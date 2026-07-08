"""Parity tests of Python AutoADAM against R's ``auto.adam()``.

R outputs are fetched live via :mod:`tests._r_bridge` (which loads the local
R source through ``devtools::load_all``), so the comparison is always against
the current checkout.

Each case checks that Python's AutoADAM selects the same:
  - Error distribution
  - ARIMA orders (AR, I, MA per lag level)
  - AICc value (within 0.05 tolerance for numerical precision)

A single Rscript invocation produces all reference outputs for all six cases
(seed-controlled data generation + ``auto.adam()`` fit). Python fits each
case lazily under a class-scoped fixture so the search runs once per case.

Skipped in CI by default (``r_parity`` marker — opt in with
``pytest -m r_parity``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smooth.adam_general.core.auto_adam import AutoADAM

from ._r_bridge import r_dict

pytestmark = pytest.mark.r_parity

# All six cases are computed in a single Rscript call: the R block sets up
# every input series, fits each ``auto.adam`` model, and returns the joint
# outputs as a nested JSON dict. This keeps the round-trip to R cheap while
# preserving full coverage.
_R_ALL_CASES = r"""
{
    library(jsonlite)
    safe_int <- function(x) if (length(x) == 0) 0L else as.integer(x)
    get_ar <- function(m) if (is.list(m$orders)) safe_int(m$orders$ar) else 0L
    get_i  <- function(m) if (is.list(m$orders)) safe_int(m$orders$i)  else 0L
    get_ma <- function(m) if (is.list(m$orders)) safe_int(m$orders$ma) else 0L
    pack <- function(m, y) list(
        model_name   = as.character(modelType(m)),
        distribution = as.character(m$distribution),
        ar_orders    = get_ar(m),
        i_orders     = get_i(m),
        ma_orders    = get_ma(m),
        aicc         = as.numeric(AICc(m)),
        series       = as.numeric(y)
    )

    # Case 1: pure ARIMA, non-seasonal
    set.seed(42); y1 <- cumsum(rnorm(100))
    m1 <- auto.adam(y1, model='NNN', lags=c(1),
                    orders=list(ar=c(2), i=c(2), ma=c(2), select=TRUE),
                    distribution=c('dnorm','dlaplace','ds'),
                    ic='AICc', silent=TRUE)

    # Case 2: pure ARIMA, seasonal
    set.seed(123)
    seasonal_comp <- rep(sin(seq(0, 2*pi, length.out=13)[-13]), 20)
    y2 <- cumsum(rnorm(240)) + seasonal_comp * 5
    m2 <- auto.adam(y2, model='NNN', lags=c(1,12),
                    orders=list(ar=c(2,2), i=c(2,1), ma=c(2,2), select=TRUE),
                    distribution=c('dnorm','dlaplace'),
                    ic='AICc', silent=TRUE)

    # Case 3: ETS only (no ARIMA), distribution selection
    set.seed(7)
    trend3 <- seq(100, 130, length.out=120)
    seasonal3 <- rep(c(1, 1.1, 1.2, 1.1, 1.0, 0.9, 0.8, 0.9, 1.0, 1.0, 1.1, 1.2), 10)
    y3 <- trend3 * seasonal3 * exp(rnorm(120, 0, 0.05))
    m3 <- auto.adam(y3, model='ZXZ', lags=c(12),
                    orders=list(ar=c(0,0), i=c(0,0), ma=c(0,0), select=FALSE),
                    distribution=c('dnorm','dlaplace','dgamma'),
                    ic='AICc', silent=TRUE)

    # Case 4: ETS + ARIMA selection, non-seasonal
    set.seed(99)
    y4 <- 100 + seq(0, 20, length.out=120) + rnorm(120, 0, 3)
    m4 <- auto.adam(y4, model='AAN', lags=c(1),
                    orders=list(ar=c(2), i=c(2), ma=c(2), select=TRUE),
                    distribution=c('dnorm','dlaplace'),
                    ic='AICc', silent=TRUE)

    # Case 5: ETSX (the xreg data is read from disk; it is a fixed fixture
    # generated once in the past — not RNG-controlled).
    xreg_data <- read.csv('python/tests/data/etsx_data.csv')
    m5 <- auto.adam(xreg_data, model='AAN',
                    orders=list(ar=c(0), i=c(0), ma=c(0), select=FALSE),
                    distribution=c('dnorm','dlaplace'),
                    regressors='use', ic='AICc', silent=TRUE)

    # Case 6: ETS + ARIMA selection, seasonal
    set.seed(55)
    trend6 <- seq(50, 100, length.out=144)
    seasonal6 <- rep(sin(seq(0, 2*pi, length.out=13)[-13]) * 10, 12)
    y6 <- trend6 + seasonal6 + rnorm(144, 0, 2)
    m6 <- auto.adam(y6, model='ZXZ', lags=c(12),
                    orders=list(ar=c(2,2), i=c(2,1), ma=c(2,2), select=TRUE),
                    distribution=c('dnorm','dlaplace'),
                    ic='AICc', silent=TRUE)

    list(
        nnn_nonseasonal     = pack(m1, y1),
        nnn_seasonal        = pack(m2, y2),
        ets_only            = pack(m3, y3),
        ets_arima           = pack(m4, y4),
        etsx                = list(  # xreg case carries X, not y
            model_name   = as.character(modelType(m5)),
            distribution = as.character(m5$distribution),
            ar_orders    = get_ar(m5),
            i_orders     = get_i(m5),
            ma_orders    = get_ma(m5),
            aicc         = as.numeric(AICc(m5))
        ),
        ets_arima_seasonal  = pack(m6, y6)
    )
}
"""


# AutoADAM kwargs for each case (mirrors the Python-side parameters in the
# original reference JSON).
PY_PARAMS = {
    "nnn_nonseasonal": dict(
        model="NNN",
        lags=[1],
        ar_order=2,
        i_order=2,
        ma_order=2,
        arima_select=True,
        distribution=["dnorm", "dlaplace", "ds"],
        ic="AICc",
    ),
    "nnn_seasonal": dict(
        model="NNN",
        lags=[1, 12],
        ar_order=[2, 2],
        i_order=[2, 1],
        ma_order=[2, 2],
        arima_select=True,
        distribution=["dnorm", "dlaplace"],
        ic="AICc",
    ),
    "ets_only": dict(
        model="ZXZ",
        lags=[12],
        ar_order=0,
        i_order=0,
        ma_order=0,
        arima_select=False,
        distribution=["dnorm", "dlaplace", "dgamma"],
        ic="AICc",
    ),
    "ets_arima": dict(
        model="AAN",
        lags=[1],
        ar_order=2,
        i_order=2,
        ma_order=2,
        arima_select=True,
        distribution=["dnorm", "dlaplace"],
        ic="AICc",
    ),
    "etsx": dict(
        model="AAN",
        ar_order=0,
        i_order=0,
        ma_order=0,
        arima_select=False,
        distribution=["dnorm", "dlaplace"],
        regressors="use",
        ic="AICc",
    ),
    "ets_arima_seasonal": dict(
        model="ZXZ",
        lags=[12],
        ar_order=[2, 2],
        i_order=[2, 1],
        ma_order=[2, 2],
        arima_select=True,
        distribution=["dnorm", "dlaplace"],
        ic="AICc",
    ),
}


@pytest.fixture(scope="module")
def r_outputs():
    return r_dict(_R_ALL_CASES)


def _normalise_orders(val):
    if isinstance(val, int):
        return [val]
    if isinstance(val, list):
        return list(val)
    return [int(val)]


def _series_for(case_name, r_outputs):
    """Return ``y`` (and ``X`` if applicable) for the given case. ``etsx``
    uses a fixed regressor fixture on disk; everything else takes ``y`` from
    R via the same seed-controlled simulation R itself used."""
    from pathlib import Path

    if case_name == "etsx":
        df = pd.read_csv(Path(__file__).parent / "data" / "etsx_data.csv")
        y = df.iloc[:, 0].to_numpy(dtype=float)
        X = df.iloc[:, 1:].to_numpy(dtype=float) if df.shape[1] > 1 else None
        return y, X
    y = np.asarray(r_outputs[case_name]["series"], dtype=float)
    return y, None


# --------------------------------------------------------------------------
# One class per case. Each class fits AutoADAM once at class-fixture level so
# the (potentially expensive) model search isn't repeated across tests.
# --------------------------------------------------------------------------


def _fit_python(case_name, r_outputs):
    y, X = _series_for(case_name, r_outputs)
    return AutoADAM(**PY_PARAMS[case_name]).fit(y, X)


class TestNNNNonseasonal:
    @pytest.fixture(scope="class")
    def m(self, r_outputs):
        return _fit_python("nnn_nonseasonal", r_outputs)

    def test_distribution(self, m, r_outputs):
        assert m._selected_distribution == r_outputs["nnn_nonseasonal"]["distribution"][0]

    def test_i_orders(self, m, r_outputs):
        assert _normalise_orders(m._selected_arima_orders["i_orders"]) == _normalise_orders(
            r_outputs["nnn_nonseasonal"]["i_orders"]
        )

    def test_ar_orders(self, m, r_outputs):
        assert _normalise_orders(
            m._selected_arima_orders["ar_orders"]
        ) == _normalise_orders(r_outputs["nnn_nonseasonal"]["ar_orders"])

    def test_ma_orders(self, m, r_outputs):
        assert _normalise_orders(
            m._selected_arima_orders["ma_orders"]
        ) == _normalise_orders(r_outputs["nnn_nonseasonal"]["ma_orders"])

    def test_aicc(self, m, r_outputs):
        assert abs(m.aicc - r_outputs["nnn_nonseasonal"]["aicc"][0]) < 0.05


class TestNNNSeasonal:
    @pytest.fixture(scope="class")
    def m(self, r_outputs):
        return _fit_python("nnn_seasonal", r_outputs)

    def test_distribution(self, m, r_outputs):
        assert m._selected_distribution == r_outputs["nnn_seasonal"]["distribution"][0]

    def test_i_orders(self, m, r_outputs):
        assert _normalise_orders(
            m._selected_arima_orders["i_orders"]
        ) == _normalise_orders(r_outputs["nnn_seasonal"]["i_orders"])

    def test_ar_orders(self, m, r_outputs):
        assert _normalise_orders(
            m._selected_arima_orders["ar_orders"]
        ) == _normalise_orders(r_outputs["nnn_seasonal"]["ar_orders"])

    def test_ma_orders(self, m, r_outputs):
        assert _normalise_orders(
            m._selected_arima_orders["ma_orders"]
        ) == _normalise_orders(r_outputs["nnn_seasonal"]["ma_orders"])

    def test_aicc(self, m, r_outputs):
        assert abs(m.aicc - r_outputs["nnn_seasonal"]["aicc"][0]) < 0.05


class TestETSOnly:
    @pytest.fixture(scope="class")
    def m(self, r_outputs):
        return _fit_python("ets_only", r_outputs)

    def test_ets_model(self, m, r_outputs):
        assert r_outputs["ets_only"]["model_name"][0] in m.model

    def test_distribution(self, m, r_outputs):
        assert m._selected_distribution == r_outputs["ets_only"]["distribution"][0]

    def test_aicc(self, m, r_outputs):
        assert abs(m.aicc - r_outputs["ets_only"]["aicc"][0]) < 0.05


class TestETSARIMA:
    @pytest.fixture(scope="class")
    def m(self, r_outputs):
        return _fit_python("ets_arima", r_outputs)

    def test_ets_model(self, m, r_outputs):
        assert r_outputs["ets_arima"]["model_name"][0] in m.model

    def test_distribution(self, m, r_outputs):
        assert m._selected_distribution == r_outputs["ets_arima"]["distribution"][0]

    def test_ar_orders(self, m, r_outputs):
        assert _normalise_orders(
            m._selected_arima_orders["ar_orders"]
        ) == _normalise_orders(r_outputs["ets_arima"]["ar_orders"])

    def test_i_orders(self, m, r_outputs):
        assert _normalise_orders(
            m._selected_arima_orders["i_orders"]
        ) == _normalise_orders(r_outputs["ets_arima"]["i_orders"])

    def test_ma_orders(self, m, r_outputs):
        assert _normalise_orders(
            m._selected_arima_orders["ma_orders"]
        ) == _normalise_orders(r_outputs["ets_arima"]["ma_orders"])

    def test_aicc(self, m, r_outputs):
        assert abs(m.aicc - r_outputs["ets_arima"]["aicc"][0]) < 0.05


class TestETSX:
    @pytest.fixture(scope="class")
    def m(self, r_outputs):
        return _fit_python("etsx", r_outputs)

    def test_ets_model(self, m, r_outputs):
        assert r_outputs["etsx"]["model_name"][0] in m.model

    def test_distribution(self, m, r_outputs):
        assert m._selected_distribution == r_outputs["etsx"]["distribution"][0]

    def test_aicc(self, m, r_outputs):
        assert abs(m.aicc - r_outputs["etsx"]["aicc"][0]) < 0.05


class TestETSARIMASeasonal:
    @pytest.fixture(scope="class")
    def m(self, r_outputs):
        return _fit_python("ets_arima_seasonal", r_outputs)

    def test_ets_model(self, m, r_outputs):
        assert r_outputs["ets_arima_seasonal"]["model_name"][0] in m.model

    def test_distribution(self, m, r_outputs):
        assert m._selected_distribution == r_outputs["ets_arima_seasonal"]["distribution"][0]

    def test_ar_orders(self, m, r_outputs):
        assert _normalise_orders(
            m._selected_arima_orders["ar_orders"]
        ) == _normalise_orders(r_outputs["ets_arima_seasonal"]["ar_orders"])

    def test_i_orders(self, m, r_outputs):
        assert _normalise_orders(
            m._selected_arima_orders["i_orders"]
        ) == _normalise_orders(r_outputs["ets_arima_seasonal"]["i_orders"])

    def test_ma_orders(self, m, r_outputs):
        assert _normalise_orders(
            m._selected_arima_orders["ma_orders"]
        ) == _normalise_orders(r_outputs["ets_arima_seasonal"]["ma_orders"])

    def test_aicc(self, m, r_outputs):
        assert abs(m.aicc - r_outputs["ets_arima_seasonal"]["aicc"][0]) < 0.05
