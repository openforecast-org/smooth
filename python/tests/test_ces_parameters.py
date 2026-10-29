"""Tests of :class:`smooth.CES` and :class:`smooth.AutoCES` that need no R.

Kept out of ``test_ces.py`` because that module is marked ``r_parity`` as a
whole; these check the Python API's own guards, and its results against values
taken from R (the R call is in each test).
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smooth import CES, AutoCES

_DATA_DIR = Path(__file__).parent / "data"

# The values from R are those of the reference BLAS. The C++ core takes its matrix
# products from the BLAS of the build, and the CES fits move with their last bits (R
# moves the same way): the criteria by up to 4e-5 with the OpenBLAS of CI on Windows,
# the bounds of reapply() by up to 3e-9
_FIT_TOLERANCE = 1e-4
_BOUND_TOLERANCE = 1e-6


class TestCESProvidedB:
    """The provided ``b`` must match the seasonality it belongs to."""

    y = pd.read_csv(_DATA_DIR / "ces_airpassengers.csv")["y"].values

    def test_full_rejects_a_real_b(self):
        with pytest.raises(ValueError, match="complex second smoothing parameter"):
            CES(seasonality="full", lags=[12], b=0.3).fit(self.y)

    def test_partial_rejects_a_complex_b(self):
        with pytest.raises(ValueError, match="real second smoothing parameter"):
            CES(seasonality="partial", lags=[12], b=complex(0.3, 0.1)).fit(self.y)

    @pytest.mark.parametrize("seasonality", ["none", "simple"])
    def test_b_is_dropped_where_there_is_none(self, seasonality):
        with pytest.warns(UserWarning, match="no second smoothing parameter"):
            m = CES(seasonality=seasonality, lags=[12], b=0.3)
        m.fit(self.y)
        assert m.b_ is None
        free = CES(seasonality=seasonality, lags=[12])
        free.fit(self.y)
        assert m.loglik == free.loglik

    def test_valid_b_is_honoured(self):
        free = CES(seasonality="full", lags=[12])
        free.fit(self.y)
        held = CES(seasonality="full", lags=[12], b=free.b_)
        held.fit(self.y)
        assert held.b_ == free.b_

        partial = CES(seasonality="partial", lags=[12], b=0.3)
        partial.fit(self.y)
        assert partial.b_ == 0.3


def test_ces_predict_keeps_the_seasonal_phase_past_one_cycle():
    """predict() beyond one cycle reads the seasonal cells in phase, as the
    forecast made at fit time does, when the sample ends mid-cycle."""
    t = np.arange(138)
    y = (
        100
        + t
        + 10 * np.sin(2 * np.pi * t / 12)
        + np.random.default_rng(3).normal(0, 2, t.size)
    )
    fitted_with_h = CES(seasonality="full", lags=[12], h=30).fit(y)
    fitted_alone = CES(seasonality="full", lags=[12]).fit(y)
    # The forecast of the C++ core from the end of the sample, with the lookup
    # table of the fit, which spans the horizon
    m = fitted_with_h
    start = m._head_geometry + m._obs_in_sample
    forecast = m._adam_cpp.forecast(
        matrixWt=np.asfortranarray(np.tile(m._mat_wt[-1:], (30, 1))),
        matrixF=np.asfortranarray(m._mat_f),
        indexLookupTable=np.asfortranarray(m._index_lookup_table[:, start : start + 30]),
        profilesRecent=np.asfortranarray(m._profiles_recent_table.copy()),
        horizon=30,
    ).forecast
    np.testing.assert_allclose(
        np.asarray(fitted_alone.predict(h=30).mean), np.ravel(forecast), rtol=1e-12
    )


def test_sum_r_adds_in_order_as_r():
    """R's sum() accumulates sequentially in long double: each 1 is lost
    against 2^64, so the total is 0 (R: sum(c(2^64, rep(1,7), -2^64, rep(0,7))))
    where NumPy's pairwise long double sum gives 7."""
    from smooth.adam_general.core.utils.utils import _sum_r

    values = np.array([2.0**64] + [1.0] * 7 + [-(2.0**64)] + [0.0] * 7)
    assert _sum_r(values) == 0.0
    assert np.array_equal(_sum_r(np.vstack([values, values]), axis=1), [0.0, 0.0])


def test_ces_takes_the_missing_values_for_gaps():
    """The missing values are skipped by the fit and are not in the loss: the
    likelihood is over the observed values, as its per-observation terms."""
    y = pd.read_csv(_DATA_DIR / "ces_airpassengers.csv")["y"].values.astype(float)
    y[[9, 49, 50, 89]] = np.nan
    with pytest.warns(UserWarning, match="NAs"):
        model = CES(seasonality="full", lags=[12]).fit(y)
    assert np.isfinite(model.loglik)
    assert np.sum(model.point_lik()) == pytest.approx(model.loglik, rel=1e-12)
    np.testing.assert_array_equal(np.isnan(model.residuals), np.isnan(y))
    assert np.all(np.isfinite(model.predict(h=12, interval="prediction").upper))


_AIR = pd.read_csv(_DATA_DIR / "ces_airpassengers.csv")["y"].to_numpy(dtype=float)


def test_auto_ces_selects_by_the_information_criterion_as_r():
    """R: auto.ces(ts(y, frequency=12)) picks CES(partial) by AICc."""
    m = AutoCES(lags=[12]).fit(_AIR)
    assert m.model_name == "CES(partial)"
    assert m.loglik == pytest.approx(-573.675717908, rel=_FIT_TOLERANCE)
    assert m.ICs == pytest.approx(
        {
            "none": 1430.97768695,
            "simple": 1333.66542447,
            "partial": 1188.82343582,
            "full": 1252.20790401,
        },
        rel=_FIT_TOLERANCE,
    )


def test_auto_ces_takes_the_abbreviated_pool():
    """R: auto.ces(ts(y, frequency=12), seasonality=c("n", "s"))."""
    m = AutoCES(lags=[12], seasonality=["n", "s"]).fit(_AIR)
    assert m.model_name == "CES(simple)"
    assert m.ICs == pytest.approx(
        {"none": 1430.97768695, "simple": 1333.66542447}, rel=_FIT_TOLERANCE
    )


def test_auto_ces_reverts_a_strange_pool_to_the_default():
    with pytest.warns(UserWarning, match="strange type of model"):
        m = AutoCES(lags=[12], seasonality=["n", "x"]).fit(_AIR)
    assert sorted(m.ICs) == ["full", "none", "partial", "simple"]
    assert m.model_name == "CES(partial)"


def test_auto_ces_fits_the_non_seasonal_model_on_non_seasonal_data():
    """R: auto.ces(y) without a frequency fits CES(none) only."""
    m = AutoCES().fit(_AIR)
    assert m.model_name == "CES(none)"
    assert list(m.ICs) == ["none"]
    assert m.loglik == pytest.approx(-710.258388775, rel=_FIT_TOLERANCE)


def test_auto_ces_drops_the_models_a_short_sample_cannot_take():
    """R: auto.ces(ts(y[1:26], frequency=12)) compares CES(none) and CES(partial)
    only, warning about the others."""
    with pytest.warns(UserWarning, match="Not enough observations for CES"):
        m = AutoCES(lags=[12]).fit(_AIR[:26])
    assert m.model_name == "CES(none)"
    assert set(m.ICs) == {"none", "partial"}
    assert m.ICs["none"] == pytest.approx(222.150196905, rel=_FIT_TOLERANCE)
    # The fit of the partial CES on 26 observations is on a flat surface, where the
    # last bits of the matrix products of the BLAS move it: 292.281 with the reference
    # BLAS (R's here), 291.650 with OpenBLAS (CI's), in R as in Python
    assert m.ICs["partial"] > m.ICs["none"]


@pytest.mark.parametrize(
    "seasonality, a, name, side, bound, tolerance",
    [
        ("none", None, "alpha_0", "max", 2.00026793268, _BOUND_TOLERANCE),
        ("none", None, "alpha_1", "min", 1.00013497312, _BOUND_TOLERANCE),
        ("partial", None, "alpha_1", "max", 1.0, _BOUND_TOLERANCE),
        # The bound of the full CES follows its estimates, which the last bits of the
        # matrix products of the BLAS move (1.837771 with the reference BLAS, 1.836848
        # with OpenBLAS)
        ("full", 1.35 + 1.01j, "beta_0", "max", 1.83777093985, 1e-3),
        ("full", 1.35 + 1.01j, "beta_1", "min", 0.94, _BOUND_TOLERANCE),
    ],
)
def test_reapply_keeps_the_draws_of_ces_stable(
    seasonality, a, name, side, bound, tolerance
):
    """The draws of the smoothing parameters are cut at the range where the model
    stays stable, with the others at their estimates, as R's cesBounds(): they
    pile up at R's bound. R: set.seed(1); m <- ces(ts(y, frequency=12),
    seasonality=..., initial="optimal", a=...); range(reapply(m,
    nsim=2000)$randomParameters[, name])."""
    m = CES(seasonality=seasonality, lags=[12], initial="optimal", a=a).fit(_AIR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        draws = np.asarray(m.reapply(nsim=2000, seed=1).random_parameters)
    draws = draws[:, m.coef_names.index(name)]
    edge = draws.max() if side == "max" else draws.min()
    assert edge == pytest.approx(bound, rel=tolerance)
    assert np.sum(draws == edge) > 100
