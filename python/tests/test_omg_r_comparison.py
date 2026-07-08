"""Exact-match tests of the Python OMG class against R's ``smooth::omg()``.

R outputs are fetched live via :mod:`tests._r_bridge` (which loads the local
R source through ``devtools::load_all``), so the comparison is always against
the current checkout.

Tolerances:
* Scalars (loss, loglik): rtol=1e-3, atol=1e-4 — tight because the optimised
  cost should agree closely.
* Fitted / forecast: rtol=3e-3, atol=1e-3 — slightly looser to absorb small
  floating-point drift between the two C++ propagations.

Skipped in CI by default (``r_parity`` marker — opt in with
``pytest -m r_parity``).
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pytest

from smooth import OMG

from ._r_bridge import r_dict

pytestmark = pytest.mark.r_parity

RTOL = 1e-3
ATOL = 1e-4
FITTED_RTOL = 3e-3
FITTED_ATOL = 1e-3


def _scenario(
    name: str,
    *,
    py_kw: dict,
    r_fit: str,
    data_key: str = "main",
) -> dict:
    return dict(name=name, py_kw=py_kw, r_fit=r_fit, data_key=data_key)


SCENARIOS: List[Dict[str, Any]] = [
    # Group H1 — basic ETS shapes
    _scenario(
        "h1_mnn_mnn",
        py_kw=dict(model_a="MNN", model_b="MNN", lags=[1]),
        r_fit="omg(y, modelA='MNN', modelB='MNN', lags=c(1))",
    ),
    _scenario(
        "h1_ann_mnn",
        py_kw=dict(model_a="ANN", model_b="MNN", lags=[1]),
        r_fit="omg(y, modelA='ANN', modelB='MNN', lags=c(1))",
    ),
    _scenario(
        "h1_ann_ann",
        py_kw=dict(model_a="ANN", model_b="ANN", lags=[1]),
        r_fit="omg(y, modelA='ANN', modelB='ANN', lags=c(1))",
    ),
    _scenario(
        "h1_aan_mnn",
        py_kw=dict(model_a="AAN", model_b="MNN", lags=[1]),
        r_fit="omg(y, modelA='AAN', modelB='MNN', lags=c(1))",
    ),
    # Group H2 — seasonal
    _scenario(
        "h2_seasonal_ana_mnm",
        py_kw=dict(model_a="ANA", model_b="MNM", lags=[1, 12]),
        r_fit="omg(ts(ys, frequency=12), modelA='ANA', modelB='MNM', lags=c(1,12))",
        data_key="seasonal",
    ),
    _scenario(
        "h2_seasonal_mnn_mnn",
        py_kw=dict(model_a="MNN", model_b="MNN", lags=[1, 12]),
        r_fit="omg(ts(ys, frequency=12), modelA='MNN', modelB='MNN', lags=c(1,12))",
        data_key="seasonal",
    ),
    # Group H3 — holdout
    _scenario(
        "h3_holdout_mnn",
        py_kw=dict(model_a="MNN", model_b="MNN", lags=[1], h=10, holdout=True),
        r_fit="omg(y, modelA='MNN', modelB='MNN', lags=c(1), h=10, holdout=TRUE)",
    ),
    _scenario(
        "h3_holdout_ann",
        py_kw=dict(model_a="ANN", model_b="ANN", lags=[1], h=10, holdout=True),
        r_fit="omg(y, modelA='ANN', modelB='ANN', lags=c(1), h=10, holdout=TRUE)",
    ),
]


_DATA_SETUP = (
    "set.seed(41); y <- rpois(200, 0.3);"
    "set.seed(13); ys <- rpois(200, 0.4);"
)


def _r_outputs_for(scenario: dict) -> dict:
    expr = (
        "{"
        + _DATA_SETUP
        + f"m <- tryCatch({scenario['r_fit']}, error=function(e) NULL);"
        " if (is.null(m) || !is.finite(m$lossValue) || m$lossValue >= 1e10) {"
        "   list(skip=TRUE)"
        " } else {"
        "   B_A <- if (length(m$modelA$B) > 0) as.numeric(m$modelA$B) else numeric(0);"
        "   B_B <- if (length(m$modelB$B) > 0) as.numeric(m$modelB$B) else numeric(0);"
        "   ot <- as.numeric(m$modelA$data != 0);"
        "   fc <- if (!is.null(m$forecast) && length(as.numeric(m$forecast)) > 0"
        "             && !all(is.na(as.numeric(m$forecast)))) {"
        "             as.numeric(m$forecast)"
        "         } else {"
        "             tryCatch(as.numeric(forecast(m, h=10)$mean),"
        "                      error=function(e) NULL)"
        "         };"
        "   list(skip=FALSE,"
        "        fitted=as.numeric(m$fitted),"
        "        residuals=ot - as.numeric(m$fitted),"
        "        coef=c(B_A, B_B),"
        "        forecast=if (!is.null(fc) && all(is.finite(fc))) fc else NULL,"
        "        loss_value=as.numeric(m$lossValue),"
        "        loglik=as.numeric(m$logLik))"
        " }"
        "}"
    )
    return r_dict(expr)


def _input_data() -> dict:
    expr = (
        "{"
        + _DATA_SETUP
        + "list(y=as.numeric(y), ys=as.numeric(ys))"
        "}"
    )
    return r_dict(expr)


@pytest.fixture(scope="module")
def inputs():
    return _input_data()


@pytest.fixture(scope="module")
def r_outputs():
    return {s["name"]: _r_outputs_for(s) for s in SCENARIOS}


def _python_fit(scenario, inputs):
    y = np.asarray(
        inputs["ys"] if scenario["data_key"] == "seasonal" else inputs["y"],
        dtype=float,
    )
    return OMG(**scenario["py_kw"]).fit(y)


def _skip_if_degenerate(scenario, r_outputs):
    skip = r_outputs[scenario["name"]].get("skip")
    if skip is not None and bool(skip[0] if isinstance(skip, list) else skip):
        pytest.skip(f"{scenario['name']}: R returned a degenerate / penalty fit")


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["name"] for s in SCENARIOS])
class TestOMGRComparison:
    def test_loss_value(self, scenario, inputs, r_outputs):
        _skip_if_degenerate(scenario, r_outputs)
        m = _python_fit(scenario, inputs)
        np.testing.assert_allclose(
            m.loss_value,
            r_outputs[scenario["name"]]["loss_value"][0],
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"loss_value mismatch for {scenario['name']}",
        )

    def test_loglik(self, scenario, inputs, r_outputs):
        _skip_if_degenerate(scenario, r_outputs)
        m = _python_fit(scenario, inputs)
        np.testing.assert_allclose(
            m.loglik,
            r_outputs[scenario["name"]]["loglik"][0],
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"loglik mismatch for {scenario['name']}",
        )

    def test_fitted(self, scenario, inputs, r_outputs):
        _skip_if_degenerate(scenario, r_outputs)
        m = _python_fit(scenario, inputs)
        ref = np.asarray(r_outputs[scenario["name"]]["fitted"], dtype=float)
        np.testing.assert_allclose(
            m.fitted,
            ref,
            rtol=FITTED_RTOL,
            atol=FITTED_ATOL,
            err_msg=f"fitted mismatch for {scenario['name']}",
        )

    def test_forecast(self, scenario, inputs, r_outputs):
        _skip_if_degenerate(scenario, r_outputs)
        ref_raw = r_outputs[scenario["name"]].get("forecast")
        if ref_raw is None:
            pytest.skip(f"{scenario['name']}: no forecast reference")
        ref = np.asarray(ref_raw, dtype=float)
        m = _python_fit(scenario, inputs)
        fc_h = scenario["py_kw"].get("h") or 10
        fc = (
            np.asarray(m._auto_forecast.mean.values, dtype=float)
            if m._auto_forecast is not None
            else np.asarray(m.predict(h=fc_h).mean.values, dtype=float)
        )
        np.testing.assert_allclose(
            fc,
            ref,
            rtol=FITTED_RTOL,
            atol=FITTED_ATOL,
            err_msg=f"forecast mismatch for {scenario['name']}",
        )
