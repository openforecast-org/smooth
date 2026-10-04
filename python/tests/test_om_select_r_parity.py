"""OM / OMG ETS model selection against R's ``om()`` / ``omg()``.

The same pool, branch-and-bound and estimator run on both sides, so the
selected model, the information criteria, the logLik and the forecasts match
to the last bits. Skipped by default (``r_parity`` marker).
"""

from __future__ import annotations

import numpy as np
import pytest

from smooth import OM, OMG

from ._r_bridge import r_dict

pytestmark = pytest.mark.r_parity

DATA = (
    "set.seed(41); "
    "y <- rbinom(364, 1, plogis(-0.3 + 1.2*sin(2*pi*(1:364)/7)))*rpois(364, 3);"
)

OM_CASES = [
    ("ZZN", [1], "odds-ratio"),
    ("ZXZ", [1, 7], "odds-ratio"),
    ("ZZZ", [1, 7], "inverse-odds-ratio"),
    ("ZXN", [1], "direct"),
    ("ZXN", [1], "auto"),
    ("ZXN", [1], "general"),
]


def _r_lags(lags):
    return "c(" + ",".join(map(str, lags)) + ")"


def _r_fit(call):
    return r_dict(
        f"{{ {DATA} m <- {call}; "
        "list(y=y, model=m$model, logLik=as.numeric(logLik(m)), "
        "ICs=if(is.null(m$ICs)) NULL else unname(m$ICs), "
        "ICnames=if(is.null(m$ICs)) NULL else names(m$ICs), "
        "f=as.numeric(forecast(m, h=7)$mean)) }"
    )


def _compare(py, r):
    assert py.model == r["model"][0]
    assert py.loglik == pytest.approx(r["logLik"][0], rel=1e-12)
    np.testing.assert_allclose(
        np.asarray(py.predict(h=7).mean), r["f"], rtol=1e-10, atol=1e-12
    )


@pytest.mark.parametrize(
    "model,lags,occurrence", OM_CASES, ids=[f"{c[0]}-{c[2]}" for c in OM_CASES]
)
def test_om_select_matches_r(model, lags, occurrence):
    r = _r_fit(
        f"om(y, model='{model}', lags={_r_lags(lags)}, occurrence='{occurrence}')"
    )
    # fit() returns the selected object for occurrence="auto", as auto.om() does
    py = OM(model=model, lags=lags, occurrence=occurrence).fit(
        np.asarray(r["y"], dtype=float)
    )
    _compare(py, r)
    if r.get("ICnames") and getattr(py, "ics", None) is not None:
        assert list(py.ics) == r["ICnames"]
        np.testing.assert_allclose(list(py.ics.values()), r["ICs"], rtol=1e-12)


@pytest.mark.parametrize(
    "model_a,model_b", [("ZXZ", "ANN"), ("ANA", "ANN"), ("ANN", "ANA")]
)
def test_omg_mixed_sides_match_r(model_a, model_b):
    r = _r_fit(f"omg(y, modelA='{model_a}', modelB='{model_b}', lags=c(1,7))")
    py = OMG(model_a=model_a, model_b=model_b, lags=[1, 7])
    py.fit(np.asarray(r["y"], dtype=float))
    _compare(py, r)
