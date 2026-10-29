"""R-Python parity of proxy_occurrence (R's proxyOccurrence) in OM and OMG: the same
estimates, likelihood and probabilities with the score proxy of the occurrence error."""

import warnings

import numpy as np
import pytest

from smooth import OM, OMG
from tests._r_bridge import r_dict

pytestmark = pytest.mark.r_parity

DATA = (
    "set.seed(5); z <- as.numeric(arima.sim(list(ar=0.8), 200));"
    " y <- rbinom(200, 1, plogis(z))*rpois(200, 3);"
)

CASES = [
    (occurrence, model, ets)
    for occurrence in ("odds-ratio", "inverse-odds-ratio", "general")
    for model in ("ANN", "MNN")
    for ets in ("conventional", "adam")
]


@pytest.mark.parametrize("occurrence,model,ets", CASES)
def test_the_score_proxy_agrees(occurrence, model, ets):
    r = r_dict(
        f"{{ {DATA} m <- om(y, '{model}', occurrence='{occurrence}', ets='{ets}',"
        " proxyOccurrence='score');"
        " list(y=y, logLik=as.numeric(logLik(m)), fitted=as.numeric(fitted(m))) }"
    )
    y = np.asarray(r["y"], dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if occurrence == "general":
            python_model = OMG(model_a=model, ets=ets, proxy_occurrence="score")
        else:
            python_model = OM(
                model=model, occurrence=occurrence, ets=ets, proxy_occurrence="score"
            )
        m = python_model.fit(y)
    assert m.loglik == pytest.approx(np.atleast_1d(r["logLik"])[0], rel=1e-6)
    np.testing.assert_allclose(
        np.asarray(m.fitted, dtype=float), np.asarray(r["fitted"]), atol=1e-6
    )
