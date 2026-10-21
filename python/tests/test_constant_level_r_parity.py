"""R-Python parity of the model with no ETS, no ARIMA and no regressors: R's
parametersChecker() makes it a constant level, in adam(), om() and omg()."""

import warnings

import numpy as np
import pytest

from smooth import ADAM, OM, OMG
from tests._r_bridge import r_dict

pytestmark = pytest.mark.r_parity

DATA = "set.seed(41); y <- (runif(100) < 0.4) * runif(100, 1, 5);"

# (R call, Python model)
CASES = {
    "adam": ("adam(y + 5, 'NNN')", lambda: ADAM(model="NNN")),
    "odds-ratio": (
        "om(y, 'NNN', occurrence='odds-ratio')",
        lambda: OM(model="NNN", occurrence="odds-ratio"),
    ),
    "inverse-odds-ratio": (
        "om(y, 'NNN', occurrence='inverse-odds-ratio')",
        lambda: OM(model="NNN", occurrence="inverse-odds-ratio"),
    ),
    "direct": (
        "om(y, 'NNN', occurrence='direct')",
        lambda: OM(model="NNN", occurrence="direct"),
    ),
    "omg": ("omg(y, 'NNN', 'NNN')", lambda: OMG(model_a="NNN", model_b="NNN")),
}


@pytest.mark.parametrize("case", list(CASES))
def test_the_constant_level_agrees(case):
    r_call, python_model = CASES[case]
    r = r_dict(
        f"{{ {DATA} m <- {r_call};"
        " B <- if(is.null(m$B)) c(m$modelA$B, m$modelB$B) else m$B;"
        " list(y=y, B=unname(B), logLik=as.numeric(logLik(m)),"
        " mean=as.numeric(forecast(m, h=3)$mean)) }"
    )
    y = np.asarray(r["y"], dtype=float) + (5 if case == "adam" else 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = python_model().fit(y)
        mean = np.asarray(model.predict(h=3).mean, dtype=float)
    assert np.allclose(model.b_value, r["B"], rtol=1e-6)
    assert model.loglik == pytest.approx(np.atleast_1d(r["logLik"])[0], rel=1e-8)
    assert np.allclose(mean, r["mean"], rtol=1e-6)
