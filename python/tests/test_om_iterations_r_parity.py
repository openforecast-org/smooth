"""R-Python parity of n_iterations and smoother in OMG and AutoOM: they reach
every model fitted, as the ellipsis of R's omg() and auto.om()."""

import warnings

import numpy as np
import pytest

from smooth import OM, OMG, AutoOM
from tests._r_bridge import r_dict

pytestmark = pytest.mark.r_parity

DATA = (
    "set.seed(5); z <- as.numeric(arima.sim(list(ar=0.8), 200));"
    " y <- rbinom(200, 1, plogis(z))*rpois(200, 3);"
)

# (R call, Python model)
CASES = {
    "omg": (
        "omg(y, 'MNN', 'MNN', nIterations=3)",
        lambda: OMG(model_a="MNN", n_iterations=3),
    ),
    "om-general": (
        "om(y, 'MNN', occurrence='general', nIterations=3, smoother='lowess')",
        lambda: OM(model="MNN", occurrence="general", n_iterations=3, smoother="lowess"),
    ),
    "auto.om": (
        "auto.om(y, 'MNN', orders=list(ar=0, i=0, ma=0), nIterations=3,"
        " smoother='lowess')",
        lambda: AutoOM(
            model="MNN",
            orders={"ar": 0, "i": 0, "ma": 0},
            n_iterations=3,
            smoother="lowess",
        ),
    ),
    "om-auto": (
        "om(y, 'MNN', nIterations=3, smoother='lowess')",
        lambda: OM(model="MNN", n_iterations=3, smoother="lowess"),
    ),
}


@pytest.mark.parametrize("case", list(CASES))
def test_the_iterations_agree(case):
    r_call, python_model = CASES[case]
    r = r_dict(
        f"{{ {DATA} m <- {r_call};"
        " list(y=y, model=m$model, logLik=as.numeric(logLik(m))) }"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = python_model().fit(np.asarray(r["y"], dtype=float))
    assert model.model_name == np.atleast_1d(r["model"])[0]
    assert model.loglik == pytest.approx(np.atleast_1d(r["logLik"])[0], rel=1e-6)


def test_omg_passes_n_iterations_to_both_sides():
    y = np.tile([0, 1, 0, 0, 2, 0, 1, 1, 0, 3], 10).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = OMG(model_a="MNN", n_iterations=3).fit(y)
    assert fit.model_a._initials["n_iterations"] == 3
    assert fit.model_b._initials["n_iterations"] == 3
