"""R-Python parity of ADAM with regressors: the smoothing parameters estimated with a
provided alpha (those of "adapt" and the trend), and a constant with a multiplicative
error."""

import warnings

import numpy as np
import pandas as pd
import pytest

from smooth import ADAM
from tests._r_bridge import r_dict

pytestmark = pytest.mark.r_parity

DATA = (
    "set.seed(41); x <- rnorm(120, 0, 1);"
    " d <- data.frame(y=100*exp(0.002*(1:120) + 0.1*x + rnorm(120, 0, 0.02)), x=x);"
)
CASES = {
    "adapt with alpha": (
        "model='ANN', persistence=list(alpha=0.3), regressors='adapt'",
        dict(model="ANN", persistence={"alpha": 0.3}, regressors="adapt"),
    ),
    "trend with alpha": (
        "model='AAN', persistence=list(alpha=0.3)",
        dict(model="AAN", persistence={"alpha": 0.3}),
    ),
    "multiplicative constant": (
        "model='MNN', constant=TRUE, initial='optimal'",
        dict(model="MNN", constant=True, initial="optimal"),
    ),
}


@pytest.mark.parametrize("case", list(CASES))
def test_the_fits_agree(case):
    r_arguments, arguments = CASES[case]
    r = r_dict(
        f"{{ {DATA} m <- adam(d, {r_arguments});"
        " list(y=d$y, x=d$x, B=unname(m$B), names=names(m$B),"
        " logLik=as.numeric(logLik(m)),"
        " persistence=unname(m$persistence), fitted=as.numeric(fitted(m)),"
        " forecast=as.numeric(forecast(m, h=5, newdata=data.frame(x=rep(0.5, 5)))$mean)) }"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ADAM(**arguments).fit(
            np.asarray(r["y"], dtype=float), pd.DataFrame({"x": r["x"]})
        )
    # The coefficients of the regressors are named after them, as in R
    assert list(fit.coef_names) == list(r["names"])
    np.testing.assert_allclose(fit.coef, r["B"], rtol=1e-6, atol=1e-8)
    assert fit.loglik == pytest.approx(r["logLik"][0], rel=1e-8)
    np.testing.assert_allclose(fit.fitted, r["fitted"], rtol=1e-8)
    forecast = fit.predict(h=5, X=np.full((5, 1), 0.5)).mean
    np.testing.assert_allclose(forecast, r["forecast"], rtol=1e-8)
