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


def test_a_series_of_zeros_and_ones_agrees():
    """Without an occurrence model the zeros are values, as in R: a dummy variable
    (forecast by ADAM when the future regressors are missing) is an ETS(ANN), not
    the naive of a single non-zero observation."""
    r = r_dict(
        "{ x <- rep(0, 60); x[21] <- 1; m <- suppressWarnings(adam(x, h=3));"
        " list(model=m$model, forecast=as.numeric(m$forecast)) }"
    )
    y = np.zeros(60)
    y[20] = 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ADAM().fit(y)
    assert fit.model_name == r["model"][0]
    np.testing.assert_allclose(fit.predict(h=3).mean, r["forecast"], rtol=1e-8)


@pytest.mark.parametrize("model, outliers", [("MAM", "use"), ("MAM", "select")])
def test_the_outliers_agree(model, outliers):
    """The dummies of the outliers, named and ordered as R's (with the leads and lags
    of "select"), the model refitted with them as specified, and their zeros in the
    forecasts."""
    r = r_dict(
        "{ y <- AirPassengers; y[c(30, 140)] <- y[c(30, 140)]*1.4;"
        f" m <- adam(y, '{model}', outliers='{outliers}');"
        " list(y=as.numeric(y), model=m$model, B=unname(coef(m)), names=names(coef(m)),"
        " logLik=as.numeric(logLik(m)), forecast=as.numeric(forecast(m, h=12)$mean)) }"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ADAM(model=model, lags=[12], outliers=outliers).fit(
            np.asarray(r["y"], dtype=float)
        )
    assert fit.model_name == r["model"][0]
    assert fit.coef_names == r["names"]
    np.testing.assert_allclose(fit.coef, r["B"], rtol=1e-6, atol=1e-8)
    assert fit.loglik == pytest.approx(r["logLik"][0], rel=1e-10)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        forecast = fit.predict(h=12).mean
    np.testing.assert_allclose(forecast, r["forecast"], rtol=1e-8)


def test_a_regressor_named_as_a_dummy_is_renamed():
    """A regressor named outlier1 becomes x.outlier1, with a warning, as in R."""
    r = r_dict(
        "{ y <- AirPassengers; y[c(30, 140)] <- y[c(30, 140)]*1.4; set.seed(1);"
        " d <- data.frame(y=as.numeric(y), outlier1=rnorm(144));"
        " m <- suppressWarnings(adam(d, 'MAM', lags=12, outliers='use'));"
        " list(y=d$y, x=d$outlier1, names=names(coef(m)), B=unname(coef(m))) }"
    )
    X = pd.DataFrame({"outlier1": r["x"]})
    with pytest.warns(UserWarning, match="Renaming them to x.outlier1"):
        fit = ADAM(model="MAM", lags=[12], outliers="use").fit(
            np.asarray(r["y"], dtype=float), X
        )
    assert fit.coef_names == r["names"]
    np.testing.assert_allclose(fit.coef, r["B"], rtol=1e-6, atol=1e-8)
