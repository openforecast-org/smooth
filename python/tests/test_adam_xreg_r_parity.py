"""R-Python parity of ADAM with regressors: the smoothing parameters estimated with a
provided alpha (those of "adapt" and the trend), a constant with a multiplicative
error, the missing values of the regressors, the initials provided for some of them,
and a regression with an occurrence model."""

import warnings

import numpy as np
import pandas as pd
import pytest

from smooth import ADAM, OM
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


# Two regressors, the second with a missing value, and an intermittent version
XREG = (
    "set.seed(41); x1 <- rnorm(60, 10, 2); x2 <- rnorm(60, 5, 1);"
    " d <- data.frame(y=50 + 2*x1 - 3*x2 + cumsum(rnorm(60)), x1=x1, x2=x2);"
    " set.seed(2); di <- transform(d, y=y*rbinom(60, 1, 0.7)); dn <- d; dn$x2[5] <- NA;"
)


def _xreg_data(r, key="y"):
    X = pd.DataFrame(
        {
            "x1": np.asarray(r["x1"], dtype=float),
            "x2": np.asarray(
                [np.nan if v in ("NA", None) else v for v in r["x2"]], dtype=float
            ),
        }
    )
    return np.asarray(r[key], dtype=float), X


@pytest.mark.parametrize(
    "r_arguments, arguments",
    [
        ("'ANN'", dict(model="ANN")),
        ("'ANN', regressors='select'", dict(model="ANN", regressors="select")),
        ("'ANN', regressors='adapt'", dict(model="ANN", regressors="adapt")),
        ("'NNN'", dict(model="NNN")),
    ],
)
def test_a_missing_regressor_is_its_first_value_as_r(r_arguments, arguments):
    r = r_dict(
        f"{{ {XREG} m <- suppressWarnings(adam(dn, {r_arguments}));"
        " list(y=dn$y, x1=dn$x1, x2=dn$x2, logLik=as.numeric(logLik(m))) }"
    )
    y, X = _xreg_data(r)
    with pytest.warns(UserWarning, match="Some variables contained NAs"):
        model = ADAM(**arguments).fit(y, X=X)
    np.testing.assert_allclose(model.loglik, r["logLik"][0], rtol=1e-8)


@pytest.mark.parametrize(
    "r_initial, initial",
    [("c(x1=2)", {"x1": 2.0}), ("c(x2=-3)", {"x2": -3.0}), ("2", [2.0])],
)
def test_the_initials_of_some_regressors_are_those_of_r(r_initial, initial):
    r = r_dict(
        f"{{ {XREG} m <- adam(d, 'ANN', initial=list(xreg={r_initial}));"
        " list(y=d$y, x1=d$x1, x2=d$x2, logLik=as.numeric(logLik(m)),"
        " xreg=unname(m$initial$xreg), nparam=nparam(m), provided=m$nParam[2,2],"
        " initialType=m$initialType) }"
    )
    y, X = _xreg_data(r)
    model = ADAM(model="ANN", initial={"xreg": initial}).fit(y, X=X)
    np.testing.assert_allclose(model.loglik, r["logLik"][0], rtol=1e-8)
    np.testing.assert_allclose(
        list(model.initial_value["xreg"].values()), r["xreg"], rtol=1e-6
    )
    assert model.nparam == r["nparam"][0]
    assert model.n_param.provided["xreg"] == r["provided"][0]
    assert model.initial_type == r["initialType"][0]


@pytest.mark.parametrize("occurrence", ["fixed", "odds-ratio"])
def test_a_regression_with_an_occurrence_is_that_of_r(occurrence):
    r = r_dict(
        f"{{ {XREG} m <- adam(di, 'NNN', occurrence='{occurrence}');"
        " f <- forecast(m, h=3, newdata=di[1:3,], interval='prediction',"
        " level=c(0.8, 0.95));"
        " list(y=di$y, x1=di$x1, x2=di$x2, coef=unname(coef(m)),"
        " logLik=as.numeric(logLik(m)), nparam=nparam(m),"
        " fitted=as.numeric(fitted(m)), residuals=as.numeric(residuals(m)),"
        " mean=as.numeric(f$mean), lower=as.numeric(f$lower),"
        " upper=as.numeric(f$upper)) }"
    )
    y, X = _xreg_data(r)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ADAM(model="NNN", occurrence=occurrence).fit(y, X=X)
    np.testing.assert_allclose(model.coef, r["coef"], rtol=1e-6)
    np.testing.assert_allclose(model.loglik, r["logLik"][0], rtol=1e-6)
    assert model.nparam == r["nparam"][0]
    np.testing.assert_allclose(np.asarray(model.fitted), r["fitted"], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(model.residuals), r["residuals"], atol=1e-8)
    forecast = model.predict(
        h=3, X=X.iloc[:3], interval="prediction", level=[0.8, 0.95]
    )
    np.testing.assert_allclose(forecast.mean.values, r["mean"], rtol=1e-6)
    np.testing.assert_allclose(
        forecast.lower.values.ravel(order="F"), r["lower"], rtol=1e-6
    )
    np.testing.assert_allclose(
        forecast.upper.values.ravel(order="F"), r["upper"], rtol=1e-6
    )


def test_the_forecast_of_a_regression_of_the_occurrence_is_that_of_r():
    r = r_dict(
        f"{{ {XREG} m <- om(di, model='NNN', occurrence='odds-ratio');"
        " list(y=di$y, x1=di$x1, x2=di$x2,"
        " mean=as.numeric(forecast(m, h=3, newdata=di[1:3,])$mean)) }"
    )
    y, X = _xreg_data(r)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = OM(model="NNN", occurrence="odds-ratio").fit(y, X=X)
    forecast = model.predict(h=3, X=X.iloc[:3])
    np.testing.assert_allclose(forecast.mean.values, r["mean"], rtol=1e-6)
