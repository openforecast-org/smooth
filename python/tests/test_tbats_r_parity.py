"""R-Python parity of TBATS: the same structure, parameters, likelihood, forecasts
and covariance as R's tbats() on the same data."""

import numpy as np
import pytest

from smooth import TBATS
from tests._r_bridge import r_dict

pytestmark = pytest.mark.r_parity

ORDERS0 = {"ar": 0, "ma": 0, "select": False}
ARMA11 = {"ar": 1, "ma": 1, "select": False}

# (R arguments, Python arguments) of the fits compared
CASES = {
    "additive": (
        "harmonics=5, trend='additive', orders=list(ar=0, ma=0, select=FALSE)",
        dict(harmonics=[5], trend="additive", orders=ORDERS0),
    ),
    "damped-arma": (
        "harmonics=5, trend='damped', orders=list(ar=1, ma=1, select=FALSE)",
        dict(harmonics=[5], trend="damped", orders=ARMA11),
    ),
    "selection": ("", dict()),
    "optimal": (
        "harmonics=3, trend='additive', orders=list(ar=0, ma=0, select=FALSE),"
        " initial='optimal'",
        dict(harmonics=[3], trend="additive", orders=ORDERS0, initial="optimal"),
    ),
    "two-stage": (
        "harmonics=3, trend='additive', orders=list(ar=1, ma=0, select=FALSE),"
        " initial='two-stage'",
        dict(
            harmonics=[3],
            trend="additive",
            orders={"ar": 1, "ma": 0, "select": False},
            initial="two-stage",
        ),
    ),
    "dgnorm": (
        "harmonics=4, trend='none', orders=list(ar=0, ma=0, select=FALSE),"
        " distribution='dgnorm'",
        dict(harmonics=[4], trend="none", orders=ORDERS0, distribution="dgnorm"),
    ),
    "usual": (
        "harmonics=4, trend='additive', orders=list(ar=0, ma=0, select=FALSE),"
        " bounds='usual'",
        dict(harmonics=[4], trend="additive", orders=ORDERS0, bounds="usual"),
    ),
}


def _r_fit(series, arguments, extra=""):
    comma = ", " if arguments else ""
    return r_dict(
        f"{{ m <- tbats({series}{comma}{arguments}); list(y=as.numeric(actuals(m)),"
        " model=m$model, B=unname(m$B), names=names(m$B),"
        f" logLik=as.numeric(logLik(m)){extra}) }}"
    )


@pytest.mark.parametrize("case", list(CASES))
def test_the_fits_agree_on_air_passengers(case):
    r_arguments, python_arguments = CASES[case]
    r = _r_fit("AirPassengers", r_arguments)
    fit = TBATS(lags=[1, 12], **python_arguments).fit(np.asarray(r["y"], dtype=float))
    assert fit.model_name == r["model"][0]
    assert fit.coef_names == r["names"]
    np.testing.assert_allclose(fit.coef, r["B"], rtol=1e-6, atol=1e-8)
    assert fit.loglik == pytest.approx(r["logLik"][0], rel=1e-10)


def test_the_selection_agrees_on_bjsales():
    r = _r_fit("BJsales", "", ", ICs=unname(m$ICs)")
    fit = TBATS().fit(np.asarray(r["y"], dtype=float))
    assert fit.model_name == r["model"][0]
    np.testing.assert_allclose(fit.coef, r["B"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(list(fit.ics.values()), r["ICs"], rtol=1e-10)


def test_the_arma_selection_agrees_on_harmonics_with_an_ar():
    r = r_dict(
        "{ set.seed(41); tt <- 1:240;"
        " y <- ts(500 + 20*sin(2*pi*tt/12) + 10*cos(2*pi*tt/12) +"
        " arima.sim(list(ar=0.7), 240, sd=5), frequency=12);"
        " m <- tbats(y, harmonics=1, trend='none');"
        " list(y=as.numeric(y), model=m$model, B=unname(m$B), ICs=unname(m$ICs),"
        " ICnames=names(m$ICs)) }"
    )
    fit = TBATS(lags=[1, 12], harmonics=[1], trend="none").fit(
        np.asarray(r["y"], dtype=float)
    )
    assert fit.model_name == r["model"][0]
    assert fit.orders_["ar"] == [1]
    assert list(fit.ics) == r["ICnames"]
    np.testing.assert_allclose(list(fit.ics.values()), r["ICs"], rtol=1e-10)
    np.testing.assert_allclose(fit.coef, r["B"], rtol=1e-6, atol=1e-8)


def test_the_forecasts_and_the_covariance_agree():
    r = _r_fit(
        "AirPassengers",
        CASES["damped-arma"][0],
        ", f=list(mean=as.numeric(forecast(m, h=12, interval='prediction')$mean),"
        " lower=as.numeric(forecast(m, h=12, interval='prediction')$lower),"
        " upper=as.numeric(forecast(m, h=12, interval='prediction')$upper)),"
        " se=unname(sqrt(diag(vcov(m)))), pointLik=as.numeric(pointLik(m))",
    )
    fit = TBATS(lags=[1, 12], **CASES["damped-arma"][1]).fit(
        np.asarray(r["y"], dtype=float)
    )
    forecast = fit.predict(h=12, interval="prediction")
    np.testing.assert_allclose(np.asarray(forecast.mean), r["f"]["mean"], rtol=1e-9)
    np.testing.assert_allclose(np.ravel(forecast.lower), r["f"]["lower"], rtol=1e-9)
    np.testing.assert_allclose(np.ravel(forecast.upper), r["f"]["upper"], rtol=1e-9)
    np.testing.assert_allclose(fit.point_lik(), r["pointLik"], rtol=1e-9)
    np.testing.assert_allclose(np.sqrt(np.diag(fit.vcov())), r["se"], rtol=1e-6)
