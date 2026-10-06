"""Missing values in ADAM, OM and OMG against R.

The missing observations are skipped by the fit (no update of the states) and are
neither zeros of the occurrence nor in the likelihood, so the estimates, the
likelihood, its per-observation terms and the covariance agree with R. Skipped by
default (``r_parity`` marker).
"""

import numpy as np
import pytest

from smooth import ADAM, CES, MSARIMA, OM, OMG, TBATS

from ._r_bridge import r_dict

pytestmark = pytest.mark.r_parity

DATA = (
    "set.seed({seed}); y <- rbinom(200,1,0.4)*exp(rnorm(200,2,0.3)); "
    "y[c(10:20,100)] <- NA;"
)

CASES = {
    "adam mixture dgamma": (
        3,
        "adam(y, 'MNN', occurrence='odds-ratio')",
        lambda: ADAM(model="MNN", occurrence="odds-ratio"),
    ),
    "adam mixture dinvgauss": (
        3,
        "adam(y, 'MNN', occurrence='odds-ratio', distribution='dinvgauss')",
        lambda: ADAM(model="MNN", occurrence="odds-ratio", distribution="dinvgauss"),
    ),
    "adam mixture dnorm": (
        3,
        "adam(y, 'MNN', occurrence='odds-ratio', distribution='dnorm')",
        lambda: ADAM(model="MNN", occurrence="odds-ratio", distribution="dnorm"),
    ),
    "om odds-ratio": (
        1,
        "om(y, model='ANN', occurrence='odds-ratio')",
        lambda: OM(model="ANN", occurrence="odds-ratio"),
    ),
    "om fixed": (
        1,
        "om(y, model='ANN', occurrence='fixed')",
        lambda: OM(model="ANN", occurrence="fixed"),
    ),
    "omg": (
        1,
        "omg(y, modelA='ANN', modelB='ANN')",
        lambda: OMG(model_a="ANN", model_b="ANN"),
    ),
    "om ARIMA": (
        1,
        "om(y, 'NNN', orders=list(ar=1,i=1,ma=1), occurrence='odds-ratio')",
        lambda: OM(
            model="NNN",
            orders={"ar": [1], "i": [1], "ma": [1]},
            occurrence="odds-ratio",
        ),
    ),
    "omg ARIMA": (
        1,
        "omg(y, modelA='NNN', modelB='NNN', ordersA=list(ar=1,i=1,ma=1),"
        " ordersB=list(ar=1,i=1,ma=1))",
        lambda: OMG(
            model_a="NNN",
            model_b="NNN",
            orders_a={"ar": [1], "i": [1], "ma": [1]},
            orders_b={"ar": [1], "i": [1], "ma": [1]},
        ),
    ),
    "omg gradient": (
        1,
        "omg(y, modelA='ANN', modelB='MNN', initial='gradient')",
        lambda: OMG(model_a="ANN", model_b="MNN", initial="gradient"),
    ),
}


def _fit(case):
    seed, r_call, make = CASES[case]
    r = r_dict(
        f"{{ {DATA.format(seed=seed)} m <- suppressWarnings({r_call}); "
        "list(y=y, ll=as.numeric(logLik(m)), nobs=attr(logLik(m), 'nobs'), "
        "pl=as.numeric(pointLik(m)), f=as.numeric(fitted(m)), "
        "fc=as.numeric(forecast(m, h=5)$mean)) }"
    )
    y = np.array([np.nan if v is None or v == "NA" else v for v in r["y"]], float)
    return make().fit(y), r


@pytest.mark.parametrize("case", list(CASES))
def test_the_fit_with_missing_values_agrees(case):
    fit, r = _fit(case)
    assert fit.loglik == pytest.approx(r["ll"][0], abs=1e-8)
    np.testing.assert_allclose(fit.point_lik(), r["pl"], atol=1e-8)
    np.testing.assert_allclose(np.asarray(fit.fitted, float), r["f"], atol=1e-8)
    np.testing.assert_allclose(np.asarray(fit.predict(h=5).mean), r["fc"], atol=1e-8)


@pytest.mark.parametrize("case", ["om odds-ratio", "omg", "omg gradient"])
def test_the_covariance_with_missing_values_agrees(case):
    seed, r_call, make = CASES[case]
    r = r_dict(
        f"{{ {DATA.format(seed=seed)} m <- suppressWarnings({r_call}); "
        "list(y=y, v=as.vector(vcov(m))) }"
    )
    y = np.array([np.nan if v is None or v == "NA" else v for v in r["y"]], float)
    fit = make().fit(y)
    np.testing.assert_allclose(np.asarray(fit.vcov()).ravel(), r["v"], rtol=1e-6)


# The gaps (missing and zero values) in the starting values of ARIMA, and the
# occurrence model of ADAM with the orders of the demand sizes
GAP_CASES = {
    "msarima NA": (
        "y <- AirPassengers; y[c(10,50,90)] <- NA;",
        "msarima(y, orders=list(ar=c(1,1),i=c(1,1),ma=c(1,1)), lags=c(1,12))",
        lambda: MSARIMA(orders={"ar": [1, 1], "i": [1, 1], "ma": [1, 1]}, lags=[1, 12]),
    ),
    **{
        f"adam ARIMA {occurrence}": (
            "set.seed(44); y <- ts(rpois(120, 0.7) * (1 + rnorm(120)^2));",
            f"adam(y, 'MNN', orders=list(ar=1, ma=1), occurrence='{occurrence}')",
            lambda occurrence=occurrence: ADAM(
                model="MNN", orders={"ar": [1], "ma": [1]}, occurrence=occurrence
            ),
        )
        for occurrence in ("odds-ratio", "general", "auto", "direct")
    },
    "tbats occurrence select": (
        TBATS_INTERMITTENT := (
            "set.seed(7); y <- ts(exp(2 + 0.4*sin(2*pi*(1:300)/7) + rnorm(300, 0, 0.3))"
            "*rbinom(300, 1, 0.7), frequency=7);"
        ),
        "tbats(y, occurrence='odds-ratio')",
        lambda: TBATS(lags=[1, 7], occurrence="odds-ratio"),
    ),
    "tbats occurrence arma": (
        TBATS_INTERMITTENT,
        "tbats(y, occurrence='odds-ratio', orders=list(ar=1, ma=1, select=FALSE))",
        lambda: TBATS(
            lags=[1, 7],
            occurrence="odds-ratio",
            orders={"ar": 1, "ma": 1, "select": False},
        ),
    ),
}


@pytest.mark.parametrize("case", list(GAP_CASES))
def test_the_fit_over_gaps_agrees(case):
    data, r_call, make = GAP_CASES[case]
    r = r_dict(
        f"{{ {data} m <- suppressWarnings({r_call}); list(y=as.numeric(y),"
        " ll=as.numeric(logLik(m)), B=unname(m$B), model=modelName(m),"
        " fc=as.numeric(forecast(m, h=5)$mean)) }"
    )
    y = np.array([np.nan if v is None or v == "NA" else v for v in r["y"]], float)
    fit = make().fit(y)
    assert fit.loglik == pytest.approx(r["ll"][0], abs=1e-8)
    np.testing.assert_allclose(fit.coef, r["B"], atol=1e-8)
    np.testing.assert_allclose(np.asarray(fit.predict(h=5).mean), r["fc"], atol=1e-8)


# The missing values as gaps in ADAM and TBATS: the filled values seed the states
# only, and everything else takes the observed values
AIRPASSENGERS_GAPS = "y <- as.numeric(AirPassengers); y[c(10,50,51,90,140)] <- NA;"
REGRESSION_GAPS = (
    "set.seed(1); x1 <- rnorm(144); x2 <- as.numeric(AirPassengers)/10+rnorm(144);"
    " y <- as.numeric(AirPassengers); y[c(10,50,51,90)] <- NA;"
    " d <- data.frame(y=y, x1=x1, x2=x2);"
)
INTERMITTENT_GAPS = (
    "set.seed(7); y <- ts(exp(2 + 0.4*sin(2*pi*(1:300)/7) + rnorm(300, 0, 0.3))*"
    "rbinom(300, 1, 0.7), frequency=7); y[c(20:25, 150)] <- NA;"
)
GAPS_CASES = {
    "adam MAM holdout": (
        AIRPASSENGERS_GAPS,
        "adam(ts(y, frequency=12), 'MAM', h=12, holdout=TRUE)",
        lambda: ADAM(model="MAM", lags=[12], h=12, holdout=True),
        False,
    ),
    "adam MSE": (
        AIRPASSENGERS_GAPS,
        "adam(y, 'ANN', loss='MSE')",
        lambda: ADAM(model="ANN", loss="MSE"),
        False,
    ),
    "adam TMSE": (
        AIRPASSENGERS_GAPS,
        "adam(y, 'ANN', loss='TMSE', h=6)",
        lambda: ADAM(model="ANN", loss="TMSE", h=6),
        False,
    ),
    "adam regressors": (
        REGRESSION_GAPS,
        "adam(d, 'MNN')",
        lambda: ADAM(model="MNN"),
        True,
    ),
    "adam select": (
        REGRESSION_GAPS,
        "adam(d, 'MNN', regressors='select')",
        lambda: ADAM(model="MNN", regressors="select"),
        True,
    ),
    "adam regression": (
        REGRESSION_GAPS,
        "adam(d, 'NNN')",
        lambda: ADAM(model="NNN"),
        True,
    ),
    "ces": (
        AIRPASSENGERS_GAPS,
        "ces(ts(y, frequency=12), seasonality='full')",
        lambda: CES(seasonality="full", lags=[12]),
        False,
    ),
    "ces TMSE": (
        AIRPASSENGERS_GAPS,
        "ces(ts(y, frequency=12), seasonality='partial', loss='TMSE', h=6)",
        lambda: CES(seasonality="partial", lags=[12], loss="TMSE", h=6),
        False,
    ),
    "tbats": (
        AIRPASSENGERS_GAPS,
        "tbats(ts(y, frequency=12), lags=c(1,12))",
        lambda: TBATS(lags=[1, 12]),
        False,
    ),
    "tbats arma": (
        AIRPASSENGERS_GAPS,
        "tbats(ts(y, frequency=12), lags=c(1,12), "
        "orders=list(ar=1, ma=1, select=FALSE))",
        lambda: TBATS(lags=[1, 12], orders={"ar": 1, "ma": 1, "select": False}),
        False,
    ),
    "tbats occurrence": (
        INTERMITTENT_GAPS,
        "tbats(y, lags=c(1,7), occurrence='odds-ratio')",
        lambda: TBATS(lags=[1, 7], occurrence="odds-ratio"),
        False,
    ),
}


def _values(values):
    return np.array([np.nan if v is None or v == "NA" else v for v in values], float)


@pytest.mark.parametrize("case", list(GAPS_CASES))
def test_the_gaps_agree(case):
    import pandas as pd

    data, r_call, make, regressors = GAPS_CASES[case]
    r = r_dict(
        f"{{ {data} m <- suppressWarnings({r_call}); list(y=as.numeric(y),"
        " x1=if(exists('x1')) x1 else NA, x2=if(exists('x2')) x2 else NA,"
        " ll=as.numeric(logLik(m)), aicc=AICc(m), f=as.numeric(fitted(m)),"
        " res=as.numeric(residuals(m)),"
        " acc=if(is.null(m$accuracy)) NA else unname(m$accuracy[c('ME','MAE')])) }"
    )
    y = _values(r["y"])
    X = pd.DataFrame({"x1": r["x1"], "x2": r["x2"]}) if regressors else None
    fit = make().fit(y, X) if regressors else make().fit(y)
    assert fit.loglik == pytest.approx(r["ll"][0], abs=1e-8)
    assert fit.aicc == pytest.approx(r["aicc"][0], abs=1e-6)
    fitted = np.asarray(fit.fitted, dtype=float)
    np.testing.assert_allclose(fitted, _values(r["f"])[: len(fitted)], atol=1e-6)
    residuals = np.asarray(fit.residuals, dtype=float)
    expected = _values(r["res"])[: len(residuals)]
    np.testing.assert_array_equal(np.isnan(residuals), np.isnan(expected))
    np.testing.assert_allclose(residuals, expected, atol=1e-6)
    if r["acc"][0] not in (None, "NA"):
        assert fit.accuracy["ME"] == pytest.approx(r["acc"][0], abs=1e-8)
        assert fit.accuracy["MAE"] == pytest.approx(r["acc"][1], abs=1e-8)


def test_the_scale_model_over_gaps_agrees():
    """sm() of a model with missing values: the gaps are neither in the loss nor
    an occurrence of the scale, as in R."""
    from smooth import sm

    r = r_dict(
        f"{{ {AIRPASSENGERS_GAPS} m <- suppressWarnings(adam(ts(y, frequency=12),"
        " 'MAM')); s <- suppressWarnings(sm(m)); list(y=as.numeric(y),"
        " ll=as.numeric(logLik(s)), B=unname(s$B)) }"
    )
    fit = sm(ADAM(model="MAM", lags=[12]).fit(_values(r["y"])))
    assert fit.loglik == pytest.approx(r["ll"][0], abs=1e-8)
    np.testing.assert_allclose(fit.coef, r["B"], atol=1e-8)
