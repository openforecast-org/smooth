"""R-Python parity of TBATS: the same structure, parameters, likelihood, forecasts
and covariance as R's tbats() on the same data."""

import warnings

import numpy as np
import pytest

from smooth import OM, TBATS
from tests._r_bridge import r_dict, r_eval

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


def test_confint_agrees():
    r = _r_fit(
        "AirPassengers",
        CASES["damped-arma"][0],
        ", ci=unname(confint(m)[,2:3])",
    )
    fit = TBATS(lags=[1, 12], **CASES["damped-arma"][1]).fit(
        np.asarray(r["y"], dtype=float)
    )
    np.testing.assert_allclose(
        fit.confint().iloc[:, 1:].to_numpy(), r["ci"], rtol=1e-5, atol=1e-9
    )


# A series with two regressors (the second is noise) and a third one of noise for the
# selection; a series whose coefficient of the first regressor drifts
XREG_DATA = (
    "set.seed(41); xr <- cbind(x1=rnorm(132, 10, 2), x2=rnorm(132));"
    " y <- ts(200 + 20*sin(2*pi*(1:132)/12) + 5*xr[,'x1'] + cumsum(rnorm(132)) +"
    " rnorm(132), frequency=12);"
    " set.seed(7); xs <- cbind(xr, noise=rnorm(132));"
    " set.seed(41); xa <- cbind(x1=rnorm(150, 10, 2), x2=rnorm(150));"
    " ya <- 200 + (5+cumsum(rnorm(150, 0, 0.05)))*xa[,'x1'] - 2*xa[,'x2'] +"
    " cumsum(rnorm(150)) + rnorm(150);"
)
R_ORDERS0 = "orders=list(ar=0, ma=0, select=FALSE)"

# (R call, the data of y and X, Python arguments) of the fits with regressors
XREG_CASES = {
    "use": (
        f"tbats(y, xreg=xr, harmonics=1, trend='none', {R_ORDERS0}, lambda=1,"
        " h=12, holdout=TRUE)",
        ("y", "xr"),
        dict(
            lags=[1, 12],
            harmonics=[1],
            trend="none",
            orders=ORDERS0,
            lambda_bc=1,
            h=12,
            holdout=True,
        ),
    ),
    "selection": (
        "tbats(y, xreg=xr, h=12, holdout=TRUE)",
        ("y", "xr"),
        dict(lags=[1, 12], h=12, holdout=True),
    ),
    "two-stage": (
        "tbats(ts(y[1:120], frequency=12), xreg=xr[1:120,], harmonics=1,"
        " trend='additive', orders=list(ar=1, ma=0, select=FALSE),"
        " initial='two-stage')",
        ("y[1:120]", "xr[1:120,]"),
        dict(
            lags=[1, 12],
            harmonics=[1],
            trend="additive",
            orders={"ar": 1, "ma": 0, "select": False},
            initial="two-stage",
        ),
    ),
    "complete": (
        f"tbats(y, xreg=xr, harmonics=1, trend='none', {R_ORDERS0},"
        " initial='complete', h=12, holdout=TRUE)",
        ("y", "xr"),
        dict(
            lags=[1, 12],
            harmonics=[1],
            trend="none",
            orders=ORDERS0,
            initial="complete",
            h=12,
            holdout=True,
        ),
    ),
    "adapt": (
        f"tbats(ya, lags=1, xreg=xa, regressors='adapt', trend='none', {R_ORDERS0})",
        ("ya", "xa"),
        dict(lags=[1], regressors="adapt", trend="none", orders=ORDERS0),
    ),
    "adapt-usual": (
        f"tbats(ya, lags=1, xreg=xa, regressors='adapt', trend='none', {R_ORDERS0},"
        " bounds='usual')",
        ("ya", "xa"),
        dict(
            lags=[1], regressors="adapt", trend="none", orders=ORDERS0, bounds="usual"
        ),
    ),
    "select": (
        "tbats(y, xreg=xs, regressors='select', h=12, holdout=TRUE)",
        ("y", "xs"),
        dict(lags=[1, 12], regressors="select", h=12, holdout=True),
    ),
}


def _r_xreg_fit(case, extra=""):
    call, (y, X), _ = XREG_CASES[case]
    return r_dict(
        f"{{ {XREG_DATA} m <- suppressWarnings({call});"
        f" list(y=as.numeric({y}), X={X}, model=m$model, B=unname(m$B),"
        " names=names(m$B), logLik=as.numeric(logLik(m)), ICs=unname(m$ICs),"
        f" ICnames=names(m$ICs), forecast=as.numeric(m$forecast){extra}) }}"
    )


@pytest.mark.parametrize("case", list(XREG_CASES))
def test_the_fits_with_regressors_agree(case):
    r = _r_xreg_fit(case)
    fit = TBATS(**XREG_CASES[case][2]).fit(
        np.asarray(r["y"], dtype=float), np.asarray(r["X"], dtype=float)
    )
    assert fit.model_name == r["model"][0]
    assert fit.coef_names == r["names"]
    np.testing.assert_allclose(fit.coef, r["B"], rtol=1e-6, atol=1e-8)
    assert fit.loglik == pytest.approx(r["logLik"][0], rel=1e-10)
    assert list(fit.ics) == r["ICnames"]
    np.testing.assert_allclose(list(fit.ics.values()), r["ICs"], rtol=1e-10)
    if fit.forecast_ is not None:
        np.testing.assert_allclose(fit.forecast_, r["forecast"], rtol=1e-10)


def test_the_intervals_with_new_regressors_agree():
    newdata = "xr[121:132,] + matrix(c(10, 0), 12, 2, byrow=TRUE)"
    r = _r_xreg_fit(
        "use",
        f", f={{ f <- forecast(m, h=12, newdata={newdata}, interval='prediction');"
        " list(mean=as.numeric(f$mean), lower=as.numeric(f$lower),"
        f" upper=as.numeric(f$upper)) }}, Xnew={newdata}",
    )
    fit = TBATS(**XREG_CASES["use"][2]).fit(
        np.asarray(r["y"], dtype=float), np.asarray(r["X"], dtype=float)
    )
    forecast = fit.predict(
        h=12, X=np.asarray(r["Xnew"], dtype=float), interval="prediction"
    )
    np.testing.assert_allclose(np.asarray(forecast.mean), r["f"]["mean"], rtol=1e-10)
    np.testing.assert_allclose(np.ravel(forecast.lower), r["f"]["lower"], rtol=1e-9)
    np.testing.assert_allclose(np.ravel(forecast.upper), r["f"]["upper"], rtol=1e-9)


@pytest.mark.parametrize("lam", ["NULL", "0"])
def test_the_mean_by_quadrature_agrees(lam):
    r = _r_fit(
        "AirPassengers",
        CASES["additive"][0] + f", lambda={lam}",
        ", mean=as.numeric(forecast(m, h=24, point='mean')$mean)",
    )
    fit = TBATS(
        lags=[1, 12],
        lambda_bc=None if lam == "NULL" else 0.0,
        **CASES["additive"][1],
    ).fit(np.asarray(r["y"], dtype=float))
    np.testing.assert_allclose(
        fit.predict(h=24, point="mean").mean, r["mean"], rtol=1e-10
    )


INTERMITTENT = (
    "set.seed(7); y <- ts(exp(2 + 0.4*sin(2*pi*(1:300)/7) + rnorm(300, 0, 0.3))*"
    "rbinom(300, 1, 0.7), frequency=7);"
)
OCCURRENCE_CASES = {
    "odds-ratio": ("'odds-ratio'", lambda y: "odds-ratio"),
    "general": ("'general'", lambda y: "general"),
    "seasonal om": (
        "om(y, model='ANA', lags=c(1,7), occurrence='odds-ratio')",
        lambda y: OM(model="ANA", lags=[1, 7], occurrence="odds-ratio").fit(y),
    ),
    "probabilities": (
        "c(rep(0.7, 300), rep(0.6, 3))",
        lambda y: np.r_[np.full(300, 0.7), np.full(3, 0.6)],
    ),
}


@pytest.mark.parametrize("case", list(OCCURRENCE_CASES))
def test_the_occurrence_mixture_agrees(case):
    r_occurrence, py_occurrence = OCCURRENCE_CASES[case]
    r = r_dict(
        f"{{ {INTERMITTENT} m <- tbats(y, occurrence={r_occurrence},"
        " orders=list(ar=0, ma=0, select=FALSE));"
        " both <- forecast(m, h=7, interval='prediction', level=c(0.8, 0.95));"
        " list(y=as.numeric(y), B=unname(m$B), logLik=as.numeric(logLik(m)),"
        " nparam=nparam(m), fitted=as.numeric(fitted(m)),"
        " skeleton=as.numeric(both$mean),"
        " lower=as.numeric(both$lower), upper=as.numeric(both$upper),"
        " mean=as.numeric(forecast(m, h=7, point='mean')$mean),"
        " median=as.numeric(forecast(m, h=7, point='median')$mean),"
        " side=as.numeric(forecast(m, h=7, interval='prediction',"
        " side='upper')$upper)) }"
    )
    y = np.asarray(r["y"], dtype=float)
    fit = TBATS(lags=[1, 7], occurrence=py_occurrence(y), orders=ORDERS0).fit(y)
    np.testing.assert_allclose(fit.coef, r["B"], rtol=1e-8, atol=1e-10)
    assert fit.loglik == pytest.approx(r["logLik"][0], rel=1e-10)
    assert fit.nparam == r["nparam"][0]
    np.testing.assert_allclose(fit.fitted, r["fitted"], rtol=1e-8)
    both = fit.predict(h=7, interval="prediction", level=[0.8, 0.95])
    np.testing.assert_allclose(both.mean, r["skeleton"], rtol=1e-10)
    np.testing.assert_allclose(both.lower.to_numpy().T.ravel(), r["lower"], atol=1e-10)
    np.testing.assert_allclose(both.upper.to_numpy().T.ravel(), r["upper"], rtol=1e-10)
    mean = fit.predict(h=7, point="mean").mean
    np.testing.assert_allclose(mean, r["mean"], rtol=1e-10)
    np.testing.assert_allclose(
        fit.predict(h=7, point="median").mean, r["median"], rtol=1e-10, atol=1e-12
    )
    upper = fit.predict(h=7, interval="prediction", side="upper").upper
    np.testing.assert_allclose(np.ravel(upper), r["side"], rtol=1e-10)


MISSING_XREG_CASES = {
    "estimated lambda": (
        "y",
        "xr",
        f"harmonics=1, trend='none', {R_ORDERS0}, h=12, holdout=TRUE",
        dict(
            lags=[1, 12],
            harmonics=[1],
            trend="none",
            orders=ORDERS0,
            h=12,
            holdout=True,
        ),
    ),
    "adapt": (
        "ya",
        "xa",
        f"lags=1, regressors='adapt', trend='none', {R_ORDERS0}",
        dict(lags=[1], regressors="adapt", trend="none", orders=ORDERS0),
    ),
}


@pytest.mark.parametrize("case", list(MISSING_XREG_CASES))
def test_the_fits_with_missing_regressors_agree(case):
    y, X, r_arguments, arguments = MISSING_XREG_CASES[case]
    r = r_dict(
        f"{{ {XREG_DATA} X <- {X}; {X}[c(15, 60), 1] <- NA;"
        f" m <- suppressWarnings(tbats({y}, xreg={X}, {r_arguments}));"
        f" list(y=as.numeric({y}), X=X, B=unname(m$B), logLik=as.numeric(logLik(m)),"
        " fitted=replace(as.numeric(fitted(m)), c(15, 60), 0),"
        " forecast=as.numeric(m$forecast)) }"
    )
    X = np.asarray(r["X"], dtype=float)
    X[[14, 59], 0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = TBATS(**arguments).fit(np.asarray(r["y"], dtype=float), X)
    np.testing.assert_allclose(fit.coef, r["B"], rtol=1e-6, atol=1e-8)
    assert fit.loglik == pytest.approx(r["logLik"][0], rel=1e-10)
    # No fitted values without the regressors
    assert np.all(np.isnan(fit.fitted[[14, 59]]))
    np.testing.assert_allclose(np.nan_to_num(fit.fitted), r["fitted"], rtol=1e-8)
    if fit.forecast_ is not None:
        np.testing.assert_allclose(fit.forecast_, r["forecast"], rtol=1e-10)


def test_the_cumulative_skeletons_agree():
    r = r_dict(
        f"{{ {INTERMITTENT} m <- tbats(y, occurrence='odds-ratio', {R_ORDERS0});"
        " a <- tbats(AirPassengers, harmonics=5, trend='additive',"
        f" {R_ORDERS0});"
        " list(y=as.numeric(y), mixture=as.numeric(forecast(m, h=7,"
        " cumulative=TRUE)$mean), air=as.numeric(forecast(a, h=12,"
        " cumulative=TRUE)$mean)) }"
    )
    y = np.asarray(r["y"], dtype=float)
    fit = TBATS(lags=[1, 7], occurrence="odds-ratio", orders=ORDERS0).fit(y)
    np.testing.assert_allclose(
        fit.predict(h=7, cumulative=True).mean, r["mixture"], rtol=1e-10
    )
    air = TBATS(lags=[1, 12], harmonics=[5], trend="additive", orders=ORDERS0).fit(
        np.asarray(r_dict("list(y=as.numeric(AirPassengers))")["y"], dtype=float)
    )
    np.testing.assert_allclose(
        air.predict(h=12, cumulative=True).mean, r["air"], rtol=1e-10
    )


GRADIENT_CASES = {
    "additive": (
        f"harmonics=5, trend='additive', {R_ORDERS0}",
        dict(harmonics=[5], trend="additive", orders=ORDERS0),
    ),
    "damped arma": (
        "harmonics=5, trend='damped', orders=list(ar=1, ma=1, select=FALSE)",
        dict(
            harmonics=[5],
            trend="damped",
            orders={"ar": 1, "ma": 1, "select": False},
        ),
    ),
    "MAE": (
        f"harmonics=5, trend='none', loss='MAE', {R_ORDERS0}",
        dict(harmonics=[5], trend="none", loss="MAE", orders=ORDERS0),
    ),
    "TMSE": (
        f"harmonics=5, trend='additive', loss='TMSE', h=6, {R_ORDERS0}",
        dict(harmonics=[5], trend="additive", loss="TMSE", h=6, orders=ORDERS0),
    ),
}


@pytest.mark.parametrize("case", list(GRADIENT_CASES))
def test_the_gradient_initials_agree(case):
    r_arguments, arguments = GRADIENT_CASES[case]
    r = r_dict(
        "{ m <- suppressWarnings(tbats(AirPassengers, initial='gradient',"
        f" {r_arguments}));"
        " list(y=as.numeric(AirPassengers), B=unname(m$B), ll=as.numeric(logLik(m)),"
        " nparam=nparam(m), fc=as.numeric(forecast(m, h=12)$mean)) }"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = TBATS(lags=[1, 12], initial="gradient", **arguments).fit(
            np.asarray(r["y"], dtype=float)
        )
    np.testing.assert_allclose(fit.coef, r["B"], rtol=1e-6, atol=1e-8)
    assert fit.loglik == pytest.approx(r["ll"][0], rel=1e-10)
    assert fit.nparam == r["nparam"][0]
    np.testing.assert_allclose(fit.predict(h=12).mean, r["fc"], rtol=1e-10)


# A custom loss of R and its Python twin, and a multistep loss over gaps
CUSTOM_LOSS_R = "function(actual, fitted, B) mean(abs(actual-fitted)^1.5)"
CUSTOM_DATA = {
    "custom": "y <- as.numeric(AirPassengers);",
    "custom gaps": "y <- as.numeric(AirPassengers); y[c(10,50,51)] <- NA;",
    "TMSE gaps": "y <- as.numeric(AirPassengers); y[c(10,50,51)] <- NA;",
}


def _custom_loss(actual, fitted, B):
    return float(np.mean(np.abs(actual - fitted) ** 1.5))


@pytest.mark.parametrize("case", list(CUSTOM_DATA))
def test_the_custom_and_multistep_losses_agree(case):
    multistep = case == "TMSE gaps"
    r_loss = "loss='TMSE', h=6" if multistep else f"loss={CUSTOM_LOSS_R}"
    r = r_dict(
        f"{{ {CUSTOM_DATA[case]} m <- suppressWarnings(tbats(ts(y, frequency=12),"
        f" harmonics=5, trend='additive', {R_ORDERS0}, {r_loss}));"
        " list(y=y, B=unname(m$B), loss=m$lossValue, ll=as.numeric(logLik(m)),"
        " fc=as.numeric(forecast(m, h=6)$mean)) }"
    )
    y = np.array([np.nan if v in (None, "NA") else v for v in r["y"]], float)
    loss = dict(loss="TMSE", h=6) if multistep else dict(loss=_custom_loss)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = TBATS(
            lags=[1, 12], harmonics=[5], trend="additive", orders=ORDERS0, **loss
        ).fit(y)
    np.testing.assert_allclose(fit.coef, r["B"], rtol=1e-6, atol=1e-8)
    assert fit.loss_value == pytest.approx(r["loss"][0], rel=1e-10)
    assert fit.loglik == pytest.approx(r["ll"][0], rel=1e-10)
    np.testing.assert_allclose(fit.predict(h=6).mean, r["fc"], rtol=1e-10)


def test_the_diagnostics_agree():
    """The diagnostics shared with ADAM, on the residuals of the transformed data."""
    r = r_dict(
        "{ m <- tbats(AirPassengers, harmonics=5, trend='damped',"
        " orders=list(ar=1, ma=0, select=FALSE), h=12, holdout=TRUE);"
        " list(y=as.numeric(AirPassengers), rstandard=as.numeric(rstandard(m)),"
        " rstudent=as.numeric(rstudent(m)), sigma=sigma(m),"
        " rmultistep=as.numeric(rmultistep(m, h=3)),"
        " analytical=as.numeric(multicov(m, h=3)),"
        " empirical=as.numeric(multicov(m, type='empirical', h=3))) }"
    )
    fit = TBATS(
        lags=[1, 12],
        harmonics=[5],
        trend="damped",
        orders={"ar": 1, "ma": 0, "select": False},
        h=12,
        holdout=True,
    ).fit(np.asarray(r["y"], dtype=float))
    np.testing.assert_allclose(fit.rstandard(), r["rstandard"], rtol=1e-8)
    np.testing.assert_allclose(fit.rstudent(), r["rstudent"], rtol=1e-8)
    np.testing.assert_allclose(fit.sigma, r["sigma"], rtol=1e-8)
    np.testing.assert_allclose(
        fit.rmultistep(h=3).to_numpy().ravel(order="F"), r["rmultistep"], rtol=1e-8
    )
    np.testing.assert_allclose(
        fit.multicov(h=3).to_numpy().ravel(), r["analytical"], rtol=1e-8
    )
    np.testing.assert_allclose(
        fit.multicov(type="empirical", h=3).to_numpy().ravel(),
        r["empirical"],
        rtol=1e-8,
    )


def test_the_eigenvalue_moduli_are_those_of_r_to_the_bit():
    # The admissible bounds take them from the routine shared with R, written without
    # LAPACK: the optima lie on the boundary, where two LAPACK builds disagree
    from smooth.adam_general._eigenCalc import eigen_moduli

    rng = np.random.default_rng(3)
    for n in (2, 7, 21):
        A = rng.normal(size=(n, n))
        values = ",".join(repr(float(v)) for v in A.ravel(order="F"))
        r = r_eval(f"sprintf('%a', eigenModuliCpp(matrix(c({values}), {n})))")
        python = np.ravel(eigen_moduli(np.asfortranarray(A)))
        assert [float.fromhex(v) for v in np.atleast_1d(r)] == list(python)
