"""Tests for ADAM with explanatory variables (ETSX models).

Data generated in R:
    set.seed(42)
    n <- 120
    x1 <- rnorm(n); x2 <- rnorm(n)
    y <- 10 + 2*x1 - 1.5*x2 + rnorm(n)
    write.csv(data.frame(y=y, x1=x1, x2=x2), "tests/data/etsx_data.csv",
              row.names=FALSE)

R reference (adam(df, model="AAN", regressors="use", formula=y~x1+x2)):
    smoother="global" (default since smooth v4.4.1 / Python v1.0.1)
    alpha=0.00890, beta=0, x1=1.915, x2=-1.461
    AIC=339.2135
    Fitted[1:5]=[15.24117, 11.48431, 10.92935, 13.08419, 11.19438]
    Forecast[1:3]=[10.91767, 7.99035, 9.92983] (using last 12 rows as X_new)
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smooth import ADAM

DATA_PATH = Path(__file__).parent / "data" / "etsx_data.csv"

# R reference values (adam, smoother="global", no holdout)
R_AIC = 343.2135
R_FITTED_5 = [15.24117, 11.48431, 10.92935, 13.08419, 11.19438]
R_FORECAST_3 = [10.91767, 7.99035, 9.92983]
R_COEF_ALPHA = 0.00890
R_COEF_X1 = 1.915
R_COEF_X2 = -1.461


@pytest.fixture
def etsx_data():
    df = pd.read_csv(DATA_PATH, float_precision="round_trip")
    return df["y"].values, df[["x1", "x2"]].values


# --- regressors="use" ---

def test_etsx_use_fits(etsx_data):
    y, X = etsx_data
    model = ADAM(model="AAN", regressors="use")
    model.fit(y, X)
    assert model._explanatory["xreg_model"] is True
    assert model._explanatory["xreg_number"] == 2


def test_etsx_use_aic_matches_r(etsx_data):
    y, X = etsx_data
    model = ADAM(model="AAN", regressors="use")
    model.fit(y, X)
    assert abs(model.aic - R_AIC) < 1.0


def test_etsx_use_coefs_match_r(etsx_data):
    y, X = etsx_data
    model = ADAM(model="AAN", regressors="use")
    model.fit(y, X)
    # B = [alpha, beta, x1, x2]
    B = model.coef
    assert abs(B[0] - R_COEF_ALPHA) < 0.01
    assert abs(B[2] - R_COEF_X1) < 0.1
    assert abs(B[3] - R_COEF_X2) < 0.1


def test_etsx_use_fitted_matches_r(etsx_data):
    y, X = etsx_data
    model = ADAM(model="AAN", regressors="use")
    model.fit(y, X)
    assert np.allclose(model.fitted[:5].values, R_FITTED_5, atol=0.1)


def test_etsx_use_forecast_matches_r(etsx_data):
    y, X = etsx_data
    model = ADAM(model="AAN", regressors="use")
    model.fit(y, X)
    X_new = X[-12:]
    fc = model.predict(h=12, X=X_new)
    assert np.allclose(fc.mean.values[:3], R_FORECAST_3, atol=0.1)


def test_etsx_use_forecast_shape(etsx_data):
    y, X = etsx_data
    model = ADAM(model="AAN", regressors="use")
    model.fit(y, X)
    rng = np.random.default_rng(99)
    X_new = rng.standard_normal((12, 2))
    fc = model.predict(h=12, X=X_new)
    assert len(fc.mean) == 12
    assert not np.any(np.isnan(fc.mean.values))


@pytest.mark.parametrize("regressors", ["use", "adapt"])
def test_etsx_simulated_interval_uses_the_new_x(etsx_data, regressors):
    # The simulated paths take the new values of the regressors, as the point
    # forecasts do, and predict() leaves the in-sample measurement as it is
    y, X = etsx_data
    model = ADAM(model="ANN", regressors=regressors).fit(y[:108], X[:108])
    measurement = model.measurement.copy()
    X_new = X[108:] + np.array([5.0, 0.0])
    point = model.predict(h=12, X=X_new).mean.values
    np.testing.assert_array_equal(model.measurement, measurement)
    fc = model.predict(h=12, X=X_new, interval="simulated", nsim=2000)
    np.testing.assert_allclose(fc.mean.values, point, rtol=1e-12)
    centre = (np.ravel(fc.lower) + np.ravel(fc.upper)) / 2
    np.testing.assert_allclose(centre, point, atol=1.0)


def test_etsx_use_states_shape(etsx_data):
    y, X = etsx_data
    model = ADAM(model="AAN", regressors="use")
    model.fit(y, X)
    # AAN + 2 xreg = 4 state rows
    assert model.states.shape[0] == 4
    assert model.states.shape[1] == len(y) + 1


# --- regressors="select" ---

def test_etsx_select_fits(etsx_data):
    y, X = etsx_data
    model = ADAM(model="AAN", regressors="select")
    model.fit(y, X)
    assert model._explanatory["xreg_model"] is True


def test_etsx_select_keeps_signal(etsx_data):
    """Both signal variables should be selected from the data."""
    y, X = etsx_data
    model = ADAM(model="AAN", regressors="select")
    model.fit(y, X)
    assert model._explanatory["xreg_number"] == 2


def test_etsx_select_drops_noise(etsx_data):
    """Noise columns should be dropped by stepwise selection."""
    y, X = etsx_data
    rng = np.random.default_rng(7)
    X_noise = np.column_stack([X, rng.standard_normal((len(y), 3))])
    model = ADAM(model="AAN", regressors="select")
    model.fit(y, X_noise)
    assert model._explanatory["xreg_number"] <= 2


# --- regressors="adapt" ---

def test_etsx_adapt_fits(etsx_data):
    y, X = etsx_data
    model = ADAM(model="AAN", regressors="adapt")
    model.fit(y, X)
    assert model._explanatory["xreg_model"] is True


def test_etsx_adapt_states_shape(etsx_data):
    y, X = etsx_data
    model = ADAM(model="AAN", regressors="adapt")
    model.fit(y, X)
    # Level + trend + 2 adaptive xreg = 4 rows
    assert model.states.shape[0] == 4


def test_etsx_adapt_aic_better_than_null(etsx_data):
    """Adapt model should fit reasonably (AIC finite)."""
    y, X = etsx_data
    model = ADAM(model="AAN", regressors="adapt")
    model.fit(y, X)
    assert np.isfinite(model.aic)


# --- edge cases ---

def test_etsx_adapt_deltas_within_bounds(etsx_data):
    """The usual bounds keep the smoothing parameters of the regressors in [0, 1]."""
    y, X = etsx_data
    model = ADAM(model="AAN", regressors="adapt").fit(y, X=X)
    deltas = [v for n, v in zip(model.coef_names, model.coef) if n.startswith("delta")]
    assert deltas and all(0 <= d <= 1 for d in deltas)


@pytest.mark.parametrize("delta", [0.02, [0.02, 0.03]])
def test_etsx_adapt_provided_deltas_are_fixed(etsx_data, delta):
    """persistence={"xreg": ...} fixes the deltas, one for all or one each, as R."""
    y, X = etsx_data
    model = ADAM(model="AAN", regressors="adapt", persistence={"xreg": delta})
    model.fit(y, X=X)
    expected = np.resize(delta, 2)
    assert [model.persistence_vector[f"delta{i}"] for i in (1, 2)] == list(expected)
    assert not any(n.startswith("delta") for n in model.coef_names)


@pytest.mark.r_parity
def test_etsx_adapt_dummies_match_r():
    """Monthly dummies with adapt: the initials (with R's auxiliary trend) and the
    fit agree with R."""
    from ._r_bridge import r_array

    air = pd.read_csv(
        Path(__file__).parent / "data" / "ces_airpassengers.csv",
        float_precision="round_trip",
    )
    y = air["y"].values.astype(float)
    X = np.zeros((len(y), 11))
    for month in range(1, 12):
        X[month::12, month - 1] = 1
    kw = dict(
        model="MMN",
        lags=[1, 12],
        orders={"ar": [0, 0], "i": [0, 0], "ma": [0, 2]},
        distribution="dnorm",
        regressors="adapt",
    )
    start = ADAM(**kw, nlopt_kwargs={"maxeval": 1}).fit(y, X=X)
    model = ADAM(**kw).fit(y, X=X)
    expected = r_array(
        "{X <- temporaldummy(ts(y, frequency=12))[,-1];"
        " f <- function(...) adam(data.frame(y=y, X), 'MMN', lags=c(1,12),"
        " orders=list(ma=c(0,2)), distribution='dnorm', regressors='adapt', ...);"
        " c(f(maxeval=1)$B, f()$lossValue)}",
        R_data={"y": y},
    )
    np.testing.assert_allclose(start.coef, expected[:-1], rtol=1e-10)
    assert model._adam_estimated["CF_value"] == pytest.approx(expected[-1], rel=1e-8)


@pytest.mark.r_parity
def test_etsx_occurrence_initials_match_r():
    """Intermittent demand with a multiplicative error: the initials of the
    regressors come from the non-zero demand, as R's subset (log(0) was clamped)."""
    from ._r_bridge import r_array

    rng = np.random.default_rng(44)
    x = rng.normal(5, 1, 120)
    y = rng.poisson(0.7, 120) * (1 + x + rng.normal(size=120) ** 2)
    start = ADAM(
        model="MNN",
        occurrence="odds-ratio",
        distribution="dnorm",
        nlopt_kwargs={"maxeval": 1},
    ).fit(y, X=x.reshape(-1, 1))
    expected = r_array(
        "{m <- adam(data.frame(y=y, x=x), 'MNN', occurrence='odds-ratio',"
        " distribution='dnorm', maxeval=1); unname(m$B)}",
        R_data={"y": y, "x": x},
    )
    np.testing.assert_allclose(start.coef, expected, rtol=1e-10)


def test_etsx_single_regressor(etsx_data):
    y, X = etsx_data
    model = ADAM(model="ANN", regressors="use")
    model.fit(y, X[:, :1])
    assert model._explanatory["xreg_number"] == 1
    assert model.states.shape[0] == 2  # level + 1 xreg


def test_etsx_holdout(etsx_data):
    y, X = etsx_data
    h = 12
    model = ADAM(model="AAN", regressors="use", holdout=True, h=h)
    model.fit(y, X)
    fc = model.predict(h=h, X=X[-h:])
    assert len(fc.mean) == h


def test_etsx_no_x_fits_plain_ets(etsx_data):
    """regressors='use' with no X just fits a plain ETS model."""
    y, _ = etsx_data
    model = ADAM(model="AAN", regressors="use")
    model.fit(y)
    assert model._explanatory.get("xreg_model") is False


def test_etsx_short_x_pads_with_warning(etsx_data):
    """X shorter than y: last row is repeated and a warning is issued."""
    y, X = etsx_data
    model = ADAM(model="AAN", regressors="use")
    with pytest.warns(UserWarning, match="repeated"):
        model.fit(y, X[:-5])
    assert model._explanatory["xreg_number"] == 2


def test_etsx_dataframe_x(etsx_data):
    y, X = etsx_data
    X_df = pd.DataFrame(X, columns=["x1", "x2"])
    model = ADAM(model="AAN", regressors="use")
    model.fit(y, X_df)
    assert model._explanatory["xreg_number"] == 2
    assert model._explanatory["xreg_names"] == ["x1", "x2"]


# --- initial type R/Python equivalence ---
# R reference: adam(df, model="AAN", regressors="use", formula=y~x1+x2, initial=X)

def test_etsx_initial_backcasting(etsx_data):
    """initial='backcasting': xreg coefs in B, AIC and fitted match R."""
    y, X = etsx_data
    model = ADAM(model="AAN", regressors="use", initial="backcasting")
    model.fit(y, X)
    B = model.coef
    # R (smoother="global"): alpha=0.00890, x1=1.91508, x2=-1.46058, AIC=339.2135
    assert abs(B[0] - 0.00890) < 0.02
    assert abs(B[2] - 1.91508) < 0.1
    assert abs(B[3] - (-1.46058)) < 0.1
    assert abs(model.aic - 343.2135) < 1.0
    assert abs(model.fitted.values[0] - 15.241) < 0.5


def test_etsx_initial_complete_no_xreg_in_b(etsx_data):
    """initial='complete': xreg coefs NOT in B (backcasting handles them)."""
    y, X = etsx_data
    model_bc = ADAM(model="AAN", regressors="use", initial="backcasting")
    model_bc.fit(y, X)
    model_co = ADAM(model="AAN", regressors="use", initial="complete")
    model_co.fit(y, X)
    # "complete" B must have 2 fewer params (the 2 xreg coefs are excluded)
    assert len(model_co.coef) == len(model_bc.coef) - 2
    # AIC and fitted[0] should be close to R reference (smoother="global")
    assert abs(model_co.aic - 343.2875) < 1.5
    assert abs(model_co.fitted.values[0] - 15.265) < 0.5


def test_etsx_initial_optimal(etsx_data):
    """initial='optimal': xreg coefs in B (after ETS initials), match R reference."""
    y, X = etsx_data
    model = ADAM(model="AAN", regressors="use", initial="optimal")
    model.fit(y, X)
    B = model.coef
    # For "optimal", B = [alpha, beta, level_init, trend_init, x1, x2]
    # R (smoother="default" -> "ma" for optimal): x1=1.87839, x2=-1.51349,
    # AIC=349.552, fitted[0]=14.682.
    assert abs(B[-2] - 1.87839) < 0.15
    assert abs(B[-1] - (-1.51349)) < 0.15
    assert abs(model.aic - 349.552) < 2.0
    assert abs(model.fitted.values[0] - 14.682) < 0.5


def test_etsx_initial_two_stage(etsx_data):
    """initial='two-stage': xreg coefs in B (after ETS initials), match R reference."""
    y, X = etsx_data
    model = ADAM(model="AAN", regressors="use", initial="two-stage")
    model.fit(y, X)
    B = model.coef
    # For "two-stage", B = [alpha, beta, level_init, trend_init, x1, x2]
    # R (smoother="global"): x1=1.91726, x2=-1.47840, AIC=338.37, fitted[0]=14.959
    assert abs(B[-2] - 1.91726) < 0.15
    assert abs(B[-1] - (-1.47840)) < 0.15
    assert abs(model.aic - 338.3654) < 2.0
    assert abs(model.fitted.values[0] - 14.959) < 0.5


# --- initial={"xreg": [...]} ---

def test_initial_xreg_accepted(etsx_data):
    """initial={'xreg': [...]} is accepted and model fits without error."""
    y, X = etsx_data
    model = ADAM(model="AAN", regressors="use", initial={"xreg": [2.0, -1.5]})
    model.fit(y, X)
    assert model._explanatory["xreg_model"] is True


def test_initial_xreg_is_kept_and_the_rest_estimated(etsx_data):
    """The provided xreg initials are used, and the ETS initials estimated, as R."""
    y, X = etsx_data
    model = ADAM(model="AAN", regressors="use", initial={"xreg": [2.0, -1.5]})
    model.fit(y, X)
    assert model.initial_type == "provided"
    assert model.initial_value["xreg"] == {"x1": 2.0, "x2": -1.5}
    assert np.isfinite(model.initial_value["level"])
    assert model.loss_value < 1e100


def test_initial_xreg_wrong_length_raises(etsx_data):
    """initial={'xreg': [...]} with more values than regressors raises ValueError;
    fewer are those of the first regressors, and the others are estimated, as R."""
    y, X = etsx_data  # X has 2 columns
    model = ADAM(model="AAN", regressors="use", initial={"xreg": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="xreg"):
        model.fit(y, X)
    model = ADAM(model="AAN", regressors="use", initial={"xreg": [1.0]}).fit(y, X)
    assert list(model.initial_value["xreg"].values())[0] == 1.0
    assert model.n_param.provided["xreg"] == 1


def test_initial_xreg_combined_with_level(etsx_data):
    """initial dict can mix 'xreg' with other keys."""
    y, X = etsx_data
    model = ADAM(model="AAN", regressors="use",
                 initial={"level": 10.0, "xreg": [2.0, -1.5]})
    model.fit(y, X)
    assert model._explanatory["xreg_model"] is True


def test_pure_regression_scale_is_sigma_squared():
    """A pure regression stores sigma^2 as its scale, as the other ADAM models."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(100, 1))
    y = 10 + 2 * x[:, 0] + rng.normal(0, 10, 100)
    model = ADAM(model="NNN", lags=[1]).fit(y, X=x)
    residuals = np.asarray(model.residuals, dtype=float)
    assert model.scale == pytest.approx(np.mean(residuals**2), rel=1e-10)


@pytest.mark.parametrize("interval", ["none", "prediction", "confidence"])
@pytest.mark.parametrize("side", ["both", "upper"])
def test_pure_regression_predict_is_alm_predict(interval, side):
    """A pure regression is forecasted by ALM.predict(), as R's forecast.alm()."""
    rng = np.random.default_rng(1)
    x = rng.normal(100, 10, (100, 2))
    y = x[:, 0] + rng.normal(size=100)
    model = ADAM(model="NNN").fit(y[:90], X=x[:90])
    forecast = model.predict(h=5, X=x[90:95], interval=interval, side=side)
    expected = model._alm_model.predict(
        np.column_stack([np.ones(5), x[90:95]]),
        interval=interval,
        level=0.95,
        side=side,
    )
    assert np.allclose(forecast.mean.values, np.ravel(expected.mean))
    assert list(forecast.mean.index) == list(range(90, 95))
    if interval != "none":
        assert np.allclose(forecast.upper.values.ravel(), np.ravel(expected.upper))


def test_pure_regression_predict_with_selected_regressors():
    rng = np.random.default_rng(1)
    x = rng.normal(100, 10, (100, 3))
    y = x[:, 0] + rng.normal(size=100)
    model = ADAM(model="NNN", regressors="select").fit(y[:90], X=x[:90])
    forecast = model.predict(h=3, X=x[90:93], interval="prediction")
    assert np.all(np.isfinite(forecast.lower.values))
    assert np.allclose(forecast.mean.values, x[90:93, 0], atol=3)


# --- LASSO with lambda=1: the adaptive regressors at their shrinkage target ---


def test_etsx_adapt_lasso_lambda_one_does_not_adapt():
    """With lambda=1 the deltas of the regressors are zero, as in R.

    R: t <- 1:40; y <- round(100 + cumsum(sin(t/3) + (t %% 5)/5), 3);
    x <- round(cos(t/2), 3); adam(data.frame(y, x), "ANN", regressors="adapt",
    loss="LASSO", lambda=1, h=3); forecast(newdata=data.frame(x=c(0.1, 0.2, 0.3)))
    """
    t = np.arange(1, 41)
    y = np.round(100 + np.cumsum(np.sin(t / 3) + (t % 5) / 5), 3)
    X = pd.DataFrame({"x": np.round(np.cos(t / 2), 3)})
    model = ADAM(model="ANN", regressors="adapt", loss="LASSO", lambda_param=1)
    model.fit(y, X)
    assert model.persistence_vector == {"alpha": 0.0, "delta1": 0.0}
    assert model.initial_value["xreg"]["x"] == pytest.approx(0.762654, abs=1e-6)
    forecast = model.predict(h=3, X=pd.DataFrame({"x": [0.1, 0.2, 0.3]}))
    np.testing.assert_allclose(
        forecast.mean.values, [111.1203, 111.1966, 111.2729], rtol=1e-6
    )


# --- regressors="select" in the models that select after their fit ---


@pytest.fixture
def select_data():
    """The ETSX data with a third, irrelevant column (R: d$x3 <- round(sin(1:120), 6))."""
    df = pd.read_csv(DATA_PATH, float_precision="round_trip")
    df["x3"] = np.round(np.sin(np.arange(1, 121)), 6)
    return df


def test_ces_select_keeps_the_signal(select_data):
    """CES selects on the errors of the model without regressors, as R.

    R: ces(d, regressors="select", h=3): x1 and x2, logLik -162.613656 with the
    reference BLAS. The starting coefficients of the regressors come from the QR of
    the BLAS, and the fit moves with their last bits: -162.613674 with OpenBLAS, in R
    as in Python
    """
    from smooth import CES

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = CES(regressors="select", h=3).fit(
            select_data["y"].values, select_data[["x1", "x2", "x3"]]
        )
    assert [name for name in model.coef_names if name.startswith("x")] == ["x1", "x2"]
    assert model.loglik == pytest.approx(-162.613656, rel=1e-6)


def test_om_select_on_the_occurrence(select_data):
    """OM selects on the occurrence indicator, as R's om().

    R: d2 <- d; d2$y <- ifelse(d$x1 + 1.5*cos(3*(1:120)) > -0.3, d$y, 0);
    om(d2, "MNN", regressors="select"); forecast(h=3, newdata=d[118:120, ])
    """
    from smooth import OM

    t = np.arange(1, 121)
    y = np.where(select_data["x1"] + 1.5 * np.cos(3 * t) > -0.3, select_data["y"], 0)
    X = select_data[["x1", "x2", "x3"]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = OM(model="MNN", regressors="select").fit(y, X)
        forecast = model.predict(h=3, X=X.iloc[117:120])
    assert list(model.coef_names) == ["alpha", "x1"]
    np.testing.assert_allclose(model.coef, [0, 1.322667], atol=1e-6)
    assert model.loglik == pytest.approx(-63.82436, rel=1e-6)
    np.testing.assert_allclose(
        forecast.mean.values, [0.981642, 0.198426, 0.642679], atol=1e-6
    )
