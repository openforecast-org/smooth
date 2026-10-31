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


# A regression with an occurrence model. R:
# set.seed(41); x <- round(rnorm(30, 10, 2), 2)
# set.seed(2); y <- round((5 + 2*x + rnorm(30)) * rbinom(30, 1, 0.7), 2)
# m <- adam(data.frame(y=y, x=x), "NNN", occurrence=occurrence)
# forecast(m, h=3, newdata=data.frame(x=c(9, 11.5, 13)), interval=..., level=...)
OCC_X = np.array(
    [
        8.41,
        10.39,
        12,
        12.58,
        11.81,
        10.99,
        11.2,
        6.84,
        12,
        14.38,
        7.58,
        8.83,
        12.11,
        9.37,
        9.89,
        10.66,
        11.33,
        11.76,
        10.41,
        14.55,
        8.2,
        14.28,
        7.67,
        9.91,
        10.83,
        13.44,
        8.43,
        7.39,
        9.1,
        6.45,
    ]
).reshape(-1, 1)
OCC_Y = np.array(
    [
        0,
        0,
        30.59,
        29.03,
        0,
        27.11,
        28.11,
        18.44,
        30.98,
        33.62,
        20.58,
        23.64,
        28.83,
        22.7,
        26.56,
        0,
        28.54,
        0,
        26.83,
        34.53,
        23.49,
        32.36,
        21.93,
        26.77,
        26.66,
        0,
        0,
        19.18,
        0,
        18.19,
    ]
)
OCC_X_NEW = np.array([[9.0], [11.5], [13.0]])
# R's mean, and the lower and upper bounds of levels 0.8 and 0.95 (column-major)
OCC_R = {
    "fixed": {
        "mean": [17.32126553, 20.8362183, 22.94518997],
        "both": (
            [22.368357, 27.16501939, 30.0148061, 21.47728738, 26.27647369, 29.10768949],
            [
                24.87145807,
                29.66103053,
                32.56298473,
                25.76252769,
                30.54957623,
                33.47010134,
            ],
        ),
        "upper": (
            None,
            [
                24.30201435,
                29.09319973,
                31.98328607,
                25.3432504,
                30.13148653,
                33.04327342,
            ],
        ),
        "lower": (
            [
                22.93780072,
                27.73285019,
                30.59450476,
                21.89656467,
                26.69456339,
                29.53451741,
            ],
            None,
        ),
        # R's confidence fails with two levels; these are its levels 0.8 and 0.95
        # one at a time
        "confidence": (
            [
                17.06222664,
                20.59021595,
                22.61443515,
                16.91291794,
                20.44842144,
                22.42378974,
            ],
            [
                17.58030441,
                21.08222066,
                23.27594479,
                17.72961312,
                21.22401517,
                23.4665902,
            ],
        ),
    },
    "odds-ratio": {
        "mean": [17.51802432, 20.78353104, 22.69219945],
        "both": (
            [
                22.36006678,
                27.16687571,
                30.02311424,
                21.47075815,
                26.2779342,
                29.11422196,
            ],
            [
                24.87974829,
                29.65917421,
                32.55467659,
                25.76905692,
                30.54811572,
                33.46356887,
            ],
        ),
        "upper": (
            None,
            [
                24.31254886,
                29.09083758,
                31.97270478,
                25.35042808,
                30.12988038,
                33.03608784,
            ],
        ),
        "lower": (
            [22.92726621, 27.73521234, 30.60508605, 21.889387, 26.69616955, 29.541703],
            None,
        ),
        "confidence": (
            [
                17.25604292,
                20.53815074,
                22.36509149,
                17.10503816,
                20.39671477,
                22.17654811,
            ],
            [
                17.78000572,
                21.02891134,
                23.01930741,
                17.93101048,
                21.17034731,
                23.20785079,
            ],
        ),
    },
}


@pytest.fixture(scope="module", params=["fixed", "odds-ratio"])
def regression_with_occurrence(request):
    model = ADAM(model="NNN", occurrence=request.param).fit(OCC_Y, X=OCC_X)
    return request.param, model


@pytest.mark.parametrize("case", ["both", "upper", "lower", "confidence"])
def test_pure_regression_with_occurrence_intervals_as_r(
    regression_with_occurrence, case
):
    """The mean of the sizes times the probability; the prediction interval of
    the sizes at the level (level-(1-p))/p, the confidence one times p, as R."""
    occurrence, model = regression_with_occurrence
    # R: coef(m)
    np.testing.assert_allclose(model.coef, [6.364684804, 1.91724697], rtol=1e-8)
    interval, side = (
        ("confidence", "both") if case == "confidence" else ("prediction", case)
    )
    forecast = model.predict(
        h=3, X=OCC_X_NEW, interval=interval, level=[0.8, 0.95], side=side
    )
    np.testing.assert_allclose(
        forecast.mean.values, OCC_R[occurrence]["mean"], rtol=1e-8
    )
    for bound, expected in zip(
        (forecast.lower, forecast.upper), OCC_R[occurrence][case]
    ):
        if expected is None:
            # R gives -Inf / Inf for the bound a one-sided interval leaves out
            assert bound is None
        else:
            np.testing.assert_allclose(
                bound.values.ravel(order="F"), expected, rtol=1e-8
            )


def test_pure_regression_predict_needs_x(regression_with_occurrence):
    _, model = regression_with_occurrence
    with pytest.raises(ValueError, match="X for the forecast horizon is needed"):
        model.predict(h=3)


def test_pure_regression_with_occurrence_confidence_of_one_level_as_r():
    model = ADAM(model="NNN", occurrence="fixed").fit(OCC_Y, X=OCC_X)
    forecast = model.predict(h=3, X=OCC_X_NEW, interval="confidence", level=0.9)
    # R: forecast(m, h=3, newdata=..., interval="confidence", level=0.9)
    np.testing.assert_allclose(
        forecast.lower.values.ravel(),
        [16.98391287, 20.51584345, 22.51443991],
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        forecast.upper.values.ravel(),
        [17.65861818, 21.15659316, 23.37594003],
        rtol=1e-8,
    )


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
    """The ETSX data with a third, irrelevant column
    (R: d$x3 <- round(sin(1:120), 6))."""
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
