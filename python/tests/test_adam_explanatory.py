"""Tests for ADAM with explanatory variables (ETSX models).

Data generated in R:
    set.seed(42)
    n <- 120
    x1 <- rnorm(n); x2 <- rnorm(n)
    y <- 10 + 2*x1 - 1.5*x2 + rnorm(n)
    write.csv(data.frame(y=y, x1=x1, x2=x2), "tests/data/etsx_data.csv", row.names=FALSE)

R reference (adam(df, model="AAN", regressors="use", formula=y~x1+x2)):
    smoother="global" (default since smooth v4.4.1 / Python v1.0.1)
    alpha=0.00890, beta=0, x1=1.915, x2=-1.461
    AIC=339.2135
    Fitted[1:5]=[15.24117, 11.48431, 10.92935, 13.08419, 11.19438]
    Forecast[1:3]=[10.91767, 7.99035, 9.92983] (using last 12 rows as X_new)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smooth import ADAM

DATA_PATH = Path(__file__).parent / "data" / "etsx_data.csv"

# R reference values (adam, smoother="global", no holdout)
R_AIC = 339.2135
R_FITTED_5 = [15.24117, 11.48431, 10.92935, 13.08419, 11.19438]
R_FORECAST_3 = [10.91767, 7.99035, 9.92983]
R_COEF_ALPHA = 0.00890
R_COEF_X1 = 1.915
R_COEF_X2 = -1.461


@pytest.fixture
def etsx_data():
    df = pd.read_csv(DATA_PATH)
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
    assert abs(model.aic - 339.2135) < 1.0
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
    assert abs(model_co.aic - 335.2486) < 1.5
    assert abs(model_co.fitted.values[0] - 15.265) < 0.5


def test_etsx_initial_optimal(etsx_data):
    """initial='optimal': xreg coefs in B (after ETS initials), match R reference."""
    y, X = etsx_data
    model = ADAM(model="AAN", regressors="use", initial="optimal")
    model.fit(y, X)
    B = model.coef
    # For "optimal", B = [alpha, beta, level_init, trend_init, x1, x2]
    # R (smoother="global"): x1=1.91839, x2=-1.47938, AIC=338.46, fitted[0]=14.956
    assert abs(B[-2] - 1.91839) < 0.15
    assert abs(B[-1] - (-1.47938)) < 0.15
    assert abs(model.aic - 338.4623) < 2.0
    assert abs(model.fitted.values[0] - 14.956) < 0.5


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


def test_initial_xreg_converges_to_same_result(etsx_data):
    """Different xreg seeds converge to the same loss value."""
    y, X = etsx_data
    m1 = ADAM(model="AAN", regressors="use", initial={"xreg": [0.0, 0.0]})
    m1.fit(y, X)
    m2 = ADAM(model="AAN", regressors="use", initial={"xreg": [2.0, -1.5]})
    m2.fit(y, X)
    assert abs(m1.loss_value - m2.loss_value) < 1.0


def test_initial_xreg_wrong_length_raises(etsx_data):
    """initial={'xreg': [...]} with wrong length raises ValueError."""
    y, X = etsx_data  # X has 2 columns
    model = ADAM(model="AAN", regressors="use", initial={"xreg": [1.0]})
    with pytest.raises(ValueError, match="xreg"):
        model.fit(y, X)


def test_initial_xreg_combined_with_level(etsx_data):
    """initial dict can mix 'xreg' with other keys."""
    y, X = etsx_data
    model = ADAM(model="AAN", regressors="use",
                 initial={"level": 10.0, "xreg": [2.0, -1.5]})
    model.fit(y, X)
    assert model._explanatory["xreg_model"] is True
