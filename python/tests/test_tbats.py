"""Tests of TBATS, mirroring tests/testthat/test_tbats.R."""

import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

from smooth import TBATS
from smooth.adam_general.core.tbats import fitter as ft
from smooth.adam_general.core.tbats import structure as st

ORDERS0 = {"ar": 0, "ma": 0, "select": False}
ARMA11 = {"ar": 1, "ma": 1, "select": False}


@pytest.fixture(scope="module")
def air():
    path = pathlib.Path(__file__).parent / "data" / "ces_airpassengers.csv"
    return pd.read_csv(path)["y"].to_numpy(dtype=float)


def _rotation_form(
    eps, frequency, gammas, level, trend, alpha, beta, s, s_star, ar, ma, arma
):
    """De Livera's rotation form with ADAM's ARMA(1,1), from the states at t=0."""
    y = np.zeros(len(eps))
    for t, e in enumerate(eps):
        y[t] = level + trend + s.sum() + arma + e
        s_new = np.cos(frequency) * s + np.sin(frequency) * s_star + gammas[:, 0] * e
        s_star = -np.sin(frequency) * s + np.cos(frequency) * s_star + gammas[:, 1] * e
        s = s_new
        level = level + trend + alpha * e
        trend = trend + beta * e
        arma = ar * arma + (ar + ma) * e
    return y


def test_errors_are_those_of_the_rotation_form():
    rng = np.random.default_rng(11)
    tt = np.arange(1, 301)
    y = (
        200
        + 0.2 * tt
        + 5 * np.sin(2 * np.pi * tt / 7)
        + 3 * np.cos(2 * np.pi * tt / 30.4375)
        + np.cumsum(rng.normal(0, 0.5, 300))
        + rng.normal(size=300)
    )
    fit = TBATS(
        lags=[1, 7, 30.4375],
        harmonics=[2, 1],
        trend="additive",
        lambda_bc=1,
        orders=ARMA11,
        initial="optimal",
    ).fit(y)
    B = dict(zip(fit.coef_names, fit.coef))
    initial = fit.initial_value
    seasonal = initial["seasonal"]
    frequency = 2 * np.pi * seasonal["j"].to_numpy() / seasonal["period"].to_numpy()
    labels = [st._period_label(p) for p in seasonal["period"]]
    gammas = np.column_stack(
        [[B[f"gamma1[{x}]"] for x in labels], [B[f"gamma2[{x}]"] for x in labels]]
    )
    a, b = seasonal["sin"].to_numpy(), seasonal["cos"].to_numpy()
    s1 = a * np.sin(frequency) + b * np.cos(frequency)
    s_star1 = a * np.cos(frequency) - b * np.sin(frequency)
    rotation = _rotation_form(
        fit.residuals,
        frequency,
        gammas,
        initial["level"],
        initial["trend"],
        B["alpha"],
        B["beta"],
        s1,
        s_star1,
        B["phi1[1]"],
        B["theta1[1]"],
        initial["arma"][0],
    )
    # lambda=1 transforms y into y-1
    assert np.max(np.abs(rotation - (y - 1))) < 1e-8


def test_backcasting_reproduces_a_noise_free_fractional_period():
    tt = np.arange(1, 201)
    y = (
        100
        + 0.5 * tt
        + 10 * np.sin(2 * np.pi * tt / 7.3)
        + 4 * np.cos(2 * np.pi * tt / 7.3)
        + 2 * np.sin(4 * np.pi * tt / 7.3)
    )
    fit = TBATS(
        lags=[1, 7.3],
        harmonics=[2],
        trend="additive",
        lambda_bc=1,
        orders=ORDERS0,
        B=np.array([0.1, 0.01, 0.01, 0.01]),
        maxeval=1,
    ).fit(y)
    assert np.max(np.abs(fit.residuals)) < 1e-8


def test_lambda_zero_is_the_model_of_the_logarithms(air):
    fit_log = TBATS(
        lags=[1, 12], harmonics=[5], trend="additive", lambda_bc=0, orders=ORDERS0
    ).fit(air)
    fit_level = TBATS(
        lags=[1, 12],
        harmonics=[5],
        trend="additive",
        lambda_bc=1,
        orders=ORDERS0,
        B=fit_log.coef,
        maxeval=1,
    ).fit(np.log(air))
    assert fit_log.loglik == pytest.approx(
        fit_level.loglik - np.log(air).sum(), rel=1e-10
    )


def test_lambda_is_estimated_in_the_unit_interval_or_falls_back(air):
    fit = TBATS(lags=[1, 12], harmonics=[5], trend="additive", orders=ORDERS0).fit(air)
    assert 0 <= fit.lambda_ <= 1
    assert "lambda" in fit.coef_names
    with pytest.warns(UserWarning, match="positive data"):
        negative = TBATS(
            lags=[1, 12], harmonics=[5], trend="additive", orders=ORDERS0
        ).fit(air - 200)
    assert negative.lambda_ == 1
    with pytest.warns(UserWarning, match="likelihood"):
        mse = TBATS(
            lags=[1, 12], harmonics=[5], trend="additive", orders=ORDERS0, loss="MSE"
        ).fit(air)
    assert mse.lambda_ == 1


@pytest.mark.parametrize("distribution", ["dnorm", "dlaplace", "ds", "dgnorm"])
def test_the_distributions_are_fitted(air, distribution):
    fit = TBATS(
        lags=[1, 12],
        harmonics=[5],
        trend="additive",
        orders=ORDERS0,
        distribution=distribution,
    ).fit(air)
    assert np.isfinite(fit.loglik)
    assert ("shape" in fit.coef_names) == (distribution == "dgnorm")


def test_a_provided_shape_is_not_estimated(air):
    fit = TBATS(
        lags=[1, 12],
        harmonics=[5],
        trend="additive",
        orders=ORDERS0,
        distribution="dgnorm",
        shape=1.5,
    ).fit(air)
    assert "shape" not in fit.coef_names


def test_usual_bounds_keep_the_response_in_the_unit_interval(air):
    fit = TBATS(
        lags=[1, 12], harmonics=[5], trend="additive", orders=ORDERS0, bounds="usual"
    ).fit(air)
    B = dict(zip(fit.coef_names, fit.coef))
    seasonal = fit.initial_value["seasonal"]
    frequency = 2 * np.pi * seasonal["j"].to_numpy() / seasonal["period"].to_numpy()
    horizons = np.arange(12)
    response = (
        B["alpha"]
        + np.cos(np.outer(horizons, frequency)) @ np.full(5, B["gamma1[12]"])
        + np.sin(np.outer(horizons, frequency)) @ np.full(5, B["gamma2[12]"])
    )
    assert np.all((response >= 0) & (response <= 1))
    assert B["beta"] <= B["alpha"]


def test_coinciding_harmonics_are_dropped_and_arma_lags_merged():
    table = st.harmonics_table([24, 168], [2, 8])
    assert len(table["frequency"]) == 2 + 7
    assert not np.any((table["period"] == 168) & (table["j"] == 7))
    spec = st.arma_spec({"ar": [1, 1, 2], "ma": 0}, [1, 7, 7.02])
    assert spec["lags"].tolist() == [1, 7]
    assert spec["ar_orders"].tolist() == [1, 2]


@pytest.mark.parametrize("initial", ["backcasting", "optimal", "two-stage"])
def test_admissible_bounds_keep_the_discount_matrix_stable(air, initial):
    fit = TBATS(
        lags=[1, 12], harmonics=[5], trend="damped", orders=ARMA11, initial=initial
    ).fit(air)
    best = fit._best
    values = ft.eigens(
        best["elements"]["mat_f"],
        best["elements"]["vec_g"],
        best["elements"]["w"],
        best["struct"],
    )
    assert values.max() <= 1 + 1e-10


def test_two_stage_cannot_end_below_the_backcasted_fit(air):
    complete = TBATS(
        lags=[1, 12], trend="damped", orders=ARMA11, initial="complete"
    ).fit(air)
    two_stage = TBATS(
        lags=[1, 12], trend="damped", orders=ARMA11, initial="two-stage"
    ).fit(air)
    assert two_stage.loglik >= complete.loglik - 1e-8


def test_the_arma_falls_back_to_none_when_it_does_not_help(air):
    fit = TBATS(lags=[1, 12], harmonics=[5], trend="additive").fit(air)
    assert sum(fit.orders_["ar"]) + sum(fit.orders_["ma"]) == 0
    assert any("+ARMA" in name for name in fit.ics)
    assert min(fit.ics.values()) == pytest.approx(fit.aicc)


def test_the_forecasts_are_the_transformed_forecasts_of_adam(air):
    fit = TBATS(
        lags=[1, 12],
        harmonics=[5],
        trend="damped",
        orders={"ar": 1, "ma": 0},
        h=12,
        holdout=True,
    ).fit(air)
    forecast = fit.predict(h=12, interval="prediction")
    np.testing.assert_allclose(np.asarray(forecast.mean), fit.forecast_, rtol=1e-10)
    lower = np.asarray(forecast.lower).ravel()
    upper = np.asarray(forecast.upper).ravel()
    mean = np.asarray(forecast.mean)
    assert np.all((lower < mean) & (mean < upper))
    # The cumulative forecasts come from the paths in the space of the data
    np.testing.assert_allclose(
        fit.predict(h=12, cumulative=True).mean, fit.forecast_.sum(), rtol=1e-10
    )
    cumulative = fit.predict(
        h=12, cumulative=True, interval="prediction", point="mean", seed=41
    )
    mean = float(cumulative.mean.iloc[0])
    assert cumulative.lower.iloc[0, 0] < mean < cumulative.upper.iloc[0, 0]
    assert mean == pytest.approx(fit.predict(h=12, point="mean").mean.sum(), rel=1e-2)


def test_point_likelihoods_sum_to_the_log_likelihood(air):
    fit = TBATS(lags=[1, 12], harmonics=[5], trend="additive", orders=ORDERS0).fit(air)
    assert fit.point_lik().sum() == pytest.approx(fit.loglik, rel=1e-12)


def test_the_covariance_is_finite(air):
    fit = TBATS(lags=[1, 12], harmonics=[5], trend="additive", orders=ORDERS0).fit(air)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        covariance = fit.vcov()
    assert np.all(np.isfinite(covariance.to_numpy()))
    assert list(covariance.index) == fit.coef_names


def test_bad_arguments_raise():
    with pytest.raises(ValueError, match="trend"):
        TBATS(trend="multiplicative")
    with pytest.raises(ValueError, match="lambda"):
        TBATS(lambda_bc=2).fit(np.arange(1.0, 30.0))


@pytest.fixture(scope="module")
def damped_arma(air):
    return TBATS(lags=[1, 12], harmonics=[5], trend="damped", orders=ARMA11).fit(air)


def test_reapply_reproduces_the_fit_with_a_tiny_covariance(air):
    fit = TBATS(
        lags=[1, 12], harmonics=[4], trend="additive", lambda_bc=1, orders=ORDERS0
    ).fit(air)
    refitted = fit.reapply(nsim=5, heuristics=1e-12, seed=41)
    assert np.max(np.abs(refitted.refitted.to_numpy() - fit.fitted[:, None])) < 1e-3
    assert refitted.states.shape[2] == 5
    assert np.all(refitted.lambdas == 1)


def test_the_intervals_with_the_uncertainty_of_the_parameters(damped_arma):
    point = np.asarray(damped_arma.predict(h=12).mean)
    for interval in ("confidence", "complete"):
        forecast = damped_arma.predict(h=12, interval=interval, nsim=20, seed=41)
        np.testing.assert_allclose(np.asarray(forecast.mean), point)
        lower = np.ravel(forecast.lower)
        upper = np.ravel(forecast.upper)
        assert np.all(lower <= upper)
    assert np.all(lower < point) and np.all(point < upper)


@pytest.mark.parametrize("initial", ["backcasting", "optimal"])
def test_the_simulations_start_from_the_initials_of_the_model(air, initial):
    fit = TBATS(
        lags=[1, 12],
        harmonics=[3],
        trend="damped",
        orders={"ar": 1, "ma": 0},
        initial=initial,
    ).fit(air)
    simulated = fit.simulate(nsim=2, seed=41)
    assert simulated.data.shape == (len(air), 2)
    first = (
        st.box_cox(simulated.data.iloc[0].to_numpy(), fit.lambda_)
        - simulated.residuals.iloc[0]
    )
    np.testing.assert_allclose(
        first, st.box_cox(fit.fitted[0], fit.lambda_), rtol=1e-10
    )


def test_confint_stays_inside_the_bounds_of_the_model(damped_arma):
    intervals = damped_arma.confint()
    assert intervals.loc["lambda"].iloc[1] >= 0 and intervals.loc["lambda"].iloc[2] <= 1
    coef = damped_arma.coef
    assert np.all((intervals.iloc[:, 1] <= coef) & (coef <= intervals.iloc[:, 2]))


def test_the_bootstrap_refits_the_model(damped_arma):
    bootstrap = damped_arma.coefbootstrap(nsim=5, seed=41)
    assert bootstrap.coefficients.shape == (
        bootstrap.nsim_effective,
        len(damped_arma.coef),
    )
    assert bootstrap.nsim_effective > 0
    covariance = damped_arma.vcov(type="bootstrap", nsim=5, seed=41)
    assert list(covariance.index) == damped_arma.coef_names


# A series with two regressors, of which the second is noise, and their future values
@pytest.fixture(scope="module")
def xreg_data():
    rng = np.random.default_rng(41)
    times = np.arange(1, 133)
    X = np.column_stack([rng.normal(10, 2, 132), rng.normal(size=132)])
    noise = np.cumsum(rng.normal(size=132)) + rng.normal(size=132)
    y = 200 + 20 * np.sin(2 * np.pi * times / 12) + 5 * X[:, 0] + noise
    return y, pd.DataFrame(X, columns=["x1", "x2"])


@pytest.fixture(scope="module")
def xreg_fit(xreg_data):
    y, X = xreg_data
    return TBATS(
        lags=[1, 12],
        harmonics=[1],
        trend="none",
        orders=ORDERS0,
        lambda_bc=1,
        h=12,
        holdout=True,
    ).fit(y, X)


def test_lambda_zero_with_a_regressor_is_the_model_of_the_logarithms(xreg_data):
    y, X = xreg_data
    arguments = dict(lags=[1, 12], harmonics=[1], trend="none", orders=ORDERS0)
    fit_log = TBATS(lambda_bc=0, **arguments).fit(y[:120], X[:120])
    fit_level = TBATS(lambda_bc=1, B=fit_log.coef, maxeval=1, **arguments).fit(
        np.log(y[:120]), X[:120]
    )
    assert fit_log.loglik == pytest.approx(
        fit_level.loglik - np.sum(np.log(y[:120])), rel=1e-8
    )


def test_the_regressors_are_estimated_and_used_in_the_forecasts(xreg_fit, xreg_data):
    _, X = xreg_data
    assert xreg_fit.model_name.startswith("TBATSX")
    assert list(xreg_fit.initial_value["xreg"]) == ["x1", "x2"]
    assert xreg_fit.initial_value["xreg"]["x1"] == pytest.approx(5, abs=0.5)
    forecast = xreg_fit.forecast_
    np.testing.assert_allclose(np.asarray(xreg_fit.predict(h=12).mean), forecast)
    future = X.to_numpy()[120:]
    np.testing.assert_allclose(
        np.asarray(xreg_fit.predict(h=12, X=future).mean), forecast, rtol=1e-12
    )
    with pytest.warns(UserWarning, match="X is not provided"):
        xreg_fit.predict(h=24)
    assert np.sum(xreg_fit.point_lik()) == pytest.approx(xreg_fit.loglik, rel=1e-8)
    assert np.all(np.isfinite(xreg_fit.vcov().to_numpy()))
    bootstrap = xreg_fit.coefbootstrap(nsim=3, seed=41)
    assert bootstrap.coefficients.shape[1] == len(xreg_fit.coef)
    assert xreg_fit.simulate(nsim=2, seed=41).data.shape == (120, 2)


def test_the_intervals_follow_the_future_regressors(xreg_fit, xreg_data):
    # The simulated paths take the new values of the regressors, as the point
    # forecasts do: the bounds move with them
    _, X = xreg_data
    future = X.to_numpy()[120:] + np.array([10.0, 0.0])
    for interval in ("approximate", "simulated", "complete"):
        forecast = xreg_fit.predict(h=12, X=future, interval=interval, nsim=200)
        mean = np.asarray(forecast.mean)
        assert np.all(mean > xreg_fit.forecast_ + 40)
        lower, upper = np.ravel(forecast.lower), np.ravel(forecast.upper)
        assert np.all((lower < mean) & (mean < upper))


def test_all_backcast_regressors_are_counted_in_the_parameters(xreg_fit, xreg_data):
    y, X = xreg_data
    fit = TBATS(
        lags=[1, 12],
        harmonics=[1],
        trend="none",
        orders=ORDERS0,
        lambda_bc=1,
        initial="complete",
        h=12,
        holdout=True,
    ).fit(y, X)
    assert not {"x1", "x2"} & set(fit.coef_names)
    assert fit.nparam == xreg_fit.nparam


@pytest.mark.parametrize("bounds", ["usual", "admissible"])
def test_adaptive_regressors_stay_within_the_bounds(bounds):
    rng = np.random.default_rng(41)
    X = np.column_stack([rng.normal(10, 2, 150), rng.normal(size=150)])
    beta = 5 + np.cumsum(rng.normal(0, 0.05, 150))
    noise = np.cumsum(rng.normal(size=150)) + rng.normal(size=150)
    y = 200 + beta * X[:, 0] - 2 * X[:, 1] + noise
    fit = TBATS(regressors="adapt", trend="none", orders=ORDERS0, bounds=bounds).fit(
        y, X
    )
    assert fit.model_name.endswith("{D}")
    deltas = fit.coef[[fit.coef_names.index(n) for n in ("delta1", "delta2")]]
    assert np.all((deltas >= 0) & (deltas <= 1))
    # The averaged condition of ADAM rejects a coefficient that explodes
    exploding = fit.coef.copy()
    exploding[fit.coef_names.index("delta1")] = 3
    assert fit._best["fitter"](exploding) is None


def test_the_selection_keeps_the_relevant_regressor(xreg_data):
    y, X = xreg_data
    X = X.assign(noise=np.random.default_rng(7).normal(size=len(y)))
    fit = TBATS(lags=[1, 12], regressors="select", h=12, holdout=True).fit(y, X)
    assert fit.xreg_names_ == ["x1"]
    assert any("+X(x1)" in name for name in fit.ics)
    np.testing.assert_allclose(np.asarray(fit.predict(h=12).mean), fit.forecast_)


def test_the_point_forecasts(air):
    fit = TBATS(lags=[1, 12], harmonics=[3], trend="additive", orders=ORDERS0).fit(air)
    skeleton = fit.predict(h=24).mean.to_numpy()
    np.testing.assert_allclose(fit.predict(h=24, point="median").mean, skeleton)
    mean = fit.predict(h=24, point="mean").mean.to_numpy()
    assert np.all(mean > skeleton)
    # lambda=0 and the S distribution: the mean does not exist
    fit = TBATS(
        lags=[1, 12],
        harmonics=[3],
        trend="additive",
        orders=ORDERS0,
        lambda_bc=0,
        distribution="ds",
    ).fit(air)
    with pytest.warns(UserWarning, match="does not exist"):
        fit.predict(h=3, point="mean", seed=41)
    with pytest.raises(ValueError, match="point"):
        fit.predict(h=3, point="mode")


@pytest.fixture(scope="module")
def intermittent():
    """Log-normal sizes with a weekly pattern, on 45% of the days."""
    rng = np.random.default_rng(41)
    t = np.arange(1, 366)
    sizes = np.exp(2 + 0.4 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 0.3, 365))
    return sizes * rng.binomial(1, 0.45, 365)


def test_the_occurrence_mixture(intermittent):
    # The zeros are fitted as values, and the Box-Cox transform falls back to 1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        TBATS(lags=[1, 7], orders=ORDERS0).fit(intermittent)
    messages = " ".join(str(w.message) for w in caught)
    assert "occurrence" in messages and "lambda=1" in messages
    fit = TBATS(
        lags=[1, 7], occurrence="odds-ratio", orders=ORDERS0, h=14, holdout=True
    ).fit(intermittent)
    occurrence = fit._occurrence["model"]
    assert fit.nparam == fit._best["n_param_estimated"] + occurrence.nparam
    np.testing.assert_allclose(fit.point_lik().sum(), fit.loglik, rtol=1e-10)
    # The sizes are log-normal
    assert abs(fit.lambda_) < 0.05
    p_forecast = np.asarray(occurrence.predict(h=14).mean)
    forecasted = fit.predict(h=14, interval="prediction")
    sizes = st.box_cox_inverse(fit._best["forecast_bc"], fit.lambda_)
    np.testing.assert_allclose(forecasted.mean, sizes * p_forecast)
    np.testing.assert_allclose(forecasted.mean, fit.forecast_)
    # The probability of no demand is above 0.5: the median and the lower bound are 0
    assert np.all(fit.predict(h=14, point="median").mean == 0)
    assert np.all(forecasted.lower.to_numpy() == 0)
    assert np.all(forecasted.upper.to_numpy().ravel() > forecasted.mean.to_numpy())
    assert np.mean(np.asarray(fit.simulate(nsim=2, seed=41).data) == 0) > 0.3
    assert np.mean(fit.reforecast(h=14, nsim=20, seed=41).paths == 0) > 0.3
    reapplied = fit.reapply(nsim=5, seed=41)
    assert np.all(reapplied.errors[fit.actuals == 0] == 0)


def test_the_cumulative_forecasts_of_the_mixture(intermittent):
    # The sum of the skeletons times the probabilities, and the quantiles of the sums
    # of the paths
    fit = TBATS(lags=[1, 7], occurrence="odds-ratio", orders=ORDERS0).fit(intermittent)
    forecasted = fit.predict(h=14)
    cumulative = fit.predict(h=14, cumulative=True, interval="prediction", seed=41)
    mean = float(cumulative.mean.iloc[0])
    assert mean == pytest.approx(forecasted.mean.sum(), rel=1e-10)
    assert cumulative.lower.iloc[0, 0] < mean < cumulative.upper.iloc[0, 0]
    reforecasted = fit.reforecast(
        h=14, cumulative=True, interval="prediction", nsim=20, seed=41
    )
    assert reforecasted.lower.iloc[0, 0] < reforecasted.upper.iloc[0, 0]


def test_the_observations_with_missing_regressors_are_dropped(xreg_data):
    y, X = xreg_data
    arguments = dict(lags=[1, 12], harmonics=[1], trend="none", orders=ORDERS0)
    X_na = X.copy()
    X_na.iloc[[14, 59], 0] = np.nan
    with pytest.warns(UserWarning, match="missing values"):
        fit = TBATS(**arguments).fit(y, X_na)
    y_na = y.copy()
    y_na[[14, 59]] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_na = TBATS(**arguments).fit(y_na, X)
    assert fit.loglik == pytest.approx(fit_na.loglik, rel=1e-12)
    np.testing.assert_allclose(fit.coef, fit_na.coef, rtol=1e-10)
    assert np.all(np.isnan(fit.fitted[[14, 59]]))
    with pytest.raises(ValueError, match="horizon"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_future_na = X.to_numpy().copy()
        X_future_na[-2:] = np.nan
        TBATS(h=12, **arguments).fit(y[:120], X_future_na)


def test_the_occurrence_errors(intermittent):
    with pytest.raises(ValueError, match="contradicts"):
        TBATS(lags=[1, 7], occurrence=np.where(intermittent > 0, 0.0, 0.5)).fit(
            intermittent
        )
    with pytest.raises(ValueError, match="multistep"):
        TBATS(lags=[1, 7], occurrence="odds-ratio", loss="MSEh", h=3).fit(intermittent)
    with pytest.raises(ValueError, match="occurrence"):
        TBATS(occurrence="sometimes")


@pytest.mark.filterwarnings("ignore:Data contains NAs")
def test_tbats_takes_the_missing_values_for_gaps():
    rng = np.random.default_rng(1)
    t = np.arange(144)
    y = np.exp(
        5 + 0.01 * t + 0.2 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.05, 144)
    )
    y[[9, 49, 50, 89]] = np.nan
    model = TBATS(lags=[1, 12]).fit(y)
    assert np.sum(model.point_lik()) == pytest.approx(model.loglik, abs=1e-8)
    assert np.all(np.isnan(model.residuals[np.isnan(y)]))
    assert np.all(np.isfinite(model.fitted))
    assert np.all(np.isfinite(model.predict(h=12, interval="prediction").upper))


def test_initial_gradient_solves_for_the_initials(air):
    """The initials are counted as with backcasting, and at the same parameters the
    solved initials fit at least as well as the backcast ones."""
    arguments = dict(lags=[1, 12], harmonics=[5], trend="additive", orders=ORDERS0)
    backcast = TBATS(**arguments).fit(air)
    gradient = TBATS(initial="gradient", **arguments).fit(air)
    assert gradient.nparam == backcast.nparam
    assert gradient.point_lik().sum() == pytest.approx(gradient.loglik, rel=1e-12)
    at_backcast = TBATS(
        initial="gradient",
        B=backcast.coef,
        maxeval=1,
        **arguments,
    ).fit(air)
    assert at_backcast.loss_value <= backcast.loss_value
