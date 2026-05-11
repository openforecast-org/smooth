"""
Unit tests for ADAM with AirPassengers dataset.

These tests use the classic AirPassengers dataset (monthly airline passengers 1949-1960)
to verify model fitting, persistence estimation, and forecasting.

The reference loss values were recorded from the implementation to ensure consistency.
"""

import numpy as np
import pytest

from smooth import ADAM


# Full AirPassengers dataset (1949-1960, 144 monthly observations)
AIRPASSENGERS = np.array([
    112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
    115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
    145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
    171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
    196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
    204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
    242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
    284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
    315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
    340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
    360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
    417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432
], dtype=float)


class TestADAMAirPassengersBasic:
    """Basic tests with AirPassengers dataset."""

    def test_dataset_properties(self):
        """Verify AirPassengers dataset properties."""
        assert len(AIRPASSENGERS) == 144
        assert AIRPASSENGERS.min() == 104
        assert AIRPASSENGERS.max() == 622
        assert np.isclose(AIRPASSENGERS.mean(), 280.298611, rtol=1e-4)

    def test_ann_model_fit(self):
        """Test ANN (simple exponential smoothing) on AirPassengers."""
        model = ADAM(model="ANN", lags=[1])
        model.fit(AIRPASSENGERS)

        # Verify model was estimated
        assert model.coef is not None
        assert model.loss_value is not None
        assert model.coef is not None

        # Reference loss value (recorded from implementation)
        expected_loss = 710.389560
        actual_loss = model.loss_value
        assert np.isclose(actual_loss, expected_loss, rtol=1e-4), \
            f"ANN loss {actual_loss} differs from expected {expected_loss}"

    def test_aan_model_fit(self):
        """Test AAN (Holt's linear trend) on AirPassengers."""
        model = ADAM(model="AAN", lags=[1])
        model.fit(AIRPASSENGERS)

        # Reference loss value
        expected_loss = 755.065923
        actual_loss = model.loss_value
        assert np.isclose(actual_loss, expected_loss, rtol=1e-4), \
            f"AAN loss {actual_loss} differs from expected {expected_loss}"

    def test_ana_model_fit(self):
        """Test ANA (seasonal, no trend) on AirPassengers."""
        model = ADAM(model="ANA", lags=[12])
        model.fit(AIRPASSENGERS)

        # Reference loss value
        expected_loss = 586.036821
        actual_loss = model.loss_value
        assert np.isclose(actual_loss, expected_loss, rtol=1e-4), \
            f"ANA loss {actual_loss} differs from expected {expected_loss}"


class TestADAMAirPassengersPersistence:
    """Tests for persistence parameter estimation on AirPassengers."""

    def test_ann_persistence_bounds(self):
        """Test that ANN persistence (alpha) is within valid bounds."""
        model = ADAM(model="ANN", lags=[1])
        model.fit(AIRPASSENGERS)

        B = model.coef
        alpha = B[0]

        # Alpha should be between 0 and 1
        assert 0 <= alpha <= 1, f"Alpha {alpha} out of bounds [0, 1]"

    def test_aan_persistence_bounds(self):
        """Test that AAN persistence (alpha, beta) is within valid bounds."""
        model = ADAM(model="AAN", lags=[1])
        model.fit(AIRPASSENGERS)

        B = model.coef
        alpha = B[0]
        beta = B[1]

        # Alpha and beta should be between 0 and 1
        assert 0 <= alpha <= 1, f"Alpha {alpha} out of bounds"
        assert 0 <= beta <= 1, f"Beta {beta} out of bounds"
        # Typically beta <= alpha for stability
        assert beta <= alpha + 0.01, f"Beta {beta} should be <= alpha {alpha}"

    def test_ana_persistence_bounds(self):
        """Test that ANA persistence (alpha, gamma) is within valid bounds."""
        model = ADAM(model="ANA", lags=[12])
        model.fit(AIRPASSENGERS)

        B = model.coef
        alpha = B[0]
        gamma = B[1]

        # Alpha and gamma should be between 0 and 1
        assert 0 <= alpha <= 1, f"Alpha {alpha} out of bounds"
        assert 0 <= gamma <= 1, f"Gamma {gamma} out of bounds"

    def test_ana_persistence_values(self):
        """Test that ANA persistence values match expected estimates."""
        model = ADAM(model="ANA", lags=[12])
        model.fit(AIRPASSENGERS)

        B = model.coef
        # Reference values from implementation
        expected_alpha = 0.34142799
        expected_gamma = 0.65856849

        assert np.isclose(B[0], expected_alpha, rtol=1e-3), \
            f"Alpha {B[0]} differs from expected {expected_alpha}"
        assert np.isclose(B[1], expected_gamma, rtol=1e-3), \
            f"Gamma {B[1]} differs from expected {expected_gamma}"


class TestADAMAirPassengersPartialPersistence:
    """Tests for partially provided persistence parameters."""

    def test_ana_partial_persistence_level(self):
        """Test ANA with only level persistence (alpha) provided."""
        model = ADAM(model="ANA", lags=[12], persistence={"level": 0.4})
        model.fit(AIRPASSENGERS)

        # Model should fit with provided alpha
        assert model.coef is not None
        assert model.persistence_level_ == 0.4

        # Loss should be higher than optimal since alpha is fixed
        # Reference: optimal loss is ~586, with alpha=0.4 it's ~649
        expected_loss = 649.145309
        actual_loss = model.loss_value
        assert np.isclose(actual_loss, expected_loss, rtol=1e-3), \
            f"Loss with alpha=0.4: {actual_loss} differs from expected {expected_loss}"

    def test_provided_persistence_affects_loss(self):
        """Test that providing persistence changes the loss."""
        # Fit without provided persistence
        model_free = ADAM(model="ANA", lags=[12])
        model_free.fit(AIRPASSENGERS)
        loss_free = model_free.loss_value

        # Fit with provided persistence (suboptimal)
        model_fixed = ADAM(model="ANA", lags=[12], persistence={"level": 0.4})
        model_fixed.fit(AIRPASSENGERS)
        loss_fixed = model_fixed.loss_value

        # Loss with fixed (suboptimal) persistence should be higher
        assert loss_fixed > loss_free, \
            f"Fixed persistence loss {loss_fixed} should be > free loss {loss_free}"


class TestADAMAirPassengersForecasting:
    """Tests for forecasting on AirPassengers."""

    def test_ana_forecast_shape(self):
        """Test that ANA forecast has correct shape."""
        model = ADAM(model="ANA", lags=[12])
        model.fit(AIRPASSENGERS)

        forecast = model.predict(h=12)

        assert forecast.shape[0] == 12
        assert "mean" in forecast.columns

    def test_ana_forecast_values(self):
        """Test ANA forecast values for first 3 periods."""
        model = ADAM(model="ANA", lags=[12])
        model.fit(AIRPASSENGERS)

        forecast = model.predict(h=12)
        forecast_mean = forecast["mean"].values

        # Reference forecast values (first 3 periods)
        expected_forecasts = [444.20935371, 419.61956321, 458.90904404]

        for i, (actual, expected) in enumerate(zip(forecast_mean[:3], expected_forecasts)):
            assert np.isclose(actual, expected, rtol=1e-4), \
                f"Forecast[{i}] = {actual} differs from expected {expected}"

    def test_forecast_positive(self):
        """Test that forecasts are positive (since data is positive)."""
        model = ADAM(model="ANA", lags=[12])
        model.fit(AIRPASSENGERS)

        forecast = model.predict(h=24)

        assert (forecast["mean"] > 0).all(), "All forecasts should be positive"

    def test_forecast_includes_intervals(self):
        """Test that forecast includes prediction intervals."""
        model = ADAM(model="ANA", lags=[12])
        model.fit(AIRPASSENGERS)

        forecast = model.predict(h=12, interval="prediction")
        columns = forecast.columns.tolist()

        # Should have lower and upper bounds
        has_lower = any("lower" in c for c in columns)
        has_upper = any("upper" in c for c in columns)

        assert has_lower, "Forecast should include lower bounds"
        assert has_upper, "Forecast should include upper bounds"

    def test_forecast_intervals_ordered(self):
        """Test that prediction intervals are properly ordered."""
        model = ADAM(model="ANA", lags=[12])
        model.fit(AIRPASSENGERS)

        forecast = model.predict(h=12, interval="prediction")

        # Find lower and upper columns
        lower_col = [c for c in forecast.columns if "lower" in c][0]
        upper_col = [c for c in forecast.columns if "upper" in c][0]

        # Lower < Mean < Upper
        assert (forecast[lower_col] <= forecast["mean"]).all(), \
            "Lower bound should be <= mean"
        assert (forecast["mean"] <= forecast[upper_col]).all(), \
            "Mean should be <= upper bound"


class TestADAMAirPassengersInitialTypes:
    """Tests for different initial value estimation methods."""

    def test_ana_backcasting_initial(self):
        """Test ANA with backcasting initial (default)."""
        model = ADAM(model="ANA", lags=[12], initial="backcasting")
        model.fit(AIRPASSENGERS)

        expected_loss = 586.036821
        actual_loss = model.loss_value
        assert np.isclose(actual_loss, expected_loss, rtol=1e-4), \
            f"ANA backcasting loss {actual_loss} differs from expected {expected_loss}"

    def test_ana_optimal_initial(self):
        """Test ANA with optimal initial values."""
        model = ADAM(model="ANA", lags=[12], initial="optimal")
        model.fit(AIRPASSENGERS)

        expected_loss = 588.474062
        actual_loss = model.loss_value
        assert np.isclose(actual_loss, expected_loss, rtol=1e-4), \
            f"ANA optimal loss {actual_loss} differs from expected {expected_loss}"

    def test_ana_two_stage_initial(self):
        """Test ANA with two-stage initial values."""
        model = ADAM(model="ANA", lags=[12], initial="two-stage")
        model.fit(AIRPASSENGERS)

        expected_loss = 586.073832
        actual_loss = model.loss_value
        assert np.isclose(actual_loss, expected_loss, rtol=1e-4), \
            f"ANA two-stage loss {actual_loss} differs from expected {expected_loss}"

    def test_aaa_backcasting_initial(self):
        """Test AAA with backcasting initial (default)."""
        model = ADAM(model="AAA", lags=[12], initial="backcasting")
        model.fit(AIRPASSENGERS)

        expected_loss = 565.313766
        actual_loss = model.loss_value
        assert np.isclose(actual_loss, expected_loss, rtol=1e-4), \
            f"AAA backcasting loss {actual_loss} differs from expected {expected_loss}"

    def test_aaa_optimal_initial(self):
        """Test AAA with optimal initial values."""
        model = ADAM(model="AAA", lags=[12], initial="optimal")
        model.fit(AIRPASSENGERS)

        expected_loss = 569.312991
        actual_loss = model.loss_value
        assert np.isclose(actual_loss, expected_loss, rtol=1e-4), \
            f"AAA optimal loss {actual_loss} differs from expected {expected_loss}"

    def test_aaa_two_stage_initial(self):
        """Test AAA with two-stage initial values."""
        model = ADAM(model="AAA", lags=[12], initial="two-stage")
        model.fit(AIRPASSENGERS)

        expected_loss = 565.123455
        actual_loss = model.loss_value
        assert np.isclose(actual_loss, expected_loss, rtol=1e-4), \
            f"AAA two-stage loss {actual_loss} differs from expected {expected_loss}"

    def test_different_initials_different_losses(self):
        """Test that different initial methods produce different losses."""
        losses = {}
        for init_type in ["backcasting", "optimal", "two-stage"]:
            model = ADAM(model="ANA", lags=[12], initial=init_type)
            model.fit(AIRPASSENGERS)
            losses[init_type] = model.loss_value

        # All losses should be different
        loss_values = list(losses.values())
        assert len(set(round(v, 4) for v in loss_values)) == len(loss_values), \
            f"Expected different losses for different initial types, got: {losses}"

    def test_optimal_estimates_more_parameters(self):
        """Test that optimal initial estimates more parameters than backcasting."""
        model_back = ADAM(model="AAA", lags=[12], initial="backcasting")
        model_back.fit(AIRPASSENGERS)

        model_opt = ADAM(model="AAA", lags=[12], initial="optimal")
        model_opt.fit(AIRPASSENGERS)

        # Optimal should have more parameters (includes initial states)
        b_back = len(model_back.coef)
        b_opt = len(model_opt.coef)

        assert b_opt > b_back, \
            f"Optimal B length {b_opt} should be > backcasting B length {b_back}"


class TestADAMAirPassengersAAAPersistence:
    """Tests for ETS(A,A,A) model with various persistence configurations."""

    def test_aaa_no_persistence(self):
        """Test AAA with no persistence provided (baseline)."""
        model = ADAM(model="AAA", lags=[12])
        model.fit(AIRPASSENGERS)

        expected_loss = 565.313766
        actual_loss = model.loss_value
        assert np.isclose(actual_loss, expected_loss, rtol=1e-4), \
            f"AAA loss {actual_loss} differs from expected {expected_loss}"

        # Check persistence values are estimated
        B = model.coef
        assert len(B) >= 3, "AAA should estimate at least 3 persistence parameters"

    def test_aaa_alpha_only(self):
        """Test AAA with only alpha=0.5 provided."""
        model = ADAM(model="AAA", lags=[12], persistence={"level": 0.5})
        model.fit(AIRPASSENGERS)

        # Verify alpha is fixed
        assert model.persistence_level_ == 0.5

        # Reference loss with alpha=0.5
        expected_loss = 636.965563
        actual_loss = model.loss_value
        assert np.isclose(actual_loss, expected_loss, rtol=1e-3), \
            f"AAA alpha=0.5 loss {actual_loss} differs from expected {expected_loss}"

    def test_aaa_alpha_beta(self):
        """Test AAA with alpha=0.5 and beta=0.1 provided."""
        model = ADAM(model="AAA", lags=[12], persistence={"level": 0.5, "trend": 0.1})
        model.fit(AIRPASSENGERS)

        # Verify persistence values are fixed
        assert model.persistence_level_ == 0.5
        assert model.persistence_trend_ == 0.1

        # Reference loss
        expected_loss = 654.328174
        actual_loss = model.loss_value
        assert np.isclose(actual_loss, expected_loss, rtol=1e-3), \
            f"AAA alpha=0.5, beta=0.1 loss {actual_loss} differs from expected {expected_loss}"

    def test_aaa_persistence_bounds(self):
        """Test that AAA estimated persistence is within valid bounds."""
        model = ADAM(model="AAA", lags=[12])
        model.fit(AIRPASSENGERS)

        B = model.coef
        alpha, beta, gamma = B[0], B[1], B[2]

        # All persistence parameters should be in [0, 1]
        assert 0 <= alpha <= 1, f"Alpha {alpha} out of bounds"
        assert 0 <= beta <= 1, f"Beta {beta} out of bounds"
        assert 0 <= gamma <= 1, f"Gamma {gamma} out of bounds"

    def test_aaa_estimated_persistence_values(self):
        """Test AAA estimated persistence values match expected."""
        model = ADAM(model="AAA", lags=[12])
        model.fit(AIRPASSENGERS)

        B = model.coef
        # Reference values
        expected_alpha = 0.24444748
        expected_beta = 0.00108759
        expected_gamma = 0.75490673

        assert np.isclose(B[0], expected_alpha, rtol=1e-3), \
            f"Alpha {B[0]} differs from expected {expected_alpha}"
        assert np.isclose(B[1], expected_beta, rtol=0.1), \
            f"Beta {B[1]} differs from expected {expected_beta}"
        assert np.isclose(B[2], expected_gamma, rtol=1e-3), \
            f"Gamma {B[2]} differs from expected {expected_gamma}"

    def test_fixed_persistence_higher_loss(self):
        """Test that fixing persistence to suboptimal values increases loss."""
        # Optimal (free) estimation
        model_free = ADAM(model="AAA", lags=[12])
        model_free.fit(AIRPASSENGERS)
        loss_free = model_free.loss_value

        # Fixed alpha
        model_fixed = ADAM(model="AAA", lags=[12], persistence={"level": 0.5})
        model_fixed.fit(AIRPASSENGERS)
        loss_fixed = model_fixed.loss_value

        assert loss_fixed > loss_free, \
            f"Fixed persistence loss {loss_fixed} should be > free loss {loss_free}"

    def test_more_fixed_params_higher_loss(self):
        """Test that fixing more parameters generally increases loss."""
        # Only alpha fixed
        model1 = ADAM(model="AAA", lags=[12], persistence={"level": 0.5})
        model1.fit(AIRPASSENGERS)
        loss1 = model1.loss_value

        # Alpha and beta fixed
        model2 = ADAM(model="AAA", lags=[12], persistence={"level": 0.5, "trend": 0.1})
        model2.fit(AIRPASSENGERS)
        loss2 = model2.loss_value

        # Fixing more parameters with suboptimal values should increase loss
        assert loss2 > loss1, \
            f"Loss with alpha+beta fixed {loss2} should be > loss with only alpha fixed {loss1}"


class TestADAMAirPassengersModelComparison:
    """Tests comparing different models on AirPassengers."""

    def test_seasonal_model_better_than_nonseasonal(self):
        """Test that seasonal model (ANA) fits better than non-seasonal (ANN)."""
        model_ann = ADAM(model="ANN", lags=[1])
        model_ann.fit(AIRPASSENGERS)
        loss_ann = model_ann.loss_value

        model_ana = ADAM(model="ANA", lags=[12])
        model_ana.fit(AIRPASSENGERS)
        loss_ana = model_ana.loss_value

        # Seasonal model should have lower loss for this seasonal data
        assert loss_ana < loss_ann, \
            f"ANA loss {loss_ana} should be < ANN loss {loss_ann}"

    def test_different_models_different_losses(self):
        """Test that different model specifications produce different losses."""
        models = {
            "ANN": ADAM(model="ANN", lags=[1]),
            "AAN": ADAM(model="AAN", lags=[1]),
            "ANA": ADAM(model="ANA", lags=[12]),
        }

        losses = {}
        for name, model in models.items():
            model.fit(AIRPASSENGERS)
            losses[name] = model.loss_value

        # All losses should be different
        loss_values = list(losses.values())
        assert len(set(loss_values)) == len(loss_values), \
            f"Expected different losses, got: {losses}"


class TestADAMMultipleSeasonalPersistence:
    """Tests for models with multiple seasonal components and partial persistence."""

    def test_two_seasonal_full_persistence(self):
        """Test model with two seasonal lags and full persistence provided."""
        # Model with two seasonal components (quarterly and annual for monthly data)
        model = ADAM(
            model="ANA",
            lags=[3, 12],
            persistence={"level": 0.3, "seasonal": [0.4, 0.5]}
        )
        model.fit(AIRPASSENGERS)

        assert model.coef is not None
        # Verify model runs without error with full seasonal specification

    def test_two_seasonal_partial_persistence_first_only(self):
        """Test model with two seasonal lags, only first gamma provided."""
        # Only gamma_1 provided, gamma_2 should be estimated
        model = ADAM(
            model="ANA",
            lags=[3, 12],
            persistence={"level": 0.3, "seasonal": [0.4]}
        )
        model.fit(AIRPASSENGERS)

        assert model.coef is not None
        # B should contain estimated gamma_2
        B = model.coef
        assert len(B) >= 1, "Should have at least one estimated parameter (gamma_2)"

    def test_two_seasonal_no_persistence(self):
        """Test model with two seasonal lags, all persistence estimated."""
        model = ADAM(model="ANA", lags=[3, 12])
        model.fit(AIRPASSENGERS)

        assert model.coef is not None
        B = model.coef
        # Should estimate alpha + gamma = 2 persistence params (lags combined)
        assert len(B) >= 2

    def test_partial_seasonal_different_from_full(self):
        """Test that partial and full seasonal persistence produce different results."""
        # Full specification
        model_full = ADAM(
            model="ANA",
            lags=[3, 12],
            persistence={"seasonal": [0.4, 0.5]}
        )
        model_full.fit(AIRPASSENGERS)
        loss_full = model_full.loss_value

        # Partial specification (only first gamma, second estimated)
        model_partial = ADAM(
            model="ANA",
            lags=[3, 12],
            persistence={"seasonal": [0.4]}
        )
        model_partial.fit(AIRPASSENGERS)
        loss_partial = model_partial.loss_value

        # Losses should differ (partial has one free parameter)
        assert loss_full != loss_partial, \
            f"Full loss {loss_full} should differ from partial loss {loss_partial}"

    def test_aaa_full_persistence_all_components(self):
        """Test AAA with full persistence including trend and seasonal."""
        model = ADAM(
            model="AAA",
            lags=[12],
            persistence={"level": 0.3, "trend": 0.05, "seasonal": [0.6]}
        )
        model.fit(AIRPASSENGERS)

        assert model.coef is not None
        # With all persistence fixed, B should not contain persistence params
        # (only initial states, phi, etc.)

    def test_aaa_partial_persistence_trend_seasonal(self):
        """Test AAA with trend provided but seasonal estimated."""
        model = ADAM(
            model="AAA",
            lags=[12],
            persistence={"level": 0.3, "trend": 0.05}
        )
        model.fit(AIRPASSENGERS)

        assert model.coef is not None
        # Should estimate gamma
        B = model.coef
        assert len(B) >= 1, "Should estimate at least gamma"

    def test_double_seasonal_estimates_three_params(self):
        """Test that ETS(M,N,M) with lags=[3,12] estimates alpha, gamma1, gamma2."""
        model = ADAM(
            model="MNM",
            lags=[3, 12],
            initial="backcasting"
        )
        model.fit(AIRPASSENGERS)

        assert model.coef is not None
        B = model.coef
        # With backcasting, should estimate exactly 3 persistence params:
        # alpha, gamma1, gamma2
        assert len(B) == 3, f"Expected 3 parameters (alpha, gamma1, gamma2), got {len(B)}"

    def test_double_seasonal_ana_estimates_three_params(self):
        """Test that ETS(A,N,A) with lags=[3,12] estimates alpha, gamma1, gamma2."""
        model = ADAM(
            model="ANA",
            lags=[3, 12],
            initial="backcasting"
        )
        model.fit(AIRPASSENGERS)

        assert model.coef is not None
        B = model.coef
        # With backcasting, should estimate exactly 3 persistence params
        assert len(B) == 3, f"Expected 3 parameters (alpha, gamma1, gamma2), got {len(B)}"

    def test_double_seasonal_partial_gamma(self):
        """Test double seasonal with only gamma1 provided."""
        model = ADAM(
            model="ANA",
            lags=[3, 12],
            persistence={"seasonal": [0.4]},
            initial="backcasting"
        )
        model.fit(AIRPASSENGERS)

        assert model.coef is not None
        B = model.coef
        # Should estimate alpha and gamma2 (gamma1 is provided)
        assert len(B) == 2, f"Expected 2 parameters (alpha, gamma2), got {len(B)}"
