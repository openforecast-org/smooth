"""
Unit tests for ADAM model combination (CCC) functionality.

Tests cover:
- Model combination with "CCC" (combine all)
- Partial combination with "CCN", "ACA", etc.
- IC weights calculation
- Combined fitted values and residuals
- Combined forecasts and prediction intervals
"""

import numpy as np
import pytest

from smooth import ADAM
from smooth.adam_general.core.utils.ic import calculate_ic_weights


class TestICWeights:
    """Tests for IC weights calculation."""

    def test_calculate_ic_weights_basic(self):
        """Test basic IC weights calculation."""
        ic_values = {"ANN": 100.0, "AAN": 98.0, "AAA": 102.0}
        weights = calculate_ic_weights(ic_values)

        # Weights should sum to 1
        assert abs(sum(weights.values()) - 1.0) < 1e-10

        # Best model (lowest IC) should have highest weight
        assert weights["AAN"] > weights["ANN"]
        assert weights["AAN"] > weights["AAA"]

    def test_calculate_ic_weights_equal(self):
        """Test IC weights with equal IC values."""
        ic_values = {"ANN": 100.0, "AAN": 100.0, "AAA": 100.0}
        weights = calculate_ic_weights(ic_values)

        # All weights should be equal
        assert abs(weights["ANN"] - weights["AAN"]) < 1e-10
        assert abs(weights["AAN"] - weights["AAA"]) < 1e-10

    def test_calculate_ic_weights_single(self):
        """Test IC weights with single model."""
        ic_values = {"ANN": 100.0}
        weights = calculate_ic_weights(ic_values)

        assert weights["ANN"] == 1.0

    def test_calculate_ic_weights_empty(self):
        """Test IC weights with empty input."""
        weights = calculate_ic_weights({})
        assert weights == {}

    def test_calculate_ic_weights_threshold(self):
        """Test that tiny weights are zeroed out."""
        # Large IC difference should result in near-zero weight for worse model
        ic_values = {"ANN": 100.0, "AAN": 150.0}
        weights = calculate_ic_weights(ic_values, threshold=1e-5)

        # Model with much higher IC should have zero weight
        assert weights["AAN"] == 0.0
        assert weights["ANN"] == 1.0


class TestADAMCombinationFit:
    """Tests for ADAM combination fitting."""

    def test_ccc_fit_basic(self, simple_series):
        """Test basic CCC model fitting."""
        model = ADAM(model="CCC", lags=[1])
        model.fit(simple_series)

        # Should be marked as combined
        assert model.is_combined

        # Should have IC weights
        assert model.ic_weights is not None
        assert abs(sum(model.ic_weights.values()) - 1.0) < 1e-10

    def test_ccc_fit_has_models(self, simple_series):
        """Test that combined model has individual models."""
        model = ADAM(model="CCC", lags=[1])
        model.fit(simple_series)

        # Should have list of models
        assert model.models is not None
        assert len(model.models) > 0

        # Each model should have name and weight
        for m in model.models:
            assert "name" in m
            assert "weight" in m
            assert m["weight"] >= 0  # All models stored; filtering at predict-time

    def test_ccc_fitted_values(self, simple_series):
        """Test that combined fitted values are computed."""
        model = ADAM(model="CCC", lags=[1])
        model.fit(simple_series)

        fitted = model.fitted
        assert fitted is not None
        assert len(fitted) == len(simple_series)
        assert not np.isnan(fitted).any()

    def test_ccc_residuals(self, simple_series):
        """Test that combined residuals are computed."""
        model = ADAM(model="CCC", lags=[1])
        model.fit(simple_series)

        residuals = model.residuals
        assert residuals is not None
        assert len(residuals) == len(simple_series)

        # Residuals should be y - fitted
        expected = np.array(simple_series) - model.fitted
        np.testing.assert_array_almost_equal(residuals, expected)

    def test_ccn_fit(self, simple_series):
        """Test CCN (combine error and trend, no seasonality) model fitting."""
        model = ADAM(model="CCN", lags=[1])
        model.fit(simple_series)

        assert model.is_combined
        assert model.ic_weights is not None

    def test_aca_fit(self, seasonal_series):
        """Test ACA (additive error, combine trend, additive season) model fitting."""
        model = ADAM(model="ACA", lags=[12])
        model.fit(seasonal_series)

        assert model.is_combined
        assert model.ic_weights is not None


class TestADAMCombinationPredict:
    """Tests for ADAM combination prediction."""

    def test_ccc_predict_basic(self, simple_series):
        """Test basic prediction with combined model."""
        model = ADAM(model="CCC", lags=[1])
        model.fit(simple_series)

        forecast = model.predict(h=10)

        assert forecast is not None
        assert len(forecast["mean"]) == 10
        assert not np.isnan(forecast["mean"]).any()

    def test_ccc_predict_intervals(self, simple_series):
        """Test prediction intervals with combined model."""
        model = ADAM(model="CCC", lags=[1])
        model.fit(simple_series)

        # predict() returns a DataFrame with "mean" column
        forecast = model.predict(h=10)

        assert "mean" in forecast.columns
        assert len(forecast["mean"]) == 10

    def test_ccn_predict(self, simple_series):
        """Test prediction with CCN model."""
        model = ADAM(model="CCN", lags=[1])
        model.fit(simple_series)

        forecast = model.predict(h=10)
        assert len(forecast["mean"]) == 10


class TestADAMCombinationProperties:
    """Tests for combined model properties."""

    def test_is_combined_false_for_regular(self, simple_series):
        """Test that is_combined is False for regular models."""
        model = ADAM(model="ANN", lags=[1])
        model.fit(simple_series)

        assert not model.is_combined

    def test_ic_weights_error_for_regular(self, simple_series):
        """Test that ic_weights raises error for regular models."""
        model = ADAM(model="ANN", lags=[1])
        model.fit(simple_series)

        with pytest.raises(ValueError, match="not fitted with combination"):
            _ = model.ic_weights

    def test_models_error_for_regular(self, simple_series):
        """Test that models raises error for regular models."""
        model = ADAM(model="ANN", lags=[1])
        model.fit(simple_series)

        with pytest.raises(ValueError, match="not fitted with combination"):
            _ = model.models

    def test_model_name_for_combined(self, simple_series):
        """Test model name for combined model has ETS prefix."""
        model = ADAM(model="CCC", lags=[1])
        model.fit(simple_series)

        # Model name should have ETS prefix with original spec
        assert model.model == "ETS(CCC)"
        assert model.model_name is not None


class TestADAMCombinationReporting:
    """A combination has no single likelihood, IC or distribution (R parity)."""

    def test_likelihood_and_ics_are_unavailable(self, simple_series):
        """R's logLik() returns NULL for a combination, so the ICs are empty."""
        model = ADAM(model="CCC", lags=[1])
        model.fit(simple_series)

        for name in ("loglik", "aic", "aicc", "bic", "bicc"):
            with pytest.warns(UserWarning, match="did you use combinations"):
                assert np.isnan(getattr(model, name))

    def test_distribution_stays_unresolved(self, simple_series):
        """Members span both error types, so no single distribution applies."""
        model = ADAM(model="CCC", lags=[1])
        model.fit(simple_series)
        assert model.distribution_ == "default"

        explicit = ADAM(model="CCC", lags=[1], distribution="dlnorm")
        explicit.fit(simple_series)
        assert explicit.distribution_ == "dlnorm"

    def test_nparam_uses_the_full_ic_weights(self, simple_series):
        """The sub-1% members are dropped from the fitted values, not the df."""
        model = ADAM(model="CCC", lags=[1])
        model.fit(simple_series)

        expected = 1.0  # the scale
        for name, weight in model.ic_weights.items():
            member = ADAM(model=name, lags=[1])
            member.fit(simple_series)
            expected += (member.nparam - 1.0) * weight

        assert model.nparam == pytest.approx(expected, rel=1e-12)


class TestADAMCombinationEdgeCases:
    """Tests for edge cases in model combination."""

    def test_single_valid_model(self, simple_series):
        """Test combination when only one model is valid."""
        # With very short series, may only have one valid model
        short = simple_series[:15]
        model = ADAM(model="CCN", lags=[1])
        model.fit(short)

        # Should still work even if only one model
        assert model.is_combined
        forecast = model.predict(h=5)
        assert len(forecast["mean"]) == 5

    def test_ccc_with_seasonal(self, seasonal_series):
        """Test CCC with seasonal data."""
        model = ADAM(model="CCC", lags=[12])
        model.fit(seasonal_series)

        assert model.is_combined
        forecast = model.predict(h=12)
        assert len(forecast["mean"]) == 12

    def test_custom_models_pool(self, simple_series):
        """Test that a list of models selects the best one."""
        model = ADAM(model=["ANN", "AAN", "MNN"], lags=[1])
        model.fit(simple_series)

        # Best model should be one from the pool (with ETS prefix)
        expected_models = [f"ETS({m})" for m in ["ANN", "AAN", "MNN"]]
        assert model.model in expected_models
