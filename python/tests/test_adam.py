"""
Unit tests for the ADAM class.

Tests cover:
- Initialization and configuration
- Model fitting
- Prediction
- Different model types (ETS combinations)
"""

import numpy as np
import pytest

from smooth import ADAM


class TestADAMInitialization:
    """Tests for ADAM initialization."""

    def test_import(self):
        """Test that ADAM can be imported from smooth."""
        from smooth import ADAM
        assert ADAM is not None

    def test_basic_init(self):
        """Test basic initialization."""
        model = ADAM(model="ANN")
        assert model is not None

    def test_init_with_lags(self):
        """Test initialization with lags."""
        model = ADAM(model="ANA", lags=[12])
        assert model is not None
        assert 12 in model.lags


class TestADAMFit:
    """Tests for ADAM fitting."""

    def test_fit_basic(self, simple_series):
        """Test basic model fitting."""
        model = ADAM(model="ANN")
        model.fit(simple_series)

        # Model should have been fitted
        assert model.coef is not None
        assert len(model.coef) > 0

    def test_fit_seasonal(self, seasonal_series):
        """Test fitting seasonal model."""
        model = ADAM(model="ANA", lags=[12])
        model.fit(seasonal_series)

        assert model.coef is not None

    def test_fit_returns_self(self, simple_series):
        """Test that fit returns self for chaining."""
        model = ADAM(model="ANN")
        result = model.fit(simple_series)

        assert result is model

    def test_fit_stores_data(self, simple_series):
        """Test that fit stores the training data."""
        model = ADAM(model="ANN")
        model.fit(simple_series)

        # data property should contain the training data
        assert model.data is not None
        assert len(model.data) == len(simple_series)


class TestADAMPredict:
    """Tests for ADAM prediction."""

    def test_predict_basic(self, simple_series):
        """Test basic prediction."""
        model = ADAM(model="ANN")
        model.fit(simple_series)

        forecast = model.predict(h=10)

        # Should return DataFrame with 'mean' column
        assert hasattr(forecast, 'shape')
        assert forecast.shape[0] == 10
        assert 'mean' in forecast.columns

    def test_predict_seasonal(self, seasonal_series):
        """Test seasonal prediction."""
        model = ADAM(model="ANA", lags=[12])
        model.fit(seasonal_series)

        forecast = model.predict(h=24)

        assert forecast.shape[0] == 24
        assert not forecast['mean'].isna().any()

    def test_predict_before_fit_raises(self):
        """Test that predict before fit raises error."""
        model = ADAM(model="ANN")

        with pytest.raises((AttributeError, ValueError, RuntimeError, KeyError)):
            model.predict(h=10)

    def test_predict_includes_intervals(self, simple_series):
        """Test that predict includes prediction intervals when requested."""
        model = ADAM(model="ANN")
        model.fit(simple_series)

        forecast = model.predict(h=10, interval="prediction")

        # Should have lower and upper bounds
        cols = forecast.columns.tolist()
        assert any('lower' in c for c in cols)
        assert any('upper' in c for c in cols)


class TestADAMModelTypes:
    """Tests for different model types."""

    @pytest.mark.parametrize("model_code", [
        "ANN",  # Simple exponential smoothing
        "AAN",  # Holt's linear trend
        "AAdN", # Damped trend
    ])
    def test_ets_models_nonseasonal(self, simple_series, model_code):
        """Test non-seasonal ETS models."""
        model = ADAM(model=model_code)
        model.fit(simple_series)
        forecast = model.predict(h=5)

        assert forecast.shape[0] == 5
        assert not forecast['mean'].isna().any()

    @pytest.mark.parametrize("model_code", [
        "ANA",  # Additive seasonality
        "AAA",  # Trend + seasonality
    ])
    def test_ets_models_seasonal(self, seasonal_series, model_code):
        """Test seasonal ETS models."""
        model = ADAM(model=model_code, lags=[12])
        model.fit(seasonal_series)
        forecast = model.predict(h=12)

        assert forecast.shape[0] == 12
        assert not forecast['mean'].isna().any()

    def test_multiplicative_error(self, multiplicative_series):
        """Test multiplicative error model."""
        model = ADAM(model="MNN")
        model.fit(multiplicative_series)
        forecast = model.predict(h=5)

        assert forecast.shape[0] == 5
        assert not forecast['mean'].isna().any()
        # Multiplicative error model forecasts should stay positive for positive data
        assert (forecast['mean'] > 0).all()


class TestADAMModelSelection:
    """Tests for automatic model selection."""

    def test_model_zzz(self, seasonal_series):
        """Test automatic model selection with ZZZ."""
        model = ADAM(model="ZZZ", lags=[12])
        model.fit(seasonal_series)

        assert model.coef is not None
        forecast = model.predict(h=12)
        assert forecast.shape[0] == 12
        assert not forecast['mean'].isna().any()

    def test_model_zxz(self, seasonal_series):
        """Test automatic selection for error and seasonality (no trend) with ZXZ."""
        model = ADAM(model="ZXZ", lags=[12])
        model.fit(seasonal_series)

        assert model.coef is not None
        forecast = model.predict(h=12)
        assert forecast.shape[0] == 12
        assert not forecast['mean'].isna().any()

    def test_model_fff(self, seasonal_series):
        """Test full model with FFF."""
        model = ADAM(model="FFF", lags=[12])
        model.fit(seasonal_series)

        assert model.coef is not None
        forecast = model.predict(h=12)
        assert forecast.shape[0] == 12
        assert not forecast['mean'].isna().any()

    def test_model_ppp(self, seasonal_series):
        """Test partial automatic selection with PPP."""
        model = ADAM(model="PPP", lags=[12])
        model.fit(seasonal_series)

        assert model.coef is not None
        forecast = model.predict(h=12)
        assert forecast.shape[0] == 12
        assert not forecast['mean'].isna().any()

    def test_model_zzz_nonseasonal(self, simple_series):
        """Test ZZZ model selection without seasonality."""
        model = ADAM(model="ZZZ", lags=[1])
        model.fit(simple_series)

        assert model.coef is not None
        forecast = model.predict(h=5)
        assert forecast.shape[0] == 5


class TestADAMEdgeCases:
    """Edge case tests for ADAM."""

    def test_short_series(self, short_series):
        """Test with short series."""
        model = ADAM(model="ANN")
        model.fit(short_series)
        forecast = model.predict(h=3)

        assert forecast.shape[0] == 3

    def test_series_with_zeros(self):
        """Test series containing zeros."""
        np.random.seed(42)
        y = np.abs(np.random.randn(50)) + 0.1
        y[10] = 0.001  # Near zero but not exactly zero
        y[20] = 0.001

        model = ADAM(model="ANN")
        model.fit(y)
        forecast = model.predict(h=5)

        assert not forecast['mean'].isna().any()

    def test_large_horizon(self, simple_series):
        """Test prediction with large horizon."""
        model = ADAM(model="ANN")
        model.fit(simple_series)
        forecast = model.predict(h=100)

        assert forecast.shape[0] == 100
        assert not forecast['mean'].isna().any()


class TestADAMAttributes:
    """Tests for ADAM attributes after fitting."""

    def test_persistence_level_attribute(self, simple_series):
        """Test that persistence level (alpha) is accessible."""
        model = ADAM(model="ANN")
        model.fit(simple_series)

        # Should have persistence_level_ (alpha parameter)
        assert hasattr(model, 'persistence_level_')
        # persistence_level_ may be None for some models, check if numeric
        if model.persistence_level_ is not None:
            assert 0 <= model.persistence_level_ <= 1

    def test_persistence_trend_attribute(self, simple_series):
        """Test that persistence trend (beta) is accessible for trend models."""
        model = ADAM(model="AAN")
        model.fit(simple_series)

        assert hasattr(model, 'persistence_trend_')

    def test_phi_attribute(self, simple_series):
        """Test that phi (damping) is accessible for damped models."""
        model = ADAM(model="AAdN")
        model.fit(simple_series)

        assert hasattr(model, 'phi_')


class TestADAMBounds:
    """Tests for parameter bounds."""

    def test_admissible_bounds_linear_series(self):
        """Test admissible bounds with linear series.

        For a linear series (1 to 20), ETS(ANN) with admissible bounds
        should produce a smoothing parameter (alpha) greater than 1,
        which is outside the usual [0,1] bounds but still admissible.
        """
        y = np.arange(1, 21, dtype=float)
        model = ADAM(model="ANN", bounds="admissible")
        model.fit(y)

        alpha = model.coef[0]
        assert alpha > 1, (
            f"Expected alpha > 1 for linear series with admissible bounds, "
            f"got {alpha}"
        )


class TestADAMConstant:
    """Tests for the constant/drift parameter."""

    @pytest.fixture
    def linear_series(self):
        np.random.seed(42)
        return np.arange(1, 61, dtype=float) + np.random.randn(60) * 0.1

    def test_ets_constant_fits(self, linear_series):
        """ETS(ANN) with constant=True fits without error."""
        model = ADAM("ANN", constant=True)
        model.fit(linear_series)
        assert np.isfinite(model.constant_value)

    def test_ets_constant_improves_fit(self, linear_series):
        """constant=True gives lower AICc than no constant on a linear-trend series."""
        m0 = ADAM("ANN").fit(linear_series)
        m1 = ADAM("ANN", constant=True).fit(linear_series)
        assert m1.loss_value <= m0.loss_value

    def test_arima_constant_fits(self, linear_series):
        """ARIMA(1,1,1) with constant=True fits without error."""
        model = ADAM("NNN", ar_order=1, i_order=1, ma_order=1, constant=True)
        model.fit(linear_series)
        assert np.isfinite(model.constant_value)

    def test_fixed_constant(self, linear_series):
        """constant=0.5 (fixed value) is stored and accessible after fit."""
        model = ADAM("ANN", constant=0.5)
        model.fit(linear_series)
        assert model.constant_value == pytest.approx(0.5)

    def test_fixed_constant_numeric_ets(self, linear_series):
        """constant=1.6 (numeric) is preserved exactly throughout optimisation."""
        model = ADAM("ANN", constant=1.6)
        model.fit(linear_series)
        assert model.constant_value == pytest.approx(1.6)

    def test_fixed_constant_numeric_arima(self, linear_series):
        """Numeric constant is preserved for ARIMA models too."""
        model = ADAM("NNN", ar_order=1, i_order=1, ma_order=1, constant=0.3)
        model.fit(linear_series)
        assert model.constant_value == pytest.approx(0.3)

    def test_constant_shown_in_summary(self, linear_series):
        """Model summary includes the constant value when constant=True."""
        model = ADAM("ANN", constant=True)
        model.fit(linear_series)
        summary = str(model)
        assert "Intercept" in summary or "constant" in summary.lower()


class TestADAMARIMAOrders:
    """Tests for ARIMA orders and lags interaction."""

    @pytest.fixture
    def series60(self):
        np.random.seed(1)
        return np.random.randn(60)

    def test_short_lags_equals_explicit_padded(self, series60):
        """lags=[12] + ar=[1] equals lags=[1,12] + ar=[1,0] (lag=1 auto-prepended)."""
        m1 = ADAM("NNN", lags=[12], ar_order=[1], i_order=[1], ma_order=[1]).fit(series60)
        m2 = ADAM("NNN", lags=[1, 12], ar_order=[1, 0], i_order=[1, 0], ma_order=[1, 0]).fit(series60)

        assert m1._arima["lags_model_arima"] == m2._arima["lags_model_arima"]
        assert abs(m1.loss_value - m2.loss_value) < 1e-6

    def test_nonseasonal_arima_at_lag1_only(self, series60):
        """ar_order=[1] with lags=[12] gives non-seasonal ARIMA (at lag 1 only)."""
        m = ADAM("NNN", lags=[12], ar_order=[1], i_order=[1], ma_order=[1]).fit(series60)

        arima = m._arima
        # After prepending lag=1, orders should be [1,0] (non-zero only at lag=1)
        assert arima["ar_orders"] == [1, 0]
        assert arima["ma_orders"] == [1, 0]

    def test_seasonal_only_arima(self, series60):
        """ar_order=[0,1] with lags=[1,12] gives seasonal-only ARIMA at lag=12."""
        m = ADAM("NNN", lags=[1, 12], ar_order=[0, 1], i_order=[0, 1], ma_order=[0, 1]).fit(series60)

        arima = m._arima
        assert arima["ar_orders"] == [0, 1]
        assert arima["ma_orders"] == [0, 1]


class TestADAMArmaFixed:
    """Tests for fixed ARMA parameters via the arma argument."""

    @pytest.fixture
    def arima_series(self):
        np.random.seed(42)
        return np.cumsum(np.random.randn(60)) + 100.0

    @pytest.fixture
    def long_series(self):
        np.random.seed(7)
        trend = np.arange(120) * 0.5
        seasonal = np.tile(np.sin(np.linspace(0, 2 * np.pi, 12)), 10)
        return trend + seasonal + np.random.randn(120) * 0.3

    def test_arma_ar_fixed_non_seasonal(self, arima_series):
        """Fixed AR coef is stored in arma_parameters, not in B."""
        m = ADAM("NNN", ar_order=1, i_order=0, ma_order=0, arma={"ar": [0.5]})
        m.fit(arima_series)
        assert m._arima["arma_parameters"] == pytest.approx([0.5])
        assert len(m.coef) == 0  # nothing estimated

    def test_arma_ma_fixed_non_seasonal(self, arima_series):
        """Fixed MA coef is stored in arma_parameters."""
        m = ADAM("NNN", ar_order=0, i_order=1, ma_order=1, arma={"ma": [0.3]})
        m.fit(arima_series)
        assert m._arima["arma_parameters"] == pytest.approx([0.3])
        assert len(m.coef) == 0

    def test_arma_both_fixed(self, arima_series):
        """Both AR and MA fixed: arma_parameters contains both in order."""
        m = ADAM("NNN", ar_order=1, i_order=1, ma_order=1,
                 arma={"ar": [0.5], "ma": [0.2]})
        m.fit(arima_series)
        assert m._arima["arma_parameters"] == pytest.approx([0.5, 0.2])
        assert len(m.coef) == 0

    def test_arma_fixed_produces_different_fitted(self, arima_series):
        """Different fixed MA values produce different fitted values."""
        m03 = ADAM("NNN", ar_order=0, i_order=1, ma_order=1, arma={"ma": [0.3]}).fit(arima_series)
        m08 = ADAM("NNN", ar_order=0, i_order=1, ma_order=1, arma={"ma": [0.8]}).fit(arima_series)
        assert not np.allclose(m03.fitted, m08.fitted)

    def test_arma_with_constant(self, arima_series):
        """Fixed arma with constant=True fits without error."""
        m = ADAM("NNN", ar_order=1, i_order=1, ma_order=1,
                 arma={"ar": [0.5], "ma": [0.2]}, constant=True)
        m.fit(arima_series)
        assert np.isfinite(m.constant_value)

    def test_arma_sarima(self, long_series):
        """Fixed arma on SARIMA fits without error."""
        m = ADAM("NNN", lags=[1, 12],
                 ar_order=[1, 0], i_order=[1, 1], ma_order=[1, 1],
                 arma={"ma": [0.3, 0.2]})
        m.fit(long_series)
        assert m._arima["arma_parameters"] is not None
        assert np.all(np.isfinite(m.fitted))

    def test_msarima_arma_fixed(self, arima_series):
        """MSARIMA with fixed arma stores fixed values in arma_parameters."""
        from smooth import MSARIMA
        m = MSARIMA(ar_order=1, i_order=1, ma_order=1,
                    arma={"ar": [0.5], "ma": [0.2]})
        m.fit(arima_series)
        assert m._arima["arma_parameters"] == pytest.approx([0.5, 0.2])
        assert len(m.coef) == 0


class TestADAMReproducibility:
    """Tests for reproducibility."""

    def test_same_seed_same_result(self, simple_series):
        """Test that same random seed gives same result."""
        np.random.seed(42)
        model1 = ADAM(model="ANN")
        model1.fit(simple_series)
        forecast1 = model1.predict(h=5)

        np.random.seed(42)
        model2 = ADAM(model="ANN")
        model2.fit(simple_series)
        forecast2 = model2.predict(h=5)

        np.testing.assert_array_almost_equal(
            forecast1['mean'].values,
            forecast2['mean'].values
        )
