"""
Unit tests for the msdecompose function.

Tests cover:
- Basic functionality and output format
- Different smoother types (lowess, ma, global)
- Additive and multiplicative decomposition
- Multiple seasonal periods
- Edge cases
"""

import numpy as np
import pytest

from smooth import msdecompose


class TestMsdecomposeBasic:
    """Basic functionality tests for msdecompose."""

    def test_import(self):
        """Test that msdecompose can be imported from smooth."""
        from smooth import msdecompose
        assert callable(msdecompose)

    def test_basic_output_format(self, seasonal_series):
        """Test that msdecompose returns correct output format."""
        result = msdecompose(seasonal_series, lags=[12])

        assert isinstance(result, dict)
        assert 'y' in result
        assert 'states' in result
        assert 'initial' in result
        assert 'seasonal' in result
        assert 'fitted' in result
        assert 'lags' in result
        assert 'type' in result
        assert 'smoother' in result

    def test_output_shapes(self, seasonal_series):
        """Test output array shapes."""
        n = len(seasonal_series)
        result = msdecompose(seasonal_series, lags=[12])

        # y should match input
        assert len(result['y']) == n

        # states should have correct shape (n, 3) for level, trend, seasonal
        assert result['states'].shape[0] == n
        assert result['states'].shape[1] == 3  # level, trend, seasonal

        # fitted should match input length
        assert len(result['fitted']) == n

        # seasonal should be a list with one array
        assert len(result['seasonal']) == 1
        assert len(result['seasonal'][0]) == n

    def test_initial_values_structure(self, seasonal_series):
        """Test initial values structure."""
        result = msdecompose(seasonal_series, lags=[12])

        assert 'nonseasonal' in result['initial']
        assert 'seasonal' in result['initial']
        assert 'level' in result['initial']['nonseasonal']
        assert 'trend' in result['initial']['nonseasonal']
        assert len(result['initial']['seasonal']) == 1
        assert len(result['initial']['seasonal'][0]) == 12


class TestMsdecomposeSmoothers:
    """Tests for different smoother types."""

    def test_lowess_smoother(self, seasonal_series):
        """Test LOWESS smoother."""
        result = msdecompose(seasonal_series, lags=[12], smoother='lowess')
        assert result['smoother'] == 'lowess'
        assert not np.any(np.isnan(result['fitted']))

    def test_ma_smoother(self, seasonal_series):
        """Test moving average smoother."""
        result = msdecompose(seasonal_series, lags=[12], smoother='ma')
        assert result['smoother'] == 'ma'
        # MA produces NaN at edges
        assert not np.all(np.isnan(result['fitted']))

    def test_global_smoother(self, seasonal_series):
        """Test global smoother."""
        result = msdecompose(seasonal_series, lags=[12], smoother='global')
        assert result['smoother'] == 'global'
        assert not np.any(np.isnan(result['fitted']))

    def test_supsmu_smoother(self, seasonal_series):
        """Test supsmu smoother (uses LOWESS internally)."""
        result = msdecompose(seasonal_series, lags=[12], smoother='supsmu')
        assert result['smoother'] == 'supsmu'
        assert not np.any(np.isnan(result['fitted']))

    def test_different_smoothers_different_results(self, seasonal_series):
        """Test that different smoothers produce different results."""
        result_lowess = msdecompose(seasonal_series, lags=[12], smoother='lowess')
        msdecompose(seasonal_series, lags=[12], smoother='ma')
        result_global = msdecompose(seasonal_series, lags=[12], smoother='global')

        # Results should differ
        assert not np.allclose(
            result_lowess['fitted'], result_global['fitted'], equal_nan=True
        )


class TestMsdecomposeTypes:
    """Tests for additive and multiplicative decomposition."""

    def test_additive_type(self, seasonal_series):
        """Test additive decomposition."""
        result = msdecompose(seasonal_series, lags=[12], type='additive')
        assert result['type'] == 'additive'

    def test_multiplicative_type(self, multiplicative_series):
        """Test multiplicative decomposition."""
        result = msdecompose(multiplicative_series, lags=[12], type='multiplicative')
        assert result['type'] == 'multiplicative'

    def test_multiplicative_positive_data(self, multiplicative_series):
        """Test that multiplicative works with positive data."""
        result = msdecompose(multiplicative_series, lags=[12], type='multiplicative')

        # Seasonal components should be around 1 for multiplicative
        seasonal_mean = np.nanmean(result['seasonal'][0])
        # After centering in log space and exponentiating, mean should be close to 1
        assert 0.5 < seasonal_mean < 2.0

    def test_invalid_type_raises(self, seasonal_series):
        """Test that invalid type raises error."""
        with pytest.raises(ValueError):
            msdecompose(seasonal_series, lags=[12], type='invalid')

    def test_invalid_smoother_raises(self, seasonal_series):
        """Test that invalid smoother raises error."""
        with pytest.raises(ValueError):
            msdecompose(seasonal_series, lags=[12], smoother='invalid')


class TestMsdecomposeMultipleLags:
    """Tests for multiple seasonal periods."""

    def test_two_lags(self):
        """Test decomposition with two seasonal periods."""
        np.random.seed(42)
        n = 336  # 2 weeks of hourly data
        t = np.arange(n)
        # Daily + weekly pattern
        daily = 5 * np.sin(2 * np.pi * t / 24)
        weekly = 10 * np.sin(2 * np.pi * t / 168)
        y = 100 + daily + weekly + np.random.randn(n) * 2

        result = msdecompose(y, lags=[24, 168])

        # Should have two seasonal components
        assert len(result['seasonal']) == 2
        assert len(result['initial']['seasonal']) == 2

        # First seasonal initial should have 24 values
        assert len(result['initial']['seasonal'][0]) == 24
        # Second seasonal initial should have 168 values
        assert len(result['initial']['seasonal'][1]) == 168

    def test_lags_sorted(self, seasonal_series):
        """Test that lags are sorted in output."""
        result = msdecompose(seasonal_series, lags=[12, 4])
        np.testing.assert_array_equal(result['lags'], np.array([4, 12]))

    def test_duplicate_lags_removed(self, seasonal_series):
        """Test that duplicate lags are removed."""
        result = msdecompose(seasonal_series, lags=[12, 12, 4, 4])
        np.testing.assert_array_equal(result['lags'], np.array([4, 12]))


class TestMsdecomposeEdgeCases:
    """Edge case tests for msdecompose."""

    def test_short_series(self, short_series):
        """Test with short series."""
        result = msdecompose(short_series, lags=[4], smoother='lowess')
        assert len(result['fitted']) == len(short_series)

    def test_series_with_nan(self, seasonal_series):
        """Test handling of NaN values."""
        y = seasonal_series.copy()
        y[10] = np.nan
        y[50] = np.nan

        result = msdecompose(y, lags=[12], smoother='lowess')

        # Should handle NaN values
        assert not np.all(np.isnan(result['fitted']))

    def test_lag_equals_one(self):
        """Test with lag=1 (no seasonality)."""
        np.random.seed(42)
        y = 100 + np.arange(50) * 0.5 + np.random.randn(50) * 2

        result = msdecompose(y, lags=[1], smoother='lowess')

        # Should work without error
        assert 'states' in result

    def test_empty_lags_list(self, seasonal_series):
        """Test with empty lags list."""
        # Empty lags should be treated as lags=[1]
        result = msdecompose(seasonal_series, lags=[])
        assert 'states' in result

    def test_ma_switches_to_lowess_for_small_sample(self):
        """Test that MA smoother switches to LOWESS for small samples."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # 5 observations

        # With lag=12 (larger than sample), should switch to LOWESS
        with pytest.warns(UserWarning, match="Moving average does not work"):
            result = msdecompose(y, lags=[12], smoother='ma')

        assert result['smoother'] == 'lowess'


class TestMsdecomposeDecompositionQuality:
    """Tests for decomposition quality."""

    def test_fitted_values_reasonable(self, airpassengers):
        """Test that fitted values are reasonable."""
        result = msdecompose(airpassengers, lags=[12], smoother='lowess')

        # Fitted should be close to actual
        residuals = airpassengers - result['fitted']
        rmse = np.sqrt(np.nanmean(residuals ** 2))

        # RMSE should be reasonable (less than 20% of mean)
        assert rmse < 0.2 * np.mean(airpassengers)

    def test_seasonal_pattern_periodic(self, seasonal_series):
        """Test that seasonal pattern is approximately periodic."""
        result = msdecompose(seasonal_series, lags=[12], smoother='lowess')

        seasonal = result['seasonal'][0]

        # First 12 values should be similar to values 12 positions later
        for i in range(12, min(24, len(seasonal))):
            # Allow some variation but should be correlated
            if not np.isnan(seasonal[i]) and not np.isnan(seasonal[i - 12]):
                diff = abs(seasonal[i] - seasonal[i - 12])
                assert diff < 10  # Reasonable threshold

    def test_trend_monotonic_for_trending_data(self):
        """Test that trend is approximately monotonic for trending data."""
        np.random.seed(42)
        n = 100
        y = 100 + np.arange(n) * 2 + np.random.randn(n) * 5

        result = msdecompose(y, lags=[1], smoother='lowess')

        # Trend (first column of states) should be mostly increasing
        trend = result['states'][:, 0]
        valid_trend = trend[~np.isnan(trend)]
        increasing = np.sum(np.diff(valid_trend) > 0) / len(np.diff(valid_trend))

        assert increasing > 0.8  # At least 80% of steps should be increasing
