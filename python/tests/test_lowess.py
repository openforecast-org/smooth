"""
Unit tests for the lowess function (provided by greybox >= 1.0.2).

``smooth`` no longer ships its own LOWESS; it relies on ``greybox.lowess``.
These tests guard that dependency and cover:
- Basic functionality and output format
- Parameter variations (f, iter, delta)
- Edge cases (small datasets, unsorted data)
- R compatibility
"""

import numpy as np
import pytest
from greybox import lowess


class TestLowessBasic:
    """Basic functionality tests for lowess."""

    def test_import(self):
        """Test that lowess can be imported from greybox."""
        from greybox import lowess
        assert callable(lowess)

    def test_basic_output_format(self):
        """Test that lowess returns correct output format."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.1, 2.0, 2.9, 4.1, 5.0])

        result = lowess(x, y)

        assert isinstance(result, dict)
        assert 'x' in result
        assert 'y' in result
        assert len(result['x']) == len(x)
        assert len(result['y']) == len(y)

    def test_output_sorted(self):
        """Test that output x values are sorted."""
        x = np.array([5.0, 2.0, 8.0, 1.0, 3.0])
        y = np.array([5.1, 2.0, 8.2, 0.9, 3.0])

        result = lowess(x, y)

        # x should be sorted
        assert np.all(result['x'][:-1] <= result['x'][1:])

    def test_smoothing_effect(self):
        """Test that lowess actually smooths the data."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y_true = np.sin(x)
        y_noisy = y_true + np.random.randn(50) * 0.5  # More noise

        result = lowess(x, y_noisy, f=0.3)  # Use smaller span

        # Smoothed values should have less variance than noisy values
        var_noisy = np.var(np.diff(y_noisy))
        var_smooth = np.var(np.diff(result['y']))

        assert var_smooth < var_noisy


class TestLowessParameters:
    """Tests for lowess parameters."""

    def test_span_parameter(self):
        """Test that different f values produce different results."""
        x = np.linspace(0, 10, 30)
        y = np.sin(x) + np.random.randn(30) * 0.2

        result_smooth = lowess(x, y, f=0.8)
        result_rough = lowess(x, y, f=0.2)

        # Different spans should produce different results
        assert not np.allclose(result_smooth['y'], result_rough['y'])

        # Larger span should produce smoother (less variable) output
        var_smooth = np.var(np.diff(result_smooth['y']))
        var_rough = np.var(np.diff(result_rough['y']))
        assert var_smooth < var_rough

    def test_iter_parameter(self):
        """Test that iter parameter affects robustness."""
        x = np.linspace(0, 10, 30)
        y = np.sin(x)
        y[15] = 10  # Add outlier

        result_iter0 = lowess(x, y, iter=0)
        result_iter3 = lowess(x, y, iter=3)

        # With more iterations, outlier influence should be reduced
        # Check values near the outlier
        assert abs(result_iter3['y'][15] - np.sin(x[15])) < abs(result_iter0['y'][15] - np.sin(x[15]))

    def test_delta_parameter(self):
        """Test that delta parameter works."""
        x = np.linspace(0, 10, 100)
        y = np.sin(x)

        # Should not raise error with custom delta
        result = lowess(x, y, delta=0.5)
        assert len(result['y']) == len(y)

    def test_default_parameters(self):
        """Test default parameter values match R."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.1, 2.0, 2.9, 4.1, 5.0])

        # Default should be f=2/3, iter=3
        result_default = lowess(x, y)
        result_explicit = lowess(x, y, f=2/3, iter=3)

        np.testing.assert_array_almost_equal(result_default['y'], result_explicit['y'])


class TestLowessInputFormats:
    """Tests for different input formats."""

    def test_2d_input(self):
        """Test that 2D array input works (R-style)."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.1, 2.0, 2.9, 4.1, 5.0])
        xy = np.column_stack([x, y])

        result_2d = lowess(xy)
        result_separate = lowess(x, y)

        np.testing.assert_array_almost_equal(result_2d['y'], result_separate['y'])

    def test_list_input(self):
        """Test that list inputs are converted correctly."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.1, 2.0, 2.9, 4.1, 5.0]

        result = lowess(x, y)

        assert isinstance(result['x'], np.ndarray)
        assert isinstance(result['y'], np.ndarray)

    def test_unsorted_input(self):
        """Test that unsorted input is handled correctly."""
        x_unsorted = np.array([3.0, 1.0, 5.0, 2.0, 4.0])
        y_unsorted = np.array([3.1, 1.0, 5.0, 2.1, 4.0])

        result = lowess(x_unsorted, y_unsorted)

        # Output should be sorted by x
        expected_x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_almost_equal(result['x'], expected_x)


class TestLowessEdgeCases:
    """Edge case tests for lowess."""

    def test_small_dataset(self):
        """Test with very small dataset."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])

        result = lowess(x, y)

        assert len(result['y']) == 3

    def test_two_points(self):
        """Test with only two points."""
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 2.0])

        result = lowess(x, y)

        assert len(result['y']) == 2

    def test_duplicate_x_values(self):
        """Test handling of duplicate x values."""
        x = np.array([1.0, 2.0, 2.0, 3.0, 4.0])
        y = np.array([1.0, 2.1, 1.9, 3.0, 4.0])

        result = lowess(x, y)

        assert len(result['y']) == len(y)

    def test_mismatched_lengths_raises(self):
        """Test that mismatched x and y lengths raise error."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0])

        with pytest.raises(ValueError):
            lowess(x, y)

    def test_missing_y_with_1d_x_raises(self):
        """Test that 1D x without y raises error."""
        x = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError):
            lowess(x)


class TestLowessNumericalStability:
    """Numerical stability tests."""

    def test_large_values(self):
        """Test with large values."""
        x = np.array([1e6, 2e6, 3e6, 4e6, 5e6])
        y = np.array([1e6, 2e6, 3e6, 4e6, 5e6])

        result = lowess(x, y)

        assert not np.any(np.isnan(result['y']))
        assert not np.any(np.isinf(result['y']))

    def test_small_values(self):
        """Test with small values."""
        x = np.array([1e-6, 2e-6, 3e-6, 4e-6, 5e-6])
        y = np.array([1e-6, 2e-6, 3e-6, 4e-6, 5e-6])

        result = lowess(x, y)

        assert not np.any(np.isnan(result['y']))
        assert not np.any(np.isinf(result['y']))

    def test_mixed_scale(self):
        """Test with mixed scale data."""
        x = np.array([0.001, 0.01, 0.1, 1.0, 10.0])
        y = np.array([0.001, 0.01, 0.1, 1.0, 10.0])

        result = lowess(x, y)

        assert not np.any(np.isnan(result['y']))
