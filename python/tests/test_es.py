"""
Unit tests for the ES (Exponential Smoothing) class.

Tests cover:
- Initialization
- Model fitting
- Prediction
"""

import numpy as np
import pytest

from smooth import ES


class TestESInitialization:
    """Tests for ES initialization."""

    def test_import(self):
        """Test that ES can be imported from smooth."""
        from smooth import ES
        assert ES is not None

    def test_basic_init(self):
        """Test basic initialization."""
        model = ES(model="ANN")
        assert model is not None

    def test_init_with_lags(self):
        """Test initialization with seasonal lags."""
        model = ES(model="ANA", lags=[12])
        assert model is not None


class TestESFit:
    """Tests for ES fitting."""

    def test_fit_basic(self, simple_series):
        """Test basic model fitting."""
        model = ES(model="ANN")
        model.fit(simple_series)

        # Model should have been fitted
        assert model.coef is not None

    def test_fit_returns_self(self, simple_series):
        """Test that fit returns self."""
        model = ES(model="ANN")
        result = model.fit(simple_series)

        assert result is model

    def test_fit_seasonal(self, seasonal_series):
        """Test fitting seasonal ES model."""
        model = ES(model="ANA", lags=[12])
        model.fit(seasonal_series)

        assert model.coef is not None


class TestESPredict:
    """Tests for ES prediction."""

    def test_predict_basic(self, simple_series):
        """Test basic prediction."""
        model = ES(model="ANN")
        model.fit(simple_series)

        forecast = model.predict(h=10)

        assert forecast.shape[0] == 10
        assert 'mean' in forecast.columns
        assert not forecast['mean'].isna().any()

    def test_predict_before_fit_raises(self):
        """Test that predict before fit raises error."""
        model = ES(model="ANN")

        with pytest.raises((AttributeError, ValueError, RuntimeError, KeyError)):
            model.predict(h=10)


class TestESModelTypes:
    """Tests for different ES model types."""

    @pytest.mark.parametrize("model_code", [
        "ANN",  # Simple
        "AAN",  # Holt
        "AAdN", # Damped
    ])
    def test_nonseasonal_models(self, simple_series, model_code):
        """Test non-seasonal ES models."""
        model = ES(model=model_code)
        model.fit(simple_series)
        forecast = model.predict(h=5)

        assert forecast.shape[0] == 5

    @pytest.mark.parametrize("model_code", [
        "ANA",  # Additive seasonal
        "AAA",  # Holt-Winters additive
    ])
    def test_seasonal_models(self, seasonal_series, model_code):
        """Test seasonal ES models."""
        model = ES(model=model_code, lags=[12])
        model.fit(seasonal_series)
        forecast = model.predict(h=12)

        assert forecast.shape[0] == 12


class TestESModelSelection:
    """Tests for automatic model selection."""

    def test_model_zzz(self, seasonal_series):
        """Test automatic model selection with ZZZ."""
        model = ES(model="ZZZ", lags=[12])
        model.fit(seasonal_series)

        assert model.coef is not None
        forecast = model.predict(h=12)
        assert forecast.shape[0] == 12
        assert not forecast['mean'].isna().any()

    def test_model_zxz(self, seasonal_series):
        """Test automatic selection for error and seasonality (no trend) with ZXZ."""
        model = ES(model="ZXZ", lags=[12])
        model.fit(seasonal_series)

        assert model.coef is not None
        forecast = model.predict(h=12)
        assert forecast.shape[0] == 12
        assert not forecast['mean'].isna().any()

    def test_model_fff(self, seasonal_series):
        """Test full model with FFF."""
        model = ES(model="FFF", lags=[12])
        model.fit(seasonal_series)

        assert model.coef is not None
        forecast = model.predict(h=12)
        assert forecast.shape[0] == 12
        assert not forecast['mean'].isna().any()

    def test_model_ppp(self, seasonal_series):
        """Test partial automatic selection with PPP."""
        model = ES(model="PPP", lags=[12])
        model.fit(seasonal_series)

        assert model.coef is not None
        forecast = model.predict(h=12)
        assert forecast.shape[0] == 12
        assert not forecast['mean'].isna().any()

    def test_model_zzz_nonseasonal(self, simple_series):
        """Test ZZZ model selection without seasonality."""
        model = ES(model="ZZZ", lags=[1])
        model.fit(simple_series)

        assert model.coef is not None
        forecast = model.predict(h=5)
        assert forecast.shape[0] == 5


class TestESEdgeCases:
    """Edge case tests for ES."""

    def test_short_series(self, short_series):
        """Test with short series."""
        model = ES(model="ANN")
        model.fit(short_series)
        forecast = model.predict(h=3)

        assert forecast.shape[0] == 3

    def test_constant_series(self):
        """Test with near-constant series."""
        # Add tiny noise to avoid degenerate case (zero variance causes scipy warnings)
        np.random.seed(42)
        y = np.ones(30) * 50 + np.random.randn(30) * 1e-6

        model = ES(model="ANN")
        model.fit(y)
        forecast = model.predict(h=5)

        # Forecast mean should be close to 50
        assert np.all(np.abs(forecast['mean'].values - 50) < 5)


class TestESInitialSeasonAndX:
    """initial_season and initial_X, as R's initialSeason and initialX of es().

    R's values from
    ``es(y4, "ANA", initial="optimal", initialSeason=c(-5, 0, -8, 13))`` and
    ``es(data.frame(y=y, x=x), "ANN", initial="optimal", initialX=2)``.
    """

    Y4 = np.array(
        [
            10,
            14,
            8,
            25,
            16,
            22,
            14,
            35,
            15,
            27,
            18,
            40,
            28,
            40,
            25,
            65,
            28,
            44,
            30,
            55,
            32,
            48,
            35,
            70,
        ],
        dtype=float,
    )
    SEASON = np.array([-5.0, 0.0, -8.0, 13.0])
    X = np.array(
        [1, 3, 2, 5, 4, 6, 2, 1, 3, 5, 4, 2, 6, 5, 3, 1, 2, 4, 6, 3], dtype=float
    ).reshape(-1, 1)
    Y = np.array(
        [
            12.1,
            16.3,
            13.8,
            20.4,
            18.2,
            22.5,
            14.4,
            12.6,
            16.9,
            20.8,
            18.7,
            14.9,
            23.1,
            21.2,
            17.0,
            13.3,
            15.1,
            19.4,
            23.6,
            17.2,
        ]
    )

    def test_a_string_initial_with_a_season_estimates_the_rest(self):
        model = ES(model="ANA", lags=[4], initial="optimal", initial_season=self.SEASON)
        assert set(model.initial) == {"seasonal"}
        model.fit(self.Y4)
        assert model.initial_type == "provided"
        np.testing.assert_array_equal(model.initial_value["seasonal"], self.SEASON)
        # R: m$B and logLik(m)
        np.testing.assert_allclose(
            model.coef, [0.4969287239, 0.3641462973, 14.60355028], rtol=1e-8
        )
        assert model.loglik == pytest.approx(-77.8776638, rel=1e-8)

    def test_a_string_initial_with_x_estimates_the_level(self):
        model = ES(model="ANN", initial="optimal", initial_X=np.array([2.0]))
        assert set(model.initial) == {"xreg"}
        model.fit(self.Y, self.X)
        assert model.initial_type == "provided"
        assert list(model.initial_value["xreg"].values()) == [2.0]
        # R: m$B, alpha and the level
        np.testing.assert_allclose(model.coef, [0.5821599012, 10.13792934], rtol=1e-8)

    def test_a_dict_initial_ignores_the_season_as_r(self):
        model = ES(
            model="ANA", lags=[4], initial={"level": 20.0}, initial_season=self.SEASON
        )
        model.fit(self.Y4)
        # R: unlist(m$initial)[-1] of es(y4, "ANA", initial=list(level=20),
        # initialSeason=c(-5, 0, -8, 13)): the seasonal states are estimated
        np.testing.assert_allclose(
            model.initial_value["seasonal"],
            [-5.230788636, 1.416607605, -7.691126749, 11.50530778],
            rtol=1e-6,
        )

    def test_a_dict_initial_ignores_x_as_r(self):
        model = ES(model="ANN", initial={"level": 10.0}, initial_X=np.array([2.0]))
        model.fit(self.Y, self.X)
        # R: es(data.frame(y=y, x=x), "ANN", initial=list(level=10), initialX=2)
        # estimates the coefficient of x
        assert list(model.initial_value["xreg"].values()) == pytest.approx(
            [2.057433895], rel=1e-6
        )

    def test_a_dict_initial_without_extras_is_kept(self):
        initial = {"level": 20.0, "seasonal": self.SEASON}
        model = ES(model="ANA", lags=[4], initial=initial)
        # A copy, the caller's dict untouched
        assert model.initial is not initial
        assert model.initial["level"] == 20.0
        assert model.initial["seasonal"] is self.SEASON
        assert set(initial) == {"level", "seasonal"}

    def test_no_initial_is_backcasting(self):
        assert ES(model="ANN", initial=None).initial == "backcasting"
