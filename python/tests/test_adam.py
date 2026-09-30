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
        assert hasattr(forecast, "shape")
        assert forecast.shape[0] == 10
        assert "mean" in forecast.columns

    def test_predict_seasonal(self, seasonal_series):
        """Test seasonal prediction."""
        model = ADAM(model="ANA", lags=[12])
        model.fit(seasonal_series)

        forecast = model.predict(h=24)

        assert forecast.shape[0] == 24
        assert not forecast["mean"].isna().any()

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
        assert any("lower" in c for c in cols)
        assert any("upper" in c for c in cols)


class TestADAMModelTypes:
    """Tests for different model types."""

    @pytest.mark.parametrize(
        "model_code",
        [
            "ANN",  # Simple exponential smoothing
            "AAN",  # Holt's linear trend
            "AAdN",  # Damped trend
        ],
    )
    def test_ets_models_nonseasonal(self, simple_series, model_code):
        """Test non-seasonal ETS models."""
        model = ADAM(model=model_code)
        model.fit(simple_series)
        forecast = model.predict(h=5)

        assert forecast.shape[0] == 5
        assert not forecast["mean"].isna().any()

    @pytest.mark.parametrize(
        "model_code",
        [
            "ANA",  # Additive seasonality
            "AAA",  # Trend + seasonality
        ],
    )
    def test_ets_models_seasonal(self, seasonal_series, model_code):
        """Test seasonal ETS models."""
        model = ADAM(model=model_code, lags=[12])
        model.fit(seasonal_series)
        forecast = model.predict(h=12)

        assert forecast.shape[0] == 12
        assert not forecast["mean"].isna().any()

    def test_multiplicative_error(self, multiplicative_series):
        """Test multiplicative error model."""
        model = ADAM(model="MNN")
        model.fit(multiplicative_series)
        forecast = model.predict(h=5)

        assert forecast.shape[0] == 5
        assert not forecast["mean"].isna().any()
        # Multiplicative error model forecasts should stay positive for positive data
        assert (forecast["mean"] > 0).all()


class TestADAMModelSelection:
    """Tests for automatic model selection."""

    def test_model_zzz(self, seasonal_series):
        """Test automatic model selection with ZZZ."""
        model = ADAM(model="ZZZ", lags=[12])
        model.fit(seasonal_series)

        assert model.coef is not None
        forecast = model.predict(h=12)
        assert forecast.shape[0] == 12
        assert not forecast["mean"].isna().any()

    def test_model_zxz(self, seasonal_series):
        """Test automatic selection for error and seasonality (no trend) with ZXZ."""
        model = ADAM(model="ZXZ", lags=[12])
        model.fit(seasonal_series)

        assert model.coef is not None
        forecast = model.predict(h=12)
        assert forecast.shape[0] == 12
        assert not forecast["mean"].isna().any()

    def test_model_fff(self, seasonal_series):
        """Test full model with FFF."""
        model = ADAM(model="FFF", lags=[12])
        model.fit(seasonal_series)

        assert model.coef is not None
        forecast = model.predict(h=12)
        assert forecast.shape[0] == 12
        assert not forecast["mean"].isna().any()

    def test_model_ppp(self, seasonal_series):
        """Test partial automatic selection with PPP."""
        model = ADAM(model="PPP", lags=[12])
        model.fit(seasonal_series)

        assert model.coef is not None
        forecast = model.predict(h=12)
        assert forecast.shape[0] == 12
        assert not forecast["mean"].isna().any()

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

        assert not forecast["mean"].isna().any()

    def test_large_horizon(self, simple_series):
        """Test prediction with large horizon."""
        model = ADAM(model="ANN")
        model.fit(simple_series)
        forecast = model.predict(h=100)

        assert forecast.shape[0] == 100
        assert not forecast["mean"].isna().any()


class TestADAMAttributes:
    """Tests for ADAM attributes after fitting."""

    def test_persistence_level_attribute(self, simple_series):
        """Test that persistence level (alpha) is accessible."""
        model = ADAM(model="ANN")
        model.fit(simple_series)

        # Should have persistence_level_ (alpha parameter)
        assert hasattr(model, "persistence_level_")
        # persistence_level_ may be None for some models, check if numeric
        if model.persistence_level_ is not None:
            assert 0 <= model.persistence_level_ <= 1

    def test_persistence_trend_attribute(self, simple_series):
        """Test that persistence trend (beta) is accessible for trend models."""
        model = ADAM(model="AAN")
        model.fit(simple_series)

        assert hasattr(model, "persistence_trend_")

    def test_phi_attribute(self, simple_series):
        """Test that phi (damping) is accessible for damped models."""
        model = ADAM(model="AAdN")
        model.fit(simple_series)

        assert hasattr(model, "phi_")


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
            f"Expected alpha > 1 for linear series with admissible bounds, got {alpha}"
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
        m1 = ADAM("NNN", lags=[12], ar_order=[1], i_order=[1], ma_order=[1]).fit(
            series60
        )
        m2 = ADAM(
            "NNN", lags=[1, 12], ar_order=[1, 0], i_order=[1, 0], ma_order=[1, 0]
        ).fit(series60)

        assert m1._arima["lags_model_arima"] == m2._arima["lags_model_arima"]
        assert abs(m1.loss_value - m2.loss_value) < 1e-6

    def test_nonseasonal_arima_at_lag1_only(self, series60):
        """ar_order=[1] with lags=[12] gives non-seasonal ARIMA (at lag 1 only)."""
        m = ADAM("NNN", lags=[12], ar_order=[1], i_order=[1], ma_order=[1]).fit(
            series60
        )

        arima = m._arima
        # lag=1 is prepended, which pads the orders to [1, 0]; the seasonal lag
        # then carries no order and the ETS part is non-seasonal, so lag 12 and
        # its zero orders are trimmed away entirely, exactly as R does
        # (adamGeneral.R:487-493, where m$lags is 1 and m$orders$ar is 1).
        assert arima["ar_orders"] == [1]
        assert arima["ma_orders"] == [1]

    def test_seasonal_only_arima(self, series60):
        """ar_order=[0,1] with lags=[1,12] gives seasonal-only ARIMA at lag=12."""
        m = ADAM(
            "NNN", lags=[1, 12], ar_order=[0, 1], i_order=[0, 1], ma_order=[0, 1]
        ).fit(series60)

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
        m = ADAM(
            "NNN", ar_order=1, i_order=1, ma_order=1, arma={"ar": [0.5], "ma": [0.2]}
        )
        m.fit(arima_series)
        assert m._arima["arma_parameters"] == pytest.approx([0.5, 0.2])
        assert len(m.coef) == 0

    def test_arma_fixed_produces_different_fitted(self, arima_series):
        """Different fixed MA values produce different fitted values."""
        m03 = ADAM("NNN", ar_order=0, i_order=1, ma_order=1, arma={"ma": [0.3]}).fit(
            arima_series
        )
        m08 = ADAM("NNN", ar_order=0, i_order=1, ma_order=1, arma={"ma": [0.8]}).fit(
            arima_series
        )
        assert not np.allclose(m03.fitted, m08.fitted)

    def test_arma_with_constant(self, arima_series):
        """Fixed arma with constant=True fits without error."""
        m = ADAM(
            "NNN",
            ar_order=1,
            i_order=1,
            ma_order=1,
            arma={"ar": [0.5], "ma": [0.2]},
            constant=True,
        )
        m.fit(arima_series)
        assert np.isfinite(m.constant_value)

    def test_arma_sarima(self, long_series):
        """Fixed arma on SARIMA fits without error."""
        m = ADAM(
            "NNN",
            lags=[1, 12],
            ar_order=[1, 0],
            i_order=[1, 1],
            ma_order=[1, 1],
            arma={"ma": [0.3, 0.2]},
        )
        m.fit(long_series)
        assert m._arima["arma_parameters"] is not None
        assert np.all(np.isfinite(m.fitted))

    def test_msarima_arma_fixed(self, arima_series):
        """MSARIMA with fixed arma stores fixed values in arma_parameters."""
        from smooth import MSARIMA

        m = MSARIMA(ar_order=1, i_order=1, ma_order=1, arma={"ar": [0.5], "ma": [0.2]})
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
            forecast1["mean"].values, forecast2["mean"].values
        )


class TestADAMLogLikDistribution:
    """logLik under non-likelihood losses uses the loss-implied distribution."""

    def _series(self):
        rng = np.random.default_rng(41)
        trend = 100 + np.cumsum(rng.normal(0, 2, 72))
        seas = np.tile([5, -3, 2, -4, 6, -6, 3, -1, 2, -2, 4, -6], 6)
        return trend + seas

    def test_loglik_is_implied_distribution_not_minus_loss(self):
        # For MSE/MAE the reported logLik must be the concentrated likelihood
        # under the loss-implied distribution (MSE<->dnorm, MAE<->dlaplace),
        # NOT -lossValue. loss_value stays the raw loss.
        from scipy import stats

        y = self._series()
        m_mse = ADAM(model="AAA", lags=[12], loss="MSE").fit(y)
        r = np.asarray(m_mse.residuals).ravel()
        n = len(r)
        sd = np.sqrt(np.sum(r**2) / n)
        dnorm_ll = float(np.sum(stats.norm.logpdf(r, 0, sd)))
        assert abs(float(m_mse.loglik) - dnorm_ll) < 1e-6
        assert float(m_mse.loss_value) < 100  # the MSE, not the likelihood

        m_mae = ADAM(model="AAA", lags=[12], loss="MAE").fit(y)
        r = np.asarray(m_mae.residuals).ravel()
        b = np.sum(np.abs(r)) / len(r)
        dlap_ll = float(np.sum(stats.laplace.logpdf(r, 0, b)))
        assert abs(float(m_mae.loglik) - dlap_ll) < 1e-6

    def test_explicit_distribution_is_honoured_over_the_loss(self):
        # An explicit distribution is honoured for the reported logLik, even
        # when the fitting loss implies a different one (mirrors R): loss=MSE
        # with distribution="dlaplace" reports the Laplace likelihood, not the
        # MSE-implied Normal. Only distribution="default" follows the loss.
        from scipy import stats

        y = self._series()
        m_default = ADAM(model="AAA", lags=[12], loss="MSE").fit(y)
        m_lap = ADAM(model="AAA", lags=[12], loss="MSE", distribution="dlaplace").fit(y)
        # The two differ: default is Normal, explicit is Laplace.
        assert abs(float(m_default.loglik) - float(m_lap.loglik)) > 1.0
        r = np.asarray(m_lap.residuals).ravel()
        b = np.sum(np.abs(r)) / len(r)
        dlap_ll = float(np.sum(stats.laplace.logpdf(r, 0, b)))
        assert abs(float(m_lap.loglik) - dlap_ll) < 1e-6

    def test_ds_distribution_is_the_s_not_students_t(self):
        # The S distribution log-density is -log(4 s^2) - sqrt(|x-mu|)/s, not a
        # Student's-t. HAM routes through it.
        y = self._series()
        m = ADAM(model="AAA", lags=[12], loss="likelihood", distribution="ds").fit(y)
        r = np.asarray(m.residuals).ravel()
        s = np.sum(np.sqrt(np.abs(r))) / (len(r) * 2)
        s_ll = float(np.sum(-np.log(4 * s**2) - np.sqrt(np.abs(r)) / s))
        assert abs(float(m.loglik) - s_ll) < 1e-6


class TestADAMPointLik:
    """point_lik() mirrors R's pointLik.adam: per-observation log-likelihood
    that sums to loglik. R-parity is covered in the comparison suite; here we
    check the self-consistency invariant across distributions."""

    def _series(self):
        np.random.seed(3)
        return 100.0 + np.cumsum(np.random.randn(120)) + np.arange(120) * 0.2

    @pytest.mark.parametrize(
        "model,dist",
        [
            ("AAN", "dnorm"),
            ("AAN", "dlaplace"),
            ("MNN", "dgamma"),
            ("MNN", "dlnorm"),
        ],
    )
    def test_point_lik_sums_to_loglik(self, model, dist):
        y = np.abs(self._series())
        m = ADAM(model=model, lags=[1], distribution=dist, initial="optimal").fit(y)
        pl = np.asarray(m.point_lik()).ravel()
        assert pl.shape == (m.nobs,)
        assert np.isclose(np.sum(pl), float(m.loglik))
        assert np.allclose(m.point_lik(log=False), np.exp(pl))


class TestADAMvcovType:
    """vcov type= parameter: opg (default) is PSD and distinct from hessian;
    bootstrap= is deprecated."""

    def _series(self):
        np.random.seed(5)
        return 100.0 + np.cumsum(np.random.randn(120)) + np.arange(120) * 0.3

    def test_opg_is_default_and_psd(self):
        m = ADAM(model="AAN", lags=[1], initial="optimal").fit(self._series())
        v_default = m.vcov()
        v_opg = m.vcov(type="opg")
        np.testing.assert_allclose(v_default.values, v_opg.values)
        sym = (v_opg.values + v_opg.values.T) / 2
        assert np.all(np.linalg.eigvalsh(sym) > -1e-6)

    def test_opg_differs_from_hessian(self):
        m = ADAM(model="AAN", lags=[1], initial="optimal").fit(self._series())
        v_opg = m.vcov(type="opg").values
        v_h = m.vcov(type="hessian").values
        assert not np.allclose(v_opg, v_h, atol=1e-6)

    def test_bootstrap_true_deprecated(self):
        import warnings

        m = ADAM(model="ANN", lags=[1], initial="optimal").fit(self._series())
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m.vcov(bootstrap=True, nsim=20)
            assert any(issubclass(x.category, DeprecationWarning) for x in w)

    def test_invalid_type_raises(self):
        m = ADAM(model="ANN", lags=[1], initial="optimal").fit(self._series())
        with pytest.raises(ValueError, match="type must be one of"):
            m.vcov(type="nonsense")


class TestADAMPersistenceNames:
    """R accepts the Greek names as aliases of the component names."""

    @staticmethod
    def _series():
        rng = np.random.default_rng(5)
        t = np.arange(96)
        return (
            100 + 0.5 * t + 8 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1.5, t.size)
        )

    def test_greek_names_match_component_names(self):
        y = self._series()
        greek = ADAM(
            model="AAdA",
            lags=[1, 12],
            persistence={"alpha": 0.3, "beta": 0.1, "gamma": 0.2},
            phi=0.9,
            initial="backcasting",
        ).fit(y)
        named = ADAM(
            model="AAdA",
            lags=[1, 12],
            persistence={"level": 0.3, "trend": 0.1, "seasonal": 0.2},
            phi=0.9,
            initial="backcasting",
        ).fit(y)
        assert greek.loglik == named.loglik

    def test_component_name_wins_over_the_alias(self):
        y = self._series()
        both = ADAM(
            model="ANN",
            lags=[1],
            persistence={"level": 0.3, "alpha": 0.9},
            initial="backcasting",
        ).fit(y)
        only = ADAM(
            model="ANN", lags=[1], persistence={"level": 0.3}, initial="backcasting"
        ).fit(y)
        assert both.loglik == only.loglik


class TestADAMARIMAInitialiser:
    """Hannan-Rissanen starting values of the ARMA parameters."""

    @pytest.fixture
    def air(self):
        """The full AirPassengers series."""
        import pathlib

        import pandas as pd

        path = pathlib.Path(__file__).parent / "data" / "ces_airpassengers.csv"
        return pd.read_csv(path)["y"].values.astype(float)

    @staticmethod
    def _hr(y, ar, ma, lags, ar_est=True, ma_est=True, arma=(), use=None, bounds=True):
        from smooth.adam_general import _ols

        lags = np.asarray(lags, dtype=np.uint64)
        return _ols.arima_hr(
            np.asarray(y, dtype=float),
            np.asarray(ar, dtype=np.uint64),
            np.asarray(ma, dtype=np.uint64),
            lags,
            ar_est,
            ma_est,
            np.asarray(arma, dtype=float),
            np.ones_like(lags) if use is None else np.asarray(use, dtype=np.uint64),
            bounds,
        )

    @staticmethod
    def _arma(n, ar, ma, seed):
        from scipy.signal import lfilter

        e = np.random.default_rng(seed).normal(size=n + 200)
        return lfilter([1, ma], [1, -ar], e)[200:]

    def test_recovers_arma(self):
        """HR lands next to the true ARMA(1,1) parameters."""
        b = self._hr(self._arma(1000, 0.6, 0.3, 41), [1], [1], [1])
        np.testing.assert_allclose(b, [0.6, 0.3], atol=0.1)

    def test_feasible_only_when_rejected(self):
        """Values move inside the boundary only if the cost function rejects them."""
        e = np.random.default_rng(45).normal(size=200)
        y = np.zeros(200)
        for t in range(1, 200):
            y[t] = 1.03 * y[t - 1] + e[t]
        ar_raw = self._hr(y, [1], [0], [1], ma_est=False, bounds=False)
        assert ar_raw[0] > 1
        assert self._hr(y, [1], [0], [1], ma_est=False)[0] == pytest.approx(0.99)

        # A non-invertible HR estimate is reflected to the invertible MA
        w = np.diff(np.random.default_rng(47).normal(size=300), n=2)
        ma = [self._hr(w, [0], [1], [1], ar_est=False, bounds=b)[0] for b in (False, True)]
        assert abs(ma[0]) < 1
        assert ma[1] == ma[0]

    def test_defaults_and_provided(self):
        """Too few seasons or a switched-off level keep the defaults."""
        y = self._arma(300, 0.6, 0.3, 41)
        short = self._hr(y[:30], [1, 1], [1, 1], [1, 12])
        np.testing.assert_array_equal(short[2:], [0.1, -0.1])
        off = self._hr(y, [1, 1], [1, 1], [1, 12], use=[1, 0])
        np.testing.assert_array_equal(off[2:], [0.1, -0.1])
        assert len(self._hr(y, [1], [1], [1], ar_est=False, arma=[0.6])) == 1

    def test_regression_residuals(self):
        """HR runs on the residuals of the regression, not on the series."""
        from scipy.signal import lfilter

        rng = np.random.default_rng(48)
        x = rng.normal(10, 5, 200)
        y = 100 + 5 * x + lfilter([1], [1, -0.6], rng.normal(size=400))[200:]
        start = ADAM(
            model="NNN",
            orders={"ar": [1]},
            constant=True,
            nlopt_kwargs={"maxeval": 1},
        ).fit(y, X=x.reshape(-1, 1))
        assert start.coef[0] == pytest.approx(0.6, abs=0.1)

    @pytest.mark.r_parity
    def test_regression_on_differences_matches_r(self):
        """With differences, HR runs on the regression of the differences, as in R."""
        from ._r_bridge import r_array

        rng = np.random.default_rng(49)
        # Short: the bridge passes the data on Rscript's command line
        x = np.cumsum(rng.normal(size=120))
        y = 100 + 3 * x + np.cumsum(self._arma(120, 0.0, 0.5, 49))
        start = ADAM(
            model="NNN",
            orders={"ar": [0], "i": [1], "ma": [1]},
            nlopt_kwargs={"maxeval": 1},
        ).fit(y, X=x.reshape(-1, 1))
        expected = r_array(
            "{m <- adam(data.frame(y=y, x=x), 'NNN', orders=list(i=1, ma=1),"
            " maxeval=1); unname(m$B['theta1[1]'])}",
            R_data={"y": y, "x": x},
        )
        np.testing.assert_allclose(start.coef[:1], expected, rtol=1e-12)

    def test_constant_is_intercept(self):
        """The constant starts consistent with AR and the fit finds the mean."""
        y = 100 + self._arma(200, 0.6, 0.3, 41)
        model = ADAM(
            model="NNN",
            orders={"ar": [1], "i": [0], "ma": [1]},
            lags=[1],
            constant=True,
        ).fit(y)
        phi, constant = model.coef[0], model.coef[-1]
        assert constant / (1 - phi) == pytest.approx(np.mean(y), rel=0.01)

    @pytest.mark.parametrize(
        "kwargs",
        [{"distribution": d} for d in ("dlaplace", "ds", "dgnorm", "dgamma", "dlnorm")]
        + [{"loss": loss} for loss in ("MAE", "HAM", "MSEh", "TMSE", "GTMSE", "GPL")],
    )
    def test_non_normal_and_losses(self, air, kwargs):
        """The fit improves on the starting point under any distribution and loss."""
        args = dict(
            model="NNN",
            orders={"ar": [1, 1], "i": [1, 1], "ma": [1, 1]},
            lags=[1, 12],
            h=12,
            **kwargs,
        )
        start = ADAM(**args, nlopt_kwargs={"maxeval": 1}).fit(air)
        model = ADAM(**args).fit(air)
        assert np.all(np.isfinite(start.coef))
        assert model._adam_estimated["CF_value"] <= start._adam_estimated["CF_value"]

    @pytest.mark.r_parity
    @pytest.mark.parametrize(
        "model, orders, lags, distribution",
        [
            ("NNN", {"ar": [1, 1], "i": [1, 1], "ma": [1, 1]}, [1, 12], "dnorm"),
            ("NNN", {"ar": [1, 1], "i": [1, 1], "ma": [1, 1]}, [1, 12], "dgamma"),
            ("MAM", {"ar": [1, 1], "i": [0, 0], "ma": [1, 1]}, [1, 12], "dgamma"),
            ("AAN", {"ar": [1], "i": [0], "ma": [1]}, [1], "dnorm"),
        ],
    )
    def test_starting_values_match_r(self, air, model, orders, lags, distribution):
        """The ARMA starting values are R's, bit for bit up to the JSON round-trip."""
        from ._r_bridge import r_array, r_to_literal

        start = ADAM(
            model=model,
            orders=orders,
            lags=lags,
            distribution=distribution,
            nlopt_kwargs={"maxeval": 1},
        ).fit(air)
        n_arma = sum(orders["ar"]) + sum(orders["ma"])
        orders_r = ",".join(f"{k}={r_to_literal(v)}" for k, v in orders.items())
        expected = r_array(
            f"{{m <- adam(ts(y, frequency=12), '{model}', orders=list({orders_r}),"
            f" lags={r_to_literal(lags)}, distribution='{distribution}', maxeval=1);"
            " unname(m$B[grepl('phi|theta', names(m$B))])}",
            R_data={"y": air},
        )
        k = len(start.coef) - n_arma
        np.testing.assert_allclose(start.coef[k:], expected, rtol=1e-12)


class TestARIMABounds:
    """Stationary AR and invertible MA, factor by factor (src/headers/arimaBounds.h)."""

    def test_reflection_matches_the_roots(self):
        """The largest reflection coefficient is below one exactly when all roots are outside."""
        from smooth.adam_general import _adamCore
        from smooth.adam_general.core.utils.polynomials import adam_polynomialiser

        core = _adamCore.adamCore(
            lags=np.ones(1, dtype=np.uint64),
            E="A",
            T="N",
            S="N",
            nNonSeasonal=0,
            nSeasonal=0,
            nETS=0,
            nArima=0,
            nXreg=0,
            nComponents=0,
            constant=False,
            adamETS=False,
        )
        rng = np.random.default_rng(3)
        for _ in range(200):
            theta = rng.uniform(-2, 2, 3)
            polys = adam_polynomialiser(
                core, theta, [0, 0], [0, 0], [1, 2], False, True, [], [1, 12]
            )
            invertible = np.all(np.abs(np.roots([theta[2], theta[1], 1])) > 1) and (
                abs(theta[0]) < 1
            )
            assert (polys["ma_reflection"] < 1) == invertible

    @pytest.mark.r_parity
    def test_bounds_match_r(self):
        """The ARIMA penalty accepts and rejects the same seasonal MA / AR as R."""
        import pathlib

        import pandas as pd

        from smooth.adam_general import _adamCore
        from smooth.adam_general.core.utils.polynomials import (
            adam_polynomialiser,
            arima_bounds_penalty,
        )

        from ._r_bridge import r_dict

        path = pathlib.Path(__file__).parent / "data" / "ces_airpassengers.csv"
        y = np.log(pd.read_csv(path)["y"].values.astype(float))
        core = _adamCore.adamCore(
            lags=np.ones(1, dtype=np.uint64),
            E="A",
            T="N",
            S="N",
            nNonSeasonal=0,
            nSeasonal=0,
            nETS=0,
            nArima=0,
            nXreg=0,
            nComponents=0,
            constant=False,
            adamETS=False,
        )
        cases = [
            ({"ar": [0, 0], "i": [1, 1], "ma": [0, 2]}, [1.058, 0.793]),
            ({"ar": [0, 0], "i": [1, 1], "ma": [0, 2]}, [0.3, -1.2]),
            ({"ar": [0, 2], "i": [1, 0], "ma": [0, 0]}, [1.2, -0.3]),
            ({"ar": [0, 2], "i": [1, 0], "ma": [0, 0]}, [0.3, 0.8]),
        ]
        for orders, values in cases:
            arima = {
                "arima_model": True,
                "ar_estimate": sum(orders["ar"]) > 0,
                "ma_estimate": sum(orders["ma"]) > 0,
            }
            polys = adam_polynomialiser(
                core,
                values,
                orders["ar"],
                orders["i"],
                orders["ma"],
                arima["ar_estimate"],
                arima["ma_estimate"],
                [],
                [1, 12],
            )
            orders_r = ",".join(
                f"{k}=c({','.join(map(str, v))})" for k, v in orders.items()
            )
            expected = r_dict(
                f"{{m <- adam(ts(y, frequency=12), 'NNN', orders=list({orders_r}),"
                f" lags=c(1,12), maxeval=1, B=c({','.join(map(str, values))}));"
                " list(loss=m$lossValue)}",
                R_data={"y": y},
            )
            assert (arima_bounds_penalty(arima, polys) == 0) == (
                expected["loss"][0] < 1e100
            )


    def test_parameter_bounds_match_the_roots(self):
        """The bounds of a parameter within its factor are those of the roots."""
        from smooth.adam_general import _ols

        values = np.array([0.1, 0.2, 0.3])
        grid = np.arange(-3, 3, 0.001)
        stable = [
            np.all(np.abs(np.roots([-0.3, -v, -0.1, 1])) > 1) for v in grid
        ]
        np.testing.assert_allclose(
            _ols.arima_parameter_bounds(values, 1, -1.0),
            [grid[stable].min(), grid[stable].max()],
            atol=2e-3,
        )
        assert np.all(np.isnan(_ols.arima_parameter_bounds(np.array([1.2]), 0, 1.0)))

    @pytest.mark.r_parity
    def test_confint_matches_r(self):
        """The ARMA confidence intervals are clamped to the same region as in R."""
        import pathlib

        import pandas as pd

        from ._r_bridge import r_array

        path = pathlib.Path(__file__).parent / "data" / "ces_airpassengers.csv"
        y = np.log(pd.read_csv(path)["y"].values.astype(float))
        orders = {"ar": [2, 1], "i": [1, 1], "ma": [1, 1]}
        model = ADAM(model="NNN", orders=orders, lags=[1, 12]).fit(y)
        expected = r_array(
            "{m <- adam(ts(y, frequency=12), 'NNN', lags=c(1,12),"
            " orders=list(ar=c(2,1), i=c(1,1), ma=c(1,1))); unname(confint(m))}",
            R_data={"y": y},
        )
        np.testing.assert_allclose(model.confint().values, expected, rtol=1e-6)

class TestADAMARIMAStates:
    """ARIMA initials: the initial state of the companion form, as in ssarima."""

    @pytest.fixture
    def air(self):
        """The full AirPassengers series."""
        import pathlib

        import pandas as pd

        path = pathlib.Path(__file__).parent / "data" / "ces_airpassengers.csv"
        return pd.read_csv(path)["y"].values.astype(float)

    def test_initials_sum_the_ari_states_by_time(self):
        """Initial k sums ari_L * x[m-L+k] over the ARI lags L >= k."""
        from smooth.adam_general.core.utils.polynomials import arima_initials

        # (1 - B)(1 - B^12): ARI lags 1, 12 and 13
        ari = np.zeros(14)
        ari[[0, 1, 12, 13]] = [1, -1, -1, 1]
        x = np.arange(1.0, 14.0)
        expected = x.copy()  # lag 13: x[k-1]
        expected[:12] -= x[1:13]  # lag 12: -x[k]
        expected[0] -= x[12]  # lag 1: -x[12]
        np.testing.assert_array_equal(arima_initials(ari, x, "A"), expected)

    def test_head_initials_read_the_fitted_heads_by_time(self):
        """The state with lag L gives the time k its fitted column max-L+k."""
        from smooth.adam_general.core.utils.polynomials import arima_head_initials

        head = np.arange(1.0, 13.0).reshape(3, 4)
        # Lags 1, 2 and 4: time 1 <- [0,3] + [1,2] + [2,0]; time 2 <- [1,3] + [2,1]
        expected = [4 + 7 + 9, 8 + 10, 11, 12]
        np.testing.assert_array_equal(
            arima_head_initials(head, [1, 2, 4], 4, "A"), expected
        )

    def test_every_initial_enters_the_fit(self, air):
        """The MA-only states have initials too, each changing the loss."""
        orders = {"ar": [0, 0], "i": [1, 0], "ma": [0, 1]}
        model = ADAM(model="NNN", orders=orders, lags=[1, 12], initial="optimal").fit(
            np.log(air)
        )
        names = [n for n in model.coef_names if n.startswith("ARIMAState")]
        assert len(names) == 12
        assert np.all(np.asarray(model.states)[:-1, :12] == 0)

    @pytest.mark.r_parity
    @pytest.mark.parametrize(
        "model, orders, lags, extra, log_data",
        [
            ("NNN", {"ar": [1, 1], "i": [1, 1], "ma": [1, 1]}, [1, 12], "", True),
            ("NNN", {"ar": [0, 2], "i": [0, 1], "ma": [0, 0]}, [1, 12], "", True),
            ("NNN", {"ar": [1, 1], "i": [1, 1], "ma": [1, 1]}, [1, 12], "dg", False),
            ("NNN", {"ar": [1, 1], "i": [0, 0], "ma": [0, 0]}, [1, 12], "c", True),
            ("NNN", {"ar": [0, 0], "i": [1, 1], "ma": [1, 1]}, [1, 12], "c", True),
            ("NNN", {"ar": [0, 0], "i": [1, 0], "ma": [0, 1]}, [1, 12], "", True),
            ("MAM", {"ar": [1, 1], "i": [0, 0], "ma": [1, 1]}, [1, 12], "dg", False),
            ("ANA", {"ar": [1, 1], "i": [0, 0], "ma": [1, 1]}, [1, 12], "", False),
        ],
    )
    def test_optimal_arima_matches_r(self, air, model, orders, lags, extra, log_data):
        """Starting B, loss and fitted values of optimal ARIMA agree with R."""
        from ._r_bridge import r_dict, r_to_literal

        y = np.log(air) if log_data else air
        distribution = "dgamma" if extra == "dg" else "dnorm"
        constant = extra == "c"
        args = dict(
            model=model,
            orders=orders,
            lags=lags,
            distribution=distribution,
            constant=constant,
            initial="optimal",
        )
        start = ADAM(**args, nlopt_kwargs={"maxeval": 1}).fit(y)
        fitted = ADAM(**args).fit(y)
        orders_r = ",".join(f"{k}={r_to_literal(v)}" for k, v in orders.items())
        call = (
            f"adam(ts(y, frequency=12), '{model}', orders=list({orders_r}),"
            f" lags={r_to_literal(lags)}, distribution='{distribution}',"
            f" constant={'TRUE' if constant else 'FALSE'}, initial='optimal'"
        )
        expected = r_dict(
            f"{{m0 <- {call}, maxeval=1); m <- {call});"
            " list(B=unname(m0$B), loss=m$lossValue, fitted=as.vector(fitted(m)))}",
            R_data={"y": y},
        )
        np.testing.assert_allclose(start.coef, expected["B"], rtol=1e-8, atol=1e-10)
        assert fitted._adam_estimated["CF_value"] == pytest.approx(
            expected["loss"][0], rel=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(fitted.fitted).ravel(), expected["fitted"], rtol=1e-5
        )

    @pytest.mark.r_parity
    @pytest.mark.parametrize(
        "orders, initial",
        [
            ({"ar": [1, 0], "i": [1, 0], "ma": [1, 2]}, "backcasting"),
            ({"ar": [1, 1], "i": [1, 1], "ma": [1, 1]}, "backcasting"),
            ({"ar": [2, 0], "i": [1, 1], "ma": [0, 1]}, "complete"),
        ],
    )
    def test_backcasted_arima_matches_r(self, air, orders, initial):
        """The seed of the backcasting does not drift between the CF calls."""
        from ._r_bridge import r_dict, r_to_literal

        model = ADAM(model="NNN", orders=orders, lags=[1, 12], initial=initial).fit(
            np.log(air)
        )
        orders_r = ",".join(f"{k}={r_to_literal(v)}" for k, v in orders.items())
        expected = r_dict(
            f"{{m <- adam(ts(y, frequency=12), 'NNN', orders=list({orders_r}),"
            f" lags=c(1,12), initial='{initial}');"
            " list(B=unname(m$B), loss=m$lossValue)}",
            R_data={"y": np.log(air)},
        )
        assert model._adam_estimated["CF_value"] == pytest.approx(
            expected["loss"][0], rel=1e-6
        )
        np.testing.assert_allclose(model.coef, expected["B"], rtol=1e-4, atol=1e-6)

    @pytest.mark.r_parity
    def test_two_stage_arima_matches_r(self, air):
        """Two-stage passes the backcasted initials on as R does."""
        from ._r_bridge import r_dict

        orders = {"ar": [1, 1], "i": [1, 1], "ma": [1, 1]}
        model = ADAM(model="NNN", orders=orders, lags=[1, 12], initial="two-stage").fit(
            np.log(air)
        )
        expected = r_dict(
            "{m <- adam(ts(y, frequency=12), 'NNN', orders=list(ar=c(1,1), i=c(1,1),"
            " ma=c(1,1)), lags=c(1,12), initial='two-stage');"
            " list(loss=m$lossValue, arima=unname(m$initial$arima))}",
            R_data={"y": np.log(air)},
        )
        assert model._adam_estimated["CF_value"] == pytest.approx(
            expected["loss"][0], rel=1e-6
        )
        arima_initials = [
            value
            for name, value in zip(model.coef_names, model.coef)
            if name.startswith("ARIMAState")
        ]
        np.testing.assert_allclose(arima_initials, expected["arima"], rtol=1e-5)
