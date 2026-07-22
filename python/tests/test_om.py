"""General-functionality tests for the OM (Occurrence Model) class.

Covers init/fit/predict for the four occurrence types supported in Stage 1
(``fixed``, ``odds-ratio``, ``inverse-odds-ratio``, ``direct``) plus
``orders``, ``X`` regressors, holdout and the property surface.

R-vs-Python exact-match tests live in ``test_om_r_comparison.py``.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from smooth import OM

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def intermittent_y():
    """Deterministic intermittent demand series, n=200."""
    rng = np.random.default_rng(41)
    return rng.poisson(0.3, 200).astype(float)


@pytest.fixture
def binary_y():
    rng = np.random.default_rng(0)
    return rng.binomial(1, 0.6, 200).astype(float)


@pytest.fixture
def regressors_X():  # noqa: N802
    rng = np.random.default_rng(7)
    return rng.standard_normal((200, 2))


# --------------------------------------------------------------------------
# Initialisation
# --------------------------------------------------------------------------


class TestInit:
    def test_default_init_does_not_raise(self):
        OM()

    @pytest.mark.parametrize(
        "occ", ["fixed", "odds-ratio", "inverse-odds-ratio", "direct"]
    )
    def test_init_each_occurrence(self, occ):
        m = OM(occurrence=occ)
        assert m.occurrence == occ
        assert m.occurrence_char in ("f", "o", "i", "d")

    def test_init_auto_returns_autoOM_instance(self):
        from smooth import AutoOM

        m = OM(occurrence="auto")
        assert isinstance(m, AutoOM)

    def test_auto_fit_returns_om_or_omg(self, intermittent_y):
        from smooth import OMG, AutoOM

        m = OM(model="MNN", occurrence="auto", lags=[1]).fit(intermittent_y)
        assert isinstance(m, (OM, OMG))
        assert not isinstance(m, AutoOM)

    def test_init_general_returns_omg(self):
        from smooth import OMG

        m = OM(occurrence="general")
        assert isinstance(m, OMG)

    def test_init_rejects_invalid(self):
        with pytest.raises(ValueError, match="Invalid occurrence"):
            OM(occurrence="silly")

    def test_init_rejects_invalid_loss(self):
        with pytest.raises(ValueError, match="Invalid loss"):
            OM(loss="not-a-real-loss")


# --------------------------------------------------------------------------
# Loss menu (mirrors R `om()`'s full single-step menu + LASSO/RIDGE + custom)
# --------------------------------------------------------------------------


class TestLossMenu:
    @pytest.mark.parametrize("loss", ["likelihood", "MSE", "MAE", "HAM"])
    def test_string_losses_fit(self, intermittent_y, loss):
        m = OM(model="MNN", occurrence="odds-ratio", loss=loss).fit(intermittent_y)
        assert m.loss_ == loss
        assert np.isfinite(m.loss_value)

    @pytest.mark.parametrize("loss", ["LASSO", "RIDGE"])
    def test_regularised_losses_fit(self, intermittent_y, loss):
        m = OM(model="MNN", occurrence="odds-ratio", loss=loss, reg_lambda=0.3).fit(
            intermittent_y
        )
        assert m.loss_ == loss
        assert np.isfinite(m.loss_value)
        assert m.reg_lambda == 0.3

    def test_custom_callable_loss(self, intermittent_y):
        def cube_abs(actual, fitted, B):  # noqa: N803
            import numpy as _np

            return float(_np.sum(_np.abs(actual - fitted) ** 3))

        m = OM(model="MNN", occurrence="odds-ratio", loss=cube_abs).fit(intermittent_y)
        assert m.loss_ == "custom"
        assert np.isfinite(m.loss_value)


# --------------------------------------------------------------------------
# Fit + predict basic flow
# --------------------------------------------------------------------------


class TestFitBasic:
    @pytest.mark.parametrize(
        "occ", ["fixed", "odds-ratio", "inverse-odds-ratio", "direct"]
    )
    def test_fit_returns_self(self, intermittent_y, occ):
        model = "ANN" if occ == "fixed" else "MNN"
        m = OM(model=model, occurrence=occ, lags=[1])
        out = m.fit(intermittent_y)
        assert out is m

    @pytest.mark.parametrize(
        "occ", ["fixed", "odds-ratio", "inverse-odds-ratio", "direct"]
    )
    def test_fitted_in_unit_interval(self, intermittent_y, occ):
        model = "ANN" if occ == "fixed" else "MNN"
        m = OM(model=model, occurrence=occ, lags=[1])
        m.fit(intermittent_y)
        f = m.fitted
        assert f.shape == intermittent_y.shape
        assert np.all(f >= 0.0 - 1e-12)
        assert np.all(f <= 1.0 + 1e-12)

    def test_fixed_is_constant_and_alpha_zero(self, intermittent_y):
        m = OM(occurrence="fixed").fit(intermittent_y)
        f = m.fitted
        # All fitted values are exactly mean(ot)
        target = float(np.mean(intermittent_y != 0))
        assert np.allclose(f, target, atol=1e-12)
        assert m.persistence_vector["alpha"] == 0.0

    def test_residuals_equal_ot_minus_fitted(self, intermittent_y):
        m = OM(model="MNN", occurrence="odds-ratio", lags=[1]).fit(intermittent_y)
        ot = (intermittent_y != 0).astype(float)
        np.testing.assert_allclose(m.residuals, ot - m.fitted, atol=1e-12)

    def test_actuals_is_binary(self, intermittent_y):
        m = OM(model="MNN", occurrence="odds-ratio", lags=[1]).fit(intermittent_y)
        a = m.actuals
        assert set(np.unique(a)).issubset({0.0, 1.0})


# --------------------------------------------------------------------------
# Properties
# --------------------------------------------------------------------------


class TestProperties:
    @pytest.fixture
    def fitted_model(self, intermittent_y):
        return OM(model="MNN", occurrence="odds-ratio", lags=[1]).fit(intermittent_y)

    def test_distribution_is_plogis(self, fitted_model):
        assert fitted_model.distribution_ == "plogis"

    def test_scale_matches_link_residual_std(self, fitted_model):
        """OM.scale / sigma == sqrt(mean(residuals²)) — mirrors R's
        ``sigma.om`` (R/om.R) so multi-step covariances on the link
        scale are well-defined."""
        expected = float(np.sqrt(np.mean(np.asarray(fitted_model.residuals) ** 2)))
        assert np.isfinite(fitted_model.scale)
        assert np.isfinite(fitted_model.sigma)
        np.testing.assert_allclose(fitted_model.scale, expected)
        np.testing.assert_allclose(fitted_model.sigma, expected)

    def test_loss_is_likelihood(self, fitted_model):
        assert fitted_model.loss_ == "likelihood"

    @pytest.mark.parametrize(
        "occ,letter",
        [
            ("fixed", "F"),
            ("odds-ratio", "O"),
            ("inverse-odds-ratio", "I"),
            ("direct", "D"),
        ],
    )
    def test_model_name_format(self, intermittent_y, occ, letter):
        model = "ANN" if occ == "fixed" else "MNN"
        m = OM(model=model, occurrence=occ, lags=[1]).fit(intermittent_y)
        assert re.match(rf"^oETS\([AM][NAM]d?[NAM]\)\[{letter}\]$", m.model_name), (
            f"got {m.model_name!r}"
        )

    def test_property_accessors_do_not_raise(self, fitted_model):
        # Touch every property in turn; none should throw
        _ = (
            fitted_model.coef,
            fitted_model.b_value,
            fitted_model.fitted,
            fitted_model.residuals,
            fitted_model.actuals,
            fitted_model.data,
            fitted_model.states,
            fitted_model.persistence_vector,
            fitted_model.transition,
            fitted_model.measurement,
            fitted_model.initial_value,
            fitted_model.initial_type,
            fitted_model.loss_value,
            fitted_model.loglik,
            fitted_model.aic,
            fitted_model.aicc,
            fitted_model.bic,
            fitted_model.bicc,
            fitted_model.nobs,
            fitted_model.nparam,
            fitted_model.error_type,
            fitted_model.model_type,
            fitted_model.model_name,
            fitted_model.lags_used,
            fitted_model.orders,
            fitted_model.time_elapsed,
        )

    def test_point_lik_sums_to_loglik(self, fitted_model):
        pl = fitted_model.point_lik()
        assert pl.shape == (fitted_model.nobs,)
        assert np.isclose(np.sum(pl), fitted_model.loglik)

    def test_point_lik_is_bernoulli(self, binary_y):
        m = OM(model="ANN", occurrence="odds-ratio", initial="optimal").fit(binary_y)
        ot = np.asarray(m._observations["ot"], dtype=float).ravel()
        p = np.asarray(m.fitted, dtype=float).ravel()
        expected = np.where(ot == 1, np.log(p), np.log(1.0 - p))
        assert np.allclose(m.point_lik(), expected)
        assert np.allclose(m.point_lik(log=False), np.exp(expected))


# --------------------------------------------------------------------------
# Predict
# --------------------------------------------------------------------------


class TestPredict:
    def test_predict_shape_and_bounds(self, intermittent_y):
        m = OM(model="MNN", occurrence="odds-ratio", lags=[1]).fit(intermittent_y)
        fc = m.predict(h=10)
        assert fc.mean.shape == (10,)
        assert np.all(fc.mean.values >= 0.0 - 1e-12)
        assert np.all(fc.mean.values <= 1.0 + 1e-12)

    def test_predict_warns_on_interval(self, intermittent_y):
        m = OM(model="MNN", occurrence="odds-ratio", lags=[1]).fit(intermittent_y)
        with pytest.warns(UserWarning, match="Intervals on the probability scale"):
            m.predict(h=5, interval="prediction")

    def test_holdout_split(self, intermittent_y):
        m = OM(model="MNN", occurrence="odds-ratio", lags=[1], h=10, holdout=True)
        m.fit(intermittent_y)
        assert m.holdout_data is not None
        assert m.holdout_data.shape == (10,)
        assert m.nobs == 200 - 10
        # Auto-forecast is populated
        assert m._auto_forecast is not None
        assert m._auto_forecast.mean.shape == (10,)


# --------------------------------------------------------------------------
# Explanatory regressors
# --------------------------------------------------------------------------


class TestRegressors:
    def test_fit_with_X_runs(self, intermittent_y, regressors_X):  # noqa: N802
        m = OM(
            model="ANN",
            occurrence="odds-ratio",
            lags=[1],
            regressors="use",
        )
        m.fit(intermittent_y, X=regressors_X)
        # coef contains alpha + 2 regression coefficients
        assert len(m.coef) >= 3
        # fitted in [0, 1]
        f = m.fitted
        assert np.all(f >= 0.0 - 1e-12)
        assert np.all(f <= 1.0 + 1e-12)

    def test_predict_with_X_future(self, intermittent_y, regressors_X):  # noqa: N802
        m = OM(
            model="ANN",
            occurrence="odds-ratio",
            lags=[1],
            regressors="use",
        )
        m.fit(intermittent_y, X=regressors_X)
        rng = np.random.default_rng(99)
        X_future = rng.standard_normal((10, 2))
        fc = m.predict(h=10, X=X_future)
        assert fc.mean.shape == (10,)
        assert np.all(fc.mean.values >= 0.0 - 1e-12)
        assert np.all(fc.mean.values <= 1.0 + 1e-12)


# --------------------------------------------------------------------------
# ARIMA orders
# --------------------------------------------------------------------------


class TestARIMA:
    @pytest.mark.parametrize(
        "orders",
        [
            {"ar": [1], "i": [0], "ma": [0]},
            {"ar": [0], "i": [1], "ma": [1]},
            {"ar": [1], "i": [1], "ma": [1]},
        ],
    )
    def test_arima_orders_fit(self, intermittent_y, orders):
        m = OM(
            model="ANN",
            occurrence="odds-ratio",
            lags=[1],
            orders=orders,
        )
        m.fit(intermittent_y)
        # Reflected back
        assert m.orders == {
            "ar": list(orders["ar"]),
            "i": list(orders["i"]),
            "ma": list(orders["ma"]),
        }
        # Model name contains ARIMA part
        assert "ARIMA" in m.model_name
        # Fitted in [0, 1]
        f = m.fitted
        assert np.all(f >= 0.0 - 1e-12)
        assert np.all(f <= 1.0 + 1e-12)


# --------------------------------------------------------------------------
# Various ETS shapes
# --------------------------------------------------------------------------


class TestETSShapes:
    @pytest.mark.parametrize(
        "model,lags",
        [
            ("ANN", [1]),
            ("MNN", [1]),
            ("AAN", [1]),
            ("MAN", [1]),
            ("AAdN", [1]),
            ("MAdN", [1]),
            ("ANA", [12]),
            ("MNM", [12]),
            ("AAA", [12]),
        ],
    )
    def test_shape_runs(self, intermittent_y, model, lags):
        m = OM(model=model, occurrence="odds-ratio", lags=lags)
        m.fit(intermittent_y)
        assert m.fitted.shape == intermittent_y.shape


# --------------------------------------------------------------------------
# Input variants
# --------------------------------------------------------------------------


class TestInputs:
    def test_integer_counts(self):
        rng = np.random.default_rng(13)
        y = rng.poisson(0.4, 150).astype(int)
        m = OM(model="MNN", occurrence="odds-ratio", lags=[1]).fit(y)
        assert m.fitted.shape == y.shape

    def test_binary_input(self, binary_y):
        m = OM(model="MNN", occurrence="odds-ratio", lags=[1]).fit(binary_y)
        assert m.fitted.shape == binary_y.shape


# --------------------------------------------------------------------------
# Regressions
# --------------------------------------------------------------------------


class TestInferenceMethods:
    """vcov / confint / summary for OM."""

    @pytest.fixture
    def fitted_om(self, intermittent_y):
        return OM(model="MNN", occurrence="odds-ratio", lags=[1]).fit(intermittent_y)

    def test_vcov_shape(self, fitted_om):
        V = fitted_om.vcov()
        assert V.shape == (fitted_om.nparam - 1, fitted_om.nparam - 1) or V.shape == (
            len(fitted_om.coef),
            len(fitted_om.coef),
        )
        # Diagonal must be non-negative (covariance variances)
        diag = np.diag(V.to_numpy())
        assert np.all(diag >= 0)

    def test_confint_columns(self, fitted_om):
        ci = fitted_om.confint(level=0.95)
        assert ci.columns.tolist() == ["S.E.", "2.5%", "97.5%"]
        # SEs must be non-negative; lo <= hi
        assert (ci["S.E."] >= 0).all()
        assert (ci["2.5%"] <= ci["97.5%"]).all()

    def test_summary_contains_occurrence_header(self, fitted_om):
        text = str(fitted_om.summary())
        assert "Occurrence model" in text
        assert "Sample size" in text

    def test_confint_level_changes_bounds(self, fitted_om):
        ci90 = fitted_om.confint(level=0.90)
        ci99 = fitted_om.confint(level=0.99)
        # Higher confidence => wider intervals
        width90 = ci90.iloc[:, 2] - ci90.iloc[:, 1]
        width99 = ci99.iloc[:, 2] - ci99.iloc[:, 1]
        assert (width99 >= width90).all()


class TestRegressions:
    """Bug-fix regression tests."""

    def test_user_supplied_persistence_does_not_crash(self, intermittent_y):
        """Bug #5: `persistence={alpha:0, beta:0}` must not crash.

        On the R side this raised "object 'nParamEstimated' not found"
        because the `nParamEstimated` variable was only initialised in the
        ``occurrence == "fixed"`` short-circuit. Python's `_fit_fixed`
        side-steps the same path, so this test pins the public-API
        contract: providing a full persistence vector is a supported
        usage and the OM must fit cleanly.
        """
        m = OM(
            model="AAN",
            occurrence="odds-ratio",
            lags=[1],
            persistence={"alpha": 0.0, "beta": 0.0},
        )
        m.fit(intermittent_y)
        # The model must have fitted — no crash, no error
        assert m.fitted.shape == intermittent_y.shape


class TestOMvcovType:
    def _y(self):
        rng = np.random.default_rng(9)
        return rng.binomial(1, 0.6, 160).astype(float)

    def test_opg_default_psd_and_genuine(self):
        m = OM(model="ANN", occurrence="odds-ratio", initial="optimal").fit(self._y())
        v_opg = m.vcov(type="opg").values
        v_h = m.vcov(type="hessian").values
        np.testing.assert_allclose(m.vcov().values, v_opg)
        assert np.all(np.linalg.eigvalsh((v_opg + v_opg.T) / 2) > -1e-6)
        assert not np.allclose(v_opg, v_h, atol=1e-6)

    def test_bootstrap_deprecated(self):
        import warnings

        m = OM(model="ANN", occurrence="odds-ratio", initial="optimal").fit(self._y())
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m.vcov(bootstrap=True, nsim=20)
            assert any(issubclass(x.category, DeprecationWarning) for x in w)
