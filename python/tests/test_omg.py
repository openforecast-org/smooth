"""General-functionality tests for the OMG (General Occurrence Model) class.

Covers init/fit/predict, sub-model inspection, combined fitted probabilities,
property surface, holdout, X regressors, ARIMA orders, various ETS shapes,
and the transparent ``OM(occurrence='general', ...)`` delegation path.

R-vs-Python exact-match tests live in ``test_omg_r_comparison.py``.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from smooth import OM, OMG

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def intermittent_y():
    """Deterministic intermittent demand, n=200."""
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


@pytest.fixture
def fitted_omg(intermittent_y):
    return OMG(model_a="MNN", lags=[1]).fit(intermittent_y)


# --------------------------------------------------------------------------
# Initialisation
# --------------------------------------------------------------------------


class TestInit:
    def test_default_init_does_not_raise(self):
        OMG()

    def test_model_b_defaults_to_model_a(self):
        m = OMG(model_a="AAN")
        assert m.model_b_spec == "AAN"

    def test_model_b_overrideable(self):
        m = OMG(model_a="ANN", model_b="MNN")
        assert m.model_a_spec == "ANN"
        assert m.model_b_spec == "MNN"

    def test_lags_default_to_none(self):
        m = OMG()
        assert m.lags is None

    def test_invalid_loss_raises(self):
        with pytest.raises(ValueError, match="loss='likelihood'"):
            OMG(loss="MAE")  # type: ignore[arg-type]

    def test_occurrence_property_is_general(self):
        m = OMG()
        assert m.occurrence == "general"


# --------------------------------------------------------------------------
# OM delegation: OM(occurrence='general') returns OMG
# --------------------------------------------------------------------------


class TestOMDelegation:
    def test_om_general_returns_omg_instance(self):
        m = OM(model="MNN", occurrence="general", lags=[1])
        assert isinstance(m, OMG)

    def test_delegated_omg_fits(self, intermittent_y):
        m = OM(model="MNN", occurrence="general", lags=[1])
        m.fit(intermittent_y)
        assert np.all(m.fitted >= 0.0)
        assert np.all(m.fitted <= 1.0)

    def test_delegated_model_a_b_same_spec(self, intermittent_y):
        m = OM(model="ANN", occurrence="general", lags=[1])
        assert isinstance(m, OMG)
        assert m.model_a_spec == "ANN"
        assert m.model_b_spec == "ANN"


# --------------------------------------------------------------------------
# Fit basic flow
# --------------------------------------------------------------------------


class TestFitBasic:
    def test_fit_returns_self(self, intermittent_y):
        m = OMG(model_a="MNN", lags=[1])
        out = m.fit(intermittent_y)
        assert out is m

    def test_fitted_shape(self, intermittent_y, fitted_omg):
        assert fitted_omg.fitted.shape == intermittent_y.shape

    def test_fitted_in_open_unit_interval(self, fitted_omg):
        f = fitted_omg.fitted
        assert np.all(f > 0.0), f"min fitted = {f.min()}"
        assert np.all(f < 1.0), f"max fitted = {f.max()}"

    def test_residuals_equal_ot_minus_fitted(self, intermittent_y, fitted_omg):
        ot = (intermittent_y != 0).astype(float)
        np.testing.assert_allclose(
            fitted_omg.residuals, ot - fitted_omg.fitted, atol=1e-12
        )

    def test_actuals_is_binary(self, fitted_omg):
        a = fitted_omg.actuals
        assert set(np.unique(a)).issubset({0.0, 1.0})

    def test_fitted_is_link_of_submodel_raw_outputs(self, intermittent_y, fitted_omg):
        """p = aFit/(aFit+bFit) must be consistent with sub-model fitted."""
        from smooth.adam_general.core.utils.omg_cost import omg_link_function

        e_a = fitted_omg._side_a["model_type_dict"]["error_type"]
        e_b = fitted_omg._side_b["model_type_dict"]["error_type"]
        p_check = omg_link_function(
            fitted_omg.model_a._prepared["y_fitted_raw"],
            fitted_omg.model_b._prepared["y_fitted_raw"],
            e_a,
            e_b,
        )
        np.testing.assert_allclose(fitted_omg.fitted, p_check, atol=1e-12)


# --------------------------------------------------------------------------
# Sub-model inspection
# --------------------------------------------------------------------------


class TestSubModels:
    def test_model_a_is_om_instance(self, fitted_omg):
        assert isinstance(fitted_omg.model_a, OM)

    def test_model_b_is_om_instance(self, fitted_omg):
        assert isinstance(fitted_omg.model_b, OM)

    def test_model_a_occurrence_is_odds_ratio(self, fitted_omg):
        assert fitted_omg.model_a._om_occurrence == "odds-ratio"

    def test_model_b_occurrence_is_inverse_odds_ratio(self, fitted_omg):
        assert fitted_omg.model_b._om_occurrence == "inverse-odds-ratio"

    def test_sub_model_fitted_shapes(self, intermittent_y, fitted_omg):
        assert fitted_omg.model_a.fitted.shape == intermittent_y.shape
        assert fitted_omg.model_b.fitted.shape == intermittent_y.shape

    def test_sub_model_fitted_in_unit_interval(self, fitted_omg):
        for sub in (fitted_omg.model_a, fitted_omg.model_b):
            f = sub.fitted
            assert np.all(f >= 0.0 - 1e-12) and np.all(f <= 1.0 + 1e-12)


# --------------------------------------------------------------------------
# Properties
# --------------------------------------------------------------------------


class TestProperties:
    def test_distribution_is_plogis(self, fitted_omg):
        assert fitted_omg.distribution_ == "plogis"

    def test_loss_is_likelihood(self, fitted_omg):
        assert fitted_omg.loss_ == "likelihood"

    def test_scale_is_nan(self, fitted_omg):
        assert np.isnan(fitted_omg.scale)
        assert np.isnan(fitted_omg.sigma)

    def test_model_name_format(self, fitted_omg):
        pattern = r"^oETS\[G\]\([AM][NAM]d?[NAM]\)\([AM][NAM]d?[NAM]\)$"
        assert re.match(pattern, fitted_omg.model_name), (
            f"got {fitted_omg.model_name!r}"
        )

    def test_loss_value_is_finite(self, fitted_omg):
        assert np.isfinite(fitted_omg.loss_value)

    def test_loglik_is_finite(self, fitted_omg):
        assert np.isfinite(fitted_omg.loglik)

    def test_loglik_equals_neg_loss(self, fitted_omg):
        np.testing.assert_allclose(fitted_omg.loglik, -fitted_omg.loss_value)

    def test_aic_aicc_bic_bicc_are_finite(self, fitted_omg):
        for v in (fitted_omg.aic, fitted_omg.aicc, fitted_omg.bic, fitted_omg.bicc):
            assert np.isfinite(v), f"IC not finite: {v}"

    def test_coef_nonempty(self, fitted_omg):
        assert len(fitted_omg.coef) > 0

    def test_nobs(self, intermittent_y, fitted_omg):
        assert fitted_omg.nobs == len(intermittent_y)

    def test_nparam_equals_coef_length(self, fitted_omg):
        assert fitted_omg.nparam == len(fitted_omg.coef)

    def test_lags_used(self, fitted_omg):
        assert fitted_omg.lags_used == [1]

    def test_time_elapsed_positive(self, fitted_omg):
        assert fitted_omg.time_elapsed > 0

    def test_property_surface_does_not_raise(self, fitted_omg):
        _ = (
            fitted_omg.fitted,
            fitted_omg.residuals,
            fitted_omg.actuals,
            fitted_omg.coef,
            fitted_omg.b_value,
            fitted_omg.loss_value,
            fitted_omg.loglik,
            fitted_omg.aic,
            fitted_omg.aicc,
            fitted_omg.bic,
            fitted_omg.bicc,
            fitted_omg.nobs,
            fitted_omg.nparam,
            fitted_omg.occurrence,
            fitted_omg.distribution_,
            fitted_omg.loss_,
            fitted_omg.scale,
            fitted_omg.sigma,
            fitted_omg.model_name,
            fitted_omg.model,
            fitted_omg.lags_used,
            fitted_omg.time_elapsed,
        )


# --------------------------------------------------------------------------
# Predict
# --------------------------------------------------------------------------


class TestPredict:
    def test_predict_shape(self, fitted_omg):
        fc = fitted_omg.predict(h=10)
        assert fc.mean.shape == (10,)

    def test_predict_in_open_unit_interval(self, fitted_omg):
        fc = fitted_omg.predict(h=10)
        p = fc.mean.values
        assert np.all(p > 0.0), f"min forecast = {p.min()}"
        assert np.all(p < 1.0), f"max forecast = {p.max()}"

    def test_predict_warns_on_interval(self, fitted_omg):
        with pytest.warns(UserWarning, match="Intervals on the probability scale"):
            fitted_omg.predict(h=5, interval="prediction")

    def test_predict_does_not_mutate_states(self, intermittent_y):
        m = OMG(model_a="MNN", lags=[1]).fit(intermittent_y)
        f1 = m.fitted.copy()
        m.predict(h=10)
        np.testing.assert_array_equal(m.fitted, f1)


# --------------------------------------------------------------------------
# Holdout
# --------------------------------------------------------------------------


class TestHoldout:
    def test_holdout_split(self, intermittent_y):
        m = OMG(model_a="MNN", lags=[1], h=10, holdout=True)
        m.fit(intermittent_y)
        assert m.holdout_data is not None
        assert m.holdout_data.shape == (10,)
        assert m.nobs == len(intermittent_y) - 10

    def test_auto_forecast_populated_when_h_gt_0(self, intermittent_y):
        m = OMG(model_a="MNN", lags=[1], h=10, holdout=True)
        m.fit(intermittent_y)
        assert m._auto_forecast is not None
        assert m._auto_forecast.mean.shape == (10,)

    def test_no_holdout_data_when_holdout_false(self, fitted_omg):
        assert fitted_omg.holdout_data is None


# --------------------------------------------------------------------------
# Explanatory regressors
# --------------------------------------------------------------------------


class TestRegressors:
    def test_fit_with_X_runs(self, intermittent_y, regressors_X):  # noqa: N802
        m = OMG(model_a="ANN", lags=[1])
        m.fit(intermittent_y, X=regressors_X)
        f = m.fitted
        assert np.all(f > 0.0)
        assert np.all(f < 1.0)

    def test_coef_length_includes_xreg(self, intermittent_y, regressors_X):  # noqa: N802
        m = OMG(model_a="ANN", lags=[1]).fit(intermittent_y, X=regressors_X)
        # Two sides, each with alpha + 2 xreg coefficients at minimum
        assert len(m.coef) >= 6

    def test_predict_with_X_future(self, intermittent_y, regressors_X):  # noqa: N802
        m = OMG(model_a="ANN", lags=[1]).fit(intermittent_y, X=regressors_X)
        rng = np.random.default_rng(99)
        X_future = rng.standard_normal((10, 2))
        fc = m.predict(h=10, X=X_future)
        assert fc.mean.shape == (10,)


# --------------------------------------------------------------------------
# ARIMA orders
# --------------------------------------------------------------------------


class TestARIMA:
    @pytest.mark.parametrize(
        "orders",
        [
            {"ar": [1], "i": [0], "ma": [0]},
            {"ar": [0], "i": [1], "ma": [1]},
        ],
    )
    def test_arima_orders_fit(self, intermittent_y, orders):
        m = OMG(model_a="ANN", lags=[1], orders_a=orders)
        m.fit(intermittent_y)
        assert np.isfinite(m.loss_value)
        f = m.fitted
        assert np.all(f > 0.0)
        assert np.all(f < 1.0)


# --------------------------------------------------------------------------
# Various ETS shapes
# --------------------------------------------------------------------------


class TestETSShapes:
    @pytest.mark.parametrize(
        "model_a,lags",
        [
            ("ANN", [1]),
            ("MNN", [1]),
            ("AAN", [1]),
            ("AAdN", [1]),
            ("ANA", [12]),
            ("MNM", [12]),
        ],
    )
    def test_shape_runs(self, intermittent_y, model_a, lags):
        m = OMG(model_a=model_a, lags=lags)
        m.fit(intermittent_y)
        assert m.fitted.shape == intermittent_y.shape
        assert np.isfinite(m.loss_value)

    def test_asymmetric_ab_shapes(self, intermittent_y):
        """Model A and B can have different ETS structures."""
        m = OMG(model_a="ANN", model_b="MNN", lags=[1])
        m.fit(intermittent_y)
        assert np.isfinite(m.loss_value)
        assert np.all(m.fitted > 0.0) and np.all(m.fitted < 1.0)


# --------------------------------------------------------------------------
# Input variants
# --------------------------------------------------------------------------


class TestInputs:
    def test_integer_counts(self):
        rng = np.random.default_rng(13)
        y = rng.poisson(0.4, 150).astype(int)
        m = OMG(model_a="MNN", lags=[1]).fit(y)
        assert m.fitted.shape == y.shape

    def test_binary_input(self, binary_y):
        m = OMG(model_a="MNN", lags=[1]).fit(binary_y)
        assert m.fitted.shape == binary_y.shape
        assert np.all(m.fitted > 0.0) and np.all(m.fitted < 1.0)
