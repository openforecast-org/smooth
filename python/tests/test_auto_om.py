"""Tests for the AutoOM class — automatic occurrence model selection.

AutoOM.fit() returns the best OM or OMG object directly (matching R's auto.om()
which returns the best om object). The returned model has all standard OM/OMG
attributes plus time_elapsed_ (total selection time).
"""

from __future__ import annotations

import numpy as np
import pytest

from smooth import OM, AutoOM, OMG


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def intermittent_y():
    rng = np.random.default_rng(41)
    return rng.poisson(0.3, 200).astype(float)


@pytest.fixture
def fitted_auto(intermittent_y):
    return AutoOM(model="MNN", lags=[1]).fit(intermittent_y)


# --------------------------------------------------------------------------
# Initialisation
# --------------------------------------------------------------------------


class TestInit:
    def test_default_init_does_not_raise(self):
        AutoOM()

    def test_invalid_occurrence_raises(self):
        with pytest.raises(ValueError, match="Unknown occurrence"):
            AutoOM(occurrence=["fixed", "bogus"])

    def test_invalid_ic_raises(self):
        with pytest.raises(ValueError, match="ic must be"):
            AutoOM(ic="WHATEVER")

    def test_string_occurrence_is_normalised_to_list(self):
        m = AutoOM(occurrence="fixed")
        assert m.occurrence == ["fixed"]

    def test_occurrence_defaults_to_all_five(self):
        m = AutoOM()
        assert set(m.occurrence) == {
            "fixed",
            "odds-ratio",
            "inverse-odds-ratio",
            "direct",
            "general",
        }


# --------------------------------------------------------------------------
# OM delegation: OM(occurrence='auto') returns AutoOM before fit
# --------------------------------------------------------------------------


class TestOMDelegation:
    def test_om_auto_returns_autoOM_instance(self):
        m = OM(model="MNN", occurrence="auto", lags=[1])
        assert isinstance(m, AutoOM)

    def test_delegated_autoOM_fits(self, intermittent_y):
        m = OM(model="MNN", occurrence="auto", lags=[1]).fit(intermittent_y)
        assert isinstance(m, (OM, OMG))
        assert np.all(m.fitted >= 0.0)
        assert np.all(m.fitted <= 1.0)


# --------------------------------------------------------------------------
# Fit — returns OM or OMG
# --------------------------------------------------------------------------


class TestFit:
    def test_fit_returns_om_or_omg(self, intermittent_y):
        out = AutoOM(model="MNN", lags=[1]).fit(intermittent_y)
        assert isinstance(out, (OM, OMG))

    def test_fitted_has_time_elapsed(self, fitted_auto):
        assert hasattr(fitted_auto, "time_elapsed_")
        assert fitted_auto.time_elapsed_ > 0

    def test_fitted_probabilities_in_range(self, fitted_auto):
        assert np.all(fitted_auto.fitted >= 0.0)
        assert np.all(fitted_auto.fitted <= 1.0)

    def test_restricted_occurrence_returns_om(self, intermittent_y):
        m = AutoOM(model="MNN", lags=[1], occurrence=["fixed", "odds-ratio"]).fit(
            intermittent_y
        )
        assert isinstance(m, OM)


# --------------------------------------------------------------------------
# Properties — all standard OM/OMG attributes are accessible
# --------------------------------------------------------------------------


class TestProperties:
    def test_fitted_shape(self, intermittent_y, fitted_auto):
        assert fitted_auto.fitted.shape == intermittent_y.shape

    def test_fitted_in_unit_interval(self, fitted_auto):
        f = fitted_auto.fitted
        assert np.all(f >= 0.0) and np.all(f <= 1.0)

    def test_residuals_shape(self, intermittent_y, fitted_auto):
        assert fitted_auto.residuals.shape == intermittent_y.shape

    def test_actuals_is_binary(self, fitted_auto):
        a = fitted_auto.actuals
        assert set(np.unique(a)).issubset({0.0, 1.0})

    def test_loss_value_is_finite(self, fitted_auto):
        assert np.isfinite(fitted_auto.loss_value)

    def test_loglik_is_finite(self, fitted_auto):
        assert np.isfinite(fitted_auto.loglik)

    def test_loglik_equals_neg_loss(self, fitted_auto):
        np.testing.assert_allclose(fitted_auto.loglik, -fitted_auto.loss_value)

    def test_aic_aicc_bic_bicc_are_finite(self, fitted_auto):
        for v in (fitted_auto.aic, fitted_auto.aicc, fitted_auto.bic, fitted_auto.bicc):
            assert np.isfinite(v), f"IC not finite: {v}"

    def test_nobs(self, intermittent_y, fitted_auto):
        assert fitted_auto.nobs == len(intermittent_y)

    def test_nparam_positive(self, fitted_auto):
        assert fitted_auto.nparam > 0

    def test_coef_nonempty(self, fitted_auto):
        assert len(fitted_auto.coef) > 0

    def test_model_name_is_string(self, fitted_auto):
        assert isinstance(fitted_auto.model_name, str)
        assert len(fitted_auto.model_name) > 0

    def test_lags_used_nonempty(self, fitted_auto):
        assert len(fitted_auto.lags_used) > 0

    def test_scale_is_nan(self, fitted_auto):
        assert np.isnan(fitted_auto.scale)
        assert np.isnan(fitted_auto.sigma)

    def test_distribution_is_plogis(self, fitted_auto):
        assert fitted_auto.distribution_ == "plogis"

    def test_loss_is_likelihood(self, fitted_auto):
        assert fitted_auto.loss_ == "likelihood"

    def test_time_elapsed_positive(self, fitted_auto):
        assert fitted_auto.time_elapsed_ > 0


# --------------------------------------------------------------------------
# Predict
# --------------------------------------------------------------------------


class TestPredict:
    def test_predict_shape(self, fitted_auto):
        fc = fitted_auto.predict(h=10)
        assert fc.mean.shape == (10,)

    def test_predict_in_unit_interval(self, fitted_auto):
        fc = fitted_auto.predict(h=10)
        p = fc.mean.values
        assert np.all(p >= 0.0) and np.all(p <= 1.0)


# --------------------------------------------------------------------------
# Holdout
# --------------------------------------------------------------------------


class TestHoldout:
    def test_holdout_data_populated(self, intermittent_y):
        m = AutoOM(model="MNN", lags=[1], h=10, holdout=True).fit(intermittent_y)
        assert m.holdout_data is not None
        assert m.holdout_data.shape == (10,)

    def test_nobs_reduced_by_h(self, intermittent_y):
        m = AutoOM(model="MNN", lags=[1], h=10, holdout=True).fit(intermittent_y)
        assert m.nobs == len(intermittent_y) - 10

    def test_no_holdout_data_when_holdout_false(self, fitted_auto):
        assert fitted_auto.holdout_data is None
