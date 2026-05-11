"""Tests for occurrence-aware prediction intervals in ADAM.

Formula tests: verify the interval adjustment formula analytically.
R comparison tests: sanity-check against R reference with loose tolerance
  (exact match is not expected because Python and R optimise to different
  model parameters; the test just confirms the magnitude is plausible).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from smooth import ADAM

DATA_DIR = Path(__file__).parent / "data"
OCC_DIR = DATA_DIR / "adam_occ_intervals"
OM_DIR = DATA_DIR / "om"


@pytest.fixture(scope="module")
def y():
    return pd.read_csv(OM_DIR / "intermittent_demand.csv")["y"].to_numpy()


@pytest.fixture(scope="module")
def r_approx():
    upper = pd.read_csv(OCC_DIR / "approximate_ann_or_upper.csv")["upper"].to_numpy()
    lower = pd.read_csv(OCC_DIR / "approximate_ann_or_lower.csv")["lower"].to_numpy()
    mean_ = pd.read_csv(OCC_DIR / "approximate_ann_or_mean.csv")["mean"].to_numpy()
    return {"upper": upper, "lower": lower, "mean": mean_}


@pytest.fixture(scope="module")
def r_sim():
    upper = pd.read_csv(OCC_DIR / "simulated_ann_or_upper.csv")["upper"].to_numpy()
    lower = pd.read_csv(OCC_DIR / "simulated_ann_or_lower.csv")["lower"].to_numpy()
    return {"upper": upper, "lower": lower}


# ---------------------------------------------------------------------------
# Approximate interval formula verification
# ---------------------------------------------------------------------------


class TestApproximateFormula:
    """Verify the compound-distribution interval formula directly against
    an analytical calculation using the same model parameters."""

    @pytest.fixture(scope="class")
    def model_and_fc(self, y):
        m = ADAM(model="ANN", lags=[1], occurrence="odds-ratio")
        m.fit(y)
        fc = m.predict(h=10, interval="approximate", level=0.95)
        return m, fc

    def test_monotonicity(self, model_and_fc):
        _, fc = model_and_fc
        mean = fc.mean.values
        lower = fc.lower.values.ravel()
        upper = fc.upper.values.ravel()
        assert np.all(lower <= mean + 1e-10), "lower > mean"
        assert np.all(mean <= upper + 1e-10), "mean > upper"

    def test_interval_label(self, model_and_fc):
        _, fc = model_and_fc
        assert fc.interval == "approximate"

    def test_finite(self, model_and_fc):
        _, fc = model_and_fc
        assert np.all(np.isfinite(fc.upper.values))
        assert np.all(np.isfinite(fc.lower.values))
        assert np.all(np.isfinite(fc.mean.values))

    def test_formula_matches_manual_calculation(self, model_and_fc, y):
        """The adjusted-level formula for approximate occurrence intervals.

        R uses loc=compound_mean and adjusts the confidence level:
          conf_adj = max(0, (conf - (1-p)) / p)
          q_low = (1 - conf_adj) / 2
          q_up  = (1 + conf_adj) / 2
          upper = compound_mean + N_ppf(q_up,  0, sigma)
          lower = compound_mean + N_ppf(q_low, 0, sigma)
        """
        m, fc = model_and_fc
        p = m._occurrence["p_forecast"]  # already computed by predict()
        if p is None:
            pytest.skip("p_forecast not stored")

        from smooth.adam_general.core.forecaster._helpers import (
            _prepare_matrices_for_forecast,
        )
        from smooth.adam_general.core.utils.var_covar import covar_anal
        from smooth.adam_general.core.utils.var_covar import sigma as sigma_fn

        gen = m._general.copy()
        gen["h"] = 10
        mat_vt, mat_wt, vec_g, mat_f = _prepare_matrices_for_forecast(
            m._prepared, m._observations, m._lags_model, gen
        )
        s2 = sigma_fn(m._observations, m._params_info, m._general, m._prepared) ** 2
        v_voc = covar_anal(
            m._lags_model["lags_model_all"], 10, mat_wt, mat_f, vec_g, s2
        )
        sigma_h = np.sqrt(np.diag(v_voc))  # (10,)

        p_arr = np.asarray(p, dtype=float)
        conf = 0.95
        conf_adj = np.maximum(0.0, (conf - (1 - p_arr)) / p_arr)
        q_low = (1 - conf_adj) / 2
        q_up = (1 + conf_adj) / 2

        mean = fc.mean.values
        expected_upper = mean + stats.norm.ppf(q_up, loc=0, scale=sigma_h)
        expected_lower = mean + stats.norm.ppf(q_low, loc=0, scale=sigma_h)

        np.testing.assert_allclose(
            fc.upper.values.ravel(), expected_upper, rtol=1e-5, atol=1e-7,
            err_msg="Upper bounds do not match the analytical formula"
        )
        np.testing.assert_allclose(
            fc.lower.values.ravel(), expected_lower, rtol=1e-5, atol=1e-7,
            err_msg="Lower bounds do not match the analytical formula"
        )


# ---------------------------------------------------------------------------
# R comparison — sanity check (model parameters may differ)
# ---------------------------------------------------------------------------


class TestApproximateVsR:
    """Sanity-check approximate intervals against R reference.

    Python and R may optimise to different parameter values; we only verify
    that the ratio upper/mean and |lower|/mean are in the same ballpark.
    """

    @pytest.fixture(scope="class")
    def fc(self, y):
        m = ADAM(model="ANN", lags=[1], occurrence="odds-ratio")
        m.fit(y)
        return m.predict(h=10, interval="approximate", level=0.95)

    def test_upper_mean_ratio_plausible(self, fc, r_approx):
        ratio_py = np.mean(fc.upper.values.ravel() / fc.mean.values)
        ratio_r = np.mean(r_approx["upper"] / r_approx["mean"])
        # Both ratios should be > 1; we don't expect exact match
        assert ratio_py > 1.0, "Python upper/mean ratio should be > 1"
        assert ratio_r > 1.0, "R upper/mean ratio should be > 1"
        # Order-of-magnitude check: within 3× of each other
        assert ratio_py / ratio_r < 3.0 and ratio_r / ratio_py < 3.0, (
            f"Ratios differ more than 3×: Python={ratio_py:.3f}, R={ratio_r:.3f}"
        )

    def test_lower_is_below_mean(self, fc):
        assert np.all(fc.lower.values.ravel() <= fc.mean.values + 1e-10)


# ---------------------------------------------------------------------------
# Simulated intervals
# ---------------------------------------------------------------------------


class TestSimulated:
    @pytest.fixture(scope="class")
    def model_and_fc(self, y):
        m = ADAM(model="ANN", lags=[1], occurrence="odds-ratio")
        m.fit(y)
        fc = m.predict(h=10, interval="simulated", level=0.95, nsim=100000)
        return m, fc

    def test_shape(self, model_and_fc):
        _, fc = model_and_fc
        assert fc.mean.shape == (10,)
        assert fc.upper.shape == (10, 1)
        assert fc.lower.shape == (10, 1)

    def test_finite(self, model_and_fc):
        _, fc = model_and_fc
        assert np.all(np.isfinite(fc.mean.values))
        assert np.all(np.isfinite(fc.upper.values))
        assert np.all(np.isfinite(fc.lower.values))

    def test_monotonicity(self, model_and_fc):
        _, fc = model_and_fc
        mean = fc.mean.values
        assert np.all(fc.lower.values.ravel() <= mean + 1e-6)
        assert np.all(mean <= fc.upper.values.ravel() + 1e-6)

    def test_lower_near_zero(self, model_and_fc):
        """Lower bounds should be near 0: with p≈0.345 the compound 2.5th
        quantile is dominated by the mass at 0 (P(Y=0) ≈ 0.655 > 0.025)."""
        _, fc = model_and_fc
        assert np.all(fc.lower.values.ravel() < 0.2), (
            "Lower bound should be near 0 for this intermittent series"
        )

    def test_r_lower_matches(self, model_and_fc, r_sim):
        """R reference lower bounds are all 0; Python should be close."""
        _, fc = model_and_fc
        np.testing.assert_allclose(
            fc.lower.values.ravel(), r_sim["lower"], atol=0.1,
            err_msg="Lower simulated bounds diverge from R reference"
        )

    def test_upper_positive(self, model_and_fc):
        _, fc = model_and_fc
        assert np.all(fc.upper.values.ravel() > 0)

    def test_interval_label(self, model_and_fc):
        _, fc = model_and_fc
        assert fc.interval == "simulated"


# ---------------------------------------------------------------------------
# Multiplicative ETS — shape / monotonicity only
# ---------------------------------------------------------------------------


class TestMultiplicativeOccurrence:
    @pytest.fixture(scope="class")
    def fc(self, y):
        m = ADAM(model="MNN", lags=[1], occurrence="odds-ratio")
        m.fit(y)
        return m.predict(h=10, interval="prediction", level=0.95, nsim=5000)

    def test_no_nan(self, fc):
        assert not np.any(np.isnan(fc.mean.values))
        assert not np.any(np.isnan(fc.upper.values))
        assert not np.any(np.isnan(fc.lower.values))

    def test_monotonicity(self, fc):
        mean = fc.mean.values
        assert np.all(fc.lower.values.ravel() <= mean + 1e-6)
        assert np.all(mean <= fc.upper.values.ravel() + 1e-6)

    def test_non_negative(self, fc):
        assert np.all(fc.mean.values >= -1e-10)


# ---------------------------------------------------------------------------
# interval="none" still works with occurrence model
# ---------------------------------------------------------------------------


class TestNoIntervalOccurrence:
    def test_predict_none_interval(self, y):
        m = ADAM(model="ANN", lags=[1], occurrence="odds-ratio")
        m.fit(y)
        fc = m.predict(h=5, interval="none")
        assert fc.lower is None
        assert fc.upper is None
        assert fc.mean.shape == (5,)
        assert np.all(fc.mean.values >= -1e-10)
