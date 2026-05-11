"""
Tests to verify that nlopt_kargs parameters flow correctly through to NLopt.

Covers both the direct estimation path (fixed model) and the model-selection
path (ZXZ, ZZZ), where a bug previously caused TypeError for unknown kwargs.
"""

import numpy as np
import pytest

from smooth import ADAM, AutoADAM

# 48-obs series: trend + seasonality (fast to fit, enough for ZXZ/ZZZ)
np.random.seed(42)
_t = np.arange(48)
Y = 100 + 0.5 * _t + 8 * np.sin(2 * np.pi * _t / 12) + np.random.randn(48) * 3


class TestNloptKargsSingleModel:
    """nlopt_kargs on fixed (non-selection) models."""

    def test_maxeval(self):
        model = ADAM(model="ANN", lags=[1], nlopt_kargs={"maxeval": 5})
        model.fit(Y)
        assert hasattr(model, "loglik")

    def test_print_level(self):
        model = ADAM(model="ANN", lags=[1], nlopt_kargs={"print_level": 0})
        model.fit(Y)
        assert hasattr(model, "loglik")

    def test_xtol_rel(self):
        model = ADAM(model="ANN", lags=[1], nlopt_kargs={"xtol_rel": 1e-4})
        model.fit(Y)
        assert hasattr(model, "loglik")

    def test_algorithm_sbplx(self):
        model = ADAM(
            model="ANN", lags=[1], nlopt_kargs={"algorithm": "NLOPT_LN_SBPLX"}
        )
        model.fit(Y)
        assert hasattr(model, "loglik")

    def test_multiple_params(self):
        model = ADAM(
            model="AAN",
            lags=[1],
            nlopt_kargs={"maxeval": 10, "print_level": 0, "xtol_rel": 1e-4},
        )
        model.fit(Y)
        assert hasattr(model, "loglik")

    def test_maxeval_affects_loglik(self):
        """Tight maxeval (very few evals) should give a different loglik than default."""
        m_tight = ADAM(model="ANN", lags=[1], nlopt_kargs={"maxeval": 2})
        m_tight.fit(Y)

        m_full = ADAM(model="ANN", lags=[1])
        m_full.fit(Y)

        # A very tight budget almost certainly gives a suboptimal solution
        assert m_tight.loglik != m_full.loglik


class TestNloptKargsModelSelection:
    """nlopt_kargs on model-selection models (ZXZ, ZZZ)."""

    def test_maxeval_with_zxz(self):
        """maxeval must not raise TypeError in the selection path — original bug."""
        model = ADAM(
            model="ZXZ",
            lags=[1, 12],
            nlopt_kargs={"maxeval": 2},
        )
        model.fit(Y)  # must not raise TypeError
        assert hasattr(model, "loglik")

    def test_maxeval_with_zzz(self):
        model = ADAM(
            model="ZZZ",
            lags=[1, 12],
            nlopt_kargs={"maxeval": 2},
        )
        model.fit(Y)
        assert hasattr(model, "loglik")

    def test_algorithm_with_selection(self):
        model = ADAM(
            model="ZXZ",
            lags=[1, 12],
            nlopt_kargs={"algorithm": "NLOPT_LN_SBPLX"},
        )
        model.fit(Y)
        assert hasattr(model, "loglik")

    def test_multiple_params_selection(self):
        model = ADAM(
            model="ZXZ",
            lags=[1, 12],
            nlopt_kargs={"maxeval": 2, "xtol_rel": 1e-4},
        )
        model.fit(Y)
        assert hasattr(model, "loglik")


class TestNloptKargsAutoADAM:
    """nlopt_kargs forwarded through AutoADAM."""

    def test_maxeval_autoadam(self):
        model = AutoADAM(
            model="ANN",
            lags=[1],
            arima_select=False,
            distribution=["dnorm"],
            nlopt_kargs={"maxeval": 5},
        )
        model.fit(Y)
        assert hasattr(model, "loglik")


class TestNloptKargsInvalid:
    """Edge cases for nlopt_kargs values."""

    def test_none_is_allowed(self):
        """nlopt_kargs=None is the default and must work."""
        model = ADAM(model="ANN", lags=[1], nlopt_kargs=None)
        model.fit(Y)
        assert hasattr(model, "loglik")

    def test_unknown_param_raises(self):
        """An unrecognised kwarg should propagate as TypeError from estimator."""
        model = ADAM(model="ANN", lags=[1], nlopt_kargs={"nonexistent_param": 1})
        with pytest.raises(TypeError):
            model.fit(Y)
