"""Smoke tests for the Python port of ``coefbootstrap``.

These tests stay in the default suite (no ``r_parity`` marker): they're
fast — ``nsim`` is kept tiny — and deterministic with a fixed ``seed``.
The cross-language parity tests live in
``test_coefbootstrap_r_parity.py`` behind the ``r_parity`` marker.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from smooth import ADAM, ES, MSARIMA, OM, OMG
from smooth.adam_general.core.utils.bootstrap import BootstrapResult


@pytest.fixture(scope="module")
def y_continuous() -> np.ndarray:
    rng = np.random.default_rng(0)
    return (100 + np.cumsum(rng.standard_normal(120))).astype(float)


@pytest.fixture(scope="module")
def y_intermittent() -> np.ndarray:
    rng = np.random.default_rng(0)
    y = rng.poisson(0.3, 200).astype(float)
    # Force at least one 0 and one non-zero observation so OMG can fit.
    y[0] = 1.0
    y[1] = 0.0
    return y


def _silence_inner_warnings():
    return warnings.catch_warnings()


def test_adam_coefbootstrap_shape(y_continuous):
    """ADAM.coefbootstrap returns the BootstrapResult shape advertised in
    the docstring (n_eff × k coef matrix, k × k vcov)."""
    with _silence_inner_warnings():
        warnings.simplefilter("ignore")
        m = ADAM(model="ANN").fit(y_continuous)
    boot = m.coefbootstrap(nsim=20, seed=42)
    assert isinstance(boot, BootstrapResult)
    k = len(m.coef_names)
    assert boot.vcov.shape == (k, k)
    assert boot.coefficients.shape == (boot.nsim_effective, k)
    assert list(boot.coefficients.columns) == list(m.coef_names)
    assert boot.method == "cr"
    assert boot.nsim == 20
    assert boot.time_elapsed > 0


def test_adam_coefbootstrap_seed_reproducible(y_continuous):
    """Same seed → identical replicate matrix (modulo optimizer noise: we
    only assert close-equal, not bit-equal, because nlopt may produce tiny
    differences across runs)."""
    with _silence_inner_warnings():
        warnings.simplefilter("ignore")
        m = ADAM(model="ANN").fit(y_continuous)
    b1 = m.coefbootstrap(nsim=15, seed=7)
    b2 = m.coefbootstrap(nsim=15, seed=7)
    np.testing.assert_allclose(
        b1.coefficients.to_numpy(),
        b2.coefficients.to_numpy(),
        rtol=1e-6,
        atol=1e-8,
    )


def test_es_inherits_coefbootstrap(y_continuous):
    """ES is an ADAM subclass — it should inherit a working coefbootstrap."""
    with _silence_inner_warnings():
        warnings.simplefilter("ignore")
        es = ES(model="AAN").fit(y_continuous)
    boot = es.coefbootstrap(nsim=10, seed=42)
    assert boot.nsim_effective > 0
    # AAN has alpha + beta (and possibly trend init / level init).
    assert {"alpha", "beta"}.issubset(set(boot.coefficients.columns))


def test_msarima_inherits_coefbootstrap(y_continuous):
    """MSARIMA also inherits from ADAM."""
    with _silence_inner_warnings():
        warnings.simplefilter("ignore")
        ar = MSARIMA(ar_order=1, i_order=0, ma_order=1).fit(y_continuous)
    boot = ar.coefbootstrap(nsim=10, seed=42)
    assert boot.nsim_effective > 0
    cols = set(boot.coefficients.columns)
    assert any(c.startswith("phi") for c in cols)
    assert any(c.startswith("theta") for c in cols)


def test_om_coefbootstrap_shape(y_intermittent):
    """OM bootstrap refits the OM (not a plain ADAM) so the occurrence
    machinery is preserved across replicates."""
    with _silence_inner_warnings():
        warnings.simplefilter("ignore")
        om = OM(model="MNN", occurrence="odds-ratio").fit(y_intermittent)
    boot = om.coefbootstrap(nsim=15, seed=42)
    assert boot.nsim_effective > 0
    assert boot.vcov.shape[0] == len(om.coef_names)
    assert boot.model == "MNN"


def test_omg_coefbootstrap_joint_names(y_intermittent):
    """OMG's joint coefficient matrix uses A:/B: prefixes for sub-model
    parameters (mirrors R's vcov.omg row/col naming)."""
    with _silence_inner_warnings():
        warnings.simplefilter("ignore")
        g = OMG(model_a="ANN", model_b="ANN").fit(y_intermittent)
    boot = g.coefbootstrap(nsim=15, seed=42)
    assert boot.nsim_effective > 0
    cols = list(boot.coefficients.columns)
    assert all(c.startswith(("A:", "B:")) for c in cols), cols
    assert boot.model == "omg"


def test_dsr_raises_not_implemented(y_continuous):
    with _silence_inner_warnings():
        warnings.simplefilter("ignore")
        m = ADAM(model="ANN").fit(y_continuous)
    with pytest.raises(NotImplementedError, match="dsr"):
        m.coefbootstrap(nsim=5, method="dsr")


try:
    import joblib as _joblib  # noqa: F401

    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False


@pytest.mark.skipif(not _HAS_JOBLIB, reason="joblib not installed")
def test_parallel_with_joblib_matches_serial(y_continuous):
    """With joblib installed, serial and parallel runs are bit-identical
    given the same seed — the indices are pre-generated upstream and the
    optimiser is deterministic, so worker scheduling is the only source
    of nondeterminism and it does not affect the *content* of each
    replicate."""
    with _silence_inner_warnings():
        warnings.simplefilter("ignore")
        m = ADAM(model="AAN").fit(y_continuous)
    boot_serial = m.coefbootstrap(nsim=20, seed=33)
    boot_par = m.coefbootstrap(nsim=20, seed=33, parallel=True)
    assert boot_serial.parallel is False
    assert boot_par.parallel is True
    np.testing.assert_allclose(
        boot_serial.coefficients.to_numpy(),
        boot_par.coefficients.to_numpy(),
        rtol=1e-6,
        atol=1e-8,
    )


@pytest.mark.skipif(not _HAS_JOBLIB, reason="joblib not installed")
def test_parallel_int_n_jobs(y_continuous):
    """``parallel=<int>`` uses exactly that many workers."""
    with _silence_inner_warnings():
        warnings.simplefilter("ignore")
        m = ADAM(model="ANN").fit(y_continuous)
    boot = m.coefbootstrap(nsim=10, seed=1, parallel=2)
    assert boot.parallel is True
    assert boot.nsim_effective == 10


def test_parallel_falls_back_when_joblib_missing(y_continuous, monkeypatch):
    """If ``joblib`` is not importable, ``parallel=True`` emits a
    ``UserWarning`` and runs serially. We simulate the missing-joblib
    state by hiding the module from ``sys.modules`` and blocking new
    imports of it via a meta_path finder."""
    import sys

    # Drop any cached joblib so the import below has to consult the finders.
    for name in [k for k in sys.modules if k == "joblib" or k.startswith("joblib.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)

    class _BlockJoblib:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "joblib" or fullname.startswith("joblib."):
                raise ImportError(f"joblib import blocked for test ({fullname})")
            return None

    monkeypatch.setattr(sys, "meta_path", [_BlockJoblib(), *sys.meta_path])

    with _silence_inner_warnings():
        warnings.simplefilter("ignore")
        m = ADAM(model="ANN").fit(y_continuous)
    with pytest.warns(UserWarning, match="joblib"):
        boot = m.coefbootstrap(nsim=5, seed=1, parallel=True)
    assert boot.parallel is False
    assert boot.nsim_effective == 5


def test_invalid_method_raises(y_continuous):
    with _silence_inner_warnings():
        warnings.simplefilter("ignore")
        m = ADAM(model="ANN").fit(y_continuous)
    with pytest.raises(ValueError, match="method must be"):
        m.coefbootstrap(nsim=5, method="bogus")


def test_vcov_bootstrap_dispatches_to_coefbootstrap(y_continuous):
    """`vcov(bootstrap=True, **kwargs)` should be the same as
    coefbootstrap(**kwargs).vcov (same seed → identical result)."""
    with _silence_inner_warnings():
        warnings.simplefilter("ignore")
        m = ADAM(model="ANN").fit(y_continuous)
    v_direct = m.coefbootstrap(nsim=10, seed=99).vcov
    v_via_vcov = m.vcov(bootstrap=True, nsim=10, seed=99)
    np.testing.assert_allclose(v_direct.to_numpy(), v_via_vcov.to_numpy())


def test_confint_bootstrap_quantile_columns(y_continuous):
    """`confint(bootstrap=True)` should produce a frame shaped like the
    Fisher-based confint (S.E. + two quantile columns) but values come from
    empirical quantiles of replicates."""
    with _silence_inner_warnings():
        warnings.simplefilter("ignore")
        m = ADAM(model="ANN").fit(y_continuous)
    ci = m.confint(bootstrap=True, nsim=20, seed=11, level=0.9)
    assert list(ci.columns) == ["S.E.", "5%", "95%"]
    # Lower < Upper for each row.
    assert (ci["5%"] <= ci["95%"]).all()


def test_omg_vcov_bootstrap_dispatch(y_intermittent):
    """OMG.vcov(bootstrap=True) routes through OMG.coefbootstrap."""
    with _silence_inner_warnings():
        warnings.simplefilter("ignore")
        g = OMG(model_a="ANN", model_b="ANN").fit(y_intermittent)
    v = g.vcov(bootstrap=True, nsim=10, seed=42)
    assert isinstance(v, pd.DataFrame)
    assert v.shape[0] == v.shape[1] == len(g.coef_names)


def test_omg_confint_bootstrap_dispatch(y_intermittent):
    """OMG.confint(bootstrap=True) returns empirical-quantile CIs."""
    with _silence_inner_warnings():
        warnings.simplefilter("ignore")
        g = OMG(model_a="ANN", model_b="ANN").fit(y_intermittent)
    ci = g.confint(bootstrap=True, nsim=10, seed=42, level=0.8)
    assert list(ci.columns) == ["S.E.", "10%", "90%"]
    assert (ci["10%"] <= ci["90%"]).all()


def test_bootstrap_size_default(y_continuous):
    """Default size still reports ``floor(0.75*nobs)`` even though the
    time-series sampler doesn't consume it — the field exists for
    parity with R and the regression-pure path."""
    with _silence_inner_warnings():
        warnings.simplefilter("ignore")
        m = ADAM(model="ANN").fit(y_continuous)
    boot = m.coefbootstrap(nsim=5, seed=1)
    assert boot.size == int(np.floor(0.75 * m.nobs))


def test_bootstrap_replace_path_size_validates(y_continuous):
    """When the caller explicitly requests iid case resampling
    (``replace=True``), ``size > nobs`` is allowed (sampling with
    replacement); without replacement (``replace=False, prob=...``),
    ``size > nobs`` raises."""
    with _silence_inner_warnings():
        warnings.simplefilter("ignore")
        m = ADAM(model="ANN").fit(y_continuous)
    prob = np.ones(m.nobs) / m.nobs
    with pytest.raises(ValueError, match="size"):
        m.coefbootstrap(nsim=2, size=m.nobs + 50, replace=False, prob=prob, seed=0)
