"""Smoke tests for the Python port of ``multicov``.

These cover the analytical and simulated paths on ADAM / ES / MSARIMA / OM
and the "not defined" raise on OMG. They stay in the default suite — fast,
deterministic on the analytical side, and noisy-but-bounded on the
simulated side (we only check shape / symmetry / PSD, not exact values).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from smooth import ADAM, ES, MSARIMA, OM, OMG


@pytest.fixture(scope="module")
def y_continuous() -> np.ndarray:
    rng = np.random.default_rng(0)
    return (100 + np.cumsum(rng.standard_normal(120))).astype(float)


@pytest.fixture(scope="module")
def y_intermittent() -> np.ndarray:
    rng = np.random.default_rng(0)
    y = rng.poisson(0.5, 200).astype(float)
    y[0] = 1.0
    y[1] = 0.0
    return y


def _silence():
    return warnings.catch_warnings()


def _is_symmetric(m: np.ndarray, atol: float = 1e-9) -> bool:
    return bool(np.allclose(m, m.T, atol=atol))


def _is_psd(m: np.ndarray, atol: float = 1e-8) -> bool:
    sym = (m + m.T) / 2
    eig = np.linalg.eigvalsh(sym)
    return bool(np.all(eig >= -atol))


# --- Analytical path ---------------------------------------------------------


def test_adam_multicov_analytical_shape_and_labels(y_continuous):
    with _silence():
        warnings.simplefilter("ignore")
        m = ADAM(model="ANN").fit(y_continuous)
    M = m.multicov(h=5)
    assert isinstance(M, pd.DataFrame)
    assert M.shape == (5, 5)
    assert list(M.columns) == [f"h{i + 1}" for i in range(5)]
    assert list(M.index) == [f"h{i + 1}" for i in range(5)]


def test_adam_multicov_analytical_symmetric_and_psd(y_continuous):
    with _silence():
        warnings.simplefilter("ignore")
        m = ADAM(model="AAN").fit(y_continuous)
    M = m.multicov(h=6).to_numpy()
    assert _is_symmetric(M)
    assert _is_psd(M)


def test_adam_multicov_analytical_diag_monotone_nondecreasing(y_continuous):
    """For a stationary ANN with σ² > 0, the diagonal variance should
    be non-decreasing in horizon (variance accumulates)."""
    with _silence():
        warnings.simplefilter("ignore")
        m = ADAM(model="ANN").fit(y_continuous)
    diag = np.diag(m.multicov(h=8).to_numpy())
    assert np.all(np.diff(diag) >= -1e-9)


def test_es_inherits_multicov(y_continuous):
    with _silence():
        warnings.simplefilter("ignore")
        es = ES(model="AAN").fit(y_continuous)
    M = es.multicov(h=4)
    assert M.shape == (4, 4)
    assert _is_symmetric(M.to_numpy())


def test_msarima_inherits_multicov(y_continuous):
    with _silence():
        warnings.simplefilter("ignore")
        ar = MSARIMA(ar_order=1, ma_order=1).fit(y_continuous)
    M = ar.multicov(h=4)
    assert M.shape == (4, 4)
    assert _is_symmetric(M.to_numpy())


def test_om_multicov_analytical_finite(y_intermittent):
    """OM's sigma is the link-scale residual std-dev
    (``sqrt(mean(residuals²))``), mirroring R's ``sigma.om`` (R/om.R).
    multicov on that scale is finite, PSD, and symmetric — interpreted
    as the multi-step covariance of the **link-transformed** forecast
    errors (logit / log-odds), not on the probability axis."""
    with _silence():
        warnings.simplefilter("ignore")
        m = OM(model="MNN", occurrence="odds-ratio").fit(y_intermittent)
    M = m.multicov(h=4).to_numpy()
    assert M.shape == (4, 4)
    assert np.all(np.isfinite(M))
    assert _is_symmetric(M)
    assert _is_psd(M)
    # Diagonal scales with σ²; σ should match `sqrt(mean(residuals²))`
    # (R/oes.R:1253 — `output$s2 <- mean(residuals²)`).
    expected_sigma = np.sqrt(np.mean(np.asarray(m.residuals) ** 2))
    np.testing.assert_allclose(m.sigma, expected_sigma)


# --- Simulated path ----------------------------------------------------------


def test_adam_multicov_simulated_shape_and_symmetry(y_continuous):
    with _silence():
        warnings.simplefilter("ignore")
        m = ADAM(model="ANN").fit(y_continuous)
    M = m.multicov(type="simulated", h=4, nsim=300).to_numpy()
    assert M.shape == (4, 4)
    assert _is_symmetric(M, atol=1e-6)
    # PSD with a slightly looser tolerance — Monte Carlo noise can create
    # tiny negative eigenvalues at modest nsim.
    assert _is_psd(M, atol=1e-4)


def test_simulated_does_not_corrupt_model_state(y_continuous):
    """multicov(type='simulated') drives `predict(scenarios=True)`
    internally; the saved/restored state should leave subsequent
    `predict()` calls undisturbed."""
    with _silence():
        warnings.simplefilter("ignore")
        m = ADAM(model="ANN").fit(y_continuous)
    fc_before = m.predict(h=3).mean.to_numpy()
    _ = m.multicov(type="simulated", h=4, nsim=100)
    fc_after = m.predict(h=3).mean.to_numpy()
    np.testing.assert_allclose(fc_before, fc_after)


# --- Edge cases & errors -----------------------------------------------------


def test_adam_multicov_empirical_shape_and_labels(y_continuous):
    with _silence():
        warnings.simplefilter("ignore")
        m = ADAM(model="AAN").fit(y_continuous)
    M = m.multicov(type="empirical", h=4)
    assert isinstance(M, pd.DataFrame)
    assert M.shape == (4, 4)
    assert list(M.columns) == [f"h{i + 1}" for i in range(4)]
    assert list(M.index) == [f"h{i + 1}" for i in range(4)]
    arr = M.to_numpy()
    assert _is_symmetric(arr)
    assert _is_psd(arr)


def test_adam_multicov_empirical_matches_rmultistep_formula(y_continuous):
    """Internal-consistency check: multicov(empirical) is literally
    ``(errorsᵀ errors) / (nobs - h)`` where ``errors = rmultistep(h)`` —
    R/adam.R:7090-7092."""
    with _silence():
        warnings.simplefilter("ignore")
        m = ADAM(model="ANN").fit(y_continuous)
    h = 5
    M = m.multicov(type="empirical", h=h).to_numpy()
    errors = m.rmultistep(h=h).to_numpy()
    expected = (errors.T @ errors) / (m.nobs - h)
    np.testing.assert_allclose(M, expected, rtol=1e-12, atol=1e-12)


def test_om_multicov_empirical_finite(y_intermittent):
    """OM inherits multicov; the empirical branch routes through
    ``rmultistep`` which works on OM (link-scale errors), producing a
    finite, symmetric, PSD covariance."""
    with _silence():
        warnings.simplefilter("ignore")
        m = OM(model="MNN", occurrence="odds-ratio").fit(y_intermittent)
    M = m.multicov(type="empirical", h=4).to_numpy()
    assert M.shape == (4, 4)
    assert np.all(np.isfinite(M))
    assert _is_symmetric(M)
    assert _is_psd(M)


def test_invalid_type_raises(y_continuous):
    with _silence():
        warnings.simplefilter("ignore")
        m = ADAM(model="ANN").fit(y_continuous)
    with pytest.raises(ValueError, match="type must be"):
        m.multicov(type="bogus", h=3)


def test_invalid_h_raises(y_continuous):
    with _silence():
        warnings.simplefilter("ignore")
        m = ADAM(model="ANN").fit(y_continuous)
    with pytest.raises(ValueError, match="h must be"):
        m.multicov(h=0)


def test_invalid_nsim_raises(y_continuous):
    with _silence():
        warnings.simplefilter("ignore")
        m = ADAM(model="ANN").fit(y_continuous)
    with pytest.raises(ValueError, match="nsim must be"):
        m.multicov(type="simulated", h=3, nsim=0)


def test_omg_multicov_raises(y_intermittent):
    """OMG doesn't have a meaningful joint multicov; method must raise
    with a clear redirect to the sub-models."""
    with _silence():
        warnings.simplefilter("ignore")
        g = OMG(model_a="ANN", model_b="ANN").fit(y_intermittent)
    with pytest.raises(NotImplementedError, match="model_a.multicov"):
        g.multicov(h=3)


def test_omg_submodel_multicov_works(y_intermittent):
    """OMG sub-models inherit multicov from ADAM. NaN-filled (no σ),
    but shape/dispatch must be correct."""
    with _silence():
        warnings.simplefilter("ignore")
        g = OMG(model_a="ANN", model_b="ANN").fit(y_intermittent)
    Ma = g.model_a.multicov(h=3)
    assert Ma.shape == (3, 3)
    assert list(Ma.columns) == ["h1", "h2", "h3"]
