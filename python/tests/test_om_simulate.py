"""Smoke tests for :meth:`smooth.OM.simulate` and :meth:`smooth.OMG.simulate`.

Covers the public surface: shape contracts, seed determinism, occurrence
scheme inheritance, OMG sub-model field access. Also pins the four
feature axes the simulator must propagate through to the latent ETS:
multi-frequency seasonal lags, xreg, ARMA orders, and OMG composition.

R-parity tests live in ``test_om_simulate_r_parity.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smooth import OM, OMG, SimulateResult


@pytest.fixture
def y_intermittent():
    """Length-60 intermittent series with gentle upward drift."""
    rng = np.random.default_rng(42)
    return rng.binomial(1, np.clip(0.3 + 0.005 * np.arange(60), 0.05, 0.95))


@pytest.fixture
def y_seasonal_intermittent():
    """Length-72 (6-year monthly) intermittent series with a sinusoid."""
    rng = np.random.default_rng(0)
    n = 72
    season = 0.2 * np.sin(2 * np.pi * np.arange(n) / 12)
    return rng.binomial(1, np.clip(0.3 + season, 0.05, 0.95))


# ----------------------------------------------------------------------
# Baseline: shape contracts + seed determinism
# ----------------------------------------------------------------------


def test_om_simulate_shapes(y_intermittent):
    m = OM(model="MNN", occurrence="odds-ratio").fit(y_intermittent)
    sim = m.simulate(nsim=3, seed=42)
    assert isinstance(sim, SimulateResult)
    assert np.asarray(sim.probability).shape == (60, 3)
    assert sim.occurrence.shape == (60, 3)
    assert set(np.unique(sim.occurrence).astype(int).tolist()) <= {0, 1}
    arr = np.asarray(sim.probability)
    assert (arr >= 0).all() and (arr <= 1).all()
    assert sim.latent is not None and sim.latent.shape == (60, 3)
    assert sim.occurrence_type == "odds-ratio"


def test_om_simulate_seed_deterministic(y_intermittent):
    m = OM(model="MNN", occurrence="odds-ratio").fit(y_intermittent)
    a = m.simulate(nsim=2, seed=7)
    b = m.simulate(nsim=2, seed=7)
    np.testing.assert_array_equal(
        np.asarray(a.probability), np.asarray(b.probability)
    )
    np.testing.assert_array_equal(a.occurrence, b.occurrence)


def test_om_simulate_nsim_1_returns_series(y_intermittent):
    m = OM(model="MNN", occurrence="direct").fit(y_intermittent)
    sim = m.simulate(nsim=1, seed=0)
    assert isinstance(sim.data, pd.Series)
    assert sim.data.shape == (60,)


# ----------------------------------------------------------------------
# Feature axis: multi-frequency / seasonal lags
# ----------------------------------------------------------------------


def test_om_simulate_seasonal_lags(y_seasonal_intermittent):
    """Seasonal OM (lags=[1, 12]) — the latent ETS uses a 12-step
    seasonal component; ``OM.simulate`` must propagate that through
    ``ADAM.simulate`` without breaking the shape.
    """
    m = OM(model="MNM", lags=[1, 12], occurrence="odds-ratio").fit(
        y_seasonal_intermittent
    )
    sim = m.simulate(nsim=2, seed=0)
    arr = np.asarray(sim.probability)
    assert arr.shape == (72, 2)
    assert sim.latent.shape == (72, 2)
    assert (arr >= 0).all() and (arr <= 1).all()


# ----------------------------------------------------------------------
# Feature axis: external regressors (xreg)
# ----------------------------------------------------------------------


def test_om_simulate_with_xreg(y_intermittent):
    """xreg-fitted OM — the ADAM.simulate path already handles xreg;
    OM.simulate must pass through unchanged."""
    n = len(y_intermittent)
    rng = np.random.default_rng(1)
    X = rng.standard_normal((n, 2))
    m = OM(model="MNN", occurrence="odds-ratio").fit(y_intermittent, X=X)
    sim = m.simulate(nsim=2, seed=0)
    assert np.asarray(sim.probability).shape == (n, 2)
    assert sim.latent.shape == (n, 2)


# ----------------------------------------------------------------------
# Feature axis: ARMA orders
# ----------------------------------------------------------------------


def test_om_simulate_with_arima_orders(y_intermittent):
    """OM with ARMA(1, 0, 1) in the latent ETS — the ARIMA path
    through ADAM.simulate must inherit cleanly."""
    m = OM(
        model="MNN",
        orders={"ar": [1], "i": [0], "ma": [1]},
        occurrence="odds-ratio",
    ).fit(y_intermittent)
    sim = m.simulate(nsim=2, seed=0)
    assert np.asarray(sim.probability).shape == (60, 2)
    assert sim.latent.shape == (60, 2)


# ----------------------------------------------------------------------
# OMG composite
# ----------------------------------------------------------------------


def test_omg_simulate_shapes_and_submodels(y_intermittent):
    m = OMG(model_a="MNN", model_b="MNN").fit(y_intermittent)
    sim = m.simulate(nsim=2, seed=42)
    assert isinstance(sim, SimulateResult)
    arr = np.asarray(sim.probability)
    assert arr.shape == (60, 2)
    assert sim.occurrence.shape == (60, 2)
    assert (arr >= 0).all() and (arr <= 1).all()
    assert sim.model_a is not None and sim.model_b is not None
    assert sim.model_a.latent.shape == (60, 2)
    assert sim.model_b.latent.shape == (60, 2)
    assert sim.occurrence_type == "general"


def test_omg_simulate_seed_deterministic(y_intermittent):
    m = OMG(model_a="MNN", model_b="MNN").fit(y_intermittent)
    a = m.simulate(nsim=2, seed=7)
    b = m.simulate(nsim=2, seed=7)
    np.testing.assert_array_equal(
        np.asarray(a.probability), np.asarray(b.probability)
    )


def test_omg_simulate_seasonal_lags(y_seasonal_intermittent):
    """Seasonal OMG — both sub-models carry the seasonal lag."""
    m = OMG(model_a="MNM", model_b="MNM", lags=[1, 12]).fit(
        y_seasonal_intermittent
    )
    sim = m.simulate(nsim=2, seed=0)
    arr = np.asarray(sim.probability)
    assert arr.shape == (72, 2)
    assert (arr >= 0).all() and (arr <= 1).all()
