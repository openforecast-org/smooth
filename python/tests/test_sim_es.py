"""Smoke tests for :func:`smooth.sim_es`.

Covers the public surface: shapes / dtypes / finiteness, the string-vs-
callable randomizer fork, the seed-determinism contract for string
randomizers, and the ``probability < 1`` intermittent path. R-parity
tests live in ``test_sim_es_r_parity.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smooth import SimulateResult, sim_es


def test_sim_es_ann_nsim1_returns_series():
    """nsim=1 returns a pandas Series of length ``obs``."""
    s = sim_es(
        model="ANN", obs=50, persistence=[0.3], initial=[100.0],
        randomizer="rnorm", seed=42,
    )
    assert isinstance(s, SimulateResult)
    assert isinstance(s.data, pd.Series)
    assert s.data.shape == (50,)
    assert isinstance(s.residuals, pd.Series)
    assert s.residuals.shape == (50,)
    assert s.states.shape == (1, 51, 1)
    assert np.all(np.isfinite(s.data.to_numpy()))


def test_sim_es_mam_seasonal_shapes():
    """A seasonal model returns the right state cube and a non-empty
    seasonal initial vector."""
    s = sim_es(
        model="MAM", obs=48, frequency=12,
        persistence=[0.1, 0.05, 0.05], phi=1.0,
        initial=[100.0, 1.0],
        initial_season=list(np.linspace(0.9, 1.1, 12)),
        randomizer="rnorm", seed=0,
    )
    assert s.states.shape == (3, 60, 1)  # 3 components × (obs + lag_max=12)
    assert s.initial_season is not None
    assert s.initial_season.shape == (1, 12)
    assert s.persistence.shape == (3, 1)
    assert np.all(np.isfinite(s.data.to_numpy()))


def test_sim_es_nsim_gt_1_returns_dataframe():
    s = sim_es(
        model="ANN", obs=30, nsim=5,
        persistence=[0.3], initial=[100.0],
        randomizer="rnorm", seed=0,
    )
    assert isinstance(s.data, pd.DataFrame)
    assert s.data.shape == (30, 5)
    assert s.states.shape == (1, 31, 5)
    assert np.all(np.isfinite(s.data.to_numpy()))


def test_sim_es_callable_randomizer_passes_through_verbatim():
    """A callable randomizer feeds errors **as given** — two calls
    with the same callable produce identical results."""
    errors = np.array([0.5, -0.3, 0.1, -0.2, 0.4, 0.0, -0.1, 0.3])

    def feed(n):
        return errors[:n]

    s1 = sim_es(model="ANN", obs=8, persistence=[0.3], initial=[100.0],
                randomizer=feed)
    s2 = sim_es(model="ANN", obs=8, persistence=[0.3], initial=[100.0],
                randomizer=feed)
    np.testing.assert_array_equal(s1.data.to_numpy(), s2.data.to_numpy())
    # First observation: y_1 = level_0 + e_1 (ANN simulator).
    assert s1.data.iloc[0] == pytest.approx(100.0 + 0.5)


def test_sim_es_callable_too_short_raises():
    def feed_short(n):
        return np.zeros(3)  # not enough for obs=10

    with pytest.raises(ValueError, match="returned 3 values, need at least"):
        sim_es(model="ANN", obs=10, persistence=[0.3], initial=[100.0],
               randomizer=feed_short)


def test_sim_es_string_randomizer_seed_deterministic():
    """Same string randomizer + same seed → identical output."""
    s1 = sim_es(model="ANN", obs=20, persistence=[0.3], initial=[100.0],
                randomizer="rnorm", seed=123)
    s2 = sim_es(model="ANN", obs=20, persistence=[0.3], initial=[100.0],
                randomizer="rnorm", seed=123)
    np.testing.assert_array_equal(s1.data.to_numpy(), s2.data.to_numpy())


def test_sim_es_intermittent_probability_below_one():
    """probability < 1 produces an occurrence mask and ``iETS(...)`` label."""
    s = sim_es(
        model="ANN", obs=80, persistence=[0.3], initial=[5.0],
        randomizer="rnorm", probability=0.4, seed=7,
    )
    assert s.model.startswith("iETS(")
    assert s.occurrence is not None
    assert s.occurrence.shape == (80, 1)
    # Most observations should be zero (probability ≈ 0.4).
    nonzero_fraction = float((s.data.to_numpy() != 0).mean())
    assert 0.2 < nonzero_fraction < 0.7


def test_sim_es_repr_matches_r_layout():
    """Single-series ETS repr lists model, nsim, persistence, true logLik."""
    s = sim_es(
        model="ANN", obs=20, persistence=[0.3], initial=[100.0],
        randomizer="rnorm", seed=0,
    )
    out = str(s)
    assert "Data generated from: ETS(ANN)" in out
    assert "Number of generated series: 1" in out
    assert "Persistence vector:" in out
    assert "alpha" in out
    assert "True likelihood:" in out


def test_sim_es_invalid_model_raises():
    with pytest.raises(ValueError, match="strange model"):
        sim_es(model="AA", obs=10)


def test_sim_es_seasonal_no_frequency_raises():
    with pytest.raises(ValueError, match="Cannot create the seasonal model"):
        sim_es(model="ANA", obs=24, frequency=1)
