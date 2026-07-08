"""Smoke tests for :func:`smooth.sim_gum`.

Covers fixed-matrix vs random-stability-sampled paths, the callable
randomizer pass-through, shape contracts, and the ``probability < 1``
intermittent branch. R-parity tests live in ``test_sim_gum_r_parity.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smooth import SimulateResult, sim_gum


def test_sim_gum_fixed_matrices():
    """With explicit measurement / transition / persistence the sampler
    skips the rejection loop and uses the supplied values verbatim."""
    s = sim_gum(
        orders=[2], lags=[1], obs=30,
        measurement=[1.0, 0.0],
        transition=[1.0, 0.0, 0.5, 0.5],
        persistence=[0.3, 0.1],
        initial=[100.0, 1.0],
        randomizer="rnorm", seed=42,
    )
    assert isinstance(s, SimulateResult)
    assert s.data.shape == (30,)
    assert np.all(np.isfinite(s.data.to_numpy()))
    # measurement / transition / persistence stored on the result
    assert s.measurement is not None
    assert s.transition is not None
    assert s.persistence is not None
    np.testing.assert_allclose(s.measurement[0], [1.0, 0.0])
    # R's column-major reshape of c(1, 0, 0.5, 0.5):
    np.testing.assert_allclose(s.transition, [[1.0, 0.5], [0.0, 0.5]])


def test_sim_gum_random_stability_sampler_terminates():
    """When matrices are ``None``, the eigenvalue-rejection sampler must
    converge well under the 2000-try cap for a small model."""
    s = sim_gum(orders=[1], lags=[1], obs=30, randomizer="rnorm", seed=42)
    assert np.all(np.isfinite(s.data.to_numpy()))


def test_sim_gum_seasonal_lags():
    s = sim_gum(
        orders=[1, 1], lags=[1, 4], obs=24,
        measurement=[1.0, 1.0],
        transition=[1.0, 0.0, 0.0, 1.0],
        persistence=[0.3, 0.2],
        initial=[100.0] * (2 * 4),  # components × lag_max
        randomizer="rnorm", seed=0,
    )
    assert s.data.shape == (24,)
    assert s.states.shape[0] == 2  # 2 components
    assert s.states.shape[1] == 24 + 4  # obs + lag_max


def test_sim_gum_nsim_gt_1_returns_dataframe():
    s = sim_gum(
        orders=[1], lags=[1], obs=30, nsim=3,
        measurement=[1.0], transition=[0.7], persistence=[0.3],
        initial=[10.0], randomizer="rnorm", seed=0,
    )
    assert isinstance(s.data, pd.DataFrame)
    assert s.data.shape == (30, 3)
    assert s.persistence.shape == (1, 3)


def test_sim_gum_callable_randomizer_deterministic():
    errors = np.linspace(-1.0, 1.0, 25)

    def feed(n):
        return errors[:n]

    s1 = sim_gum(orders=[1], lags=[1], obs=25,
                 measurement=[1.0], transition=[0.5], persistence=[0.3],
                 initial=[10.0], randomizer=feed)
    s2 = sim_gum(orders=[1], lags=[1], obs=25,
                 measurement=[1.0], transition=[0.5], persistence=[0.3],
                 initial=[10.0], randomizer=feed)
    np.testing.assert_array_equal(s1.data.to_numpy(), s2.data.to_numpy())


def test_sim_gum_invalid_orders_raises():
    with pytest.raises(ValueError, match="Negative orders"):
        sim_gum(orders=[-1], lags=[1], obs=10)


def test_sim_gum_mismatched_orders_lags_raises():
    with pytest.raises(ValueError, match="length"):
        sim_gum(orders=[1, 1], lags=[1], obs=10)


def test_sim_gum_repr_layout():
    s = sim_gum(orders=[1], lags=[1], obs=20,
                measurement=[1.0], transition=[0.5], persistence=[0.3],
                initial=[10.0], randomizer="rnorm", seed=0)
    out = str(s)
    assert "Data generated from: GUM(1[1])" in out
    assert "True likelihood:" in out
