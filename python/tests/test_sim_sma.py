"""Smoke tests for :func:`smooth.sim_sma`.

SMA is a thin wrapper over :func:`sim_ssarima`; these tests verify
shape contracts, the model-label substitution (``"SMA(<order>)"``),
and the ``arma`` / ``constant`` field-clearing that R does in its
post-processing block (R/simsma.R:87-89).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smooth import sim_sma


def test_sim_sma_order_5_returns_series():
    s = sim_sma(order=5, obs=50, initial=[10.0] * 5, randomizer="rnorm", seed=42)
    assert isinstance(s.data, pd.Series)
    assert s.data.shape == (50,)
    assert s.states.shape == (5, 51, 1)
    assert s.model == "SMA(5)"
    # R's post-processing strips arma / constant from the SMA result.
    assert s.arma is None
    assert s.constant is None
    assert np.all(np.isfinite(s.data.to_numpy()))


def test_sim_sma_nsim_gt_1_dataframe():
    s = sim_sma(order=3, obs=40, nsim=3, initial=[5.0, 5.0, 5.0],
                randomizer="rnorm", seed=0)
    assert isinstance(s.data, pd.DataFrame)
    assert s.data.shape == (40, 3)
    assert s.states.shape == (3, 41, 3)
    assert np.all(np.isfinite(s.data.to_numpy()))


def test_sim_sma_callable_randomizer_deterministic():
    errors = np.linspace(-1.0, 1.0, 30)

    def feed(n):
        return errors[:n]

    s1 = sim_sma(order=3, obs=30, initial=[0.0, 0.0, 0.0], randomizer=feed)
    s2 = sim_sma(order=3, obs=30, initial=[0.0, 0.0, 0.0], randomizer=feed)
    np.testing.assert_array_equal(s1.data.to_numpy(), s2.data.to_numpy())


def test_sim_sma_intermittent_label():
    s = sim_sma(order=2, obs=50, initial=[1.0, 1.0],
                probability=0.3, randomizer="rnorm", seed=0)
    assert s.model.startswith("iSMA")


# Zero errors: the true likelihood takes log(0), Inf as in R
_ZERO_ERRORS = pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")


@_ZERO_ERRORS
def test_sim_sma_without_errors_is_that_of_r():
    """R: sim.sma(order=2, obs=6, initial=c(2,4), randomizer="rnorm", mean=0,
    sd=0): each value the mean of the two before it, and logLik Inf."""
    s = sim_sma(
        order=2, obs=6, initial=[2.0, 4.0], randomizer="rnorm", mean=0, sd=0, seed=1
    )
    np.testing.assert_allclose(
        s.data.to_numpy(), [2.0, 5.0, 3.5, 4.25, 3.875, 4.0625], rtol=1e-14
    )
    assert np.ravel(s.log_lik).tolist() == [np.inf]


def test_sim_sma_draws_the_order_as_r():
    """Without an order, ceiling(runif(1, 0, 100)), as R."""
    expected = int(np.ceil(np.random.default_rng(3).uniform(0.0, 100.0)))
    s = sim_sma(obs=5, seed=3)
    assert s.model == f"SMA({expected})"
    assert s.states.shape[0] == expected


def test_sim_sma_rounds_the_order():
    assert sim_sma(order=-2.6, obs=5, seed=0).model == "SMA(3)"


def test_sim_sma_zero_order_raises():
    with pytest.raises(ValueError, match="SMA order must be >= 1"):
        sim_sma(order=0.4, obs=5)


@_ZERO_ERRORS
def test_sim_sma_initial_of_another_length_falls_back_to_random():
    """As R's warning: "The length of initial state vector does not correspond to
    the chosen model! Falling back to random number generator." """
    with pytest.warns(UserWarning, match="does not correspond.*Falling back"):
        s = sim_sma(
            order=2,
            obs=6,
            initial=[2.0, 4.0, 5.0],
            randomizer="rnorm",
            mean=0,
            sd=0,
            seed=1,
        )
    # Not the provided initial: the values of a random one
    assert not np.allclose(s.data.to_numpy(), [2.0, 5.0, 3.5, 4.25, 3.875, 4.0625])
    plain = sim_sma(order=2, obs=6, randomizer="rnorm", mean=0, sd=0, seed=1)
    np.testing.assert_array_equal(s.data.to_numpy(), plain.data.to_numpy())


@pytest.mark.parametrize(
    "probability, label",
    [
        (np.full(20, 1.0), "SMA(2)"),
        (np.r_[np.full(10, 1.0), np.full(10, 0.5)], "iSMA(2)"),
    ],
)
def test_sim_sma_label_with_a_vector_of_probabilities(probability, label):
    s = sim_sma(
        order=2,
        obs=20,
        initial=[1.0, 1.0],
        probability=probability,
        randomizer="rnorm",
        seed=0,
    )
    assert s.model == label
