"""Smoke tests for :func:`smooth.sim_sma`.

SMA is a thin wrapper over :func:`sim_ssarima`; these tests verify
shape contracts, the model-label substitution (``"SMA(<order>)"``),
and the ``arma`` / ``constant`` field-clearing that R does in its
post-processing block (R/simsma.R:87-89).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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
