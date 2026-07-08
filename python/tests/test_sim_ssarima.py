"""Smoke tests for :func:`smooth.sim_ssarima`.

Covers shape contracts, AR(1) / IMA(1,1) / SARIMA(1,0,0)(1,0,0)[12],
the callable-randomizer pass-through, ``constant=...`` flag, and the
``probability < 1`` intermittent branch. R-parity tests live in
``test_sim_ssarima_r_parity.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smooth import SimulateResult, sim_ssarima


def test_sim_ssarima_ar1_nsim1_shapes():
    s = sim_ssarima(
        orders={"ar": [1], "i": [0], "ma": [0]},
        lags=[1],
        obs=50,
        arma={"ar": [0.7], "ma": []},
        initial=[10.0],
        randomizer="rnorm",
        seed=42,
    )
    assert isinstance(s, SimulateResult)
    assert isinstance(s.data, pd.Series)
    assert s.data.shape == (50,)
    assert s.states.shape == (1, 51, 1)
    assert s.arma is not None
    assert float(s.arma["ar"][0, 0]) == pytest.approx(0.7)
    assert np.all(np.isfinite(s.data.to_numpy()))


def test_sim_ssarima_ima11_finite():
    s = sim_ssarima(
        orders={"ar": [0], "i": [1], "ma": [1]},
        lags=[1],
        obs=40,
        arma={"ar": [], "ma": [0.6]},
        initial=[100.0],
        randomizer="rnorm",
        seed=0,
    )
    assert s.model == "ARIMA(0,1,1)"
    assert s.data.shape == (40,)
    assert np.all(np.isfinite(s.data.to_numpy()))


def test_sim_ssarima_sarima_shapes():
    s = sim_ssarima(
        orders={"ar": [1, 1], "i": [0, 0], "ma": [0, 0]},
        lags=[1, 12],
        obs=72,
        arma={"ar": [0.4, 0.3], "ma": []},
        initial=list(np.full(13, 10.0)),
        randomizer="rnorm",
        seed=0,
    )
    assert s.data.shape == (72,)
    assert s.states.shape == (13, 73, 1)
    assert "SARIMA" in s.model
    assert np.all(np.isfinite(s.data.to_numpy()))


def test_sim_ssarima_nsim_gt_1_dataframe():
    s = sim_ssarima(
        orders={"ar": [1], "i": [0], "ma": [1]},
        lags=[1],
        obs=30,
        nsim=4,
        arma={"ar": [0.5], "ma": [0.3]},
        initial=[5.0],
        randomizer="rnorm",
        seed=0,
    )
    assert isinstance(s.data, pd.DataFrame)
    assert s.data.shape == (30, 4)
    assert np.all(np.isfinite(s.data.to_numpy()))


def test_sim_ssarima_callable_randomizer_deterministic():
    errors = np.linspace(-1.0, 1.0, 30)

    def feed(n):
        return errors[:n]

    s1 = sim_ssarima(
        orders={"ar": [1], "i": [0], "ma": [0]},
        lags=[1],
        obs=30,
        arma={"ar": [0.5], "ma": []},
        initial=[5.0],
        randomizer=feed,
    )
    s2 = sim_ssarima(
        orders={"ar": [1], "i": [0], "ma": [0]},
        lags=[1],
        obs=30,
        arma={"ar": [0.5], "ma": []},
        initial=[5.0],
        randomizer=feed,
    )
    np.testing.assert_array_equal(s1.data.to_numpy(), s2.data.to_numpy())


def test_sim_ssarima_constant_numeric_value_propagates():
    s = sim_ssarima(
        orders={"ar": [0], "i": [1], "ma": [0]},
        lags=[1],
        obs=30,
        constant=5.0,
        initial=[100.0],
        randomizer="rnorm",
        seed=0,
    )
    assert s.constant == pytest.approx(5.0)
    assert "with drift" in s.model  # I-order > 0 ⇒ "drift" label


def test_sim_ssarima_invalid_negative_orders_raises():
    with pytest.raises(ValueError, match="Negative ARIMA orders"):
        sim_ssarima(orders={"ar": [-1], "i": [0], "ma": [0]}, lags=[1])


def test_sim_ssarima_intermittent_probability_below_one():
    s = sim_ssarima(
        orders={"ar": [1], "i": [0], "ma": [0]},
        lags=[1],
        obs=80,
        arma={"ar": [0.7], "ma": []},
        initial=[5.0],
        randomizer="rnorm",
        probability=0.3,
        seed=7,
    )
    assert s.model.startswith("iARIMA")
    assert s.occurrence is not None
    assert s.occurrence.shape == (80, 1)
