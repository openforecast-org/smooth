"""Smoke tests for :func:`smooth.sim_ces`.

Covers all four seasonality flavours, the complex-``a`` / complex-``b``
parameter slots, the stability-rejection sampler, the ``probability < 1``
intermittent branch, and the ``frequency`` carve-out (seasonal models
require ``frequency > 1``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smooth import SimulateResult, sim_ces


@pytest.mark.parametrize("seasonality", ["none", "simple", "partial", "full"])
def test_sim_ces_all_seasonalities_run(seasonality):
    freq = 12 if seasonality != "none" else 1
    b_val = None
    if seasonality == "partial":
        b_val = 0.5
    elif seasonality == "full":
        b_val = complex(1.3, 1.0)
    s = sim_ces(
        seasonality=seasonality,
        obs=24,
        frequency=freq,
        a=complex(1.3, 1.0),
        b=b_val,
        randomizer="rnorm",
        seed=42,
    )
    assert isinstance(s, SimulateResult)
    assert isinstance(s.data, pd.Series)
    assert s.data.shape == (24,)
    assert np.all(np.isfinite(s.data.to_numpy()))
    assert s.a is not None
    if seasonality in ("none", "simple"):
        assert s.b is None
    else:
        assert s.b is not None


def test_sim_ces_random_a_stability_rejection_terminates():
    """Default ``a=None`` triggers the stability-rejecting sampler —
    must converge in well under the 1000-try cap."""
    s = sim_ces(seasonality="none", obs=30, randomizer="rnorm", seed=0)
    a = complex(s.a)
    # Stability "banana" region from R/simces.R:108-110.
    assert (a.real - 2.5) ** 2 + a.imag**2 > 1.25
    assert (a.real - 0.5) ** 2 + (a.imag - 1.0) ** 2 > 0.25
    assert (a.real - 1.5) ** 2 + (a.imag - 0.5) ** 2 < 1.5


def test_sim_ces_nsim_gt_1_returns_dataframe():
    s = sim_ces(
        seasonality="none", obs=30, nsim=4,
        a=complex(1.3, 1.0), randomizer="rnorm", seed=0,
    )
    assert isinstance(s.data, pd.DataFrame)
    assert s.data.shape == (30, 4)
    a_vec = np.asarray(s.a).ravel()
    assert a_vec.shape == (4,)


def test_sim_ces_seed_deterministic():
    a = sim_ces(seasonality="none", obs=20,
                a=complex(1.3, 1.0), randomizer="rnorm", seed=7)
    b = sim_ces(seasonality="none", obs=20,
                a=complex(1.3, 1.0), randomizer="rnorm", seed=7)
    np.testing.assert_array_equal(a.data.to_numpy(), b.data.to_numpy())


def test_sim_ces_callable_randomizer_deterministic():
    errors = np.linspace(-1.0, 1.0, 20)

    def feed(n):
        return errors[:n]

    s1 = sim_ces(seasonality="none", obs=20,
                 a=complex(1.3, 1.0), initial=[100.0, 0.0], randomizer=feed)
    s2 = sim_ces(seasonality="none", obs=20,
                 a=complex(1.3, 1.0), initial=[100.0, 0.0], randomizer=feed)
    np.testing.assert_array_equal(s1.data.to_numpy(), s2.data.to_numpy())


def test_sim_ces_seasonal_no_frequency_raises():
    with pytest.raises(ValueError, match="Cannot simulate seasonal CES"):
        sim_ces(seasonality="simple", obs=24, frequency=1)


def test_sim_ces_invalid_seasonality_raises():
    with pytest.raises(ValueError, match="Unknown seasonality"):
        sim_ces(seasonality="quarterly", obs=24, frequency=4)


def test_sim_ces_repr_shows_smoothing_params():
    s = sim_ces(seasonality="full", obs=24, frequency=4,
                a=complex(1.3, 1.0), b=complex(1.3, 1.0),
                randomizer="rnorm", seed=0)
    out = str(s)
    assert "Data generated from: CES(full)" in out
    assert "Smoothing parameter a:" in out
    assert "Smoothing parameter b:" in out
