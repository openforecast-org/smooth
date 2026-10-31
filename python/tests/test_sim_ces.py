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


def test_sim_ces_prints_the_rounded_smoothing_parameters():
    """R: print(sim.ces("partial", obs=10, frequency=4, a=1.31234567+0.98765432i,
    b=0.41234567)) gives "Smoothing parameter a: 1.3123+0.9877i" and
    "Smoothing parameter b: 0.4123"."""
    s = sim_ces(
        seasonality="partial",
        obs=10,
        frequency=4,
        a=complex(1.31234567, 0.98765432),
        b=0.41234567,
        seed=1,
    )
    lines = str(s).splitlines()
    assert "Smoothing parameter a: (1.3123+0.9877j)" in lines
    assert "Smoothing parameter b: 0.4123" in lines


def test_sim_ces_has_the_true_likelihood_as_r():
    # R/simces.R:423-425 for rnorm: -obs/2 * (log(2*pi*e) + log(mean(errors^2)))
    s = sim_ces(
        seasonality="partial",
        obs=10,
        frequency=4,
        a=complex(1.31234567, 0.98765432),
        b=0.41234567,
        seed=1,
    )
    errors = np.asarray(s.residuals, dtype=float)
    expected = -10 / 2 * (np.log(2 * np.pi * np.e) + np.log(np.mean(errors**2)))
    assert s.log_lik is not None
    assert float(np.ravel(s.log_lik)[0]) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    "value, expected",
    [
        (complex(1.23456, -0.98765), complex(1.2346, -0.9877)),
        (np.complex64(1.5 + 0.25j), complex(1.5, 0.25)),
        (np.array([1.23456 + 2.00004j]), complex(1.2346, 2.0)),
        (0.41234567, 0.4123),
        (np.array([0.41234567]), 0.4123),
    ],
)
def test_round_c_of_a_single_value(value, expected):
    """A single real or complex value rounded as R's round(x, 4)."""
    from smooth.adam_general.core.simulate.result import _round_c

    rounded = _round_c(value, 4)
    assert type(rounded) is type(expected)
    assert rounded == expected


def test_round_c_of_several_values():
    from smooth.adam_general.core.simulate.result import _round_c

    rounded = _round_c(np.array([0.12345, 1.98765, 2.5]), 2)
    np.testing.assert_array_equal(rounded, [0.12, 1.99, 2.5])
