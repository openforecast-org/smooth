"""Tests for the backcasting head (``head_length``).

Python mirror of ``tests/testthat/test_backcastHead.R``. The head is the
zero-error stretch that backcasting produces before the sample; ``head_length``
controls whether the final forward pass filters over the model's own backcasts.
"""

import numpy as np
import pandas as pd
import pytest

from smooth import ADAM, CES
from smooth.adam_general.core.creator.architector import adam_head_length

M = 12


@pytest.fixture(scope="module")
def y():
    """A seasonal series with a trend, long enough for several cycles."""
    rng = np.random.default_rng(11)
    t = np.arange(144)
    values = (
        120
        + 0.8 * t
        + 12 * np.sin(2 * np.pi * t / M)
        + np.cumsum(rng.normal(0, 1.2, t.size))
    )
    return pd.Series(values)


def _fitted(model):
    return np.asarray(model.fitted, dtype=float).ravel()


def _fit(y, head_length, model="AAdA", phi=0.9, **kwargs):
    return ADAM(
        model=model,
        lags=[1, M],
        persistence={"level": 0.3, "trend": 0.1, "seasonal": 0.2},
        phi=phi,
        initial="backcasting",
        head_length=head_length,
        **kwargs,
    ).fit(y)


# 1. head_length=0 reproduces the legacy zero-error head, the default filters it
def test_head_length_switches_the_filtering_on_and_off(y):
    default = _fit(y, None)
    legacy = _fit(y, 0)
    cycle = _fit(y, M)

    np.testing.assert_allclose(_fitted(default), _fitted(cycle), rtol=0, atol=0)
    assert not np.allclose(_fitted(default), _fitted(legacy))


# 2. Where the trend flip is already the exact reversal, the head changes nothing
def test_head_filtering_is_a_noop_for_ets_without_damped_trend(y):
    common = dict(
        model="AAA",
        lags=[1, M],
        persistence={"level": 0.3, "trend": 0.1, "seasonal": 0.2},
        initial="backcasting",
    )
    default = ADAM(**common).fit(y)
    legacy = ADAM(**common, head_length=0).fit(y)

    np.testing.assert_allclose(_fitted(default), _fitted(legacy), rtol=0, atol=0)


# 3. The states keep their shape whatever the head length
def test_states_keep_their_shape_whatever_the_head_length(y):
    cycle = _fit(y, M, model="AAA", phi=None)
    long = _fit(y, 2 * M, model="AAA", phi=None)

    assert np.asarray(cycle.states).shape == np.asarray(long.states).shape


# 4. The C++ kernel gets the resolved flag, and the drift is never flipped
def test_architector_sets_the_kernel_flag_and_never_flips_the_drift(y):
    default = _fit(y, None)
    assert default._adam_cpp.headLength == M
    assert default._adam_cpp.flipConstant is False

    legacy = _fit(y, 0)
    assert legacy._adam_cpp.headLength == 0

    # An odd total order of differencing used to flip the drift; ADAM carries the
    # constant in the measurement vector, so it must not.
    arima = ADAM(
        model="NNN",
        lags=[1],
        orders={"ar": [0], "i": [1], "ma": [1]},
        constant=True,
        initial="backcasting",
    ).fit(y)
    assert arima._adam_cpp.flipConstant is False


# 5. CES resolves the head the same way
def test_ces_resolves_the_head_length(y):
    default = CES(seasonality="full", lags=[M], initial="backcasting").fit(y)
    legacy = CES(
        seasonality="full", lags=[M], initial="backcasting", head_length=0
    ).fit(y)

    assert not np.allclose(_fitted(default), _fitted(legacy))


# 6. The resolver itself: absent means one cycle, 0 switches the filtering off
@pytest.mark.parametrize(
    ("requested", "geometry", "flag"),
    [
        (None, 12, 12),  # absent: one full lag cycle, filtering on
        (0, 12, 0),  # explicit 0: filtering off, geometry unchanged
        (6, 12, 12),  # below the cycle: clamped up to the cycle
        (24, 24, 24),  # above the cycle: that head length
        (500, 100, 100),  # above the sample: clamped to the sample
    ],
)
def test_adam_head_length_resolver(requested, geometry, flag):
    resolved = adam_head_length(requested, lags_model_max=12, obs_in_sample=100)
    assert resolved == {"geometry": geometry, "flag": flag}
