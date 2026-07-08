"""Smoke tests for :meth:`smooth.ADAM.simulate` (and inherited subclasses).

Verifies the fitted-model-driven simulation path: shape contracts,
seed determinism, inheritance to ``ES`` / ``MSARIMA`` / ``SMA``, and
the R-style ``" estimated via adam()"`` model-label suffix.
"""

from __future__ import annotations

import numpy as np

from smooth import ADAM, ES, MSARIMA, SMA

AIRPASSENGERS = np.array(
    [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
     115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
     145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
     171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
     196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
     204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
     242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
     284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
     315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
     340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
     360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
     417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432],
    dtype=float,
)


def _short_series():
    return np.array(
        [10, 12, 15, 13, 16, 18, 20, 19, 22, 25, 28, 30,
         11, 13, 16, 14, 17, 19, 21, 20, 23, 26, 29, 31],
        dtype=float,
    )


def test_adam_simulate_mam_returns_finite_and_right_shape():
    m = ADAM(model="MAM", lags=[12], initial="backcasting").fit(AIRPASSENGERS)
    sim = m.simulate(nsim=5, obs=len(AIRPASSENGERS), seed=0)
    assert sim.data.shape == (len(AIRPASSENGERS), 5)
    assert sim.states.shape[0] == 3  # level, trend, seasonality
    assert sim.states.shape[2] == 5
    assert np.all(np.isfinite(sim.data.to_numpy()))


def test_adam_simulate_seed_deterministic():
    m = ADAM(model="ANN", initial="backcasting").fit(_short_series())
    a = m.simulate(nsim=3, obs=20, seed=42)
    b = m.simulate(nsim=3, obs=20, seed=42)
    np.testing.assert_array_equal(a.data.to_numpy(), b.data.to_numpy())


def test_adam_simulate_default_obs_matches_in_sample():
    m = ADAM(model="ANN", initial="backcasting").fit(_short_series())
    sim = m.simulate(nsim=1, seed=0)
    assert sim.data.shape == (len(_short_series()),)


def test_adam_simulate_model_label_suffix():
    """R's ``print.adam.sim`` writes ``... estimated via adam()`` —
    our repr does too."""
    m = ADAM(model="ANN", initial="backcasting").fit(_short_series())
    sim = m.simulate(nsim=1, obs=24, seed=0)
    assert "estimated via adam()" in sim.model
    out = str(sim)
    assert "estimated via adam()" in out


def test_es_inherits_simulate():
    m = ES(model="ANN", lags=[1]).fit(_short_series())
    sim = m.simulate(nsim=2, obs=24, seed=42)
    assert sim.data.shape == (24, 2)
    assert np.all(np.isfinite(sim.data.to_numpy()))


def test_msarima_inherits_simulate():
    m = MSARIMA(
        orders={"ar": [1], "i": [0], "ma": [1]},
        lags=[1],
        initial="backcasting",
    ).fit(_short_series())
    sim = m.simulate(nsim=2, obs=24, seed=0)
    assert sim.data.shape == (24, 2)
    assert np.all(np.isfinite(sim.data.to_numpy()))


def test_sma_inherits_simulate():
    m = SMA(order=3).fit(_short_series())
    sim = m.simulate(nsim=1, obs=24, seed=0)
    assert sim.data.shape == (24,)
    assert np.all(np.isfinite(sim.data.to_numpy()))


def test_adam_simulate_callable_randomizer_passes_through():
    """A callable randomizer skips R's distribution dispatch."""
    m = ADAM(model="ANN", initial="backcasting").fit(_short_series())
    errors = np.full(24, 0.1)

    def feed(n):
        return errors[:n]

    sim = m.simulate(nsim=1, obs=24, randomizer=feed)
    # With ANN (additive errors) and the level fixed, every step
    # adds exactly 0.1 (modulo state evolution).
    assert np.all(np.isfinite(sim.data.to_numpy()))
