"""The random generators behind simulated intervals against greybox's CDFs.

The draws keep their own scipy/numpy generators for seeded reproducibility, so
each is checked against the greybox distribution function R uses as well.
"""

from pathlib import Path

import greybox as gb
import numpy as np
import pandas as pd
import pytest

from smooth import ADAM
from smooth.adam_general.core.utils.distributions import ralaplace, rs

DATA = Path(__file__).parent / "data"
N = 200_000


def _max_cdf_gap(draws, cdf, probs):
    quantiles = np.quantile(draws, probs)
    return np.max(np.abs(cdf(quantiles) - probs))


def test_rs_matches_greybox_s_distribution():
    draws = rs(N, mu=1.0, scale=0.7, random_state=42)
    probs = np.array([0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
    assert _max_cdf_gap(draws, lambda q: gb.ps(q, 1.0, 0.7), probs) < 0.005
    assert np.var(draws) == pytest.approx(120 * 0.7**4, rel=0.05)


@pytest.mark.parametrize("alpha", [0.2, 0.5, 0.8])
def test_ralaplace_matches_greybox_alaplace(alpha):
    draws = ralaplace(N, mu=0.0, scale=1.3, alpha=alpha, random_state=7)
    probs = np.array([0.05, 0.25, alpha, 0.75, 0.95])
    gap = _max_cdf_gap(draws, lambda q: gb.palaplace(q, 0.0, 1.3, alpha), probs)
    assert gap < 0.005
    assert np.mean(draws <= 0) == pytest.approx(alpha, abs=0.005)


def test_provided_gnorm_shape_survives_fit():
    y = pd.read_csv(DATA / "sm_positive.csv")["y"].to_numpy(float)
    model = ADAM(model="ANN", lags=[1], distribution="dgnorm", gnorm_shape=1.5)
    model.fit(y)
    scale = model.scale
    model.predict(h=5, interval="simulated", nsim=200)
    assert model.scale == scale
    assert model.other["shape"] == 1.5
    assert np.all(np.isfinite(model.rstandard()))
