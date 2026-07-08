"""Smoke tests for :func:`smooth.sim_oes`.

Verifies the four occurrence-combination formulas, the shape contract
on ``data`` / ``model_a`` / ``model_b`` / ``log_lik``, and the
print-method header from R's ``print.oes.sim``. R-parity tests live in
``test_sim_oes_r_parity.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smooth import SimulateResult, sim_oes


@pytest.mark.parametrize(
    "occurrence",
    ["odds-ratio", "inverse-odds-ratio", "direct", "general"],
)
def test_sim_oes_all_modes_produce_valid_probabilities(occurrence):
    s = sim_oes(
        model="MNN", obs=40, occurrence=occurrence,
        persistence=[0.1], initial=[1.0],
        model_b="MNN", persistence_b=[0.1], initial_b=[1.0],
        randomizer="rlnorm", seed=42, sdlog=0.3,
    )
    assert isinstance(s, SimulateResult)
    assert isinstance(s.data, pd.Series)
    assert s.data.shape == (40,)
    arr = s.data.to_numpy()
    assert np.all(np.isfinite(arr))
    assert (arr >= 0).all() and (arr <= 1).all()
    assert s.occurrence_type == occurrence
    # ``direct`` / ``odds-ratio`` need only A; ``inverse-odds-ratio``
    # needs only B; ``general`` needs both. The sub-model slots reflect
    # that.
    if occurrence in {"odds-ratio", "direct"}:
        assert s.model_a is not None
        assert s.model_b is None
    elif occurrence == "inverse-odds-ratio":
        assert s.model_a is None
        assert s.model_b is not None
    else:  # general
        assert s.model_a is not None
        assert s.model_b is not None


def test_sim_oes_model_labels():
    """R's label switch (R/simoes.R:145-149)."""
    assert sim_oes(
        model="MNN", obs=20, occurrence="odds-ratio",
        persistence=[0.1], initial=[1.0],
        randomizer="rlnorm", seed=0, sdlog=0.3,
    ).model == "oETS[O](MNN)"

    assert sim_oes(
        model_b="MNN", obs=20, occurrence="inverse-odds-ratio",
        persistence_b=[0.1], initial_b=[1.0],
        randomizer="rlnorm", seed=0, sdlog=0.3,
    ).model == "oETS[I](MNN)"

    assert sim_oes(
        model="MNN", obs=20, occurrence="direct",
        persistence=[0.1], initial=[0.5],
        randomizer="rlnorm", seed=0, sdlog=0.3,
    ).model == "oETS[D](MNN)"

    assert sim_oes(
        model="MNN", model_b="MNN", obs=20, occurrence="general",
        persistence=[0.1], initial=[1.0],
        persistence_b=[0.1], initial_b=[1.0],
        randomizer="rlnorm", seed=0, sdlog=0.3,
    ).model == "oETS[G](MNN)(MNN)"


def test_sim_oes_nsim_gt_1_returns_dataframe():
    s = sim_oes(
        model="MNN", obs=30, nsim=4, occurrence="general",
        persistence=[0.1], initial=[1.0],
        persistence_b=[0.1], initial_b=[1.0],
        randomizer="rlnorm", seed=0, sdlog=0.3,
    )
    assert isinstance(s.data, pd.DataFrame)
    assert s.data.shape == (30, 4)
    assert s.log_lik is not None and len(np.atleast_1d(s.log_lik)) == 4
    arr = s.data.to_numpy()
    assert (arr >= 0).all() and (arr <= 1).all()


def test_sim_oes_seed_deterministic():
    a = sim_oes(
        model="MNN", obs=20, occurrence="general",
        persistence=[0.1], initial=[1.0],
        persistence_b=[0.1], initial_b=[1.0],
        randomizer="rlnorm", seed=123, sdlog=0.3,
    )
    b = sim_oes(
        model="MNN", obs=20, occurrence="general",
        persistence=[0.1], initial=[1.0],
        persistence_b=[0.1], initial_b=[1.0],
        randomizer="rlnorm", seed=123, sdlog=0.3,
    )
    np.testing.assert_array_equal(a.data.to_numpy(), b.data.to_numpy())


def test_sim_oes_direct_clips_to_unit_interval():
    """``direct`` clips A's output to [0, 1] even when A drifts outside."""
    s = sim_oes(
        model="ANN", obs=20, occurrence="direct",
        persistence=[0.5], initial=[0.7],
        randomizer="rnorm", seed=42, sd=2.0,
    )
    arr = s.data.to_numpy()
    assert arr.min() >= 0.0
    assert arr.max() <= 1.0


def test_sim_oes_invalid_occurrence_raises():
    with pytest.raises(ValueError, match="Unknown occurrence type"):
        sim_oes(model="MNN", obs=10, occurrence="not-a-real-mode")


def test_sim_oes_repr_matches_r_layout():
    """Single-series OES repr lists model name, count, obs count, logLik."""
    s = sim_oes(
        model="MNN", obs=20, occurrence="odds-ratio",
        persistence=[0.1], initial=[1.0],
        randomizer="rlnorm", seed=0, sdlog=0.2,
    )
    out = str(s)
    assert "Data generated from: oETS[O](MNN)" in out
    assert "Number of generated series: 1" in out
    assert "Number of observations in each series: 20" in out
    assert "True likelihood:" in out
