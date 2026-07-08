"""R↔Python parity for :func:`smooth.sim_oes`.

The sub-models pull their errors via the plug-in-numbers trick.
``sim_oes`` calls ``sim_es`` twice (for models A and B); we feed the
**concatenated** error vector to a single ``ourFunction`` that doles
out the right slice to each call. With identical errors driving the
same C++ ``adamCore::simulate`` kernel in both languages, the
resulting probability series must match to floating-point.
"""

from __future__ import annotations

import numpy as np
import pytest

from smooth import sim_oes

from ._r_bridge import r_dict, r_to_literal

pytestmark = pytest.mark.r_parity


def _r_sim_oes(
    occurrence,
    obs,
    errors,
    *,
    model="MNN",
    persistence=None,
    initial=None,
    model_b=None,
    persistence_b=None,
    initial_b=None,
    randomizer_kwargs=None,
):
    """Drive R's ``sim.oes`` with a pre-generated error vector.

    The R-side ``ourFunction`` is stateful — successive calls (from
    ``sim.es`` for A, then ``sim.es`` for B) read sequential slices of
    the supplied ``errors`` vector. This matches Python's per-sub-model
    error draw exactly: ``sim_es`` for A consumes ``errors[:obs]`` and
    ``sim_es`` for B consumes ``errors[obs:2*obs]``.
    """
    R_data = {"errors": np.asarray(errors, dtype=float).ravel()}
    args = [
        f"model={r_to_literal(model)}",
        f"obs={int(obs)}",
        f"occurrence={r_to_literal(occurrence)}",
        "randomizer='ourFunction'",
    ]
    if persistence is not None:
        args.append(f"persistence={r_to_literal(persistence)}")
    if initial is not None:
        args.append(f"initial={r_to_literal(initial)}")
    if model_b is not None:
        args.append(f"modelB={r_to_literal(model_b)}")
    if persistence_b is not None:
        args.append(f"persistenceB={r_to_literal(persistence_b)}")
    if initial_b is not None:
        args.append(f"initialB={r_to_literal(initial_b)}")
    # Pass a no-op ellipsis arg so R doesn't switch to rnorm.
    if randomizer_kwargs:
        for k, v in randomizer_kwargs.items():
            args.append(f"{k}={float(v)}")
    else:
        args.append("dummy=1")
    arg_string = ",".join(args)

    # ``cursor`` is a global counter; ``ourFunction`` advances it on
    # each call. ``sim.oes`` calls ``sim.es`` up to twice (A then B);
    # each takes ``obs * nsim`` numbers.
    return r_dict(
        f"""{{
            assign('cursor', 0, envir=globalenv());
            assign('ourFunction', function(n, ...) {{
                out <- errors[(cursor + 1):(cursor + n)];
                assign('cursor', cursor + n, envir=globalenv());
                out
            }}, envir=globalenv());
            s <- suppressWarnings(sim.oes({arg_string}));
            list(
                probability = as.numeric(s$probability)
            )
        }}""",
        R_data=R_data,
    )


def test_sim_oes_odds_ratio_matches_r_with_shared_errors():
    """Odds-ratio simulates A only → ``obs`` errors total."""
    obs = 40
    rng = np.random.default_rng(0)
    errors = rng.normal(0.0, 1.0, obs)

    cursor = [0]

    def feed(n):
        out = errors[cursor[0]:cursor[0] + n]
        cursor[0] += n
        return out

    py = sim_oes(
        model="ANN", obs=obs, occurrence="odds-ratio",
        persistence=[0.1], initial=[0.5],
        randomizer=feed,
    )
    cursor[0] = 0
    r = _r_sim_oes("odds-ratio", obs, errors,
                   model="ANN", persistence=[0.1], initial=[0.5])

    np.testing.assert_allclose(
        py.data.to_numpy(), np.asarray(r["probability"]), atol=1e-10
    )


def test_sim_oes_general_matches_r_with_shared_errors():
    """General simulates both A and B → ``2 * obs`` errors total."""
    obs = 30
    rng = np.random.default_rng(0)
    errors = rng.normal(0.0, 1.0, 2 * obs)

    cursor = [0]

    def feed(n):
        out = errors[cursor[0]:cursor[0] + n]
        cursor[0] += n
        return out

    py = sim_oes(
        model="ANN", obs=obs, occurrence="general",
        persistence=[0.1], initial=[0.5],
        model_b="ANN", persistence_b=[0.1], initial_b=[0.5],
        randomizer=feed,
    )
    cursor[0] = 0
    r = _r_sim_oes(
        "general", obs, errors,
        model="ANN", persistence=[0.1], initial=[0.5],
        model_b="ANN", persistence_b=[0.1], initial_b=[0.5],
    )

    np.testing.assert_allclose(
        py.data.to_numpy(), np.asarray(r["probability"]), atol=1e-10
    )


def test_sim_oes_direct_matches_r_with_shared_errors():
    """Direct simulates A only and clips to [0, 1]."""
    obs = 30
    rng = np.random.default_rng(7)
    errors = rng.normal(0.0, 1.0, obs)

    cursor = [0]

    def feed(n):
        out = errors[cursor[0]:cursor[0] + n]
        cursor[0] += n
        return out

    py = sim_oes(
        model="ANN", obs=obs, occurrence="direct",
        persistence=[0.05], initial=[0.5],
        randomizer=feed,
    )
    cursor[0] = 0
    r = _r_sim_oes("direct", obs, errors,
                   model="ANN", persistence=[0.05], initial=[0.5])

    np.testing.assert_allclose(
        py.data.to_numpy(), np.asarray(r["probability"]), atol=1e-10
    )
