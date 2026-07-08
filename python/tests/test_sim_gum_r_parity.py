"""R↔Python parity for :func:`smooth.sim_gum`.

Plug-in-numbers trick: pre-generate identical errors and supply
explicit ``measurement`` / ``transition`` / ``persistence`` matrices
so neither side fires the stability-rejection sampler. With matching
inputs the shared C++ ``adamCore::simulate`` kernel produces
byte-identical output.
"""

from __future__ import annotations

import numpy as np
import pytest

from smooth import sim_gum

from ._r_bridge import r_dict, r_to_literal

pytestmark = pytest.mark.r_parity


def _r_sim_gum(orders, lags, obs, errors, *,
               measurement, transition, persistence, initial,
               frequency=1):
    R_data = {"errors": np.asarray(errors, dtype=float).ravel()}
    args = [
        f"orders={r_to_literal(orders)}",
        f"lags={r_to_literal(lags)}",
        f"obs={int(obs)}",
        f"frequency={int(frequency)}",
        f"measurement={r_to_literal(measurement)}",
        f"transition={r_to_literal(np.asarray(transition).ravel(order='F'))}",
        f"persistence={r_to_literal(persistence)}",
        f"initial={r_to_literal(initial)}",
        "randomizer='ourFunction'",
        "dummy=1",
    ]
    arg_string = ",".join(args)

    return r_dict(
        f"""{{
            assign('ourFunction', function(n, ...) errors[1:n],
                   envir=globalenv());
            s <- suppressWarnings(sim.gum({arg_string}));
            list(
                data = as.numeric(s$data),
                residuals = as.numeric(s$residuals)
            )
        }}""",
        R_data=R_data,
    )


def test_sim_gum_local_level_matches_r_with_shared_errors():
    """GUM(1[1]) with fixed measurement=1, transition=0.5, persistence=0.3."""
    obs = 40
    rng = np.random.default_rng(0)
    errors = rng.normal(0.0, 1.0, obs)

    def feed(n):
        return errors[:n]

    py = sim_gum(
        orders=[1], lags=[1], obs=obs,
        measurement=[1.0], transition=[0.5], persistence=[0.3],
        initial=[10.0], randomizer=feed,
    )
    r = _r_sim_gum(
        [1], [1], obs, errors,
        measurement=[1.0], transition=[0.5], persistence=[0.3],
        initial=[10.0],
    )

    np.testing.assert_allclose(
        py.data.to_numpy(), np.asarray(r["data"]), atol=1e-10
    )


def test_sim_gum_two_component_matches_r_with_shared_errors():
    """GUM(2[1]) with column-major reshape of the transition matrix."""
    obs = 40
    rng = np.random.default_rng(1)
    errors = rng.normal(0.0, 1.0, obs)

    def feed(n):
        return errors[:n]

    # Companion form, column-major: c(1, 0, 0.5, 0.5) → [[1, 0.5], [0, 0.5]]
    transition = np.array([[1.0, 0.5], [0.0, 0.5]])
    py = sim_gum(
        orders=[2], lags=[1], obs=obs,
        measurement=[1.0, 0.0],
        transition=transition.ravel(order="F").tolist(),
        persistence=[0.3, 0.1],
        initial=[100.0, 1.0],
        randomizer=feed,
    )
    r = _r_sim_gum(
        [2], [1], obs, errors,
        measurement=[1.0, 0.0],
        transition=transition,
        persistence=[0.3, 0.1],
        initial=[100.0, 1.0],
    )

    np.testing.assert_allclose(
        py.data.to_numpy(), np.asarray(r["data"]), atol=1e-10
    )
