"""R↔Python parity for :func:`smooth.sim_es`.

Uses the plug-in-numbers trick — pre-generate the error vector in
Python, pass it to R via ``randomizer="ourFunction"`` and to Python via
a callable. Both languages then drive the **same** C++
``adamCore::simulate`` kernel with the **same** errors, so the
generated series and the state cube must agree to floating-point.
"""

from __future__ import annotations

import numpy as np
import pytest

from smooth import sim_es

from ._r_bridge import r_dict, r_to_literal

pytestmark = pytest.mark.r_parity


def _r_sim_es(model, obs, errors, *, persistence=None, initial=None,
              initial_season=None, frequency=1, phi=1.0):
    """Drive R's ``sim.es`` with a pre-generated error vector.

    The R side defines ``ourFunction <- function(n, ...) errors[1:n]``
    and passes it via ``randomizer="ourFunction"``. We also pass a
    no-op ellipsis arg (``dummy=1``) so R's ``length(ellipsis)>0``
    branch fires — without it, ``sim.es`` warns "needs some arbitrary
    parameters!" and silently switches to ``rnorm`` (R/simes.R:294-297).
    """
    R_data = {"errors": np.asarray(errors, dtype=float).ravel()}
    args = [
        f"model={r_to_literal(model)}",
        f"obs={int(obs)}",
        f"frequency={int(frequency)}",
        "randomizer='ourFunction'",
    ]
    if persistence is not None:
        args.append(f"persistence={r_to_literal(persistence)}")
    if initial is not None:
        args.append(f"initial={r_to_literal(initial)}")
    if initial_season is not None:
        args.append(f"initialSeason={r_to_literal(initial_season)}")
    if phi != 1.0:
        args.append(f"phi={float(phi)}")
    args.append("dummy=1")  # bypass the switch-to-rnorm guard
    arg_string = ",".join(args)

    # Assign ``ourFunction`` to the global environment so R's
    # ``do.call("ourFunction", ...)`` (which resolves names by string
    # against ``globalenv()``) can find it. A ``local({...})`` block
    # would hide the binding from ``do.call``.
    return r_dict(
        f"""{{
            assign('ourFunction', function(n, ...) errors[1:n],
                   envir=globalenv());
            s <- suppressWarnings(sim.es({arg_string}));
            list(
                data = as.numeric(s$data),
                states = as.numeric(t(as.matrix(s$states))),
                persistence = as.numeric(s$persistence),
                residuals = as.numeric(s$residuals)
            )
        }}""",
        R_data=R_data,
    )


def test_sim_es_ann_matches_r_with_shared_errors():
    """ANN: identical errors feeding the same C++ kernel ⇒ identical data."""
    obs = 50
    errors = np.linspace(-1.5, 1.5, obs)  # deterministic, easily reproducible

    def feed(n):
        return errors[:n]

    py = sim_es(
        model="ANN", obs=obs,
        persistence=[0.3], initial=[100.0],
        randomizer=feed,
    )
    r = _r_sim_es("ANN", obs, errors, persistence=[0.3], initial=[100.0])

    np.testing.assert_allclose(
        py.data.to_numpy(), np.asarray(r["data"]), atol=1e-10
    )
    np.testing.assert_allclose(
        py.residuals.to_numpy(), np.asarray(r["residuals"]), atol=1e-10
    )


def test_sim_es_aaa_seasonal_matches_r_with_shared_errors():
    obs = 48
    rng = np.random.default_rng(0)
    errors = rng.normal(0.0, 0.5, obs)

    def feed(n):
        return errors[:n]

    persistence = [0.1, 0.05, 0.05]
    initial = [100.0, 1.0]
    initial_season = list(np.linspace(-2.0, 2.0, 12))

    py = sim_es(
        model="AAA", obs=obs, frequency=12,
        persistence=persistence, initial=initial,
        initial_season=initial_season,
        randomizer=feed,
    )
    r = _r_sim_es(
        "AAA", obs, errors,
        persistence=persistence, initial=initial,
        initial_season=initial_season, frequency=12,
    )

    np.testing.assert_allclose(
        py.data.to_numpy(), np.asarray(r["data"]), atol=1e-10
    )
