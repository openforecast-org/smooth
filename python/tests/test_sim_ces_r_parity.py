"""R↔Python parity for :func:`smooth.sim_ces`.

Plug-in-numbers trick: pre-generate identical errors and pin ``a`` /
``b`` to fixed complex values so the stability-rejection sampler never
fires. Both languages then drive the same C++ ``adamCore::simulate``
kernel with byte-identical inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from smooth import sim_ces

from ._r_bridge import r_dict, r_to_literal

pytestmark = pytest.mark.r_parity


def _r_sim_ces(seasonality, obs, errors, *, a, b=None, initial=None,
               frequency=1):
    R_data = {"errors": np.asarray(errors, dtype=float).ravel()}
    args = [
        f"seasonality={r_to_literal(seasonality)}",
        f"obs={int(obs)}",
        f"frequency={int(frequency)}",
        f"a=complex(real={float(a.real)},imaginary={float(a.imag)})",
        "randomizer='ourFunction'",
    ]
    if b is not None:
        if isinstance(b, complex):
            args.append(
                f"b=complex(real={float(b.real)},imaginary={float(b.imag)})"
            )
        else:
            args.append(f"b={float(b)}")
    if initial is not None:
        args.append(f"initial={r_to_literal(initial)}")
    args.append("dummy=1")
    arg_string = ",".join(args)

    return r_dict(
        f"""{{
            assign('ourFunction', function(n, ...) errors[1:n],
                   envir=globalenv());
            s <- suppressWarnings(sim.ces({arg_string}));
            list(
                data = as.numeric(s$data),
                residuals = as.numeric(s$residuals)
            )
        }}""",
        R_data=R_data,
    )


def test_sim_ces_none_matches_r_with_shared_errors():
    obs = 40
    rng = np.random.default_rng(0)
    errors = rng.normal(0.0, 1.0, obs)

    def feed(n):
        return errors[:n]

    a = complex(1.3, 1.0)
    py = sim_ces(seasonality="none", obs=obs, a=a,
                 initial=[100.0, 0.0], randomizer=feed)
    r = _r_sim_ces("none", obs, errors, a=a, initial=[100.0, 0.0])

    np.testing.assert_allclose(
        py.data.to_numpy(), np.asarray(r["data"]), atol=1e-9
    )


def test_sim_ces_full_matches_r_with_shared_errors():
    obs = 48
    freq = 4
    rng = np.random.default_rng(0)
    errors = rng.normal(0.0, 1.0, obs)

    def feed(n):
        return errors[:n]

    a = complex(1.3, 1.0)
    b = complex(1.3, 1.0)
    # 4 components × lag_max(=freq)=4 = 16 entries.
    # Layout per R/simces.R:262: ``rep(initialValue, each=nsim)`` fills
    # ``matInitialValue[, 1:lagsModelMax,]`` column-major — the first
    # ``components_number`` values become the lag-1 slice, etc.
    init = np.tile([10.0, 0.0, 5.0, 0.0], freq).tolist()
    py = sim_ces(seasonality="full", obs=obs, frequency=freq,
                 a=a, b=b, initial=init, randomizer=feed)
    r = _r_sim_ces("full", obs, errors, a=a, b=b,
                   initial=init, frequency=freq)

    np.testing.assert_allclose(
        py.data.to_numpy(), np.asarray(r["data"]), atol=1e-9
    )
