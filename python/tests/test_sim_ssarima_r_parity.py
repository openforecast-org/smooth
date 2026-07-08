"""R↔Python parity for :func:`smooth.sim_ssarima` and :func:`smooth.sim_sma`.

Plug-in-numbers trick: pre-generate the error vector in Python, feed
it to both R (via ``randomizer="ourFunction"``) and Python (via a
callable). Both languages drive the same C++ ``adamCore::simulate``
kernel — when the errors and the model matrices match, the output
agrees to floating-point.
"""

from __future__ import annotations

import numpy as np
import pytest

from smooth import sim_sma, sim_ssarima

from ._r_bridge import r_dict, r_to_literal

pytestmark = pytest.mark.r_parity


def _r_sim_ssarima(
    orders, lags, obs, errors, *, arma=None, initial=None,
    constant=False, frequency=1, bounds="none",
):
    """Drive R's ``sim.ssarima`` with a pre-generated error vector.

    Defines ``ourFunction`` in the global env (so ``do.call`` can find
    it) and passes a ``dummy=1`` ellipsis arg to bypass R's
    "needs-some-arbitrary-parameters" switch-to-rnorm guard
    (R/simssarima.R:594-599).
    """
    R_data = {"errors": np.asarray(errors, dtype=float).ravel()}
    args = [
        f"orders=list(ar={r_to_literal(orders['ar'])},"
        f"i={r_to_literal(orders['i'])},"
        f"ma={r_to_literal(orders['ma'])})",
        f"lags={r_to_literal(lags)}",
        f"obs={int(obs)}",
        f"frequency={int(frequency)}",
        f"bounds={r_to_literal(bounds)}",
        "randomizer='ourFunction'",
    ]
    if arma is not None:
        ar = arma.get("ar", [])
        ma = arma.get("ma", [])
        args.append(
            f"arma=list(ar={r_to_literal(ar)},ma={r_to_literal(ma)})"
        )
    if initial is not None:
        args.append(f"initial={r_to_literal(initial)}")
    if isinstance(constant, bool):
        args.append(f"constant={'TRUE' if constant else 'FALSE'}")
    else:
        args.append(f"constant={float(constant)}")
    args.append("dummy=1")
    arg_string = ",".join(args)

    return r_dict(
        f"""{{
            assign('ourFunction', function(n, ...) errors[1:n],
                   envir=globalenv());
            s <- suppressWarnings(sim.ssarima({arg_string}));
            list(
                data = as.numeric(s$data),
                residuals = as.numeric(s$residuals)
            )
        }}""",
        R_data=R_data,
    )


def _r_sim_sma(order, obs, errors, *, initial=None, frequency=1):
    R_data = {"errors": np.asarray(errors, dtype=float).ravel()}
    args = [
        f"order={int(order)}",
        f"obs={int(obs)}",
        f"frequency={int(frequency)}",
        "randomizer='ourFunction'",
    ]
    if initial is not None:
        args.append(f"initial={r_to_literal(initial)}")
    args.append("dummy=1")
    arg_string = ",".join(args)
    return r_dict(
        f"""{{
            assign('ourFunction', function(n, ...) errors[1:n],
                   envir=globalenv());
            s <- suppressWarnings(sim.sma({arg_string}));
            list(
                data = as.numeric(s$data),
                residuals = as.numeric(s$residuals)
            )
        }}""",
        R_data=R_data,
    )


def test_sim_ssarima_ar1_matches_r_with_shared_errors():
    obs = 50
    errors = np.linspace(-1.5, 1.5, obs)

    def feed(n):
        return errors[:n]

    orders = {"ar": [1], "i": [0], "ma": [0]}
    arma = {"ar": [0.5], "ma": []}

    py = sim_ssarima(
        orders=orders, lags=[1], obs=obs,
        arma=arma, initial=[10.0], randomizer=feed,
    )
    r = _r_sim_ssarima(orders, [1], obs, errors, arma=arma, initial=[10.0])

    np.testing.assert_allclose(
        py.data.to_numpy(), np.asarray(r["data"]), atol=1e-10
    )
    np.testing.assert_allclose(
        py.residuals.to_numpy(), np.asarray(r["residuals"]), atol=1e-10
    )


def test_sim_ssarima_ima11_matches_r_with_shared_errors():
    obs = 60
    rng = np.random.default_rng(0)
    errors = rng.normal(0.0, 1.0, obs)

    def feed(n):
        return errors[:n]

    orders = {"ar": [0], "i": [1], "ma": [1]}
    arma = {"ar": [], "ma": [0.4]}

    py = sim_ssarima(
        orders=orders, lags=[1], obs=obs,
        arma=arma, initial=[100.0], randomizer=feed,
    )
    r = _r_sim_ssarima(orders, [1], obs, errors, arma=arma, initial=[100.0])

    np.testing.assert_allclose(
        py.data.to_numpy(), np.asarray(r["data"]), atol=1e-9
    )


def test_sim_sma_order_3_matches_r_with_shared_errors():
    obs = 40
    rng = np.random.default_rng(0)
    errors = rng.normal(0.0, 1.0, obs)

    def feed(n):
        return errors[:n]

    py = sim_sma(order=3, obs=obs, initial=[10.0, 10.0, 10.0], randomizer=feed)
    r = _r_sim_sma(3, obs, errors, initial=[10.0, 10.0, 10.0])

    np.testing.assert_allclose(
        py.data.to_numpy(), np.asarray(r["data"]), atol=1e-10
    )
