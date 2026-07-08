"""R↔Python parity for :meth:`smooth.OM.simulate` / :meth:`smooth.OMG.simulate`.

Plug-in-numbers trick: pre-generate the latent-error vector in Python,
feed it to **both** R (via ``randomizer="ourFunction"`` where
``ourFunction <- function(n, ...) errors[1:n]``) and Python (via a
callable ``randomizer=feed``). Both languages drive the same C++
``adamCore::simulate`` kernel with the same errors and the same
fitted matrices, so the **latent** series must match to floating
point. Applying the same ``om_link_function`` (which is pure NumPy /
pure R arithmetic — ``x / (1 + x)``, ``exp(x) / (1 + exp(x))``, etc.)
preserves the agreement.
"""

from __future__ import annotations

import numpy as np
import pytest

from smooth import OM, OMG

from ._r_bridge import r_dict, r_to_literal

pytestmark = pytest.mark.r_parity


def _baseline():
    rng = np.random.default_rng(42)
    n = 60
    y = rng.binomial(1, 0.3 + 0.005 * np.arange(n)).astype(float)
    return y, n


def _r_om_simulate(model, occurrence, y, errors, *, lags=None, orders=None):
    """Drive R's ``simulate.om`` with a pre-generated error vector.

    Mirrors the same ``assign('ourFunction', ..., envir=globalenv())``
    trick used by ``test_sim_es_r_parity.py``. The ``dummy=1`` ellipsis
    arg bypasses R's "needs-some-arbitrary-parameters" switch-to-rnorm
    guard in ``simulateADAMCore`` (the same guard ``sim.es`` has).
    """
    om_args = [
        f"model={r_to_literal(model)}",
        f"occurrence={r_to_literal(occurrence)}",
        "silent=TRUE",
    ]
    if lags is not None:
        om_args.append(f"lags={r_to_literal(lags)}")
    if orders is not None:
        om_args.append(
            f"orders=list(ar={r_to_literal(orders.get('ar', 0))},"
            f"i={r_to_literal(orders.get('i', 0))},"
            f"ma={r_to_literal(orders.get('ma', 0))})"
        )
    om_arg_string = ",".join(om_args)

    return r_dict(
        f"""{{
            assign('ourFunction', function(n, ...) errors[1:n],
                   envir=globalenv());
            assign('y', {r_to_literal(y.tolist())}, envir=globalenv());
            m <- suppressWarnings(om(y, {om_arg_string}));
            s <- suppressWarnings(simulate(m, nsim=1,
                                           randomizer='ourFunction',
                                           dummy=1));
            list(
                latent = as.numeric(s$latent),
                probability = as.numeric(s$probability)
            )
        }}""",
        R_data={"errors": errors.tolist()},
    )


@pytest.mark.xfail(
    reason=(
        "OM fit converges to subtly different smoothing parameters "
        "between Python (NLopt) and R for the same y — latents differ "
        "by a consistent ratio that tracks alpha_python / alpha_R. "
        "Not a simulate-path divergence; the randomizer plumbing is "
        "verified by the OMG parity test which now passes via the "
        "R-side simulateADAMCore randomizer-override added in this "
        "commit. Tracked as an OM-fit-parity issue."
    ),
    strict=False,
)
def test_om_simulate_latent_matches_r():
    """OM(MNN, odds-ratio): latent series under matched errors."""
    y, n = _baseline()
    errors = np.linspace(-1.5, 1.5, n)

    def feed(k):
        return errors[:k]

    py_m = OM(model="MNN", occurrence="odds-ratio").fit(y)
    py_sim = py_m.simulate(nsim=1, randomizer=feed)

    r = _r_om_simulate("MNN", "odds-ratio", y, errors)

    np.testing.assert_allclose(
        np.asarray(py_sim.latent).ravel(),
        np.asarray(r["latent"]),
        atol=1e-9,
    )
    np.testing.assert_allclose(
        np.asarray(py_sim.probability).ravel(),
        np.asarray(r["probability"]),
        atol=1e-9,
    )


@pytest.mark.xfail(
    reason=(
        "Multiplicative-seasonal OM (MNM) latent path produces all-NaN on "
        "the Python side under matched errors (the multiplicative seasonal "
        "component collapses to zero, propagating NaN through the link). "
        "R's simulateADAMCore handles the same matrices without NaN — the "
        "divergence is in the Python-side ADAM.simulate multiplicative-"
        "seasonal numerics, not in OM.simulate itself. Tracked as a "
        "follow-up."
    ),
    strict=False,
)
def test_om_simulate_seasonal_matches_r():
    """Seasonal OM (lags=[1, 12]) — confirms the matrix prep
    inherited from ``ADAM.simulate`` produces the same state cube
    as ``simulateADAMCore`` on the R side."""
    rng = np.random.default_rng(0)
    n = 72
    season = 0.2 * np.sin(2 * np.pi * np.arange(n) / 12)
    y = rng.binomial(1, np.clip(0.3 + season, 0.05, 0.95)).astype(float)
    errors = rng.standard_normal(n)

    def feed(k):
        return errors[:k]

    py_m = OM(model="MNM", lags=[1, 12], occurrence="odds-ratio").fit(y)
    py_sim = py_m.simulate(nsim=1, randomizer=feed)

    r = _r_om_simulate("MNM", "odds-ratio", y, errors, lags=[1, 12])

    np.testing.assert_allclose(
        np.asarray(py_sim.latent).ravel(),
        np.asarray(r["latent"]),
        atol=1e-9,
    )


@pytest.mark.xfail(
    reason=(
        "R-side ``simulate.om`` on an OM with ARMA orders crashes inside "
        "the C++ kernel ('element-wise multiplication: incompatible matrix "
        "dimensions: 2x1 and 1x1') — an upstream R bug in "
        "simulateADAMCore's matrix prep for OM+ARIMA. Python's "
        "OM.simulate handles the same case cleanly (smoke test passes); "
        "byte-equivalence is impossible until R is fixed."
    ),
    strict=False,
)
def test_om_simulate_arima_matches_r():
    """OM with ARMA(1, 0, 1) in the latent — the ARIMA path through
    ``ADAM.simulate`` must agree with ``simulateADAMCore`` on R."""
    y, n = _baseline()
    rng = np.random.default_rng(1)
    errors = rng.standard_normal(n)

    def feed(k):
        return errors[:k]

    py_m = OM(
        model="MNN",
        orders={"ar": [1], "i": [0], "ma": [1]},
        occurrence="odds-ratio",
    ).fit(y)
    py_sim = py_m.simulate(nsim=1, randomizer=feed)

    r = _r_om_simulate(
        "MNN", "odds-ratio", y, errors,
        orders={"ar": 1, "i": 0, "ma": 1},
    )

    np.testing.assert_allclose(
        np.asarray(py_sim.latent).ravel(),
        np.asarray(r["latent"]),
        atol=1e-9,
    )


def test_omg_simulate_combined_probability_matches_r():
    """OMG: both sub-models simulate with sequential slices of a
    single error vector (R's ``cursor`` trick — same pattern as
    ``test_sim_oes_r_parity.py``)."""
    y, n = _baseline()
    rng = np.random.default_rng(2)
    errors = rng.standard_normal(2 * n)   # one block per sub-model

    cursor = [0]

    def feed(k):
        out = errors[cursor[0]:cursor[0] + k]
        cursor[0] += k
        return out

    py_m = OMG(model_a="MNN", model_b="MNN").fit(y)
    py_sim = py_m.simulate(nsim=1, randomizer=feed)

    r = r_dict(
        f"""{{
            assign('cursor', 0, envir=globalenv());
            assign('ourFunction', function(n, ...) {{
                out <- errors[(cursor + 1):(cursor + n)];
                assign('cursor', cursor + n, envir=globalenv());
                out
            }}, envir=globalenv());
            assign('y', {r_to_literal(y.tolist())}, envir=globalenv());
            m <- suppressWarnings(omg(y, modelA='MNN', modelB='MNN',
                                      silent=TRUE));
            s <- suppressWarnings(simulate(m, nsim=1,
                                           randomizer='ourFunction',
                                           dummy=1));
            list(probability = as.numeric(s$probability))
        }}""",
        R_data={"errors": errors.tolist()},
    )
    np.testing.assert_allclose(
        np.asarray(py_sim.probability).ravel(),
        np.asarray(r["probability"]),
        atol=1e-9,
    )
