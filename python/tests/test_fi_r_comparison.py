"""Parity tests of the Python Fisher Information against R's ``smooth::adam``.

R outputs are fetched live via :mod:`tests._r_bridge` (which loads the local
R source through ``devtools::load_all``), so the comparison is always against
the current checkout.

The decisive check evaluates the **Python FI at R's converged coefficients**
and compares it to R's FI: this isolates the FI computation from the small
NLopt-vs-nloptr optimiser divergence. A looser sanity check compares the FI
at Python's own optimum.

Both models are fitted with ``initial="optimal"`` so the parameter vector
contains the initial states too, matching R's default initialisation. R's
``vcov(m)`` is the inverse of the observed FI, so the FI is recovered as
``solve(vcov(m))`` at the fitted optimum.

Skipped in CI by default (``r_parity`` marker — opt in with
``pytest -m r_parity``).
"""

from __future__ import annotations

import numpy as np
import pytest

from smooth.adam_general.core.adam import ADAM
from smooth.adam_general.core.utils.var_covar import fisher_information

from ._r_bridge import r_dict

pytestmark = pytest.mark.r_parity

# Repo convention for R-parity numerics. Dominant FI entries match R to
# ~1e-4 relative; tiny initial-state entries pick up O(1e-4) noise from the
# vcov round-trip used to recover R's FI.
RTOL = 1e-3
ATOL = 1e-4

SCENARIOS = [("ann", "ANN"), ("aan", "AAN")]
SEEDS = {"ann": 41, "aan": 42}
PERSISTENCES = {"ann": "0.3", "aan": "c(0.3, 0.1)"}


def _r_fit_outputs(scenario, model):
    """One R subprocess call: simulate data, fit, return series + coef + FI."""
    seed = SEEDS[scenario]
    pers = PERSISTENCES[scenario]
    expr = (
        f"{{ set.seed({seed});"
        f"  y <- sim.es('{model}', obs=120, frequency=12, persistence={pers})$data;"
        f"  m <- adam(y, model='{model}', initial='optimal', FI=TRUE);"
        "   list(series=as.numeric(y),"
        "        coef=as.numeric(coef(m)),"
        "        coef_names=names(coef(m)),"
        "        FI=as.matrix(solve(vcov(m)))) }"
    )
    return r_dict(expr)


def _python_fi_at(m, B):
    """Python FI evaluated at ``B`` using the fitted model's internal dicts."""
    return fisher_information(
        B,
        m._model_type,
        m._components,
        m._lags_model,
        m._adam_created,
        m._persistence,
        m._initials,
        m._arima,
        m._explanatory,
        m._phi_internal,
        m._constant,
        m._observations,
        m._occurrence,
        m._general,
        m._profile,
        m._adam_cpp,
    )


@pytest.fixture(scope="module")
def r_outputs():
    """Fit each ADAM scenario once in R; share across all parametrised tests."""
    return {scenario: _r_fit_outputs(scenario, model) for scenario, model in SCENARIOS}


@pytest.mark.parametrize("scenario,model", SCENARIOS)
def test_fi_matches_r_at_same_b(scenario, model, r_outputs):
    out = r_outputs[scenario]
    y = np.asarray(out["series"], dtype=float)
    r_B = np.asarray(out["coef"], dtype=float)
    r_FI = np.asarray(out["FI"], dtype=float)  # noqa: N806

    m = ADAM(model=model, lags=[1], initial="optimal", fi=True).fit(y)

    # (a) Decisive: Python FI at R's coefficients == R's FI.
    fi_at_rB = _python_fi_at(m, r_B)  # noqa: N806
    np.testing.assert_allclose(fi_at_rB, r_FI, rtol=RTOL, atol=ATOL)

    # FI is exposed on the fitted model and is symmetric.
    assert m.fisher_information_ is not None
    assert m.fisher_information_.shape == (len(r_B), len(r_B))
    np.testing.assert_allclose(
        m.fisher_information_, m.fisher_information_.T, rtol=0, atol=1e-8
    )


@pytest.mark.parametrize("scenario,model", SCENARIOS)
def test_fi_at_python_optimum_close_to_r(scenario, model, r_outputs):
    out = r_outputs[scenario]
    y = np.asarray(out["series"], dtype=float)
    r_FI = np.asarray(out["FI"], dtype=float)  # noqa: N806

    m = ADAM(model=model, lags=[1], initial="optimal", fi=True).fit(y)

    # (b) Sanity: FI at Python's own optimum is close to R's (looser, since
    # the two optimisers can settle at slightly different coefficient vectors).
    np.testing.assert_allclose(m.fisher_information_, r_FI, rtol=5e-2, atol=1e-2)


def test_fi_absent_by_default(r_outputs):
    y = np.asarray(r_outputs["ann"]["series"], dtype=float)
    m = ADAM(model="ANN", lags=[1], initial="optimal").fit(y)
    assert m.fisher_information_ is None
