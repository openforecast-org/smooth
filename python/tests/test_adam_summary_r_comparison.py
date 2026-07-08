"""Parity tests of Python ADAM vcov/confint/summary against R's
``smooth::adam``.

R outputs are fetched live via :mod:`tests._r_bridge` (which loads the local
R source through ``devtools::load_all``), so the comparison is always
against the current checkout.

Scenarios exercise every CI-clamping path: ``ANN`` / ``AAN`` (usual ETS),
``AAdN`` with ``bounds="admissible"`` (eigen-bounds + phi), an
ARIMA(1,0,1) (AR/MA stationarity/invertibility clamping), and an
``initial="two-stage"`` variant of ``AAN`` (which produces the same
``B`` shape as ``optimal`` — locks in the per-parameter relative
Hessian step's effect on the staged path).

With the per-parameter relative Hessian step
``h_i = ε^(1/4) · max(|x_i|, 1)`` in ``src/headers/hessianCore.h``,
SEs for the large-magnitude initial states (``level``, ``trend``,
``ARIMAState*``) match R at the same tight ``rtol=2e-2`` as the
smoothing parameters.

Skipped in CI by default (``r_parity`` marker — opt in with
``pytest -m r_parity``).
"""

from __future__ import annotations

import numpy as np
import pytest

from smooth.adam_general.core.adam import ADAM
from smooth.adam_general.core.utils.var_covar import (
    fisher_information,
    invert_fisher_information,
)

from ._r_bridge import r_dict

pytestmark = pytest.mark.r_parity

# Scenario -> (Python ADAM kwargs, R sim seed, R sim model+persistence, R fit kwargs).
SCENARIOS = {
    "ann": {
        "py_kw": dict(model="ANN"),
        "r_sim": "set.seed(41); y <- sim.es('ANN', obs=120, frequency=12, "
        "persistence=0.3)$data",
        "r_fit": "adam(y, model='ANN', initial='optimal')",
    },
    "aan": {
        "py_kw": dict(model="AAN"),
        "r_sim": "set.seed(42); y <- sim.es('AAN', obs=120, frequency=12, "
        "persistence=c(0.3, 0.1))$data",
        "r_fit": "adam(y, model='AAN', initial='optimal')",
    },
    "aadn": {
        "py_kw": dict(model="AAdN", bounds="admissible"),
        "r_sim": "set.seed(43); y <- sim.es('AAdN', obs=120, frequency=12, "
        "persistence=c(0.3, 0.1), phi=0.9)$data",
        "r_fit": "adam(y, model='AAdN', initial='optimal', bounds='admissible')",
    },
    "arima": {
        "py_kw": dict(model="NNN", ar_order=1, i_order=0, ma_order=1),
        "r_sim": "set.seed(44); y <- as.numeric(arima.sim(list(ar=0.4, ma=0.3), "
        "n=300)) + 20",
        "r_fit": "adam(y, model='NNN', orders=list(ar=1, i=0, ma=1), "
        "initial='optimal')",
    },
    # `initial="two-stage"` produces the same B shape as `optimal` (initials
    # in B); the staged path is just a better optimiser seed. With the
    # per-parameter relative Hessian step in hessianCore.h, the SEs match
    # R at the same tight tolerance — locks in the two-stage parity.
    "aan_two_stage": {
        "py_kw": dict(model="AAN", initial="two-stage"),
        "r_sim": "set.seed(42); y <- sim.es('AAN', obs=120, frequency=12, "
        "persistence=c(0.3, 0.1))$data",
        "r_fit": "adam(y, model='AAN', initial='two-stage')",
    },
}


def _r_outputs_for(scenario):
    """One R subprocess call: simulate, fit, extract everything needed."""
    spec = SCENARIOS[scenario]
    expr = (
        f"{{ {spec['r_sim']};"
        f"  m <- {spec['r_fit']};"
        "   list(series=as.numeric(y),"
        "        coef=as.numeric(coef(m)),"
        "        coef_names=names(coef(m)),"
        "        vcov=as.matrix(vcov(m)),"
        "        confint=as.matrix(confint(m, level=0.95)),"
        "        confint_rownames=rownames(confint(m, level=0.95))) }"
    )
    return r_dict(expr)


@pytest.fixture(scope="module")
def r_outputs():
    """One R subprocess per scenario; cached across all parametrised tests."""
    return {name: _r_outputs_for(name) for name in SCENARIOS}


def _python_fit(scenario, r_outputs):
    spec = SCENARIOS[scenario]
    y = np.asarray(r_outputs[scenario]["series"], dtype=float)
    # Default to initial="optimal"; the scenario's py_kw may override
    # (e.g. the two-stage scenario uses initial="two-stage").
    kw = dict(lags=[1], initial="optimal", fi=True)
    kw.update(spec["py_kw"])
    return ADAM(**kw).fit(y)


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_coef_names_match_r(scenario, r_outputs):
    m = _python_fit(scenario, r_outputs)
    r_names = list(r_outputs[scenario]["coef_names"])
    assert m.coef_names == r_names


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_vcov_matches_r_at_same_b(scenario, r_outputs):
    """Decisive: invert the FI computed at R's coefficients, compare to R vcov."""
    m = _python_fit(scenario, r_outputs)
    r_B = np.asarray(r_outputs[scenario]["coef"], dtype=float)  # noqa: N806
    r_vcov = np.asarray(r_outputs[scenario]["vcov"], dtype=float)

    FI = fisher_information(  # noqa: N806
        r_B,
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
    cov = invert_fisher_information(FI)
    # With both (i) the per-parameter relative FD Hessian step in
    # hessianCore.h and (ii) the shared olsCore.h backend for the
    # msdecompose global smoother, the residual gap collapses from the
    # historical ~2% to the FD-Hessian discretisation floor (~1e-7 on
    # ETS scenarios, ~1e-4 on the ARIMA scenario where extra polynomial
    # ops accumulate).
    np.testing.assert_allclose(cov, r_vcov, rtol=1e-3, atol=1e-6)

    # Smoothing-parameter covariance block (alpha/beta/gamma/phi) matches tightly.
    names = m.coef_names
    smooth_idx = [
        i
        for i, nm in enumerate(names)
        if nm == "phi" or nm[:1] in ("a", "b", "g") and not nm.startswith("ARIMA")
    ]
    if smooth_idx:
        sub = np.ix_(smooth_idx, smooth_idx)
        np.testing.assert_allclose(cov[sub], r_vcov[sub], rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_confint_matches_r(scenario, r_outputs):
    m = _python_fit(scenario, r_outputs)
    ci = m.confint(level=0.95)
    r_ci = np.asarray(r_outputs[scenario]["confint"], dtype=float)
    r_rownames = list(r_outputs[scenario]["confint_rownames"])

    assert list(ci.index) == r_rownames

    # Confidence bounds are bounded by Python NLopt vs R nloptr
    # convergence drift on the coefficient vector itself (Python runs
    # its own fit here, so confint = py_coef ± py_SE × t-crit), not by
    # the OLS-step ULP that the shared olsCore.h closed. Keep the
    # historical 2e-2 bound for confint to reflect that optimiser floor.
    np.testing.assert_allclose(
        ci.iloc[:, 1:].to_numpy(), r_ci[:, 1:], rtol=2e-2, atol=1e-2
    )
    # SE column from confint() is computed at Python's own coefs (it
    # runs its own fit), so it inherits the same NLopt-vs-nloptr
    # optimiser-convergence floor as the lower/upper bounds. The vcov
    # parity at R's coefs is the decisive measurement
    # (test_vcov_matches_r_at_same_b) and tightened to 1e-3.
    np.testing.assert_allclose(ci["S.E."].to_numpy(), r_ci[:, 0], rtol=2e-2, atol=1e-3)


def test_summary_layout_and_significance(r_outputs):
    m = _python_fit("aan", r_outputs)
    s = str(m.summary())
    for marker in (
        "Model estimated using",
        "Coefficients:",
        "Error standard deviation:",
        "Sample size:",
        "Number of estimated parameters:",
        "Information criteria:",
    ):
        assert marker in s
    # alpha/beta are significant for this series -> at least one star.
    assert "*" in s


def test_print_and_summary_are_separate(r_outputs):
    m = _python_fit("aan", r_outputs)
    concise = str(m)  # print.adam-style
    full = str(m.summary())  # summary.adam-style
    assert concise != full
    # Concise report carries the persistence-vector marker; the summary doesn't.
    assert "Persistence vector" in concise
    assert "Persistence vector" not in full
    # The summary carries the coefficient table; the concise report doesn't.
    assert "Coefficients:" in full
