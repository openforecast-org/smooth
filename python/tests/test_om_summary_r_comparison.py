"""Parity tests of Python OM/OMG vcov/confint/summary against R's
``smooth::om()`` / ``omg()``.

R outputs are fetched live via :mod:`tests._r_bridge` (which loads the local
R source through ``devtools::load_all``), so the comparison is always against
the current checkout — no pre-generated CSV reference data.

Scenarios:

* ``om_odds_mnn`` — OM with ``MNN`` / odds-ratio occurrence
* ``om_inv_mnn`` — OM with ``MNN`` / inverse-odds-ratio occurrence
* ``omg_mnn_mnn`` — OMG (joint two-sub-model MNN/MNN) via R's
  ``om(occurrence="general")`` routing

Coefficient, log-likelihood and fitted values match R to machine precision.
Hessian-derived quantities (vcov, SE, CI) now also agree to ~1e-9 on OMG and
within ~1% on standalone OM after three alignments: (i) ``bounds="none"`` is
passed during the numerical Hessian (matches R's ``boundsFI <- "none"`` at
R/adam.R:2797), (ii) ``invert_fisher_information`` takes ``abs(diag(...))``
to mirror R/adam.R:5226, and (iii) OMG sub-models inherit the proper
parameter names from the joint initialiser so persistence-parameter
clamping fires correctly in ``confint``.

Skipped in CI by default (``r_parity`` marker — opt in with
``pytest -m r_parity``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smooth import OM, OMG

from ._r_bridge import r_dict

pytestmark = pytest.mark.r_parity

# Scenario -> (Python class, Python kwargs, R model call template). The R
# template uses ``y`` as the variable name (bound by ``r_dict`` via R_data).
SCENARIOS = {
    "om_odds_mnn": (
        OM,
        dict(model="MNN", occurrence="odds-ratio", lags=[1]),
        "om(y, model='MNN', occurrence='odds-ratio', lags=1)",
    ),
    "om_inv_mnn": (
        OM,
        dict(model="MNN", occurrence="inverse-odds-ratio", lags=[1]),
        "om(y, model='MNN', occurrence='inverse-odds-ratio', lags=1)",
    ),
    "omg_mnn_mnn": (
        OMG,
        dict(model_a="MNN", model_b="MNN", lags=[1]),
        "om(y, model='MNN', occurrence='general', lags=1)",
    ),
}

COEF_RTOL = 1e-2
COEF_ATOL = 1e-3
# After aligning Python's FI computation with R (bounds="none" during the
# Hessian, abs(diag(vcov)), proper B_names plumbed to OMG sub-models for
# clamping) AND fixing R's vcov.om to preserve the original initialType
# during its FI refit (R/om.R fitted-model intake — see comments there),
# both OM and OMG now agree with R to machine precision (~1e-9 / 1e-17).
# Tolerances reflect that: very tight, catching any future regression.
VCOV_RTOL = 1e-4
VCOV_ATOL = 1e-6
SE_RTOL = 1e-4
SE_ATOL = 1e-6
CI_RTOL = 1e-4
CI_ATOL = 1e-6


@pytest.fixture(scope="module")
def intermittent_y():
    """Deterministic intermittent demand, n=200 (matches Python OM tests)."""
    rng = np.random.default_rng(41)
    return rng.poisson(0.3, 200).astype(float)


@pytest.fixture(scope="module")
def r_outputs(intermittent_y):
    """One R subprocess call per scenario — fits + extracts coef/vcov/confint.

    ``coef.omg`` returns NULL in the R package, so for OMG we recover the
    joint coef vector from the two sub-model ``$B`` slots (mirrors
    ``OMG.coef`` in Python). Coefficient names get ``A:``/``B:`` prefixes.
    """
    out = {}
    for scenario, (_, _, r_call) in SCENARIOS.items():
        is_omg = scenario.startswith("omg")
        coef_expr = (
            "{ ba <- as.numeric(m$modelA$B); names(ba) <- names(m$modelA$B);"
            " bb <- as.numeric(m$modelB$B); names(bb) <- names(m$modelB$B);"
            " list(values=c(ba, bb),"
            " names=c(paste0('A:', names(ba)), paste0('B:', names(bb)))) }"
            if is_omg
            else "list(values=as.numeric(coef(m)), names=names(coef(m)))"
        )
        expr = (
            f"{{ m <- {r_call};"
            f" ci <- confint(m, level=0.95);"
            "  list(coef=" + coef_expr + ","
            "       vcov=as.matrix(vcov(m)),"
            "       confint=as.matrix(ci),"
            "       confint_rownames=rownames(ci)) }"
        )
        out[scenario] = r_dict(expr, R_data={"y": intermittent_y})
    return out


def _python_fit(scenario, intermittent_y):
    cls, kw, _ = SCENARIOS[scenario]
    return cls(**kw).fit(intermittent_y)


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_coef_length_matches_r(scenario, intermittent_y, r_outputs):
    m = _python_fit(scenario, intermittent_y)
    r_coef = r_outputs[scenario]["coef"]
    assert len(m.coef) == len(r_coef["values"])


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_coef_values_match_r(scenario, intermittent_y, r_outputs):
    m = _python_fit(scenario, intermittent_y)
    r_B = np.asarray(r_outputs[scenario]["coef"]["values"], dtype=float)
    py_B = np.asarray(m.coef, dtype=float)
    np.testing.assert_allclose(
        py_B,
        r_B,
        rtol=COEF_RTOL,
        atol=COEF_ATOL,
        err_msg=f"{scenario}: coefficient values diverge from R",
    )


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_vcov_shape_matches_r(scenario, intermittent_y, r_outputs):
    m = _python_fit(scenario, intermittent_y)
    V = m.vcov().to_numpy()
    r_V = np.asarray(r_outputs[scenario]["vcov"], dtype=float)
    assert V.shape == r_V.shape, f"{scenario}: vcov shape {V.shape} != R {r_V.shape}"


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_vcov_values_match_r(scenario, intermittent_y, r_outputs):
    m = _python_fit(scenario, intermittent_y)
    V = m.vcov().to_numpy()
    r_V = np.asarray(r_outputs[scenario]["vcov"], dtype=float)
    np.testing.assert_allclose(
        V,
        r_V,
        rtol=VCOV_RTOL,
        atol=VCOV_ATOL,
        err_msg=f"{scenario}: joint covariance diverges from R",
    )


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_confint_se_matches_r(scenario, intermittent_y, r_outputs):
    m = _python_fit(scenario, intermittent_y)
    ci = m.confint(level=0.95)
    r_ci = np.asarray(r_outputs[scenario]["confint"], dtype=float)
    np.testing.assert_allclose(
        ci.iloc[:, 0].to_numpy(),
        r_ci[:, 0],
        rtol=SE_RTOL,
        atol=SE_ATOL,
        err_msg=f"{scenario}: standard errors diverge from R",
    )


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_confint_bounds_match_r(scenario, intermittent_y, r_outputs):
    m = _python_fit(scenario, intermittent_y)
    ci = m.confint(level=0.95)
    r_ci = np.asarray(r_outputs[scenario]["confint"], dtype=float)
    np.testing.assert_allclose(
        ci.iloc[:, 1:].to_numpy(),
        r_ci[:, 1:],
        rtol=CI_RTOL,
        atol=CI_ATOL,
        err_msg=f"{scenario}: confidence bounds diverge from R",
    )


def test_omg_coef_names_carry_ab_prefix(intermittent_y, r_outputs):
    """Joint OMG coefficient rows must be prefixed with ``A:`` / ``B:``
    (mirrors R's ``confint.omg`` row-naming convention)."""
    m = _python_fit("omg_mnn_mnn", intermittent_y)
    names = m.coef_names
    assert any(n.startswith("A:") for n in names)
    assert any(n.startswith("B:") for n in names)
    # And confint rownames from R follow the same convention.
    r_rownames = r_outputs["omg_mnn_mnn"]["confint_rownames"]
    assert any(n.startswith("A:") for n in r_rownames)
    assert any(n.startswith("B:") for n in r_rownames)


def test_summary_renders_for_each_scenario(intermittent_y):
    """Sanity: ``summary()`` produces a non-empty string for every scenario."""
    for scenario in SCENARIOS:
        m = _python_fit(scenario, intermittent_y)
        text = str(m.summary())
        assert len(text) > 50, f"{scenario}: summary unexpectedly short"
        if scenario.startswith("om_"):
            assert "Occurrence model" in text
        elif scenario == "omg_mnn_mnn":
            assert "Sub-model A" in text and "Sub-model B" in text


def test_logLik_matches_r(intermittent_y, r_outputs):  # noqa: N802
    """Log-likelihood should match R essentially exactly — same data, same
    initialiser, same optimiser. This is the litmus test for any latent
    cost-function divergence between the two implementations."""
    for scenario in SCENARIOS:
        m = _python_fit(scenario, intermittent_y)
        # Compute R's log-likelihood inline (separate small call).
        from ._r_bridge import r_eval

        _, _, r_call = SCENARIOS[scenario]
        r_ll = r_eval(f"as.numeric(logLik({r_call}))", R_data={"y": intermittent_y})
        r_ll_scalar = float(r_ll[0] if isinstance(r_ll, list) else r_ll)
        np.testing.assert_allclose(
            float(m.loglik),
            r_ll_scalar,
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"{scenario}: log-likelihood diverges from R",
        )


# Suppress an unused import warning when pandas is not referenced elsewhere.
_ = pd  # noqa: F841
