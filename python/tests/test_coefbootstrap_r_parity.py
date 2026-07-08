"""Distributional R-parity tests for ``coefbootstrap``.

R uses base-R's RNG, Python uses NumPy's PCG64; even given the same seed
the two implementations produce different replicate *sequences*. What
*should* match is the **bootstrap distribution** they're sampling from:

1. The replicate mean converges to the original coefficient (bootstrap
   consistency). Both sides should land in the same neighbourhood.
2. The empirical variance of each parameter should agree within a small
   factor — the Monte Carlo std error at ``nsim=200`` is roughly
   ``sd / sqrt(nsim) ≈ sd / 14``, so a factor-of-3 tolerance is generous
   but not vacuous.
3. The empirical covariance is positive semi-definite on both sides.

To remove one source of divergence we generate ``y`` on the R side and
feed the *exact same* numbers into the Python fit. The remaining
divergence is the bootstrap RNG mismatch, which is what the tolerances
above absorb.

Marked ``r_parity``; opt in with ``pytest -m r_parity``. Slow:
``nsim=200`` × 5 scenarios × R subprocess + Python fit ≈ 2-3 minutes.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pytest

from smooth import ADAM, OM, OMG

from ._r_bridge import r_dict

pytestmark = pytest.mark.r_parity

# How tight we expect distributional parity to be at nsim=200.
NSIM = 200
MEAN_RTOL = 0.5  # bootstrap mean within 50% of original coef on both sides
MEAN_ATOL = 0.05  # absolute slack for parameters near zero
# Empirical variance agreement within a small factor. Looser for binary
# models (OM/OMG) where alpha sits at the [0,1] boundary and small-sample
# replicates have heavier tails.
VAR_FACTOR_DEFAULT = 3.0
VAR_FACTOR_BINARY = 30.0
VAR_FLOOR = 1e-12  # treat both diagonals below this as "didn't vary"


def _is_psd(mat: np.ndarray, atol: float = 1e-8) -> bool:
    if mat.shape[0] == 0:
        return True
    eig = np.linalg.eigvalsh((mat + mat.T) / 2)
    return bool(np.all(eig >= -atol))


def _as_list(x):
    """Normalise jsonlite's `[x]` wrapping of scalar character vectors."""
    return x if isinstance(x, list) else [x]


@dataclass
class _Scenario:
    """One R-vs-Python comparison."""

    name: str
    # R side: a `{ ... }` block returning a named list with y, coef, boot.
    r_expr: str
    # Python-side fitter — takes the y array, returns a fitted model.
    py_fit: Any  # Callable[[np.ndarray], Any]
    # Variance-agreement tolerance (factor by which py/r diagonals may
    # disagree).
    var_factor: float = VAR_FACTOR_DEFAULT


def _scenarios() -> List[_Scenario]:
    """All five model specs probed in this parity sweep.

    Each R block:
      1. Sets a seed and generates ``y`` (or uses a stable rbinom for OM/OMG).
      2. Fits the model.
      3. Runs ``coefbootstrap(m, nsim=200, method="cr")`` under a second seed.
      4. Returns: ``y`` (so Python can fit on the same numbers), original
         coef + names, boot replicate matrix, boot vcov.
    """

    def _adam_block(spec: str, fit_args: str, sim_code: str) -> str:
        return (
            "{"
            f"{sim_code}; "
            f"m <- adam(y, {fit_args}); "
            "set.seed(101); "
            f"b <- coefbootstrap(m, nsim={NSIM}, method='cr'); "
            "list(y=as.numeric(y), "
            "coef=as.numeric(coef(m)), names=names(coef(m)), "
            "boot=as.numeric(t(b$coefficients)), boot_dim=dim(b$coefficients), "
            "vcov=as.numeric(b$vcov))"
            "}"
        )

    def _om_block() -> str:
        return (
            "{"
            "set.seed(31); y <- rbinom(200, 1, 0.4); "
            "m <- om(y, model='MNN', occurrence='odds-ratio'); "
            "set.seed(101); "
            f"b <- coefbootstrap(m, nsim={NSIM}, method='cr'); "
            "list(y=as.numeric(y), "
            "coef=as.numeric(coef(m)), names=names(coef(m)), "
            "boot=as.numeric(t(b$coefficients)), boot_dim=dim(b$coefficients), "
            "vcov=as.numeric(b$vcov))"
            "}"
        )

    def _omg_block() -> str:
        return (
            "{"
            "set.seed(31); y <- rbinom(200, 1, 0.4); "
            "m <- omg(y, modelA='ANN', modelB='ANN'); "
            "set.seed(101); "
            f"b <- coefbootstrap(m, nsim={NSIM}, method='cr'); "
            "list(y=as.numeric(y), "
            "coef=as.numeric(coef(m)), names=names(coef(m)), "
            "boot=as.numeric(t(b$coefficients)), boot_dim=dim(b$coefficients), "
            "vcov=as.numeric(b$vcov))"
            "}"
        )

    def _fit_adam(spec: str, **kw):
        def _f(y):
            return ADAM(model=spec, **kw).fit(y)

        return _f

    def _fit_adam_arima(**kw):
        def _f(y):
            return ADAM(model="NNN", ar_order=1, i_order=0, ma_order=1, **kw).fit(y)

        return _f

    def _fit_om(y):
        return OM(model="MNN", occurrence="odds-ratio").fit(y)

    def _fit_omg(y):
        return OMG(model_a="ANN", model_b="ANN").fit(y)

    return [
        _Scenario(
            name="ann",
            r_expr=_adam_block(
                "ANN",
                "model='ANN', initial='optimal'",
                "set.seed(11); y <- sim.es('ANN', obs=120, frequency=12, "
                "persistence=0.3)$data",
            ),
            py_fit=_fit_adam("ANN", initial="optimal"),
        ),
        _Scenario(
            name="aan",
            r_expr=_adam_block(
                "AAN",
                "model='AAN', initial='optimal'",
                "set.seed(12); y <- sim.es('AAN', obs=120, frequency=12, "
                "persistence=c(0.3, 0.1))$data",
            ),
            py_fit=_fit_adam("AAN", initial="optimal"),
        ),
        _Scenario(
            name="arima",
            r_expr=_adam_block(
                "ARIMA(1,0,1)",
                "model='NNN', orders=list(ar=1, i=0, ma=1)",
                "set.seed(13); y <- as.numeric(arima.sim(list(ar=0.4, ma=0.3), "
                "n=200)) + 20",
            ),
            py_fit=_fit_adam_arima(),
        ),
        _Scenario(
            name="om",
            r_expr=_om_block(),
            py_fit=_fit_om,
            var_factor=VAR_FACTOR_BINARY,
        ),
        _Scenario(
            name="omg",
            r_expr=_omg_block(),
            py_fit=_fit_omg,
            var_factor=VAR_FACTOR_BINARY,
        ),
    ]


@pytest.fixture(scope="module")
def r_results() -> Dict[str, Dict[str, Any]]:
    """Fetch every scenario's R result in one go (one Rscript invocation
    keeps the test suite snappier — devtools::load_all is expensive)."""
    blocks = []
    for s in _scenarios():
        blocks.append(f"{s.name}={s.r_expr}")
    expr = "list(" + ", ".join(blocks) + ")"
    raw = r_dict(expr)
    parsed: Dict[str, Dict[str, Any]] = {}
    for name, payload in raw.items():
        # ``coef.omg`` returns an unnamed vector, so ``names(coef(m))`` is
        # NULL and JSON drops it to ``{}``. Take ``boot_dim`` as the source
        # of truth for the parameter count; use names only when they
        # actually populate.
        coef_names = _as_list(payload["names"]) if payload["names"] else []
        boot_dim = payload["boot_dim"]  # [nrow, ncol]
        nrow = int(boot_dim[0])
        k = int(boot_dim[1])
        # R sent us t(coefficients) flattened: column-major of an (nrow, k)
        # matrix transposed, i.e. row-major of (nrow, k). Reshape accordingly.
        boot = np.asarray(payload["boot"], dtype=float).reshape(nrow, k)
        vcov = np.asarray(payload["vcov"], dtype=float).reshape(k, k)
        parsed[name] = dict(
            y=np.asarray(payload["y"], dtype=float),
            coef=np.asarray(payload["coef"], dtype=float),
            names=coef_names,
            boot=boot,
            vcov=vcov,
            k=k,
        )
    return parsed


@pytest.mark.parametrize("scen", _scenarios(), ids=lambda s: s.name)
def test_coefbootstrap_distribution_parity(scen, r_results):
    """Three parity assertions per scenario.

    1. Same parameter cardinality + matching names (modulo
       Python's prefix conventions for OMG).
    2. Bootstrap replicate mean is close to the original coef on both
       sides, and the two means land in the same neighbourhood.
    3. The diagonal of the empirical vcov agrees within a factor of
       ``VAR_FACTOR`` between R and Python (parameters whose variance is
       below ``VAR_FLOOR`` on both sides are skipped — they're numerically
       constant and tolerance ratios are meaningless).
    """
    r = r_results[scen.name]
    y_r = r["y"]

    # Python fit on the *same* data R generated. Suppress inner warnings so
    # pytest's r_parity output stays tidy.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = scen.py_fit(y_r)
        boot = m.coefbootstrap(nsim=NSIM, seed=101)

    py_coef = np.asarray(m.coef, dtype=float)
    py_coef_names = list(boot.coefficients.columns)
    py_boot = boot.coefficients.to_numpy()
    py_vcov = boot.vcov.to_numpy()

    # --- (1) parameter cardinality (names may differ for OMG: R drops
    # names on `coef.omg`, Python keeps `A:alpha` / `B:alpha`; for the
    # others both sides label identically, so we cross-check by length).
    assert len(py_coef_names) == r["k"], (
        f"{scen.name}: Python has {py_coef_names}, R has {r['k']} params"
    )

    # --- structural: PSD on both sides
    assert _is_psd(py_vcov), f"{scen.name}: Python vcov is not PSD"
    assert _is_psd(r["vcov"]), f"{scen.name}: R vcov is not PSD"

    # --- (2) replicate-mean ≈ original coef (bootstrap consistency).
    # OMG's R-side ``coef`` returns an unnamed empty vector, so skip the
    # against-coef check there and only compare R vs Python means.
    py_mean = py_boot.mean(axis=0)
    r_mean = r["boot"].mean(axis=0)
    if len(r["coef"]) == r["k"]:
        np.testing.assert_allclose(
            py_mean,
            py_coef,
            rtol=MEAN_RTOL,
            atol=MEAN_ATOL,
            err_msg=f"{scen.name}: Python replicate mean ≠ original coef",
        )
        np.testing.assert_allclose(
            r_mean,
            r["coef"],
            rtol=MEAN_RTOL,
            atol=MEAN_ATOL,
            err_msg=f"{scen.name}: R replicate mean ≠ original coef",
        )
    # And Python's mean lands near R's mean (always).
    np.testing.assert_allclose(
        py_mean,
        r_mean,
        rtol=MEAN_RTOL,
        atol=MEAN_ATOL,
        err_msg=f"{scen.name}: Python and R replicate means diverge",
    )

    # --- (3) empirical variance magnitude agreement
    py_var = np.diag(py_vcov)
    r_var = np.diag(r["vcov"])
    for i, name in enumerate(py_coef_names):
        if py_var[i] < VAR_FLOOR and r_var[i] < VAR_FLOOR:
            continue  # both numerically constant — skip
        # Guard against zero on one side only (would be a real disagreement)
        max_side = max(py_var[i], r_var[i])
        min_side = min(py_var[i], r_var[i])
        if min_side <= 0:
            pytest.fail(
                f"{scen.name}/{name}: one side has zero variance "
                f"(py={py_var[i]:.3g}, r={r_var[i]:.3g})"
            )
        ratio = max_side / min_side
        assert ratio <= scen.var_factor, (
            f"{scen.name}/{name}: vcov diagonal disagrees by factor "
            f"{ratio:.2f} (py={py_var[i]:.3g}, r={r_var[i]:.3g}; "
            f"tolerance={scen.var_factor:.1f}×)"
        )


def test_r_returns_bootstrap_class():
    """Sanity that the R fixture is calling the real coefbootstrap method
    (not a stale install with a different return shape)."""
    expr = (
        "{"
        "set.seed(41); "
        "y <- sim.es('ANN', obs=80, frequency=12, persistence=0.3)$data; "
        "m <- adam(y, model='ANN', initial='optimal'); "
        "set.seed(1); b <- coefbootstrap(m, nsim=10, method='cr'); "
        "list(class=class(b)[1], method=b$method)"
        "}"
    )
    r = r_dict(expr)
    cls = r["class"]
    if isinstance(cls, list):
        cls = cls[0]
    method = r["method"]
    if isinstance(method, list):
        method = method[0]
    assert cls == "bootstrap"
    assert method == "cr"
