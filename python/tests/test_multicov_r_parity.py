"""Numerical R-parity tests for ``multicov`` (analytical mode).

Unlike the coefbootstrap parity tests — which are stochastic — the
analytical multicov is a deterministic closed-form expression from the
fitted state-space matrices ``(F, W, g, σ²)``. Given identical input ``y``
on both sides, R and Python should produce the same fitted state and
hence the same covariance to machine precision (the only divergence is
optimiser-determined floating-point noise in the fitted parameters,
which is already accounted for in the wider r_parity suite).

This file uses **the same ``y`` on both sides** by generating the series
in R and feeding the exact numbers back into the Python fit (no RNG
mismatch).

Marked ``r_parity``; the existing ``conftest.py`` hook skips it on CI.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import numpy as np
import pytest

from smooth import ADAM, OM

from ._r_bridge import r_dict

pytestmark = pytest.mark.r_parity

# After (a) the per-parameter relative FD Hessian step in hessianCore.h
# and (b) the shared olsCore.h backend for msdecompose's global smoother,
# R and Python converge to the same B on these scenarios (the OLS ULP
# that used to propagate through x0 -> NLopt -> different B is gone), so
# the analytical formula -- and the empirical path that calls the shared
# adamCore::ferrors backend -- agree to machine precision. The historical
# ANALYTICAL_RTOL = 0.05 reflected the old gap and is no longer current.
ANALYTICAL_RTOL = 1e-10
ANALYTICAL_ATOL = 1e-12
H = 6


@dataclass
class _Scenario:
    name: str
    r_expr: str
    py_fit: Callable[[np.ndarray], Any]


def _scenarios() -> List[_Scenario]:
    """R-side blocks: simulate y, fit, compute multicov, return everything."""

    def _adam_block(sim_code: str, fit_args: str) -> str:
        return (
            "{"
            f"{sim_code}; "
            f"m <- adam(y, {fit_args}); "
            f"M <- multicov(m, type='analytical', h={H}); "
            f"Memp <- multicov(m, type='empirical', h={H}); "
            "list(y=as.numeric(y), "
            "M=as.numeric(M), Mdim=dim(M), "
            "Memp=as.numeric(Memp), "
            "sigma=as.numeric(sigma(m)))"
            "}"
        )

    def _fit_adam(model: str, **kw):
        def _f(y):
            return ADAM(model=model, **kw).fit(y)

        return _f

    def _fit_adam_arima(**kw):
        def _f(y):
            return ADAM(model="NNN", ar_order=1, i_order=0, ma_order=1, **kw).fit(y)

        return _f

    def _fit_om(y):
        return OM(model="MNN", occurrence="odds-ratio").fit(y)

    om_block = (
        "{"
        "set.seed(31); y <- rbinom(200, 1, 0.4); "
        "m <- om(y, model='MNN', occurrence='odds-ratio'); "
        f"M <- multicov(m, type='analytical', h={H}); "
        f"Memp <- multicov(m, type='empirical', h={H}); "
        "list(y=as.numeric(y), "
        "M=as.numeric(M), Mdim=dim(M), "
        "Memp=as.numeric(Memp), "
        "sigma=as.numeric(sigma(m)))"
        "}"
    )

    return [
        _Scenario(
            name="ann",
            r_expr=_adam_block(
                "set.seed(11); y <- sim.es('ANN', obs=120, frequency=12, "
                "persistence=0.3)$data",
                "model='ANN', initial='optimal'",
            ),
            py_fit=_fit_adam("ANN", initial="optimal"),
        ),
        _Scenario(
            name="aan",
            r_expr=_adam_block(
                "set.seed(12); y <- sim.es('AAN', obs=120, frequency=12, "
                "persistence=c(0.3, 0.1))$data",
                "model='AAN', initial='optimal'",
            ),
            py_fit=_fit_adam("AAN", initial="optimal"),
        ),
        _Scenario(
            name="arima",
            r_expr=_adam_block(
                "set.seed(13); y <- as.numeric(arima.sim(list(ar=0.4, ma=0.3), "
                "n=200)) + 20",
                "model='NNN', orders=list(ar=1, i=0, ma=1)",
            ),
            py_fit=_fit_adam_arima(),
        ),
        _Scenario(
            name="om_mnn",
            r_expr=om_block,
            py_fit=_fit_om,
        ),
    ]


@pytest.fixture(scope="module")
def r_results() -> Dict[str, Dict[str, Any]]:
    """One Rscript subprocess for the whole suite (devtools::load_all is
    the slow part — call it once)."""
    blocks = [f"{s.name}={s.r_expr}" for s in _scenarios()]
    expr = "list(" + ", ".join(blocks) + ")"
    raw = r_dict(expr)
    parsed: Dict[str, Dict[str, Any]] = {}
    for name, payload in raw.items():
        mdim = payload["Mdim"]
        k = int(mdim[0])
        M = np.asarray(payload["M"], dtype=float).reshape(k, k)
        Memp = np.asarray(payload["Memp"], dtype=float).reshape(k, k)
        parsed[name] = dict(
            y=np.asarray(payload["y"], dtype=float),
            M=M,
            Memp=Memp,
            sigma=float(
                payload["sigma"][0]
                if isinstance(payload["sigma"], list)
                else payload["sigma"]
            ),
        )
    return parsed


@pytest.mark.parametrize("scen", _scenarios(), ids=lambda s: s.name)
def test_multicov_analytical_parity(scen, r_results):
    """Closed-form covariance: Python's multicov should match R's to a
    small relative tolerance — the only source of divergence is the
    optimiser-determined σ² (a single scalar)."""
    r = r_results[scen.name]
    y_r = r["y"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = scen.py_fit(y_r)
        M_py = m.multicov(type="analytical", h=H).to_numpy()

    M_r = r["M"]

    # Symmetry on both sides.
    assert np.allclose(M_py, M_py.T, atol=1e-10), (
        f"{scen.name}: Python multicov not symmetric"
    )
    assert np.allclose(M_r, M_r.T, atol=1e-10), f"{scen.name}: R multicov not symmetric"

    # σ²-level agreement: M scales linearly with σ², so the matrices
    # agree iff (σ²_py / σ²_r) is close to 1 and the unitless shape
    # matches.
    np.testing.assert_allclose(
        M_py,
        M_r,
        rtol=ANALYTICAL_RTOL,
        atol=ANALYTICAL_ATOL,
        err_msg=f"{scen.name}: analytical multicov disagrees with R",
    )


@pytest.mark.parametrize("scen", _scenarios(), ids=lambda s: s.name)
def test_multicov_empirical_parity(scen, r_results):
    """Empirical covariance: both R and Python call the same C++
    ``adamCore::ferrors`` backend for the per-cell residuals, so the
    only divergence is fitted-parameter optimiser noise (which feeds
    through to the rolled-forward state). Same tolerance as analytical."""
    r = r_results[scen.name]
    y_r = r["y"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = scen.py_fit(y_r)
        M_py = m.multicov(type="empirical", h=H).to_numpy()

    M_r = r["Memp"]

    assert np.allclose(M_py, M_py.T, atol=1e-10), (
        f"{scen.name}: Python empirical multicov not symmetric"
    )
    assert np.allclose(M_r, M_r.T, atol=1e-10), (
        f"{scen.name}: R empirical multicov not symmetric"
    )
    np.testing.assert_allclose(
        M_py,
        M_r,
        rtol=ANALYTICAL_RTOL,
        atol=ANALYTICAL_ATOL,
        err_msg=f"{scen.name}: empirical multicov disagrees with R",
    )


def test_multicov_simulated_shape_parity():
    """Simulated path uses different RNG between R and Python — only
    structural parity (shape, symmetry, PSD) is meaningful."""
    expr = (
        "{"
        "set.seed(11); "
        "y <- sim.es('ANN', obs=120, frequency=12, persistence=0.3)$data; "
        "m <- adam(y, model='ANN', initial='optimal'); "
        f"M <- multicov(m, type='simulated', h={H}, nsim=200); "
        "list(y=as.numeric(y), M=as.numeric(M), Mdim=dim(M))"
        "}"
    )
    r = r_dict(expr)
    M_r = np.asarray(r["M"], dtype=float).reshape(H, H)
    y_r = np.asarray(r["y"], dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = ADAM(model="ANN", initial="optimal").fit(y_r)
        M_py = m.multicov(type="simulated", h=H, nsim=200).to_numpy()

    assert M_py.shape == M_r.shape == (H, H)
    assert np.allclose(M_py, M_py.T, atol=1e-6)
    assert np.allclose(M_r, M_r.T, atol=1e-6)
    # Both PSD (allow small numerical slack on either side).
    for label, mat in (("py", M_py), ("r", M_r)):
        eig = np.linalg.eigvalsh((mat + mat.T) / 2)
        assert np.all(eig >= -1e-3), f"{label} multicov not PSD: eig={eig}"
