"""Randomizer dispatch for the ``sim_*`` family.

Mirrors R's ``do.call(randomizer, ellipsis)`` pattern in ``sim.es``
(``R/simes.R:460``), ``sim.gum``, ``sim.ssarima``, ``sim.ces``,
``sim.oes`` — accept either an R-style randomizer name (``"rnorm"``,
``"rlnorm"``, ``"rt"``, ``"rlaplace"``, ``"rs"``, ``"rgnorm"``,
``"rgamma"``, ``"rinvgauss"``, ``"rbeta"``) or a plain Python callable.

The callable form enables the R↔Python parity-test trick used in the
``test_*_r_parity`` modules: pre-generate identical errors in one
language, then pass them to both via a callable that just yields the
stored values. With identical errors feeding the **same** C++
``adamCore::simulate`` kernel, the two outputs agree to floating-point.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np
from scipy import stats as _stats

from smooth.adam_general.core.utils.distributions import (
    ralaplace,
    rgnorm,
    rlaplace,
    rs,
)

# R names accepted by ``sim.*`` (R/simes.R:117, R/simgum.R:81, etc.).
# All standardised on (sd=1 / scale=1 / ...) defaults so a plain
# ``randomizer="rnorm"`` reproduces R's default ``rnorm(n)``.
_R_DEFAULTS = {
    "rnorm",
    "rlnorm",
    "rt",
    "rlaplace",
    "rs",
    "rgnorm",
    "rgamma",
    "rinvgauss",
    "rbeta",
}


def _draw_rnorm(n, rng, **kw):
    return rng.normal(kw.get("mean", 0.0), kw.get("sd", 1.0), n)


def _draw_rlnorm(n, rng, **kw):
    return rng.lognormal(kw.get("meanlog", 0.0), kw.get("sdlog", 1.0), n)


def _draw_rt(n, rng, **kw):
    df = kw.get("df", None)
    if df is None:
        raise ValueError("'rt' requires a 'df' argument (degrees of freedom).")
    return rng.standard_t(df, n)


def _draw_rlaplace(n, rng, **kw):
    return rlaplace(n, kw.get("mu", 0.0), kw.get("b", 1.0), random_state=rng)


def _draw_rs(n, rng, **kw):
    return rs(n, kw.get("mu", 0.0), kw.get("b", 1.0), random_state=rng)


def _draw_rgnorm(n, rng, **kw):
    return rgnorm(
        n,
        kw.get("mu", 0.0),
        kw.get("alpha", 1.0),
        kw.get("beta", 2.0),
        random_state=rng,
    )


def _draw_rgamma(n, rng, **kw):
    shape = kw.get("shape", None)
    if shape is None:
        raise ValueError("'rgamma' requires a 'shape' argument.")
    return rng.gamma(shape, kw.get("scale", 1.0), n)


def _draw_rinvgauss(n, rng, **kw):
    mean = kw.get("mean", 1.0)
    shape = kw.get("shape", kw.get("dispersion", 1.0))
    lam = 1.0 / shape if "dispersion" in kw else shape
    return _stats.invgauss.rvs(mean / lam, scale=lam, size=n, random_state=rng)


def _draw_ralaplace(n, rng, **kw):
    return ralaplace(
        n,
        kw.get("mu", 0.0),
        kw.get("b", 1.0),
        kw.get("alpha", 0.5),
        random_state=rng,
    )


def _draw_rbeta(n, rng, **kw):
    shape1 = kw.get("shape1", kw.get("alpha", 1.0))
    shape2 = kw.get("shape2", kw.get("beta", 1.0))
    return rng.beta(shape1, shape2, n)


_R_DISPATCH = {
    "rnorm": _draw_rnorm,
    "rlnorm": _draw_rlnorm,
    "rt": _draw_rt,
    "rlaplace": _draw_rlaplace,
    "rs": _draw_rs,
    "rgnorm": _draw_rgnorm,
    "rgamma": _draw_rgamma,
    "rinvgauss": _draw_rinvgauss,
    "ralaplace": _draw_ralaplace,
    "rbeta": _draw_rbeta,
}


def resolve_randomizer(
    randomizer: Union[str, Callable],
    rng: Optional[np.random.Generator] = None,
    **kwargs,
) -> Callable[[int], np.ndarray]:
    """Return a callable ``f(n) -> ndarray`` for ``sim_*`` error draws.

    Parameters
    ----------
    randomizer : str | callable
        R-style randomizer name (``"rnorm"``, ``"rlnorm"``, ``"rt"``,
        ``"rlaplace"``, ``"rs"``, ``"rgnorm"``, ``"rgamma"``,
        ``"rinvgauss"``, ``"ralaplace"``, ``"rbeta"``) **or** a Python
        callable that takes a single integer ``n`` (R's ``do.call``
        convention) and returns a length-``n`` array. The latter form
        is what the R-parity tests use to feed pre-generated errors.
    rng : numpy.random.Generator, optional
        RNG to use for string-named randomizers. Ignored when
        ``randomizer`` is callable (the caller's callable owns its
        randomness).
    **kwargs
        Forwarded to the per-distribution sampler — these are R's
        ``...`` arguments (e.g. ``sd=0.5`` for ``rnorm``, ``shape=1``
        for ``rgamma``).

    Returns
    -------
    callable
        A function ``f(n)`` that returns a flat ``ndarray`` of length
        ``n``.
    """
    if callable(randomizer):

        def _from_callable(n: int) -> np.ndarray:
            out = np.asarray(randomizer(n)).reshape(-1)
            if out.shape[0] < n:
                raise ValueError(
                    f"randomizer callable returned {out.shape[0]} values, "
                    f"need at least {n}."
                )
            return out[:n]

        return _from_callable

    if not isinstance(randomizer, str):
        raise TypeError(
            f"randomizer must be a string or callable; got {type(randomizer).__name__}"
        )
    if randomizer not in _R_DISPATCH:
        raise ValueError(
            f"Unknown randomizer: {randomizer!r}. Supported: {sorted(_R_DISPATCH)}"
        )

    rng = rng if rng is not None else np.random.default_rng()
    draw = _R_DISPATCH[randomizer]

    def _from_name(n: int) -> np.ndarray:
        return np.asarray(draw(n, rng, **kwargs)).reshape(-1)

    return _from_name


def is_default_randomizer(randomizer: Union[str, Callable]) -> bool:
    """Return True iff ``randomizer`` is one of R's no-extra-args defaults.

    Used by ``sim_es`` to decide whether to apply R's variance-scaling
    rules (``R/simes.R:471-493``): "default" randomizers get error-type-
    specific scaling, custom callables get passed through unchanged.
    """
    if callable(randomizer):
        return False
    return randomizer in _R_DEFAULTS
