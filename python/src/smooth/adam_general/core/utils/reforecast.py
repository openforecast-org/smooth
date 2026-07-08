"""Container + helpers for ``ADAM.reforecast``.

Mirrors R's ``"adam.forecast"`` / ``"adam.predict"`` S3 list returned by
``reforecast.adam`` (R/reapply.R:941-1402). The point of having a
dedicated dataclass — rather than reusing :class:`ForecastResult` — is
that ``reforecast`` carries the full ``(h, nsim, nsim)`` paths cube
(R's ``$paths``) that the caller may want for downstream diagnostics.
A :meth:`to_forecast_result` helper compresses it into the standard
``ForecastResult`` shape so :meth:`ADAM.predict` can dispatch through
``reforecast`` for ``interval="complete"`` / ``"confidence"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as _stats

from smooth.adam_general.core.forecaster.result import ForecastResult
from smooth.adam_general.core.utils.distributions import (
    ralaplace,
    rgnorm,
    rs,
)


@dataclass
class ReforecastResult:
    """Container for an ``ADAM.reforecast`` run.

    Attributes
    ----------
    mean : pandas.Series
        Point forecasts indexed by the forecast period (length ``h`` for
        non-cumulative, length ``1`` for cumulative).
    lower : pandas.DataFrame | None
        Lower interval bounds; ``(h, n_levels)`` (columns labelled by the
        per-side quantile string R uses). ``None`` when
        ``interval="none"``.
    upper : pandas.DataFrame | None
        Upper interval bounds; same shape as ``lower``.
    level : list[float]
        Confidence levels used (always a list, even for a single level).
    interval : str
        ``"prediction"``, ``"confidence"`` or ``"none"``.
    side : str
        ``"both"``, ``"upper"`` or ``"lower"``.
    cumulative : bool
        Whether the point + intervals are cumulative.
    h : int
        Forecast horizon (the value passed in by the caller).
    paths : numpy.ndarray | None
        ``(h, nsim, nsim)`` cube of simulated trajectories (R's
        ``$paths``). ``None`` only when ``h<=0`` (no forward sim was
        run).
    model : str
        Model spec string.
    """

    mean: pd.Series
    lower: Optional[pd.DataFrame]
    upper: Optional[pd.DataFrame]
    level: list
    interval: str
    side: str
    cumulative: bool
    h: int
    paths: Optional[np.ndarray]
    model: str

    def __repr__(self) -> str:
        return (
            f"ReforecastResult(model={self.model!r}, h={self.h}, "
            f"interval={self.interval!r}, side={self.side!r}, "
            f"cumulative={self.cumulative}, n_levels={len(self.level)})"
        )

    def to_forecast_result(self) -> ForecastResult:
        """Project onto the standard :class:`ForecastResult` shape.

        Used by :meth:`ADAM.predict` to forward
        ``interval="complete"`` / ``"confidence"`` through ``reforecast``
        without exposing the cube to callers that only expect the
        ``mean / lower / upper / level / side / interval`` set.
        """
        return ForecastResult(
            mean=self.mean,
            lower=self.lower,
            upper=self.upper,
            level=self.level if len(self.level) > 1 else self.level[0],
            side=self.side,
            interval=self.interval,
        )


def sample_reforecast_errors(
    distribution: str,
    h: int,
    nsim: int,
    sigma: float,
    *,
    n_obs: int,
    n_param: int,
    opt_scale: float,
    shape: Optional[float] = None,
    alpha: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample ``(h, nsim, nsim)`` error draws for ``reforecast``.

    Mirrors R's ``reforecast.adam`` switch block (R/reapply.R:1254-1273).
    The per-distribution conversion from ``sigma(object)`` to the
    distributional scale parameter matches R verbatim — this is the
    correct conversion for the reforecast path, regardless of how
    ``intervals.py``'s ``generate_errors`` chooses to interpret its
    ``scale`` argument elsewhere.

    Parameters
    ----------
    distribution : str
        Distribution name (``"dnorm"``, ``"dlaplace"``, ``"ds"``,
        ``"dgnorm"``, ``"dlogis"``, ``"dt"``, ``"dalaplace"``,
        ``"dlnorm"``, ``"dinvgauss"``, ``"dgamma"``, ``"dllaplace"``,
        ``"dls"``, ``"dlgnorm"``).
    h : int
        Forecast horizon.
    nsim : int
        Number of parameter draws (the cube is ``(h, nsim, nsim)``).
    sigma : float
        Empirical residual std-dev from :attr:`ADAM.sigma`. Used by all
        distributions except ``"dt"`` (where the t-distribution is
        scaled by it externally) and ``"dlnorm"`` which uses
        ``opt_scale``.
    n_obs, n_param : int
        Used by ``"dt"`` for the degrees of freedom.
    opt_scale : float
        :attr:`ADAM.scale` — the internal optimisation scale, equal to
        R's ``extractScale(object)``. Only consumed by the ``"dlnorm"``
        branch (mean-log shift and sigma-log of the log-normal).
    shape : float, optional
        Shape parameter for ``"dgnorm"`` / ``"dlgnorm"``.
    alpha : float, optional
        Asymmetry parameter for ``"dalaplace"``.
    rng : numpy.random.Generator, optional
        RNG for reproducibility.

    Returns
    -------
    numpy.ndarray
        ``(h, nsim, nsim)`` F-ordered cube ready for
        :meth:`adamCore.reforecast`.
    """
    rng = rng or np.random.default_rng()
    n = h * nsim * nsim

    if distribution == "dnorm":
        e = rng.normal(0.0, sigma, n)
    elif distribution == "dlaplace":
        e = rng.laplace(0.0, sigma / 2.0, n)
    elif distribution == "ds":
        e = rs(n, 0.0, (sigma**2 / 120.0) ** 0.25, random_state=rng)
    elif distribution == "dgnorm":
        if shape is None:
            raise ValueError("'dgnorm' requires a shape parameter.")
        from math import gamma as _gamma

        gnorm_scale = sigma * np.sqrt(_gamma(1.0 / shape) / _gamma(3.0 / shape))
        e = rgnorm(n, 0.0, gnorm_scale, shape, random_state=rng)
    elif distribution == "dlogis":
        e = rng.logistic(0.0, sigma * np.sqrt(3.0) / np.pi, n)
    elif distribution == "dt":
        df = max(n_obs - n_param, 1)
        e = rng.standard_t(df, n)
    elif distribution == "dalaplace":
        if alpha is None:
            raise ValueError("'dalaplace' requires an alpha parameter.")
        a2 = alpha**2
        oma2 = (1.0 - alpha) ** 2
        scale = np.sqrt(sigma**2 * a2 * oma2 / (a2 + oma2))
        e = ralaplace(n, 0.0, scale, alpha, random_state=rng)
    elif distribution == "dlnorm":
        # R uses ``extractScale(object)`` (the optimisation scale) here,
        # not ``sigma(object)``. Without subtracting 1 the multiplier
        # would have mean ``exp(sigma^2/2 - sigma^2/2) = 1`` exactly.
        meanlog = -(opt_scale**2) / 2.0
        e = rng.lognormal(meanlog, opt_scale, n) - 1.0
    elif distribution == "dinvgauss":
        # rinvgauss(n, mean=1, dispersion=sigma^2) - 1
        # scipy's invgauss is parametrised as ``mu`` (mean) with
        # ``scale=lambda`` where ``lambda = 1/dispersion``.
        lam = 1.0 / (sigma**2)
        e = _stats.invgauss.rvs(1.0 / lam, scale=lam, size=n, random_state=rng) - 1.0
    elif distribution == "dgamma":
        # rgamma(n, shape=sigma^-2, scale=sigma^2) - 1
        e = rng.gamma(1.0 / (sigma**2), sigma**2, n) - 1.0
    elif distribution == "dllaplace":
        e = np.exp(rng.laplace(0.0, sigma / 2.0, n)) - 1.0
    elif distribution == "dls":
        e = np.exp(rs(n, 0.0, (sigma**2 / 120.0) ** 0.25, random_state=rng)) - 1.0
    elif distribution == "dlgnorm":
        if shape is None:
            raise ValueError("'dlgnorm' requires a shape parameter.")
        from math import gamma as _gamma

        gnorm_scale = sigma * np.sqrt(_gamma(1.0 / shape) / _gamma(3.0 / shape))
        e = np.exp(rgnorm(n, 0.0, gnorm_scale, shape, random_state=rng)) - 1.0
    else:
        raise ValueError(f"Unsupported distribution for reforecast: {distribution!r}")

    return np.asfortranarray(e.reshape((h, nsim, nsim), order="F"), dtype=np.float64)
