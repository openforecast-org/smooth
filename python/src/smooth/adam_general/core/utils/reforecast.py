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
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from smooth.adam_general.core.forecaster.result import ForecastResult
from smooth.adam_general.core.utils.distributions import generate_errors


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
    scale: Union[float, np.ndarray],
    *,
    n_obs: int,
    n_param: int,
    shape: Optional[float] = None,
    alpha: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample ``(h, nsim, nsim)`` error draws for ``reforecast``.

    Mirrors R's ``reforecast.adam`` (R/reapply.R), which draws from the same
    sampler as the simulated intervals.

    Parameters
    ----------
    distribution : str
        Distribution name, as in :func:`generate_errors`.
    h : int
        Forecast horizon.
    nsim : int
        Number of parameter draws (the cube is ``(h, nsim, nsim)``).
    scale : float or numpy.ndarray
        The de-biased scale, or one per horizon for an implanted scale model.
    n_obs, n_param : int
        Used by ``"dt"`` for the degrees of freedom.
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

    # A scale model gives one scale per horizon. R lets recycling spread it along
    # the first (horizon) axis of the F-ordered cube, which tiling reproduces.
    scale_values = np.asarray(scale, dtype=np.float64).ravel()
    scale_draw: Any = (
        np.tile(scale_values, n // scale_values.size)
        if scale_values.size > 1
        else float(scale_values[0])
    )
    e = generate_errors(
        distribution,
        n,
        scale_draw,
        obs_in_sample=n_obs,
        n_param=n_param,
        shape=shape,
        alpha=alpha,
        random_state=rng,
    )
    return np.asfortranarray(e.reshape((h, nsim, nsim), order="F"), dtype=np.float64)
