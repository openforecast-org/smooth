"""Container for the output of the ``sim_*`` / ``ADAM.simulate`` family.

Mirrors R's S3 ``"smooth.sim"`` / ``"oes.sim"`` lists returned by
``sim.es``, ``sim.gum``, ``sim.ces``, ``sim.ssarima``, ``sim.sma``,
``sim.oes`` and ``simulate.adam``. The fields are the **union** of
those R lists; each ``sim_*`` populates only the relevant slots and
leaves the rest as ``None``. The ``__repr__`` / ``__str__`` reproduce
the layout of R's ``print.smooth.sim`` / ``print.oes.sim``
(``R/methods.R:1952`` and ``R/methods.R:2149``) so that
``print(sim_result)`` looks the same on both sides of the bridge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd


@dataclass
class SimulateResult:
    """Container mirroring R's ``smooth.sim`` / ``oes.sim`` S3 list.

    Attributes
    ----------
    model : str
        Model spec string ("ETS(MAM)", "ARIMA(1,1,1)[12]", "GUM(1,1)",
        "CES(...)" or "SMA(5)"). Prefixed with ``"i"`` (e.g. ``"iETS(MNN)"``)
        when ``probability < 1`` produced an intermittent series.
    data : pandas.Series | pandas.DataFrame
        Generated time series. ``Series`` (length ``obs``) when ``nsim=1``;
        ``DataFrame`` of shape ``(obs, nsim)`` otherwise.
    states : numpy.ndarray
        State trajectory cube of shape ``(n_components, obs + lag_max, nsim)``
        (matches R's ``arrVt``). When ``nsim=1`` the ``ts`` matrix R returns
        is a 2-D ``(n_components, obs + lag_max)`` slice — we keep the cube
        shape for consistency.
    residuals : pandas.Series | pandas.DataFrame
        Error terms passed through the simulator (same shape as ``data``).
    occurrence : numpy.ndarray, optional
        ``(obs, nsim)`` 0/1 occurrence mask when ``probability < 1``;
        ``None`` otherwise.
    probability : numpy.ndarray | float, optional
        The probability vector (scalar or length-``obs``) used to draw
        the occurrence mask. ``None`` for non-intermittent series.
    persistence : numpy.ndarray, optional
        ``(persistenceLength, nsim)`` smoothing parameters
        (``alpha``, ``beta``, ``gamma``). Set by ``sim_es`` / ``sim_oes``.
    phi : float, optional
        Damping parameter. Set by ``sim_es`` / ``sim_oes`` (default ``1``).
    initial : numpy.ndarray, optional
        Initial states for level/trend (set by ``sim_es`` / ``sim_oes``).
    initial_season : numpy.ndarray, optional
        Initial seasonal states (set by ``sim_es`` / ``sim_oes``).
    measurement : numpy.ndarray, optional
        Measurement vector ``w`` (``sim_gum``).
    transition : numpy.ndarray, optional
        Transition matrix ``F`` (``sim_gum``).
    a, b : complex | numpy.ndarray, optional
        Complex smoothing parameters for CES (``sim_ces``).
    arma : dict, optional
        ``{"ar": ..., "ma": ...}`` with the ARMA coefficients used
        (``sim_ssarima``). Shape ``(sum(orders), nsim)``.
    constant : float, optional
        ARIMA constant term (``sim_ssarima``).
    profile : numpy.ndarray, optional
        Final recent-profile cube ``(n_components, lag_max, nsim)``.
    intermittent : str, optional
        ``"none"``, ``"fixed"`` or ``"tsb"`` — matches R's
        ``$intermittent`` field.
    log_lik : numpy.ndarray, optional
        Per-series true log-likelihood (R's ``$logLik``).
    other : dict
        Extra parameters captured from ``**kwargs`` (R's ``$other``).
    model_a, model_b : SimulateResult, optional
        Sub-models for ``sim_oes`` (R's ``$modelA`` / ``$modelB``).
    occurrence_type : str, optional
        ``"odds-ratio"``, ``"inverse-odds-ratio"``, ``"direct"`` or
        ``"general"`` — R's ``$occurrence`` on ``"oes.sim"``.
    """

    model: str
    data: Union[pd.Series, pd.DataFrame]
    states: np.ndarray
    residuals: Union[pd.Series, pd.DataFrame]
    occurrence: Optional[np.ndarray] = None
    probability: Optional[Union[np.ndarray, float]] = None
    persistence: Optional[np.ndarray] = None
    phi: Optional[float] = None
    initial: Optional[np.ndarray] = None
    initial_season: Optional[np.ndarray] = None
    measurement: Optional[np.ndarray] = None
    transition: Optional[np.ndarray] = None
    a: Optional[Union[complex, np.ndarray]] = None
    b: Optional[Union[complex, np.ndarray]] = None
    arma: Optional[Dict[str, np.ndarray]] = None
    constant: Optional[float] = None
    profile: Optional[np.ndarray] = None
    intermittent: Optional[str] = None
    log_lik: Optional[np.ndarray] = None
    other: Dict[str, Any] = field(default_factory=dict)
    model_a: Optional["SimulateResult"] = None
    model_b: Optional["SimulateResult"] = None
    occurrence_type: Optional[str] = None
    # Pre-link state-space output (used by ``OM.simulate`` to expose
    # the latent ETS draws so ``OMG.simulate`` can feed them to
    # ``omg_link_function`` for the joint probability).
    latent: Optional[np.ndarray] = None

    @property
    def _nsim(self) -> int:
        if isinstance(self.data, pd.Series):
            return 1
        return int(self.data.shape[1])

    @property
    def _smooth_type(self) -> str:
        """Detect which family this result belongs to — drives ``__str__``."""
        if self.model_a is not None or self.occurrence_type is not None:
            return "OES"
        if self.a is not None:
            return "CES"
        if self.arma is not None:
            return "ARIMA"
        if self.model.startswith("SMA") or self.model.startswith("iSMA"):
            return "SMA"
        if self.model.startswith("GUM") or self.model.startswith("iGUM"):
            return "GUM"
        return "ETS"

    def __repr__(self) -> str:  # noqa: D401
        return self.__str__()

    def __str__(self, digits: int = 4) -> str:
        """R-flavoured printout. Mirrors ``print.smooth.sim`` line-for-line.

        For ``nsim > 1`` R prints only the header (model + count) — we
        match that. For ``nsim == 1`` the per-family branch reproduces
        persistence / ARMA / CES / SMA / OES summaries.
        """
        nsim = self._nsim
        lines = [
            f"Data generated from: {self.model}",
            f"Number of generated series: {nsim}",
        ]
        if self._smooth_type == "OES":
            obs = (
                len(self.occurrence)
                if isinstance(self.occurrence, np.ndarray) and self.occurrence.ndim == 1
                else int(self.data.shape[0])
            )
            lines.append(f"Number of observations in each series: {obs}")

        if nsim == 1:
            kind = self._smooth_type
            if kind == "ETS":
                if self.persistence is not None:
                    lines.append("Persistence vector:")
                    names = _persistence_names(self.persistence)
                    vec = np.asarray(self.persistence).reshape(-1)
                    lines.append(_format_named_vector(names, vec, digits))
                if self.phi is not None and float(self.phi) != 1.0:
                    lines.append(f"Phi: {self.phi}")
                if self.occurrence is not None and np.any(self.occurrence != 1):
                    lines.append("The data is produced based on an occurrence model.")
                lines.append(_format_loglik(self.log_lik, digits))
            elif kind == "ARIMA":
                lines.extend(self._format_arima_lines(digits))
                lines.append(_format_loglik(self.log_lik, digits))
            elif kind == "CES":
                lines.append(f"Smoothing parameter a: {_round_c(self.a, digits)}")
                if self.b is not None:
                    lines.append(f"Smoothing parameter b: {_round_c(self.b, digits)}")
                lines.append(_format_loglik(self.log_lik, digits))
            elif kind == "SMA":
                lines.append(_format_loglik(self.log_lik, digits))
            elif kind == "OES":
                lines.append(_format_loglik(self.log_lik, digits))
            elif kind == "GUM":
                # R's ``print.smooth.sim`` has no GUM-specific branch —
                # just emit the true log-likelihood and stop.
                lines.append(_format_loglik(self.log_lik, digits))
        return "\n".join(lines)

    def _format_arima_lines(self, digits: int) -> list:
        """ARIMA block of R's ``print.smooth.sim``.

        Renders an ``AR(i)`` table indexed by lag and an ``MA(i)`` table
        indexed by lag, matching R's matrix dimnames. ``constant`` and
        differences (``I(...)``) are appended below if present.
        """
        out = []
        arma = self.arma or {}
        ar = np.asarray(arma.get("ar", [])).reshape(-1)
        ma = np.asarray(arma.get("ma", [])).reshape(-1)
        if ar.size:
            out.append("AR parameters:")
            out.append(_format_vector_block("AR", ar, digits))
        if ma.size:
            out.append("MA parameters:")
            out.append(_format_vector_block("MA", ma, digits))
        if self.constant is not None and not np.isnan(self.constant):
            out.append(f"Constant value: {round(float(self.constant), digits)}")
        return out


def _persistence_names(persistence: np.ndarray) -> list:
    """Map a persistence vector length to ETS smoothing-parameter names."""
    arr = np.asarray(persistence).reshape(-1)
    n = arr.shape[0]
    if n == 1:
        return ["alpha"]
    if n == 2:
        # Two-component vectors are either alpha+beta (T!=N) or
        # alpha+gamma (S!=N). R prefers alpha+beta when both exist.
        return ["alpha", "beta"]
    return ["alpha", "beta", "gamma"]


def _format_named_vector(names: list, values: np.ndarray, digits: int) -> str:
    """Format an R-style named numeric vector — two aligned rows."""
    name_cells = [f"{nm:>{max(len(nm), digits + 4)}}" for nm in names]
    val_cells = [
        f"{round(float(v), digits):>{max(len(names[i]), digits + 4)}}"
        for i, v in enumerate(values)
    ]
    return "  ".join(name_cells) + "\n" + "  ".join(val_cells)


def _format_vector_block(prefix: str, values: np.ndarray, digits: int) -> str:
    """Format an AR / MA coefficient vector with R-style row names."""
    rows = [f"{prefix}({i + 1})" for i in range(values.shape[0])]
    return _format_named_vector(rows, values, digits)


def _format_loglik(log_lik, digits: int) -> str:
    """One-line ``True likelihood:`` summary that matches R's output."""
    if log_lik is None:
        return "True likelihood: NA"
    arr = np.atleast_1d(np.asarray(log_lik, dtype=float))
    if arr.size == 1:
        val = float(arr[0])
        if np.isnan(val):
            return "True likelihood: NA"
        return f"True likelihood: {round(val, digits)}"
    return "True likelihood: " + ", ".join(
        "NA" if np.isnan(v) else str(round(float(v), digits)) for v in arr
    )


def _round_c(x, digits: int):
    """Round a real or complex value the same way R's ``round`` would."""
    if isinstance(x, complex):
        return complex(round(x.real, digits), round(x.imag, digits))
    arr = np.atleast_1d(np.asarray(x))
    if arr.size == 1:
        v = arr.ravel()[0]
        if isinstance(v, complex) or np.iscomplexobj(v):
            return complex(round(v.real, digits), round(v.imag, digits))
        return round(float(v), digits)
    return np.round(arr, digits)
