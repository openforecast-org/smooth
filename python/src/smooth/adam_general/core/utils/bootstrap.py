"""Shared utilities for coefficient bootstrap (case resampling).

Implements the Python equivalent of R's ``coefbootstrap`` infrastructure
shared between ADAM, OM and OMG: a ``BootstrapResult`` container that
matches the field set of R's ``"bootstrap"`` S3 class, plus the index
samplers used to build replicate datasets.

Only ``method="cr"`` (case resampling) is supported. ``method="dsr"``
(data-shape replication, R's ``greybox::dsrboot``) is not yet ported.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class BootstrapResult:
    """Container for a coefficient-bootstrap run.

    Mirrors R's ``"bootstrap"``-class list returned by ``coefbootstrap.adam``
    / ``coefbootstrap.om`` / ``coefbootstrap.omg``.

    Attributes
    ----------
    vcov : pd.DataFrame
        Empirical variance-covariance of the bootstrap replicates
        (``Xc^T Xc / nsim_effective``, ``Xc`` = mean-centred replicate
        coefficient matrix). Indexed and columned by ``coef_names``.
    coefficients : pd.DataFrame
        ``(nsim_effective, k)`` matrix of replicate coefficient vectors;
        columns are ``coef_names``.
    method : str
        Bootstrap method used. Currently always ``"cr"``.
    nsim : int
        Number of replicates requested by the caller.
    nsim_effective : int
        Replicates that actually converged (failures dropped).
    size : int
        Subsample size used per replicate.
    replace : bool
        Whether replicates were drawn with replacement.
    prob : array-like or None
        Sampling probability weights (``None`` means uniform).
    parallel : bool
        Whether parallel execution was requested. Always ``False``
        in the current Python port (a warning is emitted upstream).
    model : str
        Model spec / class that was bootstrapped (e.g. ``"ANN"``, ``"omg"``).
    time_elapsed : float
        Wall-clock seconds taken by the bootstrap.
    """

    vcov: pd.DataFrame
    coefficients: pd.DataFrame
    method: str
    nsim: int
    nsim_effective: int
    size: int
    replace: bool
    prob: Optional[NDArray]
    parallel: bool
    model: str
    time_elapsed: float
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"BootstrapResult(model={self.model!r}, method={self.method!r}, "
            f"nsim={self.nsim_effective}/{self.nsim}, "
            f"size={self.size}, k={self.coefficients.shape[1]}, "
            f"time_elapsed={self.time_elapsed:.2f}s)"
        )

    def __str__(self) -> str:
        return (
            f"Coefficient bootstrap for {self.model!r}\n"
            f"  method     : {self.method}\n"
            f"  replicates : {self.nsim_effective} of {self.nsim} converged\n"
            f"  size       : {self.size}\n"
            f"  parameters : {self.coefficients.shape[1]}\n"
            f"  elapsed    : {self.time_elapsed:.2f}s"
        )


def case_resample_indices(
    nobs: int,
    size: int,
    nsim: int,
    replace: bool,
    prob: Optional[NDArray],
    rng: np.random.Generator,
) -> NDArray:
    """Return an ``(nsim, size)`` integer index array drawn from ``[0, nobs)``.

    Mirrors R's internal ``sampler()`` helper used by ``coefbootstrap.adam``
    and ``coefbootstrap.om``: i.i.d. case resampling with or without
    replacement, optionally weighted by ``prob``.
    """
    if size > nobs and not replace:
        raise ValueError(
            f"size={size} exceeds nobs={nobs} with replace=False. "
            "Set replace=True or reduce size."
        )
    out = np.empty((nsim, size), dtype=np.int64)
    for i in range(nsim):
        out[i] = rng.choice(nobs, size=size, replace=replace, p=prob)
    return out


def moving_origin_indices(
    nobs: int,
    size: int,
    nsim: int,
    rng: np.random.Generator,
) -> NDArray:
    """Return an ``(nsim, size)`` array of contiguous time-ordered windows.

    Each row is ``arange(start, start + size)`` for a uniformly drawn
    ``start`` in ``[0, nobs - size]``. Retained for legacy callers; the
    new time-series default is :func:`time_series_sample_indices`.
    """
    if size > nobs:
        raise ValueError(f"size={size} exceeds nobs={nobs} for moving-origin sampling.")
    max_start = nobs - size
    if max_start == 0:
        starts = np.zeros(nsim, dtype=np.int64)
    else:
        starts = rng.integers(0, max_start + 1, size=nsim)
    out = starts[:, None] + np.arange(size, dtype=np.int64)[None, :]
    return out


def time_series_sample_indices(
    nobs: int,
    nsim: int,
    obs_minimum: int,
    change_origin: bool,
    rng: np.random.Generator,
) -> list[NDArray]:
    """Variable-length contiguous-window sampler matching R's ``sampler()``.

    Replicates R's case-resampling sampler for time-series models (the
    non-``regressionPure`` branch in ``coefbootstrap.adam`` /
    ``coefbootstrap.om`` / ``coefbootstrap.omg``): for each replicate

    1. Draw a length ``L = ceil(runif(obs_minimum, nobs))`` (uniform
       integer in ``[obs_minimum, nobs]``).
    2. If ``change_origin`` (i.e. ``initial_type`` is ``"backcasting"``
       or ``"complete"``), pick a starting offset uniformly in
       ``[0, nobs - L]``. Otherwise start at 0.
    3. The replicate's indices are ``arange(start, start + L)``.

    The size of each replicate therefore varies across the run. R's
    ``size`` argument is ignored on this path (matches R's behaviour).

    Returns ``nsim`` index arrays of varying length.
    """
    if obs_minimum >= nobs:
        raise ValueError(
            f"obs_minimum={obs_minimum} >= nobs={nobs}; not enough observations "
            "for case-resampling bootstrap (R would warn and fall back to dsr)."
        )
    out: list[NDArray] = []
    for _ in range(nsim):
        # R: ceil(runif(1, obs_minimum, nobs)). runif(a, b) ∈ [a, b]; ceil
        # gives integers in [obs_minimum+1, nobs]. Probability mass at the
        # boundary is a measure-zero set, so a uniform integer in
        # [obs_minimum+1, nobs] (inclusive) reproduces R's distribution.
        L = int(rng.integers(obs_minimum + 1, nobs + 1))
        if change_origin and L < nobs:
            # R: floor(runif(1, 0, nobs - L)) — uniform integer in
            # [0, nobs - L) (one less than nobs - L because R's runif open
            # at upper bound, and floor of (nobs - L) — exclusive).
            start = int(rng.integers(0, nobs - L + 1))
        else:
            start = 0
        out.append(np.arange(start, start + L, dtype=np.int64))
    return out


def empirical_vcov(coef_matrix: NDArray) -> NDArray:
    """Mean-centred cross-product empirical covariance.

    Matches R's formula on adam.R:5109 (``t(coefvcov) %*% coefvcov / nsim``
    where ``coefvcov`` is the matrix of replicate-minus-mean rows).
    """
    coef_matrix = np.asarray(coef_matrix, dtype=float)
    if coef_matrix.shape[0] < 2:
        n = coef_matrix.shape[1]
        return np.full((n, n), np.nan)
    centered = coef_matrix - coef_matrix.mean(axis=0)
    return (centered.T @ centered) / coef_matrix.shape[0]


def _resolve_n_jobs(parallel: Union[bool, int], nsim: int) -> int:
    """Translate a ``parallel`` argument into a concrete ``n_jobs`` count.

    Mirrors R's ``parallel`` handling in ``coefbootstrap.adam`` /
    ``coefbootstrap.om`` (R/adam.R:4864-4870): boolean ``True`` picks
    ``cpu_count - 1`` cores, an integer is used as-is. Clamped to
    ``[1, nsim]`` so we never spin up more workers than replicates.
    """
    if isinstance(parallel, bool):
        if not parallel:
            return 1
        n = max(1, (os.cpu_count() or 1) - 1)
    else:
        n = int(parallel)
        if n < 1:
            raise ValueError(
                f"parallel must be a positive int or bool, got {parallel!r}."
            )
    return min(n, max(1, nsim))


def run_replicates(
    fn: Callable[[int], Optional[NDArray]],
    nsim: int,
    parallel: Union[bool, int] = False,
    verbose: bool = False,
    label: str = "coefbootstrap",
) -> tuple[list[NDArray], bool]:
    """Run ``fn(i)`` for each replicate, optionally in parallel via joblib.

    Parameters
    ----------
    fn
        Worker callable. Must be picklable (top-level function — not a
        bound method or closure) when running in parallel. Returns a
        ``np.ndarray`` for a converged replicate or ``None`` to skip.
    nsim
        Number of replicates to run.
    parallel
        ``False`` runs serially. ``True`` requests parallel execution with
        ``cpu_count - 1`` workers; an integer specifies the exact worker
        count. If joblib is not installed, a one-line warning is emitted
        and the call falls back to a serial loop.
    verbose
        Print 10%-progress in serial mode; pass ``verbose=10`` through to
        ``joblib.Parallel`` in the parallel branch.
    label
        Identifier used in progress lines (``coefbootstrap`` /
        ``coefbootstrap[omg]``).

    Returns
    -------
    (replicate_coefs, parallel_used)
        ``replicate_coefs`` is the list of non-``None`` worker results,
        ordered by replicate index (joblib preserves submission order).
        ``parallel_used`` is ``True`` only when the parallel branch
        actually ran — callers persist this to ``BootstrapResult.parallel``
        so the recorded flag reflects reality.
    """
    use_parallel = bool(parallel) and (
        isinstance(parallel, int) and parallel != 0
        if not isinstance(parallel, bool)
        else parallel
    )

    if use_parallel:
        try:
            from joblib import Parallel, delayed
        except ImportError:
            warnings.warn(
                "parallel=True requested but joblib is not installed; "
                "running serially. Install with `pip install joblib` (or "
                '`pip install "smooth[parallel]"`) for parallel bootstrap.',
                UserWarning,
                stacklevel=3,
            )
            use_parallel = False

    replicate_coefs: list[NDArray] = []

    if use_parallel:
        n_jobs = _resolve_n_jobs(parallel, nsim)
        joblib_verbose = 10 if verbose else 0
        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=joblib_verbose)(
            delayed(fn)(i) for i in range(nsim)
        )
        replicate_coefs.extend(r for r in results if r is not None)
        return replicate_coefs, True

    # Serial path
    log_every = max(1, nsim // 10) if verbose else None
    for i in range(nsim):
        r = fn(i)
        if r is not None:
            replicate_coefs.append(r)
        if log_every is not None and (i + 1) % log_every == 0:
            print(
                f"  {label}: {i + 1}/{nsim} replicates "
                f"({len(replicate_coefs)} converged)"
            )
    return replicate_coefs, False


def bootstrap_confint_frame(
    boot: "BootstrapResult",
    coef_names: list[str],
    params: NDArray,
    level: float = 0.95,
    parm: Optional[Any] = None,
) -> pd.DataFrame:
    """Build a confint-style DataFrame from a :class:`BootstrapResult`.

    Mirrors the empirical-quantile branch of R's ``confint.adam`` /
    ``confint.omg`` when ``bootstrap=TRUE``: standard errors from the
    bootstrap covariance, and the two tails as empirical quantiles of the
    replicate matrix. No clamping is applied — the bootstrap replicates
    already live inside the feasible region by construction.

    Returns ``["S.E.", "<lo>%", "<hi>%"]`` with the same row index as
    ``coef_names``. Rows with no replicates (``coefficients`` empty) come
    back as NaN.
    """
    vcov_diag = np.abs(np.diag(np.asarray(boot.vcov.to_numpy(), dtype=float)))
    se = np.sqrt(vcov_diag)

    coef_matrix = boot.coefficients.to_numpy()
    if coef_matrix.shape[0] == 0:
        lo_vals = np.full(len(coef_names), np.nan)
        hi_vals = np.full(len(coef_names), np.nan)
    else:
        lo_vals = np.quantile(coef_matrix, (1 - level) / 2, axis=0)
        hi_vals = np.quantile(coef_matrix, (1 + level) / 2, axis=0)

    lo_name = f"{(1 - level) / 2 * 100:g}%"
    hi_name = f"{(1 + level) / 2 * 100:g}%"
    out = pd.DataFrame(
        np.column_stack([se, lo_vals, hi_vals]),
        index=coef_names,
        columns=["S.E.", lo_name, hi_name],
    )
    if parm is not None:
        out = out.loc[parm if isinstance(parm, (list, tuple)) else [parm]]
    return out


def _build_result(
    replicate_coefs: list[NDArray],
    coef_names: list[str],
    *,
    method: str,
    nsim: int,
    size: int,
    replace: bool,
    prob: Optional[NDArray],
    parallel: bool,
    model: str,
    time_elapsed: float,
    extra_metadata: Optional[dict[str, Any]] = None,
) -> BootstrapResult:
    """Assemble a :class:`BootstrapResult` from the converged replicate coefs."""
    if replicate_coefs:
        coef_matrix = np.vstack([np.asarray(c, dtype=float) for c in replicate_coefs])
    else:
        coef_matrix = np.zeros((0, len(coef_names)))

    vcov_arr = empirical_vcov(coef_matrix)
    coef_df = pd.DataFrame(coef_matrix, columns=coef_names)
    vcov_df = pd.DataFrame(vcov_arr, index=coef_names, columns=coef_names)

    return BootstrapResult(
        vcov=vcov_df,
        coefficients=coef_df,
        method=method,
        nsim=nsim,
        nsim_effective=coef_matrix.shape[0],
        size=size,
        replace=replace,
        prob=prob,
        parallel=parallel,
        model=model,
        time_elapsed=time_elapsed,
        metadata=extra_metadata or {},
    )
