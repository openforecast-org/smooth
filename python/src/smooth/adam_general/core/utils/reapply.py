"""Container for the result of ``ADAM.reapply``.

Mirrors R's ``"reapply"`` S3 class produced by ``reapply.adam``
(R/reapply.R:772-777). The field set and array dimensions match R
exactly so downstream code (notably ``ADAM.reforecast``) can be a
straight port.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class ReapplyResult:
    """Container for an ``ADAM.reapply`` run.

    Attributes
    ----------
    time_elapsed : float
        Wall-clock seconds taken by the reapply call.
    y : pandas.Series
        The in-sample actuals (R: ``$y``). Index matches ``ADAM.fitted``.
    states : numpy.ndarray
        State cube of shape ``(n_components, obs_in_sample + lags_model_max,
        nsim)`` — components × time × replicate, mirroring R's
        ``$states`` array.
    refitted : pandas.DataFrame
        ``(obs_in_sample, nsim)`` matrix of fitted paths (R: ``$refitted``).
        Index matches ``y.index``; columns are ``nsim1 … nsimN``.
    fitted : pandas.Series
        Conditional-mean fitted values from the underlying fit
        (R: ``$fitted``) — identical to ``ADAM.fitted`` and provided for
        plotting against ``refitted``.
    model : str
        Model spec string (e.g. ``"ETS(MAM)"``).
    transition : numpy.ndarray
        Transition-matrix cube ``(n_components, n_components, nsim)``.
    measurement : numpy.ndarray
        Measurement-matrix cube ``(obs_in_sample, n_components, nsim)``.
    persistence : pandas.DataFrame
        Persistence matrix ``(n_components, nsim)``. Row index gives the
        component names where available.
    profile : numpy.ndarray
        Final profile cube ``(n_components, lags_model_max, nsim)`` —
        the profile after each replicate's in-sample reapply run.
    random_parameters : pandas.DataFrame
        ``(nsim, n_parameters)`` sampled-from-MVN parameter matrix.
        Columns are :attr:`ADAM.coef_names`.
    nsim : int
        Number of replicates.
    """

    time_elapsed: float
    y: pd.Series
    states: np.ndarray
    refitted: pd.DataFrame
    fitted: pd.Series
    model: str
    transition: np.ndarray
    measurement: np.ndarray
    persistence: pd.DataFrame
    profile: np.ndarray
    random_parameters: pd.DataFrame
    nsim: int

    def __repr__(self) -> str:
        return (
            f"ReapplyResult(model={self.model!r}, nsim={self.nsim}, "
            f"refitted={self.refitted.shape}, "
            f"time_elapsed={self.time_elapsed:.2f}s)"
        )

    def __str__(self) -> str:
        return (
            f"Reapply for {self.model!r}\n"
            f"  replicates : {self.nsim}\n"
            f"  refitted   : {self.refitted.shape[0]} x {self.refitted.shape[1]}\n"
            f"  parameters : {self.random_parameters.shape[1]}\n"
            f"  elapsed    : {self.time_elapsed:.2f}s"
        )

    def plot(
        self,
        *,
        ax=None,
        figsize=None,
        title=None,
        ylim=None,
        ylabel="",
        legend=False,
    ):
        """Fan-chart of the refitted matrix (R: ``plot.reapply``).

        Draws five nested quantile bands of ``self.refitted`` —
        ``95%`` / ``80%`` / ``60%`` / ``40%`` / ``20%`` — shaded from
        lightest (outer) to darkest (inner), then overlays the
        actuals as a thin black line and the original point-estimate
        fitted values as a dashed purple line. Mirrors R's
        ``plot.reapply`` (R/reapply.R:816-857).

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axis to draw on. When ``None`` a fresh figure + axis are
            created.
        figsize : tuple, optional
            Forwarded to :func:`matplotlib.pyplot.subplots` when
            ``ax`` is not provided.
        title : str, optional
            Plot title. Defaults to ``"Refitted values of {model}"``.
        ylim : tuple, optional
            ``(low, high)`` y-axis limits. Defaults to the range over
            actuals + fitted.
        ylabel : str, optional
            Y-axis label. Defaults to empty (matches R).
        legend : bool, optional
            Add a legend distinguishing the actuals / fitted lines.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the plot.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        # Mirror R's nested-band design (R/reapply.R:850-854): five
        # symmetric (lower, upper) quantile pairs drawn outer-first.
        band_pairs = [
            (0.025, 0.975),  # 95%
            (0.10, 0.90),  # 80%
            (0.20, 0.80),  # 60%
            (0.30, 0.70),  # 40%
            (0.40, 0.60),  # 20%
        ]
        # R uses ``colorRampPalette(c("grey95", "darkgrey"))(5)`` —
        # lightest on the outside (95% band), darkest in the middle.
        cmap = LinearSegmentedColormap.from_list(
            "carma_grey",
            ["#F2F2F2", "#A9A9A9"],
        )
        band_colors = [cmap(i / (len(band_pairs) - 1)) for i in range(len(band_pairs))]
        edge_color = "#A9A9A9"

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or (10, 5))
        else:
            fig = ax.figure

        refitted = self.refitted.to_numpy()
        actuals = self.y.to_numpy()
        fitted = self.fitted.to_numpy()
        x_idx = self.y.index

        for color, (lo, hi) in zip(band_colors, band_pairs):
            q_lo = np.quantile(refitted, lo, axis=1)
            q_hi = np.quantile(refitted, hi, axis=1)
            ax.fill_between(
                x_idx,
                q_lo,
                q_hi,
                color=color,
                edgecolor=edge_color,
                linewidth=0.3,
            )

        ax.plot(x_idx, actuals, color="black", lw=1, label="Actuals")
        ax.plot(x_idx, fitted, color="purple", lw=2, ls="--", label="Fitted")

        ax.set_title(title or f"Refitted values of {self.model}")
        ax.set_ylabel(ylabel)
        if ylim is None:
            stack = np.concatenate([actuals, fitted])
            stack = stack[np.isfinite(stack)]
            if stack.size > 0:
                ax.set_ylim(float(stack.min()), float(stack.max()))
        else:
            ax.set_ylim(ylim)
        if legend:
            ax.legend(loc="best")

        return fig


def sampling_vcov(vcov) -> NDArray:
    """The covariance of the parameters for the draws of ``reapply``.

    Mirrors R's ``reapply_vcov``: the OPG covariance (the default) returns an
    infinite variance for a parameter the data does not identify (e.g. an
    initial that washes out when its smoothing parameter is at a bound). Such
    parameters cannot be resampled, so they are held at their point estimate
    (their row and column zeroed), and the matrix is repaired to be positive
    semi-definite.
    """
    # A copy: np.asarray can return a read-only view into a DataFrame's block
    vcov_arr = np.array(vcov, dtype=float)
    non_finite = ~np.isfinite(np.diag(vcov_arr))
    if np.any(non_finite):
        vcov_arr[non_finite, :] = 0.0
        vcov_arr[:, non_finite] = 0.0
    return _psd_correct(vcov_arr)


def _psd_correct(vcov: NDArray) -> NDArray:
    """Ensure ``vcov`` is positive semi-definite for the MVN sampler.

    Mirrors R's ``reapply.adam`` lines 96-115: if the smallest eigenvalue
    is negative, shift the diagonal by ``|min_eig| + 1e-10`` when the
    shift is small (PSD repair). When the eigenvalue is below ``-1`` the
    repair is too aggressive — fall back to the diagonal-only matrix
    which is always PSD.
    """
    vcov = np.asarray(vcov, dtype=float)
    if vcov.size == 0:
        return vcov
    try:
        eig_min = float(np.min(np.linalg.eigvalsh(vcov)))
    except np.linalg.LinAlgError:
        # ``Eigenvalues did not converge`` can fire on platform-specific
        # LAPACK iteration noise (seen on Windows wheels under
        # ``bounds="admissible"`` reapply) even when the input is a
        # well-formed covariance matrix. The diagonal-only matrix is
        # PSD by construction, so use it as the safe fallback for the
        # MVN sampler — same response as the very-negative-eigenvalue
        # branch below.
        warnings.warn(
            "Eigendecomposition of the covariance matrix did not converge; "
            "falling back to the diagonal-only matrix for MVN sampling.",
            stacklevel=3,
        )
        return np.diag(np.diag(vcov))
    if eig_min < 0:
        if eig_min > -1:
            warnings.warn(
                "The covariance matrix of parameters is not positive "
                "semi-definite; shifting the diagonal to repair it. "
                "Consider re-estimating the model with a different "
                "optimiser configuration.",
                stacklevel=3,
            )
            return vcov + (-eig_min + 1e-10) * np.eye(vcov.shape[0])
        warnings.warn(
            "The covariance matrix of parameters has a large negative "
            "eigenvalue; falling back to the diagonal-only matrix for "
            "MVN sampling. It is worth re-estimating the model.",
            stacklevel=3,
        )
        return np.diag(np.diag(vcov))
    return vcov
