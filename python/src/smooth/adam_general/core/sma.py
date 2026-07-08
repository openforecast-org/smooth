"""
Simple Moving Average (SMA) wrapper for ADAM.

Implements SMA(m) as a single-source-of-error AR(m) state-space model with
every autoregressive coefficient fixed at 1/m. Wrapping the moving average
inside ADAM gives recursive multi-step forecasts, proper forecast variance,
and information-criterion-based automatic order selection.
"""

from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray
from scipy import stats as scipy_stats

from smooth.adam_general.core.adam import ADAM


def _ic_value(loglik: float, df: int, n: int, ic: str) -> float:
    """Compute an information criterion with given df and sample size."""
    aic = -2.0 * loglik + 2.0 * df
    if ic == "AIC":
        return aic
    if ic == "AICc":
        denom = max(n - df - 1, 1)
        return aic + 2.0 * df * (df + 1) / denom
    if ic == "BIC":
        return -2.0 * loglik + np.log(n) * df
    if ic == "BICc":
        denom = max(n - df - 1, 1)
        return -2.0 * loglik + np.log(n) * df * (1.0 + (df + 1) / denom)
    raise ValueError(f"Unknown IC: {ic!r}")


class SMA(ADAM):
    """
    Simple Moving Average in Single Source of Error state space form.

    SMA(m) is an AR(m) state-space model where every AR coefficient is fixed
    at 1/m. It is implemented as a thin wrapper over :class:`ADAM` with
    ``model="NNN"`` and the AR vector hard-coded, so it inherits the full
    ADAM fit / predict / diagnostics surface (multi-step forecasts,
    prediction intervals, residual diagnostics). If ``order`` is left
    unspecified, the order is selected automatically by information
    criterion.

    Parameters
    ----------
    order : Optional[int], default=None
        Order of the moving average. If None, selected automatically using
        the information criterion (ternary search when ``fast=True``,
        sequential scan when ``fast=False``).

    ic : Literal["AIC", "AICc", "BIC", "BICc"], default="AICc"
        Information criterion used for automatic order selection.

    h : int, default=10
        Forecast horizon (used with ``holdout=True`` to reserve a test set).

    holdout : bool, default=False
        Whether to hold out the last ``h`` observations for validation.

    fast : bool, default=True
        If True, use ternary search for order selection (fast, finds a local
        minimum). If False, evaluate all orders 1 … min(200, T) sequentially.
        When a pandas Series with a DatetimeIndex is passed, the inferred
        seasonal period is always evaluated as a candidate regardless of this flag.

    verbose : int, default=0
        Verbosity level. 0 = silent.

    **kwargs
        Additional arguments passed to ADAM (e.g. ``n_iterations``).

    Attributes
    ----------
    model : str
        Model name, e.g. ``"SMA(3)"``.
    ICs_ : dict
        IC values for each evaluated order (only present after auto-selection).
        Keys are order integers, values are IC floats.

    See Also
    --------
    ADAM : Parent class; all ADAM attributes are available after ``fit()``.
    MSARIMA : General multiple-seasonal ARIMA wrapper.

    Examples
    --------
    Fixed order::

        >>> import numpy as np
        >>> from smooth import SMA
        >>> y = np.cumsum(np.random.randn(60)) + 100
        >>> model = SMA(order=4, h=5)
        >>> model.fit(y)
        >>> fc = model.predict(h=5)
        >>> fc.mean

    Auto-selected order::

        >>> model = SMA(h=5)
        >>> model.fit(y)
        >>> print(model.model)   # e.g. "SMA(3)"
        >>> print(model.ICs_)

    References
    ----------
    - Svetunkov, I., & Petropoulos, F. (2017). Old dog, new tricks: a
      modelling view of simple moving averages. International Journal of
      Production Research. https://doi.org/10.1080/00207543.2017.1380326
    """

    _BLOCKED_KWARGS = frozenset(
        {
            "model",
            "ar_order",
            "i_order",
            "ma_order",
            "orders",
            "arima_select",
            "arma",
            "initial",
            "loss",
            "bounds",
            "distribution",
            "lags",
        }
    )

    def __init__(
        self,
        order: Optional[int] = None,
        ic: Literal["AIC", "AICc", "BIC", "BICc"] = "AICc",
        h: int = 10,
        holdout: bool = False,
        fast: bool = True,
        verbose: int = 0,
        **kwargs,
    ) -> None:
        bad = self._BLOCKED_KWARGS & set(kwargs)
        if bad:
            raise ValueError(
                f"SMA() does not support these parameters: {sorted(bad)}. "
                "Use ADAM() for full model control."
            )

        self._sma_order = order
        self._sma_fast = fast
        self._ICs_array: Optional[NDArray] = None

        super().__init__(
            model="NNN",
            ar_order=1,
            i_order=0,
            ma_order=0,
            lags=[1],
            arma={"ar": [1.0]},
            initial="backcasting",
            loss="MSE",
            bounds="none",
            distribution="dnorm",
            ic=ic,
            h=h,
            holdout=holdout,
            verbose=verbose,
            **kwargs,
        )

    def fit(self, y: NDArray, X: Optional[NDArray] = None) -> "SMA":
        """Fit the SMA model to time series data."""
        # Determine order and obs_in_sample before calling super().fit()
        n = len(y)
        h_eff = (self.h or 0) if self.holdout else 0
        obs_in_sample = n - h_eff
        y_is = np.asarray(y[:obs_in_sample], dtype=float)
        ic = self.ic

        if self._sma_order is None:
            order, ICs = self._select_order(y_is, ic)
            self._ICs_array = ICs
        else:
            order = self._sma_order
            self._ICs_array = None

        if not isinstance(order, (int, np.integer)) or order < 1:
            raise ValueError(f"order must be a positive integer, got {order!r}.")
        order = int(order)
        if obs_in_sample < order:
            raise ValueError(
                f"Not enough observations ({obs_in_sample}) for order {order}."
            )

        self.ar_order = order
        self.i_order = 0
        self.ma_order = 0
        self.lags = [1]
        self.arma = {"ar": [1.0 / order] * order}
        self.initial = "backcasting"
        self.loss = "MSE"
        self.bounds = "none"
        self.distribution = "dnorm"

        super().fit(y, X)

        self.model = f"SMA({order})"
        if self._ICs_array is not None:
            self.ICs_ = {
                i + 1: v for i, v in enumerate(self._ICs_array) if not np.isnan(v)
            }

        return self

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ic_for_order(y_is: NDArray, order: int, ic: str) -> float:
        """Fit SMA(order) on y_is and return the IC (df=1: scale is the sole param)."""
        m = ADAM(
            model="NNN",
            ar_order=order,
            i_order=0,
            ma_order=0,
            lags=[1],
            arma={"ar": [1.0 / order] * order},
            initial="backcasting",
            loss="MSE",
            bounds="none",
            distribution="dnorm",
            holdout=False,
            verbose=0,
        )
        m.fit(y_is)
        n = len(y_is)
        fitted = np.asarray(m.fitted, dtype=float)
        errors = y_is - fitted
        scale = float(np.sqrt(np.sum(errors**2) / n))
        if scale <= 0:
            scale = float(np.finfo(float).eps)
        loglik = float(np.sum(scipy_stats.norm.logpdf(y_is, fitted, scale)))
        return _ic_value(loglik, df=1, n=n, ic=ic)

    def _select_order(self, y_is: NDArray, ic: str) -> tuple[int, NDArray]:
        """Return (best_order, ICs_array) via ternary or sequential search."""
        max_order = min(200, len(y_is))
        ICs = np.full(max_order, np.nan)

        def ev(k: int) -> float:
            if np.isnan(ICs[k - 1]):
                ICs[k - 1] = SMA._ic_for_order(y_is, k, ic)
            return ICs[k - 1]

        if self._sma_fast:
            i, k = 1, max_order
            ev(i)
            ev(k)
            j = (k + i) // 2
            for _ in range(max_order):
                ev(j)
                triple = [
                    (ICs[i - 1], i),
                    (ICs[j - 1], j),
                    (ICs[k - 1], k),
                ]
                sorted_t = sorted(triple, key=lambda x: x[0])
                i_new = sorted_t[0][1]
                k_new = sorted_t[1][1]
                if (i == i_new and k == k_new) or (k == i_new and i == k_new):
                    k_new = j
                i, k = min(i_new, k_new), max(i_new, k_new)
                j = (k + i) // 2
                if j == i or j == k or j == 0:
                    break

            freq = _infer_frequency(y_is)
            if freq is not None and 1 <= freq <= max_order:
                ev(freq)
        else:
            for k in range(1, max_order + 1):
                ev(k)

        order = int(np.nanargmin(ICs) + 1)
        return order, ICs


def _infer_frequency(y: NDArray) -> Optional[int]:
    """Try to read the seasonal period from a pandas Series DatetimeIndex."""
    try:
        import pandas as pd

        if isinstance(y, pd.Series) and isinstance(y.index, pd.DatetimeIndex):
            freq = y.index.freq
            if freq is not None:
                freq_map = {
                    "M": 12,
                    "MS": 12,
                    "ME": 12,
                    "Q": 4,
                    "QS": 4,
                    "QE": 4,
                    "W": 52,
                    "D": 7,
                    "h": 24,
                    "H": 24,
                }
                for k, v in freq_map.items():
                    if str(freq).startswith(k):
                        return v
    except ImportError:
        pass
    return None
