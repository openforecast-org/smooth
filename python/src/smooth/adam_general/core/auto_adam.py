"""
AutoADAM — automatic model selection wrapper for ADAM.

Provides automatic selection of:
- ARIMA orders (three-phase D→MA→AR search)
- Error distribution (tries each candidate; picks lowest IC)
- ETS model (delegated to ADAM's existing selection machinery)

Outlier detection is accepted as a parameter but not yet implemented;
a warning is issued when it is requested.
"""

import time
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray

from smooth.adam_general.core.adam import ADAM, LOSS_OPTIONS
from smooth.adam_general.core.estimator.arima_selector import arima_selector

_POSITIVE_ONLY_DISTS = {"dlnorm", "dinvgauss", "dgamma"}

_ALL_DISTRIBUTIONS = [
    "dnorm",
    "dlaplace",
    "ds",
    "dgnorm",
    "dlnorm",
    "dinvgauss",
    "dgamma",
]


class AutoADAM(ADAM):
    """
    Automatic ADAM model selection.

    Wraps :class:`ADAM` with automatic selection of ARIMA orders and error
    distribution.

    ETS model selection (ZZZ, ZXZ, FFF, CCC …) is handled by the underlying
    ADAM machinery and is not duplicated here.

    Parameters
    ----------
    model : Union[str, List[str]], default="ZXZ"
        ETS model specification passed to each internal ADAM fit.
        Supports all codes accepted by :class:`ADAM`.

    lags : Optional[List[int]], default=None
        Seasonal period(s). Lag 1 is prepended automatically when absent.

    ar_order : Union[int, List[int]], default=[3, 3]
        Maximum AR order(s) per lag level for ARIMA selection.
        Defaults to ``[3, 3]`` matching R's ``auto.adam()``.

    i_order : Union[int, List[int]], default=[2, 1]
        Maximum integration order(s) per lag level for ARIMA selection.
        Defaults to ``[2, 1]`` matching R's ``auto.adam()``.

    ma_order : Union[int, List[int]], default=[3, 3]
        Maximum MA order(s) per lag level for ARIMA selection.

    orders : Optional[Dict[str, Any]], default=None
        Dict-style alternative to the scalar max-order arguments above. A
        dict with keys ``"ar"``, ``"i"``, ``"ma"`` (each an int or list)
        and optionally ``"select"`` (bool). When provided,
        ``ar_order`` / ``i_order`` / ``ma_order`` are ignored.

    arima_select : bool, default=True
        Whether to perform ARIMA order selection. Unlike :class:`ADAM`,
        this defaults to ``True``.

    distribution : Union[str, List[str]], default=(all 7 distributions)
        Distribution(s) to try. When a list is supplied every entry is
        fitted and the one with lowest IC is kept.  A single string uses
        only that distribution (no selection loop).

    outliers : Literal["ignore", "use", "select"], default="ignore"
        Outlier handling mode.  ``"use"`` and ``"select"`` are accepted but
        not yet implemented; a :class:`UserWarning` is issued.

    level : float, default=0.99
        Confidence level for outlier detection (placeholder for future use).

    ic : Literal["AIC", "AICc", "BIC", "BICc"], default="AICc"
        Information criterion used for all model comparisons.

    loss : LOSS_OPTIONS, default="likelihood"
        Loss function for parameter estimation.

    constant : Union[bool, float], default=False
        Constant/drift term. Overridden by ARIMA selection when
        ``arima_select=True`` (the selection algorithm tests constant on/off).

    holdout : bool, default=False
        Whether to use a holdout sample.

    h : Optional[int], default=None
        Forecast horizon.

    bounds : Literal["usual", "admissible", "none"], default="usual"
        Parameter bounds type.

    initial : str or dict, default="backcasting"
        Initialisation method or fixed initial values.

    regressors : Literal["use", "select", "adapt"], default="use"
        How to handle external regressors.

    verbose : int, default=0
        Verbosity level for the *final* model fit. All intermediate
        selection fits are always silent.

    **kwargs
        Additional arguments forwarded to every internal :class:`ADAM` fit.

    See Also
    --------
    ADAM : Full ADAM model.
    ES : ETS-only wrapper.
    MSARIMA : Pure ARIMA wrapper.

    Examples
    --------
    Full automatic selection (ETS + ARIMA + distribution)::

        >>> from smooth import AutoADAM
        >>> import numpy as np
        >>> y = np.cumsum(np.random.randn(120)) + 100
        >>> model = AutoADAM(lags=[12])
        >>> model.fit(y)
        >>> print(model.model, model.distribution_)

    ARIMA-only automatic selection (no ETS)::

        >>> model = AutoADAM(model="NNN", lags=[1, 12])
        >>> model.fit(y)

    Fix the distribution, only select ARIMA orders::

        >>> model = AutoADAM(distribution="dnorm", lags=[1, 12])
        >>> model.fit(y)

    References
    ----------
    - Svetunkov, I. (2023). Forecasting and Analytics with the Augmented
      Dynamic Adaptive Model. https://openforecast.org/adam/
    """

    def __init__(
        self,
        model: Union[str, List[str]] = "ZXZ",
        lags: Optional[Union[int, List[int]]] = None,
        ar_order: Union[int, List[int], None] = None,
        i_order: Union[int, List[int], None] = None,
        ma_order: Union[int, List[int], None] = None,
        orders: Optional[Dict[str, Any]] = None,
        arima_select: bool = False,
        distribution: Union[str, List[str], None] = None,
        outliers: Literal["ignore", "use", "select"] = "ignore",
        level: float = 0.99,
        constant: Union[bool, float] = False,
        ic: Literal["AIC", "AICc", "BIC", "BICc"] = "AICc",
        loss: LOSS_OPTIONS = "likelihood",
        h: Optional[int] = None,
        holdout: bool = False,
        bounds: Literal["usual", "admissible", "none"] = "usual",
        initial: Union[str, Dict[str, Any], None] = "backcasting",
        regressors: Literal["use", "select", "adapt"] = "use",
        verbose: int = 0,
        ets: Literal["conventional", "adam"] = "conventional",
        **kwargs,
    ) -> None:
        """Initialise AutoADAM.

        Notes on ``lags`` and ARIMA-order parameters
        --------------------------------------------
        ``lags`` accepts either a scalar (``lags=12``) or a list
        (``lags=[12]``); both are equivalent.

        The ARIMA-order specification follows a precedence rule shared with
        :class:`ADAM`:

        - If ``orders`` (dict) is supplied, it is used and the three scalar
          arguments ``ar_order`` / ``i_order`` / ``ma_order`` are **ignored**
          (a warning is emitted). Order selection is on iff
          ``orders.get("select", arima_select)`` is true.
        - Else if any of the three scalar order arguments has a non-zero value,
          they are used as **fixed** orders (no selection).
        - Else (the default), no ARIMA component is fitted and no order
          selection is performed.
        """
        # Normalise scalar lags to a list so downstream code is uniform.
        if isinstance(lags, (int, np.integer)):
            lags = [int(lags)]

        # Store AutoADAM-specific params before delegating to ADAM
        self._auto_distribution_spec: List[str] = (
            list(distribution)
            if isinstance(distribution, list)
            else (
                [distribution]
                if isinstance(distribution, str)
                else list(_ALL_DISTRIBUTIONS)
            )
        )
        self._auto_outliers: str = outliers
        self._auto_level: float = level
        self._auto_verbose: int = verbose

        # Resolve the ARIMA order specification using the shared helper so
        # ADAM and AutoADAM share the same precedence rule.
        from smooth.adam_general.core.checker.arima_checks import resolve_arima_orders

        resolved, select_flag = resolve_arima_orders(
            orders, ar_order, i_order, ma_order, arima_select=arima_select
        )

        # Decide search ranges for the selection loop. When the resolved dict
        # carries scalar values they get broadcast in ``arima_selector``.
        # When orders=None and triplet=empty, the resolver returned None →
        # AutoADAM is effectively pure ETS.
        self._auto_arima_select: bool = bool(select_flag)
        if resolved is not None:
            ar_val = resolved.get("ar", 0)
            i_val = resolved.get("i", 0)
            ma_val = resolved.get("ma", 0)
            self._auto_max_ar: List[int] = (
                list(ar_val) if isinstance(ar_val, (list, tuple)) else [int(ar_val)]
            )
            self._auto_max_i: List[int] = (
                list(i_val) if isinstance(i_val, (list, tuple)) else [int(i_val)]
            )
            self._auto_max_ma: List[int] = (
                list(ma_val) if isinstance(ma_val, (list, tuple)) else [int(ma_val)]
            )
        else:
            # No ARIMA at all — set placeholder zeros so existing code paths
            # that read these attributes don't break, but ``arima_select=False``
            # ensures the selection loop is skipped.
            self._auto_max_ar = [0]
            self._auto_max_i = [0]
            self._auto_max_ma = [0]

        # Pass placeholder values to ADAM.__init__ — fit() will override them
        super().__init__(
            model=model,
            lags=lags,
            ar_order=0,
            i_order=0,
            ma_order=0,
            orders=None,
            arima_select=False,
            constant=constant,
            distribution=None,
            ic=ic,
            loss=loss,
            h=h,
            holdout=holdout,
            bounds=bounds,
            initial=initial,
            regressors=regressors,
            verbose=verbose,
            ets=ets,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, y, X=None):
        """
        Fit the AutoADAM model.

        Runs distribution loop and (optionally) ARIMA order selection, then
        copies the best fitted model's state into this instance.

        Parameters
        ----------
        y : array-like
            Time series data.
        X : array-like, optional
            External regressor matrix.

        Returns
        -------
        self
        """
        auto_start = time.time()

        # Save AutoADAM-specific attributes so they survive the state copy
        _auto_keys = [k for k in self.__dict__ if k.startswith("_auto_")]
        _auto_state = {k: self.__dict__[k] for k in _auto_keys}

        # (outlier handling is applied after best-model selection below)

        y_arr = np.asarray(y, dtype=float).ravel()

        # Candidate distributions filtered for data properties
        candidates = self._filter_distributions(y_arr)

        # Adam kwargs forwarded to internal fits (everything except selection params)
        adam_kw = dict(
            loss=self.loss,
            h=self.h,
            holdout=self.holdout,
            bounds=self.bounds,
            initial=self.initial,
            regressors=self.regressors,
            verbose=0,  # always silent during selection
        )
        # Forward any extra kwargs stored in self (e.g. occurrence, arma)
        for attr in (
            "occurrence",
            "arma",
            "persistence",
            "phi",
            "n_iterations",
            "fast",
            "smoother",
            "ets",
        ):
            val = getattr(self, attr, None)
            if val is not None:
                adam_kw[attr] = val

        lags = self.lags  # may be None (handled by ADAM internally)

        # ------------------------------------------------------------------
        # Main selection loop over distributions
        # ------------------------------------------------------------------
        results: Dict[str, Any] = {}

        ets_model = self.model if isinstance(self.model, str) else self.model[0]
        has_ets = ets_model != "NNN"

        verbose = bool(self._auto_verbose)
        if verbose:
            # Mirror R's autoadam.R style: one line, comma-separated as we go.
            print(
                "Evaluating models with different distributions... ",
                end="",
                flush=True,
            )

        for dist in candidates:
            if verbose:
                print(f"{dist}, ", end="", flush=True)
            if self._auto_arima_select:
                # ETS-first strategy (mirrors autoadam.R lines 392-423, 560-595):
                # When the model has ETS components, fit ETS-only first, then
                # use it as the baseline for ARIMA selection on its residuals.
                ets_baseline = None
                if has_ets:
                    try:
                        ets_baseline = ADAM(
                            model=ets_model,
                            lags=lags,
                            ar_order=0,
                            i_order=0,
                            ma_order=0,
                            distribution=dist,
                            **adam_kw,
                        ).fit(y_arr, X)
                    except Exception:
                        pass

                sel = arima_selector(
                    y=y_arr,
                    ets_model=ets_model,
                    max_ar_orders=self._auto_max_ar,
                    max_i_orders=self._auto_max_i,
                    max_ma_orders=self._auto_max_ma,
                    lags=lags,
                    distribution=dist,
                    ic=self.ic,
                    X=X,
                    ets_baseline=ets_baseline,
                    **adam_kw,
                )
                fitted_model = sel["model"]
                ic_val = sel["ic_value"]

                if fitted_model is None:
                    continue

                # Store ARIMA orders for reference
                results[dist] = {
                    "model": fitted_model,
                    "ic": ic_val,
                    "ar_orders": sel["ar_orders"],
                    "i_orders": sel["i_orders"],
                    "ma_orders": sel["ma_orders"],
                    "constant": sel["constant"],
                }
            else:
                # Fixed ARIMA orders: just fit once per distribution
                try:
                    m = ADAM(
                        model=self.model,
                        lags=lags,
                        ar_order=self._auto_max_ar,
                        i_order=self._auto_max_i,
                        ma_order=self._auto_max_ma,
                        constant=self.constant,
                        distribution=dist,
                        **adam_kw,
                    ).fit(y, X)
                    ic_val = _get_ic(m, self.ic)
                    results[dist] = {"model": m, "ic": ic_val}
                except Exception:
                    continue

        if not results:
            if verbose:
                print("(no distribution succeeded)")
            raise RuntimeError(
                "AutoADAM: all candidate distributions failed. "
                "Check your data and model specification."
            )

        if verbose:
            print("Done!")

        # ------------------------------------------------------------------
        # Select best distribution
        # ------------------------------------------------------------------
        best_dist = min(results, key=lambda d: results[d]["ic"])
        best_entry = results[best_dist]
        best_model = best_entry["model"]

        if verbose:
            print(f"Selected distribution: {best_dist}")
            if self._auto_arima_select and "ar_orders" in best_entry:
                print(
                    f"Selected ARIMA orders: AR={best_entry['ar_orders']}, "
                    f"I={best_entry['i_orders']}, MA={best_entry['ma_orders']}"
                )

        # ------------------------------------------------------------------
        # Outlier handling: detect on best model, refit with dummies appended
        # ------------------------------------------------------------------
        if self._auto_outliers in ("use", "select"):
            od = best_model.outlierdummy(level=self._auto_level)
            if len(od.id) > 0:
                D = (
                    ADAM._expand_outlier_dummies(od.outliers)
                    if self._auto_outliers == "select"
                    else od.outliers
                )
                h_eff = len(y_arr) - best_model.nobs
                if h_eff > 0:
                    D = np.vstack([D, np.zeros((h_eff, D.shape[1]))])
                X_new = np.hstack([X, D]) if X is not None else D
                # best_model._config was built on the first fit; restore attrs
                # so that fit() can run again (fit() expects instance attrs).
                # Skip read-only properties (e.g. `orders`) with try/except.
                for k, v in best_model._config.items():
                    try:
                        setattr(best_model, k, v)
                    except AttributeError:
                        pass
                best_model.regressors = (
                    "select" if self._auto_outliers == "select" else "use"
                )
                best_model.outliers = "ignore"
                best_model.fit(y_arr, X_new)

        # ------------------------------------------------------------------
        # Copy fitted state from best_model into self
        # ------------------------------------------------------------------
        self.__dict__.update(best_model.__dict__)

        # Restore AutoADAM-specific state
        # (overwrite any same-named keys from best_model)
        self.__dict__.update(_auto_state)

        # Patch _config to reflect original outliers setting
        if self._auto_outliers in ("use", "select") and hasattr(self, "_config"):
            self._config["outliers"] = self._auto_outliers

        # Store selection metadata
        self._selected_distribution: str = best_dist
        self._all_ic_values: Dict[str, float] = {d: results[d]["ic"] for d in results}
        if self._auto_arima_select:
            self._selected_arima_orders: Dict[str, Any] = {
                k: best_entry.get(k)
                for k in ("ar_orders", "i_orders", "ma_orders", "constant")
            }

        # Update total elapsed time to include selection
        self.time_elapsed_ = time.time() - auto_start

        return self

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _filter_distributions(self, y: NDArray) -> List[str]:
        """Return candidate distributions appropriate for the data."""
        candidates = list(self._auto_distribution_spec)

        if np.any(y <= 0):
            candidates = [d for d in candidates if d not in _POSITIVE_ONLY_DISTS]

        # Remove positive-only distributions for pure ARIMA (NNN)
        ets_model = self.model if isinstance(self.model, str) else self.model[0]
        if ets_model == "NNN":
            candidates = [d for d in candidates if d not in _POSITIVE_ONLY_DISTS]

        return candidates if candidates else ["dnorm"]

    def __repr__(self) -> str:
        """Return string representation of fitted AutoADAM model."""
        try:
            self._check_is_fitted()
            return (
                f"AutoADAM: {self.model}\n"
                f"Distribution: {self._selected_distribution}\n"
                f"IC ({self.ic}): "
                f"{self._all_ic_values.get(self._selected_distribution, 'N/A'):.4f}"
            )
        except Exception:
            return "AutoADAM (not fitted)"


def _get_ic(model, ic_name: str) -> float:
    """Extract IC value from fitted ADAM model."""
    return float(getattr(model, ic_name.lower()))
