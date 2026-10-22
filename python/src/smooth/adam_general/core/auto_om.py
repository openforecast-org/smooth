"""Automatic Occurrence Model Selection (AutoOM).

Fits an :class:`OM` (or :class:`OMG`) for each candidate occurrence type and
returns the model with the lowest information criterion.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray

from smooth.adam_general.core.checker.arima_checks import (
    _R_ORDERS,
    resolve_arima_orders,
)
from smooth.adam_general.core.estimator.arima_selector import arima_selector
from smooth.adam_general.core.om import OM, _check_om_loss
from smooth.adam_general.core.omg import OMG
from smooth.adam_general.core.utils.utils import SMOOTHER_DEFAULT, SmootherType

_VALID_OCCURRENCE = (
    "fixed",
    "odds-ratio",
    "inverse-odds-ratio",
    "direct",
    "general",
)

_IC_ATTR = {"AIC": "aic", "AICc": "aicc", "BIC": "bic", "BICc": "bicc"}


def _get_ic(model: Union[OM, OMG], ic_name: str) -> float:
    return float(getattr(model, _IC_ATTR[ic_name]))


class AutoOM:
    """Automatic occurrence-model selection.

    Fits an :class:`OM` (or :class:`OMG`) for each entry in ``occurrence``
    and returns the best-fitting model (lowest IC) directly.  The returned
    object is a plain :class:`OM` or :class:`OMG` with one extra attribute
    ``time_elapsed_`` (total selection time in seconds).

    Parameters
    ----------
    model :
        ETS spec passed to every candidate, including both sub-models of the
        general (OMG) candidate.
    lags :
        Seasonal lags shared across all candidates.
    orders :
        ARIMA orders, as R's ``auto.om()``: ``{"ar": ..., "i": ..., "ma": ...,
        "select": ...}``. With ``select=True`` they are the maximum orders, selected
        for each occurrence type on the residuals of its ETS (except for the
        general one) by ADAM's ARIMA selector. The default (None, without
        ``ar_order`` / ``i_order`` / ``ma_order``) is R's
        ``{"ar": [3, 3], "i": [2, 1], "ma": [3, 3], "select": True}``; pass
        ``{"ar": 0, "i": 0, "ma": 0}`` for no ARIMA.
    arima_select, ar_order, i_order, ma_order :
        The orders as in :class:`AutoADAM`, used when ``orders`` is None.
    occurrence :
        Sequence of occurrence types to try.
    h, holdout :
        Forecast horizon and holdout flag forwarded to every candidate.
    ic :
        Information criterion used for selection.
    loss, lambda_param :
        The loss (and the LASSO / RIDGE weight) of every candidate, as R's
        ``auto.om()`` passes them to ``om()``.
    n_iterations, smoother :
        The backcasting iterations and the ``msdecompose`` smoother of the
        initial states, as in :class:`OM`, passed to every candidate (both
        sub-models of the general one).
    """

    def __init__(
        self,
        model: str = "ZXZ",
        lags: Optional[List[int]] = None,
        orders: Optional[Dict[str, Any]] = None,
        occurrence: Union[List[str], str] = list(_VALID_OCCURRENCE),
        h: int = 0,
        holdout: bool = False,
        persistence: Optional[Dict[str, float]] = None,
        phi: Optional[float] = None,
        initial: Union[str, Dict[str, Any]] = "backcasting",
        n_iterations: Optional[int] = None,
        constant: bool = False,
        arma: Optional[Dict[str, Any]] = None,
        regressors: Literal["use", "select", "adapt"] = "use",
        ic: Literal["AIC", "AICc", "BIC", "BICc"] = "AICc",
        bounds: Literal["usual", "admissible", "none"] = "usual",
        verbose: int = 0,
        nlopt_kwargs: Optional[Dict[str, Any]] = None,
        ets: Literal["conventional", "adam"] = "conventional",
        arima_select: bool = False,
        ar_order: Union[int, List[int], None] = None,
        i_order: Union[int, List[int], None] = None,
        ma_order: Union[int, List[int], None] = None,
        loss: Any = "likelihood",
        lambda_param: Optional[float] = None,
        smoother: SmootherType = SMOOTHER_DEFAULT,
    ) -> None:
        _check_om_loss(loss)
        if isinstance(occurrence, str):
            occurrence = [occurrence]
        unknown = set(occurrence) - set(_VALID_OCCURRENCE)
        if unknown:
            raise ValueError(f"Unknown occurrence type(s): {unknown!r}")
        if ic not in _IC_ATTR:
            raise ValueError(f"ic must be one of {list(_IC_ATTR)}; got {ic!r}")
        self.model = model
        self.lags = lags
        self.orders = orders
        self.occurrence = list(occurrence)
        self.h = h
        self.holdout = holdout
        self.persistence = persistence
        self.phi = phi
        self.initial = initial
        self.n_iterations = n_iterations
        self.constant = constant
        self.arma = arma
        self.regressors = regressors
        self.ic = ic
        self.bounds = bounds
        self.verbose = verbose
        self.nlopt_kwargs = nlopt_kwargs
        if ets not in ("conventional", "adam"):
            raise ValueError(f"Invalid ets: {ets!r}. Must be 'conventional' or 'adam'.")
        self.ets = ets
        if smoother not in ("default", "ma", "lowess", "supsmu", "global"):
            raise ValueError(
                f"Invalid smoother: {smoother!r}. Must be one of "
                "'default', 'ma', 'lowess', 'supsmu', 'global'."
            )
        self.smoother = smoother
        self.arima_select = arima_select
        self.ar_order = ar_order
        self.i_order = i_order
        self.ma_order = ma_order
        # Passed to each candidate, as R's auto.om() passes its ellipsis to om()
        self.loss = loss
        self.lambda_param = lambda_param

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _common_kwargs(self, lags: Optional[List[int]]) -> Dict[str, Any]:
        return dict(
            lags=lags,
            h=self.h,
            holdout=self.holdout,
            initial=self.initial,
            n_iterations=self.n_iterations,
            smoother=self.smoother,
            ic=self.ic,
            bounds=self.bounds,
            verbose=self.verbose,
            nlopt_kwargs=self.nlopt_kwargs,
            loss=self.loss,
            lambda_param=self.lambda_param,
        )

    def _build_candidate(
        self, occ: str, orders: Optional[Dict[str, Any]], lags: Optional[List[int]]
    ) -> Union[OM, OMG]:
        common = self._common_kwargs(lags)
        if occ == "general":
            return OMG(
                model_a=self.model,
                model_b=self.model,
                orders_a=orders,
                constant_a=self.constant,
                persistence_a=self.persistence,
                phi_a=self.phi,
                arma_a=self.arma,
                regressors_a=self.regressors,
                ets=self.ets,
                **common,
            )
        return OM(
            model=self.model,
            occurrence=occ,
            orders=orders,
            constant=self.constant,
            persistence=self.persistence,
            phi=self.phi,
            arma=self.arma,
            regressors=self.regressors,
            ets=self.ets,
            **common,
        )

    def _fit_with_arima_selection(
        self, occ: str, y: NDArray, X: Optional[NDArray], lags: List[int], maxima
    ) -> Union[OM, OMG]:
        """The orders of a non-general occurrence type selected by ADAM's ARIMA
        selector on the residuals of the model without them, as R's auto.om()
        calls adam_arimaSelector() with om() as the fitter."""
        no_arima = {"ar": 0, "i": 0, "ma": 0, "select": False}
        baseline = None
        if self.model != "NNN" or X is not None:
            baseline = self._build_candidate(occ, no_arima, lags).fit(y, X)
        common = self._common_kwargs(lags)
        for key in ("lags", "ic", "verbose"):
            common.pop(key)
        selected = arima_selector(
            y=y,
            ets_model=self.model,
            max_ar_orders=maxima[0],
            max_i_orders=maxima[1],
            max_ma_orders=maxima[2],
            lags=lags,
            ic=self.ic,
            X=X,
            ets_baseline=baseline,
            fitter=OM,
            occurrence=occ,
            persistence=self.persistence,
            phi=self.phi,
            regressors=self.regressors,
            ets=self.ets,
            **common,
        )
        if selected["model"] is None:
            raise RuntimeError(f"AutoOM: arima_select failed for occurrence='{occ}'")
        return selected["model"]

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, y: NDArray, X: Optional[NDArray] = None) -> Union[OM, OMG]:
        """Fit all candidates and return the best model by IC.

        Parameters
        ----------
        y :
            Observation vector (binarised internally: any non-zero → 1).
        X :
            Optional regressor matrix forwarded to every candidate.

        Returns
        -------
        OM or OMG
            The best-fitting occurrence model, with ``time_elapsed_`` set to
            the total selection time in seconds.
        """
        start = time.time()
        best_ic = float("inf")
        best_model: Optional[Union[OM, OMG]] = None

        # The orders as R's auto.om() reads them: the maximum orders by lag, with
        # lag 1, without the lags of no order, unique and sorted
        orders, select = resolve_arima_orders(
            self.orders, self.ar_order, self.i_order, self.ma_order, self.arima_select
        )
        if orders is None:
            orders, select = dict(_R_ORDERS), True
        lags = list(self.lags or [1])
        maxima = [list(np.atleast_1d(orders.get(k, 0))) for k in ("ar", "i", "ma")]
        if any(np.any(np.asarray(m) > 0) for m in maxima):
            lags = [lag for lag in ([1] * (1 not in lags) + lags) if lag != 0]
            table: Dict[int, List[int]] = {}
            for k, lag in enumerate(lags):
                row = [int(m[k]) if k < len(m) else 0 for m in maxima]
                table[lag] = np.maximum(table.get(lag, row), row).tolist()
            kept = {lag: row for lag, row in table.items() if sum(row) > 0}
            lags = sorted(kept or {min(table): table[min(table)]})
            maxima = [[(kept or table)[lag][j] for lag in lags] for j in range(3)]
        else:
            select = False
            orders = {"ar": 0, "i": 0, "ma": 0, "select": False}
        if select and self.arma is not None:
            warnings.warn(
                "ARIMA order selection cannot be done with provided arma parameters. "
                "Dropping them.",
                stacklevel=2,
            )
            self.arma = None

        for occ in self.occurrence:
            try:
                if select and occ != "general":
                    m = self._fit_with_arima_selection(occ, y, X, lags, maxima)
                else:
                    no_select = {"ar": 0, "i": 0, "ma": 0, "select": False}
                    m = self._build_candidate(
                        occ, no_select if select else orders, lags
                    )
                    m.fit(y, X)
                ic_val = _get_ic(m, self.ic)
                if ic_val < best_ic:
                    best_ic = ic_val
                    best_model = m
            except Exception as exc:
                warnings.warn(
                    f"AutoOM: occurrence='{occ}' failed with {type(exc).__name__}: "
                    f"{exc} — skipping.",
                    stacklevel=2,
                )

        if best_model is None:
            raise RuntimeError("AutoOM: all occurrence types failed to fit.")

        best_model.time_elapsed_ = time.time() - start
        return best_model
