"""
Multiple Seasonal ARIMA (MSARIMA) wrapper for ADAM.

This module provides an MSARIMA class that wraps the ADAM model for pure
ARIMA forecasting without ETS components, mirroring R's msarima() function.
"""

from typing import Any, Dict, List, Literal, Optional, Union

from numpy.typing import NDArray

from smooth.adam_general.core.adam import ADAM, LOSS_OPTIONS


class MSARIMA(ADAM):
    """
    Multiple Seasonal ARIMA in Single Source of Error state space form.

    This class wraps ADAM with ``model="NNN"`` and ``distribution="dnorm"``
    hardcoded, providing a clean interface for pure ARIMA (and SARIMA) models
    without ETS components. It mirrors R's ``msarima()`` function.

    The default specification is ARIMA(0,1,1), matching R's default
    ``orders=list(ar=c(0), i=c(1), ma=c(1))``.

    Parameters
    ----------
    orders : Optional[Dict[str, Any]], default=None
        R-style alternative to ``ar_order``/``i_order``/``ma_order``.
        A dict with keys ``"ar"``, ``"i"``, ``"ma"`` (each an int or list
        of ints) and optionally ``"select"`` (bool). Example::

            orders={"ar": [1, 1], "i": [1, 1], "ma": [1, 1]}

        If ``ar_order``, ``i_order``, or ``ma_order`` are non-zero they
        take priority over ``orders``.

    lags : Optional[List[int]], default=None
        Seasonal period(s). E.g. ``lags=[1, 12]`` for monthly data.
        If None, defaults to ``[1]`` (non-seasonal).

    ar_order : Union[int, List[int]], default=0
        Autoregressive order(s). Matches R default ``ar=c(0)``.

    i_order : Union[int, List[int]], default=1
        Integration order(s). Matches R default ``i=c(1)``.

    ma_order : Union[int, List[int]], default=1
        Moving average order(s). Matches R default ``ma=c(1)``.

    arima_select : bool, default=False
        Whether to perform automatic ARIMA order selection. Equivalent to
        including ``"select": True`` in the ``orders`` dict.

    constant : Union[bool, float], default=False
        Whether to include a constant (drift) term. ``True`` estimates it;
        a numeric value fixes it. The model name will show "with drift"
        when ``i_order > 0``, or "with constant" otherwise.
        The fitted value is accessible via ``model.constant_value``.

    arma : Optional[Dict[str, Any]], default=None
        Fixed ARMA parameter values (not estimated). If None, all ARMA
        parameters are estimated.

    initial : Union[str, Dict[str, Any]], default="backcasting"
        Initialisation method or dict of fixed initial values.
        String options: ``"backcasting"``, ``"optimal"``, ``"complete"``,
        ``"two-stage"``.

    initial_X : Optional[NDArray], default=None
        Initial values for regressor coefficients.

    ic : Literal["AIC", "AICc", "BIC", "BICc"], default="AICc"
        Information criterion for model selection.

    loss : LOSS_OPTIONS, default="likelihood"
        Loss function for parameter estimation.

    h : Optional[int], default=None
        Forecast horizon. Can also be set in ``predict()``.

    holdout : bool, default=False
        Whether to use a holdout sample for validation.

    bounds : Literal["usual", "admissible", "none"], default="usual"
        Parameter bounds type.

    verbose : int, default=0
        Verbosity level. 0 = silent.

    regressors : Literal["use", "select", "adapt"], default="use"
        How to handle external regressors.

    **kwargs
        Additional arguments passed to ADAM.

    See Also
    --------
    ADAM : Parent class with full attribute documentation.
    ES : ETS-only wrapper (counterpart for pure ETS models).

    Examples
    --------
    Default ARIMA(0,1,1)::

        >>> from smooth import MSARIMA
        >>> import numpy as np
        >>> y = np.cumsum(np.random.randn(60)) + 100.0
        >>> model = MSARIMA()
        >>> model.fit(y)

    ARIMA(1,1,1) with drift::

        >>> model = MSARIMA(ar_order=1, i_order=1, ma_order=1, constant=True)
        >>> model.fit(y)
        >>> print(f"Drift: {model.constant_value:.4f}")

    SARIMA(1,1,1)(1,1,1)[12] via R-style dict::

        >>> model = MSARIMA(
        ...     orders={"ar": [1, 1], "i": [1, 1], "ma": [1, 1]},
        ...     lags=[1, 12],
        ... )
        >>> model.fit(y)

    References
    ----------
    - Svetunkov, I. (2023). Forecasting and Analytics with the
      Augmented Dynamic Adaptive Model. https://openforecast.org/adam/
    """

    def __init__(
        self,
        orders: Optional[Dict[str, Any]] = None,
        lags: Optional[List[int]] = None,
        ar_order: Union[int, List[int]] = 0,
        i_order: Union[int, List[int]] = 1,
        ma_order: Union[int, List[int]] = 1,
        arima_select: bool = False,
        constant: Union[bool, float] = False,
        arma: Optional[Dict[str, Any]] = None,
        initial: Union[str, Dict[str, Any], None] = "backcasting",
        initial_X: Optional[NDArray] = None,
        ic: Literal["AIC", "AICc", "BIC", "BICc"] = "AICc",
        loss: LOSS_OPTIONS = "likelihood",
        h: Optional[int] = None,
        holdout: bool = False,
        bounds: Literal["usual", "admissible", "none"] = "usual",
        verbose: int = 0,
        regressors: Literal["use", "select", "adapt"] = "use",
        **kwargs,
    ) -> None:
        """Initialize MSARIMA model with ARIMA-specific parameters."""
        _blocked = {"model", "persistence", "phi", "distribution"}
        _bad = _blocked & set(kwargs)
        if _bad:
            raise ValueError(
                f"MSARIMA() does not support these parameters: {sorted(_bad)}. "
                "Use ADAM() for full model control."
            )

        initial_value = self._build_initial(initial, initial_X)

        # When an explicit orders dict is provided, zero the scalar params so the
        # dict takes priority in ADAM's fit() merge logic.
        if orders is not None:
            ar_order, i_order, ma_order = 0, 0, 0

        super().__init__(
            model="NNN",
            lags=lags,
            ar_order=ar_order,
            i_order=i_order,
            ma_order=ma_order,
            orders=orders,
            arima_select=arima_select,
            constant=constant,
            arma=arma,
            initial=initial_value,
            distribution="dnorm",
            ic=ic,
            loss=loss,
            h=h,
            holdout=holdout,
            bounds=bounds,
            verbose=verbose,
            regressors=regressors,
            **kwargs,
        )

        self._initial_X = initial_X

    @staticmethod
    def _build_initial(
        initial: Union[str, Dict[str, Any], None],
        initial_X: Optional[NDArray],
    ) -> Union[str, Dict[str, Any]]:
        """Merge initial method/values with optional regressor initial values."""
        if isinstance(initial, dict):
            result = initial.copy()
            if initial_X is not None:
                result["xreg"] = initial_X
            return result
        if isinstance(initial, str):
            if initial_X is not None:
                return {"xreg": initial_X}
            return initial
        return "backcasting"
