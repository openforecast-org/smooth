"""
AutoMSARIMA — automatic ARIMA order selection wrapper for ADAM.

Mirrors R's ``auto.msarima()`` function: fixes ``model="NNN"`` and
``distribution="dnorm"``, always enables ARIMA order selection, and
forwards all other parameters to :class:`AutoADAM`.
"""

from typing import Any, Dict, List, Literal, Optional, Union

from numpy.typing import NDArray

from smooth.adam_general.core.adam import LOSS_OPTIONS
from smooth.adam_general.core.auto_adam import AutoADAM
from smooth.adam_general.core.msarima import MSARIMA


class AutoMSARIMA(AutoADAM):
    """
    Automatic Multiple Seasonal ARIMA with order selection.

    Wraps :class:`AutoADAM` with ``model="NNN"`` and ``distribution="dnorm"``
    fixed, providing automatic ARIMA order selection for pure ARIMA (and
    SARIMA) models without ETS components. Mirrors R's ``auto.msarima()``.

    Parameters
    ----------
    lags : Optional[List[int]], default=None
        Seasonal period(s). E.g. ``lags=[1, 12]`` for monthly data.
        If None, defaults to ``[1]`` (non-seasonal).

    ar_order : Union[int, List[int]], default=[3, 3]
        Maximum AR order(s) per lag level for selection.
        Matches R's ``orders=list(ar=c(3,3))``.

    i_order : Union[int, List[int]], default=[2, 1]
        Maximum integration order(s) per lag level.
        Matches R's ``orders=list(i=c(2,1))``.

    ma_order : Union[int, List[int]], default=[3, 3]
        Maximum MA order(s) per lag level for selection.
        Matches R's ``orders=list(ma=c(3,3))``.

    orders : Optional[Dict[str, Any]], default=None
        R-style alternative to scalar max orders. A dict with keys
        ``"ar"``, ``"i"``, ``"ma"`` (each an int or list). When provided,
        ``ar_order``/``i_order``/``ma_order`` are ignored.

    constant : Union[bool, float], default=False
        Whether to include a constant (drift) term.

    initial : Union[str, Dict[str, Any]], default="backcasting"
        Initialisation method or dict of fixed initial values.
        String options: ``"backcasting"``, ``"optimal"``, ``"complete"``,
        ``"two-stage"``.

    initial_X : Optional[NDArray], default=None
        Initial values for regressor coefficients (equivalent to R's
        ``initialX``).

    ic : Literal["AIC", "AICc", "BIC", "BICc"], default="AICc"
        Information criterion for model comparison during selection.

    loss : LOSS_OPTIONS, default="likelihood"
        Loss function for parameter estimation.

    h : Optional[int], default=None
        Forecast horizon. Can also be set in ``predict()``.

    holdout : bool, default=False
        Whether to use a holdout sample.

    bounds : Literal["usual", "admissible", "none"], default="usual"
        Parameter bounds type.

    regressors : Literal["use", "select", "adapt"], default="use"
        How to handle external regressors.

    outliers : Literal["ignore", "use", "select"], default="ignore"
        Outlier handling mode (see :class:`AutoADAM`).

    level : float, default=0.99
        Confidence level for outlier detection.

    verbose : int, default=0
        Verbosity level. 0 = silent.

    **kwargs
        Additional arguments forwarded to :class:`AutoADAM`.

    See Also
    --------
    AutoADAM : Full automatic model selection.
    MSARIMA : Fixed-order MSARIMA wrapper.

    Examples
    --------
    Automatic non-seasonal ARIMA::

        >>> from smooth import AutoMSARIMA
        >>> import numpy as np
        >>> y = np.cumsum(np.random.randn(100)) + 100.0
        >>> model = AutoMSARIMA(lags=[1])
        >>> model.fit(y)
        >>> print(model)

    Automatic seasonal ARIMA for monthly data::

        >>> model = AutoMSARIMA(lags=[1, 12])
        >>> model.fit(y)
        >>> fc = model.predict(h=24)

    References
    ----------
    - Svetunkov, I. (2023). Forecasting and Analytics with the
      Augmented Dynamic Adaptive Model. https://openforecast.org/adam/
    """

    def __init__(
        self,
        lags: Optional[List[int]] = None,
        ar_order: Union[int, List[int]] = [3, 3],  # noqa: B006
        i_order: Union[int, List[int]] = [2, 1],  # noqa: B006
        ma_order: Union[int, List[int]] = [3, 3],  # noqa: B006
        orders: Optional[Dict[str, Any]] = None,
        constant: Union[bool, float] = False,
        initial: Union[str, Dict[str, Any], None] = "backcasting",
        initial_X: Optional[NDArray] = None,
        ic: Literal["AIC", "AICc", "BIC", "BICc"] = "AICc",
        loss: LOSS_OPTIONS = "likelihood",
        h: Optional[int] = None,
        holdout: bool = False,
        bounds: Literal["usual", "admissible", "none"] = "usual",
        regressors: Literal["use", "select", "adapt"] = "use",
        outliers: Literal["ignore", "use", "select"] = "ignore",
        level: float = 0.99,
        verbose: int = 0,
        **kwargs,
    ) -> None:
        """Initialise AutoMSARIMA."""
        _blocked = {"model", "distribution", "arima_select"}
        _bad = _blocked & set(kwargs)
        if _bad:
            raise ValueError(
                f"AutoMSARIMA() does not support these parameters: {sorted(_bad)}. "
                "Use AutoADAM() for full model control."
            )

        initial_value = MSARIMA._build_initial(initial, initial_X)

        super().__init__(
            model="NNN",
            lags=lags,
            ar_order=ar_order,
            i_order=i_order,
            ma_order=ma_order,
            orders=orders,
            arima_select=True,
            distribution="dnorm",
            constant=constant,
            initial=initial_value,
            ic=ic,
            loss=loss,
            h=h,
            holdout=holdout,
            bounds=bounds,
            regressors=regressors,
            outliers=outliers,
            level=level,
            verbose=verbose,
            **kwargs,
        )

        self._initial_X = initial_X

    def __repr__(self) -> str:
        """Return string representation of fitted AutoMSARIMA model."""
        try:
            self._check_is_fitted()
            orders = getattr(self, "_selected_arima_orders", {})
            ar = orders.get("ar_orders", "?")
            d = orders.get("i_orders", "?")
            ma = orders.get("ma_orders", "?")
            return (
                f"AutoMSARIMA: ARIMA({ar},{d},{ma})\n"
                f"IC ({self.ic}): "
                f"{self._all_ic_values.get('dnorm', float('nan')):.4f}"
            )
        except Exception:
            return "AutoMSARIMA (not fitted)"
