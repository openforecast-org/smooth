"""
Exponential Smoothing (ES) wrapper for ADAM.

This module provides an ES class that wraps the ADAM model for pure
Exponential Smoothing (ETS) forecasting without ARIMA components.
"""

from typing import Any, Dict, List, Literal, Optional, Union

from numpy.typing import NDArray

from smooth.adam_general.core.adam import ADAM, LOSS_OPTIONS
from smooth.adam_general.core.utils.utils import SMOOTHER_DEFAULT, SmootherType


class ES(ADAM):
    """
    Exponential Smoothing in Single Source of Error (SSOE) state space model.

    This class is a wrapper around ADAM that provides a simplified interface
    for pure ETS (Error, Trend, Seasonal) models without ARIMA components.
    It uses normal distribution for errors and conventional ETS formulation.

    The model is specified in state-space form as:

    .. math::

        y_t &= o_t(w(v_{t-l}) + h(x_t, a_{t-1}) + r(v_{t-l})\\epsilon_t)

        v_t &= f(v_{t-l}) + g(v_{t-l})\\epsilon_t

    where:

    - :math:`y_t`: Observed value at time t
    - :math:`o_t`: Occurrence indicator (Bernoulli variable for intermittent data)
    - :math:`v_t`: State vector (level, trend, seasonal components)
    - :math:`l`: Vector of lags
    - :math:`w(\\cdot)`: Measurement function
    - :math:`r(\\cdot)`: Error function (additive or multiplicative)
    - :math:`f(\\cdot)`: Transition function
    - :math:`g(\\cdot)`: Persistence function
    - :math:`\\epsilon_t`: Error term (normal distribution)

    Parameters
    ----------
    model : Union[str, List[str]], default="ZXZ"
        The type of ETS model. The first letter stands for the type of error
        ("A" or "M"), the second for trend ("N", "A", "Ad", "M" or "Md"),
        and the third for seasonality ("N", "A" or "M").

        Examples: "ANN", "AAN", "AAdN", "AAA", "AAdA", "MAdM"

        Special codes:
        - "ZZZ": Automatic selection using information criteria
        - "XXX": Select only additive components
        - "YYY": Select only multiplicative components
        - "ZXZ": Auto-select error and seasonal, additive trend only (default)
        - "CCC": Combination of all models using AIC weights

        Can also be a list of model names for custom model pool.

    lags : Optional[Union[int, List[int]]], default=None
        Seasonal period(s). Can be a single integer or a list of integers.
        E.g., ``lags=12`` is equivalent to ``lags=[12]``.
        For monthly data with annual seasonality: ``lags=12`` or ``lags=[1, 12]``.
        If None, defaults to [1] (no seasonality).

    persistence : Optional[Dict[str, float]], default=None
        Fixed persistence (smoothing) parameters. If None, estimated.
        Keys: "alpha" (level), "beta" (trend), "gamma" (seasonal).

    phi : Optional[float], default=None
        Damping parameter for damped trend models. If None, estimated when applicable.

    initial : Union[str, Dict[str, Any]], default="backcasting"
        Method for initializing states or dictionary of fixed initial values.
        String options: "backcasting", "optimal", "complete", "two-stage"

    initial_season : Optional[NDArray], default=None
        Initial values for seasonal components. If None, estimated.
        Length should be (seasonal_period - 1) for each seasonal component.

    ic : Literal["AIC", "AICc", "BIC", "BICc"], default="AICc"
        Information criterion for model selection.

    loss : LOSS_OPTIONS, default="likelihood"
        Loss function for parameter estimation.

    h : Optional[int], default=None
        Forecast horizon. Can also be set in predict().

    holdout : bool, default=False
        Whether to use holdout sample for validation.

    bounds : Literal["usual", "admissible", "none"], default="usual"
        Parameter bounds type:
        - "usual": Standard smoothing parameter constraints
        - "admissible": Stability constraints
        - "none": No constraints

    verbose : int, default=0
        Verbosity level. 0 = silent.

    regressors : Literal["use", "select"], default="use"
        How to handle external regressors.

    initial_X : Optional[NDArray], default=None
        Initial values for regressor coefficients.

    **kwargs
        Additional arguments passed to ADAM.

    See Also
    --------
    ADAM : Parent class with full attribute documentation including
        ``persistence_vector``, ``states``, ``fitted``, ``residuals``,
        ``phi_``, ``coef``, and all other properties.

    Examples
    --------
    Simple exponential smoothing::

        >>> from smooth.adam_general.core.es import ES
        >>> import numpy as np
        >>> y = np.array([112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118])
        >>> model = ES(model="ANN")
        >>> model.fit(y)
        >>> forecasts = model.predict(h=6)

    Holt-Winters with automatic model selection::

        >>> model = ES(model="ZZZ", lags=[12])
        >>> model.fit(y)
        >>> print(f"Selected model: {model.model_type}")

    Damped trend model::

        >>> model = ES(model="AAdN")
        >>> model.fit(y)
        >>> print(f"Damping parameter: {model.phi_:.3f}")

    With external regressors::

        >>> X = np.random.randn(len(y), 2)
        >>> model = ES(model="AAN", regressors="use")
        >>> model.fit(y, X=X)

    References
    ----------
    - Svetunkov, I. (2023). Forecasting and Analytics with the
      Augmented Dynamic Adaptive Model. https://openforecast.org/adam/
    - Hyndman, R.J., et al. (2008). "Forecasting with Exponential Smoothing"

    See Also
    --------
    ADAM : Full ADAM model with ETS and ARIMA components
    """

    def __init__(
        self,
        model: Union[str, List[str]] = "ZXZ",
        lags: Optional[List[int]] = None,
        persistence: Optional[Dict[str, float]] = None,
        phi: Optional[float] = None,
        initial: Union[str, Dict[str, Any], None] = "backcasting",
        initial_season: Optional[NDArray] = None,
        ic: Literal["AIC", "AICc", "BIC", "BICc"] = "AICc",
        loss: LOSS_OPTIONS = "likelihood",
        h: Optional[int] = None,
        holdout: bool = False,
        bounds: Literal["usual", "admissible", "none"] = "usual",
        smoother: SmootherType = SMOOTHER_DEFAULT,
        verbose: int = 0,
        regressors: Literal["use", "select"] = "use",
        initial_X: Optional[NDArray] = None,
        **kwargs,
    ) -> None:
        """Initialize ES model with ETS-specific parameters.

        ``smoother`` controls the msdecompose smoother used to obtain the initial
        level, trend and seasonal components; "default" (the default) uses "ma"
        for ``initial="optimal"`` and "global" otherwise.
        """
        _blocked = {
            "ar_order",
            "i_order",
            "ma_order",
            "orders",
            "arima_select",
            "constant",
            "ets",
        }
        _bad = _blocked & set(kwargs)
        if _bad:
            raise ValueError(
                f"ES() does not support these parameters: {sorted(_bad)}. "
                "Use ADAM() for models with ARIMA components or a constant term."
            )

        # Process initial values
        initial_value = self._process_initial_params(initial, initial_season, initial_X)

        # Call parent ADAM with ETS-specific settings
        super().__init__(
            model=model,
            lags=lags,
            # No ARIMA components for ES
            ar_order=0,
            i_order=0,
            ma_order=0,
            arima_select=False,
            # ETS parameters
            persistence=persistence,
            phi=phi,
            initial=initial_value,
            # Always use normal distribution for ES
            distribution="dnorm",
            # Other parameters
            ic=ic,
            loss=loss,
            h=h,
            holdout=holdout,
            bounds=bounds,
            constant=False,
            smoother=smoother,
            verbose=verbose,
            regressors=regressors,
            ets="conventional",
            **kwargs,
        )

        # Store ES-specific parameters
        self._initial_season = initial_season
        self._initial_X = initial_X

    def _process_initial_params(
        self,
        initial: Union[str, Dict[str, Any], None],
        initial_season: Optional[NDArray],
        initial_X: Optional[NDArray],
    ) -> Union[str, Dict[str, Any]]:
        """
        Process initial parameters into format expected by ADAM.

        Parameters
        ----------
        initial : Union[str, Dict[str, Any], None]
            Initial specification (string method or dict of values).
        initial_season : Optional[NDArray]
            Initial seasonal values.
        initial_X : Optional[NDArray]
            Initial regressor coefficients.

        Returns
        -------
        Union[str, Dict[str, Any]]
            Processed initial specification.
        """
        # If initial is already a dict, add seasonal and xreg if provided
        if isinstance(initial, dict):
            initial_value = initial.copy()
            if initial_season is not None:
                initial_value["seasonal"] = initial_season
            if initial_X is not None:
                initial_value["xreg"] = initial_X
            return initial_value

        # If initial is a string method
        if isinstance(initial, str):
            # If we have additional initial values, convert to dict
            if initial_season is not None or initial_X is not None:
                initial_value = {}
                if initial_season is not None:
                    initial_value["seasonal"] = initial_season
                if initial_X is not None:
                    initial_value["xreg"] = initial_X
                return initial_value
            return initial

        # None case - use default
        return "backcasting"
