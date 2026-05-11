import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats as scipy_stats

from smooth.adam_general.core.checker import parameters_checker
from smooth.adam_general.core.creator import architector, creator
from smooth.adam_general.core.estimator import (
    estimator,
    selector,
)
from smooth.adam_general.core.forecaster import forecaster, preparator
from smooth.adam_general.core.utils.ic import calculate_ic_weights, ic_function
from smooth.adam_general.core.utils.n_param import NParam
from smooth.adam_general.core.utils.utils import SMOOTHER_DEFAULT

# Note: adam_cpp instance is stored in self and passed to forecasting functions


# Type hint groups
DISTRIBUTION_OPTIONS = Literal[
    "default", "dnorm", "dlaplace", "ds", "dgnorm", "dlnorm", "dinvgauss", "dgamma"
]

LOSS_OPTIONS = Literal[
    "likelihood",
    "GPL",
    "MSE",
    "MAE",
    "HAM",
    "MSEh",
    "MAEh",
    "HAMh",
    "MSCE",
    "MACE",
    "CHAM",
    "TMSE",
    "TMAE",
    "THAM",
    "GTMSE",
    "GTAME",
    "GTHAM",
    "LASSO",
    "RIDGE",
]

OCCURRENCE_OPTIONS = Literal[
    "none", "auto", "fixed", "general", "odds-ratio", "inverse-odds-ratio", "direct"
]

INITIAL_OPTIONS = Optional[
    Union[
        Dict[str, Any],
        Literal["backcasting", "optimal", "complete", "two-stage", "provided"],
        Tuple[str, ...],
    ]
]


@dataclass
class OutlierDummy:
    """Container returned by ADAM.outlierdummy()."""

    outliers: Optional[NDArray]  # (n, m) 0/1 matrix, one column per outlier, or None
    id: NDArray  # 0-based indices of outlier observations
    statistic: NDArray  # [lower, upper] quantile bounds
    level: float
    type: str  # "rstandard" or "rstudent"


class ADAM:
    """
    ADAM: Augmented Dynamic Adaptive Model for Time Series Forecasting.

    ADAM is an advanced state-space modeling framework that combines
    **ETS** (Error, Trend, Seasonal) and **ARIMA** components into a unified
    Single Source of Error (SSOE) model. It provides a flexible, data-driven
    approach to time series forecasting with automatic model selection,
    parameter estimation, and prediction intervals.

    **Mathematical Form**:

    The ADAM model is specified in state-space form as:

    .. math::

        y_t &= o_t(w(v_{t-l}) + h(x_t, a_{t-1}) + r(v_{t-l})\\epsilon_t)

        v_t &= f(v_{t-l}, a_{t-1}) + g(v_{t-l}, a_{t-1}, x_t)\\epsilon_t

    where:

    - :math:`y_t`: Observed value at time t
    - :math:`o_t`: Occurrence indicator (Bernoulli variable for intermittent
      data, 1 otherwise)
    - :math:`v_t`: State vector (level, trend, seasonal, ARIMA components)
    - :math:`l`: Vector of lags
    - :math:`x_t`: Vector of exogenous variables
    - :math:`a_t`: Parameters for exogenous variables
    - :math:`w(\\cdot)`: Measurement function
    - :math:`r(\\cdot)`: Error function (additive or multiplicative)
    - :math:`f(\\cdot)`: Transition function (state evolution)
    - :math:`g(\\cdot)`: Persistence function (smoothing parameters)
    - :math:`\\epsilon_t`: Error term (various distributions supported)

    **Key Features**:

    1. **Unified Framework**: Seamlessly combines ETS and ARIMA in a single model
    2. **Multiple Seasonality**: Supports multiple seasonal periods
       (e.g., daily + weekly)
    3. **Automatic Selection**: Branch & Bound algorithm for efficient model selection
    4. **Flexible Distributions**: Normal, Laplace, Gamma, Log-Normal, and more
    5. **Intermittent Demand**: Built-in occurrence models for sparse data
    6. **External Regressors**: Include covariates with adaptive or fixed coefficients
    7. **Scikit-learn Compatible**: Familiar `.fit()` and `.predict()` API

    **Model Specification**:

    Models are specified using a string notation:

    - **ETS Models**: "ETS" where E=Error, T=Trend, S=Seasonal

      * E (Error): "A" (Additive), "M" (Multiplicative)
      * T (Trend): "N" (None), "A" (Additive), "Ad" (Additive Damped),
        "M" (Multiplicative), "Md" (Multiplicative Damped)
      * S (Seasonal): "N" (None), "A" (Additive), "M" (Multiplicative)

      Examples: "ANN" (Simple Exponential Smoothing), "AAN" (Holt's Linear),
      "AAA" (Holt-Winters Additive)

    - **Automatic Selection**:

      * "ZZZ": Select best model using Branch & Bound
      * "XXX": Select only additive components
      * "YYY": Select only multiplicative components
      * "ZXZ": Auto-select error and seasonal, additive trend only (**default**, safer)
      * "FFF": Full search across all 30 ETS model types

    - **ARIMA Models**: Specified via `ar_order`, `i_order`, `ma_order` parameters

      * Supports seasonal ARIMA: SARIMA(p,d,q)(P,D,Q)m
      * Multiple seasonality: e.g., hourly data with daily (24) and
        weekly (168) patterns

    **Supported Error Distributions**:

    - **Normal** (``distribution="dnorm"``): Default for additive errors
    - **Gamma** (``distribution="dgamma"``): Default for multiplicative errors
    - **Laplace** (``distribution="dlaplace"``): For heavy-tailed errors (MAE loss)
    - **Log-Normal** (``distribution="dlnorm"``): For positive-only data
    - **Inverse Gaussian** (``distribution="dinvgauss"``): For skewed positive data
    - **S distribution** (``distribution="ds"``): For extremely heavy-tailed data
    - **Generalized Normal** (``distribution="dgnorm"``): Flexible shape parameter

    Distribution is auto-selected based on loss function if ``distribution=None``.

    **Loss Functions**:

    - ``loss="likelihood"``: Maximum likelihood estimation (**default**)
    - ``loss="MSE"``: Mean Squared Error
    - ``loss="MAE"``: Mean Absolute Error
    - ``loss="HAM"``: Half-Absolute Moment
    - ``loss="MSEh"``: Multi-step MSE (h-step ahead)
    - ``loss="LASSO"``: L1 regularization for variable selection
    - ``loss="RIDGE"``: L2 regularization for shrinkage

    **Multistep loss functions**:

    - ``loss="MSEh"``: Mean Squared Error for specific h-steps ahead
    - ``loss="TMSE"``: Trace Mean Squared Error (sum of MSEh from 1 to h)
    - ``loss="GTMSE"``: Geometric Trace Mean Squared Error (sum of logs of MSEh
      from 1 to h)
    - ``loss="MSCE"``: Mean Squared Cumulative Error (sum of MSEh from 1 to h and
      covariances between them)
    - ``loss="GPL"``: Generalised Predictive Likelihood (minimum of the determinant of
      the covariance matrix of multistep errors)

    **Initialization Methods**:

    - ``initial="optimal"``: Optimize all initial states (default)
    - ``initial="backcasting"``: Use backcasting to initialize states
    - ``initial="two-stage"``: Backcast then optimize
    - ``initial="complete"``: Pure backcasting without optimization
    - ``initial={"level": 100, ...}``: Provide custom initial states

    **Workflow Example**:

    .. code-block:: python

        from smooth import ADAM
        import numpy as np

        # Generate sample data
        y = np.array([112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118] * 3)

        # Automatic model selection
        model = ADAM(model="ZZZ", lags=[12], ic="AICc")
        ADAM_fit = model.fit(y)
        print(ADAM_fit)

        # Generate 12-step ahead forecasts with intervals
        forecasts = model.predict(h=12, level=0.95)
        print(forecasts)

        # Access fitted parameters
        print(f"Model: {model.model_name}")
        print(f"Alpha: {model.persistence_level_:.3f}")
        print(f"AICc: {model.aicc}")

    **Attributes (After Fitting)**:

    The model stores fitted results as attributes with trailing underscores
    (scikit-learn convention):

    - ``persistence_level_``: Level smoothing parameter (α)
    - ``persistence_trend_``: Trend smoothing parameter (β)
    - ``persistence_seasonal_``: Seasonal smoothing parameter(s) (γ)
    - ``phi_``: Damping parameter (φ)
    - ``initial_states_``: Estimated initial states
    - ``arma_parameters_``: AR/MA coefficients (if ARIMA)

    Additional fitted attributes:

    - ``model_name``: Full model specification string
    - ``coef``: Estimated parameter vector B
    - ``states``: State matrix over time
    - ``persistence_vector``: Named persistence parameters
    - ``transition``: Transition matrix

    **Performance Considerations**:

    - **Small Data** (T < 100): Use "backcasting" initialization,
      it's faster
    - **Large Data** (T > 1000): "optimal" initialization works well
    - **Multiple Seasonality**: Can be slow; consider simpler models
      first
    - **Model Selection**: "ZZZ" with Branch & Bound is much faster than
      "FFF" exhaustive search

    **Common Use Cases**:

    1. **Automatic Forecasting**: ``ADAM(model="ZXZ", lags=[12])`` -
       Let the model choose
    2. **Intermittent Demand**: ``ADAM(model="ANN", occurrence="auto")`` -
       For sparse data
    3. **External Regressors**: ``ADAM(model="AAN").fit(y, X=regressors)`` -
       Include covariates
    4. **Multiple Seasonality**: ``ADAM(model="AAA", lags=[24, 168])`` -
       Hourly data with daily/weekly patterns
    5. **ARIMA**: ``ADAM(model="NNN", ar_order=1, i_order=1, ma_order=1)`` -
       Pure ARIMA(1,1,1)
    6. **Custom Model**: ``ADAM(model="MAM", persistence={"alpha": 0.3})`` -
       Fix some parameters

    **Comparison to R's smooth::adam**:

    This Python implementation is a direct translation of the R smooth
    package's ``adam()`` function, maintaining mathematical equivalence
    while adapting to scikit-learn conventions:

    - R: ``adam(data, model="ZZZ", h=10)`` →
      Python: ``ADAM(model="ZZZ").fit(y).predict(h=10)``
    - R: ``persistence=list(alpha=0.3)`` → Python: ``persistence={"alpha": 0.3}``
    - R: ``orders=list(ar=c(1,1))`` → Python: ``ar_order=[1, 1]``

    **References**:

    - Svetunkov, I. (2023). Forecasting and Analytics with the Augmented Dynamic
      Adaptive Model. https://openforecast.org/adam/
    - Hyndman, R.J., et al. (2008). "Forecasting with Exponential Smoothing"
    - Svetunkov, I. & Boylan, J.E. (2017). "State-space ARIMA for
      supply-chain forecasting"
    - Svetunkov, I. & Kourentzes, N. & Killick, R. (2023). "Multi-step
      estimators and shrinkage effect in time series models".
      DOI: 10.1007/s00180-023-01377-x

    See Also
    --------
    adam.fit : Fit the ADAM model to data
    adam.predict : Generate point forecasts
    adam.predict_intervals : Generate prediction intervals
    selector : Automatic model selection function
    estimator : Parameter estimation function
    forecaster : Forecasting function
    print : Print the outputs of the ADAM class

    Examples
    --------
    Simple exponential smoothing::

        >>> from smooth import ADAM
        >>> import numpy as np
        >>> y = np.array([112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118])
        >>> model = ADAM(model="ANN", lags=[1])
        >>> model.fit(y)
        >>> forecasts = model.predict(h=6)
        >>> print(forecasts)

    Automatic model selection with multiple seasonality::

        >>> model = ADAM(model="ZZZ", lags=[12], ic="AICc")
        >>> model.fit(y)
        >>> print(f"Selected model: {model.model_name}")

    SARIMA(1,1,1)(1,1,1)₁₂::

        >>> model = ADAM(
        ...     model="NNN",  # Pure ARIMA
        ...     ar_order=[1, 1],
        ...     i_order=[1, 1],
        ...     ma_order=[1, 1],
        ...     lags=[1, 12]
        ... )
        >>> model.fit(y)
        >>> forecasts = model.predict(h=12)

    With external regressors::

        >>> X = np.random.randn(len(y), 2)  # Two regressors
        >>> X_future = np.random.randn(6, 2)  # Regressors for forecast period
        >>> model = ADAM(model="AAN", regressors="use")
        >>> model.fit(y, X=X)
        >>> forecasts = model.predict(h=6, X=X_future)

    Fix some parameters, estimate others::

        >>> model = ADAM(
        ...     model="AAA",
        ...     lags=[12],
        ...     persistence={"alpha": 0.3},  # Fix alpha, estimate beta and gamma
        ...     initial="backcasting"
        ... )
        >>> model.fit(y)

    Intermittent demand forecasting::

        >>> sparse_data = np.array([0, 0, 15, 0, 0, 23, 0, 0, 0, 18, 0, 0])
        >>> model = ADAM(model="ANN", occurrence="auto")
        >>> model.fit(sparse_data)
        >>> forecasts = model.predict(h=6)  # Accounts for zero-demand probability
    """

    def __init__(
        self,
        model: Union[str, List[str]] = "ZXZ",
        lags: Optional[NDArray] = None,
        # ARIMA specific parameters
        ar_order: Union[int, List[int]] = 0,
        i_order: Union[int, List[int]] = 0,
        ma_order: Union[int, List[int]] = 0,
        orders: Optional[Dict[str, Any]] = None,
        arima_select: bool = False,
        # end of ARIMA specific parameters
        constant: bool = False,
        regressors: Literal["use", "select", "adapt"] = "use",
        distribution: Optional[DISTRIBUTION_OPTIONS] = None,
        loss: LOSS_OPTIONS = "likelihood",
        loss_horizon: Optional[int] = None,
        # outlier detection
        outliers: Literal["ignore", "use", "select"] = "ignore",
        outliers_level: float = 0.99,
        # end of outlier detection
        ic: Literal["AIC", "AICc", "BIC", "BICc"] = "AICc",
        bounds: Literal["usual", "admissible", "none"] = "usual",
        occurrence: OCCURRENCE_OPTIONS = "none",
        # ---- These are the estimated parameters that we can choose to fix ----
        # Dictionary of terms e.g. {"alpha": 0.5, "beta": 0.5}
        persistence: Optional[Dict[str, float]] = None,
        phi: Optional[float] = None,
        initial: INITIAL_OPTIONS = "backcasting",
        # Number of iterations for backcasting (default 2 for backcasting, 1 otherwise)
        n_iterations: Optional[int] = None,
        # TODO: enforce the structure of this
        arma: Optional[Dict[str, Any]] = None,
        # ----- End of parameters----
        verbose: int = 0,
        # Parameters moved from fit
        h: Optional[int] = None,
        holdout: bool = False,
        fast: bool = False,
        lambda_param: Optional[float] = None,
        # Profile parameters
        profiles_recent_provided: bool = False,
        profiles_recent_table: Optional[Any] = None,
        # Fisher information matrix: We're skipping for now and we'll use composition
        # for it like Grid Search in scikit-learn.
        # initial values for optimization parameters:
        nlopt_initial: Optional[Dict[str, Any]] = None,
        nlopt_upper: Optional[Dict[str, Any]] = None,
        nlopt_lower: Optional[Dict[str, Any]] = None,
        nlopt_kargs: Optional[Dict[str, Any]] = None,
        # specific to losses or distributions
        reg_lambda: Optional[float] = None,
        gnorm_shape: Optional[float] = None,
        smoother: Literal["lowess", "ma", "global"] = SMOOTHER_DEFAULT,
        ets: Literal["conventional", "adam"] = "conventional",
        **kwargs,
    ) -> None:
        """
        Initialize the ADAM model with specified parameters.

        Parameters
        ----------
        model : Union[str, List[str]], default="ZXZ"
            Model specification string (e.g., "ANN" for ETS) or
            list of model strings.
        lags : Optional[Union[int, List[int]]], default=None
            Seasonal period(s). Can be a single integer or a list of integers.
            E.g., ``lags=12`` is equivalent to ``lags=[12]``.
        ar_order : Union[int, List[int]], default=0
            Autoregressive order(s) for ARIMA components.
        i_order : Union[int, List[int]], default=0
            Integration order(s) for ARIMA components.
        ma_order : Union[int, List[int]], default=0
            Moving average order(s) for ARIMA components.
        orders : Optional[Dict[str, Any]], default=None
            R-style alternative to ``ar_order``/``i_order``/``ma_order``.
            A dict with keys ``"ar"``, ``"i"``, ``"ma"`` (each an int or list
            of ints) and optionally ``"select"`` (bool).  Example::

                orders={"ar": [1, 1], "i": [1, 1], "ma": [2, 2]}

            If ``ar_order``, ``i_order``, or ``ma_order`` are non-zero they
            take priority over ``orders``.
        arima_select : bool, default=False
            Whether to perform automatic ARIMA order selection.
        constant : Union[bool, float], default=False
            Whether to include a constant (intercept/drift) term.

            - ``True``: estimate the constant as a free parameter.
            - ``False``: no constant.
            - A numeric value: fix the constant at that value (not estimated).

            The model name reflects the role of the constant:

            - **"with drift"** — when the model is ETS *or* ARIMA with any
              integration order > 0 (e.g. ``ETS(ANN) with drift``,
              ``ARIMA(1,1,1) with drift``).
            - **"with constant"** — when the model is a pure non-integrated
              ARIMA (all ``i_order = 0``), e.g. ``ARIMA(1,0,1) with constant``.

            The fitted value is accessible via ``model.constant_value``.
        regressors : Literal["use", "select", "adapt"], default="use"
            How to handle external regressors.
        distribution : Optional[DISTRIBUTION_OPTIONS], default=None
            Error distribution. If None, it is selected automatically based
            on the loss function.
        loss : LOSS_OPTIONS, default="likelihood"
            Loss function for parameter estimation.
        loss_horizon : Optional[int], default=None
            Number of steps for multi-step loss functions (e.g., MSEh).
        outliers : Literal["ignore", "use", "select"], default="ignore"
            Outlier handling: ``"ignore"`` skips detection; ``"use"`` detects
            outliers and includes their dummies as fixed regressors; ``"select"``
            also expands each dummy with lag/lead columns and lets the regressor
            selection mechanism choose which matter.
        outliers_level : float, default=0.99
            Confidence level for outlier detection.
        ic : Literal["AIC", "AICc", "BIC", "BICc"], default="AICc"
            Information criterion for model selection.
        bounds : Literal["usual", "admissible", "none"], default="usual"
            Parameter bounds specification during optimization.
        occurrence : OCCURRENCE_OPTIONS, default="none"
            Occurrence model type for intermittent data.
        persistence : Optional[Dict[str, float]], default=None
            Fixed persistence parameters (e.g., {"alpha": 0.5, "beta": 0.5}).
            If None, parameters are estimated.
        phi : Optional[float], default="backcasting"
            Fixed damping parameter for damped trend models. If None,
            estimated if applicable.
        initial : INITIAL_OPTIONS, default=None
            Method for initializing states or fixed initial states. Can be a string
            (e.g., 'optimal', 'backcasting'), a dictionary of initial state values,
            or a tuple of state names to initialize.
        arma : Optional[Dict[str, Any]], default=None
            Fixed ARMA parameters specification. If None, estimated if applicable.
        verbose : int, default=0
            Verbosity level (0=silent, higher values indicate more output).
        h : Optional[int], default=None
            Forecast horizon. If None during initialization, can be set in `predict`.
        holdout : bool, default=False
            Whether to use a holdout sample for validation during the fit process.
        fast : bool, default=False
            Whether to use faster, possibly less accurate, estimation methods.
        lambda_param : Optional[float], default=None
            Lambda parameter for Box-Cox transformation or regularization.
        profiles_recent_provided : bool, default=False
            Whether recent profiles (e.g., for exogenous variables) are provided.
        profiles_recent_table : Optional[Any], default=None
            Table containing recent profiles data.
        nlopt_initial : Optional[Dict[str, Any]], default=None
            Initial values for optimization parameters for NLopt solver.
        nlopt_upper : Optional[Dict[str, Any]], default=None
            Upper bounds for optimization parameters for NLopt solver.
        nlopt_lower : Optional[Dict[str, Any]], default=None
            Lower bounds for optimization parameters for NLopt solver.
        nlopt_kargs : Optional[Dict[str, Any]], default=None
            Additional keyword arguments for optimization. Supported keys:

            - ``print_level`` (int): Verbosity level for optimization progress
              (default=0). When >0, prints parameter vector B and cost function
              value on every iteration.
              Output format: ``Iter N: B=[val1, val2, ...] -> CF=value``
            - ``xtol_rel`` (float): Relative tolerance on optimization parameters
              (default=1e-6). Optimization stops when parameter changes are
              smaller than xtol_rel * |params|.
            - ``xtol_abs`` (float): Absolute tolerance on optimization parameters
              (default=1e-8). Optimization stops when parameter changes are
              smaller than xtol_abs.
            - ``ftol_rel`` (float): Relative tolerance on function value
              (default=1e-8). Optimization stops when CF changes are smaller
              than ftol_rel * |CF|.
            - ``ftol_abs`` (float): Absolute tolerance on function value (default=0).
              Optimization stops when CF changes are smaller than ftol_abs.
            - ``algorithm`` (str): NLopt algorithm name
              (default="NLOPT_LN_NELDERMEAD"). Common alternatives:
              "NLOPT_LN_SBPLX" (Subplex), "NLOPT_LN_COBYLA" (COBYLA),
              "NLOPT_LN_BOBYQA" (BOBYQA). Use "LN_" prefix for derivative-free
              algorithms.

            Example::

                model = ADAM(model="AAN", nlopt_kargs={
                    "print_level": 1,
                    "xtol_rel": 1e-8,
                    "algorithm": "NLOPT_LN_SBPLX"
                })
        reg_lambda : Optional[float], default=None
            Regularization parameter specifically for LASSO/RIDGE losses.
        gnorm_shape : Optional[float], default=None
            Shape parameter 's' for the generalized normal distribution.
        smoother : Literal["lowess", "ma", "global"], default="global"
            Smoother type for time series decomposition in initial state estimation.

            - "lowess": Uses LOWESS (Locally Weighted Scatterplot Smoothing) for both
              trend and seasonal pattern extraction. This is the default.
            - "ma": Uses simple moving average for both trend and seasonality.
            - "global": Uses lowess for trend and "ma" (moving average) for seasonality.
              Provides robust trend estimation while avoiding lowess instability for
              small seasonal samples.
        """
        # Start time for measuring computation duration
        self._start_time = time.time()

        # Store initialization parameters
        self.model = model
        self.lags = lags
        self.ar_order = ar_order
        self.i_order = i_order
        self.ma_order = ma_order
        self._init_orders = orders
        self.arima_select = arima_select
        self.constant = constant
        self.regressors = regressors
        self.distribution = distribution
        self.loss = loss
        self.loss_horizon = loss_horizon
        self.outliers = outliers
        self.outliers_level = outliers_level
        self.ic = ic
        self.bounds = bounds
        self.occurrence = occurrence
        self.persistence = persistence
        self.phi = phi
        self.initial = initial
        self.n_iterations = n_iterations
        self.arma = arma
        self.verbose = verbose
        self.nlopt_initial = nlopt_initial
        self.nlopt_upper = nlopt_upper
        self.nlopt_lower = nlopt_lower
        self.nlopt_kargs = nlopt_kargs
        self.reg_lambda = reg_lambda
        self.gnorm_shape = gnorm_shape
        self.smoother = smoother
        if ets not in ("conventional", "adam"):
            raise ValueError(f"Invalid ets: {ets!r}. Must be 'conventional' or 'adam'.")
        self.ets = ets

        # Store parameters that were moved from fit
        self.h = h
        self.holdout = holdout
        self.fast = fast
        # Handle 'lambda' from kwargs (since 'lambda' is a reserved word in Python)
        # Users can pass either lambda_param=0.5 or **{'lambda': 0.5}
        if "lambda" in kwargs:
            self.lambda_param = kwargs["lambda"]
        else:
            self.lambda_param = lambda_param

        # Handle 'print_level' from kwargs for convenience
        # Users can pass print_level=1 directly or via nlopt_kargs={'print_level': 1}
        if "print_level" in kwargs:
            if self.nlopt_kargs is None:
                self.nlopt_kargs = {}
            self.nlopt_kargs["print_level"] = kwargs["print_level"]

        # Store profile parameters
        self.profiles_recent_provided = profiles_recent_provided
        self.profiles_recent_table = profiles_recent_table

    def fit(
        self,
        y: NDArray,
        X: Optional[NDArray] = None,
    ):
        """
        Fit the ADAM model to time series data.

        This method estimates the model parameters, selects the best model
        (if automatic selection is enabled), and prepares the model for
        forecasting. It implements the complete ADAM estimation pipeline:
        parameter checking, model architecture creation, state-space matrix
        construction, parameter optimization, and model preparation.

        **Estimation Process**:

        1. **Parameter Validation**: Check all inputs via ``parameters_checker()``
        2. **Model Selection** (if ``model_do="select"``): Use Branch & Bound algorithm
        3. **Model Architecture**: Define components and lags via ``architector()``
        4. **Matrix Creation**: Build state-space matrices via ``creator()``
        5. **Parameter Estimation**: Optimize using NLopt via ``estimator()``
        6. **Model Preparation**: Compute fitted values and final states
           via ``preparator()``

        After fitting, the model stores all results as attributes:

        - Estimated parameters (``persistence_level_``, ``phi_``, etc.)
        - State-space matrices (``states``, ``transition``, ``measurement``)
        - Fitted values and residuals (``fitted``, ``residuals``)
        - Information criteria (``aic``, ``aicc``, ``bic``, ``bicc``)
        - Model specification (``model_name``, ``model_type``)

        Parameters
        ----------
        y : array-like, shape (T,)
            Time series data to fit. Can be:

            - ``numpy.ndarray``: Shape (T,) for univariate time series
            - ``pandas.Series``: Will use index for time information if DatetimeIndex

            Data requirements:

            - **Minimum length**: Depends on model complexity. Rule of thumb:
              T ≥ 3 × (number of parameters)
            - **Multiplicative models**: Require strictly positive data (y > 0)
            - **Missing values**: Currently not supported in Python version
            - **Frequency**: Auto-detected from pandas Series with DatetimeIndex

        X : array-like, shape (T, n_features), optional
            External regressors (explanatory variables). If provided:

            - Must have same length as ``y`` (T observations)
            - Each column is a separate regressor
            - Used only if ``regressors`` parameter was set in ``__init__``
            - Can be adaptive (with persistence) or fixed coefficients

            Example::

                >>> X = np.column_stack([trend, holidays, temperature])
                >>> model.fit(y, X=X)

        Returns
        -------
        self : ADAM
            The fitted model instance with populated attributes:

            **Fitted Parameters** (scikit-learn style with trailing underscores):

            - ``persistence_level_``: α (level smoothing), range [0, 1]
            - ``persistence_trend_``: β (trend smoothing), range [0, α]
            - ``persistence_seasonal_``: γ (seasonal smoothing), list if multiple
            - ``phi_``: Damping parameter, range [0, 1]
            - ``arma_parameters_``: AR/MA coefficients (if ARIMA)
            - ``initial_states_``: Initial state values

            **Model Components** (via properties):

            - ``model_name``: Full model specification string
            - ``coef`` / ``B``: Estimated parameter vector
            - ``states``: State matrix over time
            - ``transition``: Transition matrix
            - ``measurement``: Measurement matrix
            - ``aic``, ``aicc``, ``bic``, ``bicc``: Information criteria

        Raises
        ------
        ValueError
            If data validation fails:

            - y contains NaN values
            - y has insufficient length for model complexity
            - Multiplicative model specified for non-positive data
            - X and y have mismatched lengths

        RuntimeError
            If optimization fails to converge. Check:

            - Model specification is appropriate for data
            - Initial values are reasonable (try different ``initial`` method)
            - Bounds are not too restrictive

        Notes
        -----
        **Optimization Algorithm**:

        Uses NLopt's Nelder-Mead simplex algorithm by default (derivative-free, robust).
        For large models or difficult optimization, consider:

        - Changing ``initial`` method
        - Adjusting bounds (``bounds="usual"`` vs ``bounds="admissible"``)
        - Providing custom starting values via ``nlopt_initial`` parameter

        **Computational Complexity**:

        Fitting time depends on:

        - Sample size T: O(T) per function evaluation
        - Number of parameters k: ~40k function evaluations
        - Model selection: Estimates ~10-15 models for "ZZZ"

        Typical fitting times:

        - T=100, simple ETS: ~0.1 seconds
        - T=1000, ETS with 2 seasonalities: ~1-2 seconds
        - T=1000, automatic selection: ~10-20 seconds

        **Model Selection Details**:

        When ``model_do="select"``:

        1. Branch & Bound explores model space efficiently
        2. Each candidate model is fully estimated
        3. Best model selected based on ``ic`` criterion
        4. Selected model is re-estimated with full optimization

        To see which models were tested::

            >>> model.fit(y)
            >>> print(model.ic_selection)  # Dict of model names -> IC values

        **Holdout Validation**:

        If ``holdout=True`` in ``__init__``:

        - Last ``h`` observations are withheld for validation
        - Model estimated on first T-h observations
        - Can use holdout for out-of-sample accuracy assessment

        **Two-Stage Initialization**:

        When ``initial="two-stage"``:

        1. **Stage 1**: Quick backcasting estimation for initial states
        2. **Stage 2**: Refined optimization starting from stage 1 results

        Often provides better results than pure ``initial="optimal"`` for
        complex models.

        **Memory Usage**:

        - State matrix: O(n_components × (T + max_lag)) floats
        - Modest for typical models (~1-10 MB)
        - Multiple seasonality can increase memory usage

        See Also
        --------
        predict : Generate forecasts from fitted model
        predict_intervals : Generate prediction intervals
        estimator : Underlying estimation function
        parameters_checker : Input validation function

        Examples
        --------
        Basic fitting::

            >>> from smooth import ADAM
            >>> import numpy as np
            >>> y = np.array([
            ...     112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118
            ... ])
            >>> model = ADAM(model="ANN", lags=[1])
            >>> model.fit(y)
            >>> print(f"Alpha: {model.persistence_level_:.3f}")

        With external regressors::

            >>> X = np.random.randn(len(y), 2)
            >>> model = ADAM(model="AAN", regressors="use")
            >>> model.fit(y, X=X)
            >>> print(f"Coefficients: {model.coef}")

        Automatic model selection::

            >>> model = ADAM(model="ZZZ", lags=[12], ic="AICc")
            >>> model.fit(y)
            >>> print(f"Selected: {model.model_name}")
            >>> print(f"AICc: {model.aicc}")

        Access fitted values and residuals::

            >>> model.fit(y)
            >>> fitted = model.fitted
            >>> residuals = model.residuals
            >>> print(f"RMSE: {np.sqrt(np.mean(residuals**2)):.3f}")

        Using pandas Series with datetime index::

            >>> import pandas as pd
            >>> dates = pd.date_range(
            ...     '2020-01-01', periods=len(y), freq='M'
            ... )
            >>> y_series = pd.Series(y, index=dates)
            >>> model.fit(y_series)
            >>> # Frequency auto-detected from index
        """
        # Store fit parameters - these are now set in __init__
        # No need to call _setup_parameters as those parameters are now
        # instance attributes

        # Check parameters and prepare data
        self._check_parameters(y, X)

        # Pure-regression early exit — mirrors R/adam.R:498-629.
        if getattr(self, "_alm_model", None) is not None:
            self._populate_from_alm(y, X)
            return self

        # Fit the occurrence model first if requested (mirrors R adam.R:2160-2195).
        # p_fitted is held constant while the demand-sizes model is estimated.
        self._om_model = None
        if self._occurrence.get("occurrence_model"):
            ot_logical = self._observations["ot_logical"]
            self._observations["obs_zero"] = int(np.sum(~ot_logical))
            self._om_model = self._fit_occurrence_model(y)
            self._occurrence["p_fitted"] = self._om_model.fitted
            self._occurrence["oes_model"] = self._occurrence["occurrence"]

        # Execute model estimation or selection based on model_do
        if self._model_type["model_do"] == "estimate":
            self._execute_estimation()
        elif self._model_type["model_do"] == "select":
            # get the best model
            self._execute_selection()
            # Execute estimation for the selected model
            # Note: estimator() handles two-stage initialization internally,
            # so all models in the pool use consistent initialization
            self._execute_estimation(estimation=True)

        elif self._model_type["model_do"] == "combine":
            # Store original model specification for display (e.g., "CCC")
            self._original_model_spec = self._model_type.get("model", self.model)
            # Run model selection first to get all candidate models
            self._execute_selection()
            # Combine models using IC weights
            self._execute_combination()
            # Execute estimation for the best model (for state-space matrices)
            self._execute_estimation(estimation=True)
        else:
            model_do = self._model_type["model_do"]
            warnings.warn(
                f"Unknown model_do value: {model_do}. "
                "Expected one of: 'estimate', 'select', 'combine'"
            )

        # Prepare final results and format output data
        self._prepare_results()

        # Scale demand-sizes fitted by occurrence probability (mirrors R:1973-1975)
        if self._om_model is not None:
            import pandas as pd

            p = self._occurrence["p_fitted"]
            yf = self._prepared["y_fitted"]
            if isinstance(yf, pd.Series):
                yf = yf * p
            else:
                yf = np.asarray(yf) * np.asarray(p)
            self._prepared["y_fitted"] = yf
            self._prepared["residuals"] = np.asarray(
                self._observations["y_in_sample"]
            ) - np.asarray(yf)
            # Include occurrence model params in the total count
            self._adam_estimated["n_param_estimated"] += self._om_model.nparam

        # Store fitted parameters with trailing underscores
        self._set_fitted_attributes()

        # Auto-forecast if h > 0 (before _config consolidation so holdout is accessible)
        self._auto_predict()

        # Compute elapsed time
        self.time_elapsed_ = time.time() - self._start_time

        # Outlier detection and refitting
        # "ignore" → skip; "use" → include dummies as fixed regressors;
        # "select" → expand dummies with lag/lead and let selection prune them.
        _outliers = self.outliers
        if _outliers in ("use", "select"):
            _level = self.outliers_level
            od = self.outlierdummy(level=_level)
            if len(od.id) > 0:
                D = (
                    self._expand_outlier_dummies(od.outliers)
                    if _outliers == "select"
                    else od.outliers
                )
                h_eff = len(y) - self.nobs
                if h_eff > 0:
                    D = np.vstack([D, np.zeros((h_eff, D.shape[1]))])
                X_new = np.hstack([X, D]) if X is not None else D
                self.outliers = "ignore"
                self.regressors = "select" if _outliers == "select" else "use"
                self.fit(y, X_new)
                self._config["outliers"] = _outliers
                return self  # _config already built by the recursive fit()

        # Consolidate init params into _config and remove individual attributes
        self._config = {
            "lags": self.lags,
            "ar_order": self.ar_order,
            "i_order": self.i_order,
            "ma_order": self.ma_order,
            "orders": self._init_orders,
            "arima_select": self.arima_select,
            "constant": self.constant,
            "regressors": self.regressors,
            "distribution": self.distribution,
            "loss": self.loss,
            "loss_horizon": self.loss_horizon,
            "outliers": self.outliers,
            "outliers_level": self.outliers_level,
            "ic": self.ic,
            "bounds": self.bounds,
            "occurrence": self.occurrence,
            "persistence": self.persistence,
            "phi": self.phi,
            "initial": self.initial,
            "n_iterations": self.n_iterations,
            "arma": self.arma,
            "reg_lambda": self.reg_lambda,
            "gnorm_shape": self.gnorm_shape,
            "lambda_param": self.lambda_param,
            "fast": self.fast,
            "holdout": self.holdout,
        }
        for key in self._config:
            try:
                delattr(self, key)
            except AttributeError:
                pass

        return self

    def _set_fitted_attributes(self):
        """
        Set fitted parameters as attributes with trailing underscores.

        This follows scikit-learn conventions for fitted attributes.
        """
        # Set persistence parameters (pre-estimation values for provided params)
        if self._persistence:
            if "persistence_level" in self._persistence:
                self.persistence_level_ = self._persistence["persistence_level"]
            if "persistence_trend" in self._persistence:
                self.persistence_trend_ = self._persistence["persistence_trend"]
            if "persistence_seasonal" in self._persistence:
                self.persistence_seasonal_ = self._persistence["persistence_seasonal"]
            if "persistence_xreg" in self._persistence:
                self.persistence_xreg_ = self._persistence["persistence_xreg"]

        # For combined models, preserve original model specification with ETS prefix
        if getattr(self, "_is_combined", False):
            self.model = f"ETS({self._original_model_spec})"
            return

        # Update self.model with the selected/estimated model name
        if self._model_type:
            if hasattr(self, "_best_model") and self._best_model:
                ets_str = self._best_model
            else:
                e = self._model_type.get("error_type", "")
                t = self._model_type.get("trend_type", "")
                s = self._model_type.get("season_type", "")
                damped = self._model_type.get("damped", False)
                if damped and t not in ["N", ""]:
                    t = t + "d"
                ets_str = e + t + s if (e or t or s) else self.model

            model_parts = []
            is_ets = self._model_type.get("ets_model", False)
            has_xreg = self._explanatory.get("xreg_model", False)
            if is_ets:
                ets_prefix = "ETSX" if has_xreg else "ETS"
                model_parts.append(f"{ets_prefix}({ets_str})")

            is_arima = self._model_type.get("arima_model", False)
            if is_arima and self._arima:
                ar_orders = self._arima.get("ar_orders", [0]) or [0]
                i_orders = self._arima.get("i_orders", [0]) or [0]
                ma_orders = self._arima.get("ma_orders", [0]) or [0]
                lags = self._lags_model.get("lags_original", [1]) or [1]
                has_xreg_arima = (
                    self._explanatory.get("xreg_model", False) and not is_ets
                )
                seasonal_have_orders = any(
                    (ar_orders[j] if j < len(ar_orders) else 0) != 0
                    or (i_orders[j] if j < len(i_orders) else 0) != 0
                    or (ma_orders[j] if j < len(ma_orders) else 0) != 0
                    for j, lag in enumerate(lags)
                    if lag > 1
                )
                if all(lag == 1 for lag in lags) or not seasonal_have_orders:
                    prefix = "ARIMAX" if has_xreg_arima else "ARIMA"
                    model_parts.append(
                        f"{prefix}({ar_orders[0]},{i_orders[0]},{ma_orders[0]})"
                    )
                else:
                    prefix = "SARIMAX" if has_xreg_arima else "SARIMA"
                    arima_str = prefix
                    for j, lag in enumerate(lags):
                        p = ar_orders[j] if j < len(ar_orders) else 0
                        d = i_orders[j] if j < len(i_orders) else 0
                        q = ma_orders[j] if j < len(ma_orders) else 0
                        if p == 0 and d == 0 and q == 0:
                            continue
                        arima_str += f"({p},{d},{q})[{lag}]"
                    model_parts.append(arima_str)

            if model_parts:
                self.model = "+".join(model_parts)
            else:
                self.model = ets_str

            if self._constant and self._constant.get("constant_required", False):
                i_orders = (self._arima or {}).get("i_orders") or []
                has_drift = is_ets or any(d != 0 for d in i_orders)
                self.model += " with " + ("drift" if has_drift else "constant")

    # =========================================================================
    # Extraction Properties - R-style accessors implemented as Python properties
    # =========================================================================

    def _check_is_fitted(self):
        """Check if model has been fitted."""
        if not hasattr(self, "_prepared") or self._prepared is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

    # =========================================================================
    # R-Equivalent Properties
    # =========================================================================

    @property
    def states(self) -> NDArray:
        """
        State matrix containing component values over time (R: $states).

        The state matrix stores the evolution of all model components including
        level, trend (if present), seasonal components (if present), and ARIMA
        states (if present). Each column represents a different state component,
        and each row represents a time point.

        Returns
        -------
        NDArray
            2D array of shape (n_states, obs_in_sample + 1). Columns represent
            level, trend (if model has trend), and seasonal components (if model
            has seasonality, one column per lag).

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAA", lags=12)
        >>> model.fit(y)
        >>> states = model.states
        >>> level = states[0, :]  # Level component over time
        """
        self._check_is_fitted()
        return self._prepared["states"]

    @property
    def persistence_vector(self) -> Dict[str, Any]:
        """
        Estimated smoothing parameters (R: $persistence).

        Returns a dictionary containing the smoothing/persistence parameters
        that control how quickly the model adapts to new observations. Higher
        values mean faster adaptation (more weight on recent observations).

        Returns
        -------
        Dict[str, Any]
            Dictionary with keys ``persistence_level`` (alpha, 0-1),
            ``persistence_trend`` (beta, 0-alpha, only if model has trend),
            and ``persistence_seasonal`` (gamma, 0 to 1-alpha, only if model
            has seasonality).

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        See Also
        --------
        phi_ : Damping parameter for trend

        Examples
        --------
        >>> model = ADAM(model="AAA", lags=12)
        >>> model.fit(y)
        >>> alpha = model.persistence_vector['persistence_level']
        >>> gamma = model.persistence_vector['persistence_seasonal']
        """
        self._check_is_fitted()
        return self._prepared.get("persistence", {})

    @property
    def phi_(self) -> Optional[float]:
        """
        Damping parameter for trend component (R: $phi).

        The damping parameter (phi) controls how quickly the trend dampens
        toward zero over the forecast horizon. A value of 1.0 means no damping
        (linear trend), while values less than 1.0 cause the trend to
        gradually flatten.

        Returns
        -------
        Optional[float]
            Damping parameter value between 0 and 1, or None if the model
            does not include a damped trend component.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Notes
        -----
        Uses trailing underscore following scikit-learn convention for fitted
        parameters.

        Examples
        --------
        >>> model = ADAM(model="AAdN")  # Damped trend model
        >>> model.fit(y)
        >>> print(f"Damping: {model.phi_:.3f}")
        """
        self._check_is_fitted()
        if self._model_type.get("damped", False):
            return self._prepared.get("phi", 1.0)
        return None

    @property
    def transition(self) -> NDArray:
        """
        State transition matrix F (R: $transition).

        The transition matrix governs how states evolve from one time period
        to the next in the state-space formulation: v_t = F @ v_{t-1} + g * e_t

        Returns
        -------
        NDArray
            Square matrix of shape (n_states, n_states) defining state
            transitions. Structure depends on model components (ETS, ARIMA).

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        See Also
        --------
        measurement : Measurement matrix W
        states : State values over time

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> F = model.transition
        """
        self._check_is_fitted()
        return self._prepared["mat_f"]

    @property
    def measurement(self) -> NDArray:
        """
        Measurement matrix W (R: $measurement).

        The measurement matrix maps the state vector to the observation
        equation: y_t = W @ v_t + e_t (for additive errors).

        Returns
        -------
        NDArray
            Matrix of shape (obs_in_sample, n_states) that maps states to
            observations. For time-invariant models, all rows are identical.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        See Also
        --------
        transition : State transition matrix F
        states : State values over time

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> W = model.measurement
        """
        self._check_is_fitted()
        return self._prepared["mat_wt"]

    @property
    def initial_value(self) -> Dict[str, Any]:
        """
        Initial state values used for model fitting (R: $initial).

        Contains the starting values for each state component at time t=0,
        which serve as the foundation for the state evolution.

        Returns
        -------
        Dict[str, Any]
            Dictionary with keys ``level`` (initial level value), ``trend``
            (initial trend value, only if model has trend), and ``seasonal``
            (initial seasonal values as array, only if model has seasonality).

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        See Also
        --------
        initial_type : Method used for initialization

        Examples
        --------
        >>> model = ADAM(model="AAA", lags=12)
        >>> model.fit(y)
        >>> init_level = model.initial_value['level']
        """
        self._check_is_fitted()
        return self._prepared.get("initial_value", {})

    @property
    def initial_type(self) -> str:
        """
        Initialization method used for initial states (R: $initialType).

        Returns
        -------
        str
            One of:
            - ``"optimal"``: Initial states optimized during fitting
            - ``"backcasting"``: Initial states estimated via backcasting
            - ``"two-stage"``: Backcast then optimize
            - ``"complete"``: Pure backcasting without optimization
            - ``"provided"``: User-supplied initial values

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        See Also
        --------
        initial_value : The actual initial state values

        Examples
        --------
        >>> model = ADAM(model="AAN", initial="backcasting")
        >>> model.fit(y)
        >>> print(model.initial_type)
        'backcasting'
        """
        self._check_is_fitted()
        return self._initials.get("initial_type", "optimal")

    @property
    def loss_value(self) -> float:
        """
        Optimized loss/cost function value (R: $lossValue).

        The final value of the loss function after optimization. Lower values
        indicate better fit (for most loss functions).

        Returns
        -------
        float
            The minimized loss function value from the optimization process.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        See Also
        --------
        loss : The loss function type used
        loglik : Log-likelihood value

        Examples
        --------
        >>> model = ADAM(model="AAN", loss="MSE")
        >>> model.fit(y)
        >>> print(f"MSE: {model.loss_value:.4f}")
        """
        self._check_is_fitted()
        return self._adam_estimated["CF_value"]

    @property
    def time_elapsed(self) -> float:
        """
        Time taken to fit the model in seconds.

        Returns
        -------
        float
            Elapsed time in seconds from start to end of the fit() call.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="ZXZ", lags=12)
        >>> model.fit(y)
        >>> print(f"Fitting took {model.time_elapsed:.2f} seconds")
        """
        self._check_is_fitted()
        return self.time_elapsed_

    @property
    def data(self) -> NDArray:
        """
        In-sample training data (R: $data).

        Alias for ``actuals`` property. Returns the original time series
        data used for model fitting.

        Returns
        -------
        NDArray
            1D array of in-sample observations.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> training_data = model.data
        """
        return self.actuals

    @property
    def holdout_data(self) -> Optional[NDArray]:
        """
        Holdout validation data (R: $holdout).

        If ``holdout=True`` was specified during fitting, returns the portion
        of data withheld for validation. Otherwise returns None.

        Returns
        -------
        Optional[NDArray]
            1D array of holdout observations, or None if no holdout was used.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAN", holdout=True, h=12)
        >>> model.fit(y)
        >>> if model.holdout_data is not None:
        ...     print(f"Holdout size: {len(model.holdout_data)}")
        """
        self._check_is_fitted()
        if self._general.get("holdout"):
            return np.array(self._observations.get("y_holdout", []))
        return None

    @property
    def b_value(self) -> NDArray:
        """
        Full parameter vector B (R: $B).

        Alias for ``coef`` property. Contains all estimated parameters in a
        single vector, including persistence parameters, initial states,
        ARIMA coefficients, and other model parameters.

        Returns
        -------
        NDArray
            1D array of all model parameters.

        See Also
        --------
        coef : Primary property for parameter vector

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> all_params = model.b_value
        """
        return self.coef

    @property
    def scale(self) -> float:
        """
        Scale/dispersion parameter (R: $scale).

        Alias for ``sigma`` property. The scale parameter represents the
        estimated standard deviation of the residuals for normal distribution,
        or the analogous dispersion parameter for other distributions.

        Returns
        -------
        float
            Estimated scale parameter.

        See Also
        --------
        sigma : Primary property for scale parameter

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> print(f"Scale: {model.scale:.4f}")
        """
        return self.sigma

    @property
    def profile(self) -> Optional[Any]:
        """
        Profile information for seasonal patterns (R: $profile).

        Contains profile data used for forecasting with multiple seasonality.
        The profile captures the seasonal pattern structure.

        Returns
        -------
        Optional[Any]
            Profile table for recent observations, or None if not applicable.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAA", lags=[24, 168])
        >>> model.fit(y)
        >>> seasonal_profile = model.profile
        """
        self._check_is_fitted()
        return self._prepared.get("profiles_recent_table")

    @property
    def n_param(self) -> Any:
        """
        Parameter count information (R: $nParam).

        Provides information about the number of parameters in the model,
        distinguishing between estimated and provided parameters.

        Returns
        -------
        Any
            Parameter count information (structure may vary).

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAA", lags=12)
        >>> model.fit(y)
        >>> print(model.n_param)
        """
        self._check_is_fitted()
        return self._general.get("n_param")

    @property
    def constant_value(self) -> Optional[float]:
        """
        Constant/intercept term value (R: $constant).

        The estimated constant term in the model, if one was included.

        Returns
        -------
        Optional[float]
            Constant term value, or None if no constant was estimated.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAN", constant=True)
        >>> model.fit(y)
        >>> if model.constant_value is not None:
        ...     print(f"Constant: {model.constant_value:.4f}")
        """
        self._check_is_fitted()
        return self._prepared.get("constant")

    @property
    def distribution_(self) -> str:
        """
        Error distribution used for fitting (R: $distribution).

        The probability distribution assumed for the error term in the model.
        Uses trailing underscore to distinguish from the input parameter and
        follow scikit-learn convention for fitted attributes.

        Returns
        -------
        str
            Distribution name, one of:
            - ``"dnorm"``: Normal distribution (default for additive errors)
            - ``"dgamma"``: Gamma distribution (default for multiplicative)
            - ``"dlaplace"``: Laplace distribution
            - ``"dlnorm"``: Log-Normal distribution
            - ``"dinvgauss"``: Inverse Gaussian distribution
            - ``"ds"``: S distribution
            - ``"dgnorm"``: Generalized Normal distribution

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        See Also
        --------
        loss_ : Loss function used for estimation

        Examples
        --------
        >>> model = ADAM(model="AAN", distribution="dlaplace")
        >>> model.fit(y)
        >>> print(model.distribution_)
        'dlaplace'
        """
        self._check_is_fitted()
        return self._general.get("distribution_new", self._general.get("distribution"))

    @property
    def loss_(self) -> str:
        """
        Loss function used for parameter estimation (R: $loss).

        The objective function minimized during model fitting. Uses trailing
        underscore to distinguish from the input parameter and follow
        scikit-learn convention for fitted attributes.

        Returns
        -------
        str
            Loss function name, one of:
            - ``"likelihood"``: Maximum likelihood (default)
            - ``"MSE"``: Mean Squared Error
            - ``"MAE"``: Mean Absolute Error
            - ``"HAM"``: Half Absolute Moment
            - ``"LASSO"``: L1 regularization
            - ``"RIDGE"``: L2 regularization
            - ``"GTMSE"``: Geometric Trace MSE
            - ``"GPL"``: Generalized Predictive Likelihood

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        See Also
        --------
        loss_value : The optimized loss function value
        distribution_ : Error distribution used

        Examples
        --------
        >>> model = ADAM(model="AAN", loss="MAE")
        >>> model.fit(y)
        >>> print(model.loss_)
        'MAE'
        """
        self._check_is_fitted()
        return self._general.get("loss")

    # =========================================================================
    # Combined Model Properties
    # =========================================================================

    @property
    def is_combined(self) -> bool:
        """
        Return True if model is a combination of multiple models.

        Combined models are created using "C" in the model string
        (e.g., "CCC", "CCN", "ACA"). The combination uses IC weights to
        combine fitted values and forecasts from all models in the pool.

        Returns
        -------
        bool
            True if the model is a weighted combination of multiple models,
            False otherwise.

        Examples
        --------
        >>> model = ADAM(model="CCC", lags=[1])
        >>> model.fit(y)
        >>> model.is_combined  # True
        """
        return getattr(self, "_is_combined", False)

    @property
    def ic_weights(self) -> Dict[str, float]:
        """
        Return IC weights for combined models (R: $ICw).

        Akaike weights represent the relative likelihood of each model being
        the best model given the data. They are used to combine forecasts from
        multiple models.

        Returns
        -------
        Dict[str, float]
            Dictionary mapping model names to their IC weights. Weights sum to 1.0.

        Raises
        ------
        ValueError
            If the model has not been fitted or was not fitted with combination.

        Examples
        --------
        >>> model = ADAM(model="CCC", lags=[1])
        >>> model.fit(y)
        >>> weights = model.ic_weights
        >>> print(f"ANN weight: {weights.get('ANN', 0):.3f}")
        """
        self._check_is_fitted()
        if not getattr(self, "_is_combined", False):
            raise ValueError("Model was not fitted with combination. Use model='CCC'.")
        return self._ic_weights

    @property
    def models(self) -> List[Dict]:
        """
        Return list of individual models in the combination (R: $models).

        Each entry contains the model name, its IC weight, and the prepared
        model data for forecasting.

        Returns
        -------
        List[Dict]
            List of dictionaries containing:
            - "name": Model name string (e.g., "ANN")
            - "weight": IC weight for this model
            - "prepared": Prepared model data for forecasting

        Raises
        ------
        ValueError
            If the model has not been fitted or was not fitted with combination.

        Examples
        --------
        >>> model = ADAM(model="CCC", lags=[1])
        >>> model.fit(y)
        >>> for m in model.models:
        ...     print(f"{m['name']}: {m['weight']:.3f}")
        """
        self._check_is_fitted()
        if not getattr(self, "_is_combined", False):
            raise ValueError("Model was not fitted with combination. Use model='CCC'.")
        return self._prepared_models

    # =========================================================================
    # Existing Properties
    # =========================================================================

    @property
    def fitted(self) -> NDArray:
        """
        Return in-sample fitted values.

        For combined models (e.g., model="CCC"), returns IC-weighted combination
        of fitted values from all models in the combination.

        Returns
        -------
        NDArray
            Array of fitted values for the in-sample period.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> fitted_values = model.fitted
        """
        self._check_is_fitted()
        if getattr(self, "_is_combined", False):
            return self._combined_fitted
        return self._prepared["y_fitted"]

    @property
    def actuals(self) -> NDArray:
        """
        Return original in-sample data.

        Returns
        -------
        NDArray
            Array of original observations used for fitting.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> original_data = model.actuals
        """
        self._check_is_fitted()
        return np.array(self._observations["y_in_sample"])

    @property
    def coef(self) -> NDArray:
        """
        Return estimated coefficients (parameter vector B).

        The parameter vector B contains all optimized parameters in order:
        1. ETS persistence parameters (α, β, γ)
        2. Damping parameter (φ)
        3. Initial states
        4. ARIMA parameters (AR, MA coefficients)
        5. Regression coefficients
        6. Constant term
        7. Distribution parameters

        Returns
        -------
        NDArray
            Parameter vector B with all estimated coefficients.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> coefficients = model.coef
        """
        self._check_is_fitted()
        return self._adam_estimated["B"]

    @property
    def residuals(self) -> NDArray:
        """
        Return model residuals (errors from fitting).

        For additive error models, residuals are y_t - fitted_t.
        For multiplicative error models, residuals are y_t / fitted_t - 1.
        For combined models, returns residuals computed from the IC-weighted
        combined fitted values: y_t - combined_fitted_t.

        Returns
        -------
        NDArray
            Array of residuals from the fitted model.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> errors = model.residuals
        >>> rmse = np.sqrt(np.mean(errors**2))
        """
        self._check_is_fitted()
        if getattr(self, "_is_combined", False):
            return self._combined_residuals
        return self._prepared["residuals"]

    def rstandard(self) -> NDArray:
        """
        Return standardised residuals.

        Residuals are scaled by distribution-specific estimates of their
        scale, corrected for degrees of freedom ``df = nobs - nparam``.
        The standardisation formula depends on the fitted distribution:

        - **dnorm**: ``(e - ē) / (σ √(n/df))``
        - **dlaplace**: ``e / σ · n/df``
        - **ds**: ``(e - ē) / (σ · n/df)²``
        - **dgnorm**: ``(e - ē) / (σ^β · n/df)^(1/β)``, β = shape
        - **dlnorm**: ``exp((log(e) + σ²/2 - mean(·)) / (σ √(n/df)))``
        - **dinvgauss / dgamma**: ``e / ē``

        For a correctly specified model the result should be approximately
        distributed as the standardised version of the fitted distribution.
        Mirrors R's ``rstandard.adam()``.

        Returns
        -------
        NDArray
            Standardised residuals, length ``nobs``.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        See Also
        --------
        rstudent : Leave-one-out studentised residuals.
        outlierdummy : Outlier detection based on standardised residuals.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> std_res = model.rstandard()
        >>> # For a well-specified normal model, std_res ≈ N(0, 1)
        >>> abs(std_res.mean()) < 0.1
        True
        """
        self._check_is_fitted()
        obs = self.nobs
        df = obs - self.nparam
        errors = self.residuals.copy().astype(float)
        dist = self.distribution_
        scale = self.sigma

        if dist == "dnorm":
            mean_e = np.mean(errors)
            return (errors - mean_e) / (scale * np.sqrt(obs / df))
        elif dist == "ds":
            mean_e = np.mean(errors)
            return (errors - mean_e) / (scale * obs / df) ** 2
        elif dist == "dgnorm":
            beta = self.gnorm_shape if self.gnorm_shape is not None else 2.0
            mean_e = np.mean(errors)
            return (errors - mean_e) / (scale**beta * obs / df) ** (1.0 / beta)
        elif dist in ("dinvgauss", "dgamma"):
            return errors / np.mean(errors)
        elif dist == "dlnorm":
            log_e = np.log(errors) + scale**2 / 2
            return np.exp((log_e - np.mean(log_e)) / (scale * np.sqrt(obs / df)))
        else:  # dlaplace and other additive distributions
            return errors / scale * obs / df

    def rstudent(self) -> NDArray:
        """
        Return studentised (leave-one-out) residuals.

        Each residual is scaled by the distribution-specific scale estimate
        recomputed *without that observation* (leave-one-out). Compared to
        ``rstandard()``, the result is more sensitive to individual outliers
        because no single point inflates the global scale estimate it is
        judged against.

        The leave-one-out sums are computed in O(n) using identities such as
        ``Σ e[-i]² = Σ e² − e[i]²``, avoiding an explicit loop.
        Degrees of freedom: ``df = nobs - nparam - 1``.

        Standardisation by distribution:

        - **dnorm**: ``(e[i] - ē) / √(Σe[-i]² / df)``
        - **dlaplace**: ``(e[i] - ē) / (Σ|e[-i]| / df)``
        - **ds**: ``(e[i] - ē) / (Σ√|e[-i]| / (2 df))²``
        - **dgnorm**: ``(e[i] - ē) / (Σ|e[-i]|^β · β/df)^(1/β)``
        - **dlnorm**: ``exp(log_e[i] / √(Σlog_e[-i]² / df))``,
          where ``log_e = log(e) - mean(log e) - σ²/2``
        - **dinvgauss / dgamma**: ``e[i] / mean(e[-i])``

        Mirrors R's ``rstudent.adam()``.

        Returns
        -------
        NDArray
            Studentised residuals, length ``nobs``.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        See Also
        --------
        rstandard : Simpler standardised residuals (faster).
        outlierdummy : Outlier detection based on studentised residuals.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> stu_res = model.rstudent()
        >>> # Studentised residuals are slightly more spread than rstandard
        >>> stu_res.std() >= model.rstandard().std()
        True
        """
        self._check_is_fitted()
        obs = self.nobs
        df = obs - self.nparam - 1
        errors = self.residuals.copy().astype(float)
        dist = self.distribution_

        if dist == "dnorm":
            errors -= np.mean(errors)
            total_sq = np.sum(errors**2)
            denom = np.sqrt((total_sq - errors**2) / df)
            return errors / denom

        elif dist == "dlaplace":
            errors -= np.mean(errors)
            total_abs = np.sum(np.abs(errors))
            denom = (total_abs - np.abs(errors)) / df
            return errors / denom

        elif dist == "ds":
            errors -= np.mean(errors)
            total_sqrt_abs = np.sum(np.sqrt(np.abs(errors)))
            denom = ((total_sqrt_abs - np.sqrt(np.abs(errors))) / (2 * df)) ** 2
            return errors / denom

        elif dist == "dgnorm":
            beta = self.gnorm_shape if self.gnorm_shape is not None else 2.0
            errors -= np.mean(errors)
            total_pow = np.sum(np.abs(errors) ** beta)
            denom = ((total_pow - np.abs(errors) ** beta) * (beta / df)) ** (1.0 / beta)
            return errors / denom

        elif dist in ("dinvgauss", "dgamma"):
            total = np.sum(errors)
            mean_loo = (total - errors) / (obs - 1)
            return errors / mean_loo

        elif dist == "dlnorm":
            scale = self.sigma
            log_e = np.log(errors) - np.mean(np.log(errors)) - scale**2 / 2
            total_sq = np.sum(log_e**2)
            denom = np.sqrt((total_sq - log_e**2) / df)
            return np.exp(log_e / denom)

        else:  # default: treat like normal
            errors -= np.mean(errors)
            total_sq = np.sum(errors**2)
            denom = np.sqrt((total_sq - errors**2) / df)
            return errors / denom

    def outlierdummy(
        self,
        level: float = 0.999,
        type: str = "rstandard",  # noqa: A002
    ) -> OutlierDummy:
        """
        Detect outliers and return a matrix of indicator dummy variables.

        Computes standardised residuals (via ``rstandard()`` or ``rstudent()``),
        then derives two-sided quantile bounds ``[q_lo, q_hi]`` for the fitted
        distribution at the given confidence ``level``. Observations whose
        standardised residual falls outside these bounds are labelled outliers.

        The quantile bounds are distribution-specific:

        - **dnorm / dlnorm** (log-space): ``scipy.stats.norm.ppf``
        - **dlaplace**: ``scipy.stats.laplace.ppf``
        - **ds**: ``scipy.stats.gennorm.ppf(..., beta=0.5)``
        - **dgnorm**: ``scipy.stats.gennorm.ppf(..., beta=shape)``
        - **dgamma**: ``scipy.stats.gamma.ppf(..., a=1/σ, scale=σ)``
        - **dinvgauss**: ``scipy.stats.invgauss.ppf`` with df-corrected dispersion

        Mirrors R's ``outlierdummy.adam()`` from the **greybox** package.

        Parameters
        ----------
        level : float, default 0.999
            Two-sided confidence level for outlier detection.
            0.99 flags observations in the outer 1 % of the distribution.
        type : {"rstandard", "rstudent"}, default "rstandard"
            Which standardised residuals to use. ``"rstudent"`` is more
            sensitive to individual outliers in small samples.

        Returns
        -------
        OutlierDummy
            Dataclass with fields:

            ``outliers`` : ndarray of shape (n, m) or None
                Binary dummy matrix, one column per detected outlier.
                ``None`` when no outliers are found.
            ``id`` : ndarray of int
                0-based indices of outlier observations.
            ``statistic`` : ndarray of shape (2,)
                ``[lower, upper]`` quantile bounds used for detection.
            ``level`` : float
                The confidence level used.
            ``type`` : str
                The residual type used (``"rstandard"`` or ``"rstudent"``).

        Raises
        ------
        ValueError
            If ``type`` is not ``"rstandard"`` or ``"rstudent"``.
        ValueError
            If the model has not been fitted yet.

        See Also
        --------
        rstandard : Standardised residuals.
        rstudent : Studentised residuals.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> od = model.outlierdummy(level=0.99)
        >>> od.id          # 0-based indices of detected outliers
        >>> od.statistic   # [lower, upper] bounds, e.g. [-2.576, 2.576]
        >>> if od.outliers is not None:
        ...     # Refit with outlier dummies as exogenous regressors
        ...     model2 = ADAM(model="AAN")
        ...     model2.fit(y, X=od.outliers)
        """
        self._check_is_fitted()
        if type not in ("rstandard", "rstudent"):
            raise ValueError("type must be 'rstandard' or 'rstudent'")

        errors = self.rstandard() if type == "rstandard" else self.rstudent()
        dist = self.distribution_
        p = np.array([(1 - level) / 2, (1 + level) / 2])

        if dist == "dnorm":
            stat = scipy_stats.norm.ppf(p)
        elif dist == "dlaplace":
            stat = scipy_stats.laplace.ppf(p)
        elif dist == "ds":
            stat = scipy_stats.gennorm.ppf(p, beta=0.5)
        elif dist == "dgnorm":
            beta = self.gnorm_shape if self.gnorm_shape is not None else 2.0
            stat = scipy_stats.gennorm.ppf(p, beta=beta)
        elif dist == "dlnorm":
            errors = np.log(errors)
            stat = scipy_stats.norm.ppf(p)
        elif dist == "dgamma":
            scale = self.sigma
            stat = scipy_stats.gamma.ppf(p, a=1.0 / scale, scale=scale)
        elif dist == "dinvgauss":
            nobs, npar = self.nobs, self.nparam
            disp = float(self.sigma) * nobs / (nobs - npar)
            stat = scipy_stats.invgauss.ppf(p, mu=disp, scale=1.0 / disp)
        else:
            stat = scipy_stats.norm.ppf(p)

        outlier_ids = np.where((errors > stat[1]) | (errors < stat[0]))[0]
        n_out = len(outlier_ids)

        if n_out > 0:
            dummy_mat = np.zeros((self.nobs, n_out), dtype=float)
            dummy_mat[outlier_ids, np.arange(n_out)] = 1.0
        else:
            dummy_mat = None

        return OutlierDummy(
            outliers=dummy_mat,
            id=outlier_ids,
            statistic=stat,
            level=level,
            type=type,
        )

    @staticmethod
    def _expand_outlier_dummies(D: NDArray) -> NDArray:
        """
        Expand each outlier dummy column into three columns: lag-1, t, lead+1.

        Used by ``outliers="select"`` to allow the regressor selection
        mechanism to choose which temporal offset carries the outlier effect.

        Parameters
        ----------
        D : NDArray of shape (n, m)
            Binary dummy matrix from ``outlierdummy()``.

        Returns
        -------
        NDArray of shape (n, 3*m)
            Expanded matrix with columns ordered as
            ``[lag_1, t, lead_1, lag_2, t_2, lead_2, ...]``.
        """
        n, m = D.shape
        cols = []
        for j in range(m):
            col = D[:, j]
            cols.append(np.concatenate([[0.0], col[:-1]]))  # lag -1
            cols.append(col.copy())  # t
            cols.append(np.concatenate([col[1:], [0.0]]))  # lead +1
        return np.column_stack(cols)

    @property
    def nobs(self) -> int:
        """
        Return number of observations used for fitting.

        Returns
        -------
        int
            Number of in-sample observations.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> n = model.nobs
        """
        self._check_is_fitted()
        return len(self._observations["y_in_sample"])

    @property
    def nparam(self) -> int:
        """
        Return number of estimated parameters.

        Returns
        -------
        int
            Number of parameters estimated during optimization.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> k = model.nparam
        """
        self._check_is_fitted()
        return self._adam_estimated["n_param_estimated"]

    @property
    def sigma(self) -> float:
        """
        Return scale/standard error estimate.

        This is the estimated scale parameter of the error distribution,
        which equals the standard deviation for normal errors.

        Returns
        -------
        float
            Scale parameter estimate.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> std_error = model.sigma
        """
        self._check_is_fitted()
        return self._prepared["scale"]

    @property
    def loglik(self) -> float:
        """
        Return log-likelihood of the fitted model.

        Returns
        -------
        float
            Log-likelihood value.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> ll = model.loglik
        """
        self._check_is_fitted()
        return self._adam_estimated["log_lik_adam_value"]["value"]

    @property
    def aic(self) -> float:
        """
        Return Akaike Information Criterion.

        Returns
        -------
        float
            AIC value.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> aic_val = model.aic
        """
        self._check_is_fitted()
        from smooth.adam_general.core.utils.ic import AIC

        log_lik = self._adam_estimated["log_lik_adam_value"]
        return AIC(log_lik["value"], log_lik["nobs"], log_lik["df"])

    @property
    def aicc(self) -> float:
        """
        Return corrected Akaike Information Criterion.

        Returns
        -------
        float
            AICc value.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> aicc_val = model.aicc
        """
        self._check_is_fitted()
        from smooth.adam_general.core.utils.ic import AICc

        log_lik = self._adam_estimated["log_lik_adam_value"]
        return AICc(log_lik["value"], log_lik["nobs"], log_lik["df"])

    @property
    def bic(self) -> float:
        """
        Return Bayesian Information Criterion.

        Returns
        -------
        float
            BIC value.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> bic_val = model.bic
        """
        self._check_is_fitted()
        from smooth.adam_general.core.utils.ic import BIC

        log_lik = self._adam_estimated["log_lik_adam_value"]
        return BIC(log_lik["value"], log_lik["nobs"], log_lik["df"])

    @property
    def bicc(self) -> float:
        """
        Return corrected Bayesian Information Criterion.

        Returns
        -------
        float
            BICc value.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> bicc_val = model.bicc
        """
        self._check_is_fitted()
        from smooth.adam_general.core.utils.ic import BICc

        log_lik = self._adam_estimated["log_lik_adam_value"]
        return BICc(log_lik["value"], log_lik["nobs"], log_lik["df"])

    @property
    def error_type(self) -> str:
        """
        Return error type: 'A' (additive) or 'M' (multiplicative).

        Returns
        -------
        str
            'A' for additive errors, 'M' for multiplicative errors.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> err_type = model.error_type  # Returns 'A'
        """
        self._check_is_fitted()
        return self._model_type["error_type"]

    @property
    def model_type(self) -> str:
        """
        Return ETS model type code (e.g., 'AAN', 'AAA', 'MAdM').

        Returns just the three/four-letter ETS code, not the full model name.
        For the full model name including ARIMA orders and X indicator,
        use ``model_name``.

        Returns
        -------
        str
            ETS model code where:
            - First letter: Error type (A/M)
            - Second letter: Trend type (N/A/Ad/M/Md)
            - Third letter: Seasonal type (N/A/M)

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="ZZZ")  # Auto-selection
        >>> model.fit(y)
        >>> selected_type = model.model_type  # e.g., 'AAN'
        """
        self._check_is_fitted()
        return self._model_type.get("model", "")

    @property
    def orders(self) -> Dict[str, List[int]]:
        """
        Return ARIMA orders as dict with 'ar', 'i', 'ma' keys.

        Returns
        -------
        Dict[str, List[int]]
            Dictionary with keys 'ar', 'i', 'ma' containing lists of orders
            for each lag. For pure ETS models, returns [0] for each.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="NNN", ar_order=1, i_order=1, ma_order=1)
        >>> model.fit(y)
        >>> arima_orders = model.orders
        >>> print(arima_orders)  # {'ar': [1], 'i': [1], 'ma': [1]}
        """
        self._check_is_fitted()
        ar = self._arima.get("ar_orders")
        i = self._arima.get("i_orders")
        ma = self._arima.get("ma_orders")
        return {
            "ar": ar if ar is not None else [0],
            "i": i if i is not None else [0],
            "ma": ma if ma is not None else [0],
        }

    @property
    def model_name(self) -> str:
        """
        Return full model name string (R: modelName()).

        Returns the complete model specification including ETS type,
        ARIMA orders, and X indicator for regressors.

        Returns
        -------
        str
            Full model name, e.g., 'ETS(AAN)', 'ETS(AAA)+ARIMA(1,1,1)',
            or 'ETSX(MAN)' when regressors are included.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> name = model.model_name  # 'ETS(AAN)'
        """
        self._check_is_fitted()
        return self.model

    @property
    def lags_used(self) -> List[int]:
        """
        Return the vector of lags used in the model.

        Returns
        -------
        List[int]
            List of lag values (seasonal periods) used in the model.
            For example, [1, 12] for monthly data with annual seasonality.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAA", lags=[1, 12])
        >>> model.fit(y)
        >>> model.lags_used  # [1, 12]
        """
        self._check_is_fitted()
        return list(self._lags_model.get("lags", [1]))

    @property
    def om_model(self):
        """Fitted occurrence model (OM / OMG / AutoOM), or None."""
        self._check_is_fitted()
        return getattr(self, "_om_model", None)

    def rmultistep(self, h: int = 10) -> pd.DataFrame:
        """Return the (T-h) × h matrix of rolling in-sample multistep forecast errors.

        Translates R's rmultistep.adam() from R/rmultistep.R.
        Must be called after fit().

        Parameters
        ----------
        h : int
            Forecast horizon (number of steps ahead). Default 10.

        Returns
        -------
        pd.DataFrame
            Shape (T-h, h) where T is obs_in_sample.
        """
        from smooth.adam_general.core.forecaster._helpers import (
            _compute_multistep_errors,
        )

        self._check_is_fitted()
        gen = {**self._general, "h": h}
        mat_wt = np.asfortranarray(self._prepared["measurement"], dtype=np.float64)
        mat_f = np.asfortranarray(self._prepared["transition"], dtype=np.float64)
        errors = _compute_multistep_errors(
            self._adam_cpp,
            self._prepared,
            self._observations,
            self._lags_model,
            gen,
            mat_wt,
            mat_f,
        )
        return pd.DataFrame(errors, columns=[f"h={i + 1}" for i in range(h)])

    def predict(
        self,
        h: int,
        X: Optional[NDArray] = None,
        interval: Literal[
            "none",
            "prediction",
            "simulated",
            "approximate",
            "semiparametric",
            "nonparametric",
            "empirical",
            "confidence",
            "complete",
        ] = "none",
        level: Optional[Union[float, List[float]]] = 0.95,
        side: Literal["both", "upper", "lower"] = "both",
        cumulative: bool = False,
        nsim: int = 10000,
        occurrence: Optional[NDArray] = None,
        scenarios: bool = False,
    ) -> NDArray:
        """
        Generate forecasts using the fitted ADAM model.

        Matches R's ``forecast.adam()`` interface for interval types and
        additional parameters.

        Parameters
        ----------
        h : int
            Forecast horizon (number of steps to forecast).
        X : Optional[NDArray], default=None
            Exogenous variables for the forecast period.
            Ensure that X covers the entire forecast horizon ``h``.
        interval : str, default="none"
            Type of prediction interval to construct:

            - ``"none"``: No intervals, point forecasts only.
            - ``"prediction"``: Automatically selects ``"simulated"`` or
              ``"approximate"`` depending on the model type.
            - ``"simulated"``: Simulation-based intervals (Monte Carlo).
            - ``"approximate"``: Analytical (parametric) intervals.
            - ``"semiparametric"``: Not yet implemented.
            - ``"nonparametric"``: Not yet implemented.
            - ``"empirical"``: Not yet implemented.
            - ``"confidence"``: Not yet implemented.
            - ``"complete"``: Not yet implemented.
        level : float or list of float, default=0.95
            Confidence level(s) for prediction intervals. Accepts a single
            value (e.g. ``0.95``) or a list for multiple simultaneous levels
            (e.g. ``[0.9, 0.95, 0.99]``). Values above 1 are treated as
            percentages and divided by 100.

            Each level produces a pair of ``lower_X`` / ``upper_X`` columns
            in the output, where X is the corresponding quantile. For example,
            ``level=0.95`` with ``side="both"`` yields columns
            ``"lower_0.025"`` and ``"upper_0.975"``.
        side : str, default="both"
            Which side(s) of the intervals to compute:

            - ``"both"``: Both lower and upper bounds (default).
            - ``"upper"``: Upper bound only. Column named ``"upper_<level>"``.
            - ``"lower"``: Lower bound only. Column named ``"lower_<1-level>"``.
        cumulative : bool, default=False
            If True, return cumulative (summed) forecasts over the horizon.
        nsim : int, default=10000
            Number of simulations for simulation-based intervals.
        occurrence : Optional[NDArray], default=None
            External occurrence probabilities for the forecast period.
            Overrides the fitted model's occurrence for forecasting.
        scenarios : bool, default=False
            If True and ``interval="simulated"``, store the raw simulation
            matrix in ``self._general["_scenarios_matrix"]``.

        Returns
        -------
        ForecastResult
            Structured result with ``.mean`` (pd.Series), ``.lower`` and
            ``.upper`` (pd.DataFrame or None), ``.level``, ``.side``, and
            ``.interval`` attributes.  Use ``.to_dataframe()`` for a flat
            pd.DataFrame.

        Raises
        ------
        ValueError
            If the model has not been fitted yet or ``h`` is not set.
        """
        # Set forecast horizon
        if h is not None:
            self.h = h
            self._general["h"] = self.h
        else:
            if self._general["h"] is None:
                raise ValueError("Forecast horizon is not set.")

        self._general["interval"] = interval
        self._general["nsim"] = nsim
        self._general["cumulative"] = cumulative
        self._general["scenarios"] = scenarios

        if occurrence is not None:
            self._occurrence["occurrence"] = occurrence

        # Store new_xreg for forecast period (used in _generate_point_forecasts)
        if X is not None and self._explanatory.get("xreg_model"):
            new_xreg = np.asarray(X)
            if new_xreg.dtype.names is not None:
                new_xreg = np.column_stack(
                    [new_xreg[f].astype(float) for f in new_xreg.dtype.names]
                )
            else:
                new_xreg = new_xreg.astype(float)
            if new_xreg.ndim == 1:
                new_xreg = new_xreg.reshape(-1, 1)
            self._explanatory["new_xreg"] = new_xreg
        else:
            self._explanatory.pop("new_xreg", None)

        # Validate prediction inputs and prepare data for forecasting
        self._validate_prediction_inputs()
        self._prepare_prediction_data()

        # Pre-compute p_forecast so forecaster can use it for both point forecasts
        # and interval adjustment (R: forecast.adam lines 6134-6172, 6204-6210).
        p_forecast_arr = None
        if getattr(self, "_om_model", None) is not None:
            occ_fc = self._om_model.predict(h=h)
            p_forecast_arr = np.asarray(occ_fc.mean.values, dtype=float)
            # Store in occurrence_dict so _process_occurrence_forecast can use it
            self._occurrence["p_forecast"] = p_forecast_arr

        # Execute the prediction (forecaster handles p scaling + interval adjustment)
        predictions = self._execute_prediction(
            interval=interval,
            level=level,
            side=side,
        )

        # Post-interval NA patch (R/adam.R ~6742-6751)
        if p_forecast_arr is not None:
            if predictions.upper is not None:
                upper_vals = predictions.upper.values
                mask = np.isnan(upper_vals)
                if np.any(mask):
                    fill = (predictions.mean.values / p_forecast_arr).reshape(-1, 1)
                    predictions.upper = pd.DataFrame(
                        np.where(mask, fill, upper_vals),
                        index=predictions.upper.index,
                        columns=predictions.upper.columns,
                    )
            if predictions.lower is not None:
                lower_vals = predictions.lower.values
                if np.any(np.isnan(lower_vals)):
                    predictions.lower = pd.DataFrame(
                        np.where(np.isnan(lower_vals), 0.0, lower_vals),
                        index=predictions.lower.index,
                        columns=predictions.lower.columns,
                    )

        # Recompute accuracy measures against holdout if available
        if self._general.get("holdout", False):
            y_holdout = self._observations.get("y_holdout")
            y_in_sample = self._observations.get("y_in_sample")
            if y_holdout is not None and len(y_holdout) > 0:
                from smooth.adam_general.core.utils.printing import (
                    _compute_forecast_errors,
                )

                fc_values = np.asarray(predictions.mean, dtype=float).ravel()
                y_holdout_arr = np.asarray(y_holdout, dtype=float).ravel()
                n = min(len(fc_values), len(y_holdout_arr))
                period = (
                    max(self._lags_model.get("lags", [1])) if self._lags_model else 1
                )
                self.accuracy = _compute_forecast_errors(
                    y_holdout_arr[:n],
                    fc_values[:n],
                    np.asarray(y_in_sample, dtype=float),
                    period,
                )

        return predictions

    def predict_intervals(
        self,
        h: int,
        X: Optional[NDArray] = None,
        levels: List[float] = [0.8, 0.95],
        side: Literal["both", "upper", "lower"] = "both",
        nsim: int = 10000,
    ):
        """
        Generate prediction intervals using the fitted ADAM model.

        Convenience wrapper around ``predict()`` that defaults to
        ``interval="prediction"`` and accepts multiple confidence levels.

        Parameters
        ----------
        h : int
            Forecast horizon (number of steps to forecast).
        X : Optional[NDArray], default=None
            Exogenous variables for the forecast period.
        levels : List[float], default=[0.8, 0.95]
            Confidence levels for prediction intervals. Each level produces
            a pair of lower/upper columns in the output DataFrame. For
            example, ``levels=[0.8, 0.95]`` with ``side="both"`` yields
            columns ``"lower_0.1"``, ``"lower_0.025"``, ``"upper_0.9"``,
            ``"upper_0.975"``.
        side : Literal["both", "upper", "lower"], default="both"
            Which side(s) of the intervals to return.
        nsim : int, default=10000
            Number of simulations for simulation-based intervals.

        Returns
        -------
        ForecastResult
            Structured result with ``.mean``, ``.lower``, ``.upper`` attributes.
        """
        return self.predict(
            h=h,
            X=X,
            interval="prediction",
            level=levels,
            side=side,
            nsim=nsim,
        )

    def _check_parameters(self, ts, X=None):
        """
        Check parameters using parameters_checker and store results.

        Parameters
        ----------
        ts : NDArray or pd.Series
            Time series data (numpy array or pandas Series).
        X : NDArray or pd.DataFrame, optional
            External regressors, shape (len(ts), n_features).
        """
        # Build orders dict for parameters_checker.
        # ar_order/i_order/ma_order take priority; fall back to orders dict.
        if any(param != 0 for param in [self.ar_order, self.i_order, self.ma_order]):
            orders = {
                "ar": self.ar_order,
                "i": self.i_order,
                "ma": self.ma_order,
                "select": self.arima_select,
            }
        elif self._init_orders is not None:
            orders = {
                **self._init_orders,
                "select": self._init_orders.get("select", self.arima_select),
            }
        else:
            orders = None

        result = parameters_checker(
            ts,
            model=self.model,
            lags=self.lags,
            orders=orders,
            constant=self.constant,
            outliers=self.outliers,
            level=self.outliers_level,
            persistence=self.persistence,
            phi=self.phi,
            initial=self.initial,
            n_iterations=self.n_iterations,
            distribution=self.distribution,
            loss=self.loss,
            h=self.h,
            holdout=self.holdout,
            occurrence=self.occurrence,
            ic=self.ic,
            bounds=self.bounds,
            silent=(self.verbose == 0),
            fast=self.fast,
            lambda_param=self.lambda_param,
            X=X,
            regressors=self.regressors,
            arma=self.arma,
        )
        if not isinstance(result, tuple):
            self._alm_model = result
            return
        self._alm_model = None
        (
            self._general,
            self._observations,
            self._persistence,
            self._initials,
            self._arima,
            self._constant,
            self._model_type,
            self._components,
            self._lags_model,
            self._occurrence,
            self._phi_internal,
            self._explanatory,
            self._params_info,
        ) = result

    def _populate_from_alm(self, y, X):
        """Populate model attributes from the greybox.ALM early-exit object.

        Mirrors R/adam.R:498-629 where an alm() result is wrapped into the
        adam object when model="NNN" with regressors but no ETS/ARIMA.
        """
        alm = self._alm_model
        n = int(alm.nobs)
        y_in_sample = np.asarray(y[:n], dtype=float)
        fitted = np.asarray(alm.fitted_values_, dtype=float)

        self._observations = {
            "y_in_sample": y_in_sample,
            "ot": (y_in_sample != 0).astype(float),
        }
        self._model_type = {
            "model": "NNN",
            "ets_model": False,
            "arima_model": False,
            "xreg_model": True,
        }
        self._arima = {"ar_orders": [0], "i_orders": [0], "ma_orders": [0]}
        self._explanatory = {"xreg_model": True}
        self._general = {
            "loss": "likelihood",
            "h": getattr(self, "h", 0),
            "holdout": getattr(self, "holdout", False),
        }
        self._prepared = {
            "y_fitted": fitted,
            "residuals": y_in_sample - fitted,
        }
        self._adam_estimated = {
            "B": np.asarray(alm.coefficients),
            "n_param_estimated": int(alm.nparam),
            "log_lik_adam_value": {
                "value": float(alm.loglik),
                "nobs": n,
                "df": int(alm.nparam),
            },
        }
        self.model = "Regression"
        self.time_elapsed_ = time.time() - self._start_time

    def _handle_lasso_ridge_special_case(self):
        """
        Handle special case for LASSO/RIDGE with lambda=1.

        Sets appropriate parameter values. This is a special case where we use
        MSE to estimate initials only and disable other parameter estimation.
        """
        if self._general["loss"] in ["LASSO", "RIDGE"] and self._general["lambda"] == 1:
            if self._model_type["ets_model"]:
                # Pre-set ETS parameters
                self._persistence["persistence_estimate"] = False
                self._persistence["persistence_level_estimate"] = False
                self._persistence["persistence_trend_estimate"] = False
                self._persistence["persistence_seasonal_estimate"] = [False]
                self._persistence["persistence_level"] = 0
                self._persistence["persistence_trend"] = 0
                self._persistence["persistence_seasonal"] = [0]
                # Phi
                self._phi_internal["phi_estimate"] = False
                self._phi_internal["phi"] = 1

            if self._model_type["xreg_model"]:
                # ETSX parameters
                self._persistence["persistence_xreg_estimate"] = False
                self._persistence["persistence_xreg"] = 0

            if self._model_type["arima_model"]:
                # Pre-set ARMA parameters
                self._arima["ar_estimate"] = [False]
                self._arima["ma_estimate"] = [False]
                self._preset_arima_parameters()

            self._general["lambda"] = 0

    def _fit_occurrence_model(self, y):
        """Fit an occurrence model on ``y`` and return it.

        Mirrors R/adam.R:2161-2166 (``omModel <- om(...)``).
        Occurrence type comes from ``self._occurrence["occurrence"]``.
        """
        from smooth.adam_general.core.auto_om import AutoOM
        from smooth.adam_general.core.om import OM

        occ = self._occurrence["occurrence"]
        lags = list(self._lags_model.get("lags", [1]))
        adam_model = self._model_type.get("model", "MNN")
        common = dict(
            lags=lags,
            h=self._general.get("h", 0),
            holdout=self._general.get("holdout", False),
            ic=self._general.get("ic", "AICc"),
            bounds=self._general.get("bounds", "usual"),
            initial=self._initials.get("initial_type", "backcasting"),
        )
        if occ == "auto":
            m = AutoOM(model=adam_model, **common)
        elif occ == "general":
            from smooth.adam_general.core.omg import OMG

            m = OMG(**common)
        else:
            m = OM(model=adam_model, occurrence=occ, **common)
        m.fit(y)
        return m

    def _preset_arima_parameters(self):
        """Set up ARIMA parameters for special cases where estimation is disabled."""
        arma_parameters = []
        for i, lag in enumerate(self._lags_model["lags"]):
            if self._arima["ar_orders"][i] > 0:
                arma_parameters.extend([1] * self._arima["ar_orders"][i])
            if self._arima["ma_orders"][i] > 0:
                arma_parameters.extend([0] * self._arima["ma_orders"][i])
        self._arima["arma_parameters"] = arma_parameters

    def _execute_estimation(self, estimation=True):
        """
        Execute model estimation when model_do is 'estimate'.

        This handles parameter estimation and model creation.
        """
        # Handle special case for LASSO/RIDGE with lambda=1
        self._handle_lasso_ridge_special_case()

        # Estimate the model
        # Note: estimator() handles two-stage initialization internally
        if estimation:
            # gnorm shape handling: estimate when not given (mirrors R)
            dist = self._general.get("distribution_new") or self._general.get(
                "distribution", "dnorm"
            )
            other_parameter_estimate = dist == "dgnorm" and self.gnorm_shape is None
            if dist != "dgnorm":
                other_value = None
            elif self.gnorm_shape is None:
                other_value = 2.0
            else:
                other_value = float(self.gnorm_shape)

            nlopt_params = self.nlopt_kargs if self.nlopt_kargs else {}
            self._adam_estimated = estimator(
                general_dict=self._general,
                model_type_dict=self._model_type,
                lags_dict=self._lags_model,
                observations_dict=self._observations,
                arima_dict=self._arima,
                constant_dict=self._constant,
                explanatory_dict=self._explanatory,
                profiles_recent_table=self.profiles_recent_table,
                profiles_recent_provided=self.profiles_recent_provided,
                persistence_dict=self._persistence,
                initials_dict=self._initials,
                occurrence_dict=self._occurrence,
                phi_dict=self._phi_internal,
                components_dict=self._components,
                multisteps=self._general.get("multisteps", False),
                smoother=self.smoother,
                adam_ets=(self.ets == "adam"),
                other=other_value,
                other_parameter_estimate=other_parameter_estimate,
                **nlopt_params,
            )
            # Extract adam_cpp from estimation results
            self._adam_cpp = self._adam_estimated["adam_cpp"]

            # Store back estimated gnorm shape
            if other_parameter_estimate and "B" in self._adam_estimated:
                self.gnorm_shape = float(abs(self._adam_estimated["B"][-1]))

        # Build the model structure - architector() returns 6 values including
        # adam_cpp, but we already have adam_cpp from estimation
        (
            self._model_type,
            self._components,
            self._lags_model,
            self._observations,
            self._profile,
            _,  # adam_cpp - already stored from estimation result
        ) = architector(
            model_type_dict=self._model_type,
            lags_dict=self._lags_model,
            observations_dict=self._observations,
            arima_checked=self._arima,
            constants_checked=self._constant,
            explanatory_checked=self._explanatory,
            profiles_recent_table=self.profiles_recent_table,
            profiles_recent_provided=self.profiles_recent_provided,
            adam_ets=(self.ets == "adam"),
        )
        # print(self._components)
        # Create the model matrices
        self._adam_created = creator(
            model_type_dict=self._model_type,
            lags_dict=self._lags_model,
            profiles_dict=self._profile,
            observations_dict=self._observations,
            persistence_checked=self._persistence,
            initials_checked=self._initials,
            arima_checked=self._arima,
            constants_checked=self._constant,
            phi_dict=self._phi_internal,
            components_dict=self._components,
            explanatory_checked=self._explanatory,
            smoother=self.smoother,
        )

        # Calculate information criterion
        if estimation:
            self._ic_selection = ic_function(
                self._general["ic"], self._adam_estimated["log_lik_adam_value"]
            )

        # Update parameters number
        self._update_parameters_number(self._adam_estimated["n_param_estimated"])

    def _update_parameters_number(self, n_param_estimated):
        """
        Update the parameters number in the general dictionary.

        Parameters
        ----------
        n_param_estimated : int
            Number of estimated parameters from optimization
        """
        # Store number of estimated parameters
        self._n_param_estimated = n_param_estimated

        # Skip updating n_param for combined models
        # (already set in _execute_combination)
        if getattr(self, "_is_combined", False):
            # Legacy format for backward compatibility
            if "parameters_number" not in self._general:
                self._general["parameters_number"] = self._params_info.get(
                    "parameters_number", [[0], [0]]
                )
            return

        # Update the n_param table
        if "n_param" in self._general:
            n_param = self._general["n_param"]
            # n_param_estimated = len(B) = internal_in_B + xreg_in_B
            # Subtract xreg (correctly set by build_n_param_table) to get internal
            n_param.estimated["internal"] = (
                n_param_estimated - n_param.estimated["xreg"]
            )

            # Handle likelihood loss case - scale parameter is estimated
            if self._general["loss"] == "likelihood":
                n_param.estimated["scale"] = 1
            else:
                n_param.estimated["scale"] = 0

            # Update totals
            n_param.update_totals()

            # Store reference for easy access
            self._n_param = n_param

        # Legacy format for backward compatibility
        if "parameters_number" not in self._general:
            self._general["parameters_number"] = self._params_info.get(
                "parameters_number", [[0], [0]]
            )
        self._general["parameters_number"][0][0] = n_param_estimated

        # Handle likelihood loss case in legacy format
        if self._general["loss"] == "likelihood":
            if len(self._general["parameters_number"][0]) <= 3:
                self._general["parameters_number"][0].append(1)
            else:
                self._general["parameters_number"][0][3] = 1

        # Calculate row sums in legacy format
        if len(self._general["parameters_number"][0]) <= 4:
            self._general["parameters_number"][0].append(
                sum(self._general["parameters_number"][0][0:4])
            )
            self._general["parameters_number"][1].append(
                sum(self._general["parameters_number"][1][0:4])
            )
        else:
            self._general["parameters_number"][0][4] = sum(
                self._general["parameters_number"][0][0:4]
            )
            self._general["parameters_number"][1][4] = sum(
                self._general["parameters_number"][1][0:4]
            )

    def _execute_selection(self):
        """
        Execute model selection when model_do is 'select'.

        This handles model selection and creation of the selected model.
        """
        # Run model selection
        self._adam_selected = selector(
            model_type_dict=self._model_type,
            phi_dict=self._phi_internal,
            general_dict=self._general,
            lags_dict=self._lags_model,
            observations_dict=self._observations,
            arima_dict=self._arima,
            constant_dict=self._constant,
            explanatory_dict=self._explanatory,
            occurrence_dict=self._occurrence,
            components_dict=self._components,
            profiles_recent_table=self.profiles_recent_table,
            profiles_recent_provided=self.profiles_recent_provided,
            persistence_results=self._persistence,
            initials_results=self._initials,
            criterion=self._general["ic"],
            silent=self.verbose == 0,
            nlopt_kargs=self.nlopt_kargs,
            smoother=self.smoother,
        )
        # print(self._adam_selected)
        # print(self._adam_selected["ic_selection"])

        # Updates parametes with the selected model and updates adam_estimated
        self.select_best_model()

        # print(self._adam_selected["ic_selection"])
        # Process each selected model
        # The following commented-out loop and its associated helper method
        # calls (_update_model_from_selection, _create_matrices_for_selected_model,
        # _update_parameters_for_selected_model) appear to be placeholders
        # or remnants of a "combine" functionality that is not fully implemented
        # yet, as indicated by the NotImplementedError in the fit method for
        # model_do="combine". These will be kept for now as they might be
        # relevant for future development.
        # for i, result in enumerate(self._adam_selected["results"]):
        #     # Update model parameters with the selected model
        #     self._update_model_from_selection(i, result)
        #
        #     # Create matrices for this model
        #     self._create_matrices_for_selected_model(i)
        #
        #     # Update parameters number for this model
        #     self._update_parameters_for_selected_model(i, result)

    def select_best_model(self):
        """
        Select the best model based on information criteria and update model parameters.
        """
        self._ic_selection = self._adam_selected["ic_selection"]
        results = self._adam_selected["results"]
        # Find best model
        self._best_model = min(self._ic_selection.items(), key=lambda x: x[1])[0]
        best_id = next(
            i for i, result in enumerate(results) if result["model"] == self._best_model
        )
        # Update dictionaries with best model results
        self._model_type = results[best_id]["model_type_dict"]
        self._phi_internal = results[best_id]["phi_dict"]
        self._adam_estimated = results[best_id]["adam_estimated"]
        self._adam_cpp = self._adam_estimated["adam_cpp"]

    def _execute_combination(self):
        """
        Execute model combination using IC weights.

        Combines fitted values from multiple models using Akaike weights derived from
        information criterion values. Each model's contribution is proportional to
        its relative likelihood of being the best model.

        The combined fitted values are IC-weighted averages across all models.
        """
        import copy

        # Get IC weights from selection results
        self._ic_weights = calculate_ic_weights(self._ic_selection)
        results = self._adam_selected["results"]

        # Compute filtered weights (>= 0.01) for fitted values calculation only
        # Full model set is stored; filtering happens at predict-time
        filtered_weights = {k: v for k, v in self._ic_weights.items() if v >= 0.01}
        total_filtered = sum(filtered_weights.values())
        if total_filtered > 0:
            filtered_weights = {
                k: v / total_filtered for k, v in filtered_weights.items()
            }

        # Initialize combined fitted values
        obs_in_sample = self._observations["obs_in_sample"]
        y_fitted_combined = np.zeros(obs_in_sample)
        n_param_weighted = 0.0

        # Store ALL models for later forecasting (filtering happens at predict-time)
        self._prepared_models = []

        for result in results:
            model_name = result["model"]
            original_weight = self._ic_weights.get(model_name, 0)
            filtered_weight = filtered_weights.get(model_name, 0)

            # Make copies of dicts that get modified by architector
            lags_dict_copy = copy.deepcopy(self._lags_model)
            observations_dict_copy = copy.deepcopy(self._observations)
            model_type_dict = result["model_type_dict"].copy()
            phi_dict = result["phi_dict"].copy()

            # Call architector to get components for this model
            (
                model_type_dict,
                components_dict,
                lags_dict_copy,
                observations_dict_copy,
                profile_dict,
                _,
            ) = architector(
                model_type_dict=model_type_dict,
                lags_dict=lags_dict_copy,
                observations_dict=observations_dict_copy,
                arima_checked=self._arima,
                constants_checked=self._constant,
                explanatory_checked=self._explanatory,
                profiles_recent_table=self.profiles_recent_table,
                profiles_recent_provided=self.profiles_recent_provided,
                adam_ets=(self.ets == "adam"),
            )

            # Call creator to build matrices
            adam_created = creator(
                model_type_dict=model_type_dict,
                lags_dict=lags_dict_copy,
                profiles_dict=profile_dict,
                observations_dict=observations_dict_copy,
                persistence_checked=self._persistence,
                initials_checked=self._initials,
                arima_checked=self._arima,
                constants_checked=self._constant,
                phi_dict=phi_dict,
                components_dict=components_dict,
                explanatory_checked=self._explanatory,
                smoother=self.smoother,
            )

            # Make copy of general_dict for preparator
            general_dict_copy = copy.deepcopy(self._general)

            # Ensure distribution_new is set (needed by preparator/scaler)
            if "distribution_new" not in general_dict_copy:
                if general_dict_copy.get("distribution") == "default":
                    if general_dict_copy.get("loss") == "likelihood":
                        if model_type_dict.get("error_type") == "M":
                            general_dict_copy["distribution_new"] = "dgamma"
                        else:
                            general_dict_copy["distribution_new"] = "dnorm"
                    else:
                        general_dict_copy["distribution_new"] = "dnorm"
                else:
                    general_dict_copy["distribution_new"] = general_dict_copy.get(
                        "distribution", "dnorm"
                    )

            # Call preparator to get fitted values
            prepared = preparator(
                model_type_dict=model_type_dict,
                components_dict=components_dict,
                lags_dict=lags_dict_copy,
                matrices_dict=adam_created,
                persistence_checked=self._persistence,
                initials_checked=self._initials,
                arima_checked=self._arima,
                explanatory_checked=self._explanatory,
                phi_dict=phi_dict,
                constants_checked=self._constant,
                observations_dict=observations_dict_copy,
                occurrence_dict=self._occurrence,
                general_dict=general_dict_copy,
                profiles_dict=profile_dict,
                adam_estimated=result["adam_estimated"],
                adam_cpp=result["adam_estimated"]["adam_cpp"],
            )

            # Get fitted values and handle NaN
            fitted_values = np.asarray(prepared["y_fitted"]).flatten()
            fitted_values = np.nan_to_num(fitted_values, nan=0.0)

            # Add IC-weighted contribution (using filtered weight for fitted values)
            y_fitted_combined += fitted_values * filtered_weight
            n_param_weighted += (
                result["adam_estimated"]["n_param_estimated"] * filtered_weight
            )

            # Store for later forecasting
            # (using ORIGINAL weight - filtering at predict-time)
            self._prepared_models.append(
                {
                    "name": model_name,
                    "weight": original_weight,
                    "result": result,
                    "model_type_dict": model_type_dict,
                    "components_dict": components_dict,
                    "lags_dict": lags_dict_copy,
                    "observations_dict": observations_dict_copy,
                    "profile_dict": profile_dict,
                    "phi_dict": phi_dict,
                    "adam_created": adam_created,
                    "prepared": prepared,
                    "explanatory_dict": self._explanatory,
                    "constants_dict": self._constant,
                    "n_param_estimated": result["adam_estimated"]["n_param_estimated"],
                }
            )

        # Store combined results
        self._combined_fitted = y_fitted_combined
        self._combined_residuals = (
            np.array(self._observations["y_in_sample"]) - y_fitted_combined
        )

        # Create NParam with weighted average internal params + scale
        n_param = NParam()
        n_param.estimated["internal"] = n_param_weighted
        if self._general.get("loss") == "likelihood":
            n_param.estimated["scale"] = 1
        n_param.update_totals()
        self._n_param = n_param
        self._general["n_param"] = n_param
        self._n_param_combined = n_param.estimated["all"]  # Backward compat

        # Mark as combined model
        self._is_combined = True

        # Use the best model as the primary model for state-space matrices
        self.select_best_model()

    def _update_model_from_selection(self, index, result):
        """
        Update model parameters with the selected model parameters.

        Parameters
        ----------
        index : int
            Index of the selected model in the results list from model selection.
        result : dict
            The dictionary containing all parameters and estimation results
            for the selected model.
        """
        # Update global dictionaries with the selected model info
        self._general.update(result["general"])
        self._model_type.update(result["model_type_dict"])
        self._arima.update(result["arima_dict"])
        self._constant.update(result["constant_dict"])
        self._persistence.update(result["persistence_dict"])
        self._initials.update(result["initials_dict"])
        self._phi_internal.update(result["phi_dict"])
        self._components.update(result["components_dict"])
        self._lags_model.update(result["lags_dict"])
        self._observations.update(result["observations_dict"])
        self._profile = result.get(
            "profile_dict",
            {
                "profiles_recent_provided": self.profiles_recent_provided,
                "profiles_recent_table": self.profiles_recent_table,
            },
        )

        # Store the estimated model and adam_cpp
        self._adam_estimated = result["adam_estimated"]
        self._adam_cpp = self._adam_estimated["adam_cpp"]

    def _create_matrices_for_selected_model(self, index):
        """
        Create matrices for a selected model. This is typically used when iterating
        through models in a selection process, particularly for a "combine" feature.

        Parameters
        ----------
        index : int
            Index of the selected model.
        """
        # Create matrices for this model
        self._adam_created = creator(
            model_type_dict=self._model_type,
            lags_dict=self._lags_model,
            profiles_dict=self._profile,
            observations_dict=self._observations,
            persistence_checked=self._persistence,
            initials_checked=self._initials,
            arima_checked=self._arima,
            constants_checked=self._constant,
            phi_dict=self._phi_internal,
            components_dict=self._components,
            explanatory_checked=self._explanatory,
            smoother=self.smoother,
        )

        # Store created matrices
        self._adam_selected["results"][index]["adam_created"] = self._adam_created

    def _update_parameters_for_selected_model(self, index, result):
        """
        Update parameters number for a selected model.

        This is typically used when iterating through models in a selection
        process, particularly for a "combine" feature.

        Parameters
        ----------
        index : int
            Index of the selected model.
        result : dict
            Selected model result containing `adam_estimated` which has
            `n_param_estimated`.
        """
        # Update parameters number
        n_param_estimated = result["adam_estimated"]["n_param_estimated"]

        # Update n_param table if available
        if "n_param" in self._general and self._general["n_param"] is not None:
            n_param = self._general["n_param"]
            n_param.estimated["internal"] = n_param_estimated

            if self._general["loss"] == "likelihood":
                n_param.estimated["scale"] = 1
            else:
                n_param.estimated["scale"] = 0

            n_param.update_totals()
            self._n_param = n_param

        # Legacy format
        self._general["parameters_number"] = self._params_info.get(
            "parameters_number", [[0], [0]]
        )
        self._general["parameters_number"][0][0] = n_param_estimated

        # Handle likelihood loss case
        if self._general["loss"] == "likelihood":
            if len(self._general["parameters_number"][0]) <= 3:
                self._general["parameters_number"][0].append(1)
            else:
                self._general["parameters_number"][0][3] = 1

        # Calculate row sums
        if len(self._general["parameters_number"][0]) <= 4:
            self._general["parameters_number"][0].append(
                sum(self._general["parameters_number"][0][0:4])
            )
            self._general["parameters_number"][1].append(
                sum(self._general["parameters_number"][1][0:4])
            )
        else:
            self._general["parameters_number"][0][4] = sum(
                self._general["parameters_number"][0][0:4]
            )
            self._general["parameters_number"][1][4] = sum(
                self._general["parameters_number"][1][0:4]
            )

        # Store parameters number
        self._adam_selected["results"][index]["parameters_number"] = self._general[
            "parameters_number"
        ]

    def _prepare_results(self):
        """
        Prepare final results and format output data.

        This transforms data into appropriate formats and handles distribution
        selection.
        """
        # Transform data into appropriate classes
        self._format_time_series_data()

        # Handle distribution selection
        self._select_distribution()

        # Compute fitted values and residuals by calling preparator
        self._compute_fitted_values()

    def _format_time_series_data(self):
        """
        Format time series data into pandas Series with appropriate indexes.
        """
        if isinstance(self._observations["y_in_sample"], np.ndarray):
            # Check if frequency is a valid pandas frequency (not just "1" string)
            freq = self._observations.get("frequency", "1")
            try:
                # Try to use date_range if frequency looks valid
                if freq != "1" and isinstance(
                    self._observations.get("y_start"), (pd.Timestamp, str)
                ):
                    self._y_in_sample = pd.Series(
                        self._observations["y_in_sample"],
                        index=pd.date_range(
                            start=self._observations["y_start"],
                            periods=len(self._observations["y_in_sample"]),
                            freq=freq,
                        ),
                    )
                else:
                    # Use simple range index for non-datetime data
                    self._y_in_sample = pd.Series(self._observations["y_in_sample"])
            except (ValueError, TypeError):
                # Fallback to simple range index if date_range fails
                self._y_in_sample = pd.Series(self._observations["y_in_sample"])

            if self._general["holdout"]:
                try:
                    if freq != "1" and isinstance(
                        self._observations.get("y_forecast_start"),
                        (pd.Timestamp, str),
                    ):
                        self._y_holdout = pd.Series(
                            self._observations["y_holdout"],
                            index=pd.date_range(
                                start=self._observations["y_forecast_start"],
                                periods=len(self._observations["y_holdout"]),
                                freq=freq,
                            ),
                        )
                    else:
                        self._y_holdout = pd.Series(self._observations["y_holdout"])
                except (ValueError, TypeError):
                    self._y_holdout = pd.Series(self._observations["y_holdout"])
        else:
            self._y_in_sample = self._observations["y_in_sample"].copy()
            if self._general["holdout"]:
                self._y_holdout = pd.Series(
                    self._observations["y_holdout"],
                    index=self._observations["y_forecast_index"],
                )

    def _select_distribution(self):
        """
        Select appropriate distribution based on model and loss function.

        Distribution mapping:
        - likelihood: dnorm (additive) or dgamma (multiplicative)
        - MAE-based losses: dlaplace
        - HAM-based losses: ds
        - MSE-based and other losses: dnorm
        """
        if self._general["distribution"] == "default":
            loss = self._general["loss"]
            if loss == "likelihood":
                if self._model_type["error_type"] == "A":
                    self._general["distribution_new"] = "dnorm"
                elif self._model_type["error_type"] == "M":
                    self._general["distribution_new"] = "dgamma"
            elif loss in [
                "MAE",
                "MAEh",
                "TMAE",
                "GTMAE",
                "MACE",
                "aTMAE",
                "aGTMAE",
                "aMACE",
            ]:
                self._general["distribution_new"] = "dlaplace"
            elif loss in [
                "HAM",
                "HAMh",
                "THAM",
                "GTHAM",
                "CHAM",
                "aTHAM",
                "aGTHAM",
                "aCHAM",
            ]:
                self._general["distribution_new"] = "ds"
            elif loss in [
                "MSE",
                "MSEh",
                "TMSE",
                "GTMSE",
                "MSCE",
                "GPL",
                "aMSEh",
                "aTMSE",
                "aGTMSE",
                "aMSCE",
                "aGPL",
                "LASSO",
                "RIDGE",
                "custom",
            ]:
                self._general["distribution_new"] = "dnorm"
            else:
                # Fallback to dnorm for any unrecognized loss
                self._general["distribution_new"] = "dnorm"
        else:
            self._general["distribution_new"] = self._general["distribution"]

    def _compute_fitted_values(self):
        """
        Compute fitted values and residuals after model estimation.

        This calls preparator() to run the fitter with final parameters
        and extract fitted values, residuals, and scale.
        """
        # Set a default h if not provided (needed for preparator)
        if self.h is None:
            if self._lags_model and len(self._lags_model["lags"]) > 0:
                self._general["h"] = max(self._lags_model["lags"])
            else:
                self._general["h"] = 10

        # Call preparator to compute fitted values and residuals
        self._prepared = preparator(
            model_type_dict=self._model_type,
            components_dict=self._components,
            lags_dict=self._lags_model,
            matrices_dict=self._adam_created,
            persistence_checked=self._persistence,
            initials_checked=self._initials,
            arima_checked=self._arima,
            explanatory_checked=self._explanatory,
            phi_dict=self._phi_internal,
            constants_checked=self._constant,
            observations_dict=self._observations,
            occurrence_dict=self._occurrence,
            general_dict=self._general,
            profiles_dict=self._profile,
            adam_estimated=self._adam_estimated,
            adam_cpp=self._adam_cpp,
            bounds="usual",
            other=None,
        )

    def _auto_predict(self):
        """Run point forecasts automatically if h > 0; compute accuracy if holdout."""
        h = getattr(self, "h", None)
        if not h or h <= 0:
            return

        self._general["h"] = h
        self._prepare_prediction_data()

        if getattr(self, "_om_model", None) is not None:
            occ_fc = self._om_model.predict(h=h)
            self._occurrence["p_forecast"] = np.asarray(occ_fc.mean.values, dtype=float)

        auto_fc = forecaster(
            model_prepared=self._prepared,
            observations_dict=self._observations,
            general_dict=self._general,
            occurrence_dict=self._occurrence,
            lags_dict=self._lags_model,
            model_type_dict=self._model_type,
            explanatory_checked=self._explanatory,
            components_dict=self._components,
            constants_checked=self._constant,
            params_info=self._params_info,
            adam_cpp=self._adam_cpp,
            interval="none",
            level=0.95,
            side="both",
        )
        self._auto_forecast = auto_fc

        if not getattr(self, "holdout", False):
            return
        y_holdout = self._observations.get("y_holdout")
        y_in_sample = self._observations.get("y_in_sample")
        if y_holdout is None or len(y_holdout) == 0:
            return

        from smooth.adam_general.core.utils.printing import _compute_forecast_errors

        fc_values = np.asarray(auto_fc.mean, dtype=float).ravel()
        y_holdout_arr = np.asarray(y_holdout, dtype=float).ravel()
        n = min(len(fc_values), len(y_holdout_arr))
        period = max(self._lags_model.get("lags", [1])) if self._lags_model else 1
        self.accuracy = _compute_forecast_errors(
            y_holdout_arr[:n],
            fc_values[:n],
            np.asarray(y_in_sample, dtype=float),
            period,
        )

    def _validate_prediction_inputs(self):
        """
        Validate that the model is properly fitted before prediction.

        Raises
        ------
        ValueError
            If the model has not been fitted yet or is missing required components.
        """
        # Verify that the model has been fitted
        if not hasattr(self, "_model_type"):
            raise ValueError("Model must be fitted before prediction.")

        # Check if we have the necessary components based on the model type
        if self._model_type["model_do"] == "estimate" and not hasattr(
            self, "_adam_estimated"
        ):
            raise ValueError("Model estimation results not found.")
        elif self._model_type["model_do"] == "select" and not hasattr(
            self, "_adam_selected"
        ):
            raise ValueError("Model selection results not found.")

    def _prepare_prediction_data(self):
        """
        Prepare data for prediction by setting up necessary matrices and parameters.
        """
        # If h wasn't provided, use default h
        if self.h is None:
            if self._lags_model and len(self._lags_model["lags"]) > 0:
                self.h = max(self._lags_model["lags"])

            else:
                self.h = 10
                self._general["h"] = self.h

        # Prepare necessary data for forecasting
        self._prepared = preparator(
            # Model info
            model_type_dict=self._model_type,
            # Components info
            components_dict=self._components,
            # Lags info
            lags_dict=self._lags_model,
            # Matrices from creator
            matrices_dict=self._adam_created,
            # Parameter dictionaries
            persistence_checked=self._persistence,
            initials_checked=self._initials,
            arima_checked=self._arima,
            explanatory_checked=self._explanatory,
            phi_dict=self._phi_internal,
            constants_checked=self._constant,
            # Other parameters
            observations_dict=self._observations,
            occurrence_dict=self._occurrence,
            general_dict=self._general,
            profiles_dict=self._profile,
            # The parameter vector
            adam_estimated=self._adam_estimated,
            # adamCore C++ object
            adam_cpp=self._adam_cpp,
            # Optional parameters
            bounds="usual",
            other=None,
        )

    def _execute_prediction(
        self,
        interval="prediction",
        level=0.95,
        side="both",
    ):
        """
        Execute the forecasting process based on the prepared data.

        For combined models, runs forecaster for each model with non-zero
        IC weight and combines the results.

        Returns
        -------
        ForecastResult
            Forecast results including point forecasts and prediction intervals.
        """
        # Handle combined models
        if getattr(self, "_is_combined", False):
            return self._execute_prediction_combined(
                interval=interval,
                level=level,
                side=side,
            )

        # Standard single-model prediction
        self._forecast_results = forecaster(
            model_prepared=self._prepared,
            observations_dict=self._observations,
            general_dict=self._general,
            occurrence_dict=self._occurrence,
            lags_dict=self._lags_model,
            model_type_dict=self._model_type,
            explanatory_checked=self._explanatory,
            components_dict=self._components,
            constants_checked=self._constant,
            params_info=self._params_info,
            adam_cpp=self._adam_cpp,
            interval=interval,
            level=level,
            side=side,
        )
        return self._forecast_results

    def _execute_prediction_combined(
        self,
        interval="prediction",
        level=0.95,
        side="both",
    ):
        """
        Execute combined forecasting using IC-weighted model averaging.

        Generates forecasts from all models with non-zero IC weights and
        combines them using Akaike weights.

        Returns
        -------
        ForecastResult
            IC-weighted combined forecast results.
        """
        from smooth.adam_general.core.forecaster import forecaster_combined

        self._forecast_results = forecaster_combined(
            prepared_models=self._prepared_models,
            ic_weights=self._ic_weights,
            observations_dict=self._observations,
            general_dict=self._general,
            occurrence_dict=self._occurrence,
            params_info=self._params_info,
            interval=interval,
            level=level,
            side=side,
        )
        return self._forecast_results

    def _format_prediction_results(self):
        """
        Format the prediction results into a more user-friendly structure.

        Note: This method is defined but not explicitly called within the ADAM
        class's current public interface (fit, predict, predict_intervals).

        Returns
        -------
        ForecastResult
            Formatted prediction results including point forecasts and intervals.
        """
        return self._forecast_results

    def __str__(self) -> str:
        """
        Return a formatted string representation of the fitted model.

        Returns a comprehensive summary including model type, parameters,
        information criteria, and forecast errors (if holdout was used).

        Returns
        -------
        str
            Formatted model summary
        """
        from smooth.adam_general.core.utils.printing import model_summary

        # Check if model has been fitted
        if not hasattr(self, "_model_type"):
            return f"ADAM(model={self.model}) - not fitted"

        return model_summary(self)

    def __repr__(self) -> str:
        """
        Return a string representation of the ADAM model.

        Returns
        -------
        str
            Brief model representation
        """
        if hasattr(self, "_model_type") and self._model_type:
            model_str = self._model_type.get("model", self.model)
            if self._model_type.get("ets_model", False):
                return f"ADAM(ETS({model_str}), fitted=True)"
            return f"ADAM({model_str}, fitted=True)"
        return f"ADAM(model={self.model}, fitted=False)"

    def summary(self, digits: int = 4) -> str:
        """
        Generate a formatted summary of the fitted model.

        Parameters
        ----------
        digits : int, default=4
            Number of decimal places for numeric output

        Returns
        -------
        str
            Formatted model summary

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> print(model.summary())
        """
        from smooth.adam_general.core.utils.printing import model_summary

        if not hasattr(self, "_model_type"):
            return "Model has not been fitted yet. Call fit() first."

        return model_summary(self, digits=digits)

    def plot(  # noqa: B006
        self,
        which=[1, 2, 4, 6],
        level: float = 0.95,
        legend: bool = False,
        lowess: bool = True,
        **kwargs,
    ):
        """
        Diagnostic plots for the fitted ADAM model (R: ``plot.adam``).

        Parameters
        ----------
        which : int or list of int, optional
            Plot type(s) to produce. Default ``[1, 2, 4, 6]``.

            1  — Actuals vs Fitted

            2  — Standardised Residuals vs Fitted

            3  — Studentised Residuals vs Fitted

            4  — |Residuals| vs Fitted

            5  — Residuals² vs Fitted

            6  — Q-Q plot (distribution-specific)

            7  — Actuals and Fitted over time

            8  — Standardised Residuals vs Time

            9  — Studentised Residuals vs Time

            10 — ACF of Residuals

            11 — PACF of Residuals

            12 — Model states over time

            13 — |Standardised Residuals| vs Fitted

            14 — Standardised Residuals² vs Fitted

            15 — ACF of Squared Residuals

            16 — PACF of Squared Residuals

        level : float, optional
            Confidence level for bounds and bands. Default 0.95.
        legend : bool, optional
            Show legend on applicable plots (2, 3, 7, 8, 9). Default False.
        lowess : bool, optional
            Add LOWESS smoothing line to scatter plots. Default True.
        **kwargs
            Passed to matplotlib (e.g. ``figsize``).

        Returns
        -------
        matplotlib.figure.Figure or list[matplotlib.figure.Figure]
            Single figure when ``which`` has one element, list otherwise.

        Examples
        --------
        >>> model = ADAM(model="AAA", lags=[1, 12])
        >>> model.fit(y)
        >>> figs = model.plot()                   # default: which=[1,2,4,6]
        >>> fig  = model.plot(which=7)            # single time-series plot
        >>> figs = model.plot(which=[10, 11])     # ACF and PACF
        """
        from smooth.adam_general.core.plotting import plot_adam

        return plot_adam(
            self, which=which, level=level, legend=legend, lowess=lowess, **kwargs
        )
