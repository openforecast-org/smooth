import warnings

import numpy as np
import pandas as pd

from ._utils import _warn
from .arima_checks import _check_arima
from .data_checks import _calculate_ot_logical, _check_lags, _check_occurrence
from .model_checks import _check_ets_model
from .organizers import (
    _calculate_n_param_max,
    _calculate_parameters_number,
    _organize_components_info,
    _organize_lags_info,
    _organize_model_type_info,
    _organize_occurrence_info,
    _organize_phi_info,
)
from .parameter_checks import (
    _check_constant,
    _check_distribution_loss,
    _check_initial,
    _check_outliers,
    _check_persistence,
    _check_phi,
    _initialize_estimation_params,
)
from .sample_size import (
    _adjust_model_for_sample_size,
    _restrict_models_pool_for_sample_size,
)


def parameters_checker(
    data,
    model,
    lags,
    orders=None,
    constant=False,
    outliers="ignore",
    level=0.99,
    persistence=None,
    phi=None,
    initial=None,
    n_iterations=None,
    distribution="default",
    loss="likelihood",
    h=0,
    holdout=False,
    occurrence="none",
    ic="AICc",
    bounds="usual",
    silent=False,
    fast=False,
    lambda_param=None,
    interval="none",
    interval_level=[0.95],
    side="both",
    cumulative=False,
    nsim=10000,
    scenarios=False,
    ellipsis=None,
    X=None,
    regressors="use",
    arma=None,
):
    """
    Validate and process all ADAM model parameters before estimation.

    This is the central parameter validation function that checks all user inputs for
    consistency, converts them to standardized internal formats, and sets up the
    complete
    model specification. It acts as a gatekeeper before model estimation, ensuring that:

    - Model specifications are valid (ETS components, ARIMA orders)
    - Data properties are appropriate (sufficient observations, valid lags)
    - Parameter specifications are consistent (bounds, distributions, loss functions)
    - Initial values and persistence parameters are properly formatted
    - Information criteria and occurrence models are correctly configured

    The function performs comprehensive validation similar to R's adam() parameter
    checking,
    transforming user-friendly inputs into the detailed dictionaries required by the
    estimation engine.

    **Validation Process**:

    1. **Occurrence Checking**: Validate intermittent demand settings
    2. **Lags Validation**: Ensure lags are compatible with data length
    3. **ETS Model Parsing**: Decode model string (e.g., "AAA", "ZXZ") into components
    4. **ARIMA Validation**: Check orders and stationarity requirements
    5. **Distribution & Loss**: Verify compatibility (e.g., multiplicative error
    requires positive data)
    6. **Outliers**: Configure outlier detection if requested
    7. **Damping (φ)**: Validate damping parameter for damped trend models
    8. **Persistence**: Process smoothing parameters (α, β, γ) - fixed or to be
    estimated
    9. **Initial States**: Configure initialization method (optimal, backcasting,
    provided)
    10. **Constants**: Set up intercept term if required
    11. **Model Pool**: Generate model pool for automatic selection ("ZZZ", "XXX", etc.)
    12. **Profiles**: Initialize time-varying parameter structures
    13. **Observations**: Format data and compute necessary statistics
    14. **Assembly**: Package all validated parameters into organized dictionaries

    Parameters
    ----------
    data : array-like, pandas.Series, or pandas.DataFrame
        Time series data for model estimation. Must be numeric and one-dimensional.
        Can handle intermittent demand (data with zeros).

        - **Length requirement**: Must have sufficient observations for the model
          (at least max(lags) + max(orders) observations)
        - **Missing values**: Handled by converting to numeric with ``pd.to_numeric``

    model : str or list of str
        ETS model specification or list of models for selection.

        Model string format: ``E + T + S`` where:

        - **E** (Error): "A" (Additive) or "M" (Multiplicative)
        - **T** (Trend): "N" (None), "A" (Additive), "Ad" (Additive damped),
          "M" (Multiplicative), "Md" (Multiplicative damped)
        - **S** (Seasonality): "N" (None), "A" (Additive), "M" (Multiplicative)

        Special codes for automatic selection:

        - **"Z"**: Select from all options (Branch & Bound algorithm)
        - **"X"**: Select only additive components
        - **"Y"**: Select only multiplicative components
        - **"C"**: Combine forecasts using IC weights
        - **"P"**: Select between pure additive and pure multiplicative
        - **"F"**: Full search across all 30 possible models
        - **"S"**: Sensible 19-model pool with finite variance

        Examples: "ANN" (Simple exponential smoothing), "AAA" (Holt-Winters additive),
        "MAM" (Multiplicative error with additive trend and multiplicative seasonality),
        "ZXZ" (Auto-select error and seasonality, only additive trend)

    lags : numpy.ndarray or list
        Seasonal lags vector. For multiple seasonality, provide multiple lags.

        - Non-seasonal model: ``lags=[1]``
        - Monthly with annual seasonality: ``lags=[1, 12]``
        - Hourly with daily and weekly seasonality: ``lags=[1, 24, 168]``

        **Important**: First lag is typically 1 for level/trend. Subsequent lags define
        seasonal patterns. Length must not exceed number of observations.

    orders : dict, list, tuple, or None, default=None
        ARIMA component specification. If None, pure ETS model is estimated.

        **Format options**:

        1. **Dict format** (recommended for clarity)::

            orders = {
                'ar': [p, P],    # AR orders: non-seasonal p, seasonal P
                'i': [d, D],     # Integration: non-seasonal d, seasonal D
                'ma': [q, Q],    # MA orders: non-seasonal q, seasonal Q
                'select': False  # Whether to auto-select orders
            }

        2. **List/tuple format**: ``[p, d, q]`` for non-seasonal ARIMA(p,d,q)

        If ``'select': True``, automatic order selection is performed (similar to
        auto.arima).

        Examples:

        - ``orders={'ar': [1, 0], 'i': [1, 0], 'ma': [1, 0]}``: ARIMA(1,1,1)
        - ``orders={'ar': [0, 1], 'i': [0, 1], 'ma': [0, 1]}``: Seasonal
        ARIMA(0,0,0)(1,1,1)
        - ``orders=[1, 1, 1]``: Non-seasonal ARIMA(1,1,1)

    constant : bool or float, default=False
        Whether to include a constant (intercept) term in the model.

        - ``False``: No constant
        - ``True``: Estimate constant
        - ``float``: Fixed constant value (not estimated)

        The constant is particularly useful for:

        - Models without trend when data has non-zero mean
        - ARIMA models with drift

    outliers : str, default="ignore"
        Outlier detection and handling method.

        - **"ignore"**: No outlier handling (default)
        - **"detect"**: Detect outliers using tsoutliers package
        - **"use"**: Use provided outlier indicators

        *Note: Outlier handling is not fully implemented in Python version yet.*

    level : float, default=0.99
        Confidence level for outlier detection (if outliers != "ignore").
        Typical values: 0.95 (5% significance), 0.99 (1% significance).

    persistence : dict, list, float, or None, default=None
        Smoothing parameters specification (α, β, γ).

        **Format options**:

        1. **None** (default): All smoothing parameters are estimated
        2. **Dict format** for granular control::

            persistence = {
                'alpha': 0.3,     # Level smoothing (or None to estimate)
                'beta': 0.1,      # Trend smoothing (or None to estimate)
                'gamma': 0.05     # Seasonal smoothing (or None to estimate)
            }

        3. **List format**: ``[α, β, γ]`` with None for parameters to estimate
        4. **Float**: Single value used for all estimated smoothing parameters (starting
        value)

        **Constraints**: During estimation, smoothing parameters are constrained to
        [0,1]
        with additional restrictions: β ≤ α, γ ≤ 1-α (usual bounds).

    phi : float or None, default=None
        Damping parameter for damped trend models (Ad or Md).

        - **None**: Estimate φ (if model has damped trend)
        - **Float in (0,1]**: Fixed damping value
        - **1.0**: No damping (equivalent to non-damped trend)

        Lower values (e.g., 0.8-0.95) produce more conservative long-term forecasts
        by damping the trend contribution over time.

    initial : str, dict, list, or None, default=None
        Initial state values specification.

        **Initialization methods**:

        - **"optimal"**: Optimize initial states along with other parameters (default)
        - **"backcasting"**: Use backcasting with 2 iterations and head refinement
        - **"complete"**: Full backcasting without subsequent optimization
        - **"two-stage"**: First backcast, then optimize using backcasted values as
        starting point

        **Fixed initial values**::

            initial = {
                'level': 100,                    # Initial level
                'trend': 5,                      # Initial trend (if trendy)
                'seasonal': [0.9, 1.0, 1.1, ...] # Initial seasonal indices (if
                seasonal)
            }

        **Hybrid approach**: Dict with some values specified and others set to None for
        estimation.

    n_iterations : int or None, default=None
        Number of backcasting iterations when initial="backcasting" or "complete".

        - **None**: Use default (2 for backcasting)
        - **int**: Custom iteration count (typically 2-5)

        More iterations improve initial state estimates but increase computation time.

    distribution : str, default="default"
        Error term probability distribution.

        Supported distributions:

        - **"default"**: Automatic selection based on error type and loss

          * Additive error → Normal (dnorm)
          * Multiplicative error → Gamma (dgamma)

        - **"dnorm"**: Normal distribution (Gaussian)
        - **"dlaplace"**: Laplace distribution (for MAE loss)
        - **"ds"**: S distribution (for HAM loss)
        - **"dgnorm"**: Generalized Normal distribution
        - **"dlnorm"**: Log-Normal distribution
        - **"dgamma"**: Gamma distribution (for multiplicative errors)
        - **"dinvgauss"**: Inverse Gaussian distribution

        The distribution affects likelihood calculation and prediction intervals.

    loss : str, default="likelihood"
        Loss function for parameter optimization.

        **One-step losses**:

        - **"likelihood"**: Maximum likelihood estimation (default)
        - **"MSE"**: Mean Squared Error
        - **"MAE"**: Mean Absolute Error
        - **"HAM"**: Half Absolute Moment (geometric mean of absolute errors)
        - **"LASSO"**: L1-regularized loss (for variable selection)
        - **"RIDGE"**: L2-regularized loss (for shrinkage)

        **Multi-step losses** (h-step ahead):

        - **"MSEh"**: h-step ahead MSE
        - **"MAEh"**: h-step ahead MAE
        - **"HAMh"**: h-step ahead HAM

        For LASSO/RIDGE, set ``lambda_param`` to control regularization strength.

    h : int, default=0
        Forecast horizon (number of steps ahead to forecast).

        - Used for holdout validation if ``holdout=True``
        - Required for multi-step losses (MSEh, MAEh, HAMh)
        - Sets prediction interval horizon

    holdout : bool, default=False
        Whether to split data into training and holdout samples.

        - ``False``: Use all data for estimation
        - ``True``: Last ``h`` observations become holdout sample for validation

        Useful for out-of-sample accuracy assessment.

    occurrence : str, default="none"
        Occurrence model for intermittent demand (data with zeros).

        - **"none"**: No occurrence model (continuous demand)
        - **"auto"**: Automatically select occurrence model
        - **"fixed"**: Fixed probability
        - **"general"**: General occurrence model
        - **"odds-ratio"**: Odds-ratio based model
        - **"inverse-odds-ratio"**: Inverse odds-ratio model
        - **"direct"**: Direct probability model
        - **"provided"**: User-provided occurrence indicators

        Occurrence models are essential for intermittent demand forecasting (e.g., spare
        parts).

    ic : str, default="AICc"
        Information criterion for model selection.

        - **"AIC"**: Akaike Information Criterion
        - **"AICc"**: Corrected AIC (recommended for small samples)
        - **"BIC"**: Bayesian Information Criterion (more parsimonious)
        - **"BICc"**: Corrected BIC

        Lower IC values indicate better models. AICc is default as it performs well
        across sample sizes.

    bounds : str, default="usual"
        Parameter constraint type during optimization.

        - **"usual"**: Classical restrictions (α,β,γ ∈ [0,1], β ≤ α, γ ≤ 1-α, φ ∈ [0,1])
        - **"admissible"**: Stability constraints based on eigenvalues of transition
        matrix
        - **"none"**: No constraints (not recommended)

        "usual" bounds are recommended for most applications. "admissible" allows more
        flexibility but may produce unstable forecasts.

    silent : bool, default=False
        Whether to suppress warning messages.

        - ``False``: Display warnings about model specification issues
        - ``True``: Silent mode (no warnings)

    fast : bool, default=False
        Whether to use faster (but possibly less accurate) estimation.

        - ``False``: Standard estimation
        - ``True``: Reduced accuracy for speed (fewer iterations, looser tolerances)

    lambda_param : float or None, default=None
        Regularization parameter for LASSO/RIDGE losses.

        - **0**: No regularization (pure MSE)
        - **1**: Full regularization (parameters shrunk to zero/heavily penalized)
        - **(0,1)**: Partial regularization

        Typical values: 0.01-0.1 for moderate regularization.

    interval : str, default="prediction"
        Prediction interval type.

        - **"none"**: No intervals.
        - **"prediction"**: Auto-selects "simulated" or "approximate".
        - **"simulated"**: Simulation-based intervals.
        - **"approximate"**: Analytical (parametric) intervals.

    interval_level : list of float, default=[0.95]
        Confidence level(s) for prediction intervals.

        Examples: ``[0.80, 0.95]`` for 80% and 95% intervals.

    side : str, default="both"
        Which prediction interval bounds to compute.

        - **"both"**: Lower and upper bounds
        - **"lower"**: Lower bound only
        - **"upper"**: Upper bound only

    cumulative : bool, default=False
        Whether to compute cumulative forecasts (sum over horizon).
        Useful for total demand forecasting.

    nsim : int, default=1000
        Number of simulations for simulation-based prediction intervals.

    scenarios : int, default=100
        Number of scenarios for scenario-based forecasting.

    ellipsis : dict or None, default=None
        Additional parameters passed through (for extensibility).

    Returns
    -------
    tuple of 13 dict
        Tuple containing validated and organized parameters:

        1. **general_dict** : General configuration (loss, distribution, bounds, ic, h,
        holdout)
        2. **observations_dict** : Data and observation-related information
        3. **persistence_results** : Validated persistence parameters
        4. **initials_results** : Validated initial state specifications
        5. **arima_results** : ARIMA component specifications
        6. **constant_dict** : Constant term configuration
        7. **model_type_dict** : Model type information (ETS, ARIMA, components)
        8. **components_dict** : Component counts and structure
        9. **lags_dict** : Lag structure and related information
        10. **occurrence_dict** : Occurrence model configuration
        11. **phi_dict** : Damping parameter specification
        12. **explanatory_dict** : External regressors configuration (not fully
        implemented)
        13. **params_info** : Parameter count information

    Raises
    ------
    ValueError
        If parameters are invalid or inconsistent:

        - Data is non-numeric or empty
        - Lags exceed data length
        - Model string is malformed
        - ARIMA orders are negative
        - Incompatible distribution/loss combination
        - Insufficient data for model complexity

    Notes
    -----
    **Parameter Checking Philosophy**:

    This function aims to fail early with clear error messages rather than allowing
    invalid configurations to proceed to estimation. It provides helpful warnings when
    suboptimal choices are detected (e.g., multiplicative seasonality with negative
    data).

    **Output Structure**:

    The function consolidates parameter validation into a single entry
    point and returns structured dictionaries grouping related settings
    (model type, distribution, ARIMA spec, occurrence, observations, etc.)
    that the downstream architector / creator / estimator consume.

    **Performance**:

    Parameter checking is fast (< 1ms typically). The main computational cost is in
    model estimation, not validation.

    See Also
    --------
    estimator : Main estimation function that uses validated parameters
    selector : Model selection function
    ADAM : User-facing class that wraps parameter_checker and estimator

    Examples
    --------
    Basic validation for simple exponential smoothing::

        >>> data = np.array([10, 12, 15, 13, 16, 18, 20, 19, 22, 25])
        >>> results = parameters_checker(
        ...     data=data,
        ...     model="ANN",
        ...     lags=[1],
        ...     silent=True
        ... )
        >>> general, obs, persist, initials, arima, const, model_type, *rest = results
        >>> print(model_type['model'])
        'ANN'

    Validation with automatic model selection::

        >>> results = parameters_checker(
        ...     data=data,
        ...     model="ZXZ",  # Auto-select error and seasonality, only additive trend
        ...     lags=[1, 12],
        ...     model_do="select",
        ...     ic="AICc",
        ...     silent=True
        ... )

    ARIMA component with fixed smoothing::

        >>> results = parameters_checker(
        ...     data=data,
        ...     model="AAN",
        ...     lags=[1],
        ...     orders={'ar': [1, 0], 'i': [0, 0], 'ma': [0, 0]},
        ...     persistence={'alpha': 0.3, 'beta': 0.1},
        ...     silent=True
        ... )
    """
    #####################
    # 1) Check Occurrence
    #####################
    # Extract values if DataFrame/Series and ensure numeric
    if hasattr(data, "values"):
        data_values = data.values
        if isinstance(data_values, np.ndarray):
            data_values = data_values.flatten()
        # Convert to numeric if needed
        data_values = pd.to_numeric(data_values, errors="coerce")
    else:
        # Convert to numeric if needed
        try:
            data_values = pd.to_numeric(data, errors="coerce")
        except Exception:
            raise ValueError("Data must be numeric or convertible to numeric values")

    occ_info = _check_occurrence(data_values, occurrence, silent, holdout, h)
    obs_in_sample = occ_info["obs_in_sample"]
    obs_nonzero = occ_info["obs_nonzero"]
    occurrence_model = occ_info["occurrence_model"]

    #####################
    # X validation
    #####################
    has_xreg, X, xreg_number, xreg_names_from_input = _validate_x(
        X, obs_all=occ_info["obs_in_sample"]
    )

    #####################
    # 2) Check Lags
    #####################
    lags_info = _check_lags(lags, obs_in_sample, silent)
    validated_lags = lags_info["lags"]
    lags_model = lags_info["lags_model"]
    lags_model_seasonal = lags_info["lags_model_seasonal"]
    max_lag = lags_info["max_lag"]

    #####################
    # 3) Check ETS Model
    #####################
    ets_info = _check_ets_model(model, distribution, data, silent, max_lag)
    ets_model = ets_info["ets_model"]
    model_do = ets_info.get("model_do", "estimate")

    #####################
    # 4) Check ARIMA
    #####################
    arima_info = _check_arima(orders, validated_lags, silent, arma=arma)
    arima_model = arima_info["arima_model"]
    ar_orders = arima_info["ar_orders"]
    i_orders = arima_info["i_orders"]
    ma_orders = arima_info["ma_orders"]
    lags_model_arima = arima_info["lags_model_arima"]

    #####################
    # 5) Check Distribution & Loss
    #####################
    dist_info = _check_distribution_loss(distribution, loss, silent)
    distribution = dist_info["distribution"]
    loss = dist_info["loss"]
    loss_function = dist_info.get("loss_function", None)

    # If ETS is off (e.g. NNN), infer error_type from the distribution:
    # log/multiplicative distributions imply multiplicative errors.
    if not ets_model:
        _mult_dists = {"dinvgauss", "dlnorm", "dllaplace", "dls", "dlgnorm", "dgamma"}
        ets_info["error_type"] = "M" if distribution in _mult_dists else "A"

    # Switch usual bounds to admissible if there's no ETS — mirrors R
    # (adamGeneral.R:2829-2832: "this speeds up ARIMA"). Without this,
    # Python takes the classical polynomial-companion eigvals path in CF
    # while R takes the state-space smoothEigens path, producing different
    # penalty values at NLopt's infeasibility probes and divergent
    # optimisation trajectories on ARIMA-only models.
    if not ets_model and bounds == "usual":
        bounds = "admissible"

    #####################
    # 6) Check Outliers
    #####################
    outliers_mode = _check_outliers(outliers, silent)

    #####################
    # 7) Check Phi
    #####################
    phi_info = _check_phi(phi, ets_info["damped"], silent)
    phi_val = phi_info["phi"]
    phi_estimate = phi_info["phi_estimate"]

    #####################
    # 8) Check Persistence
    #####################
    persist_info = _check_persistence(
        persistence=persistence,
        ets_model=ets_model,
        trend_type=ets_info["trend_type"],
        season_type=ets_info["season_type"],
        lags_model_seasonal=lags_model_seasonal,
        xreg_model=has_xreg,
        silent=silent,
    )

    #####################
    # 9) Check Initial Values
    #####################
    init_info = _check_initial(
        initial=initial,
        ets_model=ets_model,
        trend_type=ets_info["trend_type"],
        season_type=ets_info["season_type"],
        arima_model=arima_model,
        xreg_model=has_xreg,
        silent=silent,
    )

    # The gradient initial-state solve profiles the loss in C++, which a custom
    # loss callable cannot cross; the fit falls back to backcasting (mirrors R)
    if init_info["initial_type"] == "gradient" and loss == "custom":
        _warn(
            'initial="gradient" is not available for custom loss functions. '
            "Backcasting will be used instead.",
            silent,
        )

    # Process n_iterations parameter (for backcasting)
    # Default behavior: 2 for backcasting/complete, 1 for optimal/two-stage
    # Track whether user explicitly provided n_iterations
    n_iterations_provided = n_iterations is not None

    if n_iterations is None:
        if init_info["initial_type"] in ["backcasting", "complete", "gradient"]:
            n_iterations = 2
        else:
            n_iterations = 1
    else:
        # Validate user-provided n_iterations
        if not isinstance(n_iterations, int) or n_iterations < 1:
            _warn(
                "n_iterations must be a positive integer. Using default value.", silent
            )
            n_iterations_provided = False
            if init_info["initial_type"] in ["backcasting", "complete"]:
                n_iterations = 2
            else:
                n_iterations = 1

    # Add to init_info
    init_info["n_iterations"] = n_iterations
    init_info["n_iterations_provided"] = n_iterations_provided
    # Propagate ARIMA initial count from arima_info (computed in _check_arima)
    if arima_model and init_info["initial_arima_number"] == 0:
        init_info["initial_arima_number"] = arima_info.get("initial_arima_number", 0)

    #####################
    # 10) Check Constant
    #####################
    constant_dict = _check_constant(constant, silent)

    #####################
    # 11) Check Bounds
    #####################
    if bounds not in ["usual", "admissible", "none"]:
        _warn(f"Unknown bounds='{bounds}'. Switching to 'usual'.", silent)
        bounds = "usual"

    #####################
    # 12) Check Holdout
    #####################
    if holdout and h <= 0:
        _warn(
            "holdout=TRUE but horizon 'h' is not positive. "
            "No real holdout can be made.",
            silent,
        )

    #####################
    # 13) Check Model Pool
    #####################
    # Check if multiplicative models are allowed
    if hasattr(data, "values"):
        actual_values = (
            data.values.flatten() if hasattr(data.values, "flatten") else data.values
        )
    else:
        actual_values = np.asarray(data).flatten()

    allow_multiplicative = not (
        (any(y <= 0 for y in actual_values if not np.isnan(y)) and not occurrence_model)
        or (occurrence_model and any(y < 0 for y in actual_values if not np.isnan(y)))
    )

    #  Calculate n_param_max to determine if pool restriction is needed (R lines
    # 2641-2651)
    model_is_trendy = ets_info["trend_type"] not in ["N", None]
    model_is_seasonal = (
        ets_info["season_type"] not in ["N", None] and len(lags_model_seasonal) > 0
    )

    n_param_max = _calculate_n_param_max(
        ets_model=ets_model,
        persistence_level_estimate=persist_info.get("persistence_level_estimate", True),
        model_is_trendy=model_is_trendy,
        persistence_trend_estimate=persist_info.get("persistence_trend_estimate", True),
        model_is_seasonal=model_is_seasonal,
        persistence_seasonal_estimate=persist_info.get(
            "persistence_seasonal_estimate", [True] * len(lags_model_seasonal)
        ),
        phi_estimate=phi_estimate,
        initial_type=init_info.get("initial_type", "optimal"),
        initial_level_estimate=init_info.get("initial_level_estimate", True),
        initial_trend_estimate=init_info.get("initial_trend_estimate", True),
        initial_seasonal_estimate=init_info.get(
            "initial_seasonal_estimate", [True] * len(lags_model_seasonal)
        ),
        lags_model_seasonal=lags_model_seasonal,
        arima_model=arima_model,
        initial_arima_number=arima_info.get("initial_arima_number", 0),
        ar_required=any(ar_orders) if ar_orders else False,
        ar_estimate=arima_info.get("ar_estimate", True),
        ar_orders=ar_orders,
        ma_required=any(ma_orders) if ma_orders else False,
        ma_estimate=arima_info.get("ma_estimate", True),
        ma_orders=ma_orders,
        xreg_model=has_xreg,
        xreg_number=xreg_number,
        initial_xreg_estimate=has_xreg,
        persistence_xreg_estimate=(has_xreg and regressors == "adapt"),
    )

    # Restrict models pool based on sample size (R lines 2641-2944)
    # Only apply if obs_nonzero <= n_param_max (R line 2655)
    pool_restriction = _restrict_models_pool_for_sample_size(
        obs_nonzero=obs_nonzero,
        lags_model_max=max_lag,
        model_do=model_do,
        error_type=ets_info["error_type"],
        trend_type=ets_info["trend_type"],
        season_type=ets_info["season_type"],
        models_pool=ets_info.get("models_pool"),
        allow_multiplicative=allow_multiplicative,
        xreg_number=xreg_number,
        silent=silent,
        n_param_max=n_param_max,
        damped=ets_info.get("damped", False),
    )

    # Update ets_info with restricted values
    model_changed = False
    if pool_restriction["error_type"] != ets_info["error_type"]:
        ets_info["error_type"] = pool_restriction["error_type"]
        model_changed = True
    if pool_restriction["trend_type"] != ets_info["trend_type"]:
        ets_info["trend_type"] = pool_restriction["trend_type"]
        model_changed = True
    if pool_restriction["season_type"] != ets_info["season_type"]:
        ets_info["season_type"] = pool_restriction["season_type"]
        model_changed = True
    if pool_restriction["models_pool"] is not None:
        ets_info["models_pool"] = pool_restriction["models_pool"]
    if pool_restriction["model_do"] != model_do:
        model_do = pool_restriction["model_do"]
        ets_info["model_do"] = model_do
    # Update damped flag if it was changed
    if pool_restriction["damped"] != ets_info.get("damped", False):
        ets_info["damped"] = pool_restriction["damped"]
        model_changed = True

    # Rebuild model string if any component changed
    if model_changed:
        new_model = ets_info["error_type"] + ets_info["trend_type"]
        if ets_info["damped"] and ets_info["trend_type"] != "N":
            new_model += "d"
        new_model += ets_info["season_type"]
        ets_info["model"] = new_model

    # Update persistence if restricted
    if pool_restriction["persistence_level"] is not None:
        persist_info["persistence"] = pool_restriction["persistence_level"]
    if not pool_restriction["persistence_estimate"]:
        persist_info["persistence_estimate"] = False
        persist_info["persistence_level_estimate"] = False

    # Update phi_estimate if restricted
    if not pool_restriction["phi_estimate"]:
        phi_estimate = False

    # Update initial if restricted
    if pool_restriction["initial_type"] is not None:
        initial = pool_restriction["initial_type"]
    if not pool_restriction["initial_estimate"]:
        init_info["initial_estimate"] = False
        init_info["initial_level_estimate"] = False

    # Setup model type dictionary
    model_type_dict = _organize_model_type_info(
        ets_info, arima_info, xreg_model=has_xreg
    )

    # Apply additional sample size adjustments
    model_type_dict = _adjust_model_for_sample_size(
        model_info=model_type_dict,
        obs_nonzero=obs_nonzero,
        lags_model_max=max_lag,
        allow_multiplicative=allow_multiplicative,
        xreg_number=xreg_number,
        silent=silent,
    )

    # Update models_pool from restriction
    if pool_restriction["models_pool"] is not None:
        model_type_dict["models_pool"] = pool_restriction["models_pool"]

    # Components info
    components_dict = _organize_components_info(
        ets_info, arima_info, lags_model_seasonal
    )

    # Create lags dictionary
    lags_dict = _organize_lags_info(
        validated_lags=validated_lags,
        lags_model=lags_model,
        lags_model_seasonal=lags_model_seasonal,
        lags_model_arima=lags_model_arima,
        xreg_model=has_xreg,
    )

    # Create occurrence dictionary
    occurrence_dict = _organize_occurrence_info(
        occurrence=occ_info["occurrence"],
        occurrence_model=occurrence_model,
        obs_in_sample=obs_in_sample,
        h=h,
    )

    # Create phi dictionary
    phi_dict = _organize_phi_info(phi_val=phi_val, phi_estimate=phi_estimate)

    # Calculate observation logical vector
    ot_info = _calculate_ot_logical(
        data=data,
        occurrence=occurrence_dict["occurrence"],
        occurrence_model=occurrence_dict["occurrence_model"],
        obs_in_sample=obs_in_sample,
        h=h,
        holdout=holdout,
    )

    # Create observations dictionary
    # Use actual y_in_sample length to account for holdout split
    actual_y_in_sample = ot_info.get("y_in_sample", data)
    actual_obs_in_sample = (
        len(actual_y_in_sample)
        if hasattr(actual_y_in_sample, "__len__")
        else obs_in_sample
    )
    observations_dict = {
        "obs_in_sample": actual_obs_in_sample,
        "obs_nonzero": obs_nonzero,
        "obs_all": occ_info["obs_all"],
        # "obs_states": obs_states,
        "ot_logical": ot_info["ot_logical"],
        "ot": ot_info["ot"],
        "y_in_sample": ot_info.get("y_in_sample", data),
        "y_holdout": ot_info.get("y_holdout", None),
        "frequency": ot_info["frequency"],
        "y_start": ot_info["y_start"],
        "y_in_sample_index": ot_info.get("y_in_sample_index", None),
        "y_forecast_start": ot_info["y_forecast_start"],
        "y_forecast_index": ot_info.get("y_forecast_index", None),
    }

    # Determine if multistep loss is used
    multistep_losses = [
        "MSEh",
        "TMSE",
        "GTMSE",
        "MSCE",
        "MAEh",
        "TMAE",
        "GTMAE",
        "MACE",
        "HAMh",
        "THAM",
        "GTHAM",
        "CHAM",
        "GPL",
        "aMSEh",
        "aTMSE",
        "aGTMSE",
        "aMSCE",
        "aGPL",
    ]
    multisteps = loss in multistep_losses

    # Create general dictionary with remaining parameters
    general_dict = {
        "distribution": distribution,
        "loss": loss,
        "multisteps": multisteps,
        "outliers": outliers_mode,
        "h": h,
        "holdout": holdout,
        "ic": ic,
        "bounds": bounds,
        "model_do": model_do,
        "fast": fast,
        "interval": interval,
        "interval_level": interval_level,
        "side": side,
        "cumulative": cumulative,
        "nsim": nsim,
        "scenarios": scenarios,
        "ellipsis": ellipsis,
        "lambda": lambda_param if lambda_param is not None else 1,
    }
    # Add custom loss function if provided
    if loss_function is not None:
        general_dict["loss_function"] = loss_function

    # Initialize estimation parameters if needed
    if model_do == "estimate":
        est_params = _initialize_estimation_params(
            loss=loss,
            lambda_param=lambda_param if lambda_param is not None else 1,
            ets_info=ets_info,
            arima_info=arima_info,
            silent=silent,
        )
        # Update general dict with estimation parameters
        general_dict.update(
            {
                "lambda": est_params.get("lambda", 1),
                "lambda_": est_params.get("lambda_"),
                "arma_params": est_params.get("arma_params", None),
            }
        )

    # Create persistence dictionary
    persistence_dict = {
        "persistence": persist_info["persistence"],
        "persistence_estimate": persist_info["persistence_estimate"],
        "persistence_level": persist_info["persistence_level"],
        "persistence_level_estimate": persist_info["persistence_level_estimate"],
        "persistence_trend": persist_info["persistence_trend"],
        "persistence_trend_estimate": persist_info["persistence_trend_estimate"],
        "persistence_seasonal": persist_info["persistence_seasonal"],
        "persistence_seasonal_estimate": persist_info["persistence_seasonal_estimate"],
        "persistence_xreg": persist_info["persistence_xreg"],
        "persistence_xreg_estimate": persist_info["persistence_xreg_estimate"],
        "persistence_xreg_provided": persist_info["persistence_xreg_provided"],
    }

    # Create initials dictionary
    initials_dict = {
        "initial": init_info["initial"],
        "initial_type": init_info["initial_type"],
        "initial_estimate": init_info["initial_estimate"],
        "initial_level": init_info["initial_level"],
        "initial_level_estimate": init_info["initial_level_estimate"],
        "initial_trend": init_info["initial_trend"],
        "initial_trend_estimate": init_info["initial_trend_estimate"],
        "initial_seasonal": init_info["initial_seasonal"],
        "initial_seasonal_estimate": init_info["initial_seasonal_estimate"],
        "initial_arima": init_info["initial_arima"],
        "initial_arima_estimate": init_info["initial_arima_estimate"],
        "initial_arima_number": init_info["initial_arima_number"],
        "initial_xreg": init_info.get("initial_xreg"),
        "initial_xreg_estimate": init_info["initial_xreg_estimate"],
        "initial_xreg_provided": init_info["initial_xreg_provided"],
        "n_iterations": init_info["n_iterations"],
        "n_iterations_provided": init_info["n_iterations_provided"],
    }

    # Create ARIMA dictionary
    arima_dict = {
        "arima_model": arima_model,
        "ar_orders": ar_orders,
        "i_orders": i_orders,
        "ma_orders": ma_orders,
        "ar_required": arima_info.get("ar_required", False),
        "i_required": arima_info.get("i_required", False),
        "ma_required": arima_info.get("ma_required", False),
        "ar_estimate": arima_info.get("ar_estimate", False),
        "ma_estimate": arima_info.get("ma_estimate", False),
        "arma_parameters": arima_info.get("arma_parameters", None),
        "non_zero_ari": arima_info.get("non_zero_ari", []),
        "non_zero_ma": arima_info.get("non_zero_ma", []),
        "select": arima_info.get("select", False),
        "components_number_arima": arima_info.get("components_number_arima", 0),
        "lags_model_arima": arima_info.get("lags_model_arima", []),
    }

    # Pure-regression early exit: when model="NNN" (no ETS, no ARIMA) and
    # regressors are provided (not adaptive), delegate directly to
    # greybox.ALM and return it instead of the 13-tuple.
    _arima_nonzero = any(
        v > 0
        for orders_list in (ar_orders or [], i_orders or [], ma_orders or [])
        for v in orders_list
    )
    if not ets_model and not _arima_nonzero and has_xreg and regressors != "adapt":
        from greybox import ALM

        n = observations_dict["obs_in_sample"]
        y_is = np.asarray(observations_dict["y_in_sample"], dtype=float)
        dist = _map_distribution_for_greybox(distribution)
        if regressors == "select":
            names = xreg_names_from_input or [f"x{i + 1}" for i in range(xreg_number)]
            X_df = pd.DataFrame(X[:n], columns=names)
            X_df.insert(0, "y", y_is)
            from greybox import stepwise

            return stepwise(X_df, ic=ic, distribution=dist, silent=True)
        alm = ALM(distribution=dist)
        alm.fit(np.column_stack([np.ones(n), X[:n]]), y_is)
        return alm

    # Build explanatory variables dictionary
    if has_xreg:
        xreg_dict = _process_xreg(
            X=X,
            regressors=regressors,
            y_in_sample=observations_dict["y_in_sample"],
            obs_in_sample=observations_dict["obs_in_sample"],
            distribution=distribution,
            e_type=ets_info["error_type"],
            ic=ic,
            xreg_names_from_input=xreg_names_from_input,
            initial_xreg=initials_dict.get("initial_xreg"),
        )
        # For "select": if no variables survived selection, disable xreg
        if not xreg_dict["xreg_model"]:
            has_xreg = False
            model_type_dict["xreg_model"] = False
        else:
            # Update persistence and initials based on regressors mode
            if regressors == "adapt":
                persistence_dict["persistence_xreg_estimate"] = True
                initials_dict["initial_xreg_estimate"] = True
            else:
                persistence_dict["persistence_xreg_estimate"] = False
                initials_dict["initial_xreg_estimate"] = True
    else:
        xreg_dict = {
            "xreg_model": False,
            "regressors": None,
            "xreg_model_initials": None,
            "xreg_data": None,
            "xreg_number": 0,
            "xreg_names": None,
            "response_name": None,
            "formula": None,
            "xreg_parameters_missing": None,
            "xreg_parameters_included": None,
            "xreg_parameters_estimated": None,
            "xreg_parameters_persistence": None,
        }

    # Calculate number of parameters using the new n_param table structure
    from smooth.adam_general.core.utils.n_param import build_n_param_table

    n_param = build_n_param_table(
        model_type_dict=model_type_dict,
        persistence_checked=persistence_dict,
        initials_checked=initials_dict,
        arima_checked=arima_dict,
        phi_dict=phi_dict,
        constants_checked=constant_dict,
        explanatory_checked=xreg_dict,
        general=general_dict,
    )

    # Also keep legacy format for backward compatibility
    params_info = _calculate_parameters_number(
        ets_info=ets_info,
        arima_info=arima_info,
        xreg_info=None,
        constant_required=constant_dict["constant_required"],
    )

    # Set parameters number in general dict
    general_dict["parameters_number"] = params_info
    general_dict["n_param"] = n_param

    # Return all dictionaries
    return (
        general_dict,
        observations_dict,
        persistence_dict,
        initials_dict,
        arima_dict,
        constant_dict,
        model_type_dict,
        components_dict,
        lags_dict,
        occurrence_dict,
        phi_dict,
        xreg_dict,
        params_info,
    )


# ---------------------------------------------------------------------------
# Explanatory-variable helpers
# ---------------------------------------------------------------------------


def _validate_x(X, obs_all):
    """Validate and normalise the X matrix; return (has_xreg, X, n_cols, names)."""
    if X is None:
        return False, None, 0, None

    if hasattr(X, "columns"):
        names = list(X.columns)
        X = X.values
    else:
        names = None

    X = np.asarray(X)
    # Structured / record arrays (e.g. from R data frames via rpy2)
    if X.dtype.names is not None:
        if names is None:
            names = list(X.dtype.names)
        X = np.column_stack([X[f].astype(float) for f in X.dtype.names])
    else:
        X = X.astype(float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n = len(X)
    if n > obs_all:
        X = X[:obs_all]
    elif n < obs_all:
        pad = obs_all - n
        warnings.warn(
            f"X has {n} rows but y has {obs_all}. "
            f"The last row of X will be repeated {pad} time(s) to match.",
            stacklevel=4,
        )
        X = np.vstack([X, np.tile(X[-1], (pad, 1))])

    return True, X, X.shape[1], names


def _map_distribution_for_greybox(distribution):
    """Map a smooth distribution string to the nearest greybox ALM distribution."""
    _map = {
        "dnorm": "dnorm",
        "dlnorm": "dlnorm",
        "dgamma": "dgamma",
        "dlaplace": "dlaplace",
        "ds": "ds",
        "dgnorm": "dgnorm",
        "dinvgauss": "dinvgauss",
        "plogis": "plogis",
    }
    return _map.get(distribution, "dnorm")


def _process_xreg(
    X,
    regressors,
    y_in_sample,
    obs_in_sample,
    distribution,
    e_type,
    ic,
    xreg_names_from_input=None,
    initial_xreg=None,
):
    """Process explanatory variables and return a populated xreg_dict.

    Parameters
    ----------
    X : np.ndarray, shape (obs_all, p)
    regressors : {"use", "select", "adapt"}
    y_in_sample : array-like, shape (obs_in_sample,)
    obs_in_sample : int
    distribution : str  smooth distribution name
    ic : str  information criterion for stepwise selection
    xreg_names_from_input : list[str] or None
    initial_xreg : np.ndarray or None
        User-supplied initial coefficient values, shape (p, 1). If provided,
        skips ALM computation and uses these values as optimizer seed.

    Returns
    -------
    dict  populated xreg_dict (xreg_model may be False if selection drops all vars)
    """
    from greybox import ALM

    n_cols = X.shape[1]
    X_in_sample = X[:obs_in_sample]
    y_is = np.asarray(y_in_sample, dtype=float)

    # Default names
    xreg_names = (
        xreg_names_from_input
        if xreg_names_from_input is not None
        else [f"x{i + 1}" for i in range(n_cols)]
    )

    # ---------- variable selection ----------
    selected_mask = np.ones(n_cols, dtype=bool)
    if regressors == "select":
        import pandas as pd
        from greybox import stepwise

        df_sw = pd.DataFrame(X_in_sample, columns=xreg_names)
        df_sw.insert(0, "y", y_is)
        sw_dist = _map_distribution_for_greybox(distribution)
        try:
            sw_model = stepwise(df_sw, ic=ic, distribution=sw_dist, silent=True)
            sel_names = sw_model._feature_names or []
            selected_mask = np.array([n in sel_names for n in xreg_names])
        except Exception:
            pass  # keep all variables on failure

        if not selected_mask.any():
            # No variables selected — return disabled xreg_dict
            return {
                "xreg_model": False,
                "regressors": regressors,
                "xreg_model_initials": None,
                "xreg_data": None,
                "xreg_number": 0,
                "xreg_names": None,
                "response_name": None,
                "formula": None,
                "xreg_parameters_missing": None,
                "xreg_parameters_included": None,
                "xreg_parameters_estimated": None,
                "xreg_parameters_persistence": None,
            }

    # Apply selection mask
    X_selected = X[:, selected_mask]
    X_in_sel = X_selected[:obs_in_sample]
    names_selected = [n for n, s in zip(xreg_names, selected_mask) if s]
    p = X_selected.shape[1]

    # ---------- initialise coefficients ----------
    if initial_xreg is not None:
        initial_xreg_flat = np.asarray(initial_xreg, dtype=float).ravel()
        if len(initial_xreg_flat) != p:
            raise ValueError(
                f"initial['xreg'] has {len(initial_xreg_flat)} values "
                f"but model has {p} regressors."
            )
        xreg_initials = [
            {"initial_xreg": initial_xreg_flat.reshape(-1, 1)},
            {"initial_xreg": initial_xreg_flat.reshape(-1, 1)},
        ]
    else:
        X_aug = np.column_stack([np.ones(obs_in_sample), X_in_sel])
        # R log-transforms y for multiplicative models with these distributions
        # (adamGeneral.R:1400). For dlnorm/dgamma/dinvgauss the distribution
        # itself handles the multiplicative structure, so no log-transform.
        _log_dists = {
            "dnorm",
            "dlaplace",
            "ds",
            "dgnorm",
            "dlogis",
            "dt",
            "dalaplace",
        }

        def _resolve_dist_for_alm(etype, dist):
            """Resolve distribution for ALM, matching R's xregInitialiser."""
            if dist == "default":
                return "dnorm" if etype == "A" else "dlnorm"
            return _map_distribution_for_greybox(dist)

        def _fit_alm(y_response, etype):
            alm_dist = _resolve_dist_for_alm(etype, distribution)
            try:
                alm = ALM(distribution=alm_dist)
                alm.fit(X_aug, y_response, feature_names=names_selected)
                return np.asarray(alm.coef, dtype=float)
            except Exception:
                return np.zeros(p)

        def _mult_response():
            """Get response for multiplicative ALM: log(y) if needed."""
            resolved = distribution if distribution != "default" else "dlnorm"
            if resolved in _log_dists:
                return np.log(np.maximum(y_is, 1e-10))
            return y_is

        if e_type == "A":
            xreg_initials = [
                {"initial_xreg": _fit_alm(y_is, "A").reshape(-1, 1)},
                None,
            ]
        elif e_type == "M":
            xreg_initials = [
                None,
                {"initial_xreg": _fit_alm(_mult_response(), "M").reshape(-1, 1)},
            ]
        else:
            # "Z" — error type selection, need both additive and multiplicative
            xreg_initials = [
                {"initial_xreg": _fit_alm(y_is, "A").reshape(-1, 1)},
                {"initial_xreg": _fit_alm(_mult_response(), "M").reshape(-1, 1)},
            ]

    return {
        "xreg_model": True,
        "regressors": regressors,
        "xreg_model_initials": xreg_initials,
        "xreg_data": X_selected,
        "xreg_number": p,
        "xreg_names": names_selected,
        "response_name": None,
        "formula": None,
        "xreg_parameters_missing": None,
        "xreg_parameters_included": None,
        "xreg_parameters_estimated": [1] * p,
        "xreg_parameters_persistence": list(range(1, p + 1)),
    }
