import warnings

import numpy as np
import pandas as pd

# Note: adam_cpp instance is passed to functions that need C++ integration
# The adamCore object is created in architector() and passed through the pipeline
from ._helpers import (
    _prepare_lookup_table,
    _prepare_matrices_for_forecast,
    _safe_create_index,
)
from .intervals import (
    ensure_level_format,
    generate_prediction_interval,
    generate_simulation_interval,
)
from .result import ForecastResult


def _prepare_forecast_index(observations_dict, general_dict):
    """
    Prepare the index for the forecast Series/DataFrame.

    Parameters
    ----------
    observations_dict : dict
        Dictionary with observation data and related information
    general_dict : dict
        Dictionary with general model parameters

    Returns
    -------
    pandas.Index
        Index for the forecast
    """
    y_forecast_start = observations_dict["y_forecast_start"]
    h = general_dict["h"]
    freq = observations_dict["frequency"]

    # Check if y_forecast_start is a valid timestamp
    try:
        # Try to create a date_range with the start
        if isinstance(y_forecast_start, (pd.Timestamp, np.datetime64)):
            observations_dict["y_forecast_index"] = pd.date_range(
                start=y_forecast_start, periods=h, freq=freq
            )
        elif isinstance(y_forecast_start, (int, np.integer)):
            # Numeric index - use RangeIndex
            observations_dict["y_forecast_index"] = pd.RangeIndex(
                start=y_forecast_start, stop=y_forecast_start + h
            )
        else:
            # Try date_range anyway, may work for some types
            observations_dict["y_forecast_index"] = pd.date_range(
                start=y_forecast_start, periods=h, freq=freq
            )
    except (TypeError, ValueError):
        # Fallback to RangeIndex
        n_obs = len(observations_dict.get("y_in_sample", []))
        observations_dict["y_forecast_index"] = pd.RangeIndex(
            start=n_obs, stop=n_obs + h
        )

    return observations_dict["y_forecast_index"]


def _check_fitted_values(model_prepared, occurrence_dict):
    """
    Check fitted values for NaNs and adjust for occurrence if needed.

    Parameters
    ----------
    model_prepared : dict
        Dictionary with the prepared model including fitted values
    occurrence_dict : dict
        Dictionary with occurrence model parameters

    Returns
    -------
    dict
        Updated model_prepared dictionary
    """
    # Check for NaNs in fitted values
    if np.any(np.isnan(model_prepared["y_fitted"])) or np.any(
        pd.isna(model_prepared["y_fitted"])
    ):
        warnings.warn(
            "Something went wrong in the estimation of the model and NaNs were "
            "produced. "
            "If this is a mixed model, consider using the pure ones instead."
        )

    # Apply occurrence model to fitted values if present
    if occurrence_dict["occurrence_model"]:
        model_prepared["y_fitted"][:] = (
            model_prepared["y_fitted"] * occurrence_dict["p_fitted"]
        )

    # Fix cases when we have zeroes in the provided occurrence
    if occurrence_dict["occurrence"] == "provided":
        model_prepared["y_fitted"][~occurrence_dict["ot_logical"]] = (
            model_prepared["y_fitted"][~occurrence_dict["ot_logical"]]
            * occurrence_dict["p_fitted"][~occurrence_dict["ot_logical"]]
        )

    return model_prepared


def _initialize_forecast_series(observations_dict, general_dict):
    """
    Initialize a pandas Series for forecasts with appropriate index.

    Parameters
    ----------
    observations_dict : dict
        Dictionary with observation data and related information
    general_dict : dict
        Dictionary with general model parameters

    Returns
    -------
    pandas.Series
        Initialized forecast series with NaN values
    """
    if general_dict["h"] <= 0:
        return None

    if not isinstance(observations_dict.get("y_in_sample"), pd.Series):
        # Use safe index creation for non-Series input
        index = _safe_create_index(
            start=observations_dict["y_forecast_start"],
            periods=general_dict["h"],
            freq=observations_dict["frequency"],
        )
        forecast_series = pd.Series(np.full(general_dict["h"], np.nan), index=index)
    else:
        forecast_series = pd.Series(
            np.full(general_dict["h"], np.nan),
            index=observations_dict["y_forecast_index"],
        )

    return forecast_series


def _determine_forecast_interval(
    general_dict, model_type_dict, explanatory_checked, lags_dict
):
    """
    Determine the appropriate interval type for forecasting.

    Parameters
    ----------
    general_dict : dict
        Dictionary with general model parameters
    model_type_dict : dict
        Dictionary with model type information
    explanatory_checked : dict
        Dictionary with external regressors information
    lags_dict : dict
        Dictionary with lag-related information

    Returns
    -------
    dict
        Updated general_dict with interval type
    """
    # If this is "prediction", do simulations for multiplicative components
    if general_dict["interval"] == "prediction":
        # Simulate stuff for the ETS only
        if (
            model_type_dict["ets_model"] or explanatory_checked["xreg_number"] > 0
        ) and (
            model_type_dict["trend_type"] == "M"
            or (
                model_type_dict["season_type"] == "M"
                and general_dict["h"] > lags_dict["lags_model_min"]
            )
        ):
            general_dict["interval"] = "simulated"
        else:
            general_dict["interval"] = "approximate"

    return general_dict


def _generate_point_forecasts(
    observations_dict,
    lags_dict,
    model_prepared,
    lookup,
    model_type_dict,
    components_dict,
    explanatory_checked,
    constants_checked,
    general_dict,
    adam_cpp,
):
    """
    Generate point forecasts using adam_cpp.forecast().

    Parameters
    ----------
    observations_dict : dict
        Dictionary with observation data and related information
    lags_dict : dict
        Dictionary with lag-related information
    model_prepared : dict
        Dictionary with the prepared model
    lookup : numpy.ndarray
        Lookup table for forecasting
    model_type_dict : dict
        Dictionary with model type information
    components_dict : dict
        Dictionary with components information
    explanatory_checked : dict
        Dictionary with external regressors information
    constants_checked : dict
        Dictionary with information about constants
    general_dict : dict
        Dictionary with general model parameters

    Returns
    -------
    numpy.ndarray
        Array of point forecasts
    """
    # Get all the necessary matrices for forecasting
    mat_vt, mat_wt, vec_g, mat_f = _prepare_matrices_for_forecast(
        model_prepared, observations_dict, lags_dict, general_dict
    )

    # Prepare data for adam_forecaster
    profiles_recent_table = np.asfortranarray(
        model_prepared["profiles_recent_table"], dtype=np.float64
    )
    index_lookup_table = np.asfortranarray(lookup, dtype=np.uint64)

    # Fix a bug I cant trace
    components_dict["components_number_ets_non_seasonal"] = (
        components_dict["components_number_ets"]
        - components_dict["components_number_ets_seasonal"]
    )

    # Call adam_cpp.forecast() with the prepared inputs
    # Note: E, T, S, nNonSeasonal, nSeasonal, nArima, nXreg, constant are set
    # during adamCore construction
    forecast_result = adam_cpp.forecast(
        matrixWt=np.asfortranarray(mat_wt, dtype=np.float64),
        matrixF=np.asfortranarray(mat_f, dtype=np.float64),
        indexLookupTable=index_lookup_table,
        profilesRecent=profiles_recent_table,
        horizon=general_dict["h"],
    )
    y_forecast = forecast_result.forecast.flatten()

    return y_forecast


def _handle_forecast_safety_checks(
    y_forecast, model_type_dict, model_prepared, general_dict
):
    """
    Perform safety checks on forecasts and issue warnings if needed.

    Parameters
    ----------
    y_forecast : numpy.ndarray
        Array of point forecasts
    model_type_dict : dict
        Dictionary with model type information
    model_prepared : dict
        Dictionary with the prepared model
    general_dict : dict
        Dictionary with general model parameters

    Returns
    -------
    numpy.ndarray
        Corrected forecast array
    """
    # Replace NaN values with zeros
    if np.any(np.isnan(y_forecast)):
        y_forecast[np.isnan(y_forecast)] = 0

    # Issue warning about potentially explosive multiplicative trend
    # Make safety checks
    # If there are NaN values
    if np.any(np.isnan(y_forecast)):
        y_forecast[np.isnan(y_forecast)] = 0

    # Make a warning about the potential explosive trend
    if (
        model_type_dict["trend_type"] == "M"
        and not model_type_dict["damped"]
        and model_prepared["profiles_recent_table"][1, 0] > 1
        and general_dict["h"] > 10
    ):
        warnings.warn(
            "Your model has a potentially explosive multiplicative trend. "
            "I cannot do anything about it, so please just be careful."
        )

    return y_forecast


def _process_occurrence_forecast(occurrence_dict, general_dict):
    """
    Process occurrence forecasts for forecast horizon.

    Parameters
    ----------
    occurrence_dict : dict
        Dictionary with occurrence model parameters
    general_dict : dict
        Dictionary with general model parameters

    Returns
    -------
    numpy.ndarray
        Array of occurrence probabilities for forecast horizon
    """
    # Initialize occurrence model flag
    occurrence_model = False
    # If the occurrence values are provided for the holdout
    if occurrence_dict.get("occurrence") is not None and isinstance(
        occurrence_dict["occurrence"], bool
    ):
        p_forecast = occurrence_dict["occurrence"] * 1
    elif occurrence_dict.get("occurrence") is not None and isinstance(
        occurrence_dict["occurrence"], (int, float)
    ):
        p_forecast = occurrence_dict["occurrence"]
    else:
        # If this is a mixture model, produce forecasts for the occurrence
        if occurrence_dict.get("occurrence_model"):
            occurrence_model = True
            if occurrence_dict["occurrence"] == "provided":
                p_forecast = np.ones(general_dict["h"])
            else:
                # TODO: Implement forecast for occurrence model
                pass
        else:
            occurrence_model = False
            # If this was provided occurrence, then use provided values
            if (
                occurrence_dict.get("occurrence") is not None
                and occurrence_dict.get("occurrence") == "provided"
                and occurrence_dict.get("p_forecast") is not None
            ):
                p_forecast = occurrence_dict["p_forecast"]
            else:
                p_forecast = np.ones(general_dict["h"])

    # Make sure that the values are of the correct length
    if general_dict["h"] < len(p_forecast):
        p_forecast = p_forecast[: general_dict["h"]]
    elif general_dict["h"] > len(p_forecast):
        p_forecast = np.concatenate(
            [p_forecast, np.repeat(p_forecast[-1], general_dict["h"] - len(p_forecast))]
        )

    return p_forecast, occurrence_model


def forecaster(
    model_prepared,
    observations_dict,
    general_dict,
    occurrence_dict,
    lags_dict,
    model_type_dict,
    explanatory_checked,
    components_dict,
    constants_checked,
    params_info,
    adam_cpp,
    interval="prediction",
    level=0.95,
    side="both",
):
    """
    Generate point forecasts and prediction intervals from an estimated ADAM model.

    This function takes a prepared (fitted) ADAM model and produces forecasts for a
    specified horizon h. It can generate point forecasts, prediction intervals, or both.
    The forecasting process uses the **recursive multi-step ahead** approach where each
    forecast step updates the state vector for the next step.

    **Forecasting Process**:

    1. **Preparation**: Set up forecast index, check fitted values, validate horizon
    2. **Lookup Table**: Create index mapping for accessing lagged states
    3. **Point Forecasts**: Call C++ forecaster to generate h-step ahead predictions
       using the state-space recursion:

       .. math::

           \\hat{y}_{T+h} = w_{T+h}' v_{T+h|T}

           v_{T+h|T} = F v_{T+h-1|T}

    4. **Safety Checks**: Ensure forecasts are valid (no NaN, appropriate bounds)
    5. **Occurrence Adjustment**: Apply occurrence probabilities for intermittent data
    6. **Cumulative Forecasts**: Sum forecasts if cumulative=True (for demand totals)
    7. **Prediction Intervals**: Generate confidence bounds using parametric,
       simulation, or bootstrap methods

    **Prediction Interval Methods**:

    - **Parametric** (default): Analytical intervals based on assumed error distribution
      and state-space variance formulas. Fast and accurate for well-specified models.

    - **Simulation**: Monte Carlo simulation of future paths using estimated model and
      error distribution. More flexible but slower. Recommended for:

      * Multiplicative error models
      * Intermittent demand
      * Non-normal distributions

    - **Bootstrap**: Resampling residuals to generate intervals (not fully
      implemented yet)

    Parameters
    ----------
    model_prepared : dict
        Prepared model from ``preparator()`` containing:

        - **'states'**: State vector matrix (n_components × T), last columns are
          used as starting point for forecasting
        - **'y_fitted'**: Fitted values for in-sample period
        - **'residuals'**: Model residuals
        - **'mat_wt'**: Measurement matrix
        - **'mat_f'**: Transition matrix
        - **'vec_g'**: Persistence vector
        - **'persistence_level'**: Estimated α (level smoothing)
        - **'persistence_trend'**: Estimated β (if trendy)
        - **'persistence_seasonal'**: Estimated γ (if seasonal)
        - **'phi'**: Damping parameter (if damped trend)
        - **'scale'**: Estimated error scale parameter
        - **'initial_level'**, **'initial_trend'**, **'initial_seasonal'**: Initial
          states
        - **'arima_polynomials'**: AR/MA polynomials (if ARIMA)

    observations_dict : dict
        Observation information containing:

        - 'y_in_sample': In-sample observed values
        - 'obs_in_sample': Number of in-sample observations
        - 'y_forecast_start': Starting time for forecasts
        - 'y_forecast_index': Pandas index for forecast period
        - 'frequency': Time series frequency (for date indexing)

    general_dict : dict
        General configuration containing:

        - **'h'**: Forecast horizon (number of steps ahead)
        - 'distribution': Error distribution ('dnorm', 'dgamma', etc.)
        - 'cumulative': Whether to produce cumulative forecasts
        - 'nsim': Number of simulations (for simulation method, default 10000)
        - 'interval': Interval type ('none', 'prediction', 'simulated', 'approximate')

    occurrence_dict : dict
        Intermittent demand configuration containing:

        - 'occurrence_model': Whether occurrence model is active
        - 'p_fitted': Fitted occurrence probabilities (if occurrence model)
        - 'ot_logical': Boolean mask for non-zero observations

    lags_dict : dict
        Lag structure containing:

        - 'lags': Primary lag vector
        - 'lags_model': Lags for each state component
        - 'lags_model_all': Complete lag specification
        - 'lags_model_max': Maximum lag (lookback period)

    model_type_dict : dict
        Model type specification containing:

        - 'error_type': 'A' (additive) or 'M' (multiplicative)
        - 'trend_type': 'N', 'A', 'Ad', 'M', 'Md'
        - 'season_type': 'N', 'A', 'M'
        - 'ets_model': Whether ETS components exist
        - 'arima_model': Whether ARIMA components exist
        - 'model_is_trendy': Trend presence flag
        - 'model_is_seasonal': Seasonality presence flag

    explanatory_checked : dict
        External regressors specification containing:

        - 'xreg_model': Whether regressors are present
        - 'xreg_number': Number of regressors
        - 'xreg_data': Regressor values for forecast period (if available)

    components_dict : dict
        Component counts containing:

        - 'components_number_all': Total state dimension
        - 'components_number_ets': ETS component count
        - 'components_number_arima': ARIMA component count

    constants_checked : dict
        Constant term specification containing:

        - 'constant_required': Whether constant is included

    params_info : list or array
        Parameter count information:

        - params_info[0][0]: Number of states
        - params_info[1][0]: Number of parameters estimated

    interval : str, default="prediction"
        Interval type, matching R's ``forecast.adam()``::

        - **"none"**: No intervals, point forecasts only.
        - **"prediction"**: Automatically selects "simulated" or "approximate".
        - **"simulated"**: Simulation-based intervals (Monte Carlo).
        - **"approximate"**: Analytical (parametric) intervals.

    level : float or list of float
        Confidence level(s) for prediction intervals.

        - Single value: e.g., 0.95 for 95% intervals
        - List: e.g., [0.80, 0.95] for 80% and 95% intervals (currently only first used)

        Standard values: 0.80 (80%), 0.90 (90%), 0.95 (95%), 0.99 (99%)

    side : str
        Which prediction interval bounds to compute:

        - **"both"**: Lower and upper bounds (default)
        - **"lower"**: Lower bound only
        - **"upper"**: Upper bound only

    Returns
    -------
    ForecastResult
        Structured result with attributes:

        - **mean**: ``pd.Series`` of point forecasts (always present).
        - **lower**: ``pd.DataFrame`` of lower bounds, columns are quantile
          values (e.g. ``0.025``).  ``None`` when ``interval="none"``
          or ``side="upper"``.
        - **upper**: ``pd.DataFrame`` of upper bounds, columns are quantile
          values (e.g. ``0.975``).  ``None`` when ``interval="none"``
          or ``side="lower"``.
        - **level**, **side**, **interval**: echo of the requested parameters.

        Also supports DataFrame-style access (``result["mean"]``,
        ``result.shape``, ``result.columns``) for backward compatibility.
        Use ``result.to_dataframe()`` for a flat ``pd.DataFrame``.

    Notes
    -----
    **Recursive Forecasting**:

    The forecaster uses a recursive approach where:

    1. At h=1: Use final state vector from fitting
    2. At h=2: Update states using h=1 forecast
    3. Continue recursively through horizon h

    This naturally propagates uncertainty as h increases.

    **State Vector Evolution**:

    For each forecast step, the state vector evolves as:

    .. math::

        v_{T+h|T} = F v_{T+h-1|T} + g \\cdot 0

    The error term is zero for point forecasts (expected value).

    **Interval Width Growth**:

    Prediction intervals widen with forecast horizon due to:

    - Accumulation of forecast errors (one-step → multi-step)
    - Uncertainty in parameter estimates
    - Uncertainty in state values

    For multiplicative error models, intervals grow faster than additive models.

    **Occurrence Probabilities**:

    For intermittent demand, forecasts are adjusted by occurrence probability:

    .. math::

        E[y_{T+h}] = E[y_{T+h} | \\text{demand}] \\times P(\\text{demand})

    Intervals account for both demand size and demand probability uncertainties.

    **Cumulative Forecasts**:

    When cumulative=True, useful for inventory management:

    - Point forecast: Sum of individual forecasts
    - Intervals: Account for correlation between forecast errors
    - Occurrence: Uses simulation method automatically (complex distribution)

    **Performance**:

    - Point forecasts only: Very fast (~1ms for h=100)
    - Parametric intervals: Fast (~5-10ms)
    - Simulation intervals: Slower (100-500ms depending on nsim)

    **Zero and Negative Forecasts**:

    - Multiplicative models: Forecasts bounded below by small positive value
    - Intermittent data: Zero forecasts occur when occurrence probability < threshold
    - Negative forecasts: Possible with additive error if data trends strongly down

    See Also
    --------
    preparator : Prepares model for forecasting (called before forecaster)
    generate_prediction_interval : Parametric interval calculation
    generate_simulation_interval : Simulation-based interval calculation
    adam_forecaster : C++ backend for fast recursive forecasting

    Examples
    --------
    Generate point forecasts only::

        >>> result = forecaster(..., interval='none')
        >>> print(result.mean)           # pd.Series of point forecasts
        >>> print(result.lower)          # None

    Generate 95% prediction intervals (auto-selects method)::

        >>> result = forecaster(..., interval='prediction', level=0.95)
        >>> print(result.lower.columns)  # Float64Index([0.025])
        >>> print(result.upper.columns)  # Float64Index([0.975])

    Generate simulation-based intervals::

        >>> result = forecaster(..., interval='simulated', level=0.90)

    Cumulative forecast for inventory planning::

        >>> result = forecaster(
        ...     general_dict={'h': 12, 'cumulative': True, ...},
        ...     interval='prediction', level=0.95, ...
        ... )
    """
    # 1. Prepare forecast index
    _prepare_forecast_index(observations_dict, general_dict)
    # 2. Check fitted values for issues and adjust for occurrence
    model_prepared = _check_fitted_values(model_prepared, occurrence_dict)

    # 3. Return empty result if horizon is zero
    if general_dict["h"] <= 0:
        return None

    # 4. Initialize forecast series structure (but we'll use numpy arrays for
    # calculations). We don't need to store this value since we're only using
    # numpy arrays internally
    _initialize_forecast_series(observations_dict, general_dict)

    # 5. Prepare lookup table for forecasting
    lookup = _prepare_lookup_table(lags_dict, observations_dict, general_dict)

    # 6. Set interval from caller and resolve "prediction" → "simulated"/"approximate"
    general_dict["interval"] = interval
    general_dict = _determine_forecast_interval(
        general_dict, model_type_dict, explanatory_checked, lags_dict
    )

    # 7. Generate point forecasts using adam_cpp.forecast()
    y_forecast_values = _generate_point_forecasts(
        observations_dict,
        lags_dict,
        model_prepared,
        lookup,
        model_type_dict,
        components_dict,
        explanatory_checked,
        constants_checked,
        general_dict,
        adam_cpp,
    )

    # 8. Perform safety checks on forecasts
    y_forecast_values = _handle_forecast_safety_checks(
        y_forecast_values, model_type_dict, model_prepared, general_dict
    )

    # Point forecasts now always come from adam_cpp.forecast(), not simulations.
    # (matches R commit 7d4d3736: forecast.adam returns $forecast, not sim mean)
    resolved_interval = general_dict["interval"]
    _cached_sim = None

    # 9. Process occurrence forecasts
    p_forecast, occurrence_model = _process_occurrence_forecast(
        occurrence_dict, general_dict
    )

    # 10. Apply occurrence probabilities and handle cumulative
    y_forecast_values = y_forecast_values * p_forecast
    if general_dict.get("cumulative"):
        y_forecast_values = np.sum(y_forecast_values)
    # In case of occurrence model use simulations - the cumulative
    # probability is complex
    if occurrence_model:
        general_dict["interval"] = "simulated"
        resolved_interval = "simulated"

    # 12. Determine index: single row for cumulative, full horizon otherwise
    is_cumulative = general_dict.get("cumulative", False)
    forecast_index = (
        observations_dict["y_forecast_index"][:1]
        if is_cumulative
        else observations_dict["y_forecast_index"]
    )

    if resolved_interval == "none":
        return ForecastResult(
            mean=pd.Series(y_forecast_values, index=forecast_index, name="mean"),
            lower=None,
            upper=None,
            level=level,
            side=side,
            interval=resolved_interval,
        )
    else:
        if level is None:
            warnings.warn("No confidence level specified. Using default level of 0.95")
            level = 0.95

        level_low, level_up = ensure_level_format(level, side)

        if resolved_interval == "simulated":
            if _cached_sim is not None:
                y_lower, y_upper, y_simulated = _cached_sim
            else:
                nsim = general_dict.get("nsim", 10000)
                y_lower, y_upper, y_simulated, _ = generate_simulation_interval(
                    y_forecast_values,
                    model_prepared,
                    general_dict,
                    observations_dict,
                    model_type_dict,
                    lags_dict,
                    components_dict,
                    explanatory_checked,
                    constants_checked,
                    params_info,
                    adam_cpp,
                    level_low,
                    level_up,
                    nsim=nsim,
                )
            if general_dict.get("scenarios", False):
                general_dict["_scenarios_matrix"] = y_simulated
        elif resolved_interval == "approximate":
            y_lower, y_upper = generate_prediction_interval(
                y_forecast_values,
                model_prepared,
                general_dict,
                observations_dict,
                model_type_dict,
                lags_dict,
                params_info,
                level_low,
                level_up,
            )
            if general_dict.get("scenarios", False):
                warnings.warn('scenarios=True requires interval="simulated". Ignored.')
        else:
            raise NotImplementedError(
                f'interval="{resolved_interval}" is not yet implemented'
            )

        n_levels = len(level_low)
        lower_df = None
        upper_df = None
        if side != "upper":
            lower_df = pd.DataFrame(
                {level_low[j]: y_lower[:, j] for j in range(n_levels)},
                index=forecast_index,
            )
        if side != "lower":
            upper_df = pd.DataFrame(
                {level_up[j]: y_upper[:, j] for j in range(n_levels)},
                index=forecast_index,
            )

    return ForecastResult(
        mean=pd.Series(y_forecast_values, index=forecast_index, name="mean"),
        lower=lower_df,
        upper=upper_df,
        level=level,
        side=side,
        interval=resolved_interval,
    )


def forecaster_combined(
    prepared_models,
    ic_weights,
    observations_dict,
    general_dict,
    occurrence_dict,
    params_info,
    interval="prediction",
    level=0.95,
    side="both",
):
    """
    Generate combined forecasts from multiple prepared models using IC weights.

    This function produces IC-weighted point forecasts and prediction intervals
    by combining forecasts from multiple ADAM models. Each model's forecast
    contribution is proportional to its Akaike weight.

    Parameters
    ----------
    prepared_models : list of dict
        List of prepared model dictionaries. Each dict must contain:

        - 'name': Model name string (e.g., 'ANN')
        - 'weight': IC weight for this model (>= 0.01 to be included)
        - 'result': Original selection result with adam_estimated
        - 'model_type_dict': Model type specification
        - 'components_dict': Component counts
        - 'lags_dict': Lag structure
        - 'prepared': Prepared model from preparator()

    ic_weights : dict
        Dictionary mapping model names to IC weights. Weights should sum to 1.0.

    observations_dict : dict
        Observation information (shared across all models).

    general_dict : dict
        General configuration containing 'h' (horizon) and other settings.

    occurrence_dict : dict
        Occurrence model specification.

    params_info : list or array
        Parameter count information for interval calculation.

    interval : str, default='prediction'
        Interval type matching R's ``forecast.adam()``:
        'none', 'prediction', 'simulated', 'approximate', etc.

    level : float, default=0.95
        Confidence level for prediction intervals (0 to 1).

    side : str, default='both'
        Interval side: 'both', 'lower', or 'upper'.

    Returns
    -------
    ForecastResult
        Structured result with ``mean`` (pd.Series), ``lower`` and
        ``upper`` (pd.DataFrame or None) containing IC-weighted combined
        forecasts.

    Notes
    -----
    The combined forecast is computed as:

    .. math::

        \\hat{y}_{T+h}^{combined} = \\sum_{m=1}^{M} w_m \\hat{y}_{T+h}^{(m)}

    where :math:`w_m` is the Akaike weight for model m.

    For prediction intervals, the same weighting is applied to lower and upper
    bounds. This is an approximation that works well when models agree.
    """
    import copy

    h = general_dict["h"]

    # Filter weights >= 0.01 and renormalize (matches R's forecast.adamCombined)
    model_weights = {m["name"]: m["weight"] for m in prepared_models}
    filtered_weights = {k: v for k, v in model_weights.items() if v >= 0.01}
    total_weight = sum(filtered_weights.values())
    if total_weight > 0:
        filtered_weights = {k: v / total_weight for k, v in filtered_weights.items()}

    # Initialize combined arrays
    has_intervals = interval != "none"
    n_out = 1 if general_dict.get("cumulative", False) else h
    y_forecast_combined = np.zeros(n_out)
    # For interval columns, accumulate by column name
    interval_accum = {}  # col_name -> np.zeros(n_out)
    forecast_index = None

    for model_info in prepared_models:
        model_name = model_info["name"]
        weight = filtered_weights.get(model_name, 0)

        if weight == 0:
            continue

        result = model_info["result"]
        prepared = model_info["prepared"]
        model_type_dict = model_info["model_type_dict"]
        components_dict = model_info["components_dict"]
        lags_dict = model_info["lags_dict"]
        adam_cpp = result["adam_estimated"]["adam_cpp"]

        general_dict_copy = copy.deepcopy(general_dict)

        model_forecast = forecaster(
            model_prepared=prepared,
            observations_dict=observations_dict,
            general_dict=general_dict_copy,
            occurrence_dict=occurrence_dict,
            lags_dict=lags_dict,
            model_type_dict=model_type_dict,
            explanatory_checked=model_info.get(
                "explanatory_dict",
                {"xreg_model": False, "xreg_number": 0},
            ),
            components_dict=components_dict,
            constants_checked=model_info.get(
                "constants_dict",
                {"constant_required": False},
            ),
            params_info=params_info,
            adam_cpp=adam_cpp,
            interval=interval,
            level=level,
            side=side,
        )

        if forecast_index is None:
            forecast_index = model_forecast.mean.index

        y_forecast_combined += (
            np.nan_to_num(model_forecast.mean.values, nan=0.0) * weight
        )

        if has_intervals:
            if model_forecast.lower is not None:
                for col in model_forecast.lower.columns:
                    key = ("lower", col)
                    if key not in interval_accum:
                        interval_accum[key] = np.zeros(n_out)
                    interval_accum[key] += (
                        np.nan_to_num(model_forecast.lower[col].values, nan=0.0)
                        * weight
                    )
            if model_forecast.upper is not None:
                for col in model_forecast.upper.columns:
                    key = ("upper", col)
                    if key not in interval_accum:
                        interval_accum[key] = np.zeros(n_out)
                    interval_accum[key] += (
                        np.nan_to_num(model_forecast.upper[col].values, nan=0.0)
                        * weight
                    )

    # Build ForecastResult
    mean_series = pd.Series(y_forecast_combined, index=forecast_index, name="mean")
    lower_df = None
    upper_df = None
    if has_intervals:
        lower_keys = sorted(k for k in interval_accum if k[0] == "lower")
        upper_keys = sorted(k for k in interval_accum if k[0] == "upper")
        if lower_keys:
            lower_df = pd.DataFrame(
                {k[1]: interval_accum[k] for k in lower_keys},
                index=forecast_index,
            )
        if upper_keys:
            upper_df = pd.DataFrame(
                {k[1]: interval_accum[k] for k in upper_keys},
                index=forecast_index,
            )

    return ForecastResult(
        mean=mean_series,
        lower=lower_df,
        upper=upper_df,
        level=level,
        side=side,
        interval=interval,
    )
