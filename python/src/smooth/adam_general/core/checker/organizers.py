import numpy as np


def _organize_model_type_info(ets_info, arima_info, xreg_model=False):
    """
    Organize model type information into a consolidated dictionary.

    Parameters
    ----------
    ets_info : dict
        ETS model information
    arima_info : dict
        ARIMA model information
    xreg_model : bool, optional
        Whether external regressors are used

    Returns
    -------
    dict
        Consolidated model type information
    """
    # Create standard model type dictionary
    model_type_dict = {
        "ets_model": ets_info["ets_model"],
        "arima_model": arima_info["arima_model"],
        "xreg_model": xreg_model,
        "model": ets_info["model"],
        "error_type": ets_info["error_type"],
        "trend_type": ets_info["trend_type"],
        "season_type": ets_info["season_type"],
        "damped": ets_info["damped"],
        "allow_multiplicative": ets_info["allow_multiplicative"],
        "model_do": ets_info.get("model_do", "estimate"),  # Use model_do from ets_info
        "models_pool": ets_info.get(
            "models_pool", None
        ),  # Use models_pool from ets_info
        "model_is_trendy": ets_info["trend_type"] != "N",
        "model_is_seasonal": ets_info["season_type"] != "N",
    }

    return model_type_dict


def _organize_components_info(ets_info, arima_info, lags_model_seasonal):
    """
    Organize model components information.

    Parameters
    ----------
    ets_info : dict
        ETS model information
    arima_info : dict
        ARIMA model information
    lags_model_seasonal : list
        List of seasonal lags

    Returns
    -------
    dict
        Dictionary with components information
    """
    # Calculate number of ETS components
    if ets_info["ets_model"]:
        components_number_ets = (
            1 + (ets_info["trend_type"] != "N") + (ets_info["season_type"] != "N")
        )
        components_number_ets_seasonal = (
            len(lags_model_seasonal) if ets_info["season_type"] != "N" else 0
        )
        components_number_ets_non_seasonal = (
            components_number_ets - components_number_ets_seasonal
        )
    else:
        components_number_ets = 0
        components_number_ets_seasonal = 0
        components_number_ets_non_seasonal = 0

    # Calculate number of ARIMA components
    components_number_arima = (
        (
            sum(arima_info["ar_orders"])
            + sum(arima_info["ma_orders"])
            + sum(arima_info["i_orders"])
        )
        if arima_info["arima_model"]
        else 0
    )

    # Create components dictionary
    components_dict = {
        "components_number_ets": components_number_ets,
        "components_number_ets_seasonal": components_number_ets_seasonal,
        "components_number_ets_non_seasonal": components_number_ets_non_seasonal,
        "components_number_arima": components_number_arima,
    }

    return components_dict


def _organize_lags_info(
    validated_lags, lags_model, lags_model_seasonal, lags_model_arima, xreg_model=False
):
    """
    Organize lags information.

    Parameters
    ----------
    validated_lags : list
        List of validated lags
    lags_model : list
        List of model lags
    lags_model_seasonal : list
        List of seasonal lags
    lags_model_arima : list
        List of ARIMA lags
    xreg_model : bool, optional
        Whether external regressors are used

    Returns
    -------
    dict
        Dictionary with lags information
    """
    # Calculate the maximum lag
    lags_model_max = max(validated_lags) if validated_lags else 1

    # Smallest seasonal lag (excluding trivial lag=1), or Inf if none
    lags_model_all = sorted(set(lags_model + lags_model_arima))
    lags_non_trivial = [lag for lag in lags_model_all if lag != 1]
    lags_model_min = min(lags_non_trivial) if lags_non_trivial else float("inf")

    # Create lags dictionary
    lags_dict = {
        "lags": validated_lags,
        "lags_model": lags_model,
        "lags_model_seasonal": lags_model_seasonal,
        "lags_model_arima": lags_model_arima,
        "lags_length": len(lags_model),
        "lags_model_max": lags_model_max,
        "lags_model_min": lags_model_min,
        "lags_model_all": lags_model_all,
    }

    return lags_dict


def _organize_occurrence_info(occurrence, occurrence_model, obs_in_sample, h=0):
    """
    Organize occurrence information.

    Parameters
    ----------
    occurrence : str
        Occurrence type
    occurrence_model : bool
        Whether occurrence model is used
    obs_in_sample : int
        Number of in-sample observations
    h : int, optional
        Forecast horizon

    Returns
    -------
    dict
        Dictionary with occurrence information
    """
    # Create occurrence dictionary
    occurrence_dict = {
        "occurrence": occurrence,
        "occurrence_model": occurrence_model,
        "oes_model": "none",
        "p_fitted": None,  # Filled by ADAM.fit() from a fitted OM/OMG/AutoOM
    }

    return occurrence_dict


def _organize_phi_info(phi_val, phi_estimate):
    """
    Organize phi (damping) parameter information.

    Parameters
    ----------
    phi_val : float
        Phi value
    phi_estimate : bool
        Whether phi should be estimated

    Returns
    -------
    dict
        Dictionary with phi information
    """
    return {"phi": phi_val, "phi_estimate": phi_estimate}


def _calculate_parameters_number(
    ets_info, arima_info, xreg_info=None, constant_required=False
):
    """Calculate number of parameters for different model components.

    Returns a 2x1 array-like structure similar to R's parametersNumber matrix:
    - Row 1: Number of states/components
    - Row 2: Number of parameters to estimate
    """
    # Initialize parameters number matrix (2x1)
    parameters_number = [[0], [0]]  # Mimics R's matrix(0,2,1)

    # Count states (first row)
    if ets_info["ets_model"]:
        # Add level component
        parameters_number[0][0] += 1
        # Add trend if present
        if ets_info["trend_type"] != "N":
            parameters_number[0][0] += 1
        # Add seasonal if present
        if ets_info["season_type"] != "N":
            parameters_number[0][0] += 1

    # Count parameters to estimate (second row)
    if ets_info["ets_model"]:
        # Level persistence
        parameters_number[1][0] += 1
        # Trend persistence if present
        if ets_info["trend_type"] != "N":
            parameters_number[1][0] += 1
            # Additional parameter for damped trend
            if ets_info["damped"]:
                parameters_number[1][0] += 1
        # Seasonal persistence if present
        if ets_info["season_type"] != "N":
            parameters_number[1][0] += 1

    # Add ARIMA parameters if present
    if arima_info["arima_model"]:
        # Add number of ARMA parameters
        parameters_number[1][0] += len(arima_info.get("arma_parameters") or [])

    # Add constant if required
    if constant_required:
        parameters_number[1][0] += 1

    # Handle pure constant model case (no ETS, no ARIMA, no xreg)
    if not ets_info["ets_model"] and not arima_info["arima_model"] and not xreg_info:
        parameters_number[0][0] = 0
        parameters_number[1][0] = 2  # Matches R code line 3047

    return parameters_number
    # return {
    #    "parameters_number": parameters_number,
    #    "n_states": parameters_number[0][0],
    #    "n_params": parameters_number[1][0]
    # }


def _calculate_n_param_max(
    ets_model,
    persistence_level_estimate,
    model_is_trendy,
    persistence_trend_estimate,
    model_is_seasonal,
    persistence_seasonal_estimate,
    phi_estimate,
    initial_type,
    initial_level_estimate,
    initial_trend_estimate,
    initial_seasonal_estimate,
    lags_model_seasonal,
    arima_model=False,
    initial_arima_number=0,
    ar_required=False,
    ar_estimate=False,
    ar_orders=None,
    ma_required=False,
    ma_estimate=False,
    ma_orders=None,
    xreg_model=False,
    xreg_number=0,
    initial_xreg_estimate=False,
    persistence_xreg_estimate=False,
):
    """
    Calculate maximum number of parameters for the model.

    Follows R's adamGeneral.R lines 2641-2651.

    Parameters
    ----------
    ets_model : bool
        Whether ETS model is used
    persistence_level_estimate : bool
        Whether level persistence is estimated
    model_is_trendy : bool
        Whether model has trend component
    persistence_trend_estimate : bool
        Whether trend persistence is estimated
    model_is_seasonal : bool
        Whether model has seasonal component
    persistence_seasonal_estimate : list or bool
        Whether seasonal persistence(s) are estimated
    phi_estimate : bool
        Whether damping parameter is estimated
    initial_type : str
        Initial type ('optimal', 'two-stage', 'backcasting', 'provided')
    initial_level_estimate : bool
        Whether initial level is estimated
    initial_trend_estimate : bool
        Whether initial trend is estimated
    initial_seasonal_estimate : list or bool
        Whether initial seasonal(s) are estimated
    lags_model_seasonal : list
        List of seasonal lags
    arima_model : bool
        Whether ARIMA model is used
    initial_arima_number : int
        Number of ARIMA initial states
    ar_required : bool
        Whether AR is required
    ar_estimate : bool
        Whether AR is estimated
    ar_orders : list
        List of AR orders
    ma_required : bool
        Whether MA is required
    ma_estimate : bool
        Whether MA is estimated
    ma_orders : list
        List of MA orders
    xreg_model : bool
        Whether xreg model is used
    xreg_number : int
        Number of external regressors
    initial_xreg_estimate : bool
        Whether xreg initials are estimated
    persistence_xreg_estimate : bool
        Whether xreg persistence is estimated

    Returns
    -------
    int
        Maximum number of parameters
    """
    # Handle list/bool for persistence_seasonal_estimate
    if isinstance(persistence_seasonal_estimate, (list, np.ndarray)):
        sum_persistence_seasonal = sum(persistence_seasonal_estimate)
    else:
        sum_persistence_seasonal = (
            int(persistence_seasonal_estimate) if persistence_seasonal_estimate else 0
        )

    # Handle list/bool for initial_seasonal_estimate - multiply by lags as in R
    # R: sum(initialSeasonalEstimate*lagsModelSeasonal)
    if lags_model_seasonal and initial_seasonal_estimate:
        if isinstance(initial_seasonal_estimate, (list, np.ndarray)):
            # Element-wise multiplication of estimates and lags, then sum
            init_seasonal_arr = np.array(initial_seasonal_estimate, dtype=int)
            lags_arr = np.array(lags_model_seasonal)
            # Broadcast if lengths differ (repeat estimates if shorter)
            if len(init_seasonal_arr) < len(lags_arr):
                init_seasonal_arr = np.tile(init_seasonal_arr, len(lags_arr))[
                    : len(lags_arr)
                ]
            elif len(init_seasonal_arr) > len(lags_arr):
                init_seasonal_arr = init_seasonal_arr[: len(lags_arr)]
            sum_initial_seasonal_lags = int(np.sum(init_seasonal_arr * lags_arr))
        else:
            # Single boolean - multiply by sum of lags
            sum_initial_seasonal_lags = int(initial_seasonal_estimate) * sum(
                lags_model_seasonal
            )
    else:
        sum_initial_seasonal_lags = 0

    # Check if initial type requires estimation
    initial_needs_estimation = initial_type in ["optimal", "two-stage"]
    initial_needs_xreg = initial_type in ["backcasting", "optimal", "two-stage"]

    # ETS component
    ets_params = 0
    if ets_model:
        ets_params = (
            int(persistence_level_estimate)
            + int(model_is_trendy) * int(persistence_trend_estimate)
            + int(model_is_seasonal) * sum_persistence_seasonal
            + int(phi_estimate)
            + int(initial_needs_estimation)
            * (
                int(initial_level_estimate)
                + int(initial_trend_estimate)
                + sum_initial_seasonal_lags
            )
        )

    # ARIMA component
    arima_params = 0
    if arima_model:
        ar_orders_sum = sum(ar_orders) if ar_orders else 0
        ma_orders_sum = sum(ma_orders) if ma_orders else 0
        arima_params = (
            int(initial_needs_estimation) * initial_arima_number
            + int(ar_required) * int(ar_estimate) * ar_orders_sum
            + int(ma_required) * int(ma_estimate) * ma_orders_sum
        )

    # Xreg component
    xreg_params = 0
    if xreg_model:
        xreg_params = xreg_number * (
            int(initial_needs_xreg) * int(initial_xreg_estimate)
            + int(persistence_xreg_estimate)
        )

    # Total: 1 (scale) + ETS + ARIMA + xreg
    return 1 + ets_params + arima_params + xreg_params
