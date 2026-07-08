import numpy as np


def _process_initial_values(
    model_type_dict,
    lags_dict,
    matrices_dict,
    initials_checked,
    arima_checked,
    explanatory_checked,
    components_dict,
):
    """
    Process initial values for the model.

    Parameters
    ----------
    model_type_dict : dict
        Model type specification
    lags_dict : dict
        Lags information
    matrices_dict : dict
        Model matrices
    initials_checked : dict
        Initial values
    arima_checked : dict
        ARIMA components specification
    explanatory_checked : dict
        Explanatory variables specification
    components_dict : dict
        Components information

    Returns
    -------
    tuple
        initial_value, initial_value_ets, initial_value_names, initial_estimated
    """
    # Initial values to return
    initial_value = [None] * (
        model_type_dict["ets_model"]
        * (
            1
            + model_type_dict["model_is_trendy"]
            + model_type_dict["model_is_seasonal"]
        )
        + arima_checked["arima_model"]
        + explanatory_checked["xreg_model"]
    )
    initial_value_ets = [None] * (
        model_type_dict["ets_model"] * len(lags_dict["lags_model"])
    )
    initial_value_names = [""] * (
        model_type_dict["ets_model"]
        * (
            1
            + model_type_dict["model_is_trendy"]
            + model_type_dict["model_is_seasonal"]
        )
        + arima_checked["arima_model"]
        + explanatory_checked["xreg_model"]
    )

    # The vector that defines what was estimated in the model
    initial_estimated = [False] * (
        model_type_dict["ets_model"]
        * (
            1
            + model_type_dict["model_is_trendy"]
            + model_type_dict["model_is_seasonal"]
            * components_dict["components_number_ets_seasonal"]
        )
        + arima_checked["arima_model"]
        + explanatory_checked["xreg_model"]
    )

    # Write down the initials of ETS
    if model_type_dict["ets_model"]:
        # Write down level, trend and seasonal
        for i in range(len(lags_dict["lags_model"])):
            # In case of level / trend, we want to get the very first value
            if lags_dict["lags_model"][i] == 1:
                initial_value_ets[i] = matrices_dict["mat_vt"][
                    i, : lags_dict["lags_model_max"]
                ][0]
            #  In cases of seasonal components, they should be at the end of the
            # pre-heat period
            #  Only extract lag-1 values (the last one is the normalized value computed
            # from others)
            else:
                start_idx = lags_dict["lags_model_max"] - lags_dict["lags_model"][i]
                seasonal_full = matrices_dict["mat_vt"][
                    i, start_idx : lags_dict["lags_model_max"]
                ].copy()

                # Renormalise seasonal initials so additive components sum to 0
                # and multiplicative components have geometric mean 1.
                if np.any(~np.isnan(seasonal_full)):
                    if model_type_dict["season_type"] == "A":
                        seasonal_full = seasonal_full - np.nanmean(seasonal_full)
                    elif model_type_dict["season_type"] == "M":
                        positive_vals = seasonal_full.copy()
                        positive_vals[positive_vals <= 0] = np.nan
                        if np.all(np.isnan(positive_vals)):
                            geo_mean = 1.0
                        else:
                            geo_mean = np.exp(np.nanmean(np.log(positive_vals)))
                            if np.isnan(geo_mean) or geo_mean == 0:
                                geo_mean = 1.0
                        seasonal_full = seasonal_full / geo_mean

                initial_value_ets[i] = seasonal_full[:-1]

        j = 0
        # Write down level in the final list
        initial_estimated[j] = initials_checked["initial_level_estimate"]
        initial_value[j] = initial_value_ets[j]
        initial_value_names[j] = "level"

        if model_type_dict["model_is_trendy"]:
            j = 1
            initial_estimated[j] = initials_checked["initial_trend_estimate"]
            # Write down trend in the final list
            initial_value[j] = initial_value_ets[j]
            # Remove the trend from ETS list
            initial_value_ets[j] = None
            initial_value_names[j] = "trend"

            # Write down the initial seasonals
            if model_type_dict["model_is_seasonal"]:
                #  Convert initial_seasonal_estimate to list if it's a boolean (for
                # single seasonality)
                if isinstance(initials_checked["initial_seasonal_estimate"], bool):
                    seasonal_estimate_list = [
                        initials_checked["initial_seasonal_estimate"]
                    ] * components_dict["components_number_ets_seasonal"]
                else:
                    seasonal_estimate_list = initials_checked[
                        "initial_seasonal_estimate"
                    ]

                initial_estimated[
                    j + 1 : j + 1 + components_dict["components_number_ets_seasonal"]
                ] = seasonal_estimate_list
                # Remove the level from ETS list
                initial_value_ets[0] = None
                j += 1
                if len(seasonal_estimate_list) > 1:
                    initial_value[j] = [x for x in initial_value_ets if x is not None]
                    initial_value_names[j] = "seasonal"
                    for k in range(components_dict["components_number_ets_seasonal"]):
                        initial_estimated[j + k] = f"seasonal{k + 1}"
                else:
                    initial_value[j] = next(
                        x for x in initial_value_ets if x is not None
                    )
                    initial_value_names[j] = "seasonal"
                    initial_estimated[j] = "seasonal"

    # Write down the ARIMA initials
    if arima_checked["arima_model"]:
        j += 1
        initial_estimated[j] = initials_checked["initial_arima_estimate"]
        if initials_checked["initial_arima_estimate"]:
            initial_value[j] = matrices_dict["mat_vt"][
                components_dict["components_number_ets"]
                + components_dict.get("components_number_arima", 0)
                - 1,
                : initials_checked["initial_arima_number"],
            ]
        else:
            initial_value[j] = initials_checked["initial_arima"]
        initial_value_names[j] = "arima"
        initial_estimated[j] = "arima"

    # Set names for initial values
    initial_value = {
        name: value for name, value in zip(initial_value_names, initial_value) if name
    }

    return initial_value, initial_value_ets, initial_value_names, initial_estimated
