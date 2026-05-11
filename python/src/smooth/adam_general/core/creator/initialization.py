import numpy as np

from smooth.adam_general.core.utils.utils import msdecompose


def _initialize_states(
    matrices,
    model_params,
    profiles_dict,
    observations_dict,
    initials_checked,
    arima_checked,
    explanatory_checked,
    constants_checked,
):
    """
    Initialize the state matrix with proper values.

    Args:
        matrices: Dictionary containing model matrices
        model_params: Dictionary containing model parameters
        profiles_dict: Dictionary containing profiles information
        observations_dict: Dictionary containing observation information
        initials_checked: Dictionary of initial values
        arima_checked: Dictionary of ARIMA parameters
        explanatory_checked: Dictionary of explanatory variables parameters
        constants_checked: Dictionary containing constant parameters

    Returns:
        Dict: Updated matrices dictionary
    """
    # Get matrices
    mat_vt = matrices["mat_vt"]

    # Get parameters
    profiles_recent_provided = model_params["profiles_recent_provided"]
    ets_model = model_params["ets_model"]
    lags_model_max = model_params["lags_model_max"]
    profiles_recent_table = model_params["profiles_recent_table"]
    # If recent profiles are not provided, initialize states
    if not profiles_recent_provided:
        # ETS model initialization
        if ets_model:
            mat_vt = _initialize_ets_states(
                mat_vt,
                model_params,
                initials_checked,
                explanatory_checked,
                observations_dict,
            )

        # ARIMA model initialization
        if arima_checked["arima_model"]:
            mat_vt = _initialize_arima_states(
                mat_vt, model_params, initials_checked, arima_checked, observations_dict
            )

        # Initialize explanatory variables
        if explanatory_checked["xreg_model"]:
            mat_vt = _initialize_xreg_states(
                mat_vt,
                model_params,
                initials_checked,
                explanatory_checked,
                arima_checked,
            )

        # Initialize constant if needed
        if constants_checked["constant_required"]:
            mat_vt = _initialize_constant(
                mat_vt,
                model_params,
                constants_checked,
                initials_checked,
                observations_dict,
                arima_checked,
                explanatory_checked,
                ets_model,
            )
    else:
        # If profiles are provided, use them directly
        mat_vt[:, 0:lags_model_max] = profiles_recent_table

    matrices["mat_vt"] = mat_vt
    return matrices


def _initialize_ets_states(
    mat_vt, model_params, initials_checked, explanatory_checked, observations_dict
):
    """
    Initialize ETS model states.

    Args:
        mat_vt: State matrix
        model_params: Dictionary containing model parameters
        initials_checked: Dictionary of initial values
        explanatory_checked: Dictionary of explanatory variables parameters
        observations_dict: Dictionary containing observation information

    Returns:
        np.ndarray: Updated state matrix
    """
    # Get parameters
    model_is_seasonal = model_params["model_is_seasonal"]
    model_is_trendy = model_params["model_is_trendy"]
    e_type = model_params["e_type"]
    lags_model = model_params["lags_model"]
    lags_model_max = model_params["lags_model_max"]
    components_number_ets_seasonal = model_params["components_number_ets_seasonal"]
    y_in_sample = model_params["y_in_sample"]
    obs_nonzero = observations_dict["obs_nonzero"]

    # If initials need to be estimated
    if initials_checked["initial_estimate"]:
        # For seasonal models
        if model_is_seasonal:
            # Handle seasonal model initialization
            if obs_nonzero >= lags_model_max * 2:
                mat_vt = _initialize_ets_seasonal_states_with_decomp(
                    mat_vt, model_params, initials_checked, explanatory_checked
                )

            else:
                mat_vt = _initialize_ets_seasonal_states_small_sample(
                    mat_vt, model_params, initials_checked, explanatory_checked
                )

        else:
            # Handle non-seasonal model initialization
            mat_vt = _initialize_ets_nonseasonal_states(
                mat_vt, model_params, initials_checked
            )
        # Handle special case for multiplicative models
        if (
            initials_checked["initial_level_estimate"]
            and e_type == "M"
            and mat_vt[0, lags_model_max - 1] == 0
        ):
            mat_vt[0, 0:lags_model_max] = np.mean(y_in_sample)

    # If initials are provided, use them directly
    elif (
        not initials_checked["initial_estimate"]
        and initials_checked["initial_type"] == "provided"
    ):
        j = 0
        mat_vt[j, 0:lags_model_max] = initials_checked["initial_level"]
        if model_is_trendy:
            j += 1
            mat_vt[j, 0:lags_model_max] = initials_checked["initial_trend"]
        if model_is_seasonal:
            for i in range(components_number_ets_seasonal):
                #  This is misaligned, but that's okay, because this goes directly to
                # profile_recent
                mat_vt[j + i, 0 : lags_model[j + i]] = initials_checked[
                    "initial_seasonal"
                ][i]

    return mat_vt


def _initialize_ets_seasonal_states_with_decomp(
    mat_vt, model_params, initials_checked, explanatory_checked
):
    """
    Initialize ETS seasonal model states using decomposition when sufficient data is
    available.

    Args:
        mat_vt: State matrix
        model_params: Dictionary containing model parameters
        initials_checked: Dictionary of initial values
        explanatory_checked: Dictionary of explanatory variables parameters

    Returns:
        np.ndarray: Updated state matrix
    """
    # Get parameters
    e_type = model_params["e_type"]
    t_type = model_params["t_type"]
    s_type = model_params["s_type"]
    lags = model_params["lags"]
    lags_model = model_params["lags_model"]
    lags_model_max = model_params["lags_model_max"]
    model_is_trendy = model_params["model_is_trendy"]
    components_number_ets_seasonal = model_params["components_number_ets_seasonal"]
    y_in_sample = model_params["y_in_sample"]
    ot_logical = model_params["ot_logical"]

    # Run both additive and multiplicative decompositions (matching R lines 892-898)
    smoother = model_params["smoother"]
    y_decomposition_additive = msdecompose(
        y_in_sample.ravel(),
        [lag for lag in lags if lag != 1],
        type="additive",
        smoother=smoother,
    )

    if any(x == "M" for x in [e_type, t_type, s_type]):
        y_decomposition_multiplicative = msdecompose(
            y_in_sample.ravel(),
            [lag for lag in lags if lag != 1],
            type="multiplicative",
            smoother=smoother,
        )
    else:
        y_decomposition_multiplicative = None

    # If either e_type or s_type are multiplicative, use multiplicative decomposition
    # This is needed for the correct seasonal indices (matching R lines 902-906)
    decomposition_type = (
        "multiplicative" if any(x == "M" for x in [e_type, s_type]) else "additive"
    )
    y_decomposition = (
        y_decomposition_multiplicative
        if decomposition_type == "multiplicative"
        else y_decomposition_additive
    )

    j = 0

    # Initialize level (matching R lines 909-919)
    if initials_checked["initial_level_estimate"]:
        # If there's a trend, use the initial from decomposition
        if model_is_trendy:
            #  Use multiplicative decomposition initial for M trend, additive for A (R
            # lines 913-914)
            if t_type == "M":
                mat_vt[j, 0:lags_model_max] = y_decomposition_multiplicative["initial"][
                    "nonseasonal"
                ]["level"]
            else:
                mat_vt[j, 0:lags_model_max] = y_decomposition_additive["initial"][
                    "nonseasonal"
                ]["level"]
        # If not trendy, use the global mean
        else:
            mat_vt[j, 0:lags_model_max] = np.mean(y_in_sample[ot_logical])

        if explanatory_checked["xreg_model"]:
            xreg_init = (
                explanatory_checked["xreg_model_initials"][0]
                if e_type == "A"
                or explanatory_checked["xreg_model_initials"][1] is None
                else explanatory_checked["xreg_model_initials"][1]
            )
            xreg_coefs = xreg_init["initial_xreg"].ravel()
            if e_type == "A":
                mat_vt[j, 0:lags_model_max] -= np.dot(
                    xreg_coefs, explanatory_checked["xreg_data"][0]
                )
            else:
                mat_vt[j, 0:lags_model_max] /= np.exp(
                    np.dot(xreg_coefs, explanatory_checked["xreg_data"][0])
                )
    else:
        mat_vt[j, 0:lags_model_max] = initials_checked["initial_level"]
    j += 1

    # Initialize trend if needed (matching R lines 936-960)
    if model_is_trendy:
        if initials_checked["initial_trend_estimate"]:
            #  Use additive decomposition initial for A trend, multiplicative for M (R
            # lines 938-949)
            if t_type == "A":
                mat_vt[j, 0:lags_model_max] = y_decomposition_additive["initial"][
                    "nonseasonal"
                ]["trend"]
                if s_type == "M":
                    #  If the initial trend is higher than the lowest value, initialise
                    # with zero.
                    #  This is a failsafe mechanism for the mixed models (R lines
                    # 940-945)
                    if mat_vt[j, 0] < 0 and abs(mat_vt[j, 0]) > min(
                        abs(y_in_sample[ot_logical])
                    ):
                        mat_vt[j, 0:lags_model_max] = 0
            elif t_type == "M":
                mat_vt[j, 0:lags_model_max] = y_decomposition_multiplicative["initial"][
                    "nonseasonal"
                ]["trend"]

                #  This is a failsafe for multiplicative trend models with negative
                # initial level (R lines 952-954)
                if np.any(mat_vt[0, 0:lags_model_max] < 0):
                    mat_vt[0, 0:lags_model_max] = y_in_sample[ot_logical][0]
        else:
            mat_vt[j, 0:lags_model_max] = initials_checked["initial_trend"]
        j += 1

    # Initialize seasonal components (matching R lines 963-1006)
    # For pure models use stuff as is
    if (
        all(x == "A" for x in [e_type, s_type])
        or all(x == "M" for x in [e_type, s_type])
        or (e_type == "A" and s_type == "M")
    ):
        for i in range(components_number_ets_seasonal):
            if initials_checked["initial_seasonal_estimate"]:
                #  Use initial["seasonal"][i] which already contains the correct number
                # of values (R line 968)
                mat_vt[i + j, 0 : lags_model[i + j]] = y_decomposition["initial"][
                    "seasonal"
                ][i]
                # Renormalise the initial seasons (R lines 975-984)
                if s_type == "A":
                    mat_vt[i + j, 0 : lags_model[i + j]] -= np.mean(
                        mat_vt[i + j, 0 : lags_model[i + j]]
                    )
                else:
                    mat_vt[i + j, 0 : lags_model[i + j]] /= np.exp(
                        np.mean(np.log(mat_vt[i + j, 0 : lags_model[i + j]]))
                    )
            else:
                mat_vt[i + j, 0 : lags_model[i + j]] = initials_checked[
                    "initial_seasonal"
                ][i]
    # For mixed models use a different set of initials (R lines 987-1006)
    elif e_type == "M" and s_type == "A":
        for i in range(components_number_ets_seasonal):
            if initials_checked["initial_seasonal_estimate"]:
                # Use initial["seasonal"][i] and apply log transformation (R line 991)
                mat_vt[i + j, 0 : lags_model[i + j]] = np.log(
                    y_decomposition["initial"]["seasonal"][i]
                ) * np.min(y_in_sample[ot_logical])
                # Renormalise the initial seasons
                if s_type == "A":
                    mat_vt[i + j, 0 : lags_model[i + j]] -= np.mean(
                        mat_vt[i + j, 0 : lags_model[i + j]]
                    )
                else:
                    mat_vt[i + j, 0 : lags_model[i + j]] /= np.exp(
                        np.mean(np.log(mat_vt[i + j, 0 : lags_model[i + j]]))
                    )
            else:
                mat_vt[i + j, 0 : lags_model[i + j]] = initials_checked[
                    "initial_seasonal"
                ][i]

    # Failsafe in case negatives were produced
    if e_type == "M" and mat_vt[0, 0] <= 0:
        mat_vt[0, 0:lags_model_max] = y_in_sample[0]

    return mat_vt


def _initialize_ets_seasonal_states_small_sample(
    mat_vt, model_params, initials_checked, explanatory_checked
):
    """
    Initialize ETS seasonal model states when limited data is available.

    Args:
        mat_vt: State matrix
        model_params: Dictionary containing model parameters
        initials_checked: Dictionary of initial values
        explanatory_checked: Dictionary of explanatory variables parameters

    Returns:
        np.ndarray: Updated state matrix
    """
    # Get parameters
    e_type = model_params["e_type"]
    t_type = model_params["t_type"]
    s_type = model_params["s_type"]
    lags_model = model_params["lags_model"]
    lags_model_max = model_params["lags_model_max"]
    model_is_trendy = model_params["model_is_trendy"]
    components_number_ets_seasonal = model_params["components_number_ets_seasonal"]
    y_in_sample = model_params["y_in_sample"]
    ot_logical = model_params["ot_logical"]

    # If either e_type or s_type are multiplicative, do multiplicative decomposition
    j = 0
    # level
    if initials_checked["initial_level_estimate"]:
        mat_vt[j, 0:lags_model_max] = np.mean(y_in_sample[0:lags_model_max])
        if explanatory_checked["xreg_model"]:
            xreg_init = (
                explanatory_checked["xreg_model_initials"][0]
                if e_type == "A"
                or explanatory_checked["xreg_model_initials"][1] is None
                else explanatory_checked["xreg_model_initials"][1]
            )
            xreg_coefs = xreg_init["initial_xreg"].ravel()
            if e_type == "A":
                mat_vt[j, 0:lags_model_max] -= np.dot(
                    xreg_coefs, explanatory_checked["xreg_data"][0]
                )
            else:
                mat_vt[j, 0:lags_model_max] /= np.exp(
                    np.dot(xreg_coefs, explanatory_checked["xreg_data"][0])
                )
    else:
        mat_vt[j, 0:lags_model_max] = initials_checked["initial_level"]
    j += 1

    # Initialize trend if needed
    if model_is_trendy:
        if initials_checked["initial_trend_estimate"]:
            if t_type == "A":
                # trend
                mat_vt[j, 0:lags_model_max] = y_in_sample[1] - y_in_sample[0]
            elif t_type == "M":
                if initials_checked["initial_level_estimate"]:
                    # level fix
                    mat_vt[j - 1, 0:lags_model_max] = np.exp(
                        np.mean(np.log(y_in_sample[ot_logical][0:lags_model_max]))
                    )
                # trend
                mat_vt[j, 0:lags_model_max] = y_in_sample[1] / y_in_sample[0]

        # Safety check for multiplicative trend
        if t_type == "M" and np.any(mat_vt[j, 0:lags_model_max] > 1.1):
            mat_vt[j, 0:lags_model_max] = 1
        else:
            mat_vt[j, 0:lags_model_max] = initials_checked["initial_trend"]
        j += 1

    # Initialize seasonal components
    if s_type == "A":
        for i in range(components_number_ets_seasonal):
            if initials_checked["initial_seasonal_estimate"]:
                mat_vt[i + j - 1, 0 : lags_model[i + j - 1]] = (
                    y_in_sample[0 : lags_model[i + j - 1]] - mat_vt[0, 0]
                )
                # Renormalise the initial seasons
                mat_vt[i + j - 1, 0 : lags_model[i + j - 1]] -= np.mean(
                    mat_vt[i + j - 1, 0 : lags_model[i + j - 1]]
                )
            else:
                mat_vt[i + j - 1, 0 : lags_model[i + j - 1]] = initials_checked[
                    "initial_seasonal"
                ][i]
    # For mixed models use a different set of initials
    else:
        for i in range(components_number_ets_seasonal):
            if initials_checked["initial_seasonal_estimate"]:
                # abs() is needed for mixed ETS+ARIMA
                mat_vt[i + j - 1, 0 : lags_model[i + j - 1]] = y_in_sample[
                    0 : lags_model[i + j - 1]
                ] / abs(mat_vt[0, 0])
                # Renormalise the initial seasons
                mat_vt[i + j - 1, 0 : lags_model[i + j - 1]] /= np.exp(
                    np.mean(np.log(mat_vt[i + j - 1, 0 : lags_model[i + j - 1]]))
                )
            else:
                mat_vt[i + j - 1, 0 : lags_model[i + j - 1]] = initials_checked[
                    "initial_seasonal"
                ][i]

    return mat_vt


def _initialize_ets_nonseasonal_states(mat_vt, model_params, initials_checked):
    """
    Initialize ETS non-seasonal model states.

    Args:
        mat_vt: State matrix
        model_params: Dictionary containing model parameters
        initials_checked: Dictionary of initial values

    Returns:
        np.ndarray: Updated state matrix
    """
    # Get parameters
    lags_model_max = model_params["lags_model_max"]
    model_is_trendy = model_params["model_is_trendy"]
    e_type = model_params["e_type"]
    t_type = model_params["t_type"]
    y_in_sample = model_params["y_in_sample"]
    ot_logical = model_params["ot_logical"]

    # This decomposition does not produce seasonal component
    # Run both additive and multiplicative decompositions (matching R lines 1021-1028)
    smoother = model_params["smoother"]
    y_decomposition_additive = msdecompose(
        y_in_sample.ravel(),
        lags=[1],
        type="additive",
        smoother=smoother,
    )

    if any(x == "M" for x in [e_type, t_type]):
        y_decomposition_multiplicative = msdecompose(
            y_in_sample.ravel(),
            lags=[1],
            type="multiplicative",
            smoother=smoother,
        )
    else:
        y_decomposition_multiplicative = None

    # level (matching R lines 1030-1044)
    if initials_checked["initial_level_estimate"]:
        # If there's a trend, use the initial from decomposition (R lines 1032-1036)
        if model_is_trendy:
            if t_type == "M":
                mat_vt[0, :lags_model_max] = y_decomposition_multiplicative["initial"][
                    "nonseasonal"
                ]["level"]
            else:
                mat_vt[0, :lags_model_max] = y_decomposition_additive["initial"][
                    "nonseasonal"
                ]["level"]
        # If not trendy, use the global mean
        else:
            mat_vt[0, :lags_model_max] = np.mean(y_in_sample[ot_logical])
    else:
        mat_vt[0, :lags_model_max] = initials_checked["initial_level"]

    # trend (matching R lines 1046-1054)
    if model_is_trendy:
        if initials_checked["initial_trend_estimate"]:
            if t_type == "A":
                mat_vt[1, 0:lags_model_max] = y_decomposition_additive["initial"][
                    "nonseasonal"
                ]["trend"]
            else:  # t_type == "M"
                mat_vt[1, 0:lags_model_max] = y_decomposition_multiplicative["initial"][
                    "nonseasonal"
                ]["trend"]
        else:
            mat_vt[1, 0:lags_model_max] = initials_checked["initial_trend"]

    # Failsafe in case negatives were produced
    if e_type == "M" and mat_vt[0, 0] <= 0:
        mat_vt[0, 0:lags_model_max] = y_in_sample[0]

    return mat_vt


def _initialize_arima_states(
    mat_vt, model_params, initials_checked, arima_checked, observations_dict
):
    """
    Initialize ARIMA model states.

    Args:
        mat_vt: State matrix
        model_params: Dictionary containing model parameters
        initials_checked: Dictionary of initial values
        arima_checked: Dictionary of ARIMA parameters
        observations_dict: Dictionary containing observation information

    Returns:
        np.ndarray: Updated state matrix
    """
    # Get parameters
    components_number_ets = model_params["components_number_ets"]
    components_number_arima = model_params["components_number_arima"]
    e_type = model_params["e_type"]
    y_in_sample = model_params["y_in_sample"]
    ot_logical = model_params["ot_logical"]

    if initials_checked["initial_arima_estimate"]:
        mat_vt[
            components_number_ets : components_number_ets + components_number_arima,
            0 : initials_checked["initial_arima_number"],
        ] = 0 if e_type == "A" else 1

        # Use original user-specified lags (not ETS-only lags which may be empty
        # for pure ARIMA models) — mirrors R's use of lags in adam.R:1091-1108
        lags_original = model_params.get("lags_original", [1]) or [1]
        obs_in_sample = model_params["obs_in_sample"]

        has_seasonal = any(lag > 1 for lag in lags_original)
        if has_seasonal and obs_in_sample > max(lags_original) * 2:
            # R uses the full time-varying seasonal series (T values) from msdecompose
            # (adam.R lines 1092-1097: tail(msdecompose(...)$seasonal, 1)[[1]]).
            # Using ["initial"]["seasonal"][-1] (12 values, static) instead of
            # ["seasonal"][-1] (T values, time-varying) gives wrong seed row values
            # for columns 12..initial_arima_number-1.
            y_decomposition = msdecompose(
                y_in_sample,
                [lag for lag in lags_original if lag != 1],
                type="additive" if e_type == "A" else "multiplicative",
                smoother=model_params["smoother"],
            )["seasonal"][-1]
        elif has_seasonal:
            y_decomposition = y_in_sample[ot_logical][:obs_in_sample]
        else:
            y_decomposition = (
                np.mean(np.diff(y_in_sample[ot_logical]))
                if e_type == "A"
                else np.exp(np.mean(np.diff(np.log(y_in_sample[ot_logical]))))
            )

        # Tile using max of original lags (R: ceiling(initialArimaNumber/max(lags)))
        max_arima_lag = max(lags_original) if has_seasonal else 1
        mat_vt[
            components_number_ets + components_number_arima - 1,
            0 : initials_checked["initial_arima_number"],
        ] = np.tile(
            y_decomposition,
            int(np.ceil(initials_checked["initial_arima_number"] / max_arima_lag)),
        )[: initials_checked["initial_arima_number"]]
    else:
        mat_vt[
            components_number_ets : components_number_ets + components_number_arima,
            0 : initials_checked["initial_arima_number"],
        ] = 0 if e_type == "A" else 1
        mat_vt[
            components_number_ets + components_number_arima - 1,
            0 : initials_checked["initial_arima_number"],
        ] = initials_checked["initial_arima"][
            : initials_checked["initial_arima_number"]
        ]

    return mat_vt


def _initialize_xreg_states(
    mat_vt, model_params, initials_checked, explanatory_checked, arima_checked
):
    """
    Initialize explanatory variables states.

    Args:
        mat_vt: State matrix
        model_params: Dictionary containing model parameters
        initials_checked: Dictionary of initial values
        explanatory_checked: Dictionary of explanatory variables parameters
        arima_checked: Dictionary of ARIMA parameters

    Returns:
        np.ndarray: Updated state matrix
    """
    # Get parameters
    components_number_ets = model_params["components_number_ets"]
    components_number_arima = model_params["components_number_arima"]
    e_type = model_params["e_type"]
    lags_model_max = model_params["lags_model_max"]

    if (
        e_type == "A"
        or initials_checked["initial_xreg_provided"]
        or explanatory_checked["xreg_model_initials"][1] is None
    ):
        mat_vt[
            components_number_ets + components_number_arima : components_number_ets
            + components_number_arima
            + explanatory_checked["xreg_number"],
            0:lags_model_max,
        ] = explanatory_checked["xreg_model_initials"][0]["initial_xreg"]
    else:
        mat_vt[
            components_number_ets + components_number_arima : components_number_ets
            + components_number_arima
            + explanatory_checked["xreg_number"],
            0:lags_model_max,
        ] = explanatory_checked["xreg_model_initials"][1]["initial_xreg"]

    return mat_vt


def _initialize_constant(
    mat_vt,
    model_params,
    constants_checked,
    initials_checked,
    observations_dict,
    arima_checked,
    explanatory_checked,
    ets_model,
):
    """
    Initialize constant term if required.

    Args:
        mat_vt: State matrix
        model_params: Dictionary containing model parameters
        constants_checked: Dictionary containing constant parameters
        initials_checked: Dictionary of initial values
        observations_dict: Dictionary containing observation information
        arima_checked: Dictionary of ARIMA parameters
        explanatory_checked: Dictionary of explanatory variables parameters
        ets_model: Boolean indicating if ETS model is used

    Returns:
        np.ndarray: Updated state matrix
    """
    # Get parameters
    components_number_ets = model_params["components_number_ets"]
    components_number_arima = model_params["components_number_arima"]
    e_type = model_params["e_type"]
    lags_model_max = model_params["lags_model_max"]
    y_in_sample = model_params["y_in_sample"]
    ot_logical = model_params["ot_logical"]

    if constants_checked["constant_estimate"]:
        # Add the mean of data
        if sum(arima_checked["i_orders"] or []) == 0 and not ets_model:
            mat_vt[
                components_number_ets
                + components_number_arima
                + explanatory_checked["xreg_number"],
                :,
            ] = np.mean(y_in_sample[ot_logical])
        else:
            if e_type == "A":
                mat_vt[
                    components_number_ets
                    + components_number_arima
                    + explanatory_checked["xreg_number"],
                    :,
                ] = np.mean(np.diff(y_in_sample[ot_logical]))
            else:
                mat_vt[
                    components_number_ets
                    + components_number_arima
                    + explanatory_checked["xreg_number"],
                    :,
                ] = np.exp(np.mean(np.diff(np.log(y_in_sample[ot_logical]))))
    else:
        mat_vt[
            components_number_ets
            + components_number_arima
            + explanatory_checked["xreg_number"],
            :,
        ] = constants_checked["constant_value"]

    # If ETS model is used, change the initial level
    if ets_model and initials_checked["initial_level_estimate"]:
        if e_type == "A":
            mat_vt[0, 0:lags_model_max] -= mat_vt[
                components_number_ets
                + components_number_arima
                + explanatory_checked["xreg_number"],
                0,
            ]
        else:
            mat_vt[0, 0:lags_model_max] /= mat_vt[
                components_number_ets
                + components_number_arima
                + explanatory_checked["xreg_number"],
                0,
            ]

    # If ARIMA is done, debias states
    if arima_checked["arima_model"] and initials_checked["initial_arima_estimate"]:
        if e_type == "A":
            mat_vt[
                components_number_ets : components_number_ets + components_number_arima,
                0 : initials_checked["initial_arima_number"],
            ] -= mat_vt[
                components_number_ets
                + components_number_arima
                + explanatory_checked["xreg_number"],
                0,
            ]
        else:
            mat_vt[
                components_number_ets : components_number_ets + components_number_arima,
                0 : initials_checked["initial_arima_number"],
            ] /= mat_vt[
                components_number_ets
                + components_number_arima
                + explanatory_checked["xreg_number"],
                0,
            ]

    return mat_vt
