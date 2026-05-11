import numpy as np

from smooth.adam_general.core.creator import architector, creator, initialiser


def _run_two_stage_estimator(
    general_dict,
    model_type_dict,
    lags_dict,
    observations_dict,
    arima_dict,
    constant_dict,
    explanatory_dict,
    profiles_recent_table,
    profiles_recent_provided,
    persistence_dict,
    initials_dict,
    phi_dict,
    components_dict,
    occurrence_dict,
    multisteps=False,
    lb=None,
    ub=None,
    maxtime=None,
    print_level=0,
    maxeval=None,
    return_matrices=False,
    xtol_rel=1e-6,
    xtol_abs=1e-8,
    ftol_rel=1e-8,
    ftol_abs=0,
    algorithm="NLOPT_LN_NELDERMEAD",
    smoother="global",
    adam_ets: bool = False,
):
    """
    Internal function to handle two-stage initialization within estimator.

    This implements the two-stage procedure:
    1. Stage 1: Run with initial_type="complete" (backcasting) to get initial parameters
    2. Extract initial states from Stage 1's fitted mat_vt
    3. Stage 2: Run optimization with original "two-stage" using B_initial from Stage 1
    """
    from .estimator import estimator

    # Stage 1: Run with "complete" to get backcasted parameters
    stage1_initials = initials_dict.copy()
    stage1_initials["initial_type"] = "complete"
    # For Stage 1 (backcasting): use user's value if provided, otherwise default to 2
    if initials_dict.get("n_iterations_provided", False):
        stage1_initials["n_iterations"] = initials_dict["n_iterations"]
    else:
        stage1_initials["n_iterations"] = 2

    adam_estimated_s1 = estimator(
        general_dict=general_dict,
        model_type_dict=model_type_dict,
        lags_dict=lags_dict,
        observations_dict=observations_dict,
        arima_dict=arima_dict,
        constant_dict=constant_dict,
        explanatory_dict=explanatory_dict,
        profiles_recent_table=profiles_recent_table,
        profiles_recent_provided=profiles_recent_provided,
        persistence_dict=persistence_dict,
        initials_dict=stage1_initials,
        phi_dict=phi_dict,
        components_dict=components_dict,
        occurrence_dict=occurrence_dict,
        multisteps=multisteps,
        maxtime=maxtime,
        print_level=print_level,
        maxeval=maxeval,
        return_matrices=True,  # Need matrices to extract initial states
        xtol_rel=xtol_rel,
        xtol_abs=xtol_abs,
        ftol_rel=ftol_rel,
        ftol_abs=ftol_abs,
        algorithm=algorithm,
        smoother=smoother,
        adam_ets=adam_ets,
    )

    # Get B vector structure with "two-stage" (includes initial states in B)
    # We need to call architector/creator/initialiser to get proper B structure
    (
        model_type_dict_s2,
        components_dict_s2,
        lags_dict_s2,
        observations_dict_s2,
        profile_dict_s2,
        _,
    ) = architector(
        model_type_dict,
        lags_dict,
        observations_dict,
        arima_dict,
        explanatory_dict,
        constant_dict,
        profiles_recent_table,
        profiles_recent_provided,
        adam_ets,
    )

    adam_created_s2 = creator(
        model_type_dict_s2,
        lags_dict_s2,
        profile_dict_s2,
        observations_dict_s2,
        persistence_dict,
        initials_dict,  # Original "two-stage"
        arima_dict,
        constant_dict,
        phi_dict,
        components_dict_s2,
        explanatory_dict,
        smoother=smoother,
    )

    b_values = initialiser(
        model_type_dict=model_type_dict_s2,
        components_dict=components_dict_s2,
        lags_dict=lags_dict_s2,
        adam_created=adam_created_s2,
        persistence_checked=persistence_dict,
        initials_checked=initials_dict,  # Original "two-stage"
        arima_checked=arima_dict,
        constants_checked=constant_dict,
        explanatory_checked=explanatory_dict,
        observations_dict=observations_dict_s2,
        bounds=general_dict["bounds"],
        phi_dict=phi_dict,
        profile_dict=profile_dict_s2,
    )

    B = b_values["B"].copy()
    Bl = b_values["Bl"].copy()
    Bu = b_values["Bu"].copy()

    # Extract results from Stage 1
    B_stage1 = adam_estimated_s1["B"]

    # Calculate nParametersBack: number of persistence, phi, and ARMA parameters
    # (excluding initials, constant, shape) - matching R's adam.R lines 2518-2522
    #  IMPORTANT: Use model_type_dict_s2 (from architector) which has correct
    # model_is_trendy
    #  and model_is_seasonal flags, not the original model_type_dict which may have
    # stale values
    # from the parent "ZXZ" model during model selection.
    n_params_back = 0
    if model_type_dict_s2.get("ets_model", False):
        if persistence_dict.get("persistence_level_estimate", False):
            n_params_back += 1
        if model_type_dict_s2.get("model_is_trendy", False) and persistence_dict.get(
            "persistence_trend_estimate", False
        ):
            n_params_back += 1
        if model_type_dict_s2.get("model_is_seasonal", False):
            persistence_seasonal_estimate = persistence_dict.get(
                "persistence_seasonal_estimate", []
            )
            if isinstance(persistence_seasonal_estimate, list):
                n_params_back += sum(persistence_seasonal_estimate)
            elif persistence_seasonal_estimate:
                n_params_back += 1
        if phi_dict.get("phi_estimate", False):
            n_params_back += 1

    if explanatory_dict.get("xreg_model", False) and persistence_dict.get(
        "persistence_xreg_estimate", False
    ):
        xreg_parameters_persistence = explanatory_dict.get(
            "xreg_parameters_persistence", [0]
        )
        n_params_back += (
            max(xreg_parameters_persistence) if xreg_parameters_persistence else 0
        )

    if arima_dict.get("arima_model", False):
        ar_orders = arima_dict.get("ar_orders", [])
        ma_orders = arima_dict.get("ma_orders", [])
        if arima_dict.get("ar_estimate", False):
            n_params_back += sum(ar_orders) if ar_orders else 0
        if arima_dict.get("ma_estimate", False):
            n_params_back += sum(ma_orders) if ma_orders else 0

    # Copy persistence/phi/ARMA from Stage 1 to B
    if n_params_back > 0:
        B[:n_params_back] = B_stage1[:n_params_back]

    # Extract initial states from Stage 1's fitted mat_vt
    mat_vt_s1 = adam_estimated_s1["matrices"]["mat_vt"]
    lags_dict_s1 = adam_estimated_s1["lags_dict"]
    lags_model_s1 = lags_dict_s1["lags_model"]
    lags_model_max_s1 = lags_dict_s1["lags_model_max"]
    components_dict_s1 = adam_estimated_s1["components_dict"]

    initial_states = []
    current_row = 0

    # Level
    if initials_dict.get("initial_level_estimate", False):
        initial_states.append(mat_vt_s1[current_row, 0])
        current_row += 1
    elif model_type_dict.get("ets_model", False):
        current_row += 1

    # Trend
    if model_type_dict.get("model_is_trendy", False):
        if initials_dict.get("initial_trend_estimate", False):
            initial_states.append(mat_vt_s1[current_row, 0])
        current_row += 1

    # Seasonal
    if model_type_dict.get("model_is_seasonal", False):
        n_seasonal = components_dict_s1.get("components_number_ets_seasonal", 0)
        seasonal_estimate = initials_dict.get(
            "initial_seasonal_estimate", [False] * n_seasonal
        )
        if not isinstance(seasonal_estimate, list):
            seasonal_estimate = [seasonal_estimate] * n_seasonal

        for i in range(n_seasonal):
            lag = lags_model_s1[current_row] if current_row < len(lags_model_s1) else 1
            if seasonal_estimate[i] if i < len(seasonal_estimate) else False:
                start_idx = lags_model_max_s1 - lag
                full_seasonal = mat_vt_s1[
                    current_row, start_idx:lags_model_max_s1
                ].copy()

                # Renormalize
                season_type = model_type_dict.get("season_type", "N")
                if season_type == "A":
                    full_seasonal = full_seasonal - np.mean(full_seasonal)
                elif season_type == "M":
                    if np.all(full_seasonal > 0):
                        geo_mean = np.exp(np.mean(np.log(full_seasonal)))
                        full_seasonal = full_seasonal / geo_mean

                # Truncate to m-1
                initial_states.extend(full_seasonal[:-1].tolist())
            current_row += 1

    # ARIMA initials
    if arima_dict.get("arima_model", False):
        n_arima = initials_dict.get("initial_arima_number", 0)
        if initials_dict.get("initial_arima_estimate", False) and n_arima > 0:
            for i in range(n_arima):
                initial_states.append(mat_vt_s1[current_row + i, lags_model_max_s1 - 1])
            current_row += n_arima

    # xreg initials
    if explanatory_dict.get("xreg_model", False):
        xreg_number = explanatory_dict.get("xreg_number", 0)
        xreg_params_estimated = explanatory_dict.get("xreg_parameters_estimated", [])
        if initials_dict.get("initial_xreg_estimate", False) and xreg_number > 0:
            n_ets = components_dict_s1.get("components_number_ets", 0)
            n_arima_components = components_dict_s1.get("components_number_arima", 0)
            xreg_start_row = n_ets + n_arima_components
            xreg_initials_all = mat_vt_s1[
                xreg_start_row : xreg_start_row + xreg_number, lags_model_max_s1 - 1
            ]
            if xreg_params_estimated is not None and len(xreg_params_estimated) > 0:
                xreg_params_estimated_arr = np.array(xreg_params_estimated)
                xreg_initials_filtered = xreg_initials_all[
                    xreg_params_estimated_arr == 1
                ]
                initial_states.extend(xreg_initials_filtered.tolist())

    # Put extracted initials into B
    if len(initial_states) > 0:
        B[n_params_back : n_params_back + len(initial_states)] = initial_states

    # Handle constant
    if constant_dict.get("constant_estimate", False):
        n_ets = components_dict_s1.get("components_number_ets", 0)
        n_arima_components = components_dict_s1.get("components_number_arima", 0)
        xreg_number = (
            explanatory_dict.get("xreg_number", 0)
            if explanatory_dict.get("xreg_model", False)
            else 0
        )
        constant_idx = n_params_back + len(initial_states)
        if constant_idx < len(B):
            constant_row = n_ets + n_arima_components + xreg_number
            if constant_row < mat_vt_s1.shape[0]:
                constant_value = mat_vt_s1[constant_row, lags_model_max_s1 - 1]
                if not np.isnan(constant_value):
                    B[constant_idx] = constant_value

    # Handle other parameters like shape
    if general_dict.get("other_parameter_estimate", False):
        if len(B_stage1) > 0:
            B[-1] = abs(B_stage1[-1])

    # Adjust bounds
    Bl[np.isnan(Bl)] = -np.inf
    Bu[np.isnan(Bu)] = np.inf
    Bl[Bl > B] = -np.inf
    Bu[Bu < B] = np.inf

    # Stage 2: Run with original "two-stage" and B_initial from Stage 1
    return estimator(
        general_dict=general_dict,
        model_type_dict=model_type_dict,
        lags_dict=lags_dict,
        observations_dict=observations_dict,
        arima_dict=arima_dict,
        constant_dict=constant_dict,
        explanatory_dict=explanatory_dict,
        profiles_recent_table=profiles_recent_table,
        profiles_recent_provided=profiles_recent_provided,
        persistence_dict=persistence_dict,
        initials_dict=initials_dict,  # Original "two-stage"
        phi_dict=phi_dict,
        components_dict=components_dict,
        occurrence_dict=occurrence_dict,
        multisteps=multisteps,
        lb=Bl,
        ub=Bu,
        maxtime=maxtime,
        print_level=print_level,
        maxeval=maxeval,
        B_initial=B,  # Use B populated from Stage 1
        return_matrices=return_matrices,
        xtol_rel=xtol_rel,
        xtol_abs=xtol_abs,
        ftol_rel=ftol_rel,
        ftol_abs=ftol_abs,
        algorithm=algorithm,
        smoother=smoother,
    )
