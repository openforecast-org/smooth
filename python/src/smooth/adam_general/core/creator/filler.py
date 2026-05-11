import numpy as np

from smooth.adam_general.core.utils.polynomials import adam_polynomialiser


def filler(
    B,
    model_type_dict,
    components_dict,
    lags_dict,
    matrices_dict,
    persistence_checked,
    initials_checked,
    arima_checked,
    explanatory_checked,
    phi_dict,
    constants_checked,
    adam_cpp=None,
):
    """
    Fill state-space matrices with parameter values from optimization vector B.

    This is the **critical bridge function** between the optimizer and the model. During
    each optimization iteration, ``filler()`` is called by ``CF()`` to populate the
    state-space matrices (mat_vt, mat_wt, mat_f, vec_g) with the current parameter
    values
    from the optimization vector B.

    The function extracts parameters from B in a specific order and places them into the
    appropriate matrix locations. This must perfectly match the parameter ordering
    defined
    by ``initialiser()``.

    **Parameter Extraction Order from B**:

    1. **ETS Persistence** (α, β, γ) → vec_g
    2. **Damping** (φ) → mat_wt and mat_f
    3. **Initial States** (l₀, b₀, s₀, ARIMA initials) → mat_vt (first max_lag columns)
    4. **ARIMA Coefficients** (AR, MA) → Converted to polynomials, stored in
    arima_polynomials
    5. **Regressor Coefficients** → mat_vt (regressor state rows)
    6. **Constant** → mat_vt (constant row)

    Parameters
    ----------
    B : numpy.ndarray
        Parameter vector from optimizer containing all estimated parameters in order:
        [persistence, phi, initials, AR, MA, xreg, constant]

    model_type_dict : dict
        Model specification (ets_model, arima_model, trend_type, etc.)

    components_dict : dict
        Component counts (components_number_ets, components_number_arima, etc.)

    lags_dict : dict
        Lag structure (lags, lags_model, lags_model_max, lags_model_seasonal)

    matrices_dict : dict
        State-space matrices to be filled (modified in-place):

        - 'mat_vt': State vector (initial values filled)
        - 'mat_wt': Measurement matrix (damping filled)
        - 'mat_f': Transition matrix (damping filled)
        - 'vec_g': Persistence vector (smoothing parameters filled)

    persistence_checked : dict
        Persistence specification indicating which parameters are estimated

    initials_checked : dict
        Initial values specification and estimation flags

    arima_checked : dict
        ARIMA specification with AR/MA orders and estimation flags

    explanatory_checked : dict
        External regressors specification

    phi_dict : dict
        Damping parameter specification

    constants_checked : dict
        Constant term specification

    Returns
    -------
    dict
        Dictionary containing:

        - **'mat_vt'**: Updated state vector matrix
        - **'mat_wt'**: Updated measurement matrix
        - **'mat_f'**: Updated transition matrix
        - **'vec_g'**: Updated persistence vector
        - **'arimaPolynomials'**: Dict with 'arPolynomial' and 'maPolynomial' (if ARIMA)

    Notes
    -----
    **Critical Performance Function**:

    This function is called thousands of times during optimization (once per CF
    evaluation).
    It must be fast and correct. Any indexing errors will cause cryptic optimization
    failures.

    **In-place Modification**:

    The matrices in matrices_dict are modified **in-place**. A copy should be made
    before
    calling if the original needs to be preserved.

    **Parameter Indexing**:

    The variable ``j`` tracks position in B. It advances as parameters are extracted.
    The order must exactly match ``initialiser()``'s parameter packing.

    **ARIMA Handling**:

    AR and MA coefficients are converted to polynomial form via
    ``adam_polynomialiser()``,
    which returns companion matrices for the state-space representation.

    See Also
    --------
    initialiser : Creates initial B vector - must match filler's extraction order
    CF : Cost function that calls filler during optimization
    creator : Creates initial matrices that filler updates

    Examples
    --------
    Fill matrices during optimization::

        >>> B = np.array([0.3, 0.1, 100, 5])  # [alpha, beta, l0, b0]
        >>> result = filler(
        ...     B=B,
        ...     model_type_dict={'ets_model': True, 'model_is_trendy': True, ...},
        ...     components_dict={'components_number_ets': 2, ...},
        ...     lags_dict={'lags_model_max': 1, ...},
        ...     matrices_dict={'mat_vt': mat_vt, 'mat_wt': mat_wt,
        ...                    'mat_f': mat_f, 'vec_g': vec_g},
        ...     persistence_checked={'persistence_estimate': True,
        ...                          'persistence_level_estimate': True,
        ...                          'persistence_trend_estimate': True, ...},
        ...     initials_checked={'initial_level_estimate': True,
        ...                       'initial_trend_estimate': True, ...},
        ...     arima_checked={'arima_model': False, ...},
        ...     explanatory_checked={'xreg_model': False, ...},
        ...     phi_dict={'phi_estimate': False, ...},
        ...     constants_checked={'constant_required': False, ...}
        ... )
        >>> print(result['vec_g'])  # [0.3, 0.1] - alpha and beta filled
        >>> print(result['mat_vt'][:, 0])  # [100, 5] - initial level and trend filled
    """
    j = 0
    # Fill in persistence
    if persistence_checked["persistence_estimate"]:
        # Persistence of ETS
        if model_type_dict["ets_model"]:
            i = 0
            # alpha
            if persistence_checked["persistence_level_estimate"]:
                j += 1
                matrices_dict["vec_g"][i] = B[j - 1]
            # beta
            if model_type_dict["model_is_trendy"]:
                i = 1
                if persistence_checked["persistence_trend_estimate"]:
                    j += 1
                    matrices_dict["vec_g"][i] = B[j - 1]

            # gamma1, gamma2, ...
            if model_type_dict["model_is_seasonal"]:
                if any(persistence_checked["persistence_seasonal_estimate"]):
                    seasonal_indices = (
                        i
                        + np.where(
                            persistence_checked["persistence_seasonal_estimate"]
                        )[0]
                        + 1
                    )
                    n_seasonal_to_estimate = sum(
                        persistence_checked["persistence_seasonal_estimate"]
                    )
                    matrices_dict["vec_g"][seasonal_indices, 0] = B[
                        j : j + n_seasonal_to_estimate
                    ]
                    j += n_seasonal_to_estimate
                i = components_dict["components_number_ets"] - 1

        # Persistence of xreg (delta parameters for adapt mode)
        if (
            explanatory_checked["xreg_model"]
            and persistence_checked["persistence_xreg_estimate"]
        ):
            xreg_persistence_number = max(
                explanatory_checked["xreg_parameters_persistence"]
            )
            xreg_start = (
                components_dict["components_number_ets"]
                + components_dict["components_number_arima"]
            )
            xreg_values = B[j : j + xreg_persistence_number][
                np.array(explanatory_checked["xreg_parameters_persistence"]) - 1
            ]
            matrices_dict["vec_g"][xreg_start : xreg_start + len(xreg_values), 0] = (
                xreg_values
            )
            j += xreg_persistence_number

    # Damping parameter
    if model_type_dict["ets_model"] and phi_dict["phi_estimate"]:
        phi_dict["phi"] = B[j]  # Update phi_dict with estimated value
        j += 1
        matrices_dict["mat_wt"][:, 1] = B[j - 1]
        matrices_dict["mat_f"][0:2, 1] = B[j - 1]

    # ARMA parameters - R lines 1377-1401
    if arima_checked["arima_model"] and adam_cpp is not None:
        # Calculate number of ARMA parameters to extract from B
        n_ar_params = (
            sum(arima_checked["ar_orders"]) if arima_checked["ar_estimate"] else 0
        )
        n_ma_params = (
            sum(arima_checked["ma_orders"]) if arima_checked["ma_estimate"] else 0
        )
        n_arma_params = n_ar_params + n_ma_params

        # Call the function returning ARI and MA polynomials - R line 1383-1385
        # adamCpp$polynomialise(B[j+1:sum(...)], arOrders, iOrders, maOrders, ...)
        arima_polynomials = adam_polynomialiser(
            adam_cpp,
            B[j : j + n_arma_params] if n_arma_params > 0 else np.array([0.0]),
            arima_checked["ar_orders"],
            arima_checked["i_orders"],
            arima_checked["ma_orders"],
            arima_checked["ar_estimate"],
            arima_checked["ma_estimate"],
            arima_checked["arma_parameters"]
            if arima_checked["arma_parameters"]
            else [],
            lags_dict.get("lags_original", lags_dict["lags"]),
        )
        # Alias: polynomialiser returns snake_case; bounds checking uses camelCase
        arima_polynomials["ariPolynomial"] = arima_polynomials["ari_polynomial"]
        arima_polynomials["arPolynomial"] = arima_polynomials["ar_polynomial"]
        arima_polynomials["maPolynomial"] = arima_polynomials["ma_polynomial"]

        # Get array views for indexing
        non_zero_ari = arima_checked["non_zero_ari"]
        non_zero_ma = arima_checked["non_zero_ma"]
        components_number_ets = components_dict["components_number_ets"]

        # Reset ARIMA section of vec_g before filling.
        # R does this implicitly via copy-on-modify; Python mutates in place,
        # so the MA += would accumulate across CF calls without this reset.
        arima_end = components_number_ets + components_dict["components_number_arima"]
        matrices_dict["vec_g"][components_number_ets:arima_end] = 0

        # Fill in the transition matrix - R lines 1388-1391
        if len(non_zero_ari) > 0:
            for row_idx in range(len(non_zero_ari)):
                poly_idx = non_zero_ari[row_idx, 0]
                state_idx = non_zero_ari[row_idx, 1]
                #  R: matF[componentsNumberETS+nonZeroARI[,2],
                # componentsNumberETS+1:...] <- -ariPolynomial[nonZeroARI[,1]]
                matrices_dict["mat_f"][
                    components_number_ets + state_idx,
                    components_number_ets : components_number_ets
                    + components_dict["components_number_arima"]
                    + constants_checked.get("constant_required", 0),
                ] = -arima_polynomials["ari_polynomial"][poly_idx]

        # Fill in the persistence vector - R lines 1392-1399
        if len(non_zero_ari) > 0:
            for row_idx in range(len(non_zero_ari)):
                poly_idx = non_zero_ari[row_idx, 0]
                state_idx = non_zero_ari[row_idx, 1]
                #  R: vecG[componentsNumberETS+nonZeroARI[,2]] <-
                # -ariPolynomial[nonZeroARI[,1]]
                matrices_dict["vec_g"][
                    components_number_ets + state_idx
                ] = -arima_polynomials["ari_polynomial"][poly_idx]

        if len(non_zero_ma) > 0:
            for row_idx in range(len(non_zero_ma)):
                poly_idx = non_zero_ma[row_idx, 0]
                state_idx = non_zero_ma[row_idx, 1]
                # R: vecG[...+nonZeroMA[,2]] += maPolynomial[nonZeroMA[,1]]
                matrices_dict["vec_g"][components_number_ets + state_idx] += (
                    arima_polynomials["ma_polynomial"][poly_idx]
                )

        j += n_arma_params
        matrices_dict["arima_polynomials"] = arima_polynomials

    # Initials of ETS
    if (
        model_type_dict["ets_model"]
        and initials_checked["initial_type"] not in ["complete", "backcasting"]
        and initials_checked["initial_estimate"]
    ):
        i = 0
        if initials_checked["initial_level_estimate"]:
            j += 1
            matrices_dict["mat_vt"][i, : lags_dict["lags_model_max"]] = B[j - 1]

        i += 1
        if (
            model_type_dict["model_is_trendy"]
            and initials_checked["initial_trend_estimate"]
        ):
            j += 1
            matrices_dict["mat_vt"][i, : lags_dict["lags_model_max"]] = B[j - 1]
            i += 1

        if model_type_dict["model_is_seasonal"] and (
            isinstance(initials_checked["initial_seasonal_estimate"], bool)
            and initials_checked["initial_seasonal_estimate"]
            or isinstance(initials_checked["initial_seasonal_estimate"], list)
            and any(initials_checked["initial_seasonal_estimate"])
        ):
            for k in range(components_dict["components_number_ets_seasonal"]):
                # Convert initial_seasonal_estimate to a list if it's not already
                # This is for handling single seasonalities
                if isinstance(initials_checked["initial_seasonal_estimate"], bool):
                    initials_checked["initial_seasonal_estimate"] = [
                        initials_checked["initial_seasonal_estimate"]
                    ] * components_dict["components_number_ets_seasonal"]

                if initials_checked["initial_seasonal_estimate"][k]:
                    # added a -1 because the seasonal index is 0-based in R
                    seasonal_index = (
                        components_dict["components_number_ets"]
                        - components_dict["components_number_ets_seasonal"]
                        + k
                    )
                    lag = lags_dict["lags"][seasonal_index]

                    matrices_dict["mat_vt"][seasonal_index, : lag - 1] = B[j : j + lag]

                    if model_type_dict["season_type"] == "A":
                        matrices_dict["mat_vt"][seasonal_index, lag - 1] = -np.sum(
                            B[j : j + lag]
                        )
                    else:  # "M"
                        matrices_dict["mat_vt"][seasonal_index, lag - 1] = 1 / np.prod(
                            B[j : j + lag]
                        )

                    j += lag - 1

    # Ensure arima_polynomials is available for initials section
    if arima_checked["arima_model"] and adam_cpp is None:
        arima_polynomials = matrices_dict.get("arima_polynomials", {})

    # Initials of ARIMA
    if arima_checked["arima_model"]:
        if (
            initials_checked["initial_type"] not in ["complete", "backcasting"]
            and initials_checked["initial_arima_estimate"]
        ):
            # print(f"DEBUG - Processing ARIMA initial values starting at index {j}")
            arima_index = (
                components_dict["components_number_ets"]
                + components_dict["components_number_arima"]
                - 1
            )

            matrices_dict["mat_vt"][
                arima_index, : initials_checked["initial_arima_number"]
            ] = B[j : j + initials_checked["initial_arima_number"]]

            ari_indices = (
                components_dict["components_number_ets"]
                + arima_checked["non_zero_ari"][:, 1]
            )
            ari_poly_vals = arima_polynomials["ariPolynomial"][
                arima_checked["non_zero_ari"][:, 0]
            ].reshape(-1, 1)
            init_vals = B[j : j + initials_checked["initial_arima_number"]]

            if model_type_dict["error_type"] == "A":
                matrices_dict["mat_vt"][
                    ari_indices, : initials_checked["initial_arima_number"]
                ] = ari_poly_vals @ init_vals.reshape(1, -1)
            else:  # "M"
                matrices_dict["mat_vt"][
                    ari_indices, : initials_checked["initial_arima_number"]
                ] = np.exp(ari_poly_vals @ np.log(init_vals).reshape(1, -1))

            j += initials_checked["initial_arima_number"]
        elif any([arima_checked["ar_estimate"], arima_checked["ma_estimate"]]):
            last_arima_row_idx = (
                components_dict["components_number_ets"]
                + components_dict["components_number_arima"]
                - 1
            )
            ari_indices = (
                components_dict["components_number_ets"]
                + arima_checked["non_zero_ari"][:, 1]
            )
            ari_poly_vals = arima_polynomials["ariPolynomial"][
                arima_checked["non_zero_ari"][:, 0]
            ].reshape(-1, 1)
            last_arima_row = matrices_dict["mat_vt"][
                last_arima_row_idx, : initials_checked["initial_arima_number"]
            ].copy()

            if model_type_dict["error_type"] == "A":
                matrices_dict["mat_vt"][
                    ari_indices, : initials_checked["initial_arima_number"]
                ] = ari_poly_vals @ last_arima_row.reshape(1, -1)
            else:  # "M"
                matrices_dict["mat_vt"][
                    ari_indices, : initials_checked["initial_arima_number"]
                ] = np.exp(ari_poly_vals @ np.log(last_arima_row).reshape(1, -1))

    # Xreg initial values
    if (
        explanatory_checked["xreg_model"]
        and (initials_checked["initial_type"] != "complete")
        and initials_checked["initial_estimate"]
        and initials_checked["initial_xreg_estimate"]
    ):
        xreg_params_est = np.asarray(
            explanatory_checked["xreg_parameters_estimated"], dtype=int
        )
        xreg_number_to_estimate = int(xreg_params_est.sum())
        xreg_indices = (
            components_dict["components_number_ets"]
            + components_dict["components_number_arima"]
            + np.where(xreg_params_est == 1)[0]
        )

        matrices_dict["mat_vt"][xreg_indices, : lags_dict["lags_model_max"]] = B[
            j : j + xreg_number_to_estimate, np.newaxis
        ]

        j += xreg_number_to_estimate

    # Constant
    constant_index = (
        components_dict["components_number_ets"]
        + components_dict["components_number_arima"]
        + explanatory_checked["xreg_number"]
    )
    if constants_checked["constant_estimate"]:
        matrices_dict["mat_vt"][constant_index, :] = B[j]
    elif constants_checked["constant_required"]:
        matrices_dict["mat_vt"][constant_index, :] = constants_checked["constant_value"]

    return {
        "mat_vt": matrices_dict["mat_vt"],
        "mat_wt": matrices_dict["mat_wt"],
        "mat_f": matrices_dict["mat_f"],
        "vec_g": matrices_dict["vec_g"],
        "arima_polynomials": matrices_dict["arima_polynomials"],
    }
