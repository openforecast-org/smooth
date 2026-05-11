import numpy as np

from smooth.adam_general.core.utils.polynomials import adam_polynomialiser
from smooth.adam_general.core.utils.utils import (
    calculate_acf,
    calculate_pacf,
)


def initialiser(
    # Model type info
    model_type_dict,
    # Components info
    components_dict,
    # Lags info
    lags_dict,
    # Matrices from creator
    adam_created,
    # Parameter dictionaries
    persistence_checked,
    initials_checked,
    arima_checked,
    constants_checked,
    explanatory_checked,
    phi_dict,
    # Other parameters
    observations_dict,
    bounds="usual",
    other=None,
    other_parameter_estimate=False,
    other_value=2.0,
    profile_dict=None,  # Added
    adam_cpp=None,
):
    """
    Initialize parameter vector and bounds for ADAM optimization.

    This function constructs the initial parameter vector **B** and its lower/upper
    bounds
    (Bl, Bu) for the nonlinear optimization process. The parameter vector contains all
    estimable parameters in a specific order, and the bounds enforce constraints during
    optimization.

    The function determines:

    1. **Parameter Count**: Calculate total number of parameters to estimate based on
    model specification
    2. **Parameter Vector B**: Assign reasonable starting values for each parameter
    3. **Lower Bounds (Bl)**: Set minimum allowed values (e.g., 0 for smoothing
    parameters)
    4. **Upper Bounds (Bu)**: Set maximum allowed values (e.g., 1 for smoothing
    parameters)
    5. **Parameter Names**: Create descriptive labels for each parameter

    **Parameter Vector Structure**:

    The optimization parameter vector B is organized as follows::

        B = [persistence_parameters,     # α, β, γ (ETS smoothing)
             phi,                        # Damping parameter (if damped trend)
             initial_states,             # l₀, b₀, s₀, ARIMA initials (if optimal)
             ar_parameters,              # AR coefficients (if ARIMA)
             ma_parameters,              # MA coefficients (if ARIMA)
             xreg_parameters,            # Regression coefficients (if regressors)
             constant]                   # Intercept (if included)

    **Typical Parameter Bounds**:

    - **Persistence (α, β, γ)**: [0, 1] with additional constraints β ≤ α, γ ≤ 1-α
    - **Damping (φ)**: [0, 1]
    - **Initial states**: Data-dependent bounds (e.g., ±3 standard deviations from mean)
    - **ARIMA (AR/MA)**: [-1, 1] for stability (tighter bounds near stationarity
    boundaries)
    - **Regressors**: Unbounded or loosely bounded
    - **Constant**: Unbounded

    Parameters
    ----------
    model_type_dict : dict
        Model specification containing:

        - 'ets_model': Whether ETS components exist
        - 'arima_model': Whether ARIMA components exist
        - 'xreg_model': Whether regressors are present
        - 'error_type': 'A' or 'M'
        - 'trend_type': 'N', 'A', 'Ad', 'M', 'Md'
        - 'season_type': 'N', 'A', 'M'
        - 'model_is_trendy': Trend presence flag
        - 'model_is_seasonal': Seasonality presence flag
        - 'damped': Damped trend flag

    components_dict : dict
        Component counts containing:

        - 'components_number_all': Total state dimension
        - 'components_number_ets': ETS component count
        - 'components_number_ets_seasonal': Seasonal component count
        - 'components_number_arima': ARIMA component count

    lags_dict : dict
        Lag structure containing:

        - 'lags': Primary lag vector
        - 'lags_model': Per-component lags
        - 'lags_model_seasonal': Seasonal lags only
        - 'lags_model_max': Maximum lag

    adam_created : dict
        State-space matrices from ``creator()`` containing:

        - 'mat_vt': State vector (used to extract initial values)
        - 'mat_wt': Measurement matrix
        - 'mat_f': Transition matrix
        - 'vec_g': Persistence vector

    persistence_checked : dict
        Persistence specification containing:

        - 'persistence_estimate': Whether to estimate any smoothing parameters
        - 'persistence_level_estimate': Whether to estimate α
        - 'persistence_trend_estimate': Whether to estimate β
        - 'persistence_seasonal_estimate': List of flags for each seasonal γ
        - 'persistence_xreg_estimate': Whether to estimate regressor persistence
        - Fixed values for non-estimated persistence parameters

    initials_checked : dict
        Initial states specification containing:

        - 'initial_type': 'optimal', 'backcasting', 'complete', or 'provided'
        - 'initial_level_estimate': Whether to optimize level initial
        - 'initial_trend_estimate': Whether to optimize trend initial
        - 'initial_seasonal_estimate': List of flags for seasonal initials
        - 'initial_arima_estimate': Whether to optimize ARIMA initials
        - 'initial_arima_number': Number of ARIMA initial states
        - 'initial_xreg_estimate': Whether to optimize regressor initials
        - Fixed initial values (if 'provided')

    arima_checked : dict
        ARIMA specification containing:

        - 'arima_model': ARIMA presence flag
        - 'ar_estimate': Whether to estimate AR coefficients
        - 'ma_estimate': Whether to estimate MA coefficients
        - 'ar_orders': AR orders per lag
        - 'ma_orders': MA orders per lag
        - 'ar_parameters': Fixed AR coefficients (if not estimated)
        - 'ma_parameters': Fixed MA coefficients (if not estimated)

    constants_checked : dict
        Constant term specification containing:

        - 'constant_required': Whether constant is included
        - 'constant_estimate': Whether to estimate constant
        - 'constant_value': Fixed constant value (if not estimated)

    explanatory_checked : dict
        External regressors specification containing:

        - 'xreg_model': Regressor presence flag
        - 'xreg_number': Number of regressors
        - 'xreg_parameters_estimated': Which regressor coefficients to estimate
        - 'xreg_parameters_persistence': Persistence for adaptive regressors

    phi_dict : dict
        Damping specification containing:

        - 'phi': Fixed damping value (if not estimated)
        - 'phi_estimate': Whether to estimate φ

    observations_dict : dict
        Observation information containing:

        - 'y_in_sample': Time series data (for computing data-dependent bounds)
        - 'obs_in_sample': Number of observations

    bounds : str, default="usual"
        Bound type specification:

        - **"usual"**: Standard bounds (α,β,γ ∈ [0,1], φ ∈ [0,1], etc.)
        - **"admissible"**: Relaxed bounds for admissible parameter space
        - **"none"**: No bounds (not recommended)

    other : float or None, default=None
        Additional distribution parameter (for certain distributions).
        Currently unused in initialiser.

    profile_dict : dict or None, default=None
        Profile matrices for time-varying parameters. Required when
        initial_type="complete" to properly extract backcasted states.

    Returns
    -------
    dict
        Dictionary containing initialization results:

        - **'B'** (numpy.ndarray): Initial parameter vector, shape (n_params,).
          Starting values for optimization. Reasonable defaults based on model type.

        - **'Bl'** (numpy.ndarray): Lower bounds, shape (n_params,).
          Minimum allowed parameter values during optimization.

        - **'Bu'** (numpy.ndarray): Upper bounds, shape (n_params,).
          Maximum allowed parameter values during optimization.

        - **'names'** (list of str): Parameter names for interpretability.
          Examples: 'alpha', 'beta', 'gamma[1]', 'phi', 'initial_level',
          'ar[1]', 'ma[1]', 'xreg[1]', 'constant'

    Notes
    -----
    **Starting Values Philosophy**:

    Good starting values accelerate convergence. This function uses:

    - **Smoothing parameters**: Start at 0.1-0.3 (conservative, data-adaptive)
    - **Damping**: Start at 0.95 (mild damping)
    - **Initial states**: Extracted from matrices populated by ``creator()``
    - **ARIMA**: Start near zero for stability
    - **Regressors**: Start at zero (assumes centering)

    **Bounds and Constraints**:

    Bounds enforce hard constraints during optimization. Additional soft constraints
    (e.g., β ≤ α) are enforced via penalty in the cost function ``CF()``.

    For "usual" bounds:

    - Persistence: [0, 1]
    - Damping: [0, 1]
    - Initial states: [min_data - 3*sd, max_data + 3*sd]
    - ARIMA: [-0.9999, 0.9999] (slightly tighter for numerical stability)

    **Parameter Count Formula**:

    Total parameters = ETS_persistence + phi + ETS_initials + ARIMA_params +
    ARIMA_initials + regressor_coeffs + regressor_initials + constant

    The exact count depends on what is estimated vs. fixed.

    **Relationship to Optimization**:

    The returned B, Bl, Bu are passed directly to NLopt. During each optimization
    iteration:

    1. NLopt proposes new B values within [Bl, Bu]
    2. ``CF()`` calls ``filler()`` to populate matrices with B
    3. ``CF()`` evaluates cost and returns to NLopt
    4. Repeat until convergence

    See Also
    --------
    creator : Create state-space matrices before calling initialiser
    filler : Fill matrices with parameter values from B during optimization
    estimator : Main estimation function that calls initialiser

    Examples
    --------
    Initialize parameters for simple exponential smoothing::

        >>> init_result = initialiser(
        ...     model_type_dict={'ets_model': True, 'error_type': 'A',
        ...                      'trend_type': 'N', 'season_type': 'N',
        ...                      'model_is_trendy': False,
        ...                      'model_is_seasonal': False, ...},
        ...     components_dict={'components_number_all': 1,
        ...                      'components_number_ets': 1, ...},
        ...     lags_dict={'lags': np.array([1]), 'lags_model_max': 1, ...},
        ...     adam_created=adam_matrices,
        ...     persistence_checked={'persistence_estimate': True,
        ...                          'persistence_level_estimate': True, ...},
        ...     initials_checked={'initial_type': 'optimal',
        ...                       'initial_level_estimate': True, ...},
        ...     arima_checked={'arima_model': False, ...},
        ...     constants_checked={'constant_required': False, ...},
        ...     explanatory_checked={'xreg_model': False, ...},
        ...     phi_dict={'phi': 1.0, 'phi_estimate': False},
        ...     observations_dict={'y_in_sample': data,
        ...                         'obs_in_sample': len(data), ...},
        ...     bounds="usual"
        ... )
        >>> print(init_result['B'])  # [0.3, 100] - alpha and initial level
        >>> print(init_result['names'])  # ['alpha', 'initial_level']
        >>> print(len(init_result['B']))  # 2 parameters

    Initialize for Holt's linear trend with backcasting::

        >>> init_result = initialiser(
        ...     model_type_dict={'ets_model': True, 'error_type': 'A',
        ...                      'trend_type': 'A', 'model_is_trendy': True, ...},
        ...     initials_checked={'initial_type': 'backcasting', ...},
        ...     # No initial states in B
        ...     persistence_checked={'persistence_estimate': True,
        ...                          'persistence_level_estimate': True,
        ...                          'persistence_trend_estimate': True, ...},
        ...     ...
        ... )
        >>> print(init_result['names']) # ['alpha', 'beta'] only - no initials with
        backcasting
    """
    # Build persistence estimate vector with proper seasonal expansion
    # Each seasonal component gets its own entry in the vector
    persistence_estimate_vector = [
        persistence_checked["persistence_level_estimate"],
        model_type_dict["model_is_trendy"]
        and persistence_checked["persistence_trend_estimate"],
    ]
    if model_type_dict["model_is_seasonal"]:
        persistence_estimate_vector.extend(
            persistence_checked["persistence_seasonal_estimate"]
        )
    total_params = (
        model_type_dict["ets_model"]
        * (sum(persistence_estimate_vector) + phi_dict["phi_estimate"])
        + explanatory_checked["xreg_model"]
        * persistence_checked["persistence_xreg_estimate"]
        * max(explanatory_checked["xreg_parameters_persistence"] or [0])
        + arima_checked["arima_model"]
        * (
            arima_checked["ar_estimate"] * sum(arima_checked["ar_orders"] or [])
            + arima_checked["ma_estimate"] * sum(arima_checked["ma_orders"] or [])
        )
        + model_type_dict["ets_model"]
        * (initials_checked["initial_type"] not in ["backcasting", "complete"])
        * (
            initials_checked["initial_level_estimate"]
            + (
                model_type_dict["model_is_trendy"]
                * initials_checked["initial_trend_estimate"]
            )
            + (
                model_type_dict["model_is_seasonal"]
                * sum(
                    initials_checked["initial_seasonal_estimate"]
                    * (np.array(lags_dict["lags_model_seasonal"] or []) - 1)
                )
            )
        )
        + (initials_checked["initial_type"] not in ["backcasting", "complete"])
        * arima_checked["arima_model"]
        * (initials_checked["initial_arima_number"] or 0)
        * initials_checked["initial_arima_estimate"]
        + (initials_checked["initial_type"] != "complete")
        * explanatory_checked["xreg_model"]
        * initials_checked["initial_xreg_estimate"]
        * sum(explanatory_checked["xreg_parameters_estimated"] or [])
        + constants_checked["constant_estimate"]
        + int(other_parameter_estimate)
    )

    B = np.zeros(total_params)
    Bl = np.zeros(total_params)
    Bu = np.zeros(total_params)
    names = []

    j = 0

    if model_type_dict["ets_model"]:
        if persistence_checked["persistence_estimate"] and any(
            persistence_estimate_vector
        ):
            if any(
                ptype == "M"
                for ptype in [
                    model_type_dict["error_type"],
                    model_type_dict["trend_type"],
                    model_type_dict["season_type"],
                ]
            ):
                if (
                    (
                        model_type_dict["error_type"] == "A"
                        and model_type_dict["trend_type"] == "A"
                        and model_type_dict["season_type"] == "M"
                    )
                    or (
                        model_type_dict["error_type"] == "A"
                        and model_type_dict["trend_type"] == "M"
                        and model_type_dict["season_type"] == "A"
                    )
                    or (
                        initials_checked["initial_type"] in ["complete", "backcasting"]
                        and (
                            (
                                model_type_dict["error_type"] == "M"
                                and model_type_dict["trend_type"] == "A"
                                and model_type_dict["season_type"] == "A"
                            )
                            or (
                                model_type_dict["error_type"] == "M"
                                and model_type_dict["trend_type"] == "A"
                                and model_type_dict["season_type"] == "M"
                            )
                        )
                    )
                ):
                    #  Match R:
                    # c(0.01,0.005,rep(0.001,componentsNumberETSSeasonal))
                    # [which(persistenceEstimateVector)]
                    initial_values = [0.01, 0.005] + [0.001] * components_dict[
                        "components_number_ets_seasonal"
                    ]
                    B[j : j + sum(persistence_estimate_vector)] = [
                        val
                        for val, estimate in zip(
                            initial_values, persistence_estimate_vector
                        )
                        if estimate
                    ]
                elif (
                    model_type_dict["error_type"] == "M"
                    and model_type_dict["trend_type"] == "M"
                    and model_type_dict["season_type"] == "A"
                ):
                    #  Match R:
                    # c(0.01,0.005,rep(0.01,componentsNumberETSSeasonal))
                    # [which(persistenceEstimateVector)]
                    initial_values = [0.01, 0.005] + [0.01] * components_dict[
                        "components_number_ets_seasonal"
                    ]
                    B[j : j + sum(persistence_estimate_vector)] = [
                        val
                        for val, estimate in zip(
                            initial_values, persistence_estimate_vector
                        )
                        if estimate
                    ]
                elif (
                    model_type_dict["error_type"] == "M"
                    and model_type_dict["trend_type"] == "A"
                ):
                    if initials_checked["initial_type"] in ["complete", "backcasting"]:
                        #  Match R:
                        # c(0.1,0.05,rep(0.3,componentsNumberETSSeasonal))
                        # [which(persistenceEstimateVector)]
                        initial_values = [0.1, 0.05] + [0.3] * components_dict[
                            "components_number_ets_seasonal"
                        ]
                        B[j : j + sum(persistence_estimate_vector)] = [
                            val
                            for val, estimate in zip(
                                initial_values, persistence_estimate_vector
                            )
                            if estimate
                        ]
                    else:
                        B[j : j + sum(persistence_estimate_vector)] = [0.2, 0.01] + [
                            0.3
                        ] * components_dict["components_number_ets_seasonal"]
                elif (
                    model_type_dict["error_type"] == "M"
                    and model_type_dict["trend_type"] == "M"
                ):
                    B[j : j + sum(persistence_estimate_vector)] = [0.1, 0.05] + [
                        0.3
                    ] * components_dict["components_number_ets_seasonal"]
                else:
                    #  Match R:
                    # c(0.1,0.05,rep(0.3,componentsNumberETSSeasonal))
                    # [which(persistenceEstimateVector)]
                    #  Always build full vector, then filter - this handles
                    # non-trendy seasonal models correctly
                    initial_values = [0.1, 0.05] + [0.3] * components_dict[
                        "components_number_ets_seasonal"
                    ]
                    B[j : j + sum(persistence_estimate_vector)] = [
                        val
                        for val, estimate in zip(
                            initial_values, persistence_estimate_vector
                        )
                        if estimate
                    ]

            else:
                #  Match R:
                # c(0.1,0.05,rep(0.3,componentsNumberETSSeasonal))
                # [which(persistenceEstimateVector)]
                #  Always build full vector, then filter - this handles
                # non-trendy seasonal models correctly
                initial_values = [0.1, 0.05] + [0.3] * components_dict[
                    "components_number_ets_seasonal"
                ]
                B[j : j + sum(persistence_estimate_vector)] = [
                    val
                    for val, estimate in zip(
                        initial_values, persistence_estimate_vector
                    )
                    if estimate
                ]

            if bounds == "usual":
                Bl[j : j + sum(persistence_estimate_vector)] = 0
                Bu[j : j + sum(persistence_estimate_vector)] = 1
            else:
                Bl[j : j + sum(persistence_estimate_vector)] = -5
                Bu[j : j + sum(persistence_estimate_vector)] = 5

            # Names for B
            if persistence_checked["persistence_level_estimate"]:
                names.append("alpha")
                j += 1
            if (
                model_type_dict["model_is_trendy"]
                and persistence_checked["persistence_trend_estimate"]
            ):
                names.append("beta")
                j += 1
            if model_type_dict["model_is_seasonal"] and any(
                persistence_checked["persistence_seasonal_estimate"]
            ):
                if components_dict["components_number_ets_seasonal"] > 1:
                    names.extend(
                        [
                            f"gamma{i}"
                            for i in range(
                                1, components_dict["components_number_ets_seasonal"] + 1
                            )
                        ]
                    )
                else:
                    names.append("gamma")
                j += sum(persistence_checked["persistence_seasonal_estimate"])

    if (
        explanatory_checked["xreg_model"]
        and persistence_checked["persistence_xreg_estimate"]
    ):
        xreg_persistence_number = max(
            explanatory_checked["xreg_parameters_persistence"]
        )
        B[j : j + xreg_persistence_number] = (
            0.01 if model_type_dict["error_type"] == "A" else 0
        )
        Bl[j : j + xreg_persistence_number] = -5
        Bu[j : j + xreg_persistence_number] = 5
        names.extend([f"delta{i + 1}" for i in range(xreg_persistence_number)])
        j += xreg_persistence_number

    if model_type_dict["ets_model"] and phi_dict["phi_estimate"]:
        B[j] = 0.95
        names.append("phi")
        Bl[j] = 0
        Bu[j] = 1
        j += 1

    if arima_checked["arima_model"]:
        if any([arima_checked["ar_estimate"], arima_checked["ma_estimate"]]):
            # Use numpy for element-wise multiplication of orders and lags
            ma_orders_arr = np.array(arima_checked["ma_orders"])
            ar_orders_arr = np.array(arima_checked["ar_orders"])
            i_orders_arr = np.array(arima_checked["i_orders"])
            lags_arr = np.array(lags_dict.get("lags_original", lags_dict["lags"]))

            acf_values = [-0.1] * int(np.sum(ma_orders_arr * lags_arr))
            pacf_values = [0.1] * int(np.sum(ar_orders_arr * lags_arr))

            if not (model_type_dict["ets_model"] or np.all(i_orders_arr == 0)):
                y_differenced = observations_dict["y_in_sample"].copy()
                # Implement differencing if needed
                if np.any(i_orders_arr > 0):
                    for i, order in enumerate(arima_checked["i_orders"]):
                        if order > 0:
                            y_differenced = np.diff(y_differenced, n=order, axis=0)

                # ACF/PACF calculation for non-seasonal models
                if np.all(lags_arr <= 1):
                    ma_total = int(np.sum(ma_orders_arr * lags_arr))
                    ar_total = int(np.sum(ar_orders_arr * lags_arr))
                    if arima_checked["ma_required"] and arima_checked["ma_estimate"]:
                        acf_vals = calculate_acf(y_differenced, nlags=max(1, ma_total))[
                            1:
                        ]
                        acf_values[: min(ma_total, len(y_differenced) - 1)] = acf_vals[
                            : min(ma_total, len(y_differenced) - 1)
                        ]
                    if arima_checked["ar_required"] and arima_checked["ar_estimate"]:
                        pacf_vals = calculate_pacf(
                            y_differenced, nlags=max(1, ar_total)
                        )
                        pacf_values[: min(ar_total, len(y_differenced) - 1)] = (
                            pacf_vals[1 : min(ar_total + 1, len(pacf_vals))]
                        )

            arma_start_index = j
            for i, lag in enumerate(lags_dict.get("lags_original", lags_dict["lags"])):
                if (
                    arima_checked["ar_required"]
                    and arima_checked["ar_estimate"]
                    and arima_checked["ar_orders"][i] > 0
                ):
                    indices = np.arange(1, arima_checked["ar_orders"][i] + 1) * lag - 1
                    vals = [
                        pacf_values[idx] if idx < len(pacf_values) else 0.1
                        for idx in indices
                    ]
                    if all(not np.isnan(v) for v in vals):
                        B[j : j + arima_checked["ar_orders"][i]] = vals
                    else:
                        B[j : j + arima_checked["ar_orders"][i]] = 0.1
                    if sum(B[j : j + arima_checked["ar_orders"][i]]) > 1:
                        B[j : j + arima_checked["ar_orders"][i]] = (
                            B[j : j + arima_checked["ar_orders"][i]]
                            / sum(B[j : j + arima_checked["ar_orders"][i]])
                            - 0.01
                        )
                    Bl[j : j + arima_checked["ar_orders"][i]] = -5
                    Bu[j : j + arima_checked["ar_orders"][i]] = 5
                    names.extend(
                        [
                            f"phi{k + 1}[{lag}]"
                            for k in range(arima_checked["ar_orders"][i])
                        ]
                    )
                    j += arima_checked["ar_orders"][i]

                if (
                    arima_checked["ma_required"]
                    and arima_checked["ma_estimate"]
                    and arima_checked["ma_orders"][i] > 0
                ):
                    indices = np.arange(1, arima_checked["ma_orders"][i] + 1) * lag - 1
                    vals = [
                        acf_values[idx] if idx < len(acf_values) else -0.1
                        for idx in indices
                    ]
                    if all(not np.isnan(v) for v in vals):
                        B[j : j + arima_checked["ma_orders"][i]] = vals
                    else:
                        B[j : j + arima_checked["ma_orders"][i]] = -0.1
                    if sum(B[j : j + arima_checked["ma_orders"][i]]) > 1:
                        B[j : j + arima_checked["ma_orders"][i]] = (
                            B[j : j + arima_checked["ma_orders"][i]]
                            / sum(B[j : j + arima_checked["ma_orders"][i]])
                            - 0.01
                        )
                    Bl[j : j + arima_checked["ma_orders"][i]] = -5
                    Bu[j : j + arima_checked["ma_orders"][i]] = 5
                    names.extend(
                        [
                            f"theta{k + 1}[{lag}]"
                            for k in range(arima_checked["ma_orders"][i])
                        ]
                    )
                    j += arima_checked["ma_orders"][i]

    #  NOTE: Removed backcasting from initialiser - CF already handles backcasting for
    # complete/backcasting modes
    # This was causing double backcasting which led to different results than R

    if (
        model_type_dict["ets_model"]
        and initials_checked["initial_type"] not in ["backcasting", "complete"]
        and initials_checked["initial_estimate"]
    ):
        if initials_checked["initial_level_estimate"]:
            B[j] = adam_created["mat_vt"][0, 0]
            Bl[j] = -np.inf if model_type_dict["error_type"] == "A" else 0
            Bu[j] = np.inf
            names.append("level")
            j += 1
        if (
            model_type_dict["model_is_trendy"]
            and initials_checked["initial_trend_estimate"]
        ):
            B[j] = adam_created["mat_vt"][1, 0]
            Bl[j] = -np.inf if model_type_dict["trend_type"] == "A" else 0
            Bu[j] = np.inf if model_type_dict["trend_type"] == "A" else 2
            names.append("trend")
            j += 1

        if model_type_dict["model_is_seasonal"] and (
            isinstance(initials_checked["initial_seasonal_estimate"], bool)
            and initials_checked["initial_seasonal_estimate"]
            or isinstance(initials_checked["initial_seasonal_estimate"], list)
            and any(initials_checked["initial_seasonal_estimate"])
        ):
            if components_dict["components_number_ets_seasonal"] > 1:
                for k in range(components_dict["components_number_ets_seasonal"]):
                    if initials_checked["initial_seasonal_estimate"][k]:
                        # Get the correct seasonal component index and lag
                        seasonal_index = (
                            components_dict["components_number_ets"]
                            - components_dict["components_number_ets_seasonal"]
                            + k
                        )
                        lag = lags_dict["lags"][seasonal_index]

                        # Get the values from mat_vt (make sure dimensions match)
                        seasonal_values = adam_created["mat_vt"][
                            seasonal_index, : lag - 1
                        ]

                        # Assign to B with matching dimensions
                        B[j : j + lag - 1] = seasonal_values

                        if model_type_dict["season_type"] == "A":
                            Bl[j : j + lag - 1] = -np.inf
                            Bu[j : j + lag - 1] = np.inf
                        else:
                            Bl[j : j + lag - 1] = 0
                            Bu[j : j + lag - 1] = np.inf
                        names.extend([f"seasonal_{m}" for m in range(2, lag)])
                        j += lag - 1
            else:
                # Get the correct seasonal component index and lag
                seasonal_index = components_dict["components_number_ets"] - 1
                temp_lag = lags_dict["lags_model"][seasonal_index]
                seasonal_values = adam_created["mat_vt"][seasonal_index, : temp_lag - 1]
                # Assign to B with matching dimensions
                B[j : j + temp_lag - 1] = seasonal_values
                if model_type_dict["season_type"] == "A":
                    Bl[j : j + temp_lag - 1] = -np.inf
                    Bu[j : j + temp_lag - 1] = np.inf
                else:
                    Bl[j : j + temp_lag - 1] = 0
                    Bu[j : j + temp_lag - 1] = np.inf
                # names.extend([f"seasonal_{m}" for m in range(2, temp_lag)])
                j += temp_lag - 1
    if (
        initials_checked["initial_type"] not in ["backcasting", "complete"]
        and arima_checked["arima_model"]
        and initials_checked["initial_arima_estimate"]
    ):
        # Fix: Python 0-based indexing requires -1 (R uses 1-based)
        B[j : j + initials_checked["initial_arima_number"]] = adam_created["mat_vt"][
            components_dict["components_number_ets"]
            + components_dict["components_number_arima"]
            - 1,
            : initials_checked["initial_arima_number"],
        ]
        # Normalize by ARI polynomial tail (matches R lines 1684-1686)
        if adam_cpp is not None:
            n_ar = (
                sum(arima_checked["ar_orders"]) if arima_checked["ar_estimate"] else 0
            )
            n_ma = (
                sum(arima_checked["ma_orders"]) if arima_checked["ma_estimate"] else 0
            )
            n_arma = n_ar + n_ma
            if n_arma > 0:
                arima_polys = adam_polynomialiser(
                    adam_cpp,
                    B[arma_start_index : arma_start_index + n_arma],
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
                ari_tail = arima_polys["ari_polynomial"][-1]
                if ari_tail != 0:
                    B[j : j + initials_checked["initial_arima_number"]] /= ari_tail
        names.extend(
            [
                f"ARIMAState{n}"
                for n in range(1, initials_checked["initial_arima_number"] + 1)
            ]
        )
        if model_type_dict["error_type"] == "A":
            Bl[j : j + initials_checked["initial_arima_number"]] = -np.inf
            Bu[j : j + initials_checked["initial_arima_number"]] = np.inf
        else:
            B[j : j + initials_checked["initial_arima_number"]] = np.abs(
                B[j : j + initials_checked["initial_arima_number"]]
            )
            Bl[j : j + initials_checked["initial_arima_number"]] = 0
            Bu[j : j + initials_checked["initial_arima_number"]] = np.inf
        j += initials_checked["initial_arima_number"]

    if (
        initials_checked["initial_xreg_estimate"]
        and explanatory_checked["xreg_model"]
        and initials_checked["initial_type"] != "complete"
    ):
        xreg_number_to_estimate = sum(explanatory_checked["xreg_parameters_estimated"])
        if xreg_number_to_estimate > 0:
            xreg_start = (
                components_dict["components_number_ets"]
                + components_dict["components_number_arima"]
            )
            B[j : j + xreg_number_to_estimate] = adam_created["mat_vt"][
                xreg_start : xreg_start + xreg_number_to_estimate, 0
            ]
            names.extend([f"xreg{idx + 1}" for idx in range(xreg_number_to_estimate)])
            Bl[j : j + xreg_number_to_estimate] = -np.inf
            Bu[j : j + xreg_number_to_estimate] = np.inf
            j += xreg_number_to_estimate

    if constants_checked["constant_estimate"]:
        j += 1
        if (
            adam_created["mat_vt"].shape[0]
            > components_dict["components_number_ets"]
            + components_dict["components_number_arima"]
            + explanatory_checked["xreg_number"]
        ):
            B[j - 1] = adam_created["mat_vt"][
                components_dict["components_number_ets"]
                + components_dict["components_number_arima"]
                + explanatory_checked["xreg_number"],
                0,
            ]
        else:
            B[j - 1] = 0  # or some other default value
        names.append(constants_checked["constant_name"] or "constant")
        if model_type_dict["ets_model"] or (
            arima_checked["i_orders"] is not None
            and sum(arima_checked["i_orders"]) != 0
        ):
            if model_type_dict["error_type"] == "A":
                Bu[j - 1] = np.quantile(
                    np.diff(
                        observations_dict["y_in_sample"][
                            observations_dict["ot_logical"]
                        ],
                        axis=0,
                    ),
                    0.6,
                )
                Bl[j - 1] = -Bu[j - 1]
            else:
                Bu[j - 1] = np.exp(
                    np.quantile(
                        np.diff(
                            np.log(
                                observations_dict["y_in_sample"][
                                    observations_dict["ot_logical"]
                                ]
                            ),
                            axis=0,
                        ),
                        0.6,
                    )
                )
                Bl[j - 1] = np.exp(
                    np.quantile(
                        np.diff(
                            np.log(
                                observations_dict["y_in_sample"][
                                    observations_dict["ot_logical"]
                                ]
                            ),
                            axis=0,
                        ),
                        0.4,
                    )
                )

            if Bu[j - 1] <= Bl[j - 1]:
                Bu[j - 1] = np.inf
                Bl[j - 1] = -np.inf if model_type_dict["error_type"] == "A" else 0

            if B[j - 1] <= Bl[j - 1]:
                Bl[j - 1] = -np.inf if model_type_dict["error_type"] == "A" else 0
            if B[j - 1] >= Bu[j - 1]:
                Bu[j - 1] = np.inf
        else:
            Bu[j - 1] = max(
                np.max(
                    np.abs(
                        observations_dict["y_in_sample"][
                            observations_dict["ot_logical"]
                        ]
                    )
                ),
                abs(B[j - 1]) * 1.01,
            )
            Bl[j - 1] = -Bu[j - 1]

    if other_parameter_estimate:
        j += 1
        B[j - 1] = other_value
        names.append("other")
        Bl[j - 1] = 0.25  # penalty kicks in below 0.25 (cost_functions.py:323)
        Bu[j - 1] = np.inf
    return {"B": B, "Bl": Bl, "Bu": Bu, "names": names}


def _extract_initialiser_params(
    model_type_dict, components_dict, lags_dict, adam_created, observations_dict
):
    """
    Extract parameters needed for initialization from input dictionaries.

    Args:
        model_type_dict: Dictionary containing model type information
        components_dict: Dictionary containing component information
        lags_dict: Dictionary containing lags information
        adam_created: Dictionary containing created model matrices
        observations_dict: Dictionary containing observation information

    Returns:
        Dict: Dictionary containing extracted parameters
    """
    # Extract observation values
    obs_states = observations_dict["obs_states"]
    obs_in_sample = observations_dict["obs_in_sample"]
    obs_all = observations_dict["obs_all"]
    ot_logical = observations_dict["ot_logical"]
    y_in_sample = observations_dict["y_in_sample"]

    # Extract model type values
    error_type = model_type_dict["error_type"]
    trend_type = model_type_dict["trend_type"]
    season_type = model_type_dict["season_type"]
    ets_model = model_type_dict["ets_model"]
    model_is_trendy = model_type_dict["model_is_trendy"]
    model_is_seasonal = model_type_dict["model_is_seasonal"]

    # Extract matrices
    mat_vt = adam_created["mat_vt"]
    mat_wt = adam_created["mat_wt"]
    mat_f = adam_created["mat_f"]
    vec_g = adam_created["vec_g"]

    # Extract component numbers
    components_number_ets = components_dict["components_number_ets"]
    components_number_ets_seasonal = components_dict["components_number_ets_seasonal"]
    components_number_arima = components_dict.get("components_number_arima", 0)

    # Extract lags info
    lags = lags_dict["lags"]
    lags_model = lags_dict["lags_model"]
    lags_model_max = lags_dict["lags_model_max"]

    return {
        "obs_states": obs_states,
        "obs_in_sample": obs_in_sample,
        "obs_all": obs_all,
        "ot_logical": ot_logical,
        "y_in_sample": y_in_sample,
        "error_type": error_type,
        "trend_type": trend_type,
        "season_type": season_type,
        "ets_model": ets_model,
        "model_is_trendy": model_is_trendy,
        "model_is_seasonal": model_is_seasonal,
        "mat_vt": mat_vt,
        "mat_wt": mat_wt,
        "mat_f": mat_f,
        "vec_g": vec_g,
        "components_number_ets": components_number_ets,
        "components_number_ets_seasonal": components_number_ets_seasonal,
        "components_number_arima": components_number_arima,
        "lags": lags,
        "lags_model": lags_model,
        "lags_model_max": lags_model_max,
    }


def _calculate_initial_parameters_and_bounds(
    model_params,
    persistence_checked,
    initials_checked,
    arima_checked,
    constants_checked,
    explanatory_checked,
    phi_dict,
    bounds,
    adam_created,  # Added
    observations_dict,  # Added
):
    """
    Calculate initial parameter vector B, bounds Bl and Bu, and names.
    Combines logic from old initialiser and _prepare_optimization_inputs.
    """
    # Extract relevant parameters from model_params and dicts
    ets_model = model_params["ets_model"]
    model_is_trendy = model_params["model_is_trendy"]
    model_is_seasonal = model_params["model_is_seasonal"]
    error_type = model_params["error_type"]
    trend_type = model_params["trend_type"]
    season_type = model_params["season_type"]
    components_number_ets = model_params["components_number_ets"]
    components_number_ets_seasonal = model_params["components_number_ets_seasonal"]
    mat_vt = adam_created["mat_vt"]  # Used for initial state estimates
    y_in_sample = observations_dict["y_in_sample"]
    ot_logical = observations_dict["ot_logical"]

    # Calculate the total number of parameters to estimate
    # --- Calculate persistence params ---
    est_level = persistence_checked.get("persistence_level_estimate", False)
    est_trend = model_is_trendy and persistence_checked.get(
        "persistence_trend_estimate", False
    )
    seas_est_val = persistence_checked.get("persistence_seasonal_estimate", False)
    num_seasonal_persistence_params = 0  # Initialize
    if isinstance(seas_est_val, bool):
        est_seasonal = model_is_seasonal and seas_est_val
        #  num_seasonal_persistence_params = 1 if est_seasonal else 0 # Old logic was
        # sum of components
        if (
            est_seasonal
        ):  # if est_seasonal is True, it means all seasonal components are estimated
            num_seasonal_persistence_params = (
                components_number_ets_seasonal
                if components_number_ets_seasonal > 0
                else 1
                if model_is_seasonal
                else 0
            )
    elif isinstance(seas_est_val, (list, np.ndarray)):
        est_seasonal = model_is_seasonal and any(seas_est_val)
        num_seasonal_persistence_params = sum(p for p in seas_est_val if p)
    else:
        est_seasonal = False
        num_seasonal_persistence_params = 0

    num_ets_persistence_params = sum(
        [
            est_level,
            est_trend,
            num_seasonal_persistence_params
            if isinstance(seas_est_val, bool) and seas_est_val
            else num_seasonal_persistence_params,
        ]
    )

    # --- Xreg persistence parameters (deltas) ---
    est_xreg_persistence = explanatory_checked.get(
        "xreg_model", False
    ) and persistence_checked.get("persistence_xreg_estimate", False)
    num_xreg_persistence_params = 0
    if est_xreg_persistence:
        #  The old code used max(explanatory_checked['xreg_parameters_persistence'] or
        # [0])
        #  This implies xreg_parameters_persistence is a list of integers indicating
        # which xreg variables have persistence
        # and the number of delta parameters is the max value in that list.
        #  For simplicity, if persistence_xreg_estimate is True, we might assume one
        # delta per xreg, or follow old logic if xreg_parameters_persistence is
        # available.
        # The old code's logic for xreg_parameters_persistence was:
        # max(explanatory_checked['xreg_parameters_persistence'] or [0])
        #  This suggests 'xreg_parameters_persistence' is a list like [1, 2] if 2 delta
        # params
        #  Let's assume for now if est_xreg_persistence is true, it's for all xreg
        # components.
        #  A more robust way would be to use
        # explanatory_checked['xreg_parameters_persistence'] if available.
        #  Given the old code: max(explanatory_checked['xreg_parameters_persistence'] or
        # [0])
        #  If 'xreg_parameters_persistence' is not in explanatory_checked or is empty,
        # this defaults to 0.
        # This needs careful check against how 'xreg_parameters_persistence' is defined.
        #  Let's assume xreg_parameters_persistence is a list like [1,1,0] for 3 xregs,
        # 2 have persistence estimates
        if explanatory_checked.get("xreg_parameters_persistence"):
            num_xreg_persistence_params = sum(
                explanatory_checked.get("xreg_parameters_persistence", [])
            )  # Number of true flags
        else:  # Fallback if not defined, assume one per xreg if main flag is true
            num_xreg_persistence_params = explanatory_checked.get("xreg_number", 0)

    # --- Calculate ARIMA params ---
    est_ar = arima_checked.get("ar_estimate", False)
    est_ma = arima_checked.get("ma_estimate", False)
    num_ar_params = sum(arima_checked.get("ar_orders", [])) if est_ar else 0
    num_ma_params = sum(arima_checked.get("ma_orders", [])) if est_ma else 0
    num_arima_params = num_ar_params + num_ma_params

    # --- Calculate Xreg params ---
    est_xreg = explanatory_checked.get("xreg_model", False) and explanatory_checked.get(
        "xreg_estimate", False
    )
    num_xreg_params = explanatory_checked.get("xreg_number", 0) if est_xreg else 0

    # --- Calculate Phi param ---
    est_phi = ets_model and model_is_trendy and phi_dict.get("phi_estimate", False)

    # --- Calculate Initial state params ---
    initials_active = initials_checked.get("initial_type") not in [
        "complete",
        "backcasting",
    ]
    est_init_level = (
        ets_model
        and initials_active
        and initials_checked.get("initial_level_estimate", False)
    )
    est_init_trend = (
        ets_model
        and model_is_trendy
        and initials_active
        and initials_checked.get("initial_trend_estimate", False)
    )

    est_init_seasonal_val = initials_checked.get("initial_seasonal_estimate", False)
    num_initial_seasonal_params = 0
    est_init_seasonal = False
    if model_is_seasonal and initials_active:
        if isinstance(est_init_seasonal_val, bool):
            if est_init_seasonal_val:
                est_init_seasonal = True
                # Needs lags_model to be accurate, using approximation
                num_initial_seasonal_params = sum(
                    lag - 1
                    for lag in model_params["lags_model"][1 + int(model_is_trendy) :]
                )
        elif isinstance(est_init_seasonal_val, (list, np.ndarray)):
            if any(est_init_seasonal_val):
                est_init_seasonal = True
                lags_seasonal = model_params["lags_model"][1 + int(model_is_trendy) :]
                num_initial_seasonal_params = sum(
                    lags_seasonal[i] - 1
                    for i, est in enumerate(est_init_seasonal_val)
                    if est
                )

    est_init_arima = (
        initials_active
        and arima_checked.get("arima_model", False)
        and initials_checked.get("initial_arima_estimate", False)
    )
    num_init_arima_params = (
        initials_checked.get("initial_arima_number", 0) if est_init_arima else 0
    )

    # --- Calculate Constant param ---
    est_constant = constants_checked.get("constant_estimate", False)

    # --- Summing up ---
    total_params = int(
        num_ets_persistence_params  # Changed from num_persistence_params
        + num_xreg_persistence_params  # Added for delta
        + num_arima_params
        + num_xreg_params
        + est_phi
        + est_init_level
        + est_init_trend
        + num_initial_seasonal_params
        + num_init_arima_params
        + est_constant
    )

    # Initialize arrays
    B = np.zeros(total_params)
    Bl = np.zeros(total_params)
    Bu = np.zeros(total_params)
    names = []
    param_idx = 0

    # --- Populate B, Bl, Bu, names based on old initialiser logic ---

    # Determine model-specific initial values for persistence parameters
    #  R has special initialization for "mixed" models (models with both A and M
    # components)
    # See R/adam.R lines 1450-1485
    initial_type = initials_checked.get("initial_type")
    is_mixed = (
        ets_model
        and any(t == "M" for t in [error_type, trend_type, season_type])
        and any(t == "A" for t in [error_type, trend_type, season_type] if t != "N")
    )

    if is_mixed:
        # M*A models (MAM, MAA, MAN) with optimal initial
        if error_type == "M" and trend_type == "A":
            if initial_type not in ["complete", "backcasting"]:
                alpha_init = 0.2
                beta_init = 0.01
                gamma_init = 0.3
            else:
                alpha_init = 0.1
                beta_init = 0.05
                gamma_init = 0.3
        # AAM, AMA
        elif (error_type == "A" and trend_type == "A" and season_type == "M") or (
            error_type == "A" and trend_type == "M" and season_type == "A"
        ):
            alpha_init = 0.01
            beta_init = 0.005
            gamma_init = 0.001
        # MMA
        elif error_type == "M" and trend_type == "M" and season_type == "A":
            alpha_init = 0.01
            beta_init = 0.005
            gamma_init = 0.01
        # MMM and other mixed
        else:
            alpha_init = 0.1
            beta_init = 0.05
            gamma_init = 0.3
    else:
        # Non-mixed models (ANN, AAA, MNM, MMM, etc.) - default values
        alpha_init = 0.1
        beta_init = 0.05
        gamma_init = 0.3

    # ETS Persistence Parameters
    if ets_model:
        # Level Persistence (alpha)
        if est_level:
            B[param_idx] = alpha_init
            if bounds == "usual":
                Bl[param_idx], Bu[param_idx] = 0, 1
            else:
                Bl[param_idx], Bu[param_idx] = -5, 5  # Old code's else
            names.append("alpha")
            param_idx += 1

        # Trend Persistence (beta)
        if est_trend:
            B[param_idx] = beta_init
            if bounds == "usual":
                Bl[param_idx], Bu[param_idx] = 0, 1
            else:
                Bl[param_idx], Bu[param_idx] = -5, 5  # Old code's else
            names.append("beta")
            param_idx += 1

        # Seasonal Persistence (gamma)
        if est_seasonal:
            B[param_idx : param_idx + num_seasonal_persistence_params] = gamma_init
            #  Old code had more complex B init for seasonal persistence based on model
            # types
            # For Bl, Bu:
            if bounds == "usual":
                Bl[param_idx : param_idx + num_seasonal_persistence_params] = 0
                Bu[param_idx : param_idx + num_seasonal_persistence_params] = 1
            else:
                Bl[param_idx : param_idx + num_seasonal_persistence_params] = -5
                Bu[param_idx : param_idx + num_seasonal_persistence_params] = (
                    5  # Old code's else
                )

            if (
                isinstance(seas_est_val, bool) and seas_est_val
            ):  # Single gamma if all estimated together
                if components_number_ets_seasonal > 1:
                    names.extend(
                        [
                            f"gamma{k + 1}"
                            for k in range(num_seasonal_persistence_params)
                        ]
                    )
                elif (
                    num_seasonal_persistence_params == 1
                ):  # handles single seasonal component
                    names.append("gamma")
            elif isinstance(seas_est_val, (list, np.ndarray)):  # individual gammas
                true_indices = [i + 1 for i, est in enumerate(seas_est_val) if est]
                if len(true_indices) == 1 and num_seasonal_persistence_params == 1:
                    names.append("gamma")
                else:
                    names.extend([f"gamma{k}" for k in true_indices])
            elif (
                num_seasonal_persistence_params == 1
            ):  # Catch single seasonal from components_number_ets_seasonal
                names.append("gamma")

            param_idx += num_seasonal_persistence_params

    # Xreg Persistence Parameters (delta) - ADDED SECTION
    if est_xreg_persistence and num_xreg_persistence_params > 0:
        # Default B values from old code (0.01 for A, 0 for M error type)
        # This depends on how xreg_parameters_persistence is structured.
        # Assuming one delta per estimated xreg persistence.
        B[param_idx : param_idx + num_xreg_persistence_params] = (
            0.01 if error_type == "A" else 0
        )
        Bl[param_idx : param_idx + num_xreg_persistence_params] = -5
        Bu[param_idx : param_idx + num_xreg_persistence_params] = 5
        #  Naming based on the number of such parameters. Old code used "delta1",
        # "delta2"...
        #  This relies on num_xreg_persistence_params correctly reflecting count of
        # deltas
        xreg_pers_indices = []
        if explanatory_checked.get("xreg_parameters_persistence"):
            xreg_pers_indices = [
                i + 1
                for i, est_flag in enumerate(
                    explanatory_checked["xreg_parameters_persistence"]
                )
                if est_flag
            ]

        if (
            len(xreg_pers_indices) == num_xreg_persistence_params
            and num_xreg_persistence_params > 0
        ):
            if len(xreg_pers_indices) == 1:
                names.append(
                    f"delta{xreg_pers_indices[0]}"
                )  # Or just "delta" if only one possible
            else:
                names.extend([f"delta{k}" for k in xreg_pers_indices])
        else:  # Fallback naming if mismatch
            names.extend([f"delta{k + 1}" for k in range(num_xreg_persistence_params)])
        param_idx += num_xreg_persistence_params

    # ARIMA Parameters
    if arima_checked.get("arima_model", False):
        ar_orders = np.asarray(arima_checked.get("ar_orders", []), dtype=int)
        i_orders = np.asarray(arima_checked.get("i_orders", []), dtype=int)
        ma_orders = np.asarray(arima_checked.get("ma_orders", []), dtype=int)
        lags_arr = np.asarray(model_params.get("lags", [1]), dtype=int)

        if len(lags_arr) < len(ar_orders):
            lags_arr = np.pad(
                lags_arr, (0, len(ar_orders) - len(lags_arr)), constant_values=1
            )
        elif len(lags_arr) > len(ar_orders):
            lags_arr = lags_arr[: len(ar_orders)]

        pacf_values = (
            np.repeat(0.1, int(np.dot(ar_orders, lags_arr))) if est_ar else np.array([])
        )
        acf_values = (
            np.repeat(-0.1, int(np.dot(ma_orders, lags_arr)))
            if est_ma
            else np.array([])
        )

        # Mirror R: for ARIMA-only with differencing, initialise via ACF/PACF on
        # differenced data.
        if (not ets_model) and (not np.all(i_orders == 0)):
            y_differenced = np.asarray(y_in_sample[ot_logical], dtype=float)
            for lag_value, diff_order in zip(lags_arr, i_orders):
                if diff_order > 0:
                    for _ in range(int(diff_order)):
                        if y_differenced.size <= lag_value:
                            break
                        y_differenced = (
                            y_differenced[lag_value:] - y_differenced[:-lag_value]
                        )

            if np.all(lags_arr <= 1):
                if est_ma and acf_values.size > 0:
                    try:
                        lag_max = max(1, int(np.dot(ma_orders, lags_arr)))
                        acf_calc = calculate_acf(y_differenced, nlags=lag_max)[1:]
                        usable = min(acf_values.size, acf_calc.size)
                        if usable > 0:
                            acf_values[:usable] = acf_calc[:usable]
                    except Exception:
                        pass
                if est_ar and pacf_values.size > 0:
                    try:
                        lag_max = max(1, int(np.dot(ar_orders, lags_arr)))
                        pacf_calc = calculate_pacf(y_differenced, nlags=lag_max)
                        usable = min(pacf_values.size, pacf_calc.size)
                        if usable > 0:
                            pacf_values[:usable] = pacf_calc[:usable]
                    except Exception:
                        pass

        if est_ar:
            for lag_idx, lag_value in enumerate(lags_arr):
                order = int(ar_orders[lag_idx]) if lag_idx < len(ar_orders) else 0
                if order <= 0:
                    continue

                idx = (np.arange(1, order + 1) * lag_value) - 1
                idx = idx[(idx >= 0) & (idx < pacf_values.size)]
                values = (
                    pacf_values[idx] if idx.size == order else np.repeat(0.1, order)
                )
                if np.any(np.isnan(values)):
                    values = np.repeat(0.1, order)
                if np.sum(values) > 1:
                    values = values / np.sum(values) - 0.01

                B[param_idx : param_idx + order] = values
                Bl[param_idx : param_idx + order] = -5
                Bu[param_idx : param_idx + order] = 5
                names.extend([f"ar_{k + 1}" for k in range(order)])
                param_idx += order

        if est_ma:
            for lag_idx, lag_value in enumerate(lags_arr):
                order = int(ma_orders[lag_idx]) if lag_idx < len(ma_orders) else 0
                if order <= 0:
                    continue

                idx = (np.arange(1, order + 1) * lag_value) - 1
                idx = idx[(idx >= 0) & (idx < acf_values.size)]
                values = acf_values[idx] if idx.size == order else np.repeat(0.1, order)
                if np.any(np.isnan(values)):
                    values = np.repeat(0.1, order)
                if np.sum(values) > 1:
                    values = values / np.sum(values) - 0.01

                B[param_idx : param_idx + order] = values
                Bl[param_idx : param_idx + order] = -5
                Bu[param_idx : param_idx + order] = 5
                names.extend([f"ma_{k + 1}" for k in range(order)])
                param_idx += order

    # Explanatory Variable Parameters
    if est_xreg:
        # Use initial values if available from _initialize_xreg_states, otherwise 0
        try:
            initial_xreg_vals = mat_vt[
                components_number_ets
                + model_params["components_number_arima"] : components_number_ets
                + model_params["components_number_arima"]
                + num_xreg_params,
                0,
            ]
            B[param_idx : param_idx + num_xreg_params] = initial_xreg_vals
        except IndexError:
            B[param_idx : param_idx + num_xreg_params] = (
                0  # Fallback if mat_vt shape is unexpected
            )

        # Old code: Bl=-inf, Bu=inf
        Bl[param_idx : param_idx + num_xreg_params] = -np.inf
        Bu[param_idx : param_idx + num_xreg_params] = np.inf
        names.extend([f"xreg_{k + 1}" for k in range(num_xreg_params)])
        param_idx += num_xreg_params

    # Phi (Damping) Parameter
    if est_phi:
        B[param_idx] = 0.95  # Default initial value
        # Old code: Bl=0, Bu=1
        Bl[param_idx], Bu[param_idx] = 0, 1
        names.append("phi")
        param_idx += 1

    # Initial ETS States
    if ets_model and initials_active:
        # Initial Level
        if est_init_level:
            B[param_idx] = mat_vt[0, 0]  # Use value from creator initialization
            # Old code: Bl = -np.inf if error_type == "A" else 0
            if error_type == "A":
                Bl[param_idx], Bu[param_idx] = -np.inf, np.inf
            else:
                Bl[param_idx], Bu[param_idx] = 0, np.inf
            names.append("init_level")
            param_idx += 1

        # Initial Trend
        if est_init_trend:
            B[param_idx] = mat_vt[1, 0]  # Use value from creator initialization
            # Old code: Bl = -np.inf if trend_type == "A" else 0
            if trend_type == "A":
                Bl[param_idx], Bu[param_idx] = -np.inf, np.inf
            else:  # Multiplicative trend
                Bl[param_idx], Bu[param_idx] = 0, np.inf
            names.append("init_trend")
            param_idx += 1

        # Initial Seasonal
        if est_init_seasonal:
            seasonal_comp_start_idx = 1 + int(model_is_trendy)
            param_count_seasonal = 0
            name_idx_seasonal = 1

            current_est_init_seas_val = est_init_seasonal_val
            # Ensure it's a list for iteration if it was a single True boolean
            if isinstance(current_est_init_seas_val, bool):
                current_est_init_seas_val = [
                    current_est_init_seas_val
                ] * components_number_ets_seasonal

            for i in range(components_number_ets_seasonal):
                if current_est_init_seas_val[i]:
                    try:
                        current_lag = model_params["lags_model"][
                            seasonal_comp_start_idx + i
                        ]
                        num_params_this_comp = current_lag - 1
                        if num_params_this_comp > 0:
                            # Get initial values from mat_vt (calculated in creator)
                            initial_vals_seas = mat_vt[
                                seasonal_comp_start_idx + i, :num_params_this_comp
                            ]
                            B[
                                param_idx + param_count_seasonal : param_idx
                                + param_count_seasonal
                                + num_params_this_comp
                            ] = initial_vals_seas

                            # Old code: Bl = -np.inf if season_type == "A" else 0
                            if season_type == "A":
                                Bl[
                                    param_idx + param_count_seasonal : param_idx
                                    + param_count_seasonal
                                    + num_params_this_comp
                                ] = -np.inf
                                Bu[
                                    param_idx + param_count_seasonal : param_idx
                                    + param_count_seasonal
                                    + num_params_this_comp
                                ] = np.inf
                            else:  # Multiplicative seasonality
                                Bl[
                                    param_idx + param_count_seasonal : param_idx
                                    + param_count_seasonal
                                    + num_params_this_comp
                                ] = 0
                                Bu[
                                    param_idx + param_count_seasonal : param_idx
                                    + param_count_seasonal
                                    + num_params_this_comp
                                ] = np.inf

                            season_suffix = (
                                f"_{name_idx_seasonal}"
                                if components_number_ets_seasonal > 1
                                and isinstance(est_init_seasonal_val, list)
                                and len(est_init_seasonal_val) > 1
                                else ""
                            )  # check if est_init_seasonal_val implies multiple
                            # distinct seasonalities
                            names.extend(
                                [
                                    f"init_seas{season_suffix}_{k + 1}"
                                    for k in range(num_params_this_comp)
                                ]
                            )
                            param_count_seasonal += num_params_this_comp
                    except IndexError:
                        # Handle potential index errors if lags_model is incorrect
                        print(
                            f"Warning: Could not determine initial seasonal "
                            f"parameters for component {i + 1}."
                        )
                        pass  # Skip this component's parameters

                    name_idx_seasonal += 1
            param_idx += param_count_seasonal

    # Initial ARIMA States
    if est_init_arima:
        if num_init_arima_params > 0:
            arima_state_start_idx = components_number_ets
            # Get initial values from mat_vt
            try:
                # Calculate expected shape and slice
                num_arima_components = model_params["components_number_arima"]
                if mat_vt.shape[1] >= num_init_arima_params:
                    initial_arima_flat = mat_vt[
                        arima_state_start_idx : arima_state_start_idx
                        + num_arima_components,
                        :num_init_arima_params,
                    ].flatten()
                    # Ensure we don't exceed the length of B array slice
                    slice_len = min(
                        len(initial_arima_flat),
                        len(B[param_idx : param_idx + num_init_arima_params]),
                    )
                    B[param_idx : param_idx + slice_len] = initial_arima_flat[
                        :slice_len
                    ]
                else:
                    B[param_idx : param_idx + num_init_arima_params] = (
                        0 if error_type == "A" else 1
                    )  # Fallback
            except IndexError:
                B[param_idx : param_idx + num_init_arima_params] = (
                    0 if error_type == "A" else 1
                )  # Fallback

            # Old code: Bl = -np.inf if error_type == "A" else 0
            if error_type == "A":
                Bl[param_idx : param_idx + num_init_arima_params] = -np.inf
                Bu[param_idx : param_idx + num_init_arima_params] = np.inf
            else:
                #  Ensure initial values are non-negative for multiplicative errors as
                # per old logic (implicitly by Bl=0)
                B[param_idx : param_idx + num_init_arima_params] = np.maximum(
                    B[param_idx : param_idx + num_init_arima_params], 0
                )  # Old used abs() then Bl=0
                Bl[param_idx : param_idx + num_init_arima_params] = 0
                Bu[param_idx : param_idx + num_init_arima_params] = np.inf
            names.extend([f"init_arima_{k + 1}" for k in range(num_init_arima_params)])
            param_idx += num_init_arima_params

    # Constant Parameter
    if est_constant:
        try:
            constant_idx_in_vt = (
                components_number_ets
                + model_params["components_number_arima"]
                + explanatory_checked.get("xreg_number", 0)
            )
            B[param_idx] = mat_vt[
                constant_idx_in_vt, 0
            ]  # Use value from creator initialization
        except IndexError:
            B[param_idx] = 0  # Fallback

        # Bounds calculation similar to old code
        if (
            arima_checked.get("i_orders")
            and sum(arima_checked.get("i_orders", [])) != 0
        ) or ets_model:
            try:
                valid_ot_logical = ot_logical & np.isfinite(y_in_sample)
                if np.sum(valid_ot_logical) > 1:
                    # Ensure y has positive values for log
                    log_y_valid = y_in_sample[valid_ot_logical]
                    if error_type != "A":
                        log_y_valid = log_y_valid[log_y_valid > 1e-10]
                        if len(log_y_valid) < 2:
                            raise ValueError("Not enough positive values for log diff")
                        diff_log_y = np.diff(np.log(log_y_valid))
                    else:
                        # diff_log_y = np.array([]) # Not needed for Additive
                        pass  # Not needed for Additive case for old bounds

                    diff_y = np.diff(y_in_sample[valid_ot_logical])

                    if error_type == "A":  # Old code direct logic
                        bound_val = np.quantile(diff_y, 0.6)  # Old: not abs()
                        # Ensure bound_val is not NaN if diff_y is empty or all same
                        if not np.isfinite(bound_val) or (
                            len(diff_y) > 0 and np.all(diff_y == diff_y[0])
                        ):  # if diff_y results in non-finite quantile
                            Bl[param_idx], Bu[param_idx] = -np.inf, np.inf
                        else:
                            Bl[param_idx] = -np.abs(
                                bound_val
                            )  # ensure symmetry around 0 if bound_val can be negative
                            Bu[param_idx] = np.abs(bound_val)
                            if Bu[param_idx] <= Bl[param_idx]:
                                Bl[param_idx], Bu[param_idx] = -np.inf, np.inf
                    elif (
                        bounds == "none"
                    ):  # New code path, keep for flexibility, though old didn't have it
                        Bl[param_idx], Bu[param_idx] = -np.inf, np.inf
                    else:  # Multiplicative or Mixed from old logic
                        Bl[param_idx] = np.exp(np.quantile(diff_log_y, 0.4))
                        Bu[param_idx] = np.exp(np.quantile(diff_log_y, 0.6))
                        if Bu[param_idx] <= Bl[param_idx] or not np.isfinite(
                            Bu[param_idx]
                        ):
                            Bl[param_idx], Bu[param_idx] = (
                                0,
                                np.inf,
                            )  # Old: 0, not 1e-10
                else:
                    raise ValueError("Not enough valid observations for diff")
            except Exception:
                # print(f"Constant bounds calculation failed: {e}") # Optional debug
                Bl[param_idx], Bu[param_idx] = (
                    (-np.inf, np.inf) if error_type == "A" else (0, np.inf)
                )  # Old: 0, not 1e-10
        else:  # Not ETS and no differencing
            y_abs_max = (
                np.abs(y_in_sample[ot_logical]).max() if np.sum(ot_logical) > 0 else 1
            )
            current_B_val = B[param_idx]  # Value already set or default 0
            Bl[param_idx] = -max(
                y_abs_max,
                abs(current_B_val) * 1.01 if np.isfinite(current_B_val) else y_abs_max,
            )  # handle non-finite B
            Bu[param_idx] = -Bl[param_idx]
            if not (
                np.isfinite(Bl[param_idx]) and np.isfinite(Bu[param_idx])
            ):  # Fallback if calculation fails
                Bl[param_idx], Bu[param_idx] = -np.inf, np.inf

        # Ensure B is within bounds
        B[param_idx] = np.clip(B[param_idx], Bl[param_idx], Bu[param_idx])

        names.append(constants_checked.get("constant_name", "constant"))
        param_idx += 1

    # Final check for parameter count consistency
    if param_idx != total_params:
        #  print(f"Warning: Parameter count mismatch! Expected {total_params}, got
        # {param_idx}. Adjusting arrays.")
        # Attempt to resize arrays - this might indicate logic error elsewhere
        B = B[:param_idx]
        Bl = Bl[:param_idx]
        Bu = Bu[:param_idx]
    # Return the dictionary in the expected format
    return {"B": B, "Bl": Bl, "Bu": Bu, "names": names}
