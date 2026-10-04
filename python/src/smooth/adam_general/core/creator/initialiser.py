import numpy as np

from smooth.adam_general import _ols  # type: ignore[attr-defined]
from smooth.adam_general.core.utils.polynomials import (
    adam_polynomialiser,
    arima_initials,
)
from smooth.adam_general.core.utils.utils import (
    _mean_r,
    _sum_r,
    msdecompose,
)


def drift_series(y, error_type, ets_model, lags, i_orders):
    """The series on the scale of the constant, from which its starting value and
    bounds come (R's ``adam_driftSeries``): differenced as the ARIMA part of the
    model differences it (in logs for a multiplicative error), whose drift it is,
    or by one step for ETS, where it is the drift of the level. With seasonal
    differencing, the drift is the change over a season, not over one step."""
    y = np.asarray(y, dtype=np.float64).ravel()
    if error_type == "M":
        y = np.log(y)
    i_orders = list(i_orders or [])
    if ets_model or sum(i_orders) == 0:
        return np.diff(y)
    for lag, order in zip(lags, i_orders):
        for _ in range(int(order)):
            y = y[int(lag) :] - y[: -int(lag)]
    return y


def _arima_initialiser(
    y_in_sample,
    ot_logical,
    ets_model,
    error_type,
    season_type,
    model_is_seasonal,
    lags,
    ar_orders,
    i_orders,
    ma_orders,
    ar_estimate,
    ma_estimate,
    arma_parameters,
    bounded,
    smoother,
    xreg_in_sample,
):
    """Hannan-Rissanen starting values of the AR / MA parameters.

    Mirrors R's ``adam_arimaInitialiser`` (R/utils-adam.R); the estimation itself
    is the shared C++ ``arimaHRCore`` (src/headers/arimaInitCore.h). The series
    is the in-sample data on the scale of the ARIMA part (logs for multiplicative
    error), with the ETS part approximated by the decomposition that gives the ETS
    initials (the same smoother), differenced as the model requires, and the
    regressors (``xreg_in_sample``, or None, differenced alike) by OLS on the
    differences. Missing and zero values are gaps, and so is any difference that
    touches one; they are zeros of the centred series. The seasonal ARIMA factors
    that coincide with the ETS seasonality keep the defaults. With ``bounded``, the
    factors that the cost function would reject (not stationary AR, not invertible
    MA, see src/headers/arimaBounds.h) are moved inside the boundary.

    Returns the AR / MA values in the order of B.
    """
    y = np.asarray(y_in_sample, dtype=np.float64).ravel().copy()
    y[~np.asarray(ot_logical, dtype=bool).ravel()] = np.nan
    lags_arr = np.asarray(lags, dtype=np.uint64).ravel()

    with np.errstate(divide="ignore", invalid="ignore"):
        if ets_model:
            decomposition = msdecompose(
                y,
                lags=[int(lag) for lag in lags_arr if lag != 1]
                if model_is_seasonal
                else [1],
                type="multiplicative"
                if "M" in (error_type, season_type)
                else "additive",
                smoother=smoother,
            )
            fitted = np.asarray(decomposition["fitted"], dtype=np.float64)
            y = np.log(y) - np.log(fitted) if error_type == "M" else y - fitted
        elif error_type == "M":
            y = np.log(y)
    # The series and the regressors, differenced alike: a difference that touches
    # a gap is a gap too
    columns = [y]
    if xreg_in_sample is not None:
        columns.append(np.asarray(xreg_in_sample, dtype=np.float64))
    y_diffs = np.column_stack(columns)
    with np.errstate(invalid="ignore"):
        for lag, order in zip(lags_arr, i_orders):
            for _ in range(int(order)):
                y_diffs = y_diffs[int(lag) :] - y_diffs[: -int(lag)]
    y = y_diffs[:, 0].copy()
    observed = np.isfinite(y)
    # Regression on the differences: in levels, an integrated error makes it spurious
    if xreg_in_sample is not None:
        xreg_diffs = np.column_stack([np.ones(len(y)), y_diffs[:, 1:]])
        y = y - xreg_diffs @ _ols.ols(xreg_diffs[observed], y[observed])
    # The gaps are zeros of the centred series (Hannan-Rissanen centres it), which
    # keeps the lags aligned without inventing differences across them
    y[~observed] = _mean_r(y[observed])

    use_level = ~(ets_model & model_is_seasonal & (lags_arr > 1))
    return _ols.arima_hr(
        y,
        np.asarray(ar_orders, dtype=np.uint64),
        np.asarray(ma_orders, dtype=np.uint64),
        lags_arr,
        bool(ar_estimate),
        bool(ma_estimate),
        np.asarray(arma_parameters if arma_parameters else [], dtype=np.float64),
        use_level.astype(np.uint64),
        bool(bounded),
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
    smoother=None,
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
        * (
            initials_checked["initial_type"]
            not in ["backcasting", "complete", "gradient"]
        )
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
        + (
            initials_checked["initial_type"]
            not in ["backcasting", "complete", "gradient"]
        )
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
                        initials_checked["initial_type"]
                        in ["complete", "backcasting", "gradient"]
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
                    if initials_checked["initial_type"] in [
                        "complete",
                        "backcasting",
                        "gradient",
                    ]:
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

    arma_start_index = j
    if arima_checked["arima_model"]:
        if any([arima_checked["ar_estimate"], arima_checked["ma_estimate"]]):
            # Use numpy for element-wise multiplication of orders and lags
            ma_orders_arr = np.array(arima_checked["ma_orders"])
            ar_orders_arr = np.array(arima_checked["ar_orders"])
            i_orders_arr = np.array(arima_checked["i_orders"])
            lags_arr = np.array(lags_dict["lags"])

            arma_values = _arima_initialiser(
                observations_dict["y_in_sample"],
                observations_dict["ot_logical"],
                model_type_dict["ets_model"],
                model_type_dict["error_type"],
                model_type_dict["season_type"],
                model_type_dict["model_is_seasonal"],
                lags_arr,
                ar_orders_arr,
                i_orders_arr,
                ma_orders_arr,
                arima_checked["ar_estimate"],
                arima_checked["ma_estimate"],
                arima_checked["arma_parameters"],
                bounds != "none",
                smoother,
                adam_created["mat_wt"][
                    : len(observations_dict["y_in_sample"]),
                    components_dict["components_number_ets"]
                    + components_dict["components_number_arima"] : components_dict[
                        "components_number_ets"
                    ]
                    + components_dict["components_number_arima"]
                    + explanatory_checked["xreg_number"],
                ]
                if explanatory_checked["xreg_model"]
                else None,
            )

            for i, lag in enumerate(lags_dict["lags"]):
                for part, prefix, required, estimate in (
                    ("ar_orders", "phi", "ar_required", "ar_estimate"),
                    ("ma_orders", "theta", "ma_required", "ma_estimate"),
                ):
                    order = arima_checked[part][i]
                    if order and arima_checked[required] and arima_checked[estimate]:
                        B[j : j + order] = arma_values[
                            j - arma_start_index : j - arma_start_index + order
                        ]
                        Bl[j : j + order] = -5
                        Bu[j : j + order] = 5
                        names.extend([f"{prefix}{k + 1}[{lag}]" for k in range(order)])
                        j += order

    # The polynomials at the starting values of the AR / MA parameters
    arima_polynomials = None
    if arima_checked["arima_model"] and adam_cpp is not None:
        arima_polynomials = adam_polynomialiser(
            adam_cpp,
            B[arma_start_index:j],
            arima_checked["ar_orders"],
            arima_checked["i_orders"],
            arima_checked["ma_orders"],
            arima_checked["ar_estimate"],
            arima_checked["ma_estimate"],
            arima_checked["arma_parameters"] or [],
            lags_dict["lags"],
        )

    #  NOTE: Removed backcasting from initialiser - CF already handles backcasting for
    # complete/backcasting modes
    # This was causing double backcasting which led to different results than R

    if (
        model_type_dict["ets_model"]
        and initials_checked["initial_type"]
        not in ["backcasting", "complete", "gradient"]
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
                        lag = lags_dict["lags_model"][seasonal_index]

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
                        names.extend([f"seasonal{k + 1}_{m}" for m in range(1, lag)])
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
                names.extend([f"seasonal_{m}" for m in range(1, temp_lag)])
                j += temp_lag - 1
    if (
        initials_checked["initial_type"] not in ["backcasting", "complete", "gradient"]
        and arima_checked["arima_model"]
        and initials_checked["initial_arima_estimate"]
    ):
        # The creator holds the pre-sample values in the last ARIMA state (see
        # arima_initials), taken through the ARI polynomial of the starting values
        m = initials_checked["initial_arima_number"]
        B[j : j + m] = arima_initials(
            arima_polynomials["ari_polynomial"],
            adam_created["arima_pre_sample"],
            model_type_dict["error_type"],
        )
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
        # The constant is the intercept of ARIMA, so it needs to agree with the AR
        # starting values (R: sum(arimaPolynomials$arPolynomial))
        if arima_polynomials is not None and not model_type_dict["ets_model"]:
            ar_at_one = _sum_r(arima_polynomials["ar_polynomial"])
            B[j - 1] = (
                B[j - 1] ** ar_at_one
                if model_type_dict["error_type"] == "M"
                else B[j - 1] * ar_at_one
            )
        names.append(constants_checked["constant_name"] or "constant")
        if model_type_dict["ets_model"] or (
            arima_checked["i_orders"] is not None
            and sum(arima_checked["i_orders"]) != 0
        ):
            drift = drift_series(
                observations_dict["y_in_sample"][observations_dict["ot_logical"]],
                model_type_dict["error_type"],
                model_type_dict["ets_model"],
                lags_dict["lags"],
                arima_checked["i_orders"],
            )
            if model_type_dict["error_type"] == "A":
                Bu[j - 1] = np.quantile(drift, 0.6)
                Bl[j - 1] = -Bu[j - 1]
            else:
                Bu[j - 1] = np.exp(np.quantile(drift, 0.6))
                Bl[j - 1] = np.exp(np.quantile(drift, 0.4))

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
        # Match R's `adam_checkOptimizer` (R/utils-adam.R:241ff) which uses a
        # near-zero lower bound on the distribution shape parameter. A
        # tighter bound (we previously used 0.25) changes NLopt's bounded
        # initial simplex even when the boundary is never active at the
        # optimum, which made Python's Nelder-Mead land in a different
        # local minimum than R's nloptr on the same cost surface (e.g.
        # AirPassengers with model="MAM", distribution="dgnorm"). The
        # soft penalty for shape < 0.25 in ``cost_functions.py`` still
        # protects against the genuinely-pathological region where the
        # density becomes too peaked.
        Bl[j - 1] = 1e-10
        Bu[j - 1] = np.inf
    return {"B": B, "Bl": Bl, "Bu": Bu, "names": names}
