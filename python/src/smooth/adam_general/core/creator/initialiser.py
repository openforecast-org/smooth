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


def _initialiser_persistence(
    error_type,
    trend_type,
    season_type,
    adam_ets,
    backcast,
    estimate_vector,
    components_number_ets_seasonal,
    bounds,
):
    """The starting values of the smoothing parameters, with their bounds and names:
    the small ones of the multiplicative models that are not safe for the
    conventional ETS (R's adam_initialiserPersistence)."""
    model = error_type + trend_type + season_type
    values = (0.1, 0.05, 0.3)
    if not adam_ets and "M" in model:
        # AAM, AMA, and with backcasting MAA and MAM
        if model in ("AAM", "AMA") or (backcast and model in ("MAA", "MAM")):
            values = (0.01, 0.005, 0.001)
        # MMA is the worst: everything near zero
        elif model == "MMA":
            values = (0.01, 0.005, 0.01)
        elif error_type == "M" and trend_type == "A" and not backcast:
            values = (0.2, 0.01, 0.3)
    n = components_number_ets_seasonal
    gammas = [f"gamma{i}" for i in range(1, n + 1)] if n > 1 else ["gamma"]
    candidates = zip(
        [values[0], values[1]] + [values[2]] * n, ["alpha", "beta"] + gammas[:n]
    )
    chosen = [pair for pair, estimate in zip(candidates, estimate_vector) if estimate]
    low, up = (0, 1) if bounds == "usual" else (-5, 5)
    return (
        [value for value, _ in chosen],
        [low] * len(chosen),
        [up] * len(chosen),
        [name for _, name in chosen],
    )


def _initialiser_arma(
    model_type_dict,
    arima_checked,
    lags_dict,
    observations_dict,
    bounds,
    smoother,
    xreg_in_sample,
    adam_cpp,
):
    """The starting values of the AR and MA parameters, filled in lags-wise, with
    their bounds and names, and the polynomials at them (R's adam_initialiserArma).
    """
    B, names = [], []
    if arima_checked["ar_estimate"] or arima_checked["ma_estimate"]:
        arma_values = _arima_initialiser(
            observations_dict["y_in_sample"],
            observations_dict["ot_logical"],
            model_type_dict["ets_model"],
            model_type_dict["error_type"],
            model_type_dict["season_type"],
            model_type_dict["model_is_seasonal"],
            np.array(lags_dict["lags"]),
            np.array(arima_checked["ar_orders"]),
            np.array(arima_checked["i_orders"]),
            np.array(arima_checked["ma_orders"]),
            arima_checked["ar_estimate"],
            arima_checked["ma_estimate"],
            arima_checked["arma_parameters"],
            bounds != "none",
            smoother,
            xreg_in_sample,
        )
        for i, lag in enumerate(lags_dict["lags"]):
            for part, prefix, required, estimate in (
                ("ar_orders", "phi", "ar_required", "ar_estimate"),
                ("ma_orders", "theta", "ma_required", "ma_estimate"),
            ):
                order = arima_checked[part][i]
                if order and arima_checked[required] and arima_checked[estimate]:
                    B.extend(arma_values[len(B) : len(B) + order])
                    names.extend(f"{prefix}{k + 1}[{lag}]" for k in range(order))
    arima_polynomials = None
    if adam_cpp is not None:
        arima_polynomials = adam_polynomialiser(
            adam_cpp,
            np.asarray(B, dtype=np.float64),
            arima_checked["ar_orders"],
            arima_checked["i_orders"],
            arima_checked["ma_orders"],
            arima_checked["ar_estimate"],
            arima_checked["ma_estimate"],
            arima_checked["arma_parameters"] or [],
            lags_dict["lags"],
        )
    return B, [-5] * len(B), [5] * len(B), names, arima_polynomials


def _initialiser_states(
    mat_vt, model_type_dict, initials_checked, components_dict, lags_dict
):
    """The initial level, trend and seasonal states from the first column of mat_vt,
    with their bounds and names (R's adam_initialiserStates)."""
    B, Bl, Bu, names = [], [], [], []
    if initials_checked["initial_level_estimate"]:
        B.append(mat_vt[0, 0])
        Bl.append(-np.inf if model_type_dict["error_type"] == "A" else 0)
        Bu.append(np.inf)
        names.append("level")
    if (
        model_type_dict["model_is_trendy"]
        and initials_checked["initial_trend_estimate"]
    ):
        additive = model_type_dict["trend_type"] == "A"
        B.append(mat_vt[1, 0])
        Bl.append(-np.inf if additive else 0)
        Bu.append(np.inf if additive else 2)
        names.append("trend")
    # The seasonal states are numbered by the component with several of them
    n = components_dict["components_number_ets_seasonal"]
    estimates = initials_checked["initial_seasonal_estimate"][:n]
    low = -np.inf if model_type_dict["season_type"] == "A" else 0
    for k, estimate in enumerate(
        estimates if model_type_dict["model_is_seasonal"] else []
    ):
        if estimate:
            row = components_dict["components_number_ets"] - n + k
            positions = range(1, lags_dict["lags_model"][row])
            B.extend(mat_vt[row, : len(positions)])
            Bl.extend([low] * len(positions))
            Bu.extend([np.inf] * len(positions))
            names.extend(f"seasonal{k + 1 if n > 1 else ''}_{m}" for m in positions)
    return B, Bl, Bu, names


def _initialiser_constant(
    constant, constant_name, ar_polynomial, model_type_dict, arima_checked, y, lags
):
    """The starting value of the constant, which agrees with the AR starting values
    in ARIMA, with its bounds from the drift of the series (R's
    adam_initialiserConstant). ``y`` is the observed in-sample data."""
    error_type = model_type_dict["error_type"]
    if ar_polynomial is not None and not model_type_dict["ets_model"]:
        ar_at_one = _sum_r(ar_polynomial)
        constant = constant**ar_at_one if error_type == "M" else constant * ar_at_one
    name = [constant_name or "constant"]
    i_orders = arima_checked["i_orders"]
    if not model_type_dict["ets_model"] and not (i_orders and sum(i_orders) != 0):
        upper = max(np.max(np.abs(y)), abs(constant) * 1.01)
        return [constant], [-upper], [upper], name
    drift = drift_series(y, error_type, model_type_dict["ets_model"], lags, i_orders)
    if error_type == "A":
        upper = np.quantile(drift, 0.6)
        lower = -upper
    else:
        upper = np.exp(np.quantile(drift, 0.6))
        lower = np.exp(np.quantile(drift, 0.4))
    # Failsafe for weird cases, when upper bound is the same or lower than the lower
    if upper <= lower:
        upper, lower = np.inf, -np.inf if error_type == "A" else 0
    # Failsafe for cases, when the B is outside of bounds
    if constant <= lower:
        lower = -np.inf if error_type == "A" else 0
    if constant >= upper:
        upper = np.inf
    return [constant], [lower], [upper], name


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
    adam_ets=False,
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
    # The vector of logicals for persistence elements, one per seasonal component
    persistence_estimate_vector = [
        persistence_checked["persistence_level_estimate"],
        model_type_dict["model_is_trendy"]
        and persistence_checked["persistence_trend_estimate"],
    ]
    if model_type_dict["model_is_seasonal"]:
        persistence_estimate_vector.extend(
            persistence_checked["persistence_seasonal_estimate"]
        )
    backcast = initials_checked["initial_type"] in [
        "backcasting",
        "complete",
        "gradient",
    ]
    error_type = model_type_dict["error_type"]
    mat_vt = adam_created["mat_vt"]
    xreg_start = (
        components_dict["components_number_ets"]
        + components_dict["components_number_arima"]
    )

    # The parts of B in this order: the persistence of states and of xreg, phi, AR
    # and MA parameters, the initials of ETS, ARIMA and xreg, the constant and the
    # other parameter, each with its bounds and names
    pieces = []
    if (
        model_type_dict["ets_model"]
        and persistence_checked["persistence_estimate"]
        and any(persistence_estimate_vector)
    ):
        pieces.append(
            _initialiser_persistence(
                error_type,
                model_type_dict["trend_type"],
                model_type_dict["season_type"],
                adam_ets,
                backcast,
                persistence_estimate_vector,
                components_dict["components_number_ets_seasonal"],
                bounds,
            )
        )

    if (
        explanatory_checked["xreg_model"]
        and persistence_checked["persistence_xreg_estimate"]
    ):
        n = max(explanatory_checked["xreg_parameters_persistence"])
        start = 0.01 if error_type == "A" else 0
        deltas = [f"delta{i + 1}" for i in range(n)]
        pieces.append(([start] * n, [-5] * n, [5] * n, deltas))

    if model_type_dict["ets_model"] and phi_dict["phi_estimate"]:
        pieces.append(([0.95], [0], [1], ["phi"]))

    arima_polynomials = None
    if arima_checked["arima_model"]:
        xreg_in_sample = None
        if explanatory_checked["xreg_model"]:
            xreg_in_sample = adam_created["mat_wt"][
                : len(observations_dict["y_in_sample"]),
                xreg_start : xreg_start + explanatory_checked["xreg_number"],
            ]
        *arma, arima_polynomials = _initialiser_arma(
            model_type_dict,
            arima_checked,
            lags_dict,
            observations_dict,
            bounds,
            smoother,
            xreg_in_sample,
            adam_cpp,
        )
        pieces.append(arma)

    if (
        model_type_dict["ets_model"]
        and not backcast
        and initials_checked["initial_estimate"]
    ):
        pieces.append(
            _initialiser_states(
                mat_vt, model_type_dict, initials_checked, components_dict, lags_dict
            )
        )

    # The ARIMA initials: the creator holds the pre-sample values in the last ARIMA
    # state (see arima_initials), taken through the ARI polynomial of the starting
    # values. The multiplicative ones are positive to avoid errors
    if (
        not backcast
        and arima_checked["arima_model"]
        and initials_checked["initial_arima_estimate"]
    ):
        m = initials_checked["initial_arima_number"]
        values = arima_initials(
            arima_polynomials["ari_polynomial"],
            adam_created["arima_pre_sample"],
            error_type,
        )
        if error_type != "A":
            values = np.abs(values)
        low = -np.inf if error_type == "A" else 0
        states = [f"ARIMAState{k}" for k in range(1, m + 1)]
        pieces.append((list(values), [low] * m, [np.inf] * m, states))

    # The initials of the estimated regressors (excluded from B under "complete")
    if (
        initials_checked["initial_xreg_estimate"]
        and explanatory_checked["xreg_model"]
        and initials_checked["initial_type"] != "complete"
    ):
        estimated = np.asarray(explanatory_checked["xreg_parameters_estimated"]) == 1
        n = int(np.sum(estimated))
        regressors = np.asarray(explanatory_checked["xreg_names"])[estimated]
        rows = xreg_start + np.flatnonzero(estimated)
        pieces.append((mat_vt[rows, 0], [-np.inf] * n, [np.inf] * n, list(regressors)))

    if constants_checked["constant_estimate"]:
        row = xreg_start + explanatory_checked["xreg_number"]
        pieces.append(
            _initialiser_constant(
                mat_vt[row, 0] if mat_vt.shape[0] > row else 0,
                constants_checked["constant_name"],
                None
                if arima_polynomials is None
                else arima_polynomials["ar_polynomial"],
                model_type_dict,
                arima_checked,
                observations_dict["y_in_sample"][observations_dict["ot_logical"]],
                lags_dict["lags"],
            )
        )

    # The shape of the distribution, with R's near-zero lower bound (a tighter one
    # changes NLopt's initial simplex even when it is never active)
    if other_parameter_estimate:
        pieces.append(([other_value], [1e-10], [np.inf], ["other"]))

    B, Bl, Bu = (
        np.array([value for piece in pieces for value in piece[i]], dtype=np.float64)
        for i in range(3)
    )
    names = [name for piece in pieces for name in piece[3]]
    return {"B": B, "Bl": Bl, "Bu": Bu, "names": names}
