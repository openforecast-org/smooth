import warnings

import numpy as np

from .initialization import _initialize_states

# Suppress divide by zero warnings
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="divide by zero encountered"
)


def creator(
    # Model type info
    model_type_dict,
    # Lags info
    lags_dict,
    # Profiles
    profiles_dict,
    # Observation info
    observations_dict,
    # Parameter dictionaries
    persistence_checked,
    initials_checked,
    arima_checked,
    constants_checked,
    phi_dict,
    # Components info
    components_dict,
    explanatory_checked=None,
    smoother="global",
):
    """
    Create state-space matrices for ADAM model representation.

    This function constructs the complete state-space representation of an ADAM
    model by building four fundamental matrices and initializing the state vector.
    These matrices define the model's dynamics and are used throughout estimation
    and forecasting.

    The ADAM state-space form is:

    .. math::

        y_t &= o_t(w_t' v_{t-l} + r_t \\epsilon_t)

        v_t &= F v_{t-l} + g \\epsilon_t

    where:

    - :math:`v_t` is the **state vector** (mat_vt) containing level, trend,
      seasonal, ARIMA, and regression components
    - :math:`w_t` is the **measurement vector** (mat_wt) extracting the observed
      value from states
    - :math:`F` is the **transition matrix** (mat_f) governing state evolution
    - :math:`g` is the **persistence vector** (vec_g) controlling error
      propagation (smoothing parameters)
    - :math:`\\epsilon_t` is the error term

    **Matrix Construction Process**:

    1. **Extract Parameters**: Parse model specification and component counts
    2. **Allocate Matrices**: Initialize matrices with appropriate dimensions
    3. **Setup Persistence** (vec_g): Place smoothing parameters (α, β, γ) for
       ETS components
    4. **Setup Measurement** (mat_wt): Define how states map to observations
    5. **Setup Transition** (mat_f): Define state evolution (identity for most,
       damping for trend)
    6. **ARIMA Polynomials**: Create companion matrices for AR/MA components
       if present
    7. **Initialize States** (mat_vt): Set initial values via backcasting,
       optimization, or user input

    **Matrix Dimensions**:

    - **mat_vt**: (n_components × T+max_lag) - State vector over time
    - **mat_wt**: (T × n_components) - Measurement weights
    - **mat_f**: (n_components × n_components) - Transition matrix
    - **vec_g**: (n_components × 1) - Persistence vector

    where T is the number of observations and n_components includes:

    - Level (1)
    - Trend (0 or 1)
    - Seasonal (sum of (lag_i - 1) for each seasonal lag)
    - ARIMA (AR + MA orders)
    - Regressors (number of exogenous variables)
    - Constant (0 or 1)

    Parameters
    ----------
    model_type_dict : dict
        Model specification containing:

        - 'ets_model': Whether ETS components exist
        - 'arima_model': Whether ARIMA components exist
        - 'xreg_model': Whether regressors are present
        - 'error_type': 'A' (additive) or 'M' (multiplicative)
        - 'trend_type': 'N', 'A', 'Ad', 'M', 'Md'
        - 'season_type': 'N', 'A', 'M'
        - 'model_is_trendy': Boolean for trend presence
        - 'model_is_seasonal': Boolean for seasonality presence
        - 'damped': Whether trend is damped

    lags_dict : dict
        Lag structure containing:

        - 'lags': Primary lag vector (e.g., [1, 12])
        - 'lags_model': Lag for each state component
        - 'lags_model_all': Full lag specification as column vector
        - 'lags_model_max': Maximum lag value (lookback period)
        - 'lags_model_seasonal': Lags for seasonal components only

    profiles_dict : dict
        Time-varying parameter configuration containing:

        - 'profiles_recent_table': Recent profile values
        - 'profiles_recent_provided': Whether user provided profiles
        - 'index_lookup_table': Index mapping for profile access

    observations_dict : dict
        Observation information containing:

        - 'obs_in_sample': Number of in-sample observations
        - 'obs_all': Total observations including holdout
        - 'obs_states': State dimension including pre-sample

    persistence_checked : dict
        Validated persistence parameters containing:

        - 'persistence_estimate': Whether to estimate smoothing parameters
        - 'persistence_level': α (level smoothing) - fixed value or None
        - 'persistence_trend': β (trend smoothing) - fixed value or None
        - 'persistence_seasonal': γ (seasonal smoothing) - list of values or None
        - 'persistence_xreg': Regressor persistence (if adaptive regressors)

    initials_checked : dict
        Initial state specification containing:

        - 'initial_type': Initialization method ('optimal', 'backcasting', 'provided')
        - 'initial_level': Fixed level initial (if provided)
        - 'initial_trend': Fixed trend initial (if provided)
        - 'initial_seasonal': Fixed seasonal initials (if provided)
        - 'initial_arima': Fixed ARIMA initials (if provided)

    arima_checked : dict
        ARIMA specification containing:

        - 'arima_model': Whether ARIMA is present
        - 'ar_orders': AR orders for each lag
        - 'ma_orders': MA orders for each lag
        - 'i_orders': Integration orders
        - 'arma_parameters': AR and MA coefficients (if provided)

    constants_checked : dict
        Constant term specification containing:

        - 'constant_required': Whether constant is included
        - 'constant_value': Fixed constant (if provided)

    phi_dict : dict
        Damping parameter containing:

        - 'phi': Damping value (0 < φ ≤ 1)
        - 'phi_estimate': Whether to estimate φ

    components_dict : dict
        Component counts containing:

        - 'components_number_all': Total state dimension
        - 'components_number_ets': Number of ETS components
        - 'components_number_ets_seasonal': Number of seasonal components
        - 'components_number_ets_non_seasonal': Level + trend count
        - 'components_number_arima': ARIMA state count
        - Additional component counts

    explanatory_checked : dict, optional
        External regressors specification containing:

        - 'xreg_model': Whether regressors exist
        - 'xreg_number': Number of regressors
        - 'mat_xt': Regressor data matrix (T × p)

    smoother : str, default="lowess"
        Smoother type for time series decomposition used in initial state estimation.

        - "lowess": Uses LOWESS for both trend and seasonal extraction
        - "ma": Uses moving average for both
        - "global": Uses lowess for trend and "ma" for seasonality

    Returns
    -------
    dict
        Dictionary containing created state-space matrices:

        - **'mat_vt'** (numpy.ndarray): State vector matrix, shape (n_components,
        T+max_lag).
          Fortran-ordered (column-major) for C++ compatibility. Contains all state
          components over time plus pre-sample period.

        - **'mat_wt'** (numpy.ndarray): Measurement matrix, shape (T, n_components).
          Fortran-ordered. Each row extracts the observation from the state vector
          at that time point.

        - **'mat_f'** (numpy.ndarray): Transition matrix, shape (n_components,
        n_components).
          Fortran-ordered. Defines how states evolve. Typically near-identity with
          damping and ARIMA companion matrices.

        - **'vec_g'** (numpy.ndarray): Persistence vector, shape (n_components,).
          Contains smoothing parameters (α, β, γ) and controls error propagation
          through states.

        - **'arima_polynomials'** (dict or None): If ARIMA is present, contains:

          * 'ar_polynomial': AR polynomial coefficients
          * 'ma_polynomial': MA polynomial coefficients
          * Companion matrices for state-space representation

    Notes
    -----
    **Matrix Order and Alignment**:

    All matrices use **Fortran order** (column-major) for efficient interfacing with
    C++ estimation routines via pybind11. This is critical for performance.

    **State Vector Structure**:

    The state vector mat_vt is organized as::

        [Level,
         Trend (if present),
         Seasonal_1[1], ..., Seasonal_1[m1-1],
         Seasonal_2[1], ..., Seasonal_2[m2-1],
         ...,
         ARIMA_states,
         Regressor_1, ..., Regressor_p,
         Constant (if present)]

    **Initialization**:

    Initial states (first max_lag columns of mat_vt) are populated based on:

    - **"backcasting"**: Iteratively fit model backwards from observation 1
    - **"optimal"**: Optimize initials along with other parameters
    - **"provided"**: Use user-specified values
    - **Default**: Simple heuristics (e.g., mean for level, classical decomposition for
    seasonals)

    **Performance**:

    Matrix creation is fast (< 1ms typically). The created matrices are then passed
    to the C++ fitter which operates on them in-place during estimation.

    See Also
    --------
    initialiser : Create initial parameter vector B and bounds for optimization
    filler : Fill matrices with parameter values from vector B
    architector : Determine model architecture before matrix creation
    estimator : Main estimation function that uses created matrices

    Examples
    --------
    Create matrices for a simple ETS(A,N,N) model::

        >>> adam_created = creator(
        ...     model_type_dict={'ets_model': True, 'error_type': 'A',
        ...                      'trend_type': 'N', 'season_type': 'N',
        ...                      'model_is_trendy': False,
        ...                      'model_is_seasonal': False, ...},
        ...     lags_dict={'lags': np.array([1]), 'lags_model': [1],
        ...                'lags_model_max': 1, ...},
        ...     profiles_dict={'profiles_recent_table': np.zeros((1, 1)), ...},
        ...     observations_dict={'obs_in_sample': 100, 'obs_states': 101, ...},
        ...     persistence_checked={'persistence_estimate': True, ...},
        ...     initials_checked={'initial_type': 'optimal', ...},
        ...     arima_checked={'arima_model': False, ...},
        ...     constants_checked={'constant_required': False, ...},
        ...     phi_dict={'phi': 1.0, 'phi_estimate': False},
        ... components_dict={'components_number_all': 1, 'components_number_ets': 1,
        ...}
        ... )
        >>> print(adam_created['mat_vt'].shape)  # (1, 101)
        >>> print(adam_created['vec_g'].shape)   # (1,)

    Create matrices for ARIMA(1,1,1)::

        >>> adam_created = creator(
        ...     model_type_dict={'arima_model': True, 'ets_model': False, ...},
        ...     arima_checked={'arima_model': True, 'ar_orders': [1], 'ma_orders': [1],
        ...                    'i_orders': [1], ...},
        ...     ...
        ... )
        >>> # ARIMA companion matrices in arima_polynomials
        >>> print(adam_created['arima_polynomials'])
    """
    # Extract data and parameters
    model_params = _extract_model_parameters(
        model_type_dict,
        components_dict,
        lags_dict,
        observations_dict,
        phi_dict,
        profiles_dict,
    )
    model_params["smoother"] = smoother

    # Setup matrices
    matrices = _setup_matrices(
        model_params,
        observations_dict,
        components_dict,
        explanatory_checked,
        constants_checked,
    )

    # Setup persistence vector and transition matrix
    matrices = _setup_persistence_vector(
        matrices,
        model_params,
        persistence_checked,
        arima_checked,
        constants_checked,
        explanatory_checked,
    )

    # Setup measurement vector and handle damping
    matrices = _setup_measurement_vector(matrices, model_params, explanatory_checked)

    # Handle ARIMA polynomials if needed
    matrices = _handle_polynomial_setup(matrices, model_params, arima_checked)
    # Initialize states
    matrices = _initialize_states(
        matrices,
        model_params,
        profiles_dict,
        observations_dict,
        initials_checked,
        arima_checked,
        explanatory_checked,
        constants_checked,
    )

    # Return created matrices
    return {
        "mat_vt": matrices["mat_vt"],
        "mat_wt": matrices["mat_wt"],
        "mat_f": matrices["mat_f"],
        "vec_g": matrices["vec_g"],
        "arima_polynomials": matrices.get("arima_polynomials", None),
    }


def _extract_model_parameters(
    model_type_dict,
    components_dict,
    lags_dict,
    observations_dict,
    phi_dict,
    profiles_dict,
):
    """
    Extract and organize model parameters from various dictionaries.

    Args:
        model_type_dict: Dictionary containing model type information
        components_dict: Dictionary containing component information
        lags_dict: Dictionary containing lags information
        observations_dict: Dictionary containing observation information
        phi_dict: Dictionary containing phi parameters
        profiles_dict: Dictionary containing profiles information

    Returns:
        Dict: Dictionary containing organized model parameters
    """
    # Extract observation values
    obs_states = observations_dict["obs_states"]
    obs_in_sample = observations_dict["obs_in_sample"]
    obs_all = observations_dict["obs_all"]
    ot_logical = observations_dict["ot_logical"]
    y_in_sample = observations_dict["y_in_sample"]
    obs_nonzero = observations_dict["obs_nonzero"]

    # Extract values from dictionaries
    ets_model = model_type_dict["ets_model"]
    e_type = model_type_dict["error_type"]
    t_type = model_type_dict["trend_type"]
    s_type = model_type_dict["season_type"]
    model_is_trendy = model_type_dict["model_is_trendy"]
    model_is_seasonal = model_type_dict["model_is_seasonal"]

    # Extract component numbers
    components_number_ets = components_dict["components_number_ets"]
    components_number_ets_seasonal = components_dict["components_number_ets_seasonal"]
    components_number_arima = components_dict.get("components_number_arima", 0)

    # Extract phi parameter
    phi = phi_dict["phi"]

    # Extract lags info
    lags = lags_dict["lags"]
    lags_model = lags_dict["lags_model"]
    lags_model_arima = lags_dict["lags_model_arima"]
    lags_model_all = lags_dict["lags_model_all"]
    lags_model_max = lags_dict["lags_model_max"]
    lags_original = lags_dict.get("lags_original", lags) or [1]

    # Extract profiles info
    profiles_recent_table = profiles_dict["profiles_recent_table"]
    profiles_recent_provided = profiles_dict["profiles_recent_provided"]

    return {
        "obs_states": obs_states,
        "obs_in_sample": obs_in_sample,
        "obs_all": obs_all,
        "ot_logical": ot_logical,
        "y_in_sample": y_in_sample,
        "obs_nonzero": obs_nonzero,
        "ets_model": ets_model,
        "e_type": e_type,
        "t_type": t_type,
        "s_type": s_type,
        "model_is_trendy": model_is_trendy,
        "model_is_seasonal": model_is_seasonal,
        "components_number_ets": components_number_ets,
        "components_number_ets_seasonal": components_number_ets_seasonal,
        "components_number_arima": components_number_arima,
        "phi": phi,
        "lags": lags,
        "lags_model": lags_model,
        "lags_model_arima": lags_model_arima,
        "lags_model_all": lags_model_all,
        "lags_model_max": lags_model_max,
        "lags_original": lags_original,
        "profiles_recent_table": profiles_recent_table,
        "profiles_recent_provided": profiles_recent_provided,
    }


def _setup_matrices(
    model_params,
    observations_dict,
    components_dict,
    explanatory_checked,
    constants_checked,
):
    """
    Setup the state, measurement, and transition matrices for the model.

    Args:
        model_params: Dictionary containing model parameters
        observations_dict: Dictionary containing observation information
        components_dict: Dictionary containing component information
        explanatory_checked: Dictionary of explanatory variables parameters
        constants_checked: Dictionary containing constant parameters

    Returns:
        Dict: Dictionary containing initialized matrices
    """
    # Get parameters
    obs_states = model_params["obs_states"]
    obs_all = model_params["obs_all"]
    components_number_ets = model_params["components_number_ets"]
    components_number_arima = model_params["components_number_arima"]

    # Matrix of states. Time in columns, components in rows
    mat_vt = np.full(
        (
            components_number_ets
            + components_number_arima
            + explanatory_checked["xreg_number"]
            + constants_checked["constant_required"],
            obs_states,
        ),
        np.nan,
    )

    # Measurement rowvector
    mat_wt = np.ones(
        (
            obs_all,
            components_number_ets
            + components_number_arima
            + explanatory_checked["xreg_number"]
            + constants_checked["constant_required"],
        )
    )

    # Transition matrix
    mat_f = np.eye(
        components_number_ets
        + components_number_arima
        + explanatory_checked["xreg_number"]
        + constants_checked["constant_required"]
    )

    # Persistence vector
    vec_g = np.zeros(
        (
            components_number_ets
            + components_number_arima
            + explanatory_checked["xreg_number"]
            + constants_checked["constant_required"],
            1,
        )
    )

    return {"mat_vt": mat_vt, "mat_wt": mat_wt, "mat_f": mat_f, "vec_g": vec_g}


def _setup_measurement_vector(matrices, model_params, explanatory_checked):
    """
    Setup the measurement vector and handle damping parameters.

    Args:
        matrices: Dictionary containing model matrices
        model_params: Dictionary containing model parameters
        explanatory_checked: Dictionary of explanatory variables parameters

    Returns:
        Dict: Updated matrices dictionary
    """
    # Get matrices
    mat_wt = matrices["mat_wt"]
    mat_f = matrices["mat_f"]

    # Get parameters
    ets_model = model_params["ets_model"]
    model_is_trendy = model_params["model_is_trendy"]
    phi = model_params["phi"]
    components_number_ets = model_params["components_number_ets"]
    components_number_arima = model_params["components_number_arima"]

    # If xreg are provided, then fill in the respective values in Wt vector
    if explanatory_checked["xreg_model"]:
        n_xreg_rows = len(explanatory_checked["xreg_data"])
        col_start = components_number_ets + components_number_arima
        col_end = col_start + explanatory_checked["xreg_number"]
        mat_wt[:n_xreg_rows, col_start:col_end] = explanatory_checked["xreg_data"]

    # Damping parameter value
    if ets_model and model_is_trendy:
        mat_f[0, 1] = phi
        mat_f[1, 1] = phi
        mat_wt[:, 1] = phi

    matrices["mat_wt"] = mat_wt
    matrices["mat_f"] = mat_f

    return matrices


def _setup_persistence_vector(
    matrices,
    model_params,
    persistence_checked,
    arima_checked,
    constants_checked,
    explanatory_checked,
):
    """
    Setup the persistence vector for the model.

    Args:
        matrices: Dictionary containing model matrices
        model_params: Dictionary containing model parameters
        persistence_checked: Dictionary of persistence parameters
        arima_checked: Dictionary of ARIMA parameters
        constants_checked: Dictionary containing constant parameters
        explanatory_checked: Dictionary of explanatory variables parameters

    Returns:
        Dict: Updated matrices dictionary
    """
    # Get matrices
    mat_f = matrices["mat_f"]
    vec_g = matrices["vec_g"]

    # Get parameters
    ets_model = model_params["ets_model"]
    model_is_trendy = model_params["model_is_trendy"]
    model_is_seasonal = model_params["model_is_seasonal"]
    components_number_arima = model_params["components_number_arima"]

    j = 0
    # ETS model, persistence
    if ets_model:
        j += 1
        if not persistence_checked["persistence_level_estimate"]:
            vec_g[j - 1, 0] = persistence_checked["persistence_level"]

        if model_is_trendy:
            j += 1
            if not persistence_checked["persistence_trend_estimate"]:
                vec_g[j - 1, 0] = persistence_checked["persistence_trend"]

        if model_is_seasonal:
            if not all(persistence_checked["persistence_seasonal_estimate"]):
                # Get indices where persistence was provided (not estimated)
                provided_indices = np.where(
                    np.logical_not(persistence_checked["persistence_seasonal_estimate"])
                )[0]
                # Get only the provided values at those indices
                provided_values = [
                    persistence_checked["persistence_seasonal"][i]
                    for i in provided_indices
                ]
                vec_g[j + provided_indices, 0] = provided_values
            j += model_params["components_number_ets_seasonal"]

    # ARIMA model, persistence
    if arima_checked["arima_model"]:
        # Remove diagonal from the ARIMA part of the matrix
        mat_f[j : j + components_number_arima, j : j + components_number_arima] = 0
        j += components_number_arima

    # Modify transition to handle drift
    if not arima_checked["arima_model"] and constants_checked["constant_required"]:
        mat_f[0, -1] = 1

    # Regression, persistence
    if explanatory_checked["xreg_model"]:
        if (
            persistence_checked["persistence_xreg_provided"]
            and not persistence_checked["persistence_xreg_estimate"]
        ):
            vec_g[j : j + explanatory_checked["xreg_number"], 0] = persistence_checked[
                "persistence_xreg"
            ]

    matrices["mat_f"] = mat_f
    matrices["vec_g"] = vec_g

    return matrices


def _handle_polynomial_setup(matrices, model_params, arima_checked):
    """
    Handle ARIMA polynomial setup if needed.

    Args:
        matrices: Dictionary containing model matrices
        model_params: Dictionary containing model parameters
        arima_checked: Dictionary of ARIMA parameters

    Returns:
        Dict: Updated matrices dictionary
    """
    # Get matrices
    mat_f = matrices["mat_f"]
    vec_g = matrices["vec_g"]

    matrices["mat_f"] = mat_f
    matrices["vec_g"] = vec_g
    matrices["arima_polynomials"] = None

    return matrices
