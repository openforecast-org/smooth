import numpy as np
import pandas as pd

from smooth.adam_general.core.creator import filler
from smooth.adam_general.core.utils.utils import scaler

from ._helpers import _safe_create_index


def _fill_matrices_if_needed(
    general_dict,
    matrices_dict,
    adam_estimated,
    model_type_dict,
    components_dict,
    lags_dict,
    persistence_checked,
    initials_checked,
    arima_checked,
    explanatory_checked,
    phi_dict,
    constants_checked,
    adam_cpp=None,
):
    """
    Fill matrices with estimated parameters if needed.

    Parameters
    ----------
    general_dict : dict
        Dictionary with general model parameters
    matrices_dict : dict
        Dictionary with model matrices
    adam_estimated : dict
        Dictionary with estimated parameters
    model_type_dict : dict
        Dictionary with model type information
    components_dict : dict
        Dictionary with model components information
    lags_dict : dict
        Dictionary with lag-related information
    persistence_checked : dict
        Dictionary with persistence parameters
    initials_checked : dict
        Dictionary with initial values
    arima_checked : dict
        Dictionary with ARIMA model parameters
    explanatory_checked : dict
        Dictionary with external regressors information
    phi_dict : dict
        Dictionary with damping parameter information
    constants_checked : dict
        Dictionary with information about constants in the model

    Returns
    -------
    dict
        Updated matrices dictionary
    """
    if general_dict.get("model_do") != "use":
        matrices_dict = filler(
            adam_estimated["B"],
            model_type_dict=model_type_dict,
            components_dict=components_dict,
            lags_dict=lags_dict,
            matrices_dict=matrices_dict,
            persistence_checked=persistence_checked,
            initials_checked=initials_checked,
            arima_checked=arima_checked,
            explanatory_checked=explanatory_checked,
            phi_dict=phi_dict,
            constants_checked=constants_checked,
            adam_cpp=adam_cpp,
        )

    return matrices_dict


def _prepare_profiles_recent_table(matrices_dict, lags_dict):
    """
    Prepare the recent profiles table for forecasting.

    Parameters
    ----------
    matrices_dict : dict
        Dictionary with model matrices
    lags_dict : dict
        Dictionary with lag-related information

    Returns
    -------
    numpy.ndarray
        Recent profiles table
    numpy.ndarray
        Initial recent profiles table
    """
    profiles_recent_table = matrices_dict["mat_vt"][:, : lags_dict["lags_model_max"]]
    profiles_recent_initial = matrices_dict["mat_vt"][
        :, : lags_dict["lags_model_max"]
    ].copy()

    return profiles_recent_table, profiles_recent_initial


def _prepare_fitter_inputs(
    observations_dict, matrices_dict, lags_dict, profiles_dict, profiles_recent_table
):
    """
    Prepare inputs for the adam_fitter function.

    Parameters
    ----------
    observations_dict : dict
        Dictionary with observation data and related information
    matrices_dict : dict
        Dictionary with model matrices
    lags_dict : dict
        Dictionary with lag-related information
    profiles_dict : dict
        Dictionary with profile information
    profiles_recent_table : numpy.ndarray
        Recent profiles table

    Returns
    -------
    tuple
        Tuple containing arrays prepared for adam_fitter
    """
    # Convert pandas Series/DataFrames to numpy arrays
    y_in_sample = np.asarray(
        observations_dict["y_in_sample"].flatten(), dtype=np.float64
    )
    ot = np.asarray(observations_dict["ot"].flatten(), dtype=np.float64)
    mat_vt = np.asfortranarray(matrices_dict["mat_vt"], dtype=np.float64)
    mat_wt = np.asfortranarray(matrices_dict["mat_wt"], dtype=np.float64)
    mat_f = np.asfortranarray(matrices_dict["mat_f"], dtype=np.float64)
    vec_g = np.asfortranarray(matrices_dict["vec_g"], dtype=np.float64)
    lags_model_all = np.asfortranarray(
        lags_dict["lags_model_all"], dtype=np.uint64
    ).reshape(-1, 1)
    index_lookup_table = np.asfortranarray(
        profiles_dict["index_lookup_table"], dtype=np.uint64
    )
    profiles_recent_table = np.asfortranarray(profiles_recent_table, dtype=np.float64)

    return (
        y_in_sample,
        ot,
        mat_vt,
        mat_wt,
        mat_f,
        vec_g,
        lags_model_all,
        index_lookup_table,
        profiles_recent_table,
    )


def _correct_multiplicative_components(
    matrices_dict, profiles_dict, model_type_dict, components_dict
):
    """
    Correct negative or NaN values in multiplicative components.

    Parameters
    ----------
    matrices_dict : dict
        Dictionary with model matrices
    profiles_dict : dict
        Dictionary with profile information
    model_type_dict : dict
        Dictionary with model type information
    components_dict : dict
        Dictionary with model components information

    Returns
    -------
    dict
        Updated matrices dictionary
    dict
        Updated profiles dictionary
    """

    # Kind of complex here. Sorry people.
    # Thanks for the nice heuristics Ivan!

    if model_type_dict["trend_type"] == "M" and (
        np.any(np.isnan(matrices_dict["mat_vt"][1, :]))
        or np.any(matrices_dict["mat_vt"][1, :] <= 0)
    ):
        i = np.where(matrices_dict["mat_vt"][1, :] <= 0)[0]
        matrices_dict["mat_vt"][1, i] = 1e-6
        profiles_dict["profiles_recent_table"][1, i] = 1e-6
    if (
        model_type_dict["season_type"] == "M"
        and np.all(
            ~np.isnan(
                matrices_dict["mat_vt"][
                    components_dict[
                        "components_number_ets_non_seasonal"
                    ] : components_dict["components_number_ets_non_seasonal"]
                    + components_dict["components_number_ets_seasonal"],
                    :,
                ]
            )
        )
        and np.any(
            matrices_dict["mat_vt"][
                components_dict["components_number_ets_non_seasonal"] : components_dict[
                    "components_number_ets_non_seasonal"
                ]
                + components_dict["components_number_ets_seasonal"],
                :,
            ]
            <= 0
        )
    ):
        i = np.where(
            matrices_dict["mat_vt"][
                components_dict["components_number_ets_non_seasonal"] : components_dict[
                    "components_number_ets_non_seasonal"
                ]
                + components_dict["components_number_ets_seasonal"],
                :,
            ]
            <= 0
        )[0]
        matrices_dict["mat_vt"][
            components_dict["components_number_ets_non_seasonal"] : components_dict[
                "components_number_ets_non_seasonal"
            ]
            + components_dict["components_number_ets_seasonal"],
            i,
        ] = 1e-6
        i = np.where(
            profiles_dict["profiles_recent_table"][
                components_dict["components_number_ets_non_seasonal"] : components_dict[
                    "components_number_ets_non_seasonal"
                ]
                + components_dict["components_number_ets_seasonal"],
                :,
            ]
            <= 0
        )[0]
        profiles_dict["profiles_recent_table"][
            components_dict["components_number_ets_non_seasonal"] : components_dict[
                "components_number_ets_non_seasonal"
            ]
            + components_dict["components_number_ets_seasonal"],
            i,
        ] = 1e-6

    return matrices_dict, profiles_dict


def _initialize_fitted_series(observations_dict):
    """
    Initialize pandas Series for fitted values and errors.

    Parameters
    ----------
    observations_dict : dict
        Dictionary with observation data and related information

    Returns
    -------
    tuple
        Tuple of (y_fitted, errors) pandas Series
    """
    if not isinstance(observations_dict["y_in_sample"], pd.Series):
        # Use safe index creation for non-Series input
        index = _safe_create_index(
            start=observations_dict["y_start"],
            periods=observations_dict["obs_in_sample"],
            freq=observations_dict["frequency"],
        )
        y_fitted = pd.Series(
            np.full(observations_dict["obs_in_sample"], np.nan), index=index
        )
        errors = pd.Series(
            np.full(observations_dict["obs_in_sample"], np.nan), index=index
        )
    else:
        y_fitted = pd.Series(
            np.full(observations_dict["obs_in_sample"], np.nan),
            index=observations_dict["y_in_sample_index"],
        )
        errors = pd.Series(
            np.full(observations_dict["obs_in_sample"], np.nan),
            index=observations_dict["y_in_sample_index"],
        )

    return y_fitted, errors


def _update_distribution(general_dict, model_type_dict):
    """
    Update distribution based on error type and loss function.

    Parameters
    ----------
    general_dict : dict
        Dictionary with general model parameters
    model_type_dict : dict
        Dictionary with model type information

    Returns
    -------
    dict
        Updated general dictionary
    """
    if general_dict["distribution"] == "default":
        if general_dict["loss"] == "likelihood":
            if model_type_dict["error_type"] == "A":
                general_dict["distribution"] = "dnorm"
            elif model_type_dict["error_type"] == "M":
                general_dict["distribution"] = "dgamma"
        elif general_dict["loss"] in ["MAEh", "MACE", "MAE"]:
            general_dict["distribution"] = "dlaplace"
        elif general_dict["loss"] in ["HAMh", "CHAM", "HAM"]:
            general_dict["distribution"] = "ds"
        elif general_dict["loss"] in ["MSEh", "MSCE", "MSE", "GPL"]:
            general_dict["distribution"] = "dnorm"
        else:
            general_dict["distribution"] = "dnorm"

    return general_dict


def _process_initial_values(
    model_type_dict,
    lags_dict,
    matrices_dict,
    components_dict,
    arima_checked,
    explanatory_checked,
    initials_checked,
):
    """
    Process and organize initial values.

    Parameters
    ----------
    model_type_dict : dict
        Dictionary with model type information
    lags_dict : dict
        Dictionary with lag-related information
    matrices_dict : dict
        Dictionary with model matrices
    components_dict : dict
        Dictionary with model components information
    arima_checked : dict
        Dictionary with ARIMA model parameters
    explanatory_checked : dict
        Dictionary with external regressors information
    initials_checked : dict
        Dictionary with initial values

    Returns
    -------
    tuple
        Tuple of (initial_value, initial_value_names, initial_estimated)
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
    j = 0
    if model_type_dict["ets_model"]:
        # Write down level, trend and seasonal
        for i in range(len(lags_dict["lags_model"])):
            # In case of level / trend, we want to get the very first value
            if lags_dict["lags_model"][i] == 1:
                initial_value_ets[i] = matrices_dict["mat_vt"][
                    i, : lags_dict["lags_model_max"]
                ][0]
            # In cases of seasonal components, they should be at the end of
            # the pre-heat period
            else:
                # print(lags_dict["lags_model"][i][0])
                # here we might have an issue for taking the first element
                start_idx = lags_dict["lags_model_max"] - lags_dict["lags_model"][i]
                initial_value_ets[i] = matrices_dict["mat_vt"][
                    i, start_idx : lags_dict["lags_model_max"]
                ]

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
            # Convert initial_seasonal_estimate to list if it's a boolean
            # (for single seasonality)
            if isinstance(initials_checked["initial_seasonal_estimate"], bool):
                seasonal_estimate_list = [
                    initials_checked["initial_seasonal_estimate"]
                ] * components_dict["components_number_ets_seasonal"]
            else:
                seasonal_estimate_list = initials_checked["initial_seasonal_estimate"]

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
                initial_value[j] = next(x for x in initial_value_ets if x is not None)
                initial_value_names[j] = "seasonal"
                initial_estimated[j] = "seasonal"

    # Write down the ARIMA initials
    if arima_checked["arima_model"]:
        if model_type_dict["ets_model"]:
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
        name: value for name, value in zip(initial_value_names, initial_value)
    }

    return initial_value, initial_value_names, initial_estimated


def _process_arma_parameters(arima_checked, adam_estimated):
    """
    Process ARMA parameters from estimates.

    Parameters
    ----------
    arima_checked : dict
        Dictionary with ARIMA model parameters
    adam_estimated : dict
        Dictionary with estimated parameters

    Returns
    -------
    dict or None
        Dictionary of AR and MA parameters or None if no ARIMA model
    """
    # TODO: B here was not defined in the function namespace. Validate if it's indeed
    # in that dictionary
    if arima_checked["arima_model"]:
        arma_parameters_list = {}
        arima_polys = adam_estimated.get("arima_polynomials", {})
        if arima_checked["ar_required"]:
            ar_poly = arima_polys.get("ar_polynomial", arima_polys.get("arPolynomial"))
            if ar_poly is not None:
                arma_parameters_list["ar"] = list(ar_poly[1:])
            else:
                arma_parameters_list["ar"] = []
        if arima_checked["ma_required"]:
            ma_poly = arima_polys.get("ma_polynomial", arima_polys.get("maPolynomial"))
            if ma_poly is not None:
                arma_parameters_list["ma"] = list(ma_poly[1:])
            else:
                arma_parameters_list["ma"] = []
    else:
        arma_parameters_list = None

    return arma_parameters_list


def _calculate_scale_parameter(
    general_dict, model_type_dict, errors, y_fitted, observations_dict, other
):
    """
    Calculate scale parameter using scaler function.

    Parameters
    ----------
    general_dict : dict
        Dictionary with general model parameters
    model_type_dict : dict
        Dictionary with model type information
    errors : pandas.Series
        Series with model errors
    y_fitted : pandas.Series
        Series with fitted values
    observations_dict : dict
        Dictionary with observation data and related information
    other : dict
        Additional parameters

    Returns
    -------
    float
        Calculated scale parameter
    """
    scale = scaler(
        general_dict["distribution_new"],
        model_type_dict["error_type"],
        errors[observations_dict["ot_logical"]],
        y_fitted[observations_dict["ot_logical"]],
        observations_dict["obs_in_sample"],
        other,
    )

    return scale


def _process_other_parameters(
    constants_checked, adam_estimated, general_dict, arima_checked, lags_dict=None
):
    """
    Process additional parameters like constants and ARIMA polynomials.

    Parameters
    ----------
    constants_checked : dict
        Dictionary with information about constants
    adam_estimated : dict
        Dictionary with estimated parameters
    general_dict : dict
        Dictionary with general model parameters
    arima_checked : dict
        Dictionary with ARIMA model parameters
    lags_dict : dict, optional
        Dictionary with lag-related information

    Returns
    -------
    tuple
        Tuple of (constant_value, other_returned)
    """
    # Get constant value
    # If constant is being estimated, get it from B (last element when estimated)
    # If not estimated but required, get the fixed value from constants_checked
    if constants_checked["constant_estimate"]:
        if len(adam_estimated["B"]) > 0:
            constant_value = adam_estimated["B"][-1]
        else:
            constant_value = 0  # Default when no parameters estimated
    elif constants_checked["constant_required"]:
        constant_value = constants_checked.get("constant_value", 0)
    else:
        constant_value = 0

    # Initialize other parameters dictionary
    other_returned = {}

    # Add LASSO/RIDGE lambda if applicable
    if general_dict["loss"] in ["LASSO", "RIDGE"]:
        other_returned["lambda"] = general_dict["lambda_"]

    # Add ARIMA polynomials if applicable
    if arima_checked["arima_model"]:
        other_returned["polynomial"] = adam_estimated["arima_polynomials"]
        other_returned["ARIMA_indices"] = {
            "nonZeroARI": arima_checked["non_zero_ari"],
            "nonZeroMA": arima_checked["non_zero_ma"],
        }

        # Create AR polynomial matrix (R: arOrders %*% lags)
        if lags_dict is not None:
            lags_original = lags_dict.get("lags_original", lags_dict["lags"])
            ar_matrix_size = int(np.dot(arima_checked["ar_orders"], lags_original))
            other_returned["ar_polynomial_matrix"] = np.zeros(
                (ar_matrix_size, ar_matrix_size)
            )

            if other_returned["ar_polynomial_matrix"].shape[0] > 1:
                # Set diagonal elements to 1 except first row/col
                other_returned["ar_polynomial_matrix"][1:-1, 2:] = np.eye(
                    other_returned["ar_polynomial_matrix"].shape[0] - 2
                )

                if arima_checked["ar_required"]:
                    other_returned["ar_polynomial_matrix"][:, 0] = -adam_estimated[
                        "arima_polynomials"
                    ]["ar_polynomial"][1:]

        other_returned["arma_parameters"] = arima_checked["arma_parameters"]

    return constant_value, other_returned


def preparator(
    # Model type info
    model_type_dict,
    # Components info
    components_dict,
    # Lags info
    lags_dict,
    # Matrices from creator
    matrices_dict,
    # Parameter dictionaries
    persistence_checked,
    initials_checked,
    arima_checked,
    explanatory_checked,
    phi_dict,
    constants_checked,
    # Other parameters
    observations_dict,
    occurrence_dict,
    general_dict,
    profiles_dict,
    # The parameter vector
    adam_estimated,
    # adamCore C++ object
    adam_cpp,
    # Optional parameters
    bounds="usual",
    other=None,
):
    """
    Prepare estimated ADAM model for forecasting by computing in-sample fit and states.

    This is the **bridge function** between model estimation and forecasting. After
    parameters are optimized by ``estimator()``, ``preparator()`` fills the state-space
    matrices with the estimated parameters, runs the model forward through the in-sample
    period to generate fitted values and final states, and packages everything needed
    for ``forecaster()`` to produce out-of-sample predictions.

    **Preparation Process**:

    1. **Matrix Filling**: If parameters were estimated (not fixed), call
       ``filler()`` to populate mat_vt, mat_wt, mat_f, vec_g with values from
       optimized parameter vector B

    2. **Profile Setup**: Prepare profile matrices for time-varying parameters
       (advanced feature, typically zeros for standard models)

    3. **Array Preparation**: Convert all inputs to proper numpy arrays with
       correct shapes and data types (Fortran-order for C++ compatibility)

    4. **Model Fitting**: Call C++ ``adam_fitter()`` to run the model forward through
       in-sample data, updating states and computing fitted values:

       .. math::

           y_t^{\\text{fitted}} = w_t' v_{t-l}

           v_t = F v_{t-l} + g \\epsilon_t

       where :math:`\\epsilon_t = (y_t - y_t^{\\text{fitted}}) / r_t`

    5. **Results Packaging**: Extract and organize:

       - Final state vector (for starting forecasts)
       - Fitted values
       - Residuals
       - Scale parameter
       - All estimated parameters (α, β, γ, φ, initials, AR/MA, etc.)

    **Outputs Used by Forecaster**:

    The prepared model dict contains everything ``forecaster()`` needs:

    - **States**: Final values to initialize forecasting
    - **Matrices**: mat_wt, mat_f, vec_g for state-space recursion
    - **Parameters**: For interval calculation (scale, smoothing params)
    - **Fitted values**: For diagnostics and residual analysis

    Parameters
    ----------
    model_type_dict : dict
        Model type specification containing:

        - 'ets_model': Whether ETS components exist
        - 'arima_model': Whether ARIMA components exist
        - 'error_type': 'A' or 'M'
        - 'trend_type': 'N', 'A', 'Ad', 'M', 'Md'
        - 'season_type': 'N', 'A', 'M'
        - 'model_is_trendy': Trend presence flag
        - 'model_is_seasonal': Seasonality presence flag

    components_dict : dict
        Component counts containing:

        - 'components_number_all': Total state dimension
        - 'components_number_ets': ETS component count
        - 'components_number_arima': ARIMA component count

    lags_dict : dict
        Lag structure containing:

        - 'lags': Primary lag vector
        - 'lags_model': Lags for each state component
        - 'lags_model_all': Complete lag specification
        - 'lags_model_max': Maximum lag (lookback period)

    matrices_dict : dict
        State-space matrices from ``creator()`` containing:

        - 'mat_vt': State vector (may have initial parameters or backcasted values)
        - 'mat_wt': Measurement matrix (may have damping placeholders)
        - 'mat_f': Transition matrix
        - 'vec_g': Persistence vector (may have smoothing parameter placeholders)

        These matrices are updated in-place if parameters were estimated.

    persistence_checked : dict
        Persistence specification containing:

        - 'persistence_estimate': Whether smoothing parameters were estimated
        - 'persistence_level_estimate': Whether α was estimated
        - 'persistence_trend_estimate': Whether β was estimated
        - 'persistence_seasonal_estimate': List of flags for γ estimation
        - Fixed values for non-estimated parameters

    initials_checked : dict
        Initial states specification containing:

        - 'initial_type': Initialization method used
        - 'initial_level_estimate': Whether l₀ was estimated
        - 'initial_trend_estimate': Whether b₀ was estimated
        - 'initial_seasonal_estimate': List of flags for s₀ estimation

    arima_checked : dict
        ARIMA specification containing:

        - 'arima_model': Whether ARIMA components exist
        - 'ar_estimate': Whether AR coefficients were estimated
        - 'ma_estimate': Whether MA coefficients were estimated
        - 'ar_orders': AR orders
        - 'ma_orders': MA orders

    explanatory_checked : dict
        External regressors specification containing:

        - 'xreg_model': Whether regressors exist
        - 'xreg_number': Number of regressors
        - 'mat_xt': Regressor data matrix

    phi_dict : dict
        Damping specification containing:

        - 'phi_estimate': Whether φ was estimated
        - 'phi': Damping value (estimated or fixed)

    constants_checked : dict
        Constant term specification containing:

        - 'constant_required': Whether constant is included
        - 'constant_estimate': Whether constant was estimated

    observations_dict : dict
        Observation information containing:

        - 'y_in_sample': Time series data
        - 'obs_in_sample': Number of observations
        - 'ot': Occurrence vector (1 for non-zero observations)

    occurrence_dict : dict
        Intermittent demand specification containing:

        - 'occurrence_model': Whether occurrence model is active
        - 'p_fitted': Fitted occurrence probabilities

    general_dict : dict
        General configuration containing:

        - 'distribution': Error distribution
        - 'loss': Loss function used in estimation

    profiles_dict : dict
        Profile matrices for time-varying parameters containing:

        - 'profiles_recent_table': Recent profile values
        - 'index_lookup_table': Index mapping for profile access

    adam_estimated : dict
        Estimation results from ``estimator()`` containing:

        - **'B'**: Optimized parameter vector
        - 'CF_value': Final cost function value
        - 'n_param_estimated': Number of estimated parameters
        - 'log_lik_adam_value': Log-likelihood information
        - 'arima_polynomials': AR/MA polynomials (if ARIMA)

    bounds : str, default="usual"
        Bound type used in estimation ('usual', 'admissible', 'none').
        Currently unused in preparator but kept for compatibility.

    other : float or None, default=None
        Additional distribution parameter (for certain distributions).
        Currently unused in preparator.

    Returns
    -------
    dict
        Prepared model dictionary containing:

        - **'states'** (numpy.ndarray): State vector matrix, shape
          (n_components, T+max_lag). Final columns (at T) are starting point
          for forecasting.

        - **'y_fitted'** (numpy.ndarray): In-sample fitted values, shape (T,)

        - **'residuals'** (numpy.ndarray): In-sample residuals, shape (T,)

        - **'mat_wt'** (numpy.ndarray): Measurement matrix (for forecasting)

        - **'mat_f'** (numpy.ndarray): Transition matrix (for forecasting)

        - **'vec_g'** (numpy.ndarray): Persistence vector (for forecasting)

        - **'scale'** (float): Error scale parameter (standard deviation for additive,
          scale for multiplicative)

        - **'persistence_level'** (float): Estimated α (if applicable)

        - **'persistence_trend'** (float): Estimated β (if trendy)

        - **'persistence_seasonal'** (list): Estimated γ values (if seasonal)

        - **'phi'** (float): Damping parameter (if damped trend)

        - **'initial_level'** (float): Level initial state

        - **'initial_trend'** (float): Trend initial state (if trendy)

        - **'initial_seasonal'** (list): Seasonal initial states (if seasonal)

        - **'ar_parameters'** (list): AR coefficients (if ARIMA)

        - **'ma_parameters'** (list): MA coefficients (if ARIMA)

        - **'xreg_parameters'** (list): Regression coefficients (if regressors)

        - **'constant'** (float): Constant term (if included)

        - **'arima_polynomials'** (dict): AR/MA polynomial matrices (if ARIMA)

        - **'loglik'** (float): Log-likelihood value

        - **'n_param'** (int): Number of estimated parameters

    Notes
    -----
    **Matrix Ordering**:

    All matrices use **Fortran order** (column-major) for C++ compatibility. Do not
    change to C-order as it will cause incorrect results in adam_fitter.

    **Fitted Values vs Residuals**:

    - **Fitted values**: One-step-ahead predictions using actual past observations
    - **Residuals**: y_t - y_fitted_t (not scaled)
    - **Scaled residuals**: ε_t = residuals_t / scale

    For multiplicative models, residuals are relative errors.

    **Initial States in Output**:

    The initial states returned (initial_level, initial_trend, etc.) are:

    - Values at time t=0 (before first observation)
    - Either estimated, backcasted, or user-provided depending on initial_type
    - Extracted from first max_lag columns of mat_vt

    **When is Filler Called?**:

    ``filler()`` is called only if parameters were actually estimated. If all parameters
    were fixed (e.g., using a previously estimated model), matrices already contain
    correct values and filler is skipped.

    **ARIMA Polynomials**:

    For ARIMA models, the arima_polynomials dict contains companion matrix
    representations of AR and MA polynomials, used for state-space forecasting.

    **Performance**:

    The C++ adam_fitter is very fast (~1-5ms for T=1000 observations). The preparator
    overhead is minimal.

    See Also
    --------
    estimator : Calls preparator after optimization to get final fitted model
    forecaster : Uses prepared model to generate forecasts
    filler : Fills matrices with parameter values (called by preparator if needed)
    adam_fitter : C++ backend for computing fitted values and states

    Examples
    --------
    Prepare model after estimation::

        >>> prepared = preparator(
        ...     model_type_dict={'ets_model': True, 'arima_model': False, ...},
        ...     components_dict={'components_number_all': 13, ...},
        ...     lags_dict={'lags': np.array([1, 12]), ...},
        ...     matrices_dict={'mat_vt': mat_vt, 'mat_wt': mat_wt, ...},
        ...     persistence_checked={'persistence_estimate': True, ...},
        ...     initials_checked={'initial_type': 'optimal', ...},
        ...     observations_dict={'y_in_sample': data, 'obs_in_sample': 100, ...},
        ...     adam_estimated={'B': optimized_params, 'log_lik_adam_value': {...}, ...}
        ...     ...
        ... )
        >>> print(prepared['y_fitted'])  # In-sample fitted values
        >>> print(prepared['states'][:, -1])  # Final state vector for forecasting
        >>> print(prepared['scale'])  # Error scale for prediction intervals

    Extract estimated parameters::

        >>> alpha = prepared['persistence_level']
        >>> beta = prepared['persistence_trend']
        >>> l0 = prepared['initial_level']
        >>> print(f"Smoothing: α={alpha:.3f}, β={beta:.3f}, Initial level: {l0:.2f}")

    Use prepared model for forecasting::

        >>> forecasts = forecaster(
        ...     model_prepared=prepared,  # Pass prepared model
        ...     observations_dict=obs_dict,
        ...     general_dict={'h': 12, ...},
        ...     ...
        ... )
    """
    # 1. Fill matrices with estimated parameters if needed
    matrices_dict = _fill_matrices_if_needed(
        general_dict,
        matrices_dict,
        adam_estimated,
        model_type_dict,
        components_dict,
        lags_dict,
        persistence_checked,
        initials_checked,
        arima_checked,
        explanatory_checked,
        phi_dict,
        constants_checked,
        adam_cpp=adam_cpp,
    )

    # 2. Prepare profiles recent table
    profiles_recent_table, profiles_recent_initial = _prepare_profiles_recent_table(
        matrices_dict, lags_dict
    )

    # 3. Prepare inputs for adam_fitter
    (
        y_in_sample,
        ot,
        mat_vt,
        mat_wt,
        mat_f,
        vec_g,
        lags_model_all,
        index_lookup_table,
        profiles_recent_table_fortran,
    ) = _prepare_fitter_inputs(
        observations_dict,
        matrices_dict,
        lags_dict,
        profiles_dict,
        profiles_recent_table,
    )

    # 4. Run adam_fitter to get fitted values and states.
    # Check if initial_type is a list or string and compute backcast correctly
    if isinstance(initials_checked["initial_type"], list):
        backcast_value_prep = any(
            [
                t == "complete" or t == "backcasting"
                for t in initials_checked["initial_type"]
            ]
        )
    else:
        backcast_value_prep = initials_checked["initial_type"] in [
            "complete",
            "backcasting",
        ]

    # Call adam_cpp.fit() with the prepared inputs
    # Note: E, T, S, nNonSeasonal, nSeasonal, nArima, nXreg, constant are set
    # during adamCore construction
    adam_fitted = adam_cpp.fit(
        matrixVt=mat_vt,
        matrixWt=mat_wt,
        matrixF=mat_f,
        vectorG=vec_g,
        indexLookupTable=index_lookup_table,
        profilesRecent=profiles_recent_table_fortran,
        vectorYt=y_in_sample,
        vectorOt=ot,
        backcast=backcast_value_prep,
        nIterations=initials_checked["n_iterations"],
        O="n",
    )
    # 5. Correct negative or NaN values in multiplicative components
    matrices_dict, profiles_dict = _correct_multiplicative_components(
        matrices_dict, profiles_dict, model_type_dict, components_dict
    )
    # 6. Initialize fitted values and errors series
    y_fitted, errors = _initialize_fitted_series(observations_dict)
    # 7. Fill in fitted values and errors from adam_cpp.fit() results
    errors[:] = adam_fitted.errors.flatten()
    y_fitted[:] = adam_fitted.fitted.flatten()
    # 8. Update distribution based on error type and loss function
    general_dict = _update_distribution(general_dict, model_type_dict)

    # 9. Process initial values for all components
    initial_value, initial_value_names, initial_estimated = _process_initial_values(
        model_type_dict,
        lags_dict,
        matrices_dict,
        components_dict,
        arima_checked,
        explanatory_checked,
        initials_checked,
    )

    # 10. Handle external regressors
    if (
        explanatory_checked["xreg_model"]
        and explanatory_checked.get("regressors") != "adapt"
    ):
        explanatory_checked["regressors"] = "use"
    elif not explanatory_checked["xreg_model"]:
        explanatory_checked["regressors"] = None

    # 11. Process ARMA parameters
    arma_parameters_list = _process_arma_parameters(arima_checked, adam_estimated)

    # 12. Calculate scale parameter
    scale = _calculate_scale_parameter(
        general_dict, model_type_dict, errors, y_fitted, observations_dict, other
    )

    # 13. Process constant and other parameters
    constant_value, other_returned = _process_other_parameters(
        constants_checked, adam_estimated, general_dict, arima_checked, lags_dict
    )

    # 14. Update parameters number
    # Ensure parameters_number has enough elements before updating
    while len(general_dict["parameters_number"][0]) < 3:
        general_dict["parameters_number"][0].append(0)
    while len(general_dict["parameters_number"][1]) < 3:
        general_dict["parameters_number"][1].append(0)
    general_dict["parameters_number"][0][2] = np.sum(
        general_dict["parameters_number"][0][:2]
    )

    # 15. Return the prepared model
    return {
        "model": model_type_dict["model"],
        "time_elapsed": None,  # Time calculation could be added if needed
        "holdout": general_dict["holdout"],
        "y_fitted": y_fitted,
        "residuals": errors,
        "states": adam_fitted.states,
        "profiles_recent_table": adam_fitted.profile,
        "persistence": matrices_dict["vec_g"],
        "transition": matrices_dict["mat_f"],
        "measurement": matrices_dict["mat_wt"],
        "mat_vt": matrices_dict["mat_vt"],
        "mat_f": matrices_dict["mat_f"],
        "mat_wt": matrices_dict["mat_wt"],
        "phi": phi_dict["phi"],
        "initial": initial_value,
        "initial_type": initials_checked["initial_type"],
        "initial_estimated": initial_estimated,
        "orders": general_dict.get("orders"),
        "arma": arma_parameters_list,
        "constant": constant_value,
        "n_param": general_dict["parameters_number"],
        "occurrence": occurrence_dict["oes_model"],
        "formula": explanatory_checked.get("formula"),
        "regressors": explanatory_checked.get("regressors"),
        "loss": general_dict["loss"],
        "loss_value": adam_estimated["CF_value"],
        "log_lik": adam_estimated["log_lik_adam_value"],
        "distribution": general_dict["distribution"],
        "scale": scale,
        "other": other_returned,
        "B": adam_estimated["B"],
        "lags": lags_dict["lags"],
        "lags_all": lags_dict["lags_model_all"],
        "FI": general_dict.get("fi"),
    }
