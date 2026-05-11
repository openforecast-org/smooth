from typing import Any, Dict, List, Union

import numpy as np

from smooth.adam_general._adamCore import adamCore


def architector(
    # Model type info
    model_type_dict: Dict[str, Any],
    # Lags info
    lags_dict: Dict[str, Any],
    # Observation info
    observations_dict: Dict[str, Any],
    # Optional model components
    arima_checked: Dict[str, Any] = None,
    explanatory_checked: Dict[str, Any] = None,
    constants_checked: Dict[str, Any] = None,
    # Profiles
    profiles_recent_table: Union[np.ndarray, None] = None,
    profiles_recent_provided: bool = False,
    adam_ets: bool = False,
) -> Dict[str, Any]:
    """
    Determine and set up ADAM model architecture before matrix creation.

    This function is the **first step** in the model estimation pipeline. It analyzes
    the model specification and data to determine the complete model structure,
    including:

    - Component counts (how many states for level, trend, seasonality, ARIMA,
    regressors)
    - Lag structure (which lags to use for each component)
    - Profile setup (time-varying parameter structures)
    - Observation indexing (including pre-sample period)

    The architector prepares all structural information needed by ``creator()`` to build
    the state-space matrices.

    **Architecture Setup Process**:

    1. **Normalize Flags**: Ensure model_is_trendy and model_is_seasonal match
    trend_type and season_type
    2. **Component Counting**: Calculate number of states for each component type
    3. **Lag Assignment**: Assign appropriate lag to each state component
    4. **Profile Creation**: Set up lookup tables for time-varying parameters (if used)
    5. **Observation Indexing**: Compute total state sequence length (obs + pre-sample)

    Parameters
    ----------
    model_type_dict : dict
        Model specification containing:

        - 'ets_model': ETS presence flag
        - 'arima_model': ARIMA presence flag
        - 'xreg_model': Regressors presence flag
        - 'error_type': 'A' or 'M'
        - 'trend_type': 'N', 'A', 'Ad', 'M', 'Md'
        - 'season_type': 'N', 'A', 'M'

        **Modified in-place** to add:

        - 'model_is_trendy': Boolean (derived from trend_type)
        - 'model_is_seasonal': Boolean (derived from season_type)

    lags_dict : dict
        Lag information containing at minimum:

        - 'lags': Primary lag vector (e.g., [1, 12])

        **Modified in-place** to add:

        - 'lags_model': Lag for each state component
        - 'lags_model_all': Column vector of all lags
        - 'lags_model_max': Maximum lag value
        - 'lags_model_seasonal': Lags for seasonal components only

    observations_dict : dict
        Observation information containing:

        - 'obs_in_sample': Number of in-sample observations
        - 'obs_all': Total observations (including holdout if applicable)

        **Modified in-place** to add:

        - 'obs_states': Total state sequence length = obs_in_sample + lags_model_max

    arima_checked : dict, optional
        ARIMA specification containing:

        - 'arima_model': ARIMA presence flag
        - 'ar_orders': AR orders per lag
        - 'ma_orders': MA orders per lag
        - 'i_orders': Integration orders

    explanatory_checked : dict, optional
        External regressors specification containing:

        - 'xreg_model': Regressor presence flag
        - 'xreg_number': Number of regressors

    constants_checked : dict, optional
        Constant term specification containing:

        - 'constant_required': Constant presence flag

    profiles_recent_table : numpy.ndarray or None, default=None
        User-provided recent profile values for time-varying parameters.
        Shape: (n_components, max_lag)

    profiles_recent_provided : bool, default=False
        Whether profiles_recent_table was provided by user (True) or should be
        initialized (False)

    Returns
    -------
    tuple of 5 dict + adamCore
        Updated and created dictionaries, plus C++ adamCore object:

        1. **model_type_dict**: Updated with model_is_trendy and model_is_seasonal flags

        2. **components_dict**: New dictionary containing component counts:

           - 'components_number_all': Total state dimension
           - 'components_number_ets': Total ETS components
           - 'components_number_ets_non_seasonal': Level + trend count
           - 'components_number_ets_seasonal': Seasonal component count
           - 'components_number_arima': ARIMA state count
           - Additional component breakdown

        3. **lags_dict**: Updated with complete lag structure:

           - 'lags_model': List of lags for each component
           - 'lags_model_all': Column vector of all lags
           - 'lags_model_max': Maximum lag
           - 'lags_model_seasonal': Seasonal lags only

        4. **observations_dict**: Updated with obs_states

        5. **profiles_dict**: New dictionary for time-varying parameters:

           - 'profiles_recent_table': Matrix for recent values
           - 'profiles_recent_provided': Whether user-provided
           - 'index_lookup_table': Index mapping for profile access

        6. **adam_cpp**: C++ adamCore object with fit, forecast, simulate methods

    Notes
    -----
    **Component Counting Logic**:

    - **Level**: Always 1 if ETS
    - **Trend**: 1 if trendy, 0 otherwise
    - **Seasonal**: sum((lag_i - 1) for each seasonal lag)
    - **ARIMA**: sum(max(ar_order, ma_order) for each lag)
    - **Regressors**: xreg_number
    - **Constant**: 1 if required, 0 otherwise

    **Lag Assignment**:

    Each state component is assigned a lag that determines when it affects observations:

    - Level: lag 1
    - Trend: lag 1
    - Seasonal for lag m: lag m
    - ARIMA: lag 1 (for non-seasonal) and lag m (for seasonal)

    **Pre-sample Period**:

    The state vector includes max_lag initial values before the first observation.
    This allows the model to have valid lagged states from time t=1 onward.

    **Profile Tables**:

    Profiles enable time-varying parameters (advanced feature). Most users won't provide
    profiles, so they're initialized as zeros.

    See Also
    --------
    creator : Uses architector output to create matrices
    estimator : Calls architector as first step
    parameters_checker : Validates inputs before architector

    Examples
    --------
    Set up architecture for Holt-Winters model::

        >>> model_type, components, lags, obs, profiles, adam_cpp = architector(
        ...     model_type_dict={'ets_model': True, 'error_type': 'A',
        ...                      'trend_type': 'A', 'season_type': 'A'},
        ...     lags_dict={'lags': np.array([1, 12])},
        ...     observations_dict={'obs_in_sample': 100, 'obs_all': 100},
        ...     arima_checked={'arima_model': False},
        ...     explanatory_checked={'xreg_model': False},
        ...     constants_checked={'constant_required': False}
        ... )
        >>> print(components['components_number_ets'])  # 1 + 1 + 11 = 13
        >>> print(lags['lags_model_max'])  # 12
        >>> print(obs['obs_states'])  # 100 + 12 = 112

    Set up architecture for ARIMA(1,1,1)::

        >>> model_type, components, lags, obs, profiles, adam_cpp = architector(
        ...     model_type_dict={'ets_model': False, 'arima_model': True},
        ...     lags_dict={'lags': np.array([1])},
        ...     observations_dict={'obs_in_sample': 100, 'obs_all': 100},
        ...     arima_checked={'arima_model': True, 'ar_orders': [1],
        ...                    'ma_orders': [1], 'i_orders': [1]},
        ...     explanatory_checked={'xreg_model': False},
        ...     constants_checked={'constant_required': True}
        ... )
        >>> print(components['components_number_arima'])  # max(1, 1) = 1
        >>> print(components['components_number_all'])  # 1 ARIMA + 1 constant = 2
    """
    # Ensure model_is_trendy and model_is_seasonal flags are consistently set.
    # A trend_type of "N" means no trend.
    model_type_dict["model_is_trendy"] = model_type_dict.get("trend_type", "N") != "N"
    # A season_type of "N" means no seasonality.
    model_type_dict["model_is_seasonal"] = (
        model_type_dict.get("season_type", "N") != "N"
    )

    # Set up components for the model
    components_dict = _setup_components(model_type_dict, arima_checked, lags_dict)
    # Set up lags
    lags_dict = _setup_lags(lags_dict, model_type_dict, components_dict)

    # Add constant lag=1 entry to lags_model_all (R: lagsModelAll <- c(lagsModelAll, 1))
    if constants_checked and constants_checked.get("constant_required", False):
        lags_dict["lags_model_all"] = lags_dict["lags_model_all"] + [1]

    # Add xreg lag=1 entries to lags_model_all
    # R: lagsModelAll <- c(lagsModelAll, rep(1, xregNumber))
    xreg_number = (
        explanatory_checked.get("xreg_number", 0) if explanatory_checked else 0
    )
    if xreg_number > 0:
        lags_dict["lags_model_all"] = lags_dict["lags_model_all"] + [1] * xreg_number

    # Calculate total number of components
    components_number_all = len(lags_dict["lags_model_all"])

    # Verify it matches the alternative calculation
    # expected_total = (
    #     components_dict['components_number_ets'] +
    #     components_dict['components_number_arima'] +
    #     (explanatory_checked.get('xreg_number', 0) if explanatory_checked else 0) +
    #  (1 if (constants_checked and constants_checked.get('constant_required', False))
    # else 0)
    # )

    # Store in components_dict
    components_dict["components_number_all"] = components_number_all

    # Set up profiles
    profiles_dict = _create_profiles(
        profiles_recent_provided, profiles_recent_table, lags_dict, observations_dict
    )

    # Update obs states
    observations_dict["obs_states"] = (
        observations_dict["obs_in_sample"] + lags_dict["lags_model_max"]
    )

    # Create C++ adam class, which will then use fit, forecast etc methods
    # This matches R implementation (adam.R line 752-758)
    adam_cpp = adamCore(
        lags=np.array(lags_dict["lags_model_all"], dtype=np.uint64),
        E=model_type_dict["error_type"],
        T=model_type_dict["trend_type"],
        S=model_type_dict["season_type"],
        nNonSeasonal=components_dict["components_number_ets_non_seasonal"],
        nSeasonal=components_dict["components_number_ets_seasonal"],
        nETS=components_dict["components_number_ets"],
        nArima=components_dict.get("components_number_arima", 0),
        nXreg=explanatory_checked.get("xreg_number", 0) if explanatory_checked else 0,
        nComponents=components_dict["components_number_all"],
        constant=constants_checked.get("constant_required", False)
        if constants_checked
        else False,
        adamETS=adam_ets,
    )

    return (
        model_type_dict,
        components_dict,
        lags_dict,
        observations_dict,
        profiles_dict,
        adam_cpp,
    )


def _setup_components(model_type_dict, arima_checked, lags_dict):
    """
    Set up components for the model architecture.

    Args:
        model_type_dict: Dictionary containing model type information
        arima_checked: Dictionary of ARIMA parameters

    Returns:
        Dict: Dictionary containing component information
    """
    # Initialize components dict
    components_dict = {}

    # Determine ETS components
    if model_type_dict["ets_model"]:
        # Basic number of components: level is always present
        components_number_ets = 1

        # Add trend component if needed
        if model_type_dict["model_is_trendy"]:
            components_number_ets += 1
        # Add seasonal components if needed
        components_number_ets_seasonal = 0

        if model_type_dict["model_is_seasonal"]:
            # Count number of seasonal components based on the original lags provided.
            # A seasonal lag is any lag > 1 in the original lags list for the model.
            original_lags = lags_dict.get("lags", [])
            components_number_ets_seasonal = sum(
                1 for lag_period in original_lags if lag_period > 1
            )
            components_number_ets += components_number_ets_seasonal
        # Store in dictionary
        components_dict["components_number_ets"] = components_number_ets
        components_dict["components_number_ets_seasonal"] = (
            components_number_ets_seasonal
        )
        components_dict["components_number_ets_non_seasonal"] = (
            components_number_ets - components_number_ets_seasonal
        )
    else:
        # No ETS components
        components_dict["components_number_ets"] = 0
        components_dict["components_number_ets_seasonal"] = 0
        components_dict["components_number_ets_non_seasonal"] = 0

    # Determine ARIMA components
    if arima_checked and arima_checked["arima_model"]:
        # R line 628: componentsNumberARIMA <- length(lagsModelARIMA);
        #  Use the pre-computed value from _check_arima() - number of unique polynomial
        # lags
        components_number_arima = arima_checked.get("components_number_arima", 0)
        components_dict["components_number_arima"] = components_number_arima
        components_dict["lags_model_arima"] = arima_checked.get("lags_model_arima", [])
    else:
        components_dict["components_number_arima"] = 0
        components_dict["lags_model_arima"] = []

    return components_dict


def _setup_lags(lags_dict, model_type_dict, components_dict):
    """
    Set up lags for the model architecture.

    Args:
        lags_dict: Dictionary containing lags information
        model_type_dict: Dictionary containing model type information
        components_dict: Dictionary containing component information

    Returns:
        Dict: Updated lags dictionary
    """
    # Extract parameters
    lags = lags_dict["lags"]

    # Calculate model lags for each component
    lags_model = []
    lags_model_seasonal = []

    # ETS components
    if model_type_dict["ets_model"]:
        # Level always has lag 1
        lags_model.append(1)

        # Trend component has lag 1
        if model_type_dict["model_is_trendy"]:
            lags_model.append(1)

        # Seasonal components have lags corresponding to seasonal periods
        if model_type_dict["model_is_seasonal"]:
            for lag in lags:
                if lag > 1:
                    lags_model.append(lag)
                    lags_model_seasonal.append(lag)

    # ARIMA components - use polynomial-derived lags from arima_checks
    lags_model_arima = components_dict.get("lags_model_arima", [])

    # Combine all lags
    lags_model_all = lags_model + lags_model_arima
    # Find maximum lag
    lags_model_max = max(lags_model_all) if lags_model_all else 1

    # Update lags dictionary
    lags_dict_updated = lags_dict.copy()
    lags_dict_updated["lags_original"] = lags_dict[
        "lags"
    ]  # R keeps original `lags` intact
    lags_dict_updated["lags_model"] = lags_model
    lags_dict_updated["lags"] = lags_model
    lags_dict_updated["lags_model_arima"] = lags_model_arima
    lags_dict_updated["lags_model_all"] = lags_model_all
    lags_dict_updated["lags_model_max"] = lags_model_max
    lags_dict_updated["lags_model_seasonal"] = lags_model_seasonal
    return lags_dict_updated


def _create_profiles(
    profiles_recent_provided, profiles_recent_table, lags_dict, observations_dict
):
    """
    Create profiles for the model architecture.

    Args:
        profiles_recent_provided: Whether recent profiles are provided
        profiles_recent_table: Table of recent profiles
        lags_dict: Dictionary containing lags information
        observations_dict: Dictionary containing observation information

    Returns:
        Dict: Dictionary containing profile information
    """
    # Initialize profiles dictionary
    profiles_dict = {
        "profiles_recent_provided": profiles_recent_provided,
        "profiles_recent_table": profiles_recent_table,
    }

    # If profiles are not provided, create them
    if not profiles_recent_provided:
        # Create profile matrices
        profiles = adam_profile_creator(
            lags_model_all=lags_dict["lags_model_all"],
            lags_model_max=lags_dict["lags_model_max"],
            obs_all=observations_dict["obs_all"],
            lags=lags_dict["lags"],
            y_index=observations_dict.get("y_index", None),
            y_classes=observations_dict.get("y_classes", None),
        )

        # Store profiles in dictionary
        profiles_dict["profiles_recent_table"] = profiles["profiles_recent_table"]
        profiles_dict["index_lookup_table"] = profiles["index_lookup_table"]
    return profiles_dict


def adam_profile_creator(
    lags_model_all: List[List[int]],
    lags_model_max: int,
    obs_all: int,
    lags: Union[List[int], None] = None,
    y_index: Union[List, None] = None,
    y_classes: Union[List, None] = None,
) -> Dict[str, np.ndarray]:
    """
    Creates recent profile and the lookup table for ADAM.

    Args:
        lags_model_all: All lags used in the model for ETS + ARIMA + xreg.
        lags_model_max: The maximum lag used in the model.
        obs_all: Number of observations to create.
        lags: The original lags provided by user (optional).
        y_index: The indices needed to get the specific dates (optional).
        y_classes: The class used for the actual data (optional).

    Returns:
        A dictionary with 'recent' (profiles_recent_table) and 'lookup'
        (index_lookup_table) as keys.
    """
    # Initialize matrices
    # Flatten lags_model_all to handle it properly
    #  This is needed because in R, the lagsModelAll is a flat vector, but in Python
    # it's a list of lists

    profiles_recent_table = np.zeros((len(lags_model_all), lags_model_max))
    index_lookup_table = np.ones((len(lags_model_all), obs_all + lags_model_max))
    profile_indices = (
        np.arange(1, lags_model_max * len(lags_model_all) + 1)
        .reshape(-1, len(lags_model_all))
        .T
    )

    # Update matrices based on lagsModelAll
    # Update matrices based on lagsModelAll
    for i, lag in enumerate(lags_model_all):
        # Create the matrix with profiles based on the provided lags.
        # For every row, fill the first 'lag' elements from 1 to lag
        profiles_recent_table[i, :lag] = np.arange(1, lag + 1)

        # For the i-th row in indexLookupTable, fill with a repeated sequence starting
        # from lagsModelMax to the end of the row.
        # The repeated sequence is the i-th row of profileIndices, repeated enough times
        # to cover 'obsAll' observations.
        # '- 1' at the end adjusts these values to Python's zero-based indexing.
        # Fix the array size mismatch - ensure we're using the correct range
        index_lookup_table[i, lags_model_max : (lags_model_max + obs_all)] = (
            np.tile(
                profile_indices[i, : lags_model_all[i]],
                int(np.ceil(obs_all / lags_model_all[i])),
            )[:obs_all]
            - 1
        )

        # Fix the head: use order-preserving unique (like R's unique()),
        # and use the full obs_all slice (not obs_all - 1)
        vals = index_lookup_table[i, lags_model_max : (lags_model_max + obs_all)]
        unique_indices = np.array(list(dict.fromkeys(vals.tolist())))

        # Use [-lags_model_max:] to replicate R's tail() — take from the END
        # not the beginning (Bug 3 fix).
        index_lookup_table[i, :lags_model_max] = np.tile(
            unique_indices, lags_model_max
        )[-lags_model_max:]

    # Convert to int!
    index_lookup_table = index_lookup_table.astype(int)

    # Note: I skip handling of special cases (e.g., daylight saving time, leap years)
    profiles = {
        "profiles_recent_table": np.array(profiles_recent_table, dtype="float64"),
        "index_lookup_table": np.array(index_lookup_table, dtype="int64"),
    }
    return profiles
