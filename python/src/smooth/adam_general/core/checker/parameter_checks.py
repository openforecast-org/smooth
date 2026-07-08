import numpy as np

from ._utils import _warn


def _check_distribution_loss(distribution, loss, silent=False):
    """
    Check distribution and loss function compatibility.

    Parameters
    ----------
    distribution : str
        Probability distribution
    loss : str or callable
        Loss function name or custom callable.
        If callable, it should accept (actual, fitted, B) arguments.
    silent : bool, optional
        Whether to suppress warnings

    Returns
    -------
    dict
        Dictionary with validated distribution, loss, and optionally loss_function
    """
    # Valid distribution types
    valid_distributions = [
        "default",
        "dnorm",
        "dlaplace",
        "ds",
        "dlogitnorm",
        "dlogis",
        "dt",
        "dgnorm",
        "dlnorm",
        "dinvgauss",
        "dgamma",
        "dpois",
        "plogis",
    ]

    # Valid loss functions
    valid_losses = [
        "likelihood",
        "GPL",
        "MSE",
        "MAE",
        "HAM",
        "MSEh",
        "MAEh",
        "HAMh",
        "MSCE",
        "MACE",
        "CHAM",
        "TMSE",
        "TMAE",
        "THAM",
        "GTMSE",
        "GTAME",
        "GTHAM",
        "LASSO",
        "RIDGE",
    ]

    # Check distribution. ``None`` is the ADAM constructor's default and
    # means "auto-select", which maps to ``"default"`` internally — no
    # warning needed for that. Warn only on actually-invalid strings.
    if distribution is None:
        distribution = "default"
    elif distribution not in valid_distributions:
        _warn(f"Unknown distribution: {distribution}. Switching to 'default'.", silent)
        distribution = "default"

    # Check loss function - handle callable custom loss
    loss_function = None
    if callable(loss):
        loss_function = loss
        loss = "custom"
    elif loss not in valid_losses:
        _warn(f"Unknown loss function: {loss}. Switching to 'likelihood'.", silent)
        loss = "likelihood"

    result = {"distribution": distribution, "loss": loss}
    if loss_function is not None:
        result["loss_function"] = loss_function
    return result


def _check_outliers(outliers_mode, silent=False):
    """
    Check outliers handling mode.

    Parameters
    ----------
    outliers_mode : str
        Outliers handling mode
    silent : bool, optional
        Whether to suppress warnings

    Returns
    -------
    str
        Validated outliers mode
    """
    valid_modes = ["ignore", "replace"]

    if outliers_mode not in valid_modes:
        _warn(f"Unknown outliers mode: {outliers_mode}. Switching to 'ignore'.", silent)
        return "ignore"

    return outliers_mode


def _check_phi(phi, damped, silent=False):
    """
    Check damping parameter phi.

    Parameters
    ----------
    phi : float or None
        Damping parameter value
    damped : bool
        Whether trend is damped
    silent : bool, optional
        Whether to suppress warnings

    Returns
    -------
    dict
        Dictionary with phi information
    """
    # If damped, but phi not provided, use default
    if damped and phi is None:
        phi = 0.95
        phi_estimate = True
    # If damped and phi provided as numeric
    elif damped and isinstance(phi, (int, float)):
        if phi >= 1 or phi <= 0:
            _warn(
                f"Invalid phi value: {phi}. Must be in (0, 1). Setting to 0.95.", silent
            )
            phi = 0.95
        phi_estimate = False
    # If damped and phi provided as boolean/string "estimate"
    elif damped and (
        phi is True or (isinstance(phi, str) and phi.lower() == "estimate")
    ):
        phi = 0.95
        phi_estimate = True
    # If not damped, phi is fixed at 1
    elif not damped:
        phi = 1.0
        phi_estimate = False
    # Default fallback
    else:
        _warn(f"Invalid phi specification: {phi}. Using default value.", silent)
        phi = 0.95 if damped else 1.0
        phi_estimate = damped

    return {"phi": phi, "phi_estimate": phi_estimate}


def _check_persistence(
    persistence,
    ets_model,
    trend_type,
    season_type,
    lags_model_seasonal,
    xreg_model=False,
    silent=False,
):
    """
    Check persistence parameters.

    Parameters
    ----------
    persistence : float, list, dict, or None
        Persistence parameter specification
    ets_model : bool
        Whether ETS model is used
    trend_type : str
        Trend component type
    season_type : str
        Seasonal component type
    lags_model_seasonal : list
        List of seasonal lags
    xreg_model : bool, optional
        Whether external regressors are used
    silent : bool, optional
        Whether to suppress warnings

    Returns
    -------
    dict
        Dictionary with persistence parameters
    """
    # Initialize defaults
    n_seasonal = len(lags_model_seasonal) if lags_model_seasonal else 0
    result = {
        "persistence": None,
        "persistence_estimate": True,
        "persistence_level": None,
        "persistence_level_estimate": True,
        "persistence_trend": None,
        "persistence_trend_estimate": True,
        "persistence_seasonal": [None] * n_seasonal,
        "persistence_seasonal_estimate": [True] * n_seasonal,
        "persistence_xreg": None,
        "persistence_xreg_estimate": True,
        "persistence_xreg_provided": False,
    }

    # Handle None case
    if persistence is None:
        return result

    # Handle single numeric value
    if isinstance(persistence, (int, float)):
        # Apply the same value to all components
        result["persistence"] = persistence
        result["persistence_level"] = persistence
        result["persistence_trend"] = persistence
        result["persistence_seasonal"] = (
            [persistence] * n_seasonal if n_seasonal > 0 else None
        )

        # Mark all as not estimated
        result["persistence_estimate"] = False
        result["persistence_level_estimate"] = False
        result["persistence_trend_estimate"] = False
        result["persistence_seasonal_estimate"] = (
            [False] * n_seasonal if n_seasonal > 0 else []
        )

        return result

    # Handle list/tuple of values
    if isinstance(persistence, (list, tuple)):
        # Check if the length is appropriate for the model
        expected_length = (
            1 + (trend_type != "N") + (len(lags_model_seasonal) > 0) + xreg_model
        )

        if len(persistence) > expected_length:
            _warn(
                f"Too many persistence values provided ({len(persistence)}). "
                f"Expected at most {expected_length}.",
                silent,
            )

        # Assign values based on position
        pos = 0
        if ets_model:
            if pos < len(persistence):
                result["persistence_level"] = persistence[pos]
                result["persistence_level_estimate"] = False
                pos += 1

            if trend_type != "N" and pos < len(persistence):
                result["persistence_trend"] = persistence[pos]
                result["persistence_trend_estimate"] = False
                pos += 1

            if len(lags_model_seasonal) > 0 and pos < len(persistence):
                # Single value applies to all seasonal components
                result["persistence_seasonal"] = [persistence[pos]] * n_seasonal
                result["persistence_seasonal_estimate"] = [False] * n_seasonal
                pos += 1

        if xreg_model and pos < len(persistence):
            result["persistence_xreg"] = persistence[pos]
            result["persistence_xreg_estimate"] = False
            result["persistence_xreg_provided"] = True

        # Mark overall persistence as not estimated
        result["persistence_estimate"] = False

        return result

    # Handle dictionary of named parameters
    if isinstance(persistence, dict):
        # Process level persistence
        if "level" in persistence:
            result["persistence_level"] = persistence["level"]
            result["persistence_level_estimate"] = False

        # Process trend persistence
        if "trend" in persistence and trend_type != "N":
            result["persistence_trend"] = persistence["trend"]
            result["persistence_trend_estimate"] = False

        # Process seasonal persistence
        if "seasonal" in persistence and len(lags_model_seasonal) > 0:
            seasonal_val = persistence["seasonal"]
            if isinstance(seasonal_val, (int, float)):
                # Single value applies to all seasonal components
                result["persistence_seasonal"] = [seasonal_val] * n_seasonal
                result["persistence_seasonal_estimate"] = [False] * n_seasonal
            elif isinstance(seasonal_val, (list, tuple)):
                # List of values - could be partial specification
                # Fill provided values, leave rest as None (to be estimated)
                seasonal_list = [None] * n_seasonal
                estimate_list = [True] * n_seasonal
                for i, val in enumerate(seasonal_val):
                    if i < n_seasonal:
                        seasonal_list[i] = val
                        estimate_list[i] = False
                result["persistence_seasonal"] = seasonal_list
                result["persistence_seasonal_estimate"] = estimate_list
            else:
                # Fallback - treat as single value
                result["persistence_seasonal"] = [seasonal_val] * n_seasonal
                result["persistence_seasonal_estimate"] = [False] * n_seasonal

        # Process xreg persistence
        if "xreg" in persistence and xreg_model:
            result["persistence_xreg"] = persistence["xreg"]
            result["persistence_xreg_estimate"] = False
            result["persistence_xreg_provided"] = True

        # Mark overall persistence as not estimated
        result["persistence_estimate"] = False

        return result

    # Invalid persistence type
    _warn(
        f"Invalid persistence type: {type(persistence)}. Using default values.", silent
    )
    return result


def _check_initial(
    initial,
    ets_model,
    trend_type,
    season_type,
    arima_model=False,
    xreg_model=False,
    silent=False,
):
    """
    Check initial values for state components.

    Parameters
    ----------
    initial : float, list, dict, or None
        Initial values specification
    ets_model : bool
        Whether ETS model is used
    trend_type : str
        Trend component type
    season_type : str
        Seasonal component type
    arima_model : bool, optional
        Whether ARIMA model is used
    xreg_model : bool, optional
        Whether external regressors are used
    silent : bool, optional
        Whether to suppress warnings

    Returns
    -------
    dict
        Dictionary with initial values
    """
    # Initialize defaults
    result = {
        "initial": None,
        "initial_type": "optimal",
        "initial_estimate": True,
        "initial_level": None,
        "initial_level_estimate": True,
        "initial_trend": None,
        "initial_trend_estimate": True,
        "initial_seasonal": None,
        "initial_seasonal_estimate": True,
        "initial_arima": None,
        "initial_arima_estimate": True,
        "initial_arima_number": 0,
        "initial_xreg": None,
        "initial_xreg_estimate": True,
        "initial_xreg_provided": False,
    }

    # Handle None case
    if initial is None:
        return result

    # Handle "optimal" or "backcasting" strings
    if isinstance(initial, str):
        if initial.lower() in ["optimal", "backcasting", "complete", "two-stage"]:
            result["initial_type"] = initial.lower()
            return result
        else:
            _warn(f"Unknown initial value method: {initial}. Using 'optimal'.", silent)
            return result

    # Handle numeric values
    if isinstance(initial, (int, float)):
        # Set all initial values to this value
        result["initial"] = initial
        result["initial_level"] = initial
        result["initial_trend"] = initial if trend_type != "N" else None
        result["initial_seasonal"] = initial if season_type != "N" else None
        result["initial_arima"] = initial if arima_model else None

        # Mark all as not estimated
        result["initial_estimate"] = False
        result["initial_level_estimate"] = False
        result["initial_trend_estimate"] = False
        result["initial_seasonal_estimate"] = False
        result["initial_arima_estimate"] = False

        # Set type to backcasting
        result["initial_type"] = "provided"

        return result

    # Handle list/tuple of values
    if isinstance(initial, (list, tuple)):
        result["initial_type"] = "provided"

        # Figure out the expected number of values
        expected_components = (
            1 + (trend_type != "N") + (season_type != "N") + arima_model
        )

        if len(initial) > expected_components:
            _warn(
                f"Too many initial values provided ({len(initial)}). "
                f"Expected at most {expected_components}.",
                silent,
            )

        # Assign values based on position
        pos = 0

        # Level
        if ets_model and pos < len(initial):
            result["initial_level"] = initial[pos]
            result["initial_level_estimate"] = False
            pos += 1

        # Trend
        if ets_model and trend_type != "N" and pos < len(initial):
            result["initial_trend"] = initial[pos]
            result["initial_trend_estimate"] = False
            pos += 1

        # Seasonal
        if ets_model and season_type != "N" and pos < len(initial):
            result["initial_seasonal"] = initial[pos]
            result["initial_seasonal_estimate"] = False
            pos += 1

        # ARIMA
        if arima_model and pos < len(initial):
            result["initial_arima"] = initial[pos]
            result["initial_arima_estimate"] = False

            # Set ARIMA initial number
            if isinstance(initial[pos], (list, np.ndarray)):
                result["initial_arima_number"] = len(initial[pos])
            else:
                result["initial_arima_number"] = 1

            pos += 1

        # Mark overall initial as not estimated
        result["initial_estimate"] = False

        return result

    # Handle dictionary of named parameters
    if isinstance(initial, dict):
        result["initial_type"] = "provided"

        # Process level initial
        if "level" in initial:
            result["initial_level"] = initial["level"]
            result["initial_level_estimate"] = False

        # Process trend initial
        if "trend" in initial and trend_type != "N":
            result["initial_trend"] = initial["trend"]
            result["initial_trend_estimate"] = False

        # Process seasonal initial
        if "seasonal" in initial and season_type != "N":
            result["initial_seasonal"] = initial["seasonal"]
            result["initial_seasonal_estimate"] = False

        # Process ARIMA initial
        if "arima" in initial and arima_model:
            result["initial_arima"] = initial["arima"]
            result["initial_arima_estimate"] = False

            # Set ARIMA initial number
            if isinstance(initial["arima"], (list, np.ndarray)):
                result["initial_arima_number"] = len(initial["arima"])
            else:
                result["initial_arima_number"] = 1

        # Process xreg initial
        if "xreg" in initial:
            result["initial_xreg"] = np.asarray(initial["xreg"], dtype=float).reshape(
                -1, 1
            )
            result["initial_xreg_provided"] = True

        # Mark overall initial as not estimated
        result["initial_estimate"] = False

        return result

    # Invalid initial type
    _warn(f"Invalid initial type: {type(initial)}. Using default values.", silent)
    return result


def _check_constant(constant, silent=False):
    """
    Check constant term parameter.

    Parameters
    ----------
    constant : bool, float, or None
        Constant term specification
    silent : bool, optional
        Whether to suppress warnings

    Returns
    -------
    dict
        Dictionary with constant term information
    """
    # Initialize with defaults
    result = {
        "constant_required": False,
        "constant_estimate": False,
        "constant_value": None,
        "constant_name": "constant",
    }

    # Handle None case
    if constant is None:
        return result

    # Handle boolean case
    if isinstance(constant, bool):
        result["constant_required"] = constant
        result["constant_estimate"] = constant
        return result

    # Handle numeric case
    if isinstance(constant, (int, float)):
        result["constant_required"] = True
        result["constant_estimate"] = False
        result["constant_value"] = float(constant)
        return result

    # Invalid type
    _warn(
        f"Invalid constant type: {type(constant)}. Using default value (False).", silent
    )
    return result


def _initialize_estimation_params(
    loss, lambda_param, ets_info, arima_info, silent=False
):
    """
    Initialize parameters for estimation.

    Parameters
    ----------
    loss : str
        Loss function name
    lambda_param : float
        Lambda parameter for LASSO/RIDGE
    ets_info : dict
        ETS model information
    arima_info : dict
        ARIMA model information
    silent : bool, optional
        Whether to suppress warnings

    Returns
    -------
    dict
        Dictionary with estimation parameters
    """
    # Initialize result
    result = {"lambda": lambda_param}

    # Handle specialized loss functions
    if loss in ["LASSO", "RIDGE"]:
        # Adjust lambda if needed
        if lambda_param <= 0:
            _warn(f"Lambda must be positive for {loss}. Setting it to 1.", silent)
            result["lambda"] = 1.0

        # Set lambda directly in the result
        result["lambda_"] = result["lambda"]

    # Add ARMA parameters if needed
    if arima_info["arima_model"]:
        result["arma_params"] = arima_info["arma_parameters"]

    return result
