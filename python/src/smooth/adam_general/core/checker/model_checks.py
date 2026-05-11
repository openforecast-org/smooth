import numpy as np

from ._utils import _warn


def _expand_component_code(comp_char, allow_multiplicative=True):
    """
    Expand a single component character into a list of valid possibilities.

    Parameters
    ----------
    comp_char : str
        Component character (E, T, S)
    allow_multiplicative : bool, optional
        Whether multiplicative models are allowed

    Returns
    -------
    list
        List of possible component values
    """
    possible = set()

    # Handle each special case:
    if comp_char == "Z":
        # Expand to A,N + M if allowed
        possible.update(["A", "N"])
        if allow_multiplicative:
            possible.add("M")
    elif comp_char == "C":
        # "C" is effectively "combine all" in R
        possible.update(["A", "N"])
        if allow_multiplicative:
            possible.add("M")
    elif comp_char == "F":
        # "full" => A, N, plus M if multiplicative
        possible.update(["A", "N"])
        if allow_multiplicative:
            possible.add("M")
    elif comp_char == "P":
        # "pure" => A plus M if multiplicative
        possible.update(["A"])
        if allow_multiplicative:
            possible.add("M")
    elif comp_char == "X":
        # R logic converts X->A
        possible.update(["A"])
    elif comp_char == "Y":
        # R logic converts Y->M if allowed, else A
        if allow_multiplicative:
            possible.update(["M"])
        else:
            possible.update(["A"])
    else:
        # If it's one of 'A','M','N' or an unknown letter, just return that
        possible.add(comp_char)

    return list(possible)


def _build_models_pool_from_components(
    error_type_char,
    trend_type_char,
    season_type_char,
    damped_original_model,
    allow_multiplicative,
    max_lag=1,
):
    """
    Build a models pool by fully enumerating expansions for E, T, S.
    This version aims to replicate the pool generation of the older
    _generate_models_pool.

    Parameters
    ----------
    error_type_char : str
        Error component character (e.g., 'Z', 'A', 'F', 'P')
        These are characters *after* initial processing in _check_model_composition
        (e.g., C->Z, M->A if not allow_multiplicative for specific components).
    trend_type_char : str
        Trend component character, similarly processed.
    season_type_char : str
        Seasonal component character, similarly processed.
    damped_original_model : bool
        Damping flag from the original model string (largely ignored here for pool
        generation,
        as damping is part of trend strings like "Ad").
    allow_multiplicative : bool
        Whether multiplicative models are generally allowed for pool expansion (e.g. for
        Z).
    max_lag : int, default=1
        Maximum lag value. If <= 1, seasonality is not allowed.

    Returns
    -------
    tuple
        (candidate_models, combined_mode_placeholder)
    """
    candidate_models = []

    # If max_lag <= 1, no seasonality is possible - set to "N" silently
    if max_lag <= 1:
        season_type_char = "N"

    # Handle full pool case ("F")
    if "F" in [error_type_char, trend_type_char, season_type_char]:
        candidate_models = [
            "ANN",
            "AAN",
            "AAdN",
            "AMN",
            "AMdN",
            "ANA",
            "AAA",
            "AAdA",
            "AMA",
            "AMdA",
            "ANM",
            "AAM",
            "AAdM",
            "AMM",
            "AMdM",
        ]
        if allow_multiplicative:
            candidate_models.extend(
                [
                    "MNN",
                    "MAN",
                    "MAdN",
                    "MMN",
                    "MMdN",
                    "MNA",
                    "MAA",
                    "MAdA",
                    "MMA",
                    "MMdA",
                    "MNM",
                    "MAM",
                    "MAdM",
                    "MMM",
                    "MMdM",
                ]
            )
        # Filter out seasonal models if max_lag <= 1
        if max_lag <= 1:
            candidate_models = [m for m in candidate_models if m[-1] == "N"]
    # Handle pure models case ("P")
    elif "P" in [error_type_char, trend_type_char, season_type_char]:
        candidate_models = ["ANN", "AAN", "AAdN", "ANA", "AAA", "AAdA"]
        if allow_multiplicative:
            candidate_models.extend(["MNN", "MMN", "MMdN", "MNM", "MMM", "MMdM"])
        # Filter out seasonal models if max_lag <= 1
        if max_lag <= 1:
            candidate_models = [m for m in candidate_models if m[-1] == "N"]
    # Handle standard selection case
    else:
        # Check for multiplicative requests on non-positive data
        if not allow_multiplicative:
            if error_type_char in ["Y", "Z"]:
                import warnings

                warnings.warn(
                    "Multiplicative error models cannot be used on non-positive data. "
                    "Switching to additive error (A).",
                    UserWarning,
                )
                error_type_char = "A"
            if trend_type_char in ["Y", "Z"]:
                import warnings

                warnings.warn(
                    "Multiplicative trend models cannot be used on non-positive data. "
                    "Switching to additive trend selection (X).",
                    UserWarning,
                )
                trend_type_char = "X"
            if season_type_char in ["Y", "Z"]:
                import warnings

                warnings.warn(
                    "Multiplicative seasonal models cannot be used on "
                    "non-positive data. Switching to additive seasonal "
                    "selection (X).",
                    UserWarning,
                )
                season_type_char = "X"

        # Determine Error Options
        if error_type_char in ["A", "M"]:
            actual_error_options = [error_type_char]
        else:  # For 'Z', 'N', 'X', 'Y' etc.
            actual_error_options = ["A", "M"] if allow_multiplicative else ["A"]

        # Determine Trend Options
        # Note: trend_type_char would already be 'A' or 'Ad' if originally 'M' or 'Md'
        #  and allow_multiplicative was false, due to pre-processing in
        # _check_model_composition.
        if trend_type_char in ["N", "A", "M", "Ad", "Md"]:
            actual_trend_options_with_damping = [trend_type_char]
        elif trend_type_char == "Z":
            actual_trend_options_with_damping = ["N", "A", "M", "Ad", "Md"]
        elif trend_type_char == "Y":
            actual_trend_options_with_damping = ["N", "M", "Md"]
        else:  # 'X' or any other
            actual_trend_options_with_damping = ["N", "A", "Ad"]

        # Determine Season Options
        # Note: season_type_char would already be 'A' if originally 'M'
        # and allow_multiplicative was false.
        if season_type_char in ["N", "A", "M"]:
            actual_season_options = [season_type_char]
        elif season_type_char == "Z":
            actual_season_options = ["N", "A", "M"]
        elif season_type_char == "Y":
            actual_season_options = ["N", "M"]
        else:  # 'X' or any other
            actual_season_options = ["N", "A"]

        for e_opt in actual_error_options:
            for t_opt in actual_trend_options_with_damping:
                # t_opt already includes 'd' if it's a damped trend (e.g., "Ad")
                for s_opt in actual_season_options:
                    candidate_models.append(f"{e_opt}{t_opt}{s_opt}")

    candidate_models = sorted(list(set(candidate_models)))
    # combined_mode is determined upstream in _check_model_composition and used there.
    # This function just needs to return the pool.
    return candidate_models, False


def _check_model_composition(
    model_str, allow_multiplicative=True, silent=False, max_lag=1
):
    """
    Parse and validate model composition string.

    Parameters
    ----------
    model_str : str
        String like "ANN", "ZZZ", etc. representing model components
    allow_multiplicative : bool, optional
        Whether multiplicative models are allowed
    silent : bool, optional
        Whether to suppress warnings
    max_lag : int, default=1
        Maximum lag value. If <= 1, seasonality is not allowed.

    Returns
    -------
    dict
        Dictionary with model components and configuration
    """
    # Initialize defaults
    error_type = trend_type = season_type = "N"
    damped = False
    model_do = "estimate"
    models_pool = None

    # Validate model string
    if not isinstance(model_str, str):
        if not silent:
            _warn(
                f"Invalid model type: {model_str}. "
                "Should be a string. Switching to 'ZZZ'."
            )
        model_str = "ZZZ"

    # Handle 4-character models (with damping)
    if len(model_str) == 4:
        error_type = model_str[0]
        trend_type = model_str[1]
        season_type = model_str[3]
        if model_str[2] != "d":
            if not silent:
                _warn(f"Invalid damped trend specification in {model_str}. Using 'd'.")
        damped = True

    # Handle 3-character models
    elif len(model_str) == 3:
        error_type = model_str[0]
        trend_type = model_str[1]
        season_type = model_str[2]
        damped = trend_type in ["Z", "X", "Y"]

    else:
        if not silent:
            _warn(f"Invalid model string length: {model_str}. Switching to 'ZZZ'.")
        model_str = "ZZZ"
        error_type = trend_type = season_type = "Z"
        damped = True

    # Validate components
    valid_error = ["Z", "X", "Y", "A", "M", "C", "N", "F", "P", "S"]
    valid_trend = ["Z", "X", "Y", "N", "A", "M", "C", "F", "P", "S"]
    valid_season = ["Z", "X", "Y", "N", "A", "M", "C", "F", "P", "S"]

    if error_type not in valid_error:
        if not silent:
            _warn(f"Invalid error type: {error_type}. Switching to 'Z'.")
        error_type = "Z"
        model_do = "select"

    if trend_type not in valid_trend:
        if not silent:
            _warn(f"Invalid trend type: {trend_type}. Switching to 'Z'.")
        trend_type = "Z"
        model_do = "select"

    if season_type not in valid_season:
        if not silent:
            _warn(f"Invalid seasonal type: {season_type}. Switching to 'Z'.")
        season_type = "Z"
        model_do = "select"

    # Handle model selection/combination mode
    # Track if combination was requested (to avoid overriding later)
    is_combination = "C" in [error_type, trend_type, season_type]
    if is_combination:
        model_do = "combine"
        # Replace C with Z for actual fitting
        if error_type == "C":
            error_type = "Z"
        if trend_type == "C":
            trend_type = "Z"
        if season_type == "C":
            season_type = "Z"

    # Handle sensible pool case ("S") - matches R adamGeneral.R lines 393-430
    if "S" in [error_type, trend_type, season_type]:
        model_do = "select"
        # Build sensible pool (19 models with finite variance)
        sensible_pool = [
            "ANN",
            "AAN",
            "AAdN",
            "ANA",
            "AAA",
            "AAdA",
            "MNN",
            "MAN",
            "MAdN",
            "MNA",
            "MAA",
            "MAdA",
            "MNM",
            "MAM",
            "MAdM",
            "MMN",
            "MMdN",
            "MMM",
            "MMdM",
        ]

        # Filter by error type if not S
        if error_type != "S":
            target = {"X": "A", "Y": "M"}.get(error_type, error_type)
            sensible_pool = [m for m in sensible_pool if m[0] == target]
        else:
            error_type = "Z"

        # Filter by trend type if not S
        if trend_type != "S":
            if trend_type == "X":
                sensible_pool = [m for m in sensible_pool if m[1] in ["A", "N"]]
            elif trend_type == "Y":
                sensible_pool = [m for m in sensible_pool if m[1] in ["M", "N"]]
            else:
                sensible_pool = [m for m in sensible_pool if m[1] == trend_type]
        else:
            trend_type = "Z"

        # Filter by season type if not S
        if season_type != "S":
            if season_type == "X":
                sensible_pool = [m for m in sensible_pool if m[-1] in ["A", "N"]]
            elif season_type == "Y":
                sensible_pool = [m for m in sensible_pool if m[-1] in ["M", "N"]]
            else:
                sensible_pool = [m for m in sensible_pool if m[-1] == season_type]
        else:
            season_type = "Z"

        # Filter for max_lag (no seasonality if <= 1)
        if max_lag <= 1:
            sensible_pool = [m for m in sensible_pool if m[-1] == "N"]

        # Apply multiplicative restrictions if data has non-positive values
        if not allow_multiplicative:
            sensible_pool = [m for m in sensible_pool if m[0] == "A"]
            sensible_pool = [m for m in sensible_pool if m[1] != "M"]
            sensible_pool = [m for m in sensible_pool if m[-1] != "M"]

        models_pool = sensible_pool

    elif any(
        c in ["Z", "X", "Y", "F", "P"] for c in [error_type, trend_type, season_type]
    ):
        # Don't override "combine" mode - only set to "select" if not already combining
        if not is_combination:
            model_do = "select"

    # Handle multiplicative restrictions
    if not allow_multiplicative:
        if error_type == "M":
            error_type = "A"
        if trend_type == "M":
            trend_type = "A"
        if season_type == "M":
            season_type = "A"
        if error_type == "Y":
            error_type = "X"
        if trend_type == "Y":
            trend_type = "X"
        if season_type == "Y":
            season_type = "X"

    # Generate models pool if needed
    # Only pre-generate pool when components are specific (not Z, X, Y)
    # When Z, X, or Y is present, leave models_pool as None for branch-and-bound
    use_branch_and_bound = any(
        c in ["Z", "X", "Y"] for c in [error_type, trend_type, season_type]
    )
    if model_do in ["select", "combine"] and not use_branch_and_bound:
        models_pool, _ = _build_models_pool_from_components(
            error_type, trend_type, season_type, damped, allow_multiplicative, max_lag
        )

    # Return model components and info
    return {
        "model": model_str,
        "error_type": error_type,
        "trend_type": trend_type,
        "season_type": season_type,
        "damped": damped,
        "model_do": model_do,
        "models_pool": models_pool,
        "allow_multiplicative": allow_multiplicative,
    }


def _generate_models_pool(
    error_type, trend_type, season_type, allow_multiplicative, silent=False
):
    """
    Generate a pool of models based on component specifications.

    Parameters
    ----------
    error_type : str
        Error component type
    trend_type : str
        Trend component type
    season_type : str
        Seasonal component type
    allow_multiplicative : bool
        Whether multiplicative models are allowed
    silent : bool, optional
        Whether to suppress warnings

    Returns
    -------
    tuple
        (pool_small, pool_errors, pool_trends, pool_seasonals, check_trend,
        check_seasonal)
    """
    # Print status if not silent
    if not silent:
        print("Forming the pool of models based on... ", end="")

    # Define the whole pool of errors, trends, and seasonals
    if not allow_multiplicative:
        pool_errors = ["A"]
        pool_trends = ["N", "A", "Ad"]
        pool_seasonals = ["N", "A"]
    else:
        pool_errors = ["A", "M"]
        pool_trends = ["N", "A", "Ad", "M", "Md"]
        pool_seasonals = ["N", "A", "M"]

    # Prepare error type
    if error_type != "Z":
        pool_errors = [error_type]
        pool_errors_small = [error_type]
    else:
        pool_errors_small = ["A"]

    # Prepare trend type
    if trend_type != "Z":
        if trend_type == "X":
            pool_trends_small = ["N", "A"]
            pool_trends = ["N", "A", "Ad"]
            check_trend = True
        elif trend_type == "Y":
            pool_trends_small = ["N", "M"]
            pool_trends = ["N", "M", "Md"]
            check_trend = True
        else:
            damped = "d" in trend_type
            if damped:
                pool_trends = pool_trends_small = [trend_type]
            else:
                pool_trends = pool_trends_small = [trend_type]
            check_trend = False
    else:
        pool_trends_small = ["N", "A"]
        check_trend = True

    # Prepare seasonal type
    if season_type != "Z":
        if season_type == "X":
            pool_seasonals = pool_seasonals_small = ["N", "A"]
            check_seasonal = True
        elif season_type == "Y":
            pool_seasonals_small = ["N", "M"]
            pool_seasonals = ["N", "M"]
            check_seasonal = True
        else:
            pool_seasonals_small = [season_type]
            pool_seasonals = [season_type]
            check_seasonal = False
    else:
        pool_seasonals_small = ["N", "A", "M"]
        check_seasonal = True

    # Create the small pool
    pool_small = []
    for error in pool_errors_small:
        for trend in pool_trends_small:
            for seasonal in pool_seasonals_small:
                pool_small.append(error + trend + seasonal)

    # Align error and seasonality, if the error was not forced to be additive
    if any(model[2] == "M" for model in pool_small) and error_type not in ["A", "X"]:
        for i, model in enumerate(pool_small):
            if model[2] == "M":
                pool_small[i] = "M" + model[1:]

    return (
        pool_small,
        pool_errors,
        pool_trends,
        pool_seasonals,
        check_trend,
        check_seasonal,
    )


def _check_ets_model(model, distribution, data, silent=False, max_lag=1):
    """
    Check ETS model and validate compatibility with data.

    Parameters
    ----------
    model : str
        Model string specification
    distribution : str
        Probability distribution to use
    data : array-like
        Time series data
    silent : bool, optional
        Whether to suppress warnings
    max_lag : int, default=1
        Maximum lag value. If <= 1, seasonality is not allowed.

    Returns
    -------
    dict
        Dictionary with ETS model information
    """
    # Determine if multiplicative models are allowed
    data_arr = np.array(data)
    allow_multiplicative = not np.any(np.asarray(data_arr) <= 0)

    # Handle list/tuple of model strings (R: model=c("ANN","AAN","AAA"))
    if isinstance(model, (list, tuple)):
        # Check if these are ETS model strings (3-4 chars each)
        all_ets = all(isinstance(m, str) and len(m) in (3, 4) for m in model)
        if not all_ets:
            # Not ETS model strings — treat as ARIMA orders
            return {
                "ets_model": False,
                "model": "ANN",
                "error_type": "A",
                "trend_type": "N",
                "season_type": "N",
                "damped": False,
                "allow_multiplicative": allow_multiplicative,
            }

        # Separate combination ("C") models from regular ones
        has_combiner = [any(c == "C" for c in m) for m in model]
        regular_pool = [m for m, is_c in zip(model, has_combiner) if not is_c]
        regular_pool = list(dict.fromkeys(regular_pool))  # unique, order-preserved

        if any(has_combiner):
            # If any combination models, infer base as CCC/CCN
            base_model = "CCC" if any(m[-1] != "N" for m in model) else "CCN"
        else:
            # Infer base model: fix components that are uniform across pool
            base_model = ["Z", "Z", "Z"]
            if all(m[-1] == "N" for m in regular_pool):
                base_model[2] = "N"
            if all(m[1] == "N" for m in regular_pool):
                base_model[1] = "N"
            base_model = "".join(base_model)

        # Parse the inferred base model normally
        model_info = _check_model_composition(
            base_model, allow_multiplicative, silent, max_lag
        )
        # Override models_pool with the user-provided list
        model_info["models_pool"] = regular_pool if regular_pool else None
        model_info["ets_model"] = True
        model_info["allow_multiplicative"] = allow_multiplicative
        return model_info

    # If ARIMA, return default ETS off settings
    if model in ["auto.arima", "ARIMA"]:
        return {
            "ets_model": False,
            "model": "ANN",
            "error_type": "A",
            "trend_type": "N",
            "season_type": "N",
            "damped": False,
            "allow_multiplicative": allow_multiplicative,
        }

    # Check if pure ARIMA is requested (not hybrid)
    if isinstance(model, str) and model.upper() in ["ARIMA", "AUTO.ARIMA"]:
        return {
            "ets_model": False,
            "model": "ANN",
            "error_type": "A",
            "trend_type": "N",
            "season_type": "N",
            "damped": False,
            "allow_multiplicative": allow_multiplicative,
        }

    # Check if NNN (pure ARIMA, no ETS)
    if isinstance(model, str) and model.upper() == "NNN":
        return {
            "ets_model": False,
            "model": "NNN",
            "error_type": "A",
            "trend_type": "N",
            "season_type": "N",
            "damped": False,
            "allow_multiplicative": allow_multiplicative,
        }

    # Check if this is an ETS model
    if isinstance(model, str) and len(model) in [3, 4]:
        # Parse model configuration
        model_info = _check_model_composition(
            model, allow_multiplicative, silent, max_lag
        )

        # Check for multiplicative compatibility with data
        if not allow_multiplicative:
            if model_info["error_type"] == "M":
                _warn(
                    "Switching to additive error because data has non-positive values.",
                    silent,
                )
                model_info["error_type"] = "A"
            if model_info["trend_type"] == "M":
                _warn(
                    "Switching to additive trend because data has non-positive values.",
                    silent,
                )
                model_info["trend_type"] = "A"
            if model_info["season_type"] == "M":
                _warn(
                    "Switching to additive seasonal because data has "
                    "non-positive values.",
                    silent,
                )
                model_info["season_type"] = "A"

            # Update model string based on changes
            model_str = f"{model_info['error_type']}{model_info['trend_type']}"
            if model_info["damped"]:
                model_str += "d"
            model_str += model_info["season_type"]
            model_info["model"] = model_str

        # Indicate this is an ETS model
        model_info["ets_model"] = True
        model_info["allow_multiplicative"] = allow_multiplicative

        return model_info

    # Default to ANN for unrecognized models
    _warn(f"Unknown model type: {model}. Switching to 'ANN'.", silent)
    return {
        "ets_model": True,
        "model": "ANN",
        "error_type": "A",
        "trend_type": "N",
        "season_type": "N",
        "damped": False,
        "allow_multiplicative": allow_multiplicative,
    }
