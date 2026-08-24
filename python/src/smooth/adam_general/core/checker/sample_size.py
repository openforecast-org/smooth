from ._utils import _warn


def _restrict_models_pool_for_sample_size(
    obs_nonzero,
    lags_model_max,
    model_do,
    error_type,
    trend_type,
    season_type,
    models_pool,
    allow_multiplicative=True,
    xreg_number=0,
    silent=False,
    n_param_max=None,
    damped=False,
):
    """
    Restrict the models pool based on sample size.

    Follows the logic from R's adamGeneral.R lines 2641-2944.
    Only applies restrictions when obs_nonzero <= n_param_max.

    Parameters
    ----------
    obs_nonzero : int
        Number of non-zero observations
    lags_model_max : int
        Maximum lag value
    model_do : str
        Model action: 'estimate', 'select', 'combine', or 'use'
    error_type : str
        Error type character
    trend_type : str
        Trend type character
    season_type : str
        Season type character
    models_pool : list or None
        Current models pool
    allow_multiplicative : bool
        Whether multiplicative models are allowed
    xreg_number : int
        Number of external regressors
    silent : bool
        Whether to suppress warnings
    n_param_max : int or None
        Maximum number of parameters. If None, restrictions always apply.
        If provided, restrictions only apply when obs_nonzero <= n_param_max.
    damped : bool
        Whether the model has damped trend

    Returns
    -------
    dict
        Updated model configuration with restricted pool
    """
    n_param_exo = xreg_number * 2  # xreg coefficients + persistence

    result = {
        "model_do": model_do,
        "error_type": error_type,
        "trend_type": trend_type,
        "season_type": season_type,
        "models_pool": models_pool,
        "persistence_level": None,
        "persistence_estimate": True,
        "initial_type": None,
        "initial_estimate": True,
        "phi_estimate": True,
        "damped": damped,
    }

    # Only apply restrictions if n_param_max is None or obs_nonzero <= n_param_max
    # Following R line 2655: if(obsNonzero <= nParamMax)
    if n_param_max is not None and obs_nonzero > n_param_max:
        return result

    # Print message when restrictions are being applied (R lines 2657-2661)
    if n_param_max is not None and not silent:
        print(
            f"Number of non-zero observations is {obs_nonzero}, "
            f"while the maximum number of parameters to estimate is {n_param_max}.\n"
            "Updating pool of models."
        )

    # If pool not specified and select/combine mode, build restricted pool
    if (
        obs_nonzero > (3 + n_param_exo)
        and models_pool is None
        and model_do in ["select", "combine"]
    ):
        new_pool = ["ANN"]
        if allow_multiplicative:
            new_pool.append("MNN")

        # A model belongs to the branch of its *component* letter, not its error
        # letter: MAN has an additive trend and only a multiplicative error, so
        # it sits with AAN under "Z"/"X"/"A" -- likewise MAdN with AAdN, and MNA
        # with ANA. Mirrors R/adamGeneral.R:2483-2507.
        # Enough for trend model
        if obs_nonzero > (5 + n_param_exo):
            if trend_type in ["Z", "X", "A"]:
                new_pool.extend(["AAN", "MAN"])
            if allow_multiplicative and trend_type in ["Z", "Y", "M"]:
                new_pool.extend(["AMN", "MMN"])

        # Enough for damped trend model
        if obs_nonzero > (6 + n_param_exo):
            if trend_type in ["Z", "X", "A"]:
                new_pool.extend(["AAdN", "MAdN"])
            if allow_multiplicative and trend_type in ["Z", "Y", "M"]:
                new_pool.extend(["AMdN", "MMdN"])

        # Enough for seasonal model
        if obs_nonzero > lags_model_max and lags_model_max != 1:
            if season_type in ["Z", "X", "A"]:
                new_pool.extend(["ANA", "MNA"])
            if allow_multiplicative and season_type in ["Z", "Y", "M"]:
                new_pool.extend(["ANM", "MNM"])

        # Enough for seasonal model with trend
        if (
            obs_nonzero > (6 + lags_model_max + n_param_exo)
            and obs_nonzero > 2 * lags_model_max
            and lags_model_max != 1
        ):
            if trend_type in ["Z", "X", "A"] and season_type in ["Z", "X", "A"]:
                new_pool.append("AAA")
            if allow_multiplicative:
                if trend_type in ["Z", "X", "A"] and season_type in ["Z", "Y", "A"]:
                    new_pool.append("MAA")
                if trend_type in ["Z", "X", "A"] and season_type in ["Z", "Y", "M"]:
                    new_pool.extend(["AAM", "MAM"])
                if trend_type in ["Z", "Y", "M"] and season_type in ["Z", "X", "A"]:
                    new_pool.extend(["AMA", "MMA"])
                if trend_type in ["Z", "Y", "M"] and season_type in ["Z", "Y", "M"]:
                    new_pool.extend(["AMM", "MMM"])

        result["models_pool"] = new_pool
        if model_do == "combine":
            result["model_do"] = "combine"
        else:
            result["model_do"] = "select"

        if not silent:
            _warn(
                f"Not enough observations for full model pool. "
                f"Fitting restricted pool: {new_pool}"
            )

    # If pool is provided, filter it based on available observations
    elif obs_nonzero > (3 + n_param_exo) and models_pool is not None:
        filtered_pool = list(models_pool)

        # Remove damped seasonal models if not enough obs
        if obs_nonzero <= (6 + lags_model_max + 1 + n_param_exo):
            filtered_pool = [
                m for m in filtered_pool if not (len(m) == 4 and m[-1] in ["A", "M"])
            ]

        # Remove seasonal + trend models if not enough obs
        if obs_nonzero <= (5 + lags_model_max + 1 + n_param_exo):
            filtered_pool = [
                m for m in filtered_pool if not (m[1] != "N" and m[-1] != "N")
            ]

        # Remove seasonal models if not enough obs
        if obs_nonzero <= lags_model_max:
            filtered_pool = [m for m in filtered_pool if m[-1] == "N"]

        # Remove damped trend if not enough obs
        if obs_nonzero <= (6 + n_param_exo):
            filtered_pool = [m for m in filtered_pool if len(m) != 4]

        # Remove any trend if not enough obs
        if obs_nonzero <= (5 + n_param_exo):
            filtered_pool = [m for m in filtered_pool if m[1] == "N"]

        result["models_pool"] = list(set(filtered_pool))
        if not silent and len(filtered_pool) < len(models_pool):
            _warn(f"Pool restricted due to sample size: {filtered_pool}")

    # Handle estimate/use mode (R lines 2778-2832)
    elif obs_nonzero > (3 + n_param_exo) and model_do in ["estimate", "use"]:
        # Build model string: E + T + (d if damped) + S
        model = error_type + trend_type
        if damped:
            model += "d"
        model += season_type

        # 1. Remove damped from seasonal models if not enough obs (R lines 2780-2790)
        if obs_nonzero <= (6 + lags_model_max + 1 + n_param_exo):
            if len(model) == 4:  # Damped model with seasonal
                if not silent:
                    _warn(
                        f"Not enough non-zero observations for ETS({model})! "
                        "Fitting what I can..."
                    )
                model = model[:2] + model[3]  # Remove 'd': AAdA -> AAA

        # 2. Remove trend from seasonal models if not enough obs (R lines 2791-2798)
        if obs_nonzero <= (5 + lags_model_max + 1 + n_param_exo):
            if model[1] != "N":  # Has trend
                if not silent:
                    _warn(
                        f"Not enough non-zero observations for ETS({model})! "
                        "Fitting what I can..."
                    )
                model = model[0] + "N" + model[2]  # Remove trend: AAA -> ANA

        # 3. Remove seasonal if not enough obs (R lines 2799-2805)
        if obs_nonzero <= lags_model_max:
            if model[-1] != "N":  # Has seasonal
                if not silent:
                    _warn(
                        f"Not enough non-zero observations for ETS({model})! "
                        "Fitting what I can..."
                    )
                model = model[:2] + "N"  # Remove seasonal: ANA -> ANN

        #  4. Remove damped from non-seasonal models if not enough obs (R lines
        # 2806-2814)
        if obs_nonzero <= (6 + n_param_exo):
            if len(model) == 4:  # Damped model (non-seasonal at this point)
                if not silent:
                    _warn(
                        f"Not enough non-zero observations for ETS({model})! "
                        "Fitting what I can..."
                    )
                model = model[:2] + model[3]  # Remove 'd': AAdN -> AAN

        # 5. Remove any trend if not enough obs (R lines 2815-2821)
        if obs_nonzero <= (5 + n_param_exo):
            if model[1] != "N":  # Has trend
                if not silent:
                    _warn(
                        f"Not enough non-zero observations for ETS({model})! "
                        "Fitting what I can..."
                    )
                model = model[0] + "N" + model[2]  # Remove trend: AAN -> ANN

        # Update result based on simplified model (R lines 2822-2832)
        result["error_type"] = model[0]
        result["trend_type"] = model[1]
        result["season_type"] = model[-1]  # Last character is always season

        # Update damped and phi_estimate based on final model
        model_is_trendy = model[1] != "N"
        result["damped"] = model_is_trendy and len(model) == 4
        result["phi_estimate"] = result["damped"]

    # Extreme cases
    if obs_nonzero == 4:
        if error_type in ["A", "M"]:
            result["model_do"] = "estimate"
            result["trend_type"] = "N"
            result["season_type"] = "N"
        else:
            result["models_pool"] = ["ANN", "MNN"] if allow_multiplicative else ["ANN"]
            result["model_do"] = "select"
            result["error_type"] = "Z"
            result["trend_type"] = "N"
            result["season_type"] = "N"
            if not silent:
                _warn("Very small sample. Only level models available.")
        result["phi_estimate"] = False
        result["damped"] = False

    elif obs_nonzero == 3:
        if error_type in ["A", "M"]:
            result["model_do"] = "estimate"
            result["trend_type"] = "N"
            result["season_type"] = "N"
        else:
            result["models_pool"] = ["ANN", "MNN"] if allow_multiplicative else ["ANN"]
            result["model_do"] = "select"
            result["error_type"] = "Z"
            result["trend_type"] = "N"
            result["season_type"] = "N"
        result["persistence_level"] = 0
        result["persistence_estimate"] = False
        result["phi_estimate"] = False
        result["damped"] = False
        if not silent:
            _warn("Very small sample. Persistence set to zero.")

    elif obs_nonzero == 2:
        result["models_pool"] = None
        result["persistence_level"] = 0
        result["persistence_estimate"] = False
        result["initial_type"] = "provided"
        result["initial_estimate"] = False
        result["model_do"] = "use"
        result["error_type"] = "A"
        result["trend_type"] = "N"
        result["season_type"] = "N"
        result["phi_estimate"] = False
        result["damped"] = False
        if not silent:
            _warn("Sample too small. Using fixed ANN model.")

    elif obs_nonzero == 1:
        result["models_pool"] = None
        result["persistence_level"] = 0
        result["persistence_estimate"] = False
        result["initial_type"] = "provided"
        result["initial_estimate"] = False
        result["model_do"] = "use"
        result["error_type"] = "A"
        result["trend_type"] = "N"
        result["season_type"] = "N"
        result["phi_estimate"] = False
        result["damped"] = False
        if not silent:
            _warn("Only one observation. Using Naive forecast.")

    elif obs_nonzero == 0:
        result["models_pool"] = None
        result["persistence_level"] = 0
        result["persistence_estimate"] = False
        result["initial_type"] = "provided"
        result["initial_estimate"] = False
        result["model_do"] = "use"
        result["error_type"] = "A"
        result["trend_type"] = "N"
        result["season_type"] = "N"
        result["phi_estimate"] = False
        result["damped"] = False
        if not silent:
            _warn("No non-zero observations. Forecast will be zero.")

    elif obs_nonzero <= 3 + n_param_exo:
        raise ValueError(
            f"Not enough observations ({obs_nonzero}) for model estimation."
        )

    return result


def _adjust_model_for_sample_size(
    model_info,
    obs_nonzero,
    lags_model_max,
    allow_multiplicative=True,
    xreg_number=0,
    silent=False,
):
    """
    Adjust model complexity based on sample size.

    Parameters
    ----------
    model_info : dict
        Model type information
    obs_nonzero : int
        Number of non-zero observations
    lags_model_max : int
        Maximum lag value
    allow_multiplicative : bool, optional
        Whether multiplicative models are allowed
    xreg_number : int, optional
        Number of external regressors
    silent : bool, optional
        Whether to suppress warnings

    Returns
    -------
    dict
        Updated model information
    """
    result = model_info.copy()

    # Effective sample size after accounting for lags
    effective_sample = obs_nonzero - lags_model_max

    # Calculate number of parameters needed
    n_params = 0

    # Level parameter
    if model_info["ets_model"]:
        n_params += 1

    # Trend parameter
    if model_info["ets_model"] and model_info["trend_type"] != "N":
        n_params += 1

        # Damping parameter
        if model_info["damped"]:
            n_params += 1

    # Seasonal parameters
    if (
        model_info["ets_model"]
        and model_info["season_type"] != "N"
        and model_info.get("model_is_seasonal", False)
    ):
        # One parameter per seasonal lag
        n_params += model_info.get("components_number_ets_seasonal", 0)

    # External regressors
    n_params += xreg_number

    # Check if we have enough observations
    if effective_sample <= n_params:
        # Not enough observations, simplify the model
        _warn(
            f"Not enough non-zero observations ({obs_nonzero}) for the model "
            f"with {n_params} parameters. Switching to a simpler model.",
            silent,
        )

        # Start by reducing seasonality
        if result["season_type"] != "N":
            result["season_type"] = "N"
            result["model_is_seasonal"] = False
            n_params -= model_info.get("components_number_ets_seasonal", 0)

        # Then remove damping
        if result["damped"] and result["trend_type"] != "N":
            result["damped"] = False
            n_params -= 1

        # Finally remove trend
        if effective_sample <= n_params and result["trend_type"] != "N":
            result["trend_type"] = "N"
            result["model_is_trendy"] = False
            n_params -= 1

        # Update model string
        if result["damped"] and result["trend_type"] != "N":
            result["model"] = (
                f"{result['error_type']}{result['trend_type']}d{result['season_type']}"
            )
        else:
            result["model"] = (
                f"{result['error_type']}{result['trend_type']}{result['season_type']}"
            )

    return result
