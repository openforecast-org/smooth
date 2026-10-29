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
    n_param_exo=0,
    silent=False,
    n_param_max=None,
    damped=False,
):
    """
    Restrict the models pool based on sample size.

    Translates the ETS part of R's "Checks for the potential number of degrees of
    freedom" (R/adamGeneral.R).
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
    n_param_exo : int
        Number of estimated parameters of the regressors (R's nParamExo)
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

    # R's warning, for the model as it is when it is raised
    model = error_type + trend_type + ("d" if damped else "") + season_type
    fitting_what_i_can = (
        "Not enough of non-zero observations for the fit of ETS({})! "
        "Fitting what I can..."
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
                new_pool.append("AAN")
                # MAN sits in this branch for its additive trend, but its error
                # is multiplicative and must be allowed.
                if allow_multiplicative:
                    new_pool.append("MAN")
            if allow_multiplicative and trend_type in ["Z", "Y", "M"]:
                new_pool.extend(["AMN", "MMN"])

        # Enough for damped trend model
        if obs_nonzero > (6 + n_param_exo):
            if trend_type in ["Z", "X", "A"]:
                new_pool.append("AAdN")
                if allow_multiplicative:
                    new_pool.append("MAdN")
            if allow_multiplicative and trend_type in ["Z", "Y", "M"]:
                new_pool.extend(["AMdN", "MMdN"])

        # Enough for seasonal model
        if obs_nonzero > lags_model_max and lags_model_max != 1:
            if season_type in ["Z", "X", "A"]:
                new_pool.append("ANA")
                if allow_multiplicative:
                    new_pool.append("MNA")
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

        _warn(fitting_what_i_can.format(model), silent)

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

        result["models_pool"] = list(dict.fromkeys(filtered_pool))
        result["model_do"] = "combine" if model_do == "combine" else "select"
        _warn(fitting_what_i_can.format(model), silent)

    # Handle estimate/use mode (R lines 2778-2832)
    elif obs_nonzero > (3 + n_param_exo) and model_do in ["estimate", "use"]:
        seasonal = season_type != "N"

        # Remove damped from seasonal models if not enough obs
        if (
            len(model) == 4
            and seasonal
            and obs_nonzero <= (6 + lags_model_max + 1 + n_param_exo)
        ):
            _warn(fitting_what_i_can.format(model), silent)
            model = model[:2] + model[3]

        # Remove trend from seasonal models if not enough obs
        if (
            model[1] != "N"
            and seasonal
            and obs_nonzero <= (5 + lags_model_max + 1 + n_param_exo)
        ):
            _warn(fitting_what_i_can.format(model), silent)
            model = model[0] + "N" + model[-1]

        # Remove seasonal if not enough obs
        if seasonal and obs_nonzero <= lags_model_max:
            _warn(fitting_what_i_can.format(model), silent)
            model = model[:-1] + "N"

        # Remove damped if not enough obs
        if len(model) == 4 and obs_nonzero <= (6 + n_param_exo):
            _warn(fitting_what_i_can.format(model), silent)
            model = model[:2] + model[3]

        # Remove any trend if not enough obs
        if model[1] != "N" and obs_nonzero <= (5 + n_param_exo):
            _warn(fitting_what_i_can.format(model), silent)
            model = model[0] + "N" + model[-1]

        # Update result based on simplified model (R lines 2822-2832)
        result["error_type"] = model[0]
        result["trend_type"] = model[1]
        result["season_type"] = model[-1]  # Last character is always season

        # Update damped and phi_estimate based on final model
        model_is_trendy = model[1] != "N"
        result["damped"] = model_is_trendy and len(model) == 4
        result["phi_estimate"] = result["damped"]

    # Extreme cases, after the branches above as in R
    elif obs_nonzero == 4:
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
            _warn(
                "You have a very small sample. "
                "The only available model is level model.",
                silent,
            )
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
        _warn(
            "I did not have enough of non-zero observations, "
            "so persistence value was set to zero.",
            silent,
        )

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
        _warn(
            "I did not have enough of non-zero observations, so persistence value "
            "was set to zero and initial was preset.",
            silent,
        )

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
        _warn(
            "I did not have enough of non-zero observations, so I used Naive.",
            silent,
        )

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
        _warn("You have a sample with zeroes only. Your forecast will be zero.", silent)

    else:
        raise ValueError("Not enough observations... Even for fitting of ETS('ANN')!")

    return result
