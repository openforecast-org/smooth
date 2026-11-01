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
    if n_param_max is not None:
        if obs_nonzero > n_param_max:
            return result
        # Print message when restrictions are being applied (R lines 2657-2661)
        if not silent:
            print(
                f"Number of non-zero observations is {obs_nonzero}, "
                f"while the maximum number of parameters to estimate is {n_param_max}."
                "\nUpdating pool of models."
            )

    if obs_nonzero <= 3 + n_param_exo:
        _restrict_tiny_sample(
            result, obs_nonzero, error_type, allow_multiplicative, silent
        )
        return result

    # R's warning, for the model as it is when it is raised
    model = error_type + trend_type + "d" * damped + season_type
    # A pool: built for the selection, or what the sample can fit of the one given
    if models_pool is not None or model_do in ["select", "combine"]:
        result["models_pool"] = (
            _filter_small_sample_pool(
                models_pool, obs_nonzero, lags_model_max, n_param_exo
            )
            if models_pool is not None
            else _build_small_sample_pool(
                obs_nonzero,
                lags_model_max,
                trend_type,
                season_type,
                allow_multiplicative,
                n_param_exo,
            )
        )
        result["model_do"] = "combine" if model_do == "combine" else "select"
        _warn(_FITTING_WHAT_I_CAN.format(model), silent)
    # The model simplified to what the sample can fit (R lines 2778-2832)
    else:
        model = _simplify_small_sample_model(
            model, obs_nonzero, lags_model_max, n_param_exo, silent
        )
        result.update(
            error_type=model[0],
            trend_type=model[1],
            season_type=model[-1],
            damped=len(model) == 4,
            phi_estimate=len(model) == 4,
        )
    return result


# The candidates of the pool of a small sample, in R's order, with what they need: a
# trend (T), a damped one (D) or none (N), and a seasonality (S) or none (N)
_SMALL_SAMPLE_POOL = (
    ("ANN", "NN"), ("MNN", "NN"),
    ("AAN", "TN"), ("MAN", "TN"), ("AMN", "TN"), ("MMN", "TN"),
    ("AAdN", "DN"), ("MAdN", "DN"), ("AMdN", "DN"), ("MMdN", "DN"),
    ("ANA", "NS"), ("MNA", "NS"), ("ANM", "NS"), ("MNM", "NS"),
    ("AAA", "TS"), ("MAA", "TS"), ("AAM", "TS"), ("MAM", "TS"),
    ("AMA", "TS"), ("MMA", "TS"), ("AMM", "TS"), ("MMM", "TS"),
)  # fmt: skip

_FITTING_WHAT_I_CAN = (
    "Not enough of non-zero observations for the fit of ETS({})! Fitting what I can..."
)


def _build_small_sample_pool(
    obs_nonzero,
    lags_model_max,
    trend_type,
    season_type,
    allow_multiplicative,
    n_param_exo,
):
    """The pool of the selection on a small sample (R/adamGeneral.R): the candidates
    the sample can fit, with the components the model letters allow."""
    seasonal = lags_model_max != 1
    enough = {
        "NN": True,
        "TN": obs_nonzero > 5 + n_param_exo,
        "DN": obs_nonzero > 6 + n_param_exo,
        "NS": seasonal and obs_nonzero > lags_model_max,
        "TS": seasonal
        and obs_nonzero > max(6 + lags_model_max + n_param_exo, 2 * lags_model_max),
    }
    trend_allowed = {"N": True, "A": trend_type in "ZXA", "M": trend_type in "ZYM"}
    season_allowed = {"N": True, "A": season_type in "ZXA", "M": season_type in "ZYM"}
    return [
        model
        for model, needs in _SMALL_SAMPLE_POOL
        if enough[needs]
        and trend_allowed[model[1]]
        and season_allowed[model[-1]]
        and (allow_multiplicative or "M" not in model)
    ]


def _filter_small_sample_pool(models_pool, obs_nonzero, lags_model_max, n_param_exo):
    """The models of a pool that a small sample can fit: those it does not simplify."""
    return list(
        dict.fromkeys(
            model
            for model in models_pool
            if _simplify_small_sample_model(
                model, obs_nonzero, lags_model_max, n_param_exo, True
            )
            == model
        )
    )


def _simplify_small_sample_model(
    model, obs_nonzero, lags_model_max, n_param_exo, silent
):
    """The model without the damping, the trend and the seasonality, in this order,
    while the sample cannot fit it (R lines 2778-2832)."""
    # The observations needed by a trend (T), a damped one (D) or none (N), with a
    # seasonality (S) or without (N); the level alone needs none
    needed = {
        "TN": 5 + n_param_exo,
        "DN": 6 + n_param_exo,
        "NS": lags_model_max,
        "TS": 6 + lags_model_max + n_param_exo,
        "DS": 7 + lags_model_max + n_param_exo,
    }
    while True:
        trend = "D" if len(model) == 4 else ("N" if model[1] == "N" else "T")
        components = trend + ("N" if model[-1] == "N" else "S")
        if components == "NN" or obs_nonzero > needed[components]:
            return model
        _warn(_FITTING_WHAT_I_CAN.format(model), silent)
        if trend == "D":
            model = model[:2] + model[3:]
        elif trend == "T":
            model = model[0] + "N" + model[-1]
        else:
            model = model[:-1] + "N"


def _restrict_tiny_sample(
    result, obs_nonzero, error_type, allow_multiplicative, silent
):
    """The model of the samples of zero to four non-zero observations, in ``result``
    (R's extreme cases of small samples)."""
    if obs_nonzero > 4:
        raise ValueError("Not enough observations... Even for fitting of ETS('ANN')!")
    result.update(phi_estimate=False, damped=False, trend_type="N", season_type="N")
    # A level model: estimated, or selected without the error type
    if obs_nonzero >= 3:
        if error_type in ["A", "M"]:
            result["model_do"] = "estimate"
        else:
            result.update(
                models_pool=["ANN", "MNN"] if allow_multiplicative else ["ANN"],
                model_do="select",
                error_type="Z",
            )
            if obs_nonzero == 4:
                _warn(
                    "You have a very small sample. "
                    "The only available model is level model.",
                    silent,
                )
        if obs_nonzero == 3:
            result.update(persistence_level=0, persistence_estimate=False)
            _warn(
                "I did not have enough of non-zero observations, "
                "so persistence value was set to zero.",
                silent,
            )
        return
    # The level model with zero persistence and the initial preset
    result.update(
        models_pool=None,
        persistence_level=0,
        persistence_estimate=False,
        initial_type="provided",
        initial_estimate=False,
        model_do="use",
        error_type="A",
    )
    _warn(
        {
            2: "I did not have enough of non-zero observations, so persistence value "
            "was set to zero and initial was preset.",
            1: "I did not have enough of non-zero observations, so I used Naive.",
            0: "You have a sample with zeroes only. Your forecast will be zero.",
        }[obs_nonzero],
        silent,
    )
