import numpy as np

from ._utils import _warn
from .arima_checks import _check_arima
from .organizers import _calculate_n_param_max


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


def _check_sample_size(
    ets_info,
    arima_info,
    persist_info,
    init_info,
    phi_estimate,
    model_do,
    obs_nonzero,
    obs_in_sample,
    y_na_values,
    data,
    occ_info,
    distribution,
    validated_lags,
    lags_model_seasonal,
    max_lag,
    has_xreg,
    xreg_number,
    regressors,
    silent,
):
    """The model, the pool and the parameters that the sample can fit (R's
    adam_checkSampleSize): ARIMA is dropped next to ETS, backcast or replaced by
    ARIMA(0,1,1) alone, the pool of ETS is restricted, and the level of the samples
    too small to estimate anything is set. Returns the updated checks of ETS, ARIMA,
    the persistence and the initials, phi_estimate, model_do and obs_nonzero."""
    ets_model = ets_info["ets_model"]
    arima_model = arima_info["arima_model"]
    ar_orders = arima_info["ar_orders"]
    ma_orders = arima_info["ma_orders"]
    # Whether multiplicative models are allowed (resolved in _check_ets_model)
    allow_multiplicative = ets_info["allow_multiplicative"]

    #  Calculate n_param_max to determine if pool restriction is needed (R lines
    # 2641-2651)
    model_is_trendy = ets_info["trend_type"] not in ["N", None]
    model_is_seasonal = (
        ets_info["season_type"] not in ["N", None] and len(lags_model_seasonal) > 0
    )

    # The occurrence model is fitted to all the observed values, not to the non-zero
    # ones, as R
    if distribution == "plogis":
        obs_nonzero = obs_in_sample - int(np.sum(y_na_values[:obs_in_sample]))

    n_param_arguments = dict(
        ets_model=ets_model,
        persistence_level_estimate=persist_info.get("persistence_level_estimate", True),
        model_is_trendy=model_is_trendy,
        persistence_trend_estimate=persist_info.get("persistence_trend_estimate", True),
        model_is_seasonal=model_is_seasonal,
        persistence_seasonal_estimate=persist_info.get(
            "persistence_seasonal_estimate", [True] * len(lags_model_seasonal)
        ),
        phi_estimate=phi_estimate,
        initial_type=init_info.get("initial_type", "optimal"),
        initial_level_estimate=init_info.get("initial_level_estimate", True),
        initial_trend_estimate=init_info.get("initial_trend_estimate", True),
        initial_seasonal_estimate=init_info["initial_seasonal_estimate"],
        lags_model_seasonal=lags_model_seasonal,
        arima_model=arima_model,
        initial_arima_number=arima_info.get("initial_arima_number", 0),
        ar_required=any(ar_orders) if ar_orders else False,
        ar_estimate=arima_info.get("ar_estimate", True),
        ar_orders=ar_orders,
        ma_required=any(ma_orders) if ma_orders else False,
        ma_estimate=arima_info.get("ma_estimate", True),
        ma_orders=ma_orders,
        xreg_model=has_xreg,
        xreg_number=xreg_number,
        initial_xreg_estimate=has_xreg,
        persistence_xreg_estimate=(has_xreg and regressors == "adapt"),
    )

    # ARIMA on a small sample, as R: dropped next to ETS; alone, backcast if that
    # saves enough, else ARIMA(0,1,1)
    n_param_max = _calculate_n_param_max(**n_param_arguments)
    initial_optimal = init_info.get("initial_type") in ["optimal", "two-stage"]
    initial_arima_number = arima_info.get("initial_arima_number", 0)
    if obs_nonzero <= n_param_max and arima_model and not arima_info["select"]:
        if ets_model:
            _warn(
                "I don't have enough observations to fit ETS with ARIMA terms. "
                "I will construct the simple ETS."
            )
            arima_info = _check_arima(None, validated_lags, silent)
        elif initial_optimal and (obs_nonzero > n_param_max - initial_arima_number):
            _warn(
                f"The number of parameter to estimate is {n_param_max}, while the "
                f"number of observations is {obs_nonzero}. Switching initial to "
                "'backcasting' to save some degrees of freedom."
            )
            init_info["initial_type"] = "complete"
            # R sets the iterations from the initial after this check
            if not init_info["n_iterations_provided"]:
                init_info["n_iterations"] = 2
        else:
            _warn(
                f"The number of parameter to estimate is {n_param_max}, while the "
                f"number of observations is {obs_nonzero}. Switching to "
                "ARIMA(0,1,1) model."
            )
            arima_info = _check_arima({"ar": 0, "i": 1, "ma": 1}, [1], silent)
        arima_model = arima_info["arima_model"]
        ar_orders = arima_info["ar_orders"]
        ma_orders = arima_info["ma_orders"]
        init_info["initial_arima_number"] = arima_info["initial_arima_number"]
        n_param_arguments.update(
            arima_model=arima_model,
            initial_arima_number=arima_info["initial_arima_number"],
            ar_required=arima_info["ar_required"],
            ar_estimate=arima_info["ar_estimate"],
            ar_orders=ar_orders,
            ma_required=arima_info["ma_required"],
            ma_estimate=arima_info["ma_estimate"],
            ma_orders=ma_orders,
            initial_type=init_info["initial_type"],
        )
    if (
        arima_model
        and obs_nonzero
        < (init_info.get("initial_type") in ["optimal", "two-stage"])
        * arima_info.get("initial_arima_number", 0)
        and not arima_info["select"]
    ):
        raise ValueError("Not enough observations for such a complicated model.")
    n_param_max = _calculate_n_param_max(**n_param_arguments)

    # Restrict models pool based on sample size (R's "Checks for the potential
    # number of degrees of freedom"), for ETS only
    pool_restriction = _restrict_models_pool_for_sample_size(
        obs_nonzero=obs_nonzero,
        lags_model_max=max_lag,
        model_do=model_do,
        error_type=ets_info["error_type"],
        trend_type=ets_info["trend_type"],
        season_type=ets_info["season_type"],
        models_pool=ets_info.get("models_pool"),
        allow_multiplicative=allow_multiplicative,
        n_param_exo=xreg_number
        * (int(has_xreg) + int(has_xreg and regressors == "adapt")),
        silent=silent,
        # R checks the ETS part for an ETS model only
        n_param_max=n_param_max if ets_model else -1,
        damped=ets_info.get("damped", False),
    )

    # Update ets_info with restricted values
    model_changed = False
    if pool_restriction["error_type"] != ets_info["error_type"]:
        ets_info["error_type"] = pool_restriction["error_type"]
        model_changed = True
    if pool_restriction["trend_type"] != ets_info["trend_type"]:
        ets_info["trend_type"] = pool_restriction["trend_type"]
        model_changed = True
    if pool_restriction["season_type"] != ets_info["season_type"]:
        ets_info["season_type"] = pool_restriction["season_type"]
        model_changed = True
    if pool_restriction["models_pool"] is not None:
        ets_info["models_pool"] = pool_restriction["models_pool"]
    if pool_restriction["model_do"] != model_do:
        model_do = pool_restriction["model_do"]
        ets_info["model_do"] = model_do
    # Update damped flag if it was changed
    if pool_restriction["damped"] != ets_info.get("damped", False):
        ets_info["damped"] = pool_restriction["damped"]
        model_changed = True

    # Rebuild model string if any component changed
    if model_changed:
        new_model = ets_info["error_type"] + ets_info["trend_type"]
        if ets_info["damped"] and ets_info["trend_type"] != "N":
            new_model += "d"
        new_model += ets_info["season_type"]
        ets_info["model"] = new_model

    # Update persistence if restricted: alpha is set, as R's persistenceLevel
    if pool_restriction["persistence_level"] is not None:
        persist_info["persistence_level"] = pool_restriction["persistence_level"]
    if not pool_restriction["persistence_estimate"]:
        persist_info["persistence_estimate"] = False
        persist_info["persistence_level_estimate"] = False

    # Update phi_estimate if restricted
    if not pool_restriction["phi_estimate"]:
        phi_estimate = False

    # Update initial if restricted
    if pool_restriction["initial_type"] is not None:
        init_info["initial_type"] = pool_restriction["initial_type"]
    if not pool_restriction["initial_estimate"]:
        init_info["initial_estimate"] = False
        init_info["initial_level_estimate"] = False
    # The level of the samples too small to estimate anything, as R: the mean of the
    # values the occurrence marks (all of them without one), zero without any
    if pool_restriction["model_do"] == "use" and obs_nonzero <= 2:
        y_is = np.asarray(data, dtype=float).ravel()[:obs_in_sample]
        no_occurrence = (
            isinstance(occ_info["occurrence"], str) and occ_info["occurrence"] == "none"
        )
        marked = ~np.isnan(y_is) & ((y_is != 0) | no_occurrence)
        init_info["initial_level"] = (
            float(np.mean(y_is[marked])) if obs_nonzero > 0 else 0.0
        )

    return (
        ets_info,
        arima_info,
        persist_info,
        init_info,
        phi_estimate,
        model_do,
        obs_nonzero,
    )
