import numpy as np

from ._utils import _warn

# The full, the pure and the safe pools ("F", "P" and "S" in the model), in R's order
_NAMED_MODELS_POOLS = {
    "F": (
        "ANN", "AAN", "AAdN", "AMN", "AMdN", "ANA", "AAA", "AAdA", "AMA", "AMdA",
        "ANM", "AAM", "AAdM", "AMM", "AMdM", "MNN", "MAN", "MAdN", "MMN", "MMdN",
        "MNA", "MAA", "MAdA", "MMA", "MMdA", "MNM", "MAM", "MAdM", "MMM", "MMdM",
    ),
    "P": (
        "ANN", "AAN", "AAdN", "ANA", "AAA", "AAdA",
        "MNN", "MMN", "MMdN", "MNM", "MMM", "MMdM",
    ),
    "S": (
        "ANN", "AAN", "AAdN", "ANA", "AAA", "AAdA", "MNN", "MAN", "MAdN", "MNA",
        "MAA", "MAdA", "MNM", "MAM", "MAdM", "MMN", "MMdN", "MMM", "MMdM",
    ),
}  # fmt: skip

# The components a letter of the model allows: X the additive or none, Y the
# multiplicative or none, the selection letters all
_ALLOWED_COMPONENTS = {"X": "AN", "Y": "MN", "N": "N", "A": "A", "M": "M"}


def _named_models_pool(letter, error_type, trend_type, season_type):
    """The full, the pure or the safe pool (``letter`` "F", "P" or "S"), narrowed by
    the other letters of the model, as R's."""
    allowed = [
        _ALLOWED_COMPONENTS.get(component, "NAM")
        for component in (error_type, trend_type, season_type)
    ]
    return [
        model
        for model in _NAMED_MODELS_POOLS[letter]
        if model[0] in allowed[0] and model[1] in allowed[1] and model[-1] in allowed[2]
    ]


def _parse_model_string(model_str, silent):
    """The letters of the model, its damping and what to do with it: the invalid
    letters are selected, "C" combines."""
    model_do = "estimate"
    if not isinstance(model_str, str):
        _warn(
            f"Invalid model type: {model_str}. Should be a string. Switching to 'ZZZ'.",
            silent,
        )
        model_str = "ZZZ"
    if len(model_str) not in (3, 4):
        _warn(f"Invalid model string length: {model_str}. Switching to 'ZZZ'.", silent)
        model_str = "ZZZ"
    if len(model_str) == 4 and model_str[2] != "d":
        _warn(f"Invalid damped trend specification in {model_str}. Using 'd'.", silent)
    # The letters with "C" for the combination of the selected models
    letters = model_str.replace("C", "Z")
    components = [letters[0], letters[1], letters[-1]]
    damped = len(model_str) == 4 or model_str[1] in ["Z", "X", "Y"]

    for i, name in enumerate(("error", "trend", "seasonal")):
        if components[i] not in "ZXYAMNFPS":
            _warn(f"Invalid {name} type: {components[i]}. Switching to 'Z'.", silent)
            components[i] = "Z"
            model_do = "select"
    if letters != model_str:
        model_do = "combine"
    return model_str, *components, damped, model_do


def _restrict_to_additive(
    model_str, error_type, trend_type, season_type, models_pool, silent
):
    """The model on non-positive data, as R's (R/adamGeneral.R, allowMultiplicative):
    an explicit pool amended, a selection switched to the additive models, a warning
    for a specific multiplicative model."""
    if models_pool is not None:
        pool_additive = [m for m in models_pool if "M" not in (m[0], m[1], m[-1])]
        if len(pool_additive) != len(models_pool):
            models_pool = pool_additive
            # The full and the pure pools are amended silently
            if not set("PF") & set(model_str):
                _warn(
                    "Only additive models are allowed for your data. "
                    "Amending the pool.",
                    silent,
                )
    if model_str in ("PPP", "FFF", "YYY") or "Z" in model_str:
        error_type, models_pool = "A", None
        trend_type, season_type = (
            "X" if component in ("Y", "Z", "P", "F") else component
            for component in (trend_type, season_type)
        )
        _warn(
            "Only additive models are allowed for your data. "
            "Changing the selection mechanism.",
            silent,
        )
    elif "M" in (error_type, trend_type, season_type):
        _warn(
            f"Your data contains non-positive values, so the "
            f"ETS({model_str}) might break down.",
            silent,
        )
    return model_str, error_type, trend_type, season_type, models_pool


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
    model_str, error_type, trend_type, season_type, damped, model_do = (
        _parse_model_string(model_str, silent)
    )
    models_pool = None
    components = (error_type, trend_type, season_type)

    # The full, the pure or the safe pool, unless the models are combined, as R's
    named = next((letter for letter in "FPS" if letter in components), None)
    if named is not None:
        error_type, trend_type, season_type = (
            "Z" if component in "FPS" else component for component in components
        )
        if model_do != "combine":
            models_pool = _named_models_pool(named, *components)
            model_do = "select"
    # The letters of the selection; the others are estimated
    elif model_do != "combine" and any(c in ["Z", "X", "Y"] for c in components):
        model_do = "select"

    # Non-positive data: multiplicative components are not applicable
    if not allow_multiplicative:
        model_str, error_type, trend_type, season_type, models_pool = (
            _restrict_to_additive(
                model_str, error_type, trend_type, season_type, models_pool, silent
            )
        )

    # A seasonal model needs a seasonal lag. R warns and drops the seasonal
    # component when the maximum lag is 1 (R/adamGeneral.R, "unity lags").
    if season_type != "N" and max_lag <= 1:
        if season_type not in "ZXY":
            _warn(
                "Cannot build the seasonal model on data with the unity lags.\n"
                f"Switching to non-seasonal model: ETS({model_str[:-1]}N)",
                silent,
            )
        season_type = "N"
        model_str = model_str[:-1] + "N"
        if models_pool is not None:
            models_pool = [m for m in models_pool if m[-1] == "N"]

    # All the letters fixed by the data or the lags: that model, as R's
    fixed = error_type + trend_type + season_type
    if model_do != "estimate" and fixed[0] in "AM" and not set("ZXY") & set(fixed):
        models_pool = [error_type + trend_type + "d" * damped + season_type]

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


def _check_ets_model(
    model, distribution, data, silent=False, max_lag=1, occurrence_model=False
):
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
    occurrence_model : bool, default=False
        Whether an occurrence model is in use. Zeroes are expected there, so
        only negative values rule multiplicative components out.

    Returns
    -------
    dict
        Dictionary with ETS model information
    """
    # Determine if multiplicative models are allowed (R/adamGeneral.R):
    # zeroes only rule them out when there is no occurrence part.
    data_arr = np.asarray(data)
    allow_multiplicative = not np.any(
        data_arr < 0 if occurrence_model else data_arr <= 0
    )

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
        # Override models_pool with the user-provided list, without the seasonal
        # models when there is no seasonal lag, as R
        if model_info["season_type"] == "N":
            regular_pool = [m for m in regular_pool if m[-1] == "N"]
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
