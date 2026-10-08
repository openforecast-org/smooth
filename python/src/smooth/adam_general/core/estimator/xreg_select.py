"""``regressors="select"`` in the estimator, as R's ``adam()``: the model is estimated
without the regressors, ``stepwise()`` selects them on its errors, and the model with
the selected ones is estimated from its parameters (R/adam.R, ``estimator()``)."""

import numpy as np

from smooth.adam_general.core.utils.cost_functions import fit_at_parameters
from smooth.adam_general.core.utils.utils import observed_mask, xreg_selector

from .optimization import _set_distribution

# The distributions of the ratios 1+e/mu, and those with a shape parameter
LOG_DISTRIBUTIONS = ("dlnorm", "dllaplace", "dls", "dlgnorm", "dinvgauss", "dgamma")
SHAPE_DISTRIBUTIONS = ("dalaplace", "dgnorm", "dlgnorm", "dt")
# Multiplicative errors with these distributions are regressed in logarithms
LEVEL_DISTRIBUTIONS = ("dnorm", "dlaplace", "ds", "dgnorm", "dlogis", "dt", "dalaplace")
# The losses of ALM for the multistep ones
ALM_LOSSES = {
    **dict.fromkeys(("MSEh", "TMSE", "GTMSE", "MSCE"), "MSE"),
    **dict.fromkeys(("MAEh", "TMAE", "GTMAE", "MACE"), "MAE"),
    **dict.fromkeys(("HAMh", "THAM", "GTHAM", "CHAM"), "HAM"),
}


def selection_errors(fitted, distribution, model_type_dict):
    """The errors of the model in the form of the distribution, for ``stepwise()``."""
    errors = np.ravel(fitted.errors).astype(float)
    e_type = model_type_dict["error_type"]
    if distribution in LOG_DISTRIBUTIONS and e_type == "A":
        errors = 1 + errors / np.ravel(fitted.fitted)
    errors = errors + (e_type == "M")
    types = (
        e_type,
        model_type_dict.get("trend_type", "N"),
        model_type_dict.get("season_type", "N"),
    )
    if distribution in LOG_DISTRIBUTIONS and "A" in types and np.any(errors <= 0):
        errors[errors <= 0] = 1e-100
    return errors


def selection_initials(y, X, names, general_dict, distribution, e_type, time=None):
    """The initial coefficients of the selected regressors: ALM with a trend (the
    time of the observations, ``time``), which is dropped, unless one of them is the
    trend."""
    from greybox import ALM

    obs = len(y)
    log_y = e_type == "M" and general_dict["distribution"] in LEVEL_DISTRIBUTIONS
    columns = [np.ones(obs), X]
    if "trend" not in names:
        columns.append(np.arange(1.0, obs + 1) if time is None else time)
    loss = ALM_LOSSES.get(general_dict["loss"], general_dict["loss"])
    alm = ALM(distribution=distribution, loss=loss)
    alm.fit(np.column_stack(columns), np.log(y) if log_y else y)
    return np.asarray(alm.coef, dtype=float)[: X.shape[1]]


def estimate_and_select(estimator, arguments):
    """The model without the regressors; with the regressors selected on its errors
    if there are any, estimated from its parameters with ``regressors="use"``. The
    result carries the explanatory dict of the model in ``explanatory_dict``."""
    explanatory = arguments["explanatory_dict"]
    select = explanatory["select"]
    without = {key: value for key, value in explanatory.items() if key != "select"}
    first = estimator(
        **{**arguments, "explanatory_dict": without, "return_matrices": True}
    )

    model_type = arguments["model_type_dict"]
    general = _set_distribution(arguments["general_dict"], model_type)
    distribution = general["distribution_new"]
    B = first["B"]
    other = arguments["other"]
    df = len(B) + 1
    if distribution in SHAPE_DISTRIBUTIONS and first["other_parameter_estimate"]:
        other = abs(B[-1])
        df -= 1
    fitted = fit_at_parameters(
        B,
        model_type,
        first["components_dict"],
        first["lags_dict"],
        first["matrices"],
        arguments["persistence_dict"],
        arguments["initials_dict"],
        arguments["arima_dict"],
        without,
        arguments["phi_dict"],
        arguments["constant_dict"],
        arguments["observations_dict"],
        general,
        first["profile_dict"],
        first["adam_cpp"],
        other,
    )
    obs = arguments["observations_dict"]["obs_in_sample"]
    # On the observed values only: the filled ones are not data
    observed = observed_mask(arguments["observations_dict"])
    names = list(select["names"])
    selected = xreg_selector(
        selection_errors(fitted, distribution, model_type)[observed],
        select["X"][:obs][observed],
        names,
        general["ic"],
        df,
        distribution,
        other,
    )
    if len(selected) == 0:
        return {**first, "explanatory_dict": without}

    columns = [names.index(name) for name in selected]
    X = select["X"][:, columns]
    initials = selection_initials(
        np.asarray(arguments["observations_dict"]["y_in_sample"], dtype=float)[
            observed
        ],
        X[:obs][observed],
        selected,
        general,
        distribution,
        model_type["error_type"],
        np.arange(1.0, obs + 1)[observed],
    )
    xreg = select["build"](
        X=X, xreg_names_from_input=selected, initial_xreg=initials.reshape(-1, 1)
    )
    xreg["xreg_columns"] = columns
    result = estimator(
        **{
            **arguments,
            "explanatory_dict": xreg,
            "model_type_dict": {**model_type, "xreg_model": True},
            "persistence_dict": {
                **arguments["persistence_dict"],
                "persistence_xreg_estimate": False,
            },
            "initials_dict": {
                **arguments["initials_dict"],
                "initial_xreg_estimate": True,
            },
            "B_initial": dict(zip(first["B_names"], B)),
            "lb": None,
            "ub": None,
        }
    )
    return {**result, "explanatory_dict": xreg}
