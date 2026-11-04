"""Admissible-region bounds for ADAM parameters.

Direct translations of R's ``eigenValues`` / ``eigenBounds`` /
``arimaParameterBounds`` (``R/adam.R``), used by ``confint`` and ``reapply`` to
clamp parameters to the region in which the model is stable
(``bounds="admissible"``) or stationary/invertible (ARIMA).
"""

import numpy as np

from smooth.adam_general import _ols  # type: ignore[attr-defined]
from smooth.adam_general._eigenCalc import smooth_eigens


def eigen_values(
    vec_g,
    transition,
    measurement,
    lags_model_all,
    xreg_model,
    obs_in_sample,
    has_delta=False,
    xreg_number=0,
    constant_required=False,
):
    """Whether the discount matrix has an eigenvalue outside the unit circle.

    Mirrors R's ``eigenValues`` (``R/adam.R:4332``):
    ``any(smoothEigens(...) > 1 - 1e-10)``.
    """
    eigenvalues = smooth_eigens(
        persistence=np.asfortranarray(
            np.asarray(vec_g, dtype=np.float64).reshape(-1, 1)
        ),
        transition=np.asfortranarray(transition, dtype=np.float64),
        measurement=np.asfortranarray(measurement, dtype=np.float64),
        lags_model_all=np.asarray(lags_model_all, dtype=np.int32),
        xreg_model=bool(xreg_model),
        obs_in_sample=int(obs_in_sample),
        has_delta=bool(has_delta),
        xreg_number=int(xreg_number),
        constant_required=bool(constant_required),
    )
    return bool(np.any(np.asarray(eigenvalues) > 1 - 1e-10))


def eigen_bounds(vec_g, variable_index, **static_args):
    """Stability bounds for a single persistence parameter.

    Translation of R's ``eigenBounds``: the interval around the value of
    ``vec_g[variable_index]`` on which the model stays forecastable, found by the
    search of the ARMA bounds (``stability_interval``, ``arimaBounds.h``) within
    [-5, 5]; ``(-inf, inf)`` when it is not forecastable at its value.
    ``static_args`` are forwarded to :func:`eigen_values` (everything except
    ``vec_g``).
    """
    g = np.asarray(vec_g, dtype=float).copy()

    def stable(value):
        g[variable_index] = value
        return not eigen_values(g, **static_args)

    lower, upper = _ols.stability_interval(
        stable, float(vec_g[variable_index]), 0.01, 5.0
    )
    if np.isnan(lower) or np.isnan(upper):
        return -np.inf, np.inf
    return float(lower), float(upper)


def ces_bounds(params, names, variable_index, vec_g, **static_args):
    """Stability bounds for a single smoothing parameter of CES.

    Translation of R's ``cesBounds``: the interval around its value on which the
    model stays forecastable, with the other parameters at their values, found by
    the same search as :func:`eigen_bounds`. As there, but each parameter moves
    both the transition and the persistence, so both are rebuilt from ``params``:
    each complex (alpha_0, alpha_1) and (beta_0, beta_1) gives a pair of states,
    after the level and potential for beta, and the real beta of the partial
    seasonality one. ``static_args`` are those of :func:`eigen_values`.
    """
    params = np.asarray(params, dtype=float).copy()
    real = [i for i, n in enumerate(names) if n.startswith(("alpha_0", "beta_0"))]
    imaginary = [i for i, n in enumerate(names) if n.startswith(("alpha_1", "beta_1"))]
    partial = [i for i, n in enumerate(names) if n == "beta" or n.startswith("beta[")]
    shift = 0 if any(n.startswith("alpha") for n in names) else 2
    rows = 2 * np.arange(len(real)) + shift
    transition = np.array(static_args.pop("transition"), dtype=float)
    g = np.asarray(vec_g, dtype=float).copy()

    def stable(value):
        params[variable_index] = value
        transition[rows, rows + 1] = params[imaginary] - 1
        transition[rows + 1, rows + 1] = 1 - params[real]
        g[rows] = params[real] - params[imaginary]
        g[rows + 1] = params[real] + params[imaginary]
        g[2 : 2 + len(partial)] = params[partial]
        return not eigen_values(g, transition, **static_args)

    lower, upper = _ols.stability_interval(
        stable, float(params[variable_index]), 0.01, 5.0
    )
    if np.isnan(lower) or np.isnan(upper):
        return -np.inf, np.inf
    return float(lower), float(upper)


def arima_parameter_bounds(names, params, arima, lags):
    """Stationarity / invertibility bounds of the ARMA parameters among ``names``.

    Translation of R's ``arimaParameterBounds``: each parameter is bounded within
    its factor (the same type and lag), with the others at their values, by the
    shared C++ ``arimaParameterBounds`` (src/headers/arimaBounds.h). The fixed
    values come from ``arima["arma_parameters"]`` (lag by lag, AR then MA). The
    parameters of factors that are not stationary / invertible are left out.

    Returns a dict of parameter name to (lower, upper).
    """
    estimated = dict(zip(names, params))
    fixed = iter(arima.get("arma_parameters") or [])
    factors: dict = {}
    for lag, ar_order, ma_order in zip(lags, arima["ar_orders"], arima["ma_orders"]):
        for kind, order, estimate in (
            ("phi", ar_order, arima["ar_estimate"]),
            ("theta", ma_order, arima["ma_estimate"]),
        ):
            for j in range(int(order)):
                name = f"{kind}{j + 1}[{lag}]"
                value = estimated[name] if estimate else next(fixed)
                factors.setdefault((kind, lag), []).append((name, value))

    bounds = {}
    for (kind, _), members in factors.items():
        values = np.array([value for _, value in members], dtype=float)
        for j, (name, _) in enumerate(members):
            if name not in estimated:
                continue
            lower, upper = _ols.arima_parameter_bounds(
                values, j, -1.0 if kind == "phi" else 1.0
            )
            if np.isfinite(lower) and np.isfinite(upper):
                bounds[name] = (float(lower), float(upper))
    return bounds
