"""Admissible-region bounds for ADAM parameters.

Direct translations of R's ``eigenValues`` / ``eigenBounds`` /
``arimaParameterBounds`` (``R/adam.R``), used by ``confint`` and ``reapply`` to
clamp parameters to the region in which the model is stable
(``bounds="admissible"``) or stationary/invertible (ARIMA).
"""

import numpy as np

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
    ``any(smoothEigens(...) > 1 + 1e-10)``.
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
    return bool(np.any(np.asarray(eigenvalues) > 1 + 1e-10))


def eigen_bounds(vec_g, variable_index, **static_args):
    """Stability bounds for a single persistence parameter.

    Translation of R's ``eigenBounds`` (``R/adam.R:4341``): grid-search the value
    of ``vec_g[variable_index]`` from -5 upwards (step 0.01) for the lower bound
    and from 5 downwards for the upper bound, keeping the discount matrix stable.
    ``static_args`` are forwarded to :func:`eigen_values` (everything except
    ``vec_g``).
    """
    g = np.asarray(vec_g, dtype=float).copy()

    # Lower bound
    g[variable_index] = -5.0
    while eigen_values(g, **static_args):
        g[variable_index] += 0.01
        if g[variable_index] > 5:
            g[variable_index] = -5.0
            break
    lower_bound = g[variable_index] - 0.01

    # Upper bound
    g[variable_index] = 5.0
    while eigen_values(g, **static_args):
        g[variable_index] -= 0.01
        if g[variable_index] < -5:
            g[variable_index] = 5.0
            break
    upper_bound = g[variable_index] + 0.01

    return lower_bound, upper_bound


def arima_parameter_bounds(names, params, arima, lags):
    """Stationarity / invertibility bounds of the ARMA parameters among ``names``.

    Translation of R's ``arimaParameterBounds``: each parameter is bounded within
    its factor (the same type and lag), with the others at their values, by the
    shared C++ ``arimaParameterBounds`` (src/headers/arimaBounds.h). The fixed
    values come from ``arima["arma_parameters"]`` (all AR, then all MA). The
    parameters of factors that are not stationary / invertible are left out.

    Returns a dict of parameter name to (lower, upper).
    """
    from smooth.adam_general import _ols

    estimated = dict(zip(names, params))
    fixed = list(arima.get("arma_parameters") or [])
    factors: dict = {}
    position = 0
    for kind, orders, estimate in (
        ("phi", arima["ar_orders"], arima["ar_estimate"]),
        ("theta", arima["ma_orders"], arima["ma_estimate"]),
    ):
        for lag, order in zip(lags, orders):
            for j in range(int(order)):
                name = f"{kind}{j + 1}[{lag}]"
                value = estimated[name] if estimate else fixed[position]
                factors.setdefault((kind, lag), []).append((name, value))
                position += 1

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
