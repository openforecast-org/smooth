"""Admissible-region bounds for ADAM parameters.

Direct translations of R's ``eigenValues`` / ``eigenBounds`` /
``arPolinomialsBounds`` (``R/adam.R:4332-4403``), used by ``confint`` to clamp
parameter confidence intervals to the region in which the model is stable
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


def ar_polynomial_bounds(ar_polynomial_matrix, ar_polynomial, variable_index):
    """Stationarity bounds for a single AR parameter.

    Translation of R's ``arPolinomialsBounds`` (``R/adam.R:4370``): grid-search
    the AR coefficient at ``variable_index`` so the AR companion matrix has all
    eigenvalue moduli <= 1, with a 20-step stopping criterion in each direction.
    """
    mat = np.array(ar_polynomial_matrix, dtype=float)
    ar = np.asarray(ar_polynomial, dtype=float).copy()
    stopping_criteria = 20

    def _roots_outside():
        mat[:, 0] = -ar[1:]
        try:
            eig = np.linalg.eigvals(mat)
        except np.linalg.LinAlgError:
            # ``Eigenvalues did not converge`` happens on numerically
            # pathological companion matrices that can appear at the
            # extreme ends of the grid search — treat them as outside
            # the stability region so the search advances toward the
            # admissible interior.
            return True
        return bool(np.any(np.abs(eig) > 1))

    # Lower bound
    ar[variable_index] = -5.0
    i = 1
    while _roots_outside():
        ar[variable_index] += 0.01
        i += 1
        if i >= stopping_criteria:
            break
    lower_bound = ar[variable_index] - 0.01

    # Upper bound
    ar[variable_index] = 5.0
    i = 1
    while _roots_outside():
        ar[variable_index] -= 0.01
        i += 1
        if i >= stopping_criteria:
            break
    upper_bound = ar[variable_index] + 0.01

    return lower_bound, upper_bound
