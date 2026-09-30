"""
ARIMA polynomial utilities for ADAM models.

This module provides the interface to the C++ polynomialise method for computing
ARIMA polynomial coefficients used in state-space representation.
"""

import numpy as np


def adam_polynomialiser(
    adam_cpp,
    B,
    ar_orders,
    i_orders,
    ma_orders,
    ar_estimate,
    ma_estimate,
    arma_parameters,
    lags,
):
    """
    Compute ARIMA polynomials using the C++ adamCore.polynomialise method.

    This function wraps the C++ ``polynomialise`` method exposed via
    pybind11. Given the optimisation parameter vector ``B`` and the
    AR/I/MA order specification, it builds the ARMA polynomials used by
    the state-space fitter.

    Parameters
    ----------
    adam_cpp : adamCore
        The C++ adamCore object (must be initialized before calling)
    B : array-like
        Parameter vector containing AR/MA coefficients to extract if estimating
    ar_orders : array-like
        AR orders for each lag (e.g., [1] for AR(1), [1, 1] for seasonal)
    i_orders : array-like
        Integration (differencing) orders for each lag
    ma_orders : array-like
        MA orders for each lag
    ar_estimate : bool
        Whether AR parameters should be extracted from B
    ma_estimate : bool
        Whether MA parameters should be extracted from B
    arma_parameters : array-like or None
        Fixed AR/MA parameters if not estimating, empty if estimating
    lags : array-like
        Lag values corresponding to each order (e.g., [1] for non-seasonal, [1, 12] for
        monthly)

    Returns
    -------
    dict
        Dictionary with polynomial arrays:
        - 'ar_polynomial': AR polynomial coefficients
        - 'i_polynomial': Integration polynomial coefficients
        - 'ari_polynomial': Combined ARI polynomial (AR * I)
        - 'ma_polynomial': MA polynomial coefficients
    """
    # Convert inputs to correct numpy array types for C++ binding
    B_arr = np.asarray(B, dtype=np.float64).flatten()
    ar_orders_arr = np.asarray(ar_orders, dtype=np.uint64).flatten()
    i_orders_arr = np.asarray(i_orders, dtype=np.uint64).flatten()
    ma_orders_arr = np.asarray(ma_orders, dtype=np.uint64).flatten()
    lags_arr = np.asarray(lags, dtype=np.uint64).flatten()

    # Handle arma_parameters - must be a float array for C++
    if arma_parameters is None or len(arma_parameters) == 0:
        arma_params_arr = np.array([], dtype=np.float64)
    else:
        arma_params_arr = np.asarray(arma_parameters, dtype=np.float64).flatten()

    # Call the C++ polynomialise method
    result = adam_cpp.polynomialise(
        B_arr,
        ar_orders_arr,
        i_orders_arr,
        ma_orders_arr,
        ar_estimate,
        ma_estimate,
        arma_params_arr,
        lags_arr,
    )

    # Convert C++ PolyResult struct to Python dict with numpy arrays
    # Flatten to 1D: C++ arma::vec may come back as (n,1) via pybind11
    return {
        "ar_polynomial": np.asarray(result.arPolynomial).flatten(),
        "i_polynomial": np.asarray(result.iPolynomial).flatten(),
        "ari_polynomial": np.asarray(result.ariPolynomial).flatten(),
        "ma_polynomial": np.asarray(result.maPolynomial).flatten(),
        "ar_reflection": float(result.arReflection),
        "ma_reflection": float(result.maReflection),
    }


def arima_bounds_penalty(arima_checked, arima_polynomials):
    """Penalty for a not stationary AR or not invertible MA part.

    R's ARIMA check in ``adam_bounds_checker()``: the largest reflection
    coefficient of the estimated factors (``src/headers/arimaBounds.h``, returned
    by :func:`adam_polynomialiser`), which is below one exactly when each AR factor
    is stationary and each MA one invertible.
    """
    if not arima_checked["arima_model"] or not (
        arima_checked["ar_estimate"] or arima_checked["ma_estimate"]
    ):
        return 0.0
    reflection = max(
        arima_checked["ar_estimate"] * arima_polynomials["ar_reflection"],
        arima_checked["ma_estimate"] * arima_polynomials["ma_reflection"],
    )
    return 1e100 * reflection if reflection >= 1 else 0.0


def arima_initials(ari_polynomial, x, error_type):
    """ARIMA initials from the pre-sample values (R's ``adam_arimaInitials()``).

    In the profile table, the column k of the ARIMA state with lag L is its value
    at time k-L, so the fitted value at time k (k <= m, the largest ARIMA lag)
    receives the column k of all the states with lags L >= k and nothing else
    from the head. Only these m sums
    are identified, which makes the ARIMA initials the same as the initial state
    of ssarima's companion form. They are held by the state with the largest lag,
    the last one, the other heads being zero (one for "M", where the sums are
    products).

    With zero errors before the sample, the ARI state with lag L is ``ari_L``
    times the pre-sample value ``x`` in the convention of
    :func:`arima_pre_sample` (``-y``, ``1/y`` for "M"; oldest first, ``x[m-1]``
    at time 0), so the initial k is the sum of ``ari_L * x[m-L+k]`` over the ARI
    lags L >= k.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    m = x.size
    if error_type == "M":
        x = np.log(x)
    ari_polynomial = np.asarray(ari_polynomial, dtype=np.float64).ravel()
    initials = np.zeros(m)
    for lag in np.flatnonzero(ari_polynomial[1:]) + 1:
        initials[:lag] += ari_polynomial[lag] * x[m - lag :]
    return np.exp(initials) if error_type == "M" else initials


def arima_head_initials(mat_vt_arima, lags_model_arima, lags_model_max, error_type):
    """ARIMA initials implied by the heads of the fitted ARIMA states.

    R's ``adam_arimaHeadInitials()``: the sums of :func:`arima_initials` for the
    states estimated in any way, e.g. by backcasting. The fitted states are
    aligned in time: the head column c (0-based) is the time c+1-lags_model_max,
    so the state with lag L gives to the time k its column lags_model_max-L+k-1.
    """
    lags_arima = np.asarray(lags_model_arima, dtype=int).ravel()
    head = np.asarray(mat_vt_arima, dtype=np.float64)[:, :lags_model_max]
    if error_type == "M":
        head = np.log(head)
    initials = np.array(
        [
            sum(
                head[i, lags_model_max - lag + k - 1]
                for i, lag in enumerate(lags_arima)
                if lag >= k
            )
            for k in range(1, int(lags_arima.max()) + 1)
        ]
    )
    return np.exp(initials) if error_type == "M" else initials


def arima_pre_sample(
    y_in_sample, ot_logical, error_type, lags, i_orders, m, constant_level, smoother
):
    """Pre-sample values of the series for the ARIMA initials.

    R's ``adam_arimaPreSample()``: the decomposition that gives the ETS initials
    (the same smoother), extended ``m`` observations backwards (the level, the
    trend if the model has differences, and the seasonal patterns) on the scale of
    the ARIMA part. ``msdecompose`` already extrapolates its initial level to the
    time 1-lags_max, over any gap of the smoother. With no differences, the
    constant is the level, and the values are deviations from it. Returned as
    ``-y`` (``1/y`` for "M"), oldest first (see :func:`arima_initials`).
    """
    from smooth.adam_general.core.utils.utils import msdecompose

    y = np.asarray(y_in_sample, dtype=np.float64).ravel().copy()
    y[~np.asarray(ot_logical, dtype=bool).ravel()] = np.nan
    if error_type == "M":
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.log(y)
        y[~np.isfinite(y)] = np.nan
    lags = [int(lag) for lag in np.asarray(lags).ravel()]
    n_valid = int(np.sum(~np.isnan(y)))
    seasonal_lags = [lag for lag in lags if lag > 1 and lag * 2 < n_valid]
    decomposition_lags = seasonal_lags if seasonal_lags else [1]
    decomposition = msdecompose(y, lags=decomposition_lags, smoother=smoother)
    level = float(decomposition["initial"]["nonseasonal"]["level"])
    trend = float(decomposition["initial"]["nonseasonal"]["trend"])
    slope = (
        trend if any(int(order) > 0 for order in np.asarray(i_orders).ravel()) else 0.0
    )
    times = np.arange(1 - m, 1)
    # The level at the time 1, then the slope from there
    y_pre = level + trend * max(decomposition_lags) + slope * (times - 1)
    for i, lag in enumerate(seasonal_lags):
        pattern = np.asarray(decomposition["seasonal"][i])
        y_pre = y_pre + pattern[(times - 1) % lag]
    if constant_level is not None and not np.any(np.asarray(i_orders) > 0):
        y_pre = y_pre - (
            np.log(constant_level) if error_type == "M" else constant_level
        )
    return np.exp(-y_pre) if error_type == "M" else -y_pre
