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
    }


def arima_states(ari_polynomial, non_zero_ari, x, error_type):
    """ARI states of ADAM ARIMA from its initials (R's ``adam_arimaStates()``).

    ADAM's ARI state with lag L is ``v_{L,t} = eta_L y_t`` (plus the MA term), so
    with zero errors before the sample all of them follow from the pre-sample
    values of the series. The initials ``x`` are these values in the convention
    of ``initial["arima"]``: ``-y`` (``1/y`` for "M"), oldest first, ``x[m-1]``
    at time 0, ``m`` being the largest ARI lag. The component with lag L reads
    its column c at time c-L, so it gets ``ari_L * x[m-L+c]``; its columns
    beyond L are cyclic repeats.

    Returns the states of the ARI rows in the order of ``non_zero_ari``, whose
    rows are ``[lag, state index]``.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    m = x.size
    if error_type == "M":
        x = np.log(x)
    columns = np.arange(m)
    non_zero_ari = np.atleast_2d(np.asarray(non_zero_ari, dtype=int))
    states = np.empty((non_zero_ari.shape[0], m))
    for i, lag in enumerate(non_zero_ari[:, 0]):
        states[i] = ari_polynomial[lag] * x[m - lag + (columns % lag)]
    return np.exp(states) if error_type == "M" else states


def arima_initial_row(non_zero_ari, components_number_arima):
    """Index within the ARIMA block of the state holding the initials.

    The ARI component with the largest lag, or the last ARIMA state for a pure
    MA (R's ``adam_arimaInitialRow()``, 0-based here).
    """
    non_zero_ari = np.atleast_2d(np.asarray(non_zero_ari, dtype=int))
    if non_zero_ari.size > 0:
        return int(non_zero_ari[np.argmax(non_zero_ari[:, 0]), 1])
    return components_number_arima - 1


def arima_pre_sample(
    y_in_sample, ot_logical, error_type, lags, i_orders, m, constant_level=None
):
    """Pre-sample values of the series for the ARIMA initials.

    R's ``adam_arimaPreSample()``: a lowess decomposition extended ``m``
    observations backwards (the level, the slope if the model has differences,
    and the seasonal patterns) on the scale of the ARIMA part. With no
    differences, the constant is the level, and the values are deviations from
    it. Returned in the convention of ``initial["arima"]`` (see
    :func:`arima_states`).
    """
    from smooth.adam_general.core.utils.utils import _mean_r, msdecompose

    y = np.asarray(y_in_sample, dtype=np.float64).ravel().copy()
    y[~np.asarray(ot_logical, dtype=bool).ravel()] = np.nan
    if error_type == "M":
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.log(y)
        y[~np.isfinite(y)] = np.nan
    lags = [int(lag) for lag in np.asarray(lags).ravel()]
    n_valid = int(np.sum(~np.isnan(y)))
    seasonal_lags = [lag for lag in lags if lag > 1 and lag * 2 < n_valid]
    decomposition = msdecompose(
        y, lags=seasonal_lags if seasonal_lags else [1], smoother="lowess"
    )
    trend = np.asarray(decomposition["states"])[:, 0]
    slope = 0.0
    if any(int(order) > 0 for order in np.asarray(i_orders).ravel()):
        head = min(trend.size, max(24, 2 * max(lags)))
        slope = _mean_r(np.diff(trend[:head]))
    times = np.arange(1 - m, 1)
    y_pre = trend[0] + slope * (times - 1)
    for i, lag in enumerate(seasonal_lags):
        pattern = np.asarray(decomposition["seasonal"][i])
        y_pre = y_pre + pattern[(times - 1) % lag]
    if constant_level is not None:
        y_pre = y_pre - (
            np.log(constant_level) if error_type == "M" else constant_level
        )
    return np.exp(-y_pre) if error_type == "M" else -y_pre


def arima_collect_initials(
    mat_vt,
    components_number_ets,
    components_number_arima,
    non_zero_ari,
    initial_arima_number,
    ari_polynomial,
    error_type,
):
    """ARIMA initials from the fitted states (R's ``adam_initial_collector()``).

    The state with the largest ARI lag holds ``ari_m`` times the initials, so
    they are its first ``initial_arima_number`` values divided by the last
    coefficient of the ARI polynomial (on logs for "M").
    """
    row = components_number_ets + arima_initial_row(
        non_zero_ari, components_number_arima
    )
    values = np.asarray(mat_vt[row, :initial_arima_number], dtype=np.float64)
    tail = None if ari_polynomial is None else float(np.asarray(ari_polynomial)[-1])
    if tail is None or tail == 0:
        return values
    if error_type == "M":
        return np.exp(np.log(values) / tail)
    return values / tail
