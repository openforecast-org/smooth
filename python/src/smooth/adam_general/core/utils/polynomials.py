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

    This function wraps the C++ polynomialise method exposed via pybind11.
    It mirrors R's call: adamCpp$polynomialise(B, arOrders, iOrders, maOrders,
                                               arEstimate, maEstimate, armaParameters,
                                               lags)

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
