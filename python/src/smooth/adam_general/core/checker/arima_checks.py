import numpy as np


def _expand_orders(orders):
    """
    Expand ARIMA orders into AR, I, MA components.

    Parameters
    ----------
    orders : list, tuple, int, dict, or None
        ARIMA order specification. Can be:
        - dict with 'ar', 'i', 'ma' keys
        - list/tuple of [ar, i, ma] values
        - single int (interpreted as AR order)

    Returns
    -------
    tuple
        (ar_orders, i_orders, ma_orders)
    """
    # Default values
    ar_orders = i_orders = ma_orders = [0]

    if orders is None:
        return ar_orders, i_orders, ma_orders

    # Handle dict input (from ADAM class)
    if isinstance(orders, dict):
        ar = orders.get("ar", 0)
        i = orders.get("i", 0)
        ma = orders.get("ma", 0)
        ar_orders = [ar] if isinstance(ar, (int, float)) else list(ar) if ar else [0]
        i_orders = [i] if isinstance(i, (int, float)) else list(i) if i else [0]
        ma_orders = [ma] if isinstance(ma, (int, float)) else list(ma) if ma else [0]
    # Handle list/tuple input
    elif isinstance(orders, (list, tuple)):
        if len(orders) >= 3:
            ar_orders = (
                [orders[0]] if isinstance(orders[0], (int, float)) else orders[0]
            )
            i_orders = [orders[1]] if isinstance(orders[1], (int, float)) else orders[1]
            ma_orders = (
                [orders[2]] if isinstance(orders[2], (int, float)) else orders[2]
            )
    elif isinstance(orders, (int, float)):
        # Single value -> assume AR order
        ar_orders = [orders]

    return ar_orders, i_orders, ma_orders


def _get_polynomial_indices_from_cpp(ar_orders, i_orders, ma_orders, lags):
    """
    Use C++ polynomialise to get correct polynomial indices matching R's algorithm.

    This function calls the C++ polynomialise with dummy parameters to extract
    the correct polynomial structure (which indices are non-zero) matching exactly
    what R's implementation produces.
    """
    try:
        from smooth.adam_general._adamCore import adamCore
    except ImportError:
        return None

    try:
        ar_orders_arr = np.array(ar_orders, dtype=np.uint64)
        i_orders_arr = np.array(i_orders, dtype=np.uint64)
        ma_orders_arr = np.array(ma_orders, dtype=np.uint64)
        lags_arr = np.array(lags, dtype=np.uint64)

        n_ar = int(np.sum(ar_orders_arr))
        n_ma = int(np.sum(ma_orders_arr))
        n_arma = n_ar + n_ma

        if n_arma == 0:
            return None

        # Use distinct values to prevent polynomial term cancellation
        # (e.g. ARIMA(2,1,0) with equal phi1=phi2 makes the B² term vanish)
        dummy_B = np.arange(1, n_arma + 1, dtype=np.float64) * 0.1

        # Create minimal adamCore instance for polynomialise
        # max_lag = max(lags_arr) if len(lags_arr) > 0 else 1
        dummy_lags = np.array([1], dtype=np.uint64)
        adam_cpp = adamCore(
            dummy_lags,  # lags
            "N",  # E
            "N",  # T
            "N",  # S
            0,  # nNonSeasonal
            0,  # nSeasonal
            0,  # nETS
            n_arma,  # nArima
            0,  # nXreg
            n_arma,  # nComponents
            False,  # constant
            False,  # adamETS
        )

        result = adam_cpp.polynomialise(
            dummy_B,
            ar_orders_arr,
            i_orders_arr,
            ma_orders_arr,
            n_ar > 0,
            n_ma > 0,
            np.array([], dtype=np.float64),
            lags_arr,
        )

        ari_poly = np.asarray(result.ariPolynomial).flatten()
        ma_poly = np.asarray(result.maPolynomial).flatten()

        # Use actual polynomial values (with non-zero params) to find structural indices
        # This correctly captures cross-terms from polynomial multiplication
        ari_indices = list(np.where(np.abs(ari_poly[1:]) > 1e-10)[0] + 1)
        ma_indices = list(np.where(np.abs(ma_poly[1:]) > 1e-10)[0] + 1)

        if len(ari_indices) == 0 and len(ma_indices) == 0:
            return None

        # Keep polynomial positions as direct numpy indices:
        # coefficient for B^k is stored at index k.
        all_lag_values = sorted(set(list(ari_indices) + list(ma_indices)))
        lags_model_arima = all_lag_values

        components_number_arima = len(lags_model_arima)
        initial_arima_number = max(lags_model_arima) if lags_model_arima else 0

        # Map lag/polynomial position -> ARIMA state index.
        ari_state_map = {idx: i for i, idx in enumerate(lags_model_arima)}
        ma_state_map = {idx: i for i, idx in enumerate(lags_model_arima)}

        # Handle empty indices case - must create 2D array with shape (0, 2)
        if len(ari_indices) > 0:
            non_zero_ari = np.array(
                [[idx, ari_state_map[idx]] for idx in ari_indices], dtype=int
            )
        else:
            non_zero_ari = np.zeros((0, 2), dtype=int)

        if len(ma_indices) > 0:
            non_zero_ma = np.array(
                [[idx, ma_state_map[idx]] for idx in ma_indices], dtype=int
            )
        else:
            non_zero_ma = np.zeros((0, 2), dtype=int)

        return (
            non_zero_ari,
            non_zero_ma,
            lags_model_arima,
            components_number_arima,
            initial_arima_number,
        )

    except Exception:
        # Return None on any error to fall back to Python algorithm
        return None


def _check_arima(orders, validated_lags, silent=False, arma=None):
    """
    Check and validate ARIMA model specification.

    This function mirrors R's parametersChecker ARIMA handling in adamGeneral.R lines
    519-666.

    Parameters
    ----------
    orders : list, tuple, int, or None
        ARIMA order specification
    validated_lags : list
        List of validated lags (must include 1 as first element for non-seasonal)
    silent : bool, optional
        Whether to suppress warnings

    Returns
    -------
    dict
        Dictionary with ARIMA model information including:
        - non_zero_ari: Nx2 matrix [polynomial_index, state_index] (0-indexed for
        Python)
        - non_zero_ma: Nx2 matrix [polynomial_index, state_index] (0-indexed for Python)
        - lags_model_arima: list of lag values for ARIMA states
        - components_number_arima: number of ARIMA state components
    """

    # Initialize with default values - R lines 652-666
    arima_result = {
        "arima_model": False,
        "ar_orders": None,
        "i_orders": None,
        "ma_orders": None,
        "ar_required": False,
        "i_required": False,
        "ma_required": False,
        "ar_estimate": False,
        "ma_estimate": False,
        "arma_parameters": None,
        "lags_model_arima": [],
        "non_zero_ari": np.zeros((0, 2), dtype=int),
        "non_zero_ma": np.zeros((0, 2), dtype=int),
        "components_number_arima": 0,
        "components_names_arima": [],
        "initial_arima_number": 0,
        "select": False,
    }

    # If no orders specified, return default values
    if orders is None:
        return arima_result

    # Parse orders into components - R lines 521-538
    ar_orders, i_orders, ma_orders = _expand_orders(orders)

    # Check for valid ARIMA component - R lines 541-542
    if (sum(ar_orders) + sum(i_orders) + sum(ma_orders)) == 0:
        return arima_result

    arima_result["arima_model"] = True

    # See if AR/I/MA is needed - R lines 544-560
    ar_required = sum(ar_orders) > 0
    i_required = sum(i_orders) > 0
    ma_required = sum(ma_orders) > 0

    arima_result["ar_required"] = ar_required
    arima_result["i_required"] = i_required
    arima_result["ma_required"] = ma_required

    # Ensure lags start with 1
    lags = list(validated_lags) if validated_lags else [1]
    if lags[0] != 1:
        lags = [1] + lags

    # Define maxOrder and align orders with lags - R lines 563-578
    max_order = max(len(ar_orders), len(i_orders), len(ma_orders), len(lags))

    # Pad orders with zeros to match max_order
    ar_orders = list(ar_orders) + [0] * (max_order - len(ar_orders))
    i_orders = list(i_orders) + [0] * (max_order - len(i_orders))
    ma_orders = list(ma_orders) + [0] * (max_order - len(ma_orders))

    # If lags shorter than max_order, filter orders by non-zero lags - R lines 573-578
    if len(lags) < max_order:
        lags_new = list(lags) + [0] * (max_order - len(lags))
        # Filter to keep only orders where lags are non-zero
        ar_orders = [ar_orders[i] for i in range(len(lags_new)) if lags_new[i] != 0]
        i_orders = [i_orders[i] for i in range(len(lags_new)) if lags_new[i] != 0]
        ma_orders = [ma_orders[i] for i in range(len(lags_new)) if lags_new[i] != 0]
    else:
        # Make sure lags matches the length
        lags = list(lags[:max_order])

    # If after filtering all orders are zero, return no ARIMA - R lines 632-646
    if all(o == 0 for o in ar_orders + i_orders + ma_orders):
        return arima_result

    arima_result["ar_orders"] = ar_orders
    arima_result["i_orders"] = i_orders
    arima_result["ma_orders"] = ma_orders

    cpp_result = _get_polynomial_indices_from_cpp(ar_orders, i_orders, ma_orders, lags)

    if cpp_result is not None:
        (
            non_zero_ari,
            non_zero_ma,
            lags_model_arima,
            components_number_arima,
            initial_arima_number,
        ) = cpp_result
        # Component names - R lines 629-630
        if components_number_arima > 1:
            components_names_arima = [
                f"ARIMAState{i + 1}" for i in range(components_number_arima)
            ]
        else:
            components_names_arima = (
                ["ARIMAState1"] if components_number_arima > 0 else []
            )
    else:
        # Fall back to Python algorithm for simple cases or when C++ fails
        # Define the non-zero values via polynomial computation - R lines 580-616
        ari_values = []
        ma_values = []

        def _unique_preserve_order(values):
            seen = set()
            out = []
            for value in values:
                if value not in seen:
                    seen.add(value)
                    out.append(value)
            return out

        for i in range(len(lags)):
            ari_for_lag = [0]
            if ar_orders[i] > 0:
                ari_for_lag.extend(range(1, ar_orders[i] + 1))
            if i_orders[i] > 0:
                ari_for_lag.extend(
                    range(ar_orders[i] + 1, ar_orders[i] + i_orders[i] + 1)
                )
            ari_for_lag = _unique_preserve_order([v * lags[i] for v in ari_for_lag])
            ari_values.append(ari_for_lag)

            ma_for_lag = [0]
            if ma_orders[i] > 0:
                ma_for_lag.extend(range(1, ma_orders[i] + 1))
            ma_for_lag = _unique_preserve_order([v * lags[i] for v in ma_for_lag])
            ma_values.append(ma_for_lag)

        def expand_polynomial(values_list):
            if len(values_list) == 0:
                return [0]
            dims = [len(values) for values in values_list]
            result = np.zeros(dims, dtype=int)
            for axis, values in enumerate(values_list):
                shape = [1] * len(values_list)
                shape[axis] = len(values)
                result = result + np.asarray(values, dtype=int).reshape(shape)
            return result.flatten(order="F").tolist()

        ari_polynomial = expand_polynomial(ari_values)
        ma_polynomial = expand_polynomial(ma_values)

        def _unique_positive_in_order(values):
            seen = set()
            out = []
            for value in values:
                if value > 0 and value not in seen:
                    seen.add(value)
                    out.append(value)
            return out

        non_zero_ari_lags = _unique_positive_in_order(ari_polynomial)
        non_zero_ma_lags = _unique_positive_in_order(ma_polynomial)

        lags_model_arima = sorted(set(non_zero_ari_lags + non_zero_ma_lags))

        if len(lags_model_arima) == 0:
            arima_result["arima_model"] = False
            return arima_result

        ari_state_map = {lag: idx for idx, lag in enumerate(lags_model_arima)}
        ma_state_map = ari_state_map

        non_zero_ari = [[lag, ari_state_map[lag]] for lag in non_zero_ari_lags]
        non_zero_ma = [[lag, ma_state_map[lag]] for lag in non_zero_ma_lags]

        non_zero_ari = (
            np.array(non_zero_ari, dtype=int)
            if non_zero_ari
            else np.zeros((0, 2), dtype=int)
        )
        non_zero_ma = (
            np.array(non_zero_ma, dtype=int)
            if non_zero_ma
            else np.zeros((0, 2), dtype=int)
        )

        components_number_arima = len(lags_model_arima)
        initial_arima_number = max(lags_model_arima) if lags_model_arima else 0
        # Component names
        if components_number_arima > 1:
            components_names_arima = [
                f"ARIMAState{i + 1}" for i in range(components_number_arima)
            ]
        else:
            components_names_arima = (
                ["ARIMAState1"] if components_number_arima > 0 else []
            )

    # Update result
    arima_result["non_zero_ari"] = non_zero_ari
    arima_result["non_zero_ma"] = non_zero_ma
    arima_result["lags_model_arima"] = lags_model_arima
    arima_result["components_number_arima"] = components_number_arima
    arima_result["components_names_arima"] = components_names_arima
    arima_result["initial_arima_number"] = initial_arima_number

    # Set estimation flags - always estimate parameters if they are required
    arima_result["ar_estimate"] = ar_required
    arima_result["ma_estimate"] = ma_required

    # Initialize ARMA parameters (will be filled during estimation)
    arima_parameters = []
    if ar_required:
        for i in range(len(ar_orders)):
            for j in range(ar_orders[i]):
                arima_parameters.append(0.0)
    if ma_required:
        for i in range(len(ma_orders)):
            for j in range(ma_orders[i]):
                arima_parameters.append(0.0)

    arima_result["arma_parameters"] = arima_parameters if arima_parameters else None

    # Apply fixed ARMA values from `arma` dict
    if arma is not None and isinstance(arma, dict):
        ar_fixed = arma.get("ar", None)
        ma_fixed = arma.get("ma", None)
        n_ar = sum(ar_orders)
        n_ma = sum(ma_orders)

        if ar_fixed is not None and ar_required:
            ar_fixed = (
                [ar_fixed] if isinstance(ar_fixed, (int, float)) else list(ar_fixed)
            )
            arima_result["ar_estimate"] = False
            arima_parameters[:n_ar] = ar_fixed[:n_ar]

        if ma_fixed is not None and ma_required:
            ma_fixed = (
                [ma_fixed] if isinstance(ma_fixed, (int, float)) else list(ma_fixed)
            )
            arima_result["ma_estimate"] = False
            arima_parameters[n_ar : n_ar + n_ma] = ma_fixed[:n_ma]

        arima_result["arma_parameters"] = arima_parameters if arima_parameters else None

    return arima_result
