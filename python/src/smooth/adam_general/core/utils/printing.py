"""
Printing utilities for ADAM models.

This module provides functions to generate formatted summaries of fitted ADAM models,
similar to R's print.adam() method.
"""

from typing import Any, Dict, Optional

import numpy as np

from smooth.adam_general.core.utils.ic import AIC, BIC, AICc, BICc


def _format_distribution(distribution: str, other: Optional[Dict] = None) -> str:
    """
    Format distribution name for display.

    Parameters
    ----------
    distribution : str
        Distribution code (e.g., 'dnorm', 'dgamma')
    other : Optional[Dict]
        Dictionary with additional parameters (shape, nu, etc.)

    Returns
    -------
    str
        Human-readable distribution name
    """
    other = other or {}

    dist_names = {
        "dnorm": "Normal",
        "dlaplace": "Laplace",
        "ds": "S",
        "dlogis": "Logistic",
        "dlnorm": "Log-Normal",
        "dllaplace": "Log-Laplace",
        "dls": "Log-S",
        "dinvgauss": "Inverse Gaussian",
        "dgamma": "Gamma",
    }

    if distribution == "dgnorm":
        shape = other.get("shape", "?")
        return (
            f"Generalised Normal with shape={shape:.4f}"
            if isinstance(shape, float)
            else f"Generalised Normal with shape={shape}"
        )
    elif distribution == "dt":
        nu = other.get("nu", "?")
        return (
            f"Student t with df={nu:.4f}"
            if isinstance(nu, float)
            else f"Student t with df={nu}"
        )
    elif distribution == "dalaplace":
        alpha = other.get("alpha", "?")
        return (
            f"Asymmetric Laplace with alpha={alpha:.4f}"
            if isinstance(alpha, float)
            else f"Asymmetric Laplace with alpha={alpha}"
        )
    elif distribution == "dlgnorm":
        shape = other.get("shape", "?")
        return (
            f"Log-Generalised Normal with shape={shape:.4f}"
            if isinstance(shape, float)
            else f"Log-Generalised Normal with shape={shape}"
        )

    return dist_names.get(distribution, distribution)


def _get_persistence_from_model(model: Any) -> Dict[str, Any]:
    """
    Extract persistence parameters from a fitted ADAM model.

    Parameters
    ----------
    model : ADAM
        Fitted ADAM model

    Returns
    -------
    Dict[str, Any]
        Dictionary with persistence values: alpha, beta, gamma (list), xreg
    """
    result = {"alpha": None, "beta": None, "gamma": [], "xreg": None}

    # Try to get from _prepared first (if predict was called)
    if hasattr(model, "_prepared") and model._prepared:
        vec_g = model._prepared.get("persistence")
        if vec_g is not None and len(vec_g) > 0:
            return _extract_persistence_from_vec_g(model, vec_g)

    # Otherwise, fill matrices to get vec_g
    if (
        hasattr(model, "_adam_estimated")
        and model._adam_estimated
        and hasattr(model, "_adam_created")
        and model._adam_created
    ):
        B = model._adam_estimated.get("B")
        if B is not None and len(B) > 0:
            try:
                from smooth.adam_general.core.creator import filler

                matrices_copy = {
                    "mat_vt": model._adam_created["mat_vt"].copy(),
                    "mat_wt": model._adam_created["mat_wt"].copy(),
                    "mat_f": model._adam_created["mat_f"].copy(),
                    "vec_g": model._adam_created["vec_g"].copy(),
                    "arima_polynomials": model._adam_created.get("arima_polynomials"),
                }

                filled = filler(
                    B=B,
                    model_type_dict=model._model_type,
                    components_dict=model._components,
                    lags_dict=model._lags_model,
                    matrices_dict=matrices_copy,
                    persistence_checked=model._persistence,
                    initials_checked=model._initials,
                    arima_checked=model._arima,
                    explanatory_checked=model._explanatory,
                    phi_dict=model._phi_internal,
                    constants_checked=model._constant,
                )

                vec_g = filled.get("vec_g")
                if vec_g is not None:
                    return _extract_persistence_from_vec_g(model, vec_g)
            except Exception:
                pass

    return result


def _extract_persistence_from_vec_g(model: Any, vec_g: np.ndarray) -> Dict[str, Any]:
    """
    Extract persistence values from vec_g array.

    Parameters
    ----------
    model : ADAM
        Fitted ADAM model
    vec_g : np.ndarray
        Persistence vector

    Returns
    -------
    Dict[str, Any]
        Dictionary with persistence values
    """
    result = {"alpha": None, "beta": None, "gamma": [], "xreg": None}

    if vec_g is None or len(vec_g) == 0:
        return result

    vec_g = np.array(vec_g).flatten()
    idx = 0

    model_type = model._model_type if hasattr(model, "_model_type") else {}
    components = model._components if hasattr(model, "_components") else {}

    n_ets = components.get("components_number_ets", 0)
    n_ets_seasonal = components.get("components_number_ets_seasonal", 0)

    # Alpha (level)
    if n_ets > 0 and idx < len(vec_g):
        result["alpha"] = vec_g[idx]
        idx += 1

    # Beta (trend)
    if model_type.get("model_is_trendy", False) and idx < len(vec_g):
        result["beta"] = vec_g[idx]
        idx += 1

    # Gamma (seasonal) - can be multiple
    if model_type.get("model_is_seasonal", False) and n_ets_seasonal > 0:
        for _ in range(n_ets_seasonal):
            if idx < len(vec_g):
                result["gamma"].append(vec_g[idx])
                idx += 1

    return result


def _format_persistence_vector(model: Any, digits: int = 4) -> str:
    """
    Format persistence vector for display.

    Parameters
    ----------
    model : ADAM
        Fitted ADAM model
    digits : int
        Number of decimal places

    Returns
    -------
    str
        Formatted persistence vector string
    """
    persistence = _get_persistence_from_model(model)

    names = []
    values = []

    # Level (alpha)
    if persistence["alpha"] is not None:
        names.append("alpha")
        values.append(persistence["alpha"])

    # Trend (beta)
    if persistence["beta"] is not None:
        names.append("beta")
        values.append(persistence["beta"])

    # Seasonal (gamma)
    if persistence["gamma"]:
        for i, val in enumerate(persistence["gamma"]):
            if len(persistence["gamma"]) > 1:
                names.append(f"gamma{i + 1}")
            else:
                names.append("gamma")
            values.append(val)

    if not names:
        return ""

    # Format header and values
    name_strs = [f"{n:>{max(len(n), digits + 2)}}" for n in names]
    value_strs = [
        f"{v:{max(len(names[i]), digits + 2)}.{digits}f}" for i, v in enumerate(values)
    ]

    header = " ".join(name_strs)
    vals = " ".join(value_strs)

    return f"{header}\n{vals}"


def _format_param_matrix(row_labels, col_labels, matrix, digits):
    """Format a parameter matrix (rows=orders, cols=lags) as aligned text like R."""
    col_w = max(digits + 7, max(len(c) for c in col_labels))
    row_w = max(len(r) for r in row_labels)
    # Header
    header = " " * row_w + "  " + "  ".join(c.rjust(col_w) for c in col_labels)
    lines = [header]
    for r_label, row in zip(row_labels, matrix):
        cells = []
        for val in row:
            if val is None:
                cells.append("NA".rjust(col_w))
            else:
                cells.append(f"{val:.{digits}f}".rjust(col_w))
        lines.append(r_label.ljust(row_w) + "  " + "  ".join(cells))
    return "\n".join(lines)


def _format_arma_parameters(model: Any, digits: int = 4) -> str:
    """Format ARMA parameters as R-style matrix (rows=orders, cols=lags)."""
    if not hasattr(model, "_arima") or not model._arima:
        return ""
    if not model._arima.get("arima_model", False):
        return ""

    arima = model._arima
    ar_orders = arima.get("ar_orders") or []
    ma_orders = arima.get("ma_orders") or []

    # Get lags from lags_original (the input lags, aligned with ar/ma orders)
    lags = []
    if hasattr(model, "_lags_model") and model._lags_model:
        lags = (
            model._lags_model.get("lags_original")
            or model._lags_model.get("lags")
            or []
        )
    if not lags and hasattr(model, "_config"):
        lags = model._config.get("lags") or [1]

    # Get polynomial values from _prepared
    # arma["ar"] = arPolynomial[1:], arma["ma"] = maPolynomial[1:]
    ar_poly = []
    ma_poly = []
    if hasattr(model, "_prepared") and model._prepared:
        arma = model._prepared.get("arma") or {}
        ar_poly = list(arma.get("ar") or [])
        ma_poly = list(arma.get("ma") or [])

    def _get_ar_val(order_j, lag):
        """AR(j) at lag: display = -arPolynomial[j*lag] = raw B value."""
        pos = order_j * lag - 1
        if pos < len(ar_poly):
            return -float(ar_poly[pos])
        return None

    def _get_ma_val(order_j, lag):
        """MA(j) at lag: display = maPolynomial[j*lag] = raw B value."""
        pos = order_j * lag - 1
        if pos < len(ma_poly):
            return float(ma_poly[pos])
        return None

    blocks = []

    # AR block
    if ar_poly and any(o > 0 for o in ar_orders):
        n = min(len(ar_orders), len(lags))
        active = [(k, lags[k]) for k in range(n) if ar_orders[k] > 0]
        if active:
            max_ar = max(ar_orders[k] for k, _ in active)
            col_labels = [f"Lag {lag}" for _, lag in active]
            row_labels = [f"AR({j + 1})" for j in range(max_ar)]
            matrix = [
                [
                    _get_ar_val(j + 1, lag) if j < ar_orders[k] else None
                    for k, lag in active
                ]
                for j in range(max_ar)
            ]
            blocks.append(_format_param_matrix(row_labels, col_labels, matrix, digits))

    # MA block
    if ma_poly and any(o > 0 for o in ma_orders):
        n = min(len(ma_orders), len(lags))
        active = [(k, lags[k]) for k in range(n) if ma_orders[k] > 0]
        if active:
            max_ma = max(ma_orders[k] for k, _ in active)
            col_labels = [f"Lag {lag}" for _, lag in active]
            row_labels = [f"MA({j + 1})" for j in range(max_ma)]
            matrix = [
                [
                    _get_ma_val(j + 1, lag) if j < ma_orders[k] else None
                    for k, lag in active
                ]
                for j in range(max_ma)
            ]
            blocks.append(_format_param_matrix(row_labels, col_labels, matrix, digits))

    return "\n".join(blocks)


def _format_information_criteria(
    log_lik_value: float, nobs: int, n_params: int, digits: int = 4
) -> str:
    """
    Format information criteria for display.

    Parameters
    ----------
    log_lik_value : float
        Log-likelihood value
    nobs : int
        Number of observations
    n_params : int
        Number of estimated parameters
    digits : int
        Number of decimal places

    Returns
    -------
    str
        Formatted information criteria string
    """
    aic = AIC(log_lik_value, nobs, n_params)
    aicc = AICc(log_lik_value, nobs, n_params)
    bic = BIC(log_lik_value, nobs, n_params)
    bicc = BICc(log_lik_value, nobs, n_params)

    # Format with alignment
    width = max(digits + 5, 8)
    header = f"{'AIC':>{width}} {'AICc':>{width}} {'BIC':>{width}} {'BICc':>{width}}"
    values = (
        f"{aic:{width}.{digits}f} "
        f"{aicc:{width}.{digits}f} "
        f"{bic:{width}.{digits}f} "
        f"{bicc:{width}.{digits}f}"
    )

    return f"{header}\n{values}"


def _compute_forecast_errors(
    y_holdout: np.ndarray,
    y_fitted_holdout: np.ndarray,
    y_in_sample: np.ndarray,
    period: int = 1,
) -> Dict[str, float]:
    """
    Compute forecast error metrics using greybox.measures().

    Parameters
    ----------
    y_holdout : np.ndarray
        Actual holdout values
    y_fitted_holdout : np.ndarray
        Forecasted values for holdout period
    y_in_sample : np.ndarray
        In-sample actual values (for scaling)
    period : int
        Unused; kept for API compatibility.

    Returns
    -------
    Dict[str, float]
        Dictionary of error metrics
    """
    from greybox.point_measures import measures

    m = measures(y_holdout, y_fitted_holdout, y_in_sample)
    m["RMSE"] = np.sqrt(m["MSE"])
    return m


def _format_forecast_errors(errors: Dict[str, float], digits: int = 3) -> str:
    """
    Format forecast errors for display.

    Parameters
    ----------
    errors : Dict[str, float]
        Dictionary of error metrics
    digits : int
        Number of decimal places

    Returns
    -------
    str
        Formatted error metrics string
    """
    lines = []

    # Line 1: ME, MAE, RMSE
    line1 = "; ".join(
        [
            f"ME: {errors['ME']:.{digits}f}",
            f"MAE: {errors['MAE']:.{digits}f}",
            f"RMSE: {errors['RMSE']:.{digits}f}",
        ]
    )
    lines.append(line1)

    # Line 2: sCE, Asymmetry, sMAE, sMSE (as percentages)
    line2 = "; ".join(
        [
            f"sCE: {errors['sCE'] * 100:.{digits}f}%",
            f"Asymmetry: {errors['asymmetry'] * 100:.1f}%",
            f"sMAE: {errors['sMAE'] * 100:.{digits}f}%",
            f"sMSE: {errors['sMSE'] * 100:.{digits}f}%",
        ]
    )
    lines.append(line2)

    # Line 3: MASE, RMSSE, rMAE, rRMSE
    line3 = "; ".join(
        [
            f"MASE: {errors['MASE']:.{digits}f}",
            f"RMSSE: {errors['RMSSE']:.{digits}f}",
            f"rMAE: {errors['rMAE']:.{digits}f}",
            f"rRMSE: {errors['rRMSE']:.{digits}f}",
        ]
    )
    lines.append(line3)

    return "\n".join(lines)


def model_summary_combined(model: Any, digits: int = 4) -> str:
    """
    Generate a formatted summary for a combined ADAM model.

    This function produces output for models fitted with combination
    (e.g., model="CCC"), showing the number of models combined and
    their average number of parameters.

    Parameters
    ----------
    model : ADAM
        Fitted combined ADAM model instance
    digits : int, default=4
        Number of decimal places for numeric output

    Returns
    -------
    str
        Formatted combined model summary string
    """
    lines = []

    # Time elapsed
    if hasattr(model, "time_elapsed_"):
        lines.append(f"Time elapsed: {model.time_elapsed_:.2f} seconds")

    # Model type - show original model spec (e.g., "CCC")
    model_name = _get_model_name(model)
    lines.append(f"Model estimated: {model_name}")
    lines.append("Loss function type: likelihood")
    lines.append("")

    # Number of models in the pool (all models stored, filtering at predict-time)
    n_models = len(model._prepared_models)
    lines.append(f"Number of models combined: {n_models}")

    # Sample size
    nobs = _get_nobs(model)
    lines.append(f"Sample size: {nobs}")

    # Average number of parameters (weighted)
    if hasattr(model, "_n_param") and model._n_param:
        avg_params = model._n_param.estimated["all"]
    else:
        avg_params = getattr(model, "_n_param_combined", 0)
    lines.append(f"Average number of estimated parameters: {avg_params:.{digits}f}")

    # Average degrees of freedom
    avg_df = nobs - avg_params
    lines.append(f"Average number of degrees of freedom: {avg_df:.{digits}f}")

    return "\n".join(lines)


def model_summary(model: Any, digits: int = 4) -> str:
    """
    Generate a formatted summary of a fitted ADAM model.

    This function produces output similar to R's print.adam() method.
    For combined models (model="CCC"), dispatches to model_summary_combined().

    Parameters
    ----------
    model : ADAM
        Fitted ADAM model instance
    digits : int, default=4
        Number of decimal places for numeric output

    Returns
    -------
    str
        Formatted model summary string

    Examples
    --------
    >>> from smooth import ADAM
    >>> model = ADAM(model="AAN")
    >>> model.fit(y)
    >>> print(model_summary(model))
    """
    # Dispatch to combined summary if this is a combined model
    if getattr(model, "_is_combined", False):
        return model_summary_combined(model, digits)

    lines = []

    # Time elapsed
    if hasattr(model, "time_elapsed_"):
        lines.append(f"Time elapsed: {model.time_elapsed_:.2f} seconds")

    # Model type
    model_name = _get_model_name(model)
    function_name = _get_function_name(model)
    lines.append(f"Model estimated using {function_name}() function: {model_name}")

    # Initialization type
    init_type = _get_initialization_type(model)
    lines.append(f"With {init_type} initialisation")

    # Distribution
    distribution = _get_distribution(model)
    dist_str = _format_distribution(distribution, getattr(model, "other", None))
    lines.append(f"Distribution assumed in the model: {dist_str}")

    # Loss function
    loss_str = _format_loss(model, digits)
    lines.append(loss_str)

    # Constant/Drift if present
    constant_str = _format_constant(model, digits)
    if constant_str:
        lines.append(constant_str)

    # Persistence vector
    if _is_ets_model(model):
        persistence_str = _format_persistence_vector(model, digits)
        if persistence_str:
            has_xreg = (
                hasattr(model, "_explanatory")
                and bool(model._explanatory)
                and model._explanatory.get("xreg_model", False)
            )
            g_label = (
                "Persistence vector g (excluding xreg):"
                if has_xreg
                else "Persistence vector g:"
            )
            lines.append(g_label)
            lines.append(persistence_str)

        # Damping parameter
        phi_str = _format_phi(model, digits)
        if phi_str:
            lines.append(phi_str)

    # ARMA parameters
    arma_str = _format_arma_parameters(model, digits)
    if arma_str:
        lines.append("ARMA parameters of the model:")
        lines.append(arma_str)

    # Sample size and parameters
    nobs = _get_nobs(model)
    n_params = _get_n_params(model)
    lines.append(f"Sample size: {nobs}")
    lines.append(f"Number of estimated parameters: {n_params}")
    lines.append(f"Number of degrees of freedom: {nobs - n_params}")

    # Information criteria
    if _can_compute_ic(model):
        log_lik = _get_log_likelihood(model)
        if log_lik is not None:
            lines.append("Information criteria:")
            lines.append(_format_information_criteria(log_lik, nobs, n_params, digits))
    else:
        lines.append(
            "Information criteria are unavailable for the chosen loss & distribution."
        )

    # Forecast errors (if holdout)
    errors_str = _format_holdout_errors(model, digits)
    if errors_str:
        lines.append("")
        lines.append("Forecast errors:")
        lines.append(errors_str)

    return "\n".join(lines)


def _build_model_name(model: Any) -> str:
    """Build the full R-style model name (e.g. 'SARIMA(1,1,2)[1](1,1,2)[12]')."""
    model_str = getattr(model, "model", "") or ""
    ets_model = False
    arima_model = False
    ar_orders = i_orders = ma_orders = []
    lags = [1]
    xreg_model = False
    n_ets_seasonal = 0
    constant_estimate = False
    regressors = "use"

    if hasattr(model, "_model_type") and model._model_type:
        ets_model = model._model_type.get("ets_model", False)
    if hasattr(model, "_arima") and model._arima:
        arima_model = model._arima.get("arima_model", False)
        ar_orders = model._arima.get("ar_orders") or []
        i_orders = model._arima.get("i_orders") or []
        ma_orders = model._arima.get("ma_orders") or []
    if hasattr(model, "_lags_model") and model._lags_model:
        lags = model._lags_model.get("lags", [1]) or [1]
    if hasattr(model, "_explanatory") and model._explanatory:
        xreg_model = model._explanatory.get("xreg_model", False)
    if hasattr(model, "_components") and model._components:
        n_ets_seasonal = model._components.get("components_number_ets_seasonal", 0)
    if hasattr(model, "_constant") and model._constant:
        constant_estimate = model._constant.get("constant_estimate", False)
    if hasattr(model, "_config"):
        regressors = model._config.get("regressors", "use")

    name = ""

    # ETS part
    if ets_model and model_str != "NNN":
        xstr = "X" if xreg_model else ""
        name = f"ETS{xstr}({model_str})"
        if n_ets_seasonal > 1:
            seasonal_lags = [str(lag) for lag in lags if lag != 1]
            name += f"[{', '.join(seasonal_lags)}]"

    # ARIMA part
    if arima_model:
        if ets_model:
            name += "+"
        # Non-seasonal: all lags == 1, or all seasonal-lag orders are zero
        seasonal_lags_idx = [k for k, lag in enumerate(lags) if lag > 1]
        is_nonseasonal = all(lag == 1 for lag in lags) or all(
            (k >= len(ar_orders) or ar_orders[k] == 0)
            and (k >= len(i_orders) or i_orders[k] == 0)
            and (k >= len(ma_orders) or ma_orders[k] == 0)
            for k in seasonal_lags_idx
        )
        xstr = "X" if (xreg_model and not ets_model) else ""
        if is_nonseasonal:
            p = ar_orders[0] if ar_orders else 0
            d = i_orders[0] if i_orders else 0
            q = ma_orders[0] if ma_orders else 0
            name += f"ARIMA{xstr}({p},{d},{q})"
        else:
            name += f"SARIMA{xstr}"
            for k, lag in enumerate(lags):
                p = ar_orders[k] if k < len(ar_orders) else 0
                d = i_orders[k] if k < len(i_orders) else 0
                q = ma_orders[k] if k < len(ma_orders) else 0
                if p == 0 and d == 0 and q == 0:
                    continue
                name += f"({p},{d},{q})[{lag}]"

    if not ets_model and not arima_model:
        if model_str == "NNN":
            if xreg_model:
                name = "Regression" if regressors != "adapt" else "Dynamic regression"
            else:
                name = "Constant level"
        elif xreg_model:
            name = "Regression" if regressors != "adapt" else "Dynamic regression"

    if (ets_model or arima_model) and constant_estimate:
        constant_name = "drift" if model_str != "NNN" else "constant"
        name += f" with {constant_name}"

    return name or model_str or "Unknown"


def _get_model_name(model: Any) -> str:
    """Get the model name string."""
    if hasattr(model, "model") and model.model:
        return model.model
    return "Unknown"


def _get_function_name(model: Any) -> str:
    """Get the function name used to estimate the model."""
    if model.__class__.__name__ == "ES":
        return "ES"
    return "ADAM"


def _get_initialization_type(model: Any) -> str:
    """Get the initialization type string."""
    if hasattr(model, "_initials") and model._initials:
        return model._initials.get("initial_type", "unknown")
    if hasattr(model, "_config"):
        initial = model._config.get("initial")
        if isinstance(initial, str):
            return initial
        return "provided"
    return "unknown"


def _get_distribution(model: Any) -> str:
    """Get the distribution code."""
    if hasattr(model, "_general") and model._general:
        return model._general.get(
            "distribution_new", model._general.get("distribution", "dnorm")
        )
    if hasattr(model, "_config"):
        dist = model._config.get("distribution")
        if dist:
            return dist
    return "dnorm"


def _format_loss(model: Any, digits: int) -> str:
    """Format loss function information."""
    loss = "likelihood"
    loss_value = None
    lambda_val = None

    if hasattr(model, "_general") and model._general:
        loss = model._general.get("loss", "likelihood")
        if loss in ["LASSO", "RIDGE"]:
            lambda_val = model._general.get("lambda")

    if hasattr(model, "_adam_estimated") and model._adam_estimated:
        if "CF_value" in model._adam_estimated:
            loss_value = model._adam_estimated["CF_value"]

    result = f"Loss function type: {loss}"
    if loss_value is not None:
        result += f"; Loss function value: {loss_value:.{digits}f}"
    if lambda_val is not None:
        result += f"; lambda= {lambda_val}"

    return result


def _format_constant(model: Any, digits: int) -> str:
    """Format constant/drift value if present."""
    if hasattr(model, "_constant") and model._constant:
        if model._constant.get("constant_required", False):
            constant_val = model.constant_value
            if constant_val is not None:
                return f"Intercept/Drift value: {constant_val:.{digits}f}"
    return ""


def _format_phi(model: Any, digits: int) -> str:
    """Format damping parameter if present."""
    if not hasattr(model, "_model_type") or not model._model_type:
        return ""

    if not model._model_type.get("damped", False):
        return ""

    # Skip if phi is not estimated (means model isn't actually damped)
    if hasattr(model, "_phi_internal") and model._phi_internal:
        if not model._phi_internal.get("phi_estimate", False):
            return ""

    phi_val = None

    if hasattr(model, "_phi_internal") and model._phi_internal:
        phi_val = model._phi_internal.get("phi")

    # Fallback to phi_ property
    if phi_val is None:
        try:
            phi_val = model.phi_
        except (ValueError, AttributeError):
            pass

    # Try _prepared
    if phi_val is None and hasattr(model, "_prepared") and model._prepared:
        phi_val = model._prepared.get("phi")

    if phi_val is not None:
        return f"Damping parameter: {phi_val:.{digits}f}"

    return ""


def _is_ets_model(model: Any) -> bool:
    """Check if model has ETS components."""
    if hasattr(model, "_model_type") and model._model_type:
        return model._model_type.get("ets_model", False)
    return False


def _get_nobs(model: Any) -> int:
    """Get number of observations."""
    if hasattr(model, "_observations") and model._observations:
        return model._observations.get("obs_in_sample", 0)
    return 0


def _get_n_params(model: Any) -> int:
    """Get number of estimated parameters (for degrees of freedom calculation)."""
    # Use n_param property via _general
    if hasattr(model, "_general") and model._general:
        n_param = model._general.get("n_param")
        if n_param:
            return n_param.n_param_estimated

        # Legacy format fallback
        params_number = model._general.get("parameters_number", [[0]])
        if params_number and len(params_number) > 0:
            return (
                params_number[0][0]
                if isinstance(params_number[0], list)
                else params_number[0]
            )

    # Fallback to _n_param_estimated
    if hasattr(model, "_n_param_estimated"):
        return model._n_param_estimated

    return 0


def _format_n_param_table(model: Any) -> str:
    """Format the n_param table for display."""
    n_param = None

    if hasattr(model, "_general") and model._general:
        n_param = model._general.get("n_param")

    if n_param is None:
        return ""

    return str(n_param)


def _can_compute_ic(model: Any) -> bool:
    """Check if information criteria can be computed."""
    loss = "likelihood"
    distribution = "dnorm"

    if hasattr(model, "_general") and model._general:
        loss = model._general.get("loss", "likelihood")
        distribution = model._general.get(
            "distribution_new", model._general.get("distribution", "dnorm")
        )

    # IC available for likelihood or matching loss/distribution combos
    if loss == "likelihood":
        return True
    if loss in ["MSE", "MSEh", "MSCE", "GPL"] and distribution == "dnorm":
        return True
    if loss in ["MAE", "MAEh", "MACE"] and distribution == "dlaplace":
        return True
    if loss in ["HAM", "HAMh", "CHAM"] and distribution == "ds":
        return True

    return False


def _get_log_likelihood(model: Any) -> Optional[float]:
    """Get log-likelihood value."""
    if hasattr(model, "_adam_estimated") and model._adam_estimated:
        log_lik_dict = model._adam_estimated.get("log_lik_adam_value", {})
        if isinstance(log_lik_dict, dict):
            return log_lik_dict.get("value")
        return log_lik_dict
    return None


def _format_holdout_errors(model: Any, digits: int) -> str:
    """Format holdout forecast errors if available."""
    if not hasattr(model, "_general") or not model._general:
        return ""

    if not model._general.get("holdout", False):
        return ""

    if not hasattr(model, "_observations") or not model._observations:
        return ""

    y_holdout = model._observations.get("y_holdout")
    y_in_sample = model._observations.get("y_in_sample")

    if y_holdout is None or len(y_holdout) == 0:
        return ""

    # Fast path: accuracy already computed by _auto_predict()
    if hasattr(model, "accuracy") and model.accuracy:
        return _format_forecast_errors(model.accuracy, digits)

    # Fallback: compute from forecast results (manual predict or auto-forecast)
    fc = None
    if hasattr(model, "_forecast_results") and model._forecast_results is not None:
        fc = model._forecast_results
    elif hasattr(model, "_auto_forecast") and model._auto_forecast is not None:
        fc = model._auto_forecast
    if fc is None:
        return ""

    y_forecast = np.asarray(fc.mean, dtype=float).ravel()
    if len(y_forecast) < len(y_holdout):
        return ""
    y_forecast = y_forecast[: len(y_holdout)]

    period = 1
    if hasattr(model, "_lags_model") and model._lags_model:
        lags = model._lags_model.get("lags", [1])
        period = max(lags) if lags else 1

    errors = _compute_forecast_errors(y_holdout, y_forecast, y_in_sample, period)
    return _format_forecast_errors(errors, digits)
