"""
CES-specific cost function for optimization.

Translates R/adam-ces.R CF() function (lines 486-563).
Wraps ces_filler + adamCore.fit() + loss computation.
"""

import math

import numpy as np

from smooth.adam_general._eigenCalc import smooth_eigens
from smooth.adam_general.core.ces.filler import ces_filler

_R_LN_SQRT_2PI = 0.918938533204672741780329736406


def ces_cf(
    B,
    # Matrices (originals — will be copied)
    mat_vt,
    mat_wt,
    mat_f,
    vec_g,
    # CES parameter dicts
    a,
    b,
    # CES config
    seasonality,
    n_seasonal,
    lags_model_seasonal,
    lags_model_max,
    initial_type,
    xreg_model,
    xreg_number,
    initial_xreg_estimate,
    components_number,
    # ADAM profile structures
    lags_model_all,
    index_lookup_table,
    profiles_recent_table,
    # Data
    y_in_sample,
    ot,
    ot_logical,
    obs_in_sample,
    n_iterations,
    # Config
    bounds,
    loss,
    h,
    multisteps,
    adam_cpp,
    initial_value=None,
):
    """
    CES cost function evaluated during optimization.

    Parameters
    ----------
    B : np.ndarray
        Current parameter vector from optimizer.
    mat_vt, mat_wt, mat_f, vec_g : np.ndarray
        Original state-space matrices (copied before modification).
    a, b : dict
        CES smoothing parameter specifications.
    seasonality : str
        Seasonality type.
    bounds : str
        "admissible" or "none".
    loss : str
        Loss function name.
    h : int
        Forecast horizon (for multistep losses).
    multisteps : bool
        Whether using multistep loss.
    adam_cpp : adamCore
        C++ adamCore instance.

    Returns
    -------
    float
        Cost function value.
    """
    # Copy matrices to avoid in-place mutation across calls
    mat_f_copy = mat_f.copy()
    vec_g_copy = vec_g.copy()
    mat_vt_copy = mat_vt.copy()

    # Fill matrices with current B values — R line 488
    elements = ces_filler(
        B=B,
        mat_vt=mat_vt_copy,
        mat_f=mat_f_copy,
        vec_g=vec_g_copy,
        a=a,
        b=b,
        seasonality=seasonality,
        n_seasonal=n_seasonal,
        lags_model_seasonal=lags_model_seasonal,
        lags_model_max=lags_model_max,
        initial_type=initial_type,
        xreg_model=xreg_model,
        xreg_number=xreg_number,
        initial_xreg_estimate=initial_xreg_estimate,
        components_number=components_number,
        initial_value=initial_value,
    )

    # Admissible bounds check — R lines 490-497
    if bounds == "admissible":
        eigen_values = smooth_eigens(
            persistence=np.asfortranarray(
                elements["vec_g"].reshape(-1, 1), dtype=np.float64
            ),
            transition=np.asfortranarray(elements["mat_f"], dtype=np.float64),
            measurement=np.asfortranarray(mat_wt, dtype=np.float64),
            lags_model_all=np.asarray(lags_model_all, dtype=np.int32),
            xreg_model=xreg_model,
            obs_in_sample=obs_in_sample,
            has_delta=False,
            xreg_number=xreg_number,
            constant_required=False,
        )
        if np.any(eigen_values > 1 + 1e-50):
            return 1e100 * np.max(eigen_values)

    # Update mat_vt initial columns and profiles — R lines 499-501
    mat_vt_copy[:, :lags_model_max] = elements["vt"]
    profiles_recent_copy = profiles_recent_table.copy()
    profiles_recent_copy[:] = elements["vt"]

    # Ensure Fortran order for C++
    mat_vt_f = np.asfortranarray(mat_vt_copy, dtype=np.float64)
    mat_wt_f = np.asfortranarray(mat_wt, dtype=np.float64)
    mat_f_f = np.asfortranarray(elements["mat_f"], dtype=np.float64)
    vec_g_f = np.asfortranarray(elements["vec_g"].ravel(), dtype=np.float64)
    ilt_f = np.asfortranarray(index_lookup_table, dtype=np.uint64)
    prt_f = np.asfortranarray(profiles_recent_copy, dtype=np.float64)
    y_f = np.asfortranarray(y_in_sample, dtype=np.float64).ravel()
    ot_f = np.asfortranarray(ot, dtype=np.float64).ravel()

    backcast = initial_type in ("complete", "backcasting", "gradient")

    # Additive SSOE: initial="gradient" profiles the initials by least squares
    # (the CES state-space is linear/additive); otherwise the ordinary fit.
    from smooth.adam_general.core.utils.gradient import adam_fit_or_gradient

    adam_fitted = adam_fit_or_gradient(
        adam_cpp=adam_cpp,
        mat_vt=mat_vt_f,
        mat_wt=mat_wt_f,
        mat_f=mat_f_f,
        vec_g=vec_g_f,
        index_lookup_table=ilt_f,
        profiles_recent_table=prt_f,
        y_in_sample=y_f,
        ot=ot_f,
        initial_type=initial_type,
        n_iterations=int(n_iterations),
        backcast_value=backcast,
        model_type_dict={
            "ets_model": False,
            "arima_model": True,
            "xreg_model": bool(xreg_model),
            "error_type": "A",
            "trend_type": "N",
            "season_type": "N",
            "model_is_trendy": False,
            "model_is_seasonal": False,
        },
        components_dict={
            "components_number_ets": 0,
            "components_number_ets_seasonal": 0,
            "components_number_ets_non_seasonal": 0,
            "components_number_arima": int(components_number),
        },
        lags_dict={
            "lags_model_max": int(lags_model_max),
            "lags_model": lags_model_all,
            "lags_model_all": lags_model_all,
            "lags_model_seasonal": lags_model_seasonal,
        },
        obs_in_sample=obs_in_sample,
        o_type="n",
        loss=loss,
        distribution="dnorm",
        horizon=h or 0,
        multisteps=multisteps,
    )

    errors = np.array(adam_fitted.errors).ravel()
    fitted = np.array(adam_fitted.fitted).ravel()

    # Compute loss — R lines 510-555
    if not multisteps:
        if loss == "likelihood":
            # CES scaler: sqrt(sum(errors^2)/obs) — R line 482
            errors_ot = errors[ot_logical]
            errors_sum = math.fsum(float(error) ** 2 for error in errors_ot)
            scale = math.sqrt(errors_sum / obs_in_sample)
            y_ot = y_in_sample[ot_logical]
            fitted_ot = fitted[ot_logical]
            log_scale = math.log(scale)
            cf_value = math.fsum(
                _R_LN_SQRT_2PI
                + 0.5 * ((float(actual) - float(fitted_value)) / scale) ** 2
                + log_scale
                for actual, fitted_value in zip(y_ot, fitted_ot)
            )
        elif loss == "MSE":
            cf_value = np.sum(errors**2) / obs_in_sample
        elif loss == "MAE":
            cf_value = np.sum(np.abs(errors)) / obs_in_sample
        elif loss == "HAM":
            cf_value = np.sum(np.sqrt(np.abs(errors))) / obs_in_sample
        else:
            cf_value = np.sum(errors**2) / obs_in_sample
    else:
        # Multistep errors — R lines 534-555
        adam_errors = adam_cpp.ferrors(
            adam_fitted.states,
            mat_wt_f,
            mat_f_f,
            ilt_f,
            prt_f,
            int(h),
            y_f,
        ).errors
        adam_errors = np.array(adam_errors)

        if loss == "MSEh":
            cf_value = np.sum(adam_errors[:, h - 1] ** 2) / (obs_in_sample - h)
        elif loss == "TMSE":
            cf_value = np.sum(np.sum(adam_errors**2, axis=0) / (obs_in_sample - h))
        elif loss == "GTMSE":
            cf_value = np.sum(
                np.log(np.sum(adam_errors**2, axis=0) / (obs_in_sample - h))
            )
        elif loss == "MSCE":
            cf_value = np.sum(np.sum(adam_errors, axis=1) ** 2) / (obs_in_sample - h)
        elif loss == "GPL":
            cf_value = np.log(
                np.linalg.det(adam_errors.T @ adam_errors / (obs_in_sample - h))
            )
        else:
            cf_value = np.sum(adam_errors**2) / obs_in_sample

    if np.isnan(cf_value) or np.isinf(cf_value):
        cf_value = 1e300

    return cf_value
