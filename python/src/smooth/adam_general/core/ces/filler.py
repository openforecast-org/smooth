"""
CES-specific parameter filling from optimization vector B.

Translates R/adam-ces.R filler() function (lines 269-377).
Maps B vector elements to matF, vecG, and matVt initial states.
"""

import numpy as np


def ces_filler(
    B,
    mat_vt,
    mat_f,
    vec_g,
    a,
    b,
    seasonality,
    n_seasonal,
    lags_model_seasonal,
    lags_model_max,
    initial_type,
    xreg_model=False,
    xreg_number=0,
    initial_xreg_estimate=False,
    components_number=0,
    initial_value=None,
):
    """
    Fill CES state-space matrices with parameter values from B.

    Parameters
    ----------
    B : np.ndarray
        Optimization parameter vector.
    mat_vt : np.ndarray
        State vector matrix (modified via copy).
    mat_f : np.ndarray
        Transition matrix (modified via copy).
    vec_g : np.ndarray
        Persistence vector (modified via copy).
    a : dict
        Complex smoothing parameter a. Keys: 'value', 'estimate'.
    b : dict
        Smoothing parameter b (real for partial, complex for full).
        Keys: 'value', 'estimate', 'number'.
    seasonality : str
        One of "none", "simple", "partial", "full".
    n_seasonal : int
        Number of seasonal components.
    lags_model_seasonal : list of int
        Seasonal lag values.
    lags_model_max : int
        Maximum lag.
    initial_type : str
        Initialization type: "backcasting", "optimal", "two-stage", "complete",
        "provided".
    xreg_model : bool
        Whether exogenous regressors are present.
    xreg_number : int
        Number of regressors.
    initial_xreg_estimate : bool
        Whether xreg initials are estimated.
    components_number : int
        Number of CES components (excluding xreg).
    initial_value : np.ndarray, optional
        Provided initial values (when initial_type="provided").

    Returns
    -------
    dict
        {'mat_f': array, 'vec_g': array, 'vt': array}
        vt is the initial portion of mat_vt (first lags_model_max columns).
    """
    n_coefficients = 0
    j = 0

    # --- Parameter a --- R lines 275-308
    if seasonality != "simple":
        if a["estimate"]:
            # R: matF[1,2] <- B[2]-1; matF[2,2] <- 1-B[1]
            mat_f[0, 1] = B[1] - 1
            mat_f[1, 1] = 1 - B[0]
            # R: vecG[1:2,] <- c(B[1]-B[2], B[1]+B[2])
            vec_g[0, 0] = B[0] - B[1]
            vec_g[1, 0] = B[0] + B[1]
            n_coefficients += 2
        else:
            a_val = a["value"]
            mat_f[0, 1] = a_val.imag - 1
            mat_f[1, 1] = 1 - a_val.real
            vec_g[0, 0] = a_val.real - a_val.imag
            vec_g[1, 0] = a_val.real + a_val.imag
    else:
        # Simple seasonality — R lines 292-308
        if a["estimate"]:
            for i in range(n_seasonal):
                # R: matF[i*2, i*2] <- 1-B[nCoefficients+i*2-1]  (1-based)
                # Python 0-based: mat_f[2*i+1, 2*i+1] = 1-B[n_coefficients+2*i]
                mat_f[2 * i + 1, 2 * i + 1] = 1 - B[n_coefficients + 2 * i]
                # R: matF[i*2-1, i*2] <- B[nCoefficients+i*2]-1
                mat_f[2 * i, 2 * i + 1] = B[n_coefficients + 2 * i + 1] - 1
                # R: vecG[-c(1,0)+2*i,] = c(B[..i*2-1]-B[..i*2], B[..i*2-1]+B[..i*2])
                vec_g[2 * i, 0] = (
                    B[n_coefficients + 2 * i] - B[n_coefficients + 2 * i + 1]
                )
                vec_g[2 * i + 1, 0] = (
                    B[n_coefficients + 2 * i] + B[n_coefficients + 2 * i + 1]
                )
            n_coefficients += 2 * n_seasonal
        else:
            for i in range(n_seasonal):
                a_val = a["value"][i] if n_seasonal > 1 else a["value"]
                mat_f[2 * i + 1, 2 * i + 1] = 1 - a_val.real
                mat_f[2 * i, 2 * i + 1] = a_val.imag - 1
                vec_g[2 * i, 0] = a_val.real - a_val.imag
                vec_g[2 * i + 1, 0] = a_val.real + a_val.imag

    # --- Parameter b --- R lines 311-340
    if seasonality == "partial":
        if b["estimate"]:
            # R: vecG[2+1:nSeasonal,] <- B[nCoefficients+1:nSeasonal]
            for i in range(n_seasonal):
                vec_g[2 + i, 0] = B[n_coefficients + i]
            n_coefficients += n_seasonal
        else:
            for i in range(n_seasonal):
                b_val = b["value"][i] if n_seasonal > 1 else b["value"]
                vec_g[2 + i, 0] = b_val

    elif seasonality == "full":
        if b["estimate"]:
            for i in range(n_seasonal):
                # R: matF[2+i*2, 2+i*2] <- 1-B[nCoefficients+i*2-1]
                mat_f[2 + 2 * i + 1, 2 + 2 * i + 1] = 1 - B[n_coefficients + 2 * i]
                # R: matF[2+i*2-1, 2+i*2] <- B[nCoefficients+i*2]-1
                mat_f[2 + 2 * i, 2 + 2 * i + 1] = B[n_coefficients + 2 * i + 1] - 1
                # R: vecG[2-c(1,0)+2*i,] = c(B[..]-B[..], B[..]+B[..])
                vec_g[2 + 2 * i, 0] = (
                    B[n_coefficients + 2 * i] - B[n_coefficients + 2 * i + 1]
                )
                vec_g[2 + 2 * i + 1, 0] = (
                    B[n_coefficients + 2 * i] + B[n_coefficients + 2 * i + 1]
                )
            n_coefficients += 2 * n_seasonal
        else:
            for i in range(n_seasonal):
                b_val = b["value"][i] if n_seasonal > 1 else b["value"]
                mat_f[2 + 2 * i + 1, 2 + 2 * i + 1] = 1 - b_val.real
                mat_f[2 + 2 * i, 2 + 2 * i + 1] = b_val.imag - 1
                vec_g[2 + 2 * i, 0] = b_val.real - b_val.imag
                vec_g[2 + 2 * i + 1, 0] = b_val.real + b_val.imag

    # --- Initial states --- R lines 342-368
    vt = mat_vt[:, :lags_model_max].copy()

    if initial_type in ("optimal", "two-stage"):
        # Non-seasonal part — R lines 345-348
        if seasonality != "simple":
            vt[0, :lags_model_max] = B[n_coefficients]
            vt[1, :lags_model_max] = B[n_coefficients + 1]
            n_coefficients += 2
            j += 2

        # Seasonal part — R lines 351-364
        if seasonality in ("simple", "full"):
            for i in range(n_seasonal):
                lag = lags_model_seasonal[i]
                vt[j : j + 2, :lag] = B[
                    n_coefficients : n_coefficients + 2 * lag
                ].reshape(2, lag, order="F")
                n_coefficients += lag * 2
                j += 2
        elif seasonality == "partial":
            for i in range(n_seasonal):
                lag = lags_model_seasonal[i]
                vt[j, :lag] = B[n_coefficients : n_coefficients + lag]
                n_coefficients += lag
                j += 1

    elif initial_type == "provided" and initial_value is not None:
        vt[:, :lags_model_max] = initial_value

    # --- Xreg initials --- R lines 371-374
    if xreg_model and initial_xreg_estimate and initial_type != "complete":
        vt[j : j + xreg_number, :] = B[
            n_coefficients : n_coefficients + xreg_number, np.newaxis
        ]
        n_coefficients += xreg_number

    return {"mat_f": mat_f, "vec_g": vec_g, "vt": vt}
