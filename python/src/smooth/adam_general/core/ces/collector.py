"""
What CES reports from its parameters.

Translates ces_collector() of R/adam-ces.R.
"""

import numpy as np


def ces_collector(
    B,
    a,
    b,
    mat_vt,
    seasonality,
    n_seasonal,
    lags_model_max,
    lags_model_all,
    components_number,
    xreg_model,
    xreg_number,
):
    """The complex smoothing parameters a and b from B (updated in place), the
    initial states of CES and of the regressors (``mat_vt`` is time by components),
    and the name of the model (R's ces_collector)."""
    # Reconstruct complex a and b from B — R lines 1048-1093
    n_coefficients = 0
    if a["estimate"]:
        if seasonality != "simple":
            a["value"] = complex(B[0], B[1])
            n_coefficients = 2
        else:
            a_vals = []
            for i in range(n_seasonal):
                a_vals.append(
                    complex(B[n_coefficients + 2 * i], B[n_coefficients + 2 * i + 1])
                )
            a["value"] = a_vals if n_seasonal > 1 else a_vals[0]
            n_coefficients += 2 * n_seasonal

    if b["estimate"]:
        if seasonality == "partial":
            b_vals = B[n_coefficients : n_coefficients + n_seasonal]
            b["value"] = b_vals.tolist() if n_seasonal > 1 else float(b_vals[0])
            n_coefficients += n_seasonal
        elif seasonality == "full":
            b_vals = []
            for i in range(n_seasonal):
                b_vals.append(
                    complex(
                        B[n_coefficients + 2 * i],
                        B[n_coefficients + 2 * i + 1],
                    )
                )
            b["value"] = b_vals if n_seasonal > 1 else b_vals[0]
            n_coefficients += 2 * n_seasonal

    # Initial values — R lines 1021-1044
    # mat_vt is (time, components) after transpose
    initial_states = {}
    if seasonality == "none":
        initial_states["nonseasonal"] = mat_vt[0, 0:2]
    elif seasonality == "simple":
        initial_states["seasonal"] = mat_vt[:lags_model_max, : n_seasonal * 2]
    else:
        initial_states["nonseasonal"] = mat_vt[0, 0:2]
        seasonal_mask = np.array(lags_model_all[:components_number]) != 1
        initial_states["seasonal"] = mat_vt[:lags_model_max, :components_number][
            :, seasonal_mask
        ]
    if xreg_model:
        initial_states["xreg"] = mat_vt[
            0, components_number : components_number + xreg_number
        ]

    # Model name — R lines 1096-1104
    model_name = "CES"
    if xreg_model:
        model_name += "X"
    model_name += f"({seasonality})"
    return a, b, initial_states, model_name
