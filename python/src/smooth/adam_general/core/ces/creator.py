"""
CES-specific state-space matrix construction.

Translates R/adam-ces.R creator() function (lines 379-478).
Builds matF, vecG, matWt, matVt for CES with four seasonality types.
"""

import numpy as np

from smooth.adam_general.core.creator.architector import adam_profile_creator
from smooth.adam_general.core.utils.utils import msdecompose


def ces_creator(
    seasonality,
    n_seasonal,
    lags_model_seasonal,
    lags_model_all,
    lags_model_max,
    components_number,
    xreg_number,
    obs_in_sample,
    obs_states,
    obs_all,
    y_in_sample,
    y_frequency,
    lags,
    xreg_data=None,
    xreg_model_initials=None,
    xreg_names=None,
):
    """
    Create CES state-space matrices.

    Parameters
    ----------
    seasonality : str
        One of "none", "simple", "partial", "full".
    n_seasonal : int
        Number of seasonal components (length of lags_model_seasonal).
    lags_model_seasonal : list of int
        Seasonal lag values.
    lags_model_all : list of int
        All lags including xreg (flat list).
    lags_model_max : int
        Maximum lag value.
    components_number : int
        Number of CES state components (excluding xreg).
    xreg_number : int
        Number of exogenous regressors.
    obs_in_sample : int
        Number of in-sample observations.
    obs_states : int
        Number of state columns (obs_in_sample + lags_model_max).
    obs_all : int
        Total observations including holdout.
    y_in_sample : np.ndarray
        In-sample time series values.
    y_frequency : int
        Time series frequency.
    lags : list of int
        Original user-provided lags.
    xreg_data : np.ndarray, optional
        Exogenous regressor data matrix.
    xreg_model_initials : dict, optional
        Initial values for xreg coefficients.
    xreg_names : list of str, optional
        Names of exogenous regressors.

    Returns
    -------
    dict
        Dictionary with keys: 'profiles_recent_table', 'index_lookup_table',
        'mat_f', 'vec_g', 'mat_wt', 'mat_vt', 'row_names'.
    """
    total_components = components_number + xreg_number

    # Create ADAM profiles — R line 385-388
    adam_profiles = adam_profile_creator(
        lags_model_all=lags_model_all,
        lags_model_max=lags_model_max,
        obs_all=obs_all,
        lags=lags,
    )
    profiles_recent_table = adam_profiles["profiles_recent_table"]
    index_lookup_table = adam_profiles["index_lookup_table"]

    # Base matrices — R lines 390-396
    mat_f = np.eye(total_components, order="F")
    mat_f[1, 0] = 1  # R: matF[2,1] <- 1
    vec_g = np.zeros((total_components, 1), order="F")
    mat_wt = np.ones((obs_in_sample, total_components), order="F")
    mat_wt[:, 1] = 0  # R: matWt[,2] <- 0
    mat_vt = np.zeros((total_components, obs_states), order="F")

    row_names = [""] * total_components

    # Seasonal decomposition if needed — R line 400-401.  We use
    # smoother="global" instead of the historical default "lowess"
    # because Python's greybox.lowess does not match R's stats::lowess
    # byte-for-byte (~3e-13 output diff on real data), and the seed
    # cascade through the CES recursion turns that into 4-ULP CF
    # differences that send BOBYQA into a different basin on partial /
    # full seasonality.  The shared olsCore.h backend behind
    # smoother="global" produces byte-identical decompositions across
    # languages (see msdecompose r-parity wiki).  R's CES creator is
    # updated in parallel to call msdecompose(..., smoother="global").
    if seasonality != "none":
        decomp = msdecompose(
            y_in_sample,
            lags=lags_model_seasonal,
            type="additive",
            smoother="global",
        )
        y_decomposed_seasonal = decomp["seasonal"]

    # Fill matrices for each seasonality type — R lines 404-467
    if seasonality == "full":
        row_names[0] = "level"
        row_names[1] = "potential"
        if n_seasonal > 1:
            for i in range(n_seasonal):
                row_names[2 + 2 * i] = f"seasonal 1[{lags_model_seasonal[i]}]"
                row_names[2 + 2 * i + 1] = f"seasonal 2[{lags_model_seasonal[i]}]"
            # R: matVt[1,1:lagsModelMax] <- mean(yInSample[1:lagsModelMax])
            level_init = np.mean(y_in_sample[:lags_model_max])
            mat_vt[0, :lags_model_max] = level_init
            mat_vt[1, :lags_model_max] = level_init / 1.1
            for i in range(n_seasonal):
                # R: matF[2+2*i, 2+2*i-1] <- 1
                mat_f[2 + 2 * i + 1, 2 + 2 * i] = 1  # 0-based
                # R: matWt[,2+2*i] <- 0
                mat_wt[:, 2 + 2 * i + 1] = 0  # 0-based
                # R: matVt[2+i*2-1, 1:lagsModelMax]
                seasonal_vals = y_decomposed_seasonal[i][:lags_model_max]
                mat_vt[2 + 2 * i, :lags_model_max] = seasonal_vals
                mat_vt[2 + 2 * i + 1, :lags_model_max] = seasonal_vals / 1.1
        else:
            # Single seasonal — R lines 418-426
            mat_f[3, 2] = 1  # R: matF[4,3] <- 1
            mat_wt[:, 3] = 0  # R: matWt[,4] <- 0
            row_names = ["level", "potential", "seasonal 1", "seasonal 2"]
            level_init = np.mean(y_in_sample[:lags_model_max])
            mat_vt[0, :lags_model_max] = level_init
            mat_vt[1, :lags_model_max] = level_init / 1.1
            mat_vt[2, :lags_model_max] = y_decomposed_seasonal[0][:lags_model_max]
            mat_vt[3, :lags_model_max] = mat_vt[2, :lags_model_max] / 1.1

    elif seasonality == "partial":
        row_names[0] = "level"
        row_names[1] = "potential"
        if n_seasonal > 1:
            for i in range(n_seasonal):
                row_names[2 + i] = f"seasonal[{lags_model_seasonal[i]}]"
            level_init = np.mean(y_in_sample[:lags_model_max])
            mat_vt[0, :lags_model_max] = level_init
            mat_vt[1, :lags_model_max] = level_init / 1.1
            for i in range(n_seasonal):
                mat_vt[2 + i, :lags_model_max] = y_decomposed_seasonal[i][
                    :lags_model_max
                ]
        else:
            row_names = ["level", "potential", "seasonal"]
            level_init = np.mean(y_in_sample[:lags_model_max])
            mat_vt[0, :lags_model_max] = level_init
            mat_vt[1, :lags_model_max] = level_init / 1.1
            mat_vt[2, :lags_model_max] = y_decomposed_seasonal[0][:lags_model_max]

    elif seasonality == "simple":
        if n_seasonal > 1:
            for i in range(n_seasonal):
                row_names[2 * i] = f"level.s[{lags_model_seasonal[i]}]"
                row_names[2 * i + 1] = f"potential.s[{lags_model_seasonal[i]}]"
            # R: matVt[(1:nSeasonal)*2-1, 1:lagsModelMax]
            for i in range(n_seasonal):
                mat_vt[2 * i, :lags_model_max] = y_in_sample[:lags_model_max]
                mat_vt[2 * i + 1, :lags_model_max] = (
                    mat_vt[2 * i, :lags_model_max] / 1.1
                )
            for i in range(n_seasonal):
                # R: matF[2*i, 2*i-1] <- 1
                mat_f[2 * i + 1, 2 * i] = 1  # 0-based
                # R: matWt[, 2*i] <- 0
                mat_wt[:, 2 * i + 1] = 0  # 0-based
        else:
            row_names = ["level.s", "potential.s"]
            mat_vt[0, :lags_model_max] = y_in_sample[:lags_model_max]
            mat_vt[1, :lags_model_max] = mat_vt[0, :lags_model_max] / 1.1

    else:
        # seasonality == "none" — R lines 463-467
        row_names = ["level", "potential"]
        init_len = min(max(10, y_frequency), obs_in_sample)
        level_init = np.mean(y_in_sample[:init_len])
        mat_vt[0, 0] = level_init
        mat_vt[1, 0] = level_init / 1.1

    # Add xreg names and initial values — R lines 469-473
    if xreg_number > 0 and xreg_names is not None:
        for i in range(xreg_number):
            if components_number + i < len(row_names):
                row_names[components_number + i] = xreg_names[i]
            else:
                row_names.append(xreg_names[i])

    if xreg_number > 0 and xreg_model_initials is not None:
        mat_vt[components_number : components_number + xreg_number, 0] = (
            xreg_model_initials
        )
        mat_wt[:, components_number : components_number + xreg_number] = xreg_data[
            :obs_in_sample
        ]

    # Ensure Fortran order for C++
    mat_f = np.asfortranarray(mat_f, dtype=np.float64)
    vec_g = np.asfortranarray(vec_g, dtype=np.float64)
    mat_wt = np.asfortranarray(mat_wt, dtype=np.float64)
    mat_vt = np.asfortranarray(mat_vt, dtype=np.float64)

    return {
        "profiles_recent_table": profiles_recent_table,
        "index_lookup_table": index_lookup_table,
        "mat_f": mat_f,
        "vec_g": vec_g,
        "mat_wt": mat_wt,
        "mat_vt": mat_vt,
        "row_names": row_names,
    }
