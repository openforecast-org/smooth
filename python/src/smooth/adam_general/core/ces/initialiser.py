"""
CES-specific initial parameter vector construction.

Translates R/adam-ces.R initialiser() function (lines 680-746).
Builds the initial B vector for optimization. CES uses NO explicit bounds.
"""

import numpy as np


def ces_initialiser(
    a,
    b,
    seasonality,
    n_seasonal,
    lags_model_seasonal,
    lags_model_max,
    mat_vt,
    initial_type,
    components_number,
    xreg_model=False,
    xreg_number=0,
    xreg_names=None,
):
    """
    Create initial parameter vector B for CES optimization.

    Parameters
    ----------
    a : dict
        Complex smoothing parameter a. Keys: 'value', 'estimate', 'number'.
    b : dict
        Smoothing parameter b. Keys: 'value', 'estimate', 'number'.
    seasonality : str
        One of "none", "simple", "partial", "full".
    n_seasonal : int
        Number of seasonal components.
    lags_model_seasonal : list of int
        Seasonal lag values.
    lags_model_max : int
        Maximum lag.
    mat_vt : np.ndarray
        State vector matrix (from creator, used for initial state values).
    initial_type : str
        Initialization type.
    components_number : int
        Number of CES state components (excluding xreg).
    xreg_model : bool
        Whether exogenous regressors are present.
    xreg_number : int
        Number of regressors.
    xreg_names : list of str, optional
        Names of xreg variables.

    Returns
    -------
    np.ndarray
        Initial parameter vector B (1-D).
    """
    B = []
    names = []

    # --- Parameter a --- R lines 683-699
    if a["estimate"]:
        if seasonality != "simple":
            B.extend([1.3, 1.0])
            names.extend(["alpha_0", "alpha_1"])
        else:
            if n_seasonal > 1:
                for i in range(n_seasonal):
                    B.extend([1.3, 1.0])
                    lag = lags_model_seasonal[i]
                    names.extend([f"alpha_0[{lag}]", f"alpha_1[{lag}]"])
            else:
                B.extend([1.3, 1.0])
                names.extend(["alpha_0", "alpha_1"])

    # --- Parameter b --- R lines 702-721
    if b["estimate"]:
        if seasonality == "partial":
            if n_seasonal > 1:
                for i in range(n_seasonal):
                    B.append(0.1)
                    names.append(f"beta[{lags_model_seasonal[i]}]")
            else:
                B.append(0.1)
                names.append("beta")
        else:
            # seasonality == "full"
            if n_seasonal > 1:
                for i in range(n_seasonal):
                    B.extend([1.3, 1.0])
                    lag = lags_model_seasonal[i]
                    names.extend([f"beta_0[{lag}]", f"beta_1[{lag}]"])
            else:
                B.extend([1.3, 1.0])
                names.extend(["beta_0", "beta_1"])

    # --- Initial states --- R lines 724-740
    if initial_type not in ("backcasting", "complete"):
        if seasonality != "simple":
            # R: B <- c(B, matVt[1:2, 1])
            B.extend(mat_vt[0:2, 0].tolist())
            names.extend(["level_0", "potential_0"])

        if seasonality == "simple":
            # R: B <- c(B, matVt[1:(nSeasonal*2), 1:lagsModelMax])
            # R stores column-major, so flatten by column
            for col in range(lags_model_max):
                for row in range(n_seasonal * 2):
                    B.append(mat_vt[row, col])
                    names.append(f"state_{row}_lag_{col}")
        elif seasonality == "partial":
            # R: B <- c(B, matVt[2+(1:nSeasonal), 1:lagsModelMax])
            for col in range(lags_model_max):
                for i in range(n_seasonal):
                    B.append(mat_vt[2 + i, col])
                    names.append(f"seasonal_{i}_lag_{col}")
        elif seasonality == "full":
            # R: B <- c(B, matVt[2+(1:(nSeasonal*2)), 1:lagsModelMax])
            for col in range(lags_model_max):
                for i in range(n_seasonal * 2):
                    B.append(mat_vt[2 + i, col])
                    names.append(f"seasonal_{i}_lag_{col}")

    # --- Xreg initials --- R lines 742-744
    if xreg_model and initial_type != "complete":
        xreg_init = mat_vt[components_number : components_number + xreg_number, 0]
        B.extend(xreg_init.tolist())
        if xreg_names is not None:
            names.extend(xreg_names)
        else:
            names.extend([f"xreg_{i}" for i in range(xreg_number)])

    return np.array(B, dtype=np.float64)
