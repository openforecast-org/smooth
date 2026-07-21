import math

import numpy as np
from numpy.linalg import eigvals

from smooth.adam_general._eigenCalc import smooth_eigens
from smooth.adam_general.core.creator import filler
from smooth.adam_general.core.utils.gradient import adam_fit_or_gradient
from smooth.adam_general.core.utils.utils import (
    calculate_entropy,
    calculate_likelihood,
    calculate_multistep_loss,
    scaler,
)


def trim_b_for_penalty(
    B,  # noqa: N803
    components_dict,
    persistence_checked,
    explanatory_checked,
    phi_dict,
    arima_checked,
    initials_checked,
    model_type_dict,
    lags_dict,
    general,
):
    """Build the trimmed/transformed parameter vector for LASSO/RIDGE penalty.

    Mirrors the block in the additive/ARIMA ADAM CF (R/adam.R:894-937,
    `cost_functions.py:544-677` pre-refactor): drops initial-state entries
    from ``B``, shrinks ``phi`` and the AR coefficients to ``1 - x``
    (so the penalty pushes them toward 1, not 0), keeps MA as-is, and
    normalises xreg coefficients by the additive ``y_denominator``.

    Shared by ADAM's ``CF``, ``om_cf`` (for ``OM``) and ``omg_cf``
    (for ``OMG``) so the regularisation acts on the exact same parameter
    subset across all three.
    """
    components_ets = int(components_dict.get("components_number_ets", 0) or 0)
    persistence_xreg_est = int(
        persistence_checked.get("persistence_xreg_estimate", False) or 0
    )
    xreg_num = int(explanatory_checked.get("xreg_number", 0) or 0)
    phi_est = int(phi_dict.get("phi_estimate", False) or 0)
    ar_orders = arima_checked.get("ar_orders") if arima_checked else None
    ma_orders = arima_checked.get("ma_orders") if arima_checked else None
    arima_model = arima_checked.get("arima_model", False) if arima_checked else False

    if ar_orders is None or (hasattr(ar_orders, "__len__") and len(ar_orders) == 0):
        ar_orders = [0]
    if ma_orders is None or (hasattr(ma_orders, "__len__") and len(ma_orders) == 0):
        ma_orders = [0]

    persistence_to_skip = (
        components_ets
        + persistence_xreg_est * xreg_num
        + phi_est
        + sum(ar_orders)
        + sum(ma_orders)
    )

    B_penalty = np.asarray(B, dtype=float).copy()  # noqa: N806

    if phi_est:
        phi_idx = components_ets + persistence_xreg_est * xreg_num
        if phi_idx < len(B_penalty):
            B_penalty[phi_idx] = 1 - B_penalty[phi_idx]

    j = components_ets + persistence_xreg_est * xreg_num + phi_est

    if arima_model and (sum(ma_orders) > 0 or sum(ar_orders) > 0):
        lags = lags_dict.get("lags", [1]) if lags_dict else [1]
        if not hasattr(lags, "__len__"):
            lags = [lags]
        for i in range(len(lags)):
            ar_order_i = ar_orders[i] if i < len(ar_orders) else 0
            ma_order_i = ma_orders[i] if i < len(ma_orders) else 0
            if ar_order_i > 0 and j + ar_order_i <= len(B_penalty):
                B_penalty[j : j + ar_order_i] = 1 - B_penalty[j : j + ar_order_i]
            j += ar_order_i + ma_order_i

    initial_type = (
        initials_checked.get("initial_type", "optimal")
        if initials_checked
        else "optimal"
    )
    if isinstance(initial_type, list):
        initial_type_matches = any(
            t in ("optimal", "backcasting", "two-stage") for t in initial_type
        )
    else:
        initial_type_matches = initial_type in (
            "optimal",
            "backcasting",
            "two-stage",
        )

    if initial_type_matches:
        if xreg_num > 0:
            denominator = general.get("denominator")
            error_type = model_type_dict.get("error_type", "A")
            if error_type == "A" and denominator is not None:
                B_penalty = np.concatenate(
                    [
                        B_penalty[:persistence_to_skip],
                        B_penalty[-xreg_num:] / denominator,
                    ]
                )
            else:
                B_penalty = np.concatenate(
                    [B_penalty[:persistence_to_skip], B_penalty[-xreg_num:]]
                )
        else:
            B_penalty = B_penalty[:persistence_to_skip]

    return B_penalty


def CF(  # noqa: N802
    B,
    model_type_dict,
    components_dict,
    lags_dict,
    matrices_dict,
    persistence_checked,
    initials_checked,
    arima_checked,
    explanatory_checked,
    phi_dict,
    constants_checked,
    observations_dict,
    profile_dict,
    general,
    adam_cpp,
    bounds="usual",
    other=None,
    otherParameterEstimate=False,
    arPolynomialMatrix=None,
    maPolynomialMatrix=None,
    regressors=None,
    return_fitted=False,
):
    """
    Cost Function for ADAM model parameter estimation.

    This function calculates the value of the cost function (CF) for given parameters
    during the optimization process. The CF is minimized to find optimal parameter
    values
    for the ADAM model. The function implements various loss functions (likelihood, MSE,
    MAE,
    HAM, LASSO, RIDGE) and enforces parameter constraints (bounds).

    The cost function is evaluated as follows:

    1. **Parameter Filling**: Fill model matrices with current parameter values from
    vector B
    2. **Bounds Checking**: Apply parameter constraints based on the 'bounds' setting:

       - **"usual"**: Classical restrictions on smoothing parameters:

         * ETS smoothing parameters: :math:`0 \\leq \\alpha, \\beta, \\gamma \\leq 1`
         * Trend constraint: :math:`\\beta \\leq \\alpha`
         * Seasonal constraint: :math:`\\gamma \\leq 1 - \\alpha`
         * Damping constraint: :math:`0 \\leq \\phi \\leq 1`
         * ARIMA stationarity: AR and MA polynomial roots outside unit circle

       - **"admissible"**: Check eigenvalues of the state transition matrix to ensure
       stability
       - **None**: No bounds checking

    3. **Model Fitting**: Call C++ fitter to compute fitted values and errors
    4. **Loss Calculation**: Compute loss based on specified loss function:

       - **"likelihood"**: Negative log-likelihood for specified distribution
       - **"MSE"**: Mean Squared Error
       - **"MAE"**: Mean Absolute Error
       - **"HAM"**: Half Absolute Moment (square root of absolute errors)
       - **"LASSO"** / **"RIDGE"**: Regularized loss with L1/L2 penalties

    Parameters
    ----------
    B : numpy.ndarray
        Parameter vector containing (in order):

        - ETS persistence parameters (α, β, γ)
        - Damping parameter (φ)
        - Initial states
        - ARIMA parameters (AR, MA coefficients)
        - Regression coefficients
        - Constant term
        - Distribution parameters (if estimated)

    model_type_dict : dict
        Model type specification containing:

        - 'error_type': Error type ('A' for additive, 'M' for multiplicative)
        - 'trend_type': Trend type ('N', 'A', 'Ad', 'M', 'Md')
        - 'season_type': Seasonality type ('N', 'A', 'M')
        - 'ets_model': Whether ETS components are present
        - 'arima_model': Whether ARIMA components are present
        - 'model_is_trendy': Whether trend is present
        - 'model_is_seasonal': Whether seasonality is present

    components_dict : dict
        Components information containing:

        - 'components_number_ets': Total number of ETS components
        - 'components_number_ets_seasonal': Number of seasonal ETS components
        - 'components_number_ets_non_seasonal': Number of non-seasonal ETS components
        - 'components_number_arima': Number of ARIMA components

    lags_dict : dict
        Lag structure information containing:

        - 'lags': Vector of lags for each seasonal component
        - 'lags_model': Lags for each model component
        - 'lags_model_all': Complete lag specification
        - 'lags_model_max': Maximum lag value

    matrices_dict : dict
        State-space matrices from creator(), modified in-place:

        - 'mat_vt': State vector matrix
        - 'mat_wt': Measurement matrix
        - 'mat_f': Transition matrix
        - 'vec_g': Persistence vector

    persistence_checked : dict
        Persistence parameters specification from checker()
    initials_checked : dict
        Initial values specification containing:

        - 'initial_type': Initialization method ('optimal', 'backcasting', 'complete')
        - 'n_iterations': Number of backcasting iterations

    arima_checked : dict
        ARIMA specification containing:

        - 'arima_model': Whether ARIMA is present
        - 'ar_estimate': Whether to estimate AR parameters
        - 'ma_estimate': Whether to estimate MA parameters
        - 'ar_required': Whether AR is required
        - 'ma_required': Whether MA is required
        - 'ar_orders': AR orders for each lag
        - 'ma_orders': MA orders for each lag

    explanatory_checked : dict
        External regressors specification containing:

        - 'xreg_model': Whether external regressors are present
        - 'xreg_number': Number of external regressors

    phi_dict : dict
        Damping parameter specification containing:

        - 'phi_estimate': Whether to estimate damping parameter
        - 'phi': Current damping parameter value

    constants_checked : dict
        Constant term specification containing:

        - 'constant_required': Whether a constant is included
        - 'constant_estimate': Whether to estimate the constant

    observations_dict : dict
        Observations information containing:

        - 'y_in_sample': In-sample time series values
        - 'ot': Occurrence variable (for intermittent data)
        - 'ot_logical': Boolean mask for non-zero observations
        - 'obs_in_sample': Number of in-sample observations
        - 'obs_zero': Number of zero observations
        - 'occurrence_model': Whether occurrence model is present

    profile_dict : dict
        Profile matrices for time-varying parameters containing:

        - 'profiles_recent_table': Recent values for profile initialization
        - 'index_lookup_table': Index lookup for profile access

    general : dict
        General model configuration containing:

        - 'loss': Loss function ('likelihood', 'MSE', 'MAE', 'HAM', 'LASSO', 'RIDGE')
        - 'distribution_new': Error distribution ('dnorm', 'dlaplace', 'ds', 'dgnorm',
        'dlnorm', 'dgamma', 'dinvgauss')
        - 'multisteps': Whether to use multistep loss
        - 'lambda': Regularization parameter for LASSO/RIDGE
        - 'denominator': Scaling denominator for LASSO/RIDGE
        - 'y_denominator': Y-value scaling for LASSO/RIDGE

    bounds : str, optional
        Type of bounds to enforce:

        - "usual": Classical parameter restrictions (default)
        - "admissible": Admissibility constraints based on eigenvalues
        - None: No bounds checking

    other : float, optional
        Additional distribution parameters (e.g., shape parameter for generalized
        normal)
    otherParameterEstimate : bool, optional
        Whether to estimate distribution parameters from B vector
    arPolynomialMatrix : numpy.ndarray, optional
        Companion matrix for AR polynomial (for bounds checking)
    maPolynomialMatrix : numpy.ndarray, optional
        Companion matrix for MA polynomial (for bounds checking)
    regressors : str, optional
        Regressor handling method ('use', 'select', 'adapt')

    Returns
    -------
    float
        Cost function value. Returns a large penalty (1e100 or 1e300) if constraints are
        violated or computation fails. Otherwise returns the computed loss value based
        on
        the specified loss function.

    Notes
    -----
    The function is called repeatedly during optimization by NLopt. It performs the
    following:

    1. Fills model matrices with current parameter values using ``filler()``
    2. Checks parameter bounds and returns penalty if violated
    3. Calls C++ ``adam_fitter()`` to compute fitted values and errors
    4. Calculates and returns the appropriate loss function value

    **Important Implementation Details**:

    - Matrices are passed to C++ by reference and may be modified
    - Arrays must be in Fortran order for C++ compatibility
    - Copies are made of matrices to avoid cross-iteration contamination
    - NaN values in CF result in a large penalty (1e300)

    **Parameter Constraints**:

    For "usual" bounds, violations return penalty 1e100:

    - ETS smoothing parameters outside [0,1]
    - Trend parameter β > α (violates smoothness)
    - Seasonal parameter γ > 1-α
    - Damping φ outside [0,1]
    - ARIMA polynomial roots inside unit circle (non-stationary)

    For "admissible" bounds, violations return penalty 1e100 × max(eigenvalue):

    - Eigenvalues of state transition matrix > 1 (unstable system)

    See Also
    --------
    log_Lik_ADAM : Calculate log-likelihood for fitted model
    filler : Fill model matrices with parameter values
    adam_fitter : C++ function for model fitting

    References
    ----------
    .. [1] Svetunkov, I. (2023). "Smooth forecasting with the smooth package in R".
           arXiv:2301.01790.
    .. [2] Hyndman, R.J., Koehler, A.B., Ord, J.K., and Snyder, R.D. (2008).
           "Forecasting with Exponential Smoothing: The State Space Approach".
           Springer-Verlag.

    Examples
    --------
    This function is typically called internally during optimization::

        >>> # During optimization, NLopt calls CF repeatedly
        >>> cf_value = CF(B=initial_params, model_type_dict=..., components_dict=...,
        ...)
        >>> # If cf_value is large (1e100), constraints were violated
    """

    # R passes adamCreated$matVt as a parameter to every CF call, so each call
    # gets a fresh copy (R copy-on-modify). Python mutates matrices_dict["mat_vt"]
    # in-place. The elif branch in filler writes ari_poly*seed to the seed row
    # (last ARIMA state), so without restoration the seed accumulates as
    # ari_poly^N*seed after N calls. Save it before filler; restore after the
    # Fortran copy so C++ still receives the correctly filled value.
    arima_seed_backup = None
    if (
        arima_checked["arima_model"]
        and any([arima_checked["ar_estimate"], arima_checked["ma_estimate"]])
        and initials_checked["initial_type"] in ["complete", "backcasting", "gradient"]
    ):
        seed_row_idx = (
            components_dict["components_number_ets"]
            + components_dict["components_number_arima"]
            - 1
        )
        n_init = initials_checked["initial_arima_number"]
        arima_seed_backup = (
            seed_row_idx,
            n_init,
            matrices_dict["mat_vt"][seed_row_idx, :n_init].copy(),
        )

    # Fill in the matrices
    adam_elements = filler(
        B,
        model_type_dict,
        components_dict,
        lags_dict,
        matrices_dict,
        persistence_checked,
        initials_checked,
        arima_checked,
        explanatory_checked,
        phi_dict,
        constants_checked,
        adam_cpp=adam_cpp,
    )

    # Capture the filler-modified state for C++ and profiles before restoring.
    # Profiles and the Fortran copy must capture the mutated seed row (ari_poly*seed).
    profile_dict["profiles_recent_table"][:] = adam_elements["mat_vt"][
        :, : lags_dict["lags_model_max"]
    ]
    # An explicit copy: the gradient dispatcher overwrites the head of mat_vt
    # in place, and np.asfortranarray would alias an already-Fortran-ordered
    # array, leaking the solved initials into the shared creator matrices
    # (R's copy-on-modify semantics never leak them)
    mat_vt = np.array(adam_elements["mat_vt"], dtype=np.float64, order="F")

    # Restore the seed row of mat_vt before the next CF call so each
    # optimiser step starts from the same initial state (the C++ fitter may
    # mutate the row in place).
    if arima_seed_backup is not None:
        idx, n, row = arima_seed_backup
        matrices_dict["mat_vt"][idx, :n] = row

    # If we estimate parameters of distribution, take it from the B vector
    if otherParameterEstimate:
        other = abs(B[-1])
        if general["distribution_new"] in ["dgnorm", "dlgnorm"] and other < 0.25:
            return 1e10 / other
    # Check the bounds, classical restrictions
    # print(components_dict['components_number_ets_non_seasonal'])

    if bounds == "usual":
        if arima_checked["arima_model"] and any(
            [arima_checked["ar_estimate"], arima_checked["ma_estimate"]]
        ):
            if (
                arima_checked["ar_estimate"]
                and np.all(-adam_elements["arima_polynomials"]["arPolynomial"][1:] > 0)
                and sum(-adam_elements["arima_polynomials"]["arPolynomial"][1:]) >= 1
            ):
                arPolynomialMatrix[:, 0] = -adam_elements["arima_polynomials"][
                    "arPolynomial"
                ][1:]
                arPolyroots = np.abs(eigvals(arPolynomialMatrix))
                if any(arPolyroots > 1):
                    return 1e100 * max(arPolyroots)

            if (
                arima_checked["ma_estimate"]
                and sum(adam_elements["arima_polynomials"]["maPolynomial"][1:]) >= 1
            ):
                maPolynomialMatrix[:, 0] = adam_elements["arima_polynomials"][
                    "maPolynomial"
                ][1:]
                maPolyroots = np.abs(eigvals(maPolynomialMatrix))
                if any(maPolyroots > 1):
                    return 1e100 * max(abs(maPolyroots))

        if model_type_dict["ets_model"]:
            if any(
                adam_elements["vec_g"][: components_dict["components_number_ets"]] > 1
            ) or any(
                adam_elements["vec_g"][: components_dict["components_number_ets"]] < 0
            ):
                return 1e300
            if model_type_dict["model_is_trendy"]:
                if adam_elements["vec_g"][1] > adam_elements["vec_g"][0]:
                    return 1e300
                if model_type_dict["model_is_seasonal"] and any(
                    adam_elements["vec_g"][
                        components_dict[
                            "components_number_ets_non_seasonal"
                        ] : components_dict["components_number_ets_non_seasonal"]
                        + components_dict["components_number_ets_seasonal"]
                    ]
                    > (1 - adam_elements["vec_g"][0])
                ):
                    return 1e300

            elif model_type_dict["model_is_seasonal"] and any(
                adam_elements["vec_g"][
                    components_dict[
                        "components_number_ets_non_seasonal"
                    ] : components_dict["components_number_ets_non_seasonal"]
                    + components_dict["components_number_ets_seasonal"]
                ]
                > (1 - adam_elements["vec_g"][0])
            ):
                return 1e300

            if phi_dict["phi_estimate"] and (
                adam_elements["mat_f"][1, 1] > 1 or adam_elements["mat_f"][1, 1] < 0
            ):
                return 1e300

        if explanatory_checked["xreg_model"] and regressors == "adapt":
            xreg_start = (
                components_dict["components_number_ets"]
                + components_dict["components_number_arima"]
            )
            deltas = adam_elements["vec_g"][
                xreg_start : xreg_start + explanatory_checked["xreg_number"]
            ]
            if np.any(deltas > 1) or np.any(deltas < 0):
                return 1e100 * np.max(np.abs(deltas - 0.5))

    elif bounds == "admissible":
        if arima_checked["arima_model"]:
            # Mirror R's admissible AR-check (utils-adam.R:1484-1486):
            # AND of (all -arPoly[1:] > 0) AND (sum -arPoly[1:] >= 1).  The
            # legacy `or sum(...) < 0` clause was a stale rule that fired the
            # AR-eigvals penalty on B-points where R passes through to the
            # state-space smoothEigens check; on ARIMA-only fits with negative
            # AR seeds (e.g. NLopt simplex probes with ar1[1] ~= -2) it pushed
            # Py into a different basin from R.
            ar_neg_coefs = -adam_elements["arima_polynomials"]["arPolynomial"][1:]
            if (
                arima_checked["ar_estimate"]
                and np.all(np.asarray(ar_neg_coefs) > 0)
                and sum(ar_neg_coefs) >= 1
            ):
                arPolynomialMatrix[:, 0] = ar_neg_coefs
                eigenValues = np.abs(eigvals(arPolynomialMatrix))
                if any(eigenValues > 1):
                    return 1e100 * np.max(eigenValues)

        if model_type_dict["ets_model"] or arima_checked["arima_model"]:
            has_delta = explanatory_checked["xreg_model"] and regressors == "adapt"
            eigenValues = smooth_eigens(
                persistence=np.asfortranarray(
                    adam_elements["vec_g"].reshape(-1, 1), dtype=np.float64
                ),
                transition=np.asfortranarray(adam_elements["mat_f"], dtype=np.float64),
                measurement=np.asfortranarray(
                    adam_elements["mat_wt"], dtype=np.float64
                ),
                lags_model_all=np.asarray(lags_dict["lags_model_all"], dtype=np.int32),
                xreg_model=explanatory_checked["xreg_model"],
                obs_in_sample=observations_dict["obs_in_sample"],
                has_delta=has_delta,
                xreg_number=explanatory_checked["xreg_number"],
                constant_required=constants_checked["constant_required"],
            )
            if np.any(eigenValues > 1 + 1e-50):
                return 1e100 * np.max(eigenValues)

    y_in_sample = np.asarray(observations_dict["y_in_sample"], dtype=np.float64)
    ot = np.asarray(observations_dict["ot"], dtype=np.float64)

    mat_wt = np.asfortranarray(adam_elements["mat_wt"], dtype=np.float64)
    mat_f = np.asfortranarray(adam_elements["mat_f"], dtype=np.float64)
    vec_g = np.asfortranarray(adam_elements["vec_g"], dtype=np.float64)

    index_lookup_table = np.asfortranarray(
        profile_dict["index_lookup_table"], dtype=np.uint64
    )
    profiles_recent_table = np.asfortranarray(
        profile_dict["profiles_recent_table"], dtype=np.float64
    )

    # Check if initial_type is a list or string and compute backcast correctly
    if isinstance(initials_checked["initial_type"], list):
        backcast_value = any(
            [
                t == "complete" or t == "backcasting"
                for t in initials_checked["initial_type"]
            ]
        )
    else:
        backcast_value = initials_checked["initial_type"] in ["complete", "backcasting"]

    # Call adam_cpp.fit() (or the gradient initial-state solve) via the shared
    # dispatcher — parity with R's adam_fitOrGradient. Parameters that were passed
    # to adam_fitter are now stored in adam_cpp (E, T, S, etc.).
    adam_fitted = adam_fit_or_gradient(
        adam_cpp=adam_cpp,
        mat_vt=mat_vt,
        mat_wt=mat_wt,
        mat_f=mat_f,
        vec_g=vec_g,
        index_lookup_table=index_lookup_table,
        profiles_recent_table=profiles_recent_table,
        y_in_sample=y_in_sample,
        ot=ot,
        initial_type=initials_checked["initial_type"],
        n_iterations=initials_checked["n_iterations"],
        backcast_value=backcast_value,
        model_type_dict=model_type_dict,
        components_dict=components_dict,
        lags_dict=lags_dict,
        obs_in_sample=observations_dict["obs_in_sample"],
        loss=general["loss"],
        distribution=general.get(
            "distribution_new", general.get("distribution", "default")
        ),
        other=other,
        horizon=general.get("h", 0),
        multisteps=general["multisteps"],
    )

    # The in-sample fitted values and errors at this B, returned directly for
    # the OPG covariance: the caller recomputes the concentrated scale (via
    # scaler) and forms the per-observation likelihood. Mirrors om_cf/omg_cf's
    # return_fitted.
    if return_fitted:
        return (
            np.asarray(adam_fitted.fitted).ravel(),
            np.asarray(adam_fitted.errors).ravel(),
        )

    # adam_fitted.errors = np.repeat()

    # print('adam_fitted')
    # print(adam_fitted)
    if not general["multisteps"]:
        if general["loss"] == "likelihood":
            scale = scaler(
                general["distribution_new"],
                model_type_dict["error_type"],
                adam_fitted.errors[observations_dict["ot_logical"]],
                adam_fitted.fitted[observations_dict["ot_logical"]],
                observations_dict["obs_in_sample"],
                other,
            )
            # Calculate the likelihood.  Use math.fsum (Shewchuk exact) to
            # mirror R's LDOUBLE-accumulator sum() — NumPy's default pairwise
            # sum drifts by 1 ULP per call against R, which through NLopt's
            # deterministic simplex moves becomes a different convergence
            # point on flat ARIMA cost surfaces.  Same class of fix as the
            # msdecompose summation-order work.
            ll = calculate_likelihood(
                general["distribution_new"],
                model_type_dict["error_type"],
                observations_dict["y_in_sample"][observations_dict["ot_logical"]],
                adam_fitted.fitted[observations_dict["ot_logical"]],
                scale,
                other,
            )
            CFValue = -math.fsum(np.asarray(ll, dtype=np.float64).ravel().tolist())
            # Differential entropy for the logLik of occurrence model
            if observations_dict.get("occurrence_model", False) or any(
                ~observations_dict["ot_logical"]
            ):
                CFValueEntropy = calculate_entropy(
                    general["distribution_new"],
                    scale,
                    other,
                    observations_dict["obs_zero"],
                    adam_fitted.fitted[~observations_dict["ot_logical"]],
                )
                if np.isnan(CFValueEntropy) or CFValueEntropy < 0:
                    CFValueEntropy = np.inf
                CFValue += CFValueEntropy

        elif general["loss"] == "MSE":
            CFValue = np.sum(adam_fitted.errors**2) / observations_dict["obs_in_sample"]
        elif general["loss"] == "MAE":
            CFValue = (
                np.sum(np.abs(adam_fitted.errors)) / observations_dict["obs_in_sample"]
            )
        elif general["loss"] == "HAM":
            CFValue = (
                np.sum(np.sqrt(np.abs(adam_fitted.errors)))
                / observations_dict["obs_in_sample"]
            )
        elif general["loss"] in ["LASSO", "RIDGE"]:
            # Trim B for penalty term (shared helper — also called by om_cf
            # and omg_cf so OM/OMG penalise the exact same parameter subset
            # ADAM does).
            B_penalty = trim_b_for_penalty(  # noqa: N806
                B,
                components_dict,
                persistence_checked,
                explanatory_checked,
                phi_dict,
                arima_checked,
                initials_checked,
                model_type_dict,
                lags_dict,
                general,
            )

            # Flatten errors to avoid potential broadcasting issues
            errors_flat = np.asarray(adam_fitted.errors).ravel()

            # Calculate error term based on error type
            error_type = model_type_dict.get("error_type", "A")
            obs_in_sample = observations_dict.get("obs_in_sample", len(errors_flat))
            lambda_val = general.get("lambda", 0)

            if error_type == "A":
                # Additive errors: normalize by y_denominator
                y_denom = general.get("y_denominator", 1)
                if y_denom is None or y_denom <= 0:
                    y_denom = 1  # Fallback to 1
                error_term = (
                    (1 - lambda_val)
                    * np.linalg.norm(errors_flat / y_denom)
                    / np.sqrt(obs_in_sample)
                )
                CFValue = error_term
            else:  # "M"
                # Multiplicative errors: use log(1 + errors)
                log_arg = 1 + errors_flat
                if np.any(log_arg <= 0):
                    CFValue = 1e100
                else:
                    error_term = (
                        (1 - lambda_val)
                        * np.linalg.norm(np.log(log_arg))
                        / np.sqrt(obs_in_sample)
                    )
                    CFValue = error_term

            # Add penalty term (LASSO = L1, RIDGE = L2)
            if general["loss"] == "LASSO":
                CFValue += lambda_val * np.sum(np.abs(B_penalty))
            else:  # "RIDGE"
                CFValue += lambda_val * np.linalg.norm(B_penalty)

        elif general["loss"] == "custom":
            # Ensure arrays are 1D to avoid broadcasting issues
            # (armadillo vectors are column vectors that may become (n,1) shaped arrays)
            fitted_1d = np.asarray(adam_fitted.fitted).ravel()
            CFValue = general["loss_function"](
                actual=y_in_sample, fitted=fitted_1d, B=B
            )
    else:
        # Multistep loss functions (MSEh, TMSE, GTMSE, MSCE, etc.)
        h = general["h"]
        obs_in_sample = observations_dict["obs_in_sample"]

        # Get multistep forecast errors using ferrors method
        error_result = adam_cpp.ferrors(
            matrixVt=adam_fitted.states,
            matrixWt=mat_wt,
            matrixF=mat_f,
            indexLookupTable=index_lookup_table,
            profilesRecent=profiles_recent_table,
            horizon=h,
            vectorYt=y_in_sample,
        )
        adam_errors = error_result.errors  # Matrix: (obs_in_sample - h) x h

        # Calculate loss based on type
        loss = general["loss"]
        CFValue = calculate_multistep_loss(loss, adam_errors, obs_in_sample, h)

    if np.isnan(CFValue) or np.isinf(CFValue):
        CFValue = 1e300

    return CFValue


def log_Lik_ADAM(  # noqa: N802
    B,
    model_type_dict,
    components_dict,
    lags_dict,
    adam_created,
    persistence_dict,
    initials_dict,
    arima_dict,
    explanatory_dict,
    phi_dict,
    constant_dict,
    observations_dict,
    occurrence_dict,
    general_dict,
    profile_dict,
    adam_cpp,
    multisteps=False,
    otherParameterEstimate=False,
):
    """
    Calculate log-likelihood for the ADAM model.

    This function computes the log-likelihood value for an ADAM model with given
    parameters.
    The log-likelihood is used for model selection (via information criteria) and for
    computing confidence intervals. The function handles various loss functions and can
    compute both one-step-ahead and multi-step-ahead likelihoods.

    The log-likelihood is calculated as:

    .. math::

        \\ell = -\\text{CF}(B)

    where CF is the cost function value. For occurrence models (intermittent data), the
    log-likelihood is augmented with the log-probability of the occurrence process:

    .. math::

        \\ell_{\\text{total}} = \\ell + \\sum_{t \\in \\mathcal{D}} \\log p_t +
                                \\sum_{t \\notin \\mathcal{D}} \\log(1-p_t)

    where :math:`\\mathcal{D}` is the set of time points with non-zero demand, and
    :math:`p_t` is the probability of occurrence at time t.

    Parameters
    ----------
    B : numpy.ndarray
        Parameter vector containing (in order):

        - ETS persistence parameters (α, β, γ)
        - Damping parameter (φ)
        - Initial states
        - ARIMA parameters (AR, MA coefficients)
        - Regression coefficients
        - Constant term
        - Distribution parameters (if estimated)

    model_type_dict : dict
        Model type specification containing:

        - 'error_type': Error type ('A' for additive, 'M' for multiplicative)
        - 'trend_type': Trend type ('N', 'A', 'Ad', 'M', 'Md')
        - 'season_type': Seasonality type ('N', 'A', 'M')
        - 'ets_model': Whether ETS components are present
        - 'arima_model': Whether ARIMA components are present

    components_dict : dict
        Components information containing:

        - 'components_number_ets': Total number of ETS components
        - 'components_number_ets_seasonal': Number of seasonal ETS components
        - 'components_number_arima': Number of ARIMA components

    lags_dict : dict
        Lag structure information containing:

        - 'lags': Vector of lags for each seasonal component
        - 'lags_model': Lags for each model component
        - 'lags_model_all': Complete lag specification
        - 'lags_model_max': Maximum lag value

    adam_created : dict
        State-space matrices from creator():

        - 'mat_vt': State vector matrix
        - 'mat_wt': Measurement matrix
        - 'mat_f': Transition matrix
        - 'vec_g': Persistence vector

    persistence_dict : dict
        Persistence parameters specification from checker()
    initials_dict : dict
        Initial values specification containing:

        - 'initial_type': Initialization method ('optimal', 'backcasting', 'complete')
        - 'n_iterations': Number of backcasting iterations

    arima_dict : dict
        ARIMA specification containing:

        - 'arima_model': Whether ARIMA is present
        - 'ar_estimate': Whether to estimate AR parameters
        - 'ma_estimate': Whether to estimate MA parameters
        - 'ar_required': Whether AR is required
        - 'ma_required': Whether MA is required

    explanatory_dict : dict
        External regressors specification containing:

        - 'xreg_model': Whether external regressors are present
        - 'xreg_number': Number of external regressors

    phi_dict : dict
        Damping parameter specification containing:

        - 'phi_estimate': Whether to estimate damping parameter
        - 'phi': Current damping parameter value

    constant_dict : dict
        Constant term specification containing:

        - 'constant_required': Whether a constant is included
        - 'constant_estimate': Whether to estimate the constant

    observations_dict : dict
        Observations information containing:

        - 'y_in_sample': In-sample time series values
        - 'ot': Occurrence variable (for intermittent data)
        - 'ot_logical': Boolean mask for non-zero observations
        - 'obs_in_sample': Number of in-sample observations
        - 'obs_zero': Number of zero observations

    occurrence_dict : dict
        Occurrence model information containing:

        - 'occurrence_model': Whether occurrence model is present
        - 'p_fitted': Fitted probabilities of occurrence

    general_dict : dict
        General model configuration containing:

        - 'loss': Loss function ('likelihood', 'MSE', 'MAE', 'HAM', 'LASSO', 'RIDGE',
        multistep variants)
        - 'distribution_new': Error distribution ('dnorm', 'dlaplace', 'ds', etc.)
        - 'h': Forecast horizon (for multistep losses)

    profile_dict : dict
        Profile matrices for time-varying parameters containing:

        - 'profiles_recent_table': Recent values for profile initialization
        - 'index_lookup_table': Index lookup for profile access

    multisteps : bool, optional
        Whether to use multi-step-ahead likelihood calculation (default: False).
        If True, computes likelihood based on multi-step forecasts.

    Returns
    -------
    float or dict
        Log-likelihood value. For standard likelihood calculation, returns a float.
        The value represents the natural logarithm of the likelihood function,
        where higher (less negative) values indicate better fit.

        For LASSO/RIDGE losses, returns 0 as these do not have a proper likelihood.

    Notes
    -----
    **Distribution Mapping**:

    For non-likelihood loss functions, the function maps to appropriate distributions:

    - MSE → Normal distribution (dnorm)
    - MAE → Laplace distribution (dlaplace)
    - HAM → S distribution (ds)

    **Multi-step Likelihood**:

    For multi-step loss functions (MSEh, MAEh, HAMh, etc.), concentrated likelihoods are
    computed:

    - MSEh, TMSE, MSCE: :math:`-\\frac{T-h}{2}(\\log(2\\pi) + 1 + \\log(\\text{loss}))`
    - MAEh, TMAE, MACE: :math:`-(T-h)(\\log(2) + 1 + \\log(\\text{loss}))`
    - HAMh, THAM, CHAM: :math:`-(T-h)(\\log(4) + 2 + 2\\log(\\text{loss}))`

    where T is the sample size and h is the forecast horizon.

    **Occurrence Model**:

    For intermittent data with occurrence models, the total likelihood combines:

    1. Conditional likelihood given non-zero demand
    2. Probability of occurrence/non-occurrence

    **Multiplicative Models**:

    For multiplicative error models in multistep context, the likelihood is adjusted by
    the Jacobian term: :math:`-\\sum_t \\log|y_t|` to account for the log
    transformation.

    See Also
    --------
    CF : Cost function used during optimization
    ic_function : Calculate information criteria from log-likelihood

    References
    ----------
    .. [1] Svetunkov, I. (2023). "Smooth forecasting with the smooth package in R".
           arXiv:2301.01790.
    .. [2] Snyder, R.D., Ord, J.K., Koehler, A.B., McLaren, K.R., and Beaumont, A.N.
    (2017).
           "Forecasting compositional time series: A state space approach".
           International Journal of Forecasting, 33(2), 502-512.

    Examples
    --------
    Calculate log-likelihood for estimated parameters::

        >>> loglik = log_Lik_ADAM(
        ...     B=estimated_params,
        ...     model_type_dict=model_type,
        ...     components_dict=components,
        ...     lags_dict=lags,
        ...     adam_created=matrices,
        ...     persistence_dict=persistence,
        ...     initials_dict=initials,
        ...     arima_dict=arima,
        ...     explanatory_dict=explanatory,
        ...     phi_dict=phi,
        ...     constant_dict=constants,
        ...     observations_dict=observations,
        ...     occurrence_dict=occurrence,
        ...     general_dict=general,
        ...     profile_dict=profile
        ... )
        >>> print(f"Log-likelihood: {loglik}")

    For multi-step likelihood::

        >>> loglik_multistep = log_Lik_ADAM(
        ...     B=estimated_params,
        ...     ...,
        ...     multisteps=True
        ... )
    """

    if not multisteps:
        # print(profile_dict)
        if general_dict["loss"] in ["LASSO", "RIDGE"]:
            return 0
        else:
            # The reported log-likelihood for a fit-only loss is the
            # concentrated likelihood, NOT -loss. Remap the loss to
            # "likelihood" and let CF compute it (CF reads general["loss"]/
            # ["distribution_new"], so the remap must land on those keys — a
            # copy, general_dict is shared). distribution_new is already
            # resolved upstream (the user's explicit choice, or the loss-implied
            # default: MSE->dnorm, MAE->dlaplace, HAM->ds), so it is used as-is:
            # an explicitly-selected distribution is honoured for the logLik
            # even when the loss implies a different one; only the default falls
            # back to the loss-implied distribution. Mirrors R/adam.R:1009-1017.
            general_for_ll = dict(general_dict)
            general_for_ll["loss"] = (
                "likelihood"
                if general_dict["loss"] in ["MSE", "MAE", "HAM"]
                else general_dict["loss"]
            )

            # Call CF function with bounds="none"
            logLikReturn = -CF(
                B,
                model_type_dict,
                components_dict,
                lags_dict,
                adam_created,
                persistence_dict,
                initials_dict,
                arima_dict,
                explanatory_dict,
                phi_dict,
                constant_dict,
                observations_dict,
                profile_dict,
                general_for_ll,
                adam_cpp,
                bounds=None,
                otherParameterEstimate=otherParameterEstimate,
            )

            # Handle occurrence model
            if occurrence_dict["occurrence_model"]:
                if np.isinf(logLikReturn):
                    logLikReturn = 0
                if any(
                    1 - occurrence_dict["p_fitted"][~observations_dict["ot_logical"]]
                    == 0
                ) or any(
                    occurrence_dict["p_fitted"][observations_dict["ot_logical"]] == 0
                ):
                    pt_new = occurrence_dict["p_fitted"][
                        (occurrence_dict["p_fitted"] != 0)
                        & (occurrence_dict["p_fitted"] != 1)
                    ]
                    ot_new = observations_dict["ot"][
                        (occurrence_dict["p_fitted"] != 0)
                        & (occurrence_dict["p_fitted"] != 1)
                    ]
                    if len(pt_new) == 0:
                        return logLikReturn
                    else:
                        return (
                            logLikReturn
                            + np.sum(np.log(pt_new[ot_new == 1]))
                            + np.sum(np.log(1 - pt_new[ot_new == 0]))
                        )
                else:
                    return (
                        logLikReturn
                        + np.sum(
                            np.log(
                                occurrence_dict["p_fitted"][
                                    observations_dict["ot_logical"]
                                ]
                            )
                        )
                        + np.sum(
                            np.log(
                                1
                                - occurrence_dict["p_fitted"][
                                    ~observations_dict["ot_logical"]
                                ]
                            )
                        )
                    )
            else:
                return logLikReturn

    else:
        # Call CF function with bounds="none"
        logLikReturn = CF(
            B,
            model_type_dict,
            components_dict,
            lags_dict,
            adam_created,
            persistence_dict,
            initials_dict,
            arima_dict,
            explanatory_dict,
            phi_dict,
            constant_dict,
            observations_dict,
            profile_dict,
            general_dict,
            adam_cpp,
            bounds=None,
        )

        # Concentrated log-likelihoods for the multistep losses
        if general_dict["loss"] in ["MSEh", "aMSEh", "TMSE", "aTMSE", "MSCE", "aMSCE"]:
            # is horizon different than h?
            logLikReturn = (
                -(observations_dict["obs_in_sample"] - general_dict["h"])
                / 2
                * (np.log(2 * np.pi) + 1 + np.log(logLikReturn))
            )
        elif general_dict["loss"] in ["GTMSE", "aGTMSE"]:
            logLikReturn = (
                -(observations_dict["obs_in_sample"] - general_dict["h"])
                / 2
                * (np.log(2 * np.pi) + 1 + logLikReturn)
            )
        elif general_dict["loss"] in ["MAEh", "TMAE", "GTMAE", "MACE"]:
            logLikReturn = -(observations_dict["obs_in_sample"] - general_dict["h"]) * (
                np.log(2) + 1 + np.log(logLikReturn)
            )
        elif general_dict["loss"] in ["HAMh", "THAM", "GTHAM", "CHAM"]:
            logLikReturn = -(observations_dict["obs_in_sample"] - general_dict["h"]) * (
                np.log(4) + 2 + 2 * np.log(logLikReturn)
            )
        elif general_dict["loss"] in ["GPL", "aGPL"]:
            logLikReturn = (
                -(observations_dict["obs_in_sample"] - general_dict["h"])
                / 2
                * (
                    general_dict["h"] * np.log(2 * np.pi)
                    + general_dict["h"]
                    + logLikReturn
                )
                / general_dict["h"]
            )

        # Make likelihood comparable
        logLikReturn = (
            logLikReturn
            / (observations_dict["obs_in_sample"] - general_dict["h"])
            * observations_dict["obs_in_sample"]
        )

        # Handle multiplicative model
        if model_type_dict["ets_model"] and model_type_dict["error_type"] == "M":
            # Fill in the matrices
            adam_elements = filler(
                B,
                model_type_dict,
                components_dict,
                lags_dict,
                adam_created,
                persistence_dict,
                initials_dict,
                arima_dict,
                explanatory_dict,
                phi_dict,
                constant_dict,
                adam_cpp,
            )

            # Write down the initials in the recent profile
            profile_dict["profiles_recent_table"][:] = adam_elements["mat_vt"][
                :, : lags_dict["lags_model_max"]
            ]

            # Fit the model again to extract the fitted values.
            # Check if initial_type is a list or string and compute backcast correctly
            if isinstance(initials_dict["initial_type"], list):
                backcast_value_log = any(
                    [
                        t == "complete" or t == "backcasting"
                        for t in initials_dict["initial_type"]
                    ]
                )
            else:
                backcast_value_log = initials_dict["initial_type"] in [
                    "complete",
                    "backcasting",
                ]

            # Convert stuff to numpy arrays with float64 - C++ requires that
            y_in_sample = np.asarray(observations_dict["y_in_sample"], dtype=np.float64)
            ot = np.asarray(observations_dict["ot"], dtype=np.float64)
            # Explicit copy -- see the matching comment in CF()
            mat_vt = np.array(adam_elements["mat_vt"], dtype=np.float64, order="F")
            mat_wt = np.asfortranarray(adam_elements["mat_wt"], dtype=np.float64)
            mat_f = np.asfortranarray(
                adam_elements["mat_f"], dtype=np.float64
            )  # Also copy mat_f since it's passed by reference
            vec_g = np.asfortranarray(
                adam_elements["vec_g"], dtype=np.float64
            )  # Make sure it's a 1D array
            index_lookup_table = np.asfortranarray(
                profile_dict["index_lookup_table"], dtype=np.uint64
            )
            profiles_recent_table = np.asfortranarray(
                profile_dict["profiles_recent_table"], dtype=np.float64
            )

            adam_fitted = adam_fit_or_gradient(
                adam_cpp=adam_cpp,
                mat_vt=mat_vt,
                mat_wt=mat_wt,
                mat_f=mat_f,
                vec_g=vec_g,
                index_lookup_table=index_lookup_table,
                profiles_recent_table=profiles_recent_table,
                y_in_sample=y_in_sample,
                ot=ot,
                initial_type=initials_dict["initial_type"],
                n_iterations=initials_dict["n_iterations"],
                backcast_value=backcast_value_log,
                model_type_dict=model_type_dict,
                components_dict=components_dict,
                lags_dict=lags_dict,
                obs_in_sample=observations_dict["obs_in_sample"],
                loss=general_dict["loss"],
                distribution=general_dict.get(
                    "distribution_new", general_dict.get("distribution", "default")
                ),
                # This refit only runs for the multistep losses, whose codes do
                # not use the distribution parameter
                other=None,
                horizon=general_dict.get("h", 0),
                multisteps=general_dict["multisteps"],
            )

            logLikReturn -= np.sum(np.log(np.abs(adam_fitted.fitted)))

        return logLikReturn
