import numpy as np

from smooth.adam_general._eigenCalc import smooth_eigens
from smooth.adam_general.core.creator import filler
from smooth.adam_general.core.utils.gradient import adam_fit_or_gradient
from smooth.adam_general.core.utils.polynomials import arima_bounds_penalty
from smooth.adam_general.core.utils.utils import (
    _log_r,
    _sum_r,
    calculate_entropy,
    calculate_likelihood,
    calculate_multistep_loss,
    complete_windows,
    multistep_log_lik,
    observed_mask,
    scaler,
)


def lasso_denominators(loss, mat_wt, components_dict, xreg_number, y_in_sample):
    """The scales of LASSO / RIDGE (R's ``adam_lassoDenominators``).

    The standard deviations of the explanatory variables (``denominator``), which
    normalise their parameters, and of the differenced series (``y_denominator``),
    which normalises the errors of ADAM; None for the other losses.
    """
    result = {"denominator": None, "y_denominator": None}
    if loss in ("LASSO", "RIDGE"):
        if xreg_number > 0:
            start = components_dict["components_number_ets"] + components_dict.get(
                "components_number_arima", 0
            )
            denominator = np.std(
                np.asarray(mat_wt)[:, start : start + xreg_number], axis=0, ddof=1
            )
            denominator[np.isinf(denominator)] = 1
            result["denominator"] = denominator
        result["y_denominator"] = max(
            np.std(np.diff(np.asarray(y_in_sample, dtype=float).ravel()), ddof=1), 1
        )
    return result


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
    constant_estimate=False,
    other_parameter_estimate=False,
):
    """The parameters of the LASSO / RIDGE penalty (R's CF in R/adam.R).

    The estimated parameters, found in ``B`` as the filler does, shifted to be zero
    at their shrinkage targets: no smoothing, ``phi`` and AR of one, MA and
    regressors of zero, the additive regressors normalised by the
    ``denominator``. The initial states are not shrunk.

    Shared by ADAM's ``CF``, ``om_cf`` (for ``OM``) and ``omg_cf`` (for ``OMG``).
    """
    B = np.asarray(B, dtype=float)  # noqa: N806
    ets_model = model_type_dict["ets_model"]
    arima_model = bool(arima_checked and arima_checked.get("arima_model", False))
    xreg_model = bool(explanatory_checked.get("xreg_model", False))

    n_smoothing = 0
    if persistence_checked.get("persistence_estimate", False):
        if ets_model:
            n_smoothing += int(persistence_checked["persistence_level_estimate"])
            if model_type_dict["model_is_trendy"]:
                n_smoothing += int(persistence_checked["persistence_trend_estimate"])
            if model_type_dict["model_is_seasonal"]:
                n_smoothing += int(
                    np.sum(persistence_checked["persistence_seasonal_estimate"])
                )
        if xreg_model and persistence_checked.get("persistence_xreg_estimate", False):
            n_smoothing += max(explanatory_checked["xreg_parameters_persistence"])
    n_phi = int(bool(ets_model and phi_dict.get("phi_estimate", False)))

    ar_estimated: list = []
    if arima_model:
        ar_est = bool(arima_checked["ar_estimate"])
        ma_est = bool(arima_checked["ma_estimate"])
        for ar_order, ma_order in zip(
            arima_checked["ar_orders"], arima_checked["ma_orders"]
        ):
            ar_estimated += [True] * (ar_est * ar_order) + [False] * (ma_est * ma_order)
    arma = B[n_smoothing + n_phi : n_smoothing + n_phi + len(ar_estimated)]
    ar_mask = np.asarray(ar_estimated, dtype=bool)

    n_xreg = 0
    xreg_estimated = None
    if (
        xreg_model
        and initials_checked.get("initial_type") != "complete"
        and initials_checked.get("initial_estimate", False)
        and initials_checked.get("initial_xreg_estimate", False)
    ):
        xreg_estimated = (
            np.asarray(explanatory_checked["xreg_parameters_estimated"], dtype=int) == 1
        )
        n_xreg = int(xreg_estimated.sum())
    end = len(B) - int(other_parameter_estimate) - int(constant_estimate)
    xreg = B[end - n_xreg : end]
    denominator = general.get("denominator")
    if (
        n_xreg > 0
        and model_type_dict.get("error_type") == "A"
        and denominator is not None
    ):
        xreg = xreg / np.asarray(denominator, dtype=float)[xreg_estimated]

    return np.concatenate(
        [
            B[:n_smoothing],
            1 - B[n_smoothing : n_smoothing + n_phi],
            1 - arma[ar_mask],
            arma[~ar_mask],
            xreg,
        ]
    )


def adam_bounds_checker(
    adam_elements,
    bounds,
    model_type_dict,
    components_dict,
    lags_dict,
    arima_checked,
    explanatory_checked,
    phi_dict,
    constants_checked,
    obs_in_sample,
):
    """Penalty for parameters outside the bounds, or 0 (R's ``adam_bounds_checker``).

    Shared by ``CF``, ``om_cf`` and ``omg_cf``: stationarity / invertibility of
    the ARIMA factors, then the ``"usual"`` restrictions of ETS and of the
    ``regressors="adapt"`` smoothing parameters, or the ``"admissible"`` check of
    the discount matrix of the non-ARIMA components.
    """
    adapt = explanatory_checked.get("regressors") == "adapt"
    # Stationary AR and invertible MA, factor by factor
    if bounds != "none":
        penalty = arima_bounds_penalty(
            arima_checked, adam_elements["arima_polynomials"]
        )
        if penalty > 0:
            return penalty

    if bounds == "usual":
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

        if explanatory_checked["xreg_model"] and adapt:
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
        # The discount matrix split by lags is meaningless for the lagged ARIMA,
        # which has one state per lag: its states are checked above instead
        n_ets = components_dict["components_number_ets"]
        components_other = np.setdiff1d(
            np.arange(len(lags_dict["lags_model_all"])),
            np.arange(n_ets, n_ets + components_dict["components_number_arima"]),
        )
        if components_other.size > 0 and (
            model_type_dict["ets_model"] or arima_checked["arima_model"]
        ):
            has_delta = explanatory_checked["xreg_model"] and adapt
            eigenValues = smooth_eigens(
                persistence=np.asfortranarray(
                    adam_elements["vec_g"][components_other].reshape(-1, 1),
                    dtype=np.float64,
                ),
                transition=np.asfortranarray(
                    adam_elements["mat_f"][np.ix_(components_other, components_other)],
                    dtype=np.float64,
                ),
                measurement=np.asfortranarray(
                    adam_elements["mat_wt"][:, components_other], dtype=np.float64
                ),
                lags_model_all=np.asarray(lags_dict["lags_model_all"], dtype=np.int32)[
                    components_other
                ],
                xreg_model=explanatory_checked["xreg_model"],
                obs_in_sample=obs_in_sample,
                has_delta=has_delta,
                xreg_number=explanatory_checked["xreg_number"],
                constant_required=constants_checked["constant_required"],
            )
            if np.any(eigenValues > 1 + 1e-50):
                return 1e100 * np.max(eigenValues)

    return 0


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

    # Capture the filled state for C++ and the profiles
    profile_dict["profiles_recent_table"][:] = adam_elements["mat_vt"][
        :, : lags_dict["lags_model_max"]
    ]
    # An explicit copy: the gradient dispatcher overwrites the head of mat_vt
    # in place, and np.asfortranarray would alias an already-Fortran-ordered
    # array, leaking the solved initials into the shared creator matrices
    # (R's copy-on-modify semantics never leak them)
    mat_vt = np.array(adam_elements["mat_vt"], dtype=np.float64, order="F")

    # If we estimate parameters of distribution, take it from the B vector
    if otherParameterEstimate:
        other = abs(B[-1])
        if general["distribution_new"] in ["dgnorm", "dlgnorm"] and other < 0.25:
            return 1e10 / other
    # Check the bounds, classical restrictions
    # print(components_dict['components_number_ets_non_seasonal'])

    penalty = adam_bounds_checker(
        adam_elements,
        bounds,
        model_type_dict,
        components_dict,
        lags_dict,
        arima_checked,
        explanatory_checked,
        phi_dict,
        constants_checked,
        observations_dict["obs_in_sample"],
    )
    if penalty > 0:
        return penalty

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
        xreg_number=int(explanatory_checked.get("xreg_number", 0) or 0),
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

    # The missing values are not in the loss: the errors are zero there, and the
    # losses are divided by the observed values
    observed = observed_mask(observations_dict)
    obs_observed = int(np.sum(observed))
    if not general["multisteps"]:
        if general["loss"] == "likelihood":
            scale = scaler(
                general["distribution_new"],
                model_type_dict["error_type"],
                adam_fitted.errors[observations_dict["ot_logical"]],
                adam_fitted.fitted[observations_dict["ot_logical"]],
                obs_observed,
                other,
            )
            # Aggregate through _sum_r: R accumulates sum() in a long double
            # register, and NumPy's default pairwise sum drifts by 1 ULP per
            # call against it, which through NLopt's deterministic simplex
            # moves becomes a different convergence point on flat ARIMA cost
            # surfaces.  _sum_r *is* that accumulator, so it mirrors R more
            # literally than exact summation would -- and it stays vectorised,
            # where math.fsum(...tolist()) walked the array in Python.
            ll = calculate_likelihood(
                general["distribution_new"],
                model_type_dict["error_type"],
                observations_dict["y_in_sample"][observations_dict["ot_logical"]],
                adam_fitted.fitted[observations_dict["ot_logical"]],
                scale,
                other,
            )
            CFValue = -_sum_r(np.asarray(ll, dtype=np.float64).ravel())
            # Differential entropy for the logLik of occurrence model, over the
            # observed zeros: a missing observation is not a zero
            ot_zero = ~observations_dict["ot_logical"]
            if observations_dict.get("y_na_values") is not None:
                ot_zero = ot_zero & ~observations_dict["y_na_values"]
            if observations_dict.get("occurrence_model", False) or any(ot_zero):
                CFValueEntropy = calculate_entropy(
                    general["distribution_new"],
                    scale,
                    other,
                    int(np.sum(ot_zero)),
                    adam_fitted.fitted[ot_zero],
                )
                # NaN means something is wrong; a negative entropy (it should not
                # be) becomes zero, so that occurrence does not distort the sizes
                if np.isnan(CFValueEntropy):
                    CFValueEntropy = np.inf
                elif CFValueEntropy < 0:
                    CFValueEntropy = 0.0
                CFValue += CFValueEntropy

        elif general["loss"] == "MSE":
            CFValue = _sum_r(adam_fitted.errors**2) / obs_observed
        elif general["loss"] == "MAE":
            CFValue = _sum_r(np.abs(adam_fitted.errors)) / obs_observed
        elif general["loss"] == "HAM":
            CFValue = _sum_r(np.sqrt(np.abs(adam_fitted.errors))) / obs_observed
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
                constants_checked["constant_estimate"],
                otherParameterEstimate,
            )

            # Flatten errors to avoid potential broadcasting issues
            errors_flat = np.asarray(adam_fitted.errors).ravel()

            # Calculate error term based on error type
            error_type = model_type_dict.get("error_type", "A")
            obs_in_sample = obs_observed
            lambda_val = general.get("lambda", 0)

            if error_type == "A":
                # Additive errors: normalize by y_denominator
                y_denom = general.get("y_denominator", 1)
                if y_denom is None or y_denom <= 0:
                    y_denom = 1  # Fallback to 1
                # R's sqrt(sum(x^2)/n): np.linalg.norm rounds differently
                error_term = (1 - lambda_val) * np.sqrt(
                    _sum_r((errors_flat / y_denom) ** 2) / obs_in_sample
                )
                CFValue = error_term
            else:  # "M"
                # Multiplicative errors: use log(1 + errors)
                log_arg = 1 + errors_flat
                if np.any(log_arg <= 0):
                    CFValue = 1e100
                else:
                    error_term = (1 - lambda_val) * np.sqrt(
                        _sum_r(_log_r(log_arg) ** 2) / obs_in_sample
                    )
                    CFValue = error_term

            # Add penalty term (LASSO = L1, RIDGE = L2)
            if general["loss"] == "LASSO":
                CFValue += lambda_val * _sum_r(np.abs(B_penalty))
            else:  # "RIDGE"
                CFValue += lambda_val * np.sqrt(_sum_r(np.asarray(B_penalty) ** 2))

        elif general["loss"] == "custom":
            # Ensure arrays are 1D to avoid broadcasting issues
            # (armadillo vectors are column vectors that may become (n,1) shaped arrays)
            fitted_1d = np.asarray(adam_fitted.fitted).ravel()
            CFValue = general["loss_function"](
                actual=np.asarray(y_in_sample).ravel()[observed],
                fitted=fitted_1d[observed],
                B=B,
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
        adam_errors = error_result.errors  # Matrix: (obs_in_sample - h + 1) x h
        # The windows with all their targets observed
        adam_errors = adam_errors[complete_windows(observed, h)]

        # Calculate loss based on type
        loss = general["loss"]
        CFValue = calculate_multistep_loss(loss, adam_errors, len(adam_errors) + h, h)

    # A perfect fit (-inf) is kept, as in R
    if np.isnan(CFValue) or CFValue == np.inf:
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

    where T-h is replaced by the number of windows with all their targets observed
    (T-h+1 without missing values).

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

            # Handle occurrence model: the observed zeros, a missing value is not in
            # the likelihood
            if occurrence_dict["occurrence_model"]:
                if np.isinf(logLikReturn):
                    logLikReturn = 0
                p_fitted = occurrence_dict["p_fitted"]
                ot_logical = observations_dict["ot_logical"]
                observed = observed_mask(observations_dict)
                zero = ~ot_logical & observed
                if any(1 - p_fitted[zero] == 0) or any(p_fitted[ot_logical] == 0):
                    usable = (p_fitted != 0) & (p_fitted != 1) & observed
                    pt_new = p_fitted[usable]
                    ot_new = observations_dict["ot"][usable]
                    if len(pt_new) == 0:
                        return logLikReturn
                    else:
                        return (
                            logLikReturn
                            + _sum_r(_log_r(pt_new[ot_new == 1]))
                            + _sum_r(_log_r(1 - pt_new[ot_new == 0]))
                        )
                else:
                    return (
                        logLikReturn
                        + _sum_r(_log_r(p_fitted[ot_logical]))
                        + _sum_r(_log_r(1 - p_fitted[zero]))
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

        # Concentrated log-likelihoods for the multistep losses, over the windows
        # with all their targets observed
        logLikReturn = multistep_log_lik(
            logLikReturn,
            general_dict["loss"],
            general_dict["h"],
            observed_mask(observations_dict),
        )

        # Handle multiplicative model
        if model_type_dict["ets_model"] and model_type_dict["error_type"] == "M":
            # This refit only runs for the multistep losses, whose codes do not
            # use the distribution parameter
            adam_fitted = fit_at_parameters(
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
                general_dict,
                profile_dict,
                adam_cpp,
            )

            logLikReturn -= _sum_r(_log_r(np.abs(adam_fitted.fitted)))

        return logLikReturn


def fit_at_parameters(
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
    general_dict,
    profile_dict,
    adam_cpp,
    other=None,
):
    """The fit of the model at the parameters B, as R's ``filler()`` followed by
    ``adam_fitOrGradient()``: the initials are written into the recent profile."""
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
    profile_dict["profiles_recent_table"][:] = adam_elements["mat_vt"][
        :, : lags_dict["lags_model_max"]
    ]
    initial_type = initials_dict["initial_type"]
    types = initial_type if isinstance(initial_type, list) else [initial_type]
    # Explicit copies: the C++ fitter writes into its arguments
    return adam_fit_or_gradient(
        adam_cpp=adam_cpp,
        mat_vt=np.array(adam_elements["mat_vt"], dtype=np.float64, order="F"),
        mat_wt=np.asfortranarray(adam_elements["mat_wt"], dtype=np.float64),
        mat_f=np.asfortranarray(adam_elements["mat_f"], dtype=np.float64),
        vec_g=np.asfortranarray(adam_elements["vec_g"], dtype=np.float64),
        index_lookup_table=np.asfortranarray(
            profile_dict["index_lookup_table"], dtype=np.uint64
        ),
        profiles_recent_table=np.asfortranarray(
            profile_dict["profiles_recent_table"], dtype=np.float64
        ),
        y_in_sample=np.asarray(observations_dict["y_in_sample"], dtype=np.float64),
        ot=np.asarray(observations_dict["ot"], dtype=np.float64),
        initial_type=initial_type,
        n_iterations=initials_dict["n_iterations"],
        backcast_value=any(t in ("complete", "backcasting") for t in types),
        model_type_dict=model_type_dict,
        components_dict=components_dict,
        lags_dict=lags_dict,
        obs_in_sample=observations_dict["obs_in_sample"],
        loss=general_dict["loss"],
        distribution=general_dict.get(
            "distribution_new", general_dict.get("distribution", "default")
        ),
        other=other,
        horizon=general_dict.get("h", 0),
        multisteps=general_dict["multisteps"],
        xreg_number=int(explanatory_dict.get("xreg_number", 0) or 0),
    )
