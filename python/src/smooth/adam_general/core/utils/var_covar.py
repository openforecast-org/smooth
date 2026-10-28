import greybox as gb
import numpy as np


def filter_variances(
    adam_cpp,
    lags_model_all,
    head_length,
    mat_wt,
    mat_f,
    vec_g,
    profile,
    ot_logical,
    h=0,
    logs=False,
    power=2,
    noise=None,
):
    """The variances of the errors relative to the one-step one (or with ``power=1``
    the multipliers of their means in logs) from the filter of ``adam_cpp.filter``
    over the periods without an observed value (R's ``adam_gapVarianceModel``). They
    do not depend on the data or on the initial states, so the filter runs on ones
    from ``profile`` without backcasting; the conventional ETS needs the moments of
    its errors at the scale (``noise`` of :func:`gap_drift`). With ``h>0``, the h
    periods after the sample are unobserved, with the last measurement, and their
    values are returned: those of the forecast."""
    # Imported here: the creator imports this module
    from smooth.adam_general.core.creator import adam_profile_creator

    ot = np.asarray(ot_logical, dtype=bool).ravel()
    obs = ot.size
    lags_model_max = int(np.max(lags_model_all))
    head_length = max(int(head_length), lags_model_max)
    rows = np.r_[np.arange(obs), np.repeat(obs - 1, h)]
    lookup = adam_profile_creator(
        lags_model_all=lags_model_all,
        lags_model_max=lags_model_max,
        obs_all=obs + h,
        head_length=head_length,
    )["index_lookup_table"]
    profile = np.asarray(profile, dtype=float)[:, :lags_model_max]
    filtered = adam_cpp.filter(
        matrixVt=np.zeros((profile.shape[0], obs + h + head_length), order="F"),
        matrixWt=np.asfortranarray(np.asarray(mat_wt, dtype=float)[rows]),
        matrixF=np.asfortranarray(mat_f, dtype=float),
        vectorG=np.asfortranarray(vec_g, dtype=float).reshape(-1, 1),
        indexLookupTable=np.asfortranarray(lookup, dtype=np.uint64),
        profilesRecent=np.asfortranarray(profile),
        vectorYt=np.ones(obs + h),
        vectorOt=np.r_[ot, np.zeros(h, dtype=bool)].astype(float),
        backcast=False,
        nIterations=1,
        logs=bool(logs),
        scale=0.0,
        noiseMean=np.empty(0) if noise is None else np.asarray(noise[0], dtype=float),
        noiseCovariance=(
            np.empty((0, 0))
            if noise is None
            else np.asfortranarray(noise[1], dtype=float)
        ),
    )
    values = np.ravel(
        2 * np.asarray(filtered.innovationsScale) if power == 1 else filtered.variances
    )
    return values[obs:] if h > 0 else values


# The Gauss-Legendre rule with 32 nodes on (0,1), for the moments of a function of the
# error through its quantiles: the same numbers as R's adam_gaussLegendre
_GAUSS_LEGENDRE_NODES = np.array(
    [
        0.001368069075259215,
        0.007194244227365809,
        0.017618872206246805,
        0.03254696203113017,
        0.051839422116973954,
        0.07531619313371501,
        0.10275810201602881,
        0.13390894062985514,
        0.16847786653489238,
        0.20614212137961885,
        0.24655004553388532,
        0.28932436193468236,
        0.33406569885893617,
        0.38035631887393145,
        0.42776401920860174,
        0.4758461671561308,
        0.5241538328438692,
        0.5722359807913983,
        0.6196436811260685,
        0.6659343011410639,
        0.7106756380653176,
        0.7534499544661146,
        0.7938578786203812,
        0.8315221334651076,
        0.8660910593701449,
        0.8972418979839711,
        0.924683806866285,
        0.948160577883026,
        0.9674530379688698,
        0.9823811277937532,
        0.9928057557726342,
        0.9986319309247408,
    ]
)
_GAUSS_LEGENDRE_WEIGHTS = np.array(
    [
        0.003509305004735253,
        0.008137197365452872,
        0.012696032654631012,
        0.017136931456510882,
        0.021417949011113418,
        0.025499029631188046,
        0.029342046739267783,
        0.03291111138818084,
        0.03617289705442417,
        0.039096947893535114,
        0.041655962113473353,
        0.04382604650220189,
        0.04558693934788189,
        0.046922199540402255,
        0.047819360039637354,
        0.04827004425736383,
        0.04827004425736383,
        0.047819360039637354,
        0.046922199540402255,
        0.04558693934788189,
        0.04382604650220189,
        0.041655962113473353,
        0.039096947893535114,
        0.03617289705442417,
        0.03291111138818084,
        0.029342046739267783,
        0.025499029631188046,
        0.021417949011113418,
        0.017136931456510882,
        0.012696032654631012,
        0.008137197365452872,
        0.003509305004735253,
    ]
)


def gap_drift(
    vec_g, ot_logical, distribution, adam_ets, scale, filtered=None, noise=None
):
    """The moments of the errors of a pure multiplicative ETS in logs at the scale,
    and the drift of its states after the periods without an observed size, as R's
    ``adam_gapDrift()``.

    The noise of ``adam_cpp.filter`` is the mean and the covariance of
    ``u = log(1+e)`` and of the innovations of the states, ``g u`` for ADAM ETS and
    ``log(1 + g e)`` for the conventional one, from the quantiles of ``1+e`` at the
    Gauss-Legendre nodes. With the filter, the mean and the variance of the
    logarithm of the drift of the states at the observed sizes are
    ``N(mu (m-1), v (f-1))`` for ADAM ETS, whose filter runs with the error of mean
    zero and the variance one (``f`` and ``m`` being its variances and multipliers
    of the mean, and ``mu`` and ``v`` those of ``u``), and ``N(0, f - v)`` for the
    conventional one, whose filter runs with these moments. Returns the mean, the
    variance and the noise.
    """
    from smooth.adam_general.core.utils.utils import _log_r, _sum_r

    ot = np.asarray(ot_logical, dtype=bool).ravel()
    if noise is None:
        vec_g = np.asarray(vec_g, dtype=float).ravel()
        nodes = _GAUSS_LEGENDRE_NODES
        if distribution == "dgamma":
            quantiles = gb.qgamma(nodes, shape=1 / scale, scale=scale)
        elif distribution == "dinvgauss":
            quantiles = gb.qinvgauss(nodes, 1.0, scale)
        else:
            quantiles = gb.qlnorm(nodes, -scale / 2, np.sqrt(scale))
        quantiles = np.asarray(quantiles, dtype=float)
        errors = _log_r(quantiles)
        if adam_ets:
            states = errors[:, None] * vec_g[None, :]
        else:
            states = _log_r(1 + (quantiles - 1)[:, None] * vec_g[None, :])
        innovations = np.column_stack([errors, states])
        # Their means and covariances by the sums of R
        weights = _GAUSS_LEGENDRE_WEIGHTS
        noise_mean = _sum_r(weights[:, None] * innovations, axis=0)
        n = innovations.shape[1]
        noise_covariance = np.array(
            [
                [
                    _sum_r(weights * innovations[:, i] * innovations[:, j])
                    - noise_mean[i] * noise_mean[j]
                    for j in range(n)
                ]
                for i in range(n)
            ]
        )
        noise = (noise_mean, noise_covariance)
    mean = np.zeros(int(ot.sum()))
    variance = np.zeros(int(ot.sum()))
    if filtered is not None:
        variances = np.ravel(filtered.variances)[ot]
        if adam_ets:
            multipliers = 2 * np.ravel(filtered.innovationsScale)[ot]
            mean = noise[0][0] * (multipliers - 1)
            variance = noise[1][0, 0] * (variances - 1)
        else:
            variance = variances - noise[1][0, 0]
    return mean, variance, noise


def covar_anal(lags_model, h, measurement, transition, persistence, s2):
    """The analytical covariance matrix of the 1..h steps ahead errors (R's
    ``covarAnal()``), from the C++ shared with R (``src/headers/covarAnalCore.h``).

    Parameters:
    - lags_model: model lags assigned to each state (e.g., [1, 1, 12])
    - h: forecast horizon
    - measurement: measurement matrix, of which the first row is used
    - transition: transition matrix (n_components x n_components)
    - persistence: persistence vector (n_components,)
    - s2: one-step-ahead variance

    Returns:
    - covar_mat: covariance matrix (h x h)
    """
    from smooth.adam_general._adamCore import covar_anal_cpp

    # Owned Fortran-ordered copies, which carma takes as Armadillo matrices
    def fortran(x):
        return np.array(x, dtype=np.float64, order="F", copy=True)

    return np.asarray(
        covar_anal_cpp(
            fortran(np.ravel(lags_model)),
            int(h),
            fortran(np.atleast_2d(measurement)),
            fortran(transition),
            fortran(np.ravel(persistence)),
            float(s2),
        )
    )


def var_anal(lags_model, h, measurement, transition, persistence, s2):
    """
    Returns variances for the multiplicative error ETS models. Corrected Python version.

    Parameters:
    - lags_model: list or array, model lags assigned to each state (e.g., [1, 1, 12])
    - h: int, forecast horizon
    - measurement: array, measurement vector (Should be 1D, shape (n_components,))
    - transition: array, transition matrix (n_components x n_components)
    - persistence: array, persistence vector (k_states,)
    - s2: float, one-step-ahead variance

    Returns:
    - var_mat: array, variance vector (h,)
    """
    # Ensure inputs are numpy arrays and persistence is 1D
    lags_model = np.array(lags_model)
    measurement = np.array(measurement)
    transition = np.array(transition)
    persistence = np.array(persistence).flatten()  # Ensure persistence is 1D

    # Prepare the necessary parameters
    lags_unique = np.unique(
        lags_model
    )  # All unique lags present in the model definition
    steps = np.sort(
        lags_unique[lags_unique <= h]
    )  # Unique lags <= horizon h used for array_persistence_q
    steps_number = len(steps)
    n_components = transition.shape[0]  # Number of rows/cols in transition matrix
    k_states = len(persistence)  # Number of state components

    # --- Input dimension validation (optional but recommended) ---
    if n_components != transition.shape[1]:
        raise ValueError("Transition matrix must be square.")
    # Ensure measurement is treated as 1D for validation
    if measurement.ndim > 1:
        # Attempt to flatten if it makes sense (e.g., row or column vector)
        if measurement.size == n_components:
            measurement = measurement.flatten()
        else:
            raise ValueError(
                f"Measurement shape {measurement.shape} cannot be flattened "
                f"to match n_components {n_components}."
            )
    if measurement.ndim != 1 or measurement.shape[0] != n_components:
        raise ValueError(
            f"Measurement shape {measurement.shape} incompatible with "
            f"n_components {n_components}. Expecting ({n_components},)."
        )
    if lags_model.shape[0] != k_states:
        raise ValueError(
            f"lags_model length {lags_model.shape[0]} "
            f"must match persistence length {k_states}."
        )
    if k_states != n_components:
        raise ValueError(
            f"Number of states ({k_states}) from persistence vector does not "
            f"match transition matrix dimension ({n_components}). "
            "Check model definition."
        )
    # --- End Validation ---

    # Prepare the persistence array (array_persistence_q)
    # This array stores diagonal persistence matrices sliced according to steps
    array_persistence_q = np.zeros((n_components, n_components, steps_number))
    # Use the already flattened persistence array here
    diag_matrix_full = np.diag(persistence)  # Now guaranteed to be k_states x k_states

    for i_step in range(steps_number):
        mask = lags_model == steps[i_step]  # Boolean array of length k_states
        if np.sum(mask) > 0:
            # Assign relevant columns from diag_matrix_full to the slice
            # This indexing should now work correctly
            array_persistence_q[:, mask, i_step] = diag_matrix_full[:, mask]

    ## The matrices that will be used in the loop
    matrix_persistence_q = np.zeros((n_components, n_components))
    iq = np.zeros(1)  # Accumulator, use array element for direct update
    ik = np.eye(k_states)  # Identity matrix

    # The vector of variances, initialized to zeros (like R's rep(0,h))
    var_mat = np.zeros(h)

    # Calculate log variances for steps 2 to h (Python indices 1 to h-1)
    if h > 1:
        # Outer loop: R i goes 2..h. Python i goes 1..h-1. (Py_i = R_i - 1)
        for i in range(1, h):  # Corresponds to h steps 2, 3, ..., h
            iq[0] = 0.0  # Reset accumulator for R step i+1
            # R's inner loop limit: sum(steps < i), where i is R index (2..h)
            # Python equivalent limit: sum(steps < Py_i + 1) = sum(steps < i + 1)
            num_inner_loops = np.sum(steps < (i + 1))

            #  Inner loop: R k goes 1..num_inner_loops. Python k_idx goes
            # 0..num_inner_loops-1. (Py_k_idx = R_k - 1)
            for k_idx in range(num_inner_loops):
                # Get persistence slice corresponding to the k_idx-th step (< i+1)
                matrix_persistence_q[:] = array_persistence_q[:, :, k_idx]

                # Get the k_idx-th unique lag overall (used in power calculation)
                # R code uses lagsUnique[k], which maps to lags_unique[k_idx]
                current_lag = lags_unique[k_idx]
                if current_lag <= 0:
                    raise ValueError(f"Invalid lag found in lags_unique: {current_lag}")

                # Calculate the power exponent using the *correct* step index
                # R uses ceiling(i / lagsUnique[k]) - 1, where i is R index (2..h)
                # Python needs ceiling((Py_i + 1) / lags_unique[k_idx]) - 1
                #             = ceiling((i + 1) / lags_unique[k_idx]) - 1
                power_val = np.ceil((i + 1) / current_lag) - 1
                power_val = int(power_val)  # Ensure integer for matrix_power

                # Perform the matrix calculations
                try:
                    mat_pers_q_pow2 = matrix_power_wrap(matrix_persistence_q, 2)
                    term1 = mat_pers_q_pow2 * s2
                    term2 = matrix_power_wrap(ik + term1, power_val)
                    term3 = term2 - ik
                    iq[0] += np.sum(np.diag(term3))  # Accumulate sum of diagonal
                except Exception as e:
                    print(
                        f"Error in var_anal calculation for h={i + 1}, "
                        f"k_idx={k_idx}, lag={current_lag}, power={power_val}: {e}"
                    )
                    iq[0] = np.nan  # Propagate error as NaN
                    break  # Exit inner loop for this step i

            #  Assign log(iq) to the variance matrix (index i corresponds to R's i+1
            # step)
            if np.isnan(iq[0]):
                var_mat[i] = np.nan
            elif iq[0] <= 0:
                var_mat[i] = -np.inf if iq[0] == 0 else np.nan  # Match R log behavior
            else:
                var_mat[i] = np.log(iq[0])

    # Final Adjustments - applied in the same order as R
    # 1. Apply exp and multiply by (1 + s2)
    var_mat = np.exp(var_mat) * (1 + s2)

    # 2. Adjust the first element (index 0) - corresponds to R's varMat[1] adjustment
    if h > 0:
        # R: varMat[1] <- varMat[1] - 1. Initially exp(0)*(1+s2) -> 1+s2. Result s2.
        var_mat[0] = var_mat[0] - 1

    # 3. Adjust elements from index 1 onwards - corresponds to R's varMat[-1] adjustment
    if h > 1:
        # R: varMat[-1] <- varMat[-1] + s2 (elements 2..h)
        # Python: elements 1..h-1
        var_mat[1:] = var_mat[1:] + s2

    # Optional: Replace any remaining non-finite values with NaN
    var_mat[~np.isfinite(var_mat)] = np.nan

    return var_mat


# I did not use the C++ wrapper for simplicity here
# speed is alrigh here
def matrix_power_wrap(matrix, power):
    """
    Helper function to compute matrix power. Handles integer powers >= 0.
    """
    power = int(power)
    if power < 0:
        raise ValueError(f"Matrix power calculation received negative power: {power}")
    elif power == 0:
        return np.eye(matrix.shape[0])
    else:
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"Matrix must be square for matrix power. Got shape: {matrix.shape}"
            )
        if not np.isfinite(matrix).all():
            # Or handle differently if necessary
            raise ValueError("Matrix contains non-finite values.")
        try:
            # Use numpy's matrix_power for integer exponents
            return np.linalg.matrix_power(matrix, power)
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(
                f"Numpy matrix_power failed for power {power}: {e}"
            ) from e
        except ValueError as e:  # Catch other potential numpy errors
            raise ValueError(
                f"Error during numpy matrix_power for power {power}: {e}"
            ) from e


def fisher_information(
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
    step_size=None,
    other=None,
    other_parameter_estimate=False,
):
    """Observed Fisher Information matrix at the parameter vector ``B``.

    Direct translation of R's ``FI <- -hessian(logLikADAM, B, h=stepSize)``
    (``R/adam.R:2799``). The Hessian uses the pracma-exact finite-difference
    scheme implemented in the C++ ``_numDeriv.hessian`` and is taken of the
    log-likelihood (``log_Lik_ADAM`` ≡ R's ``logLikADAM``); the observed
    information is its negative.

    Parameters
    ----------
    B : array-like
        Parameter vector at which to evaluate the information matrix
        (typically the estimated optimum).
    step_size : float, optional
        Absolute finite-difference step ``h``. Defaults to
        ``np.finfo(float).eps ** 0.25`` (``≈ 1.22e-4``), matching R's
        ``.Machine$double.eps^(1/4)``.

    Returns
    -------
    numpy.ndarray
        Symmetric ``len(B) × len(B)`` observed Fisher Information matrix.
    """
    from smooth.adam_general._numDeriv import hessian as _hessian_cpp
    from smooth.adam_general.core.utils.cost_functions import log_Lik_ADAM

    B = np.asarray(B, dtype=float)
    h = step_size if step_size else float(np.finfo(float).eps ** 0.25)

    def loglik(b):
        return float(
            log_Lik_ADAM(
                np.asarray(b, dtype=float),
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
                otherParameterEstimate=other_parameter_estimate,
            )
        )

    return -np.asarray(_hessian_cpp(loglik, B, h))


def numerical_hessian(callable_, B, step_size=None):  # noqa: N803
    """Numerical Hessian of a scalar function via the pracma-exact scheme.

    Wraps the C++ ``_numDeriv.hessian`` so OM/OMG/ADAM can compute observed
    Fisher Information from their own cost functions without each rebuilding
    the loglik dispatch in :func:`fisher_information`.

    Parameters
    ----------
    callable_ : Callable[[np.ndarray], float]
        Scalar function of a 1-D parameter vector.
    B : array-like
        Point at which to evaluate the Hessian (typically the estimated
        optimum).
    step_size : float, optional
        Absolute finite-difference step ``h``. Defaults to
        ``np.finfo(float).eps ** 0.25`` (matches R's ``.Machine$double.eps^(1/4)``).

    Returns
    -------
    numpy.ndarray
        Symmetric ``len(B) × len(B)`` Hessian matrix.
    """
    from smooth.adam_general._numDeriv import hessian as _hessian_cpp

    B = np.asarray(B, dtype=float)
    h = step_size if step_size else float(np.finfo(float).eps ** 0.25)
    return np.asarray(_hessian_cpp(lambda b: float(callable_(b)), B, h))


def resolve_covar_type(type_, bootstrap=False):
    """Resolve the covariance ``type`` for the vcov/confint/summary methods.

    Mirrors R's ``covarTypeResolver``: ``type`` is one of ``"opg"`` (the
    default), ``"hessian"`` or ``"bootstrap"``. The deprecated ``bootstrap=True``
    switch maps to ``"bootstrap"`` with a ``DeprecationWarning``.
    """
    import warnings

    if bootstrap:
        warnings.warn(
            "bootstrap=True is deprecated; use type='bootstrap' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        return "bootstrap"
    if type_ is None:
        return "opg"
    if type_ not in ("opg", "hessian", "bootstrap"):
        raise ValueError(
            f"type must be one of 'opg', 'hessian', 'bootstrap'; got {type_!r}"
        )
    return type_


def covar_opg(parameter_values, point_lik_at, obs_in_sample, loglik, step_size=None):
    """OPG / BHHH covariance ``J^-1`` with ``J = sum_t s_t s_t'``.

    Direct translation of R's ``covarOPGCore`` (R/helper.R). The per-observation
    score ``s_t`` of the log-likelihood is estimated by central differences of
    ``point_lik_at`` (a callable mapping a parameter vector to the length-``T``
    per-observation log-likelihood), and the covariance is the inverse of the
    outer-product-of-gradients matrix ``J``. ``J`` is positive semi-definite by
    construction, so the covariance is finite at boundary estimates where the
    observed Hessian is indefinite.

    A reproduction guard requires ``point_lik_at(parameter_values)`` to sum to
    ``loglik`` (else returns ``None``, so the caller falls back to the Hessian).
    At an active bound one perturbation side may leave the feasible region and
    return a non-finite vector; the difference then drops to one-sided against
    the reproduced base. Directions with a (near-)zero score are dropped and
    given an infinite variance (unidentified parameters).

    Returns
    -------
    numpy.ndarray or None
        The ``len(B) x len(B)`` covariance, or ``None`` if the reproduction
        guard trips or a score cannot be evaluated.
    """
    b_values = np.asarray(parameter_values, dtype=float)
    n_param = len(b_values)
    if n_param == 0:
        return None
    h_default = step_size if step_size else float(np.finfo(float).eps ** 0.25)

    base_lik = point_lik_at(b_values)
    if (
        base_lik is None
        or len(base_lik) != obs_in_sample
        or not np.isclose(float(np.sum(base_lik)), float(loglik), atol=1e-4, rtol=0.0)
    ):
        return None
    base_lik = np.asarray(base_lik, dtype=float)

    def _side_ok(vec):
        return (
            vec is not None and len(vec) == obs_in_sample and np.all(np.isfinite(vec))
        )

    scores = np.full((obs_in_sample, n_param), np.nan)
    for j in range(n_param):
        h_step = h_default * max(1.0, abs(b_values[j]))
        up = b_values.copy()
        up[j] += h_step
        down = b_values.copy()
        down[j] -= h_step
        lik_up = point_lik_at(up)
        lik_down = point_lik_at(down)
        up_ok = _side_ok(lik_up)
        down_ok = _side_ok(lik_down)
        if up_ok and down_ok:
            scores[:, j] = (np.asarray(lik_up) - np.asarray(lik_down)) / (2 * h_step)
        elif up_ok:
            scores[:, j] = (np.asarray(lik_up) - base_lik) / h_step
        elif down_ok:
            scores[:, j] = (base_lik - np.asarray(lik_down)) / h_step
        else:
            return None

    if not np.all(np.isfinite(scores)):
        return None

    j_matrix = scores.T @ scores
    diag_j = np.diag(j_matrix)
    # Drop parameters whose per-observation scores carry (near-)zero information
    # relative to the best-identified one: at a boundary MLE such a parameter is
    # only weakly identified (its OPG score is tiny at the optimum even though it
    # still enters the likelihood), so its OPG variance is enormous. Reporting an
    # infinite variance flags it as effectively unidentified -- a clearer signal
    # than a huge finite number, and it makes reapply()/reforecast() hold the
    # parameter at its estimate rather than inject nonsense-wide draws. Use
    # type="hessian" for a finite (still large) standard error there.
    keep = (diag_j > np.max(diag_j) * 1e-10) & np.isfinite(diag_j)
    vcov = np.full((n_param, n_param), np.inf)
    if np.any(keep):
        # The Moore-Penrose pseudo-inverse via the symmetric eigen-decomposition,
        # dropping the (near-)zero-eigenvalue directions: the inverse when J is
        # well conditioned, a pooled variance for collinear parameters. As R, which
        # does not use solve(): its singularity check at the machine epsilon is
        # decided by the last bits of the LU.
        vals, vecs = np.linalg.eigh(j_matrix[np.ix_(keep, keep)])
        positive = vals > np.max(vals) * 1e-10
        if not np.any(positive):
            return None
        vecs_keep = vecs[:, positive]
        vcov_keep = vecs_keep @ (vecs_keep.T / vals[positive][:, None])
        vcov[np.ix_(keep, keep)] = vcov_keep
    return vcov


def invert_fisher_information(FI):  # noqa: N803
    """Invert an observed Fisher Information matrix to a covariance matrix.

    Reproduces R's ``vcov.adam`` inversion (``R/adam.R:5210``): "broken"
    variables — rows that are all zero or contain NaN — carry no information and
    are excluded from the inversion, then their rows/columns are set to ``inf``.
    The remaining sub-matrix is inverted via a Cholesky solve, falling back to a
    general solve and finally to a ``1e100`` diagonal if it is singular.

    Parameters
    ----------
    FI : array-like
        Observed Fisher Information matrix.

    Returns
    -------
    numpy.ndarray
        Covariance matrix (FI inverse) with broken rows/cols set to ``inf``.
    """
    FI = np.array(FI, dtype=float)
    n = FI.shape[0]

    broken = np.all(FI == 0, axis=1) | np.any(np.isnan(FI), axis=1)
    good = ~broken

    out = np.zeros((n, n))
    if np.any(good):
        sub = FI[np.ix_(good, good)]
        try:
            chol = np.linalg.cholesky(sub)
            inv = np.linalg.solve(chol.T, np.linalg.solve(chol, np.eye(sub.shape[0])))
        except np.linalg.LinAlgError:
            try:
                inv = np.linalg.solve(sub, np.eye(sub.shape[0]))
            except np.linalg.LinAlgError:
                inv = np.diag(np.full(sub.shape[0], 1e100))
        out[np.ix_(good, good)] = inv

    out[broken, :] = np.inf
    out[:, broken] = np.inf

    # Mirror R (R/adam.R:5226, R/omg.R:1690): "Just in case, take absolute
    # values for the diagonal in order to avoid possible issues with FI".
    # Without this, a non-positive-semi-definite FI can produce negative
    # variances on the diagonal (downstream ``sqrt(abs(diag(V)))`` recovers
    # the SE, but a user inspecting ``vcov`` directly would see something
    # algebraically impossible).
    diag_idx = np.arange(n)
    out[diag_idx, diag_idx] = np.abs(out[diag_idx, diag_idx])
    return out
