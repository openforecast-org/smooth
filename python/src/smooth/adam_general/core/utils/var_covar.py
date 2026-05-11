import numpy as np


def sigma(observations_dict, params_info, general, prepared_model):
    """
    Calculate error scale parameter (standard deviation) for ADAM model.

    This function computes the scale parameter σ (sigma) of the error distribution,
    which characterizes the magnitude of forecast errors. The calculation method
    depends on the error distribution specified in the model.

    The scale parameter is used for:

    - **Prediction intervals**: Determines interval width
    - **Log-likelihood**: Part of the probability density function
    - **Model diagnostics**: Measures model fit quality
    - **Simulation**: Governs error generation in forecasting

    **Calculation Method**:

    The scale is estimated as the square root of the mean squared (transformed)
    residuals:

    .. math::

        \\hat{\\sigma} = \\sqrt{\\frac{1}{T - k} \\sum_{t=1}^T r_t^2}

    where:

    - T = number of observations
    - k = number of estimated parameters
    - r_t = transformed residuals (transformation depends on distribution)

    For different distributions, the residuals are transformed as:

    - **Normal, Laplace, S, Generalized Normal, t, Logistic, Asymmetric Laplace**:
      :math:`r_t = \\epsilon_t` (untransformed residuals)

    - **Log-Normal, Log-Laplace, Log-S**:
      :math:`r_t = \\log(\\epsilon_t)` (log-transformed residuals)

    - **Inverse Gaussian, Gamma**:
      :math:`r_t = \\epsilon_t` (untransformed, for multiplicative errors)

    Parameters
    ----------
    observations_dict : dict
        Observation information containing:

        - 'obs_in_sample': Number of in-sample observations (T)

    params_info : list or array
        Parameter counts containing:

        - params_info[0][-1]: Total number of parameters estimated (k)

    general : dict
        General model configuration containing:

        - **'distribution'**: Error distribution name. Supported values:

          * 'dnorm': Normal distribution
          * 'dlaplace': Laplace (double exponential)
          * 'ds': S distribution
          * 'dgnorm': Generalized normal
          * 'dt': Student's t
          * 'dlogis': Logistic
          * 'dalaplace': Asymmetric Laplace
          * 'dlnorm': Log-normal
          * 'dllaplace': Log-Laplace
          * 'dls': Log-S
          * 'dinvgauss': Inverse Gaussian
          * 'dgamma': Gamma

    prepared_model : dict
        Prepared model from ``preparator()`` containing:

        - **'residuals'**: In-sample residuals (y_t - y_fitted_t), pandas Series or
        ndarray.
          May contain NaN values which are excluded from calculation.

    Returns
    -------
    float
        Estimated scale parameter σ (sigma). Always positive.

        - For additive errors: Interpreted as standard deviation of errors
        - For multiplicative errors: Scale of relative errors

    Notes
    -----
    **Degrees of Freedom Adjustment**:

    The denominator is T - k (degrees of freedom) to provide an unbiased estimate.
    If T - k ≤ 0 (sample too small or too many parameters), uses biased estimator
    with denominator T instead.

    **Missing Values**:

    NaN values in residuals are automatically excluded from the calculation. The
    effective sample size is the number of non-NaN residuals.

    **Distribution-Specific Notes**:

    - **Log-distributions**: Work on log-scale to accommodate positive-only data.
      The sigma is the standard deviation of log(errors), not errors themselves.

    - **Gamma and Inverse Gaussian**: For multiplicative error models. The formula
      uses residuals directly (not residuals - 1) to match R implementation.

    - **Generalized Log-Normal**: Currently commented out, would require additional
      scale extraction step.

    **Relationship to Likelihood**:

    The scale parameter σ appears in the likelihood function. For normal errors:

    .. math::

        \\log L = -\\frac{T}{2}\\log(2\\pi) - \\frac{T}{2}\\log(\\sigma^2) -
        \\frac{1}{2\\sigma^2}\\sum r_t^2

    Maximizing likelihood is equivalent to minimizing σ² for normal distribution.

    **Scale vs Variance**:

    - **scale (σ)**: Square root of variance, same units as data
    - **variance (σ²)**: Second moment, squared units
    - **s2**: One-step-ahead variance (used in var_anal, covar_anal)

    In ADAM, "scale" typically refers to σ, while "s2" refers to σ².

    **Performance**:

    Very fast computation (~1ms), dominated by residual squaring operation.

    See Also
    --------
    preparator : Computes residuals used in sigma calculation
    covar_anal : Uses s2 = sigma² for covariance matrix calculation
    var_anal : Uses s2 for variance calculation
    log_Lik_ADAM : Uses sigma in likelihood computation

    Examples
    --------
    Calculate scale from fitted model::

        >>> sigma_estimate = sigma(
        ...     observations_dict={'obs_in_sample': 100},
        ...     params_info=[[...], [...], [5]],  # 5 parameters
        ...     general={'distribution': 'dnorm'},
        ...     prepared_model={'residuals': residuals_series}
        ... )
        >>> print(f"Error standard deviation: {sigma_estimate:.4f}")

    Compare scales across distributions::

        >>> # Normal error model
        >>> sigma_norm = sigma(obs_dict, params, {'distribution': 'dnorm'}, model)
        >>> # Laplace error model
        >>> sigma_laplace = sigma(obs_dict, params, {'distribution': 'dlaplace'}, model)
        >>> # Laplace typically has smaller sigma for same data

    Use sigma for prediction interval width::

        >>> sigma_hat = sigma(obs_dict, params, general, prepared_model)
        >>> # 95% prediction interval (for normal errors):
        >>> interval_width = 1.96 * sigma_hat
    """

    params_number = params_info[0][-1]

    # In case of likelihood, scale is not calculated towards parameters for variance
    if general["loss"] == "likelihood" and len(params_info[0]) > 1:
        params_number = params_number - params_info[0][1]

    vals = observations_dict["obs_in_sample"] - params_number
    # If the sample is too small, then use biased estimator
    if vals <= 0:
        vals = observations_dict["obs_in_sample"]

    residuals = prepared_model["residuals"]
    non_nan_mask = ~residuals.isna()

    # Calculate sigma based on distribution type
    r = residuals[non_nan_mask]
    if general["distribution"] in [
        "dnorm",
        "dlaplace",
        "ds",
        "dgnorm",
        "dt",
        "dlogis",
        "dalaplace",
        "dinvgauss",
        "dgamma",
    ]:
        return np.linalg.norm(r) / np.sqrt(vals)
    elif general["distribution"] in ["dlnorm", "dllaplace", "dls"]:
        # elif general['distribution'] == 'dlgnorm':
        # we need the extract_scale() function here
        #    sigma = (np.log(r - extract_scale()**2/2)**2).sum()
        return np.linalg.norm(np.log(r)) / np.sqrt(vals)


def covar_anal(lags_model, h, measurement, transition, persistence, s2):
    """
    Returns analytical conditional h-steps ahead covariance matrix. Corrected Python
    version.
    This is used in covar() method and in the construction of parametric prediction
    intervals.

    Parameters:
    - lags_model: list or array, model lags assigned to each state (e.g., [1, 1, 12])
    - h: int, forecast horizon
    - measurement: array, measurement matrix (typically h x n_components from
    forecaster).
                   The function logic uses the first row measurement[0, :].
    - transition: array, transition matrix (n_components x n_components)
    - persistence: array, persistence vector (k_states,)
    - s2: float, one-step-ahead variance

    Returns:
    - covar_mat: array, covariance matrix (h x h)
    """
    # Ensure inputs are numpy arrays and persistence is 1D
    lags_model = np.array(lags_model)
    measurement_matrix = np.array(measurement)  # Keep original name for clarity
    transition = np.array(transition)
    persistence = np.array(persistence).flatten()  # Ensure persistence is 1D

    n_components = transition.shape[0]
    k_states = len(persistence)
    # print(len(measurement_matrix)) # Optional debug print

    # --- Basic Input Validation ---
    if n_components != transition.shape[1]:
        raise ValueError("Transition matrix must be square.")
    # Validate measurement matrix dimensions more robustly
    if measurement_matrix.ndim != 2 or measurement_matrix.shape[1] != n_components:
        raise ValueError(
            f"Measurement matrix shape {measurement_matrix.shape} incompatible "
            f"with n_components {n_components}. Expecting (>=1, {n_components})."
        )
    if lags_model.shape[0] != k_states:
        raise ValueError(
            f"lags_model length {lags_model.shape[0]} "
            f"must match persistence length {k_states}."
        )
    if (
        k_states > n_components
    ):  # Allow k_states <= n_components (e.g. ARIMA components)
        raise ValueError(
            f"Number of states ({k_states}) from persistence vector "
            f"cannot exceed transition matrix dimension ({n_components})."
        )
    if measurement_matrix.shape[0] < 1:
        raise ValueError("Measurement matrix must have at least one row.")
    # --- End Validation ---

    # Use the first row of the measurement matrix, similar to R logic assumption
    measurement_vector = measurement_matrix[0, :]  # Shape (n_components,)

    covar_mat = np.eye(h)
    min_lag = np.min(lags_model) if len(lags_model) > 0 else h + 1  # Handle empty lags

    if h > min_lag:
        lags_unique = np.unique(lags_model)
        steps = np.sort(lags_unique[lags_unique <= h])
        steps_number = len(steps)
        if steps_number == 0:  # No relevant lags within horizon
            # Multiply the matrix by the one-step-ahead variance
            covar_mat = covar_mat * s2
            return covar_mat

        array_transition = np.zeros((n_components, n_components, steps_number))
        # This array stores slices of the measurement_vector based on lags
        array_measurement = np.zeros((1, n_components, steps_number))

        for i in range(steps_number):
            mask = lags_model == steps[i]  # k_states long boolean mask
            # Need to map k_states mask to n_components columns
            component_mask = np.zeros(n_components, dtype=bool)
            if k_states == n_components:
                component_mask = mask
            else:
                # Assuming persistence corresponds to the first k_states components
                #  This might need adjustment based on specific model structure if
                # k_states < n_components
                component_mask[:k_states] = mask

            if np.sum(component_mask) > 0:
                array_transition[:, component_mask, i] = transition[:, component_mask]
                # Assign parts of the single measurement_vector
                array_measurement[0, component_mask, i] = measurement_vector[
                    component_mask
                ]

        # Holds values corresponding to R's cValues[i+1]
        c_values = np.zeros(h)

        # Prepare transition array
        transition_powered = np.zeros((n_components, n_components, h, steps_number))
        # Initialize first min(steps) time steps (Python index 0 to min(steps)-1)
        current_min_step = min(steps) if len(steps) > 0 else 0
        # Corrected loop: Iterate through time steps AND step numbers for initialization
        for i in range(current_min_step):
            for k in range(
                steps_number
            ):  # Add loop over the 4th dimension (steps_number)
                transition_powered[:, :, i, k] = np.eye(
                    n_components
                )  # Assign to specific [:,:,i,k] slice

        # Generate values for the transition matrix
        # R loops i from (min(steps)+1) to h. Python loops i from min(steps) to h-1.
        for i in range(current_min_step, h):
            #  R loops k from 1 to sum(steps<i). Python loops k from 0 to sum(steps <
            # i+1)-1
            num_inner_loops_k = np.sum(steps < (i + 1))
            for k in range(num_inner_loops_k):
                # This needs to be produced only for the lower lag (k=0).
                # Then it will be reused for the higher ones.
                if k == 0:  # R's k==1
                    #  R loops j from 1 to sum(steps<i). Python loops j from 0 to
                    # sum(steps < i+1)-1
                    num_inner_loops_j = np.sum(steps < (i + 1))  # Same limit as k loop
                    for j in range(num_inner_loops_j):
                        # Condition uses R's i, which is Python's i + 1
                        if steps[j] == 0:
                            continue  # Avoid division by zero
                        if (i + 1 - steps[k]) / steps[j] > 1:  # Use Py i+1
                            transition_new = array_transition[:, :, j]
                        else:
                            transition_new = np.eye(n_components)

                        # Indexing transition_powered uses Python's i, k, j
                        past_index = i - steps[j]
                        if past_index < 0:  # Ensure index is valid
                            #  This case might indicate an issue or need specific
                            # handling.
                            # For now, assume identity if index is invalid.
                            past_transition_powered = np.eye(n_components)
                        else:
                            past_transition_powered = transition_powered[
                                :, :, past_index, k
                            ]

                        # If this is a zero matrix, do simple multiplication
                        if np.all(transition_powered[:, :, i, k] == 0):
                            transition_powered[:, :, i, k] = (
                                transition_new @ past_transition_powered
                            )
                        else:
                            # Check that the multiplication is not an identity matrix
                            new_term = transition_new @ past_transition_powered
                            if not np.allclose(
                                new_term, np.eye(n_components)
                            ):  # Use allclose for float comparison
                                transition_powered[:, :, i, k] = (
                                    transition_powered[:, :, i, k] + new_term
                                )
                else:  # k > 0 (R's k > 1)
                    # Copy the structure from the lower lags (k=0)
                    # R: transitionPowered[,,i-steps[k]+1,1]; (Index 1 for k=1)
                    # Py: transition_powered[:, :, i-steps[k]+1, 0] (Index 0 for k=0)
                    time_index_copy = i - steps[k] + 1
                    if (
                        time_index_copy < 0 or time_index_copy >= h
                    ):  # Ensure index is valid within h dimension
                        #  Handle invalid index - maybe copy identity or latest
                        # available?
                        # Copying identity might be safest default if state is unknown.
                        transition_powered[:, :, i, k] = np.eye(n_components)
                    else:
                        transition_powered[:, :, i, k] = transition_powered[
                            :, :, time_index_copy, 0
                        ]

                # Generate values of cj
                #  Uses Python's i, k. Stores result in c_values[i] (maps to R's
                # cValues[i+1])
                #  Ensure array_measurement slice has correct shape (1, n_components)
                # before matmul
                meas_slice = array_measurement[:, :, k]
                if meas_slice.shape != (1, n_components):
                    # This case shouldn't happen with current logic, but good practice
                    raise ValueError(
                        f"Unexpected shape for array_measurement slice: "
                        f"{meas_slice.shape}"
                    )

                c_values[i] = (
                    c_values[i]
                    + (meas_slice @ transition_powered[:, :, i, k] @ persistence)[0]
                )

        # Fill in diagonals
        # R loops i from 2 to h. Uses cValues[i].
        #  Python loops i from 1 to h-1. Needs value corresponding to R's cValues[i+1],
        # which is Py's c_values[i].
        # <<< FIX START >>>
        for i in range(1, h):
            # Index i is valid for c_values (0 to h-1)
            # Ensure c_values[i] is not NaN before squaring
            if not np.isnan(c_values[i]):
                c_val_sq = (
                    c_values[i] ** 2
                )  # Use c_values[i] corresponding to R's cValues[i+1]
                # Ensure covar_mat[i-1, i-1] is not NaN before adding
                if not np.isnan(covar_mat[i - 1, i - 1]):
                    covar_mat[i, i] = covar_mat[i - 1, i - 1] + c_val_sq
                else:
                    covar_mat[i, i] = np.nan  # Propagate NaN
            else:
                covar_mat[i, i] = np.nan  # Propagate NaN
        # <<< FIX END >>>

        # Fill in off-diagonals
        # R loops i, j from 1 to h. Python loops i, j from 0 to h-1.
        for i in range(h):
            for j in range(h):
                if i == j:
                    continue
                elif i == 0:  # R's i==1
                    #  R uses cValues[j]. Python needs element corresponding to R's
                    # cValues[j+1], which is Py's c_values[j].
                    if j >= 0 and j < len(
                        c_values
                    ):  # Check index validity for c_values[j]
                        covar_mat[i, j] = c_values[
                            j
                        ]  # Use c_values[j] instead of c_values[j-1]
                    else:
                        #  Handle cases where index j might be out of bounds for
                        # c_values (shouldn't happen if j<h)
                        # Add check for NaN propagation
                        if j >= 0 and j < len(c_values) and np.isnan(c_values[j]):
                            covar_mat[i, j] = np.nan
                        # Explicitly handle Py j=0 case if needed.
                        # Maybe should be 0? R's cValues[1] is 0.
                        elif j == 0:
                            # Tentatively set to 0.0 based on R cValues[1]
                            covar_mat[i, j] = 0.0
                        #  If j is out of bounds, something else is wrong. Let it raise
                        # IndexError or handle as NaN?
                elif i > j:
                    covar_mat[i, j] = covar_mat[j, i]  # Symmetry
                else:  # i < j
                    # R: covarMat[i-1,j-1] + covarMat[1,j] * covarMat[1,i];
                    # Py: covar_mat[i-1, j-1] + covar_mat[0, j] * covar_mat[0, i]
                    #  This recursive relation should now use the correctly filled first
                    # row/col
                    if (i - 1) >= 0 and (j - 1) >= 0:  # Check indices
                        term1 = covar_mat[i - 1, j - 1]
                        term2 = covar_mat[0, j]
                        term3 = covar_mat[0, i]
                        # Check for NaN before calculation
                        if not (np.isnan(term1) or np.isnan(term2) or np.isnan(term3)):
                            covar_mat[i, j] = term1 + term2 * term3
                        else:
                            covar_mat[i, j] = (
                                np.nan
                            )  # Propagate NaN if components are NaN
                    else:
                        #  Handle cases where indices i-1 or j-1 are invalid (shouldn't
                        # happen if i < j and i >= 1)
                        covar_mat[i, j] = np.nan  # Or some other default?

    # Multiply the matrix by the one-step-ahead variance
    covar_mat = covar_mat * s2

    # Replace NaNs that might have occurred due to index issues or errors, if desired
    # covar_mat = np.nan_to_num(covar_mat, nan=0.0) # Optional: replace NaN with 0

    return covar_mat


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
