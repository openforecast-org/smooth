import nlopt
import numpy as np

from smooth.adam_general.core.creator import architector, creator, filler, initialiser
from smooth.adam_general.core.utils.gradient import adam_fit_or_gradient

from .optimization import (
    _calculate_loglik,
    _configure_optimizer,
    _create_objective_function,
    _run_optimization,
    _set_distribution,
    _setup_arima_polynomials,
    _setup_optimization_parameters,
)
from .two_stage import _run_two_stage_estimator


def estimator(
    general_dict,
    model_type_dict,
    lags_dict,
    observations_dict,
    arima_dict,
    constant_dict,
    explanatory_dict,
    profiles_recent_table,
    profiles_recent_provided,
    persistence_dict,
    initials_dict,
    phi_dict,
    components_dict,
    occurrence_dict,
    multisteps=False,
    lb=None,
    ub=None,
    maxtime=None,
    print_level=0,  # 1 or 0
    maxeval=None,
    B_initial=None,
    return_matrices=False,
    other=None,
    other_parameter_estimate=False,
    # NLopt parameters
    xtol_rel=1e-6,
    xtol_abs=1e-8,
    ftol_rel=1e-8,
    ftol_abs=0,
    algorithm="NLOPT_LN_NELDERMEAD",
    smoother="global",
    adam_ets: bool = False,
    fi: bool = False,
    step_size=None,
):
    """
    Estimate parameters for ADAM model using non-linear optimization.

    This function coordinates the complete parameter estimation process for an ADAM
    (Augmented Dynamic Adaptive Model) by setting up the state-space model structure,
    initializing parameters, and executing optimization via the NLopt library. The
    estimation minimizes a cost function (likelihood, MSE, MAE, etc.) to find optimal
    parameter values.

    The ADAM model is represented in state-space form as:

    .. math::

        y_t &= o_t(w(v_{t-l}) + h(x_t, a_{t-1}) + r(v_{t-l})\\epsilon_t)

        v_t &= f(v_{t-l}, a_{t-1}) + g(v_{t-l}, a_{t-1}, x_t)\\epsilon_t

    where:

    - :math:`y_t` is the observed value at time t
    - :math:`v_t` is the state vector
    - :math:`o_t` is the occurrence indicator (for intermittent data)
    - :math:`w(\\cdot)` is the measurement function
    - :math:`h(\\cdot)` is the exogenous variables function
    - :math:`r(\\cdot)` is the error function
    - :math:`f(\\cdot)` is the transition function
    - :math:`g(\\cdot)` is the persistence function
    - :math:`\\epsilon_t` is the error term

    **Estimation Algorithm**:

    1. **Architecture Setup**: Call ``architector()`` to define model structure,
    determine
       component counts, and set up lag structures
    2. **Matrix Creation**: Call ``creator()`` to build initial state-space matrices
       (measurement, transition, persistence)
    3. **Parameter Initialization**: Call ``initialiser()`` to construct the initial
       parameter vector B and bounds (lower/upper limits)
    4. **Distribution Selection**: Map loss function to appropriate error distribution
       (e.g., MSE → Normal, MAE → Laplace)
    5. **Optimization Setup**: Configure NLopt with Nelder-Mead algorithm, set
    tolerances
       and iteration limits
    6. **Objective Function**: Create wrapper for ``CF()`` cost function
    7. **Optimization Execution**: Run NLopt to minimize cost function
    8. **Log-likelihood Calculation**: Compute final log-likelihood using
    ``log_Lik_ADAM()``
    9. **Results Assembly**: Package estimated parameters, matrices, and diagnostics

    **Optimization Method**:

    Uses NLopt's Nelder-Mead (LN_NELDERMEAD) algorithm:

    - Gradient-free local optimization suitable for non-smooth cost functions
    - Tolerances: xtol_rel=1e-6, ftol_rel=1e-8, xtol_abs=1e-8
    - Maximum evaluations: 40 × len(B) for standard models, 150 × len(B) for models
      with external regressors
    - Default timeout: 30 minutes

    **Parameter Vector Structure**:

    The optimization parameter vector B contains (in order):

    1. **ETS Persistence**: α (level), β (trend), γ (seasonal)
    2. **Damping**: φ (if damped trend is present)
    3. **Initial States**: l₀ (level), b₀ (trend), s₀ (seasonal), ARIMA initial states
    4. **ARIMA Parameters**: AR coefficients (φ₁, φ₂, ...), MA coefficients (θ₁, θ₂,
    ...)
    5. **Regression Coefficients**: Weights for external regressors
    6. **Constant Term**: Intercept (if included)
    7. **Distribution Parameters**: Shape parameters for certain distributions

    Parameters
    ----------
    general_dict : dict
        General model configuration containing:

        - 'loss': Loss function ('likelihood', 'MSE', 'MAE', 'HAM', 'LASSO', 'RIDGE')
        - 'distribution': Error distribution specification
        - 'bounds': Parameter bounds type ('usual', 'admissible', 'none')
        - 'ic': Information criterion for model selection ('AIC', 'AICc', 'BIC', 'BICc')
        - 'h': Forecast horizon
        - 'holdout': Whether holdout sample is used

    model_type_dict : dict
        Model type specification containing:

        - 'model': Model string (e.g., "ANN", "AAA", "MAM")
        - 'error_type': Error type ('A' for additive, 'M' for multiplicative)
        - 'trend_type': Trend type ('N', 'A', 'Ad', 'M', 'Md')
        - 'season_type': Seasonality type ('N', 'A', 'M')
        - 'ets_model': Whether ETS components are present
        - 'arima_model': Whether ARIMA components are present
        - 'xreg_model': Whether external regressors are present

    lags_dict : dict
        Lag structure information containing:

        - 'lags': Primary lag vector (e.g., [1, 12] for monthly data with annual
        seasonality)
        - 'lags_model': Lags for each state component
        - 'lags_model_all': Complete lag specification for all components
        - 'lags_model_max': Maximum lag value (defines pre-sample period)

    observations_dict : dict
        Time series data containing:

        - 'y_in_sample': Observed values for estimation
        - 'y_holdout': Holdout sample (if applicable)
        - 'obs_in_sample': Number of in-sample observations
        - 'ot': Occurrence vector (1 for non-zero, 0 for zero observations)
        - 'ot_logical': Boolean mask for non-zero observations

    arima_dict : dict
        ARIMA specification containing:

        - 'arima_model': Whether ARIMA components exist
        - 'ar_orders': AR order for each lag (e.g., [1, 1] for SARIMA)
        - 'i_orders': Integration order for each lag
        - 'ma_orders': MA order for each lag
        - 'ar_estimate': Whether to estimate AR parameters
        - 'ma_estimate': Whether to estimate MA parameters
        - 'ar_required': Whether AR is included
        - 'ma_required': Whether MA is included

    constant_dict : dict
        Constant term specification containing:

        - 'constant_required': Whether model includes a constant
        - 'constant_estimate': Whether to estimate the constant
        - 'constant_name': Variable name for constant in B vector

    explanatory_dict : dict
        External regressors specification containing:

        - 'xreg_model': Whether regressors are present
        - 'xreg_number': Number of regressors
        - 'xreg_names': Names of regressor variables
        - 'mat_xt': Regressor data matrix

    profiles_recent_table : numpy.ndarray
        Recent profile values for time-varying parameters (advanced feature)
    profiles_recent_provided : bool
        Whether user provided custom profile values
    persistence_dict : dict
        Persistence parameters containing:

        - 'persistence_estimate': Whether to estimate persistence
        - 'persistence_level_estimate': Estimate α (level smoothing)
        - 'persistence_trend_estimate': Estimate β (trend smoothing)
        - 'persistence_seasonal_estimate': Estimate γ (seasonal smoothing)
        - 'persistence_level': Fixed value for α (if not estimated)
        - 'persistence_trend': Fixed value for β (if not estimated)
        - 'persistence_seasonal': Fixed value for γ (if not estimated)

    initials_dict : dict
        Initial states specification containing:

        - 'initial_type': Initialization method:

          * 'backcasting': Use backcasting with refinement
          * 'optimal': Optimize initial states
          * 'two-stage': First backcasting then optimisation
          * 'complete': Full backcasting without refinement
          * 'provided': User-provided initial values

        - 'initial_level': Fixed level initial (if provided)
        - 'initial_trend': Fixed trend initial (if provided)
        - 'initial_seasonal': Fixed seasonal initials (if provided)
        - 'initial_level_estimate': Whether to optimize level initial
        - 'initial_trend_estimate': Whether to optimize trend initial
        - 'initial_seasonal_estimate': Whether to optimize seasonal initials
        - 'n_iterations': Backcasting iteration count (typically 2)

    phi_dict : dict
        Damping parameter specification containing:

        - 'phi_estimate': Whether to estimate damping parameter
        - 'phi': Fixed damping value (if not estimated, typically 1.0 for undamped)

    components_dict : dict
        Component counts containing:

        - 'components_number_ets': Total ETS components
        - 'components_number_ets_seasonal': Number of seasonal components
        - 'components_number_ets_non_seasonal': Number of non-seasonal ETS components
        - 'components_number_arima': Number of ARIMA state components
        - 'components_number_all': Total state dimension

    occurrence_dict : dict
        Intermittent data specification containing:

        - 'occurrence_model': Whether intermittent demand model is used
        - 'occurrence_type': Type of occurrence model
        - 'ot': Occurrence vector

    multisteps : bool, default=False
        Whether to use multi-step-ahead cost function (e.g., for MSEh, MAEh).
        If True, errors are computed over h-step-ahead forecasts rather than
        one-step-ahead.
    lb : numpy.ndarray, optional
        Lower bounds for parameters. If None, computed by ``initialiser()``.
        Shape must match B.
    ub : numpy.ndarray, optional
        Upper bounds for parameters. If None, computed by ``initialiser()``.
        Shape must match B.
    maxtime : float, optional
        Maximum optimization time in seconds. If None, defaults to 1800 seconds (30
        min).
    print_level : int, default=1
        Verbosity level:

        - 0: Silent (no output)
        - 1: Minimal output (currently suppressed in favor of general_dict['silent'])

    maxeval : int, optional
        Maximum number of cost function evaluations. If None, computed as:

        - Standard models: 40 × len(B)
        - Models with regressors: max(1500, 150 × len(B))

    B_initial : numpy.ndarray, optional
        Initial parameter vector to start optimization from. If provided, it overrides
        the default initialization computed by ``initialiser()``. Useful for:

        - Two-stage initialization (backcasting → optimal)
        - Warm-starting from previous estimates
        - Custom starting values

        Shape must match the parameter vector structure.
    return_matrices : bool, default=False
        Whether to return state-space matrices in the result dictionary. Useful for
        two-stage initialization where backcasted states are needed. If True, returns:

        - 'matrices': Updated state-space matrices with backcasted/optimized states
        - 'lags_dict': Updated lags dictionary
        - 'profile_dict': Updated profile dictionary
        - 'components_dict': Components information

    smoother : str, default="global"
        Smoother type for time series decomposition used in initial state estimation.

        - "lowess": Uses LOWESS for both trend and seasonal extraction
        - "ma": Uses moving average for both
        - "global": Uses lowess for trend and "ma" for seasonality

    Returns
    -------
    dict
        Dictionary containing estimation results:

        - **'B'** (numpy.ndarray): Optimized parameter vector
        - **'CF_value'** (float): Final cost function value at optimum
        - **'n_param_estimated'** (int): Number of estimated parameters
        - **'log_lik_adam_value'** (dict): Log-likelihood information with keys:

          * 'value': Log-likelihood value
          * 'nobs': Number of observations
          * 'df': Degrees of freedom (parameters + scale)

        - **'arima_polynomials'** (dict): AR and MA polynomial coefficients (if ARIMA
        present)

        If `return_matrices=True`, additionally includes:

        - **'matrices'** (dict): Updated state-space matrices (mat_vt, mat_wt, mat_f,
        vec_g)
        - **'lags_dict'** (dict): Lags information
        - **'profile_dict'** (dict): Profile matrices
        - **'components_dict'** (dict): Component counts

    Raises
    ------
    RuntimeError
        If optimization fails to converge or encounters numerical errors

    Notes
    -----
    **Special Cases**:

    1. **LASSO/RIDGE with λ=1**: Parameters are preset to zero, only initials are
    estimated
       using MSE
    2. **Two-stage initialization**: When initial_type='two-stage', the function is
    called
       twice:

       - Stage 1: initial_type='complete' (backcasting)
       - Stage 2: initial_type='optimal' using Stage 1 results as B_initial

    **Optimization Troubleshooting**:

    - If CF returns 1e100, parameter constraints were violated
    - If CF returns 1e300, NaN was encountered during computation
    - Increase maxeval if optimization terminates prematurely
    - Check bounds if estimated parameters are at boundaries

    **Performance Considerations**:

    - C++ ``adam_fitter()`` is called at each iteration (major computational cost)
    - Larger models (many seasonalities, high ARIMA orders) require more iterations
    - Backcasting (initial_type='backcasting' or 'complete') adds overhead but improves
      initial state estimates

    See Also
    --------
    architector : Set up model architecture
    creator : Create state-space matrices
    initialiser : Initialize parameter vector and bounds
    CF : Cost function for optimization
    log_Lik_ADAM : Calculate log-likelihood
    selector : Automatic model selection

    References
    ----------
    .. [1] Svetunkov, I. (2023). "Smooth forecasting with the smooth package in R".
           arXiv:2301.01790.
    .. [2] Hyndman, R.J., Koehler, A.B., Ord, J.K., and Snyder, R.D. (2008).
           "Forecasting with Exponential Smoothing: The State Space Approach".
           Springer-Verlag.
    .. [3] Johnson, S.G. The NLopt nonlinear-optimization package.
           https://nlopt.readthedocs.io/

    Examples
    --------
    Estimate parameters for an ETS(A,A,A) model::

        >>> adam_estimated = estimator(
        ... general_dict={'loss': 'likelihood', 'distribution': 'default', 'bounds':
        'usual', ...},
        ... model_type_dict={'model': 'AAA', 'error_type': 'A', 'trend_type': 'A',
        'season_type': 'A', ...},
        ...     lags_dict={'lags': np.array([1, 12]), ...},
        ... observations_dict={'y_in_sample': y_data, 'obs_in_sample': len(y_data),
        ...},
        ...     arima_dict={'arima_model': False, ...},
        ...     constant_dict={'constant_required': False, ...},
        ...     explanatory_dict={'xreg_model': False, ...},
        ...     profiles_recent_table=None,
        ...     profiles_recent_provided=False,
        ...     persistence_dict={'persistence_estimate': True, ...},
        ...     initials_dict={'initial_type': 'backcasting', ...},
        ...     phi_dict={'phi_estimate': False, 'phi': 1.0},
        ...     components_dict={...},
        ...     occurrence_dict={'occurrence_model': False, ...}
        ... )
        >>> print(f"Estimated parameters: {adam_estimated['B']}")
        >>> print(f"Log-likelihood: {adam_estimated['log_lik_adam_value']['value']}")

    Two-stage initialization example::

        >>> # Stage 1: Backcasting
        >>> initials_dict['initial_type'] = 'complete'
        >>> stage1 = estimator(..., return_matrices=True)
        >>> # Stage 2: Optimal
        >>> initials_dict['initial_type'] = 'optimal'
        >>> B_initial = np.concatenate([stage1['B'], extracted_initials_from_stage1])
        >>> stage2 = estimator(..., B_initial=B_initial)
    """
    # Handle two-stage initialization internally
    if initials_dict.get("initial_type") == "two-stage" and B_initial is None:
        return _run_two_stage_estimator(
            general_dict=general_dict,
            model_type_dict=model_type_dict,
            lags_dict=lags_dict,
            observations_dict=observations_dict,
            arima_dict=arima_dict,
            constant_dict=constant_dict,
            explanatory_dict=explanatory_dict,
            profiles_recent_table=profiles_recent_table,
            profiles_recent_provided=profiles_recent_provided,
            persistence_dict=persistence_dict,
            initials_dict=initials_dict,
            phi_dict=phi_dict,
            components_dict=components_dict,
            occurrence_dict=occurrence_dict,
            multisteps=multisteps,
            lb=lb,
            ub=ub,
            maxtime=maxtime,
            print_level=print_level,
            maxeval=maxeval,
            return_matrices=return_matrices,
            xtol_rel=xtol_rel,
            xtol_abs=xtol_abs,
            ftol_rel=ftol_rel,
            ftol_abs=ftol_abs,
            algorithm=algorithm,
            smoother=smoother,
        )

    # Step 1: Set up model structure
    # Simple call of the architector - also creates adam_cpp object
    (
        model_type_dict,
        components_dict,
        lags_dict,
        observations_dict,
        profile_dict,
        adam_cpp,
    ) = architector(
        model_type_dict,
        lags_dict,
        observations_dict,
        arima_dict,
        explanatory_dict,
        constant_dict,
        profiles_recent_table,
        profiles_recent_provided,
        adam_ets,
    )

    # Step 2: Create model matrices
    # Simple call of the creator
    adam_created = creator(
        model_type_dict,
        lags_dict,
        profile_dict,
        observations_dict,
        persistence_dict,
        initials_dict,
        arima_dict,
        constant_dict,
        phi_dict,
        components_dict,
        explanatory_dict,
        smoother=smoother,
    )
    # Step 3: Initialize parameters
    b_values = initialiser(
        model_type_dict=model_type_dict,
        components_dict=components_dict,
        lags_dict=lags_dict,
        adam_created=adam_created,
        persistence_checked=persistence_dict,
        initials_checked=initials_dict,
        arima_checked=arima_dict,
        constants_checked=constant_dict,
        explanatory_checked=explanatory_dict,
        observations_dict=observations_dict,
        bounds=general_dict["bounds"],
        phi_dict=phi_dict,
        profile_dict=profile_dict,
        adam_cpp=adam_cpp,
        other_parameter_estimate=other_parameter_estimate,
        other_value=other if other is not None else 2.0,
    )
    # Get initial parameter vector and bounds; user-provided values are used as-is
    B = np.asarray(B_initial, dtype=float) if B_initial is not None else b_values["B"]
    lb = np.asarray(lb, dtype=float) if lb is not None else b_values["Bl"]
    ub = np.asarray(ub, dtype=float) if ub is not None else b_values["Bu"]

    # Step 4: Set up ARIMA polynomials if needed
    ar_polynomial_matrix, ma_polynomial_matrix = _setup_arima_polynomials(
        model_type_dict, arima_dict, lags_dict
    )

    # Step 5: Set appropriate distribution
    general_dict = _set_distribution(general_dict, model_type_dict)

    # Step 6: Configure optimization parameters
    maxeval_used, general_dict = _setup_optimization_parameters(
        general_dict,
        explanatory_dict,
        B,
        maxeval,
        adam_created,
        observations_dict,
        multisteps,
        components_dict,
    )

    # Step 7: Create and configure optimizer
    # Convert algorithm string to nlopt constant
    nlopt_algorithm = getattr(
        nlopt, algorithm.replace("NLOPT_", ""), nlopt.LN_NELDERMEAD
    )
    opt = nlopt.opt(nlopt_algorithm, len(B))
    opt = _configure_optimizer(
        opt,
        lb,
        ub,
        maxeval_used,
        maxtime,
        xtol_rel=xtol_rel,
        xtol_abs=xtol_abs,
        ftol_rel=ftol_rel,
        ftol_abs=ftol_abs,
    )

    # Step 8: Create objective function
    objective_wrapper = _create_objective_function(
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
        print_level,
        ar_polynomial_matrix=ar_polynomial_matrix,
        ma_polynomial_matrix=ma_polynomial_matrix,
        other=other,
        other_parameter_estimate=other_parameter_estimate,
    )

    # Set objective function
    opt.set_min_objective(objective_wrapper)

    # Step 9: Run optimization
    B[:] = _run_optimization(opt, B)
    CF_value = opt.last_optimum_value()

    # Step 9a: a BOBYQA refinement pass was tried in the past but produced
    # less stable ARIMA estimates, so only the Nelder-Mead pass is run.

    # Step 10a: Retry optimisation with zero smoothing parameters if the
    # initial optimisation failed to converge (non-finite or penalty-valued
    # cost function).
    if not np.isfinite(CF_value) or CF_value >= 1e300:
        # Calculate number of ETS persistence parameters (alpha, beta, gamma)
        components_number_ets = 0
        if model_type_dict["ets_model"]:
            # Build persistence estimate vector with proper seasonal expansion
            persistence_estimate_vector = [
                persistence_dict["persistence_level_estimate"],
                model_type_dict["model_is_trendy"]
                and persistence_dict["persistence_trend_estimate"],
            ]
            if model_type_dict["model_is_seasonal"]:
                persistence_estimate_vector.extend(
                    persistence_dict["persistence_seasonal_estimate"]
                )
            components_number_ets = sum(persistence_estimate_vector)
            if components_number_ets > 0:
                B[:components_number_ets] = 0

        if arima_dict["arima_model"]:
            # Calculate starting index for ARIMA parameters
            #  Match R's calculation exactly: componentsNumberETS +
            # persistenceXregEstimate*xregNumber
            # Note: R's retry code doesn't account for phi, so we match that behavior
            ar_ma_start = components_number_ets
            if (
                explanatory_dict["xreg_model"]
                and persistence_dict["persistence_xreg_estimate"]
            ):
                ar_ma_start += max(
                    explanatory_dict["xreg_parameters_persistence"] or [0]
                )

            # Calculate number of ARIMA parameters
            ar_orders = arima_dict.get("ar_orders", [])
            ma_orders = arima_dict.get("ma_orders", [])
            ar_estimate = arima_dict.get("ar_estimate", False)
            ma_estimate = arima_dict.get("ma_estimate", False)

            ar_count = sum(ar_orders) if ar_estimate else 0
            ma_count = sum(ma_orders) if ma_estimate else 0
            ar_ma_count = ar_count + ma_count

            if ar_ma_count > 0:
                B[ar_ma_start : ar_ma_start + ar_ma_count] = 0.01

        # Retry optimization with reset parameters
        opt2 = nlopt.opt(nlopt_algorithm, len(B))
        opt2 = _configure_optimizer(
            opt2,
            lb,
            ub,
            maxeval_used,
            maxtime,
            xtol_rel=xtol_rel,
            xtol_abs=xtol_abs,
            ftol_rel=ftol_rel,
            ftol_abs=ftol_abs,
        )
        opt2.set_min_objective(objective_wrapper)
        B[:] = _run_optimization(opt2, B)
        CF_value = opt2.last_optimum_value()

    # A fix for the special case of LASSO/RIDGE with lambda==1
    if (
        any(general_dict["loss"] == loss_type for loss_type in ["LASSO", "RIDGE"])
        and general_dict["lambda"] == 1
    ):
        CF_value = 0

    n_param_estimated = len(B)

    # For initial="gradient" the ETS initials are not in B, but they are
    # genuinely solved (by least squares / Gauss-Newton), so they count as
    # degrees of freedom just like initial="optimal". This applies only in the
    # gradient solve's scope (pure ETS); with ARIMA or xreg present gradient
    # falls back to backcasting and keeps the historical zero-initial-df count,
    # as do backcasting / complete. Mirrors R (R/adam.R).
    gradient_in_scope = (
        model_type_dict["ets_model"]
        and not arima_dict["arima_model"]
        and not model_type_dict.get("xreg_model", False)
    )
    if initials_dict["initial_type"] == "gradient" and gradient_in_scope:
        seasonal_estimate = initials_dict.get("initial_seasonal_estimate", [])
        if not isinstance(seasonal_estimate, (list, tuple, np.ndarray)):
            seasonal_estimate = [seasonal_estimate]
        lags_model_seasonal = lags_dict.get("lags_model_seasonal", []) or []
        seasonal_df = sum(
            bool(est) * (int(lag) - 1)
            for est, lag in zip(seasonal_estimate, lags_model_seasonal)
        )
        n_states_backcasting = (
            bool(initials_dict.get("initial_level_estimate", True))
            + model_type_dict["model_is_trendy"]
            * bool(initials_dict.get("initial_trend_estimate", True))
            + model_type_dict["model_is_seasonal"] * seasonal_df
        )
        n_param_estimated += int(n_states_backcasting)

    # Step 11: Calculate log-likelihood
    log_lik_adam_value = _calculate_loglik(
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
        multisteps,
        n_param_estimated,
        other_parameter_estimate=other_parameter_estimate,
    )

    # Step 11a: Fisher Information matrix (observed), if requested. Mirrors the
    # FI block in R/adam.R:2698-2836 — the negative Hessian of the
    # log-likelihood at the estimated B, computed in the same scope so the
    # estimation-time flags/matrices match the fitted parameters.
    FI = None
    if fi and not multisteps and general_dict["loss"] not in ("LASSO", "RIDGE"):
        from smooth.adam_general.core.utils.var_covar import fisher_information

        FI = fisher_information(
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
            step_size=step_size,
            other=other,
            other_parameter_estimate=other_parameter_estimate,
        )

    # Step 12: Prepare and return results
    result = {
        "B": B,
        "other_parameter_estimate": other_parameter_estimate,
        "CF_value": CF_value,
        "n_param_estimated": n_param_estimated,
        "log_lik_adam_value": log_lik_adam_value,
        "arima_polynomials": adam_created["arima_polynomials"],
        "adam_cpp": adam_cpp,  # Always return adam_cpp for forecasting
        "FI": FI,
        "B_names": list(b_values["names"]),
    }

    if return_matrices:
        # Ensure matrices are updated with final B values and backcasted states
        # The CF function uses copies, so we need to update the originals

        # Fill matrices with final B
        filler(
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
            adam_cpp=adam_cpp,
        )

        # Run the backcasting fit to update states. In-scope ETS gradient is
        # excluded: its initials are re-solved from the pristine msdecompose seed
        # inside preparator(), and running the fit here would overwrite the seed
        # head with evolved states, breaking the (seed-dependent) multiplicative
        # Gauss-Newton solve downstream. Out-of-scope gradient (ARIMA / xreg)
        # falls back to backcasting, so it needs this refresh just like
        # backcasting / complete.
        run_state_refit = initials_dict["initial_type"] in [
            "complete",
            "backcasting",
        ] or (initials_dict["initial_type"] == "gradient" and not gradient_in_scope)
        if run_state_refit:
            mat_vt = np.asfortranarray(adam_created["mat_vt"], dtype=np.float64)
            mat_wt = np.asfortranarray(adam_created["mat_wt"], dtype=np.float64)
            mat_f = np.asfortranarray(adam_created["mat_f"], dtype=np.float64)
            vec_g = np.asfortranarray(adam_created["vec_g"], dtype=np.float64)
            index_lookup_table = np.asfortranarray(
                profile_dict["index_lookup_table"], dtype=np.uint64
            )
            profiles_recent_table = np.asfortranarray(
                profile_dict["profiles_recent_table"], dtype=np.float64
            )
            y_in_sample = np.asfortranarray(
                observations_dict["y_in_sample"], dtype=np.float64
            )
            ot = np.asfortranarray(observations_dict["ot"], dtype=np.float64)

            adam_fit_or_gradient(
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
                n_iterations=initials_dict.get("n_iterations", 2) or 2,
                backcast_value=True,
                model_type_dict=model_type_dict,
                components_dict=components_dict,
                lags_dict=lags_dict,
                obs_in_sample=observations_dict["obs_in_sample"],
            )

            # Update original matrices
            adam_created["mat_vt"][:] = mat_vt[:]

        result["matrices"] = adam_created
        result["lags_dict"] = lags_dict
        result["profile_dict"] = profile_dict
        result["components_dict"] = components_dict

    return result
