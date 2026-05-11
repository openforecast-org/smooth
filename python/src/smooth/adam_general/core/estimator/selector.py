import math

from smooth.adam_general.core.utils.ic import ic_function

from .estimator import estimator


def _form_model_pool(model_type_dict, silent=False, lags_dict=None):
    """
    Form a pool of models based on model type specifications.

    Parameters
    ----------
    model_type_dict : dict
        Model type specification
    silent : bool, optional
        Whether to suppress progress messages
    lags_dict : dict, optional
        Lags information including lags_model_max

    Returns
    -------
    tuple
        pool_small, pool_errors, pool_trends, pool_seasonals, check_trend,
        check_seasonal
    """
    # Check if the pool was provided
    if model_type_dict["models_pool"] is not None:
        return model_type_dict["models_pool"], None, None, None, None, None

    # Print status if not silent
    if not silent:
        print("Forming the pool of models based on... ", end="")

    # Check if seasonal models are possible based on lags
    max_lag = lags_dict.get("lags_model_max", 1) if lags_dict else 1
    has_seasonality = max_lag > 1

    # Define the whole pool of errors, trends, and seasonals
    if not model_type_dict["allow_multiplicative"]:
        pool_errors = ["A"]
        pool_trends = ["N", "A", "Ad"]
        pool_seasonals = ["N", "A"] if has_seasonality else ["N"]
    else:
        pool_errors = ["A", "M"]
        pool_trends = ["N", "A", "Ad", "M", "Md"]
        pool_seasonals = ["N", "A", "M"] if has_seasonality else ["N"]

    # Prepare error type
    if model_type_dict["error_type"] != "Z":
        if model_type_dict["error_type"] == "X":
            pool_errors = ["A"]
            pool_errors_small = "A"
        elif model_type_dict["error_type"] == "Y":
            pool_errors = ["M"]
            pool_errors_small = "M"
        else:
            pool_errors = [model_type_dict["error_type"]]
            pool_errors_small = model_type_dict["error_type"]
    else:
        pool_errors_small = "A"

    # Prepare trend type
    if model_type_dict["trend_type"] != "Z":
        if model_type_dict["trend_type"] == "X":
            pool_trends_small = ["N", "A"]
            pool_trends = ["N", "A", "Ad"]
            check_trend = True
        elif model_type_dict["trend_type"] == "Y":
            pool_trends_small = ["N", "M"]
            pool_trends = ["N", "M", "Md"]
            check_trend = True
        else:
            if model_type_dict["damped"]:
                pool_trends = pool_trends_small = [model_type_dict["trend_type"] + "d"]
            else:
                pool_trends = pool_trends_small = [model_type_dict["trend_type"]]
            check_trend = False
    else:
        pool_trends_small = ["N", "A"]
        check_trend = True

    # Prepare seasonal type
    # If no seasonality is possible (max_lag <= 1), force to "N"
    if not has_seasonality:
        pool_seasonals = pool_seasonals_small = ["N"]
        check_seasonal = False
    elif model_type_dict["season_type"] != "Z":
        if model_type_dict["season_type"] == "X":
            pool_seasonals = pool_seasonals_small = ["N", "A"]
            check_seasonal = True
        elif model_type_dict["season_type"] == "Y":
            pool_seasonals_small = ["N", "M"]
            pool_seasonals = ["N", "M"]
            check_seasonal = True
        else:
            pool_seasonals_small = [model_type_dict["season_type"]]
            pool_seasonals = [model_type_dict["season_type"]]
            check_seasonal = False
    else:
        pool_seasonals_small = ["N", "A", "M"]
        check_seasonal = True

    # Create the small pool
    pool_small = []
    for error in pool_errors_small:
        for trend in pool_trends_small:
            for seasonal in pool_seasonals_small:
                pool_small.append(error + trend + seasonal)

    # Align error and seasonality, if the error was not forced to be additive
    # For 3-character models, the seasonal is at index 2 (model[2])
    # For 4-character models, the seasonal is the last character (model[-1])
    #  Since pool_small only contains 3-character models at this stage, use model[-1]
    # for safety
    if any(model[-1] == "M" for model in pool_small) and model_type_dict[
        "error_type"
    ] not in ["A", "X"]:
        for i, model in enumerate(pool_small):
            if model[-1] == "M":
                pool_small[i] = "M" + model[1:]

    return (
        pool_small,
        pool_errors,
        pool_trends,
        pool_seasonals,
        check_trend,
        check_seasonal,
    )


def _estimate_model(
    model_type_dict_temp,
    phi_dict_temp,
    general_dict,
    lags_dict,
    observations_dict,
    arima_dict,
    constant_dict,
    explanatory_dict,
    profiles_recent_table,
    profiles_recent_provided,
    persistence_results,
    initials_results,
    occurrence_dict,
    components_dict,
    nlopt_kargs=None,
    smoother="global",
):
    """
    Estimate a single model and calculate its information criterion.

    Parameters
    ----------
    model_type_dict_temp : dict
        Temporary model type dictionary for this model
    phi_dict_temp : dict
        Temporary phi dictionary for this model
    general_dict : dict
        General model parameters
    lags_dict : dict
        Lags information
    observations_dict : dict
        Observations information
    arima_dict : dict
        ARIMA components specification
    constant_dict : dict
        Constant term specification
    explanatory_dict : dict
        Explanatory variables specification
    profiles_recent_table : array-like
        Recent profiles table
    profiles_recent_provided : bool
        Whether profiles were provided by user
    persistence_results : dict
        Persistence parameters
    initials_results : dict
        Initial values
    occurrence_dict : dict
        Occurrence model information
    components_dict : dict
        Components information

    Returns
    -------
    dict
        Dictionary containing estimation results and model information
    """
    # Estimate the model
    adam_estimated = estimator(
        general_dict=general_dict,
        model_type_dict=model_type_dict_temp,
        lags_dict=lags_dict,
        observations_dict=observations_dict,
        arima_dict=arima_dict,
        constant_dict=constant_dict,
        explanatory_dict=explanatory_dict,
        profiles_recent_table=profiles_recent_table,
        profiles_recent_provided=profiles_recent_provided,
        persistence_dict=persistence_results,
        initials_dict=initials_results,
        occurrence_dict=occurrence_dict,
        phi_dict=phi_dict_temp,
        components_dict=components_dict,
        **(nlopt_kargs or {}),
        smoother=smoother,
    )

    # Calculate information criterion
    IC = ic_function(general_dict["ic"], loglik=adam_estimated["log_lik_adam_value"])

    # Return results
    return {
        "adam_estimated": adam_estimated,
        "IC": IC,
        "model_type_dict": model_type_dict_temp,
        "phi_dict": phi_dict_temp,
        "model": model_type_dict_temp["model"],
    }


def _run_branch_and_bound(
    pool_small,
    model_type_dict,
    phi_dict,
    general_dict,
    lags_dict,
    observations_dict,
    arima_dict,
    constant_dict,
    explanatory_dict,
    profiles_recent_table,
    profiles_recent_provided,
    persistence_results,
    initials_results,
    occurrence_dict,
    components_dict,
    pool_seasonals,
    pool_trends,
    check_seasonal,
    check_trend,
    nlopt_kargs=None,
    smoother="global",
    silent=False,
):
    """
    Run branch and bound algorithm to efficiently search model space.

    Parameters
    ----------
    pool_small : list
        Small pool of models to test
    model_type_dict : dict
        Model type specification
    phi_dict : dict
        Damping parameter specification
    general_dict : dict
        General model parameters
    lags_dict : dict
        Lags information
    observations_dict : dict
        Observations information
    arima_dict : dict
        ARIMA components specification
    constant_dict : dict
        Constant term specification
    explanatory_dict : dict
        Explanatory variables specification
    profiles_recent_table : array-like
        Recent profiles table
    profiles_recent_provided : bool
        Whether profiles were provided by user
    persistence_results : dict
        Persistence parameters
    initials_results : dict
        Initial values
    occurrence_dict : dict
        Occurrence model information
    components_dict : dict
        Components information
    pool_seasonals : list
        Pool of possible seasonal components
    pool_trends : list
        Pool of possible trend components
    check_seasonal : bool
        Whether to check seasonal components
    check_trend : bool
        Whether to check trend components

    Returns
    -------
    tuple
        (results, models_tested, pool_seasonals, pool_trends)
        - results: list of estimation results for tested models
        - models_tested: list of model strings that were tested
        - pool_seasonals: updated list of seasonal components to include in final pool
        - pool_trends: updated list of trend components to include in final pool
    """

    # Helper function to get seasonal type from model string (last character)
    def get_seasonal(model_str):
        return model_str[-1]

    # Helper function to get trend type from model string (second character)
    def get_trend(model_str):
        return model_str[1]

    # Helper function to find model index in pool_small
    def find_model_index(pool, seasonal=None, trend=None, error=None, exclude=None):
        """Find first model matching criteria, returns 1-indexed position or None."""
        if exclude is None:
            exclude = []
        for k, model in enumerate(pool):
            if model in exclude:
                continue
            if seasonal is not None and get_seasonal(model) != seasonal:
                continue
            if trend is not None and get_trend(model) != trend:
                continue
            if error is not None and model[0] != error:
                continue
            return k + 1  # 1-indexed
        return None

    # Helper function to estimate a model and store results
    def estimate_and_store(model_str, results_list, results_dict):
        """Estimate a model if not already tested, return results index."""
        if model_str in results_dict:
            return results_dict[model_str]

        idx = len([r for r in results_list if r is not None])

        model_type_dict_temp = model_type_dict.copy()
        model_type_dict_temp["model"] = model_str
        phi_dict_temp = phi_dict.copy()

        e_type = model_str[0]
        t_type = model_str[1]
        s_type = model_str[-1]

        model_type_dict_temp["error_type"] = e_type
        model_type_dict_temp["trend_type"] = t_type
        model_type_dict_temp["season_type"] = s_type

        if len(model_str) == 4:
            phi_dict_temp["phi"] = 0.95
            phi_dict_temp["phi_estimate"] = True
        else:
            phi_dict_temp["phi"] = 1
            phi_dict_temp["phi_estimate"] = False

        result = _estimate_model(
            model_type_dict_temp,
            phi_dict_temp,
            general_dict,
            lags_dict,
            observations_dict,
            arima_dict,
            constant_dict,
            explanatory_dict,
            profiles_recent_table,
            profiles_recent_provided,
            persistence_results,
            initials_results,
            occurrence_dict,
            components_dict,
            nlopt_kargs=nlopt_kargs,
            smoother=smoother,
        )

        result["Etype"] = e_type
        result["Ttype"] = t_type
        result["Stype"] = s_type
        result["model"] = model_str

        if phi_dict_temp["phi_estimate"]:
            result["phi_dict"]["phi"] = result["adam_estimated"]["B"].get("phi", 0.95)
        else:
            result["phi_dict"]["phi"] = 1

        results_list.append(result)
        results_dict[model_str] = idx
        return idx

    # Ensure pool_seasonals and pool_trends are lists
    if isinstance(pool_seasonals, str):
        pool_seasonals = [pool_seasonals]
    if isinstance(pool_trends, str):
        pool_trends = [pool_trends]

    # Make copies to avoid modifying the original lists
    pool_seasonals = list(pool_seasonals)
    pool_trends = list(pool_trends)

    # Initialize
    results = []
    results_dict = {}  # model_str -> index in results
    models_tested = []

    # Branch and bound algorithm
    # Step 1: Estimate baseline model (ANN or equivalent)
    baseline_model = pool_small[0]  # Typically "ANN"
    baseline_idx = estimate_and_store(baseline_model, results, results_dict)
    models_tested.append(baseline_model)
    best_ic = results[baseline_idx]["IC"]

    # Step 2: Check seasonality (if check_seasonal is True)
    if check_seasonal and len(pool_seasonals) > 1:
        # Find model with additive seasonality and same trend as baseline
        seasonal_model_a = find_model_index(
            pool_small,
            seasonal="A",
            trend=get_trend(baseline_model),
            exclude=models_tested,
        )

        if seasonal_model_a is not None:
            model_a = pool_small[seasonal_model_a - 1]
            idx_a = estimate_and_store(model_a, results, results_dict)
            models_tested.append(model_a)
            ic_a = results[idx_a]["IC"]

            if ic_a < best_ic:
                # Seasonality helps - remove "N" from pool, check multiplicative
                pool_seasonals = [s for s in pool_seasonals if s != "N"]
                best_ic = ic_a

                # Check multiplicative seasonality if available
                if "M" in pool_seasonals:
                    seasonal_model_m = find_model_index(
                        pool_small,
                        seasonal="M",
                        trend=get_trend(baseline_model),
                        exclude=models_tested,
                    )
                    if seasonal_model_m is not None:
                        model_m = pool_small[seasonal_model_m - 1]
                        idx_m = estimate_and_store(model_m, results, results_dict)
                        models_tested.append(model_m)
                        ic_m = results[idx_m]["IC"]

                        if ic_m < best_ic:
                            # Multiplicative is better
                            pool_seasonals = ["M"]
                            best_ic = ic_m
                        else:
                            # Additive is better
                            pool_seasonals = ["A"]
                else:
                    pool_seasonals = ["A"]

                # Now check trend with the selected seasonal
                if check_trend and len(pool_trends) > 1:
                    trend_model = find_model_index(
                        pool_small,
                        seasonal=pool_seasonals[0],
                        trend="A",
                        exclude=models_tested,
                    )
                    if trend_model is not None:
                        model_t = pool_small[trend_model - 1]
                        idx_t = estimate_and_store(model_t, results, results_dict)
                        models_tested.append(model_t)
                        ic_t = results[idx_t]["IC"]

                        if ic_t < best_ic:
                            # Trend helps - keep all trend options
                            pool_trends = [t for t in pool_trends if t != "N"]
                        else:
                            # No trend needed
                            pool_trends = ["N"]
            else:
                # No seasonality - pool_seasonals = ["N"]
                pool_seasonals = ["N"]

                # Check trend: AAN vs ANN
                if check_trend and len(pool_trends) > 1:
                    trend_model = find_model_index(
                        pool_small, seasonal="N", trend="A", exclude=models_tested
                    )
                    if trend_model is not None:
                        model_t = pool_small[trend_model - 1]
                        idx_t = estimate_and_store(model_t, results, results_dict)
                        models_tested.append(model_t)
                        ic_t = results[idx_t]["IC"]

                        if ic_t < best_ic:
                            # Trend helps - keep all trend options
                            pool_trends = [t for t in pool_trends if t != "N"]
                            best_ic = ic_t
                        else:
                            # No trend helps - only check MNN for error type
                            pool_trends = ["N"]

                            # Check MNN if multiplicative error is allowed
                            if model_type_dict.get("allow_multiplicative", True):
                                error_model = find_model_index(
                                    pool_small,
                                    seasonal="N",
                                    trend="N",
                                    error="M",
                                    exclude=models_tested,
                                )
                                if error_model is not None:
                                    model_e = pool_small[error_model - 1]
                                    estimate_and_store(model_e, results, results_dict)
                                    models_tested.append(model_e)
                                    #  IC comparison for error type will be done in full
                                    # pool estimation
        else:
            # No seasonal model to test, keep checking trend
            pool_seasonals = ["N"]
    elif not check_seasonal:
        # Seasonality already determined, check trend
        if check_trend and len(pool_trends) > 1:
            trend_model = find_model_index(
                pool_small, seasonal=pool_seasonals[0], trend="A", exclude=models_tested
            )
            if trend_model is not None:
                model_t = pool_small[trend_model - 1]
                idx_t = estimate_and_store(model_t, results, results_dict)
                models_tested.append(model_t)
                ic_t = results[idx_t]["IC"]

                if ic_t < best_ic:
                    pool_trends = [t for t in pool_trends if t != "N"]
                else:
                    pool_trends = ["N"]

    return results, models_tested, pool_seasonals, pool_trends


def _estimate_all_models(
    models_pool,
    model_type_dict,
    phi_dict,
    general_dict,
    lags_dict,
    observations_dict,
    arima_dict,
    constant_dict,
    explanatory_dict,
    profiles_recent_table,
    profiles_recent_provided,
    persistence_results,
    initials_results,
    occurrence_dict,
    components_dict,
    silent=False,
    nlopt_kargs=None,
    # Pre-computed results from branch-and-bound
    precomputed_results=None,
    precomputed_models=None,
    smoother="global",
):
    """
    Estimate all models in the provided pool.

    Parameters
    ----------
    models_pool : list
        List of models to estimate
    model_type_dict : dict
        Model type specification
    phi_dict : dict
        Damping parameter specification
    general_dict : dict
        General model parameters
    lags_dict : dict
        Lags information
    observations_dict : dict
        Observations information
    arima_dict : dict
        ARIMA components specification
    constant_dict : dict
        Constant term specification
    explanatory_dict : dict
        Explanatory variables specification
    profiles_recent_table : array-like
        Recent profiles table
    profiles_recent_provided : bool
        Whether profiles were provided by user
    persistence_results : dict
        Persistence parameters
    initials_results : dict
        Initial values
    occurrence_dict : dict
        Occurrence model information
    components_dict : dict
        Components information
    silent : bool, optional
        Whether to suppress progress messages
    precomputed_results : list, optional
        Results from branch-and-bound for models already estimated
    precomputed_models : list, optional
        List of model strings already estimated in branch-and-bound

    Returns
    -------
    list
        List of results for each model
    """
    # Build a dictionary of precomputed results
    precomputed_dict = {}
    if precomputed_results is not None and precomputed_models is not None:
        for model_str, result in zip(precomputed_models, precomputed_results):
            if result is not None:
                precomputed_dict[model_str] = result

    models_number = len(models_pool)
    results = [None] * models_number

    # Print progress message if not silent
    if not silent:
        print("Estimation progress:    ", end="")

    # Estimate each model
    for j in range(models_number):
        if not silent:
            if j == 0:
                print("\b", end="")
            print("\b" * (len(str(round((j) / models_number * 100))) + 1), end="")
            print(f"{round((j + 1) / models_number * 100)}%", end="")

        model_current = models_pool[j]

        # Check if this model was already estimated in branch-and-bound
        if model_current in precomputed_dict:
            results[j] = precomputed_dict[model_current]
            continue

        # Create copies for this model
        model_type_dict_temp = model_type_dict.copy()
        phi_dict_temp = phi_dict.copy()

        # Set model parameters
        model_type_dict_temp["error_type"] = model_current[0]
        model_type_dict_temp["trend_type"] = model_current[1]

        if len(model_current) == 4:
            # 4-character model means damped (e.g., "AAdN")
            phi_dict_temp["phi"] = 0.95
            model_type_dict_temp["season_type"] = model_current[3]
            model_type_dict_temp["damped"] = True
            phi_dict_temp["phi_estimate"] = True
        else:
            # 3-character model means not damped (e.g., "AAN")
            phi_dict_temp["phi"] = 1
            model_type_dict_temp["season_type"] = model_current[2]
            model_type_dict_temp["damped"] = False
            phi_dict_temp["phi_estimate"] = False

        # Estimate the model
        results[j] = {}
        results[j]["adam_estimated"] = estimator(
            general_dict=general_dict,
            model_type_dict=model_type_dict_temp,
            lags_dict=lags_dict,
            observations_dict=observations_dict,
            arima_dict=arima_dict,
            constant_dict=constant_dict,
            explanatory_dict=explanatory_dict,
            profiles_recent_table=profiles_recent_table,
            profiles_recent_provided=profiles_recent_provided,
            persistence_dict=persistence_results,
            initials_dict=initials_results,
            occurrence_dict=occurrence_dict,
            phi_dict=phi_dict_temp,
            components_dict=components_dict,
            **(nlopt_kargs or {}),
            smoother=smoother,
        )
        results[j]["IC"] = ic_function(
            general_dict["ic"],
            loglik=results[j]["adam_estimated"]["log_lik_adam_value"],
        )
        results[j]["model_type_dict"] = model_type_dict_temp
        results[j]["phi_dict"] = phi_dict_temp
        results[j]["model"] = model_current
    if not silent:
        print("... Done!")

    return results


def selector(
    model_type_dict,
    phi_dict,
    general_dict,
    lags_dict,
    observations_dict,
    arima_dict,
    constant_dict,
    explanatory_dict,
    occurrence_dict,
    components_dict,
    profiles_recent_table,
    profiles_recent_provided,
    persistence_results,
    initials_results,
    criterion="AICc",
    silent=False,
    nlopt_kargs=None,
    smoother="global",
):
    """
    Automatic model selection for ADAM using information criteria and Branch & Bound.

    This function implements the automatic model selection procedure for ADAM models
    by creating a pool of candidate models and selecting the best one based on an
    information criterion (AIC, AICc, BIC, or BICc). The selection process uses a
    **Branch and Bound algorithm** to efficiently explore the model space without
    estimating every possible combination.

    The function supports several selection modes triggered by special codes in the
    model specification string:

    - **"ZZZ"**: Select from all possible models using Branch and Bound
    - **"XXX"**: Select only additive components (A, N)
    - **"YYY"**: Select only multiplicative components (M, N)
    - **"PPP"**: Select between pure additive (AAA) and pure multiplicative (MMM)
    - **"FFF"**: Full search across all 30 ETS model types
    - **"CCC"**: Combine models using IC weights (not fully implemented in Python yet)
    - Mixed codes like **"ZXZ"**: Auto-select error and seasonality, only additive trend

    **Selection Algorithm**:

    1. **Pool Formation**: Create initial model pool based on specification:

       - Identify which components (error, trend, seasonality) need selection
       - Generate small pool for Branch and Bound exploration
       - Generate full pool for final estimation

    2. **Branch and Bound** (if applicable):

       - Start with simplest model (typically "ANN")
       - Iteratively explore promising branches (add/change components)
       - Prune branches that cannot improve upon best IC found
       - Significantly faster than exhaustive search

    3. **Full Estimation**: Estimate all models in the final pool

    4. **Selection**: Choose model with best (lowest) information criterion

    **Model Pool Generation**:

    The pool size depends on the specification:

    - **"ZZZ"**: Up to 30 models (5 errors × 3 trends × 2 seasonalities)
    - **"XXX"**: Up to 6 models (additive-only: A × {N,A,Ad} × {N,A})
    - **"YYY"**: Up to 10 models (multiplicative-focused)
    - **"FFF"**: All 30 models estimated exhaustively
    - Custom pool: Use provided list (e.g., ["ANN", "AAN", "AAA"])

    Parameters
    ----------
    model_type_dict : dict
        Model specification containing:

        - 'model': Model string with selection codes (e.g., "ZXZ", "FFF")
        - 'error_type': Error component ('Z' for auto-select, or 'A'/'M' for fixed)
        - 'trend_type': Trend component ('Z', 'X', 'Y', or specific like 'A', 'Ad', 'N')
        - 'season_type': Seasonality component ('Z', 'X', 'Y', or 'N', 'A', 'M')
        - 'models_pool': List of model strings (if pre-specified pool)
        - 'allow_multiplicative': Whether multiplicative models are allowed
        (data-dependent)
        - 'model_do': Action type ('select', 'combine', or 'estimate')

    phi_dict : dict
        Damping parameter specification containing:

        - 'phi_estimate': Whether to estimate damping
        - 'phi': Fixed damping value (if not estimated)

    general_dict : dict
        General configuration containing:

        - 'loss': Loss function ('likelihood', 'MSE', 'MAE', etc.)
        - 'distribution': Error distribution
        - 'bounds': Parameter bounds type
        - 'h': Forecast horizon
        - 'holdout': Whether holdout validation is used

    lags_dict : dict
        Lag structure containing:

        - 'lags': Primary lag vector (e.g., [1, 12])
        - 'lags_model': Lags for each component
        - 'lags_model_max': Maximum lag

    observations_dict : dict
        Time series data containing:

        - 'y_in_sample': Observed values
        - 'obs_in_sample': Number of observations
        - 'ot': Occurrence vector (for intermittent data)

    arima_dict : dict
        ARIMA specification containing:

        - 'arima_model': Whether ARIMA components exist
        - 'ar_orders': AR orders
        - 'ma_orders': MA orders
        - 'i_orders': Integration orders

    constant_dict : dict
        Constant term specification containing:

        - 'constant_required': Whether constant is included
        - 'constant_estimate': Whether to estimate constant

    explanatory_dict : dict
        External regressors specification containing:

        - 'xreg_model': Whether regressors are present
        - 'xreg_number': Number of regressors

    occurrence_dict : dict
        Intermittent demand specification containing:

        - 'occurrence_model': Whether occurrence model is used
        - 'occurrence_type': Type of occurrence model

    components_dict : dict
        Component counts (passed through to estimator)

    profiles_recent_table : numpy.ndarray
        Recent profile values for time-varying parameters

    profiles_recent_provided : bool
        Whether user provided custom profiles

    persistence_results : dict
        Persistence parameters containing:

        - 'persistence_estimate': Whether to estimate smoothing parameters
        - Fixed values for non-estimated parameters

    initials_results : dict
        Initial states specification containing:

        - 'initial_type': Initialization method
        - Flags for which initials to estimate

    criterion : str, default="AICc"
        Information criterion for model selection:

        - **"AIC"**: Akaike Information Criterion (penalizes complexity moderately)
        - **"AICc"**: Corrected AIC (recommended for small samples, **default**)
        - **"BIC"**: Bayesian Information Criterion (more parsimonious than AIC)
        - **"BICc"**: Corrected BIC

        Lower IC values indicate better models. AICc is default as it performs
        well across sample sizes and is the standard in forecast package.

    silent : bool, default=False
        Whether to suppress progress messages during model estimation.
        If False, prints which models are being estimated.

    Returns
    -------
    dict
        Dictionary containing selection results:

        - **'results'** (list of dict): Estimation results for each model in pool.
          Each dict contains:

          * 'model': Model string (e.g., "ANN", "MAM")
          * 'IC': Information criterion value
          * 'loglik': Log-likelihood
          * 'n_param': Number of parameters
          * 'B': Estimated parameter vector
          * Additional estimation outputs

        - **'ic_selection'** (dict): Mapping of model names to IC values.
          Keys are model strings, values are IC scores. NaN values are replaced
          with 1e100 for comparison purposes.

    Notes
    -----
    **Branch and Bound Algorithm**:

    The Branch and Bound method is a heuristic that exploits the nested structure
    of ETS models. It works by:

    1. Starting with the simplest model (no trend, no seasonality)
    2. "Branching" by adding/changing one component at a time
    3. "Bounding" by not exploring branches worse than current best + tolerance

    This can reduce the search from 30 models to ~10-15 models in typical cases,
    with minimal risk of missing the global optimum.

    **Performance Considerations**:

    - Branch and Bound: Estimates ~10-15 models typically
    - Full pool ("FFF"): Estimates all 30 models (slower but exhaustive)
    - Custom pool: Estimates only specified models (fastest)

    Estimation time is roughly proportional to number of models × optimization time
    per model. For large datasets or complex models, consider using a smaller pool.

    **Multiplicative Model Restrictions**:

    Multiplicative error and seasonality require strictly positive data. If data
    contains zeros or negatives, `allow_multiplicative=False` is automatically set,
    restricting the pool to additive models only.

    **Model Combination (CCC)**:

    The "CCC" option for combining model forecasts using IC weights is mentioned in
    R documentation but not fully implemented in Python version yet. Use "ZZZ" for
    selection instead.

    **Equivalent to R's auto.adam()**:

    This function implements Python equivalent of R's `auto.adam()` from smooth package.

    See Also
    --------
    estimator : Estimates a single model (called by selector for each candidate)
    _form_model_pool : Creates the pool of candidate models
    _run_branch_and_bound : Implements Branch & Bound algorithm
    _estimate_all_models : Estimates all models in the pool

    Examples
    --------
    Automatic selection with default AICc::

        >>> results = selector(
        ...     model_type_dict={'model': 'ZZZ', 'error_type': 'Z',
        ...                      'trend_type': 'Z', 'season_type': 'Z',
        ...                      'models_pool': None, ...},
        ...     general_dict={'loss': 'likelihood', 'h': 10, ...},
        ...     lags_dict={'lags': np.array([1, 12]), ...},
        ...     observations_dict={'y_in_sample': data, ...},
        ...     criterion='AICc',
        ...     ...
        ... )
        >>> best_model = min(results['ic_selection'], key=results['ic_selection'].get)
        >>> print(f"Best model: {best_model}, AICc:
        {results['ic_selection'][best_model]}")

    Select from additive models only::

        >>> results = selector(
        ...     model_type_dict={'model': 'XXX', 'error_type': 'X',
        ...                      'trend_type': 'X', 'season_type': 'X', ...},
        ...     criterion='BIC',  # Use BIC for more parsimonious selection
        ...     ...
        ... )
        >>> # Will only consider: ANN, AAN, AAdN, ANA, AAA, AAdA

    Use custom pool::

        >>> results = selector(
        ...     model_type_dict={'model': 'ZZZ',
        ...                      'models_pool': ['ANN', 'AAN', 'AAdN', 'ANA', 'AAA']},
        ...     criterion='AICc',
        ...     ...
        ... )
        >>> # Estimates only the 5 specified models
    """
    # Set the information criterion in general_dict
    general_dict["ic"] = criterion
    # Step 1: Form the model pool
    (
        pool_small,
        pool_errors,
        pool_trends,
        pool_seasonals,
        check_trend,
        check_seasonal,
    ) = _form_model_pool(model_type_dict, silent, lags_dict)
    # Step 2: Run branch and bound if pool was not provided

    # Initialize variables for precomputed results
    bb_results = None
    bb_models_tested = None

    # Check if we're in combination mode - skip B&B and use full pool
    is_combining = model_type_dict.get("model_do") == "combine"

    if model_type_dict["models_pool"] is None:
        if is_combining:
            # For combination, use full pool without Branch & Bound filtering
            # This generates all 30 models (2 errors × 5 trends × 3 seasonals)
            bb_models_tested = []
            # Keep original pools from _form_model_pool (not filtered by B&B)
        else:
            # Run branch and bound to select models for model selection
            bb_results, bb_models_tested, pool_seasonals, pool_trends = (
                _run_branch_and_bound(
                    pool_small,
                    model_type_dict,
                    phi_dict,
                    general_dict,
                    lags_dict,
                    observations_dict,
                    arima_dict,
                    constant_dict,
                    explanatory_dict,
                    profiles_recent_table,
                    profiles_recent_provided,
                    persistence_results,
                    initials_results,
                    occurrence_dict,
                    components_dict,
                    pool_seasonals,
                    pool_trends,
                    check_seasonal,
                    check_trend,
                    nlopt_kargs=nlopt_kargs,
                    smoother=smoother,
                    silent=silent,
                )
            )

        # Prepare a bigger pool based on the small one
        # Ensure pool_seasonals and pool_trends are lists
        if isinstance(pool_seasonals, str):
            pool_seasonals = [pool_seasonals]
        if isinstance(pool_trends, str):
            pool_trends = [pool_trends]

        # Generate models in the same order as R:
        # paste0(rep(poolErrors, each=len(poolTrends)*len(poolSeasonals)),
        #        poolTrends,
        #        rep(poolSeasonals, each=len(poolTrends)))
        # This is: for e in errors: for s in seasonals: for t in trends
        generated_models = [
            e + t + s
            for e in pool_errors
            for s in pool_seasonals  # outer loop (slower varying)
            for t in pool_trends  # inner loop (faster varying)
        ]

        # Use dict.fromkeys() to preserve insertion order while removing duplicates
        # (mimics R's unique(c(...)) behavior)
        combined_models = bb_models_tested + generated_models
        model_type_dict["models_pool"] = list(dict.fromkeys(combined_models))

    # Step 3: Estimate all models in the pool (skip already tested models)
    results = _estimate_all_models(
        model_type_dict["models_pool"],
        model_type_dict,
        phi_dict,
        general_dict,
        lags_dict,
        observations_dict,
        arima_dict,
        constant_dict,
        explanatory_dict,
        profiles_recent_table,
        profiles_recent_provided,
        persistence_results,
        initials_results,
        occurrence_dict,
        components_dict,
        silent,
        nlopt_kargs=nlopt_kargs,
        precomputed_results=bb_results,
        precomputed_models=bb_models_tested,
        smoother=smoother,
    )
    # print(results)

    # Step 4: Extract ICs and find the best model
    models_number = len(model_type_dict["models_pool"])
    ic_selection = [results[j]["IC"] for j in range(models_number)]
    # Create dictionary with model names and ICs
    ic_selection_dict = dict(zip(model_type_dict["models_pool"], ic_selection))
    # Replace NaN values with large number
    ic_selection = [1e100 if math.isnan(x) else x for x in ic_selection]

    return {"results": results, "ic_selection": ic_selection_dict}


# def _generate_forecasts(
#     general_dict,
#     observations_dict,
#     matrices_dict,
#     lags_dict,
#     profiles_dict,
#     model_type_dict,
#     components_dict,
#     explanatory_checked,
#     constants_checked,
# ):
#     """
#     Generate forecasts if the horizon is non-zero.
#
#     Parameters
#     ----------
#     general_dict : dict
#         General model parameters
#     observations_dict : dict
#         Observations information
#     matrices_dict : dict
#         Model matrices
#     lags_dict : dict
#         Lags information
#     profiles_dict : dict
#         Profiles information
#     model_type_dict : dict
#         Model type specification
#     components_dict : dict
#         Components information
#     explanatory_checked : dict
#         Explanatory variables specification
#     constants_checked : dict
#         Constant term specification
#
#     Returns
#     -------
#     pandas.Series
#         Forecasted values
#     """
#     # If horizon is zero, return an empty series
#     if general_dict["h"] <= 0:
#         if any(observations_dict.get("y_classes", []) == "ts"):
#             return pd.Series(
#                 [np.nan],
#                 index=pd.date_range(
#                     start=observations_dict["y_forecast_start"],
#                     periods=1,
#                     freq=observations_dict["y_frequency"],
#                 ),
#             )
#         else:
#             return pd.Series(
#                 np.full(general_dict["horizon"], np.nan),
#                 index=observations_dict["y_forecast_index"],
#             )
#
#     # Create forecast Series
#     if not isinstance(observations_dict.get("y_in_sample"), pd.Series):
#         y_forecast = pd.Series(
#             np.full(general_dict["h"], np.nan),
#             index=pd.date_range(
#                 start=observations_dict["y_forecast_start"],
#                 periods=general_dict["h"],
#                 freq=observations_dict["frequency"],
#             ),
#         )
#     else:
#         y_forecast = pd.Series(
#             np.full(general_dict["h"], np.nan),
#             index=observations_dict["y_forecast_index"],
#         )
#
#     # Generate forecasts
#     forecasts = adam_forecaster(
#         matrixWt=matrices_dict["mat_wt"][-general_dict["h"] :],
#         matrixF=matrices_dict["mat_f"],
#         lags=lags_dict["lags_model_all"],
#         indexLookupTable=profiles_dict["index_lookup_table"],
#         profilesRecent=profiles_dict["profiles_recent_table"],
#         E=model_type_dict["error_type"],
#         T=model_type_dict["trend_type"],
#         S=model_type_dict["season_type"],
#         nNonSeasonal=components_dict["components_number_ets"],
#         nSeasonal=components_dict["components_number_ets_seasonal"],
#         nArima=components_dict.get("components_number_arima", 0),
#         nXreg=explanatory_checked["xreg_number"],
#         constant=constants_checked["constant_required"],
#         horizon=general_dict["h"],
#     ).flatten()
#
#     # Fill in forecast values
#     y_forecast[:] = forecasts
#
#     # Replace NaNs with zeros
#     if np.any(np.isnan(y_forecast)):
#         y_forecast[np.isnan(y_forecast)] = 0
#
#     return y_forecast
#
#
# def _update_distribution(general_dict, model_type_dict):
#     """
#     Update distribution based on error term and loss function.
#
#     Parameters
#     ----------
#     general_dict : dict
#         General model parameters
#     model_type_dict : dict
#         Model type specification
#
#     Returns
#     -------
#     dict
#         Updated general_dict
#     """
#     general_dict_updated = general_dict.copy()
#
#     if general_dict["distribution"] == "default":
#         if general_dict["loss"] == "likelihood":
#             if model_type_dict["error_type"] == "A":
#                 general_dict_updated["distribution"] = "dnorm"
#             elif model_type_dict["error_type"] == "M":
#                 general_dict_updated["distribution"] = "dgamma"
#         elif general_dict["loss"] in ["MAEh", "MACE", "MAE"]:
#             general_dict_updated["distribution"] = "dlaplace"
#         elif general_dict["loss"] in ["HAMh", "CHAM", "HAM"]:
#             general_dict_updated["distribution"] = "ds"
#         elif general_dict["loss"] in ["MSEh", "MSCE", "MSE", "GPL"]:
#             general_dict_updated["distribution"] = "dnorm"
#         else:
#             general_dict_updated["distribution"] = "dnorm"
#
#     return general_dict_updated
#
