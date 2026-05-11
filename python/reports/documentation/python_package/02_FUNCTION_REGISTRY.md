# Python Package Function Registry

Every function in python/src/smooth/. Source: `grep -E "^def [a-zA-Z_][a-zA-Z0-9_]*\("` (includes CF, AIC, etc.).

| Function | Module | Line | Role |
|----------|--------|------|------|
| adam_fitter | _adam_general | 15 | C++ fit wrapper |
| adam_forecaster | _adam_general | 96 | C++ forecast wrapper |
| adam_simulator | _adam_general | 152 | C++ simulate wrapper |
| initialiser | initialiser | 10 | Initial B and bounds |
| _extract_initialiser_params | initialiser | 869 | Extract params |
| _calculate_initial_parameters_and_bounds | initialiser | 941 | Calculate bounds |
| lowess_r | utils | 18 | R-compatible LOWESS |
| msdecompose | utils | 260 | Multiple seasonal decomposition |
| calculate_acf | utils | 744 | ACF |
| calculate_pacf | utils | 761 | PACF |
| calculate_likelihood | utils | 778 | Likelihood |
| calculate_entropy | utils | 828 | Entropy |
| calculate_multistep_loss | utils | 856 | Multistep loss |
| scaler | utils | 893 | Error scaler |
| architector | architector | 8 | Model architecture |
| _setup_components | architector | 284 | ETS/ARIMA components |
| _setup_lags | architector | 348 | Lag structure |
| _create_profiles | architector | 403 | Profiles |
| adam_profile_creator | architector | 442 | Profile/lookup creator |
| filler | filler | 6 | Fill matrices from B |
| adam_polynomialiser | polynomials | 11 | ARIMA polynomials |
| _fill_matrices_if_needed | preparator | 10 | Fill if needed |
| _prepare_profiles_recent_table | preparator | 79 | Profiles table |
| _prepare_fitter_inputs | preparator | 105 | Fitter inputs |
| _correct_multiplicative_components | preparator | 159 | Multiplicative correction |
| _initialize_fitted_series | preparator | 256 | Fitted series |
| _update_distribution | preparator | 296 | Distribution update |
| _process_initial_values | preparator | 330 | Initial values |
| _process_arma_parameters | preparator | 489 | ARMA params |
| _calculate_scale_parameter | preparator | 528 | Scale |
| _process_other_parameters | preparator | 566 | Other params |
| preparator | preparator | 642 | Main preparator |
| _setup_arima_polynomials | optimization | 8 | ARIMA polynomials |
| _set_distribution | optimization | 46 | Distribution |
| _setup_optimization_parameters | optimization | 81 | Opt params |
| _configure_optimizer | optimization | 167 | NLopt config |
| _create_objective_function | optimization | 230 | Objective |
| _run_optimization | optimization | 339 | Run opt |
| _calculate_loglik | optimization | 365 | Log-lik |
| _initialize_states | initialization | 6 | State init |
| _initialize_ets_states | initialization | 88 | ETS states |
| _initialize_ets_seasonal_states_with_decomp | initialization | 163 | Seasonal (decomp) |
| _initialize_ets_seasonal_states_small_sample | initialization | 345 | Seasonal (small) |
| _initialize_ets_nonseasonal_states | initialization | 450 | Non-seasonal ETS |
| _initialize_arima_states | initialization | 529 | ARIMA states |
| _initialize_xreg_states | initialization | 598 | Xreg states |
| _initialize_constant | initialization | 642 | Constant |
| _expand_component_code | model_checks | 6 | Expand component |
| _build_models_pool_from_components | model_checks | 61 | Build pool |
| _check_model_composition | model_checks | 231 | Model composition |
| _generate_models_pool | model_checks | 437 | Generate pool |
| _check_ets_model | model_checks | 544 | ETS model check |
| parameters_checker | parameters_checker | 32 | Main checker |
| estimator | estimator | 18 | Main estimator |
| _form_model_pool | selector | 8 | Form pool |
| _estimate_model | selector | 123 | Estimate one |
| _run_branch_and_bound | selector | 224 | B&B selection |
| _estimate_all_models | selector | 546 | Estimate all |
| selector | selector | 710 | Main selector |
| _prepare_forecast_index | forecaster | 20 | Forecast index |
| _check_fitted_values | forecaster | 67 | Fitted check |
| _initialize_forecast_series | forecaster | 109 | Forecast series |
| _determine_forecast_interval | forecaster | 145 | Interval type |
| _generate_point_forecasts | forecaster | 186 | Point forecasts |
| _handle_forecast_safety_checks | forecaster | 259 | Safety checks |
| _process_occurrence_forecast | forecaster | 306 | Occurrence |
| _prepare_forecast_intervals | forecaster | 365 | Intervals |
| _format_forecast_output | forecaster | 399 | Format output |
| forecaster | forecaster | 442 | Main forecaster |
| _safe_create_index | _helpers | 7 | Index creation |
| _prepare_lookup_table | _helpers | 48 | Lookup table |
| _prepare_matrices_for_forecast | _helpers | 78 | Matrices for forecast |
| ensure_level_format | intervals | 18 | Level format |
| generate_prediction_interval | intervals | 36 | Prediction interval |
| generate_simulation_interval | intervals | 444 | Simulation interval |
| _process_initial_values | initial_values | 4 | Initial values (estimator) |
| _run_two_stage_estimator | two_stage | 6 | Two-stage |
| _expand_orders | arima_checks | 4 | Expand orders |
| _check_arima | arima_checks | 52 | ARIMA check |
| _check_distribution_loss | parameter_checks | 6 | Dist/loss |
| _check_outliers | parameter_checks | 82 | Outliers |
| _check_phi | parameter_checks | 107 | Phi |
| _check_persistence | parameter_checks | 156 | Persistence |
| _check_initial | parameter_checks | 326 | Initial |
| _check_constant | parameter_checks | 507 | Constant |
| _initialize_estimation_params | parameter_checks | 555 | Init params |
| _check_occurrence | data_checks | 6 | Occurrence |
| _check_lags | data_checks | 77 | Lags |
| _calculate_ot_logical | data_checks | 140 | OT logical |
| _restrict_models_pool_for_sample_size | sample_size | 4 | Restrict pool |
| _adjust_model_for_sample_size | sample_size | 336 | Adjust model |
| _organize_model_type_info | organizers | 4 | Model type |
| _organize_components_info | organizers | 44 | Components |
| _organize_lags_info | organizers | 100 | Lags |
| _organize_occurrence_info | organizers | 141 | Occurrence |
| _organize_phi_info | organizers | 175 | Phi |
| _calculate_parameters_number | organizers | 194 | N param |
| _calculate_n_param_max | organizers | 253 | N param max |
| _warn | _utils | 1 | Warn |
| creator | creator | 15 | Main creator |
| _extract_model_parameters | creator | 356 | Extract params |
| _setup_matrices | creator | 440 | Setup matrices |
| _setup_measurement_vector | creator | 511 | Measurement |
| _setup_persistence_vector | creator | 555 | Persistence |
| _handle_polynomial_setup | creator | 638 | Polynomial setup |
| sigma | var_covar | 4 | Scale |
| covar_anal | var_covar | 212 | Covariance analytical |
| var_anal | var_covar | 485 | Variance analytical |
| matrix_power_wrap | var_covar | 644 | Matrix power |
| CF | cost_functions | 14 | Cost function for optimization |
| log_Lik_ADAM | cost_functions | 700 | Log-likelihood |
| AIC | ic | 4 | AIC |
| AICc | ic | 31 | AICc |
| BIC | ic | 66 | BIC |
| BICc | ic | 95 | BICc |
| ic_function | ic | 130 | IC (AIC, BIC, etc.) |
| rlaplace | distributions | 12 | Laplace RNG |
| rs | distributions | 43 | S dist RNG |
| rgnorm | distributions | 83 | Gnorm RNG |
| ralaplace | distributions | 123 | Asymmetric Laplace RNG |
| generate_errors | distributions | 165 | Error generation |
| normalize_errors | distributions | 276 | Normalize errors |
| create_n_param | n_param | 129 | N param table |
| count_internal_params | n_param | 181 | Count internal |
| count_xreg_params | n_param | 299 | Count xreg |
| build_n_param_table | n_param | 334 | Build table |
| lowess | lowess | 13 | LOWESS (user-facing) |
