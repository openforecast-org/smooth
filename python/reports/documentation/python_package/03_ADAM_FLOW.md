# Python ADAM Flow

Call graph for ADAM.fit() and ADAM.predict().

## Fit Flow

```
ADAM.fit(y, X)
  │
  ├─► _check_parameters()
  │     └─► parameters_checker() [checker/parameters_checker.py]
  │           ├─► _check_model_composition()
  │           ├─► _check_arima()
  │           ├─► _check_lags()
  │           ├─► _check_persistence()
  │           ├─► _check_initial()
  │           └─► ... (data_checks, parameter_checks, organizers)
  │
  ├─► _execute_estimation() or _execute_selection()
  │     ├─► architector() [creator/architector.py]
  │     ├─► creator() [creator/creator.py]
  │     ├─► If selection: selector() → estimator() per candidate
  │     └─► estimator() [estimator/estimator.py]
  │           ├─► initialiser() [creator/initialiser.py]
  │           ├─► _create_objective_function()
  │           │     └─► CF() [utils/cost_functions.py]
  │           │           ├─► filler() [creator/filler.py]
  │           │           └─► adam_fitter() [_adam_general.py]
  │           └─► _run_optimization() [optimization.py]
  │
  └─► _prepare_results()
```

## Predict Flow

```
ADAM.predict(h, X)
  │
  ├─► _validate_prediction_inputs()
  ├─► _prepare_prediction_data()
  │     └─► preparator() [forecaster/preparator.py]
  │           ├─► _fill_matrices_if_needed() → filler() if needed
  │           ├─► _prepare_profiles_recent_table()
  │           ├─► _prepare_fitter_inputs() (in-sample if needed)
  │           └─► _initialize_fitted_series()
  │
  ├─► _execute_prediction()
  │     └─► forecaster() [forecaster/forecaster.py]
  │           ├─► _prepare_forecast_index()
  │           ├─► _prepare_lookup_table() [_helpers]
  │           ├─► _prepare_matrices_for_forecast()
  │           ├─► _generate_point_forecasts() → adam_forecaster()
  │           └─► _prepare_forecast_intervals() if needed
  │
  └─► return forecasts
```

## Key Function Roles

- **parameters_checker**: Single entry for all validation
- **architector**: Components, lags, profiles
- **creator**: mat_vt, mat_wt, mat_f, vec_g
- **filler**: Populate matrices from B (called in CF and preparator)
- **initialiser**: B, Bl, Bu for optimization
- **estimator**: NLopt loop, CF, adam_fitter
- **preparator**: Ready model for forecasting
- **forecaster**: adam_forecaster + intervals
