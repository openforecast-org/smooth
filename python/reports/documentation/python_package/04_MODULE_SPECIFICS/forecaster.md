# Python: forecaster Module

**Path**: python/src/smooth/adam_general/core/forecaster/

## Structure

| File | Purpose |
|------|---------|
| forecaster.py | forecaster() |
| preparator.py | preparator() |
| intervals.py | generate_prediction_interval, generate_simulation_interval |
| _helpers.py | _prepare_lookup_table, _prepare_matrices_for_forecast |

## preparator

Prepares fitted model for forecasting:
- _fill_matrices_if_needed (calls filler if matrices not final)
- _prepare_profiles_recent_table
- _prepare_fitter_inputs (run adam_fitter if need in-sample states)
- _initialize_fitted_series
- _process_arma_parameters, _calculate_scale_parameter

## forecaster

1. _prepare_forecast_index
2. _check_fitted_values
3. _initialize_forecast_series
4. _prepare_lookup_table, _prepare_matrices_for_forecast
5. _generate_point_forecasts → adam_forecaster()
6. _handle_forecast_safety_checks
7. _process_occurrence_forecast if occurrence
8. _prepare_forecast_intervals (parametric or simulation)
9. _format_forecast_output

## intervals

- generate_prediction_interval: Parametric intervals (sigma, covar_anal, var_anal)
- generate_simulation_interval: Simulation-based intervals

## R Mapping

- R forecasterwrap (Rcpp) → adam_forecaster
- R ssIntervals → intervals.generate_prediction_interval
- R preparator logic inside adam/forecast.adam → preparator
