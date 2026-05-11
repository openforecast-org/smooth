# Python: checker Module

**Path**: python/src/smooth/adam_general/core/checker/

## Structure

| File | Purpose |
|------|---------|
| parameters_checker.py | Main entry: parameters_checker() |
| model_checks.py | _check_model_composition, _check_ets_model, models pool |
| arima_checks.py | _check_arima, _expand_orders |
| data_checks.py | _check_occurrence, _check_lags, _calculate_ot_logical |
| parameter_checks.py | _check_persistence, _check_initial, _check_phi, _check_constant |
| sample_size.py | _restrict_models_pool_for_sample_size, _adjust_model_for_sample_size |
| organizers.py | _organize_model_type_info, _organize_lags_info, etc. |
| _utils.py | _warn |

## parameters_checker Flow

1. Process observations (y, X, holdout)
2. _check_model_composition (model string, Z/X/Y/C codes)
3. _check_arima (orders, lags)
4. _check_lags
5. _check_occurrence
6. _check_persistence
7. _check_initial
8. _check_phi
9. _check_constant
10. _check_distribution_loss
11. Organizers build model_type_dict, lags_dict, etc.
12. Returns dicts: model_type_dict, lags_dict, observations_dict, explanatory_checked, initials_checked, etc.

## R Mapping

- parametersChecker (adamGeneral.R) → parameters_checker
- Model/lag/initial checks scattered in R → organized into checker submodules
