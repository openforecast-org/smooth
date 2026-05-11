# Python Package File Index

Every Python module in `python/src/smooth/` with path, line count, purpose, and key exports.

| Path | Lines | Main Purpose | Key Exports |
|------|-------|--------------|-------------|
| smooth/__init__.py | 6 | Package init | ADAM, ES, msdecompose, lowess |
| smooth/adam_general/__init__.py | 7 | Subpackage init | - |
| smooth/adam_general/_adam_general.py | 219 | C++ bindings | adam_fitter, adam_forecaster, adam_simulator |
| smooth/adam_general/core/__init__.py | 4 | Core init | - |
| smooth/adam_general/core/adam.py | 2349 | Main ADAM class | ADAM |
| smooth/adam_general/core/es.py | 266 | ES wrapper | ES |
| smooth/adam_general/core/checker/__init__.py | 3 | Checker init | - |
| smooth/adam_general/core/checker/parameters_checker.py | 1047 | Main checker | parameters_checker |
| smooth/adam_general/core/checker/model_checks.py | 658 | Model composition | _check_model_composition, _check_ets_model |
| smooth/adam_general/core/checker/arima_checks.py | 272 | ARIMA validation | _check_arima |
| smooth/adam_general/core/checker/data_checks.py | 282 | Data/occurrence | _check_occurrence, _check_lags |
| smooth/adam_general/core/checker/parameter_checks.py | 596 | Params | _check_persistence, _check_initial |
| smooth/adam_general/core/checker/sample_size.py | 435 | Sample size | _restrict_models_pool_for_sample_size |
| smooth/adam_general/core/checker/organizers.py | 411 | Organizers | _organize_model_type_info |
| smooth/adam_general/core/checker/_utils.py | 13 | Utils | _warn |
| smooth/adam_general/core/creator/__init__.py | 12 | Creator init | - |
| smooth/adam_general/core/creator/architector.py | 515 | Architecture | architector, adam_profile_creator |
| smooth/adam_general/core/creator/creator.py | 703 | Matrix creation | creator |
| smooth/adam_general/core/creator/filler.py | 455 | Fill matrices | filler |
| smooth/adam_general/core/creator/initialiser.py | 1631 | Initial params | initialiser |
| smooth/adam_general/core/creator/initialization.py | 748 | State init | _initialize_states |
| smooth/adam_general/core/estimator/__init__.py | 7 | Estimator init | - |
| smooth/adam_general/core/estimator/estimator.py | 780 | Estimation | estimator |
| smooth/adam_general/core/estimator/selector.py | 1244 | Model selection | selector |
| smooth/adam_general/core/estimator/optimization.py | 455 | Optimization | _run_optimization |
| smooth/adam_general/core/estimator/two_stage.py | 329 | Two-stage init | _run_two_stage_estimator |
| smooth/adam_general/core/estimator/initial_values.py | 179 | Initial values | _process_initial_values |
| smooth/adam_general/core/forecaster/__init__.py | 4 | Forecaster init | - |
| smooth/adam_general/core/forecaster/forecaster.py | 881 | Forecast logic | forecaster |
| smooth/adam_general/core/forecaster/preparator.py | 1139 | Prepare for forecast | preparator |
| smooth/adam_general/core/forecaster/intervals.py | 720 | Prediction intervals | generate_prediction_interval |
| smooth/adam_general/core/forecaster/_helpers.py | 117 | Helpers | _prepare_lookup_table |
| smooth/adam_general/core/utils/__init__.py | 8 | Utils init | - |
| smooth/adam_general/core/utils/cost_functions.py | 1169 | Cost functions | CF, log_Lik_ADAM |
| smooth/adam_general/core/utils/ic.py | 160 | IC | ic_function |
| smooth/adam_general/core/utils/likelihood.py | 0 | (empty placeholder) | - |
| smooth/adam_general/core/utils/utils.py | 989 | General utils | msdecompose, lowess_r, scaler |
| smooth/adam_general/core/utils/var_covar.py | 671 | Variance | sigma, covar_anal, var_anal |
| smooth/adam_general/core/utils/polynomials.py | 93 | Polynomials | adam_polynomialiser |
| smooth/adam_general/core/utils/distributions.py | 305 | Distributions | generate_errors |
| smooth/adam_general/core/utils/n_param.py | 404 | N param | create_n_param |
| smooth/adam_general/core/utils/printing.py | 828 | Printing | model_summary |
| smooth/lowess.py | 132 | LOWESS | lowess |

## Entry Points

- **ADAM**: `smooth.adam_general.core.adam.ADAM`
- **ES**: `smooth.adam_general.core.es.ES`
- **msdecompose**: `smooth.adam_general.core.utils.utils.msdecompose`
- **lowess**: `smooth.lowess.lowess`
