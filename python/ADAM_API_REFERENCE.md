# ADAM Class API Reference

Complete reference for `smooth.adam_general.core.adam.ADAM`.

---

## Constructor Parameters (`__init__`)

### Model Specification

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str \| list[str]` | `"ZXZ"` | ETS model string (e.g. `"ANN"`, `"ZZZ"`, `"CCC"`) or list of models |
| `lags` | `list[int] \| None` | `None` | Seasonal period(s), e.g. `[1, 12]` for monthly with annual seasonality |

### ARIMA Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ar_order` | `int \| list[int]` | `0` | Autoregressive order(s) |
| `i_order` | `int \| list[int]` | `0` | Integration (differencing) order(s) |
| `ma_order` | `int \| list[int]` | `0` | Moving average order(s) |
| `arima_select` | `bool` | `False` | Automatic ARIMA order selection |

### Estimation Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `constant` | `bool` | `False` | Include constant/intercept term |
| `regressors` | `"use" \| "select" \| "adapt"` | `"use"` | How to handle external regressors |
| `distribution` | `str \| None` | `None` | Error distribution (`"dnorm"`, `"dgamma"`, `"dlaplace"`, `"dlnorm"`, `"dinvgauss"`, `"ds"`, `"dgnorm"`). Auto-selected if `None` |
| `loss` | `str` | `"likelihood"` | Loss function (`"likelihood"`, `"MSE"`, `"MAE"`, `"HAM"`, `"MSEh"`, `"TMSE"`, `"GTMSE"`, `"GPL"`, `"MSCE"`, `"LASSO"`, `"RIDGE"`, etc.) |
| `loss_horizon` | `int \| None` | `None` | Steps for multi-step loss functions |
| `ic` | `"AIC" \| "AICc" \| "BIC" \| "BICc"` | `"AICc"` | Information criterion for model selection |
| `bounds` | `"usual" \| "admissible" \| "none"` | `"usual"` | Parameter bounds during optimization |

### Outlier Detection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `outliers` | `"ignore" \| "detect" \| "use"` | `"ignore"` | Outlier handling method |
| `outliers_level` | `float` | `0.99` | Confidence level for outlier detection |

### Fixed Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `persistence` | `dict \| None` | `None` | Fixed smoothing params, e.g. `{"alpha": 0.3, "beta": 0.1}` |
| `phi` | `float \| None` | `None` | Fixed damping parameter (0-1) |
| `initial` | `str \| dict \| None` | `"backcasting"` | Initialization method (`"optimal"`, `"backcasting"`, `"two-stage"`, `"complete"`, `"provided"`) or dict of values |
| `n_iterations` | `int \| None` | `None` | Backcasting iterations (default: 2 for backcasting, 1 otherwise) |
| `arma` | `dict \| None` | `None` | Fixed ARMA parameters |

### Occurrence / Intermittent Demand

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `occurrence` | `str` | `"none"` | Occurrence model type (`"none"`, `"auto"`, `"fixed"`, `"general"`, `"odds-ratio"`, `"inverse-odds-ratio"`, `"direct"`) |

### Optimizer Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nlopt_initial` | `dict \| None` | `None` | Initial values for NLopt optimizer |
| `nlopt_upper` | `dict \| None` | `None` | Upper bounds for optimizer |
| `nlopt_lower` | `dict \| None` | `None` | Lower bounds for optimizer |
| `nlopt_kwargs` | `dict \| None` | `None` | Extra NLopt options: `print_level`, `xtol_rel`, `xtol_abs`, `ftol_rel`, `ftol_abs`, `algorithm` |

### Other

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `verbose` | `int` | `0` | Verbosity level (0 = silent) |
| `h` | `int \| None` | `None` | Forecast horizon (can also be set in `predict()`) |
| `holdout` | `bool` | `False` | Withhold last `h` observations for validation |
| `fast` | `bool` | `False` | Use faster, possibly less accurate estimation |
| `lambda_param` | `float \| None` | `None` | Box-Cox transformation or regularization lambda |
| `frequency` | `str \| None` | `None` | Time series frequency (e.g. `"D"`, `"M"`); auto-detected from pandas DatetimeIndex |
| `profiles_recent_provided` | `bool` | `False` | Whether recent seasonal profiles are provided |
| `profiles_recent_table` | `Any \| None` | `None` | Table of recent profile data |
| `reg_lambda` | `float \| None` | `None` | Regularization parameter for LASSO/RIDGE losses |
| `gnorm_shape` | `float \| None` | `None` | Shape parameter for generalized normal distribution |
| `smoother` | `"lowess" \| "ma" \| "global"` | `"lowess"` | Smoother for decomposition in initial state estimation |

---

## Public Methods

### `fit(y, X=None)`

Fit the ADAM model to time series data. Returns `self`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `y` | `NDArray \| pd.Series` | Time series data, shape `(T,)` |
| `X` | `NDArray \| None` | External regressors, shape `(T, n_features)` |

After fitting, all properties and internal attributes are populated. Init parameters are consolidated into `self._config` and removed from the instance.

### `predict(h, X=None, interval="none", level=0.95, side="both", cumulative=False, nsim=10000, occurrence=None, scenarios=False)`

Generate forecasts. Returns `ForecastResult`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `h` | `int` | — | Forecast horizon |
| `X` | `NDArray \| None` | `None` | Future regressors, shape `(h, n_features)` |
| `interval` | `str` | `"none"` | Interval type: `"none"`, `"prediction"`, `"simulated"`, `"approximate"` |
| `level` | `float \| list[float]` | `0.95` | Confidence level(s) for intervals |
| `side` | `str` | `"both"` | `"both"`, `"upper"`, or `"lower"` |
| `cumulative` | `bool` | `False` | Cumulative forecasts over horizon |
| `nsim` | `int` | `10000` | Simulations for simulation-based intervals |
| `occurrence` | `NDArray \| None` | `None` | External occurrence probabilities |
| `scenarios` | `bool` | `False` | Store raw simulation matrix |

### `predict_intervals(h, X=None, levels=[0.8, 0.95], side="both", nsim=10000)`

Convenience wrapper around `predict()` with `interval="prediction"`. Returns `ForecastResult`.

### `summary(digits=4)`

Return formatted model summary string.

### `select_best_model()`

Select best model from selection results based on IC. Updates `_model_type`, `_phi_internal`, `_adam_estimated`, and `_adam_cpp` with best model's values.

---

## Properties (read-only, available after `fit()`)

### State-Space Matrices

| Property | Return Type | Description |
|----------|-------------|-------------|
| `states` | `NDArray` | State matrix `mat_vt`, shape `(n_states, T+1)` |
| `transition` | `NDArray` | Transition matrix `F`, shape `(n_states, n_states)` |
| `measurement` | `NDArray` | Measurement matrix `W`, shape `(T, n_states)` |
| `persistence_vector` | `dict` | Smoothing params dict (keys: `persistence_level`, `persistence_trend`, `persistence_seasonal`) |
| `phi_` | `float \| None` | Damping parameter (None if no damped trend) |
| `initial_value` | `dict` | Initial state values (keys: `level`, `trend`, `seasonal`) |
| `initial_type` | `str` | Initialization method used (`"optimal"`, `"backcasting"`, etc.) |

### Loss and Fit Information

| Property | Return Type | Description |
|----------|-------------|-------------|
| `loss_value` | `float` | Optimized cost function value |
| `loss_` | `str` | Loss function name used (trailing `_` to avoid name conflict) |
| `distribution_` | `str` | Error distribution used (trailing `_` to avoid name conflict) |
| `sigma` | `float` | Scale/standard error estimate |
| `scale` | `float` | Alias for `sigma` |
| `loglik` | `float` | Log-likelihood value |
| `time_elapsed` | `float` | Fitting time in seconds |

### Data Access

| Property | Return Type | Description |
|----------|-------------|-------------|
| `fitted` | `NDArray` | In-sample fitted values (IC-weighted for combined models) |
| `residuals` | `NDArray` | Model residuals |
| `actuals` | `NDArray` | Original in-sample observations |
| `data` | `NDArray` | Alias for `actuals` |
| `holdout_data` | `NDArray \| None` | Holdout observations (None if `holdout=False`) |
| `nobs` | `int` | Number of in-sample observations |

### Coefficients and Parameters

| Property | Return Type | Description |
|----------|-------------|-------------|
| `coef` | `NDArray` | Estimated parameter vector B |
| `b_value` | `NDArray` | Alias for `coef` |
| `nparam` | `int` | Number of estimated parameters |
| `n_param` | `Any` | Parameter count information object |
| `constant_value` | `float \| None` | Constant/intercept term value |
| `profile` | `Any \| None` | Seasonal profile information |

### Information Criteria

| Property | Return Type | Description |
|----------|-------------|-------------|
| `aic` | `float` | Akaike Information Criterion |
| `aicc` | `float` | Corrected AIC |
| `bic` | `float` | Bayesian Information Criterion |
| `bicc` | `float` | Corrected BIC |

### Model Specification

| Property | Return Type | Description |
|----------|-------------|-------------|
| `model_name` | `str` | Full model name, e.g. `"ETS(AAN)"`, `"ETS(AAA)+ARIMA(1,1,1)"` |
| `model_type` | `str` | ETS code only, e.g. `"AAN"` |
| `error_type` | `str` | `"A"` (additive) or `"M"` (multiplicative) |
| `orders` | `dict` | ARIMA orders: `{"ar": [...], "i": [...], "ma": [...]}` |
| `lags_used` | `list[int]` | Lag values used in the model |

### Combined Model Properties

| Property | Return Type | Description |
|----------|-------------|-------------|
| `is_combined` | `bool` | True if model uses IC-weighted combination |
| `ic_weights` | `dict[str, float]` | Model name -> IC weight mapping (raises if not combined) |
| `models` | `list[dict]` | List of individual model dicts with `name`, `weight`, `prepared`, etc. (raises if not combined) |

---

## Dunder Methods

| Method | Description |
|--------|-------------|
| `__str__()` | Formatted model summary via `model_summary()`. Shows `"ADAM(model=...) - not fitted"` before fitting |
| `__repr__()` | Brief representation, e.g. `"ADAM(ETS(AAN), fitted=True)"` or `"ADAM(model=ZXZ, fitted=False)"` |

---

## Private Methods

### Core Pipeline

| Method | Description |
|--------|-------------|
| `_check_parameters(ts)` | Run `parameters_checker()` and populate all internal dicts |
| `_execute_estimation(estimation=True)` | Run estimator + architector + creator + IC calculation |
| `_execute_selection()` | Run `selector()` for model selection, then `select_best_model()` |
| `_execute_combination()` | Combine models using IC weights; sets `_is_combined=True` |
| `_prepare_results()` | Format data, select distribution, compute fitted values |

### Estimation Helpers

| Method | Description |
|--------|-------------|
| `_handle_lasso_ridge_special_case()` | Handle LASSO/RIDGE with `lambda=1` (disable estimation, use MSE for initials) |
| `_preset_arima_parameters()` | Set ARIMA params when estimation is disabled |
| `_update_parameters_number(n_param_estimated)` | Update `NParam` object and legacy `parameters_number` format |
| `_set_fitted_attributes()` | Set `persistence_level_`, `persistence_trend_`, etc. and build `self.model` string |

### Result Preparation

| Method | Description |
|--------|-------------|
| `_format_time_series_data()` | Convert arrays to `pd.Series` with DatetimeIndex if available |
| `_select_distribution()` | Map loss function to default distribution when `distribution="default"` |
| `_compute_fitted_values()` | Call `preparator()` to get fitted values, residuals, scale |

### Prediction Pipeline

| Method | Description |
|--------|-------------|
| `_check_is_fitted()` | Raise `ValueError` if `_prepared` is not set |
| `_validate_prediction_inputs()` | Verify model is fitted and has required components |
| `_prepare_prediction_data()` | Call `preparator()` to set up matrices for forecasting |
| `_execute_prediction(interval, level, side)` | Run `forecaster()` or delegate to `_execute_prediction_combined()` |
| `_execute_prediction_combined(interval, level, side)` | Run `forecaster_combined()` for IC-weighted combined forecasts |
| `_format_prediction_results()` | Return stored `_forecast_results` (not called by current public API) |

### Selection Helpers (legacy/placeholder)

| Method | Description |
|--------|-------------|
| `_update_model_from_selection(index, result)` | Update all dicts from a selection result |
| `_create_matrices_for_selected_model(index)` | Call `creator()` for a selected model |
| `_update_parameters_for_selected_model(index, result)` | Update parameter counts for a selected model |

---

## Internal Attributes (set during `fit()`)

### Configuration

| Attribute | Type | Description |
|-----------|------|-------------|
| `_config` | `dict` | Consolidated init params after `fit()` (keys: `lags`, `ar_order`, `i_order`, `ma_order`, `distribution`, `loss`, `ic`, `bounds`, `holdout`, `fast`, etc.) |
| `_start_time` | `float` | `time.time()` at init |
| `time_elapsed_` | `float` | Total fit duration in seconds |

### Parameter Dicts (from `parameters_checker`)

| Attribute | Type | Description |
|-----------|------|-------------|
| `_general` | `dict` | General settings (h, loss, distribution, ic, holdout, n_param, etc.) |
| `_observations` | `dict` | Data arrays (`y_in_sample`, `y_holdout`, `obs_in_sample`, frequency info) |
| `_persistence` | `dict` | Persistence/smoothing params and estimation flags |
| `_initials` | `dict` | Initial state values and type |
| `_arima` | `dict` | ARIMA orders, parameters, estimation flags |
| `_constant` | `dict` | Constant term settings |
| `_model_type` | `dict` | Model specification (`error_type`, `trend_type`, `season_type`, `damped`, `model_do`, `ets_model`, `arima_model`) |
| `_components` | `dict` | Component structure info |
| `_lags_model` | `dict` | Lag configuration (`lags`, `lags_model_max`, etc.) |
| `_occurrence` | `dict` | Occurrence model settings |
| `_phi_internal` | `dict` | Phi (damping) value and estimation flag |
| `_explanatory` | `dict` | Exogenous regressor settings (`xreg_model`, etc.) |
| `_params_info` | `dict` | Parameter information (bounds, `parameters_number`) |

### Estimation Results

| Attribute | Type | Description |
|-----------|------|-------------|
| `_adam_estimated` | `dict` | Estimation output (`B`, `CF_value`, `log_lik_adam_value`, `n_param_estimated`, `adam_cpp`) |
| `_adam_cpp` | `object` | C++ adamCore object for fitting/forecasting |
| `_adam_created` | `dict` | Creator output (state-space matrices before filling) |
| `_adam_selected` | `dict` | Selector output (`results` list, `ic_selection` dict) |
| `_ic_selection` | `dict` | Model name -> IC value mapping |
| `_best_model` | `str` | Best model name from selection |
| `_profile` | `dict` | Seasonal profile data |
| `_prepared` | `dict` | Preparator output (`mat_vt`, `mat_f`, `mat_wt`, `y_fitted`, `residuals`, `scale`, `persistence`, `phi`, `initial_value`, `constant_value`, `profiles_recent_table`) |

### Fitted Parameter Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `persistence_level_` | `float` | Level smoothing parameter (alpha) |
| `persistence_trend_` | `float` | Trend smoothing parameter (beta) |
| `persistence_seasonal_` | `float \| list` | Seasonal smoothing parameter(s) (gamma) |
| `persistence_xreg_` | `float` | Regressor persistence parameter |
| `model` | `str` | Updated after fit to full name, e.g. `"ETS(AAN)"` |

### Combined Model Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `_is_combined` | `bool` | True when model uses IC-weighted combination |
| `_ic_weights` | `dict[str, float]` | Model name -> IC weight |
| `_prepared_models` | `list[dict]` | Per-model data (name, weight, result, prepared, model_type_dict, components_dict, lags_dict, observations_dict, profile_dict, phi_dict, adam_created, explanatory_dict, constants_dict) |
| `_combined_fitted` | `NDArray` | IC-weighted combined fitted values |
| `_combined_residuals` | `NDArray` | `y - combined_fitted` |
| `_n_param_combined` | `float` | Weighted average of estimated params |
| `_original_model_spec` | `str` | Original model string (e.g. `"CCC"`) before selection |

### Prediction Results

| Attribute | Type | Description |
|-----------|------|-------------|
| `_forecast_results` | `ForecastResult` | Last prediction output (`.mean`, `.lower`, `.upper`, `.level`, `.side`, `.interval`) |
| `_y_in_sample` | `pd.Series` | Formatted in-sample data |
| `_y_holdout` | `pd.Series` | Formatted holdout data |
| `_n_param_estimated` | `int` | Number of estimated parameters |
| `_n_param` | `NParam` | Parameter count object |
