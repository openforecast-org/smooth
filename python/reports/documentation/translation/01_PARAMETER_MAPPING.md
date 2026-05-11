# R to Python Parameter Mapping

## Main Model Parameters

| R (adam/es) | Python (ADAM/ES) |
|-------------|------------------|
| model | model |
| lags | lags |
| orders (list: ar, i, ma) | ar_order, i_order, ma_order |
| persistence | persistence (dict or None) |
| phi | phi |
| initial | initial |
| initialSeason | initial_season |
| distribution | distribution |
| loss | loss |
| ic | ic |
| bounds | bounds |
| h | h (in predict) |
| holdout | holdout |
| xreg | X (in fit) |
| regressors | regressors |
| initialX | initial_xreg |

## Fitted Attributes (R → Python)

| R (model$...) | Python (model._) |
|---------------|------------------|
| model$persistence | persistence_level_, persistence_trend_, persistence_seasonal_ |
| model$phi | phi_ |
| model$initial | initial_states_ |
| model$scale | scale_ |
| model$logLik | (log_likelihood in results) |
| model$states | (in matrices_dict) |
| model$fitted | y_fitted |
| model$residuals | errors |
| model$orders | ar_order, i_order, ma_order (from model_type_dict) |

## Initial Options

| R | Python |
|---|--------|
| "backcasting" | "backcasting" |
| "optimal" | "optimal" |
| "two-stage" | "two-stage" |
| "complete" | "complete" |
| "provided" (with initial values) | dict or list of values |

## Distribution

| R | Python |
|---|--------|
| "default" | "default" |
| "dnorm" | "dnorm" |
| "dlaplace" | "dlaplace" |
| "dgamma" | "dgamma" |
| etc. | Same string names |
