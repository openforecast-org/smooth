# R to Python Data Structure Mapping

## State-Space Matrices

| R | Python |
|---|--------|
| matVt | mat_vt (matrices_dict) |
| matF | mat_f |
| matw / matWt | mat_wt |
| vecg | vec_g |
| matxt | mat_xt (xreg) |
| matat | R-only (oes, ssXreg) |
| matFX | R-only (oes, ssXreg) |

R uses camelCase; Python uses snake_case. Same mathematical meaning. Note: matat, matFX are R-only; Python has mat_xt but full xreg/occurrence state matrices not yet mirrored.

## Checker / Parameters Output

| R (list) | Python (dict) |
|----------|---------------|
| ParentEnvironment / env | Not used; Python passes explicitly |
| modelType / Etype, Ttype, Stype | model_type_dict |
| lagsModel, lagsModelAll | lags_dict |
| yInSample, yHoldout | observations_dict |
| otLogical | observations_dict or occurrence |
| initialType, initialValue | initials_checked |

## Estimator Output

| R | Python |
|---|--------|
| B (vector) | B (array) |
| matVt (filled) | matrices_dict["mat_vt"] |
| yFitted | y_fitted |
| errors | errors |
| scale | scale |
| logLik | log_likelihood |

## Forecast Output

| R | Python |
|---|--------|
| yForecast (ts) | forecasts (Series or array) |
| lower, upper (intervals) | lower, upper in predict_intervals |
| model$forecast | model.predict(h) |

## Profile / Lookup

| R | Python |
|---|--------|
| profilesRecentTable | profiles_recent_table |
| indexLookupTable | index_lookup_table |
