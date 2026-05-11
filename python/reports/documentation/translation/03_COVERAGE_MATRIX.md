# Translation Coverage Matrix

What is translated vs not. Based on exports and test files.

## Full Translation

| Feature | R | Python |
|---------|---|--------|
| ETS (ANN, AAN, AAA, etc.) | es(), adam() | ES, ADAM |
| model selection (ZZZ, XXX, YYY) | adam(), es() | ADAM, ES |
| lags | lags | lags |
| persistence, phi | persistence, phi | persistence, phi |
| initial (backcasting, optimal, two-stage, complete) | adam | ADAM |
| distribution (dnorm, dgamma, etc.) | adam | ADAM |
| loss (likelihood, MSE, MAE, etc.) | adam | ADAM |
| xreg | xreg | X |
| msdecompose | msdecompose | msdecompose |
| lowess | stats::lowess | lowess |
| Prediction intervals (parametric) | forecast(interval=) | predict_intervals |
| Two-stage initialization | adam(initial="two-stage") | ADAM(initial="two-stage") |

## Partial Translation

| Feature | R | Python |
|---------|---|--------|
| ARIMA | adam(orders=), ssarima | ADAM(ar_order, i_order, ma_order) - in progress |
| Occurrence | adam(occurrence=), oes | occurrence in checker/forecaster; no standalone oes |
| Simulation intervals | forecast(interval="simulated") | generate_simulation_interval |
| Multi-seasonal | lags=c(1,12,4) | lags=[1,12,4] (supported) |

## Not Translated

| Feature | R | Python |
|---------|---|--------|
| gum | gum() | - |
| ces | ces() | - |
| sma | sma() | - |
| oes (standalone) | oes() | - |
| oesg | oesg() | - |
| sparma | sparma() | - |
| cma | cma() | - |
| smoothCombine | smoothCombine() | - |
| reapply | reapply() | - |
| reforecast | reforecast() | - |
| sim.es, sim.ces, etc. | sim.* | - |
| auto.gum, auto.ces, etc. | auto.* (except auto.adam) | - |
| Scale model (sm.adam) | sm.adam, implant | - |
| rmultistep | rmultistep() | - |
