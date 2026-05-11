# R to Python Translation Index

Master lookup: R function → Python equivalent. Verified from codebase.

## Core Pipeline

| R | Python |
|---|--------|
| parametersChecker (adamGeneral.R) | parameters_checker (checker/parameters_checker.py) |
| architector (adam.R local) | architector (creator/architector.py) |
| creator (adam.R local) | creator (creator/creator.py) |
| filler (adam.R local) | filler (creator/filler.py) |
| initialiser (adam.R local) | initialiser (creator/initialiser.py) |
| adam C++ fit | adam_fitter → _adamCore.fit |
| forecasterwrap | adam_forecaster → _adamCore.forecast |
| adamProfileCreator | adam_profile_creator (architector.py) |

## User Entry Points

| R | Python |
|---|--------|
| adam() | ADAM class |
| es() | ES class |
| msdecompose() | msdecompose (utils/utils.py) |
| (no direct lowess export) | lowess (lowess.py) |

## Utilities

| R | Python |
|---|--------|
| likelihoodFunction | CF, log_Lik_ADAM (cost_functions.py) |
| ICFunction | ic_function (ic.py) |
| covarAnal | covar_anal (var_covar.py) |
| adamVarAnal | var_anal (var_covar.py) |
| matrixPowerWrap | matrix_power_wrap (var_covar.py) |
| adam_polynomialiser (C++) | adam_polynomialiser (polynomials.py) |

## Not Translated (R Only)

- gum, ces, ssarima (as separate entry), msarima, sma
- oes, oesg as standalone
- reapply, reforecast
- sim.*, auto.gum, auto.ces, auto.ssarima, auto.msarima
- sparma, cma, smoothCombine
- ssInput, ssForecaster, ssIntervals, ssXreg (logic distributed in Python)

## Partial / In Progress

- ARIMA: Python has ar_order, i_order, ma_order in ADAM; ssarima-specific logic not fully mirrored
- Occurrence: Python has _check_occurrence, occurrence in forecaster; full oes API not exposed
