# R Package Overview

## Architecture

The smooth R package implements forecasting using state-space models. The centerpiece is **ADAM** (Augmented Dynamic Adaptive Model), combining ETS, ARIMA, and regression in a Single Source of Error framework.

## Package Layout

```
R/
├── adam.R              # Main adam() + local architector/creator/filler/initialiser
├── adamGeneral.R       # parametersChecker (called by adam)
├── adam-es.R           # es() - ETS wrapper
├── adam-gum.R          # gum() - GUM wrapper
├── adam-ces.R          # ces() - CES wrapper
├── adam-ssarima.R      # ssarima() - State-space ARIMA
├── adam-msarima.R      # msarima() - Multi-seasonal ARIMA
├── adam-sma.R          # sma() - Simple moving average
├── ssfunctions.R       # ssInput, ssForecaster, ssIntervals, ssXreg, likelihoodFunction, ICFunction
├── helper.R            # calculateBackcastingDF, dfDiscounter, componentsDefiner, checkers
├── methods.R           # S3 methods (plot, coef, sigma, forecast, etc.)
├── oes.R / oesg.R      # Occurrence models
├── msdecompose.R       # Multiple seasonal decomposition
└── auto*.R, sim*.R, ...# Auto selection, simulation
```

## Entry Points and Call Flow

### adam() (R/adam.R line 326)

Main entry. Internally:

1. Calls `parametersChecker()` from adamGeneral.R
2. Defines local `architector`, `creator`, `filler`, `initialiser`
3. For estimation: architector -> creator -> initialiser -> optimization (uses filler + C++ adamCore)
4. For forecasting: filler -> forecasterwrap (Rcpp)

### es() (R/adam-es.R line 224)

Wraps adam() with ETS defaults. Calls adam() via `adam(y, model=..., orders=list(ar=c(0),i=c(0),ma=c(0)), ...)`.

### gum(), ces(), ssarima(), msarima(), sma()

Each has its own wrapper that either calls adam() with specific orders or uses ssfunctions.R (ssInput, ssForecaster) for non-ADAM models.

### ssfunctions.R

- **ssInput**: Shared input checker for es, gum, ces, ssarima, smoothC
- **ssIntervals**: Prediction interval calculation
- **ssForecaster**: Calls `forecasterwrap` (Rcpp) for point forecasts
- **ssXreg**: Exogenous variable handling
- **likelihoodFunction**, **ICFunction**: Likelihood and IC for optimization

## C++ Integration (R)

- **RcppExports.R**: forecasterwrap, matrixPowerWrap, occurenceFitterWrap, occurrenceOptimizerWrap, etc.
- **adamGeneral.cpp**: Exposes adamCore class (fit, forecast, simulate)
- **ssGeneral.cpp**, **ssOccurrence.cpp**: State-space and occurrence ops
- **eigenCalc.cpp**, **matrixPowerWrap.cpp**: Eigenvalues, matrix power

## NAMESPACE Exports

Key exports: adam, es, gum, ces, ssarima, msarima, sma, oes, oesg, msdecompose, auto.adam, auto.ces, auto.gum, auto.ssarima, auto.msarima, smoothCombine, sim.es, sim.ces, sim.gum, sim.ssarima, sim.sma, sim.oes, reapply, reforecast, cma, sparma, pls, sowhat, lags, orders, modelName, modelType, plus S3 methods.
