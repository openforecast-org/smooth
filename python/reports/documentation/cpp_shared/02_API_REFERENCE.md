# adamCore API Reference

From src/headers/adamCore.h and bindings in adamGeneral.cpp / adamPython.cpp.

## Constructor

```
adamCore(lags, E, T, S, nNonSeasonal, nSeasonal, nETS, nArima, nXreg, nComponents, constant, adamETS)
```

- E, T, S: Error, Trend, Seasonal type chars ('A', 'M', 'N', etc.)
- nNonSeasonal, nSeasonal: ETS component counts
- nArima, nXreg: ARIMA and xreg counts
- adamETS: true for ETS models

## Methods

### polynomialise(B, arOrders, iOrders, maOrders, arEstimate, maEstimate, armaParameters, lagsARIMA)

Returns PolyResult: arPolynomial, iPolynomial, ariPolynomial, maPolynomial. Used for ARIMA polynomial setup.

### fit(matrixVt, matrixWt, matrixF, vectorG, indexLookupTable, profilesRecent, vectorYt, vectorOt, backcast, nIterations, refineHead)

Returns FitResult: states, fitted, errors, profile. Core fitting routine.

### forecast(matrixWt, matrixF, indexLookupTable, profilesRecent, horizon)

Returns ForecastResult: forecast (vector). Point forecast.

### ferrors(matrixVt, matrixWt, matrixF, indexLookupTable, profilesRecent, horizon, vectorYt)

Returns ErrorResult: errors. Forecast errors.

### simulate(matrixErrors, matrixOt, arrayVt, matrixWt, arrayF, matrixG, indexLookupTable, profilesRecent, E)

Returns SimulateResult: states, data. Simulation.

### reapply(matrixYt, matrixOt, arrayVt, arrayWt, arrayF, matrixG, indexLookupTable, arrayProfilesRecent, backcast, refineHead)

Returns ReapplyResult. Re-fit with new data.

### reforecast(arrayErrors, arrayOt, arrayWt, arrayF, matrixG, indexLookupTable, arrayProfileRecent, E)

Returns ReforecastResult. Re-forecast from errors.

## Python Wrapper (_adam_general.py)

- adam_fitter: Creates adamCore, calls fit, returns dict
- adam_forecaster: Creates adamCore, calls forecast, returns array
- adam_simulator: Creates adamCore, calls simulate, returns dict
