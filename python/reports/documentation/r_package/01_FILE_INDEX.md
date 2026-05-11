# R Package File Index

Every R file in `R/` with path, line count, purpose, and key functions.

| File | Path | Lines | Main Purpose | Key Functions |
|------|------|-------|--------------|---------------|
| adam.R | R/adam.R | 9371 | Main ADAM model; architector/creator/filler/initialiser as local | adam, adamProfileCreator, forecast.adam, predict.adam |
| adamGeneral.R | R/adamGeneral.R | 3382 | Parameter validation | parametersChecker |
| adam-ces.R | R/adam-ces.R | 1204 | CES wrapper | ces, creator, filler, initialiser |
| adam-es.R | R/adam-es.R | 413 | ETS/ES wrapper | es |
| adam-gum.R | R/adam-gum.R | 1006 | GUM wrapper | gum, filler |
| adam-msarima.R | R/adam-msarima.R | 321 | Multi-seasonal ARIMA | msarima |
| adam-sma.R | R/adam-sma.R | 341 | Simple moving average | sma |
| adam-ssarima.R | R/adam-ssarima.R | 1341 | State-space ARIMA | ssarima, filler, initialiser |
| arimaCompact.R | R/arimaCompact.R | 89 | ARIMA compact representation | (utilities) |
| autoadam.R | R/autoadam.R | 982 | Auto ADAM selection | auto.adam |
| autoces.R | R/autoces.R | 189 | Auto CES | auto.ces |
| autogum.R | R/autogum.R | 253 | Auto GUM | auto.gum |
| automsarima.R | R/automsarima.R | 752 | Auto MSARIMA | auto.msarima |
| autossarima.R | R/autossarima.R | 693 | Auto SSARIMA | auto.ssarima |
| cma.R | R/cma.R | 184 | Centered moving average | cma |
| depricator.R | R/depricator.R | 9 | Deprecation helper | depricator |
| globals.R | R/globals.R | 169 | Global variables | - |
| helper.R | R/helper.R | 321 | Helper functions | calculateBackcastingDF, dfDiscounter, componentsDefiner, etsChecker |
| isFunctions.R | R/isFunctions.R | 107 | Type checks | is.smooth, is.adam, is.oes, etc. |
| iss.R | R/iss.R | 71 | Intermittent setup | intermittentParametersSetter |
| methods.R | R/methods.R | 2532 | S3 methods (plot, coef, sigma, etc.) | orders, lags, plot.smooth, coef.smooth |
| msdecompose.R | R/msdecompose.R | 486 | Multiple seasonal decomposition | msdecompose |
| oes.R | R/oes.R | 1291 | Occurrence ETS | oes |
| oesg.R | R/oesg.R | 1092 | Occurrence ETS general | oesg |
| reapply.R | R/reapply.R | 1401 | Reapply/reforecast | reapply, reforecast |
| rmultistep.R | R/rmultistep.R | 123 | Multi-step forecasting | rmultistep |
| RcppExports.R | R/RcppExports.R | 31 | Rcpp wrappers | forecasterwrap, matrixPowerWrap, occurenceFitterWrap |
| simces.R | R/simces.R | 471 | Simulate CES | sim.ces |
| simes.R | R/simes.R | 615 | Simulate ETS | sim.es |
| simgum.R | R/simgum.R | 436 | Simulate GUM | sim.gum |
| simoes.R | R/simoes.R | 152 | Simulate OES | sim.oes |
| simsma.R | R/simsma.R | 91 | Simulate SMA | sim.sma |
| simssarima.R | R/simssarima.R | 752 | Simulate SSARIMA | sim.ssarima |
| sm.R | R/sm.R | 421 | Scale model for ADAM | sm.adam, implant.adam |
| smooth-package.R | R/smooth-package.R | 86 | Package setup | - |
| smoothCombine.R | R/smoothCombine.R | 348 | Combine forecasts | smoothCombine |
| sowhat.R | R/sowhat.R | 38 | Release checks | sowhat |
| sparma.R | R/sparma.R | 691 | Seasonal PARMA | sparma |
| ssfunctions.R | R/ssfunctions.R | 2816 | Core SS functions | ssInput, ssIntervals, ssForecaster, ssXreg, likelihoodFunction, ICFunction |
| variance-covariance.R | R/variance-covariance.R | 136 | Variance-covariance | covarAnal, adamVarAnal |
| zzz.R | R/zzz.R | 32 | Package load/unload | .onAttach, .onLoad, .onUnload |

## Entry Points (User-Facing)

- **adam**: `adam.R` line 326
- **es**: `adam-es.R` line 224 (wraps adam)
- **gum**: `adam-gum.R` line 98
- **ces**: `adam-ces.R` line 91
- **ssarima**: `adam-ssarima.R` line 110
- **msarima**: `adam-msarima.R` line 192
- **sma**: `adam-sma.R` line 101
- **oes**: `oes.R` line 107
- **msdecompose**: `msdecompose.R` line 54
- **auto.adam**: `autoadam.R` line 19
