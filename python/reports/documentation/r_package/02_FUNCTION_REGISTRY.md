# R Package Function Registry

Every function defined in R/, with file and approximate line. Source: `grep -E "^[a-zA-Z0-9_.]+ *<- *function" R/*.R`.

| Function | File | Line | Role |
|----------|------|------|------|
| calculateBackcastingDF | helper.R | 6 | Backcasting degrees of freedom |
| dfDiscounter | helper.R | 64 | Discount factor |
| dfDiscounterFit | helper.R | 158 | Discount factor for fit |
| componentsDefiner | helper.R | 196 | Define ETS components |
| etsChecker | helper.R | 260 | ETS model checker |
| arimaChecker | helper.R | 264 | ARIMA checker |
| gumChecker | helper.R | 268 | GUM checker |
| ssarimaChecker | helper.R | 272 | SSARIMA checker |
| cesChecker | helper.R | 276 | CES checker |
| sparmaChecker | helper.R | 280 | SPARMA checker |
| smoothEigens | helper.R | 286 | Eigenvalue calc (R wrapper) |
| adam | adam.R | 326 | Main ADAM model entry |
| adamProfileCreator | adam.R | 4881 | Profile/lookup table creator |
| adamETSChecker | adam.R | 4982 | ETS checker for adam object |
| modelType.adam | adam.R | 4993 | Model type extractor |
| errorType.adam | adam.R | 5007 | Error type extractor |
| trendType | adam.R | 5019 | Trend type |
| seasonType | adam.R | 5023 | Season type |
| orders.adam | adam.R | 5029 | ARIMA orders |
| lags.adam | adam.R | 5035 | Lags |
| plot.adam | adam.R | 5041 | Plot method |
| print.adam | adam.R | 5853 | Print method |
| print.adamCombined | adam.R | 6066 | Print combined |
| eigenValues | adam.R | 6108 | Eigenvalues |
| eigenBounds | adam.R | 6117 | Eigenvalue bounds |
| arPolinomialsBounds | adam.R | 6146 | AR polynomial bounds |
| confint.adam | adam.R | 6184 | Confidence intervals |
| coef.adam | adam.R | 6382 | Coefficients |
| sigma.adam | adam.R | 6389 | Scale/sigma |
| summary.adam | adam.R | 6425 | Summary |
| as.data.frame.summary.adam | adam.R | 6499 | Coerce summary |
| summary.adamCombined | adam.R | 6504 | Summary combined |
| print.summary.adam | adam.R | 6509 | Print summary |
| xtable.adam | adam.R | 6600 | xtable |
| xtable.summary.adam | adam.R | 6609 | xtable summary |
| coefbootstrap.adam | adam.R | 6621 | Bootstrap coef |
| vcov.adam | adam.R | 6888 | Variance-covariance |
| actuals.adam | adam.R | 7011 | Actual values |
| nobs.adam | adam.R | 7026 | Number of observations |
| residuals.adam | adam.R | 7031 | Residuals |
| rstandard.adam | adam.R | 7054 | Standardized residuals |
| rstudent.adam | adam.R | 7103 | Studentized residuals |
| outlierdummy.adam | adam.R | 7190 | Outlier dummy |
| predict.adam | adam.R | 7236 | Predict |
| plot.adam.predict | adam.R | 7511 | Plot predict |
| forecast.adam | adam.R | 7610 | Forecast |
| forecast.adamCombined | adam.R | 8568 | Forecast combined |
| print.adam.forecast | adam.R | 8659 | Print forecast |
| plot.adam.forecast | adam.R | 8677 | Plot forecast |
| multicov.adam | adam.R | 8810 | Multi-step covariance |
| pointLik.adam | adam.R | 8997 | Point likelihood |
| simulate.adam | adam.R | 9115 | Simulate |
| print.adam.sim | adam.R | 9364 | Print simulation |
| ssarima | adam-ssarima.R | 110 | SSARIMA entry |
| gum | adam-gum.R | 98 | GUM entry |
| smoothEigensR | RcppExports.R | 4 | Rcpp eigenvalues |
| matrixPowerWrap | RcppExports.R | 8 | Rcpp matrix power |
| forecasterwrap | RcppExports.R | 12 | Rcpp forecaster |
| occurenceFitterWrap | RcppExports.R | 16 | Rcpp occurrence fitter |
| occurrenceOptimizerWrap | RcppExports.R | 20 | Rcpp occurrence optimizer |
| occurenceGeneralFitterWrap | RcppExports.R | 24 | Rcpp general occurrence |
| occurrenceGeneralOptimizerWrap | RcppExports.R | 28 | Rcpp general occurrence optimizer |
| ces | adam-ces.R | 91 | CES entry |
| auto.gum | autogum.R | 20 | Auto GUM |
| sm.adam | sm.R | 6 | Scale model for ADAM |
| extractScale.smooth | sm.R | 351 | Extract scale |
| extractSigma.smooth | sm.R | 377 | Extract sigma |
| implant.adam | sm.R | 408 | Implant scale model |
| auto.ces | autoces.R | 21 | Auto CES |
| auto.ssarima | autossarima.R | 77 | Auto SSARIMA |
| sparma | sparma.R | 62 | SPARMA entry |
| msdecompose | msdecompose.R | 54 | Decomposition entry |
| actuals.msdecompose | msdecompose.R | 282 | Actuals |
| errorType.msdecompose | msdecompose.R | 287 | Error type |
| fitted.msdecompose | msdecompose.R | 297 | Fitted |
| forecast.msdecompose | msdecompose.R | 308 | Forecast |
| is.msdecompose | msdecompose.R | 377 | Type check |
| is.msdecompose.forecast | msdecompose.R | 383 | Type check |
| lags.msdecompose | msdecompose.R | 388 | Lags |
| modelType.msdecompose | msdecompose.R | 393 | Model type |
| nobs.msdecompose | msdecompose.R | 398 | Nobs |
| nparam.msdecompose | msdecompose.R | 403 | N param |
| plot.msdecompose | msdecompose.R | 409 | Plot |
| print.msdecompose | msdecompose.R | 462 | Print |
| residuals.msdecompose | msdecompose.R | 469 | Residuals |
| sigma.msdecompose | msdecompose.R | 479 | Sigma |
| auto.msarima | automsarima.R | 22 | Auto MSARIMA |
| es | adam-es.R | 224 | ES/ETS entry |
| msarima | adam-msarima.R | 192 | MSARIMA entry |
| parametersChecker | adamGeneral.R | 1 | Main parameter checker |
| rmultistep | rmultistep.R | 28 | Multi-step entry |
| rmultistep.default | rmultistep.R | 33 | Default |
| rmultistep.adam | rmultistep.R | 40 | ADAM method |
| ssInput | ssfunctions.R | 2 | SS input checker |
| ssIntervals | ssfunctions.R | 1521 | Prediction intervals |
| ssForecaster | ssfunctions.R | 2102 | SS forecaster |
| ssXreg | ssfunctions.R | 2326 | Xreg handling |
| likelihoodFunction | ssfunctions.R | 2698 | Likelihood |
| ICFunction | ssfunctions.R | 2775 | Information criterion |
| sim.sma | simsma.R | 48 | Sim SMA |
| orders | methods.R | 60 | Generic orders |
| lags | methods.R | 65 | Generic lags |
| modelName | methods.R | 70 | Generic modelName |
| modelType | methods.R | 75 | Generic modelType |
| AICc.smooth | methods.R | 81 | AICc |
| BICc.smooth | methods.R | 102 | BICc |
| multicov | methods.R | 165 | multicov generic |
| multicov.default | methods.R | 169 | Default |
| multicov.smooth | methods.R | 178 | Smooth method |
| logLik.smooth | methods.R | 316 | Log-likelihood |
| logLik.smooth.sim | methods.R | 326 | Log-lik sim |
| nobs.smooth | methods.R | 333 | Nobs generic |
| nobs.smooth.sim | methods.R | 338 | Nobs sim |
| nparam.smooth | methods.R | 350 | N param |
| pls | methods.R | 402 | PLS generic |
| pls.default | methods.R | 407 | PLS default |
| pls.smooth | methods.R | 421 | PLS smooth |
| sigma.smooth | methods.R | 555 | Sigma generic |
| sigma.smooth.sim | methods.R | 565 | Sigma sim |
| pointLik.smooth | methods.R | 572 | Point lik |
| pointLik.oes | methods.R | 598 | Point lik oes |
| coef.smooth | methods.R | 614 | Coef generic |
| fitted.smooth | methods.R | 673 | Fitted |
| fitted.smooth.forecast | methods.R | 677 | Fitted forecast |
| forecast.oes | methods.R | 683 | Forecast oes |
| actuals.smooth | methods.R | 726 | Actuals |
| actuals.smooth.forecast | methods.R | 730 | Actuals forecast |
| lags.default | methods.R | 736 | Lags default |
| lags.ets | methods.R | 741 | Lags ets |
| lags.ar | methods.R | 751 | Lags ar |
| lags.Arima | methods.R | 756 | Lags Arima |
| lags.smooth | methods.R | 761 | Lags smooth |
| errorType.smooth | methods.R | 828 | Error type |
| modelLags | methods.R | 879 | Model lags |
| modelName.default | methods.R | 916 | modelName default |
| modelName.ar | methods.R | 921 | modelName ar |
| modelName.lm | methods.R | 926 | modelName lm |
| modelName.Arima | methods.R | 931 | modelName Arima |
| modelName.ets | methods.R | 943 | modelName ets |
| modelName.forecast | methods.R | 948 | modelName forecast |
| modelName.smooth | methods.R | 953 | modelName smooth |
| modelType.default | methods.R | 959 | modelType default |
| modelType.smooth | methods.R | 964 | modelType smooth |
| modelType.oesg | methods.R | 997 | modelType oesg |
| modelType.ets | methods.R | 1002 | modelType ets |
| orders.default | methods.R | 1008 | orders default |
| orders.smooth | methods.R | 1013 | orders smooth |
| orders.ar | methods.R | 1083 | orders ar |
| orders.Arima | methods.R | 1088 | orders Arima |
| plot.smooth | methods.R | 1188 | Plot smooth |
| plot.smoothC | methods.R | 1840 | Plot smoothC |
| plot.smooth.sim | methods.R | 1846 | Plot sim |
| plot.smooth.forecast | methods.R | 1880 | Plot forecast |
| plot.oes | methods.R | 1895 | Plot oes |
| plot.oes.sim | methods.R | 1901 | Plot oes sim |
| print.smooth.sim | methods.R | 1936 | Print sim |
| print.smooth.forecast | methods.R | 2050 | Print forecast |
| print.oes | methods.R | 2080 | Print oes |
| print.oes.sim | methods.R | 2133 | Print oes sim |
| residuals.smooth | methods.R | 2162 | Residuals |
| rstandard.smooth | methods.R | 2173 | Rstandard |
| rstudent.smooth | methods.R | 2196 | Rstudent |
| outlierdummy.smooth | methods.R | 2229 | Outlier dummy |
| simulate.smooth | methods.R | 2259 | Simulate |
| smoothType | methods.R | 2401 | Smooth type |
| summary.smooth | methods.R | 2443 | Summary |
| summary.smooth.forecast | methods.R | 2448 | Summary forecast |
| accuracy.smooth | methods.R | 2486 | Accuracy |
| accuracy.smooth.forecast | methods.R | 2503 | Accuracy forecast |
| paletteDetector | methods.R | 2519 | Palette |
| oes | oes.R | 107 | OES entry |
| intermittentParametersSetter | iss.R | 1 | Intermittent setup |
| sim.ces | simces.R | 78 | Sim CES |
| sim.es | simes.R | 113 | Sim ETS |
| reapply | reapply.R | 75 | Reapply generic |
| reapply.default | reapply.R | 78 | Reapply default |
| reapply.adam | reapply.R | 87 | Reapply adam |
| reapply.adamCombined | reapply.R | 781 | Reapply combined |
| plot.reapply | reapply.R | 816 | Plot reapply |
| plot.reapplyCombined | reapply.R | 860 | Plot combined |
| print.reapply | reapply.R | 906 | Print reapply |
| print.reapplyCombined | reapply.R | 914 | Print combined |
| reforecast | reapply.R | 923 | Reforecast generic |
| reforecast.default | reapply.R | 929 | Reforecast default |
| reforecast.adam | reapply.R | 941 | Reforecast adam |
| cma | cma.R | 68 | CMA entry |
| sim.ssarima | simssarima.R | 83 | Sim SSARIMA |
| oesg | oesg.R | 95 | OESG entry |
| sim.gum | simgum.R | 76 | Sim GUM |
| sma | adam-sma.R | 101 | SMA entry |
| smoothCombine | smoothCombine.R | 92 | Combine entry |
| auto.adam | autoadam.R | 19 | Auto ADAM |
| covarAnal | variance-covariance.R | 1 | Covariance analytical |
| adamVarAnal | variance-covariance.R | 87 | ADAM variance analytical |
| sowhat | sowhat.R | 27 | Sowhat |
| release_questions | sowhat.R | 36 | Release checks |
| sim.oes | simoes.R | 84 | Sim OES |
| is.smooth | isFunctions.R | 37 | Type check |
| is.smoothC | isFunctions.R | 43 | Type check |
| is.msarima | isFunctions.R | 49 | Type check |
| is.oes | isFunctions.R | 55 | Type check |
| is.oesg | isFunctions.R | 61 | Type check |
| is.smooth.sim | isFunctions.R | 67 | Type check |
| is.smooth.forecast | isFunctions.R | 73 | Type check |
| is.adam | isFunctions.R | 99 | Type check |
| is.adam.sim | isFunctions.R | 105 | Type check |
| depricator | depricator.R | 1 | Deprecation |
| .onAttach | zzz.R | 1 | Attach hook |
| .onLoad | zzz.R | 25 | Load hook |
| .onUnload | zzz.R | 30 | Unload hook |

## Local Functions (inside adam())

These are defined inside `adam()` and not exported. They appear in adam.R around lines 656-1844:
- **architector** (656): Define model architecture, lags, profiles
- **creator** (750): Build state-space matrices (matVt, matF, vecg, matw)
- **filler** (1194): Populate matrices from parameter vector B
- **initialiser** (1402): Initial parameter vector and bounds for optimization
