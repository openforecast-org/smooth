# Global variables used across smooth package functions
# This consolidates all utils::globalVariables() calls from various files
# Variables are grouped by the primary functions/files that use them

utils::globalVariables(c(

  # Variables injected via list2env() in adam.R, adamGeneral.R, om.R, omg.R, ssarima.R, ces.R, gum.R
  "B", "FI",
  "initialXregProvided",
  "lb", "ub",
  "otherReturned",
  "persistenceXregProvided",
  "print_level",
  "profilesRecentInitial",
  "select",

  # adam.R - Main ADAM function and related infrastructure
  "adamCpp", "adamETS", "adamFitted",
  "algorithm",
  "allowMultiplicative",
  "arEstimate", "arOrders", "arRequired",
  "arimaModel", "arimaPolynomials", "armaParameters",
  "componentsNamesARIMA", "componentsNamesETS",
  "componentsNumberARIMA", "componentsNumberETS",
  "componentsNumberETSNonSeasonal", "componentsNumberETSSeasonal",
  "constantEstimate", "constantName", "constantRequired", "constantValue",
  "damped", "dataStart",
  "digits",
  "etsModel",
  "ftol_abs", "ftol_rel",
  "horizon",
  "icFunction",
  "indexLookupTable",
  "initialArima", "initialArimaEstimate", "initialArimaNumber",
  "initialEstimate",
  "initialLevel", "initialLevelEstimate",
  "initialSeasonal", "initialSeasonalEstimate", "initialSeasonEstimate",
  "initialTrend", "initialTrendEstimate",
  "iOrders", "iRequired",
  "lagsModel", "lagsModelAll", "lagsModelARIMA", "lagsModelSeasonal",
  "lambda", "lossFunction",
  "maEstimate", "maOrders", "maRequired",
  "matF", "matVt", "matWt",
  "maxeval", "maxtime",
  "modelIsMultiplicative", "modelIsSeasonal", "modelIsTrendy",
  "nComponentsAll", "nComponentsNonSeasonal", "nIterations",
  "nParamEstimated",
  "nonZeroARI", "nonZeroMA",
  "other", "otherParameterEstimate",
  "persistenceLevel", "persistenceLevelEstimate",
  "persistenceSeasonal", "persistenceSeasonalEstimate",
  "persistenceTrend", "persistenceTrendEstimate",
  "pForecast", "profilesRecentTable",
  "responseName",
  "smoother", "stepSize",
  "vecG",
  "xregModelInitials",
  "xregParametersEstimated", "xregParametersIncluded",
  "xregParametersMissing", "xregParametersPersistence",
  "xtol_abs", "xtol_rel",
  "yClasses", "yDenominator",
  "yForecastIndex", "yIndexAll", "yInSampleIndex",
  "yNAValues", "yStart",

  # ssfunctions.R - State space utility functions
  "AR", "MA",
  "CF",
  "constant", "cumulative",
  "errors",
  "Etype",
  "h", "holdout",
  "imodel", "interval", "intervalType",
  "lags", "level", "loss",
  "matat", "matFX", "matvt", "matxt", "measurement", "model", "multisteps",
  "nExovars", "nParam",
  "obsAll", "obsInSample", "obsNonzero", "obsStates", "obsZero",
  "oesmodel", "ot", "orders",
  "pFitted",
  "rounded",
  "Stype",
  "transition", "Ttype",
  "vecgX",
  "xreg",
  "y", "yFitted",

  # adam-ces.R, adam-gum.R, adam-ssarima.R - Model-specific wrappers
  # (ces, gum, ssarima - all share similar xreg and distribution variables)
  "distribution",
  "initialXregEstimate",
  "otLogical",
  "persistenceXreg", "persistenceXregEstimate",
  "xregData", "xregModel", "xregNames", "xregNumber",
  "yFrequency", "yHoldout", "yIndex",

  # autossarima.R - Automatic ARIMA selection
  "ar.orders", "i.orders", "ma.orders",
  "initialType",
  "silent", "silentGraph", "silentLegend",

  # oes.R - Occurrence ETS model
  "initialValue",
  "lagsModelMax",
  "modelDo",

  # oesg.R - Occurrence ETS general model
  "modelsPool",
  "parametersNumber",
  "regressors",
  "updateX",

  # iss.R - Intermittent state space utilities
  "obs",
  "occurrenceModel", "occurrenceModelProvided",
  "yInSample",

  # sparma.R - Sparse ARMA (minimal - only constant parameters)
  # constantRequired, constantEstimate already listed under adam.R

  #### Additional variables used across multiple functions

  # cfObjective - Cost/objective function (5 files: adam-sma.R, cma.R, methods.R, ssfunctions.R, RcppExports)
  "cfObjective",

  # dataFreq - Data frequency/periodicity (9 files: adam-ssarima.R, automsarima.R, autossarima.R,
  #            iss.R, methods.R, oesg.R, oes.R, ssfunctions.R, RcppExports)
  "dataFreq",

  # errors.mat - Multi-step forecasting error matrix (4 files: methods.R, oes.R, ssfunctions.R, C++ interface)
  "errors.mat",

  # initial - Initial values/states parameter (33 files: ubiquitous across all models, simulations, utilities)
  "initial",

  # matw - Measurement vector/matrix (9 files: oesg.R, oes.R, RcppExports.R, simces.R, simes.R,
  #        simgum.R, simssarima.R, ssfunctions.R, C++ interface)
  "matw",

  # nComponents - Number of model components/states (8 files: autogum.R, methods.R, oesg.R, oes.R,
  #               randomARIMA.R, ssfunctions.R, variance-covariance.R, internal utilities)
  "nComponents",

  # nParamOccurrence - Number of parameters in occurrence model (5 files: adamGeneral.R, adam.R,
  #                    iss.R, ssfunctions.R, intermittent demand functions)
  "nParamOccurrence",

  # occurrence - Occurrence model type for intermittent demand (30+ files: adam.R, all adam-*.R wrappers,
  #              oes.R, oesg.R, adamGeneral.R, iss.R, all auto functions with occurrence support)
  "occurrence",

  # persistenceEstimate - Flag for estimating persistence parameters (8 files: adamGeneral.R, adam.R,
  #                       adam-gum.R, adam-ssarima.R, oesg.R, oes.R, ssfunctions.R, parameter handlers)
  "persistenceEstimate",

  # phiEstimate - Flag for estimating phi (damping) parameter (6 files: adamGeneral.R, adam.R,
  #               oesg.R, oes.R, ssfunctions.R, damping parameter handling)
  "phiEstimate",

  # s2 - Variance estimate (sigma-squared) for inference and intervals (14 files: adam.R, adam-es.R,
  #      adam-msarima.R, adam-sma.R, automsarima.R, autossarima.R, cma.R, methods.R, oesg.R,
  #      oes.R, smoothCombine.R, ssfunctions.R, variance-covariance.R, all interval calculations)
  "s2",

  # silentText - Text output suppression flag (5 files: automsarima.R, oesg.R, oes.R,
  #              ssfunctions.R, output control)
  "silentText",

  # vecg - Persistence vector (lowercase 'g') in state space formulation (6 files: oesg.R, oes.R,
  #        RcppExports.R, simssarima.R, ssfunctions.R, C++ interface)
  "vecg",

  # yForecastStart - Starting point for forecast period (14 files: adam.R, adam-ces.R, adam-gum.R,
  #                  adam-ssarima.R, adamGeneral.R, automsarima.R, autossarima.R, msdecompose.R,
  #                  oesg.R, oes.R, reapply.R, smoothCombine.R, sparma.R, ssfunctions.R)
  "yForecastStart",

  # yLower, yUpper - Prediction interval bounds (8 files each: adam.R, automsarima.R, autossarima.R,
  #                  oesg.R, oes.R, reapply.R, ssfunctions.R, all interval calculations)
  "yLower", "yUpper"
))
