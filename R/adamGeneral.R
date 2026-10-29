commonParametersChecker <- function(data, model, lags, formulaToUse, orders, constant=FALSE, arma,
                                    persistence, phi, initial,
                                    distribution=c("default","dnorm","dlaplace","ds","dgnorm",
                                                   "dlnorm","dinvgauss","dgamma","plogis"),
                                    loss, h, holdout, occurrence,
                                    ic=c("AICc","AIC","BIC","BICc"), bounds=c("usual","admissible","none"),
                                    regressors, yName,
                                    silent, modelDo,
                                    ellipsis, fast=FALSE){

    #### Data ####
    list2env(adam_checkData(data, lags, h, holdout, yName, modelDo, formulaToUse),
             envir=environment());

    #### Check what is used for the model ####
    list2env(adam_checkModel(fast, model, modelDo),
             envir=environment());

    #### Check the lags vector ####
    list2env(adam_checkLags(lags, obsInSample), envir=environment());

    #### ARIMA term ####
    list2env(adam_checkArima(Stype, arma, componentsNamesETS, componentsNumberETS, distribution,
                  etsModel, fast, lags, model, modelDo, modelIsTrendy, modelsPool, orders,
                  parametersNumber),
             envir=environment());
    # The orders are selected by auto.adam(), which calls the checker again
    if(select){
        return(list(select=TRUE, distribution=distribution));
    }

    #### Loss function type ####
    list2env(adam_checkLoss(loss, h), envir=environment());

    #### Explanatory variables: xregModel and regressors ####
    regressors <- match.arg(regressors,c("use","select","adapt"));
    xregModel <- !is.null(xregData);
    # The formula term of each expanded regressor, for regressors="select" with a formula
    xregTerms <- NULL;

    #### Persistence provided ####
    list2env(adam_checkPersistence(Stype, Ttype, componentsNumberETSSeasonal, etsModel, lagsLength,
                  lagsModelSeasonal, modelDo, modelIsTrendy, parametersNumber, persistence,
                  xregModel),
             envir=environment());

    #### Phi ####
    list2env(adam_checkPhi(phi, etsModel, modelDo, damped, parametersNumber), envir=environment());

    #### This needs to be amended after developing the first prototype! ####
    # If we have the zoo class and weird lags, amend profiles
    # Weird lags (dst and fractional): 24, 24*2 (half hour), 52, 24*4 (15 minutes), 7*24, 7*48, 365, 24*52, 24*365
    # if(any(yClasses=="zoo") && any(lags %in% c(24, 48, 52, 96, 168, 336, 365, 1248, 8760))){
        # For hourly, half-hourly and quarter hour data, just amend the profiles for DST.
        # For daily, repeat profile of 28th on 29th February.
        # For weekly, repeat the last week, when we have 53 instead of 52.
    # }

    #### Occurrence variable ####
    list2env(adam_checkOccurrence(Etype, Stype, Ttype, etsModel, h, lagsModelMax, model,
                  modelsPool, obsInSample, occurrence, y, yClasses, yFrequency, yInSample,
                  yNAValues, yStart),
             envir=environment());

    #### Initial values ####
    list2env(adam_checkInitial(Etype, Stype, arimaModel, componentsNumberETSSeasonal, distribution,
                  etsModel, initial, initialArimaNumber, lagsModelSeasonal, loss, modelDo,
                  modelIsTrendy, parametersNumber, silent, xregModel),
             envir=environment());

    #### xreg preparation ####
    xregChecked <- adam_checkXreg(Etype, arimaModel, bounds, data, distribution, ellipsis, etsModel,
                              formula, formulaToUse, h, holdout, ic, initialType, initialXreg,
                              initialXregEstimate, lagsModelAll, loss, lossFunction, obsAll, obsInSample,
                              occurrence, occurrenceMissingOnly, occurrenceModel, occurrenceProbability,
                              omModel, otLogical, parametersNumber, persistenceEstimate,
                              persistenceLevelEstimate, persistenceSeasonalEstimate, persistenceTrendEstimate,
                              persistenceXreg, persistenceXregEstimate, regressors, responseName, xregData,
                              xregModel, xregTerms, y, yHoldout, yInSample, yNAValues, yName);
    # A pure regression is the alm() model, returned as it is
    if(is.alm(xregChecked)){
        return(xregChecked);
    }
    list2env(xregChecked, envir=environment());

    #### Conclusions about the initials ####
    # Make sure that only important elements are estimated.
    if(!modelIsTrendy){
        initialTrendEstimate <- FALSE;
        initialTrend <- NULL;
    }
    if(!modelIsSeasonal){
        initialSeasonalEstimate <- FALSE;
        initialSeasonal <- NULL;
    }
    if(!arimaModel){
        initialArimaEstimate <- FALSE;
        initialArima <- NULL;
    }
    if(!xregModel){
        initialXregEstimate <- FALSE;
        initialXreg <- NULL;
    }

    # If we don't need to estimate anything, flag initialEstimate
    if(!any(c(etsModel && initialLevelEstimate,
              (etsModel && modelIsTrendy && initialTrendEstimate),
              (etsModel && modelIsSeasonal && all(initialSeasonalEstimate)),
              (arimaModel && initialArimaEstimate),
              (xregModel && initialXregEstimate)))){
        initialEstimate[] <- FALSE;
    }
    else{
        initialEstimate[] <- TRUE;
    }

    # If at least something is provided, flag it as "provided"
    if((etsModel && !initialLevelEstimate) ||
       (etsModel && modelIsTrendy && !initialTrendEstimate) ||
       (etsModel && modelIsSeasonal && any(!initialSeasonalEstimate)) ||
       (arimaModel && !initialArimaEstimate) ||
       (xregModel && !initialXregEstimate)){
        initialType[] <- "provided";
    }

    # Observations in the states matrix
    # Define the number of cols that should be in the matvt
    obsStates <- obsInSample + lagsModelMax;

    if(any(yInSample[!yNAValues[1:obsInSample]]<=0) &&
       any(distribution==c("dinvgauss","dgamma","dlnorm","dllaplace","dls","dlgnorm")) &&
       !occurrenceModel && (occurrence!="provided")){
        warning(paste0("You have non-positive values in the data. ",
                       "The distribution ",distribution," does not support that. ",
                       "This might lead to problems in the estimation."),
                call.=FALSE);
    }

    # Update the number of parameters
    if(occurrenceModelProvided){
        parametersNumber[2,3] <- nparam(omModel);
        pForecast <- c(forecast(omModel, h=h)$mean);
    }

    #### Information Criteria ####
    ic <- match.arg(ic,c("AICc","AIC","BIC","BICc"));
    icFunction <- switch(ic,
                         "AIC"=AIC,
                         "AICc"=AICc,
                         "BIC"=BIC,
                         "BICc"=BICc);

    #### Bounds for the smoothing parameters ####
    bounds <- match.arg(bounds,c("usual","admissible","none"));


    #### Checks for the potential number of degrees of freedom ####
    list2env(adam_checkSampleSize(Etype, Stype, Ttype, allowMultiplicative, arEstimate, arOrders,
                  arRequired, arimaModel, componentsNamesARIMA, componentsNumberARIMA,
                  componentsNumberETS, damped, distribution, etsModel, iOrders, iRequired,
                  initialArimaNumber, initialEstimate, initialLevel, initialLevelEstimate,
                  initialSeasonalEstimate, initialTrendEstimate, initialType, initialXregEstimate,
                  lagsModelARIMA, lagsModelAll, lagsModelMax, lagsModelSeasonal, maEstimate,
                  maOrders, maRequired, model, modelDo, modelIsSeasonal, modelIsTrendy, modelsPool,
                  nonZeroARI, nonZeroMA, obsInSample, obsNonzero, occurrence, occurrenceModel,
                  occurrenceModelProvided, orders, otLogical, parametersNumber,
                  persistenceEstimate, persistenceLevel, persistenceLevelEstimate,
                  persistenceSeasonalEstimate, persistenceTrendEstimate, persistenceXregEstimate,
                  phiEstimate, select, silent, xregModel, xregNumber, yInSample, yNAValues),
             envir=environment());

    #### Process ellipsis ####
    list2env(adam_checkEllipsis(ellipsis, loss, distribution, initialType, lags, arimaModel,
                                lagsModelMax, obsInSample),
             envir=environment());

    #### Constant ####
    list2env(adam_checkConstant(constant, iOrders, etsModel), envir=environment());

    # If there is no model, return a constant level
    if(!etsModel && !arimaModel && !xregModel){
        modelsPool <- NULL;
        if(!constantRequired){
            constantEstimate <- constantRequired <- TRUE;
            constantName <- "constant";
        }

        model <- "NNN";
        if(is.null(B)){
            modelDo <- "estimate";
        }
        Etype <- switch(distribution,
                        "default"=,"dnorm"=,"dlaplace"=,"ds"=,"dgnorm"=,"dlogis"=,"plogis"=,"dt"="A",
                        "dlnorm"=,"dllaplace"=,"dls"=,"dlgnorm"=,"dinvgauss"=,"dgamma"="M",
                        "A");
        Ttype <- "N";
        Stype <- "N";
        phiEstimate <- FALSE;
        parametersNumber[1,1] <- 0;
        parametersNumber[2,1] <- 2;
    }

    # LASSO / RIDGE with lambda=1 is the penalty alone, which is zero with the parameters
    # at their shrinkage targets: no smoothing, phi=1, AR=1 and MA=0. They are set to
    # them, leaving the initials and the regressors, estimated with lambda=0 in adam(),
    # or nothing, and then the model is used as it is
    if(any(loss==c("LASSO","RIDGE")) && lambda==1){
        if(etsModel){
            persistenceEstimate[] <- persistenceLevelEstimate[] <- persistenceTrendEstimate[] <-
                persistenceSeasonalEstimate[] <- phiEstimate[] <- FALSE;
            persistenceLevel <- persistenceTrend <- persistenceSeasonal <- 0;
            phi <- 1;
        }
        if(xregModel){
            persistenceXregEstimate[] <- FALSE;
            persistenceXreg <- 0;
        }
        if(arimaModel){
            arEstimate[] <- maEstimate[] <- FALSE;
            armaParameters <- NULL;
            for(i in seq_along(lags)){
                armaParameters <- c(armaParameters,
                                    setNames(rep(1, arOrders[i]),
                                             paste0("phi", seq_len(arOrders[i]), "[", lags[i], "]")),
                                    setNames(rep(0, maOrders[i]),
                                             paste0("theta", seq_len(maOrders[i]), "[", lags[i], "]")));
            }
        }
    }

    # See if the estimation of the model is not needed (do we estimate anything?)
    if(!any(c(etsModel & c(persistenceLevelEstimate, persistenceTrendEstimate,
                           persistenceSeasonalEstimate, phiEstimate,
                           all(initialType!=c("complete","backcasting","gradient")) & c(initialLevelEstimate,
                                                                             initialTrendEstimate,
                                                                             initialSeasonalEstimate)),
              arimaModel & c(arEstimate, maEstimate,
                             all(initialType!=c("complete","backcasting","gradient")) & initialEstimate & initialArimaEstimate),
              xregModel & c(persistenceXregEstimate, (initialType!="complete") & initialXregEstimate),
              constantEstimate,
              otherParameterEstimate))){
        modelDo <- "use";
    }

    # Switch usual bounds to the admissible if there's no ETS - this speeds up ARIMA
    if(!etsModel && bounds=="usual"){
        bounds[] <- "admissible";
    }

    # If we do model selection / combination with non-standard losses, complain
    if(any(modelDo==c("select","combine")) &&
       ((any(loss==c("MSE","MSEh","MSCE","GPL")) && all(distribution!=c("default","dnorm"))) ||
        (any(loss==c("MAE","MAEh","MACE")) && all(distribution!=c("default","dlaplace"))) ||
        (any(loss==c("HAM","HAMh","CHAM")) && all(distribution!=c("default","ds"))) ||
        (any(loss==c("TMSE","GTMSE","TMAE","GTMAE","THAM","GTHAM"))))){
        warning("The model selection only works in case of loss='likelihood'. I hope you know what you are doing.",
                call.=FALSE);
    }

    # Just in case, give names to yHoldout and yInSample
    colnames(yInSample) <- responseName;
    if(holdout){
        colnames(yHoldout) <- responseName;
    }

    #### Return the validated parameters as a named list ####
    return(list(
        # Actuals
        y = y,
        yHoldout = yHoldout,
        yInSample = yInSample,
        yNAValues = yNAValues, occurrenceMissingOnly = occurrenceMissingOnly,
        # Index and structure
        yClasses = yClasses,
        yIndex = yIndex,
        yInSampleIndex = yInSampleIndex,
        yForecastIndex = yForecastIndex,
        yIndexAll = yIndexAll,
        yFrequency = yFrequency,
        yStart = yStart,
        yForecastStart = yForecastStart,
        horizon = h,
        h = h,
        holdout = holdout,
        # Observation counts
        obsInSample = obsInSample,
        obsAll = obsAll,
        obsStates = obsStates,
        obsNonzero = obsNonzero,
        obsZero = obsZero,
        parametersNumber = parametersNumber,
        # Model type
        etsModel = etsModel,
        model = model,
        Etype = Etype,
        Ttype = Ttype,
        Stype = Stype,
        modelIsTrendy = modelIsTrendy,
        modelIsSeasonal = modelIsSeasonal,
        modelsPool = modelsPool,
        damped = damped,
        modelDo = modelDo,
        allowMultiplicative = allowMultiplicative,
        # Component counts and names
        componentsNumberETS = componentsNumberETS,
        componentsNamesETS = componentsNamesETS,
        componentsNumberETSNonSeasonal = componentsNumberETS - componentsNumberETSSeasonal,
        componentsNumberETSSeasonal = componentsNumberETSSeasonal,
        componentsNumberARIMA = componentsNumberARIMA,
        componentsNamesARIMA = componentsNamesARIMA,
        # Lags
        lags = lags,
        lagsModel = lagsModel,
        lagsModelSeasonal = lagsModelSeasonal,
        lagsModelARIMA = lagsModelARIMA,
        lagsModelAll = lagsModelAll,
        lagsModelMax = lagsModelMax,
        headLength = headLength,
        # Persistence
        persistence = persistence,
        persistenceEstimate = persistenceEstimate,
        persistenceLevel = persistenceLevel,
        persistenceLevelEstimate = persistenceLevelEstimate,
        persistenceTrend = persistenceTrend,
        persistenceTrendEstimate = persistenceTrendEstimate,
        persistenceSeasonal = persistenceSeasonal,
        persistenceSeasonalEstimate = persistenceSeasonalEstimate,
        persistenceXreg = persistenceXreg,
        persistenceXregEstimate = persistenceXregEstimate,
        persistenceXregProvided = persistenceXregProvided,
        # phi
        phi = phi,
        phiEstimate = phiEstimate,
        # Initials
        initial = initial,
        initialType = initialType,
        initialEstimate = initialEstimate,
        initialLevel = initialLevel,
        initialLevelEstimate = initialLevelEstimate,
        initialTrend = initialTrend,
        initialTrendEstimate = initialTrendEstimate,
        initialSeasonal = initialSeasonal,
        initialSeasonalEstimate = initialSeasonalEstimate,
        initialArima = initialArima,
        initialArimaEstimate = initialArimaEstimate,
        initialArimaNumber = initialArimaNumber,
        initialXregEstimate = initialXregEstimate,
        initialXregProvided = initialXregProvided,
        # Occurrence model
        omModel = omModel,
        occurrenceModel = occurrenceModel,
        occurrenceModelProvided = occurrenceModelProvided,
        occurrence = occurrence,
        pFitted = pFitted,
        pForecast = pForecast,
        ot = ot,
        otLogical = otLogical,
        # Distribution, loss, bounds and IC
        distribution = distribution,
        loss = loss,
        lossFunction = lossFunction,
        multisteps = multisteps,
        ic = ic,
        icFunction = icFunction,
        bounds = bounds,
        # ARIMA components
        arimaModel = arimaModel,
        orders = orders,
        arOrders = arOrders,
        iOrders = iOrders,
        maOrders = maOrders,
        arRequired = arRequired,
        iRequired = iRequired,
        maRequired = maRequired,
        arEstimate = arEstimate,
        maEstimate = maEstimate,
        armaParameters = armaParameters,
        nonZeroARI = nonZeroARI,
        nonZeroMA = nonZeroMA,
        select = select,
        # Explanatory variables
        xregModel = xregModel,
        regressors = regressors,
        xregModelInitials = xregModelInitials,
        data = data,
        xregData = xregData,
        xregNumber = xregNumber,
        xregNames = xregNames,
        responseName = responseName,
        formula = formulaToUse,
        xregTerms = xregTerms,
        xregParametersMissing = xregParametersMissing,
        xregParametersIncluded = xregParametersIncluded,
        xregParametersEstimated = xregParametersEstimated,
        xregParametersPersistence = xregParametersPersistence,
        # Constant
        constantRequired = constantRequired,
        constantEstimate = constantEstimate,
        constantValue = constantValue,
        constantName = constantName,
        # Optimisation
        maxeval = maxeval,
        maxtime = maxtime,
        xtol_rel = xtol_rel,
        xtol_abs = xtol_abs,
        ftol_rel = ftol_rel,
        ftol_abs = ftol_abs,
        algorithm = algorithm,
        print_level = print_level,
        B = B,
        lb = lb,
        ub = ub,
        # Distribution shape and regularisation
        other = other,
        otherParameterEstimate = otherParameterEstimate,
        lambda = lambda,
        # Misc
        nIterations = nIterations,
        smoother = smoother,
        FI = FI,
        stepSize = stepSize
    ));
}

#### adamSpecificChecker: thin wrapper adding outliers/distribution early-exit ####
adamSpecificChecker <- function(data, model, lags, formulaToUse, orders, constant=FALSE, arma,
                                outliers=c("ignore","use","select"), level=0.99,
                                persistence, phi, initial,
                                distribution=c("default","dnorm","dlaplace",
                                               "ds","dgnorm","dlnorm","dinvgauss","dgamma","plogis"),
                                loss, h, holdout, occurrence,
                                ic=c("AICc","AIC","BIC","BICc"),
                                bounds=c("usual","admissible","none"),
                                regressors, yName,
                                silent, modelDo,
                                ellipsis, fast=FALSE){
    outliers <- match.arg(outliers)
    if(!fast){
        distribution <- match.arg(distribution[1],
                                   c("default","dnorm","dlaplace",
                                     "ds","dgnorm","dlnorm","dinvgauss","dgamma","plogis"))
    }

    if(outliers != "ignore"){
        return(list(select=TRUE, distribution=distribution, outliers=outliers))
    }

    result <- commonParametersChecker(data=data, model=model, lags=lags,
                                      formulaToUse=formulaToUse,
                                      orders=orders, constant=constant, arma=arma,
                                      persistence=persistence, phi=phi, initial=initial,
                                      distribution=distribution, loss=loss,
                                      h=h, holdout=holdout,
                                      occurrence=occurrence, ic=ic, bounds=bounds,
                                      regressors=regressors, yName=yName,
                                      silent=silent, modelDo=modelDo,
                                      ellipsis=ellipsis, fast=fast)

    if(is.alm(result)){
        return(result)
    }

    return(c(result, list(outliers=outliers)))
}

#### Backward-compatible alias ####
parametersChecker <- adamSpecificChecker;
