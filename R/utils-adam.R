#### Internal helper functions shared across ADAM-family models ####
# These are extracted from parametersChecker() to allow reuse by om() and
# other future functions without duplicating logic.

#### Data preparation ####
adam_checkData <- function(data, lags, h, holdout, yName, modelDo, formulaToUse) {
    responseName <- yName

    # Extract from sim/Mdata objects
    if(is.adam.sim(data) || is.smooth.sim(data)){
        data <- data$data
        lags <- frequency(data)
    }
    else if(inherits(data,"Mdata")){
        h <- data$h
        holdout <- TRUE
        if(modelDo!="use"){
            lags <- frequency(data$x)
        }
        data <- ts(c(data$x,data$xx),start=start(data$x),frequency=frequency(data$x))
    }

    # Extract index
    ### tsibble has its own index function, so shit happens because of it...
    if(inherits(data,"tbl_ts")){
        yIndex <- data[[1]]
        if(any(duplicated(yIndex))){
            warning(paste0("You have duplicated time stamps in the variable ",yName,
                           ". I will refactor this."),call.=FALSE)
            yIndex <- yIndex[1] + c(1:length(data[[1]])) * diff(tail(yIndex,2))
        }
    }
    else{
        yIndex <- try(time(data),silent=TRUE)
        if(inherits(yIndex,"try-error")){
            if(!is.data.frame(data) && !is.null(dim(data))){
                yIndex <- as.POSIXct(rownames(data))
            }
            else if(is.data.frame(data)){
                yIndex <- c(1:nrow(data))
            }
            else{
                yIndex <- c(1:length(data))
            }
        }
    }
    yClasses <- class(data)

    # Multi-column data: extract response and xreg
    if(!is.null(ncol(data)) && ncol(data)>1){
        xregData <- data
        if(inherits(data,"tbl_df") || inherits(data,"tbl")){
            data <- as.data.frame(data)
        }
        if(!is.null(formulaToUse)){
            responseName <- all.vars(formulaToUse)[1]
            y <- data[,responseName]
        }
        else{
            responseName <- colnames(xregData)[1]
            if(inherits(data,"tbl_ts")){
                y <- data$value
            }
            else if(inherits(data,"data.table") || inherits(data,"data.frame")){
                y <- data[[1]]
            }
            else if(inherits(data,"zoo")){
                if(ncol(data)>1){
                    xregData <- as.data.frame(data)
                }
                y <- zoo(data[,1],order.by=time(data))
            }
            else{
                y <- data[,1]
            }
        }
        yIndex <- try(time(y),silent=TRUE)
        if(inherits(yIndex,"try-error")){
            if(!is.null(dim(data))){
                yIndex <- try(as.POSIXct(rownames(data)),silent=TRUE)
                if(inherits(yIndex,"try-error")){
                    yIndex <- c(1:nrow(data))
                }
            }
            else{
                yIndex <- c(1:length(y))
            }
        }
        else{
            yClasses <- class(y)
        }
    }
    else{
        xregData <- NULL
        if(!is.null(ncol(data)) && !is.null(colnames(data)[1])){
            responseName <- colnames(data)[1]
            y <- data[,1]
        }
        else{
            y <- data
        }
    }

    # deparse(substitute(data)) in the caller returns the whole series whenever
    # the data arrives by value -- do.call(adam, list(y)), for instance. The
    # resulting "name" is useless and, past R's 10000-byte limit on names, makes
    # the default formula unbuildable, so fall back to a plain name.
    responseName <- make.names(responseName)
    if(nchar(responseName)>100){
        responseName <- "y"
    }
    if(nchar(yName)>100){
        yName <- "data"
    }

    obsAll <- length(y) + (1 - holdout)*h
    obsInSample <- length(y) - holdout*h

    if(obsInSample<=0){
        stop("The number of in-sample observations is not positive. Cannot do anything.",
             call.=FALSE)
    }

    # The missing values, filled for the initialisation and skipped by the fit
    yNAValues <- is.na(y)
    if(any(yNAValues)){
        warning("Data contains NAs. The values will be ignored during the model construction.",
                call.=FALSE)
        # The estimates rest on the observed values only
        if(sum(!yNAValues[1:obsInSample]) < obsInSample/2){
            warning("More than half of the in-sample data is missing (", sum(yNAValues[1:obsInSample]),
                    " of ", obsInSample, "): the estimates rest on few observations.", call.=FALSE)
        }
        # The values for the initialisation only, as the fit skips them: a polynomial of the
        # time and harmonics fitted to the observed in-sample values, shared with Python.
        # Those of the holdout are placeholders, which nothing takes for data
        lagMax <- max(max(lags), 10);
        yFilled <- naFillCpp(as.numeric(y), lagMax);
        yFilled[1:obsInSample] <- naFillCpp(as.numeric(y[1:obsInSample]), lagMax);
        y[yNAValues] <- yFilled[yNAValues]
        if(!is.null(xregData)){
            xregData[yNAValues,responseName] <- y[yNAValues]
        }
    }

    # Determine ts class
    if(all(yClasses=="integer") || all(yClasses=="numeric") ||
       all(yClasses=="data.frame") || all(yClasses=="matrix")){
        if(any(class(yIndex) %in% c("POSIXct","Date"))){
            yClasses <- "zoo"
        }
        else{
            yClasses <- "ts"
        }
    }
    yFrequency <- frequency(y)
    yStart <- yIndex[1]
    yInSample <- matrix(y[1:obsInSample],ncol=1)
    if(holdout){
        yForecastStart <- yIndex[obsInSample+1]
        yHoldout <- matrix(y[-c(1:obsInSample)],ncol=1)
        yForecastIndex <- yIndex[-c(1:obsInSample)]
        yInSampleIndex <- yIndex[c(1:obsInSample)]
        yIndexAll <- yIndex
    }
    else{
        yInSampleIndex <- yIndex
        if(any(yClasses=="ts")){
            yIndexDiff <- deltat(yIndex)
            yForecastIndex <- yIndex[obsInSample]+yIndexDiff*c(1:max(h,1))
        }
        else{
            yIndexDiff <- diff(tail(yIndex,2))
            yForecastIndex <- yIndex[obsInSample]+yIndexDiff*c(1:max(h,1))
        }
        yForecastStart <- yIndex[obsInSample]+yIndexDiff
        yHoldout <- NULL
        yIndexAll <- c(yIndex,yForecastIndex)
    }

    if(!is.numeric(yInSample)){
        stop("The provided data is not numeric! Can't construct any model!", call.=FALSE)
    }

    # Add trend variable to xreg if requested via formula but not present
    if(!is.null(formulaToUse) &&
       any(all.vars(formulaToUse)=="trend") && all(colnames(xregData)!="trend")){
        if(!is.null(xregData)){
            xregData <- cbind(xregData,trend=c(1:obsAll))
        }
        else{
            xregData <- cbind(y=y,trend=c(1:obsAll))
        }
    }

    parametersNumber <- matrix(0,2,5,
                               dimnames=list(c("Estimated","Provided"),
                                             c("nParamInternal","nParamXreg",
                                               "nParamOccurrence","nParamScale","nParamAll")))

    return(list(
        y = y,
        yHoldout = yHoldout,
        yInSample = yInSample,
        yNAValues = yNAValues,
        yIndex = yIndex,
        yClasses = yClasses,
        yFrequency = yFrequency,
        yStart = yStart,
        yForecastStart = yForecastStart,
        yInSampleIndex = yInSampleIndex,
        yForecastIndex = yForecastIndex,
        yIndexAll = yIndexAll,
        obsInSample = obsInSample,
        obsAll = obsAll,
        xregData = xregData,
        responseName = responseName,
        yName = yName,
        parametersNumber = parametersNumber,
        lags = lags,
        h = h,
        holdout = holdout
    ))
}

#### Optimiser / ellipsis parameter processing ####
adam_checkOptimizer <- function(ellipsis, loss, distribution, initialType, lags, arimaModel) {
    if(is.null(ellipsis$maxeval)){
        maxeval <- NULL
        if(any(lags>24) && arimaModel && any(initialType==c("optimal","two-stage"))){
            warning(paste0("The estimation of ARIMA model with initial='optimal' on",
                           " high frequency data might take more time to converge.",
                           " Consider setting maxeval to a higher value (e.g.",
                           " maxeval=10000) or using initial='backcasting'."),
                    call.=FALSE, immediate.=TRUE)
        }
    }
    else{
        maxeval <- ellipsis$maxeval
    }
    maxtime <- if(is.null(ellipsis$maxtime)) -1 else ellipsis$maxtime
    xtol_rel <- if(is.null(ellipsis$xtol_rel)) 1E-6 else ellipsis$xtol_rel
    xtol_abs <- if(is.null(ellipsis$xtol_abs)) 1E-8 else ellipsis$xtol_abs
    ftol_rel <- if(is.null(ellipsis$ftol_rel)) 1E-8 else ellipsis$ftol_rel
    ftol_abs <- if(is.null(ellipsis$ftol_abs)) 0 else ellipsis$ftol_abs
    algorithm <- if(is.null(ellipsis$algorithm)) "NLOPT_LN_NELDERMEAD" else ellipsis$algorithm
    print_level <- if(is.null(ellipsis$print_level)) 0 else ellipsis$print_level
    lb <- if(is.null(ellipsis$lb)) NULL else ellipsis$lb
    ub <- if(is.null(ellipsis$ub)) NULL else ellipsis$ub
    B  <- if(is.null(ellipsis$B))  NULL else ellipsis$B

    lambda <- other <- NULL
    otherParameterEstimate <- FALSE

    if(any(loss==c("LASSO","RIDGE"))){
        if(is.null(ellipsis$lambda)){
            warning("You have not provided lambda parameter. I will set it to zero.", call.=FALSE)
            lambda <- 0
        }
        else{
            lambda <- ellipsis$lambda
        }
    }

    if(distribution=="dalaplace"){
        if(is.null(ellipsis$alpha)){
            other <- 0.5
            otherParameterEstimate <- TRUE
        }
        else{
            other <- ellipsis$alpha
            otherParameterEstimate <- FALSE
        }
        names(other) <- "alpha"
    }
    else if(any(distribution==c("dgnorm","dlgnorm"))){
        if(is.null(ellipsis$shape)){
            other <- 2
            otherParameterEstimate <- TRUE
        }
        else{
            other <- ellipsis$shape
            otherParameterEstimate <- FALSE
        }
        names(other) <- "shape"
    }
    else if(distribution=="dt"){
        if(is.null(ellipsis$nu)){
            other <- 2
            otherParameterEstimate <- TRUE
        }
        else{
            other <- ellipsis$nu
            otherParameterEstimate <- FALSE
        }
        names(other) <- "nu"
    }

    if(is.null(ellipsis$nIterations)){
        nIterations <- 1
        if(any(initialType==c("complete","backcasting","gradient"))){
            nIterations[] <- 2
        }
    }
    else{
        nIterations <- ellipsis$nIterations
    }

    smoother <- if(is.null(ellipsis$smoother)) "default" else ellipsis$smoother
    # smoother="default" resolves to the centred moving average for the optimal
    # initialisation and to the global model for every other initialisation.
    if(smoother=="default"){
        smoother <- if(initialType=="optimal") "ma" else "global"
    }
    FI <- if(is.null(ellipsis$FI)) FALSE else ellipsis$FI
    stepSize <- if(is.null(ellipsis$stepSize)) .Machine$double.eps^(1/4) else ellipsis$stepSize

    return(list(
        maxeval = maxeval,
        maxtime = maxtime,
        xtol_rel = xtol_rel,
        xtol_abs = xtol_abs,
        ftol_rel = ftol_rel,
        ftol_abs = ftol_abs,
        algorithm = algorithm,
        print_level = print_level,
        lb = lb,
        ub = ub,
        B = B,
        lambda = lambda,
        other = other,
        otherParameterEstimate = otherParameterEstimate,
        nIterations = nIterations,
        smoother = smoother,
        FI = FI,
        stepSize = stepSize
    ))
}

#### Degrees of freedom of backcast / complete / gradient initial states ####
# The identifiable count of the initial-state design when the initials are
# obtained by backcasting, complete backcasting or the gradient solve. Those
# initials are determined from the data, so they consume degrees of freedom
# exactly as optimised ones do (see dfInitialsETSLevelSeasonal()). Returns 0
# for initialType="optimal" (initials sit in B and are counted via length(B),
# with the seasonal redundancy handled by the caller) and for "provided".
# Shared by the estimator-free "use" paths of adam()/om() and by omg().
dfInitialsBackcast <- function(etsModel, modelIsSeasonal, modelIsTrendy,
                               lagsModelSeasonal, initialLevelEstimate,
                               initialTrendEstimate, initialSeasonalEstimate,
                               arimaModel, initialArimaNumber, initialArimaEstimate,
                               xregModel, xregNumber, initialXregEstimate,
                               initialType){
    if(!any(initialType==c("backcasting","complete","gradient"))){
        return(0);
    }
    dfInitials <- 0;
    if(etsModel){
        seasonalLagsEstimated <- if(modelIsSeasonal){
            lagsModelSeasonal[as.logical(initialSeasonalEstimate)];
        } else {
            numeric(0);
        }
        dfInitials <- dfInitialsETSLevelSeasonal(seasonalLagsEstimated,
                                                 as.logical(initialLevelEstimate)) +
            modelIsTrendy*initialTrendEstimate;
    }
    if(arimaModel){
        dfInitials <- dfInitials + initialArimaNumber*initialArimaEstimate;
    }
    if(xregModel && any(initialType=="complete")){
        dfInitials <- dfInitials + xregNumber*initialXregEstimate;
    }
    return(dfInitials);
}

# Resolves the user's headLength argument (may be NULL) into:
# - geometry: the head length used for adamProfileCreator() / obsStates (always
#   at least lagsModelMax, clamped to obsInSample);
# - flag: what is assigned to adamCpp$headLength (0L switches head filtering off).
# NULL (absent): geometry=lagsModelMax, flag=lagsModelMax (filtering on, one cycle).
# hl==0: geometry=lagsModelMax, flag=0L (filtering off, legacy behaviour).
# hl>0: geometry=flag=max(lagsModelMax, min(round(hl), obsInSample)).
adam_headLength <- function(hl, lagsModelMax, obsInSample){
    hlRequested <- if(is.null(hl)) lagsModelMax else max(0, round(hl))
    geometry <- max(lagsModelMax, min(hlRequested, obsInSample))
    flag <- if(hlRequested==0) 0L else as.integer(geometry)
    return(list(geometry=geometry, flag=flag))
}

#### Model architecture and initial matrix creation ####
adam_architector <- function(etsModel, Etype, Ttype, Stype, lags, lagsModelSeasonal,
                             xregNumber, obsInSample, initialType,
                             arimaModel, lagsModelARIMA, xregModel, constantRequired,
                             componentsNumberARIMA,
                             obsAll, yIndexAll, yClasses, adamETS,
                             profilesRecentTable=NULL, profilesRecentProvided=FALSE,
                             flipConstant=FALSE, headLength=NULL){
    if(etsModel){
        modelIsTrendy <- Ttype != "N"
        if(modelIsTrendy){
            lagsModel <- matrix(c(1, 1), ncol=1)
            componentsNamesETS <- c("level", "trend")
        }
        else{
            lagsModel <- matrix(c(1), ncol=1)
            componentsNamesETS <- c("level")
        }
        modelIsSeasonal <- Stype != "N"
        if(modelIsSeasonal){
            lagsModel <- matrix(c(lagsModel, lagsModelSeasonal), ncol=1)
            componentsNumberETSSeasonal <- length(lagsModelSeasonal)
            if(componentsNumberETSSeasonal > 1){
                componentsNamesETS <- c(componentsNamesETS,
                                        paste0("seasonal", c(1:componentsNumberETSSeasonal)))
            }
            else{
                componentsNamesETS <- c(componentsNamesETS, "seasonal")
            }
        }
        else{
            componentsNumberETSSeasonal <- 0
        }
        lagsModelAll <- lagsModel
        componentsNumberETS <- length(lagsModel)
    }
    else{
        modelIsTrendy <- modelIsSeasonal <- FALSE
        componentsNumberETS <- componentsNumberETSSeasonal <- 0
        componentsNamesETS <- NULL
        lagsModelAll <- lagsModel <- NULL
    }

    if(arimaModel){
        lagsModelAll <- matrix(c(lagsModel, lagsModelARIMA), ncol=1)
    }

    if(constantRequired){
        lagsModelAll <- matrix(c(lagsModelAll, 1), ncol=1)
    }

    if(xregModel){
        lagsModelAll <- matrix(c(lagsModelAll, rep(1, xregNumber)), ncol=1)
    }

    lagsModelMax <- max(lagsModelAll)
    headLengthResolved <- adam_headLength(headLength, lagsModelMax, obsInSample)
    headLength <- headLengthResolved$geometry
    obsStates <- obsInSample + headLength

    adamProfiles <- adamProfileCreator(lagsModelAll, lagsModelMax, obsAll,
                                       lags=lags, yIndex=yIndexAll, yClasses=yClasses,
                                       headLength=headLength)
    if(profilesRecentProvided){
        profilesRecentTable <- profilesRecentTable[, 1:lagsModelMax, drop=FALSE]
    }
    else{
        profilesRecentTable <- adamProfiles$recent
    }
    indexLookupTable <- adamProfiles$lookup

    componentsNumberETSNonSeasonal <- componentsNumberETS - componentsNumberETSSeasonal
    adamCpp <- new(adamCore,
                   lagsModelAll, Etype, Ttype, Stype,
                   componentsNumberETSNonSeasonal,
                   componentsNumberETSSeasonal,
                   componentsNumberETS, componentsNumberARIMA,
                   xregNumber, length(lagsModelAll),
                   constantRequired, adamETS)
    # Flip the drift sign in the backward pass of backcasting when the total
    # order of ARIMA differencing is odd (time reversal changes the drift by
    # (-1)^(d+D)) — the ARIMA analog of the ETS trend reversal
    adamCpp$flipConstant <- flipConstant
    # Head length for backcasting: one full lag cycle by default, so the head is
    # filtered against the model's own backcasts. 0 switches the filtering off.
    adamCpp$headLength <- headLengthResolved$flag

    return(list(
        lagsModel = lagsModel,
        lagsModelAll = lagsModelAll,
        lagsModelMax = lagsModelMax,
        headLength = headLength,
        componentsNumberETS = componentsNumberETS,
        componentsNumberETSSeasonal = componentsNumberETSSeasonal,
        componentsNumberETSNonSeasonal = componentsNumberETSNonSeasonal,
        componentsNamesETS = componentsNamesETS,
        obsStates = obsStates,
        modelIsTrendy = modelIsTrendy,
        modelIsSeasonal = modelIsSeasonal,
        indexLookupTable = indexLookupTable,
        profilesRecentTable = profilesRecentTable,
        adamCpp = adamCpp
    ))
}

adam_creator <- function(etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal,
                         lags, lagsModel, lagsModelARIMA, lagsModelAll, lagsModelMax,
                         profilesRecentTable=NULL, profilesRecentProvided=FALSE,
                         obsStates, obsInSample, obsAll,
                         componentsNumberETS, componentsNumberETSSeasonal,
                         componentsNamesETS, otLogical, yInSample,
                         persistence, persistenceEstimate,
                         persistenceLevel, persistenceLevelEstimate,
                         persistenceTrend, persistenceTrendEstimate,
                         persistenceSeasonal, persistenceSeasonalEstimate,
                         persistenceXreg, persistenceXregEstimate, persistenceXregProvided,
                         phi,
                         initialType, initialEstimate,
                         initialLevel, initialLevelEstimate, initialTrend, initialTrendEstimate,
                         initialSeasonal, initialSeasonalEstimate,
                         initialArima, initialArimaEstimate, initialArimaNumber,
                         initialXregEstimate, initialXregProvided,
                         arimaModel, arRequired, iRequired, maRequired, armaParameters,
                         arOrders, iOrders, maOrders,
                         componentsNumberARIMA, componentsNamesARIMA,
                         xregModel, xregModelInitials, xregData, xregNumber, xregNames,
                         xregParametersPersistence,
                         constantRequired, constantEstimate, constantValue, constantName,
                         adamCpp,
                         arEstimate, maEstimate, smoother, nonZeroARI, nonZeroMA){

    nComponents <- componentsNumberETS+componentsNumberARIMA+xregNumber+constantRequired
    componentNames <- c(componentsNamesETS, componentsNamesARIMA, xregNames, constantName)

    matVt <- matrix(NA, nComponents, obsStates,
                    dimnames=list(componentNames, NULL))

    matWt <- matrix(1, obsAll, nComponents,
                    dimnames=list(NULL, componentNames))

    if(xregModel){
        matWt[,componentsNumberETS+componentsNumberARIMA+1:xregNumber] <- xregData
    }

    matF <- diag(nComponents)

    vecG <- matrix(0, nComponents, 1,
                   dimnames=list(componentNames, NULL))

    j <- 0
    if(etsModel){
        j <- j+1
        rownames(vecG)[j] <- "alpha"
        if(!persistenceLevelEstimate){
            vecG[j,] <- persistenceLevel
        }
        if(modelIsTrendy){
            j <- j+1
            rownames(vecG)[j] <- "beta"
            if(!persistenceTrendEstimate){
                vecG[j,] <- persistenceTrend
            }
        }
        if(modelIsSeasonal){
            if(!all(persistenceSeasonalEstimate)){
                vecG[j+which(!persistenceSeasonalEstimate),] <- persistenceSeasonal
            }
            if(componentsNumberETSSeasonal>1){
                rownames(vecG)[j+c(1:componentsNumberETSSeasonal)] <-
                    paste0("gamma", c(1:componentsNumberETSSeasonal))
            }
            else{
                rownames(vecG)[j+1] <- "gamma"
            }
            j <- j+componentsNumberETSSeasonal
        }
    }

    if(arimaModel){
        matF[j+1:componentsNumberARIMA,j+1:componentsNumberARIMA] <- 0
        if(componentsNumberARIMA>1){
            rownames(vecG)[j+1:componentsNumberARIMA] <- paste0("psi",c(1:componentsNumberARIMA))
        }
        else{
            rownames(vecG)[j+1:componentsNumberARIMA] <- "psi"
        }
        j <- j+componentsNumberARIMA
    }

    if(!arimaModel && constantRequired){
        matF[1,ncol(matF)] <- 1
    }

    if(xregModel){
        if(persistenceXregProvided && !persistenceXregEstimate){
            vecG[j+1:xregNumber,] <- persistenceXreg
        }
        rownames(vecG)[j+1:xregNumber] <- paste0("delta",xregParametersPersistence)
    }

    if(etsModel && modelIsTrendy){
        matF[1,2] <- phi
        matF[2,2] <- phi
        matWt[,2] <- phi
    }

    if(arimaModel && (!arEstimate && !maEstimate)){
        arimaPolynomials <- lapply(
            adamCpp$polynomialise(0, arOrders, iOrders, maOrders,
                                  arEstimate, maEstimate, armaParameters, lags),
            as.vector)
        if(nrow(nonZeroARI)>0){
            matF[componentsNumberETS+nonZeroARI[,2],componentsNumberETS+nonZeroARI[,2]] <-
                -arimaPolynomials$ariPolynomial[nonZeroARI[,1]]
        }
        if(nrow(nonZeroARI)>0){
            vecG[componentsNumberETS+nonZeroARI[,2]] <-
                -arimaPolynomials$ariPolynomial[nonZeroARI[,1]]
        }
        if(nrow(nonZeroMA)>0){
            vecG[componentsNumberETS+nonZeroMA[,2]] <- vecG[componentsNumberETS+nonZeroMA[,2]] +
                arimaPolynomials$maPolynomial[nonZeroMA[,1]]
        }
    }
    else{
        arimaPolynomials <- NULL
    }

    if(!profilesRecentProvided){
        if(etsModel){
            if(initialEstimate){
                if(modelIsSeasonal){
                    yDecompositionAdditive <- msdecompose(yInSample, lags=lags[lags!=1],
                                                          type="additive",
                                                          smoother=smoother)
                    if(any(c(Etype,Ttype,Stype)=="M")){
                        yDecompositionMultiplicative <- msdecompose(yInSample, lags=lags[lags!=1],
                                                                    type="multiplicative",
                                                                    smoother=smoother)
                    }
                    decompositionType <- c("additive","multiplicative")[any(c(Etype,Stype)=="M")+1]
                    yDecomposition <- switch(decompositionType,
                                             "additive"=yDecompositionAdditive,
                                             "multiplicative"=yDecompositionMultiplicative)
                    j <- 1
                    if(initialLevelEstimate){
                        if(modelIsTrendy){
                            matVt[j,1:lagsModelMax] <- switch(Ttype,
                                                              "M"=yDecompositionMultiplicative$initial$nonseasonal[1],
                                                              yDecompositionAdditive$initial$nonseasonal[1])
                        }
                        else{
                            matVt[j,1:lagsModelMax] <- mean(yInSample[otLogical])
                        }
                        if(xregModel){
                            if(Etype=="A"){
                                matVt[j,1:lagsModelMax] <- matVt[j,1:lagsModelMax] -
                                    as.vector(xregModelInitials[[1]]$initialXreg %*% xregData[1,])
                            }
                            else{
                                matVt[j,1:lagsModelMax] <- matVt[j,1:lagsModelMax] /
                                    as.vector(exp(xregModelInitials[[2]]$initialXreg %*%
                                                      xregData[1,]))
                            }
                        }
                    }
                    else{
                        matVt[j,1:lagsModelMax] <- initialLevel
                    }
                    j <- j+1
                    if(modelIsTrendy){
                        if(initialTrendEstimate){
                            if(Ttype=="A"){
                                matVt[j,1:lagsModelMax] <-
                                    yDecompositionAdditive$initial$nonseasonal[2]
                                if(Stype=="M"){
                                    if(matVt[j,1]<0 &&
                                       abs(matVt[j,1])>min(abs(yInSample[otLogical]))){
                                        matVt[j,1:lagsModelMax] <- 0
                                    }
                                }
                            }
                            else if(Ttype=="M"){
                                matVt[j,1:lagsModelMax] <-
                                    yDecompositionMultiplicative$initial$nonseasonal[2]
                                if(any(matVt[1,1:lagsModelMax]<0)){
                                    matVt[1,1:lagsModelMax] <- yInSample[otLogical][1]
                                }
                            }
                        }
                        else{
                            matVt[j,1:lagsModelMax] <- initialTrend
                        }
                        j <- j+1
                    }
                    if(all(c(Etype,Stype)=="A") || all(c(Etype,Stype)=="M") ||
                       (Etype=="A" & Stype=="M")){
                        for(i in 1:componentsNumberETSSeasonal){
                            if(initialSeasonalEstimate[i]){
                                matVt[i+j-1,1:lagsModel[i+j-1]] <-
                                    yDecomposition$initial$seasonal[[i]]
                                if(Stype=="A"){
                                    matVt[i+j-1,1:lagsModel[i+j-1]] <-
                                        matVt[i+j-1,1:lagsModel[i+j-1]] -
                                        mean(matVt[i+j-1,1:lagsModel[i+j-1]])
                                }
                                else{
                                    matVt[i+j-1,1:lagsModel[i+j-1]] <-
                                        matVt[i+j-1,1:lagsModel[i+j-1]] /
                                        exp(mean(log(matVt[i+j-1,1:lagsModel[i+j-1]])))
                                }
                            }
                            else{
                                matVt[i+j-1,1:lagsModel[i+j-1]] <- initialSeasonal[[i]]
                            }
                        }
                    }
                    else if(Etype=="M" && Stype=="A"){
                        for(i in 1:componentsNumberETSSeasonal){
                            if(initialSeasonalEstimate[i]){
                                matVt[i+j-1,1:lagsModel[i+j-1]] <-
                                    log(yDecomposition$initial$seasonal[[i]]) *
                                    min(yInSample[otLogical])
                                if(Stype=="A"){
                                    matVt[i+j-1,1:lagsModel[i+j-1]] <-
                                        matVt[i+j-1,1:lagsModel[i+j-1]] -
                                        mean(matVt[i+j-1,1:lagsModel[i+j-1]])
                                }
                                else{
                                    matVt[i+j-1,1:lagsModel[i+j-1]] <-
                                        matVt[i+j-1,1:lagsModel[i+j-1]] /
                                        exp(mean(log(matVt[i+j-1,1:lagsModel[i+j-1]])))
                                }
                            }
                            else{
                                matVt[i+j-1,1:lagsModel[i+j-1]] <- initialSeasonal[[i]]
                            }
                        }
                    }
                    if(Etype=="M" && matVt[1,1]<=0){
                        matVt[1,1:lagsModelMax] <- yInSample[1]
                    }
                }
                else{
                    yDecompositionAdditive <- msdecompose(yInSample, lags=1,
                                                          type="additive",
                                                          smoother=smoother)
                    if(any(c(Etype,Ttype)=="M")){
                        yDecompositionMultiplicative <- msdecompose(yInSample, lags=1,
                                                                    type="multiplicative",
                                                                    smoother=smoother)
                    }
                    if(initialLevelEstimate){
                        if(modelIsTrendy){
                            matVt[1,1:lagsModelMax] <- switch(Ttype,
                                                              "M"=yDecompositionMultiplicative$initial$nonseasonal[1],
                                                              yDecompositionAdditive$initial$nonseasonal[1])
                        }
                        else{
                            matVt[1,1:lagsModelMax] <- mean(yInSample[otLogical])
                        }
                    }
                    else{
                        matVt[1,1:lagsModelMax] <- initialLevel
                    }
                    if(modelIsTrendy){
                        if(initialTrendEstimate){
                            matVt[2,1:lagsModelMax] <- switch(Ttype,
                                                              "A"=yDecompositionAdditive$initial$nonseasonal[2],
                                                              "M"=yDecompositionMultiplicative$initial$nonseasonal[2])
                        }
                        else{
                            matVt[2,1:lagsModelMax] <- initialTrend
                        }
                    }
                    if(Etype=="M" && matVt[1,1]<=0){
                        matVt[1,1:lagsModelMax] <- yInSample[1]
                    }
                }
                if(initialLevelEstimate && Etype=="M" && matVt[1,lagsModelMax]==0){
                    matVt[1,1:lagsModelMax] <- mean(yInSample)
                }
            }
            else if(!initialEstimate && initialType=="provided"){
                j <- 1
                matVt[j,1:lagsModelMax] <- initialLevel
                if(modelIsTrendy){
                    j <- j+1
                    matVt[j,1:lagsModelMax] <- initialTrend
                }
                if(modelIsSeasonal){
                    for(i in 1:componentsNumberETSSeasonal){
                        matVt[j+i,1:lagsModel[j+i]] <- initialSeasonal[[i]]
                    }
                }
                j <- j+componentsNumberETSSeasonal
            }
        }

        # ARIMA states start from zero errors and zero deviations. The initials are
        # set after the constant, which they may depend on
        if(arimaModel){
            matVt[componentsNumberETS+1:componentsNumberARIMA, 1:lagsModelMax] <- switch(Etype, "A"=0, "M"=1)
        }

        if(xregModel){
            if(Etype=="A" || initialXregProvided || is.null(xregModelInitials[[2]])){
                matVt[componentsNumberETS+componentsNumberARIMA+1:xregNumber, 1:lagsModelMax] <- 0
                matVt[names(xregModelInitials[[1]]$initialXreg), 1:lagsModelMax] <-
                    xregModelInitials[[1]]$initialXreg
            }
            else{
                matVt[componentsNumberETS+componentsNumberARIMA+1:xregNumber, 1:lagsModelMax] <- 0
                matVt[names(xregModelInitials[[2]]$initialXreg), 1:lagsModelMax] <-
                    xregModelInitials[[2]]$initialXreg
            }
        }

        if(constantRequired){
            if(constantEstimate){
                if(sum(iOrders)==0 && !etsModel){
                    matVt[componentsNumberETS+componentsNumberARIMA+xregNumber+1,] <-
                        mean(yInSample[otLogical])
                }
                else{
                    driftSeries <- adam_driftSeries(yInSample[otLogical], Etype, etsModel, lags, iOrders)
                    matVt[componentsNumberETS+componentsNumberARIMA+xregNumber+1,] <-
                        switch(Etype, "A"=mean(driftSeries), "M"=exp(mean(driftSeries)))
                }
            }
            else{
                matVt[componentsNumberETS+componentsNumberARIMA+xregNumber+1,] <- constantValue
            }
            if(etsModel && initialLevelEstimate){
                if(Etype=="A"){
                    matVt[1,1:lagsModelMax] <- matVt[1,1:lagsModelMax] -
                        matVt[componentsNumberETS+componentsNumberARIMA+xregNumber+1,1]
                }
                else{
                    matVt[1,1:lagsModelMax] <- matVt[1,1:lagsModelMax] /
                        matVt[componentsNumberETS+componentsNumberARIMA+xregNumber+1,1]
                }
            }
        }

        # ARIMA initials. The provided ones (see adam_arimaInitials) are used as they
        # are. The estimated ones start from the pre-sample values of the series,
        # which the initialiser (optimal) or the filler (backcasting) turn into the
        # initials with the ARI polynomial. They stay neutral for ETS+ARIMA, where
        # ETS carries the level and seasonality.
        if(arimaModel){
            initialRow <- componentsNumberETS+componentsNumberARIMA
            if(!initialArimaEstimate){
                matVt[initialRow, 1:initialArimaNumber] <- initialArima[1:initialArimaNumber]
            }
            else if(!etsModel){
                constantLevel <- if(constantRequired){
                    matVt[componentsNumberETS+componentsNumberARIMA+xregNumber+1,1]}
                matVt[initialRow, 1:initialArimaNumber] <-
                    adam_arimaPreSample(yInSample, otLogical, Etype, lags, iOrders,
                                        initialArimaNumber, constantLevel, smoother)
            }
        }
    }
    else{
        matVt[,1:lagsModelMax] <- profilesRecentTable
    }

    return(list(matVt=matVt, matWt=matWt, matF=matF, vecG=vecG, arimaPolynomials=arimaPolynomials))
}

adam_filler <- function(B,
                        etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal,
                        componentsNumberETS, componentsNumberETSNonSeasonal,
                        componentsNumberETSSeasonal, componentsNumberARIMA,
                        lags, lagsModel, lagsModelMax,
                        matVt, matWt, matF, vecG,
                        persistenceEstimate, persistenceLevelEstimate, persistenceTrendEstimate,
                        persistenceSeasonalEstimate, persistenceXregEstimate,
                        phiEstimate,
                        initialType, initialEstimate,
                        initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                        initialArimaEstimate, initialXregEstimate,
                        arimaModel, arEstimate, maEstimate, arOrders, iOrders, maOrders,
                        arRequired, maRequired, armaParameters,
                        nonZeroARI, nonZeroMA, arimaPolynomials,
                        xregModel, xregNumber,
                        xregParametersMissing, xregParametersIncluded,
                        xregParametersEstimated, xregParametersPersistence,
                        constantEstimate,
                        adamCpp,
                        constantRequired, initialArimaNumber){

    j <- 0
    # Fill in persistence
    if(persistenceEstimate){
        # Persistence of ETS
        if(etsModel){
            i <- 1
            # alpha
            if(persistenceLevelEstimate){
                j[] <- j+1
                vecG[i] <- B[j]
            }
            # beta
            if(modelIsTrendy){
                i[] <- 2
                if(persistenceTrendEstimate){
                    j[] <- j+1
                    vecG[i] <- B[j]
                }
            }
            # gamma1, gamma2, ...
            if(modelIsSeasonal){
                if(any(persistenceSeasonalEstimate)){
                    vecG[i+which(persistenceSeasonalEstimate)] <-
                        B[j+c(1:sum(persistenceSeasonalEstimate))]
                    j[] <- j+sum(persistenceSeasonalEstimate)
                }
                i[] <- componentsNumberETS
            }
        }

        # Persistence of xreg
        if(xregModel && persistenceXregEstimate){
            xregPersistenceNumber <- max(xregParametersPersistence)
            vecG[componentsNumberETS+componentsNumberARIMA+1:length(xregParametersPersistence)] <-
                B[j+1:xregPersistenceNumber][xregParametersPersistence]
            j[] <- j+xregPersistenceNumber
        }
    }

    # Damping parameter
    if(etsModel && phiEstimate){
        j[] <- j+1
        matWt[,2] <- B[j]
        matF[1:2,2] <- B[j]
    }

    # ARMA parameters. This goes before xreg in persistence
    if(arimaModel){
        # Call the function returning ARI and MA polynomials
        arimaPolynomials <- lapply(
            adamCpp$polynomialise(B[j+1:sum(c(arOrders*arEstimate,maOrders*maEstimate))],
                                  arOrders, iOrders, maOrders,
                                  arEstimate, maEstimate, armaParameters, lags),
            as.vector)

        # Fill in the transition matrix
        if(nrow(nonZeroARI)>0){
            matF[componentsNumberETS+nonZeroARI[,2],
                 componentsNumberETS+1:(componentsNumberARIMA+constantRequired)] <-
                -arimaPolynomials$ariPolynomial[nonZeroARI[,1]]
        }
        # Fill in the persistence vector
        if(nrow(nonZeroARI)>0){
            vecG[componentsNumberETS+nonZeroARI[,2]] <-
                -arimaPolynomials$ariPolynomial[nonZeroARI[,1]]
        }
        if(nrow(nonZeroMA)>0){
            vecG[componentsNumberETS+nonZeroMA[,2]] <- vecG[componentsNumberETS+nonZeroMA[,2]] +
                arimaPolynomials$maPolynomial[nonZeroMA[,1]]
        }
        j[] <- j+sum(c(arOrders*arEstimate,maOrders*maEstimate))
    }

    # Initials of ETS if something needs to be estimated
    if(etsModel && all(initialType!=c("complete","backcasting","gradient")) && initialEstimate){
        i <- 1
        if(initialLevelEstimate){
            j[] <- j+1
            matVt[i,1:lagsModelMax] <- B[j]
        }
        i[] <- i+1
        if(modelIsTrendy && initialTrendEstimate){
            j[] <- j+1
            matVt[i,1:lagsModelMax] <- B[j]
            i[] <- i+1
        }
        if(modelIsSeasonal && any(initialSeasonalEstimate)){
            for(k in 1:componentsNumberETSSeasonal){
                if(initialSeasonalEstimate[k]){
                    matVt[componentsNumberETSNonSeasonal+k,
                          2:lagsModel[componentsNumberETSNonSeasonal+k]-1] <-
                        B[j+2:(lagsModel[componentsNumberETSNonSeasonal+k])-1]
                    matVt[componentsNumberETSNonSeasonal+k,
                          lagsModel[componentsNumberETSNonSeasonal+k]] <-
                        switch(Stype,
                               "A"=-sum(B[j+2:(lagsModel[componentsNumberETSNonSeasonal+k])-1]),
                               "M"=1/prod(B[j+2:(lagsModel[componentsNumberETSNonSeasonal+k])-1]))
                    j[] <- j+lagsModel[componentsNumberETSNonSeasonal+k]-1
                }
            }
        }
    }

    # Initials of ARIMA, held by the state with the largest lag (see adam_arimaInitials):
    # the estimated ones, or, for the backcasting, the pre-sample values stored by the
    # creator taken through the current ARI polynomial
    if(arimaModel && initialArimaEstimate){
        if(all(initialType!=c("complete","backcasting","gradient"))){
            matVt[componentsNumberETS+componentsNumberARIMA, 1:initialArimaNumber] <- B[j+1:initialArimaNumber]
            j[] <- j+initialArimaNumber
        }
        else{
            matVt[componentsNumberETS+componentsNumberARIMA, 1:initialArimaNumber] <-
                adam_arimaInitials(arimaPolynomials$ariPolynomial,
                                   matVt[componentsNumberETS+componentsNumberARIMA, 1:initialArimaNumber], Etype)
        }
    }

    # Initials of the xreg. Kept in B for every initial type except "complete"
    # (backcast). Under "gradient" the affine / GN initial-state solve overwrites
    # the xreg cells of matVt, so the B entry is a no-op there (mirrors how the
    # occurrence path handles a gradient-solved xreg); it still counts once
    # towards the parameter total via length(B).
    if(xregModel && (initialType!="complete") && initialEstimate && initialXregEstimate){
        xregNumberToEstimate <- sum(xregParametersEstimated)
        matVt[componentsNumberETS+componentsNumberARIMA+which(xregParametersEstimated==1),
              1:lagsModelMax] <- B[j+1:xregNumberToEstimate]
        j[] <- j+xregNumberToEstimate
    }

    # Constant
    if(constantEstimate){
        matVt[componentsNumberETS+componentsNumberARIMA+xregNumber+1,] <- B[j+1]
    }

    return(list(matVt=matVt, matWt=matWt, matF=matF, vecG=vecG, arimaPolynomials=arimaPolynomials))
}

# The ARIMA initials. In the profile table, the column k of the ARIMA state with
# lag L is its value at time k-L, so the fitted value at time k (k <= m, the
# largest ARIMA lag) receives the column k of all the states with lags L >= k
# and nothing else from the head.
# Only these m sums are identified, which makes the ARIMA initials the same as
# the initial state of ssarima's companion form. They are held by the state
# with the largest lag, the last one (lagsModelARIMA is sorted), the other heads
# being zero (one for "M", where the sums are products).
# With zero errors before the sample, the ARI state with lag L is ari_L times
# the pre-sample value x in the convention of adam_arimaPreSample (-y, 1/y for
# "M"; oldest first, x[m] at time 0), so the initial k is the sum of
# ari_L * x[m-L+k] over the ARI lags L >= k.
adam_arimaInitials <- function(ariPolynomial, x, Etype){
    m <- length(x)
    x <- switch(Etype, "M"=log(x), x)
    initials <- numeric(m)
    for(lag in which(ariPolynomial[-1]!=0)){
        initials[1:lag] <- initials[1:lag] + ariPolynomial[lag+1] * x[m-lag+1:lag]
    }
    return(switch(Etype, "M"=exp(initials), initials))
}

# The ARIMA initials implied by the heads of the fitted ARIMA states (the sums
# above), for the states estimated in any way, e.g. by backcasting. The fitted
# states are aligned in time: the head column c is the time c-lagsModelMax, so
# the state with lag L gives to the time k its column lagsModelMax-L+k.
adam_arimaHeadInitials <- function(matVtARIMA, lagsModelARIMA, lagsModelMax, Etype){
    head <- switch(Etype, "M"=log(matVtARIMA[,1:lagsModelMax,drop=FALSE]),
                   matVtARIMA[,1:lagsModelMax,drop=FALSE])
    initials <- vapply(1:max(lagsModelARIMA), function(k){
        rows <- which(lagsModelARIMA>=k)
        return(sum(head[cbind(rows, lagsModelMax-lagsModelARIMA[rows]+k)]))
    }, numeric(1))
    return(switch(Etype, "M"=exp(initials), initials))
}

# The series on the scale of the constant, from which its starting value and bounds
# come: differenced as the ARIMA part of the model differences it (in logs for a
# multiplicative error), whose drift it is, or by one step for ETS, where it is the
# drift of the level. With seasonal differencing, the drift is the change over a
# season, not over one step.
adam_driftSeries <- function(y, Etype, etsModel, lags, iOrders){
    if(Etype=="M"){
        y <- log(y)
    }
    if(etsModel || all(iOrders==0)){
        return(diff(y))
    }
    for(i in seq_along(iOrders)){
        if(iOrders[i]>0){
            y <- diff(y, lag=lags[i], differences=iOrders[i])
        }
    }
    return(y)
}

# Pre-sample values of the series for the ARIMA initials: the decomposition that
# gives the ETS initials (the same smoother), extended m observations backwards
# (the level, the trend if the model has differences, and the seasonal patterns)
# on the scale of the ARIMA part. msdecompose() already extrapolates its initial
# level to the time 1-lagsMax, over any gap of the smoother. With no differences,
# the constant is the level, and the values are deviations from it.
# Returned as -y (1/y for "M"), oldest first (see adam_arimaInitials).
adam_arimaPreSample <- function(yInSample, otLogical, Etype, lags, iOrders, m, constantLevel, smoother){
    y <- as.vector(yInSample)
    y[!otLogical] <- NA
    if(Etype=="M"){
        y <- suppressWarnings(log(y))
        y[!is.finite(y)] <- NA
    }
    seasonalLags <- lags[lags>1 & lags*2<sum(!is.na(y))]
    decompositionLags <- if(length(seasonalLags)>0) seasonalLags else 1
    yDecomposition <- msdecompose(y, lags=decompositionLags, smoother=smoother)
    slope <- if(any(iOrders>0)) yDecomposition$initial$nonseasonal["trend"] else 0
    times <- (1-m):0
    # The level at the time 1, then the slope from there
    yPre <- yDecomposition$initial$nonseasonal["level"] +
        yDecomposition$initial$nonseasonal["trend"]*max(decompositionLags) + slope*(times-1)
    for(i in seq_along(seasonalLags)){
        yPre <- yPre + yDecomposition$seasonal[[i]][((times-1) %% seasonalLags[i]) + 1]
    }
    if(!is.null(constantLevel) && all(iOrders==0)){
        yPre <- yPre - switch(Etype, "M"=log(constantLevel), constantLevel)
    }
    return(as.vector(switch(Etype, "M"=exp(-yPre), -yPre)))
}

# Starting values of the AR / MA parameters via the Hannan-Rissanen method
# (src/headers/arimaInitCore.h). The series is the in-sample data on the scale of
# the ARIMA part (logs for multiplicative error), with the ETS part approximated
# by the decomposition that gives the ETS initials (the same smoother) and the
# regressors (xregInSample, a matrix or NULL) by OLS on what is left, differenced
# as the model requires. The seasonal
# ARIMA factors that coincide with the ETS seasonality keep the defaults.
# With bounded, the factors that the cost function would reject (not stationary AR,
# not invertible MA, see src/headers/arimaBounds.h) are moved inside the boundary.
# Returns the AR / MA values in the order of B.
adam_arimaInitialiser <- function(yInSample, otLogical, etsModel, Etype, Stype, modelIsSeasonal,
                                  lags, arOrders, iOrders, maOrders, arEstimate, maEstimate,
                                  armaParameters, bounded, smoother, xregInSample){
    # Missing and zero (intermittent) values are treated as NAs, the gaps
    y <- as.vector(yInSample)
    y[!otLogical] <- NA
    if(etsModel){
        yDecomposition <- msdecompose(y, lags=if(modelIsSeasonal) lags[lags!=1] else 1,
                                      type=c("additive","multiplicative")[any(c(Etype,Stype)=="M")+1],
                                      smoother=smoother)
        y <- switch(Etype, "M"=log(y) - log(yDecomposition$fitted), y - yDecomposition$fitted)
    }
    else if(Etype=="M"){
        y <- log(y)
    }
    # The series and the regressors, differenced alike: a difference that touches a
    # gap is a gap too
    yDiffs <- cbind(as.vector(y), xregInSample)
    for(i in which(iOrders>0)){
        yDiffs <- diff(yDiffs, lag=lags[i], differences=iOrders[i])
    }
    y <- yDiffs[,1]
    observed <- is.finite(y)
    # Regression on the differences: in levels, an integrated error makes it spurious
    if(!is.null(xregInSample)){
        xregDiffs <- cbind(1, yDiffs[,-1,drop=FALSE])
        y <- as.vector(y - xregDiffs %*% olsCpp(xregDiffs[observed,,drop=FALSE], y[observed]))
    }
    # The gaps are zeros of the centred series (Hannan-Rissanen centres it), which keeps
    # the lags aligned without inventing differences across them
    y[!observed] <- mean(y[observed])

    useLevel <- !(etsModel & modelIsSeasonal & lags>1)
    return(as.vector(arimaHRCpp(y, arOrders, maOrders, lags, arEstimate, maEstimate,
                                if(is.null(armaParameters)) numeric(0) else armaParameters,
                                useLevel, bounded)))
}

adam_initialiser <- function(etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal,
                             componentsNumberETSNonSeasonal, componentsNumberETSSeasonal,
                             componentsNumberETS,
                             lags, lagsModel, lagsModelSeasonal, lagsModelARIMA, lagsModelMax,
                             matVt,
                             persistenceEstimate, persistenceLevelEstimate,
                             persistenceTrendEstimate,
                             persistenceSeasonalEstimate, persistenceXregEstimate,
                             phiEstimate, initialType, initialEstimate,
                             initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                             initialArimaEstimate, initialXregEstimate,
                             arimaModel, arRequired, maRequired, arEstimate, maEstimate,
                             arOrders, maOrders,
                             componentsNumberARIMA, componentsNamesARIMA, initialArimaNumber,
                             xregModel, xregNumber,
                             xregParametersEstimated, xregParametersPersistence,
                             constantEstimate, constantName, otherParameterEstimate,
                             adamCpp,
                             ets, bounds, yInSample, otLogical, iOrders, armaParameters, other,
                             smoother, matWt){
    # The vector of logicals for persistence elements
    persistenceEstimateVector <- c(persistenceLevelEstimate,
                                   modelIsTrendy&persistenceTrendEstimate,
                                   modelIsSeasonal&persistenceSeasonalEstimate)

    # The order:
    # Persistence of states and for xreg, phi, AR and MA parameters,
    # initials, initialsARIMA, initials for xreg
    B <- Bl <- Bu <- vector("numeric",
                            # Values of the persistence vector + phi
                            etsModel*(persistenceLevelEstimate +
                                          modelIsTrendy*persistenceTrendEstimate +
                                          modelIsSeasonal*sum(persistenceSeasonalEstimate) +
                                          phiEstimate) +
                                xregModel*persistenceXregEstimate*max(xregParametersPersistence) +
                                # AR and MA values
                                arimaModel*(arEstimate*sum(arOrders)+maEstimate*sum(maOrders)) +
                                # initials of ETS
                                etsModel*all(initialType!=c("complete","backcasting","gradient"))*
                                (initialLevelEstimate +
                                     (modelIsTrendy*initialTrendEstimate) +
                                     (modelIsSeasonal*
                                          sum(initialSeasonalEstimate*(lagsModelSeasonal-1)))) +
                                # initials of ARIMA
                                all(initialType!=c("complete","backcasting","gradient"))*
                                arimaModel*initialArimaNumber*initialArimaEstimate +
                                # initials of xreg (in B unless backcast under "complete")
                                (initialType!="complete")*xregModel*initialXregEstimate*
                                sum(xregParametersEstimated) +
                                constantEstimate + otherParameterEstimate)

    j <- 0
    if(etsModel){
        # Fill in persistence
        if(persistenceEstimate && any(persistenceEstimateVector)){
            if(ets=="conventional" && any(c(Etype,Ttype,Stype)=="M")){
                # A special type of model which is not safe: AAM, MAA, MAM
                if((Etype=="A" && Ttype=="A" && Stype=="M") ||
                   (Etype=="A" && Ttype=="M" && Stype=="A") ||
                   (any(initialType==c("complete","backcasting","gradient")) &&
                    ((Etype=="M" && Ttype=="A" && Stype=="A") ||
                     (Etype=="M" && Ttype=="A" && Stype=="M")))){
                    B[1:sum(persistenceEstimateVector)] <-
                        c(0.01,0.005,rep(0.001,componentsNumberETSSeasonal))[
                            which(persistenceEstimateVector)]
                }
                # MMA is the worst. Set everything to zero and see if anything can be done...
                else if((Etype=="M" && Ttype=="M" && Stype=="A")){
                    B[1:sum(persistenceEstimateVector)] <-
                        c(0.01,0.005,rep(0.01,componentsNumberETSSeasonal))[
                            which(persistenceEstimateVector)]
                }
                else if(Etype=="M" && Ttype=="A"){
                    if(any(initialType==c("complete","backcasting","gradient"))){
                        B[1:sum(persistenceEstimateVector)] <-
                            c(0.1,0.05,rep(0.3,componentsNumberETSSeasonal))[
                                which(persistenceEstimateVector)]
                    }
                    else{
                        B[1:sum(persistenceEstimateVector)] <-
                            c(0.2,0.01,rep(0.3,componentsNumberETSSeasonal))[
                                which(persistenceEstimateVector)]
                    }
                }
                else if(Etype=="M" && Ttype=="M"){
                    B[1:sum(persistenceEstimateVector)] <-
                        c(0.1,0.05,rep(0.3,componentsNumberETSSeasonal))[
                            which(persistenceEstimateVector)]
                }
                else{
                    B[1:sum(persistenceEstimateVector)] <-
                        c(0.1,0.05,rep(0.3,componentsNumberETSSeasonal))[
                            which(persistenceEstimateVector)]
                }
            }
            else{
                B[1:sum(persistenceEstimateVector)] <-
                    c(0.1, 0.05, rep(0.3, componentsNumberETSSeasonal))[
                        which(persistenceEstimateVector)]
            }
            if(bounds=="usual"){
                Bl[1:sum(persistenceEstimateVector)] <- rep(0, sum(persistenceEstimateVector))
                Bu[1:sum(persistenceEstimateVector)] <- rep(1, sum(persistenceEstimateVector))
            }
            else{
                Bl[1:sum(persistenceEstimateVector)] <- rep(-5, sum(persistenceEstimateVector))
                Bu[1:sum(persistenceEstimateVector)] <- rep(5, sum(persistenceEstimateVector))
            }
            # Names for B
            if(persistenceLevelEstimate){
                j[] <- j+1
                names(B)[j] <- "alpha"
            }
            if(modelIsTrendy && persistenceTrendEstimate){
                j[] <- j+1
                names(B)[j] <- "beta"
            }
            if(modelIsSeasonal && any(persistenceSeasonalEstimate)){
                if(componentsNumberETSSeasonal>1){
                    names(B)[j+c(1:sum(persistenceSeasonalEstimate))] <-
                        paste0("gamma",c(1:componentsNumberETSSeasonal))
                }
                else{
                    names(B)[j+1] <- "gamma"
                }
                j[] <- j+sum(persistenceSeasonalEstimate)
            }
        }
    }

    # Persistence if xreg is provided
    if(xregModel && persistenceXregEstimate){
        xregPersistenceNumber <- max(xregParametersPersistence)
        B[j+1:xregPersistenceNumber] <- rep(switch(Etype,"A"=0.01,"M"=0),xregPersistenceNumber)
        Bl[j+1:xregPersistenceNumber] <- rep(-5, xregPersistenceNumber)
        Bu[j+1:xregPersistenceNumber] <- rep(5, xregPersistenceNumber)
        names(B)[j+1:xregPersistenceNumber] <- paste0("delta",c(1:xregPersistenceNumber))
        j[] <- j+xregPersistenceNumber
    }

    # Damping parameter
    if(etsModel && phiEstimate){
        j[] <- j+1
        B[j] <- 0.95
        names(B)[j] <- "phi"
        Bl[j] <- 0
        Bu[j] <- 1
    }

    # ARIMA parameters (AR / MA)
    if(arimaModel){
        # This index is needed to get the correct polynomials
        k <- j
        # These are filled in lags-wise
        if(any(c(arEstimate,maEstimate))){
            armaValues <- adam_arimaInitialiser(yInSample, otLogical, etsModel, Etype, Stype,
                                                modelIsSeasonal, lags, arOrders, iOrders, maOrders,
                                                arEstimate, maEstimate, armaParameters,
                                                bounds!="none", smoother,
                                                if(xregModel){matWt[seq_along(yInSample),
                                                                    componentsNumberETS+componentsNumberARIMA+
                                                                        1:xregNumber, drop=FALSE]})
            for(i in 1:length(lags)){
                if(arRequired && arEstimate && arOrders[i]>0){
                    B[j+c(1:arOrders[i])] <- armaValues[j-k+c(1:arOrders[i])]
                    Bl[j+c(1:arOrders[i])] <- -5
                    Bu[j+c(1:arOrders[i])] <- 5
                    names(B)[j+1:arOrders[i]] <- paste0("phi",1:arOrders[i],"[",lags[i],"]")
                    j[] <- j + arOrders[i]
                }
                if(maRequired && maEstimate && maOrders[i]>0){
                    B[j+c(1:maOrders[i])] <- armaValues[j-k+c(1:maOrders[i])]
                    Bl[j+c(1:maOrders[i])] <- -5
                    Bu[j+c(1:maOrders[i])] <- 5
                    names(B)[j+1:maOrders[i]] <- paste0("theta",1:maOrders[i],"[",lags[i],"]")
                    j[] <- j + maOrders[i]
                }
            }
        }

        arimaPolynomials <- lapply(
            adamCpp$polynomialise(B[k+1:sum(c(arOrders*arEstimate,maOrders*maEstimate))],
                                  arOrders, iOrders, maOrders,
                                  arEstimate, maEstimate, armaParameters, lags),
            as.vector)
    }

    # Initials
    if(etsModel && all(initialType!=c("complete","backcasting","gradient")) && initialEstimate){
        if(initialLevelEstimate){
            j[] <- j+1
            B[j] <- matVt[1,1]
            names(B)[j] <- "level"
            if(Etype=="A"){
                Bl[j] <- -Inf
                Bu[j] <- Inf
            }
            else{
                Bl[j] <- 0
                Bu[j] <- Inf
            }
        }
        if(modelIsTrendy && initialTrendEstimate){
            j[] <- j+1
            B[j] <- matVt[2,1]
            names(B)[j] <- "trend"
            if(Ttype=="A"){
                Bl[j] <- -Inf
                Bu[j] <- Inf
            }
            else{
                Bl[j] <- 0
                Bu[j] <- 2
            }
        }
        if(modelIsSeasonal && any(initialSeasonalEstimate)){
            if(componentsNumberETSSeasonal>1){
                for(k in 1:componentsNumberETSSeasonal){
                    if(initialSeasonalEstimate[k]){
                        B[j+2:lagsModel[componentsNumberETSNonSeasonal+k]-1] <-
                            matVt[componentsNumberETSNonSeasonal+k,
                                  2:lagsModel[componentsNumberETSNonSeasonal+k]-1]
                        names(B)[j+2:(lagsModel[componentsNumberETSNonSeasonal+k])-1] <-
                            paste0("seasonal",k,"_",
                                   2:lagsModel[componentsNumberETSNonSeasonal+k]-1)
                        if(Stype=="A"){
                            Bl[j+2:lagsModel[componentsNumberETSNonSeasonal+k]-1] <- -Inf
                            Bu[j+2:lagsModel[componentsNumberETSNonSeasonal+k]-1] <- Inf
                        }
                        else{
                            Bl[j+2:lagsModel[componentsNumberETSNonSeasonal+k]-1] <- 0
                            Bu[j+2:lagsModel[componentsNumberETSNonSeasonal+k]-1] <- Inf
                        }
                        j[] <- j+(lagsModelSeasonal[k]-1)
                    }
                }
            }
            else{
                B[j+2:(lagsModel[componentsNumberETS])-1] <-
                    matVt[componentsNumberETS,2:lagsModel[componentsNumberETS]-1]
                names(B)[j+2:(lagsModel[componentsNumberETS])-1] <-
                    paste0("seasonal_",2:lagsModel[componentsNumberETS]-1)
                if(Stype=="A"){
                    Bl[j+2:(lagsModel[componentsNumberETS])-1] <- -Inf
                    Bu[j+2:(lagsModel[componentsNumberETS])-1] <- Inf
                }
                else{
                    Bl[j+2:(lagsModel[componentsNumberETS])-1] <- 0
                    Bu[j+2:(lagsModel[componentsNumberETS])-1] <- Inf
                }
                j[] <- j+(lagsModel[componentsNumberETS]-1)
            }
        }
    }

    # ARIMA initials
    if(arimaModel && all(initialType!=c("complete","backcasting","gradient")) && initialArimaEstimate){
        # The creator holds the pre-sample values in the last ARIMA state (see
        # adam_arimaInitials), taken through the ARI polynomial of the starting values
        B[j+1:initialArimaNumber] <-
            adam_arimaInitials(arimaPolynomials$ariPolynomial,
                               matVt[componentsNumberETS+componentsNumberARIMA, 1:initialArimaNumber], Etype)
        names(B)[j+1:initialArimaNumber] <- paste0("ARIMAState",1:initialArimaNumber)

        if(Etype=="A"){
            Bl[j+1:initialArimaNumber] <- -Inf
            Bu[j+1:initialArimaNumber] <- Inf
        }
        else{
            # Make sure that ARIMA states are positive to avoid errors
            B[j+1:initialArimaNumber] <- abs(B[j+1:initialArimaNumber])
            Bl[j+1:initialArimaNumber] <- 0
            Bu[j+1:initialArimaNumber] <- Inf
        }
        j[] <- j+initialArimaNumber
    }

    # Initials of the xreg (excluded from B only under "complete")
    if((initialType!="complete") && initialXregEstimate){
        xregNumberToEstimate <- sum(xregParametersEstimated)
        B[j+1:xregNumberToEstimate] <-
            matVt[componentsNumberETS+componentsNumberARIMA+
                      which(xregParametersEstimated==1),1]
        names(B)[j+1:xregNumberToEstimate] <-
            rownames(matVt)[componentsNumberETS+componentsNumberARIMA+
                                which(xregParametersEstimated==1)]
        if(Etype=="A"){
            Bl[j+1:xregNumberToEstimate] <- -Inf
            Bu[j+1:xregNumberToEstimate] <- Inf
        }
        else{
            Bl[j+1:xregNumberToEstimate] <- -Inf
            Bu[j+1:xregNumberToEstimate] <- Inf
        }
        j[] <- j+xregNumberToEstimate
    }

    if(constantEstimate){
        j[] <- j+1
        B[j] <- matVt[componentsNumberETS+componentsNumberARIMA+xregNumber+1,1]
        # The constant is the intercept of ARIMA, so it needs to agree with the AR starting values
        if(arimaModel && !etsModel){
            B[j] <- switch(Etype,
                           "M"=B[j]^sum(arimaPolynomials$arPolynomial),
                           B[j]*sum(arimaPolynomials$arPolynomial))
        }
        names(B)[j] <- constantName
        if(etsModel || sum(iOrders)!=0){
            driftSeries <- adam_driftSeries(yInSample[otLogical], Etype, etsModel, lags, iOrders)
            if(Etype=="A"){
                Bu[j] <- quantile(driftSeries,0.6)
                Bl[j] <- -Bu[j]
            }
            else{
                Bu[j] <- exp(quantile(driftSeries,0.6))
                Bl[j] <- exp(quantile(driftSeries,0.4))
            }

            # Failsafe for weird cases, when upper bound is the same or lower than the lower one
            if(Bu[j]<=Bl[j]){
                Bu[j] <- Inf
                Bl[j] <- switch(Etype,"A"=-Inf,"M"=0)
            }

            # Failsafe for cases, when the B is outside of bounds
            if(B[j]<=Bl[j]){
                Bl[j] <- switch(Etype,"A"=-Inf,"M"=0)
            }
            if(B[j]>=Bu[j]){
                Bu[j] <- Inf
            }
        }
        else{
            Bu[j] <- max(abs(yInSample[otLogical]),abs(B[j])*1.01)
            Bl[j] <- -Bu[j]
        }
    }

    # Add lambda if it is needed
    if(otherParameterEstimate){
        j[] <- j+1
        B[j] <- other
        names(B)[j] <- "other"
        Bl[j] <- 1e-10
        Bu[j] <- Inf
    }

    return(list(B=B,Bl=Bl,Bu=Bu))
}

adam_scaler <- function(distribution, Etype, errors, yFitted, obsInSample, other){
    return(switch(distribution,
                  "dnorm"=sum(errors^2)/obsInSample,
                  "dlaplace"=sum(abs(errors))/obsInSample,
                  "ds"=sum(sqrt(abs(errors))) / (obsInSample*2),
                  "dgnorm"=(other*sum(abs(errors)^other)/obsInSample)^{1/other},
                  "dalaplace"=sum(errors*(other-(errors<=0)*1))/obsInSample,
                  # Log-domain distributions: route 1+errors (or 1+errors/yFitted)
                  # through as.complex() and take abs() of the resulting log so
                  # the scale stays finite when the argument is non-positive.
                  # Equivalent Python: abs(log((1+errors).astype(complex))). The
                  # outer abs() is the modulus of the complex log, replacing the
                  # earlier Re()/abs() of a real arg pattern.
                  "dlnorm"=2*abs(switch(Etype,
                                        "A"=1-sqrt(abs(1-sum(abs(log(as.complex(1+errors/yFitted)))^2)/
                                                           obsInSample)),
                                        "M"=1-sqrt(abs(1-sum(abs(log(as.complex(1+errors)))^2)/obsInSample)))),
                  "dllaplace"=switch(Etype,
                                     "A"=sum(abs(log(as.complex(1+errors/yFitted))))/obsInSample,
                                     "M"=sum(abs(log(as.complex(1+errors))))/obsInSample),
                  "dls"=switch(Etype,
                               "A"=sum(sqrt(abs(log(as.complex(1+errors/yFitted)))))/obsInSample,
                               "M"=sum(sqrt(abs(log(as.complex(1+errors)))))/obsInSample),
                  "dlgnorm"=switch(Etype,
                                   "A"=abs((other*sum(abs(log(as.complex(1+errors/yFitted)))^other)/
                                                obsInSample)^{1/other}),
                                   "M"=abs((other*sum(abs(log(as.complex(1+errors)))^other)/
                                                obsInSample)^{1/other})),
                  "dinvgauss"=switch(Etype,
                                     "A"=sum((errors/yFitted)^2/(1+errors/yFitted))/obsInSample,
                                     "M"=sum((errors)^2/(1+errors))/obsInSample),
                  "dgamma"=switch(Etype,
                                  "A"=sum((errors/yFitted)^2)/obsInSample,
                                  "M"=sum(errors^2)/obsInSample)))
}

# The scale is the parameter of the distribution as written in the ADAM monograph
# (Tables 11.1-11.2): sigma^2 for dnorm, dlnorm, dinvgauss and dgamma, s for the
# others. The variance of the error is proportional to scale^p, with p returned here.
adam_scalePower <- function(distribution){
    return(switch(distribution,
                  "ds"=,"dls"=4,
                  "dlaplace"=,"dalaplace"=,"dgnorm"=,"dllaplace"=,"dlgnorm"=,"dlogis"=2,
                  1));
}

# De-bias the scale in the variance space: the variance is multiplied by obs/df,
# so the scale is multiplied by (obs/df)^(1/p).
adam_scaleDebias <- function(scale, distribution, obs, df){
    return(scale*(obs/df)^(1/adam_scalePower(distribution)));
}

# Degrees of freedom for de-biasing the scale: the observed sizes minus the
# parameters, without the scale ones when they were estimated by likelihood.
adam_dfScale <- function(object){
    nParam <- nparam(object);
    if(!is.null(object$loss) && object$loss=="likelihood"){
        nParam[] <- nParam - object$nParam[1,4];
    }
    df <- adam_nobsObserved(object) - nParam;
    if(df<=0){
        df[] <- adam_nobsObserved(object);
    }
    return(df);
}

# The variance of the error term implied by the scale (see adam_scalePower)
adam_scaleVariance <- function(scale, distribution, other){
    return(switch(distribution,
                  "dlaplace"=,"dllaplace"=2*scale^2,
                  "ds"=,"dls"=120*scale^4,
                  "dgnorm"=,"dlgnorm"=scale^2*gamma(3/other$shape)/gamma(1/other$shape),
                  "dalaplace"=scale^2/(other$alpha^2*(1-other$alpha)^2/(other$alpha^2+(1-other$alpha)^2)),
                  "dlogis"=scale^2*pi^2/3,
                  scale));
}

# The de-biased variance of the error term implied by the scale, or by the scale
# model's values. This, not sigma(), feeds the analytical intervals, so they come
# from the same estimate of the distribution as the likelihood and the simulations.
adam_varianceDebiased <- function(object, scaleValue=extractScale(object)){
    # An occurrence model has no scale: its errors are on the link scale
    if(is.occurrence(object)){
        return(sigma(object)^2);
    }
    return(adam_scaleVariance(scaleValue, object$distribution, object$other)*
               adam_nobsObserved(object)/adam_dfScale(object));
}

# The windows of the multistep errors (row i of ferrors() has the targets i..i+h-1)
# whose targets are all observed: the losses over the missing values are not taken
adam_completeWindows <- function(observed, h){
    missingCount <- cumsum(c(0, !observed));
    rows <- seq_len(max(length(observed)-h+1, 0));
    return(missingCount[rows+h] - missingCount[rows] == 0);
}

# The multistep loss over the windows with all their targets observed (the multistep
# losses of adam(), ces(), gum() and ssarima()): adamErrors are the errors of ferrors()
adam_multistepLoss <- function(adamErrors, loss, h, observed){
    adamErrors <- adamErrors[adam_completeWindows(observed, h),,drop=FALSE];
    nWindows <- nrow(adamErrors);
    # Not done yet: "aMSEh","aTMSE","aGTMSE","aMSCE","aGPL"
    return(switch(loss,
                  "MSEh"=sum(adamErrors[,h]^2)/nWindows,
                  "TMSE"=sum(colSums(adamErrors^2)/nWindows),
                  "GTMSE"=sum(log(colSums(adamErrors^2)/nWindows)),
                  "MSCE"=sum(rowSums(adamErrors)^2)/nWindows,
                  "MAEh"=sum(abs(adamErrors[,h]))/nWindows,
                  "TMAE"=sum(colSums(abs(adamErrors))/nWindows),
                  "GTMAE"=sum(log(colSums(abs(adamErrors))/nWindows)),
                  "MACE"=sum(abs(rowSums(adamErrors)))/nWindows,
                  "HAMh"=sum(sqrt(abs(adamErrors[,h])))/nWindows,
                  "THAM"=sum(colSums(sqrt(abs(adamErrors)))/nWindows),
                  "GTHAM"=sum(log(colSums(sqrt(abs(adamErrors)))/nWindows)),
                  "CHAM"=sum(sqrt(abs(rowSums(adamErrors))))/nWindows,
                  "GPL"=log(det(t(adamErrors) %*% adamErrors/nWindows)),
                  0));
}

# The concentrated log-likelihood of a multistep loss over the windows with all their
# targets observed, rescaled to the observed values to be comparable with the
# one-step likelihoods (taking T instead of T-h is not well motivated at the moment)
adam_multistepLogLik <- function(lossValue, loss, h, observed){
    nWindows <- sum(adam_completeWindows(observed, h));
    logLikValue <- -switch(loss,
                           "MSEh"=, "aMSEh"=, "TMSE"=, "aTMSE"=, "MSCE"=, "aMSCE"=
                               nWindows/2*(log(2*pi)+1+log(lossValue)),
                           "GTMSE"=, "aGTMSE"=
                               nWindows/2*(log(2*pi)+1+lossValue),
                           "MAEh"=, "TMAE"=, "GTMAE"=, "MACE"=
                               nWindows*(log(2)+1+log(lossValue)),
                           "HAMh"=, "THAM"=, "GTHAM"=, "CHAM"=
                               nWindows*(log(4)+2+2*log(lossValue)),
                           #### Divide GPL by h in order to make it comparable with the univariate ones
                           "GPL"=, "aGPL"=
                               nWindows/2*(h*log(2*pi)+h+lossValue)/h);
    return(logLikValue / nWindows * sum(observed));
}

# The accuracy on the observed values of the holdout, scaled by the observed in-sample
# ones: the missing values (NA) are not compared with anything
adam_accuracy <- function(holdout, forecast, inSample){
    observed <- !is.na(as.vector(holdout));
    if(!any(observed)){
        return(NULL);
    }
    inSample <- as.vector(inSample);
    return(measures(as.vector(holdout)[observed], as.vector(forecast)[observed],
                    inSample[!is.na(inSample)]));
}

# The variance of the error at each observed size relative to the one-step variance.
# After j-1 periods without an observed size (the zeros of an occurrence model and the
# missing values), the states have gone on with zero errors, and the error of a pure
# additive model at the next observed size sums the one-step errors since then, with the
# coefficients c_i of covarAnal(): its variance is 1 + sum(c_i^2, i<j). The first gap
# counts from the initial states
adam_gapVariance <- function(lagsModelAll, matWt, matF, vecG, otLogical){
    indices <- which(otLogical);
    gaps <- diff(c(0, indices));
    gapVariance <- rep(1, length(otLogical));
    if(any(gaps>1)){
        gapVariance[indices] <- diag(covarAnal(lagsModelAll, max(gaps), matWt[1,,drop=FALSE],
                                               matF, vecG, 1))[gaps];
    }
    return(gapVariance);
}

# The gap variance of the observations of a fitted model (ones unless the model is pure
# additive with the normal distribution), at its observed sizes
adam_gapVarianceModel <- function(object){
    y <- as.vector(actuals(object));
    otLogical <- !is.na(y);
    if(is.list(object$occurrence) && any(as.vector(tbats_pFitted(object))[otLogical]!=1)){
        otLogical[] <- otLogical & y!=0;
    }
    gapVariance <- rep(1, length(y));
    # A scale model (sm()) has its own residuals, standardised already
    if(object$distribution=="dnorm" && errorType(object)=="A" && !grepl("M", modelType(object)) &&
       !is.scale(object)){
        gapVariance[] <- adam_gapVariance(modelLags(object), object$measurement, object$transition,
                                          matrix(object$persistence, ncol=1), otLogical);
    }
    return(gapVariance);
}

# The observed sizes, which the scale is divided by: the missing values are not, and
# neither are the zeros of an occurrence model, whose sizes are not observed
adam_nobsObserved <- function(object){
    y <- as.vector(actuals(object));
    observed <- !is.na(y);
    # An occurrence model (om(), with its type in $occurrence) has no sizes
    if(is.list(object$occurrence) && any(as.vector(tbats_pFitted(object))[observed]!=1)){
        observed[] <- observed & y!=0;
    }
    return(sum(observed));
}

# The de-biased variance from the scale model's forecasts
adam_scaleModelVariance <- function(object, h, newdata){
    scaleValue <- forecast(object$scale,h=h,newdata=newdata,interval="none")$mean;
    scaleValue[] <- adam_varianceDebiased(object, scaleValue);
    return(scaleValue);
}

# The scale for simulations, de-biased in the variance space with the df of the
# location model. scaleValue is either the scale or the scale model's values.
adam_scaleSimulation <- function(object, scaleValue){
    return(adam_scaleDebias(scaleValue, object$distribution, adam_nobsObserved(object), adam_dfScale(object)));
}

# Random errors of the model for the provided scale (see adam_scalePower).
# dnorm and dlnorm take the square root of their scale, sigma^2.
adam_errorsSimulate <- function(n, distribution, scale, other, dfT){
    return(switch(distribution,
                  "plogis"=,
                  "dnorm"=rnorm(n, 0, sqrt(scale)),
                  "dlaplace"=rlaplace(n, 0, scale),
                  "ds"=rs(n, 0, scale),
                  "dgnorm"=rgnorm(n, 0, scale, other$shape),
                  "dlogis"=rlogis(n, 0, scale),
                  "dt"=rt(n, dfT),
                  "dalaplace"=ralaplace(n, 0, scale, other$alpha),
                  "dlnorm"=rlnorm(n, -scale/2, sqrt(scale))-1,
                  "dinvgauss"=rinvgauss(n, 1, dispersion=scale)-1,
                  "dgamma"=rgamma(n, shape=scale^{-1}, scale=scale)-1,
                  "dllaplace"=exp(rlaplace(n, 0, scale))-1,
                  "dls"=exp(rs(n, 0, scale))-1,
                  "dlgnorm"=exp(rgnorm(n, 0, scale, other$shape))-1));
}

# The power-law quantile A1*j^A2, j=1..h, of the multistep errors at the level,
# minimising the pinball loss (Taylor & Bunn). For a given A2 the best A1 is the
# weighted quantile of e/j^A2 with weights j^A2, so only A2 is optimised.
adam_quantilePower <- function(errors, level){
    h <- ncol(errors);
    horizons <- col(errors)[!is.na(errors)];
    errors <- errors[!is.na(errors)];
    # The weighted quantile of errors/j^power: the first value reaching the level
    quantileA1 <- function(power){
        weights <- horizons^power;
        ratios <- errors / weights;
        ordering <- order(ratios);
        weightsCumulative <- cumsum(weights[ordering]);
        return(ratios[ordering][which(weightsCumulative >= level*weightsCumulative[length(ordering)])[1]]);
    }
    pinball <- function(power){
        residualsQuantile <- errors - quantileA1(power)*horizons^power;
        return((1-level)*sum(abs(residualsQuantile[residualsQuantile<0])) +
                   level*sum(abs(residualsQuantile[residualsQuantile>=0])));
    }
    power <- nloptr(0.5, pinball, lb=-2, ub=4,
                    opts=list(algorithm="NLOPT_LN_NELDERMEAD", xtol_rel=1e-8, xtol_abs=1e-6,
                              maxeval=500))$solution;
    return(quantileA1(power)*c(1:h)^power);
}

#### IC weights (Akaike weights) ####
adam_ic_weights <- function(icSelection, threshold=1e-5){
    icBest <- min(icSelection);
    weights <- exp(-0.5*(icSelection - icBest)) / sum(exp(-0.5*(icSelection - icBest)));
    weights[weights < threshold] <- 0;
    weights <- weights / sum(weights);
    return(weights);
}

#### Bounds checker for ARIMA stationarity and ETS smoothing parameter constraints ####
# The scales of LASSO / RIDGE: the standard deviations of the explanatory variables
# (denominator), which normalise their parameters, and of the differenced series
# (yDenominator), which normalises the errors of adam()
adam_lassoDenominators <- function(loss, matWt, componentsNumberETS, componentsNumberARIMA,
                                   xregNumber, yInSample){
    denominator <- yDenominator <- NULL;
    if(any(loss==c("LASSO","RIDGE"))){
        if(xregNumber>0){
            denominator <- apply(matWt[,componentsNumberETS+componentsNumberARIMA+1:xregNumber,drop=FALSE],
                                 2, sd);
            denominator[is.infinite(denominator)] <- 1;
        }
        yDenominator <- max(sd(diff(yInSample)),1);
    }
    return(list(denominator=denominator, yDenominator=yDenominator));
}

# The estimated parameters of the LASSO / RIDGE penalty, found in B as the filler does,
# shifted to be zero at their shrinkage targets: no smoothing, phi and AR of one ("no
# good understanding how to shrink ARMA"), MA and regressors of zero, the additive
# regressors normalised by their standard deviations (denominator). The initial states
# are not shrunk. Shared by adam(), om() and omg()
adam_penaltyParameters <- function(B, Etype, etsModel, modelIsTrendy, modelIsSeasonal,
                                   persistenceEstimate, persistenceLevelEstimate,
                                   persistenceTrendEstimate, persistenceSeasonalEstimate,
                                   xregModel, persistenceXregEstimate, xregParametersPersistence,
                                   phiEstimate, arimaModel, arEstimate, maEstimate, arOrders, maOrders,
                                   initialType, initialEstimate, initialXregEstimate,
                                   xregParametersEstimated, constantEstimate, otherParameterEstimate,
                                   denominator){
    nSmoothing <- persistenceEstimate *
        (etsModel * (persistenceLevelEstimate + modelIsTrendy*persistenceTrendEstimate +
                         modelIsSeasonal*sum(persistenceSeasonalEstimate)) +
             xregModel * persistenceXregEstimate * max(c(0, xregParametersPersistence)));
    nPhi <- etsModel * phiEstimate;
    arOrdersEstimated <- arimaModel * arEstimate * arOrders;
    maOrdersEstimated <- arimaModel * maEstimate * maOrders;
    arma <- B[nSmoothing + nPhi + seq_len(sum(arOrdersEstimated, maOrdersEstimated))];
    arEstimated <- as.logical(unlist(lapply(seq_along(arOrdersEstimated), function(i){
        return(rep(c(TRUE,FALSE), c(arOrdersEstimated[i], maOrdersEstimated[i])))})));
    xregEstimated <- xregModel && (initialType!="complete") && initialEstimate && initialXregEstimate;
    nXreg <- xregEstimated * sum(xregParametersEstimated);
    xreg <- B[length(B) - otherParameterEstimate - constantEstimate - nXreg + seq_len(nXreg)];
    if(Etype=="A" && nXreg>0){
        xreg <- xreg / denominator[xregParametersEstimated==1];
    }
    return(c(B[seq_len(nSmoothing)], 1-B[nSmoothing + seq_len(nPhi)],
             1-arma[arEstimated], arma[!arEstimated], xreg));
}

adam_bounds_checker <- function(adamElements, arimaPolynomials,
                                bounds,
                                etsModel, modelIsTrendy, modelIsSeasonal,
                                componentsNumberETS, componentsNumberETSNonSeasonal,
                                componentsNumberETSSeasonal,
                                arimaModel, arEstimate, maEstimate,
                                xregModel, regressors, xregNumber, componentsNumberARIMA,
                                lagsModelAll, obsInSample,
                                phiEstimate){
    # Stationary AR and invertible MA, factor by factor (src/headers/arimaBounds.h)
    if(bounds!="none" && arimaModel){
        arimaReflection <- max(arEstimate*arimaPolynomials$arReflection,
                               maEstimate*arimaPolynomials$maReflection);
        if(arimaReflection>=1){
            return(1E+100*arimaReflection);
        }
    }

    if(bounds=="usual"){

        if(etsModel){
            if(any(adamElements$vecG[1:componentsNumberETS]>1) ||
               any(adamElements$vecG[1:componentsNumberETS]<0)){
                return(1E+300);
            }
            if(modelIsTrendy){
                if(adamElements$vecG[2] > adamElements$vecG[1]){
                    return(1E+300);
                }
                if(modelIsSeasonal &&
                   any(adamElements$vecG[componentsNumberETSNonSeasonal+c(1:componentsNumberETSSeasonal)] >
                       (1-adamElements$vecG[1]))){
                    return(1E+300);
                }
            }
            else{
                if(modelIsSeasonal &&
                   any(adamElements$vecG[componentsNumberETSNonSeasonal+c(1:componentsNumberETSSeasonal)] >
                       (1-adamElements$vecG[1]))){
                    return(1E+300);
                }
            }
            if(phiEstimate && (adamElements$matF[2,2]>1 || adamElements$matF[2,2]<0)){
                return(1E+300);
            }
        }

        if(xregModel && regressors=="adapt"){
            if(any(adamElements$vecG[componentsNumberETS+componentsNumberARIMA+1:xregNumber]>1) ||
               any(adamElements$vecG[componentsNumberETS+componentsNumberARIMA+1:xregNumber]<0)){
                return(1E+100*max(abs(
                    adamElements$vecG[componentsNumberETS+componentsNumberARIMA+1:xregNumber]-0.5
                )));
            }
        }
    }
    else if(bounds=="admissible"){
        # The discount matrix split by lags is meaningless for the lagged ARIMA,
        # which has one state per lag: its states are checked above instead
        componentsOther <- setdiff(seq_along(lagsModelAll), componentsNumberETS+seq_len(componentsNumberARIMA))
        if(length(componentsOther)>0){
            eigenValues <- smoothEigens(adamElements$vecG[componentsOther,,drop=FALSE],
                                        adamElements$matF[componentsOther,componentsOther,drop=FALSE],
                                        adamElements$matWt[,componentsOther,drop=FALSE],
                                        lagsModelAll[componentsOther], xregModel, obsInSample);
            if(any(eigenValues>1+1E-50)){
                return(1E+100*max(eigenValues));
            }
        }
    }
    return(0);
}

#### xreg variable selector using stepwise regression on residuals ####
adam_xreg_selector <- function(errors, xregData, obsInSample, ic, df, distribution,
                               occurrence, other){
    alpha <- shape <- nu <- NULL;
    if(distribution=="dalaplace"){
        alpha <- other;
    }
    else if(any(distribution==c("dgnorm","dlgnorm"))){
        shape <- other;
    }
    else if(distribution=="dt"){
        nu <- other;
    }
    stepwiseModel <- suppressWarnings(
        stepwise(data.frame(errorsIvan41=errors, xregData[1:obsInSample,,drop=FALSE]),
                 ic=ic, df=df, distribution=distribution, occurrence=occurrence, silent=TRUE,
                 alpha=alpha, shape=shape, nu=nu)
    );
    return(list(initialXreg=coef(stepwiseModel)[-1], other=stepwiseModel$other,
                formula=formula(stepwiseModel)));
}

#### Model name assembler ####
adam_model_name <- function(etsModel, model, xregModel, arimaModel,
                            arOrders, iOrders, maOrders, lags,
                            regressors, constantRequired, constantName,
                            occurrence, componentsNumberETSSeasonal,
                            prefix = "i"){
    modelName <- "";
    if(etsModel){
        if(model!="NNN"){
            modelName <- "ETS";
            if(xregModel){
                modelName <- paste0(modelName,"X");
            }
            modelName <- paste0(modelName,"(",model,")");
            if(componentsNumberETSSeasonal>1){
                modelName <- paste0(modelName,"[",
                                    paste0(lags[lags!=1], collapse=", "),"]");
            }
        }
    }
    if(arimaModel){
        if(etsModel){
            modelName <- paste0(modelName,"+");
        }
        if(all(lags==1) || (all(arOrders[lags>1]==0) && all(iOrders[lags>1]==0) &&
                            all(maOrders[lags>1]==0))){
            modelName <- paste0(modelName,"ARIMA");
            if(!etsModel && xregModel){
                modelName <- paste0(modelName,"X");
            }
            modelName <- paste0(modelName,"(",arOrders[1],",",iOrders[1],",",
                                maOrders[1],")");
        }
        else{
            modelName <- paste0(modelName,"SARIMA");
            if(!etsModel && xregModel){
                modelName <- paste0(modelName,"X");
            }
            for(i in 1:length(arOrders)){
                if(all(arOrders[i]==0) && all(iOrders[i]==0) && all(maOrders[i]==0)){
                    next;
                }
                modelName <- paste0(modelName,"(",arOrders[i],",",iOrders[i],",",
                                    maOrders[i],")[",lags[i],"]");
            }
        }
    }
    if(regressors=="adapt"){
        modelName <- paste0(modelName,"{D}");
    }
    if(!etsModel && !arimaModel){
        if(model=="NNN"){
            modelName <- "Constant level";
        }
        else if(regressors=="adapt"){
            modelName <- "Dynamic regression";
        }
        else{
            modelName <- "Regression";
        }
    }
    else{
        if(constantRequired){
            modelName <- paste0(modelName," with ",constantName);
        }
    }
    if(!is.null(occurrence) && all(occurrence!=c("n","none"))){
        modelName <- paste0(prefix,modelName,
                            switch(occurrence,
                                   "f"=,"fixed"="[F]",
                                   "d"=,"direct"="[D]",
                                   "o"=,"odds-ratio"="[O]",
                                   "i"=,"inverse-odds-ratio"="[I]",
                                   "g"=,"general"="[G]",
                                   ""));
    }
    return(modelName);
}

#### Initial values collector ####
adam_initial_collector <- function(matVt, etsModel, modelIsTrendy, modelIsSeasonal,
                                   lagsModel, lagsModelMax,
                                   initialLevelEstimate, initialTrendEstimate,
                                   initialSeasonalEstimate,
                                   componentsNumberETSSeasonal,
                                   arimaModel, initialArimaEstimate, initialArima,
                                   initialArimaNumber,
                                   componentsNumberETS, componentsNumberARIMA,
                                   arimaPolynomials, Etype,
                                   xregModel, initialXregEstimate, xregNumber, lagsModelARIMA){
    initialValue <- vector("list", etsModel*(1+modelIsTrendy+modelIsSeasonal)+arimaModel+xregModel);
    initialValueETS <- vector("list", etsModel*length(lagsModel));
    initialValueNames <- vector("character", etsModel*(1+modelIsTrendy+modelIsSeasonal)+arimaModel+xregModel);
    # The vector that defines what was estimated in the model
    initialEstimated <- vector("logical", etsModel*(1+modelIsTrendy+modelIsSeasonal*componentsNumberETSSeasonal)+
                                   arimaModel+xregModel);

    # Write down the initials of ETS
    j <- 0;
    if(etsModel){
        # Write down level, trend and seasonal
        for(i in 1:length(lagsModel)){
            # In case of level / trend, we want to get the very first value
            if(lagsModel[i]==1){
                initialValueETS[[i]] <- head(matVt[i,1:lagsModelMax],1);
            }
            # In cases of seasonal components, they should be at the end of the pre-heat period
            else{
                initialValueETS[[i]] <- tail(matVt[i,1:lagsModelMax],lagsModel[i]);
            }
        }
        j[] <- j+1;
        # Write down level in the final list
        initialEstimated[j] <- initialLevelEstimate;
        initialValue[[j]] <- initialValueETS[[j]];
        initialValueNames[j] <- c("level");
        names(initialEstimated)[j] <- initialValueNames[j];
        if(modelIsTrendy){
            j[] <- 2;
            initialEstimated[j] <- initialTrendEstimate;
            # Write down trend in the final list
            initialValue[[j]] <- initialValueETS[[j]];
            # Remove the trend from ETS list
            initialValueETS[[j]] <- NULL;
            initialValueNames[j] <- c("trend");
            names(initialEstimated)[j] <- initialValueNames[j];
        }
        # Write down the initial seasonals
        if(modelIsSeasonal){
            initialEstimated[j+c(1:componentsNumberETSSeasonal)] <- initialSeasonalEstimate;
            # Remove the level from ETS list
            initialValueETS[[1]] <- NULL;
            j[] <- j+1;
            if(length(initialSeasonalEstimate)>1){
                initialValue[[j]] <- initialValueETS;
                initialValueNames[[j]] <- "seasonal";
                names(initialEstimated)[j+0:(componentsNumberETSSeasonal-1)] <-
                    paste0(initialValueNames[j],c(1:componentsNumberETSSeasonal));
            }
            else{
                initialValue[[j]] <- initialValueETS[[1]];
                initialValueNames[[j]] <- "seasonal";
                names(initialEstimated)[j] <- initialValueNames[j];
            }
        }
    }

    # Write down the ARIMA initials
    if(arimaModel){
        j[] <- j+1;
        initialEstimated[j] <- initialArimaEstimate;
        if(initialArimaEstimate){
            initialValue[[j]] <- adam_arimaHeadInitials(matVt[componentsNumberETS+1:componentsNumberARIMA,,drop=FALSE],
                                                        lagsModelARIMA, lagsModelMax, Etype);
        }
        else{
            initialValue[[j]] <- initialArima;
        }
        initialValueNames[j] <- "arima";
        names(initialEstimated)[j] <- initialValueNames[j];
    }
    # Write down the xreg initials
    if(xregModel){
        j[] <- j+1;
        initialEstimated[j] <- initialXregEstimate;
        initialValue[[j]] <- matVt[componentsNumberETS+componentsNumberARIMA+1:xregNumber,lagsModelMax];
        initialValueNames[j] <- "xreg";
        names(initialEstimated)[j] <- initialValueNames[j];
    }
    names(initialValue) <- initialValueNames;

    return(list(initialValue=initialValue, initialEstimated=initialEstimated));
}

#### ETS model selector (branch-and-bound + full pool) ####
adam_selector <- function(estimator_fn, model, modelsPool, allowMultiplicative,
                          modelDo="estimate",
                          etsModel, Etype, Ttype, Stype, damped, lags,
                          lagsModelSeasonal, lagsModelARIMA,
                          obsStates, obsInSample,
                          yInSample, persistence, persistenceEstimate,
                          persistenceLevel, persistenceLevelEstimate,
                          persistenceTrend, persistenceTrendEstimate,
                          persistenceSeasonal, persistenceSeasonalEstimate,
                          persistenceXreg, persistenceXregEstimate, persistenceXregProvided,
                          phi, phiEstimate,
                          initialType, initialLevel, initialTrend, initialSeasonal,
                          initialArima, initialEstimate,
                          initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                          initialArimaEstimate, initialXregEstimate, initialXregProvided,
                          arimaModel, arRequired, iRequired, maRequired, armaParameters,
                          componentsNumberARIMA, componentsNamesARIMA,
                          formula, xregModel, xregModelInitials, xregData,
                          xregNumber, xregNames, regressors,
                          xregParametersMissing, xregParametersIncluded,
                          xregParametersEstimated, xregParametersPersistence,
                          constantRequired, constantEstimate, constantValue, constantName,
                          ot, otLogical, occurrenceModel, pFitted, icFunction,
                          bounds, loss, lossFunction, distribution,
                          horizon, multisteps, other, otherParameterEstimate, lambda,
                          silent, B){
    if(is.null(modelsPool)){
        if(!allowMultiplicative){
            poolErrors <- c("A");
            poolTrends <- c("N","A","Ad");
            poolSeasonals <- c("N","A");
        }
        else{
            poolErrors <- c("A","M");
            poolTrends <- c("N","A","Ad","M","Md");
            poolSeasonals <- c("N","A","M");
        }

        if(Etype!="Z"){
            poolErrors <- poolErrorsSmall <- Etype;
        }
        else{
            poolErrorsSmall <- "A";
        }

        if(Ttype!="Z"){
            if(Ttype=="X"){
                poolTrendsSmall <- c("N","A");
                poolTrends <- c("N","A","Ad");
                checkTrend <- TRUE;
            }
            else if(Ttype=="Y"){
                poolTrendsSmall <- c("N","M");
                poolTrends <- c("N","M","Md");
                checkTrend <- TRUE;
            }
            else{
                if(damped){
                    poolTrends <- poolTrendsSmall <- paste0(Ttype,"d");
                }
                else{
                    poolTrends <- poolTrendsSmall <- Ttype;
                }
                checkTrend <- FALSE;
            }
        }
        else{
            poolTrendsSmall <- c("N","A");
            checkTrend <- TRUE;
        }

        if(Stype!="Z"){
            if(Stype=="X"){
                poolSeasonals <- poolSeasonalsSmall <- c("N","A");
                checkSeasonal <- TRUE;
            }
            else if(Stype=="Y"){
                poolSeasonalsSmall <- c("N","M");
                poolSeasonals <- c("N","M");
                checkSeasonal <- TRUE;
            }
            else{
                poolSeasonalsSmall <- Stype;
                poolSeasonals <- Stype;
                checkSeasonal <- FALSE;
            }
        }
        else{
            poolSeasonalsSmall <- c("N","A","M");
            checkSeasonal <- TRUE;
        }

        # For combination, use the full pool — no branch-and-bound
        if(modelDo == "combine"){
            modelsPool <- paste0(rep(poolErrors, each=length(poolTrends)*length(poolSeasonals)),
                                 rep(poolTrends, each=length(poolSeasonals)),
                                 rep(poolSeasonals, length(poolErrors)*length(poolTrends)));
            j <- 0;
            results <- vector("list", length(modelsPool));
        }
        else{
            if(!silent){
                cat("Forming the pool of models based on... ");
            }
            poolSmall <- paste0(rep(poolErrorsSmall, length(poolTrendsSmall)*length(poolSeasonalsSmall)),
                                rep(poolTrendsSmall, each=length(poolSeasonalsSmall)),
                                rep(poolSeasonalsSmall, length(poolTrendsSmall)));
            if(any(substr(poolSmall,3,3)=="M") && all(Etype!=c("A","X"))){
                multiplicativeSeason <- (substr(poolSmall,3,3)=="M");
                poolSmall[multiplicativeSeason] <- paste0("M", substr(poolSmall[multiplicativeSeason],2,3));
            }
            modelsTested <- NULL;
            modelCurrent <- NA;

            j <- 1;
            i <- 0;
            check <- TRUE;
            besti <- bestj <- 1;
            results <- vector("list", length(poolSmall));

            while(check){
                i <- i + 1;
                modelCurrent[] <- poolSmall[j];
                if(!silent){
                    cat(modelCurrent,"\b, ");
                }
                Etype[] <- substring(modelCurrent,1,1);
                Ttype[] <- substring(modelCurrent,2,2);
                if(nchar(modelCurrent)==4){
                    phi[] <- 0.95;
                    phiEstimate[] <- TRUE;
                    Stype[] <- substring(modelCurrent,4,4);
                }
                else{
                    phi[] <- 1;
                    phiEstimate[] <- FALSE;
                    Stype[] <- substring(modelCurrent,3,3);
                }

                results[[i]] <- estimator_fn(etsModel, Etype, Ttype, Stype, lags,
                                             lagsModelSeasonal, lagsModelARIMA,
                                             obsStates, obsInSample,
                                             yInSample, persistence, persistenceEstimate,
                                             persistenceLevel, persistenceLevelEstimate,
                                             persistenceTrend, persistenceTrendEstimate,
                                             persistenceSeasonal, persistenceSeasonalEstimate,
                                             persistenceXreg, persistenceXregEstimate,
                                             persistenceXregProvided,
                                             phi, phiEstimate,
                                             initialType, initialLevel, initialTrend, initialSeasonal,
                                             initialArima, initialEstimate,
                                             initialLevelEstimate, initialTrendEstimate,
                                             initialSeasonalEstimate,
                                             initialArimaEstimate, initialXregEstimate,
                                             initialXregProvided,
                                             arimaModel, arRequired, iRequired, maRequired,
                                             armaParameters,
                                             componentsNumberARIMA, componentsNamesARIMA,
                                             formula, xregModel, xregModelInitials, xregData,
                                             xregNumber, xregNames, regressors,
                                             xregParametersMissing, xregParametersIncluded,
                                             xregParametersEstimated, xregParametersPersistence,
                                             constantRequired, constantEstimate, constantValue,
                                             constantName,
                                             ot, otLogical, occurrenceModel, pFitted,
                                             bounds, loss, lossFunction, distribution,
                                             horizon, multisteps, other, otherParameterEstimate,
                                             lambda, B);
                results[[i]]$IC <- icFunction(results[[i]]$logLikADAMValue);
                results[[i]]$Etype <- Etype;
                results[[i]]$Ttype <- Ttype;
                results[[i]]$Stype <- Stype;
                results[[i]]$phiEstimate <- phiEstimate;
                if(phiEstimate){
                    results[[i]]$phi <- results[[i]]$B[names(results[[i]]$B)=="phi"];
                }
                else{
                    results[[i]]$phi <- 1;
                }
                results[[i]]$model <- modelCurrent;
                modelsTested <- c(modelsTested, modelCurrent);

                if(j>1){
                    if(results[[besti]]$IC <= results[[i]]$IC){
                        if(substring(modelCurrent,2,2)==substring(poolSmall[bestj],2,2)){
                            poolSeasonals <- results[[besti]]$Stype;
                            checkSeasonal <- FALSE;
                            j <- which(poolSmall!=poolSmall[bestj] &
                                           substring(poolSmall,nchar(poolSmall),nchar(poolSmall))==poolSeasonals);
                        }
                        else{
                            poolTrends <- results[[bestj]]$Ttype;
                            checkTrend[] <- FALSE;
                        }
                    }
                    else{
                        if(substring(modelCurrent,2,2) == substring(poolSmall[besti],2,2)){
                            poolSeasonals <- poolSeasonals[poolSeasonals!=results[[besti]]$Stype];
                            if(length(poolSeasonals)>1){
                                bestj[] <- j;
                                besti[] <- i;
                                j <- 3;
                            }
                            else{
                                bestj[] <- j;
                                besti[] <- i;
                                j <- which(substring(poolSmall,nchar(poolSmall),nchar(poolSmall))==poolSeasonals &
                                               substring(poolSmall,2,2)!=substring(modelCurrent,2,2));
                                checkSeasonal[] <- FALSE;
                            }
                        }
                        else{
                            poolTrends <- poolTrends[poolTrends!=results[[bestj]]$Ttype];
                            besti[] <- i;
                            bestj[] <- j;
                            checkTrend[] <- FALSE;
                        }
                    }

                    if(all(!c(checkTrend,checkSeasonal))){
                        check[] <- FALSE;
                    }
                }
                else{
                    j <- 2;
                }

                if(length(j)==0){
                    j <- length(poolSmall);
                }
                if(j>length(poolSmall)){
                    check[] <- FALSE;
                }
            }

            modelsPool <- unique(c(modelsTested,
                                   paste0(rep(poolErrors, each=length(poolTrends)*length(poolSeasonals)),
                                          poolTrends,
                                          rep(poolSeasonals, each=length(poolTrends)))));
            j <- length(modelsTested);
        }
    }
    else{
        j <- 0;
        results <- vector("list", length(modelsPool));
    }
    modelsNumber <- length(modelsPool);

    if(!silent){
        cat("Estimation progress:    ");
    }
    while(j < modelsNumber){
        j <- j + 1;
        if(!silent){
            if(j==1){
                cat("\b");
            }
            cat(paste0(rep("\b",nchar(round((j-1)/modelsNumber,2)*100)+1),collapse=""));
            cat(round(j/modelsNumber,2)*100,"\b%");
        }

        modelCurrent <- modelsPool[j];
        Etype <- substring(modelCurrent,1,1);
        Ttype <- substring(modelCurrent,2,2);
        if(nchar(modelCurrent)==4){
            phi[] <- 0.95;
            Stype <- substring(modelCurrent,4,4);
            phiEstimate <- TRUE;
        }
        else{
            phi[] <- 1;
            Stype <- substring(modelCurrent,3,3);
            phiEstimate <- FALSE;
        }

        results[[j]] <- estimator_fn(etsModel, Etype, Ttype, Stype, lags,
                                     lagsModelSeasonal, lagsModelARIMA,
                                     obsStates, obsInSample,
                                     yInSample, persistence, persistenceEstimate,
                                     persistenceLevel, persistenceLevelEstimate,
                                     persistenceTrend, persistenceTrendEstimate,
                                     persistenceSeasonal, persistenceSeasonalEstimate,
                                     persistenceXreg, persistenceXregEstimate,
                                     persistenceXregProvided,
                                     phi, phiEstimate,
                                     initialType, initialLevel, initialTrend, initialSeasonal,
                                     initialArima, initialEstimate,
                                     initialLevelEstimate, initialTrendEstimate,
                                     initialSeasonalEstimate,
                                     initialArimaEstimate, initialXregEstimate,
                                     initialXregProvided,
                                     arimaModel, arRequired, iRequired, maRequired,
                                     armaParameters,
                                     componentsNumberARIMA, componentsNamesARIMA,
                                     formula, xregModel, xregModelInitials, xregData,
                                     xregNumber, xregNames, regressors,
                                     xregParametersMissing, xregParametersIncluded,
                                     xregParametersEstimated, xregParametersPersistence,
                                     constantRequired, constantEstimate, constantValue,
                                     constantName,
                                     ot, otLogical, occurrenceModel, pFitted,
                                     bounds, loss, lossFunction, distribution,
                                     horizon, multisteps, other, otherParameterEstimate,
                                     lambda, B);
        results[[j]]$IC <- icFunction(results[[j]]$logLikADAMValue);
        results[[j]]$Etype <- Etype;
        results[[j]]$Ttype <- Ttype;
        results[[j]]$Stype <- Stype;
        results[[j]]$phiEstimate <- phiEstimate;
        if(phiEstimate){
            results[[j]]$phi <- results[[j]]$B[names(results[[j]]$B)=="phi"];
        }
        else{
            results[[j]]$phi <- 1;
        }
        results[[j]]$model <- modelCurrent;
    }

    if(!silent){
        cat("... Done! \n");
    }

    icSelection <- vector("numeric", modelsNumber);
    for(i in 1:modelsNumber){
        icSelection[i] <- results[[i]]$IC;
    }
    names(icSelection) <- modelsPool;
    icSelection[is.nan(icSelection)] <- 1E100;

    return(list(results=results, icSelection=icSelection));
}

#### Parallel setup / teardown ####
adam_setupParallel <- function(parallel, nModels){
    if(is.numeric(parallel)){
        nCores <- parallel;
        parallel <- TRUE;
    }
    else{
        nCores <- min(parallel::detectCores() - 1, nModels);
    }

    if(parallel){
        if(!requireNamespace("foreach", quietly=TRUE)){
            stop("In order to run the function in parallel, 'foreach' package must be installed.", call.=FALSE);
        }
        if(!requireNamespace("parallel", quietly=TRUE)){
            stop("In order to run the function in parallel, 'parallel' package must be installed.", call.=FALSE);
        }

        if(Sys.info()['sysname']=="Windows"){
            if(requireNamespace("doParallel", quietly=TRUE)){
                cat("Setting up", nCores, "clusters using 'doParallel'...\n");
                cluster <- parallel::makeCluster(nCores);
                doParallel::registerDoParallel(cluster);
            }
            else{
                stop("Sorry, but in order to run the function in parallel, you need 'doParallel' package.", call.=FALSE);
            }
        }
        else{
            if(requireNamespace("doMC", quietly=TRUE)){
                doMC::registerDoMC(nCores);
                cluster <- NULL;
            }
            else if(requireNamespace("doParallel", quietly=TRUE)){
                cat("Setting up", nCores, "clusters using 'doParallel'...\n");
                cluster <- parallel::makeCluster(nCores);
                doParallel::registerDoParallel(cluster);
            }
            else{
                stop(paste0("Sorry, but in order to run the function in parallel, you need either ",
                            "'doMC' (prefered) or 'doParallel' package."), call.=FALSE);
            }
        }
    }
    else{
        cluster <- NULL;
    }

    return(list(parallel=parallel, cluster=cluster, nCores=nCores));
}

adam_teardownParallel <- function(cluster){
    if(!is.null(cluster)){
        parallel::stopCluster(cluster);
    }
}

#### ARIMA order selector ####
adam_arimaSelector <- function(data, model, lags, arMax, iMax, maMax,
                                h, holdout,
                                persistence, phi, initial,
                                ic, bounds, silent, regressors,
                                testModelETS,
                                fitter = adam,
                                fitter_args = list(),
                                IC = AICc,
                                formula = NULL,
                                ets = "conventional",
                                responseName = "y",
                                obsInSample, ...){
    silentDebug <- FALSE;
    env <- environment();
    # Strip 'constant' from ... — it is added explicitly in each do.call below,
    # and a duplicate named argument causes an error in R.
    dots <- list(...);
    dots[["constant"]] <- NULL;

    modelOriginal <- model;
    etsModelType <- model;

    if(is.adam(testModelETS)){
        model <- etsModelType <- modelType(testModelETS);
        ic_val <- tryCatch(IC(testModelETS),
                           warning = function(w) numeric(0),
                           error   = function(e) numeric(0));
        ICOriginal <- if(length(ic_val) > 0) ic_val else Inf;
    }
    else{
        ICOriginal <- Inf;
    }

    if(!silent){
        cat(" Selecting ARIMA orders... ");
    }

    # Kick off IMA elements if ETS was fitted
    if(any(substr(etsModelType,1,1) %in% c("A","M"))){
        iMax[lags==1] <- 0;
        maMax[lags==1] <- 0;
    }
    if(any(substr(etsModelType,3,3)=="d")){
        arMax[lags==1] <- 0;
    }
    if(any(substr(etsModelType,nchar(etsModelType),nchar(etsModelType)) %in% c("A","M"))){
        iMax[lags!=1] <- 0;
        maMax[lags!=1] <- 0;
    }

    if(all(maMax==0)){
        nModelsARIMA <- prod(iMax + 1) * (1 + sum(arMax));
    }
    else{
        nModelsARIMA <- prod(iMax + 1) * (1 + sum(maMax*(1 + sum(arMax))));
    }

    ordersLength <- length(lags);
    lagsMax <- max(lags);
    lagsTest <- maTest <- arTest <- rep(0,ordersLength);
    arBest <- maBest <- iBest <- rep(0,ordersLength);
    arBestLocal <- maBestLocal <- arBest;

    iCombinations <- prod(iMax+1);
    iOrders <- matrix(0,iCombinations*2,ncol=ordersLength+1);

    if(any(iMax!=0)){
        iOrders[,1] <- rep(c(0:iMax[1]),times=prod(iMax[-1]+1));
        if(ordersLength>1){
            for(seasLag in 2:ordersLength){
                iOrders[,seasLag] <- rep(c(0:iMax[seasLag]),each=prod(iMax[1:(seasLag-1)]+1));
            }
        }
    }
    iOrders[1:iCombinations+iCombinations,] <- iOrders[1:iCombinations,];
    iOrders[,ordersLength+1] <- rep(c(0,1),each=iCombinations);

    iOrdersICs <- vector("numeric",iCombinations*2);
    iOrdersICs[1] <- ICOriginal;

    if(!silent){
        cat("\nSelecting differences... ");
    }

    # Base call arguments shared by every fitter call.
    # The data is passed as the symbol `data`, not by value: the fitter takes the
    # response name from deparse(substitute(data)), so a by-value call deparsed
    # the whole series into that name. Beyond roughly a thousand observations
    # this exceeds R's 10000-byte limit on names, and building the default
    # formula from it aborted the selection with "variable names are limited to
    # 10000 bytes". do.call() below evaluates the symbol in this frame.
    base_call <- c(list(data=quote(data), model=model, lags=lags,
                        formula=formula, h=h, holdout=holdout,
                        persistence=persistence, phi=phi, initial=initial,
                        ic=ic, bounds=bounds, regressors=regressors,
                        silent=TRUE, ets=ets, environment=env),
                   fitter_args);

    for(d in 2:(iCombinations*2)){
        testModel <- try(do.call(fitter,
                                 c(base_call,
                                   list(orders=list(ar=0, i=iOrders[d,1:ordersLength], ma=0),
                                        constant=(iOrders[d,ordersLength+1]==1)),
                                   dots)),
                         silent=TRUE);
        if(!inherits(testModel,"try-error")){
            iOrdersICs[d] <- IC(testModel);
        }
        else{
            iOrdersICs[d] <- Inf;
        }
    }
    d <- which.min(iOrdersICs);
    iBest <- iOrders[d,1:ordersLength];
    constantValue <- iOrders[d,ordersLength+1]==1;

    # The winner is refitted from a cold start, exactly as a direct call with
    # these orders would be. Carrying the search's parameter vector over as a
    # warm start made auto.msarima() report a different optimum than msarima()
    # with the very same orders, which is confusing and not reproducible by the
    # user. Both are truncated at the same maxeval, so the warm start only moved
    # the truncation point. It also left bestIC (taken from the cold loop fit
    # below) describing a different fit than the bestModel returned alongside it.
    bestModel <- testModel <- do.call(fitter,
                                      c(base_call,
                                        list(orders=list(ar=0, i=iBest, ma=0),
                                             constant=constantValue),
                                        dots));
    bestIC <- iOrdersICs[d];

    if(silentDebug){
        cat("Best IC:",bestIC,"\n");
    }
    maTest <- rep(0,ordersLength);
    arTest <- rep(0,ordersLength);

    if(!silent){
        cat("\nSelecting ARMA... |");
        mSymbols <- c("/","-","\\","|","/","-","\\","|","/","-","\\","|","/","-","\\","|");
    }

    for(i in ordersLength:1){
        if(!silent){
            m <- 1;
        }
        if(maMax[i]!=0){
            maBestNotFound <- TRUE;
            while(maBestNotFound){
                if(!silent){
                    m <- m+1;
                    cat("\b");
                    cat(mSymbols[m]);
                }
                acfValues <- acf(residuals(bestModel), lag.max=max((maMax*lags)[i]*2,obsInSample/2)+1, plot=FALSE)$acf[-1];
                maTest[i] <- which.max(abs(acfValues[c(1:maMax[i])*lags[i]]));

                testModel <- try(do.call(fitter,
                                         c(base_call,
                                           list(orders=list(ar=arBest, i=iBest, ma=maTest),
                                                constant=constantValue),
                                           dots)),
                                 silent=TRUE);
                ICValue <- if(inherits(testModel,"try-error")) Inf else IC(testModel);

                if(silentDebug){
                    cat("\nTested MA:", maTest, "IC:", ICValue);
                }
                if(!is.na(ICValue) && ICValue < bestIC){
                    maBest[i] <- maTest[i];
                    bestIC <- ICValue;
                    bestModel <- testModel;
                }
                else{
                    maTest[i] <- maBest[i];
                    maBestNotFound[] <- FALSE;
                }
            }
        }

        if(arMax[i]!=0){
            arBestNotFound <- TRUE;
            while(arBestNotFound){
                if(!silent){
                    m <- m+1;
                    cat("\b");
                    cat(mSymbols[m]);
                }
                pacfValues <- pacf(residuals(bestModel), lag.max=max((arMax*lags)[i]*2,obsInSample/2)+1, plot=FALSE)$acf;
                arTest[i] <- which.max(abs(pacfValues[c(1:arMax[i])*lags[i]]));

                testModel <- try(do.call(fitter,
                                         c(base_call,
                                           list(orders=list(ar=arTest, i=iBest, ma=maBest),
                                                constant=constantValue),
                                           dots)),
                                 silent=TRUE);
                ICValue <- if(inherits(testModel,"try-error")) Inf else IC(testModel);
                if(silentDebug){
                    cat("\nTested AR:", arTest, "IC:", ICValue);
                }

                if(!is.na(ICValue) && ICValue < bestIC){
                    arBest[i] <- arTest[i];
                    bestIC <- ICValue;
                    bestModel <- testModel;
                }
                else{
                    arTest[i] <- arBest[i];
                    arBestNotFound[] <- FALSE;
                }
            }
        }
    }

    # Additional checks for ARIMA(0,d,d) models
    additionalModels <- NULL;
    if(any(maMax!=0) && any(iMax!=0)){
        additionalModels <- iOrders[1:iCombinations,1:ordersLength,drop=FALSE];
        # These are IMA(d,d) candidates: the MA order is set equal to the
        # differencing one, so a row is only usable when d does not exceed maMax
        # at EVERY lag. The conditions must accumulate -- assigning them in turn
        # kept only the last lag's verdict and let through candidates asking for
        # an MA order the user excluded.
        modelsLeft <- rep(TRUE,iCombinations);
        for(i in 1:ordersLength){
            modelsLeft[] <- modelsLeft & (additionalModels[,i] <= maMax[i]);
        }
        additionalModels <- additionalModels[modelsLeft,,drop=FALSE];
    }

    # Row 1 is the all-zero differencing case and always survives the filter, so
    # a single row means there is nothing left to test.
    if(!is.null(additionalModels) && nrow(additionalModels)>1){
        imaOrdersICs <- vector("numeric",nrow(additionalModels));
        imaOrdersICs[] <- Inf;
        for(d in 2:nrow(additionalModels)){
            testModel <- try(do.call(fitter,
                                     c(base_call,
                                       list(orders=list(ar=0,
                                                        i=additionalModels[d,1:ordersLength],
                                                        ma=additionalModels[d,1:ordersLength]),
                                            constant=FALSE),
                                       dots)),
                             silent=TRUE);

            if(!inherits(testModel,"try-error")){
                imaOrdersICs[d] <- IC(testModel);
            }
            else{
                imaOrdersICs[d] <- Inf;
            }
            if(silentDebug){
                cat("\nAdditional Model:", additionalModels[d,1:ordersLength], "IC:", imaOrdersICs[d]);
            }
        }

        d <- which.min(imaOrdersICs);
        imaBest <- additionalModels[d,1:ordersLength];
        if(imaOrdersICs[d]<bestIC){
            arBest <- 0;
            iBest <- maBest <- imaBest;
            constantValue <- FALSE;
            bestModel <- do.call(fitter,
                                 c(base_call,
                                   list(orders=list(ar=0, i=iBest, ma=maBest),
                                        constant=constantValue),
                                   dots));
        }
    }

    # Refit full model if ARIMA was selected on ETS residuals
    if(is.adam(testModelETS)){
        bestModel <- do.call(fitter,
                             c(base_call,
                               list(orders=list(ar=arBest, i=iBest, ma=maBest),
                                    constant=constantValue),
                               dots));

        if(IC(bestModel) >= ICOriginal){
            bestModel <- testModelETS;
        }
    }

    if(!silent){
        cat("\nThe best ARIMA is selected. ");
    }

    bestModel$formula[[2]] <- as.name(responseName);
    colnames(bestModel$data)[1] <- responseName;
    if(holdout){
        colnames(bestModel$holdout)[1] <- responseName;
    }

    return(bestModel);
}
