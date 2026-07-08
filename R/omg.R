#' General occurrence model
#'
#' Fits two parallel ETS occurrence models (A: odds-ratio, B: inverse-odds-ratio)
#' jointly using a shared Bernoulli log-likelihood.  The combined probability
#' at each time point is \eqn{p_t = p_{At} / (p_{At} + p_{Bt})}.
#'
#' @param data Binary time series (0/1), vector or data frame.
#' @param modelA ETS model string for model A (default \code{"MNN"}).
#' @param modelB ETS model string for model B (default \code{"MNN"}).
#' @param ordersA ARIMA orders list for model A.
#' @param ordersB ARIMA orders list for model B.
#' @param constantA Logical, include constant in model A.
#' @param constantB Logical, include constant in model B.
#' @param formulaA Formula for exogenous variables in model A.
#' @param formulaB Formula for exogenous variables in model B.
#' @param regressorsA How to handle regressors in model A.
#' @param regressorsB How to handle regressors in model B.
#' @param persistenceA Persistence vector for model A.
#' @param persistenceB Persistence vector for model B.
#' @param phiA Damping parameter for model A.
#' @param phiB Damping parameter for model B.
#' @param armaA ARMA parameters for model A.
#' @param armaB ARMA parameters for model B.
#' @param etsA ETS variant for model A (\code{"conventional"} or \code{"adam"}).
#' @param etsB ETS variant for model B.
#' @param lags Seasonal lags (shared).
#' @param h Forecast horizon.
#' @param holdout If \code{TRUE}, hold out the last \code{h} observations.
#' @param initial Initialisation method (shared).
#' @param loss Loss function (shared).
#' @param ic Information criterion (shared).
#' @param bounds Parameter bounds type (shared).
#' @param model An already-fitted \code{omg} object. When supplied, the
#'   per-side parameters are lifted from \code{model$modelA} and
#'   \code{model$modelB} and no estimation is performed; passing
#'   \code{FI=TRUE} alongside computes the observed Fisher information
#'   over the joint parameter vector (the path used by
#'   \code{vcov.omg}).
#' @param silent If \code{TRUE}, suppress output.
#' @param ... Additional arguments passed to the optimiser.
#'
#' @return An object of class \code{c("omg","om","smooth")}.
#'
#' @seealso \link{om}, \link{forecast.omg}
#'
#' @examples
#' set.seed(41)
#' y <- rpois(100, 0.5)
#' m <- omg(y)
#' forecast(m, h=10)
#' @export
omg <- function(data,
                modelA = "MNN", modelB = modelA,
                ordersA = list(ar=c(0), i=c(0), ma=c(0), select=FALSE),
                ordersB = ordersA,
                constantA = FALSE, constantB = constantA,
                formulaA = NULL, formulaB = formulaA,
                regressorsA = c("use","select","adapt"),
                regressorsB = regressorsA,
                persistenceA = NULL, persistenceB = persistenceA,
                phiA = NULL, phiB = phiA,
                armaA = NULL, armaB = armaA,
                etsA = c("conventional","adam"),
                etsB = etsA,
                lags = c(frequency(data)),
                h = 0, holdout = FALSE,
                initial = c("backcasting","optimal","two-stage","complete"),
                loss = c("likelihood","MSE","MAE","HAM","LASSO","RIDGE"),
                ic = c("AICc","AIC","BIC","BICc"),
                bounds = c("usual","admissible","none"),
                model = NULL,
                silent = TRUE, ...) {

    startTime <- Sys.time()
    cl        <- match.call()

    # Capture ellipsis early so FI / stepSize / B / lb / ub passed via ...
    # are visible to the fitted-object intake below.
    ellipsis <- list(...)

    # If a fitted omg object is passed via `model`, lift its parameters out
    # of model$modelA / model$modelB and set modelDo_user="use" so the
    # optimiser is skipped. Mirrors om()'s Phase 1 intake. This is the
    # canonical entry for vcov(omg_obj): vcov re-calls omg(..., model=obj,
    # FI=TRUE, stepSize=...).
    if(is.omg(model)){
        # A-side
        modelA       <- modelType(model$modelA)
        persistenceA <- model$modelA$persistence
        phiA         <- model$modelA$phi
        armaA        <- model$modelA$arma
        ordersA      <- model$modelA$orders
        regressorsA  <- model$modelA$regressors
        constantA    <- if(is.null(model$modelA$constant)) FALSE else model$modelA$constant
        if(is.null(formulaA)) { formulaA <- formula(model$modelA) }
        # B-side
        modelB       <- modelType(model$modelB)
        persistenceB <- model$modelB$persistence
        phiB         <- model$modelB$phi
        armaB        <- model$modelB$arma
        ordersB      <- model$modelB$orders
        regressorsB  <- model$modelB$regressors
        constantB    <- if(is.null(model$modelB$constant)) FALSE else model$modelB$constant
        if(is.null(formulaB)) { formulaB <- formula(model$modelB) }
        # Shared
        if(!is.null(model$lags)) { lags <- model$lags }
        if(!is.null(model$loss)) { loss <- model$loss }
        if(!is.null(model$modelA$bounds))      { bounds  <- model$modelA$bounds }
        if(!is.null(model$modelA$initialType)) { initial <- model$modelA$initialType }
        # Joint B (concatenation: A then B).
        ellipsis$B   <- c(model$modelA$B, model$modelB$B)
        # A/B split point for the joint B. The initialiser produces empty
        # BValues when persistence and initials are "provided", so we cannot
        # rely on its nParamsA for the FI computation.
        nParamsA_use <- length(model$modelA$B)
        modelDo_user <- "use"
    } else {
        modelDo_user <- NULL
        nParamsA_use <- NULL
    }

    # Custom callable loss (same convention as adam() / om()).
    if(is.function(loss)){
        omgUserLossFunction <- loss
        loss <- "custom"
    } else {
        omgUserLossFunction <- NULL
        loss <- match.arg(loss)
    }
    # Regularisation weight for LASSO/RIDGE (mirrors adam()'s ellipsis$lambda).
    lambda <- if(is.null(ellipsis$lambda)) 0 else as.numeric(ellipsis$lambda)
    ic         <- match.arg(ic)
    bounds     <- match.arg(bounds)
    initial    <- match.arg(initial)
    regressorsA <- match.arg(regressorsA)
    # Explicit choices here — defensive against the `regressorsB = regressorsA`
    # formal default getting evaluated AFTER the line above reassigned
    # regressorsA, which would make `match.arg(regressorsB)` see choices of
    # length 1 and reject any multi-element supplied value. Mirrors etsB below.
    regressorsB <- match.arg(regressorsB, c("use","select","adapt"))
    etsA       <- match.arg(etsA)
    etsB       <- match.arg(etsB, c("conventional","adam"))
    # `ellipsis` was already populated by the fitted-object intake at the top
    # of omg() (may carry $B injected from model$modelA$B / model$modelB$B).
    # Only initialise it here if the intake didn't run.
    if(!exists("ellipsis", inherits=FALSE)) { ellipsis <- list(...) }

    adamETSA <- (etsA == "adam")
    adamETSB <- (etsB == "adam")

    #### Shared data preparation ####
    if(is.data.frame(data)) {
        yName <- colnames(data)[1]
    } else {
        yName <- paste0(deparse(substitute(data)), collapse="")
        if(length(yName) == 0 || is.null(yName)) { yName <- "y" }
    }
    modelDo <- "estimate"
    # If we lifted a fitted omg object earlier, switch to the "use" path so
    # the optimiser is skipped and (optionally) the hessian is computed.
    if(!is.null(modelDo_user)) { modelDo <- modelDo_user }

    dataChecked <- adam_checkData(data, lags, h, holdout, yName, modelDo, formulaA)
    list2env(dataChecked, envir=environment())

    lossArg <- if(loss == "likelihood") "likelihood" else loss

    # Pre-select ETS model when wildcard characters (Z/X/Y) are present
    if(grepl("[ZXYFPS]", modelA)) {
        preA  <- om(data=data, model=modelA, lags=lags, orders=ordersA,
                    constant=constantA, formula=formulaA, regressors=regressorsA,
                    occurrence="odds-ratio", loss=loss, h=0, holdout=FALSE,
                    persistence=persistenceA, phi=phiA, initial=initial,
                    arma=armaA, ic=ic, bounds=bounds, ets=etsA, silent=TRUE, ...)
        modelA <- modelType(preA)
    }
    if(grepl("[ZXYFPS]", modelB)) {
        preB  <- om(data=data, model=modelB, lags=lags, orders=ordersB,
                    constant=constantB, formula=formulaB, regressors=regressorsB,
                    occurrence="inverse-odds-ratio", loss=loss, h=0, holdout=FALSE,
                    persistence=persistenceB, phi=phiB, initial=initial,
                    arma=armaB, ic=ic, bounds=bounds, ets=etsB, silent=TRUE, ...)
        modelB <- modelType(preB)
    }

    #### parametersChecker for A and B ####
    checkerA <- parametersChecker(data=data, model=modelA, lags=lags,
                                  formulaToUse=formulaA, orders=ordersA,
                                  constant=constantA, arma=armaA,
                                  persistence=persistenceA, phi=phiA,
                                  initial=initial, distribution="dnorm",
                                  loss=lossArg,
                                  h=h, holdout=holdout, occurrence="odds-ratio",
                                  ic=ic, bounds=bounds, regressors=regressorsA,
                                  yName=yName, silent=silent, modelDo=modelDo,
                                  ellipsis=ellipsis, fast=FALSE)

    checkerB <- parametersChecker(data=data, model=modelB, lags=lags,
                                  formulaToUse=formulaB, orders=ordersB,
                                  constant=constantB, arma=armaB,
                                  persistence=persistenceB, phi=phiB,
                                  initial=initial, distribution="dnorm",
                                  loss=lossArg,
                                  h=h, holdout=holdout, occurrence="inverse-odds-ratio",
                                  ic=ic, bounds=bounds, regressors=regressorsB,
                                  yName=yName, silent=silent, modelDo=modelDo,
                                  ellipsis=ellipsis, fast=FALSE)

    if(isTRUE(checkerA$select)) {
        stop("ARIMA order selection is not supported in omg(). Specify fixed orders.", call.=FALSE)
    }
    if(isTRUE(checkerB$select)) {
        stop("ARIMA order selection is not supported in omg(). Specify fixed orders.", call.=FALSE)
    }

    #### Shared binary indicators (same data for A and B) ####
    ot         <- checkerA$ot
    otLogical  <- checkerA$otLogical
    oInSample  <- matrix(as.numeric(ot), ncol=1)
    if(holdout) {
        yHoldout <- checkerA$yHoldout
        yHoldout[] <- (yHoldout != 0) * 1
    }

    occurrenceModel <- FALSE
    yFitted         <- matrix(rep(mean(oInSample), obsInSample), ncol=1)
    refineHead      <- TRUE

    #### Optimiser settings ####
    optimSettings <- adam_checkOptimizer(ellipsis=ellipsis, loss=loss,
                                         distribution="dnorm",
                                         initialType=checkerA$initialType,
                                         lags=lags,
                                         arimaModel=checkerA$arimaModel)
    list2env(optimSettings, envir=environment())

    otLogicalInternal    <- otLogical
    otLogicalInternal[]  <- TRUE

    #### Inner joint estimator ####
    omgEstimator <- function() {

        # omgCF_local is defined at file scope (top of this file) — pure
        # function over its explicit args, reachable from both the optimiser
        # path here and from the modelDo=="use" branch for FI computation.

        # Architecture for A
        adamArchitectA <- adam_architector(
            checkerA$etsModel, checkerA$Etype, checkerA$Ttype, checkerA$Stype,
            lags, checkerA$lagsModelSeasonal,
            checkerA$xregNumber, obsInSample, checkerA$initialType,
            checkerA$arimaModel, checkerA$lagsModelARIMA,
            checkerA$xregModel, checkerA$constantRequired,
            checkerA$componentsNumberARIMA, obsAll, yIndexAll, yClasses, adamETSA)

        # Architecture for B — pad obsAll so B's lookup table covers A's loop bound
        lagsModelMaxA  <- adamArchitectA$lagsModelMax
        obsAllB_opt    <- max(obsAll, obsInSample + lagsModelMaxA)
        adamArchitectB <- adam_architector(
            checkerB$etsModel, checkerB$Etype, checkerB$Ttype, checkerB$Stype,
            lags, checkerB$lagsModelSeasonal,
            checkerB$xregNumber, obsInSample, checkerB$initialType,
            checkerB$arimaModel, checkerB$lagsModelARIMA,
            checkerB$xregModel, checkerB$constantRequired,
            checkerB$componentsNumberARIMA, obsAllB_opt, yIndexAll, yClasses, adamETSB)
        obsStatesB_opt <- max(adamArchitectB$obsStates, obsInSample + lagsModelMaxA)

        adamCppA <- adamArchitectA$adamCpp
        adamCppB <- adamArchitectB$adamCpp

        # Creator for A (Etype forced to "A")
        adamCreatedA <- adam_creator(
            checkerA$etsModel, Etype="A",
            Ttype=switch(checkerA$Ttype, "N"="N", "A"),
            Stype="A",
            adamArchitectA$modelIsTrendy, adamArchitectA$modelIsSeasonal,
            lags, adamArchitectA$lagsModel, checkerA$lagsModelARIMA,
            adamArchitectA$lagsModelAll, adamArchitectA$lagsModelMax,
            adamArchitectA$profilesRecentTable, FALSE,
            adamArchitectA$obsStates, obsInSample, obsAll,
            adamArchitectA$componentsNumberETS,
            adamArchitectA$componentsNumberETSSeasonal,
            adamArchitectA$componentsNamesETS, otLogicalInternal, ot,
            checkerA$persistence, checkerA$persistenceEstimate,
            checkerA$persistenceLevel, checkerA$persistenceLevelEstimate,
            checkerA$persistenceTrend, checkerA$persistenceTrendEstimate,
            checkerA$persistenceSeasonal, checkerA$persistenceSeasonalEstimate,
            checkerA$persistenceXreg, checkerA$persistenceXregEstimate,
            checkerA$persistenceXregProvided,
            checkerA$phi,
            checkerA$initialType, checkerA$initialEstimate,
            checkerA$initialLevel, checkerA$initialLevelEstimate,
            checkerA$initialTrend, checkerA$initialTrendEstimate,
            checkerA$initialSeasonal, checkerA$initialSeasonalEstimate,
            checkerA$initialArima, checkerA$initialArimaEstimate,
            checkerA$initialArimaNumber,
            checkerA$initialXregEstimate, checkerA$initialXregProvided,
            checkerA$arimaModel, checkerA$arRequired, checkerA$iRequired,
            checkerA$maRequired, checkerA$armaParameters,
            checkerA$arOrders, checkerA$iOrders, checkerA$maOrders,
            checkerA$componentsNumberARIMA, checkerA$componentsNamesARIMA,
            checkerA$xregModel, checkerA$xregModelInitials, checkerA$xregData,
            checkerA$xregNumber, checkerA$xregNames,
            checkerA$xregParametersPersistence,
            checkerA$constantRequired, checkerA$constantEstimate,
            checkerA$constantValue, checkerA$constantName,
            adamCppA,
            checkerA$arEstimate, checkerA$maEstimate, smoother,
            checkerA$nonZeroARI, checkerA$nonZeroMA)

        # Creator for B (Etype forced to "A")
        adamCreatedB <- adam_creator(
            checkerB$etsModel, Etype="A",
            Ttype=switch(checkerB$Ttype, "N"="N", "A"),
            Stype="A",
            adamArchitectB$modelIsTrendy, adamArchitectB$modelIsSeasonal,
            lags, adamArchitectB$lagsModel, checkerB$lagsModelARIMA,
            adamArchitectB$lagsModelAll, adamArchitectB$lagsModelMax,
            adamArchitectB$profilesRecentTable, FALSE,
            obsStatesB_opt, obsInSample, obsAll,
            adamArchitectB$componentsNumberETS,
            adamArchitectB$componentsNumberETSSeasonal,
            adamArchitectB$componentsNamesETS, otLogicalInternal, ot,
            checkerB$persistence, checkerB$persistenceEstimate,
            checkerB$persistenceLevel, checkerB$persistenceLevelEstimate,
            checkerB$persistenceTrend, checkerB$persistenceTrendEstimate,
            checkerB$persistenceSeasonal, checkerB$persistenceSeasonalEstimate,
            checkerB$persistenceXreg, checkerB$persistenceXregEstimate,
            checkerB$persistenceXregProvided,
            checkerB$phi,
            checkerB$initialType, checkerB$initialEstimate,
            checkerB$initialLevel, checkerB$initialLevelEstimate,
            checkerB$initialTrend, checkerB$initialTrendEstimate,
            checkerB$initialSeasonal, checkerB$initialSeasonalEstimate,
            checkerB$initialArima, checkerB$initialArimaEstimate,
            checkerB$initialArimaNumber,
            checkerB$initialXregEstimate, checkerB$initialXregProvided,
            checkerB$arimaModel, checkerB$arRequired, checkerB$iRequired,
            checkerB$maRequired, checkerB$armaParameters,
            checkerB$arOrders, checkerB$iOrders, checkerB$maOrders,
            checkerB$componentsNumberARIMA, checkerB$componentsNamesARIMA,
            checkerB$xregModel, checkerB$xregModelInitials, checkerB$xregData,
            checkerB$xregNumber, checkerB$xregNames,
            checkerB$xregParametersPersistence,
            checkerB$constantRequired, checkerB$constantEstimate,
            checkerB$constantValue, checkerB$constantName,
            adamCppB,
            checkerB$arEstimate, checkerB$maEstimate, smoother,
            checkerB$nonZeroARI, checkerB$nonZeroMA)

        # Initial transforms
        adamCreatedA$matVt <- om_initial_transform(
            adamCreatedA$matVt, "odds-ratio", checkerA$Etype,
            checkerA$Ttype, checkerA$Stype, checkerA$etsModel,
            adamArchitectA$modelIsTrendy, adamArchitectA$modelIsSeasonal,
            checkerA$initialLevelEstimate, checkerA$initialTrendEstimate,
            checkerA$initialSeasonalEstimate,
            adamArchitectA$componentsNumberETS,
            adamArchitectA$componentsNumberETSNonSeasonal,
            adamArchitectA$componentsNumberETSSeasonal,
            adamArchitectA$lagsModel, adamArchitectA$lagsModelMax,
            checkerA$lagsModelSeasonal, obsInSample, ot,
            checkerA$arimaModel, checkerA$componentsNumberARIMA,
            checkerA$initialArimaEstimate, checkerA$initialArimaNumber,
            checkerA$xregModel, checkerA$xregNumber,
            checkerA$initialXregEstimate,
            checkerA$constantRequired, checkerA$constantEstimate)

        adamCreatedB$matVt <- om_initial_transform(
            adamCreatedB$matVt, "inverse-odds-ratio", checkerB$Etype,
            checkerB$Ttype, checkerB$Stype, checkerB$etsModel,
            adamArchitectB$modelIsTrendy, adamArchitectB$modelIsSeasonal,
            checkerB$initialLevelEstimate, checkerB$initialTrendEstimate,
            checkerB$initialSeasonalEstimate,
            adamArchitectB$componentsNumberETS,
            adamArchitectB$componentsNumberETSNonSeasonal,
            adamArchitectB$componentsNumberETSSeasonal,
            adamArchitectB$lagsModel, adamArchitectB$lagsModelMax,
            checkerB$lagsModelSeasonal, obsInSample, ot,
            checkerB$arimaModel, checkerB$componentsNumberARIMA,
            checkerB$initialArimaEstimate, checkerB$initialArimaNumber,
            checkerB$xregModel, checkerB$xregNumber,
            checkerB$initialXregEstimate,
            checkerB$constantRequired, checkerB$constantEstimate)

        # Initial B vectors
        BValuesA <- adam_initialiser(
            checkerA$etsModel, checkerA$Etype, checkerA$Ttype, checkerA$Stype,
            adamArchitectA$modelIsTrendy, adamArchitectA$modelIsSeasonal,
            adamArchitectA$componentsNumberETSNonSeasonal,
            adamArchitectA$componentsNumberETSSeasonal,
            adamArchitectA$componentsNumberETS,
            lags, adamArchitectA$lagsModel, checkerA$lagsModelSeasonal,
            checkerA$lagsModelARIMA, adamArchitectA$lagsModelMax,
            adamCreatedA$matVt,
            checkerA$persistenceEstimate, checkerA$persistenceLevelEstimate,
            checkerA$persistenceTrendEstimate,
            checkerA$persistenceSeasonalEstimate,
            checkerA$persistenceXregEstimate,
            checkerA$phiEstimate, checkerA$initialType, checkerA$initialEstimate,
            checkerA$initialLevelEstimate, checkerA$initialTrendEstimate,
            checkerA$initialSeasonalEstimate,
            checkerA$initialArimaEstimate, checkerA$initialXregEstimate,
            checkerA$arimaModel, checkerA$arRequired, checkerA$maRequired,
            checkerA$arEstimate, checkerA$maEstimate,
            checkerA$arOrders, checkerA$maOrders,
            checkerA$componentsNumberARIMA, checkerA$componentsNamesARIMA,
            checkerA$initialArimaNumber,
            checkerA$xregModel, checkerA$xregNumber,
            checkerA$xregParametersEstimated, checkerA$xregParametersPersistence,
            checkerA$constantEstimate, checkerA$constantName,
            checkerA$otherParameterEstimate,
            adamCppA,
            etsA, bounds, ot, otLogicalInternal,
            checkerA$iOrders, checkerA$armaParameters, checkerA$other)

        BValuesB <- adam_initialiser(
            checkerB$etsModel, checkerB$Etype, checkerB$Ttype, checkerB$Stype,
            adamArchitectB$modelIsTrendy, adamArchitectB$modelIsSeasonal,
            adamArchitectB$componentsNumberETSNonSeasonal,
            adamArchitectB$componentsNumberETSSeasonal,
            adamArchitectB$componentsNumberETS,
            lags, adamArchitectB$lagsModel, checkerB$lagsModelSeasonal,
            checkerB$lagsModelARIMA, adamArchitectB$lagsModelMax,
            adamCreatedB$matVt,
            checkerB$persistenceEstimate, checkerB$persistenceLevelEstimate,
            checkerB$persistenceTrendEstimate,
            checkerB$persistenceSeasonalEstimate,
            checkerB$persistenceXregEstimate,
            checkerB$phiEstimate, checkerB$initialType, checkerB$initialEstimate,
            checkerB$initialLevelEstimate, checkerB$initialTrendEstimate,
            checkerB$initialSeasonalEstimate,
            checkerB$initialArimaEstimate, checkerB$initialXregEstimate,
            checkerB$arimaModel, checkerB$arRequired, checkerB$maRequired,
            checkerB$arEstimate, checkerB$maEstimate,
            checkerB$arOrders, checkerB$maOrders,
            checkerB$componentsNumberARIMA, checkerB$componentsNamesARIMA,
            checkerB$initialArimaNumber,
            checkerB$xregModel, checkerB$xregNumber,
            checkerB$xregParametersEstimated, checkerB$xregParametersPersistence,
            checkerB$constantEstimate, checkerB$constantName,
            checkerB$otherParameterEstimate,
            adamCppB,
            etsB, bounds, ot, otLogicalInternal,
            checkerB$iOrders, checkerB$armaParameters, checkerB$other)

        # Capture user-supplied B / lb / ub from ellipses BEFORE the joint
        # defaults shadow them (B, lb, ub were extracted by
        # adam_checkOptimizer() and list2env()'d into the surrounding omg()
        # frame; the joint default assignments below would otherwise mask
        # them as local variables).
        userB  <- B
        userLb <- lb
        userUb <- ub

        B_A <- BValuesA$B
        B_B <- BValuesB$B
        nParamsA <- length(B_A)
        B_used  <- c(B_A, B_B)
        lb      <- c(BValuesA$Bl, BValuesB$Bl)
        ub      <- c(BValuesA$Bu, BValuesB$Bu)

        # Override with user-supplied values when given. B is the JOINT
        # vector; named B is name-matched onto B_used, unnamed B is assigned
        # positionally — mirrors the adam.R/om.R pattern.
        if(!is.null(userB)){
            if(!is.null(names(userB))){
                userB <- userB[names(userB) %in% names(B_used)]
                B_used[names(userB)] <- userB
            } else {
                B_used[] <- userB
            }
        }
        if(!is.null(userLb)){
            lb[] <- userLb
        }
        if(!is.null(userUb)){
            ub[] <- userUb
        }

        # Mixed model checks
        EtypeA <- checkerA$Etype; TtypeA <- checkerA$Ttype; StypeA <- checkerA$Stype
        EtypeB <- checkerB$Etype; TtypeB <- checkerB$Ttype; StypeB <- checkerB$Stype

        if((EtypeA=="A" && TtypeA=="A" && StypeA=="M") ||
           (EtypeA=="A" && TtypeA=="M" && StypeA=="A") ||
           (EtypeA=="M" && TtypeA=="A" && StypeA=="A") ||
           (EtypeA=="A" && TtypeA=="M" && StypeA=="N") ||
           (EtypeA=="M" && TtypeA=="M" && StypeA=="A") ||
           (EtypeA=="M" && TtypeA=="N" && StypeA=="A") ||
           (EtypeA=="A" && TtypeA=="N" && StypeA=="M")) {
            B_used[seq_len(nParamsA)] <- 0
        }
        if((EtypeB=="A" && TtypeB=="A" && StypeB=="M") ||
           (EtypeB=="A" && TtypeB=="M" && StypeB=="A") ||
           (EtypeB=="M" && TtypeB=="A" && StypeB=="A") ||
           (EtypeB=="A" && TtypeB=="M" && StypeB=="N") ||
           (EtypeB=="M" && TtypeB=="M" && StypeB=="A") ||
           (EtypeB=="M" && TtypeB=="N" && StypeB=="A") ||
           (EtypeB=="A" && TtypeB=="N" && StypeB=="M")) {
            B_used[seq_len(length(B_B)) + nParamsA] <- 0
        }

        # ARIMA companion matrices for A
        if(checkerA$arimaModel) {
            arPolynomialMatrixA <- matrix(0, checkerA$arOrders %*% lags,
                                          checkerA$arOrders %*% lags)
            if(nrow(arPolynomialMatrixA) > 1) {
                arPolynomialMatrixA[2:nrow(arPolynomialMatrixA)-1,
                                    2:nrow(arPolynomialMatrixA)] <-
                    diag(nrow(arPolynomialMatrixA) - 1)
            }
            maPolynomialMatrixA <- matrix(0, checkerA$maOrders %*% lags,
                                          checkerA$maOrders %*% lags)
            if(nrow(maPolynomialMatrixA) > 1) {
                maPolynomialMatrixA[2:nrow(maPolynomialMatrixA)-1,
                                    2:nrow(maPolynomialMatrixA)] <-
                    diag(nrow(maPolynomialMatrixA) - 1)
            }
        } else {
            arPolynomialMatrixA <- maPolynomialMatrixA <- NULL
        }

        # ARIMA companion matrices for B
        if(checkerB$arimaModel) {
            arPolynomialMatrixB <- matrix(0, checkerB$arOrders %*% lags,
                                          checkerB$arOrders %*% lags)
            if(nrow(arPolynomialMatrixB) > 1) {
                arPolynomialMatrixB[2:nrow(arPolynomialMatrixB)-1,
                                    2:nrow(arPolynomialMatrixB)] <-
                    diag(nrow(arPolynomialMatrixB) - 1)
            }
            maPolynomialMatrixB <- matrix(0, checkerB$maOrders %*% lags,
                                          checkerB$maOrders %*% lags)
            if(nrow(maPolynomialMatrixB) > 1) {
                maPolynomialMatrixB[2:nrow(maPolynomialMatrixB)-1,
                                    2:nrow(maPolynomialMatrixB)] <-
                    diag(nrow(maPolynomialMatrixB) - 1)
            }
        } else {
            arPolynomialMatrixB <- maPolynomialMatrixB <- NULL
        }

        # B-side scalars for omfitGeneral
        nNonSeasonalB <- adamArchitectB$componentsNumberETSNonSeasonal
        nSeasonalB    <- adamArchitectB$componentsNumberETSSeasonal
        nETSB         <- adamArchitectB$componentsNumberETS
        nArimaB       <- checkerB$componentsNumberARIMA
        nXregB        <- checkerB$xregNumber
        nComponentsB  <- length(adamArchitectB$lagsModelAll)

        # Explicit nloptr args
        nloptrArgs <- list(
            # A-side
            etsModelA=checkerA$etsModel, EtypeA=EtypeA, TtypeA=TtypeA, StypeA=StypeA,
            modelIsTrendyA=adamArchitectA$modelIsTrendy,
            modelIsSeasonalA=adamArchitectA$modelIsSeasonal,
            componentsNumberETSA=adamArchitectA$componentsNumberETS,
            componentsNumberETSNonSeasonalA=adamArchitectA$componentsNumberETSNonSeasonal,
            componentsNumberETSSeasonalA=adamArchitectA$componentsNumberETSSeasonal,
            componentsNumberARIMAA=checkerA$componentsNumberARIMA,
            lags=lags,
            lagsModelA=adamArchitectA$lagsModel,
            lagsModelMaxA=adamArchitectA$lagsModelMax,
            lagsModelAllA=adamArchitectA$lagsModelAll,
            indexLookupTableA=adamArchitectA$indexLookupTable,
            profilesRecentTableA=adamArchitectA$profilesRecentTable,
            matVtA=adamCreatedA$matVt, matWtA=adamCreatedA$matWt,
            matFA=adamCreatedA$matF, vecGA=adamCreatedA$vecG,
            persistenceEstimateA=checkerA$persistenceEstimate,
            persistenceLevelEstimateA=checkerA$persistenceLevelEstimate,
            persistenceTrendEstimateA=checkerA$persistenceTrendEstimate,
            persistenceSeasonalEstimateA=checkerA$persistenceSeasonalEstimate,
            persistenceXregEstimateA=checkerA$persistenceXregEstimate,
            phiEstimateA=checkerA$phiEstimate,
            initialTypeA=checkerA$initialType, initialEstimateA=checkerA$initialEstimate,
            initialLevelEstimateA=checkerA$initialLevelEstimate,
            initialTrendEstimateA=checkerA$initialTrendEstimate,
            initialSeasonalEstimateA=checkerA$initialSeasonalEstimate,
            initialArimaEstimateA=checkerA$initialArimaEstimate,
            initialXregEstimateA=checkerA$initialXregEstimate,
            initialArimaNumberA=checkerA$initialArimaNumber,
            arimaModelA=checkerA$arimaModel,
            arEstimateA=checkerA$arEstimate, maEstimateA=checkerA$maEstimate,
            arOrdersA=checkerA$arOrders, iOrdersA=checkerA$iOrders,
            maOrdersA=checkerA$maOrders,
            arRequiredA=checkerA$arRequired, maRequiredA=checkerA$maRequired,
            armaParametersA=checkerA$armaParameters,
            nonZeroARIA=checkerA$nonZeroARI, nonZeroMAA=checkerA$nonZeroMA,
            arimaPolynomialsA=adamCreatedA$arimaPolynomials,
            arPolynomialMatrixA=arPolynomialMatrixA,
            maPolynomialMatrixA=maPolynomialMatrixA,
            xregModelA=checkerA$xregModel, xregNumberA=checkerA$xregNumber,
            xregParametersMissingA=checkerA$xregParametersMissing,
            xregParametersIncludedA=checkerA$xregParametersIncluded,
            xregParametersEstimatedA=checkerA$xregParametersEstimated,
            xregParametersPersistenceA=checkerA$xregParametersPersistence,
            constantRequiredA=checkerA$constantRequired,
            constantEstimateA=checkerA$constantEstimate,
            adamCppA=adamCppA,
            # B-side
            etsModelB=checkerB$etsModel, EtypeB=EtypeB, TtypeB=TtypeB, StypeB=StypeB,
            modelIsTrendyB=adamArchitectB$modelIsTrendy,
            modelIsSeasonalB=adamArchitectB$modelIsSeasonal,
            componentsNumberETSB=adamArchitectB$componentsNumberETS,
            componentsNumberETSNonSeasonalB=adamArchitectB$componentsNumberETSNonSeasonal,
            componentsNumberETSSeasonalB=adamArchitectB$componentsNumberETSSeasonal,
            componentsNumberARIMAB=checkerB$componentsNumberARIMA,
            lagsModelB=adamArchitectB$lagsModel,
            lagsModelMaxB=adamArchitectB$lagsModelMax,
            lagsModelAllB=adamArchitectB$lagsModelAll,
            indexLookupTableB=adamArchitectB$indexLookupTable,
            profilesRecentTableB=adamArchitectB$profilesRecentTable,
            matVtB=adamCreatedB$matVt, matWtB=adamCreatedB$matWt,
            matFB=adamCreatedB$matF, vecGB=adamCreatedB$vecG,
            persistenceEstimateB=checkerB$persistenceEstimate,
            persistenceLevelEstimateB=checkerB$persistenceLevelEstimate,
            persistenceTrendEstimateB=checkerB$persistenceTrendEstimate,
            persistenceSeasonalEstimateB=checkerB$persistenceSeasonalEstimate,
            persistenceXregEstimateB=checkerB$persistenceXregEstimate,
            phiEstimateB=checkerB$phiEstimate,
            initialTypeB=checkerB$initialType, initialEstimateB=checkerB$initialEstimate,
            initialLevelEstimateB=checkerB$initialLevelEstimate,
            initialTrendEstimateB=checkerB$initialTrendEstimate,
            initialSeasonalEstimateB=checkerB$initialSeasonalEstimate,
            initialArimaEstimateB=checkerB$initialArimaEstimate,
            initialXregEstimateB=checkerB$initialXregEstimate,
            initialArimaNumberB=checkerB$initialArimaNumber,
            arimaModelB=checkerB$arimaModel,
            arEstimateB=checkerB$arEstimate, maEstimateB=checkerB$maEstimate,
            arOrdersB=checkerB$arOrders, iOrdersB=checkerB$iOrders,
            maOrdersB=checkerB$maOrders,
            arRequiredB=checkerB$arRequired, maRequiredB=checkerB$maRequired,
            armaParametersB=checkerB$armaParameters,
            nonZeroARIB=checkerB$nonZeroARI, nonZeroMAB=checkerB$nonZeroMA,
            arimaPolynomialsB=adamCreatedB$arimaPolynomials,
            arPolynomialMatrixB=arPolynomialMatrixB,
            maPolynomialMatrixB=maPolynomialMatrixB,
            xregModelB=checkerB$xregModel, xregNumberB=checkerB$xregNumber,
            xregParametersMissingB=checkerB$xregParametersMissing,
            xregParametersIncludedB=checkerB$xregParametersIncluded,
            xregParametersEstimatedB=checkerB$xregParametersEstimated,
            xregParametersPersistenceB=checkerB$xregParametersPersistence,
            constantRequiredB=checkerB$constantRequired,
            constantEstimateB=checkerB$constantEstimate,
            adamCppB=adamCppB,
            # B-side scalars for omfitGeneral
            nNonSeasonalB=nNonSeasonalB, nSeasonalB=nSeasonalB, nETSB=nETSB,
            nArimaB=nArimaB, nXregB=nXregB, nComponentsB=nComponentsB,
            adamETSB_flag=adamETSB,
            # Shared
            bounds=bounds, regressors=regressorsA,
            ot=ot, otLogical=otLogical, obsInSample=obsInSample,
            nIterations=nIterations, refineHead=refineHead,
            nParamsA=nParamsA,
            loss=loss, lossFunction=omgUserLossFunction, lambda=lambda)

        # --------------------------------------------------------------
        # FI placeholder. Populated when modelDo=="use" and ellipsis$FI is
        # TRUE; otherwise stays NULL. Mirrors the FI path in om()'s use
        # branch — hessian of -logLik (= what omgCF_local returns) IS the
        # observed Fisher Information.
        # --------------------------------------------------------------
        FIMatrixUse <- NULL

        if(identical(modelDo, "use")) {
            # No optimisation. Joint B comes straight from the fitted-object
            # intake at the top of omg(): ellipsis$B was set to
            # c(model$modelA$B, model$modelB$B). DO NOT use userB here —
            # the name-match override block above filters userB against
            # B_used's names, which is empty when persistence and initials
            # arrive as "provided" from the fitted model.
            B_use <- if(!is.null(ellipsis$B)) ellipsis$B else B_used
            res <- list(solution=as.numeric(B_use),
                        objective=NA_real_)

            if(isTRUE(ellipsis$FI) && length(B_use) > 0) {
                stepSize <- if(is.null(ellipsis$stepSize)) {
                    .Machine$double.eps^(1/4)
                } else {
                    ellipsis$stepSize
                }
                Bnames <- names(B_use)
                if(is.null(Bnames)) {
                    Bnames <- c(names(BValuesA$B), names(BValuesB$B))
                    names(B_use) <- Bnames
                }
                # Split into A-half and B-half by position, and derive the
                # *EstimateFI flags from each half's names — exactly the
                # same name-matching pattern as om()'s use-branch FI block.
                .deriveFI <- function(half_names, nCompSeas, archHas_arima, archHas_xreg,
                                      xreg_names) {
                    pLvl <- any(half_names == "alpha")
                    pTrd <- any(half_names == "beta")
                    if(any(substr(half_names, 1, 5) == "gamma")) {
                        gmsk <- substr(half_names, 1, 5) == "gamma"
                        if(sum(gmsk) == 1) {
                            pSea <- TRUE
                        } else {
                            pSea <- vector("logical", nCompSeas)
                            pSea[as.numeric(substr(half_names, 6, 6)[gmsk])] <- TRUE
                        }
                    } else { pSea <- FALSE }
                    pXrg   <- any(substr(half_names, 1, 5) == "delta")
                    pEst   <- any(c(pLvl, pTrd, pSea, pXrg))
                    phiEst <- any(half_names == "phi")
                    iLvl   <- any(half_names == "level")
                    iTrd   <- any(half_names == "trend")
                    if(any(substr(half_names, 1, 8) == "seasonal")) {
                        sn   <- half_names[substr(half_names, 1, 8) == "seasonal"]
                        iSea <- vector("logical", nCompSeas)
                        if(any(substr(sn, 1, 9) == "seasonal_")) {
                            iSea[] <- TRUE
                        } else {
                            iSea[unique(as.numeric(substr(sn, 9, 9)))] <- TRUE
                        }
                    } else { iSea <- FALSE }
                    iAri <- if(archHas_arima) any(substr(half_names, 1, 10) == "ARIMAState") else FALSE
                    iXrg <- if(archHas_xreg)  any(xreg_names %in% half_names) else FALSE
                    iEst <- any(c(iLvl, iTrd, iSea, iAri, iXrg))
                    list(pEst=pEst, pLvl=pLvl, pTrd=pTrd, pSea=pSea, pXrg=pXrg,
                         phiEst=phiEst, iLvl=iLvl, iTrd=iTrd, iSea=iSea,
                         iAri=iAri, iXrg=iXrg, iEst=iEst)
                }

                # Use the fitted-object split point when available — the
                # initialiser's nParamsA is 0 for the "use" path (everything
                # provided), so it would put all of B into the B-side half.
                nParamsA_FI <- if(!is.null(nParamsA_use)) nParamsA_use else nParamsA
                names_A <- Bnames[seq_len(nParamsA_FI)]
                names_B <- Bnames[seq_len(length(B_use) - nParamsA_FI) + nParamsA_FI]

                xnamesA <- if(checkerA$xregModel) colnames(checkerA$xregData) else character(0)
                xnamesB <- if(checkerB$xregModel) colnames(checkerB$xregData) else character(0)

                fA <- .deriveFI(names_A, adamArchitectA$componentsNumberETSSeasonal,
                                checkerA$arimaModel, checkerA$xregModel, xnamesA)
                fB <- .deriveFI(names_B, adamArchitectB$componentsNumberETSSeasonal,
                                checkerB$arimaModel, checkerB$xregModel, xnamesB)

                iTypeAFI <- if(fA$iEst) "optimal" else checkerA$initialType
                iTypeBFI <- if(fB$iEst) "optimal" else checkerB$initialType

                nlaFI <- nloptrArgs
                # A-side overrides
                nlaFI$persistenceEstimateA         <- fA$pEst
                nlaFI$persistenceLevelEstimateA    <- fA$pLvl
                nlaFI$persistenceTrendEstimateA    <- fA$pTrd
                nlaFI$persistenceSeasonalEstimateA <- fA$pSea
                nlaFI$persistenceXregEstimateA     <- fA$pXrg
                nlaFI$phiEstimateA                 <- fA$phiEst
                nlaFI$initialTypeA                 <- iTypeAFI
                nlaFI$initialEstimateA             <- fA$iEst
                nlaFI$initialLevelEstimateA        <- fA$iLvl
                nlaFI$initialTrendEstimateA        <- fA$iTrd
                nlaFI$initialSeasonalEstimateA     <- fA$iSea
                nlaFI$initialArimaEstimateA        <- fA$iAri
                nlaFI$initialXregEstimateA         <- fA$iXrg
                # B-side overrides
                nlaFI$persistenceEstimateB         <- fB$pEst
                nlaFI$persistenceLevelEstimateB    <- fB$pLvl
                nlaFI$persistenceTrendEstimateB    <- fB$pTrd
                nlaFI$persistenceSeasonalEstimateB <- fB$pSea
                nlaFI$persistenceXregEstimateB     <- fB$pXrg
                nlaFI$phiEstimateB                 <- fB$phiEst
                nlaFI$initialTypeB                 <- iTypeBFI
                nlaFI$initialEstimateB             <- fB$iEst
                nlaFI$initialLevelEstimateB        <- fB$iLvl
                nlaFI$initialTrendEstimateB        <- fB$iTrd
                nlaFI$initialSeasonalEstimateB     <- fB$iSea
                nlaFI$initialArimaEstimateB        <- fB$iAri
                nlaFI$initialXregEstimateB         <- fB$iXrg
                # Override the joint split point so omgCF_local splits B at
                # the fitted-model boundary, not at the (zero-length)
                # initialiser boundary.
                nlaFI$nParamsA <- nParamsA_FI
                # Disable bounds so omgCF_local never short-circuits with
                # 1e+300 during the hessian probes.
                nlaFI$bounds <- "none"
                if(checkerA$arimaModel) {
                    nlaFI$arPolynomialMatrixA <- NULL
                    nlaFI$maPolynomialMatrixA <- NULL
                }
                if(checkerB$arimaModel) {
                    nlaFI$arPolynomialMatrixB <- NULL
                    nlaFI$maPolynomialMatrixB <- NULL
                }

                CFAtOptimum <- do.call(omgCF_local, c(list(B=B_use), nlaFI))
                omgCF_for_FI <- function(B) {
                    names(B) <- Bnames
                    val <- tryCatch(suppressWarnings(
                                        do.call(omgCF_local, c(list(B=B), nlaFI))),
                                    error = function(e) CFAtOptimum + 1e6)
                    if(!is.finite(val)) { val <- CFAtOptimum + 1e6 }
                    return(val)
                }
                FIMatrixUse <- try(suppressWarnings(
                                       hessianCpp(omgCF_for_FI, B_use, h=stepSize)),
                                   silent=TRUE)
                if(inherits(FIMatrixUse, "try-error") ||
                   any(!is.finite(FIMatrixUse))) {
                    FIMatrixUse <- NULL
                } else {
                    colnames(FIMatrixUse) <- Bnames
                    rownames(FIMatrixUse) <- Bnames
                }
            }
            # Fall through to the standard B_joint / return(list(...)) below.
        } else if(length(B_used) == 0){
            res <- list(solution=B_used,
                        objective=do.call(omgCF_local, c(list(B=B_used), nloptrArgs)));
        } else {
            maxevalUsed <- if(is.null(maxeval)) length(B_used) * 40L else maxeval
            res <- suppressWarnings(
                do.call(nloptr,
                        c(list(x0=B_used, eval_f=omgCF_local, lb=lb, ub=ub,
                               opts=list(algorithm=algorithm, xtol_rel=xtol_rel, xtol_abs=xtol_abs,
                                         ftol_rel=ftol_rel, ftol_abs=ftol_abs,
                                         maxeval=maxevalUsed, maxtime=maxtime,
                                         print_level=print_level)),
                          nloptrArgs)))
            res$call <- quote(nloptr(x0=B_used, eval_f=omgCF_local, lb=lb, ub=ub, opts=opts));

            # Retry from a small-persistence safe point if the first run hit
            # the infeasibility plateau, but only when the user did NOT supply
            # their own B — otherwise their B is the authoritative starting
            # point. Mirrors the failsafe in om()'s retry block: all params
            # set to 0.001 with the two leading alphas (A-side and B-side)
            # bumped to 0.01, which keeps the multiplicative recursion stable.
            if(is.null(userB) && (is.infinite(res$objective) || res$objective == 1e+300)) {
                B_used[] <- 0.001
                B_used[1] <- 0.01                # alpha for A-side
                B_used[nParamsA + 1] <- 0.01     # alpha for B-side
                res <- suppressWarnings(
                    do.call(nloptr,
                            c(list(x0=B_used, eval_f=omgCF_local, lb=lb, ub=ub,
                                   opts=list(algorithm=algorithm, xtol_rel=xtol_rel, xtol_abs=xtol_abs,
                                             ftol_rel=ftol_rel, ftol_abs=ftol_abs,
                                             maxeval=maxevalUsed, maxtime=maxtime,
                                             print_level=print_level)),
                              nloptrArgs)))
                res$call <- quote(nloptr(x0=B_used, eval_f=omgCF_local, lb=lb, ub=ub, opts=opts));
            }
        }

        B_joint <- res$solution
        names(B_joint) <- c(names(BValuesA$B), names(BValuesB$B))
        CFValue <- res$objective

        return(list(
            B_A       = B_joint[seq_len(nParamsA)],
            B_B       = B_joint[seq_len(length(B_joint) - nParamsA) + nParamsA],
            CFValue   = CFValue,
            logLikValue = -CFValue,
            nParamsA  = nParamsA,
            nParamsB  = length(B_joint) - nParamsA,
            adamArchitectA = adamArchitectA,
            adamArchitectB = adamArchitectB,
            adamCreatedA   = adamCreatedA,
            adamCreatedB   = adamCreatedB,
            # The exact arg-set that nloptr (and every CF call) saw. Made
            # available downstream so any future per-side use can pull from
            # the same source the optimiser used. adamCppA/adamCppB are not
            # returned separately — they live in adamArchitectA/B$adamCpp.
            nloptrArgs = nloptrArgs,
            # Observed Fisher Information at the supplied B. Non-NULL only
            # when modelDo=="use" + ellipsis$FI=TRUE; otherwise NULL.
            FI = FIMatrixUse))
    }

    #### Final fit ####

    omgFinalFit <- function(res, hLocal=0) {
        checker       <- res$checker
        adamArchitect <- res$adamArchitect
        adamCreated   <- res$adamCreated
        occurrence    <- res$occurrence
        regressors    <- res$regressors
        occurrenceChar <- if(occurrence == "odds-ratio") "o" else "i"

        adamFilled <- adam_filler(res$B,
                                  checker$etsModel, checker$Etype, checker$Ttype, checker$Stype,
                                  checker$modelIsTrendy, checker$modelIsSeasonal,
                                  adamArchitect$componentsNumberETS,
                                  adamArchitect$componentsNumberETSNonSeasonal,
                                  adamArchitect$componentsNumberETSSeasonal,
                                  checker$componentsNumberARIMA,
                                  lags, adamArchitect$lagsModel, adamArchitect$lagsModelMax,
                                  adamCreated$matVt, adamCreated$matWt, adamCreated$matF, adamCreated$vecG,
                                  checker$persistenceEstimate, checker$persistenceLevelEstimate,
                                  checker$persistenceTrendEstimate, checker$persistenceSeasonalEstimate,
                                  checker$persistenceXregEstimate, checker$phiEstimate,
                                  checker$initialType, checker$initialEstimate,
                                  checker$initialLevelEstimate, checker$initialTrendEstimate,
                                  checker$initialSeasonalEstimate, checker$initialArimaEstimate,
                                  checker$initialXregEstimate,
                                  checker$arimaModel, checker$arEstimate, checker$maEstimate,
                                  checker$arOrders, checker$iOrders, checker$maOrders,
                                  checker$arRequired, checker$maRequired, checker$armaParameters,
                                  checker$nonZeroARI, checker$nonZeroMA, adamCreated$arimaPolynomials,
                                  checker$xregModel, checker$xregNumber,
                                  checker$xregParametersMissing, checker$xregParametersIncluded,
                                  checker$xregParametersEstimated, checker$xregParametersPersistence,
                                  checker$constantEstimate, adamArchitect$adamCpp,
                                  checker$constantRequired, checker$initialArimaNumber)

        prof <- adamFilled$matVt[, seq_len(adamArchitect$lagsModelMax), drop=FALSE]
        adamFitted <- adamArchitect$adamCpp$fit(
            adamFilled$matVt, adamFilled$matWt, adamFilled$matF, adamFilled$vecG,
            adamArchitect$indexLookupTable, prof,
            as.numeric(ot), as.numeric(ot),
            any(checker$initialType == c("complete","backcasting")),
            nIterations, refineHead, occurrenceChar)

        yFitted <- adamFitted$fitted

        if(is.null(res$logLikADAMValue)) {
            pFitted <- omLinkFunction(as.numeric(yFitted), checker$Etype, occurrence)
            ot_vec  <- as.numeric(oInSample)
            ll <- sum(ot_vec * log(pmax(pFitted, 1e-15)) +
                          (1 - ot_vec) * log(pmax(1 - pFitted, 1e-15)))
            res$logLikADAMValue <- ll
            res$CFValue <- -ll
        }

        statesRaw <- adamFitted$states[,
                                       (adamArchitect$lagsModelMax+1):ncol(adamFitted$states), drop=FALSE]
        compNames <- rownames(adamCreated$matVt)
        if(!is.null(compNames)) rownames(statesRaw) <- compNames

        if(any(yClasses == "ts")) {
            yFitted <- ts(yFitted, start=yStart, frequency=yFrequency)
            errors  <- ts(as.numeric(oInSample) - yFitted, start=yStart, frequency=yFrequency)
            matVt   <- ts(t(statesRaw), start=yStart, frequency=yFrequency)
        } else {
            yFitted <- zoo(yFitted, order.by=yInSampleIndex)
            errors  <- zoo(as.numeric(oInSample) - yFitted, order.by=yInSampleIndex)
            matVt   <- zoo(t(statesRaw), order.by=yInSampleIndex)
        }

        if(hLocal > 0) {
            yForecast <- if(any(yClasses=="ts")) {
                ts(vector("numeric", hLocal), start=yForecastStart, frequency=yFrequency)
            } else {
                zoo(vector("numeric", hLocal), order.by=yForecastIndex[1])
            }
            forecastIndexLookup <- adamArchitect$indexLookupTable[,
                                   adamArchitect$lagsModelMax + obsInSample + seq_len(hLocal), drop=FALSE]
            yForecast[] <- adamArchitect$adamCpp$forecast(
                tail(adamFilled$matWt, hLocal), adamFilled$matF,
                forecastIndexLookup, adamFitted$profile, hLocal)$forecast
        } else {
            yForecast <- NULL
        }

        modelStr  <- paste0(checker$Etype, checker$Ttype,
                            "d"[checker$phiEstimate], checker$Stype)
        modelName <- adam_model_name(
            checker$etsModel, modelStr,
            checker$xregModel, checker$arimaModel,
            checker$arOrders, checker$iOrders, checker$maOrders, lags,
            regressors, checker$constantRequired, checker$constantName,
            occurrence, adamArchitect$componentsNumberETSSeasonal,
            prefix = "o")

        vecGFinal <- adamFilled$vecG
        if(adamArchitect$componentsNumberETS > 0) {
            persistenceVec <- as.vector(vecGFinal)[seq_len(adamArchitect$componentsNumberETS)]
            names(persistenceVec) <- rownames(vecGFinal)[seq_len(adamArchitect$componentsNumberETS)]
        } else {
            persistenceVec <- numeric(0)
        }

        initialCollected <- adam_initial_collector(
            adamFitted$states[, seq_len(adamArchitect$lagsModelMax), drop=FALSE],
            checker$etsModel, checker$modelIsTrendy, checker$modelIsSeasonal,
            adamArchitect$lagsModel, adamArchitect$lagsModelMax,
            checker$initialLevelEstimate, checker$initialTrendEstimate,
            checker$initialSeasonalEstimate,
            adamArchitect$componentsNumberETSSeasonal,
            checker$arimaModel, checker$initialArimaEstimate,
            checker$initialArima, checker$initialArimaNumber,
            adamArchitect$componentsNumberETS, checker$componentsNumberARIMA,
            adamFilled$arimaPolynomials, checker$Etype,
            checker$xregModel, checker$initialXregEstimate, checker$xregNumber)

        if(checker$arimaModel && (checker$arRequired || checker$maRequired)) {
            armaParametersList <- vector("list", checker$arRequired + checker$maRequired)
            j <- 1L
            if(checker$arRequired && checker$arEstimate) {
                armaParametersList[[j]] <- res$B[nchar(names(res$B))>3 &
                                                     substr(names(res$B),1,3)=="phi"]
                names(armaParametersList)[j] <- "ar"; j <- j + 1L
            } else if(checker$arRequired) {
                armaParametersList[[j]] <- checker$armaParameters[
                    substr(names(checker$armaParameters),1,3)=="phi"]
                names(armaParametersList)[j] <- "ar"; j <- j + 1L
            }
            if(checker$maRequired && checker$maEstimate) {
                armaParametersList[[j]] <- res$B[substr(names(res$B),1,5)=="theta"]
                names(armaParametersList)[j] <- "ma"
            } else if(checker$maRequired) {
                armaParametersList[[j]] <- checker$armaParameters[
                    substr(names(checker$armaParameters),1,5)=="theta"]
                names(armaParametersList)[j] <- "ma"
            }
        } else {
            armaParametersList <- NULL
        }

        parNum <- checker$parametersNumber
        parNum[1,1] <- res$nParamEstimated
        parNum[1,5] <- sum(parNum[1,1:4])
        parNum[2,5] <- sum(parNum[2,1:4])

        subModel <- list(
            model       = modelName,
            timeElapsed = Sys.time() - startTime,
            data        = yInSample,
            fitted      = yFitted,
            residuals   = errors,
            forecast    = yForecast,
            states      = matVt,
            profile     = adamFitted$profile,
            profileInitial = prof,
            persistence = persistenceVec,
            phi         = if(checker$phiEstimate) res$B["phi"] else checker$phi,
            transition  = adamFilled$matF,
            measurement = adamFilled$matWt,
            initial     = initialCollected$initialValue,
            initialType = checker$initialType,
            initialEstimated = initialCollected$initialEstimated,
            orders      = list(ar=checker$arOrders, i=checker$iOrders, ma=checker$maOrders),
            arma        = armaParametersList,
            constant    = if(checker$constantRequired) {
                if(checker$constantEstimate) res$B[checker$constantName] else checker$constantValue
            } else NULL,
            nParam      = parNum,
            occurrence  = occurrence,
            formula     = checker$formula,
            regressors  = regressors,
            loss        = loss,
            lossValue   = res$CFValue,
            lossFunction = NULL,
            logLik      = res$logLikADAMValue,
            distribution = "plogis",
            scale       = NA,
            other       = NULL,
            B           = res$B,
            lags        = lags,
            lagsAll     = adamArchitect$lagsModelAll,
            ets         = checker$etsModel,
            res         = res,
            FI          = NULL,
            adamCpp     = adamArchitect$adamCpp,
            bounds      = bounds,
            call        = cl)

        if(holdout) {
            subModel$holdout  <- yHoldout
        }

        # Tag the sub-model with an ``omg_submodel`` class ahead of ``om``
        # so ``actuals()`` can dispatch to a custom method that returns the
        # latent (unobservable) value the sub-model was implicitly fitting,
        # rather than the binary occurrence indicator.
        class(subModel) <- c("omg_submodel","om","adam","smooth","occurrence")
        return(subModel)
    }

    #### Run estimation ####
    jointResult <- omgEstimator()

    resA <- list(
        B               = jointResult$B_A,
        nParamEstimated = jointResult$nParamsA,
        logLikADAMValue = NULL,
        CFValue         = 0,
        FI              = NULL,
        checker         = checkerA,
        adamArchitect   = jointResult$adamArchitectA,
        adamCreated     = jointResult$adamCreatedA,
        # Same arg-set the optimiser saw — available for downstream use.
        nloptrArgs      = jointResult$nloptrArgs,
        occurrence      = "odds-ratio",
        regressors      = regressorsA)

    resB <- list(
        B               = jointResult$B_B,
        nParamEstimated = jointResult$nParamsB,
        logLikADAMValue = NULL,
        CFValue         = 0,
        FI              = NULL,
        checker         = checkerB,
        adamArchitect   = jointResult$adamArchitectB,
        adamCreated     = jointResult$adamCreatedB,
        nloptrArgs      = jointResult$nloptrArgs,
        occurrence      = "inverse-odds-ratio",
        regressors      = regressorsB)

    modelA <- omgFinalFit(resA, hLocal=h)
    modelB <- omgFinalFit(resB, hLocal=h)

    EtypeA <- errorType(modelA)
    EtypeB <- errorType(modelB)

    yFittedA  <- as.vector(modelA$fitted)
    yFittedB  <- as.vector(modelB$fitted)
    yFitted   <- modelA$fitted
    yFitted[] <- omgLinkFunction(yFittedA, yFittedB, EtypeA, EtypeB)

    if(h > 0) {
        yForecast <- modelA$forecast;
        yForecast[] <- omgLinkFunction(modelA$forecast, modelB$forecast, EtypeA, EtypeB)
    }
    else {
        # h=0: mirror om()'s convention (R/om.R:294, 319) and populate
        # ``$forecast`` with a one-element ``ts(NA)`` / ``zoo(NA)``
        # placeholder. Without it, ``plot.smooth`` strips ``$forecast``
        # from the ellipsis (`ellipsis$forecast <- NULL` removes the slot)
        # and ``graphmaker`` errors with "argument 'forecast' is missing".
        if(any(yClasses == "ts")) {
            yForecast <- ts(NA, start=yForecastStart, frequency=yFrequency);
        }
        else {
            yForecast <- zoo(NA, order.by=yForecastIndex[1]);
        }
    }

    modelName <- paste0("oETS[G](", modelType(modelA), ")(", modelType(modelB), ")")

    # Wrap the in-sample series with its original class (ts / zoo) so the
    # top-level omg object's ``data`` matches what ``om(y, ...)`` stores.
    # Mirrors om.R:983-987.
    if(any(yClasses == "ts")) {
        yData <- ts(yInSample, start=yStart, frequency=yFrequency);
    } else {
        yData <- zoo(yInSample, order.by=yInSampleIndex);
    }

    result <- list(
        model       = modelName,
        modelA      = modelA,
        modelB      = modelB,
        # Store the raw in-sample series at the top level so ``actuals.omg``
        # can return it with the same class (ts / zoo / numeric) that the
        # standalone ``actuals.om`` would on a fresh ``om(y, ...)`` call.
        data        = yData,
        fitted      = yFitted,
        forecast    = yForecast,
        occurrence  = "general",
        lags        = lags,
        lossValue   = jointResult$CFValue,
        logLik      = jointResult$logLikValue,
        nParam      = {
            nParamMat <- matrix(0, 2, 5,
                                dimnames=list(c("Estimated","Provided"),
                                              c("nParamInternal","nParamXreg","nParamOccurrence",
                                                "nParamScale","nParamAll")))
            nParamMat[1,1] <- jointResult$nParamsA + jointResult$nParamsB
            nParamMat[1,5] <- nParamMat[1,1]
            nParamMat[2,1:4] <- modelA$nParam[2,1:4] + modelB$nParam[2,1:4]
            nParamMat[2,5]   <- sum(nParamMat[2,1:4])
            nParamMat
        },
        distribution = "plogis",
        loss        = loss,
        lossFunction = omgUserLossFunction,
        lambda      = lambda,
        call        = cl,
        timeElapsed = Sys.time() - startTime,
        # FI is non-NULL only when omg() was called with model=<fitted omg>
        # and FI=TRUE (the vcov.omg path).
        FI          = jointResult$FI)

    if(holdout) {
        result$holdout  <- yHoldout
        result$accuracy <- measures(as.vector(yHoldout), yForecast,
                                    as.vector(yInSample))
    }

    class(result) <- c("omg","om","smooth","occurrence")

    if(!silent){
        plot(result, 7)
    }

    return(result)
}

omgLinkFunction <- function(fittedA, fittedB, EtypeA, EtypeB) {
    if(EtypeA == "A" && EtypeB == "A") {
        return(1 / (1 + exp(fittedB - fittedA)))
    }
    else if(EtypeA == "M" && EtypeB == "M") {
        return(1 / (1 + fittedB / fittedA))
    }
    else if(EtypeA == "M" && EtypeB == "A") {
        return(1 / (1 + exp(fittedB - log(fittedA))))
    }
    else {
        return(1 / (1 + exp(log(fittedB) - fittedA)))
    }
}

# File-scope joint cost function for omg. Used by omgEstimator() during
# optimisation and by the modelDo=="use" branch of omg() for FI computation.
# Takes all dependencies as explicit arguments — no closure state.
omgCF_local <- function(B,
                        # A-side
                        etsModelA, EtypeA, TtypeA, StypeA,
                        modelIsTrendyA, modelIsSeasonalA,
                        componentsNumberETSA, componentsNumberETSNonSeasonalA,
                        componentsNumberETSSeasonalA, componentsNumberARIMAA,
                        lags, lagsModelA, lagsModelMaxA, lagsModelAllA,
                        indexLookupTableA, profilesRecentTableA,
                        matVtA, matWtA, matFA, vecGA,
                        persistenceEstimateA, persistenceLevelEstimateA,
                        persistenceTrendEstimateA, persistenceSeasonalEstimateA,
                        persistenceXregEstimateA, phiEstimateA,
                        initialTypeA, initialEstimateA,
                        initialLevelEstimateA, initialTrendEstimateA,
                        initialSeasonalEstimateA, initialArimaEstimateA,
                        initialXregEstimateA, initialArimaNumberA,
                        arimaModelA, arEstimateA, maEstimateA,
                        arOrdersA, iOrdersA, maOrdersA,
                        arRequiredA, maRequiredA, armaParametersA,
                        nonZeroARIA, nonZeroMAA, arimaPolynomialsA,
                        arPolynomialMatrixA, maPolynomialMatrixA,
                        xregModelA, xregNumberA,
                        xregParametersMissingA, xregParametersIncludedA,
                        xregParametersEstimatedA, xregParametersPersistenceA,
                        constantRequiredA, constantEstimateA,
                        adamCppA,
                        # B-side
                        etsModelB, EtypeB, TtypeB, StypeB,
                        modelIsTrendyB, modelIsSeasonalB,
                        componentsNumberETSB, componentsNumberETSNonSeasonalB,
                        componentsNumberETSSeasonalB, componentsNumberARIMAB,
                        lagsModelB, lagsModelMaxB, lagsModelAllB,
                        indexLookupTableB, profilesRecentTableB,
                        matVtB, matWtB, matFB, vecGB,
                        persistenceEstimateB, persistenceLevelEstimateB,
                        persistenceTrendEstimateB, persistenceSeasonalEstimateB,
                        persistenceXregEstimateB, phiEstimateB,
                        initialTypeB, initialEstimateB,
                        initialLevelEstimateB, initialTrendEstimateB,
                        initialSeasonalEstimateB, initialArimaEstimateB,
                        initialXregEstimateB, initialArimaNumberB,
                        arimaModelB, arEstimateB, maEstimateB,
                        arOrdersB, iOrdersB, maOrdersB,
                        arRequiredB, maRequiredB, armaParametersB,
                        nonZeroARIB, nonZeroMAB, arimaPolynomialsB,
                        arPolynomialMatrixB, maPolynomialMatrixB,
                        xregModelB, xregNumberB,
                        xregParametersMissingB, xregParametersIncludedB,
                        xregParametersEstimatedB, xregParametersPersistenceB,
                        constantRequiredB, constantEstimateB,
                        adamCppB,
                        # B-side scalars for omfitGeneral
                        nNonSeasonalB, nSeasonalB, nETSB,
                        nArimaB, nXregB, nComponentsB, adamETSB_flag,
                        # Shared
                        bounds, regressors,
                        ot, otLogical, obsInSample,
                        nIterations, refineHead, nParamsA,
                        loss = "likelihood",
                        lossFunction = NULL,
                        lambda = 0) {

    B_A <- B[seq_len(nParamsA)]
    B_B <- B[seq_len(length(B) - nParamsA) + nParamsA]

    elemA <- adam_filler(B_A,
                         etsModelA, EtypeA, TtypeA, StypeA,
                         modelIsTrendyA, modelIsSeasonalA,
                         componentsNumberETSA, componentsNumberETSNonSeasonalA,
                         componentsNumberETSSeasonalA, componentsNumberARIMAA,
                         lags, lagsModelA, lagsModelMaxA,
                         matVtA, matWtA, matFA, vecGA,
                         persistenceEstimateA, persistenceLevelEstimateA,
                         persistenceTrendEstimateA, persistenceSeasonalEstimateA,
                         persistenceXregEstimateA, phiEstimateA,
                         initialTypeA, initialEstimateA,
                         initialLevelEstimateA, initialTrendEstimateA,
                         initialSeasonalEstimateA, initialArimaEstimateA,
                         initialXregEstimateA,
                         arimaModelA, arEstimateA, maEstimateA,
                         arOrdersA, iOrdersA, maOrdersA,
                         arRequiredA, maRequiredA, armaParametersA,
                         nonZeroARIA, nonZeroMAA, arimaPolynomialsA,
                         xregModelA, xregNumberA,
                         xregParametersMissingA, xregParametersIncludedA,
                         xregParametersEstimatedA, xregParametersPersistenceA,
                         constantEstimateA, adamCppA,
                         constantRequiredA, initialArimaNumberA)

    elemB <- adam_filler(B_B,
                         etsModelB, EtypeB, TtypeB, StypeB,
                         modelIsTrendyB, modelIsSeasonalB,
                         componentsNumberETSB, componentsNumberETSNonSeasonalB,
                         componentsNumberETSSeasonalB, componentsNumberARIMAB,
                         lags, lagsModelB, lagsModelMaxB,
                         matVtB, matWtB, matFB, vecGB,
                         persistenceEstimateB, persistenceLevelEstimateB,
                         persistenceTrendEstimateB, persistenceSeasonalEstimateB,
                         persistenceXregEstimateB, phiEstimateB,
                         initialTypeB, initialEstimateB,
                         initialLevelEstimateB, initialTrendEstimateB,
                         initialSeasonalEstimateB, initialArimaEstimateB,
                         initialXregEstimateB,
                         arimaModelB, arEstimateB, maEstimateB,
                         arOrdersB, iOrdersB, maOrdersB,
                         arRequiredB, maRequiredB, armaParametersB,
                         nonZeroARIB, nonZeroMAB, arimaPolynomialsB,
                         xregModelB, xregNumberB,
                         xregParametersMissingB, xregParametersIncludedB,
                         xregParametersEstimatedB, xregParametersPersistenceB,
                         constantEstimateB, adamCppB,
                         constantRequiredB, initialArimaNumberB)

    penaltyA <- adam_bounds_checker(elemA, elemA$arimaPolynomials, bounds,
                                    etsModelA, modelIsTrendyA, modelIsSeasonalA,
                                    componentsNumberETSA, componentsNumberETSNonSeasonalA,
                                    componentsNumberETSSeasonalA,
                                    arimaModelA, arEstimateA, maEstimateA,
                                    xregModelA, regressors, xregNumberA, componentsNumberARIMAA,
                                    lagsModelAllA, obsInSample,
                                    arPolynomialMatrixA, maPolynomialMatrixA, phiEstimateA)

    penaltyB <- adam_bounds_checker(elemB, elemB$arimaPolynomials, bounds,
                                    etsModelB, modelIsTrendyB, modelIsSeasonalB,
                                    componentsNumberETSB, componentsNumberETSNonSeasonalB,
                                    componentsNumberETSSeasonalB,
                                    arimaModelB, arEstimateB, maEstimateB,
                                    xregModelB, regressors, xregNumberB, componentsNumberARIMAB,
                                    lagsModelAllB, obsInSample,
                                    arPolynomialMatrixB, maPolynomialMatrixB, phiEstimateB)

    if(penaltyA + penaltyB > 0) { return(1e+300) }

    profilesRecentTableA[] <- elemA$matVt[, seq_len(lagsModelMaxA)]
    profilesRecentTableB[] <- elemB$matVt[, seq_len(lagsModelMaxB)]

    res <- adamCppA$omfitGeneral(
        elemA$matVt, elemA$matWt, elemA$matF, elemA$vecG,
        indexLookupTableA, profilesRecentTableA,
        EtypeB, TtypeB, StypeB,
        nNonSeasonalB, nSeasonalB, nETSB,
        nArimaB, nXregB, nComponentsB,
        constantRequiredB, adamETSB_flag,
        elemB$matVt, elemB$matWt, elemB$matF, elemB$vecG,
        indexLookupTableB, profilesRecentTableB,
        as.numeric(ot),
        any(initialTypeA == c("complete","backcasting")),
        nIterations, refineHead)

    pCombined <- omgLinkFunction(res$fittedA, res$fittedB, EtypeA, EtypeB)

    if(any(is.nan(pCombined)) || any(pCombined <= 0) || any(pCombined >= 1)) {
        return(1e+300)
    }

    # Loss dispatch — joint Bernoulli for "likelihood", probability-scale
    # residual for MSE/MAE/HAM, regularised for LASSO/RIDGE, user callable
    # for "custom". The C++ joint state-space step ran first either way;
    # this just decides what scalar to hand to nloptr.
    errors <- as.numeric(ot) - pCombined
    if(loss == "custom"){
        return(lossFunction(actual=as.numeric(ot), fitted=pCombined, B=B))
    } else if(loss == "likelihood"){
        return(-(sum(log(pCombined[otLogical])) +
                     sum(log(1 - pCombined[!otLogical]))))
    } else if(loss == "MSE"){
        return(mean(errors^2))
    } else if(loss == "MAE"){
        return(mean(abs(errors)))
    } else if(loss == "HAM"){
        return(mean(sqrt(abs(errors))))
    } else if(any(loss == c("LASSO","RIDGE"))){
        errorTerm <- (1 - lambda) * sqrt(mean(errors^2))
        if(loss == "LASSO"){
            return(errorTerm + lambda * sum(abs(B)))
        } else {
            return(errorTerm + lambda * sqrt(sum(B^2)))
        }
    } else {
        # Fallback to likelihood for any unrecognised string.
        return(-(sum(log(pCombined[otLogical])) +
                     sum(log(1 - pCombined[!otLogical]))))
    }
}

# Per-side ETS "usual"-bounds confint correction for one omg sub-model.
# Mirrors the ETS block of confint.adam (R/adam.R:4441-4541). Takes the
# side's parameter values, their standard errors (from the JOINT vcov) and
# the two t-quantiles; returns a 2-column [lower, upper] matrix. The
# admissible-bounds and ARIMA branches of confint.adam are not reproduced
# here — occurrence sub-models are plain ETS with bounds="usual"; for any
# other bounds type the plain t-interval is returned.
omgConfintSide <- function(subModel, params, se, tLo, tHi){
    pn <- names(params)
    n  <- length(params)
    bnd <- matrix(0, n, 2, dimnames=list(pn, NULL))
    bnd[,1] <- tLo * se
    bnd[,2] <- tHi * se

    etsModel   <- isTRUE(subModel$ets)
    boundsType <- subModel$bounds

    if(etsModel && !is.null(boundsType) && boundsType=="usual"){
        if(any(pn=="alpha")){
            bnd["alpha",1] <- max(-params["alpha"], bnd["alpha",1])
            bnd["alpha",2] <- min(1-params["alpha"], bnd["alpha",2])
        }
        if(any(pn=="beta")){
            bnd["beta",1] <- max(-params["beta"], bnd["beta",1])
            if(any(pn=="alpha")){
                bnd["beta",2] <- min(params["alpha"]-params["beta"], bnd["beta",2])
            }
            else{
                bnd["beta",2] <- min(subModel$persistence["alpha"]-params["beta"], bnd["beta",2])
            }
        }
        if(any(substr(pn,1,5)=="gamma")){
            gammas <- which(substr(pn,1,5)=="gamma")
            bnd[gammas,1] <- apply(cbind(bnd[gammas,1], -params[gammas]),1,max)
            if(any(pn=="alpha")){
                bnd[gammas,2] <- apply(cbind(bnd[gammas,2],
                                             (1-params["alpha"])-params[gammas]),1,min)
            }
            else{
                bnd[gammas,2] <- apply(cbind(bnd[gammas,2],
                                             (1-subModel$persistence["alpha"])-params[gammas]),1,min)
            }
        }
        if(any(substr(pn,1,5)=="delta")){
            deltas <- which(substr(pn,1,5)=="delta")
            bnd[deltas,1] <- apply(cbind(bnd[deltas,1], -params[deltas]),1,max)
            bnd[deltas,2] <- apply(cbind(bnd[deltas,2], 1-params[deltas]),1,min)
        }
        if(any(pn=="phi")){
            bnd["phi",1] <- max(-params["phi"], bnd["phi",1])
            bnd["phi",2] <- min(1-params["phi"], bnd["phi",2])
        }
    }

    bnd[] <- bnd + params
    return(bnd)
}

#' @export
coefbootstrap.omg <- function(object, nsim=1000, size=floor(0.75*nobs(object)),
                              replace=FALSE, prob=NULL, parallel=FALSE,
                              method=c("cr","dsr"), ...){

    startTime <- Sys.time();

    cl <- match.call();
    yInSample <- actuals(object);

    method <- match.arg(method);

    if(is.numeric(parallel)){
        nCores <- parallel;
        parallel <- TRUE;
    }
    else if(is.logical(parallel) && parallel){
        nCores <- min(parallel::detectCores() - 1, nsim);
    }

    if(parallel){
        if(!requireNamespace("foreach", quietly = TRUE)){
            stop("In order to run the function in parallel, 'foreach' package must be installed.", call. = FALSE);
        }
        if(!requireNamespace("parallel", quietly = TRUE)){
            stop("In order to run the function in parallel, 'parallel' package must be installed.", call. = FALSE);
        }
        if(Sys.info()['sysname']=="Windows"){
            if(requireNamespace("doParallel", quietly = TRUE)){
                cluster <- parallel::makeCluster(nCores);
                doParallel::registerDoParallel(cluster);
            }
            else{
                stop("Sorry, but in order to run the function in parallel, you need 'doParallel' package.",
                     call. = FALSE);
            }
        }
        else{
            if(requireNamespace("doMC", quietly = TRUE)){
                doMC::registerDoMC(nCores);
                cluster <- NULL;
            }
            else if(requireNamespace("doParallel", quietly = TRUE)){
                cluster <- parallel::makeCluster(nCores);
                doParallel::registerDoParallel(cluster);
            }
            else{
                stop("Sorry, but in order to run the function in parallel, you need either 'doMC' (prefered) or 'doParallel' packages.",
                     call. = FALSE);
            }
        }
    }

    # Joint coefficients, with A:/B: prefixed names (mirrors confint.omg)
    coefficientsOriginal <- c(object$modelA$B, object$modelB$B);
    names(coefficientsOriginal) <- c(paste0("A:", names(object$modelA$B)),
                                     paste0("B:", names(object$modelB$B)));
    nVariables <- length(coefficientsOriginal);
    variablesNames <- names(coefficientsOriginal);
    nParamsA <- length(object$modelA$B);
    obsInsample <- nobs(object);
    # omg's lags() generic errors on the oETS[G] name — use the stored value.
    lags <- object$lags;
    yData <- object$modelA$data;

    coefBootstrap <- matrix(0, nsim, nVariables, dimnames=list(NULL, variablesNames));
    indices <- c(1:obsInsample);

    # Build a fresh omg() call from the fitted sub-model specs. We never use
    # object$call (its head may be om() when the object came from
    # om(occurrence="general")) and never call om() — every replicate goes
    # through omg() directly.
    newCall <- quote(omg());
    newCall$data    <- yData;
    newCall$modelA  <- modelType(object$modelA);
    newCall$modelB  <- modelType(object$modelB);
    newCall$ordersA <- object$modelA$orders;
    newCall$ordersB <- object$modelB$orders;
    newCall$lags    <- lags;
    newCall$silent  <- TRUE;
    newCall$holdout <- FALSE;
    newCall$B  <- coefficientsOriginal;
    newCall$lb <- rep(-Inf, nVariables);
    newCall$ub <- rep(Inf,  nVariables);

    regressionPure <- FALSE;

    obsMinimum <- max(lags, nVariables) + 2;
    if(obsMinimum>=obsInsample && method=="cr"){
        warning("Not enough observations to do Case Resampling bootstrap. Changing method to 'dsr'.",
                call.=FALSE, immediate.=TRUE);
        method <- "dsr";
    }

    changeOrigin <- any(object$modelA$initialType==c("backcasting","complete"));

    sampler <- function(indices,size,replace,prob,regressionPure=FALSE,changeOrigin=FALSE){
        if(regressionPure){
            return(sample(indices,size=size,replace=replace,prob=prob));
        }
        else{
            indices <- c(1:ceiling(runif(1,obsMinimum,obsInsample)));
            startingIndex <- 0;
            if(changeOrigin){
                startingIndex <- floor(runif(1,0,obsInsample-max(indices)));
            }
            return(startingIndex+indices);
        }
    }

    # Extract the joint coefficient vector from a re-fitted omg model.
    jointCoef <- function(testModel){
        c(testModel$modelA$B, testModel$modelB$B);
    }

    #### Bootstrap the data
    if(method=="dsr"){
        type <- "additive";
        if(all(yInSample>=0) && any(yInSample>1)){
            type[] <- "multiplicative";
        }
        dataBoot <- dsrboot(yInSample, nsim=nsim, type=type, intermittent=FALSE);
        if(!parallel){
            for(i in 1:nsim){
                newCall$data <- dataBoot$boot[,i];
                testModel <- tryCatch(suppressWarnings(eval(newCall)), error=function(e) NULL);
                if(!is.null(testModel)){
                    testCoef <- jointCoef(testModel);
                    if(length(testCoef)==nVariables){ coefBootstrap[i,] <- testCoef; }
                }
            }
        }
        else{
            coefBootstrapParallel <- foreach::`%dopar%`(foreach::foreach(i=1:nsim),{
                newCall$data <- dataBoot$boot[,i];
                testModel <- tryCatch(eval(newCall), error=function(e) NULL);
                if(is.null(testModel)){ return(NULL); }
                return(jointCoef(testModel));
            })
            for(i in 1:nsim){
                if(!is.null(coefBootstrapParallel[[i]]) &&
                   length(coefBootstrapParallel[[i]])==nVariables){
                    coefBootstrap[i,] <- coefBootstrapParallel[[i]];
                }
            }
        }
    }
    else{
        #### Case Resampling bootstrap
        # Resample short / sparse subsamples that the joint model cannot be
        # re-estimated on, so every replicate is a genuine re-estimation with
        # the provided joint B as the starting point.
        maxAttempts <- 100L;
        refitOMG <- function(){
            testModel <- NULL; testCoef <- NULL; attempt <- 0L;
            while(is.null(testCoef) && attempt < maxAttempts){
                attempt <- attempt + 1L;
                subsetValues <- sampler(indices,size,replace,prob,regressionPure,changeOrigin);
                newCall$data <- yData[subsetValues];
                testModel <- tryCatch(suppressWarnings(eval(newCall)), error=function(e) NULL);
                if(!is.null(testModel)){
                    cand <- jointCoef(testModel);
                    if(length(cand)==nVariables){ testCoef <- cand; }
                }
            }
            return(testCoef);
        }
        if(!parallel){
            for(i in 1:nsim){
                testCoef <- refitOMG();
                if(!is.null(testCoef)){ coefBootstrap[i,] <- testCoef; }
            }
        }
        else{
            coefBootstrapParallel <- foreach::`%dopar%`(foreach::foreach(i=1:nsim),{
                refitOMG();
            })
            for(i in 1:nsim){
                if(!is.null(coefBootstrapParallel[[i]]) &&
                   length(coefBootstrapParallel[[i]])==nVariables){
                    coefBootstrap[i,] <- coefBootstrapParallel[[i]];
                }
            }
        }
    }

    if(parallel && !is.null(cluster)){
        parallel::stopCluster(cluster);
    }

    coefBootstrap[is.na(coefBootstrap)] <- 0;
    colnames(coefBootstrap) <- variablesNames;

    coefvcov <- coefBootstrap - matrix(coefficientsOriginal, nsim, nVariables, byrow=TRUE);

    return(structure(list(vcov=(t(coefvcov) %*% coefvcov)/nsim,
                          coefficients=coefBootstrap, method=method,
                          nsim=nsim, size=NA, replace=NA, prob=NA,
                          parallel=parallel, model=as.symbol("omg"),
                          timeElapsed=Sys.time()-startTime),
                     class="bootstrap"));
}

#' @export
confint.omg <- function(object, parm, level=0.95, bootstrap=FALSE, ...){
    confintNames <- c(paste0((1-level)/2*100,"%"),
                      paste0((1+level)/2*100,"%"))

    if(bootstrap){
        # Empirical bootstrap quantiles, mirroring confint.adam.
        coefValues <- coefbootstrap(object, ...)
        out <- cbind(sqrt(diag(coefValues$vcov)),
                     apply(coefValues$coefficients, 2, quantile, probs=(1-level)/2),
                     apply(coefValues$coefficients, 2, quantile, probs=(1+level)/2))
        colnames(out) <- c("S.E.", confintNames)
    }
    else{
        V  <- vcov(object, ...)               # JOINT covariance (vcov.omg)
        SE <- sqrt(abs(diag(V)))

        coefJoint <- c(object$modelA$B, object$modelB$B)
        nParamsA  <- length(object$modelA$B)
        idxA <- seq_len(nParamsA)
        idxB <- seq_len(length(coefJoint) - nParamsA) + nParamsA

        # Base t-interval half-widths with the JOINT nobs / nparam, exactly like
        # confint.adam (R/adam.R:4430-4431).
        tLo <- qt((1-level)/2, df=nobs(object)-nparam(object))
        tHi <- qt((1+level)/2, df=nobs(object)+nparam(object))

        sideA <- omgConfintSide(object$modelA, coefJoint[idxA], SE[idxA], tLo, tHi)
        sideB <- omgConfintSide(object$modelB, coefJoint[idxB], SE[idxB], tLo, tHi)

        out <- rbind(cbind(SE[idxA], sideA),
                     cbind(SE[idxB], sideB))
        colnames(out) <- c("S.E.", confintNames)
        rownames(out) <- c(paste0("A:", names(object$modelA$B)),
                           paste0("B:", names(object$modelB$B)))
    }

    if(!missing(parm)){
        out <- out[parm, , drop=FALSE]
    }
    return(out)
}

#' @export
vcov.omg <- function(object, bootstrap=FALSE, heuristics=NULL, ...){
    ellipsis <- list(...)

    if(!is.null(heuristics) && is.numeric(heuristics)){
        # Heuristic shortcut over the joint coef vector
        return(diag(abs(c(object$modelA$B, object$modelB$B)) * heuristics))
    }

    if(bootstrap){
        return(coefbootstrap(object, ...)$vcov)
    }

    h <- if(any(!is.na(object$forecast))) length(object$forecast) else 0
    stepSize <- if(is.null(ellipsis$stepSize)) {
        .Machine$double.eps^(1/4)
    } else {
        ellipsis$stepSize
    }

    # Data is stored on the sub-models, not at the omg top level.
    yData <- object$modelA$data

    modelReturn <- suppressWarnings(
        omg(yData, h=h, model=object, FI=TRUE, stepSize=stepSize))

    if(is.null(modelReturn$FI)){
        stop("Could not compute Fisher Information for this omg model. ",
             "Try a different stepSize.", call.=FALSE)
    }

    brokenVariables <- apply(modelReturn$FI==0, 1, all) |
                       apply(is.nan(modelReturn$FI), 1, any)
    if(any(brokenVariables)){
        modelReturn <- suppressWarnings(
            omg(yData, h=h, model=object, FI=TRUE,
                stepSize=.Machine$double.eps^(1/6)))
        brokenVariables <- apply(modelReturn$FI==0, 1, all)
    }
    if(any(is.nan(modelReturn$FI))){
        stop("Fisher Information contains NaN; try a different stepSize ",
             "(e.g. stepSize=1e-6).", call.=FALSE)
    }
    if(any(eigen(modelReturn$FI, only.values=TRUE)$values < 0)){
        warning("Observed Fisher Information is not positive semi-definite; ",
                "covariance matrix may be unreliable.", call.=FALSE)
    }

    FIMatrix <- modelReturn$FI[!brokenVariables, !brokenVariables, drop=FALSE]
    vcovMatrix <- try(chol2inv(chol(FIMatrix)), silent=TRUE)
    if(inherits(vcovMatrix, "try-error")){
        vcovMatrix <- try(solve(FIMatrix, diag(ncol(FIMatrix)), tol=1e-20),
                          silent=TRUE)
        if(inherits(vcovMatrix, "try-error")){
            warning("Hessian is singular; cannot invert.", call.=FALSE)
            vcovMatrix <- diag(1e+100, ncol(FIMatrix))
        }
    }
    modelReturn$FI[!brokenVariables, !brokenVariables] <- vcovMatrix
    modelReturn$FI[brokenVariables, ] <- Inf
    modelReturn$FI[, brokenVariables] <- Inf
    diag(modelReturn$FI) <- abs(diag(modelReturn$FI))
    return(modelReturn$FI)
}

#' @rdname forecast.smooth
#' @export
forecast.omg <- function(object, h=10, ...) {
    if(is.null(h)) { h <- length(object$forecast) }
    fcA <- forecast.adam(object$modelA, h=h, interval="none",
                         level=0.95, side="both", cumulative=FALSE, ...)
    fcB <- forecast.adam(object$modelB, h=h, interval="none",
                         level=0.95, side="both", cumulative=FALSE, ...)
    EtypeA      <- errorType(object$modelA)
    EtypeB      <- errorType(object$modelB)
    yForecastA  <- as.vector(fcA$mean)
    yForecastB  <- as.vector(fcB$mean)
    yForecast   <- fcA$mean
    yForecast[] <- omgLinkFunction(yForecastA, yForecastB, EtypeA, EtypeB)
    return(structure(
        list(mean=yForecast, lower=fcA$lower, upper=fcA$upper,
             model=object, level=fcA$level, interval=fcA$interval,
             side=fcA$side, cumulative=fcA$cumulative, h=h,
             scenarios=fcA$scenarios),
        class=c("adam.forecast","smooth.forecast","forecast")))
}

#' @export
actuals.omg <- function(object, ...) {
    # Mirror actuals.om's class-preserving binary indicator, but use the
    # top-level object$data so the returned object has the same class
    # (ts / zoo / numeric) as the omg-input series — not the sub-model's
    # post-fit data, which may have been class-stripped.
    if(is.null(object$data)){
        return(actuals.om(object$modelA, ...));
    }
    yObs <- if(is.data.frame(object$data) || is.matrix(object$data)) object$data[,1] else object$data;
    yObs[] <- (yObs != 0) * 1;
    return(yObs);
}

#' @export
actuals.omg_submodel <- function(object, ...) {
    # Reconstruct the latent (unobservable) value the sub-model was implicitly
    # fitting, before the link function turned it into a probability. The
    # sub-model class is ``omg_submodel`` (set in omgFinalFit), so this method
    # dispatches in preference to ``actuals.om`` which would return the
    # binary occurrence indicator.

    return(fitted(object) + residuals(object));
}

#' @export
print.omg <- function(x, digits=4, ...) {
    cat("Time elapsed:", round(as.numeric(x$timeElapsed, units="secs"), 2), "seconds")
    cat(paste0("\nGeneral occurrence model: ", x$model))
    stripModel <- function(m) sub("^o", "", sub("\\[.*\\]$", "", m))
    cat(paste0("\nModel A: ", stripModel(x$modelA$model)))
    cat(paste0("\nModel B: ", stripModel(x$modelB$model)))

    distrib <- switch(x$distribution,
                      "dnorm"     = "Normal",
                      "dlaplace"  = "Laplace",
                      "ds"        = "S",
                      "dgnorm"    = paste0("Generalised Normal with shape=", round(x$other$shape, digits)),
                      "dlogis"    = "Logistic",
                      "plogis"    = "Cumulative Logistic",
                      "dt"        = paste0("Student t with df=", round(x$other$nu, digits)),
                      "dalaplace" = paste0("Asymmetric Laplace with alpha=", round(x$other$alpha, digits)),
                      "dlnorm"    = "Log-Normal",
                      "dllaplace" = "Log-Laplace",
                      "dls"       = "Log-S",
                      "dlgnorm"   = paste0("Log-Generalised Normal with shape=", round(x$other$shape, digits)),
                      "dinvgauss" = "Inverse Gaussian",
                      "dgamma"    = "Gamma")
    cat(paste0("\n\nDistribution assumed in the model: ", distrib))
    cat(paste0("\nLoss function type: ", x$loss))
    if(!is.null(x$lossValue)) {
        cat(paste0("; Loss function value: ", round(x$lossValue, digits)))
    }

    cat(paste0("\n\nSample size: ", nobs(x)))
    cat(paste0("\nNumber of estimated parameters: ", round(nparam(x), digits)))
    cat(paste0("\nNumber of degrees of freedom: ", nobs(x) - round(nparam(x), digits)))
    if(x$nParam[2,5] > 0) {
        cat(paste0("\nNumber of provided parameters: ", x$nParam[2,5]))
    }
    ICs <- c(AIC=AIC(x), AICc=AICc(x), BIC=BIC(x), BICc=BICc(x))
    cat("\nInformation criteria:\n")
    print(round(ICs, digits))
    return(invisible(x))
}

#' @export
summary.omg <- function(object, level=0.95, bootstrap=FALSE, ...) {
    ci <- confint(object, level=level, bootstrap=bootstrap, ...)   # joint table, A:/B: rows

    nParamsA <- length(object$modelA$B)
    idxA <- seq_len(nParamsA)
    idxB <- seq_len(nrow(ci) - nParamsA) + nParamsA

    buildTable <- function(vals, ciRows){
        tab <- cbind(vals, ciRows)
        colnames(tab) <- c("Estimate","Std. Error",
                           paste0("Lower ",(1-level)/2*100,"%"),
                           paste0("Upper ",(1+level)/2*100,"%"))
        rownames(tab) <- sub("^[AB]:", "", rownames(ciRows))
        list(table=tab,
             significance=!(tab[,3]<=0 & tab[,4]>=0))
    }
    A <- buildTable(object$modelA$B, ci[idxA, , drop=FALSE])
    B <- buildTable(object$modelB$B, ci[idxB, , drop=FALSE])

    ourReturn <- list(
        model         = object$model,
        modelAName    = object$modelA$model,
        modelBName    = object$modelB$model,
        occurrence    = "General",
        distribution  = object$distribution,
        coefficientsA = A$table, significanceA = A$significance,
        coefficientsB = B$table, significanceB = B$significance,
        loss          = object$loss,
        lossValue     = object$lossValue,
        nobs          = nobs(object),
        nparam        = nparam(object),
        nParam        = object$nParam,
        call          = object$call,
        ICs           = c(AIC=AIC(object), AICc=AICc(object),
                          BIC=BIC(object), BICc=BICc(object))
    )
    return(structure(ourReturn, class="summary.omg"))
}

#' @export
print.summary.omg <- function(x, ...){
    ellipsis <- list(...)
    digits <- if(is.null(ellipsis$digits)) 4 else ellipsis$digits

    cat(paste0("\nModel estimated using omg() function: ", x$model))
    cat(paste0("\nOccurrence model type: ", x$occurrence))
    cat(paste0("\nDistribution used in the estimation: Cumulative Logistic"))
    cat(paste0("\nLoss function type: ", x$loss))
    if(!is.null(x$lossValue)){
        cat(paste0("; Loss function value: ", round(x$lossValue, digits)))
    }

    # Per-model blocks: model name as the heading directly above its
    # coefficient table.
    printModelBlock <- function(label, name, tab, sig){
        cat(paste0("\n\n", label, ": ", name, "\n"))
        stars <- setNames(vector("character", length(sig)), names(sig))
        stars[sig] <- "*"
        print(data.frame(round(tab, digits), stars,
                         check.names=FALSE, fix.empty.names=FALSE))
    }
    printModelBlock("Model A", x$modelAName, x$coefficientsA, x$significanceA)
    printModelBlock("Model B", x$modelBName, x$coefficientsB, x$significanceB)

    cat(paste0("\nSample size: ", x$nobs))
    cat(paste0("\nNumber of estimated parameters: ", x$nparam))
    cat(paste0("\nNumber of degrees of freedom: ", x$nobs - x$nparam))
    if(x$nParam[2,5] > 0){
        cat(paste0("\nNumber of provided parameters: ", x$nParam[2,5]))
    }
    cat("\nInformation criteria:\n")
    print(round(x$ICs, digits))
    return(invisible(x))
}

#' @export
as.data.frame.summary.omg <- function(x, ...){
    A <- as.data.frame(x$coefficientsA); rownames(A) <- paste0("A:", rownames(A))
    B <- as.data.frame(x$coefficientsB); rownames(B) <- paste0("B:", rownames(B))
    return(rbind(A, B))
}

#' @importFrom stats rstandard
#' @export
rstandard.omg <- function(model, ...){
    obs <- nobs(model);
    df  <- obs - nparam(model);
    p   <- as.numeric(model$fitted);
    e   <- as.numeric(actuals(model)) - p;
    return(e / sqrt(p * (1 - p)) * sqrt(obs / df));
}

#' @importFrom stats rstudent
#' @export
rstudent.omg <- function(model, ...){
    obs <- nobs(model);
    df  <- obs - nparam(model) - 1;
    p   <- as.numeric(model$fitted);
    e   <- as.numeric(actuals(model)) - p;
    return(e / sqrt(p * (1 - p)) * sqrt(obs / df));
}

#' @rdname simulate.om
#' @export
simulate.omg <- function(object, nsim=1, seed=NULL, obs=nobs(object), ...){
    startTime <- Sys.time();
    if(!is.null(seed)){
        set.seed(seed);
    }
    # Sub-calls pass ``seed=NULL`` so they don't re-seed; they consume
    # the now-pinned global RNG state in order, giving a reproducible
    # joint result from one master ``seed``.

    simA <- simulate(object$modelA, nsim=nsim, obs=obs, ...);
    simB <- simulate(object$modelB, nsim=nsim, obs=obs, ...);

    # Combine the **latent** series via ``omgLinkFunction`` — that
    # function operates on the pre-link magnitudes, not on the
    # post-link probabilities. ``simulate.om`` exposes ``$latent``
    # for exactly this reason.
    EtypeA <- errorType(object$modelA);
    EtypeB <- errorType(object$modelB);
    obsInSample <- obs;
    probMat <- matrix(omgLinkFunction(c(simA$latent), c(simB$latent),
                                      EtypeA, EtypeB),
                      obsInSample, nsim);
    probMat[] <- pmin(pmax(probMat, 0), 1);
    occurrenceData <- matrix(rbinom(obsInSample*nsim, 1, c(probMat)),
                             obsInSample, nsim);

    # Preserve ts/zoo timing — borrow the carrier from the A sub-sim
    # (its dimensions and time index match what we need).
    probability <- simA$probability;
    probability[] <- probMat;
    occurrenceOut <- simA$data;
    occurrenceOut[] <- occurrenceData;

    safeProb <- pmax(probMat, .Machine$double.eps);
    if(nsim==1){
        logLik <- sum(log(safeProb));
    }
    else{
        logLik <- colSums(log(safeProb));
    }

    return(structure(list(timeElapsed = Sys.time() - startTime,
                          model       = object$model,
                          occurrence  = "general",
                          probability = probability,
                          data        = occurrenceOut,
                          ot          = occurrenceOut,
                          modelA      = simA,
                          modelB      = simB,
                          logLik      = logLik,
                          other       = list(...)),
                     class=c("omg.sim","oes.sim","smooth.sim")));
}
