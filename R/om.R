#' Occurrence Model
#'
#' Function returns the occurrence part of the ADAM model with the specified
#' probability update and model types.
#'
#' The function estimates probability of demand occurrence, using the selected
#' ADAM state space model. It supports ETS, ARIMA and explanatory variables,
#' also allowing to have multiple frequencies and doing variables selection.
#' It is an ADAM analogue for the binary occurrence variable modelling.
#'
#' For the details about the model and its implementation, see the respective
#' vignette: \code{vignette("om","smooth")}
#'
#' @param data Numeric vector, time series, or data frame. Non-binary input is
#'   automatically binarised: any non-zero value becomes 1.
#' @param model Three-letter ETS specification such as \code{"MNN"} or
#'   \code{"AAN"}. Automatic selection with \code{"Z"} / \code{"X"} /
#'   \code{"Y"} wildcards is supported.
#' @param lags Vector of seasonal lags. Defaults to \code{frequency(data)}.
#' @param orders ARIMA orders list: \code{list(ar, i, ma, select)}.
#' @param constant Logical; whether to include a constant term.
#' @param formula Optional formula for external regressors.
#' @param regressors How to handle regressors: \code{"use"},
#'   \code{"select"}, or \code{"adapt"}.
#' @param occurrence Type of link function mapping state to probability:
#'   \code{"fixed"} (constant), \code{"odds-ratio"}, \code{"inverse-odds-ratio"},
#'   or \code{"direct"}.
#' @param loss Loss function: \code{"likelihood"} (Bernoulli) or \code{"MSE"}.
#' @param h Forecast horizon.
#' @param holdout If \code{TRUE}, a holdout sample of size \code{h} is withheld.
#' @param persistence Optional persistence (smoothing) parameter vector.
#' @param phi Optional damping parameter.
#' @param initial Initialisation method: \code{"backcasting"}, \code{"optimal"},
#'   \code{"two-stage"}, \code{"complete"}, or \code{"gradient"} (solves the
#'   initial states by profiling the occurrence loss; falls back to backcasting
#'   for a custom loss or an out-of-scope specification).
#' @param arma Optional fixed ARMA parameters.
#' @param ic Information criterion for model selection.
#' @param bounds Parameter bounds type.
#' @param ets Type of ETS model: \code{"conventional"} or \code{"adam"}.
#' @param smoother The smoother used by \link[smooth]{msdecompose} to obtain the
#' initial level, trend and seasonal indices (and the seasonal profiles for
#' multiple seasonal models). \code{"default"} (the default) uses \code{"ma"} for
#' \code{initial="optimal"} and \code{"global"} otherwise; \code{"ma"},
#' \code{"lowess"}, \code{"supsmu"} and \code{"global"} force the respective smoother.
#' @param silent If \code{TRUE}, suppresses output and plot.
#' @param ... Additional arguments passed to the optimiser (\code{maxeval},
#'   \code{xtol_rel}, \code{algorithm}, \code{print_level}).
#'
#' @return An object of class \code{c("om","adam","smooth")}.
#'
#' @seealso \link{forecast.om}, \link{adam}
#'
#' @examples
#' set.seed(42)
#' y <- rbinom(120, 1, 0.6)
#' m <- om(y, model="MNN", occurrence="odds-ratio")
#' forecast(m, h=12)
#'
#' @rdname om
#' @export
om <- function(data,
               model = "ZXZ",
               lags  = c(frequency(data)),
               orders = list(ar=c(0), i=c(0), ma=c(0), select=FALSE),
               constant = FALSE,
               formula  = NULL,
               regressors = c("use","select","adapt"),
               occurrence = c("auto","fixed","odds-ratio","inverse-odds-ratio","direct","general"),
               loss = c("likelihood","MSE","MAE","HAM","LASSO","RIDGE"),
               h = 0, holdout = FALSE,
               persistence = NULL, phi = NULL,
               initial = c("backcasting","optimal","two-stage","complete","gradient"),
               arma = NULL,
               ic = c("AICc","AIC","BIC","BICc"),
               bounds = c("usual","admissible","none"),
               ets = c("conventional","adam"),
               smoother = c("default","ma","lowess","supsmu","global"),
               silent = TRUE, ...){

    startTime <- Sys.time();
    cl <- match.call();

    # Capture ellipsis early so FI / stepSize / B / lb / ub passed via ...
    # are visible downstream. Mirrors adam.R.
    ellipsis <- list(...);
    # The msdecompose() smoother for the initial states, passed via ellipsis so
    # the internal adam_checkOptimizer() resolves "default" once initialType is known.
    smoother <- match.arg(smoother);
    ellipsis$smoother <- smoother;

    # If a fitted om object is passed via `model`, lift its parameters out
    # and set modelDo="use" so the optimiser is skipped. Mirrors
    # adam.R:354-424. This is the canonical entry point for vcov(om_obj),
    # which re-calls om(..., model=object, FI=TRUE, stepSize=...).
    if(is.om(model)){
        initial      <- model$initial;
        persistence  <- model$persistence;
        phi          <- model$phi;
        occurrence   <- model$occurrence;
        bounds       <- model$bounds;
        loss         <- model$loss;
        if(!is.null(model$ic)){
            ic <- model$ic;
        }
        ellipsis$B   <- model$B;
        lags         <- model$lags;
        orders       <- model$orders;
        constant     <- if(is.null(model$constant)) FALSE else model$constant;
        arma         <- model$arma;
        if(is.null(formula)){
            formula <- formula(model);
        }
        regressors   <- model$regressors;
        initialType  <- model$initialType;
        # When the original fit used backcasting / complete, the FI must be
        # computed on the SAME objective the optimiser saw — backcasting
        # active, ``nIterations=2``, and initials derived from the data by
        # the C++ kernel. Pass the type as a STRING to parametersChecker so
        # it keeps all ``initial*Estimate=TRUE`` flags and lets
        # ``adam_creator`` seed ``matVt`` from the data; ``adam_cpp$fit``
        # then runs backcasting cleanly. Pinning the converged
        # ``model$profileInitial`` as the recent profile while telling C++
        # to backcast produces NaN in fitted values (the two seeds disagree).
        # For ``optimal`` / ``provided`` fits, the converged numeric initials
        # ARE the correct seed, so keep the original behaviour for that case.
        if(any(model$initialType == c("backcasting","complete","gradient"))){
            initial                <- model$initialType;
            profilesRecentTable    <- NULL;
            profilesRecentProvided <- FALSE;
        }
        else{
            profilesRecentTable    <- model$profileInitial;
            profilesRecentProvided <- TRUE;
        }
        # NOTE: do NOT propagate model$ets — that is the etsModel boolean
        # flag, not the om() `ets` argument (which is a character of
        # c("conventional","adam")). Leave `ets` at the user's default.
        # Collapse the fitted object down to its ETS spec string so the rest
        # of om() treats `model` as a normal spec.
        model        <- modelType(model);
        modelDo_user <- "use";
    } else {
        modelDo_user <- NULL;
    }

    occurrence <- match.arg(occurrence);
    # Resolve `regressors` early — both the auto.om() and omg() forwarding
    # blocks below pass it through as-is, and if it's still the
    # multi-element formal default at that point, downstream match.arg
    # calls (notably omg.R's `match.arg(regressorsB)`) will fail.
    regressors <- match.arg(regressors);
    if(occurrence == "auto") {
        result <- auto.om(data=data, model=model, lags=lags, orders=orders,
                          formula=formula, regressors=regressors,
                          h=h, holdout=holdout,
                          persistence=persistence, phi=phi,
                          initial=initial, arma=arma,
                          ic=ic, bounds=bounds, silent=silent, ets=ets,
                          occurrence=c("fixed","general","odds-ratio","inverse-odds-ratio","direct"),
                          constant=constant, loss=loss, ...)
        result$call <- match.call()
        return(result)
    }
    if(occurrence == "general") {
        result <- omg(data=data, modelA=model, modelB=model,
                      ordersA=orders, ordersB=orders,
                      constantA=constant, constantB=constant,
                      formulaA=formula, formulaB=formula,
                      regressorsA=regressors, regressorsB=regressors,
                      persistenceA=persistence, persistenceB=persistence,
                      phiA=phi, phiB=phi,
                      armaA=arma, armaB=arma,
                      etsA=ets, etsB=ets,
                      lags=lags, h=h, holdout=holdout,
                      initial=initial, loss=loss, ic=ic,
                      bounds=bounds, silent=silent, ...)
        result$call <- match.call()
        return(result)
    }
    occurrenceType <- occurrence;
    # Custom callable loss — same convention as adam() (R/adamGeneral.R:574-602):
    # flip the string flag to "custom" and stash the function for the cost
    # dispatch. Use a uniquely-named slot so the later
    # ``list2env(checkerReturn, ...)`` blast doesn't overwrite the closure
    # with the NULL ``lossFunction`` produced by ``commonParametersChecker``
    # (which sees the string ``"custom"`` and has no callable to capture).
    if(is.function(loss)){
        omUserLossFunction <- loss;
        loss <- "custom";
    }
    else {
        omUserLossFunction <- NULL;
        loss <- match.arg(loss);
    }
    ic <- match.arg(ic);
    bounds <- match.arg(bounds);

    # Regularisation weight — mirrors adam()'s LASSO/RIDGE convention
    # (passed via ellipsis in adam(); promoted to a first-class arg here
    # for clarity).
    lambda <- if(is.null(ellipsis$lambda)) 0 else as.numeric(ellipsis$lambda);
    # `regressors` is resolved earlier (above the auto.om/omg forwarding
    # blocks) — no need to repeat.
    ets <- match.arg(ets);
    # Do not overwrite ellipsis here — it may already hold values pulled out
    # of a fitted-object intake at the top of the function (ellipsis$B etc.).
    if(!exists("ellipsis", inherits=FALSE)){
        ellipsis <- list(...);
    }

    occurrenceChar <- switch(occurrence,
                             "odds-ratio"          = "o",
                             "inverse-odds-ratio"  = "i",
                             "direct"              = "d",
                             "fixed"               = "f",
                             "n");

    #### Data preparation ####
    yName <- paste0(deparse(substitute(data)), collapse="");
    if(length(yName)==0 || is.null(yName)){
        yName <- "y";
    }
    modelDo <- "estimate";
    # If we lifted a fitted om object earlier, switch to the "use" path.
    if(!is.null(modelDo_user)){
        modelDo <- modelDo_user;
    }

    dataChecked <- adam_checkData(data, lags, h, holdout, yName, modelDo, formula);
    list2env(dataChecked, envir=environment());

    #### Force ETS(A,N,N) with persistence=0 for "fixed" occurrence ####
    if(occurrence == "fixed"){
        model <- "ANN";
        persistence <- 0;
        initial <- "optimal";
        modelDo <- "use";
    }

    #### Call parametersChecker ####
    checkerReturn <- parametersChecker(data=data, model=model, lags=lags,
                                       formulaToUse=formula, orders=orders,
                                       constant=constant, arma=arma,
                                       persistence=persistence, phi=phi,
                                       initial=initial,
                                       distribution="plogis",
                                       loss=if(loss=="likelihood") "likelihood" else loss,
                                       h=h, holdout=holdout,
                                       occurrence=occurrence,
                                       ic=ic, bounds=bounds, regressors=regressors,
                                       yName=yName,
                                       silent=silent, modelDo=modelDo,
                                       ellipsis=ellipsis, fast=FALSE);

    #### Pure regression: alm was returned directly by the checker ####
    if(is.alm(checkerReturn)){
        obsInSample <- nobs(checkerReturn);
        nParam <- length(coef(checkerReturn));

        modelReturned <- list(model="Regression");
        modelReturned$timeElapsed <- Sys.time() - startTime;
        modelReturned$call <- cl;
        if(is.null(formula)){
            formula <- formula(checkerReturn);
        }
        if(holdout){
            colnames(data) <- make.names(colnames(data), unique=TRUE);
            modelReturned$holdout <- data[obsInSample + c(1:h),,drop=FALSE];
        }
        else{
            modelReturned$holdout <- NULL;
        }
        responseName <- all.vars(formula)[1];
        y <- data[, responseName];
        yIndex <- try(time(y), silent=TRUE);
        if(inherits(yIndex, "try-error")){
            if(!is.data.frame(data) && !is.null(dim(data))){
                yIndex <- as.POSIXct(rownames(data));
            }
            else if(is.data.frame(data)){
                yIndex <- c(1:nrow(data));
            }
            else{
                yIndex <- c(1:length(data));
            }
        }

        if(inherits(y, "zoo")){
            modelReturned$data      <- data[1:obsInSample,,drop=FALSE];
            modelReturned$fitted    <- zoo(fitted(checkerReturn),    order.by=yIndex[1:obsInSample]);
            modelReturned$residuals <- zoo(residuals(checkerReturn), order.by=yIndex[1:obsInSample]);
            if(h > 0){
                if(holdout){
                    modelReturned$forecast <- zoo(
                        forecast(checkerReturn, h=h, newdata=tail(data,h), interval="none")$mean,
                        order.by=yIndex[obsInSample + 1:h]);
                }
                else{
                    modelReturned$forecast <- zoo(
                        forecast(checkerReturn, h=h, interval="none")$mean,
                        order.by=yIndex[obsInSample + 1:h]);
                }
            }
            else{
                modelReturned$forecast <- zoo(NA, order.by=yIndex[obsInSample + 1]);
            }
            modelReturned$states <- zoo(
                matrix(coef(checkerReturn), obsInSample + 1, nParam, byrow=TRUE,
                       dimnames=list(NULL, names(coef(checkerReturn)))),
                order.by=c(yIndex[1] - diff(yIndex[1:2]), yIndex[1:obsInSample]));
        }
        else{
            yFrequency <- frequency(y);
            modelReturned$data      <- ts(data[1:obsInSample,,drop=FALSE], start=yIndex[1], frequency=yFrequency);
            modelReturned$fitted    <- ts(fitted(checkerReturn),    start=yIndex[1], frequency=yFrequency);
            modelReturned$residuals <- ts(residuals(checkerReturn), start=yIndex[1], frequency=yFrequency);
            if(h > 0){
                if(holdout){
                    modelReturned$forecast <- ts(
                        forecast(checkerReturn, h=h, newdata=tail(data,h), interval="none")$mean,
                        start=yIndex[obsInSample + 1], frequency=yFrequency);
                }
                else{
                    modelReturned$forecast <- ts(
                        as.numeric(forecast(checkerReturn, h=h, interval="none")$mean),
                        start=yIndex[obsInSample] + diff(yIndex[1:2]), frequency=yFrequency);
                }
            }
            else{
                modelReturned$forecast <- ts(NA, start=yIndex[obsInSample] + diff(yIndex[1:2]), frequency=yFrequency);
            }
            modelReturned$states <- ts(
                matrix(coef(checkerReturn), obsInSample + 1, nParam, byrow=TRUE,
                       dimnames=list(NULL, names(coef(checkerReturn)))),
                start=yIndex[1] - diff(yIndex[1:2]), frequency=yFrequency);
        }
        modelReturned$persistence <- rep(0, nParam);
        names(modelReturned$persistence) <- paste0("delta", c(1:nParam));
        modelReturned$phi <- 1;
        modelReturned$transition <- diag(nParam);
        modelReturned$measurement <- checkerReturn$data;
        modelReturned$measurement[,1] <- 1;
        colnames(modelReturned$measurement) <- colnames(modelReturned$states);
        modelReturned$initial <- list(xreg=coef(checkerReturn));
        modelReturned$initialType <- "optimal";
        modelReturned$initialEstimated <- TRUE;
        names(modelReturned$initialEstimated) <- "xreg";
        modelReturned$orders <- list(ar=0, i=0, ma=0);
        modelReturned$arma <- NULL;
        parametersNumber <- matrix(0, 2, 5,
                                   dimnames=list(c("Estimated","Provided"),
                                                 c("nParamInternal","nParamXreg","nParamOccurrence","nParamScale","nParamAll")));
        parametersNumber[1, 2] <- nParam;
        parametersNumber[1, 5] <- nParam;
        modelReturned$nParam        <- parametersNumber;
        modelReturned$formula       <- formula(checkerReturn);
        modelReturned$regressors    <- "use";
        modelReturned$loss          <- checkerReturn$loss;
        modelReturned$lossValue     <- checkerReturn$lossValue;
        modelReturned$lossFunction  <- checkerReturn$lossFunction;
        modelReturned$logLik        <- logLik(checkerReturn);
        modelReturned$distribution  <- checkerReturn$distribution;
        modelReturned$scale         <- checkerReturn$scale;
        modelReturned$other         <- checkerReturn$other;
        modelReturned$B             <- coef(checkerReturn);
        modelReturned$lags          <- 1;
        modelReturned$lagsAll       <- rep(1, nParam);
        modelReturned$FI            <- checkerReturn$FI;
        modelReturned$occurrence    <- occurrence;
        if(holdout){
            modelReturned$accuracy <- measures(modelReturned$holdout[, responseName],
                                               modelReturned$forecast,
                                               modelReturned$data[, responseName]);
        }
        else{
            modelReturned$accuracy <- NULL;
        }
        class(modelReturned) <- c("om","adam","smooth","occurrence");
        if(!silent){
            plot(modelReturned, 7);
        }
        return(modelReturned);
    }

    list2env(checkerReturn, envir=environment());
    # ``commonParametersChecker`` resets ``lossFunction`` to ``NULL`` for a
    # string-valued loss; restore the user-provided callable captured above
    # so the optimiser-side cost dispatch can call it.
    if(loss == "custom"){
        lossFunction <- omUserLossFunction;
    }

    # Delegate ARIMA order selection to auto.om() with the current occurrence type.
    if(is.list(orders) && !is.null(orders$select) && isTRUE(orders$select)){
        result <- auto.om(data=data, model=model, lags=lags,
                          orders=orders, formula=formula,
                          regressors=regressors, occurrence=occurrence,
                          h=h, holdout=holdout,
                          persistence=persistence, phi=phi,
                          initial=initial, arma=arma,
                          ic=ic, bounds=bounds,
                          silent=silent, ets=ets,
                          constant=constant, loss=loss, ...);
        result$call <- match.call();
        return(result);
    }

    occurrence <- occurrenceType;

    # For "fixed": set initial level analytically and disable estimation
    if(occurrenceType == "fixed"){
        if(initialLevelEstimate){
            initialLevel <- mean(ot);
        }
        initialType <- "provided";
        initialLevelEstimate <- FALSE;
        initialEstimate <- FALSE;
        persistenceEstimate <- FALSE;
        persistenceLevelEstimate <- FALSE;
        occurrenceChar <- "d";
        nParamEstimated <- 1;
        modelDo <- "use";
    }

    # If the user supplied a complete persistence vector (or any other input
    # forces modelDo="use" outside the "fixed" branch above), nParamEstimated
    # has not been set yet — but downstream code (omFinalFit, IC) reads it.
    if(!exists("nParamEstimated", inherits=FALSE)){
        nParamEstimated <- 0;
    }

    # Binary indicators (ot from checker is already binary when occurrence != "none")
    yInSample[] <- (yInSample!=0)*1;
    if(holdout){
        yHoldout[] <- (yHoldout != 0) * 1;
        if(any(yClasses=="ts")){
            yHoldout <- ts(yHoldout, start=yForecastStart, frequency=yFrequency);
        } else {
            yHoldout <- zoo(yHoldout, order.by=yForecastIndex);
        }
    }

    # Override occurrence-related flags set by checker
    occurrenceModel <- FALSE;
    oesModel <- NULL;
    yFitted <- matrix(rep(mean(yInSample), obsInSample), ncol=1);
    adamETS <- (ets == "adam");

    #### Optimiser settings ####
    optimSettings <- adam_checkOptimizer(ellipsis=ellipsis, loss=loss, distribution="dnorm",
                                         initialType=initialType, lags=lags,
                                         arimaModel=arimaModel);
    list2env(optimSettings, envir=environment());

    # This is the internal variable, which should be equal to 1 everywhere
    # This is to create all necessary objects correctly.
    otLogicalInternal <- otLogical;
    otLogicalInternal[] <- TRUE;

    #### Inner estimator for a single ETS/ARIMA model ####
    omEstimator <- function(etsModel, Etype, Ttype, Stype, lags,
                            lagsModelSeasonal, lagsModelARIMA,
                            obsStates, obsInSample,
                            yInSample, persistence, persistenceEstimate,
                            persistenceLevel, persistenceLevelEstimate,
                            persistenceTrend, persistenceTrendEstimate,
                            persistenceSeasonal, persistenceSeasonalEstimate,
                            persistenceXreg, persistenceXregEstimate,
                            persistenceXregProvided,
                            phi, phiEstimate,
                            initialType, initialLevel, initialTrend,
                            initialSeasonal, initialArima, initialEstimate,
                            initialLevelEstimate, initialTrendEstimate,
                            initialSeasonalEstimate, initialArimaEstimate,
                            initialXregEstimate, initialXregProvided,
                            arimaModel, arRequired, iRequired, maRequired,
                            armaParameters,
                            componentsNumberARIMA, componentsNamesARIMA,
                            formula, xregModel, xregModelInitials, xregData,
                            xregNumber, xregNames, regressors,
                            xregParametersMissing, xregParametersIncluded,
                            xregParametersEstimated, xregParametersPersistence,
                            constantRequired, constantEstimate, constantValue,
                            constantName,
                            ot, otLogical, occurrenceModel, yFitted,
                            bounds, loss, lossFunction, distribution,
                            horizon, multisteps, other, otherParameterEstimate,
                            lambda, B){

        # omCF_local is defined at file scope (top of this file) — it is a
        # pure function over its explicit arguments, so we just use the
        # file-scope version. nloptr will receive it via eval_f below.

        adamArchitect <- adam_architector(etsModel, Etype, Ttype, Stype, lags,
                                          lagsModelSeasonal,
                                          xregNumber, obsInSample, initialType,
                                          arimaModel, lagsModelARIMA, xregModel,
                                          constantRequired,
                                          componentsNumberARIMA,
                                          obsAll, yIndexAll, yClasses, adamETS);
        list2env(adamArchitect, environment());

        # Etype="A" is needed for the decomposition to work in case of 0/1 data
        adamCreated <- adam_creator(etsModel, Etype="A", Ttype=switch(Ttype, "N"="N", "A"), Stype="A",
                                    modelIsTrendy, modelIsSeasonal,
                                    lags, lagsModel, lagsModelARIMA, lagsModelAll,
                                    lagsModelMax,
                                    profilesRecentTable, FALSE,
                                    obsStates, obsInSample,
                                    obsAll,
                                    componentsNumberETS, componentsNumberETSSeasonal,
                                    componentsNamesETS, otLogicalInternal, ot,
                                    persistence, persistenceEstimate,
                                    persistenceLevel, persistenceLevelEstimate,
                                    persistenceTrend, persistenceTrendEstimate,
                                    persistenceSeasonal, persistenceSeasonalEstimate,
                                    persistenceXreg, persistenceXregEstimate,
                                    persistenceXregProvided,
                                    phi,
                                    initialType, initialEstimate,
                                    initialLevel, initialLevelEstimate,
                                    initialTrend, initialTrendEstimate,
                                    initialSeasonal, initialSeasonalEstimate,
                                    initialArima, initialArimaEstimate,
                                    initialArimaNumber,
                                    initialXregEstimate, initialXregProvided,
                                    arimaModel, arRequired, iRequired, maRequired,
                                    armaParameters,
                                    arOrders, iOrders, maOrders,
                                    componentsNumberARIMA, componentsNamesARIMA,
                                    xregModel, xregModelInitials, xregData,
                                    xregNumber, xregNames,
                                    xregParametersPersistence,
                                    constantRequired, constantEstimate,
                                    constantValue, constantName,
                                    adamCpp,
                                    arEstimate, maEstimate, smoother,
                                    nonZeroARI, nonZeroMA);

        adamCreated$matVt <- om_initial_transform(
            adamCreated$matVt, occurrence, Etype, Ttype, Stype,
            etsModel,
            modelIsTrendy, modelIsSeasonal,
            initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
            componentsNumberETS,
            componentsNumberETSNonSeasonal, componentsNumberETSSeasonal,
            lagsModel, lagsModelMax, lagsModelSeasonal,
            obsInSample, ot,
            arimaModel, componentsNumberARIMA,
            initialArimaEstimate, initialArimaNumber,
            xregModel, xregNumber, initialXregEstimate,
            constantRequired, constantEstimate);

        BValues <- adam_initialiser(etsModel, Etype, Ttype, Stype,
                                    modelIsTrendy, modelIsSeasonal,
                                    componentsNumberETSNonSeasonal,
                                    componentsNumberETSSeasonal,
                                    componentsNumberETS,
                                    lags, lagsModel, lagsModelSeasonal,
                                    lagsModelARIMA, lagsModelMax,
                                    adamCreated$matVt,
                                    persistenceEstimate, persistenceLevelEstimate,
                                    persistenceTrendEstimate,
                                    persistenceSeasonalEstimate,
                                    persistenceXregEstimate,
                                    phiEstimate, initialType, initialEstimate,
                                    initialLevelEstimate, initialTrendEstimate,
                                    initialSeasonalEstimate,
                                    initialArimaEstimate, initialXregEstimate,
                                    arimaModel, arRequired, maRequired,
                                    arEstimate, maEstimate,
                                    arOrders, maOrders,
                                    componentsNumberARIMA, componentsNamesARIMA,
                                    initialArimaNumber,
                                    xregModel, xregNumber,
                                    xregParametersEstimated, xregParametersPersistence,
                                    constantEstimate, constantName,
                                    otherParameterEstimate,
                                    adamCpp,
                                    ets, bounds, ot, otLogicalInternal,
                                    iOrders, armaParameters, other);

        # Respect user-supplied B / lb / ub from ellipses (mirrors adam.R
        # lines 1229-1361). Named B is filtered by name match against the
        # initialiser's parameter set; unnamed B is taken as-is but gets the
        # initialiser's names. lb / ub fall back to BValues only when NULL.
        if(!is.null(B)){
            if(!is.null(names(B))){
                B <- B[names(B) %in% names(BValues$B)];
                BValues$B[] <- B;
            }
            else{
                BValues$B[] <- B;
                names(B) <- names(BValues$B);
            }
            B_used <- BValues$B;
        }
        else{
            B_used <- BValues$B;
        }

        if(is.null(lb)){
            lb <- BValues$Bl;
        }
        if(is.null(ub)){
            ub <- BValues$Bu;
        }

        # Treat the dangerous mixed models — but ONLY when the user did not
        # supply their own B via ellipses. A user-provided B is treated as
        # the authoritative starting point.
        if(is.null(B) &&
           ((Etype=="A" && Ttype=="A" && Stype=="M") ||
            (Etype=="A" && Ttype=="M" && Stype=="A") ||
            (Etype=="M" && Ttype=="A" && Stype=="A") ||
            (Etype=="M" && Ttype=="A" && Stype=="N") ||
            (Etype=="A" && Ttype=="M" && Stype=="N") ||
            (Etype=="M" && Ttype=="M" && Stype=="A") ||
            (Etype=="M" && Ttype=="N" && Stype=="A") ||
            (Etype=="A" && Ttype=="N" && Stype=="M") ||
            occurrence=="direct")){
            B_used[] <- 0;
            B_used[1] <- 0.1;
        }

        # ARIMA companion matrices for bounds checking
        if(arimaModel){
            arPolynomialMatrix <- matrix(0, arOrders %*% lags, arOrders %*% lags);
            if(nrow(arPolynomialMatrix) > 1){
                arPolynomialMatrix[2:nrow(arPolynomialMatrix)-1, 2:nrow(arPolynomialMatrix)] <-
                    diag(nrow(arPolynomialMatrix) - 1);
            }
            maPolynomialMatrix <- matrix(0, maOrders %*% lags, maOrders %*% lags);
            if(nrow(maPolynomialMatrix) > 1){
                maPolynomialMatrix[2:nrow(maPolynomialMatrix)-1, 2:nrow(maPolynomialMatrix)] <-
                    diag(nrow(maPolynomialMatrix) - 1);
            }
        } else {
            arPolynomialMatrix <- maPolynomialMatrix <- NULL;
        }

        # All arguments needed by omCF_local are passed explicitly to nloptr
        # below, so the cost function never reads them from the surrounding
        # closure (which is shared across the model-pool loop and could leak
        # the original model's flags into a different submodel's evaluation).
        nloptrArgs <- list(
            etsModel=etsModel, Etype=Etype, Ttype=Ttype, Stype=Stype,
            modelIsTrendy=modelIsTrendy, modelIsSeasonal=modelIsSeasonal,
            componentsNumberETS=componentsNumberETS,
            componentsNumberETSNonSeasonal=componentsNumberETSNonSeasonal,
            componentsNumberETSSeasonal=componentsNumberETSSeasonal,
            componentsNumberARIMA=componentsNumberARIMA,
            lags=lags, lagsModel=lagsModel, lagsModelMax=lagsModelMax,
            lagsModelAll=lagsModelAll,
            indexLookupTable=indexLookupTable,
            profilesRecentTable=profilesRecentTable,
            matVt=adamCreated$matVt, matWt=adamCreated$matWt,
            matF=adamCreated$matF, vecG=adamCreated$vecG,
            persistenceEstimate=persistenceEstimate,
            persistenceLevelEstimate=persistenceLevelEstimate,
            persistenceTrendEstimate=persistenceTrendEstimate,
            persistenceSeasonalEstimate=persistenceSeasonalEstimate,
            persistenceXregEstimate=persistenceXregEstimate,
            phiEstimate=phiEstimate,
            initialType=initialType, initialEstimate=initialEstimate,
            initialLevelEstimate=initialLevelEstimate,
            initialTrendEstimate=initialTrendEstimate,
            initialSeasonalEstimate=initialSeasonalEstimate,
            initialArimaEstimate=initialArimaEstimate,
            initialXregEstimate=initialXregEstimate,
            initialArimaNumber=initialArimaNumber,
            arimaModel=arimaModel, arEstimate=arEstimate, maEstimate=maEstimate,
            arOrders=arOrders, iOrders=iOrders, maOrders=maOrders,
            arRequired=arRequired, maRequired=maRequired,
            armaParameters=armaParameters,
            nonZeroARI=nonZeroARI, nonZeroMA=nonZeroMA,
            arimaPolynomials=adamCreated$arimaPolynomials,
            arPolynomialMatrix=arPolynomialMatrix,
            maPolynomialMatrix=maPolynomialMatrix,
            xregModel=xregModel, xregNumber=xregNumber,
            xregParametersMissing=xregParametersMissing,
            xregParametersIncluded=xregParametersIncluded,
            xregParametersEstimated=xregParametersEstimated,
            xregParametersPersistence=xregParametersPersistence,
            constantRequired=constantRequired,
            constantEstimate=constantEstimate,
            bounds=bounds, regressors=regressors, loss=loss,
            ot=ot, otLogical=otLogical, obsInSample=obsInSample,
            nIterations=nIterations,
            occurrence=occurrence, occurrenceChar=occurrenceChar,
            adamCpp=adamCpp,
            lambda=lambda, lossFunction=lossFunction);

        # If there is nothing to estimate (e.g. a degenerate/tiny sample where
        # the initialiser produced an empty parameter vector), skip nloptr —
        # calling it with a zero-length x0 errors. Just evaluate the cost once
        # at the empty B. Mirrors the guard in omgEstimator().
        if(length(B_used) == 0){
            res <- list(solution=B_used,
                        objective=do.call(omCF_local, c(list(B=B_used), nloptrArgs)));
        }
        else{
            maxevalUsed <- if(is.null(maxeval)) length(B_used) * 40L else maxeval;
            res <- suppressWarnings(do.call(nloptr,
                                            c(list(x0=B_used, eval_f=omCF_local,
                                                   lb=lb, ub=ub,
                                                   opts=list(algorithm=algorithm,
                                                             xtol_rel=xtol_rel, xtol_abs=xtol_abs,
                                                             ftol_rel=ftol_rel, ftol_abs=ftol_abs,
                                                             maxeval=maxevalUsed, maxtime=maxtime,
                                                             print_level=print_level)),
                                              nloptrArgs)));
            res$call <- quote(nloptr(x0=B_used, eval_f=omCF_local, lb=lb, ub=ub, opts=opts));

            # Retry from BValues$B if the first run hit the infeasibility plateau,
            # but only when the user did NOT supply their own B — their B is the
            # authoritative starting point and must not be silently replaced.
            if(is.null(B) && (is.infinite(res$objective) || res$objective == 1e+300)){
                B_used[] <- BValues$B;
                B_used[] <- 0.001;
                B_used[1] <- 0.01;
                res <- suppressWarnings(do.call(nloptr,
                                                c(list(x0=B_used, eval_f=omCF_local,
                                                       lb=lb, ub=ub,
                                                       opts=list(algorithm=algorithm,
                                                                 xtol_rel=xtol_rel, xtol_abs=xtol_abs,
                                                                 ftol_rel=ftol_rel, ftol_abs=ftol_abs,
                                                                 maxeval=maxevalUsed, maxtime=maxtime,
                                                                 print_level=print_level)),
                                                  nloptrArgs)));
                res$call <- quote(nloptr(x0=B_used, eval_f=omCF_local, lb=lb, ub=ub, opts=opts));
            }
        }

        B_used <- res$solution;
        names(B_used) <- names(BValues$B);
        CFValue <- res$objective;

        # Initial states consume the same df however obtained (mirrors adam):
        # added when backcast/complete/gradient (not in B), redundancy subtracted
        # when optimised. The occurrence model is Bernoulli, so there is no scale
        # parameter. See dfInitialsETSLevelSeasonal().
        etsRedundancy <- 0;
        dfInitials <- 0;
        if(etsModel){
            seasonalLagsEstimated <- if(modelIsSeasonal){
                lagsModelSeasonal[as.logical(initialSeasonalEstimate)];
            } else {
                numeric(0);
            }
            dfLevelSeasonal <- dfInitialsETSLevelSeasonal(seasonalLagsEstimated,
                                                          as.logical(initialLevelEstimate));
            naiveLevelSeasonal <- initialLevelEstimate +
                modelIsSeasonal*sum(initialSeasonalEstimate*(lagsModelSeasonal-1));
            etsRedundancy <- naiveLevelSeasonal - dfLevelSeasonal;
            dfInitials <- dfLevelSeasonal + modelIsTrendy*initialTrendEstimate;
        }
        if(arimaModel){
            dfInitials <- dfInitials + initialArimaNumber*initialArimaEstimate;
        }
        nStatesBackcasting <- 0;
        if(any(initialType==c("backcasting","complete","gradient"))){
            nStatesBackcasting <- dfInitials;
            etsRedundancy <- 0;
        }
        nParamEstimated <- length(B_used) + nStatesBackcasting - etsRedundancy;
        logLikValue <- -CFValue;

        # Fisher Information is NOT computed inside omEstimator. The canonical
        # path is vcov(om_object) -> om(model=object, FI=TRUE) which routes
        # via the modelDo=="use" branch where the hessian is taken at the
        # fixed B. This mirrors how adam() handles FI (adam.R:2698+).
        FIMatrix <- NULL;

        return(list(B=B_used, CFValue=CFValue, nParamEstimated=nParamEstimated,
                    logLikADAMValue=logLikValue,
                    # Fields not present in nloptrArgs:
                    xregData=xregData, xregNames=xregNames,
                    xregModelInitials=xregModelInitials, formula=formula,
                    res=res, FI=FIMatrix,
                    iRequired=iRequired,
                    adamArchitect=adamArchitect, adamCreated=adamCreated,
                    # The exact arg-set that nloptr (and every CF call) saw.
                    # Downstream code reads model/component/persistence/initial
                    # flags from here — there is no need to also duplicate them
                    # into this return list.
                    nloptrArgs=nloptrArgs));
    }

    #### IC function with shared environment for nParam ####
    .icEnv <- new.env(parent=emptyenv());
    .icEnv$nP <- 0L;

    icFunction <- function(ll){
        nP <- .icEnv$nP;
        llObj <- structure(ll, nobs=obsInSample, df=nP, class="logLik");
        return(switch(ic,
                      "AIC"  = AIC(llObj),
                      "AICc" = AICc(llObj),
                      "BIC"  = BIC(llObj),
                      "BICc" = BICc(llObj)));
    };
    icFunctionWrap <- function(ll, nP, obsIS=obsInSample){
        .icEnv$nP <- nP;
        return(icFunction(ll));
    };

    #### Helper: build a full om object from an estimator result ####
    omFinalFit <- function(res, hLocal=0, fullObject=FALSE,
                           yFittedOverride=NULL, yForecastOverride=NULL){
        otLogicalInternal <- otLogical;
        otLogicalInternal[] <- TRUE;

        # Prefer the exact arg-bundle the optimiser/CF saw if it was passed
        # through. This avoids any divergence between the matrices adam_filler
        # is fed here vs what every CF eval was fed.
        nla <- res$nloptrArgs

        if(!is.null(res$adamArchitect)){
            # Reuse objects built before nloptr — avoids any mismatch between the
            # matrices the optimiser saw and those rebuilt here from scratch.
            adamArchitect <- res$adamArchitect;
            adamCreated   <- res$adamCreated;
        } else {
            adamArchitect <- adam_architector(nla$etsModel, nla$Etype, nla$Ttype, nla$Stype,
                                              lags, lagsModelSeasonal,
                                              xregNumber, obsInSample, initialType,
                                              nla$arimaModel, lagsModelARIMA,
                                              xregModel, constantRequired,
                                              componentsNumberARIMA,
                                              obsAll, yIndexAll, yClasses, adamETS);
            adamCreated <- adam_creator(nla$etsModel, nla$Etype, nla$Ttype, nla$Stype,
                                        nla$modelIsTrendy, nla$modelIsSeasonal,
                                        lags, adamArchitect$lagsModel, lagsModelARIMA,
                                        adamArchitect$lagsModelAll, adamArchitect$lagsModelMax,
                                        adamArchitect$profilesRecentTable, FALSE,
                                        adamArchitect$obsStates, obsInSample,
                                        obsAll,
                                        adamArchitect$componentsNumberETS,
                                        adamArchitect$componentsNumberETSSeasonal,
                                        adamArchitect$componentsNamesETS, otLogicalInternal, ot,
                                        persistence, nla$persistenceEstimate,
                                        persistenceLevel, nla$persistenceLevelEstimate,
                                        persistenceTrend, nla$persistenceTrendEstimate,
                                        persistenceSeasonal, nla$persistenceSeasonalEstimate,
                                        persistenceXreg, nla$persistenceXregEstimate,
                                        persistenceXregProvided,
                                        phi,
                                        initialType, nla$initialEstimate,
                                        initialLevel, nla$initialLevelEstimate,
                                        initialTrend, nla$initialTrendEstimate,
                                        initialSeasonal, nla$initialSeasonalEstimate,
                                        initialArima, nla$initialArimaEstimate,
                                        initialArimaNumber,
                                        nla$initialXregEstimate, initialXregProvided,
                                        nla$arimaModel, nla$arRequired, res$iRequired, nla$maRequired,
                                        armaParameters,
                                        nla$arOrders, nla$iOrders, nla$maOrders,
                                        componentsNumberARIMA, componentsNamesARIMA,
                                        xregModel, xregModelInitials, xregData,
                                        xregNumber, xregNames,
                                        xregParametersPersistence,
                                        constantRequired, constantEstimate,
                                        constantValue, constantName,
                                        nla$adamCpp,
                                        nla$arEstimate, nla$maEstimate, smoother,
                                        nonZeroARI, nonZeroMA);
            adamCreated$matVt <- om_initial_transform(
                adamCreated$matVt, occurrence, nla$Etype, nla$Ttype, nla$Stype,
                nla$etsModel,
                nla$modelIsTrendy, nla$modelIsSeasonal,
                nla$initialLevelEstimate, nla$initialTrendEstimate, nla$initialSeasonalEstimate,
                adamArchitect$componentsNumberETS,
                adamArchitect$componentsNumberETSNonSeasonal,
                adamArchitect$componentsNumberETSSeasonal,
                adamArchitect$lagsModel, adamArchitect$lagsModelMax, lagsModelSeasonal,
                obsInSample, ot,
                nla$arimaModel, componentsNumberARIMA,
                nla$initialArimaEstimate, initialArimaNumber,
                xregModel, xregNumber, nla$initialXregEstimate,
                constantRequired, constantEstimate);
        }

        adamFilled <- adam_filler(res$B,
                                  nla$etsModel, nla$Etype, nla$Ttype, nla$Stype,
                                  nla$modelIsTrendy, nla$modelIsSeasonal,
                                  adamArchitect$componentsNumberETS, adamArchitect$componentsNumberETSNonSeasonal,
                                  adamArchitect$componentsNumberETSSeasonal, componentsNumberARIMA,
                                  lags, adamArchitect$lagsModel, adamArchitect$lagsModelMax,
                                  adamCreated$matVt, adamCreated$matWt, adamCreated$matF, adamCreated$vecG,
                                  nla$persistenceEstimate, nla$persistenceLevelEstimate,
                                  nla$persistenceTrendEstimate, nla$persistenceSeasonalEstimate,
                                  nla$persistenceXregEstimate, nla$phiEstimate,
                                  initialType, nla$initialEstimate,
                                  nla$initialLevelEstimate, nla$initialTrendEstimate,
                                  nla$initialSeasonalEstimate, nla$initialArimaEstimate,
                                  nla$initialXregEstimate,
                                  nla$arimaModel, nla$arEstimate, nla$maEstimate,
                                  nla$arOrders, nla$iOrders, nla$maOrders,
                                  nla$arRequired, nla$maRequired, armaParameters,
                                  nonZeroARI, nonZeroMA, adamCreated$arimaPolynomials,
                                  xregModel, xregNumber,
                                  xregParametersMissing, xregParametersIncluded,
                                  xregParametersEstimated, xregParametersPersistence,
                                  constantEstimate, adamArchitect$adamCpp,
                                  constantRequired, initialArimaNumber);
        prof <- adamFilled$matVt[, 1:adamArchitect$lagsModelMax, drop=FALSE];
        # Same dispatcher as omCF_local: re-solves the gradient initials from
        # the seed at the optimum (or falls back to backcasting)
        adamFitted <- adam_fitOrGradient(adamFilled$matVt, adamFilled$matWt,
                                         adamFilled$matF, adamFilled$vecG,
                                         adamArchitect$indexLookupTable, prof,
                                         as.numeric(ot), as.numeric(ot),
                                         initialType, nIterations, adamArchitect$adamCpp,
                                         nla$etsModel, nla$arimaModel, xregModel,
                                         nla$Etype, nla$Ttype, nla$Stype,
                                         adamArchitect$componentsNumberETS,
                                         adamArchitect$componentsNumberETSSeasonal,
                                         adamArchitect$componentsNumberETSNonSeasonal,
                                         adamArchitect$lagsModel, adamArchitect$lagsModelMax,
                                         obsInSample, loss, oType=occurrenceChar,
                                         componentsNumberARIMA=componentsNumberARIMA,
                                         lagsModelAll=adamArchitect$lagsModelAll,
                                         xregNumber=xregNumber);
        yFitted <- omLinkFunction(adamFitted$fitted, nla$Etype, occurrence);

        # The occurrence model's log-likelihood is ALWAYS the Bernoulli
        # likelihood of the fitted probabilities. For loss="likelihood" the
        # cost already is that (up to sign), but for the fit-only losses
        # (MSE/MAE/HAM/...) the cost measures fit, not likelihood, so the
        # logLik must be computed here from the fitted probabilities rather
        # than taken from the cost. No flooring: an infeasible fitted
        # probability surfaces as -Inf instead of being silently clipped.
        # (The empty-B estimator path also arrives here with logLik absent.)
        if(is.null(res$logLikADAMValue) || loss != "likelihood"){
            ot_vec   <- as.numeric(yInSample);
            yfit_vec <- as.numeric(yFitted);
            ll <- sum(ot_vec * log(yfit_vec) + (1 - ot_vec) * log(1 - yfit_vec));
            res$logLikADAMValue <- ll;
        }

        # The reported lossValue is the fitting loss evaluated on the final
        # fitted probabilities, ungated. The infeasibility guard inside the
        # cost (return 1e300 when p leaves [0,1]) only exists to steer the
        # optimiser; it must not leak into the reported loss when the loss
        # optimum itself is infeasible (an MAE/HAM optimum can put p<0). This
        # keeps lossValue = the actual loss the gradient/optimiser minimised,
        # while logLik stays the (possibly NaN) Bernoulli — they are allowed to
        # be misaligned. LASSO/RIDGE/custom keep the optimiser's value (their
        # penalty terms are not recoverable from the fitted alone).
        errorsFinal <- as.numeric(yInSample) - as.numeric(yFitted);
        res$CFValue <- switch(loss,
                              "likelihood" = -res$logLikADAMValue,
                              "MSE"  = mean(errorsFinal^2),
                              "MAE"  = mean(abs(errorsFinal)),
                              "HAM"  = mean(sqrt(abs(errorsFinal))),
                              if(is.null(res$CFValue)) res$objective else res$CFValue);

        # Forecast
        if(hLocal > 0){
            yForecast <- adamArchitect$adamCpp$forecast(tail(adamFilled$matWt, hLocal),
                                                        adamFilled$matF,
                                                        adamArchitect$indexLookupTable[, adamArchitect$lagsModelMax +
                                                                                           obsInSample + 1:hLocal,
                                                                                       drop=FALSE],
                                                        adamFitted$profile, hLocal)$forecast;
            yForecast <- omLinkFunction(yForecast, nla$Etype, occurrence);
            yForecast[is.nan(yForecast)] <- 0;
        }

        if(!fullObject){
            if(hLocal == 0){
                return(yFitted);
            }
            return(list(fitted=yFitted, forecast=as.vector(yForecast)));
        }

        # States
        statesRaw <- adamFitted$states[, (adamArchitect$lagsModelMax+1):ncol(adamFitted$states), drop=FALSE];
        compNames <- rownames(adamCreated$matVt);
        if(!is.null(compNames)){
            rownames(statesRaw) <- compNames;
        }

        # Wrap as ts/zoo
        if(any(yClasses == "ts")){
            yFitted    <- ts(yFitted, start=yStart, frequency=yFrequency);
            errors <- ts(as.numeric(yInSample) - yFitted, start=yStart, frequency=yFrequency);
            matVt    <- ts(t(statesRaw), start=yStart, frequency=yFrequency);
        } else {
            yFitted    <- zoo(yFitted, order.by=yInSampleIndex);
            errors <- zoo(as.numeric(yInSample) - yFitted, order.by=yInSampleIndex);
            matVt    <- zoo(t(statesRaw), order.by=yInSampleIndex);
        }

        # Forecast ts
        if(hLocal > 0 && !is.null(yForecast)){
            if(any(yClasses == "ts")){
                yForecast <- ts(yForecast, start=yForecastStart, frequency=yFrequency);
            } else {
                yForecast <- zoo(yForecast, order.by=yForecastIndex);
            }
        } else {
            yForecast <- if(any(yClasses=="ts")){
                ts(NA, start=yForecastStart, frequency=yFrequency);
            } else {
                zoo(NA, order.by=yForecastIndex[1]);
            }
        }

        # Use overrides for combination, otherwise leave model-fitted values in place
        if(!is.null(yFittedOverride)){
            yFitted[] <- yFittedOverride;
        }
        if(!is.null(yForecastOverride)){
            yForecast[] <- yForecastOverride;
        }

        # Model name
        modelStr <- paste0(nla$Etype, nla$Ttype, "d"[nla$phiEstimate], nla$Stype);
        modelName <- adam_model_name(nla$etsModel, modelStr, xregModel, nla$arimaModel,
                                     nla$arOrders, nla$iOrders, nla$maOrders, lags,
                                     regressors, constantRequired, constantName,
                                     occurrenceType, adamArchitect$componentsNumberETSSeasonal,
                                     prefix = "o");

        # Persistence vector
        vecGFinal <- adamFilled$vecG;
        if(adamArchitect$componentsNumberETS > 0){
            persistenceVec <- as.vector(vecGFinal)[1:adamArchitect$componentsNumberETS];
            names(persistenceVec) <- rownames(vecGFinal)[1:adamArchitect$componentsNumberETS];
        } else {
            persistenceVec <- numeric(0);
        }

        # Initial values
        initialCollected <- adam_initial_collector(
            adamFitted$states[, 1:adamArchitect$lagsModelMax, drop=FALSE],
            nla$etsModel, nla$modelIsTrendy, nla$modelIsSeasonal,
            adamArchitect$lagsModel, adamArchitect$lagsModelMax,
            nla$initialLevelEstimate, nla$initialTrendEstimate, nla$initialSeasonalEstimate,
            adamArchitect$componentsNumberETSSeasonal,
            nla$arimaModel, nla$initialArimaEstimate, initialArima, initialArimaNumber,
            adamArchitect$componentsNumberETS, componentsNumberARIMA,
            adamFilled$arimaPolynomials, nla$Etype,
            xregModel, nla$initialXregEstimate, xregNumber);

        # ARMA parameters
        if(nla$arimaModel && (nla$arRequired || nla$maRequired)){
            armaParametersList <- vector("list", nla$arRequired + nla$maRequired);
            j <- 1L;
            if(nla$arRequired && nla$arEstimate){
                armaParametersList[[j]] <- res$B[nchar(names(res$B))>3 &
                                                     substr(names(res$B),1,3)=="phi"];
                names(armaParametersList)[j] <- "ar";
                j <- j + 1L;
            } else if(nla$arRequired){
                armaParametersList[[j]] <- armaParameters[substr(names(armaParameters),1,3)=="phi"];
                names(armaParametersList)[j] <- "ar";
                j <- j + 1L;
            }
            if(nla$maRequired && nla$maEstimate){
                armaParametersList[[j]] <- res$B[substr(names(res$B),1,5)=="theta"];
                names(armaParametersList)[j] <- "ma";
            } else if(nla$maRequired){
                armaParametersList[[j]] <- armaParameters[substr(names(armaParameters),1,5)=="theta"];
                names(armaParametersList)[j] <- "ma";
            }
        } else {
            armaParametersList <- NULL;
        }

        # Parameter counts
        parNum <- parametersNumber;
        parNum[1,1] <- res$nParamEstimated;
        parNum[1,5] <- sum(parNum[1,1:4]);
        parNum[2,5] <- sum(parNum[2,1:4]);

        if(any(yClasses == "ts")){
            yInSample <- ts(yInSample, start=yStart, frequency=yFrequency);
        } else {
            yInSample <- zoo(yInSample, order.by=yInSampleIndex);
        }

        subModel <- list(
            model = modelName,
            timeElapsed = Sys.time() - startTime,
            data = yInSample,
            fitted = yFitted,
            residuals = errors,
            forecast = yForecast,
            states = matVt,
            profile = adamFitted$profile,
            profileInitial = prof,
            persistence = persistenceVec,
            phi = if(nla$phiEstimate) res$B["phi"] else phi,
            transition = adamFilled$matF,
            measurement = adamFilled$matWt,
            initial = initialCollected$initialValue,
            initialType = initialType,
            initialEstimated = initialCollected$initialEstimated,
            orders = list(ar=nla$arOrders, i=nla$iOrders, ma=nla$maOrders),
            arma = armaParametersList,
            constant = if(constantRequired) {
                if(constantEstimate) res$B[constantName] else constantValue
            } else NULL,
            nParam = parNum,
            occurrence = occurrenceType,
            formula = formula,
            regressors = regressors,
            loss = loss,
            lossValue = res$CFValue,
            lossFunction = lossFunction,
            logLik = res$logLikADAMValue,
            distribution = "plogis",
            scale = NA,
            other = if(exists("otherReturned", inherits=FALSE)) otherReturned else NULL,
            B = res$B,
            lags = lags,
            lagsAll = adamArchitect$lagsModelAll,
            ets = nla$etsModel,
            res = res$res,
            FI = res$FI,
            adamCpp = adamArchitect$adamCpp,
            bounds = bounds,
            call = cl
        );

        if(holdout){
            subModel$holdout <- yHoldout;
            subModel$accuracy <- measures(as.vector(yHoldout), yForecast,
                                          as.vector(yInSample));
        }

        class(subModel) <- c("om","adam","smooth","occurrence");
        return(subModel);
    };

    #### Model selection or combination ####
    if(modelDo %in% c("select","combine")){
        omEstimatorWrapper <- function(etsModel, Etype, Ttype, Stype, lags,
                                       lagsModelSeasonal, lagsModelARIMA,
                                       obsStates, obsInSample,
                                       yInSample, persistence, persistenceEstimate,
                                       persistenceLevel, persistenceLevelEstimate,
                                       persistenceTrend, persistenceTrendEstimate,
                                       persistenceSeasonal, persistenceSeasonalEstimate,
                                       persistenceXreg, persistenceXregEstimate,
                                       persistenceXregProvided,
                                       phi, phiEstimate,
                                       initialType, initialLevel, initialTrend,
                                       initialSeasonal, initialArima, initialEstimate,
                                       initialLevelEstimate, initialTrendEstimate,
                                       initialSeasonalEstimate, initialArimaEstimate,
                                       initialXregEstimate, initialXregProvided,
                                       arimaModel, arRequired, iRequired, maRequired,
                                       armaParameters,
                                       componentsNumberARIMA, componentsNamesARIMA,
                                       formula, xregModel, xregModelInitials, xregData,
                                       xregNumber, xregNames, regressors,
                                       xregParametersMissing, xregParametersIncluded,
                                       xregParametersEstimated, xregParametersPersistence,
                                       constantRequired, constantEstimate, constantValue,
                                       constantName,
                                       ot, otLogical, occurrenceModel, yFitted,
                                       bounds, loss, lossFunction, distribution,
                                       horizon, multisteps, other, otherParameterEstimate,
                                       lambda, B){
            res <- omEstimator(etsModel, Etype, Ttype, Stype, lags,
                               lagsModelSeasonal, lagsModelARIMA,
                               obsStates, obsInSample,
                               yInSample, persistence, persistenceEstimate,
                               persistenceLevel, persistenceLevelEstimate,
                               persistenceTrend, persistenceTrendEstimate,
                               persistenceSeasonal, persistenceSeasonalEstimate,
                               persistenceXreg, persistenceXregEstimate,
                               persistenceXregProvided,
                               phi, phiEstimate,
                               initialType, initialLevel, initialTrend,
                               initialSeasonal, initialArima, initialEstimate,
                               initialLevelEstimate, initialTrendEstimate,
                               initialSeasonalEstimate, initialArimaEstimate,
                               initialXregEstimate, initialXregProvided,
                               arimaModel, arRequired, iRequired, maRequired,
                               armaParameters,
                               componentsNumberARIMA, componentsNamesARIMA,
                               formula, xregModel, xregModelInitials, xregData,
                               xregNumber, xregNames, regressors,
                               xregParametersMissing, xregParametersIncluded,
                               xregParametersEstimated, xregParametersPersistence,
                               constantRequired, constantEstimate, constantValue,
                               constantName,
                               ot, otLogical, occurrenceModel, yFitted,
                               bounds, loss, lossFunction, "dnorm",
                               horizon, multisteps, other, otherParameterEstimate,
                               lambda, B);
            .icEnv$nP <- res$nParamEstimated;
            res$IC <- icFunction(res$logLikADAMValue);
            return(res);
        }

        adamSelected <- adam_selector(omEstimatorWrapper,
                                      model, modelsPool, allowMultiplicative,
                                      modelDo=modelDo,
                                      etsModel, Etype, Ttype, Stype, damped, lags,
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
                                      ot, otLogical, occurrenceModel, yFitted,
                                      icFunction,
                                      bounds, loss, lossFunction, "dnorm",
                                      horizon, multisteps, other, otherParameterEstimate,
                                      lambda, silent, B);

        icSelection <- adamSelected$icSelection;
        bestIdx <- which.min(icSelection)[1];
        modelOriginal <- model;
        estimatorResult <- adamSelected$results[[bestIdx]];
        # The selected sub-model's Etype/Ttype/Stype/persistence/initial flags
        # live inside nloptrArgs (single source of truth). Unpack it FIRST so
        # those values land in the outer scope; then unpack the top-level keys
        # (which include adamArchitect, adamCreated, B, etc.).
        list2env(estimatorResult$nloptrArgs, environment());
        list2env(estimatorResult, environment());

        if(modelDo == "combine"){
            icWeights <- adam_ic_weights(icSelection);
            # temporary weights, dropping the very small ones
            icWeightsTemp <- icWeights
            icWeightsTemp[icWeightsTemp<1e-5] <- 0;
            # Calculate sensible weights
            wSum <- sum(icWeightsTemp);
            # Amend the weights if they don't add up to 1
            if(wSum > 0 && wSum!=1){
                icWeights <- icWeights / wSum;
            }

            yFittedCombined <- matrix(0, obsInSample, 1);
            yForecastCombined <- if(h > 0) numeric(h) else NULL;
            individualModels <- vector("list", length(icWeights));
            for(i in seq_along(icWeights)){
                subModel <- tryCatch(omFinalFit(adamSelected$results[[i]],
                                                hLocal=h, fullObject=TRUE),
                                     error=function(e){
                                         message("om(): combine: model ", i, " fitter failed (",
                                                 conditionMessage(e), "), dropping from average.");
                                         icWeights[i] <<- 0;
                                         NULL;
                                     });
                if(is.null(subModel)){
                    next;
                }
                individualModels[[i]] <- subModel;
                # Don't add thingy if the weight is low or there are NaNs
                if(icWeights[i] >= 1e-5 && !any(is.nan(subModel$fitted))){
                    yFittedCombined[] <- yFittedCombined +
                        icWeights[i] * subModel$fitted;
                    if(h > 0){
                        yForecastCombined[] <- yForecastCombined +
                            icWeights[i] * subModel$forecast;
                    }
                }
            }
            names(individualModels) <- if(!is.null(names(icSelection))) names(icSelection) else
                paste0("model", seq_along(individualModels));
        }

        adamArchitect <- adam_architector(etsModel, Etype, Ttype, Stype, lags, lagsModelSeasonal,
                                          xregNumber, obsInSample, initialType,
                                          arimaModel, lagsModelARIMA, xregModel, constantRequired,
                                          componentsNumberARIMA,
                                          obsAll, yIndexAll, yClasses, adamETS);
        list2env(adamArchitect, environment());
    }
    else if(modelDo == "use"){
        # No estimation needed — parameters fully specified (e.g. "fixed"
        # occurrence). Build adamArchitect / adamCreated / nloptrArgs here so
        # estimatorResult matches the structure returned by omEstimator and
        # omFinalFit can take its "reuse" path (no rebuild).

        adamArchitectUse <- adam_architector(etsModel, Etype, Ttype, Stype, lags,
                                             lagsModelSeasonal,
                                             xregNumber, obsInSample, initialType,
                                             arimaModel, lagsModelARIMA, xregModel,
                                             constantRequired,
                                             componentsNumberARIMA,
                                             obsAll, yIndexAll, yClasses, adamETS);

        # Same Etype="A" decomposition trick as omEstimator (line 479): keeps
        # the matrix structure stable on 0/1 data even when the model is M.
        adamCreatedUse <- adam_creator(etsModel, Etype="A",
                                       Ttype=switch(Ttype, "N"="N", "A"), Stype="A",
                                       modelIsTrendy, modelIsSeasonal,
                                       lags, adamArchitectUse$lagsModel, lagsModelARIMA,
                                       adamArchitectUse$lagsModelAll,
                                       adamArchitectUse$lagsModelMax,
                                       adamArchitectUse$profilesRecentTable, FALSE,
                                       adamArchitectUse$obsStates, obsInSample,
                                       obsAll,
                                       adamArchitectUse$componentsNumberETS,
                                       adamArchitectUse$componentsNumberETSSeasonal,
                                       adamArchitectUse$componentsNamesETS,
                                       otLogicalInternal, ot,
                                       persistence, persistenceEstimate,
                                       persistenceLevel, persistenceLevelEstimate,
                                       persistenceTrend, persistenceTrendEstimate,
                                       persistenceSeasonal, persistenceSeasonalEstimate,
                                       persistenceXreg, persistenceXregEstimate,
                                       persistenceXregProvided,
                                       phi,
                                       initialType, initialEstimate,
                                       initialLevel, initialLevelEstimate,
                                       initialTrend, initialTrendEstimate,
                                       initialSeasonal, initialSeasonalEstimate,
                                       initialArima, initialArimaEstimate,
                                       initialArimaNumber,
                                       initialXregEstimate, initialXregProvided,
                                       arimaModel, arRequired, iRequired, maRequired,
                                       armaParameters,
                                       arOrders, iOrders, maOrders,
                                       componentsNumberARIMA, componentsNamesARIMA,
                                       xregModel, xregModelInitials, xregData,
                                       xregNumber, xregNames,
                                       xregParametersPersistence,
                                       constantRequired, constantEstimate,
                                       constantValue, constantName,
                                       adamArchitectUse$adamCpp,
                                       arEstimate, maEstimate, smoother,
                                       nonZeroARI, nonZeroMA);

        adamCreatedUse$matVt <- om_initial_transform(
            adamCreatedUse$matVt, occurrence, Etype, Ttype, Stype,
            etsModel,
            modelIsTrendy, modelIsSeasonal,
            initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
            adamArchitectUse$componentsNumberETS,
            adamArchitectUse$componentsNumberETSNonSeasonal,
            adamArchitectUse$componentsNumberETSSeasonal,
            adamArchitectUse$lagsModel, adamArchitectUse$lagsModelMax,
            lagsModelSeasonal,
            obsInSample, ot,
            arimaModel, componentsNumberARIMA,
            initialArimaEstimate, initialArimaNumber,
            xregModel, xregNumber, initialXregEstimate,
            constantRequired, constantEstimate);

        # ARIMA companion matrices (parallel to omEstimator lines 578-590).
        if(arimaModel){
            arPolynomialMatrixUse <- matrix(0, arOrders %*% lags, arOrders %*% lags);
            if(nrow(arPolynomialMatrixUse) > 1){
                arPolynomialMatrixUse[2:nrow(arPolynomialMatrixUse)-1,
                                      2:nrow(arPolynomialMatrixUse)] <-
                    diag(nrow(arPolynomialMatrixUse) - 1);
            }
            maPolynomialMatrixUse <- matrix(0, maOrders %*% lags, maOrders %*% lags);
            if(nrow(maPolynomialMatrixUse) > 1){
                maPolynomialMatrixUse[2:nrow(maPolynomialMatrixUse)-1,
                                      2:nrow(maPolynomialMatrixUse)] <-
                    diag(nrow(maPolynomialMatrixUse) - 1);
            }
        } else {
            arPolynomialMatrixUse <- NULL;
            maPolynomialMatrixUse <- NULL;
        }

        # Full nloptrArgs, identical shape to omEstimator's (line 597-642).
        nloptrArgsUse <- list(
            etsModel=etsModel, Etype=Etype, Ttype=Ttype, Stype=Stype,
            modelIsTrendy=modelIsTrendy, modelIsSeasonal=modelIsSeasonal,
            componentsNumberETS=adamArchitectUse$componentsNumberETS,
            componentsNumberETSNonSeasonal=adamArchitectUse$componentsNumberETSNonSeasonal,
            componentsNumberETSSeasonal=adamArchitectUse$componentsNumberETSSeasonal,
            componentsNumberARIMA=componentsNumberARIMA,
            lags=lags,
            lagsModel=adamArchitectUse$lagsModel,
            lagsModelMax=adamArchitectUse$lagsModelMax,
            lagsModelAll=adamArchitectUse$lagsModelAll,
            indexLookupTable=adamArchitectUse$indexLookupTable,
            profilesRecentTable=adamArchitectUse$profilesRecentTable,
            matVt=adamCreatedUse$matVt, matWt=adamCreatedUse$matWt,
            matF=adamCreatedUse$matF, vecG=adamCreatedUse$vecG,
            persistenceEstimate=persistenceEstimate,
            persistenceLevelEstimate=persistenceLevelEstimate,
            persistenceTrendEstimate=persistenceTrendEstimate,
            persistenceSeasonalEstimate=persistenceSeasonalEstimate,
            persistenceXregEstimate=persistenceXregEstimate,
            phiEstimate=phiEstimate,
            initialType=initialType, initialEstimate=initialEstimate,
            initialLevelEstimate=initialLevelEstimate,
            initialTrendEstimate=initialTrendEstimate,
            initialSeasonalEstimate=initialSeasonalEstimate,
            initialArimaEstimate=initialArimaEstimate,
            initialXregEstimate=initialXregEstimate,
            initialArimaNumber=initialArimaNumber,
            arimaModel=arimaModel, arEstimate=arEstimate, maEstimate=maEstimate,
            arOrders=arOrders, iOrders=iOrders, maOrders=maOrders,
            arRequired=arRequired, maRequired=maRequired,
            armaParameters=armaParameters,
            nonZeroARI=nonZeroARI, nonZeroMA=nonZeroMA,
            arimaPolynomials=adamCreatedUse$arimaPolynomials,
            arPolynomialMatrix=arPolynomialMatrixUse,
            maPolynomialMatrix=maPolynomialMatrixUse,
            xregModel=xregModel, xregNumber=xregNumber,
            xregParametersMissing=xregParametersMissing,
            xregParametersIncluded=xregParametersIncluded,
            xregParametersEstimated=xregParametersEstimated,
            xregParametersPersistence=xregParametersPersistence,
            constantRequired=constantRequired,
            constantEstimate=constantEstimate,
            bounds=bounds, regressors=regressors, loss=loss,
            ot=ot, otLogical=otLogical, obsInSample=obsInSample,
            nIterations=nIterations,
            occurrence=occurrence, occurrenceChar=occurrenceChar,
            adamCpp=adamArchitectUse$adamCpp);

        # Fisher Information at the supplied / fitted parameters. Mirrors
        # the FI block in adam.R:2698+ — the hessian of -logLik (which is
        # what omCF_local returns) IS the observed Fisher Information.
        # This is the path vcov.om uses: om(..., model=object, FI=TRUE, ...).
        # The trick: omCF_local only varies the elements of B that the
        # *Estimate flags say are estimated. With a fitted-model intake,
        # those flags are FALSE (parameters are "provided"), so the CF
        # ignores B and the hessian comes out as zero. We override the
        # flags here based on names(B) so the hessian is taken over
        # exactly the parameters that appear in B.
        FIMatrixUse <- NULL;
        if(isTRUE(ellipsis$FI)){
            stepSize <- if(is.null(ellipsis$stepSize)) {
                .Machine$double.eps^(1/4);
            } else {
                ellipsis$stepSize;
            };
            B_for_FI <- ellipsis$B;
            if(!is.null(B_for_FI) && length(B_for_FI) > 0){
                # Derive *EstimateFI flags from names(B) — adam.R:2768+
                Bnames <- names(B_for_FI);
                ncSeas <- adamArchitectUse$componentsNumberETSSeasonal;
                pLvlFI   <- any(Bnames=="alpha");
                pTrdFI   <- any(Bnames=="beta");
                if(any(substr(Bnames,1,5)=="gamma")){
                    gammasMask <- substr(Bnames,1,5)=="gamma";
                    if(sum(gammasMask)==1){
                        pSeaFI <- TRUE;
                    }
                    else{
                        pSeaFI <- vector("logical", ncSeas);
                        pSeaFI[as.numeric(substr(Bnames,6,6)[gammasMask])] <- TRUE;
                    }
                } else {
                    pSeaFI <- FALSE;
                }
                pXrgFI    <- any(substr(Bnames,1,5)=="delta");
                pEstFI    <- any(c(pLvlFI, pTrdFI, pSeaFI, pXrgFI));
                phiEstFI  <- any(Bnames=="phi");
                iLvlFI    <- any(Bnames=="level");
                iTrdFI    <- any(Bnames=="trend");
                if(any(substr(Bnames,1,8)=="seasonal")){
                    sn <- Bnames[substr(Bnames,1,8)=="seasonal"];
                    iSeaFI <- vector("logical", ncSeas);
                    if(any(substr(sn,1,9)=="seasonal_")){
                        iSeaFI[] <- TRUE;
                    } else {
                        iSeaFI[unique(as.numeric(substr(sn,9,9)))] <- TRUE;
                    }
                } else {
                    iSeaFI <- FALSE;
                }
                iAriFI <- if(arimaModel) any(substr(Bnames,1,10)=="ARIMAState") else FALSE;
                iXrgFI <- if(xregModel) any(colnames(xregData) %in% Bnames) else FALSE;
                iEstFI <- any(c(iLvlFI, iTrdFI, iSeaFI, iAriFI, iXrgFI));
                # If initials are in B, FI sees the model as initial="optimal";
                # otherwise leave the user's initialType (backcasting / provided).
                iTypeFI <- if(iEstFI) "optimal" else initialType;

                # Patch nloptrArgsUse with the FI-specific flags and disable
                # bounds so omCF_local never short-circuits with 1e+300.
                nlaFI <- nloptrArgsUse;
                nlaFI$persistenceEstimate        <- pEstFI;
                nlaFI$persistenceLevelEstimate   <- pLvlFI;
                nlaFI$persistenceTrendEstimate   <- pTrdFI;
                nlaFI$persistenceSeasonalEstimate<- pSeaFI;
                nlaFI$persistenceXregEstimate    <- pXrgFI;
                nlaFI$phiEstimate                <- phiEstFI;
                nlaFI$initialType                <- iTypeFI;
                nlaFI$initialEstimate            <- iEstFI;
                nlaFI$initialLevelEstimate       <- iLvlFI;
                nlaFI$initialTrendEstimate       <- iTrdFI;
                nlaFI$initialSeasonalEstimate    <- iSeaFI;
                nlaFI$initialArimaEstimate       <- iAriFI;
                nlaFI$initialXregEstimate        <- iXrgFI;
                nlaFI$bounds                     <- "none";
                if(arimaModel){
                    nlaFI$arPolynomialMatrix <- NULL;
                    nlaFI$maPolynomialMatrix <- NULL;
                }

                CFAtOptimum <- do.call(omCF_local,
                                       c(list(B=B_for_FI), nlaFI));
                omCF_for_FI <- function(B){
                    names(B) <- Bnames;
                    val <- tryCatch(suppressWarnings(
                                        do.call(omCF_local,
                                                c(list(B=B), nlaFI))),
                                    error = function(e) CFAtOptimum + 1e6);
                    if(!is.finite(val)){
                        val <- CFAtOptimum + 1e6;
                    }
                    return(val);
                }
                FIMatrixUse <- try(suppressWarnings(
                                       hessianCpp(omCF_for_FI, B_for_FI,
                                                  h=stepSize)),
                                   silent=TRUE);
                if(inherits(FIMatrixUse, "try-error") ||
                   any(!is.finite(FIMatrixUse))){
                    FIMatrixUse <- NULL;
                } else {
                    colnames(FIMatrixUse) <- Bnames;
                    rownames(FIMatrixUse) <- Bnames;
                }
            }
        }

        # Degrees of freedom for the "use" path: nothing is optimised, but the
        # backcast / complete / gradient initial states are still determined from
        # the data and consume df. The occurrence model is Bernoulli, so there is
        # no scale. "fixed" occurrence instead estimates a single probability
        # level (nParamEstimated=1, set above); every other use case counts its
        # backcast initials here.
        if(occurrenceType != "fixed"){
            nParamEstimated <- dfInitialsBackcast(etsModel, modelIsSeasonal, modelIsTrendy,
                                                  lagsModelSeasonal, initialLevelEstimate,
                                                  initialTrendEstimate, initialSeasonalEstimate,
                                                  arimaModel, initialArimaNumber, initialArimaEstimate,
                                                  xregModel, xregNumber, initialXregEstimate,
                                                  initialType);
        }

        estimatorResult <- list(
            B=if(is.null(ellipsis$B)) numeric(0) else ellipsis$B,
            CFValue=0, nParamEstimated=nParamEstimated,
            logLikADAMValue=NULL,
            xregData=xregData, xregNames=xregNames,
            xregModelInitials=xregModelInitials, formula=formula,
            res=list(objective=0), FI=FIMatrixUse,
            iRequired=iRequired,
            adamArchitect=adamArchitectUse, adamCreated=adamCreatedUse,
            nloptrArgs=nloptrArgsUse);
    }
    else{
        estimatorResult <- omEstimator(etsModel, Etype, Ttype, Stype, lags,
                                       lagsModelSeasonal, lagsModelARIMA,
                                       obsStates, obsInSample,
                                       yInSample, persistence, persistenceEstimate,
                                       persistenceLevel, persistenceLevelEstimate,
                                       persistenceTrend, persistenceTrendEstimate,
                                       persistenceSeasonal, persistenceSeasonalEstimate,
                                       persistenceXreg, persistenceXregEstimate,
                                       persistenceXregProvided,
                                       phi, phiEstimate,
                                       initialType, initialLevel, initialTrend,
                                       initialSeasonal, initialArima, initialEstimate,
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
                                       ot, otLogical, occurrenceModel, yFitted,
                                       bounds, loss, lossFunction, "dnorm",
                                       horizon, multisteps, other, otherParameterEstimate,
                                       lambda, B);
        list2env(estimatorResult$nloptrArgs, environment());
        list2env(estimatorResult, environment());
    }

    #### Build return object via omFinalFit ####
    if(modelDo == "combine"){
        modelReturned <- omFinalFit(estimatorResult, hLocal=h, fullObject=TRUE,
                                    yFittedOverride=yFittedCombined,
                                    yForecastOverride=yForecastCombined);
        modelReturned$model <- adam_model_name(etsModel, modelOriginal, xregModel, arimaModel,
                                               arOrders, iOrders, maOrders, lags,
                                               regressors, constantRequired, constantName,
                                               occurrenceType, componentsNumberETSSeasonal,
                                               prefix = "o");
        modelReturned$models <- individualModels;
        modelReturned$ICw <- icWeights;
        modelReturned$ICs <- icSelection;
        # Weighted estimated parameters across the combined pool (mirrors adam())
        nParamMat <- modelReturned$nParam;
        nParamMat[1,1] <- sum(sapply(individualModels,
                                     function(x) if(is.null(x)) 0 else nparam(x)) *
                                  icWeights);
        nParamMat[1,5] <- sum(nParamMat[1,1:4]);
        nParamMat[2,5] <- sum(nParamMat[2,1:4]);
        modelReturned$nParam <- nParamMat;
        class(modelReturned) <- c("omCombined","om","adam","smooth","occurrence");
    } else {
        modelReturned <- omFinalFit(estimatorResult, hLocal=h, fullObject=TRUE);
        if(modelDo == "select"){
            modelReturned$ICs <- icSelection;
        }
    }

    if(!silent){
        plot(modelReturned, 7);
    }

    return(modelReturned);
}

# Transform initial state values for occurrence models.
# Overwrites matVt level, trend, and seasonal rows with values on the
# correct occurrence-model scale.  Only touches components whose
# corresponding *Estimate flag is TRUE (i.e. not user-provided).
# Follows the pattern of oesInitialiser in oes.R lines 288-368.
om_initial_transform <- function(matVt, occurrence, Etype, Ttype, Stype,
                                 etsModel,
                                 modelIsTrendy, modelIsSeasonal,
                                 initialLevelEstimate, initialTrendEstimate,
                                 initialSeasonalEstimate,
                                 componentsNumberETS,
                                 componentsNumberETSNonSeasonal,
                                 componentsNumberETSSeasonal,
                                 lagsModel, lagsModelMax, lagsModelSeasonal,
                                 obsInSample, ot,
                                 arimaModel, componentsNumberARIMA,
                                 initialArimaEstimate, initialArimaNumber,
                                 xregModel, xregNumber, initialXregEstimate,
                                 constantRequired, constantEstimate){

    occurrenceTransformer <- function(value){
        value <- switch(occurrence,
                        "odds-ratio"         = value / (1 - value),
                        "inverse-odds-ratio" = (1 - value) / value,
                        "fixed"              =,
                        "direct"             = value,
                        value);
        if(Etype == "A" && occurrence %in% c("odds-ratio","inverse-odds-ratio")){
            value <- log(value);
        }
        return(value);
    }

    # j tracks the number of ETS rows already consumed in matVt.
    # When etsModel is FALSE (e.g. pure ARIMA), there is no level row at index 1.
    j <- 0;
    levelOriginal <- if(etsModel) matVt[1, 1] else NA_real_;

    if(etsModel){
        #### Level ####
        if(initialLevelEstimate){
            # Failsafe in case of negative level
            if(matVt[1, 1]<0 || matVt[1, 1]>1){
                levelOriginal <- mean(ot);
                matVt[1, 1:lagsModelMax] <- levelOriginal;
            }
            matVt[1, 1:lagsModelMax] <- occurrenceTransformer(matVt[1, 1:lagsModelMax]);
        }
        j <- 1;

        #### Trend ####
        if(modelIsTrendy){
            if(initialTrendEstimate){
                # levels <- switch(Ttype,
                #                  "A"=levelOriginal + c(0, matVt[j+1, 1]),
                #                  "M"=levelOriginal * c(1, matVt[j+1, 1]));
                # levels[] <- occurrenceTransformer(levels);
                # if(Ttype=="A"){
                #     matVt[j+1, 1:lagsModelMax] <- diff(levels);
                # }
                # else{
                #     matVt[j+1, 1:lagsModelMax] <- levels[2]/levels[1];
                # }
                if(Ttype=="A"){
                    matVt[j+1, 1:lagsModelMax] <- 0;
                }
                else{
                    matVt[j+1, 1:lagsModelMax] <- 1;
                }
            }
            j[] <- j + 1;
        }

        #### Seasonal ####
        if(modelIsSeasonal){
            if(any(initialSeasonalEstimate)){
                for(i in 1:componentsNumberETSSeasonal){
                    seasonalOcc <- matVt[j+i, 1:lagsModelSeasonal[i]];
                    if(Stype=="M"){
                        # Transform this into the "multiplicative" seasonality
                        seasonalOcc[] <- seasonalOcc / levelOriginal + 1;
                    }
                    else{
                        # If additive, normalise
                        seasonalOcc[] <- seasonalOcc - mean(seasonalOcc);
                    }
                    matVt[j+i, 1:lagsModelSeasonal[i]] <- seasonalOcc;
                }
            }
            j[] <- j + componentsNumberETSSeasonal;
        }
    }

    #### ARIMA ####
    # ARIMA initial states live in the same raw state-space as the level
    # *after* the level has been mapped onto the model-native scale; they are
    # not probabilities. Running occurrenceTransformer() on them turns the
    # default seed of 0 into log(0) = -Inf for Etype="A", which corrupts the
    # initial parameter vector handed to nloptr.
    if(arimaModel){
        j[] <- j + componentsNumberARIMA;
    }

    #### Xreg ####
    # xreg coefficients are regression weights, not probabilities. They can be
    # negative; running log(value/(1-value)) on a negative coefficient gives
    # NaN, which corrupts the initial parameter vector handed to nloptr.
    if(xregModel){
        j[] <- j + xregNumber;
    }

    #### Constant ####
    # Same reasoning as for xreg/ARIMA: the constant is on the same scale as
    # the (already-transformed) level, so transforming it again is a category
    # error. Leaving it untouched.

    return(matVt);
}


omLinkFunction <- function(x, Etype, occurrence){
    switch(occurrence,
           "odds-ratio"         = switch(Etype,
                                         "M" = x / (1 + x),
                                         "A" = exp(x) / (1 + exp(x)),
                                         x / (1 + x)),
           "inverse-odds-ratio" = switch(Etype,
                                         "M" = 1 / (1 + x),
                                         "A" = 1 / (1 + exp(x)),
                                         1 / (1 + x)),
           "fixed"              =,
           "direct"             = pmin(pmax(x, 0), 1),
           x);
}

# File-scope occurrence-model cost function. Used by omEstimator() during
# optimisation and by the modelDo=="use" branch of om() for FI computation.
# Takes all dependencies as explicit arguments — no closure state.
omCF_local <- function(B,
                       etsModel, Etype, Ttype, Stype,
                       modelIsTrendy, modelIsSeasonal,
                       componentsNumberETS, componentsNumberETSNonSeasonal,
                       componentsNumberETSSeasonal, componentsNumberARIMA,
                       lags, lagsModel, lagsModelMax, lagsModelAll,
                       indexLookupTable, profilesRecentTable,
                       matVt, matWt, matF, vecG,
                       persistenceEstimate, persistenceLevelEstimate,
                       persistenceTrendEstimate, persistenceSeasonalEstimate,
                       persistenceXregEstimate, phiEstimate,
                       initialType, initialEstimate,
                       initialLevelEstimate, initialTrendEstimate,
                       initialSeasonalEstimate, initialArimaEstimate,
                       initialXregEstimate, initialArimaNumber,
                       arimaModel, arEstimate, maEstimate,
                       arOrders, iOrders, maOrders,
                       arRequired, maRequired, armaParameters,
                       nonZeroARI, nonZeroMA, arimaPolynomials,
                       arPolynomialMatrix, maPolynomialMatrix,
                       xregModel, xregNumber,
                       xregParametersMissing, xregParametersIncluded,
                       xregParametersEstimated, xregParametersPersistence,
                       constantRequired, constantEstimate,
                       bounds, regressors, loss,
                       ot, otLogical, obsInSample,
                       nIterations,
                       occurrence, occurrenceChar,
                       adamCpp,
                       lambda = 0, lossFunction = NULL){
    adamElements <- adam_filler(B,
                                etsModel, Etype, Ttype, Stype,
                                modelIsTrendy, modelIsSeasonal,
                                componentsNumberETS, componentsNumberETSNonSeasonal,
                                componentsNumberETSSeasonal, componentsNumberARIMA,
                                lags, lagsModel, lagsModelMax,
                                matVt, matWt, matF, vecG,
                                persistenceEstimate, persistenceLevelEstimate,
                                persistenceTrendEstimate, persistenceSeasonalEstimate,
                                persistenceXregEstimate, phiEstimate,
                                initialType, initialEstimate,
                                initialLevelEstimate, initialTrendEstimate,
                                initialSeasonalEstimate, initialArimaEstimate,
                                initialXregEstimate,
                                arimaModel, arEstimate, maEstimate,
                                arOrders, iOrders, maOrders,
                                arRequired, maRequired, armaParameters,
                                nonZeroARI, nonZeroMA, arimaPolynomials,
                                xregModel, xregNumber,
                                xregParametersMissing, xregParametersIncluded,
                                xregParametersEstimated, xregParametersPersistence,
                                constantEstimate, adamCpp,
                                constantRequired, initialArimaNumber);
    penalty <- adam_bounds_checker(adamElements, adamElements$arimaPolynomials,
                                   bounds,
                                   etsModel, modelIsTrendy, modelIsSeasonal,
                                   componentsNumberETS, componentsNumberETSNonSeasonal,
                                   componentsNumberETSSeasonal,
                                   arimaModel, arEstimate, maEstimate,
                                   xregModel, regressors, xregNumber,
                                   componentsNumberARIMA,
                                   lagsModelAll, obsInSample,
                                   arPolynomialMatrix, maPolynomialMatrix,
                                   phiEstimate);
    if(penalty != 0){
        return(penalty);
    }
    profilesRecentTable[] <- adamElements$matVt[, 1:lagsModelMax];
    # Fit dispatcher: for initial="gradient" this solves the initials by
    # profiling the occurrence loss over the probability residuals (in C++);
    # otherwise (or when out of scope) it is the ordinary occurrence fit, with
    # gradient joining the backcasting group as a fall-back.
    adamFitted <- adam_fitOrGradient(adamElements$matVt, adamElements$matWt,
                                     adamElements$matF, adamElements$vecG,
                                     indexLookupTable, profilesRecentTable,
                                     as.numeric(ot), as.numeric(ot),
                                     initialType, nIterations, adamCpp,
                                     etsModel, arimaModel, xregModel, Etype, Ttype, Stype,
                                     componentsNumberETS, componentsNumberETSSeasonal,
                                     componentsNumberETSNonSeasonal, lagsModel, lagsModelMax,
                                     obsInSample, loss, oType=occurrenceChar,
                                     componentsNumberARIMA=componentsNumberARIMA,
                                     lagsModelAll=lagsModelAll, xregNumber=xregNumber);
    yFitted <- omLinkFunction(adamFitted$fitted, Etype, occurrence);
    if(any(is.nan(yFitted)) || any(yFitted<0) || any(yFitted>1)){
        return(1e+300);
    }
    # Loss dispatch — mirrors R/adam.R:885-940 single-step block. ``errors``
    # are on the probability scale (``ot - yFitted``) since the OM target is
    # binary. The LASSO/RIDGE branch follows the same penalty structure as
    # adam() (R/adam.R:894-937).
    errors <- as.numeric(ot) - yFitted;
    if(loss == "custom"){
        CFValue <- lossFunction(actual=as.numeric(ot), fitted=yFitted, B=B);
    }
    else if(loss == "likelihood"){
        CFValue <- -(sum(log(yFitted[otLogical])) + sum(log(1 - yFitted[!otLogical])));
    }
    else if(loss == "MSE"){
        CFValue <- mean(errors^2);
    }
    else if(loss == "MAE"){
        CFValue <- mean(abs(errors));
    }
    else if(loss == "HAM"){
        CFValue <- mean(sqrt(abs(errors)));
    }
    else if(any(loss == c("LASSO","RIDGE"))){
        # Trim initials out of B for the penalty — same convention as
        # adam() (R/adam.R:897-916). For non-ARIMA OM the typical B is just
        # the persistence; we keep that intact and drop nothing if there
        # are no initials in B.
        BPenalty <- B;
        errorTerm <- (1 - lambda) * sqrt(mean(errors^2));
        if(loss == "LASSO"){
            CFValue <- errorTerm + lambda * sum(abs(BPenalty));
        } else {
            CFValue <- errorTerm + lambda * sqrt(sum(BPenalty^2));
        }
    }
    else {
        # Fallback to MSE — preserves the pre-existing behaviour for any
        # unknown loss string that match.arg didn't catch upstream.
        CFValue <- mean(errors^2);
    }
    return(CFValue)
}

#' @export
coefbootstrap.om <- function(object, nsim=1000, size=floor(0.75*nobs(object)),
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

    # Coefficients of the model
    coefficientsOriginal <- coef(object);
    nVariables <- length(coefficientsOriginal);
    variablesNames <- names(coefficientsOriginal);
    obsInsample <- nobs(object);

    coefBootstrap <- matrix(0, nsim, nVariables, dimnames=list(NULL, variablesNames));
    indices <- c(1:obsInsample);

    # Form the call for om()
    newCall <- object$call;
    newCall$silent <- TRUE;
    if(newCall[[1]]=="auto.om"){
        newCall[[1]] <- as.symbol("om");
    }
    newCall$holdout <- FALSE;
    newCall$model <- modelType(object);
    newCall$occurrence <- object$occurrence;
    if(!is.null(object$call$orders$select)){
        newCall$orders <- orders(object);
        newCall$orders$select <- FALSE;
    }
    newCall$constant <- !is.null(object$constant);
    lags <- lags(object);
    newCall$lags <- lags;
    newCall$B <- object$B;
    newCall$lb <- rep(-Inf, length(object$B));
    newCall$ub <- rep(Inf, length(object$B));
    newCall$data <- object$data;

    regressionPure <- substr(object$model,1,10)=="Regression";

    obsMinimum <- max(lags, nVariables) + 2;
    if(obsMinimum>=obsInsample && method=="cr"){
        warning("Not enough observations to do Case Resampling bootstrap. Changing method to 'dsr'.",
                call.=FALSE, immediate.=TRUE);
        method <- "dsr";
    }

    # If this is backcasting (or gradient, which also derives the initials
    # from the data), do sampling with moving origin
    changeOrigin <- any(object$initialType==c("backcasting","complete","gradient"));

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

    #### Bootstrap the data
    if(method=="dsr"){
        #### Data Shape Replication bootstrap (on the 0/1 occurrence response)
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
                    coefBootstrap[i,variablesNames %in% names(coef(testModel))] <- coef(testModel);
                }
            }
        }
        else{
            coefBootstrapParallel <- foreach::`%dopar%`(foreach::foreach(i=1:nsim),{
                newCall$data <- dataBoot$boot[,i];
                testModel <- tryCatch(eval(newCall), error=function(e) NULL);
                if(is.null(testModel)){ return(NULL); }
                return(coef(testModel));
            })
            for(i in 1:nsim){
                if(!is.null(coefBootstrapParallel[[i]])){
                    coefBootstrap[i,variablesNames %in% names(coefBootstrapParallel[[i]])] <- coefBootstrapParallel[[i]];
                }
            }
        }
    }
    else{
        #### Case Resampling bootstrap
        # A subsample can be too short / too sparse (few non-zero occurrences)
        # for the model to be re-estimated. Such draws are resampled (up to
        # maxAttempts) so every replicate is a genuine re-estimation with the
        # provided B as the starting point.
        maxAttempts <- 100L;
        refitOM <- function(){
            testModel <- NULL; attempt <- 0L;
            while(is.null(testModel) && attempt < maxAttempts){
                attempt <- attempt + 1L;
                subsetValues <- sampler(indices,size,replace,prob,regressionPure,changeOrigin);
                newCall$data <- object$data[subsetValues];
                testModel <- tryCatch(suppressWarnings(eval(newCall)), error=function(e) NULL);
                if(!is.null(testModel) && length(coef(testModel))==0){ testModel <- NULL; }
            }
            return(testModel);
        }
        if(!parallel){
            for(i in 1:nsim){
                testModel <- refitOM();
                if(!is.null(testModel)){
                    coefBootstrap[i,variablesNames %in% names(coef(testModel))] <- coef(testModel);
                }
            }
        }
        else{
            coefBootstrapParallel <- foreach::`%dopar%`(foreach::foreach(i=1:nsim),{
                testModel <- refitOM();
                if(is.null(testModel)){ return(NULL); }
                return(coef(testModel));
            })
            for(i in 1:nsim){
                if(!is.null(coefBootstrapParallel[[i]])){
                    coefBootstrap[i,variablesNames %in% names(coefBootstrapParallel[[i]])] <- coefBootstrapParallel[[i]];
                }
            }
        }
    }

    if(parallel && !is.null(cluster)){
        parallel::stopCluster(cluster);
    }

    # Get rid of NAs. They mean "zero"
    coefBootstrap[is.na(coefBootstrap)] <- 0;
    colnames(coefBootstrap) <- names(coefficientsOriginal);

    # Centre the coefficients for the calculation of the vcov
    coefvcov <- coefBootstrap - matrix(coefficientsOriginal, nsim, nVariables, byrow=TRUE);

    return(structure(list(vcov=(t(coefvcov) %*% coefvcov)/nsim,
                          coefficients=coefBootstrap, method=method,
                          nsim=nsim, size=NA, replace=NA, prob=NA,
                          parallel=parallel, model=object$call[[1]], timeElapsed=Sys.time()-startTime),
                     class="bootstrap"));
}

#' @export
vcov.om <- function(object, type=c("opg","hessian","bootstrap"),
                    bootstrap=FALSE, heuristics=NULL, ...){
    ellipsis <- list(...);
    type <- covarTypeResolver(type, bootstrap);

    if(!is.null(heuristics) && is.numeric(heuristics)){
        return(diag(abs(coef(object)) * heuristics));
    }

    if(type=="bootstrap"){
        return(coefbootstrap(object, ...)$vcov);
    }

    stepSize <- if(is.null(ellipsis$stepSize)) {
        .Machine$double.eps^(1/4);
    } else {
        ellipsis$stepSize;
    };

    # OPG / BHHH covariance J = sum_t s_t s_t' -- PSD by construction, so it
    # returns finite standard errors at boundary estimates where the observed
    # Fisher Information (below) is indefinite. Falls back to the Hessian if the
    # reproduction guard trips (covarOPGom returns NULL).
    if(type=="opg"){
        vcovOPG <- covarOPGom(object, stepSize=stepSize);
        if(!is.null(vcovOPG)){
            return(vcovOPG);
        }
        warning("The OPG covariance could not be computed for this om model; ",
                "falling back to the observed Fisher Information.", call.=FALSE);
    }

    h <- if(any(!is.na(object$forecast))) length(object$forecast) else 0;

    modelReturn <- suppressWarnings(
        om(object$data, h=h, model=object,
           formula=formula(object), FI=TRUE, stepSize=stepSize));

    if(is.null(modelReturn$FI)){
        stop("Could not compute Fisher Information for this om model. ",
             "Try a different stepSize.", call.=FALSE);
    }

    # Rows / cols that are all zero (or contain NaN) carry no information.
    brokenVariables <- apply(modelReturn$FI==0, 1, all) |
                       apply(is.nan(modelReturn$FI), 1, any);
    if(any(brokenVariables)){
        modelReturn <- suppressWarnings(
            om(object$data, h=h, model=object,
               formula=formula(object), FI=TRUE,
               stepSize=.Machine$double.eps^(1/6)));
        brokenVariables <- apply(modelReturn$FI==0, 1, all);
    }
    if(any(is.nan(modelReturn$FI))){
        stop("Fisher Information contains NaN; try a different stepSize ",
             "(e.g. stepSize=1e-6).", call.=FALSE);
    }
    if(any(eigen(modelReturn$FI, only.values=TRUE)$values < 0)){
        warning("Observed Fisher Information is not positive semi-definite; ",
                "covariance matrix may be unreliable.", call.=FALSE);
    }

    FIMatrix <- modelReturn$FI[!brokenVariables, !brokenVariables, drop=FALSE];
    vcovMatrix <- try(chol2inv(chol(FIMatrix)), silent=TRUE);
    if(inherits(vcovMatrix, "try-error")){
        vcovMatrix <- try(solve(FIMatrix, diag(ncol(FIMatrix)), tol=1e-20),
                          silent=TRUE);
        if(inherits(vcovMatrix, "try-error")){
            warning("Hessian is singular; cannot invert.", call.=FALSE);
            vcovMatrix <- diag(1e+100, ncol(FIMatrix));
        }
    }
    modelReturn$FI[!brokenVariables, !brokenVariables] <- vcovMatrix;
    modelReturn$FI[brokenVariables, ] <- Inf;
    modelReturn$FI[, brokenVariables] <- Inf;
    diag(modelReturn$FI) <- abs(diag(modelReturn$FI));
    return(modelReturn$FI);
}

#' @rdname forecast.smooth
#' @export
forecast.om <- function(object, h=10, ...){
    # Intervals on the probability scale are not implemented yet, so the
    # underlying forecast.adam() call is forced to interval="none". The
    # remaining slots (level, side, cumulative) are set to their defaults so
    # the returned structure is shape-compatible with forecast.adam().
    fc <- forecast.adam(object, h=h, interval="none",
                        level=0.95, side="both", cumulative=FALSE, ...);

    Etype <- errorType(object);
    occurrence <- object$occurrence;
    fc$mean[] <- omLinkFunction(fc$mean, Etype, occurrence);
    if(occurrence %in% c("odds-ratio","inverse-odds-ratio") && Etype == "A"){
        fc$mean[is.nan(fc$mean)] <- 1;
    }

    return(fc);
}

#' @export
actuals.om <- function(object, ...){
    if(is.null(object$data)){
        return(NULL);
    }
    yObs <- if(is.data.frame(object$data) || is.matrix(object$data)) object$data[,1] else object$data;
    yObs[] <- (yObs != 0) * 1;
    return(yObs);
}

#' @importFrom stats sigma
#' @export
sigma.om <- function(object, ...){
    # `sigma.adam` dispatches on `object$distribution`, which is "plogis"
    # for occurrence models -- not in its switch table, so it would
    # return `numeric(0)`. Use the link-scale RMS of the residuals
    # instead: occurrence residuals live on the logit / log-odds scale,
    # so their RMS is a meaningful scale parameter for the underlying
    # ETS. Without this, `s2 = NA^2 = NA` propagates through `covarAnal`
    # and breaks `multicov(om_obj)` and downstream callers.
    e <- residuals(object);
    if(is.null(e) || length(e)==0) return(NA_real_);
    return(sqrt(mean(e^2, na.rm=TRUE)));
}

#' @export
print.om <- function(x, ...){
    cat("Occurrence model\n");
    print.adam(x, ...);
}

#' @export
summary.om <- function(object, ...){
    cat("Occurrence model\n");
    summary.adam(object, ...);
}

#' @export
print.omCombined <- function(x, ...){
    cat("Occurrence model\n");
    print.adamCombined(x, ...);
}

#' @export
summary.omCombined <- function(object, ...){
    cat("Occurrence model\n");
    summary.adamCombined(object, ...);
}

#' @export
forecast.omCombined <- function(object, h=NULL, ...){
    # Interval-related slots are fixed internally so the returned structure
    # matches forecast.adamCombined's field set; intervals on the probability
    # scale are not implemented yet.
    interval   <- "none";
    level      <- 0.95;
    side       <- "both";
    cumulative <- FALSE;

    obsInSample <- nobs(object);
    yClasses <- class(actuals(object));
    if(is.null(h)){
        h <- length(object$forecast);
    }
    nLevels <- length(level);

    if(any(yClasses == "ts")){
        yForecastStart <- time(actuals(object))[obsInSample] + deltat(actuals(object));
        yFrequency <- frequency(actuals(object));
        yForecast <- ts(rep(0, h), start=yForecastStart, frequency=yFrequency);
        yLower <- yUpper <- ts(matrix(NA_real_, h, nLevels),
                               start=yForecastStart, frequency=yFrequency);
    } else {
        yIndex <- time(actuals(object));
        yForecastIndex <- yIndex[obsInSample] + diff(tail(yIndex, 2)) * (1:h);
        yForecast <- zoo(rep(0, h), order.by=yForecastIndex);
        yLower <- yUpper <- zoo(matrix(NA_real_, h, nLevels),
                                order.by=yForecastIndex);
    }

    object$ICw[object$ICw < 1e-2] <- 0;
    object$ICw[] <- object$ICw / sum(object$ICw);

    for(i in seq_along(object$models)){
        if(object$ICw[i] == 0){
            next;
        }
        fc_i <- forecast.adam(object$models[[i]], h=h, interval=interval,
                              level=level, side=side, cumulative=cumulative, ...);
        Etype_i <- errorType(object$models[[i]]);
        fc_i$mean[] <- omLinkFunction(fc_i$mean, Etype_i, object$occurrence);
        yForecast[] <- yForecast + fc_i$mean * object$ICw[i];
    }

    object$models <- NULL;
    return(structure(list(mean=yForecast,
                          lower=yLower, upper=yUpper,
                          model=object,
                          level=level, interval=interval,
                          side=side, cumulative=cumulative, h=h),
                     class=c("adam.forecast","smooth.forecast","forecast")));
}

#' @importFrom stats rstandard
#' @export
rstandard.om <- function(model, ...){
    obs <- nobs(model);
    df  <- obs - nparam(model);
    p   <- as.numeric(model$fitted);
    e   <- as.numeric(model$residuals);
    return(e / sqrt(p * (1 - p)) * sqrt(obs / df));
}

#' @importFrom stats rstudent
#' @export
rstudent.om <- function(model, ...){
    obs <- nobs(model);
    df  <- obs - nparam(model) - 1;
    p   <- as.numeric(model$fitted);
    e   <- as.numeric(model$residuals);
    return(e / sqrt(p * (1 - p)) * sqrt(obs / df));
}

#' @title Simulate methods for occurrence (om/omg) state-space models
#' @description Re-simulates probabilities and occurrence indicators
#' from a fitted \code{om} or \code{omg} model. The latent ETS is
#' simulated via the shared C++ kernel (the same one
#' \code{simulate.adam} uses), the latent series is mapped to a
#' probability via \code{omLinkFunction} (or \code{omgLinkFunction}
#' for \code{omg}), and a binomial draw with that probability gives
#' the 0/1 occurrence series.
#'
#' @param object An object of class \code{om} (or \code{omg}).
#' @param nsim Number of simulated series to draw.
#' @param seed Optional integer; forwarded to \code{set.seed} at the
#'   start of the simulation. Matches \code{stats::simulate}'s
#'   generic signature. When \code{NULL} (default) the global RNG
#'   state is used unchanged.
#' @param obs Number of observations per simulated series. Defaults
#'   to the in-sample length.
#' @param ... Currently unused; kept for forward compatibility.
#'
#' @return An S3 list of class \code{c("om.sim","oes.sim","smooth.sim")}
#'   (or \code{c("omg.sim","oes.sim","smooth.sim")} for \code{omg})
#'   with fields:
#'   \describe{
#'     \item{\code{$probability}}{Simulated probability series of
#'       shape \code{(obs, nsim)} -- the equivalent of
#'       \code{sim.oes()}'s \code{$probability} output.}
#'     \item{\code{$data}}{0/1 occurrence indicators of shape
#'       \code{(obs, nsim)}, drawn via \code{rbinom} with the
#'       simulated probability.}
#'     \item{\code{$states}, \code{$residuals}}{Latent state cube
#'       and the errors used internally.}
#'     \item{\code{$model}, \code{$occurrence}}{Identifiers carried
#'       over from the fit.}
#'     \item{\code{$latent}}{Pre-link state-space output --
#'       internal, used by \code{simulate.omg} to combine sub-models.}
#'   }
#'
#' @details
#' \code{print()} on the returned object dispatches to
#' \code{print.oes.sim} via the inherited \code{"oes.sim"} class.
#'
#' @examples
#' \dontrun{
#' set.seed(7)
#' y <- rbinom(120, 1, prob=0.3 + 0.005*(1:120))
#' m <- om(y, model="MNN", occurrence="odds-ratio", silent=TRUE)
#' sim <- simulate(m, nsim=5, seed=42)
#' range(sim$probability)
#' table(sim$data)
#' }
#'
#' @rdname simulate.om
#' @export
simulate.om <- function(object, nsim=1, seed=NULL, obs=nobs(object), ...){
    startTime <- Sys.time();
    if(!is.null(seed)){
        set.seed(seed);
    }

    # 1. Simulate the latent ETS via the shared helper. ``om`` inherits
    #    from ``adam`` so all required fields are present on ``object``.
    inner <- simulateADAMCore(object, nsim=nsim, obs=obs, ...);

    # 2. Apply the inverse link: latent state -> probability.
    Etype       <- errorType(object);
    occurrence  <- object$occurrence;
    obsInSample <- inner$obsInSample;

    latentMatrix <- matrix(inner$data, obsInSample, nsim);
    probability  <- matrix(omLinkFunction(c(latentMatrix), Etype, occurrence),
                           obsInSample, nsim);
    # Numerical guard: omLinkFunction's "fixed" / "direct" branches
    # already clip, but odds-ratio paths can overshoot under extreme
    # latent noise. Clamp uniformly here.
    probability[] <- pmin(pmax(probability, 0), 1);

    # 3. Draw 0/1 occurrence indicators.
    occurrenceData <- matrix(rbinom(obsInSample*nsim, 1, c(probability)),
                             obsInSample, nsim);

    # 4. Preserve time series structure from the in-sample series.
    yInSample <- actuals(object);
    yClasses  <- class(yInSample);
    colnames(probability)    <- paste0("nsim", c(1:nsim));
    colnames(occurrenceData) <- paste0("nsim", c(1:nsim));
    if(any(yClasses=="zoo")){
        yIndex <- time(yInSample);
        yIndexDiff <- diff(head(yIndex, 2));
        yTime <- yIndex[1] + yIndexDiff * c(1:(obsInSample-1));
        probability    <- zoo(probability,    order.by=yTime);
        occurrenceData <- zoo(occurrenceData, order.by=yTime);
    }
    else{
        probability <- ts(probability, start=start(yInSample),
                          frequency=frequency(yInSample));
        occurrenceData <- ts(occurrenceData, start=start(yInSample),
                             frequency=frequency(yInSample));
    }

    # Per-series log-likelihood, mirroring ``sim.oes`` (R/simoes.R:138-143).
    # Operate on a plain matrix so the ts/zoo attributes don't get
    # mangled by ``pmax``.
    probMat  <- matrix(as.numeric(probability), obsInSample, nsim);
    safeProb <- pmax(probMat, .Machine$double.eps);
    if(nsim==1){
        logLik <- sum(log(safeProb));
    }
    else{
        logLik <- colSums(log(safeProb));
    }

    return(structure(list(timeElapsed = Sys.time() - startTime,
                          model       = object$model,
                          occurrence  = occurrence,
                          probability = probability,
                          data        = occurrenceData,
                          ot          = occurrenceData,   # alias used by print.oes.sim
                          states      = inner$states,
                          residuals   = inner$matErrors,
                          latent      = latentMatrix,
                          logLik      = logLik,
                          other       = inner$ellipsis),
                     class=c("om.sim","oes.sim","smooth.sim")));
}
