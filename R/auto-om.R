#' Automatic Occurrence Model Selection
#'
#' Fits \link{om} for each supplied \code{occurrence} type and returns the
#' model with the lowest information criterion.
#'
#' @param data Numeric vector, time series, or data frame. Non-binary input is
#'   automatically binarised: any non-zero value becomes 1.
#' @param model Three-letter ETS specification (wildcards \code{"Z"}/\code{"X"}/\code{"Y"} supported).
#' @param lags Vector of seasonal lags. Defaults to \code{frequency(data)}.
#' @param orders ARIMA orders: \code{list(ar, i, ma, select=TRUE/FALSE)}.
#'   When \code{select=TRUE}, ARIMA orders are selected automatically for each
#'   occurrence type.
#' @param formula Optional formula for external regressors.
#' @param regressors How to handle regressors: \code{"use"}, \code{"select"}, or \code{"adapt"}.
#' @param occurrence Character vector of occurrence link types to try:
#'   \code{"fixed"}, \code{"odds-ratio"}, \code{"inverse-odds-ratio"}, \code{"direct"}.
#' @param h Forecast horizon.
#' @param holdout If \code{TRUE}, a holdout sample of size \code{h} is withheld.
#' @param persistence Optional persistence (smoothing) parameter vector.
#' @param phi Optional damping parameter.
#' @param initial Initialisation method: \code{"backcasting"}, \code{"optimal"},
#'   \code{"two-stage"}, or \code{"complete"}.
#' @param arma Optional fixed ARMA parameters.
#' @param ic Information criterion used for selection.
#' @param bounds Parameter bounds type.
#' @param silent If \code{TRUE}, suppresses progress messages and plots.
#' @param parallel If \code{TRUE} (or a core count), fit occurrence types in
#'   parallel.  Requires \code{foreach} and \code{doMC} / \code{doParallel}.
#' @param ets ETS flavour passed to \link{om}: \code{"conventional"} or \code{"adam"}.
#' @param ... Additional arguments forwarded to \link{om}.
#'
#' @return The best \code{om} object (lowest IC), with \code{$call} set to
#'   the \code{auto.om()} call and \code{$timeElapsed} recording wall time.
#'
#' @seealso \link{om}, \link{auto.adam}
#'
#' @examples
#' set.seed(42)
#' y <- rbinom(120, 1, 0.6)
#' \donttest{m <- auto.om(y, occurrence = c("fixed", "odds-ratio"))}
#'
#' @export
auto.om <- function(data,
                    model      = "ZXZ",
                    lags       = c(frequency(data)),
                    orders     = list(ar=c(3,3), i=c(2,1), ma=c(3,3), select=TRUE),
                    formula    = NULL,
                    regressors = c("use","select","adapt"),
                    occurrence = c("fixed","odds-ratio","inverse-odds-ratio","direct","general"),
                    h = 0, holdout = FALSE,
                    persistence = NULL, phi = NULL,
                    initial = c("backcasting","optimal","two-stage","complete","gradient"),
                    arma = NULL,
                    ic      = c("AICc","AIC","BIC","BICc"),
                    bounds  = c("usual","admissible","none"),
                    silent  = TRUE,
                    parallel = FALSE,
                    ets     = c("conventional","adam"),
                    ...){
    startTime <- Sys.time();
    cl <- match.call();

    #### IC / matching ####
    ic      <- match.arg(ic, c("AICc","AIC","BIC","BICc"));
    IC      <- switch(ic, "AIC"=AIC, "AICc"=AICc, "BIC"=BIC, "BICc"=BICc);
    initial <- match.arg(initial);
    ets     <- match.arg(ets);
    regressors <- match.arg(regressors);

    occurrence <- match.arg(occurrence,
                            c("fixed","odds-ratio","inverse-odds-ratio","direct","general"),
                            several.ok=TRUE);

    if(any(unlist(strsplit(model,""))=="C")){
        modelDo <- "combine";
    }
    else{
        modelDo <- "select";
    }

    #### Data prep ####
    if(is.adam.sim(data) || is.smooth.sim(data)){
        data <- data$data;
    }
    else if(inherits(data,"Mdata")){
        h <- data$h;
        holdout <- TRUE;
        lags <- frequency(data$x);
        data <- ts(c(data$x, data$xx), start=start(data$x), frequency=frequency(data$x));
    }

    if(is.null(dim(data))){
        obsInSample <- length(data) - holdout*h;
    }
    else{
        obsInSample <- nrow(data) - holdout*h;
        if(is.null(formula)){
            responseName <- colnames(data)[1];
        }
        else{
            responseName <- all.vars(formula)[1];
        }
    }

    if(!is.null(dim(data))){
        yInSample <- if(is.null(formula)) data[1:obsInSample, 1] else
            data[1:obsInSample, all.vars(formula)[1]];
    }
    else{
        yInSample  <- data[1:obsInSample];
        responseName <- paste0(deparse(substitute(data)), collapse="");
        responseName <- make.names(responseName, unique=TRUE);
    }

    #### Model flags ####
    etsModel  <- all(model != "NNN");
    xregModel <- (!is.null(dim(data)) && ncol(data) > 1);

    #### ARIMA parameter processing ####
    if(is.list(orders)){
        arimaModel       <- any(c(orders$ar, orders$i, orders$ma) > 0);
    }
    else{
        arimaModel <- any(orders > 0);
    }

    if(arimaModel){
        arimaModelSelect <- FALSE;
        if(is.list(orders)){
            arimaModelSelect <- isTRUE(orders$select);
            arMax <- orders$ar;
            iMax  <- orders$i;
            maMax <- orders$ma;
        }
        else{
            arMax <- orders[1];
            iMax  <- orders[2];
            maMax <- orders[3];
        }

        if(any(c(arMax, iMax, maMax) < 0)){
            stop("Funny guy! How am I gonna construct a model with negative order?", call.=FALSE);
        }

        if(sum(lags==1) == 0){
            lags <- c(1, lags);
        }
        if(any(lags==0)){
            arMax <- arMax[lags!=0];
            iMax  <- iMax[lags!=0];
            maMax <- maMax[lags!=0];
            lags  <- lags[lags!=0];
        }

        maxorder <- max(length(arMax), length(iMax), length(maMax), length(lags));
        if(length(arMax) != maxorder){ arMax <- c(arMax, rep(0, maxorder-length(arMax))); }
        if(length(iMax)  != maxorder){ iMax  <- c(iMax,  rep(0, maxorder-length(iMax)));  }
        if(length(maMax) != maxorder){ maMax <- c(maMax, rep(0, maxorder-length(maMax))); }

        if(any((arMax + iMax + maMax) == 0)){
            orders2leave <- (arMax + iMax + maMax) != 0;
            if(all(!orders2leave)){ orders2leave <- lags == min(lags); }
            arMax <- arMax[orders2leave];
            iMax  <- iMax[orders2leave];
            maMax <- maMax[orders2leave];
            lags  <- lags[orders2leave];
        }

        if(length(unique(lags)) != length(lags)){
            lagsNew  <- unique(lags);
            arMaxNew <- iMaxNew <- maMaxNew <- lagsNew;
            for(i in seq_along(lagsNew)){
                arMaxNew[i] <- max(arMax[which(lags==lagsNew[i])], na.rm=TRUE);
                iMaxNew[i]  <- max(iMax[which(lags==lagsNew[i])],  na.rm=TRUE);
                maMaxNew[i] <- max(maMax[which(lags==lagsNew[i])], na.rm=TRUE);
            }
            arMax <- arMaxNew;
            iMax  <- iMaxNew;
            maMax <- maMaxNew;
            lags  <- lagsNew;
        }

        arMax <- arMax[order(lags)];
        iMax  <- iMax[order(lags)];
        maMax <- maMax[order(lags)];
        lags  <- sort(lags);
    }
    else{
        arMax <- iMax <- maMax <- NULL;
        arimaModelSelect <- FALSE;
        orders <- c(0,0,0);
    }

    nModels <- length(occurrence);

    #### Parallel setup ####
    setupResult <- adam_setupParallel(parallel, nModels);
    parallel    <- setupResult$parallel;
    cluster     <- setupResult$cluster;

    if(!silent){
        if(!parallel){
            cat("Evaluating occurrence models... ");
        }
        else{
            cat("Working... ");
        }
    }

    #### omReturner: fits om for each occurrence type ####
    omReturner <- function(data, model, lags, orders,
                           occurrence, h, holdout,
                           persistence, phi, initial, arma,
                           ic, bounds, regressors, parallel,
                           arimaModelSelect, arMax, iMax, maMax, ...){
        ordersToUse <- if(arimaModelSelect) list(ar=0, i=0, ma=0, select=FALSE) else orders;

        if(!parallel){
            selectedModels <- vector("list", length(occurrence));
            for(i in seq_along(occurrence)){
                if(!silent){ cat(occurrence[i], "\b, "); }
                if(etsModel || xregModel || (arimaModel && !arimaModelSelect) ||
                   occurrence[i] == "general"){
                    selectedModels[[i]] <- om(data=data, model=model, lags=lags,
                                              orders=ordersToUse,
                                              occurrence=occurrence[i], formula=formula,
                                              h=h, holdout=holdout,
                                              persistence=persistence, phi=phi,
                                              initial=initial, arma=arma,
                                              ic=ic, bounds=bounds,
                                              regressors=regressors,
                                              silent=TRUE, ets=ets, ...);
                }
                if(arimaModelSelect && occurrence[i] != "general"){
                    selectedModels[[i]] <- adam_arimaSelector(
                        data=data, model=model,
                        lags=lags, arMax=arMax, iMax=iMax, maMax=maMax,
                        h=h, holdout=holdout,
                        persistence=persistence, phi=phi, initial=initial,
                        ic=ic, bounds=bounds, silent=silent, regressors=regressors,
                        testModelETS=selectedModels[[i]],
                        fitter=om, fitter_args=list(occurrence=occurrence[i]),
                        IC=IC, formula=formula, ets=ets,
                        responseName=responseName, obsInSample=obsInSample, ...);
                }
            }
        }
        else{
            selectedModels <- foreach::`%dopar%`(
                foreach::foreach(i=seq_along(occurrence)),
                {
                    if(etsModel || xregModel || occurrence[i] == "general"){
                        testModel <- om(data=data, model=model, lags=lags,
                                        orders=ordersToUse,
                                        occurrence=occurrence[i], formula=formula,
                                        h=h, holdout=holdout,
                                        persistence=persistence, phi=phi,
                                        initial=initial, arma=arma,
                                        ic=ic, bounds=bounds,
                                        regressors=regressors,
                                        silent=TRUE, ets=ets, ...);
                    }
                    else{
                        testModel <- NULL;
                    }
                    if(arimaModelSelect && occurrence[i] != "general"){
                        testModel <- adam_arimaSelector(
                            data=data, model=model,
                            lags=lags, arMax=arMax, iMax=iMax, maMax=maMax,
                            h=h, holdout=holdout,
                            persistence=persistence, phi=phi, initial=initial,
                            ic=ic, bounds=bounds, silent=TRUE, regressors=regressors,
                            testModelETS=testModel,
                            fitter=om, fitter_args=list(occurrence=occurrence[i]),
                            IC=IC, formula=formula, ets=ets,
                            responseName=responseName, obsInSample=obsInSample, ...);
                    }
                    return(testModel);
                });
        }
        return(selectedModels);
    }

    if(arimaModelSelect && !is.null(arma)){
        warning("ARIMA order selection cannot be done with provided arma parameters. Dropping them.",
                call.=FALSE);
        arma <- NULL;
    }

    #### Run models ####
    if(!arimaModelSelect){
        selectedModels <- omReturner(data, model, lags, orders,
                                     occurrence, h, holdout,
                                     persistence, phi, initial, arma,
                                     ic, bounds, regressors, parallel,
                                     arimaModelSelect, arMax, iMax, maMax, ...);
    }
    else if(etsModel || xregModel){
        selectedModels <- omReturner(data, model, lags, orders,
                                     occurrence, h, holdout,
                                     persistence, phi, initial, arma,
                                     ic, bounds, regressors, parallel,
                                     arimaModelSelect, arMax, iMax, maMax, ...);
    }
    else{
        if(!parallel){
            selectedModels <- vector("list", length(occurrence));
            for(i in seq_along(occurrence)){
                if(!silent){ cat(occurrence[i], "\b: "); }
                if(occurrence[i] != "general"){
                    selectedModels[[i]] <- adam_arimaSelector(
                        data=data, model=model,
                        lags=lags, arMax=arMax, iMax=iMax, maMax=maMax,
                        h=h, holdout=holdout,
                        persistence=persistence, phi=phi, initial=initial,
                        ic=ic, bounds=bounds, silent=silent, regressors=regressors,
                        testModelETS=NULL,
                        fitter=om, fitter_args=list(occurrence=occurrence[i]),
                        IC=IC, formula=formula, ets=ets,
                        responseName=responseName, obsInSample=obsInSample, ...);
                } else {
                    selectedModels[[i]] <- om(data=data, model=model, lags=lags,
                                              orders=c(0,0,0), occurrence="general",
                                              formula=formula, h=h, holdout=holdout,
                                              persistence=persistence, phi=phi,
                                              initial=initial, arma=arma, ic=ic, bounds=bounds,
                                              regressors=regressors, silent=TRUE, ets=ets, ...);
                }
            }
        }
        else{
            selectedModels <- foreach::`%dopar%`(
                foreach::foreach(i=seq_along(occurrence)),
                {
                    if(occurrence[i] != "general"){
                        testModel <- adam_arimaSelector(
                            data=data, model=model,
                            lags=lags, arMax=arMax, iMax=iMax, maMax=maMax,
                            h=h, holdout=holdout,
                            persistence=persistence, phi=phi, initial=initial,
                            ic=ic, bounds=bounds, silent=TRUE, regressors=regressors,
                            testModelETS=NULL,
                            fitter=om, fitter_args=list(occurrence=occurrence[i]),
                            IC=IC, formula=formula, ets=ets,
                            responseName=responseName, obsInSample=obsInSample, ...);
                    } else {
                        testModel <- om(data=data, model=model, lags=lags,
                                        orders=c(0,0,0), occurrence="general",
                                        formula=formula, h=h, holdout=holdout,
                                        persistence=persistence, phi=phi,
                                        initial=initial, arma=arma, ic=ic, bounds=bounds,
                                        regressors=regressors, silent=TRUE, ets=ets, ...);
                    }
                    return(testModel);
                });
        }
    }

    #### Select best ####
    safeIC <- function(m){
        if(!is.null(m$ICw) && !is.null(m$ICs) && is.numeric(m$ICs) && is.numeric(m$ICw)){
            return(as.numeric(m$ICs[is.finite(m$ICs)] %*% m$ICw[is.finite(m$ICs)]));
        }
        tryCatch(IC(m), warning=function(w) Inf, error=function(e) Inf);
    }
    ICValues <- vapply(selectedModels, safeIC, numeric(1));

    best <- which.min(ICValues);
    selectedModels[[best]]$timeElapsed <- Sys.time() - startTime;
    selectedModels[[best]]$call <- cl;

    if(!silent){
        cat("Done!\n");
        plot(selectedModels[[best]], 7);
    }

    adam_teardownParallel(cluster);

    return(selectedModels[[best]]);
}
