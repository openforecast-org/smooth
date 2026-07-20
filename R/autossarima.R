#' State Space ARIMA
#'
#' Function selects the best State Space ARIMA based on information criteria,
#' using fancy branch and bound mechanism. The resulting model can be not
#' optimal in IC meaning, but it is usually reasonable.
#'
#' The function constructs bunch of ARIMAs in Single Source of Error
#' state space form (see \link[smooth]{ssarima} documentation) and selects the
#' best one based on information criterion. The mechanism is described in
#' Svetunkov & Boylan (2019).
#'
#' Due to the flexibility of the model, multiple seasonalities can be used. For
#' example, something crazy like this can be constructed:
#' SARIMA(1,1,1)(0,1,1)[24](2,0,1)[24*7](0,0,1)[24*30], but the estimation may
#' take a lot of time... It is recommended to use \link[smooth]{auto.msarima} in
#' cases with more than one seasonality and high frequencies.
#'
#' For some more information about the model and its implementation, see the
#' vignette: \code{vignette("ssarima","smooth")}
#'
#' @template ssBasicParam
#' @template ssAdvancedParam
#' @template ssXregParam
#' @template ssInitialParam
#' @template ssAuthor
#' @template ssKeywords
#'
#' @template ssGeneralRef
#' @template ssIntermittentRef
#' @template ssARIMARef
#'
#' @param ic The information criterion to use in the model selection.
#' @param orders List of maximum orders to check, containing vector variables
#' \code{ar}, \code{i} and \code{ma}. If a variable is not provided in the
#' list, then it is assumed to be equal to zero. At least one variable should
#' have the same length as \code{lags}.
#' @param lags Defines lags for the corresponding orders (see examples). The
#' length of \code{lags} must correspond to the length of \code{orders}. There
#' is no restrictions on the length of \code{lags} vector.
#' @param fast If \code{TRUE}, then some of the orders of ARIMA are
#' skipped. This is not advised for models with \code{lags} greater than 12.
#' @param constant If \code{NULL}, then the function will check if constant is
#' needed. if \code{TRUE}, then constant is forced in the model. Otherwise
#' constant is not used.
#' @param ...  Other non-documented parameters. For example \code{FI=TRUE} will
#' make the function also produce Fisher Information matrix, which then can be
#' used to calculated variances of parameters of the model.  Maximum orders to
#' check can also be specified separately, however \code{orders} variable must
#' be set to \code{NULL}: \code{ar.orders} - Maximum order of AR term. Can be
#' vector, defining max orders of AR, SAR etc.  \code{i.orders} - Maximum order
#' of I. Can be vector, defining max orders of I, SI etc.  \code{ma.orders} -
#' Maximum order of MA term. Can be vector, defining max orders of MA, SMA etc.
#' @return Object of class "smooth" is returned. See \link[smooth]{ssarima} for
#' details.
#' @seealso \code{\link[smooth]{es}, \link[smooth]{ces},
#' \link[smooth]{sim.es}, \link[smooth]{gum}, \link[smooth]{ssarima}}
#'
#' @examples
#'
#' \donttest{set.seed(41)}
#' \donttest{x <- rnorm(118,100,3)}
#'
#' # The best ARIMA for the data
#' \donttest{ourModel <- auto.ssarima(x,orders=list(ar=c(2,1),i=c(1,1),ma=c(2,1)),lags=c(1,12),
#'                                    h=18,holdout=TRUE)}
#'
#' # The other one using optimised states
#' \donttest{auto.ssarima(x,orders=list(ar=c(3,2),i=c(2,1),ma=c(3,2)),lags=c(1,12),
#'                        initial="two",h=18,holdout=TRUE)}
#'
#' \donttest{summary(ourModel)
#' forecast(ourModel)
#' plot(forecast(ourModel))}
#'
#' @rdname ssarima
#' @export
auto.ssarima <- function(y, orders=list(ar=c(3,3),i=c(2,1),ma=c(3,3)), lags=c(1,frequency(y)),
                         fast=TRUE, constant=NULL,
                         initial=c("backcasting","optimal","two-stage","complete","gradient"),
                         loss=c("likelihood","MSE","MAE","HAM","MSEh","TMSE","GTMSE","MSCE","GPL"),
                         ic=c("AICc","AIC","BIC","BICc"),
                         h=0, holdout=FALSE, bounds=c("admissible","usual","none"), silent=TRUE,
                         xreg=NULL, regressors=c("use","select","adapt"),
                         ...){
# Function estimates several ssarima models and selects the best one using the selected information criterion.
#
#    Copyright (C) 2015 - Inf  Ivan Svetunkov

# Start measuring the time of calculations
    startTime <- Sys.time();

    ### Depricate the old parameters
    ellipsis <- list(...);

    ic <- match.arg(ic);
    IC <- switch(ic,
                 "AIC"=AIC,
                 "AICc"=AICc,
                 "BIC"=BIC,
                 "BICc"=BICc);

    # Switch of combinations
    combine <- FALSE;

    # If this is Mcomp data, then take the frequency from it
    if(any(class(y)=="Mdata")){
        if(all(lags %in% c(1,frequency(y)))){
            lags <- unique(c(lags,frequency(y$x)));
        }
        yInSample <- y$x;
        # Measure the sample size based on what was provided as data
        obsInSample <- length(y$x) - holdout*h;
    }
    else{
        # Measure the sample size based on what was provided as data
        obsInSample <- length(y) - holdout*h;
        yInSample <- y[1:obsInSample];
    }

    arMax <- orders$ar;
    iMax <- orders$i;
    maMax <- orders$ma;

    if(is.null(constant)){
        constantCheck <- TRUE;
        constantValue <- TRUE;
    }
    else{
        if(is.logical(constant)){
            constantCheck <- FALSE;
            constantValue <- constant;
        }
        else{
            constant <- NULL;
            constantCheck <- TRUE;
            constantValue <- TRUE;
            warning("Strange value of constant parameter. We changed it to default value.");
        }
    }

    if(any(is.complex(c(arMax,iMax,maMax,lags)))){
        stop("Come on! Be serious! This is ARIMA, not CES!",call.=FALSE);
    }

    if(any(c(arMax,iMax,maMax)<0)){
        stop("Funny guy! How am I gonna construct a model with negative order?",call.=FALSE);
    }

    if(any(c(lags)<0)){
        stop("Right! Why don't you try complex lags then, mister smart guy?",call.=FALSE);
    }

    # If there are zero lags, drop them
    if(any(lags==0)){
        arMax <- arMax[lags!=0];
        iMax <- iMax[lags!=0];
        maMax <- maMax[lags!=0];
        lags <- lags[lags!=0];
    }

    # Define maxorder and make all the values look similar (for the polynomials)
    maxorder <- max(length(arMax),length(iMax),length(maMax));
    if(length(arMax)!=maxorder){
        arMax <- c(arMax,rep(0,maxorder-length(arMax)));
    }
    if(length(iMax)!=maxorder){
        iMax <- c(iMax,rep(0,maxorder-length(iMax)));
    }
    if(length(maMax)!=maxorder){
        maMax <- c(maMax,rep(0,maxorder-length(maMax)));
    }

    # If zeroes are defined as orders for some lags, drop them.
    if(any((arMax + iMax + maMax)==0)){
        orders2leave <- (arMax + iMax + maMax)!=0;
        if(all(!orders2leave)){
            orders2leave <- lags==min(lags);
        }
        arMax <- arMax[orders2leave];
        iMax <- iMax[orders2leave];
        maMax <- maMax[orders2leave];
        lags <- lags[orders2leave];
    }

    # Get rid of duplicates in lags
    if(length(unique(lags))!=length(lags)){
        lagsNew <- unique(lags);
        arMaxNew <- iMaxNew <- maMaxNew <- lagsNew;
        for(i in 1:length(lagsNew)){
            arMaxNew[i] <- max(arMax[which(lags==lagsNew[i])],na.rm=TRUE);
            iMaxNew[i] <- max(iMax[which(lags==lagsNew[i])],na.rm=TRUE);
            maMaxNew[i] <- max(maMax[which(lags==lagsNew[i])],na.rm=TRUE);
        }
        arMax <- arMaxNew;
        iMax <- iMaxNew;
        maMax <- maMaxNew;
        lags <- lagsNew;
    }

    # Order things, so we would deal with the lowest level of seasonality first
    arMax <- arMax[order(lags,decreasing=FALSE)];
    iMax <- iMax[order(lags,decreasing=FALSE)];
    maMax <- maMax[order(lags,decreasing=FALSE)];
    lags <- sort(lags,decreasing=FALSE);

# 1 stands for constant, the other one stands for variance
    nParamMax <- (1 + max(arMax %*% lags + iMax %*% lags,maMax %*% lags)
                  + sum(arMax) + sum(maMax) + constantCheck);

# Try to figure out if the number of parameters can be tuned in order to fit something smaller on small samples
# Don't try to fix anything if the number of seasonalities is greater than 2
    if(length(lags)<=2){
        if(obsInSample <= nParamMax){
            armaLength <- length(arMax);
            while(obsInSample <= nParamMax){
                if(any(c(arMax[armaLength],maMax[armaLength])>0)){
                    arMax[armaLength] <- max(0,arMax[armaLength] - 1);
                    nParamMax <- max(arMax %*% lags + iMax %*% lags,maMax %*% lags) + sum(arMax) + sum(maMax) + 1 + 1;
                    if(obsInSample <= nParamMax){
                        maMax[armaLength] <- max(0,maMax[armaLength] - 1);
                        nParamMax <- max(arMax %*% lags + iMax %*% lags,maMax %*% lags) + sum(arMax) + sum(maMax) + 1 + 1;
                    }
                }
                else{
                    if(armaLength==2){
                        arMax[1] <- arMax[1] - 1;
                        nParamMax <- max(arMax %*% lags + iMax %*% lags,maMax %*% lags) + sum(arMax) + sum(maMax) + 1 + 1;
                        if(obsInSample <= nParamMax){
                            maMax[1] <- maMax[1] - 1;
                            nParamMax <- max(arMax %*% lags + iMax %*% lags,maMax %*% lags) + sum(arMax) + sum(maMax) + 1 + 1;
                        }
                    }
                    else{
                        break;
                    }
                }
                if(all(c(arMax,maMax)==0)){
                    if(iMax[armaLength]>0){
                        iMax[armaLength] <- max(0,iMax[armaLength] - 1);
                        nParamMax <- max(arMax %*% lags + iMax %*% lags,maMax %*% lags) + sum(arMax) + sum(maMax) + 1 + 1;
                    }
                    else if(iMax[1]>0){
                        if(obsInSample <= nParamMax){
                            iMax[1] <- max(0,iMax[1] - 1);
                            nParamMax <- max(arMax %*% lags + iMax %*% lags,maMax %*% lags) + sum(arMax) + sum(maMax) + 1 + 1;
                        }
                    }
                    else{
                        break;
                    }
                }

            }
                nParamMax <- max(arMax %*% lags + iMax %*% lags,maMax %*% lags) + sum(arMax) + sum(maMax) + 1 + 1;
        }
    }

    if(obsInSample <= nParamMax){
        message(paste0("Not enough observations for the reasonable fit. Number of possible parameters is ",
                        nParamMax," while the number of observations is ",obsInSample,"!"));
        stop("Redefine maximum orders and try again.",call.=FALSE)
    }

# 1 stands for constant/no constant, another one stands for ARIMA(0,0,0)
    if(all(maMax==0)){
        nModels <- prod(iMax + 1) * (1 + sum(arMax)) + constantCheck;
    }
    else{
        nModels <- prod(iMax + 1) * (1 + sum(maMax*(1 + sum(arMax)))) + constantCheck;
    }
    testModel <- list(NA);
# Array with elements x maxorders x horizon x point/lower/upper
    if(combine){
        testForecasts <- list(NA);
        testFitted <- list(NA);
        testICs <- list(NA);
        testLevels <- list(NA);
        testStates <- list(NA);
        testTransition <- list(NA);
        testPersistence <- list(NA);
    }
    ICValue <- 1E+100;
    m <- 0;
    # constant <- TRUE;

    lagsTest <- maTest <- arTest <- rep(0,length(lags));
    arBest <- maBest <- iBest <- rep(0,length(lags));
    arBestLocal <- maBestLocal <- arBest;

#### Function corrects IC taking number of parameters on previous step ####
    icCorrector <- function(icValue, nParam, obsInSample, nParamNew){
        if(ic=="AIC"){
            llikelihood <- (2*nParam - icValue)/2;
            correction <- 2*nParamNew - 2*llikelihood;
        }
        else if(ic=="AICc"){
            llikelihood <- (2*nParam*obsInSample/(obsInSample-nParam-1) - icValue)/2;
            correction <- 2*nParamNew*obsInSample/(obsInSample-nParamNew-1) - 2*llikelihood;
        }
        else if(ic=="BIC"){
            llikelihood <- (nParam*log(obsInSample) - icValue)/2;
            correction <- nParamNew*log(obsInSample) - 2*llikelihood;
        }
        else if(ic=="BICc"){
            llikelihood <- ((nParam*log(obsInSample)*obsInSample)/(obsInSample-nParam-1) - icValue)/2;
            correction <- (nParamNew*log(obsInSample)*obsInSample)/(obsInSample-nParamNew-1) - 2*llikelihood;
        }

        return(correction);
    }

### If for some reason we have model with zeroes for orders, return it.
    if(all(c(arMax,iMax,maMax)==0)){
        cat("\b\b\b\bDone!\n");
        bestModel <- ssarima(y, orders=list(ar=arBest,i=(iBest),ma=(maBest)), lags=(lags),
                             constant=constantValue, regressors=regressors,
                             initial=initial, loss=loss,
                             h=h, holdout=holdout, bounds=bounds, silent=TRUE,
                             xreg=xreg);
        return(bestModel);
    }

    iOrders <- matrix(0,prod(iMax+1),ncol=length(iMax));

##### Loop for differences #####
    if(any(iMax!=0)){
        # Prepare table with differences
        iOrders[,1] <- rep(c(0:iMax[1]),times=prod(iMax[-1]+1));
        if(length(iMax)>1){
            for(seasLag in 2:length(iMax)){
                iOrders[,seasLag] <- rep(c(0:iMax[seasLag]),each=prod(iMax[1:(seasLag-1)]+1))
            }
        }
    }

    # Start the loop with differences
    for(d in 1:nrow(iOrders)){
        m <- m + 1;
        # if(!silent){
        #     cat(paste0(rep("\b",nchar(round(m/nModels,2)*100)+1),collapse=""));
        #     cat(paste0(round((m)/nModels,2)*100,"%"));
        # }
        nParamOriginal <- 1;
        if(!silent){
            cat("\nI: ");cat(iOrders[d,]);cat(", ");
        }
        testModel <- ssarima(y, orders=list(ar=0,i=iOrders[d,],ma=0), lags=lags,
                             constant=constantValue, regressors=regressors,
                             initial=initial, loss=loss,
                             h=h, holdout=holdout, bounds=bounds, silent=TRUE,
                             xreg=xreg);
        ICValue <- IC(testModel);
        if(combine){
            testForecasts[[m]] <- matrix(NA,h,3);
            testForecasts[[m]][,1] <- testModel$forecast;
            testForecasts[[m]][,2] <- testModel$lower;
            testForecasts[[m]][,3] <- testModel$upper;
            testFitted[[m]] <- testModel$fitted;
            testICs[[m]] <- ICValue;
            testLevels[[m]] <- 1;
            testStates[[m]] <- testModel$states;
            testTransition[[m]] <- testModel$transition;
            testPersistence[[m]] <- testModel$persistence;
        }
        if(!silent){
            cat(ICValue); cat("\n");
        }
        if(m==1){
            bestIC <- ICValue;
            dataMA <- dataI <- testModel$residuals;
            iBest <- iOrders[d,];
            bestICAR <- bestICI <- bestICMA <- bestIC;
        }
        else{
            if(ICValue < bestICI){
                bestICI <- ICValue;
                dataMA <- dataI <- testModel$residuals;
                if(ICValue < bestIC){
                    iBest <- iOrders[d,];
                    bestIC <- ICValue;
                    maBest <- arBest <- rep(0,length(arTest));
                }
            }
            else{
                if(fast){
                    m <- m + sum(maMax*(1 + sum(arMax)));
                    next;
                }
            }
        }

        ##### Loop for MA #####
        if(any(maMax!=0)){
            bestICMA <- bestICI;
            maBestLocal <- maTest <- rep(0,length(maTest));
            for(seasSelectMA in 1:length(lags)){
                if(maMax[seasSelectMA]!=0){
                    for(maSelect in 1:maMax[seasSelectMA]){
                        m <- m + 1;
                        # if(!silent){
                        #     cat(paste0(rep("\b",nchar(round(m/nModels,2)*100)+1),collapse=""));
                        #     cat(paste0(round((m)/nModels,2)*100,"%"));
                        # }
                        maTest[seasSelectMA] <- maMax[seasSelectMA] - maSelect + 1;
                        nParamMA <- sum(maTest);
                        nParamNew <- nParamOriginal + nParamMA;

                        if(!silent){
                            cat("MA: ");cat(maTest);cat(", ");
                        }
                        testModel <- ssarima(dataI, orders=list(ar=0,i=0,ma=maTest), lags=lags,
                                             constant=FALSE, initial=initial, loss=loss,
                                             h=h, holdout=FALSE, bounds=bounds, silent=TRUE);
                        ICValue <- icCorrector(IC(testModel), nParamMA, obsInSample, nParamNew);
                        if(combine){
                            testForecasts[[m]] <- matrix(NA,h,3);
                            testForecasts[[m]][,1] <- testModel$forecast;
                            testForecasts[[m]][,2] <- testModel$lower;
                            testForecasts[[m]][,3] <- testModel$upper;
                            testFitted[[m]] <- testModel$fitted;
                            testICs[[m]] <- ICValue;
                            testLevels[[m]] <- 2;
                            testStates[[m]] <- testModel$states;
                            testTransition[[m]] <- testModel$transition;
                            testPersistence[[m]] <- testModel$persistence;
                        }
                        if(!silent){
                            cat(ICValue); cat("\n");
                        }
                        if(ICValue < bestICMA){
                            bestICMA <- ICValue;
                            maBestLocal <- maTest;
                            if(ICValue < bestIC){
                                bestIC <- bestICMA;
                                iBest <- iOrders[d,];
                                maBest <- maTest;
                                arBest <- rep(0,length(arTest));
                            }
                            dataMA <- testModel$residuals;
                        }
                        else{
                            if(fast){
                                m <- m + maTest[seasSelectMA] * (1 + sum(arMax)) - 1;
                                maTest <- maBestLocal;
                                break;
                            }
                            else{
                                maTest <- maBestLocal;
                            }
                        }

                        ##### Loop for AR #####
                        if(any(arMax!=0)){
                            bestICAR <- bestICMA;
                            arBestLocal <- arTest <- rep(0,length(arTest));
                            for(seasSelectAR in 1:length(lags)){
                                lagsTest[seasSelectAR] <- lags[seasSelectAR];
                                if(arMax[seasSelectAR]!=0){
                                    for(arSelect in 1:arMax[seasSelectAR]){
                                        m <- m + 1;
                                        # if(!silent){
                                        #     cat(paste0(rep("\b",nchar(round(m/nModels,2)*100)+1),collapse=""));
                                        #     cat(paste0(round((m)/nModels,2)*100,"%"));
                                        # }
                                        arTest[seasSelectAR] <- arMax[seasSelectAR] - arSelect + 1;
                                        nParamAR <- sum(arTest);
                                        nParamNew <- nParamOriginal + nParamMA + nParamAR;

                                        if(!silent){
                                            cat("AR: ");cat(arTest);cat(", ");
                                        }
                                        testModel <- ssarima(dataMA, orders=list(ar=arTest,i=0,ma=0), lags=lags,
                                                             constant=FALSE, initial=initial, loss=loss,
                                                             h=h, holdout=FALSE, bounds=bounds, silent=TRUE);
                                        ICValue <- icCorrector(IC(testModel), nParamAR, obsInSample, nParamNew);
                                        if(combine){
                                            testForecasts[[m]] <- matrix(NA,h,3);
                                            testForecasts[[m]][,1] <- testModel$forecast;
                                            testForecasts[[m]][,2] <- testModel$lower;
                                            testForecasts[[m]][,3] <- testModel$upper;
                                            testFitted[[m]] <- testModel$fitted;
                                            testICs[[m]] <- ICValue;
                                            testLevels[[m]] <- 3;
                                            testStates[[m]] <- testModel$states;
                                            testTransition[[m]] <- testModel$transition;
                                            testPersistence[[m]] <- testModel$persistence;
                                        }
                                        if(!silent){
                                            cat(ICValue); cat("\n");
                                        }
                                        if(ICValue < bestICAR){
                                            bestICAR <- ICValue;
                                            arBestLocal <- arTest;
                                            if(ICValue < bestIC){
                                                bestIC <- ICValue;
                                                iBest <- iOrders[d,];
                                                arBest <- arTest;
                                                maBest <- maTest;
                                            }
                                        }
                                        else{
                                            if(fast){
                                                m <- m + arTest[seasSelectAR] - 1;
                                                arTest <- arBestLocal;
                                                break;
                                            }
                                            else{
                                                arTest <- arBestLocal;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        else{
            ##### Loop for AR #####
            if(any(arMax!=0)){
                bestICAR <- bestICMA;
                arBestLocal <- arTest <- rep(0,length(arTest));
                for(seasSelectAR in 1:length(lags)){
                    lagsTest[seasSelectAR] <- lags[seasSelectAR];
                    if(arMax[seasSelectAR]!=0){
                        for(arSelect in 1:arMax[seasSelectAR]){
                            m <- m + 1;
                            # if(!silent){
                            #     cat(paste0(rep("\b",nchar(round(m/nModels,2)*100)+1),collapse=""));
                            #     cat(paste0(round((m)/nModels,2)*100,"%"));
                            # }
                            arTest[seasSelectAR] <- arMax[seasSelectAR] - arSelect + 1;
                            nParamAR <- sum(arTest);
                            nParamNew <- nParamOriginal + nParamAR;

                            if(!silent){
                                cat("AR: ");cat(arTest);cat(", ");
                            }
                            testModel <- ssarima(dataMA, orders=list(ar=arTest,i=0,ma=0), lags=lags,
                                                 constant=FALSE, initial=initial, loss=loss,
                                                 h=h, holdout=FALSE, bounds=bounds, silent=TRUE);
                            ICValue <- icCorrector(IC(testModel), nParamAR, obsInSample, nParamNew);
                            if(combine){
                                testForecasts[[m]] <- matrix(NA,h,3);
                                testForecasts[[m]][,1] <- testModel$forecast;
                                testForecasts[[m]][,2] <- testModel$lower;
                                testForecasts[[m]][,3] <- testModel$upper;
                                testFitted[[m]] <- testModel$fitted;
                                testICs[[m]] <- ICValue;
                                testLevels[[m]] <- 3;
                                testStates[[m]] <- testModel$states;
                                testTransition[[m]] <- testModel$transition;
                                testPersistence[[m]] <- testModel$persistence;
                            }
                            if(!silent){
                                cat(ICValue); cat("\n");
                            }
                            if(ICValue < bestICAR){
                                bestICAR <- ICValue;
                                arBestLocal <- arTest;
                                if(ICValue < bestIC){
                                    bestIC <- ICValue;
                                    iBest <- iOrders[d,];
                                    arBest <- arTest;
                                    maBest <- maTest;
                                }
                            }
                            else{
                                if(fast){
                                    m <- m + arTest[seasSelectAR] - 1;
                                    arTest <- arBestLocal;
                                    break;
                                }
                                else{
                                    arTest <- arBestLocal;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

#### Test the constant ####
    if(constantCheck){
        m <- m + 1;
        # if(!silent){
        #     cat(paste0(rep("\b",nchar(round(m/nModels,2)*100)+1),collapse=""));
        #     cat(paste0(round((m)/nModels,2)*100,"%"));
        # }

        if(any(c(arBest,iBest,maBest)!=0)){
            testModel <- ssarima(y, orders=list(ar=(arBest),i=(iBest),ma=(maBest)), lags=(lags),
                                 constant=FALSE, regressors=regressors,
                                 initial=initial, loss=loss,
                                 h=h, holdout=holdout, bounds=bounds, silent=TRUE,
                                 xreg=xreg);
            ICValue <- IC(testModel);
            if(combine){
                testForecasts[[m]] <- matrix(NA,h,3);
                testForecasts[[m]][,1] <- testModel$forecast;
                testForecasts[[m]][,2] <- testModel$lower;
                testForecasts[[m]][,3] <- testModel$upper;
                testFitted[[m]] <- testModel$fitted;
                testICs[[m]] <- ICValue;
                testLevels[[m]] <- 1;
                testStates[[m]] <- testModel$states;
                testTransition[[m]] <- testModel$transition;
                testPersistence[[m]] <- testModel$persistence;
            }
            # cat("Constant: ");print(ICValue);
            if(ICValue < bestIC){
                bestModel <- testModel;
                constantValue <- FALSE;
            }
            else{
                constantValue <- TRUE;
            }
        }
    }

    if(combine){
        testICs <- unlist(testICs);
        testLevels <- unlist(testLevels);
        testForecasts <- array(unlist(testForecasts),c(h,3,length(testICs)));
        testFitted <- matrix(unlist(testFitted),ncol=length(testICs));
        icWeights <- exp(-0.5*(testICs-min(testICs)))/sum(exp(-0.5*(testICs-min(testICs))));

        testForecastsNew <- testForecasts;
        testFittedNew <- testFitted;
        for(i in 1:length(testLevels)){
            if(testLevels[i]==1){
                j <- i;
            }
            else if(testLevels[i]==2){
                k <- i;
                testForecastsNew[,,i] <- testForecasts[,,j] + testForecasts[,,i];
                testFittedNew[,i] <- testFitted[,j] + testFitted[,i];
            }
            else if(testLevels[i]==3){
                testForecastsNew[,,i] <- testForecasts[,,j] + testForecasts[,,k] + testForecasts[,,i];
                testFittedNew[,i] <- testFitted[,j] + testFitted[,k] + testFitted[,i];
            }
        }
        yForecast <- testForecasts[[1]]

        yForecast <- ts(testForecastsNew[,1,] %*% icWeights,start=yForecastStart,frequency=dataFreq);
        yLower <- ts(testForecastsNew[,2,] %*% icWeights,start=yForecastStart,frequency=dataFreq);
        yUpper <- ts(testForecastsNew[,3,] %*% icWeights,start=yForecastStart,frequency=dataFreq);
        yFitted <- ts(testFittedNew %*% icWeights,start=dataStart,frequency=dataFreq);
        modelname <- "ARIMA combined";

        errors <- ts(yInSample-c(yFitted),start=dataStart,frequency=dataFreq);
        yHoldout <- ts(y[(obsInSample+1):obsAll],start=yForecastStart,frequency=dataFreq);
        s2 <- mean(errors^2);
        errormeasures <- measures(yHoldout,yForecast,yInSample);
        ICs <- c(t(testICs) %*% icWeights);
        names(ICs) <- ic;

        bestModel <- list(data=y, model=modelname, timeElapsed=Sys.time()-startTime,
                          initialType=initialType,
                          fitted=yFitted, forecast=yForecast,
                          lower=yLower, upper=yUpper, residuals=errors,
                          s2=s2, holdout=yHoldout,
                          regressors=regressors, formula=formula,
                          ICs=ICs, ICw=icWeights, lossValue=NULL, loss=loss, accuracy=errormeasures);

        bestModel <- structure(bestModel,class="smooth");
    }
    else{
        #### Reestimate the best model in order to get rid of bias ####
        bestModel <- ssarima(y, orders=list(ar=(arBest),i=(iBest),ma=(maBest)), lags=(lags),
                             constant=constantValue, regressors=regressors,
                             initial=initial, loss=loss,
                             h=h, holdout=holdout, bounds=bounds, silent=TRUE,
                             xreg=xreg);

        bestModel$timeElapsed <- Sys.time()-startTime;
    }

    if(!silent){
        cat("... Done! \n");
    }

##### Make a plot #####
    if(!silent){
        plot(bestModel, 7)
    }

    return(bestModel);
}
