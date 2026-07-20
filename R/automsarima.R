#' @aliases auto.msarima
#'
#' @examples
#'
#' x <- rnorm(118,100,3)
#' # The best ARIMA for the data
#' ourModel <- auto.msarima(x,orders=list(ar=c(2,1),i=c(1,1),ma=c(2,1)),lags=c(1,12),
#'                      h=18,holdout=TRUE)
#'
#' # The other one using optimised states
#' \donttest{auto.msarima(x,orders=list(ar=c(3,2),i=c(2,1),ma=c(3,2)),lags=c(1,12),
#'                        h=18,holdout=TRUE)}
#'
#' # And now combined ARIMA
#' \donttest{auto.msarima(x,orders=list(ar=c(3,2),i=c(2,1),ma=c(3,2)),lags=c(1,12),
#'                        combine=TRUE,h=18,holdout=TRUE)}
#'
#' plot(forecast(ourModel, h=18, interval="simulated"))
#'
#' @rdname msarima
#' @export
auto.msarima <- function(y, orders=list(ar=c(3,3),i=c(2,1),ma=c(3,3)), lags=c(1,frequency(y)),
                         initial=c("backcasting","optimal","two-stage","complete","gradient"),
                         ic=c("AICc","AIC","BIC","BICc"),
                         loss=c("likelihood","MSE","MAE","HAM","MSEh","TMSE","GTMSE","MSCE","GPL"),
                         h=0, holdout=FALSE,
                         bounds=c("usual","admissible","none"),
                         silent=TRUE,
                         xreg=NULL, regressors=c("use","select","adapt"), initialX=NULL, ...){
# This is a wrapper function for adam with order selection

    if(!is.null(xreg) && is.numeric(y)){
        data <- cbind(y=as.data.frame(y),as.data.frame(xreg));
        data <- as.matrix(data)
        data <- ts(data, start=start(y), frequency=frequency(y));
        colnames(data)[1] <- "y";
        # Give name to the explanatory variables if they do not have them
        if(is.null(names(xreg))){
            if(!is.null(ncol(xreg))){
                colnames(data)[-1] <- paste0("x",c(1:ncol(xreg)));
            }
            else{
                colnames(data)[-1] <- "x";
            }
        }
    }
    else{
        data <- y;
    }

    # Fix orders
    orders$select <- TRUE;

    return(adam(data, model="NNN", orders=orders, lags=lags, distribution="dnorm",
                initial=initial, loss=loss, h=h, holdout=holdout, bounds=bounds,
                silent=silent, regressors=regressors, adamETS=FALSE, ...))

# Function estimates several msarima models and selects the best one using the selected information criterion.
#
#    Copyright (C) 2015 - 2016  Ivan Svetunkov
#
# Start measuring the time of calculations
#     startTime <- Sys.time();
#
#     ### Depricate the old parameters
#     ellipsis <- list(...)
#     ellipsis <- depricator(ellipsis, "xregDo", "regressors");
#
#     updateX <- FALSE;
#     persistenceX <- transitionX <- NULL;
#     occurrence <- "none";
#     oesmodel <- "MNN";
#
# # Add all the variables in ellipsis to current environment
#     list2env(ellipsis,environment());
#
#     if(!is.null(orders)){
#         arMax <- orders$ar;
#         iMax <- orders$i;
#         maMax <- orders$ma;
#     }
#
# # If orders are provided in ellipsis via arMax, write them down.
#     if(exists("ar.orders",inherits=FALSE)){
#         if(is.null(ar.orders)){
#             arMax <- 0;
#         }
#         else{
#             arMax <- ar.orders;
#         }
#     }
#     else{
#         if(is.null(orders)){
#             arMax <- 0;
#         }
#     }
#     if(exists("i.orders",inherits=FALSE)){
#         if(is.null(i.orders)){
#             iMax <- 0;
#         }
#         else{
#             iMax <- i.orders;
#         }
#     }
#     else{
#         if(is.null(orders)){
#             iMax <- 0;
#         }
#     }
#     if(exists("ma.orders",inherits=FALSE)){
#         if(is.null(ma.orders)){
#             maMax <- 0;
#         }
#         else{
#             maMax <- ma.orders
#         }
#     }
#     else{
#         if(is.null(orders)){
#             maMax <- 0;
#         }
#     }
#
#     interval <- "none";
#     cumulative <- FALSE;
#
# ##### Set environment for ssInput and make all the checks #####
#     environment(ssAutoInput) <- environment();
#     ssAutoInput("auto.msarima",ParentEnvironment=environment());
#
#     if(is.null(constant)){
#         constantCheck <- TRUE;
#         constantValue <- TRUE;
#     }
#     else{
#         if(is.logical(constant)){
#             constantCheck <- FALSE;
#             constantValue <- constant;
#         }
#         else{
#             constant <- NULL;
#             constantCheck <- TRUE;
#             constantValue <- TRUE;
#             warning("Strange value of constant parameter. We changed it to the default value.");
#         }
#     }
#
#     if(any(is.complex(c(arMax,iMax,maMax,lags)))){
#         stop("Come on! Be serious! This is ARIMA, not CES!",call.=FALSE);
#     }
#
#     if(any(c(arMax,iMax,maMax)<0)){
#         stop("Funny guy! How am I gonna construct a model with negative order?",call.=FALSE);
#     }
#
#     if(any(c(lags)<0)){
#         stop("Right! Why don't you try complex lags then, mister smart guy?",call.=FALSE);
#     }
#
#     # If there are zero lags, drop them
#     if(any(lags==0)){
#         arMax <- arMax[lags!=0];
#         iMax <- iMax[lags!=0];
#         maMax <- maMax[lags!=0];
#         lags <- lags[lags!=0];
#     }
#
#     # Define maxorder and make all the values look similar (for the polynomials)
#     maxorder <- max(length(arMax),length(iMax),length(maMax));
#     if(length(arMax)!=maxorder){
#         arMax <- c(arMax,rep(0,maxorder-length(arMax)));
#     }
#     if(length(iMax)!=maxorder){
#         iMax <- c(iMax,rep(0,maxorder-length(iMax)));
#     }
#     if(length(maMax)!=maxorder){
#         maMax <- c(maMax,rep(0,maxorder-length(maMax)));
#     }
#
#     # If zeroes are defined as orders for some lags, drop them.
#     if(any((arMax + iMax + maMax)==0)){
#         orders2leave <- (arMax + iMax + maMax)!=0;
#         if(all(!orders2leave)){
#             orders2leave <- lags==min(lags);
#         }
#         arMax <- arMax[orders2leave];
#         iMax <- iMax[orders2leave];
#         maMax <- maMax[orders2leave];
#         lags <- lags[orders2leave];
#     }
#
#     # Get rid of duplicates in lags
#     if(length(unique(lags))!=length(lags)){
#         if(dataFreq!=1){
#             warning(paste0("'lags' variable contains duplicates: (",paste0(lags,collapse=","),"). Getting rid of some of them."),call.=FALSE);
#         }
#         lagsNew <- unique(lags);
#         arMaxNew <- iMaxNew <- maMaxNew <- lagsNew;
#         for(i in 1:length(lagsNew)){
#             arMaxNew[i] <- max(arMax[which(lags==lagsNew[i])],na.rm=TRUE);
#             iMaxNew[i] <- max(iMax[which(lags==lagsNew[i])],na.rm=TRUE);
#             maMaxNew[i] <- max(maMax[which(lags==lagsNew[i])],na.rm=TRUE);
#         }
#         arMax <- arMaxNew;
#         iMax <- iMaxNew;
#         maMax <- maMaxNew;
#         lags <- lagsNew;
#     }
#
#     # Order things, so we would deal with the lowest level of seasonality first
#     arMax <- arMax[order(lags,decreasing=FALSE)];
#     iMax <- iMax[order(lags,decreasing=FALSE)];
#     maMax <- maMax[order(lags,decreasing=FALSE)];
#     lags <- sort(lags,decreasing=FALSE);
#
# # 1 stands for constant, the other one stands for variance
#     nParamMax <- (1 + max(arMax %*% lags + iMax %*% lags,maMax %*% lags)
#                   + sum(arMax) + sum(maMax) + constantCheck);
#
# # Try to figure out if the number of parameters can be tuned in order to fit something smaller on small samples
# # Don't try to fix anything if the number of seasonalities is greater than 2
#     if(length(lags)<=2){
#         if(obsNonzero <= nParamMax){
#             armaLength <- length(arMax);
#             while(obsNonzero <= nParamMax){
#                 if(any(c(arMax[armaLength],maMax[armaLength])>0)){
#                     arMax[armaLength] <- max(0,arMax[armaLength] - 1);
#                     nParamMax <- max(arMax %*% lags + iMax %*% lags,maMax %*% lags) + sum(arMax) + sum(maMax) + 1 + 1;
#                     if(obsNonzero <= nParamMax){
#                         maMax[armaLength] <- max(0,maMax[armaLength] - 1);
#                         nParamMax <- max(arMax %*% lags + iMax %*% lags,maMax %*% lags) + sum(arMax) + sum(maMax) + 1 + 1;
#                     }
#                 }
#                 else{
#                     if(armaLength==2){
#                         arMax[1] <- arMax[1] - 1;
#                         nParamMax <- max(arMax %*% lags + iMax %*% lags,maMax %*% lags) + sum(arMax) + sum(maMax) + 1 + 1;
#                         if(obsNonzero <= nParamMax){
#                             maMax[1] <- maMax[1] - 1;
#                             nParamMax <- max(arMax %*% lags + iMax %*% lags,maMax %*% lags) + sum(arMax) + sum(maMax) + 1 + 1;
#                         }
#                     }
#                     else{
#                         break;
#                     }
#                 }
#                 if(all(c(arMax,maMax)==0)){
#                     if(iMax[armaLength]>0){
#                         iMax[armaLength] <- max(0,iMax[armaLength] - 1);
#                         nParamMax <- max(arMax %*% lags + iMax %*% lags,maMax %*% lags) + sum(arMax) + sum(maMax) + 1 + 1;
#                     }
#                     else if(iMax[1]>0){
#                         if(obsNonzero <= nParamMax){
#                             iMax[1] <- max(0,iMax[1] - 1);
#                             nParamMax <- max(arMax %*% lags + iMax %*% lags,maMax %*% lags) + sum(arMax) + sum(maMax) + 1 + 1;
#                         }
#                     }
#                     else{
#                         break;
#                     }
#                 }
#
#             }
#                 nParamMax <- max(arMax %*% lags + iMax %*% lags,maMax %*% lags) + sum(arMax) + sum(maMax) + 1 + 1;
#         }
#     }
#
#     if(obsNonzero <= nParamMax){
#         message(paste0("Not enough observations for the reasonable fit. Number of possible parameters is ",
#                         nParamMax," while the number of observations is ",obsNonzero,"!"));
#         stop("Redefine maximum orders and try again.",call.=FALSE)
#     }
#
# # 1 stands for constant/no constant, another one stands for ARIMA(0,0,0)
#     if(all(maMax==0)){
#         nModels <- prod(iMax + 1) * (1 + sum(arMax)) + constantCheck;
#     }
#     else{
#         nModels <- prod(iMax + 1) * (1 + sum(maMax*(1 + sum(arMax)))) + constantCheck;
#     }
#     testModel <- list(NA);
# # Array with elements x maxorders x horizon x point/lower/upper
#     if(combine){
#         testForecasts <- list(NA);
#         testFitted <- list(NA);
#         testICs <- list(NA);
#         testLevels <- list(NA);
#         testStates <- list(NA);
#         testTransition <- list(NA);
#         testPersistence <- list(NA);
#     }
#     ICValue <- 1E+100;
#     m <- 0;
#     # constant <- TRUE;
#
#     lagsTest <- maTest <- arTest <- rep(0,length(lags));
#     arBest <- maBest <- iBest <- rep(0,length(lags));
#     arBestLocal <- maBestLocal <- arBest;
#
# #### Function corrects IC taking number of parameters on previous step ####
#     icCorrector <- function(icValue, nParam, obsNonzero, nParamNew){
#         if(ic=="AIC"){
#             llikelihood <- (2*nParam - icValue)/2;
#             correction <- 2*nParamNew - 2*llikelihood;
#         }
#         else if(ic=="AICc"){
#             llikelihood <- (2*nParam*obsNonzero/(obsNonzero-nParam-1) - icValue)/2;
#             correction <- 2*nParamNew*obsNonzero/(obsNonzero-nParamNew-1) - 2*llikelihood;
#         }
#         else if(ic=="BIC"){
#             llikelihood <- (nParam*log(obsNonzero) - icValue)/2;
#             correction <- nParamNew*log(obsNonzero) - 2*llikelihood;
#         }
#         else if(ic=="BICc"){
#             llikelihood <- ((nParam*log(obsNonzero)*obsNonzero)/(obsNonzero-nParam-1) - icValue)/2;
#             correction <- (nParamNew*log(obsNonzero)*obsNonzero)/(obsNonzero-nParamNew-1) - 2*llikelihood;
#         }
#
#         return(correction);
#     }
#
#     if(!silentText){
#         cat("Estimation progress:     ");
#     }
#
# ### If for some reason we have model with zeroes for orders, return it.
#     if(all(c(arMax,iMax,maMax)==0)){
#         cat("\b\b\b\bDone!\n");
#         bestModel <- msarima(y, orders=list(ar=arBest,i=(iBest),ma=(maBest)), lags=(lags),
#                              constant=constantValue, initial=initialType, loss=loss,
#                              h=h, holdout=holdout, #cumulative=cumulative,
#                              # interval=intervalType, level=level,
#                              bounds=bounds, silent=TRUE,
#                              xreg=xreg, regressors=regressors, initialX=initialX, FI=FI);
#         return(bestModel);
#     }
#
#     iOrders <- matrix(0,prod(iMax+1),ncol=length(iMax));
#
# ##### Loop for differences #####
#     if(any(iMax!=0)){
#         # Prepare table with differences
#         iOrders[,1] <- rep(c(0:iMax[1]),times=prod(iMax[-1]+1));
#         if(length(iMax)>1){
#             for(seasLag in 2:length(iMax)){
#                 iOrders[,seasLag] <- rep(c(0:iMax[seasLag]),each=prod(iMax[1:(seasLag-1)]+1))
#             }
#         }
#     }
#
#     # Start the loop with differences
#     for(d in 1:nrow(iOrders)){
#         m <- m + 1;
#         if(!silentText){
#             cat(paste0(rep("\b",nchar(round(m/nModels,2)*100)+1),collapse=""));
#             cat(paste0(round((m)/nModels,2)*100,"%"));
#         }
#         # Originally, we only have a constant
#         nParamOriginal <- 1;
#         if(silent[1]=="d"){
#             cat("I: ");cat(iOrders[d,]);cat(", ");
#         }
#         testModel <- msarima(y, orders=list(ar=0,i=iOrders[d,],ma=0), lags=lags,
#                              constant=constantValue, initial=initialType, loss=loss,
#                              h=h, holdout=holdout, #cumulative=cumulative,
#                              # interval=intervalType, level=level,
#                              bounds=bounds, silent=TRUE,
#                              xreg=xreg, regressors=regressors, initialX=initialX, FI=FI);
#         ICValue <- testModel$ICs;
#         if(combine){
#             testForecasts[[m]] <- matrix(NA,h,3);
#             testForecasts[[m]][,1] <- testModel$forecast;
#             # testForecasts[[m]][,2] <- testModel$lower;
#             # testForecasts[[m]][,3] <- testModel$upper;
#             testFitted[[m]] <- testModel$fitted;
#             testICs[[m]] <- ICValue;
#             testLevels[[m]] <- 1;
#             testStates[[m]] <- testModel$states;
#             testTransition[[m]] <- testModel$transition;
#             testPersistence[[m]] <- testModel$persistence;
#         }
#         if(silent[1]=="d"){
#             cat(ICValue); cat("\n");
#         }
#         if(m==1){
#             bestIC <- ICValue;
#             dataMA <- dataI <- testModel$residuals;
#             iBest <- iOrders[d,];
#             bestICAR <- bestICI <- bestICMA <- bestIC;
#         }
#         else{
#             if(ICValue < bestICI){
#                 bestICI <- ICValue;
#                 dataMA <- dataI <- testModel$residuals;
#                 if(ICValue < bestIC){
#                     iBest <- iOrders[d,];
#                     bestIC <- ICValue;
#                     maBest <- arBest <- rep(0,length(arTest));
#                 }
#             }
#             else{
#                 if(fast){
#                     m <- m + sum(maMax*(1 + sum(arMax)));
#                     next;
#                 }
#             }
#         }
#
#         ##### Loop for MA #####
#         if(any(maMax!=0)){
#             bestICMA <- bestICI;
#             maBestLocal <- maTest <- rep(0,length(maTest));
#             for(seasSelectMA in 1:length(lags)){
#                 if(maMax[seasSelectMA]!=0){
#                     for(maSelect in 1:maMax[seasSelectMA]){
#                         m <- m + 1;
#                         if(!silentText){
#                             cat(paste0(rep("\b",nchar(round(m/nModels,2)*100)+1),collapse=""));
#                             cat(paste0(round((m)/nModels,2)*100,"%"));
#                         }
#                         maTest[seasSelectMA] <- maMax[seasSelectMA] - maSelect + 1;
#
#                         if(silent[1]=="d"){
#                             cat("MA: ");cat(maTest);cat(", ");
#                         }
#                         testModel <- msarima(dataI, orders=list(ar=0,i=0,ma=maTest), lags=lags,
#                                              constant=FALSE, initial=initialType, loss=loss,
#                                              h=h, holdout=FALSE,
#                                              # interval=intervalType, level=level,
#                                              bounds=bounds, silent=TRUE,
#                                              xreg=NULL, regressors="use", initialX=initialX, FI=FI);
#                         # Exclude the variance from the number of parameters
#                         nParamMA <- nparam(testModel)-1;
#                         nParamNew <- nParamOriginal + nParamMA;
#                         ICValue <- icCorrector(testModel$ICs, nParamMA, obsNonzero, nParamNew);
#                         if(combine){
#                             testForecasts[[m]] <- matrix(NA,h,3);
#                             testForecasts[[m]][,1] <- testModel$forecast;
#                             # testForecasts[[m]][,2] <- testModel$lower;
#                             # testForecasts[[m]][,3] <- testModel$upper;
#                             testFitted[[m]] <- testModel$fitted;
#                             testICs[[m]] <- ICValue;
#                             testLevels[[m]] <- 2;
#                             testStates[[m]] <- testModel$states;
#                             testTransition[[m]] <- testModel$transition;
#                             testPersistence[[m]] <- testModel$persistence;
#                         }
#                         if(silent[1]=="d"){
#                             cat(ICValue); cat("\n");
#                         }
#                         if(ICValue < bestICMA){
#                             bestICMA <- ICValue;
#                             maBestLocal <- maTest;
#                             if(ICValue < bestIC){
#                                 bestIC <- bestICMA;
#                                 iBest <- iOrders[d,];
#                                 maBest <- maTest;
#                                 arBest <- rep(0,length(arTest));
#                             }
#                             dataMA <- testModel$residuals;
#                         }
#                         else{
#                             if(fast){
#                                 m <- m + maTest[seasSelectMA] * (1 + sum(arMax)) - 1;
#                                 maTest <- maBestLocal;
#                                 break;
#                             }
#                             else{
#                                 maTest <- maBestLocal;
#                             }
#                         }
#
#                         ##### Loop for AR #####
#                         if(any(arMax!=0)){
#                             bestICAR <- bestICMA;
#                             arBestLocal <- arTest <- rep(0,length(arTest));
#                             for(seasSelectAR in 1:length(lags)){
#                                 lagsTest[seasSelectAR] <- lags[seasSelectAR];
#                                 if(arMax[seasSelectAR]!=0){
#                                     for(arSelect in 1:arMax[seasSelectAR]){
#                                         m <- m + 1;
#                                         if(!silentText){
#                                             cat(paste0(rep("\b",nchar(round(m/nModels,2)*100)+1),collapse=""));
#                                             cat(paste0(round((m)/nModels,2)*100,"%"));
#                                         }
#                                         arTest[seasSelectAR] <- arMax[seasSelectAR] - arSelect + 1;
#
#                                         if(silent[1]=="d"){
#                                             cat("AR: ");cat(arTest);cat(", ");
#                                         }
#                                         testModel <- msarima(dataMA, orders=list(ar=arTest,i=0,ma=0), lags=lags,
#                                                              constant=FALSE, initial=initialType, loss=loss,
#                                                              h=h, holdout=FALSE,
#                                                              # interval=intervalType, level=level,
#                                                              bounds=bounds, silent=TRUE,
#                                                              xreg=NULL, regressors="use", initialX=initialX, FI=FI);
#                                         # Exclude the variance from the number of parameters
#                                         nParamAR <- nparam(testModel)-1;
#                                         nParamNew <- nParamOriginal + nParamMA + nParamAR;
#                                         ICValue <- icCorrector(testModel$ICs, nParamAR, obsNonzero, nParamNew);
#                                         if(combine){
#                                             testForecasts[[m]] <- matrix(NA,h,3);
#                                             testForecasts[[m]][,1] <- testModel$forecast;
#                                             # testForecasts[[m]][,2] <- testModel$lower;
#                                             # testForecasts[[m]][,3] <- testModel$upper;
#                                             testFitted[[m]] <- testModel$fitted;
#                                             testICs[[m]] <- ICValue;
#                                             testLevels[[m]] <- 3;
#                                             testStates[[m]] <- testModel$states;
#                                             testTransition[[m]] <- testModel$transition;
#                                             testPersistence[[m]] <- testModel$persistence;
#                                         }
#                                         if(silent[1]=="d"){
#                                             cat(ICValue); cat("\n");
#                                         }
#                                         if(ICValue < bestICAR){
#                                             bestICAR <- ICValue;
#                                             arBestLocal <- arTest;
#                                             if(ICValue < bestIC){
#                                                 bestIC <- ICValue;
#                                                 iBest <- iOrders[d,];
#                                                 arBest <- arTest;
#                                                 maBest <- maTest;
#                                             }
#                                         }
#                                         else{
#                                             if(fast){
#                                                 m <- m + arTest[seasSelectAR] - 1;
#                                                 arTest <- arBestLocal;
#                                                 break;
#                                             }
#                                             else{
#                                                 arTest <- arBestLocal;
#                                             }
#                                         }
#                                     }
#                                 }
#                             }
#                         }
#                     }
#                 }
#             }
#         }
#         else{
#             ##### Loop for AR #####
#             if(any(arMax!=0)){
#                 bestICAR <- bestICMA;
#                 arBestLocal <- arTest <- rep(0,length(arTest));
#                 for(seasSelectAR in 1:length(lags)){
#                     lagsTest[seasSelectAR] <- lags[seasSelectAR];
#                     if(arMax[seasSelectAR]!=0){
#                         for(arSelect in 1:arMax[seasSelectAR]){
#                             m <- m + 1;
#                             if(!silentText){
#                                 cat(paste0(rep("\b",nchar(round(m/nModels,2)*100)+1),collapse=""));
#                                 cat(paste0(round((m)/nModels,2)*100,"%"));
#                             }
#                             arTest[seasSelectAR] <- arMax[seasSelectAR] - arSelect + 1;
#                             nParamAR <- sum(arTest);
#                             nParamNew <- nParamOriginal + nParamAR;
#
#                             if(silent[1]=="d"){
#                                 cat("AR: ");cat(arTest);cat(", ");
#                             }
#                             testModel <- msarima(dataMA, orders=list(ar=arTest,i=0,ma=0), lags=lags,
#                                                  constant=FALSE, initial=initialType, loss=loss,
#                                                  h=h, holdout=FALSE,
#                                                  # interval=intervalType, level=level,
#                                                  bounds=bounds, silent=TRUE,
#                                                  xreg=NULL, regressors="use", initialX=initialX, FI=FI);
#                             ICValue <- icCorrector(testModel$ICs, nParamAR, obsNonzero, nParamNew);
#                             if(combine){
#                                 testForecasts[[m]] <- matrix(NA,h,3);
#                                 testForecasts[[m]][,1] <- testModel$forecast;
#                                 # testForecasts[[m]][,2] <- testModel$lower;
#                                 # testForecasts[[m]][,3] <- testModel$upper;
#                                 testFitted[[m]] <- testModel$fitted;
#                                 testICs[[m]] <- ICValue;
#                                 testLevels[[m]] <- 3;
#                                 testStates[[m]] <- testModel$states;
#                                 testTransition[[m]] <- testModel$transition;
#                                 testPersistence[[m]] <- testModel$persistence;
#                             }
#                             if(silent[1]=="d"){
#                                 cat(ICValue); cat("\n");
#                             }
#                             if(ICValue < bestICAR){
#                                 bestICAR <- ICValue;
#                                 arBestLocal <- arTest;
#                                 if(ICValue < bestIC){
#                                     bestIC <- ICValue;
#                                     iBest <- iOrders[d,];
#                                     arBest <- arTest;
#                                     maBest <- maTest;
#                                 }
#                             }
#                             else{
#                                 if(fast){
#                                     m <- m + arTest[seasSelectAR] - 1;
#                                     arTest <- arBestLocal;
#                                     break;
#                                 }
#                                 else{
#                                     arTest <- arBestLocal;
#                                 }
#                             }
#                         }
#                     }
#                 }
#             }
#         }
#     }
#
# #### Test the constant ####
#     if(constantCheck){
#         m <- m + 1;
#         if(!silentText){
#             cat(paste0(rep("\b",nchar(round(m/nModels,2)*100)+1),collapse=""));
#             cat(paste0(round((m)/nModels,2)*100,"%"));
#         }
#
#         if(any(c(arBest,iBest,maBest)!=0)){
#             testModel <- msarima(y, orders=list(ar=(arBest),i=(iBest),ma=(maBest)), lags=(lags),
#                                  constant=FALSE, initial=initialType, loss=loss,
#                                  h=h, holdout=holdout, #cumulative=cumulative,
#                                  # interval=intervalType, level=level,
#                                  bounds=bounds, silent=TRUE,
#                                  xreg=xreg, regressors=regressors, initialX=initialX, FI=FI);
#             ICValue <- testModel$ICs;
#             if(combine){
#                 testForecasts[[m]] <- matrix(NA,h,3);
#                 testForecasts[[m]][,1] <- testModel$forecast;
#                 # testForecasts[[m]][,2] <- testModel$lower;
#                 # testForecasts[[m]][,3] <- testModel$upper;
#                 testFitted[[m]] <- testModel$fitted;
#                 testICs[[m]] <- ICValue;
#                 testLevels[[m]] <- 1;
#                 testStates[[m]] <- testModel$states;
#                 testTransition[[m]] <- testModel$transition;
#                 testPersistence[[m]] <- testModel$persistence;
#             }
#             if(silent[1]=="d"){
#                 cat("No constant: "); cat(ICValue); cat("\n");
#             }
#             if(ICValue < bestIC){
#                 bestModel <- testModel;
#                 constantValue <- FALSE;
#                 bestIC <- ICValue;
#             }
#             else{
#                 constantValue <- TRUE;
#             }
#         }
#     }
#     if(silent[1]=="d"){
#         cat("Best IC: "); cat(bestIC); cat("\n");
#     }
#
#     if(combine){
#         testICs <- unlist(testICs);
#         testLevels <- unlist(testLevels);
#         testForecasts <- array(unlist(testForecasts),c(h,3,length(testICs)));
#         testFitted <- matrix(unlist(testFitted),ncol=length(testICs));
#         icWeights <- exp(-0.5*(testICs-min(testICs)))/sum(exp(-0.5*(testICs-min(testICs))));
#
#         testForecastsNew <- testForecasts;
#         testFittedNew <- testFitted;
#         for(i in 1:length(testLevels)){
#             if(testLevels[i]==1){
#                 j <- i;
#             }
#             else if(testLevels[i]==2){
#                 k <- i;
#                 testForecastsNew[,,i] <- testForecasts[,,j] + testForecasts[,,i];
#                 testFittedNew[,i] <- testFitted[,j] + testFitted[,i];
#             }
#             else if(testLevels[i]==3){
#                 testForecastsNew[,,i] <- testForecasts[,,j] + testForecasts[,,k] + testForecasts[,,i];
#                 testFittedNew[,i] <- testFitted[,j] + testFitted[,k] + testFitted[,i];
#             }
#         }
#         yForecast <- ts(testForecastsNew[,1,] %*% icWeights,start=yForecastStart,frequency=dataFreq);
#         yLower <- ts(testForecastsNew[,2,] %*% icWeights,start=yForecastStart,frequency=dataFreq);
#         yUpper <- ts(testForecastsNew[,3,] %*% icWeights,start=yForecastStart,frequency=dataFreq);
#         yFitted <- ts(testFittedNew %*% icWeights,start=dataStart,frequency=dataFreq);
#         modelname <- "ARIMA combined";
#
#         errors <- ts(yInSample-c(yFitted),start=dataStart,frequency=dataFreq);
#         yHoldout <- ts(y[(obsNonzero+1):obsAll],start=yForecastStart,frequency=dataFreq);
#         s2 <- mean(errors^2);
#         errormeasures <- measures(yHoldout,yForecast,yInSample);
#         ICs <- c(t(testICs) %*% icWeights);
#         names(ICs) <- ic;
#
#         bestModel <- list(model=modelname,timeElapsed=Sys.time()-startTime,
#                           initialType=initialType,
#                           fitted=yFitted,forecast=yForecast, #cumulative=cumulative,
#                           lower=yLower,upper=yUpper,residuals=errors,s2=s2,
#                           #interval=intervalType,level=level,
#                           y=y,holdout=yHoldout,
#                           xreg=xreg, regressors=regressors, initialX=initialX,
#                           ICs=ICs,ICw=icWeights,lossValue=NULL,loss=loss,accuracy=errormeasures);
#
#         bestModel <- structure(bestModel,class=c("smooth","msarima"));
#     }
#     else{
#         #### Reestimate the best model in order to get rid of bias ####
#         bestModel <- msarima(y, orders=list(ar=(arBest),i=(iBest),ma=(maBest)), lags=(lags),
#                              constant=constantValue, initial=initialType, loss=loss,
#                              h=h, holdout=holdout, #cumulative=cumulative,
#                              # interval=intervalType, level=level,
#                              bounds=bounds, silent=TRUE,
#                              xreg=xreg, regressors=regressors, initialX=initialX, FI=FI);
#
#         yFitted <- bestModel$fitted;
#         yForecast <- bestModel$forecast;
#         yUpper <- bestModel$upper;
#         yLower <- bestModel$lower;
#         modelname <- bestModel$model;
#
#         bestModel$timeElapsed <- Sys.time()-startTime;
#     }
#
#     if(!silentText){
#         cat("... Done! \n");
#     }
#
# ##### Make a plot #####
#     if(!silentGraph){
#         yForecastNew <- yForecast;
#         yUpperNew <- yUpper;
#         yLowerNew <- yLower;
#         # if(cumulative){
#         #     yForecastNew <- ts(rep(yForecast/h,h),start=yForecastStart,frequency=dataFreq)
#         #     if(interval){
#         #         yUpperNew <- ts(rep(yUpper/h,h),start=yForecastStart,frequency=dataFreq)
#         #         yLowerNew <- ts(rep(yLower/h,h),start=yForecastStart,frequency=dataFreq)
#         #     }
#         # }
#
#         # if(interval){
#         #     graphmaker(actuals=y,forecast=yForecastNew,fitted=yFitted, lower=yLowerNew,upper=yUpperNew,
#         #                level=level,legend=!silentLegend,main=modelname,cumulative=cumulative);
#         # }
#         # else{
#             graphmaker(actuals=y,forecast=yForecastNew,fitted=yFitted,
#                        legend=!silentLegend,main=modelname,cumulative=FALSE);
#         # }
#     }
#
#     return(bestModel);
}
