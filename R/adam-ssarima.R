
#' State Space ARIMA
#'
#' Function constructs State Space ARIMA, estimating AR, MA terms and initial
#' states.
#'
#' The model, implemented in this function, is discussed in Svetunkov & Boylan
#' (2019).
#'
#' The basic ARIMA(p,d,q) used in the function has the following form:
#'
#' \eqn{(1 - B)^d (1 - a_1 B - a_2 B^2 - ... - a_p B^p) y_[t] = (1 + b_1 B +
#' b_2 B^2 + ... + b_q B^q) \epsilon_[t] + c}
#'
#' where \eqn{y_[t]} is the actual values, \eqn{\epsilon_[t]} is the error term,
#' \eqn{a_i, b_j} are the parameters for AR and MA respectively and \eqn{c} is
#' the constant. In case of non-zero differences \eqn{c} acts as drift.
#'
#' This model is then transformed into ARIMA in the Single Source of Error
#' State space form (proposed in Snyder, 1985):
#'
#' \eqn{y_{t} = w' v_{t-l} + \epsilon_{t}}
#'
#' \eqn{v_{t} = F v_{t-l} + g_t \epsilon_{t}}
#'
#' where \eqn{v_{t}} is the state vector (defined based on
#' \code{orders}) and \eqn{l} is the vector of \code{lags}, \eqn{w_t} is the
#' \code{measurement} vector (with explanatory variables if provided), \eqn{F}
#' is the \code{transition} matrix, \eqn{g_t} is the \code{persistence} vector
#' (which includes explanatory variables if they were used).
#'
#' Due to the flexibility of the model, multiple seasonalities can be used. For
#' example, something crazy like this can be constructed:
#' SARIMA(1,1,1)(0,1,1)[24](2,0,1)[24*7](0,0,1)[24*30], but the estimation may
#' take a lot of time... If you plan estimating a model with more than one
#' seasonality, it is recommended to use \link[smooth]{msarima} instead.
#'
#' The model selection for SSARIMA is done by the \link[smooth]{auto.ssarima} function.
#'
#' For some more information about the model and its implementation, see the
#' vignette: \code{vignette("ssarima","smooth")}
#'
#' @template ssBasicParam
#' @template ssAdvancedParam
#' @template ssXregParam
#' @template ssAuthor
#' @template ssKeywords
#'
#' @template ADAMInitial
#'
#' @template smoothRef
#' @template ssGeneralRef
#'
#' @param orders Order of the model. Specified as a list, similar to how it is done
#' in  \link[smooth]{adam}, containing vector variables \code{ar}, \code{i} and
#' \code{ma}. If a variable is not provided in the list, then it is assumed to be
#' equal to zero. At least one variable should have the same length as \code{lags}.
#' If the vector is provided (e.g. \code{c(0,1,1)}), a non-seasonal ARIMA will
#' be constructed. In case of \code{auto.ssarima()}, this should be a list of
#' maximum orders to check.
#' The orders are set by a user. If you want the automatic order selection,
#' then use \link[smooth]{auto.ssarima} function instead.
#' @param lags Specifies what should be the seasonal component lag in ARIMA.
#' e.g. \code{lags=c(1,12)} will lead to the seasonal ARIMA with m=12.
#' This can accept several lags, supporting multiple seasonal ETS and ARIMA models.
#' However, due to the model architecture, SSARIMA is slow, and it is recommended to use
#' \link[smooth]{msarima}, \link[smooth]{auto.msarima} or \link[smooth]{adam} in
#' case of multiple frequencies.
#' @param constant If \code{TRUE}, constant term is included in the model. Depending
#' on the order I(d), this can lead to a model with an intercept or a drift. Can
#' also be a number (constant value).
#' In case of \code{auto.ssarima()}, can also be \code{NULL}, in which case the
#' function will check if constant is needed.
#' @param arma Either the named list or a vector with AR / MA parameters ordered lag-wise.
#' The number of elements should correspond to the specified orders e.g.
#' \code{orders=list(ar=c(1,1),ma=c(1,1)), lags=c(1,4), arma=list(ar=c(0.9,0.8),ma=c(-0.3,0.3))}
#' @param model A previously estimated ssarima model, if provided, the function
#' will not estimate anything and will use all its parameters.
#' @param bounds What type of bounds to use in the model estimation. The first
#' letter can be used instead of the whole word. In case of \code{ssarima()}, the
#' "usual" means restricting AR and MA parameters to lie between -1 and 1.
#' @param ...  Other non-documented parameters. See \link[smooth]{adam} for
#' details.
#'
#' @return Object of class "adam" is returned with similar elements to the
#' \link[smooth]{adam} function.
#'
#' @seealso \code{\link[smooth]{auto.ssarima}, \link[smooth]{auto.msarima}, \link[smooth]{adam},
#'  \link[smooth]{es}, \link[smooth]{ces}}
#'
#' @examples
#' # ARIMA(1,1,1) fitted to some data
#' ourModel <- ssarima(rnorm(118,100,3),orders=list(ar=c(1),i=c(1),ma=c(1)),lags=c(1))
#'
#' # Model with the same lags and orders, applied to a different data
#' ssarima(rnorm(118,100,3),orders=orders(ourModel),lags=lags(ourModel))
#'
#' # The same model applied to a different data
#' ssarima(rnorm(118,100,3),model=ourModel)
#'
#' # Example of SARIMA(2,0,0)(1,0,0)[4]
#' \donttest{ssarima(rnorm(118,100,3),orders=list(ar=c(2,1)),lags=c(1,4))}
#'
#' # SARIMA(1,1,1)(0,0,1)[4] with different initialisations
#' \donttest{ssarima(rnorm(118,100,3),orders=list(ar=c(1),i=c(1),ma=c(1,1)),
#'         lags=c(1,4),h=18,holdout=TRUE,initial="backcasting")}
#'
#' @rdname ssarima
#' @export
ssarima <- function(y, orders=list(ar=c(0),i=c(1),ma=c(1)), lags=c(1, frequency(y)),
                    constant=FALSE, arma=NULL, model=NULL,
                    initial=c("backcasting","optimal","two-stage","complete"),
                    loss=c("likelihood","MSE","MAE","HAM","MSEh","TMSE","GTMSE","MSCE","GPL"),
                    h=0, holdout=FALSE, bounds=c("admissible","usual","none"), silent=TRUE,
                    xreg=NULL, regressors=c("use","select","adapt"), initialX=NULL,
                    ...){
##### Function constructs SARIMA model (possible triple seasonality) using state space approach
#
#    Copyright (C) 2016 - Inf Ivan Svetunkov
# Start measuring the time of calculations
    startTime <- Sys.time();
    cl <- match.call();
    # Record the parental environment. Needed for ARIMA initialisation
    env <- parent.frame();
    ellipsis <- list(...);

    # Check seasonality and loss
    loss <- match.arg(loss);

    # paste0() is needed in order to get rid of potential issues with names
    yName <- paste0(deparse(substitute(y)),collapse="");

    # Assume that the model is not provided
    profilesRecentProvided <- FALSE;
    profilesRecentTable <- NULL;

    initialOriginal <- initial;
    # If a previous model provided as a model, write down the variables
    if(!is.null(model)){
        if(is.null(model$model)){
            stop("The provided model is not SSARIMA.",call.=FALSE);
        }
        else if(smoothType(model)!="SSARIMA"){
            stop("The provided model is not SSARIMA.",call.=FALSE);
        }
        # This needs to be fixed to align properly in case of various seasonals
        profilesRecentInitial <- profilesRecentTable <- model$profileInitial;
        profilesRecentProvided[] <- TRUE;
        # This is needed to save initials and to avoid the standard checks
        initial <- initialValueProvided <- model$initial;
        initialOriginal <- model$initialType;
        seasonality <- model$seasonality;
        measurement <- model$measurement;
        transition <- model$transition;
        ellipsis$B <- coef(model);
        lags <- lags(model);
        orders <- orders(model);
        arma <- model$arma;
        arimaPolynomials <- model$other$polynomial;
        arPolynomialMatrix <- model$other$arPolynomialMatrix;
        maPolynomialMatrix <- model$other$maPolynomialMatrix;
        constant <- model$constant;
        if(is.null(constant)){
            constant <- FALSE;
        }

        modelDo <- modelDoOriginal <- "use";
    }
    else{
        modelDo <- modelDoOriginal <- "estimate";
        initialValueProvided <- NULL;
        arimaPolynomials <- NULL;
        arPolynomialMatrix <- maPolynomialMatrix <- NULL;
    }

    # SSARIMA is checked as ADAM ARIMA
    model <- "NNN";

    # Form the data from the provided y and xreg
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

    # If initial was provided, trick parametersChecker
    if(!is.character(initial)){
        initialValueProvided <- initial;
        initial <- "optimal";

        if(is.list(initialValueProvided)){
            initialX <- initialValueProvided$xreg;
        }
    }
    else{
        initialOriginal <- match.arg(initial);
    }
    if(!is.null(initialX)){
        initial <- list(xreg=initialX);
    }

    boundsOriginal <- match.arg(bounds);

    # Default parameters for the wrapper
    distribution <- "dnorm";
    formula <- NULL;
    ic <- "AICc";
    level <- 0.99;
    occurrence <- "none";
    outliers <- "ignore";
    persistence <- NULL;
    phi <- NULL;

    ##### Make all the checks #####
    checkerReturn <- parametersChecker(data=data, model, lags, formulaToUse=formula,
                                       orders=orders,
                                       constant=constant, arma=arma,
                                       outliers=outliers, level=level,
                                       persistence=persistence, phi=phi, initial,
                                       distribution=distribution, loss, h, holdout, occurrence=occurrence,
                                       # This is not needed by the gum() function
                                       ic=ic, bounds=boundsOriginal,
                                       regressors=regressors, yName=yName,
                                       silent, modelDo, ellipsis, fast=FALSE);

    # If the regression was returned, just return it
    if(is.alm(checkerReturn)){
        return(checkerReturn);
    }
    list2env(checkerReturn, envir=environment());

    # A fix to make sure that usual bounds are possible
    bounds <- boundsOriginal;

    # This is the variable needed for the C++ code to determine whether the head of data needs to be
    # refined. In case of SSARIMA this only creates a mess
    refineHead <- TRUE;

    ##### Elements of SSARIMA #####
    filler <- function(B, matVt, matF, vecG, matWt, arRequired=TRUE, maRequired=TRUE, arEstimate=TRUE, maEstimate=TRUE){

        j <- 0;
        # ARMA parameters. This goes before xreg in persistence
        if(arimaModel){
            # This is a failsafe for cases, when model doesn't have any parameters (e.g. I(d) with backcasting)
            if(is.null(B)){
                arimaPolynomials <- lapply(adamCpp$polynomialise(0,
                                                                 arOrders, iOrders, maOrders,
                                                                 arEstimate, maEstimate, armaParameters, lags), as.vector);
            }
            else{
                # Call the function returning ARI and MA polynomials
                arimaPolynomials <- lapply(adamCpp$polynomialise(B[1:sum(c(arOrders*arEstimate,maOrders*maEstimate))],
                                                                 arOrders, iOrders, maOrders,
                                                                 arEstimate, maEstimate, armaParameters, lags), as.vector);
            }

            if(arRequired || any(iOrders>0)){
                # Reset the places for the ma polynomial not to duplicate the values
                vecG[1:length(arimaPolynomials$maPolynomial[-1]),] <- 0
                # Fill in the transition matrix
                matF[1:length(arimaPolynomials$ariPolynomial[-1]),1] <- -arimaPolynomials$ariPolynomial[-1];
                # Fill in the persistence vector
                vecG[1:length(arimaPolynomials$ariPolynomial[-1]),1] <- -arimaPolynomials$ariPolynomial[-1];
                if(maRequired){
                    vecG[1:length(arimaPolynomials$maPolynomial[-1]),1] <- vecG[1:length(arimaPolynomials$maPolynomial[-1]),1] +
                        arimaPolynomials$maPolynomial[-1];
                }
            }
            else{
                if(maRequired){
                    vecG[1:length(arimaPolynomials$maPolynomial[-1]),1] <- arimaPolynomials$maPolynomial[-1];
                }
            }
            j[] <- j+sum(c(arOrders*arEstimate,maOrders*maEstimate));
        }

        # Fill in persistence
        if(xregModel && persistenceEstimate && persistenceXregEstimate){
            # Persistence of xreg
            xregPersistenceNumber <- max(xregParametersPersistence);
            vecG[j+componentsNumberARIMA+1:length(xregParametersPersistence)] <-
                B[j+1:xregPersistenceNumber][xregParametersPersistence];
            j[] <- j+xregPersistenceNumber;
        }

        # Initials of ARIMA
        if(arimaModel && initialArimaEstimate && (any(initialType==c("optimal","two-stage")))){
            matVt[1:initialArimaNumber, 1] <- B[j+1:initialArimaNumber];

            j[] <- j+initialArimaNumber;
        }

        # Initials of the xreg
        if(xregModel && (initialType!="complete") && initialEstimate && initialXregEstimate){
            xregNumberToEstimate <- sum(xregParametersEstimated);
            matVt[componentsNumberARIMA+which(xregParametersEstimated==1),
                  1:lagsModelMax] <- B[j+1:xregNumberToEstimate];
            j[] <- j+xregNumberToEstimate;
            # Normalise initials
            for(i in which(xregParametersMissing!=0)){
                matVt[componentsNumberARIMA+i,
                      1:lagsModelMax] <- -sum(matVt[componentsNumberARIMA+
                                                        which(xregParametersIncluded==xregParametersMissing[i]),
                                                    1:lagsModelMax]);
            }
        }

        # Constant
        if(constantEstimate){
            matVt[componentsNumberARIMA+xregNumber+1,] <- B[j+1];
        }

        return(list(matVt=matVt, matWt=matWt, matF=matF, vecG=vecG, arimaPolynomials=arimaPolynomials));
    }

    ##### Function returns scale parameter for the provided parameters #####
    scaler <- function(errors, obsInSample){
        return(sqrt(sum(errors^2)/obsInSample));
    }

    ##### Cost function #####
    CF <- function(B, matVt, matF, vecG, matWt, arRequired=TRUE, maRequired=TRUE,
                   arEstimate=TRUE, maEstimate=TRUE){
        # Obtain the main elements
        elements <- filler(B, matVt, matF, vecG, matWt, arRequired=arRequired, maRequired=maRequired,
                           arEstimate=arEstimate, maEstimate=maEstimate);

        #### The usual bounds ####
        if(bounds=="usual"){
            # Stationarity and invertibility conditions for ARIMA
            if(arimaModel && any(c(arEstimate,maEstimate))){
                # Calculate the polynomial roots for AR
                if(arEstimate &&
                   any(abs(elements$arimaPolynomials$maPolynomial[-1])>=1)){
                   # all(elements$arimaPolynomials$arPolynomial[-1]>0) &&
                   # sum(-(elements$arimaPolynomials$arPolynomial[-1]))>=1){
                    # arPolynomialMatrix[,1] <- -elements$arimaPolynomials$arPolynomial[-1];
                    # arPolyroots <- abs(eigen(arPolynomialMatrix, symmetric=FALSE, only.values=TRUE)$values);
                    # if(any(arPolyroots>1)){
                        return(1E+100);
                    # }
                }
                # Calculate the polynomial roots of MA
                if(maEstimate &&
                   any(abs(elements$arimaPolynomials$maPolynomial[-1])>=1)){
                   # sum(elements$arimaPolynomials$maPolynomial[-1])>=1){
                    # maPolynomialMatrix[,1] <- elements$arimaPolynomials$maPolynomial[-1];
                    # maPolyroots <- abs(eigen(maPolynomialMatrix, symmetric=FALSE, only.values=TRUE)$values);
                    # if(any(maPolyroots>1)){
                    #     return(1E+100*max(abs(maPolyroots)));
                    # }
                    return(1E+100);
                }
            }

            # Smoothing parameters for the explanatory variables (0, 1) region
            if(xregModel && regressors=="adapt"){
                if(any(elements$vecG[componentsNumberARIMA+1:xregNumber]>1) ||
                   any(elements$vecG[componentsNumberARIMA+1:xregNumber]<0)){
                    return(1E+100*max(abs(elements$vecG[componentsNumberARIMA+1:xregNumber]-0.5)));
                }
            }
        }
        #### The admissible bounds ####
        else if(bounds=="admissible"){
            if(arimaModel){
                # Stationarity condition of ARIMA
                # Calculate the polynomial roots for AR
                if(arEstimate &&
                   (all(-elements$arimaPolynomials$arPolynomial[-1]>0) &
                    sum(-(elements$arimaPolynomials$arPolynomial[-1]))>=1)){
                    arPolynomialMatrix[,1] <- -elements$arimaPolynomials$arPolynomial[-1];
                    eigenValues <- abs(eigen(arPolynomialMatrix, symmetric=FALSE, only.values=TRUE)$values);
                    if(any(eigenValues>1)){
                        return(1E+100*max(eigenValues));
                    }
                }

                # Stability / invertibility condition
                eigenValues <- smoothEigens(elements$vecG, elements$matF, matWt,
                                            lagsModelAll, xregModel, obsInSample);
                if(any(eigenValues>1+1E-50)){
                    return(1E+100*max(eigenValues));
                }

                # # Stability/Invertibility condition of ARIMA
                # if(xregModel){
                #     if(regressors=="adapt"){
                #         # We check the condition on average
                #         eigenValues <- abs(eigen((elements$matF -
                #                                       diag(as.vector(elements$vecG)) %*%
                #                                       t(measurementInverter(elements$matWt[1:obsInSample,,drop=FALSE])) %*%
                #                                       elements$matWt[1:obsInSample,,drop=FALSE] / obsInSample),
                #                                  symmetric=FALSE, only.values=TRUE)$values);
                #     }
                #     else{
                #         # We drop the X parts from matrices
                #         indices <- c(1:(componentsNumberARIMA))
                #         eigenValues <- abs(eigen(elements$matF[indices,indices,drop=FALSE] -
                #                                      elements$vecG[indices,,drop=FALSE] %*%
                #                                      elements$matWt[obsInSample,indices,drop=FALSE],
                #                                  symmetric=FALSE, only.values=TRUE)$values);
                #     }
                # }
                # else{
                #     if(arimaModel && maEstimate && (sum(elements$arimaPolynomials$maPolynomial[-1])>=1 |
                #                                                  sum(elements$arimaPolynomials$maPolynomial[-1])<0)){
                #         eigenValues <- abs(eigen(elements$matF -
                #                                      elements$vecG %*% elements$matWt[obsInSample,,drop=FALSE],
                #                                  symmetric=FALSE, only.values=TRUE)$values);
                #     }
                #     else{
                #         eigenValues <- 0;
                #     }
                # }
                # if(any(eigenValues>1+1E-50)){
                #     return(1E+100*max(eigenValues));
                # }
            }
        }

        # Write down the initials in the recent profile
        matVt[,1] <- profilesRecentTable[] <- elements$matVt[,1, drop=FALSE];

        adamFitted <- adamCpp$fit(matVt, elements$matWt,
                                  elements$matF, elements$vecG,
                                  indexLookupTable, profilesRecentTable,
                                  yInSample, ot,
                                  any(initialType==c("complete","backcasting")), nIterations,
                                  refineHead, "n");

        if(!multisteps){
            if(loss=="likelihood"){
                # Scale for different functions
                scale <- scaler(adamFitted$errors[otLogical], obsInSample);

                # Calculate the likelihood
                CFValue <- -sum(dnorm(x=yInSample[otLogical],
                                      mean=adamFitted$fitted[otLogical],
                                      sd=scale, log=TRUE));
            }
            else if(loss=="MSE"){
                CFValue <- sum(adamFitted$errors^2)/obsInSample;
            }
            else if(loss=="MAE"){
                CFValue <- sum(abs(adamFitted$errors))/obsInSample;
            }
            else if(loss=="HAM"){
                CFValue <- sum(sqrt(abs(adamFitted$errors)))/obsInSample;
            }
            else if(loss=="custom"){
                CFValue <- lossFunction(actual=yInSample,fitted=adamFitted$fitted,B=B);
            }
        }
        else{
            # Call for the Rcpp function to produce a matrix of multistep errors
            adamErrors <- adamCpp$ferrors(adamFitted$states, elements$matWt,
                                          elements$matF,
                                          indexLookupTable, profilesRecentTable,
                                          h, yInSample)$errors;

            # Not done yet: "aMSEh","aTMSE","aGTMSE","aMSCE","aGPL"
            CFValue <- switch(loss,
                              "MSEh"=sum(adamErrors[,h]^2)/(obsInSample-h),
                              "TMSE"=sum(colSums(adamErrors^2)/(obsInSample-h)),
                              "GTMSE"=sum(log(colSums(adamErrors^2)/(obsInSample-h))),
                              "MSCE"=sum(rowSums(adamErrors)^2)/(obsInSample-h),
                              "MAEh"=sum(abs(adamErrors[,h]))/(obsInSample-h),
                              "TMAE"=sum(colSums(abs(adamErrors))/(obsInSample-h)),
                              "GTMAE"=sum(log(colSums(abs(adamErrors))/(obsInSample-h))),
                              "MACE"=sum(abs(rowSums(adamErrors)))/(obsInSample-h),
                              "HAMh"=sum(sqrt(abs(adamErrors[,h])))/(obsInSample-h),
                              "THAM"=sum(colSums(sqrt(abs(adamErrors)))/(obsInSample-h)),
                              "GTHAM"=sum(log(colSums(sqrt(abs(adamErrors)))/(obsInSample-h))),
                              "CHAM"=sum(sqrt(abs(rowSums(adamErrors))))/(obsInSample-h),
                              "GPL"=log(det(t(adamErrors) %*% adamErrors/(obsInSample-h))),
                              0);
        }

        if(is.na(CFValue) || is.nan(CFValue)){
            CFValue[] <- 1e+300;
        }

        return(CFValue);
    }

    #### Likelihood function ####
    logLikFunction <- function(B, matVt, matF, vecG, matWt,
                               arRequired=TRUE, maRequired=TRUE,
                               arEstimate=TRUE, maEstimate=TRUE){
        return(-CF(B, matVt=matVt, matF=matF, vecG=vecG, matWt=matWt,
                   arRequired=arRequired, maRequired=maRequired,
                   arEstimate=arEstimate, maEstimate=maEstimate));
    }

    #### Basic ARIMA parameters ####
    # Fix orders if they were all zero and became NULL
    if(is.null(arOrders)){
        arOrders <- 0;
        iOrders <- 0;
        maOrders <- 0;
    }

    # If there are zero lags, drop them
    if(any(lags==0)){
        arOrders <- arOrders[lags!=0];
        iOrders <- iOrders[lags!=0];
        maOrders <- maOrders[lags!=0];
        lags <- lags[lags!=0];
    }

    if(any(lags>48)){
        warning(paste0("SSARIMA is quite slow with lags greater than 48. ",
                       "It is recommended to use MSARIMA in this case instead."),
                call.=FALSE);
    }

    # Define maxorder and make all the values look similar (for the polynomials)
    maxorder <- max(length(arOrders),length(iOrders),length(maOrders));
    if(length(arOrders)!=maxorder){
        arOrders <- c(arOrders,rep(0,maxorder-length(arOrders)));
    }
    if(length(iOrders)!=maxorder){
        iOrders <- c(iOrders,rep(0,maxorder-length(iOrders)));
    }
    if(length(maOrders)!=maxorder){
        maOrders <- c(maOrders,rep(0,maxorder-length(maOrders)));
    }

    if((length(lags)!=length(arOrders)) & (length(lags)!=length(iOrders)) & (length(lags)!=length(maOrders))){
        stop("Seasonal lags do not correspond to any element of SARIMA",call.=FALSE);
    }

    # Get rid of duplicates in lags
    if(length(unique(lags))!=length(lags)){
        if(dataFreq!=1){
            warning(paste0("'lags' variable contains duplicates: (",paste0(lags,collapse=","),
                           "). Getting rid of some of them."),call.=FALSE);
        }
        lagsNew <- unique(lags);
        arOrdersNew <- iOrdersNew <- maOrdersNew <- lagsNew;
        for(i in 1:length(lagsNew)){
            arOrdersNew[i] <- max(arOrders[which(lags==lagsNew[i])]);
            iOrdersNew[i] <- max(iOrders[which(lags==lagsNew[i])]);
            maOrdersNew[i] <- max(maOrders[which(lags==lagsNew[i])]);
        }
        arOrders <- arOrdersNew;
        iOrders <- iOrdersNew;
        maOrders <- maOrdersNew;
        lags <- lagsNew;
    }
    # Recollect the orders to reuse them in other functions
    orders <- list(ar=arOrders, i=iOrders, ma=maOrders);

    componentsNumberARIMA <- max(arOrders %*% lags + iOrders %*% lags, maOrders %*% lags);

    # componentsNumberAll is the ARIMA components + intercept/drift
    componentsNumberAll <- componentsNumberARIMA + constantRequired;

    lagsModelAll <- matrix(rep(1,componentsNumberAll+xregNumber),ncol=1);
    lagsModelMax <- 1;

    # Create C++ adam class, which will then use fit, forecast etc methods
    adamCpp <- new(adamCore,
                   lagsModelAll, Etype, Ttype, Stype,
                   componentsNumberETSNonSeasonal,
                   componentsNumberETSSeasonal,
                   componentsNumberETS, componentsNumberARIMA,
                   xregNumber, length(lagsModelAll),
                   constantRequired, FALSE);

    if(!is.null(initialValueProvided)){
        initialType <- "provided";
    }
    initialValue <- initialValueProvided;

    initialValueARIMA <- initialValue;
    if(is.list(initialValue)){
        xregModelInitials[[1]][[1]] <- initialValue$xreg;
        initialValueARIMA <- initialValue$arima;
    }

    initial <- initialOriginal;

    if(any(initialType==c("optimal","two-stage"))){
        initialArimaNumber <- componentsNumberARIMA;
    }

    ##### Preset values of matvt and other matrices ######
    matF <- diag(componentsNumberAll+xregNumber);
    vecG <- matrix(0,componentsNumberAll+xregNumber,1,
                   dimnames=list(c(paste0("psi",1:(componentsNumberARIMA))[componentsNumberARIMA>0],
                                   paste0("delta",1:xregNumber)[xregNumber>0],
                                   constantName), NULL));
    matWt <- matrix(1, obsInSample, componentsNumberAll+xregNumber);
    matVt <- matrix(0, componentsNumberAll+xregNumber, obsStates,
                    dimnames=list(c(paste0("Component ",1:(componentsNumberARIMA))[componentsNumberARIMA>0],
                                    xregNames, constantName), NULL));

    if(componentsNumberARIMA > 0){
        # Transition matrix, measurement vector and persistence vector + state vector
        if(componentsNumberARIMA>1){
            matF[1,1] <- 0;
            matF[componentsNumberARIMA,componentsNumberARIMA] <- 0;
            matF[1:(componentsNumberARIMA-1),2:componentsNumberARIMA] <- diag(componentsNumberARIMA-1);
            matWt[,2:componentsNumberARIMA] <- 0;
        }
        if(initialType=="provided"){
            matVt[1:componentsNumberARIMA,1] <- initialValueARIMA;
            initialArimaEstimate <- FALSE;
        }
        else{
            yDifferenced <- yInSample;
            # If the model has differences, take them
            if(any(iOrders>0) && (any(arOrders>0) || any(maOrders>0) || constantRequired)){
                for(i in 1:length(iOrders)){
                    if(iOrders[i]>0){
                        yDifferenced <- diff(yDifferenced,lag=lags[i],differences=iOrders[i]);
                    }
                }
            }
            obsDiff <- length(yDifferenced)
            # If the number of components is larger than the sample size (after taking differences)
            if(obsDiff<componentsNumberARIMA){
                matVt[1:componentsNumberARIMA,1] <- mean(yDifferenced[obsDiff:1]);
            }
            else if(all(iOrders==0) || (any(lags==1) && iOrders[lags==1]==0) ||
               obsDiff<(componentsNumberARIMA+as.vector(iOrders %*% lags)-1)){
                matVt[1:componentsNumberARIMA,1] <- yDifferenced[min(componentsNumberARIMA,obsInSample):1]-
                    mean(yDifferenced[min(componentsNumberARIMA,obsInSample):1]);
            }
            else{
                # matVt[1:componentsNumberARIMA,1] <- yDifferenced[1:componentsNumberARIMA + as.vector(iOrders %*% lags)-1];
                matVt[1:componentsNumberARIMA,1] <- yDifferenced[1:componentsNumberARIMA];
            }
            matVt[1,1] <- yInSample[1];
        }
    }

    # Add parameters for the X
    if(xregModel){
        matWt[,componentsNumberARIMA+c(1:xregNumber)] <- xregData[1:obsInSample,];
        if(initialXregEstimate && initialType!="complete"){
            matVt[componentsNumberARIMA+c(1:xregNumber),1] <- xregModelInitials[[1]][[1]];
        }
    }

    # Add element for the intercept in the transition matrix
    if(constantRequired){
        matF[1,componentsNumberAll+xregNumber] <- 1;
        # If the model has differences, set this to zero
        if(constantEstimate){
            if(any(iOrders>0)){
                matVt[componentsNumberAll+xregNumber,] <- 0;
            }
            else{
                matVt[componentsNumberAll+xregNumber,] <- mean(yInSample[1:componentsNumberAll]);
            }
        }
        else{
            matVt[componentsNumberAll+xregNumber,] <- constantValue;
        }
    }

    ##### Pre-set yFitted, yForecast, errors and basic parameters #####
    # Prepare fitted and error with ts / zoo
    if(any(yClasses=="ts")){
        yFitted <- ts(rep(NA,obsInSample), start=yStart, frequency=yFrequency);
        errors <- ts(rep(NA,obsInSample), start=yStart, frequency=yFrequency);
    }
    else{
        yFitted <- zoo(rep(NA,obsInSample), order.by=yInSampleIndex);
        errors <- zoo(rep(NA,obsInSample), order.by=yInSampleIndex);
    }
    yForecast <- rep(NA, h);

    # Values for occurrence. No longer supported in ssarima()
    parametersNumber[1,3] <- parametersNumber[2,3] <- 0;
    # Xreg parameters
    parametersNumber[1,2] <- xregNumber + sum(persistenceXreg);
    # Scale value
    parametersNumber[1,4] <- 1;

    # Fix modelDo based on everything that needs to be estimated
    modelDo <- c("use","estimate")[any(arEstimate, maEstimate,
                                       initialArimaEstimate & any(initialType==c("optimal","two-stage")),
                                       persistenceEstimate, persistenceXregEstimate,
                                       initialXregEstimate, constantEstimate)+1];

    #### If we need to estimate the model ####
    if(modelDo=="estimate"){
        # Create ADAM profiles for correct treatment of seasonality
        adamProfiles <- adamProfileCreator(lagsModelAll, lagsModelMax, obsAll,
                                           lags=lags, yIndex=yIndexAll, yClasses=yClasses);
        profilesRecentTable <- adamProfiles$recent;
        if(initialType=="provided"){
            profilesRecentTable[1:componentsNumberARIMA,1] <- matVt[1:componentsNumberARIMA,1];
        }
        indexLookupTable <- adamProfiles$lookup;

        #### Initialise vector B ####
        initialiser <- function(...){
            B <- Bl <- Bu <- vector("numeric",
                                    # Dynamic ADAMX
                                    xregModel*persistenceXregEstimate*max(xregParametersPersistence) +
                                        # AR and MA values
                                        (arEstimate*sum(arOrders)+maEstimate*sum(maOrders)) +
                                        # initials of ARIMA
                                        all(initialType!=c("complete","backcasting"))*initialArimaNumber*initialArimaEstimate +
                                        # initials of xreg
                                        (initialType!="complete")*xregModel*initialXregEstimate*sum(xregParametersEstimated) +
                                        constantEstimate);

            j <- 0;
            if(xregModel && persistenceXregEstimate){
                xregPersistenceNumber <- max(xregParametersPersistence);
                B[j+1:xregPersistenceNumber] <- rep(0.01, xregPersistenceNumber);
                Bl[j+1:xregPersistenceNumber] <- rep(-5, xregPersistenceNumber);
                Bu[j+1:xregPersistenceNumber] <- rep(5, xregPersistenceNumber);
                names(B)[j+1:xregPersistenceNumber] <- paste0("delta",c(1:xregPersistenceNumber));
                j[] <- j+xregPersistenceNumber;
            }

            # These are filled in lags-wise
            if(any(c(arEstimate,maEstimate))){
                acfValues <- rep(-0.1, maOrders %*% lags);
                pacfValues <- rep(0.1, arOrders %*% lags);
                if(!all(iOrders==0)){
                    yDifferenced <- yInSample;
                    # If the model has differences, take them
                    if(any(iOrders>0)){
                        for(i in 1:length(iOrders)){
                            if(iOrders[i]>0){
                                yDifferenced <- diff(yDifferenced,lag=lags[i],differences=iOrders[i]);
                            }
                        }
                    }
                    # Do ACF/PACF initialisation only for non-seasonal models
                    # if(all(lags<=1)){
                    if(maRequired && maEstimate){
                        # If the sample is smaller than lags, it will be substituted by default values
                        acfValues[1:min(maOrders %*% lags, length(yDifferenced)-1)] <-
                            acf(yDifferenced,lag.max=max(1,maOrders %*% lags),plot=FALSE)$acf[-1];
                    }
                    if(arRequired && arEstimate){
                        # If the sample is smaller than lags, it will be substituted by default values
                        pacfValues[1:min(arOrders %*% lags, length(yDifferenced)-1)] <-
                            pacf(yDifferenced,lag.max=max(1,arOrders %*% lags),plot=FALSE)$acf;
                    }
                    # }
                }
                for(i in 1:length(lags)){
                    if(arRequired && arEstimate && arOrders[i]>0){
                        if(all(!is.nan(pacfValues[c(1:arOrders[i])*lags[i]]))){
                            B[j+c(1:arOrders[i])] <- pacfValues[c(1:arOrders[i])*lags[i]];
                        }
                        else{
                            B[j+c(1:arOrders[i])] <- 0.1;
                        }
                        if(sum(B[j+c(1:arOrders[i])])>1){
                            B[j+c(1:arOrders[i])] <- B[j+c(1:arOrders[i])] / sum(B[j+c(1:arOrders[i])]) - 0.01;
                        }
                        # B[j+c(1:arOrders[i])] <- rep(0.1,arOrders[i]);
                        Bl[j+c(1:arOrders[i])] <- -5;
                        Bu[j+c(1:arOrders[i])] <- 5;
                        names(B)[j+1:arOrders[i]] <- paste0("phi",1:arOrders[i],"[",lags[i],"]");
                        j[] <- j + arOrders[i];
                    }
                    if(maRequired && maEstimate && maOrders[i]>0){
                        if(all(!is.nan(acfValues[c(1:maOrders[i])*lags[i]]))){
                            B[j+c(1:maOrders[i])] <- acfValues[c(1:maOrders[i])*lags[i]];
                        }
                        else{
                            B[j+c(1:maOrders[i])] <- 0.1;
                        }
                        if(sum(B[j+c(1:maOrders[i])])>1){
                            B[j+c(1:maOrders[i])] <- B[j+c(1:maOrders[i])] / sum(B[j+c(1:maOrders[i])]) - 0.01;
                        }
                        # B[j+c(1:maOrders[i])] <- rep(-0.1,maOrders[i]);
                        Bl[j+c(1:maOrders[i])] <- -5;
                        Bu[j+c(1:maOrders[i])] <- 5;
                        names(B)[j+1:maOrders[i]] <- paste0("theta",1:maOrders[i],"[",lags[i],"]");
                        j[] <- j + maOrders[i];
                    }
                }
            }

            # ARIMA initials
            if(all(initialType!=c("complete","backcasting")) && initialArimaEstimate){
                B[j+1:initialArimaNumber] <- matVt[1:initialArimaNumber,1];
                names(B)[j+1:initialArimaNumber] <- paste0("ARIMAState",1:initialArimaNumber);

                Bl[j+1:initialArimaNumber] <- -Inf;
                Bu[j+1:initialArimaNumber] <- Inf;
                j[] <- j+initialArimaNumber;
            }

            # Initials of the xreg
            if(xregModel && initialType!="complete" && initialXregEstimate){
                xregNumberToEstimate <- sum(xregParametersEstimated);
                B[j+1:xregNumberToEstimate] <- matVt[componentsNumberARIMA+
                                                         which(xregParametersEstimated==1),1];
                names(B)[j+1:xregNumberToEstimate] <- xregNames;
                if(Etype=="A"){
                    Bl[j+1:xregNumberToEstimate] <- -Inf;
                    Bu[j+1:xregNumberToEstimate] <- Inf;
                }
                else{
                    Bl[j+1:xregNumberToEstimate] <- -Inf;
                    Bu[j+1:xregNumberToEstimate] <- Inf;
                }
                j[] <- j+xregNumberToEstimate;
            }

            if(constantEstimate){
                j[] <- j+1;
                B[j] <- matVt[componentsNumberARIMA+xregNumber+1,1];
                names(B)[j] <- constantName;
                if(sum(iOrders)!=0){
                    Bu[j] <- quantile(diff(yInSample[otLogical]),0.6);
                    Bl[j] <- -Bu[j];

                    # Failsafe for weird cases, when upper bound is the same or lower than the lower one
                    if(Bu[j]<=Bl[j]){
                        Bu[j] <- Inf;
                        Bl[j] <- -Inf;
                    }

                    # Failsafe for cases, when the B is outside of bounds
                    if(B[j]<=Bl[j]){
                        Bl[j] <- -Inf;
                    }
                    if(B[j]>=Bu[j]){
                        Bu[j] <- Inf;
                    }
                }
                else{
                    Bu[j] <- max(abs(yInSample[otLogical]),abs(B[j])*1.01);
                    Bl[j] <- -Bu[j];
                }
            }
            return(list(B=B, Bu=Bu, Bl=Bl));
        }

        BValues <- initialiser();

        ##### Pre-heat initial parameters by doing the backcasted ARIMA ####
        if(arimaModel && initialType=="two-stage" && is.null(B)){
            # Estimate ARIMA with backcasting first
            clNew <- cl;
            # If environment is provided, use it
            if(!is.null(ellipsis$environment)){
                env <- ellipsis$environment;
            }
            # Use complete backcasting
            clNew$initial <- "complete";
            # Shut things up
            clNew$silent <- TRUE;
            # If this is an xreg model, we do selection, and there's no formula, create one
            if(xregModel && !is.null(clNew$regressors) && clNew$regressors=="select"){
                clNew$formula <- as.formula(paste0(responseName,"~",paste0(xregNames,collapse="+")));
            }
            # Switch off regressors selection
            if(!is.null(clNew$regressors) && clNew$regressors=="select"){
                clNew$regressors <- "use";
            }
            # Get rid of explanatory variables if they are not needed
            if(!xregModel && (!is.null(ncol(data)) && ncol(data)>1)){
                clNew$data <- data[,responseName];
            }
            # Call for ADAM with backcasting
            ssarimaBack <- suppressWarnings(eval(clNew, envir=env));
            B <- BValues$B;
            # Number of smoothing, dampening and ARMA parameters
            nParametersBack <- (xregModel*persistenceXregEstimate*max(xregParametersPersistence) +
                                    # AR and MA values
                                    arimaModel*(arEstimate*sum(arOrders)+maEstimate*sum(maOrders)));
            if(nParametersBack>0){
                # Use the estimated parameters
                B[1:nParametersBack] <- ssarimaBack$B[1:nParametersBack];
            }

            # If there are explanatory variables, use only those initials that are required
            if(xregModel){
                ssarimaBack$initial$xreg <- ssarimaBack$initial$xreg[xregParametersEstimated==1];
            }

            initialsUnlisted <- unlist(ssarimaBack$initial);
            # If initials are reasonable, use them
            if(!any(is.na(initialsUnlisted))){
                B[nParametersBack + c(1:length(initialsUnlisted))] <- initialsUnlisted;
            }

            # If the constant is used and it's good, record it
            if(constantEstimate && !is.na(ssarimaBack$constant)){
                B[nParametersBack+componentsNumberARIMA+xregNumber+1] <- ssarimaBack$constant;
            }

            # Make sure that the bounds are reasonable
            # if(any(is.na(lb))){
            #     lb[is.na(lb)] <- -Inf;
            # }
            # if(any(lb>B)){
            #     lb[lb>B] <- -Inf;
            # }
            # if(any(is.na(ub))){
            #     ub[is.na(ub)] <- Inf;
            # }
            # if(any(ub<B)){
            #     ub[ub<B] <- Inf;
            # }
        }

        if(is.null(B)){
            B <- BValues$B;
            # Parameter bounds.
            # Not used in the estimation because they lead to wrong estimates
            # lb <- BValues$Bl;
            # ub <- BValues$Bu;
        }

        # Companion matrices for the polynomials calculation -> stationarity / stability checks
        if(arimaModel){
            # AR polynomials
            arPolynomialMatrix <- matrix(0, arOrders %*% lags, arOrders %*% lags);
            if(nrow(arPolynomialMatrix)>1){
                arPolynomialMatrix[2:nrow(arPolynomialMatrix)-1,2:nrow(arPolynomialMatrix)] <- diag(nrow(arPolynomialMatrix)-1);
            }
            # MA polynomials
            maPolynomialMatrix <- matrix(0, maOrders %*% lags, maOrders %*% lags);
            if(nrow(maPolynomialMatrix)>1){
                maPolynomialMatrix[2:nrow(maPolynomialMatrix)-1,2:nrow(maPolynomialMatrix)] <- diag(nrow(maPolynomialMatrix)-1);
            }
        }

        #### Parameters of the nloptr and optimisation ####
        # Print level defined
        print_level_hidden <- print_level;
        if(print_level==41){
            cat("Initial parameters:",B,"\n");
            print_level[] <- 0;
        }

        # maxeval based on what was provided
        maxevalUsed <- maxeval;
        if(is.null(maxeval)){
            maxevalUsed <- length(B) * 40;
            if(xregModel && initialType!="complete"){
                maxevalUsed[] <- length(B) * 100;
                maxevalUsed[] <- max(1000,maxevalUsed);
            }
        }

        # Tuning the best obtained values using Nelder-Mead
        res <- suppressWarnings(nloptr(B, CF,# lb=lb, ub=ub,
                                       opts=list(algorithm=algorithm, xtol_rel=xtol_rel, xtol_abs=xtol_abs,
                                                 ftol_rel=ftol_rel, ftol_abs=ftol_abs,
                                                 maxeval=maxevalUsed, maxtime=maxtime, print_level=print_level),
                                       matVt=matVt, matF=matF, vecG=vecG, matWt=matWt,
                                       arRequired=arRequired, maRequired=maRequired,
                                       arEstimate=arEstimate, maEstimate=maEstimate));

        if(print_level_hidden>0){
            print(res);
        }

        B[] <- res$solution;
        CFValue <- res$objective;

        nStatesBackcasting <- 0;
        # Calculate the number of degrees of freedom coming from states in case of backcasting
        if(any(initialType==c("backcasting","complete"))){
            # Obtain the main elements
            ssarimaFilled <- filler(B, matVt, matF, vecG, matWt,
                                    arRequired=arRequired, maRequired=maRequired,
                                    arEstimate=arEstimate, maEstimate=maEstimate);

            nStatesBackcasting[] <- calculateBackcastingDF(profilesRecentTable, lagsModelAll,
                                                           FALSE, Stype, componentsNumberETSNonSeasonal,
                                                           componentsNumberETSSeasonal, ssarimaFilled$vecG, ssarimaFilled$matF,
                                                           obsInSample, lagsModelMax, indexLookupTable,
                                                           adamCpp, dfForBack);
        }

        # Parameters estimated + variance
        nParamEstimated <- length(B) + (loss=="likelihood")*1 + nStatesBackcasting;

        # Prepare for fitting
        elements <- filler(B, matVt, matF, vecG, matWt, arRequired=arRequired, maRequired=maRequired,
                           arEstimate=arEstimate, maEstimate=maEstimate);
        matF[] <- elements$matF;
        vecG[] <- elements$vecG;
        matVt[,1] <- elements$matVt[,1];
        matWt[] <- elements$matWt;
        arimaPolynomials <- elements$arimaPolynomials;

        # Write down the initials in the recent profile
        profilesRecentInitial <- profilesRecentTable[] <- matVt[,1,drop=FALSE];
        parametersNumber[1,1] <- length(B) + nStatesBackcasting;
    }
    #### If we just use the provided values ####
    else{
        # Setting this to zero is essential for the polynomialiser to work
        if(is.null(B)){
            B <- 0;
        }

        # Create index lookup table
        adamProfiles <- adamProfileCreator(lagsModelAll, lagsModelMax, obsAll,
                                       lags=lags, yIndex=yIndexAll, yClasses=yClasses);
        indexLookupTable <- adamProfiles$lookup;
        if(is.null(profilesRecentTable)){
            profilesRecentInitial <- profilesRecentTable <- adamProfiles$recent;
        }

        if(any(initialType==c("optimal","two-stage","provided"))){
            initialType <- "provided";
            profilesRecentInitial[,1] <- profilesRecentTable[,1] <- matVt[,1];
        }
        else{
            initialType <- initialOriginal[1];
        }

        initialXregEstimateOriginal <- initialXregEstimate;
        initialXregEstimate <- FALSE;

        # Prepare for fitting
        elements <- filler(armaParameters, matVt, matF, vecG, matWt,
                           arRequired=arRequired, maRequired=maRequired,
                           arEstimate=arEstimate, maEstimate=maEstimate);
        matF[] <- elements$matF;
        vecG[] <- elements$vecG;
        matVt[,1] <- profilesRecentTable[] <- elements$matVt[,1, drop=FALSE];
        matWt[] <- elements$matWt;
        arimaPolynomials <- elements$arimaPolynomials;

        CFValue <- CF(B, matVt, matF, vecG, matWt,
                      arRequired=arRequired, maRequired=maRequired,
                      arEstimate=arEstimate, maEstimate=maEstimate);
        res <- NULL;

        # Only variance is estimated
        nParamEstimated <- 1;

        initialType <- initialOriginal;
        initialXregEstimate <- initialXregEstimateOriginal;
    }


    #### Fisher Information ####
    if(FI){
        # Substitute values to get hessian
        if(any(substr(names(B),1,3)=="phi")){
            arEstimateOriginal <- arEstimate;
            arEstimate <- arRequired;
        }
        if(any(substr(names(B),1,5)=="theta")){
            maEstimateOriginal <- maEstimate;
            maEstimate <- maRequired;
        }
        if(any(substr(names(B),1,10)=="ARIMAState")){
            initialArimaEstimateOriginal <- initialArimaEstimate;
            initialArimaEstimate <- TRUE;
        }

        initialTypeOriginal <- initialType;
        initialType <- switch(initialType,
                              "complete"=,
                              "backcasting"="provided",
                              initialType);
        if(!is.null(initialValueProvided$xreg) && initialOriginal!="complete"){
            initialXregEstimateOriginal <- initialXregEstimate;
            initialXregEstimate <- TRUE;
        }
        if(constantRequired){
            constantEstimateOriginal <- constantEstimate;
            constantEstimate <- constantRequired;
        }
        boundsOriginal <- bounds;
        bounds <- "none";

        # Calculate hessian
        FI <- -hessian(logLikFunction, B, h=stepSize, matVt=matVt, matF=matF, vecG=vecG, matWt=matWt,
                       arRequired=arRequired, maRequired=maRequired,
                       arEstimate=arEstimate, maEstimate=maEstimate);
        colnames(FI) <- rownames(FI) <- names(B);

        if(any(substr(names(B),1,3)=="phi")){
            arEstimate <- arEstimateOriginal;
        }
        if(any(substr(names(B),1,5)=="theta")){
            maEstimate <- maEstimateOriginal;
        }
        if(any(substr(names(B),1,10)=="ARIMAState")){
            initialArimaEstimate <- initialArimaEstimateOriginal;
        }
        initialType <- initialTypeOriginal;
        if(!is.null(initialValueProvided$xreg) && initialOriginal!="complete"){
            initialXregEstimate <- initialXregEstimateOriginal;
        }
        if(constantRequired){
            constantEstimate <- constantEstimateOriginal;
        }
        bounds <- boundsOriginal;
    }
    else{
        FI <- NA;
    }

    # In case of likelihood, we typically have one more parameter to estimate - scale.
    logLikValue <- structure(logLikFunction(B, matVt=matVt, matF=matF, vecG=vecG, matWt=matWt,
                                            arRequired=arRequired, maRequired=maRequired,
                                            arEstimate=arEstimate, maEstimate=maEstimate),
                             nobs=obsInSample, df=nParamEstimated, class="logLik");

    adamFitted <- adamCpp$fit(matVt, matWt,
                              matF, vecG,
                              indexLookupTable, profilesRecentTable,
                              yInSample, ot,
                              any(initialType==c("complete","backcasting")), nIterations,
                              refineHead, "n");

    errors[] <- adamFitted$errors;
    yFitted[] <- adamFitted$fitted;
    # Write down the recent profile for future use
    profilesRecentTable <- adamFitted$profile;
    matVt[] <- adamFitted$states;

    # Write down the initials in the recent profile
    if(!any(initialType==c("complete","backcasting"))){
        profilesRecentInitial <- matVt[,1,drop=FALSE];
    }

    scale <- scaler(adamFitted$errors[otLogical], obsInSample);

    if(any(yClasses=="ts")){
        yForecast <- ts(rep(NA, max(1,h)), start=yForecastStart, frequency=yFrequency);
    }
    else{
        yForecast <- zoo(rep(NA, max(1,h)), order.by=yForecastIndex);
    }
    if(h>0){
        yForecast[] <- adamCpp$forecast(tail(matWt,h), matF,
                                        indexLookupTable[,lagsModelMax+obsInSample+c(1:h),drop=FALSE],
                                        profilesRecentTable,
                                        h)$forecast;
    }
    else{
        yForecast[] <- NA;
    }

    ##### Do final check and make some preparations for output #####
    # Write down initials of states vector and exogenous
    if(initialType!="provided"){
        initialValue <- list();
        if(arimaModel){
            initialValue$arima <- matVt[1:componentsNumberARIMA,1];
        }
    }
    if(xregModel){
        initialValue$xreg <- matVt[componentsNumberARIMA+1:xregNumber,1];
    }
    if(constantRequired){
        constantValue <- matVt[componentsNumberAll+xregNumber,1]
    }
    else{
        constantValue <- NULL;
    }
    if(arimaModel){
        armaParametersList <- vector("list",arRequired+maRequired);
        j <- 1;
        if(arRequired && arEstimate){
            # Avoid damping parameter phi
            armaParametersList[[j]] <- B[nchar(names(B))>3 & substr(names(B),1,3)=="phi"];
            names(armaParametersList)[j] <- "ar";
            j[] <- j+1;
        }
        # If this was provided
        else if(arRequired && !arEstimate){
            # Avoid damping parameter phi
            armaParametersList[[j]] <- armaParameters[substr(names(armaParameters),1,3)=="phi"];
            names(armaParametersList)[j] <- "ar";
            j[] <- j+1;
        }
        if(maRequired && maEstimate){
            armaParametersList[[j]] <- B[substr(names(B),1,5)=="theta"];
            names(armaParametersList)[j] <- "ma";
        }
        else if(maRequired && !maEstimate){
            armaParametersList[[j]] <- armaParameters[substr(names(armaParameters),1,5)=="theta"];
            names(armaParametersList)[j] <- "ma";
        }

        otherReturned <- list(polynomial=arimaPolynomials,
                              arPolynomialMatrix=arPolynomialMatrix,
                              maPolynomialMatrix=maPolynomialMatrix,
                              ARIMAIndices=list(nonZeroARI=nonZeroARI,nonZeroMA=nonZeroMA));
    }
    else{
        armaParametersList <- NULL;
        otherReturned <- NULL;
    }

    parametersNumber[1,5] <- sum(parametersNumber[1,])

    # Right down the smoothing parameters
    nCoefficients <- 0;

    modelName <- "SSARIMA";
    if(xregModel){
        modelName[] <- paste0(modelName,"X");
    }
    if(arimaModel){
        # Either the lags are non-seasonal, or there are no orders for seasonal lags
        if(all(lags==1) || (all(arOrders[lags>1]==0) && all(iOrders[lags>1]==0) && all(maOrders[lags>1]==0))){
            modelName[] <- paste0(modelName,"(",arOrders[1],",",iOrders[1],",",maOrders[1],")");
        }
        else{
            for(i in 1:length(arOrders)){
                if(all(arOrders[i]==0) && all(iOrders[i]==0) && all(maOrders[i]==0)){
                    next;
                }
                modelName[] <- paste0(modelName,"(",arOrders[i],",");
                modelName[] <- paste0(modelName,iOrders[i],",");
                modelName[] <- paste0(modelName,maOrders[i],")[",lags[i],"]");
            }
        }
    }
    if(regressors=="adapt"){
        modelName[] <- paste0(modelName,"{D}");
    }
    if(constantRequired){
        modelName[] <- paste0(modelName," with ",constantName);
    }

    if(all(occurrence!=c("n","none"))){
        modelName[] <- paste0("i",modelName);
    }

    parametersNumber[1,5] <- sum(parametersNumber[1,1:4]);
    parametersNumber[2,5] <- sum(parametersNumber[2,1:4]);

    ##### Deal with the holdout sample #####
    if(holdout && h>0){
        errormeasures <- measures(yHoldout,yForecast,yInSample);
    }
    else{
        errormeasures <- NULL;
    }

    # Amend the class of state matrix
    if(any(yClasses=="ts")){
        matVt <- ts(t(matVt), start=(yIndex[1]-(yIndex[2]-yIndex[1])*lagsModelMax), frequency=yFrequency);
    }
    else{
        yStatesIndex <- yInSampleIndex[1] - lagsModelMax*diff(tail(yInSampleIndex,2)) +
            c(1:lagsModelMax-1)*diff(tail(yInSampleIndex,2));
        yStatesIndex <- c(yStatesIndex, yInSampleIndex);
        matVt <- zoo(t(matVt), order.by=yStatesIndex);
    }

    # Transform everything into appropriate classes
    if(any(yClasses=="ts")){
        yInSample <- ts(yInSample,start=yStart, frequency=yFrequency);
        if(holdout){
            yHoldout <- ts(as.matrix(yHoldout), start=yForecastStart, frequency=yFrequency);
        }
    }
    else{
        yInSample <- zoo(yInSample, order.by=yInSampleIndex);
        if(holdout){
            yHoldout <- zoo(as.matrix(yHoldout), order.by=yForecastIndex);
        }
    }

    ##### Return values #####
    modelReturned <- structure(list(model=modelName, timeElapsed=Sys.time()-startTime,
                                    call=cl, orders=orders, lags=lags,
                                    arma=armaParametersList, other=otherReturned,
                                    data=yInSample, holdout=yHoldout, fitted=yFitted, residuals=errors,
                                    forecast=yForecast, states=matVt, accuracy=errormeasures,
                                    profile=profilesRecentTable, profileInitial=profilesRecentInitial,
                                    persistence=vecG[,1], transition=matF,
                                    measurement=matWt, initial=initialValue, initialType=initialType,
                                    constant=constantValue, nParam=parametersNumber,
                                    formula=formula, regressors=regressors,
                                    loss=loss, lossValue=CFValue, lossFunction=lossFunction, logLik=logLikValue,
                                    ICs=setNames(c(AIC(logLikValue), AICc(logLikValue), BIC(logLikValue), BICc(logLikValue)),
                                                 c("AIC","AICc","BIC","BICc")),
                                    distribution=distribution, bounds=bounds,
                                    scale=scale, B=B, lags=lags, lagsAll=lagsModelAll, res=res, FI=FI,
                                    adamCpp=adamCpp),
                               class=c("adam","smooth"));

    # Fix data and holdout if we had explanatory variables
    if(!is.null(xregData) && !is.null(ncol(data))){
        # Remove redundant columns from the data
        modelReturned$data <- data[1:obsInSample,,drop=FALSE];
        if(holdout){
            modelReturned$holdout <- data[obsInSample+c(1:h),,drop=FALSE];
        }
        # Fix the ts class, which is destroyed during subsetting
        if(all(yClasses!="zoo")){
            if(is.data.frame(data)){
                modelReturned$data[,responseName] <- ts(modelReturned$data[,responseName],
                                                        start=yStart, frequency=yFrequency);
                if(holdout){
                    modelReturned$holdout[,responseName] <- ts(modelReturned$holdout[,responseName],
                                                               start=yForecastStart, frequency=yFrequency);
                }
            }
            else{
                modelReturned$data <- ts(modelReturned$data, start=yStart, frequency=yFrequency);
                if(holdout){
                    modelReturned$holdout <- ts(modelReturned$holdout, start=yForecastStart, frequency=yFrequency);
                }
            }
        }
    }


    ##### Make a plot #####
    if(!silent){
        plot(modelReturned, 7)
    }

    return(modelReturned);
}
