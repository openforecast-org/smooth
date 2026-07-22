#' Generalised Univariate Model
#'
#' Function constructs Generalised Univariate Model, estimating matrices F, w,
#' vector g and initial parameters.
#'
#' The function estimates the Single Source of Error state space model of the
#' following type:
#'
#' \deqn{y_{t} = w_t' v_{t-l} + \epsilon_{t}}
#'
#' \deqn{v_{t} = F v_{t-l} + g_{t} \epsilon_{t}}
#'
#' where \eqn{v_{t}} is the state vector (defined using \code{orders}) and
#' \eqn{l} is the vector of \code{lags}, \eqn{w_t} is the \code{measurement}
#' vector (which includes fixed elements and explanatory variables),
#' \eqn{F} is the \code{transition} matrix, \eqn{g_t} is the \code{persistence}
#' vector (includes explanatory variables as well if provided), finally,
#' \eqn{\epsilon_{t}} is the error term.
#'
#' For some more information about the model and its implementation, see the
#' vignette: \code{vignette("gum","smooth")}
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
#' @param orders Order of the model. Specified as vector of number of states
#' with different lags. For example, \code{orders=c(1,1)} means that there are
#' two states: one of the first lag type, the second of the second type.
#' In case of \code{auto.gum()}, this parameters is the value of the max order
#' to check.
#' @param lags Defines lags for the corresponding orders. If, for example,
#' \code{orders=c(1,1)} and lags are defined as \code{lags=c(1,12)}, then the
#' model will have two states: the first will have lag 1 and the second will
#' have lag 12. The length of \code{lags} must correspond to the length of
#' \code{orders}. In case of the \code{auto.gum()}, the value of the maximum
#' lag to check. This should usually be a maximum frequency of the data.
#' @param type Type of model. Can either be \code{"additive"} or
#' \code{"multiplicative"}. The latter means that the GUM is fitted on
#' log-transformed data. In case of \code{auto.gum()}, can also be \code{"select"},
#' implying automatic selection of the type.
#' @param persistence Persistence vector \eqn{g}, containing smoothing
#' parameters. If \code{NULL}, then estimated.
#' @param transition Transition matrix \eqn{F}. Can be provided as a vector.
#' Matrix will be formed using the default \code{matrix(transition,nc,nc)},
#' where \code{nc} is the number of components in the state vector. If
#' \code{NULL}, then estimated.
#' @param measurement Measurement vector \eqn{w}. If \code{NULL}, then
#' estimated.
#' @param bounds The type of bounds for the parameters to use in the model
#' estimation. Can be either \code{admissible} - guaranteeing the stability of the
#' model, \code{"usual"} restrict all the parameters with the (0, 1) region,
#' or \code{none} - no restrictions (potentially dangerous).
#' @param model A previously estimated GUM model, if provided, the function
#' will not estimate anything and will use all its parameters.
#' @param ...  Other non-documented parameters. See \link[smooth]{adam} for
#' details.  However, there are several unique parameters passed to the optimiser
#' in comparison with \code{adam}:
#' 1. \code{algorithm0}, which defines what algorithm to use in nloptr for the initial
#' optimisation. By default, this is "NLOPT_LN_BOBYQA".
#' 2. \code{algorithm} determines the second optimiser. By default this is
#' "NLOPT_LN_NELDERMEAD".
#' 3. maxeval0 and maxeval, that determine the number of iterations for the two
#' optimisers. By default, \code{maxeval0=maxeval=40*k}, where
#' k is the number of estimated parameters.
#' 4. xtol_rel0 and xtol_rel, which are 1e-8 and 1e-6 respectively.
#' There are also ftol_rel0, ftol_rel, ftol_abs0 and ftol_abs, which by default
#' are set to values explained in the \code{nloptr.print.options()} function.
#'
#' @return Object of class "adam" is returned with similar elements to the
#' \link[smooth]{adam} function.
#'
#' @seealso \code{\link[smooth]{adam}, \link[smooth]{es}, \link[smooth]{ces}}
#'
#' @examples
#' gum(BJsales, h=8, holdout=TRUE)
#'
#' \donttest{ourModel <- gum(rnorm(118,100,3), orders=c(2,1), lags=c(1,4), h=18, holdout=TRUE)}
#'
#' # Redo previous model on a new data and produce prediction interval
#' \donttest{gum(rnorm(118,100,3), model=ourModel, h=18)}
#'
#' # Produce something crazy with optimal initials (not recommended)
#' \donttest{gum(rnorm(118,100,3), orders=c(1,1,1), lags=c(1,3,5), h=18, holdout=TRUE, initial="o")}
#'
#' # Simpler model estimated using trace forecast error loss function and its analytical analogue
#' \donttest{gum(rnorm(118,100,3), orders=c(1), lags=c(1), h=18, holdout=TRUE, bounds="n", loss="TMSE")}
#'
#' @rdname gum
#' @export
gum <- function(y, orders=c(1,1), lags=c(1,frequency(y)), type=c("additive","multiplicative"),
                initial=c("backcasting","optimal","two-stage","complete","gradient"),
                persistence=NULL, transition=NULL, measurement=rep(1,sum(orders)),
                loss=c("likelihood","MSE","MAE","HAM","MSEh","TMSE","GTMSE","MSCE","GPL"),
                h=0, holdout=FALSE, bounds=c("usual","admissible","none"), silent=TRUE,
                model=NULL, xreg=NULL, regressors=c("use","select","adapt","integrate"), initialX=NULL, ...){
# General Univariate Model function. Paper to follow... at some point... maybe.
#
#    Copyright (C) 2016 - Inf Ivan Svetunkov

# Start measuring the time of calculations
    startTime <- Sys.time();
    cl <- match.call();
    # Record the parental environment. Needed for optimal initialisation
    env <- parent.frame();

    ellipsis <- list(...);

    # Check seasonality and loss
    type <- match.arg(type);
    loss <- match.arg(loss);
    boundsOriginal <- match.arg(bounds);

    # paste0() is needed in order to get rid of potential issues with names
    yName <- paste0(deparse(substitute(data)),collapse="");

    # Assume that the model is not provided
    profilesRecentProvided <- FALSE;
    profilesRecentTable <- NULL;

    # If a previous model provided as a model, write down the variables
    if(!is.null(model)){
        if(is.null(model$model)){
            stop("The provided model is not GUM.",call.=FALSE);
        }
        else if(smoothType(model)!="GUM"){
            stop("The provided model is not GUM.",call.=FALSE);
        }
        # This needs to be fixed to align properly in case of various seasonals
        profilesRecentInitial <- profilesRecentTable <- model$profileInitial;
        profilesRecentProvided[] <- TRUE;
        # This is needed to save initials and to avoid the standard checks
        initialValueProvided <- model$initial;
        initialOriginal <- initial <- model$initialType;
        seasonality <- model$seasonality;
        # matVt <- t(model$states);
        measurement <- model$measurement;
        transition <- model$transition;
        persistenceOriginal <- model$persistence;
        ellipsis$B <- coef(model);
        lags <- lags(model);
        orders <- orders(model);

        model <- model$model;
        model <- NULL;
        modelDo <- modelDoOriginal <- "use";
    }
    else{
        modelDo <- modelDoOriginal <- "estimate";
        initialOriginal <- initial;
        initialValueProvided <- NULL;
        persistenceOriginal <- persistence;
    }

    # If this is Mcomp data, then take the frequency from it
    if(any(class(data)=="Mdata") && all(lags %in% c(1,frequency(data)))){
        lags <- c(1,frequency(data$x));
    }

    orders <- orders[order(lags)];
    lags <- sort(lags);
    # Remove redundant lags (if present)
    lags <- lags[!is.na(orders)];
    # Remove NAs (if lags are longer than orders)
    orders <- orders[!is.na(orders)];

    # GUM is "checked" as ARIMA
    model <- "NNN";
    ordersOriginal <- orders;
    lagsOriginal <- lags;

    # Specific checks for orders and lags
    if(any(is.complex(c(orders,lags)))){
        stop("Complex values? Right! Come on! Be real!", call.=FALSE);
    }
    if(any(c(orders)<0)){
        stop("Funny guy! How am I gonna construct a model with negative orders?", call.=FALSE);
    }
    if(any(c(lags)<0)){
        stop("Right! Why don't you try complex lags then, mister smart guy?", call.=FALSE);
    }

    # If there are zero lags, drop them
    if(any(lags==0)){
        orders <- orders[lags!=0];
        lags <- lags[lags!=0];
    }
    # If zeroes are defined for some orders, drop them.
    if(any(orders==0)){
        lags <- lags[orders!=0];
        orders <- orders[orders!=0];
    }

    # Get rid of duplicates in lags
    if(length(unique(lags))!=length(lags)){
        lagsNew <- unique(lags);
        ordersNew <- lagsNew;
        for(i in 1:length(lagsNew)){
            ordersNew[i] <- max(orders[which(lags==lagsNew[i])]);
        }
        orders <- ordersNew;
        lags <- lagsNew;
    }

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
    }
    if(!is.null(initialX)){
        initial <- list(xreg=initialX);
    }

    # Hack parametersChecker if initial="integrate"
    regressorsIntegrate <- FALSE;
    regressors <- match.arg(regressors);
    if(regressors=="integrate"){
        regressorsIntegrate <- TRUE;
        regressors <- "adapt";
    }

    # Default parameters for the wrapper
    constant <- FALSE;
    distribution <- "dnorm";
    formula <- NULL;
    ic <- "AICc";
    level <- 0.99;
    occurrence <- "none";
    outliers <- "ignore";
    phi <- NULL;

    ##### Set environment for ssInput and make all the checks #####
    checkerReturn <- parametersChecker(data=data, model, lags, formulaToUse=formula,
                                       orders=list(ar=c(orders),i=c(0),ma=c(0),select=FALSE),
                                       constant=constant, arma=NULL,
                                       outliers=outliers, level=level,
                                       persistence=NULL, phi=phi, initial,
                                       distribution=distribution, loss, h, holdout, occurrence=occurrence,
                                       # This is not needed by the gum() function
                                       ic=ic, bounds=bounds[1],
                                       regressors=regressors, yName=yName,
                                       silent, modelDo, ellipsis, fast=FALSE);
    list2env(checkerReturn, envir=environment());

    # Return to the original ones. The parametersChecker will force "admissible"
    bounds <- boundsOriginal

    # Check whether the multiplicative model is applicable
    if(type=="multiplicative"){
        if(any(yInSample<=0)){
            warning("Multiplicative model can only be used on positive data. Switching to the additive one.",
                    call.=FALSE);
            modelIsMultiplicative <- FALSE;
            type <- "additive";
        }
        else{
            yInSample <- log(yInSample);
            modelIsMultiplicative <- TRUE;
        }
    }
    else{
        modelIsMultiplicative <- FALSE;
    }

    ##### Elements of GUM #####
    filler <- function(B, vt, matF, vecG, matWt){

        nCoefficients <- 0;
        if(persistenceEstimate){
            vecG[1:componentsNumberAll,] <- B[nCoefficients+(1:componentsNumberAll)];
            nCoefficients[] <- nCoefficients + componentsNumberAll;
        }

        if(transitionEstimate){
            matF[1:componentsNumberAll,1:componentsNumberAll] <- B[nCoefficients+(1:(componentsNumberAll^2))];
            nCoefficients[] <- nCoefficients + componentsNumberAll^2;
        }

        if(measurementEstimate){
            matWt[,1:componentsNumber] <- B[nCoefficients+(1:componentsNumber)];
            nCoefficients[] <- nCoefficients + componentsNumber;
        }

        if(any(initialType==c("optimal","two-stage"))){
            for(i in 1:componentsNumber){
                vt[i,1:lagsModelMax] <- rep(B[nCoefficients+(1:lagsModel[i])], lagsModelMax)[1:lagsModelMax];
                nCoefficients[] <- nCoefficients + lagsModel[i];
            }
        }

        # In case of backcasting, we still estimate initials of xreg
        if(xregModel && initialXregEstimate && initialType!="complete"){
            vt[componentsNumber+c(1:xregNumber),1:lagsModelMax] <- B[nCoefficients+(1:xregNumber)];
        }

        return(list(matWt=matWt,matF=matF,vecG=vecG,vt=vt));
    }

    ##### Function returns scale parameter for the provided parameters #####
    scaler <- function(errors, obsInSample){
        return(sqrt(sum(errors^2)/obsInSample));
    }

    ##### Cost function for GUM #####
    CF <- function(B, matVt, matF, vecG, matWt){
        # Obtain the elements of GUM
        elements <- filler(B, matVt[,1:lagsModelMax,drop=FALSE], matF, vecG, matWt);

        if(bounds=="admissible"){
            # Stability / invertibility condition
            eigenValues <- smoothEigens(elements$vecG, elements$matF, matWt,
                                        lagsModelAll, xregModel, obsInSample);
            if(any(eigenValues>1+1E-50)){
                return(1E+100*max(eigenValues));
            }
        }
        else if(bounds=="usual"){
            if(any(B>1) || any(B<0)){
                return(1E+100);
            }
        }

        # Write down the initials in the recent profile
        matVt[,1:lagsModelMax] <- profilesRecentTable[] <- elements$vt;

        # Additive SSOE: initial="gradient" profiles the state initials by least
        # squares (falls back to backcasting for a multiplicative GUM).
        adamFitted <- adam_fitOrGradient(matVt, elements$matWt,
                                         elements$matF, elements$vecG,
                                         indexLookupTable, profilesRecentTable,
                                         yInSample, ot, initialType, nIterations, adamCpp,
                                         FALSE, TRUE, xregModel, Etype, "N", "N",
                                         0, 0, 0, lagsModelAll, lagsModelMax,
                                         obsInSample, loss, "dnorm", NULL, 0, FALSE, "n",
                                         componentsNumberARIMA, lagsModelAll);

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
    logLikFunction <- function(B, matVt, matF, vecG, matWt){
        return(-CF(B, matVt=matVt, matF=matF, vecG=vecG, matWt=matWt));
    }

    initialValue <- initialValueProvided;
    initial <- initialOriginal;
    persistence <- persistenceOriginal;

    # Reuse initials if they were provided
    if(!is.null(initialValue)){
        initialType <- "provided";
    }

    orders <- ordersOriginal;
    lags <- lagsOriginal;

    # Fixes to get correct number of components and lags
    # lagsModel is the lags of GUM
    lagsModel <- matrix(rep(lags, times=orders),ncol=1);
    # lagsModelAll is all the lags, GUM+X
    if(xregModel){
        lagsModelAll <- c(lagsModel, rep(1, xregNumber));
    }
    else{
        lagsModelAll <- lagsModel;
    }
    lagsModelMax <- max(lagsModelAll);
    obsStates[] <- obsInSample + lagsModelMax;

    # The reversed lags to fill in values in the state vector
    # lagsModelRev <- lagsModelMax - lagsModel + 1;
    componentsNumberARIMA <- componentsNumber <- sum(orders);
    # Record how many values in the initial state vector need to be estimated
    initialsNumber <- orders %*% lags;

    # componentsNumberAll is used to fill in all matrices
    componentsNumberAll <- componentsNumber
    if(regressorsIntegrate){
        regressors <- "integrate";
        componentsNumberAll <- componentsNumber+xregNumber;
    }

    # Create C++ adam class, which will then use fit, forecast etc methods
    adamCpp <- new(adamCore,
                   lagsModelAll, Etype, Ttype, Stype,
                   componentsNumberETSNonSeasonal,
                   componentsNumberETSSeasonal,
                   componentsNumberETS, componentsNumberARIMA,
                   xregNumber, length(lagsModelAll),
                   constantRequired, FALSE);

    matF <- diag(componentsNumber+xregNumber);
    vecG <- matrix(0,componentsNumber+xregNumber,1);
    if(xregModel){
        rownames(vecG) <- c(paste0("g",1:componentsNumber),
                            paste0("delta",1:xregNumber));
    }
    else{
        rownames(vecG) <- paste0("g",1:componentsNumber);
    }
    matWt <- matrix(1, obsInSample, componentsNumber+xregNumber);
    matVt <- matrix(0, componentsNumber+xregNumber, obsStates,
                    dimnames=list(c(paste0("Component ",1:(componentsNumber)), xregNames), NULL));

    # Fixes for what to estimate
    persistenceEstimate <- is.null(persistence);
    transitionEstimate <- is.null(transition);
    measurementEstimate <- is.null(measurement);
    initialEstimate <- is.null(initialValue);

    # Provided measurement should be just a vector for the dynamic elements
    if(!measurementEstimate){
        matWt[,1:componentsNumber] <- matrix(measurement, obsInSample, componentsNumber, byrow=TRUE);
    }
    if(!transitionEstimate){
        matF[1:componentsNumberAll,1:componentsNumberAll] <- transition;
    }
    if(!persistenceEstimate){
        vecG[1:componentsNumberAll,] <- persistence;
    }
    if(!initialEstimate){
        matVt[1:componentsNumber,1:lagsModelMax] <- initialValue$endogenous;
        if(xregModel){
            matVt[componentsNumber+1:xregNumber,1:lagsModelMax] <- initialValue$xreg;
        }
        initialType <- "provided";
    }
    else{
        # if(initialType!="complete"){
        slope <- (cov(yInSample[1:min(max(12,lagsModelMax),obsInSample),],c(1:min(max(12,lagsModelMax),obsInSample)))/
                      var(c(1:min(max(12,lagsModelMax),obsInSample))));
        intercept <- (sum(yInSample[1:min(max(12,lagsModelMax),obsInSample),])/min(max(12,lagsModelMax),obsInSample) -
                          slope * (sum(c(1:min(max(12,lagsModelMax),obsInSample)))/
                                       min(max(12,lagsModelMax),obsInSample) - 1));

        vtvalues <- vector("numeric", initialsNumber);
        nCoefficients <- 0;
        if(any(lags==1) && length(orders[lags==1])>=1){
            vtvalues[nCoefficients+1] <- intercept;
            nCoefficients[] <- nCoefficients + 1;
        }
        if(any(lags==1) && length(orders[lags==1])>1){
            vtvalues[nCoefficients+1] <- slope;
            nCoefficients[] <- nCoefficients + 1;
        }
        if((initialsNumber)>2){
            # rep is needed to make things work for the small samples
            vtvalues[nCoefficients + 1:(initialsNumber - nCoefficients)] <-
                rep(yInSample[1:min(initialsNumber - nCoefficients,obsInSample),],
                    ceiling(obsInSample/initialsNumber)+1)[1:(initialsNumber - nCoefficients)];
        }

        nCoefficients[] <- 0;
        for(i in 1:componentsNumber){
            matVt[i,1:lagsModel[i]] <- vtvalues[nCoefficients+(1:lagsModel[i])];
            nCoefficients[] <- nCoefficients + lagsModel[i];
        }
        # }
    }

    # Add parameters for the X
    if(xregModel){
        matWt[,componentsNumber+c(1:xregNumber)] <- xregData[1:obsInSample,];
        if(initialXregEstimate && initialType!="complete"){
            matVt[componentsNumber+c(1:xregNumber),1] <- xregModelInitials[[1]][[1]];
        }
    }

    # A hack in case all the parameters were provided
    if(modelDoOriginal=="use"){
        modelDo <- modelDoOriginal;
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

    # Values for occurrence. No longer supported in gum()
    parametersNumber[1,3] <- parametersNumber[2,3] <- 0;
    # Xreg parameters
    parametersNumber[1,2] <- xregNumber + sum(persistenceXreg);
    # Scale value
    parametersNumber[1,4] <- 1;

    #### If we need to estimate the model ####
    if(modelDo=="estimate"){
        # Create ADAM profiles for correct treatment of seasonality
        adamProfiles <- adamProfileCreator(lagsModelAll, lagsModelMax, obsAll,
                                           lags=lags, yIndex=yIndexAll, yClasses=yClasses);
        profilesRecentTable <- adamProfiles$recent;
        indexLookupTable <- adamProfiles$lookup;

        if(is.null(B)){
            B <- vector("numeric", persistenceEstimate*componentsNumberAll +
                            transitionEstimate*componentsNumberAll^2 +
                            measurementEstimate*componentsNumber +
                            initialEstimate*any(initialType==c("optimal","two-stage"))*sum(initialsNumber) +
                            xregNumber*initialXregEstimate*(initialType!="complete"));
            names(B) <- c(paste0("g",1:componentsNumberAll)[persistenceEstimate*(1:componentsNumberAll)],
                          paste0("F",paste0(rep(1:componentsNumberAll,each=componentsNumberAll),
                                            rep(1:componentsNumberAll,times=componentsNumberAll))
                          )[transitionEstimate*(1:(componentsNumberAll^2))],
                          paste0("w",1:componentsNumber)[measurementEstimate*(1:componentsNumber)],
                          paste0("vt",1:sum(initialsNumber))[initialEstimate*any(initialType==c("optimal","two-stage"))*(1:sum(initialsNumber))],
                          xregNames[(1:xregNumber)*initialXregEstimate*(initialType!="complete")]);

            nCoefficients <- 0;
            if(persistenceEstimate){
                B[nCoefficients+1:componentsNumberAll] <- rep(0.1, componentsNumberAll);
                nCoefficients[] <- nCoefficients + componentsNumberAll;
            }

            if(transitionEstimate){
                B[nCoefficients+1:(componentsNumberAll^2)] <- as.numeric(matF[1:componentsNumberAll,1:componentsNumberAll]);
                nCoefficients[] <- nCoefficients + componentsNumberAll^2;
            }

            if(measurementEstimate){
                B[nCoefficients+1:componentsNumber] <- rep(1, componentsNumber);
                nCoefficients[] <- nCoefficients + componentsNumber;
            }

            # In case of optimal, get some initials from backcasting
            if(initialType=="two-stage" && is.null(B)){
                clNew <- cl;
                # If environment is provided, use it
                if(!is.null(ellipsis$environment)){
                    env <- ellipsis$environment;
                }
                # Use complete backcasting
                clNew$initial <- "complete";
                # Shut things up
                clNew$silent <- TRUE;
                # Switch off regressors selection
                if(!is.null(clNew$regressors) && clNew$regressors=="select"){
                    clNew$regressors <- "use";
                }

                # Call for GUM with backcasting
                gumBack <- suppressWarnings(eval(clNew, envir=env));
                B[1:nCoefficients] <- gumBack$B;

                # B <- c(B, gumBack$initial$endogenous);
                for(i in 1:componentsNumber){
                    # B[nCoefficients+(1:lagsModel[i])] <- matVt[i,lagsModelRev[i]:lagsModelMax];
                    B[nCoefficients+(1:lagsModel[i])] <- gumBack$initial$endogenous[i,1:lagsModel[i]];
                    nCoefficients[] <- nCoefficients + lagsModel[i];
                }
            }

            if(xregModel && initialXregEstimate && initialType!="complete"){
                B[nCoefficients+1:xregNumber] <- matVt[-c(1:componentsNumber),lagsModelMax];
            }
        }


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

        # Values for the preliminary optimiser
        if(is.null(ellipsis$algorithm0)){
            algorithm0 <- "NLOPT_LN_BOBYQA";
        }
        else{
            algorithm0 <- ellipsis$algorithm0;
        }
        if(is.null(ellipsis$maxeval0)){
            maxeval0 <- maxevalUsed;
        }
        else{
            maxeval0 <- ellipsis$maxeval0;
        }
        if(is.null(ellipsis$maxtime0)){
            maxtime0 <- -1;
        }
        else{
            maxtime0 <- ellipsis$maxtime0;
        }
        if(is.null(ellipsis$xtol_rel0)){
            xtol_rel0 <- 1e-8;
        }
        else{
            xtol_rel0 <- ellipsis$xtol_rel0;
        }
        if(is.null(ellipsis$xtol_abs0)){
            xtol_abs0 <- 0;
        }
        else{
            xtol_abs0 <- ellipsis$xtol_abs0;
        }
        if(is.null(ellipsis$ftol_rel0)){
            ftol_rel0 <- 0;
        }
        else{
            ftol_rel0 <- ellipsis$ftol_rel0;
        }
        if(is.null(ellipsis$ftol_abs0)){
            ftol_abs0 <- 0;
        }
        else{
            ftol_abs0 <- ellipsis$ftol_abs0;
        }

        nloptrArgs <- list(matVt=matVt, matF=matF, vecG=vecG, matWt=matWt);

        if(length(B)==0){
            # Nothing to estimate (persistence, transition and initials all
            # provided): evaluate the cost once instead of calling nloptr with a
            # zero-length x0, which errors ("x0 must have length > 0"). Mirrors
            # the empty-B guard in om().
            CFValue <- do.call(CF, c(list(B), nloptrArgs));
            res <- NULL;
        }
        else{
            # First run of BOBYQA to get better values of B
            opts <- list(algorithm=algorithm0, xtol_rel=xtol_rel0, xtol_abs=xtol_abs0,
                         ftol_rel=ftol_rel0, ftol_abs=ftol_abs0,
                         maxeval=maxeval0, maxtime=maxtime0, print_level=print_level);
            res <- do.call(nloptr, c(list(x0=B, eval_f=CF, opts=opts), nloptrArgs));
            res$call <- quote(nloptr(x0=B, eval_f=CF, opts=opts));

            if(print_level_hidden>0){
                print(res);
            }

            B[] <- res$solution;

            # Tuning the best obtained values using Nelder-Mead
            opts <- list(algorithm=algorithm, xtol_rel=xtol_rel, xtol_abs=xtol_abs,
                         ftol_rel=ftol_rel, ftol_abs=ftol_abs,
                         maxeval=maxevalUsed, maxtime=maxtime, print_level=print_level);
            res <- suppressWarnings(do.call(nloptr, c(list(x0=B, eval_f=CF, opts=opts), nloptrArgs)));
            res$call <- quote(nloptr(x0=B, eval_f=CF, opts=opts));

            if(print_level_hidden>0){
                print(res);
            }

            B[] <- res$solution;
            CFValue <- res$objective;
        }

        # Identifiable initial-state df of GUM: the number of GUM state initials
        # (= sum of the component lags; xreg is counted separately in [1,2]),
        # the same whether they are optimised or backcast/complete/gradient. GUM
        # has no multi-seasonal shared-frequency redundancy (structural = rank).
        nStatesBackcasting <- 0;
        if(any(initialType==c("backcasting","complete","gradient"))){
            nStatesBackcasting[] <- sum(lagsModelAll) - xregNumber;
        }

        # Parameters estimated + variance. The scale is always an estimated
        # parameter (every reported logLik is a concentrated likelihood).
        nParamEstimated <- length(B) + 1 + nStatesBackcasting;

        # Prepare for fitting
        elements <- filler(B, matVt[,1:lagsModelMax,drop=FALSE], matF, vecG, matWt);
        matF[] <- elements$matF;
        vecG[] <- elements$vecG;
        matVt[,1:lagsModelMax] <- elements$vt;
        matWt[] <- elements$matWt;

        # Write down the initials in the recent profile
        profilesRecentInitial <- profilesRecentTable[] <- matVt[,1:lagsModelMax,drop=FALSE];
        parametersNumber[1,1] <- length(B) + nStatesBackcasting;
    }
    #### If we just use the provided values ####
    else{
        # Create index lookup table
        indexLookupTable <- adamProfileCreator(lagsModelAll, lagsModelMax, obsAll,
                                           lags=lags, yIndex=yIndexAll, yClasses=yClasses)$lookup;
        if(any(initialType==c("optimal","two-stage","provided"))){
            initialType <- "provided";
        }
        else{
            initialType <- initialOriginal[1];
        }
        # initialValue <- profilesRecentInitial;
        initialXregEstimateOriginal <- initialXregEstimate;
        initialXregEstimate <- FALSE;

        CFValue <- CF(B, matVt, matF, vecG, matWt);
        res <- NULL;

        # Nothing is optimised, but backcast / complete / gradient initials are
        # still determined from the data and consume df exactly as in the
        # estimated branch above (line 748-755); the scale is always estimated.
        nStatesBackcasting <- 0;
        if(any(initialType==c("backcasting","complete","gradient"))){
            nStatesBackcasting[] <- sum(lagsModelAll) - xregNumber;
        }
        nParamEstimated <- nStatesBackcasting + 1;
        parametersNumber[1,1] <- nStatesBackcasting;

        initialXregEstimate <- initialXregEstimateOriginal;
    }

    #### Fisher Information ####
    if(FI){
        # Substitute values to get hessian
        if(any(substr(names(B),1,1)=="g")){
            persistenceEstimateOriginal <- persistenceEstimate;
            persistenceEstimate <- TRUE;
        }
        if(any(substr(names(B),1,1)=="F")){
            transitionEstimateOriginal <- transitionEstimate;
            transitionEstimate <- TRUE;
        }
        if(any(substr(names(B),1,1)=="w")){
            measurementEstimateOriginal <- measurementEstimate;
            measurementEstimate <- TRUE;
        }
        # initialTypeOriginal <- initialType;
        # initialType <- "optimal";
        if(!is.null(initialValueProvided$xreg) && initialOriginal!="complete"){
            initialXregEstimateOriginal <- initialXregEstimate;
            initialXregEstimate <- TRUE;
        }

        logLikFunction_FI <- function(B) logLikFunction(B, matVt=matVt, matF=matF, vecG=vecG, matWt=matWt);
        FI <- -hessianCpp(logLikFunction_FI, B, h=stepSize);
        colnames(FI) <- rownames(FI) <- names(B);

        if(any(substr(names(B),1,1)=="g")){
            persistenceEstimate <- persistenceEstimateOriginal;
        }
        if(any(substr(names(B),1,1)=="F")){
            transitionEstimate <- transitionEstimateOriginal;
        }
        if(any(substr(names(B),1,1)=="w")){
            measurementEstimate <- measurementEstimateOriginal;
        }
        # initialType <- initialTypeOriginal;
        if(!is.null(initialValueProvided$xreg) && initialOriginal!="complete"){
            initialXregEstimate <- initialXregEstimateOriginal;
        }
    }
    else{
        FI <- NA;
    }

    # In case of likelihood, we typically have one more parameter to estimate - scale.
    logLikValue <- structure(logLikFunction(B, matVt=matVt, matF=matF, vecG=vecG, matWt=matWt),
                             nobs=obsInSample, df=nParamEstimated, class="logLik");

    adamFitted <- adam_fitOrGradient(matVt, matWt,
                                     matF, vecG,
                                     indexLookupTable, profilesRecentTable,
                                     yInSample, ot, initialType, nIterations, adamCpp,
                                     FALSE, TRUE, xregModel, Etype, "N", "N",
                                     0, 0, 0, lagsModelAll, lagsModelMax,
                                     obsInSample, loss, "dnorm", NULL, 0, FALSE, "n",
                                     componentsNumberARIMA, lagsModelAll);

    errors[] <- adamFitted$errors;
    yFitted[] <- adamFitted$fitted;
    # Write down the recent profile for future use
    profilesRecentTable <- adamFitted$profile;
    matVt[] <- adamFitted$states;
    if(!any(initialType==c("complete","backcasting"))){
        profilesRecentInitial <- matVt[,1:lagsModelMax,drop=FALSE];
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
        initialValue <- list(endogenous=matVt[1:componentsNumber,1:lagsModelMax,drop=FALSE]);
        # initialValue <- vector("list", 1*(seasonality!="simple") + 1*(seasonality!="none") + xregModel);
        # if(seasonality=="none"){
        #     names(initialValue) <- c("nonseasonal","xreg")[c(TRUE,xregModel)]
        #     initialValue$nonseasonal <- matVt[1:2,1];
        # }
        # else if(seasonality=="simple"){
        #     names(initialValue) <- c("seasonal","xreg")[c(TRUE,xregModel)]
        #     initialValue$seasonal <- matVt[1:2,1:lagsModelMax];
        # }
        # else{
        #     names(initialValue) <- c("nonseasonal","seasonal","xreg")[c(TRUE,TRUE,xregModel)]
        #     initialValue$nonseasonal <- matVt[1:2,1];
        #     initialValue$seasonal <- matVt[lagsModelAll!=1,1:lagsModelMax];
        # }

        # if(initialType=="optimal"){
        #     parametersNumber[1,1] <- (parametersNumber[1,1] + initialsNumber);
        # }
    }
    if(xregModel){
        initialValue$xreg <- matVt[componentsNumber+1:xregNumber,1];
    }
    parametersNumber[1,5] <- sum(parametersNumber[1,])

    # Right down the smoothing parameters
    nCoefficients <- 0;

    modelname <- "GUM";
    if(type=="multiplicative"){
        modelname[] <- paste0("log",modelname);
    }
    if(xregModel){
        modelname[] <- paste0(modelname,"X");
    }
    modelname[] <- paste0(modelname,"(",paste(orders,"[",lags,"]",collapse=",",sep=""),")");

    if(all(occurrence!=c("n","none"))){
        modelname[] <- paste0("i",modelname);
    }

    parametersNumber[1,5] <- sum(parametersNumber[1,1:4]);
    parametersNumber[2,5] <- sum(parametersNumber[2,1:4]);

    ##### Prepare objects to return #####

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

    # If this was a log-models, fix it
    if(modelIsMultiplicative){
        logLikValue[] <- logLikValue - sum(yInSample);
        yInSample[] <- exp(yInSample);
        yFitted[] <- exp(yFitted);
        yForecast[] <- exp(yForecast);
    }

    if(holdout && h>0){
        errormeasures <- measures(yHoldout,yForecast,yInSample);
    }
    else{
        errormeasures <- NULL;
    }

    ##### Return values #####
    modelReturned <- structure(list(model=modelname, timeElapsed=Sys.time()-startTime,
                                    call=cl, orders=orders, lags=lags, type=type,
                                    data=yInSample, holdout=yHoldout, fitted=yFitted, residuals=errors,
                                    forecast=yForecast, states=matVt, accuracy=errormeasures,
                                    profile=profilesRecentTable, profileInitial=profilesRecentInitial,
                                    persistence=vecG[,1], transition=matF,
                                    measurement=matWt, initial=initialValue, initialType=initialType,
                                    nParam=parametersNumber,
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

    if(!silent){
        plot(modelReturned, 7)
    }

    return(modelReturned);
}
