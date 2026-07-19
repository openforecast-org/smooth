#' Complex Exponential Smoothing
#'
#' Function estimates CES in state space form with information potential equal
#' to errors and returns several variables.
#'
#' The function estimates Complex Exponential Smoothing in the state space form
#' described in Svetunkov et al. (2022) with the information potential
#' equal to the approximation error.
#'
#' The \code{auto.ces()} function implements the automatic seasonal component
#' selection based on information criteria.
#'
#' \code{ces()} uses two optimisers to get good estimates of parameters. By default
#' these are BOBYQA and then Nelder-Mead. This can be regulated via \code{...} - see
#' details below.
#'
#' For some more information about the model and its implementation, see the
#' vignette: \code{vignette("ces","smooth")}
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
#' @template ssADAMRef
#' @template ssGeneralRef
#' @template ssCESRef
#'
#' @param seasonality The type of seasonality used in CES. Can be: \code{none}
#' - No seasonality; \code{simple} - Simple seasonality, using lagged CES
#' (based on \code{t-m} observation, where \code{m} is the seasonality lag);
#' \code{partial} - Partial seasonality with the real seasonal component
#' (equivalent to additive seasonality); \code{full} - Full seasonality with
#' complex seasonal component (can do both multiplicative and additive
#' seasonality, depending on the data). First letter can be used instead of
#' full words.
#'
#' In case of the \code{auto.ces()} function, this parameter defines which models
#' to try.
#' @param lags Vector of lags to use in the model. Allows defining multiple seasonal models.
#' @param a First complex smoothing parameter. Should be a complex number.
#'
#' NOTE! CES is very sensitive to a and b values so it is advised either to
#' leave them alone, or to use values from previously estimated model.
#' @param b Second complex smoothing parameter. Can be real if
#' \code{seasonality="partial"}. In case of \code{seasonality="full"} must be
#' complex number.
#' @param bounds The type of bounds for the persistence to use in the model
#' estimation. Can be either \code{admissible} - guaranteeing the stability of the
#' model, or \code{none} - no restrictions (potentially dangerous).
#' @param model A previously estimated GUM model, if provided, the function
#' will not estimate anything and will use all its parameters.
#' @param ...  Other non-documented parameters. See \link[smooth]{adam} for
#' details. However, there are several unique parameters passed to the optimiser
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
#' @seealso \code{\link[smooth]{adam}, \link[smooth]{es}}
#'
#' @examples
#' y <- rnorm(100,10,3)
#' ces(y, h=20, holdout=FALSE)
#'
#' y <- 500 - c(1:100)*0.5 + rnorm(100,10,3)
#' ces(y, h=20, holdout=TRUE)
#'
#' ces(BJsales, h=8, holdout=TRUE)
#'
#' \donttest{ces(AirPassengers, h=18, holdout=TRUE, seasonality="s")
#' ces(AirPassengers, h=18, holdout=TRUE, seasonality="p")
#' ces(AirPassengers, h=18, holdout=TRUE, seasonality="f")}

#' @rdname ces
#' @export
ces <- function(y, seasonality=c("none","simple","partial","full"), lags=c(frequency(y)),
                initial=c("backcasting","optimal","two-stage","complete","gradient"), a=NULL, b=NULL,
                loss=c("likelihood","MSE","MAE","HAM","MSEh","TMSE","GTMSE","MSCE","GPL"),
                h=0, holdout=FALSE, bounds=c("admissible","none"), silent=TRUE,
                model=NULL, xreg=NULL, regressors=c("use","select","adapt"), initialX=NULL, ...){
# Function estimates CES in state space form with sigma = error
# and returns complex smoothing parameter value, fitted values,
# residuals, point and interval forecasts, matrix of CES components and values of
# information criteria.
#
#    Copyright (C) 2015 - Inf  Ivan Svetunkov


# Start measuring the time of calculations
    startTime <- Sys.time();
    cl <- match.call();
    # Record the parental environment. Needed for optimal initialisation
    env <- parent.frame();

    ellipsis <- list(...);

    # Check seasonality and loss
    seasonality <- match.arg(seasonality);
    loss <- match.arg(loss);

    # paste0() is needed in order to get rid of potential issues with names
    yName <- paste0(deparse(substitute(y)),collapse="");

    # Assume that the model is not provided
    profilesRecentProvided <- FALSE;
    profilesRecentTable <- NULL;

    # If a previous model provided as a model, write down the variables
    if(!is.null(model)){
        if(is.null(model$model)){
            stop("The provided model is not CES.",call.=FALSE);
        }
        else if(smoothType(model)!="CES"){
            stop("The provided model is not CES.",call.=FALSE);
        }
        # This needs to be fixed to align properly in case of various seasonals
        profilesRecentInitial <- profilesRecentTable <- model$profileInitial;
        profilesRecentProvided[] <- TRUE;
        # This is needed to save initials and to avoid the standard checks
        initialValueProvided <- model$initial;
        initialOriginal <- initial <- model$initialType;
        a <- model$parameters$a;
        b <- model$parameters$b;
        seasonality <- model$seasonality;
        matVt <- t(model$states);
        matWt <- model$measurement;
        matF <- model$transition;
        vecG <- as.matrix(model$persistence);
        ellipsis$B <- coef(model);
        lags <- lags(model);

        model <- model$model;
        model <- NULL;
        modelDo <- modelDoOriginal <- "use";
    }
    else{
        modelDo <- modelDoOriginal <- "estimate";
        initialValueProvided <- NULL;
        initialOriginal <- initial;
    }

    a <- list(value=a);
    b <- list(value=b);

    if(is.null(a$value)){
        a$estimate <- TRUE;
    }
    else{
        a$estimate <- FALSE;
    }
    if(is.null(b$value) && any(seasonality==c("partial","full"))){
        b$estimate <- TRUE;
    }
    else{
        b$estimate <- FALSE;
    }

    if(seasonality=="partial"){
        b$number <- 1;
    }
    else if(seasonality=="full"){
        b$number <- 2;
    }
    else{
        b$number <- 0;
    }

    # Make it look like ANN/ANA/AAA (to get correct lagsModelAll)
    model <- switch(seasonality,
                    "none"="AAN",
                    "partial"="AAA",
                    "ANA");

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

    # Default parameters for the wrapper
    constant <- FALSE;
    distribution <- "dnorm";
    formula <- NULL;
    ic <- "AICc";
    level <- 0.99;
    occurrence <- "none";
    orders <- list(ar=c(0),i=c(0),ma=c(0),select=FALSE);
    outliers <- "ignore";
    persistence <- NULL;
    phi <- NULL;

    ##### Set environment for ssInput and make all the checks #####
    checkerReturn <- parametersChecker(data=data, model, lags, formulaToUse=formula,
                                       orders=orders,
                                       constant=constant, arma=NULL,
                                       outliers=outliers, level=level,
                                       persistence=persistence, phi=phi, initial,
                                       distribution=distribution, loss, h, holdout, occurrence=occurrence,
                                       # This is not needed by the function
                                       ic=ic, bounds=bounds[1],
                                       regressors=regressors, yName=yName,
                                       silent, modelDo, ellipsis, fast=FALSE);
    list2env(checkerReturn, envir=environment());

    # Fix lagsModel and Ttype for CES. This is needed because the function drops duplicate seasonal lags
    # if(seasonality=="simple"){
    #     # Build our own lags, no non-seasonal ones
    #     lagsModelSeasonal <- lags <- rep(lags[lags!=1], each=2);
    #     lagsModelAll <- lagsModel <- matrix(lags,ncol=1);
    # }
    # else if(seasonality=="full"){
    #     # Remove unit lags
    #     lagsModelSeasonal <- lags <- rep(lags[lags!=1], each=2);
    #     # Build our own
    #     lags <- c(1,1,lags);
    #     lagsModelAll <- lagsModel <- matrix(lags,ncol=1);
    # }
    # else{
    #     Ttype <- "N";
    #     model <- "ANN";
    # }

    ##### Elements of CES #####
    filler <- function(B, matVt, matF, vecG, a, b){

        # j is for states in matVt, nCoefficients is for the places in B
        j <- 0;
        nCoefficients <- 0;
        # No seasonality
        if(seasonality!="simple"){
            if(a$estimate){
                matF[1,2] <- B[2]-1;
                matF[2,2] <- 1-B[1];
                vecG[1:2,] <- c(B[1]-B[2],
                                B[1]+B[2]);
                nCoefficients[] <- nCoefficients + 2;
            }
            else{
                matF[1,2] <- Im(a$value)-1;
                matF[2,2] <- 1-Re(a$value);
                vecG[1:2,] <- c(Re(a$value)-Im(a$value),
                                Re(a$value)+Im(a$value));
            }
        }
        # Simple seasonality, lagged CES
        else{
            if(a$estimate){
                for(i in 1:nSeasonal){
                    matF[i*2,i*2] <- 1-B[nCoefficients+i*2-1];
                    matF[i*2-1,i*2] <- B[nCoefficients+i*2]-1;
                    vecG[-c(1,0)+2*i,] <- c(B[nCoefficients+i*2-1]-B[nCoefficients+i*2],
                                    B[nCoefficients+i*2-1]+B[nCoefficients+i*2]);
                }
                nCoefficients[] <- nCoefficients + 2*nSeasonal;
            }
            else{
                for(i in 1:nSeasonal){
                    matF[i*2,i*2] <- 1-Re(a$value[i]);
                    matF[i*2-1,i*2] <- Im(a$value[i])-1;
                    vecG[-c(1,0)+2*i,] <- c(Re(a$value[i])-Im(a$value[i]),
                                             Re(a$value[i])+Im(a$value[i]));
                }
            }
        }

        if(seasonality=="partial"){
            # Partial seasonality with a real part only
            if(b$estimate){
                vecG[2+1:nSeasonal,] <- B[nCoefficients+1:nSeasonal];
                nCoefficients[] <- nCoefficients + nSeasonal;
            }
            else{
                vecG[2+1:nSeasonal,] <- b$value;
            }
        }
        else if(seasonality=="full"){
            # Full seasonality with both real and imaginary parts
            if(b$estimate){
                for(i in 1:nSeasonal){
                    matF[2+i*2,2+i*2] <- 1-B[nCoefficients+i*2-1];
                    matF[2+i*2-1,2+i*2] <- B[nCoefficients+i*2]-1;
                    vecG[2-c(1,0)+2*i,] <- c(B[nCoefficients+i*2-1]-B[nCoefficients+i*2],
                                    B[nCoefficients+i*2-1]+B[nCoefficients+i*2]);
                }
                nCoefficients[] <- nCoefficients + 2*nSeasonal;
            }
            else{
                for(i in 1:nSeasonal){
                    matF[2+i*2,2+i*2] <- 1-Re(b$value[i]);
                    matF[2+i*2-1,2+i*2] <- Im(b$value[i])-1;
                    vecG[2-c(1,0)+2*i,] <- c(Re(b$value[i])-Im(b$value[i]),
                                             Re(b$value[i])+Im(b$value[i]));
                }
            }
        }

        vt <- matVt[,1:lagsModelMax,drop=FALSE];
        if(any(initialType==c("optimal","two-stage"))){
            # Fill in the non-seasonal part
            if(seasonality!="simple"){
                vt[1:2,1:lagsModelMax] <- B[nCoefficients+(1:2)];
                nCoefficients[] <- nCoefficients + 2;
                j[] <- j+2;
            }

            if(any(seasonality==c("simple","full"))){
                for(i in 1:nSeasonal){
                    vt[j+1:2,1:lagsModelSeasonal[i]] <- B[nCoefficients+(1:(2*lagsModelSeasonal[i]))];
                    nCoefficients[] <- nCoefficients + lagsModelSeasonal[i]*2;
                    j[] <- j+2;
                }
            }
            else if(seasonality=="partial"){
                for(i in 1:nSeasonal){
                    vt[j+1,1:lagsModelSeasonal[i]] <- B[nCoefficients+(1:lagsModelSeasonal[i])];
                    nCoefficients[] <- nCoefficients + lagsModelSeasonal[i];
                    j[] <- j+1;
                }
            }
        }
        else if(initialType=="provided"){
            vt[,1:lagsModelMax] <- initialValue;
        }

        # If exogenous are included
        if(xregModel && initialXregEstimate && initialType!="complete"){
            vt[j+(1:xregNumber),] <- B[nCoefficients+(1:xregNumber)];
            nCoefficients[] <- nCoefficients + xregNumber;
        }

        return(list(matF=matF,vecG=vecG,vt=vt));
    }

    creator <- function(seasonality, xregModel,
                        lagsModelAll, lagsModelMax, obsAll, lags, yIndexAll, yClasses,
                        lagsModelSeasonal, nSeasonal,
                        componentsNumber, xregNumber, obsInSample, obsStates, xregNames,
                        yFrequency, xregModelInitials){
        # Create ADAM profiles for correct treatment of seasonality
        adamProfiles <- adamProfileCreator(lagsModelAll, lagsModelMax, obsAll,
                                           lags=lags, yIndex=yIndexAll, yClasses=yClasses);
        profilesRecentTable <- adamProfiles$recent;
        indexLookupTable <- adamProfiles$lookup;

        matF <- diag(componentsNumber+xregNumber);
        matF[2,1] <- 1;
        vecG <- matrix(0,componentsNumber+xregNumber,1);
        matWt <- matrix(1, obsInSample, componentsNumber+xregNumber);
        matWt[,2] <- 0;
        matVt <- matrix(0, componentsNumber+xregNumber, obsStates);
        # Fill something in, we'll amend later
        rownames(matVt) <- rep(" ", componentsNumber+xregNumber);

        if(seasonality!="none"){
            # smoother="global" instead of the historical default "lowess":
            # Python's greybox.lowess does not match R's stats::lowess
            # byte-for-byte (~3e-13 output diff on real data); that ULP-level
            # gap propagates through the CES seasonal seed into different
            # BOBYQA basins on partial / full seasonality. The shared
            # olsCore.h backend behind smoother="global" produces
            # byte-identical decompositions across languages.
            yDecomposedSeasonal <- msdecompose(yInSample, lags=lagsModelSeasonal,
                                                type="additive", smoother="global")$seasonal;
        }

        # Fill in matrices for each of the special cases
        if(seasonality=="full"){
            rownames(matVt)[1:2] <- c("level", "potential")
            if(nSeasonal>1){
                rownames(matVt)[-c(1:2)] <- c(paste0(rep(c("seasonal 1", "seasonal 2"),nSeasonal),"[",
                                                     rep(lagsModelSeasonal,each=2),"]"), xregNames);
                matVt[1,1:lagsModelMax] <- mean(yInSample[1:lagsModelMax]);
                matVt[2,1:lagsModelMax] <- matVt[1,1:lagsModelMax]/1.1;
                for(i in 1:nSeasonal){
                    matF[2+2*i,2+2*i-1] <- 1;
                    matWt[,2+2*i] <- 0;
                    matVt[2+i*2-1,1:lagsModelMax] <- yDecomposedSeasonal[[i]][1:lagsModelMax];
                    matVt[2+i*2,1:lagsModelMax] <- matVt[2+i*2-1,1:lagsModelMax]/1.1;
                }
            }
            else{
                matF[4,3] <- 1;
                matWt[,4] <- 0;
                rownames(matVt) <- c("level", "potential", "seasonal 1", "seasonal 2", xregNames);
                matVt[1,1:lagsModelMax] <- mean(yInSample[1:lagsModelMax]);
                matVt[2,1:lagsModelMax] <- matVt[1,1:lagsModelMax]/1.1;
                matVt[3,1:lagsModelMax] <- yDecomposedSeasonal[[1]][1:lagsModelMax];
                matVt[4,1:lagsModelMax] <- matVt[3,1:lagsModelMax]/1.1;
            }
        }
        else if(seasonality=="partial"){
            rownames(matVt)[1:2] <- c("level", "potential")
            if(nSeasonal>1){
                rownames(matVt)[-c(1:2)] <- c(paste0(rep(c("seasonal"),nSeasonal),"[",
                                                     lagsModelSeasonal,"]"), xregNames);
                matVt[1,1:lagsModelMax] <- mean(yInSample[1:lagsModelMax]);
                matVt[2,1:lagsModelMax] <- matVt[1,1:lagsModelMax]/1.1;
                for(i in 1:nSeasonal){
                    matVt[2+i,1:lagsModelMax] <- yDecomposedSeasonal[[i]][1:lagsModelMax];
                }
            }
            else{
                rownames(matVt) <- c("level", "potential", "seasonal", xregNames);
                matVt[1,1:lagsModelMax] <- mean(yInSample[1:lagsModelMax]);
                matVt[2,1:lagsModelMax] <- matVt[1,1:lagsModelMax]/1.1;
                matVt[3,1:lagsModelMax] <- yDecomposedSeasonal[[1]][1:lagsModelMax];
            }
        }
        else if(seasonality=="simple"){
            if(nSeasonal>1){
                rownames(matVt) <- c(paste0(rep(c("level.s", "potential.s"),nSeasonal),"[",
                                            rep(lagsModelSeasonal,each=2),"]"), xregNames);
                matVt[(1:nSeasonal)*2-1,1:lagsModelMax] <- yInSample[1:lagsModelMax];
                matVt[(1:nSeasonal)*2,1:lagsModelMax] <- matVt[(1:nSeasonal)*2-1,1:lagsModelMax]/1.1;
                for(i in 1:nSeasonal){
                    matF[2*i,2*i-1] <- 1;
                    matWt[,2*i] <- 0;
                }
            }
            else{
                rownames(matVt) <- c("level.s", "potential.s", xregNames);
                matVt[1,1:lagsModelMax] <- yInSample[1:lagsModelMax];
                matVt[2,1:lagsModelMax] <- matVt[1,1:lagsModelMax]/1.1;
            }
        }
        else{
            rownames(matVt) <- c("level", "potential", xregNames);
            matVt[1:componentsNumber,1] <- c(mean(yInSample[1:min(max(10,yFrequency),obsInSample)]),
                                             mean(yInSample[1:min(max(10,yFrequency),obsInSample)])/1.1);
        }

        # Add parameters for the X
        if(xregModel){
            matVt[componentsNumber+c(1:xregNumber),1] <- xregModelInitials[[1]][[1]];
            matWt[,componentsNumber+c(1:xregNumber)] <- xregData[1:obsInSample,];
        }

        return(list(profilesRecentTable=profilesRecentTable,
                    indexLookupTable=indexLookupTable, matF=matF,
                    vecG=vecG, matWt=matWt, matVt=matVt));
    }

    ##### Function returns scale parameter for the provided parameters #####
    scaler <- function(errors, obsInSample){
        return(sqrt(sum(errors^2)/obsInSample));
    }

    ##### Cost function for CES #####
    CF <- function(B, matVt, matF, vecG, a, b){
        # Obtain the elements of CES
        elements <- filler(B, matVt, matF, vecG, a, b);

        if(bounds=="admissible"){
            # Stability / invertibility condition
            eigenValues <- smoothEigens(elements$vecG, elements$matF, matWt,
                                        lagsModelAll, xregModel, obsInSample);
            if(any(eigenValues>1+1E-50)){
                return(1E+100*max(eigenValues));
            }
        }

        matVt[,1:lagsModelMax] <- elements$vt;
        # Write down the initials in the recent profile
        profilesRecentTable[] <- elements$vt;

        adamFitted <- adamCpp$fit(matVt, matWt,
                                  elements$matF, elements$vecG,
                                  indexLookupTable, profilesRecentTable,
                                  yInSample, ot,
                                  any(initialType==c("complete","backcasting","gradient")), nIterations, "n");

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
            adamErrors <- adamCpp$ferrors(adamFitted$states, matWt,
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
    logLikFunction <- function(B, matVt, matF, vecG, a, b){
        return(-CF(B, matVt=matVt, matF=matF, vecG=vecG, a=a, b=b));
    }


    #### ! In order for CES to work on the ADAM engine, pretend that it is ARIMA ####
    # So, componentsNumberETS should be zero and componentsNumberARIMA = componentsNumber

    nSeasonal <- length(lagsModelSeasonal);
    # Create all the necessary matrices and vectors
    componentsNumberARIMA <- componentsNumber <- switch(seasonality,
                                                        "none"=2,
                                                        "simple"=2*nSeasonal,
                                                        "partial"=2+nSeasonal,
                                                        "full"=2+2*nSeasonal);

    componentsNumberETS <- componentsNumberETSSeasonal <- componentsNumberETSNonSeasonal <- 0;

    lagsModelAll <- lagsModel <- matrix(c(switch(seasonality,
                                                 "none"=c(1,1),
                                                 "simple"=rep(lagsModelSeasonal,each=2),
                                                 "partial"=c(1,1,lagsModelSeasonal),
                                                 "full"=c(1,1,rep(lagsModelSeasonal,each=2))),
                                          rep(1, xregNumber)),
                                        ncol=1);

    Stype <- Ttype <- "N";
    model <- "ANN";


    # Create C++ adam class, which will then use fit, forecast etc methods
    adamCpp <- new(adamCore,
                   lagsModelAll, Etype, Ttype, Stype,
                   componentsNumberETSNonSeasonal,
                   componentsNumberETSSeasonal,
                   componentsNumberETS, componentsNumberARIMA,
                   xregNumber, length(lagsModelAll),
                   constantRequired, FALSE);

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

    # Values for occurrence. No longer supported in ces()
    parametersNumber[1,3] <- parametersNumber[2,3] <- 0;
    # Xreg parameters
    parametersNumber[1,2] <- xregNumber + sum(persistenceXreg);
    # Scale value
    parametersNumber[1,4] <- 1;

    # A hack in case the parameters were provided
    modelDo <- modelDoOriginal;

    if(!a$estimate && !b$estimate &&
       (initialType=="complete" || (initialType=="backcasting" && !xregModel))){
        modelDo <- "use";
    }

    #### If we need to estimate the model ####
    if(modelDo=="estimate"){
        cesCreated <- creator(seasonality, xregModel,
                              lagsModelAll, lagsModelMax, obsAll, lags, yIndexAll, yClasses,
                              lagsModelSeasonal, nSeasonal,
                              componentsNumber, xregNumber, obsInSample, obsStates, xregNames,
                              yFrequency, xregModelInitials);

        list2env(cesCreated, environment());
        ##### Check number of observations vs number of max parameters #####
        # if(obsNonzero <= nParamMax){
        #     if(regressors=="select"){
        #         if(obsNonzero <= (nParamMax - nParamExo)){
        #             warning(paste0("Not enough observations for the reasonable fit. Number of parameters is ",
        #                         nParamMax," while the number of observations is ",obsNonzero - nParamExo,"!"),call.=FALSE);
        #             tinySample <- TRUE;
        #         }
        #         else{
        #             warning(paste0("The potential number of exogenous variables is higher than the number of observations. ",
        #                            "This may cause problems in the estimation."),call.=FALSE);
        #         }
        #     }
        #     else{
        #         warning(paste0("Not enough observations for the reasonable fit. Number of parameters is ",
        #                        nParamMax," while the number of observations is ",obsNonzero,"!"),call.=FALSE);
        #         tinySample <- TRUE;
        #     }
        # }
        # else{
        #     tinySample <- FALSE;
        # }

        # If this is tiny sample, use SES instead
        # if(tinySample){
        #     warning("Not enough observations to fit CES. Switching to ETS(A,N,N).",call.=FALSE);
        #     return(es(y,"ANN",initial=initial,loss=loss,
        #               h=h,holdout=holdout,cumulative=cumulative,
        #               interval=interval,level=level,
        #               occurrence=occurrence,
        #               oesmodel=oesmodel,
        #               bounds="u",
        #               silent=silent,
        #               xreg=xreg,regressors=regressors,initialX=initialX,
        #               updateX=updateX,persistenceX=persistenceX,transitionX=transitionX));
        # }

        # Initialisation before the optimiser
        # if(any(initialType=="optimal",a$estimate,b$estimate)){
        initialiser <- function(...){
            B <- NULL;
            # If we don't need to estimate a
            if(a$estimate){
                if(seasonality!="simple"){
                    B <- setNames(c(1.3,1),
                                  c("alpha_0","alpha_1"));
                }
                else{
                    if(nSeasonal>1){
                        B <- c(B,
                               setNames(rep(c(1.3, 1), nSeasonal),
                                        paste0(rep(c("alpha_0","alpha_1"), nSeasonal),
                                               "[",rep(lagsModelSeasonal, each=2), "]")));
                    }
                    else{
                        B <- setNames(c(1.3,1),
                                      c("alpha_0","alpha_1"));
                    }
                }
            }

            if(b$estimate){
                if(seasonality=="partial"){
                    if(nSeasonal>1){
                        B <- c(B, setNames(rep(0.1, nSeasonal), paste0("beta[", lagsModelSeasonal, "]")));
                    }
                    else{
                        B <- c(B, setNames(0.1, "beta"));
                    }
                }
                else{
                    if(nSeasonal>1){
                        B <- c(B,
                               setNames(rep(c(1.3, 1), nSeasonal),
                                        paste0(rep(c("beta_0","beta_1"), nSeasonal),
                                               "[",rep(lagsModelSeasonal, each=2), "]")));
                    }
                    else{
                        B <- c(B, setNames(c(1.3, 1), c("beta_0","beta_1")));
                    }
                }
            }

            if(all(initialType!=c("backcasting","complete","gradient"))){
                # Record the level and potential
                if(seasonality!="simple"){
                    B <- c(B, matVt[1:2,1]);
                }

                # Record seasonal indices
                if(seasonality=="simple"){
                    B <- c(B, matVt[1:(nSeasonal*2),1:lagsModelMax]);
                }
                else if(seasonality=="partial"){
                    B <- c(B, matVt[2+(1:nSeasonal),1:lagsModelMax]);
                }
                else if(seasonality=="full"){
                    B <- c(B, matVt[2+(1:(nSeasonal*2)),1:lagsModelMax]);
                }
            }

            if(xregModel && initialType!="complete"){
                B <- c(B, setNames(matVt[-c(1:componentsNumber),1], xregNames));
            }
            return(B);
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

            # Call for CES with backcasting
            cesBack <- suppressWarnings(eval(clNew, envir=env));
            B <- cesBack$B;
            # Vector of initial estimates of parameters
            if(seasonality!="simple"){
                B <- c(B, cesBack$initial$nonseasonal);
            }
            if(seasonality!="none"){
                BSeasonal <- as.vector(cesBack$initial$seasonal);
                if(seasonality=="partial"){
                    names(BSeasonal) <- paste0("seasonal_", c(1:lagsModelMax));
                }
                else{
                    names(BSeasonal) <- paste0(rep(c("seasonal 1_","seasonal 2_"), times=lagsModelMax),
                                               rep(c(1:lagsModelMax), each=2))
                }
                B <- c(B, BSeasonal);
            }
            if(xregModel){
                B <- c(B, cesBack$initial$xreg);
            }
        }

        if(is.null(B)){
            B <- initialiser();
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
            if(xregModel){
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

        nloptrArgs <- list(matVt=matVt, matF=matF, vecG=vecG, a=a, b=b);

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

        # Identifiable initial-state df of CES: the same count whether the
        # initials are optimised or backcast/complete/gradient. CES has no
        # multi-seasonal shared-frequency redundancy, so it is the plain
        # structural count (matching the "optimal" branch of the nParam table).
        cesInitialCount <- 2*(seasonality!="simple") +
                           lagsModelMax*(seasonality!="none") +
                           lagsModelMax*any(seasonality==c("full","simple"));
        nStatesBackcasting <- 0;
        if(any(initialType==c("backcasting","complete","gradient"))){
            nStatesBackcasting[] <- cesInitialCount;
        }

        # Parameters estimated + variance. The scale is always an estimated
        # parameter (every reported logLik is a concentrated likelihood).
        nParamEstimated <- length(B) + 1 + nStatesBackcasting;
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
        initialValue <- profilesRecentTable;
        initialXregEstimateOriginal <- initialXregEstimate;
        initialXregEstimate <- FALSE;

        # If matF doesn't exist, this must be a new model with all parameters provided
        # So, we need to create the basic matrices.
        if(!exists("matF", inherits=FALSE)){
            cesCreated <- creator(seasonality, xregModel,
                                  lagsModelAll, lagsModelMax, obsAll, lags, yIndexAll, yClasses,
                                  componentsNumber, xregNumber, obsInSample, obsStates, xregNames,
                                  yFrequency, xregModelInitials);

            list2env(cesCreated, environment());
        }

        CFValue <- CF(B, matVt, matF, vecG, a, b);
        res <- NULL;

        # Only variance is estimated
        nParamEstimated <- (loss=="likelihood")*1;

        initialXregEstimate <- initialXregEstimateOriginal;
    }

    # Prepare for fitting
    elements <- filler(B, matVt, matF, vecG, a, b);
    matF <- elements$matF;
    vecG <- elements$vecG;
    matVt[,1:lagsModelMax] <- elements$vt;

    # Write down the initials in the recent profile
    profilesRecentTable[] <- elements$vt;
    if(!profilesRecentProvided){
        profilesRecentInitial <- elements$vt;
    }

    #### Fisher Information ####
    if(FI){
        # Substitute values to get hessian
        if(any(substr(names(B),1,5)=="alpha")){
            a$estimateOriginal <- a$estimate;
            a$estimate <- TRUE;
        }
        if(any(substr(names(B),1,4)=="beta")){
            b$estimateOriginal <- b$estimate;
            b$estimate <- TRUE;
        }
        # initialTypeOriginal <- initialType;
        # initialType <- "optimal";
        if(!is.null(initialValueProvided$xreg) && initialOriginal!="complete"){
            initialXregEstimateOriginal <- initialXregEstimate;
            initialXregEstimate <- TRUE;
        }
        # This is needed to have some likelihood returned in case of boundary situations
        boundsOriginal <- bounds
        bounds <- "none"

        logLikFunction_FI <- function(B) logLikFunction(B, matVt=matVt, matF=matF, vecG=vecG, a=a, b=b);
        FI <- -hessianCpp(logLikFunction_FI, B, h=stepSize);
        colnames(FI) <- rownames(FI) <- names(B);

        if(any(substr(names(B),1,5)=="alpha")){
            a$estimate <- a$estimateOriginal;
        }
        if(any(substr(names(B),1,4)=="beta")){
            b$estimate <- b$estimateOriginal;
        }
        bounds <- boundsOriginal
        if(!is.null(initialValueProvided$xreg) && initialOriginal!="complete"){
            initialXregEstimate <- initialXregEstimateOriginal;
        }
    }
    else{
        FI <- NA;
    }

    # In case of likelihood, we typically have one more parameter to estimate - scale.
    logLikValue <- structure(logLikFunction(B, matVt=matVt, matF=matF, vecG=vecG, a=a, b=b),
                             nobs=obsInSample, df=nParamEstimated, class="logLik");

    adamFitted <- adamCpp$fit(matVt, matWt,
                              matF, vecG,
                              indexLookupTable, profilesRecentTable,
                              yInSample, ot,
                              any(initialType==c("complete","backcasting","gradient")), nIterations, "n");

    errors[] <- adamFitted$errors;
    yFitted[] <- adamFitted$fitted;
    # Write down the recent profile for future use
    profilesRecentTable <- adamFitted$profile;
    matVt[] <- adamFitted$states;
    if(!any(initialType==c("complete","backcasting","gradient"))){
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
        initialValue <- vector("list", 1*(seasonality!="simple") + 1*(seasonality!="none") + xregModel);
        if(seasonality=="none"){
            names(initialValue) <- c("nonseasonal","xreg")[c(TRUE,xregModel)]
            initialValue$nonseasonal <- matVt[1:2,1];
        }
        else if(seasonality=="simple"){
            names(initialValue) <- c("seasonal","xreg")[c(TRUE,xregModel)]
            initialValue$seasonal <- matVt[1:(nSeasonal*2),1:lagsModelMax];
        }
        else{
            names(initialValue) <- c("nonseasonal","seasonal","xreg")[c(TRUE,TRUE,xregModel)]
            initialValue$nonseasonal <- matVt[1:2,1];
            initialValue$seasonal <- matVt[lagsModelAll!=1,1:lagsModelMax];
        }

        # Initials count in the parameter total however they are obtained
        # (optimised or backcast/complete/gradient) -- the same identifiable
        # count either way.
        if(any(initialType==c("optimal","two-stage","backcasting","complete","gradient"))){
            parametersNumber[1,1] <- (parametersNumber[1,1] + 2*(seasonality!="simple") +
                                      lagsModelMax*(seasonality!="none") + lagsModelMax*any(seasonality==c("full","simple")));
        }
    }
    if(xregModel){
        initialValue$xreg <- matVt[componentsNumber+1:xregNumber,1];
    }
    parametersNumber[1,5] <- sum(parametersNumber[1,])

    # Right down the smoothing parameters
    nCoefficients <- 0;
    if(a$estimate){
        if(seasonality!="simple"){
            a$value <- complex(real=B[1],imaginary=B[2]);
            nCoefficients <- 2;
            parametersNumber[1,1] <- parametersNumber[1,1] + 2;
            names(a$value) <- "a0+ia1";
        }
        else{
            a$value <- complex(real=B[nCoefficients+(1:nSeasonal)*2-1], imaginary=B[nCoefficients+(1:nSeasonal)*2]);
            if(nSeasonal>1){
                names(a$value) <- paste0("a0+ia1[",lagsModelSeasonal,"]");
            }
            else{
                names(a$value) <- "a0+ia1";
            }
        }
    }

    if(b$estimate){
        if(seasonality=="partial"){
            b$value <- B[nCoefficients+1:nSeasonal];
            parametersNumber[1,1] <- parametersNumber[1,1] + nSeasonal;
        }
        else if(seasonality=="full"){
            b$value <- complex(real=B[nCoefficients+(1:nSeasonal)*2-1], imaginary=B[nCoefficients+(1:nSeasonal)*2]);
            parametersNumber[1,1] <- parametersNumber[1,1] + nSeasonal*2;
        }
    }
    if(b$number!=0){
        if(is.complex(b$value)){
            if(nSeasonal>1){
                names(b$value) <- paste0("b0+ib1[",lagsModelSeasonal,"]");
            }
            else{
                names(b$value) <- "b0+ib1";
            }
        }
        else{
            if(nSeasonal>1){
                names(b$value) <- paste0("b[",lagsModelSeasonal,"]");
            }
            else{
                names(b$value) <- "b";
            }
        }
    }

    modelname <- "CES";
    if(xregModel){
        modelname[] <- paste0(modelname,"X");
    }
    modelname[] <- paste0(modelname,"(",seasonality,")");

    if(all(occurrence!=c("n","none"))){
        modelname[] <- paste0("i",modelname);
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

    ##### Print output #####
    # if(!silent){
    #     if(any(abs(eigen(matF - vecG %*% matWt, only.values=TRUE)$values)>(1 + 1E-10))){
    #         if(bounds!="a"){
    #             warning("Unstable model was estimated! Use bounds='admissible' to address this issue!",call.=FALSE);
    #         }
    #         else{
    #             warning("Something went wrong in optimiser - unstable model was estimated! Please report this error to the maintainer.",
    #                     call.=FALSE);
    #         }
    #     }
    # }

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
    modelReturned <- structure(list(model=modelname, timeElapsed=Sys.time()-startTime,
                                    call=cl, parameters=list(a=a$value, b=b$value), seasonality=seasonality,
                                    data=yInSample, holdout=yHoldout, fitted=yFitted, residuals=errors,
                                    forecast=yForecast, states=matVt, accuracy=errormeasures,
                                    profile=profilesRecentTable, profileInitial=profilesRecentInitial,
                                    persistence=vecG[,1], transition=matF,
                                    measurement=matWt, initial=initialValue, initialType=initialType,
                                    nParam=parametersNumber,
                                    formula=formula, regressors=regressors,
                                    loss=loss, lossValue=CFValue, lossFunction=lossFunction, logLik=logLikValue,
                                    # ICs=setNames(c(AIC(logLikValue), AICc(logLikValue), BIC(logLikValue), BICc(logLikValue)),
                                    #              c("AIC","AICc","BIC","BICc")),
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
