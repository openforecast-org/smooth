#### Refitter and reforecaster ####
#' Reapply the model with randomly generated initial parameters and produce forecasts
#'
#' \code{reapply} function generates the parameters based on the values in the provided
#' object and then reapplies the same model with those parameters to the data, getting
#' the fitted paths and updated states. \code{reforecast} function uses those values
#' in order to produce forecasts for the \code{h} steps ahead.
#'
#' The main motivation of the function is to take the randomness due to the in-sample
#' estimation of parameters into account when fitting the model and to propagate
#' this randomness to the forecasts. The methods can be considered as a special case
#' of recursive bootstrap.
#'
#' @template ssAuthor
#' @template ssKeywords
#'
#' @param object Model estimated using one of the functions of smooth package.
#' @param nsim Number of paths to generate (number of simulations to do).
#' @param h Forecast horizon.
#' @param newdata The new data needed in order to produce forecasts.
#' @param type The type of covariance matrix of parameters to use for the
#' refitting. One of \code{"opg"} (the default, the OPG / BHHH estimator
#' \eqn{J=\sum_t s_t s_t'} built from the per-observation scores, positive
#' semi-definite by construction), \code{"hessian"} (the observed Fisher
#' Information, i.e. the inverse numerical Hessian of the log-likelihood) or
#' \code{"bootstrap"}. See \link[stats]{vcov} on the fitted model for details.
#' @param bootstrap Deprecated in favour of \code{type="bootstrap"}. The logical,
#' which determines, whether to use bootstrap for the covariance matrix of
#' parameters or not. \code{bootstrap=TRUE} still works but issues a warning.
#' @param heuristics The value for proportion to use for heuristic estimation of the
#' standard deviation of parameters. If \code{NULL}, it is not used.
#' @param occurrence The vector containing the future occurrence variable
#' (values in [0,1]), if it is known.
#' @param interval What type of mechanism to use for interval construction. The options
#' include \code{interval="none"}, \code{interval="prediction"} (prediction intervals)
#' and \code{interval="confidence"} (intervals for the point forecast). The other options
#' are not supported and do not make much sense for the refitted model.
#' @param level Confidence level. Defines width of prediction interval.
#' @param side Defines, whether to provide \code{"both"} sides of prediction
#' interval or only \code{"upper"}, or \code{"lower"}.
#' @param cumulative If \code{TRUE}, then the cumulative forecast and prediction
#' interval are produced instead of the normal ones. This is useful for
#' inventory control systems.
#' @param ... Other parameters passed to \code{reapply()} and \code{mean()} functions in case of
#' \code{reforecast} (\code{trim} parameter in \code{mean()} is set to
#' 0.01 by default) and to \code{vcov} in case of \code{reapply}.
#' @return \code{reapply()} returns object of the class "reapply", which contains:
#' \itemize{
#' \item \code{timeElapsed} - Time elapsed for the code execution;
#' \item \code{y} - The actual values;
#' \item \code{states} - The array of states of the model;
#' \item \code{refitted} - The matrix with fitted values, where columns correspond
#' to different paths;
#' \item \code{fitted} - The vector of fitted values (conditional mean);
#' \item \code{model} - The name of the constructed model;
#' \item \code{transition} - The array of transition matrices;
#' \item \code{measurement} - The array of measurement matrices;
#' \item \code{persistence} - The matrix of persistence vectors (paths in columns);
#' \item \code{profile} - The array of profiles obtained by the end of each fit.
#' }
#'
#' \code{reforecast()} returns the object of the class \link[smooth]{forecast.smooth},
#' which contains in addition to the standard list the variable \code{paths} - all
#' simulated trajectories with h in rows, simulated future paths for each state in
#' columns and different states (obtained from \code{reapply()} function) in the
#' third dimension.
#'
#' @references \itemize{
#' \item Berndt, E., Hall, B., Hall, R., & Hausman, J. (1974). Estimation and
#' Inference in Nonlinear Structural Models. Annals of Economic and Social
#' Measurement, 3(4), 653--665. (The OPG / BHHH covariance estimator used by
#' \code{type="opg"}.)
#' }
#' @seealso \link[smooth]{forecast.smooth}
#' @examples
#'
#' x <- rnorm(100,0,1)
#'
#' # Just as example. orders and lags do not return anything for ces() and es(). But modelType() does.
#' ourModel <- adam(x, "ANN")
#' refittedModel <- reapply(ourModel, nsim=50)
#' plot(refittedModel)
#'
#' ourForecast <- reforecast(ourModel, nsim=50)
#'
#' @rdname reapply
#' @export reapply
reapply <- function(object, nsim=1000, type=c("opg","hessian","bootstrap"),
                    bootstrap=FALSE, heuristics=NULL, ...) UseMethod("reapply")

#' @export
reapply.default <- function(object, nsim=1000, type=c("opg","hessian","bootstrap"),
                            bootstrap=FALSE, heuristics=NULL, ...){
    warning(paste0("The method is not implemented for the object of the class ",class(object)[1]),
            call.=FALSE);
    return(structure(list(states=object$states, fitted=fitted(object)),
                     class="reapply"));
}

#' @importFrom MASS mvrnorm
#' @export
reapply.adam <- function(object, nsim=1000, type=c("opg","hessian","bootstrap"),
                         bootstrap=FALSE, heuristics=NULL, ...){
    type <- covarTypeResolver(type, bootstrap);
    # Start measuring the time of calculations
    startTime <- Sys.time();
    parametersNames <- names(coef(object));

    # Check whether we deal with adam ETS or the conventional
    adamETS <- adamETSChecker(object);

    vcovAdam <- suppressWarnings(vcov(object, type=type, heuristics=heuristics, nsim=nsim, ...));
    # The OPG covariance (the default) returns an infinite variance for a
    # parameter the data does not identify (e.g. an initial that washes out when
    # its smoothing parameter is at a bound). Such parameters cannot be
    # resampled, so hold them at their point estimate — zero their row/column —
    # before the multivariate-normal draw.
    nonFiniteVar <- !is.finite(diag(vcovAdam));
    if(any(nonFiniteVar)){
        warning(paste0("The covariance of ",
                       paste0(parametersNames[nonFiniteVar], collapse=", "),
                       " is not finite (the data does not identify these), ",
                       "so they are held at their estimates in the refit."),
                call.=FALSE, immediate.=TRUE);
        vcovAdam[nonFiniteVar,] <- 0;
        vcovAdam[,nonFiniteVar] <- 0;
    }
    # Check if the matrix is positive definite
    vcovEigen <- min(eigen(vcovAdam, only.values=TRUE)$values);
    if(vcovEigen<0){
        if(vcovEigen>-1){
            warning(paste0("The covariance matrix of parameters is not positive semi-definite. ",
                           "I will try fixing this, but it might make sense re-estimating the model, tuning the optimiser."),
                    call.=FALSE, immediate.=TRUE);
            # Tune the thing a bit - one of simple ways to fix the issue
            epsilon <- -vcovEigen+1e-10;
            vcovAdam[] <- vcovAdam + epsilon*diag(nrow(vcovAdam));
        }
        else{
            warning(paste0("The covariance matrix of parameters is not positive semi-definite. ",
                           "I cannot fix it, so I will use the diagonal only. ",
                           "It makes sense to re-estimate the model, tuning the optimiser. ",
                           "For example, try reoptimising providing initial vector of parameters 'B=object$B'."),
                    call.=FALSE, immediate.=TRUE);
            vcovAdam[] <- diag(diag(vcovAdam));
        }
    }

    # All the variables needed in the refitter
    yInSample <- actuals(object);
    yClasses <- class(yInSample);
    parametersNumber <- length(parametersNames);
    obsInSample <- nobs(object);
    Etype <- errorType(object);
    Ttype <- trendType(object);
    Stype <- seasonType(object);
    lags <- lags(object);
    lagsSeasonal <- lags[lags!=1];
    nSeasonal <- length(lagsSeasonal);
    lagsModelAll <- modelLags(object);
    lagsModelMax <- max(lagsModelAll);
    persistence <- as.matrix(object$persistence);
    # If there is xreg, but no deltas, increase persistence by including zeroes
    # This can be considered as a failsafe mechanism
    if(ncol(object$data)>1 && !any(substr(names(object$persistence),1,5)=="delta")){
        persistence <- rbind(persistence,matrix(rep(0,sum(object$nParam[,2])),ncol=1));
    }

    cesModel <- cesChecker(object);
    etsModel <- etsChecker(object);
    arimaModel <- arimaChecker(object);
    gumModel <- gumChecker(object);
    ssarimaModel <- ssarimaChecker(object);

    # Get componentsNumberETS, seasonal and componentsNumberARIMA
    componentsDefined <- componentsDefiner(object);
    componentsNumberETS <- componentsDefined$componentsNumberETS;
    componentsNumberETSSeasonal <- componentsDefined$componentsNumberETSSeasonal;
    componentsNumberETSNonSeasonal <- componentsDefined$componentsNumberETSNonSeasonal;
    componentsNumberARIMA <- componentsDefined$componentsNumberARIMA;
    constantRequired <- componentsDefined$constantRequired;

    # Prepare variables for xreg
    if(!is.null(object$initial$xreg)){
        xregModel <- TRUE;

        #### Create xreg vectors ####
        xreg <- object$data;
        formula <- formula(object)
        responseName <- all.vars(formula)[1];
        # Robustify the names of variables
        colnames(xreg) <- make.names(colnames(xreg),unique=TRUE);
        # The names of the original variables
        xregNamesOriginal <- all.vars(formula)[-1];
        # Levels for the factors
        xregFactorsLevels <- lapply(xreg,levels);
        xregFactorsLevels[[responseName]] <- NULL;
        # Expand the variables. We cannot use alm, because it is based on obsInSample
        xregData <- model.frame(formula,data=as.data.frame(xreg));
        # Binary, flagging factors in the data
        xregFactors <- (attr(terms(xregData),"dataClasses")=="factor")[-1];
        # Get the names from the standard model.matrix
        xregNames <- colnames(model.matrix(xregData,data=xregData));
        interceptIsPresent <- FALSE;
        if(any(xregNames=="(Intercept)")){
            interceptIsPresent[] <- TRUE;
            xregNames <- xregNames[xregNames!="(Intercept)"];
        }
        # Expanded stuff with all levels for factors
        if(any(xregFactors)){
            xregModelMatrix <- model.matrix(xregData,xregData,
                                            contrasts.arg=lapply(xregData[attr(terms(xregData),"dataClasses")=="factor"],
                                                                 contrasts, contrasts=FALSE));
            xregNamesModified <- colnames(xregModelMatrix)[-1];
        }
        else{
            xregModelMatrix <- model.matrix(xregData,data=xregData);
            xregNamesModified <- xregNames;
        }
        xregData <- as.matrix(xregModelMatrix);
        # Remove intercept
        if(interceptIsPresent){
            xregData <- xregData[,-1,drop=FALSE];
        }
        xregNumber <- ncol(xregData);

        # The indices of the original parameters
        xregParametersMissing <- setNames(vector("numeric",xregNumber),xregNamesModified);
        # # The indices of the original parameters
        xregParametersIncluded <- setNames(vector("numeric",xregNumber),xregNamesModified);
        # The vector, marking the same values of smoothing parameters
        if(interceptIsPresent){
            xregParametersPersistence <- setNames(attr(xregModelMatrix,"assign")[-1],xregNamesModified);
        }
        else{
            xregParametersPersistence <- setNames(attr(xregModelMatrix,"assign"),xregNamesModified);
        }

        # If there are factors not in the alm data, create additional initials
        if(any(!(xregNamesModified %in% xregNames))){
            xregAbsent <- !(xregNamesModified %in% xregNames);
            # Go through new names and find, where they came from. Then get the missing parameters
            for(i in which(xregAbsent)){
                # Find the name of the original variable
                # Use only the last value... hoping that the names like x and x1 are not used.
                xregNameFound <- tail(names(sapply(xregNamesOriginal,grepl,xregNamesModified[i])),1);
                # Get the indices of all k-1 levels
                xregParametersIncluded[xregNames[xregNames %in% paste0(xregNameFound,
                                                                       xregFactorsLevels[[xregNameFound]])]] <- i;
                # Get the index of the absent one
                xregParametersMissing[i] <- i;
            }
            # Write down the new parameters
            xregNames <- xregNamesModified;
        }
        # The vector of parameters that should be estimated (numeric + original levels of factors)
        xregParametersEstimated <- xregParametersIncluded
        xregParametersEstimated[xregParametersEstimated!=0] <- 1;
        xregParametersEstimated[xregParametersMissing==0 & xregParametersIncluded==0] <- 1;
    }
    else{
        xregModel <- FALSE;
        xregNumber <- 0;
        xregParametersMissing <- 0;
        xregParametersIncluded <- 0;
        xregParametersEstimated <- 0;
        xregParametersPersistence <- 0;
    }
    indexLookupTable <- adamProfileCreator(lagsModelAll, lagsModelMax, obsInSample)$lookup;

    # Create C++ adam class
    adamCpp <- new(adamCore,
                   lagsModelAll, Etype, Ttype, Stype,
                   componentsNumberETSNonSeasonal,
                   componentsNumberETSSeasonal,
                   componentsNumberETS, componentsNumberARIMA,
                   xregNumber, length(lagsModelAll),
                   constantRequired, adamETS);
    # Drift flips sign in the backcasting backward pass when the total order
    # of differencing is odd — the ARIMA analog of the ETS trend reversal
    adamCpp$flipConstant <- constantRequired && !is.null(object$orders) &&
        (sum(object$orders$i) %% 2 == 1);

    # Generate the data from the multivariate normal
    randomParameters <- mvrnorm(nsim, coef(object), vcovAdam);

    #### Rectify the random values for smoothing parameters ####
    if(etsModel){
        # Usual bounds
        if(object$bounds=="usual"){
            # Set the bounds for alpha
            if(any(parametersNames=="alpha")){
                randomParameters[randomParameters[,"alpha"]<0,"alpha"] <- 0;
                randomParameters[randomParameters[,"alpha"]>1,"alpha"] <- 1;
            }
            # Set the bounds for beta
            if(any(parametersNames=="beta")){
                randomParameters[randomParameters[,"beta"]<0,"beta"] <- 0;
                randomParameters[randomParameters[,"beta"]>randomParameters[,"alpha"],"beta"] <-
                    randomParameters[randomParameters[,"beta"]>randomParameters[,"alpha"],"alpha"];
            }
            # Set the bounds for gamma
            if(any(substr(parametersNames,1,5)=="gamma")){
                gammas <- which(substr(colnames(randomParameters),1,5)=="gamma");
                for(i in 1:length(gammas)){
                    randomParameters[randomParameters[,gammas[i]]<0,gammas[i]] <- 0;
                    randomParameters[randomParameters[,gammas[i]]>randomParameters[,"alpha"],
                                     gammas[i]] <- 1-
                        randomParameters[randomParameters[,gammas[i]]>randomParameters[,"alpha"],"alpha"];
                }
            }
            # Set the bounds for phi
            if(any(parametersNames=="phi")){
                randomParameters[randomParameters[,"phi"]<0,"phi"] <- 0;
                randomParameters[randomParameters[,"phi"]>1,"phi"] <- 1;
            }
        }
        # Admissible bounds
        else if(object$bounds=="admissible"){
            # Check, if there is alpha
            if(any(parametersNames=="alpha")){
                alphaBounds <- eigenBounds(object, persistence,
                                           variableNumber=which(names(object$persistence)=="alpha"));
                randomParameters[randomParameters[,"alpha"]<alphaBounds[1],"alpha"] <- alphaBounds[1];
                randomParameters[randomParameters[,"alpha"]>alphaBounds[2],"alpha"] <- alphaBounds[2];
            }
            # Check, if there is beta
            if(any(parametersNames=="beta")){
                betaBounds <- eigenBounds(object, persistence,
                                          variableNumber=which(names(object$persistence)=="beta"));
                randomParameters[randomParameters[,"beta"]<betaBounds[1],"beta"] <- betaBounds[1];
                randomParameters[randomParameters[,"beta"]>betaBounds[2],"beta"] <- betaBounds[2];
            }
            # Check, if there are gammas
            if(any(substr(parametersNames,1,5)=="gamma")){
                gammas <- which(substr(parametersNames,1,5)=="gamma");
                for(i in 1:length(gammas)){
                    gammaBounds <- eigenBounds(object, persistence,
                                               variableNumber=which(substr(names(object$persistence),1,5)=="gamma")[i]);
                    randomParameters[randomParameters[,gammas[i]]<gammaBounds[1],gammas[i]] <- gammaBounds[1];
                    randomParameters[randomParameters[,gammas[i]]>gammaBounds[2],gammas[i]] <- gammaBounds[2];
                }
            }
            # Check, if there are deltas (for xreg)
            # if(any(substr(parametersNames,1,5)=="delta")){
            #     deltas <- which(substr(parametersNames,1,5)=="delta");
            #     for(i in 1:length(deltas)){
            #         deltaBounds <- eigenBounds(object, persistence,
            #                                    variableNumber=which(substr(names(object$persistence),1,5)=="delta")[i]);
            #         randomParameters[randomParameters[,deltas[i]]<deltaBounds[1],deltas[i]] <- deltaBounds[1];
            #         randomParameters[randomParameters[,deltas[i]]>deltaBounds[2],deltas[i]] <- deltaBounds[2];
            #     }
            # }
        }

        # States
        # Set the bounds for trend
        if(Ttype=="M" && any(parametersNames=="trend")){
            randomParameters[randomParameters[,"trend"]<0,"trend"] <- 1e-6;
        }
        # Seasonality
        if(Stype=="M" && any(substr(parametersNames,1,8)=="seasonal")){
            seasonals <- which(substr(parametersNames,1,8)=="seasonal");
            for(i in seasonals){
                randomParameters[randomParameters[,i]<0,i] <- 1e-6;
            }
        }
    }

    if(cesModel){
        alpha0 <- which(substr(parametersNames,1,7)=="alpha_0");
        alpha1 <- which(substr(parametersNames,1,7)=="alpha_1");
        beta <- which(substr(parametersNames,1,4)=="beta");
        beta0 <- which(substr(parametersNames,1,6)=="beta_0");
        beta1 <- which(substr(parametersNames,1,6)=="beta_1");

        # Check, if there is alpha_0
        if(length(alpha0)>0){
            for(i in 1:length(alpha0)){
                alphaBounds <- eigenBounds(object, persistence,
                                           variableNumber=alpha0[i]);
                randomParameters[randomParameters[,alpha0[i]]<alphaBounds[1],alpha0[i]] <- alphaBounds[1];
                randomParameters[randomParameters[,alpha0[i]]>alphaBounds[2],alpha0[i]] <- alphaBounds[2];
            }
        }
        # Check, if there is alpha_1
        if(length(alpha1)>0){
            for(i in 1:length(alpha1)){
                alphaBounds <- eigenBounds(object, persistence,
                                           variableNumber=alpha1[i]);
                randomParameters[randomParameters[,alpha1[i]]<alphaBounds[1],alpha1[i]] <- alphaBounds[1];
                randomParameters[randomParameters[,alpha1[i]]>alphaBounds[2],alpha1[i]] <- alphaBounds[2];
            }
        }
        # Check, if there is a unique beta from partial seasonal model
        betaUnique <- beta[!(beta %in% beta0) & !(beta %in% beta1)];
        if(length(betaUnique)>0){
            for(i in 1:length(betaUnique)){
                alphaBounds <- eigenBounds(object, persistence,
                                           variableNumber=betaUnique[i]);
                randomParameters[randomParameters[,betaUnique[i]]<alphaBounds[1],betaUnique[i]] <- alphaBounds[1];
                randomParameters[randomParameters[,betaUnique[i]]>alphaBounds[2],betaUnique[i]] <- alphaBounds[2];
            }
        }
        # Check, if there is alpha_0
        if(length(beta0)>0){
            for(i in 1:length(beta0)){
                alphaBounds <- eigenBounds(object, persistence,
                                           variableNumber=beta0[i]);
                randomParameters[randomParameters[,beta0[i]]<alphaBounds[1],beta0[i]] <- alphaBounds[1];
                randomParameters[randomParameters[,beta0[i]]>alphaBounds[2],beta0[i]] <- alphaBounds[2];
            }
        }
        # Check, if there is alpha_1
        if(length(beta1)>0){
            for(i in 1:length(beta1)){
                alphaBounds <- eigenBounds(object, persistence,
                                           variableNumber=beta1[i]);
                randomParameters[randomParameters[,beta1[i]]<alphaBounds[1],beta1[i]] <- alphaBounds[1];
                randomParameters[randomParameters[,beta1[i]]>alphaBounds[2],beta1[i]] <- alphaBounds[2];
            }
        }
    }

    # Correct the bounds for the ARIMA model
    if(arimaModel){
        #### Deal with ARIMA parameters ####
        ariPolynomial <- object$other$polynomial$ariPolynomial;
        arPolynomial <- object$other$polynomial$arPolynomial;
        maPolynomial <- object$other$polynomial$maPolynomial;
        nonZeroARI <- object$other$ARIMAIndices$nonZeroARI;
        nonZeroMA <- object$other$ARIMAIndices$nonZeroMA;
        arPolynomialMatrix <- object$other$arPolynomialMatrix;
        # Locate all thetas for ARIMA
        thetas <- which(substr(parametersNames,1,5)=="theta");
        # Locate phi for ARIMA (they are always phi1, phi2 etc)
        phis <- which((substr(parametersNames,1,3)=="phi") & (nchar(parametersNames)>3));
        # Do loop for thetas
        if(length(thetas)>0){
            # MA parameters
            for(i in 1:length(thetas)){
                psiBounds <- eigenBounds(object, persistence,
                                         variableNumber=which(substr(names(object$persistence),1,3)=="psi")[nonZeroMA[i,2]]);
                # If there are ARI elements in persistence, subtract (-(-x)) them to get proper bounds
                if(any(nonZeroARI[,2]==i)){
                    ariIndex <- which(nonZeroARI[,2]==i);
                    randomParameters[randomParameters[,thetas[i]]-ariPolynomial[nonZeroARI[ariIndex,1]]<psiBounds[1],thetas[i]] <-
                        psiBounds[1]+ariPolynomial[nonZeroARI[ariIndex,1]];
                    randomParameters[randomParameters[,thetas[i]]-ariPolynomial[nonZeroARI[ariIndex,1]]>psiBounds[2],thetas[i]] <-
                        psiBounds[2]+ariPolynomial[nonZeroARI[ariIndex,1]];
                }
                else{
                    randomParameters[randomParameters[,thetas[i]]<psiBounds[1],thetas[i]] <- psiBounds[1];
                    randomParameters[randomParameters[,thetas[i]]>psiBounds[2],thetas[i]] <- psiBounds[2];
                }
            }
        }
        # Locate phi for ARIMA (they are always phi1, phi2 etc)
        if(length(phis)>0){
            # AR parameters
            for(i in 1:length(phis)){
                # Get bounds for AR based on stationarity condition
                phiBounds <- arPolinomialsBounds(arPolynomialMatrix, arPolynomial,
                                                 which(arPolynomial==arPolynomial[arPolynomial!=0][-1][i]));

                randomParameters[randomParameters[,phis[i]]<phiBounds[1],phis[i]] <- phiBounds[1];
                randomParameters[randomParameters[,phis[i]]>phiBounds[2],phis[i]] <- phiBounds[2];
            }
        }
    }

    # Set the bounds for deltas
    if(any(substr(parametersNames,1,5)=="delta")){
        deltas <- which(substr(colnames(randomParameters),1,5)=="delta");
        randomParameters[,deltas][randomParameters[,deltas]<0] <- 0;
        randomParameters[,deltas][randomParameters[,deltas]>1] <- 1;
    }

    #### Prepare the necessary matrices ####
    # States are defined similar to how it is done in adam.
    # Inserting the existing one is needed in order to deal with the case, when one of the initials was provided
    arrVt <- array(t(object$states),c(ncol(object$states),nrow(object$states),nsim),
                   dimnames=list(colnames(object$states),NULL,paste0("nsim",c(1:nsim))));
    # Set the proper time stamps for the fitted
    if(any(yClasses=="zoo")){
        fittedMatrix <- zoo(array(NA,c(obsInSample,nsim),
                                  dimnames=list(NULL,paste0("nsim",c(1:nsim)))),
                            order.by=time(yInSample));
    }
    else{
        fittedMatrix <- ts(array(NA,c(obsInSample,nsim),
                                 dimnames=list(NULL,paste0("nsim",c(1:nsim)))),
                           start=start(yInSample), frequency=frequency(yInSample));
    }

    # Transition and measurement
    arrF <- array(object$transition,c(dim(object$transition),nsim));
    arrWt <- array(object$measurement,c(dim(object$measurement),nsim));

    # Persistence matrix
    # The first one is a failsafe mechanism for xreg
    matG <- array(object$persistence, c(length(object$persistence), nsim),
                  dimnames=list(names(object$persistence), paste0("nsim",c(1:nsim))));

    #### Fill in the values in matrices ####
    # k is the index for randomParameters columns
    k <- 0;
    # Fill in the ETS parameters
    if(etsModel){
        if(any(parametersNames=="alpha")){
            matG["alpha",] <- randomParameters[,"alpha"];
            k <- k+1;
        }
        if(any(parametersNames=="beta")){
            matG["beta",] <- randomParameters[,"beta"];
            k <- k+1;
        }
        if(any(substr(parametersNames,1,5)=="gamma")){
            gammas <- which(substr(colnames(randomParameters),1,5)=="gamma");
            matG[colnames(randomParameters)[gammas],] <- t(randomParameters[,gammas,drop=FALSE]);
            k <- k+length(gammas);
        }

        # If we have phi, update the transition and measurement matrices
        if(any(parametersNames=="phi")){
            arrF[1,2,] <- arrF[2,2,] <- randomParameters[,"phi"];
            arrWt[,2,] <- matrix(randomParameters[,"phi"],nrow(object$measurement),nsim,byrow=TRUE);
            k <- k+1;
        }
    }

    # Transition and persistence of CES
    if(cesModel){
        # j is for states in matVt
        j <- 0;
        # No seasonality
        if(length(alpha0)>0){
            if(length(alpha0)==1){
                arrF[1,2,] <- randomParameters[,"alpha_1"] - 1;
                arrF[2,2,] <- 1 - randomParameters[,"alpha_0"];
                matG[1,] <- randomParameters[,"alpha_0"] - randomParameters[,"alpha_1"];
                matG[2,] <- randomParameters[,"alpha_0"] + randomParameters[,"alpha_1"];
                k[] <- k + 2;
            }
            # Simple seasonality, lagged CES
            else{
                for(i in 1:nSeasonal){
                    arrF[i*2-1,i*2,] <- randomParameters[,alpha1[i]] - 1;
                    arrF[i*2,i*2,] <- 1-randomParameters[,alpha0[i]];
                    matG[2*i-1,] <- randomParameters[,alpha0[i]] - randomParameters[,alpha1[i]];
                    matG[2*i,] <- randomParameters[,alpha0[i]] + randomParameters[,alpha1[i]];
                    k[] <- k + 2;
                }
                # }
            }
        }

        if(length(beta)>0){
            if(object$seasonality=="partial"){
                # Partial seasonality with a real part only
                for(i in 1:nSeasonal){
                    matG[k+i,] <- randomParameters[,beta[i]];
                }
                k[] <- k + nSeasonal;
            }
            else if(object$seasonality=="full"){
                # Full seasonality with both real and imaginary parts
                for(i in 1:nSeasonal){
                    arrF[k+i*2-1,k+i*2,] <- randomParameters[,beta1[i]]-1;
                    arrF[k+i*2,k+i*2,] <- 1 - randomParameters[,beta0[i]];
                    matG[k+2*i-1,] <- randomParameters[,beta0[i]] - randomParameters[,beta1[i]];
                    matG[k+2*i,] <- randomParameters[,beta0[i]] + randomParameters[,beta1[i]];
                }
                k[] <- k + 2*nSeasonal;
            }
        }
    }

    # Transition, persistence and measurement of GUM
    # if(gumModel){}

    if(xregModel && any(substr(parametersNames,1,5)=="delta")){
        deltas <- which(substr(colnames(randomParameters),1,5)=="delta");
        matG[colnames(randomParameters)[deltas],] <- t(randomParameters[,deltas,drop=FALSE]);
        k <- k+length(deltas);
    }

    # Fill in the persistence and transition for ARIMA
    if(arimaModel){
        if(is.list(object$orders)){
            arOrders <- object$orders$ar;
            iOrders <- object$orders$i;
            maOrders <- object$orders$ma;
        }
        else if(is.vector(object$orders)){
            arOrders <- object$orders[1];
            iOrders <- object$orders[2];
            maOrders <- object$orders[3];
        }

        # See if AR is needed
        arRequired <- FALSE;
        if(sum(arOrders)>0){
            arRequired[] <- TRUE;
        }
        # See if I is needed
        iRequired <- FALSE;
        if(sum(iOrders)>0){
            iRequired[] <- TRUE;
        }
        # See if I is needed
        maRequired <- FALSE;
        if(sum(maOrders)>0){
            maRequired[] <- TRUE;
        }

        # Define maxOrder and make all the values look similar (for the polynomials)
        maxOrder <- max(length(arOrders),length(iOrders),length(maOrders),length(lags));
        if(length(arOrders)!=maxOrder){
            arOrders <- c(arOrders,rep(0,maxOrder-length(arOrders)));
        }
        if(length(iOrders)!=maxOrder){
            iOrders <- c(iOrders,rep(0,maxOrder-length(iOrders)));
        }
        if(length(maOrders)!=maxOrder){
            maOrders <- c(maOrders,rep(0,maxOrder-length(maOrders)));
        }
        if(length(lags)!=maxOrder){
            lagsNew <- c(lags,rep(0,maxOrder-length(lags)));
            arOrders <- arOrders[lagsNew!=0];
            iOrders <- iOrders[lagsNew!=0];
            maOrders <- maOrders[lagsNew!=0];
        }
        # The provided parameters
        armaParameters <- object$other$armaParameters;
        # Check if the AR / MA parameters were estimated
        arEstimate <- any((substr(parametersNames,1,3)=="phi") & (nchar(parametersNames)>3))
        maEstimate <- any(substr(parametersNames,1,5)=="theta");

        # polyIndex is the index of the phi / theta parameters -1
        if(any(c(arEstimate,maEstimate))){
            polyIndex <- min(which((substr(parametersNames,1,3)=="phi") & (nchar(parametersNames)>3)),
                             which(substr(parametersNames,1,5)=="theta")) -1;
        }
        # If AR / MA are not estimated, then we don't care
        else{
            polyIndex <- -1;
        }

        for(i in 1:nsim){
            # Call the function returning ARI and MA polynomials
            arimaPolynomials <- lapply(adamCpp$polynomialise(randomParameters[i,polyIndex+1:sum(c(arOrders*arEstimate,maOrders*maEstimate))],
                                                             arOrders, iOrders, maOrders,
                                                             arEstimate, maEstimate, armaParameters, lags), as.vector);

            # Fill in the transition and persistence matrices
            if(nrow(nonZeroARI)>0){
                arrF[componentsNumberETS+nonZeroARI[,2],componentsNumberETS+1:componentsNumberARIMA,i] <-
                    -arimaPolynomials$ariPolynomial[nonZeroARI[,1]];
                matG[componentsNumberETS+nonZeroARI[,2],i] <- -arimaPolynomials$ariPolynomial[nonZeroARI[,1]];
            }
            if(nrow(nonZeroMA)>0){
                matG[componentsNumberETS+nonZeroMA[,2],i] <- matG[componentsNumberETS+nonZeroMA[,2],i] +
                    arimaPolynomials$maPolynomial[nonZeroMA[,1]];
            }
        }
        k <- k+sum(c(arOrders*arEstimate,maOrders*maEstimate));
    }

    # j is the index for the components in the profile
    j <- 0
    # Fill in the profile values
    profilesRecentArray <- array(object$profileInitial,c(dim(object$profile),nsim));
    if(etsModel){
        j <- j+1;
        if(any(parametersNames=="level")){
            profilesRecentArray[j,1,] <- randomParameters[,"level"];
            k <- k+1;
        }
        if(any(parametersNames=="trend")){
            j <- j+1;
            profilesRecentArray[j,1,] <- randomParameters[,"trend"];
            k <- k+1;
        }
        if(any(substr(parametersNames,1,8)=="seasonal")){
            # If there is only one seasonality
            if(any(substr(parametersNames,1,9)=="seasonal_")){
                initialSeasonalIndices <- 1;
                seasonalNames <- "seasonal"
            }
            # If there are several
            else{
                # This assumes that we cannot have more than 9 seasonalities.
                initialSeasonalIndices <- as.numeric(unique(substr(parametersNames[substr(parametersNames,1,8)=="seasonal"],9,9)));
                seasonalNames <- unique(substr(parametersNames[substr(parametersNames,1,8)=="seasonal"],1,9));
            }
            for(i in initialSeasonalIndices){
                profilesRecentArray[j+i,1:(lagsSeasonal[i]-1),] <-
                    t(randomParameters[,paste0(seasonalNames[i],"_",c(1:(lagsSeasonal[i]-1)))]);
                profilesRecentArray[j+i,lagsSeasonal[i],] <-
                    switch(Stype,
                           "A"=-apply(profilesRecentArray[j+i,1:(lagsSeasonal[i]-1),,drop=FALSE],3,sum),
                           "M"=1/apply(profilesRecentArray[j+i,1:(lagsSeasonal[i]-1),,drop=FALSE],3,prod),
                           0);
            }
            j <- j+max(initialSeasonalIndices);
            k <- k+length(initialSeasonalIndices);
        }
    }
    # CES states
    # if(cesModel){}
    # GUM states
    # if(gumModel){}
    # ARIMA states in the profileRecent
    if(arimaModel){
        # See if the initials were estimated
        # initialArimaNumber <- sum(substr(parametersNames,1,10)=="ARIMAState");
        initialArimaNumber <- sum(substr(colnames(object$states),1,10)=="ARIMAState");

        # This is needed in order to propagate initials of ARIMA to all components
        if(any(object$initialType==c("optimal","two-stage")) && any(c(arEstimate,maEstimate))){
            if(nrow(nonZeroARI)>0 && nrow(nonZeroARI)>=nrow(nonZeroMA)){
                for(i in 1:nsim){
                    # Call the function returning ARI and MA polynomials
                    ### This is not optimal, as the polynomialiser() is called twice (for parameters and here),
                    ### but this is simpler
                    arimaPolynomials <- lapply(adamCpp$polynomialise(randomParameters[i,polyIndex+1:sum(c(arOrders*arEstimate,maOrders*maEstimate))],
                                                                     arOrders, iOrders, maOrders,
                                                                     arEstimate, maEstimate, armaParameters, lags), as.vector);
                    profilesRecentArray[j+componentsNumberARIMA, 1:initialArimaNumber, i] <-
                        randomParameters[i, k+1:initialArimaNumber];
                    profilesRecentArray[j+nonZeroARI[,2], 1:initialArimaNumber, i] <-
                        switch(Etype,
                               "A"= arimaPolynomials$ariPolynomial[nonZeroARI[,1]] %*%
                                   t(profilesRecentArray[j+componentsNumberARIMA,
                                                         1:initialArimaNumber, i]),
                               "M"=exp(arimaPolynomials$ariPolynomial[nonZeroARI[,1]] %*%
                                           t(log(profilesRecentArray[j+componentsNumberARIMA,
                                                                     1:initialArimaNumber, i]))));
                }
            }
            else{
                for(i in 1:nsim){
                    # Call the function returning ARI and MA polynomials
                    arimaPolynomials <- lapply(adamCpp$polynomialise(randomParameters[i,polyIndex+1:sum(c(arOrders*arEstimate,maOrders*maEstimate))],
                                                                     arOrders, iOrders, maOrders,
                                                                     arEstimate, maEstimate, armaParameters, lags), as.vector);
                    profilesRecentArray[componentsNumberETS+componentsNumberARIMA, 1:initialArimaNumber, i] <-
                        randomParameters[i, k+1:initialArimaNumber];
                    profilesRecentArray[j+nonZeroMA[,2], 1:initialArimaNumber, i] <-
                        switch(Etype,
                               "A"=arimaPolynomials$maPolynomial[nonZeroMA[,1]] %*%
                                   t(profilesRecentArray[componentsNumberETS+componentsNumberARIMA,
                                                         1:initialArimaNumber, i]),
                               "M"=exp(arimaPolynomials$maPolynomial[nonZeroMA[,1]] %*%
                                           t(log(profilesRecentArray[componentsNumberETS+componentsNumberARIMA,
                                                                     1:initialArimaNumber, i]))));
                }
            }
        }
        j <- j+initialArimaNumber;
        k <- k+initialArimaNumber;
    }
    # Regression part
    if(xregModel){
        xregNumberToEstimate <- sum(xregParametersEstimated);
        profilesRecentArray[j+which(xregParametersEstimated==1),1,] <- t(randomParameters[,k+1:xregNumberToEstimate]);
        # Normalise initials
        for(i in which(xregParametersMissing!=0)){
            profilesRecentArray[j+i,1,] <- -colSums(profilesRecentArray[j+which(xregParametersEstimated==1),1,]);
        }
        j[] <- j+xregNumberToEstimate;
        k[] <- k+xregNumberToEstimate;
    }
    if(constantRequired){
        profilesRecentArray[j+1,1,] <- randomParameters[,k+1];
    }

    if(is.null(object$occurrence) || !is.occurrence(object$occurrence)){
        ot <- matrix(rep(1, obsInSample));
        pt <- rep(1, obsInSample);
    }
    else{
        ot <- matrix(actuals(object$occurrence));
        pt <- fitted(object$occurrence);
    }

    yt <- matrix(actuals(object));

    # Refit the model with the new parameter
    adamRefitted <- adamCpp$reapply(yt, ot,
                                    arrVt, arrWt,
                                    arrF, matG,
                                    indexLookupTable, profilesRecentArray,
                                    any(object$initialType==c("backcasting","complete")))

    arrVt[] <- adamRefitted$states;
    fittedMatrix[] <- adamRefitted$fitted * as.vector(pt);
    profilesRecentArray[] <- adamRefitted$profile;

    # If this was a model in logarithms (e.g. ARIMA for sm), then take exponent
    if(any(unlist(gregexpr("in logs",object$model))!=-1)){
        fittedMatrix[] <- exp(fittedMatrix);
    }

    return(structure(list(timeElapsed=Sys.time()-startTime,
                          y=actuals(object), states=arrVt, refitted=fittedMatrix,
                          fitted=fitted(object), model=object$model,
                          transition=arrF, measurement=arrWt, persistence=matG,
                          profile=profilesRecentArray, randomParameters=randomParameters),
                     class="reapply"));
}

#' @export
reapply.adamCombined <- function(object, nsim=1000, type=c("opg","hessian","bootstrap"),
                                 bootstrap=FALSE, heuristics=NULL, ...){
    type <- covarTypeResolver(type, bootstrap);
    startTime <- Sys.time();

    # Remove ICw, which are lower than 0.001
    object$ICw[object$ICw<1e-2] <- 0;
    object$ICw[] <- object$ICw / sum(object$ICw);

    # List of refitted matrices
    yRefitted <- vector("list", length(object$models));
    names(yRefitted) <- names(object$models);

    for(i in 1:length(object$models)){
        if(object$ICw[i]==0){
            next;
        }
        yRefitted[[i]] <- reapply(object$models[[i]], nsim=nsim, type=type,
                                  heuristics=heuristics, ...)$refitted;
    }

    # Get rid of specific models to save RAM
    object$models <- NULL;

    # Keep only the used weights
    yRefitted <- yRefitted[object$ICw!=0];
    object$ICw <- object$ICw[object$ICw!=0];

    return(structure(list(timeElapsed=Sys.time()-startTime,
                          y=actuals(object), refitted=yRefitted,
                          fitted=fitted(object), model=object$model,
                          ICw=object$ICw),
                     class=c("reapplyCombined","reapply")));
}


#' @importFrom grDevices rgb colorRampPalette palette
#' @export
plot.reapply <- function(x, ...){
    paletteBasic <- paletteDetector(c("black","red","purple","blue","darkgrey","grey95"));

    nLevels <- 5
    cols <- colorRampPalette(c(paletteBasic[6],paletteBasic[5]))(nLevels)[findInterval(1:nLevels,
                                                                                       seq(1, nLevels, length.out=nLevels))];

    ellipsis <- list(...);
    ellipsis$x <- actuals(x);

    if(any(class(ellipsis$x)=="zoo")){
        yQuantiles <- zoo(matrix(0,length(ellipsis$x),11),order.by=time(ellipsis$x));
    }
    else{
        yQuantiles <- ts(matrix(0,length(ellipsis$x),11),start=start(ellipsis$x),frequency=frequency(ellipsis$x));
    }
    quantileseq <- seq(0,1,length.out=11);
    yQuantiles[,1] <- apply(x$refitted,1,quantile,0.975,na.rm=TRUE);
    yQuantiles[,11] <- apply(x$refitted,1,quantile,0.025,na.rm=TRUE);
    for(i in 2:10){
        yQuantiles[,i] <- apply(x$refitted,1,quantile,quantileseq[i],na.rm=TRUE);
    }

    if(is.null(ellipsis$ylim)){
        ellipsis$ylim <- range(c(as.vector(ellipsis$x),as.vector(fitted(x))),na.rm=TRUE);
    }
    if(is.null(ellipsis$main)){
        ellipsis$main <- paste0("Refitted values of ",x$model);
    }
    if(is.null(ellipsis$ylab)){
        ellipsis$ylab <- "";
    }

    do.call(plot, ellipsis);
    for(i in 1:nLevels){
        polygon(c(time(yQuantiles),rev(time(yQuantiles))), c(as.vector(yQuantiles[,i]),
                                                             rev(as.vector(yQuantiles[,11-i+1]))),
                col=cols[i], border=paletteBasic[5])
    }
    lines(ellipsis$x,col=paletteBasic[1],lwd=1);
    lines(fitted(x),col=paletteBasic[3],lwd=2,lty=2);
}

#' @export
plot.reapplyCombined <- function(x, ...){
    paletteBasic <- paletteDetector(c("black","red","purple","blue","darkgrey","grey95"));

    nLevels <- 5
    cols <- colorRampPalette(c(paletteBasic[6],paletteBasic[5]))(nLevels)[findInterval(1:nLevels,
                                                                                       seq(1, nLevels, length.out=nLevels))];

    ellipsis <- list(...);
    ellipsis$x <- actuals(x);

    if(any(class(ellipsis$x)=="zoo")){
        yQuantiles <- zoo(matrix(0,length(ellipsis$x),11),order.by=time(ellipsis$x));
    }
    else{
        yQuantiles <- ts(matrix(0,length(ellipsis$x),11),start=start(ellipsis$x),frequency=frequency(ellipsis$x));
    }
    quantileseq <- seq(0,1,length.out=11);
    for(j in 1:length(x$refitted)){
        yQuantiles[,1] <- yQuantiles[,1] + apply(x$refitted[[j]],1,quantile,0.975,na.rm=TRUE)* x$ICw[j];
        yQuantiles[,11] <- yQuantiles[,11] + apply(x$refitted[[j]],1,quantile,0.025,na.rm=TRUE)* x$ICw[j];
        for(i in 2:10){
            yQuantiles[,i] <- yQuantiles[,i] + apply(x$refitted[[j]],1,quantile,quantileseq[i],na.rm=TRUE)* x$ICw[j];
        }
    }

    if(is.null(ellipsis$ylim)){
        ellipsis$ylim <- range(c(as.vector(ellipsis$x),as.vector(fitted(x))),na.rm=TRUE);
    }
    if(is.null(ellipsis$main)){
        ellipsis$main <- paste0("Refitted values of ",x$model);
    }
    if(is.null(ellipsis$ylab)){
        ellipsis$ylab <- "";
    }

    do.call(plot, ellipsis);
    for(i in 1:nLevels){
        polygon(c(time(yQuantiles),rev(time(yQuantiles))), c(as.vector(yQuantiles[,i]),
                                                             rev(as.vector(yQuantiles[,11-i+1]))),
                col=cols[i], border=paletteBasic[5])
    }
    lines(ellipsis$x,col=paletteBasic[1],lwd=1);
    lines(fitted(x),col=paletteBasic[3],lwd=2,lty=2);
}

#' @export
print.reapply <- function(x, ...){
    nsim <- ncol(x$refitted);
    cat("Time elapsed:",round(as.numeric(x$timeElapsed,units="secs"),2),"seconds");
    cat("\nModel refitted:",x$model);
    cat("\nNumber of simulation paths produced:",nsim);
}

#' @export
print.reapplyCombined <- function(x, ...){
    nsim <- ncol(x$refitted[[1]]);
    cat("Time elapsed:",round(as.numeric(x$timeElapsed,units="secs"),2),"seconds");
    cat("\nModel refitted:",x$model);
    cat("\nNumber of simulation paths produced:",nsim);
}

#' @rdname reapply
#' @export reforecast
reforecast <- function(object, h=10, newdata=NULL, occurrence=NULL,
                       interval=c("prediction", "confidence", "none"),
                       level=0.95, side=c("both","upper","lower"), cumulative=FALSE,
                       nsim=100, ...) UseMethod("reforecast")

#' @export
reforecast.default <- function(object, h=10, newdata=NULL, occurrence=NULL,
                               interval=c("prediction", "confidence", "none"),
                               level=0.95, side=c("both","upper","lower"), cumulative=FALSE,
                               nsim=100, ...){
    warning(paste0("The method is not implemented for the object of the class ,",class(object)[1]),
            call.=FALSE);
    return(forecast(object=object, h=h, newdata=newdata, occurrence=occurrence,
                    interval=interval, level=level, side=side, cumulative=cumulative,
                    nsim=nsim, ...));
}

#' @export
reforecast.adam <- function(object, h=10, newdata=NULL, occurrence=NULL,
                            interval=c("prediction", "confidence", "none"),
                            level=0.95, side=c("both","upper","lower"), cumulative=FALSE,
                            nsim=100, type=c("opg","hessian","bootstrap"),
                            bootstrap=FALSE, heuristics=NULL, ...){
    type <- covarTypeResolver(type, bootstrap);

    objectRefitted <- reapply(object, nsim=nsim, type=type, heuristics=heuristics, ...);
    ellipsis <- list(...);

    # Check whether we deal with adam ETS or the conventional
    adamETS <- adamETSChecker(object);

    # If the trim is not provided, set it to 1%
    if(is.null(ellipsis$trim)){
        trim <- 0.01;
    }
    else{
        trim <- ellipsis$trim;
    }

    #### <--- This part is widely a copy-paste from forecast.adam()
    interval <- match.arg(interval[1],c("none", "prediction", "confidence","simulated"));
    side <- match.arg(side);

    # Model type
    model <- modelType(object);
    Etype <- errorType(object);
    Ttype <- substr(model,2,2);
    Stype <- substr(model,nchar(model),nchar(model));

    # Technical parameters
    lagsModelAll <- modelLags(object);
    lagsModelMax <- max(lagsModelAll);
    profilesRecentArray <- objectRefitted$profile;

    # Get componentsNumberETS, seasonal and componentsNumberARIMA
    componentsDefined <- componentsDefiner(object);
    componentsNumberETS <- componentsDefined$componentsNumberETS;
    componentsNumberETSSeasonal <- componentsDefined$componentsNumberETSSeasonal;
    componentsNumberETSNonSeasonal <- componentsDefined$componentsNumberETSNonSeasonal;
    componentsNumberARIMA <- componentsDefined$componentsNumberARIMA;
    constantRequired <- componentsDefined$constantRequired;

    obsStates <- nrow(object$states);
    obsInSample <- nobs(object);
    indexLookupTable <- adamProfileCreator(lagsModelAll, lagsModelMax,
                                           obsInSample+h)$lookup[,-c(1:(obsInSample+lagsModelMax)),drop=FALSE];

    yClasses <- class(actuals(object));

    if(any(yClasses=="ts")){
        # ts structure
        if(h>0){
            yForecastStart <- time(actuals(object))[obsInSample]+deltat(actuals(object));
        }
        else{
            yForecastStart <- time(actuals(object))[1];
        }
        yFrequency <- frequency(actuals(object));
    }
    else{
        # zoo thingy
        yIndex <- time(actuals(object));
        if(h>0){
            yForecastIndex <- yIndex[obsInSample]+diff(tail(yIndex,2))*c(1:h);
        }
        else{
            yForecastIndex <- yIndex;
        }
    }

    # How many levels did user asked to produce
    nLevels <- length(level);
    # Cumulative forecasts have only one observation
    if(cumulative){
        # hFinal is the number of elements we will have in the final forecast
        hFinal <- 1;
    }
    else{
        if(h>0){
            hFinal <- h;
        }
        else{
            hFinal <- obsInSample;
        }
    }

    # Create necessary matrices for the forecasts
    if(any(yClasses=="ts")){
        yForecast <- ts(vector("numeric", hFinal), start=yForecastStart, frequency=yFrequency);
        yUpper <- yLower <- ts(matrix(0,hFinal,nLevels), start=yForecastStart, frequency=yFrequency);
    }
    else{
        yForecast <- zoo(vector("numeric", hFinal), order.by=yForecastIndex);
        yUpper <- yLower <- zoo(matrix(0,hFinal,nLevels), order.by=yForecastIndex);
    }

    # If the occurrence values are provided for the holdout
    if(!is.null(occurrence) && is.numeric(occurrence)){
        pForecast <- occurrence;
    }
    else{
        # If this is a mixture model, produce forecasts for the occurrence
        if(is.occurrence(object$occurrence)){
            occurrenceModel <- TRUE;
            if(object$occurrence$occurrence=="provided"){
                pForecast <- rep(1,h);
            }
            else{
                pForecast <- forecast(object$occurrence,h=h,newdata=newdata)$mean;
            }
        }
        else{
            occurrenceModel <- FALSE;
            # If this was provided occurrence, then use provided values
            if(!is.null(object$occurrence) && is.list(object$occurrence) &&
               !is.null(object$occurrence$occurrence) &&
               (object$occurrence$occurrence=="provided")){
                pForecast <- object$occurrence$forecast;
            }
            else{
                pForecast <- rep(1, h);
            }
        }
    }

    # Make sure that the values are of the correct length
    if(h<length(pForecast)){
        pForecast <- pForecast[1:h];
    }
    else if(h>length(pForecast)){
        pForecast <- c(pForecast, rep(tail(pForecast,1), h-length(pForecast)));
    }

    # Set the levels
    if(interval!="none"){
        # Fix just in case a silly user used 95 etc instead of 0.95
        if(any(level>1)){
            level[] <- level / 100;
        }
        levelLow <- levelUp <- matrix(0,hFinal,nLevels);
        levelNew <- matrix(level,nrow=hFinal,ncol=nLevels,byrow=TRUE);

        # If this is an occurrence model, then take probability into account in the level.
        # This correction is only needed for approximate, because the others contain zeroes
        if(side=="both"){
            levelLow[] <- (1-levelNew)/2;
            levelUp[] <- (1+levelNew)/2;
        }
        else if(side=="upper"){
            levelLow[] <- 0;
            levelUp[] <- levelNew;
        }
        else{
            levelLow[] <- 1-levelNew;
            levelUp[] <- 1;
        }
        levelLow[levelLow<0] <- 0;
        levelUp[levelUp<0] <- 0;
    }

    #### Return adam.predict if h<=0 ####
    # If the horizon is zero, just construct fitted and potentially confidence interval thingy
    if(h<=0){
        # If prediction interval is needed, this can be done with predict.adam
        if(any(interval==c("prediction","none"))){
            warning(paste0("You've set h=",h," and interval=\"",interval,
                           "\". There is no point in using reforecast() function for your task. ",
                           "Using predict() method instead."),
                    call.=FALSE);
            return(predict(object, newdata=newdata,
                           interval=interval,
                           level=level, side=side, ...));
        }

        yForecast[] <- rowMeans(objectRefitted$refitted);
        if(interval=="confidence"){
            for(i in 1:hFinal){
                yLower[i,] <- quantile(objectRefitted$refitted[i,],levelLow[i,]);
                yUpper[i,] <- quantile(objectRefitted$refitted[i,],levelUp[i,]);
            }
        }
        return(structure(list(mean=yForecast, lower=yLower, upper=yUpper, model=object,
                              level=level, interval=interval, side=side),
                         class=c("adam.predict","adam.forecast")));
    }

    #### All the important matrices ####
    # Last h observations of measurement
    arrWt <- objectRefitted$measurement[obsInSample-c(h:1)+1,,,drop=FALSE];
    # If the forecast horizon is higher than the in-sample, duplicate the last value in matWt
    if(dim(arrWt)[1]<h){
        arrWt <- array(tail(arrWt,1), c(h, ncol(arrWt), nsim), dimnames=list(NULL,colnames(arrWt),NULL));
    }

    # Deal with explanatory variables
    if(ncol(object$data)>1){
        xregNumber <- length(object$initial$xreg);
        xregNames <- names(object$initial$xreg);
        # The newdata is not provided
        if(is.null(newdata) && ((!is.null(object$holdout) && nrow(object$holdout)<h) ||
                                is.null(object$holdout))){
            # Salvage what data we can (if there is something)
            if(!is.null(object$holdout)){
                hNeeded <- h-nrow(object$holdout);
                xreg <- tail(object$data,h);
                xreg[1:nrow(object$holdout),] <- object$holdout;
            }
            else{
                hNeeded <- h;
                xreg <- tail(object$data,h);
            }

            if(is.matrix(xreg)){
                warning("The newdata is not provided.",
                        "Predicting the explanatory variables based on the in-sample data.",
                        call.=FALSE);
                for(i in 1:xregNumber){
                    xreg[,i] <- adam(object$data[,i+1],h=hNeeded,silent=TRUE)$forecast;
                }
            }
            else{
                warning("The newdata is not provided. Using last h in-sample observations instead.",
                        call.=FALSE);
            }
        }
        else if(is.null(newdata) && !is.null(object$holdout) && nrow(object$holdout)>=h){
            xreg <- object$holdout[1:h,,drop=FALSE];
        }
        else{
            # If this is not a matrix / data.frame, then convert to one
            if(!is.data.frame(newdata) && !is.matrix(newdata)){
                newdata <- as.data.frame(newdata);
                colnames(newdata) <- "xreg";
            }
            if(nrow(newdata)<h){
                warning(paste0("The newdata has ",nrow(newdata)," observations, while ",h," are needed. ",
                               "Using the last available values as future ones."),
                        call.=FALSE);
                newnRows <- h-nrow(newdata);
                # xreg <- rbind(as.matrix(newdata),matrix(rep(tail(newdata,1),each=newnRows),newnRows,ncol(newdata)));
                xreg <- newdata[c(1:nrow(newdata),rep(nrow(newdata)),each=newnRows),];
            }
            else if(nrow(newdata)>h){
                warning(paste0("The newdata has ",nrow(newdata)," observations, while only ",h," are needed. ",
                               "Using the last ",h," of them."),
                        call.=FALSE);
                xreg <- tail(newdata,h);
            }
            else{
                xreg <- newdata;
            }
        }

        # If the names are wrong, transform to data frame and expand
        if(!all(xregNames %in% colnames(xreg))){
            xreg <- as.data.frame(xreg);
        }

        # Expand the xreg if it is data frame to get the proper matrix
        if(is.data.frame(xreg)){
            testFormula <- formula(object);
            # Remove response variable
            testFormula[[2]] <- NULL;
            # Expand the variables. We cannot use alm, because it is based on obsInSample
            xregData <- model.frame(testFormula,data=xreg);
            # Binary, flagging factors in the data
            # Expanded stuff with all levels for factors
            if(any((attr(terms(xregData),"dataClasses")=="factor")[-1])){
                xregModelMatrix <- model.matrix(xregData,xregData,
                                                contrasts.arg=lapply(xregData[attr(terms(xregData),"dataClasses")=="factor"],
                                                                     contrasts, contrasts=FALSE));
            }
            else{
                xregModelMatrix <- model.matrix(xregData,data=xregData);
            }
            colnames(xregModelMatrix) <- make.names(colnames(xregModelMatrix), unique=TRUE);
            newdata <- as.matrix(xregModelMatrix)[,xregNames,drop=FALSE];
            rm(xregData,xregModelMatrix);
        }
        else{
            newdata <- xreg[,xregNames];
        }
        rm(xreg);

        arrWt[,componentsNumberETS+componentsNumberARIMA+c(1:xregNumber),] <- newdata;
    }
    else{
        xregNumber <- 0;
    }

    # Create C++ adam class
    adamCpp <- new(adamCore,
                   lagsModelAll, Etype, Ttype, Stype,
                   componentsNumberETSNonSeasonal,
                   componentsNumberETSSeasonal,
                   componentsNumberETS, componentsNumberARIMA,
                   xregNumber, length(lagsModelAll),
                   constantRequired, adamETS);

    #### Simulate the data ####
    # If scale model is included, produce forecasts
    if(is.scale(object$scale)){
        sigmaValue <- forecast(object$scale,h=h,newdata=newdata,interval="none")$mean;
    }
    else{
        sigmaValue <- sigma(object);
    }
    # This stuff is needed in order to produce adequate values for weird models
    EtypeModified <- Etype;
    if(Etype=="A" && any(object$distribution==c("dlnorm","dinvgauss","dgamma","dls","dllaplace"))){
        EtypeModified[] <- "M";
    }
    # Matrix for the errors
    arrErrors <- array(switch(object$distribution,
                              "dnorm"=rnorm(h*nsim^2, 0, sigmaValue),
                              "dlaplace"=rlaplace(h*nsim^2, 0, sigmaValue/2),
                              "ds"=rs(h*nsim^2, 0, (sigmaValue^2/120)^0.25),
                              "dgnorm"=rgnorm(h*nsim^2, 0,
                                              sigmaValue*sqrt(gamma(1/object$other$shape)/gamma(3/object$other$shape)),
                                              object$other$shape),
                              "dlogis"=rlogis(h*nsim^2, 0, sigmaValue*sqrt(3)/pi),
                              "dt"=rt(h*nsim^2, obsInSample-nparam(object)),
                              "dalaplace"=ralaplace(h*nsim^2, 0,
                                                    sqrt(sigmaValue^2*object$other$alpha^2*(1-object$other$alpha)^2/
                                                             (object$other$alpha^2+(1-object$other$alpha)^2)),
                                                    object$other$alpha),
                              "dlnorm"=rlnorm(h*nsim^2, -extractScale(object)^2/2, extractScale(object))-1,
                              "dinvgauss"=rinvgauss(h*nsim^2, 1, dispersion=sigmaValue^2)-1,
                              "dgamma"=rgamma(h*nsim^2, shape=sigmaValue^{-2}, scale=sigmaValue^2)-1,
                              "dllaplace"=exp(rlaplace(h*nsim^2, 0, sigmaValue/2))-1,
                              "dls"=exp(rs(h*nsim^2, 0, (sigmaValue^2/120)^0.25))-1,
                              "dlgnorm"=exp(rgnorm(h*nsim^2, 0,
                                                   sigmaValue*sqrt(gamma(1/object$other$shape)/gamma(3/object$other$shape))))-1),
                       c(h,nsim,nsim));
    # Normalise errors in order not to get ridiculous things on small nsim
    if(nsim<=500){
        if(Etype=="A"){
            arrErrors[] <- arrErrors - array(apply(arrErrors,1,mean),c(h,nsim,nsim));
        }
        else{
            arrErrors[] <- (1+arrErrors) / array(apply(1+arrErrors,1,mean),c(h,nsim,nsim))-1;
        }
    }
    # Array of the simulated data
    arrayYSimulated <- array(0,c(h,nsim,nsim));
    # Start the loop... might take some time
    arrayYSimulated[] <- adamCpp$reforecast(arrErrors, array(rbinom(h*nsim^2, 1, pForecast), c(h,nsim,nsim)),
                                            arrWt,
                                            objectRefitted$transition, objectRefitted$persistence,
                                            indexLookupTable, profilesRecentArray,
                                            EtypeModified)$data;

    #### Note that the cumulative doesn't work with oes at the moment!
    if(cumulative){
        yForecast[] <- mean(apply(arrayYSimulated,1,sum,na.rm=TRUE,trim=trim));
        if(interval!="none"){
            yLower[] <- quantile(apply(arrayYSimulated,1,sum,na.rm=TRUE),levelLow,type=7);
            yUpper[] <- quantile(apply(arrayYSimulated,1,sum,na.rm=TRUE),levelUp,type=7);
        }
    }
    else{
        yForecast[] <- apply(arrayYSimulated,1,mean,na.rm=TRUE,trim=trim);
        if(interval=="prediction"){
            for(i in 1:h){
                for(j in 1:nLevels){
                    yLower[i,j] <- quantile(arrayYSimulated[i,,],levelLow[i,j],na.rm=TRUE,type=7);
                    yUpper[i,j] <- quantile(arrayYSimulated[i,,],levelUp[i,j],na.rm=TRUE,type=7);
                }
            }
        }
        else if(interval=="confidence"){
            for(i in 1:h){
                yLower[i,] <- quantile(apply(arrayYSimulated[i,,],2,mean,na.rm=TRUE,trim=trim),levelLow[i,],na.rm=TRUE,type=7);
                yUpper[i,] <- quantile(apply(arrayYSimulated[i,,],2,mean,na.rm=TRUE,trim=trim),levelUp[i,],na.rm=TRUE,type=7);
            }
        }
    }

    # Fix of prediction intervals depending on what has happened
    if(interval!="none"){
        # Make sensible values out of those weird quantiles
        if(!cumulative){
            if(any(levelLow==0)){
                # zoo does not like, when you work with matrices of indices... silly thing
                yBoundBuffer <- levelLow;
                yBoundBuffer[] <- yLower
                if(Etype=="A"){
                    yBoundBuffer[levelLow==0] <- -Inf;
                    yLower[] <- yBoundBuffer;
                }
                else{
                    yBoundBuffer[levelLow==0] <- 0;
                    yLower[] <- yBoundBuffer;
                }
            }
            if(any(levelUp==1)){
                # zoo does not like, when you work with matrices of indices... silly thing
                yBoundBuffer <- levelUp;
                yBoundBuffer[] <- yUpper
                yBoundBuffer[levelUp==1] <- Inf;
                yUpper[] <- yBoundBuffer;
            }
        }
        else{
            if(Etype=="A" && any(levelLow==0)){
                yLower[] <- -Inf;
            }
            else if(Etype=="M" && any(levelLow==0)){
                yLower[] <- 0;
            }
            if(any(levelUp==1)){
                yUpper[] <- Inf;
            }
        }

        # Substitute NAs and NaNs with zeroes
        if(any(is.nan(yLower)) || any(is.na(yLower))){
            yLower[is.nan(yLower)] <- switch(Etype,"A"=0,"M"=1);
            yLower[is.na(yLower)] <- switch(Etype,"A"=0,"M"=1);
        }
        if(any(is.nan(yUpper)) || any(is.na(yUpper))){
            yUpper[is.nan(yUpper)] <- switch(Etype,"A"=0,"M"=1);
            yUpper[is.na(yUpper)] <- switch(Etype,"A"=0,"M"=1);
        }

        # Check what we have from the occurrence model
        if(occurrenceModel){
            # If there are NAs, then there's no variability and no intervals.
            if(any(is.na(yUpper))){
                yUpper[is.na(yUpper)] <- (yForecast/pForecast)[is.na(yUpper)];
            }
            if(any(is.na(yLower))){
                yLower[is.na(yLower)] <- 0;
            }
        }

        colnames(yLower) <- switch(side,
                                   "both"=paste0("Lower bound (",(1-level)/2*100,"%)"),
                                   "lower"=paste0("Lower bound (",(1-level)*100,"%)"),
                                   "upper"=rep("Lower 0%",nLevels));

        colnames(yUpper) <- switch(side,
                                   "both"=paste0("Upper bound (",(1+level)/2*100,"%)"),
                                   "lower"=rep("Upper 100%",nLevels),
                                   "upper"=paste0("Upper bound (",level*100,"%)"));
    }
    else{
        yUpper[] <- yLower[] <- NA;
    }

    # If this was a model in logarithms (e.g. ARIMA for sm), then take exponent
    if(any(unlist(gregexpr("in logs",object$model))!=-1)){
        yForecast[] <- exp(yForecast);
        yLower[] <- exp(yLower);
        yUpper[] <- exp(yUpper);
    }

    structure(list(mean=yForecast, lower=yLower, upper=yUpper, model=object,
                   level=level, interval=interval, side=side, cumulative=cumulative,
                   h=h, paths=arrayYSimulated),
              class=c("adam.forecast","smooth.forecast","forecast"));
}
