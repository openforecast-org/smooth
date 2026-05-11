#' Functions that extract values from the fitted model
#'
#' These functions allow extracting orders and lags for \code{ssarima()}, \code{gum()} and \code{sma()},
#' type of model from \code{es()} and \code{ces()} and name of model.
#'
#' \code{orders()} and \code{lags()} are useful only for SSARIMA, GUM and SMA. They return \code{NA} for other functions.
#' This can also be applied to \code{arima()}, \code{Arima()} and \code{auto.arima()} functions from stats and forecast packages.
#' \code{modelType()} is useful only for ETS and CES. They return \code{NA} for other functions.
#' This can also be applied to \code{ets()} function from forecast package. \code{errorType}
#' extracts the type of error from the model (either additive or multiplicative). Finally, \code{modelName}
#' returns the name of the fitted model. For example, "ARIMA(0,1,1)". This is purely descriptive and
#' can also be applied to non-smooth classes, so that, for example, you can easily extract the name
#' of the fitted AR model from \code{ar()} function from \code{stats} package.
#'
#' @template ssAuthor
#' @template ssKeywords
#'
#' @aliases orders
#' @param object Model estimated using one of the functions of smooth package.
#' @param ... Currently nothing is accepted via ellipsis.
#' @return     Either vector, scalar or list with values is returned.
#' \code{orders()} in case of ssarima returns list of values:
#' \itemize{
#' \item \code{ar} - AR orders.
#' \item \code{i} - I orders.
#' \item \code{ma} - MA orders.
#' }
#' \code{lags()} returns the vector of lags of the model.
#' All the other functions return strings of character.
#' @seealso \link[smooth]{ssarima}, \link[smooth]{msarima}
#' @examples
#'
#' x <- rnorm(100,0,1)
#'
#' # Just as example. orders and lags do not return anything for ces() and es(). But modelType() does.
#' ourModel <- ces(x, h=10)
#' orders(ourModel)
#' lags(ourModel)
#' modelType(ourModel)
#' modelName(ourModel)
#'
#' # And as another example it does the opposite for gum() and ssarima()
#' ourModel <- gum(x, h=10, orders=c(1,1), lags=c(1,4))
#' orders(ourModel)
#' lags(ourModel)
#' modelType(ourModel)
#' modelName(ourModel)
#'
#' # Finally these values can be used for simulate functions or original functions.
#' ourModel <- auto.ssarima(x)
#' ssarima(x, orders=orders(ourModel), lags=lags(ourModel), constant=ourModel$constant)
#' sim.ssarima(orders=orders(ourModel), lags=lags(ourModel), constant=ourModel$constant)
#'
#' ourModel <- es(x)
#' es(x, model=modelType(ourModel))
#' sim.es(model=modelType(ourModel))
#'
#' @rdname orders
#' @export orders
orders <- function(object, ...) UseMethod("orders")

#' @aliases lags
#' @rdname orders
#' @export lags
lags <- function(object, ...) UseMethod("lags")

#' @aliases modelName
#' @rdname orders
#' @export modelName
modelName <- function(object, ...) UseMethod("modelName")

#' @aliases modelType
#' @rdname orders
#' @export modelType
modelType <- function(object, ...) UseMethod("modelType")

##### Likelihood function and stuff #####

#' @importFrom greybox AICc
#' @export
AICc.smooth <- function(object, ...){
    llikelihood <- logLik(object);
    nParamAll <- nparam(object);
    llikelihood <- llikelihood[1:length(llikelihood)];

    if(!is.null(object$occurrence)){
        obs <- sum(object$fitted!=0);
        nParamSizes <- nParamAll - object$nParam[1,3];
        IC <- (2*nParamAll - 2*llikelihood +
                   2*nParamSizes*(nParamSizes + 1) / (obs - nParamSizes - 1));
    }
    else{
        obs <- nobs(object);
        IC <- 2*nParamAll - 2*llikelihood + 2 * nParamAll * (nParamAll + 1) / (obs - nParamAll - 1);
    }

    return(IC);
}

#' @importFrom greybox BICc
#' @export
BICc.smooth <- function(object, ...){
    llikelihood <- logLik(object);
    nParamAll <- nparam(object);
    llikelihood <- llikelihood[1:length(llikelihood)];

    if(!is.null(object$occurrence)){
        obs <- sum(object$fitted!=0);
        nParamSizes <- nParamAll - object$nParam[1,3];
        IC <- - 2*llikelihood + (nParamSizes * log(obs) * obs) / (obs - nParamSizes - 1);
    }
    else{
        obs <- nobs(object);
        IC <- - 2*llikelihood + (nParamAll * log(obs) * obs) / (obs - nParamAll - 1);
    }

    return(IC);
}

#' Function returns the multiple steps ahead covariance matrix of forecast errors
#'
#' This function extracts covariance matrix of 1 to h steps ahead forecast errors for
#' \code{adam()}, \code{ssarima()}, \code{gum()}, \code{sma()}, \code{es()} and
#' \code{ces()} models.
#'
#' The function returns either scalar (if it is a non-smooth model)
#' or the matrix of (h x h) size with variances and covariances of 1 to h steps ahead
#' forecast errors.
#'
#' @template ssAuthor
#' @template ssKeywords
#'
#' @param object Model estimated using one of the functions of smooth package.
#' @param type What method to use in order to produce covariance matrix:
#' \enumerate{
#' \item \code{analytical} - based on the state space structure of the model and the
#' one-step-ahead forecast error. This works for pure additive and pure multiplicative
#' models. The values for the mixed models might be off.
#' \item \code{empirical} - based on the in-sample 1 to h steps ahead forecast errors
#' (works fine on larger samples);
#' \item \code{simulated} - the data is simulated from the estimated model, then the
#' same model is applied to it and then the empirical 1 to h steps ahead forecast
#' errors are produced;
#' }
#' @param h Forecast horizon to use in the calculations.
#' @param nsim Number of iterations to produce in the simulation. Only needed if
#' \code{type="simulated"}
#' @param ... Other parameters passed to simulate function (if \code{type="simulated"}
#' is used). These are \code{obs} and \code{seed}. By default \code{obs=1000}.
#' This approach increases the accuracy of covariance matrix on small samples
#' and intermittent data;
#' @return Scalar in cases of non-smooth functions. (h x h) matrix otherwise.
#'
#' @seealso \link[smooth]{orders}
#' @examples
#'
#' x <- rnorm(100,0,1)
#'
#' # A simple example with a 5x5 covariance matrix
#' ourModel <- ces(x, h=5)
#' multicov(ourModel)
#'
#' @rdname multicov
#' @export multicov
multicov <-  function(object, type=c("analytical","empirical","simulated"), h=10, nsim=1000,
                      ...) UseMethod("multicov")

#' @export
multicov.default <- function(object, type=c("analytical","empirical","simulated"), h=10, nsim=1000,
                             ...){
    # Function extracts the conditional variances from the model
    return(sigma(object)^2);
}

#' @aliases multicov.smooth
#' @rdname multicov
#' @export
multicov.smooth <- function(object, type=c("analytical","empirical","simulated"), h=10, nsim=1000,
                            ...){
    # Function extracts the conditional variances from the model

    if(is.smoothC(object)){
        stop("Sorry, but covariance matrix is not available for the combinations.",
            call.=FALSE)
    }

    type <- substr(type[1],1,1);

    if(is.null(object$persistence) & any(type==c("a","s"))){
        warning(paste0("The provided model does not contain the components necessary for the ",
                       "derivation of the covariance matrix.\n",
                       "Did you combine forecasts? Switching to 'empirical'"),
                call.=FALSE);
        type <- "e";
    }

    if(!is.null(object$occurrence) & type=="e"){
        warning(paste0("Empirical covariance matrix can be very inaccurate in cases of ",
                       "intemittent models.\nWe recommend using type='s' or type='a' instead."),
                call.=FALSE);
    }

    # Empirical covariance matrix
    if(type=="e"){
        if(errorType(object)=="A"){
            errors <- rmultistep(object, h=h);
        }
        else{
            errors <- log(1 + rmultistep(object, h=h));
        }
        if(!is.null(object$occurrence)){
            obs <- t((errors!=0)*1) %*% (errors!=0)*1;
            obs[obs==0] <- 1;
            df <- obs - nparam(object);
            df[df<=0] <- obs[df<=0];
        }
        else{
            obs <- matrix(nobs(object),ncol(errors),ncol(errors));
            df <- obs - nparam(object);
            df[df<=0] <- obs[df<=0];
        }
        covarMat <- t(errors) %*% errors / df;
    }
    # Simulated covariance matrix
    else if(type=="s"){
        smoothType <- smoothType(object);
        ellipsis <- list(...);
        if(any(names(ellipsis)=="obs")){
            obs <- ellipsis$obs;
        }
        else{
            obs <- length(actuals(object));
        }
        if(any(names(ellipsis)=="seed")){
            seed <- ellipsis$seed;
        }
        else{
            seed <- NULL;
        }

        h <- length(object$forecast);
        # if(smoothType=="ETS"){
        #     smoothFunction <- es;
        # }
        # # GUM models
        # else if(smoothType=="GUM"){
        #     smoothFunction <- gum;
        # }
        # # SSARIMA models
        # else if(smoothType=="ARIMA"){
        #     smoothFunction <- ssarima;
        # }
        # # CES models
        # else if(smoothType=="CES"){
        #     smoothFunction <- ces;
        # }
        # # SMA models
        # else if(smoothType=="SMA"){
        #     smoothFunction <- sma;
        # }

        covarArray <- array(NA,c(h,h,nsim));

        newData <- simulate(object, nsim=nsim, obs=obs, seed=seed);
        for(i in 1:nsim){
            # Apply the model to the simulated data
            # smoothModel <- smoothFunction(newData$data[,i], model=object, h=h);
            # Remove first h-1 and last values.
            # errors <- smoothModel$errors;
            errors <- rmultistep(object, h=h)

            # Transform errors if needed
            if(errorType(object)=="M"){
                # Remove zeroes if they are present
                errors <- errors[!apply(errors==0,1,any),];
                errors <- log(1 + errors);
            }

            # Calculate covariance matrix
            if(!is.null(object$occurrence)){
                obsInSample <- t((errors!=0)*1) %*% (errors!=0)*1;
                obsInSample[obsInSample==0] <- 1;
            }
            else{
                obsInSample <- matrix(nrow(errors),ncol(errors),ncol(errors));
            }
            covarArray[,,i] <- t(errors) %*% errors / obsInSample;
        }

        covarMat <- apply(covarArray,c(1,2),mean);
    }
    # Analytical covariance matrix
    else if(type=="a"){
        if(!is.null(object$occurrence)){
            ot <- (residuals(object)!=0)*1;
        }
        else{
            ot <- rep(1,nobs(object));
        }
        # h <- length(object$forecast);
        lagsModel <- modelLags(object);
        s2 <- sigma(object)^2;
        persistence <- matrix(object$persistence,length(object$persistence),1);
        transition <- object$transition;
        measurement <- object$measurement;

        covarMat <- covarAnal(lagsModel, h, measurement, transition, persistence, s2);

    }
    return(covarMat);
    # correlation matrix: multicov(test) / sqrt(diag(multicov(test)) %*% t(diag(multicov(test))))
}

#' @importFrom stats logLik
#' @export
logLik.smooth <- function(object, ...){
    if(is.null(object$logLik)){
        warning("The likelihood of this model is unavailable. Hint: did you use combinations?");
        return(NULL);
    }
    else{
        return(structure(object$logLik,nobs=nobs(object),df=nparam(object),class="logLik"));
    }
}
#' @export
logLik.smooth.sim <- function(object, ...){
    obs <- nobs(object);
    return(structure(object$logLik,nobs=obs,df=0,class="logLik"));
}

#' @importFrom stats nobs
#' @export
nobs.smooth <- function(object, ...){
    return(length(object$fitted));
}
#' @method nobs smooth.sim
#' @export
nobs.smooth.sim <- function(object, ...){
    if(is.null(dim(object$data))){
        return(length(object$data));
    }
    else{
        return(nrow(object$data));
    }
}

#' @importFrom greybox nparam

#' @export
nparam.smooth <- function(object, ...){
    if(is.null(object$nParam)){
        warning("Number of parameters of the model is unavailable. Hint: did you use combinations?",
                call.=FALSE);
        return(NULL);
    }
    else{
        return(object$nParam[1,ncol(object$nParam)]);
    }
}

#' Prediction Likelihood Score
#'
#' Function estimates Prediction Likelihood Score for the provided model
#'
#' Prediction likelihood score (PLS) is based on either normal or log-normal
#' distribution of errors. This is extracted from the provided model. The likelihood
#' based on the distribution of 1 to h steps ahead forecast errors is used in the process.
#'
#' @template ssAuthor
#' @template ssKeywords
#'
#' @param object The model estimated using smooth functions. This thing also accepts
#' other models (e.g. estimated using functions from forecast package), but may not always
#' work properly with them.
#' @param holdout The values for the holdout part of the sample. If the model was fitted
#' on the data with the \code{holdout=TRUE}, then the parameter is not needed.
#' @param ... Parameters passed to multicov function. The function is called in order to get
#' the covariance matrix of 1 to h steps ahead forecast errors.
#'
#' @return A value of the log-likelihood.
#' @references \itemize{
#' %\item Eltoft, T., Taesu, K., Te-Won, L. (2006). On the multivariate Laplace
#' distribution. IEEE Signal Processing Letters. 13 (5): 300-303.
#' \doi{10.1109/LSP.2006.870353} - this is not yet used in the function.
#' \item Snyder, R. D., Ord, J. K., Beaumont, A., 2012. Forecasting the intermittent
#' demand for slow-moving inventories: A modelling approach. International
#' Journal of Forecasting 28 (2), 485-496.
#' \item Kolassa, S., 2016. Evaluating predictive count data distributions in retail
#' sales forecasting. International Journal of Forecasting 32 (3), 788-803..
#' }
#' @examples
#'
#' # Generate data, apply es() with the holdout parameter and calculate PLS
#' x <- rnorm(100,0,1)
#' ourModel <- es(x, h=10, holdout=TRUE)
#' pls(ourModel, type="a")
#' pls(ourModel, type="e")
#' pls(ourModel, type="s", obs=100, nsim=100)
#'
#' @rdname pls
#' @export pls
pls <-  function(object, holdout=NULL, ...) UseMethod("pls")
# Function calculates PLS based on the provided model

#' @importFrom stats dnorm
#' @export
pls.default <- function(object, holdout=NULL, ...){
    if(is.null(holdout)){
        stop("We need the values from the holdout in order to proceed.",
             call.=FALSE);
    }
    h <- length(holdout);
    yForecast <- forecast(object, h=h)$mean;

    return(sum(dnorm(holdout,yForecast,sigma(object),log=TRUE)));
}

#' @rdname pls
#' @aliases pls.smooth
#' @export
pls.smooth <- function(object, holdout=NULL, ...){
    if(is.smoothC(object)){
        stop("Sorry, but PLS is not available for the combinations.",
             call.=FALSE)
    }
    # If holdout is provided, check it and use it. Otherwise try extracting from the model
    yForecast <- object$forecast;
    covarMat <- multicov(object, ...);
    if(!is.null(holdout)){
        if(length(yForecast)!=length(holdout)){
            if(is.null(object$holdout)){
                stop("The forecast of the model does not correspond to the provided holdout.",
                     call.=FALSE);
            }
            else{
                holdout <- object$holdout;
            }
        }
    }
    else{
        if(all(is.na(object$holdout))){
            stop("No values for the holdout are available. Cannot proceed.",
                 call.=FALSE);
        }
        holdout <- object$holdout;
    }
    h <- length(holdout);

    Etype <- errorType(object);
    loss <- object$loss;
    if(any(loss==c("MAE","MAEh","TMAE","GTMAE","MACE"))){
        loss <- "MAE";
    }
    else if(any(loss==c("HAM","HAMh","THAM","GTHAM","CHAM"))){
        loss <- "HAM";
    }
    else{
        loss <- "MSE";
    }

    densityFunction <- function(loss, ...){
        if(loss=="MAE"){
        # This is a simplification. The real multivariate Laplace is bizarre!
            scale <- sqrt(diag(covarMat)/2);
            plsValue <- sum(dlaplace(errors, 0, scale, log=TRUE));
        }
        else if(loss=="HAM"){
        # This is a simplification. We don't have multivariate HAM yet.
            scale <- (diag(covarMat)/120)^0.25;
            plsValue <- sum(ds(errors, 0, scale, log=TRUE));
        }
        else{
            if(is.infinite(det(covarMat))){
                plsValue <- -as.vector((log(2*pi)+(abs(determinant(covarMat)$modulus)))/2 +
                                           (t(errors) %*% solve(covarMat) %*% errors) / 2);
            }
            # This is the case with overfitting the data
            else if(det(covarMat)==0){
                # If there is any non-zero error, then it means that the model is completely wrong (because it predicts that sigma=0)
                if(any(errors!=0)){
                    plsValue <- -Inf;
                }
                else{
                    plsValue <- 0;
                }
            }
            else{
                # Here and later in the code the abs() is needed for weird cases of wrong covarMat
                plsValue <- -as.vector(log(2*pi*abs(det(covarMat)))/2 +
                                           (t(errors) %*% solve(covarMat) %*% errors) / 2);
            }
        }
        return(plsValue);
    }

    # Additive models
    if(Etype=="A"){
        # Non-intermittent data
        if(is.null(object$occurrence)){
            errors <- holdout - yForecast;
            plsValue <- densityFunction(loss, errors, covarMat);
        }
        # Intermittent data
        else{
            ot <- holdout!=0;
            pForecast <- object$occurrence$forecast;
            errors <- holdout - yForecast / pForecast;
            if(all(ot)){
                plsValue <- densityFunction(loss, errors, covarMat) + sum(log(pForecast));
            }
            else if(all(!ot)){
                plsValue <- sum(log(1-pForecast));
            }
            else{
                errors[!ot] <- 0;

                plsValue <- densityFunction(loss, errors, covarMat);
                plsValue <- plsValue + sum(log(pForecast[ot])) + sum(log(1-pForecast[!ot]));
            }
        }
    }
    # Multiplicative models
    else{
        # Non-intermittent data
        if(is.null(object$occurrence)){
            errors <- log(holdout) - log(yForecast);
            plsValue <- densityFunction(loss, errors, covarMat) - sum(log(holdout));
        }
        # Intermittent data
        else{
            ot <- holdout!=0;
            pForecast <- object$occurrence$forecast;
            errors <- log(holdout) - log(yForecast / pForecast);
            if(all(ot)){
                plsValue <- (densityFunction(loss, errors, covarMat) - sum(log(holdout)) +
                             sum(log(pForecast)));
            }
            else if(all(!ot)){
                plsValue <- sum(log(1-pForecast));
            }
            else{
                errors[!ot] <- 0;

                plsValue <- densityFunction(loss, errors, covarMat) - sum(log(holdout[ot]));
                plsValue <- plsValue + sum(log(pForecast[ot])) + sum(log(1-pForecast[!ot]));
            }
        }
    }

    return(plsValue);
}

#' @importFrom stats sigma
#' @export
sigma.smooth <- function(object, ...){
    if(!is.null(object$s2)){
        return(sqrt(object$s2));
    }
    else{
        return(NULL);
    }
}

#' @export
sigma.smooth.sim <- function(object, ...){
    return(sqrt(mean(residuals(object)^2)));
}

#### pointLik for smooth ####
#' @importFrom greybox pointLik
#' @export
pointLik.smooth <- function(object, log=TRUE, ...){
    obs <- nobs(object);
    errors <- residuals(object);
    likValues <- vector("numeric",obs);
    loss <- object$loss;

    if(errorType(object)=="M"){
        likValues <- likValues - log(actuals(object));
    }

    if(any(loss==c("MAE","MAEh","TMAE","GTMAE","MACE"))){
        likValues <- likValues + dlaplace(errors, 0, mean(abs(errors)), log=log);
    }
    else if(any(loss==c("HAM","HAMh","THAM","GTHAM","CHAM"))){
        likValues <- likValues + ds(errors, 0, mean(sqrt(abs(errors))/2), log=log);
    }
    else{
        likValues <- likValues + dnorm(errors, 0, sqrt(mean(abs(errors)^2)), log=log);
    }

    likValues <- ts(as.vector(likValues), start=start(errors), frequency=frequency(errors));

    return(likValues);
}

#' @export
pointLik.oes <- function(object, log=TRUE, ...){
    ot <- actuals(object);
    pFitted <- fitted(object);
    likValues <- vector("numeric",nobs(object));
    likValues[ot==1] <- log(pFitted[ot==1]);
    likValues[ot==0] <- log(1-pFitted[ot==0]);
    likValues <- ts(likValues, start=start(ot), frequency=frequency(ot));

    if(!log){
        likValues[] <- exp(likValues);
    }
    return(likValues);
}

#' @export
pointLik.om <- function(object, log=TRUE, ...){
    ot <- as.numeric(actuals(object));
    pFitted <- as.numeric(fitted(object));
    otLogical <- ot == 1;
    likValues <- numeric(length(ot));
    likValues[otLogical]  <- log(pFitted[otLogical]);
    likValues[!otLogical] <- log(1 - pFitted[!otLogical]);
    fittedTS <- fitted(object);
    likValues <- ts(likValues, start=start(fittedTS), frequency=frequency(fittedTS));
    if(!log){
        likValues[] <- exp(likValues);
    }
    return(likValues);
}

#### Extraction of parameters of models ####
#' @export
coef.smooth <- function(object, ...)
{
    smoothType <- smoothType(object);
    if(smoothType=="CES"){
        parameters <- c(object$a,object$b);
    }
    else if(smoothType=="ETS"){
        if(any(unlist(gregexpr("C",modelType(object)))==-1)){
            # If this was normal ETS, return values
            parameters <- c(object$persistence,object$initial,object$initialSeason,object$initialX);
        }
        else{
            # If we did combinations, we cannot return anything
            message("Combination of models was done, so there are no coefficients to return");
            parameters <- NULL;
        }
    }
    else if(smoothType=="GUM"){
        parameters <- c(object$measurement,object$transition,object$persistence,object$initial);
        names(parameters) <- c(paste0("Measurement ",c(1:length(object$measurement))),
                               paste0("Transition ",c(1:length(object$transition))),
                               paste0("Persistence ",c(1:length(object$persistence))),
                               paste0("Initial ",c(1:length(object$initial))));
    }
    else if(smoothType=="ARIMA"){
        if(any(unlist(gregexpr("combine",object$model))==-1)){
            # If this was normal ARIMA, return values
            namesConstant <- NamesMA <- NamesAR <- parameters <- NULL;
            if(any(object$AR!=0)){
                parameters <- c(parameters,object$AR);
                NamesAR <- paste(rownames(object$AR),rep(colnames(object$AR),each=nrow(object$AR)),sep=", ");
            }
            if(any(object$MA!=0)){
                parameters <- c(parameters,object$MA);
                NamesMA <- paste(rownames(object$MA),rep(colnames(object$MA),each=nrow(object$MA)),sep=", ")
            }
            if(object$constant!=0){
                parameters <- c(parameters,object$constant);
                namesConstant <- "Constant";
            }
            names(parameters) <- c(NamesAR,NamesMA,namesConstant);
            parameters <- parameters[parameters!=0];
        }
        else{
            # If we did combinations, we cannot return anything
            message("Combination of models was done, so there are no coefficients to return");
            parameters <- NULL;
        }
    }
    else if(smoothType=="SMA"){
        parameters <- object$persistence;
    }

    return(parameters);
}


#### Fitted, forecast and actual values ####
#' @export
fitted.smooth <- function(object, ...){
    return(object$fitted);
}
#' @export
fitted.smooth.forecast <- function(object, ...){
    return(fitted(object$model));
}

#' @rdname forecast.smooth
#' @export
forecast.oes <- function(object, h=10,
                         # interval=c("parametric","semiparametric","nonparametric","none"),
                         # level=0.95, side=c("both","upper","lower"),
                         ...){
    # side <- match.arg(side);
    # This correction is needed in order to reduce the level and then just use one bound
    # if(any(side==c("upper","lower"))){
    #     levelNew <- level*2-1;
    # }
    # else{
    #     levelNew <- level;
    # }

    if(is.oesg(object)){
        newModel <- oesg(actuals(object),modelA=object$modelA,modelB=object$modelB,
                         h=h,silent="all",...);
    }
    else{
        newModel <- oes_old(actuals(object),model=object,h=h,silent="all",...);
    }

    # Remove the redundant values, if they were produced
    # if(side=="upper"){
    #     newModel$lower[] <- NA;
    #     newModel$level <- level;
    # }
    # else if(side=="lower"){
    #     newModel$upper[] <- NA;
    #     newModel$level <- level;
    # }

    output <- list(model=object, method=object$model, mean=newModel$forecast,
                   forecast=newModel$forecast, interval="none"
                   # lower=newModel$lower,upper=newModel$upper,level=levelNew,
                   # interval=interval,side=side
                   );

    return(structure(output,class=c("smooth.forecast","forecast")));
}

#' @importFrom stats window
#' @importFrom greybox actuals
#' @export
actuals.smooth <- function(object, ...){
    return(window(object$y,start=start(object$y),end=end(object$fitted)));
}
#' @export
actuals.smooth.forecast <- function(object, ...){
    return(window(actuals(object$model),start=start(actuals(object$model)),end=end(fitted(object$model))));
}

#### Function extracts lags of provided model ####
#' @export
lags.default <- function(object, ...){
    return(NA);
}

#' @export
lags.ets <- function(object, ...){
    modelName <- modelType(object);
    lags <- c(1);
    if(substr(modelName,nchar(modelName),nchar(modelName))!="N"){
        lags <- c(lags,frequency(actuals(object)));
    }
    return(lags);
}

#' @export
lags.ar <- function(object, ...){
    return(1);
}

#' @export
lags.Arima <- function(object, ...){
    return(c(1,object$arma[5]));
}

#' @export
lags.smooth <- function(object, ...){
    model <- object$model;
    smoothType <- smoothType(object);
    if(!is.null(model)){
        if(smoothType=="GUM"){
            lags <- as.numeric(substring(model,unlist(gregexpr("\\[",model))+1,unlist(gregexpr("\\]",model))-1));
        }
        else if(smoothType=="ARIMA"){
            if(any(unlist(gregexpr("combine",object$model))==-1)){
                if(any(unlist(gregexpr("\\[",model))!=-1)){
                    lags <- as.numeric(substring(model,unlist(gregexpr("\\[",model))+1,unlist(gregexpr("\\]",model))-1));
                }
                else{
                    lags <- 1;
                }
            }
            else{
                warning("ARIMA was combined and we cannot extract lags anymore. Sorry!",call.=FALSE);
            }
        }
        else if(smoothType=="SMA"){
            lags <- 1;
        }
        else if(smoothType=="ETS"){
            modelName <- modelType(object);
            lags <- c(1);
            if(substr(modelName,nchar(modelName),nchar(modelName))!="N"){
                lags <- c(lags,frequency(actuals(object)));
            }
        }
        else if(smoothType=="CES"){
            modelName <- modelType(object);
            dataFreq <- frequency(actuals(object));
            if(modelName=="none"){
                lags <- c(1);
            }
            else if(modelName=="simple"){
                lags <- c(dataFreq);
            }
            else if(modelName=="partial"){
                lags <- c(1,dataFreq);
            }
            else if(modelName=="full"){
                lags <- c(1,dataFreq);
            }
            else{
                stop("Sorry, but we cannot identify the type of the provided model.",
                     call.=FALSE);
            }
        }
        else{
            lags <- NA;
        }
    }
    else{
        lags <- NA;
    }

    return(lags);
}

#' @export
lags.smooth.sim <- lags.smooth;

#### Function extracts type of error in the model: "A" or "M" ####
#' @importFrom greybox errorType
#' @export
errorType.smooth <- function(object, ...){
    smoothType <- smoothType(object);
    # ETS models
    if(smoothType=="ETS"){
        if(any(substr(modelType(object),1,1)==c("A","X"))){
            Etype <- "A";
        }
        else if(any(substr(modelType(object),1,1)==c("M","Y"))){
            Etype <- "M";
        }
        else{
            stop("Sorry, but we cannot define error type for this type of model",
                 call.=FALSE);
        }
    }
    # GUM models
    else if(smoothType=="GUM"){
        if(substr(modelName(object),1,1)=="M"){
            Etype <- "M";
        }
        else{
            Etype <- "A";
        }
    }
    # SSARIMA models
    else if(any(smoothType==c("ARIMA","SSARIMA"))){
        Etype <- "A";
    }
    # CES models
    else if(smoothType=="CES"){
        Etype <- "A";
    }
    # SMA models
    else if(smoothType=="SMA"){
        Etype <- "A";
    }
    # SMA models
    else if(smoothType=="CMA"){
        Etype <- "A";
    }
    else{
        stop(paste0("Sorry but we cannot identify error type for the model '",object$model),
             call.=FALSE);
    }
    return(Etype);
}

#' @export
errorType.smooth.sim <- errorType.smooth;

##### Function returns the modelLags from the model - internal function #####
modelLags <- function(object, ...){
    modelLags <- NA;
    if(is.msarima(object)){
        modelLags <- object$modelLags;
    }
    else if(is.adam(object)){
        modelLags <- object$lagsAll;
    }
    else{
        smoothType <- smoothType(object);
        if(smoothType=="ETS"){
            modelLags <- matrix(rep(lags(object),times=orders(object)),ncol=1);
        }
        else if(smoothType=="GUM"){
            modelLags <- matrix(rep(lags(object),times=orders(object)),ncol=1);
        }
        else if(smoothType=="ARIMA"){
            ordersARIMA <- orders(object);
            nComponents <- max(ordersARIMA$ar %*% lags(object) + ordersARIMA$i %*% lags(object),
                               ordersARIMA$ma %*% lags(object));
            modelLags <- matrix(rep(1,times=nComponents),ncol=1);
            if(is.numeric(object$constant)){
                modelLags <- rbind(modelLags,1);
            }
        }
        else if(smoothType=="CES"){
            modelLags <- matrix(rep(lags(object),times=orders(object)),ncol=1);
        }
        else if(smoothType=="SMA"){
            modelLags <- matrix(rep(1,times=orders(object)),ncol=1);
        }
    }
    return(modelLags);
}

#### Function extracts the full name of a model. For example "ETS(AAN)" ####
#' @export
modelName.default <- function(object, ...){
    return(NA);
}

#' @export
modelName.ar <- function(object, ...){
    return(paste0("AR(",object$order,")"));
}

#' @export
modelName.lm <- function(object, ...){
    return("Regression");
}

#' @export
modelName.Arima <- function(object, ...){
    ordersArma <- object$arma;
    if(all(ordersArma[c(3,4,7)]==0)){
        return(paste0("ARIMA(",paste(ordersArma[c(1,6,2)],collapse=","),")"));
    }
    else{
        return(paste0("SARIMA(",paste(ordersArma[c(1,6,2)],collapse=","),")(",
                      paste(ordersArma[c(3,7,4)],collapse=","),")[",ordersArma[5],"]"));
    }
}

#' @export
modelName.ets <- function(object, ...){
    return(object$method);
}

#' @export
modelName.forecast <- function(object, ...){
    return(object$method);
}

#' @export
modelName.smooth <- function(object, ...){
    return(object$model);
}

#### Function extracts type of model. For example "AAN" from ets ####
#' @export
modelType.default <- function(object, ...){
    return(NA);
}

#' @export
modelType.smooth <- function(object, ...){
    model <- object$model;
    smoothType <- smoothType(object);

    if(smoothType=="ETS"){
        modelType <- substring(model,unlist(gregexpr("\\(",model))+1,unlist(gregexpr("\\)",model))-1);
    }
    else if(smoothType=="CES"){
        modelType <- substring(model,unlist(gregexpr("\\(",model))+1,unlist(gregexpr("\\)",model))-1);
        if(modelType=="n"){
            modelType <- "none";
        }
        else if(modelType=="s"){
            modelType <- "simple";
        }
        else if(modelType=="p"){
            modelType <- "partial";
        }
        else{
            modelType <- "full";
        }
    }
    else{
        modelType <- NA;
    }

    return(modelType);
}

#' @export
modelType.smooth.sim <- modelType.smooth;

#' @export
modelType.oesg <- function(object, ...){
    return(modelType(object$modelA));
}

#' @export
modelType.ets <- function(object, ...){
    return(gsub(",","",substring(object$method,5,nchar(object$method)-1)));
}

#### Function extracts orders of provided model ####
#' @export
orders.default <- function(object, ...){
    return(NA);
}

#' @export
orders.smooth <- function(object, ...){
    smoothType <- smoothType(object);
    model <- object$model;
    if(!is.null(model)){
        if(smoothType=="GUM"){
            orders <- as.numeric(substring(model,unlist(gregexpr("\\[",model))-1,unlist(gregexpr("\\[",model))-1));
        }
        else if(smoothType=="ARIMA"){
            if(any(unlist(gregexpr("combine",object$model))==-1)){
                arima.orders <- paste0(c("",substring(model,unlist(gregexpr("\\(",model))+1,unlist(gregexpr("\\)",model))-1),"")
                                       ,collapse=";");
                comas <- unlist(gregexpr("\\,",arima.orders));
                semicolons <- unlist(gregexpr("\\;",arima.orders));
                ar.orders <- as.numeric(substring(arima.orders,semicolons[-length(semicolons)]+1,comas[2*(1:(length(comas)/2))-1]-1));
                i.orders <- as.numeric(substring(arima.orders,comas[2*(1:(length(comas)/2))-1]+1,comas[2*(1:(length(comas)/2))-1]+1));
                ma.orders <- as.numeric(substring(arima.orders,comas[2*(1:(length(comas)/2))]+1,semicolons[-1]-1));

                orders <- list(ar=ar.orders,i=i.orders,ma=ma.orders);
            }
            else{
                warning("ARIMA was combined and we cannot extract orders anymore. Sorry!",call.=FALSE);
            }
        }
        else if(smoothType=="SMA"){
            orders <- as.numeric(substring(model,unlist(gregexpr("\\(",model))+1,unlist(gregexpr("\\)",model))-1));
        }
        else if(smoothType=="ETS"){
            modelName <- modelType(object);
            orders <- 1;
            if(substr(modelName,2,2)!="N"){
                orders <- 2;
            }
            if(substr(modelName,nchar(modelName),nchar(modelName))!="N"){
                orders <- c(orders, 1);
            }
        }
        else if(smoothType=="CES"){
            modelName <- modelType(object);
            if(modelName=="none"){
                orders <- 2;
            }
            else if(modelName=="simple"){
                orders <- 2;
            }
            else if(modelName=="partial"){
                orders <- c(2,1);
            }
            else if(modelName=="full"){
                orders <- c(2,2);
            }
            else{
                stop("Sorry, but we cannot identify the type of the provided model.",
                     call.=FALSE);
            }
        }
        else{
            orders <- NA;
        }
    }
    else{
        orders <- NA;
    }

    return(orders);
}

#' @export
orders.smooth.sim <- orders.smooth;

#' @export
orders.ar <- function(object, ...){
    return(list(ar=object$order));
}

#' @export
orders.Arima <- function(object, ...){
    model <- object$arma;

    ar.orders <- c(model[1],model[3]);
    i.orders <-  c(model[6],model[7]);
    ma.orders <- c(model[2],model[4]);

    orders <- list(ar=ar.orders,i=i.orders,ma=ma.orders);

    return(orders);
}

#### Plots of smooth objects ####
#' Plots for the fit and states
#'
#' The function produces diagnostics plots for a \code{smooth} model
#'
#' The list of produced plots includes:
#' \enumerate{
#' \item Actuals vs Fitted values. Allows analysing, whether there are any issues in the fit.
#' Does the variability of actuals increase with the increase of fitted values? Is the relation
#' well captured? They grey line on the plot corresponds to the perfect fit of the model.
#' \item Standardised residuals vs Fitted. Plots the points and the confidence bounds
#' (red lines) for the specified confidence \code{level}. Useful for the analysis of outliers;
#' \item Studentised residuals vs Fitted. This is similar to the previous plot, but with the
#' residuals divided by the scales with the leave-one-out approach. Should be more sensitive
#' to outliers;
#' \item Absolute residuals vs Fitted. Useful for the analysis of heteroscedasticity;
#' \item Squared residuals vs Fitted - similar to (3), but with squared values;
#' \item Q-Q plot with the specified distribution. Can be used in order to see if the
#' residuals follow the assumed distribution. The type of distribution depends on the one used
#' in the estimation (see \code{distribution} parameter in \link[greybox]{alm});
#' \item ACF of the residuals. Are the residuals autocorrelated? See \link[stats]{acf} for
#' details;
#' \item Fitted over time. Plots actuals (black line), fitted values (purple line), point forecast
#' (blue line) and prediction interval (grey lines). Can be used in order to make sure that the model
#' did not miss any important events over time;
#' \item Standardised residuals vs Time. Useful if you want to see, if there is autocorrelation or
#' if there is heteroscedasticity in time. This also shows, when the outliers happen;
#' \item Studentised residuals vs Time. Similar to previous, but with studentised residuals;
#' \item PACF of the residuals. No, really, are they autocorrelated? See pacf function from stats
#' package for details;
#' \item Plot of the states of the model. It is not recommended to produce this plot together with
#' the others, because there might be several states, which would cause the creation of a different
#' canvas. In case of "msdecompose", this will produce the decomposition of the series into states
#' on a different canvas;
#' \item Absolute standardised residuals vs Fitted. Similar to the previous, but with absolute
#' values. This is more relevant to the models where scale is calculated as an absolute value of
#' something (e.g. Laplace);
#' \item Squared standardised residuals vs Fitted. This is an additional plot needed to diagnose
#' heteroscedasticity in a model with varying scale. The variance on this plot will be constant if
#' the adequate model for \code{scale} was constructed. This is more appropriate for normal and
#' the related distributions.
#' }
#' Which of the plots to produce, is specified via the \code{which} parameter.
#'
#' @param x Estimated smooth model.
#' @param which Which of the plots to produce. The possible options (see details for explanations):
#' \enumerate{
#' \item Actuals vs Fitted values;
#' \item Standardised residuals vs Fitted;
#' \item Studentised residuals vs Fitted;
#' \item Absolute residuals vs Fitted;
#' \item Squared residuals vs Fitted;
#' \item Q-Q plot with the specified distribution;
#' \item Fitted over time;
#' \item Standardised residuals vs Time;
#' \item Studentised residuals vs Time;
#' \item ACF of the residuals;
#' \item PACF of the residuals;
#' \item Plot of states of the model;
#' \item Absolute standardised residuals vs Fitted;
#' \item Squared standardised residuals vs Fitted;
#' \item ACF of the squared residuals;
#' \item PACF of the squared residuals.
#' }
#' @param level Confidence level. Defines width of confidence interval. Used in plots (2), (3), (7), (8),
#' (9), (10) and (11).
#' @param legend If \code{TRUE}, then the legend is produced on plots (2), (3) and (7).
#' @param ask Logical; if \code{TRUE}, the user is asked to press Enter before each plot.
#' @param lowess Logical; if \code{TRUE}, LOWESS lines are drawn on scatterplots, see \link[stats]{lowess}.
#' @param ... The parameters passed to the plot functions. Recommended to use with separate plots.
#' @return The function produces the number of plots, specified in the parameter \code{which}.
#'
#' @template ssAuthor
#' @seealso \link[greybox]{plot.greybox}
#' @keywords ts univar
#' @examples
#'
#' ourModel <- es(c(rnorm(50,100,10),rnorm(50,120,10)), "ANN", h=10)
#' par(mfcol=c(3,4))
#' plot(ourModel, c(1:11))
#' plot(ourModel, 12)
#'
#' @importFrom stats ppoints qqnorm qqplot qqline acf pacf lowess sd na.pass
#' @importFrom grDevices dev.interactive devAskNewPage
#' @importFrom graphics plot text
#' @importFrom greybox is.occurrence
#' @rdname plot.smooth
#' @export
plot.smooth <- function(x, which=c(1,2,4,6), level=0.95, legend=FALSE,
                        ask=prod(par("mfcol")) < length(which) && dev.interactive(),
                        lowess=TRUE, ...){
    ellipsis <- list(...);

    # Define, whether to wait for the hit of "Enter"
    if(ask){
        oask <- devAskNewPage(TRUE);
        on.exit(devAskNewPage(oask));
    }

    # 1. Fitted vs Actuals values
    plot1 <- function(x, ...){
        ellipsis <- list(...);

        # Get the actuals and the fitted values
        ellipsis$y <- as.vector(actuals(x));
        if(is.occurrence(x)){
            if(any(x$distribution==c("plogis","pnorm"))){
                ellipsis$y <- (ellipsis$y!=0)*1;
            }
        }
        ellipsis$x <- as.vector(fitted(x));

        # If this is a mixture model, remove zeroes
        if(is.occurrence(x$occurrence)){
            ellipsis$x <- ellipsis$x[ellipsis$y!=0];
            ellipsis$y <- ellipsis$y[ellipsis$y!=0];
        }

        # Remove NAs
        if(any(is.na(ellipsis$x))){
            ellipsis$y <- ellipsis$y[!is.na(ellipsis$x)];
            ellipsis$x <- ellipsis$x[!is.na(ellipsis$x)];
        }
        if(any(is.na(ellipsis$y))){
            ellipsis$x <- ellipsis$x[!is.na(ellipsis$y)];
            ellipsis$y <- ellipsis$y[!is.na(ellipsis$y)];
        }

        # Title
        if(!any(names(ellipsis)=="main")){
            ellipsis$main <- "Actuals vs Fitted";
        }
        # If type and ylab are not provided, set them...
        if(!any(names(ellipsis)=="type")){
            ellipsis$type <- "p";
        }
        if(!any(names(ellipsis)=="ylab")){
            ellipsis$ylab <- "Actuals";
        }
        if(!any(names(ellipsis)=="xlab")){
            ellipsis$xlab <- "Fitted";
        }
        # xlim and ylim
        if(!any(names(ellipsis)=="xlim")){
            ellipsis$xlim <- range(c(ellipsis$x,ellipsis$y));
        }
        if(!any(names(ellipsis)=="ylim")){
            ellipsis$ylim <- range(c(ellipsis$x,ellipsis$y));
        }

        # Start plotting
        do.call(plot,ellipsis);
        abline(a=0,b=1,col="grey",lwd=2,lty=2)
        if(lowess){
            lines(lowess(ellipsis$x, ellipsis$y), col=2);
        }
    }

    # 2 and 3: Standardised  / studentised residuals vs Fitted
    plot2 <- function(x, type="rstandard", ...){
        ellipsis <- list(...);

        ellipsis$x <- as.vector(fitted(x));
        if(type=="rstandard"){
            ellipsis$y <- as.vector(rstandard(x));
            yName <- "Standardised";
        }
        else{
            ellipsis$y <- as.vector(rstudent(x));
            yName <- "Studentised";
        }

        if(!any(names(ellipsis)=="main")){
            ellipsis$main <- paste0(yName," Residuals vs Fitted");
        }

        if(!any(names(ellipsis)=="xlab")){
            ellipsis$xlab <- "Fitted";
        }
        if(!any(names(ellipsis)=="ylab")){
            ellipsis$ylab <- paste0(yName," Residuals");
        }

        if(legend){
            if(ellipsis$x[length(ellipsis$x)]>mean(ellipsis$x)){
                legendPosition <- "bottomright";
            }
            else{
                legendPosition <- "topright";
            }
        }

        # Get the IDs of outliers and statistic
        outliers <- outlierdummy(x, level=level, type=type);
        outliersID <- outliers$id;
        statistic <- outliers$statistic;

        # Substitute zeroes with NAs if there was an occurrence
        if(is.occurrence(x$occurrence)){
            ellipsis$x[actuals(x$occurrence)==0] <- NA;
        }

        if(!any(names(ellipsis)=="ylim")){
            ellipsis$ylim <- range(c(ellipsis$y,statistic), na.rm=TRUE)*1.2;
            if(legend){
                if(legendPosition=="bottomright"){
                    ellipsis$ylim[1] <- ellipsis$ylim[1] - 0.2*diff(ellipsis$ylim);
                }
                else{
                    ellipsis$ylim[2] <- ellipsis$ylim[2] + 0.2*diff(ellipsis$ylim);
                }
            }
        }

        xRange <- range(ellipsis$x, na.rm=TRUE);
        xRange[1] <- xRange[1] - sd(ellipsis$x, na.rm=TRUE);
        xRange[2] <- xRange[2] + sd(ellipsis$x, na.rm=TRUE);

        do.call(plot,ellipsis);
        abline(h=0, col="grey", lty=2);
        polygon(c(xRange,rev(xRange)),c(statistic[1],statistic[1],statistic[2],statistic[2]),
                col="lightgrey", border=NA, density=10);
        abline(h=statistic, col=2, lty=2);
        if(length(outliersID)>0){
            points(ellipsis$x[outliersID], ellipsis$y[outliersID], pch=16);
            text(ellipsis$x[outliersID], ellipsis$y[outliersID], labels=outliersID, pos=(ellipsis$y[outliersID]>0)*2+1);
        }
        if(lowess){
            # Remove NAs
            if(any(is.na(ellipsis$x))){
                ellipsis$y <- ellipsis$y[!is.na(ellipsis$x)];
                ellipsis$x <- ellipsis$x[!is.na(ellipsis$x)];
            }
            lines(lowess(ellipsis$x, ellipsis$y), col=2);
        }

        if(legend){
            if(lowess){
                legend(legendPosition,
                       legend=c(paste0(round(level,3)*100,"% bounds"),"outside the bounds","LOWESS line"),
                       col=c(2, 1 ,2), lwd=c(1,NA,1), lty=c(2,1,1), pch=c(NA,16,NA));
            }
            else{
                legend(legendPosition,
                       legend=c(paste0(round(level,3)*100,"% bounds"),"outside the bounds"),
                       col=c(2, 1), lwd=c(1,NA), lty=c(2,1), pch=c(NA,16));
            }
        }
    }

    # 4 and 5. Fitted vs |Residuals| or Fitted vs Residuals^2
    plot3 <- function(x, type="abs", ...){
        ellipsis <- list(...);

        ellipsis$x <- as.vector(fitted(x));
        if(type=="abs"){
            ellipsis$y <- abs(as.vector(residuals(x)));
        }
        else{
            ellipsis$y <- as.vector(residuals(x))^2;
        }

        if(is.occurrence(x$occurrence)){
            ellipsis$x <- ellipsis$x[ellipsis$y!=0];
            ellipsis$y <- ellipsis$y[ellipsis$y!=0];
        }
        # Remove NAs
        if(any(is.na(ellipsis$x))){
            ellipsis$x <- ellipsis$x[!is.na(ellipsis$x)];
            ellipsis$y <- ellipsis$y[!is.na(ellipsis$y)];
        }

        if(!any(names(ellipsis)=="main")){
            if(type=="abs"){
                ellipsis$main <- "|Residuals| vs Fitted";
            }
            else{
                ellipsis$main <- "Residuals^2 vs Fitted";
            }
        }

        if(!any(names(ellipsis)=="xlab")){
            ellipsis$xlab <- "Fitted";
        }
        if(!any(names(ellipsis)=="ylab")){
            if(type=="abs"){
                ellipsis$ylab <- "|Residuals|";
            }
            else{
                ellipsis$ylab <- "Residuals^2";
            }
        }

        do.call(plot,ellipsis);
        abline(h=0, col="grey", lty=2);
        if(lowess){
            lines(lowess(ellipsis$x, ellipsis$y), col=2);
        }
    }

    # 6. Q-Q with the specified distribution
    plot4 <- function(x, ...){
        ellipsis <- list(...);

        ellipsis$y <- as.vector(residuals(x));
        if(is.occurrence(x$occurrence)){
            ellipsis$y <- ellipsis$y[actuals(x$occurrence)!=0];
        }

        if(!any(names(ellipsis)=="xlab")){
            ellipsis$xlab <- "Theoretical Quantile";
        }
        if(!any(names(ellipsis)=="ylab")){
            ellipsis$ylab <- "Actual Quantile";
        }

        if(any(x$loss==c("MAEh","TMAE","GTMAE","MACE"))){
            if(!any(names(ellipsis)=="main")){
                ellipsis$main <- "QQ-plot of Laplace distribution";
            }
            ellipsis$x <- qlaplace(ppoints(500), mu=0, scale=x$scale);

            do.call(qqplot, ellipsis);
            qqline(ellipsis$y, distribution=function(p) qlaplace(p, mu=0, scale=x$scale));
        }
        else if(any(x$loss==c("HAMh","THAM","GTHAM","CHAM"))){
            if(!any(names(ellipsis)=="main")){
                ellipsis$main <- "QQ-plot of S distribution";
            }
            ellipsis$x <- qs(ppoints(500), mu=0, scale=x$scale);

            do.call(qqplot, ellipsis);
            qqline(ellipsis$y, distribution=function(p) qs(p, mu=0, scale=x$scale));
        }
        else{
            if(!any(names(ellipsis)=="main")){
                ellipsis$main <- "QQ plot of normal distribution";
            }

            do.call(qqnorm, ellipsis);
            qqline(ellipsis$y);
        }
    }

    # 7. Basic plot over time
    plot5 <- function(x, ...){
        ellipsis <- list(...);

        ellipsis$actuals <- actuals(x);
        if(!is.null(x$holdout)){
            yHoldout <- x$holdout;
            if(is.zoo(ellipsis$actuals)){
                ellipsis$actuals <- zoo(c(as.vector(ellipsis$actuals),as.vector(yHoldout)),
                                        order.by=c(time(ellipsis$actuals),time(yHoldout)));
            }
            else{
                ellipsis$actuals <- ts(c(ellipsis$actuals,yHoldout),
                                       start=start(ellipsis$actuals),
                                       frequency=frequency(ellipsis$actuals));
            }
        }
        if(is.null(ellipsis$main)){
            ellipsis$main <- x$model;
        }
        ellipsis$forecast <- x$forecast;
        ellipsis$fitted <- fitted(x);
        ellipsis$legend <- FALSE;
        ellipsis$parReset <- FALSE;
        if(!any(x$interval==c("none","n"))){
            ellipsis$lower <- x$lower;
            ellipsis$upper <- x$upper;
            ellipsis$level <- x$level;
        }

        do.call(graphmaker, ellipsis);
    }

    # 8 and 9. Standardised / Studentised residuals vs time
    plot6 <- function(x, type="rstandard", ...){

        ellipsis <- list(...);
        if(type=="rstandard"){
            ellipsis$x <- rstandard(x);
            yName <- "Standardised";
        }
        else{
            ellipsis$x <- rstudent(x);
            yName <- "Studentised";
        }

        if(!any(names(ellipsis)=="main")){
            ellipsis$main <- paste0(yName," Residuals vs Time");
        }

        if(!any(names(ellipsis)=="xlab")){
            ellipsis$xlab <- "Time";
        }
        if(!any(names(ellipsis)=="ylab")){
            ellipsis$ylab <- paste0(yName," Residuals");
        }

        # If type and ylab are not provided, set them...
        if(!any(names(ellipsis)=="type")){
            ellipsis$type <- "l";
        }

        # Get the IDs of outliers and statistic
        outliers <- outlierdummy(x, level=level, type=type);
        outliersID <- outliers$id;
        statistic <- outliers$statistic;

        if(!any(names(ellipsis)=="ylim")){
            ellipsis$ylim <- range(c(ellipsis$x,statistic),na.rm=TRUE)*1.2;
        }

        if(legend){
            legendPosition <- "topright";
            ellipsis$ylim[2] <- ellipsis$ylim[2] + 0.2*diff(ellipsis$ylim);
            ellipsis$ylim[1] <- ellipsis$ylim[1] - 0.2*diff(ellipsis$ylim);
        }

        # Start plotting
        do.call(plot,ellipsis);
        if(is.occurrence(x$occurrence)){
            points(ellipsis$x);
        }
        if(length(outliersID)>0){
            points(time(ellipsis$x)[outliersID], ellipsis$x[outliersID], pch=16);
            text(time(ellipsis$x)[outliersID], ellipsis$x[outliersID], labels=outliersID, pos=(ellipsis$x[outliersID]>0)*2+1);
        }
        if(lowess){
            # Substitute NAs with the mean
            if(any(is.na(ellipsis$x))){
                ellipsis$x[is.na(ellipsis$x)] <- mean(ellipsis$x, na.rm=TRUE);
            }
            lines(lowess(c(1:length(ellipsis$x)),ellipsis$x), col=2);
        }
        abline(h=0, col="grey", lty=2);
        abline(h=statistic[1], col=2, lty=2);
        abline(h=statistic[2], col=2, lty=2);
        polygon(c(1:nobs(x), c(nobs(x):1)),
                c(rep(statistic[1],nobs(x)), rep(statistic[2],nobs(x))),
                col="lightgrey", border=NA, density=10);
        if(legend){
            legend(legendPosition,legend=c("Residuals",paste0(level*100,"% prediction interval")),
                   col=c(1,2), lwd=rep(1,3), lty=c(1,1,2));
        }
    }

    # 10 and 11. ACF and PACF
    plot7 <- function(x, type="acf", squared=FALSE, ...){
        ellipsis <- list(...);

        if(!any(names(ellipsis)=="main")){
            if(type=="acf"){
                if(squared){
                    ellipsis$main <- "Autocorrelation Function of Squared Residuals";
                }
                else{
                    ellipsis$main <- "Autocorrelation Function of Residuals";
                }
            }
            else{
                if(squared){
                    ellipsis$main <- "Partial Autocorrelation Function of Squared Residuals";
                }
                else{
                    ellipsis$main <- "Partial Autocorrelation Function of Residuals";
                }
            }
        }

        if(!any(names(ellipsis)=="xlab")){
            ellipsis$xlab <- "Lags";
        }
        if(!any(names(ellipsis)=="ylab")){
            if(type=="acf"){
                ellipsis$ylab <- "ACF";
            }
            else{
                ellipsis$ylab <- "PACF";
            }
        }

        if(!any(names(ellipsis)=="ylim")){
            ellipsis$ylim <- c(-1,1);
        }

        if(squared){
            if(type=="acf"){
                theValues <- acf(as.vector(residuals(x)^2), plot=FALSE, na.action=na.pass)
            }
            else{
                theValues <- pacf(as.vector(residuals(x)^2), plot=FALSE, na.action=na.pass);
            }
        }
        else{
            if(type=="acf"){
                theValues <- acf(as.vector(residuals(x)), plot=FALSE, na.action=na.pass)
            }
            else{
                theValues <- pacf(as.vector(residuals(x)), plot=FALSE, na.action=na.pass);
            }
        }
        ellipsis$x <- switch(type,
                             "acf"=theValues$acf[-1],
                             "pacf"=theValues$acf);
        statistic <- qnorm(c((1-level)/2, (1+level)/2),0,sqrt(1/nobs(x)));

        ellipsis$type <- "h"

        do.call(plot,ellipsis);
        abline(h=0, col=1, lty=1);
        abline(h=statistic, col=2, lty=2);
        if(any(ellipsis$x>statistic[2] | ellipsis$x<statistic[1])){
            outliersID <- which(ellipsis$x >statistic[2] | ellipsis$x <statistic[1]);
            points(outliersID, ellipsis$x[outliersID], pch=16);
            text(outliersID, ellipsis$x[outliersID], labels=outliersID, pos=(ellipsis$x[outliersID]>0)*2+1);
        }
    }

    # 12. Plot of states
    plot8 <- function(x, ...){
        parDefault <- par(no.readonly=TRUE);
        on.exit(par(parDefault), add=TRUE);
        smoothType <- smoothType(x);
        if(smoothType=="ETS"){
            if(any(unlist(gregexpr("C",x$model))==-1)){
                statesNames <- c("actuals",colnames(x$states),"residuals");
                x$states <- cbind(actuals(x),x$states,residuals(x));
                colnames(x$states) <- statesNames;
                if(ncol(x$states)>10){
                    message("Too many states. Plotting them on several canvases.");
                    if(is.null(ellipsis$main)){
                        ellipsisMain <- NULL;
                    }
                    else{
                        ellipsisMain <- ellipsis$main;
                    }
                    nPlots <- ceiling(ncol(x$states)/10);
                    for(i in 1:nPlots){
                        if(is.null(ellipsisMain)){
                            ellipsis$main <- paste0("States of ",x$model,", part ",i);
                        }
                        ellipsis$x <- x$states[,(1+(i-1)*10):min(i*10,ncol(x$states)),drop=FALSE];
                        do.call(plot, ellipsis);
                    }
                }
                else{
                    if(ncol(x$states)<=5){
                        ellipsis$nc <- 1;
                    }
                    if(is.null(ellipsis$main)){
                        ellipsis$main <- paste0("States of ",x$model);
                    }
                    ellipsis$x <- x$states;
                    do.call(plot, ellipsis);
                }
            }
            else{
                # If we did combinations, we cannot return anything
                message("Combination of models was done. Sorry, but there is nothing to plot.");
            }
        }
        else if(smoothType=="CMA"){
            ellipsis$actuals <- actuals(x);
            ellipsis$forecast <- x$forecast;
            ellipsis$fitted <- x$fitted;
            ellipsis$legend <- FALSE;
            ellipsis$vline <- FALSE;
            if(is.null(ellipsis$main)){
                ellipsis$main <- x$model;
            }
            do.call(graphmaker, ellipsis);
        }
        else{
            if(any(unlist(gregexpr("combine",x$model))!=-1)){
                # If we did combinations, we cannot do anything
                message("Combination of models was done. Sorry, but there is nothing to plot.");
            }
            else{
                statesNames <- c(colnames(x$states),"residuals");
                x$states <- cbind(x$states,residuals(x));
                colnames(x$states) <- statesNames;
                if(ncol(x$states)>10){
                    message("Too many states. Plotting them one by one on several graphs.");
                    if(is.null(ellipsis$main)){
                        ellipsisMain <- NULL;
                    }
                    else{
                        ellipsisMain <- ellipsis$main;
                    }
                    nPlots <- ceiling(ncol(x$states)/10);
                    for(i in 1:nPlots){
                        if(is.null(ellipsisMain)){
                            ellipsis$main <- paste0("States of ",x$model,", part ",i);
                        }
                        ellipsis$x <- x$states[,(1+(i-1)*10):min(i*10,ncol(x$states))];
                        do.call(plot, ellipsis);
                    }
                }
                else{
                    if(is.null(ellipsis$main)){
                        ellipsis$main <- paste0("States of ",x$model);
                    }
                    ellipsis$x <- x$states;
                    do.call(plot, ellipsis);
                }
            }
        }
    }

    # 13 and 14. Fitted vs (std. Residuals)^2 or Fitted vs |std. Residuals|
    plot9 <- function(x, type="abs", ...){
        ellipsis <- list(...);

        ellipsis$x <- as.vector(fitted(x));
        ellipsis$y <- as.vector(rstandard(x));
        if(any(x$distribution==c("dinvgauss","dgamma"))){
            ellipsis$y[] <- log(ellipsis$y);
        }
        if(type=="abs"){
            ellipsis$y[] <- abs(ellipsis$y);
        }
        else{
            ellipsis$y[] <- ellipsis$y^2;
        }

        if(is.occurrence(x$occurrence)){
            ellipsis$x <- ellipsis$x[actuals(x$occurrence)!=0];
            ellipsis$y <- ellipsis$y[actuals(x$occurrence)!=0];
        }
        # Remove NAs
        if(any(is.na(ellipsis$x))){
            ellipsis$x <- ellipsis$x[!is.na(ellipsis$x)];
            ellipsis$y <- ellipsis$y[!is.na(ellipsis$y)];
        }

        if(!any(names(ellipsis)=="main")){
            if(type=="abs"){
                if(any(x$distribution==c("dinvgauss","dgamma","dlnorm","dllaplace","dls","dlgnorm"))){
                    ellipsis$main <- paste0("|log(Standardised Residuals)| vs Fitted");
                }
                else{
                    ellipsis$main <- "|Standardised Residuals| vs Fitted";
                }
            }
            else{
                if(any(x$distribution==c("dinvgauss","dgamma","dlnorm","dllaplace","dls","dlgnorm"))){
                    ellipsis$main <- paste0("log(Standardised Residuals)^2 vs Fitted");
                }
                else{
                    ellipsis$main <- "Standardised Residuals^2 vs Fitted";
                }
            }
        }

        if(!any(names(ellipsis)=="xlab")){
            ellipsis$xlab <- "Fitted";
        }
        if(!any(names(ellipsis)=="ylab")){
            if(type=="abs"){
                if(any(x$distribution==c("dinvgauss","dgamma","dlnorm","dllaplace","dls","dlgnorm"))){
                    ellipsis$ylab <- "|log(Standardised Residuals)|";
                }
                else{
                    ellipsis$ylab <- "|Standardised Residuals|";
                }
            }
            else{
                if(any(x$distribution==c("dinvgauss","dgamma","dlnorm","dllaplace","dls","dlgnorm"))){
                    ellipsis$ylab <- "log(Standardised Residuals)^2";
                }
                else{
                    ellipsis$ylab <- "Standardised Residuals^2";
                }
            }
        }

        do.call(plot,ellipsis);
        abline(h=0, col="grey", lty=2);
        if(lowess){
            lines(lowess(ellipsis$x[!is.na(ellipsis$y)], ellipsis$y[!is.na(ellipsis$y)]), col=2);
        }
    }

    # Do plots
    for(i in which){
        if(any(i==1)){
            plot1(x, ...);
        }
        else if(any(i==2)){
            plot2(x, ...);
        }
        else if(any(i==3)){
            plot2(x, "rstudent", ...);
        }
        else if(any(i==4)){
            plot3(x, ...);
        }
        else if(any(i==5)){
            plot3(x, type="squared", ...);
        }
        else if(any(i==6)){
            plot4(x, ...);
        }
        else if(any(i==7)){
            plot5(x, ...);
        }
        else if(any(i==8)){
            plot6(x, ...);
        }
        else if(any(i==9)){
            plot6(x, "rstudent", ...);
        }
        else if(any(i==10)){
            plot7(x, type="acf", ...);
        }
        else if(any(i==11)){
            plot7(x, type="pacf", ...);
        }
        else if(any(i==12)){
            plot8(x, ...);
        }
        else if(any(i==13)){
            plot9(x, ...);
        }
        else if(any(i==14)){
            plot9(x, type="squared", ...);
        }
        else if(any(i==15)){
            plot7(x, type="acf", squared=TRUE, ...);
        }
        else if(any(i==16)){
            plot7(x, type="pacf", squared=TRUE, ...);
        }
    }
}

#' @export
plot.smoothC <- function(x, ...){
    graphmaker(actuals(x), x$forecast, x$fitted, x$lower, x$upper, x$level,
               main="Combined smooth forecasts");
}

#' @export
plot.smooth.sim <- function(x, ...){
    ellipsis <- list(...);
    if(is.null(ellipsis$main)){
        ellipsis$main <- x$model;
    }

    if(is.null(dim(x$data))){
        nsim <- 1;
    }
    else{
        nsim <- dim(x$data)[2];
    }

    if(nsim==1){
        if(is.null(ellipsis$ylab)){
            ellipsis$ylab <- "Data";
        }
        ellipsis$x <- x$data;
        do.call(plot, ellipsis);
    }
    else{
        randomNumber <- ceiling(runif(1,1,nsim));
        message(paste0("You have generated ",nsim," time series. Not sure which of them to plot.\n",
                       "Please use plot(ourSimulation$data[,k]) instead. Plotting randomly selected series N",randomNumber,"."));
        if(is.null(ellipsis$ylab)){
            ellipsis$ylab <- paste0("Series N",randomNumber);
        }
        ellipsis$x <- x$data[,randomNumber];
        do.call(plot, ellipsis);
    }
}

#' @method plot smooth.forecast
#' @export
plot.smooth.forecast <- function(x, ...){
    yActuals <- actuals(x$model);
    if(!is.null(x$model$holdout)){
        yActuals <- ts(c(yActuals,x$model$holdout), start=start(yActuals), frequency=frequency(yActuals));
        yActuals <- window(yActuals, start=start(yActuals), end=min(tail(time(x$mean),1),tail(time(yActuals),1)));
    }
    if(!any(x$interval==c("none","n"))){
        graphmaker(yActuals,x$mean,fitted(x$model),x$lower,x$upper,x$level,...);
    }
    else{
        graphmaker(yActuals,x$mean,fitted(x$model),...);
    }
}

#' @export
plot.oes <- function(x, which=7, ...){
    # This is needed, because diagnostics doesn't make sense in case of oes
    plot.smooth(x, which=which, ...);
}

#' @export
plot.oes.sim <- function(x, ...){
    ellipsis <- list(...);
    if(is.null(ellipsis$main)){
        ellipsis$main <- x$model;
    }

    if(is.null(dim(x$ot))){
        nsim <- 1;
    }
    else{
        nsim <- dim(x$ot)[2];
    }

    if(nsim==1){
        if(is.null(ellipsis$ylab)){
            ellipsis$ylab <- "Data";
        }
        ellipsis$x <- x$probability;
        do.call(plot, ellipsis);
    }
    else{
        randomNumber <- ceiling(runif(1,1,nsim));
        message(paste0("You have generated ",nsim," time series. Not sure which of them to plot.\n",
                       "Please use plot(ourSimulation$probability[,k]) instead. Plotting randomly selected series N",randomNumber,"."));
        if(is.null(ellipsis$ylab)){
            ellipsis$ylab <- paste0("Series N",randomNumber);
        }
        ellipsis$x <- x$probability[,randomNumber];
        do.call(plot, ellipsis);
    }
}


#### Prints of smooth ####
#' @export
print.smooth.sim <- function(x, ...){
    ellipsis <- list(...);
    if(!any(names(ellipsis)=="digits")){
        digits <- 4;
    }
    else{
        digits <- ellipsis$digits;
    }

    smoothType <- smoothType(x);
    if(is.null(dim(x$data))){
        nsim <- 1
    }
    else{
        nsim <- dim(x$data)[2]
    }

    cat(paste0("Data generated from: ",x$model,"\n"));
    cat(paste0("Number of generated series: ",nsim,"\n"));

    if(nsim==1){
        if(smoothType=="ETS"){
            cat(paste0("Persistence vector: \n"));
            xPersistence <- as.vector(x$persistence);
            names(xPersistence) <- rownames(x$persistence);
            print(round(xPersistence,digits));
            if(x$phi!=1){
                cat(paste0("Phi: ",x$phi,"\n"));
            }
            if(any(x$occurrence!=1)){
                cat(paste0("The data is produced based on an occurrence model.\n"));
            }
            cat(paste0("True likelihood: ",round(x$logLik,digits),"\n"));
        }
        else if(smoothType=="ARIMA"){
            ar.orders <- orders(x)$ar;
            i.orders <- orders(x)$i;
            ma.orders <- orders(x)$ma;
            lags <- lags(x);
            # AR terms
            if(any(ar.orders!=0)){
                ARterms <- matrix(0,max(ar.orders),sum(ar.orders!=0),
                                  dimnames=list(paste0("AR(",c(1:max(ar.orders)),")"),
                                                paste0("Lag ",lags[ar.orders!=0])));
            }
            else{
                ARterms <- matrix(0,1,1);
            }
            # Differences
            if(any(i.orders!=0)){
                Iterms <- matrix(0,1,length(i.orders),
                                 dimnames=list("I(...)",paste0("Lag ",lags)));
                Iterms[,] <- i.orders;
            }
            else{
                Iterms <- 0;
            }
            # MA terms
            if(any(ma.orders!=0)){
                MAterms <- matrix(0,max(ma.orders),sum(ma.orders!=0),
                                  dimnames=list(paste0("MA(",c(1:max(ma.orders)),")"),
                                                paste0("Lag ",lags[ma.orders!=0])));
            }
            else{
                MAterms <- matrix(0,1,1);
            }

            n.coef <- ar.coef <- ma.coef <- 0;
            ar.i <- ma.i <- 1;
            for(i in 1:length(ar.orders)){
                if(ar.orders[i]!=0){
                    ARterms[1:ar.orders[i],ar.i] <- x$arma$ar[ar.coef+(1:ar.orders[i])];
                    ar.coef <- ar.coef + ar.orders[i];
                    ar.i <- ar.i + 1;
                }
                if(ma.orders[i]!=0){
                    MAterms[1:ma.orders[i],ma.i] <- x$arma$ma[ma.coef+(1:ma.orders[i])];
                    ma.coef <- ma.coef + ma.orders[i];
                    ma.i <- ma.i + 1;
                }
            }

            if(!is.null(x$arma$ar)){
                cat(paste0("AR parameters: \n"));
                print(round(ARterms,digits));
            }
            if(!is.null(x$arma$ma)){
                cat(paste0("MA parameters: \n"));
                print(round(MAterms,digits));
            }
            if(!is.na(x$constant)){
                cat(paste0("Constant value: ",round(x$constant,digits),"\n"));
            }
            cat(paste0("True likelihood: ",round(x$logLik,digits),"\n"));
        }
        else if(smoothType=="CES"){
            cat(paste0("Smoothing parameter a: ",round(x$a,digits),"\n"));
            if(!is.null(x$b)){
                if(is.complex(x$b)){
                    cat(paste0("Smoothing parameter b: ",round(x$b,digits),"\n"));
                }
                else{
                    cat(paste0("Smoothing parameter b: ",round(x$b,digits),"\n"));
                }
            }
            cat(paste0("True likelihood: ",round(x$logLik,digits),"\n"));
        }
        else if(smoothType=="SMA"){
            cat(paste0("True likelihood: ",round(x$logLik,digits),"\n"));
        }
    }
}

#' @export
print.smooth.forecast <- function(x, ...){
    if(!any(x$interval==c("none","n"))){
        level <- x$level;
        if(level>1){
            level <- level/100;
        }
        if(x$side=="both"){
            output <- cbind(x$mean,x$lower,x$upper);
            colnames(output) <- c("Point forecast",
                                  paste0("Lower bound (",(1-level)/2*100,"%)"),
                                  paste0("Upper bound (",(1+level)/2*100,"%)"));
        }
        else if(x$side=="upper"){
            output <- cbind(x$mean,x$upper);
            colnames(output) <- c("Point forecast",
                                  paste0("Upper bound (",level*100,"%)"));
        }
        else if(x$side=="lower"){
            output <- cbind(x$mean,x$lower);
            colnames(output) <- c("Point forecast",
                                  paste0("Lower bound (",level*100,"%)"));
        }
    }
    else{
        output <- x$mean;
    }
    print(output);
}

#' @export
print.oes <- function(x, ...){
    ellipsis <- list(...);
    if(!any(names(ellipsis)=="digits")){
        digits <- 4;
    }
    else{
        digits <- ellipsis$digits;
    }

    occurrence <- x$occurrence
    if(occurrence=="general"){
        occurrence <- "General";
    }
    else if(occurrence=="direct"){
        occurrence <- "Direct probability";
    }
    else if(occurrence=="fixed"){
        occurrence <- "Fixed probability";
    }
    else if(occurrence=="inverse-odds-ratio"){
        occurrence <- "Inverse odds ratio";
    }
    else if(occurrence=="odds-ratio"){
        occurrence <- "Odds ratio";
    }
    else{
        occurrence <- "None";
    }
    ICs <- round(c(AIC(x),AICc(x),BIC(x),BICc(x)),digits);
    names(ICs) <- c("AIC","AICc","BIC","BICc");
    cat(paste0("Occurrence state space model estimated: ",occurrence,"\n"));
    if(!is.null(x$model)){
        cat(paste0("Underlying ETS model: ",x$model,"\n"));
    }
    if(!is.null(x$persistence) && (x$occurrence!="fixed")){
        cat("Smoothing parameters:\n");
        print(round(x$persistence[,1],digits));
    }
    if(!is.null(x$initial)){
        cat("Vector of initials:\n");
        print(round(x$initial,digits));
    }
    if(!is.null(sigma(x))){
        cat("\nError standard deviation: "); cat(round(sigma(x),digits));
    }
    cat("\nSample size: "); cat(nobs(x));
    cat("\nNumber of estimated parameters: "); cat(nparam(x));
    cat("\nNumber of degrees of freedom: "); cat(nobs(x)-nparam(x));
    cat("\nInformation criteria: \n");
    print(ICs);
}

#' @export
print.oes.sim <- function(x, ...){
    ellipsis <- list(...);
    if(!any(names(ellipsis)=="digits")){
        digits <- 4;
    }
    else{
        digits <- ellipsis$digits;
    }

    if(is.null(dim(x$ot))){
        nsim <- 1;
        obs <- length(x$ot)
    }
    else{
        nsim <- dim(x$ot)[2];
        obs <- dim(x$ot)[1];
    }

    cat(paste0("Data generated from: ",x$model,"\n"));
    cat(paste0("Number of generated series: ",nsim,"\n"));
    cat(paste0("Number of observations in each series: ",obs,"\n"));

    if(nsim==1){
        cat(paste0("True likelihood: ",round(x$logLik,digits),"\n"));
    }
}

#### Residuals for provided object ####
#' @export
residuals.smooth <- function(object, ...){
    if(errorType(object)=="A"){
        return(object$residuals);
    }
    else{
        return(log(1+object$residuals));
    }
}

#' @importFrom stats rstandard
#' @export
rstandard.smooth <- function(model, ...){
    obs <- nobs(model);
    df <- obs - nparam(model);

    # If this is an occurrence model, then only modify the non-zero obs
    if(is.occurrence(model$occurrence)){
        residsToGo <- (actuals(model$occurrence)!=0);
    }
    else{
        residsToGo <- rep(TRUE,obs);
    }

    errors <- residuals(model, ...);
    errors[] <- (errors - mean(errors[residsToGo], na.rm=TRUE)) / sqrt(sigma(model)^2 * obs / df);
    # Fill in values with NAs if there is occurrence model
    if(is.occurrence(model$occurrence)){
        errors[!residsToGo] <- NA;
    }
    return(errors);
}

#' @importFrom stats rstudent
#' @export
rstudent.smooth <- function(model, ...){
    obs <- nobs(model);
    df <- obs - nparam(model) - 1;
    # If this is an occurrence model, then only modify the non-zero obs
    if(is.occurrence(model$occurrence)){
        residsToGo <- (actuals(model$occurrence)!=0);
    }
    else{
        residsToGo <- rep(TRUE,obs);
    }
    rstudentised <- errors <- residuals(model, ...);
    errors[] <- errors - mean(errors, na.rm=TRUE);
    # Prepare the residuals
    if(errorType(model)=="M"){
        for(i in which(residsToGo)){
            rstudentised[i] <- errors[i] / sqrt(sum(errors[-i]^2, na.rm=TRUE) / df);
        }
    }
    else{
        for(i in which(residsToGo)){
            rstudentised[i] <- errors[i] / sqrt(sum(errors[-i]^2, na.rm=TRUE) / df);
        }
    }
    # Fill in values with NAs if there is occurrence model
    if(is.occurrence(model$occurrence)){
        rstudentised[!residsToGo] <- NA;
    }

    return(rstudentised);
}

#' @importFrom greybox outlierdummy
#' @export
outlierdummy.smooth <- function(object, level=0.999, type=c("rstandard","rstudent"), ...){
    # Function returns the matrix of dummies with outliers
    type <- match.arg(type);
    errors <- switch(type,"rstandard"=rstandard(object),"rstudent"=rstudent(object));
    statistic <- switch(object$loss,
                      "MAE"=,"MAEh"=,"MACE"=,"TMAE"=,"GTMAE"=qlaplace(c((1-level)/2, (1+level)/2), 0, 1),
                      "HAM"=,"HAMh"=,"CHAM"=,"THAM"=,"GTHAM"=qs(c((1-level)/2, (1+level)/2), 0, 1),
                      qnorm(c((1-level)/2, (1+level)/2), 0, 1));

    outliersID <- which(errors>statistic[2] | errors<statistic[1]);
    outliersNumber <- length(outliersID);
    if(outliersNumber>0){
        outliers <- ts(matrix(0, nobs(object), outliersNumber,
                              dimnames=list(NULL,
                                            paste0("outlier",c(1:outliersNumber)))),
                       start=start(actuals(object)), frequency=frequency(actuals(object)));
        outliers[cbind(outliersID,c(1:outliersNumber))] <- 1;
    }
    else{
        outliers <- NULL;
    }

    return(structure(list(outliers=outliers, statistic=statistic, id=outliersID,
                          level=level, type=type),
                     class="outlierdummy"));
}

#### Simulate data using provided object ####
#' @importFrom utils tail
#' @export
simulate.smooth <- function(object, nsim=1, seed=NULL, obs=NULL, ...){
    ellipsis <- list(...);
    smoothType <- smoothType(object);
    if(is.null(obs)){
        obs <- nobs(object);
    }
    if(!is.null(seed)){
        set.seed(seed);
    }

    # Start a list of arguments
    args <- vector("list",0);

    loss <- object$loss;
    if(any(loss==c("MAE","MAEh","TMAE","GTMAE","MACE"))){
        randomizer <- "rlaplace";
        if(!is.null(ellipsis$mu)){
            args$mu <- ellipsis$mu;
        }
        else{
            args$mu <- 0;
        }

        if(!is.null(ellipsis$scale)){
            args$scale <- ellipsis$scale;
        }
        else{
            args$scale <- mean(abs(residuals(object)));
        }
    }
    else if(any(loss==c("HAM","HAMh","THAM","GTHAM","CHAM"))){
        randomizer <- "rs";
        if(!is.null(ellipsis$mu)){
            args$mu <- ellipsis$mu;
        }
        else{
            args$mu <- 0;
        }

        if(!is.null(ellipsis$scale)){
            args$scale <- ellipsis$scale;
        }
        else{
            args$scale <- mean(sqrt(abs(residuals(object))));
        }
    }
    else{
        randomizer <- "rnorm";
        if(!is.null(ellipsis$mean)){
            args$mean <- ellipsis$mean;
        }
        else{
            args$mean <- 0;
        }

        if(!is.null(ellipsis$sd)){
            args$sd <- ellipsis$sd;
        }
        else{
            args$sd <- sigma(object);
        }
    }
    args$randomizer <- randomizer;
    args$frequency <- frequency(actuals(object));
    args$obs <- obs;
    args$nsim <- nsim;
    args$initial <- object$initial;
    # If this is an occurrence model, use the fitted values for the probabilities
    if(is.occurrence(object$occurrence)){
        args$probability <- fitted(object$occurrence);
    }
    else{
        args$probability <- 1;
    }

    if(smoothType=="ETS"){
        model <- modelType(object);
        if(any(unlist(gregexpr("C",model))==-1)){
            args$model <- model;
            args$phi <- object$phi;
            args$persistence <- object$persistence;
            if(is.list(args$initial)){
                args$initialSeason <- object$initial$seasonal;
                args$initial <- unlist(args$initial[c("level","trend")]);
            }

            if(errorType(object)=="M" && args$mean==0){
                args$mean <- 1;
            }

            simulatedData <- do.call("sim.es",args);
        }
        else{
            message("Sorry, but we cannot simulate data from combined model.");
            simulatedData <- NA;
        }
    }
    else if(smoothType=="ARIMA"){
        if(any(unlist(gregexpr("combine",object$model))==-1)){
            args$orders <- orders(object);
            args$lags <- lags(object);
            args$AR <- object$AR;
            args$MA <- object$MA;
            args$constant <- object$constant;

            simulatedData <- do.call("sim.ssarima",args);
        }
        else{
            message("Sorry, but we cannot simulate data from combined model.");
            simulatedData <- NA;
        }
    }
    else if(smoothType=="CES"){
        args$seasonality <- modelType(object);
        args$a <- object$a;
        args$b <- object$b;

        simulatedData <- do.call("sim.ces",args);
    }
    else if(smoothType=="GUM"){
        args$orders <- orders(object);
        args$lags <- lags(object);
        args$measurement <- object$measurement;
        args$transition <- object$transition;
        args$persistence <- object$persistence;

        simulatedData <- do.call("sim.gum",args);
    }
    else if(smoothType=="SMA"){
        args$order <- orders(object)[1];

        simulatedData <- do.call("sim.sma",args);
    }
    else{
        model <- substring(object$model,1,unlist(gregexpr("\\(",object$model))[1]-1);
        message(paste0("Sorry, but simulate is not yet available for the model ",model,"."));
        simulatedData <- NA;
    }
    return(simulatedData);
}

#### Type of smooth model. Internal function ####
smoothType <- function(object, ...){
    if(!is.list(object$model)){
        if(gregexpr("ETS",object$model)!=-1){
            smoothType <- "ETS";
        }
        else if(gregexpr("CES",object$model)!=-1){
            smoothType <- "CES";
        }
        else if(gregexpr("SSARIMA",object$model)!=-1){
            smoothType <- "SSARIMA";
        }
        else if(gregexpr("ARIMA",object$model)!=-1){
            smoothType <- "ARIMA";
        }
        else if(gregexpr("GUM",object$model)!=-1){
            smoothType <- "GUM";
        }
        else if(gregexpr("SMA",object$model)!=-1){
            smoothType <- "SMA";
        }
        else if(gregexpr("CMA",object$model)!=-1){
            smoothType <- "CMA";
        }
        else if(gregexpr("SpARMA",object$model)!=-1){
            smoothType <- "SpARMA";
        }
        else if(gregexpr("Constant",object$model)!=-1){
            smoothType <- "ARIMA";
        }
        else{
            smoothType <- "none";
        }
    }
    else{
        smoothType <- "smoothCombine";
    }

    return(smoothType);
}

#### Summary of objects ####
#' @export
summary.smooth <- function(object, ...){
    print(object);
}

#' @export
summary.smooth.forecast <- function(object, ...){
    print(object);
}


#### Accuracy measures ####

#' @importFrom generics accuracy
#' @export
generics::accuracy

#' Error measures for an estimated model
#'
#' Function produces error measures for the provided object and the holdout values of the
#' response variable. Note that instead of parameters \code{x}, \code{test}, the function
#' accepts the vector of values in \code{holdout}. Also, the parameters \code{d} and \code{D}
#' are not supported - MASE is always calculated via division by first differences.
#'
#' The function is a wrapper for the \link[greybox]{measures} function and is implemented
#' for convenience.
#'
#' @template ssAuthor
#'
#' @param object The estimated model or a forecast from the estimated model generated via
#' either \code{predict()} or \code{forecast()} functions.
#' @param holdout The vector of values of the response variable in the holdout (test) set.
#' If not provided, then the function will return the in-sample error measures. If the
#' \code{holdout=TRUE} parameter was used in the estimation of a model, the holdout values
#' will be extracted automatically.
#' @param ... Other variables passed to the \code{forecast()} function (e.g. \code{newdata}).
#' @examples
#'
#' y <- rnorm(100, 100, 10)
#' ourModel <- adam(y, holdout=TRUE, h=10)
#' accuracy(ourModel)
#'
#' @rdname accuracy
#' @export
accuracy.smooth <- function(object, holdout=NULL, ...){
    if(is.null(object$holdout)){
        if(is.null(holdout)){
            return(measures(actuals(object), fitted(object), actuals(object)));
        }
        else{
            h <- length(holdout);
            return(measures(holdout, forecast(object, h=h, ...)$mean, actuals(object)));
        }
    }
    else{
        return(object$accuracy);
    }
}

#' @rdname accuracy
#' @export
accuracy.smooth.forecast <- function(object, holdout=NULL, ...){
    if(is.null(holdout)){
        if(is.null(object$model$holdout)){
            return(measures(actuals(object), fitted(object), actuals(object)));
        }
        else{
            return(object$model$accuracy);
        }
    }
    else{
        h <- length(holdout);
        return(measures(holdout, object$mean, actuals(object)));
    }
}

# Function detects the palette and sets its own colours if the default one is used
paletteDetector <- function(colours){
    # If the default palette is used, define the new one in the provided colours
    paletteBasic <- palette();
    palette("default");
    paletteDefault <- palette();
    if(all(paletteBasic %in% paletteDefault) &&
       all(paletteBasic==paletteDefault)){
        paletteBasic <- colours;
    }
    else{
        palette(paletteBasic);
    }
    return(paletteBasic);
}
