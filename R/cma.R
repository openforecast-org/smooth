#' Centered Moving Average
#'
#' Function constructs centered moving average based on state space SMA
#'
#' If the order is odd, then the function constructs SMA(order) and
#' shifts it back in time. Otherwise an AR(order+1) model is constructed
#' with the preset parameters:
#'
#' \deqn{phi_i = {0.5,1,1,...,0.5} / order}
#'
#' This then corresponds to the centered MA with 0.5 weight for the
#' first observation and 0.5 weight for an additional one. e.g. if this is
#' monthly data and we use order=12, then half of the first January and
#' half of the new one is taken.
#'
#' This is not a forecasting tool. This is supposed to smooth the time
#' series in order to find trend. So don't expect any forecasts from this
#' function!
#'
#' @template ssAuthor
#' @template ssKeywords
#'
#' @template smoothRef
#'
#' @param y Vector or ts object, containing data needed to be smoothed.
#' @param order Order of centered moving average. If \code{NULL}, then the
#' function will try to select order of SMA based on information criteria.
#' See \link[smooth]{sma} for details.
#' @param silent If \code{TRUE}, then plot is not produced. Otherwise, there
#' is a plot...
#' @param ... Nothing. Needed only for the transition to the new name of variables.
#' @return Object of class "smooth" is returned. It contains the list of the
#' following values:
#'
#' \itemize{
#' \item \code{model} - the name of the estimated model.
#' \item \code{timeElapsed} - time elapsed for the construction of the model.
#' \item \code{order} - order of the moving average.
#' \item \code{nParam} - table with the number of estimated / provided parameters.
#' If a previous model was reused, then its initials are reused and the number of
#' provided parameters will take this into account.
#' \item \code{fitted} - the fitted values, shifted in time.
#' \item \code{forecast} - NAs, because this function does not produce forecasts.
#' \item \code{residuals} - the residuals of the SMA / AR model.
#' \item \code{s2} - variance of the residuals (taking degrees of freedom into
#' account) of the SMA / AR model.
#' \item \code{y} - the original data.
#' \item \code{ICs} - values of information criteria from the respective SMA or
#' AR model. Includes AIC, AICc, BIC and BICc.
#' \item \code{logLik} - log-likelihood of the SMA / AR model.
#' \item \code{lossValue} - Cost function value (for the SMA / AR model).
#' \item \code{loss} - Type of loss function used in the estimation.
#' }
#'
#' @seealso \code{\link[smooth]{es}, \link[smooth]{ssarima}}
#'
#' @examples
#'
#' # CMA of specific order
#' ourModel <- cma(rnorm(118,100,3),order=12)
#'
#' # CMA of arbitrary order
#' ourModel <- cma(rnorm(118,100,3))
#'
#' summary(ourModel)
#'
#' @export cma
cma <- function(y, order=NULL, silent=TRUE, ...){

# Start measuring the time of calculations
    startTime <- Sys.time();

    cl <- match.call();

    holdout <- FALSE;
    h <- 0;

    ##### data #####
    if(any(is.smooth.sim(y))){
        y <- y$data;
    }
    else if(any(class(y)=="Mdata")){
        y <- ts(c(y$x,y$xx),start=start(y$x),frequency=frequency(y$x));
    }

    if(!is.numeric(y)){
        stop("The provided data is not a vector or ts object! Can't construct any model!", call.=FALSE);
    }
    if(!is.null(ncol(y))){
        if(ncol(y)>1){
            stop("The provided data is not a vector! Can't construct any model!", call.=FALSE);
        }
    }
    # Check the data for NAs
    if(any(is.na(y))){
        if(!silent){
            warning("Data contains NAs. These observations will be substituted by zeroes.",call.=FALSE);
        }
        y[is.na(y)] <- 0;
    }

    # Define obs, the number of observations of in-sample
    obsInSample <- length(y) - holdout*h;

    # Define obsAll, the overal number of observations (in-sample + holdout)
    obsAll <- length(y) + (1 - holdout)*h;

    # If obsInSample is negative, this means that we can't do anything...
    if(obsInSample<=0){
        stop("Not enough observations in sample.",call.=FALSE);
    }
    # Define the actual values
    datafreq <- frequency(y);
    dataStart <- start(y);
    yInSample <- ts(y[1:obsInSample], start=dataStart, frequency=datafreq);

    # Order of the model
    if(!is.null(order)){
        if(!is.numeric(order) || length(order)!=1){
            stop("The order should be a single number.",call.=FALSE);
        }
        if(order<1){
            stop("The order of the model must be a positive number.",call.=FALSE);
        }
        if(obsInSample < order){
            stop("Sorry, but we don't have enough observations for that order.",call.=FALSE);
        }
        orderSelect <- FALSE;
    }
    else{
        orderSelect <- TRUE;
    }

    if(orderSelect){
        order <- orders(sma(yInSample))[1];
    }

    if((order %% 2)!=0 || (order==obsInSample)){
        smaModel <- sma(yInSample, order=order, h=order, holdout=FALSE, silent=TRUE);
        yFitted <- c(smaModel$fitted[-c(1:((order+1)/2))],smaModel$forecast);
        logLik <- smaModel$logLik;
        errors <- residuals(smaModel);
    }
    else{
        ssarimaModel <- msarima(yInSample, orders=c(order+1,0,0), arma=list(ar=c(0.5,rep(1,order-1),0.5)/order),
                                h=order, holdout=FALSE, silent=TRUE, initial="backcasting");
        # The AR(order+1) fitted value at t is centred at t-order/2-1
        yFitted <- c(ssarimaModel$fitted[-c(1:(order/2+1))],ssarimaModel$forecast);
        smaModel <- sma(yInSample, order=1, h=order, holdout=FALSE, silent=TRUE);
        logLik <- ssarimaModel$logLik;
        errors <- residuals(ssarimaModel);
    }
    yForecast <- ts(NA, start=start(smaModel$forecast), frequency=datafreq);
    yFitted <- ts(yFitted[1:obsInSample], start=dataStart, frequency=datafreq);
    modelname <- paste0("CMA(",order,")");
    nParam <- smaModel$nParam;
    s2 <- sum(errors^2)/(obsInSample - 2);
    cfObjective <- mean(errors^2);

    model <- structure(list(model=modelname,timeElapsed=Sys.time()-startTime,
                            order=order, nParam=nParam, distribution="dnorm",
                            fitted=yFitted,forecast=yForecast,residuals=errors,s2=s2,
                            data=smaModel$data, formula=smaModel$formula, call=cl,
                            initialType=smaModel$initialType,
                            logLik=logLik,lossValue=cfObjective,loss="MSE"),
                       class=c("adam","smooth"));

    ICs <- c(AIC(model),AICc(model),BIC(model),BICc(model));
    names(ICs) <- c("AIC","AICc","BIC","BICc");
    model$ICs <- ICs;

    if(!silent){
        plot(model, 7);
    }

    return(model);
}
