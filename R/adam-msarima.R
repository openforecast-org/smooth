#' Multiple Seasonal ARIMA
#'
#' Function constructs Multiple Seasonal State Space ARIMA, estimating AR, MA
#' terms and initial states. It is a wrapper of \link[smooth]{adam} function.
#'
#' The model, implemented in this function differs from the one in
#' \link[smooth]{ssarima} function (Svetunkov & Boylan, 2019), but it is more
#' efficient and better fitting the data (which might be a limitation).
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
#' State space form (based by Snyder, 1985, but in a slightly different
#' formulation):
#'
#' \eqn{y_{t} = o_{t} (w' v_{t-l} + x_t a_{t-1} + \epsilon_{t})}
#'
#' \eqn{v_{t} = F v_{t-l} + g \epsilon_{t}}
#'
#' \eqn{a_{t} = F_{X} a_{t-1} + g_{X} \epsilon_{t} / x_{t}}
#'
#' Where \eqn{o_{t}} is the Bernoulli distributed random variable (in case of
#' normal data equal to 1), \eqn{v_{t}} is the state vector (defined based on
#' \code{orders}) and \eqn{l} is the vector of \code{lags}, \eqn{x_t} is the
#' vector of exogenous parameters. \eqn{w} is the \code{measurement} vector,
#' \eqn{F} is the \code{transition} matrix, \eqn{g} is the \code{persistence}
#' vector, \eqn{a_t} is the vector of parameters for exogenous variables,
#' \eqn{F_{X}} is the \code{transitionX} matrix and \eqn{g_{X}} is the
#' \code{persistenceX} matrix. The main difference from \link[smooth]{ssarima}
#' function is that this implementation skips zero polynomials, substantially
#' decreasing the dimension of the transition matrix. As a result, this
#' function works faster than \link[smooth]{ssarima} on high frequency data,
#' and it is more accurate.
#'
#' Due to the flexibility of the model, multiple seasonalities can be used. For
#' example, something crazy like this can be constructed:
#' SARIMA(1,1,1)(0,1,1)[24](2,0,1)[24*7](0,0,1)[24*30], but the estimation may
#' take some time... Still this should be estimated in finite time (not like
#' with \code{ssarima}).
#'
#' The \code{auto.msarima} function constructs several ARIMA models in Single
#' Source of Error state space form based on \code{adam} function (see
#' \link[smooth]{adam} documentation) and selects the best one based on the
#' selected information criterion.
#'
#' For some additional details see the vignettes: \code{vignette("adam","smooth")}
#' and \code{vignette("ssarima","smooth")}
#'
#' @template ssBasicParam
#' @template ssAdvancedParam
#' @template ssXregParam
#' @template ssAuthor
#' @template ssKeywords
#'
#' @template ADAMInitial
#'
#' @template ssGeneralRef
#' @template ssARIMARef
#'
#' @param orders List of orders, containing vector variables \code{ar},
#' \code{i} and \code{ma}. Example:
#' \code{orders=list(ar=c(1,2),i=c(1),ma=c(1,1,1))}. If a variable is not
#' provided in the list, then it is assumed to be equal to zero. At least one
#' variable should have the same length as \code{lags}. Another option is to
#' specify orders as a vector of a form \code{orders=c(p,d,q)}. The non-seasonal
#' ARIMA(p,d,q) is constructed in this case.
#' For \code{auto.msarima} this is the list of maximum orders to check,
#' containing vector variables \code{ar}, \code{i} and \code{ma}. If a variable
#' is not provided in the list, then it is assumed to be equal to zero. At least
#' one variable should have the same length as \code{lags}.
#' @param lags Defines lags for the corresponding orders (see examples above).
#' The length of \code{lags} must correspond to the length of either \code{ar},
#' \code{i} or \code{ma} in \code{orders} variable. There is no restrictions on
#' the length of \code{lags} vector. It is recommended to order \code{lags}
#' ascending.
#' The orders are set by a user. If you want the automatic order selection,
#' then use \link[smooth]{auto.msarima} function instead.
#' @param bounds What type of bounds to use in the model estimation. The first
#' letter can be used instead of the whole word. \code{"admissible"} guarantee
#' that the model is stable. \code{"usual"} are not supported due to restrictions
#' in \code{adam()}.
#' @param constant If \code{TRUE}, constant term is included in the model. Can
#' also be a number (constant value). For \code{auto.msarima}, if \code{NULL},
#' then the function will check if constant is needed.
#' @param arma Either the named list or a vector with AR / MA parameters ordered lag-wise.
#' The number of elements should correspond to the specified orders e.g.
#' \code{orders=list(ar=c(1,1),ma=c(1,1)), lags=c(1,4), arma=list(ar=c(0.9,0.8),ma=c(-0.3,0.3))}
#' @param model Previously estimated MSARIMA model.
#' @param initial Can be either character or a vector of initial states.
#' If it is character, then it can be \code{"optimal"}, meaning that all initial
#' states are optimised, or \code{"backcasting"}, meaning that the initials of
#' dynamic part of the model are produced using backcasting procedure (advised
#' for data with high frequency). In the latter case, the parameters of the
#' explanatory variables are optimised. This is recommended for ARIMAX
#' model. Alternatively, you can set \code{initial="complete"} backcasting,
#' which means that all states (including explanatory variables) are initialised
#' via backcasting.
#' @param ...  Other non-documented parameters. see \link[smooth]{adam} for
#' details.
#'
#' \code{FI=TRUE} will make the function produce Fisher Information matrix,
#' which then can be used to calculated variances of parameters of the model.
#'
#' @return Object of class "adam" is returned. It contains the list of the
#' following values:
#'
#' \itemize{
#' \item \code{model} - the name of the estimated model.
#' \item \code{timeElapsed} - time elapsed for the construction of the model.
#' \item \code{states} - the matrix of the fuzzy components of msarima, where
#' \code{rows} correspond to time and \code{cols} to states.
#' \item \code{transition} - matrix F.
#' \item \code{persistence} - the persistence vector. This is the place, where
#' smoothing parameters live.
#' \item \code{measurement} - measurement vector of the model.
#' \item \code{arma} - list of AR/MA parameters
#' \item \code{constant} - the value of the constant term.
#' \item \code{initialType} - Type of the initial values used.
#' \item \code{initial} - the initial values of the state vector (extracted
#' from \code{states}).
#' \item \code{nParam} - table with the number of estimated / provided parameters.
#' If a previous model was reused, then its initials are reused and the number of
#' provided parameters will take this into account.
#' \item \code{fitted} - the fitted values.
#' \item \code{forecast} - the point forecast.
#' \item \code{lower} - the lower bound of prediction interval. When
#' \code{interval="none"} then NA is returned.
#' \item \code{upper} - the higher bound of prediction interval. When
#' \code{interval="none"} then NA is returned.
#' \item \code{residuals} - the residuals of the estimated model.
#' \item \code{errors} - The matrix of 1 to h steps ahead errors. Only returned when the
#' multistep losses are used and semiparametric interval is needed.
#' \item \code{s2} - variance of the residuals (taking degrees of freedom into
#' account).
#' \item \code{interval} - type of interval asked by user.
#' \item \code{level} - confidence level for interval.
#' \item \code{cumulative} - whether the produced forecast was cumulative or not.
#' \item \code{y} - the original data.
#' \item \code{holdout} - the holdout part of the original data.
#' \item \code{xreg} - provided vector or matrix of exogenous variables. If
#' \code{regressors="s"}, then this value will contain only selected exogenous
#' variables.
#' \item \code{initialX} - initial values for parameters of exogenous
#' variables.
#' \item \code{ICs} - values of information criteria of the model. Includes
#' AIC, AICc, BIC and BICc.
#' \item \code{logLik} - log-likelihood of the function.
#' \item \code{lossValue} - Cost function value.
#' \item \code{loss} - Type of loss function used in the estimation.
#' \item \code{FI} - Fisher Information. Equal to NULL if \code{FI=FALSE}
#' or when \code{FI} is not provided at all.
#' \item \code{accuracy} - vector of accuracy measures for the holdout sample.
#' In case of non-intermittent data includes: MPE, MAPE, SMAPE, MASE, sMAE,
#' RelMAE, sMSE and Bias coefficient (based on complex numbers). In case of
#' intermittent data the set of errors will be: sMSE, sPIS, sCE (scaled
#' cumulative error) and Bias coefficient. This is available only when
#' \code{holdout=TRUE}.
#' \item \code{B} - the vector of all the estimated parameters.
#' }
#'
#' @seealso \code{\link[smooth]{adam}, \link[smooth]{orders},
#' \link[smooth]{es}, \link[smooth]{auto.ssarima}}
#'
#' @examples
#'
#' x <- rnorm(118,100,3)
#'
#' # The simple call of ARIMA(1,1,1):
#' ourModel <- msarima(x,orders=c(1,1,1),lags=1,h=18,holdout=TRUE)
#'
#' # Example of SARIMA(2,0,0)(1,0,0)[4]
#' msarima(x,orders=list(ar=c(2,1)),lags=c(1,4),h=18,holdout=TRUE)
#'
#' # SARIMA of a peculiar order on AirPassengers data
#' ourModel <- msarima(AirPassengers,orders=list(ar=c(1,0,3),i=c(1,0,1),ma=c(0,1,2)),
#'                     lags=c(1,6,12),h=10,holdout=TRUE)
#'
#' # ARIMA(1,1,1) with Mean Squared Trace Forecast Error
#' msarima(x,orders=c(1,1,1),lags=1,h=18,holdout=TRUE,loss="TMSE")
#'
#' plot(forecast(ourModel, h=18, interval="approximate"))
#'
#' @rdname msarima
#' @export
msarima <- function(y, orders=list(ar=c(0),i=c(1),ma=c(1)), lags=c(1),
                    constant=FALSE, arma=NULL, model=NULL,
                    initial=c("backcasting","optimal","two-stage","complete","gradient"),
                    ic=c("AICc","AIC","BIC","BICc"),
                    loss=c("likelihood","MSE","MAE","HAM","MSEh","TMSE","GTMSE","MSCE","GPL"),
                    h=0, holdout=FALSE,
                    bounds=c("usual","admissible","none"),
                    silent=TRUE,
                    xreg=NULL, regressors=c("use","select","adapt"), initialX=NULL,
                    ...){
    # Copyright (C) 2022 - Inf  Ivan Svetunkov

    # Start measuring the time of calculations
    startTime <- Sys.time();
    cl <- match.call();
    env <- environment();
    ellipsis <- list(...);

    # Check if the simulated thing is provided
    if(is.smooth.sim(y)){
        if(smoothType(y)=="ARIMA"){
            model <- y;
            y <- y$data;
        }
    }
    else if(is.smooth(y)){
        model <- y;
        y <- y$y;
    }

    # If a previous model provided as a model, write down the variables
    if(!is.null(model)){
        if(is.null(model$model)){
            stop("The provided model is not ARIMA.",call.=FALSE);
        }
        else if(smoothType(model)!="ARIMA"){
            stop("The provided model is not ARIMA.",call.=FALSE);
        }

# If this is a normal ARIMA, do things
        if(any(unlist(gregexpr("combine",model$model))==-1)){

            return(adam(data=y, model=model,
                        orders=orders, lags=lags, constant=constant,
                        arma=arma,
                        loss=loss, h=h, holdout=holdout, initial=initial,
                        ic=ic, bounds=bounds[1], distribution="dnorm",
                        silent=silent, regressors=regressors[1], environment=env, ...));
        }
        else{
            stop("The provided model is a combination of ARIMAs. We cannot fit that.",call.=FALSE);
        }
    }

    # Fix lags and orders if lags=1 was dropped
    if(length(lags)==1 && lags>1){
        lags <- c(1,lags);
        if(is.list(orders)){
            if(all(sapply(orders,length)==1)){
                for(i in 1:length(orders)){
                    orders[[i]] <- c(0,orders[[i]]);
                }
            }
        }
        else{
            orders <- list(ar=c(0,orders[1]),i=c(0,orders[2]),ma=c(0,orders[3]));
        }
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

    initialValue <- vector("list",2);
    names(initialValue) <- c("arima","xreg");
    # Prepare initials if they are numeric
    if(is.numeric(initial)){
        initialValue$arima <- initial;
    }
    else{
        initialValue$arma <- NULL;
    }
    if(!is.null(initialX)){
        initialValue$xreg <- initialX;
    }
    else{
        initialValue$xreg <- NULL;
    }
    if(all(sapply(initialValue, is.null))){
        initialValue <- initial[1];
    }

    # Warnings about the interval and cumulative
    if(!is.null(ellipsis$interval) && ellipsis$interval!="none"){
        warning("Parameter \"interval\" is no longer supported in msarima(). ",
                "Please use forecast() method to produce prediction interval.")
    }

    if(!is.null(ellipsis$cumulative) && ellipsis$cumulative!="none"){
        warning("Parameter \"cumulative\" is no longer supported in msarima(). ",
                "Please use forecast() method to produce cumulative values.")
    }

    ourModel <- adam(data=data, model="NNN",
                     orders=orders, lags=lags, constant=constant,
                     arma=arma,
                     loss=loss, h=h, holdout=holdout, initial=initialValue,
                     ic=ic, bounds=bounds[1], distribution="dnorm",
                     silent=silent, regressors=regressors[1], environment=env, ...);
    ourModel$call <- cl;
    ourModel$timeElapsed=Sys.time()-startTime;

    return(ourModel);
}
