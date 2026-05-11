#' Simple Moving Average
#'
#' Function constructs state space simple moving average of predefined order
#'
#' The function constructs AR model in the Single Source of Error state space form
#' based on the idea that:
#'
#' \eqn{y_{t} = \frac{1}{n} \sum_{j=1}^n y_{t-j}}
#'
#' which is AR(n) process, that can be modelled using:
#'
#' \eqn{y_{t} = w' v_{t-1} + \epsilon_{t}}
#'
#' \eqn{v_{t} = F v_{t-1} + g \epsilon_{t}}
#'
#' Where \eqn{v_{t}} is a state vector.
#'
#' For some more information about the model and its implementation, see the
#' vignette: \code{vignette("sma","smooth")}
#'
#' @template ssBasicParam
#' @template ssAuthor
#' @template ssKeywords
#'
#' @template smoothRef
#'
#' @param order Order of simple moving average. If \code{NULL}, then it is
#' selected automatically using information criteria.
#' @param fast if \code{TRUE}, then the modified Ternary search is used to
#' find the optimal order of the model. This does not guarantee the optimal
#' solution, but gives a reasonable one (local minimum).
#' @param ...  Other non-documented parameters.  For example parameter
#' \code{model} can accept a previously estimated SMA model and use its
#' parameters.
#' @return Object of class "smooth" is returned. It contains the list of the
#' following values:
#'
#' \itemize{
#' \item \code{model} - the name of the estimated model.
#' \item \code{timeElapsed} - time elapsed for the construction of the model.
#' \item \code{states} - the matrix of the fuzzy components of ssarima, where
#' \code{rows} correspond to time and \code{cols} to states.
#' \item \code{transition} - matrix F.
#' \item \code{persistence} - the persistence vector. This is the place, where
#' smoothing parameters live.
#' \item \code{measurement} - measurement vector of the model.
#' \item \code{order} - order of moving average.
#' \item \code{initial} - Initial state vector values.
#' \item \code{initialType} - Type of initial values used.
#' \item \code{nParam} - table with the number of estimated / provided parameters.
#' If a previous model was reused, then its initials are reused and the number of
#' provided parameters will take this into account.
#' \item \code{fitted} - the fitted values.
#' \item \code{forecast} - the point forecast.
#' \item \code{lower} - the lower bound of prediction interval. When
#' \code{interval=FALSE} then NA is returned.
#' \item \code{upper} - the higher bound of prediction interval. When
#' \code{interval=FALSE} then NA is returned.
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
#' \item \code{ICs} - values of information criteria of the model. Includes AIC,
#' AICc, BIC and BICc.
#' \item \code{logLik} - log-likelihood of the function.
#' \item \code{lossValue} - Cost function value.
#' \item \code{loss} - Type of loss function used in the estimation.
#' \item \code{accuracy} - vector of accuracy measures for the
#' holdout sample. Includes: MPE, MAPE, SMAPE, MASE, sMAE, RelMAE, sMSE and
#' Bias coefficient (based on complex numbers). This is available only when
#' \code{holdout=TRUE}.
#' }
#'
#' @references \itemize{
#' \item Svetunkov, I., & Petropoulos, F. (2017). Old dog, new tricks: a
#' modelling view of simple moving averages. International Journal of
#' Production Research, 7543(January), 1-14.
#' \doi{10.1080/00207543.2017.1380326}
#' }
#'
#' @seealso \code{\link[stats]{filter}, \link[smooth]{adam}, \link[smooth]{msarima}}
#'
#' @examples
#'
#' # SMA of specific order
#' ourModel <- sma(rnorm(118,100,3), order=12, h=18, holdout=TRUE)
#'
#' # SMA of arbitrary order
#' ourModel <- sma(rnorm(118,100,3), h=18, holdout=TRUE)
#'
#' plot(forecast(ourModel, h=18, interval="empirical"))
#'
#' @rdname sma
#' @export
sma <- function(y, order=NULL, ic=c("AICc","AIC","BIC","BICc"),
                h=10, holdout=FALSE, silent=TRUE, fast=TRUE,
                ...){
# Function constructs simple moving average in state space model

#    Copyright (C) 2022  Ivan Svetunkov

# Start measuring the time of calculations
    startTime <- Sys.time();
    cl <- match.call();

    ellipsis <- list(...);

# Add all the variables in ellipsis to current environment
    list2env(ellipsis,environment());

    # Check if the simulated thing is provided
    if(is.smooth.sim(y)){
        if(smoothType(y)=="SMA"){
            model <- y;
            y <- y$data;
        }
    }
    else if(is.smooth(y)){
        model <- y;
        y <- y$y;
    }
    else{
        model <- ellipsis$model;

        if(inherits(y,"Mdata")){
            h <- y$h;
            holdout <- TRUE;
            y <- ts(c(y$x,y$xx),start=start(y$x),frequency=frequency(y$x));
        }
    }

    # If a previous model provided as a model, write down the variables
    if(!is.null(model)){
        if(is.null(model$model)){
            stop("The provided model is not Simple Moving Average!",call.=FALSE);
        }
        else if(smoothType(model)!="SMA"){
            stop("The provided model is not Simple Moving Average!",call.=FALSE);
        }
        else{
            order <- model$orders[1];
        }
    }


    #### Information Criteria ####
    ic <- match.arg(ic,c("AICc","AIC","BIC","BICc"));
    icFunction <- switch(ic,
                         "AIC"=AIC,
                         "AICc"=AICc,
                         "BIC"=BIC,
                         "BICc"=BICc);

    obsAll <- length(y) + (1 - holdout)*h;
    obsInSample <- length(y) - holdout*h;
    yInSample <- y[1:obsInSample];
    yFrequency <- frequency(y);

    if(!is.null(order)){
        if(obsInSample < order){
            stop("Sorry, but we don't have enough observations for that order.",call.=FALSE);
        }

        if(!is.numeric(order)){
            stop("The provided order is not numeric.",call.=FALSE);
        }
        else{
            if(length(order)!=1){
                warning("The order should be a scalar. Using the first provided value.",call.=FALSE);
                order <- order[1];
            }

            if(order<1){
                stop("The order of the model must be a positive number.",call.=FALSE);
            }
        }
        orderSelect <- FALSE;
    }
    else{
        orderSelect <- TRUE;
    }

    # ETS type
    Etype <- "A";
    Ttype <- Stype <- "N";

    componentsNumberETS <- componentsNumberETSSeasonal <- componentsNumberETSNonSeasonal <- xregNumber <- 0;
    constantRequired <- FALSE;
    ot <- yInSample;
    ot[] <- 1;


    CreatorSMA <- function(order){
        lagsModelAll <- matrix(1:order, ncol=1);
        lagsModelMax <- max(lagsModelAll);
        obsStates <- obsInSample+lagsModelMax;

        # # Create ADAM profiles
        adamProfiles <- adamProfileCreator(lagsModelAll, lagsModelMax, obsAll);

        indexLookupTable <- adamProfiles$lookup;
        profilesRecentTable <- adamProfiles$recent;
        profilesRecentTable[order,1:order] <- mean(yInSample[1:order]);

        # State space matrices
        matF <- matrix(1/order,order,order);
        matWt <- matrix(1,obsInSample,order,byrow=TRUE);
        vecG <- matrix(1/order,order);
        matVt <- matrix(0,order,obsStates);

        # Create C++ adam class, which will then use fit, forecast etc methods
        adamCpp <- new(adamCore,
                       lagsModelAll, Etype, Ttype, Stype,
                       componentsNumberETSNonSeasonal,
                       componentsNumberETSSeasonal,
                       componentsNumberETS, order,
                       xregNumber, length(lagsModelAll),
                       constantRequired, FALSE);

        #### Fitter and the losses calculation ####
        adamFitted <- adamCpp$fit(matVt, matWt,
                                  matF, vecG,
                                  indexLookupTable, profilesRecentTable,
                                  yInSample, ot,
                                  TRUE, 2,
                                  TRUE, "n");

        # Get scale, cf, logLik and IC
        scale <- sqrt(sum(adamFitted$errors^2)/obsInSample);
        cfObjective <- sum(dnorm(x=yInSample, mean=adamFitted$fitted, sd=scale, log=TRUE));

        nStatesBackcasting <- 0;
        #### This is switched off because in sma() the initial values have almost no effect
        # on the final values. This is because the weights are 1/n, and the difference
        # between g=0 and g=1/n is almost non-existent

        # Calculate the number of degrees of freedom coming from states in case of backcasting
        # nStatesBackcasting[] <- calculateBackcastingDF(profilesRecentTable, lagsModelAll,
        #                                                FALSE, Stype, componentsNumberETSNonSeasonal,
        #                                                componentsNumberETSSeasonal, vecG, matF,
        #                                                obsInSample, lagsModelMax, indexLookupTable,
        #                                                adamCpp);

        logLik <- structure(cfObjective, nobs=obsInSample, df=1+nStatesBackcasting, class="logLik");
        ICValue <- icFunction(logLik);

        return(ICValue);
    }


    # Select the order of sma
    if(orderSelect){
        maxOrder <- min(200,obsInSample);
        ICs <- rep(NA,maxOrder);
        if(fast){
            # The lowest bound
            iNew <- i <- 1;
            ICs[i] <- CreatorSMA(i);
            # The highest bound
            kNew <- k <- maxOrder
            ICs[k] <- CreatorSMA(k);
            # The middle point
            j <- floor((k+i)/2);
            # If the new IC is the same as one of the old ones, stop
            optimalICNotFound <- TRUE;

            # Track number of iterations
            m <- 1;
            while(optimalICNotFound){
                # Escape the loop if it takes too much time
                m[] <- m+1
                if(m>maxOrder){
                    break;
                }
                ICs[j] <- CreatorSMA(j);
                if(!silent){
                    cat(paste0("Order ", i, " - ", round(ICs[i],4), "; "));
                    cat(paste0("Order ", j, " - ", round(ICs[j],4), "; "));
                    cat(paste0("Order " , k, " - ", round(ICs[k],4), "\n"));
                }

                # Move the bounds
                iNew[] <- which(min(c(ICs[i],ICs[j],ICs[k]))==ICs);
                kNew[] <- which(sort(c(ICs[i],ICs[j],ICs[k]))[2]==ICs);

                # If both bounds haven't changed, move the higher one to the middle
                if(i==iNew && k==kNew || k==iNew && i==kNew){
                    kNew[] <- j;
                }

                i[] <- min(iNew, kNew);
                k[] <- max(iNew, kNew);
                j[] <- floor((k+i)/2);

                # If the new IC is the same as one of the old ones, stop
                optimalICNotFound[] <- j!=i && j!=k && j!=0;
            }
            # Check a specific order equal to frequency of the data
            if(is.na(ICs[yFrequency]) && obsInSample>=yFrequency){
                ICs[yFrequency] <- CreatorSMA(yFrequency);
                if(!silent){
                    cat(paste0("Order " , yFrequency, " - ", round(ICs[yFrequency],4), "\n"));
                }
            }
        }
        else{
            for(i in 1:maxOrder){
                order <- i;
                ICs[i] <- CreatorSMA(i);
                if(!silent){
                    cat(paste0("Order " , i, " - ", round(ICs[i],4), "\n"));
                }
            }
        }
        order <- which.min(ICs)[1];
    }
    smaModel <- adam(y, model="NNN", orders=c(order,0,0), lags=1,
                     h=h, holdout=holdout, ic=ic, silent=TRUE,
                     arma=rep(1/order,order), initial="backcasting",
                     loss="MSE", bounds="none");

    smaModel$model <- paste0("SMA(",order,")");
    smaModel$timeElapsed <- Sys.time()-startTime;
    smaModel$call <- cl;
    if(orderSelect){
        smaModel$ICs <- ICs;
        names(smaModel$ICs) <- c(1:length(ICs));
    }

    if(!silent){
        plot(smaModel, 7);
    }

    return(smaModel);
}
