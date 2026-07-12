#' Combination of forecasts of state space models
#'
#' Function constructs ETS, SSARIMA, CES, GUM and SMA and combines their
#' forecasts using IC weights.
#'
#' The combination of these models using information criteria weights is
#' possible because they are all formulated in Single Source of Error
#' framework. Due to the the complexity of some of the models, the
#' estimation process may take some time. So be patient.
#'
#' The prediction interval are combined either probability-wise or
#' quantile-wise (Lichtendahl et al., 2013), which may take extra time,
#' because we need to produce all the distributions for all the models.
#' This can be sped up with the smaller value for bins parameter, but
#' the resulting interval may be imprecise.
#'
#' @template ssBasicParam
#' @template ssAdvancedParam
#' @template ssXregParam
#' @template ssIntervals
#' @template ssAuthor
#' @template ssKeywords
#'
#' @template ssGeneralRef
#' @template ssETSRef
#' @template ssIntervalsRef
#'
#' @param models List of the estimated smooth models to use in the
#' combination. If \code{NULL}, then all the models are estimated
#' in the function.
#' @param initial Can be \code{"optimal"}, meaning that the initial
#' states are optimised, or \code{"backcasting"}, meaning that the
#' initials are produced using backcasting procedure.
#' @param bounds What type of bounds to use in the model estimation. The first
#' letter can be used instead of the whole word. \code{"usual"} implies restrictions on
#' the smoothing parameter, guaranteeing that the exponential smoothing behaves
#' as an averaging model. \code{"admissible"} guarantee that the model is stable.
#' @param bins The number of bins for the prediction interval.
#' The lower value means faster work of the function, but less
#' precise estimates of the quantiles. This needs to be an even
#' number.
#' @param intervalCombine How to average the prediction interval:
#' quantile-wise (\code{"quantile"}) or probability-wise
#' (\code{"probability"}).
#' @param ... This currently determines nothing.
#'
#' \itemize{
#' \item \code{timeElapsed} - time elapsed for the construction of the model.
#' \item \code{initialType} - type of the initial values used.
#' \item \code{fitted} - fitted values of ETS.
#' \item \code{quantiles} - the 3D array of produced quantiles if \code{interval!="none"}
#' with the dimensions: (number of models) x (bins) x (h).
#' \item \code{forecast} - point forecast of ETS.
#' \item \code{lower} - lower bound of prediction interval. When \code{interval="none"}
#' then NA is returned.
#' \item \code{upper} - higher bound of prediction interval. When \code{interval="none"}
#' then NA is returned.
#' \item \code{residuals} - residuals of the estimated model.
#' \item \code{s2} - variance of the residuals (taking degrees of freedom into account).
#' \item \code{interval} - type of interval asked by user.
#' \item \code{level} - confidence level for interval.
#' \item \code{cumulative} - whether the produced forecast was cumulative or not.
#' \item \code{y} - original data.
#' \item \code{holdout} - holdout part of the original data.
#' \item \code{xreg} - provided vector or matrix of exogenous variables. If \code{regressors="s"},
#' then this value will contain only selected exogenous variables.
#' \item \code{ICs} - values of information criteria of the model. Includes AIC, AICc, BIC and BICc.
#' \item \code{accuracy} - vector of accuracy measures for the holdout sample. In
#' case of non-intermittent data includes: MPE, MAPE, SMAPE, MASE, sMAE,
#' RelMAE, sMSE and Bias coefficient (based on complex numbers). In case of
#' intermittent data the set of errors will be: sMSE, sPIS, sCE (scaled
#' cumulative error) and Bias coefficient.
#' }
#'
#' @seealso \code{\link[smooth]{es}, \link[smooth]{auto.ssarima},
#' \link[smooth]{auto.ces}, \link[smooth]{auto.gum}, \link[smooth]{sma}}
#'
#' @examples
#'
#' \dontrun{ourModel <- smoothCombine(BJsales,interval="p")
#' plot(ourModel)}
#'
#' # models parameter accepts either previously estimated smoothCombine
#' # or a manually formed list of smooth models estimated in sample:
#' \dontrun{smoothCombine(BJsales,models=ourModel)}
#'
#' \dontrun{models <- list(es(BJsales), sma(BJsales))
#' smoothCombine(BJsales,models=models)}
#'
#' @importFrom stats fitted
#' @export smoothCombine
smoothCombine <- function(y, models=NULL,
                          initial=c("backcasting", "optimal", "two-stage", "complete", "gradient"),
                          ic=c("AICc","AIC","BIC","BICc"),
                          loss=c("MSE","MAE","HAM","MSEh","TMSE","GTMSE","MSCE"),
                          h=10, holdout=FALSE, cumulative=FALSE,
                          interval=c("none", "prediction", "confidence", "simulated",
                                     "approximate", "semiparametric", "nonparametric",
                                     "empirical","complete"),
                          level=0.95,
                          bins=200, intervalCombine=c("quantile","probability"),
                          bounds=c("usual","admissible","none"),
                          silent=TRUE,
                          xreg=NULL, regressors=c("use","select"), initialX=NULL, ...){
# Copyright (C) 2018 - Inf  Ivan Svetunkov

# Start measuring the time of calculations
    startTime <- Sys.time();

    if(any(is.smoothC(models))){
        ourQuantiles <- models$quantiles;
        models <- models$models;
    }
    else{
        ourQuantiles <- NA;
    }

    ### Depricate the old parameters
    ellipsis <- list(...)
    ellipsis <- depricator(ellipsis, "xregDo", "regressors");

    updateX <- FALSE;
    persistenceX <- transitionX <- NULL;
    occurrence <- "none";
    oesmodel <- "MNN";
    intervalOriginal <- interval;

    initial <- match.arg(initial);
    ic <- match.arg(ic);
    IC <- switch(ic,
                 "AIC"=AIC,
                 "AICc"=AICc,
                 "BIC"=BIC,
                 "BICc"=BICc);

    loss <- match.arg(loss);
    intervalType <- match.arg(interval);
    intervalCombine <- match.arg(intervalCombine);
    bounds <- match.arg(bounds);
    regressors <- match.arg(regressors);

    modelsNotProvided <- is.null(models);
    if(modelsNotProvided){
        nModels <- 5;
    }
    else{
        nModels <- length(models);
    }


    #### Model selection, if none is provided ####
    if(modelsNotProvided){
        if(!silent){
            cat("Estimating models... ");
        }
        if(!silent){
            cat("ES");
        }
        esModel <- es(y,initial=initial,ic=ic,loss=loss,h=h,holdout=holdout,
                      bounds=bounds,silent=TRUE,
                      xreg=xreg,regressors=regressors, initialX=initialX);
        if(!silent){
            cat(", CES");
        }
        cesModel <- auto.ces(y,initial=initial,ic=ic,loss=loss,h=h,holdout=holdout,
                             bounds=bounds,silent=TRUE,
                             xreg=xreg,regressors=regressors, initialX=initialX);
        if(!silent){
            cat(", SSARIMA");
        }
        ssarimaModel <- auto.ssarima(y,initial=initial,ic=ic,loss=loss,h=h,holdout=holdout,
                                     bounds=bounds,silent=TRUE,
                                     xreg=xreg,regressors=regressors, initialX=initialX);
        if(!silent){
            cat(", GUM");
        }
        gumModel <- auto.gum(y,initial=initial,ic=ic,loss=loss,h=h,holdout=holdout,
                             bounds=bounds,silent=TRUE,
                             xreg=xreg,regressors=regressors, initialX=initialX);
        if(!silent){
            cat(", SMA");
        }
        smaModel <- sma(y,ic=ic,h=h,holdout=holdout,
                        silent=TRUE);
        if(!silent){
            cat(". Done!\n");
        }
        models <- list(esModel, cesModel, ssarimaModel, gumModel, smaModel);
        names(models) <- c("ETS","CES","SSARIMA","GUM","SMA");
    }

    yForecastTest <- forecast(models[[1]],h=h,interval="none");
    yHoldout <- yForecastTest$model$holdout;
    yInSample <- actuals(yForecastTest$model);
    yStart <- start(yInSample);
    yFrequency <- frequency(yInSample);
    yForecastStart <- start(yForecastTest$mean);

    # Calculate AIC weights
    ICs <- unlist(lapply(models, IC));
    if(is.null(names(models))){
        names(models) <- unlist(lapply(models,smoothType));
    }
    names(ICs) <- paste0(names(models), " ", ic);

    icBest <- min(ICs);
    icWeights <- exp(-0.5*(ICs-icBest)) / sum(exp(-0.5*(ICs-icBest)));

    modelsForecasts <- lapply(models, forecast, h=h, interval=intervalType,
                              level=0, cumulative=cumulative, newdata=xreg);

    yForecast <- as.matrix(as.data.frame(lapply(modelsForecasts,`[[`,"mean")));
    yForecast <- ts(c(yForecast %*% icWeights),start=yForecastStart,frequency=yFrequency);

    yFitted <- as.matrix(as.data.frame(lapply(models,fitted)));
    yFitted <- ts(c(yFitted %*% icWeights),start=yStart,frequency=yFrequency);

    lower <- upper <- NA;

    if(intervalType!="none"){
        #### This part is for combining the prediction interval ####
        quantilesReturned <- matrix(NA,2,h,dimnames=list(c("Lower","Upper"),paste0("h",c(1:h))));
        # Minimum and maximum quantiles
        minMaxQuantiles <- matrix(NA,2,h);

        if(intervalCombine=="probability"){
            # Probability-based combination
            if((abs(bins) %% 2)<=1e-100){
                bins <- bins-1;
            }

            # If the quantiles are provided and bins in them are larger
            # or equal to what the user asked, then don't recalculate them.
            quantilesRedo <- TRUE;
            if(!any(is.na(ourQuantiles))){
                if(dim(ourQuantiles)[2]>=bins){
                    quantilesRedo <- FALSE;
                }
            }

            # Prepare the matrix for the sequences from min to max for each h
            ourSequence <- array(NA,c(bins,h));

            # If quantiles weren't provided by the previous model, produce them
            if(quantilesRedo){

                # This is needed for appropriate combination of prediction interval
                ourQuantiles <- array(NA,c(nModels,bins,h),dimnames=list(names(models),
                                                                         c(1:bins)/(bins+1),
                                                                         colnames(quantilesReturned)));

                # Write down the median values for all the models
                ourQuantiles[,"0.5",] <- t(as.matrix(as.data.frame(lapply(modelsForecasts,`[[`,"lower"))));

                if(!silent){
                    cat("Constructing prediction interval...    ");
                }
                # Do loop writing down all the quantiles
                for(j in 1:((bins-1)/2)){
                    if(!silent){
                        if(j==1){
                            cat("\b");
                        }
                        cat(paste0(rep("\b",nchar(round((j-1)/((bins-1)/2),2)*100)+1),collapse=""));
                        cat(paste0(round(j/((bins-1)/2),2)*100,"%"));
                    }
                    modelsForecasts <- lapply(models,forecast,h=h,interval=intervalType,
                                              level=j*2/(bins+1),holdout=holdout,cumulative=cumulative,
                                              xreg=xreg);

                    ourQuantiles[,(bins+1)/2-j,] <- t(as.matrix(as.data.frame(lapply(modelsForecasts,
                                                                                     `[[`,"lower"))));
                    ourQuantiles[,(bins+1)/2+j,] <- t(as.matrix(as.data.frame(lapply(modelsForecasts,
                                                                                     `[[`,"upper"))));
                }
            }

            # Write down minimum and maximum values between the models for each horizon
            minMaxQuantiles[1,] <- apply(ourQuantiles,3,min);
            minMaxQuantiles[2,] <- apply(ourQuantiles,3,max);
            # Prepare an array with the new combined probabilities
            newProbabilities <- array(NA,c(bins,h),dimnames=list(c(1:bins),dimnames(ourQuantiles)[[3]]));
            for(j in 1:h){
                ourSequence[,j] <- seq(minMaxQuantiles[1,j],minMaxQuantiles[2,j],length.out=bins);
                for(k in 1:bins){
                    newProbabilities[k,j] <- sum(icWeights %*% (ourQuantiles[,,j] <= ourSequence[k,j])) / (bins+1);
                }
            }

            # The quantilesReturned - quantiles, for which the newP is the first time > than selected value
            for(j in 1:h){
                quantilesReturned[1,j] <- ourSequence[newProbabilities[,j]>=(1-level)/2,j][1];
                quantilesReturned[2,j] <- ourSequence[newProbabilities[,j]>=(1+level)/2,j][1];
            }

            if(!silent){
                cat(" Done!\n");
            }
        }
        else{
            modelsForecasts <- lapply(models,forecast,h=h,interval=intervalType,
                                      level=level,holdout=holdout,cumulative=cumulative,
                                      xreg=xreg);

            quantilesReturned[1,] <- icWeights %*% t(as.matrix(as.data.frame(lapply(modelsForecasts,
                                                                                    `[[`,"lower"))));
            quantilesReturned[2,] <- icWeights %*% t(as.matrix(as.data.frame(lapply(modelsForecasts,
                                                                                    `[[`,"upper"))));
        }

        lower <- ts(quantilesReturned[1,],start=yForecastStart,frequency=yFrequency);
        upper <- ts(quantilesReturned[2,],start=yForecastStart,frequency=yFrequency);
    }

    yInSample <- yInSample[1:length(yFitted)];
    errors <- c(yInSample[1:length(yFitted)])-c(yFitted);
    s2 <- mean(errors^2);

    ##### Now let's deal with holdout #####
    if(holdout){
        if(cumulative){
            errormeasures <- measures(sum(yHoldout),yForecast,h*yInSample);
        }
        else{
            errormeasures <- measures(yHoldout,yForecast,yInSample);
        }

        if(cumulative){
            yHoldout <- ts(sum(yHoldout),start=yForecastStart,frequency=yFrequency);
        }
    }
    else{
        errormeasures <- NA;
    }

    model <- structure(list(timeElapsed=Sys.time()-startTime, models=models, initialType=initial,
                            fitted=yFitted, quantiles=ourQuantiles,
                            forecast=yForecast, lower=lower, upper=upper, residuals=errors, s2=s2,
                            interval=intervalType, level=level, cumulative=cumulative,
                            y=y, holdout=yHoldout, ICs=ICs, ICw=icWeights, loss=loss,
                            lossValue=NULL,accuracy=errormeasures),
                       class=c("smoothC","smooth"));

    if(!silent){
        plot(model, 7)
    }
    return(model);
}
