#' @param ic The information criterion to use in the model selection.
#' @examples
#'
#' y <- ts(rnorm(100,10,3),frequency=12)
#' # CES with and without holdout
#' auto.ces(y,h=20,holdout=TRUE)
#' auto.ces(y,h=20,holdout=FALSE)
#'
#'
#' # Selection between "none" and "full" seasonalities
#' \donttest{auto.ces(AirPassengers, h=12, holdout=TRUE,
#'                    seasonality=c("n","f"), ic="AIC")}
#'
#' ourModel <- auto.ces(AirPassengers)
#'
#' \donttest{summary(ourModel)}
#' forecast(ourModel, h=12)
#'
#' @rdname ces
#' @export
auto.ces <- function(y, seasonality=c("none","simple","partial","full"), lags=c(frequency(y)),
                     initial=c("backcasting","optimal","two-stage","complete","gradient"),
                     ic=c("AICc","AIC","BIC","BICc"),
                     loss=c("likelihood","MSE","MAE","HAM","MSEh","TMSE","GTMSE","MSCE","GPL"),
                     h=0, holdout=FALSE,
                     bounds=c("admissible","none"),
                     silent=TRUE, xreg=NULL, regressors=c("use","select","adapt"), ...){
#  Function estimates several CES models in state space form with sigma = error,
#  chooses the one with the lowest ic value and returns complex smoothing parameter
#  value, fitted values, residuals, point and interval forecasts, matrix of CES components
#  and values of information criteria

#    Copyright (C) 2015 - Inf Ivan Svetunkov

# Start measuring the time of calculations
    startTime <- Sys.time();

    # Record the call and amend it
    cl <- match.call();
    cl[[1]] <- substitute(smooth::ces);
    # cl[[1]] <- substitute(ces);
    # Make sure that the thing is silent
    cl$silent <- TRUE;

    # Record the parental environment. Needed for optimal initialisation
    env <- parent.frame();
    cl$environment <- env;

    ### Depricate the old parameters
    ellipsis <- list(...);

    # If this is simulated, extract the actuals
    if(is.adam.sim(y) || is.smooth.sim(y)){
        y <- y$data;
    }
    # If this is Mdata, use all the available stuff
    else if(inherits(y,"Mdata")){
        h <- y$h;
        holdout <- TRUE;
        lags <- frequency(y$x);
        y <- ts(c(y$x,y$xx),start=start(y$x),frequency=frequency(y$x));
    }

    # Measure the sample size based on what was provided as data
    obsInSample <- length(y) - holdout*h;

# If the pool of models is wrong, fall back to default
    modelsOk <- rep(FALSE,length(seasonality));
    modelsOk[] <- seasonality %in% c("n","s","p","f","none","simple","partial","full");

    if(!all(modelsOk)){
        message("The pool of models includes a strange type of model! Reverting to default pool.");
        seasonality <- c("n","s","p","f");
    }
    seasonality <- substr(seasonality,1,1);

    ic <- match.arg(ic);
    IC <- switch(ic,
                 "AIC"=AIC,
                 "AICc"=AICc,
                 "BIC"=BIC,
                 "BICc"=BICc);

    initial <- match.arg(initial);
    yFrequency <- max(lags);

    # Define maximum needed number of parameters
    if(any(seasonality=="n")){
    # 1 is for variance, 2 is for complex smoothing parameter
        nParamMax <- 3;
        if(any(initial==c("optimal","two-stage"))){
            nParamMax <- nParamMax + 2;
        }
    }
    if(any(seasonality=="p")){
        nParamMax <- 4;
        if(any(initial==c("optimal","two-stage"))){
            nParamMax <- nParamMax + 2 + yFrequency;
        }
        if(obsInSample <= nParamMax){
            warning("The sample is too small. We cannot use partial seasonal model.",call.=FALSE);
            seasonality <- seasonality[seasonality!="p"];
        }
    }
    if(any(seasonality=="s")){
        nParamMax <- 3;
        if(any(initial==c("optimal","two-stage"))){
            nParamMax <- nParamMax + 2*yFrequency;
        }
        if(obsInSample <= nParamMax){
            warning("The sample is too small. We cannot use simple seasonal model.",call.=FALSE);
            seasonality <- seasonality[seasonality!="s"];
        }
    }
    if(any(seasonality=="f")){
        nParamMax <- 5;
        if(any(initial==c("optimal","two-stage"))){
            nParamMax <- nParamMax + 2 + 2*yFrequency;
        }
        if(obsInSample <= nParamMax){
            warning("The sample is too small. We cannot use full seasonal model.",call.=FALSE);
            seasonality <- seasonality[seasonality!="f"];
        }
    }

    if(yFrequency==1){
        if(!silent){
            message("The data is not seasonal. Simple CES was the only solution here.");
        }
        cl$seasonality <- "none";
        return(eval(cl, envir=env));
    }

# Check the number of observations and number of parameters.
    if(any(seasonality=="f") & (obsInSample <= yFrequency*2 + 2 + 4 + 1)){
        warning("Sorry, but you don't have enough observations for CES(f).",call.=FALSE);
        seasonality <- seasonality[seasonality!="f"];
    }
    if(any(seasonality=="p") & (obsInSample <= yFrequency + 2 + 3 + 1)){
        warning("Sorry, but you don't have enough observations for CES(p).",call.=FALSE);
        seasonality <- seasonality[seasonality!="p"];
    }
    if(any(seasonality=="s") & (obsInSample <= yFrequency*2 + 2 + 1)){
        warning("Sorry, but you don't have enough observations for CES(s).",call.=FALSE);
        seasonality <- seasonality[seasonality!="s"];
    }

    # Get back to the full names
    seasonalityTypes <- c("none","simple","partial","full");
    seasonality <- seasonalityTypes[substr(seasonalityTypes,1,1) %in% seasonality];

    CESModel <- vector("list",length(seasonality));
    names(CESModel) <- seasonality
    ICs <- vector("numeric", length(seasonality));

    if(!silent){
        cat("Estimating CES with seasonality: ");
    }
    # ivan41 is needed to avoid conflicts with using index i
    for(ivan41 in 1:length(seasonality)){
        if(!silent){
            cat(paste0('"',seasonality[ivan41],'" '));
        }

        cl$seasonality <- seasonality[ivan41];
        CESModel[[ivan41]] <- eval(cl, envir=env);
    }
    ICs <- sapply(CESModel, IC);

    bestModel <- CESModel[[which(ICs==min(ICs))[1]]];

    modelname <- bestModel$model;

    if(!silent){
        bestModelType <- seasonality[which(ICs==min(ICs))[1]];
        cat(" \n");
        cat(paste0('The best model is with seasonality = "',bestModelType,'"\n'));
    }

##### Make a plot #####
    if(!silent){
        plot(bestModel, 7)
    }

    bestModel$ICs <- ICs;
    bestModel$timeElapsed <- Sys.time()-startTime;

    return(bestModel);
}
