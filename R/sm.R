#' @param object The model previously estimated using \code{adam()} function.
#'
#' @rdname adam
#' @importFrom greybox sm is.scale extractScale extractSigma
#' @export
sm.adam <- function(object, model="YYY", lags=NULL,
                    orders=list(ar=c(0),i=c(0),ma=c(0),select=FALSE),
                    constant=FALSE, formula=NULL,
                    regressors=c("use","select","adapt"), data=NULL,
                    persistence=NULL, phi=NULL, initial=c("optimal","backcasting"), arma=NULL,
                    ic=c("AICc","AIC","BIC","BICc"), bounds=c("usual","admissible","none"),
                    silent=TRUE, ...){
    # The function creates a scale model for the provided model
    # occurrence and distribution are extracted from the model.
    # loss can only be likelihood (for now). For future: allow MSE, MAE, HAM, GPL and other similar
    # outliers are not detected
    # Start measuring the time of calculations
    startTime <- Sys.time();
    distribution <- object$distribution;

    if(object$loss!="likelihood"){
        stop("sm() only works with models estimated via maximisation of likelihood. ",
             "Yours was estimated via ", object$loss,". Cannot proceed.", call.=FALSE);
    }

    # If one of the following is used, warn the user
    # if(any(distribution==c("dgamma","dinvgauss"))){
    #     warning("Please note that the scale model for Gamma and Inverse Gaussian distributions ",
    #             "does not produce standardised residuals",
    #             call.=FALSE);
    # }

    if(is.null(data)){
        data <- object$data;
    }

    cl <- match.call();
    # Start a new call for the adam function
    newCall <- as.list(cl);

    # Extract the "other" value
    if(length(object$other)>0){
        other <- switch(distribution,
                        "dgnorm"=,
                        "dlgnorm"=object$other$shape,
                        "dalaplace"=object$other$alpha,
                        NULL);
    }
    else{
        other <- NULL;
    }

    # Extract the name of response variable
    responseName <- as.character(formula(object)[[2]]);

    # Actuals, fitted, residuals, scale from the original model
    yInSampleSM <- actuals(object);
    yFittedSM <- fitted(object);
    et <- residuals(object);
    scaleSM <- object$scale;

    # Get error type and sample size
    EtypeSM <- errorType(object);
    obsInSample <- nobs(object);
    holdout <- !is.null(object$holdout);
    h <- length(object$forecast);
    if(holdout){
        obsAll <- obsInSample + h;
    }

    # Occurrence logical for intermittent model
    if(is.occurrence(object$occurrence)){
        otLogical <- yInSampleSM!=0;
        occurrence <- object$occurrence;
        occurrenceModel <- TRUE;
    }
    else{
        otLogical <- rep(TRUE,obsInSample);
        occurrence <- NULL;
        occurrenceModel <- FALSE;
    }

    #### The custom loss function to estimate parameters of the model ####
    lossFunction <- function(actual,fitted,B,xreg=NULL){
        if(logModelSM){
            fitted[] <- exp(fitted);
        }
        CFValue <- -sum(switch(distribution,
                               "dnorm"=switch(EtypeSM,
                                              "A"=dnorm(x=yInSampleSM[otLogical], mean=yFittedSM[otLogical],
                                                        sd=sqrt(fitted[otLogical]), log=TRUE),
                                              "M"=dnorm(x=yInSampleSM[otLogical], mean=yFittedSM[otLogical],
                                                        sd=sqrt(fitted[otLogical])*yFittedSM[otLogical], log=TRUE)),
                               "dlaplace"=switch(EtypeSM,
                                                 "A"=dlaplace(q=yInSampleSM[otLogical], mu=yFittedSM[otLogical],
                                                              scale=fitted[otLogical], log=TRUE),
                                                 "M"=dlaplace(q=yInSampleSM[otLogical], mu=yFittedSM[otLogical],
                                                              scale=fitted[otLogical]*yFittedSM[otLogical], log=TRUE)),
                               "ds"=switch(EtypeSM,
                                           "A"=ds(q=yInSampleSM[otLogical], mu=yFittedSM[otLogical],
                                                  scale=fitted[otLogical], log=TRUE),
                                           "M"=ds(q=yInSampleSM[otLogical], mu=yFittedSM[otLogical],
                                                  scale=fitted[otLogical]*sqrt(yFittedSM[otLogical]), log=TRUE)),
                               "dgnorm"=switch(EtypeSM,
                                               "A"=dgnorm(q=yInSampleSM[otLogical],mu=yFittedSM[otLogical],
                                                          scale=fitted[otLogical], shape=other, log=TRUE),
                                               # suppressWarnings is needed, because the check is done for scalar alpha
                                               "M"=suppressWarnings(dgnorm(q=yInSampleSM[otLogical],
                                                                           mu=yFittedSM[otLogical],
                                                                           scale=fitted[otLogical]*(yFittedSM[otLogical])^other,
                                                                           shape=other, log=TRUE))),
                               # "dlogis"=switch(EtypeSM,
                               #                 "A"=dlogis(x=yInSampleSM[otLogical],
                               #                            location=yFittedSM[otLogical],
                               #                            scale=fitted[otLogical], log=TRUE),
                               #                 "M"=dlogis(x=yInSampleSM[otLogical],
                               #                            location=yFittedSM[otLogical],
                               #                            scale=fitted[otLogical]*yFittedSM[otLogical], log=TRUE)),
                               # "dalaplace"=switch(EtypeSM,
                               #                    "A"=dalaplace(q=yInSampleSM[otLogical],
                               #                                  mu=yFittedSM[otLogical],
                               #                                  scale=fitted[otLogical], alpha=other, log=TRUE),
                               #                    "M"=dalaplace(q=yInSampleSM[otLogical],
                               #                                  mu=yFittedSM[otLogical],
                               #                                  scale=fitted[otLogical]*yFittedSM[otLogical],
                               #                                  alpha=other, log=TRUE)),
                               # "dlnorm"=dlnorm(x=yInSampleSM[otLogical],
                               #                 meanlog=Re(log(as.complex(yFittedSM[otLogical])))-scaleSM^2/2-log(fitted[otLogical]),
                               #                 sdlog=scaleSM, log=TRUE),
                               # "dllaplace"=dlaplace(q=log(yInSampleSM[otLogical]),
                               #                      mu=Re(log(as.complex(yFittedSM[otLogical]))),
                               #                      scale=fitted[otLogical], log=TRUE) -log(yInSampleSM[otLogical]),
                               # "dls"=ds(q=log(yInSampleSM[otLogical]),
                               #          mu=Re(log(as.complex(yFittedSM[otLogical]))),
                               #          scale=fitted[otLogical], log=TRUE) -log(yInSampleSM[otLogical]),
                               # "dlgnorm"=dgnorm(q=log(yInSampleSM[otLogical]),
                               #                  mu=Re(log(as.complex(yFittedSM[otLogical]))),
                               #                  scale=fitted[otLogical], shape=other, log=TRUE) -log(yInSampleSM[otLogical]),
                               # abs() is needed for rare cases, when negative values are produced for E="A" models
                               "dlnorm"=dlnorm(x=yInSampleSM[otLogical],
                                               meanlog=Re(log(as.complex(yFittedSM[otLogical])))-fitted[otLogical]^2/2,
                                               sdlog=sqrt(fitted[otLogical]), log=TRUE),
                               "dinvgauss"=dinvgauss(x=yInSampleSM[otLogical], mean=abs(yFittedSM[otLogical]),
                                                     dispersion=abs(fitted[otLogical]/yFittedSM[otLogical]), log=TRUE),
                               "dgamma"=dgamma(x=yInSampleSM[otLogical], shape=1/fitted[otLogical],
                                               scale=fitted[otLogical]*yFittedSM[otLogical], log=TRUE)
        ));

        # The differential entropy for the models with the missing data
        if(occurrenceModel){
            CFValue[] <- CFValue + sum(switch(distribution,
                                              "dnorm" =,
                                              # "dfnorm" =,
                                              # "dbcnorm" =,
                                              # "dlogitnorm" =,
                                              "dlnorm" = (log(sqrt(2*pi)*fitted[!otLogical])+0.5),
                                              # "dlgnorm" =,
                                              "dgnorm" =(1/other-
                                                                      log(other /
                                                                              (2*fitted[!otLogical]*gamma(1/other)))),
                                              "dinvgauss" = (0.5*(log(pi/2)+1+suppressWarnings(log(fitted[!otLogical])))),
                                              "dgamma" = (1/fitted[!otLogical] + log(fitted[!otLogical]) +
                                                                      log(gamma(1/fitted[!otLogical])) +
                                                                      (1-1/fitted[!otLogical])*digamma(1/fitted[!otLogical])),
                                              # "dalaplace" =,
                                              # "dllaplace" =,
                                              "dlaplace" = (1 + log(2*fitted[!otLogical])),
                                              # "dls" =,
                                              "ds" = (2 + 2*log(2*fitted[!otLogical])),
                                              # "dlogis" = obsZero*2,
                                              # "dt" = ((fitted[!otLogical]+1)/2 *
                                              #                     (digamma((fitted[!otLogical]+1)/2)-digamma(fitted[!otLogical]/2)) +
                                              #                     log(sqrt(fitted[!otLogical]) * beta(fitted[!otLogical]/2,0.5))),
                                              # "dchisq" = (log(2)*gamma(fitted[!otLogical]/2)-
                                              #                         (1-fitted[!otLogical]/2)*digamma(fitted[!otLogical]/2)+
                                              #                         fitted[!otLogical]/2),
                                              0
            ));
        }
        return(CFValue);
    }

    # Remove sm.adam and "object"
    newCall[[1]] <- NULL;
    newCall$object <- NULL;

    # Transform residuals for the model fit
    # These should align with how the scale is calculated
    # Assign into the non-zero positions rather than over the whole vector: the
    # right-hand side is only sum(otLogical) long for an occurrence model, so
    # `et[] <- ...` recycled it across obsInSample slots and misaligned the
    # scale response (and warned when the two are not multiples).
    et[otLogical] <- switch(distribution,
                   "dnorm"=et[otLogical]^2,
                   "dlaplace"=,
                   "dalaplace"=abs(et[otLogical]),
                   "ds"=0.5*abs(et[otLogical])^0.5,
                   # "dgnorm"=(other*sum(abs(errors)^other)/obsInSample)^{1/other}
                   "dgnorm"=(other*abs(et[otLogical])^other)^{1/other},
                   "dlnorm"=log(et[otLogical])^2,
                   "dgamma"=(et[otLogical]-1)^2,
                   "dinvgauss"=(et[otLogical]-1)^2/et[otLogical]);

    # Substitute the original data with the error term
    data[1:obsInSample,responseName] <- et;
    # If there was a holdout, add values to the "data"
    if(holdout){
        # This shit is needed because data.frame has issues with ts
        if(!is.null(ncol(data)) && ncol(data)>1){
            newCall$data <- as.data.frame(matrix(NA,obsAll,ncol(object$holdout),
                                                 dimnames=list(NULL,colnames(data))));
            newCall$data[c(1:obsInSample),] <- data;
            newCall$data[-c(1:obsInSample),] <- object$holdout;
        }
        else{
            newCall$data <- rbind(data,object$holdout);
        }
        # Get the errors for the holdout
        etForecast <- switch(EtypeSM,
                             "A"=object$holdout[,responseName]-object$forecast,
                             "M"=(object$holdout[,responseName]-object$forecast)/object$forecast);
        newCall$data[obsInSample+1:h,responseName] <- switch(distribution,
                                                             "dnorm"=etForecast^2,
                                                             # "dalaplace"=,
                                                             "dlaplace"=abs(etForecast),
                                                             "ds"=0.5*abs(etForecast)^0.5,
                                                             "dgnorm"=(other*abs(etForecast)^other)^{1/other},
                                                             "dlnorm"=switch(EtypeSM,
                                                                             "A"=(log(1+etForecast/object$forecast)^2),
                                                                             "M"=(log(1+etForecast)^2)),
                                                             "dgamma"=switch(EtypeSM,
                                                                             "A"=(etForecast/object$forecast)^2,
                                                                             "M"=(etForecast)^2),
                                                             "dinvgauss"=switch(EtypeSM,
                                                                                "A"=(etForecast/object$forecast)^2/(etForecast/object$forecast+1),
                                                                                "M"=(etForecast)^2/(1+etForecast)));
    }
    else{
        newCall$data <- data;
    }

    # If the parameters weren't provided, use default values
    if(is.null(newCall$model)){
        newCall$model <- "YYY";
    }
    else{
        if(any(substr(newCall$model,1,1) %in% c("Z","F","P")) ||
           any(substr(newCall$model,2,2) %in% c("Z","F","P")) ||
           any(substr(newCall$model,nchar(newCall$model),nchar(newCall$model)) %in% c("Z","F","P"))){
            warning("This type of model selection is not supported by the sm() function.",
                    call.=FALSE);
        }
    }

    # lags from the main model
    if(is.null(newCall$lags)){
        newCall$lags <- lags(object);
    }
    if(is.null(newCall$constant)){
        newCall$constant <- FALSE;
    }
    # If the formula is not provided, ignore explanatory variables
    if(is.null(newCall$formula)){
        newCall$formula <- NULL;
        responseName <- colnames(newCall$data)[1];
        newCall$data <- newCall$data[,1,drop=FALSE];
    }
    else{
        responseName <- as.character(newCall$formula[[2]]);
    }
    if(is.null(newCall$regressors)){
        newCall$regressors <- "use";
    }

    # If we have a model with additive distributions, do some tricks for ARIMA to fit it in logs
    if((!is.null(orders) || !is.null(formula) || any(substr(newCall$model,1,1) %in% c("A","X"))) &&
       !any(substr(newCall$model,1,1) %in% c("M","Y")) &&
       any(distribution==c("dnorm","dlaplace","ds","dgnorm"))){
        warning("This type of model can only be applied to the data in logarithms. Amending the data",
                call.=FALSE);
        logModelSM <- TRUE;
        newCall$data[,responseName] <- log(newCall$data[,responseName]);
    }
    else{
        logModelSM <- FALSE;
    }

    newCall$h <- h;
    newCall$holdout <- holdout;
    newCall$loss <- lossFunction;
    newCall$occurrence <- object$occurrence;
    newCall$distribution <- object$distribution;
    newCall$outliers <- "ignore";
    newCall$silent <- TRUE;
    if(any(distribution==c("dgnorm","dlgnorm"))){
        newCall$shape <- other;
    }
    else if(distribution=="dalaplace"){
        newCall$alpha <- other;
    }

    adamModel <- do.call(adam, as.list(newCall));

    nVariables <- nparam(adamModel);
    # Replace the logLik first: assigning the attribute before this line set it
    # on the object that the next statement then discarded, so the df never
    # reached the output and the ICs used nparam(scale) alone.
    adamModel$logLik <- -adamModel$lossValue;
    # -1 is needed to remove the scale from the number of parameters
    attr(adamModel$logLik,"df") <- nVariables + nparam(object)-1;
    # object$nParam[1,5] <- object$nParam[1,5]-1;
    # object$nParam[1,1] <- object$nParam[1,1]-1;
    # # Redo nParam table. Record scale parameters in the respective column
    # adamModel$nParam[,4] <- adamModel$nParam[,5];
    # # Use original nparam
    # adamModel$nParam[,1:3] <- object$nParam[,1:3];
    # adamModel$nParam[,5] <- rowSums(adamModel$nParam[,1:4]);

    # Fix fitted and forecast if logARIMA was used
    if(logModelSM){
        adamModel$fitted <- exp(adamModel$fitted);
        adamModel$forecast <- exp(adamModel$forecast);
        adamModel$data[,responseName] <- exp(adamModel$data[,responseName]);
        adamModel$holdout[,responseName] <- exp(adamModel$holdout[,responseName]);
        adamModel$model <- paste0(adamModel$model," in logs");
    }

    # Produce standardised residuals
    adamModel$residuals[] <- switch(distribution,
                                    # N(0, 1)
                                    "dnorm"=as.vector(residuals(object))/sqrt(fitted(adamModel)),
                                    # Laplace(0, 1)
                                    "dlaplace"=as.vector(residuals(object))/fitted(adamModel),
                                    # S(0, 1)
                                    "ds"=as.vector(residuals(object))/fitted(adamModel)^2,
                                    # GN(0, 1, beta)
                                    "dgnorm"=as.vector(residuals(object))/fitted(adamModel)^{1/other},
                                    # Make this logN(-1/2, 1)
                                    "dlnorm"=exp((log(as.vector(residuals(object)))+fitted(adamModel)^2/2-0.5)/fitted(adamModel))-1,
                                    # This becomes Gamma(sigma^-2, 1)
                                    "dgamma"=as.vector(residuals(object))/sqrt(fitted(adamModel))-1,
                                    # IG(sigma^2, 1)
                                    "dinvgauss"=adamModel$residuals,
                                    # All the others
                                    as.vector(residuals(object))/fitted(adamModel));

    adamModel$loss <- "likelihood";
    adamModel$call <- cl;
    adamModel$timeElapsed <- Sys.time()-startTime

    # Reclass the output to the scale model
    class(adamModel) <- c("sm.adam","adam","smooth","scale");

    return(adamModel);
}

#' @export
extractScale.smooth <- function(object, ...){
    if(is.scale(object$scale)){
        return(switch(object$distribution,
                      "dnorm"=,
                      "dlnorm"=sqrt(fitted(object$scale)),
                      "ds"=fitted(object$scale)^2,
                      # $other is a list (list(shape=...)), so the shape has to
                      # be named -- extractSigma.smooth below already does.
                      "dgnorm"=fitted(object$scale)^{1/object$scale$other$shape},
                      fitted(object$scale)));
    }
    else{
        if(is.scale(object)){
            return(1);
            # return(switch(object$distribution,
            #           "dnorm"=,
            #           "dlnorm"=sqrt(fitted(object)),
            #           "ds"=fitted(object)^2,
            #           "dgnorm"=fitted(object)^{1/object$other},
            #           fitted(object)));
        }
        else{
            return(object$scale);
        }
    }
}

#' @export
extractSigma.smooth <- function(object, ...){
    if(is.scale(object$scale)){
        return(switch(object$distribution,
                      "dnorm"=,
                      "dlnorm"=,
                      "dlogitnorm"=,
                      "dbcnorm"=,
                      "dfnorm"=,
                      "dinvgauss"=,
                      "dgamma"=extractScale(object),
                      "dlaplace"=,
                      "dllaplace"=sqrt(2*extractScale(object)),
                      "ds"=,
                      "dls"=sqrt(120*(extractScale(object)^4)),
                      "dgnorm"=,
                      "dlgnorm"=sqrt(extractScale(object)^2*gamma(3/object$other$shape) / gamma(1/object$other$shape)),
                      "dlogis"=extractScale(object)*pi/sqrt(3),
                      "dt"=1/sqrt(1-2/extractScale(object)),
                      "dalaplace"=extractScale(object)/sqrt((object$other$alpha^2*(1-object$other$alpha)^2)*
                                                                (object$other$alpha^2+(1-object$other$alpha)^2)),
                      # For now sigma is returned for: dpois, dnbinom, dchisq, dbeta and plogis, pnorm.
                      sigma(object)
        ));
    }
    else{
        return(sigma(object));
    }
}

#' @importFrom greybox implant
#' @export
implant.adam <- function(location, scale, ...){
    if(!is.scale(scale)){
        stop("sm is not a scale model. Cannot procede.",
             call.=FALSE)
    }
    location$scale <- scale;
    location$logLik <- logLik(scale);
    location$lossValue <- scale$lossValue;
    location$nParam[,4] <- scale$nParam[,5];
    location$nParam[,5] <- rowSums(location$nParam[,1:4]);
    location$call$scale <- formula(scale);

    return(location);
}
