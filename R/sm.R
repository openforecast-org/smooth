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

    # Occurrence logical for intermittent model. The occurrence of a model with
    # missing values only (zero at the gaps, one elsewhere) is not one
    if(is.occurrence(object$occurrence) &&
       any(fitted(object$occurrence)[!is.na(yInSampleSM)]<1)){
        otLogical <- yInSampleSM!=0;
        occurrence <- object$occurrence;
        occurrenceModel <- TRUE;
    }
    else{
        otLogical <- rep(TRUE,obsInSample);
        occurrence <- NULL;
        occurrenceModel <- FALSE;
    }
    # The missing values are gaps, as in the location model: adam() hands the custom
    # loss the observed values only, so the data of the loss are those too
    observedSM <- !is.na(yInSampleSM);
    otLogical[!observedSM] <- FALSE;
    ySM <- yInSampleSM[observedSM];
    fSM <- yFittedSM[observedSM];
    otSM <- otLogical[observedSM];

    #### The custom loss function to estimate parameters of the model ####
    lossFunction <- function(actual,fitted,B,xreg=NULL){
        if(logModelSM){
            fitted[] <- exp(fitted);
        }
        values <- sm_logDensities(ySM, fSM, fitted, distribution, EtypeSM, other, otSM, occurrenceModel);
        # The densities first, then the entropies, as the likelihood sums them
        return(-sum(values[otSM]) - sum(values[!otSM]));
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
                   "dlaplace"=abs(et[otLogical]),
                   "dalaplace"=et[otLogical]*(other-(et[otLogical]<=0)*1),
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
                                                             "dalaplace"=etForecast*(other-(etForecast<=0)*1),
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
    newCall$occurrence <- occurrence;
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
    # The likelihood of the occurrence model is in the joint one, as in the location model
    adamModel$logLik <- -adamModel$lossValue +
        if(occurrenceModel) sum(as.vector(pointLik(occurrence))[observedSM]) else 0;
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
        if(holdout){
            adamModel$holdout[,responseName] <- exp(adamModel$holdout[,responseName]);
        }
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
                                    "dgnorm"=as.vector(residuals(object))/fitted(adamModel),
                                    # Make this logN(-1/2, 1): (log(1+e) + sigma^2/2) / sigma ~ N(0, 1);
                                    # residuals() adds 1
                                    "dlnorm"=exp((log(as.vector(residuals(object)))+fitted(adamModel)/2)/
                                                     sqrt(fitted(adamModel))-0.5)-1,
                                    # (1+e) = sigma^2 eta, with eta ~ Gamma(sigma^-2, 1); residuals() adds 1
                                    "dgamma"=as.vector(residuals(object))/fitted(adamModel)-1,
                                    # IG(sigma^2, 1)
                                    "dinvgauss"=adamModel$residuals,
                                    # All the others
                                    as.vector(residuals(object))/fitted(adamModel));

    adamModel$loss <- "likelihood";
    # The data of the likelihood of the location model, for pointLik()
    adamModel$location <- list(y=as.vector(yInSampleSM), mu=as.vector(yFittedSM), Etype=EtypeSM,
                               other=other, otLogical=otLogical, occurrenceModel=occurrenceModel);
    adamModel$call <- cl;
    adamModel$timeElapsed <- Sys.time()-startTime

    # Reclass the output to the scale model
    class(adamModel) <- c("sm.adam","adam","smooth","scale");

    return(adamModel);
}

#' @param object The model estimated with \code{tbats()}, for \code{sm()}: the scale of
#' its error term is modelled by TBATS (with the arguments and defaults of
#' \code{tbats()}, but the lags of \code{object} by default) on the transformed errors
#' in the space of the Box-Cox transformed data (the squares for \code{"dnorm"}, the
#' absolute values for \code{"dlaplace"}, ..., divided by the exponent of the mean of
#' their logarithm at a unit scale, so that their logarithms are unbiased for the
#' log-scale), with lambda 0, by the joint likelihood
#' of the model's data. \link[greybox]{implant} puts it in \code{object}, whose
#' forecasts then have the scale of each horizon. With an occurrence model, the scale
#' model shares it, and the zeros take the differential entropy at their scale.
#'
#' @rdname tbats
#' @export
sm.adamTBATS <- function(object, lags=NULL, harmonics=NULL,
                         trend=c("auto","none","additive","damped"),
                         orders=list(ar=3, ma=3, select=TRUE),
                         xreg=NULL, regressors=c("use","select","adapt"),
                         ic=c("AICc","AIC","BIC","BICc"),
                         persistence=NULL, phi=NULL,
                         initial=c("backcasting","optimal","two-stage","complete","gradient"), arma=NULL,
                         bounds=c("admissible","usual","none"), silent=TRUE, ...){
    startTime <- Sys.time();
    cl <- match.call();
    if(object$loss!="likelihood"){
        stop("sm() only works with models estimated via maximisation of likelihood. ",
             "Yours was estimated via ", object$loss,". Cannot proceed.", call.=FALSE);
    }
    distribution <- object$distribution;
    shape <- object$other$shape;

    # The model in the space of the transformed data, where its distribution is. With an
    # occurrence model, the sizes, and their predictions at the zeros
    y <- actuals(object);
    observed <- !is.na(as.vector(y));
    otLogical <- tbats_sizes(as.vector(y), object);
    occurrenceModel <- !is.null(object$occurrence);
    objectBC <- tbats_boxCoxObject(object);
    yBC <- as.vector(objectBC$data[,1]);
    muBC <- as.vector(fitted(objectBC));
    errors <- yBC - muBC;
    sizes <- as.vector(fitted(object))/tbats_pFitted(object);
    muBC[!otLogical & observed] <- tbats_boxCox(sizes[!otLogical & observed], object$lambda);
    # The scale of each error, as in sm.adam()
    response <- switch(distribution,
                       "dnorm"=errors^2,
                       "dlaplace"=abs(errors),
                       "ds"=0.5*abs(errors)^0.5,
                       "dgnorm"=(shape*abs(errors)^shape)^(1/shape));
    # The logarithm of the transformed error is biased for the log-scale by the mean of
    # its logarithm at a unit scale (that of chi-squared with one degree of freedom for
    # dnorm, -1.27): removed, so that the scale model in logs follows the scale, and its
    # states (backcast, or updated by the errors in logs) are not off by a factor
    logBias <- switch(distribution,
                      "dnorm"=digamma(0.5)+log(2),
                      "dlaplace"=digamma(1),
                      "ds"=digamma(2)-log(2),
                      "dgnorm"=(log(shape)+digamma(1/shape))/shape);
    response[] <- response*exp(-logBias);

    # The joint log-likelihood of the observed values given the scale, the exponent of
    # the fitted values of the scale model in logs: the densities of the sizes, and minus
    # the entropy at the zeros of an occurrence model
    ySM <- yBC[observed];
    muSM <- muBC[observed];
    otSM <- otLogical[observed];
    lossFunction <- function(actual, fitted, B){
        return(-sum(sm_logDensities(ySM, muSM, exp(fitted), distribution, "A", shape,
                                    otSM, occurrenceModel)));
    }
    if(is.null(lags)){
        lags <- object$lags;
    }
    scaleModel <- tbats(ts(response, start=start(y), frequency=frequency(y)), lags=lags,
                        harmonics=harmonics, trend=trend, lambda=0, orders=orders, xreg=xreg,
                        regressors=regressors, distribution=distribution, loss=lossFunction, ic=ic,
                        persistence=persistence, phi=phi, initial=initial, arma=arma, bounds=bounds,
                        occurrence=if(occurrenceModel) object$occurrence else "none",
                        silent=silent, shape=shape, ...);

    # The scale model is fitted with the occurrence model, whose zeros it sees, but it is the
    # scale: its fitted values and forecasts are not multiplied by the probabilities. The
    # occurrence model stays in the likelihood
    if(occurrenceModel){
        scaleModel$fitted[] <- as.vector(fitted(scaleModel))/tbats_pFitted(scaleModel);
        scaleModel$occurrence <- NULL;
    }

    # The log-likelihood of the data, with the Jacobian of the transform (at the predicted
    # sizes for the zeros) and the parameters of both models (one scale)
    jacobian <- replace((object$lambda-1)*log(ifelse(otLogical, as.vector(y), sizes)), !observed, 0);
    scaleModel$logLik <- structure(as.numeric(logLik(scaleModel)) + sum(jacobian), nobs=sum(observed),
                                   df=nparam(scaleModel)+nparam(object)-1, class="logLik");
    # The standardised residuals
    scaleValues <- as.vector(fitted(scaleModel));
    scaleModel$residuals[] <- errors / switch(distribution, "dnorm"=sqrt(scaleValues),
                                              "ds"=scaleValues^2, scaleValues);
    scaleModel$loss <- "likelihood";
    # The data of the likelihood of the model, for pointLik()
    scaleModel$location <- list(y=yBC, mu=muBC, Etype="A", other=shape, otLogical=otLogical,
                                occurrenceModel=occurrenceModel, jacobian=jacobian,
                                occurrence=object$occurrence);
    scaleModel$call <- cl;
    scaleModel$timeElapsed <- Sys.time()-startTime;
    class(scaleModel) <- c("sm.adam","adamTBATS","adam","smooth","scale");
    return(scaleModel);
}

#' @export
extractScale.smooth <- function(object, ...){
    # The scale model returns the scale itself (sigma^2 for dnorm, s for dlaplace etc)
    if(is.scale(object$scale)){
        return(fitted(object$scale));
    }
    else if(is.scale(object)){
        return(1);
    }
    else{
        return(object$scale);
    }
}

#' @export
extractSigma.smooth <- function(object, ...){
    if(is.scale(object$scale)){
        return(switch(object$distribution,
                      "dt"=1/sqrt(1-2/extractScale(object)),
                      # For now sigma is returned for: dpois, dnbinom, dchisq, dbeta and plogis, pnorm.
                      "dpois"=,"dnbinom"=,"dchisq"=,"dbeta"=,"plogis"=,"pnorm"=sigma(object),
                      sqrt(adam_scaleVariance(extractScale(object), object$distribution, object$other))
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

# The log-likelihood of the location model's observations given the scale, observation
# by observation (the likelihood of sm()): the log-densities of the values with demand,
# and minus the differential entropy at the zeros of an occurrence model
sm_logDensities <- function(y, mu, scale, distribution, Etype, other, otLogical, occurrenceModel){
    EtypeSM <- Etype;
    values <- rep(0, length(y));
    values[otLogical] <- switch(distribution,
                           "dnorm"=switch(EtypeSM,
                                          "A"=dnorm(x=y[otLogical], mean=mu[otLogical],
                                                    sd=sqrt(scale[otLogical]), log=TRUE),
                                          "M"=dnorm(x=y[otLogical], mean=mu[otLogical],
                                                    sd=sqrt(scale[otLogical])*mu[otLogical], log=TRUE)),
                           "dlaplace"=switch(EtypeSM,
                                             "A"=dlaplace(q=y[otLogical], mu=mu[otLogical],
                                                          scale=scale[otLogical], log=TRUE),
                                             "M"=dlaplace(q=y[otLogical], mu=mu[otLogical],
                                                          scale=scale[otLogical]*mu[otLogical], log=TRUE)),
                           "ds"=switch(EtypeSM,
                                       "A"=ds(q=y[otLogical], mu=mu[otLogical],
                                              scale=scale[otLogical], log=TRUE),
                                       "M"=ds(q=y[otLogical], mu=mu[otLogical],
                                              scale=scale[otLogical]*sqrt(mu[otLogical]), log=TRUE)),
                           "dgnorm"=switch(EtypeSM,
                                           "A"=dgnorm(q=y[otLogical],mu=mu[otLogical],
                                                      scale=scale[otLogical], shape=other, log=TRUE),
                                           # suppressWarnings is needed, because the check is done for scalar alpha
                                           "M"=suppressWarnings(dgnorm(q=y[otLogical],
                                                                       mu=mu[otLogical],
                                                                       scale=scale[otLogical]*mu[otLogical],
                                                                       shape=other, log=TRUE))),
                           # "dlogis"=switch(EtypeSM,
                           #                 "A"=dlogis(x=y[otLogical],
                           #                            location=mu[otLogical],
                           #                            scale=scale[otLogical], log=TRUE),
                           #                 "M"=dlogis(x=y[otLogical],
                           #                            location=mu[otLogical],
                           #                            scale=scale[otLogical]*mu[otLogical], log=TRUE)),
                           "dalaplace"=switch(EtypeSM,
                                              "A"=dalaplace(q=y[otLogical],
                                                            mu=mu[otLogical],
                                                            scale=scale[otLogical], alpha=other, log=TRUE),
                                              "M"=dalaplace(q=y[otLogical],
                                                            mu=mu[otLogical],
                                                            scale=scale[otLogical]*mu[otLogical],
                                                            alpha=other, log=TRUE)),
                           # "dlnorm"=dlnorm(x=y[otLogical],
                           #                 meanlog=Re(log(as.complex(mu[otLogical])))-scaleSM^2/2-log(scale[otLogical]),
                           #                 sdlog=scaleSM, log=TRUE),
                           # "dllaplace"=dlaplace(q=log(y[otLogical]),
                           #                      mu=Re(log(as.complex(mu[otLogical]))),
                           #                      scale=scale[otLogical], log=TRUE) -log(y[otLogical]),
                           # "dls"=ds(q=log(y[otLogical]),
                           #          mu=Re(log(as.complex(mu[otLogical]))),
                           #          scale=scale[otLogical], log=TRUE) -log(y[otLogical]),
                           # "dlgnorm"=dgnorm(q=log(y[otLogical]),
                           #                  mu=Re(log(as.complex(mu[otLogical]))),
                           #                  scale=scale[otLogical], shape=other, log=TRUE) -log(y[otLogical]),
                           # abs() is needed for rare cases, when negative values are produced for E="A" models
                           "dlnorm"=dlnorm(x=y[otLogical],
                                           meanlog=Re(log(as.complex(mu[otLogical])))-scale[otLogical]/2,
                                           sdlog=sqrt(scale[otLogical]), log=TRUE),
                           "dinvgauss"=dinvgauss(x=y[otLogical], mean=abs(mu[otLogical]),
                                                 dispersion=abs(scale[otLogical]/mu[otLogical]), log=TRUE),
                           "dgamma"=dgamma(x=y[otLogical], shape=1/scale[otLogical],
                                           scale=scale[otLogical]*mu[otLogical], log=TRUE));
    if(occurrenceModel){
        values[!otLogical] <- -switch(distribution,
                                          # The scale is sigma^2 for dnorm and dlnorm
                                          "dnorm" = (log(sqrt(2*pi*scale[!otLogical]))+0.5),
                                          # "dfnorm" =,
                                          # "dbcnorm" =,
                                          # "dlogitnorm" =,
                                          "dlnorm" = (log(sqrt(2*pi*scale[!otLogical]))+0.5-scale[!otLogical]/2),
                                          # "dlgnorm" =,
                                          "dgnorm" =(1/other-
                                                                  log(other /
                                                                          (2*scale[!otLogical]*gamma(1/other)))),
                                          "dinvgauss" = (0.5*(log(pi/2)+1+suppressWarnings(log(scale[!otLogical])))),
                                          "dgamma" = (1/scale[!otLogical] + log(scale[!otLogical]) +
                                                                  log(gamma(1/scale[!otLogical])) +
                                                                  (1-1/scale[!otLogical])*digamma(1/scale[!otLogical])),
                                          "dalaplace" =,
                                          # "dllaplace" =,
                                          "dlaplace" = (1 + log(2*scale[!otLogical])),
                                          # "dls" =,
                                          "ds" = (2 + 2*log(2*scale[!otLogical])),
                                          # "dlogis" = obsZero*2,
                                          # "dt" = ((scale[!otLogical]+1)/2 *
                                          #                     (digamma((scale[!otLogical]+1)/2)-digamma(scale[!otLogical]/2)) +
                                          #                     log(sqrt(scale[!otLogical]) * beta(scale[!otLogical]/2,0.5))),
                                          # "dchisq" = (log(2)*gamma(scale[!otLogical]/2)-
                                          #                         (1-scale[!otLogical]/2)*digamma(scale[!otLogical]/2)+
                                          #                         scale[!otLogical]/2),
                                          0
        );
    }
    return(values);
}

# The terms of the likelihood of the scale model, observation by observation: those of
# the location model's observations given the scale (zero at the missing values), which
# sum to its logLik()
#' @export
pointLik.sm.adam <- function(object, log=TRUE, ...){
    location <- object$location;
    observed <- !is.na(location$y);
    values <- rep(0, length(location$y));
    values[observed] <- sm_logDensities(location$y[observed], location$mu[observed],
                                        as.vector(fitted(object))[observed], object$distribution,
                                        location$Etype, location$other, location$otLogical[observed],
                                        location$occurrenceModel);
    # The likelihood of the occurrence model, as in logLik()
    if(location$occurrenceModel){
        # [[ ]]: $ would match occurrenceModel partially
        occurrence <- if(is.null(location[["occurrence"]])) object$occurrence else location[["occurrence"]];
        values[observed] <- values[observed] + as.vector(pointLik(occurrence))[observed];
    }
    # The Jacobian of the Box-Cox transform of tbats()
    if(!is.null(location$jacobian)){
        values[] <- values + location$jacobian;
    }
    if(!log){
        values <- exp(values);
    }
    return(values);
}
