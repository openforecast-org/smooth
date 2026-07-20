#### Helper functions used by adam() and others

#### Identifiable degrees of freedom of ETS initial states ####
# Greatest common divisor (Euclid) -- base R has none.
dfGCD <- function(a, b){
    while(b != 0){
        t <- b;
        b <- a %% b;
        a <- t;
    }
    return(a);
}

# Identifiable degrees of freedom of the ETS level + seasonal initial states.
# A period-m seasonal component spans the frequencies {0, 1/m, ..., (m-1)/m}, and
# two blocks share the gcd(m_i, m_j) of them; the dimension of the sum of these
# cyclic subspaces is the inclusion-exclusion over gcd of every non-empty subset.
# The level enters as a period-1 block. For a single seasonal m this returns
# m (= level + (m-1)); for coprime periods the naive sum(m_i-1)+level; for
# non-coprime periods it drops by the shared-frequency overlap. seasonalLags are
# the periods whose initials are estimated; levelEstimated adds the level block.
dfInitialsETSLevelSeasonal <- function(seasonalLags, levelEstimated){
    blocks <- numeric(0);
    if(levelEstimated){
        blocks <- c(blocks, 1);
    }
    blocks <- c(blocks, seasonalLags);
    if(length(blocks) == 0){
        return(0);
    }
    total <- 0;
    for(k in 1:length(blocks)){
        combsK <- combn(blocks, k, simplify=FALSE);
        gcdsK <- vapply(combsK, function(s){ Reduce(dfGCD, s); }, numeric(1));
        total <- total + (-1)^(k+1) * sum(gcdsK);
    }
    return(total);
}

#### Small technical functions returning types of models and components ####
# Function defines number of components based on the model type
componentsDefiner <- function(object){
    etsModel <- etsChecker(object);
    arimaModel <- arimaChecker(object);
    cesModel <- cesChecker(object);
    gumModel <- gumChecker(object);
    ssarimaModel <- ssarimaChecker(object);
    sparmaModel <- sparmaChecker(object);

    if(cesModel){
        componentsNumberETS <- componentsNumberETSSeasonal <- componentsNumberETSNonSeasonal <- 0;
        componentsNumberARIMA <- length(object$initial$nonseasonal);
        # If seasonal is formed via a matrix, this must be "simple" or a "full" model
        if(!is.null(object$initial$seasonal)){
            # If this is not a matrix then we have only one seasonal component
            if(is.matrix(object$initial$seasonal)){
                componentsNumberARIMA[] <- componentsNumberARIMA + nrow(object$initial$seasonal);
            }
            else{
                componentsNumberARIMA[] <- componentsNumberARIMA + 1;
            }
        }
    }
    else if(gumModel){
        componentsNumberETS <- componentsNumberETSSeasonal <- componentsNumberETSNonSeasonal <- 0;
        componentsNumberARIMA <- sum(orders(object));
    }
    else if(ssarimaModel){
        arimaOrders <- orders(object);
        lags <- lags(object);
        componentsNumberETS <- componentsNumberETSSeasonal <- componentsNumberETSNonSeasonal <- 0;
        componentsNumberARIMA <- max(arimaOrders$ar %*% lags + arimaOrders$i %*% lags, arimaOrders$ma %*% lags);
    }
    else if(sparmaModel){
        componentsNumberETS <- componentsNumberETSSeasonal <- componentsNumberETSNonSeasonal <- 0;
        componentsNumberARIMA <- length(modelLags(object));
    }
    else{
        if(!is.null(object$initial$seasonal)){
            if(is.list(object$initial$seasonal)){
                componentsNumberETSSeasonal <- length(object$initial$seasonal);
            }
            else{
                componentsNumberETSSeasonal <- 1;
            }
        }
        else{
            componentsNumberETSSeasonal <- 0;
        }
        componentsNumberETSNonSeasonal <- length(object$initial$level) + length(object$initial$trend);
        componentsNumberETS <- componentsNumberETSNonSeasonal + componentsNumberETSSeasonal;
        componentsNumberARIMA <- sum(substr(colnames(object$states),1,10)=="ARIMAState");
    }

    # See if constant is required
    constantRequired <- !is.null(object$constant);

    return(list(componentsNumberETS=componentsNumberETS,
                componentsNumberETSNonSeasonal=componentsNumberETSNonSeasonal,
                componentsNumberETSSeasonal=componentsNumberETSSeasonal,
                componentsNumberARIMA=componentsNumberARIMA,
                constantRequired=constantRequired))
}


etsChecker <- function(object){
    return(any(unlist(gregexpr("ETS",object$model))!=-1));
}

arimaChecker <- function(object){
    return(any(unlist(gregexpr("ARIMA",object$model))!=-1));
}

gumChecker <- function(object){
    return(smoothType(object)=="GUM");
}

ssarimaChecker <- function(object){
    return(smoothType(object)=="SSARIMA");
}

cesChecker <- function(object){
    return(smoothType(object)=="CES");
}

sparmaChecker <- function(object){
    return(smoothType(object)=="SpARMA");
}


#### The function that returns the eigen values for specified parameters ADAM ####
smoothEigens <- function(persistence, transition, measurement,
                         lagsModelAll, xregModel, obsInSample){
    persistenceNames <- names(persistence);
    hasDelta <- any(substr(persistenceNames,1,5)=="delta");
    xregNumber <- sum(substr(persistenceNames,1,5)=="delta");
    constantRequired <- any(persistenceNames %in% c("constant","drift"));

    return(smoothEigensR(persistence, transition, measurement,
                         lagsModelAll, xregModel, obsInSample,
                         hasDelta, xregNumber, constantRequired));

    # lagsUnique <- unique(lagsModelAll);
    # lagsUniqueLength <- length(lagsUnique);
    # eigenValues <- vector("numeric", lagsUniqueLength);
    # # Check eigen values per unique component (unique lag)
    # #### !!!! Eigen values checks do not work for xreg. So, check the average condition
    # if(xregModel && any(substr(names(persistence),1,5)=="delta")){
    #     # We check the condition on average
    #     return(eigen((transition -
    #                       diag(as.vector(persistence)) %*%
    #                       t(measurementInverter(measurement[1:obsInSample,,drop=FALSE])) %*%
    #                       measurement[1:obsInSample,,drop=FALSE] / obsInSample),
    #                  symmetric=FALSE, only.values=TRUE)$values);
    # }
    # else{
    #     for(i in 1:lagsUniqueLength){
    #         eigenValues[which(lagsModelAll==lagsUnique[i])] <-
    #             eigen(transition[lagsModelAll==lagsUnique[i], lagsModelAll==lagsUnique[i], drop=FALSE] -
    #                       persistence[lagsModelAll==lagsUnique[i],,drop=FALSE] %*%
    #                       measurement[obsInSample,lagsModelAll==lagsUnique[i],drop=FALSE],
    #                   symmetric=FALSE, only.values=TRUE)$values
    #     }
    # }
    # return(eigenValues);
}


# Internal: OPG / BHHH covariance of the parameters. The observed Fisher
# Information (numerical Hessian of the log-likelihood) can be indefinite when a
# parameter sits at an active bound (e.g. a smoothing parameter estimated at 0),
# so its inverse is not a valid covariance. The OPG estimator J = sum_t s_t s_t'
# of the per-observation scores s_t = d(logLik_t)/d(theta) is PSD by
# construction and gives finite standard errors for boundary-but-identified
# parameters. Scores are obtained by central-differencing pointLik() (the
# per-observation log-density, which already dispatches over every distribution
# and model type) at parameter values perturbed via the model's own call, so the
# selected ETS/ARIMA structure and lags are preserved. Returns the covariance
# matrix, or NULL to signal the caller to fall back to the Hessian.
covarOPG <- function(object, stepSize=.Machine$double.eps^(1/4)){
    B <- coef(object);
    parametersNames <- names(B);
    nParam <- length(B);
    if(nParam==0 || object$loss!="likelihood"){
        return(NULL);
    }

    # Initial states are estimated parameters ONLY for initial="optimal". For
    # backcasting / gradient / complete they are derived, not free, so they must
    # not be perturbed: the score is taken over the dynamics parameters
    # (persistence / phi / arma) only, and the initials are re-derived (via the
    # model's own initialType) at every perturbed evaluation.
    initialType <- object$initialType;
    initialsEstimated <- identical(initialType, "optimal");

    persistenceNames <- names(object$persistence);
    armaAr <- object$arma$ar;
    armaMa <- object$arma$ma;
    armaNames <- c(names(armaAr), names(armaMa));
    initialList <- object$initial;
    xregInitNames <- names(initialList$xreg);

    # Which engine fitted this object: the state-space ARIMA engines (ssarima,
    # sparma) use their own ARIMA representation, so they must be re-fitted with
    # their own function (an adam() refit would reconstruct a different model).
    # CES and GUM have their own parameterisations (complex smoothing, free
    # transition) whose perturbation maps are not yet implemented; they return
    # NULL here and the caller falls back to the Hessian.
    engine <- if(ssarimaChecker(object)){ "ssarima"; }
              else if(sparmaChecker(object)){ "sparma"; }
              else if(cesChecker(object) || gumChecker(object)){ NULL; }
              else { "adam"; }
    if(is.null(engine)){
        return(NULL);
    }

    # Structural refit arguments reconstructed from the object.
    modelString <- modelType(object);
    modelLags <- object$lags;
    modelOrders <- if(!is.null(object$orders)){ object$orders; } else { NULL; }
    regressorsMode <- object$regressors;

    # Per-observation log-likelihood of the model rebuilt at a perturbed
    # parameter set: everything provided (empty B, single evaluation), the
    # selected structure and lags preserved. Initials are re-derived for
    # non-optimal initialisation.
    evalPointLik <- function(persistence, phi, arma, initialArg){
        if(engine=="adam"){
            args <- list(data=object$data, model=modelString, lags=modelLags,
                         persistence=persistence, phi=phi, initial=initialArg,
                         h=0, FI=FALSE, silent=TRUE);
            if(!is.null(regressorsMode)){ args$regressors <- regressorsMode; }
        }
        else if(engine=="ssarima"){
            # ssarima's data argument is named `y`, and it takes explicit lags.
            args <- list(y=object$data, lags=modelLags, initial=initialArg,
                         h=0, FI=FALSE, silent=TRUE);
        }
        else{
            # sparma: data argument is `data`, no lags argument.
            args <- list(data=object$data, initial=initialArg,
                         h=0, FI=FALSE, silent=TRUE);
        }
        if(!is.null(modelOrders)){ args$orders <- modelOrders; }
        if(length(arma)>0){ args$arma <- arma; }
        modelLocal <- try(suppressWarnings(do.call(engine, args)), silent=TRUE);
        if(inherits(modelLocal,"try-error")){
            return(NULL);
        }
        return(as.numeric(pointLik(modelLocal)));
    }

    # For optimal the perturbed initial list is passed; otherwise the initialType
    # string, so the initials are re-derived at the perturbed dynamics.
    baseArma <- if(length(armaNames)>0){ object$arma; } else { list(); }
    baseInitialArg <- if(initialsEstimated){ initialList; } else { initialType; }

    # Reproduction guard: the reconstructed model at the UNPERTURBED parameters
    # must reproduce the object's likelihood. If not (e.g. regressors="select"/
    # "adapt", or any structure the refit cannot rebuild exactly), the scores
    # would be meaningless, so fall back to the Hessian path.
    baseLik <- evalPointLik(object$persistence, object$phi, baseArma, baseInitialArg);
    if(is.null(baseLik) ||
       !isTRUE(all.equal(sum(baseLik), as.numeric(logLik(object)), tolerance=1e-4))){
        return(NULL);
    }

    obsInSample <- nobs(object);
    scores <- matrix(NA_real_, obsInSample, nParam, dimnames=list(NULL, parametersNames));

    # Build the perturbed parameter set for one named parameter.
    perturbSet <- function(nameJ, delta){
        persistence <- object$persistence;
        phi <- object$phi;
        arma <- baseArma;
        initialArg <- baseInitialArg;
        if(nameJ %in% persistenceNames){
            persistence[nameJ] <- persistence[nameJ]+delta;
        }
        else if(nameJ=="phi"){
            phi <- phi+delta;
        }
        else if(nameJ %in% names(armaAr)){
            arma$ar[nameJ] <- arma$ar[nameJ]+delta;
        }
        else if(nameJ %in% names(armaMa)){
            arma$ma[nameJ] <- arma$ma[nameJ]+delta;
        }
        else if(initialsEstimated){
            # Initial-state parameter (estimated only under initial="optimal").
            if(nameJ=="level"){
                initialArg$level <- initialArg$level+delta;
            }
            else if(nameJ=="trend"){
                initialArg$trend <- initialArg$trend+delta;
            }
            else if(substr(nameJ,1,8)=="seasonal"){
                idx <- as.integer(sub("^seasonal_?","",nameJ));
                if(is.list(initialArg$seasonal)){
                    initialArg$seasonal[[1]][idx] <- initialArg$seasonal[[1]][idx]+delta;
                }
                else{
                    initialArg$seasonal[idx] <- initialArg$seasonal[idx]+delta;
                }
            }
            else if(substr(nameJ,1,10)=="ARIMAState"){
                idx <- as.integer(sub("^ARIMAState","",nameJ));
                initialArg$arima[idx] <- initialArg$arima[idx]+delta;
            }
            else if(nameJ %in% xregInitNames){
                initialArg$xreg[nameJ] <- initialArg$xreg[nameJ]+delta;
            }
            else{
                return(NULL);
            }
        }
        else{
            # A non-dynamics parameter with non-optimal initials should not occur.
            return(NULL);
        }
        return(list(persistence=persistence, phi=phi, arma=arma, initialArg=initialArg));
    }

    for(j in seq_len(nParam)){
        nameJ <- parametersNames[j];
        hStep <- stepSize*max(1, abs(B[j]));
        setUp <- perturbSet(nameJ, hStep);
        setDown <- perturbSet(nameJ, -hStep);
        if(is.null(setUp) || is.null(setDown)){
            return(NULL);
        }
        likUp <- evalPointLik(setUp$persistence, setUp$phi, setUp$arma, setUp$initialArg);
        likDown <- evalPointLik(setDown$persistence, setDown$phi, setDown$arma, setDown$initialArg);
        if(is.null(likUp) || is.null(likDown) ||
           length(likUp)!=obsInSample || length(likDown)!=obsInSample){
            return(NULL);
        }
        scores[,j] <- (likUp-likDown)/(2*hStep);
    }

    if(any(!is.finite(scores))){
        return(NULL);
    }

    # J = S'S is PSD by construction. Drop directions with (near-)zero
    # information, matching the brokenVariables logic of the Hessian path.
    J <- crossprod(scores);
    diagJ <- diag(J);
    keep <- diagJ > max(diagJ)*1e-10 & is.finite(diagJ);
    vcovMatrix <- matrix(Inf, nParam, nParam, dimnames=list(parametersNames, parametersNames));
    if(any(keep)){
        vcovKeep <- try(solve(J[keep,keep,drop=FALSE]), silent=TRUE);
        if(inherits(vcovKeep,"try-error")){
            return(NULL);
        }
        vcovMatrix[keep,keep] <- vcovKeep;
    }
    return(vcovMatrix);
}
