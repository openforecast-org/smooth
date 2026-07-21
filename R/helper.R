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
# Shared OPG assembly. Central-differences the per-observation log-likelihood
# over the estimated parameters, forms J = sum_t s_t s_t' (PSD by construction)
# and inverts it, dropping directions with (near-)zero information (the
# brokenVariables logic of the Hessian path). `perturbedPointLik(j, delta)` is an
# engine-specific closure that returns the per-observation log-likelihood vector
# for the model rebuilt with the j-th parameter perturbed by delta -- with
# j=NULL, delta=0 giving the base fit -- or NULL on failure. Indexing by position
# (not name) supports engines whose coefficients share names (seasonal CES).
# `parameterValues` is the named coefficient vector (for the step sizes and the
# result dimnames). A reproduction guard requires the base fit to reproduce the
# object's likelihood, so a refit that cannot rebuild the model falls back.
covarOPGCore <- function(object, parameterValues, perturbedPointLik,
                         stepSize=.Machine$double.eps^(1/4)){
    parametersNames <- names(parameterValues);
    nParam <- length(parameterValues);
    if(nParam==0 || object$loss!="likelihood"){
        return(NULL);
    }
    obsInSample <- nobs(object);

    baseLik <- perturbedPointLik(NULL, 0);
    if(is.null(baseLik) || length(baseLik)!=obsInSample ||
       !isTRUE(all.equal(sum(baseLik), as.numeric(logLik(object)), tolerance=1e-4))){
        return(NULL);
    }

    scores <- matrix(NA_real_, obsInSample, nParam);
    for(j in seq_len(nParam)){
        hStep <- stepSize*max(1, abs(parameterValues[j]));
        likUp <- perturbedPointLik(j, hStep);
        likDown <- perturbedPointLik(j, -hStep);
        if(is.null(likUp) || is.null(likDown) ||
           length(likUp)!=obsInSample || length(likDown)!=obsInSample){
            return(NULL);
        }
        scores[,j] <- (likUp-likDown)/(2*hStep);
    }
    if(any(!is.finite(scores))){
        return(NULL);
    }

    J <- crossprod(scores);
    diagJ <- diag(J);
    keep <- diagJ > max(diagJ)*1e-10 & is.finite(diagJ);
    vcovMatrix <- matrix(Inf, nParam, nParam, dimnames=list(parametersNames, parametersNames));
    if(any(keep)){
        Jkeep <- J[keep,keep,drop=FALSE];
        vcovKeep <- try(solve(Jkeep), silent=TRUE);
        if(inherits(vcovKeep,"try-error")){
            # Ill-conditioned (collinear parameters): the plain inverse fails, so
            # use the Moore-Penrose pseudo-inverse via the symmetric eigen-
            # decomposition, dropping the (near-)zero-eigenvalue directions. Still
            # PSD; the parameters spanning the collinear null space get a pooled
            # variance rather than a spurious solve() failure.
            eigenJ <- eigen(Jkeep, symmetric=TRUE);
            positive <- eigenJ$values > max(eigenJ$values)*1e-10;
            if(!any(positive)){
                return(NULL);
            }
            vectorsKeep <- eigenJ$vectors[,positive,drop=FALSE];
            vcovKeep <- vectorsKeep %*% (t(vectorsKeep)/eigenJ$values[positive]);
        }
        vcovMatrix[keep,keep] <- vcovKeep;
    }
    return(vcovMatrix);
}

# OPG covariance for the ETS / ARIMA family (adam and the state-space ARIMA
# engine ssarima). Initial states are estimated parameters ONLY for
# initial="optimal"; for backcasting / gradient / complete they are derived, so
# the score spans the dynamics (persistence / phi / arma) only and the initials
# are re-derived (via the model's own initialType) at each perturbed evaluation.
# CES / GUM / sparma have their own parameterisations and dedicated functions
# (covarOPGces / covarOPGgum / covarOPGsparma).
covarOPG <- function(object, stepSize=.Machine$double.eps^(1/4)){
    engine <- if(ssarimaChecker(object)){ "ssarima"; } else { "adam"; }

    initialType <- object$initialType;
    initialsEstimated <- identical(initialType, "optimal");
    persistenceNames <- names(object$persistence);
    armaAr <- object$arma$ar;
    armaMa <- object$arma$ma;
    initialList <- object$initial;
    xregInitNames <- names(initialList$xreg);

    modelLags <- object$lags;
    modelString <- modelType(object);
    modelOrders <- if(!is.null(object$orders)){ object$orders; } else { NULL; }
    regressorsMode <- object$regressors;
    baseArma <- if(length(c(names(armaAr), names(armaMa)))>0){ object$arma; } else { list(); }
    baseInitialArg <- if(initialsEstimated){ initialList; } else { initialType; }

    refit <- function(persistence, phi, arma, initialArg){
        if(engine=="adam"){
            args <- list(data=object$data, model=modelString, lags=modelLags,
                         persistence=persistence, phi=phi, initial=initialArg,
                         h=0, FI=FALSE, silent=TRUE);
            if(!is.null(regressorsMode)){ args$regressors <- regressorsMode; }
        }
        else{
            # ssarima: data argument is named `y`.
            args <- list(y=object$data, lags=modelLags, initial=initialArg,
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

    parametersNames <- names(coef(object));
    perturbedPointLik <- function(j, delta){
        persistence <- object$persistence;
        phi <- object$phi;
        arma <- baseArma;
        initialArg <- baseInitialArg;
        if(!is.null(j)){
            nameJ <- parametersNames[j];
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
                return(NULL);
            }
        }
        return(refit(persistence, phi, arma, initialArg));
    }

    return(covarOPGCore(object, coef(object), perturbedPointLik, stepSize));
}

# OPG covariance for CES. CES has its own parameterisation: the smoothing
# parameters are complex (a, and b for seasonal models) with the coefficients
# alpha_0 / alpha_1 = Re(a) / Im(a) (and beta_0 / beta_1 = Re(b) / Im(b)); the
# remaining coefficients are the initial states, stored in `profileInitial`.
# ces() re-fitted with model=<perturbed clone> holds all of them (it reads
# parameters$a, parameters$b and profileInitial), so the clone is perturbed
# directly. Mirrors how the CES Hessian FI is computed (via model=object).
covarOPGces <- function(object, stepSize=.Machine$double.eps^(1/4)){
    parameterValues <- coef(object);
    parametersNames <- names(parameterValues);
    aValue <- object$parameters$a;
    bValue <- object$parameters$b;
    seasonalityType <- object$seasonality;

    # The smoothing coefficients (alpha_0/alpha_1 = Re/Im of a, beta_0/beta_1 =
    # Re/Im of b) come first; the rest are initial states in profileInitial
    # (states x head lags). The seasonal initial coefficients are unnamed, so the
    # initials are mapped BY POSITION: the non-seasonal states (level, potential;
    # absent for seasonality="simple") are a single free value each, repeated
    # across the head columns -- perturbing one perturbs the whole row; the
    # seasonal states fill their rows across the head columns in column-major
    # order (verified against coef()).
    nSmoothing <- sum(grepl("^(alpha|beta)_[01]$", parametersNames));
    profile <- object$profileInitial;
    nCol <- ncol(profile);
    nNonSeasonalRows <- if(identical(seasonalityType, "simple")){ 0L } else { 2L };
    seasonalRows <- if(nrow(profile)>nNonSeasonalRows){
                        (nNonSeasonalRows+1):nrow(profile);
                    } else { integer(0); }
    nSeasonalRows <- length(seasonalRows);

    # The profileInitial cells perturbed by the k-th initial coefficient.
    initialCells <- function(k){
        if(k<=nNonSeasonalRows){
            return(list(rows=k, cols=seq_len(nCol)));   # constant state -> whole row
        }
        ks <- k-nNonSeasonalRows;
        list(rows=seasonalRows[((ks-1) %% nSeasonalRows)+1],
             cols=((ks-1) %/% nSeasonalRows)+1);
    }

    perturbedPointLik <- function(j, delta){
        clone <- object;
        if(!is.null(j)){
            nameJ <- parametersNames[j];
            if(nameJ=="alpha_0"){
                clone$parameters$a <- complex(real=Re(aValue)+delta, imaginary=Im(aValue));
            }
            else if(nameJ=="alpha_1"){
                clone$parameters$a <- complex(real=Re(aValue), imaginary=Im(aValue)+delta);
            }
            else if(nameJ=="beta_0"){
                clone$parameters$b <- complex(real=Re(bValue)+delta, imaginary=Im(bValue));
            }
            else if(nameJ=="beta_1"){
                clone$parameters$b <- complex(real=Re(bValue), imaginary=Im(bValue)+delta);
            }
            else{
                cell <- initialCells(j-nSmoothing);
                clone$profileInitial[cell$rows, cell$cols] <-
                    clone$profileInitial[cell$rows, cell$cols]+delta;
            }
        }
        modelLocal <- try(suppressWarnings(
            ces(object$data, seasonality=seasonalityType, lags=object$lags,
                model=clone, h=0)), silent=TRUE);
        if(inherits(modelLocal,"try-error")){
            return(NULL);
        }
        return(as.numeric(pointLik(modelLocal)));
    }

    return(covarOPGCore(object, parameterValues, perturbedPointLik, stepSize));
}

# OPG covariance for GUM. Its coefficients are the persistence g1..gK, the free
# transition matrix F<row><col> (row-major) and the initial states vt1..vtK
# (estimated only under initial="optimal"). Initials are perturbed as free
# parameters only for optimal, where the model is re-fitted from a perturbed
# clone (gum() reads persistence, transition and -- for the initials -- the
# coefficient vector B, the same mechanism the GUM Hessian FI uses). For
# backcasting / gradient / complete the initials are derived, so the score spans
# g / F only and the initials are re-derived by re-fitting with the perturbed
# persistence / transition provided and the model's own initialType.
covarOPGgum <- function(object, stepSize=.Machine$double.eps^(1/4)){
    parameterValues <- coef(object);
    parametersNames <- names(parameterValues);
    persistenceNames <- names(object$persistence);
    transitionDim <- ncol(object$transition);
    transitionNames <- grep("^F", parametersNames, value=TRUE);   # row-major order
    initialsEstimated <- identical(object$initialType, "optimal");

    perturbTransitionCell <- function(transition, nameJ, delta){
        k <- match(nameJ, transitionNames);
        row <- ((k-1) %/% transitionDim)+1;
        col <- ((k-1) %% transitionDim)+1;
        transition[row,col] <- transition[row,col]+delta;
        return(transition);
    }

    perturbedPointLik <- function(j, delta){
        if(initialsEstimated){
            clone <- object;
            if(!is.null(j)){
                nameJ <- parametersNames[j];
                if(nameJ %in% persistenceNames){
                    clone$persistence[nameJ] <- clone$persistence[nameJ]+delta;
                }
                else if(substr(nameJ,1,1)=="F"){
                    clone$transition <- perturbTransitionCell(clone$transition, nameJ, delta);
                }
                else{
                    # Initial state (vt): held through the coefficient vector B.
                    clone$B[nameJ] <- clone$B[nameJ]+delta;
                }
            }
            modelLocal <- try(suppressWarnings(
                gum(object$data, orders=object$orders, lags=object$lags,
                    model=clone, h=0)), silent=TRUE);
        }
        else{
            # Re-derive the initials: perturb only the provided dynamics.
            persistence <- object$persistence;
            transition <- object$transition;
            if(!is.null(j)){
                nameJ <- parametersNames[j];
                if(nameJ %in% persistenceNames){
                    persistence[nameJ] <- persistence[nameJ]+delta;
                }
                else if(substr(nameJ,1,1)=="F"){
                    transition <- perturbTransitionCell(transition, nameJ, delta);
                }
                else{
                    return(NULL);
                }
            }
            modelLocal <- try(suppressWarnings(
                gum(object$data, orders=object$orders, lags=object$lags,
                    persistence=persistence, transition=transition,
                    initial=object$initialType, h=0)), silent=TRUE);
        }
        if(inherits(modelLocal,"try-error")){
            return(NULL);
        }
        return(as.numeric(pointLik(modelLocal)));
    }

    return(covarOPGCore(object, parameterValues, perturbedPointLik, stepSize));
}

# OPG covariance for the sparse ARMA engine. The coefficients are the free AR
# (phi<k>) and MA (theta<k>) parameters and, under initial="optimal", the ARIMA
# state initials (initial1..initialN). sparma() holds provided arma coefficients
# (arEstimate=FALSE fills matF / vecG from them) and, for the initials, accepts
# the free state values directly (front-padded to the dense companion length
# internally), so the model is re-fitted with the perturbed arma / initials
# supplied. For backcasting / gradient / complete the initials are derived, so
# the score spans the arma parameters only and the initials are re-derived.
covarOPGsparma <- function(object, stepSize=.Machine$double.eps^(1/4)){
    parameterValues <- coef(object);
    parametersNames <- names(parameterValues);
    armaArNames <- names(object$arma$ar);
    armaMaNames <- names(object$arma$ma);
    initialsEstimated <- identical(object$initialType, "optimal");
    freeInitial <- parameterValues[grepl("^initial", parametersNames)];

    perturbedPointLik <- function(j, delta){
        arma <- object$arma;
        initialArg <- if(initialsEstimated){ as.numeric(freeInitial); } else { object$initialType; }
        if(!is.null(j)){
            nameJ <- parametersNames[j];
            if(nameJ %in% armaArNames){
                arma$ar[nameJ] <- arma$ar[nameJ]+delta;
            }
            else if(nameJ %in% armaMaNames){
                arma$ma[nameJ] <- arma$ma[nameJ]+delta;
            }
            else if(initialsEstimated && grepl("^initial", nameJ)){
                idx <- as.integer(sub("^initial", "", nameJ));
                initialArg[idx] <- initialArg[idx]+delta;
            }
            else{
                return(NULL);
            }
        }
        modelLocal <- try(suppressWarnings(
            sparma(object$data, orders=object$orders, arma=arma,
                   initial=initialArg, h=0)), silent=TRUE);
        if(inherits(modelLocal,"try-error")){
            return(NULL);
        }
        return(as.numeric(pointLik(modelLocal)));
    }

    return(covarOPGCore(object, parameterValues, perturbedPointLik, stepSize));
}
