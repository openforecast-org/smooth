#' Trigonometric Box-Cox ARMA Trend Seasonal model
#'
#' Function constructs the TBATS model of De Livera et al. (2011) in the Single
#' Source of Error state space framework of ADAM: the level and trend of ETS, a
#' trigonometric seasonality for each period (fractional periods are allowed) and
#' an ARMA in ADAM's form, all in the space of the Box-Cox transformed data.
#'
#' The model is estimated on \eqn{y^{(\lambda)}_t}, the Box-Cox transform of the
#' data:
#'
#' \deqn{y^{(\lambda)}_t = w' v_{t-l} + \epsilon_t}
#'
#' \deqn{v_t = F v_{t-l} + g \epsilon_t}
#'
#' where \eqn{v_t} contains the level, the trend, two states for every harmonic
#' and the ARMA states. A harmonic \eqn{j} of the period \eqn{m} with the
#' frequency \eqn{\lambda_j = 2 \pi j / m} adds
#' \eqn{u_t = v_{1,t-1} + v_{2,t-2}} to the measurement, with
#' \eqn{(1 - 2 \cos \lambda_j B + B^2) u_t = (\gamma_1 B + (\sin \lambda_j \gamma_2 - \cos \lambda_j \gamma_1) B^2) \epsilon_t},
#' which is the seasonal component of De Livera et al. (2011), the smoothing
#' parameters \eqn{\gamma_1} and \eqn{\gamma_2} being shared by the harmonics of
#' the same period. The ARMA is the one of \link[smooth]{adam}: a separate
#' component, not a model of the error term.
#'
#' The point forecasts are the medians, the inverse Box-Cox transform of the
#' point forecasts of the transformed data.
#'
#' @template ssAuthor
#' @template ssKeywords
#' @template smoothRef
#' @template ssADAMRef
#'
#' @references \itemize{
#' \item De Livera, A.M., Hyndman, R.J., Snyder, R.D. (2011). Forecasting time
#' series with complex seasonal patterns using exponential smoothing. Journal of
#' the American Statistical Association, 106(496), 1513-1527.
#' }
#'
#' @param y Vector or ts object, containing the data needed to be forecasted.
#' @param lags The lags of the model: 1 and the seasonal periods, which can be
#' fractional (e.g. \code{c(1, 7, 365.25)}). The harmonics are fitted for every
#' lag above 1.
#' @param harmonics The number of harmonics for each lag above 1. If \code{NULL},
#' they are selected by the information criterion on the global model (a
#' regression on a trend and the Fourier terms). The number of harmonics of the
#' period \eqn{m} has to be below \eqn{m/2}.
#' @param trend The type of trend: \code{"none"}, \code{"additive"},
#' \code{"damped"} or \code{"auto"}, which selects between them by the
#' information criterion.
#' @param lambda The Box-Cox parameter. If \code{NULL}, it is estimated in [0, 1]
#' together with the other parameters (with \code{loss="likelihood"} only;
#' otherwise it is set to 1).
#' @param orders The orders of the ARMA, \code{list(ar, ma, select)}, aligned with
#' \code{lags} as in \link[smooth]{adam}. A single value refers to the lag 1. The
#' lags of the ARMA are truncated to integers (365.25 becomes 365). With
#' \code{select=TRUE}, the orders are the maxima: the trend is chosen without ARMA,
#' the orders up to them are screened with Hannan-Rissanen (the lags one at a time,
#' from the largest) on the residuals of the global model, the winner is fitted, and
#' the ARMA is kept only if it improves the information criterion. All the values are
#' returned in \code{ICs}.
#' @param distribution The distribution of the error term in the space of the
#' Box-Cox transformed data: \code{"dnorm"}, \code{"dlaplace"}, \code{"ds"} or
#' \code{"dgnorm"} (the shape is estimated unless \code{shape} is provided in
#' ellipsis).
#' @param loss The loss function, see \link[smooth]{adam}.
#' @param ic The information criterion used in the selection.
#' @param h The forecast horizon.
#' @param holdout If \code{TRUE}, the holdout of the size \code{h} is taken from
#' the data.
#' @param initial The initialisation: \code{"backcasting"} (default),
#' \code{"optimal"}, \code{"two-stage"} or \code{"complete"} (the same as
#' backcasting here).
#' @param bounds The bounds of the parameters: \code{"admissible"} (default)
#' guarantees the stability of the model (the eigenvalues of the discount matrix
#' of the level, trend and harmonics lie in the unit circle and the ARMA is
#' stationary and invertible), \code{"usual"} keeps the smoothing parameters in
#' their usual region (the response of the level and seasonality to the error
#' stays in [0, 1] over the seasonal cycle), which does not guarantee the
#' stability, \code{"none"} only keeps \eqn{\lambda} in [0, 1].
#' @param silent If \code{TRUE}, nothing is printed.
#' @param model A previously estimated TBATS model, if provided, the function
#' will not estimate anything and will use all its parameters.
#' @param ...  Other non-documented parameters, see \link[smooth]{adam}:
#' \code{B}, \code{lb}, \code{ub}, \code{maxeval}, \code{maxtime},
#' \code{algorithm}, \code{xtol_rel}, \code{xtol_abs}, \code{ftol_rel},
#' \code{ftol_abs}, \code{print_level}, \code{nIterations}, \code{headLength},
#' \code{FI}, \code{stepSize} and \code{shape}.
#'
#' @return Object of class "adam" is returned with similar elements to the
#' \link[smooth]{adam} function, together with \code{lambda}, \code{harmonics}
#' and \code{periods}.
#'
#' @seealso \code{\link[smooth]{adam}, \link[smooth]{ces}, \link[smooth]{msarima}}
#'
#' @examples
#' tbats(AirPassengers, orders=list(ar=0, ma=0, select=FALSE), h=12, holdout=TRUE)
#'
#' @rdname tbats
#' @export
tbats <- function(y, lags=c(1, frequency(y)), harmonics=NULL,
                  trend=c("auto","none","additive","damped"),
                  lambda=NULL, orders=list(ar=3, ma=3, select=TRUE),
                  distribution=c("dnorm","dlaplace","ds","dgnorm"),
                  loss=c("likelihood","MSE","MAE","HAM","MSEh","TMSE","GTMSE","MSCE","GPL"),
                  ic=c("AICc","AIC","BIC","BICc"), h=0, holdout=FALSE,
                  initial=c("backcasting","optimal","two-stage","complete"),
                  bounds=c("admissible","usual","none"), silent=TRUE, model=NULL, ...){
    startTime <- Sys.time();
    cl <- match.call();
    ellipsis <- list(...);
    yName <- paste0(deparse(substitute(y)), collapse="");

    trend <- match.arg(trend);
    distribution <- match.arg(distribution);
    loss <- match.arg(loss);
    ic <- match.arg(ic);
    initial <- match.arg(initial);
    bounds <- match.arg(bounds);
    # The Box-Cox parameter is kept apart: the checker returns LASSO's lambda
    lambdaProvided <- lambda;
    modelDo <- "estimate";
    armaSpecProvided <- NULL;

    # A previous model fixes the structure and the parameters
    if(!is.null(model)){
        if(!tbatsChecker(model)){
            stop("The provided model is not TBATS.", call.=FALSE);
        }
        lags <- model$lags;
        harmonics <- model$harmonics;
        trend <- model$trendType;
        armaSpecProvided <- model$armaSpec;
        distribution <- model$distribution;
        loss <- model$loss;
        initial <- model$initialType;
        bounds <- model$bounds;
        lambdaProvided <- if(is.null(model$B) || !any(names(model$B)=="lambda")) model$lambda else NULL;
        if(any(names(model$B)=="shape")){
            ellipsis$shape <- NULL;
        }
        else if(distribution=="dgnorm"){
            ellipsis$shape <- model$other$shape;
        }
        ellipsis$B <- model$B;
        modelDo <- "use";
    }

    # The checker handles the data, the holdout and the optimiser settings in ellipsis
    checked <- parametersChecker(data=y, model="NNN", lags=1, formulaToUse=NULL,
                                 orders=list(ar=0,i=0,ma=0,select=FALSE), constant=FALSE,
                                 arma=NULL, outliers="ignore", level=0.99,
                                 persistence=NULL, phi=NULL, initial=initial,
                                 distribution=distribution, loss=loss, h=h, holdout=holdout,
                                 occurrence="none", ic=ic, bounds=bounds, regressors="use",
                                 yName=yName, silent=silent, modelDo=modelDo,
                                 ellipsis=ellipsis, fast=FALSE);
    checked$headLengthUser <- ellipsis$headLength;
    checked$modelDo <- modelDo;
    checked$bounds <- bounds;
    yInSample <- as.vector(checked$yInSample);
    if(any(!is.finite(yInSample))){
        stop("tbats() does not support missing values yet.", call.=FALSE);
    }

    #### The structure ####
    periods <- sort(unique(lags[lags>1]));
    lambdaSpec <- tbats_lambdaSpec(lambdaProvided, yInSample, loss);
    armaSpec <- if(is.null(armaSpecProvided)) tbats_armaSpec(orders, lags) else armaSpecProvided;
    # With the selection, the trend is chosen without ARMA
    armaSpecFit <- if(armaSpec$select) tbats_armaBuild(0, 0, 1) else armaSpec;
    trendTypes <- if(trend=="auto") c("none","additive","damped") else trend;

    # Harmonics from the global model, at the starting value of lambda
    if(is.null(harmonics)){
        harmonics <- tbats_harmonicsSelect(yInSample, periods, any(trendTypes!="none"),
                                           lambdaSpec, checked$icFunction, ic);
    }
    else{
        if(length(harmonics)!=length(periods)){
            stop("harmonics should have one value per lag above 1.", call.=FALSE);
        }
        kMax <- ceiling(periods/2)-1;
        if(any(harmonics>kMax)){
            warning("The number of harmonics has to be below half of the period. Reducing it.",
                    call.=FALSE);
            harmonics <- pmin(harmonics, kMax);
        }
    }
    harmonicTable <- tbats_harmonics(periods, harmonics);

    #### Fit the candidates and select ####
    candidates <- lapply(trendTypes, function(trendType){
        return(tbats_fit(yInSample, trendType, harmonicTable, armaSpecFit, lambdaSpec,
                         distribution, initial, checked));
    });
    ICs <- sapply(candidates, function(candidate){
        return(tbats_IC(candidate$logLik, ic));
    });
    names(ICs) <- trendTypes;
    best <- candidates[[which.min(ICs)]];

    # The ARMA orders, screened on the residuals of the global model at the lambda of the
    # best one: on its errors, the adaptive level hides an AR in a near-unit MA root.
    # The winner is fitted and kept only if it beats the models without ARMA
    if(armaSpec$select && armaSpec$nParam>0){
        X <- tbats_design(length(yInSample), best$trendType!="none", harmonicTable);
        armaSpecBest <- tbats_armaSelect(qr.resid(qr(X), tbats_boxCox(yInSample, best$elements$lambda)),
                                         armaSpec, distribution, best$elements$shape,
                                         best$nParamEstimated, ic);
        if(armaSpecBest$nParam>0){
            candidate <- tbats_fit(yInSample, best$trendType, harmonicTable, armaSpecBest, lambdaSpec,
                                   distribution, initial, checked);
            icCandidate <- tbats_IC(candidate$logLik, ic);
            if(icCandidate<min(ICs)){
                best <- candidate;
            }
            ICs[paste0(candidate$trendType, "+ARMA(", paste(armaSpecBest$arOrders, collapse=","), ";",
                       paste(armaSpecBest$maOrders, collapse=","), ")")] <- icCandidate;
        }
    }

    return(tbats_return(best, checked, cl, startTime, periods, harmonics, ICs, silent));
}

#### Box-Cox ####
# The Box-Cox transform and its inverse
#' @keywords internal
tbats_boxCox <- function(y, lambda){
    if(lambda==0){
        return(log(y));
    }
    return((y^lambda-1)/lambda);
}

#' @keywords internal
tbats_boxCoxInverse <- function(z, lambda){
    z <- as.numeric(z);
    if(lambda==0){
        return(exp(z));
    }
    return(pmax(lambda*z+1, 0)^(1/lambda));
}

# How lambda is treated: estimated in [0, 1] only with the likelihood and positive
# data; otherwise fixed (provided, or 1 with a message)
#' @keywords internal
tbats_lambdaSpec <- function(lambda, y, loss){
    if(!is.null(lambda)){
        if(lambda<0 || lambda>1){
            stop("lambda should lie in [0, 1].", call.=FALSE);
        }
        if(any(y<=0) && lambda!=1){
            warning("The Box-Cox transform needs positive data. Setting lambda=1.", call.=FALSE);
            lambda <- 1;
        }
        return(list(estimate=FALSE, value=lambda));
    }
    if(any(y<=0)){
        warning("The Box-Cox transform needs positive data. Setting lambda=1.", call.=FALSE);
        return(list(estimate=FALSE, value=1));
    }
    if(loss!="likelihood"){
        message("lambda is estimated with loss=\"likelihood\" only. Setting lambda=1.");
        return(list(estimate=FALSE, value=1));
    }
    return(list(estimate=TRUE, value=NULL));
}

#### Harmonics and the global model ####
# The harmonics of the periods, without those of a longer period whose frequency
# coincides with a harmonic of a shorter one (the 7th of 168 is the 1st of 24)
#' @keywords internal
tbats_harmonics <- function(periods, harmonics){
    table <- data.frame(period=numeric(0), j=integer(0), frequency=numeric(0));
    for(i in seq_along(periods)){
        if(harmonics[i]>0){
            table <- rbind(table, data.frame(period=periods[i], j=1:harmonics[i],
                                             frequency=2*pi*(1:harmonics[i])/periods[i]));
        }
    }
    return(table[!duplicated(round(table$frequency, 10)),,drop=FALSE]);
}

# The design of the global model: an intercept, a trend and the Fourier terms
#' @keywords internal
tbats_design <- function(obs, trendIn, harmonicTable){
    times <- 1:obs;
    X <- matrix(1, obs, 1);
    if(trendIn){
        X <- cbind(X, times);
    }
    if(nrow(harmonicTable)>0){
        angles <- outer(times, harmonicTable$frequency);
        X <- cbind(X, sin(angles), cos(angles));
    }
    return(X);
}

# The maximum of the profile log-likelihood of the global model in lambda, with the
# Jacobian
#' @keywords internal
tbats_lambdaProfile <- function(y, qrX){
    obs <- length(y);
    logY <- sum(log(y));
    profile <- function(lambda){
        rss <- sum(qr.resid(qrX, tbats_boxCox(y, lambda))^2);
        return(-obs/2*log(rss/obs) + (lambda-1)*logY);
    }
    # Rounded: Brent's search finds it to about 1e-4, and the last bits of the
    # least squares differ between linear algebra libraries (the Python port)
    return(round(optimize(profile, c(0, 1), maximum=TRUE)$maximum, 8));
}

# The starting value of lambda (or its fixed value) for a design
#' @keywords internal
tbats_lambdaStart <- function(y, X, lambdaSpec){
    if(!lambdaSpec$estimate){
        return(lambdaSpec$value);
    }
    return(tbats_lambdaProfile(y, qr(X)));
}

# The number of harmonics of each period by the information criterion of the global
# model, one period at a time, stopping after two harmonics without improvement
#' @keywords internal
tbats_harmonicsSelect <- function(y, periods, trendIn, lambdaSpec, icFunction, ic){
    harmonics <- rep(0, length(periods));
    if(length(periods)==0){
        return(harmonics);
    }
    kMax <- pmax(ceiling(periods/2)-1, 0);
    obs <- length(y);
    lambda <- tbats_lambdaStart(y, tbats_design(obs, trendIn, tbats_harmonics(periods, pmin(kMax, 3))),
                                lambdaSpec);
    yBC <- tbats_boxCox(y, lambda);
    icValue <- function(harmonicsTest){
        X <- tbats_design(obs, trendIn, tbats_harmonics(periods, harmonicsTest));
        if(ncol(X)>=obs-1){
            return(Inf);
        }
        rss <- sum(qr.resid(qr(X), yBC)^2);
        logLikValue <- structure(-obs/2*(log(2*pi*rss/obs)+1), nobs=obs, df=ncol(X)+1, class="logLik");
        return(tbats_IC(logLikValue, ic));
    }
    icBest <- icValue(harmonics);
    for(i in seq_along(periods)){
        failures <- 0;
        harmonicsTest <- harmonics;
        while(harmonicsTest[i]<kMax[i] && failures<2){
            harmonicsTest[i] <- harmonicsTest[i]+1;
            icTest <- icValue(harmonicsTest);
            if(icTest<icBest){
                icBest <- icTest;
                harmonics[i] <- harmonicsTest[i];
                failures <- 0;
            }
            else{
                failures <- failures+1;
            }
        }
    }
    return(harmonics);
}

#### ARMA ####
# The orders of the ARMA aligned with the lags, truncated to integer lags and merged
#' @keywords internal
tbats_armaSpec <- function(orders, lags){
    select <- isTRUE(orders$select);
    ar <- if(is.null(orders$ar)) 0 else orders$ar;
    ma <- if(is.null(orders$ma)) 0 else orders$ma;
    armaLags <- if(length(ar)==1 && length(ma)==1) 1 else trunc(lags);
    ar <- rep(ar, length.out=length(armaLags));
    ma <- rep(ma, length.out=length(armaLags));
    return(tbats_armaBuild(ar, ma, armaLags, select));
}

# The specification of the ARMA from its orders per lag
#' @keywords internal
tbats_armaBuild <- function(ar, ma, armaLags, select=FALSE){
    lagsUnique <- sort(unique(armaLags));
    arOrders <- sapply(lagsUnique, function(lag) max(ar[armaLags==lag]));
    maOrders <- sapply(lagsUnique, function(lag) max(ma[armaLags==lag]));
    keep <- (arOrders+maOrders)>0;
    arOrders <- arOrders[keep];
    maOrders <- maOrders[keep];
    lagsUnique <- lagsUnique[keep];
    # The lags of the states: the powers of B in the AR and MA polynomials
    powers <- function(orders){
        if(length(orders)==0){
            return(numeric(0));
        }
        grid <- expand.grid(lapply(seq_along(orders), function(i) (0:orders[i])*lagsUnique[i]));
        return(setdiff(unique(rowSums(grid)), 0));
    }
    stateLags <- sort(unique(c(powers(arOrders), powers(maOrders))));
    names <- unlist(lapply(seq_along(lagsUnique), function(i){
        return(c(if(arOrders[i]>0) paste0("phi", seq_len(arOrders[i]), "[", lagsUnique[i], "]"),
                 if(maOrders[i]>0) paste0("theta", seq_len(maOrders[i]), "[", lagsUnique[i], "]")));
    }));
    return(list(select=select, arOrders=arOrders, maOrders=maOrders, lags=lagsUnique,
                stateLags=stateLags, nParam=sum(arOrders+maOrders), names=names));
}

# The ARMA orders screened with Hannan-Rissanen on the residuals of a model without ARMA,
# one lag at a time from the largest. The IC of a candidate comes from the likelihood of
# its innovations on a common sample: the rest of the model is common to all of them
#' @keywords internal
tbats_armaSelect <- function(errors, armaSpec, distribution, shape, nParamBase, ic){
    lags <- armaSpec$lags;
    arOrders <- maOrders <- rep(0, length(lags));
    obs <- length(errors);
    nDrop <- min(sum(armaSpec$arOrders*lags), floor(obs/4));
    obsUsed <- obs-nDrop;
    for(i in order(lags, decreasing=TRUE)){
        screen <- arimaHRSelectCpp(errors, arOrders, maOrders, lags, i-1,
                                   armaSpec$arOrders[i], armaSpec$maOrders[i], TRUE);
        nParam <- nParamBase + sum(arOrders[-i]+maOrders[-i]) + rowSums(screen$orders);
        ICs <- sapply(seq_len(nrow(screen$orders)), function(j){
            innovations <- screen$innovations[nDrop+seq_len(obsUsed), j];
            logLikValue <- structure(tbats_logLik(innovations, distribution, shape, obsUsed),
                                     nobs=obsUsed, df=nParam[j], class="logLik");
            return(tbats_IC(logLikValue, ic));
        });
        winner <- which.min(ICs);
        arOrders[i] <- screen$orders[winner,1];
        maOrders[i] <- screen$orders[winner,2];
    }
    return(tbats_armaBuild(arOrders, maOrders, lags));
}

# The information criterion of a log-likelihood
#' @keywords internal
tbats_IC <- function(logLikValue, ic){
    return(switch(ic, "AIC"=AIC(logLikValue), "AICc"=AICc(logLikValue),
                  "BIC"=BIC(logLikValue), "BICc"=BICc(logLikValue)));
}

#### Structure ####
# The parts of the model that do not depend on the parameters: the lags, the rows of
# the components, the fixed transition blocks of the harmonics and the matrices of
# the usual bounds
#' @keywords internal
tbats_structure <- function(trendType, harmonicTable, armaSpec, periods){
    trendIn <- trendType!="none";
    nETS <- 1+trendIn;
    nHarmonics <- nrow(harmonicTable);
    nArma <- length(armaSpec$stateLags);
    lagsModelAll <- c(rep(1, nETS), rep(c(1, 2), nHarmonics), armaSpec$stateLags);
    nComponents <- length(lagsModelAll);
    harmonicRows <- nETS + 2*seq_len(nHarmonics) - 1;
    armaRows <- nETS + 2*nHarmonics + seq_len(nArma);
    matF <- diag(0, nComponents);
    matF[1, 1] <- 1;
    for(i in seq_len(nHarmonics)){
        rows <- harmonicRows[i]+0:1;
        matF[rows, rows] <- c(2*cos(harmonicTable$frequency[i]), -1) %o% c(1, 1);
    }
    # The response of the level and seasonality over the longest cycle
    horizons <- 0:(ceiling(max(c(1, periods)))-1);
    angles <- outer(horizons, harmonicTable$frequency);
    periodIndex <- match(harmonicTable$period, periods);
    componentNames <- c("level", if(trendIn) "trend",
                        if(nHarmonics>0) paste0(rep(c("s","s*"), nHarmonics), rep(harmonicTable$j, each=2),
                                                "[", rep(round(harmonicTable$period, 4), each=2), "]"),
                        if(nArma>0) paste0("ARMAState", seq_len(nArma)));
    return(list(trendType=trendType, trendIn=trendIn, damped=trendType=="damped",
                nETS=nETS, nHarmonics=nHarmonics, nArma=nArma, nComponents=nComponents,
                lagsModelAll=lagsModelAll, lagsModelMax=max(lagsModelAll),
                harmonicRows=harmonicRows, armaRows=armaRows, matF=matF,
                harmonicTable=harmonicTable, periods=periods, periodIndex=periodIndex,
                responseCos=cos(angles), responseSin=sin(angles),
                armaLagMax=if(nArma>0) max(armaSpec$stateLags) else 0,
                componentNames=componentNames));
}

#### Initial states ####
# The initial states of the global model: the level and trend at t=0 and the
# Fourier coefficients of the harmonics
#' @keywords internal
tbats_globalStates <- function(beta, struct){
    beta[!is.finite(beta)] <- 0;
    nH <- struct$nHarmonics;
    k <- 1+struct$trendIn;
    return(list(level=beta[1], trend=if(struct$trendIn) beta[2] else 0,
                sinCoef=beta[k+seq_len(nH)], cosCoef=beta[k+nH+seq_len(nH)]));
}

# The recent profile from the initial states. The level and trend sit lagsModelMax-1
# steps before t=0 (the head refinement walks them to t=0 with the damping); a harmonic with
# s(t) = a sin(lambda t) + b cos(lambda t) holds 2cos(lambda) s(0) in its lag-1 state
# and -s(-1), -s(0) in the two cells of its lag-2 state; the ARMA initials sit in the
# last ARMA state
#' @keywords internal
tbats_profile <- function(states, armaInitial, struct, phi=1){
    profile <- matrix(0, struct$nComponents, struct$lagsModelMax);
    # One step back from (l_t, b_t) is (l_t - b_t, b_t / phi)
    trends <- states$trend/phi^(seq_len(struct$lagsModelMax)-1);
    profile[1, 1] <- states$level - sum(trends[-struct$lagsModelMax]);
    if(struct$trendIn){
        profile[2, 1] <- trends[struct$lagsModelMax];
    }
    if(struct$nHarmonics>0){
        frequency <- struct$harmonicTable$frequency;
        s0 <- states$cosCoef;
        sm1 <- -states$sinCoef*sin(frequency) + states$cosCoef*cos(frequency);
        profile[struct$harmonicRows, 1] <- 2*cos(frequency)*s0;
        profile[struct$harmonicRows+1, 1] <- -sm1;
        profile[struct$harmonicRows+1, 2] <- -s0;
    }
    if(struct$nArma>0){
        profile[struct$nComponents, 1:struct$armaLagMax] <- armaInitial;
    }
    return(profile);
}

# The identified initials read back from the states (columns up to t=0)
#' @keywords internal
tbats_initialsRead <- function(matVt, struct){
    L <- struct$lagsModelMax;
    states <- list(level=matVt[1, L], trend=if(struct$trendIn) matVt[2, L] else 0,
                   sinCoef=numeric(0), cosCoef=numeric(0));
    if(struct$nHarmonics>0){
        frequency <- struct$harmonicTable$frequency;
        rows <- struct$harmonicRows;
        s0 <- -matVt[rows+1, L];
        s1 <- matVt[rows, L] + matVt[rows+1, L-1];
        sm1 <- 2*cos(frequency)*s0 - s1;
        states$cosCoef <- s0;
        states$sinCoef <- (s0*cos(frequency) - sm1)/sin(frequency);
    }
    armaInitial <- if(struct$nArma>0) matVt[struct$nComponents, L-struct$armaLagMax+1:struct$armaLagMax] else numeric(0);
    return(list(states=states, arma=armaInitial));
}

# The labels of the harmonics, "j[period]"
#' @keywords internal
tbats_harmonicLabels <- function(struct){
    if(struct$nHarmonics==0){
        return(character(0));
    }
    return(paste0(struct$harmonicTable$j, "[", round(struct$harmonicTable$period, 4), "]"));
}

#### Parameters ####
# The names, starting values and bounds of the parameter vector
#' @keywords internal
tbats_B <- function(struct, armaSpec, armaStart, lambdaSpec, lambdaStart, distribution,
                    otherParameterEstimate, initialEstimate, bounds){
    nPeriods <- length(unique(struct$harmonicTable$period));
    periodsUsed <- struct$periods[sort(unique(struct$periodIndex))];
    B <- c(alpha=0.1,
           beta=if(struct$trendIn) 0.05,
           phi=if(struct$damped) 0.95,
           # No seasonal smoothing: on the boundary of the admissible region, which any
           # small value can cross
           setNames(rep(0, 2*nPeriods),
                    paste0(rep(c("gamma1[","gamma2["), nPeriods), rep(round(periodsUsed, 4), each=2), "]")[seq_len(2*nPeriods)]),
           setNames(armaStart, armaSpec$names));
    if(initialEstimate){
        nH <- struct$nHarmonics;
        harmonicNames <- c(paste0("sin", tbats_harmonicLabels(struct)), paste0("cos", tbats_harmonicLabels(struct)));
        harmonicNames <- harmonicNames[seq_len(2*nH)];
        B <- c(B, level=0, trend=if(struct$trendIn) 0,
               setNames(rep(0, 2*nH), harmonicNames),
               setNames(rep(0, struct$armaLagMax), if(struct$armaLagMax>0) paste0("ARMAState", seq_len(struct$armaLagMax))));
    }
    if(lambdaSpec$estimate){
        B <- c(B, lambda=lambdaStart);
    }
    if(distribution=="dgnorm" && otherParameterEstimate){
        B <- c(B, shape=2);
    }
    lb <- rep(-Inf, length(B));
    ub <- rep(Inf, length(B));
    if(bounds=="usual"){
        persistence <- names(B) %in% c("alpha","beta","phi");
        lb[persistence] <- 0;
        ub[persistence] <- 1;
    }
    lb[names(B)=="lambda"] <- 0;
    ub[names(B)=="lambda"] <- 1;
    lb[names(B)=="shape"] <- 0;
    return(list(B=B, lb=lb, ub=ub));
}

# The elements of the model for the parameter vector: the matrices, the initial
# deviations, lambda and the shape, and a penalty when the bounds are violated
#' @keywords internal
tbats_filler <- function(B, struct, armaSpec, lambdaSpec, other, initialEstimate, bounds, adamCpp){
    get <- function(name, default){
        return(if(any(names(B)==name)) B[[name]] else default);
    }
    alpha <- B[["alpha"]];
    beta <- get("beta", 0);
    phi <- get("phi", 1);
    periodsUsed <- struct$periods[sort(unique(struct$periodIndex))];
    gammaIndex <- match(struct$harmonicTable$period, periodsUsed);
    gamma1 <- B[grepl("^gamma1\\[", names(B))][gammaIndex];
    gamma2 <- B[grepl("^gamma2\\[", names(B))][gammaIndex];
    lambda <- get("lambda", lambdaSpec$value);
    shape <- get("shape", if(is.null(other)) 2 else other);
    penalty <- 0;

    matF <- struct$matF;
    vecG <- matrix(0, struct$nComponents, 1);
    w <- rep(1, struct$nComponents);
    vecG[1] <- alpha;
    if(struct$trendIn){
        matF[1:2, 2] <- phi;
        w[2] <- phi;
        vecG[2] <- beta;
    }
    if(struct$nHarmonics>0){
        frequency <- struct$harmonicTable$frequency;
        vecG[struct$harmonicRows] <- gamma1;
        vecG[struct$harmonicRows+1] <- sin(frequency)*gamma2 - cos(frequency)*gamma1;
    }
    if(struct$nArma>0){
        polynomials <- adamCpp$polynomialise(B[armaSpec$names], armaSpec$arOrders,
                                             rep(0, length(armaSpec$lags)), armaSpec$maOrders,
                                             TRUE, TRUE, numeric(0), armaSpec$lags);
        ar <- -polynomials$ariPolynomial[armaSpec$stateLags+1];
        ar[!is.finite(ar)] <- 0;
        ma <- polynomials$maPolynomial[armaSpec$stateLags+1];
        ma[!is.finite(ma)] <- 0;
        matF[struct$armaRows, struct$armaRows] <- ar %o% rep(1, struct$nArma);
        vecG[struct$armaRows] <- ar + ma;
        if(bounds!="none"){
            reflection <- max(polynomials$arReflection*(sum(armaSpec$arOrders)>0),
                              polynomials$maReflection*(sum(armaSpec$maOrders)>0));
            if(reflection>=1){
                penalty <- penalty + 1E+100*reflection;
            }
        }
    }

    if(lambda<0 || lambda>1 || shape<=0){
        penalty <- penalty + 1E+100;
    }
    if(bounds=="usual"){
        response <- alpha;
        if(struct$nHarmonics>0){
            response <- alpha + struct$responseCos %*% gamma1 + struct$responseSin %*% gamma2;
        }
        if(any(c(alpha, phi)<0) || any(c(alpha, phi)>1) || beta<0 || beta>alpha ||
           any(response<0) || any(response>1)){
            penalty <- penalty + 1E+100;
        }
    }
    else if(bounds=="admissible"){
        eigenValues <- tbats_eigens(matF, vecG, w, struct);
        if(any(eigenValues>1+1E-10)){
            penalty <- penalty + 1E+100*max(eigenValues);
        }
    }

    deviations <- NULL;
    if(initialEstimate){
        nH <- struct$nHarmonics;
        deviations <- list(level=B[["level"]], trend=get("trend", 0),
                           sinCoef=B[paste0("sin", tbats_harmonicLabels(struct))[seq_len(nH)]],
                           cosCoef=B[paste0("cos", tbats_harmonicLabels(struct))[seq_len(nH)]],
                           arma=if(struct$armaLagMax>0) B[paste0("ARMAState", seq_len(struct$armaLagMax))] else numeric(0));
    }
    return(list(matF=matF, vecG=vecG, w=w, phi=phi, lambda=lambda, shape=shape,
                deviations=deviations, penalty=penalty));
}

# The starting seasonal smoothing parameters for the admissible bounds: no smoothing
# sits on the boundary of the region, so the optimiser would start half in the
# penalty. The point of a small grid, shared by the periods, that is the furthest
# inside the region
#' @keywords internal
tbats_gammaStart <- function(B, struct, armaSpec, lambdaSpec, other, initialEstimate, adamCpp){
    gammas <- grepl("^gamma", names(B));
    grid <- expand.grid(gamma1=c(-0.01, -0.001, 0, 0.001, 0.01), gamma2=c(-0.01, -0.001, 0, 0.001, 0.01));
    eigenMax <- apply(grid, 1, function(gamma){
        BTest <- B;
        BTest[grepl("^gamma1", names(B))] <- gamma[1];
        BTest[grepl("^gamma2", names(B))] <- gamma[2];
        elements <- tbats_filler(BTest, struct, armaSpec, lambdaSpec, other, initialEstimate, "none", adamCpp);
        return(max(tbats_eigens(elements$matF, elements$vecG, elements$w, struct)));
    });
    best <- which.min(eigenMax);
    if(eigenMax[best]<1){
        B[grepl("^gamma1", names(B))] <- grid$gamma1[best];
        B[grepl("^gamma2", names(B))] <- grid$gamma2[best];
    }
    return(B);
}

# The moduli of the eigenvalues of the discount matrix of the level, trend and
# harmonics, on the lag-expanded form: a harmonic is (v1_t, v2_t, v2_{t-1})
#' @keywords internal
tbats_eigens <- function(matF, vecG, w, struct){
    nETS <- struct$nETS;
    nH <- struct$nHarmonics;
    k <- nETS + 3*nH;
    Fe <- matrix(0, k, k);
    ge <- rep(0, k);
    we <- rep(0, k);
    Fe[1:nETS, 1:nETS] <- matF[1:nETS, 1:nETS];
    ge[1:nETS] <- vecG[1:nETS];
    we[1:nETS] <- w[1:nETS];
    for(i in seq_len(nH)){
        rowOld <- struct$harmonicRows[i];
        rows <- nETS + 3*(i-1) + 1:3;
        eta <- matF[rowOld+0:1, rowOld];
        Fe[rows, rows] <- rbind(c(eta[1], 0, eta[1]), c(eta[2], 0, eta[2]), c(0, 1, 0));
        ge[rows[1:2]] <- vecG[rowOld+0:1];
        we[rows] <- c(1, 0, 1);
    }
    return(Mod(eigen(Fe - ge %o% we, only.values=TRUE)$values));
}

#### The fitter ####
# One fit of a fixed structure: the estimation of the parameters, the final fit and
# the forecasts, all in the space of the Box-Cox transformed data
#' @keywords internal
tbats_fit <- function(y, trendType, harmonicTable, armaSpec, lambdaSpec, distribution,
                      initial, checked){
    obs <- length(y);
    struct <- tbats_structure(trendType, harmonicTable, armaSpec, sort(unique(harmonicTable$period)));
    if(nrow(harmonicTable)==0){
        struct$periods <- numeric(0);
    }
    X <- tbats_design(obs, struct$trendIn, harmonicTable);
    qrX <- qr(X);
    lambdaStart <- tbats_lambdaStart(y, X, lambdaSpec);
    yBCStart <- tbats_boxCox(y, lambdaStart);
    logY <- if(lambdaSpec$estimate || lambdaStart!=1) sum(log(y)) else 0;

    adamCpp <- new(adamCore, struct$lagsModelAll, "A", if(struct$trendIn) "A" else "N", "N",
                   struct$nETS, 0, struct$nETS, struct$nComponents-struct$nETS, 0,
                   struct$nComponents, FALSE, FALSE);
    headLength <- adam_headLength(checked$headLengthUser, struct$lagsModelMax, obs);
    adamCpp$headLength <- headLength$flag;
    lookup <- adamProfileCreator(struct$lagsModelAll, struct$lagsModelMax, obs+max(checked$h, 1),
                                 headLength=headLength$geometry)$lookup;
    matVt <- matrix(0, struct$nComponents, obs+headLength$geometry);
    ot <- rep(1, obs);

    initialType <- initial;
    backcast <- any(initialType==c("backcasting","complete"));
    initialEstimate <- any(initialType==c("optimal","two-stage"));

    # The starting values of the ARMA from Hannan-Rissanen on the global residuals
    armaStart <- numeric(0);
    if(armaSpec$nParam>0){
        armaStart <- as.vector(arimaHRCpp(qr.resid(qrX, yBCStart), armaSpec$arOrders, armaSpec$maOrders,
                                          armaSpec$lags, TRUE, TRUE, numeric(0),
                                          rep(1, length(armaSpec$lags)), checked$bounds!="none"));
    }

    other <- checked$other;
    otherEstimate <- distribution=="dgnorm" && isTRUE(checked$otherParameterEstimate);
    BList <- tbats_B(struct, armaSpec, armaStart, lambdaSpec, lambdaStart, distribution,
                     otherEstimate, initialEstimate, checked$bounds);
    if(checked$bounds=="admissible" && struct$nHarmonics>0){
        BList$B <- tbats_gammaStart(BList$B, struct, armaSpec, lambdaSpec, other, initialEstimate, adamCpp);
    }

    #### The cost function ####
    fitStates <- function(elements){
        yBC <- if(lambdaSpec$estimate) tbats_boxCox(y, elements$lambda) else yBCStart;
        states <- tbats_globalStates(qr.coef(qrX, yBC), struct);
        armaInitial <- rep(0, struct$armaLagMax);
        if(!is.null(elements$deviations)){
            states$level <- states$level + elements$deviations$level;
            states$trend <- states$trend + elements$deviations$trend;
            states$sinCoef <- states$sinCoef + elements$deviations$sinCoef;
            states$cosCoef <- states$cosCoef + elements$deviations$cosCoef;
            armaInitial <- elements$deviations$arma;
        }
        profile <- tbats_profile(states, armaInitial, struct, elements$phi);
        fitted <- adamCpp$fit(matVt, matrix(elements$w, obs, struct$nComponents, byrow=TRUE),
                              elements$matF, elements$vecG, lookup, profile,
                              yBC, ot, backcast, checked$nIterations, "n");
        fitted$yBC <- yBC;
        fitted$profileInitial <- profile;
        return(fitted);
    }
    lossValue <- function(B, lossUsed){
        # nloptr drops the names
        names(B) <- names(BList$B);
        elements <- tbats_filler(B, struct, armaSpec, lambdaSpec, other, initialEstimate,
                                 checked$bounds, adamCpp);
        if(elements$penalty>0){
            return(elements$penalty);
        }
        fitted <- fitStates(elements);
        errors <- fitted$errors;
        if(any(lossUsed==c("likelihood","MSE","MAE","HAM","custom"))){
            value <- switch(lossUsed,
                            "likelihood"=-tbats_logLik(errors, distribution, elements$shape, obs) -
                                (elements$lambda-1)*logY,
                            "MSE"=sum(errors^2)/obs,
                            "MAE"=sum(abs(errors))/obs,
                            "HAM"=sum(sqrt(abs(errors)))/obs,
                            "custom"=checked$lossFunction(actual=fitted$yBC, fitted=fitted$fitted, B=B));
        }
        else{
            hor <- checked$h;
            adamErrors <- adamCpp$ferrors(fitted$states, matrix(elements$w, obs, struct$nComponents, byrow=TRUE),
                                          elements$matF, lookup, fitted$profileInitial, hor, fitted$yBC)$errors;
            value <- switch(lossUsed,
                            "MSEh"=sum(adamErrors[,hor]^2)/(obs-hor),
                            "TMSE"=sum(colSums(adamErrors^2)/(obs-hor)),
                            "GTMSE"=sum(log(colSums(adamErrors^2)/(obs-hor))),
                            "MSCE"=sum(rowSums(adamErrors)^2)/(obs-hor),
                            "GPL"=log(det(t(adamErrors) %*% adamErrors/(obs-hor))));
        }
        if(!is.finite(value)){
            value <- 1E+300;
        }
        return(value);
    }
    CF <- function(B){
        return(lossValue(B, checked$loss));
    }
    # The fit at any parameters, for reapply(): NULL where they violate the bounds
    fitter <- function(B){
        names(B) <- names(BList$B);
        elements <- tbats_filler(B, struct, armaSpec, lambdaSpec, other, initialEstimate,
                                 checked$bounds, adamCpp);
        if(elements$penalty>0){
            return(NULL);
        }
        fitted <- fitStates(elements);
        return(list(fitted=fitted$fitted, states=fitted$states, profile=fitted$profile,
                    matF=elements$matF, vecG=elements$vecG, w=elements$w, lambda=elements$lambda));
    }

    #### Estimation ####
    B <- BList$B;
    if(initialType=="two-stage"){
        # Backcast first, then optimise all from its parameters and initials
        backcastFit <- tbats_fit(y, trendType, harmonicTable, armaSpec, lambdaSpec, distribution,
                                 "complete", checked);
        common <- intersect(names(B), names(backcastFit$B));
        B[common] <- backcastFit$B[common];
        B <- tbats_deviations(B, backcastFit, struct, qrX, y, lambdaSpec);
    }
    if(!is.null(checked$B)){
        if(length(checked$B)!=length(B)){
            stop(paste0("The provided B has ", length(checked$B), " values, while the model needs ",
                        length(B), ": ", paste(names(B), collapse=", "), "."), call.=FALSE);
        }
        B[] <- checked$B;
    }
    lb <- if(is.null(checked$lb)) BList$lb else checked$lb;
    ub <- if(is.null(checked$ub)) BList$ub else checked$ub;
    res <- NULL;
    if(checked$modelDo=="estimate" && length(B)>0){
        # The harmonics make the surface flat: 40 evaluations per parameter, as in
        # adam(), stopped well short of the optimum on AirPassengers
        maxevalUsed <- if(is.null(checked$maxeval)) length(B)*200 else checked$maxeval;
        printLevel <- checked$print_level;
        if(printLevel==41){
            cat("Initial parameters:", B, "\n");
            printLevel <- 0;
        }
        opts <- list(algorithm=checked$algorithm, xtol_rel=checked$xtol_rel, xtol_abs=checked$xtol_abs,
                     ftol_rel=checked$ftol_rel, ftol_abs=checked$ftol_abs,
                     maxeval=maxevalUsed, maxtime=checked$maxtime, print_level=printLevel);
        res <- suppressWarnings(nloptr(B, CF, lb=lb, ub=ub, opts=opts));
        # Stuck on a penalty: restart from no smoothing, unless B was provided
        if(is.null(checked$B) && (is.infinite(res$objective) || res$objective>=1E+100)){
            B[grepl("^(alpha|beta|gamma)", names(B))] <- 0;
            res <- suppressWarnings(nloptr(B, CF, lb=lb, ub=ub, opts=opts));
        }
        if(checked$print_level>0){
            print(res);
        }
        B[] <- res$solution;
    }
    lossFinal <- CF(B);

    #### The final fit ####
    elements <- tbats_filler(B, struct, armaSpec, lambdaSpec, other, initialEstimate,
                             checked$bounds, adamCpp);
    fitted <- fitStates(elements);
    states <- fitted$states;
    if(headLength$geometry>struct$lagsModelMax){
        states <- states[, -c(1:(headLength$geometry-struct$lagsModelMax)), drop=FALSE];
    }
    rownames(states) <- struct$componentNames;
    initialRead <- tbats_initialsRead(states, struct);

    # The identified initials are counted whether they are optimised or backcast
    nInitials <- 1 + struct$trendIn + 2*struct$nHarmonics + struct$armaLagMax;
    nParamEstimated <- length(B)*(checked$modelDo=="estimate") + 1 + nInitials*backcast;
    logLikValue <- -lossValue(B, "likelihood");

    # The Hessian of the log-likelihood
    FI <- NA;
    if(isTRUE(checked$FI) && length(B)>0){
        FI <- -hessianCpp(function(BNew) -lossValue(BNew, "likelihood"), B, h=checked$stepSize);
        colnames(FI) <- rownames(FI) <- names(B);
    }

    scale <- tbats_scale(fitted$errors, distribution, elements$shape, obs);
    forecastBC <- NULL;
    if(checked$h>0){
        forecastBC <- adamCpp$forecast(matrix(elements$w, checked$h, struct$nComponents, byrow=TRUE),
                                       elements$matF,
                                       lookup[, headLength$geometry+obs+1:checked$h, drop=FALSE],
                                       fitted$profile, checked$h)$forecast;
    }

    return(list(B=B, res=res, lossValue=lossFinal,
                logLik=structure(logLikValue, nobs=obs, df=nParamEstimated, class="logLik"),
                nParamEstimated=nParamEstimated, nInitials=nInitials*backcast,
                struct=struct, armaSpec=armaSpec, elements=elements, fitted=fitted, states=states,
                initialRead=initialRead, scale=scale, forecastBC=forecastBC, FI=FI,
                trendType=trendType, initialType=initialType, distribution=distribution,
                adamCpp=adamCpp, lookup=lookup, headLength=headLength, fitter=fitter,
                y=y, lambdaSpec=lambdaSpec, checked=checked));
}

# The initial deviations from the global model that reproduce the initials of a fit
#' @keywords internal
tbats_deviations <- function(B, backcastFit, struct, qrX, y, lambdaSpec){
    lambda <- backcastFit$elements$lambda;
    states <- tbats_globalStates(qr.coef(qrX, tbats_boxCox(y, lambda)), struct);
    read <- backcastFit$initialRead;
    B[["level"]] <- read$states$level - states$level;
    if(struct$trendIn){
        B[["trend"]] <- read$states$trend - states$trend;
    }
    if(struct$nHarmonics>0){
        labels <- tbats_harmonicLabels(struct);
        B[paste0("sin", labels)] <- read$states$sinCoef - states$sinCoef;
        B[paste0("cos", labels)] <- read$states$cosCoef - states$cosCoef;
    }
    if(struct$armaLagMax>0){
        B[paste0("ARMAState", seq_len(struct$armaLagMax))] <- read$arma;
    }
    return(B);
}

#### Scale and likelihood ####
# The scale as in the ADAM monograph: sigma^2 for dnorm, s for the others
#' @keywords internal
tbats_scale <- function(errors, distribution, shape, obs){
    return(adam_scaler(distribution, "A", errors, NULL, obs, shape));
}

# The log-likelihood of the errors in the space of the transformed data
#' @keywords internal
tbats_logLik <- function(errors, distribution, shape, obs){
    scale <- tbats_scale(errors, distribution, shape, obs);
    return(sum(switch(distribution,
                      "dnorm"=dnorm(errors, 0, sqrt(scale), log=TRUE),
                      "dlaplace"=dlaplace(errors, 0, scale, log=TRUE),
                      "ds"=ds(errors, 0, scale, log=TRUE),
                      "dgnorm"=dgnorm(errors, 0, scale, shape, log=TRUE))));
}

#### The returned object ####
#' @keywords internal
tbats_return <- function(best, checked, cl, startTime, periods, harmonics, ICs, silent){
    struct <- best$struct;
    lambda <- best$elements$lambda;
    obs <- length(best$y);
    yClasses <- checked$yClasses;

    makeSeries <- function(values){
        if(any(yClasses=="ts")){
            return(ts(values, start=checked$yStart, frequency=checked$yFrequency));
        }
        return(zoo(values, order.by=checked$yInSampleIndex));
    }
    yFitted <- makeSeries(tbats_boxCoxInverse(best$fitted$fitted, lambda));
    errors <- makeSeries(best$fitted$errors);
    if(checked$h>0){
        yForecast <- tbats_boxCoxInverse(best$forecastBC, lambda);
        if(any(yClasses=="ts")){
            yForecast <- ts(yForecast, start=checked$yForecastStart, frequency=checked$yFrequency);
        }
        else{
            yForecast <- zoo(yForecast, order.by=checked$yForecastIndex);
        }
    }
    else{
        yForecast <- ts(NA, start=checked$yForecastStart, frequency=checked$yFrequency);
    }

    # The name: TBATS(lambda, {p,q}, phi, <m1,k1>, ...)
    armaSpec <- best$armaSpec;
    seasonalPart <- if(struct$nHarmonics>0){
        counts <- table(factor(struct$harmonicTable$period, levels=periods));
        paste0(", ", paste0("<", round(periods, 4), ",", as.vector(counts), ">")[counts>0], collapse="")
    } else "";
    phiValue <- if(struct$damped) round(best$B[["phi"]], 3) else "-";
    modelName <- paste0("TBATS(", round(lambda, 3), ", {", sum(armaSpec$arOrders), ",",
                        sum(armaSpec$maOrders), "}, ", phiValue, seasonalPart, ")");

    initialValue <- list(level=best$initialRead$states$level);
    if(struct$trendIn){
        initialValue$trend <- best$initialRead$states$trend;
    }
    if(struct$nHarmonics>0){
        initialValue$seasonal <- data.frame(period=struct$harmonicTable$period, j=struct$harmonicTable$j,
                                            sin=best$initialRead$states$sinCoef,
                                            cos=best$initialRead$states$cosCoef);
    }
    if(struct$nArma>0){
        initialValue$arma <- best$initialRead$arma;
    }

    parametersNumber <- matrix(0, 2, 5, dimnames=list(c("Estimated","Provided"),
                                                      c("nParamInternal","nParamXreg","nParamOccurrence",
                                                        "nParamScale","nParamAll")));
    parametersNumber[1,1] <- best$nParamEstimated - 1;
    parametersNumber[1,4] <- 1;
    parametersNumber[1,5] <- sum(parametersNumber[1,1:4]);
    parametersNumber[2,1] <- length(best$B)*(checked$modelDo=="use");
    parametersNumber[2,5] <- sum(parametersNumber[2,1:4]);

    yInSample <- checked$yInSample;
    yHoldout <- checked$yHoldout;
    # Keep the ts class of the data, as adam() does
    if(is.ts(yFitted)){
        yInSample <- ts(yInSample, start=start(yFitted), frequency=frequency(yFitted));
        if(!is.null(yHoldout) && is.ts(yForecast)){
            yHoldout <- ts(yHoldout, start=start(yForecast), frequency=frequency(yForecast));
        }
    }
    errormeasures <- NULL;
    if(checked$holdout && checked$h>0){
        errormeasures <- measures(yHoldout, yForecast, yInSample);
    }

    persistence <- best$elements$vecG[,1];
    names(persistence) <- struct$componentNames;
    matF <- best$elements$matF;
    dimnames(matF) <- list(struct$componentNames, struct$componentNames);
    matWt <- matrix(best$elements$w, obs, struct$nComponents, byrow=TRUE,
                    dimnames=list(NULL, struct$componentNames));
    orders <- list(ar=armaSpec$arOrders, i=rep(0, length(armaSpec$lags)), ma=armaSpec$maOrders);
    arma <- NULL;
    if(armaSpec$nParam>0){
        arPart <- best$B[grepl("^phi[0-9]", names(best$B))];
        maPart <- best$B[grepl("^theta", names(best$B))];
        arma <- list(ar=if(length(arPart)>0) arPart, ma=if(length(maPart)>0) maPart);
    }
    other <- if(best$distribution=="dgnorm") list(shape=best$elements$shape) else list();

    modelReturned <- structure(list(model=modelName, timeElapsed=Sys.time()-startTime,
                                    call=cl, data=yInSample, holdout=yHoldout,
                                    fitted=yFitted, residuals=errors, forecast=yForecast,
                                    states=t(best$states), accuracy=errormeasures,
                                    profile=best$fitted$profile, profileInitial=best$fitted$profileInitial,
                                    persistence=persistence, phi=if(struct$damped) best$B[["phi"]] else 1,
                                    transition=matF, measurement=matWt,
                                    initial=initialValue, initialType=best$initialType,
                                    orders=orders, arma=arma, armaSpec=armaSpec, armaLags=armaSpec$lags,
                                    lambda=lambda,
                                    harmonics=harmonics, periods=periods, trendType=best$trendType,
                                    nParam=parametersNumber,
                                    formula=checked$formula,
                                    loss=checked$loss, lossValue=best$lossValue, lossFunction=checked$lossFunction,
                                    logLik=best$logLik,
                                    ICs=ICs,
                                    distribution=best$distribution, other=other, bounds=checked$bounds,
                                    scale=best$scale, B=best$B, lags=c(1, periods),
                                    lagsAll=struct$lagsModelAll, res=best$res, FI=best$FI,
                                    adamCpp=best$adamCpp, fitter=best$fitter),
                               class=c("tbats","adam","smooth"));
    if(!silent){
        plot(modelReturned, 7);
    }
    return(modelReturned);
}

#### Methods ####
# The model in the space of the Box-Cox transform, as an adam object: the adam
# methods work there, and the tbats ones transform their results back
#' @keywords internal
tbats_boxCoxObject <- function(object){
    lambda <- object$lambda;
    objectBC <- object;
    class(objectBC) <- c("adam","smooth");
    yBC <- tbats_boxCox(actuals(object), lambda);
    objectBC$data[] <- yBC;
    objectBC$fitted[] <- yBC - residuals(object);
    objectBC$forecast[] <- tbats_boxCox(object$forecast, lambda);
    if(!is.null(object$holdout)){
        objectBC$holdout[] <- suppressWarnings(tbats_boxCox(object$holdout, lambda));
    }
    return(objectBC);
}

# The quantiles of the transformed data map onto those of the data, so the
# forecasts and the bounds are transformed back; the point forecast is the median
#' @keywords internal
tbats_boxCoxForecast <- function(result, object){
    lambda <- object$lambda;
    for(element in c("mean","lower","upper")){
        if(!is.null(result[[element]])){
            result[[element]][] <- tbats_boxCoxInverse(result[[element]], lambda);
        }
    }
    if(is.matrix(result$scenarios)){
        result$scenarios[] <- tbats_boxCoxInverse(result$scenarios, lambda);
    }
    result$model <- object;
    return(result);
}

# The refits at parameters drawn from their distribution. Each draw has its own
# lambda, so its states are in the space of its own transform; the refitted values
# are transformed back. A draw outside the bounds is pulled towards the estimate,
# to the last point of the segment between them that satisfies the bounds: the
# estimate often lies on the boundary of the admissible region, where most draws
# would otherwise be rejected.
#' @export
reapply.tbats <- function(object, nsim=1000, type=c("opg","hessian","bootstrap"),
                          bootstrap=FALSE, heuristics=NULL, ...){
    startTime <- Sys.time();
    type <- covarTypeResolver(type, bootstrap);
    parameters <- coef(object);
    vcovMatrix <- vcov(object, type=type, heuristics=heuristics, ...);
    eigenMin <- min(eigen(vcovMatrix, symmetric=TRUE, only.values=TRUE)$values);
    if(eigenMin<0){
        vcovMatrix <- vcovMatrix + diag(abs(eigenMin)+1e-10, nrow(vcovMatrix));
    }
    draws <- matrix(MASS::mvrnorm(nsim, parameters, vcovMatrix), ncol=length(parameters),
                    dimnames=list(NULL, names(parameters)));
    refits <- vector("list", nsim);
    for(i in 1:nsim){
        refit <- object$fitter(draws[i,]);
        if(is.null(refit)){
            # Bisect the share of the deviation from the estimate that keeps the bounds
            inside <- 0;
            outside <- 1;
            for(iteration in 1:20){
                share <- (inside+outside)/2;
                if(is.null(object$fitter(parameters + share*(draws[i,]-parameters)))){
                    outside <- share;
                }
                else{
                    inside <- share;
                }
            }
            draws[i,] <- parameters + inside*(draws[i,]-parameters);
            refit <- object$fitter(draws[i,]);
        }
        refits[[i]] <- refit;
    }
    obs <- nobs(object);
    nComponents <- ncol(object$states);
    lagsModelMax <- max(object$lagsAll);
    lambdas <- sapply(refits, function(refit) refit$lambda);
    refitted <- matrix(sapply(1:nsim, function(i) tbats_boxCoxInverse(refits[[i]]$fitted, lambdas[i])),
                       obs, nsim, dimnames=list(NULL, paste0("nsim",1:nsim)));
    if(any(class(actuals(object))=="ts")){
        refitted <- ts(refitted, start=start(actuals(object)), frequency=frequency(actuals(object)));
    }
    states <- array(sapply(refits, function(refit){
        return(refit$states[, ncol(refit$states)-(obs+lagsModelMax)+1:(obs+lagsModelMax), drop=FALSE]);
    }), c(nComponents, obs+lagsModelMax, nsim));
    return(structure(list(timeElapsed=Sys.time()-startTime,
                          y=actuals(object), states=states, refitted=refitted,
                          fitted=fitted(object), model=object$model,
                          transition=array(sapply(refits, function(refit) refit$matF),
                                           c(nComponents, nComponents, nsim)),
                          measurement=array(sapply(refits, function(refit) rep(refit$w, each=obs)),
                                            c(obs, nComponents, nsim)),
                          persistence=matrix(sapply(refits, function(refit) refit$vecG[,1]), nComponents, nsim),
                          profile=array(sapply(refits, function(refit) refit$profile),
                                        c(nComponents, lagsModelMax, nsim)),
                          randomParameters=draws, lambda=lambdas,
                          errors=sapply(1:nsim, function(i){
                              return(tbats_boxCox(as.numeric(actuals(object)), lambdas[i]) - refits[[i]]$fitted);
                          })),
                     class="reapply"));
}

# The forecasts with the uncertainty of the parameters: for each draw of reapply(),
# its point forecasts ("confidence") or its simulated paths with the scale of its
# own residuals ("prediction"), in the space of its own transform and transformed
# back. The point forecast stays the median of the model.
#' @export
reforecast.tbats <- function(object, h=10, newdata=NULL, occurrence=NULL,
                             interval=c("prediction", "confidence", "none"),
                             level=0.95, side=c("both","upper","lower"), cumulative=FALSE,
                             nsim=100, type=c("opg","hessian","bootstrap"),
                             bootstrap=FALSE, heuristics=NULL, ...){
    interval <- match.arg(interval);
    side <- match.arg(side);
    if(cumulative && object$lambda!=1){
        stop("Cumulative forecasts of tbats() are only available for lambda=1.", call.=FALSE);
    }
    objectRefitted <- reapply(object, nsim=nsim, type=type, bootstrap=bootstrap, heuristics=heuristics, ...);
    obs <- nobs(object);
    lagsModelAll <- object$lagsAll;
    lagsModelMax <- max(lagsModelAll);
    nComponents <- length(lagsModelAll);
    lookup <- adamProfileCreator(lagsModelAll, lagsModelMax, obs+h)$lookup[,-c(1:(obs+lagsModelMax)),drop=FALSE];
    draws <- objectRefitted$randomParameters;
    dfScale <- adam_dfScale(object);
    paths <- vector("list", nsim);
    for(j in 1:nsim){
        matWt <- matrix(objectRefitted$measurement[1,,j], h, nComponents, byrow=TRUE);
        matF <- objectRefitted$transition[,,j];
        profile <- matrix(objectRefitted$profile[,,j], nComponents, lagsModelMax);
        lambda <- objectRefitted$lambda[j];
        if(interval=="prediction"){
            shape <- if(any(colnames(draws)=="shape")) draws[j,"shape"] else object$other$shape;
            scale <- adam_scaleDebias(tbats_scale(objectRefitted$errors[,j], object$distribution, shape, obs),
                                      object$distribution, obs, dfScale);
            errors <- array(adam_errorsSimulate(h*nsim, object$distribution, scale, list(shape=shape), dfScale),
                            c(h, nsim, 1));
            simulated <- object$adamCpp$reforecast(errors, array(1, c(h, nsim, 1)),
                                                   array(matWt, c(h, nComponents, 1)),
                                                   array(matF, c(nComponents, nComponents, 1)),
                                                   matrix(objectRefitted$persistence[,j], nComponents, 1),
                                                   lookup, array(profile, c(nComponents, lagsModelMax, 1)),
                                                   "A")$data;
            paths[[j]] <- matrix(tbats_boxCoxInverse(simulated, lambda), h, nsim);
        }
        else{
            paths[[j]] <- matrix(tbats_boxCoxInverse(object$adamCpp$forecast(matWt, matF, lookup, profile, h)$forecast,
                                                     lambda), h, 1);
        }
    }
    paths <- do.call(cbind, paths);
    if(cumulative){
        paths <- matrix(colSums(paths), 1);
    }
    levelLow <- switch(side, "both"=(1-level)/2, "upper"=rep(0, length(level)), "lower"=1-level);
    levelUp <- switch(side, "both"=(1+level)/2, "upper"=level, "lower"=rep(1, length(level)));
    yLower <- t(apply(paths, 1, quantile, probs=levelLow, na.rm=TRUE, names=FALSE));
    yUpper <- t(apply(paths, 1, quantile, probs=levelUp, na.rm=TRUE, names=FALSE));
    if(length(level)==1){
        yLower <- matrix(yLower, ncol=1);
        yUpper <- matrix(yUpper, ncol=1);
    }
    yLower[, levelLow==0] <- 0;
    yUpper[, levelUp==1] <- Inf;
    pointForecast <- forecast(tbats_boxCoxObject(object), h=h, interval="none")$mean;
    yForecast <- pointForecast;
    yForecast[] <- tbats_boxCoxInverse(pointForecast, object$lambda);
    if(cumulative){
        yForecast <- sum(yForecast);
    }
    makeLike <- function(values){
        values <- ts(values, start=start(pointForecast), frequency=frequency(pointForecast));
        return(values);
    }
    yLower <- makeLike(yLower);
    yUpper <- makeLike(yUpper);
    colnames(yLower) <- paste0("Lower bound (", levelLow*100, "%)");
    colnames(yUpper) <- paste0("Upper bound (", levelUp*100, "%)");
    return(structure(list(mean=yForecast, lower=yLower, upper=yUpper, model=object,
                          level=level, interval=interval, side=side, cumulative=cumulative, h=h,
                          scenarios=FALSE),
                     class=c("adam.forecast","smooth.forecast","forecast")));
}

#' @export
forecast.tbats <- function(object, h=10, newdata=NULL, occurrence=NULL,
                           interval=c("none", "prediction", "confidence", "simulated",
                                      "approximate", "semiparametric", "nonparametric",
                                      "empirical","complete"),
                           level=0.95, side=c("both","upper","lower"), cumulative=FALSE, nsim=NULL,
                           scenarios=FALSE, ...){
    if(cumulative && object$lambda!=1){
        stop("Cumulative forecasts of tbats() are only available for lambda=1: ",
             "the sums of the transformed values do not transform back.", call.=FALSE);
    }
    objectBC <- tbats_boxCoxObject(object);
    # The parameter uncertainty comes from reforecast(), which reapplies the model
    # with the lambda of each draw and returns the bounds of the data
    if(h>0 && any(interval[1]==c("confidence","complete"))){
        return(reforecast(object, h=h, newdata=newdata, occurrence=occurrence,
                          interval=switch(interval[1], "confidence"="confidence", "prediction"),
                          level=level, side=match.arg(side), cumulative=cumulative,
                          nsim=if(is.null(nsim)) 100 else nsim, ...));
    }
    result <- forecast(objectBC, h=h, newdata=newdata, occurrence=occurrence,
                       interval=interval, level=level, side=side, cumulative=cumulative, nsim=nsim,
                       scenarios=scenarios, ...);
    return(tbats_boxCoxForecast(result, object));
}

#' @export
predict.tbats <- function(object, newdata=NULL, interval=c("none", "confidence", "prediction"),
                          level=0.95, side=c("both","upper","lower"), ...){
    result <- predict(tbats_boxCoxObject(object), newdata=newdata, interval=interval,
                      level=level, side=side, ...);
    return(tbats_boxCoxForecast(result, object));
}

# The log-densities of the data: those of the transformed data and the Jacobian
#' @export
pointLik.tbats <- function(object, log=TRUE, ...){
    likValues <- pointLik(tbats_boxCoxObject(object), log=TRUE) +
        (object$lambda-1)*log(as.numeric(actuals(object)));
    if(!log){
        likValues <- exp(likValues);
    }
    return(likValues);
}

# The OPG covariance of the parameters: the scores of the log-densities, refitting
# the model at each perturbed parameter
#' @keywords internal
covarOPGtbats <- function(object, stepSize=.Machine$double.eps^(1/4)){
    parameterValues <- coef(object);
    y <- actuals(object);
    perturbedPointLik <- function(j, delta){
        clone <- object;
        if(!is.null(j)){
            clone$B[j] <- clone$B[j]+delta;
        }
        modelLocal <- try(suppressWarnings(tbats(y, model=clone, h=0)), silent=TRUE);
        if(inherits(modelLocal, "try-error")){
            return(NULL);
        }
        return(as.numeric(pointLik(modelLocal)));
    }
    return(covarOPGCore(object, parameterValues, perturbedPointLik, stepSize));
}

#' @export
simulate.tbats <- function(object, nsim=1, seed=NULL, obs=nobs(object), ...){
    result <- simulate(tbats_boxCoxObject(object), nsim=nsim, seed=seed, obs=obs, ...);
    result$data[] <- tbats_boxCoxInverse(result$data, object$lambda);
    result$model <- object$model;
    return(result);
}
