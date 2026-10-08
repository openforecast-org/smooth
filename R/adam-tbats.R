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
#' The point forecasts are by default the skeleton (\code{point="skeleton"}, see
#' \link[smooth]{forecast.adam}): the inverse Box-Cox transform of the point forecasts
#' of the transformed data, the median of the forecast distribution.
#' \code{point="median"} gives the same values, and \code{point="mean"} the mean:
#' with \code{distribution="dnorm"} by Gauss-Hermite quadrature over the normal
#' forecast distribution of the transformed data, with the variance of the
#' approximate interval (exactly \eqn{\exp(\mu+\sigma^2/2)} for \eqn{\lambda=0}),
#' otherwise from \code{nsim} simulated paths transformed back. With \eqn{\lambda=0},
#' \code{ds} and \code{dgnorm} with a shape below one have no mean, so the
#' simulated one is unstable (a warning).
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
#' @param y Vector or ts object, containing the data needed to be forecasted. The
#' missing values (\code{NA}) are gaps: the global model is fitted to the observed
#' values, the states move through the transition without an update at the gaps, the
#' likelihood and the information criteria count the observed values only, and the
#' ARMA screen takes zeros at the gaps of the residuals. The residuals are \code{NA}
#' there, and the fitted values are the predictions of the model.
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
#' @param xreg The explanatory variables: a numeric matrix or data frame with a
#' column per variable and a row per observation of \code{y}, holdout included; the
#' rows beyond it give the future values for the forecast stored in the model (\code{h}).
#' They enter the model in the space of the Box-Cox transformed data, as in the ETSX of
#' \link[smooth]{adam}. Factors should be converted into dummy variables. The future
#' values for \code{forecast()} are taken from its \code{newdata}. The observations
#' with a missing value of a regressor are dropped (gaps of \code{y}, with a
#' warning), and their fitted values are \code{NA}; the future values cannot be missing.
#' @param regressors How to treat the explanatory variables: \code{"use"} them as
#' they are (constant coefficients), \code{"select"} them as \link[smooth]{adam}
#' does (\code{stepwise()} on the errors of the model chosen without them, which is
#' refitted with the selected ones and kept if it improves the information criterion),
#' or \code{"adapt"} their coefficients over time (one smoothing parameter per
#' regressor, \code{delta1}, ..., as in \link[smooth]{adam}).
#' @param occurrence The occurrence of an intermittent demand. \code{"none"} (the
#' default) fits the zeros as values, with a warning. A fitted \link[smooth]{om} (or
#' \code{omg()}, \code{oes()}) model is used as it is, and a numeric vector gives the
#' probabilities of occurrence (or 0/1) of the in-sample observations, and possibly of
#' the horizon. \code{"fixed"}, \code{"auto"}, \code{"odds-ratio"},
#' \code{"inverse-odds-ratio"}, \code{"direct"} and \code{"general"} fit
#' \code{om(y, model="ZXN", lags=1)} of that type: a level-only occurrence with the
#' trend selected, as the seasonal pattern of the probability is hard to find in zeros
#' and ones (a seasonal one can be provided as a fitted \code{om()}). The sizes are then
#' modelled on the non-zero observations: the Box-Cox transform, its Jacobian and the
#' global model use them only, and the states evolve through the zeros. The
#' log-likelihood and the number of parameters include those of the occurrence model, and
#' the fitted values and forecasts are the probability times those of the sizes (see
#' \code{point} in \link[smooth]{forecast.adam}).
#' @param distribution The distribution of the error term in the space of the
#' Box-Cox transformed data: \code{"auto"} (default), \code{"dnorm"},
#' \code{"dlaplace"}, \code{"ds"} or \code{"dgnorm"} (the shape is estimated unless
#' \code{shape} is provided in ellipsis). With \code{"auto"}, the model is selected
#' with \code{"dgnorm"}, whose shape nests the others (2 is the normal, 1 the Laplace
#' and 0.5 the S distribution), and the named distribution closest to the estimated
#' shape on the log scale is then fitted on the selected structure (from the default
#' starting values and from the estimates of \code{"dgnorm"}, the higher likelihood
#' kept). Its information criterion is added to \code{ICs}. With a loss other than the
#' likelihood, \code{"auto"} is \code{"dlaplace"} for \code{"MAE"}, \code{"ds"} for
#' \code{"HAM"} and \code{"dnorm"} otherwise, as in \link[smooth]{adam}.
#' @param loss The loss function, see \link[smooth]{adam}. A custom loss is a function
#' of \code{actual}, \code{fitted} and \code{B}, as in \link[smooth]{adam}, which
#' receives the observed values with demand and their fitted values in the space of the
#' Box-Cox transformed data (lambda is then 1, as with the other losses than the
#' likelihood).
#' @param ic The information criterion used in the selection.
#' @param h The forecast horizon.
#' @param holdout If \code{TRUE}, the holdout of the size \code{h} is taken from
#' the data.
#' @param persistence The smoothing parameters, as in \link[smooth]{adam}: a vector
#' (level, trend, a pair \eqn{\gamma_1, \gamma_2} per period of \code{lags} above 1,
#' the smoothing parameters of the regressors with \code{regressors="adapt"}), used only
#' when the structure is not selected, or a named list with \code{level}, \code{trend},
#' \code{seasonal} (a list with one vector \code{c(gamma1, gamma2)} per period) and
#' \code{xreg}, where only the provided elements are fixed. The values for the components
#' that the model does not have are ignored.
#' @param phi The damping parameter, used with \code{trend="damped"}. With
#' \code{trend="auto"}, it is estimated with a warning, as in \link[smooth]{adam}.
#' @param initial The initialisation: \code{"backcasting"} (default),
#' \code{"optimal"}, \code{"two-stage"}, \code{"complete"} (the same as
#' backcasting here) or \code{"gradient"}, which solves for the initial states by
#' profiling the estimation loss at the current parameters, as in
#' \link[smooth]{adam}: the model is additive in the space of the transformed data,
#' so this is a linear solve (with weighted least squares sweeps for the robust and
#' multistep losses). The coefficients of the regressors are estimated with the
#' other parameters. With a custom loss, it is switched to \code{"backcasting"} with a
#' warning.
#' The initial states can also be provided, in the space of the Box-Cox transformed data
#' (so they are meaningful with a provided \code{lambda}), as in \link[smooth]{adam}: a
#' vector (level, trend, then for each period the sine coefficients of its harmonics
#' followed by their cosine coefficients, the ARMA states, the coefficients of the
#' regressors), used only when the structure is not selected, or a named list with
#' \code{level}, \code{trend}, \code{seasonal} (a list with one vector of the sine and then
#' the cosine coefficients per period, which needs \code{harmonics}), \code{arma} and
#' \code{xreg}. The states not provided are estimated, with \code{"optimal"}
#' initialisation.
#' @param arma The parameters of the ARMA, as in \link[smooth]{adam}: a list with
#' \code{ar} and \code{ma}, or a vector with the AR and then the MA parameters of each
#' lag, lag by lag. If provided, the orders are not selected.
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
#' @return Object of class \code{c("adamTBATS","adam","smooth")} is returned with
#' similar elements to the \link[smooth]{adam} function, together with
#' \code{lambda}, \code{harmonics}, \code{periods} and the information criteria of
#' the fitted candidates in \code{ICs}. The methods of \link[smooth]{adam} apply:
#' \code{forecast()} and \code{predict()} work in the space of the transformed data
#' and transform the results back (the point forecasts are the skeleton, the
#' medians, by default; the cumulative ones are the sums of the skeletons, or the
#' mean or median of the sums of the paths simulated and transformed back, whose
#' quantiles give the bounds), \code{interval="confidence"} and
#' \code{"complete"} come from \code{reforecast()}, which refits the model at each
#' draw of the parameters with its own \eqn{\lambda}, and \code{confint()} keeps
#' the intervals inside the bounds of the model.
#'
#' @seealso \code{\link[smooth]{adam}, \link[smooth]{ces}, \link[smooth]{msarima}}
#'
#' @examples
#' ourModel <- tbats(AirPassengers, orders=list(ar=0, ma=0, select=FALSE), h=12, holdout=TRUE)
#' forecast(ourModel, h=12, interval="prediction")
#'
#' @rdname tbats
#' @export
tbats <- function(y, lags=c(1, frequency(y)), harmonics=NULL,
                  trend=c("auto","none","additive","damped"),
                  lambda=NULL, orders=list(ar=3, ma=3, select=TRUE),
                  xreg=NULL, regressors=c("use","select","adapt"),
                  occurrence=c("none","auto","fixed","general","odds-ratio","inverse-odds-ratio","direct"),
                  distribution=c("auto","dnorm","dlaplace","ds","dgnorm"),
                  loss=c("likelihood","MSE","MAE","HAM","MSEh","TMSE","GTMSE","MSCE","GPL"),
                  ic=c("AICc","AIC","BIC","BICc"), h=0, holdout=FALSE,
                  persistence=NULL, phi=NULL,
                  initial=c("backcasting","optimal","two-stage","complete","gradient"), arma=NULL,
                  bounds=c("admissible","usual","none"), silent=TRUE, model=NULL, ...){
    startTime <- Sys.time();
    cl <- match.call();
    ellipsis <- list(...);
    yName <- paste0(deparse(substitute(y)), collapse="");

    trend <- match.arg(trend);
    distribution <- match.arg(distribution);
    # A function is a custom loss, handled by the checker
    if(!is.function(loss)){
        loss <- match.arg(loss);
    }
    ic <- match.arg(ic);
    # The provided initials are estimated where they are missing, as in adam()
    initialProvided <- NULL;
    if(is.numeric(initial) || is.list(initial)){
        initialProvided <- initial;
        initial <- "optimal";
    }
    initial <- match.arg(initial);
    bounds <- match.arg(bounds);
    regressors <- match.arg(regressors);
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
        # The occurrence model of the model, as it is
        if(!is.null(model$occurrence)){
            occurrence <- model$occurrence;
        }
        trend <- model$trendType;
        armaSpecProvided <- model$armaSpec;
        distribution <- model$distribution;
        loss <- if(model$loss=="custom") model$lossFunction else model$loss;
        initial <- model$initialType;
        bounds <- model$bounds;
        # The regressors of the model, selected or not, are used as they are
        if(!is.null(model$xregNames)){
            regressors <- if(model$regressors=="adapt") "adapt" else "use";
            if(is.null(xreg)){
                xreg <- rbind(model$data[, model$xregNames, drop=FALSE],
                              model$holdout[, model$xregNames, drop=FALSE]);
            }
        }
        # The values provided to the model, as they were applied
        persistence <- model$provided$persistence;
        phi <- model$provided$phi;
        arma <- model$provided$arma;
        initialProvided <- model$provided$initial;
        if(initial=="provided"){
            initial <- "optimal";
        }
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
    # The provided values, as adam() takes them: phi needs a preselected trend, and an unnamed
    # vector needs a preselected structure to be matched with it
    selection <- trend=="auto" || is.null(harmonics) || isTRUE(orders$select) || regressors=="select";
    if(!is.null(phi) && trend=="auto"){
        warning("Predefined phi can only be used with a preselected trend. Changing to estimation.",
                call.=FALSE);
        phi <- NULL;
    }
    if(!is.null(persistence) && !is.list(persistence) && selection){
        warning(paste0("Predefined persistence vector can only be used with a preselected structure.\n",
                       "Changing to estimation of persistence values."), call.=FALSE);
        persistence <- NULL;
    }
    if(!is.null(initialProvided) && !is.list(initialProvided) && selection){
        warning(paste0("Predefined initials vector can only be used with a preselected structure.\n",
                       "Changing to estimation of initials."), call.=FALSE);
        initialProvided <- NULL;
    }
    initialList <- is.list(initialProvided);
    if(initialList && !is.null(initialProvided$seasonal) && is.null(harmonics)){
        warning("Initial seasonal coefficients need the harmonics to be provided. Estimating them.",
                call.=FALSE);
        initialProvided$seasonal <- NULL;
    }
    if(initialList && !is.null(initialProvided$arma) && isTRUE(orders$select) && is.null(arma)){
        warning("Initial ARMA states need the orders to be preselected. Estimating them.", call.=FALSE);
        initialProvided$arma <- NULL;
    }
    if(initialList && !is.null(initialProvided$xreg) && regressors=="select"){
        warning("Initial values of the regressors cannot be used with their selection. Estimating them.",
                call.=FALSE);
        initialProvided$xreg <- NULL;
    }
    # The ARMA parameters fix its orders
    if(!is.null(arma)){
        orders$select <- FALSE;
    }
    if(!is.null(initialProvided) && initial!="optimal"){
        initial <- "optimal";
    }

    # "auto": the structure is selected with dgnorm, then refitted with the named
    # distribution closest to its shape; the other losses imply one, as in adam()
    distributionAuto <- distribution=="auto";
    if(distributionAuto){
        lossName <- if(is.function(loss)) "custom" else loss;
        distribution <- switch(lossName, "likelihood"="dgnorm", "MAE"="dlaplace", "HAM"="ds", "dnorm");
        distributionAuto <- distribution=="dgnorm";
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
    # The initialisation the checker settled on (backcasting for "gradient" with a
    # custom loss)
    initial <- checked$initialType;
    checked$headLengthUser <- ellipsis$headLength;
    checked$modelDo <- modelDo;
    checked$bounds <- bounds;
    checked[["tbatsProvided"]] <- list(persistence=persistence, phi=phi, arma=arma, initial=initialProvided);
    # The missing values are gaps: the checker filled them, but the global model, the fit and
    # the likelihood use the observed values only
    yInSample <- as.vector(checked$yInSample);
    yInSample[checked$yNAValues[seq_along(yInSample)]] <- NA;
    xregSpec <- tbats_xreg(xreg, length(yInSample), checked$h, regressors);
    yInSample[xregSpec$missing] <- NA;
    occurrenceSpec <- tbats_occurrence(occurrence, yInSample, checked$loss);
    # Under a name of its own: checked has occurrence elements of adam(), which $ matches
    # partially
    checked[["tbatsOccurrence"]] <- occurrenceSpec;
    otLogical <- occurrenceSpec$otLogical;

    #### The structure ####
    periods <- sort(unique(lags[lags>1]));
    lambdaSpec <- tbats_lambdaSpec(lambdaProvided, yInSample[otLogical], checked$loss);
    armaSpec <- if(is.null(armaSpecProvided)) tbats_armaSpec(orders, lags) else armaSpecProvided;
    # The provided AR or MA parameters of a wrong number are estimated, as in adam()
    if(!is.null(arma)){
        armaKinds <- if(is.list(arma)) arma[intersect(c("ar","ma"), names(arma))] else list(arma=arma);
        armaNeeded <- c(ar=sum(armaSpec$arOrders), ma=sum(armaSpec$maOrders), arma=armaSpec$nParam);
        for(kind in names(armaKinds)){
            if(length(armaKinds[[kind]])!=armaNeeded[[kind]]){
                warning(paste0("The number of provided ", toupper(kind), " parameters is ",
                               length(armaKinds[[kind]]), ", while the orders imply ", armaNeeded[[kind]],
                               ". Estimating them."), call.=FALSE);
                armaKinds[kind] <- list(NULL);
            }
        }
        checked[["tbatsProvided"]]["arma"] <- list(if(is.list(arma)) armaKinds else armaKinds$arma);
    }
    # With the selection, the trend is chosen without ARMA, and without the regressors
    armaSpecFit <- if(armaSpec$select) tbats_armaBuild(0, 0, 1) else armaSpec;
    xregSpecFit <- if(regressors=="select") NULL else xregSpec;
    trendTypes <- if(trend=="auto") c("none","additive","damped") else trend;

    # Harmonics from the global model, at the starting value of lambda
    if(is.null(harmonics)){
        harmonics <- tbats_harmonicsSelect(yInSample, periods, any(trendTypes!="none"),
                                           lambdaSpec, checked$icFunction, ic, xregSpecFit$data, otLogical);
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
    # The trends are warm started from the model without it, with no trend smoothing
    candidates <- vector("list", length(trendTypes));
    for(i in seq_along(trendTypes)){
        candidates[[i]] <- tbats_fit(yInSample, trendTypes[i], harmonicTable, armaSpecFit, lambdaSpec,
                                     distribution, initial, checked, xregSpecFit,
                                     if(i>1) c(candidates[[1]]$B, beta=0));
    }
    ICs <- sapply(candidates, function(candidate){
        return(tbats_IC(candidate$logLik, ic));
    });
    names(ICs) <- trendTypes;
    best <- candidates[[which.min(ICs)]];

    # The regressors selected by stepwise() on the errors of the best model, as in adam();
    # the model with them is kept only if it beats the one without
    if(regressors=="select" && !is.null(xregSpec)){
        shapeEstimated <- any(names(best$B)=="shape");
        selected <- names(adam_xreg_selector(best$fitted$errors[otLogical], xregSpec$data[otLogical,,drop=FALSE],
                                             sum(otLogical), ic,
                                             length(best$B)+1-shapeEstimated, distribution, "none",
                                             best$elements$shape)$initialXreg);
        xregSpecSelected <- tbats_xregSubset(xregSpec, make.names(selected));
        if(!is.null(xregSpecSelected)){
            candidate <- tbats_fit(yInSample, best$trendType, harmonicTable, armaSpecFit, lambdaSpec,
                                   distribution, initial, checked, xregSpecSelected, best$B);
            icCandidate <- tbats_IC(candidate$logLik, ic);
            if(icCandidate<min(ICs)){
                best <- candidate;
                xregSpecFit <- xregSpecSelected;
            }
            ICs[paste0(candidate$trendType, "+X(", paste(xregSpecSelected$names, collapse=","), ")")] <- icCandidate;
        }
    }

    # The ARMA orders, screened on the residuals of the global model at the lambda of the
    # best one: on its errors, the adaptive level hides an AR in a near-unit MA root.
    # The winner is fitted and kept only if it beats the models without ARMA
    if(armaSpec$select && armaSpec$nParam>0){
        X <- tbats_design(length(yInSample), best$trendType!="none", harmonicTable,
                          xregSpecFit$data)[otLogical,,drop=FALSE];
        residuals <- tbats_qrResid(tbats_qr(X), tbats_boxCox(yInSample[otLogical], best$elements$lambda));
        armaSpecBest <- tbats_armaSelect(tbats_gapped(residuals, otLogical), armaSpec, distribution,
                                         best$elements$shape, best$nParamEstimated, ic, otLogical);
        if(armaSpecBest$nParam>0){
            # Warm started from the best model, with no ARMA
            candidate <- tbats_fit(yInSample, best$trendType, harmonicTable, armaSpecBest, lambdaSpec,
                                   distribution, initial, checked, xregSpecFit,
                                   c(best$B, setNames(rep(0, armaSpecBest$nParam), armaSpecBest$names)));
            icCandidate <- tbats_IC(candidate$logLik, ic);
            if(icCandidate<min(ICs)){
                best <- candidate;
            }
            ICs[paste0(candidate$trendType, "+ARMA(", paste(armaSpecBest$arOrders, collapse=","), ";",
                       paste(armaSpecBest$maOrders, collapse=","), ")")] <- icCandidate;
        }
    }

    if(distributionAuto){
        best <- tbats_closest(best, yInSample, harmonicTable, lambdaSpec, initial, checked, xregSpecFit);
        ICs[best$distribution] <- tbats_IC(best$logLik, ic);
    }

    return(tbats_return(best, checked, cl, startTime, periods, harmonics, ICs, silent));
}

# The named distribution closest to the shape of dgnorm on the log scale (S 0.5, Laplace
# 1, normal 2), fitted on the structure selected with dgnorm from the default start and
# from its estimates without the shape, the higher likelihood kept
tbats_closest <- function(best, y, harmonicTable, lambdaSpec, initial, checked, xregSpec){
    shapes <- c(ds=0.5, dlaplace=1, dnorm=2);
    distribution <- names(shapes)[which.min(abs(log(best$elements$shape)-log(shapes)))];
    checked["other"] <- list(NULL);
    checked$otherParameterEstimate <- FALSE;
    keep <- names(best$B)!="shape";
    checked["lb"] <- list(checked$lb[keep]);
    checked["ub"] <- list(checked$ub[keep]);
    starts <- list(NULL, best$B[keep]);
    fits <- lapply(starts, function(B){
        checked["B"] <- list(B);
        return(tbats_fit(y, best$trendType, harmonicTable, best$armaSpec, lambdaSpec, distribution,
                         initial, checked, xregSpec));
    });
    return(fits[[which.max(sapply(fits, function(fit) as.numeric(fit$logLik)))]]);
}

#### Explanatory variables ####
# The explanatory variables: numeric, one column per variable, named; the in-sample
# rows and those of the horizon (the holdout or the future), the last row repeated
# when they do not reach it
tbats_xreg <- function(xreg, obsInSample, h, regressors){
    if(is.null(xreg)){
        return(NULL);
    }
    xreg <- as.data.frame(xreg);
    if(!all(sapply(xreg, is.numeric))){
        stop("xreg should contain numeric variables only: convert the factors into dummy variables.",
             call.=FALSE);
    }
    xreg <- as.matrix(xreg);
    if(is.null(colnames(xreg))){
        colnames(xreg) <- paste0("x", seq_len(ncol(xreg)));
    }
    colnames(xreg) <- make.names(colnames(xreg), unique=TRUE);
    if(nrow(xreg)<obsInSample){
        stop("xreg has fewer rows than the in-sample data.", call.=FALSE);
    }
    # The observations with a missing regressor are dropped: they are gaps of the response
    missing <- rowSums(!is.finite(xreg[1:obsInSample,,drop=FALSE]))>0;
    if(any(missing)){
        warning("xreg has missing values: the observations of their rows are dropped.", call.=FALSE);
    }
    if(nrow(xreg)<obsInSample+h){
        warning("xreg does not cover the horizon h. Repeating its last row.", call.=FALSE);
        xreg <- xreg[c(1:nrow(xreg), rep(nrow(xreg), obsInSample+h-nrow(xreg))),,drop=FALSE];
    }
    if(h>0 && any(!is.finite(xreg[obsInSample+1:h,]))){
        stop("xreg has missing values in the horizon: the forecasts need them.", call.=FALSE);
    }
    return(list(data=xreg[1:obsInSample,,drop=FALSE],
                future=if(h>0) xreg[obsInSample+1:h,,drop=FALSE],
                names=colnames(xreg), number=ncol(xreg), regressors=regressors, missing=missing));
}

# Some of the regressors, used as they are (NULL if none)
tbats_xregSubset <- function(xregSpec, names){
    if(length(names)==0){
        return(NULL);
    }
    return(list(data=xregSpec$data[, names, drop=FALSE],
                future=if(!is.null(xregSpec$future)) xregSpec$future[, names, drop=FALSE],
                names=names, number=length(names), regressors="use"));
}

# The measurement matrix for the rows of the regressors (or a number of rows). Their
# missing values are placeholders: the fit skips those observations
tbats_matWt <- function(w, struct, rows, xregData=NULL){
    matWt <- matrix(w, rows, struct$nComponents, byrow=TRUE);
    if(struct$nXreg>0){
        matWt[, struct$xregRows] <- xregData;
        matWt[is.na(matWt)] <- 0;
    }
    return(matWt);
}

#### Box-Cox ####
# The Box-Cox transform and its inverse
tbats_boxCox <- function(y, lambda){
    if(lambda==0){
        return(log(y));
    }
    return((y^lambda-1)/lambda);
}

tbats_boxCoxInverse <- function(z, lambda){
    z <- as.numeric(z);
    if(lambda==0){
        return(exp(z));
    }
    return(pmax(lambda*z+1, 0)^(1/lambda));
}

# The occurrence of an intermittent demand: a provided om() / omg() / oes() model, the
# provided probabilities (or 0/1), or an om() with the level only (and the trend
# selected) for the occurrence type: its seasonal pattern is hard to find in zeros and
# ones. The non-zero observations, and the log-likelihood and parameters of the
# occurrence, which are added to those of the sizes. The missing values are neither:
# otLogical marks the observed (and non-zero) values that the sizes are fitted to
tbats_occurrence <- function(occurrence, y, loss){
    obs <- length(y);
    observed <- !is.na(y);
    otLogical <- observed & (y!=0);
    otLogical[!observed] <- FALSE;
    none <- list(model=NULL, otLogical=observed, logLik=0, nParam=0, pFitted=rep(1, obs));
    if(is.occurrence(occurrence)){
        omModel <- occurrence;
        if(length(fitted(omModel))!=obs){
            stop("The occurrence model should be fitted to the in-sample data.", call.=FALSE);
        }
    }
    else if(is.numeric(occurrence) || is.logical(occurrence)){
        probabilities <- as.numeric(occurrence);
        if(length(probabilities)<obs || any(probabilities<0 | probabilities>1)){
            stop("The provided occurrence should have the probabilities (in [0, 1]) of the in-sample ",
                 "observations, and possibly of the horizon.", call.=FALSE);
        }
        pFitted <- probabilities[1:obs];
        if(any(pFitted[otLogical]==0) || any(pFitted[!otLogical & observed]==1)){
            stop("The provided occurrence contradicts the data.", call.=FALSE);
        }
        omModel <- list(occurrence="provided", fitted=pFitted, forecast=probabilities[-(1:obs)],
                        logLik=sum(log(pFitted[otLogical])) + sum(log(1-pFitted[!otLogical & observed])));
        return(list(model=omModel, otLogical=otLogical, logLik=omModel$logLik, nParam=0,
                    pFitted=pFitted));
    }
    else{
        occurrence <- match.arg(occurrence[1], c("none","auto","fixed","general","odds-ratio",
                                                 "inverse-odds-ratio","direct"));
        if(occurrence=="none"){
            if(any(y==0, na.rm=TRUE)){
                warning("The data has zeros, which are fitted as values. For an intermittent demand, ",
                        "use the occurrence argument.", call.=FALSE);
            }
            return(none);
        }
        # No zeros: nothing to model
        if(all(otLogical[observed])){
            return(none);
        }
        omModel <- om(y, model="ZXN", lags=1, occurrence=occurrence, silent=TRUE);
    }
    if(any(loss==c("MSEh","TMSE","GTMSE","MSCE","GPL"))){
        stop("The multistep losses are not available with an occurrence model.", call.=FALSE);
    }
    return(list(model=omModel, otLogical=otLogical, logLik=as.numeric(logLik(omModel)),
                nParam=nparam(omModel), pFitted=as.vector(fitted(omModel))));
}

# The probabilities of occurrence for the horizon: forecasts of the occurrence model, or
# the provided ones (the last of them repeated)
tbats_pForecast <- function(object, h){
    occurrence <- object$occurrence;
    if(is.null(occurrence) || h<=0){
        return(rep(1, max(h, 0)));
    }
    if(is.occurrence(occurrence)){
        return(as.vector(forecast(occurrence, h=h)$mean));
    }
    pForecast <- c(occurrence$forecast, rep(tail(c(occurrence$fitted, occurrence$forecast), 1), h));
    return(pForecast[1:h]);
}

# The transformed sizes, zero where there is no demand (the fitter skips them)
tbats_boxCoxSizes <- function(y, lambda, otLogical){
    yBC <- rep(0, length(y));
    yBC[otLogical] <- tbats_boxCox(y[otLogical], lambda);
    return(yBC);
}

# How lambda is treated: estimated in [0, 1] only with the likelihood and positive
# data; otherwise fixed (provided, or 1 with a message)
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

# The design of the global model: an intercept, a trend, the Fourier terms and the
# regressors
tbats_design <- function(obs, trendIn, harmonicTable, xregData=NULL){
    times <- 1:obs;
    X <- matrix(1, obs, 1);
    if(trendIn){
        X <- cbind(X, times);
    }
    if(nrow(harmonicTable)>0){
        angles <- outer(times, harmonicTable$frequency);
        X <- cbind(X, sin(angles), cos(angles));
    }
    return(cbind(X, xregData));
}

# The least squares of the global model through the Householder QR shared with Python
# (src/headers/olsCore.h, BLAS-free), so that the two agree to the last bit; R's qr()
# is LINPACK's and numpy's LAPACK's, which round differently
tbats_qr <- function(X){
    return(householderQRCpp(X));
}

tbats_qrCoef <- function(qrX, y){
    return(as.vector(householderCoefCpp(qrX$qr, qrX$qraux, qrX$rDiag, y)));
}

tbats_qrResid <- function(qrX, y){
    return(as.vector(householderResidCpp(qrX$qr, qrX$qraux, qrX$rDiag, y)));
}

# The maximum of the profile log-likelihood of the global model in lambda, with the
# Jacobian
tbats_lambdaProfile <- function(y, qrX){
    obs <- length(y);
    logY <- sum(log(y));
    profile <- function(lambda){
        rss <- sum(tbats_qrResid(qrX, tbats_boxCox(y, lambda))^2);
        return(-obs/2*log(rss/obs) + (lambda-1)*logY);
    }
    # Rounded: Brent's search finds it to about 1e-4, and the last bits of the
    # least squares differ between linear algebra libraries (the Python port)
    return(round(optimize(profile, c(0, 1), maximum=TRUE)$maximum, 8));
}

# The starting value of lambda (or its fixed value) for a design
tbats_lambdaStart <- function(y, X, lambdaSpec){
    if(!lambdaSpec$estimate){
        return(lambdaSpec$value);
    }
    return(tbats_lambdaProfile(y, tbats_qr(X)));
}

# The number of harmonics of each period by the information criterion of the global
# model, one period at a time, stopping after two harmonics without improvement
tbats_harmonicsSelect <- function(y, periods, trendIn, lambdaSpec, icFunction, ic, xregData=NULL,
                                  otLogical=rep(TRUE, length(y))){
    harmonics <- rep(0, length(periods));
    if(length(periods)==0){
        return(harmonics);
    }
    kMax <- pmax(ceiling(periods/2)-1, 0);
    # The non-zero observations of an occurrence model, with their time index
    obsAll <- length(y);
    design <- function(harmonicsTest){
        return(tbats_design(obsAll, trendIn, tbats_harmonics(periods, harmonicsTest),
                            xregData)[otLogical,,drop=FALSE]);
    }
    y <- y[otLogical];
    obs <- length(y);
    lambda <- tbats_lambdaStart(y, design(pmin(kMax, 3)), lambdaSpec);
    yBC <- tbats_boxCox(y, lambda);
    icValue <- function(harmonicsTest){
        X <- design(harmonicsTest);
        if(ncol(X)>=obs-1){
            return(Inf);
        }
        rss <- sum(tbats_qrResid(tbats_qr(X), yBC)^2);
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

# The residuals of the observed values at their places and zeros at the gaps (the zeros
# of the occurrence), so that the lags of the ARMA stay aligned. The residuals of the
# global model are centred, and so is the series in Hannan-Rissanen
tbats_gapped <- function(residuals, otLogical){
    gapped <- numeric(length(otLogical));
    gapped[otLogical] <- residuals;
    return(gapped);
}

# The ARMA orders screened with Hannan-Rissanen on the residuals of a model without ARMA,
# one lag at a time from the largest. The IC of a candidate comes from the likelihood of
# its innovations on a common sample of the observed values: the rest of the model is
# common to all of them
tbats_armaSelect <- function(errors, armaSpec, distribution, shape, nParamBase, ic,
                             observed=rep(TRUE, length(errors))){
    lags <- armaSpec$lags;
    arOrders <- maOrders <- rep(0, length(lags));
    obs <- length(errors);
    nDrop <- min(sum(armaSpec$arOrders*lags), floor(obs/4));
    used <- seq_len(obs)>nDrop & observed;
    obsUsed <- sum(used);
    for(i in order(lags, decreasing=TRUE)){
        screen <- arimaHRSelectCpp(errors, arOrders, maOrders, lags, i-1,
                                   armaSpec$arOrders[i], armaSpec$maOrders[i], TRUE);
        nParam <- nParamBase + sum(arOrders[-i]+maOrders[-i]) + rowSums(screen$orders);
        ICs <- sapply(seq_len(nrow(screen$orders)), function(j){
            innovations <- screen$innovations[used, j];
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
tbats_IC <- function(logLikValue, ic){
    return(switch(ic, "AIC"=AIC(logLikValue), "AICc"=AICc(logLikValue),
                  "BIC"=BIC(logLikValue), "BICc"=BICc(logLikValue)));
}

#### Structure ####
# The parts of the model that do not depend on the parameters: the lags, the rows of
# the components, the fixed transition blocks of the harmonics and the matrices of
# the usual bounds
tbats_structure <- function(trendType, harmonicTable, armaSpec, periods, xregSpec=NULL){
    trendIn <- trendType!="none";
    nETS <- 1+trendIn;
    nHarmonics <- nrow(harmonicTable);
    nArma <- length(armaSpec$stateLags);
    nXreg <- if(is.null(xregSpec)) 0 else xregSpec$number;
    lagsModelAll <- c(rep(1, nETS), rep(c(1, 2), nHarmonics), armaSpec$stateLags, rep(1, nXreg));
    nComponents <- length(lagsModelAll);
    harmonicRows <- nETS + 2*seq_len(nHarmonics) - 1;
    armaRows <- nETS + 2*nHarmonics + seq_len(nArma);
    xregRows <- nETS + 2*nHarmonics + nArma + seq_len(nXreg);
    matF <- diag(0, nComponents);
    matF[1, 1] <- 1;
    matF[cbind(xregRows, xregRows)] <- 1;
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
                        if(nArma>0) paste0("ARMAState", seq_len(nArma)),
                        xregSpec$names);
    return(list(trendType=trendType, trendIn=trendIn, damped=trendType=="damped",
                nETS=nETS, nHarmonics=nHarmonics, nArma=nArma, nXreg=nXreg, nComponents=nComponents,
                lagsModelAll=lagsModelAll, lagsModelMax=max(lagsModelAll),
                harmonicRows=harmonicRows, armaRows=armaRows, xregRows=xregRows, matF=matF,
                xreg=xregSpec, xregAdapt=nXreg>0 && xregSpec$regressors=="adapt",
                harmonicTable=harmonicTable, periods=periods, periodIndex=periodIndex,
                # The periods with harmonics, and the one of each harmonic among them
                periodsUsed=periods[sort(unique(periodIndex))],
                gammaIndex=match(periodIndex, sort(unique(periodIndex))),
                responseCos=cos(angles), responseSin=sin(angles),
                armaLagMax=if(nArma>0) max(armaSpec$stateLags) else 0,
                componentNames=componentNames));
}

#### Initial states ####
# The initial states of the global model: the level and trend at t=0 and the
# Fourier coefficients of the harmonics
tbats_globalStates <- function(beta, struct){
    beta[!is.finite(beta)] <- 0;
    nH <- struct$nHarmonics;
    k <- 1+struct$trendIn;
    return(list(level=beta[1], trend=if(struct$trendIn) beta[2] else 0,
                sinCoef=beta[k+seq_len(nH)], cosCoef=beta[k+nH+seq_len(nH)],
                xreg=beta[k+2*nH+seq_len(struct$nXreg)]));
}

# The recent profile from the initial states. The level and trend sit lagsModelMax-1
# steps before t=0 (the head refinement walks them to t=0 with the damping); a harmonic with
# s(t) = a sin(lambda t) + b cos(lambda t) holds 2cos(lambda) s(0) in its lag-1 state
# and -s(-1), -s(0) in the two cells of its lag-2 state; the ARMA initials sit in the
# last ARMA state
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
        profile[struct$armaRows[struct$nArma], 1:struct$armaLagMax] <- armaInitial;
    }
    if(struct$nXreg>0){
        profile[struct$xregRows, 1] <- states$xreg;
    }
    return(profile);
}

# The identified initials read back from the states (columns up to t=0)
tbats_initialsRead <- function(matVt, struct){
    L <- struct$lagsModelMax;
    states <- list(level=matVt[1, L], trend=if(struct$trendIn) matVt[2, L] else 0,
                   sinCoef=numeric(0), cosCoef=numeric(0), xreg=matVt[struct$xregRows, L]);
    if(struct$nHarmonics>0){
        frequency <- struct$harmonicTable$frequency;
        rows <- struct$harmonicRows;
        s0 <- -matVt[rows+1, L];
        s1 <- matVt[rows, L] + matVt[rows+1, L-1];
        sm1 <- 2*cos(frequency)*s0 - s1;
        states$cosCoef <- s0;
        states$sinCoef <- (s0*cos(frequency) - sm1)/sin(frequency);
    }
    armaInitial <- if(struct$nArma>0) matVt[struct$armaRows[struct$nArma], L-struct$armaLagMax+1:struct$armaLagMax] else numeric(0);
    return(list(states=states, arma=armaInitial));
}

# The labels of the harmonics, "j[period]"
tbats_harmonicLabels <- function(struct){
    if(struct$nHarmonics==0){
        return(character(0));
    }
    return(paste0(struct$harmonicTable$j, "[", round(struct$harmonicTable$period, 4), "]"));
}

#### Provided parameters ####
# The parameters and initial states provided by the user for a structure, as in adam():
# persistence, phi and the ARMA on the names of B, the initials as the states in the space
# of the transformed data, which replace the global ones (and the B entries they make
# redundant). The persistence is a vector (level, trend, a pair per period, the deltas of
# the regressors) or a list with level, trend, seasonal (a pair per period) and xreg; the
# initials are a vector or a list with level, trend, seasonal (per period, the sines and
# then the cosines of its harmonics), arma and xreg; the ARMA is a vector or a list with ar
# and ma, ordered lag-wise
tbats_provided <- function(provided, struct, armaSpec){
    nPeriods <- length(struct$periods);
    nXreg <- struct$nXreg;
    # The positions of an unnamed vector of values per period after the first ones
    splitPeriods <- function(values, first, sizes){
        ends <- first+cumsum(sizes);
        return(lapply(seq_len(nPeriods), function(i) values[ends[i]-sizes[i]+seq_len(sizes[i])]));
    }
    rowsPeriod <- lapply(struct$periods, function(period) which(struct$harmonicTable$period==period));

    values <- numeric(0);
    persistence <- provided$persistence;
    if(!is.null(persistence) && !is.list(persistence)){
        k <- 1+struct$trendIn;
        persistence <- list(level=persistence[1], trend=if(struct$trendIn) persistence[2],
                            seasonal=splitPeriods(persistence, k, rep(2, nPeriods)),
                            xreg=persistence[-seq_len(k+2*nPeriods)]);
    }
    if(!is.null(persistence$level)){
        values["alpha"] <- persistence$level;
    }
    if(!is.null(persistence$trend) && struct$trendIn){
        values["beta"] <- persistence$trend;
    }
    for(i in seq_along(persistence$seasonal)){
        if(length(rowsPeriod[[i]])>0){
            values[paste0(c("gamma1[","gamma2["), round(struct$periods[i], 4), "]")] <- persistence$seasonal[[i]];
        }
    }
    if(length(persistence$xreg)>0 && struct$xregAdapt){
        values[paste0("delta", seq_len(nXreg))] <- persistence$xreg;
    }
    if(!is.null(provided$phi) && struct$damped){
        values["phi"] <- provided$phi;
    }
    arma <- provided$arma;
    if(!is.null(arma) && armaSpec$nParam>0){
        if(is.list(arma)){
            if(!is.null(arma$ar)){
                values[armaSpec$names[grepl("^phi", armaSpec$names)]] <- arma$ar;
            }
            if(!is.null(arma$ma)){
                values[armaSpec$names[grepl("^theta", armaSpec$names)]] <- arma$ma;
            }
        }
        else{
            values[armaSpec$names] <- arma;
        }
    }

    # The initials, NA where they are estimated
    initial <- provided$initial;
    if(!is.null(initial) && !is.list(initial)){
        k <- 1+struct$trendIn;
        sizes <- 2*sapply(rowsPeriod, length);
        initial <- list(level=initial[1], trend=if(struct$trendIn) initial[2],
                        seasonal=splitPeriods(initial, k, sizes),
                        arma=initial[k+sum(sizes)+seq_len(struct$armaLagMax)],
                        xreg=initial[k+sum(sizes)+struct$armaLagMax+seq_len(nXreg)]);
    }
    states <- list(level=NA, trend=NA, sinCoef=rep(NA, struct$nHarmonics),
                   cosCoef=rep(NA, struct$nHarmonics), xreg=rep(NA, nXreg));
    if(!is.null(initial$level)){
        states$level <- initial$level;
    }
    if(!is.null(initial$trend) && struct$trendIn){
        states$trend <- initial$trend;
    }
    for(i in seq_along(initial$seasonal)){
        rows <- rowsPeriod[[i]];
        if(length(initial$seasonal[[i]])!=2*length(rows)){
            stop(paste0("The initial seasonal coefficients of the period ", struct$periods[i], " should be ",
                        2*length(rows), " values: the sines and then the cosines of its harmonics."),
                 call.=FALSE);
        }
        states$sinCoef[rows] <- initial$seasonal[[i]][seq_along(rows)];
        states$cosCoef[rows] <- initial$seasonal[[i]][length(rows)+seq_along(rows)];
    }
    if(length(initial$xreg)>0){
        states$xreg[] <- initial$xreg;
    }
    armaStates <- if(length(initial$arma)>0 && struct$armaLagMax>0) initial$arma else NULL;
    labels <- tbats_harmonicLabels(struct);
    drop <- c(if(!is.na(states$level)) "level", if(!is.na(states$trend)) "trend",
              paste0("sin", labels)[!is.na(states$sinCoef)], paste0("cos", labels)[!is.na(states$cosCoef)],
              if(!is.null(armaStates)) paste0("ARMAState", seq_len(struct$armaLagMax)),
              struct$xreg$names[!is.na(states$xreg)]);
    return(list(B=values, states=states, arma=armaStates, drop=drop,
                number=length(values)+sum(!is.na(unlist(states)))+length(armaStates),
                initial=length(drop)>0));
}

#### Parameters ####
# The names, starting values and bounds of the parameter vector
tbats_B <- function(struct, armaSpec, armaStart, lambdaSpec, lambdaStart, distribution,
                    otherParameterEstimate, initialEstimate, xregEstimate, bounds){
    nPeriods <- length(unique(struct$harmonicTable$period));
    periodsUsed <- struct$periodsUsed;
    B <- c(alpha=0.1,
           beta=if(struct$trendIn) 0.05,
           phi=if(struct$damped) 0.95,
           # No seasonal smoothing: on the boundary of the admissible region, which any
           # small value can cross
           setNames(rep(0, 2*nPeriods),
                    paste0(rep(c("gamma1[","gamma2["), nPeriods), rep(round(periodsUsed, 4), each=2), "]")[seq_len(2*nPeriods)]),
           setNames(rep(0.01, struct$nXreg*struct$xregAdapt),
                    if(struct$xregAdapt) paste0("delta", seq_len(struct$nXreg))),
           setNames(armaStart, armaSpec$names));
    if(initialEstimate){
        nH <- struct$nHarmonics;
        harmonicNames <- c(paste0("sin", tbats_harmonicLabels(struct)), paste0("cos", tbats_harmonicLabels(struct)));
        harmonicNames <- harmonicNames[seq_len(2*nH)];
        B <- c(B, level=0, trend=if(struct$trendIn) 0,
               setNames(rep(0, 2*nH), harmonicNames),
               setNames(rep(0, struct$armaLagMax), if(struct$armaLagMax>0) paste0("ARMAState", seq_len(struct$armaLagMax))));
    }
    # The regressors: deviations from the global model
    if(xregEstimate){
        B <- c(B, setNames(rep(0, struct$nXreg), struct$xreg$names));
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
        persistence <- names(B) %in% c("alpha","beta","phi") | grepl("^delta", names(B));
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
tbats_filler <- function(B, struct, armaSpec, lambdaSpec, other, initialEstimate, bounds, adamCpp,
                         xregEstimate=FALSE){
    get <- function(name, default){
        return(if(any(names(B)==name)) B[[name]] else default);
    }
    alpha <- B[["alpha"]];
    beta <- get("beta", 0);
    phi <- get("phi", 1);
    gamma1 <- B[grepl("^gamma1\\[", names(B))][struct$gammaIndex];
    gamma2 <- B[grepl("^gamma2\\[", names(B))][struct$gammaIndex];
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

    if(struct$xregAdapt){
        deltas <- B[paste0("delta", seq_len(struct$nXreg))];
        vecG[struct$xregRows] <- deltas;
        if(bounds=="usual" && (any(deltas<0) || any(deltas>1))){
            penalty <- penalty + 1E+100;
        }
        # The averaged condition of adam() for the regressors, separately from the rest
        else if(bounds=="admissible"){
            # over the rows of the observations the fit takes
            xregObserved <- struct$xreg$data[complete.cases(struct$xreg$data),,drop=FALSE];
            xregEigens <- abs(smoothEigensR(matrix(deltas), diag(struct$nXreg), xregObserved,
                                            rep(1L, struct$nXreg), TRUE, nrow(xregObserved),
                                            TRUE, struct$nXreg, FALSE));
            if(any(xregEigens>1+1E-10)){
                penalty <- penalty + 1E+100*max(xregEigens);
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
    xregDeviations <- if(xregEstimate) B[struct$xreg$names] else rep(0, struct$nXreg);
    return(list(matF=matF, vecG=vecG, w=w, phi=phi, lambda=lambda, shape=shape,
                deviations=deviations, xregDeviations=xregDeviations, penalty=penalty));
}

# The starting seasonal smoothing parameters for the admissible bounds: no smoothing
# sits on the boundary of the region, so the optimiser would start half in the
# penalty. The point of a small grid, shared by the periods, that is the furthest
# inside the region
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
# harmonics. A harmonic is reduced to (s_t, v2_t) with s_t = v1_t + v2_{t-1}: the
# lag-expanded form (v1_t, v2_t, v2_{t-1}) only adds a zero eigenvalue, as the discount
# matrix sends v1_t - v2_{t-1} to zero. The eigenvalues come from the LAPACK-free routine
# shared with Python (eigenModuliCore): the optima often lie on the boundary, where two
# LAPACK builds disagreed in the last bit on which parameters were admissible
tbats_eigens <- function(matF, vecG, w, struct){
    nETS <- struct$nETS;
    nH <- struct$nHarmonics;
    k <- nETS + 2*nH;
    Fe <- matrix(0, k, k);
    ge <- rep(0, k);
    we <- rep(0, k);
    Fe[1:nETS, 1:nETS] <- matF[1:nETS, 1:nETS];
    ge[1:nETS] <- vecG[1:nETS];
    we[1:nETS] <- w[1:nETS];
    for(i in seq_len(nH)){
        rowOld <- struct$harmonicRows[i];
        rows <- nETS + 2*(i-1) + 1:2;
        Fe[rows, rows] <- cbind(matF[rowOld+0:1, rowOld], c(1, 0));
        ge[rows] <- vecG[rowOld+0:1];
        we[rows[1]] <- 1;
    }
    return(eigenModuliCpp(Fe - ge %o% we));
}

# The starting values within a hair of a finite bound, moved onto it: NLopt's default
# initial step shrinks to the distance to the bound, so the simplex of Nelder-Mead
# collapses within ~1e-14 of it (nloptr then fails and returns the start), while on the
# bound it steps inwards. The values outside the bounds are left as they are
tbats_ontoBounds <- function(B, lb, ub, tol=1e-10){
    nearUpper <- is.finite(ub) & ub-B>=0 & ub-B<tol*pmax(1, abs(ub));
    nearLower <- is.finite(lb) & B-lb>=0 & B-lb<tol*pmax(1, abs(lb));
    B[nearUpper] <- ub[nearUpper];
    B[nearLower] <- lb[nearLower];
    return(B);
}

#### The fitter ####
# One fit of a fixed structure: the estimation of the parameters, the final fit and
# the forecasts, all in the space of the Box-Cox transformed data
tbats_fit <- function(y, trendType, harmonicTable, armaSpec, lambdaSpec, distribution,
                      initial, checked, xregSpec=NULL, BStart=NULL){
    obs <- length(y);
    struct <- tbats_structure(trendType, harmonicTable, armaSpec, sort(unique(harmonicTable$period)),
                              xregSpec);
    if(nrow(harmonicTable)==0){
        struct$periods <- numeric(0);
    }
    # With an occurrence model, the sizes: the global model, the transform and its
    # Jacobian on the non-zero observations, which keep their time index
    occurrenceSpec <- checked[["tbatsOccurrence"]];
    otLogical <- occurrenceSpec$otLogical;
    obsNonzero <- sum(otLogical);
    X <- tbats_design(obs, struct$trendIn, harmonicTable, xregSpec$data)[otLogical,,drop=FALSE];
    qrX <- tbats_qr(X);
    lambdaStart <- tbats_lambdaStart(y[otLogical], X, lambdaSpec);
    yBCStart <- tbats_boxCoxSizes(y, lambdaStart, otLogical);
    logY <- if(lambdaSpec$estimate || lambdaStart!=1) sum(log(y[otLogical])) else 0;

    adamCpp <- new(adamCore, struct$lagsModelAll, "A", if(struct$trendIn) "A" else "N", "N",
                   struct$nETS, 0, struct$nETS, struct$nComponents-struct$nETS-struct$nXreg, struct$nXreg,
                   struct$nComponents, FALSE, FALSE);
    headLength <- adam_headLength(checked$headLengthUser, struct$lagsModelMax, obs);
    adamCpp$headLength <- headLength$flag;
    # The harmonics make F large and mostly zeros
    adamCpp$sparseTransition <- TRUE;
    lookup <- adamProfileCreator(struct$lagsModelAll, struct$lagsModelMax, obs+max(checked$h, 1),
                                 headLength=headLength$geometry)$lookup;
    matVt <- matrix(0, struct$nComponents, obs+headLength$geometry);
    ot <- otLogical*1;

    initialType <- initial;
    backcast <- any(initialType==c("backcasting","complete"));
    # The initials of the states that the fit determines (backcast or solved)
    initialsProfiled <- backcast || initialType=="gradient";
    initialEstimate <- any(initialType==c("optimal","two-stage"));
    # The coefficients of the regressors are estimated unless all is backcast, as in adam()
    xregEstimate <- struct$nXreg>0 && initialType!="complete";

    # The starting values of the ARMA from Hannan-Rissanen on the global residuals
    armaStart <- numeric(0);
    if(armaSpec$nParam>0){
        armaStart <- as.vector(arimaHRCpp(tbats_gapped(tbats_qrResid(qrX, yBCStart[otLogical]), otLogical),
                                          armaSpec$arOrders, armaSpec$maOrders,
                                          armaSpec$lags, TRUE, TRUE, numeric(0),
                                          rep(1, length(armaSpec$lags)), checked$bounds!="none"));
    }

    other <- checked$other;
    otherEstimate <- distribution=="dgnorm" && isTRUE(checked$otherParameterEstimate);
    BList <- tbats_B(struct, armaSpec, armaStart, lambdaSpec, lambdaStart, distribution,
                     otherEstimate, initialEstimate, xregEstimate, checked$bounds);
    # The provided values: in the full vector (the starting gammas around them), out of B
    provided <- tbats_provided(checked[["tbatsProvided"]], struct, armaSpec);
    BAll <- BList$B;
    BAll[names(provided$B)] <- provided$B;
    if(checked$bounds=="admissible" && struct$nHarmonics>0){
        BAll <- tbats_gammaStart(BAll, struct, armaSpec, lambdaSpec, other, initialEstimate, adamCpp);
        BAll[names(provided$B)] <- provided$B;
    }
    estimated <- !(names(BAll) %in% c(names(provided$B), provided$drop));
    BList <- list(B=BAll[estimated], lb=BList$lb[estimated], ub=BList$ub[estimated]);
    # The full vector for the parameters of B (named), the provided ones included
    BFull <- function(B){
        BAll[names(BList$B)] <- B;
        return(BAll);
    }

    #### The cost function ####
    # The transformed data, the initial profile and the measurement of the elements
    fitInputs <- function(elements){
        yBC <- if(lambdaSpec$estimate) tbats_boxCoxSizes(y, elements$lambda, otLogical) else yBCStart;
        states <- tbats_globalStates(tbats_qrCoef(qrX, yBC[otLogical]), struct);
        armaInitial <- rep(0, struct$armaLagMax);
        if(!is.null(elements$deviations)){
            states$level <- states$level + elements$deviations$level;
            states$trend <- states$trend + elements$deviations$trend;
            states$sinCoef <- states$sinCoef + elements$deviations$sinCoef;
            states$cosCoef <- states$cosCoef + elements$deviations$cosCoef;
            armaInitial <- elements$deviations$arma;
        }
        states$xreg <- states$xreg + elements$xregDeviations;
        for(component in names(provided$states)){
            known <- !is.na(provided$states[[component]]);
            states[[component]][known] <- provided$states[[component]][known];
        }
        if(!is.null(provided$arma)){
            armaInitial <- provided$arma;
        }
        return(list(yBC=yBC, profile=tbats_profile(states, armaInitial, struct, elements$phi),
                    matWt=tbats_matWt(elements$w, struct, obs, xregSpec$data)));
    }
    fitStates <- function(elements){
        inputs <- fitInputs(elements);
        yBC <- inputs$yBC;
        profile <- inputs$profile;
        matWt <- inputs$matWt;
        # "gradient" solves for the initials of the states by least squares (the
        # model is additive in the transformed space), the regressors staying in B
        fitted <- adam_fitOrGradient(matVt, matWt, elements$matF, elements$vecG, lookup, profile,
                                     yBC, ot, initialType, checked$nIterations, adamCpp,
                                     TRUE, struct$nComponents>struct$nETS+struct$nXreg, FALSE, "A",
                                     if(struct$trendIn) "A" else "N", "N",
                                     struct$nETS, 0, struct$nETS, struct$lagsModelAll,
                                     struct$lagsModelMax, obs, checked$loss, distribution,
                                     elements$shape, checked$h,
                                     any(checked$loss==c("MSEh","TMSE","GTMSE","MSCE","GPL")), "n",
                                     struct$nComponents-struct$nETS-struct$nXreg,
                                     struct$lagsModelAll, struct$nXreg);
        fitted$matWt <- matWt;
        fitted$yBC <- yBC;
        return(fitted);
    }
    lossValue <- function(B, lossUsed){
        # nloptr drops the names
        names(B) <- names(BList$B);
        elements <- tbats_filler(BFull(B), struct, armaSpec, lambdaSpec, other, initialEstimate,
                                 checked$bounds, adamCpp, xregEstimate);
        if(elements$penalty>0){
            return(elements$penalty);
        }
        fitted <- fitStates(elements);
        errors <- fitted$errors[otLogical];
        if(any(lossUsed==c("likelihood","MSE","MAE","HAM","custom"))){
            value <- switch(lossUsed,
                            "likelihood"=-tbats_logLik(errors, distribution, elements$shape, obsNonzero) -
                                (elements$lambda-1)*logY,
                            "MSE"=sum(errors^2)/obsNonzero,
                            "MAE"=sum(abs(errors))/obsNonzero,
                            "HAM"=sum(sqrt(abs(errors)))/obsNonzero,
                            # On the observed values with demand, in the transformed space
                            "custom"=checked$lossFunction(actual=fitted$yBC[otLogical],
                                                          fitted=fitted$fitted[otLogical], B=B));
        }
        else{
            hor <- checked$h;
            adamErrors <- adamCpp$ferrors(fitted$states, fitted$matWt,
                                          elements$matF, lookup, fitted$profileInitial, hor, fitted$yBC)$errors;
            # The windows with all their targets observed
            adamErrors <- adamErrors[adam_completeWindows(!is.na(y), hor),,drop=FALSE];
            nWindows <- nrow(adamErrors);
            value <- switch(lossUsed,
                            "MSEh"=sum(adamErrors[,hor]^2)/nWindows,
                            "TMSE"=sum(colSums(adamErrors^2)/nWindows),
                            "GTMSE"=sum(log(colSums(adamErrors^2)/nWindows)),
                            "MSCE"=sum(rowSums(adamErrors)^2)/nWindows,
                            "GPL"=log(det(t(adamErrors) %*% adamErrors/nWindows)));
        }
        if(!is.finite(value)){
            value <- 1E+300;
        }
        return(value);
    }
    CF <- function(B){
        return(lossValue(B, checked$loss));
    }
    # Whether the parameters satisfy the bounds, for the draws of reapply()
    inBounds <- function(B){
        names(B) <- names(BList$B);
        return(tbats_filler(BFull(B), struct, armaSpec, lambdaSpec, other, initialEstimate,
                            checked$bounds, adamCpp, xregEstimate)$penalty==0);
    }

    # The refits at the draws of the parameters (rows), for reapply(): the C++ refitter
    # over all of them at once, each with its own matrices, initial profile and, with
    # lambda, data. The solved initials of "gradient" are kept as their deviations from
    # the global model (profileOffset, set after the estimation)
    profileOffset <- 0;
    refitBackcast <- backcast;
    refitter <- function(draws){
        nsim <- nrow(draws);
        nComponents <- struct$nComponents;
        draws <- matrix(draws, nsim, dimnames=list(NULL, names(BList$B)));
        refits <- lapply(1:nsim, function(i){
            elements <- tbats_filler(BFull(draws[i,]), struct, armaSpec, lambdaSpec, other, initialEstimate,
                                     checked$bounds, adamCpp, xregEstimate);
            return(c(elements, fitInputs(elements)));
        });
        # The missing values are skipped, as in the fit
        yBC <- sapply(refits, function(refit) refit$yBC);
        missing <- is.na(yBC[,1]) | is.na(ot);
        yBC[missing,] <- 0;
        otRefit <- replace(ot, missing, 0);
        arrF <- array(sapply(refits, function(refit) refit$matF), c(nComponents, nComponents, nsim));
        arrWt <- array(sapply(refits, function(refit) refit$matWt), c(obs, nComponents, nsim));
        matG <- sapply(refits, function(refit) refit$vecG[,1]);
        profiles <- array(sapply(refits, function(refit) refit$profile + profileOffset),
                          c(nComponents, struct$lagsModelMax, nsim));
        refitted <- adamCpp$reapply(yBC, matrix(otRefit), array(0, c(dim(matVt), nsim)), arrWt,
                                    arrF, matrix(matG, nComponents), lookup, profiles, refitBackcast);
        return(list(fitted=refitted$fitted, states=refitted$states, profile=refitted$profile,
                    matF=arrF, matWt=arrWt, vecG=matrix(matG, nComponents),
                    lambda=sapply(refits, function(refit) refit$lambda)));
    }

    #### Estimation ####
    B <- BList$B;
    # The warm start from the parameters of a related fit, kept if it beats the default
    if(!is.null(BStart)){
        common <- intersect(names(B), names(BStart));
        BWarm <- B;
        BWarm[common] <- BStart[common];
        if(CF(BWarm)<CF(B)){
            B <- BWarm;
        }
    }
    # Backcast first, then optimise all from its parameters and initials, unless B is provided
    if(initialType=="two-stage" && is.null(checked$B)){
        backcastFit <- tbats_fit(y, trendType, harmonicTable, armaSpec, lambdaSpec, distribution,
                                 "complete", checked, xregSpec, BStart);
        common <- intersect(names(B), names(backcastFit$B));
        B[common] <- backcastFit$B[common];
        B <- tbats_deviations(B, backcastFit, struct, qrX, y[otLogical], lambdaSpec)[names(BList$B)];
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
        B <- tbats_ontoBounds(B, lb, ub);
        res <- suppressWarnings(nloptr(B, CF, lb=lb, ub=ub, opts=opts));
        # Stuck on a penalty: restart from no smoothing, unless B was provided
        if(is.null(checked$B) && (is.infinite(res$objective) || res$objective>=1E+100)){
            B[grepl("^(alpha|beta|gamma)", names(B))] <- 0;
            res <- suppressWarnings(nloptr(tbats_ontoBounds(B, lb, ub), CF, lb=lb, ub=ub, opts=opts));
        }
        if(checked$print_level>0){
            print(res);
        }
        B[] <- res$solution;
    }
    lossFinal <- CF(B);

    #### The final fit ####
    elements <- tbats_filler(BFull(B), struct, armaSpec, lambdaSpec, other, initialEstimate,
                             checked$bounds, adamCpp, xregEstimate);
    fitted <- fitStates(elements);
    states <- fitted$states;
    if(headLength$geometry>struct$lagsModelMax){
        states <- states[, -c(1:(headLength$geometry-struct$lagsModelMax)), drop=FALSE];
    }
    rownames(states) <- struct$componentNames;
    initialRead <- tbats_initialsRead(states, struct);
    # The profile of the identified initials (the backcast ones with backcasting),
    # where the simulations start
    # The solved initials of "gradient" for the refits; a failed solve falls back to
    # the backcast fit
    if(initialType=="gradient"){
        profileOffset <- fitted$profileInitial - fitInputs(elements)$profile;
        refitBackcast <- all(profileOffset==0);
    }
    fitted$profileInitial <- tbats_profile(initialRead$states, initialRead$arma, struct, elements$phi);

    # The identified initials are counted whether they are optimised, backcast or solved
    nInitials <- 1 + struct$trendIn + 2*struct$nHarmonics + struct$armaLagMax;
    # A reused model provides all of them but the scale, as in adam()
    nParamModel <- length(B) + nInitials*initialsProfiled + struct$nXreg*!xregEstimate;
    reused <- checked$modelDo=="use";
    nParamEstimated <- 1 + nParamModel*!reused;
    # The likelihood of the occurrence model is added, as its parameters are
    logLikValue <- -lossValue(B, "likelihood") + occurrenceSpec$logLik;

    # The Hessian of the log-likelihood
    FI <- NA;
    if(isTRUE(checked$FI) && length(B)>0){
        FI <- -hessianCpp(function(BNew) -lossValue(BNew, "likelihood"), B, h=checked$stepSize);
        colnames(FI) <- rownames(FI) <- names(B);
    }

    scale <- tbats_scale(fitted$errors[otLogical], distribution, elements$shape, obsNonzero);
    forecastBC <- NULL;
    if(checked$h>0){
        forecastBC <- adamCpp$forecast(tbats_matWt(elements$w, struct, checked$h, xregSpec$future),
                                       elements$matF,
                                       lookup[, headLength$geometry+obs+1:checked$h, drop=FALSE],
                                       fitted$profile, checked$h)$forecast;
    }

    return(list(B=B, BFull=BFull(B), nParamProvided=provided$number+nParamModel*reused,
                initialProvided=provided$initial,
                res=res, lossValue=lossFinal,
                logLik=structure(logLikValue, nobs=sum(!is.na(y)), df=nParamEstimated+occurrenceSpec$nParam*!reused,
                                 class="logLik"),
                nParamEstimated=nParamEstimated, nInitials=nInitials*initialsProfiled,
                struct=struct, armaSpec=armaSpec, elements=elements, fitted=fitted, states=states,
                initialRead=initialRead, scale=scale, forecastBC=forecastBC, FI=FI,
                trendType=trendType, initialType=initialType, distribution=distribution,
                adamCpp=adamCpp, lookup=lookup, headLength=headLength, inBounds=inBounds, refitter=refitter,
                y=y, lambdaSpec=lambdaSpec, checked=checked, xregEstimate=xregEstimate));
}

# The initial deviations from the global model that reproduce the initials of a fit
tbats_deviations <- function(B, backcastFit, struct, qrX, y, lambdaSpec){
    lambda <- backcastFit$elements$lambda;
    states <- tbats_globalStates(tbats_qrCoef(qrX, tbats_boxCox(y, lambda)), struct);
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
    if(struct$nXreg>0){
        B[struct$xreg$names] <- read$states$xreg - states$xreg;
    }
    return(B);
}

#### Scale and likelihood ####
# The scale as in the ADAM monograph: sigma^2 for dnorm, s for the others
tbats_scale <- function(errors, distribution, shape, obs){
    return(adam_scaler(distribution, "A", errors, NULL, obs, shape));
}

# The log-likelihood of the errors in the space of the transformed data
tbats_logLik <- function(errors, distribution, shape, obs){
    return(sum(tbats_logDensities(errors, distribution, shape, tbats_scale(errors, distribution, shape, obs))));
}

# The log-densities of the errors in the space of the transformed data
tbats_logDensities <- function(errors, distribution, shape, scale){
    return(switch(distribution,
                  "dnorm"=dnorm(errors, 0, sqrt(scale), log=TRUE),
                  "dlaplace"=dlaplace(errors, 0, scale, log=TRUE),
                  "ds"=ds(errors, 0, scale, log=TRUE),
                  "dgnorm"=dgnorm(errors, 0, scale, shape, log=TRUE)));
}

# The fitted probabilities of occurrence (ones without an occurrence model)
tbats_pFitted <- function(object){
    occurrence <- object$occurrence;
    if(is.null(occurrence)){
        return(rep(1, nobs(object)));
    }
    return(as.vector(if(is.occurrence(occurrence)) fitted(occurrence) else occurrence$fitted));
}

#### The returned object ####
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
    # With an occurrence model, the probability times the sizes
    occurrenceSpec <- checked[["tbatsOccurrence"]];
    yFitted <- tbats_boxCoxInverse(best$fitted$fitted, lambda) * occurrenceSpec$pFitted;
    # No fitted value without the regressors
    if(struct$nXreg>0){
        yFitted[rowSums(is.na(struct$xreg$data))>0] <- NA;
    }
    yFitted <- makeSeries(yFitted);
    # No errors at the missing values
    errors <- makeSeries(replace(best$fitted$errors, is.na(best$y), NA));
    if(checked$h>0){
        yForecast <- tbats_boxCoxInverse(best$forecastBC, lambda) *
            tbats_pForecast(list(occurrence=occurrenceSpec$model), checked$h);
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
    phiValue <- if(struct$damped) round(best$BFull[["phi"]], 3) else "-";
    modelName <- paste0("TBATS", if(struct$nXreg>0) "X", "(", round(lambda, 3), ", {", sum(armaSpec$arOrders), ",",
                        sum(armaSpec$maOrders), "}, ", phiValue, seasonalPart, ")", if(struct$xregAdapt) "{D}");

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
    if(struct$nXreg>0){
        initialValue$xreg <- setNames(best$initialRead$states$xreg, struct$xreg$names);
    }

    parametersNumber <- matrix(0, 2, 5, dimnames=list(c("Estimated","Provided"),
                                                      c("nParamInternal","nParamXreg","nParamOccurrence",
                                                        "nParamScale","nParamAll")));
    parametersNumber[1,1] <- best$nParamEstimated - 1;
    parametersNumber[1,3] <- occurrenceSpec$nParam*(checked$modelDo=="estimate");
    parametersNumber[1,4] <- 1;
    parametersNumber[1,5] <- sum(parametersNumber[1,1:4]);
    parametersNumber[2,1] <- best$nParamProvided;
    parametersNumber[2,3] <- occurrenceSpec$nParam*(checked$modelDo=="use");
    parametersNumber[2,5] <- sum(parametersNumber[2,1:4]);

    yInSample <- replace(checked$yInSample, is.na(best$y), NA);
    # The holdout keeps its missing values, which the checker filled
    yHoldout <- checked$yHoldout;
    if(!is.null(yHoldout)){
        yHoldout <- replace(yHoldout, checked$yNAValues[-seq_along(best$y)], NA);
    }
    # Keep the ts class of the data, as adam() does
    if(is.ts(yFitted)){
        yInSample <- ts(yInSample, start=start(yFitted), frequency=frequency(yFitted));
        if(!is.null(yHoldout) && is.ts(yForecast)){
            yHoldout <- ts(yHoldout, start=start(yForecast), frequency=frequency(yForecast));
        }
    }
    errormeasures <- NULL;
    if(checked$holdout && checked$h>0){
        errormeasures <- adam_accuracy(yHoldout, yForecast, yInSample);
    }
    # The regressors sit next to the response in the data, as in adam()
    formula <- checked$formula;
    if(struct$nXreg>0){
        responseName <- all.vars(formula)[1];
        yInSample <- cbind(yInSample, struct$xreg$data);
        colnames(yInSample) <- c(responseName, struct$xreg$names);
        if(!is.null(yHoldout)){
            yHoldout <- cbind(yHoldout, struct$xreg$future);
            colnames(yHoldout) <- colnames(yInSample);
        }
        formula <- as.formula(paste0("`", responseName, "`~",
                                     paste0("`", struct$xreg$names, "`", collapse="+")));
    }

    persistence <- best$elements$vecG[,1];
    names(persistence) <- struct$componentNames;
    matF <- best$elements$matF;
    dimnames(matF) <- list(struct$componentNames, struct$componentNames);
    matWt <- best$fitted$matWt;
    dimnames(matWt) <- list(NULL, struct$componentNames);
    orders <- list(ar=armaSpec$arOrders, i=rep(0, length(armaSpec$lags)), ma=armaSpec$maOrders);
    arma <- NULL;
    if(armaSpec$nParam>0){
        arPart <- best$BFull[grepl("^phi[0-9]", names(best$BFull))];
        maPart <- best$BFull[grepl("^theta", names(best$BFull))];
        # Only the parts the model has, as in adam()
        arma <- Filter(length, list(ar=arPart, ma=maPart));
    }
    other <- if(best$distribution=="dgnorm") list(shape=best$elements$shape) else list();

    modelReturned <- structure(list(model=modelName, timeElapsed=Sys.time()-startTime,
                                    call=cl, data=yInSample, holdout=yHoldout,
                                    fitted=yFitted, residuals=errors, forecast=yForecast,
                                    states=t(best$states), accuracy=errormeasures,
                                    profile=best$fitted$profile, profileInitial=best$fitted$profileInitial,
                                    persistence=persistence, phi=if(struct$damped) best$BFull[["phi"]] else 1,
                                    transition=matF, measurement=matWt,
                                    initial=initialValue,
                                    initialType=if(best$initialProvided) "provided" else best$initialType,
                                    orders=orders, arma=arma, armaSpec=armaSpec, armaLags=armaSpec$lags,
                                    lambda=lambda,
                                    harmonics=harmonics, periods=periods, trendType=best$trendType,
                                    nParam=parametersNumber,
                                    formula=formula, xregNames=struct$xreg$names,
                                    regressors=struct$xreg$regressors,
                                    occurrence=occurrenceSpec$model,
                                    loss=checked$loss, lossValue=best$lossValue, lossFunction=checked$lossFunction,
                                    logLik=best$logLik,
                                    ICs=ICs,
                                    distribution=best$distribution, other=other, bounds=checked$bounds,
                                    scale=best$scale, B=best$B, provided=checked[["tbatsProvided"]],
                                    lags=c(1, periods),
                                    lagsAll=struct$lagsModelAll, res=best$res, FI=best$FI,
                                    adamCpp=best$adamCpp, inBounds=best$inBounds, refitter=best$refitter),
                               class=c("adamTBATS","adam","smooth"));
    if(!silent){
        plot(modelReturned, 7);
    }
    return(modelReturned);
}

#### Methods ####
# The model in the space of the Box-Cox transform, as an adam object: the adam
# methods work there, and the tbats ones transform their results back
tbats_boxCoxObject <- function(object){
    lambda <- object$lambda;
    objectBC <- object;
    class(objectBC) <- c("adam","smooth");
    # The sizes: the occurrence is taken into account in the space of the data. They are
    # zero where there is no demand, so that nobs(all=FALSE) and adam_dfScale() count the
    # non-zero observations, and their scale is divided by all the observed values, as
    # adam()'s of an occurrence model, which adam_varianceDebiased() multiplies by T/df.
    # The missing values are NA, neither zeros nor observations
    objectBC$occurrence <- NULL;
    y <- as.numeric(actuals(object));
    otLogical <- tbats_sizes(y, object);
    # The missing values stay missing
    yBC <- replace(tbats_boxCoxSizes(y, lambda, otLogical), is.na(y), NA);
    objectBC$scale <- adam_scaleDebias(object$scale, object$distribution, sum(otLogical), sum(!is.na(y)));
    # The response only: the regressors stay as they are
    objectBC$data[,1] <- yBC;
    objectBC$fitted[] <- yBC - residuals(object);
    objectBC$forecast[] <- tbats_boxCox(object$forecast, lambda);
    if(!is.null(object$holdout)){
        objectBC$holdout[,1] <- suppressWarnings(tbats_boxCox(object$holdout[,1], lambda));
    }
    return(objectBC);
}

# The values the sizes are fitted to: the observed ones, non-zero with an occurrence model
tbats_sizes <- function(y, object){
    return(!is.na(y) & (y!=0 | is.null(object$occurrence)));
}

# The quantiles of the transformed data map onto those of the data, so the
# forecasts and the bounds are transformed back; the point forecast is the median
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

# The in-sample multistep forecast errors of the transformed data, with the C++ core
# of the model: rmultistep.adam would compare its forecasts with the data
#' @export
rmultistep.adamTBATS <- function(object, h=10, ...){
    y <- as.numeric(actuals(object));
    yBC <- tbats_boxCoxSizes(y, object$lambda, tbats_sizes(y, object));
    lookup <- adamProfileCreator(object$lagsAll, max(object$lagsAll), nobs(object))$lookup;
    errors <- object$adamCpp$ferrors(t(object$states), object$measurement, object$transition,
                                     lookup, object$profileInitial, h, matrix(yBC))$errors;
    if(any(class(actuals(object))=="ts")){
        return(ts(errors, start=start(actuals(object)), frequency=frequency(actuals(object))));
    }
    return(zoo(errors, order.by=time(actuals(object))));
}

# The refits at parameters drawn from their distribution. Each draw has its own
# lambda, so its states are in the space of its own transform; the refitted values
# are transformed back. A draw outside the bounds is pulled towards the estimate,
# to the last point of the segment between them that satisfies the bounds: the
# estimate often lies on the boundary of the admissible region, where most draws
# would otherwise be rejected.
#' @export
reapply.adamTBATS <- function(object, nsim=1000, type=c("opg","hessian","bootstrap"),
                          bootstrap=FALSE, heuristics=NULL, ...){
    startTime <- Sys.time();
    type <- covarTypeResolver(type, bootstrap);
    parameters <- coef(object);
    vcovMatrix <- reapply_vcov(object, type, heuristics, nsim, ...);
    draws <- matrix(MASS::mvrnorm(nsim, parameters, vcovMatrix), ncol=length(parameters),
                    dimnames=list(NULL, names(parameters)));
    for(i in 1:nsim){
        draws[i,] <- tbats_pullBack(object, parameters, draws[i,]);
    }
    refits <- object$refitter(draws);
    obs <- nobs(object);
    lagsModelMax <- max(object$lagsAll);
    lambdas <- refits$lambda;
    refitted <- matrix(sapply(1:nsim, function(i) tbats_boxCoxInverse(refits$fitted[,i], lambdas[i])),
                       obs, nsim, dimnames=list(NULL, paste0("nsim",1:nsim))) * tbats_pFitted(object);
    if(any(class(actuals(object))=="ts")){
        refitted <- ts(refitted, start=start(actuals(object)), frequency=frequency(actuals(object)));
    }
    nStates <- dim(refits$states)[2];
    return(structure(list(timeElapsed=Sys.time()-startTime,
                          y=actuals(object), states=refits$states[, nStates-(obs+lagsModelMax)+1:(obs+lagsModelMax),, drop=FALSE],
                          refitted=refitted, fitted=fitted(object), model=object$model,
                          transition=refits$matF, measurement=refits$matWt,
                          persistence=refits$vecG, profile=refits$profile,
                          randomParameters=draws, lambda=lambdas,
                          errors=sapply(1:nsim, function(i){
                              # The sizes: no error where there is no demand, NA where the
                              # value is missing
                              y <- as.numeric(actuals(object));
                              otLogical <- tbats_sizes(y, object);
                              errors <- (tbats_boxCoxSizes(y, lambdas[i], otLogical) - refits$fitted[,i]) * otLogical;
                              return(replace(errors, is.na(y), NA));
                          })),
                     class="reapply"));
}

# The call refitting the model with its structure, starting from its parameters
# without bounds, as coefbootstrap() does for adam()
tbats_refitCall <- function(object){
    armaSpec <- object$armaSpec;
    position <- match(trunc(object$lags), armaSpec$lags);
    orders <- list(ar=ifelse(is.na(position), 0, armaSpec$arOrders[position]),
                   ma=ifelse(is.na(position), 0, armaSpec$maOrders[position]), select=FALSE);
    arguments <- list(y=NULL, lags=object$lags, harmonics=object$harmonics, trend=object$trendType,
                      lambda=if(any(names(object$B)=="lambda")) NULL else object$lambda,
                      orders=orders, distribution=object$distribution,
                      loss=if(object$loss=="custom") object$lossFunction else object$loss,
                      initial=object$initialType, bounds=object$bounds,
                      B=object$B, lb=rep(-Inf, length(object$B)), ub=rep(Inf, length(object$B)),
                      silent=TRUE);
    if(!is.null(object$xregNames)){
        arguments$regressors <- if(object$regressors=="adapt") "adapt" else "use";
    }
    # The occurrence model is refitted on the sample, with its type
    if(!is.null(object$occurrence)){
        if(!is.occurrence(object$occurrence)){
            stop("The refits need an occurrence model, not the provided probabilities.", call.=FALSE);
        }
        arguments$occurrence <- object$occurrence$occurrence;
    }
    if(object$distribution=="dgnorm" && !any(names(object$B)=="shape")){
        arguments$shape <- object$other$shape;
    }
    return(as.call(c(as.name("tbats"), arguments)));
}

# The point of the segment from the estimate to a point that is the furthest from
# the estimate and satisfies the bounds, by bisection: the estimate often lies on the
# boundary of the admissible region
tbats_pullBack <- function(object, parameters, point){
    if(object$inBounds(point)){
        return(point);
    }
    inside <- 0;
    outside <- 1;
    for(iteration in 1:20){
        share <- (inside+outside)/2;
        if(!object$inBounds(parameters + share*(point-parameters))){
            outside <- share;
        }
        else{
            inside <- share;
        }
    }
    return(parameters + inside*(point-parameters));
}

# The bounds of the confidence intervals of the parameters (as deviations from the
# estimates) moved inside the bounds of the model, one parameter at a time
tbats_confintBounds <- function(object, parameters, bounds){
    for(j in seq_along(parameters)){
        for(side in 1:2){
            point <- parameters;
            point[j] <- parameters[j] + bounds[j,side];
            bounds[j,side] <- tbats_pullBack(object, parameters, point)[j] - parameters[j];
        }
    }
    return(bounds);
}

# The forecasts with the uncertainty of the parameters: for each draw of reapply(),
# its point forecasts ("confidence") or its simulated paths with the scale of its
# own residuals ("prediction"), in the space of its own transform and transformed
# back. The point forecast is the skeleton of the model, or the mean / median of the
# paths.
#' @export
reforecast.adamTBATS <- function(object, h=10, newdata=NULL, occurrence=NULL,
                             interval=c("prediction", "confidence", "none"),
                             level=0.95, side=c("both","upper","lower"), cumulative=FALSE,
                             nsim=100, type=c("opg","hessian","bootstrap"),
                             bootstrap=FALSE, heuristics=NULL, point=c("skeleton","mean","median"), ...){
    interval <- match.arg(interval);
    point <- match.arg(point);
    side <- match.arg(side);
    objectRefitted <- reapply(object, nsim=nsim, type=type, bootstrap=bootstrap, heuristics=heuristics, ...);
    obs <- nobs(object);
    lagsModelAll <- object$lagsAll;
    lagsModelMax <- max(lagsModelAll);
    nComponents <- length(lagsModelAll);
    lookup <- adamProfileCreator(lagsModelAll, lagsModelMax, obs+h)$lookup[,-c(1:(obs+lagsModelMax)),drop=FALSE];
    draws <- objectRefitted$randomParameters;
    dfScale <- adam_dfScale(object);
    # The sizes of an occurrence model are the non-zero observed values
    otLogical <- tbats_sizes(as.numeric(actuals(object)), object);
    # The future values of the regressors, as forecast.adam() takes them
    xregRows <- which(colnames(object$states) %in% object$xregNames);
    xregFuture <- if(length(xregRows)>0) adam_xregNewdata(object, h, tbats_newdata(object, newdata));
    paths <- vector("list", nsim);
    for(j in 1:nsim){
        matWt <- matrix(objectRefitted$measurement[1,,j], h, nComponents, byrow=TRUE);
        matWt[, xregRows] <- xregFuture;
        matF <- objectRefitted$transition[,,j];
        profile <- matrix(objectRefitted$profile[,,j], nComponents, lagsModelMax);
        lambda <- objectRefitted$lambda[j];
        if(interval=="prediction"){
            shape <- if(any(colnames(draws)=="shape")) draws[j,"shape"] else object$other$shape;
            errorsSizes <- objectRefitted$errors[otLogical,j];
            scale <- adam_scaleDebias(tbats_scale(errorsSizes, object$distribution, shape, length(errorsSizes)),
                                      object$distribution, length(errorsSizes), dfScale);
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
    # The occurrence drawn with its probabilities; the cumulative values are the sums of
    # the paths in the space of the data
    pForecast <- tbats_pForecast(object, h);
    paths <- tbats_occurrenceDraws(paths, pForecast);
    if(cumulative){
        paths <- matrix(colSums(paths), 1);
    }
    bounds <- tbats_pathsBounds(paths, level, side);
    yLower <- bounds$lower;
    yUpper <- bounds$upper;
    levelLow <- bounds$levelLow;
    levelUp <- bounds$levelUp;
    pointForecast <- forecast(tbats_boxCoxObject(object), h=h, newdata=tbats_newdata(object, newdata),
                              interval="none")$mean;
    yForecast <- pointForecast;
    yForecast[] <- tbats_boxCoxInverse(pointForecast, object$lambda) * pForecast;
    if(cumulative){
        yForecast <- sum(yForecast);
    }
    if(point!="skeleton"){
        yForecast[] <- apply(paths, 1, switch(point, "mean"=mean, "median"=median), na.rm=TRUE);
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
forecast.adamTBATS <- function(object, h=10, newdata=NULL, occurrence=NULL,
                           interval=c("none", "prediction", "confidence", "simulated",
                                      "approximate", "semiparametric", "nonparametric",
                                      "empirical","complete"),
                           level=0.95, side=c("both","upper","lower"), cumulative=FALSE, nsim=NULL,
                           scenarios=FALSE, point=c("skeleton","mean","median"), ...){
    point <- match.arg(point);
    objectBC <- tbats_boxCoxObject(object);
    newdata <- tbats_newdata(object, newdata);
    # The parameter uncertainty comes from reforecast(), which reapplies the model
    # with the lambda of each draw and returns the bounds of the data
    if(h>0 && any(interval[1]==c("confidence","complete"))){
        return(reforecast(object, h=h, newdata=newdata, occurrence=occurrence,
                          interval=switch(interval[1], "confidence"="confidence", "prediction"),
                          level=level, side=match.arg(side), cumulative=cumulative,
                          nsim=if(is.null(nsim)) 100 else nsim, point=point, ...));
    }
    # The probabilities of occurrence: provided, or forecast by the occurrence model
    pForecast <- if(is.null(occurrence)) tbats_pForecast(object, h) else
        rep(as.numeric(occurrence), length.out=max(h, 0));
    intermittent <- any(pForecast<1);
    # The sums of the back-transformed values, with the occurrence: from simulated paths in
    # the space of the data, as adam() does with an occurrence model
    if(cumulative && h>0 && (object$lambda!=1 || intermittent)){
        return(tbats_cumulative(object, objectBC, h, newdata, interval[1], level, match.arg(side),
                                nsim, point, pForecast, ...));
    }
    # The sizes. The median of the transformed data is its skeleton, and transforms back
    # into the median of the data
    result <- forecast(objectBC, h=h, newdata=newdata, interval=interval, level=level, side=side,
                       cumulative=cumulative, nsim=nsim, scenarios=scenarios, ...);
    result <- tbats_boxCoxForecast(result, object);
    if(point=="mean" && object$lambda!=1 && h>0){
        result$mean[] <- tbats_mean(object, objectBC, h, newdata, nsim, ...);
    }
    # The mixture of no demand and the sizes: the skeleton and the mean multiplied by the
    # probability, the median and the bounds its quantiles
    if(intermittent && h>0){
        quantiles <- function(probs, intervalUsed){
            return(tbats_mixtureQuantiles(object, objectBC, h, newdata, intervalUsed, probs,
                                          pForecast, nsim, ...));
        }
        result$mean[] <- if(point=="median") quantiles(0.5, if(interval[1]=="none") "prediction" else
            interval[1]) else result$mean * pForecast;
        if(interval[1]!="none"){
            level[level>1] <- level[level>1]/100;
            side <- match.arg(side);
            result$lower[] <- quantiles(switch(side, "both"=(1-level)/2, "upper"=rep(0, length(level)),
                                               "lower"=1-level), interval[1]);
            result$upper[] <- quantiles(switch(side, "both"=(1+level)/2, "upper"=level,
                                               "lower"=rep(1, length(level))), interval[1]);
        }
    }
    result$point <- point;
    return(result);
}

# The quantiles of the mixture of no demand and the sizes: zero below the probability of no
# demand, otherwise the quantile (q-(1-p))/p of the sizes, by the method of the interval
tbats_mixtureQuantiles <- function(object, objectBC, h, newdata, interval, probs, pForecast, nsim, ...){
    quantiles <- matrix(0, h, length(probs));
    for(p in unique(pForecast)){
        rows <- which(pForecast==p);
        sizeLevels <- (probs-(1-p))/p;
        quantiles[rows, sizeLevels>=1] <- Inf;
        columns <- which(sizeLevels>0 & sizeLevels<1);
        if(length(columns)>0){
            bounds <- forecast(objectBC, h=h, newdata=newdata, interval=interval, level=sizeLevels[columns],
                               side="upper", nsim=nsim, ...)$upper;
            quantiles[rows, columns] <- matrix(tbats_boxCoxInverse(bounds, object$lambda), h)[rows,];
        }
    }
    return(quantiles);
}

# The mean of the forecast distribution of the data: Gauss-Hermite quadrature over the
# normal forecast distribution of the transformed data, with the variance of the
# approximate interval (the closed form for lambda=0); for the other distributions the
# mean of the simulated paths transformed back
#' @importFrom statmod gauss.quad
tbats_mean <- function(object, objectBC, h, newdata, nsim, ...){
    lambda <- object$lambda;
    if(object$distribution=="dnorm"){
        # The bound at the level 2*pnorm(1)-1 is one standard deviation away
        bounds <- forecast(objectBC, h=h, newdata=newdata, interval="approximate",
                           level=2*pnorm(1)-1, side="both");
        mu <- as.vector(bounds$mean);
        sigma <- as.vector(bounds$upper) - mu;
        if(lambda==0){
            return(exp(mu+sigma^2/2));
        }
        quadrature <- gauss.quad(50, kind="hermite");
        return(sapply(seq_len(h), function(i){
            return(sum(quadrature$weights *
                           tbats_boxCoxInverse(mu[i]+sqrt(2)*sigma[i]*quadrature$nodes, lambda)) / sqrt(pi));
        }));
    }
    if(lambda==0 && (object$distribution=="ds" ||
                     (object$distribution=="dgnorm" && object$other$shape<1))){
        warning("With lambda=0 and the ", object$distribution, " distribution, the mean of the ",
                "forecast distribution does not exist: the simulated one is unstable and grows ",
                "with nsim.", call.=FALSE);
    }
    return(rowMeans(tbats_paths(object, objectBC, h, newdata, nsim, rep(1, h), ...)));
}

# The simulated paths of the data (h x nsim): those of the transformed data from the
# forecaster, transformed back, with the occurrence drawn with its probabilities
tbats_paths <- function(object, objectBC, h, newdata, nsim, pForecast, ...){
    paths <- forecast(objectBC, h=h, newdata=newdata, interval="simulated",
                      nsim=if(is.null(nsim)) 10000 else nsim, scenarios=TRUE, ...)$scenarios;
    return(tbats_occurrenceDraws(matrix(tbats_boxCoxInverse(paths, object$lambda), h), pForecast));
}

# The paths with the occurrence drawn with its probabilities (unchanged without occurrence)
tbats_occurrenceDraws <- function(paths, pForecast){
    if(any(pForecast<1)){
        paths[] <- paths * rbinom(length(paths), 1, pForecast);
    }
    return(paths);
}

# The bounds of the paths (a row per horizon): their quantiles at the levels of the side,
# zero and Inf at the levels 0 and 1
tbats_pathsBounds <- function(paths, level, side){
    level[level>1] <- level[level>1]/100;
    levelLow <- switch(side, "both"=(1-level)/2, "upper"=rep(0, length(level)), "lower"=1-level);
    levelUp <- switch(side, "both"=(1+level)/2, "upper"=level, "lower"=rep(1, length(level)));
    quantiles <- function(probs){
        return(matrix(t(apply(paths, 1, quantile, probs=probs, na.rm=TRUE, names=FALSE)),
                      nrow(paths), length(probs)));
    }
    yLower <- quantiles(levelLow);
    yUpper <- quantiles(levelUp);
    yLower[, levelLow==0] <- 0;
    yUpper[, levelUp==1] <- Inf;
    return(list(lower=yLower, upper=yUpper, levelLow=levelLow, levelUp=levelUp));
}

# The cumulative forecast from the paths of the data: the sum of the skeletons (times the
# probabilities), or the mean or median of the sums of the paths, and their quantiles
tbats_cumulative <- function(object, objectBC, h, newdata, interval, level, side, nsim, point,
                             pForecast, ...){
    totals <- matrix(colSums(tbats_paths(object, objectBC, h, newdata, nsim, pForecast, ...)), 1);
    # The structure of the cumulative forecast of the transformed data
    result <- forecast(objectBC, h=h, newdata=newdata,
                       interval=if(interval=="none") "none" else "simulated",
                       level=level, side=side, cumulative=TRUE, nsim=10, ...);
    skeleton <- forecast(objectBC, h=h, newdata=newdata, interval="none")$mean;
    result$mean[] <- switch(point,
                            "skeleton"=sum(tbats_boxCoxInverse(skeleton, object$lambda) * pForecast),
                            "mean"=mean(totals),
                            "median"=median(totals));
    if(interval!="none"){
        bounds <- tbats_pathsBounds(totals, level, side);
        result$lower[] <- bounds$lower;
        result$upper[] <- bounds$upper;
        result$interval <- interval;
    }
    result$model <- object;
    result$point <- point;
    return(result);
}

# The future values of the regressors with their names, when given without them
tbats_newdata <- function(object, newdata){
    if(!is.null(newdata) && !is.null(object$xregNames) && is.null(colnames(newdata)) &&
       NCOL(newdata)==length(object$xregNames)){
        newdata <- matrix(newdata, ncol=length(object$xregNames),
                          dimnames=list(NULL, object$xregNames));
    }
    return(newdata);
}

#' @export
predict.adamTBATS <- function(object, newdata=NULL, interval=c("none", "confidence", "prediction"),
                          level=0.95, side=c("both","upper","lower"), ...){
    result <- predict(tbats_boxCoxObject(object), newdata=tbats_newdata(object, newdata), interval=interval,
                      level=level, side=side, ...);
    return(tbats_boxCoxForecast(result, object));
}

# The log-densities of the data: those of the transformed data and the Jacobian
#' @export
pointLik.adamTBATS <- function(object, log=TRUE, ...){
    y <- as.numeric(actuals(object));
    # The missing values are not in the likelihood: their values stay zero
    observed <- !is.na(y);
    if(is.null(object$occurrence)){
        likValues <- pointLik(tbats_boxCoxObject(object), log=TRUE) + (object$lambda-1)*log(y);
    }
    # The occurrence, and the sizes with the Jacobian where there is a demand
    else{
        otLogical <- tbats_sizes(y, object);
        pFitted <- tbats_pFitted(object);
        likValues <- log(1-pFitted);
        likValues[otLogical] <- log(pFitted[otLogical]) +
            tbats_logDensities(as.numeric(residuals(object))[otLogical], object$distribution,
                               object$other$shape, object$scale) + (object$lambda-1)*log(y[otLogical]);
    }
    likValues[!observed] <- 0;
    if(!log){
        likValues <- exp(likValues);
    }
    return(likValues);
}

# The OPG covariance of the parameters: the scores of the log-densities, refitting
# the model at each perturbed parameter
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
simulate.adamTBATS <- function(object, nsim=1, seed=NULL, obs=nobs(object), ...){
    result <- simulate(tbats_boxCoxObject(object), nsim=nsim, seed=seed, obs=obs, ...);
    # The sizes, and the occurrence drawn with the fitted probabilities
    result$data[] <- tbats_boxCoxInverse(result$data, object$lambda) *
        rbinom(obs*nsim, 1, rep(tbats_pFitted(object), length.out=obs));
    result$model <- object$model;
    return(result);
}
