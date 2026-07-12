#' @keywords internal
#' Gradient initialisation for adam().
#'
#' Solves for the initial state by least squares instead of the backcasting
#' filter heuristic. In a Single Source of Error model the in-sample residuals
#' are, at fixed persistence, an affine function of the initial state (for
#' additive models), so the SSE-optimal initials are a single linear solve —
#' no backward pass, no time reversal, no fixed-point iteration to diverge.
#'
#' Implemented as a wrapper around the existing provided-initial path so the
#' core estimation code is left untouched:
#'   1. fit with initial="backcasting" to obtain the persistence / selected model;
#'   2. probe the forward pass at unit perturbations of each free initial to
#'      build the design, then solve;
#'   3. refit with the solved initials (initial=<list>).
#'
#' Phase 1 supports additive ETS models (no ARIMA, no xreg, single seasonality).
#' Anything else falls back to backcasting with a message, so behaviour never
#' silently changes for unsupported specifications.
#'
#' @param cl The original adam() call (with initial="gradient").
#' @param envir The environment in which to evaluate cl (the caller's frame).
adam_gradient <- function(cl, envir){
    # Step 1: backcasting fit -> persistence and (if selected) the model
    cl0 <- cl
    cl0$initial <- "backcasting"
    m0 <- eval(cl0, envir)

    # Determine free-initial structure from the fitted object
    trendy <- !is.null(m0$initial$trend)
    seasonal <- !is.null(m0$initial$seasonal)
    seasVec <- NULL
    if(seasonal){
        seasVec <- if(is.list(m0$initial$seasonal)) m0$initial$seasonal[[1]] else m0$initial$seasonal
    }
    mSeason <- length(seasVec)

    # Phase 1 support gate: additive ETS, no ARIMA, no xreg, at most one seasonality
    modelStr <- modelType(m0)
    Etype <- substr(modelStr, 1, 1)
    hasARIMA <- !is.null(m0$arma) && length(unlist(m0$arma)) > 0 &&
        any(unlist(m0$arma) != 0)
    hasXreg <- !is.null(m0$initial$xreg)
    multiSeasonal <- is.list(m0$initial$seasonal) && length(m0$initial$seasonal) > 1
    supported <- (Etype == "A") && !grepl("M", modelStr) &&
        !hasARIMA && !hasXreg && !multiSeasonal

    if(!supported){
        warning(paste0("initial=\"gradient\" currently supports additive ETS models ",
                       "(no ARIMA/xreg, single seasonality). Using backcasting for this model."),
                call.=FALSE)
        return(m0)
    }

    # Map between the free-initial vector and the initial=list() form
    v0ToList <- function(v0){
        lst <- list(level = v0[1])
        idx <- 2
        if(trendy){ lst$trend <- v0[idx]; idx <- idx + 1 }
        if(seasonal){ lst$seasonal <- v0[idx:(idx + mSeason - 1)] }
        return(lst)
    }
    listToV0 <- function(init){
        return(c(init$level,
                 if(trendy){ init$trend } else { NULL },
                 if(seasonal){ if(is.list(init$seasonal)) init$seasonal[[1]] else init$seasonal } else { NULL }))
    }

    persUsed <- m0$persistence
    phiUsed <- m0$phi

    # Oracle: in-sample residuals of one forward pass at the given initials,
    # with persistence / model / phi held fixed. Reuses the provided-initial path.
    oracle <- function(v0){
        clo <- cl
        clo$initial <- v0ToList(v0)
        clo$model <- modelStr
        clo$persistence <- persUsed
        clo$phi <- phiUsed
        res <- try(as.numeric(residuals(eval(clo, envir))), silent = TRUE)
        if(inherits(res, "try-error")){ return(NULL) }
        return(res)
    }

    # Step 2: build the design by unit probes, then solve (additive => exact)
    b0 <- listToV0(m0$initial)
    e0 <- oracle(b0)
    if(is.null(e0)){
        warning("initial=\"gradient\" solve failed; using backcasting.", call.=FALSE)
        return(m0)
    }
    k <- length(b0)
    designA <- matrix(0, length(e0), k)
    for(j in 1:k){
        bj <- b0
        bj[j] <- bj[j] + 1
        ej <- oracle(bj)
        if(is.null(ej)){
            warning("initial=\"gradient\" solve failed; using backcasting.", call.=FALSE)
            return(m0)
        }
        designA[, j] <- e0 - ej
    }
    # min || e0 - A d ||, pivoted QR handles the seasonal sum-zero rank deficiency
    qrA <- qr(designA)
    d <- qr.coef(qrA, e0)
    d[is.na(d)] <- 0
    v0Hat <- b0 + d

    # Step 3: final fit with the solved initials
    clf <- cl
    clf$initial <- v0ToList(v0Hat)
    clf$model <- modelStr
    clf$persistence <- persUsed
    clf$phi <- phiUsed
    mFinal <- try(eval(clf, envir), silent = TRUE)
    if(inherits(mFinal, "try-error")){
        warning("initial=\"gradient\" final fit failed; using backcasting.", call.=FALSE)
        return(m0)
    }
    mFinal$initialType <- "gradient"
    return(mFinal)
}
