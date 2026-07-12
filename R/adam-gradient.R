# Internal helper (not exported, not documented for users).
#
# Least-squares initial-state solve for ETS ("gradient" initialisation).
# Given fixed persistence (the current matF / vecG) and a seeded recent profile,
# this solves for the initial state that minimises the in-sample SSE. In a Single
# Source of Error model the residuals are, at fixed persistence, an affine
# function of the initial state for additive models (exact one-shot least
# squares) and a smooth nonlinear function otherwise (Gauss-Newton). It does not
# run the backcasting backward pass at all.
#
# The forward-pass oracle is adamCpp$reapply(backcast=FALSE): each slice of the
# profile cube is one probe, run through the same forward machinery the final fit
# uses, in a single C++ call. Returns the solved recent profile (same shape as
# `profile`), or NULL on failure.
adam_gradientSolve <- function(adamCpp, matWt, matF, vecG, indexLookupTable,
                               profile, yInSample, ot, layout, lagsModelMax){
    y <- as.numeric(yInSample)
    obs <- length(y)
    nComp <- nrow(profile)
    Etype <- layout$Etype
    additive <- layout$additive

    # Free-initial probe map: each entry is a set of (row, cols) in the profile
    # that move together as one free parameter.
    probes <- layout$probes
    nFree <- length(probes)
    if(nFree == 0){ return(NULL) }

    # Residuals of one forward pass for a set of candidate profiles (slices).
    # Returns an obs x nSlices matrix of residuals, or NULL on failure.
    residualsFor <- function(profList){
        nSlices <- length(profList)
        arrProf <- array(0, c(nComp, lagsModelMax, nSlices))
        for(s in 1:nSlices){ arrProf[, , s] <- profList[[s]] }
        arrVt <- array(0, c(nComp, obs + lagsModelMax, nSlices))
        arrWt <- array(matWt, c(dim(matWt), nSlices))
        arrF <- array(matF, c(dim(matF), nSlices))
        matG <- matrix(as.numeric(vecG), nComp, nSlices)
        res <- try(adamCpp$reapply(matrix(y, obs, 1), matrix(as.numeric(ot), obs, 1),
                                   arrVt, arrWt, arrF, matG, indexLookupTable,
                                   arrProf, FALSE), silent = TRUE)
        if(inherits(res, "try-error")){ return(NULL) }
        fitted <- res$fitted
        if(any(!is.finite(fitted))){ return(NULL) }
        E <- switch(Etype, "M" = y / fitted - 1, y - fitted)
        if(any(!is.finite(E))){ return(NULL) }
        return(E)
    }

    # Apply a step in free-parameter coordinates to a profile.
    applyStep <- function(prof, step){
        for(j in 1:nFree){
            p <- probes[[j]]
            prof[p$row, p$cols] <- prof[p$row, p$cols] + step[j]
        }
        return(prof)
    }

    if(additive){
        # Affine: build the design with unit probes, solve once.
        slices <- vector("list", nFree + 1)
        slices[[1]] <- profile
        for(j in 1:nFree){
            pj <- profile; pp <- probes[[j]]
            pj[pp$row, pp$cols] <- pj[pp$row, pp$cols] + 1
            slices[[j + 1]] <- pj
        }
        E <- residualsFor(slices)
        if(is.null(E)){ return(NULL) }
        e0 <- E[, 1]
        designA <- matrix(0, obs, nFree)
        for(j in 1:nFree){ designA[, j] <- e0 - E[, j + 1] }
        # Solve min ||e0 - A d|| in C++ (pivoted QR, rank-deficiency aware) so the
        # result is identical in the Python port that shares the same core.
        d <- as.numeric(olsCpp(designA, e0, 1e-7))
        d[!is.finite(d)] <- 0
        return(applyStep(profile, d))
    }
    else{
        # Nonlinear: Gauss-Newton with finite-difference Jacobian and line search.
        prof <- profile
        E0 <- residualsFor(list(prof))
        if(is.null(E0)){ return(NULL) }
        e0 <- E0[, 1]; fCur <- sum(e0^2)
        for(iter in 1:15){
            slices <- vector("list", nFree)
            hs <- numeric(nFree)
            for(j in 1:nFree){
                pp <- probes[[j]]
                h <- 1e-4 * max(1, abs(prof[pp$row, pp$cols[1]]))
                pj <- prof; pj[pp$row, pp$cols] <- pj[pp$row, pp$cols] + h
                slices[[j]] <- pj; hs[j] <- h
            }
            E <- residualsFor(slices)
            if(is.null(E)){ break }
            Jm <- matrix(0, obs, nFree)
            for(j in 1:nFree){ Jm[, j] <- (E[, j] - e0) / hs[j] }
            # Gauss-Newton step via the C++ solve (parity with Python).
            step <- tryCatch(-as.numeric(olsCpp(Jm, e0, 1e-7)),
                             error = function(e) rep(NA_real_, nFree))
            step[!is.finite(step)] <- 0
            if(sqrt(sum(step^2)) < 1e-8){ break }
            improved <- FALSE
            for(tt in c(1, 0.5, 0.25, 0.125, 0.0625, 0.03125)){
                Et <- residualsFor(list(applyStep(prof, tt * step)))
                if(!is.null(Et) && sum(Et[, 1]^2) < fCur){
                    prof <- applyStep(prof, tt * step); e0 <- Et[, 1]; fCur <- sum(e0^2)
                    improved <- TRUE; break
                }
            }
            if(!improved){ break }
        }
        return(prof)
    }
}

# Internal helper (not exported, not documented for users).
#
# Build the free-initial probe map for gradient initialisation of an ETS model.
# Returns NULL when the model is out of scope (ARIMA / xreg / non-ETS), so the
# caller falls back to backcasting.
adam_gradientLayout <- function(etsModel, arimaModel, xregModel, Etype, Ttype, Stype,
                                componentsNumberETS, componentsNumberETSSeasonal,
                                componentsNumberETSNonSeasonal, lagsModel, lagsModelMax){
    if(!etsModel || arimaModel || xregModel){ return(NULL) }
    probes <- list()
    # level: row 1, all head columns move together
    probes[[length(probes) + 1]] <- list(row = 1, cols = 1:lagsModelMax)
    # trend: row 2
    if(Ttype != "N"){
        probes[[length(probes) + 1]] <- list(row = 2, cols = 1:lagsModelMax)
    }
    # seasonal: each cell of each seasonal row is a free parameter
    if(Stype != "N" && componentsNumberETSSeasonal > 0){
        for(i in 1:componentsNumberETSSeasonal){
            r <- componentsNumberETSNonSeasonal + i
            for(c in 1:lagsModel[r]){
                probes[[length(probes) + 1]] <- list(row = r, cols = c)
            }
        }
    }
    additive <- (Etype == "A") && (Ttype %in% c("N", "A")) && (Stype %in% c("N", "A"))
    return(list(probes = probes, Etype = Etype, additive = additive))
}

# Internal helper (not exported, not documented for users).
#
# Fit dispatcher: runs the gradient initial-state solve when
# initialType=="gradient" and the model is in scope (ETS, no ARIMA/xreg),
# otherwise the ordinary adamCore$fit (with the backcast flag). Gradient always
# fits with backcast=FALSE from the solved initials; it never runs the backward
# pass. Out-of-scope gradient models fall back to backcasting.
adam_fitOrGradient <- function(matVt, matWt, matF, vecG, indexLookupTable, profile,
                               yInSample, ot, initialType, nIterations, adamCpp,
                               etsModel, arimaModel, xregModel, Etype, Ttype, Stype,
                               componentsNumberETS, componentsNumberETSSeasonal,
                               componentsNumberETSNonSeasonal, lagsModel, lagsModelMax){
    if(any(initialType == "gradient")){
        layout <- adam_gradientLayout(etsModel, arimaModel, xregModel, Etype, Ttype, Stype,
                                      componentsNumberETS, componentsNumberETSSeasonal,
                                      componentsNumberETSNonSeasonal, lagsModel, lagsModelMax)
        if(!is.null(layout)){
            solved <- adam_gradientSolve(adamCpp, matWt, matF, vecG, indexLookupTable,
                                         profile, yInSample, ot, layout, lagsModelMax)
            if(!is.null(solved)){
                matVt[, 1:lagsModelMax] <- solved
                # A single forward pass from the solved initials (backcast=FALSE).
                # nIterations must be 1 here: with no backward pass, a second
                # iteration would re-run headFillFwd on the profile already
                # mutated by the first forward pass and diverge.
                return(adamCpp$fit(matVt, matWt, matF, vecG, indexLookupTable, solved,
                                   yInSample, ot, FALSE, 1, "n"))
            }
        }
    }
    return(adamCpp$fit(matVt, matWt, matF, vecG, indexLookupTable, profile,
                       yInSample, ot,
                       any(initialType == c("complete", "backcasting", "gradient")),
                       nIterations, "n"))
}
