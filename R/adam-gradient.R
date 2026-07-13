# Internal helper (not exported, not documented for users).
#
# Least-squares initial-state solve for ETS ("gradient" initialisation).
# Given fixed persistence (the current matF / vecG) and a seeded recent profile,
# this solves for the initial state that minimises the in-sample SSE. It does not
# run the backcasting backward pass at all.
#
# All the numerical work happens in C++ (adamCpp$gradientSolve, shared with the
# Python build for exact parity): for additive ETS the residuals are affine in
# the initial state, so the design is propagated analytically alongside a single
# forward pass and solved by one pivoted-QR least squares; otherwise
# finite-difference Gauss-Newton with a line search is used, warm-started from
# the previous call's solution. Returns the solved recent profile (same shape as
# `profile`), or NULL on failure.
adam_gradientSolve <- function(adamCpp, matWt, matF, vecG, indexLookupTable,
                               profile, yInSample, ot, probeBasis, lagsModelMax,
                               obsInSample){
    solved <- adamCpp$gradientSolve(yInSample, ot, matWt, matF, vecG,
                                    indexLookupTable, profile, probeBasis, 15, TRUE)
    if(length(solved) == 0){ return(NULL) }
    return(solved)
}

# Internal helper (not exported, not documented for users).
#
# Build the probe basis for gradient initialisation of an ETS model: a
# (nComponents*lagsModelMax x nFree) 0/1 matrix whose column j marks the profile
# cells (in column-major order, matching the C++ lookup indices) that move
# together as one free initial parameter. Level and trend span all head columns;
# each seasonal cell is its own parameter. Returns NULL when the model is out of
# scope (ARIMA / xreg / non-ETS), so the caller falls back to backcasting.
adam_gradientProbeBasis <- function(etsModel, arimaModel, xregModel, Ttype, Stype,
                                    componentsNumberETSSeasonal,
                                    componentsNumberETSNonSeasonal,
                                    nComponents, lagsModel, lagsModelMax){
    if(!etsModel || arimaModel || xregModel){ return(NULL) }
    # Column-major cell index of profile[row, col] is row + (col-1)*nComponents
    cellsOf <- function(row, cols){ row + (cols - 1) * nComponents }
    probes <- list(cellsOf(1, 1:lagsModelMax))
    if(Ttype != "N"){
        probes[[length(probes) + 1]] <- cellsOf(2, 1:lagsModelMax)
    }
    if(Stype != "N" && componentsNumberETSSeasonal > 0){
        for(i in 1:componentsNumberETSSeasonal){
            r <- componentsNumberETSNonSeasonal + i
            for(j in 1:lagsModel[r]){
                probes[[length(probes) + 1]] <- cellsOf(r, j)
            }
        }
    }
    probeBasis <- matrix(0, nComponents * lagsModelMax, length(probes))
    for(j in seq_along(probes)){
        probeBasis[probes[[j]], j] <- 1
    }
    return(probeBasis)
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
                               componentsNumberETSNonSeasonal, lagsModel, lagsModelMax,
                               obsInSample){
    if(any(initialType == "gradient")){
        probeBasis <- adam_gradientProbeBasis(etsModel, arimaModel, xregModel,
                                              Ttype, Stype,
                                              componentsNumberETSSeasonal,
                                              componentsNumberETSNonSeasonal,
                                              nrow(profile), lagsModel, lagsModelMax)
        if(!is.null(probeBasis)){
            solved <- adam_gradientSolve(adamCpp, matWt, matF, vecG, indexLookupTable,
                                         profile, yInSample, ot, probeBasis,
                                         lagsModelMax, obsInSample)
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
