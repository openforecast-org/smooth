# Internal helper (not exported, not documented for users).
#
# Map the estimation loss + distribution pair onto the C++ gradientSolve loss
# code and its parameters (codes documented in src/headers/adamGradient.h).
# This table must stay mirror-identical to adam_gradient_loss_code() in the
# Python build. Returns list(code, params), or NULL for custom loss functions
# (which cannot cross into C++), making the caller fall back to backcasting.
# Losses without a matching rho (LASSO/RIDGE, the absolute multistep variants,
# the remaining distributions) keep the SSE default "S", which is exact for
# likelihood+dnorm and the previous behaviour otherwise. The multiplicative
# likelihood codes ("l", "g", "i") are mapped for Etype=="M" only: with an
# additive error those likelihoods depend on e/yhat, which is not separable in
# the residuals.
adam_gradientLossCode <- function(loss, distribution, Etype, other, horizon,
                                  multisteps){
    if(loss == "custom"){ return(NULL) }
    # Resolve distribution="default" the same way estimator()/preparator() do
    if(distribution == "default"){
        distribution <- switch(loss,
                               "likelihood"=switch(Etype, "A"="dnorm", "M"="dgamma"),
                               "MAEh"=, "MACE"=, "MAE"="dlaplace",
                               "HAMh"=, "CHAM"=, "HAM"="ds",
                               "MSEh"=, "MSCE"=, "MSE"=, "GPL"=, "dnorm");
    }
    if(multisteps){
        code <- switch(loss,
                       "MSEh"="h", "TMSE"="T", "GTMSE"="t", "MSCE"="C", "GPL"="P",
                       "S");
        if(code != "S" && is.numeric(horizon) && horizon >= 1){
            return(list(code=code, params=as.numeric(horizon)));
        }
        return(list(code="S", params=0));
    }
    code <- switch(loss,
                   "MAE"="A",
                   "HAM"="H",
                   "likelihood"=switch(distribution,
                                       "dlaplace"="A",
                                       "ds"="H",
                                       "dgnorm"="G",
                                       "dlnorm"=switch(Etype, "M"="l", "S"),
                                       "dgamma"=switch(Etype, "M"="g", "S"),
                                       "dinvgauss"=switch(Etype, "M"="i", "S"),
                                       "S"),
                   "S");
    params <- 0;
    if(code == "G"){
        if(is.null(other) || !is.finite(other)){ code <- "S" }
        else{ params <- as.numeric(other) }
    }
    return(list(code=code, params=params));
}

# Internal helper (not exported, not documented for users).
#
# Initial-state solve for ETS ("gradient" initialisation). Given fixed
# persistence (the current matF / vecG) and a seeded recent profile, this
# solves for the initial state that minimises the estimation loss (lossCode
# from adam_gradientLossCode). It does not run the backcasting backward pass
# at all.
#
# All the numerical work happens in C++ (adamCpp$gradientSolve, shared with the
# Python build for exact parity): for additive ETS the residuals are affine in
# the initial state, so the design is propagated analytically alongside a single
# forward pass and solved by pivoted-QR least squares (with IRLS sweeps or the
# multistep designs for the non-SSE losses); otherwise loss-aware Gauss-Newton
# with an analytic Jacobian and a Levenberg-Marquardt fallback is used. Returns
# the solved recent profile (same shape as `profile`), or NULL on failure.
adam_gradientSolve <- function(adamCpp, matWt, matF, vecG, indexLookupTable,
                               profile, yInSample, ot, probeBasis, lagsModelMax,
                               obsInSample, lossCode){
    solved <- adamCpp$gradientSolve(yInSample, ot, matWt, matF, vecG,
                                    indexLookupTable, profile, probeBasis, 15, TRUE,
                                    lossCode$code, lossCode$params)
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
# initialType=="gradient" and the model is in scope (ETS, no ARIMA/xreg, a
# loss that maps onto a C++ loss code), otherwise the ordinary adamCore$fit
# (with the backcast flag). Gradient always fits with backcast=FALSE from the
# solved initials; it never runs the backward pass. Out-of-scope gradient
# models (including custom loss functions) fall back to backcasting.
adam_fitOrGradient <- function(matVt, matWt, matF, vecG, indexLookupTable, profile,
                               yInSample, ot, initialType, nIterations, adamCpp,
                               etsModel, arimaModel, xregModel, Etype, Ttype, Stype,
                               componentsNumberETS, componentsNumberETSSeasonal,
                               componentsNumberETSNonSeasonal, lagsModel, lagsModelMax,
                               obsInSample, loss, distribution, other, horizon,
                               multisteps){
    if(any(initialType == "gradient")){
        lossCode <- adam_gradientLossCode(loss, distribution, Etype, other,
                                          horizon, multisteps)
        probeBasis <- adam_gradientProbeBasis(etsModel, arimaModel, xregModel,
                                              Ttype, Stype,
                                              componentsNumberETSSeasonal,
                                              componentsNumberETSNonSeasonal,
                                              nrow(profile), lagsModel, lagsModelMax)
        if(!is.null(probeBasis) && !is.null(lossCode)){
            solved <- adam_gradientSolve(adamCpp, matWt, matF, vecG, indexLookupTable,
                                         profile, yInSample, ot, probeBasis,
                                         lagsModelMax, obsInSample, lossCode)
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
