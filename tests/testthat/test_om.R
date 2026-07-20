context("Tests for om() function")

# Reproducible intermittent demand series and a data.frame with regressors,
# reused across the tests below.
set.seed(41)
nObs <- 200
yIntermittent <- rpois(nObs, 0.3)
yIntermittentTS <- ts(yIntermittent, frequency=12)
dfIntermittent <- data.frame(y=yIntermittent,
                             x1=rnorm(nObs),
                             x2=rnorm(nObs),
                             x3=rnorm(nObs))


#### Group A: occurrence types and ETS variants ####

# 1. Fixed occurrence uses ETS(ANN) with persistence=0
test_that("Fixed occurrence produces constant probability model", {
    testModel <- om(yIntermittent, occurrence="f")
    expect_s3_class(testModel, "om")
    expect_s3_class(testModel, "adam")
    expect_equal(testModel$model, "oETS(ANN)[F]")
    expect_equal(testModel$occurrence, "fixed")
    expect_false(is.null(testModel$states))
    expect_equal(as.numeric(testModel$persistence), 0)
    expect_equal(testModel$distribution, "plogis")
    expect_true(is.na(testModel$scale))
})

# 2. Fixed occurrence with holdout populates accuracy and forecast
test_that("Fixed occurrence with holdout populates accuracy", {
    testModel <- om(yIntermittent, occurrence="fixed", h=10, holdout=TRUE)
    expect_equal(testModel$model, "oETS(ANN)[F]")
    expect_length(testModel$forecast, 10)
    expect_false(is.null(testModel$accuracy))
    expect_false(is.null(testModel$holdout))
})

# 3. Odds-ratio with simple ETS(MNN) -> full path
test_that("Odds-ratio with ETS(MNN)", {
    testModel <- om(yIntermittent, occurrence="o", model="MNN")
    expect_s3_class(testModel, "om")
    expect_match(testModel$model, "^oETS")
    expect_equal(testModel$occurrence, "odds-ratio")
    expect_false(is.null(testModel$states))
    expect_true(length(testModel$persistence) >= 1)
    expect_true(all(testModel$fitted >= 0 & testModel$fitted <= 1))
})

# 4. Inverse-odds-ratio with additive trend ETS(AAN)
test_that("Inverse-odds-ratio with ETS(AAN)", {
    testModel <- om(yIntermittent, occurrence="i", model="AAN")
    expect_match(testModel$model, "^oETS")
    expect_equal(testModel$occurrence, "inverse-odds-ratio")
    expect_true(length(testModel$persistence) >= 2)
})

# 5. Direct link with ETS(MNN)
test_that("Direct link with ETS(MNN)", {
    testModel <- om(yIntermittent, occurrence="d", model="MNN")
    expect_equal(testModel$occurrence, "direct")
    expect_match(testModel$model, "^oETS")
    expect_true(all(testModel$fitted >= 0 & testModel$fitted <= 1))
})

# 6. Holdout on full path
test_that("Holdout on full path produces forecast and accuracy", {
    testModel <- om(yIntermittent, occurrence="o", model="MNN", h=10, holdout=TRUE)
    expect_length(testModel$forecast, 10)
    expect_false(is.null(testModel$accuracy))
    expect_false(is.null(testModel$holdout))
})

# 7. Multiple seasonalities
test_that("Odds-ratio with multiple seasonalities", {
    skip_on_cran()
    testModel <- om(yIntermittentTS, occurrence="o", model="MNM", lags=c(1,12))
    expect_match(testModel$model, "^oETS")
    expect_true(length(testModel$persistence) >= 1)
    expect_false(is.null(testModel$states))
})

# 8. Automatic ETS model selection
test_that("Automatic ETS selection (model=ZZN)", {
    skip_on_cran()
    testModel <- om(yIntermittent, occurrence="o", model="ZZN")
    expect_match(testModel$model, "^oETS")
    expect_s3_class(testModel, "om")
})

# 9. Damped trend with phi provided
test_that("Damped trend with phi provided", {
    skip_on_cran()
    testModel <- om(yIntermittent, occurrence="o", model="AAdN", phi=0.95)
    expect_match(testModel$model, "^oETS")
    expect_equal(testModel$phi, 0.95)
})


#### Group B: pure ARIMA ####

# 10. Pure AR(1)
test_that("Pure ARIMA(1,0,0)", {
    testModel <- om(yIntermittent, occurrence="o", model="NNN",
                    orders=list(ar=1, i=0, ma=0))
    expect_match(testModel$model, "^oARIMA")
    expect_equal(testModel$orders$ar, 1)
    expect_equal(testModel$orders$i, 0)
    expect_equal(testModel$orders$ma, 0)
    expect_false(is.null(testModel$arma))
})

# 11. ARIMA(0,1,1)
test_that("Pure ARIMA(0,1,1)", {
    testModel <- om(yIntermittent, occurrence="o", model="NNN",
                    orders=list(ar=0, i=1, ma=1))
    expect_match(testModel$model, "^oARIMA")
    expect_equal(testModel$orders$i, 1)
    expect_equal(testModel$orders$ma, 1)
})

# 12. Seasonal ARIMA
test_that("Seasonal ARIMA(1,0,0)(1,0,0)[12]", {
    skip_on_cran()
    testModel <- om(yIntermittentTS, occurrence="o", model="NNN",
                    lags=c(1,12),
                    orders=list(ar=c(1,1), i=c(0,0), ma=c(0,0)))
    expect_match(testModel$model, "^oSARIMA|^oARIMA")
    expect_equal(testModel$orders$ar, c(1,1))
})

# 13. ARIMA order selection
test_that("ARIMA order selection", {
    skip_on_cran()
    # AR-structured binary series so that ARIMA order selection finds an improvement
    set.seed(42)
    yAR <- as.numeric(arima.sim(list(ar=0.9), n=200) * 2 > 0)
    testModel <- om(yAR, occurrence="o", model="NNN",
                    orders=list(ar=2, i=0, ma=2, select=TRUE))
    expect_match(testModel$model, "^oARIMA")
    expect_s3_class(testModel, "om")
})


#### Group C: explanatory variables (xreg / formula) ####

# 14. Regressors used as-is
test_that("Regressors via formula, regressors='use'", {
    testModel <- om(dfIntermittent, occurrence="o", model="MNN",
                    formula=y~x1+x2, regressors="use")
    expect_match(testModel$model, "^oETSX")
    expect_false(is.null(testModel$formula))
    expect_false(is.null(testModel$regressors))
})

# 15. Regressors selected by IC
test_that("Regressors via formula, regressors='select'", {
    skip_on_cran()
    testModel <- om(dfIntermittent, occurrence="o", model="MNN",
                    formula=y~x1+x2+x3, regressors="select")
    expect_match(testModel$model, "^oETS")
    expect_s3_class(testModel, "om")
})

# 16. Regressors with adaptive coefficients
test_that("Regressors via formula, regressors='adapt'", {
    skip_on_cran()
    testModel <- om(dfIntermittent, occurrence="o", model="MNN",
                    formula=y~x1+x2, regressors="adapt")
    expect_match(testModel$model, "\\{D\\}")
    expect_s3_class(testModel, "om")
})


#### Group D: combinations of ETS, ARIMA, and regression ####

# 17. ETS(MNN) + AR(1)
test_that("ETS(MNN) + ARIMA(1,0,0)", {
    skip_on_cran()
    testModel <- om(yIntermittent, occurrence="o", model="MNN",
                    orders=list(ar=1, i=0, ma=0))
    expect_match(testModel$model, "^oETS\\(MNN\\)\\+ARIMA")
    expect_true(length(testModel$persistence) >= 1)
    expect_equal(testModel$orders$ar, 1)
})

# 18. ETS + ARIMA + regression
test_that("ETS(MNN) + ARIMA(1,0,1) + regression", {
    skip_on_cran()
    testModel <- om(dfIntermittent, occurrence="o", model="MNN",
                    orders=list(ar=1, i=0, ma=1),
                    formula=y~x1+x2)
    expect_match(testModel$model, "^oETSX")
    expect_match(testModel$model, "ARIMA")
    expect_equal(testModel$orders$ar, 1)
    expect_equal(testModel$orders$ma, 1)
})

# 19. ETS(AAN) + seasonal ARIMA + regression
test_that("ETS(AAN) + seasonal ARIMA + regression", {
    skip_on_cran()
    dfIntermittentSeasonal <- dfIntermittent
    dfIntermittentSeasonal$y <- ts(dfIntermittent$y, frequency=12)
    testModel <- om(dfIntermittentSeasonal, occurrence="o", model="AAN",
                    lags=c(1,12),
                    orders=list(ar=c(0,1), i=c(0,0), ma=c(0,0)),
                    formula=y~x1)
    expect_s3_class(testModel, "om")
    expect_match(testModel$model, "^oETSX")
    expect_true(length(testModel$persistence) >= 1)
})

# 20. Alternative loss on combined ETS + ARIMA
test_that("loss='MSE' on ETS(AAN) + ARIMA(1,0,0)", {
    skip_on_cran()
    testModel <- om(yIntermittent, occurrence="o", model="AAN",
                    orders=list(ar=1, i=0, ma=0), loss="MSE")
    expect_equal(testModel$loss, "MSE")
    expect_match(testModel$model, "^oETS")
    expect_equal(testModel$orders$ar, 1)
})


#### Group E: combined forecasts and Fisher Information ####

# 21. Combined forecast (model="CCN")
test_that("Combined forecast with model='CCN'", {
    skip_on_cran()
    testModel <- om(yIntermittent, occurrence="o", model="CCN")
    expect_s3_class(testModel, "omCombined")
    expect_s3_class(testModel, "om")
    expect_false(is.null(testModel$models))
    expect_false(is.null(testModel$ICw))
    expect_equal(sum(testModel$ICw), 1, tolerance=1e-8)
    expect_true(all(testModel$fitted >= 0 & testModel$fitted <= 1))
})

# 22. Combined forecast with holdout
test_that("Combined forecast with holdout populates accuracy", {
    skip_on_cran()
    testModel <- om(yIntermittent, occurrence="o", model="CCN", h=10, holdout=TRUE)
    expect_length(testModel$forecast, 10)
    expect_false(is.null(testModel$accuracy))
})

# These tests are switched off because this is not yet properly implemented
# # 23. Fisher Information when FI=TRUE
# test_that("FI=TRUE returns a Hessian matrix on the full path", {
#     skip_on_cran()
#     testModel <- om(yIntermittent, occurrence="o", model="AAN",
#                     initial="optimal", FI=TRUE)
#     expect_false(is.null(testModel$FI))
#     expect_true(is.matrix(testModel$FI))
#     expect_equal(nrow(testModel$FI), length(testModel$B))
#     expect_equal(rownames(testModel$FI), names(testModel$B))
# })
#
# # 24. FI default (FALSE) returns NULL
# test_that("FI defaults to NULL", {
#     testModel <- om(yIntermittent, occurrence="o", model="MNN")
#     expect_null(testModel$FI)
# })

# ---------------------------------------------------------------------
# vcov.om — phase 1 of vcov mechanism mirroring vcov.adam
# ---------------------------------------------------------------------

test_that("vcov.om returns a finite, well-formed matrix (ANN + odds-ratio)", {
    set.seed(7);
    y <- rbinom(60, 1, 0.4);
    m <- suppressWarnings(om(y, "ANN", occurrence="odds-ratio"));

    V <- suppressWarnings(vcov(m));
    expect_true(is.matrix(V));
    expect_equal(nrow(V), length(coef(m)));
    expect_equal(ncol(V), length(coef(m)));
    expect_true(all(is.finite(V)));         # no Inf / NaN
    expect_true(all(abs(V) < 1e+50));       # no 1e+100 singular fallback
    expect_equal(V, t(V), tolerance=1e-8);  # symmetric
    expect_true(all(diag(V) >= 0));         # non-negative diagonal
})

test_that("vcov.om works for occurrence='direct'", {
    set.seed(11);
    y <- rbinom(60, 1, 0.3);
    m <- suppressWarnings(om(y, "MNN", occurrence="direct"));
    V <- suppressWarnings(vcov(m));
    expect_true(is.matrix(V));
    expect_true(all(is.finite(V)));
    expect_true(all(abs(V) < 1e+50));
})

test_that("vcov.om works for occurrence='inverse-odds-ratio'", {
    set.seed(13);
    y <- rbinom(60, 1, 0.6);
    m <- suppressWarnings(om(y, "ANN", occurrence="inverse-odds-ratio"));
    V <- suppressWarnings(vcov(m));
    expect_true(is.matrix(V));
    expect_true(all(is.finite(V)));
    expect_true(all(abs(V) < 1e+50));
})

test_that("vcov.om returns a multi-parameter matrix for AAN", {
    set.seed(17);
    y <- rbinom(80, 1, 0.5);
    m <- suppressWarnings(om(y, "AAN", occurrence="odds-ratio"));
    V <- suppressWarnings(vcov(m));
    expect_true(is.matrix(V));
    expect_equal(nrow(V), length(coef(m)));
    expect_equal(ncol(V), length(coef(m)));
    expect_true(all(is.finite(V)));
    expect_true(all(abs(V) < 1e+50));
    expect_equal(V, t(V), tolerance=1e-8);
})

# ---------------------------------------------------------------------
# Regression: om() with occurrence="general" forwards to omg() and must
# work with the default un-resolved `regressors` vector. The earlier bug
# was that om() passed regressors=c("use","select","adapt") to omg()
# without first running match.arg, and omg.R's `match.arg(regressorsB)`
# then failed because its formal default `regressorsB = regressorsA` had
# already been overwritten by a successful match.arg(regressorsA).
# ---------------------------------------------------------------------

test_that("om() with occurrence='general' / 'gen' forwards to omg() without error", {
    set.seed(41);
    x <- sim.oes("MNN", 120, frequency=12, occurrence="general",
                 persistence=0.01, initial=2, initialB=1);
    x <- sim.es("MNN", 120, frequency=12, probability=x$probability,
                persistence=0.1);
    expect_error(
        suppressWarnings(om(x$data, "MMN", occurrence="gen", silent=TRUE, h=12)),
        NA
    );
    expect_error(
        suppressWarnings(om(x$data, "MNN", occurrence="general", silent=TRUE, h=12)),
        NA
    );
})

test_that("om() forwards regressors correctly to omg() across spellings", {
    set.seed(41);
    x <- sim.oes("MNN", 120, frequency=12, occurrence="general",
                 persistence=0.01, initial=2, initialB=1);
    x <- sim.es("MNN", 120, frequency=12, probability=x$probability,
                persistence=0.1);
    for (occ in c("gen", "general")) {
        m <- suppressWarnings(om(x$data, "MNN", occurrence=occ,
                                  silent=TRUE, h=12));
        expect_s3_class(m, "omg");
    }
    # Explicit length-1 regressors — the path that was already working.
    m <- suppressWarnings(om(x$data, "MNN", occurrence="general",
                              regressors="use", silent=TRUE, h=12));
    expect_s3_class(m, "omg");
})

# ---------------------------------------------------------------------
# coefbootstrap.om — bootstrap covariance of the occurrence-model coefs
# ---------------------------------------------------------------------

test_that("coefbootstrap.om returns a bootstrap object", {
    set.seed(41);
    x <- sim.oes("MNN", 120, frequency=12, occurrence="general",
                 persistence=0.01, initial=2, initialB=1);
    x <- sim.es("MNN", 120, frequency=12, probability=x$probability, persistence=0.1);
    m <- suppressWarnings(om(x$data, "MNN", occurrence="odds-ratio", silent=TRUE));
    bs <- suppressWarnings(coefbootstrap(m, nsim=20));
    expect_s3_class(bs, "bootstrap");
    expect_equal(nrow(bs$coefficients), 20);
    expect_equal(ncol(bs$coefficients), length(coef(m)));
    expect_equal(dim(bs$vcov), c(length(coef(m)), length(coef(m))));
    expect_true(all(is.finite(bs$vcov)));
})

# ---------------------------------------------------------------------
# vcov / confint / summary with bootstrap=TRUE for om
# ---------------------------------------------------------------------

test_that("vcov/confint/summary accept bootstrap=TRUE for om", {
    set.seed(41);
    x <- sim.oes("MNN", 120, frequency=12, occurrence="general",
                 persistence=0.01, initial=2, initialB=1);
    x <- sim.es("MNN", 120, frequency=12, probability=x$probability, persistence=0.1);
    m <- suppressWarnings(om(x$data, "MNN", occurrence="odds-ratio", silent=TRUE));

    set.seed(1); V <- suppressWarnings(vcov(m, bootstrap=TRUE, nsim=20));
    expect_equal(dim(V), c(length(coef(m)), length(coef(m))));
    expect_true(all(is.finite(V)));

    set.seed(1); ci <- suppressWarnings(confint(m, bootstrap=TRUE, nsim=20));
    expect_equal(nrow(ci), length(coef(m)));
    expect_true(all(is.finite(ci)));

    set.seed(1); s <- suppressWarnings(summary(m, bootstrap=TRUE, nsim=20));
    expect_s3_class(s, "summary.adam");
})

# ---------------------------------------------------------------------
# Loss menu — single-step losses, LASSO / RIDGE with lambda, callable
# (mirrors adam()'s ``loss`` plumbing; om() previously only honoured
#  "likelihood" and "MSE" and silently routed everything else to MSE).
# ---------------------------------------------------------------------

test_that("om() honours all single-step loss strings", {
    set.seed(31); y <- rbinom(150, 1, 0.4)
    for(L in c("likelihood", "MSE", "MAE", "HAM")){
        m <- om(y, model="MNN", occurrence="odds-ratio", loss=L)
        expect_equal(m$loss, L)
        expect_true(is.finite(m$lossValue))
    }
})

test_that("om() runs LASSO and RIDGE with explicit lambda", {
    set.seed(31); y <- rbinom(150, 1, 0.4)
    for(L in c("LASSO", "RIDGE")){
        m <- om(y, model="MNN", occurrence="odds-ratio", loss=L, lambda=0.3)
        expect_equal(m$loss, L)
        expect_true(is.finite(m$lossValue))
    }
})

test_that("om() accepts a callable for custom loss", {
    set.seed(31); y <- rbinom(150, 1, 0.4)
    my_loss <- function(actual, fitted, B) sum(abs(actual - fitted)^3)
    m <- om(y, model="MNN", occurrence="odds-ratio", loss=my_loss)
    expect_equal(m$loss, "custom")
    expect_true(is.function(m$lossFunction))
    expect_true(is.finite(m$lossValue))
})

test_that("initial=\"gradient\" solves the om initials by profiling the loss", {
    set.seed(5)
    ot <- ts(rbinom(120, 1, plogis(seq(-1, 1.5, length.out=120))))
    for(occ in c("odds-ratio","inverse-odds-ratio","direct")){
        mB <- om(ot, model="MNN", occurrence=occ, initial="backcasting", silent=TRUE)
        mO <- om(ot, model="MNN", occurrence=occ, initial="optimal", silent=TRUE)
        mG <- om(ot, model="MNN", occurrence=occ, initial="gradient", silent=TRUE)
        expect_equal(mG$initialType, "gradient")
        expect_true(is.finite(logLik(mG)))
        # Gradient must at least match backcasting and land near the optimal
        # fit (it profiles the same likelihood over the initials).
        expect_gte(as.numeric(logLik(mG)), as.numeric(logLik(mB)) - 1e-4)
        expect_lt(abs(as.numeric(logLik(mG)) - as.numeric(logLik(mO))),
                  0.01 * abs(as.numeric(logLik(mO))))
    }
})

test_that("om gradient minimises the chosen loss over the initials", {
    set.seed(5)
    otRaw <- rbinom(120, 1, plogis(seq(-1, 1.5, length.out=120)))
    ot <- ts(otRaw)
    mg <- om(ot, model="MNN", occurrence="odds-ratio", initial="gradient",
             persistence=0.1, loss="MSE", silent=TRUE)
    grid <- seq(0.05, 3, by=0.02)
    mseGrid <- min(sapply(grid, function(l){
        m <- suppressWarnings(om(ot, model="MNN", occurrence="odds-ratio",
                                 persistence=0.1, initial=list(level=l),
                                 loss="MSE", silent=TRUE))
        mean((otRaw - fitted(m))^2)
    }))
    # The analytic solve must match or beat the coarse grid optimum.
    expect_lte(mean((otRaw - fitted(mg))^2), mseGrid + 1e-6)
})

test_that("om gradient falls back to backcasting for a custom loss", {
    set.seed(5)
    ot <- ts(rbinom(120, 1, plogis(seq(-1, 1.5, length.out=120))))
    lossCustom <- function(actual, fitted, B){ sum(abs(actual - fitted)) }
    mB <- om(ot, model="MNN", occurrence="odds-ratio", initial="backcasting",
             loss=lossCustom, silent=TRUE)
    mG <- suppressMessages(om(ot, model="MNN", occurrence="odds-ratio",
                              initial="gradient", loss=lossCustom, silent=TRUE))
    expect_equal(mG$lossValue, mB$lossValue, tolerance = 1e-8)
})
