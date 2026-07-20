context("Tests for omg() function")

set.seed(41)
y <- rpois(100, 0.5)
testModel  <- omg(y)
testModelH <- omg(y, h=10, holdout=TRUE)

# 1. Class inheritance
test_that("omg() returns an omg/om/smooth object", {
    expect_s3_class(testModel, "omg")
    expect_s3_class(testModel, "om")
    expect_s3_class(testModel, "smooth")
})

# 2. is.omg() predicate
test_that("is.omg() identifies omg objects correctly", {
    expect_true(is.omg(testModel))
    expect_false(is.omg(om(y, occurrence="odds-ratio")))
    expect_false(is.omg(list()))
})

# 3. Fixed slots
test_that("omg() occurrence slot is 'general'", {
    expect_equal(testModel$occurrence, "general")
})

test_that("omg() lags slot is populated", {
    expect_false(is.null(testModel$lags))
    expect_true(length(testModel$lags) >= 1)
})

test_that("omg() call slot records omg, not om", {
    expect_equal(as.character(testModel$call)[1], "omg")
})

test_that("omg() timeElapsed is populated", {
    expect_false(is.null(testModel$timeElapsed))
})

# 4. Sub-model structure
test_that("omg() modelA and modelB are om objects", {
    expect_s3_class(testModel$modelA, "om")
    expect_s3_class(testModel$modelB, "om")
})

test_that("omg() modelA uses odds-ratio, modelB uses inverse-odds-ratio", {
    expect_equal(testModel$modelA$occurrence, "odds-ratio")
    expect_equal(testModel$modelB$occurrence, "inverse-odds-ratio")
})

test_that("omg() respects modelA and modelB arguments", {
    m <- omg(y, modelA="ANN", modelB="MNN")
    expect_match(m$modelA$model, "ANN")
    expect_match(m$modelB$model, "MNN")
})

# 5. Fitted values
test_that("omg() fitted values are in (0, 1)", {
    fp <- as.numeric(testModel$fitted)
    expect_true(all(fp > 0 & fp < 1))
})

test_that("omg() fitted is the coupled probability, consistent with logLik", {
    # The reported fitted is the coupled probability from the recursion the
    # optimiser minimised (not the standalone per-side refit), so its Bernoulli
    # log-likelihood equals the reported logLik exactly. modelA/modelB stay
    # standalone fits for parameter diagnostics.
    pf <- as.numeric(testModel$fitted)
    ot <- as.numeric(y != 0)
    expect_true(all(pf > 0 & pf < 1))
    ll <- sum(ot * log(pf) + (1 - ot) * log(1 - pf))
    expect_equal(ll, as.numeric(logLik(testModel)), tolerance = 1e-9)
})

# 6. Holdout and forecast
test_that("omg() h and holdout are respected", {
    expect_length(testModelH$forecast, 10)
    expect_false(is.null(testModelH$accuracy))
    expect_false(is.null(testModelH$holdout))
})

test_that("omg() forecast values are in (0, 1)", {
    fc <- as.numeric(testModelH$forecast)
    expect_true(all(fc > 0 & fc < 1))
})

test_that("omg() internal forecast matches forecast.omg output", {
    fc <- forecast(testModelH, h=10)
    expect_equal(as.numeric(testModelH$forecast), as.numeric(fc$mean), tolerance=1e-10)
})

# h=0 + silent=FALSE used to crash with "argument 'forecast' is missing,
# with no default" because omg() left $forecast=NULL and plot.smooth would
# then drop the slot from its do.call to graphmaker. Mirror om()'s NA
# placeholder convention so plot() can be called on an h=0 fit.
test_that("omg() populates $forecast with NA when h=0, and plot works", {
    pdf(NULL); on.exit(dev.off(), add=TRUE)
    m <- om(rpois(100, 1), occurrence="general", silent=FALSE)
    expect_false(is.null(m$forecast))
    expect_equal(length(m$forecast), 1)
    expect_true(is.na(as.numeric(m$forecast)))
    # plot() on h=0 should not error
    expect_silent(plot(m, 7))
})

# 7. forecast.omg dispatch
test_that("forecast(omg_obj) returns adam.forecast with expected fields", {
    m  <- omg(y, h=12)
    fc <- forecast(m, h=12)
    expect_s3_class(fc, "adam.forecast")
    expect_equal(names(fc),
                 c("mean","lower","upper","model","level","interval",
                   "side","cumulative","h","scenarios"))
    expect_equal(fc$interval, "none")
    expect_equal(fc$level, 0.95)
})

test_that("forecast.omg values equal omgLinkFunction of forecast.adam sub-model outputs", {
    m   <- omg(y, h=10)
    fc  <- forecast(m, h=10)
    fcA <- forecast.adam(m$modelA, h=10, interval="none", level=0.95, side="both", cumulative=FALSE)
    fcB <- forecast.adam(m$modelB, h=10, interval="none", level=0.95, side="both", cumulative=FALSE)
    fA  <- as.vector(fcA$mean)
    fB  <- as.vector(fcB$mean)
    expected <- fA / (fA + fB)
    expect_equal(as.numeric(fc$mean), expected, tolerance=1e-10)
})

# 8. actuals.omg / actuals.omg_submodel
test_that("actuals(omg_obj) returns the binary occurrence indicator", {
    expect_equal(as.numeric(actuals(testModel)), (y != 0) * 1)
})

test_that("actuals(omg_obj$modelA) reconstructs the latent value", {
    # Sub-models carry class 'omg_submodel'; actuals.omg_submodel returns
    # fitted + residuals (OM stores residuals additively regardless of
    # error type, so the same formula recovers the latent value for both
    # 'A' and 'M' sub-models).
    fA <- fitted(testModel$modelA)
    rA <- residuals(testModel$modelA)
    expect_equal(as.numeric(actuals(testModel$modelA)), as.numeric(fA + rA))
})

# 9. print / summary
test_that("print.omg outputs expected header and model lines", {
    out <- capture.output(print(testModel))
    expect_true(any(grepl("General occurrence model", out)))
    expect_true(any(grepl("Model A", out)))
    expect_true(any(grepl("Model B", out)))
})

test_that("summary.omg runs without error", {
    expect_silent(summary(testModel))
})

# 10. ETS model variants for A and B
test_that("omg() with different ETS model combinations runs without error", {
    expect_s3_class(omg(y, modelA="ANN", modelB="ANN"), "omg")
    expect_s3_class(omg(y, modelA="AAN", modelB="MNN"), "omg")
    expect_s3_class(omg(y, modelA="MAdN", modelB="ANN"), "omg")
    expect_s3_class(omg(y, modelA="AAdN", modelB="MNN"), "omg")
})

# 11. Damped trend: model name must contain the "d" character
test_that("omg() with damped trend models includes 'd' in sub-model names", {
    m <- omg(y, modelA="MAdN", modelB="AAdN")
    expect_match(m$modelA$model, "MAdN")
    expect_match(m$modelB$model, "AAdN")
})

# 12. Shared parameters: initial, loss, ic
test_that("omg() with initial='optimal' runs without error", {
    expect_s3_class(omg(y, initial="optimal"), "omg")
})

test_that("omg() with loss='MSE' runs without error", {
    expect_s3_class(omg(y, loss="MSE"), "omg")
})

test_that("omg() with ic='BIC' runs without error", {
    expect_s3_class(omg(y, ic="BIC"), "omg")
})

# 13. Asymmetric A/B pair: sub-model fitted values should differ
test_that("omg() with asymmetric model pair produces different sub-model fitted values", {
    m <- omg(y, modelA="MNN", modelB="AAN")
    expect_false(isTRUE(all.equal(as.numeric(m$modelA$fitted),
                                  as.numeric(m$modelB$fitted))))
})

# 14. etsA / etsB variant
test_that("omg() with etsA='adam' runs without error", {
    expect_s3_class(omg(y, etsA="adam"), "omg")
})

test_that("omg() with etsB='adam' runs without error", {
    expect_s3_class(omg(y, etsB="adam"), "omg")
})

# 15. Exogenous variables via formulaA
test_that("omg() with formulaA and exogenous data runs without error", {
    set.seed(1)
    xreg <- data.frame(y=y, x=rnorm(100))
    m <- omg(xreg, formulaA=y~x)
    expect_s3_class(m, "omg")
})

# 16. Custom persistence fixed via persistenceA
test_that("omg() with fixed persistenceA is respected in modelA", {
    m <- omg(y, modelA="MNN", persistenceA=0.2)
    expect_s3_class(m, "omg")
    expect_equal(as.numeric(m$modelA$persistence), 0.2)
})

# ---------------------------------------------------------------------
# vcov.omg — phase 2 of vcov mechanism mirroring vcov.adam / vcov.om
# ---------------------------------------------------------------------

test_that("vcov.omg returns a finite joint covariance matrix", {
    set.seed(12);
    y <- rbinom(150, 1, 0.5);
    m <- suppressWarnings(omg(y, modelA="ANN", modelB="ANN", silent=TRUE));
    V <- suppressWarnings(vcov(m));
    expect_true(is.matrix(V));
    expect_equal(nrow(V), length(m$modelA$B) + length(m$modelB$B));
    expect_equal(ncol(V), nrow(V));
    expect_true(all(is.finite(V)));        # no Inf / NaN
    expect_true(all(abs(V) < 1e+50));      # no 1e+100 singular fallback
    expect_equal(V, t(V), tolerance=1e-3); # symmetric
    expect_true(all(diag(V) >= 0));        # non-negative diagonal
})

test_that("vcov.omg dimension matches the joint B vector", {
    set.seed(7);
    y <- rbinom(150, 1, 0.5);
    m <- suppressWarnings(omg(y, modelA="ANN", modelB="ANN", silent=TRUE));
    V <- suppressWarnings(vcov(m));
    nJoint <- length(m$modelA$B) + length(m$modelB$B);
    expect_equal(dim(V), c(nJoint, nJoint));
    expect_true(is.matrix(V));
})

test_that("vcov.omg respects the heuristics argument", {
    set.seed(11);
    y <- rbinom(60, 1, 0.4);
    m <- suppressWarnings(omg(y, modelA="ANN", modelB="ANN", silent=TRUE));
    V <- vcov(m, heuristics=0.1);
    expect_true(is.matrix(V));
    expect_equal(nrow(V), length(m$modelA$B) + length(m$modelB$B));
})

# ---------------------------------------------------------------------
# confint.omg / summary.omg — joint CI table and per-model summary blocks
# ---------------------------------------------------------------------

test_that("confint.omg returns a finite joint CI table", {
    set.seed(12);
    y <- rbinom(150, 1, 0.5);
    m <- suppressWarnings(omg(y, modelA="ANN", modelB="ANN", silent=TRUE));
    ci <- suppressWarnings(confint(m));
    expect_true(is.matrix(ci));
    expect_equal(nrow(ci), length(m$modelA$B) + length(m$modelB$B));
    expect_equal(ncol(ci), 3);                       # S.E., lower, upper
    expect_true(all(is.finite(ci)));
    expect_true(any(grepl("^A:", rownames(ci))));
    expect_true(any(grepl("^B:", rownames(ci))));
})

test_that("summary.omg builds two coefficient sub-tables", {
    set.seed(12);
    y <- rbinom(150, 1, 0.5);
    m <- suppressWarnings(omg(y, modelA="ANN", modelB="ANN", silent=TRUE));
    s <- suppressWarnings(summary(m));
    expect_s3_class(s, "summary.omg");
    expect_true(is.matrix(s$coefficientsA));
    expect_true(is.matrix(s$coefficientsB));
    expect_equal(nrow(s$coefficientsA), length(m$modelA$B));
    expect_equal(nrow(s$coefficientsB), length(m$modelB$B));
    expect_equal(colnames(s$coefficientsA),
                 c("Estimate","Std. Error","Lower 2.5%","Upper 97.5%"));
    expect_true(all(is.finite(s$coefficientsA)));
    expect_true(all(is.finite(s$coefficientsB)));
})

test_that("print.summary.omg prints per-model blocks", {
    set.seed(12);
    y <- rbinom(150, 1, 0.5);
    m <- suppressWarnings(omg(y, modelA="ANN", modelB="ANN", silent=TRUE));
    s <- suppressWarnings(summary(m));
    expect_output(print(s), "Model A:");
    expect_output(print(s), "Model B:");
})

# ---------------------------------------------------------------------
# coefbootstrap.omg — joint bootstrap covariance over c(modelA$B, modelB$B)
# ---------------------------------------------------------------------

test_that("coefbootstrap.omg returns a joint bootstrap object", {
    set.seed(41);
    x <- sim.oes("MNN", 120, frequency=12, occurrence="general",
                 persistence=0.01, initial=2, initialB=1);
    x <- sim.es("MNN", 120, frequency=12, probability=x$probability, persistence=0.1);
    m <- suppressWarnings(omg(x$data, modelA="ANN", modelB="ANN", silent=TRUE));
    nJoint <- length(m$modelA$B) + length(m$modelB$B);
    bs <- suppressWarnings(coefbootstrap(m, nsim=20));
    expect_s3_class(bs, "bootstrap");
    expect_equal(nrow(bs$coefficients), 20);
    expect_equal(ncol(bs$coefficients), nJoint);
    expect_equal(dim(bs$vcov), c(nJoint, nJoint));
    expect_true(all(is.finite(bs$vcov)));
})

# ---------------------------------------------------------------------
# vcov / confint / summary with bootstrap=TRUE for omg
# ---------------------------------------------------------------------

test_that("vcov/confint/summary accept bootstrap=TRUE for omg", {
    set.seed(41);
    x <- sim.oes("MNN", 120, frequency=12, occurrence="general",
                 persistence=0.01, initial=2, initialB=1);
    x <- sim.es("MNN", 120, frequency=12, probability=x$probability, persistence=0.1);
    m <- suppressWarnings(omg(x$data, modelA="ANN", modelB="ANN", silent=TRUE));
    nJoint <- length(m$modelA$B) + length(m$modelB$B);

    set.seed(1); V <- suppressWarnings(vcov(m, bootstrap=TRUE, nsim=20));
    expect_equal(dim(V), c(nJoint, nJoint));
    expect_true(all(is.finite(V)));

    set.seed(1); ci <- suppressWarnings(confint(m, bootstrap=TRUE, nsim=20));
    expect_equal(nrow(ci), nJoint);
    expect_equal(ncol(ci), 3);
    expect_true(all(is.finite(ci)));

    set.seed(1); s <- suppressWarnings(summary(m, bootstrap=TRUE, nsim=20));
    expect_s3_class(s, "summary.omg");
    expect_true(is.matrix(s$coefficientsA));
    expect_true(is.matrix(s$coefficientsB));
})

# ---------------------------------------------------------------------
# Loss menu — single-step losses, LASSO / RIDGE with lambda, callable
# (the joint omfitGeneral step still runs first; the loss decides the
#  scalar handed to nloptr).
# ---------------------------------------------------------------------

test_that("omg() honours all single-step loss strings", {
    set.seed(31); y <- rbinom(150, 1, 0.4)
    for(L in c("likelihood", "MSE", "MAE", "HAM")){
        m <- omg(y, modelA="ANN", modelB="ANN", loss=L)
        expect_equal(m$loss, L)
        expect_true(is.finite(m$lossValue))
    }
})

test_that("omg() runs LASSO and RIDGE with explicit lambda", {
    set.seed(31); y <- rbinom(150, 1, 0.4)
    for(L in c("LASSO", "RIDGE")){
        m <- omg(y, modelA="ANN", modelB="ANN", loss=L, lambda=0.3)
        expect_equal(m$loss, L)
        expect_true(is.finite(m$lossValue))
    }
})

test_that("omg() accepts a callable for custom loss", {
    set.seed(31); y <- rbinom(150, 1, 0.4)
    my_loss <- function(actual, fitted, B) sum(abs(actual - fitted)^3)
    m <- omg(y, modelA="ANN", modelB="ANN", loss=my_loss)
    expect_equal(m$loss, "custom")
    expect_true(is.function(m$lossFunction))
    expect_true(is.finite(m$lossValue))
})
