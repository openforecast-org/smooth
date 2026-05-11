context("Tests for oes() function (om() wrapper)")

set.seed(41)
y <- rpois(100, 0.5)

# 1. Returned object inherits from the om/adam/smooth class chain
testModel <- oes(y, occurrence="f")
test_that("oes() returns an om/adam/smooth object", {
    expect_s3_class(testModel, "om")
    expect_s3_class(testModel, "adam")
    expect_s3_class(testModel, "smooth")
})

# 2. Each occurrence type round-trips through match.arg correctly
test_that("oes() preserves the occurrence type for each link", {
    expect_equal(oes(y, occurrence="f")$occurrence, "fixed")
    expect_equal(oes(y, occurrence="o")$occurrence, "odds-ratio")
    expect_equal(oes(y, occurrence="i")$occurrence, "inverse-odds-ratio")
    expect_equal(oes(y, occurrence="d")$occurrence, "direct")
})

# 3. Model-name format follows the new oETS(...)[X] convention
test_that("oes() renders the model name with oETS prefix and link tag", {
    expect_match(oes(y, occurrence="o", model="MNN")$model, "^oETS\\(MNN\\)\\[O\\]$")
    expect_match(oes(y, occurrence="i", model="ANN")$model, "^oETS\\(ANN\\)\\[I\\]$")
    expect_match(oes(y, occurrence="d", model="AAN")$model, "^oETS\\(AAN\\)\\[D\\]$")
})

# 4. Damped trend "d" survives the wrapper (regression test for paste0 + phiEstimate)
test_that("oes() includes 'd' in the model name when phi is estimated", {
    expect_match(oes(y, occurrence="o", model="MAdN")$model, "MAdN")
    expect_match(oes(y, occurrence="d", model="AAdN")$model, "AAdN")
})

# 5. Persistence vector length tracks Ttype across the wrapper
test_that("oes() persistence has the expected length per ETS variant", {
    expect_length(oes(y, occurrence="o", model="MNN")$persistence, 1)
    expect_length(oes(y, occurrence="o", model="MMN")$persistence, 2)
    expect_length(oes(y, occurrence="o", model="AAN")$persistence, 2)
})

# 6. Holdout / forecast horizon
test_that("oes() honours h and holdout", {
    m <- oes(y, occurrence="o", model="MNN", h=10, holdout=TRUE)
    expect_length(m$forecast, 10)
    expect_false(is.null(m$accuracy))
    expect_false(is.null(m$holdout))
})

# 7. forecast.om dispatch returns the expected adam.forecast structure
test_that("forecast(oes_obj) dispatches via forecast.om and matches forecast.adam fields", {
    m  <- oes(y, occurrence="o", model="MNN", h=12)
    fc <- forecast(m, h=12)
    expect_s3_class(fc, "adam.forecast")
    expect_equal(names(fc),
                 c("mean","lower","upper","model","level","interval",
                   "side","cumulative","h","scenarios"))
    expect_equal(fc$interval, "none")
    expect_equal(fc$level, 0.95)
})

# 8. Reported lossValue matches Bernoulli -logLik recomputed from fitted values
test_that("oes() lossValue matches manually-computed Bernoulli -logLik", {
    for(occ in c("o","i","d")){
        m <- oes(y, occurrence=occ, model="AAN", h=10)
        fp <- as.numeric(fitted(m))
        ot <- as.numeric(actuals(m))
        ll <- -(sum(log(fp[ot==1])) + sum(log(1-fp[ot==0])))
        expect_equal(m$lossValue, ll, tolerance=1e-8,
                     info=paste0("occurrence=", occ))
    }
})

# 9. Combination path (model="CCN") returns omCombined and reweighs nParam
test_that("oes(model='CCN') produces an omCombined with weighted nParam", {
    skip_on_cran()
    m <- oes(y, occurrence="d", model="CCN", h=12)
    expect_s3_class(m, "omCombined")
    expect_true(length(m$models) > 1)
    expect_equal(sum(m$ICw), 1, tolerance=1e-8)
    # Weighted nParam should be a non-integer in non-degenerate cases.
    expect_true(m$nParam[1, "nParamAll"] > 0)
})

# 10. The wrapper attaches its own call (so users see oes(...) not om(...))
test_that("oes() records its own call on the returned model", {
    m <- oes(y, occurrence="o", model="MNN")
    expect_equal(as.character(m$call)[1], "oes")
})

# auto.om tests
# 11. Basic usage: returns an om object with a valid occurrence type
test_that("auto.om() returns an om object with one of the tested occurrence types", {
    m <- auto.om(y, orders=list(ar=0, i=0, ma=0, select=FALSE))
    expect_s3_class(m, "om")
    expect_true(m$occurrence %in% c("fixed","odds-ratio","inverse-odds-ratio","direct"))
})

# 12. Restricted occurrence vector: winner must be from the supplied set
test_that("auto.om() with restricted occurrence tries only those types", {
    m <- auto.om(y, occurrence=c("odds-ratio","direct"),
                 orders=list(ar=0, i=0, ma=0, select=FALSE))
    expect_true(m$occurrence %in% c("odds-ratio","direct"))
})

# 13. Holdout + forecast horizon
test_that("auto.om() honours h and holdout", {
    m <- auto.om(y, occurrence=c("fixed","odds-ratio"),
                 orders=list(ar=0, i=0, ma=0, select=FALSE),
                 h=10, holdout=TRUE)
    expect_false(is.null(m$accuracy))
})

# 14. call slot records auto.om, not om
test_that("auto.om() records its own call on the returned model", {
    m <- auto.om(y, occurrence=c("fixed","odds-ratio"),
                 orders=list(ar=0, i=0, ma=0, select=FALSE))
    expect_equal(as.character(m$call)[1], "auto.om")
})

# 15. om() with occurrence="fixed" must have a valid logLik (regression test for NULL logLik bug)
test_that("om() fixed occurrence populates logLik and AICc", {
    m <- om(y, model="ANN", occurrence="fixed")
    expect_false(is.null(m$logLik))
    expect_true(is.finite(m$logLik))
    expect_true(is.finite(AICc(m)))
})

# 16. auto.om() with default args (ARIMA selection, fixed in candidate set) must not crash
test_that("auto.om() with default orders runs without error", {
    skip_on_cran()
    m <- auto.om(y)
    expect_s3_class(m, "om")
    expect_true(m$occurrence %in% c("fixed","odds-ratio","inverse-odds-ratio","direct"))
})
