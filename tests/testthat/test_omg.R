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

test_that("omg() fitted values equal pA / (pA + pB)", {
    pA       <- as.numeric(testModel$modelA$fitted)
    pB       <- as.numeric(testModel$modelB$fitted)
    expected <- pA / (pA + pB)
    expect_equal(as.numeric(testModel$fitted), expected, tolerance=1e-10)
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

# 8. actuals.omg
test_that("actuals(omg_obj) matches actuals from modelA", {
    expect_equal(actuals(testModel), actuals(testModel$modelA))
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
