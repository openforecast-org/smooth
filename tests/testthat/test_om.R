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
