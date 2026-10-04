context("Tests for gum() function")

# Basic GUM selection
testModel <- gum(BJsales, orders=c(2,1),lags=c(1,4), silent=TRUE, initial="optimal")
test_that("Test if GUM worked on BJsales", {
    expect_equal(testModel$model, "GUM(2[1],1[4])")
})

# Reuse previous GUM
test_that("Reuse previous GUM on BJsales", {
    expect_equal(gum(BJsales, model=testModel, silent=TRUE)$lossValue, testModel$lossValue)
})

# Test some crazy order of GUM
test_that("Test if crazy order GUM was estimated on BJsales", {
    skip_on_cran()
    testModel <- gum(BJsales, orders=c(1,1,1), lags=c(1,3,5), h=18, holdout=TRUE, initial="o", silent=TRUE)
    expect_equal(testModel$model, "GUM(1[1],1[3],1[5])")
})

# Test how different passed values are accepted by GUM
test_that("Test initials, measurement, transition and persistence of GUM on AirPassengers", {
    skip_on_cran()
    testModel <- gum(AirPassengers, orders=c(1,1,1), lags=c(1,3,5),
                     h=18, holdout=TRUE, initial="o", silent=TRUE)
    expect_equal(gum(AirPassengers, orders=c(1,1,1), lags=c(1,3,5),
                     initial=testModel$initial, h=18, holdout=TRUE, silent=TRUE)$initial, testModel$initial)
    expect_equal(gum(AirPassengers, orders=c(1,1,1), lags=c(1,3,5),
                     measurement=testModel$measurement, h=18, holdout=TRUE, silent=TRUE)$measurement, testModel$measurement)
    expect_equal(gum(AirPassengers, orders=c(1,1,1), lags=c(1,3,5),
                     transition=testModel$transition, h=18, holdout=TRUE, silent=TRUE)$transition, testModel$transition)
    expect_equal(gum(AirPassengers, orders=c(1,1,1), lags=c(1,3,5),
                     persistence=testModel$persistence, h=18, holdout=TRUE, silent=TRUE)$persistence, testModel$persistence)
})

# Test selection of exogenous with GUM
test_that("Select exogenous variables for GUMX on BJsales", {
    skip_on_cran()
    xregData <- cbind(y=BJsales, x=BJsales.lead)
    testModel <- gum(xregData, h=18, holdout=TRUE, silent=TRUE)
    expect_match(errorType(testModel),"A")
})

# Use automatic GUM
test_that("Use automatic GUM on BJsales", {
    skip_on_cran()
    expect_equal(auto.gum(BJsales, silent=TRUE)$loss, "likelihood")
})

test_that("gum() takes the missing values for gaps", {
    y <- AirPassengers;
    y[c(10, 50, 51, 90)] <- NA;
    testModel <- suppressWarnings(gum(y, orders=c(1,1), lags=c(1,12)));
    expect_equal(attr(logLik(testModel), "nobs"), sum(!is.na(y)));
    expect_equal(as.numeric(logLik(testModel)), sum(pointLik(testModel)));
    expect_true(all(is.na(residuals(testModel)[is.na(y)])));
    expect_true(all(is.finite(forecast(testModel, h=12, interval="prediction")$upper)));
})
