context("Tests for ces() function");

# Basic CES selection
testModel <- auto.ces(AirPassengers, silent=TRUE);
test_that("Test CES selection on BJsales", {
    expect_match(smoothType(testModel), "CES");
})

# Reuse previous CES
test_that("Test on AirPassengers, predefined CES", {
    expect_equal(ces(AirPassengers, model=testModel, silent=TRUE)$loss, testModel$loss);
})

# Test trace cost function for CES
testModel <- ces(AirPassengers, seasonality="f", h=18, holdout=TRUE, silent=TRUE)
test_that("Test AICc of CES based on MSTFE on AirPassengers", {
    expect_equal(as.numeric(logLik(testModel)), as.numeric(testModel$logLik));
})

# Test how different passed values are accepted by CES
test_that("Test provided a and b of CES on AirPassengers", {
    expect_equal(ces(AirPassengers, seasonality="f", a=testModel$parameters$a, silent=TRUE)$parameters$a,
                 testModel$parameters$a);
    expect_equal(ces(AirPassengers, seasonality="f", b=testModel$parameters$b, silent=TRUE)$parameters$b,
                 testModel$parameters$b);
})

# The OPG covariance must work for every seasonality: "partial" carries a real
# b named "beta" (not the beta_0/beta_1 pair), which used to be missed by the
# smoothing-coefficient count and threw "subscript out of bounds".
test_that("vcov(type='opg') works for every CES seasonality", {
    for (s in c("none", "simple", "partial", "full")) {
        m <- ces(AirPassengers, seasonality = s, initial = "optimal", silent = TRUE)
        v <- vcov(m, type = "opg")
        expect_equal(nrow(v), length(coef(m)))
        expect_false(any(is.nan(diag(v))))
    }
})

# The provided b must match the seasonality it belongs to: "partial" carries one
# real coefficient, "full" a complex pair, "none"/"simple" none at all. A
# wrong-shaped b used to reach the cost function and surface only as a 1e+300
# penalty masquerading as a fit.
test_that("provided b is validated against the seasonality", {
    expect_error(ces(AirPassengers, seasonality="full", b=0.3, silent=TRUE),
                 "complex second smoothing parameter")
    expect_error(ces(AirPassengers, seasonality="partial",
                     b=complex(real=0.3, imaginary=0.1), silent=TRUE),
                 "real second smoothing parameter")
    for (s in c("none", "simple")) {
        expect_warning(m <- ces(AirPassengers, seasonality=s, b=0.3, silent=TRUE),
                       "no second smoothing parameter")
        expect_null(m$parameters$b)
        expect_equal(as.numeric(logLik(m)),
                     as.numeric(logLik(ces(AirPassengers, seasonality=s, silent=TRUE))))
    }
    mFull <- ces(AirPassengers, seasonality="full", silent=TRUE)
    expect_equal(ces(AirPassengers, seasonality="full", b=mFull$parameters$b,
                     silent=TRUE)$parameters$b, mFull$parameters$b)
    expect_equal(as.numeric(ces(AirPassengers, seasonality="partial", b=0.3,
                                silent=TRUE)$parameters$b), 0.3)
})

# Test selection of exogenous with CES
test_that("Use exogenous variables for CESX on BJsales", {
    skip_on_cran()
    testModel <- ces(BJsales, h=18, holdout=TRUE, silent=TRUE, regressors="use", xreg=BJsales.lead)
    expect_equal(length(testModel$initial$xreg),1);
})
