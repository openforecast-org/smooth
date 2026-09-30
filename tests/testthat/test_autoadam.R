context("Baseline tests for auto.adam() — pin behavior before refactoring")

xreg <- data.frame(y=AirPassengers,
                   x=factor(temporaldummy(AirPassengers, factors=TRUE)))

# 1. ETS distribution selection
test_that("auto.adam() ETS distribution selection on AirPassengers", {
    skip_on_cran()
    set.seed(42)
    m <- auto.adam(AirPassengers, "ZZZ",
                   distribution=c("dnorm","dlnorm","dgamma"), silent=TRUE)
    expect_equal(m$distribution, "dnorm")
    expect_equal(modelType(m), "MAM")
    expect_equal(AICc(m), 1085.408, tolerance=0.01)
    expect_equal(length(m$persistence), 3)
})

# 2. ARIMA order selection on BJsales
test_that("auto.adam() ARIMA selection (NNN) on BJsales", {
    skip_on_cran()
    set.seed(42)
    m <- auto.adam(BJsales, "NNN",
                   orders=list(ar=c(2,0), i=c(2,0), ma=c(2,0), select=TRUE),
                   distribution="dnorm", silent=TRUE)
    expect_equal(modelType(m), "NNN")
    # Re-pinned after the stationarity / invertibility checks factor by factor:
    # the old checks let ARIMA(2,1,2) reach a non-invertible MA (root 0.994); with
    # the MA kept invertible, ARIMA(1,1,2) with drift wins (AICc 523.568 against
    # 527.259 before), without the nearly cancelling AR / MA pair.
    expect_equal(AICc(m), 523.568, tolerance=0.01)
    expect_equal(m$distribution, "dnorm")
    expect_equal(as.numeric(m$arma[[1]]), 0.830514, tolerance=1e-3)
})

# 3. ETS + ARIMA selection on AirPassengers
test_that("auto.adam() ETS+ARIMA selection on AirPassengers", {
    skip_on_cran()
    set.seed(42)
    m <- auto.adam(AirPassengers, "ZZZ",
                   orders=list(ar=c(2,1), i=c(1,0), ma=c(2,1), select=TRUE),
                   distribution=c("dnorm","dgamma"), lags=c(1,12), silent=TRUE)
    expect_equal(m$distribution, "dnorm")
    expect_equal(modelType(m), "MAM")
    expect_equal(AICc(m), 1085.408, tolerance=0.01)
})

# 4. Regressors = "use"
test_that("auto.adam() with regressors='use' on AirPassengers+xreg", {
    skip_on_cran()
    set.seed(42)
    m <- auto.adam(xreg, "ZZZ",
                   distribution=c("dnorm","dlnorm"), lags=c(1,12),
                   regressors="use", silent=TRUE)
    expect_equal(m$distribution, "dlnorm")
    expect_equal(modelType(m), "ANM")
    expect_equal(AICc(m), 1110.799, tolerance=0.01)
    expect_equal(length(m$persistence), 15)
})

# 5. Regressors = "select"
test_that("auto.adam() with regressors='select' on AirPassengers+xreg", {
    skip_on_cran()
    set.seed(42)
    m <- auto.adam(xreg, "ZZZ",
                   distribution=c("dnorm","dlnorm"), lags=c(1,12),
                   regressors="select", silent=TRUE)
    expect_equal(m$distribution, "dnorm")
    expect_equal(modelType(m), "MAM")
    expect_equal(AICc(m), 1085.408, tolerance=0.01)
})

# 6. Regressors = "adapt"
test_that("auto.adam() with regressors='adapt' on AirPassengers+xreg", {
    skip_on_cran()
    set.seed(42)
    m <- auto.adam(xreg, "ZZZ",
                   distribution=c("dnorm","dlnorm"), lags=c(1,12),
                   regressors="adapt", silent=TRUE)
    # With the ETS smoother for the Hannan-Rissanen values of ETS+ARIMA, an AR(1)
    # on top of ETSX(MMN) lowers AICc from 1090.929 (ETSX(MMN) with dlnorm). With the
    # Hannan-Rissanen values on the residuals of the regression, the dlnorm one reaches
    # 1083.697 and beats the dnorm one (1083.931 before, 1084.049 now)
    expect_equal(m$distribution, "dlnorm")
    expect_equal(modelType(m), "MMN")
    expect_equal(orders(m)$ar, c(1, 0))
    expect_equal(AICc(m), 1083.697, tolerance=0.01)
})

# 7. Outliers = "use"
test_that("auto.adam() with outliers='use' on BJsales", {
    skip_on_cran()
    set.seed(42)
    m <- auto.adam(BJsales, "ZZZ",
                   distribution=c("dnorm","dlaplace"), outliers="use", silent=TRUE)
    expect_equal(m$distribution, "dnorm")
    expect_equal(modelType(m), "AMdN")
    expect_equal(AICc(m), 525.256, tolerance=0.01)
})

# 8. Outliers = "select"
test_that("auto.adam() with outliers='select' on BJsales", {
    skip_on_cran()
    set.seed(42)
    m <- auto.adam(BJsales, "ZZZ",
                   distribution=c("dnorm","dlaplace"), outliers="select", silent=TRUE)
    expect_equal(m$distribution, "dnorm")
    expect_equal(modelType(m), "AMdN")
    expect_equal(AICc(m), 525.592, tolerance=0.01)
})

# 9. Long series: the ARIMA selector must not deparse the data into a name
test_that("auto.msarima() selects orders on a series longer than R's name limit", {
    skip_on_cran()
    set.seed(42)
    y <- ts(100 + cumsum(rnorm(1500)))
    m <- auto.msarima(y, orders=list(ar=1, i=1, ma=1), lags=1,
                      h=10, holdout=TRUE, silent=TRUE)
    # The response name comes from deparse(substitute(data)) inside the fitter;
    # deparsing 1500 observations into it breaks R's 10000-byte limit on names.
    expect_lt(nchar(colnames(m$data)[1]), 100)
    expect_true(grepl("ARIMA", m$model))
})

# The smoother reaches the models of the ARIMA order selection
test_that("auto.msarima() passes the smoother on", {
    y <- log(AirPassengers)
    lossValues <- sapply(c("ma","lowess"), function(s){
        auto.msarima(y, lags=c(1,12), initial="optimal", smoother=s)$lossValue
    })
    expect_equal(unname(lossValues["lowess"]),
                 msarima(y, orders=list(ar=c(0,0),i=c(1,1),ma=c(1,1)), lags=c(1,12),
                         initial="optimal", smoother="lowess")$lossValue)
    expect_false(isTRUE(all.equal(lossValues[1], lossValues[2])))
})
