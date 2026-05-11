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
    expect_equal(AICc(m), 1053.632, tolerance=0.01)
    expect_equal(length(m$persistence), 10)
})

# 2. ARIMA order selection on BJsales
test_that("auto.adam() ARIMA selection (NNN) on BJsales", {
    skip_on_cran()
    set.seed(42)
    m <- auto.adam(BJsales, "NNN",
                   orders=list(ar=c(2,0), i=c(2,0), ma=c(2,0), select=TRUE),
                   distribution="dnorm", silent=TRUE)
    expect_equal(modelType(m), "NNN")
    expect_equal(AICc(m), 521.6353, tolerance=0.01)
    expect_equal(m$distribution, "dnorm")
    expect_equal(as.numeric(m$arma[[1]]),
                 c(0.26485, 0.49232), tolerance=1e-3)
    expect_equal(as.numeric(m$arma[[2]]),
                 c(-0.05723, -0.32949), tolerance=1e-3)
})

# 3. ETS + ARIMA selection on AirPassengers
test_that("auto.adam() ETS+ARIMA selection on AirPassengers", {
    skip_on_cran()
    set.seed(42)
    m <- auto.adam(AirPassengers, "ZZZ",
                   orders=list(ar=c(2,1), i=c(1,0), ma=c(2,1), select=TRUE),
                   distribution=c("dnorm","dgamma"), lags=c(1,12), silent=TRUE)
    expect_equal(m$distribution, "dgamma")
    expect_equal(modelType(m), "MAM")
    expect_equal(AICc(m), 1053.839, tolerance=0.01)
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
    expect_equal(AICc(m), 1077.515, tolerance=0.01)
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
    expect_equal(AICc(m), 1053.632, tolerance=0.01)
})

# 6. Regressors = "adapt"
test_that("auto.adam() with regressors='adapt' on AirPassengers+xreg", {
    skip_on_cran()
    set.seed(42)
    m <- auto.adam(xreg, "ZZZ",
                   distribution=c("dnorm","dlnorm"), lags=c(1,12),
                   regressors="adapt", silent=TRUE)
    expect_equal(m$distribution, "dnorm")
    expect_equal(modelType(m), "MMN")
    expect_equal(AICc(m), 1081.797, tolerance=0.01)
})

# 7. Outliers = "use"
test_that("auto.adam() with outliers='use' on BJsales", {
    skip_on_cran()
    set.seed(42)
    m <- auto.adam(BJsales, "ZZZ",
                   distribution=c("dnorm","dlaplace"), outliers="use", silent=TRUE)
    expect_equal(m$distribution, "dnorm")
    expect_equal(modelType(m), "AMdN")
    expect_equal(AICc(m), 520.8226, tolerance=0.01)
})

# 8. Outliers = "select"
test_that("auto.adam() with outliers='select' on BJsales", {
    skip_on_cran()
    set.seed(42)
    m <- auto.adam(BJsales, "ZZZ",
                   distribution=c("dnorm","dlaplace"), outliers="select", silent=TRUE)
    expect_equal(m$distribution, "dnorm")
    expect_equal(modelType(m), "AMdN")
    expect_equal(AICc(m), 521.0947, tolerance=0.01)
})
