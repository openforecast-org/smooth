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
    # Re-pinned after the structural-df change: ARIMA(0,2,2) is now selected
    # (initials count towards df, penalising larger AR/MA orders).
    expect_equal(AICc(m), 527.457, tolerance=0.01)
    expect_equal(m$distribution, "dnorm")
    expect_equal(as.numeric(m$arma[[1]]),
                 c(-0.73604, -0.03876), tolerance=1e-3)
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
    expect_equal(m$distribution, "dlnorm")
    expect_equal(modelType(m), "MMN")
    expect_equal(AICc(m), 1090.929, tolerance=0.01)
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
