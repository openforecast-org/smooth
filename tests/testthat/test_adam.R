context("Tests for ADAM")

#### Basic ETS stuff ####
# Basic ADAM selection
testModel <- adam(BJsales, "ZZZ")
test_that("ADAM ETS(ZZZ) selection on BJsales", {
    expect_match(errorType(testModel), "A")
})

# Basic ADAM selection on 2568
testModel <- adam(AirPassengers, "ZZZ")
test_that("ADAM ETS(ZZZ) selection on AirPassengers", {
    expect_match(errorType(testModel), "M")
})

# Full ADAM selection
test_that("ADAM ETS(PPP) selection on BJsales", {
    skip_on_cran()
    testModel <- adam(BJsales, "PPP")
    expect_match(modelType(testModel), "AAdN")
})

# ADAM with specified pool
test_that("ADAM selection with a pool on BJsales", {
    skip_on_cran()
    testModel <- adam(BJsales, c("ANN","MNN","AAN","MAN"))
    expect_match(modelType(testModel), "AAN")
})

# ADAM forecasts with simulated interval
test_that("ADAM forecast with simulated interval", {
    skip_on_cran()
    testForecast <- forecast(testModel,h=8,interval="sim",level=c(0.9,0.95))
    expect_equal(ncol(testForecast$lower), 2)
})

# ADAM combination
test_that("ADAM ETS(CCC) on BJsales", {
    skip_on_cran()
    testModel <- adam(BJsales, "CCC")
    expect_match(modelType(testModel), "CCN")
})

# ADAM forecasts with approximated interval
test_that("ADAM forecast with simulated interval", {
    skip_on_cran()
    testForecast <- forecast(testModel,h=8,interval="app",level=c(0.9,0.95),side="upper")
    expect_equal(ncol(testForecast$lower), 2)
})


#### Advanced losses for ADAM ####
# ADAM with GN distribution
test_that("ADAM ETS(MAN) with Generalised Normal on BJsales", {
    skip_on_cran()
    testModel <- adam(BJsales, "MAN", distribution="dgnorm")
    expect_match(testModel$distribution, "dgnorm")
})

# ADAM with MSE
test_that("ADAM ETS(MAN) with MSE on BJsales", {
    skip_on_cran()
    testModel <- adam(BJsales, "MAN", loss="MSE")
    expect_match(testModel$loss, "MSE")
})

# ADAM with MSEh
test_that("ADAM ETS(MAN) with MSE on BJsales", {
    skip_on_cran()
    testModel <- adam(BJsales, "MAN", loss="MSEh",h=12)
    expect_match(testModel$loss, "MSEh")
})

# ADAM with GTMSE
testModel <- adam(BJsales, "MAN", loss="GTMSE",h=12)
test_that("ADAM ETS(MAN) with GTMSE on BJsales", {
    skip_on_cran()
    expect_match(testModel$loss, "GTMSE")
})

# ADAM with GPL
test_that("ADAM ETS(MAN) with GPL on BJsales", {
    skip_on_cran()
    testModel <- adam(BJsales, "MAN", loss="GPL",h=12)
    expect_match(testModel$loss, "GPL")
})

# ADAM with LASSO
test_that("ADAM ETS(MAN) with LASSO on BJsales", {
    skip_on_cran()
    testModel <- adam(BJsales, "MAN", loss="LASSO", lambda=0.5)
    expect_match(testModel$loss, "LASSO")
})

# ADAM with custom loss function
test_that("ADAM ETS(AAN) with custom loss on BJsales", {
    skip_on_cran()
    loss <- function(actual, fitted, B){
        return(sum(abs(actual-fitted)^3))
    }
    testModel <- adam(BJsales, "AAN", loss=loss)
    expect_match(testModel$loss, "custom")
})


#### ETS + occurrence model ####
# Generate intermittent data
set.seed(41)
x <- sim.oes("MNN", 120, frequency=12, occurrence="general", persistence=0.01, initial=2, initialB=1)
x <- sim.es("MNN", 120, frequency=12, probability=x$probability, persistence=0.1)

# iETS(M,N,N)_G
test_that("ADAM iETS(MNN) with general occurrence", {
    skip_on_cran()
    testModel <- adam(x$data, "MNN", occurrence="general")
    expect_match(testModel$occurrence$occurrence, "general")
})

# iETS(M,M,M)_A
test_that("ADAM iETS(MMM) with direct occurrence", {
    skip_on_cran()
    testModel <- adam(x$data, "MMM", occurrence="direct")
    expect_match(errorType(testModel), "M")
})

# iETS(M,M,N)_A
test_that("ADAM iETS(MMN) with auto occurrence", {
    skip_on_cran()
    testModel <- adam(x$data, "MMN", occurrence="auto")
    expect_match(errorType(testModel), "M")
})

# iETS(Z,Z,N)_A
test_that("ADAM iETS(MMN) with auto occurrence", {
    skip_on_cran()
    testModel <- adam(x$data, "ZZN", occurrence="auto")
    expect_true(is.occurrence(testModel$occurrence))
})

# Forecasts from the model
test_that("Froecast from ADAM iETS(ZZZ)", {
    skip_on_cran()
    testForecast <- forecast(testModel, h=18, interval="semi")
    expect_true(is.adam(testForecast$model))
})


#### ETS with several seasonalities ####
# Double seasonality on AirPassengers
test_that("ADAM ETS(YYY) with double seasonality on AirPassengers", {
    skip_on_cran()
    testModel <- adam(AirPassengers, "YYY", lags=c(1,3,12), h=18)
    expect_identical(testModel$lags, c(1,3,12))
})

# Double seasonality on AirPassengers
test_that("ADAM ETS(FFF) + backcasting with double seasonality on AirPassengers", {
    skip_on_cran()
    testModel <- adam(AirPassengers, "FFF", lags=c(1,3,12), h=18, initial="backcasting")
    expect_identical(testModel$lags, c(1,3,12))
})

# Double seasonality on AirPassengers
test_that("ADAM ETS(CCC) with double seasonality on AirPassengers", {
    skip_on_cran()
    testModel <- adam(AirPassengers, "CCC", lags=c(1,3,12), h=18)
    expect_identical(testModel$models[[1]]$lags, c(1,3,12))
})


#### ETSX / Regression + formula ####
# ETSX on AirPassengers
xreg <- data.frame(y=AirPassengers, x=factor(temporaldummy(AirPassengers,factors=TRUE)))
test_that("ADAM ETSX(MMN) on AirPassengers", {
    skip_on_cran()
    testModel <- adam(xreg, "MMN", h=18, holdout=TRUE)
    expect_false(ncol(testModel$data)==1)
})

# ETSX selection on AirPassengers
test_that("ADAM ETSX(ZZZ) + xreg selection on AirPassengers", {
    skip_on_cran()
    testModel <- adam(xreg, "ZZZ", h=18, holdout=TRUE, regressors="select")
    expect_equal(testModel$regressors,"use")
})

# ETSX adaption on AirPassengers
test_that("ADAM ETSX(MMN) + xreg adapt on AirPassengers", {
    skip_on_cran()
    testModel <- adam(xreg, "MMN", h=18, holdout=TRUE, regressors="adapt")
    expect_match(testModel$regressors, "adapt")
})

# Forecast from ETSX with formula
test_that("Forecast for ADAM adaptive regression on AirPassengers", {
    skip_on_cran()
    testForecast <- forecast(testModel, h=18, newxreg=tail(xreg, 18), interval="simulated")
    expect_equal(testForecast$level, 0.95)
})

# ETSX with formula
test_that("ADAM ETSX(MMN) + xreg formula on AirPassengers", {
    skip_on_cran()
    testModel <- adam(xreg, "MMN", h=18, holdout=TRUE, formula=y~x, distribution="dnorm")
    expect_match(testModel$regressors, "use")
})

# Forecast from ETSX with formula
test_that("Forecast for ADAM ETSX(MMN) + xreg formula on AirPassengers", {
    skip_on_cran()
    testForecast <- forecast(testModel, h=18, newxreg=tail(xreg, 18), interval="nonp")
    expect_equal(testForecast$level, 0.95)
})

# Pure regression
test_that("ADAM regression (ALM) on AirPassengers", {
    skip_on_cran()
    testModel <- adam(xreg, "NNN", h=18, holdout=TRUE, formula=y~x+trend, distribution="dlnorm")
    expect_equal(modelType(testModel),"NNN")
})


#### ETS + ARIMA / ARIMA + ARIMAX ####
### ETS + ARIMA
# ETS(ANN) + ARIMA(0,2,2)
test_that("ADAM ETS(ANN) + ARIMA(0,2,2) on BJsales", {
    skip_on_cran()
    testModel <- adam(BJsales, "ANN", orders=c(0,2,2))
    expect_match(modelType(testModel), "ANN")
})

# ETS(ANN) + ARIMA(0,2,2) backcasting
test_that("ADAM ETS(ANN) + ARIMA(0,2,2) with backcasting on BJsales", {
    skip_on_cran()
    testModel <- adam(BJsales, "ANN", orders=c(1,1,2), initial="backcasting")
    expect_match(modelType(testModel), "ANN")
})

# ETS(ZZZ) + ARIMA(0,2,2)
test_that("ADAM ETS(ZZZ) + ARIMA(0,2,2) with logN on BJsales", {
    skip_on_cran()
    testModel <- adam(BJsales, "ZZZ", orders=c(2,0,2), distribution="dlnorm")
    expect_match(testModel$distribution, "dlnorm")
})

# ETS(ZZZ) + SARIMA(2,1,2)(2,1,1)[12]
test_that("ADAM ETS(ZZZ) + SARIMA(2,1,2)(2,1,1)[12] with logS on AirPassengers", {
    skip_on_cran()
    testModel <- adam(AirPassengers, "ZZZ", orders=list(ar=c(2,2),i=c(1,1), ma=c(2,1)), distribution="ds")
    expect_match(testModel$distribution, "ds")
})

# Forecast from ETS(ZZZ) + SARIMA(2,1,2)(2,1,1)[12]
test_that("Forecast of ADAM ETS(ZZZ) + SARIMA(2,1,2)(2,1,1)[12] with S", {
    skip_on_cran()
    testForecast <- forecast(testModel, h=18, interval="prediction", side="upper")
    expect_match(testForecast$side, "upper")
})

### ARIMA / ARIMAX
# Pure SARIMA(2,1,2)(2,1,1)[12], Normal
test_that("ADAM SARIMA(2,1,2)(2,1,2)[12] with Logistic on AirPassengers", {
    skip_on_cran()
    testModel <- adam(AirPassengers, "NNN", orders=list(ar=c(2,2),i=c(1,1), ma=c(2,2)), distribution="dgnorm")
    expect_match(testModel$distribution, "dgnorm")
})

# Forecast from SARIMA(2,1,2)(2,1,2)[12]
test_that("Forecast of ADAM SARIMA(2,1,2)(2,1,2)[12]", {
    skip_on_cran()
    testForecast <- forecast(testModel, h=18, interval="approximate", side="lower")
    expect_match(testForecast$side, "lower")
})

# ARIMAX
test_that("ADAM SARIMAX on AirPassengers", {
    skip_on_cran()
    testModel <- adam(xreg, "NNN", h=18, orders=list(ar=c(2,0),i=c(1,0), ma=c(2,1)), holdout=TRUE, formula=y~x)
    expect_match(testModel$distribution, "dnorm")
})

# ARIMAX with dynamic xreg
test_that("ADAM SARIMAX with dynamic xreg on AirPassengers", {
    skip_on_cran()
    testModel <- adam(xreg, "NNN", h=18, orders=list(ar=c(2,0),i=c(1,0), ma=c(2,1)), holdout=TRUE, formula=y~x, regressors="adapt")
    expect_equal(length(testModel$persistence), 15)
})

#### Provided initial / persistence / phi / arma / B / reuse the model ####
### Initials
# ETS(MMM) with provided level
test_that("ADAM ETS(MMM) with provided level on AirPassengers", {
    skip_on_cran()
    testModel <- adam(AirPassengers, "MMM", initial=list(level=5000))
    expect_false(testModel$initialEstimated["level"])
})

# ETS(MMM) with provided trend
test_that("ADAM ETS(MMM) with provided trend on AirPassengers", {
    skip_on_cran()
    testModel <- adam(AirPassengers, "MMM", initial=list(trend=1))
    expect_false(testModel$initialEstimated["trend"])
})

# ETS(MMM) with provided seasonal
test_that("ADAM ETS(MMM) with provided seasonal components on AirPassengers", {
    skip_on_cran()
    testModel <- adam(AirPassengers, "MMM", initial=list(seasonal=AirPassengers[1:12]))
    expect_false(testModel$initialEstimated["seasonal"])
})

# ETSX(MMN) with provided xreg initials
test_that("ADAM ETSX(MMN) with provided xreg initials on AirPassengers", {
    skip_on_cran()
    testModel <- adam(xreg, "MMN", h=18, holdout=TRUE, formula=y~x,
                      initial=list(xreg=c(-0.35,-.34,.27,-.46,.07,-0.28,-0.24,0.05,-0.28,-0.34,-0.01)))
    expect_false(testModel$initialEstimated["xreg"])
})

# ETS(ANN) + ARIMA(0,2,2) with provided initials for ARIMA
test_that("ADAM ETS(ANN) + ARIMA(0,2,2) with initials for ARIMA on BJsales", {
    skip_on_cran()
    testModel <- adam(BJsales, "ANN", orders=c(0,2,2), initial=list(arima=BJsales[1:2]))
    expect_false(testModel$initialEstimated["arima"])
})

# All provided initials
test_that("ADAM ETSX(MMM)+ARIMA(0,0,2) with provided initials on AirPassengers", {
    skip_on_cran()
    testModel <- adam(xreg, "MMM", formula=y~x, orders=c(0,0,2), lags=c(1,12))
    testModel <- adam(xreg, "MMM", formula=y~x, orders=c(0,0,2), lags=c(1,12), initial=testModel$initial)
    expect_true(all(!testModel$initialEstimated))
})

### Persistence
# ETS(MMM) with provided alpha
test_that("ADAM ETS(MMM) with provided alpha on AirPassengers", {
    skip_on_cran()
    testModel <- adam(AirPassengers, "MMM", persistence=list(alpha=0.1))
    expect_equivalent(testModel$persistence["alpha"],0.1)
})

# ETS(MMM) with provided beta
test_that("ADAM ETS(MMM) with provided beta on AirPassengers", {
    skip_on_cran()
    testModel <- adam(AirPassengers, "MMM", persistence=list(beta=0.1))
    expect_equivalent(testModel$persistence["beta"],0.1)
})

# ETS(MMM) with provided gamma
test_that("ADAM ETS(MMM) with provided gamma on AirPassengers", {
    skip_on_cran()
    testModel <- adam(AirPassengers, "MMM", persistence=list(gamma=0.1))
    expect_equivalent(testModel$persistence["gamma"],0.1)
})

# ETS(MMN) with provided deltas
test_that("ADAM ETS(MMN) with provided deltas on AirPassengers", {
    skip_on_cran()
    testModel <- adam(xreg, "MMN", formula=y~x, persistence=list(delta=0.01), regressors="adapt")
    expect_equivalent(testModel$persistence[substr(names(testModel$persistence),1,5)=="delta"],rep(0.01,12))
})

### Phi
# ETS(MMdM) with provided phi
test_that("ADAM ETS(MMdM) with provided phi on AirPassengers", {
    skip_on_cran()
    testModel <- adam(AirPassengers, "MMdM", phi=0.99)
    expect_equivalent(testModel$phi,0.99)
})

### arma parameters
# Provided AR parameters
test_that("ADAM ETS(MMM)+ARIMA(2,0,2) with provided AR on AirPassengers", {
    skip_on_cran()
    testModel <- adam(AirPassengers, "MMM", orders=c(2,0,2), arma=list(ar=c(0.2,0.3)))
    expect_equivalent(testModel$arma$ar,c(0.2,0.3))
})

# Provided MA parameters
test_that("ADAM ETS(MMM)+ARIMA(2,0,2) with provided MA on AirPassengers", {
    skip_on_cran()
    testModel <- adam(AirPassengers, "MMM", orders=c(2,0,2), arma=list(ma=c(-0.2,-0.4)))
    expect_equivalent(testModel$arma$ma,c(-0.2,-0.4))
})

# Provided ARMA parameters
test_that("ADAM ETS(MMM)+ARIMA(2,0,2) with provided ARMA on AirPassengers", {
    skip_on_cran()
    testModel <- adam(AirPassengers, "MMM", orders=c(2,0,2), arma=list(ar=c(0.2,0.3), ma=c(-0.2,-0.4)))
    expect_equivalent(testModel$arma$ar,c(0.2,0.3))
    expect_equivalent(testModel$arma$ma,c(-0.2,-0.4))
})

### B
# Provided starting parameters
test_that("ADAM ARIMA(2,0,2) with provided B on AirPassengers", {
    skip_on_cran()
    testModel <- adam(AirPassengers, "NNN", orders=c(2,0,2))
    testModel <- adam(AirPassengers, "NNN", orders=c(2,0,2), B=testModel$B)
    expect_equivalent(testModel$model,"ARIMA(2,0,2)")
})

### Model reused
# Reuse ETS
test_that("Reuse ADAM ETS(MMdM) on AirPassengers", {
    skip_on_cran()
    testModel <- adam(AirPassengers, "MMdM")
    testModelNew <- adam(AirPassengers, testModel)
    expect_equal(testModel$model,testModelNew$model)
    expect_equal(nparam(testModelNew),1)
})

# Reuse ARIMA
test_that("Reuse ADAM SARIMA(2,1,2)(0,0,1)[12] on AirPassengers", {
    skip_on_cran()
    testModel <- adam(AirPassengers, "NNN", h=18, orders=list(ar=c(2,0),i=c(1,0), ma=c(2,1)), holdout=TRUE)
    testModelNew <- adam(AirPassengers, testModel)
    expect_equal(testModel$model,testModelNew$model)
    expect_equal(nparam(testModelNew),1)
})

# Reuse ARIMAX
test_that("Reuse ADAM SARIMAX(2,1,2)(0,0,1)[12] with dynamic xreg on AirPassengers", {
    skip_on_cran()
    testModel <- adam(xreg, "NNN", h=18, orders=list(ar=c(2,0),i=c(1,0), ma=c(2,1)), holdout=TRUE, formula=y~x, regressors="adapt")
    testModelNew <- adam(xreg, testModel)
    expect_equal(testModel$persistence,testModelNew$persistence)
    expect_equal(nparam(testModelNew),1)
})

# Reuse ETSX + ARIMA
test_that("Reuse ADAM ETSX(ANN)+SARIMA(2,1,2)(0,0,1)[12] on AirPassengers", {
    skip_on_cran()
    testModel <- adam(xreg, "ANN", h=18, orders=list(ar=c(2,0),i=c(1,0), ma=c(2,1)), holdout=TRUE, formula=y~x)
    testModelNew <- adam(xreg, testModel)
    expect_equal(testModel$persistence,testModelNew$persistence)
    expect_equal(nparam(testModelNew),1)
})


#### auto.adam ####
# Select the best distribution for ETS(ZZZ) on 2568
test_that("Best auto.adam on AirPassengers", {
    skip_on_cran()
    testModel <- auto.adam(AirPassengers, "ZZZ")
    expect_match(testModel$loss, "likelihood")
})

# Outliers detection for ETS on series BJsales of M1 in parallel
test_that("Detect outliers for ETS(ZZZ) on BJsales", {
    skip_on_cran()
    testModel <- auto.adam(BJsales, "ZZZ", outliers="use")
    expect_match(testModel$loss, "likelihood")
})

# Best ARIMA on the 2568
test_that("Best auto.adam ARIMA on AirPassengers", {
    skip_on_cran()
    testModel <- auto.adam(AirPassengers, "NNN", orders=list(ar=c(3,2),i=c(2,1),ma=c(3,2),select=TRUE))
    expect_match(testModel$loss, "likelihood")
})

# Outliers detection for ARIMA on series BJsales of M1 in parallel
test_that("Detect outliers for ARIMA on BJsales", {
    skip_on_cran()
    testModel <- auto.adam(BJsales, "NNN", orders=list(ar=c(3,2),i=c(2,1),ma=c(3,2),select=TRUE),
                           outliers="use")
    expect_match(modelType(testModel),"NNN")
})

# Best ETS+ARIMA+Regression on the 2568
# Summary of the best model
test_that("Best auto.adam ETS+ARIMA+Regression on AirPassengers", {
    skip_on_cran()
    testModel <- auto.adam(xreg, "ZZZ", orders=list(ar=c(3,2),i=c(2,1),ma=c(3,2),select=TRUE),
                           lags=c(1,12), regressors="select", initial="back")
    testSummary <- summary(testModel)
    expect_match(testSummary$loss, "likelihood")
})

# Best ETS+ARIMA+Regression on the 2568
test_that("Best auto.adam ETS+ARIMA+Regression+outliers on AirPassengers", {
    skip_on_cran()
    testModel <- auto.adam(xreg, "ZZZ", orders=list(ar=c(3,2),i=c(2,1),ma=c(3,2),select=TRUE),
                           outliers="use", regressors="use", initial="back")
    expect_match(testModel$loss, "likelihood")
})

# Data passed by value, e.g. do.call(adam, list(y))
test_that("The response name survives data passed by value", {
    skip_on_cran()
    set.seed(42)
    y <- ts(100 + cumsum(rnorm(1500)))
    # deparse(substitute(data)) returns the whole series here, which is past R's
    # 10000-byte limit on names, so the checker must fall back to a plain one.
    expect_lt(nchar(colnames(do.call(adam, list(y, model="ANN", silent=TRUE))$data)[1]), 100)
    expect_lt(nchar(colnames(do.call(ces, list(y, silent=TRUE))$data)[1]), 100)
    expect_lt(nchar(colnames(do.call(ssarima, list(y, silent=TRUE))$data)[1]), 100)
})

#### Hannan-Rissanen starting values of the ARMA parameters ####
# arimaHRCpp() with the bounds code of adam's "usual" unless given
hr <- function(y, ar, ma, lags, arEstimate=TRUE, maEstimate=TRUE, arma=numeric(0),
               useLevel=rep(1, length(lags)), bounds=TRUE){
    return(as.vector(smooth:::arimaHRCpp(y, ar, ma, lags, arEstimate, maEstimate, arma, useLevel, bounds)))
}

test_that("Hannan-Rissanen recovers the parameters of ARMA(1,1) and SARMA(1,1)(1,1)[12]", {
    set.seed(41)
    y <- as.vector(arima.sim(list(ar=0.6, ma=0.3), 300))
    expect_equal(hr(y, 1, 1, 1), c(0.6, 0.3), tolerance=0.1, check.attributes=FALSE)
    set.seed(42)
    y <- as.vector(arima.sim(list(ar=c(0.7, rep(0,10), 0.5, -0.35), ma=c(0.4, rep(0,10), 0.3, 0.12)), 600))
    expect_equal(hr(y, c(1,1), c(1,1), c(1,12)), c(0.7, 0.4, 0.5, 0.3), tolerance=0.2,
                 check.attributes=FALSE)
})

test_that("Hannan-Rissanen values move inside the boundary only if the cost function rejects them", {
    set.seed(45)
    e <- rnorm(200)
    y <- numeric(200)
    for(t in 2:200){
        y[t] <- 1.03*y[t-1] + e[t]
    }
    arRaw <- hr(y, 1, 0, 1, maEstimate=FALSE, bounds=FALSE)
    expect_gt(arRaw, 1)
    expect_equal(hr(y, 1, 0, 1, maEstimate=FALSE, bounds=TRUE), 0.99)
    # Over-differenced white noise: HR lands on a non-invertible MA, which is
    # reflected to the invertible one with or without the bounds
    set.seed(47)
    w <- diff(rnorm(300), differences=2)
    maValues <- sapply(c(FALSE, TRUE), function(b){hr(w, 0, 1, 1, arEstimate=FALSE, bounds=b)})
    expect_lt(abs(maValues[1]), 1)
    expect_equal(maValues[2], maValues[1])
    # An invertible MA(2) with a coefficient above one is kept as it is
    set.seed(46)
    x <- as.vector(arima.sim(list(ma=c(1.6, 0.64)), 1000))
    expect_equal(hr(x, 0, 2, 1, arEstimate=FALSE, bounds=TRUE), hr(x, 0, 2, 1, arEstimate=FALSE, bounds=FALSE))
    expect_gt(max(abs(hr(x, 0, 2, 1, arEstimate=FALSE, bounds=TRUE))), 1)
})

test_that("Hannan-Rissanen falls back to the defaults and respects the provided values", {
    set.seed(41)
    y <- as.vector(arima.sim(list(ar=0.6, ma=0.3), 300))
    # Too few seasons for the seasonal level, and a level switched off
    expect_equal(hr(y[1:30], c(1,1), c(1,1), c(1,12))[3:4], c(0.1, -0.1))
    expect_equal(hr(y, c(1,1), c(1,1), c(1,12), useLevel=c(1,0))[3:4], c(0.1, -0.1))
    expect_equal(hr(numeric(0), 1, 1, 1), c(0.1, -0.1), check.attributes=FALSE)
    # AR provided: only MA is returned
    expect_length(hr(y, 1, 1, 1, arEstimate=FALSE, arma=0.6), 1)
    testModel <- msarima(AirPassengers, orders=list(ar=1,i=1,ma=1), arma=list(ar=0.5))
    expect_equal(testModel$arma$ar, 0.5, check.attributes=FALSE)
})

test_that("ARIMA starting values with missing and intermittent data", {
    y <- AirPassengers
    y[c(10,50,90)] <- NA
    expect_true(all(is.finite(msarima(y, orders=list(ar=1,i=1,ma=1), maxeval=1)$B)))
    set.seed(44)
    y <- ts(rpois(120, 0.7) * (1 + rnorm(120)^2))
    expect_true(all(is.finite(adam(y, "MNN", orders=list(ar=1), occurrence="odds-ratio", maxeval=1)$B)))
    # ETS with ARIMA and no differencing: the series is a ts without regressors
    expect_true(all(is.finite(adam(AirPassengers, "ANA", orders=list(ar=c(1,1), ma=c(1,1)),
                                   lags=c(1,12), initial="optimal", maxeval=1)$B)))
})

test_that("ARIMA with constant starts from the intercept consistent with AR", {
    set.seed(41)
    y <- ts(100 + arima.sim(list(ar=0.6, ma=0.3), 200))
    testModel <- msarima(y, orders=list(ar=1,i=0,ma=1), constant=TRUE)
    # The implied mean of the series, constant / (1 - phi)
    expect_equal(coef(testModel)["constant"] / (1 - coef(testModel)["phi1[1]"]), mean(y),
                 tolerance=0.01, check.attributes=FALSE)
})

test_that("A regressor that is a trend is kept rather than the auxiliary trend of its initials", {
    # alm(y~x+trend) dropped x, collinear with the trend, and the creator then failed
    x <- (0:299)/10
    testModel <- adam(data.frame(y=2*x+sin(x), x=x), "NNN", orders=list(i=1, ma=1), maxeval=1)
    expect_true(any(names(coef(testModel))=="x"))
})

test_that("Hannan-Rissanen runs on the residuals of the regression", {
    set.seed(48)
    x <- rnorm(200, 10, 5)
    dat <- data.frame(y=100 + 5*x + as.vector(arima.sim(list(ar=0.6), 200)), x=x)
    # On the series itself, the regressor hides the AR(1): phi started at -0.03
    for(testModel in list(adam(dat, "NNN", orders=list(ar=1), constant=TRUE, maxeval=1),
                          ssarima(dat, orders=list(ar=1), constant=TRUE, maxeval=1))){
        expect_equal(testModel$B[["phi1[1]"]], 0.6, tolerance=0.1)
    }
    # With differences, the regression is on the differenced series and regressors
    set.seed(49)
    x <- cumsum(rnorm(300))
    y <- 100 + 3*x + cumsum(as.vector(arima.sim(list(ma=0.5), 300)))
    xregDiffs <- cbind(1, diff(x))
    residuals <- diff(y) - xregDiffs %*% olsCpp(xregDiffs, diff(y))
    expect_equal(adam(data.frame(y=y, x=x), "NNN", orders=list(i=1, ma=1), maxeval=1)$B[["theta1[1]"]],
                 hr(residuals, 0, 1, 1, arEstimate=FALSE))
    # The seasonal MA taken from the series with monthly dummies stuck the fit at 826.34
    xreg <- data.frame(y=as.vector(AirPassengers), temporaldummy(AirPassengers)[,-1])
    testModel <- suppressWarnings(adam(xreg, "MMN", lags=c(1,12), orders=list(ma=c(0,2)),
                                       distribution="dnorm", regressors="adapt"))
    expect_lt(testModel$lossValue, 516)
})

test_that("ETS+ARIMA keeps the defaults for the seasonal ARIMA at the ETS seasonal lag", {
    testModel <- adam(AirPassengers, "MAM", orders=list(ar=c(1,1),ma=c(1,1)), lags=c(1,12), maxeval=1)
    expect_equal(testModel$B[c("phi1[12]","theta1[12]")], c(0.1, -0.1), check.attributes=FALSE)
    # The start is feasible: no penalty from the bounds
    expect_lt(testModel$lossValue, 1e+100)
})

test_that("ARIMA starting values under non-normal distributions and non-likelihood losses", {
    skip_on_cran()
    orders <- list(ar=c(1,1),i=c(1,1),ma=c(1,1))
    for(distribution in c("dlaplace","ds","dgnorm","dgamma","dinvgauss","dlnorm")){
        start <- adam(AirPassengers, "NNN", orders=orders, lags=c(1,12), distribution=distribution, maxeval=1)
        testModel <- adam(AirPassengers, "NNN", orders=orders, lags=c(1,12), distribution=distribution)
        expect_true(all(is.finite(start$B)))
        expect_lte(testModel$lossValue, start$lossValue)
    }
    for(loss in c("MAE","HAM","MSEh","TMSE","GTMSE","MSCE","GPL")){
        start <- adam(AirPassengers, "NNN", orders=orders, lags=c(1,12), loss=loss, h=12, maxeval=1)
        testModel <- adam(AirPassengers, "NNN", orders=orders, lags=c(1,12), loss=loss, h=12)
        expect_true(all(is.finite(start$B)))
        expect_lte(testModel$lossValue, start$lossValue)
    }
})

#### ARIMA initials: the initial state of the companion form ####
# The fitted values of a pure ARI model with the initials built from the
# pre-sample values y_pre are the ARI predictions from the series extended by y_pre
ariFitError <- function(y, arOrders, iOrders, lags, arValues, distribution="dnorm"){
    orders <- list(ar=arOrders, i=iOrders, ma=rep(0, length(lags)))
    arma <- if(sum(arOrders)>0) list(ar=unlist(arValues)) else NULL
    fit <- function(B){
        return(adam(y, "NNN", orders=orders, lags=lags, initial="optimal", arma=arma,
                    distribution=distribution, B=B, maxeval=1))
    }
    template <- fit(NULL)$B
    set.seed(1)
    yPre <- as.vector(y)[1] + rnorm(length(template), 0, sd(y)/4)
    # Expanded ARI polynomial
    ari <- 1
    for(j in seq_along(lags)){
        factorAR <- c(1, rep(0, length(arValues[[j]])*lags[j]))
        factorAR[seq_along(arValues[[j]])*lags[j]+1] <- -arValues[[j]]
        ari <- convolve(ari, rev(factorAR), type="open")
        for(d in seq_len(iOrders[j])){
            ari <- convolve(ari, rev(c(1, rep(0, lags[j]-1), -1)), type="open")
        }
    }
    Etype <- if(distribution=="dgamma") "M" else "A"
    B <- adam_arimaInitials(ari, switch(Etype, "M"=1/yPre, -yPre), Etype)
    names(B) <- names(template)
    extended <- c(yPre, as.vector(y))
    if(distribution=="dgamma"){
        extended <- log(extended)
    }
    n <- length(B) + 20
    predicted <- sapply(1:n, function(t){-sum(ari[-1] * extended[t + length(B) - seq_along(ari[-1])])})
    if(distribution=="dgamma"){
        predicted <- exp(predicted)
    }
    return(max(abs(predicted - fitted(fit(B))[1:n])))
}

test_that("ARIMA initials give the ARI predictions from the pre-sample values", {
    expect_lt(ariFitError(AirPassengers, c(0,0), c(1,1), c(1,12), list(numeric(0), numeric(0))), 1e-8)
    expect_lt(ariFitError(AirPassengers, c(1,1), c(1,1), c(1,12), list(0.3, 0.4)), 1e-8)
    expect_lt(ariFitError(AirPassengers, c(0,2), c(0,1), c(1,12), list(numeric(0), c(0.3,0.2))), 1e-8)
    expect_lt(ariFitError(AirPassengers, c(1,1), c(1,1), c(1,12), list(0.3, 0.4), "dgamma"), 1e-8)
})

test_that("ARIMA initials of a double seasonal model give the ARI predictions", {
    skip_on_cran()
    set.seed(3)
    y <- ts(100 + cumsum(rnorm(800))/5 + 5*sin(2*pi*(1:800)/24) + 3*sin(2*pi*(1:800)/168), frequency=24)
    expect_lt(ariFitError(y, c(1,1,0), c(0,1,1), c(1,24,168), list(0.5, 0.3, numeric(0))), 1e-8)
})

test_that("Two-stage passes the backcasted ARIMA initials on without loss", {
    skip_on_cran()
    y <- log(AirPassengers)
    for(orders in list(list(ar=c(0,0),i=c(1,1),ma=c(1,1)), list(ar=c(1,1),i=c(1,1),ma=c(1,1)))){
        backcasted <- msarima(y, orders=orders, lags=c(1,12), initial="backcasting")
        B <- c(backcasted$B, unlist(backcasted$initial$arima))
        names(B) <- names(msarima(y, orders=orders, lags=c(1,12), initial="optimal", maxeval=1)$B)
        expect_lte(msarima(y, orders=orders, lags=c(1,12), initial="optimal", B=B, maxeval=1)$lossValue,
                   backcasted$lossValue + 1e-6)
    }
})

test_that("The backcasting seed follows the ARMA parameters", {
    # Near the unit root the two backcasting iterations do not forget the seed, so
    # it has to be built with the current ARI polynomial, not the differences only
    testModel <- adam(BJsales, "NNN", orders=list(ar=2,i=1,ma=2), constant=TRUE, maxeval=1)
    B <- setNames(c(-0.0365, 0.7256, 0.2485, -0.4908, 0.1378), names(testModel$B))
    lossDefault <- adam(BJsales, "NNN", orders=list(ar=2,i=1,ma=2), constant=TRUE, B=B, maxeval=1)$lossValue
    lossConverged <- adam(BJsales, "NNN", orders=list(ar=2,i=1,ma=2), constant=TRUE, B=B, maxeval=1,
                          nIterations=20)$lossValue
    expect_lt(abs(lossDefault - lossConverged), 0.01)
})

test_that("Every ARIMA initial enters the fit, as in ssarima", {
    orders <- list(ar=c(0,0),i=c(1,0),ma=c(0,1))
    testModel <- msarima(log(AirPassengers), orders=orders, lags=c(1,12), initial="optimal")
    initials <- grep("ARIMAState", names(testModel$B))
    expect_length(initials, 12)
    expect_equal(nparam(testModel),
                 nparam(ssarima(log(AirPassengers), orders=orders, lags=c(1,12), initial="optimal")))
    lossChange <- sapply(initials, function(i){
        B <- testModel$B
        B[i] <- B[i] + 0.1
        return(msarima(log(AirPassengers), orders=orders, lags=c(1,12), initial="optimal",
                       B=B, maxeval=1)$lossValue - testModel$lossValue)
    })
    expect_true(all(abs(lossChange) > 1e-8))
})

#### Stationarity and invertibility, factor by factor ####
test_that("confint keeps the ARMA parameters within their stationary / invertible factors", {
    # A parameter of an AR(3) with the others fixed, against the roots
    x <- c(0.1, 0.2, 0.3)
    grid <- seq(-3, 3, 0.001)
    stable <- sapply(grid, function(v){x[2] <- v; return(all(Mod(polyroot(c(1,-x)))>1))})
    expect_equal(as.vector(arimaParameterBoundsCpp(c(0.1, 0.2, 0.3), 1, -1)), range(grid[stable]),
                 tolerance=2e-3)
    expect_equal(as.vector(arimaParameterBoundsCpp(c(1.2, 0.5), 1, 1)), c(0.2, 1))
    expect_true(all(is.nan(arimaParameterBoundsCpp(1.2, 0, 1))))
    # The old AR bounds stopped at +-4.82; phi1 of a stationary AR(2) lies within (phi2-1, 1-phi2)
    testModel <- msarima(BJsales, orders=list(ar=2, i=1, ma=2))
    phi2 <- coef(testModel)[["phi2[1]"]]
    expect_equal(unname(confint(testModel)["phi1[1]", 2:3]), c(phi2-1, 1-phi2), tolerance=1e-6)
    # The upper bound of the admissible alpha of ANN, (0, 2), was stuck at 5.01
    testModel <- adam(BJsales, "ANN", bounds="admissible")
    expect_equal(unname(eigenBounds(testModel, as.matrix(testModel$persistence), 1)[2]), 2.01)
})

test_that("The ARIMA bounds reject exactly the non-invertible MA and non-stationary AR", {
    y <- log(AirPassengers)
    lossAt <- function(orders, values, fitter=msarima){
        B <- setNames(values, names(fitter(y, orders=orders, lags=c(1,12), maxeval=1)$B))
        return(fitter(y, orders=orders, lags=c(1,12), B=B, maxeval=1)$lossValue)
    }
    # The MA is 1 + theta_1 B^12 + theta_2 B^24
    ordersMA <- list(ar=c(0,0), i=c(1,1), ma=c(0,2))
    for(theta in list(c(1.058,0.793), c(-1.058,0.793), c(0.5,0.6), c(0.3,-1.2), c(1.2,0.1))){
        invertible <- all(Mod(polyroot(c(1, theta)))>1)
        expect_equal(lossAt(ordersMA, theta) < 1E+100, invertible)
        expect_equal(lossAt(ordersMA, theta, ssarima) < 1E+100, invertible)
    }
    # The AR is 1 - phi_1 B^12 - phi_2 B^24
    ordersAR <- list(ar=c(0,2), i=c(1,0), ma=c(0,0))
    for(phi in list(c(0.5,0.4), c(1.2,-0.3), c(-0.5,0.6), c(0.3,0.8))){
        expect_equal(lossAt(ordersAR, phi) < 1E+100, all(Mod(polyroot(c(1, -phi)))>1))
    }
})

test_that("A fit stuck on a penalty restarts from zero smoothing and small ARMA parameters", {
    skip_on_cran()
    xreg <- data.frame(y=AirPassengers, x=factor(temporaldummy(AirPassengers, factors=TRUE)))
    # The first run starts on the NaN penalty and moves to the bounds penalties
    expect_no_warning(testModel <- adam(xreg, "MMN", lags=c(1,12), orders=list(ar=c(0,0),i=c(0,0),ma=c(0,2)),
                                        regressors="adapt", distribution="dnorm"))
    expect_lt(testModel$lossValue, 1E+100)
})

test_that("LASSO / RIDGE with lambda=1 fix the parameters at their shrinkage targets", {
    # Nothing is left with backcasting: the model is used as it is, with a zero loss
    testModel <- adam(AirPassengers, "AAN", loss="RIDGE", lambda=1)
    expect_length(testModel$B, 0)
    expect_equal(unname(testModel$persistence), c(0, 0))
    expect_equal(testModel$lossValue, 0)
    testModel <- adam(log(AirPassengers), "NNN", orders=list(ar=c(1,1),i=c(1,1),ma=c(1,1)),
                      lags=c(1,12), loss="LASSO", lambda=1)
    expect_equal(unname(unlist(testModel$arma)), c(1, 1, 0, 0))
    # With optimal initials, only these are estimated
    expect_named(adam(AirPassengers, "AAN", loss="RIDGE", lambda=1, initial="optimal")$B,
                 c("level", "trend"))
})

test_that("LASSO / RIDGE shrink the estimated parameters wherever they are in B", {
    set.seed(3)
    x <- data.frame(y=as.vector(AirPassengers), x1=rnorm(144), x2=rnorm(144, 5))
    # The constant after the regressors is not shrunk, the regressors are
    testModel <- adam(x, "NNN", orders=list(ar=1,i=1,ma=1), constant=TRUE, loss="LASSO", lambda=0.1)
    expect_equal(unname(coef(testModel)[c("x1","x2")]), c(0, 0), tolerance=1e-4)
    expect_gt(abs(coef(testModel)[["drift"]]), 1)
    # With the AR provided, the MA parameters are shrunk to zero, not to one
    testModel <- adam(log(AirPassengers), "NNN", orders=list(ar=c(1,1),i=c(1,1),ma=c(1,1)),
                      lags=c(1,12), arma=list(ar=c(0.2,0.3)), loss="LASSO", lambda=0.1)
    expect_equal(unname(coef(testModel)), c(0, 0), tolerance=1e-3)
})

test_that("Pure regression stores sigma^2 as the scale, as the other ADAM models", {
    testModel <- adam(cbind(y=BJsales, x=BJsales.lead), "NNN", silent=TRUE);
    expect_equal(testModel$scale, mean(residuals(testModel)^2));
})

test_that("Pure regression is forecasted and predicted by forecast.alm() / predict.alm()", {
    xreg <- data.frame(y=as.vector(BJsales), x=as.vector(BJsales.lead));
    testModel <- adam(xreg, "NNN", h=10, holdout=TRUE, silent=TRUE);
    almModel <- alm(y~x, head(xreg, 140));
    for(interval in c("none", "prediction", "confidence")){
        testForecast <- forecast(testModel, h=10, interval=interval);
        almForecast <- forecast(almModel, h=10, newdata=tail(xreg, 10), interval=interval);
        expect_equal(as.vector(testForecast$mean), as.vector(almForecast$mean));
        expect_equal(start(testForecast$mean)[1], 141);
        if(interval!="none"){
            expect_equal(as.vector(testForecast$lower), as.vector(almForecast$lower));
            expect_equal(as.vector(predict(testModel, interval=interval)$upper),
                         as.vector(predict(almModel, interval=interval)$upper));
        }
    }
})

# Every path of a draw starts from the profile and uses the persistence of that draw
test_that("reforecast keeps the paths of a draw on its own profile and persistence", {
    fit <- adam(BJsales, "ANN");
    h <- 3;
    paths <- fit$adamCpp$reforecast(array(1, c(h, 2, 2)), array(1, c(h, 2, 2)), array(1, c(h, 1, 2)),
                                     array(1, c(1, 1, 2)), matrix(c(0, 1), 1, 2),
                                     matrix(0L, 1, h), array(200, c(1, 1, 2)), "A")$data;
    expect_equal(paths[,,1], matrix(201, h, 2));
    expect_equal(paths[,,2], matrix(201:203, h, 2));
});

# Without newdata, each regressor is forecast by adam() into its own column
test_that("the regressors forecast without newdata land in their own columns", {
    set.seed(41);
    x1 <- rnorm(120, 10, 2);
    x2 <- 50+cumsum(rnorm(120));
    data <- cbind(y=100+3*x1+cumsum(rnorm(120)), x1=x1, x2=x2);
    fit <- adam(data, "ANN", formula=y~x1+x2);
    forecasted <- suppressWarnings(forecast(fit, h=5));
    x1Forecast <- adam(data[,"x1"], h=5, silent=TRUE)$forecast;
    x2Forecast <- adam(data[,"x2"], h=5, silent=TRUE)$forecast;
    expect_equal(as.numeric(forecasted$mean),
                 as.numeric(fit$states[nrow(fit$states),"level"] + fit$initial$xreg[["x1"]]*x1Forecast +
                                fit$initial$xreg[["x2"]]*x2Forecast), tolerance=1e-10);
});

test_that("a named B is matched by name, not by position", {
    testModel <- adam(BJsales, "AAN", distribution="dgnorm");
    refit <- adam(BJsales, "AAN", distribution="dgnorm", B=rev(testModel$B), maxeval=1);
    expect_equal(refit$B, testModel$B);
});

test_that("the models of a combination count their regressors once", {
    xregData <- cbind(y=BJsales, x=BJsales.lead);
    combined <- adam(xregData, "CXN");
    expect_match(combined$model, "^ETSX");
    expect_equal(nparam(combined$models$ANN), nparam(adam(xregData, "ANN")));
});

test_that("the point forecast is the skeleton, the mean or the median", {
    testModel <- adam(BJsales, "ANN");
    skeleton <- forecast(testModel, h=12)$mean;
    expect_equal(forecast(testModel, h=12, point="mean")$mean, skeleton);
    expect_equal(forecast(testModel, h=12, point="median")$mean, skeleton);
    # A multiplicative error on additive components: the skeleton is the mean, and
    # the median is lower, the distribution being skewed to the right
    testModel <- adam(BJsales, "MNN");
    skeleton <- forecast(testModel, h=12)$mean;
    expect_equal(forecast(testModel, h=12, point="mean")$mean, skeleton);
    expect_true(all(forecast(testModel, h=12, point="median")$mean < skeleton));
    # A multiplicative trend: the mean is the skeleton at the first step only
    testModel <- adam(BJsales, "MMdN");
    set.seed(41);
    forecasted <- forecast(testModel, h=12, point="mean")$mean;
    skeleton <- forecast(testModel, h=12)$mean;
    expect_equal(forecasted[1], skeleton[1]);
    expect_equal(as.numeric(forecasted), as.numeric(skeleton), tolerance=1e-3);
    # The median of an intermittent demand with a probability below 0.5 is zero
    set.seed(3);
    yIntermittent <- rpois(120, 0.6)*rep(c(5,1),60);
    testModel <- adam(yIntermittent, "MNN", occurrence="odds-ratio");
    set.seed(41);
    expect_true(all(forecast(testModel, h=6, point="median")$mean==0));
    expect_equal(forecast(testModel, h=6, point="mean")$mean, forecast(testModel, h=6)$mean);
    expect_error(forecast(testModel, h=6, point="mode"));
});

test_that("reforecast() totals each path for the cumulative forecasts", {
    testModel <- adam(BJsales, "AAN");
    set.seed(41);
    forecasted <- reforecast(testModel, h=5, cumulative=TRUE, nsim=50);
    expect_equal(as.numeric(forecasted$mean), sum(forecast(testModel, h=5)$mean));
    expect_true(forecasted$lower < forecasted$mean && forecasted$mean < forecasted$upper);
    set.seed(41);
    expect_equal(as.numeric(reforecast(testModel, h=5, cumulative=TRUE, nsim=50, point="mean")$mean),
                 sum(forecast(testModel, h=5)$mean), tolerance=1e-2);
});

test_that("adam keeps the missing values apart from the zeros of the occurrence", {
    set.seed(3)
    y <- rbinom(200, 1, 0.4) * exp(rnorm(200, 2, 0.3))
    y[c(10:20, 100)] <- NA
    for(distribution in c("dgamma", "dinvgauss", "dnorm")){
        testModel <- suppressWarnings(adam(y, "MNN", occurrence="odds-ratio",
                                           distribution=distribution))
        expect_equal(attr(logLik(testModel), "nobs"), sum(!is.na(y)))
        expect_equal(as.numeric(logLik(testModel)), sum(pointLik(testModel)))
    }
    # Without the occurrence, the missing values are skipped in the same way
    testModel <- suppressWarnings(adam(y + 1, "ANN"))
    expect_equal(attr(logLik(testModel), "nobs"), sum(!is.na(y)))
    expect_equal(as.numeric(logLik(testModel)), sum(pointLik(testModel), na.rm=TRUE))
})

test_that("adam warns when more than half of the data is missing", {
    y <- rnorm(100, 100, 10)
    y[1:60] <- NA
    expect_warning(adam(y, "ANN"), "More than half of the in-sample data is missing")
})

test_that("adam uses the filled values only for the initialisation", {
    y <- AirPassengers
    y[c(10, 50, 51, 90, 140)] <- NA
    testModel <- suppressWarnings(adam(y, "MAM", h=12, holdout=TRUE))
    # Missing values alone are not an occurrence model
    expect_equal(modelName(testModel), "ETS(MAM)")
    expect_true(all(is.finite(fitted(testModel))))
    expect_equal(AICc(testModel), AICc(logLik(testModel)))
    # The holdout keeps its gap, and the accuracy is on its observed values
    expect_true(is.na(testModel$holdout[8, 1]))
    expect_equal(testModel$accuracy[["ME"]],
                 mean(as.numeric(testModel$holdout[-8, 1]) - as.numeric(testModel$forecast)[-8]))
    # The holdout does not enter the fill of the in-sample gaps
    yChanged <- y
    yChanged[144] <- 10000
    expect_equal(coef(suppressWarnings(adam(yChanged, "MAM", h=12, holdout=TRUE))), coef(testModel))
    # The multistep loss is over the windows with all their targets observed
    expect_true(is.finite(suppressWarnings(adam(y, "ANN", loss="TMSE", h=6))$lossValue))
    expect_true(all(is.finite(forecast(testModel, h=6, interval="semiparametric")$upper)))
    expect_false(all(is.na(rstudent(testModel))))
})

test_that("A regression with missing values keeps its rows aligned", {
    set.seed(1)
    xreg <- data.frame(y=as.numeric(AirPassengers), x1=rnorm(144), x2=as.numeric(AirPassengers)/10+rnorm(144))
    xreg$y[c(10, 50, 51)] <- NA
    testModel <- suppressWarnings(adam(xreg, "NNN", h=6, holdout=TRUE))
    reference <- adam(xreg[-c(10, 50, 51),], "NNN", h=6, holdout=TRUE)
    expect_equal(coef(testModel), coef(reference))
    expect_equal(as.numeric(logLik(testModel)), as.numeric(logLik(reference)))
    expect_true(is.na(residuals(testModel)[10]))
    expect_true(is.finite(fitted(testModel)[10]))
})

test_that("rstudent of dgamma leaves its own observation out with an occurrence model", {
    set.seed(3)
    y <- rbinom(200, 1, 0.4) * exp(rnorm(200, 2, 0.3))
    testModel <- adam(y, "MNN", occurrence="odds-ratio")
    errors <- residuals(testModel)
    used <- which(y!=0)
    i <- used[5]
    expect_equal(as.numeric(rstudent(testModel)[i]), as.numeric(errors[i] / mean(errors[used[used!=i]])))
})

test_that("The empirical interval at h=1 takes the errors, as the multistep ones", {
    testModel <- adam(AirPassengers, "MAM", h=12, holdout=TRUE)
    errors <- as.vector(testModel$residuals)
    expect_equal(as.numeric(forecast(testModel, h=1, interval="empirical")$upper),
                 as.numeric(testModel$forecast[1]) * (1 + quantile(errors, 0.975, type=7)),
                 check.attributes=FALSE)
})

test_that("The smoothing parameters of the regressors with a provided alpha", {
    set.seed(41)
    x <- rnorm(120, 0, 1)
    xregData <- data.frame(y=100*exp(0.002*(1:120) + 0.1*x + rnorm(120, 0, 0.02)), x=x)
    testModel <- adam(xregData, "ANN", persistence=list(alpha=0.3), regressors="adapt")
    expect_equal(testModel$persistence[["alpha"]], 0.3)
    expect_equal(testModel$persistence[["delta1"]], testModel$B[["delta1"]])
    # The usual bounds of delta hold with one regressor too
    expect_true(testModel$B[["delta1"]]>=0 && testModel$B[["delta1"]]<=1)
})

test_that("The draws of reapply() start the states of the regressors", {
    set.seed(41)
    x <- rnorm(120, 10, 2)
    xregData <- data.frame(y=100 + 0.5*(1:120) + 10*sin(2*pi*(1:120)/12) + 3*x + rnorm(120), x=x)
    for(model in c("AAN","AAA")){
        for(initial in c("backcasting","optimal")){
            testModel <- adam(xregData, model, lags=c(1,12), initial=initial)
            refitted <- reapply(testModel, nsim=5)
            expect_equal(refitted$states["x",1,], refitted$randomParameters[,"x"], check.attributes=FALSE)
        }
    }
})

test_that("The constant is a ratio with a multiplicative error and regressors", {
    set.seed(41)
    x <- rnorm(120, 0, 1)
    xregData <- data.frame(y=100*exp(0.002*(1:120) + 0.1*x + rnorm(120, 0, 0.02)), x=x)
    testModel <- adam(xregData, "MNN", constant=TRUE, initial="optimal")
    expect_equal(as.numeric(fitted(testModel)[2]),
                 as.numeric(testModel$states[2,"level"] * testModel$B[["drift"]] *
                                exp(testModel$B[["x"]] * x[2])))
})

test_that("ets='adam' changes nothing without ETS components", {
    for(orders in list(list(ar=1,i=1,ma=1), list(ar=1,i=1,ma=0))){
        testModel <- adam(BJsales, "NNN", orders=orders, ets="conventional")
        testModelADAM <- adam(BJsales, "NNN", orders=orders, ets="adam")
        expect_equal(as.numeric(logLik(testModelADAM)), as.numeric(logLik(testModel)))
        expect_equal(testModelADAM$B, testModel$B)
    }
})
