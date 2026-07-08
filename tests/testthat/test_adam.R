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
