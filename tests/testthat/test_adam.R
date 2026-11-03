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

# The occurrence and the regressors are counted once in a combination
test_that("ADAM combination counts the occurrence and the regressors once", {
    skip_on_cran()
    set.seed(41);
    y <- rgamma(120, shape=5, scale=2) * rbinom(120, 1, 0.6);
    testCombined <- adam(y, "CNN", occurrence="fixed");
    expect_equal(nparam(testCombined), sum(sapply(testCombined$models, nparam) * testCombined$ICw));
    xregData <- data.frame(y=as.numeric(BJsales), x=rnorm(150), z=rnorm(150));
    testCombined <- adam(xregData, "CNN", regressors="select");
    expect_equal(nparam(testCombined), sum(sapply(testCombined$models, nparam) * testCombined$ICw));
    expect_equal(testCombined$nParam[1,2],
                 sum(sapply(testCombined$models, function(x){x$nParam[1,2]}) * testCombined$ICw));
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

# The regression keeps the factors of a data frame, takes the intercept of the means with the
# provided coefficients, and works without an intercept
test_that("ADAM regression with factors, provided coefficients and no intercept", {
    set.seed(41);
    regressionData <- data.frame(y=rnorm(60, 50), x1=rnorm(60, 10), x2=rnorm(60, 5),
                                 colour=factor(sample(c("red","green","blue"), 60, TRUE)));
    expect_true(is.factor(suppressWarnings(adam(regressionData, "NNN"))$data$colour));
    testModel <- adam(regressionData[,1:3], "NNN", initial=list(xreg=c(x1=2, x2=-3)));
    expect_equal(coef(testModel),
                 c("(Intercept)"=with(regressionData, mean(y)-2*mean(x1)+3*mean(x2)), x1=2, x2=-3));
    testModel <- suppressWarnings(adam(regressionData, "NNN", formula=y~x1+x2-1));
    expect_equal(dim(testModel$measurement), c(60, 2));
})

# The prediction interval of an additive error with a distribution of 1+e is around the fitted
test_that("predict() of ETS(A,A,N) with dgamma brackets the fitted values", {
    testModel <- adam(BJsales, "AAN", distribution="dgamma");
    testPredict <- predict(testModel, interval="prediction");
    expect_true(all(testPredict$lower < testPredict$mean & testPredict$mean < testPredict$upper));
})

# The names with special characters stay in the data of the model, the model takes the
# syntactic ones, and the forecasts take the newdata with the names of the user
test_that("ADAM with special characters in the names of the variables", {
    set.seed(41);
    specialData <- data.frame(`my y`=rnorm(60, 50), `x 1`=rnorm(60, 10), check.names=FALSE);
    testModel <- suppressWarnings(adam(specialData, "ANN", h=3, holdout=TRUE));
    expect_equal(colnames(testModel$data), c("my y", "x 1"));
    expect_length(forecast(testModel, h=3, newdata=specialData[58:60,])$mean, 3);
    expect_length(suppressWarnings(adam(specialData, "NNN", h=3))$forecast, 3);
    expect_length(suppressWarnings(adam(specialData, "ANN", formula=`my y`~`x 1`, h=3))$forecast, 3);
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
    expect_warning(testModel <- auto.adam(AirPassengers, "NNN", orders=list(ar=c(3,2),i=c(2,1),ma=c(3,2),select=TRUE)),
                   "Not all distributions make sense for the pure ARIMA")
    expect_match(testModel$loss, "likelihood")
})

# Outliers detection for ARIMA on series BJsales of M1 in parallel
test_that("Detect outliers for ARIMA on BJsales", {
    skip_on_cran()
    expect_warning(testModel <- auto.adam(BJsales, "NNN", orders=list(ar=c(3,2),i=c(2,1),ma=c(3,2),select=TRUE),
                                          outliers="use"),
                   "Not all distributions make sense for the pure ARIMA")
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
    expect_warning(testModel <- msarima(y, orders=list(ar=1,i=1,ma=1), maxeval=1), "Data contains NAs")
    expect_true(all(is.finite(testModel$B)))
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
    # The admissible alpha of ANN is [0, 2): alpha=2 puts the root of the MA polynomial 1 + B on
    # the unit circle, and alpha=0 cancels the unit root of the level. The bounds are bisected to
    # the tolerance of the condition (a grid of 0.01 gave (-0.01, 2); the upper one was once 5.01)
    testModel <- adam(BJsales, "ANN", bounds="admissible")
    expect_equal(eigenBounds(testModel, as.matrix(testModel$persistence), 1), c(0, 2), tolerance=1e-8)
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

test_that("sigma() of the additive distributions is the root of the de-biased mean squared error", {
    for(distribution in c("dnorm", "dlaplace", "ds", "dgnorm")){
        testModel <- adam(BJsales, "AAN", distribution=distribution, silent=TRUE);
        df <- nobs(testModel) - nparam(testModel) + testModel$nParam[1,4];
        expect_equal(sigma(testModel), sqrt(sum(residuals(testModel)^2)/df));
    }
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

test_that("sm() and pls() take the missing values for gaps", {
    y <- AirPassengers;
    y[c(10, 50, 51, 90)] <- NA;
    testModel <- suppressWarnings(adam(y, "MAM", h=12, holdout=TRUE));
    # The gaps of the location model are not an occurrence of the scale
    scaleModel <- suppressWarnings(sm(testModel));
    expect_false(grepl("iETS", scaleModel$model));
    expect_equal(attr(logLik(scaleModel), "nobs"), sum(!is.na(y[1:132])));
    # pls() over the holdout of any length, and the observed values of it only
    holdout <- as.vector(testModel$holdout);
    holdout[c(3, 5)] <- NA;
    covarMat <- multicov(testModel, h=12)[-c(3, 5), -c(3, 5)];
    errors <- log(holdout[-c(3, 5)]) - log(as.vector(testModel$forecast)[-c(3, 5)]);
    expect_equal(pls(testModel, holdout=holdout),
                 -as.vector(log(2*pi*det(covarMat))/2 + t(errors) %*% solve(covarMat) %*% errors/2) -
                     sum(log(holdout[-c(3, 5)])));
})

test_that("sm() takes the observations with zero errors as its missing values", {
    testModel <- adam(AirPassengers, "ANN", distribution="dlaplace");
    zeros <- which(residuals(testModel)==0);
    expect_length(zeros, 5);
    # A seasonal scale model stays multiplicative, its states are not updated on the zeros, and
    # its likelihood keeps them
    expect_warning(scaleModel <- sm(testModel, lags=c(1,12)), "observations with zero errors");
    expect_match(scaleModel$model, "^ETS\\(M");
    expect_null(scaleModel$occurrence);
    expect_true(all(is.finite(fitted(scaleModel)) & fitted(scaleModel)>0));
    expect_equal(sum(pointLik(scaleModel)), as.numeric(logLik(scaleModel)));
    # The same as the data of the scale model missing there, for the fixed model
    expect_warning(scaleModel <- sm(testModel, model="MNN", persistence=0.3), "observations with zero errors");
    gapModel <- suppressWarnings(adam(replace(abs(residuals(testModel)), zeros, NA), "MNN",
                                      persistence=0.3, initial=scaleModel$initial$level,
                                      distribution="dlaplace"));
    expect_equal(as.vector(fitted(scaleModel)), as.vector(fitted(gapModel)));
})

# The arguments of sm() reach adam() as values: through the variables of a function, they were
# not found, and a model through a variable was taken for an additive one and fitted in logs
test_that("sm() takes the values of its arguments", {
    testModel <- adam(AirPassengers, "ANN");
    scaleLiteral <- suppressWarnings(sm(testModel, model="MNN", lags=12));
    scaleFunction <- function(object, modelSM, lagsSM){
        return(suppressWarnings(sm(object, model=modelSM, lags=lagsSM)));
    }
    scaleVariable <- scaleFunction(testModel, "MNN", 12);
    expect_equal(scaleVariable$model, scaleLiteral$model);
    expect_equal(logLik(scaleVariable), logLik(scaleLiteral));
})

# A regression of the scale predicts it at the zero errors, its missing values: it failed there
test_that("sm() with a regression and zero errors", {
    set.seed(1);
    testData <- data.frame(y=as.numeric(AirPassengers), x=rnorm(144));
    testModel <- adam(testData, "ANN", formula=y~x);
    expect_true(any(residuals(testModel)==0));
    scaleModel <- suppressWarnings(sm(testModel, model="NNN", formula=y~x));
    expect_equal(scaleModel$model, "Regression in logs");
    expect_equal(as.vector(fitted(scaleModel)), as.vector(exp(scaleModel$B[1]+scaleModel$B[2]*testData$x)));
})

# A regression with a provided occurrence has its fitted values at their rows, the prediction
# times the occurrence: they were recycled over all the observations
test_that("A regression with a provided occurrence", {
    set.seed(1);
    testData <- data.frame(y=as.numeric(AirPassengers), x=rnorm(144));
    zeros <- c(1,50);
    testData$y[zeros] <- 0;
    testOccurrence <- replace(rep(1,144), zeros, 0);
    expect_no_warning(testModel <- adam(testData, "NNN", formula=y~x, occurrence=testOccurrence));
    expect_equal(as.vector(fitted(testModel)),
                 as.vector(testModel$B[1]+testModel$B[2]*testData$x)*testOccurrence);
    expect_equal(as.vector(residuals(testModel))[zeros], c(0,0));
})

test_that("The point likelihoods of sm() sum to its log-likelihood", {
    set.seed(41);
    yIntermittent <- rbinom(200, 1, 0.6) * exp(rnorm(200, 2, 0.3));
    for(testModel in list(adam(AirPassengers, "MAM"), adam(AirPassengers, "ANN", distribution="dlaplace"),
                          adam(yIntermittent, "MNN", occurrence="odds-ratio"))){
        # The random walk of ANN gives observations with zero errors, the gaps of the scale model
        if(testModel$distribution=="dlaplace"){
            expect_warning(scaleModel <- sm(testModel), "observations with zero errors");
        }
        else{
            scaleModel <- sm(testModel);
        }
        expect_equal(sum(pointLik(scaleModel)), as.numeric(logLik(scaleModel)));
    }
})

test_that("The multistep errors start from the states before their targets", {
    # Row i forecasts the targets i..i+h-1, so the first column is the residuals,
    # and MSEh with h=1 is the MSE
    for(model in list(adam(AirPassengers, "ANN"), adam(AirPassengers, "AAA", lags=12),
                      adam(AirPassengers, "NNN", orders=list(ar=2, i=1)))){
        errors <- rmultistep(model, h=3);
        expect_equal(nrow(errors), nobs(model)-2);
        expect_equal(as.numeric(errors[,1]), as.numeric(residuals(model))[1:nrow(errors)],
                     tolerance=1e-8);
    }
    fitMSE <- adam(BJsales, "ANN", persistence=0.3, initial="optimal", loss="MSE");
    fitMSEh <- adam(BJsales, "ANN", persistence=0.3, initial="optimal", loss="MSEh", h=1,
                    B=fitMSE$B, maxeval=1);
    expect_equal(fitMSEh$lossValue, fitMSE$lossValue, tolerance=1e-10);
});

# Over the periods without an observed size, the states of a pure additive model with the
# normal distribution are filtered: for ETS(A,N,N) the variance of the level relative to
# sigma^2 grows by alpha^2 over them and becomes P (1-alpha)^2 / (P+1) at an observed size,
# and the variance of the error is sigma^2 (P+1), in the likelihood, the scale, the
# diagnostics and the forecasts
test_that("the filter of a pure additive model over the zeros and the missing values", {
    set.seed(41);
    e <- rnorm(300, 0, 5);
    level <- 100 + cumsum(0.3*e);
    y <- (c(100, level[-300]) + e)*rbinom(300, 1, 0.5);
    y[c(50, 51, 299, 300)] <- NA;
    testModel <- suppressWarnings(adam(y, "ANN", occurrence="fixed", distribution="dnorm"));
    otLogical <- !is.na(y) & y!=0;
    alpha <- testModel$persistence[1];
    gapVariance <- rep(1, 303);
    P <- 0;
    for(t in 1:303){
        gapVariance[t] <- P + 1;
        P <- if(t<=300 && otLogical[t]){P*(1-alpha)^2/(P+1)}else{P+alpha^2};
    }
    expect_equal(adam_gapVarianceModel(testModel), gapVariance[1:300]);
    expect_equal(testModel$scale, mean(residuals(testModel)[otLogical]^2/gapVariance[1:300][otLogical]));
    expect_equal(sum(pointLik(testModel)), as.numeric(logLik(testModel)));
    expect_equal(var(rstandard(testModel)[otLogical]), 1, tolerance=0.1);
    # The h steps after the sample are unobserved too
    s2 <- adam_varianceDebiased(testModel);
    testForecast <- forecast(testModel, h=3, interval="approximate", level=0.9, occurrence=rep(1, 3));
    expect_equal(as.vector(testForecast$upper - testForecast$mean),
                 qnorm(0.95, 0, sqrt(s2*gapVariance[301:303])));
})


# The pure multiplicative ADAM ETS is additive in logs, where its states are filtered: for
# ETS(M,N,N), log(1+e) is N(-s a/2, s k), with k = P+1 as for ETS(A,N,N) and a = 1 - 2x, x
# being the derivative of the mean of the log of the level in the scale, which decreases
# by alpha/2 over the periods without an observed size and is corrected by the gain
# (P+alpha)/k at the observed ones, and the scale is the maximum likelihood of that
test_that("the filter of the pure multiplicative ADAM ETS with dlnorm", {
    set.seed(41);
    u <- rnorm(300, -0.1, sqrt(0.2));
    level <- 10*exp(cumsum(c(0, 0.3*u[-300])));
    y <- level*exp(u)*rbinom(300, 1, 0.4);
    testModel <- adam(y, "MNN", occurrence="fixed", distribution="dlnorm", ets="adam");
    otLogical <- y!=0;
    alpha <- testModel$persistence[1];
    gapVariance <- gapMean <- rep(1, 300);
    P <- x <- 0;
    for(t in 1:300){
        gapVariance[t] <- P + 1;
        gapMean[t] <- 1 - 2*x;
        x <- x - alpha/2 + otLogical[t]*(P+alpha)/(P+1)*(0.5-x);
        P <- if(otLogical[t]){P*(1-alpha)^2/(P+1)}else{P+alpha^2};
    }
    expect_equal(adam_gapVarianceModel(testModel), gapVariance);
    expect_equal(adam_gapVarianceModel(testModel, 1), gapMean);
    expect_equal(sum(pointLik(testModel)), as.numeric(logLik(testModel)));
    # The scale maximises the likelihood given the other parameters
    logResiduals <- log(residuals(testModel)[otLogical]);
    likelihood <- function(scale){
        sum(dnorm(logResiduals, -scale*gapMean[otLogical]/2, sqrt(scale*gapVariance[otLogical]),
                  log=TRUE));
    }
    expect_equal(testModel$scale, optimize(likelihood, c(0.01, 2), maximum=TRUE)$maximum,
                 tolerance=1e-4);
})


# The log-normal 1+e has the mean of one and the variance s of its log: the one-step
# interval is qlnorm(-s/2, sqrt(s)) around the fitted values, and for the pure
# multiplicative ADAM ETS the h-step one is qlnorm(-s a_h/2, sqrt(s k_h)) exactly
test_that("the dlnorm intervals come from logN(-s/2, s)", {
    set.seed(41);
    u <- rnorm(300, -0.1, sqrt(0.2));
    y <- 10*exp(cumsum(c(0, 0.3*u[-300])))*exp(u);
    testModel <- adam(y, "MNN", distribution="dlnorm", ets="adam", h=5);
    s2 <- adam_varianceDebiased(testModel);
    testPredict <- predict(testModel, interval="prediction", level=0.9);
    expect_equal(as.vector(testPredict$upper/fitted(testModel)), rep(qlnorm(0.95, -s2/2, sqrt(s2)), 300));
    testForecast <- forecast(testModel, h=5, interval="approximate", level=0.9);
    alpha <- testModel$persistence[1];
    expect_equal(as.vector(testForecast$upper/testForecast$mean),
                 qlnorm(0.95, -s2*(1+(0:4)*alpha)/2, sqrt(s2*(1+(0:4)*alpha^2))));
})

# The intervals of an occurrence model are of the sizes, so they are around the fitted
# values of the sizes, not around their products with the probability
test_that("the intervals of an occurrence model are around the sizes", {
    set.seed(41);
    y <- rnorm(300, 100, 10)*rbinom(300, 1, 0.4);
    testModel <- adam(y, "ANN", occurrence="fixed", h=5);
    p <- fitted(testModel$occurrence)[1];
    s2 <- adam_varianceDebiased(testModel);
    testPredict <- predict(testModel, interval="prediction", level=0.9);
    expect_equal(as.vector(testPredict$upper - fitted(testModel)/p),
                 rep(qnorm((1+(0.9-(1-p))/p)/2, 0, sqrt(s2)), 300));
    testForecast <- forecast(testModel, h=1, interval="approximate", level=0.9);
    expect_equal(as.vector(testForecast$upper - testForecast$mean/p),
                 qnorm((1+(0.9-(1-p))/p)/2, 0, sqrt(s2)));
})


# The analytical covariance of the multistep errors of ETS(A,N,N) has 1 + (i-1) alpha^2 on
# the diagonal and alpha + (i-1) alpha^2 above it, from the C++ shared with Python
test_that("covarAnal() of ETS(A,N,N) is the closed form", {
    covarMat <- covarAnal(1, 5, matrix(1), matrix(1), 0.3, 2);
    expect_equal(diag(covarMat), 2*(1 + (0:4)*0.09));
    expect_equal(covarMat[1,-1], rep(2*0.3, 4));
    expect_equal(covarMat[3,5], 2*(0.3 + 2*0.09));
})

# The moments of the errors in logs of the conventional ETS(M,N,N), u = log(1+e) and
# log(1 + alpha e), are those of the simulated ones. Over j-1 periods without an observed size
# after an observed one, its filter moves the level by them, so the variance of its drift at the
# next observed size is (j-1) times that of log(1 + alpha e); and the fixed point of the moment
# equations with such a drift gives the scale
test_that("the drift of the states of the pure multiplicative ETS after the gaps", {
    set.seed(41);
    noise <- adam_gapDrift(c(0.3), c(TRUE, TRUE), "dgamma", FALSE, 0.2)$noise;
    errors <- rgamma(3e5, shape=5, scale=0.2);
    innovations <- cbind(log(errors), log(1 + 0.3*(errors - 1)));
    expect_equal(noise$mean, colMeans(innovations), tolerance=1e-2);
    expect_equal(noise$covariance, cov(innovations), tolerance=1e-2);

    e <- rgamma(300, shape=5, scale=0.2);
    y <- 10*cumprod(c(1, 1+0.3*(e[-300]-1)))*e;
    y[c(50, 51, 52)] <- 0;
    y[200:300] <- y[200:300]*rbinom(101, 1, 0.5);
    testModel <- adam(y, "MNN", occurrence="fixed", distribution="dgamma");
    otLogical <- y!=0;
    noise <- adam_gapDrift(testModel$persistence, otLogical, "dgamma", FALSE, testModel$scale)$noise;
    drift <- attr(adam_gapVarianceModel(testModel), "drift");
    expect_equal(drift$variance[1:49], rep(0, 49));
    expect_equal(drift$variance[50], 3*noise$covariance[2,2]);

    # The scale of errors (1+e) D - 1 is s, with the drift D of the simulated innovations
    gaps <- rgeom(1e5, 0.35) + 1;
    otLogical <- rep(FALSE, sum(gaps));
    otLogical[cumsum(gaps)] <- TRUE;
    driftValues <- sapply(gaps, function(j){prod(1 + 0.3*(rgamma(j-1, shape=5, scale=0.2) - 1))});
    errors <- rgamma(1e5, shape=5, scale=0.2)*driftValues - 1;
    gapDrift <- function(scale){
        noise <- adam_gapDrift(c(0.3), otLogical, "dgamma", FALSE, scale)$noise;
        return(list(mean=(gaps-1)*noise$mean[2], variance=(gaps-1)*noise$covariance[2,2]));
    }
    expect_equal(adam_scaler("dgamma", "M", errors, NULL, 1e5, NULL, gapDrift=gapDrift), 0.2, tolerance=2e-2);
})

# The Gamma, the Inverse Gaussian and the conventional log-normal after the periods without
# an observed size: the point likelihoods sum to the log-likelihood, and the fits without
# such periods are those of the one-step density
test_that("the likelihood of the pure multiplicative ETS with the drift after the gaps", {
    set.seed(41);
    e <- rgamma(300, shape=5, scale=0.2);
    y <- 10*cumprod(c(1, 1+0.3*(e[-300]-1)))*e*rbinom(300, 1, 0.35);
    for(distribution in c("dgamma","dinvgauss","dlnorm")){
        testModel <- adam(y, "MNN", occurrence="fixed", distribution=distribution);
        expect_equal(sum(pointLik(testModel)), as.numeric(logLik(testModel)));
    }
    # ADAM ETS, whose filter runs in logs with the errors of mean zero and the variance one
    for(distribution in c("dgamma","dinvgauss")){
        testModel <- adam(y, "MNN", occurrence="fixed", distribution=distribution, ets="adam");
        expect_equal(sum(pointLik(testModel)), as.numeric(logLik(testModel)));
    }
    testModel <- adam(AirPassengers, "MNN", distribution="dgamma");
    # residuals() of the Gamma are the ratios 1+e
    expect_equal(testModel$scale, mean((residuals(testModel)-1)^2));
})

# The pure additive models with the Laplace, the S and the Generalised Normal distributions
# take the variances of the filter after the periods without an observed size, with the scale
# s k^(1/p) of the variance k times the one-step one, p of adam_scalePower()
test_that("the filter of the pure additive models with the other distributions", {
    set.seed(41);
    y <- (100 + cumsum(rnorm(300, 0, 3)) + rlaplace(300, 0, 3))*rbinom(300, 1, 0.4);
    otLogical <- y!=0;
    for(distribution in c("dlaplace","ds","dgnorm")){
        testModel <- adam(y, "ANN", occurrence="fixed", distribution=distribution);
        gapVariance <- adam_gapVarianceModel(testModel);
        expect_true(all(gapVariance[otLogical] >= 1) && any(gapVariance[otLogical] > 1));
        expect_equal(sum(pointLik(testModel)), as.numeric(logLik(testModel)));
    }
    testModel <- adam(y, "ANN", occurrence="fixed", distribution="dlaplace");
    gapVariance <- adam_gapVarianceModel(testModel)[otLogical];
    expect_equal(testModel$scale, mean(abs(residuals(testModel)[otLogical])/sqrt(gapVariance)));
})

# With the k last periods without an observed size, the forecasts of the pure multiplicative
# ETS with the drift of the states start from the last observed size, as h+k steps ahead: for
# ETS(M,N,N), the level at that size, with the h+k step variance, and the simulated forecasts of
# any model, which then take the uncertainty of the states over those periods as the filter
test_that("the forecasts after the last observed size", {
    set.seed(41);
    e <- rgamma(300, shape=5, scale=0.2);
    y <- 10*cumprod(c(1, 1+0.3*(e[-300]-1)))*e*rbinom(300, 1, 0.6);
    y[295:300] <- 0;
    testModel <- adam(y, "MNN", occurrence="fixed", distribution="dgamma");
    last <- max(which(y!=0));
    trailing <- 300 - last;
    testForecast <- forecast(testModel, h=3, interval="approximate", level=0.9, occurrence=rep(1, 3));
    expect_equal(as.vector(testForecast$mean),
                 rep(unname(testModel$states[nrow(testModel$states)-trailing, 1]), 3));
    variances <- tail(adamVarAnal(1, 3+trailing, matrix(1), matrix(1), testModel$persistence,
                                  mean(adam_varianceDebiased(testModel))), 3);
    expect_equal(as.vector(testForecast$upper/testForecast$mean),
                 qgamma(0.95, shape=1/variances, scale=variances));

    z <- as.numeric(AirPassengers);
    z[141:144] <- NA;
    testModel <- suppressWarnings(adam(z, "AAN", distribution="dnorm"));
    testApproximate <- forecast(testModel, h=3, interval="approximate", level=0.9);
    set.seed(41);
    testSimulated <- forecast(testModel, h=3, interval="simulated", level=0.9, nsim=50000);
    expect_equal(as.vector(testSimulated$upper), as.vector(testApproximate$upper), tolerance=5e-3);
})

test_that("the cumulative forecasts and the scale models after the last observed size", {
    z <- as.numeric(AirPassengers);
    z[c(10,50,51,90,140,142:144)] <- NA;
    testModel <- suppressWarnings(adam(z, "AAN", distribution="dnorm"));
    # The variance of the sum over the h steps after the sample, from the last observed value
    testForecast <- forecast(testModel, h=4, interval="approximate", level=0.9, cumulative=TRUE);
    variances <- covarAnal(1, 7, matrix(1,1,2), testModel$transition, testModel$persistence,
                           mean(adam_varianceDebiased(testModel)))[4:7,4:7];
    expect_equal(as.vector(testForecast$upper), sum(forecast(testModel, h=4)$mean) +
                     qnorm(0.95)*sqrt(sum(variances)));
    set.seed(41);
    testSimulated <- forecast(testModel, h=4, interval="simulated", level=0.9, cumulative=TRUE, nsim=50000);
    expect_equal(as.vector(testSimulated$upper), as.vector(testForecast$upper), tolerance=5e-3);

    # The scale of the k last periods is the fitted one of the scale model
    testModel <- implant(testModel, suppressWarnings(sm(testModel, model="MNN")));
    testApproximate <- forecast(testModel, h=4, interval="approximate", level=0.9);
    set.seed(41);
    testSimulated <- forecast(testModel, h=4, interval="simulated", level=0.9, nsim=50000);
    expect_equal(as.vector(testSimulated$upper), as.vector(testApproximate$upper), tolerance=5e-3);
    testModel <- suppressWarnings(adam(z, "MNM", distribution="dgamma"));
    testModel <- implant(testModel, suppressWarnings(sm(testModel, model="MNN")));
    expect_length(predict(testModel)$mean, length(z));

    # The profile at the last observed value, with the corrections of the filter after the
    # missing values inside the sample, gives the forecasts from the end of the sample
    for(testModel in list(suppressWarnings(adam(ts(z, frequency=12), "ANA", distribution="dnorm")),
                          suppressWarnings(adam(ts(z, frequency=12), "NNN", lags=c(1,12),
                                                orders=list(ar=c(0,1), i=c(1,1), ma=c(1,1)))),
                          suppressWarnings(tbats(ts(z, frequency=12), distribution="dnorm", lambda=1,
                                                 trend="none", harmonics=2,
                                                 orders=list(ar=0, ma=0, select=FALSE))))){
        expect_equal(as.vector(forecast(testModel, h=4, interval="simulated", nsim=10)$mean),
                     as.vector(forecast(testModel, h=4)$mean));
    }
})

# ADAM ETS moves the level by (1+e)^alpha, whose mean is below one: the mean drifts below the
# skeleton after the first step, and the prediction interval is simulated
test_that("the mean of ADAM ETS with a multiplicative error", {
    set.seed(41);
    e <- rgamma(300, shape=5, scale=0.2);
    y <- 10*cumprod(c(1, 1+0.3*(e[-300]-1)))*e;
    testModel <- adam(y, "MNN", distribution="dgamma", ets="adam");
    scaleValue <- adam_scaleSimulation(testModel, testModel$scale);
    alpha <- testModel$persistence[1];
    drift <- exp(lgamma(1/scaleValue+alpha) + alpha*log(scaleValue) - lgamma(1/scaleValue));
    skeleton <- as.vector(forecast(testModel, h=6)$mean);
    set.seed(41);
    testForecast <- forecast(testModel, h=6, point="mean", nsim=100000);
    expect_equal(as.vector(testForecast$mean)[1], skeleton[1]);
    expect_equal(as.vector(testForecast$mean), skeleton*drift^(0:5), tolerance=3e-3);
    expect_equal(forecast(testModel, h=6, interval="prediction", nsim=10)$interval, "simulated");
})


# The variance of a sum of the multiplicative errors needs the covariances of the horizons
test_that("the cumulative intervals of the multiplicative models are simulated", {
    testModel <- adam(AirPassengers, "MNM", lags=12, h=4, holdout=FALSE);
    set.seed(41);
    testForecast <- forecast(testModel, h=4, interval="approximate", cumulative=TRUE, nsim=20000);
    expect_equal(testForecast$interval, "simulated");
    set.seed(41);
    testSimulated <- forecast(testModel, h=4, interval="simulated", cumulative=TRUE, nsim=20000);
    expect_equal(testForecast$lower, testSimulated$lower);
    expect_equal(as.vector(testForecast$mean), sum(forecast(testModel, h=4)$mean), tolerance=1e-3);
    testModel <- adam(AirPassengers, "ANN", h=4, holdout=FALSE);
    expect_equal(forecast(testModel, h=4, interval="approximate", cumulative=TRUE)$interval,
                 "approximate");
})

# The scale model in logs sums its forecasts in the space of the data
test_that("the cumulative forecasts and the reforecasts of sm() in logs", {
    testModel <- suppressWarnings(sm(adam(AirPassengers, "ANA", lags=12, h=4, holdout=FALSE), model="ANN"));
    expect_match(testModel$model, "in logs");
    expect_equal(as.vector(forecast(testModel, h=4, cumulative=TRUE)$mean),
                 sum(forecast(testModel, h=4)$mean));
    set.seed(41);
    expect_equal(apply(reapply(testModel, nsim=100)$refitted, 1, median), as.vector(fitted(testModel)),
                 tolerance=0.05);
    set.seed(41);
    testForecast <- forecast(testModel, h=2, interval="simulated", nsim=20000);
    set.seed(41);
    testReforecast <- reforecast(testModel, h=2, nsim=200);
    expect_equal(as.vector(testReforecast$lower), as.vector(testForecast$lower), tolerance=0.2);
})

# The fixed point of the scale diverges on this series (|h'(s)| > 1): the scale is the root of
# its moment equations, and the fit leaves its starting values
test_that("the scale of ADAM ETS after the gaps where its fixed point diverges", {
    set.seed(1);
    q <- rgamma(200, 2, scale=0.5);
    y <- 100*exp(cumsum(c(0, 0.6*log(q))))[1:200]*q*rep(c(0,0,0,0,1), 40);
    for(distribution in c("dgamma","dinvgauss")){
        testModel <- suppressWarnings(adam(y, "MNN", occurrence="fixed", distribution=distribution, ets="adam"));
        expect_true(is.finite(testModel$scale));
        expect_true(is.finite(logLik(testModel)) && logLik(testModel) > -1e300);
    }
})

# The approximate intervals of ADAM ETS with the Gamma and the Inverse Gaussian take the drift of
# the states over the horizon (and over the last unobserved periods), close to the simulated ones
test_that("the approximate intervals of ADAM ETS with the drift", {
    set.seed(5);
    q <- rgamma(120, 1/0.1, scale=0.1);
    y <- 100*exp(cumsum(c(0, 0.3*log(q))))[1:120]*q*(1+0.2*sin(2*pi*(1:120)/12));
    y[115:120] <- NA;
    for(distribution in c("dgamma","dinvgauss")){
        testModel <- suppressWarnings(adam(y, "MNM", lags=12, distribution=distribution, ets="adam"));
        testApproximate <- forecast(testModel, h=12, interval="approximate", level=0.9);
        set.seed(1);
        testSimulated <- forecast(testModel, h=12, interval="simulated", level=0.9, nsim=20000);
        expect_equal(as.vector(testApproximate$upper), as.vector(testSimulated$upper), tolerance=0.05);
        expect_equal(as.vector(testApproximate$lower), as.vector(testSimulated$lower), tolerance=0.05);
    }
})

# FI=TRUE returns the Fisher Information matrix of the estimated model, not TRUE
test_that("ADAM with FI=TRUE returns the Fisher Information matrix", {
    testModel <- adam(BJsales, "AAdN", FI=TRUE);
    expect_true(is.matrix(testModel$FI));
    expect_equal(colnames(testModel$FI), names(coef(testModel)));
    expect_equal(testModel$FI, adam(BJsales, model=testModel, FI=TRUE)$FI);
})

# The Asymmetric Laplace is not a distribution of adam() and is rejected by it
test_that("ADAM rejects distribution=\"dalaplace\"", {
    expect_error(adam(AirPassengers, "MNN", distribution="dalaplace"), "should be one of");
    expect_error(auto.adam(AirPassengers, "MNN", distribution="dalaplace"));
})

# The provided initials follow the order of the documentation: level, trend, seasonal, ARIMA, xreg
test_that("ADAM uses the provided initials of all the components", {
    seasonals <- c(rep(-10,3), rep(10,3), rep(0,6));
    y <- ts(100 + rep(seasonals,10) + sin(1:120), frequency=12);
    testModel <- adam(y, "ANA", initial=c(100,seasonals), persistence=c(0.1,0.1));
    expect_equal(unname(testModel$initial$level), 100);
    expect_equal(unname(testModel$initial$seasonal), seasonals);
    expect_equal(testModel$nParam["Provided","nParamInternal"], 15);
    expect_equal(unname(adam(y, "ANA", initial=list(100,NULL,seasonals))$initial$seasonal),
                 seasonals);
    expect_equal(unname(adam(y, "AAN", initial=list(100,1))$initial$trend), 1);

    xreg <- data.frame(y=as.vector(y), a=cos(1:120), b=sin(1:120/3));
    testModel <- adam(xreg, "ANN", initial=list(level=100, xreg=c(b=-3,a=2)));
    expect_equal(testModel$initial$xreg, c(a=2,b=-3));
    expect_equal(adam(xreg, "ANN", initial=c(100,2,-3))$initial$xreg, c(a=2,b=-3));
})

# A model that selected no regressors is reused on the response only
test_that("ADAM reuses a model with no regressors selected", {
    xreg <- data.frame(y=100+sin(1:120), a=cos(1:120/7), b=sin(1:120/11));
    testModel <- adam(xreg, "ANN", regressors="select");
    expect_null(testModel$initial$xreg);
    testReused <- adam(xreg, model=testModel, regressors="select");
    expect_equal(as.numeric(logLik(testReused)), as.numeric(logLik(testModel)));
})

# The formula is applied to the data first, and the regressors are selected over its columns
test_that("ADAM with formula and regressors='select' selects the transformed regressors", {
    set.seed(41);
    x1 <- runif(150, 1, 10);
    xregData <- data.frame(y=100+8*log(x1)+rnorm(150), x1=x1, x2=rnorm(150), x3=rnorm(150));
    testModel <- adam(xregData, "ANN", formula=y~log(x1)+x2+x3, regressors="select",
                      holdout=TRUE, h=10);
    expect_true("log(x1)" %in% attr(terms(formula(testModel)), "term.labels"));
    expect_true("log(x1)" %in% names(coef(testModel)));
    expect_length(forecast(testModel, h=10, newdata=xregData[141:150,])$mean, 10);
})

test_that("ADAM with formula and regressors='select' refits without regressors when none is selected", {
    set.seed(41);
    xregData <- data.frame(y=100+cumsum(rnorm(120)), x1=runif(120, 1, 10), x2=rnorm(120));
    testModel <- adam(xregData, "ANN", formula=y~log(x1)+x2, regressors="select");
    expect_equal(names(coef(testModel)), "alpha");
    expect_equal(formula(testModel), y~1, ignore_attr=TRUE);
    expect_equal(colnames(coefbootstrap(testModel, nsim=2)$coefficients), "alpha");
})

# The regressors are selected on the non-zero values, whatever the occurrence model
test_that("ADAM with an occurrence model and regressors='select'", {
    set.seed(41);
    xregData <- data.frame(y=0, a=rnorm(150), n1=rnorm(150), n2=rnorm(150));
    xregData$y <- rbinom(150, 1, plogis(2*xregData$a)) * (10+3*xregData$a+rnorm(150));
    omModel <- om(xregData$y[1:140], "MNN", occurrence="odds-ratio");
    occurrences <- list("auto","fixed","odds-ratio","inverse-odds-ratio","direct","general",omModel);
    for(occurrence in occurrences){
        testModel <- adam(xregData[1:140,], "ANN", occurrence=occurrence, regressors="select");
        expect_true("a" %in% names(coef(testModel)));
        expect_true(all(is.finite(forecast(testModel, h=10, newdata=xregData[141:150,])$mean)));
    }
    testModel <- adam(xregData[1:140,], "MNN", occurrence="auto", regressors="select",
                      formula=y~a+n1);
    expect_true("a" %in% names(coef(testModel)));
    # Without newdata, h beyond the sample takes the last values of the regressors
    testModel <- adam(xregData, "ANN", occurrence="auto", regressors="select", h=5);
    expect_length(testModel$forecast, 5);
    # The occurrence model takes the newdata as given, with its own regressors
    testModel <- adam(xregData[1:140,], "ANN", occurrence="direct", regressors="select");
    expect_equal(forecast(testModel, h=10, newdata=xregData[141:150,])$mean,
                 forecast(testModel, h=10, newdata=xregData[141:150,],
                          occurrence=forecast(testModel$occurrence, h=10,
                                              newdata=xregData[141:150,])$mean)$mean);
})

test_that("ADAM counts the provided initial level as provided and the rest as estimated", {
    testModel <- adam(AirPassengers, "AAA", lags=12, initial=list(level=100));
    # alpha, beta, gamma, the trend and 11 seasonal initials
    expect_equal(testModel$nParam[,"nParamInternal"], c(Estimated=15, Provided=1));
    expect_equal(dfInitialsETSLevelSeasonal(12, FALSE), 12);
})

test_that("ADAM takes the provided MA once, and reapply() draws around the fitted ARIMA", {
    # The MA of a lag without ARI was added twice to the persistence (psi = 1 for ma = 0.5)
    testModel <- adam(BJsales, "NNN", orders=list(ma=1), arma=list(ma=0.5), h=0);
    expect_equal(unname(testModel$persistence), 0.5);
    testModel <- adam(BJsales, "NNN", orders=list(ma=1), h=0);
    testReapply <- reapply(testModel, nsim=20);
    expect_true(all(abs(testReapply$persistence - testModel$persistence) < 0.2));
})

test_that("The admissible bounds of ETS+ARIMA check the whole model", {
    # The penalty of the bounds checker for a model with an estimated MA of the given reflection
    checker <- function(model, maReflection, etsModel){
        adamElements <- list(vecG=matrix(model$persistence, dimnames=list(names(model$persistence), NULL)),
                             matF=model$transition, matWt=model$measurement);
        return(adam_bounds_checker(adamElements, list(arReflection=0, maReflection=maReflection),
                                   "admissible", etsModel, FALSE, FALSE, etsModel*1, etsModel*1, 0,
                                   TRUE, FALSE, TRUE, FALSE, "use", 0, 1,
                                   modelLags(model), nobs(model), FALSE));
    }
    # Each part is admissible (alpha in (0, 2), |theta| < 1), the whole model is not: 1 + B - 0.1 B^2
    # has a root inside the unit circle
    testModel <- adam(BJsales, "ANN", orders=list(ma=1), persistence=c(alpha=1.9), arma=list(ma=0.1), h=0);
    expect_gt(checker(testModel, 0.1, TRUE), 0);
    # The MA root on the unit circle cancels with the difference: a constant level plus noise
    testModel <- adam(BJsales, "NNN", orders=list(i=1, ma=1), arma=list(ma=-1), h=0);
    expect_equal(checker(testModel, 1, FALSE), 0);
    # Without a difference to cancel with, a non-invertible MA is not admissible
    testModel <- adam(BJsales, "NNN", orders=list(ma=1), arma=list(ma=1.2), h=0);
    expect_gt(checker(testModel, 1.2, FALSE), 0);
})

test_that("coefbootstrap() refits a model with provided probabilities of occurrence", {
    set.seed(7);
    y <- ts(exp(2 + rnorm(150, 0, 0.3)) * rbinom(150, 1, 0.7), frequency=7);
    probabilities <- rep(c(0.6, 0.8), 75);
    testModel <- adam(y, "MNN", occurrence=probabilities);
    expect_true(all(is.finite(coefbootstrap(testModel, nsim=5)$coefficients)));
    # Each refit takes the probabilities of its observations
    newCall <- adam_bootstrapData(testModel$call, "data", y[11:60], probabilities[11:60]);
    expect_equal(newCall$occurrence, probabilities[11:60]);
})

test_that("adam() with the provided occurrence or probabilities of occurrence", {
    set.seed(7);
    y <- ts(exp(2 + rnorm(150, 0, 0.3)) * rbinom(150, 1, 0.7), frequency=7);
    nonzero <- y != 0;
    # Probabilities: a mixture with known probabilities, the sizes fitted to the non-zeros
    probabilities <- rep(c(0.6, 0.8), 75);
    testModel <- adam(y, "MNN", occurrence=c(probabilities, rep(0.75, 5)), h=5);
    level <- testModel$states[1:150, 1];
    sizes <- sum(dgamma(y[nonzero], shape=1/testModel$scale,
                        scale=testModel$scale*level[nonzero], log=TRUE));
    expect_equal(as.numeric(logLik(testModel)),
                 sizes + sum(log(probabilities[nonzero])) + sum(log(1-probabilities[!nonzero])));
    expect_equal(sum(pointLik(testModel)), as.numeric(logLik(testModel)));
    expect_equal(as.vector(fitted(testModel)), level*probabilities);
    expect_equal(as.vector(forecast(testModel, h=5)$mean), rep(0.75*tail(level, 1), 5));
    expect_warning(forecast(testModel, h=7), "horizon is not provided");
    expect_warning(forecast(adam(y, "MNN", occurrence=probabilities), h=3),
                   "horizon is not provided");
    # Zeros and ones: used as they are, without uncertainty
    testModel <- adam(y, "MNN", occurrence=c(nonzero*1, rep(1, 5)), h=5);
    expect_equal(sum(pointLik(testModel)), as.numeric(logLik(testModel)));
    expect_equal(as.vector(fitted(testModel))[!nonzero], rep(0, sum(!nonzero)));
    expect_warning(adam(y, "MNN", occurrence=probabilities, h=5), "horizon is not provided");
})

#### Printing and plotting ####
set.seed(41);
xregData <- data.frame(y=rnorm(100, 100, 5), x1=rnorm(100));
xregData$y <- xregData$y + 3*xregData$x1;
set.seed(7);
yIntermittent <- ts(exp(2 + rnorm(120, 0, 0.3)) * rbinom(120, 1, 0.7), frequency=12);
printModels <- list(ets=adam(BJsales, "AAdN", h=10, holdout=TRUE),
                    arima=adam(AirPassengers, "NNN", lags=c(1,12),
                               orders=list(ar=c(1,0), i=c(1,1), ma=c(1,1))),
                    xreg=adam(xregData, "ANN", regressors="use"),
                    occurrence=adam(yIntermittent, "MNN", occurrence="fixed"),
                    combined=adam(BJsales, "CCN", h=10, holdout=TRUE));

test_that("print() of adam() shows the model, its parameters and its holdout errors", {
    testModel <- printModels$ets;
    printed <- paste(capture.output(print(testModel)), collapse="\n");
    expect_match(printed, "Model estimated using adam() function: ETS(AAdN)", fixed=TRUE);
    expect_match(printed, paste(format(round(testModel$persistence, 4)), collapse=" +"));
    expect_match(printed, paste0("Damping parameter: ", round(testModel$phi, 4)), fixed=TRUE);
    expect_match(printed, "Sample size: 140", fixed=TRUE);
    expect_match(printed, paste0("ME: ", round(testModel$accuracy["ME"], 3), "; MAE: ",
                                 round(testModel$accuracy["MAE"], 3)), fixed=TRUE);
    # ARIMA: the AR and MA parameters by lag
    printed <- paste(capture.output(print(printModels$arima)), collapse="\n");
    expect_match(printed, "SARIMA(1,1,1)[1](0,1,1)[12]", fixed=TRUE);
    for(value in round(unlist(printModels$arima$arma), 4)){
        expect_match(printed, format(value), fixed=TRUE);
    }
    expect_match(paste(capture.output(print(printModels$xreg)), collapse="\n"),
                 "ETSX(ANN)", fixed=TRUE);
    printed <- paste(capture.output(print(printModels$occurrence)), collapse="\n");
    expect_match(printed, "iETS(MNN)[F]", fixed=TRUE);
    expect_match(printed, "Occurrence model type: Fixed probability", fixed=TRUE);
    expect_match(printed, "Mixture of Bernoulli and Gamma", fixed=TRUE);
})

test_that("print() of a combination of adam() models shows the combination", {
    testModel <- printModels$combined;
    printed <- paste(capture.output(print(testModel)), collapse="\n");
    expect_match(printed, "Model estimated: ETS(CCN)", fixed=TRUE);
    expect_match(printed, paste0("Number of models combined: ", length(testModel$models)), fixed=TRUE);
    expect_match(printed, "Sample size: 140", fixed=TRUE);
    expect_match(printed, "Forecast errors:", fixed=TRUE);
})

test_that("summary() of adam() prints the estimates and the scale", {
    testModel <- printModels$ets;
    printed <- paste(capture.output(print(summary(testModel))), collapse="\n");
    expect_match(printed, "ETS(AAdN)", fixed=TRUE);
    for(value in round(coef(testModel)[c("alpha","beta","phi")], 4)){
        expect_match(printed, format(value), fixed=TRUE);
    }
    expect_match(printed, paste0("Error standard deviation: ", round(sigma(testModel), 4)), fixed=TRUE);
})

test_that("plot() of adam() draws each of its plots", {
    for(modelName in c("ets", "arima", "xreg", "occurrence")){
        for(i in 1:16){
            pdf(NULL);
            dev.control("enable");
            plot(printModels[[modelName]], which=i);
            expect_gt(length(recordPlot()[[1]]), 0, label=paste(modelName, "plot", i));
            dev.off();
        }
    }
    pdf(NULL);
    dev.control("enable");
    plot(printModels$ets, which=c(1,2,3,7), legend=TRUE, lowess=FALSE, level=0.9);
    expect_gt(length(recordPlot()[[1]]), 0);
    dev.off();
    # A combination has no states of its own
    for(i in setdiff(1:16, c(6,12))){
        pdf(NULL);
        dev.control("enable");
        plot(printModels$combined, which=i);
        expect_gt(length(recordPlot()[[1]]), 0, label=paste("combined plot", i));
        dev.off();
    }
    pdf(NULL);
    expect_message(plot(printModels$combined, which=12), "nothing to plot");
    # The Q-Q plot of a combination said nothing and drew nothing
    expect_message(plot(printModels$combined, which=6), "nothing to plot");
    dev.off();
})

test_that("forecast() of a combination is the forecast of its models weighted by IC", {
    testModel <- printModels$combined;
    weights <- testModel$ICw;
    weights[weights<1e-2] <- 0;
    weights <- weights / sum(weights);
    testForecast <- forecast(testModel, h=5, interval="prediction");
    expect_equal(as.vector(testForecast$mean),
                 Reduce(`+`, Map(function(model, weight){
                     return(as.vector(forecast(model, h=5)$mean)*weight);
                 }, testModel$models, weights)));
    expect_true(all(testForecast$lower < testForecast$mean & testForecast$mean < testForecast$upper));
    # The forecast of the combination of the occurrence models
    testModel <- om(yIntermittent, occurrence="odds-ratio", model="CCN");
    weights <- testModel$ICw;
    weights[weights<1e-2] <- 0;
    weights <- weights / sum(weights);
    expect_equal(as.vector(forecast(testModel, h=5)$mean),
                 Reduce(`+`, Map(function(model, weight){
                     return(as.vector(forecast(model, h=5)$mean)*weight);
                 }, testModel$models, weights)));
})

test_that("print() and plot() of the forecasts of adam()", {
    testForecast <- forecast(printModels$ets, h=5, interval="prediction");
    printed <- capture.output(print(testForecast));
    expect_true(any(grepl("Point forecast +Lower bound \\(2.5%\\) +Upper bound \\(97.5%\\)", printed)));
    expect_true(any(grepl(format(round(testForecast$mean[1], 4), nsmall=4), printed, fixed=TRUE)));
    printed <- capture.output(print(forecast(printModels$ets, h=2, interval="prediction",
                                             side="upper", level=0.9)));
    expect_true(any(grepl("Point forecast +Upper bound \\(90%\\)", printed)));
    printed <- capture.output(print(forecast(printModels$ets, h=2, interval="prediction",
                                             side="lower", level=c(0.8,0.9))));
    expect_true(any(grepl("Lower bound \\(20%\\) +Lower bound \\(10%\\)", printed)));
    printed <- capture.output(print(forecast(printModels$ets, h=2)));
    expect_false(any(grepl("bound", printed)));
    for(modelName in c("ets", "occurrence", "combined")){
        pdf(NULL);
        dev.control("enable");
        plot(forecast(printModels[[modelName]], h=10, interval="prediction"));
        expect_gt(length(recordPlot()[[1]]), 0, label=paste(modelName, "forecast plot"));
        dev.off();
    }
})

test_that("print() of the forecast of msdecompose() shows its intervals", {
    testForecast <- forecast(msdecompose(AirPassengers, lags=12), h=3, interval="parametric");
    printed <- capture.output(print(testForecast));
    expect_true(any(grepl("Point forecast +Lower bound \\(2.5%\\) +Upper bound \\(97.5%\\)", printed)));
    expect_true(any(grepl(format(round(testForecast$upper[1], 4), nsmall=4), printed, fixed=TRUE)));
})

test_that("reapply() of adam() and of a combination, and their plots", {
    set.seed(1);
    testReapply <- reapply(printModels$combined, nsim=20);
    expect_s3_class(testReapply, "reapplyCombined");
    # The models with a weight of at least 0.01, with their weights renormalised
    weights <- printModels$combined$ICw[printModels$combined$ICw>=1e-2];
    expect_equal(testReapply$ICw, weights / sum(weights));
    expect_equal(names(testReapply$refitted), names(weights));
    for(refitted in testReapply$refitted){
        expect_equal(dim(refitted), c(nobs(printModels$combined), 20));
    }
    pdf(NULL);
    dev.control("enable");
    plot(testReapply);
    expect_gt(length(recordPlot()[[1]]), 0);
    dev.off();
    testReapply <- reapply(printModels$ets, nsim=20);
    expect_equal(dim(testReapply$refitted), c(nobs(printModels$ets), 20));
    pdf(NULL);
    dev.control("enable");
    plot(testReapply);
    expect_gt(length(recordPlot()[[1]]), 0);
    dev.off();
})

#### The sample-size check: the models that fit on small samples ####
test_that("ETS with ARIMA on a small sample drops the ARIMA", {
    expect_warning(testModel <- adam(AirPassengers[1:10], "AAN", orders=list(ar=2,i=1,ma=2),
                                     initial="optimal"),
                   "enough observations to fit ETS with ARIMA");
    expect_equal(testModel$model, "ETS(AAN)");
})

test_that("ARIMA on a small sample switches to backcasting or to ARIMA(0,1,1)", {
    expect_warning(testModel <- adam(AirPassengers[1:7], "NNN", orders=list(ar=2,i=1,ma=2),
                                     initial="optimal"),
                   "Switching initial to 'backcasting'");
    expect_equal(testModel$initialType, "complete");
    expect_warning(testModel <- adam(AirPassengers[1:5], "NNN", orders=list(ar=3,i=2,ma=3),
                                     initial="optimal"),
                   "Switching to ARIMA\\(0,1,1\\)");
    expect_equal(orders(testModel), list(ar=0, i=1, ma=1));
})

test_that("a small sample restricts the pool to the models it can fit", {
    expect_warning(testModel <- adam(AirPassengers[1:6], "ZZZ", initial="optimal"),
                   "Fitting what I can");
    # Six observations fit the level and the undamped trend models, not the damped ones
    expect_true(all(nchar(names(testModel$ICs))==3));
    expect_warning(testModel <- adam(AirPassengers[1:6], c("ANN","AAN","AAdN","MMdN"),
                                     initial="optimal"),
                   "Fitting what I can");
    expect_setequal(names(testModel$ICs), c("ANN","AAN"));
})

test_that("the full, the pure and the safe pools take the other letters of the model", {
    expect_equal(adam_modelsPoolNamed("F", "F", "N", "N"), c("ANN","MNN"));
    expect_equal(adam_modelsPoolNamed("P", "P", "A", "N"), c("AAN","AAdN"));
    expect_equal(adam_modelsPoolNamed("F", "F", "X", "N"), c("ANN","AAN","AAdN","MNN","MAN","MAdN"));
    expect_equal(adam_modelsPoolNamed("S", "S", "Z", "N"), c("ANN","AAN","AAdN","MNN","MAN","MAdN","MMN","MMdN"));
    expect_equal(adam_checkModel(FALSE, "PYN", "estimate")$modelsPool, c("ANN","MNN","MMN","MMdN"));
    # The pool given as a vector, its strange models, and the combination of the rest
    expect_error(adam_modelsPoolCheck(c("ANN","AQN")), "strange model");
    expect_equal(adam_modelsPoolCheck(c("ANN","AAN","CCN")), list(model="CCN", modelsPool=c("ANN","AAN")));
    expect_equal(adam_modelsPoolCheck(c("ANN","AAN","CCC"))$model, "CCC");
    expect_equal(adam_modelsPoolCheck(c("ANN","MNN"))$model, "ZNN");
    expect_message(expect_equal(adam_modelComponents("AAxN")$model, "AAdN"), "strange model");
    expect_equal(suppressMessages(adam_modelComponents("AN"))[c("model","damped")],
                 list(model="ZZZ", damped=TRUE));
    # The non-positive data: the additive pool, the additive selection, or a warning
    expect_warning(expect_equal(adam_modelsAdditive("SSS", "Z", "Z", "N", c("ANN","MNN"))$modelsPool, "ANN"),
                   "Amending the pool");
    expect_warning(expect_null(adam_modelsAdditive("ZZZ", "Z", "Z", "Z", NULL)$modelsPool),
                   "Changing the selection mechanism");
    expect_warning(adam_modelsAdditive("MNN", "M", "N", "N", NULL), "might break down");
})

test_that("the helpers of the small samples build, filter and simplify as the sample allows", {
    # Six observations: a trend, but no damping and no seasonality
    expect_equal(adam_samplePoolBuild(6, 12, "Z", "Z", TRUE, 0), c("ANN","MNN","AAN","MAN","AMN","MMN"));
    expect_equal(adam_samplePoolBuild(30, 12, "Z", "Z", FALSE, 0), c("ANN","AAN","AAdN","ANA","AAA"));
    # MAA has an additive seasonality: in the pool for "X", not for "Y"
    expect_true("MAA" %in% adam_samplePoolBuild(30, 12, "Z", "X", TRUE, 0));
    expect_false("MAA" %in% adam_samplePoolBuild(30, 12, "Z", "Y", TRUE, 0));
    # Eight observations with lag 4: the trend with the seasonality needs ten, the damped one eleven
    expect_equal(adam_samplePoolFilter(c("ANN","AAN","AAdN","ANA","AAA","AAdA","ANN"), 8, 4, 0),
                 c("ANN","AAN","AAdN","ANA"));
    caught <- capture_warnings(model <- adam_sampleModelSimplify("AAdA", 8, 4, 0));
    expect_equal(model, "ANA");
    expect_length(caught, 2);
    expect_equal(suppressWarnings(adam_sampleModelSimplify("MAA", 3, 4, 0)), "MNN");
    expect_equal(suppressWarnings(adam_sampleModelSimplify("AAdA", 0, 12, 0)), "ANN");
    # The tiny samples: a level model, estimated, selected, or used with the initial preset
    expect_warning(adam_sampleTiny(3, 3, "Z", FALSE, c(1,2,3), rep(TRUE,3), matrix(0,2,1)),
                   "persistence value was set to zero");
    tiny <- suppressWarnings(adam_sampleTiny(2, 3, "M", TRUE, c(4,0,6), c(TRUE,FALSE,TRUE), matrix(0,2,1)));
    expect_equal(tiny[c("model","modelDo","initialLevel","initialType")],
                 list(model="ANN", modelDo="use", initialLevel=5, initialType="provided"));
    expect_equal(tiny$parametersNumber[,1], c(0,2));
    tiny <- suppressWarnings(adam_sampleTiny(0, 3, "A", TRUE, c(0,0,0), rep(FALSE,3), matrix(0,2,1)));
    expect_equal(tiny[c("initialLevel","occurrence")], list(initialLevel=0, occurrence="none"));
    expect_equal(adam_sampleTiny(4, 4, "A", TRUE, 1:4, rep(TRUE,4), matrix(0,2,1))$model, "ANN");
    expect_error(adam_sampleTiny(5, 5, "A", TRUE, 1:5, rep(TRUE,5), matrix(0,2,1)), "Even for fitting");
})

# The provided ARMA parameters: a named list, a list without names and a vector
test_that("The provided ARMA parameters are read lag by lag", {
    # A named list of a wrong length switches that part to estimation
    expect_warning(arma <- adam_armaParameters(list(ar=0.5, ma=0.2), c(1,1), c(0,1), c(1,4), TRUE, TRUE),
                   "AR parameters is 1 while I need 2");
    expect_equal(arma, list(armaParameters=c("theta1[4]"=0.2), arEstimate=TRUE, maEstimate=FALSE));
    # A list without names holds the AR, then the MA, in every lag
    expect_equal(adam_armaParameters(list(0.5, 0.2), c(1,0), c(0,1), c(1,4), TRUE, TRUE)$armaParameters,
                 c("phi1[1]"=0.5, "theta1[4]"=0.2));
    # A vector goes lag by lag, and nothing is returned without the AR and the MA
    expect_equal(adam_armaParameters(c(0.1,0.2,0.3), c(1,1), c(1,0), c(1,4), TRUE, TRUE)$armaParameters,
                 c("phi1[1]"=0.1, "theta1[1]"=0.2, "phi1[4]"=0.3));
    expect_null(adam_armaParameters(NULL, 0, 0, 1, FALSE, FALSE)$armaParameters);
})

test_that("an estimated model loses its damping and then its trend as the sample shrinks", {
    # Only the steps that apply to the model warn
    caught <- capture_warnings(testModel <- adam(AirPassengers[1:6], "AAdN", initial="optimal"));
    expect_equal(modelType(testModel), "AAN");
    expect_length(grep("Fitting what I can", caught), 1);
    caught <- capture_warnings(testModel <- adam(AirPassengers[1:5], "AAdN", initial="optimal"));
    expect_equal(modelType(testModel), "ANN");
    expect_length(grep("Fitting what I can", caught), 2);
})

test_that("a seasonal model on less than two seasons drops the trend and uses LOWESS", {
    caught <- capture_warnings(testModel <- adam(AirPassengers[1:14], "AAA", lags=12,
                                                    initial="optimal"));
    expect_equal(modelType(testModel), "ANA");
    expect_true(any(grepl("Switching smoother to LOWESS", caught)));
})

test_that("four observations with a regressor fit and name the level model", {
    expect_no_warning(testModel <- adam(data.frame(y=AirPassengers[1:4], x=c(1,3,2,5)), "AAN",
                                        initial="optimal"));
    expect_equal(modelType(testModel), "ANN");
    expect_equal(testModel$model, "ETSX(ANN)");
    expect_warning(testModel <- adam(data.frame(y=AirPassengers[1:4], x=c(1,3,2,5)), "ZZN",
                                     initial="optimal"),
                   "only available model is level model");
    expect_true(modelType(testModel) %in% c("ANN","MNN"));
})

test_that("three observations or fewer fix the persistence and then the initial", {
    expect_warning(testModel <- adam(AirPassengers[1:3], "ANN", initial="optimal"),
                   "persistence value was set to zero");
    expect_equal(unname(testModel$persistence), 0);
    expect_warning(testModel <- adam(AirPassengers[1:2], "ANN"),
                   "initial was preset");
    expect_equal(testModel$initialType, "provided");
    expect_equal(unname(testModel$initial$level), mean(AirPassengers[1:2]));
    # With an occurrence, the level of the demand sizes: the mean of the non-zero values
    expect_warning(testModel <- adam(c(0,0,5,0,7,0,0,0), "ANN", occurrence="fixed"),
                   "initial was preset");
    expect_equal(unname(testModel$initial$level), 6);
    expect_warning(testModel <- adam(AirPassengers[1], "ANN"), "used Naive");
    expect_equal(unname(testModel$initial$level), AirPassengers[1]);
    # A single zero is the Naive of zero
    expect_warning(testModel <- adam(0, "ANN"), "used Naive");
    expect_equal(unname(testModel$initial$level), 0);
})

#### The regressors: the missing values, the provided initials and the occurrence ####
set.seed(41);
xregX1 <- rnorm(60, 10, 2);
xregX2 <- rnorm(60, 5, 1);
xregData <- data.frame(y=50 + 2*xregX1 - 3*xregX2 + cumsum(rnorm(60)), x1=xregX1, x2=xregX2);

test_that("a missing value of a regressor is the first observed value, whatever the way", {
    xregMissing <- xregFilled <- xregData;
    xregMissing$x1[5] <- NA;
    xregFilled$x1[5] <- xregData$x1[1];
    for(arguments in list(list(model="ANN"), list(model="ANN", regressors="select"),
                          list(model="ANN", regressors="adapt"), list(model="NNN"),
                          list(model="ANN", initial=list(xreg=c(x1=2, x2=-3))))){
        label <- paste(unlist(arguments), collapse=" ");
        expect_warning(testModel <- do.call(adam, c(list(xregMissing), arguments)),
                       "Some variables contained NAs", label=label);
        expect_equal(logLik(testModel), logLik(do.call(adam, c(list(xregFilled), arguments))),
                     label=label);
    }
    # A matrix as a data frame
    expect_warning(testModel <- adam(as.matrix(xregMissing), "ANN"), "Some variables contained NAs");
    expect_equal(logLik(testModel), logLik(adam(xregFilled, "ANN")));
})

test_that("the initials provided for some regressors are kept, the others estimated", {
    testModel <- adam(xregData, "ANN", initial=list(xreg=c(x1=2)));
    expect_equal(unname(testModel$initial$xreg["x1"]), 2);
    expect_false(testModel$initial$xreg["x2"]==0);
    expect_equal(testModel$nParam[2,2], 1);
    expect_equal(nparam(testModel), nparam(adam(xregData, "ANN"))-1);
    # Unnamed values are in the order of the variables
    expect_equal(testModel$initial$xreg, adam(xregData, "ANN", initial=list(xreg=2))$initial$xreg);
    # All of them provided
    testModel <- adam(xregData, "ANN", initial=list(xreg=c(x1=2, x2=-3)));
    expect_equal(unname(testModel$initial$xreg), c(2, -3));
    expect_equal(testModel$nParam[2,2], 2);
})

test_that("a regression with an occurrence model fits the sizes to the non-zero values", {
    set.seed(2);
    xregIntermittent <- transform(xregData, y=y*rbinom(60, 1, 0.7));
    nonZero <- xregIntermittent$y!=0;
    testModel <- adam(xregIntermittent, "NNN", occurrence="fixed");
    sizes <- greybox::alm(y~x1+x2, data=xregIntermittent[nonZero,]);
    occurrenceModel <- om(xregIntermittent$y, occurrence="fixed");
    expect_equal(coef(testModel), coef(sizes));
    expect_equal(as.numeric(logLik(testModel)), as.numeric(logLik(sizes)+logLik(occurrenceModel)));
    expect_equal(testModel$nParam[1,3], 1);
    # The fitted values are the sizes times the probability, the residuals are zero at the zeros
    expect_equal(as.vector(fitted(testModel))[nonZero], fitted(sizes)*mean(nonZero),
                 ignore_attr=TRUE);
    expect_true(all(residuals(testModel)[!nonZero]==0));
    # The forecast: the sizes times the probability, and the interval of the sizes with the
    # level (level-(1-p))/p, as forecast.adam()
    testForecast <- forecast(testModel, h=3, newdata=xregIntermittent[1:3,], interval="prediction");
    sizesForecast <- forecast(sizes, h=3, newdata=xregIntermittent[1:3,], interval="prediction",
                              level=(0.95-(1-mean(nonZero)))/mean(nonZero));
    expect_equal(as.vector(testForecast$mean), as.vector(sizesForecast$mean)*mean(nonZero));
    expect_equal(as.vector(testForecast$upper), as.vector(sizesForecast$upper));
    # The confidence interval of the sizes times the probability, for one and several levels
    sizesForecast <- forecast(sizes, h=3, newdata=xregIntermittent[1:3,], interval="confidence",
                              level=c(0.8,0.95));
    testForecast <- forecast(testModel, h=3, newdata=xregIntermittent[1:3,], interval="confidence",
                             level=c(0.8,0.95));
    expect_equal(as.vector(testForecast$lower), as.vector(sizesForecast$lower)*mean(nonZero));
    expect_equal(as.vector(testForecast$upper), as.vector(sizesForecast$upper)*mean(nonZero));
    testForecast <- forecast(testModel, h=3, newdata=xregIntermittent[1:3,], interval="confidence");
    expect_equal(as.vector(testForecast$upper), as.vector(sizesForecast$upper[,2])*mean(nonZero));
    # The same with the selection of the regressors and with a fitted om()
    expect_equal(logLik(adam(xregIntermittent, "NNN", occurrence="fixed", regressors="select")),
                 logLik(testModel));
    expect_equal(logLik(adam(xregIntermittent, "NNN", occurrence=occurrenceModel)), logLik(testModel));
})

test_that("the regressors that cannot be used are dealt with", {
    expect_warning(adam(xregData, "ANN", regressors="select", initial=list(xreg=c(x1=2, x2=-3))),
                   "Variables selection does not work with the provided initials");
    expect_warning(adam(xregData, "ANN", regressors="select", persistence=list(xreg=c(0.1,0.1))),
                   "cannot do variables selection with the provided smoothing parameters");
    expect_warning(testModel <- adam(xregData, "ANN", regressors="adapt", persistence=list(xreg=c(0.1,0.1,0.1))),
                   "length of the provided persistence for the xreg variables is wrong");
    expect_equal(testModel$model, "ETSX(ANN){D}");
    expect_warning(testModel <- adam(data.frame(y=xregData$y, x=1), "ANN"),
                   "did not have any variability");
    expect_equal(testModel$model, "ETS(ANN)");
    # A character variable is a factor, with all its levels
    xregCharacter <- data.frame(xregData, z=rep(c("a","b","c"), 20));
    expect_warning(testModel <- adam(xregCharacter, "ANN"), "character variables");
    expect_equal(names(testModel$initial$xreg), c("x1","x2","za","zb","zc"));
    xregFactor <- transform(xregCharacter, z=factor(z));
    expect_equal(logLik(testModel), logLik(adam(xregFactor, "ANN")));
    expect_equal(logLik(suppressWarnings(adam(xregCharacter, "ANN", regressors="select"))),
                 logLik(adam(xregFactor, "ANN", regressors="select")));
    # Special characters in the names of the variables
    testModel <- adam(setNames(xregData, c("y","x 1","x-2")), "ANN");
    expect_equal(names(testModel$initial$xreg), c("x.1","x.2"));
})

test_that("the selection of the regressors expands the factors with or without a formula", {
    set.seed(41);
    xregZ <- rep(c("a","b","c"), 20);
    xregFactor <- data.frame(y=50 + 2*xregData$x1 + c(a=0,b=5,c=-4)[xregZ] + cumsum(rnorm(60)),
                             x1=xregData$x1, x2=rnorm(60), z=factor(xregZ));
    testModel <- adam(xregFactor, "ANN", regressors="select");
    expect_true(all(c("x1","za","zb","zc") %in% names(testModel$initial$xreg)));
    expect_equal(logLik(testModel),
                 logLik(adam(xregFactor, "ANN", regressors="select", formula=y~x1+x2+z)));
})


#### The damping parameter ####
test_that("phi is used, estimated or one depending on the model", {
    # Provided for a damped trend: used, and counted as provided
    testModel <- adam(BJsales, "AAdN", phi=0.9);
    expect_equal(testModel$phi, 0.9);
    expect_false("phi" %in% names(coef(testModel)));
    expect_equal(testModel$nParam["Provided","nParamInternal"], 1);
    # Not provided: estimated for a damped trend, one otherwise and for ARIMA
    testModel <- adam(BJsales, "AAdN");
    expect_true("phi" %in% names(coef(testModel)));
    expect_equal(testModel$phi, coef(testModel)[["phi"]]);
    expect_equal(adam(BJsales, "AAN")$phi, 1);
    expect_equal(adam(BJsales, "NNN", orders=list(ar=1,i=1), phi=0.9)$phi, 1);
})

test_that("phi of a model without the damped trend is ignored", {
    for(phi in list(0.9, "a")){
        expect_warning(testModel <- adam(BJsales, "AAN", phi=phi), "phi is used only with a damped trend");
        expect_equal(testModel$phi, 1);
        expect_equal(logLik(testModel), logLik(adam(BJsales, "AAN")));
    }
    # The phi of an undamped model, as when it is reused, is ignored silently
    expect_no_warning(adam(BJsales, "AAN", phi=1));
    expect_no_warning(adam(BJsales, model=adam(BJsales, "AAN")));
})

test_that("phi that cannot be used is estimated with a warning", {
    expect_warning(testModel <- adam(BJsales, "ZZN", phi=0.9),
                   "Predefined phi can only be used with preselected ETS model");
    expect_s3_class(testModel, "adam");
    expect_warning(testModel <- adam(BJsales, "AAdN", phi="a"), "Provided value of phi is meaningless");
    expect_true("phi" %in% names(coef(testModel)));
    for(phi in c(-0.1, 3)){
        expect_warning(testModel <- adam(BJsales, "AAdN", phi=phi), "Damping parameter should lie in \\(0, 2\\)");
        expect_true("phi" %in% names(coef(testModel)), label=phi);
        expect_true(testModel$phi>=0 && testModel$phi<=1, label=phi);
    }
})

#### Methods of the smooth class ####
test_that("modelLags() returns the lags of the states", {
    testModel <- adam(BJsales, "AAN");
    expect_equal(modelLags(testModel), testModel$lagsAll);
    expect_equal(as.vector(modelLags(testModel)), c(1,1));
    expect_equal(as.vector(modelLags(adam(AirPassengers, "MAM", lags=12))), c(1,1,12));
    # The objects of the old msarima() kept them in modelLags
    testModel <- structure(list(modelLags=matrix(c(1,1,12), ncol=1)), class=c("msarima","smooth"));
    expect_equal(modelLags(testModel), matrix(c(1,1,12), ncol=1));
})

test_that("pointLik() of msdecompose() is the density of its residuals", {
    testModel <- msdecompose(AirPassengers, lags=12, type="multiplicative");
    yValues <- as.vector(actuals(testModel));
    errors <- as.vector(residuals(testModel));
    # The residuals of the multiplicative decomposition are in logs: the log-normal density
    expect_s3_class(pointLik(testModel), "ts");
    expect_equal(tsp(pointLik(testModel)), tsp(AirPassengers));
    expect_equal(as.vector(pointLik(testModel)),
                 dlnorm(yValues, log(as.vector(fitted(testModel))), sqrt(mean(errors^2)), log=TRUE));
    expect_equal(as.vector(pointLik(testModel, log=FALSE)),
                 dlnorm(yValues, log(as.vector(fitted(testModel))), sqrt(mean(errors^2))));
    testModel <- msdecompose(AirPassengers, lags=12, type="additive");
    errors <- as.vector(residuals(testModel));
    expect_equal(as.vector(pointLik(testModel)), dnorm(errors, 0, sqrt(mean(errors^2)), log=TRUE));
    expect_equal(as.vector(pointLik(testModel, log=FALSE)), dnorm(errors, 0, sqrt(mean(errors^2))));
    # The Laplace density for the MAE losses and the S one for HAM
    testModel$loss <- "MAE";
    expect_equal(as.vector(pointLik(testModel)), dlaplace(errors, 0, mean(abs(errors)), log=TRUE));
    testModel$loss <- "HAM";
    expect_equal(as.vector(pointLik(testModel)), ds(errors, 0, mean(sqrt(abs(errors)))/2, log=TRUE));
})

test_that("plot() of the forecast of msdecompose() draws it with and without the interval", {
    testModel <- msdecompose(AirPassengers, lags=12, type="multiplicative");
    plotLength <- c(parametric=0, none=0);
    for(interval in names(plotLength)){
        testForecast <- forecast(testModel, h=6, interval=interval);
        expect_s3_class(testForecast, "smooth.forecast");
        pdf(NULL);
        dev.control("enable");
        plot(testForecast);
        plotLength[interval] <- length(recordPlot()[[1]]);
        dev.off();
    }
    expect_gt(plotLength[["none"]], 0);
    # The interval adds its bounds to the plot
    expect_gt(plotLength[["parametric"]], plotLength[["none"]]);
})

test_that("accuracy() of adam() and of its forecasts", {
    holdout <- BJsales[141:150];
    testModel <- adam(BJsales, "AAN", h=10, holdout=TRUE);
    expect_equal(accuracy(testModel), testModel$accuracy);
    expect_equal(accuracy(testModel)[["ME"]], mean(holdout - testModel$forecast));
    expect_equal(accuracy(forecast(testModel, h=10)), testModel$accuracy);
    # Without the holdout: the in-sample measures, or those on the provided holdout
    testModel <- adam(BJsales[1:140], "AAN");
    expect_equal(accuracy(testModel), measures(actuals(testModel), fitted(testModel), actuals(testModel)));
    expect_equal(accuracy(testModel)[["ME"]], mean(residuals(testModel)));
    expect_equal(accuracy(testModel, holdout=holdout),
                 measures(holdout, forecast(testModel, h=10)$mean, actuals(testModel)));
    testForecast <- forecast(testModel, h=10);
    expect_equal(accuracy(testForecast), accuracy(testModel));
    expect_equal(accuracy(testForecast, holdout=holdout), accuracy(testModel, holdout=holdout));
})

test_that("plot() of predict() of adam() draws the fitted values and their interval", {
    testModel <- adam(BJsales, "AAN");
    testPredict <- predict(testModel);
    expect_s3_class(testPredict, "adam.predict");
    pdf(NULL);
    dev.control("enable");
    plot(testPredict);
    plotLength <- length(recordPlot()[[1]]);
    expect_gt(plotLength, 0);
    # The interval spans the y-axis, unless the limits are provided
    testPredict <- predict(testModel, interval="prediction");
    plot(testPredict);
    expect_gt(length(recordPlot()[[1]]), plotLength);
    yRange <- range(c(actuals(testModel), testPredict$mean, testPredict$lower, testPredict$upper));
    expect_equal(par("usr")[3:4], yRange + c(-1,1)*0.04*diff(yRange));
    plot(testPredict, ylim=c(150,300));
    expect_equal(par("usr")[3:4], c(150,300) + c(-1,1)*0.04*150);
    dev.off();
})

test_that("The gap variances of a model without the filter come from its matrices", {
    yGaps <- BJsales;
    yGaps[c(20,21,50)] <- NA;
    expect_warning(testModel <- adam(yGaps, "ANN", persistence=0.3), "Data contains NAs");
    # As in the models fitted without the filter (cma()): the error after j-1 missing values
    # sums j one-step errors, with the variance 1+(j-1)*alpha^2 and the mean multiplier 1+(j-1)*alpha
    testModel$adamCpp <- NULL;
    gapVariance <- gapMean <- rep(1, 150);
    gapVariance[c(22,51)] <- 1 + c(2,1)*0.3^2;
    gapMean[c(22,51)] <- 1 + c(2,1)*0.3;
    expect_equal(adam_gapVarianceModel(testModel), gapVariance);
    expect_equal(adam_gapVarianceModel(testModel, 1), gapMean);
    expect_equal(as.vector(adam_gapResiduals(testModel)), as.vector(residuals(testModel))/sqrt(gapVariance));
    expect_equal(as.vector(pointLik(testModel))[c(19,22,51)],
                 dnorm(BJsales[c(19,22,51)], as.vector(fitted(testModel))[c(19,22,51)],
                       sqrt(testModel$scale*gapVariance[c(19,22,51)]), log=TRUE));
})

test_that("extractSigma() of adam() is sigma() or the one implied by the scale model", {
    testModel <- adam(BJsales, "ANN");
    expect_equal(extractSigma(testModel), sigma(testModel));
    # The scale of the normal distribution is the variance, the one of Laplace is sigma/sqrt(2)
    expect_warning(scaleModel <- implant(testModel, sm(testModel)), "observations with zero errors");
    expect_length(extractSigma(scaleModel), nobs(testModel));
    expect_equal(extractSigma(scaleModel), sqrt(extractScale(scaleModel)));
    testModel <- adam(BJsales, "ANN", distribution="dlaplace");
    expect_warning(scaleModel <- implant(testModel, sm(testModel)), "observations with zero errors");
    expect_equal(extractSigma(scaleModel), sqrt(2)*extractScale(scaleModel));
})

#### The start-up message ####
test_that(".onAttach() shows the version of smooth and, at times, a hint", {
    # The seed that makes the random number of the message satisfy the condition
    findSeed <- function(condition){
        seed <- 1;
        set.seed(seed);
        while(!condition(sample(c(1:100), 1))){
            seed <- seed + 1;
            set.seed(seed);
        }
        return(seed);
    }
    startUpMessage <- paste0("This is package \"smooth\", v", packageVersion("smooth"), "\n");
    hints <- c("have you already tried adam() function", "you can visit my website",
               "you can read the online monograph", "File an issue on github");
    for(i in 1:4){
        set.seed(findSeed(function(value){return(value==i)}));
        printed <- tryCatch(smooth:::.onAttach("", "smooth"), message=function(m){return(conditionMessage(m))});
        expect_true(startsWith(printed, startUpMessage), label=i);
        expect_match(printed, hints[i], fixed=TRUE);
    }
    set.seed(findSeed(function(value){return(value>4)}));
    expect_equal(tryCatch(smooth:::.onAttach("", "smooth"), message=function(m){return(conditionMessage(m))}),
                 paste0(startUpMessage, "\n"));
})

# Every non-positive multiplicative trend and seasonal value of the states and of the profile
test_that("Non-positive multiplicative components are replaced", {
    matVt <- matrix(c(1,2,3,4,5, 1,-1,2,0,-3, 1,-1,1,1,2, 2,1,0,1,-1), 4, 5, byrow=TRUE);
    profile <- matrix(c(1,2, -1,2, 1,-2, 0,1), 4, 2, byrow=TRUE);
    corrected <- smooth:::adam_correctMultiplicativeComponents(matVt, profile, "M", "M", 2, 2);
    expected <- matVt;
    expected[-1,][expected[-1,]<=0] <- 1e-6;
    expect_equal(corrected$matVt, expected);
    expect_equal(corrected$profilesRecentTable, matrix(c(1,2, 1e-6,2, 1,1e-6, 1e-6,1), 4, 2, byrow=TRUE));
})

# A pool of seasonal models only, on a sample shorter than the lag: the lag is dropped, and the
# selection is over the non-seasonal models
test_that("A pool without non-seasonal models on a short sample", {
    testModel <- suppressWarnings(adam(as.numeric(AirPassengers)[1:8], c("MAM","ANA"), lags=12));
    expect_equal(testModel$model, "ETS(ANN)");
})
