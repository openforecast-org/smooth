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

test_that("gum() reports the likelihood at the fitted parameters, not the loss", {
    for(loss in c("MSE", "MAE")){
        testModel <- gum(AirPassengers, orders=c(1,1), lags=c(1,12), loss=loss);
        expect_equal(as.numeric(logLik(testModel)), sum(pointLik(testModel)));
        expect_true(as.numeric(logLik(testModel)) < 0);
    }
})

# The refit of vcov(type="hessian") keeps the estimated initials in B: the Hessian
# is the Fisher Information of the fit (with positive variances), not Inf for the initials
test_that("vcov(type='hessian') of gum() is the inverse FI of the fit", {
    m <- gum(AirPassengers, initial="optimal", FI=TRUE, stepSize=.Machine$double.eps^(1/4))
    v <- suppressWarnings(vcov(m, type="hessian", stepSize=.Machine$double.eps^(1/4)))
    expect_true(all(is.finite(diag(v))))
    expect_equal(diag(v), abs(diag(suppressWarnings(solve(m$FI)))), ignore_attr=TRUE)
})

# The model is fitted in logs: its cumulative values are the sums in the space of the data
test_that("the cumulative forecasts and the reforecasts of the multiplicative gum()", {
    testModel <- gum(AirPassengers, orders=c(1,1), lags=c(1,12), type="multiplicative", h=4, holdout=FALSE);
    expect_equal(as.vector(forecast(testModel, h=4, cumulative=TRUE)$mean),
                 sum(forecast(testModel, h=4)$mean));
    set.seed(41);
    testForecast <- forecast(testModel, h=4, interval="approximate", cumulative=TRUE, nsim=20000);
    expect_equal(testForecast$interval, "simulated");
    expect_true(testForecast$lower < sum(forecast(testModel, h=4)$mean) && testForecast$upper > testForecast$mean);
    expect_equal(as.vector(forecast(testModel, h=4, interval="simulated", cumulative=TRUE, side="upper",
                                  nsim=100)$lower), 0);
    set.seed(41);
    expect_equal(apply(reapply(testModel, nsim=100)$refitted, 1, median), as.vector(fitted(testModel)),
                 tolerance=0.05);
    set.seed(41);
    testReforecast <- reforecast(testModel, h=4, cumulative=TRUE, nsim=200);
    # With the uncertainty of the parameters, the bounds of the sums are wider
    expect_true(testReforecast$lower < testForecast$lower && testReforecast$upper > testForecast$upper);
    expect_equal(as.vector(testReforecast$lower), as.vector(testForecast$lower), tolerance=0.15);
    # The point likelihoods of the logarithms with the Jacobian, and the OPG covariance
    expect_equal(sum(pointLik(testModel)), as.numeric(logLik(testModel)));
    expect_no_warning(vcov(testModel));
    # The parameters drawn by reapply() enter the model
    expect_true(sd(reapply(testModel, nsim=10)$refitted[10,]) > 0);
})

# regressors="select" keeps the true regressor and drops the noise, as adam() does
test_that("gum() selects the regressors as adam()", {
    set.seed(3)
    xreg <- data.frame(y=0, a=rnorm(120), b=rnorm(120), c=rnorm(120))
    xreg$y <- 100 + 5*xreg$a + rnorm(120, 0, 2)
    xreg <- ts(xreg, frequency=12)
    testModel <- gum(xreg, regressors="select")
    expect_equal(names(testModel$initial$xreg), "a")
    expect_equal(names(testModel$initial$xreg), names(adam(xreg, "ANN", regressors="select")$initial$xreg))
})

# auto.gum() and the orders and lags of the simulated GUM
set.seed(41);
quarterly <- ts(100 + rnorm(48, 0, 3) + rep(c(5,-3,2,-4), 12), frequency=4);
testModel <- auto.gum(quarterly, orders=2, lags=4);
test_that("auto.gum() selects the GUM with the lowest information criterion", {
    # The candidates with one component of each lag; the seasonal one is the best
    candidates <- sapply(1:4, function(lag) AICc(gum(quarterly, orders=1, lags=lag)));
    expect_equal(testModel$model, "GUM(1[4])");
    expect_equal(AICc(testModel), min(candidates));
    expect_equal(AICc(testModel), AICc(gum(quarterly, orders=1, lags=4)));
})

test_that("auto.gum() checks its orders and the type of the model", {
    expect_error(auto.gum(quarterly, orders=0), "zero lags / orders");
    expect_warning(auto.gum(quarterly-200, orders=1, lags=2, type="multiplicative"),
                   "Switching to the additive one");
})

test_that("orders() and lags() of a simulated GUM are those of the simulation", {
    testData <- sim.gum(orders=c(1,1), lags=c(1,4), obs=40);
    expect_equal(testData$model, "GUM(1[1],1[4])");
    expect_equal(orders(testData), c(1,1));
    expect_equal(lags(testData), c(1,4));
})
