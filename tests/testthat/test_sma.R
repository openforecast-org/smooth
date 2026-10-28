context("Tests for sma()")

# sma() drives the C++ adamCore$fit() directly (not through adam()), so it is
# the one estimator whose C++ call signature is not exercised by the other
# model tests. A stale fit() call here previously slipped through the unit
# tests and only surfaced when the vignettes were rebuilt during R CMD check.
set.seed(41)
y <- ts(cumsum(rnorm(120, 0.1, 1)) + 100, frequency=12)

test_that("sma() with a specified order fits and forecasts", {
    m <- sma(y, order=12, h=8, holdout=TRUE, silent=TRUE)
    expect_match(m$model, "SMA\\(12\\)")
    expect_true(is.finite(logLik(m)))
    expect_equal(length(m$forecast), 8)
})

test_that("sma() selects the order automatically", {
    m <- sma(y, h=8, silent=TRUE)
    expect_match(m$model, "SMA\\([0-9]+\\)")
    expect_true(is.finite(AICc(m)))
})

test_that("sma() forecast() method works", {
    m <- sma(y, order=6, h=8, silent=TRUE)
    fc <- forecast(m, h=8, interval="parametric")
    expect_equal(length(fc$mean), 8)
    expect_true(all(fc$lower <= fc$upper))
})

# The missing values are gaps in the selection of the order and in the fit: the states are
# filtered over them, the likelihood is over the observed values, and the information
# criterion of the selected order is that of the model
test_that("sma() takes the missing values for gaps", {
    yNA <- y;
    yNA[c(30, 31, 60)] <- NA;
    m <- suppressWarnings(sma(yNA, h=8, holdout=TRUE, silent=TRUE));
    expect_equal(sum(pointLik(m)), as.numeric(logLik(m)));
    expect_equal(as.numeric(m$ICs[m$orders[1]]), as.numeric(AICc(m)));
})

test_that("orders() and lags() of a simulated SMA are those of the simulation", {
    testData <- sim.sma(order=3, obs=40);
    expect_equal(testData$model, "SMA(3)");
    expect_equal(orders(testData), 3);
    expect_equal(lags(testData), 1);
})

# cma() of an odd order is the centred moving average of the data
test_that("cma() of an odd order is the centred moving average", {
    passengers <- as.vector(AirPassengers);
    testModel <- cma(passengers, order=3);
    inner <- 2:(length(passengers)-1);
    expect_equal(as.vector(fitted(testModel))[inner],
                 as.vector(stats::filter(passengers, rep(1/3, 3), sides=2))[inner]);
    expect_error(cma(passengers[1:5], order=10), "enough observations");
    expect_error(cma(passengers, order=0), "positive number");
})

# smoothCombine() with the models provided
combinedModels <- list(es(BJsales, "AAN"), es(BJsales, "AAdN"), sma(BJsales));
testModel <- smoothCombine(BJsales, models=combinedModels, h=5, interval="prediction");
test_that("smoothCombine() weighs the forecasts of the models by their AICc", {
    ICs <- sapply(combinedModels, AICc);
    weights <- exp(-0.5*(ICs-min(ICs))) / sum(exp(-0.5*(ICs-min(ICs))));
    expect_equal(unname(testModel$ICw), weights);
    forecasts <- sapply(combinedModels, function(model) forecast(model, h=5)$mean);
    expect_equal(as.vector(testModel$forecast), as.vector(forecasts %*% weights));
    expect_true(all(testModel$lower < testModel$forecast & testModel$forecast < testModel$upper));
    expect_equal(extractSigma(testModel), sigma(testModel));
})

test_that("smoothCombine() reuses a combination", {
    reused <- smoothCombine(BJsales, models=testModel, h=5, interval="prediction");
    expect_equal(reused$forecast, testModel$forecast);
})

# The forecasts and plots of msdecompose()
testModel <- msdecompose(AirPassengers, lags=12, type="multiplicative");
test_that("forecast() of msdecompose() has the horizon and its interval", {
    testForecast <- forecast(testModel, h=12, level=0.9, model="MAN");
    expect_length(testForecast$mean, 12);
    expect_true(all(testForecast$lower < testForecast$mean & testForecast$mean < testForecast$upper));
    expect_equal(testForecast$level, 0.9);
})

test_that("plot() of msdecompose() draws all its plots", {
    # Each plot on a new device, which has nothing drawn before it
    for(i in 1:12){
        pdf(NULL);
        dev.control("enable");
        plot(testModel, which=i);
        expect_gt(length(recordPlot()[[1]]), 0);
        dev.off();
    }
})
