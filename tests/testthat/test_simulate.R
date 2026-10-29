context("Tests for simulate() functions")

#### ETS ####
testData <- sim.es("MNN", frequency=12, bounds="a", obs=100)
test_that("ETS(MNN) simulated with admissible bounds", {
    expect_match(testData$model, "MNN")
})

testData <- sim.es("AAdM", frequency=12, phi=0.9, obs=120)
test_that("ETS(AAdM) simulated with phi=0.9", {
    expect_match(testData$model, "AAdM")
})

testData <- sim.es("MNN", frequency=12, obs=120, nsim=100, probability=0.2)
test_that("iETS(MNN) simulated with probability=0.2 and nsim=100", {
    expect_match(testData$model, "MNN")
})

testModel <- es(AirPassengers, "ANA", h=18, silent=TRUE)
test_that("ETS(ANA) simulated from estimated model", {
    expect_match(simulate(testModel,nsim=10,seed=5,obs=100)$model, "ANA")
})

#### SSARIMA ####
testModel <- auto.ssarima(BJsales, h=8, silent=TRUE)
test_that("ARIMA(0,1,3) with drift simulated from estimated model", {
    expect_match(errorType(simulate(testModel,nsim=10,seed=5,obs=100)), "A")
})

test_that("ARIMA(0,1,1) with intermittent data", {
    expect_match(sim.ssarima(nsim=10,obs=100,probability=0.2)$model, "iARIMA")
})

#### CES ####
testModel <- auto.ces(BJsales, h=8, silent=TRUE)
test_that("CES(n) simulated from estimated model", {
    expect_match(simulate(testModel,nsim=10,seed=5,obs=100)$model, "(n)")
})

test_that("CES(s) with some random parameters", {
    expect_match(sim.ces(seasonality="s",frequency=4,nsim=1,obs=100)$model, "(s)")
})

test_that("CES(p) with some random A parameter and fixed b=0.1 ", {
    expect_equal(sim.ces(seasonality="p",frequency=4,b=0.1,nsim=1,obs=100)$b[1], 0.1)
})

test_that("CES(f) with intermittent data", {
    expect_match(sim.ces(seasonality="f",frequency=12,nsim=10,obs=100,probability=0.2)$model, "iCES")
})

#### GUM ####
testModel <- gum(BJsales, orders=1, lags=1, h=8, silent=TRUE)
test_that("GUM(1[1]) simulated from estimated model", {
    expect_match(simulate(testModel,nsim=10,seed=5,obs=100)$model, "GUM")
})

test_that("GUM(1[1]) with intermittent data", {
    expect_match(sim.gum(nsim=10,obs=100,probability=0.2)$model, "iGUM")
})

#### SMA ####
test_that("SMA(10) with intermittent data", {
    expect_match(sim.sma(10,nsim=10,obs=100,probability=0.2)$model, "iSMA")
})

#### OM ####
test_that("simulate.om returns probability + binary data with right shape", {
    set.seed(42)
    y <- rbinom(60, 1, prob=0.3 + 0.005*(1:60))
    m <- suppressWarnings(
        om(y, model="MNN", occurrence="odds-ratio", silent=TRUE)
    )
    sim <- simulate(m, nsim=3, seed=42)

    expect_s3_class(sim, "om.sim")
    expect_s3_class(sim, "oes.sim")
    expect_equal(dim(sim$probability), c(60, 3))
    expect_equal(dim(sim$data),        c(60, 3))
    expect_true(all(sim$probability >= 0 & sim$probability <= 1))
    expect_true(all(sim$data %in% c(0, 1)))
    expect_equal(sim$occurrence, "odds-ratio")
})

test_that("simulate.om is seed-deterministic", {
    set.seed(1)
    y <- rbinom(40, 1, 0.4)
    m <- suppressWarnings(om(y, model="MNN", occurrence="odds-ratio",
                             silent=TRUE))
    a <- simulate(m, nsim=2, seed=7)
    b <- simulate(m, nsim=2, seed=7)
    expect_equal(unclass(a$probability), unclass(b$probability))
    expect_equal(unclass(a$data),        unclass(b$data))
})

test_that("simulate.om dispatches to print.oes.sim cleanly", {
    set.seed(3)
    y <- rbinom(40, 1, 0.4)
    m_om <- suppressWarnings(om(y, model="MNN", occurrence="direct",
                                silent=TRUE))
    expect_output(print(simulate(m_om, nsim=1, seed=0)),
                  "Data generated from:")
})

#### OMG ####
test_that("simulate.omg returns probability + binary data + sub-sims", {
    set.seed(2)
    y <- rbinom(80, 1, prob=0.3)
    m <- suppressWarnings(omg(y, modelA="MNN", modelB="MNN", silent=TRUE))
    sim <- simulate(m, nsim=2, seed=42)

    expect_s3_class(sim, "omg.sim")
    expect_s3_class(sim, "oes.sim")
    expect_equal(dim(sim$probability), c(80, 2))
    expect_equal(dim(sim$data),        c(80, 2))
    expect_true(all(sim$probability >= 0 & sim$probability <= 1))
    expect_true(all(sim$data %in% c(0, 1)))
    expect_s3_class(sim$modelA, "om.sim")
    expect_s3_class(sim$modelB, "om.sim")
    expect_equal(sim$occurrence, "general")
})

test_that("simulate.omg dispatches to print.oes.sim cleanly", {
    set.seed(3)
    y <- rbinom(40, 1, 0.4)
    m_omg <- suppressWarnings(omg(y, modelA="MNN", modelB="MNN",
                                  silent=TRUE))
    expect_output(print(simulate(m_omg, nsim=1, seed=0)),
                  "Data generated from:")
})

#### Regression: simulate.adam after the simulateADAMCore split ####
test_that("simulate.adam still works after the simulateADAMCore split", {
    set.seed(4)
    y <- ts(rnorm(60, 100, 5), frequency=12)
    m <- adam(y, model="ANN", silent=TRUE)
    sim <- simulate(m, nsim=2, seed=42)
    expect_s3_class(sim, "adam.sim")
    expect_equal(dim(sim$data), c(60, 2))
})

#### Head-refinement alignment between sim.es and es/adam fitters ####
# Before the head-refinement fix the simulator started from the raw initial
# level while the fitter walked (m-1) trend steps forward across the seasonal
# head. On ETS(A,A,A) with lags=12 this produced a systematic offset of
# (m-1) * trend on the first simulated observation. Guard against regression.
test_that("sim.es reproduces the fitted first observation for trended-seasonal ETS", {
    fit <- es(AirPassengers, model="AAA", h=0, silent=TRUE)
    initVec <- c(fit$initial$level, fit$initial$trend)
    initSeasVec <- fit$initial$seasonal
    simObj <- sim.es(model="AAA", obs=length(AirPassengers), nsim=1,
                     persistence=fit$persistence, phi=fit$phi,
                     initial=initVec, initialSeason=initSeasVec,
                     frequency=12,
                     randomizer="rnorm", mean=0, sd=1e-12)
    expect_equal(as.numeric(simObj$data)[1],
                 as.numeric(fitted(fit))[1],
                 tolerance=1e-6)
})

test_that("sim.es bounds each seasonal smoothing parameter by its own alpha", {
    set.seed(1)
    x <- sim.es("ANA", frequency=4, obs=20, nsim=200)
    expect_true(all(x$persistence[2,] <= 1-x$persistence[1,]))
})

test_that("sim.es gives each inverse Gaussian simulation its own likelihood", {
    set.seed(2)
    y <- sim.es("MNN", obs=30, nsim=3, randomizer="rinvgauss", mean=1, dispersion=0.1)
    e <- y$residuals
    d <- y$data
    expected <- sapply(1:3, function(i){
        return(-0.5*(30*(log(mean(e[,i]^2/(1+e[,i]))/(2*pi))-1) +
                         sum(log(d[,i]/(1+e[,i]))) - 3*sum(log(d[,i]))))
    })
    expect_equal(as.numeric(y$logLik), expected)
})

test_that("sim.ssarima() uses its constant and its generated initial states", {
    # The constant state was one whatever the constant, and the generated initials
    # were replaced by ones, which also set the scale of the errors
    set.seed(1);
    testData <- sim.ssarima(orders=list(ar=0,i=1,ma=0), obs=2000, constant=5);
    expect_equal(mean(diff(as.vector(testData$data))), 5, tolerance=0.05);
    # The errors are scaled by the square root of the first state, drawn in [0, 1000]
    errorVariances <- replicate(20, var(as.vector(sim.ssarima(orders=list(ar=2,i=0,ma=2),
                                                              obs=60)$residuals)));
    expect_gt(median(errorVariances), 10);
})

#### Printing and plotting ####
test_that("print() of the simulated data shows the model and its parameters", {
    set.seed(5);
    testData <- sim.es("AAdN", obs=50, phi=0.9);
    printed <- paste(capture.output(print(testData)), collapse="\n");
    expect_match(printed, "Data generated from: ETS(AAdN)", fixed=TRUE);
    expect_match(printed, "Number of generated series: 1", fixed=TRUE);
    expect_match(printed, paste(format(round(as.vector(testData$persistence), 4)), collapse=" +"));
    expect_match(printed, "Phi: 0.9", fixed=TRUE);
    expect_match(printed, paste0("True likelihood: ", round(testData$logLik, 4)), fixed=TRUE);
    expect_match(paste(capture.output(print(sim.es("MNN", obs=50, probability=0.5))), collapse="\n"),
                 "produced based on an occurrence model", fixed=TRUE);
    # ARIMA: the parameters by lag and the constant
    testData <- sim.ssarima(orders=list(ar=1,i=1,ma=1), obs=50, arma=list(ar=0.5, ma=-0.3),
                            constant=2);
    printed <- capture.output(print(testData));
    expect_true(any(grepl("ARIMA(1,1,1) with drift", printed, fixed=TRUE)));
    expect_true(any(grepl("^AR\\(1\\) +0.5$", printed)));
    expect_true(any(grepl("^MA\\(1\\) +-0.3$", printed)));
    expect_true(any(grepl("Constant value: 2", printed, fixed=TRUE)));
    testData <- sim.ces("n", obs=50);
    printed <- paste(capture.output(print(testData)), collapse="\n");
    expect_match(printed, "Data generated from: CES(none)", fixed=TRUE);
    expect_match(printed, paste0("Smoothing parameter a: ", round(testData$a, 4)), fixed=TRUE);
    expect_match(paste(capture.output(print(sim.gum(orders=c(2), lags=c(1), obs=50))), collapse="\n"),
                 "Data generated from: GUM(2[1])", fixed=TRUE);
    testData <- sim.sma(3, obs=50);
    printed <- paste(capture.output(print(testData)), collapse="\n");
    expect_match(printed, "Data generated from: SMA(3)", fixed=TRUE);
    expect_match(printed, paste0("True likelihood: ", round(testData$logLik, 4)), fixed=TRUE);
    expect_match(paste(capture.output(print(sim.es("ANN", obs=50, nsim=3))), collapse="\n"),
                 "Number of generated series: 3", fixed=TRUE);
    expect_match(paste(capture.output(print(sim.oes("MNN", obs=50))), collapse="\n"),
                 "Data generated from: oETS[O](MNN)", fixed=TRUE);
})

test_that("plot() of the simulated data draws the series", {
    set.seed(5);
    for(testData in list(sim.es("AAdN", obs=50, phi=0.9), sim.ssarima(obs=50), sim.ces("n", obs=50),
                         sim.gum(orders=c(2), lags=c(1), obs=50), sim.sma(3, obs=50),
                         sim.oes("MNN", obs=50))){
        pdf(NULL);
        dev.control("enable");
        plot(testData);
        expect_gt(length(recordPlot()[[1]]), 0, label=testData$model);
        dev.off();
    }
    pdf(NULL);
    dev.control("enable");
    expect_message(plot(sim.es("ANN", obs=50, nsim=3)), "Not sure which of them to plot");
    expect_gt(length(recordPlot()[[1]]), 0);
    dev.off();
})

# The simulations have no actuals: the lags take the frequency of the generated data, and
# the CES type is read from its name in words (it was "full" for every type but "none")
test_that("orders() and lags() of the ETS and CES simulations", {
    expect_equal(lags(sim.es("ANA", obs=40, frequency=4)), c(1, 4));
    expect_equal(lags(sim.es("ANN", obs=40, frequency=4)), 1);
    expected <- list(none=list(orders=2, lags=1), simple=list(orders=2, lags=4),
                     partial=list(orders=c(2,1), lags=c(1,4)), full=list(orders=c(2,2), lags=c(1,4)));
    for(type in names(expected)){
        testData <- sim.ces(type, obs=40, frequency=4);
        expect_equal(modelType(testData), type);
        expect_equal(orders(testData), expected[[type]]$orders);
        expect_equal(lags(testData), expected[[type]]$lags);
    }
})

# sim.oes() with several series: the probability has a column per series (it failed),
# and print() counts the observations from it (it printed 0)
test_that("sim.oes() generates several series and prints their size", {
    for(occurrence in c("odds-ratio","inverse-odds-ratio","direct","general")){
        testData <- sim.oes("MNN", obs=50, nsim=3, occurrence=occurrence, initial=0.5);
        expect_equal(dim(testData$probability), c(50, 3));
        expect_length(testData$logLik, 3);
    }
    expect_true(any(grepl("Number of generated series: 3", capture.output(print(testData)))));
    expect_true(any(grepl("Number of observations in each series: 50",
                          capture.output(print(sim.oes("MNN", obs=50, initial=0.5))))));
})

test_that("print() of a GUM simulation shows its matrices and likelihood", {
    testData <- sim.gum(orders=c(1,1), lags=c(1,4), obs=40);
    printed <- capture.output(print(testData));
    expect_true(all(c("Persistence vector: ", "Transition matrix: ", "Measurement vector: ") %in% printed));
    expect_true(any(grepl(paste0("True likelihood: ", round(testData$logLik, 4)), printed, fixed=TRUE)));
})
