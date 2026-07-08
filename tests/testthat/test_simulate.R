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
