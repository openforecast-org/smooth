context("Tests for the backcasting head (headLength)")

y <- AirPassengers
m <- frequency(y)

# 1. headLength=0 reproduces the legacy zero-error head, the default filters it
test_that("headLength switches the head filtering on and off", {
    fDefault <- adam(y, model="AAdA", persistence=c(0.3,0.1,0.2), phi=0.9,
                     initial="backcasting", silent=TRUE)
    fLegacy  <- adam(y, model="AAdA", persistence=c(0.3,0.1,0.2), phi=0.9,
                     initial="backcasting", silent=TRUE, headLength=0)
    fCycle   <- adam(y, model="AAdA", persistence=c(0.3,0.1,0.2), phi=0.9,
                     initial="backcasting", silent=TRUE, headLength=m)
    expect_equal(as.numeric(fitted(fDefault)), as.numeric(fitted(fCycle)))
    expect_false(isTRUE(all.equal(as.numeric(fitted(fDefault)), as.numeric(fitted(fLegacy)))))
})

# 2. Where the trend flip is already the exact reversal, the head changes nothing
test_that("head filtering is a no-op for ETS without a damped trend", {
    fDefault <- adam(y, model="AAA", persistence=c(0.3,0.1,0.2),
                     initial="backcasting", silent=TRUE)
    fLegacy  <- adam(y, model="AAA", persistence=c(0.3,0.1,0.2),
                     initial="backcasting", silent=TRUE, headLength=0)
    expect_equal(as.numeric(fitted(fDefault)), as.numeric(fitted(fLegacy)))
})

# 3. initial means the same object as under initial="optimal": the state before y_1
test_that("initial keeps its meaning whatever the head length", {
    fCycle <- adam(y, model="AAA", persistence=c(0.3,0.1,0.2),
                   initial="backcasting", silent=TRUE, headLength=m)
    fLong  <- adam(y, model="AAA", persistence=c(0.3,0.1,0.2),
                   initial="backcasting", silent=TRUE, headLength=2*m)
    fOpt   <- adam(y, model="AAA", persistence=c(0.3,0.1,0.2),
                   initial="optimal", silent=TRUE)
    expect_equal(length(unlist(fCycle$initial)), length(unlist(fOpt$initial)))
    expect_equal(length(unlist(fLong$initial)), length(unlist(fOpt$initial)))
    expect_equal(nrow(fCycle$states), nrow(fLong$states))
})

# 4. Without backcasting the fit does not depend on nIterations
test_that("nIterations is ignored when there is no backcasting", {
    f1 <- adam(y, model="AAA", persistence=c(0.3,0.1,0.2),
               initial=list(level=y[1], trend=0, seasonal=rep(0,m)), silent=TRUE, nIterations=1)
    f3 <- adam(y, model="AAA", persistence=c(0.3,0.1,0.2),
               initial=list(level=y[1], trend=0, seasonal=rep(0,m)), silent=TRUE, nIterations=3)
    expect_equal(as.numeric(fitted(f1)), as.numeric(fitted(f3)))
})

# 5. ssarima builds a non-explosive filter for a pure MA model
test_that("ssarima leaves no identity transition on pure MA models", {
    fMA <- ssarima(y, orders=list(ar=0,i=0,ma=1), lags=1, initial="backcasting", silent=TRUE)
    expect_equal(as.numeric(fMA$transition[1,1]), 0)
    expect_true(all(is.finite(residuals(fMA))))
})

# 6. Each state crosses a turn of the backcast by its own lag, so a series that a
# time-symmetric model fits exactly is reproduced exactly by backcasting
test_that("backcasting reproduces noise-free series of time-symmetric models", {
    tt <- 1:120
    line <- ts(100 + 0.5*tt)
    seasonal <- ts(100 + 0.5*tt + rep(c(5,3,-2,-6,-4,0,2,6,4,-1,-3,-4), length.out=120),
                   frequency=12)
    # The trend is flipped and the level moves one step at each turn
    expect_lt(max(abs(residuals(adam(line, "AAN", persistence=c(0,0))))), 1e-8)
    # ARIMA states with lags 1 and 2 move one and two steps
    expect_lt(max(abs(residuals(adam(line, "NNN", orders=list(i=2, ma=2), arma=list(ma=c(-1.2,0.4)),
                                     constant=FALSE)))), 1e-8)
    # The airline model, with states at lags 1, 12 and 13, converges to the exact fit
    expect_lt(max(abs(residuals(adam(seasonal, "NNN", lags=c(1,12), orders=list(i=c(1,1), ma=c(1,1)),
                                     arma=list(ma=c(-0.5,-0.5)), constant=FALSE, nIterations=5)))), 1e-8)
})
