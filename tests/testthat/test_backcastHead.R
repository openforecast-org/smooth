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
