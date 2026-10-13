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
