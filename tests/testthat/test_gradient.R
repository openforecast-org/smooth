context("Tests for initial=\"gradient\"")

# initial="gradient" solves for the initial state by least squares. It fixes a
# backcasting divergence for seasonal additive ETS with a trend and a large
# seasonal smoothing parameter, and must leave the cases backcasting already
# handles well unchanged, and fall back safely for unsupported specifications.

# Deterministic additive seasonal series with trend (worst-cell persistence)
m <- 12
s0 <- 10 * sin(2 * pi * (1:m) / m); s0 <- s0 - mean(s0)
set.seed(9282)
xSeas <- sim.es("AAA", obs = 120, frequency = m,
                persistence = c(0.6, 0.15, 0.35),
                initial = c(100, 2), initialSeason = s0)
sseTrue <- sum(xSeas$residuals^2)

test_that("gradient fixes the seasonal-trend backcasting divergence", {
    fB <- adam(xSeas$data, "AAA", persistence = c(0.6, 0.15, 0.35),
               initial = "backcasting", silent = TRUE)
    fG <- adam(xSeas$data, "AAA", persistence = c(0.6, 0.15, 0.35),
               initial = "gradient", silent = TRUE)
    deficitB <- (sum(residuals(fB)^2) - sseTrue) / 100
    deficitG <- (sum(residuals(fG)^2) - sseTrue) / 100
    # backcasting diverges badly here; gradient must land near -rank (13), i.e.
    # a small negative deficit. Assert it is healthy and far below backcasting.
    expect_gt(deficitB, 100)
    expect_lt(deficitG, 5)
    expect_equal(fG$initialType, "gradient")
})

test_that("gradient does not disturb a well-behaved seasonal case", {
    # Low-persistence DGP: backcasting is already healthy here (deficit ~ -rank).
    persLow <- c(0.1, 0.05, 0.1)
    set.seed(9282)
    xLow <- sim.es("AAA", obs = 120, frequency = m, persistence = persLow,
                   initial = c(100, 2), initialSeason = s0)
    sseLow <- sum(xLow$residuals^2)
    fB <- adam(xLow$data, "AAA", persistence = persLow, initial = "backcasting", silent = TRUE)
    fG <- adam(xLow$data, "AAA", persistence = persLow, initial = "gradient", silent = TRUE)
    dB <- (sum(residuals(fB)^2) - sseLow) / 100
    dG <- (sum(residuals(fG)^2) - sseLow) / 100
    # both healthy; gradient must be no worse than backcasting, and negative
    expect_lt(dG, dB + 2)
    expect_lt(dG, 0)
})

test_that("gradient works for non-seasonal additive ETS", {
    set.seed(41)
    y <- ts(cumsum(rnorm(80, 0.2, 1)) + 100)
    fG <- adam(y, "AAN", initial = "gradient", silent = TRUE)
    expect_equal(fG$initialType, "gradient")
    expect_true(is.finite(logLik(fG)))
})

test_that("gradient handles multiplicative / mixed ETS via Gauss-Newton", {
    # Positive multiplicative-seasonal series (low persistence keeps it positive).
    set.seed(42)
    s0m <- 1 + 0.15 * sin(2 * pi * (1:12) / 12)
    xM <- sim.es("MAM", obs = 120, frequency = 12, persistence = c(0.2, 0.05, 0.1),
                 initial = c(100, 1.002), initialSeason = s0m)
    skip_if(any(!is.finite(xM$data)) || min(xM$data) <= 0)
    sseM <- sum(xM$residuals^2)
    fB <- adam(xM$data, "MAM", persistence = c(0.2, 0.05, 0.1),
               initial = "backcasting", silent = TRUE)
    fG <- adam(xM$data, "MAM", persistence = c(0.2, 0.05, 0.1),
               initial = "gradient", silent = TRUE)
    # Multiplicative models have no backcasting divergence; gradient must run
    # (initType set) and be no worse than backcasting.
    expect_equal(fG$initialType, "gradient")
    expect_true(is.finite(logLik(fG)))
    dB <- (sum(residuals(fB)^2) - sseM) / 100
    dG <- (sum(residuals(fG)^2) - sseM) / 100
    expect_lt(dG, dB + 1)
})

test_that("gradient solves additive ARIMA initials to optimal quality", {
    # Additive ARIMA is in scope: the gradient solve profiles the initials by
    # least squares (residuals are affine in the initial profile for additive
    # SSOE), matching the optimal-initials SSE at the same dynamics and beating
    # backcasting.
    set.seed(1)
    y <- ts(cumsum(rnorm(80, 0.1, 1)) + 100)
    af <- list(ar = 0.4, ma = 0.3)
    fO <- adam(y, "NNN", orders = list(ar = 1, i = 1, ma = 1), arma = af,
               initial = "optimal", loss = "MSE", silent = TRUE)
    fB <- adam(y, "NNN", orders = list(ar = 1, i = 1, ma = 1), arma = af,
               initial = "backcasting", loss = "MSE", silent = TRUE)
    fG <- adam(y, "NNN", orders = list(ar = 1, i = 1, ma = 1), arma = af,
               initial = "gradient", loss = "MSE", silent = TRUE)
    expect_equal(sum(residuals(fG)^2), sum(residuals(fO)^2), tolerance = 1e-3)
    expect_lt(sum(residuals(fG)^2), sum(residuals(fB)^2))
})

test_that("multiplicative ARIMA gradient falls back to backcasting", {
    # Multiplicative ARIMA would take the ETS-only Gauss-Newton Jacobian path,
    # so it falls back to backcasting (identical result), not the affine solve.
    set.seed(1)
    y <- ts(abs(cumsum(rnorm(80, 0.1, 1))) + 100)
    fB <- adam(y, "MNN", orders = list(ar = 1, i = 1, ma = 1),
               initial = "backcasting", silent = TRUE)
    fG <- suppressWarnings(adam(y, "MNN", orders = list(ar = 1, i = 1, ma = 1),
                                initial = "gradient", silent = TRUE))
    expect_equal(logLik(fG), logLik(fB), tolerance = 1e-8)
})

test_that("gradient estimates persistence jointly (nested, no backcasting)", {
    # Persistence not provided: the optimiser estimates it while the initials
    # are solved by gradient each evaluation. Must recover a healthy fit on the
    # otherwise-divergent seasonal-trend case.
    fG <- adam(xSeas$data, "AAA", initial = "gradient", silent = TRUE)
    expect_equal(fG$initialType, "gradient")
    expect_true(is.finite(logLik(fG)))
    expect_lt((sum(residuals(fG)^2) - sseTrue) / 100, 5)
})

# The loss-aware solve: initial="gradient" profiles the actual estimation loss
# (not the SSE surrogate), so at fixed persistence the gradient fit must never
# lose to backcasting on the loss itself.

set.seed(41)
xLoss <- ts(100 + cumsum(rnorm(72, 0, 2)) +
                rep(c(5, -3, 2, -4, 6, -6, 3, -1, 2, -2, 4, -6), 6), frequency = 12)
xLossM <- ts((100 + cumsum(rnorm(72, 0, 2))) *
                 rep(c(1.05, 0.97, 1.02, 0.96, 1.06, 0.94,
                       1.03, 0.99, 1.02, 0.98, 1.04, 0.94), 6), frequency = 12)

test_that("gradient profiles the robust one-step losses (MAE, HAM)", {
    for (loss in c("MAE", "HAM")) {
        fB <- adam(xLoss, "AAA", initial = "backcasting", loss = loss, silent = TRUE)
        fG <- adam(xLoss, "AAA", initial = "gradient", loss = loss, silent = TRUE)
        expect_lt(fG$lossValue, fB$lossValue)
    }
})

test_that("gradient handles exact zero residuals in the MAE weights", {
    # A locally constant series produces exact-zero residuals: the IRLS sweep
    # must drop those rows (no epsilon floor) and stay finite.
    yTies <- ts(rep(c(100, 100, 100, 105), 20), frequency = 4)
    fG <- adam(yTies, "ANA", initial = "gradient", loss = "MAE", silent = TRUE)
    expect_true(is.finite(fG$lossValue))
})

test_that("gradient profiles the multiplicative likelihoods", {
    # dgamma (the multiplicative default), dlnorm and dinvgauss get their exact
    # rho in the Gauss-Newton branch; the likelihood must not lose to backcasting.
    for (dist in c("default", "dlnorm", "dinvgauss")) {
        args <- list(data = xLossM, model = "MAM", initial = "backcasting", silent = TRUE)
        if (dist != "default") { args$distribution <- dist }
        fB <- do.call(adam, args)
        args$initial <- "gradient"
        fG <- do.call(adam, args)
        expect_lt(fG$lossValue, fB$lossValue + 1e-8)
    }
})

test_that("gradient solves the additive multistep losses on the multistep design", {
    for (loss in c("MSEh", "TMSE", "GTMSE", "MSCE", "GPL")) {
        fB <- adam(xLoss, "AAA", initial = "backcasting", loss = loss, h = 6,
                   holdout = FALSE, silent = TRUE)
        fG <- adam(xLoss, "AAA", initial = "gradient", loss = loss, h = 6,
                   holdout = FALSE, silent = TRUE)
        expect_lt(fG$lossValue, fB$lossValue)
    }
})

test_that("gradient with a custom loss function falls back to backcasting", {
    lossCustom <- function(actual, fitted, B) { sum(abs(actual - fitted)^1.5) }
    fB <- adam(xLoss, "AAA", initial = "backcasting", loss = lossCustom, silent = TRUE)
    fG <- suppressMessages(adam(xLoss, "AAA", initial = "gradient",
                                loss = lossCustom, silent = TRUE))
    expect_equal(fG$lossValue, fB$lossValue, tolerance = 1e-8)
})

test_that("gradient solves additive SSOE initials for CES/GUM/SSARIMA/SPARMA", {
    # These are additive SSOE models, so the affine least-squares gradient solve
    # applies: it must run (not fall back), differ from backcasting, and not lose
    # to backcasting on the in-sample SSE.
    set.seed(1)
    y <- ts(100 + cumsum(rnorm(120, 0, 2)))
    engines <- list(
        SSARIMA = function(i) ssarima(y, orders = list(ar = 1, i = 1, ma = 1), lags = 1,
                                      initial = i, loss = "MSE", silent = TRUE),
        SPARMA  = function(i) sparma(y, orders = list(ar = 1, ma = 1), lags = 1,
                                     initial = i, loss = "MSE", silent = TRUE),
        GUM     = function(i) gum(y, orders = 2, lags = 1, initial = i, loss = "MSE", silent = TRUE),
        CES     = function(i) ces(y, seasonality = "none", initial = i, loss = "MSE", silent = TRUE))
    for(fn in engines){
        mb <- fn("backcasting"); mg <- fn("gradient")
        sseB <- sum(residuals(mb)^2); sseG <- sum(residuals(mg)^2)
        expect_true(is.finite(sseG))
        expect_false(isTRUE(all.equal(sseB, sseG, tolerance = 1e-6)))  # genuinely solved
        expect_lte(sseG, sseB * 1.001 + 1e-6)                          # no worse than backcasting
    }
})
