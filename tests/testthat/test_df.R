context("Degrees-of-freedom accounting (structural, init-invariant)")

set.seed(1)
yA  <- ts(100 + cumsum(rnorm(96, 0, 2)), frequency = 12)
yS  <- ts(100 + cumsum(rnorm(96, 0, 2)) +
              rep(c(5, -3, 2, -4, 6, -6, 3, -1, 2, -2, 4, -6), 8), frequency = 12)
yN  <- ts(100 + cumsum(rnorm(240, 0, 2)), frequency = 8)

# 1. Initialisation-method invariance: the same model has the same df whether
#    the initials are optimised, backcast or gradient-solved.
test_that("nparam is identical across initialisation methods", {
    for (mod in c("ANN", "AAN", "AAdN")) {
        no <- nparam(adam(yA, mod, initial = "optimal", silent = TRUE))
        nb <- nparam(adam(yA, mod, initial = "backcasting", silent = TRUE))
        ng <- nparam(adam(yA, mod, initial = "gradient", silent = TRUE))
        expect_equal(nb, no)
        expect_equal(ng, no)
    }
    # seasonal
    no <- nparam(adam(yS, "AAA", initial = "optimal", silent = TRUE))
    nb <- nparam(adam(yS, "AAA", initial = "backcasting", silent = TRUE))
    expect_equal(nb, no)
    expect_equal(no, 17)   # 3 smoothing + 13 initials + 1 scale
    # ARIMA
    no <- nparam(adam(yA, "NNN", orders = list(ar = 2, i = 1, ma = 2),
                      initial = "optimal", silent = TRUE))
    nb <- nparam(adam(yA, "NNN", orders = list(ar = 2, i = 1, ma = 2),
                      initial = "backcasting", silent = TRUE))
    expect_equal(nb, no)
})

# 2. Non-coprime multi-seasonal: the shared-frequency redundancy is removed for
#    ALL init methods, including optimal.
test_that("non-coprime multi-seasonal df uses the gcd correction", {
    no <- nparam(adam(yN, "AAA", lags = c(1, 4, 8), initial = "optimal", silent = TRUE))
    nb <- nparam(adam(yN, "AAA", lags = c(1, 4, 8), initial = "backcasting", silent = TRUE))
    # 4 smoothing + 9 initials (IE(1,4,8)=8 + trend) + 1 scale = 14 (naive would be 17)
    expect_equal(no, 14)
    expect_equal(nb, 14)
})

# 3. The inclusion-exclusion helper matches known ranks.
test_that("dfInitialsETSLevelSeasonal matches the identifiable ranks", {
    f <- smooth:::dfInitialsETSLevelSeasonal
    expect_equal(f(numeric(0), TRUE), 1)      # level only
    expect_equal(f(12, TRUE), 12)             # level + seasonal m=12
    expect_equal(f(c(4, 8), TRUE), 8)         # nested, gcd 4
    expect_equal(f(c(4, 6), TRUE), 8)         # gcd 2
    expect_equal(f(c(3, 7), TRUE), 9)         # coprime (level + seasonal, no trend)
    expect_equal(f(c(2, 4, 8), TRUE), 8)      # multiple overlaps
})

# 4. Estimate-gating: provided parameters are not counted.
test_that("only estimated parameters are counted", {
    mEst <- adam(yA, "AAN", initial = "optimal", silent = TRUE)
    # Provide alpha; estimate the rest -> one fewer estimated parameter.
    mProv <- adam(yA, "AAN", initial = "optimal",
                  persistence = list(alpha = 0.1), silent = TRUE)
    expect_equal(nparam(mProv), nparam(mEst) - 1)
})

# 5. xreg regressors="adapt" counts the delta persistence parameters.
test_that("regressors='adapt' counts the delta parameters", {
    x1 <- rnorm(96); x2 <- rnorm(96)
    dat <- data.frame(y = as.numeric(yA) + 2 * x1 - x2, x1 = x1, x2 = x2)
    mUse   <- adam(dat, "ANN", regressors = "use", initial = "optimal", silent = TRUE)
    mAdapt <- adam(dat, "ANN", regressors = "adapt", initial = "optimal", silent = TRUE)
    # adapt adds one delta per regressor
    expect_equal(nparam(mAdapt), nparam(mUse) + 2)
})

# 6. Scale is a parameter for every loss (concentrated likelihood).
test_that("the scale counts for non-likelihood losses too", {
    mLik <- adam(yA, "AAN", loss = "likelihood", initial = "optimal", silent = TRUE)
    mMSE <- adam(yA, "AAN", loss = "MSE", initial = "optimal", silent = TRUE)
    expect_equal(nparam(mMSE), nparam(mLik))       # both include the scale
    expect_equal(mMSE$nParam[1, 4], 1)             # scale cell
})

# 7. Occurrence models: Bernoulli has no scale; initials are counted.
test_that("om counts initials and has no scale", {
    set.seed(5); ot <- ts(rbinom(120, 1, plogis(seq(-1, 1.5, length.out = 120))))
    no <- nparam(om(ot, model = "MNM", occurrence = "odds-ratio", lags = c(1, 12),
                    initial = "optimal", silent = TRUE))
    nb <- nparam(om(ot, model = "MNM", occurrence = "odds-ratio", lags = c(1, 12),
                    initial = "backcasting", silent = TRUE))
    expect_equal(nb, no)
    expect_equal(no, 14)                           # 2 smoothing + 12 initials, no scale
    mo <- om(ot, model = "MNN", occurrence = "odds-ratio", initial = "optimal", silent = TRUE)
    expect_equal(mo$nParam[1, 4], 0)               # Bernoulli: no scale
})

# 8. CES / GUM / SSARIMA backcasting == optimal.
test_that("CES/GUM/SSARIMA count initials under backcasting", {
    for (fn in list(
        function(ini) ces(yA, seasonality = "none", initial = ini, silent = TRUE),
        function(ini) gum(yA, orders = 2, lags = 1, initial = ini, silent = TRUE),
        function(ini) ssarima(yA, orders = list(ar = 1, i = 1, ma = 1), lags = 1,
                              initial = ini, silent = TRUE))) {
        expect_equal(nparam(fn("backcasting")), nparam(fn("optimal")))
    }
})

# 9. SMA df is order + 1 (AR with unit-root coefficients).
test_that("sma df is order + 1", {
    yy <- ts(rnorm(120, 100, 3))
    expect_equal(nparam(sma(yy, order = 1, silent = TRUE)), 2)
    expect_equal(nparam(sma(yy, order = 4, silent = TRUE)), 5)
    expect_equal(nparam(sma(yy, order = 12, silent = TRUE)), 13)
})
