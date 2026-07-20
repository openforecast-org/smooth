context("OPG / BHHH covariance (vcov opg=TRUE)")

# The observed Fisher Information (numerical Hessian) can be indefinite when a
# smoothing parameter is estimated at an active bound (e.g. beta = 0), so its
# inverse is not a valid covariance. The OPG estimator J = sum_t s_t s_t' is PSD
# by construction and returns finite standard errors there.

test_that("opg covariance is PSD and finite at a boundary estimate", {
    # AirPassengers MAM drives the trend smoothing beta to its 0 bound.
    m <- suppressWarnings(adam(AirPassengers, "MAM", lags = c(1, 12), initial = "backcasting"))
    skip_if(abs(as.numeric(m$persistence["beta"])) > 1e-3)  # only if beta really at bound
    vO <- suppressWarnings(vcov(m, opg = TRUE))
    expect_equal(dim(vO), c(3L, 3L))
    expect_true(all(eigen(vO, only.values = TRUE)$values > -1e-8))  # PSD
    se <- sqrt(diag(vO))
    expect_true(all(is.finite(se)))
    expect_true(se["beta"] > 0)  # finite, positive SE for the boundary parameter
})

test_that("opg matches the Hessian in a well-identified interior case", {
    set.seed(20)
    x <- sim.es("ANN", obs = 300, persistence = 0.35, initial = 100)
    m <- suppressWarnings(adam(x$data, "ANN", initial = "optimal"))
    skip_if(as.numeric(m$persistence["alpha"]) < 0.05 ||
            as.numeric(m$persistence["alpha"]) > 0.95)  # need interior
    seO <- sqrt(diag(suppressWarnings(vcov(m, opg = TRUE))))["alpha"]
    seH <- sqrt(diag(suppressWarnings(vcov(m))))["alpha"]
    # Same ballpark (OPG and observed information agree up to finite-sample noise)
    expect_lt(abs(seO - seH) / seH, 0.5)
})

test_that("opg returns a full PSD matrix with estimated initials", {
    m <- suppressWarnings(adam(AirPassengers, "MAM", lags = c(1, 12), initial = "optimal"))
    vO <- suppressWarnings(vcov(m, opg = TRUE))
    expect_equal(nrow(vO), length(coef(m)))
    expect_true(all(eigen(vO, only.values = TRUE)$values > -1e-6))
})

test_that("opg covers ARIMA (arma + estimated initials)", {
    m <- suppressWarnings(adam(BJsales, "NNN", orders = list(ar = 1, i = 1, ma = 1),
                               initial = "optimal"))
    vO <- suppressWarnings(vcov(m, opg = TRUE))
    expect_equal(nrow(vO), length(coef(m)))
    fin <- is.finite(diag(vO))
    expect_true(all(eigen(vO[fin, fin, drop = FALSE], only.values = TRUE)$values > -1e-6))
})

test_that("opg covers xreg (ETSX regression coefficients)", {
    set.seed(1)
    xdf <- data.frame(y = as.numeric(AirPassengers), z1 = rnorm(144), z2 = rnorm(144))
    m <- suppressWarnings(adam(xdf, "ANN", regressors = "use", initial = "optimal"))
    vO <- suppressWarnings(vcov(m, opg = TRUE))
    expect_equal(nrow(vO), length(coef(m)))
    expect_true(all(is.finite(sqrt(diag(vO)))))
})

test_that("opg perturbs only dynamics for backcasting (initials re-derived)", {
    # Initials are not free parameters under backcasting, so the OPG covariance
    # spans the dynamics (persistence) only -- the same set the Hessian FI does.
    m <- suppressWarnings(adam(BJsales, "NNN", orders = list(ar = 1, i = 1, ma = 1),
                               initial = "backcasting"))
    vO <- suppressWarnings(vcov(m, opg = TRUE))
    expect_equal(rownames(vO), names(coef(m)))       # arma only, no ARIMAState rows
    expect_false(any(grepl("^ARIMAState", rownames(vO))))
    expect_true(all(eigen(vO, only.values = TRUE)$values > -1e-8))
})

test_that("opg falls back to the Hessian when the refit cannot reproduce", {
    # regressors="select" is re-selected on refit, so the reproduction guard
    # fails and the OPG path falls back to the Hessian covariance.
    set.seed(1)
    xdf <- data.frame(y = as.numeric(AirPassengers),
                      z1 = rnorm(144), z2 = rnorm(144), z3 = rnorm(144))
    m <- suppressWarnings(adam(xdf, "ANN", regressors = "select", initial = "optimal"))
    vO <- suppressWarnings(vcov(m, opg = TRUE))       # falls back, warns
    vH <- suppressWarnings(vcov(m))
    expect_equal(dim(vO), dim(vH))
})

test_that("opg=FALSE leaves the default covariance unchanged", {
    m <- suppressWarnings(adam(AirPassengers, "ANN", initial = "optimal"))
    expect_equal(suppressWarnings(vcov(m)), suppressWarnings(vcov(m, opg = FALSE)))
})
