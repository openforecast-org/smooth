context("OPG / BHHH covariance (vcov type=\"opg\")")

# The observed Fisher Information (numerical Hessian) can be indefinite when a
# smoothing parameter is estimated at an active bound (e.g. beta = 0), so its
# inverse is not a valid covariance. The OPG estimator J = sum_t s_t s_t' is PSD
# by construction and returns finite standard errors there.

test_that("opg covariance is PSD and finite at a boundary estimate", {
    # AirPassengers MAM drives the trend smoothing beta to its 0 bound.
    m <- suppressWarnings(adam(AirPassengers, "MAM", lags = c(1, 12), initial = "backcasting"))
    skip_if(abs(as.numeric(m$persistence["beta"])) > 1e-3)  # only if beta really at bound
    vO <- suppressWarnings(vcov(m, type = "opg"))
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
    seO <- sqrt(diag(suppressWarnings(vcov(m, type = "opg"))))["alpha"]
    seH <- sqrt(diag(suppressWarnings(vcov(m, type = "hessian"))))["alpha"]
    # Same ballpark (OPG and observed information agree up to finite-sample noise)
    expect_lt(abs(seO - seH) / seH, 0.5)
})

test_that("opg returns a full PSD matrix with estimated initials", {
    m <- suppressWarnings(adam(AirPassengers, "MAM", lags = c(1, 12), initial = "optimal"))
    vO <- suppressWarnings(vcov(m, type = "opg"))
    expect_equal(nrow(vO), length(coef(m)))
    expect_true(all(eigen(vO, only.values = TRUE)$values > -1e-6))
})

test_that("opg covers ARIMA (arma + estimated initials)", {
    m <- suppressWarnings(adam(BJsales, "NNN", orders = list(ar = 1, i = 1, ma = 1),
                               initial = "optimal"))
    vO <- suppressWarnings(vcov(m, type = "opg"))
    expect_equal(nrow(vO), length(coef(m)))
    fin <- is.finite(diag(vO))
    expect_true(all(eigen(vO[fin, fin, drop = FALSE], only.values = TRUE)$values > -1e-6))
})

test_that("opg covers xreg (ETSX regression coefficients)", {
    set.seed(1)
    xdf <- data.frame(y = as.numeric(AirPassengers), z1 = rnorm(144), z2 = rnorm(144))
    m <- suppressWarnings(adam(xdf, "ANN", regressors = "use", initial = "optimal"))
    vO <- suppressWarnings(vcov(m, type = "opg"))
    expect_equal(nrow(vO), length(coef(m)))
    expect_true(all(is.finite(sqrt(diag(vO)))))
})

test_that("opg perturbs only dynamics for backcasting (initials re-derived)", {
    # Initials are not free parameters under backcasting, so the OPG covariance
    # spans the dynamics (persistence) only -- the same set the Hessian FI does.
    m <- suppressWarnings(adam(BJsales, "NNN", orders = list(ar = 1, i = 1, ma = 1),
                               initial = "backcasting"))
    vO <- suppressWarnings(vcov(m, type = "opg"))
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
    vO <- suppressWarnings(vcov(m, type = "opg"))       # falls back, warns
    vH <- suppressWarnings(vcov(m, type = "hessian"))
    expect_equal(dim(vO), dim(vH))
})

test_that("opg covers the state-space ARIMA engine (ssarima)", {
    # ssarima uses its own ARIMA representation, so the OPG refit is dispatched
    # to ssarima() (not adam()); it must compute a PSD covariance rather than
    # fall back to the Hessian.
    m <- suppressWarnings(ssarima(BJsales, orders = list(ar = 1, i = 1, ma = 1),
                                  initial = "optimal", h = 0))
    vO <- suppressWarnings(vcov(m, type = "opg"))
    vH <- suppressWarnings(vcov(m, type = "hessian"))
    expect_false(isTRUE(all.equal(vO, vH, tolerance = 1e-6)))  # genuinely OPG
    fin <- is.finite(diag(vO))
    expect_true(all(eigen(vO[fin, fin, drop = FALSE], only.values = TRUE)$values > -1e-6))
})

test_that("opg for ssarima backcasting spans dynamics only", {
    m <- suppressWarnings(ssarima(BJsales, orders = list(ar = 2, i = 1, ma = 1),
                                  initial = "backcasting", h = 0))
    vO <- suppressWarnings(vcov(m, type = "opg"))
    expect_false(any(grepl("^ARIMAState", rownames(vO))))     # initials re-derived
    expect_true(all(eigen(vO, only.values = TRUE)$values > -1e-8))
})

test_that("opg covers non-seasonal CES (complex smoothing + initials)", {
    m <- suppressWarnings(ces(AirPassengers, seasonality = "none",
                              initial = "optimal", h = 0))
    vO <- suppressWarnings(vcov(m, type = "opg"))
    vH <- suppressWarnings(vcov(m, type = "hessian"))
    expect_false(isTRUE(all.equal(vO, vH, tolerance = 1e-6)))  # genuinely OPG
    fin <- is.finite(diag(vO))
    expect_true(all(eigen(vO[fin, fin, drop = FALSE], only.values = TRUE)$values > -1e-6))
})

test_that("opg for CES backcasting spans the smoothing parameters only", {
    m <- suppressWarnings(ces(AirPassengers, seasonality = "none",
                              initial = "backcasting", h = 0))
    vO <- suppressWarnings(vcov(m, type = "opg"))
    expect_equal(rownames(vO), names(coef(m)))          # alpha_0, alpha_1 only
    expect_true(all(eigen(vO, only.values = TRUE)$values > -1e-8))
})

test_that("opg covers seasonal CES (level/potential rows + seasonal cells)", {
    m <- suppressWarnings(ces(AirPassengers, seasonality = "full", lags = 12,
                              initial = "optimal", h = 0))
    vO <- suppressWarnings(vcov(m, type = "opg"))
    vH <- suppressWarnings(vcov(m, type = "hessian"))
    expect_equal(dim(vO), c(length(coef(m)), length(coef(m))))
    expect_false(isTRUE(all.equal(vO, vH, tolerance = 1e-6)))  # genuinely OPG
    fin <- is.finite(diag(vO))
    expect_true(all(eigen(vO[fin, fin, drop = FALSE], only.values = TRUE)$values > -1e-6))
})

test_that("opg covers GUM (persistence g + transition F + vt initials)", {
    m <- suppressWarnings(gum(AirPassengers, orders = 2, lags = 1,
                              initial = "optimal", h = 0))
    vO <- suppressWarnings(vcov(m, type = "opg"))
    vH <- suppressWarnings(vcov(m, type = "hessian"))
    expect_equal(dim(vO), c(length(coef(m)), length(coef(m))))
    expect_false(isTRUE(all.equal(vO, vH, tolerance = 1e-6)))  # genuinely OPG
    fin <- is.finite(diag(vO))
    expect_true(all(eigen(vO[fin, fin, drop = FALSE], only.values = TRUE)$values > -1e-6))

    ms <- suppressWarnings(gum(AirPassengers, orders = 1, lags = 12,
                               initial = "optimal", h = 0))
    vOs <- suppressWarnings(vcov(ms, type = "opg"))
    fins <- is.finite(diag(vOs))
    expect_true(all(eigen(vOs[fins, fins, drop = FALSE], only.values = TRUE)$values > -1e-6))
})

test_that("opg for GUM backcasting spans dynamics only and stays PSD", {
    # Requires the empty-B fix (providing g + F + backcasting no longer errors)
    # and the pseudo-inverse fallback for the ill-conditioned score matrix.
    m <- suppressWarnings(gum(AirPassengers, orders = 2, lags = 1,
                              initial = "backcasting", h = 0))
    vO <- suppressWarnings(vcov(m, type = "opg"))
    vH <- suppressWarnings(vcov(m, type = "hessian"))
    expect_equal(rownames(vO), names(coef(m)))          # g / F only, no vt
    expect_false(any(grepl("^vt", rownames(vO))))
    expect_false(isTRUE(all.equal(vO, vH, tolerance = 1e-6)))  # genuinely OPG
    expect_true(all(eigen(vO, only.values = TRUE)$values > -1e-6))
})

test_that("opg covers sparma (held arma + provided initials round-trip)", {
    # sparma now holds provided arma coefficients and re-accepts its own initial
    # states (front-padded to the dense companion length), so the OPG refit
    # reproduces the fit and returns a genuine (non-Hessian) PSD covariance.
    m <- suppressWarnings(sparma(BJsales, orders = list(ar = 1, ma = 1),
                                 initial = "optimal", h = 0))
    vO <- suppressWarnings(vcov(m, type = "opg"))
    vH <- suppressWarnings(vcov(m, type = "hessian"))
    expect_equal(dim(vO), c(length(coef(m)), length(coef(m))))
    expect_false(isTRUE(all.equal(vO, vH, tolerance = 1e-6)))  # genuinely OPG
    expect_true(all(eigen(vO, only.values = TRUE)$values > -1e-6))
})

test_that("opg covers sparse-order sparma (phi2 with phi1 absent)", {
    # The sparse AR structure (only phi2 free) exercises the dense->sparse
    # initial mapping; OPG must still produce a PSD covariance of the right size.
    m <- suppressWarnings(sparma(BJsales, orders = list(ar = 2, ma = 1),
                                 initial = "optimal", h = 0))
    vO <- suppressWarnings(vcov(m, type = "opg"))
    expect_equal(dim(vO), c(length(coef(m)), length(coef(m))))
    fin <- is.finite(diag(vO))
    expect_true(all(eigen(vO[fin, fin, drop = FALSE], only.values = TRUE)$values > -1e-6))
})

test_that("opg for sparma backcasting spans the arma parameters only", {
    m <- suppressWarnings(sparma(BJsales, orders = list(ar = 2, ma = 1),
                                 initial = "backcasting", h = 0))
    vO <- suppressWarnings(vcov(m, type = "opg"))
    expect_equal(rownames(vO), names(coef(m)))         # phi2, theta1 -- no initials
    expect_false(any(grepl("^initial", rownames(vO))))
    expect_true(all(eigen(vO, only.values = TRUE)$values > -1e-8))
})

test_that("opg covers the occurrence model om (Bernoulli score)", {
    set.seed(42)
    yb <- rbinom(140, 1, 0.55)
    m <- suppressWarnings(om(yb, "ANN", occurrence = "odds-ratio", initial = "optimal", h = 0))
    vO <- suppressWarnings(vcov(m, type = "opg"))
    vH <- suppressWarnings(vcov(m, type = "hessian"))
    expect_equal(dim(vO), c(length(coef(m)), length(coef(m))))
    expect_false(isTRUE(all.equal(vO, vH, tolerance = 1e-6)))  # genuinely OPG
    expect_true(all(eigen(vO, only.values = TRUE)$values > -1e-6))
    expect_true(all(is.finite(sqrt(diag(vO)))))
})

test_that("opg for om backcasting spans the dynamics only", {
    set.seed(7)
    yb <- rbinom(140, 1, 0.5)
    m <- suppressWarnings(om(yb, "AAN", occurrence = "odds-ratio", initial = "backcasting", h = 0))
    vO <- suppressWarnings(vcov(m, type = "opg"))
    expect_false(any(grepl("^level$|^trend$", rownames(vO))))  # initials re-derived
    expect_true(all(eigen(vO, only.values = TRUE)$values > -1e-8))
})

test_that("opg covers the coupled occurrence model omg (joint score)", {
    # Exercises the one-sided-difference fallback: the coupled occurrence link
    # returns NaN for a negative persistence, so a boundary alpha=0 can only be
    # differenced on the feasible side.
    set.seed(11)
    yb <- rbinom(150, 1, 0.5)
    m <- suppressWarnings(omg(yb, modelA = "ANN", modelB = "ANN", initial = "optimal", h = 0))
    vO <- suppressWarnings(vcov(m, type = "opg"))
    vH <- suppressWarnings(vcov(m, type = "hessian"))
    nJoint <- length(c(m$modelA$B, m$modelB$B))
    expect_equal(dim(vO), c(nJoint, nJoint))
    expect_false(isTRUE(all.equal(vO, vH, tolerance = 1e-6)))  # genuinely OPG
    expect_true(all(eigen(vO, only.values = TRUE)$values > -1e-6))
    expect_true(all(is.finite(sqrt(diag(vO)))))
})

test_that("omg re-fit via model= reproduces the original fit (optimal + backcasting)", {
    # Regression guard for the joint-B re-entry: providing persistence used to
    # flip its Estimate flag off and shift the seed, corrupting the re-fit.
    set.seed(3)
    yb <- rbinom(130, 1, 0.5)
    for (init in c("optimal", "backcasting")) {
        m <- suppressWarnings(omg(yb, modelA = "AAN", modelB = "ANN", initial = init, h = 0))
        r <- suppressWarnings(omg(m$modelA$data, model = m, h = 0))
        expect_equal(as.numeric(logLik(r)), as.numeric(logLik(m)), tolerance = 1e-6)
    }
})

test_that("type default is opg, and the deprecated bootstrap= still maps through", {
    m <- suppressWarnings(adam(AirPassengers, "ANN", initial = "optimal"))
    # Default is now the OPG covariance.
    expect_equal(suppressWarnings(vcov(m)), suppressWarnings(vcov(m, type = "opg")))
    # type="hessian" is genuinely different from the default OPG.
    expect_false(isTRUE(all.equal(suppressWarnings(vcov(m, type = "hessian")),
                                  suppressWarnings(vcov(m, type = "opg")),
                                  tolerance = 1e-6)))
    # Deprecated bootstrap=TRUE warns and routes to the bootstrap covariance.
    expect_warning(vcov(m, bootstrap = TRUE, nsim = 50), "deprecated")
})
