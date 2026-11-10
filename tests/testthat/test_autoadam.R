context("Baseline tests for auto.adam() — pin behavior before refactoring")

xreg <- data.frame(y=AirPassengers,
                   x=factor(temporaldummy(AirPassengers, factors=TRUE)))

# 1. ETS distribution selection
test_that("auto.adam() ETS distribution selection on AirPassengers", {
    skip_on_cran()
    set.seed(42)
    m <- auto.adam(AirPassengers, "ZZZ",
                   distribution=c("dnorm","dlnorm","dgamma"), silent=TRUE)
    expect_equal(m$distribution, "dnorm")
    expect_equal(modelType(m), "MAM")
    expect_equal(AICc(m), 1085.408, tolerance=0.01)
    expect_equal(length(m$persistence), 3)
})

# The positive distributions on non-positive data leave nothing to estimate
test_that("auto.adam() stops when all the distributions are dropped", {
    expect_error(suppressWarnings(auto.adam(BJsales - 250, "ANN",
                                            distribution=c("dlnorm","dgamma"))),
                 "All the provided distributions were dropped")
})

# 2. ARIMA order selection on BJsales
test_that("auto.adam() ARIMA selection (NNN) on BJsales", {
    skip_on_cran()
    set.seed(42)
    m <- auto.adam(BJsales, "NNN",
                   orders=list(ar=c(2,0), i=c(2,0), ma=c(2,0), select=TRUE),
                   distribution="dnorm", silent=TRUE)
    expect_equal(modelType(m), "NNN")
    # Re-pinned after the drift flip in backcasting: ARIMA(0,1,0) with drift now
    # fits its first observation, so the residuals that guide the ACF / PACF steps
    # of the selection change and the path ends at ARIMA(2,1,2) with drift (AICc
    # 527.452; ARIMA(1,1,2) with drift, selected before, has 523.443 but is not
    # on the new path).
    expect_equal(AICc(m), 527.452, tolerance=0.01)
    expect_equal(m$distribution, "dnorm")
    expect_equal(as.numeric(m$arma[[1]]), c(-0.026419, 0.719587), tolerance=1e-3)
})

# 3. ETS + ARIMA selection on AirPassengers
test_that("auto.adam() ETS+ARIMA selection on AirPassengers", {
    skip_on_cran()
    set.seed(42)
    m <- auto.adam(AirPassengers, "ZZZ",
                   orders=list(ar=c(2,1), i=c(1,0), ma=c(2,1), select=TRUE),
                   distribution=c("dnorm","dgamma"), lags=c(1,12), silent=TRUE)
    expect_equal(m$distribution, "dnorm")
    expect_equal(modelType(m), "MAM")
    expect_equal(AICc(m), 1085.408, tolerance=0.01)
})

# 4. Regressors = "use"
test_that("auto.adam() with regressors='use' on AirPassengers+xreg", {
    skip_on_cran()
    set.seed(42)
    m <- auto.adam(xreg, "ZZZ",
                   distribution=c("dnorm","dlnorm"), lags=c(1,12),
                   regressors="use", silent=TRUE)
    expect_equal(m$distribution, "dlnorm")
    expect_equal(modelType(m), "ANM")
    # Re-pinned after the per-lag turns of backcasting: ETSX(ANM) with drift fits
    # better (AICc 1096.216 against 1104.974 before)
    expect_equal(AICc(m), 1096.216, tolerance=0.01)
    expect_equal(length(m$persistence), 15)
})

# 5. Regressors = "select"
test_that("auto.adam() with regressors='select' on AirPassengers+xreg", {
    skip_on_cran()
    set.seed(42)
    m <- auto.adam(xreg, "ZZZ",
                   distribution=c("dnorm","dlnorm"), lags=c(1,12),
                   regressors="select", silent=TRUE)
    expect_equal(m$distribution, "dnorm")
    expect_equal(modelType(m), "MAM")
    expect_equal(AICc(m), 1085.408, tolerance=0.01)
})

# 6. Regressors = "adapt"
test_that("auto.adam() with regressors='adapt' on AirPassengers+xreg", {
    skip_on_cran()
    set.seed(42)
    m <- auto.adam(xreg, "ZZZ",
                   distribution=c("dnorm","dlnorm"), lags=c(1,12),
                   regressors="adapt", silent=TRUE)
    # With the ETS smoother for the Hannan-Rissanen values of ETS+ARIMA, an AR(1)
    # on top of ETSX(MMN) lowers AICc from 1090.929 (ETSX(MMN) with dlnorm). With the
    # Hannan-Rissanen values on the residuals of the regression, the dlnorm one reaches
    # 1083.697 and beats the dnorm one (1083.931 before, 1084.049 now)
    expect_equal(m$distribution, "dlnorm")
    expect_equal(modelType(m), "MMN")
    expect_equal(orders(m)$ar, c(1, 0))
    expect_equal(AICc(m), 1083.697, tolerance=0.01)
})

# 7. Outliers = "use"
test_that("auto.adam() with outliers='use' on BJsales", {
    skip_on_cran()
    set.seed(42)
    m <- auto.adam(BJsales, "ZZZ",
                   distribution=c("dnorm","dlaplace"), outliers="use", silent=TRUE)
    expect_equal(m$distribution, "dnorm")
    expect_equal(modelType(m), "AMdN")
    expect_equal(AICc(m), 525.256, tolerance=0.01)
})

# 8. Outliers = "select"
test_that("auto.adam() with outliers='select' on BJsales", {
    skip_on_cran()
    set.seed(42)
    m <- auto.adam(BJsales, "ZZZ",
                   distribution=c("dnorm","dlaplace"), outliers="select", silent=TRUE)
    expect_equal(m$distribution, "dnorm")
    expect_equal(modelType(m), "AMdN")
    expect_equal(AICc(m), 525.592, tolerance=0.01)
})

# 9. Long series: the ARIMA selector must not deparse the data into a name
test_that("auto.msarima() selects orders on a series longer than R's name limit", {
    skip_on_cran()
    set.seed(42)
    y <- ts(100 + cumsum(rnorm(1500)))
    m <- auto.msarima(y, orders=list(ar=1, i=1, ma=1), lags=1,
                      h=10, holdout=TRUE, silent=TRUE)
    # The response name comes from deparse(substitute(data)) inside the fitter;
    # deparsing 1500 observations into it breaks R's 10000-byte limit on names.
    expect_lt(nchar(colnames(m$data)[1]), 100)
    expect_true(grepl("ARIMA", m$model))
})

# The smoother reaches the models of the ARIMA order selection
test_that("auto.msarima() passes the smoother on", {
    y <- log(AirPassengers)
    lossValues <- sapply(c("ma","lowess"), function(s){
        auto.msarima(y, lags=c(1,12), initial="optimal", smoother=s)$lossValue
    })
    expect_equal(unname(lossValues["lowess"]),
                 msarima(y, orders=list(ar=c(0,0),i=c(1,1),ma=c(1,1)), lags=c(1,12),
                         initial="optimal", smoother="lowess")$lossValue)
    expect_false(isTRUE(all.equal(lossValues[1], lossValues[2])))
})

# The regressors provided in xreg are named after their columns, or x1, x2, ...
test_that("auto.msarima() names the regressors of xreg", {
    set.seed(3)
    xregValues <- cbind(rnorm(60, 10, 1), rnorm(60))
    y <- 5 + 2*xregValues[,1] + cumsum(rnorm(60, 0, 0.5))
    orders <- list(ar=1, i=1, ma=0)
    testModel <- auto.msarima(y, orders=orders, lags=1, xreg=xregValues[,1])
    expect_equal(colnames(testModel$data), c("y","x"))
    expect_equal(as.vector(actuals(testModel)), y)
    expect_equal(as.vector(testModel$data[,"x"]), xregValues[,1])
    expect_match(testModel$model, "ARIMAX", fixed=TRUE)
    testModel <- auto.msarima(y, orders=orders, lags=1, xreg=xregValues)
    expect_equal(colnames(testModel$data), c("y","x1","x2"))
    testModel <- auto.msarima(y, orders=orders, lags=1, xreg=data.frame(a=xregValues[,1], b=xregValues[,2]))
    expect_equal(colnames(testModel$data), c("y","a","b"))
    expect_equal(names(testModel$initial$xreg), c("a","b"))
    # The names of the columns of a matrix too, in msarima() and ssarima() as well
    colnames(xregValues) <- c("a","b")
    for(modelFunction in list(auto.msarima, msarima, ssarima)){
        testModel <- modelFunction(y, orders=orders, lags=1, xreg=xregValues)
        expect_equal(colnames(testModel$data), c("y","a","b"))
    }
})

#### Parallel calculations ####
test_that("auto.adam() in parallel selects the model of the sequential selection", {
    skip_on_cran()
    skip_if_not_installed("foreach")
    skip_if_not(requireNamespace("doMC", quietly=TRUE) || requireNamespace("doParallel", quietly=TRUE))
    y <- BJsales[1:60]
    testModel <- auto.adam(y, "ANN", distribution=c("dnorm","dlaplace"))
    capture.output(parallelModel <- auto.adam(y, "ANN", distribution=c("dnorm","dlaplace"), parallel=2))
    foreach::registerDoSEQ()
    expect_equal(parallelModel$distribution, testModel$distribution)
    expect_equal(coef(parallelModel), coef(testModel))
    expect_equal(parallelModel$ICs, testModel$ICs)
})

test_that("adam_setupParallel() sets up the cores, or says which package is missing", {
    expect_equal(adam_setupParallel(FALSE, 3),
                 list(parallel=FALSE, cluster=NULL, nCores=min(parallel::detectCores()-1, 3)))
    # The function with the installed packages and the system replaced
    setupMocked <- function(installed, sysname="Linux"){
        setupFunction <- adam_setupParallel
        environment(setupFunction) <- list2env(list(requireNamespace=function(package, ...){
                                                        return(any(package==installed))
                                                    },
                                                    Sys.info=function(){return(c(sysname=sysname))}),
                                               parent=environment(adam_setupParallel))
        return(setupFunction)
    }
    expect_error(setupMocked(c("parallel","doMC"))(2, 3), "'foreach' package must be installed")
    expect_error(setupMocked(c("foreach","doMC"))(2, 3), "'parallel' package must be installed")
    expect_error(setupMocked(c("foreach","parallel"))(2, 3),
                 "either 'doMC' (prefered) or 'doParallel' package", fixed=TRUE)
    expect_error(setupMocked(c("foreach","parallel","doMC"), "Windows")(2, 3),
                 "you need 'doParallel' package", fixed=TRUE)
    # doParallel sets up a cluster without doMC and on Windows
    skip_if_not_installed("foreach")
    skip_if_not_installed("doParallel")
    for(sysname in c("Linux","Windows")){
        expect_output(setupResult <- setupMocked(c("foreach","parallel","doParallel"), sysname)(1, 3),
                      "Setting up 1 clusters using 'doParallel'", fixed=TRUE)
        expect_s3_class(setupResult$cluster, "cluster")
        expect_equal(foreach::getDoParName(), "doParallelSNOW", label=sysname)
        adam_teardownParallel(setupResult$cluster)
    }
    foreach::registerDoSEQ()
    # doMC registers the cores without a cluster
    skip_if_not_installed("doMC")
    expect_equal(setupMocked(c("foreach","parallel","doMC"))(2, 3), list(parallel=TRUE, cluster=NULL, nCores=2))
    expect_equal(foreach::getDoParName(), "doMC")
    expect_equal(foreach::getDoParWorkers(), 2)
    foreach::registerDoSEQ()
})
