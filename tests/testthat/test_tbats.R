context("Tests for tbats()")

orders0 <- list(ar=0, ma=0, select=FALSE)

# De Livera's rotation form with ADAM's ARMA(1,1), from the states at t=0
rotationForm <- function(eps, frequency, gammas, level, trend, phi, alpha, beta, s, sStar,
                         arPhi=0, maTheta=0, arma=0){
    y <- numeric(length(eps));
    for(t in seq_along(eps)){
        y[t] <- level + phi*trend + sum(s) + arma + eps[t];
        sNew <- cos(frequency)*s + sin(frequency)*sStar + gammas[,1]*eps[t];
        sStar <- -sin(frequency)*s + cos(frequency)*sStar + gammas[,2]*eps[t];
        s <- sNew;
        level <- level + phi*trend + alpha*eps[t];
        trend <- phi*trend + beta*eps[t];
        arma <- arPhi*arma + (arPhi+maTheta)*eps[t];
    }
    return(y);
}

test_that("the errors are those of De Livera's rotation form at the same parameters", {
    set.seed(11);
    tt <- 1:300;
    y <- ts(200 + 0.2*tt + 5*sin(2*pi*tt/7) + 3*cos(2*pi*tt/30.4375) + cumsum(rnorm(300, 0, 0.5)) +
                rnorm(300), frequency=7);
    fit <- tbats(y, lags=c(1, 7, 30.4375), harmonics=c(2, 1), trend="additive", lambda=1,
                 orders=list(ar=1, ma=1, select=FALSE), initial="optimal");
    B <- fit$B;
    seasonal <- fit$initial$seasonal;
    frequency <- 2*pi*seasonal$j/seasonal$period;
    gammas <- cbind(B[paste0("gamma1[", round(seasonal$period, 4), "]")],
                    B[paste0("gamma2[", round(seasonal$period, 4), "]")]);
    # The harmonic contributes s(t) = a sin(lambda t) + b cos(lambda t) to y_t at t=1,
    # so the rotation state before y_1 is s(1) and its pair s*(1)
    s1 <- seasonal$sin*sin(frequency) + seasonal$cos*cos(frequency);
    sStar1 <- seasonal$sin*cos(frequency) - seasonal$cos*sin(frequency);
    e <- as.numeric(residuals(fit));
    yRotation <- rotationForm(e, frequency, gammas, fit$initial$level, fit$initial$trend, 1,
                              B[["alpha"]], B[["beta"]], s1, sStar1,
                              B[["phi1[1]"]], B[["theta1[1]"]], fit$initial$arma);
    # lambda=1 transforms y into y-1
    expect_lt(max(abs(yRotation - (as.numeric(y)-1))), 1e-8);
});

test_that("backcasting reproduces a noise-free series with a fractional period", {
    tt <- 1:200;
    y <- ts(100 + 0.5*tt + 10*sin(2*pi*tt/7.3) + 4*cos(2*pi*tt/7.3) + 2*sin(4*pi*tt/7.3));
    fit <- tbats(y, lags=c(1, 7.3), harmonics=2, trend="additive", lambda=1, orders=orders0,
                 B=c(alpha=0.1, beta=0.01, `gamma1[7.3]`=0.01, `gamma2[7.3]`=0.01), maxeval=1);
    expect_lt(max(abs(residuals(fit))), 1e-8);
});

test_that("lambda=0 is the model of the logarithms", {
    y <- AirPassengers;
    fitLog <- tbats(y, harmonics=5, trend="additive", lambda=0, orders=orders0);
    fitLevel <- tbats(log(y), harmonics=5, trend="additive", lambda=1, orders=orders0,
                      B=fitLog$B, maxeval=1);
    expect_equal(as.numeric(logLik(fitLog)), as.numeric(logLik(fitLevel)) - sum(log(y)), tolerance=1e-8);
});

test_that("lambda is estimated in [0, 1] and falls back to 1 when it cannot be", {
    fit <- tbats(AirPassengers, harmonics=5, trend="additive", orders=orders0);
    expect_true(fit$lambda>=0 && fit$lambda<=1);
    expect_true(any(names(fit$B)=="lambda"));
    expect_warning(fitNegative <- tbats(AirPassengers-200, harmonics=5, trend="additive", orders=orders0),
                   "positive data");
    expect_equal(fitNegative$lambda, 1);
    expect_message(fitMSE <- tbats(AirPassengers, harmonics=5, trend="additive", orders=orders0, loss="MSE"),
                   "likelihood");
    expect_equal(fitMSE$lambda, 1);
});

test_that("all the distributions are fitted and the dgnorm shape is estimated", {
    for(distribution in c("dnorm","dlaplace","ds","dgnorm")){
        fit <- tbats(AirPassengers, harmonics=5, trend="additive", orders=orders0,
                     distribution=distribution);
        expect_true(is.finite(logLik(fit)));
    }
    expect_true(any(names(fit$B)=="shape"));
    fitShape <- tbats(AirPassengers, harmonics=5, trend="additive", orders=orders0,
                      distribution="dgnorm", shape=1.5);
    expect_false(any(names(fitShape$B)=="shape"));
});

test_that("the usual bounds keep the response to an error in [0, 1] over the cycle", {
    fit <- tbats(AirPassengers, harmonics=5, trend="additive", orders=orders0, bounds="usual");
    seasonal <- fit$initial$seasonal;
    frequency <- 2*pi*seasonal$j/seasonal$period;
    horizons <- 0:11;
    response <- fit$B[["alpha"]] + cos(outer(horizons, frequency)) %*% rep(fit$B[["gamma1[12]"]], 5) +
        sin(outer(horizons, frequency)) %*% rep(fit$B[["gamma2[12]"]], 5);
    expect_true(all(response>=0 & response<=1));
    expect_true(fit$B[["beta"]]<=fit$B[["alpha"]]);
});

test_that("coinciding harmonics are dropped and the ARMA lags are truncated and merged", {
    harmonics <- tbats_harmonics(c(24, 168), c(2, 8));
    expect_equal(nrow(harmonics), 2 + 7);
    expect_false(any(harmonics$period==168 & harmonics$j==7));
    armaSpec <- tbats_armaSpec(list(ar=c(1, 1, 2), ma=0), c(1, 7, 7.02));
    expect_equal(armaSpec$lags, c(1, 7));
    expect_equal(armaSpec$arOrders, c(1, 2));
});

test_that("a refit with model= reproduces the fit", {
    fit <- tbats(AirPassengers, harmonics=5, trend="damped", orders=list(ar=1, ma=1, select=FALSE),
                 distribution="dgnorm");
    refit <- tbats(AirPassengers, model=fit);
    expect_equal(as.numeric(logLik(refit)), as.numeric(logLik(fit)), tolerance=1e-10);
    expect_equal(as.numeric(fitted(refit)), as.numeric(fitted(fit)), tolerance=1e-10);
});

test_that("the admissible bounds (the default) keep the discount matrix stable", {
    for(initial in c("backcasting","optimal","two-stage")){
        fit <- tbats(AirPassengers, harmonics=5, trend="damped", orders=list(ar=1, ma=1, select=FALSE),
                     initial=initial);
        struct <- list(nETS=2, nHarmonics=5, harmonicRows=2+2*(1:5)-1);
        eigenValues <- tbats_eigens(fit$transition, matrix(fit$persistence), fit$measurement[1,], struct);
        expect_lte(max(eigenValues), 1+1e-10);
    }
});

test_that("two-stage starts from the backcasted fit and cannot end below it", {
    for(y in list(AirPassengers, BJsales)){
        fitBackcast <- tbats(y, trend="damped", orders=list(ar=1, ma=1, select=FALSE), initial="complete");
        fitTwoStage <- tbats(y, trend="damped", orders=list(ar=1, ma=1, select=FALSE), initial="two-stage");
        expect_gte(as.numeric(logLik(fitTwoStage)), as.numeric(logLik(fitBackcast)) - 1e-8);
    }
});

test_that("the forecasts are those of adam in the Box-Cox space transformed back", {
    fit <- tbats(AirPassengers, harmonics=5, trend="damped", orders=list(ar=1, ma=0, select=FALSE),
                 h=12, holdout=TRUE);
    expect_equal(as.numeric(forecast(fit, h=12)$mean), as.numeric(fit$forecast), tolerance=1e-8);
    for(interval in c("prediction","simulated")){
        set.seed(41);
        forecastBC <- forecast(tbats_boxCoxObject(fit), h=12, interval=interval);
        set.seed(41);
        forecastTBATS <- forecast(fit, h=12, interval=interval);
        expect_equal(as.numeric(forecastTBATS$lower),
                     tbats_boxCoxInverse(forecastBC$lower, fit$lambda), tolerance=1e-8);
        expect_true(all(forecastTBATS$lower<forecastTBATS$mean & forecastTBATS$mean<forecastTBATS$upper));
    }
    expect_error(forecast(fit, h=12, cumulative=TRUE), "Cumulative");
    predicted <- predict(fit, interval="prediction");
    expect_true(all(predicted$lower<predicted$upper));
});

test_that("the point likelihoods sum to the log-likelihood with the Jacobian", {
    fit <- tbats(AirPassengers, harmonics=5, trend="additive", orders=orders0);
    expect_equal(sum(pointLik(fit)), as.numeric(logLik(fit)), tolerance=1e-8);
});

test_that("reapply refits the model and reforecast produces the intervals", {
    fit <- tbats(AirPassengers, harmonics=5, trend="damped", orders=list(ar=1, ma=0, select=FALSE));
    # The tiny covariance returns the fitted values
    refitted <- reapply(fit, nsim=5, heuristics=1e-12);
    expect_lt(max(abs(refitted$refitted - as.numeric(fitted(fit)))), 1e-2);
    set.seed(41);
    refitted <- reapply(fit, nsim=20);
    expect_true(all(refitted$lambda>=0 & refitted$lambda<=1));
    for(interval in c("confidence","complete")){
        set.seed(41);
        forecasted <- forecast(fit, h=12, interval=interval, nsim=20);
        expect_true(all(forecasted$lower<=forecasted$upper));
    }
    expect_true(all(is.finite(vcov(fit))));
    expect_equal(nrow(confint(fit)), length(fit$B));
    simulated <- simulate(fit, nsim=3, seed=41);
    expect_equal(dim(simulated$data), c(length(actuals(fit)), 3));
});

test_that("the Hannan-Rissanen screen gives the estimates of each order", {
    set.seed(41);
    y <- as.numeric(arima.sim(list(ar=0.6, ma=0.3), 300));
    screen <- arimaHRSelectCpp(y, c(0, 1), c(0, 0), c(1, 12), 0, 2, 1, TRUE);
    expect_equal(dim(screen$innovations), c(300, 6));
    for(i in 1:nrow(screen$orders)){
        parameters <- arimaHRCpp(y, c(screen$orders[i,1], 1), c(screen$orders[i,2], 0), c(1, 12),
                                 TRUE, TRUE, numeric(0), c(1, 1), TRUE);
        expect_equal(screen$parameters[i, seq_along(parameters)], as.vector(parameters), tolerance=1e-12);
    }
});

test_that("the ARMA selection finds an AR(1) and falls back to no ARMA", {
    set.seed(41);
    tt <- 1:240;
    y <- ts(500 + 20*sin(2*pi*tt/12) + 10*cos(2*pi*tt/12) + arima.sim(list(ar=0.7), 240, sd=5), frequency=12);
    fit <- tbats(y, harmonics=1, trend="none");
    expect_equal(fit$orders$ar, 1);
    expect_equal(fit$orders$ma, 0);
    expect_equal(min(fit$ICs), as.numeric(AICc(fit)));
    refit <- tbats(y, model=fit);
    expect_equal(as.numeric(logLik(refit)), as.numeric(logLik(fit)), tolerance=1e-10);
    fitAir <- tbats(AirPassengers, harmonics=5, trend="additive");
    expect_equal(sum(fitAir$orders$ar+fitAir$orders$ma), 0);
});
