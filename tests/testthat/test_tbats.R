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
    # The cumulative forecasts come from the paths in the space of the data
    expect_equal(as.numeric(forecast(fit, h=12, cumulative=TRUE)$mean), sum(fit$forecast),
                 tolerance=1e-8);
    set.seed(41);
    cumulative <- forecast(fit, h=12, cumulative=TRUE, interval="prediction", point="mean");
    expect_true(cumulative$lower<cumulative$mean && cumulative$mean<cumulative$upper);
    expect_equal(as.numeric(cumulative$mean), sum(forecast(fit, h=12, point="mean")$mean),
                 tolerance=1e-2);
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

test_that("confint stays inside the bounds of the model and the bootstrap refits it", {
    fit <- tbats(AirPassengers, harmonics=5, trend="damped", orders=list(ar=1, ma=1, select=FALSE));
    intervals <- confint(fit);
    expect_true(intervals["lambda",2]>=0 && intervals["lambda",3]<=1);
    expect_true(all(intervals[,2]<=coef(fit) & coef(fit)<=intervals[,3]));
    fitUsual <- tbats(AirPassengers, harmonics=4, trend="additive", orders=orders0, bounds="usual");
    intervalsUsual <- confint(fitUsual);
    expect_true(intervalsUsual["alpha",2]>=0 && intervalsUsual["alpha",3]<=1);
    set.seed(41);
    bootstrap <- coefbootstrap(fit, nsim=5);
    expect_equal(dim(bootstrap$coefficients), c(5, length(coef(fit))));
    expect_true(all(is.finite(bootstrap$vcov)));
});

test_that("the simulations start from the initials of the model", {
    for(initial in c("backcasting","optimal")){
        fit <- tbats(AirPassengers, harmonics=3, trend="damped", orders=list(ar=1, ma=0, select=FALSE),
                     initial=initial);
        simulated <- simulate(fit, nsim=1, seed=41);
        fittedFirst <- tbats_boxCox(simulated$data[1], fit$lambda) - simulated$residuals[1];
        expect_equal(as.numeric(fittedFirst), tbats_boxCox(as.numeric(fitted(fit)[1]), fit$lambda),
                     tolerance=1e-10);
    }
});

# A series with two regressors, of which the second is noise, and their future values
set.seed(41);
xregTest <- cbind(x1=rnorm(132, 10, 2), x2=rnorm(132));
yXreg <- ts(200 + 20*sin(2*pi*(1:132)/12) + 5*xregTest[,"x1"] + cumsum(rnorm(132)) + rnorm(132),
            frequency=12);

test_that("with regressors, TBATS is ADAM's ETSX at the same parameters", {
    y <- as.numeric(yXreg[1:120]);
    xreg <- xregTest[1:120,];
    fitADAM <- adam(cbind(y=y, xreg), "ANN", formula=y~x1+x2, initial="optimal");
    # TBATS keeps the deviations of the level and coefficients from the global model,
    # and lambda=1 transforms y into y-1
    beta <- qr.coef(qr(tbats_design(120, FALSE, tbats_harmonics(numeric(0), numeric(0)), xreg)), y-1);
    B <- c(alpha=coef(fitADAM)[["alpha"]], level=coef(fitADAM)[["level"]]-1-beta[[1]],
           x1=coef(fitADAM)[["x1"]]-beta[[2]], x2=coef(fitADAM)[["x2"]]-beta[[3]]);
    fit <- tbats(y, lags=1, xreg=xreg, trend="none", lambda=1, orders=orders0, initial="optimal",
                 B=B, maxeval=1);
    expect_equal(as.numeric(logLik(fit)), as.numeric(logLik(fitADAM)), tolerance=1e-10);
    expect_equal(as.numeric(fitted(fit)), as.numeric(fitted(fitADAM)), tolerance=1e-10);
});

test_that("lambda=0 with a regressor is the model of the logarithms with it", {
    fitLog <- tbats(yXreg[1:120], lags=c(1,12), xreg=xregTest[1:120,], harmonics=1, trend="none",
                    lambda=0, orders=orders0);
    fitLevel <- tbats(log(yXreg[1:120]), lags=c(1,12), xreg=xregTest[1:120,], harmonics=1, trend="none",
                      lambda=1, orders=orders0, B=coef(fitLog), maxeval=1);
    expect_equal(as.numeric(logLik(fitLog)), as.numeric(logLik(fitLevel)) - sum(log(yXreg[1:120])),
                 tolerance=1e-8);
});

test_that("the regressors are estimated and their future values used in the forecasts", {
    fit <- tbats(yXreg, xreg=xregTest, harmonics=1, trend="none", orders=orders0, lambda=1,
                 h=12, holdout=TRUE);
    expect_equal(names(fit$initial$xreg), c("x1","x2"));
    expect_equal(fit$initial$xreg[["x1"]], 5, tolerance=0.1);
    expect_match(fit$model, "^TBATSX");
    expect_equal(as.numeric(forecast(fit, h=12)$mean), as.numeric(fit$forecast), tolerance=1e-10);
    expect_equal(as.numeric(forecast(fit, h=12, newdata=unname(xregTest[121:132,]))$mean),
                 as.numeric(fit$forecast), tolerance=1e-10);
    expect_warning(forecast(fit, h=24), "newdata");
    expect_equal(sum(pointLik(fit)), as.numeric(logLik(fit)), tolerance=1e-8);
    refit <- tbats(yXreg, model=fit, h=12, holdout=TRUE);
    expect_equal(as.numeric(logLik(refit)), as.numeric(logLik(fit)), tolerance=1e-10);
    expect_true(all(is.finite(vcov(fit))));
    set.seed(41);
    expect_equal(dim(coefbootstrap(fit, nsim=3)$coefficients), c(3, length(coef(fit))));
    set.seed(41);
    forecasted <- forecast(fit, h=12, interval="complete", nsim=20);
    expect_true(all(forecasted$lower<fit$forecast & fit$forecast<forecasted$upper));
    expect_equal(dim(simulate(fit, nsim=2, seed=41)$data), c(120, 2));
    fitComplete <- tbats(yXreg, xreg=xregTest, harmonics=1, trend="none", orders=orders0, lambda=1,
                         initial="complete", h=12, holdout=TRUE);
    expect_false(any(names(coef(fitComplete)) %in% c("x1","x2")));
    expect_equal(nparam(fitComplete), nparam(fit));
});

test_that("adaptive regressors are ADAM's ETSX{D} and stay within the bounds", {
    set.seed(41);
    xreg <- cbind(x1=rnorm(150, 10, 2), x2=rnorm(150));
    y <- 200 + (5+cumsum(rnorm(150, 0, 0.05)))*xreg[,"x1"] - 2*xreg[,"x2"] + cumsum(rnorm(150)) + rnorm(150);
    fitADAM <- adam(cbind(y=y, xreg), "ANN", formula=y~x1+x2, initial="optimal", regressors="adapt");
    beta <- qr.coef(qr(tbats_design(150, FALSE, tbats_harmonics(numeric(0), numeric(0)), xreg)), y-1);
    parameters <- coef(fitADAM);
    B <- c(parameters[c("alpha","delta1","delta2")], level=parameters[["level"]]-1-beta[[1]],
           x1=parameters[["x1"]]-beta[[2]], x2=parameters[["x2"]]-beta[[3]]);
    fit <- tbats(y, lags=1, xreg=xreg, regressors="adapt", trend="none", lambda=1, orders=orders0,
                 initial="optimal", B=B, maxeval=1);
    expect_equal(as.numeric(logLik(fit)), as.numeric(logLik(fitADAM)), tolerance=1e-10);
    for(bounds in c("usual","admissible")){
        fit <- tbats(y, lags=1, xreg=xreg, regressors="adapt", trend="none", orders=orders0, bounds=bounds);
        deltas <- coef(fit)[c("delta1","delta2")];
        expect_match(fit$model, "\\{D\\}$");
        expect_true(all(deltas>=0 & deltas<=1));
    }
    refit <- tbats(y, lags=1, model=fit);
    expect_equal(as.numeric(logLik(refit)), as.numeric(logLik(fit)), tolerance=1e-10);
    # The averaged condition of adam() rejects a coefficient that explodes
    expect_false(fit$inBounds(replace(coef(fit), "delta1", 3)));
});

test_that("the selection keeps the relevant regressor and drops the noise", {
    set.seed(7);
    fit <- tbats(yXreg, xreg=cbind(xregTest, noise=rnorm(132)), regressors="select", h=12, holdout=TRUE);
    expect_equal(names(fit$initial$xreg), "x1");
    expect_equal(colnames(fit$data)[-1], "x1");
    expect_true(any(grepl("\\+X\\(x1\\)", names(fit$ICs))));
    expect_equal(as.numeric(forecast(fit, h=12)$mean), as.numeric(fit$forecast), tolerance=1e-10);
});

test_that("the point forecasts: the skeleton is the median, the mean by quadrature", {
    fit <- tbats(AirPassengers, harmonics=3, trend="additive", orders=orders0);
    skeleton <- forecast(fit, h=24)$mean;
    expect_equal(forecast(fit, h=24, point="median")$mean, skeleton);
    forecastedMean <- forecast(fit, h=24, point="mean")$mean;
    expect_true(all(forecastedMean > skeleton));
    # The quadrature is the mean of the simulated paths of the same distribution
    set.seed(41);
    paths <- forecast(tbats_boxCoxObject(fit), h=24, interval="simulated", nsim=100000,
                      scenarios=TRUE)$scenarios;
    expect_equal(as.numeric(forecastedMean), rowMeans(matrix(tbats_boxCoxInverse(paths, fit$lambda), 24)),
                 tolerance=1e-3);
    # lambda=0: the mean of the log-normal
    fit <- tbats(AirPassengers, harmonics=3, trend="additive", orders=orders0, lambda=0);
    bounds <- forecast(tbats_boxCoxObject(fit), h=12, interval="approximate", level=2*pnorm(1)-1);
    expect_equal(as.numeric(forecast(fit, h=12, point="mean")$mean),
                 as.numeric(exp(bounds$mean + (bounds$upper-bounds$mean)^2/2)));
    # With lambda=0 and the S distribution, the mean does not exist
    fit <- tbats(AirPassengers, harmonics=3, trend="additive", orders=orders0, lambda=0,
                 distribution="ds");
    set.seed(41);
    expect_warning(forecast(fit, h=3, point="mean"), "does not exist");
    set.seed(41);
    expect_true(all(forecast(fit, h=6, point="median", interval="complete", nsim=10)$mean>0));
});

# An intermittent demand: log-normal sizes with a weekly pattern, on 45% of the days
set.seed(41);
yIntermittent <- ts(exp(2 + 0.4*sin(2*pi*(1:365)/7) + rnorm(365, 0, 0.3))*rbinom(365, 1, 0.45),
                    frequency=7);

test_that("the occurrence: the sizes on the non-zero observations, the forecasts of the mixture", {
    expect_warning(tbats(yIntermittent, orders=orders0), "occurrence");
    fit <- tbats(yIntermittent, occurrence="odds-ratio", orders=orders0, h=14, holdout=TRUE);
    expect_equal(fit$nParam[1,3], nparam(fit$occurrence));
    expect_equal(sum(pointLik(fit)), as.numeric(logLik(fit)), tolerance=1e-8);
    # The sizes are log-normal
    expect_equal(fit$lambda, 0, tolerance=0.05);
    pForecast <- as.vector(forecast(fit$occurrence, h=14)$mean);
    skeleton <- forecast(tbats_boxCoxObject(fit), h=14)$mean;
    forecasted <- forecast(fit, h=14, interval="prediction");
    expect_equal(as.numeric(forecasted$mean), tbats_boxCoxInverse(skeleton, fit$lambda)*pForecast);
    expect_equal(as.numeric(forecasted$mean), as.numeric(fit$forecast));
    # The scale of the sizes is de-biased by the non-zero observations, as that of tbats
    objectBC <- tbats_boxCoxObject(fit);
    expect_equal(adam_dfScale(objectBC), adam_dfScale(fit));
    expect_equal(adam_varianceDebiased(objectBC),
                 fit$scale*sum(actuals(fit)!=0)/adam_dfScale(fit));
    # The probability of no demand is above 0.5: the median and the lower bound are zero
    expect_true(all(forecast(fit, h=14, point="median")$mean==0));
    expect_true(all(forecasted$lower==0) && all(forecasted$upper>forecasted$mean));
    expect_true(all(forecast(fit, h=14, point="mean")$mean>forecasted$mean));
    set.seed(41);
    expect_true(mean(simulate(fit, nsim=2)$data==0)>0.3);
    refit <- tbats(yIntermittent, model=fit, h=14, holdout=TRUE);
    expect_equal(as.numeric(logLik(refit)), as.numeric(logLik(fit)));
    # The cumulative forecasts of the mixture: the sum of the skeletons times the
    # probabilities, and the quantiles of the sums of the paths
    cumulative <- forecast(fit, h=14, cumulative=TRUE, interval="prediction");
    expect_equal(as.numeric(cumulative$mean), sum(forecasted$mean), tolerance=1e-8);
    expect_true(cumulative$lower<cumulative$mean && cumulative$mean<cumulative$upper);
    set.seed(41);
    reforecasted <- reforecast(fit, h=14, cumulative=TRUE, interval="prediction", nsim=20);
    expect_true(reforecasted$lower<reforecasted$upper);
});

test_that("the observations with missing regressors are dropped", {
    set.seed(41);
    xreg <- cbind(x1=rnorm(132, 10, 2));
    y <- ts(200 + 20*sin(2*pi*(1:132)/12) + 5*xreg[,1] + cumsum(rnorm(132)), frequency=12);
    xregNA <- xreg;
    xregNA[c(15, 60)] <- NA;
    expect_warning(fit <- tbats(y, xreg=xregNA, harmonics=1, trend="none", orders=orders0),
                   "missing values");
    yNA <- y;
    yNA[c(15, 60)] <- NA;
    fitNA <- suppressWarnings(tbats(yNA, xreg=xreg, harmonics=1, trend="none", orders=orders0));
    expect_equal(as.numeric(logLik(fit)), as.numeric(logLik(fitNA)));
    expect_equal(coef(fit), coef(fitNA));
    expect_true(all(is.na(fitted(fit)[c(15, 60)])));
    expect_error(suppressWarnings(tbats(y[1:120], xreg=xregNA[c(1:130, NA, NA),,drop=FALSE],
                                        h=12)), "horizon");
})

test_that("tbats takes the missing values for gaps", {
    y <- AirPassengers
    y[c(10, 50, 51, 90)] <- NA
    testModel <- suppressWarnings(tbats(y, lags=c(1,12)))
    expect_equal(attr(logLik(testModel), "nobs"), sum(!is.na(y)))
    expect_equal(as.numeric(logLik(testModel)), sum(pointLik(testModel)))
    expect_true(all(is.na(residuals(testModel)[is.na(y)])))
    expect_true(all(is.finite(fitted(testModel))))
    expect_equal(AICc(testModel), AICc(logLik(testModel)))
    expect_true(all(is.finite(sqrt(diag(vcov(testModel))))))
    expect_true(all(is.finite(forecast(testModel, h=12, interval="prediction")$upper)))
    # With an occurrence model, the missing values are neither zeros nor demand
    set.seed(7)
    y <- ts(exp(2 + rnorm(300, 0, 0.3))*rbinom(300, 1, 0.7), frequency=7)
    y[c(20:25, 150)] <- NA
    testModel <- suppressWarnings(tbats(y, occurrence="odds-ratio"))
    expect_equal(attr(logLik(testModel), "nobs"), sum(!is.na(y)))
    expect_equal(as.numeric(logLik(testModel)), sum(pointLik(testModel)))
})

test_that("initial='gradient' solves for the initials of the states", {
    fitBackcast <- tbats(AirPassengers, harmonics=5, trend="additive", orders=orders0);
    fitGradient <- tbats(AirPassengers, harmonics=5, trend="additive", orders=orders0, initial="gradient");
    # The initials are counted as with backcasting, and the likelihood is that of the fit
    expect_equal(nparam(fitGradient), nparam(fitBackcast));
    expect_equal(as.numeric(logLik(fitGradient)), sum(pointLik(fitGradient)));
    # At the same parameters, the solved initials fit at least as well as the backcast ones
    atBackcast <- tbats(AirPassengers, harmonics=5, trend="additive", orders=orders0, initial="gradient",
                        B=fitBackcast$B, maxeval=1);
    expect_lte(atBackcast$lossValue, fitBackcast$lossValue);
    expect_true(all(is.finite(forecast(fitGradient, h=12, interval="prediction")$upper)));
})

test_that("tbats() takes a custom loss", {
    lossMSE <- function(actual, fitted, B){
        return(mean((actual-fitted)^2));
    }
    fitCustom <- tbats(AirPassengers, harmonics=5, trend="additive", orders=orders0, loss=lossMSE);
    fitMSE <- tbats(AirPassengers, harmonics=5, trend="additive", orders=orders0, loss="MSE");
    expect_equal(fitCustom$loss, "custom");
    expect_equal(fitCustom$lossValue, fitMSE$lossValue);
    expect_equal(fitCustom$B, fitMSE$B);
    # The refits take the function back
    expect_equal(tbats(AirPassengers, model=fitCustom)$lossValue, fitCustom$lossValue);
    expect_s3_class(coefbootstrap(fitCustom, nsim=3), "bootstrap");
})

test_that("rmultistep() of tbats() gives the errors of the transformed data", {
    fit <- tbats(AirPassengers, lags=c(1,12), harmonics=3, trend="none",
                 orders=list(ar=0, ma=0, select=FALSE));
    errors <- rmultistep(fit, h=3);
    expect_equal(dim(errors), c(nobs(fit)-3, 3));
    # Of the logarithms, not of the passengers
    expect_lt(max(abs(errors)), 1);
    expect_true(all(is.finite(multicov(fit, type="empirical", h=3))));
});
