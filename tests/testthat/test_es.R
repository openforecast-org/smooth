context("Tests for es() function");

# Basic ETS selection
testModel <- es(BJsales, silent=TRUE);
test_that("Test ETS selection on BJsales", {
    expect_equal(length(testModel$ICs), 5);
})

test_that("Test damped-trend ETS on BJsales", {
    expect_equal(es(BJsales,model="AAdN", silent=TRUE)$phi, 0.88, tolerance=0.1);
})

# Reuse previous ETS
test_that("Test on BJsales, predefined ETS", {
    expect_equal(es(BJsales, model=testModel, silent=TRUE)$cf, testModel$cf);
})

# Test combinations of ETS
test_that("Test ETS(CCC) with BIC on AirPassengers", {
    skip_on_cran
    testModel <- es(AirPassengers, "CCC", silent=TRUE, ic="BIC");
    expect_equal(testModel$scale, mean(residuals(testModel)^2));
})

# Test model selection of non-multiplicative trend ETS
test_that("Test ETS(MXM) with AIC on AirPassengers", {
    skip_on_cran()
    testModel <- es(AirPassengers, "MXM", silent=TRUE, ic="AIC");
    expect_match(testModel$loss, "likelihood");
})

# Test trace cost function for ETS
testModel <- es(AirPassengers, model="MAdM", h=18, holdout=TRUE, silent=TRUE)
test_that("Test AIC of ETS on AirPassengers", {
    expect_equal(as.numeric(round(AICc(testModel),2)), as.numeric(round(testModel$ICs,2)));
})

# Test how different passed values are accepted by ETS
test_that("Test initials, initialSeason and persistence of ETS on AirPassengers", {
    skip_on_cran()
    expect_equal(es(AirPassengers, model="MAdM", initial=testModel$initial, silent=TRUE)$initial, testModel$initial);
    expect_equal(es(AirPassengers, model="MAdM", persistence=testModel$persistence, silent=TRUE)$persistence, testModel$persistence);
    expect_equal(es(AirPassengers, model="MAdM", initialSeason=testModel$initialSeason, silent=TRUE)$initialSeason, testModel$initialSeason);
    expect_equal(es(AirPassengers, model="MAdM", phi=testModel$phi, silent=TRUE)$phi, testModel$phi);
})

x <- BJsales.lead;
y <- BJsales;
# Test selection of exogenous with ETS
test_that("Use exogenous variables for ETS on BJsales", {
    skip_on_cran()
    testModel <- es(y, h=18, holdout=TRUE, xreg=xregExpander(x), silent=TRUE, regressors="use")
    expect_equal(length(testModel$initial$xreg),3);
})

# Test combination of ETS with exogenous selection
test_that("Select exogenous variables for ETSX combined on BJsales", {
    skip_on_cran()
    testModel <- es(y, "CCC", h=18, holdout=TRUE, xreg=x, silent=TRUE, regressors="select")
    expect_match(modelType(testModel), "CCN");
})
