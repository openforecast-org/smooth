context("Tests for sparma() function");

# SpARMA with basic orders
testModel <- sparma(BJsales, orders=c(2,1), silent=TRUE);
test_that("Test sparma(2,1)", {
    expect_match(testModel$model, "SpARMA\\(2;1\\)");
})

# SpARMA with basic orders and constant
testModel <- sparma(BJsales, orders=c(2,1), silent=TRUE, constant=TRUE);
test_that("Test sparma(2,1) with constant", {
    expect_true(!is.null(testModel$constant));
})

# SpARMA with vectors for orders
testModel <- sparma(BJsales, orders=list(ar=c(2,5,7), ma=c(1,3,6)),
                    silent=TRUE);
test_that("Test SpARMA(2,5,7;1,3,6)", {
    expect_match(testModel$model, "SpARMA\\(2,5,7;1,3,6\\)");
})

# SpARMA with zero AR orders
testModel <- sparma(BJsales, orders=list(ar=0, ma=c(1,3,6)),
                    silent=TRUE);
test_that("Test SpARMA with zero AR", {
    expect_match(testModel$model, "SpARMA\\(0;1,3,6\\)");
})

# SpARMA with zero AR orders
testModel <- sparma(BJsales, orders=list(ar=c(2,5,7), ma=0),
                    silent=TRUE);
test_that("Test SpARMA with zero MA", {
    expect_match(testModel$model, "SpARMA\\(2,5,7;0\\)");
})


test_that("sparma() takes the missing values for gaps", {
    y <- AirPassengers;
    y[c(10, 50, 51, 90)] <- NA;
    testModel <- suppressWarnings(sparma(y, orders=c(1,1)));
    expect_equal(attr(logLik(testModel), "nobs"), sum(!is.na(y)));
    expect_equal(as.numeric(logLik(testModel)), sum(pointLik(testModel)));
    expect_true(all(is.na(residuals(testModel)[is.na(y)])));
    expect_true(all(is.finite(forecast(testModel, h=12, interval="prediction")$upper)));
})
