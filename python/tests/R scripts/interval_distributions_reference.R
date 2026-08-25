## interval_distributions_reference.R
## Prediction intervals for every error distribution, additive and
## multiplicative error, as ground truth for the Python port.
##
##   Rscript "python/tests/R scripts/interval_distributions_reference.R"

library(smooth); library(jsonlite)
OUT <- "python/tests/data"
y <- read.csv(file.path(OUT, "sm_positive.csv"))$y
res <- list()
for (mo in c("ANN", "MNN")) {
    for (d in c("dnorm","dlaplace","ds","dgnorm","dlnorm","dgamma","dinvgauss")) {
        m <- try(adam(y, mo, lags=1, distribution=d, silent=TRUE), silent=TRUE)
        if (inherits(m, "try-error")) next
        f <- forecast(m, h=8, interval="prediction", level=0.95)
        res[[paste0(mo,"_",d)]] <- list(
            mean=as.numeric(f$mean), lower=as.numeric(f$lower),
            upper=as.numeric(f$upper), sigma=as.numeric(sigma(m)))
        cat(sprintf("  %s %-10s lower[1]=%10.4f upper[1]=%10.4f\n", mo, d, f$lower[1], f$upper[1]))
    }
}
write(toJSON(res, auto_unbox=TRUE, digits=NA), file.path(OUT, "interval_distributions_reference.json"))
cat("wrote", length(res), "cases\n")
