## sm_forecast_reference.R
## Ground truth for Phase 3: forecasts and prediction intervals from a model
## with an implanted scale model.
##
##   Rscript "python/tests/R scripts/sm_forecast_reference.R"

library(smooth); library(jsonlite)
OUT_DIR <- "python/tests/data"
y <- read.csv(file.path(OUT_DIR, "sm_positive.csv"))$y
results <- list()

for (d in c("dnorm", "dlaplace", "ds", "dgnorm", "dlnorm", "dgamma", "dinvgauss")) {
    loc <- adam(y, "ANN", lags = 1, distribution = d, silent = TRUE)
    s   <- suppressWarnings(sm(loc))
    loc2 <- implant(loc, s)

    # Analytical intervals (the default path for additive-error ETS)
    fa <- forecast(loc2, h = 12, interval = "prediction", level = 0.95)
    # The scale model's own forecast, which drives the time-varying sigma
    sf <- forecast(s, h = 12, interval = "none")$mean

    results[[d]] <- list(
        scale_forecast = as.numeric(sf),
        mean  = as.numeric(fa$mean),
        lower = as.numeric(fa$lower),
        upper = as.numeric(fa$upper),
        extract_scale_head = as.numeric(head(extractScale(loc2), 5)),
        logLik = as.numeric(loc2$logLik),
        nparam = as.integer(nparam(loc2))
    )
    cat(sprintf("  %-10s sf[1]=%.6f lower[1]=%.4f upper[1]=%.4f\n",
                d, sf[1], fa$lower[1], fa$upper[1]))
}
write(toJSON(results, auto_unbox = TRUE, digits = NA),
      file.path(OUT_DIR, "sm_forecast_reference.json"))
cat("wrote sm_forecast_reference.json -", length(results), "cases\n")
