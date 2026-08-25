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

## Simulated-interval path, with and without a scale model
sim <- list()
for (d in c("dnorm", "dlaplace", "ds", "dgnorm")) {
    loc <- adam(y, "ANN", lags = 1, distribution = d, silent = TRUE)
    s   <- suppressWarnings(sm(loc))
    loc2 <- implant(loc, s)
    sf <- as.numeric(forecast(s, h = 12, interval = "none")$mean)
    # the de-biased scale R feeds to the simulator (R/adam.R:6388-6400)
    nP <- nparam(loc2) - loc2$nParam[1, 4]
    df <- nobs(loc2, all = FALSE) - nP
    if (df <= 0) df <- nobs(loc2, all = FALSE)
    obs <- nobs(loc2)
    sim[[d]] <- list(
        scale_forecast = sf,
        sim_scale = as.numeric(switch(d,
            "dlnorm" = , "dnorm" = (sf * obs / df)^0.5,
            "dgnorm" = ((sf^loc2$other$shape) * obs / df)^(1 / loc2$other$shape),
            sf * obs / df)),
        nparam_for_variance = as.integer(nP), df = as.integer(df), obs = as.integer(obs))
    cat(sprintf("  sim %-9s scale[1]=%.6f\n", d, sim[[d]]$sim_scale[1]))
}
write(toJSON(sim, auto_unbox = TRUE, digits = NA),
      file.path(OUT_DIR, "sm_simulated_reference.json"))
