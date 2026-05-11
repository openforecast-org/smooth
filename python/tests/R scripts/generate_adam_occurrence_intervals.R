#!/usr/bin/env Rscript
# Generate ADAM occurrence-interval reference data for Python parity tests.
#
# Outputs (written to python/tests/data/adam_occ_intervals/):
#   approximate_ann_or_upper.csv   upper bounds, 95 % CI, approximate
#   approximate_ann_or_lower.csv   lower bounds, 95 % CI, approximate
#   approximate_ann_or_mean.csv    point forecasts (== demand * p_forecast)
#   simulated_ann_or_upper.csv     upper bounds, 95 % CI, nsim=100000
#   simulated_ann_or_lower.csv     lower bounds, 95 % CI, nsim=100000

library(smooth)
set.seed(42)

out_dir <- "python/tests/data/adam_occ_intervals"
if (!dir.exists(out_dir)) {
    dir.create(out_dir, recursive = TRUE)
}

y <- read.csv("python/tests/data/om/intermittent_demand.csv")$y
h <- 10L

# ---- Approximate ----
m_approx <- adam(
    y,
    model    = "ANN",
    lags     = c(1),
    occurrence = "o",
    silent   = TRUE
)
fc_approx <- forecast(m_approx, h = h, interval = "approximate", level = 0.95)

write.csv(
    data.frame(upper = as.numeric(fc_approx$upper)),
    file = file.path(out_dir, "approximate_ann_or_upper.csv"),
    row.names = FALSE
)
write.csv(
    data.frame(lower = as.numeric(fc_approx$lower)),
    file = file.path(out_dir, "approximate_ann_or_lower.csv"),
    row.names = FALSE
)
write.csv(
    data.frame(mean = as.numeric(fc_approx$mean)),
    file = file.path(out_dir, "approximate_ann_or_mean.csv"),
    row.names = FALSE
)

cat("Approximate: upper =", as.numeric(fc_approx$upper), "\n")
cat("Approximate: lower =", as.numeric(fc_approx$lower), "\n")
cat("Approximate: mean  =", as.numeric(fc_approx$mean),  "\n")

# ---- Simulated (nsim = 100000 for Monte Carlo convergence) ----
set.seed(42)
m_sim <- adam(
    y,
    model    = "ANN",
    lags     = c(1),
    occurrence = "o",
    silent   = TRUE
)
fc_sim <- forecast(m_sim, h = h, interval = "simulated", level = 0.95, nsim = 100000)

write.csv(
    data.frame(upper = as.numeric(fc_sim$upper)),
    file = file.path(out_dir, "simulated_ann_or_upper.csv"),
    row.names = FALSE
)
write.csv(
    data.frame(lower = as.numeric(fc_sim$lower)),
    file = file.path(out_dir, "simulated_ann_or_lower.csv"),
    row.names = FALSE
)

cat("Simulated: upper =", as.numeric(fc_sim$upper), "\n")
cat("Simulated: lower =", as.numeric(fc_sim$lower), "\n")
cat("Done. Files written to", out_dir, "\n")
