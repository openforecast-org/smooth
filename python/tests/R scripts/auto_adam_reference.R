## auto_adam_reference.R
## Generates reference datasets and runs auto.adam() to produce ground-truth outputs
## for Python AutoADAM comparison tests.
##
## Run from the smooth package root:
##   Rscript "python/tests/R scripts/auto_adam_reference.R"
##
## Outputs written to:  python/tests/data/auto_adam_reference.json
##                      python/tests/data/ref_*.csv

library(smooth)
library(jsonlite)

OUT_DIR <- "python/tests/data"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

results <- list()

## Helper: safely extract integer ARIMA orders from fitted model
get_ar <- function(m) { if(is.list(m$orders)) as.integer(m$orders$ar) else 0L }
get_i  <- function(m) { if(is.list(m$orders)) as.integer(m$orders$i)  else 0L }
get_ma <- function(m) { if(is.list(m$orders)) as.integer(m$orders$ma) else 0L }

## ── Case 1: Pure ARIMA, non-seasonal ─────────────────────────────────────────
set.seed(42)
y1 <- cumsum(rnorm(100))
write.csv(data.frame(y = y1), file.path(OUT_DIR, "ref_nnn_nonseasonal.csv"),
          row.names = FALSE)

m1 <- auto.adam(y1, model = "NNN", lags = c(1),
                orders = list(ar = c(2), i = c(2), ma = c(2), select = TRUE),
                distribution = c("dnorm", "dlaplace", "ds"),
                ic = "AICc", silent = TRUE)
cat("Case 1:", modelType(m1), "dist:", m1$distribution, "AICc:", AICc(m1), "\n")
cat("  AR:", get_ar(m1), "I:", get_i(m1), "MA:", get_ma(m1), "\n")

results[["nnn_nonseasonal"]] <- list(
    data_file = "ref_nnn_nonseasonal.csv",
    python_params = list(
        model = "NNN", lags = list(1L),
        ar_order = 2L, i_order = 2L, ma_order = 2L,
        arima_select = TRUE,
        distribution = list("dnorm", "dlaplace", "ds"),
        ic = "AICc"
    ),
    model_name   = modelType(m1),
    distribution = m1$distribution,
    ar_orders    = get_ar(m1),
    i_orders     = get_i(m1),
    ma_orders    = get_ma(m1),
    aicc         = as.numeric(AICc(m1))
)

## ── Case 2: Pure ARIMA, seasonal ─────────────────────────────────────────────
set.seed(123)
seasonal_comp <- rep(sin(seq(0, 2 * pi, length.out = 13)[-13]), 20)
y2 <- cumsum(rnorm(240)) + seasonal_comp * 5
write.csv(data.frame(y = y2), file.path(OUT_DIR, "ref_nnn_seasonal.csv"),
          row.names = FALSE)

m2 <- auto.adam(y2, model = "NNN", lags = c(1, 12),
                orders = list(ar = c(2, 2), i = c(2, 1), ma = c(2, 2), select = TRUE),
                distribution = c("dnorm", "dlaplace"),
                ic = "AICc", silent = TRUE)
cat("Case 2:", modelType(m2), "dist:", m2$distribution, "AICc:", AICc(m2), "\n")
cat("  AR:", get_ar(m2), "I:", get_i(m2), "MA:", get_ma(m2), "\n")

results[["nnn_seasonal"]] <- list(
    data_file = "ref_nnn_seasonal.csv",
    python_params = list(
        model = "NNN", lags = list(1L, 12L),
        ar_order = list(2L, 2L), i_order = list(2L, 1L), ma_order = list(2L, 2L),
        arima_select = TRUE,
        distribution = list("dnorm", "dlaplace"),
        ic = "AICc"
    ),
    model_name   = modelType(m2),
    distribution = m2$distribution,
    ar_orders    = get_ar(m2),
    i_orders     = get_i(m2),
    ma_orders    = get_ma(m2),
    aicc         = as.numeric(AICc(m2))
)

## ── Case 3: ETS only, no ARIMA, distribution selection ───────────────────────
set.seed(7)
trend3    <- seq(100, 130, length.out = 120)
seasonal3 <- rep(c(1, 1.1, 1.2, 1.1, 1.0, 0.9, 0.8, 0.9, 1.0, 1.0, 1.1, 1.2), 10)
y3 <- trend3 * seasonal3 * exp(rnorm(120, 0, 0.05))
write.csv(data.frame(y = y3), file.path(OUT_DIR, "ref_ets_only.csv"),
          row.names = FALSE)

m3 <- auto.adam(y3, model = "ZXZ", lags = c(12),
                orders = list(ar = c(0, 0), i = c(0, 0), ma = c(0, 0), select = FALSE),
                distribution = c("dnorm", "dlaplace", "dgamma"),
                ic = "AICc", silent = TRUE)
cat("Case 3:", modelType(m3), "dist:", m3$distribution, "AICc:", AICc(m3), "\n")

results[["ets_only"]] <- list(
    data_file = "ref_ets_only.csv",
    python_params = list(
        model = "ZXZ", lags = list(12L),
        ar_order = 0L, i_order = 0L, ma_order = 0L,
        arima_select = FALSE,
        distribution = list("dnorm", "dlaplace", "dgamma"),
        ic = "AICc"
    ),
    model_name   = modelType(m3),
    distribution = m3$distribution,
    ar_orders    = 0L,
    i_orders     = 0L,
    ma_orders    = 0L,
    aicc         = as.numeric(AICc(m3))
)

## ── Case 4: ETS + ARIMA selection, non-seasonal ──────────────────────────────
set.seed(99)
y4 <- 100 + seq(0, 20, length.out = 120) + rnorm(120, 0, 3)
write.csv(data.frame(y = y4), file.path(OUT_DIR, "ref_ets_arima.csv"),
          row.names = FALSE)

m4 <- auto.adam(y4, model = "AAN", lags = c(1),
                orders = list(ar = c(2), i = c(2), ma = c(2), select = TRUE),
                distribution = c("dnorm", "dlaplace"),
                ic = "AICc", silent = TRUE)
cat("Case 4:", modelType(m4), "dist:", m4$distribution, "AICc:", AICc(m4), "\n")
cat("  AR:", get_ar(m4), "I:", get_i(m4), "MA:", get_ma(m4), "\n")

results[["ets_arima"]] <- list(
    data_file = "ref_ets_arima.csv",
    python_params = list(
        model = "AAN", lags = list(1L),
        ar_order = 2L, i_order = 2L, ma_order = 2L,
        arima_select = TRUE,
        distribution = list("dnorm", "dlaplace"),
        ic = "AICc"
    ),
    model_name   = modelType(m4),
    distribution = m4$distribution,
    ar_orders    = get_ar(m4),
    i_orders     = get_i(m4),
    ma_orders    = get_ma(m4),
    aicc         = as.numeric(AICc(m4))
)

## ── Case 5: ETSX with xreg ───────────────────────────────────────────────────
xreg_data <- read.csv(file.path(OUT_DIR, "etsx_data.csv"))

m5 <- auto.adam(xreg_data, model = "AAN",
                orders = list(ar = c(0), i = c(0), ma = c(0), select = FALSE),
                distribution = c("dnorm", "dlaplace"),
                regressors = "use",
                ic = "AICc", silent = TRUE)
cat("Case 5:", modelType(m5), "dist:", m5$distribution, "AICc:", AICc(m5), "\n")

results[["etsx"]] <- list(
    data_file = "etsx_data.csv",
    python_params = list(
        model = "AAN",
        ar_order = 0L, i_order = 0L, ma_order = 0L,
        arima_select = FALSE,
        distribution = list("dnorm", "dlaplace"),
        regressors = "use",
        ic = "AICc"
    ),
    model_name   = modelType(m5),
    distribution = m5$distribution,
    ar_orders    = 0L,
    i_orders     = 0L,
    ma_orders    = 0L,
    aicc         = as.numeric(AICc(m5))
)

## ── Case 6: ETS + ARIMA selection, seasonal ──────────────────────────────────
set.seed(55)
trend6    <- seq(50, 100, length.out = 144)
seasonal6 <- rep(sin(seq(0, 2 * pi, length.out = 13)[-13]) * 10, 12)
y6 <- trend6 + seasonal6 + rnorm(144, 0, 2)
write.csv(data.frame(y = y6), file.path(OUT_DIR, "ref_ets_arima_seasonal.csv"),
          row.names = FALSE)

m6 <- auto.adam(y6, model = "ZXZ", lags = c(12),
                orders = list(ar = c(2, 2), i = c(2, 1), ma = c(2, 2), select = TRUE),
                distribution = c("dnorm", "dlaplace"),
                ic = "AICc", silent = TRUE)
cat("Case 6:", modelType(m6), "dist:", m6$distribution, "AICc:", AICc(m6), "\n")
cat("  AR:", get_ar(m6), "I:", get_i(m6), "MA:", get_ma(m6), "\n")

results[["ets_arima_seasonal"]] <- list(
    data_file = "ref_ets_arima_seasonal.csv",
    python_params = list(
        model = "ZXZ", lags = list(12L),
        ar_order = list(2L, 2L), i_order = list(2L, 1L), ma_order = list(2L, 2L),
        arima_select = TRUE,
        distribution = list("dnorm", "dlaplace"),
        ic = "AICc"
    ),
    model_name   = modelType(m6),
    distribution = m6$distribution,
    ar_orders    = get_ar(m6),
    i_orders     = get_i(m6),
    ma_orders    = get_ma(m6),
    aicc         = as.numeric(AICc(m6))
)

## ── Write JSON ────────────────────────────────────────────────────────────────
json_path <- file.path(OUT_DIR, "auto_adam_reference.json")
write(toJSON(results, pretty = TRUE, auto_unbox = TRUE), json_path)
cat("\nReference outputs written to:", json_path, "\n")
