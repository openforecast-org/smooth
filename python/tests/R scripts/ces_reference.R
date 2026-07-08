## ces_reference.R
## Generates reference datasets and runs ces() / auto.ces() to produce
## ground-truth outputs for Python CES comparison tests.
##
## Run from the smooth package root:
##   Rscript "python/tests/R scripts/ces_reference.R"
##
## Outputs written to:  python/tests/data/ces_reference.json
##                      python/tests/data/ces_*.csv

library(smooth)
library(jsonlite)

OUT_DIR <- "python/tests/data"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

results <- list()

## ── Helper: extract all comparison fields from a CES model ──────────────────
extract_ces <- function(m, h) {
    fitted_vals <- as.numeric(m$fitted)
    residual_vals <- as.numeric(m$residuals)
    forecast_vals <- as.numeric(m$forecast)
    states_mat <- as.matrix(m$states)

    # ICs via logLik object
    ll <- m$logLik
    list(
        model_name   = m$model,
        seasonality  = m$seasonality,
        a_real       = Re(m$parameters$a),
        a_imag       = Im(m$parameters$a),
        b_real       = if (!is.null(m$parameters$b)) Re(m$parameters$b) else NULL,
        b_imag       = if (!is.null(m$parameters$b)) Im(m$parameters$b) else NULL,
        loss_value   = as.numeric(m$lossValue),
        logLik       = as.numeric(ll),
        nParam       = attr(ll, "df"),
        aic          = as.numeric(AIC(ll)),
        aicc         = as.numeric(AICc(ll)),
        bic          = as.numeric(BIC(ll)),
        bicc         = as.numeric(BICc(ll)),
        scale        = as.numeric(m$scale),
        B            = as.numeric(m$B),
        persistence  = as.numeric(m$persistence),
        fitted       = fitted_vals,
        residuals    = residual_vals,
        forecast     = forecast_vals,
        states_nrow  = nrow(states_mat),
        states_ncol  = ncol(states_mat),
        states_first_row = as.numeric(states_mat[1, ]),
        states_last_row  = as.numeric(states_mat[nrow(states_mat), ])
    )
}

## ── Dataset 1: AirPassengers ────────────────────────────────────────────────
y_air <- as.numeric(AirPassengers)
air_path <- file.path(OUT_DIR, "ces_airpassengers.csv")
write.csv(data.frame(y = y_air), air_path,
          row.names = FALSE)
y_air <- ts(read.csv(air_path)$y, frequency = 12)

## ── Dataset 2: Simulated non-seasonal ───────────────────────────────────────
set.seed(42)
y_nonseasonal <- 100 + cumsum(rnorm(120, 0.5, 3))
nonseasonal_path <- file.path(OUT_DIR, "ces_nonseasonal.csv")
write.csv(data.frame(y = y_nonseasonal), nonseasonal_path,
          row.names = FALSE)
y_nonseasonal <- read.csv(nonseasonal_path)$y

## ── Dataset 3: Simulated quarterly seasonal ─────────────────────────────────
set.seed(77)
n3 <- 80
trend3 <- seq(50, 90, length.out = n3)
seasonal3 <- rep(c(10, -5, 3, -8), n3 / 4)
y_quarterly <- trend3 + seasonal3 + rnorm(n3, 0, 2)
quarterly_path <- file.path(OUT_DIR, "ces_quarterly.csv")
write.csv(data.frame(y = y_quarterly), quarterly_path,
          row.names = FALSE)
y_quarterly <- ts(read.csv(quarterly_path)$y, frequency = 4)

h <- 12

## ── CES none on AirPassengers ───────────────────────────────────────────────
m_none_air <- ces(y_air, seasonality = "none", h = h,
                  holdout = TRUE, silent = TRUE)
cat("CES none Air:", m_none_air$model, "logLik:", logLik(m_none_air), "\n")
results[["none_airpassengers"]] <- c(
    list(data_file = "ces_airpassengers.csv",
         python_params = list(seasonality = "none", h = h, holdout = TRUE,
                              lags = list(12L))),
    extract_ces(m_none_air, h)
)

## ── CES simple on AirPassengers ─────────────────────────────────────────────
m_simple_air <- ces(y_air, seasonality = "simple", h = h,
                    holdout = TRUE, silent = TRUE)
cat("CES simple Air:", m_simple_air$model, "logLik:", logLik(m_simple_air), "\n")
results[["simple_airpassengers"]] <- c(
    list(data_file = "ces_airpassengers.csv",
         python_params = list(seasonality = "simple", h = h, holdout = TRUE,
                              lags = list(12L))),
    extract_ces(m_simple_air, h)
)

## ── CES partial on AirPassengers ────────────────────────────────────────────
m_partial_air <- ces(y_air, seasonality = "partial", h = h,
                     holdout = TRUE, silent = TRUE)
cat("CES partial Air:", m_partial_air$model, "logLik:", logLik(m_partial_air), "\n")
results[["partial_airpassengers"]] <- c(
    list(data_file = "ces_airpassengers.csv",
         python_params = list(seasonality = "partial", h = h, holdout = TRUE,
                              lags = list(12L))),
    extract_ces(m_partial_air, h)
)

## ── CES full on AirPassengers ───────────────────────────────────────────────
m_full_air <- ces(y_air, seasonality = "full", h = h,
                  holdout = TRUE, silent = TRUE)
cat("CES full Air:", m_full_air$model, "logLik:", logLik(m_full_air), "\n")
results[["full_airpassengers"]] <- c(
    list(data_file = "ces_airpassengers.csv",
         python_params = list(seasonality = "full", h = h, holdout = TRUE,
                              lags = list(12L))),
    extract_ces(m_full_air, h)
)

## ── CES none on non-seasonal data ───────────────────────────────────────────
m_none_ns <- ces(y_nonseasonal, seasonality = "none", h = 10,
                 holdout = TRUE, silent = TRUE)
cat("CES none nonseasonal:", m_none_ns$model, "logLik:", logLik(m_none_ns), "\n")
results[["none_nonseasonal"]] <- c(
    list(data_file = "ces_nonseasonal.csv",
         python_params = list(seasonality = "none", h = 10L, holdout = TRUE,
                              lags = list(1L))),
    extract_ces(m_none_ns, 10)
)

## ── CES none on quarterly data ──────────────────────────────────────────────
m_none_q <- ces(y_quarterly, seasonality = "none", h = 8,
                holdout = TRUE, silent = TRUE)
cat("CES none quarterly:", m_none_q$model, "logLik:", logLik(m_none_q), "\n")
results[["none_quarterly"]] <- c(
    list(data_file = "ces_quarterly.csv",
         python_params = list(seasonality = "none", h = 8L, holdout = TRUE,
                              lags = list(4L))),
    extract_ces(m_none_q, 8)
)

## ── CES simple on quarterly data ────────────────────────────────────────────
m_simple_q <- ces(y_quarterly, seasonality = "simple", h = 8,
                  holdout = TRUE, silent = TRUE)
cat("CES simple quarterly:", m_simple_q$model, "logLik:", logLik(m_simple_q), "\n")
results[["simple_quarterly"]] <- c(
    list(data_file = "ces_quarterly.csv",
         python_params = list(seasonality = "simple", h = 8L, holdout = TRUE,
                              lags = list(4L))),
    extract_ces(m_simple_q, 8)
)

## ── CES partial on quarterly data ───────────────────────────────────────────
m_partial_q <- ces(y_quarterly, seasonality = "partial", h = 8,
                   holdout = TRUE, silent = TRUE)
cat("CES partial quarterly:", m_partial_q$model, "logLik:", logLik(m_partial_q), "\n")
results[["partial_quarterly"]] <- c(
    list(data_file = "ces_quarterly.csv",
         python_params = list(seasonality = "partial", h = 8L, holdout = TRUE,
                              lags = list(4L))),
    extract_ces(m_partial_q, 8)
)

## ── CES full on quarterly data ──────────────────────────────────────────────
m_full_q <- ces(y_quarterly, seasonality = "full", h = 8,
                holdout = TRUE, silent = TRUE)
cat("CES full quarterly:", m_full_q$model, "logLik:", logLik(m_full_q), "\n")
results[["full_quarterly"]] <- c(
    list(data_file = "ces_quarterly.csv",
         python_params = list(seasonality = "full", h = 8L, holdout = TRUE,
                              lags = list(4L))),
    extract_ces(m_full_q, 8)
)

## ── AutoCES on AirPassengers ────────────────────────────────────────────────
m_auto_air <- auto.ces(y_air, h = h, holdout = TRUE, silent = TRUE)
cat("AutoCES Air:", m_auto_air$model, "seasonality:", m_auto_air$seasonality, "\n")

# Collect ICs for all seasonality types
auto_ics <- list()
for (s in c("none", "simple", "partial", "full")) {
    tryCatch({
        ms <- ces(y_air, seasonality = s, h = h, holdout = TRUE, silent = TRUE)
        auto_ics[[s]] <- as.numeric(AICc(logLik(ms)))
    }, error = function(e) {
        auto_ics[[s]] <<- NA
    })
}

results[["auto_airpassengers"]] <- c(
    list(data_file = "ces_airpassengers.csv",
         python_params = list(h = h, holdout = TRUE, lags = list(12L))),
    extract_ces(m_auto_air, h),
    list(auto_ics = auto_ics,
         selected_seasonality = m_auto_air$seasonality)
)

## ── AutoCES on quarterly data ───────────────────────────────────────────────
m_auto_q <- auto.ces(y_quarterly, h = 8,
                     holdout = TRUE, silent = TRUE)
cat("AutoCES quarterly:", m_auto_q$model, "seasonality:", m_auto_q$seasonality, "\n")

results[["auto_quarterly"]] <- c(
    list(data_file = "ces_quarterly.csv",
         python_params = list(h = 8L, holdout = TRUE, lags = list(4L))),
    extract_ces(m_auto_q, 8),
    list(selected_seasonality = m_auto_q$seasonality)
)

## ── CES with optimal initial on AirPassengers ───────────────────────────────
m_opt_air <- ces(y_air, seasonality = "partial", h = h,
                 holdout = TRUE, initial = "optimal", silent = TRUE)
cat("CES partial optimal Air:", m_opt_air$model, "logLik:", logLik(m_opt_air), "\n")
results[["partial_optimal_airpassengers"]] <- c(
    list(data_file = "ces_airpassengers.csv",
         python_params = list(seasonality = "partial", h = h, holdout = TRUE,
                              lags = list(12L), initial = "optimal")),
    extract_ces(m_opt_air, h)
)

## ── Write JSON ──────────────────────────────────────────────────────────────
json_path <- file.path(OUT_DIR, "ces_reference.json")
write(toJSON(results, pretty = TRUE, auto_unbox = TRUE, digits = 15), json_path)
cat("\nCES reference outputs written to:", json_path, "\n")
cat("Cases:", length(results), "\n")
