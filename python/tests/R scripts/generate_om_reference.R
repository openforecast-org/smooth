## generate_om_reference.R
##
## Generates per-scenario reference outputs for the Python OM class
## comparison tests. For each scenario we write:
##   * one CSV per vector quantity (fitted, forecast, residuals, coef, states)
##   * one CSV with named scalars (loss_value, loglik, aicc, alpha, beta, gamma, phi)
##   * a manifest (scenarios.csv) with the input file and the fit parameters
##
## Run from the smooth package root:
##   Rscript "python/tests/R scripts/generate_om_reference.R"
##
## All outputs land under python/tests/data/om/.

library(smooth)

OUT_DIR <- "python/tests/data/om"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ── Deterministic input series ──────────────────────────────────────────────
set.seed(41)
N <- 200
y <- rpois(N, 0.3)
write.csv(data.frame(y = y),
          file.path(OUT_DIR, "intermittent_demand.csv"),
          row.names = FALSE)

set.seed(7)
X <- matrix(rnorm(N * 2), nrow = N, ncol = 2,
            dimnames = list(NULL, c("x1", "x2")))
write.csv(as.data.frame(X),
          file.path(OUT_DIR, "intermittent_demand_X.csv"),
          row.names = FALSE)

set.seed(13)
ys <- rpois(N, 0.4)
write.csv(data.frame(y = ys),
          file.path(OUT_DIR, "intermittent_demand_seasonal.csv"),
          row.names = FALSE)

# ── Helpers ────────────────────────────────────────────────────────────────
write_vec <- function(scenario_dir, name, x) {
    write.csv(data.frame(value = as.numeric(x)),
              file.path(scenario_dir, paste0(name, ".csv")),
              row.names = FALSE)
}

write_scalars <- function(scenario_dir, kv) {
    df <- data.frame(name  = names(kv),
                     value = as.numeric(unlist(kv)),
                     stringsAsFactors = FALSE)
    write.csv(df, file.path(scenario_dir, "scalars.csv"), row.names = FALSE)
}

run_scenario <- function(name, model_obj, holdout_data = NULL) {
    cat(sprintf("== %s : %s loss=%.6f loglik=%.6f\n",
                name, model_obj$model, model_obj$lossValue, model_obj$logLik))
    sd <- file.path(OUT_DIR, name)
    dir.create(sd, showWarnings = FALSE, recursive = TRUE)

    write_vec(sd, "fitted",   as.numeric(model_obj$fitted))
    write_vec(sd, "residuals", as.numeric(model_obj$residuals))
    if (length(model_obj$B) > 0) {
        write_vec(sd, "coef", as.numeric(model_obj$B))
    } else {
        write.csv(data.frame(value = numeric(0)),
                  file.path(sd, "coef.csv"), row.names = FALSE)
    }
    existing_fc <- if (!is.null(model_obj$forecast) &&
                       length(as.numeric(model_obj$forecast)) > 0 &&
                       !all(is.na(as.numeric(model_obj$forecast))))
                       as.numeric(model_obj$forecast)
                   else NULL
    if (!is.null(existing_fc)) {
        write_vec(sd, "forecast", existing_fc)
    } else {
        fc <- tryCatch(as.numeric(forecast(model_obj, h = 10)$mean),
                       error = function(e) NULL)
        if (!is.null(fc) && length(fc) > 0 && all(is.finite(fc))) {
            write_vec(sd, "forecast", fc)
        }
    }

    persistence <- as.numeric(model_obj$persistence)
    persistence_names <- names(model_obj$persistence)

    scalars <- list(
        loss_value = model_obj$lossValue,
        loglik     = model_obj$logLik,
        aicc       = AICc(model_obj),
        nparam     = sum(model_obj$nParam[1, 1:4])
    )
    if (!is.null(persistence) && length(persistence) > 0) {
        for (i in seq_along(persistence)) {
            scalars[[persistence_names[i]]] <- persistence[i]
        }
    }
    if (!is.null(model_obj$phi)) {
        scalars[["phi"]] <- as.numeric(model_obj$phi)
    }
    write_scalars(sd, scalars)
}

# A scenario is included only when R produces a meaningful, finite log-lik.
# Several combinations (e.g. MAN with binary data, ARIMA with binary data)
# return the optimiser penalty value (>1e100) inside R itself; we skip those
# for the comparison harness because there is no genuine reference to match.

safe_run <- function(name, expr) {
    res <- tryCatch(expr, error = function(e) {
        cat(sprintf("** SKIP %s : R error: %s\n", name, conditionMessage(e)))
        return(NULL)
    })
    if (is.null(res)) return(invisible())
    if (!is.finite(res$lossValue) || res$lossValue >= 1e10) {
        cat(sprintf("** SKIP %s : R loss not finite (%s)\n", name, res$lossValue))
        return(invisible())
    }
    run_scenario(name, res)
}

# ── Group A: well-posed ETS shapes only ────────────────────────────────────
safe_run("g1_fixed_ann",
         om(y, model = "ANN", occurrence = "fixed", lags = c(1)))
for (mdl in c("ANN", "MNN", "AAN")) {
    for (occ in c("odds-ratio", "inverse-odds-ratio", "direct")) {
        name <- sprintf("g1_%s_%s",
                        gsub("-", "_", occ, fixed = TRUE),
                        tolower(mdl))
        safe_run(name,
                 om(y, model = mdl, occurrence = occ, lags = c(1)))
    }
}

# ── Group B: seasonal cases ────────────────────────────────────────────────
safe_run("g2_seasonal_ana",
         om(ts(ys, frequency = 12), model = "ANA",
            occurrence = "odds-ratio", lags = c(1, 12)))
safe_run("g2_seasonal_mnm",
         om(ts(ys, frequency = 12), model = "MNM",
            occurrence = "odds-ratio", lags = c(1, 12)))

# ── Group C: ARIMA (unblocked by Fix A — om_initial_transform no longer
#                    runs the probability transform on ARIMA initial states)
safe_run("g3_arima_100_or",
         om(y, model = "ANN", occurrence = "odds-ratio", lags = c(1),
            orders = list(ar = c(1), i = c(0), ma = c(0))))
safe_run("g3_arima_011_ior",
         om(y, model = "ANN", occurrence = "inverse-odds-ratio", lags = c(1),
            orders = list(ar = c(0), i = c(1), ma = c(1))))
safe_run("g3_arima_111_or",
         om(y, model = "ANN", occurrence = "odds-ratio", lags = c(1),
            orders = list(ar = c(1), i = c(1), ma = c(1))))

# ── Group D: explanatory regressors (unblocked by Fix A — xreg coefficients
#                    are no longer mauled by the probability transform)
df <- data.frame(y = y, x1 = X[, 1], x2 = X[, 2])
safe_run("g4_xreg_or",
         om(df, model = "ANN", occurrence = "odds-ratio", lags = c(1),
            formula = y ~ x1 + x2))
safe_run("g4_xreg_ior",
         om(df, model = "ANN", occurrence = "inverse-odds-ratio", lags = c(1),
            formula = y ~ x1 + x2))

# ── Group E: holdout (R handles these reliably) ────────────────────────────
safe_run("g5_holdout_or",
         om(y, model = "MNN", occurrence = "odds-ratio", lags = c(1),
            h = 10, holdout = TRUE))
safe_run("g5_holdout_fixed",
         om(y, model = "ANN", occurrence = "fixed", lags = c(1),
            h = 10, holdout = TRUE))

cat("\nDone. Outputs in", OUT_DIR, "\n")
