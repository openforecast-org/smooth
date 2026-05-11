## generate_omg_reference.R
##
## Generates per-scenario reference outputs for the Python OMG class
## comparison tests. For each scenario we write:
##   * scalars.csv  — loss_value, loglik, aicc
##   * fitted.csv   — combined probability vector
##   * coef.csv     — joint B = c(modelA$B, modelB$B)
##   * forecast.csv — combined forecast (scenarios with h > 0 only)
##
## Run from the smooth package root:
##   Rscript "python/tests/R scripts/generate_omg_reference.R"
##
## All outputs land under python/tests/data/omg/.

library(smooth)
# Load dev package if omg() is not in the installed version.
# In a docker container with pkgload available, run via:
#   LD_PRELOAD=/opt/conda/lib/libstdc++.so.6 R -e \
#     ".libPaths(c('/tmp/Rlib',.libPaths())); pkgload::load_all('.'); \
#      source('python/tests/R scripts/generate_omg_reference.R')"
# Or simply ensure pkgload::load_all('.') was called before sourcing this file.

OUT_DIR <- "python/tests/data/omg"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ── Shared input series (same seeds as generate_om_reference.R) ─────────────
set.seed(41)
N <- 200
y <- rpois(N, 0.3)

set.seed(7)
X <- matrix(rnorm(N * 2), nrow = N, ncol = 2,
            dimnames = list(NULL, c("x1", "x2")))

set.seed(13)
ys <- rpois(N, 0.4)

# ── Helpers ─────────────────────────────────────────────────────────────────
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

run_scenario <- function(name, model_obj) {
    cat(sprintf("== %s : %s  loss=%.6f  loglik=%.6f\n",
                name, model_obj$model, model_obj$lossValue, model_obj$logLik))
    sd <- file.path(OUT_DIR, name)
    dir.create(sd, showWarnings = FALSE, recursive = TRUE)

    write_vec(sd, "fitted",    as.numeric(model_obj$fitted))
    # residuals = ot - fitted; R's omg() doesn't store them directly
    ot <- as.numeric(model_obj$modelA$data != 0)
    write_vec(sd, "residuals", ot - as.numeric(model_obj$fitted))

    # Joint coef = c(B_A, B_B) — mirrors Python's OMG.coef
    B_A <- if (length(model_obj$modelA$B) > 0) as.numeric(model_obj$modelA$B) else numeric(0)
    B_B <- if (length(model_obj$modelB$B) > 0) as.numeric(model_obj$modelB$B) else numeric(0)
    write_vec(sd, "coef", c(B_A, B_B))

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

    scalars <- list(
        loss_value = model_obj$lossValue,
        loglik     = model_obj$logLik,
        aicc       = AICc(model_obj),
        nparam     = model_obj$nParam[1, 1]
    )
    write_scalars(sd, scalars)
}

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

# ── Group A: basic ETS shapes ────────────────────────────────────────────────
safe_run("h1_mnn_mnn",
         omg(y, modelA = "MNN", modelB = "MNN", lags = c(1)))
safe_run("h1_ann_mnn",
         omg(y, modelA = "ANN", modelB = "MNN", lags = c(1)))
safe_run("h1_ann_ann",
         omg(y, modelA = "ANN", modelB = "ANN", lags = c(1)))
safe_run("h1_aan_mnn",
         omg(y, modelA = "AAN", modelB = "MNN", lags = c(1)))

# ── Group B: seasonal ────────────────────────────────────────────────────────
safe_run("h2_seasonal_ana_mnm",
         omg(ts(ys, frequency = 12), modelA = "ANA", modelB = "MNM",
             lags = c(1, 12)))
safe_run("h2_seasonal_mnn_mnn",
         omg(ts(ys, frequency = 12), modelA = "MNN", modelB = "MNN",
             lags = c(1, 12)))

# ── Group C: holdout ─────────────────────────────────────────────────────────
safe_run("h3_holdout_mnn",
         omg(y, modelA = "MNN", modelB = "MNN", lags = c(1),
             h = 10, holdout = TRUE))
safe_run("h3_holdout_ann",
         omg(y, modelA = "ANN", modelB = "ANN", lags = c(1),
             h = 10, holdout = TRUE))

cat("\nDone. Outputs in", OUT_DIR, "\n")
