## sm_reference.R
## Ground truth for the Python port of sm() -- the scale model for adam().
##
## Run from the smooth package root:
##   Rscript "python/tests/R scripts/sm_reference.R"
##
## Outputs:  python/tests/data/sm_reference.json
##           python/tests/data/sm_*.csv

library(smooth)
library(jsonlite)

OUT_DIR <- "python/tests/data"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

## ---- Data ----------------------------------------------------------------
set.seed(41)
n <- 120
trend <- 100 + 0.4 * seq_len(n)
seasonal <- 12 * sinpi(2 * seq_len(n) / 12)
y_pos <- trend + seasonal + rnorm(n, 0, 6)          # strictly positive, seasonal
write.csv(data.frame(y = y_pos), file.path(OUT_DIR, "sm_positive.csv"), row.names = FALSE)

set.seed(42)
y_int <- rpois(60, 1.2) * 1.0                        # intermittent
write.csv(data.frame(y = y_int), file.path(OUT_DIR, "sm_intermittent.csv"), row.names = FALSE)

## ---- Helpers -------------------------------------------------------------
vec <- function(x) if (is.null(x)) NULL else as.numeric(x)

extract_sm <- function(s, loc) {
    ll <- logLik(s)
    list(
        sm_model      = s$model,
        sm_logLik     = as.numeric(ll),
        sm_df         = as.integer(attr(ll, "df")),
        sm_df_stored  = as.integer(attr(s$logLik, "df")),
        sm_nparam     = as.integer(nparam(s)),
        sm_lossValue  = as.numeric(s$lossValue),
        sm_scale      = as.numeric(s$scale),
        sm_fitted     = vec(fitted(s)),
        sm_residuals  = vec(residuals(s)),
        sm_actuals    = vec(actuals(s)),
        sm_forecast   = vec(s$forecast),
        loc_logLik    = as.numeric(logLik(loc)),
        loc_nparam    = as.integer(nparam(loc)),
        loc_scale     = as.numeric(loc$scale)
    )
}

results <- list()
run <- function(id, y, locArgs, smArgs = list()) {
    loc <- try(do.call(adam, c(list(y), locArgs, list(silent = TRUE))), silent = TRUE)
    if (inherits(loc, "try-error")) {
        results[[id]] <<- list(error = paste("location:", as.character(loc))); return(invisible())
    }
    s <- try(suppressWarnings(do.call(sm, c(list(loc), smArgs))), silent = TRUE)
    if (inherits(s, "try-error")) {
        results[[id]] <<- list(error = paste("sm:", as.character(s))); return(invisible())
    }
    out <- extract_sm(s, loc)
    out$data <- if (identical(y, y_int)) "intermittent" else "positive"
    out$loc_model <- locArgs$model
    out$distribution <- locArgs$distribution
    results[[id]] <<- out
    cat(sprintf("  %-34s %-22s logLik=%12.6f df=%d\n", id, out$sm_model, out$sm_logLik, out$sm_df))
}

## ---- Cases ---------------------------------------------------------------
cat("Additive-error location models:\n")
for (d in c("dnorm", "dlaplace", "ds", "dgnorm", "dlnorm", "dgamma", "dinvgauss")) {
    run(paste0("ANN_", d), y_pos, list(model = "ANN", lags = 1, distribution = d))
}
cat("Multiplicative-error location models:\n")
for (d in c("dnorm", "dlaplace", "ds", "dgnorm", "dlnorm", "dgamma", "dinvgauss")) {
    run(paste0("MNN_", d), y_pos, list(model = "MNN", lags = 1, distribution = d))
}
cat("Seasonal, explicit sm model, holdout, occurrence:\n")
run("AAA_dnorm_seasonal", y_pos, list(model = "AAA", lags = c(1, 12), distribution = "dnorm"))
run("ANN_dnorm_smMNN",    y_pos, list(model = "ANN", lags = 1, distribution = "dnorm"),
    list(model = "MNN"))
run("ANN_dnorm_holdout",  y_pos, list(model = "ANN", lags = 1, distribution = "dnorm",
                                      h = 12, holdout = TRUE))
run("MNN_dnorm_holdout",  y_pos, list(model = "MNN", lags = 1, distribution = "dnorm",
                                      h = 12, holdout = TRUE))
run("occ_dnorm",          y_int, list(model = "ANN", lags = 1, distribution = "dnorm",
                                      occurrence = "odds-ratio"))
run("occ_MNN_dnorm",      y_int, list(model = "MNN", lags = 1, distribution = "dnorm",
                                      occurrence = "odds-ratio"))

json_path <- file.path(OUT_DIR, "sm_reference.json")
write(toJSON(results, pretty = TRUE, auto_unbox = TRUE, digits = NA), json_path)
cat("\nwrote", json_path, "-", length(results), "cases\n")
