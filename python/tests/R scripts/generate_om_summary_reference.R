# Generates vcov / confint / coef reference fixtures for the Python OM/OMG
# parity test (python/tests/test_om_summary_r_comparison.py). Run from the
# repo root:
#   Rscript "python/tests/R scripts/generate_om_summary_reference.R"
#
# For each scenario it writes, into python/tests/data/om_summary/<name>/:
#   series.csv   — the input intermittent-demand series (so Python uses identical data)
#   coef.csv     — estimated coefficients (name,value)
#   vcov.csv     — vcov(m) covariance matrix (no row names; column order = coef order)
#   confint.csv  — confint(m, level=0.95): S.E. + lower/upper bounds (with row names)
#
# OMG uses smooth::om(..., occurrence = "general"), which the R package routes
# to the omg class. Joint vcov / confint coefficient rows are prefixed `A:` /
# `B:` to identify the sub-model — Python OMG mirrors that convention.
#
# Uses devtools::load_all(".") to run the *local* R source — the installed
# (CRAN/library) version may lag the current branch (vcov.om in particular is
# only registered in the local source as of v4.5.0.41006). Run from the repo
# root so the load_all path resolves correctly.
devtools::load_all(".")

outRoot <- file.path("python", "tests", "data", "om_summary")

writeScenario <- function(name, y, fit) {
    dir <- file.path(outRoot, name)
    dir.create(dir, recursive = TRUE, showWarnings = FALSE)

    # OMG: ``coef(fit)`` returns NULL (no coef.omg method); recover the joint
    # vector from the two sub-models so the CSV is consistent with what the
    # Python ``OMG.coef`` attribute exposes.
    if (inherits(fit, "omg")) {
        bA <- as.numeric(fit$modelA$B); names(bA) <- names(fit$modelA$B)
        bB <- as.numeric(fit$modelB$B); names(bB) <- names(fit$modelB$B)
        cf <- c(bA, bB)
        names(cf) <- c(paste0("A:", names(bA)), paste0("B:", names(bB)))
    } else {
        cf <- coef(fit)
    }
    V  <- vcov(fit)
    ci <- confint(fit, level = 0.95)

    write.csv(data.frame(value = as.numeric(y)),
              file.path(dir, "series.csv"), row.names = FALSE)
    write.csv(data.frame(name = names(cf), value = as.numeric(cf)),
              file.path(dir, "coef.csv"), row.names = FALSE)
    write.csv(as.data.frame(V),
              file.path(dir, "vcov.csv"), row.names = FALSE)
    write.csv(as.data.frame(ci),
              file.path(dir, "confint.csv"))  # keep row names

    cat("Wrote", name, "- params:", paste(names(cf), collapse = ", "), "\n")
    print(round(ci, 5))
}

# ── Deterministic input — matches the Python intermittent fixture
# (rpois(200, 0.3) via numpy default_rng(41) is RNG-different; the R series
# below is used in *both* sides so the comparison is on identical data).
set.seed(41)
yIntermittent <- rpois(200, 0.3)

# Scenario 1 — OM with MNN and odds-ratio occurrence (single ETS persistence)
fitOMOddsMNN <- om(yIntermittent, model = "MNN",
                   occurrence = "odds-ratio", lags = 1)
writeScenario("om_odds_mnn", yIntermittent, fitOMOddsMNN)

# Scenario 2 — OM with MNN and inverse-odds-ratio occurrence
fitOMInvMNN <- om(yIntermittent, model = "MNN",
                  occurrence = "inverse-odds-ratio", lags = 1)
writeScenario("om_inv_mnn", yIntermittent, fitOMInvMNN)

# Scenario 3 — OMG: joint two-sub-model fit. om(occurrence = "general")
# routes to the omg class internally; vcov / confint return joint quantities
# with `A:` / `B:` prefixed coefficient names.
fitOMG <- om(yIntermittent, model = "MNN",
             occurrence = "general", lags = 1)
writeScenario("omg_mnn_mnn", yIntermittent, fitOMG)
