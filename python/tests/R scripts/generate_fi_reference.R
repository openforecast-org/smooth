# Generates Fisher Information reference fixtures for the Python parity test
# (python/tests/test_fi_r_comparison.py). Run from the repo root:
#   Rscript "python/tests/R scripts/generate_fi_reference.R"
#
# For each scenario it writes, into python/tests/data/fi/<name>/:
#   series.csv  — the input series (so Python uses identical data)
#   coef.csv    — estimated coefficients (name,value), the B at which FI is taken
#   FI.csv      — the observed Fisher Information matrix. R exposes the
#                 covariance via vcov() (= FI^-1, built by inverting the
#                 observed FI in vcov.adam), so the FI is recovered as
#                 solve(vcov(m)) at the fitted optimum.
library(smooth)

outRoot <- file.path("python", "tests", "data", "fi")

writeScenario <- function(name, y, model) {
    dir <- file.path(outRoot, name)
    dir.create(dir, recursive = TRUE, showWarnings = FALSE)

    m <- adam(y, model = model, initial = "optimal", FI = TRUE)

    write.csv(data.frame(value = as.numeric(y)),
              file.path(dir, "series.csv"), row.names = FALSE)
    write.csv(data.frame(name = names(coef(m)), value = as.numeric(coef(m))),
              file.path(dir, "coef.csv"), row.names = FALSE)
    write.csv(as.data.frame(solve(vcov(m))),
              file.path(dir, "FI.csv"), row.names = FALSE)

    cat("Wrote", name, "- params:", paste(names(coef(m)), collapse = ", "), "\n")
}

set.seed(41)
yANN <- sim.es("ANN", obs = 120, frequency = 12, persistence = 0.3)$data
writeScenario("ann", yANN, "ANN")

set.seed(42)
yAAN <- sim.es("AAN", obs = 120, frequency = 12,
               persistence = c(0.3, 0.1))$data
writeScenario("aan", yAAN, "AAN")
