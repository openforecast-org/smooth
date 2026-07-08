# Generates vcov / confint / coef reference fixtures for the Python parity test
# (python/tests/test_adam_summary_r_comparison.py). Run from the repo root:
#   Rscript "python/tests/R scripts/generate_summary_reference.R"
#
# For each scenario it writes, into python/tests/data/summary/<name>/:
#   series.csv   — the input series (so Python uses identical data)
#   coef.csv     — estimated coefficients (name,value)
#   vcov.csv     — vcov(m) covariance matrix
#   confint.csv  — confint(m, level=0.95): S.E. + lower/upper bounds (with row names)
library(smooth)

outRoot <- file.path("python", "tests", "data", "summary")

writeScenario <- function(name, y, ...) {
    dir <- file.path(outRoot, name)
    dir.create(dir, recursive = TRUE, showWarnings = FALSE)

    m <- adam(y, initial = "optimal", ...)

    write.csv(data.frame(value = as.numeric(y)),
              file.path(dir, "series.csv"), row.names = FALSE)
    write.csv(data.frame(name = names(coef(m)), value = as.numeric(coef(m))),
              file.path(dir, "coef.csv"), row.names = FALSE)
    write.csv(as.data.frame(vcov(m)),
              file.path(dir, "vcov.csv"), row.names = FALSE)
    ci <- confint(m, level = 0.95)
    write.csv(as.data.frame(ci), file.path(dir, "confint.csv"))  # keep row names

    cat("Wrote", name, "- params:", paste(names(coef(m)), collapse = ", "), "\n")
    print(round(ci, 5))
}

set.seed(41)
yANN <- sim.es("ANN", obs = 120, frequency = 12, persistence = 0.3)$data
writeScenario("ann", yANN, model = "ANN")

set.seed(42)
yAAN <- sim.es("AAN", obs = 120, frequency = 12, persistence = c(0.3, 0.1))$data
writeScenario("aan", yAAN, model = "AAN")

set.seed(43)
yAAdN <- sim.es("AAdN", obs = 120, frequency = 12,
                persistence = c(0.3, 0.1), phi = 0.9)$data
writeScenario("aadn", yAAdN, model = "AAdN", bounds = "admissible")

set.seed(44)
yARIMA <- as.numeric(arima.sim(list(ar = 0.4, ma = 0.3), n = 300)) + 20
writeScenario("arima", yARIMA, model = "NNN",
              orders = list(ar = 1, i = 0, ma = 1))
