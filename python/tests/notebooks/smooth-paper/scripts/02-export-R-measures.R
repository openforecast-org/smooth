# Per-series, per-method error measures for the R implementations.
#
# Computes, for every series of M1 + M3 + Tourism (5,315 in total):
#   RMSSE       - point accuracy, as reported in Tables 6 and 7 of the paper
#   pinball     - scaled pinball loss, averaged over the 99 quantile levels
#   coverage[k] - empirical P(y <= q_tau) at each of the 99 levels, from which
#                 the mean calibration error is derived
#
# Results are cached per method in Data/r-benchmark-results.RData.  A method
# already present in the cache is NOT recomputed: re-running the script only
# fills in what is missing, so adding a method costs one method's worth of work
# rather than a full re-run.  Set RERUN to a comma-separated list of method
# names (or "all") to force specific ones.
#
# Parallelism: the worker pool is registered ONCE, below, and reused for every
# method.  Registering inside the per-method loop forks a fresh pool per
# foreach() call without reaping the previous one, which silently doubles the
# process count.  WORKERS must leave headroom for the parent process: this
# machine has 32 cores and must never carry more than 32 processes in total.
#
# Usage:  Rscript 02-export-R-measures.R [<output-dir>]
#         RERUN="forecast ets" Rscript 02-export-R-measures.R

suppressMessages({
	library(Mcomp); library(Tcomp); library(smooth); library(greybox)
	library(forecast); library(doMC); library(foreach)
})

args   <- commandArgs(trailingOnly = TRUE)
outDir <- if (length(args) > 0) args[1] else "Data"
dir.create(outDir, showWarnings = FALSE, recursive = TRUE)
cacheFile <- file.path(outDir, "r-benchmark-results.RData")

WORKERS <- min(30L, parallel::detectCores())   # + 1 parent, never above 32
registerDoMC(WORKERS)                          # register ONCE, reuse for all methods

LEVELS <- round(seq(0.01, 0.99, 0.01), 2)
TWOPCT <- round(100 * (1 - 2 * LEVELS[LEVELS < 0.5]))   # 98, 96, ..., 2
MEASURES <- c("rmsse", "pinball", paste0("cov", seq_along(LEVELS)))

# (label, model class, fitting function, interval type; NA = forecast package)
METHODS <- list(
	list("ADAM ETS back",  "ets",   function(x) adam(x, "ZXZ", initial = "back"), "prediction"),
	list("ES back",        "ets",   function(x) es(x, "ZXZ", initial = "back"),   "parametric"),
	list("ES opt",         "ets",   function(x) es(x, "ZXZ", initial = "opt"),    "parametric"),
	list("forecast ets",   "ets",   function(x) ets(x),                            NA),
	list("MSARIMA back",   "arima", function(x) auto.msarima(x, initial = "back"), "parametric"),
	list("MSARIMA opt",    "arima", function(x) auto.msarima(x, initial = "opt"),  "parametric"),
	list("forecast arima", "arima", function(x) auto.arima(x),                     NA)
)

# smooth reports interval levels as fractions, forecast as percentages.
quantilesFromForecast <- function(object) {
	h <- length(object$mean)
	lower <- as.matrix(object$lower); upper <- as.matrix(object$upper)
	lvl <- as.numeric(object$level)
	if (max(lvl, na.rm = TRUE) <= 1) lvl <- lvl * 100
	Q <- matrix(NA_real_, nrow = h, ncol = length(LEVELS))
	for (k in seq_along(LEVELS)) {
		tau <- LEVELS[k]
		if (abs(tau - 0.5) < 1e-9)   Q[, k] <- as.vector(object$mean)
		else if (tau < 0.5)          Q[, k] <- lower[, which.min(abs(lvl - 100 * (1 - 2 * tau)))]
		else                         Q[, k] <- upper[, which.min(abs(lvl - 100 * (2 * tau - 1)))]
	}
	t(apply(Q, 1, sort))          # monotone rearrangement, as in the Python scoring
}

scoreSeries <- function(object, holdout, insample) {
	holdout <- as.vector(holdout); insample <- as.vector(insample)
	Q <- quantilesFromForecast(object)
	scaleValue <- mean(abs(diff(insample)))
	if (!is.finite(scaleValue) || scaleValue == 0) scaleValue <- 1
	differences <- holdout - Q
	tauMatrix <- matrix(LEVELS, nrow = length(holdout), ncol = length(LEVELS), byrow = TRUE)
	pinball <- mean(colMeans(pmax(tauMatrix * differences,
								  (tauMatrix - 1) * differences)) / scaleValue)
	coverage <- colMeans(Q >= holdout)
	rmsse <- as.numeric(measures(holdout, as.vector(object$mean), insample)["RMSSE"])
	c(rmsse, pinball, coverage)
}

datasets <- c(M1, M3, tourism)
N <- length(datasets)
labels <- sapply(METHODS, `[[`, 1)

# ---- load the cache, or start an empty one -------------------------------
if (file.exists(cacheFile)) {
	load(cacheFile)                                   # brings in rResults
	cat("loaded cache:", cacheFile, "\n")
} else {
	rResults <- array(NA_real_, c(length(labels), N, length(MEASURES)),
					  dimnames = list(labels, NULL, MEASURES))
}
# tolerate a cache written before a method was added
if (!all(labels %in% dimnames(rResults)[[1]])) {
	grown <- array(NA_real_, c(length(labels), N, length(MEASURES)),
				   dimnames = list(labels, NULL, MEASURES))
	keep <- intersect(labels, dimnames(rResults)[[1]])
	grown[keep, , ] <- rResults[keep, , ]
	rResults <- grown
}

forced <- strsplit(Sys.getenv("RERUN", ""), ",")[[1]]
forced <- trimws(forced[nzchar(forced)])

cat("series:", N, "| workers:", WORKERS, "(+1 parent, cap 32)\n")

for (m in seq_along(METHODS)) {
	label <- METHODS[[m]][[1]]; fitFun <- METHODS[[m]][[3]]; intType <- METHODS[[m]][[4]]
	done <- !all(is.na(rResults[label, , "rmsse"]))
	if (done && !(label %in% forced) && !("all" %in% forced)) {
		cat(sprintf("  %-16s cached, skipping\n", label)); next
	}
	t0 <- Sys.time()
	res <- foreach(i = 1:N, .combine = "cbind",
				   .packages = c("smooth", "greybox", "forecast")) %dopar% {
		s <- datasets[[i]]
		tryCatch({
			fit <- fitFun(if (is.na(intType)) s$x else s)
			fc <- if (is.na(intType)) forecast(fit, h = s$h, level = TWOPCT)
				  else forecast(fit, h = s$h, interval = intType, level = TWOPCT / 100)
			scoreSeries(fc, s$xx, s$x)
		}, error = function(e) rep(NA_real_, length(MEASURES)))
	}
	rResults[label, , ] <- t(res)
	save(rResults, file = cacheFile)                  # checkpoint after each method
	cat(sprintf("  %-16s medRMSSE=%.4f medPinball=%.4f fail=%d [%.1f min]\n",
				label, median(rResults[label, , "rmsse"], na.rm = TRUE),
				median(rResults[label, , "pinball"], na.rm = TRUE),
				sum(is.na(rResults[label, , "rmsse"])),
				as.numeric(difftime(Sys.time(), t0, units = "mins"))))
	flush.console()
}

# ---- flat CSVs for the plotting script ----------------------------------
for (kind in c("ets", "arima")) {
	sel <- labels[sapply(METHODS, `[[`, 2) == kind]
	for (measure in c("rmsse", "pinball")) {
		mat <- t(rResults[sel, , measure, drop = FALSE][, , 1])
		colnames(mat) <- sel
		write.csv(mat, file.path(outDir, sprintf("r-%s-%s.csv", kind, measure)),
				  row.names = FALSE)
	}
}
cat("wrote", cacheFile, "and the r-{ets,arima}-{rmsse,pinball}.csv extracts\n")
cat("R-EXPORT-DONE\n")
