# Multiple Comparison with the Best (Nemenyi) for the ETS and ARIMA benchmarks.
#
# Merges the per-series error measures exported by
#   01-export-python-measures.py  (Python competitors)
#   02-export-R-measures.R        (smooth and forecast, run in R)
# and applies rmcb() from greybox, which ranks the methods on every series and
# draws the Nemenyi intervals: methods whose intervals overlap are not
# distinguishable at the stated confidence level.
#
# Produces Images/mcb-ets-<measure>.pdf and Images/mcb-arima-<measure>.pdf.
#
# Usage:  Rscript 03-nemenyi-mcb.R [<data-dir>] [<image-dir>]

suppressMessages(library(greybox))

args <- commandArgs(trailingOnly=TRUE)
dataDir  <- if(length(args) > 0) args[1] else "Data"
imageDir <- if(length(args) > 1) args[2] else "Images"
dir.create(imageDir, showWarnings=FALSE, recursive=TRUE)

# Column labels as they should appear in the plots.
prettyNames <- c("ADAM ETS back"="ADAM ETS",
				 "ES back"="ES back",
				 "ES opt"="ES opt",
				 "forecast ets"="forecast::ets",
				 "MSARIMA back"="MSARIMA back",
				 "MSARIMA opt"="MSARIMA opt",
				 "forecast arima"="auto.arima",
				 "statsforecast AutoETS"="statsforecast",
				 "sktime AutoETS"="sktime",
				 "skforecast AutoETS"="skforecast",
				 "aeon AutoETS"="aeon",
				 "statsforecast AutoARIMA"="statsforecast",
				 "skforecast Arima"="skforecast",
				 "aeon AutoARIMA"="aeon")

runMCB <- function(tag, measure, mainText){
	rFile  <- file.path(dataDir, sprintf("r-%s-%s.csv", tag, measure))
	pyFile <- file.path(dataDir, sprintf("python-%s-%s.csv", tag, measure))
	rPart  <- read.csv(rFile,  check.names=FALSE)
	pyPart <- read.csv(pyFile, check.names=FALSE)
	stopifnot(nrow(rPart) == nrow(pyPart))
	errors <- cbind(rPart, pyPart)

	# Drop methods that produce no value at all (aeon has no quantile API, so it
	# cannot be scored on pinball) and then the series where any method is
	# missing, since rmcb() needs a complete block design.
	errors <- errors[, colSums(is.na(errors)) < nrow(errors), drop=FALSE]
	complete <- complete.cases(errors)
	if(sum(!complete) > 0)
		cat(sprintf("  %s/%s: dropping %d incomplete series of %d\n",
					tag, measure, sum(!complete), nrow(errors)))
	errors <- errors[complete, , drop=FALSE]

	labels <- colnames(errors)
	labels[labels %in% names(prettyNames)] <- prettyNames[labels[labels %in% names(prettyNames)]]
	colnames(errors) <- labels

	result <- rmcb(as.matrix(errors), level=0.95, outplot="none")

	# The method names sit in the left margin, so size it from the longest label
	# rather than guessing: strwidth() in inches divided by the line height
	# (par("csi")) gives the margin in the "lines" units that par(mar=) expects.
	# Measure on a throwaway null device -- calling plot.new() on the output
	# device would emit a blank first page, which is what \includegraphics picks up.
	pdf(NULL, width=4.8, height=4.4, pointsize=13); plot.new()
	labelInches <- max(strwidth(names(result$mean), units="inches"))
	lineInches <- par("csi")
	dev.off()
	leftMargin <- ceiling(labelInches / lineInches) + 2

	outFile <- file.path(imageDir, sprintf("mcb-%s-%s.pdf", tag, measure))
	pdf(outFile, width=4.8, height=4.4, pointsize=13)
	par(mar=c(4.2, leftMargin, 2.6, 0.8), cex.main=0.85)
	plot(result, outplot="lines", main=mainText)
	dev.off()

	# The output must be a single non-empty page; a second page means something
	# advanced the device before plotting.
	pages <- length(grep("/Type\\s*/Page[^s]", readLines(outFile, warn=FALSE)))
	cat(sprintf("  wrote %s (left margin %d lines, %.2f in for '%s')\n",
				outFile, leftMargin, labelInches,
				names(result$mean)[which.max(nchar(names(result$mean)))]))

	ranks <- sort(result$mean)
	cat("  mean ranks:", paste(sprintf("%s=%.2f", names(ranks), ranks), collapse="  "), "\n")
	invisible(result)
}

cat("ETS:\n")
runMCB("ets", "rmsse",   "RMSSE")
runMCB("ets", "pinball", "Scaled pinball loss")

cat("ARIMA:\n")
runMCB("arima", "rmsse",   "RMSSE")
runMCB("arima", "pinball", "Scaled pinball loss")

cat("MCB-DONE\n")
