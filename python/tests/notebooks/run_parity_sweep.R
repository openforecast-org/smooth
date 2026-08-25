## run_parity_sweep.R -- R half of the R/Python parity sweep.
##
## Fits ADAM ETS and auto.msarima to every M1 + M3 + Tourism series and records
## what the two languages must agree on: the selected model, the log-likelihood,
## the parameter count and AICc. The Python half is run_parity_sweep.py; compare
## with compare_parity_sweep.py.
##
## Run from the repository root:
##   Rscript "python/tests/notebooks/run_parity_sweep.R" <output.csv> [limit]

suppressMessages(library(smooth))
suppressMessages(library(doMC))
suppressMessages(library(foreach))

args <- commandArgs(TRUE)
outPath <- if (length(args) >= 1) args[1] else "parity-r.csv"
limit <- if (length(args) >= 2) as.numeric(args[2]) else 0

## Mcomp/Tcomp cannot be attached here (their dependency chain is broken in this
## container), so load the data objects directly.
e  <- new.env(); lazyLoad(file.path(system.file("data", package = "Mcomp"), "Rdata"), envir = e)
e2 <- new.env(); lazyLoad(file.path(system.file("data", package = "Tcomp"), "Rdata"), envir = e2)

pick <- function(lst) if (limit > 0) lst[seq_len(min(limit, length(lst)))] else lst
datasets <- c(pick(e$M1), pick(e$M3), pick(e2$tourism))
cat(length(datasets), "series\n")

## registerDoMC once, outside the loop: calling it inside forks a fresh pool per
## foreach() without reaping the previous one.
registerDoMC(min(30, detectCores()))

rows <- foreach(i = seq_along(datasets), .combine = c) %dopar% {
    s <- datasets[[i]]
    y <- as.numeric(s$x)
    p <- frequency(s$x)
    lg <- if (p > 1) c(1, p) else c(1)
    spec <- if (p > 1) "ZXZ" else "ZXN"
    out <- character(0)

    fitOne <- function(label, expr) {
        t0 <- Sys.time()
        m <- try(suppressWarnings(expr()), silent = TRUE)
        el <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
        if (inherits(m, "try-error")) {
            sprintf("%s|%s|ERROR|nan|nan|nan|nan", s$sn, label)
        } else {
            sprintf("%s|%s|%s|%.10f|%g|%.6f|%.3f", s$sn, label, m$model,
                    as.numeric(logLik(m)), nparam(m), AICc(m), el)
        }
    }

    out <- c(out, fitOne("ETS", function()
        adam(y, spec, lags = lg, initial = "backcasting", silent = TRUE)))
    out <- c(out, fitOne("ARIMA", function()
        auto.msarima(y, lags = lg, initial = "backcasting", silent = TRUE)))
    out
}

writeLines(rows, outPath)
cat("wrote", outPath, "-", length(rows), "rows\n")
