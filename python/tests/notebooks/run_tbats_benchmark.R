# The R forecasters of the TBATS benchmark on M4 hourly (see run_tbats_benchmark.py).
#
#   Rscript run_tbats_benchmark.R forecast smooth-R [--workers 4]
#
# "forecast" is forecast::tbats() and "smooth-R" is smooth's tbats(), the latter from
# the library in SMOOTH_R_LIB when it is set. The data must have been downloaded and
# verified by `python run_tbats_benchmark.py data`. Each series is written to
# CACHE/tbats-m4-hourly/<method>/<series>.csv in the format of the Python runner.

arguments <- commandArgs(TRUE)
lib <- Sys.getenv("SMOOTH_R_LIB")
suppressMessages({
    if (nzchar(lib)) {
        library(smooth, lib.loc=lib)
    } else {
        library(smooth)
    }
    library(forecast)
    library(parallel)
})

cache <- Sys.getenv("SMOOTH_BENCH_CACHE", path.expand("~/.cache/smooth-benchmarks"))
forecasts <- file.path(cache, "tbats-m4-hourly")
h <- 48
levels <- round(seq(0.01, 0.99, 0.01), 2)
two <- round(1 - 2 * levels[levels < 0.5], 2)
timeout <- as.numeric(Sys.getenv("BENCH_TASK_TIMEOUT", "1800"))
qcols <- sprintf("q%02d", round(100 * levels))

# The 99 quantiles from the two-sided intervals of the levels two, the point at 0.5
quantiles <- function(point, lower, upper) {
    return(cbind(as.matrix(lower), as.numeric(point), as.matrix(upper)[, ncol(upper):1]))
}

forecasters <- list(
    "forecast"=function(x) {
        model <- forecast::tbats(msts(x, seasonal.periods=c(24, 168)), use.parallel=FALSE)
        fc <- forecast::forecast(model, h=h, level=100 * two)
        return(list(Q=quantiles(fc$mean, fc$lower, fc$upper), model=as.character(model)))
    },
    "smooth-R"=function(x) {
        model <- smooth::tbats(ts(x, frequency=24), lags=c(1, 24, 168))
        fc <- generics::forecast(model, h=h, interval="prediction", level=two, side="both")
        return(list(Q=quantiles(fc$mean, fc$lower, fc$upper), model=model$model))
    }
)

# Fit and forecast one series, timing both; a failure is written with NaN quantiles
task <- function(method, sid, x) {
    start <- Sys.time()
    result <- tryCatch({
        setTimeLimit(elapsed=timeout, transient=TRUE)
        forecasters[[method]](x)
    }, error=function(e) {
        list(Q=matrix(NaN, h, 99), model=substr(paste0("ERROR: ", conditionMessage(e)), 1, 200))
    })
    setTimeLimit(elapsed=Inf)
    seconds <- as.numeric(difftime(Sys.time(), start, units="secs"))
    Q <- result$Q
    colnames(Q) <- qcols
    frame <- data.frame(series=sid, n=length(x), time=seconds, model=result$model, h=1:h,
                        point=Q[, 50], Q, check.names=FALSE)
    path <- file.path(forecasts, method, paste0(sid, ".csv"))
    write.csv(frame, paste0(path, ".tmp"), row.names=FALSE)
    file.rename(paste0(path, ".tmp"), path)
    return(sid)
}

train <- read.csv(file.path(cache, "m4", "Hourly-train.csv"), row.names=1)
series <- lapply(rownames(train), function(sid) {
    return(as.numeric(na.omit(unlist(train[sid, ]))))
})
names(series) <- rownames(train)
workers <- if ("--workers" %in% arguments) {
    as.integer(arguments[which(arguments == "--workers") + 1])
} else {
    4
}

# Load the code paths before the timed fits; the forked workers inherit them
t <- 0:399
warmup <- 10 + sin(2 * pi * t / 24) + sin(2 * pi * t / 168) + t / 100
for (method in intersect(arguments, names(forecasters))) {
    dir.create(file.path(forecasts, method), recursive=TRUE, showWarnings=FALSE)
    invisible(suppressWarnings(forecasters[[method]](warmup)))
    done <- sub("\\.csv$", "", list.files(file.path(forecasts, method), pattern="\\.csv$"))
    todo <- setdiff(names(series), done)
    cat(sprintf("%s: %d of %d series to forecast, %d workers\n", method, length(todo),
                length(series), workers))
    invisible(mclapply(todo, function(sid) {
        return(suppressWarnings(task(method, sid, series[[sid]])))
    }, mc.cores=workers, mc.preschedule=FALSE))
}
