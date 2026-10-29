# Forecasts of the benchmarks of SB23 section 5 on M5 series: Croston, SBA (tsintermittent::
# crost) and TSB (tsintermittent::tsb) with their default settings, point forecasts only; HSP
# and NegBin (counter::hsp, counter::negbin) with the upper quantiles at 0.80, 0.90, 0.95 and
# 0.99 (side="upper"). The last 14 days are held out, leading zeros removed (load.py).
# Usage: Rscript forecast_benchmarks.R <sales_train> <sales_test> <rows.csv> <workers>
# Writes benchmark_forecasts.csv to $CLAIMSET_RUN_DIR in the columns of iets_forecasts.csv.
suppressMessages({library(tsintermittent); library(counter)})
args <- commandArgs(TRUE)
h <- 14
levels <- c(0.8, 0.9, 0.95, 0.99)
rows <- read.csv(args[3])$row
train <- read.csv(args[1], check.names=FALSE)
test <- read.csv(args[2], check.names=FALSE)
values <- cbind(as.matrix(train[rows+1, -(1:5)]), as.matrix(test[rows+1, -(1:5)]))

forecastSeries <- function(i){
    y <- as.numeric(values[i, ])
    y <- y[which(y != 0)[1]:length(y)]
    yIn <- head(y, -h)
    out <- NULL
    add <- function(method, mean, upper, seconds, status){
        return(data.frame(row=rows[i], method=method, horizon=1:h, mean=mean,
                          q0.8=upper[,1], q0.9=upper[,2], q0.95=upper[,3], q0.99=upper[,4],
                          seconds=seconds, status=status))
    }
    if(sum(yIn != 0) < 5){
        for(method in c("Croston","SBA","TSB","HSP","NegBin")){
            out <- rbind(out, add(method, NA, matrix(NA, h, 4), 0, "excluded"))
        }
        return(out)
    }
    point <- list(Croston=function() crost(yIn, h=h, type="croston")$frc.out,
                  SBA=function() crost(yIn, h=h, type="sba")$frc.out,
                  TSB=function() tsb(yIn, h=h)$frc.out)
    for(method in names(point)){
        t0 <- Sys.time()
        result <- tryCatch(list(mean=as.numeric(point[[method]]()), status="ok"),
                           error=function(e) list(mean=rep(NA, h), status=conditionMessage(e)))
        out <- rbind(out, add(method, result$mean, matrix(NA, h, 4),
                              as.numeric(Sys.time()-t0, units="secs"), result$status))
    }
    counters <- list(HSP=hsp, NegBin=negbin)
    for(method in names(counters)){
        t0 <- Sys.time()
        result <- tryCatch({
            fits <- lapply(levels, function(l) counters[[method]](y, h=h, holdout=TRUE,
                                                                   level=l, side="upper"))
            list(mean=as.numeric(fits[[1]]$forecast),
                 upper=sapply(fits, function(f) as.numeric(f$upper)), status="ok")
        }, error=function(e) list(mean=rep(NA, h), upper=matrix(NA, h, 4),
                                  status=conditionMessage(e)))
        out <- rbind(out, add(method, result$mean, result$upper,
                              as.numeric(Sys.time()-t0, units="secs"), result$status))
    }
    return(out)
}

forecasts <- do.call(rbind, parallel::mclapply(seq_along(rows), forecastSeries,
                                               mc.cores=as.integer(args[4])))
outdir <- Sys.getenv("CLAIMSET_RUN_DIR", ".")
write.csv(forecasts, file.path(outdir, "benchmark_forecasts.csv"), row.names=FALSE)
cat(length(rows), "series\n")
