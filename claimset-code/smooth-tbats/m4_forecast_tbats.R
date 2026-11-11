# forecast::tbats on the M4 hourly samples prepared by m4_hourly.py.
# Usage: Rscript m4_forecast_tbats.R <input.csv> <output.csv> <workers> <h>
# The input has one row per observation: series, y (the fitting sample, in order).
# The output has one row per series and step: series, h, mean, lower, upper, seconds,
# status. Each fit is single-threaded; the series are spread over <workers> processes.
suppressMessages(library(forecast))
args <- commandArgs(trailingOnly=TRUE)
input <- read.csv(args[1], stringsAsFactors=FALSE)
workers <- as.integer(args[3])
h <- as.integer(args[4])
series <- unique(input$series)

forecastOne <- function(id){
    y <- input$y[input$series==id]
    started <- proc.time()[["elapsed"]]
    result <- tryCatch({
        fit <- tbats(msts(y, seasonal.periods=c(24, 168)), use.parallel=FALSE, num.cores=1)
        fc <- forecast(fit, h=h, level=95)
        data.frame(mean=as.numeric(fc$mean), lower=as.numeric(fc$lower),
                   upper=as.numeric(fc$upper), status="ok")
    }, error=function(e){
        return(data.frame(mean=rep(NA_real_, h), lower=NA_real_, upper=NA_real_,
                          status=paste("error:", conditionMessage(e))));
    })
    seconds <- proc.time()[["elapsed"]] - started
    return(cbind(data.frame(series=id, h=1:h), result, seconds=seconds))
}

output <- do.call(rbind, parallel::mclapply(series, forecastOne, mc.cores=workers,
                                            mc.preschedule=FALSE))
write.csv(output, args[2], row.names=FALSE)
