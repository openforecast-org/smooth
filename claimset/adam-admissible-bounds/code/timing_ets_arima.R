# Fit time and log-likelihood of ARIMA and ETS+ARIMA models with bounds="admissible", with the
# check of the parts apart (the package at the first root) and of the whole model (the second)
args <- commandArgs(trailingOnly=TRUE)
roots <- c(split=args[1], whole=args[2])
repeats <- as.integer(args[3])
fits <- list(
    "ARIMA(1,1,1)(1,1,1)[12]"=function() adam(AirPassengers, "NNN", lags=c(1,12), h=0,
                                              orders=list(ar=c(1,1), i=c(1,1), ma=c(1,1))),
    "ETS(A,N,A)+ARIMA(1,0,1)"=function() adam(AirPassengers, "ANA", orders=list(ar=1, ma=1), h=0,
                                              bounds="admissible"),
    "ETS(A,Ad,N)+ARIMA(0,1,2)"=function() adam(BJsales, "AAdN", orders=list(i=1, ma=2), h=0,
                                               bounds="admissible"),
    "ssarima (1,1,1)(1,1,1)[12]"=function() ssarima(AirPassengers, lags=c(1,12), h=0,
                                                   orders=list(ar=c(1,1), i=c(1,1), ma=c(1,1)))
)

results <- list()
for(version in names(roots)){
    # Installed in a temporary library, compiled as for users (load_all() compiles with -O0)
    library <- file.path(tempdir(), version)
    dir.create(library)
    pkgbuild::clean_dll(roots[version])
    install.packages(roots[version], repos=NULL, type="source", lib=library, quiet=TRUE)
    pkgbuild::clean_dll(roots[version])
    out <- callr::r(function(library, fits, repeats){
        suppressMessages(library(smooth, lib.loc=library))
        lapply(fits, function(fit){
            environment(fit) <- globalenv()
            times <- sapply(seq_len(repeats), function(i) system.time(model <- fit())["elapsed"])
            list(time=median(times), logLik=as.numeric(logLik(model)))
        })
    }, args=list(library=library, fits=fits, repeats=repeats))
    for(model in names(out)){
        conditions <- list(model=model, check=version, repeats=repeats)
        results[[length(results)+1]] <- list(quantity="median fit time", value=out[[model]]$time,
                                             unit="s", conditions=conditions)
        results[[length(results)+1]] <- list(quantity="log-likelihood of the fit",
                                             value=out[[model]]$logLik, conditions=conditions)
        cat(version, model, out[[model]]$time, out[[model]]$logLik, "\n")
    }
}
if(nzchar(Sys.getenv("CLAIMSET_RESULTS"))){
    jsonlite::write_json(results, Sys.getenv("CLAIMSET_RESULTS"), auto_unbox=TRUE, digits=NA)
}
