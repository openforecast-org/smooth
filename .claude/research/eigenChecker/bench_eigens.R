# Speed of the polynomial check (smoothEigensPolyR) against the per-lag check (smoothEigensR):
# per call on fixed models, and per fit with smoothEigens() switched to the polynomial check
args <- commandArgs(trailingOnly=TRUE)
root <- args[1]
repeats <- as.integer(args[2])
suppressMessages(pkgload::load_all(root, quiet=TRUE))
set.seed(41)

results <- list()
add <- function(quantity, value, ...) {
    results[[length(results) + 1]] <<- list(quantity=quantity, value=value, conditions=list(...))
}

# Matrices of additive ETS with a level, an optional trend and seasonal lags
etsMatrices <- function(alpha, gamma=NULL, lags=NULL, beta=NULL, phi=1) {
    k <- 1 + (!is.null(beta)) + length(lags)
    F <- diag(k)
    g <- c(alpha, beta, gamma)
    w <- rep(1, k)
    if(!is.null(beta)) {
        F[1, 2] <- F[2, 2] <- w[2] <- phi
    }
    return(list(g=matrix(g), F=F, w=matrix(w, 1), lags=as.integer(c(1, if(!is.null(beta)) 1, lags))))
}
gumMatrices <- function(lags, sd=0.3) {
    k <- length(lags)
    return(list(g=matrix(rnorm(k, 0, sd / 2)), F=matrix(rnorm(k^2, 0, sd), k), w=matrix(rnorm(k, 1, sd / 2), 1),
                lags=as.integer(lags)))
}
models <- list("ETS(A,N,N)"=etsMatrices(0.3),
               "ETS(A,Ad,N)"=etsMatrices(0.3, beta=0.1, phi=0.95),
               "ETS(A,N,A), m=12"=etsMatrices(0.3, 0.1, 12),
               "ETS(A,Ad,A), m=12"=etsMatrices(0.3, 0.1, 12, beta=0.1, phi=0.95),
               "ETS(A,N,A,A), m=24 and 168"=etsMatrices(0.3, c(0.1, 0.05), c(24, 168)),
               "GUM, orders (1,1), lags (1,12)"=gumMatrices(c(1, 12)),
               "GUM, orders (2,2), lags (1,12)"=gumMatrices(c(1, 1, 12, 12)),
               "GUM, orders (1,1,1), lags (1,24,168)"=gumMatrices(c(1, 24, 168)))

timePerCall <- function(f, m) {
    time <- system.time(for(i in 1:repeats) f(m$g, m$F, m$w, m$lags, FALSE, 1L, FALSE))[["elapsed"]]
    return(time / repeats * 1e6)
}
for(name in names(models)) {
    m <- models[[name]]
    add("mean time per call, microseconds", timePerCall(smoothEigensR, m), check="per-lag", model=name)
    add("mean time per call, microseconds", timePerCall(smoothEigensPolyR, m), check="polynomial", model=name)
}

# Whole fits with bounds="admissible", per-lag check against the polynomial one
smoothEigensOld <- smoothEigens
smoothEigensPoly <- function(persistence, transition, measurement, lagsModelAll, xregModel, obsInSample){
    persistenceNames <- names(persistence);
    hasDelta <- any(substr(persistenceNames,1,5)=="delta");
    xregNumber <- sum(substr(persistenceNames,1,5)=="delta");
    constantRequired <- any(persistenceNames %in% c("constant","drift"));
    return(as.numeric(smoothEigensPolyR(persistence, transition, measurement, lagsModelAll, xregModel,
                                        obsInSample, hasDelta, xregNumber, constantRequired)));
}
hourly <- ts(100 + 10 * sin(2 * pi * (1:1008) / 24) + 5 * sin(2 * pi * (1:1008) / 168) + rnorm(1008),
             frequency=24)
fits <- list("gum(AirPassengers, orders=c(1,1), lags=c(1,12))"=
                 function() gum(AirPassengers, orders=c(1,1), lags=c(1,12), bounds="admissible"),
             "gum(AirPassengers, orders=c(2,1), lags=c(1,12))"=
                 function() gum(AirPassengers, orders=c(2,1), lags=c(1,12), bounds="admissible"),
             "adam(AirPassengers, \"AAdA\")"=
                 function() adam(AirPassengers, "AAdA", bounds="admissible"),
             "adam(hourly, \"ANA\", lags=c(24,168))"=
                 function() adam(hourly, "ANA", lags=c(24,168), bounds="admissible"),
             "ces(AirPassengers, \"full\")"=
                 function() ces(AirPassengers, seasonality="full", bounds="admissible"))
calls <- 0
for(check in c("per-lag", "polynomial")) {
    counted <- if(check == "per-lag") smoothEigensOld else smoothEigensPoly
    assignInNamespace("smoothEigens", function(...) {calls <<- calls + 1; counted(...)}, "smooth")
    for(name in names(fits)) {
        calls <- 0
        time <- system.time(fit <- fits[[name]]())[["elapsed"]]
        add("time of the fit, seconds", time, check=check, fit=name)
        add("calls of smoothEigens() in the fit", calls, check=check, fit=name)
        add("time of the fit per call of smoothEigens(), microseconds", time / max(calls, 1) * 1e6,
            check=check, fit=name)
    }
}
assignInNamespace("smoothEigens", smoothEigensOld, "smooth")

for(r in results) {
    cat(r$quantity, unlist(r$conditions), r$value, "\n")
}
path <- Sys.getenv("CLAIMSET_RESULTS")
if(nzchar(path)) {
    jsonlite::write_json(results, path, auto_unbox=TRUE, digits=NA)
}
