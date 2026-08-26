suppressMessages({library(Mcomp);library(Tcomp);library(smooth);library(greybox)
                  library(doMC);library(foreach)})
registerDoMC(parallel::detectCores())
d <- c(M1,M3,tourism); N <- length(d)
res <- foreach(k=1:(2*N), .combine="rbind", .packages=c("smooth","greybox")) %dopar% {
  init <- if(k<=N) "backcasting" else "optimal"
  i <- if(k<=N) k else k-N
  s <- d[[i]]
  out <- tryCatch({
    m <- auto.msarima(s, initial=init)
    fc <- forecast(m, h=s$h)
    o <- m$orders
    spec <- paste0("ar", paste(o$ar,collapse="."), "_i", paste(o$i,collapse="."),
                   "_ma", paste(o$ma,collapse="."))
    data.frame(idx=i-1, init=init, model=as.character(m$model), spec=spec,
               loglik=as.numeric(logLik(m)), aicc=as.numeric(AICc(m)),
               nparam=nparam(m),
               rmsse=as.numeric(measures(as.vector(s$xx), as.vector(fc$mean),
                                         as.vector(s$x))["RMSSE"]),
               fsum=sum(as.numeric(fc$mean)), stringsAsFactors=FALSE)
  }, error=function(e) data.frame(idx=i-1, init=init, model="ERROR", spec=NA,
                                  loglik=NA, aicc=NA, nparam=NA, rmsse=NA, fsum=NA,
                                  stringsAsFactors=FALSE))
  out
}
write.csv(res, "r_arima.csv", row.names=FALSE)
cat("wrote r_arima.csv", nrow(res), "\n")
