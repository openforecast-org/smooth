# Matrices of pure ARIMA (+constant, +static regressor) models as adam() builds them
suppressMessages(pkgload::load_all("/home/user/smooth", quiet=TRUE))
set.seed(as.integer(Sys.getenv("SEED", "1")))
n <- as.integer(Sys.getenv("NMOD", "40"))
out <- list()
stat_ar <- function(p){ # stationary AR coefficients of order p via reflection coefs
  a <- numeric(0)
  for(j in seq_len(p)){ k <- runif(1,-0.9,0.9); a <- c(a - k*rev(a), k) }
  a
}
for(i in seq_len(n)){
  m <- sample(c(4,12),1)
  seasonal <- runif(1) < 0.5
  lags <- if(seasonal) c(1,m) else 1
  ord <- function(mx) sapply(lags, function(l) sample(0:mx[1+(l>1)],1))
  ar <- ord(c(2,1)); ii <- ord(c(1,1)); ma <- ord(c(2,1))
  if(sum(ar,ii,ma)==0) next
  arc <- unlist(lapply(ar, function(p) if(p>0) stat_ar(p) else NULL))
  mac <- unlist(lapply(ma, function(q) if(q>0) runif(q,-1.4,1.4) else NULL))
  # unit-root MA at lag 1 with a difference
  if(ii[1]>0 && ma[1]==1 && runif(1)<0.5) mac[1] <- -1
  arma <- list(ar=arc, ma=mac); arma <- arma[sapply(arma,length)>0]
  if(length(arma)==0) arma <- NULL
  kind <- sample(c("plain","constant","xreg"),1)
  y <- ts(100+cumsum(rnorm(120)), frequency=m)
  x <- rnorm(120)
  orders <- list(ar=ar, i=ii, ma=ma)
  mod <- try(suppressWarnings(
    if(kind=="plain") adam(y,"NNN",orders=orders,lags=lags,arma=arma,h=0)
    else if(kind=="constant") adam(y,"NNN",orders=orders,lags=lags,arma=arma,constant=TRUE,h=0)
    else adam(data.frame(y=as.numeric(y),x=x),"NNN",orders=orders,lags=lags,arma=arma,regressors="use",h=0, frequency=m)), silent=TRUE)
  if(inherits(mod,"try-error")) { cat("fail", i, kind, "\n"); next }
  out[[length(out)+1]] <- list(kind=kind, m=m, lags=lags, ar=as.list(ar), i=as.list(ii), ma=as.list(ma),
        arc=as.numeric(arc), mac=as.numeric(mac),
        F=mod$transition, g=as.numeric(mod$persistence), w=as.numeric(tail(mod$measurement,1)),
        lagsAll=as.numeric(mod$lagsAll), states=colnames(mod$states))
}
jsonlite::write_json(out, Sys.getenv("OUT"), digits=NA, auto_unbox=FALSE)
cat("models", length(out), "\n")
