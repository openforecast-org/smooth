# The real R adam_bounds_checker() on pure-ARIMA matrices built by adam(): shortcut path against a forced whole-model check
suppressMessages(pkgload::load_all("/home/user/smooth", quiet=TRUE))
mods <- jsonlite::fromJSON(Sys.getenv("MATS"), simplifyVector=FALSE)
u <- function(x) unlist(x)
nInv <- nWrong <- nNon <- nNonDiff <- 0
for(mo in mods){
  kind <- u(mo$kind); k <- length(u(mo$lagsAll))
  F <- matrix(unlist(mo$F), k, k, byrow=TRUE); g <- matrix(u(mo$g), k, 1); w <- matrix(u(mo$w), 1)
  lags <- u(mo$lags); mq <- u(mo$ma); mac <- u(mo$mac)
  poly <- 1; ia <- 0
  for(j in seq_along(lags)){ q <- mq[j]; if(q>0){ f <- numeric(q*lags[j]+1); f[1] <- 1; f[(1:q)*lags[j]+1] <- mac[ia+1:q]; ia <- ia+q
        poly <- convolve(poly, rev(f), type="open") } }
  invertible <- min(Mod(polyroot(poly))) > 1+1e-9
  xreg <- kind=="xreg"
  elements <- list(vecG=g, matF=F, matWt=w)
  ar <- list(arReflection=0, maReflection=if(invertible) 0.5 else 2)
  pen <- adam_bounds_checker(elements, ar, "admissible", FALSE, FALSE, FALSE, 0, 0, 0, TRUE, FALSE, FALSE,
                             xreg, "use", as.integer(xreg), k-xreg-(kind=="constant"), u(mo$lagsAll), 1L, FALSE)
  full <- any(smoothEigens(g, F, w, u(mo$lagsAll), xreg, 1L) > 1+1e-50)
  if(invertible){ nInv <- nInv+1; nWrong <- nWrong + (pen!=0 || full) } else { nNon <- nNon+1; nNonDiff <- nNonDiff + ((pen!=0) != full) }
}
res <- list(list(quantity="models with all MA roots outside the unit circle, real R adam_bounds_checker with etsModel FALSE", value=nInv, conditions=list(models=length(mods))),
  list(quantity="of those, models with a penalty from the shortcut path or from a forced whole-model smoothEigens()", value=nWrong, conditions=list(models=nInv)),
  list(quantity="models with an MA root on or inside the circle; their penalty differs from the forced whole-model check", value=nNonDiff, conditions=list(models=nNon)))
jsonlite::write_json(res, Sys.getenv("CLAIMSET_RESULTS"), auto_unbox=TRUE, digits=NA)
for(r in res) cat(r$quantity, r$value, "\n")
