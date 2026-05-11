#' Occurrence ETS model
#'
#' Wrapper of \link[smooth]{om} that fits an occurrence (probability) ETS
#' model to a univariate intermittent time series. ARIMA components, ARMA
#' parameters, and explanatory-variable formulas are disabled (\code{orders},
#' \code{arma}, \code{formula} are all forced to \code{NULL}); use
#' \link[smooth]{om} directly if those features are needed.
#'
#' This is the analogue of \link[smooth]{es} for occurrence models: a
#' lightweight ETS-only entry point that delegates the heavy lifting to
#' \link[smooth]{om}.
#'
#' @param y Univariate numeric vector or time-series. Non-binary input is
#'   binarised (any non-zero value becomes 1).
#' @param model Three-letter ETS specification (e.g. \code{"MNN"},
#'   \code{"AAdN"}). Wildcards \code{"Z"}/\code{"X"}/\code{"Y"} trigger
#'   automatic selection.
#' @param lags Vector of seasonal lags. Defaults to \code{frequency(y)}.
#' @param persistence Optional persistence (smoothing) parameter vector.
#' @param phi Optional damping parameter. Only used for damped-trend models.
#' @param initial Initialisation method: \code{"backcasting"},
#'   \code{"optimal"}, \code{"two-stage"}, or \code{"complete"}.
#' @param occurrence Type of link function mapping the state to a
#'   probability: \code{"fixed"}, \code{"odds-ratio"},
#'   \code{"inverse-odds-ratio"}, or \code{"direct"}.
#' @param ic Information criterion for model selection.
#' @param h Forecast horizon.
#' @param holdout If \code{TRUE}, a holdout sample of size \code{h} is
#'   withheld.
#' @param bounds Parameter bounds type.
#' @param ets Type of ETS model: \code{"conventional"} or \code{"adam"}.
#' @param xreg Optional numeric vector or matrix of exogenous regressors, aligned
#'   with \code{y}. Merged with \code{y} into a multi-column data matrix before
#'   being passed to \link[smooth]{om}.
#' @param regressors How to handle regressors: \code{"use"} or \code{"select"}.
#' @param silent If \code{TRUE}, suppresses output and plot.
#' @param ... Additional arguments forwarded to \link[smooth]{om} (e.g.
#'   \code{maxeval}, \code{xtol_rel}, \code{algorithm}, \code{print_level}).
#'
#' @return An object of class \code{c("om","adam","smooth")}.
#'
#' @seealso \link[smooth]{om}, \link[smooth]{es}
#'
#' @examples
#' set.seed(42)
#' y <- rbinom(120, 1, 0.6)
#' m <- oes(y, model="MNN", occurrence="odds-ratio")
#' forecast(m, h=12)
#'
#' @export
oes <- function(y, model="MNN", lags=c(frequency(y)),
                persistence=NULL, phi=NULL,
                initial=c("backcasting","optimal","two-stage","complete"),
                occurrence=c("auto","fixed","odds-ratio","inverse-odds-ratio","direct","general"),
                ic=c("AICc","AIC","BIC","BICc"),
                h=0, holdout=FALSE,
                bounds=c("usual","admissible","none"),
                ets=c("conventional","adam"),
                xreg=NULL, regressors=c("use","select"),
                silent=TRUE, ...){

    startTime <- Sys.time();
    cl <- match.call();

    occurrence <- match.arg(occurrence);
    initial    <- match.arg(initial);
    regressors <- match.arg(regressors);
    if(!is.null(xreg)){
        if(is.matrix(xreg)){
            data <- as.matrix(cbind(y=as.data.frame(y), as.data.frame(xreg[1:length(y),])));
        }
        else{
            data <- as.matrix(cbind(y=as.data.frame(y), as.data.frame(xreg[1:length(y)])));
        }
        data <- ts(data, start=start(y), frequency=frequency(y));
        colnames(data)[1] <- "y";
        if(is.null(colnames(xreg))){
            if(!is.null(ncol(xreg))){
                colnames(data)[-1] <- paste0("x", c(1:ncol(xreg)));
            }
            else{
                colnames(data)[-1] <- "x";
            }
        }
    }
    else{
        data <- y;
    }

    # ARIMA / ARMA / regression-formula are disabled in the ETS-only wrapper.
    # om() requires the orders list to have the ar/i/ma/select fields, so the
    # "NULL" intent is expressed as the no-ARIMA default below.
    ourModel <- om(data=data, model=model, lags=lags,
                   orders=list(ar=0, i=0, ma=0, select=FALSE),
                   formula=NULL, arma=NULL,
                   persistence=persistence, phi=phi,
                   initial=initial, occurrence=occurrence,
                   ic=ic, h=h, holdout=holdout,
                   bounds=bounds, ets=ets,
                   regressors=regressors,
                   silent=silent, ...);
    ourModel$call <- cl;
    ourModel$timeElapsed <- Sys.time() - startTime;
    return(ourModel);
}
