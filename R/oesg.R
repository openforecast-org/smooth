#' Occurrence ETS general model
#'
#' Wrapper of \link[smooth]{omg} that fits a general occurrence (probability)
#' model — two parallel ETS sub-models A and B combined via a Beta-distribution
#' link — to a univariate intermittent time series. ARIMA components and
#' regression formulas are disabled; use \link[smooth]{omg} directly if those
#' are needed.
#'
#' @param y Univariate numeric vector or time series. Non-binary input is
#'   binarised internally by \link[smooth]{omg}.
#' @param modelA Three-letter ETS specification for sub-model A.
#' @param modelB Three-letter ETS specification for sub-model B.
#'   Defaults to \code{modelA}.
#' @param lags Vector of seasonal lags. Defaults to \code{frequency(y)}.
#' @param persistenceA Optional persistence vector for sub-model A.
#' @param persistenceB Optional persistence vector for sub-model B.
#'   Defaults to \code{persistenceA}.
#' @param phiA Optional damping parameter for sub-model A.
#' @param phiB Optional damping parameter for sub-model B. Defaults to \code{phiA}.
#' @param initial Initialisation method passed to both sub-models.
#' @param ic Information criterion for model selection.
#' @param h Forecast horizon.
#' @param holdout If \code{TRUE}, a holdout of size \code{h} is withheld.
#' @param bounds Parameter bounds type.
#' @param etsA ETS type for sub-model A: \code{"conventional"} or \code{"adam"}.
#' @param etsB ETS type for sub-model B. Defaults to \code{etsA}.
#' @param xregA Optional numeric vector or matrix of exogenous regressors for
#'   sub-model A, aligned with \code{y}.
#' @param xregB Optional numeric vector or matrix of exogenous regressors for
#'   sub-model B, aligned with \code{y}.
#' @param regressorsA How to handle \code{xregA}: \code{"use"} or \code{"select"}.
#' @param regressorsB How to handle \code{xregB}. Defaults to \code{regressorsA}.
#' @param silent If \code{TRUE}, suppresses output and plot.
#' @param ... Additional arguments forwarded to \link[smooth]{omg}.
#'
#' @return An object of class \code{c("omg","om","smooth","occurrence")}.
#'
#' @seealso \link[smooth]{omg}, \link[smooth]{oes}
#'
#' @examples
#' set.seed(42)
#' y <- rbinom(120, 1, 0.3)
#' m <- oesg(y, modelA="MNN", modelB="MNN")
#' forecast(m, h=12)
#'
#' @export
oesg <- function(y, modelA="MNN", modelB=modelA,
                 lags=c(frequency(y)),
                 persistenceA=NULL, persistenceB=persistenceA,
                 phiA=NULL, phiB=phiA,
                 initial=c("backcasting","optimal","two-stage","complete","gradient"),
                 ic=c("AICc","AIC","BIC","BICc"),
                 h=0, holdout=FALSE,
                 bounds=c("usual","admissible","none"),
                 etsA=c("conventional","adam"), etsB=etsA,
                 xregA=NULL, xregB=NULL,
                 regressorsA=c("use","select"), regressorsB=regressorsA,
                 silent=TRUE, ...) {
    startTime <- Sys.time();
    cl <- match.call();

    initial     <- match.arg(initial);
    regressorsA <- match.arg(regressorsA);
    regressorsB <- match.arg(regressorsB);

    colsA <- colsB <- NULL;
    baseData <- data.frame(y=as.vector(y));
    if(!is.null(xregA)) {
        xA <- if(is.matrix(xregA)) xregA[seq_len(length(y)),,drop=FALSE] else xregA[seq_len(length(y))];
        xA_df <- as.data.frame(xA);
        if(is.null(colnames(xA_df))) {
            colnames(xA_df) <- paste0("xA", seq_len(ncol(xA_df)));
        }
        colsA <- colnames(xA_df);
        baseData <- cbind(baseData, xA_df);
    }
    if(!is.null(xregB)) {
        xB <- if(is.matrix(xregB)) xregB[seq_len(length(y)),,drop=FALSE] else xregB[seq_len(length(y))];
        xB_df <- as.data.frame(xB);
        if(is.null(colnames(xB_df))) {
            colnames(xB_df) <- paste0("xB", seq_len(ncol(xB_df)));
        }
        colsB <- colnames(xB_df);
        baseData <- cbind(baseData, xB_df);
    }
    if(!is.null(xregA) || !is.null(xregB)) {
        data <- ts(as.matrix(baseData), start=start(y), frequency=frequency(y));
        colnames(data)[1] <- "y";
    } else {
        data <- y;
    }
    formulaA <- if(!is.null(colsA)) as.formula(paste("~", paste(colsA, collapse="+"))) else NULL;
    formulaB <- if(!is.null(colsB)) as.formula(paste("~", paste(colsB, collapse="+"))) else NULL;

    ourModel <- omg(data=data, modelA=modelA, modelB=modelB, lags=lags,
                    ordersA=list(ar=0, i=0, ma=0, select=FALSE),
                    ordersB=list(ar=0, i=0, ma=0, select=FALSE),
                    formulaA=formulaA, formulaB=formulaB,
                    armaA=NULL, armaB=NULL,
                    persistenceA=persistenceA, persistenceB=persistenceB,
                    phiA=phiA, phiB=phiB,
                    initial=initial, ic=ic, h=h, holdout=holdout,
                    bounds=bounds, etsA=etsA, etsB=etsB,
                    regressorsA=regressorsA, regressorsB=regressorsB,
                    silent=silent, ...);
    ourModel$call <- cl;
    ourModel$timeElapsed <- Sys.time() - startTime;
    return(ourModel);
}
