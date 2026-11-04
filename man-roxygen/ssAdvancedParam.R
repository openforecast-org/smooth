#' @param loss The type of Loss Function used in optimization. \code{loss} can
#' be: \code{likelihood} (assuming Normal distribution of error term),
#' \code{MSE} (Mean Squared Error), \code{MAE} (Mean Absolute Error),
#' \code{HAM} (Half Absolute Moment), \code{TMSE} - Trace Mean Squared Error,
#' \code{GTMSE} - Geometric Trace Mean Squared Error, \code{MSEh} - optimisation
#' using only h-steps ahead error, \code{MSCE} - Mean Squared Cumulative Error.
#' If \code{loss!="MSE"}, then likelihood and model selection is done based
#' on equivalent \code{MSE}. Model selection in this cases becomes not optimal.
#'
#' There are also available analytical approximations for multistep functions:
#' \code{aMSEh}, \code{aTMSE} and \code{aGTMSE}. These can be useful in cases
#' of small samples.
#'
#' Finally, just for fun the absolute and half analogues of multistep estimators
#' are available: \code{MAEh}, \code{TMAE}, \code{GTMAE}, \code{MACE}, \code{TMAE},
#' \code{HAMh}, \code{THAM}, \code{GTHAM}, \code{CHAM}.
#'
#' A custom loss is a function of \code{actual}, \code{fitted} and \code{B}, as in
#' \link[smooth]{adam}, for example
#' \code{function(actual, fitted, B) return(mean(abs(actual-fitted)))}.
