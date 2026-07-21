#' Smooth package
#'
#' Package contains functions implementing Single Source of Error state space models for
#' purposes of time series analysis and forecasting.
#'
#' \tabular{ll}{ Package: \tab smooth\cr Type: \tab Package\cr Date: \tab
#' 2016-01-27 - Inf\cr License: \tab GPL-2 \cr } The following functions are
#' included in the package:
#' \itemize{
#' \item \link[smooth]{es} - Exponential Smoothing in Single Source of Errors State Space form.
#' \item \link[smooth]{ces} - Complex Exponential Smoothing.
#' \item \link[smooth]{gum} - Generalised Exponential Smoothing.
#' \item \link[smooth]{ssarima} - SARIMA in state space framework.
#' % \item \link[smooth]{nus} - Non-Uniform Smoothing.
#' \item \link[smooth]{auto.ces} - Automatic selection between seasonal and non-seasonal CES.
#' \item \link[smooth]{auto.ssarima} - Automatic selection of ARIMA orders.
#' \item \link[smooth]{sma} - Simple Moving Average in state space form.
#' \item \link[smooth]{smoothCombine} - the function that combines forecasts from es(),
#' ces(), gum(), ssarima() and sma() functions.
#' \item \link[smooth]{cma} - Centered Moving Average. This is for smoothing time series,
#' not for forecasting.
#' \item \link[smooth]{sim.es} - simulate time series using ETS as a model.
#' \item \link[smooth]{sim.ces} - simulate time series using CES as a model.
#' \item \link[smooth]{sim.ssarima} - simulate time series using SARIMA as a model.
#' \item \link[smooth]{sim.gum} - simulate time series using GUM as a model.
#' \item \link[smooth]{sim.sma} - simulate time series using SMA.
#' \item \link[smooth]{sim.oes} - simulate time series based on occurrence part of ETS model.
#' \item \link[smooth]{oes} - occurrence part of the intermittent state space model.
#' }
#' There are also several methods implemented in the package for the classes
#' "smooth" and "smooth.sim":
#' \itemize{
#' \item \link[smooth]{orders} - extracts orders of the fitted model.
#' \item lags - extracts lags of the fitted model.
#' \item modelType - extracts type of the fitted model.
#' \item forecast - produces forecast using provided model.
#' \item \link[smooth]{multicov} - returns covariance matrix of multiple steps ahead forecast errors.
#' \item \link[smooth]{pls} - returns Prediction Likelihood Score.
#' \item \link[greybox]{nparam} - returns number of the estimated parameters.
#' \item fitted - extracts fitted values from provided model.
#' \item getResponse - returns actual values from the provided model.
#' \item residuals - extracts residuals of provided model.
#' \item plot - plots either states of the model or produced forecast (depending on what object
#' is passed).
#' \item simulate - uses sim functions (\link[smooth]{sim.es}, \link[smooth]{sim.ces},
#' \link[smooth]{sim.ssarima}, \link[smooth]{sim.gum}, \link[smooth]{sim.sma} and
#' \link[smooth]{sim.oes}) in order to simulate data using the provided object.
#' \item summary - provides summary of the object.
#' \item AICc, BICc - return, guess what...
#' }
#'
#' @name smooth
#' @aliases smooth-package
#' @template ssAuthor
#'
#' @seealso \code{\link[greybox]{forecast}, \link[smooth]{es},
#' \link[smooth]{ssarima}, \link[smooth]{ces}, \link[smooth]{gum}}
#'
#' @template ssGeneralRef
#' @template ssIntermittentRef
#' @template ssCESRef
#' @template smoothRef
#' @template ssETSRef
#' @template ssIntervalsRef
#' @template ssKeywords
#'
#' @examples
#'
#' \donttest{y <- ts(rnorm(100,10,3), frequency=12)
#'
#' adam(y, h=20, holdout=TRUE)
#' es(y, h=20, holdout=TRUE)
#' gum(y, h=20, holdout=TRUE)
#' auto.ces(y, h=20, holdout=TRUE)
#' auto.ssarima(y, h=20, holdout=TRUE)}
#'
#' @import zoo Rcpp
#' @importFrom nloptr nloptr
#' @importFrom graphics abline layout legend lines par points polygon
#' @importFrom stats AIC BIC cov decompose median coef optimize nlminb cor var simulate lm as.formula residuals
#' @importFrom stats dbeta qnorm qt qlnorm quantile rbinom rlnorm rnorm rt runif plnorm pnorm
#' @importFrom stats deltat end frequency is.ts start time ts
#' @importFrom utils packageVersion combn modifyList
#' @importFrom greybox xregExpander stepwise qs qlaplace ps plaplace ds dlaplace graphmaker measures hm
#' @useDynLib smooth
NULL
