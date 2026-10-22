#' @param initial Can be either character or a list, or a vector of initial states.
#' If it is character, then it can be \code{"backcasting"}, meaning that the initials of
#' dynamic part of the model are produced using backcasting procedure (advised
#' for data with high frequency), or \code{"optimal"}, meaning that all initial
#' states are optimised, or \code{"two-stage"}, meaning that optimisation is done
#' after the backcasting, refining the states. In case of backcasting, the parameters of the
#' explanatory variables are optimised. Alternatively, you can set \code{initial="complete"}
#' backcasting, which means that all states (including explanatory variables) are initialised
#' via backcasting.
#' Finally, \code{initial="gradient"} solves for the initial states by profiling
#' the estimation loss at the current parameters (a linear solve for the additive
#' models), as in \link[smooth]{adam}. It works for ETS, ARIMA, CES and GUM, with
#' the built-in losses; the multiplicative CES and GUM, and the ARIMA and the
#' explanatory variables of a model with multiplicative components, fall back to
#' backcasting. With a custom loss, it is switched to \code{"backcasting"} with a
#' warning.
