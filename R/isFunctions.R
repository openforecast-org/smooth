#' Smooth classes checkers
#'
#' Functions to check if an object is of the specified class
#'
#' The list of functions includes:
#' \itemize{
#' \item \code{is.smooth()} tests if the object was produced by a smooth function
#' (e.g. \link[smooth]{es} / \link[smooth]{ces} / \link[smooth]{ssarima} /
#' \link[smooth]{gum} / \link[smooth]{sma} / \link[smooth]{msarima});
#' \item \code{is.msarima()} tests if the object was produced by the
#' \link[smooth]{msarima} function;
#' \item \code{is.smoothC()} tests if the object was produced by a combination
#' function (currently applies only to \link[smooth]{smoothCombine});
#' \item \code{is.oes()} tests if the object was produced by \link[smooth]{oes}
#' function;
#' \item \code{is.smooth.sim()} tests if the object was produced by simulate functions
#' (e.g. \link[smooth]{sim.es} / \link[smooth]{sim.ces} / \link[smooth]{sim.ssarima}
#' / \link[smooth]{sim.sma} / \link[smooth]{sim.gum});
#' \item \code{is.smooth.forecast()} checks if the forecast was produced from a smooth
#' function using forecast() function.
#' }
#'
#' @param x The object to check.
#' @return \code{TRUE} if this is the specified class and \code{FALSE} otherwise.
#'
#' @template ssAuthor
#' @keywords ts univar
#' @examples
#'
#' ourModel <- msarima(rnorm(100,100,10))
#'
#' is.smooth(ourModel)
#' is.msarima(ourModel)
#'
#' @rdname isFunctions
#' @export
is.smooth <- function(x){
    return(inherits(x,"smooth"))
}

#' @rdname isFunctions
#' @export
is.smoothC <- function(x){
    return(inherits(x,"smoothC"))
}

#' @rdname isFunctions
#' @export
is.msarima <- function(x){
    return(inherits(x,"msarima"))
}

#' @rdname isFunctions
#' @export
is.oes <- function(x){
    return(inherits(x,"oes"))
}

#' @rdname isFunctions
#' @export
is.oesg <- function(x){
    return(inherits(x,"oesg"))
}

#' @rdname isFunctions
#' @export
is.om <- function(x){
    return(inherits(x,"om"))
}

#' @rdname isFunctions
#' @export
is.omg <- function(x){
    return(inherits(x,"omg"))
}

#' @rdname isFunctions
#' @export
is.smooth.sim <- function(x){
    return(inherits(x,"smooth.sim"))
}

#' @rdname isFunctions
#' @export
is.smooth.forecast <- function(x){
    return(inherits(x,"smooth.forecast"))
}

#' ADAM classes checkers
#'
#' Functions to check if an object is of the specified class
#'
#' The list of functions includes:
#' \itemize{
#' \item \code{is.adam()} tests if the object was produced by a \link[smooth]{adam} function
#' \item \code{is.adam.sim()} tests if the object was produced by sim.adam() function (not implemented yet);
#' }
#'
#' @param x The object to check.
#' @return \code{TRUE} if this is the specified class and \code{FALSE} otherwise.
#'
#' @template ssAuthor
#' @keywords ts univar
#' @examples
#'
#' ourModel <- adam(rnorm(100,100,10))
#' is.adam(ourModel)
#'
#' @rdname isFunctions
#' @export
is.adam <- function(x){
    return(inherits(x,"adam"))
}

#' @rdname isFunctions
#' @export
is.adam.sim <- function(x){
    return(inherits(x,"sim.adam"))
}
