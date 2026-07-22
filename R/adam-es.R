
#' Exponential Smoothing in SSOE state space model
#'
#' Function constructs ETS model and returns forecast, fitted values, errors
#' and matrix of states. It is a wrapper of \link[smooth]{adam} function.
#'
#' Function estimates ETS in a form of the Single Source of Error state space
#' model of the following type:
#'
#' \deqn{y_{t} = o_t (w(v_{t-l}) + h(x_t, a_{t-1}) + r(v_{t-l}) \epsilon_{t})}
#'
#' \deqn{v_{t} = f(v_{t-l}) + g(v_{t-l}) \epsilon_{t}}
#'
#' \deqn{a_{t} = F_{X} a_{t-1} + g_{X} \epsilon_{t} / x_{t}}
#'
#' Where \eqn{o_{t}} is the Bernoulli distributed random variable (in case of
#' normal data it equals to 1 for all observations), \eqn{v_{t}} is the state
#' vector and \eqn{l} is the vector of lags, \eqn{x_t} is the vector of
#' exogenous variables. w(.) is the measurement function, r(.) is the error
#' function, f(.) is the transition function, g(.) is the persistence
#' function and h(.) is the explanatory variables function. \eqn{a_t} is the
#' vector of parameters for exogenous variables, \eqn{F_{X}} is the
#' \code{transitionX} matrix and \eqn{g_{X}} is the \code{persistenceX} matrix.
#' Finally, \eqn{\epsilon_{t}} is the error term.
#'
#' For the details see Svetunkov (2023) or Hyndman et al.(2008).
#'
#' For some more information about the model and its implementation, see the
#' vignette: \code{vignette("es","smooth")}.
#'
#' Also, there are posts about the functions of the package smooth on the
#' website of Ivan Svetunkov:
#' \url{https://openforecast.org/category/r-en/smooth/} - they explain the
#' underlying models and how to use the functions.
#'
#'
#' @template ssBasicParam
#' @template ssAdvancedParam
#' @template ssXregParam
#' @template ssPersistenceParam
#' @template ssAuthor
#' @template ssKeywords
#'
#' @template ADAMInitial
#'
#' @template ssGeneralRef
#' @template ssETSRef
#' @template ssADAMRef
#'
#' @param model The type of ETS model. The first letter stands for the type of
#' the error term ("A" or "M"), the second (and sometimes the third as well) is for
#' the trend ("N", "A", "Ad", "M" or "Md"), and the last one is for the type of
#' seasonality ("N", "A" or "M"). So, the function accepts words with 3 or 4
#' characters: \code{ANN}, \code{AAN}, \code{AAdN}, \code{AAA}, \code{AAdA},
#' \code{MAdM} etc. \code{ZZZ} means that the model will be selected based on the
#' chosen information criteria type. Models pool can be restricted with additive
#' only components. This is done via \code{model="XXX"}. For example, making
#' selection between models with none / additive / damped additive trend
#' component only (i.e. excluding multiplicative trend) can be done with
#' \code{model="ZXZ"}. Furthermore, selection between multiplicative models
#' (excluding additive components) is regulated using \code{model="YYY"}. This
#' can be useful for positive data with low values (for example, slow moving
#' products). Finally, if \code{model="CCC"}, then all the models are estimated
#' and combination of their forecasts using AIC weights is produced (Kolassa,
#' 2011). This can also be regulated. For example, \code{model="CCN"} will
#' combine forecasts of all non-seasonal models and \code{model="CXY"} will
#' combine forecasts of all the models with non-multiplicative trend and
#' non-additive seasonality with either additive or multiplicative error. Not
#' sure why anyone would need this thing, but it is available.
#'
#' The parameter \code{model} can also be a vector of names of models for a
#' finer tuning (pool of models). For example, \code{model=c("ANN","AAA")} will
#' estimate only two models and select the best of them.
#'
#' Also \code{model} can accept a previously estimated ES or ETS (from forecast
#' package) model and use all its parameters.
#'
#' Keep in mind that model selection with "Z" components uses Branch and Bound
#' algorithm and may skip some models that could have slightly smaller
#' information criteria.
#' @param lags Defines lags for the corresponding components. All components
#' count, starting from level, so ETS(M,M,M) model for monthly data will have
#' \code{lags=c(1,1,12)}. However, the function will also accept \code{lags=c(12)},
#' assuming that the lags 1 were dropped.
#' @param phi Value of damping parameter. If \code{NULL} then it is estimated.
#' @param initialSeason Vector of initial values for seasonal components. If
#' \code{NULL}, they are estimated during optimisation.
#' @param bounds What type of bounds to use in the model estimation. The first
#' letter can be used instead of the whole word. \code{"usual"} implies restrictions on
#' the smoothing parameter, guaranteeing that the exponential smoothing behaves
#' as an averaging model. \code{"admissible"} guarantee that the model is stable.
#' @param smoother The smoother used by the \link[smooth]{msdecompose} function to
#' obtain the initial states - the level and the trend, together with the initial
#' seasonal indices (and the seasonal profiles in the case of multiple seasonal
#' models). \code{smoother="default"} (the default) resolves to \code{"ma"} (the
#' centred moving average) when \code{initial="optimal"} and to \code{"global"}
#' (a global model fitted to the data) for every other initialisation method. The
#' other values \code{"ma"}, \code{"lowess"} (\link[stats]{lowess}), \code{"supsmu"}
#' (\link[stats]{supsmu}) and \code{"global"} force the respective smoother
#' irrespective of the initialisation.
#' @param ...  Other non-documented parameters. For example \code{FI=TRUE} will
#' make the function also produce Fisher Information matrix, which then can be
#' used to calculated variances of smoothing parameters and initial states of
#' the model.
#' Parameters \code{B}, \code{lb} and \code{ub} can be passed via
#' ellipsis as well. In this case they will be used for optimisation. \code{B}
#' sets the initial values before the optimisation, \code{lb} and
#' \code{ub} define lower and upper bounds for the search inside of the
#' specified \code{bounds}. These values should have length equal to the number
#' of parameters to estimate.
#' You can also pass two parameters to the optimiser: 1. \code{maxeval} - maximum
#' number of evaluations to carry on; 2. \code{xtol_rel} - the precision of the
#' optimiser. The default values used in es() are \code{maxeval=500} and
#' \code{xtol_rel=1e-8}. You can read more about these parameters in the
#' documentation of \link[nloptr]{nloptr} function.
#' @return Object of class "adam" is returned. It contains the list of the
#' following values for classical ETS models:
#'
#' \itemize{
#' \item \code{model} - type of constructed model.
#' \item \code{formula} - mathematical formula, describing interactions between
#' components of es() and exogenous variables.
#' \item \code{timeElapsed} - time elapsed for the construction of the model.
#' \item \code{states} - matrix of the components of ETS.
#' \item \code{persistence} - persistence vector. This is the place, where
#' smoothing parameters live.
#' \item \code{phi} - value of damping parameter.
#' \item \code{transition} - transition matrix of the model.
#' \item \code{measurement} - measurement vector of the model.
#' \item \code{initialType} - type of the initial values used.
#' \item \code{initial} - initial values of the state vector (non-seasonal).
#' \item \code{initialSeason} - initial values of the seasonal part of state vector.
#' \item \code{nParam} - table with the number of estimated / provided parameters.
#' If a previous model was reused, then its initials are reused and the number of
#' provided parameters will take this into account.
#' \item \code{fitted} - fitted values of ETS. In case of the intermittent model, the
#' fitted are multiplied by the probability of occurrence.
#' \item \code{forecast} - the point forecast for h steps ahead (by default NA is returned). NOTE
#' that these do not always correspond to the conditional expectations. See ADAM
#' textbook, Section 4.4. for details (\url{https://openforecast.org/adam/ETSTaxonomyMaths.html}),
#' \item \code{lower} - lower bound of prediction interval. When \code{interval="none"}
#' then NA is returned.
#' \item \code{upper} - higher bound of prediction interval. When \code{interval="none"}
#' then NA is returned.
#' \item \code{residuals} - residuals of the estimated model.
#' \item \code{errors} - trace forecast in-sample errors, returned as a matrix. Only returned when the
#' multistep losses are used and semiparametric interval is needed.
#' \item \code{s2} - variance of the residuals (taking degrees of freedom into account).
#' This is an unbiased estimate of variance.
#' \item \code{interval} - type of interval asked by user.
#' \item \code{level} - confidence level for interval.
#' \item \code{cumulative} - whether the produced forecast was cumulative or not.
#' \item \code{y} - original data.
#' \item \code{holdout} - holdout part of the original data.
#' \item \code{xreg} - provided vector or matrix of exogenous variables. If \code{regressors="s"},
#' then this value will contain only selected exogenous variables.
#' \item \code{initialX} - initial values for parameters of exogenous variables.
#' \item \code{ICs} - values of information criteria of the model. Includes AIC, AICc, BIC and BICc.
#' \item \code{logLik} - concentrated log-likelihood of the function.
#' \item \code{lossValue} - loss function value.
#' \item \code{loss} - type of loss function used in the estimation.
#' \item \code{FI} - Fisher Information. Equal to NULL if \code{FI=FALSE} or when \code{FI}
#' is not provided at all.
#' \item \code{accuracy} - vector of accuracy measures for the holdout sample. In
#' case of non-intermittent data includes: MPE, MAPE, SMAPE, MASE, sMAE,
#' RelMAE, sMSE and Bias coefficient (based on complex numbers). In case of
#' intermittent data the set of errors will be: sMSE, sPIS, sCE (scaled
#' cumulative error) and Bias coefficient. This is available only when
#' \code{holdout=TRUE}.
#' \item \code{B} - the vector of all the estimated parameters.
#' }
#'
#' If combination of forecasts is produced (using \code{model="CCC"}), then a
#' shorter list of values is returned:
#'
#' \itemize{
#' \item \code{model},
#' \item \code{timeElapsed},
#' \item \code{initialType},
#' \item \code{fitted},
#' \item \code{forecast},
#' \item \code{lower},
#' \item \code{upper},
#' \item \code{residuals},
#' \item \code{s2} - variance of additive error of combined one-step-ahead forecasts,
#' \item \code{interval},
#' \item \code{level},
#' \item \code{cumulative},
#' \item \code{y},
#' \item \code{holdout},
#' \item \code{ICs} - combined ic,
#' \item \code{ICw} - ic weights used in the combination,
#' \item \code{loss},
#' \item \code{xreg},
#' \item \code{accuracy}.
#' }
#' @seealso \code{\link[smooth]{adam}, \link[greybox]{forecast},
#' \link[stats]{ts}, \link[smooth]{sim.es}}
#'
#' @examples
#'
#' # See how holdout and trace parameters influence the forecast
#' es(BJsales,model="AAdN",h=8,holdout=FALSE,loss="MSE")
#' \donttest{es(AirPassengers,model="MAM",h=18,holdout=TRUE,loss="TMSE")}
#'
#' # Model selection example
#' es(BJsales,model="ZZN",ic="AIC",h=8,holdout=FALSE,bounds="a")
#'
#' # Model selection. Compare AICc of these two models:
#' \donttest{es(AirPassengers,"ZZZ",h=10,holdout=TRUE)
#' es(AirPassengers,"MAdM",h=10,holdout=TRUE)}
#'
#' # Model selection, excluding multiplicative trend
#' \donttest{es(AirPassengers,model="ZXZ",h=8,holdout=TRUE)}
#'
#' # Combination example
#' \donttest{es(BJsales,model="CCN",h=8,holdout=TRUE)}
#'
#' # Model selection using a specified pool of models
#' ourModel <- es(AirPassengers,model=c("ANN","AAM","AMdA"),h=18)
#'
#' # Produce forecast and prediction interval
#' forecast(ourModel, h=18, interval="parametric")
#'
#' # Semiparametric interval example
#' \donttest{forecast(ourModel, h=18, interval="semiparametric")}
#'
#' # This will be the same model as in previous line but estimated on new portion of data
#' \donttest{es(BJsales,model=ourModel,h=18,holdout=FALSE)}
#'
#' @rdname es
#' @export
es <- function(y, model="ZXZ", lags=c(frequency(y)), persistence=NULL, phi=NULL,
               initial=c("backcasting","optimal","two-stage","complete","gradient"),initialSeason=NULL,
               ic=c("AICc","AIC","BIC","BICc"),
               loss=c("likelihood","MSE","MAE","HAM","MSEh","TMSE","GTMSE","MSCE","GPL"),
               h=0, holdout=FALSE,
               bounds=c("usual","admissible","none"),
               smoother=c("default","ma","lowess","supsmu","global"),
               silent=TRUE,
               xreg=NULL, regressors=c("use","select"), initialX=NULL, ...){
    # Copyright (C) 2022 - Inf  Ivan Svetunkov

    # Start measuring the time of calculations
    startTime <- Sys.time();
    cl <- match.call();
    ellipsis <- list(...);

    # Check if the simulated thing is provided
    if(is.smooth.sim(y)){
        if(smoothType(y)=="ETS"){
            model <- y;
            y <- y$data;
        }
    }
    else if(is.smooth(y)){
        model <- y;
        y <- y$y;
    }

    # If a previous model provided as a model, write down the variables
    if(is.smooth(model) || is.smooth.sim(model)){
        if(smoothType(model)!="ETS"){
            stop("The provided model is not ETS.",call.=FALSE);
        }

        # If this is the simulated data, extract the parameters
        if(is.smooth.sim(model) & !is.null(dim(model$data))){
            warning("The provided model has several submodels. Choosing a random one.",call.=FALSE);
            i <- round(runif(1,1:length(model$persistence)));
            persistence <- model$persistence[,i];
            initial <- model$initial[,i];
            initialSeason <- model$initialSeason[,i];
        }
        else{
            persistence <- model$persistence;
            initial <- model$initial;
            # initialSeason <- model$initialSeason;
            if(any(model$probability!=1)){
                occurrence <- "a";
            }
        }
        phi <- model$phi;
        if(is.null(xreg) && is.list(model$initial) && !is.null(model$initial$xreg)){
            # Exctract xreg
            xreg <- model$data[,all.vars(model$formula)[-1]];
        }

        # initialX <- model$initialX;

        # if(is.adam(model)){
        #     y <- model$data;
        # }
        # else{
        #     y <- model$y;
        # }

        model <- modelType(model);
        # if(any(unlist(gregexpr("C",model))!=-1)){
        #     initial <- "o";
        # }
    }
    else if(inherits(model,"ets")){
        # Extract smoothing parameters
        i <- 1;
        persistence <- coef(model)[i];
        if(model$components[2]!="N"){
            i <- i+1;
            persistence <- c(persistence,coef(model)[i]);
            if(model$components[3]!="N"){
                i <- i+1;
                persistence <- c(persistence,coef(model)[i]);
            }
        }
        else{
            if(model$components[3]!="N"){
                i <- i+1;
                persistence <- c(persistence,coef(model)[i]);
            }
        }

        # Damping parameter
        if(model$components[4]=="TRUE"){
            i <- i+1;
            phi <- coef(model)[i];
        }

        # Initials
        i <- i+1;
        initial <- coef(model)[i];
        if(model$components[2]!="N"){
            i <- i+1;
            initial <- c(initial,coef(model)[i]);
        }

        # Initials of seasonal component
        if(model$components[3]!="N"){
            if(model$components[2]!="N"){
                initialSeason <- rev(model$states[1,-c(1:2)]);
            }
            else{
                initialSeason <- rev(model$states[1,-c(1)]);
            }
        }
        model <- modelType(model);
    }
    else if(is.character(model)){
        # Everything is okay
    }
    else{
        warning("A model of an unknown class was provided. Switching to 'ZZZ'.",call.=FALSE);
        model <- "ZZZ";
    }

    # Merge y and xreg into one data frame
    if(!is.null(xreg) && is.numeric(y)){
        if(is.matrix(xreg)){
            data <- as.matrix(cbind(y=as.data.frame(y),as.data.frame(xreg[1:length(y),])));
        }
        else{
            data <- as.matrix(cbind(y=as.data.frame(y),as.data.frame(xreg[1:length(y)])));
        }
        data <- ts(data, start=start(y), frequency=frequency(y));
        colnames(data)[1] <- "y";
        # Give name to the explanatory variables if they do not have them
        if(is.null(colnames(xreg))){
            if(!is.null(ncol(xreg))){
                colnames(data)[-1] <- paste0("x",c(1:ncol(xreg)));
            }
            else{
                colnames(data)[-1] <- "x";
            }
        }
    }
    else{
        data <- y;
    }

    # If the initial is provided in the old school style
    if(!is.list(initial)){
        # Prepare initials if they are numeric
        initialValue <- vector("list",(!is.null(initial))*1 +(!is.null(initialSeason))*1 +(!is.null(initialX))*1);
        names(initialValue) <- c("level","seasonal","xreg")[c(!is.null(initial),!is.null(initialSeason),!is.null(initialX))];
        if(is.numeric(initial)){
            initialValue <- switch(length(initial),
                                   "1"=list(level=initial[1]),
                                   "2"=list(level=initial[1],
                                            trend=initial[2]));
        }
        if(!is.null(initialSeason)){
            initialValue$seasonal <- initialSeason;
        }
        if(!is.null(initialX)){
            initialValue$xreg <- initialX;
        }
        if(length(initialValue)==1 && is.null(initialValue$level)){
            initialValue <- initial;
        }
    }
    else{
        initialValue <- initial;
    }

    # Warnings about the interval and cumulative
    if(!is.null(ellipsis$interval) && ellipsis$interval!="none"){
        warning("Parameter \"interval\" is no longer supported in es(). ",
                "Please use forecast() method to produce prediction interval.")
    }

    if(!is.null(ellipsis$cumulative) && ellipsis$cumulative!="none"){
        warning("Parameter \"cumulative\" is no longer supported in es(). ",
                "Please use forecast() method to produce cumulative values.")
    }

    ourModel <- adam(data=data, model=model, lags=lags, persistence=persistence, phi=phi,
                     loss=loss, h=h, holdout=holdout, initial=initialValue,
                     ic=ic, bounds=bounds, distribution="dnorm", smoother=smoother,
                     silent=silent, regressors=regressors[1], ets="conventional", ...);
    ourModel$call <- cl;
    ourModel$timeElapsed=Sys.time()-startTime;

    return(ourModel);
}
