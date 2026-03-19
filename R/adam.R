#' ADAM is Augmented Dynamic Adaptive Model
#'
#' Function constructs an advanced Single Source of Error model, based on ETS
#' taxonomy and ARIMA elements
#'
#' Function estimates ADAM in a form of the Single Source of Error state space
#' model of the following type:
#'
#' \deqn{y_{t} = o_t (w(v_{t-l}) + h(x_t, a_{t-1}) + r(v_{t-l}) \epsilon_{t})}
#'
#' \deqn{v_{t} = f(v_{t-l}, a_{t-1}) + g(v_{t-l}, a_{t-1}, x_{t}) \epsilon_{t}}
#'
#' Where \eqn{o_{t}} is the Bernoulli distributed random variable (in case of
#' normal data it equals to 1 for all observations), \eqn{v_{t}} is the state
#' vector and \eqn{l} is the vector of lags, \eqn{x_t} is the vector of
#' exogenous variables. w(.) is the measurement function, r(.) is the error
#' function, f(.) is the transition function, g(.) is the persistence
#' function and \eqn{a_t} is the vector of parameters for exogenous variables.
#' Finally, \eqn{\epsilon_{t}} is the error term.
#'
#' The implemented model allows introducing several seasonal states and supports
#' intermittent data via the \code{occurrence} variable.
#'
#' The error term \eqn{\epsilon_t} can follow different distributions, which
#' are regulated via the \code{distribution} parameter. This includes:
#' \enumerate{
#' \item \code{default} - Normal distribution is used for the Additive error models,
#' Gamma is used for the Multiplicative error models.
#' \item dnorm - \link[stats]{Normal} distribution,
#' \item \link[greybox]{dlaplace} - Laplace distribution,
#' \item \link[greybox]{ds} - S distribution,
#' \item \link[greybox]{dgnorm} - Generalised Normal distribution,
# \item \link[stats]{dlogis} - Logistic Distribution,
# \item \link[stats]{dt} - T distribution,
# \item \link[greybox]{dalaplace} - Asymmetric Laplace distribution,
#' \item \link[stats]{dlnorm} - Log-Normal distribution,
# \item dllaplace - Log-Laplace distribution,
# \item dls - Log-S distribution,
# \item dlgnorm - Log-Generalised Normal distribution,
# \item \link[greybox]{dbcnorm} - Box-Cox normal distribution,
#' \item \link[stats]{dgamma} - Gamma distribution,
#' \item \link[statmod]{dinvgauss} - Inverse Gaussian distribution,
#' }
#'
#' For some more information about the model and its implementation, see the
#' vignette: \code{vignette("adam","smooth")}. The more detailed explanation
#' of ADAM is provided by Svetunkov (2021).
#'
#' The function \code{auto.adam()} tries out models with the specified
#' distributions and returns the one with the most suitable one based on selected
#' information criterion.
#'
#' \link[greybox]{sm}.adam method estimates the scale model for the already
#' estimated adam. In order for ADAM to take the SM model into account, the
#' latter needs to be recorded in the former, amending the likelihood and the number
#' of degrees of freedom. This can be done using \link[greybox]{implant} method.
#'
#' @template ssAuthor
#' @template ssKeywords
#'
#' @template ADAMDataFormulaRegLossSilentHHoldout
#'
#' @template smoothRef
#' @template ssADAMRef
#' @template ssGeneralRef
#' @template ssIntermittentRef
#' @template ssETSRef
#'
#' @param model The type of ETS model. The first letter stands for the type of
#' the error term ("A" or "M"), the second (and sometimes the third as well) is for
#' the trend ("N", "A", "Ad", "M" or "Md"), and the last one is for the type of
#' seasonality ("N", "A" or "M"). In case of several lags, the seasonal components
#' are assumed to be the same. The model is then printed out as
#' ETS(M,Ad,M)[m1,m2,...], where m1, m2, ... are the lags specified by the
#' \code{lags} parameter.
#' There are several options for the \code{model} besides the conventional ones,
#' which rely on information criteria:
#' \enumerate{
#' \item \code{model="ZZZ"} means that the model will be selected based on the
#' chosen information criteria type. The Branch and Bound is used in the process.
#' \item \code{model="XXX"} means that only additive components are tested, using
#' Branch and Bound.
#' \item \code{model="YYY"} implies selecting between multiplicative components.
#' \item \code{model="CCC"} triggers the combination of forecasts of models using
#' information criteria weights (Kolassa, 2011).
#' \item combinations between these four and the classical components are also
#' accepted. For example, \code{model="CAY"} will combine models with additive
#' trend and either none or multiplicative seasonality.
#' \item \code{model="PPP"} will produce the selection between pure additive and
#' pure multiplicative models. "P" stands for "Pure". This cannot be mixed with
#' other types of components.
#' \item \code{model="FFF"} will select between all the 30 types of models. "F"
#' stands for "Full". This cannot be mixed with other types of components.
#' \item \code{model="SSS"} creates a pool of 19 standard sensible models, which
#' have finite variance. This can be combined with "X" or "Y" to further restrict
#' the pool. For example, \code{model="SXS"} corresponds to the default pool of 15
#' models in the \code{ets()} function from the forecast package.
#' \item The parameter \code{model} can also be a vector of names of models for a
#' finer tuning (pool of models). For example, \code{model=c("ANN","AAA")} will
#' estimate only two models and select the best of them.
#' }
#'
#' Also, \code{model} can accept a previously estimated adam and use all
#' its parameters.
#'
#' Keep in mind that model selection with "Z" components uses Branch and Bound
#' algorithm and may skip some models that could have slightly smaller
#' information criteria. If you want to do a exhaustive search, you would need
#' to list all the models to check as a vector.
#'
#' The default value is set to \code{"ZXZ"}, because the multiplicative trend is explosive
#' and dangerous. It should be used only for each separate time series, not for the
#' automated predictions for big  datasets.
#'
#' @param lags Defines lags for the corresponding components. All components
#' count, starting from level, so ETS(M,M,M) model for monthly data will have
#' \code{lags=c(1,1,12)}. However, the function will also accept \code{lags=c(12)},
#' assuming that the lags 1 were dropped. In case of ARIMA, lags specify what should be
#' the seasonal component lag. e.g. \code{lags=c(1,12)} will lead to the
#' seasonal ARIMA with m=12. This can accept several lags, supporting multiple seasonal ETS
#' and ARIMA models.
#' @param orders The order of ARIMA to be included in the model. This should be passed
#' either as a vector (in which case the non-seasonal ARIMA is assumed) or as a list of
#' a type \code{orders=list(ar=c(p,P),i=c(d,D),ma=c(q,Q))}, in which case the \code{lags}
#' variable is used in order to determine the seasonality m. See \link[smooth]{msarima}
#' for details.
#' In addition, \code{orders} accepts one more parameter: \code{orders=list(select=FALSE)}.
#' If \code{TRUE}, then the function will select the most appropriate order using a
#' mechanism similar to \code{auto.msarima()}, but implemented in \code{auto.adam()}.
#' The values \code{list(ar=...,i=...,ma=...)} specify the maximum orders to check in
#' this case.
#' @param constant Logical, determining, whether the constant is needed in the model or not.
#' This is mainly needed for ARIMA part of the model, but can be used for ETS as well. In
#' case of pure regression, this is completely ignored (use \code{formula} instead).
#' @param distribution what density function to assume for the error term. The full
#' name of the distribution should be provided, starting with the letter "d" -
#' "density". The names align with the names of distribution functions in R.
#' For example, see \link[stats]{dnorm}. For detailed explanation of available
#' distributions, see vignette in greybox package: \code{vignette("greybox","alm")}.
#' @param persistence Persistence vector \eqn{g}, containing smoothing
#' parameters. If \code{NULL}, then estimated. Can be also passed as a names list of
#' the type: \code{persistence=list(level=0.1, trend=0.05, seasonal=c(0.1,0.2),
#' xreg=c(0.1,0.2))}. Dropping some elements from the named list will make the function
#' estimate them. e.g. if you don't specify seasonal in the persistence for the ETS(M,N,M)
#' model, it will be estimated.
#' @param phi Value of damping parameter. If \code{NULL} then it is estimated.
#' Only applicable for damped-trend models.
#' @param initial Can be either character or a list, or a vector of initial states.
#' If it is character, then it can be \code{"backcasting"}, meaning that the initials of
#' dynamic part of the model are produced using backcasting procedure (advised
#' for data with high frequency), or \code{"optimal"}, meaning that all initial
#' states are optimised, or \code{"two-stage"}, meaning that optimisation is done
#' after the backcasting, refining the states. In case of backcasting, the parameters of the
#' explanatory variables are optimised. This is recommended for ETSX and ARIMAX
#' models. Alternatively, you can set \code{initial="complete"} backcasting,
#' which means that all states (including explanatory variables) are initialised
#' via backcasting.
#'
#' If a use provides a list of values, it is recommended to use the named one and
#' to provide the initial components that are available. For example:
#' \code{initial=list(level=1000,trend=10,seasonal=list(c(1,2),c(1,2,3,4)),
#' arima=1,xreg=100)}. If some of the components are needed by the model, but are
#' not provided in the list, they will be estimated. If the vector is provided,
#' then it is expected that the components will be provided in the same order as above,
#' one after another without any gaps.
#' @param arma Either the named list or a vector with AR / MA parameters ordered lag-wise.
#' The number of elements should correspond to the specified orders e.g.
#' \code{orders=list(ar=c(1,1),ma=c(1,1)), lags=c(1,4), arma=list(ar=c(0.9,0.8),ma=c(-0.3,0.3))}
#' @param occurrence The type of model used in probability estimation. Can be
#' \code{"none"} - none,
#' \code{"fixed"} - constant probability,
#' \code{"general"} - the general Beta model with two parameters,
#' \code{"odds-ratio"} - the Odds-ratio model with b=1 in Beta distribution,
#' \code{"inverse-odds-ratio"} - the model with a=1 in Beta distribution,
#' \code{"direct"} - the TSB-like (Teunter et al., 2011) probability update
#' mechanism a+b=1,
#' \code{"auto"} - the automatically selected type of occurrence model.
#'
#' The type of model used in the occurrence is equal to the one provided in the
#' \code{model} parameter.
#'
#' Also, a model produced using \link[smooth]{oes} or \link[greybox]{alm} function
#' can be used here.
#' @param ic The information criterion to use in the model selection / combination
#' procedure.
#' @param bounds The type of bounds for the persistence to use in the model
#' estimation. Can be either \code{admissible} - guaranteeing the stability of the
#' model, \code{usual} - restricting the values with (0, 1) or \code{none} - no
#' restrictions (potentially dangerous).
#' @param ets Parameter determining, which ETS formulation to use. If \code{ets="conventional"},
#' the one from Hyndman et al. (2008) is used. In case of \code{ets="adam"}, ADAM reformulation
#' that updates multiplicative components differently is used. The latter is closer
#' to applying ETS to log-transformed data when multiplicative components are used. This helps
#' makig trend less explosive.
#' @param ...  Other non-documented parameters. For example, \code{FI=TRUE} will
#' make the function also produce Fisher Information matrix, which then can be
#' used to calculated variances of smoothing parameters and initial states of
#' the model. This is calculated based on the hessian of log-likelihood function and
#' accepts \code{stepSize} parameter, determining how it is calculated. The default value
#' is \code{stepSize=.Machine$double.eps^(1/4)}. This is used in the \link[stats]{vcov} method.
#' Number of iterations inside the backcasting loop to do is regulated with \code{nIterations}
#' parameter. By default it is set to 2. Furthermore, starting values of parameters can be
#' passed via \code{B}, while the upper and lower bounds should be passed in \code{ub}
#' and \code{lb} respectively. In this case they will be used for optimisation. These
#' values should have the length equal to the number of parameters to estimate in
#' the following order:
#' \enumerate{
#' \item All smoothing parameters (for the states and then for the explanatory variables);
#' \item Damping parameter (if needed);
#' \item ARMA parameters;
#' \item All the initial values (for the states and then for the explanatory variables).
#' }
#' You can also pass parameters to the optimiser in order to fine tune its work:
#' \itemize{
#' \item \code{maxeval} - maximum number of evaluations to carry out. The default is 40 per
#' estimated parameter for ETS and / or ARIMA and at least 1000 if explanatory variables
#' are introduced in the model (100 per parameter for explanatory variables, but not less
#' than 1000);
#' \item \code{maxtime} - stop, when the optimisation time (in seconds) exceeds this;
#' \item \code{xtol_rel} - the relative precision of the optimiser (the default is 1E-6);
#' \item \code{xtol_abs} - the absolute precision of the optimiser (the default is 1E-8);
#' \item \code{ftol_rel} - the stopping criterion in case of the relative change in the loss
#' function (the default is 1E-8);
#' \item \code{ftol_abs} - the stopping criterion in case of the absolute change in the loss
#' function (the default is 0 - not used);
#' \item \code{algorithm} - the algorithm to use in optimisation
#' (by default, \code{"NLOPT_LN_NELDERMEAD"} is used);
#' \item \code{print_level} - the level of output for the optimiser (0 by default).
#' If equal to 41, then the detailed results of the optimisation are returned.
#' }
#' You can read more about these parameters by running the function
#' \link[nloptr]{nloptr.print.options}.
#' It is also possible to regulate what smoother to use to get initial seasonal indices
#' from the \link[smooth]{msdecompose} function via the \code{smoother} parameter. The default
#' value is \code{smoother="lowess"}.
#' Finally, the parameter \code{lambda} for LASSO / RIDGE, \code{alpha} for the Asymmetric
#' Laplace, \code{shape} for the Generalised Normal and \code{nu} for Student's distributions
#' can be provided here as well.
#'
#' @return Object of class "adam" is returned. It contains the list of the
#' following values:
#' \itemize{
#' \item \code{model} - the name of the constructed model,
#' \item \code{timeElapsed} - the time elapsed for the estimation of the model,
#' \item \code{data} - the in-sample part of the data used for the training of the model. Includes
#' the actual values in the first column,
#' \item \code{holdout} - the holdout part of the data, excluded for purposes of model evaluation,
#' \item \code{fitted} - the vector of fitted values,
#' \item \code{residuals} - the vector of residuals,
#' \item \code{forecast} - the point forecast for h steps ahead (by default NA is returned). NOTE
#' that these do not always correspond to the conditional expectations for ETS models. See ADAM
#' textbook, Section 6.4. for details (\url{https://openforecast.org/adam/ETSTaxonomyMaths.html}),
#' \item \code{states} - the matrix of states with observations in rows and states in columns,
#' \item \code{persisten} - the vector of smoothing parameters,
#' \item \code{phi} - the value of damping parameter,
#' \item \code{transition} - the transition matrix,
#' \item \code{measurement} - the measurement matrix with observations in rows and state elements
#' in columns,
#' \item \code{initial} - the named list of initial values, including level, trend, seasonal, ARIMA
#' and xreg components,
#' \item \code{initialEstimated} - the named vector, defining which of the initials were estimated in
#' the model,
#' \item \code{initialType} - the type of initialisation used (backcasting/optimal/two-stage/complete/provided),
#' \item \code{orders} - the orders of ARIMA used in the estimation,
#' \item \code{constant} - the value of the constant (if it was included),
#' \item \code{arma} - the list of AR / MA parameters used in the model,
#' \item \code{nParam} - the matrix of the estimated / provided parameters,
#' \item \code{occurrence} - the oes model used for the occurrence part of the model,
#' \item \code{formula} - the formula used for the explanatory variables expansion,
#' \item \code{loss} - the type of loss function used in the estimation,
#' \item \code{lossValue} - the value of that loss function,
#' \item \code{logLik} - the value of the log-likelihood,
#' \item \code{distribution} - the distribution function used in the calculation of the likelihood,
#' \item \code{scale} - the value of the scale parameter,
#' \item \code{lambda} - the value of the parameter used in LASSO / dalaplace / dt,
#' \item \code{B} - the vector of all estimated parameters,
#' \item \code{lags} - the vector of lags used in the model construction,
#' \item \code{lagsAll} - the vector of the internal lags used in the model,
#' \item \code{profile} - the matrix with the profile used in the construction of the model,
#' \item \code{profileInitial} - the matrix with the initial profile (for the before the sample values),
#' \item \code{call} - the call used in the evaluation,
#' \item \code{bounds} - the type of bounds used in the process,
#' \item \code{res} - result of the model estimation, the output of the \code{nloptr()} function, explaining
#' how optimisation went,
#' \item \code{other} - the list with other parameters, such as shape for distributions or ARIMA
#' polynomials.
#' }
#'
#' @seealso \code{\link[smooth]{es}, \link[smooth]{msarima}}
#'
#' @examples
#'
#' ### The main examples are provided in the adam vignette, check it out via:
#' \dontrun{vignette("adam","smooth")}
#'
#' # Model selection using a specified pool of models
#' ourModel <- adam(rnorm(100,100,10), model=c("ANN","ANA","AAA"), lags=c(5,10))
#' \donttest{adamSummary <- summary(ourModel)
#' xtable(adamSummary)}
#'
#' \donttest{forecast(ourModel)
#' par(mfcol=c(3,4))
#' plot(ourModel, c(1:11))}
#'
#' # Model combination using a specified pool
#' \donttest{ourModel <- adam(rnorm(100,100,10), model=c("ANN","AAN","MNN","CCC"),
#'                           lags=c(5,10))}
#'
#' # ADAM ARIMA
#' \donttest{ourModel <- adam(rnorm(100,100,10), model="NNN",
#'                           lags=c(1,4), orders=list(ar=c(1,0),i=c(1,0),ma=c(1,1)))}
#'
#' @importFrom greybox dlaplace dalaplace ds dgnorm
#' @importFrom greybox stepwise alm is.occurrence is.alm polyprod
#' @importFrom stats dnorm dlogis dt dlnorm dgamma frequency confint vcov predict
#' @importFrom stats formula update model.frame model.matrix contrasts setNames terms reformulate
#' @importFrom stats acf pacf
#' @importFrom statmod dinvgauss
#' @importFrom nloptr nloptr
#' @importFrom pracma hessian
#' @importFrom zoo zoo
#' @importFrom utils head
#' @importFrom methods new
#' @rdname adam
#' @export adam
adam <- function(data, model="ZXZ", lags=c(frequency(data)), orders=list(ar=c(0),i=c(0),ma=c(0),select=FALSE),
                 constant=FALSE, formula=NULL, regressors=c("use","select","adapt"),
                 occurrence=c("none","auto","fixed","general","odds-ratio","inverse-odds-ratio","direct"),
                 distribution=c("default","dnorm","dlaplace","ds","dgnorm",
                                "dlnorm","dinvgauss","dgamma"),
                 loss=c("likelihood","MSE","MAE","HAM","LASSO","RIDGE","MSEh","TMSE","GTMSE","MSCE","GPL"),
                 outliers=c("ignore","use","select"), level=0.99,
                 h=0, holdout=FALSE,
                 persistence=NULL, phi=NULL, initial=c("backcasting","optimal","two-stage","complete"), arma=NULL,
                 ic=c("AICc","AIC","BIC","BICc"), bounds=c("usual","admissible","none"),
                 silent=TRUE, ets=c("conventional","adam"), ...){
    # Copyright (C) 2019 - Inf  Ivan Svetunkov

    # Start measuring the time of calculations
    startTime <- Sys.time();

    cl <- match.call();
    # Record the parental environment. Needed for ARIMA initialisation
    env <- parent.frame();
    ellipsis <- list(...);
    # Assume that the model is not provided
    profilesRecentProvided <- FALSE;
    profilesRecentTable <- NULL;

    # paste0() is needed in order to get rid of potential issues with names
    yName <- paste0(deparse(substitute(data)),collapse="");

    # If a previous model is provided as a model, write down the variables
    if(is.adam(model) || is.adam.sim(model)){
        # If this is the simulated data, extract the parameters
        # if(is.adam.sim(model) & !is.null(dim(model$data))){
        #     warning("The provided model has several submodels. Choosing a random one.",call.=FALSE);
        #     i <- round(runif(1,1:length(model$persistence)));
        #     persistence <- model$persistence[,i];
        #     initial <- model$initial[,i];
        #     initialSeason <- model$initialSeason[,i];
        #     if(any(model$iprob!=1)){
        #         occurrence <- "a";
        #     }
        # }
        # else{
        initial <- model$initial;
        initialEstimated <- model$initialEstimated;
        distribution <- model$distribution;
        loss <- model$loss;
        persistence <- model$persistence;
        phi <- model$phi;
        # If this was complete backcasting, set the basic one
        if(model$initialType=="complete"){
            initial <- "complete";
        }
        # If it was backcasting, but no xreg, set it as one
        else if(model$initialType=="backcasting"){
            if(!is.null(model$initial$xreg)){
                initial <- model$initial;
            }
            else{
                initial <- "backcasting";
            }
        }
        else{
            initial <- model$initial;
        }
        occurrence <- model$occurrence;
        ic <- model$ic;
        bounds <- model$bounds;
        # lambda for LASSO
        ellipsis$lambda <- model$other$lambda;
        # parameters for distributions
        ellipsis$alpha <- model$other$alpha;
        ellipsis$shape <- model$other$shape;
        ellipsis$nu <- model$other$nu;
        ellipsis$B <- model$B;
        CFValue <- model$lossValue;
        logLikADAMValue <- logLik(model);
        lagsModelAll <- modelLags(model);
        # This needs to be fixed to align properly in case of various seasonals
        profilesRecentTable <- model$profileInitial;
        profilesRecentProvided[] <- TRUE;
        regressors <- model$regressors;
        if(is.null(formula)){
            formula <- formula(model);
        }

        # Parameters of the original ARIMA model
        lags <- lags(model);
        orders <- orders(model);
        constant <- model$constant;
        if(is.null(constant)){
            constant <- FALSE;
        }
        arma <- model$arma;

        model <- modelType(model);
        modelDo <- "use";
        # if(any(unlist(gregexpr("C",model))!=-1)){
        #     initial <- "o";
        # }
    }
    else if(inherits(model,"ets")){
        # Extract smoothing parameters
        i <- 1;
        lags <- 1;
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
        # Initial for the trend
        if(model$components[2]!="N"){
            i <- i+1;
            lags <- c(lags,1);
            initial <- c(initial,coef(model)[i]);
        }

        # Initials of seasonal component
        if(model$components[3]!="N"){
            if(model$components[2]!="N"){
                initial <- c(initial,rev(model$states[1,-c(1:2)]));
            }
            else{
                initial <- c(initial,rev(model$states[1,-c(1)]));
            }
            lags <- c(lags,model$m);
        }
        model <- modelType(model);
        distribution <- "dnorm";
        loss <- "likelihood";
        modelDo <- "use"
    }
    else if(is.character(model)){
        modelDo <- "";
        # Everything is okay
    }
    else{
        modelDo <- "";
        warning("A model of an unknown class was provided. Switching to 'ZZZ'.",call.=FALSE);
        model <- "ZZZ";
    }

    ets <- match.arg(ets);
    adamETS <- ets=="adam";

    #### Check the parameters of the function and create variables based on them ####
    checkerReturn <- parametersChecker(data, model, lags, formula, orders, constant, arma,
                                       outliers, level,
                                       persistence, phi, initial,
                                       distribution, loss, h, holdout, occurrence, ic, bounds,
                                       regressors, yName,
                                       silent, modelDo, ParentEnvironment=environment(), ellipsis, fast=FALSE);

    #### Return regression if it is pure ####
    if(is.alm(checkerReturn)){
        obsInSample <- nobs(checkerReturn);
        nParam <- length(checkerReturn$coefficient);

        modelReturned <- list(model="Regression");
        modelReturned$timeElapsed <- Sys.time()-startTime;
        modelReturned$call <- checkerReturn$call;
        if(is.null(formula)){
            formula <- formula(checkerReturn);
        }
        if(holdout){
            # Robustify the names of variables
            colnames(data) <- make.names(colnames(data),unique=TRUE);
            modelReturned$holdout <- data[obsInSample+c(1:h),,drop=FALSE];
        }
        else{
            modelReturned$holdout <- NULL;
        }
        responseName <- all.vars(formula)[1];
        y <- data[,responseName];
        # Extract indeces from the data
        yIndex <- try(time(y),silent=TRUE);
        # If we cannot extract time, do something
        if(inherits(yIndex,"try-error")){
            if(!is.data.frame(data) && !is.null(dim(data))){
                yIndex <- as.POSIXct(rownames(data));
            }
            else if(is.data.frame(data)){
                yIndex <- c(1:nrow(data));
            }
            else{
                yIndex <- c(1:length(data));
            }
        }

        # Prepare fitted, residuals and the forecasts
        if(inherits(y ,"zoo")){
            modelReturned$data <- data[1:obsInSample,,drop=FALSE];
            modelReturned$fitted <- zoo(fitted(checkerReturn), order.by=yIndex[1:obsInSample]);
            modelReturned$residuals <- zoo(residuals(checkerReturn), order.by=yIndex[1:obsInSample]);
            # If we need to forecast and we had holdout=FALSE...
            if(h>0){
                if(holdout){
                    modelReturned$forecast <- zoo(forecast(checkerReturn,h=h,newdata=tail(data,h),interval="none")$mean,
                                                  order.by=yIndex[obsInSample+1:h]);
                }
                else{
                    modelReturned$forecast <- zoo(forecast(checkerReturn,h=h,interval="none")$mean,
                                                  order.by=yIndex[obsInSample+1:h]);
                }
            }
            else{
                modelReturned$forecast <- zoo(NA, order.by=yIndex[obsInSample+1]);
            }
            modelReturned$states <- zoo(matrix(coef(checkerReturn), obsInSample+1, nParam, byrow=TRUE,
                                               dimnames=list(NULL, names(coef(checkerReturn)))),
                                        order.by=c(yIndex[1]-diff(yIndex[1:2]),yIndex[1:obsInSample]));
        }
        else{
            yFrequency <- frequency(y);
            modelReturned$data <- ts(data[1:obsInSample,,drop=FALSE], start=yIndex[1], frequency=yFrequency);
            modelReturned$fitted <- ts(fitted(checkerReturn), start=yIndex[1], frequency=yFrequency);
            modelReturned$residuals <- ts(residuals(checkerReturn), start=yIndex[1], frequency=yFrequency);
            if(h>0){
                if(holdout){
                    modelReturned$forecast <- ts(forecast(checkerReturn,h=h,newdata=tail(data,h),interval="none")$mean,
                                                 start=yIndex[obsInSample+1], frequency=yFrequency);
                }
                else{
                    modelReturned$forecast <- ts(as.numeric(forecast(checkerReturn,h=h,interval="none")$mean),
                                                 start=yIndex[obsInSample]+diff(yIndex[1:2]), frequency=yFrequency);
                }
            }
            else{
                modelReturned$forecast <- ts(NA, start=yIndex[obsInSample]+diff(yIndex[1:2]), frequency=yFrequency);
            }
            modelReturned$states <- ts(matrix(coef(checkerReturn), obsInSample+1, nParam, byrow=TRUE,
                                              dimnames=list(NULL, names(coef(checkerReturn)))),
                                       start=yIndex[1]-diff(yIndex[1:2]), frequency=yFrequency);
        }
        modelReturned$persistence <- rep(0, nParam);
        names(modelReturned$persistence) <- paste0("delta",c(1:nParam));
        modelReturned$phi <- 1;
        modelReturned$transition <- diag(nParam);
        modelReturned$measurement <- checkerReturn$data;
        modelReturned$measurement[,1] <- 1;
        colnames(modelReturned$measurement) <- colnames(modelReturned$states);
        modelReturned$initial <- list(xreg=coef(checkerReturn));
        modelReturned$initialType <- "optimal";
        modelReturned$initialEstimated <- TRUE;
        names(modelReturned$initialEstimated) <- "xreg";
        modelReturned$orders <- list(ar=0,i=0,ma=0);
        modelReturned$arma <- NULL;
        # Number of estimated parameters
        parametersNumber <- matrix(0,2,5,
                                   dimnames=list(c("Estimated","Provided"),
                                                 c("nParamInternal","nParamXreg","nParamOccurrence","nParamScale","nParamAll")));
        parametersNumber[1,2] <- nParam;
        if(is.occurrence(checkerReturn$occurrence)){
            parametersNumber[1,3] <- nParam;
        }
        parametersNumber[1,5] <- sum(parametersNumber[1,1:3]);
        modelReturned$nParam <- parametersNumber;
        modelReturned$formula <- formula(checkerReturn);
        modelReturned$regressors <- "use";
        modelReturned$loss <- checkerReturn$loss;
        modelReturned$lossValue <- checkerReturn$lossValue;
        modelReturned$lossFunction <- checkerReturn$lossFunction;
        modelReturned$logLik <- logLik(checkerReturn);
        modelReturned$distribution <- checkerReturn$distribution;
        modelReturned$scale <- checkerReturn$scale;
        modelReturned$other <- checkerReturn$other;
        modelReturned$B <- coef(checkerReturn);
        modelReturned$lags <- 1;
        modelReturned$lagsAll <- rep(1,nParam);
        modelReturned$FI <- checkerReturn$FI;
        modelReturned$occurrence <- checkerReturn$occurrence;
        if(holdout){
            # This won't work if transformations of the response variable are done...
            modelReturned$accuracy <- measures(modelReturned$holdout[,responseName],modelReturned$forecast,
                                               modelReturned$data[,responseName]);
        }
        else{
            modelReturned$accuracy <- NULL;
        }
        class(modelReturned) <- c("adam","smooth");
        if(!silent){
            plot(modelReturned,7)
        }

        return(modelReturned);
    }

    #### If select was provided in the model, do auto.adam selection ####
    if(!is.null(checkerReturn$select) && checkerReturn$select){
        return(do.call("auto.adam",list(data=substitute(data), model=model, lags=lags, orders=orders,
                                        formula=formula, regressors=regressors,
                                        distribution=distribution, loss=loss,
                                        h=h, holdout=holdout, outliers=outliers, level=level,
                                        persistence=persistence, phi=phi, initial=initial, arma=arma,
                                        occurrence=occurrence,
                                        ic=ic, bounds=bounds, silent=silent, ...)));
    }

    # This is the variable needed for the C++ code to determine whether the head of data needs to be
    # refined. Only needed for the ETS(*,Z,*) models
    refineHead <- TRUE;
    # if(arimaModel){
    #     refineHead[] <- FALSE;
    # }
    # if(initialType!="backcasting" | componentsNumberARIMA==0){
    #     refineHead[] <- TRUE;
    # }
    # if(initialType=="provided"){
    #     refineHead[] <- FALSE;
    # }

    #### The function creates the technical variables (lags etc) based on the type of the model ####
    architector <- function(etsModel, Etype, Ttype, Stype, lags, lagsModelSeasonal,
                            xregNumber, obsInSample, initialType,
                            arimaModel, lagsModelARIMA, xregModel, constantRequired,
                            profilesRecentTable=NULL, profilesRecentProvided=FALSE){
        # If there is ETS
        if(etsModel){
            modelIsTrendy <- Ttype!="N";
            if(modelIsTrendy){
                # Make lags (1, 1)
                lagsModel <- matrix(c(1,1),ncol=1);
                componentsNamesETS <- c("level","trend");
            }
            else{
                # Make lags (1, ...)
                lagsModel <- matrix(c(1),ncol=1);
                componentsNamesETS <- c("level");
            }
            modelIsSeasonal <- Stype!="N";
            if(modelIsSeasonal){
                # If the lags are for the non-seasonal model
                lagsModel <- matrix(c(lagsModel,lagsModelSeasonal),ncol=1);
                componentsNumberETSSeasonal <- length(lagsModelSeasonal);
                if(componentsNumberETSSeasonal>1){
                    componentsNamesETS <- c(componentsNamesETS,paste0("seasonal",c(1:componentsNumberETSSeasonal)));
                }
                else{
                    componentsNamesETS <- c(componentsNamesETS,"seasonal");
                }
            }
            else{
                componentsNumberETSSeasonal <- 0;
            }
            lagsModelAll <- lagsModel;

            componentsNumberETS <- length(lagsModel);
        }
        else{
            modelIsTrendy <- modelIsSeasonal <- FALSE;
            componentsNumberETS <- componentsNumberETSSeasonal <- 0;
            componentsNamesETS <- NULL;
            lagsModelAll <- lagsModel <- NULL;
        }

        # If there is ARIMA
        if(arimaModel){
            lagsModelAll <- matrix(c(lagsModel,lagsModelARIMA), ncol=1);
        }

        # If constant is needed, add it
        if(constantRequired){
            lagsModelAll <- matrix(c(lagsModelAll,1), ncol=1);
        }

        # If there are xreg
        if(xregModel){
            lagsModelAll <- matrix(c(lagsModelAll,rep(1,xregNumber)), ncol=1);
        }

        lagsModelMax <- max(lagsModelAll);

        # Define the number of cols that should be in the matvt
        obsStates <- obsInSample + lagsModelMax;

        # Create ADAM profiles for correct treatment of seasonality
        adamProfiles <- adamProfileCreator(lagsModelAll, lagsModelMax, obsAll,
                                           lags=lags, yIndex=yIndexAll, yClasses=yClasses);
        if(profilesRecentProvided){
            profilesRecentTable <- profilesRecentTable[,1:lagsModelMax,drop=FALSE];
        }
        else{
            profilesRecentTable <- adamProfiles$recent;
        }
        indexLookupTable <- adamProfiles$lookup;

        # Create C++ adam class, which will then use fit, forecast etc methods
        adamCpp <- new(adamCore,
                       lagsModelAll, Etype, Ttype, Stype,
                       componentsNumberETSNonSeasonal,
                       componentsNumberETSSeasonal,
                       componentsNumberETS, componentsNumberARIMA,
                       xregNumber, length(lagsModelAll),
                       constantRequired, adamETS);

        return(list(lagsModel=lagsModel,lagsModelAll=lagsModelAll, lagsModelMax=lagsModelMax,
                    componentsNumberETS=componentsNumberETS, componentsNumberETSSeasonal=componentsNumberETSSeasonal,
                    componentsNumberETSNonSeasonal=componentsNumberETS-componentsNumberETSSeasonal,
                    componentsNamesETS=componentsNamesETS, obsStates=obsStates, modelIsTrendy=modelIsTrendy,
                    modelIsSeasonal=modelIsSeasonal,
                    indexLookupTable=indexLookupTable, profilesRecentTable=profilesRecentTable,
                    adamCpp=adamCpp));
    }

    #### The function creates the necessary matrices based on the model and provided parameters ####
    # This is needed in order to initialise the estimation
    creator <- function(etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal,
                        lags, lagsModel, lagsModelARIMA, lagsModelAll, lagsModelMax,
                        profilesRecentTable=NULL, profilesRecentProvided=FALSE,
                        obsStates, obsInSample, obsAll, componentsNumberETS, componentsNumberETSSeasonal,
                        componentsNamesETS, otLogical, yInSample,
                        # Persistence and phi
                        persistence, persistenceEstimate,
                        persistenceLevel, persistenceLevelEstimate, persistenceTrend, persistenceTrendEstimate,
                        persistenceSeasonal, persistenceSeasonalEstimate,
                        persistenceXreg, persistenceXregEstimate, persistenceXregProvided,
                        phi,
                        # Initials
                        initialType, initialEstimate,
                        initialLevel, initialLevelEstimate, initialTrend, initialTrendEstimate,
                        initialSeasonal, initialSeasonalEstimate,
                        initialArima, initialArimaEstimate, initialArimaNumber,
                        initialXregEstimate, initialXregProvided,
                        # ARIMA elements
                        arimaModel, arRequired, iRequired, maRequired, armaParameters,
                        arOrders, iOrders, maOrders,
                        componentsNumberARIMA, componentsNamesARIMA,
                        # Explanatory variables
                        xregModel, xregModelInitials, xregData, xregNumber, xregNames,
                        xregParametersPersistence,
                        # Constant
                        constantRequired, constantEstimate, constantValue, constantName,
                        adamCpp){

        # Matrix of states. Time in columns, components in rows
        matVt <- matrix(NA, componentsNumberETS+componentsNumberARIMA+xregNumber+constantRequired, obsStates,
                        dimnames=list(c(componentsNamesETS,componentsNamesARIMA,xregNames,constantName),NULL));

        # Measurement rowvector
        matWt <- matrix(1, obsAll, componentsNumberETS+componentsNumberARIMA+xregNumber+constantRequired,
                        dimnames=list(NULL,c(componentsNamesETS,componentsNamesARIMA,xregNames,constantName)));

        # If xreg are provided, then fill in the respective values in Wt vector
        if(xregModel){
            matWt[,componentsNumberETS+componentsNumberARIMA+1:xregNumber] <- xregData;
        }

        # Transition matrix
        matF <- diag(componentsNumberETS+componentsNumberARIMA+xregNumber+constantRequired);

        # Persistence vector
        vecG <- matrix(0, componentsNumberETS+componentsNumberARIMA+xregNumber+constantRequired, 1,
                       dimnames=list(c(componentsNamesETS,componentsNamesARIMA,xregNames,constantName),NULL));

        j <- 0;
        # ETS model, persistence
        if(etsModel){
            j <- j+1;
            rownames(vecG)[j] <- "alpha";
            if(!persistenceLevelEstimate){
                vecG[j,] <- persistenceLevel;
            }
            if(modelIsTrendy){
                j <- j+1;
                rownames(vecG)[j] <- "beta";
                if(!persistenceTrendEstimate){
                    vecG[j,] <- persistenceTrend;
                }
            }
            if(modelIsSeasonal){
                if(!all(persistenceSeasonalEstimate)){
                    vecG[j+which(!persistenceSeasonalEstimate),] <- persistenceSeasonal;
                }
                if(componentsNumberETSSeasonal>1){
                    rownames(vecG)[j+c(1:componentsNumberETSSeasonal)] <- paste0("gamma",c(1:componentsNumberETSSeasonal));
                }
                else{
                    rownames(vecG)[j+1] <- "gamma";
                }
                j <- j+componentsNumberETSSeasonal;
            }
        }

        # ARIMA model, names for persistence
        if(arimaModel){
            # Remove diagonal from the ARIMA part of the matrix
            matF[j+1:componentsNumberARIMA,j+1:componentsNumberARIMA] <- 0;
            if(componentsNumberARIMA>1){
                rownames(vecG)[j+1:componentsNumberARIMA] <- paste0("psi",c(1:componentsNumberARIMA));
            }
            else{
                rownames(vecG)[j+1:componentsNumberARIMA] <- "psi";
            }
            j <- j+componentsNumberARIMA;
        }

        # Modify transition to do drift
        if(!arimaModel && constantRequired){
            matF[1,ncol(matF)] <- 1;
        }

        # Regression, persistence
        if(xregModel){
            if(persistenceXregProvided && !persistenceXregEstimate){
                vecG[j+1:xregNumber,] <- persistenceXreg;
            }
            rownames(vecG)[j+1:xregNumber] <- paste0("delta",xregParametersPersistence);
        }

        # Damping parameter value
        if(etsModel && modelIsTrendy){
            matF[1,2] <- phi;
            matF[2,2] <- phi;

            matWt[,2] <- phi;
        }

        # If the arma parameters were provided, fill in the persistence
        if(arimaModel && (!arEstimate && !maEstimate)){
            # Call polynomial
            arimaPolynomials <- lapply(adamCpp$polynomialise(0, arOrders, iOrders, maOrders,
                                                             arEstimate, maEstimate, armaParameters, lags), as.vector);
            # Fill in the transition matrix
            if(nrow(nonZeroARI)>0){
                matF[componentsNumberETS+nonZeroARI[,2],componentsNumberETS+nonZeroARI[,2]] <-
                    -arimaPolynomials$ariPolynomial[nonZeroARI[,1]];
            }
            # Fill in the persistence vector
            if(nrow(nonZeroARI)>0){
                vecG[componentsNumberETS+nonZeroARI[,2]] <- -arimaPolynomials$ariPolynomial[nonZeroARI[,1]];
            }
            if(nrow(nonZeroMA)>0){
                vecG[componentsNumberETS+nonZeroMA[,2]] <- vecG[componentsNumberETS+nonZeroMA[,2]] +
                    arimaPolynomials$maPolynomial[nonZeroMA[,1]];
            }
        }
        else{
            arimaPolynomials <- NULL;
        }

        if(!profilesRecentProvided){
            # ETS model, initial state
            # If something needs to be estimated...
            if(etsModel){
                if(initialEstimate){
                    # For the seasonal models
                    if(modelIsSeasonal){
                        # !!! Use deterministic trend. This way g=0 means we fit the global model to the data
                        yDecompositionAdditive <- msdecompose(yInSample, lags=lags[lags!=1],
                                                              type="additive",
                                                              smoother=smoother);
                        if(any(c(Etype,Ttype,Stype)=="M")){
                            yDecompositionMultiplicative <- msdecompose(yInSample, lags=lags[lags!=1],
                                                                        type="multiplicative",
                                                                        smoother=smoother);
                        }

                        # If either Etype or Stype are multiplicative, use multiplicative decomposition
                        # This is needed for the correct seasonal indices
                        decompositionType <- c("additive","multiplicative")[any(c(Etype,Stype)=="M")+1];
                        yDecomposition <- switch(decompositionType,
                                                 "additive"=yDecompositionAdditive,
                                                 "multiplicative"=yDecompositionMultiplicative);
                        j <- 1;
                        # level
                        if(initialLevelEstimate){
                            # If there's a trend, use the intercept from the deterministic one
                            if(modelIsTrendy){
                                matVt[j,1:lagsModelMax] <- switch(Ttype,
                                                                  "M"=yDecompositionMultiplicative$initial$nonseasonal[1],
                                                                  yDecompositionAdditive$initial$nonseasonal[1]);
                            }
                            # If not, use the global mean
                            else{
                                matVt[j,1:lagsModelMax] <- mean(yInSample[otLogical]);
                            }
                            if(xregModel){
                                if(Etype=="A"){
                                    matVt[j,1:lagsModelMax] <- matVt[j,1:lagsModelMax] -
                                        as.vector(xregModelInitials[[1]]$initialXreg %*% xregData[1,]);
                                }
                                else{
                                    matVt[j,1:lagsModelMax] <- matVt[j,1:lagsModelMax] /
                                        as.vector(exp(xregModelInitials[[2]]$initialXreg %*% xregData[1,]));
                                }
                            }
                        }
                        else{
                            matVt[j,1:lagsModelMax] <- initialLevel;
                        }
                        j <- j+1;
                        # If trend is needed
                        if(modelIsTrendy){
                            if(initialTrendEstimate){
                                if(Ttype=="A"){
                                    matVt[j,1:lagsModelMax] <- yDecompositionAdditive$initial$nonseasonal[2];
                                    if(Stype=="M"){
                                        # If the initial trend is higher than the lowest value, initialise with zero.
                                        # This is a failsafe mechanism for the mixed models
                                        if(matVt[j,1]<0 && abs(matVt[j,1])>min(abs(yInSample[otLogical]))){
                                            matVt[j,1:lagsModelMax] <- 0;
                                        }
                                    }
                                }
                                else if(Ttype=="M"){
                                    matVt[j,1:lagsModelMax] <- yDecompositionMultiplicative$initial$nonseasonal[2];

                                    # This is a failsafe for multiplicative trend models, so that the thing does not explode
                                    if(any(matVt[1,1:lagsModelMax]<0)){
                                        matVt[1,1:lagsModelMax] <- yInSample[otLogical][1];
                                    }
                                }
                            }
                            else{
                                matVt[j,1:lagsModelMax] <- initialTrend;
                            }
                            j <- j+1;
                        }
                        #### Seasonal components
                        # For pure models use stuff as is
                        if(all(c(Etype,Stype)=="A") || all(c(Etype,Stype)=="M") ||
                           (Etype=="A" & Stype=="M")){
                            for(i in 1:componentsNumberETSSeasonal){
                                if(initialSeasonalEstimate[i]){
                                    matVt[i+j-1,1:lagsModel[i+j-1]] <- yDecomposition$initial$seasonal[[i]];
                                    # Renormalise the initial seasons
                                    if(Stype=="A"){
                                        matVt[i+j-1,1:lagsModel[i+j-1]] <-
                                            matVt[i+j-1,1:lagsModel[i+j-1]] -
                                            mean(matVt[i+j-1,1:lagsModel[i+j-1]]);
                                    }
                                    else{
                                        matVt[i+j-1,1:lagsModel[i+j-1]] <-
                                            matVt[i+j-1,1:lagsModel[i+j-1]] /
                                            exp(mean(log(matVt[i+j-1,1:lagsModel[i+j-1]])));
                                    }
                                }
                                else{
                                    matVt[i+j-1,1:lagsModel[i+j-1]] <- initialSeasonal[[i]];
                                }
                            }
                        }
                        # For mixed models use a different set of initials
                        else if(Etype=="M" && Stype=="A"){
                            for(i in 1:componentsNumberETSSeasonal){
                                if(initialSeasonalEstimate[i]){
                                    matVt[i+j-1,1:lagsModel[i+j-1]] <-
                                        log(yDecomposition$initial$seasonal[[i]])*min(yInSample[otLogical]);
                                    # Renormalise the initial seasons
                                    if(Stype=="A"){
                                        matVt[i+j-1,1:lagsModel[i+j-1]] <- matVt[i+j-1,1:lagsModel[i+j-1]] -
                                            mean(matVt[i+j-1,1:lagsModel[i+j-1]]);
                                    }
                                    else{
                                        matVt[i+j-1,1:lagsModel[i+j-1]] <- matVt[i+j-1,1:lagsModel[i+j-1]] /
                                            exp(mean(log(matVt[i+j-1,1:lagsModel[i+j-1]])));
                                    }
                                }
                                else{
                                    matVt[i+j-1,1:lagsModel[i+j-1]] <- initialSeasonal[[i]];
                                }
                            }
                        }

                        # Failsafe in case negatives were produced
                        if(Etype=="M" && matVt[1,1]<=0){
                            matVt[1,1:lagsModelMax] <- yInSample[1];
                        }
                    }
                    # Non-seasonal models
                    else{
                        # This decomposition does not produce seasonal component
                        # If either Etype or Stype are multiplicative, do multiplicative decomposition
                        # decompositionType <- c("additive","multiplicative")[any(c(Etype,Stype)=="M")+1];
                        # # !!! Use deterministic trend. This way g=0 means we fit the global model to the data
                        # yDecomposition <- msdecompose(yInSample, lags=1, type=decompositionType,
                        #                               smoother=smoother);
                        yDecompositionAdditive <- msdecompose(yInSample, lags=1,
                                                              type="additive",
                                                              smoother=smoother);
                        if(any(c(Etype,Ttype)=="M")){
                            yDecompositionMultiplicative <- msdecompose(yInSample, lags=1,
                                                                        type="multiplicative",
                                                                        smoother=smoother);
                        }
                        # level
                        if(initialLevelEstimate){
                            # If there's a trend, use the intercept from the deterministic one
                            if(modelIsTrendy){
                                # matVt[1,1:lagsModelMax] <- mean(yInSample[1:max(lagsModelMax,ceiling(obsInSample*0.2))]);
                                matVt[1,1:lagsModelMax] <- switch(Ttype,
                                                                  "M"=yDecompositionMultiplicative$initial$nonseasonal[1],
                                                                  yDecompositionAdditive$initial$nonseasonal[1]);
                            }
                            # If not, use the global mean
                            else{
                                matVt[1,1:lagsModelMax] <- mean(yInSample[otLogical]);
                            }
                        }
                        else{
                            matVt[1,1:lagsModelMax] <- initialLevel;
                        }
                        if(modelIsTrendy){
                            if(initialTrendEstimate){
                                matVt[2,1:lagsModelMax] <- switch(Ttype,
                                                                  "A"=yDecompositionAdditive$initial$nonseasonal[2],
                                                                  "M"=yDecompositionMultiplicative$initial$nonseasonal[2]);
                            }
                            else{
                                matVt[2,1:lagsModelMax] <- initialTrend;
                            }
                        }
                        # Failsafe in case negatives were produced
                        if(Etype=="M" && matVt[1,1]<=0){
                            matVt[1,1:lagsModelMax] <- yInSample[1];
                        }
                    }

                    if(initialLevelEstimate && Etype=="M" && matVt[1,lagsModelMax]==0){
                        matVt[1,1:lagsModelMax] <- mean(yInSample);
                    }
                }
                # Else, insert the provided ones... make sure that this is not a backcasting
                else if(!initialEstimate && initialType=="provided"){
                    j <- 1;
                    matVt[j,1:lagsModelMax] <- initialLevel;
                    if(modelIsTrendy){
                        j <- j+1;
                        matVt[j,1:lagsModelMax] <- initialTrend;
                    }
                    if(modelIsSeasonal){
                        for(i in 1:componentsNumberETSSeasonal){
                            # This is misaligned, but that's okay, because this goes directly to profileRecent
                            # matVt[j+i,(lagsModelMax-lagsModel[j+i])+1:lagsModel[j+i]] <- initialSeasonal[[i]];
                            matVt[j+i,1:lagsModel[j+i]] <- initialSeasonal[[i]];
                        }
                    }
                    j <- j+componentsNumberETSSeasonal;
                }
            }

            # If ARIMA orders are specified, prepare initials
            if(arimaModel){
                if(initialArimaEstimate){
                    matVt[componentsNumberETS+1:componentsNumberARIMA, 1:initialArimaNumber] <-
                        switch(Etype, "A"=0, "M"=1);
                    # Do decomposition if we have at least 2 years of data
                    if(any(lags>1) && obsInSample > max(lags)*2){
                        yDecomposition <- tail(msdecompose(yInSample,
                                                           lags=lags[lags!=1],
                                                           type=switch(Etype,
                                                                       "A"="additive",
                                                                       "M"="multiplicative"),
                                                           smoother=smoother)$seasonal,1)[[1]];
                    }
                    else if(any(lags>1) && obsInSample <= max(lags)*2){
                        yDecomposition <- yInSample[otLogical][1:obsInSample];
                    }
                    else{
                        yDecomposition <- switch(Etype,
                                                 "A"=mean(diff(yInSample[otLogical])),
                                                 "M"=exp(mean(diff(log(yInSample[otLogical])))));
                    }
                    matVt[componentsNumberETS+componentsNumberARIMA, 1:initialArimaNumber] <-
                        rep(yDecomposition,ceiling(initialArimaNumber/max(lags)))[1:initialArimaNumber];
                    # rep(yInSample[1:initialArimaNumber],each=componentsNumberARIMA);
                }
                else{
                    # Fill in the matrix with 0 / 1, just in case if the state will not be updated anymore
                    matVt[componentsNumberETS+1:componentsNumberARIMA, 1:initialArimaNumber] <-
                        switch(Etype, "A"=0, "M"=1);
                    # Insert the provided initials
                    matVt[componentsNumberETS+componentsNumberARIMA, 1:initialArimaNumber] <-
                        initialArima[1:initialArimaNumber];
                }
            }

            # Fill in the initials for xreg
            if(xregModel){
                if(Etype=="A" || initialXregProvided || is.null(xregModelInitials[[2]])){
                    # This step is needed to address the potential issue with dropped level
                    # of factor variables or variables due to multicollinearity
                    matVt[componentsNumberETS+componentsNumberARIMA+1:xregNumber,
                          1:lagsModelMax] <- 0;
                    matVt[names(xregModelInitials[[1]]$initialXreg),
                          1:lagsModelMax] <- xregModelInitials[[1]]$initialXreg;
                }
                else{
                    # This step is needed to address the potential issue with dropped level
                    # of factor variables or variables due to multicollinearity
                    matVt[componentsNumberETS+componentsNumberARIMA+1:xregNumber,
                          1:lagsModelMax] <- 0;
                    matVt[names(xregModelInitials[[2]]$initialXreg),
                          1:lagsModelMax] <- xregModelInitials[[2]]$initialXreg;
                }
            }

            # Add constant if needed
            if(constantRequired){
                if(constantEstimate){
                    # Add the mean of data
                    if(sum(iOrders)==0 && !etsModel){
                        matVt[componentsNumberETS+componentsNumberARIMA+xregNumber+1,] <- mean(yInSample[otLogical]);
                    }
                    # Add first differences
                    else{
                        matVt[componentsNumberETS+componentsNumberARIMA+xregNumber+1,] <-
                            switch(Etype,
                                   "A"=mean(diff(yInSample[otLogical])),
                                   "M"=exp(mean(diff(log(yInSample[otLogical])))));
                    }
                }
                else{
                    matVt[componentsNumberETS+componentsNumberARIMA+xregNumber+1,] <- constantValue;
                }
                # If ETS model is used, change the initial level
                if(etsModel && initialLevelEstimate){
                    if(Etype=="A"){
                        matVt[1,1:lagsModelMax] <- matVt[1,1:lagsModelMax] -
                            matVt[componentsNumberETS+componentsNumberARIMA+xregNumber+1,1];
                    }
                    else{
                        matVt[1,1:lagsModelMax] <- matVt[1,1:lagsModelMax] /
                            matVt[componentsNumberETS+componentsNumberARIMA+xregNumber+1,1];
                    }
                }
                # If ARIMA is done, debias states
                if(arimaModel && initialArimaEstimate){
                    if(Etype=="A"){
                        matVt[componentsNumberETS+nonZeroARI[,2],1:initialArimaNumber] <-
                            matVt[componentsNumberETS+nonZeroARI[,2],1:initialArimaNumber] -
                            matVt[componentsNumberETS+componentsNumberARIMA+xregNumber+1,1];
                    }
                    else{
                        matVt[componentsNumberETS+nonZeroARI[,2],1:initialArimaNumber] <-
                            matVt[componentsNumberETS+nonZeroARI[,2],1:initialArimaNumber] /
                            matVt[componentsNumberETS+componentsNumberARIMA+xregNumber+1,1];
                    }
                }
            }
        }
        else{
            matVt[,1:lagsModelMax] <- profilesRecentTable;
        }

        return(list(matVt=matVt, matWt=matWt, matF=matF, vecG=vecG, arimaPolynomials=arimaPolynomials));
    }

    #### The function fills in the existing matrices with values of A ####
    # This is needed in order to do the estimation and the fit
    filler <- function(B,
                       etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal,
                       componentsNumberETS, componentsNumberETSNonSeasonal,
                       componentsNumberETSSeasonal, componentsNumberARIMA,
                       lags, lagsModel, lagsModelMax,
                       # The main matrices
                       matVt, matWt, matF, vecG,
                       # Persistence and phi
                       persistenceEstimate, persistenceLevelEstimate, persistenceTrendEstimate,
                       persistenceSeasonalEstimate, persistenceXregEstimate,
                       phiEstimate,
                       # Initials
                       initialType, initialEstimate,
                       initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                       initialArimaEstimate, initialXregEstimate,
                       # ARIMA
                       arimaModel, arEstimate, maEstimate, arOrders, iOrders, maOrders,
                       arRequired, maRequired, armaParameters,
                       nonZeroARI, nonZeroMA, arimaPolynomials,
                       # Explanatory variables
                       xregModel, xregNumber,
                       xregParametersMissing, xregParametersIncluded,
                       xregParametersEstimated, xregParametersPersistence,
                       # Constant
                       constantEstimate,
                       adamCpp){

        j <- 0;
        # Fill in persistence
        if(persistenceEstimate){
            # Persistence of ETS
            if(etsModel){
                i <- 1;
                # alpha
                if(persistenceLevelEstimate){
                    j[] <- j+1;
                    vecG[i] <- B[j];
                }
                # beta
                if(modelIsTrendy){
                    i[] <- 2;
                    if(persistenceTrendEstimate){
                        j[] <- j+1;
                        vecG[i] <- B[j];
                    }
                }
                # gamma1, gamma2, ...
                if(modelIsSeasonal){
                    if(any(persistenceSeasonalEstimate)){
                        vecG[i+which(persistenceSeasonalEstimate)] <- B[j+c(1:sum(persistenceSeasonalEstimate))];
                        j[] <- j+sum(persistenceSeasonalEstimate);
                    }
                    i[] <- componentsNumberETS;
                }
            }

            # Persistence of xreg
            if(xregModel && persistenceXregEstimate){
                xregPersistenceNumber <- max(xregParametersPersistence);
                vecG[j+componentsNumberARIMA+1:length(xregParametersPersistence)] <-
                    B[j+1:xregPersistenceNumber][xregParametersPersistence];
                j[] <- j+xregPersistenceNumber;
            }
        }

        # Damping parameter
        if(etsModel && phiEstimate){
            j[] <- j+1;
            matWt[,2] <- B[j];
            matF[1:2,2] <- B[j];
        }

        # ARMA parameters. This goes before xreg in persistence
        if(arimaModel){
            # Call the function returning ARI and MA polynomials
            arimaPolynomials <- lapply(adamCpp$polynomialise(B[j+1:sum(c(arOrders*arEstimate,maOrders*maEstimate))],
                                                             arOrders, iOrders, maOrders,
                                                             arEstimate, maEstimate, armaParameters, lags), as.vector);

            # Fill in the transition matrix
            if(nrow(nonZeroARI)>0){
                matF[componentsNumberETS+nonZeroARI[,2],componentsNumberETS+1:(componentsNumberARIMA+constantRequired)] <-
                    -arimaPolynomials$ariPolynomial[nonZeroARI[,1]];
            }
            # Fill in the persistence vector
            if(nrow(nonZeroARI)>0){
                vecG[componentsNumberETS+nonZeroARI[,2]] <- -arimaPolynomials$ariPolynomial[nonZeroARI[,1]];
            }
            if(nrow(nonZeroMA)>0){
                vecG[componentsNumberETS+nonZeroMA[,2]] <- vecG[componentsNumberETS+nonZeroMA[,2]] +
                    arimaPolynomials$maPolynomial[nonZeroMA[,1]];
            }
            j[] <- j+sum(c(arOrders*arEstimate,maOrders*maEstimate));
        }

        # Initials of ETS if something needs to be estimated
        if(etsModel && all(initialType!=c("complete","backcasting")) && initialEstimate){
            i <- 1;
            if(initialLevelEstimate){
                j[] <- j+1;
                matVt[i,1:lagsModelMax] <- B[j];
            }
            i[] <- i+1;
            if(modelIsTrendy && initialTrendEstimate){
                j[] <- j+1;
                matVt[i,1:lagsModelMax] <- B[j];
                i[] <- i+1;
            }
            if(modelIsSeasonal && any(initialSeasonalEstimate)){
                for(k in 1:componentsNumberETSSeasonal){
                    if(initialSeasonalEstimate[k]){
                        matVt[componentsNumberETSNonSeasonal+k, 2:lagsModel[componentsNumberETSNonSeasonal+k]-1] <-
                            B[j+2:(lagsModel[componentsNumberETSNonSeasonal+k])-1];
                        matVt[componentsNumberETSNonSeasonal+k, lagsModel[componentsNumberETSNonSeasonal+k]] <-
                            switch(Stype,
                                   "A"=-sum(B[j+2:(lagsModel[componentsNumberETSNonSeasonal+k])-1]),
                                   "M"=1/prod(B[j+2:(lagsModel[componentsNumberETSNonSeasonal+k])-1]));
                        j[] <- j+lagsModel[componentsNumberETSNonSeasonal+k]-1;
                    }
                }
            }
        }

        # Initials of ARIMA
        if(arimaModel){
            if(all(initialType!=c("complete","backcasting")) && initialArimaEstimate){
                # matVt[componentsNumberETS+componentsNumberARIMA, 1:initialArimaNumber] <- B[j+1:initialArimaNumber];
                # for(i in (componentsNumberARIMA-1):1){
                #     indeces <-
                #         (1+lagsModelAll[componentsNumberETS+i+1] -
                #              lagsModelAll[componentsNumberETS+i]):lagsModelAll[componentsNumberETS+i+1];
                #     matVt[componentsNumberETS+i,
                #           1:lagsModelAll[componentsNumberETS+i]] <-
                #         (matVt[componentsNumberETS+componentsNumberARIMA, indeces] -
                #              # We need a sum of states here...
                #              matVt[componentsNumberETS+i+1, 1:lagsModelAll[componentsNumberETS+i]]);
                # }

                matVt[componentsNumberETS+nonZeroARI[,2], 1:initialArimaNumber] <-
                    switch(Etype,
                           "A"=arimaPolynomials$ariPolynomial[nonZeroARI[,1]] %*% t(B[j+1:initialArimaNumber]),
                           "M"=exp(arimaPolynomials$ariPolynomial[nonZeroARI[,1]] %*% t(log(B[j+1:initialArimaNumber]))));

                # switch(Etype,
                #        "A"=arimaPolynomials$ariPolynomial[nonZeroARI[,1]] %*% t(B[j+1:initialArimaNumber]) /
                #            tail(arimaPolynomials$ariPolynomial,1),
                #        "M"=exp(arimaPolynomials$ariPolynomial[nonZeroARI[,1]] %*% t(log(B[j+1:initialArimaNumber])) /
                #                    tail(arimaPolynomials$ariPolynomial,1)));
                j[] <- j+initialArimaNumber;
            }
            # This is needed in order to propagate initials of ARIMA to all components
            else if(any(c(arEstimate,maEstimate))){
                # if(nrow(nonZeroARI)>0 && nrow(nonZeroARI)>=nrow(nonZeroMA)){
                # if(nrow(nonZeroARI)>0){
                matVt[componentsNumberETS+nonZeroARI[,2], 1:initialArimaNumber] <-
                    switch(Etype,
                           "A"= arimaPolynomials$ariPolynomial[nonZeroARI[,1]] %*%
                               t(matVt[componentsNumberETS+componentsNumberARIMA, 1:initialArimaNumber]),
                           "M"=exp(arimaPolynomials$ariPolynomial[nonZeroARI[,1]] %*%
                                       t(log(matVt[componentsNumberETS+componentsNumberARIMA, 1:initialArimaNumber]))));

                # switch(Etype,
                #        "A"= arimaPolynomials$ariPolynomial[nonZeroARI[,1]] %*%
                #            t(matVt[componentsNumberETS+componentsNumberARIMA, 1:initialArimaNumber]) /
                #            tail(arimaPolynomials$ariPolynomial,1),
                #        "M"=exp(arimaPolynomials$ariPolynomial[nonZeroARI[,1]] %*%
                #                    t(log(matVt[componentsNumberETS+componentsNumberARIMA, 1:initialArimaNumber])) /
                #                    tail(arimaPolynomials$ariPolynomial,1)));

                # }
                # else{
                #     matVt[componentsNumberETS+nonZeroMA[,2],
                #           1:initialArimaNumber] <-
                #         switch(Etype,
                #                "A"=arimaPolynomials$maPolynomial[nonZeroMA[,1]] %*%
                #                    t(matVt[componentsNumberETS+componentsNumberARIMA, 1:initialArimaNumber]) /
                #                    tail(arimaPolynomials$maPolynomial,1),
                #                "M"=exp(arimaPolynomials$maPolynomial[nonZeroMA[,1]] %*%
                #                            t(log(matVt[componentsNumberETS+componentsNumberARIMA, 1:initialArimaNumber])) /
                #                            tail(arimaPolynomials$maPolynomial,1)));
                # }
            }
        }

        # Initials of the xreg
        if(xregModel && (initialType!="complete") && initialEstimate && initialXregEstimate){
            xregNumberToEstimate <- sum(xregParametersEstimated);
            matVt[componentsNumberETS+componentsNumberARIMA+which(xregParametersEstimated==1),
                  1:lagsModelMax] <- B[j+1:xregNumberToEstimate];
            j[] <- j+xregNumberToEstimate;
            # Normalise initials
            # for(i in which(xregParametersMissing!=0)){
            #     matVt[componentsNumberETS+componentsNumberARIMA+i,
            #           1:lagsModelMax] <- -sum(matVt[componentsNumberETS+componentsNumberARIMA+
            #                                             which(xregParametersIncluded==xregParametersMissing[i]),
            #                                         1:lagsModelMax]);
            # }
        }

        # Constant
        if(constantEstimate){
            matVt[componentsNumberETS+componentsNumberARIMA+xregNumber+1,] <- B[j+1];
        }

        return(list(matVt=matVt, matWt=matWt, matF=matF, vecG=vecG, arimaPolynomials=arimaPolynomials));
    }

    #### The function initialises the vector B for ETS ####
    initialiser <- function(etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal,
                            componentsNumberETSNonSeasonal, componentsNumberETSSeasonal, componentsNumberETS,
                            lags, lagsModel, lagsModelSeasonal, lagsModelARIMA, lagsModelMax,
                            matVt,
                            # persistence values
                            persistenceEstimate, persistenceLevelEstimate, persistenceTrendEstimate,
                            persistenceSeasonalEstimate, persistenceXregEstimate,
                            # initials
                            phiEstimate, initialType, initialEstimate,
                            initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                            initialArimaEstimate, initialXregEstimate,
                            # ARIMA elements
                            arimaModel, arRequired, maRequired, arEstimate, maEstimate, arOrders, maOrders,
                            componentsNumberARIMA, componentsNamesARIMA, initialArimaNumber,
                            # Explanatory variables
                            xregModel, xregNumber,
                            xregParametersEstimated, xregParametersPersistence,
                            # Constant and other stuff
                            constantEstimate, constantName, otherParameterEstimate,
                            adamCpp){
        # The vector of logicals for persistence elements
        persistenceEstimateVector <- c(persistenceLevelEstimate,modelIsTrendy&persistenceTrendEstimate,
                                       modelIsSeasonal&persistenceSeasonalEstimate);

        # The order:
        # Persistence of states and for xreg, phi, AR and MA parameters, initials, initialsARIMA, initials for xreg
        B <- Bl <- Bu <- vector("numeric",
                                # Values of the persistence vector + phi
                                etsModel*(persistenceLevelEstimate + modelIsTrendy*persistenceTrendEstimate +
                                              modelIsSeasonal*sum(persistenceSeasonalEstimate) + phiEstimate) +
                                    xregModel*persistenceXregEstimate*max(xregParametersPersistence) +
                                    # AR and MA values
                                    arimaModel*(arEstimate*sum(arOrders)+maEstimate*sum(maOrders)) +
                                    # initials of ETS
                                    etsModel*all(initialType!=c("complete","backcasting"))*
                                    (initialLevelEstimate +
                                         (modelIsTrendy*initialTrendEstimate) +
                                         (modelIsSeasonal*sum(initialSeasonalEstimate*(lagsModelSeasonal-1)))) +
                                    # initials of ARIMA
                                    all(initialType!=c("complete","backcasting"))*arimaModel*initialArimaNumber*initialArimaEstimate +
                                    # initials of xreg
                                    (initialType!="complete")*xregModel*initialXregEstimate*sum(xregParametersEstimated) +
                                    constantEstimate + otherParameterEstimate);

        j <- 0;
        if(etsModel){
            # Fill in persistence
            if(persistenceEstimate && any(persistenceEstimateVector)){
                if(ets=="conventional" && any(c(Etype,Ttype,Stype)=="M")){
                    # A special type of model which is not safe: AAM, MAA, MAM
                    if((Etype=="A" && Ttype=="A" && Stype=="M") || (Etype=="A" && Ttype=="M" && Stype=="A") ||
                       (any(initialType==c("complete","backcasting")) &&
                        ((Etype=="M" && Ttype=="A" && Stype=="A") || (Etype=="M" && Ttype=="A" && Stype=="M")))){
                        B[1:sum(persistenceEstimateVector)] <-
                            c(0.01,0.005,rep(0.001,componentsNumberETSSeasonal))[which(persistenceEstimateVector)];
                    }
                    # MMA is the worst. Set everything to zero and see if anything can be done...
                    else if((Etype=="M" && Ttype=="M" && Stype=="A")){
                        B[1:sum(persistenceEstimateVector)] <-
                            c(0.01,0.005,rep(0.01,componentsNumberETSSeasonal))[which(persistenceEstimateVector)];
                    }
                    else if(Etype=="M" && Ttype=="A"){
                        if(any(initialType==c("complete","backcasting"))){
                            B[1:sum(persistenceEstimateVector)] <-
                                c(0.1,0.05,rep(0.3,componentsNumberETSSeasonal))[which(persistenceEstimateVector)];
                        }
                        else{
                            B[1:sum(persistenceEstimateVector)] <-
                                c(0.2,0.01,rep(0.3,componentsNumberETSSeasonal))[which(persistenceEstimateVector)];
                        }
                    }
                    else if(Etype=="M" && Ttype=="M"){
                        B[1:sum(persistenceEstimateVector)] <-
                            c(0.1,0.05,rep(0.3,componentsNumberETSSeasonal))[which(persistenceEstimateVector)];
                    }
                    else{
                        B[1:sum(persistenceEstimateVector)] <-
                            c(0.1,0.05,rep(0.3,componentsNumberETSSeasonal))[which(persistenceEstimateVector)];
                    }
                }
                else{
                    B[1:sum(persistenceEstimateVector)] <-
                        c(0.1,0.05,rep(0.3,componentsNumberETSSeasonal))[which(persistenceEstimateVector)];
                }
                if(bounds=="usual"){
                    Bl[1:sum(persistenceEstimateVector)] <- rep(0, sum(persistenceEstimateVector));
                    Bu[1:sum(persistenceEstimateVector)] <- rep(1, sum(persistenceEstimateVector));
                }
                else{
                    Bl[1:sum(persistenceEstimateVector)] <- rep(-5, sum(persistenceEstimateVector));
                    Bu[1:sum(persistenceEstimateVector)] <- rep(5, sum(persistenceEstimateVector));
                }
                # Names for B
                if(persistenceLevelEstimate){
                    j[] <- j+1
                    names(B)[j] <- "alpha";
                }
                if(modelIsTrendy && persistenceTrendEstimate){
                    j[] <- j+1
                    names(B)[j] <- "beta";
                }
                if(modelIsSeasonal && any(persistenceSeasonalEstimate)){
                    if(componentsNumberETSSeasonal>1){
                        names(B)[j+c(1:sum(persistenceSeasonalEstimate))] <-
                            paste0("gamma",c(1:componentsNumberETSSeasonal));
                    }
                    else{
                        names(B)[j+1] <- "gamma";
                    }
                    j[] <- j+sum(persistenceSeasonalEstimate);
                }
            }
        }

        # Persistence if xreg is provided
        if(xregModel && persistenceXregEstimate){
            xregPersistenceNumber <- max(xregParametersPersistence);
            B[j+1:xregPersistenceNumber] <- rep(switch(Etype,"A"=0.01,"M"=0),xregPersistenceNumber);
            Bl[j+1:xregPersistenceNumber] <- rep(-5, xregPersistenceNumber);
            Bu[j+1:xregPersistenceNumber] <- rep(5, xregPersistenceNumber);
            names(B)[j+1:xregPersistenceNumber] <- paste0("delta",c(1:xregPersistenceNumber));
            j[] <- j+xregPersistenceNumber;
        }

        # Damping parameter
        if(etsModel && phiEstimate){
            j[] <- j+1;
            B[j] <- 0.95;
            names(B)[j] <- "phi";
            Bl[j] <- 0;
            Bu[j] <- 1;
        }

        # ARIMA parameters (AR / MA)
        if(arimaModel){
            # This index is needed to get the correct polynomials
            k <- j
            # These are filled in lags-wise
            if(any(c(arEstimate,maEstimate))){
                acfValues <- rep(-0.1, maOrders %*% lags);
                pacfValues <- rep(0.1, arOrders %*% lags);
                # If this is ETS + ARIMA model or no differences model, then don't bother with initials
                # The latter does not make sense because of non-stationarity in ACF / PACF
                # Otherwise use ACF / PACF values as starting parameters for ARIMA
                if(!(etsModel || all(iOrders==0))){
                    yDifferenced <- yInSample;
                    # If the model has differences, take them
                    if(any(iOrders>0)){
                        for(i in 1:length(iOrders)){
                            if(iOrders[i]>0){
                                yDifferenced <- diff(yDifferenced,lag=lags[i],differences=iOrders[i]);
                            }
                        }
                    }
                    # Do ACF/PACF initialisation only for non-seasonal models
                    if(all(lags<=1)){
                        if(maRequired && maEstimate){
                            # If the sample is smaller than lags, it will be substituted by default values
                            acfValues[1:min(maOrders %*% lags, length(yDifferenced)-1)] <-
                                acf(yDifferenced,lag.max=max(1,maOrders %*% lags),plot=FALSE)$acf[-1];
                        }
                        if(arRequired && arEstimate){
                            # If the sample is smaller than lags, it will be substituted by default values
                            pacfValues[1:min(arOrders %*% lags, length(yDifferenced)-1)] <-
                                pacf(yDifferenced,lag.max=max(1,arOrders %*% lags),plot=FALSE)$acf;
                        }
                    }
                }
                for(i in 1:length(lags)){
                    if(arRequired && arEstimate && arOrders[i]>0){
                        if(all(!is.nan(pacfValues[c(1:arOrders[i])*lags[i]]))){
                            B[j+c(1:arOrders[i])] <- pacfValues[c(1:arOrders[i])*lags[i]];
                        }
                        else{
                            B[j+c(1:arOrders[i])] <- 0.1;
                        }
                        if(sum(B[j+c(1:arOrders[i])])>1){
                            B[j+c(1:arOrders[i])] <- B[j+c(1:arOrders[i])] / sum(B[j+c(1:arOrders[i])]) - 0.01;
                        }
                        # B[j+c(1:arOrders[i])] <- rep(0.1,arOrders[i]);
                        Bl[j+c(1:arOrders[i])] <- -5;
                        Bu[j+c(1:arOrders[i])] <- 5;
                        names(B)[j+1:arOrders[i]] <- paste0("phi",1:arOrders[i],"[",lags[i],"]");
                        j[] <- j + arOrders[i];
                    }
                    if(maRequired && maEstimate && maOrders[i]>0){
                        if(all(!is.nan(acfValues[c(1:maOrders[i])*lags[i]]))){
                            B[j+c(1:maOrders[i])] <- acfValues[c(1:maOrders[i])*lags[i]];
                        }
                        else{
                            B[j+c(1:maOrders[i])] <- 0.1;
                        }
                        if(sum(B[j+c(1:maOrders[i])])>1){
                            B[j+c(1:maOrders[i])] <- B[j+c(1:maOrders[i])] / sum(B[j+c(1:maOrders[i])]) - 0.01;
                        }
                        # B[j+c(1:maOrders[i])] <- rep(-0.1,maOrders[i]);
                        Bl[j+c(1:maOrders[i])] <- -5;
                        Bu[j+c(1:maOrders[i])] <- 5;
                        names(B)[j+1:maOrders[i]] <- paste0("theta",1:maOrders[i],"[",lags[i],"]");
                        j[] <- j + maOrders[i];
                    }
                }
            }

            arimaPolynomials <- lapply(adamCpp$polynomialise(B[k+1:sum(c(arOrders*arEstimate,maOrders*maEstimate))],
                                                             arOrders, iOrders, maOrders,
                                                             arEstimate, maEstimate, armaParameters, lags), as.vector);
        }

        # Initials
        if(etsModel && all(initialType!=c("complete","backcasting")) && initialEstimate){
            if(initialLevelEstimate){
                j[] <- j+1;
                B[j] <- matVt[1,1];
                names(B)[j] <- "level";
                if(Etype=="A"){
                    Bl[j] <- -Inf;
                    Bu[j] <- Inf;
                }
                else{
                    Bl[j] <- 0;
                    Bu[j] <- Inf;
                }
            }
            if(modelIsTrendy && initialTrendEstimate){
                j[] <- j+1;
                B[j] <- matVt[2,1];
                names(B)[j] <- "trend";
                if(Ttype=="A"){
                    Bl[j] <- -Inf;
                    Bu[j] <- Inf;
                }
                else{
                    Bl[j] <- 0;
                    # 2 is already too much for the multiplicative model
                    Bu[j] <- 2;
                }
            }
            if(modelIsSeasonal && any(initialSeasonalEstimate)){
                if(componentsNumberETSSeasonal>1){
                    for(k in 1:componentsNumberETSSeasonal){
                        if(initialSeasonalEstimate[k]){
                            # -1 is needed in order to remove the redundant seasonal element (normalisation)
                            B[j+2:lagsModel[componentsNumberETSNonSeasonal+k]-1] <-
                                matVt[componentsNumberETSNonSeasonal+k, 2:lagsModel[componentsNumberETSNonSeasonal+k]-1];
                            names(B)[j+2:(lagsModel[componentsNumberETSNonSeasonal+k])-1] <-
                                paste0("seasonal",k,"_",2:lagsModel[componentsNumberETSNonSeasonal+k]-1);
                            if(Stype=="A"){
                                Bl[j+2:lagsModel[componentsNumberETSNonSeasonal+k]-1] <- -Inf;
                                Bu[j+2:lagsModel[componentsNumberETSNonSeasonal+k]-1] <- Inf;
                            }
                            else{
                                Bl[j+2:lagsModel[componentsNumberETSNonSeasonal+k]-1] <- 0;
                                Bu[j+2:lagsModel[componentsNumberETSNonSeasonal+k]-1] <- Inf;
                            }
                            j[] <- j+(lagsModelSeasonal[k]-1);
                        }
                    }
                }
                else{
                    # -1 is needed in order to remove the redundant seasonal element (normalisation)
                    B[j+2:(lagsModel[componentsNumberETS])-1] <- matVt[componentsNumberETS,2:lagsModel[componentsNumberETS]-1];
                    names(B)[j+2:(lagsModel[componentsNumberETS])-1] <- paste0("seasonal_",2:lagsModel[componentsNumberETS]-1);
                    if(Stype=="A"){
                        Bl[j+2:(lagsModel[componentsNumberETS])-1] <- -Inf;
                        Bu[j+2:(lagsModel[componentsNumberETS])-1] <- Inf;
                    }
                    else{
                        Bl[j+2:(lagsModel[componentsNumberETS])-1] <- 0;
                        Bu[j+2:(lagsModel[componentsNumberETS])-1] <- Inf;
                    }
                    j[] <- j+(lagsModel[componentsNumberETS]-1);
                }
            }
        }

        # ARIMA initials
        if(arimaModel && all(initialType!=c("complete","backcasting")) && initialArimaEstimate){
            B[j+1:initialArimaNumber] <- head(matVt[componentsNumberETS+componentsNumberARIMA,1:lagsModelMax],initialArimaNumber);
            names(B)[j+1:initialArimaNumber] <- paste0("ARIMAState",1:initialArimaNumber);

            # Fix initial state if the polynomial is not zero
            if(tail(arimaPolynomials$ariPolynomial,1)!=0){
                B[j+1:initialArimaNumber] <- B[j+1:initialArimaNumber] / tail(arimaPolynomials$ariPolynomial,1);
            }

            if(Etype=="A"){
                Bl[j+1:initialArimaNumber] <- -Inf;
                Bu[j+1:initialArimaNumber] <- Inf;
            }
            else{
                # Make sure that ARIMA states are positive to avoid errors
                B[j+1:initialArimaNumber] <- abs(B[j+1:initialArimaNumber]);
                Bl[j+1:initialArimaNumber] <- 0;
                Bu[j+1:initialArimaNumber] <- Inf;
            }
            j[] <- j+initialArimaNumber;
        }

        # Initials of the xreg
        if(initialType!="complete" && initialXregEstimate){
            xregNumberToEstimate <- sum(xregParametersEstimated);
            B[j+1:xregNumberToEstimate] <- matVt[componentsNumberETS+componentsNumberARIMA+
                                                     which(xregParametersEstimated==1),1];
            names(B)[j+1:xregNumberToEstimate] <- rownames(matVt)[componentsNumberETS+componentsNumberARIMA+
                                                                      which(xregParametersEstimated==1)];
            if(Etype=="A"){
                Bl[j+1:xregNumberToEstimate] <- -Inf;
                Bu[j+1:xregNumberToEstimate] <- Inf;
            }
            else{
                Bl[j+1:xregNumberToEstimate] <- -Inf;
                Bu[j+1:xregNumberToEstimate] <- Inf;
            }
            j[] <- j+xregNumberToEstimate;
        }

        if(constantEstimate){
            j[] <- j+1;
            B[j] <- matVt[componentsNumberETS+componentsNumberARIMA+xregNumber+1,1];
            names(B)[j] <- constantName;
            if(etsModel || sum(iOrders)!=0){
                if(Etype=="A"){
                    Bu[j] <- quantile(diff(yInSample[otLogical]),0.6);
                    Bl[j] <- -Bu[j];
                }
                else{
                    Bu[j] <- exp(quantile(diff(log(yInSample[otLogical])),0.6));
                    Bl[j] <- exp(quantile(diff(log(yInSample[otLogical])),0.4));
                }

                # Failsafe for weird cases, when upper bound is the same or lower than the lower one
                if(Bu[j]<=Bl[j]){
                    Bu[j] <- Inf;
                    Bl[j] <- switch(Etype,"A"=-Inf,"M"=0);
                }

                # Failsafe for cases, when the B is outside of bounds
                if(B[j]<=Bl[j]){
                    Bl[j] <- switch(Etype,"A"=-Inf,"M"=0);
                }
                if(B[j]>=Bu[j]){
                    Bu[j] <- Inf;
                }
            }
            else{
                # if(Etype=="A"){
                # B[j]*1.01 is needed to make sure that the bounds cover the initial value
                Bu[j] <- max(abs(yInSample[otLogical]),abs(B[j])*1.01);
                Bl[j] <- -Bu[j];
                # }
                # else{
                #     Bu[j] <- 1.5;
                #     Bl[j] <- 0.1;
                # }
                # If this is just a constant
            }
        }

        # Add lambda if it is needed
        if(otherParameterEstimate){
            j[] <- j+1;
            B[j] <- other;
            names(B)[j] <- "other";
            Bl[j] <- 1e-10;
            Bu[j] <- Inf;
        }

        return(list(B=B,Bl=Bl,Bu=Bu));
    }

    ##### Function returns scale parameter for the provided parameters #####
    scaler <- function(distribution, Etype, errors, yFitted, obsInSample, other){
        # as.complex() is needed in order to make the optimiser work in exotic cases
        return(switch(distribution,
                      "dnorm"=sqrt(sum(errors^2)/obsInSample),
                      "dlaplace"=sum(abs(errors))/obsInSample,
                      "ds"=sum(sqrt(abs(errors))) / (obsInSample*2),
                      "dgnorm"=(other*sum(abs(errors)^other)/obsInSample)^{1/other},
                      # "dlogis"=sqrt(sum(errors^2)/obsInSample * 3 / pi^2),
                      # "dt"=sqrt(sum(errors^2)/obsInSample),
                      "dalaplace"=sum(errors*(other-(errors<=0)*1))/obsInSample,
                      # This condition guarantees that E(1+e_t)=1
                      # abs is needed for cases, when we get imaginary values - a failsafe
                      "dlnorm"=sqrt(2*abs(switch(Etype,
                                                 "A"=1-sqrt(abs(1-sum(log(abs(1+errors/yFitted))^2)/obsInSample)),
                                                 "M"=1-sqrt(abs(1-sum(log(1+errors)^2)/obsInSample))))),
                      # "A"=Re(sqrt(sum(log(as.complex(1+errors/yFitted))^2)/obsInSample)),
                      # "M"=sqrt(sum(log(1+errors)^2)/obsInSample)),
                      "dllaplace"=switch(Etype,
                                         "A"=Re(sum(abs(log(as.complex(1+errors/yFitted))))/obsInSample),
                                         "M"=sum(abs(log(1+errors))/obsInSample)),
                      "dls"=switch(Etype,
                                   "A"=Re(sum(sqrt(abs(log(as.complex(1+errors/yFitted))))/obsInSample)),
                                   "M"=sum(sqrt(abs(log(1+errors)))/obsInSample)),
                      "dlgnorm"=switch(Etype,
                                       "A"=Re((other*sum(abs(log(as.complex(1+errors/yFitted)))^other)/obsInSample)^{1/other}),
                                       "M"=(other*sum(abs(log(as.complex(1+errors)))^other)/obsInSample)^{1/other}),
                      "dinvgauss"=switch(Etype,
                                         "A"=sum((errors/yFitted)^2/(1+errors/yFitted))/obsInSample,
                                         "M"=sum((errors)^2/(1+errors))/obsInSample),
                      "dgamma"=switch(Etype,
                                      "A"=sum((errors/yFitted)^2)/obsInSample,
                                      "M"=sum(errors^2)/obsInSample)
                      # "M"=mean((errors)^2/(1+errors))),
        ));
    }

    ##### Cost Function for ETS #####
    CF <- function(B,
                   etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal, yInSample,
                   ot, otLogical, occurrenceModel, obsInSample,
                   componentsNumberETS, componentsNumberETSSeasonal, componentsNumberETSNonSeasonal,
                   componentsNumberARIMA,
                   lags, lagsModel, lagsModelAll, lagsModelMax,
                   indexLookupTable, profilesRecentTable,
                   # The main matrices
                   matVt, matWt, matF, vecG,
                   # Persistence and phi
                   persistenceEstimate, persistenceLevelEstimate, persistenceTrendEstimate,
                   persistenceSeasonalEstimate, persistenceXregEstimate, phiEstimate,
                   # Initials
                   initialType, initialEstimate,
                   initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                   initialArimaEstimate, initialXregEstimate,
                   # ARIMA
                   arimaModel, nonZeroARI, nonZeroMA, arEstimate, maEstimate, arimaPolynomials,
                   arOrders, iOrders, maOrders, arRequired, maRequired, armaParameters,
                   # xreg
                   xregModel, xregNumber,
                   xregParametersMissing, xregParametersIncluded,
                   xregParametersEstimated, xregParametersPersistence,
                   # Constant
                   constantRequired, constantEstimate,
                   # Other stuff
                   bounds, loss, lossFunction, distribution, horizon, multisteps,
                   denominator=NULL, yDenominator=NULL,
                   other, otherParameterEstimate, lambda,
                   arPolynomialMatrix, maPolynomialMatrix,
                   adamCpp){

        # Fill in the matrices
        adamElements <- filler(B,
                               etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal,
                               componentsNumberETS, componentsNumberETSNonSeasonal,
                               componentsNumberETSSeasonal, componentsNumberARIMA,
                               lags, lagsModel, lagsModelMax,
                               matVt, matWt, matF, vecG,
                               persistenceEstimate, persistenceLevelEstimate, persistenceTrendEstimate,
                               persistenceSeasonalEstimate, persistenceXregEstimate,
                               phiEstimate,
                               initialType, initialEstimate,
                               initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                               initialArimaEstimate, initialXregEstimate,
                               arimaModel, arEstimate, maEstimate, arOrders, iOrders, maOrders,
                               arRequired, maRequired, armaParameters,
                               nonZeroARI, nonZeroMA, arimaPolynomials,
                               xregModel, xregNumber,
                               xregParametersMissing, xregParametersIncluded,
                               xregParametersEstimated, xregParametersPersistence, constantEstimate,
                               adamCpp);

        # If we estimate parameters of distribution, take it from the B vector
        if(otherParameterEstimate){
            # Take absolute value, just to be on safe side. We don't need negatives anyway.
            other[] <- abs(B[length(B)]);

            # Beta in GN is restricted by 0.25 if it is optimised.
            if(any(distribution==c("dgnorm","dlgnorm")) && other<0.25){
                return(1E+10/other);
            }
        }

        # Check the bounds, classical restrictions
        #### The usual bounds ####
        if(bounds=="usual"){
            # Stationarity and invertibility conditions for ARIMA
            if(arimaModel && any(c(arEstimate,maEstimate))){
                # Calculate the polynomial roots for AR
                if(arEstimate &&
                   (all(-adamElements$arimaPolynomials$arPolynomial[-1]>0) &
                    sum(-(adamElements$arimaPolynomials$arPolynomial[-1]))>=1)){
                    arPolynomialMatrix[,1] <- -adamElements$arimaPolynomials$arPolynomial[-1];
                    arPolyroots <- abs(eigen(arPolynomialMatrix, symmetric=FALSE, only.values=TRUE)$values);
                    if(any(arPolyroots>1)){
                        return(1E+100*max(arPolyroots));
                    }
                }
                # Calculate the polynomial roots of MA
                if(maEstimate && sum(adamElements$arimaPolynomials$maPolynomial[-1])>=1){
                    maPolynomialMatrix[,1] <- adamElements$arimaPolynomials$maPolynomial[-1];
                    maPolyroots <- abs(eigen(maPolynomialMatrix, symmetric=FALSE, only.values=TRUE)$values);
                    if(any(maPolyroots>1)){
                        return(1E+100*max(abs(maPolyroots)));
                    }
                }
            }

            # Smoothing parameters & phi restrictions in case of ETS
            if(etsModel){
                if(any(adamElements$vecG[1:componentsNumberETS]>1) || any(adamElements$vecG[1:componentsNumberETS]<0)){
                    return(1E+300);
                }
                if(modelIsTrendy){
                    if((adamElements$vecG[2]>adamElements$vecG[1])){
                        return(1E+300);
                    }
                    if(modelIsSeasonal && any(adamElements$vecG[componentsNumberETSNonSeasonal+c(1:componentsNumberETSSeasonal)]>
                                              (1-adamElements$vecG[1]))){
                        return(1E+300);
                    }
                }
                else{
                    if(modelIsSeasonal && any(adamElements$vecG[componentsNumberETSNonSeasonal+c(1:componentsNumberETSSeasonal)]>
                                              (1-adamElements$vecG[1]))){
                        return(1E+300);
                    }
                }

                # This is the restriction on the damping parameter
                if(phiEstimate && (adamElements$matF[2,2]>1 || adamElements$matF[2,2]<0)){
                    return(1E+300);
                }
            }

            # Smoothing parameters for the explanatory variables (0, 1) region
            if(xregModel && regressors=="adapt"){
                if(any(adamElements$vecG[componentsNumberETS+componentsNumberARIMA+1:xregNumber]>1) ||
                   any(adamElements$vecG[componentsNumberETS+componentsNumberARIMA+1:xregNumber]<0)){
                    return(1E+100*max(abs(adamElements$vecG[componentsNumberETS+componentsNumberARIMA+1:xregNumber]-0.5)));
                }
            }
        }
        #### The admissible bounds ####
        else if(bounds=="admissible"){
            # Stationarity condition of ARIMA
            if(arimaModel){
                # Calculate the polynomial roots for AR
                if(arEstimate &&
                   (all(-adamElements$arimaPolynomials$arPolynomial[-1]>0) &
                    sum(-(adamElements$arimaPolynomials$arPolynomial[-1]))>=1)){
                    arPolynomialMatrix[,1] <- -adamElements$arimaPolynomials$arPolynomial[-1];
                    eigenValues <- abs(eigen(arPolynomialMatrix, symmetric=FALSE, only.values=TRUE)$values);
                    if(any(eigenValues>1)){
                        return(1E+100*max(eigenValues));
                    }
                }
            }

            # Stability / invertibility condition for ETS/ARIMA/Dynamic regression.
            eigenValues <- smoothEigens(adamElements$vecG, adamElements$matF, adamElements$matWt,
                                        lagsModelAll, xregModel, obsInSample);
            if(any(eigenValues>1+1E-50)){
                return(1E+100*max(eigenValues));
            }
        }

        # Write down the initials in the recent profile
        profilesRecentTable[] <- adamElements$matVt[,1:lagsModelMax];
        # print(round(B,3))
        # print(adamElements$vecG)
        # print(profilesRecentTable)

        #### Fitter and the losses calculation ####
        adamFitted <- adamCpp$fit(adamElements$matVt, adamElements$matWt,
                                  adamElements$matF, adamElements$vecG,
                                  indexLookupTable, profilesRecentTable,
                                  yInSample, ot,
                                  any(initialType==c("complete","backcasting")), nIterations,
                                  refineHead);

        if(!multisteps){
            if(loss=="likelihood"){
                # Scale for different functions
                scale <- scaler(distribution, Etype, adamFitted$errors[otLogical],
                                adamFitted$fitted[otLogical], obsInSample, other);

                # Calculate the likelihood
                ## as.complex() is needed for failsafe in case of exotic models
                CFValue <- -sum(switch(distribution,
                                       "dnorm"=switch(Etype,
                                                      "A"=dnorm(x=yInSample[otLogical], mean=adamFitted$fitted[otLogical],
                                                                sd=scale, log=TRUE),
                                                      "M"=dnorm(x=yInSample[otLogical], mean=adamFitted$fitted[otLogical],
                                                                sd=scale*adamFitted$fitted[otLogical], log=TRUE)),
                                       "dlaplace"=switch(Etype,
                                                         "A"=dlaplace(q=yInSample[otLogical], mu=adamFitted$fitted[otLogical],
                                                                      scale=scale, log=TRUE),
                                                         "M"=dlaplace(q=yInSample[otLogical], mu=adamFitted$fitted[otLogical],
                                                                      scale=scale*adamFitted$fitted[otLogical], log=TRUE)),
                                       "ds"=switch(Etype,
                                                   "A"=ds(q=yInSample[otLogical],mu=adamFitted$fitted[otLogical],
                                                          scale=scale, log=TRUE),
                                                   "M"=ds(q=yInSample[otLogical],mu=adamFitted$fitted[otLogical],
                                                          scale=scale*sqrt(adamFitted$fitted[otLogical]), log=TRUE)),
                                       "dgnorm"=switch(Etype,
                                                       "A"=dgnorm(q=yInSample[otLogical],mu=adamFitted$fitted[otLogical],
                                                                  scale=scale, shape=other, log=TRUE),
                                                       # suppressWarnings is needed, because the check is done for scalar alpha
                                                       "M"=suppressWarnings(dgnorm(q=yInSample[otLogical],
                                                                                   mu=adamFitted$fitted[otLogical],
                                                                                   scale=scale*(adamFitted$fitted[otLogical]),
                                                                                   shape=other, log=TRUE))),
                                       # "dlogis"=switch(Etype,
                                       #                 "A"=dlogis(x=yInSample[otLogical],
                                       #                            location=adamFitted$fitted[otLogical],
                                       #                            scale=scale, log=TRUE),
                                       #                 "M"=dlogis(x=yInSample[otLogical],
                                       #                            location=adamFitted$fitted[otLogical],
                                       #                            scale=scale*adamFitted$fitted[otLogical], log=TRUE)),
                                       # "dt"=switch(Etype,
                                       #             "A"=dt(adamFitted$errors[otLogical], df=abs(other), log=TRUE),
                                       #             "M"=dt(adamFitted$errors[otLogical]*adamFitted$fitted[otLogical],
                                       #                    df=abs(other), log=TRUE)),
                                       "dalaplace"=switch(Etype,
                                                          "A"=dalaplace(q=yInSample[otLogical],
                                                                        mu=adamFitted$fitted[otLogical],
                                                                        scale=scale, alpha=other, log=TRUE),
                                                          "M"=dalaplace(q=yInSample[otLogical],
                                                                        mu=adamFitted$fitted[otLogical],
                                                                        scale=scale*adamFitted$fitted[otLogical],
                                                                        alpha=other, log=TRUE)),
                                       "dlnorm"=dlnorm(x=yInSample[otLogical],
                                                       meanlog=Re(log(as.complex(adamFitted$fitted[otLogical])))-scale^2/2,
                                                       sdlog=scale, log=TRUE),
                                       "dllaplace"=dlaplace(q=log(yInSample[otLogical]),
                                                            mu=Re(log(as.complex(adamFitted$fitted[otLogical]))),
                                                            scale=scale, log=TRUE) -log(yInSample[otLogical]),
                                       "dls"=ds(q=log(yInSample[otLogical]),
                                                mu=Re(log(as.complex(adamFitted$fitted[otLogical]))),
                                                scale=scale, log=TRUE) -log(yInSample[otLogical]),
                                       "dlgnorm"=dgnorm(q=log(yInSample[otLogical]),
                                                        mu=Re(log(as.complex(adamFitted$fitted[otLogical]))),
                                                        scale=scale, shape=other, log=TRUE) -log(yInSample[otLogical]),
                                       # abs() is needed for rare cases, when negative values are produced for E="A" models
                                       "dinvgauss"=dinvgauss(x=yInSample[otLogical], mean=abs(adamFitted$fitted[otLogical]),
                                                             dispersion=abs(scale/adamFitted$fitted[otLogical]), log=TRUE),
                                       # abs() is a failsafe mechanism for weird cases of negative values in mixed models
                                       "dgamma"=dgamma(x=yInSample[otLogical], shape=1/scale,
                                                       scale=scale*abs(adamFitted$fitted[otLogical]), log=TRUE)
                ));

                # Differential entropy for the logLik of occurrence model
                if(occurrenceModel || any(!otLogical)){
                    CFValueEntropy <- switch(distribution,
                                             "dnorm" = obsZero*(log(sqrt(2*pi)*scale)+0.5),
                                             "dlnorm" = obsZero*(log(sqrt(2*pi)*scale)+0.5)-scale^2/2,
                                             "dlogis" = obsZero*2,
                                             "dlaplace" =,
                                             "dllaplace" =,
                                             "dalaplace" = obsZero*(1 + log(2*scale)),
                                             "ds" =,
                                             "dls" = obsZero*(2 + 2*log(2*scale)),
                                             "dgnorm" =,
                                             "dlgnorm" = obsZero*(1/other-log(other/(2*scale*gamma(1/other)))),
                                             "dt" = obsZero*((scale+1)/2 *
                                                                 (digamma((scale+1)/2)-digamma(scale/2)) +
                                                                 log(sqrt(scale) * beta(scale/2,0.5))),
                                             # "dinvgauss" = obsZero*(0.5*(log(pi/2)+1+suppressWarnings(log(scale)))));
                                             # "dinvgauss" =0);
                                             "dinvgauss" = 0.5*(obsZero*(log(pi/2)+1+suppressWarnings(log(scale)))-
                                                                    sum(log(adamFitted$fitted[!otLogical]))),
                                             "dgamma" = obsZero*(1/scale + log(gamma(1/scale)) +
                                                                     (1-1/scale)*digamma(1/scale)) +
                                                 sum(log(scale*adamFitted$fitted[!otLogical]))
                    );
                    # If the entropy is NA then something is wrong. It shouldn't be!
                    if(is.na(CFValueEntropy)){
                        CFValueEntropy[] <- Inf;
                    }
                    # If it is negative (it shouldn't be), substitute with zero.
                    # Otherwise occurrence screws the demand sizes model
                    if(CFValueEntropy<0){
                        CFValueEntropy[] <- 0;
                    }
                    CFValue <- CFValue + CFValueEntropy;
                }
            }
            else if(loss=="MSE"){
                CFValue <- sum(adamFitted$errors^2)/obsInSample;
            }
            else if(loss=="MAE"){
                CFValue <- sum(abs(adamFitted$errors))/obsInSample;
            }
            else if(loss=="HAM"){
                CFValue <- sum(sqrt(abs(adamFitted$errors)))/obsInSample;
            }
            else if(any(loss==c("LASSO","RIDGE"))){
                ### All of this is needed in order to get rid of initial level, trend, seasonal and xreg parameters
                # Define, how many elements to skip (we don't normalise smoothing parameters)
                persistenceToSkip <- componentsNumberETS + persistenceXregEstimate*xregNumber +
                    phiEstimate + sum(arOrders) + sum(maOrders);

                # Shrink phi to 1
                if(phiEstimate){
                    B[componentsNumberETS + persistenceXregEstimate*xregNumber + 1] <-
                        1-B[componentsNumberETS + persistenceXregEstimate*xregNumber + 1];
                }
                j <- componentsNumberETS + persistenceXregEstimate*xregNumber + phiEstimate;

                # No good understanding how to shrink ARMA. Do these just because:
                # Shrink AR parameters to 1 and
                # Shrink MA parameters to 0
                if(arimaModel && (sum(maOrders)>0 || sum(arOrders)>0)){
                    for(i in 1:length(lags)){
                        B[j+c(1:arOrders[i])] <- 1-B[j+c(1:arOrders[i])];
                        B[j+arOrders[i]+c(1:maOrders[i])] <- B[j+arOrders[i]+c(1:maOrders[i])];
                        j[] <- j+arOrders[i]+maOrders[i];
                    }
                }

                # Don't do anything with the initial states of ETS and ARIMA. Just drop them (don't shrink)
                if(any(initialType==c("backcasting","optimal","two-stage"))){
                    # If there are explanatory variables, shrink their parameters
                    if(xregNumber>0){
                        # Normalise parameters of xreg if they are additive. Otherwise leave - they will be small and close to zero
                        B <- switch(Etype,
                                    "A"=c(B[1:persistenceToSkip],tail(B,xregNumber) / denominator),
                                    "M"=c(B[1:persistenceToSkip],tail(B,xregNumber)));
                    }
                    else{
                        B <- B[1:persistenceToSkip];
                    }
                }

                CFValue <- (switch(Etype,
                                   "A"=(1-lambda)* sqrt(sum((adamFitted$errors/yDenominator)^2)/obsInSample),
                                   "M"=(1-lambda)* sqrt(sum(log(1+adamFitted$errors)^2)/obsInSample)) +
                                switch(loss,
                                       "LASSO"=lambda * sum(abs(B)),
                                       "RIDGE"=lambda * sqrt(sum(B^2))));
            }
            else if(loss=="custom"){
                CFValue <- lossFunction(actual=yInSample,fitted=adamFitted$fitted,B=B);
            }
        }
        else{
            # Call for the Rcpp function to produce a matrix of multistep errors
            adamErrors <- adamCpp$ferrors(adamFitted$states, adamElements$matWt,
                                          adamElements$matF,
                                          indexLookupTable, profilesRecentTable,
                                          h, yInSample)$errors;

            # Not done yet: "aMSEh","aTMSE","aGTMSE","aMSCE","aGPL"
            CFValue <- switch(loss,
                              "MSEh"=sum(adamErrors[,h]^2)/(obsInSample-h),
                              "TMSE"=sum(colSums(adamErrors^2)/(obsInSample-h)),
                              "GTMSE"=sum(log(colSums(adamErrors^2)/(obsInSample-h))),
                              "MSCE"=sum(rowSums(adamErrors)^2)/(obsInSample-h),
                              "MAEh"=sum(abs(adamErrors[,h]))/(obsInSample-h),
                              "TMAE"=sum(colSums(abs(adamErrors))/(obsInSample-h)),
                              "GTMAE"=sum(log(colSums(abs(adamErrors))/(obsInSample-h))),
                              "MACE"=sum(abs(rowSums(adamErrors)))/(obsInSample-h),
                              "HAMh"=sum(sqrt(abs(adamErrors[,h])))/(obsInSample-h),
                              "THAM"=sum(colSums(sqrt(abs(adamErrors)))/(obsInSample-h)),
                              "GTHAM"=sum(log(colSums(sqrt(abs(adamErrors)))/(obsInSample-h))),
                              "CHAM"=sum(sqrt(abs(rowSums(adamErrors))))/(obsInSample-h),
                              "GPL"=log(det(t(adamErrors) %*% adamErrors/(obsInSample-h))),
                              0);

        }

        if(is.na(CFValue) || is.nan(CFValue)){
            CFValue[] <- 1e+300;
        }

        return(CFValue);
    }

    #### The function returns log-likelihood of the model ####
    logLikADAM <- function(B,
                           etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal, yInSample,
                           ot, otLogical, occurrenceModel, pFitted, obsInSample,
                           componentsNumberETS, componentsNumberETSSeasonal, componentsNumberETSNonSeasonal,
                           componentsNumberARIMA,
                           lags, lagsModel, lagsModelAll, lagsModelMax,
                           indexLookupTable, profilesRecentTable,
                           matVt, matWt, matF, vecG,
                           persistenceEstimate, persistenceLevelEstimate, persistenceTrendEstimate,
                           persistenceSeasonalEstimate, persistenceXregEstimate,
                           phiEstimate, initialType, initialEstimate,
                           initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                           initialArimaEstimate, initialXregEstimate,
                           arimaModel, nonZeroARI, nonZeroMA, arEstimate, maEstimate, arimaPolynomials,
                           arOrders, iOrders, maOrders, arRequired, maRequired, armaParameters,
                           xregModel, xregNumber,
                           xregParametersMissing, xregParametersIncluded,
                           xregParametersEstimated, xregParametersPersistence,
                           constantRequired, constantEstimate,
                           bounds, loss, lossFunction, distribution, horizon, multisteps,
                           denominator=NULL, yDenominator=NULL,
                           other, otherParameterEstimate, lambda,
                           arPolynomialMatrix, maPolynomialMatrix,
                           adamCpp){

        if(!multisteps){
            if(any(loss==c("LASSO","RIDGE"))){
                return(0);
            }
            else{
                distributionNew <- switch(loss,
                                          "MSE"="dnorm",
                                          "MAE"="dlaplace",
                                          "HAM"="ds",
                                          distribution);

                lossNew <- switch(loss,
                                  "MSE"=,"MAE"=,"HAM"="likelihood",
                                  loss)

                # bounds="none" switches off the checks of parameters.
                logLikReturn <- -CF(B,
                                    etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal, yInSample,
                                    ot, otLogical, occurrenceModel, obsInSample,
                                    componentsNumberETS, componentsNumberETSSeasonal, componentsNumberETSNonSeasonal,
                                    componentsNumberARIMA,
                                    lags, lagsModel, lagsModelAll, lagsModelMax,
                                    indexLookupTable, profilesRecentTable,
                                    matVt, matWt, matF, vecG,
                                    persistenceEstimate, persistenceLevelEstimate, persistenceTrendEstimate,
                                    persistenceSeasonalEstimate, persistenceXregEstimate,
                                    phiEstimate, initialType, initialEstimate,
                                    initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                                    initialArimaEstimate, initialXregEstimate,
                                    arimaModel, nonZeroARI, nonZeroMA, arEstimate, maEstimate, arimaPolynomials,
                                    arOrders, iOrders, maOrders, arRequired, maRequired, armaParameters,
                                    xregModel, xregNumber,
                                    xregParametersMissing, xregParametersIncluded,
                                    xregParametersEstimated, xregParametersPersistence,
                                    constantRequired, constantEstimate,
                                    bounds="none", lossNew, lossFunction, distributionNew, horizon, multisteps,
                                    denominator, yDenominator, other, otherParameterEstimate, lambda,
                                    arPolynomialMatrix, maPolynomialMatrix, adamCpp);

                # print(B);
                # print(logLikReturn)

                # If this is an occurrence model, add the probabilities
                if(occurrenceModel){
                    if(is.infinite(logLikReturn)){
                        logLikReturn[] <- 0;
                    }
                    if(any(c(1-pFitted[!otLogical]==0,pFitted[otLogical]==0))){
                        # return(-Inf);
                        ptNew <- pFitted[(pFitted!=0) & (pFitted!=1)];
                        otNew <- ot[(pFitted!=0) & (pFitted!=1)];

                        # Just return the original likelihood if the probability is weird
                        if(length(ptNew)==0){
                            return(logLikReturn);
                        }
                        else{
                            return(logLikReturn + sum(log(ptNew[otNew==1])) + sum(log(1-ptNew[otNew==0])));
                        }
                    }
                    else{
                        return(logLikReturn + sum(log(pFitted[otLogical])) + sum(log(1-pFitted[!otLogical])));
                    }
                }
                else{
                    return(logLikReturn);
                }
            }
        }
        else{
            # Use the predictive likelihoods from the GPL paper:
            # - Normal for MSEh, MSCE, GPL and their analytical counterparts
            # - Laplace for MAEh and MACE,
            # - S for HAMh and CHAM
            # bounds="none" switches off the checks of parameters.
            logLikReturn <- CF(B,
                               etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal, yInSample,
                               ot, otLogical, occurrenceModel, obsInSample,
                               componentsNumberETS, componentsNumberETSSeasonal, componentsNumberETSNonSeasonal,
                               componentsNumberARIMA,
                               lags, lagsModel, lagsModelAll, lagsModelMax,
                               indexLookupTable, profilesRecentTable,
                               matVt, matWt, matF, vecG,
                               persistenceEstimate, persistenceLevelEstimate, persistenceTrendEstimate,
                               persistenceSeasonalEstimate, persistenceXregEstimate,
                               phiEstimate, initialType, initialEstimate,
                               initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                               initialArimaEstimate, initialXregEstimate,
                               arimaModel, nonZeroARI, nonZeroMA, arEstimate, maEstimate, arimaPolynomials,
                               arOrders, iOrders, maOrders, arRequired, maRequired, armaParameters,
                               xregModel, xregNumber,
                               xregParametersMissing, xregParametersIncluded,
                               xregParametersEstimated, xregParametersPersistence,
                               constantRequired, constantEstimate,
                               bounds="none", loss, lossFunction, distribution, horizon, multisteps,
                               denominator, yDenominator,
                               other, otherParameterEstimate, lambda,
                               arPolynomialMatrix, maPolynomialMatrix, adamCpp);

            # Concentrated log-likelihoods for the multistep losses
            logLikReturn[] <- -switch(loss,
                                      "MSEh"=, "aMSEh"=, "TMSE"=, "aTMSE"=, "MSCE"=, "aMSCE"=
                                          (obsInSample-h)/2*(log(2*pi)+1+log(logLikReturn)),
                                      "GTMSE"=, "aGTMSE"=
                                          (obsInSample-h)/2*(log(2*pi)+1+logLikReturn),
                                      "MAEh"=, "TMAE"=, "GTMAE"=, "MACE"=
                                          (obsInSample-h)*(log(2)+1+log(logLikReturn)),
                                      "HAMh"=, "THAM"=, "GTHAM"=, "CHAM"=
                                          (obsInSample-h)*(log(4)+2+2*log(logLikReturn)),
                                      #### Divide GPL by 8 in order to make it comparable with the univariate ones
                                      "GPL"=, "aGPL"=
                                          (obsInSample-h)/2*(h*log(2*pi)+h+logLikReturn)/h);

            # This is not well motivated at the moment, but should make likelihood comparable, taking T instead of T-h
            logLikReturn[] <- logLikReturn / (obsInSample-h) * obsInSample;

            # In case of multiplicative model, we assume a normal or similar distribution
            if(Etype=="M"){
                # Fill in the matrices
                adamElements <- filler(B,
                                       etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal,
                                       componentsNumberETS, componentsNumberETSNonSeasonal,
                                       componentsNumberETSSeasonal, componentsNumberARIMA,
                                       lags, lagsModel, lagsModelMax,
                                       matVt, matWt, matF, vecG,
                                       persistenceEstimate, persistenceLevelEstimate, persistenceTrendEstimate,
                                       persistenceSeasonalEstimate, persistenceXregEstimate,
                                       phiEstimate,
                                       initialType, initialEstimate,
                                       initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                                       initialArimaEstimate, initialXregEstimate,
                                       arimaModel, arEstimate, maEstimate, arOrders, iOrders, maOrders,
                                       arRequired, maRequired, armaParameters,
                                       nonZeroARI, nonZeroMA, arimaPolynomials,
                                       xregModel, xregNumber,
                                       xregParametersMissing, xregParametersIncluded,
                                       xregParametersEstimated, xregParametersPersistence, constantEstimate,
                                       adamCpp);

                # Write down the initials in the recent profile
                profilesRecentTable[] <- adamElements$matVt[,1:lagsModelMax];

                # Fit the model again to extract the fitted values
                adamFitted <- adamCpp$fit(adamElements$matVt, adamElements$matWt,
                                          adamElements$matF, adamElements$vecG,
                                          indexLookupTable, profilesRecentTable,
                                          yInSample, ot,
                                          any(initialType==c("complete","backcasting")), nIterations,
                                          refineHead);
                logLikReturn[] <- logLikReturn - sum(log(abs(adamFitted$fitted)));
            }

            return(logLikReturn);
        }
    }

    #### The function estimates the ETS model and returns B, logLik, nParam and CF(B) ####
    estimator <- function(etsModel, Etype, Ttype, Stype, lags, lagsModelSeasonal, lagsModelARIMA,
                          obsStates, obsInSample,
                          yInSample, persistence, persistenceEstimate,
                          persistenceLevel, persistenceLevelEstimate,
                          persistenceTrend, persistenceTrendEstimate,
                          persistenceSeasonal, persistenceSeasonalEstimate,
                          persistenceXreg, persistenceXregEstimate, persistenceXregProvided,
                          phi, phiEstimate,
                          initialType, initialLevel, initialTrend, initialSeasonal,
                          initialArima, initialEstimate,
                          initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                          initialArimaEstimate, initialXregEstimate, initialXregProvided,
                          arimaModel, arRequired, iRequired, maRequired, armaParameters,
                          componentsNumberARIMA, componentsNamesARIMA,
                          formula, xregModel, xregModelInitials, xregData, xregNumber, xregNames, regressors,
                          xregParametersMissing, xregParametersIncluded,
                          xregParametersEstimated, xregParametersPersistence,
                          constantRequired, constantEstimate, constantValue, constantName,
                          ot, otLogical, occurrenceModel, pFitted,
                          bounds, loss, lossFunction, distribution,
                          horizon, multisteps, other, otherParameterEstimate, lambda, B){

        # Create the basic variables
        adamArchitect <- architector(etsModel, Etype, Ttype, Stype, lags, lagsModelSeasonal,
                                     xregNumber, obsInSample, initialType,
                                     arimaModel, lagsModelARIMA, xregModel, constantRequired,
                                     profilesRecentTable, profilesRecentProvided);
        list2env(adamArchitect, environment());

        # Create the matrices for the specific ETS model
        adamCreated <- creator(etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal,
                               lags, lagsModel, lagsModelARIMA, lagsModelAll, lagsModelMax,
                               profilesRecentTable, profilesRecentProvided,
                               obsStates, obsInSample, obsAll, componentsNumberETS, componentsNumberETSSeasonal,
                               componentsNamesETS, otLogical, yInSample,
                               persistence, persistenceEstimate,
                               persistenceLevel, persistenceLevelEstimate, persistenceTrend, persistenceTrendEstimate,
                               persistenceSeasonal, persistenceSeasonalEstimate,
                               persistenceXreg, persistenceXregEstimate, persistenceXregProvided,
                               phi,
                               initialType, initialEstimate,
                               initialLevel, initialLevelEstimate, initialTrend, initialTrendEstimate,
                               initialSeasonal, initialSeasonalEstimate,
                               initialArima, initialArimaEstimate, initialArimaNumber,
                               initialXregEstimate, initialXregProvided,
                               arimaModel, arRequired, iRequired, maRequired, armaParameters,
                               arOrders, iOrders, maOrders,
                               componentsNumberARIMA, componentsNamesARIMA,
                               xregModel, xregModelInitials, xregData, xregNumber, xregNames,
                               xregParametersPersistence,
                               constantRequired, constantEstimate, constantValue, constantName,
                               adamCpp);

        # Initialise B
        BValues <- initialiser(etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal,
                               componentsNumberETSNonSeasonal, componentsNumberETSSeasonal, componentsNumberETS,
                               lags, lagsModel, lagsModelSeasonal, lagsModelARIMA, lagsModelMax,
                               adamCreated$matVt,
                               persistenceEstimate, persistenceLevelEstimate, persistenceTrendEstimate,
                               persistenceSeasonalEstimate, persistenceXregEstimate,
                               phiEstimate, initialType, initialEstimate,
                               initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                               initialArimaEstimate, initialXregEstimate,
                               arimaModel, arRequired, maRequired, arEstimate, maEstimate, arOrders, maOrders,
                               componentsNumberARIMA, componentsNamesARIMA, initialArimaNumber,
                               xregModel, xregNumber,
                               xregParametersEstimated, xregParametersPersistence,
                               constantEstimate, constantName, otherParameterEstimate,
                               adamCpp);
        if(!is.null(B)){
            if(!is.null(names(B))){
                B <- B[names(B) %in% names(BValues$B)];
                BValues$B[] <- B;
            }
            else{
                BValues$B[] <- B;
                names(B) <- names(BValues$B);
            }
        }
        # print(BValues$B);

        #### Preheating initial parameters ####
        # Preheat the initial parameters Do this only for two-stage initials and if B is not provided
        if(initialType=="two-stage" && is.null(B)){
            clNew <- cl;
            # If environment is provided, use it
            if(!is.null(ellipsis$environment)){
                env <- ellipsis$environment;
            }
            # Modify model in case of ETS+ARIMA
            if(etsModel){
                clNew$model <- paste0(Etype, Ttype, "d"[phiEstimate], Stype);
            }
            # Use complete backcasting
            clNew$initial <- "complete";
            # Shut things up
            clNew$silent <- TRUE;
            # If this is an xreg model, we do selection, and there's no formula, create one
            if(xregModel && !is.null(clNew$regressors) && clNew$regressors=="select"){
                # Matrix allows to avoid unnecessary data expansions via model.matrix
                clNew$data <- as.matrix(cbind(as.data.frame(y), xregData));
                colnames(clNew$data)[[1]] <- responseName;
                # clNew$formula <- as.formula(paste0(responseName,"~",paste0(xregNames,collapse="+")));
                # Remove formula - we don't need it for the provided data
                clNew$formula <- NULL;
            }
            clNew$fast <- TRUE;
            # Switch off regressors selection
            if(!is.null(clNew$regressors) && clNew$regressors=="select"){
                clNew$regressors <- "use";
            }
            # Get rid of explanatory variables if they are not needed
            if(!xregModel && (!is.null(ncol(data)) && ncol(data)>1)){
                clNew$data <- data[,responseName];
            }
            # Call for ADAM with backcasting
            adamBack <- suppressWarnings(eval(clNew, envir=env));
            # Vector of initial estimates of parameters
            B <- BValues$B;
            # Number of smoothing, dampening and ARMA parameters
            nParametersBack <- (etsModel*(persistenceLevelEstimate + modelIsTrendy*persistenceTrendEstimate +
                                              modelIsSeasonal*sum(persistenceSeasonalEstimate) + phiEstimate) +
                                    xregModel*persistenceXregEstimate*max(xregParametersPersistence) +
                                    # AR and MA values
                                    arimaModel*(arEstimate*sum(arOrders)+maEstimate*sum(maOrders)));
            if(nParametersBack>0){
                # Use the estimated parameters
                B[1:nParametersBack] <- adamBack$B[1:nParametersBack];
            }

            # Renormalise seasonal initials and then remove the redundant ones
            if(modelIsSeasonal){
                if(length(lagsModelSeasonal)>1){
                    for(i in 1:length(lagsModelSeasonal)){
                        adamBack$initial$seasonal[[i]][] <-
                            switch(Stype,
                                   "A"=adamBack$initial$seasonal[[i]] - mean(adamBack$initial$seasonal[[i]]),
                                   "M"=adamBack$initial$seasonal[[i]] /
                                       prod(adamBack$initial$seasonal[[i]])^{1/length(adamBack$initial$seasonal[[i]])});
                        adamBack$initial$seasonal[[i]] <- adamBack$initial$seasonal[[i]][1:(lagsModelSeasonal[i]-1)];
                    }
                }
                else{
                    adamBack$initial$seasonal[] <-
                        switch(Stype,
                               "A"=adamBack$initial$seasonal - mean(adamBack$initial$seasonal),
                               "M"=adamBack$initial$seasonal /
                                   prod(adamBack$initial$seasonal)^{1/length(adamBack$initial$seasonal)});
                    adamBack$initial$seasonal <- adamBack$initial$seasonal[1:(lagsModelSeasonal-1)];
                }
            }

            # If there are explanatory variables, use only those initials that are required
            if(xregModel){
                adamBack$initial$xreg <- adamBack$initial$xreg[xregParametersEstimated==1];
            }

            initialsUnlisted <- unlist(adamBack$initial);
            # If initials are reasonable, use them
            if(!any(is.na(initialsUnlisted))){
                B[nParametersBack + c(1:length(initialsUnlisted))] <- initialsUnlisted;
            }

            # If the constant is used and it's good, record it
            if(constantEstimate && !is.na(adamBack$constant)){
                B[nParametersBack+componentsNumberETS+componentsNumberARIMA+xregNumber+1] <- adamBack$constant;
            }

            # Other parameters (shape etc)
            if(otherParameterEstimate){
                B[length(B)] <- abs(tail(adamBack$B,1));
            }

            # Parameter bounds
            lb <- BValues$Bl;
            ub <- BValues$Bu;

            # Make sure that the bounds are reasonable
            if(any(is.na(lb))){
                lb[is.na(lb)] <- -Inf;
            }
            if(any(lb>B)){
                lb[lb>B] <- -Inf;
            }
            if(any(is.na(ub))){
                ub[is.na(ub)] <- Inf;
            }
            if(any(ub<B)){
                ub[ub<B] <- Inf;
            }
        }

        # Create the vector of initials for the optimisation
        if(is.null(B)){
            B <- BValues$B
        }
        if(is.null(lb)){
            lb <- BValues$Bl;
        }
        if(is.null(ub)){
            ub <- BValues$Bu;
        }

        # Companion matrices for the polynomials calculation -> stationarity/stability checks
        if(arimaModel){
            # AR polynomials
            arPolynomialMatrix <- matrix(0, arOrders %*% lags, arOrders %*% lags);
            if(nrow(arPolynomialMatrix)>1){
                arPolynomialMatrix[2:nrow(arPolynomialMatrix)-1,2:nrow(arPolynomialMatrix)] <- diag(nrow(arPolynomialMatrix)-1);
            }
            # MA polynomials
            maPolynomialMatrix <- matrix(0, maOrders %*% lags, maOrders %*% lags);
            if(nrow(maPolynomialMatrix)>1){
                maPolynomialMatrix[2:nrow(maPolynomialMatrix)-1,2:nrow(maPolynomialMatrix)] <- diag(nrow(maPolynomialMatrix)-1);
            }
        }
        else{
            maPolynomialMatrix <- arPolynomialMatrix <- NULL;
        }

        # If the distribution is default, change it according to the error term
        if(distribution=="default"){
            distributionNew <- switch(loss,
                                      "likelihood"= switch(Etype, "A"= "dnorm", "M"= "dgamma"),
                                      "MAEh"=, "MACE"=, "MAE"= "dlaplace",
                                      "HAMh"=, "CHAM"=, "HAM"= "ds",
                                      "MSEh"=, "MSCE"=, "MSE"=, "GPL"=, "dnorm");
        }
        else{
            distributionNew <- distribution;
        }
        # print(B)
        # print(BValues)
        # print(Etype)
        # print(Ttype)
        # print(Stype)
        # print(arOrders)
        # stop()

        print_level_hidden <- print_level;
        if(print_level==41){
            cat("Initial parameters:",B,"\n");
            print_level[] <- 0;
        }

        maxevalUsed <- maxeval;
        if(is.null(maxeval)){
            maxevalUsed <- length(B) * 40;
            # If this is pure ARIMA, take more time
            # if(arimaModel && !etsModel){
            #     maxevalUsed <- length(B) * 80;
            # }
            # # If it is xregModel, do at least 500 iterations
            # else
            if(xregModel){
                maxevalUsed[] <- length(B) * 100;
                maxevalUsed[] <- max(1000,maxevalUsed);
            }
        }

        # Prepare the denominator needed for the shrinkage of explanatory variables in LASSO / RIDGE
        if(any(loss==c("LASSO","RIDGE"))){
            if(xregNumber>0){
                denominator <- apply(adamCreated$matWt[,componentsNumberETS+componentsNumberARIMA+1:xregNumber,
                                                       drop=FALSE], 2, sd);
                denominator[is.infinite(denominator)] <- 1;
            }
            else{
                denominator <- NULL;
            }
            yDenominator <- max(sd(diff(yInSample)),1);
        }
        else{
            denominator <- NULL;
            yDenominator <- NULL;
        }

        ##### Parameter estimation ####
        # Parameters are chosen to speed up the optimisation process and have decent accuracy
        res <- suppressWarnings(nloptr(B, CF, lb=lb, ub=ub,
                                       opts=list(algorithm=algorithm, xtol_rel=xtol_rel, xtol_abs=xtol_abs,
                                                 ftol_rel=ftol_rel, ftol_abs=ftol_abs,
                                                 maxeval=maxevalUsed, maxtime=maxtime, print_level=print_level),
                                       etsModel=etsModel, Etype=Etype, Ttype=Ttype, Stype=Stype, modelIsTrendy=modelIsTrendy,
                                       modelIsSeasonal=modelIsSeasonal, yInSample=yInSample,
                                       ot=ot, otLogical=otLogical, occurrenceModel=occurrenceModel, obsInSample=obsInSample,
                                       componentsNumberETS=componentsNumberETS,
                                       componentsNumberETSSeasonal=componentsNumberETSSeasonal,
                                       componentsNumberETSNonSeasonal=componentsNumberETSNonSeasonal,
                                       componentsNumberARIMA=componentsNumberARIMA,
                                       lags=lags, lagsModel=lagsModel, lagsModelAll=lagsModelAll, lagsModelMax=lagsModelMax,
                                       indexLookupTable=indexLookupTable, profilesRecentTable=profilesRecentTable,
                                       matVt=adamCreated$matVt, matWt=adamCreated$matWt,
                                       matF=adamCreated$matF, vecG=adamCreated$vecG,
                                       persistenceEstimate=persistenceEstimate, persistenceLevelEstimate=persistenceLevelEstimate,
                                       persistenceTrendEstimate=persistenceTrendEstimate,
                                       persistenceSeasonalEstimate=persistenceSeasonalEstimate,
                                       persistenceXregEstimate=persistenceXregEstimate,
                                       phiEstimate=phiEstimate, initialType=initialType,
                                       initialEstimate=initialEstimate, initialLevelEstimate=initialLevelEstimate,
                                       initialTrendEstimate=initialTrendEstimate, initialSeasonalEstimate=initialSeasonalEstimate,
                                       initialArimaEstimate=initialArimaEstimate, initialXregEstimate=initialXregEstimate,
                                       arimaModel=arimaModel, nonZeroARI=nonZeroARI, nonZeroMA=nonZeroMA,
                                       arimaPolynomials=adamCreated$arimaPolynomials,
                                       arEstimate=arEstimate, maEstimate=maEstimate,
                                       arOrders=arOrders, iOrders=iOrders, maOrders=maOrders,
                                       arRequired=arRequired, maRequired=maRequired, armaParameters=armaParameters,
                                       xregModel=xregModel, xregNumber=xregNumber,
                                       xregParametersMissing=xregParametersMissing,
                                       xregParametersIncluded=xregParametersIncluded,
                                       xregParametersEstimated=xregParametersEstimated,
                                       xregParametersPersistence=xregParametersPersistence,
                                       constantRequired=constantRequired, constantEstimate=constantEstimate,
                                       bounds=bounds, loss=loss, lossFunction=lossFunction, distribution=distributionNew,
                                       horizon=horizon, multisteps=multisteps,
                                       denominator=denominator, yDenominator=yDenominator,
                                       other=other, otherParameterEstimate=otherParameterEstimate, lambda=lambda,
                                       arPolynomialMatrix=arPolynomialMatrix, maPolynomialMatrix=maPolynomialMatrix,
                                       adamCpp=adamCpp));

        if(is.infinite(res$objective) || res$objective==1e+300){
            # If the optimisation didn't work, give it another try with zero initials for smoothing parameters
            if(etsModel){
                B[1:componentsNumberETS] <- 0;
            }
            if(arimaModel){
                B[componentsNumberETS+persistenceXregEstimate*xregNumber+
                      c(1:sum(arOrders*arEstimate,maOrders*maEstimate))] <- 0.01;
            }
            # print(B)
            res <- suppressWarnings(nloptr(B, CF, lb=lb, ub=ub,
                                           opts=list(algorithm=algorithm, xtol_rel=xtol_rel,
                                                     ftol_rel=ftol_rel, ftol_abs=ftol_abs,
                                                     maxeval=maxevalUsed, maxtime=maxtime, print_level=print_level),
                                           etsModel=etsModel, Etype=Etype, Ttype=Ttype, Stype=Stype, modelIsTrendy=modelIsTrendy,
                                           modelIsSeasonal=modelIsSeasonal, yInSample=yInSample,
                                           ot=ot, otLogical=otLogical, occurrenceModel=occurrenceModel, obsInSample=obsInSample,
                                           componentsNumberETS=componentsNumberETS,
                                           componentsNumberETSSeasonal=componentsNumberETSSeasonal,
                                           componentsNumberETSNonSeasonal=componentsNumberETSNonSeasonal,
                                           componentsNumberARIMA=componentsNumberARIMA,
                                           lags=lags, lagsModel=lagsModel, lagsModelAll=lagsModelAll, lagsModelMax=lagsModelMax,
                                           indexLookupTable=indexLookupTable, profilesRecentTable=profilesRecentTable,
                                           matVt=adamCreated$matVt, matWt=adamCreated$matWt,
                                           matF=adamCreated$matF, vecG=adamCreated$vecG,
                                           persistenceEstimate=persistenceEstimate,
                                           persistenceLevelEstimate=persistenceLevelEstimate,
                                           persistenceTrendEstimate=persistenceTrendEstimate,
                                           persistenceSeasonalEstimate=persistenceSeasonalEstimate,
                                           persistenceXregEstimate=persistenceXregEstimate,
                                           phiEstimate=phiEstimate, initialType=initialType,
                                           initialEstimate=initialEstimate, initialLevelEstimate=initialLevelEstimate,
                                           initialTrendEstimate=initialTrendEstimate, initialSeasonalEstimate=initialSeasonalEstimate,
                                           initialArimaEstimate=initialArimaEstimate, initialXregEstimate=initialXregEstimate,
                                           arimaModel=arimaModel, nonZeroARI=nonZeroARI, nonZeroMA=nonZeroMA,
                                           arimaPolynomials=adamCreated$arimaPolynomials,
                                           arEstimate=arEstimate, maEstimate=maEstimate,
                                           arOrders=arOrders, iOrders=iOrders, maOrders=maOrders,
                                           arRequired=arRequired, maRequired=maRequired, armaParameters=armaParameters,
                                           xregModel=xregModel, xregNumber=xregNumber,
                                           xregParametersMissing=xregParametersMissing,
                                           xregParametersIncluded=xregParametersIncluded,
                                           xregParametersEstimated=xregParametersEstimated,
                                           xregParametersPersistence=xregParametersPersistence,
                                           constantRequired=constantRequired, constantEstimate=constantEstimate,
                                           bounds=bounds, loss=loss, lossFunction=lossFunction, distribution=distributionNew,
                                           horizon=horizon, multisteps=multisteps,
                                           denominator=denominator, yDenominator=yDenominator,
                                           other=other, otherParameterEstimate=otherParameterEstimate, lambda=lambda,
                                           arPolynomialMatrix=arPolynomialMatrix, maPolynomialMatrix=maPolynomialMatrix,
                                           adamCpp=adamCpp));
        }

        if(print_level_hidden>0){
            print(res);
        }

        ##### !!! Check the obtained parameters and the loss value and remove redundant parameters !!! #####
        # Cases to consider:
        # 1. Some smoothing parameters are zero or one;
        # 2. The cost function value is -Inf (due to no variability in the sample);

        # Prepare the values to return
        B[] <- res$solution;
        CFValue <- res$objective;
        # A fix for the special case of LASSO/RIDGE with lambda==1
        if(any(loss==c("LASSO","RIDGE")) && lambda==1){
            CFValue[] <- 0;
        }

        nStatesBackcasting <- 0;
        # Calculate the number of degrees of freedom coming from states in case of backcasting
        if(any(initialType==c("backcasting","complete"))){
            # Fill in the matrices. This is now needed for the correct calculation of the df
            adamFilled <- filler(B,
                                 etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal,
                                 componentsNumberETS, componentsNumberETSNonSeasonal,
                                 componentsNumberETSSeasonal, componentsNumberARIMA,
                                 lags, lagsModel, lagsModelMax,
                                 adamCreated$matVt, adamCreated$matWt, adamCreated$matF, adamCreated$vecG,
                                 persistenceEstimate, persistenceLevelEstimate, persistenceTrendEstimate,
                                 persistenceSeasonalEstimate, persistenceXregEstimate,
                                 phiEstimate,
                                 initialType, initialEstimate,
                                 initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                                 initialArimaEstimate, initialXregEstimate,
                                 arimaModel, arEstimate, maEstimate, arOrders, iOrders, maOrders,
                                 arRequired, maRequired, armaParameters,
                                 nonZeroARI, nonZeroMA, adamCreated$arimaPolynomials,
                                 xregModel, xregNumber,
                                 xregParametersMissing, xregParametersIncluded,
                                 xregParametersEstimated, xregParametersPersistence, constantEstimate,
                                 adamCpp);

            nStatesBackcasting[] <- calculateBackcastingDF(profilesRecentTable, lagsModelAll,
                                                           etsModel, Stype, componentsNumberETSNonSeasonal,
                                                           componentsNumberETSSeasonal, adamFilled$vecG, adamFilled$matF,
                                                           obsInSample, lagsModelMax, indexLookupTable,
                                                           adamCpp, dfForBack);
        }

        nParamEstimated <- length(B) + nStatesBackcasting;
        # Return a proper logLik class
        logLikADAMValue <- structure(logLikADAM(B,
                                                etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal, yInSample,
                                                ot, otLogical, occurrenceModel, pFitted, obsInSample,
                                                componentsNumberETS, componentsNumberETSSeasonal, componentsNumberETSNonSeasonal,
                                                componentsNumberARIMA,
                                                lags, lagsModel, lagsModelAll, lagsModelMax,
                                                indexLookupTable, profilesRecentTable,
                                                adamCreated$matVt, adamCreated$matWt, adamCreated$matF, adamCreated$vecG,
                                                persistenceEstimate, persistenceLevelEstimate, persistenceTrendEstimate,
                                                persistenceSeasonalEstimate, persistenceXregEstimate,
                                                phiEstimate, initialType, initialEstimate,
                                                initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                                                initialArimaEstimate, initialXregEstimate,
                                                arimaModel, nonZeroARI, nonZeroMA, arEstimate, maEstimate,
                                                adamCreated$arimaPolynomials,
                                                arOrders, iOrders, maOrders, arRequired, maRequired, armaParameters,
                                                xregModel, xregNumber,
                                                xregParametersMissing, xregParametersIncluded,
                                                xregParametersEstimated, xregParametersPersistence,
                                                constantRequired, constantEstimate,
                                                bounds, loss, lossFunction, distributionNew, horizon, multisteps,
                                                denominator, yDenominator, other, otherParameterEstimate, lambda,
                                                arPolynomialMatrix, maPolynomialMatrix, adamCpp),
                                     # In case of likelihood, we typically have one more parameter to estimate - scale.
                                     nobs=obsInSample,df=nParamEstimated+(loss=="likelihood"),class="logLik");
        xregIndex <- 1;
        #### If we do variables selection, do it here, then reestimate the model. ####
        if(regressors=="select"){
            # This is a failsafe for weird cases, when something went wrong with estimation
            if(any(is.nan(adamCreated$matVt[,1:lagsModelMax]))){
                adamCreated[] <- creator(etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal,
                                         lags, lagsModel, lagsModelARIMA, lagsModelAll, lagsModelMax,
                                         profilesRecentTable, profilesRecentProvided,
                                         obsStates, obsInSample, obsAll, componentsNumberETS, componentsNumberETSSeasonal,
                                         componentsNamesETS, otLogical, yInSample,
                                         persistence, persistenceEstimate,
                                         persistenceLevel, persistenceLevelEstimate, persistenceTrend, persistenceTrendEstimate,
                                         persistenceSeasonal, persistenceSeasonalEstimate,
                                         persistenceXreg, persistenceXregEstimate, persistenceXregProvided,
                                         phi,
                                         initialType, initialEstimate,
                                         initialLevel, initialLevelEstimate, initialTrend, initialTrendEstimate,
                                         initialSeasonal, initialSeasonalEstimate,
                                         initialArima, initialArimaEstimate, initialArimaNumber,
                                         initialXregEstimate, initialXregProvided,
                                         arimaModel, arRequired, iRequired, maRequired, armaParameters,
                                         arOrders, iOrders, maOrders,
                                         componentsNumberARIMA, componentsNamesARIMA,
                                         xregModel, xregModelInitials, xregData, xregNumber, xregNames,
                                         xregParametersPersistence,
                                         constantRequired, constantEstimate, constantValue, constantName,
                                         adamCpp);
            }
            # Fill in the matrices
            adamCreated[] <- filler(B,
                                    etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal,
                                    componentsNumberETS, componentsNumberETSNonSeasonal,
                                    componentsNumberETSSeasonal, componentsNumberARIMA,
                                    lags, lagsModel, lagsModelMax,
                                    adamCreated$matVt, adamCreated$matWt, adamCreated$matF, adamCreated$vecG,
                                    persistenceEstimate, persistenceLevelEstimate, persistenceTrendEstimate,
                                    persistenceSeasonalEstimate, persistenceXregEstimate,
                                    phiEstimate,
                                    initialType, initialEstimate,
                                    initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                                    initialArimaEstimate, initialXregEstimate,
                                    arimaModel, arEstimate, maEstimate, arOrders, iOrders, maOrders,
                                    arRequired, maRequired, armaParameters,
                                    nonZeroARI, nonZeroMA, adamCreated$arimaPolynomials,
                                    xregModel, xregNumber,
                                    xregParametersMissing, xregParametersIncluded,
                                    xregParametersEstimated, xregParametersPersistence, constantEstimate,
                                    adamCpp);

            # Write down the initials in the recent profile
            profilesRecentTable[] <- adamCreated$matVt[,1:lagsModelMax];

            # Fit the model to the data
            adamFitted <- adamCpp$fit(adamCreated$matVt, adamCreated$matWt,
                                      adamCreated$matF, adamCreated$vecG,
                                      indexLookupTable, profilesRecentTable,
                                      yInSample, ot,
                                      any(initialType==c("complete","backcasting")), nIterations,
                                      refineHead);

            # Extract the errors correctly
            errors <- switch(distributionNew,
                             "dlnorm"=, "dllaplace"=, "dls"=,
                             "dlgnorm"=, "dinvgauss"=, "dgamma"=switch(Etype,
                                                                       "A"=1+adamFitted$errors/adamFitted$fitted,
                                                                       "M"=adamFitted$errors),
                             "dnorm"=, "dlaplace"=, "ds"=, "dgnorm"=, "dlogis"=, "dt"=, "dalaplace"=,adamFitted$errors);
            # Extract the errors and amend them to correspond to the distribution
            errors[] <- errors + switch(Etype,"A"=0,"M"=1);

            # This is failsafe for cases, when errors contain negative values, although they shouldn't
            if(any(distributionNew==c("dinvgauss","dgamma","dlnorm","dllaplace","dls","dlgnorm")) &&
               any(c(Etype,Ttype,Stype)=="A") && any(errors<=0)){
                errors[errors<=0] <- 1e-100;
            }

            df <- length(B)+1;
            if(any(distributionNew==c("dalaplace","dgnorm","dlgnorm","dt")) && otherParameterEstimate){
                other <- abs(B[length(B)]);
                df[] <- df - 1;
            }

            # Call the xregSelector providing the original matrix with the data
            xregIndex[] <- switch(Etype,"A"=1,"M"=2);
            xregModelInitials[[xregIndex]] <- xregSelector(errors=errors,
                                                           xregData=xregDataOriginal[1:obsInSample,
                                                                                     colnames(xregDataOriginal)!=responseName,
                                                                                     drop=FALSE],
                                                           ic=ic,
                                                           df=df, distribution=distributionNew, occurrence=oesModel,
                                                           other=other);
            xregNumber <- length(xregModelInitials[[xregIndex]]$initialXreg);
            xregNames <- names(xregModelInitials[[xregIndex]]$initialXreg);

            # Fix the names of variables
            xregNames[] <- make.names(xregNames, unique=TRUE);

            # If there are some variables, then do the proper reestimation and return the new values
            if(xregNumber>0){
                xregModel[] <- TRUE;
                initialXregEstimate[] <- initialXregEstimateOriginal;
                persistenceXregEstimate[] <- persistenceXregEstimateOriginal;
                # xregData <- xregDataOriginal[,xregNames,drop=FALSE];

                # Redefine loss for ALM
                lossNew <- switch(loss,
                                  "MSEh"=,"TMSE"=,"GTMSE"=,"MSCE"="MSE",
                                  "MAEh"=,"TMAE"=,"GTMAE"=,"MACE"="MAE",
                                  "HAMh"=,"THAM"=,"GTHAM"=,"CHAM"="HAM",
                                  loss);
                if(lossNew=="custom"){
                    lossNew <- lossFunction;
                }

                # Fix the name of the response variable
                xregModelInitials[[xregIndex]]$formula[[2]] <- as.name(responseName);
                formula <- xregModelInitials[[xregIndex]]$formula;
                xregModelInitials[[which(c(1,2)!=xregIndex)]]$formula <- formulaToUse <- formula;

                # Fix formula if dnorm / dlaplace / ds etc are used for Etype=="M"
                trendIncluded <- any(all.vars(formulaToUse)[-1]=="trend");
                if((length(formulaToUse[[2]])==1 ||
                    (length(formulaToUse[[2]])>1 & !any(as.character(formulaToUse[[2]])=="log"))) &&
                   (Etype=="M" && any(distribution==c("dnorm","dlaplace","ds","dgnorm","dlogis","dt","dalaplace")))){
                    if(trendIncluded){
                        formulaToUse <- update(formulaToUse,log(.)~.);
                    }
                    else{
                        formulaToUse <- update(formulaToUse,log(.)~.+trend);
                    }
                }
                else{
                    if(!trendIncluded){
                        formulaToUse <- update(formulaToUse,.~.+trend);
                    }
                }

                # Estimate alm again in order to get proper initials
                almModel <- do.call(alm,list(formula=formulaToUse,
                                             data=data[1:obsInSample,,drop=FALSE],
                                             distribution=distributionNew, loss=lossNew, occurrence=oesModel));

                # Remove trend
                if(!trendIncluded){
                    almModel$coefficients <- almModel$coefficients[names(almModel$coefficients)!="trend"];
                    almModel$data <- almModel$data[,colnames(almModel$data)!="trend",drop=FALSE];
                }
                almIntercept <- almModel$coefficients["(Intercept)"];
                xregModelInitials[[xregIndex]]$initialXreg <- coef(almModel)[-1];

                # Create the B vector to speed up the calculation
                B <- c(B, xregModelInitials[[xregIndex]]$initialXreg);

                #### Fix xreg vectors based on the selected stuff ####
                xregNames <- colnames(almModel$data)[-1];

                # Robustify the names of variables
                colnames(data) <- make.names(colnames(data),unique=TRUE);
                # The names of the original variables
                xregNamesOriginal <- all.vars(formula)[-1];
                # Expand the variables. We cannot use alm, because it is based on obsInSample
                xregData <- model.frame(formula,data=as.data.frame(data));
                # Binary, flagging factors in the data
                xregFactors <- (attr(terms(xregData),"dataClasses")=="factor")[-1];
                # Expanded stuff with all levels for factors
                if(any(xregFactors)){
                    # Levels for the factors
                    xregFactorsLevels <- lapply(data,levels);
                    xregFactorsLevels[[responseName]] <- NULL;
                    xregModelMatrix <- model.matrix(xregData,xregData,
                                                    contrasts.arg=lapply(xregData[attr(terms(xregData),"dataClasses")=="factor"],
                                                                         contrasts, contrasts=FALSE));
                    xregNamesModified <- colnames(xregModelMatrix)[-1];
                }
                else{
                    xregModelMatrix <- model.matrix(xregData,data=xregData);
                    xregNamesModified <- xregNames;
                }
                xregData <- as.matrix(xregModelMatrix);
                # Remove intercept
                interceptIsPresent <- FALSE;
                if(any(colnames(xregData)=="(Intercept)")){
                    interceptIsPresent[] <- TRUE;
                    xregData <- xregData[,-1,drop=FALSE];
                }
                xregNumber <- ncol(xregData);

                # If there are factors not in the alm data, create additional initials
                if(any(xregFactors) && any(!(xregNamesModified %in% xregNames))){
                    # The indices of the original parameters
                    xregParametersMissing <- setNames(vector("numeric",xregNumber),xregNamesModified);
                    # # The indices of the original parameters
                    xregParametersIncluded <- setNames(vector("numeric",xregNumber),xregNamesModified);
                    # The vector, marking the same values of smoothing parameters
                    if(interceptIsPresent){
                        xregParametersPersistence <- setNames(attr(xregModelMatrix,"assign")[-1],xregNamesModified);
                    }
                    else{
                        xregParametersPersistence <- setNames(attr(xregModelMatrix,"assign"),xregNamesModified);
                    }

                    xregAbsent <- !(xregNamesModified %in% xregNames);
                    xregParametersNew <- setNames(rep(NA,xregNumber),xregNamesModified);
                    xregParametersNew[!xregAbsent] <- xregModelInitials[[xregIndex]]$initialXreg;
                    # Go through new names and find, where they came from. Then get the missing parameters
                    for(i in which(xregAbsent)){
                        # Find the name of the original variable
                        # Use only the last value... hoping that the names like x and x1 are not used.
                        xregNameFoundID <- sapply(xregNamesOriginal,grepl,xregNamesModified[i]);
                        xregNameFound <- tail(names(xregNameFoundID)[xregNameFoundID],1);
                        # Get the indices of all k-1 levels
                        xregParametersIncluded[xregNames[xregNames %in% paste0(xregNameFound,
                                                                               xregFactorsLevels[[xregNameFound]])]] <- i;
                        # Get the index of the absent one
                        xregParametersMissing[i] <- i;

                        # Fill in the absent one, add intercept
                        xregParametersNew[i] <- almIntercept;
                        xregParametersNew[xregNamesModified[xregParametersIncluded==i]] <- almIntercept +
                            xregParametersNew[xregNamesModified[xregParametersIncluded==i]];
                        # normalise all of them
                        xregParametersNew[xregNamesModified[c(which(xregParametersIncluded==i),i)]] <-
                            xregParametersNew[xregNamesModified[c(which(xregParametersIncluded==i),i)]] -
                            mean(xregParametersNew[xregNamesModified[c(which(xregParametersIncluded==i),i)]]);
                    }
                    # Write down the new parameters
                    xregModelInitials[[xregIndex]]$initialXreg <- xregParametersNew;
                    xregNames <- xregNamesModified;

                    # The vector of parameters that should be estimated (numeric + original levels of factors)
                    xregParametersEstimated <- xregParametersIncluded
                    xregParametersEstimated[xregParametersEstimated!=0] <- 1;
                    xregParametersEstimated[xregParametersMissing==0 & xregParametersIncluded==0] <- 1;
                }
                else{
                    xregFactors <- FALSE;
                    xregParametersPersistence <- setNames(c(1:xregNumber),xregNames);
                    xregParametersEstimated <- setNames(rep(1,xregNumber),xregNames);
                    xregParametersMissing <- setNames(rep(0,xregNumber),xregNames);
                    xregParametersIncluded <- setNames(c(1:xregNumber),xregNames);
                }

                return(estimator(etsModel, Etype, Ttype, Stype, lags, lagsModelSeasonal, lagsModelARIMA,
                                 obsStates, obsInSample,
                                 yInSample, persistence, persistenceEstimate,
                                 persistenceLevel, persistenceLevelEstimate,
                                 persistenceTrend, persistenceTrendEstimate,
                                 persistenceSeasonal, persistenceSeasonalEstimate,
                                 persistenceXreg, persistenceXregEstimate, persistenceXregProvided,
                                 phi, phiEstimate,
                                 initialType, initialLevel, initialTrend, initialSeasonal,
                                 initialArima, initialEstimate,
                                 initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                                 initialArimaEstimate, initialXregEstimate, initialXregProvided,
                                 arimaModel, arRequired, iRequired, maRequired, armaParameters,
                                 componentsNumberARIMA, componentsNamesARIMA,
                                 formula, xregModel, xregModelInitials, xregData, xregNumber, xregNames, regressors="use",
                                 xregParametersMissing, xregParametersIncluded,
                                 xregParametersEstimated, xregParametersPersistence,
                                 constantRequired, constantEstimate, constantValue, constantName,
                                 ot, otLogical, occurrenceModel, pFitted,
                                 bounds, loss, lossFunction, distribution,
                                 horizon, multisteps, other, otherParameterEstimate, lambda, B));
            }
        }

        return(list(B=B, CFValue=CFValue, nParamEstimated=nParamEstimated,
                    logLikADAMValue=logLikADAMValue,
                    xregModel=xregModel, xregData=xregData, xregNumber=xregNumber,
                    xregNames=xregNames, xregModelInitials=xregModelInitials, formula=formula,
                    initialXregEstimate=initialXregEstimate, persistenceXregEstimate=persistenceXregEstimate,
                    xregParametersMissing=xregParametersMissing,xregParametersIncluded=xregParametersIncluded,
                    xregParametersEstimated=xregParametersEstimated,xregParametersPersistence=xregParametersPersistence,
                    arimaPolynomials=adamCreated$arimaPolynomials,
                    res=res,adamCpp=adamCpp));
    }


    #### The function creates a pool of models and selects the best of them ####
    selector <- function(model, modelsPool, allowMultiplicative,
                         etsModel, Etype, Ttype, Stype, damped, lags,
                         lagsModelSeasonal, lagsModelARIMA,
                         obsStates, obsInSample,
                         yInSample, persistence, persistenceEstimate,
                         persistenceLevel, persistenceLevelEstimate,
                         persistenceTrend, persistenceTrendEstimate,
                         persistenceSeasonal, persistenceSeasonalEstimate,
                         persistenceXreg, persistenceXregEstimate, persistenceXregProvided,
                         phi, phiEstimate,
                         initialType, initialLevel, initialTrend, initialSeasonal,
                         initialArima, initialEstimate,
                         initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                         initialArimaEstimate, initialXregEstimate, initialXregProvided,
                         arimaModel, arRequired, iRequired, maRequired, armaParameters,
                         componentsNumberARIMA, componentsNamesARIMA,
                         xregModel, xregModelInitials, xregData, xregNumber, xregNames, regressors,
                         xregParametersMissing, xregParametersIncluded,
                         xregParametersEstimated, xregParametersPersistence,
                         constantRequired, constantEstimate, constantValue, constantName,
                         ot, otLogical, occurrenceModel, pFitted, icFunction,
                         bounds, loss, lossFunction, distribution,
                         horizon, multisteps, other, otherParameterEstimate, lambda){

        # Check if the pool was provided. In case of "no", form the big and the small ones
        if(is.null(modelsPool)){
            # The variable saying that the pool was not provided.
            if(!silent){
                cat("Forming the pool of models based on... ");
            }

            # Define the whole pool of errors
            if(!allowMultiplicative){
                poolErrors <- c("A");
                poolTrends <- c("N","A","Ad");
                poolSeasonals <- c("N","A");
            }
            else{
                poolErrors <- c("A","M");
                poolTrends <- c("N","A","Ad","M","Md");
                poolSeasonals <- c("N","A","M");
            }

            # Some preparation variables
            # If Etype is not Z, then check on additive errors
            if(Etype!="Z"){
                poolErrors <- poolErrorsSmall <- Etype;
            }
            else{
                poolErrorsSmall <- "A";
            }

            # If Ttype is not Z, then create a pool with specified type
            if(Ttype!="Z"){
                if(Ttype=="X"){
                    poolTrendsSmall <- c("N","A");
                    poolTrends <- c("N","A","Ad");
                    checkTrend <- TRUE;
                }
                else if(Ttype=="Y"){
                    poolTrendsSmall <- c("N","M");
                    poolTrends <- c("N","M","Md");
                    checkTrend <- TRUE;
                }
                else{
                    if(damped){
                        poolTrends <- poolTrendsSmall <- paste0(Ttype,"d");
                    }
                    else{
                        poolTrends <- poolTrendsSmall <- Ttype;
                    }
                    checkTrend <- FALSE;
                }
            }
            else{
                poolTrendsSmall <- c("N","A");
                checkTrend <- TRUE;
            }

            # If Stype is not Z, then create specific pools
            if(Stype!="Z"){
                if(Stype=="X"){
                    poolSeasonals <- poolSeasonalsSmall <- c("N","A");
                    checkSeasonal <- TRUE;
                }
                else if(Stype=="Y"){
                    poolSeasonalsSmall <- c("N","M");
                    poolSeasonals <- c("N","M");
                    checkSeasonal <- TRUE;
                }
                else{
                    poolSeasonalsSmall <- Stype;
                    poolSeasonals <- Stype;
                    checkSeasonal <- FALSE;
                }
            }
            else{
                poolSeasonalsSmall <- c("N","A","M");
                checkSeasonal <- TRUE;
            }

            # If ZZZ, then the vector is: "ANN" "ANA" "ANM" "AAN" "AAA" "AAM"
            # Otherwise id depends on the provided restrictions
            poolSmall <- paste0(rep(poolErrorsSmall,length(poolTrendsSmall)*length(poolSeasonalsSmall)),
                                rep(poolTrendsSmall,each=length(poolSeasonalsSmall)),
                                rep(poolSeasonalsSmall,length(poolTrendsSmall)));
            # Align error and seasonality, if the error was not forced to be additive
            # The new pool: "ANN" "ANA" "MNM" "AAN" "AAA" "MAM"
            if(any(substr(poolSmall,3,3)=="M") && all(Etype!=c("A","X"))){
                multiplicativeSeason <- (substr(poolSmall,3,3)=="M");
                poolSmall[multiplicativeSeason] <- paste0("M",substr(poolSmall[multiplicativeSeason],2,3));
            }
            modelsTested <- NULL;
            modelCurrent <- NA;

            # Counter + checks for the components
            j <- 1;
            i <- 0;
            check <- TRUE;
            besti <- bestj <- 1;
            results <- vector("list",length(poolSmall));

            #### Branch and bound is here ####
            while(check){
                i <- i + 1;
                modelCurrent[] <- poolSmall[j];
                if(!silent){
                    cat(modelCurrent,"\b, ");
                }
                Etype[] <- substring(modelCurrent,1,1);
                Ttype[] <- substring(modelCurrent,2,2);
                if(nchar(modelCurrent)==4){
                    phi[] <- 0.95;
                    phiEstimate[] <- TRUE;
                    Stype[] <- substring(modelCurrent,4,4);
                }
                else{
                    phi[] <- 1;
                    phiEstimate[] <- FALSE;
                    Stype[] <- substring(modelCurrent,3,3);
                }

                results[[i]] <- estimator(etsModel, Etype, Ttype, Stype, lags, lagsModelSeasonal, lagsModelARIMA,
                                          obsStates, obsInSample,
                                          yInSample, persistence, persistenceEstimate,
                                          persistenceLevel, persistenceLevelEstimate,
                                          persistenceTrend, persistenceTrendEstimate,
                                          persistenceSeasonal, persistenceSeasonalEstimate,
                                          persistenceXreg, persistenceXregEstimate, persistenceXregProvided,
                                          phi, phiEstimate,
                                          initialType, initialLevel, initialTrend, initialSeasonal,
                                          initialArima, initialEstimate,
                                          initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                                          initialArimaEstimate, initialXregEstimate, initialXregProvided,
                                          arimaModel, arRequired, iRequired, maRequired, armaParameters,
                                          componentsNumberARIMA, componentsNamesARIMA,
                                          formula, xregModel, xregModelInitials, xregData, xregNumber, xregNames, regressors,
                                          xregParametersMissing, xregParametersIncluded,
                                          xregParametersEstimated, xregParametersPersistence,
                                          constantRequired, constantEstimate, constantValue, constantName,
                                          ot, otLogical, occurrenceModel, pFitted,
                                          bounds, loss, lossFunction, distribution,
                                          horizon, multisteps, other, otherParameterEstimate, lambda, B);
                results[[i]]$IC <- icFunction(results[[i]]$logLikADAMValue);
                results[[i]]$Etype <- Etype;
                results[[i]]$Ttype <- Ttype;
                results[[i]]$Stype <- Stype;
                results[[i]]$phiEstimate <- phiEstimate;
                if(phiEstimate){
                    results[[i]]$phi <- results[[i]]$B[names(results[[i]]$B)=="phi"];
                }
                else{
                    results[[i]]$phi <- 1;
                }
                results[[i]]$model <- modelCurrent;

                modelsTested <- c(modelsTested,modelCurrent);

                if(j>1){
                    # If the first is better than the second, then choose first
                    if(results[[besti]]$IC <= results[[i]]$IC){
                        # If Ttype is the same, then we check seasonality
                        if(substring(modelCurrent,2,2)==substring(poolSmall[bestj],2,2)){
                            poolSeasonals <- results[[besti]]$Stype;
                            checkSeasonal <- FALSE;
                            j <- which(poolSmall!=poolSmall[bestj] &
                                           substring(poolSmall,nchar(poolSmall),nchar(poolSmall))==poolSeasonals);
                        }
                        # Otherwise we checked trend
                        else{
                            poolTrends <- results[[bestj]]$Ttype;
                            checkTrend[] <- FALSE;
                        }
                    }
                    else{
                        # If the trend is the same
                        if(substring(modelCurrent,2,2) == substring(poolSmall[besti],2,2)){
                            poolSeasonals <- poolSeasonals[poolSeasonals!=results[[besti]]$Stype];
                            if(length(poolSeasonals)>1){
                                # Select another seasonal model, that is not from the previous iteration and not the current one
                                bestj[] <- j;
                                besti[] <- i;
                                # j[] <- 3;
                                j <- 3;
                            }
                            else{
                                bestj[] <- j;
                                besti[] <- i;
                                # Move to checking the trend
                                j <- which(substring(poolSmall,nchar(poolSmall),nchar(poolSmall))==poolSeasonals &
                                               substring(poolSmall,2,2)!=substring(modelCurrent,2,2));
                                checkSeasonal[] <- FALSE;
                            }
                        }
                        else{
                            poolTrends <- poolTrends[poolTrends!=results[[bestj]]$Ttype];
                            besti[] <- i;
                            bestj[] <- j;
                            checkTrend[] <- FALSE;
                        }
                    }

                    if(all(!c(checkTrend,checkSeasonal))){
                        check[] <- FALSE;
                    }
                }
                else{
                    j <- 2;
                }

                # If this is NULL, then this was a short pool and we checked everything
                if(length(j)==0){
                    j <- length(poolSmall);
                }
                if(j>length(poolSmall)){
                    check[] <- FALSE;
                }
            }

            # Prepare a bigger pool based on the small one
            modelsPool <- unique(c(modelsTested,
                                   paste0(rep(poolErrors,each=length(poolTrends)*length(poolSeasonals)),
                                          poolTrends,
                                          rep(poolSeasonals,each=length(poolTrends)))));
            j <- length(modelsTested);
        }
        else{
            j <- 0;
            results <- vector("list",length(modelsPool));
        }
        modelsNumber <- length(modelsPool);

        #### Run the full pool of models ####
        if(!silent){
            cat("Estimation progress:    ");
        }
        # Start loop of models
        while(j < modelsNumber){
            j <- j + 1;
            if(!silent){
                if(j==1){
                    cat("\b");
                }
                cat(paste0(rep("\b",nchar(round((j-1)/modelsNumber,2)*100)+1),collapse=""));
                cat(round(j/modelsNumber,2)*100,"\b%");
            }

            modelCurrent <- modelsPool[j];
            # print(modelCurrent)
            Etype <- substring(modelCurrent,1,1);
            Ttype <- substring(modelCurrent,2,2);
            if(nchar(modelCurrent)==4){
                phi[] <- 0.95;
                Stype <- substring(modelCurrent,4,4);
                phiEstimate <- TRUE;
            }
            else{
                phi[] <- 1;
                Stype <- substring(modelCurrent,3,3);
                phiEstimate <- FALSE;
            }

            results[[j]] <- estimator(etsModel, Etype, Ttype, Stype, lags, lagsModelSeasonal, lagsModelARIMA,
                                      obsStates, obsInSample,
                                      yInSample, persistence, persistenceEstimate,
                                      persistenceLevel, persistenceLevelEstimate,
                                      persistenceTrend, persistenceTrendEstimate,
                                      persistenceSeasonal, persistenceSeasonalEstimate,
                                      persistenceXreg, persistenceXregEstimate, persistenceXregProvided,
                                      phi, phiEstimate,
                                      initialType, initialLevel, initialTrend, initialSeasonal,
                                      initialArima, initialEstimate,
                                      initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                                      initialArimaEstimate, initialXregEstimate, initialXregProvided,
                                      arimaModel, arRequired, iRequired, maRequired, armaParameters,
                                      componentsNumberARIMA, componentsNamesARIMA,
                                      formula, xregModel, xregModelInitials, xregData, xregNumber, xregNames, regressors,
                                      xregParametersMissing, xregParametersIncluded,
                                      xregParametersEstimated, xregParametersPersistence,
                                      constantRequired, constantEstimate, constantValue, constantName,
                                      ot, otLogical, occurrenceModel, pFitted,
                                      bounds, loss, lossFunction, distribution,
                                      horizon, multisteps, other, otherParameterEstimate, lambda, B);
            results[[j]]$IC <- icFunction(results[[j]]$logLikADAMValue);
            results[[j]]$Etype <- Etype;
            results[[j]]$Ttype <- Ttype;
            results[[j]]$Stype <- Stype;
            results[[j]]$phiEstimate <- phiEstimate;
            if(phiEstimate){
                results[[j]]$phi <- results[[j]]$B[names(results[[j]]$B)=="phi"];
            }
            else{
                results[[j]]$phi <- 1;
            }
            results[[j]]$model <- modelCurrent;
        }

        if(!silent){
            cat("... Done! \n");
        }

        # Extract ICs and find the best
        icSelection <- vector("numeric",modelsNumber);
        for(i in 1:modelsNumber){
            icSelection[i] <- results[[i]]$IC;
        }
        names(icSelection) <- modelsPool;

        icSelection[is.nan(icSelection)] <- 1E100;

        return(list(results=results,icSelection=icSelection));
    }

    ##### Function uses residuals in order to determine the needed xreg #####
    xregSelector <- function(errors, xregData, ic, df, distribution, occurrence, other){
        alpha <- shape <- nu <- NULL;
        if(distribution=="dalaplace"){
            alpha <- other;
        }
        else if(any(distribution==c("dgnorm","dlgnorm"))){
            shape <- other;
        }
        else if(distribution=="dt"){
            nu <- other;
        }
        stepwiseModel <- suppressWarnings(stepwise(data.frame(errorsIvan41=errors,xregData[1:obsInSample,,drop=FALSE]),
                                                   ic=ic, df=df, distribution=distribution, occurrence=occurrence, silent=TRUE,
                                                   alpha=alpha, shape=shape, nu=nu));
        return(list(initialXreg=coef(stepwiseModel)[-1],other=stepwiseModel$other,formula=formula(stepwiseModel)));
    }

    ##### Function prepares all the matrices and vectors for return #####
    preparator <- function(B, etsModel, Etype, Ttype, Stype,
                           lagsModel, lagsModelMax, lagsModelAll,
                           componentsNumberETS, componentsNumberETSSeasonal,
                           xregNumber, distribution, loss,
                           persistenceEstimate, persistenceLevelEstimate, persistenceTrendEstimate,
                           persistenceSeasonalEstimate, persistenceXregEstimate,
                           phiEstimate, otherParameterEstimate,
                           initialType, initialEstimate,
                           initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                           initialArimaEstimate, initialXregEstimate,
                           matVt, matWt, matF, vecG,
                           occurrenceModel, ot, oesModel,
                           parametersNumber, CFValue,
                           arimaModel, arRequired, maRequired,
                           arEstimate, maEstimate, arOrders, iOrders, maOrders,
                           nonZeroARI, nonZeroMA,
                           arimaPolynomials, armaParameters,
                           constantRequired, constantEstimate, adamCpp){

        if(modelDo!="use"){
            # Fill in the matrices
            adamElements <- filler(B,
                                   etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal,
                                   componentsNumberETS, componentsNumberETSNonSeasonal,
                                   componentsNumberETSSeasonal, componentsNumberARIMA,
                                   lags, lagsModel, lagsModelMax,
                                   matVt, matWt, matF, vecG,
                                   persistenceEstimate, persistenceLevelEstimate, persistenceTrendEstimate,
                                   persistenceSeasonalEstimate, persistenceXregEstimate,
                                   phiEstimate,
                                   initialType, initialEstimate,
                                   initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                                   initialArimaEstimate, initialXregEstimate,
                                   arimaModel, arEstimate, maEstimate, arOrders, iOrders, maOrders,
                                   arRequired, maRequired, armaParameters,
                                   nonZeroARI, nonZeroMA, arimaPolynomials,
                                   xregModel, xregNumber,
                                   xregParametersMissing, xregParametersIncluded,
                                   xregParametersEstimated, xregParametersPersistence, constantEstimate,
                                   adamCpp);
            list2env(adamElements, environment());
        }

        # Write down phi
        if(phiEstimate){
            phi[] <- B[names(B)=="phi"];
        }

        # Write down the initials in the recent profile
        profilesRecentTable[] <- matVt[,1:lagsModelMax];
        profilesRecentInitial <- matVt[,1:lagsModelMax, drop=FALSE];

        # Fit the model to the data
        adamFitted <- adamCpp$fit(matVt, matWt,
                                  matF, vecG,
                                  indexLookupTable, profilesRecentTable,
                                  yInSample, ot,
                                  any(initialType==c("complete","backcasting")), nIterations,
                                  refineHead);

        matVt[] <- adamFitted$states;

        # Write down the recent profile for future use
        profilesRecentTable <- adamFitted$profile;

        # Make sure that there are no negative values in multiplicative components
        # This might appear in case of bounds="a"
        if(Ttype=="M" && (any(is.na(matVt[2,])) || any(matVt[2,]<=0))){
            i <- which(any(matVt[2,]<=0));
            matVt[2,i] <- 1e-6;
            profilesRecentTable[2,i] <- 1e-6;
        }

        if(Stype=="M" && all(!is.na(matVt[componentsNumberETSNonSeasonal+1:componentsNumberETSSeasonal,])) &&
           any(matVt[componentsNumberETSNonSeasonal+1:componentsNumberETSSeasonal,]<=0)){
            i <- which(matVt[componentsNumberETSNonSeasonal+1:componentsNumberETSSeasonal,]<=0);
            matVt[componentsNumberETSNonSeasonal+1:componentsNumberETSSeasonal,i] <- 1e-6;
            i <- which(profilesRecentTable[componentsNumberETSNonSeasonal+1:componentsNumberETSSeasonal,]<=0);
            profilesRecentTable[componentsNumberETSNonSeasonal+1:componentsNumberETSSeasonal,i] <- 1e-6;
        }

        # Prepare fitted and error with ts / zoo
        if(any(yClasses=="ts")){
            yFitted <- ts(rep(NA,obsInSample), start=yStart, frequency=yFrequency);
            errors <- ts(rep(NA,obsInSample), start=yStart, frequency=yFrequency);
        }
        else{
            yFitted <- zoo(rep(NA,obsInSample), order.by=yInSampleIndex);
            errors <- zoo(rep(NA,obsInSample), order.by=yInSampleIndex);
        }

        errors[] <- adamFitted$errors;
        yFitted[] <- adamFitted$fitted;
        # Check what was returned in the end
        if(any(is.nan(yFitted)) || any(is.na(yFitted))){
            warning("Something went wrong in the estimation of the model and NaNs were produced. ",
                    "If this is a mixed model, consider using the pure ones instead.",
                    call.=FALSE, immediate.=TRUE);
        }
        if(occurrenceModel){
            yFitted[] <- yFitted * pFitted;
        }

        # Fix the cases, when we have zeroes in the provided occurrence
        if(occurrence=="provided"){
            yFitted[!otLogical] <- yFitted[!otLogical] * pFitted[!otLogical];
        }

        # Produce forecasts if the horizon is non-zero
        if(horizon>0){
            if(any(yClasses=="ts")){
                yForecast <- ts(rep(NA, horizon), start=yForecastStart, frequency=yFrequency);
            }
            else{
                yForecast <- zoo(rep(NA, horizon), order.by=yForecastIndex);
            }

            yForecast[] <- adamCpp$forecast(tail(matWt,horizon), matF,
                                            indexLookupTable[,lagsModelMax+obsInSample+c(1:horizon),drop=FALSE],
                                            profilesRecentTable,
                                            horizon)$forecast;
            #### Make safety checks
            # If there are NaN values
            if(any(is.nan(yForecast))){
                yForecast[is.nan(yForecast)] <- 0;
            }

            # Amend forecasts, multiplying by probability
            if(occurrenceModel && !occurrenceModelProvided){
                yForecast[] <- yForecast * c(suppressWarnings(forecast(oesModel, h=h))$mean);
            }
            else if((occurrenceModel && occurrenceModelProvided) || occurrence=="provided"){
                yForecast[] <- yForecast * pForecast;
            }
        }
        else{
            if(any(yClasses=="ts")){
                yForecast <- ts(NA, start=yForecastStart, frequency=yFrequency);
            }
            else{
                yForecast <- zoo(rep(NA, horizon), order.by=yForecastIndex);
            }
        }

        # If the distribution is default, change it according to the error term
        if(distribution=="default"){
            distribution[] <- switch(loss,
                                     "likelihood"= switch(Etype, "A"= "dnorm", "M"= "dgamma"),
                                     "MAEh"=, "MACE"=, "MAE"= "dlaplace",
                                     "HAMh"=, "CHAM"=, "HAM"= "ds",
                                     "MSEh"=, "MSCE"=, "MSE"=, "GPL"=, "dnorm");
        }

        #### Initial values to return ####
        initialValue <- vector("list", etsModel*(1+modelIsTrendy+modelIsSeasonal)+arimaModel+xregModel);
        initialValueETS <- vector("list", etsModel*length(lagsModel));
        initialValueNames <- vector("character", etsModel*(1+modelIsTrendy+modelIsSeasonal)+arimaModel+xregModel);
        # The vector that defines what was estimated in the model
        initialEstimated <- vector("logical", etsModel*(1+modelIsTrendy+modelIsSeasonal*componentsNumberETSSeasonal)+
                                       arimaModel+xregModel);

        # Write down the initials of ETS
        j <- 0;
        if(etsModel){
            # Write down level, trend and seasonal
            for(i in 1:length(lagsModel)){
                # In case of level / trend, we want to get the very first value
                if(lagsModel[i]==1){
                    initialValueETS[[i]] <- head(matVt[i,1:lagsModelMax],1);
                }
                # In cases of seasonal components, they should be at the end of the pre-heat period
                else{
                    initialValueETS[[i]] <- tail(matVt[i,1:lagsModelMax],lagsModel[i]);
                }
            }
            j[] <- j+1;
            # Write down level in the final list
            initialEstimated[j] <- initialLevelEstimate;
            initialValue[[j]] <- initialValueETS[[j]];
            initialValueNames[j] <- c("level");
            names(initialEstimated)[j] <- initialValueNames[j];
            if(modelIsTrendy){
                j[] <- 2;
                initialEstimated[j] <- initialTrendEstimate;
                # Write down trend in the final list
                initialValue[[j]] <- initialValueETS[[j]];
                # Remove the trend from ETS list
                initialValueETS[[j]] <- NULL;
                initialValueNames[j] <- c("trend");
                names(initialEstimated)[j] <- initialValueNames[j];
            }
            # Write down the initial seasonals
            if(modelIsSeasonal){
                initialEstimated[j+c(1:componentsNumberETSSeasonal)] <- initialSeasonalEstimate;
                # Remove the level from ETS list
                initialValueETS[[1]] <- NULL;
                j[] <- j+1;
                if(length(initialSeasonalEstimate)>1){
                    initialValue[[j]] <- initialValueETS;
                    initialValueNames[[j]] <- "seasonal";
                    names(initialEstimated)[j+0:(componentsNumberETSSeasonal-1)] <-
                        paste0(initialValueNames[j],c(1:componentsNumberETSSeasonal));
                }
                else{
                    initialValue[[j]] <- initialValueETS[[1]];
                    initialValueNames[[j]] <- "seasonal";
                    names(initialEstimated)[j] <- initialValueNames[j];
                }
            }
        }

        # Write down the ARIMA initials
        if(arimaModel){
            j[] <- j+1;
            initialEstimated[j] <- initialArimaEstimate;
            if(initialArimaEstimate){
                initialValue[[j]] <- head(matVt[componentsNumberETS+componentsNumberARIMA,],initialArimaNumber);
                # Fix the values to get proper initials, not just the values of states
                if(tail(arimaPolynomials$ariPolynomial,1)!=0){
                    initialValue[[j]] <- switch(Etype,
                                                "A"=initialValue[[j]] / tail(arimaPolynomials$ariPolynomial,1),
                                                "M"=exp(log(initialValue[[j]]) / tail(arimaPolynomials$ariPolynomial,1)));
                    # initialValue[[j]] <- initialValue[[j]] / tail(arimaPolynomials$ariPolynomial,1);
                }
            }
            else{
                initialValue[[j]] <- initialArima;
            }
            initialValueNames[j] <- "arima";
            names(initialEstimated)[j] <- initialValueNames[j];
        }
        # Write down the xreg initials
        if(xregModel){
            j[] <- j+1;
            initialEstimated[j] <- initialXregEstimate;
            initialValue[[j]] <- matVt[componentsNumberETS+componentsNumberARIMA+1:xregNumber,lagsModelMax];
            initialValueNames[j] <- "xreg";
            names(initialEstimated)[j] <- initialValueNames[j];
        }
        names(initialValue) <- initialValueNames;

        #### Persistence to return ####
        persistence <- as.vector(vecG);
        names(persistence) <- rownames(vecG);

        # Remove xreg persistence from the returned vector
        if(xregModel && regressors!="adapt"){
            # persistence <- persistence[substr(names(persistence),1,5)!="delta"];
            # We've selected the variables, so there's nothing to select anymore
            regressors <- "use";
        }
        else if(!xregModel){
            regressors <- NULL;
        }

        if(arimaModel){
            armaParametersList <- vector("list",arRequired+maRequired);
            j[] <- 1;
            if(arRequired && arEstimate){
                # Avoid damping parameter phi
                armaParametersList[[j]] <- B[nchar(names(B))>3 & substr(names(B),1,3)=="phi"];
                names(armaParametersList)[j] <- "ar";
                j[] <- j+1;
            }
            # If this was provided
            else if(arRequired && !arEstimate){
                # Avoid damping parameter phi
                armaParametersList[[j]] <- armaParameters[substr(names(armaParameters),1,3)=="phi"];
                names(armaParametersList)[j] <- "ar";
                j[] <- j+1;
            }
            if(maRequired && maEstimate){
                armaParametersList[[j]] <- B[substr(names(B),1,5)=="theta"];
                names(armaParametersList)[j] <- "ma";
            }
            else if(maRequired && !maEstimate){
                armaParametersList[[j]] <- armaParameters[substr(names(armaParameters),1,5)=="theta"];
                names(armaParametersList)[j] <- "ma";
            }
        }
        else{
            armaParametersList <- NULL;
        }

        if(any(distribution==c("dalaplace","dgnorm","dlgnorm","dt")) && otherParameterEstimate){
            other <- abs(tail(B,1));
        }
        # which() is needed in order to overcome weird behaviour of zoo
        scale <- scaler(distribution, Etype, errors[which(otLogical)], yFitted[which(otLogical)], obsInSample, other);

        # Record constant if it was estimated
        if(constantEstimate){
            constantValue <- B[constantName];
        }

        # Prepare the list of distribution parameters to return
        otherReturned <- vector("list",1);
        # Write down parameters for distribution. It is always positive, so take abs
        if(otherParameterEstimate){
            otherReturned[[1]] <- abs(tail(B,1));
        }
        else{
            otherReturned[[1]] <- other;
        }
        # Give names to the other values
        if(distribution=="dalaplace"){
            names(otherReturned) <- "alpha";
        }
        else if(any(distribution==c("dgnorm","dlgnorm"))){
            names(otherReturned) <- "shape";
        }
        else if(any(distribution==c("dt"))){
            names(otherReturned) <- "nu";
        }
        # LASSO / RIDGE lambda
        if(any(loss==c("LASSO","RIDGE"))){
            otherReturned$lambda <- lambda;
        }
        # Return ARIMA polynomials and indices for persistence and transition
        if(arimaModel){
            otherReturned$polynomial <- arimaPolynomials;
            otherReturned$ARIMAIndices <- list(nonZeroARI=nonZeroARI,nonZeroMA=nonZeroMA);
            otherReturned$arPolynomialMatrix <- matrix(0, arOrders %*% lags, arOrders %*% lags);
            if(nrow(otherReturned$arPolynomialMatrix)>1){
                otherReturned$arPolynomialMatrix[2:nrow(otherReturned$arPolynomialMatrix)-1,
                                                 2:nrow(otherReturned$arPolynomialMatrix)] <-
                    diag(nrow(otherReturned$arPolynomialMatrix)-1);
                if(arRequired){
                    otherReturned$arPolynomialMatrix[,1] <- -arimaPolynomials$arPolynomial[-1];
                }
            }
            otherReturned$armaParameters <- armaParameters;
        }

        # Amend the class of state matrix
        if(any(yClasses=="ts")){
            matVt <- ts(t(matVt), start=(time(y)[1]-deltat(y)*lagsModelMax), frequency=yFrequency);
        }
        else{
            yStatesIndex <- yInSampleIndex[1] - lagsModelMax*diff(tail(yInSampleIndex,2)) +
                c(1:lagsModelMax-1)*diff(tail(yInSampleIndex,2));
            yStatesIndex <- c(yStatesIndex, yInSampleIndex);
            matVt <- zoo(t(matVt), order.by=yStatesIndex);
        }

        parametersNumber[2,5] <- sum(parametersNumber[2,1:4]);

        return(list(model=NA, timeElapsed=NA,
                    data=cbind(NA,xregData), holdout=NULL, fitted=yFitted, residuals=errors,
                    forecast=yForecast, states=matVt,
                    profile=profilesRecentTable, profileInitial=profilesRecentInitial,
                    persistence=persistence, phi=phi, transition=matF,
                    measurement=matWt, initial=initialValue, initialType=initialType,
                    initialEstimated=initialEstimated, orders=orders, arma=armaParametersList,
                    constant=constantValue, nParam=parametersNumber,
                    occurrence=oesModel, formula=formula, regressors=regressors,
                    loss=loss, lossValue=CFValue, logLik=logLikADAMValue, distribution=distribution,
                    scale=scale, other=otherReturned, B=B, lags=lags, lagsAll=lagsModelAll, ets=ets,
                    res=res, FI=FI, adamCpp=adamCpp));
    }

    #### Deal with occurrence model ####
    if(occurrenceModel && !occurrenceModelProvided){
        modelForOES <- model;
        if(model=="NNN"){
            modelForOES[] <- "MNN";
        }
        oesModel <- suppressWarnings(oes(ot, model=modelForOES, occurrence=occurrence, ic=ic, h=horizon,
                                         holdout=FALSE, bounds="usual", xreg=xregData, regressors=regressors, silent=TRUE));
        pFitted[] <- fitted(oesModel);
        parametersNumber[1,3] <- nparam(oesModel);
        # print(oesModel)
        # This should not happen, but just in case...
        if(oesModel$occurrence=="n"){
            occurrence <- "n";
            otLogical <- rep(TRUE,obsInSample);
            occurrenceModel <- FALSE;
            ot <- matrix(otLogical*1,ncol=1);
            obsNonzero <- sum(ot);
            obsZero <- obsInSample - obsNonzero;
            Etype[] <- switch(Etype,
                              "M"="A",
                              "Y"=,
                              "Z"="X",
                              Etype);
            Ttype[] <- switch(Ttype,
                              "M"="A",
                              "Y"=,
                              "Z"="X",
                              Ttype);
            Stype[] <- switch(Stype,
                              "M"="A",
                              "Y"=,
                              "Z"="X",
                              Stype);
        }
    }
    else if(occurrenceModel && occurrenceModelProvided){
        parametersNumber[2,3] <- nparam(oesModel);
    }

    xregDataOriginal <- xregData;
    ##### Prepare stuff for the variables selection if regressors="select" #####
    if(regressors=="select"){
        # First, record the original parameters
        xregExistOriginal <- xregModel;
        initialXregsProvidedOriginal <- initialXregProvided;
        initialXregEstimateOriginal <- initialXregEstimate;
        persistenceXregOriginal <- persistenceXreg;
        persistenceXregProvidedOriginal <- persistenceXregProvided;
        persistenceXregEstimateOriginal <- persistenceXregEstimate;
        xregModelOriginal <- xregModelInitials;
        xregNumberOriginal <- xregNumber;
        xregNamesOriginal <- xregNames;

        # Set the parameters to zero and do simple ETS
        xregModel[] <- FALSE;
        initialXregProvided <- FALSE;
        initialXregEstimate[] <- FALSE;
        persistenceXreg <- 0;
        persistenceXregProvided <- FALSE;
        persistenceXregEstimate[] <- FALSE;
        xregData <- NULL;
        xregNumber[] <- 0;
        xregNames <- NULL;
    }

    ##### Estimate the specified model #####
    if(modelDo=="estimate"){
        # If this is LASSO/RIDGE with lambda=1, use MSE to estimate initials only
        lambdaOriginal <- lambda;
        if(any(loss==c("LASSO","RIDGE")) && lambda==1){
            if(etsModel){
                # Pre-set ETS parameters
                persistenceEstimate[] <- FALSE;
                persistenceLevelEstimate[] <- persistenceTrendEstimate[] <-
                    persistenceSeasonalEstimate[] <- FALSE;
                persistenceLevel <- persistenceTrend <- persistenceSeasonal <- 0;
                # Phi
                phiEstimate[] <- FALSE;
                phi <- 1;
            }
            if(xregModel){
                # ETSX parameters
                persistenceXregEstimate[] <- FALSE;
                persistenceXreg <- 0;
            }
            if(arimaModel){
                # Pre-set ARMA parameters
                arEstimate[] <- FALSE;
                maEstimate[] <- FALSE;
                armaParameters <- vector("numeric",sum(arOrders)+sum(maOrders));
                j <- 0;
                for(i in 1:length(lags)){
                    if(arOrders[i]>0){
                        armaParameters[j+1:arOrders[i]] <- 1;
                        names(armaParameters)[j+c(1:arOrders[i])] <- paste0("phi",1:arOrders[i],"[",lags[i],"]");
                        j <- j + arOrders[i];
                    }
                    if(maOrders[i]>0){
                        armaParameters[j+1:maOrders[i]] <- 0;
                        names(armaParameters)[j+c(1:maOrders[i])] <- paste0("theta",1:maOrders[i],"[",lags[i],"]");
                        j <- j + maOrders[i];
                    }
                }
            }
            lambda <- 0;
        }

        # Estimate the parameters of the demand sizes model
        adamEstimated <- estimator(etsModel, Etype, Ttype, Stype, lags, lagsModelSeasonal, lagsModelARIMA,
                                   obsStates, obsInSample,
                                   yInSample, persistence, persistenceEstimate,
                                   persistenceLevel, persistenceLevelEstimate,
                                   persistenceTrend, persistenceTrendEstimate,
                                   persistenceSeasonal, persistenceSeasonalEstimate,
                                   persistenceXreg, persistenceXregEstimate, persistenceXregProvided,
                                   phi, phiEstimate,
                                   initialType, initialLevel, initialTrend, initialSeasonal,
                                   initialArima, initialEstimate,
                                   initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                                   initialArimaEstimate, initialXregEstimate, initialXregProvided,
                                   arimaModel, arRequired, iRequired, maRequired, armaParameters,
                                   componentsNumberARIMA, componentsNamesARIMA,
                                   formula, xregModel, xregModelInitials, xregData, xregNumber, xregNames, regressors,
                                   xregParametersMissing, xregParametersIncluded,
                                   xregParametersEstimated, xregParametersPersistence,
                                   constantRequired, constantEstimate, constantValue, constantName,
                                   ot, otLogical, occurrenceModel, pFitted,
                                   bounds, loss, lossFunction, distribution,
                                   horizon, multisteps, other, otherParameterEstimate, lambda, B);
        list2env(adamEstimated, environment());

        # A fix for the special case of lambda==1
        lambda <- lambdaOriginal;

        #### This part is needed in order for the filler to do its job later on
        # Create the basic variables based on the estimated model
        adamArchitect <- architector(etsModel, Etype, Ttype, Stype, lags, lagsModelSeasonal,
                                     xregNumber, obsInSample, initialType,
                                     arimaModel, lagsModelARIMA, xregModel, constantRequired,
                                     profilesRecentTable, profilesRecentProvided);
        list2env(adamArchitect, environment());

        # Create the matrices for the specific ETS model
        adamCreated <- creator(etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal,
                               lags, lagsModel, lagsModelARIMA, lagsModelAll, lagsModelMax,
                               profilesRecentTable, profilesRecentProvided,
                               obsStates, obsInSample, obsAll, componentsNumberETS, componentsNumberETSSeasonal,
                               componentsNamesETS, otLogical, yInSample,
                               persistence, persistenceEstimate,
                               persistenceLevel, persistenceLevelEstimate, persistenceTrend, persistenceTrendEstimate,
                               persistenceSeasonal, persistenceSeasonalEstimate,
                               persistenceXreg, persistenceXregEstimate, persistenceXregProvided,
                               phi,
                               initialType, initialEstimate,
                               initialLevel, initialLevelEstimate, initialTrend, initialTrendEstimate,
                               initialSeasonal, initialSeasonalEstimate,
                               initialArima, initialArimaEstimate, initialArimaNumber,
                               initialXregEstimate, initialXregProvided,
                               arimaModel, arRequired, iRequired, maRequired, armaParameters,
                               arOrders, iOrders, maOrders,
                               componentsNumberARIMA, componentsNamesARIMA,
                               xregModel, xregModelInitials, xregData, xregNumber, xregNames,
                               xregParametersPersistence,
                               constantRequired, constantEstimate, constantValue, constantName,
                               adamCpp);
        list2env(adamCreated, environment());

        icSelection <- icFunction(adamEstimated$logLikADAMValue);

        ####!!! If the occurrence is auto, then compare this with the model with no occurrence !!!####

        parametersNumber[1,1] <- nParamEstimated;
        if(xregModel){
            parametersNumber[1,2] <- sum(xregParametersEstimated)*initialXregEstimate +
                max(xregParametersPersistence)*persistenceXregEstimate;
            parametersNumber[1,1] <- parametersNumber[1,1] - parametersNumber[1,2];
        }
        # If we used likelihood, scale was estimated
        if((loss=="likelihood")){
            parametersNumber[1,4] <- 1;
        }
        parametersNumber[1,5] <- sum(parametersNumber[1,1:4]);
        parametersNumber[2,5] <- sum(parametersNumber[2,1:4]);
    }
    #### Selection of the best model ####
    else if(modelDo=="select"){
        adamSelected <- selector(model, modelsPool, allowMultiplicative,
                                 etsModel, Etype, Ttype, Stype, damped, lags,
                                 lagsModelSeasonal, lagsModelARIMA,
                                 obsStates, obsInSample,
                                 yInSample, persistence, persistenceEstimate,
                                 persistenceLevel, persistenceLevelEstimate,
                                 persistenceTrend, persistenceTrendEstimate,
                                 persistenceSeasonal, persistenceSeasonalEstimate,
                                 persistenceXreg, persistenceXregEstimate, persistenceXregProvided,
                                 phi, phiEstimate,
                                 initialType, initialLevel, initialTrend, initialSeasonal,
                                 initialArima, initialEstimate,
                                 initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                                 initialArimaEstimate, initialXregEstimate, initialXregProvided,
                                 arimaModel, arRequired, iRequired, maRequired, armaParameters,
                                 componentsNumberARIMA, componentsNamesARIMA,
                                 xregModel, xregModelInitials, xregData, xregNumber, xregNames, regressors,
                                 xregParametersMissing, xregParametersIncluded,
                                 xregParametersEstimated, xregParametersPersistence,
                                 constantRequired, constantEstimate, constantValue, constantName,
                                 ot, otLogical, occurrenceModel, pFitted, icFunction,
                                 bounds, loss, lossFunction, distribution,
                                 horizon, multisteps, other, otherParameterEstimate, lambda);

        icSelection <- adamSelected$icSelection;
        # Take the parameters of the best model
        list2env(adamSelected$results[[which.min(icSelection)[1]]], environment());

        #### This part is needed in order for the filler to do its job later on
        # Create the basic variables based on the estimated model
        adamArchitect <- architector(etsModel, Etype, Ttype, Stype, lags, lagsModelSeasonal,
                                     xregNumber, obsInSample, initialType,
                                     arimaModel, lagsModelARIMA, xregModel, constantRequired,
                                     profilesRecentTable, profilesRecentProvided);
        list2env(adamArchitect, environment());

        # Create the matrices for the specific ETS model
        adamCreated <- creator(etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal,
                               lags, lagsModel, lagsModelARIMA, lagsModelAll, lagsModelMax,
                               profilesRecentTable, profilesRecentProvided,
                               obsStates, obsInSample, obsAll, componentsNumberETS, componentsNumberETSSeasonal,
                               componentsNamesETS, otLogical, yInSample,
                               persistence, persistenceEstimate,
                               persistenceLevel, persistenceLevelEstimate, persistenceTrend, persistenceTrendEstimate,
                               persistenceSeasonal, persistenceSeasonalEstimate,
                               persistenceXreg, persistenceXregEstimate, persistenceXregProvided,
                               phi,
                               initialType, initialEstimate,
                               initialLevel, initialLevelEstimate, initialTrend, initialTrendEstimate,
                               initialSeasonal, initialSeasonalEstimate,
                               initialArima, initialArimaEstimate, initialArimaNumber,
                               initialXregEstimate, initialXregProvided,
                               arimaModel, arRequired, iRequired, maRequired, armaParameters,
                               arOrders, iOrders, maOrders,
                               componentsNumberARIMA, componentsNamesARIMA,
                               xregModel, xregModelInitials, xregData, xregNumber, xregNames,
                               xregParametersPersistence,
                               constantRequired, constantEstimate, constantValue, constantName,
                               adamCpp);
        list2env(adamCreated, environment());

        parametersNumber[1,1] <- nParamEstimated;
        if(xregModel){
            parametersNumber[1,2] <- xregNumber*initialXregEstimate + xregNumber*persistenceXregEstimate;
            parametersNumber[1,1] <- parametersNumber[1,1] - parametersNumber[1,2];
        }
        # If we used likelihood, scale was estimated
        if((loss=="likelihood")){
            parametersNumber[1,4] <- 1;
        }
        parametersNumber[1,5] <- sum(parametersNumber[1,1:4]);
        parametersNumber[2,5] <- sum(parametersNumber[2,1:4]);
    }
    #### Combination of models ####
    else if(modelDo=="combine"){
        modelOriginal <- model;
        # If the pool is not provided, then create one
        if(is.null(modelsPool)){
            # Define the whole pool of errors
            if(!allowMultiplicative){
                poolErrors <- c("A");
                poolTrends <- c("N","A","Ad");
                poolSeasonals <- c("N","A");
            }
            else{
                poolErrors <- c("A","M");
                poolTrends <- c("N","A","Ad","M","Md");
                poolSeasonals <- c("N","A","M");
            }

            # Some preparation variables
            # If Etype is not Z, then check on additive errors
            if(Etype!="Z"){
                poolErrors <- switch(Etype,
                                     "N"="N",
                                     "A"=,
                                     "X"="A",
                                     "M"=,
                                     "Y"="M");
            }

            # If Ttype is not Z, then create a pool with specified type
            if(Ttype!="Z"){
                poolTrends <- switch(Ttype,
                                     "N"="N",
                                     "A"=ifelse(damped,"Ad","A"),
                                     "M"=ifelse(damped,"Md","M"),
                                     "X"=c("N","A","Ad"),
                                     "Y"=c("N","M","Md"));
            }

            # If Stype is not Z, then crete specific pools
            if(Stype!="Z"){
                poolSeasonals <- switch(Stype,
                                        "N"="N",
                                        "A"="A",
                                        "X"=c("N","A"),
                                        "M"="M",
                                        "Y"=c("N","M"));
            }

            modelsPool <- paste0(rep(poolErrors,length(poolTrends)*length(poolSeasonals)),
                                 rep(poolTrends,each=length(poolSeasonals)),
                                 rep(poolSeasonals,length(poolTrends)));
        }

        adamSelected <-  selector(model, modelsPool, allowMultiplicative,
                                  etsModel, Etype, Ttype, Stype, damped, lags,
                                  lagsModelSeasonal, lagsModelARIMA,
                                  obsStates, obsInSample,
                                  yInSample, persistence, persistenceEstimate,
                                  persistenceLevel, persistenceLevelEstimate,
                                  persistenceTrend, persistenceTrendEstimate,
                                  persistenceSeasonal, persistenceSeasonalEstimate,
                                  persistenceXreg, persistenceXregEstimate, persistenceXregProvided,
                                  phi, phiEstimate,
                                  initialType, initialLevel, initialTrend, initialSeasonal,
                                  initialArima, initialEstimate,
                                  initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                                  initialArimaEstimate, initialXregEstimate, initialXregProvided,
                                  arimaModel, arRequired, iRequired, maRequired, armaParameters,
                                  componentsNumberARIMA, componentsNamesARIMA,
                                  xregModel, xregModelInitials, xregData, xregNumber, xregNames, regressors,
                                  xregParametersMissing, xregParametersIncluded,
                                  xregParametersEstimated, xregParametersPersistence,
                                  constantRequired, constantEstimate, constantValue, constantName,
                                  ot, otLogical, occurrenceModel, pFitted, icFunction,
                                  bounds, loss, lossFunction, distribution,
                                  horizon, multisteps, other, otherParameterEstimate, lambda);

        icSelection <- adamSelected$icSelection;

        icBest <- min(icSelection);
        adamSelected$icWeights  <- (exp(-0.5*(icSelection-icBest)) /
                                        sum(exp(-0.5*(icSelection-icBest))));

        # This is a failsafe mechanism, just to make sure that the ridiculous models don't impact forecasts
        adamSelected$icWeights[adamSelected$icWeights<1e-5] <- 0
        adamSelected$icWeights <- adamSelected$icWeights/sum(adamSelected$icWeights);

        # adamArchitect <- vector("list",10)
        for(i in 1:length(adamSelected$results)){
            # Take the parameters of the best model
            list2env(adamSelected$results[[i]], environment());

            #### This part is needed in order for the filler to do its job later on
            # Create the basic variables based on the estimated model
            adamArchitect <- architector(etsModel, Etype, Ttype, Stype, lags, lagsModelSeasonal,
                                         xregNumber, obsInSample, initialType,
                                         arimaModel, lagsModelARIMA, xregModel, constantRequired,
                                         profilesRecentTable, profilesRecentProvided);
            list2env(adamArchitect, environment());

            adamSelected$results[[i]]$modelIsTrendy <- adamArchitect$modelIsTrendy;
            adamSelected$results[[i]]$modelIsSeasonal <- adamArchitect$modelIsSeasonal;
            adamSelected$results[[i]]$lagsModel <- adamArchitect$lagsModel;
            adamSelected$results[[i]]$lagsModelAll <- adamArchitect$lagsModelAll;
            adamSelected$results[[i]]$lagsModelMax <- adamArchitect$lagsModelMax;
            adamSelected$results[[i]]$profilesRecentTable <- adamArchitect$profilesRecentTable;
            adamSelected$results[[i]]$indexLookupTable <- adamArchitect$indexLookupTable;
            adamSelected$results[[i]]$componentsNumberETS <- adamArchitect$componentsNumberETS;
            adamSelected$results[[i]]$componentsNumberETSSeasonal <- adamArchitect$componentsNumberETSSeasonal;
            adamSelected$results[[i]]$componentsNumberETSNonSeasonal <- adamArchitect$componentsNumberETSNonSeasonal;
            adamSelected$results[[i]]$componentsNamesETS <- adamArchitect$componentsNamesETS;
            adamSelected$results[[i]]$adamCpp <- adamArchitect$adamCpp;

            # Create the matrices for the specific ETS model
            adamCreated <- creator(etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal,
                                   lags, lagsModel, lagsModelARIMA, lagsModelAll, lagsModelMax,
                                   profilesRecentTable, profilesRecentProvided,
                                   obsStates, obsInSample, obsAll, componentsNumberETS, componentsNumberETSSeasonal,
                                   componentsNamesETS, otLogical, yInSample,
                                   persistence, persistenceEstimate,
                                   persistenceLevel, persistenceLevelEstimate, persistenceTrend, persistenceTrendEstimate,
                                   persistenceSeasonal, persistenceSeasonalEstimate,
                                   persistenceXreg, persistenceXregEstimate, persistenceXregProvided,
                                   phi,
                                   initialType, initialEstimate,
                                   initialLevel, initialLevelEstimate, initialTrend, initialTrendEstimate,
                                   initialSeasonal, initialSeasonalEstimate,
                                   initialArima, initialArimaEstimate, initialArimaNumber,
                                   initialXregEstimate, initialXregProvided,
                                   arimaModel, arRequired, iRequired, maRequired, armaParameters,
                                   arOrders, iOrders, maOrders,
                                   componentsNumberARIMA, componentsNamesARIMA,
                                   xregModel, xregModelInitials, xregData, xregNumber, xregNames,
                                   xregParametersPersistence,
                                   constantRequired, constantEstimate, constantValue, constantName,
                                   adamCpp);

            adamSelected$results[[i]]$matVt <- adamCreated$matVt;
            adamSelected$results[[i]]$matWt <- adamCreated$matWt;
            adamSelected$results[[i]]$matF <- adamCreated$matF;
            adamSelected$results[[i]]$vecG <- adamCreated$vecG;
            adamSelected$results[[i]]$arimaPolynomials <- adamCreated$arimaPolynomials;

            parametersNumber[1,1] <- adamSelected$results[[i]]$nParamEstimated;
            if(xregModel){
                parametersNumber[1,2] <- xregNumber*initialXregEstimate + xregNumber*persistenceXregEstimate;
            }
            # If we used likelihood, scale was estimated
            if((loss=="likelihood")){
                parametersNumber[1,4] <- 1;
            }
            parametersNumber[1,5] <- sum(parametersNumber[1,1:4]);
            parametersNumber[2,5] <- sum(parametersNumber[2,1:4]);

            adamSelected$results[[i]]$parametersNumber <- parametersNumber;
        }
    }
    #### Use the provided model ####
    else if(modelDo=="use"){
        # Remove bounds. The user knows what they are doing... probably
        bounds <- "none";

        # If the distribution is default, change it according to the error term
        if(distribution=="default"){
            distributionNew <- switch(loss,
                                      "likelihood"= switch(Etype, "A"= "dnorm", "M"= "dgamma"),
                                      "MAEh"=, "MACE"=, "MAE"= "dlaplace",
                                      "HAMh"=, "CHAM"=, "HAM"= "ds",
                                      "MSEh"=, "MSCE"=, "MSE"=, "GPL"=, "dnorm");
        }
        else{
            distributionNew <- distribution;
        }

        # Create the basic variables
        adamArchitect <- architector(etsModel, Etype, Ttype, Stype, lags, lagsModelSeasonal,
                                     xregNumber, obsInSample, initialType,
                                     arimaModel, lagsModelARIMA, xregModel, constantRequired,
                                     profilesRecentTable, profilesRecentProvided);
        list2env(adamArchitect, environment());

        # Create the matrices for the specific ETS model
        adamCreated <- creator(etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal,
                               lags, lagsModel, lagsModelARIMA, lagsModelAll, lagsModelMax,
                               profilesRecentTable, profilesRecentProvided,
                               obsStates, obsInSample, obsAll, componentsNumberETS, componentsNumberETSSeasonal,
                               componentsNamesETS, otLogical, yInSample,
                               persistence, persistenceEstimate,
                               persistenceLevel, persistenceLevelEstimate, persistenceTrend, persistenceTrendEstimate,
                               persistenceSeasonal, persistenceSeasonalEstimate,
                               persistenceXreg, persistenceXregEstimate, persistenceXregProvided,
                               phi,
                               initialType, initialEstimate,
                               initialLevel, initialLevelEstimate, initialTrend, initialTrendEstimate,
                               initialSeasonal, initialSeasonalEstimate,
                               initialArima, initialArimaEstimate, initialArimaNumber,
                               initialXregEstimate, initialXregProvided,
                               arimaModel, arRequired, iRequired, maRequired, armaParameters,
                               arOrders, iOrders, maOrders,
                               componentsNumberARIMA, componentsNamesARIMA,
                               xregModel, xregModelInitials, xregData, xregNumber, xregNames,
                               xregParametersPersistence,
                               constantRequired, constantEstimate, constantValue, constantName,
                               adamCpp);
        list2env(adamCreated, environment());

        # Prepare the denominator needed for the shrinkage of explanatory variables in LASSO / RIDGE
        if(xregNumber>0 && any(loss==c("LASSO","RIDGE"))){
            denominator <- apply(matWt, 2, sd);
            denominator[is.infinite(denominator)] <- 1;
            yDenominator <- max(sd(diff(yInSample)),1);
        }
        else{
            denominator <- NULL;
            yDenominator <- NULL;
        }

        CFValue <- CF(B=0, etsModel=etsModel, Etype=Etype, Ttype=Ttype, Stype=Stype, modelIsTrendy=modelIsTrendy,
                      modelIsSeasonal=modelIsSeasonal, yInSample=yInSample,
                      ot=ot, otLogical=otLogical, occurrenceModel=occurrenceModel, obsInSample=obsInSample,
                      componentsNumberETS=componentsNumberETS, componentsNumberETSSeasonal=componentsNumberETSSeasonal,
                      componentsNumberETSNonSeasonal=componentsNumberETSNonSeasonal,
                      componentsNumberARIMA=componentsNumberARIMA,
                      lags=lags, lagsModel=lagsModel, lagsModelAll=lagsModelAll, lagsModelMax=lagsModelMax,
                      indexLookupTable=indexLookupTable, profilesRecentTable=profilesRecentTable,
                      matVt=matVt, matWt=matWt, matF=matF, vecG=vecG,
                      persistenceEstimate=persistenceEstimate,
                      persistenceLevelEstimate=persistenceLevelEstimate,
                      persistenceTrendEstimate=persistenceTrendEstimate,
                      persistenceSeasonalEstimate=persistenceSeasonalEstimate,
                      persistenceXregEstimate=persistenceXregEstimate,
                      phiEstimate=phiEstimate, initialType=initialType,
                      initialEstimate=initialEstimate, initialLevelEstimate=initialLevelEstimate,
                      initialTrendEstimate=initialTrendEstimate, initialSeasonalEstimate=initialSeasonalEstimate,
                      initialArimaEstimate=initialArimaEstimate, initialXregEstimate=initialXregEstimate,
                      arimaModel=arimaModel, nonZeroARI=nonZeroARI, nonZeroMA=nonZeroMA,
                      arimaPolynomials=arimaPolynomials,
                      arEstimate=arEstimate, maEstimate=maEstimate,
                      arOrders=arOrders, iOrders=iOrders, maOrders=maOrders,
                      arRequired=arRequired, maRequired=maRequired, armaParameters=armaParameters,
                      xregModel=xregModel, xregNumber=xregNumber,
                      xregParametersMissing=xregParametersMissing,
                      xregParametersIncluded=xregParametersIncluded,
                      xregParametersEstimated=xregParametersEstimated,
                      xregParametersPersistence=xregParametersPersistence,
                      constantRequired=constantRequired, constantEstimate=constantEstimate,
                      bounds=bounds, loss=loss, lossFunction=lossFunction, distribution=distributionNew,
                      horizon=horizon, multisteps=multisteps,
                      denominator=denominator, yDenominator=yDenominator,
                      other=other, otherParameterEstimate=otherParameterEstimate, lambda=lambda,
                      arPolynomialMatrix=NULL, maPolynomialMatrix=NULL,
                      adamCpp);

        parametersNumber[1,1] <- parametersNumber[1,5] <- 1;
        logLikADAMValue <- structure(logLikADAM(B=0,
                                                etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal, yInSample,
                                                ot, otLogical, occurrenceModel, pFitted, obsInSample,
                                                componentsNumberETS, componentsNumberETSSeasonal, componentsNumberETSNonSeasonal,
                                                componentsNumberARIMA,
                                                lags, lagsModel, lagsModelAll, lagsModelMax,
                                                indexLookupTable, profilesRecentTable,
                                                matVt, matWt, matF, vecG,
                                                persistenceEstimate, persistenceLevelEstimate, persistenceTrendEstimate,
                                                persistenceSeasonalEstimate, persistenceXregEstimate,
                                                phiEstimate, initialType, initialEstimate,
                                                initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                                                initialArimaEstimate, initialXregEstimate,
                                                arimaModel, nonZeroARI, nonZeroMA, arEstimate, maEstimate, arimaPolynomials,
                                                arOrders, iOrders, maOrders, arRequired, maRequired, armaParameters,
                                                xregModel, xregNumber,
                                                xregParametersMissing, xregParametersIncluded,
                                                xregParametersEstimated, xregParametersPersistence,
                                                constantRequired, constantEstimate,
                                                bounds, loss, lossFunction, distributionNew, horizon,
                                                multisteps, denominator, yDenominator, other, otherParameterEstimate, lambda,
                                                arPolynomialMatrix=NULL, maPolynomialMatrix=NULL,
                                                adamCpp)
                                     ,nobs=obsInSample,df=parametersNumber[1,5],class="logLik")

        icSelection <- icFunction(logLikADAMValue);
        # If Fisher Information is required, do that analytically
        if(FI){
            # If B is not provided, then use the standard thing
            if(is.null(B)){
                BValues <- initialiser(etsModel, Etype, Ttype, Stype, modelIsTrendy, modelIsSeasonal,
                                       componentsNumberETSNonSeasonal, componentsNumberETSSeasonal, componentsNumberETS,
                                       lags, lagsModel, lagsModelSeasonal, lagsModelARIMA, lagsModelMax,
                                       matVt,
                                       TRUE, TRUE, modelIsTrendy, rep(modelIsSeasonal,componentsNumberETSSeasonal), FALSE,
                                       damped, "optimal", TRUE,
                                       TRUE, TRUE, rep(modelIsSeasonal,componentsNumberETSSeasonal),
                                       arimaModel, xregModel,
                                       arimaModel, arRequired, maRequired, arRequired, maRequired, arOrders, maOrders,
                                       componentsNumberARIMA, componentsNamesARIMA, initialArimaNumber,
                                       xregModel, xregNumber,
                                       xregParametersEstimated, xregParametersPersistence,
                                       constantRequired, constantName, FALSE,
                                       adamCpp);
                # Create the vector of initials for the optimisation
                B <- BValues$B;
            }

            # Reset persistence, just to make sure that there are no duplicates
            vecG[] <- 0;

            initialTypeFI <- initialType;
            # initialTypeFI <- switch(initialType,
            #                         "complete"=,
            #                         "backcasting"="provided",
            #                         initialType);
            initialEstimateFI <- FALSE;
            # Define parameters just for FI calculation
            if(initialTypeFI=="provided"){
                initialLevelEstimateFI <- any(names(B)=="level");
                initialTrendEstimateFI <- any(names(B)=="trend");
                if(any(substr(names(B),1,8)=="seasonal")){
                    initialSeasonalEstimateFI <- vector("logical", componentsNumberETSSeasonal);
                    seasonalNames <- names(B)[substr(names(B),1,8)=="seasonal"];
                    # If there is only one seasonality
                    if(any(substr(seasonalNames,1,9)=="seasonal_")){
                        initialSeasonalEstimateFI[] <- TRUE;
                    }
                    # If there are several
                    else{
                        initialSeasonalEstimateFI[unique(as.numeric(substr(seasonalNames,9,9)))] <- TRUE;
                    }
                }
                else{
                    initialSeasonalEstimateFI <- FALSE;
                }

                if(arimaModel){
                    initialArimaEstimateFI <- any(substr(names(B),1,10)=="ARIMAState");
                }
                else{
                    initialArimaEstimateFI <- FALSE;
                }

                if(xregModel){
                    initialXregEstimateFI <- any(colnames(xregData) %in% names(B));
                }
                else{
                    initialXregEstimateFI <- FALSE;
                }

                initialTypeFI <- "optimal";
                initialEstimateFI <- any(c(initialLevelEstimateFI,initialTrendEstimateFI,initialSeasonalEstimateFI,
                                           initialArimaEstimateFI, initialXregEstimateFI));
            }

            # If smoothing parameters were estimated, then alpha should be in the list
            persistenceLevelEstimateFI <- any(names(B)=="alpha");
            persistenceTrendEstimateFI <- any(names(B)=="beta");
            if(any(substr(names(B),1,5)=="gamma")){
                gammas <- (substr(names(B),1,5)=="gamma");
                if(sum(gammas)==1){
                    persistenceSeasonalEstimateFI <- TRUE;
                }
                else{
                    persistenceSeasonalEstimateFI <- vector("logical",componentsNumberETSSeasonal);
                    persistenceSeasonalEstimateFI[as.numeric(substr(names(B),6,6)[gammas])] <- TRUE;
                }
            }
            else{
                persistenceSeasonalEstimateFI <- FALSE;
            }
            persistenceXregEstimateFI <- any(substr(names(B),1,5)=="delta");
            persistenceEstimateFI <- any(c(persistenceLevelEstimateFI,persistenceTrendEstimateFI,
                                           persistenceSeasonalEstimateFI,persistenceXregEstimateFI));
            phiEstimateFI <- any(names(B)=="phi");
            otherParameterEstimateFI <- any(names(B)=="other");

            # Stuff for the ARIMA elements
            if(arimaModel){
                maEstimateFI <- maRequired;
                arEstimateFI <- arRequired;
                maPolynomialMatrix <- arPolynomialMatrix <- NULL;
            }

            # This is needed in order to avoid the 1e+300 in the CF
            boundsFI <- "none";

            FI <- -hessian(logLikADAM, B, etsModel=etsModel, Etype=Etype, Ttype=Ttype, Stype=Stype, modelIsTrendy=modelIsTrendy,
                           modelIsSeasonal=modelIsSeasonal, yInSample=yInSample,
                           ot=ot, otLogical=otLogical, occurrenceModel=occurrenceModel, pFitted=pFitted, obsInSample=obsInSample,
                           componentsNumberETS=componentsNumberETS, componentsNumberETSSeasonal=componentsNumberETSSeasonal,
                           componentsNumberETSNonSeasonal=componentsNumberETSNonSeasonal,
                           componentsNumberARIMA=componentsNumberARIMA,
                           lags=lags, lagsModel=lagsModel, lagsModelAll=lagsModelAll, lagsModelMax=lagsModelMax,
                           indexLookupTable=indexLookupTable, profilesRecentTable=profilesRecentTable,
                           matVt=matVt, matWt=matWt, matF=matF, vecG=vecG,
                           persistenceEstimate=persistenceEstimateFI, persistenceLevelEstimate=persistenceLevelEstimateFI,
                           persistenceTrendEstimate=persistenceTrendEstimateFI,
                           persistenceSeasonalEstimate=persistenceSeasonalEstimateFI,
                           persistenceXregEstimate=persistenceXregEstimateFI,
                           phiEstimate=phiEstimateFI, initialType=initialTypeFI,
                           initialEstimate=initialEstimateFI, initialLevelEstimate=initialLevelEstimateFI,
                           initialTrendEstimate=initialTrendEstimateFI, initialSeasonalEstimate=initialSeasonalEstimateFI,
                           initialArimaEstimate=initialArimaEstimateFI, initialXregEstimate=initialXregEstimateFI,
                           arimaModel=arimaModel, nonZeroARI=nonZeroARI, nonZeroMA=nonZeroMA,
                           arEstimate=arEstimateFI, maEstimate=maEstimateFI, arimaPolynomials=arimaPolynomials,
                           arOrders=arOrders, iOrders=iOrders, maOrders=maOrders,
                           arRequired=arRequired, maRequired=maRequired, armaParameters=armaParameters,
                           xregModel=xregModel, xregNumber=xregNumber,
                           xregParametersMissing=xregParametersMissing,
                           xregParametersIncluded=xregParametersIncluded,
                           xregParametersEstimated=xregParametersEstimated,
                           xregParametersPersistence=xregParametersPersistence,
                           constantRequired=constantRequired, constantEstimate=constantRequired,
                           bounds=boundsFI, loss=loss, lossFunction=lossFunction, distribution=distribution,
                           horizon=horizon, multisteps=multisteps,
                           denominator=denominator, yDenominator=yDenominator,
                           other=other, otherParameterEstimate=otherParameterEstimateFI, lambda=lambda,
                           arPolynomialMatrix=arPolynomialMatrix, maPolynomialMatrix=maPolynomialMatrix,
                           adamCpp=adamCpp,
                           h=stepSize);

            colnames(FI) <- names(B);
            rownames(FI) <- names(B);
        }
        else{
            FI <- NULL;
        }
        res <- NULL;
    }

    # Transform everything into appropriate classes
    if(any(yClasses=="ts")){
        yInSample <- ts(yInSample,start=yStart, frequency=yFrequency);
        if(holdout){
            yHoldout <- ts(as.matrix(yHoldout), start=yForecastStart, frequency=yFrequency);
        }
    }
    else{
        yInSample <- zoo(yInSample, order.by=yInSampleIndex);
        if(holdout){
            yHoldout <- zoo(as.matrix(yHoldout), order.by=yForecastIndex);
        }
    }

    #### Prepare the return if we didn't combine anything ####
    if(modelDo!="combine"){
        modelReturned <- preparator(B, etsModel, Etype, Ttype, Stype,
                                    lagsModel, lagsModelMax, lagsModelAll,
                                    componentsNumberETS, componentsNumberETSSeasonal,
                                    xregNumber, distribution, loss,
                                    persistenceEstimate, persistenceLevelEstimate, persistenceTrendEstimate,
                                    persistenceSeasonalEstimate, persistenceXregEstimate,
                                    phiEstimate, otherParameterEstimate,
                                    initialType, initialEstimate,
                                    initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                                    initialArimaEstimate, initialXregEstimate,
                                    matVt, matWt, matF, vecG,
                                    occurrenceModel, ot, oesModel,
                                    parametersNumber, CFValue,
                                    arimaModel, arRequired, maRequired,
                                    arEstimate, maEstimate, arOrders, iOrders, maOrders,
                                    nonZeroARI, nonZeroMA,
                                    arimaPolynomials, armaParameters,
                                    constantRequired, constantEstimate, adamCpp);

        # Prepare the name of the model
        modelName <- "";
        if(etsModel){
            if(model!="NNN"){
                modelName[] <- "ETS";
                if(xregModel){
                    modelName[] <- paste0(modelName,"X");
                }
                modelName[] <- paste0(modelName,"(",model,")");
                if(componentsNumberETSSeasonal>1){
                    modelName[] <- paste0(modelName,"[",paste0(lags[lags!=1], collapse=", "),"]");
                }
            }
        }
        if(arimaModel){
            if(etsModel){
                modelName[] <- paste0(modelName,"+");
            }
            # Either the lags are non-seasonal, or there are no orders for seasonal lags
            if(all(lags==1) || (all(arOrders[lags>1]==0) && all(iOrders[lags>1]==0) && all(maOrders[lags>1]==0))){
                modelName[] <- paste0(modelName,"ARIMA");
                if(!etsModel && xregModel){
                    modelName[] <- paste0(modelName,"X");
                }
                modelName[] <- paste0(modelName,"(",arOrders[1],",",iOrders[1],",",maOrders[1],")");
            }
            else{
                modelName[] <- paste0(modelName,"SARIMA");
                if(!etsModel && xregModel){
                    modelName[] <- paste0(modelName,"X");
                }
                for(i in 1:length(arOrders)){
                    if(all(arOrders[i]==0) && all(iOrders[i]==0) && all(maOrders[i]==0)){
                        next;
                    }
                    modelName[] <- paste0(modelName,"(",arOrders[i],",");
                    modelName[] <- paste0(modelName,iOrders[i],",");
                    modelName[] <- paste0(modelName,maOrders[i],")[",lags[i],"]");
                }
            }
        }
        if(regressors=="adapt"){
            modelName[] <- paste0(modelName,"{D}");
        }
        if(!etsModel && !arimaModel){
            if(model=="NNN"){
                modelName[] <- "Constant level";
            }
            else if(regressors=="adapt"){
                modelName[] <- paste0("Dynamic regression");
            }
            else{
                modelName[] <- paste0("Regression");
            }
        }
        else{
            if(constantRequired){
                modelName[] <- paste0(modelName," with ",constantName);
            }
        }
        if(all(occurrence!=c("n","none"))){
            modelName[] <- paste0("i",modelName,
                                  switch(occurrence,
                                         "f"=,"fixed"="[F]",
                                         "d"=,"direct"="[D]",
                                         "o"=,"odds-ratio"="[O]",
                                         "i"=,"invese-odds-ratio"="[I]",
                                         "g"=,"general"="[G]",
                                         ""));
        }

        modelReturned$model <- modelName;
        modelReturned$timeElapsed <- Sys.time()-startTime;
        if(!is.null(xregData) && !is.null(ncol(data))){
            # Remove redundant columns from the data
            modelReturned$data <- data[1:obsInSample,,drop=FALSE];
            if(holdout){
                modelReturned$holdout <- data[obsInSample+c(1:h),,drop=FALSE];
            }
            # Fix the ts class, which is destroyed during subsetting
            if(all(yClasses!="zoo")){
                if(is.data.frame(data)){
                    modelReturned$data[,responseName] <- ts(modelReturned$data[,responseName],
                                                            start=yStart, frequency=yFrequency);
                    if(holdout){
                        modelReturned$holdout[,responseName] <- ts(modelReturned$holdout[,responseName],
                                                                   start=yForecastStart, frequency=yFrequency);
                    }
                }
                else{
                    modelReturned$data <- ts(modelReturned$data, start=yStart, frequency=yFrequency);
                    if(holdout){
                        modelReturned$holdout <- ts(modelReturned$holdout, start=yForecastStart, frequency=yFrequency);
                    }
                }
            }
        }
        else{
            modelReturned$data <- yInSample;
            modelReturned$holdout <- yHoldout;
        }
        if(any(yNAValues)){
            modelReturned$data[yNAValues[1:obsInSample],responseName] <- NA;
            if(holdout && length(yNAValues)==obsAll){
                modelReturned$holdout[yNAValues[-c(1:obsInSample)],responseName] <- NA;
            }
            modelReturned$residuals[yNAValues[1:obsInSample]] <- NA;
        }

        class(modelReturned) <- c("adam","smooth");
    }
    #### Return the combined model ####
    else{
        modelReturned <- list(models=vector("list",length(adamSelected$results)));
        yFittedCombined <- rep(0,obsInSample);
        if(h>0){
            yForecastCombined <- rep(0,h);
        }
        else{
            yForecastCombined <- NA;
        }
        parametersNumberOverall <- parametersNumber;

        for(i in 1:length(adamSelected$results)){
            list2env(adamSelected$results[[i]], environment());
            modelReturned$models[[i]] <- preparator(B, etsModel, Etype, Ttype, Stype,
                                                    lagsModel, lagsModelMax, lagsModelAll,
                                                    componentsNumberETS, componentsNumberETSSeasonal,
                                                    xregNumber, distribution, loss,
                                                    persistenceEstimate, persistenceLevelEstimate, persistenceTrendEstimate,
                                                    persistenceSeasonalEstimate, persistenceXregEstimate,
                                                    phiEstimate, otherParameterEstimate,
                                                    initialType, initialEstimate,
                                                    initialLevelEstimate, initialTrendEstimate, initialSeasonalEstimate,
                                                    initialArimaEstimate, initialXregEstimate,
                                                    matVt, matWt, matF, vecG,
                                                    occurrenceModel, ot, oesModel,
                                                    parametersNumber, CFValue,
                                                    arimaModel, arRequired, maRequired,
                                                    arEstimate, maEstimate, arOrders, iOrders, maOrders,
                                                    nonZeroARI, nonZeroMA,
                                                    arimaPolynomials, armaParameters,
                                                    constantRequired, constantEstimate, adamCpp);
            modelReturned$models[[i]]$fitted[is.na(modelReturned$models[[i]]$fitted)] <- 0;
            yFittedCombined[] <- yFittedCombined + modelReturned$models[[i]]$fitted * adamSelected$icWeights[i];
            if(h>0){
                modelReturned$models[[i]]$forecast[is.na(modelReturned$models[[i]]$forecast)] <- 0;
                yForecastCombined[] <- yForecastCombined + modelReturned$models[[i]]$forecast * adamSelected$icWeights[i];
            }

            # Prepare the name of the model
            modelName <- "";
            if(xregModel){
                modelName[] <- "ETSX";
            }
            else{
                modelName[] <- "ETS";
            }
            modelName[] <- paste0(modelName,"(",model,")");
            if(all(occurrence!=c("n","none"))){
                modelName[] <- paste0("i",modelName);
            }
            if(componentsNumberETSSeasonal>1){
                modelName[] <- paste0(modelName,"[",paste0(lags[lags!=1], collapse=", "),"]");
            }
            if(arimaModel){
                # Either the lags are non-seasonal, or there are no orders for seasonal lags
                if(all(lags==1) || (all(arOrders[lags>1]==0) && all(iOrders[lags>1]==0) && all(maOrders[lags>1]==0))){
                    modelName[] <- paste0(modelName,"+ARIMA(",arOrders[1],",",iOrders[1],",",maOrders[1],")");
                }
                else{
                    modelName[] <- paste0(modelName,"+SARIMA");
                    for(i in 1:length(arOrders)){
                        if(all(arOrders[i]==0) && all(iOrders[i]==0) && all(maOrders[i]==0)){
                            next;
                        }
                        modelName[] <- paste0(modelName,"(",arOrders[i],",");
                        modelName[] <- paste0(modelName,iOrders[i],",");
                        modelName[] <- paste0(modelName,maOrders[i],")[",lags[i],"]");
                    }
                }
            }
            if(!etsModel && !arimaModel){
                if(model=="NNN"){
                    modelName[] <- "Constant level";
                }
                else if(regressors=="adapt"){
                    modelName[] <- paste0("Dynamic regression");
                }
                else{
                    modelName[] <- paste0("Regression");
                }
            }
            else{
                if(constantRequired){
                    modelName[] <- paste0(modelName," with ",constantName);
                }
            }
            if(all(occurrence!=c("n","none"))){
                modelName[] <- paste0("i",modelName);
            }

            modelReturned$models[[i]]$model <- modelName;
            modelReturned$models[[i]]$timeElapsed <- Sys.time()-startTime;
            if(!is.null(xregData) && !is.null(ncol(data))){
                modelReturned$models[[i]]$data <- data[1:obsInSample,,drop=FALSE];
                if(holdout){
                    modelReturned$models[[i]]$holdout <- data[obsInSample+c(1:h),,drop=FALSE];
                }
                # Fix the ts class, which is destroyed during subsetting
                if(all(yClasses!="zoo")){
                    if(is.data.frame(data)){
                        modelReturned$models[[i]]$data[,responseName] <- ts(modelReturned$models[[i]]$data[,responseName],
                                                                            start=yStart, frequency=yFrequency);
                        if(holdout){
                            modelReturned$models[[i]]$holdout[,responseName] <- ts(modelReturned$models[[i]]$holdout[,responseName],
                                                                                   start=yForecastStart, frequency=yFrequency);
                        }
                    }
                    else{
                        modelReturned$models[[i]]$data <- ts(modelReturned$models[[i]]$data, start=yStart, frequency=yFrequency);
                        if(holdout){
                            modelReturned$models[[i]]$holdout <- ts(modelReturned$models[[i]]$holdout, start=yForecastStart, frequency=yFrequency);
                        }
                    }
                }
            }
            else{
                modelReturned$models[[i]]$data <- yInSample;
                modelReturned$models[[i]]$holdout <- yHoldout;
            }
            if(any(yNAValues)){
                modelReturned$models[[i]]$data[yNAValues[1:obsInSample],responseName] <- NA;
                if(holdout && length(yNAValues)==obsAll){
                    modelReturned$models[[i]]$holdout[yNAValues[-c(1:obsInSample)],responseName] <- NA;
                }
                modelReturned$models[[i]]$residuals[yNAValues[1:obsInSample]] <- NA;
            }
            modelReturned$models[[i]]$call <- cl;

            # Amend the call so that each sub-model can be used separately
            modelReturned$models[[i]]$call$model <- model;

            modelReturned$models[[i]]$bounds <- bounds;

            class(modelReturned$models[[i]]) <- c("adam","smooth");
        }

        names(modelReturned$models) <- names(adamSelected$icWeights);

        # Record the original name of the model.
        model[] <- modelOriginal;
        # Prepare the name of the model
        modelName <- "";
        if(xregModel){
            modelName[] <- "ETSX";
        }
        else{
            modelName[] <- "ETS";
        }
        modelName[] <- paste0(modelName,"(",model,")");
        if(all(occurrence!=c("n","none"))){
            modelName[] <- paste0("i",modelName);
        }
        if(componentsNumberETSSeasonal>1){
            modelName[] <- paste0(modelName,"[",paste0(lags[lags!=1], collapse=", "),"]");
        }
        if(arimaModel){
            # Either the lags are non-seasonal, or there are no orders for seasonal lags
            if(all(lags==1) || (all(arOrders[lags>1]==0) && all(iOrders[lags>1]==0) && all(maOrders[lags>1]==0))){
                modelName[] <- paste0(modelName,"+ARIMA(",arOrders[1],",",iOrders[1],",",maOrders[1],")");
            }
            else{
                modelName[] <- paste0(modelName,"+SARIMA");
                for(i in 1:length(arOrders)){
                    if(all(arOrders[i]==0) && all(iOrders[i]==0) && all(maOrders[i]==0)){
                        next;
                    }
                    modelName[] <- paste0(modelName,"(",arOrders[i],",");
                    modelName[] <- paste0(modelName,iOrders[i],",");
                    modelName[] <- paste0(modelName,maOrders[i],")[",lags[i],"]");
                }
            }
        }
        if(all(occurrence!=c("n","none"))){
            modelName[] <- paste0("i",modelName);
        }
        modelReturned$model <- modelName;
        modelReturned$formula <- as.formula(paste0(responseName,"~."));
        modelReturned$timeElapsed <- Sys.time()-startTime;
        if(!is.null(xregDataOriginal)){
            modelReturned$data <- data[1:obsInSample,,drop=FALSE];
            if(holdout){
                modelReturned$holdout <- data[obsInSample+c(1:h),,drop=FALSE];
            }
            # Fix the ts class, which is destroyed during subsetting
            if(all(yClasses!="zoo")){
                if(is.data.frame(data)){
                    modelReturned$data[,responseName] <- ts(modelReturned$data[,responseName],
                                                            start=yStart, frequency=yFrequency);
                    if(holdout){
                        modelReturned$holdout[,responseName] <- ts(modelReturned$holdout[,responseName],
                                                                   start=yForecastStart, frequency=yFrequency);
                    }
                }
                else{
                    modelReturned$data <- ts(modelReturned$data, start=yStart, frequency=yFrequency);
                    if(holdout){
                        modelReturned$holdout <- ts(modelReturned$holdout, start=yForecastStart, frequency=yFrequency);
                    }
                }
            }
        }
        else{
            modelReturned$data <- yInSample;
            modelReturned$holdout <- yHoldout;
        }
        modelReturned$fitted <- ts(yFittedCombined,start=yStart, frequency=yFrequency);
        modelReturned$residuals <- yInSample - yFittedCombined;
        if(any(yNAValues)){
            modelReturned$data[yNAValues[1:obsInSample],responseName] <- NA;
            if(holdout && length(yNAValues)==obsAll){
                modelReturned$holdout[yNAValues[-c(1:obsInSample)],responseName] <- NA;
            }
            modelReturned$residuals[yNAValues[1:obsInSample]] <- NA;
        }
        modelReturned$forecast <- ts(yForecastCombined,start=yForecastStart, frequency=yFrequency);
        # Weighted estimated parameters minus scale
        parametersNumberOverall[1,1] <- (sapply(modelReturned$models, nparam)-(loss=="likelihood")*1) %*% adamSelected$icWeights;
        parametersNumberOverall[1,5] <- sum(parametersNumberOverall[1,1:4]);
        parametersNumberOverall[2,5] <- sum(parametersNumberOverall[2,1:4]);
        modelReturned$nParam <- parametersNumberOverall;
        modelReturned$ICw <- adamSelected$icWeights;
        # These two are needed just to make basic methods work
        modelReturned$distribution <- distribution;
        modelReturned$scale <- sqrt(mean(modelReturned$residuals^2,na.rm=TRUE));
        class(modelReturned) <- c("adamCombined","adam","smooth");
    }
    modelReturned$ICs <- icSelection;
    modelReturned$lossFunction <- lossFunction;
    modelReturned$call <- cl;
    modelReturned$bounds <- bounds;

    # Error measures if there is a holdout
    if(holdout){
        modelReturned$accuracy <- measures(yHoldout,modelReturned$forecast,yInSample);
    }

    if(!silent){
        plot(modelReturned, 7);
    }

    return(modelReturned);
}

#### Small useful ADAM functions ####
# These functions are faster than which() and tail() for vectors are.
# The main gain is in polinomialiser()
# whichFast <- function(x){
#     return(c(1:length(x))[x]);
# }
# tailFast <- function(x,...){
#     return(x[length(x)]) ;
# }

# This function creates recent profile and the lookup table for adam
#' @importFrom greybox detectdst
adamProfileCreator <- function(lagsModelAll, lagsModelMax, obsAll,
                               lags=NULL, yIndex=NULL, yClasses=NULL){
    # lagsModelAll - all lags used in the model for ETS + ARIMA + xreg
    # lagsModelMax - the maximum lag used in the model
    # obsAll - number of observations to create
    # lags - the original lags provided by user (no lags for ARIMA etc). Needed in order to see
    #        if weird frequencies are used.
    # yIndex - the indices needed in order to get the weird dates.
    # yClass - the class used for the actuals. If zoo, magic will happen here.
    # Create the matrix with profiles, based on provided lags
    profilesRecentTable <- matrix(0,length(lagsModelAll),lagsModelMax,
                                  dimnames=list(lagsModelAll,NULL));
    # Create the lookup table
    indexLookupTable <- matrix(1,length(lagsModelAll),obsAll+lagsModelMax,
                               dimnames=list(lagsModelAll,NULL));
    # Modify the lookup table in order to get proper indices in C++
    profileIndices <- matrix(c(1:(lagsModelMax*length(lagsModelAll))),length(lagsModelAll));

    for(i in 1:length(lagsModelAll)){
        profilesRecentTable[i,1:lagsModelAll[i]] <- 1:lagsModelAll[i];
        # -1 is needed to align this with C++ code
        indexLookupTable[i,lagsModelMax+c(1:obsAll)] <- rep(profileIndices[i,1:lagsModelAll[i]],
                                                            ceiling(obsAll/lagsModelAll[i]))[1:obsAll] -1;
        # Fix the head of the data, before the sample starts
        indexLookupTable[i,1:lagsModelMax] <- tail(rep(unique(indexLookupTable[i,lagsModelMax+c(1:obsAll)]),lagsModelMax),
                                                   lagsModelMax);
    }

    # Do shifts for proper lags only:
    # Check lags variable for 24 / 24*7 / 24*365 / 48 / 48*7 / 48*365 / 365 / 52
    # If they are there, find the DST / Leap moments
    # Then amend respective lookup values of profile, shifting them around
    if(any(yClasses=="zoo") && !is.null(yIndex) && !is.numeric(yIndex)){
        # If this is weekly data, duplicate 52, when 53 is used
        if(any(lags==52) && any(strftime(yIndex,format="%W")=="53")){
            shiftRows <- lagsModelAll==52;
            # If the data does not start with 1, proceed
            if(all(which(strftime(yIndex,format="%W")=="53")!=1)){
                indexLookupTable[shiftRows,which(strftime(yIndex,format="%W")=="53")] <-
                    indexLookupTable[shiftRows,which(strftime(yIndex,format="%W")=="53")-1];
            }
        }

        #### If this is daily and we have 365 days of year, locate 29th February and use 28th instead
        if(any(c(365,365*48,365*24) %in% lags) && any(strftime(yIndex,format="%d/%m")=="29/02")){
            shiftValue <- c(365,365*48,365*24)[c(365,365*48,365*24) %in% lags]/365;
            shiftRows <- lagsModelAll %in% c(365,365*48,365*24);
            # If the data does not start with 1/24/48, proceed (otherwise we refer to negative numbers)
            if(!any(which(strftime(yIndex,format="%d/%m")=="29/02") %in% shiftValue)){
                indexLookupTable[shiftRows,which(strftime(yIndex,format="%d/%m")=="29/02")] <-
                    indexLookupTable[shiftRows,which(strftime(yIndex,format="%d/%m")=="29/02")-shiftValue];
            }
        }

        #### If this is hourly; Locate DST and do shifts for specific observations
        if(any(c(24,24*7,24*365,48,48*7,48*365) %in% lags)){
            shiftRows <- lagsModelAll %in% c(24,48,24*7,48*7,24*365,48*365);
            # If this is hourly data, then shift 1 hour. If it is halfhourly, shift 2 hours
            shiftValue <- 1;
            if(any(c(48,48*7,48*365) %in% lags)){
                shiftValue[] <- 2;
            }
            # Get the start and the end of DST
            dstValues <- detectdst(yIndex);
            # If there are DST issues, do something
            doShifts <- !is.null(dstValues) && ((nrow(dstValues$start)!=0) | (nrow(dstValues$end)!=0))
            if(doShifts){
                # If the start date is not positioned before the end, introduce the artificial one
                if(nrow(dstValues$start)==0 ||
                   (nrow(dstValues$end)>0 && dstValues$start$id[1]>dstValues$end$id[1])){
                    dstValues$start <- rbind(data.frame(id=1,date=yIndex[1]),dstValues$start);
                }
                # If the end date is not present or the length of the end is not the same as the start,
                # set the end of series as one
                if(nrow(dstValues$end)==0 ||
                   nrow(dstValues$end)<nrow(dstValues$start)){
                    dstValues$end <- rbind(dstValues$end,data.frame(id=obsAll,date=tail(yIndex,1)));
                }
                # Shift everything from start to end dates by 1 obs forward.
                for(i in 1:nrow(dstValues$start)){
                    # If the end date is natural, just shift
                    if(dstValues$end$id[i]+shiftValue<=obsAll){
                        indexLookupTable[shiftRows,dstValues$start$id[i]:dstValues$end$id[i]] <-
                            indexLookupTable[shiftRows,dstValues$start$id[i]:dstValues$end$id[i]+shiftValue];
                    }
                    # If it isn't, we need to come up with the values for the end of sample
                    else{
                        indexLookupTable[shiftRows,dstValues$start$id[i]:dstValues$end$id[i]] <-
                            indexLookupTable[shiftRows,dstValues$start$id[i]:dstValues$end$id[i]-lagsModelMax+shiftValue];
                    }
                }
            }
        }
    }

    return(list(recent=profilesRecentTable,lookup=indexLookupTable));
}


#### Small technical functions returning types of models and components ####
# Function checks whether the conventional ETS or the ADAM version was used
adamETSChecker <- function(object){
    if(!is.null(object$ets) && object$ets=="adam"){
        adamETS <- TRUE;
    }
    else{
        adamETS <- FALSE;
    }
    return(adamETS);
}

#' @export
modelType.adam <- function(object, ...){
    etsModel <- etsChecker(object);
    if(etsModel){
        modelType <- substring(object$model,
                               unlist(gregexpr("\\(",object$model))+1,
                               unlist(gregexpr("\\)",object$model))-1)[1];
    }
    else{
        modelType <- "NNN";
    }
    return(modelType)
}

#' @export
errorType.adam <- function(object, ...){
    model <- modelType(object);
    if(model=="NNN"){
        return(switch(object$distribution,
                      "dnorm"=,"dlaplace"=,"ds"=,"dgnorm"=,"dlogis"=,"dt"=,"dalaplace"="A",
                      "dlnorm"=,"dllaplace"=,"dls"=,"dlgnorm"=,"dinvgauss"=,"dgamma"="M"));
    }
    else{
        return(substr(model,1,1));
    }
}

trendType <- function(object){
    return(substr(modelType(object),2,2));
}

seasonType <- function(object){
    model <- modelType(object);
    return(substr(model,nchar(model),nchar(model)));
}

#' @export
orders.adam <- function(object, ...){
    return(object$orders);
}

#### Technical methods ####
#' @export
lags.adam <- function(object, ...){
    return(object$lags);
}

#' @rdname plot.smooth
#' @export
plot.adam <- function(x, which=c(1,2,4,6), level=0.95, legend=FALSE,
                      ask=prod(par("mfcol")) < length(which) && dev.interactive(),
                      lowess=TRUE, ...){
    ellipsis <- list(...);

    # Define, whether to wait for the hit of "Enter"
    if(ask){
        devAskNewPage(TRUE);
        on.exit(devAskNewPage(FALSE));
    }

    # Warn if the diagnostis will be done for scale
    if(is.scale(x$scale) && any(which %in% c(2:6,8,9,13,14))){
        message("Note that residuals diagnostics plots are produced for scale model");
    }

    # 1. Fitted vs Actuals values
    plot1 <- function(x, ...){
        ellipsis <- list(...);

        # Get the actuals and the fitted values
        ellipsis$y <- as.vector(actuals(x));
        if(is.occurrence(x)){
            if(any(x$distribution==c("plogis","pnorm")) || x$occurrence!="none"){
                ellipsis$y <- (ellipsis$y!=0)*1;
            }
        }
        ellipsis$x <- as.vector(fitted(x));

        # If this is a mixture model, remove zeroes
        if(is.occurrence(x$occurrence)){
            ellipsis$x <- ellipsis$x[ellipsis$y!=0];
            ellipsis$y <- ellipsis$y[ellipsis$y!=0];
        }

        # Remove NAs
        if(any(is.na(ellipsis$x))){
            ellipsis$y <- ellipsis$y[!is.na(ellipsis$x)];
            ellipsis$x <- ellipsis$x[!is.na(ellipsis$x)];
        }
        if(any(is.na(ellipsis$y))){
            ellipsis$x <- ellipsis$x[!is.na(ellipsis$y)];
            ellipsis$y <- ellipsis$y[!is.na(ellipsis$y)];
        }

        # Title
        if(!any(names(ellipsis)=="main")){
            ellipsis$main <- "Actuals vs Fitted";
        }
        # If type and ylab are not provided, set them...
        if(!any(names(ellipsis)=="type")){
            ellipsis$type <- "p";
        }
        if(!any(names(ellipsis)=="ylab")){
            ellipsis$ylab <- "Actuals";
        }
        if(!any(names(ellipsis)=="xlab")){
            ellipsis$xlab <- "Fitted";
        }
        # xlim and ylim
        if(!any(names(ellipsis)=="xlim")){
            ellipsis$xlim <- range(c(ellipsis$x,ellipsis$y));
        }
        if(!any(names(ellipsis)=="ylim")){
            ellipsis$ylim <- range(c(ellipsis$x,ellipsis$y));
        }

        # Start plotting
        do.call(plot,ellipsis);
        abline(a=0,b=1,col="grey",lwd=2,lty=2)
        if(lowess){
            lines(lowess(ellipsis$x, ellipsis$y), col=2);
        }
    }

    # 2 and 3: Standardised  / studentised residuals vs Fitted
    plot2 <- function(x, type="rstandard", ...){
        ellipsis <- list(...);

        # Amend to do analysis of residuals of scale model
        if(is.scale(x$scale)){
            x <- x$scale;
        }

        ellipsis$x <- as.vector(fitted(x));
        if(type=="rstandard"){
            ellipsis$y <- as.vector(rstandard(x));
            yName <- "Standardised";
        }
        else{
            ellipsis$y <- as.vector(rstudent(x));
            yName <- "Studentised";
        }

        if(is.occurrence(x$occurrence)){
            ellipsis$x <- ellipsis$x[actuals(x$occurrence)!=0];
            ellipsis$y <- ellipsis$y[actuals(x$occurrence)!=0];
        }

        # Remove NAs
        if(any(is.na(ellipsis$x))){
            ellipsis$x <- ellipsis$x[!is.na(ellipsis$x)];
            ellipsis$y <- ellipsis$y[!is.na(ellipsis$y)];
        }

        # Main, labs etc
        if(!any(names(ellipsis)=="main")){
            if(any(x$distribution==c("dinvgauss","dgamma","dlnorm","dllaplace","dls","dlgnorm"))){
                ellipsis$main <- paste0("log(",yName," Residuals) vs Fitted");
            }
            else{
                ellipsis$main <- paste0(yName," Residuals vs Fitted");
            }
        }

        if(!any(names(ellipsis)=="xlab")){
            ellipsis$xlab <- "Fitted";
        }
        if(!any(names(ellipsis)=="ylab")){
            ellipsis$ylab <- paste0(yName," Residuals");
        }

        if(legend){
            if(ellipsis$x[length(ellipsis$x)]>mean(ellipsis$x)){
                legendPosition <- "bottomright";
            }
            else{
                legendPosition <- "topright";
            }
        }

        # Get the IDs of outliers and statistic
        outliers <- outlierdummy(x, level=level, type=type);
        statistic <- outliers$statistic;

        # Analyse stuff in logarithms if the error is multiplicative
        if(any(x$distribution==c("dinvgauss","dgamma"))){
            ellipsis$y[] <- log(ellipsis$y);
            statistic <- log(statistic);
        }
        else if(any(x$distribution==c("dlnorm","dllaplace","dls","dlgnorm"))){
            ellipsis$y[] <- log(ellipsis$y);
        }
        outliers <- which(ellipsis$y >statistic[2] | ellipsis$y <statistic[1]);
        # cat(paste0(round(length(outliers)/length(ellipsis$y),3)*100,"% of values are outside the bounds\n"));


        if(!any(names(ellipsis)=="ylim")){
            ellipsis$ylim <- range(c(ellipsis$y,statistic), na.rm=TRUE)*1.2;
            if(legend){
                if(legendPosition=="bottomright"){
                    ellipsis$ylim[1] <- ellipsis$ylim[1] - 0.2*diff(ellipsis$ylim);
                }
                else{
                    ellipsis$ylim[2] <- ellipsis$ylim[2] + 0.2*diff(ellipsis$ylim);
                }
            }
        }

        xRange <- range(ellipsis$x, na.rm=TRUE);
        xRange[1] <- xRange[1] - sd(ellipsis$x, na.rm=TRUE);
        xRange[2] <- xRange[2] + sd(ellipsis$x, na.rm=TRUE);

        do.call(plot,ellipsis);
        abline(h=0, col="grey", lty=2);
        polygon(c(xRange,rev(xRange)),c(statistic[1],statistic[1],statistic[2],statistic[2]),
                col="lightgrey", border=NA, density=10);
        abline(h=statistic, col=2, lty=2);
        if(length(outliers)>0){
            points(ellipsis$x[outliers], ellipsis$y[outliers], pch=16);
            text(ellipsis$x[outliers], ellipsis$y[outliers], labels=outliers, pos=(ellipsis$y[outliers]>0)*2+1);
        }
        if(lowess){
            lines(lowess(ellipsis$x[!is.na(ellipsis$y)], ellipsis$y[!is.na(ellipsis$y)]), col=2);
        }

        if(legend){
            if(lowess){
                legend(legendPosition,
                       legend=c(paste0(round(level,3)*100,"% bounds"),"outside the bounds","LOWESS line"),
                       col=c(2, 1, 2), lwd=c(1,NA,1), lty=c(2,1,1), pch=c(NA,16,NA));
            }
            else{
                legend(legendPosition,
                       legend=c(paste0(round(level,3)*100,"% bounds"),"outside the bounds"),
                       col=c(2, 1), lwd=c(1,NA), lty=c(2,1), pch=c(NA,16));
            }
        }
    }

    # 4 and 5. Fitted vs |Residuals| or Fitted vs Residuals^2
    plot3 <- function(x, type="abs", ...){
        ellipsis <- list(...);

        # Amend to do analysis of residuals of scale model
        if(is.scale(x$scale)){
            x <- x$scale;
        }

        ellipsis$x <- as.vector(fitted(x));
        ellipsis$y <- as.vector(residuals(x));
        if(any(x$distribution==c("dinvgauss","dlnorm","dllaplace","dls","dlgnorm"))){
            ellipsis$y[] <- log(ellipsis$y);
        }
        if(type=="abs"){
            ellipsis$y[] <- abs(ellipsis$y);
        }
        else{
            ellipsis$y[] <- as.vector(ellipsis$y)^2;
        }

        if(is.occurrence(x$occurrence)){
            ellipsis$x <- ellipsis$x[ellipsis$y!=0];
            ellipsis$y <- ellipsis$y[ellipsis$y!=0];
        }
        # Remove NAs
        if(any(is.na(ellipsis$x))){
            ellipsis$x <- ellipsis$x[!is.na(ellipsis$x)];
            ellipsis$y <- ellipsis$y[!is.na(ellipsis$y)];
        }

        if(!any(names(ellipsis)=="main")){
            if(type=="abs"){
                if(any(x$distribution==c("dinvgauss","dlnorm","dllaplace","dls","dlgnorm"))){
                    ellipsis$main <- "|log(Residuals)| vs Fitted";
                }
                else{
                    ellipsis$main <- "|Residuals| vs Fitted";
                }
            }
            else{
                if(any(x$distribution==c("dinvgauss","dlnorm","dllaplace","dls","dlgnorm"))){
                    ellipsis$main <- "log(Residuals)^2 vs Fitted";
                }
                else{
                    ellipsis$main <- "Residuals^2 vs Fitted";
                }
            }
        }

        if(!any(names(ellipsis)=="xlab")){
            ellipsis$xlab <- "Fitted";
        }
        if(!any(names(ellipsis)=="ylab")){
            if(type=="abs"){
                ellipsis$ylab <- "|Residuals|";
            }
            else{
                ellipsis$ylab <- "Residuals^2";
            }
        }

        do.call(plot,ellipsis);
        abline(h=0, col="grey", lty=2);
        if(lowess){
            lines(lowess(ellipsis$x[!is.na(ellipsis$y)], ellipsis$y[!is.na(ellipsis$y)]), col=2);
        }
    }

    # 6. Q-Q with the specified distribution
    plot4 <- function(x, ...){
        ellipsis <- list(...);

        # Amend to do analysis of residuals of scale model
        if(is.scale(x$scale)){
            x <- x$scale;
        }

        ellipsis$y <- as.vector(residuals(x));
        if(is.occurrence(x$occurrence)){
            ellipsis$y <- ellipsis$y[actuals(x$occurrence)!=0];
        }

        if(!any(names(ellipsis)=="xlab")){
            ellipsis$xlab <- "Theoretical Quantile";
        }
        if(!any(names(ellipsis)=="ylab")){
            ellipsis$ylab <- "Actual Quantile";
        }

        if(any(x$distribution=="dnorm")){
            if(!any(names(ellipsis)=="main")){
                ellipsis$main <- "QQ plot of Normal distribution";
            }

            do.call(qqnorm, ellipsis);
            qqline(ellipsis$y);
        }
        else if(any(x$distribution=="dlnorm")){
            if(!any(names(ellipsis)=="main")){
                ellipsis$main <- "QQ plot of Log-Normal distribution";
            }
            ellipsis$x <- qlnorm(ppoints(500), meanlog=-extractScale(x)^2/2, sdlog=extractScale(x));

            do.call(qqplot, ellipsis);
            qqline(ellipsis$y, distribution=function(p) qlnorm(p, meanlog=-extractScale(x)^2/2, sdlog=extractScale(x)));
        }
        else if(x$distribution=="dlaplace"){
            if(!any(names(ellipsis)=="main")){
                ellipsis$main <- "QQ-plot of Laplace distribution";
            }
            ellipsis$x <- qlaplace(ppoints(500), mu=0, scale=extractScale(x));

            do.call(qqplot, ellipsis);
            qqline(ellipsis$y, distribution=function(p) qlaplace(p, mu=0, scale=extractScale(x)));
        }
        else if(x$distribution=="dllaplace"){
            if(!any(names(ellipsis)=="main")){
                ellipsis$main <- "QQ-plot of Log-Laplace distribution";
            }
            ellipsis$x <- exp(qlaplace(ppoints(500), mu=0, scale=extractScale(x)));

            do.call(qqplot, ellipsis);
            qqline(ellipsis$y, distribution=function(p) exp(qlaplace(p, mu=0, scale=extractScale(x))));
        }
        else if(x$distribution=="ds"){
            if(!any(names(ellipsis)=="main")){
                ellipsis$main <- "QQ-plot of S distribution";
            }
            ellipsis$x <- qs(ppoints(500), mu=0, scale=extractScale(x));

            do.call(qqplot, ellipsis);
            qqline(ellipsis$y, distribution=function(p) qs(p, mu=0, scale=extractScale(x)));
        }
        else if(x$distribution=="dls"){
            if(!any(names(ellipsis)=="main")){
                ellipsis$main <- "QQ-plot of Log-S distribution";
            }
            ellipsis$x <- exp(qs(ppoints(500), mu=0, scale=extractScale(x)));

            do.call(qqplot, ellipsis);
            qqline(ellipsis$y, distribution=function(p) exp(qs(p, mu=0, scale=extractScale(x))));
        }
        else if(x$distribution=="dgnorm"){
            if(!any(names(ellipsis)=="main")){
                ellipsis$main <- paste0("QQ-plot of Generalised Normal distribution with shape=",round(x$other$shape,3));
            }
            ellipsis$x <- qgnorm(ppoints(500), mu=0, scale=extractScale(x), shape=x$other$shape);

            do.call(qqplot, ellipsis);
            qqline(ellipsis$y, distribution=function(p) qgnorm(p, mu=0, scale=extractScale(x), shape=x$other$shape));
        }
        else if(x$distribution=="dlgnorm"){
            if(!any(names(ellipsis)=="main")){
                ellipsis$main <- paste0("QQ-plot of Log-Generalised Normal distribution with shape=",round(x$other$shape,3));
            }
            ellipsis$x <- exp(qgnorm(ppoints(500), mu=0, scale=extractScale(x), shape=x$other$shape));

            do.call(qqplot, ellipsis);
            qqline(ellipsis$y, distribution=function(p) exp(qgnorm(p, mu=0, scale=extractScale(x), shape=x$other$shape)));
        }
        else if(x$distribution=="dlogis"){
            if(!any(names(ellipsis)=="main")){
                ellipsis$main <- "QQ-plot of Logistic distribution";
            }
            ellipsis$x <- qlogis(ppoints(500), location=0, scale=extractScale(x));

            do.call(qqplot, ellipsis);
            qqline(ellipsis$y, distribution=function(p) qlogis(p, location=0, scale=extractScale(x)));
        }
        else if(x$distribution=="dt"){
            # Standardise residuals
            ellipsis$y[] <- ellipsis$y / sd(ellipsis$y);
            if(!any(names(ellipsis)=="main")){
                ellipsis$main <- "QQ-plot of Student's distribution";
            }
            ellipsis$x <- qt(ppoints(500), df=x$other$nu);

            do.call(qqplot, ellipsis);
            qqline(ellipsis$y, distribution=function(p) qt(p, df=x$other$nu));
        }
        else if(x$distribution=="dalaplace"){
            if(!any(names(ellipsis)=="main")){
                ellipsis$main <- paste0("QQ-plot of Asymmetric Laplace with alpha=",round(x$other$alpha,3));
            }
            ellipsis$x <- qalaplace(ppoints(500), mu=0, scale=extractScale(x), alpha=x$other$alpha);

            do.call(qqplot, ellipsis);
            qqline(ellipsis$y, distribution=function(p) qalaplace(p, mu=0, scale=extractScale(x), alpha=x$other$alpha));
        }
        else if(x$distribution=="dinvgauss"){
            if(is.scale(x)){
                # Transform residuals for something meaningful
                # This is not 100% accurate, because the dispersion should change as well as mean...
                if(!any(names(ellipsis)=="main")){
                    ellipsis$main <- "QQ-plot of Chi-Squared distribution";
                }
                ellipsis$x <- qchisq(ppoints(500), df=1);

                do.call(qqplot, ellipsis);
                qqline(ellipsis$y, distribution=function(p) qchisq(p, df=1));
            }
            else{
                # Transform residuals for something meaningful
                # This is not 100% accurate, because the dispersion should change as well as mean...
                if(!any(names(ellipsis)=="main")){
                    ellipsis$main <- "QQ-plot of Inverse Gaussian distribution";
                }
                ellipsis$x <- qinvgauss(ppoints(500), mean=1, dispersion=extractScale(x));

                do.call(qqplot, ellipsis);
                qqline(ellipsis$y, distribution=function(p) qinvgauss(p, mean=1, dispersion=extractScale(x)));
            }
        }
        else if(x$distribution=="dgamma"){
            # Transform residuals for something meaningful
            # This is not 100% accurate, because the dispersion should change as well as mean...
            if(!any(names(ellipsis)=="main")){
                ellipsis$main <- "QQ-plot of Gamma distribution";
            }
            ellipsis$x <- qgamma(ppoints(500), shape=1/extractScale(x), scale=extractScale(x));

            do.call(qqplot, ellipsis);
            qqline(ellipsis$y, distribution=function(p) qgamma(p, shape=1/extractScale(x), scale=extractScale(x)));
        }
    }

    # 7. Basic plot over time
    plot5 <- function(x, ...){
        ellipsis <- list(...);

        ellipsis$fitted <- fitted(x);
        ellipsis$actuals <- actuals(x);
        if(!is.null(x$holdout)){
            responseName <- all.vars(formula(x))[1];
            yHoldout <- x$holdout[,responseName]
            if(inherits(yHoldout,"tbl_df") || inherits(yHoldout,"tbl")){
                yHoldout <- yHoldout[[1]];
            }
            if(is.zoo(ellipsis$fitted)){
                ellipsis$actuals <- zoo(c(as.vector(ellipsis$actuals),as.vector(yHoldout)),
                                        order.by=c(time(ellipsis$fitted),time(yHoldout)));
            }
            else{
                ellipsis$actuals <- ts(c(as.vector(ellipsis$actuals),as.vector(yHoldout)),
                                       start=start(ellipsis$fitted),
                                       frequency=frequency(ellipsis$fitted));
            }
        }
        # Reclass the actuals just in case
        else{
            if(is.zoo(ellipsis$fitted)){
                ellipsis$actuals <- zoo(as.vector(ellipsis$actuals),
                                        order.by=time(ellipsis$fitted));
            }
            else{
                ellipsis$actuals <- ts(ellipsis$actuals,
                                       start=start(ellipsis$fitted),
                                       frequency=frequency(ellipsis$fitted));
            }
        }
        if(is.null(ellipsis$main)){
            ellipsis$main <- x$model;
        }
        ellipsis$forecast <- x$forecast;
        ellipsis$legend <- FALSE;
        ellipsis$parReset <- FALSE;

        do.call(graphmaker, ellipsis);
    }

    # 8 and 9. Standardised / Studentised residuals vs time
    plot6 <- function(x, type="rstandard", ...){

        # Amend to do analysis of residuals of scale model
        if(is.scale(x$scale)){
            x <- x$scale;
        }

        ellipsis <- list(...);
        if(type=="rstandard"){
            ellipsis$x <- rstandard(x);
            yName <- "Standardised";
        }
        else{
            ellipsis$x <- rstudent(x);
            yName <- "Studentised";
        }

        # If there is occurrence part, substitute zeroes with NAs
        if(is.occurrence(x$occurrence)){
            ellipsis$x[actuals(x$occurrence)==0] <- NA;
        }

        # Main, labs etc
        if(!any(names(ellipsis)=="main")){
            if(any(x$distribution==c("dinvgauss","dgamma","dlnorm","dllaplace","dls","dlgnorm"))){
                ellipsis$main <- paste0("log(",yName," Residuals) vs Time");
            }
            else{
                ellipsis$main <- paste0(yName," Residuals vs Time");
            }
        }

        if(!any(names(ellipsis)=="xlab")){
            ellipsis$xlab <- "Time";
        }
        if(!any(names(ellipsis)=="ylab")){
            ellipsis$ylab <- paste0(yName," Residuals");
        }

        # If type and ylab are not provided, set them...
        if(!any(names(ellipsis)=="type")){
            ellipsis$type <- "l";
        }

        # Get the IDs of outliers and statistic
        outliers <- outlierdummy(x, level=level, type=type);
        statistic <- outliers$statistic;

        # Analyse stuff in logarithms if the error is multiplicative
        if(any(x$distribution==c("dinvgauss","dgamma"))){
            ellipsis$x[] <- log(ellipsis$x);
            statistic <- log(statistic);
        }
        else if(any(x$distribution==c("dlnorm","dllaplace","dls","dlgnorm"))){
            ellipsis$x[] <- log(ellipsis$x);
        }
        outliers <- which(ellipsis$x >statistic[2] | ellipsis$x <statistic[1]);


        if(!any(names(ellipsis)=="ylim")){
            ellipsis$ylim <- c(-max(abs(ellipsis$x),na.rm=TRUE),
                               max(abs(ellipsis$x),na.rm=TRUE))*1.2;
        }

        if(legend){
            legendPosition <- "topright";
            ellipsis$ylim[2] <- ellipsis$ylim[2] + 0.2*diff(ellipsis$ylim);
            ellipsis$ylim[1] <- ellipsis$ylim[1] - 0.2*diff(ellipsis$ylim);
        }

        # Start plotting
        do.call(plot,ellipsis);
        if(length(outliers)>0){
            points(time(ellipsis$x)[outliers], ellipsis$x[outliers], pch=16);
            text(time(ellipsis$x)[outliers], ellipsis$x[outliers], labels=outliers, pos=(ellipsis$x[outliers]>0)*2+1);
        }
        # If there is occurrence model, plot points to fill in breaks
        if(is.occurrence(x$occurrence)){
            points(time(ellipsis$x), ellipsis$x);
        }
        if(lowess){
            # Substitute NAs with the mean
            if(any(is.na(ellipsis$x))){
                ellipsis$x[is.na(ellipsis$x)] <- mean(ellipsis$x, na.rm=TRUE);
            }
            lines(lowess(c(1:length(ellipsis$x)),ellipsis$x), col=2);
        }
        abline(h=0, col="grey", lty=2);
        abline(h=statistic[1], col=2, lty=2);
        abline(h=statistic[2], col=2, lty=2);
        polygon(c(1:nobs(x), c(nobs(x):1)),
                c(rep(statistic[1],nobs(x)), rep(statistic[2],nobs(x))),
                col="lightgrey", border=NA, density=10);
        if(legend){
            legend(legendPosition,legend=c("Residuals",paste0(level*100,"% prediction interval")),
                   col=c(1,2), lwd=rep(1,3), lty=c(1,1,2));
        }
    }

    # 10 and 11. ACF and PACF
    plot7 <- function(x, type="acf", squared=FALSE, ...){
        ellipsis <- list(...);

        if(!any(names(ellipsis)=="main")){
            if(type=="acf"){
                if(squared){
                    ellipsis$main <- "Autocorrelation Function of Squared Residuals";
                }
                else{
                    ellipsis$main <- "Autocorrelation Function of Residuals";
                }
            }
            else{
                if(squared){
                    ellipsis$main <- "Partial Autocorrelation Function of Squared Residuals";
                }
                else{
                    ellipsis$main <- "Partial Autocorrelation Function of Residuals";
                }
            }
        }

        if(!any(names(ellipsis)=="xlab")){
            ellipsis$xlab <- "Lags";
        }
        if(!any(names(ellipsis)=="ylab")){
            if(type=="acf"){
                ellipsis$ylab <- "ACF";
            }
            else{
                ellipsis$ylab <- "PACF";
            }
        }

        if(!any(names(ellipsis)=="ylim")){
            ellipsis$ylim <- c(-1,1);
        }

        if(squared){
            if(type=="acf"){
                theValues <- acf(as.vector(residuals(x)^2), plot=FALSE, na.action=na.pass);
            }
            else{
                theValues <- pacf(as.vector(residuals(x)^2), plot=FALSE, na.action=na.pass);
            }
        }
        else{
            if(type=="acf"){
                theValues <- acf(as.vector(residuals(x)), plot=FALSE, na.action=na.pass);
            }
            else{
                theValues <- pacf(as.vector(residuals(x)), plot=FALSE, na.action=na.pass);
            }
        }
        ellipsis$x <- switch(type,
                             "acf"=theValues$acf[-1],
                             "pacf"=theValues$acf);
        statistic <- qnorm(c((1-level)/2, (1+level)/2),0,sqrt(1/nobs(x)));

        ellipsis$type <- "h"

        do.call(plot,ellipsis);
        abline(h=0, col=1, lty=1);
        abline(h=statistic, col=2, lty=2);
        if(any(ellipsis$x>statistic[2] | ellipsis$x<statistic[1])){
            outliers <- which(ellipsis$x >statistic[2] | ellipsis$x <statistic[1]);
            points(outliers, ellipsis$x[outliers], pch=16);
            text(outliers, ellipsis$x[outliers], labels=outliers, pos=(ellipsis$x[outliers]>0)*2+1);
        }
    }

    # 12. Plot of states
    plot8 <- function(x, ...){
        parDefault <- par(no.readonly=TRUE);
        on.exit(par(parDefault), add=TRUE);
        if(smoothType(x)=="CES" || any(unlist(gregexpr("C",x$model))==-1)){
            statesNames <- c("actuals",colnames(x$states),"residuals");
            x$states <- cbind(actuals(x),x$states,residuals(x));
            colnames(x$states) <- statesNames;
            # If we have the xts class, remove it. This is a bugfix
            if(inherits(x$states,"xts")){
                class(x$states) <- "zoo";
            }
            if(ncol(x$states)>10){
                message("Too many states. Plotting them on several canvases.");
                if(is.null(ellipsis$main)){
                    ellipsisMain <- NULL;
                }
                else{
                    ellipsisMain <- ellipsis$main;
                }
                nPlots <- ceiling(ncol(x$states)/10);
                for(i in 1:nPlots){
                    if(is.null(ellipsisMain)){
                        ellipsis$main <- paste0("States of ",x$model,", part ",i);
                    }
                    ellipsis$x <- x$states[,(1+(i-1)*10):min(i*10,ncol(x$states)),drop=FALSE];
                    do.call(plot, ellipsis);
                }
            }
            else{
                ellipsis$x <- x$states;
                if(ncol(x$states)<=5){
                    ellipsis$nc <- 1;
                }
                if(is.null(ellipsis$main)){
                    ellipsis$main <- paste0("States of ",x$model);
                }
                do.call(plot, ellipsis);
            }
        }
        else{
            # If we did combinations, we cannot return anything
            message("Combination of models was done. Sorry, but there is nothing to plot.");
        }
    }

    # 13 and 14. Fitted vs (std. Residuals)^2 or Fitted vs |std. Residuals|
    plot9 <- function(x, type="abs", ...){
        ellipsis <- list(...);

        # Amend to do analysis of residuals of scale model
        if(is.scale(x$scale)){
            x <- x$scale;
        }

        ellipsis$x <- as.vector(fitted(x));
        ellipsis$y <- as.vector(rstandard(x));
        if(any(x$distribution==c("dinvgauss","dgamma"))){
            ellipsis$y[] <- log(ellipsis$y);
        }
        if(type=="abs"){
            ellipsis$y[] <- abs(ellipsis$y);
        }
        else{
            ellipsis$y[] <- ellipsis$y^2;
        }

        if(is.occurrence(x$occurrence)){
            ellipsis$x <- ellipsis$x[actuals(x$occurrence)!=0];
            ellipsis$y <- ellipsis$y[actuals(x$occurrence)!=0];
        }
        # Remove NAs
        if(any(is.na(ellipsis$x))){
            ellipsis$x <- ellipsis$x[!is.na(ellipsis$x)];
            ellipsis$y <- ellipsis$y[!is.na(ellipsis$y)];
        }

        if(!any(names(ellipsis)=="main")){
            if(type=="abs"){
                if(any(x$distribution==c("dinvgauss","dgamma","dlnorm","dllaplace","dls","dlgnorm"))){
                    ellipsis$main <- paste0("|log(Standardised Residuals)| vs Fitted");
                }
                else{
                    ellipsis$main <- "|Standardised Residuals| vs Fitted";
                }
            }
            else{
                if(any(x$distribution==c("dinvgauss","dgamma","dlnorm","dllaplace","dls","dlgnorm"))){
                    ellipsis$main <- paste0("log(Standardised Residuals)^2 vs Fitted");
                }
                else{
                    ellipsis$main <- "Standardised Residuals^2 vs Fitted";
                }
            }
        }

        if(!any(names(ellipsis)=="xlab")){
            ellipsis$xlab <- "Fitted";
        }
        if(!any(names(ellipsis)=="ylab")){
            if(type=="abs"){
                if(any(x$distribution==c("dinvgauss","dgamma","dlnorm","dllaplace","dls","dlgnorm"))){
                    ellipsis$ylab <- "|log(Standardised Residuals)|";
                }
                else{
                    ellipsis$ylab <- "|Standardised Residuals|";
                }
            }
            else{
                if(any(x$distribution==c("dinvgauss","dgamma","dlnorm","dllaplace","dls","dlgnorm"))){
                    ellipsis$ylab <- "log(Standardised Residuals)^2";
                }
                else{
                    ellipsis$ylab <- "Standardised Residuals^2";
                }
            }
        }

        do.call(plot,ellipsis);
        abline(h=0, col="grey", lty=2);
        if(lowess){
            lines(lowess(ellipsis$x[!is.na(ellipsis$y)], ellipsis$y[!is.na(ellipsis$y)]), col=2);
        }
    }

    # Do plots
    for(i in which){
        if(any(i==1)){
            plot1(x, ...);
        }
        else if(any(i==2)){
            plot2(x, ...);
        }
        else if(any(i==3)){
            plot2(x, "rstudent", ...);
        }
        else if(any(i==4)){
            plot3(x, ...);
        }
        else if(any(i==5)){
            plot3(x, type="squared", ...);
        }
        else if(any(i==6)){
            plot4(x, ...);
        }
        else if(any(i==7)){
            plot5(x, ...);
        }
        else if(any(i==8)){
            plot6(x, ...);
        }
        else if(any(i==9)){
            plot6(x, "rstudent", ...);
        }
        else if(any(i==10)){
            plot7(x, type="acf", ...);
        }
        else if(any(i==11)){
            plot7(x, type="pacf", ...);
        }
        else if(any(i==12)){
            plot8(x, ...);
        }
        else if(any(i==13)){
            plot9(x, ...);
        }
        else if(any(i==14)){
            plot9(x, type="squared", ...);
        }
        else if(any(i==15)){
            plot7(x, type="acf", squared=TRUE, ...);
        }
        else if(any(i==16)){
            plot7(x, type="pacf", squared=TRUE, ...);
        }
    }
}

#' @export
print.adam <- function(x, digits=4, ...){
    if(is.scale(x)){
        cat("**Scale Model**\n");
    }
    etsModel <- etsChecker(x);
    arimaModel <- arimaChecker(x);
    cesModel <- cesChecker(x);
    sparmaModel <- sparmaChecker(x);
    adamETS <- adamETSChecker(x);

    cat("Time elapsed:",round(as.numeric(x$timeElapsed,units="secs"),2),"seconds");
    # tail all.vars is needed in case smooth::adam() was used
    cat(paste0("\nModel estimated using ",tail(all.vars(x$call[[1]]),1),
               "() function: ",ifelse(adamETS, "ADAM ", ""),x$model));
    cat(paste0("\nWith ", x$initialType, " initialisation"));
    if(is.scale(x$scale)){
        cat("\nScale model estimated with sm():",x$scale$model);
    }

    if(is.occurrence(x$occurrence)){
        occurrence <- switch(x$occurrence$occurrence,
                             "f"=,
                             "fixed"="Fixed probability",
                             "o"=,
                             "odds-ratio"="Odds ratio",
                             "i"=,
                             "inverse-odds-ratio"="Inverse odds ratio",
                             "d"=,
                             "direct"="Direct",
                             "g"=,
                             "general"="General",
                             "p"=,
                             "provided"="Provided by user");
        cat("\nOccurrence model type:",occurrence);
    }

    distrib <- switch(x$distribution,
                      "dnorm" = "Normal",
                      "dlaplace" = "Laplace",
                      "ds" = "S",
                      "dgnorm" = paste0("Generalised Normal with shape=",round(x$other$shape, digits)),
                      "dlogis" = "Logistic",
                      "dt" = paste0("Student t with df=",round(x$other$nu, digits)),
                      "dalaplace" = paste0("Asymmetric Laplace with alpha=",round(x$other$alpha,digits)),
                      "dlnorm" = "Log-Normal",
                      "dllaplace" = "Log-Laplace",
                      "dls" = "Log-S",
                      "dlgnorm" = paste0("Log-Generalised Normal with shape=",round(x$other$shape, digits)),
                      # "dbcnorm" = paste0("Box-Cox Normal with lambda=",round(x$other$lambda,2)),
                      "dinvgauss" = "Inverse Gaussian",
                      "dgamma" = "Gamma"
    );
    if(is.occurrence(x$occurrence)){
        distrib <- paste0("Mixture of Bernoulli and ", distrib);
    }
    cat("\nDistribution assumed in the model:", distrib);

    cat("\nLoss function type:",x$loss);
    if(!is.null(x$lossValue)){
        cat("; Loss function value:",round(x$lossValue,digits));
        if(any(x$loss==c("LASSO","RIDGE"))){
            cat("; lambda=",x$other$lambda);
        }
    }

    # If there is a Intercept/drift
    if(!is.null(x$constant)){
        cat("\nIntercept/Drift value:", round(x$constant, digits));
    }

    if(cesModel){
        if(!is.null(x$parameters)){
            cat("\n");
            print(round(x$parameters$a,digits));
            if(!is.null(x$parameters$b)){
                if(is.complex(x$parameters$b)){
                    print(round(x$parameters$b,digits));
                }
                else{
                    print(round((x$parameters$b),digits));
                }
            }
        }
    }

    if(etsModel){
        if(!is.null(x$persistence)){
            cat("\nPersistence vector g");
            if(ncol(x$data)>1){
                cat(" (excluding xreg):\n");
            }
            else{
                cat(":\n");
            }
            persistence <- x$persistence[substr(names(x$persistence),1,5)!="delta"];
            if(arimaModel){
                persistence <- persistence[substr(names(persistence),1,3)!="psi"];
            }
            # If there is constant, don't include the stuff
            if(!is.null(x$constant)){
                persistence <- persistence[substr(names(persistence),1,8)!="constant"];
                persistence <- persistence[substr(names(persistence),1,8)!="drift"];
            }
            print(round(persistence,digits));
        }

        if(!is.null(x$phi)){
            if(gregexpr("d",modelType(x))!=-1){
                cat("Damping parameter:", round(x$phi,digits));
            }
        }
    }

    # If this is ARIMA model
    if(!sparmaModel &&
       (!is.null(x$arma) && (!is.null(x$arma$ar) || !is.null(x$arma$ma)))){
        ordersModel <- orders(x);
        # If the order was just a vector
        if(!is.list(ordersModel)){
            ordersModel <- list(ar=ordersModel[1], i=ordersModel[2], ma=ordersModel[3]);
        }
        lagsModel <- lags(x);
        cat("\nARMA parameters of the model:\n");
        if(!is.null(x$arma$ar)){
            # cat("AR:\n")
            arMatrix <- matrix(NA,max(ordersModel$ar),length(lagsModel[ordersModel$ar!=0]),
                               dimnames=list(paste0("AR(",1:max(ordersModel$ar),")"),
                                             paste0("Lag ",lagsModel[ordersModel$ar!=0],"")));
            arNumber <- 0;
            # Remove zero lags
            ordersModel$ar <- ordersModel$ar[ordersModel$ar!=0];
            for(i in 1:length(ordersModel$ar)){
                arMatrix[(1:ordersModel$ar[i]),i] <- x$arma$ar[arNumber+(1:ordersModel$ar[i])];
                arNumber <- arNumber + ordersModel$ar[i];
            }
            print(round(arMatrix, digits));
        }
        if(!is.null(x$arma$ma)){
            # cat("MA:\n")
            # print(round(x$arma$ma,digits));
            maMatrix <- matrix(NA,max(ordersModel$ma),length(lagsModel[ordersModel$ma!=0]),
                               dimnames=list(paste0("MA(",1:max(ordersModel$ma),")"),
                                             paste0("Lag ",lagsModel[ordersModel$ma!=0],"")))
            maNumber <- 0;
            ordersModel$ma <- ordersModel$ma[ordersModel$ma!=0];
            for(i in 1:length(ordersModel$ma)){
                maMatrix[(1:ordersModel$ma[i]),i] <- x$arma$ma[maNumber+(1:ordersModel$ma[i])];
                maNumber <- maNumber + ordersModel$ma[i];
            }
            print(round(maMatrix, digits));
        }
    }
    else if(sparmaModel){
        ordersModel <- orders(x);
        if(any(ordersModel$ar!=0)){
            cat(paste0("\nAR(", ordersModel$ar, "): ", round(x$arma$ar, digits)));
        }
        if(any(ordersModel$ma!=0)){
            cat(paste0("\nMA(",ordersModel$ma,"): ", round(x$arma$ma, digits)));
        }
    }

    cat("\nSample size:", nobs(x));
    cat("\nNumber of estimated parameters:", round(nparam(x), digits));
    cat("\nNumber of degrees of freedom:", nobs(x)-round(nparam(x), digits));
    if(x$nParam[2,5]>0){
        cat("\nNumber of provided parameters:", x$nParam[2,5]);
    }

    if(x$loss=="likelihood" ||
       (any(x$loss==c("MSE","MSEh","MSCE","GPL")) & (x$distribution=="dnorm")) ||
       (any(x$loss==c("aMSE","aMSEh","aMSCE","aGPL")) & (x$distribution=="dnorm")) ||
       (any(x$loss==c("MAE","MAEh","MACE")) & (x$distribution=="dlaplace")) ||
       (any(x$loss==c("HAM","HAMh","CHAM")) & (x$distribution=="ds"))){
        ICs <- c(AIC(x),AICc(x),BIC(x),BICc(x));
        names(ICs) <- c("AIC","AICc","BIC","BICc");
        cat("\nInformation criteria:\n");
        print(round(ICs,digits));
    }
    else{
        cat("\nInformation criteria are unavailable for the chosen loss & distribution.\n");
    }

    # If there are accuracy measures, print them out
    if(!is.null(x$accuracy)){
        cat("\nForecast errors:\n");
        if(is.null(x$occurrence)){
            cat(paste(paste0("ME: ",round(x$accuracy["ME"],3)),
                      paste0("MAE: ",round(x$accuracy["MAE"],3)),
                      paste0("RMSE: ",round(sqrt(x$accuracy["MSE"]),3),"\n")
                      ,sep="; "));
            cat(paste(paste0("sCE: ",round(x$accuracy["sCE"],5)*100,"%"),
                      paste0("Asymmetry: ",round(x$accuracy["asymmetry"],3)*100,"%"),
                      paste0("sMAE: ",round(x$accuracy["sMAE"],5)*100,"%"),
                      paste0("sMSE: ",round(x$accuracy["sMSE"],5)*100,"%\n")
                      ,sep="; "));
            cat(paste(paste0("MASE: ",round(x$accuracy["MASE"],3)),
                      paste0("RMSSE: ",round(x$accuracy["RMSSE"],3)),
                      paste0("rMAE: ",round(x$accuracy["rMAE"],3)),
                      paste0("rRMSE: ",round(x$accuracy["rRMSE"],3),"\n")
                      ,sep="; "));
        }
        else{
            cat(paste(paste0("Asymmetry: ",round(x$accuracy["asymmetry"],5)*100,"%"),
                      paste0("sMSE: ",round(x$accuracy["sMSE"],5)*100,"%"),
                      paste0("rRMSE: ",round(x$accuracy["rRMSE"],3)),
                      paste0("sPIS: ",round(x$accuracy["sPIS"],5)*100,"%"),
                      paste0("sCE: ",round(x$accuracy["sCE"],5)*100,"%\n"),sep="; "));
        }
    }
}

#' @export
print.adamCombined <- function(x, digits=4, ...){
    cat("Time elapsed:",round(as.numeric(x$timeElapsed,units="secs"),2),"seconds");
    cat("\nModel estimated:",x$model);
    cat("\nLoss function type:",x$models[[1]]$loss);

    cat("\n\nNumber of models combined:", length(x$ICw));
    cat("\nSample size: "); cat(nobs(x));
    cat("\nAverage number of estimated parameters:", round(nparam(x),digits=digits));
    cat("\nAverage number of degrees of freedom:", round(nobs(x)-nparam(x),digits=digits));

    if(!is.null(x$accuracy)){
        cat("\n\nForecast errors:\n");
        if(is.null(x$occurrence)){
            cat(paste(paste0("ME: ",round(x$accuracy["ME"],3)),
                      paste0("MAE: ",round(x$accuracy["MAE"],3)),
                      paste0("RMSE: ",round(sqrt(x$accuracy["MSE"]),3),"\n")
                      # paste0("Bias: ",round(x$accuracy["cbias"],3)*100,"%"),
                      ,sep="; "));
            cat(paste(paste0("sCE: ",round(x$accuracy["sCE"],5)*100,"%"),
                      paste0("sMAE: ",round(x$accuracy["sMAE"],5)*100,"%"),
                      paste0("sMSE: ",round(x$accuracy["sMSE"],5)*100,"%\n")
                      ,sep="; "));
            cat(paste(paste0("MASE: ",round(x$accuracy["MASE"],3)),
                      paste0("RMSSE: ",round(x$accuracy["RMSSE"],3)),
                      paste0("rMAE: ",round(x$accuracy["rMAE"],3)),
                      paste0("rRMSE: ",round(x$accuracy["rRMSE"],3),"\n")
                      ,sep="; "));
        }
        else{
            cat(paste(paste0("Bias: ",round(x$accuracy["cbias"],5)*100,"%"),
                      paste0("sMSE: ",round(x$accuracy["sMSE"],5)*100,"%"),
                      paste0("rRMSE: ",round(x$accuracy["rRMSE"],3)),
                      paste0("sPIS: ",round(x$accuracy["sPIS"],5)*100,"%"),
                      paste0("sCE: ",round(x$accuracy["sCE"],5)*100,"%\n"),sep="; "));
        }
    }
}

#### Coefficients ####
#### The functions needed for confint and reapply

# The function returns TRUE if the condition is violated
eigenValues <- function(object, persistence=NULL){
    if(is.null(persistence)){
        persistence <- object$persistence;
    }
    return(any(smoothEigens(matrix(persistence), object$transition, object$measurement,
                            modelLags(object), ncol(object$data)>1, nobs(object))>1+1E-10));
}

# The function that returns the bounds for persistence parameters, based on eigen values
eigenBounds <- function(object, persistence, variableNumber=1){
    # The lower bound
    persistence[variableNumber,] <- -5;
    eigenValuesTested <- eigenValues(object, persistence);
    while(eigenValuesTested){
        persistence[variableNumber,] <- persistence[variableNumber,] + 0.01;
        eigenValuesTested[] <- eigenValues(object, persistence);
        if(persistence[variableNumber,]>5){
            persistence[variableNumber,] <- -5;
            break;
        }
    }
    lowerBound <- persistence[variableNumber,]-0.01;
    # The upper bound
    persistence[variableNumber,] <- 5;
    eigenValuesTested <- eigenValues(object, persistence);
    while(eigenValuesTested){
        persistence[variableNumber,] <- persistence[variableNumber,] - 0.01;
        eigenValuesTested[] <- eigenValues(object, persistence);
        if(persistence[variableNumber,]<-5){
            persistence[variableNumber,] <- 5;
            break;
        }
    }
    upperBound <- persistence[variableNumber,]+0.01;
    return(c(lowerBound, upperBound));
}

# Function for the bounds of the AR parameters
arPolinomialsBounds <- function(arPolynomialMatrix,arPolynomial,variableNumber){
    # The lower bound
    arPolynomial[variableNumber] <- -5;
    arPolynomialMatrix[,1] <- -arPolynomial[-1];
    arPolyroots <- any(abs(eigen(arPolynomialMatrix, symmetric=FALSE, only.values=TRUE)$values)>1);
    stoppingCriteria <- 20;
    i <- 1;
    while(arPolyroots){
        arPolynomial[variableNumber] <- arPolynomial[variableNumber] +0.01;
        arPolynomialMatrix[,1] <- -arPolynomial[-1];
        arPolyroots[] <- any(abs(eigen(arPolynomialMatrix, symmetric=FALSE, only.values=TRUE)$values)>1);
        i[] <- i+1;
        if(i>=stoppingCriteria){
            break;
        }
    }
    lowerBound <- arPolynomial[variableNumber]-0.01;
    # The upper bound
    arPolynomial[variableNumber] <- 5;
    arPolynomialMatrix[,1] <- -arPolynomial[-1];
    arPolyroots <- any(abs(eigen(arPolynomialMatrix, symmetric=FALSE, only.values=TRUE)$values)>1);
    i[] <- 1;
    while(arPolyroots){
        arPolynomial[variableNumber] <- arPolynomial[variableNumber] -0.01;
        arPolynomialMatrix[,1] <- -arPolynomial[-1];
        arPolyroots[] <- any(abs(eigen(arPolynomialMatrix, symmetric=FALSE, only.values=TRUE)$values)>1);
        i[] <- i+1;
        if(i>=stoppingCriteria){
            break;
        }
    }
    upperBound <- arPolynomial[variableNumber]+0.01;
    return(c(lowerBound, upperBound));
}


# Confidence intervals
#' @export
confint.adam <- function(object, parm, level=0.95, bootstrap=FALSE, ...){
    parameters <- coef(object);
    confintNames <- c(paste0((1-level)/2*100,"%"),
                      paste0((1+level)/2*100,"%"));

    if(bootstrap){
        coefValues <- coefbootstrap(object, ...);
        adamReturn <- cbind(sqrt(diag(coefValues$vcov)),
                            apply(coefValues$coefficients,2,quantile,probs=(1-level)/2),
                            apply(coefValues$coefficients,2,quantile,probs=(1+level)/2));
        colnames(adamReturn) <- c("S.E.",confintNames);
    }
    else{
        adamVcov <- vcov(object, ...);
        adamSD <- sqrt(abs(diag(adamVcov)));
        parametersNames <- names(adamSD);
        nParam <- length(adamSD);
        etsModel <- etsChecker(object);
        arimaModel <- arimaChecker(object);
        adamCoefBounds <- matrix(0,nParam,2,
                                 dimnames=list(parametersNames,NULL));
        # Fill in the values with normal bounds
        adamCoefBounds[,1] <- qt((1-level)/2, df=nobs(object)-nparam(object))*adamSD;
        adamCoefBounds[,2] <- qt((1+level)/2, df=nobs(object)+nparam(object))*adamSD;

        persistence <- as.matrix(object$persistence);
        # If there is xreg, but no deltas, increase persistence by including zeroes
        # This can be considered as a failsafe mechanism
        if(ncol(object$data)>1 && !any(substr(names(object$persistence),1,5)=="delta")){
            persistence <- rbind(persistence,matrix(rep(0,sum(object$nParam[,2])),ncol=1));
        }

        # Correct the bounds for the ETS model
        if(etsModel){
            #### The usual bounds ####
            if(object$bounds=="usual"){
                # Check, if there is alpha
                if(any(parametersNames=="alpha")){
                    adamCoefBounds["alpha",1] <- max(-parameters["alpha"],adamCoefBounds["alpha",1]);
                    adamCoefBounds["alpha",2] <- min(1-parameters["alpha"],adamCoefBounds["alpha",2]);
                }
                # Check, if there is beta
                if(any(parametersNames=="beta")){
                    adamCoefBounds["beta",1] <- max(-parameters["beta"],adamCoefBounds["beta",1]);
                    if(any(parametersNames=="alpha")){
                        adamCoefBounds["beta",2] <- min(parameters["alpha"]-parameters["beta"],adamCoefBounds["beta",2]);
                    }
                    else{
                        adamCoefBounds["beta",2] <- min(object$persistence["alpha"]-parameters["beta"],adamCoefBounds["beta",2]);
                    }
                }
                # Check, if there are gammas
                if(any(substr(parametersNames,1,5)=="gamma")){
                    gammas <- which(substr(parametersNames,1,5)=="gamma");
                    adamCoefBounds[gammas,1] <- apply(cbind(adamCoefBounds[gammas,1],-parameters[gammas]),1,max);
                    if(any(parametersNames=="alpha")){
                        adamCoefBounds[gammas,2] <- apply(cbind(adamCoefBounds[gammas,2],
                                                                (1-parameters["alpha"])-parameters[gammas]),1,min);
                    }
                    else{
                        adamCoefBounds[gammas,2] <- apply(cbind(adamCoefBounds[gammas,2],
                                                                (1-object$persistence["alpha"])-parameters[gammas]),1,min);
                    }
                }
                # Check, if there are deltas (for xreg)
                if(any(substr(parametersNames,1,5)=="delta")){
                    deltas <- which(substr(parametersNames,1,5)=="delta");
                    adamCoefBounds[deltas,1] <- apply(cbind(adamCoefBounds[deltas,1],-parameters[deltas]),1,max);
                    adamCoefBounds[deltas,2] <- apply(cbind(adamCoefBounds[deltas,2],1-parameters[deltas]),1,min);
                }
                # These are "usual" bounds for phi. We don't care about other bounds
                if(any(parametersNames=="phi")){
                    adamCoefBounds["phi",1] <- max(-parameters["phi"],adamCoefBounds["phi",1]);
                    adamCoefBounds["phi",2] <- min(1-parameters["phi"],adamCoefBounds["phi",2]);
                }
            }
            #### Admissible bounds ####
            else if(object$bounds=="admissible"){
                # Check, if there is alpha
                if(any(parametersNames=="alpha")){
                    alphaBounds <- eigenBounds(object, persistence,
                                               variableNumber=which(names(object$persistence)=="alpha"));
                    adamCoefBounds["alpha",1] <- max(alphaBounds[1]-parameters["alpha"],adamCoefBounds["alpha",1]);
                    adamCoefBounds["alpha",2] <- min(alphaBounds[2]-parameters["alpha"],adamCoefBounds["alpha",2]);
                }
                # Check, if there is beta
                if(any(parametersNames=="beta")){
                    betaBounds <- eigenBounds(object, persistence,
                                              variableNumber=which(names(object$persistence)=="beta"));
                    adamCoefBounds["beta",1] <- max(betaBounds[1]-parameters["beta"],adamCoefBounds["beta",1]);
                    adamCoefBounds["beta",2] <- min(betaBounds[2]-parameters["beta"],adamCoefBounds["beta",2]);
                }
                # Check, if there are gammas
                if(any(substr(parametersNames,1,5)=="gamma")){
                    gammas <- which(substr(parametersNames,1,5)=="gamma");
                    for(i in 1:length(gammas)){
                        gammaBounds <- eigenBounds(object, persistence,
                                                   variableNumber=which(substr(names(object$persistence),1,5)=="gamma")[i]);
                        adamCoefBounds[gammas[i],1] <- max(gammaBounds[1]-parameters[gammas[i]],adamCoefBounds[gammas[i],1]);
                        adamCoefBounds[gammas[i],2] <- min(gammaBounds[2]-parameters[gammas[i]],adamCoefBounds[gammas[i],2]);
                    }
                }
                # Check, if there are deltas (for xreg)
                if(any(substr(parametersNames,1,5)=="delta")){
                    deltas <- which(substr(parametersNames,1,5)=="delta");
                    for(i in 1:length(deltas)){
                        deltaBounds <- eigenBounds(object, persistence,
                                                   variableNumber=which(substr(names(object$persistence),1,5)=="delta")[i]);
                        adamCoefBounds[deltas[i],1] <- max(deltaBounds[1]-parameters[deltas[i]],adamCoefBounds[deltas[i],1]);
                        adamCoefBounds[deltas[i],2] <- min(deltaBounds[2]-parameters[deltas[i]],adamCoefBounds[deltas[i],2]);
                    }
                }
            }

            # Restrictions on the initials for the multiplicative models (greater than zero)
            # Level
            # if(errorType(object)=="M" && any(parametersNames=="level")){
            #     adamCoefBounds["level",1] <- max(-parameters["level"],adamCoefBounds["level",1]);
            #     adamCoefBounds["level",2] <- max(-parameters["level"],adamCoefBounds["level",2]);
            # }
            adamModelType <- modelType(object);
            # Trend
            if(substr(adamModelType,2,2)=="M" && any(parametersNames=="trend")){
                adamCoefBounds["trend",1] <- max(-parameters["trend"],adamCoefBounds["trend",1]);
                adamCoefBounds["trend",2] <- max(-parameters["trend"],adamCoefBounds["trend",2]);
            }
            # Seasonality
            if(substr(adamModelType,nchar(adamModelType),nchar(adamModelType))=="M" &&
               any(substr(parametersNames,1,8)=="seasonal")){
                seasonals <- which(substr(parametersNames,1,8)=="seasonal");
                adamCoefBounds[seasonals,1] <- max(-parameters[seasonals],adamCoefBounds[seasonals,1]);
                adamCoefBounds[seasonals,2] <- max(-parameters[seasonals],adamCoefBounds[seasonals,2]);
            }
        }

        # Correct the bounds for the ARIMA model
        if(arimaModel){
            #### Deal with ARIMA parameters ####
            ariPolynomial <- object$other$polynomial$ariPolynomial;
            arPolynomial <- object$other$polynomial$arPolynomial;
            maPolynomial <- object$other$polynomial$maPolynomial;
            nonZeroARI <- object$other$ARIMAIndices$nonZeroARI;
            nonZeroMA <- object$other$ARIMAIndices$nonZeroMA;
            arPolynomialMatrix <- object$other$arPolynomialMatrix;
            # Locate all thetas for ARIMA
            thetas <- which(substr(parametersNames,1,5)=="theta");
            # Locate phi for ARIMA (they are always phi1, phi2 etc)
            phis <- which((substr(parametersNames,1,3)=="phi") & (nchar(parametersNames)>3));
            # Do loop for thetas
            if(length(thetas)>0){
                # MA parameters
                for(i in 1:length(thetas)){
                    # In this case, we check, where the standard condition is violated for an element of persistence,
                    # and then substitute the ARI part from that.
                    psiBounds <- eigenBounds(object, persistence,
                                             variableNumber=which(substr(names(object$persistence),1,3)=="psi")[nonZeroMA[i,2]]);
                    # If there are ARI elements in persistence, subtract (-(-x)) them to get proper bounds
                    if(any(nonZeroARI[,2]==i)){
                        ariIndex <- which(nonZeroARI[,2]==i);
                        adamCoefBounds[thetas[i],1] <- max(psiBounds[1]-parameters[thetas[i]]+ariPolynomial[nonZeroARI[ariIndex,1]],
                                                           adamCoefBounds[thetas[i],1]);
                        adamCoefBounds[thetas[i],2] <- min(psiBounds[2]-parameters[thetas[i]]+ariPolynomial[nonZeroARI[ariIndex,1]],
                                                           adamCoefBounds[thetas[i],2]);
                    }
                    else{
                        adamCoefBounds[thetas[i],1] <- max(psiBounds[1]-parameters[thetas[i]], adamCoefBounds[thetas[i],1]);
                        adamCoefBounds[thetas[i],2] <- min(psiBounds[2]-parameters[thetas[i]], adamCoefBounds[thetas[i],2]);
                    }
                }
            }
            # Locate phi for ARIMA (they are always phi1, phi2 etc)
            if(length(phis)>0){
                # AR parameters
                for(i in 1:length(phis)){
                    # Get bounds for AR based on stationarity condition
                    phiBounds <- arPolinomialsBounds(arPolynomialMatrix, arPolynomial,
                                                     which(arPolynomial==arPolynomial[arPolynomial!=0][-1][i]));

                    adamCoefBounds[phis[i],1] <- max(phiBounds[1]-parameters[phis[i]], adamCoefBounds[phis[i],1]);
                    adamCoefBounds[phis[i],2] <- min(phiBounds[2]-parameters[phis[i]], adamCoefBounds[phis[i],2]);
                }
            }
        }

        adamCoefBounds[] <- adamCoefBounds+parameters;
        adamReturn <- cbind(adamSD,adamCoefBounds);
        colnames(adamReturn) <- c("S.E.", confintNames);
    }

    # If parm was not provided, return everything.
    if(!exists("parm",inherits=FALSE)){
        parm <- names(adamSD);
    }

    return(adamReturn[parm,,drop=FALSE]);
}

#' @export
coef.adam <- function(object, ...){
    return(object$B);
}


#' @importFrom stats sigma
#' @export
sigma.adam <- function(object, ...){

    nParam <- nparam(object);

    # In case of likelihood, sigma was calculated as a parameter
    # This is not needed for unbiassing sigma
    if((!is.null(object$loss) && object$loss=="likelihood") ||
       (inherits(object, "adamCombined") && object$models[[1]]$loss=="likelihood")){
        nParam[] <- nParam - object$nParam[1,4];
    }

    df <- nobs(object, all=FALSE)-nParam;
    # If the sample is too small, then use biased estimator
    if(df<=0){
        df[] <- nobs(object);
    }

    return(sqrt(switch(object$distribution,
                       "dnorm"=,
                       "dlaplace"=,
                       "ds"=,
                       "dgnorm"=,
                       "dt"=,
                       "dlogis"=,
                       "dalaplace"=sum(residuals(object)^2,na.rm=TRUE),
                       "dlnorm"=,
                       "dllaplace"=,
                       "dls"=sum(log(residuals(object))^2,na.rm=TRUE),
                       "dlgnorm"=sum(log(residuals(object)-extractScale(object)^2/2)^2,na.rm=TRUE),
                       "dinvgauss"=,
                       "dgamma"=sum((residuals(object)-1)^2,na.rm=TRUE)
    )
    /df));
}

#' @export
summary.adam <- function(object, level=0.95, bootstrap=FALSE, ...){
    ourReturn <- list(model=object$model,responseName=all.vars(formula(object))[1]);

    occurrence <- NULL;
    if(is.occurrence(object$occurrence)){
        occurrence <- switch(object$occurrence$occurrence,
                             "f"=,
                             "fixed"="Fixed probability",
                             "o"=,
                             "odds-ratio"="Odds ratio",
                             "i"=,
                             "inverse-odds-ratio"="Inverse odds ratio",
                             "d"=,
                             "direct"="Direct",
                             "g"=,
                             "general"="General");
    }
    ourReturn$occurrence <- occurrence;
    ourReturn$distribution <- object$distribution;

    # Collect parameters and their standard errors
    parametersValues <- coef(object);
    if(!is.null(parametersValues)){
        parametersConfint <- confint(object, level=level, bootstrap=bootstrap, ...);
        # Record the type of bootstrap done
        ellipsis <- list(...);
        if(!is.null(ellipsis$method)){
            ourReturn$method <- ellipsis$method;
        }
        else{
            ourReturn$method <- "cr";
        }
        if(is.null(parametersValues)){
            if(ncol(object$data)>1 && all(object$persistenceXreg!=0)){
                parametersValues <- c(object$persistence,object$persistenceXreg,object$initial,object$initialXreg);
            }
            else{
                parametersValues <- c(object$persistence,object$initial);
            }
            warning(paste0("Parameters are not available. You have probably provided them in the model, ",
                           "so there was nothing to estimate. I extracted smoothing parameters and initials."),
                    call.=FALSE);
        }
        parametersTable <- cbind(parametersValues,parametersConfint);
        rownames(parametersTable) <- rownames(parametersConfint);
        colnames(parametersTable) <- c("Estimate","Std. Error",
                                       paste0("Lower ",(1-level)/2*100,"%"),
                                       paste0("Upper ",(1+level)/2*100,"%"));
        ourReturn$coefficients <- parametersTable;
        # Mark those that are significant on the selected level
        ourReturn$significance <- !(parametersTable[,3]<=0 & parametersTable[,4]>=0);
    }
    ourReturn$loss <- object$loss;
    ourReturn$lossValue <- object$lossValue;
    ourReturn$nobs <- nobs(object);
    ourReturn$nparam <- nparam(object);
    ourReturn$nParam <- object$nParam;
    ourReturn$call <- object$call;
    ourReturn$other <- object$other;
    ourReturn$sigma <- sigma(object);

    if(object$loss=="likelihood" ||
       (any(object$loss==c("MSE","MSEh","MSCE")) & (object$distribution=="dnorm")) ||
       (any(object$loss==c("MAE","MAEh","MACE")) & (object$distribution=="dlaplace")) ||
       (any(object$loss==c("HAM","HAMh","CHAM")) & (object$distribution=="ds"))){
        ICs <- c(AIC(object),AICc(object),BIC(object),BICc(object));
        names(ICs) <- c("AIC","AICc","BIC","BICc");
        ourReturn$ICs <- ICs;
    }
    ourReturn$bootstrap <- bootstrap;
    return(structure(ourReturn, class="summary.adam"));
}

#' @export
as.data.frame.summary.adam <- function(x, ...){
    return(as.data.frame(x$coefficients, ...));
}

#' @export
summary.adamCombined <- function(object, ...){
    return(print.adamCombined(object, ...));
}

#' @export
print.summary.adam <- function(x, ...){
    ellipsis <- list(...);
    if(!any(names(ellipsis)=="digits")){
        digits <- 4;
    }
    else{
        digits <- ellipsis$digits;
    }

    cat(paste0("\nModel estimated using ",tail(all.vars(x$call[[1]]),1),
               "() function: ",x$model));
    cat("\nResponse variable:", paste0(x$responseName,collapse=""));

    if(!is.null(x$occurrence)){
        cat("\nOccurrence model type:",x$occurrence);
    }

    distrib <- switch(x$distribution,
                      "dnorm" = "Normal",
                      "dlaplace" = "Laplace",
                      "ds" = "S",
                      "dgnorm" = paste0("Generalised Normal with shape=",round(x$other$shape,digits)),
                      "dlogis" = "Logistic",
                      "dt" = paste0("Student t with df=",round(x$other$nu, digits)),
                      "dalaplace" = paste0("Asymmetric Laplace with alpha=",round(x$other$alpha,digits)),
                      "dlnorm" = "Log-Normal",
                      "dllaplace" = "Log-Laplace",
                      "dls" = "Log-S",
                      "dlgnorm" = paste0("Log-Generalised Normal with shape=",round(x$other$shape,digits)),
                      # "dbcnorm" = paste0("Box-Cox Normal with lambda=",round(x$other$lambda,2)),
                      "dinvgauss" = "Inverse Gaussian",
                      "dgamma" = "Gamma"
    );
    if(!is.null(x$occurrence)){
        distrib <- paste0("\nMixture of Bernoulli and ", distrib);
    }
    cat("\nDistribution used in the estimation:", distrib);

    cat("\nLoss function type:",x$loss);
    if(!is.null(x$lossValue)){
        cat("; Loss function value:",round(x$lossValue,digits));
        if(any(x$loss==c("LASSO","RIDGE"))){
            cat("; lambda=",x$other$lambda);
        }
    }

    if(x$bootstrap){
        cat("\n");
        cat(switch(x$method,
                   "cr"="Case Resampling",
                   "dsr"="Data Shape Replication"),
            "bootstrap was used for the estimation of uncertainty of parameters");
    }

    if(!is.null(x$coefficients)){
        cat("\nCoefficients:\n");
        stars <- setNames(vector("character",length(x$significance)),
                          names(x$significance));
        stars[x$significance] <- "*";
        print(data.frame(round(x$coefficients,digits),stars,
                         check.names=FALSE,fix.empty.names=FALSE));
    }
    else{
        cat("\nAll coefficients were provided");
    }

    cat("\nError standard deviation:", round(x$sigma,digits));
    cat("\nSample size:", x$nobs);
    cat("\nNumber of estimated parameters:", x$nparam);
    cat("\nNumber of degrees of freedom:", x$nobs-x$nparam);
    if(x$nParam[2,4]>0){
        cat("\nNumber of provided parameters:", x$nParam[2,4]);
    }

    if(x$loss=="likelihood" ||
       (any(x$loss==c("MSE","MSEh","MSCE")) & (x$distribution=="dnorm")) ||
       (any(x$loss==c("MAE","MAEh","MACE")) & (x$distribution=="dlaplace")) ||
       (any(x$loss==c("HAM","HAMh","CHAM")) & (x$distribution=="ds"))){
        cat("\nInformation criteria:\n");
        print(round(x$ICs,digits));
    }
    else{
        cat("\nInformation criteria are unavailable for the chosen loss & distribution.\n");
    }
}

#' @export
xtable::xtable

#' @importFrom xtable xtable
#' @export
xtable.adam <- function(x, caption = NULL, label = NULL, align = NULL, digits = NULL,
                        display = NULL, auto = FALSE, ...){
    adamSummary <- summary(x);
    return(do.call("xtable", list(x=adamSummary,
                                  caption=caption, label=label, align=align, digits=digits,
                                  display=display, auto=auto, ...)));
}

#' @export
xtable.summary.adam <- function(x, caption = NULL, label = NULL, align = NULL, digits = NULL,
                                display = NULL, auto = FALSE, ...){
    # Substitute class with lm
    class(x) <- "summary.lm";
    return(do.call("xtable", list(x=x,
                                  caption=caption, label=label, align=align, digits=digits,
                                  display=display, auto=auto, ...)));
}


#' @importFrom greybox coefbootstrap dsrboot
#' @export
coefbootstrap.adam <- function(object, nsim=1000, size=floor(0.75*nobs(object)),
                               replace=FALSE, prob=NULL, parallel=FALSE,
                               method=c("cr","dsr"), ...){

    startTime <- Sys.time();

    cl <- match.call();
    yInSample <- actuals(object);
    cesModel <- cesChecker(object);
    gumModel <- gumChecker(object);
    ssarimaModel <- ssarimaChecker(object);

    method <- match.arg(method);

    if(is.numeric(parallel)){
        nCores <- parallel;
        parallel <- TRUE;
    }
    else if(is.logical(parallel) && parallel){
        # Detect number of cores for parallel calculations
        nCores <- min(parallel::detectCores() - 1, nsim);
    }

    # If they asked for parallel, make checks and try to do that
    if(parallel){
        if(!requireNamespace("foreach", quietly = TRUE)){
            stop("In order to run the function in parallel, 'foreach' package must be installed.", call. = FALSE);
        }
        if(!requireNamespace("parallel", quietly = TRUE)){
            stop("In order to run the function in parallel, 'parallel' package must be installed.", call. = FALSE);
        }

        # Check the system and choose the package to use
        if(Sys.info()['sysname']=="Windows"){
            if(requireNamespace("doParallel", quietly = TRUE)){
                cluster <- parallel::makeCluster(nCores);
                doParallel::registerDoParallel(cluster);
            }
            else{
                stop("Sorry, but in order to run the function in parallel, you need 'doParallel' package.",
                     call. = FALSE);
            }
        }
        else{
            if(requireNamespace("doMC", quietly = TRUE)){
                doMC::registerDoMC(nCores);
                cluster <- NULL;
            }
            else if(requireNamespace("doParallel", quietly = TRUE)){
                cluster <- parallel::makeCluster(nCores);
                doParallel::registerDoParallel(cluster);
            }
            else{
                stop("Sorry, but in order to run the function in parallel, you need either 'doMC' (prefered) or 'doParallel' packages.",
                     call. = FALSE);
            }
        }
    }

    # Coefficients of the model
    coefficientsOriginal <- coef(object);
    nVariables <- length(coefficientsOriginal);
    variablesNames <- names(coefficientsOriginal);
    # interceptIsNeeded <- any(variablesNames=="(Intercept)");
    # variablesNamesMade <- make.names(variablesNames);
    # if(interceptIsNeeded){
    #     variablesNamesMade[1] <- variablesNames[1];
    # }
    obsInsample <- nobs(object);

    # The matrix with coefficients
    coefBootstrap <- matrix(0, nsim, nVariables, dimnames=list(NULL, variablesNames));
    # Indices for the observations to use and the vector of subsets
    indices <- c(1:obsInsample);

    # Form the call for alm
    newCall <- object$call;
    # Switch off this, just in case.
    newCall$silent <- TRUE;
    # If this was auto.adam, use just adam
    if(newCall[[1]]=="auto.adam"){
        newCall[[1]] <- as.symbol("adam");
    }
    newCall$formula <- formula(object);
    if(!is.null(newCall$regressors)){
        newCall$regressors <- switch(newCall$regressors,"select"="use",newCall$regressors);
    }
    # This is based on the split data, so no need to do holdout
    newCall$holdout <- FALSE;
    newCall$distribution <- object$distribution;
    if(object$loss=="custom"){
        newCall$loss <- object$lossFunction;
    }
    else{
        newCall$loss <- object$loss;
    }
    # If ETS was selected
    if(!any(c(cesModel,gumModel,ssarimaModel)) && any(object$call!=modelType(object))){
        newCall$model <- modelType(object);
    }
    # If ARIMA was selected
    if(!is.null(object$call$orders$select)){
        newCall$orders <- orders(object);
        newCall$orders$select <- FALSE;
    }
    newCall$constant <- !is.null(object$constant);

    newCall$outliers <- "ignore";

    # Get lags and the minimum possible sample (2 seasons)
    lags <- lags(object);
    # This is needed for cases, when lags changed in the function
    newCall$lags <- lags;
    # Number of variables or a seasonal cycle + 2 (for security)
    obsMinimum <- max(lags, nVariables) + 2;

    # IF the sample is too small, make a warning and switch to dsr.
    if(obsMinimum>=obsInsample && method=="cr"){
        warning("Not enough observations to do Case Resampling bootstrap. Changing method to 'dsr'.",
                call.=FALSE, immediate.=TRUE);
        method <- "dsr";
    }

    # If this is ARIMA, and the size wasn't specified, make it changeable
    if(substr(object$model,1,10)=="Regression"){
        regressionPure <- TRUE;
    }
    else{
        regressionPure <- FALSE;
    }

    # If the additional parameter was not estimated, provide it
    if(all(names(object$B)!="other")){
        if(any(object$distribution==c("dchisq","dt"))){
            newCall$nu <- object$other$nu;
        }
        else if(object$distribution=="dalaplace"){
            newCall$alpha <- object$other$alpha;
        }
        else if(object$distribution=="dbcnorm"){
            newCall$lambdaBC <- object$other$lambdaBC;
        }
        else if(any(object$distribution==c("dgnorm","dlgnorm"))){
            newCall$shape <- object$other$shape;
        }
    }
    newCall$occurrence <- object$occurrence;

    # If this is backcasting, do sampling with moving origin
    changeOrigin <- FALSE;
    if(any(object$initialType==c("backcasting","complete"))){
        changeOrigin[] <- TRUE;
    }

    # Use the available parameters as starting point
    newCall$B <- object$B;
    newCall$lb <- rep(-Inf, length(object$B));
    newCall$ub <- rep(Inf, length(object$B));
    newCall$data <- object$data;

    # Function creates a random sample. Needed for dynamic models
    sampler <- function(indices,size,replace,prob,regressionPure=FALSE,changeOrigin=FALSE){
        if(regressionPure){
            return(sample(indices,size=size,replace=replace,prob=prob));
        }
        else{
            indices <- c(1:ceiling(runif(1,obsMinimum,obsInsample)));
            startingIndex <- 0
            if(changeOrigin){
                startingIndex <- floor(runif(1,0,obsInsample-max(indices)));
            }
            # This way we return the continuous sample, starting from the first observation
            return(startingIndex+indices);
        }
    }

    #### Bootstrap the data
    if(method=="dsr"){
        #### Data Shape Replication bootstrap
        responseName <- all.vars(formula(object))[1];
        # Create a new dataset
        newData <- replicate(nsim, newCall$data, simplify=FALSE);
        newCall$formula <- as.formula(paste0(responseName,"~."));
        type <- "multiplicative";
        if(any(yInSample<0)){
            type[] <- "additive";
        }
        #
        # Only bootstrap the response variable
        dataBoot <- dsrboot(yInSample, nsim=nsim, type=type, intermittent=FALSE);
        for(i in 1:nsim){
            newData[[i]][,responseName] <- dataBoot$boot[,i];
        }
        # This chunk of code does the bootstrap for explanatory variables as well
        # dataBoot <- suppressWarnings(apply(newCall$data, 2, dsrboot,
        #                                    nsim=nsim, type=type, intermittent=FALSE));
        # nLevels <- length(dataBoot);
        # # Fill in the list of data
        # for(i in 1:nsim){
        #     for(j in 1:nLevels){
        #         newData[[i]][,j] <- dataBoot[[j]]$boot[,i];
        #     }
        # }

        # Do the bootstrap
        if(!parallel){
            for(i in 1:nsim){
                newCall$data[] <- newData[[i]];
                testModel <- suppressWarnings(eval(newCall));
                coefBootstrap[i,variablesNames %in% names(coef(testModel))] <- coef(testModel);
            }
        }
        else{
            # We don't do rbind for security reasons - in order to deal with skipped variables
            coefBootstrapParallel <- foreach::`%dopar%`(foreach::foreach(i=1:nsim),{
                newCall$data[] <- newData[[i]];
                testModel <- eval(newCall);
                return(coef(testModel));
            })
            # Prepare the matrix with parameters
            for(i in 1:nsim){
                coefBootstrap[i,variablesNames %in% names(coefBootstrapParallel[[i]])] <- coefBootstrapParallel[[i]];
            }
        }
    }
    else{
        #### Case Resampling bootstrap
        if(!parallel){
            for(i in 1:nsim){
                subsetValues <- sampler(indices,size,replace,prob,regressionPure,changeOrigin);
                newCall$data <- object$data[subsetValues,,drop=FALSE];
                testModel <- suppressWarnings(eval(newCall));
                coefBootstrap[i,variablesNames %in% names(coef(testModel))] <- coef(testModel);
            }
        }
        else{
            # We don't do rbind for security reasons - in order to deal with skipped variables
            coefBootstrapParallel <- foreach::`%dopar%`(foreach::foreach(i=1:nsim),{
                subsetValues <- sampler(indices,size,replace,prob,regressionPure,changeOrigin);
                newCall$data <- object$data[subsetValues,,drop=FALSE];
                testModel <- eval(newCall);
                return(coef(testModel));
            })
            # Prepare the matrix with parameters
            for(i in 1:nsim){
                coefBootstrap[i,variablesNames %in% names(coefBootstrapParallel[[i]])] <- coefBootstrapParallel[[i]];
            }
        }
    }

    # Get rid of NAs. They mean "zero"
    coefBootstrap[is.na(coefBootstrap)] <- 0;

    # Rename the variables to the originals
    colnames(coefBootstrap) <- names(coefficientsOriginal);

    # Centre the coefficients for the calculation of the vcov
    coefvcov <- coefBootstrap - matrix(coefficientsOriginal, nsim, nVariables, byrow=TRUE);

    return(structure(list(vcov=(t(coefvcov) %*% coefvcov)/nsim,
                          coefficients=coefBootstrap, method=method,
                          nsim=nsim, size=NA, replace=NA, prob=NA,
                          parallel=parallel, model=object$call[[1]], timeElapsed=Sys.time()-startTime),
                     class="bootstrap"));
}

#' @export
vcov.adam <- function(object, bootstrap=FALSE, heuristics=NULL, ...){
    ellipsis <- list(...);

    # Heuristics is to set variance equal to sqrt(heuristics)% of values
    if(!is.null(heuristics)){
        if(is.numeric(heuristics)){
            return(diag(abs(coef(object))*heuristics));
        }
    }

    if(bootstrap){
        return(coefbootstrap(object, ...)$vcov);
    }
    else{
        # If the forecast is in numbers, then use its length as a horizon
        if(any(!is.na(object$forecast))){
            h <- length(object$forecast)
        }
        else{
            h <- 0;
        }
        if(substr(object$model,1,10)=="Regression"){
            modelFormula <- formula(object);
            testModel <- structure(list(call=object$call,
                                        data=as.matrix(model.matrix(modelFormula,
                                                                    data=model.frame(modelFormula,
                                                                                     data=as.data.frame(object$data)))),
                                        distribution=object$distribution, occurrence=object$occurrence,
                                        coefficients=coef(object), logLik=logLik(object),
                                        residuals=residuals(object), df=nparam(object), loss=object$loss,
                                        other=object$other),
                                   class=c("alm","greybox"));
            testModel$call$formula <- modelFormula;
            testModel$data[,1] <- object$data[,1];
            colnames(testModel$data)[1] <- all.vars(modelFormula)[1];
            return(vcov(testModel));
        }
        else{
            cesModel <- cesChecker(object);
            gumModel <- gumChecker(object);
            ssarimaModel <- ssarimaChecker(object);
            if(cesModel){
                # If this is not provided by user, use a low value. CES is sensitive, so the small steps are required
                if(is.null(ellipsis$stepSize)){
                    ellipsis$stepSize <- 1e-8;
                }
                modelReturn <- suppressWarnings(ces(object$data, h=0, model=object, formula=formula(object),
                                                    FI=TRUE, stepSize=ellipsis$stepSize));
            }
            else if(gumModel){
                modelReturn <- suppressWarnings(gum(object$data, h=0, model=object, formula=formula(object),
                                                    FI=TRUE, stepSize=ellipsis$stepSize));
            }
            else if(ssarimaModel){
                modelReturn <- suppressWarnings(ssarima(object$data, h=0, model=object, formula=formula(object),
                                                        FI=TRUE, stepSize=ellipsis$stepSize));
            }
            else{
                modelReturn <- suppressWarnings(adam(object$data, h=0, model=object, formula=formula(object),
                                                     FI=TRUE, stepSize=ellipsis$stepSize));
            }
            # If any row contains all zeroes, then it means that the variable does not impact the likelihood. Invert the matrix without it.
            brokenVariables <- apply(modelReturn$FI==0,1,all) | apply(is.nan(modelReturn$FI),1,any);
            # If there are issues, try the same stuff, but with a different step size for hessian
            if(any(brokenVariables)){
                if(cesModel){
                    modelReturn <- suppressWarnings(ces(object$data, h=0, model=object, formula=formula(object),
                                                        FI=TRUE, stepSize=.Machine$double.eps^(1/6)));
                }
                else if(gumModel){
                    modelReturn <- suppressWarnings(gum(object$data, h=0, model=object, formula=formula(object),
                                                        FI=TRUE, stepSize=.Machine$double.eps^(1/6)));
                }
                else if(ssarimaModel){
                    modelReturn <- suppressWarnings(ssarima(object$data, h=0, model=object, formula=formula(object),
                                                            FI=TRUE, stepSize=.Machine$double.eps^(1/6)));
                }
                else{
                    modelReturn <- suppressWarnings(adam(object$data, h=0, model=object, formula=formula(object),
                                                         FI=TRUE, stepSize=.Machine$double.eps^(1/6)));
                }
                brokenVariables <- apply(modelReturn$FI==0,1,all);
            }
            # If there are NaNs, then this has not been estimated well
            if(any(is.nan(modelReturn$FI))){
                stop("The Fisher Information cannot be calculated numerically with provided parameters - it contains NaNs.",
                     "Try setting stepSize for the hessian to something like stepSize=1e-6 or using the bootstrap.", call.=FALSE);
            }
            if(any(eigen(modelReturn$FI,only.values=TRUE)$values<0)){
                warning("Observed Fisher Information is not positive semi-definite, ",
                        "which might mean that the likelihood was not maximised properly. ",
                        "There's not much I can do with that. ",
                        "Just keep in mind that there might be some issues ",
                        "with the covariance matrix of parameters.", call.=FALSE);
            }
            FIMatrix <- modelReturn$FI[!brokenVariables,!brokenVariables,drop=FALSE];

            vcovMatrix <- try(chol2inv(chol(FIMatrix)), silent=TRUE);
            if(inherits(vcovMatrix,"try-error")){
                vcovMatrix <- try(solve(FIMatrix, diag(ncol(FIMatrix)), tol=1e-20), silent=TRUE);
                if(inherits(vcovMatrix,"try-error")){
                    warning(paste0("Sorry, but the hessian is singular, so I could not invert it.\n",
                                   "I failed to produce the covariance matrix of parameters. Shame on me!"),
                            call.=FALSE);
                    vcovMatrix <- diag(1e+100,ncol(FIMatrix));
                }
            }
            # If there were broken variables, reproduce the zero elements.
            # Reuse FI object in order to preserve memory. The names of cols / rows should be fine.
            modelReturn$FI[!brokenVariables,!brokenVariables] <- vcovMatrix;
            modelReturn$FI[brokenVariables,] <- modelReturn$FI[,brokenVariables] <- Inf;

            # Just in case, take absolute values for the diagonal (in order to avoid possible issues with FI)
            diag(modelReturn$FI) <- abs(diag(modelReturn$FI));
            return(modelReturn$FI);
        }
    }
}

#### Residuals and actuals functions ####

#' @importFrom greybox actuals
#' @export
actuals.adam <- function(object, all=TRUE, ...){
    responseName <- all.vars(formula(object))[1];
    if(all){
        response <- object$data[,responseName];
    }
    else{
        response <- object$data[object$data[,responseName]!=0,responseName];
    }
    if(inherits(response,"tbl")){
        response <- response[[1]];
    }
    return(response);
}

#' @export
nobs.adam <- function(object, ...){
    return(length(actuals(object, ...)));
}

#' @export
residuals.adam <- function(object, ...){
    return(switch(object$distribution,
                  "dlnorm"=,
                  "dllaplace"=,
                  "dls"=,
                  "dlgnorm"=,
                  "dgamma"=,
                  "dinvgauss"=switch(errorType(object),
                                     # abs() is needed in order to avoid the weird cases
                                     "A"=abs(1+object$residuals/fitted(object)),
                                     "M"=1+object$residuals),
                  "dnorm"=,
                  "dlaplace"=,
                  "ds"=,
                  "dgnorm"=,
                  "dlogis"=,
                  "dt"=,
                  "dalaplace"=,
                  object$residuals));
}

#' @importFrom stats rstandard
#' @export
rstandard.adam <- function(model, ...){
    obs <- nobs(model);
    df <- obs - nparam(model);
    errors <- residuals(model);
    # If this is an occurrence model, then only modify the non-zero obs
    # Also, if there are NAs in actuals, consider them as occurrence
    if(is.occurrence(model$occurrence)){
        residsToGo <- which(actuals(model$occurrence)!=0 & !is.na(actuals(model)));
    }
    else{
        residsToGo <- c(1:obs);
    }

    if(any(model$distribution==c("dt","dnorm"))){
        return((errors - mean(errors[residsToGo])) / sqrt(extractScale(model)^2 * obs / df));
    }
    else if(model$distribution=="ds"){
        return((errors - mean(errors[residsToGo])) / (extractScale(model) * obs / df)^2);
    }
    else if(model$distribution=="dls"){
        errors[] <- log(errors);
        return(exp((errors - mean(errors[residsToGo])) / (extractScale(model) * obs / df)^2));
    }
    else if(model$distribution=="dgnorm"){
        return((errors - mean(errors[residsToGo])) / (extractScale(model)^model$other$shape * obs / df)^{1/model$other$shape});
    }
    else if(model$distribution=="dlgnorm"){
        errors[] <- log(errors);
        return(exp((errors - mean(errors[residsToGo])) / (extractScale(model)^model$other$shape * obs / df)^{1/model$other$shape}));
    }
    else if(any(model$distribution==c("dinvgauss","dgamma"))){
        return(errors / mean(errors[residsToGo]));
    }
    else if(model$distribution=="dlnorm"){
        # Debias the residuals
        errors[] <- log(errors) + extractScale(model)^2/2;
        return(exp((errors - mean(errors[residsToGo])) / sqrt(extractScale(model)^2 * obs / df)));
    }
    else if(model$distribution=="dllaplace"){
        errors[] <- log(errors);
        return(exp((errors - mean(errors[residsToGo])) / extractScale(model) * obs / df));
    }
    else{
        return(errors / extractScale(model) * obs / df);
    }
}

#' @importFrom stats rstudent
#' @export
rstudent.adam <- function(model, ...){
    obs <- nobs(model);
    df <- obs - nparam(model) - 1;
    rstudentised <- errors <- residuals(model);
    # If this is an occurrence model, then only modify the non-zero obs
    # Also, if there are NAs in actuals, consider them as occurrence
    if(is.occurrence(model$occurrence)){
        residsToGo <- which(actuals(model$occurrence)!=0 & !is.na(actuals(model)));
    }
    else{
        residsToGo <- c(1:obs);
    }
    if(any(model$distribution==c("dt","dnorm"))){
        errors[] <- errors - mean(errors);
        for(i in residsToGo){
            rstudentised[i] <- errors[i] / sqrt(sum(errors[-i]^2,na.rm=TRUE) / df);
        }
    }
    else if(model$distribution=="dlaplace"){
        errors[] <- errors - mean(errors);
        for(i in residsToGo){
            rstudentised[i] <- errors[i] / (sum(abs(errors[-i]),na.rm=TRUE) / df);
        }
    }
    else if(model$distribution=="dlnorm"){
        errors[] <- log(errors) - mean(log(errors)) - extractScale(model)^2/2;
        for(i in residsToGo){
            rstudentised[i] <- exp(errors[i] / sqrt(sum(errors[-i]^2,na.rm=TRUE) / df));
        }
    }
    else if(model$distribution=="dllaplace"){
        errors[] <- log(errors) - mean(log(errors));
        for(i in residsToGo){
            rstudentised[i] <- exp(errors[i] / (sum(abs(errors[-i]),na.rm=TRUE) / df));
        }
    }
    else if(model$distribution=="ds"){
        errors[] <- errors - mean(errors);
        for(i in residsToGo){
            rstudentised[i] <- errors[i] / (sum(sqrt(abs(errors[-i])),na.rm=TRUE) / (2*df))^2;
        }
    }
    else if(model$distribution=="dls"){
        errors[] <- log(errors) - mean(log(errors));
        for(i in residsToGo){
            rstudentised[i] <- exp(errors[i] / (sum(sqrt(abs(errors[-i])),na.rm=TRUE) / (2*df))^2);
        }
    }
    else if(model$distribution=="dgnorm"){
        errors[] <- errors - mean(errors);
        for(i in residsToGo){
            rstudentised[i] <- errors[i] /  (sum(abs(errors[-i])^model$other$shape) * (model$other$shape/df))^{1/model$other$shape};
        }
    }
    else if(model$distribution=="dlgnorm"){
        errors[] <- log(errors) - mean(log(errors));
        for(i in residsToGo){
            rstudentised[i] <- errors[i] /  (sum(abs(errors[-i])^model$other$shape) * (model$other$shape/df))^{1/model$other$shape};
        }
    }
    else if(model$distribution=="dalaplace"){
        for(i in residsToGo){
            rstudentised[i] <- errors[i] / (sum(errors[-i] * (model$other$alpha - (errors[-i]<=0)*1),na.rm=TRUE) / df);
        }
    }
    else if(model$distribution=="dlogis"){
        errors[] <- errors - mean(errors);
        for(i in residsToGo){
            rstudentised[i] <- errors[i] / (sqrt(sum(errors[-i]^2,na.rm=TRUE) / df) * sqrt(3) / pi);
        }
    }
    else if(any(model$distribution==c("dinvgauss","dgamma"))){
        for(i in residsToGo){
            rstudentised[i] <- errors[i] / mean(errors[residsToGo][-i],na.rm=TRUE);
        }
    }
    else{
        for(i in residsToGo){
            rstudentised[i] <- errors[i] / sqrt(sum(errors[-i]^2,na.rm=TRUE) / df);
        }
    }
    return(rstudentised);
}

#' @importFrom greybox outlierdummy
#' @importFrom stats qchisq
#' @export
outlierdummy.adam <- function(object, level=0.999, type=c("rstandard","rstudent"), ...){
    # Function returns the matrix of dummies with outliers
    type <- match.arg(type);
    errors <- switch(type,"rstandard"=rstandard(object),"rstudent"=rstudent(object));
    statistic <- switch(object$distribution,
                        "dlaplace"=,
                        "dllaplace"=qlaplace(c((1-level)/2, (1+level)/2), 0, 1),
                        "dalaplace"=qalaplace(c((1-level)/2, (1+level)/2), 0, 1, object$other$alpha),
                        "dlogis"=qlogis(c((1-level)/2, (1+level)/2), 0, 1),
                        "dt"=qt(c((1-level)/2, (1+level)/2), nobs(object)-nparam(object)),
                        "dgnorm"=,
                        "dlgnorm"=qgnorm(c((1-level)/2, (1+level)/2), 0, 1, object$other$shape),
                        "ds"=,
                        "dls"=qs(c((1-level)/2, (1+level)/2), 0, 1),
                        # In the next one, the scale is debiased, taking n-k into account
                        "dinvgauss"=qinvgauss(c((1-level)/2, (1+level)/2), mean=1,
                                              dispersion=mean(extractScale(object)) * nobs(object) /
                                                  (nobs(object)-nparam(object))),
                        "dgamma"=qgamma(c((1-level)/2, (1+level)/2), shape=1/extractScale(object), scale=extractScale(object)),
                        qnorm(c((1-level)/2, (1+level)/2), 0, 1));
    # Fix for IG in case of scale - it should be chi-squared
    if(is.scale(object) && object$distribution=="dinvgauss"){
        statistic <- qchisq(c((1-level)/2, (1+level)/2), 1);
    }
    if(any(object$distribution==c("dlnorm","dllaplace","dls","dlgnorm"))){
        errors[] <- log(errors);
    }
    outliersID <- which(errors>statistic[2] | errors<statistic[1]);
    outliersNumber <- length(outliersID);
    if(outliersNumber>0){
        outliers <- matrix(0, nobs(object), outliersNumber,
                           dimnames=list(rownames(actuals(object)),
                                         paste0("outlier",c(1:outliersNumber))));
        outliers[cbind(outliersID,c(1:outliersNumber))] <- 1;
    }
    else{
        outliers <- NULL;
    }

    return(structure(list(outliers=outliers, statistic=statistic, id=outliersID,
                          level=level, type=type),
                     class="outlierdummy"));
}

#### Predict and forecast functions ####
#' @export
predict.adam <- function(object, newdata=NULL, interval=c("none", "confidence", "prediction"),
                         level=0.95, side=c("both","upper","lower"), ...){

    interval <- match.arg(interval);
    obsInSample <- nobs(object);

    # Indices and classes of the original data
    yIndex <- time(actuals(object));
    yClasses <- class(actuals(object));
    if(any(yClasses=="ts")){
        # ts structure
        yStart <- yIndex[1];
        yFrequency <- frequency(actuals(object));
    }

    # Check if newdata is provided
    if(!is.null(newdata)){
        # If this is not a matrix / data.frame, then convert to one
        if(!is.data.frame(newdata) && !is.matrix(newdata)){
            newdata <- as.data.frame(newdata);
            colnames(newdata) <- "xreg";
        }
        h <- nrow(newdata);
        # If the newdata is provided, then just do forecasts for that part
        if(any(interval==c("none","prediction","confidence"))){
            if(interval==c("prediction")){
                interval[] <- "simulated";
            }
            return(forecast(object, h=h, newdata=newdata,
                            interval=interval,
                            level=level, side=side, ...));
        }
    }
    else{
        # If there are no newdata, then we need to produce fitted with / without interval
        if(interval=="none"){
            return(structure(list(mean=fitted(object), lower=NA, upper=NA, model=object,
                                  level=level, interval=interval, side=side),
                             class=c("adam.predict","adam.forecast")));
        }
        # Otherwise we do one-step-ahead prediction / confidence interval
        else{
            yForecast <- fitted(object);
        }
    }

    ##### Prediction interval for in sample only! #####
    side <- match.arg(side);

    if(length(level)>1){
        warning(paste0("Sorry, but I only support scalar for the level, ",
                       "when constructing in-sample interval. ",
                       "Using the first provided value."),
                call.=FALSE);
        level <- level[1];
    }
    # Fix just in case a silly user used 95 etc instead of 0.95
    if(level>1){
        level[] <- level / 100;
    }

    # Basic parameters
    model <- modelType(object);
    Etype <- errorType(object);

    # Extract variance and amend it in case of confidence interval
    s2 <- sigma(object)^2;

    # If this is a mixture model, produce forecasts for the occurrence
    if(!is.null(object$occurrence)){
        occurrenceModel <- TRUE;
        pForecast <- fitted(object$occurrence);
    }
    else{
        occurrenceModel <- FALSE;
        pForecast <- rep(1, obsInSample);
    }

    # If this is an occurrence model, then take probability into account in the level.
    if(occurrenceModel && (interval=="prediction")){
        levelNew <- (level-(1-pForecast))/pForecast;
        levelNew[levelNew<0] <- 0;
    }
    else{
        levelNew <- level;
    }

    levelLow <- levelUp <- vector("numeric",obsInSample);
    if(side=="both"){
        levelLow[] <- (1-levelNew)/2;
        levelUp[] <- (1+levelNew)/2;
    }
    else if(side=="upper"){
        levelLow[] <- rep(0,length(levelNew));
        levelUp[] <- levelNew;
    }
    else{
        levelLow[] <- 1-levelNew;
        levelUp[] <- rep(1,length(levelNew));
    }
    levelLow[levelLow<0] <- 0;
    levelUp[levelUp<0] <- 0;

    nLevels <- 1;
    # Matrices for levels

    # Create necessary matrices for the forecasts
    if(any(yClasses=="ts")){
        yUpper <- yLower <- ts(matrix(0,obsInSample,nLevels), start=yStart, frequency=yFrequency);
    }
    else{
        yUpper <- yLower <- zoo(matrix(0,obsInSample,nLevels), order.by=yIndex);
    }
    colnames(yLower) <- switch(side,
                               "both"=paste0("Lower bound (",(1-level)/2*100,"%)"),
                               "lower"=paste0("Lower bound (",(1-level)*100,"%)"),
                               "upper"=rep("Lower 0%",nLevels));

    colnames(yUpper) <- switch(side,
                               "both"=paste0("Upper bound (",(1+level)/2*100,"%)"),
                               "lower"=rep("Upper 100%",nLevels),
                               "upper"=paste0("Upper bound (",level*100,"%)"));

    #### Call reapply if this is confidence ####
    if(interval=="confidence"){
        yFittedMatrix <- reapply(object, ...);
        for(i in 1:obsInSample){
            yUpper[i] <- quantile(yFittedMatrix$refitted[i,], levelLow[i], na.rm=TRUE);
            yLower[i] <- quantile(yFittedMatrix$refitted[i,], levelUp[i], na.rm=TRUE);
        }
        return(structure(list(mean=yForecast, lower=yLower, upper=yUpper, model=object,
                              level=level, interval=interval, side=side),
                         class=c("adam.predict","adam.forecast")))
    }

    #### Produce the prediction intervals ####
    if(object$distribution=="dnorm"){
        if(Etype=="A"){
            yLower[] <- qnorm(levelLow, 0, sqrt(s2));
            yUpper[] <- qnorm(levelUp, 0, sqrt(s2));
        }
        else{
            yLower[] <- qnorm(levelLow, 1, sqrt(s2));
            yUpper[] <- qnorm(levelUp, 1, sqrt(s2));
        }
    }
    else if(object$distribution=="dlaplace"){
        if(Etype=="A"){
            yLower[] <- qlaplace(levelLow, 0, sqrt(s2/2));
            yUpper[] <- qlaplace(levelUp, 0, sqrt(s2/2));
        }
        else{
            yLower[] <- qlaplace(levelLow, 1, sqrt(s2/2));
            yUpper[] <- qlaplace(levelUp, 1, sqrt(s2/2));
        }
    }
    else if(object$distribution=="ds"){
        if(Etype=="A"){
            yLower[] <- qs(levelLow, 0, (s2/120)^0.25);
            yUpper[] <- qs(levelUp, 0, (s2/120)^0.25);
        }
        else{
            yLower[] <- qs(levelLow, 1, (s2/120)^0.25);
            yUpper[] <- qs(levelUp, 1, (s2/120)^0.25);
        }
    }
    else if(object$distribution=="dgnorm"){
        scale <- sqrt(s2*(gamma(1/object$other$shape)/gamma(3/object$other$shape)));
        if(Etype=="A"){
            yLower[] <- suppressWarnings(qgnorm(levelLow, 0, scale, object$other$shape));
            yUpper[] <- suppressWarnings(qgnorm(levelUp, 0, scale, object$other$shape));
        }
        else{
            yLower[] <- suppressWarnings(qgnorm(levelLow, 1, scale, object$other$shape));
            yUpper[] <- suppressWarnings(qgnorm(levelUp, 1, scale, object$other$shape));
        }
    }
    else if(object$distribution=="dlogis"){
        if(Etype=="A"){
            yLower[] <- qlogis(levelLow, 0, sqrt(s2*3)/pi);
            yUpper[] <- qlogis(levelUp, 0, sqrt(s2*3)/pi);
        }
        else{
            yLower[] <- qlogis(levelLow, 1, sqrt(s2*3)/pi);
            yUpper[] <- qlogis(levelUp, 1, sqrt(s2*3)/pi);
        }
    }
    else if(object$distribution=="dt"){
        df <- nobs(object) - nparam(object);
        if(Etype=="A"){
            yLower[] <- sqrt(s2)*qt(levelLow, df);
            yUpper[] <- sqrt(s2)*qt(levelUp, df);
        }
        else{
            yLower[] <- (1 + sqrt(s2)*qt(levelLow, df));
            yUpper[] <- (1 + sqrt(s2)*qt(levelUp, df));
        }
    }
    else if(object$distribution=="dalaplace"){
        alpha <- object$other$alpha;
        if(Etype=="A"){
            yLower[] <- qalaplace(levelLow, 0,
                                  sqrt(s2*alpha^2*(1-alpha)^2/(alpha^2+(1-alpha)^2)), alpha);
            yUpper[] <- qalaplace(levelUp, 0,
                                  sqrt(s2*alpha^2*(1-alpha)^2/(alpha^2+(1-alpha)^2)), alpha);
        }
        else{
            yLower[] <- qalaplace(levelLow, 1,
                                  sqrt(s2*alpha^2*(1-alpha)^2/(alpha^2+(1-alpha)^2)), alpha);
            yUpper[] <- qalaplace(levelUp, 1,
                                  sqrt(s2*alpha^2*(1-alpha)^2/(alpha^2+(1-alpha)^2)), alpha);
        }
    }
    else if(object$distribution=="dlnorm"){
        # Take into account the logN restrictions
        yLower[] <- qlnorm(levelLow, -(1-sqrt(abs(1-s2)))^2, sqrt(2*(1-sqrt(abs(1-s2)))));
        yUpper[] <- qlnorm(levelUp, -(1-sqrt(abs(1-s2)))^2, sqrt(2*(1-sqrt(abs(1-s2)))));
    }
    else if(object$distribution=="dllaplace"){
        yLower[] <- exp(qlaplace(levelLow, 0, sqrt(s2/2)));
        yUpper[] <- exp(qlaplace(levelUp, 0, sqrt(s2/2)));
    }
    else if(object$distribution=="dls"){
        yLower[] <- exp(qs(levelLow, 0, (s2/120)^0.25));
        yUpper[] <- exp(qs(levelUp, 0, (s2/120)^0.25));
    }
    else if(object$distribution=="dlgnorm"){
        scale <- sqrt(s2*(gamma(1/object$other$shape)/gamma(3/object$other$shape)));
        yLower[] <- suppressWarnings(exp(qgnorm(levelLow, 0, scale, object$other$shape)));
        yUpper[] <- suppressWarnings(exp(qgnorm(levelUp, 0, scale, object$other$shape)));
    }
    else if(object$distribution=="dinvgauss"){
        yLower[] <- qinvgauss(levelLow, 1, dispersion=s2);
        yUpper[] <- qinvgauss(levelUp, 1, dispersion=s2);
    }
    else if(object$distribution=="dgamma"){
        yLower[] <- qgamma(levelLow, shape=1/s2, scale=s2);
        yUpper[] <- qgamma(levelUp, shape=1/s2, scale=s2);
    }

    #### Clean up the produced values for the interval ####
    # Make sensible values out of those weird quantiles
    if(Etype=="A"){
        yLower[levelLow==0] <- -Inf;
    }
    else{
        yLower[levelLow==0] <- 0;
    }
    yUpper[levelUp==1] <- Inf;

    # Substitute NAs and NaNs with zeroes
    if(any(is.nan(yLower)) || any(is.na(yLower))){
        yLower[is.nan(yLower)] <- 0;
        yLower[is.na(yLower)] <- 0;
    }
    if(any(is.nan(yUpper)) || any(is.na(yUpper))){
        yUpper[is.nan(yUpper)] <- 0;
        yUpper[is.na(yUpper)] <- 0;
    }

    if(Etype=="A"){
        yLower[] <- yForecast + yLower;
        yUpper[] <- yForecast + yUpper;
    }
    else{
        yLower[] <- yForecast * yLower;
        yUpper[] <- yForecast * yUpper;
    }

    return(structure(list(mean=yForecast, lower=yLower, upper=yUpper, model=object,
                          level=level, interval=interval, side=side),
                     class=c("adam.predict","adam.forecast")));
}

#' @export
plot.adam.predict <- function(x, ...){
    paletteBasic <- paletteDetector(c("black","red","purple","blue","darkgrey"));

    ellipsis <- list(...);
    if(is.null(ellipsis$ylim)){
        ellipsis$ylim <- range(c(actuals(x$model),x$mean,x$lower,x$upper),na.rm=TRUE);
    }
    ellipsis$x <- actuals(x$model);
    do.call(plot, ellipsis);
    lines(x$mean,col=paletteBasic[3],lwd=2,lty=2);
    if(x$interval!="none"){
        lines(x$lower,col=paletteBasic[5],lwd=3,lty=2);
        lines(x$upper,col=paletteBasic[5],lwd=3,lty=2);
    }
}

#' Forecasting time series using smooth functions
#'
#' Function produces conditional expectation (point forecasts) and prediction
#' intervals for the estimated model.
#'
#' By default the function will generate conditional expectations from the
#' estimated model and will also produce a variety of prediction intervals
#' based on user preferences.
#'
#' @aliases forecast forecast.smooth
#' @param object Time series model for which forecasts are required.
#' @param h Forecast horizon.
#' @param level Confidence level. Defines width of prediction interval.
#' @param side Defines, whether to provide \code{"both"} sides of prediction
#' interval or only \code{"upper"}, or \code{"lower"}.
#' @param ...  Other arguments passed to \code{reforecast()}, or \code{predict()}
#' functions.
#' @param newdata The new data needed in order to produce forecasts.
#' @param nsim Number of iterations to do in cases of \code{interval="simulated"},
#' \code{interval="prediction"} (for mixed and multiplicative model),
#' \code{interval="confidence"} and \code{interval="complete"}.
#' The default value for the prediction / simulated interval is 1000. In case of
#' confidence or complete intervals, this is set to 100.
#' @param interval What type of mechanism to use for interval construction.
#' the recommended option is \code{interval="prediction"}, which will use analytical
#' solutions for pure additive models and simulations for the others.
#' \code{interval="simulated"} is the slowest method, but is robust to the type of
#' model. \code{interval="approximate"} (aka \code{interval="parametric"}) uses
#' analytical formulae for conditional h-steps ahead variance, but is approximate
#' for the non-additive error models. \code{interval="semiparametric"} relies on the
#' multiple steps ahead forecast error (extracted via \code{rmultistep} method) and on
#' the assumed distribution of the error term. \code{interval="nonparametric"} uses
#' Taylor & Bunn (1999) approach with quantile regressions. \code{interval="empirical"}
#' constructs intervals based on empirical quantiles of multistep forecast errors.
#' \code{interval="complete"} will call for \code{reforecast()} function and produce
#' interval based on the uncertainty around the parameters of the model.
#' Finally, \code{interval="confidence"} tries to generate the confidence intervals
#' for the point forecast based on the \code{reforecast} method.
#' @param cumulative If \code{TRUE}, then the cumulative forecast and prediction
#' interval are produced instead of the normal ones. This is useful for
#' inventory control systems.
#' @param occurrence The vector containing the future occurrence variable
#' (values in [0,1]), if it is known.
#' @param scenarios Binary, defining whether to return scenarios produced via
#' simulations or not. Only works if \code{interval="simulated"}. If \code{TRUE}
#' the object will contain \code{scenarios} variable.
#' @return Returns object of class "smooth.forecast", which contains:
#'
#' \itemize{
#' \item \code{model} - the estimated model (ES / CES / GUM / SSARIMA).
#' \item \code{method} - the name of the estimated model (ES / CES / GUM / SSARIMA).
#' \item \code{forecast} aka \code{mean} - point forecasts of the model
#' (conditional mean).
#' \item \code{lower} - lower bound of prediction interval.
#' \item \code{upper} - upper bound of prediction interval.
#' \item \code{level} - confidence level.
#' \item \code{interval} - binary variable (whether interval were produced or not).
#' \item \code{scenarios} - in case of \code{forecast.adam()} and
#' \code{interval="simulated"} returns matrix with scenarios (future paths) that were
#' used in simulations.
#' }
#' @template ssAuthor
#' @template ssIntervalsRef
#' @seealso \code{\link[generics]{forecast}}
#' @references \itemize{
#' \item Hyndman, R.J., Koehler, A.B., Ord, J.K., and Snyder, R.D. (2008)
#' Forecasting with exponential smoothing: the state space approach,
#' Springer-Verlag.}
#'
#' @keywords ts univar
#' @examples
#'
#' ourModel <- es(rnorm(100,0,1), h=10)
#' forecast(ourModel, h=10, interval="parametric")
#'
#' @rdname forecast.smooth
#' @importFrom generics forecast
#' @importFrom stats rnorm rlogis rt rlnorm rgamma
#' @importFrom stats qnorm qlogis qt qlnorm qgamma
#' @importFrom statmod rinvgauss qinvgauss
#' @importFrom greybox rlaplace rs ralaplace rgnorm
#' @importFrom greybox qlaplace qs qalaplace qgnorm
#' @export
forecast.adam <- function(object, h=10, newdata=NULL, occurrence=NULL,
                          interval=c("none", "prediction", "confidence", "simulated",
                                     "approximate", "semiparametric", "nonparametric",
                                     "empirical","complete"),
                          level=0.95, side=c("both","upper","lower"), cumulative=FALSE, nsim=NULL,
                          scenarios=FALSE, ...){

    ellipsis <- list(...);

    # Check whether we deal with adam ETS or the conventional
    adamETS <- adamETSChecker(object);

    interval <- match.arg(interval[1],c("none", "simulated", "approximate", "semiparametric",
                                        "nonparametric", "confidence", "parametric","prediction",
                                        "empirical","complete"));
    # If the horizon is zero, just construct fitted and potentially confidence interval thingy
    if(h<=0){
        if(all(interval!=c("none","confidence"))){
            interval[] <- "prediction";
        }
        return(predict(object, newdata=newdata,
                       interval=interval,
                       level=level, side=side, ...));
    }
    else{
        if(interval=="confidence"){
            if(is.null(nsim)){
                nsim <- 100;
            }
            return(reforecast(object, h=h, newdata=newdata, occurrence=occurrence,
                              interval=interval, level=level, side=side, cumulative=cumulative,
                              nsim=nsim, ...));
        }
    }

    if(interval=="parametric"){
        interval <- "prediction";
    }
    else if(interval=="complete"){
        if(is.null(nsim)){
            nsim <- 100;
        }
        return(reforecast(object, h=h, newdata=newdata, occurrence=occurrence,
                          interval="prediction", level=level, side=side, cumulative=cumulative,
                          nsim=nsim, ...));
    }
    side <- match.arg(side);

    # If nsim is null, set it to 10000
    if(is.null(nsim)){
        nsim <- 10000;
    }

    # Model type
    model <- modelType(object);
    Etype <- errorType(object);
    Ttype <- trendType(object);
    damped <- substr(model,3,3)=="d";
    Stype <- seasonType(object);

    etsModel <- etsChecker(object);
    # arimaModel <- arimaChecker(object);
    # cesModel <- cesChecker(object);
    gumModel <- gumChecker(object);
    # ssarimaModel <- ssarimaChecker(object);

    # Technical parameters
    lagsModelAll <- modelLags(object);
    lagsModelMax <- max(lagsModelAll);
    # This is needed in order to see, whether h>m or not in seasonal models
    lagsModelMin <- lagsModelAll[lagsModelAll!=1];
    if(length(lagsModelMin)==0){
        lagsModelMin <- Inf;
    }
    else{
        lagsModelMin <- min(lagsModelMin);
    }
    profilesRecentTable <- object$profile;

    # Get componentsNumberETS and componentsNumberARIMA
    componentsDefined <- componentsDefiner(object);
    componentsNumberETS <- componentsDefined$componentsNumberETS;
    componentsNumberETSSeasonal <- componentsDefined$componentsNumberETSSeasonal;
    componentsNumberETSNonSeasonal <- componentsDefined$componentsNumberETSNonSeasonal;
    componentsNumberARIMA <- componentsDefined$componentsNumberARIMA;
    constantRequired <- componentsDefined$constantRequired;

    obsStates <- nrow(object$states);
    obsInSample <- nobs(object);

    yIndex <- time(actuals(object));
    yClasses <- class(actuals(object));
    # Create indices for the future
    if(any(yClasses=="ts")){
        # ts structure
        yForecastStart <- time(actuals(object))[obsInSample]+deltat(actuals(object));
        yFrequency <- frequency(actuals(object));
        yForecastIndex <- yIndex[obsInSample]+deltat(yIndex)*c(1:h);
    }
    else{
        # zoo
        yIndex <- time(actuals(object));
        yForecastIndex <- yIndex[obsInSample]+diff(tail(yIndex,2))*c(1:h);
    }

    # Get the lookup table
    indexLookupTable <- adamProfileCreator(lagsModelAll, lagsModelMax, obsInSample+h,
                                           lags(object), c(yIndex,yForecastIndex),
                                           yClasses)$lookup[,-c(1:(obsInSample+lagsModelMax)),drop=FALSE];

    # All the important matrices
    matVt <- t(object$states[obsStates-(lagsModelMax:1)+1,,drop=FALSE]);
    matWt <- tail(object$measurement,h);
    # If the forecast horizon is higher than the in-sample, duplicate the last value in matWt
    if(nrow(matWt)<h){
        matWt <- matrix(tail(matWt,1), nrow=h, ncol=ncol(matWt), dimnames=list(NULL,colnames(matWt)), byrow=TRUE);
    }
    vecG <- matrix(object$persistence, ncol=1);

    # Deal with explanatory variables
    if(ncol(object$data)>1){
        xregNumber <- length(object$initial$xreg);
        xregNames <- names(object$initial$xreg);
        # The newdata is not provided
        if(is.null(newdata) && ((!is.null(object$holdout) && nrow(object$holdout)<h) ||
                                is.null(object$holdout))){
            # Salvage what data we can (if there is something)
            if(!is.null(object$holdout)){
                hNeeded <- h-nrow(object$holdout);
                xreg <- tail(object$data,h);
                xreg[1:nrow(object$holdout),] <- object$holdout;
            }
            else{
                hNeeded <- h;
                xreg <- tail(object$data,h);
            }

            if(is.matrix(xreg)){
                warning("The newdata is not provided.",
                        "Predicting the explanatory variables based on what I have in-sample.",
                        call.=FALSE);
                for(i in 1:xregNumber){
                    xreg[,i] <- adam(object$data[,i+1],h=hNeeded,silent=TRUE)$forecast;
                }
            }
            else{
                warning("The newdata is not provided. Using last h in-sample observations instead.",
                        call.=FALSE);
            }
        }
        # The newdata is not provided, but we have holdout
        else if(is.null(newdata) && !is.null(object$holdout) && nrow(object$holdout)>=h){
            xreg <- object$holdout[1:h,,drop=FALSE];
        }
        # The newdata is provided
        else{
            # If this is not a matrix / data.frame, then convert to one
            if(!is.data.frame(newdata) && !is.matrix(newdata)){
                newdata <- as.data.frame(newdata);
                colnames(newdata) <- "xreg";
            }
            if(nrow(newdata)<h){
                warning(paste0("The newdata has ",nrow(newdata)," observations, while ",h," are needed. ",
                               "Using the last available values as future ones."),
                        call.=FALSE);
                newnRows <- h-nrow(newdata);
                xreg <- newdata[c(1:nrow(newdata),rep(nrow(newdata)),each=newnRows),];
                # xreg <- rbind(newdata,
                #               data.frame(matrix(rep(tail(newdata,1),each=newnRows),
                #                                 newnRows,ncol(newdata),
                #                                 dimnames=list(NULL,colnames(newdata))))
                #               );
            }
            else if(nrow(newdata)>h){
                warning(paste0("The newdata has ",nrow(newdata)," observations, while only ",h," are needed. ",
                               "Using the last ",h," of them."),
                        call.=FALSE);
                xreg <- tail(newdata,h);
            }
            else{
                xreg <- newdata;
            }

            if(any(is.na(xreg))){
                warning("The newdata has NAs. This might cause some issues.",
                        call.=FALSE);
            }
        }

        # If the user asked for trend, but it's not in the data, add it
        if(any(all.vars(formula(object))=="trend") && all(colnames(object$data)!="trend")){
            xreg <- cbind(xreg,trend=nobs(object)+c(1:h));
        }

        # If the names are wrong, transform to data frame and expand
        if(!all(xregNames %in% colnames(xreg)) && !is.data.frame(xreg)){
            xreg <- as.data.frame(xreg);
        }

        # Expand the xreg if it is data frame to get the proper matrix
        if(is.data.frame(xreg)){
            testFormula <- formula(object);
            # Remove response variable
            testFormula[[2]] <- NULL;
            colnames(xreg) <- make.names(colnames(xreg));
            # Expand the variables. We cannot use alm, because it is based on obsInSample
            xregData <- model.frame(testFormula,data=xreg);
            # Binary, flagging factors in the data
            # Expanded stuff with all levels for factors

            if(any((attr(terms(xregData),"dataClasses")=="factor"))){
                xregModelMatrix <- model.matrix(xregData,xregData,
                                                contrasts.arg=lapply(xregData[attr(terms(xregData),"dataClasses")=="factor"],
                                                                     contrasts, contrasts=FALSE));
            }
            else{
                xregModelMatrix <- model.matrix(xregData,data=xregData);
            }
            xregNames[] <- make.names(xregNames, unique=TRUE);
            colnames(xregModelMatrix) <- make.names(colnames(xregModelMatrix), unique=TRUE);
            newdata <- as.matrix(xregModelMatrix)[,xregNames,drop=FALSE];
            rm(xregData,xregModelMatrix);
        }
        else{
            colnames(xreg) <- make.names(colnames(xreg));
            newdata <- xreg[,xregNames,drop=FALSE];
        }
        rm(xreg);

        # From 1 to nrow to address potential missing values
        matWt[1:nrow(newdata),componentsNumberETS+componentsNumberARIMA+c(1:xregNumber)] <- newdata;
    }
    else{
        xregNumber <- 0;
        # If the user asked for trend, but it's not in the data, add it
        if(any(all.vars(formula(object))=="trend") && all(colnames(object$data)!="trend")){
            xreg <- matrix(nobs(object)+c(1:h),h,1);
            xregNumber <- 1;
        }
    }
    matF <- object$transition;

    # If this is "prediction", do simulations for multiplicative components
    if(interval=="prediction"){
        # Simulate stuff for the ETS only
        if((etsModel || xregNumber>0) &&
           (Ttype=="M" || (Stype=="M" & h>lagsModelMin))){
            interval <- "simulated";
        }
        else{
            interval <- "approximate";
        }
    }

    # Create C++ adam class, which will then use fit, forecast etc methods
    adamCpp <- object$adamCpp;
    # adamCpp <- new(adamCore,
    #                lagsModelAll, Etype, Ttype, Stype,
    #                componentsNumberETSNonSeasonal,
    #                componentsNumberETSSeasonal,
    #                componentsNumberETS, componentsNumberARIMA,
    #                xregNumber, length(lagsModelAll),
    #                constantRequired, adamETS);

    ##### Produce point forecasts, not mean forecasts from the model! #####
    # Produce point forecasts for non-multiplicative trend / seasonality
    # Do this for cases, when h<=m as well and prediction /confidence / simulated interval
    # if(Ttype!="M" && (Stype!="M" | (Stype=="M" & h<=lagsModelMin)) ||
    #    any(interval==c("nonparametric","semiparametric","empirical","approximate"))){
        adamForecast <- adamCpp$forecast(matWt, matF,
                                         indexLookupTable, profilesRecentTable,
                                         h)$forecast;
    # }
    # else{
    #     # If we do simulations, leave it for later
    #     if(interval=="simulated"){
    #         adamForecast <- rep(0, h);
    #     }
    #     # If we don't, do simulations to get mean
    #     else{
    #         adamForecast <- forecast(object, h=h, newdata=newdata, occurrence=occurrence,
    #                                  interval="simulated",
    #                                  level=level, side="both", cumulative=cumulative, nsim=nsim, ...)$mean;
    #     }
    # }

    #### Make safety checks
    # If there are NaN values
    if(any(is.nan(adamForecast))){
        adamForecast[is.nan(adamForecast)] <- 0;
    }

    # Make a warning about the potential explosive trend
    if(Ttype=="M" && !damped && profilesRecentTable[2,1]>1 && h>10){
        warning("Your model has a potentially explosive multiplicative trend. ",
                "I cannot do anything about it, so please just be careful.",
                call.=FALSE);
    }

    occurrenceModel <- FALSE;
    # If the occurrence values are provided for the holdout
    if(!is.null(occurrence) && is.logical(occurrence)){
        pForecast <- occurrence*1;
    }
    else if(!is.null(occurrence) && is.numeric(occurrence)){
        pForecast <- occurrence;
    }
    else{
        # If this is a mixture model, produce forecasts for the occurrence
        if(is.occurrence(object$occurrence)){
            occurrenceModel[] <- TRUE;
            if(object$occurrence$occurrence=="provided"){
                pForecast <- rep(1,h);
            }
            else{
                pForecast <- forecast(object$occurrence, h=h, newdata=newdata)$mean;
            }
        }
        else{
            occurrenceModel[] <- FALSE;
            # If this was provided occurrence, then use provided values
            if(!is.null(object$occurrence) && !is.null(object$occurrence$occurrence) &&
               (object$occurrence$occurrence=="provided") && !is.na(object$occurrence$forecast)){
                pForecast <- object$occurrence$forecast;
            }
            else{
                pForecast <- rep(1, h);
            }
        }
    }
    # Make sure that the values are of the correct length
    if(h<length(pForecast)){
        pForecast <- pForecast[1:h];
    }
    else if(h>length(pForecast)){
        pForecast <- c(pForecast, rep(tail(pForecast,1), h-length(pForecast)));
    }

    # How many levels did user asked to produce
    nLevels <- length(level);
    # Cumulative forecasts have only one observation
    if(cumulative){
        # hFinal is the number of elements we will have in the final forecast
        hFinal <- 1;
        # In case of occurrence model use simulations - the cumulative probability is a bitch
        if(occurrenceModel){
            interval[] <- "simulated";
        }
    }
    else{
        hFinal <- h;
    }

    # Create necessary matrices for the forecasts
    if(any(yClasses=="ts")){
        yForecast <- ts(vector("numeric", hFinal), start=yForecastStart, frequency=yFrequency);
        yUpper <- yLower <- ts(matrix(0,hFinal,nLevels), start=yForecastStart, frequency=yFrequency);
    }
    else{
        if(cumulative){
            yForecast <- zoo(vector("numeric", hFinal), order.by=yForecastIndex[1]);
            yUpper <- yLower <- zoo(matrix(0,hFinal,nLevels), order.by=yForecastIndex[1]);
        }
        else{
            yForecast <- zoo(vector("numeric", hFinal), order.by=yForecastIndex);
            yUpper <- yLower <- zoo(matrix(0,hFinal,nLevels), order.by=yForecastIndex);
        }
    }
    # Fill in the point forecasts
    if(cumulative){
        yForecast[] <- sum(as.vector(adamForecast) * as.vector(pForecast));
    }
    else{
        yForecast[] <- as.vector(adamForecast) * as.vector(pForecast);
    }

    if(interval!="none"){
        # Fix just in case a silly user used 95 etc instead of 0.95
        if(any(level>1)){
            level[] <- level / 100;
        }
        levelLow <- levelUp <- matrix(0,nrow=hFinal,ncol=nLevels);
        levelNew <- matrix(level,nrow=hFinal,ncol=nLevels,byrow=TRUE);

        # If this is an occurrence model, then take probability into account in the level.
        # This correction is only needed for approximate, because the others contain zeroes
        if(occurrenceModel && interval=="approximate"){
            levelNew[] <- (levelNew-(1-as.vector(pForecast)))/as.vector(pForecast);
            levelNew[levelNew<0] <- 0;
        }
        if(side=="both"){
            levelLow[] <- (1-levelNew)/2;
            levelUp[] <- (1+levelNew)/2;
        }
        else if(side=="upper"){
            levelLow[] <- 0;
            levelUp[] <- levelNew;
        }
        else{
            levelLow[] <- 1-levelNew;
            levelUp[] <- 1;
        }
        levelLow[levelLow<0] <- 0;
        levelUp[levelUp<0] <- 0;
    }

    #### Simulated interval ####
    if(interval=="simulated"){
        arrVt <- array(NA, c(componentsNumberETS+componentsNumberARIMA+xregNumber+constantRequired, h+lagsModelMax, nsim));
        arrVt[,1:lagsModelMax,] <- rep(matVt,nsim);
        # Number of degrees of freedom to de-bias scales
        df <- (nobs(object, all=FALSE)-nparam(object));
        # If the sample is too small, then use biased estimator
        if(df<=0){
            df[] <- nobs(object, all=FALSE);
        }
        # If scale model is included, produce forecasts
        if(is.scale(object$scale)){
            # as.vector is needed to declass the mean.
            scaleValue <- as.vector(forecast(object$scale,h=h,newdata=newdata,interval="none")$mean);
            # De-bias the scales and transform to the appropriate scale
            # dnorm, dlnorm fit model on square residuals
            # dgnorm needs to be done with ^beta to get to 1/T part
            # The rest do not require transformations, only de-bias
            scaleValue[] <- switch(object$distribution,
                                   "dlnorm"=,
                                   "dnorm"=(scaleValue*obsInSample/df)^0.5,
                                   "dgnorm"=((scaleValue^object$other$shape)*obsInSample/df)^{1/object$other$shape},
                                   scaleValue*obsInSample/df);
        }
        else{
            scaleValue <- object$scale*obsInSample/df;
        }
        matErrors <- matrix(switch(object$distribution,
                                   "dnorm"=rnorm(h*nsim, 0, scaleValue),
                                   "dlaplace"=rlaplace(h*nsim, 0, scaleValue),
                                   "ds"=rs(h*nsim, 0, scaleValue),
                                   "dgnorm"=rgnorm(h*nsim, 0, scaleValue, object$other$shape),
                                   "dlogis"=rlogis(h*nsim, 0, scaleValue),
                                   "dt"=rt(h*nsim, obsInSample-nparam(object)),
                                   "dalaplace"=ralaplace(h*nsim, 0, scaleValue, object$other$alpha),
                                   "dlnorm"=rlnorm(h*nsim, -scaleValue^2/2, scaleValue)-1,
                                   "dinvgauss"=rinvgauss(h*nsim, 1, dispersion=scaleValue)-1,
                                   "dgamma"=rgamma(h*nsim, shape=scaleValue^{-1}, scale=scaleValue)-1,
                                   "dllaplace"=exp(rlaplace(h*nsim, 0, scaleValue))-1,
                                   "dls"=exp(rs(h*nsim, 0, scaleValue))-1,
                                   "dlgnorm"=exp(rgnorm(h*nsim, 0, scaleValue, object$other$shape))-1
        ),
        h,nsim);
        # Normalise errors in order not to get ridiculous things on small nsim
        if(nsim<=500){
            if(Etype=="A"){
                matErrors[] <- matErrors - array(apply(matErrors,1,mean),c(h,nsim));
            }
            else{
                matErrors[] <- (1+matErrors) / array(apply(1+matErrors,1,mean),c(h,nsim))-1;
            }
        }
        # This stuff is needed in order to produce adequate values for weird models
        EtypeModified <- Etype;
        if(Etype=="A" && any(object$distribution==c("dlnorm","dinvgauss","dgamma","dls","dllaplace"))){
            EtypeModified[] <- "M";
        }

        # States, Errors, Ot, Transition, Measurement, Persistence
        ySimulated <- adamCpp$simulate(matErrors, matrix(rbinom(h*nsim, 1, pForecast), h, nsim),
                                       arrVt, matWt,
                                       array(matF,c(dim(matF),nsim)),
                                       matrix(vecG, componentsNumberETS+componentsNumberARIMA+
                                                  xregNumber+constantRequired, nsim),
                                       indexLookupTable,
                                       array(profilesRecentTable,
                                             c(componentsNumberETS+componentsNumberARIMA+xregNumber+constantRequired,
                                               lagsModelMax,
                                               nsim)),
                                       EtypeModified)$data;

        #### Note that the cumulative doesn't work with oes at the moment!
        if(cumulative){
            yForecast[] <- mean(colSums(ySimulated,na.rm=TRUE));
            yLower[] <- quantile(colSums(ySimulated,na.rm=TRUE),levelLow,type=7);
            yUpper[] <- quantile(colSums(ySimulated,na.rm=TRUE),levelUp,type=7);
        }
        else{
            for(i in 1:h){
                ##### We now do point forecasts, not mean #####
                # if(Ttype=="M" || (Stype=="M" & h>lagsModelMin)){
                #     # Trim 1% of values just to resolve some issues with outliers
                #     yForecast[i] <- mean(ySimulated[i,],na.rm=TRUE,trim=0.01);
                # }
                yLower[i,] <- quantile(ySimulated[i,],levelLow[i,],na.rm=TRUE,type=7);
                yUpper[i,] <- quantile(ySimulated[i,],levelUp[i,],na.rm=TRUE,type=7);
            }
        }
        # This step is needed in order to make intervals similar between the different methods
        if(Etype=="A"){
            yLower[] <- yLower - yForecast;
            yUpper[] <- yUpper - yForecast;
        }
        else{
            yLower[] <- yLower / yForecast;
            yUpper[] <- yUpper / yForecast;
            # Substitute NaNs with zeroes - it means that both values were originally zeroes
            yLower[as.vector(is.nan(yLower))] <- 0;
            yUpper[as.vector(is.nan(yUpper))] <- 0;
        }
    }
    else{
        #### Approximate and confidence interval ####
        # Produce covariance matrix and use it
        if(any(interval=="approximate")){
            # The variance of the model
            s2 <- sigma(object)^2;
            # If scale model is included, produce forecasts
            if(is.scale(object$scale)){
                # Number of degrees of freedom to de-bias the variance
                df <- (nobs(object, all=FALSE)-nparam(object));
                # If the sample is too small, then use biased estimator
                if(df<=0){
                    df[] <- nobs(object, all=FALSE);
                }
                s2Forecast <- forecast(object$scale,h=h,newdata=newdata,interval="none")$mean;
                # Transform scales into the variances
                # dnorm, dlnorm, dgamma and dinvgauss return scales that are equal to variances
                s2Forecast[] <- switch(object$distribution,
                                       "dlaplace"=2*s2Forecast^2,
                                       "ds"=120*s2Forecast^4,
                                       "dgnorm"=s2Forecast^2*gamma(3/object$other$shape)/gamma(1/object$other$shape),
                                       "dalaplace"=s2Forecast^2/(object$other$alpha^2*(1-object$other$alpha)^2/
                                                                     (object$other$alpha^2+(1-object$other$alpha)^2)),
                                       s2Forecast)*obsInSample/df;
            }
            # IG and Lnorm can use approximations from the multiplications
            if(etsModel && any(object$distribution==c("dinvgauss","dgamma","dlnorm","dllaplace","dls","dlgnorm")) && Etype=="M"){


                vcovMulti <- adamVarAnal(lagsModelAll, h, matWt[1,,drop=FALSE], matF, vecG, s2);
                #print(vcovMulti)
                if(is.scale(object$scale)){
                    # Fix the matrix with the time varying variance
                    vcovMulti[] <- vcovMulti / s2 * (sqrt(s2Forecast) %*% t(sqrt(s2Forecast)));
                }
                if(any(object$distribution==c("dlnorm","dls","dllaplace","dlgnorm"))){
                    vcovMulti[] <- log(1+vcovMulti);
                }

                # We don't do correct cumulatives in this case...
                if(cumulative){
                    vcovMulti <- sum(vcovMulti);
                }
            }
            else{
                vcovMulti <- covarAnal(lagsModelAll, h, matWt[1,,drop=FALSE], matF, vecG, s2);
                if(is.scale(object$scale)){
                    # Fix the matrix with the time varying variance
                    vcovMulti[] <- vcovMulti / s2 * (sqrt(s2Forecast) %*% t(sqrt(s2Forecast)));
                }

                # Do either the variance of sum, or a diagonal
                if(cumulative){
                    vcovMulti <- sum(vcovMulti);
                }
                else{
                    vcovMulti <- diag(vcovMulti);
                }
            }

        }
        #### Semiparametric, nonparametric and empirical interval ####
        # Extract multistep errors and calculate the covariance matrix
        else if(any(interval==c("semiparametric","nonparametric","empirical"))){
            if(h>1){
                adamErrors <- as.matrix(rmultistep(object, h=h));

                if(any(object$distribution==c("dinvgauss","dgamma","dlnorm","dls","dllaplace","dlgnorm")) && (Etype=="A")){
                    yFittedMatrix <- adamErrors;
                    for(i in 1:h){
                        yFittedMatrix[,i] <- fitted(object)[1:(obsInSample-h)+i];
                    }
                    adamErrors[] <- adamErrors/yFittedMatrix;
                }

                if(interval=="semiparametric"){
                    # Do either the variance of sum, or a diagonal
                    if(cumulative){
                        vcovMulti <- sum(t(adamErrors) %*% adamErrors / (obsInSample-h));
                    }
                    else{
                        vcovMulti <- diag(t(adamErrors) %*% adamErrors / (obsInSample-h));
                    }
                }
                # For nonparametric and cumulative...
                else{
                    if(cumulative){
                        adamErrors <- matrix(apply(adamErrors, 2, sum),obsInSample-h,1);
                    }
                }
            }
            else{
                # If scale model is included, produce forecasts
                if(is.scale(object$scale)){
                    # Number of degrees of freedom to de-bias the variance
                    df <- (nobs(object, all=FALSE)-nparam(object));
                    # If the sample is too small, then use biased estimator
                    if(df<=0){
                        df[] <- nobs(object, all=FALSE);
                    }
                    vcovMulti <- forecast(object$scale,h=h,newdata=newdata,interval="none")$mean;
                    # Transform scales into the variances
                    # dnorm, dlnorm, dgamma and dinvgauss return scales that are equal to variances
                    vcovMulti[] <- switch(object$distribution,
                                          "dlaplace"=2*vcovMulti^2,
                                          "ds"=120*vcovMulti^4,
                                          "dgnorm"=vcovMulti^2*gamma(3/object$other$shape)/gamma(1/object$other$shape),
                                          "dalaplace"=vcovMulti^2/(object$other$alpha^2*(1-object$other$alpha)^2/
                                                                       (object$other$alpha^2+(1-object$other$alpha)^2)),
                                          vcovMulti)*obsInSample/df;
                }
                else{
                    vcovMulti <- sigma(object)^2;
                }
                adamErrors <- as.matrix(residuals(object));
            }
        }

        # Calculate interval for approximate and semiparametric
        if(any(interval==c("approximate","semiparametric"))){
            if(object$distribution=="dnorm"){
                if(Etype=="A"){
                    yLower[] <- qnorm(levelLow, 0, sqrt(vcovMulti));
                    yUpper[] <- qnorm(levelUp, 0, sqrt(vcovMulti));
                }
                else{
                    yLower[] <- qnorm(levelLow, 1, sqrt(vcovMulti));
                    yUpper[] <- qnorm(levelUp, 1, sqrt(vcovMulti));
                }
            }
            else if(object$distribution=="dlaplace"){
                if(Etype=="A"){
                    yLower[] <- qlaplace(levelLow, 0, sqrt(vcovMulti/2));
                    yUpper[] <- qlaplace(levelUp, 0, sqrt(vcovMulti/2));
                }
                else{
                    yLower[] <- qlaplace(levelLow, 1, sqrt(vcovMulti/2));
                    yUpper[] <- qlaplace(levelUp, 1, sqrt(vcovMulti/2));
                }
            }
            else if(object$distribution=="ds"){
                if(Etype=="A"){
                    yLower[] <- qs(levelLow, 0, (vcovMulti/120)^0.25);
                    yUpper[] <- qs(levelUp, 0, (vcovMulti/120)^0.25);
                }
                else{
                    yLower[] <- qs(levelLow, 1, (vcovMulti/120)^0.25);
                    yUpper[] <- qs(levelUp, 1, (vcovMulti/120)^0.25);
                }
            }
            else if(object$distribution=="dgnorm"){
                scale <- sqrt(vcovMulti*(gamma(1/object$other$shape)/gamma(3/object$other$shape)));
                if(Etype=="A"){
                    yLower[] <- suppressWarnings(qgnorm(levelLow, 0, scale, object$other$shape));
                    yUpper[] <- suppressWarnings(qgnorm(levelUp, 0, scale, object$other$shape));
                }
                else{
                    yLower[] <- suppressWarnings(qgnorm(levelLow, 1, scale, object$other$shape));
                    yUpper[] <- suppressWarnings(qgnorm(levelUp, 1, scale, object$other$shape));
                }
            }
            else if(object$distribution=="dlogis"){
                if(Etype=="A"){
                    yLower[] <- qlogis(levelLow, 0, sqrt(vcovMulti*3)/pi);
                    yUpper[] <- qlogis(levelUp, 0, sqrt(vcovMulti*3)/pi);
                }
                else{
                    yLower[] <- qlogis(levelLow, 1, sqrt(vcovMulti*3)/pi);
                    yUpper[] <- qlogis(levelUp, 1, sqrt(vcovMulti*3)/pi);
                }
            }
            else if(object$distribution=="dt"){
                df <- nobs(object) - nparam(object);
                if(Etype=="A"){
                    yLower[] <- sqrt(vcovMulti)*qt(levelLow, df);
                    yUpper[] <- sqrt(vcovMulti)*qt(levelUp, df);
                }
                else{
                    yLower[] <- (1 + sqrt(vcovMulti)*qt(levelLow, df));
                    yUpper[] <- (1 + sqrt(vcovMulti)*qt(levelUp, df));
                }
            }
            else if(object$distribution=="dalaplace"){
                alpha <- object$other$alpha;
                if(Etype=="A"){
                    yLower[] <- qalaplace(levelLow, 0,
                                          sqrt(vcovMulti*alpha^2*(1-alpha)^2/(alpha^2+(1-alpha)^2)), alpha);
                    yUpper[] <- qalaplace(levelUp, 0,
                                          sqrt(vcovMulti*alpha^2*(1-alpha)^2/(alpha^2+(1-alpha)^2)), alpha);
                }
                else{
                    yLower[] <- qalaplace(levelLow, 1,
                                          sqrt(vcovMulti*alpha^2*(1-alpha)^2/(alpha^2+(1-alpha)^2)), alpha);
                    yUpper[] <- qalaplace(levelUp, 1,
                                          sqrt(vcovMulti*alpha^2*(1-alpha)^2/(alpha^2+(1-alpha)^2)), alpha);
                }
            }
            else if(object$distribution=="dlnorm"){
                yLower[] <- qlnorm(levelLow, sqrt(abs(1-vcovMulti))-1, sqrt(vcovMulti));
                yUpper[] <- qlnorm(levelUp, sqrt(abs(1-vcovMulti))-1, sqrt(vcovMulti));
                if(Etype=="A"){
                    yLower[] <- (yLower-1)*yForecast;
                    yUpper[] <-(yUpper-1)*yForecast;
                }
            }
            else if(object$distribution=="dllaplace"){
                yLower[] <- exp(qlaplace(levelLow, 0, sqrt(vcovMulti/2)));
                yUpper[] <- exp(qlaplace(levelUp, 0, sqrt(vcovMulti/2)));
                if(Etype=="A"){
                    yLower[] <- (yLower-1)*yForecast;
                    yUpper[] <-(yUpper-1)*yForecast;
                }
            }
            else if(object$distribution=="dls"){
                yLower[] <- exp(qs(levelLow, 0, (vcovMulti/120)^0.25));
                yUpper[] <- exp(qs(levelUp, 0, (vcovMulti/120)^0.25));
                if(Etype=="A"){
                    yLower[] <- (yLower-1)*yForecast;
                    yUpper[] <-(yUpper-1)*yForecast;
                }
            }
            else if(object$distribution=="dlgnorm"){
                scale <- sqrt(vcovMulti*(gamma(1/object$other$shape)/gamma(3/object$other$shape)));
                yLower[] <- suppressWarnings(exp(qgnorm(levelLow, 0, scale, object$other$shape)));
                yUpper[] <- suppressWarnings(exp(qgnorm(levelUp, 0, scale, object$other$shape)));
                if(Etype=="A"){
                    yLower[] <- (yLower-1)*yForecast;
                    yUpper[] <-(yUpper-1)*yForecast;
                }
            }
            else if(object$distribution=="dinvgauss"){
                yLower[] <- qinvgauss(levelLow, 1, dispersion=vcovMulti);
                yUpper[] <- qinvgauss(levelUp, 1, dispersion=vcovMulti);
                if(Etype=="A"){
                    yLower[] <- (yLower-1)*yForecast;
                    yUpper[] <-(yUpper-1)*yForecast;
                }
            }
            else if(object$distribution=="dgamma"){
                yLower[] <- qgamma(levelLow, shape=1/vcovMulti, scale=vcovMulti);
                yUpper[] <- qgamma(levelUp, shape=1/vcovMulti, scale=vcovMulti);
                if(Etype=="A"){
                    yLower[] <- (yLower-1)*yForecast;
                    yUpper[] <-(yUpper-1)*yForecast;
                }
            }

        }

        # Empirical, based on specific quantiles
        else if(interval=="empirical"){
            for(i in 1:h){
                yLower[i,] <- quantile(adamErrors[,i],levelLow[i,],na.rm=TRUE,type=7);
                yUpper[i,] <- quantile(adamErrors[,i],levelUp[i,],na.rm=TRUE,type=7);
            }

            if(Etype=="M"){
                yLower[] <- 1+yLower;
                yUpper[] <- 1+yUpper;
            }
            else if(Etype=="A" & any(object$distribution==c("dinvgauss","dgamma","dlnorm","dllaplace","dls","dlgnorm"))){
                yLower[] <- yLower*yForecast;
                yUpper[] <- yUpper*yForecast;
            }
        }
        # Use Taylor & Bunn approach for the nonparametric ones
        #### Nonparametric intervals, regression ####
        else if(interval=="nonparametric"){
            if(h>1){
                # This is needed in order to see if quant regression can be used
                if(all(levelLow==levelLow[1,])){
                    levelLow <- levelLow[1,,drop=FALSE];
                }
                if(all(levelUp==levelUp[1,])){
                    levelUp <- levelUp[1,,drop=FALSE];
                }

                # Do quantile regression for h>1 and scalars for the level (no change across h)
                # transpose is needed in order to compare correctly
                if(all(t(levelNew)==levelNew[1,])){
                    # Quantile regression function
                    intervalQuantile <- function(A, alpha){
                        ee[] <- adamErrors - (A[1]*xe^A[2]);
                        return((1-alpha)*sum(abs(ee[ee<0]))+alpha*sum(abs(ee[ee>=0])));
                    }

                    ee <- adamErrors;
                    xe <- matrix(c(1:h),nrow=nrow(ee),ncol=ncol(ee),byrow=TRUE);

                    for(i in 1:nLevels){
                        # lower quantiles
                        A <- nlminb(rep(1,2),intervalQuantile,alpha=levelLow[1,i])$par;
                        yLower[,i] <- A[1]*c(1:h)^A[2];

                        # upper quantiles
                        A[] <- nlminb(rep(1,2),intervalQuantile,alpha=levelUp[1,i])$par;
                        yUpper[,i] <- A[1]*c(1:h)^A[2];
                    }
                }
                # Otherwise just return quantiles of errors
                else{
                    if(cumulative){
                        yLower[] <- quantile(adamErrors,levelLow,type=7);
                        yUpper[] <- quantile(adamErrors,levelUp,type=7);
                    }
                    else{
                        for(i in 1:h){
                            yLower[i] <- quantile(adamErrors[,i],levelLow[i],na.rm=TRUE,type=7);
                            yUpper[i] <- quantile(adamErrors[,i],levelUp[i],na.rm=TRUE,type=7);
                        }
                    }
                }
            }
            else{
                yLower[] <- quantile(adamErrors,levelLow,type=7);
                yUpper[] <- quantile(adamErrors,levelUp,type=7);
            }

            if(Etype=="M"){
                yLower[] <- 1+yLower;
                yUpper[] <- 1+yUpper;
            }
            else if(Etype=="A" & any(object$distribution==c("dinvgauss","dgamma","dlnorm","dllaplace","dls","dlgnorm"))){
                yLower[] <- yLower*yForecast;
                yUpper[] <- yUpper*yForecast;
            }
        }
        else{
            yUpper[] <- yLower[] <- NA;
        }
    }



    # Fix of prediction intervals depending on what has happened
    if(interval!="none"){
        # Make sensible values out of those weird quantiles
        if(!cumulative){
            if(any(levelLow==0)){
                # zoo does not like, when you work with matrices of indices... silly thing
                yBoundBuffer <- levelLow;
                yBoundBuffer[] <- yLower
                if(Etype=="A"){
                    yBoundBuffer[levelLow==0] <- -Inf;
                    yLower[] <- yBoundBuffer;
                }
                else{
                    yBoundBuffer[levelLow==0] <- 0;
                    yLower[] <- yBoundBuffer;
                }
            }
            if(any(levelUp==1)){
                # zoo does not like, when you work with matrices of indices... silly thing
                yBoundBuffer <- levelUp;
                yBoundBuffer[] <- yUpper
                yBoundBuffer[levelUp==1] <- Inf;
                yUpper[] <- yBoundBuffer;
            }
        }
        else{
            if(Etype=="A" && any(levelLow==0)){
                yLower[] <- -Inf;
            }
            else if(Etype=="M" && any(levelLow==0)){
                yLower[] <- 0;
            }
            if(any(levelUp==1)){
                yUpper[] <- Inf;
            }
        }

        # Substitute NAs and NaNs with zeroes
        if(any(is.nan(yLower)) || any(is.na(yLower))){
            yLower[is.nan(yLower)] <- switch(Etype,"A"=0,"M"=1);
            yLower[is.na(yLower)] <- switch(Etype,"A"=0,"M"=1);
        }
        if(any(is.nan(yUpper)) || any(is.na(yUpper))){
            yUpper[is.nan(yUpper)] <- switch(Etype,"A"=0,"M"=1);
            yUpper[is.na(yUpper)] <- switch(Etype,"A"=0,"M"=1);
        }

        # Do intervals around the forecasts...
        if(Etype=="A"){
            yLower[] <- yForecast + yLower;
            yUpper[] <- yForecast + yUpper;
        }
        else{
            yLower[] <- yForecast*yLower;
            yUpper[] <- yForecast*yUpper;
        }

        # Check what we have from the occurrence model
        if(occurrenceModel){
            # If there are NAs, then there's no variability and no intervals.
            if(any(is.na(yUpper))){
                yUpper[is.na(yUpper)] <- (yForecast/pForecast)[is.na(yUpper)];
            }
            if(any(is.na(yLower))){
                yLower[is.na(yLower)] <- 0;
            }
        }

        colnames(yLower) <- switch(side,
                                   "both"=paste0("Lower bound (",(1-level)/2*100,"%)"),
                                   "lower"=paste0("Lower bound (",(1-level)*100,"%)"),
                                   "upper"=rep("Lower 0%",nLevels));

        colnames(yUpper) <- switch(side,
                                   "both"=paste0("Upper bound (",(1+level)/2*100,"%)"),
                                   "lower"=rep("Upper 100%",nLevels),
                                   "upper"=paste0("Upper bound (",level*100,"%)"));
    }



    # If this was a model in logarithms (e.g. ARIMA for sm), then take exponent
    if(any(unlist(gregexpr("in logs",object$model))!=-1)){
        yForecast[] <- exp(yForecast);
        yLower[] <- exp(yLower);
        yUpper[] <- exp(yUpper);
    }

    if(!scenarios){
        ySimulated <- scenarios;
    }
    else{
        if(interval=="simulated"){
            colnames(ySimulated) <- paste0("nsim",1:nsim);
            rownames(ySimulated) <- paste0("h",1:h);
        }
        else{
            warning("Scenarios are only available when interval=\"simulated\".",
                    call.=FALSE);
            ySimulated <- FALSE;
        }
    }

    # Fix values if we have a log-model in case of GUM
    if(gumModel && object$type=="multiplicative"){
        yForecast[] <- exp(yForecast);
        yLower[] <- exp(yLower);
        yUpper[] <- exp(yUpper);

    }


    return(structure(list(mean=yForecast, lower=yLower, upper=yUpper, model=object,
                          level=level, interval=interval, side=side, cumulative=cumulative, h=h,
                          scenarios=ySimulated),
                     class=c("adam.forecast","smooth.forecast","forecast")));
}

#' @export
forecast.adamCombined <- function(object, h=10, newdata=NULL,
                                  interval=c("none", "prediction", "confidence", "simulated",
                                             "approximate", "semiparametric", "nonparametric",
                                             "empirical","complete"),
                                  level=0.95, side=c("both","upper","lower"), cumulative=FALSE, nsim=NULL, ...){

    interval <- match.arg(interval[1],c("none", "simulated", "approximate", "semiparametric",
                                        "nonparametric", "confidence", "parametric","prediction",
                                        "empirical","complete"));
    side <- match.arg(side);

    yClasses <- class(actuals(object));
    obsInSample <- nobs(object);

    if(any(yClasses=="ts")){
        # ts structure
        yForecastStart <- time(actuals(object))[obsInSample]+deltat(actuals(object));
        yFrequency <- frequency(actuals(object));
    }
    else{
        # zoo thingy
        yIndex <- time(actuals(object));
        yForecastIndex <- yIndex[obsInSample]+diff(tail(yIndex,2))*c(1:h);
    }

    # How many levels did user asked to produce
    nLevels <- length(level);
    # Cumulative forecasts have only one observation
    if(cumulative){
        # hFinal is the number of elements we will have in the final forecast
        hFinal <- 1;
    }
    else{
        hFinal <- h;
    }

    # Create necessary matrices for the forecasts
    if(any(yClasses=="ts")){
        yForecast <- ts(vector("numeric", hFinal), start=yForecastStart, frequency=yFrequency);
        yUpper <- yLower <- ts(matrix(0,hFinal,nLevels), start=yForecastStart, frequency=yFrequency);
    }
    else{
        yForecast <- zoo(vector("numeric", hFinal), order.by=yForecastIndex);
        yUpper <- yLower <- zoo(matrix(0,hFinal,nLevels), order.by=yForecastIndex);
    }

    # Remove ICw, which are lower than 0.001
    object$ICw[object$ICw<1e-2] <- 0;
    object$ICw[] <- object$ICw / sum(object$ICw);

    # The list contains 10 elements
    adamForecasts <- vector("list", 10);
    names(adamForecasts)[c(1:3)] <- c("mean","lower","upper");
    for(i in 1:length(object$models)){
        if(object$ICw[i]==0){
            next;
        }
        adamForecasts[] <- forecast.adam(object$models[[i]], h=h, newdata=newdata,
                                         interval=interval,
                                         level=level, side=side, cumulative=cumulative, nsim=nsim, ...);
        yForecast[] <- yForecast + adamForecasts$mean * object$ICw[i];
        yUpper[] <- yUpper + adamForecasts$upper * object$ICw[i];
        yLower[] <- yLower + adamForecasts$lower * object$ICw[i];
    }

    # Fix the names of the columns
    if(interval!="none"){
        colnames(yLower) <- colnames(adamForecasts$lower);
        colnames(yUpper) <- colnames(adamForecasts$upper);
    }

    # Fix the content of upper / lower bounds
    if(side=="upper"){
        yLower[] <- -Inf;
    }
    else if(side=="lower"){
        yUpper[] <- Inf;
    }

    # Get rid of specific models to save RAM
    object$models <- NULL;

    return(structure(list(mean=yForecast, lower=yLower, upper=yUpper, model=object,
                          level=level, interval=interval, side=side, cumulative=cumulative, h=h),
                     class=c("adam.forecast","smooth.forecast","forecast")));
}

# Failsafe for now. Ideally, should be removed
forecast.smooth <- forecast.adam;

#' @export
print.adam.forecast <- function(x, ...){
    if(x$interval!="none"){
        returnedValue <- switch(x$side,
                                "both"=cbind(x$mean,x$lower,x$upper),
                                "lower"=cbind(x$mean,x$lower),
                                "upper"=cbind(x$mean,x$upper));
        colnames(returnedValue) <- switch(x$side,
                                          "both"=c("Point forecast",colnames(x$lower),colnames(x$upper)),
                                          "lower"=c("Point forecast",colnames(x$lower)),
                                          "upper"=c("Point forecast",colnames(x$upper)))
    }
    else{
        returnedValue <- x$mean;
    }
    print(returnedValue);
}

#' @export
plot.adam.forecast <- function(x, ...){
    yClasses <- class(actuals(x));
    digits <- 2;

    ellipsis <- list(...);

    if(is.null(ellipsis$legend)){
        ellipsis$legend <- FALSE;
        ellipsis$parReset <- FALSE;
    }

    if(is.null(ellipsis$main)){
        distrib <- switch(x$model$distribution,
                          "dnorm" = "Normal",
                          "dlogis" = "Logistic",
                          "dlaplace" = "Laplace",
                          "ds" = "S",
                          "dgnorm" = paste0("Generalised Normal with shape=",round(x$model$other$shape,digits)),
                          "dalaplace" = paste0("Asymmetric Laplace with alpha=",round(x$model$other$alpha,digits)),
                          "dt" = paste0("Student t with df=",round(x$model$other$nu, digits)),
                          "dlnorm" = "Log-Normal",
                          "dllaplace" = "Log-Laplace",
                          "dls" = "Log-S",
                          "dgnorm" = paste0("Log-Generalised Normal with shape=",round(x$model$other$shape,digits)),
                          # "dbcnorm" = paste0("Box-Cox Normal with lambda=",round(x$other$lambda,2)),
                          "dinvgauss" = "Inverse Gaussian",
                          "dgamma" = "Gamma",
                          "default"
        );
        ellipsis$main <- paste0("Forecast from ",x$model$model," with ",distrib," distribution");
    }

    if(!is.null(x$model$holdout)){
        responseName <- all.vars(formula(x$model))[1];
        yHoldout <- x$model$holdout[,responseName];
        if(any(yClasses=="ts")){
            ellipsis$actuals <- ts(c(actuals(x$model),yHoldout),
                                   start=start(actuals(x$model)),
                                   frequency=frequency(actuals(x$model)));
        }
        else{
            ellipsis$actuals <- zoo(c(as.vector(actuals(x$model)),as.vector(yHoldout)),
                                    order.by=c(time(actuals(x$model)),time(yHoldout)));
        }
    }
    else{
        ellipsis$actuals <- actuals(x$model);
    }

    ellipsis$forecast <- x$mean;
    ellipsis$lower <- x$lower;
    ellipsis$upper <- x$upper;
    ellipsis$fitted <- fitted(x);
    ellipsis$level <- x$level;

    if(x$cumulative){
        # Create cumulative actuals and fitted values
        obsInSample <- nobs(x$model);
        obsNew <- floor(obsInSample/x$h);
        yCum <- yFittedCum <- vector("numeric", obsNew);
        for(i in 1:obsNew){
            yCum[i] <- sum(ellipsis$actuals[obsInSample-(x$h:1)+1-(obsNew-i)*x$h], na.rm=TRUE);
            yFittedCum[i] <- sum(ellipsis$fitted[obsInSample-(x$h:1)+1-(obsNew-i)*x$h], na.rm=TRUE);
        }

        # Get deltat to see where to place forecast
        if(!is.null(x$model$holdout)){
            yDeltat <- 0;
            yCum <- c(yCum, sum(ellipsis$actuals[obsInSample+(1:x$h)], na.rm=TRUE));
        }
        else{
            yDeltat <- deltat(ellipsis$actuals);
        }
        yFreqCum <- frequency(ellipsis$actuals)/x$h;

        # Form new ts objects based on the cumulative values
        # We ignore zoo here because who cares...
        ellipsis$actuals <- ts(yCum,
                               start=start(ellipsis$actuals),
                               frequency=yFreqCum);
        ellipsis$fitted <- ts(yFittedCum,
                              start=start(ellipsis$actuals),
                              frequency=yFreqCum);
        ellipsis$forecast <- ts(ellipsis$forecast,
                                start=end(ellipsis$actuals)+yDeltat,
                                frequency=frequency(ellipsis$forecast));
        ellipsis$lower <- ts(ellipsis$lower,
                             start=end(ellipsis$actuals)+yDeltat,
                             frequency=frequency(ellipsis$forecast));
        ellipsis$upper <- ts(ellipsis$upper,
                             start=end(ellipsis$actuals)+yDeltat,
                             frequency=frequency(ellipsis$forecast));
        ellipsis$main <- paste0("Cumulative ", ellipsis$main);
        ellipsis$vline <- FALSE;

        # if(any(yClasses=="ts")){
        #     ellipsis$forecast <- ts(ellipsis$forecast / x$h,
        #                             start=start(ellipsis$forecast),
        #                             frequency=frequency(ellipsis$forecast));
        #     ellipsis$lower <- ts(ellipsis$lower / x$h,
        #                          start=start(ellipsis$lower),
        #                          frequency=frequency(ellipsis$lower));
        #     ellipsis$upper <- ts(ellipsis$upper / x$h,
        #                          start=start(ellipsis$upper),
        #                          frequency=frequency(ellipsis$upper));
        #     ellipsis$main <- paste0("Mean ", ellipsis$main);
        # }
        # else{
        #     ellipsis$forecast <- zoo(ellipsis$forecast / x$h,
        #                              order.by=time(ellipsis$forecast)+c(1:x$h)-1);
        #     ellipsis$lower <- zoo(ellipsis$lower / x$h,
        #                           order.by=time(ellipsis$lower)+c(1:x$h)-1);
        #     ellipsis$upper <- zoo(ellipsis$upper / x$h,
        #                           order.by=time(ellipsis$upper)+c(1:x$h)-1);
        #     ellipsis$main <- paste0("Mean ", ellipsis$main);
        #     ellipsis$actuals <- zoo(c(as.vector(actuals(x$model)),as.vector(yHoldout)),
        #                             order.by=c(time(actuals(x$model)),time(yHoldout)));
        # }
    }

    do.call(graphmaker, ellipsis);

    # A fix for weird frequencies for the cumulative forecasts
    if(x$cumulative){
        points(ellipsis$actuals);
        abline(v=tail(time(ellipsis$fitted),1),col=2,lwd=2);
    }
}


#### Other methods ####

#' @export
multicov.adam <- function(object, type=c("analytical","empirical","simulated"), h=10, nsim=1000,
                          ...){
    type <- match.arg(type);

    # Check whether we deal with adam ETS or the conventional
    adamETS <- adamETSChecker(object);

    # Model type
    Ttype <- substr(modelType(object),2,2);

    lagsModelAll <- modelLags(object);
    lagsModelMax <- max(lagsModelAll);
    lagsOriginal <- lags(object);
    if(Ttype!="N"){
        lagsOriginal <- c(1,lagsOriginal);
    }

    # Get componentsNumberETS, seasonal and componentsNumberARIMA
    componentsDefined <- componentsDefiner(object);
    componentsNumberETS <- componentsDefined$componentsNumberETS;
    componentsNumberETSSeasonal <- componentsDefined$componentsNumberETSSeasonal;
    componentsNumberETSNonSeasonal <- componentsDefined$componentsNumberETSNonSeasonal;
    componentsNumberARIMA <- componentsDefined$componentsNumberARIMA;
    constantRequired <- componentsDefined$constantRequired;

    s2 <- sigma(object)^2;
    matWt <- tail(object$measurement,h);
    vecG <- matrix(object$persistence, ncol=1);
    if(ncol(object$data)>1){
        xregNumber <- ncol(object$data)-1;
    }
    else{
        xregNumber <- 0;
    }
    matF <- object$transition;

    if(type=="analytical"){
        covarMat <- covarAnal(lagsModelAll, h, matWt[1,,drop=FALSE], matF, vecG, s2);
    }
    else if(type=="empirical"){
        adamErrors <- rmultistep(object, h=h);
        covarMat <- t(adamErrors) %*% adamErrors / (nobs(object) - h);
    }
    else if(type=="simulated"){
        # This code is based on the forecast.adam() with simulations
        obsInSample <- nobs(object, all=FALSE);
        Etype <- errorType(object);
        Stype <- substr(modelType(object),nchar(modelType(object)),nchar(modelType(object)));

        # Get the lookup table
        indexLookupTable <- adamProfileCreator(lagsModelAll, lagsModelMax,
                                               obsInSample+h)$lookup[,-c(1:(obsInSample+lagsModelMax)),drop=FALSE];
        profilesRecentTable <- object$profile;

        lagsModelMin <- lagsModelAll[lagsModelAll!=1];
        if(length(lagsModelMin)==0){
            lagsModelMin <- Inf;
        }
        else{
            lagsModelMin <- min(lagsModelMin);
        }

        # Create C++ adam class
        adamCpp <- object$adamCpp;
        # adamCpp <- new(adamCore,
        #                lagsModelAll, Etype, Ttype, Stype,
        #                componentsNumberETSNonSeasonal,
        #                componentsNumberETSSeasonal,
        #                componentsNumberETS, componentsNumberARIMA,
        #                xregNumber, length(lagsModelAll),
        #                constantRequired, adamETS);

        matVt <- t(tail(object$states,lagsModelMax));

        # If this is a mixture model, produce forecasts for the occurrence
        if(is.occurrence(object$occurrence)){
            if(object$occurrence$occurrence=="provided"){
                pForecast <- rep(1,h);
            }
            else{
                pForecast <- forecast(object$occurrence,h=h)$mean;
            }
        }
        else{
            # If this was provided occurrence, then use provided values
            if(!is.null(object$occurrence) && !is.null(object$occurrence$occurrence) &&
               (object$occurrence$occurrence=="provided")){
                pForecast <- object$occurrence$forecast;
            }
            else{
                pForecast <- rep(1, h);
            }
        }

        arrVt <- array(NA, c(componentsNumberETS+componentsNumberARIMA+xregNumber+constantRequired, h+lagsModelMax, nsim));
        arrVt[,1:lagsModelMax,] <- rep(matVt,nsim);
        # Number of degrees of freedom to de-bias scales
        df <- obsInSample-nparam(object);
        # If the sample is too small, then use biased estimator
        if(df<=0){
            df[] <- obsInSample;
        }
        # If scale model is included, produce forecasts
        if(is.scale(object$scale)){
            # as.vector is needed to declass the mean.
            scaleValue <- as.vector(forecast(object$scale,h=h,interval="none")$mean);
            # De-bias the scales and transform to the appropriate scale
            # dnorm, dlnorm fit model on square residuals
            # dgnorm needs to be done with ^beta to get to 1/T part
            # The rest do not require transformations, only de-bias
            scaleValue[] <- switch(object$distribution,
                                   "dlnorm"=,
                                   "dnorm"=(scaleValue*obsInSample/df)^0.5,
                                   "dgnorm"=((scaleValue^object$other$shape)*obsInSample/df)^{1/object$other$shape},
                                   scaleValue*obsInSample/df);
        }
        else{
            scaleValue <- object$scale*obsInSample/df;
        }
        matErrors <- matrix(switch(object$distribution,
                                   "dnorm"=rnorm(h*nsim, 0, scaleValue),
                                   "dlaplace"=rlaplace(h*nsim, 0, scaleValue),
                                   "ds"=rs(h*nsim, 0, scaleValue),
                                   "dgnorm"=rgnorm(h*nsim, 0, scaleValue, object$other$shape),
                                   "dlogis"=rlogis(h*nsim, 0, scaleValue),
                                   "dt"=rt(h*nsim, obsInSample-nparam(object)),
                                   "dalaplace"=ralaplace(h*nsim, 0, scaleValue, object$other$alpha),
                                   "dlnorm"=rlnorm(h*nsim, -scaleValue^2/2, scaleValue)-1,
                                   "dinvgauss"=rinvgauss(h*nsim, 1, dispersion=scaleValue)-1,
                                   "dgamma"=rgamma(h*nsim, shape=scaleValue^{-1}, scale=scaleValue)-1,
                                   "dllaplace"=exp(rlaplace(h*nsim, 0, scaleValue))-1,
                                   "dls"=exp(rs(h*nsim, 0, scaleValue))-1,
                                   "dlgnorm"=exp(rgnorm(h*nsim, 0, scaleValue, object$other$shape))-1
        ),
        h,nsim);
        # Normalise errors in order not to get ridiculous things on small nsim
        if(nsim<=500){
            if(Etype=="A"){
                matErrors[] <- matErrors - array(apply(matErrors,1,mean),c(h,nsim));
            }
            else{
                matErrors[] <- (1+matErrors) / array(apply(1+matErrors,1,mean),c(h,nsim))-1;
            }
        }
        # This stuff is needed in order to produce adequate values for weird models
        EtypeModified <- Etype;
        if(Etype=="A" && any(object$distribution==c("dlnorm","dinvgauss","dgamma","dls","dllaplace"))){
            EtypeModified[] <- "M";
        }

        # States, Errors, Ot, Transition, Measurement, Persistence
        ySimulated <- adamCpp$simulate(matErrors, matrix(rbinom(h*nsim, 1, pForecast), h, nsim),
                                       arrVt, matWt,
                                       array(matF,c(dim(matF),nsim)),
                                       matrix(vecG, componentsNumberETS+componentsNumberARIMA+
                                                  xregNumber+constantRequired, nsim),
                                       indexLookupTable,
                                       array(profilesRecentTable,
                                             c(componentsNumberETS+componentsNumberARIMA+xregNumber+constantRequired,
                                               lagsModelMax,
                                               nsim)),
                                       EtypeModified)$data;

        yForecast <- vector("numeric", h);
        for(i in 1:h){
            if(Ttype=="M" || (Stype=="M" & h>lagsModelMin)){
                # Trim 1% of values just to resolve some issues with outliers
                yForecast[i] <- mean(ySimulated[i,],na.rm=TRUE,trim=0.01);
            }
            else{
                yForecast[i] <- mean(ySimulated[i,],na.rm=TRUE);
            }
            ySimulated[i,] <- ySimulated[i,]-yForecast[i];
            # If it is the multiplicative error, return epsilon_t
            if(Etype=="M"){
                ySimulated[i,] <- ySimulated[i,]/yForecast[i];
            }
        }

        covarMat <- (ySimulated %*% t(ySimulated))/nsim;
    }
    rownames(covarMat) <- colnames(covarMat) <- paste0("h",c(1:h));

    return(covarMat);
}

#' @export
pointLik.adam <- function(object, log=TRUE, ...){
    distribution <- object$distribution;
    yInSample <- actuals(object);
    obsInSample <- nobs(object);
    if(is.occurrence(object$occurrence)){
        otLogical <- yInSample!=0;
        yFitted <- fitted(object) / fitted(object$occurrence);
    }
    else{
        otLogical <- rep(TRUE, obsInSample);
        yFitted <- fitted(object);
    }
    scale <- extractScale(object);
    other <- switch(distribution,
                    "dalaplace"=object$other$alpha,
                    "dgnorm"=,"dlgnorm"=object$other$shape,
                    "dt"=object$other$nu);
    Etype <- errorType(object);

    likValues <- vector("numeric",obsInSample);
    likValues[otLogical] <- switch(distribution,
                                   "dnorm"=switch(Etype,
                                                  "A"=dnorm(x=yInSample[otLogical], mean=yFitted[otLogical],
                                                            sd=scale, log=TRUE),
                                                  "M"=dnorm(x=yInSample[otLogical], mean=yFitted[otLogical],
                                                            sd=scale*yFitted[otLogical], log=TRUE)),
                                   "dlaplace"=switch(Etype,
                                                     "A"=dlaplace(q=yInSample[otLogical], mu=yFitted[otLogical],
                                                                  scale=scale, log=TRUE),
                                                     "M"=dlaplace(q=yInSample[otLogical], mu=yFitted[otLogical],
                                                                  scale=scale*yFitted[otLogical], log=TRUE)),
                                   "ds"=switch(Etype,
                                               "A"=ds(q=yInSample[otLogical],mu=yFitted[otLogical],
                                                      scale=scale, log=TRUE),
                                               "M"=ds(q=yInSample[otLogical],mu=yFitted[otLogical],
                                                      scale=scale*sqrt(yFitted[otLogical]), log=TRUE)),
                                   "dgnorm"=switch(Etype,
                                                   "A"=dgnorm(q=yInSample[otLogical], mu=yFitted[otLogical],
                                                              scale=scale, shape=other, log=TRUE),
                                                   "M"=suppressWarnings(dgnorm(q=yInSample[otLogical], mu=yFitted[otLogical],
                                                                               scale=scale*yFitted[otLogical], shape=other,
                                                                               log=TRUE))),
                                   "dlogis"=switch(Etype,
                                                   "A"=dlogis(x=yInSample[otLogical], location=yFitted[otLogical],
                                                              scale=scale, log=TRUE),
                                                   "M"=dlogis(x=yInSample[otLogical], location=yFitted[otLogical],
                                                              scale=scale*yFitted[otLogical], log=TRUE)),
                                   "dt"=switch(Etype,
                                               "A"=dt(residuals(object)[otLogical], df=abs(other), log=TRUE),
                                               "M"=dt(residuals(object)[otLogical]*yFitted[otLogical],
                                                      df=abs(other), log=TRUE)),
                                   "dalaplace"=switch(Etype,
                                                      "A"=dalaplace(q=yInSample[otLogical], mu=yFitted[otLogical],
                                                                    scale=scale, alpha=other, log=TRUE),
                                                      "M"=dalaplace(q=yInSample[otLogical], mu=yFitted[otLogical],
                                                                    scale=scale*yFitted[otLogical], alpha=other, log=TRUE)),
                                   "dlnorm"=dlnorm(x=yInSample[otLogical],
                                                   meanlog=log(yFitted[otLogical]) -scale^2/2,
                                                   sdlog=scale, log=TRUE),
                                   "dllaplace"=dlaplace(q=log(yInSample[otLogical]), mu=log(yFitted[otLogical]),
                                                        scale=scale, log=TRUE),
                                   "dls"=ds(q=log(yInSample[otLogical]), mu=log(yFitted[otLogical]),
                                            scale=scale, log=TRUE),
                                   "dlgnorm"=dgnorm(q=log(yInSample[otLogical]), mu=log(yFitted[otLogical]),
                                                    scale=scale, shape=other, log=TRUE),
                                   "dinvgauss"=dinvgauss(x=yInSample[otLogical], mean=yFitted[otLogical],
                                                         dispersion=scale/yFitted[otLogical], log=TRUE),
                                   "dgamma"=dgamma(x=yInSample[otLogical], shape=1/scale,
                                                   scale=scale*yFitted[otLogical], log=TRUE)
    );
    if(any(distribution==c("dllaplace","dls","dlgnorm"))){
        likValues[otLogical] <- likValues[otLogical] - log(yInSample[otLogical]);
    }

    # If this is a mixture model, take the respective probabilities into account (differential entropy)
    if(is.occurrence(object$occurrence)){
        likValues[!otLogical] <- -switch(distribution,
                                         "dnorm" = (log(sqrt(2*pi)*scale)+0.5),
                                         "dlnorm" = (log(sqrt(2*pi)*scale)+0.5) -scale^2/2,
                                         "dlogis" = 2,
                                         "dlaplace" =,
                                         "dllaplace" =,
                                         "dalaplace" = (1 + log(2*scale)),
                                         "dt" = ((scale+1)/2 * (digamma((scale+1)/2)-digamma(scale/2)) +
                                                     log(sqrt(scale) * beta(scale/2,0.5))),
                                         "ds" =,
                                         "dls" = (2 + 2*log(2*scale)),
                                         "dgnorm" =,
                                         "dlgnorm" = 1/other-log(other/(2*scale*gamma(1/other))),
                                         "dinvgauss" = (0.5*(log(pi/2)+1+log(scale))),
                                         "dgamma" = (1/scale + log(scale*yFitted[!otLogical]) +
                                                         log(gamma(1/scale)) + (1-1/scale)*digamma(1/scale))
        );

        likValues[] <- likValues + pointLik(object$occurrence);
    }
    likValues <- ts(likValues, start=start(yFitted), frequency=frequency(yFitted));

    if(!log){
        likValues[] <- exp(likValues);
    }

    return(likValues);
}

#### Simulate function ####

#' @param obs Number of observations to produce in the simulated data.
#' @param nsim Number of series to generate from the model.
#' @param seed Random seed used in simulation of data.
#' @examples
#' # Fit ADAM to the data
#' ourModel <- adam(rnorm(100,100,10), model="AAdN")
#' # Simulate the data
#' x <- simulate(ourModel)
#'
#' @rdname adam
#' @export
simulate.adam <- function(object, nsim=1, seed=NULL, obs=nobs(object), ...){
    # Start measuring the time of calculations
    startTime <- Sys.time();

    ellipsis <- list(...);

    # Check whether we deal with adam ETS or the conventional
    adamETS <- adamETSChecker(object);

    if(!is.null(seed)){
        set.seed(seed);
    }

    # All the variables needed in the function
    yInSample <- actuals(object);
    yClasses <- class(yInSample);
    obsInSample <- obs;
    Etype <- errorType(object);
    Ttype <- substr(modelType(object),2,2);
    Stype <- substr(modelType(object),nchar(modelType(object)),nchar(modelType(object)));
    lags <- object$lags;
    lagsSeasonal <- lags[lags!=1];
    lagsModelAll <- object$lagsAll;
    lagsModelMax <- max(lagsModelAll);
    persistence <- as.matrix(object$persistence);
    # If there is xreg, but no deltas, increase persistence by including zeroes
    # This can be considered as a failsafe mechanism
    if(ncol(object$data)>1 && !any(substr(names(object$persistence),1,5)=="delta")){
        persistence <- rbind(persistence,matrix(rep(0,sum(object$nParam[,2])),ncol=1));
    }

    # Get componentsNumberETS, seasonal and componentsNumberARIMA
    componentsDefined <- componentsDefiner(object);
    componentsNumberETS <- componentsDefined$componentsNumberETS;
    componentsNumberETSSeasonal <- componentsDefined$componentsNumberETSSeasonal;
    componentsNumberETSNonSeasonal <- componentsDefined$componentsNumberETSNonSeasonal;
    componentsNumberARIMA <- componentsDefined$componentsNumberARIMA;
    constantRequired <- componentsDefined$constantRequired;

    # Prepare variables for xreg
    if(!is.null(object$initial$xreg)){
        xregModel <- TRUE;

        #### Create xreg vectors ####
        xreg <- object$data;
        formula <- formula(object)
        responseName <- all.vars(formula)[1];
        # Robustify the names of variables
        colnames(xreg) <- make.names(colnames(xreg),unique=TRUE);
        # The names of the original variables
        xregNamesOriginal <- all.vars(formula)[-1];
        # Levels for the factors
        xregFactorsLevels <- lapply(xreg,levels);
        xregFactorsLevels[[responseName]] <- NULL;
        # Expand the variables. We cannot use alm, because it is based on obsInSample
        xregData <- model.frame(formula,data=as.data.frame(xreg));
        # Binary, flagging factors in the data
        xregFactors <- (attr(terms(xregData),"dataClasses")=="factor")[-1];
        # Get the names from the standard model.matrix
        xregNames <- colnames(model.matrix(xregData,data=xregData));
        interceptIsPresent <- FALSE;
        if(any(xregNames=="(Intercept)")){
            interceptIsPresent[] <- TRUE;
            xregNames <- xregNames[xregNames!="(Intercept)"];
        }
        # Expanded stuff with all levels for factors
        if(any(xregFactors)){
            xregModelMatrix <- model.matrix(xregData,xregData,
                                            contrasts.arg=lapply(xregData[attr(terms(xregData),"dataClasses")=="factor"],
                                                                 contrasts, contrasts=FALSE));
            xregNamesModified <- colnames(xregModelMatrix)[-1];
        }
        else{
            xregModelMatrix <- model.matrix(xregData,data=xregData);
            xregNamesModified <- xregNames;
        }
        xregData <- as.matrix(xregModelMatrix);
        # Remove intercept
        if(interceptIsPresent){
            xregData <- xregData[,-1,drop=FALSE];
        }
        xregNumber <- ncol(xregData);

        # The indices of the original parameters
        xregParametersMissing <- setNames(vector("numeric",xregNumber),xregNamesModified);
        # # The indices of the original parameters
        xregParametersIncluded <- setNames(vector("numeric",xregNumber),xregNamesModified);
        # The vector, marking the same values of smoothing parameters
        if(interceptIsPresent){
            xregParametersPersistence <- setNames(attr(xregModelMatrix,"assign")[-1],xregNamesModified);
        }
        else{
            xregParametersPersistence <- setNames(attr(xregModelMatrix,"assign"),xregNamesModified);
        }

        # If there are factors not in the alm data, create additional initials
        if(any(!(xregNamesModified %in% xregNames))){
            xregAbsent <- !(xregNamesModified %in% xregNames);
            # Go through new names and find, where they came from. Then get the missing parameters
            for(i in which(xregAbsent)){
                # Find the name of the original variable
                # Use only the last value... hoping that the names like x and x1 are not used.
                xregNameFound <- tail(names(sapply(xregNamesOriginal,grepl,xregNamesModified[i])),1);
                # Get the indices of all k-1 levels
                xregParametersIncluded[xregNames[xregNames %in% paste0(xregNameFound,
                                                                       xregFactorsLevels[[xregNameFound]])]] <- i;
                # Get the index of the absent one
                xregParametersMissing[i] <- i;
            }
            # Write down the new parameters
            xregNames <- xregNamesModified;
        }
        # The vector of parameters that should be estimated (numeric + original levels of factors)
        xregParametersEstimated <- xregParametersIncluded
        xregParametersEstimated[xregParametersEstimated!=0] <- 1;
        xregParametersEstimated[xregParametersMissing==0 & xregParametersIncluded==0] <- 1;
    }
    else{
        xregModel <- FALSE;
        xregNumber <- 0;
        xregParametersMissing <- 0;
        xregParametersIncluded <- 0;
        xregParametersEstimated <- 0;
        xregParametersPersistence <- 0;
    }
    profiles <- adamProfileCreator(lagsModelAll, lagsModelMax, obsInSample);
    indexLookupTable <- profiles$lookup;

    # Create C++ adam class
    adamCpp <- object$adamCpp;
    # adamCpp <- new(adamCore,
    #                lagsModelAll, Etype, Ttype, Stype,
    #                componentsNumberETSNonSeasonal,
    #                componentsNumberETSSeasonal,
    #                componentsNumberETS, componentsNumberARIMA,
    #                xregNumber, length(lagsModelAll),
    #                constantRequired, adamETS);

    #### Prepare the necessary matrices ####
    # States are defined similar to how it is done in adam.
    arrVt <- array(t(object$states),c(ncol(object$states),nrow(object$states)+obsInSample-nobs(object),nsim),
                   dimnames=list(colnames(object$states),NULL,paste0("nsim",c(1:nsim))));

    # Set profile, which is used in the data generation
    # profilesRecentTable <- t(object$states[1:lagsModelMax,]);
    profilesRecentTable <- object$profileInitial;

    # Transition and measurement
    arrF <- array(object$transition,c(dim(object$transition),nsim));
    matWt <- object$measurement;
    if(nrow(matWt)<obsInSample){
        matWt <- rbind(matWt,
                       matrix(rep(tail(matWt,1),each=obsInSample-nrow(matWt)),
                              obsInSample-nrow(matWt), ncol(matWt)));
    }

    # Persistence matrix
    matG <- array(persistence, c(length(persistence), nsim),
                  dimnames=list(names(persistence), paste0("nsim",c(1:nsim))));

    if(is.null(object$occurrence)){
        pt <- rep(1, obsInSample);
    }
    else{
        pt <- fitted(object$occurrence);
    }

    # Number of degrees of freedom to de-bias scales
    df <- obsInSample-nparam(object);
    # If the sample is too small, then use biased estimator
    if(df<=0){
        df[] <- obsInSample;
    }

    # If scale model is included, produce forecasts
    if(is.scale(object$scale)){
        # as.vector is needed to declass the mean.
        scaleValue <- as.vector(fitted(object$scale));
        # De-bias the scales and transform to the appropriate scale
        # dnorm, dlnorm fit model on square residuals
        # dgnorm needs to be done with ^beta to get to 1/T part
        # The rest do not require transformations, only de-bias
        scaleValue[] <- switch(object$distribution,
                               "dlnorm"=,
                               "dnorm"=(scaleValue*obsInSample/df)^0.5,
                               "dgnorm"=((scaleValue^object$other$shape)*obsInSample/df)^{1/object$other$shape},
                               scaleValue*obsInSample/df);
    }
    else{
        scaleValue <- object$scale*obsInSample/df;
    }
    matErrors <- matrix(switch(object$distribution,
                               "dnorm"=rnorm(obsInSample*nsim, 0, scaleValue),
                               "dlaplace"=rlaplace(obsInSample*nsim, 0, scaleValue),
                               "ds"=rs(obsInSample*nsim, 0, scaleValue),
                               "dgnorm"=rgnorm(obsInSample*nsim, 0, scaleValue, object$other$shape),
                               "dlogis"=rlogis(obsInSample*nsim, 0, scaleValue),
                               "dt"=rt(obsInSample*nsim, obsInSample-nparam(object)),
                               "dalaplace"=ralaplace(obsInSample*nsim, 0, scaleValue, object$other$alpha),
                               "dlnorm"=rlnorm(obsInSample*nsim, -scaleValue^2/2, scaleValue)-1,
                               "dinvgauss"=rinvgauss(obsInSample*nsim, 1, dispersion=scaleValue)-1,
                               "dgamma"=rgamma(obsInSample*nsim, shape=scaleValue^{-1}, scale=scaleValue)-1,
                               "dllaplace"=exp(rlaplace(obsInSample*nsim, 0, scaleValue))-1,
                               "dls"=exp(rs(obsInSample*nsim, 0, scaleValue))-1,
                               "dlgnorm"=exp(rgnorm(obsInSample*nsim, 0, scaleValue, object$other$shape))-1
    ), obsInSample, nsim);

    # This stuff is needed in order to produce adequate values for weird models
    EtypeModified <- Etype;
    if(Etype=="A" && any(object$distribution==c("dlnorm","dinvgauss","dgamma","dls","dllaplace"))){
        EtypeModified[] <- "M";
    }

    # Refit the model with the new parameter
    ySimulated <- adamCpp$simulate(matErrors, matrix(rbinom(obsInSample*nsim, 1, pt), obsInSample, nsim),
                                   arrVt, matWt,
                                   arrF, matG,
                                   indexLookupTable,
                                   array(profilesRecentTable,
                                         c(componentsNumberETS+componentsNumberARIMA+xregNumber+constantRequired,
                                           lagsModelMax,
                                           nsim)),
                                   EtypeModified);

    # Set the proper time stamps for the fitted
    if(any(yClasses=="zoo")){
        # Get indices for the cases, when obsInSample was provided by user
        yIndex <- time(yInSample)
        yIndexDiff <- diff(head(yIndex,2));
        yTime <- yIndex[1]+yIndexDiff*c(1:(obsInSample-1));
        matrixYt <- zoo(array(ySimulated$data,c(obsInSample,nsim),
                              dimnames=list(NULL,paste0("nsim",c(1:nsim)))),
                        order.by=yTime);
    }
    else{
        matrixYt <- ts(array(ySimulated$data,c(obsInSample,nsim),
                             dimnames=list(NULL,paste0("nsim",c(1:nsim)))),
                       start=start(yInSample), frequency=frequency(yInSample));
    }

    return(structure(list(timeElapsed=Sys.time()-startTime, model=object$model, distribution=object$distribution,
                          data=matrixYt, states=ySimulated$states, persistence=object$persistence,
                          measurement=matWt, transition=object$transition, initial=object$initial,
                          probability=pt, occurrence=object$occurrence,
                          residuals=matErrors, other=ellipsis),
                     class=c("adam.sim","smooth.sim")));
}

#' @export
print.adam.sim <- function(x, ...){
    cat(paste0("Data generated from: ",x$model," estimated via adam()\n"));
    cat(paste0("Number of generated series: ",ncol(x$data),"\n"));
}

##### Other methods to implement #####
# accuracy.adam <- function(object, holdout, ...){}
# pls.adam
