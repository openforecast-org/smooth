#' Occurrence ETS model (legacy implementation)
#'
#' Function returns the occurrence part of iETS model with the specified
#' probability update and model types. This is the original \code{oes()}
#' implementation, kept under the name \code{oes_old()} as a fallback after
#' the new wrapper around \link[smooth]{om} took over the \code{oes()} name.
#'
#' The function estimates probability of demand occurrence, using the selected
#' ETS state space models.
#'
#' For the details about the model and its implementation, see the respective
#' vignette: \code{vignette("oes","smooth")}
#'
#' @template ssIntermittentRef
#' @template ssInitialParam
#' @template ssPersistenceParam
#' @template ssXregParam
#' @template ssAuthor
#' @template ssKeywords
#'
#' @param y Either numeric vector or time series vector.
#' @param model The type of ETS model used for the estimation. Normally this should
#' be \code{"MNN"} or any other pure multiplicative or additive model. The model
#' selection is available here (although it's not fast), so you can use, for example,
#' \code{"YYN"} and \code{"XXN"} for selecting between the pure multiplicative and
#' pure additive models respectively. Using mixed models is possible, but not
#' recommended.
#' @param occurrence The type of model used in probability estimation. Can be
#' \code{"none"} - none,
#' \code{"fixed"} - constant probability,
#' \code{"odds-ratio"} - the Odds-ratio model with b=1 in Beta distribution,
#' \code{"inverse-odds-ratio"} - the model with a=1 in Beta distribution,
#' \code{"direct"} - the TSB-like (Teunter et al., 2011) probability update
#' mechanism a+b=1,
#' \code{"auto"} - the automatically selected type of occurrence model,
#' \code{"general"} - the general Beta model with two parameters. This will call
#' \code{oesg()} function with two similar ETS models and the same provided
#' parameters (initials and smoothing).
#' @param phi The value of the dampening parameter. Used only for damped-trend models.
#' @param ic The information criteria to use in case of model selection.
#' @param h The forecast horizon.
#' @param holdout If \code{TRUE}, holdout sample of size \code{h} is taken from
#' the end of the data.
#' @param bounds What type of bounds to use in the model estimation. The first
#' letter can be used instead of the whole word.
#' @param silent If \code{silent="none"}, then nothing is silent, everything is
#' printed out and drawn. \code{silent="all"} means that nothing is produced or
#' drawn (except for warnings). In case of \code{silent="graph"}, no graph is
#' produced. If \code{silent="legend"}, then legend of the graph is skipped.
#' And finally \code{silent="output"} means that nothing is printed out in the
#' console, but the graph is produced. \code{silent} also accepts \code{TRUE}
#' and \code{FALSE}. In this case \code{silent=TRUE} is equivalent to
#' \code{silent="all"}, while \code{silent=FALSE} is equivalent to
#' \code{silent="none"}. The parameter also accepts first letter of words ("n",
#' "a", "g", "l", "o").
#' @param initialSeason The vector of the initial seasonal components. If \code{NULL},
#' then it is estimated.
#' @param ... The parameters passed to the optimiser, such as \code{maxeval},
#' \code{xtol_rel}, \code{algorithm} and \code{print_level}. The description of
#' these is printed out by \code{nloptr.print.options()} function from the \code{nloptr}
#' package. The default values in the oes function are \code{maxeval=500},
#' \code{xtol_rel=1E-8}, \code{algorithm="NLOPT_LN_NELDERMEAD"} and \code{print_level=0}.
#' @return The object of class "occurrence" is returned. It contains following list of
#' values:
#'
#' \itemize{
#' \item \code{model} - the type of the estimated ETS model;
#' \item \code{timeElapsed} - the time elapsed for the construction of the model;
#' \item \code{fitted} - the fitted values for the probability;
#' \item \code{fittedModel} - the fitted values of the underlying ETS model, where applicable
#' (only for occurrence=c("o","i","d"));
#' \item \code{forecast} - the forecast of the probability for \code{h} observations ahead;
#' \item \code{forecastModel} - the forecast of the underlying ETS model, where applicable
#' (only for occurrence=c("o","i","d"));
#' \item \code{lower} - the lower bound of the interval if \code{interval!="none"};
#' \item \code{upper} - the upper bound of the interval if \code{interval!="none"};
#' \item \code{lowerModel} - the lower bound of the interval of the underlying ETS model
#' if \code{interval!="none"};
#' \item \code{upperModel} - the upper bound of the interval of the underlying ETS model
#' if \code{interval!="none"};
#' \item \code{states} - the values of the state vector;
#' \item \code{logLik} - the log-likelihood value of the model;
#' \item \code{nParam} - the number of parameters in the model (the matrix is returned);
#' \item \code{residuals} - the residuals of the model;
#' \item \code{y} - actual values of occurrence (zeros and ones).
#' \item \code{persistence} - the vector of smoothing parameters;
#' \item \code{phi} - the value of the damped trend parameter;
#' \item \code{initial} - initial values of the state vector;
#' \item \code{initialSeason} - the matrix of initials seasonal states;
#' \item \code{occurrence} - the type of the occurrence model;
#' \item \code{updateX} - boolean, defining, if the states of exogenous variables were
#' estimated as well.
#' \item \code{initialX} - initial values for parameters of exogenous variables.
#' \item \code{persistenceX} - persistence vector g for exogenous variables.
#' \item \code{transitionX} - transition matrix F for exogenous variables.
#' \item \code{accuracy} - The error measures for the forecast (in case of \code{holdout=TRUE}).
#' \item \code{B} - the vector of all the estimated parameters (in case of "odds-ratio",
#' "inverse-odds-ratio" and "direct" models).
#' }
#' @seealso \code{\link[smooth]{adam}, \link[smooth]{oesg}, \link[smooth]{es},
#' \link[smooth]{oes}, \link[smooth]{om}}
#' @examples
#'
#' y <- rpois(100,0.1)
#' oes_old(y, occurrence="auto")
#'
#' oes_old(y, occurrence="f")
#'
#' @export
oes_old <- function(y, model="MNN", persistence=NULL, initial="optimal", initialSeason=NULL, phi=NULL,
                occurrence=c("fixed","general","odds-ratio","inverse-odds-ratio","direct","auto","none"),
                ic=c("AICc","AIC","BIC","BICc"), h=0, holdout=FALSE,
                # interval=c("none","parametric","likelihood","semiparametric","nonparametric"), level=0.95,
                bounds=c("usual","admissible","none"),
                silent=c("all","graph","legend","output","none"),
                xreg=NULL, regressors=c("use","select"), initialX=NULL,
                ...){
    # Function returns the occurrence part of the intermittent state space model

# Start measuring the time of calculations
    startTime <- Sys.time();

    # Set the defaults for the parameters that are no longer supported
    interval <- "none";
    level <- 0.95;
    updateX <- FALSE;
    transitionX <- NULL;
    persistenceX <- NULL;

    initial <- initial[1];

    # Options for the fitter and forecaster:
    # O: M / A odds-ratio - "odds-ratio"
    # I: - M / A inverse-odds-ratio - "inverse-odds-ratio"
    # G: - M / A general model - "general" <- This should rely on a vector-based model
    # P: - M Probability based (TSB like) - "direct"
    # Not in the fitter:
    # F: - fixed

    # If the model was passed in the occurrence part, deal with it
    if(is.oes(occurrence)){
        model <- occurrence;
    }

    # If the model is oesg, use it
    if(is.oesg(model)){
        return(oesg(y, modelA=model$modelA, modelB=model$modelB, h=h, holdout=holdout,
                    interval=interval, level=level, bounds=bounds,
                    silent=silent, ...));
    }
    else if(is.oes(model)){
        persistence <- model$persistence;
        phi <- model$phi;
        initial <- model$initial;
        initialSeason <- model$initialSeason;
        xreg <- model$xreg;
        occurrence <- model$occurrence;
        initialX <- model$initialX;
        updateX <- model$updateX;
        transitionX <- model$transitionX;
        persistenceX <- model$persistenceX;
        B <- model$B;
        model <- modelType(model);
    }

    ##### Preparations #####
    occurrence <- substr(match.arg(occurrence,c("fixed","general","odds-ratio","inverse-odds-ratio","direct","auto","none")),1,1);

    if(is.smooth.sim(y)){
        y <- y$data;
    }

    # Add all the variables in ellipsis to current environment
    # list2env(list(...),environment());
    ellipsis <- list(...);
    ellipsis <- depricator(ellipsis, "xregDo", "regressors");

    # Parameters for the nloptr
    if(any(names(ellipsis)=="maxeval")){
        maxeval <- ellipsis$maxeval;
    }
    else{
        maxeval <- 500;
    }
    if(any(names(ellipsis)=="xtol_rel")){
        xtol_rel <- ellipsis$xtol_rel;
    }
    else{
        xtol_rel <- 1e-8;
    }
    if(any(names(ellipsis)=="algorithm")){
        algorithm <- ellipsis$algorithm;
    }
    else{
        algorithm <- "NLOPT_LN_NELDERMEAD";
    }
    if(any(names(ellipsis)=="print_level")){
        print_level <- ellipsis$print_level;
    }
    else{
        print_level <- 0;
    }

    #### These are needed in order for ssInput to go forward
    loss <- "MSE";
    oesmodel <- NULL;

    ##### Set environment for ssInput and make all the checks #####
    environment(ssInput) <- environment();
    ssInput("oes",ParentEnvironment=environment());

    ### Produce vectors with zeroes and ones, fixed probability and the number of ones.
    ot <- (yInSample!=0)*1;
    otAll <- (y!=0)*1;
    iprob <- mean(ot);
    obsOnes <- sum(ot);

    if(all(ot==ot[1])){
        warning(paste0("There is no variability in the occurrence of the variable in-sample.\n",
                       "Switching to occurrence='none'."),call.=FALSE)
        occurrence <- "n";
    }

    ##### Prepare exogenous variables #####
    xregdata <- ssXreg(y=otAll, Etype="A", xreg=xreg, updateX=updateX, ot=rep(1,obsInSample),
                       persistenceX=persistenceX, transitionX=transitionX, initialX=initialX,
                       obsInSample=obsInSample, obsAll=obsAll, obsStates=obsStates,
                       lagsModelMax=1, h=h, regressors=regressors, silent=silentText,
                       allowMultiplicative=FALSE);

    nExovars <- xregdata$nExovars;
    matxt <- xregdata$matxt;
    matat <- t(xregdata$matat);
    xregEstimate <- xregdata$xregEstimate;
    matFX <- xregdata$matFX;
    vecgX <- xregdata$vecgX;
    xregNames <- colnames(matxt);
    initialXEstimate <- xregdata$initialXEstimate;
    xreg <- xregdata$xreg;

    #### The functions for the O, I, and P models ####
    if(any(occurrence==c("o","i","d"))){
        ##### Initialiser of oes #####
        # This creates the states, transition, persistence and measurement matrices
        oesInitialiser <- function(Etype, Ttype, Stype, damped, phiEstimate, occurrence,
                                   dataFreq, obsInSample, obsAll, obsStates, ot,
                                   persistenceEstimate, persistence, initialType, initialValue,
                                   initialSeasonEstimate, initialSeason){
            # Define the lags of the model, number of components and max lag
            lagsModel <- 1;
            statesNames <- "level";
            if(Ttype!="N"){
                lagsModel <- c(lagsModel, 1);
                statesNames <- c(statesNames, "trend");
            }
            nComponentsNonSeasonal <- length(lagsModel);
            if(modelIsSeasonal){
                lagsModel <- c(lagsModel, dataFreq);
                statesNames <- c(statesNames, "seasonal");
            }
            nComponentsAll <- length(lagsModel);
            lagsModelMax <- max(lagsModel);

            # Transition matrix
            matF <- diag(nComponentsAll);
            if(Ttype!="N"){
                matF[1,2] <- 1;
            }

            # Measurement vector
            matw <- matrix(1, 1, nComponentsAll, dimnames=list(NULL, statesNames));

            if(damped && !phiEstimate){
                matF[c(1,2),2] <- phi;
                matw[2] <- phi;
            }

            # Persistence vector. The initials are set here!
            if(persistenceEstimate){
                vecg <- matrix(0.1, nComponentsAll, 1);
            }
            else{
                vecg <- matrix(persistence, nComponentsAll, 1);
            }
            rownames(vecg) <- statesNames;

            # The matrix of states
            matvt <- matrix(NA, nComponentsAll, obsStates, dimnames=list(statesNames, NULL));

            # Define initial states. The initials are set here!
            if(initialType!="p"){
                initialStates <- rep(0, nComponentsNonSeasonal);
                initialStates[1] <- mean(ot);
                if(Ttype=="M"){
                    initialStates[2] <- 1;
                }
                else if(Ttype=="A"){
                    initialStates[2] <- 0;
                }
                if(occurrence=="o"){
                    initialStates[1] <- initialStates[1] / (1 - initialStates[1]);
                }
                else if(occurrence=="i"){
                    initialStates[1] <- (1-initialStates[1]) / initialStates[1];
                }
                # Initials specifically for ETS(A,M,N) and alike
                if(Etype=="A" && any(occurrence==c("o","i")) && Ttype=="M" && (initialStates[1]>1)){
                    initialStates[1] <- log(initialStates[1]);
                }

                matvt[1,1:lagsModelMax] <- initialStates[1];
                if(Ttype!="N"){
                    matvt[2,1:lagsModelMax] <- initialStates[2];
                }
            }
            else{
                matvt[1:nComponentsNonSeasonal,1:lagsModelMax] <- initialValue;
            }

            # Define the seasonals
            if(modelIsSeasonal){
                if(initialSeasonEstimate){
                    XValues <- matrix(rep(diag(lagsModelMax),ceiling(obsInSample/lagsModelMax)),lagsModelMax)[,1:obsInSample];
                    # The seasonal values should be between -1 and 1
                    initialSeasonValue <- solve(XValues %*% t(XValues)) %*% XValues %*% (ot - mean(ot));
                    # But make sure that it lies there
                    if(any(abs(initialSeasonValue)>1)){
                        initialSeasonValue <- initialSeasonValue / (max(abs(initialSeasonValue)) + 1E-10);
                    }

                    # Correct seasonals for the two models
                    if(any(occurrence==c("o","i"))){
                        # If there are some boundary values, move them a bit
                        if(any(abs(initialSeasonValue)==1)){
                            initialSeasonValue[initialSeasonValue==1] <- 1 - 1E-10;
                            initialSeasonValue[initialSeasonValue==-1] <- -1 + 1E-10;
                        }
                        # Transform this into the probability scale
                        initialSeasonValue <- (initialSeasonValue + 1) / 2;
                        if(occurrence=="o"){
                            initialSeasonValue <- initialSeasonValue / (1 - initialSeasonValue);
                        }
                        else{
                            initialSeasonValue <- (1 - initialSeasonValue) / initialSeasonValue;
                        }

                        # Transform to the adequate scale
                        if(Stype=="A"){
                            initialSeasonValue <- log(initialSeasonValue);
                        }
                    }
                    else{
                        if(Stype=="M"){
                            initialSeasonValue <- exp(initialSeasonValue);
                        }
                    }

                    # Normalise the initials
                    if(Stype=="A"){
                        initialSeasonValue <- initialSeasonValue - mean(initialSeasonValue);
                    }
                    else{
                        initialSeasonValue <- exp(log(initialSeasonValue) - mean(log(initialSeasonValue)));
                    }

                    # Write down the initial seasons into the state matrix
                    matvt[nComponentsAll,1:lagsModelMax] <- initialSeasonValue;
                }
                else{
                    matvt[nComponentsAll,1:lagsModelMax] <- initialSeason;
                }
            }

            return(list(nComponentsAll=nComponentsAll, nComponentsNonSeasonal=nComponentsNonSeasonal,
                        lagsModelMax=lagsModelMax, lagsModel=lagsModel,
                        matvt=matvt, vecg=vecg, matF=matF, matw=matw));
        }

        ##### Fill in the elements of oes #####
        # This takes the existing matrices and fills them in
        oesElements <- function(B, lagsModel, Ttype, Stype, damped,
                                nComponentsAll, nComponentsNonSeasonal, nExovars, lagsModelMax,
                                persistenceEstimate, initialType, phiEstimate, modelIsSeasonal, initialSeasonEstimate,
                                xregEstimate, initialXEstimate, updateX,
                                matvt, vecg, matF, matw, matat, matFX, vecgX){
            i <- 0;
            if(persistenceEstimate){
                vecg[] <- B[1:nComponentsAll];
                i[] <- nComponentsAll;
            }
            if(damped && phiEstimate){
                i[] <- i + 1;
                matF[,nComponentsNonSeasonal] <- B[i];
                matw[,nComponentsNonSeasonal] <- B[i];
            }
            if(initialType=="o"){
                matvt[1:nComponentsNonSeasonal,1:lagsModelMax] <- B[i+c(1:nComponentsNonSeasonal)];
                i[] <- i + nComponentsNonSeasonal;
            }
            if(modelIsSeasonal && initialSeasonEstimate){
                matvt[nComponentsAll,1:lagsModelMax] <- B[i+c(1:lagsModelMax)];
                i[] <- i + lagsModelMax;
            }
            if(xregEstimate){
                if(initialXEstimate){
                    matat[,1:lagsModelMax] <- B[i+c(1:nExovars)];
                    i[] <- i + nExovars;
                }
                if(updateX){
                    matFX[] <- B[i+c(1:(nExovars^2))];
                    i[] <- i + nExovars^2;

                    vecgX[] <- B[i+c(1:nExovars)];
                }
            }

            return(list(vecg=vecg, matF=matF, matw=matw, matvt=matvt,
                        matat=matat, matFX=matFX, vecgX=vecgX));
        }

        ##### B values for estimation #####
        # Function constructs default bounds where B values should lie
        BValues <- function(bounds, Ttype, Stype, damped, phiEstimate, persistenceEstimate,
                            initialType, modelIsSeasonal, initialSeasonEstimate,
                            xregEstimate, initialXEstimate, updateX,
                            lagsModelMax, nComponentsAll, nComponentsNonSeasonal,
                            vecg, matvt, matat, matFX, vecgX, xregNames, nExovars){
            B <- NA;
            lb <- NA;
            ub <- NA;

            #### Usual bounds ####
            if(bounds=="u"){
                # Smoothing parameters
                if(persistenceEstimate){
                    B <- c(B,vecg);
                    lb <- c(lb,rep(0,nComponentsAll));
                    ub <- c(ub,rep(1,nComponentsAll));
                }
                # Phi
                if(damped && phiEstimate){
                    B <- c(B,0.95);
                    lb <- c(lb,0);
                    ub <- c(ub,1);
                }
                # Initial states
                if(initialType=="o"){
                    if(Etype=="A"){
                        B <- c(B,matvt[1:nComponentsNonSeasonal,lagsModelMax]);
                        lb <- c(lb,-Inf);
                        ub <- c(ub,Inf);
                    }
                    else{
                        if(Ttype=="A"){
                            # This is something like ETS(M,A,N), so set level to mean, trend to zero for stability
                            B <- c(B,mean(ot[1:min(dataFreq,obsInSample)]),1E-5);
                            lb <- c(lb,-Inf);
                            ub <- c(ub,Inf);
                        }
                        else{
                            B <- c(B,abs(matvt[1:nComponentsNonSeasonal,lagsModelMax]));
                            lb <- c(lb,1E-10);
                            ub <- c(ub,Inf);
                        }
                    }
                    if(Ttype=="A"){
                        lb <- c(lb,-Inf);
                        ub <- c(ub,Inf);
                    }
                    else if(Ttype=="M"){
                        lb <- c(lb,1E-20);
                        ub <- c(ub,3);
                    }
                    # Initial seasonals
                    if(modelIsSeasonal){
                        if(initialSeasonEstimate){
                            B <- c(B,matvt[nComponentsAll,1:lagsModelMax]);
                            if(Stype=="A"){
                                lb <- c(lb,rep(-Inf,lagsModelMax));
                                ub <- c(ub,rep(Inf,lagsModelMax));
                            }
                            else{
                                lb <- c(lb,matvt[nComponentsAll,1:lagsModelMax]*0.1);
                                ub <- c(ub,matvt[nComponentsAll,1:lagsModelMax]*10);
                            }
                        }
                    }
                }
            }
            #### Admissible bounds ####
            else if(bounds=="a"){
                # Smoothing parameters
                if(persistenceEstimate){
                    B <- c(B,vecg);
                    lb <- c(lb,rep(-5,nComponentsAll));
                    ub <- c(ub,rep(5,nComponentsAll));
                }
                # Phi
                if(damped && phiEstimate){
                    B <- c(B,0.95);
                    lb <- c(lb,0);
                    ub <- c(ub,1);
                }
                # Initial states
                if(initialType=="o"){
                    if(Etype=="A"){
                        B <- c(B,matvt[1:nComponentsNonSeasonal,lagsModelMax]);
                        lb <- c(lb,-Inf);
                        ub <- c(ub,Inf);
                    }
                    else{
                        if(Ttype=="A"){
                            # This is something like ETS(M,A,N), so set level to mean, trend to zero for stability
                            B <- c(B,mean(ot[1:min(dataFreq,obsInSample)]),1E-5);
                            lb <- c(lb,-Inf);
                            ub <- c(ub,Inf);
                        }
                        else{
                            B <- c(B,abs(matvt[1:nComponentsNonSeasonal,lagsModelMax]));
                            lb <- c(lb,1E-10);
                            ub <- c(ub,Inf);
                        }
                    }
                    if(Ttype=="A"){
                        lb <- c(lb,-Inf);
                        ub <- c(ub,Inf);
                    }
                    else if(Ttype=="M"){
                        lb <- c(lb,1E-20);
                        ub <- c(ub,3);
                    }
                    # Initial seasonals
                    if(modelIsSeasonal){
                        if(initialSeasonEstimate){
                            B <- c(B,matvt[nComponentsAll,1:lagsModelMax]);
                            if(Stype=="A"){
                                lb <- c(lb,rep(-Inf,lagsModelMax));
                                ub <- c(ub,rep(Inf,lagsModelMax));
                            }
                            else{
                                lb <- c(lb,matvt[nComponentsAll,1:lagsModelMax]*0.1);
                                ub <- c(ub,matvt[nComponentsAll,1:lagsModelMax]*10);
                            }
                        }
                    }
                }
            }
            #### No bounds ####
            else{
                # Smoothing parameters
                if(persistenceEstimate){
                    B <- c(B,vecg);
                    lb <- c(lb,rep(-Inf,nComponentsAll));
                    ub <- c(ub,rep(Inf,nComponentsAll));
                }
                # Phi
                if(damped && phiEstimate){
                    B <- c(B,0.95);
                    lb <- c(lb,-Inf);
                    ub <- c(ub,Inf);
                }
                # Initial states
                if(initialType=="o"){
                    if(Etype=="A"){
                        B <- c(B,matvt[1:nComponentsNonSeasonal,lagsModelMax]);
                        lb <- c(lb,-Inf);
                        ub <- c(ub,Inf);
                    }
                    else{
                        if(Ttype=="A"){
                            # This is something like ETS(M,A,N), so set level to mean, trend to zero for stability
                            B <- c(B,mean(ot[1:min(dataFreq,obsInSample)]),1E-5);
                            lb <- c(lb,-Inf);
                            ub <- c(ub,Inf);
                        }
                        else{
                            B <- c(B,abs(matvt[1:nComponentsNonSeasonal,lagsModelMax]));
                            lb <- c(lb,1E-10);
                            ub <- c(ub,Inf);
                        }
                    }
                    if(Ttype=="A"){
                        lb <- c(lb,-Inf);
                        ub <- c(ub,Inf);
                    }
                    else if(Ttype=="M"){
                        lb <- c(lb,1E-20);
                        ub <- c(ub,3);
                    }
                    # Initial seasonals
                    if(modelIsSeasonal){
                        if(initialSeasonEstimate){
                            B <- c(B,matvt[nComponentsAll,1:lagsModelMax]);
                            if(Stype=="A"){
                                lb <- c(lb,rep(-Inf,lagsModelMax));
                                ub <- c(ub,rep(Inf,lagsModelMax));
                            }
                            else{
                                lb <- c(lb,matvt[nComponentsAll,1:lagsModelMax]*0.1);
                                ub <- c(ub,matvt[nComponentsAll,1:lagsModelMax]*10);
                            }
                        }
                    }
                }
            }

            #### Explanatory variables ####
            if(xregEstimate){
                # Initial values of at
                if(initialXEstimate){
                    B <- c(B,matat[xregNames,1]);
                    lb <- c(lb,rep(-Inf,nExovars));
                    ub <- c(ub,rep(Inf,nExovars));
                }
                if(updateX){
                    # Initials for the transition matrix
                    B <- c(B,as.vector(matFX));
                    lb <- c(lb,rep(-Inf,nExovars^2));
                    ub <- c(ub,rep(Inf,nExovars^2));

                    # Initials for the persistence matrix
                    B <- c(B,as.vector(vecgX));
                    lb <- c(lb,rep(-Inf,nExovars));
                    ub <- c(ub,rep(Inf,nExovars));
                }
            }

            # Clean and remove NAs
            B <- as.numeric(B[!is.na(B)]);
            lb <- as.numeric(lb[!is.na(lb)]);
            ub <- as.numeric(ub[!is.na(ub)]);

            return(list(B=B,lb=lb,ub=ub));
        }

        ##### Cost Function for oes #####
        CF <- function(B, lagsModel, Etype, Ttype, Stype, occurrence, damped,
                       nComponentsAll, nComponentsNonSeasonal, nExovars, lagsModelMax,
                       persistenceEstimate, initialType, phiEstimate, modelIsSeasonal, initialSeasonEstimate,
                       xregEstimate, updateX, initialXEstimate,
                       matvt, vecg, matF, matw, matat, matFX, vecgX, matxt,
                       ot, bounds){

            elements <- oesElements(B, lagsModel, Ttype, Stype, damped,
                                    nComponentsAll, nComponentsNonSeasonal, nExovars, lagsModelMax,
                                    persistenceEstimate, initialType, phiEstimate, modelIsSeasonal, initialSeasonEstimate,
                                    xregEstimate, initialXEstimate, updateX,
                                    matvt, vecg, matF, matw, matat, matFX, vecgX);

            cfRes <- occurrenceOptimizerWrap(elements$matvt, elements$matF, elements$matw, elements$vecg, ot,
                                             lagsModel, Etype, Ttype, Stype, occurrence,
                                             matxt, elements$matat, elements$matFX, elements$vecgX,
                                             bounds);

            if(is.nan(cfRes) | is.na(cfRes) | is.infinite(cfRes)){
                cfRes <- 1e+500;
            }

            return(cfRes);
        }
    }
    else if(any(occurrence==c("f","n"))){
        modelDo <- "estimate";
        model <- paste0(Etype,"NN");
    }

    ICFunction <- switch(ic,
                         "AIC"=AIC,
                         "AICc"=AICc,
                         "BIC"=BIC,
                         "BICc"=BICc);

    ##### Estimate the model #####
    if(modelDo=="estimate"){
        ##### General model - from oesg() #####
        if(occurrence=="g"){
            return(oesg(y, modelA=model, modelB=model, persistenceA=persistence, persistenceB=persistence, phiA=phi, phiB=phi,
                        initialA=initial, initialB=initial, initialSeasonA=initialSeason, initialSeasonB=initialSeason,
                        ic=ic, h=h, holdout=holdout, interval=interval, level=level, bounds=bounds,
                        silent=silent, xregA=xreg, xregB=xreg, regressorsA=regressors, regressorsB=regressors, updateXA=updateX, updateXB=updateX,
                        persistenceXA=persistenceX, persistenceXB=persistenceX, transitionXA=transitionX, transitionXB=transitionX,
                        initialXA=initialX, initialXB=initialX, ...));
        }
        ##### Fixed probability #####
        else if(occurrence=="f"){
            model <- "MNN";
            if(initialType!="o"){
                pt <- ts(matrix(rep(initial,obsInSample),obsInSample,1), start=dataStart, frequency=dataFreq);
            }
            else{
                initial <- iprob;
                pt <- ts(matrix(rep(initial,obsInSample),obsInSample,1), start=dataStart, frequency=dataFreq);
            }
            names(initial) <- "level";
            if(h>0){
                pForecast <- ts(rep(pt[1],h), start=yForecastStart, frequency=dataFreq);
            }
            else{
                pForecast <- NA;
            }
            yForecast <- log(pForecast/(1-pForecast));
            errors <- ts((ot-pt+1)/2, start=dataStart, frequency=dataFreq);
            errors[] <- log(errors / (1-errors));
            s2 <- mean(errors^2);

            # If interal is needed, transform the error and use normal distribution
            if(interval){
                if(intervalType!="l"){
                    df <- obsInSample - 1;
                }
                else{
                    df <- obsInSample;
                }
                if(df>0){
                    upperquant <- qt((1+level)/2,df=df);
                    lowerquant <- qt((1-level)/2,df=df);
                }
                else{
                    upperquant <- sqrt(1/((1-level)/2));
                    lowerquant <- -upperquant;
                }
                yUpper <- yForecast + upperquant * sqrt(s2);
                yLower <- yForecast + lowerquant * sqrt(s2);
                pUpper <- exp(yUpper) / (1+exp(yUpper));
                pLower <- exp(yLower) / (1+exp(yLower));
            }
            else{
                yUpper <- yLower <- pUpper <- pLower <- NA;
            }

            parametersNumber[1,c(1,4)] <- 1;

            output <- list(fitted=pt, forecast=pForecast, lower=pLower, upper=pUpper,
                           states=pt, lowerModel=yLower, upperModel=yUpper, forecastModel=yForecast,
                           nParam=parametersNumber, residuals=errors, y=otAll,
                           persistence=matrix(0,1,1,dimnames=list("level",NULL)),
                           initial=initial, initialSeason=NULL);
        }
        ##### Odds-ratio, inverse and direct models #####
        else if(any(occurrence==c("o","i","d"))){
            # Initialise the model
            basicparams <- oesInitialiser(Etype, Ttype, Stype, damped, phiEstimate, occurrence,
                                          dataFreq, obsInSample, obsAll, obsStates, ot,
                                          persistenceEstimate, persistence, initialType, initialValue,
                                          initialSeasonEstimate, initialSeason);
            list2env(basicparams, environment());
            # nComponentsAll, nComponentsNonSeasonal, lagsModelMax, lagsModel, matvt, vecg, matF, matw

            if(damped){
                model <- paste0(Etype,Ttype,"d",Stype);
            }
            else{
                model <- paste0(Etype,Ttype,Stype);
            }

            #### Start the optimisation ####
            if(any(c(persistenceEstimate,initialType=="o",phiEstimate,initialSeasonEstimate,
                     xregEstimate,initialXEstimate))){
                # Prepare the parameters
                B <- BValues(bounds, Ttype, Stype, damped, phiEstimate, persistenceEstimate,
                             initialType, modelIsSeasonal, initialSeasonEstimate,
                             xregEstimate, initialXEstimate, updateX,
                             lagsModelMax, nComponentsAll, nComponentsNonSeasonal,
                             vecg, matvt, matat, matFX, vecgX, xregNames, nExovars);

                if(length(B$B)>0){
                    nloptrArgs <- list(lagsModel=lagsModel, Etype=Etype, Ttype=Ttype, Stype=Stype,
                                       occurrence=occurrence,
                                       nComponentsAll=nComponentsAll,
                                       nComponentsNonSeasonal=nComponentsNonSeasonal,
                                       nExovars=nExovars, lagsModelMax=lagsModelMax, damped=damped,
                                       persistenceEstimate=persistenceEstimate,
                                       initialType=initialType, phiEstimate=phiEstimate,
                                       modelIsSeasonal=modelIsSeasonal,
                                       initialSeasonEstimate=initialSeasonEstimate,
                                       xregEstimate=xregEstimate, initialXEstimate=initialXEstimate,
                                       updateX=updateX,
                                       matvt=matvt, vecg=vecg, matF=matF, matw=matw,
                                       matat=matat, matFX=matFX, vecgX=vecgX, matxt=matxt,
                                       ot=ot, bounds=bounds);
                    opts <- list(algorithm=algorithm, xtol_rel=xtol_rel,
                                 maxeval=maxeval, print_level=print_level);

                    # Run the optimisation
                    res <- do.call(nloptr,
                                   c(list(x0=B$B, eval_f=CF, lb=B$lb, ub=B$ub, opts=opts),
                                     nloptrArgs));
                    res$call <- quote(nloptr(x0=B$B, eval_f=CF, lb=B$lb, ub=B$ub, opts=opts));

                    # If the smoothing parameters are high, try different initialisation and reestimate
                    if(persistenceEstimate && any(res$solution[c(1:(1+(Ttype!="N")*1+(Stype!="N")*1))]>0.5)){
                        B$B[c(1:(1+(Ttype!="N")*1+(Stype!="N")*1))] <- 0.01;
                        res2 <- do.call(nloptr,
                                        c(list(x0=B$B, eval_f=CF, lb=B$lb, ub=B$ub, opts=opts),
                                          nloptrArgs));
                        res2$call <- quote(nloptr(x0=B$B, eval_f=CF, lb=B$lb, ub=B$ub, opts=opts));
                        # If the new optimal is better than the old, use it
                        if(res$objective > res2$objective){
                            res <- res2;
                        }
                    }
                    B <- res$solution;
                }

                # Parameters estimated. The variance is not estimated, so not needed
                parametersNumber[1,1] <- length(B);

                # Write down phi if it was estimated
                if(damped && phiEstimate){
                    phi <- B[nComponentsAll+1];
                }
            }

            ##### Deal with the fitting and the forecasts #####
            elements <- oesElements(B, lagsModel, Ttype, Stype, damped,
                                    nComponentsAll, nComponentsNonSeasonal, nExovars, lagsModelMax,
                                    persistenceEstimate, initialType, phiEstimate, modelIsSeasonal, initialSeasonEstimate,
                                    xregEstimate, initialXEstimate, updateX,
                                    matvt, vecg, matF, matw, matat, matFX, vecgX);
            matF[] <- elements$matF;
            matw[] <- elements$matw;
            vecg[] <- elements$vecg;
            matFX[] <- elements$matFX;
            vecgX[] <- elements$vecgX;

            # Produce fitted values
            fitting <- occurenceFitterWrap(elements$matvt, matF, matw, vecg, ot,
                                           lagsModel, Etype, Ttype, Stype, occurrence,
                                           matxt, elements$matat, matFX, vecgX);
            # Write down the values in order to preserve the names, then transpose
            matvt[] <- fitting$matvt;
            matvt <- t(matvt);
            matat[] <- fitting$matat;
            matat <- t(matat);

            pFitted <- ts(fitting$pfit,start=dataStart,frequency=dataFreq);
            yFitted <- ts(fitting$yfit,start=dataStart,frequency=dataFreq);
            errors <- ts(fitting$errors,start=dataStart,frequency=dataFreq);

            if(fitting$warning){
                warning(paste0("Unreasonable values of states were produced in the estimation. ",
                               "So, we substituted them with the previous values.\nThis is because the model ETS(",
                               model,") is unstable."),
                        call.=FALSE);
            }

            ####!!! The analogue of this needs to be written in ssOccurrence.cpp, but with the correct errors ####
            errors.mat <- matrix(0,1,1);
            # if(h>0){
            #     errors.mat <- ts(errorerwrap(matvt, matF, matw, yInSample,
            #                                  h, Etype, Ttype, Stype, lagsModel,
            #                                  matxt, matat, matFX, ot),
            #                      start=dataStart,frequency=dataFreq);
            #     colnames(errors.mat) <- paste0("Error",c(1:h));
            # }
            # else{
            #     errors.mat <- NA;
            # }

            #### Produce forecasts
            # This chunk is needed in order for the default ssForecaster to work
            occurrenceOriginal <- occurrence;
            cumulative <- FALSE;
            if(h>0){
                pForecast <- rep(1, h);
            }
            else{
                pForecast <- NA;
            }
            environment(ssForecaster) <- environment();
            # This is needed for the degrees of freedom calculation
            nParam <- length(B);
            # Call the forecaster
            ssForecaster(ParentEnvironment=environment());
            # Revert to the original occurrence
            occurrence[] <- occurrenceOriginal;

            # Generate the probability forecasts from the yForecast
            pForecast[] <- switch(occurrence,
                                  "o" = switch(Etype,
                                               "M"=yForecast/(1+yForecast),
                                               "A"=exp(yForecast)/(1+exp(yForecast))),
                                  "i" = switch(Etype,
                                               "M"=1/(1+yForecast),
                                               "A"=1/(1+exp(yForecast))),
                                  "d" = sapply(sapply(as.vector(yForecast),min,1),max,0));
            # This is usually due to the exp(big number), which corresponds to 1
            if(any(occurrence==c("i","o")) && Etype=="A" && any(is.nan(pForecast))){
                pForecast[is.nan(pForecast)] <- 1;
            }

            pForecast <- ts(pForecast, start=yForecastStart, frequency=dataFreq);

            if(interval){
                pLower <- switch(occurrence,
                                      "o" = switch(Etype,
                                                   "M"=yLower/(1+yLower),
                                                   "A"=exp(yLower)/(1+exp(yLower))),
                                      "i" = switch(Etype,
                                                   "M"=1/(1+yLower),
                                                   "A"=1/(1+exp(yLower))),
                                      "d" = sapply(sapply(as.vector(yLower),min,1),max,0));
                pUpper <- switch(occurrence,
                                 "o" = switch(Etype,
                                              "M"=yUpper/(1+yUpper),
                                              "A"=exp(yUpper)/(1+exp(yUpper))),
                                 "i" = switch(Etype,
                                              "M"=1/(1+yUpper),
                                              "A"=1/(1+exp(yUpper))),
                                 "d" = sapply(sapply(as.vector(yUpper),min,1),max,0));
            }
            else{
                pUpper <- pLower <- NA;
            }

            parametersNumber[1,4] <- sum(parametersNumber[1,1:3]);
            parametersNumber[2,4] <- sum(parametersNumber[2,1:3]);

            # Merge states of vt and at if the xreg was provided
            if(!is.null(xreg)){
                matvt <- cbind(matvt,matat);
                xreg <- matxt;
            }

            if(modelIsSeasonal){
                initialSeason <- matvt[1:lagsModelMax,nComponentsAll];
            }
            else{
                initialSeason <- NULL;
            }

            #### Form the output ####
            output <- list(fitted=pFitted, forecast=pForecast, lower=pLower, upper=pUpper,
                           states=ts(matvt, start=(time(y)[1] - deltat(y)*lagsModelMax),
                                     frequency=dataFreq),
                           lowerModel=yLower, upperModel=yUpper,
                           nParam=parametersNumber, residuals=errors, y=otAll,
                           persistence=vecg, phi=phi, initial=matvt[1,1:nComponentsNonSeasonal],
                           initialSeason=initialSeason, fittedModel=yFitted, forecastModel=yForecast,
                           initialX=matat[1,], xreg=xreg, updateX=updateX, transitionX=matFX, persistenceX=vecgX,
                           B=B);
        }
        #### Automatic model selection ####
        else if(occurrence=="a"){
            occurrencePool <- c("f","o","i","d","g","n");
            occurrencePoolLength <- length(occurrencePool);
            occurrenceModels <- vector("list",occurrencePoolLength);
            for(i in 1:occurrencePoolLength){
                occurrenceModels[[i]] <- oes_old(y=y,model=model,occurrence=occurrencePool[i],
                                             ic=ic, h=h, holdout=holdout,
                                             interval=interval, level=level,
                                             bounds=bounds,
                                             silent=TRUE,
                                             xreg=xreg, regressors=regressors, updateX=updateX, ...);
            }
            ICBest <- which.min(sapply(occurrenceModels, ICFunction))[1]
            occurrence <- occurrencePool[ICBest];

            if(!silentGraph){
                graphmaker(actuals=otAll,forecast=occurrenceModels[[ICBest]]$forecast,fitted=occurrenceModels[[ICBest]]$fitted,
                           legend=!silentLegend,main=occurrenceModels[[ICBest]]$model);
            }
            return(occurrenceModels[[ICBest]]);
        }
        #### None ####
        else{
            pt <- ts(rep(1,obsInSample),start=dataStart,frequency=dataFreq);
            if(h>0){
                pForecast <- ts(rep(1,h), start=yForecastStart, frequency=dataFreq);
            }
            else{
                pForecast <- NA;
            }
            errors <- ts(ot-1, start=dataStart, frequency=dataFreq);
            parametersNumber[] <- 0;
            output <- list(fitted=pt, forecast=pForecast, lower=NA, upper=NA,
                           states=pt,
                           nParam=parametersNumber, residuals=errors, y=pt,
                           persistence=NULL, initial=NULL, initialSeason=NULL);
        }
    }
    else if(modelDo=="select"){
        if(!is.null(modelsPool)){
            modelsNumber <- length(modelsPool);
            # List for the estimated models in the pool
            results <- as.list(c(1:modelsNumber));
            j <- 0;
        }
        ##### Use branch-and-bound from es() to form the initial pool #####
        else{
            # Define the pool of models in case of "ZZZ" or "CCC" to select from
            poolErrors <- c("A","M");
            poolTrends <- c("N","A","Ad","M","Md");
            poolSeasonals <- c("N","A","M");

            if(all(Etype!=c("Z","C"))){
                poolErrors <- Etype;
            }

            # List for the estimated models in the pool
            results <- list(NA);

            ### Use brains in order to define models to estimate ###
            if(modelDo=="select" &
               (any(c(Ttype,Stype)=="X") | any(c(Ttype,Stype)=="Y") | any(c(Ttype,Stype)=="Z"))){
                if(!silentText){
                    cat("Forming the pool of models based on... ");
                }

                # poolErrorsSmall is needed for the priliminary selection
                if(Etype!="Z"){
                    poolErrors <- poolErrorsSmall <- Etype;
                }
                else{
                    poolErrorsSmall <- "A";
                }

                # Define the trends to check
                if(Ttype!="Z"){
                    if(Ttype=="X"){
                        poolTrendSmall <- c("N","A");
                        poolTrends <- c("N","A","Ad");
                        trendCheck <- TRUE;
                    }
                    else if(Ttype=="Y"){
                        poolTrendSmall <- c("N","M");
                        poolTrends <- c("N","M","Md");
                        trendCheck <- TRUE;
                    }
                    else{
                        if(damped){
                            poolTrendSmall <- paste0(Ttype,"d");
                            poolTrends <- poolTrendSmall;
                        }
                        else{
                            poolTrendSmall <- Ttype;
                            poolTrends <- Ttype;
                        }
                        trendCheck <- FALSE;
                    }
                }
                else{
                    poolTrendSmall <- c("N","A");
                    trendCheck <- TRUE;
                }

                # Define seasonality to check
                if(Stype!="Z"){
                    if(Stype=="X"){
                        poolSeasonalSmall <- c("N","A");
                        poolSeasonals <- c("N","A");
                        seasonalCheck <- TRUE;
                    }
                    else if(Stype=="Y"){
                        poolSeasonalSmall <- c("N","M");
                        poolSeasonals <- c("N","M");
                        seasonalCheck <- TRUE;
                    }
                    else{
                        poolSeasonalSmall <- Stype;
                        poolSeasonals <- Stype;
                        seasonalCheck <- FALSE;
                    }
                }
                else{
                    poolSeasonalSmall <- c("N","A","M");
                    seasonalCheck <- TRUE;
                }

                # If ZZZ, then the vector is: "ANN" "ANA" "ANM" "AAN" "AAA" "AAM"
                poolSmall <- paste0(rep(poolErrorsSmall,length(poolTrendSmall)*length(poolSeasonalSmall)),
                                    rep(poolTrendSmall,each=length(poolSeasonalSmall)),
                                    rep(poolSeasonalSmall,length(poolTrendSmall)));
                modelTested <- NULL;
                modelCurrent <- "";

                # Counter + checks for the components
                j <- 1;
                i <- 0;
                check <- TRUE;
                besti <- bestj <- 1;

                #### Branch and bound starts here ####
                while(check){
                    i <- i + 1;
                    modelCurrent[] <- poolSmall[j];
                    if(!silentText){
                        cat(paste0(modelCurrent,", "));
                    }

                    results[[i]] <- oes_old(y, model=modelCurrent, occurrence=occurrence, h=h, holdout=holdout,
                                        bounds=bounds, silent=TRUE, xreg=xreg, regressors=regressors);

                    modelTested <- c(modelTested,modelCurrent);

                    if(j>1){
                        # If the first is better than the second, then choose first
                        if(ICFunction(results[[besti]]) <= ICFunction(results[[i]])){
                            # If Ttype is the same, then we checked seasonality
                            if(substring(modelCurrent,2,2) == substring(poolSmall[bestj],2,2)){
                                poolSeasonals <- substr(modelType(results[[besti]]),
                                                        nchar(modelType(results[[besti]])),
                                                        nchar(modelType(results[[besti]])));
                                seasonalCheck <- FALSE;
                                j <- which(poolSmall!=poolSmall[bestj] &
                                               substring(poolSmall,nchar(poolSmall),nchar(poolSmall))==poolSeasonals);
                            }
                            # Otherwise we checked trend
                            else{
                                poolTrends <- substr(modelType(results[[besti]]),2,2);
                                trendCheck <- FALSE;
                            }
                        }
                        else{
                            if(substring(modelCurrent,2,2) == substring(poolSmall[besti],2,2)){
                                poolSeasonals <- poolSeasonals[poolSeasonals!=substr(modelType(results[[besti]]),
                                                                                     nchar(modelType(results[[besti]])),
                                                                                     nchar(modelType(results[[besti]])))];
                                if(length(poolSeasonals)>1){
                                    # Select another seasonal model, that is not from the previous iteration and not the current one
                                    bestj <- j;
                                    besti <- i;
                                    j <- 3;
                                }
                                else{
                                    bestj <- j;
                                    besti <- i;
                                    j <- which(substring(poolSmall,nchar(poolSmall),nchar(poolSmall))==poolSeasonals &
                                                   substring(poolSmall,2,2)!=substring(modelCurrent,2,2));
                                    seasonalCheck <- FALSE;
                                }
                            }
                            else{
                                poolTrends <- poolTrends[poolTrends!=substr(modelType(results[[besti]]),2,2)];
                                besti <- i;
                                bestj <- j;
                                trendCheck <- FALSE;
                            }
                        }

                        if(all(!c(trendCheck,seasonalCheck))){
                            check <- FALSE;
                        }
                    }
                    else{
                        j <- 2;
                    }
                }

                modelsPool <- paste0(rep(poolErrors,each=length(poolTrends)*length(poolSeasonals)),
                                     poolTrends,
                                     rep(poolSeasonals,each=length(poolTrends)));

                modelsPool <- unique(c(modelTested,modelsPool));
                modelsNumber <- length(modelsPool);
                j <- length(modelTested);
            }
            else{
                # Make the corrections in the pool for combinations
                if(all(Ttype!=c("Z","C"))){
                    if(Ttype=="Y"){
                        poolTrends <- c("N","M","Md");
                    }
                    else if(Ttype=="X"){
                        poolTrends <- c("N","A","Ad");
                    }
                    else{
                        if(damped){
                            poolTrends <- paste0(Ttype,"d");
                        }
                        else{
                            poolTrends <- Ttype;
                        }
                    }
                }
                if(all(Stype!=c("Z","C"))){
                    if(Stype=="Y"){
                        poolTrends <- c("N","M");
                    }
                    else if(Stype=="X"){
                        poolTrends <- c("N","A");
                    }
                    else{
                        poolSeasonals <- Stype;
                    }
                }

                modelsNumber <- (length(poolErrors)*length(poolTrends)*length(poolSeasonals));
                modelsPool <- paste0(rep(poolErrors,each=length(poolTrends)*length(poolSeasonals)),
                                     poolTrends,
                                     rep(poolSeasonals,each=length(poolTrends)));
                j <- 0;
            }
        }

        ##### Check models in the smaller pool #####
        if(!silentText){
            cat("Estimation progress:    ");
        }
        while(j < modelsNumber){
            j <- j + 1;
            if(!silentText){
                if(j==1){
                    cat("\b");
                }
                cat(paste0(rep("\b",nchar(round((j-1)/modelsNumber,2)*100)+1),collapse=""));
                cat(paste0(round(j/modelsNumber,2)*100,"%"));
            }

            modelCurrent <- modelsPool[j];

            results[[j]] <- oes_old(y, model=modelCurrent, occurrence=occurrence, h=h, holdout=holdout,
                                bounds=bounds, silent=TRUE, xreg=xreg, regressors=regressors);
        }

        if(!silentText){
            cat("... Done! \n");
        }

        # Write down the ICs of all the tested models
        icSelection <- sapply(results, ICFunction);
        names(icSelection) <- modelsPool;
        icSelection[is.nan(icSelection)] <- 1E100;
        icBest <- which.min(icSelection);

        output <- results[[icBest]];
        output$ICs <- icSelection;
        occurrence[] <- output$occurrence;
        model[] <- modelsPool[icBest];
        pForecast <- output$forecast;
    }
    else{
        stop("The model combination is not implemented in oes just yet", call.=FALSE);
    }

    # If there was a holdout, measure the accuracy
    if(holdout){
        yHoldout <- ts(otAll[(obsInSample+1):obsAll],start=yForecastStart,frequency=dataFreq);
        output$accuracy <- measures(yHoldout,pForecast,ot);
    }
    else{
        yHoldout <- NA;
        output$accuracy <- NA;
    }

    # Occurrence and the model name
    if(!is.null(xreg)){
        modelname <- "oETSX";
    }
    else{
        modelname <- "oETS";
    }
    output$s2 <- mean(output$residuals^2);
    output$occurrence <- switch(occurrence,
                                "f"="fixed",
                                "o"="odds-ratio",
                                "i"="inverse-odds-ratio",
                                "g"="general",
                                "d"="direct",
                                "n"="none",
                                occurrence);
    output$model <- paste0(modelname,"[",toupper(substr(occurrence,1,1)),"]","(",model,")");
    output$timeElapsed <- Sys.time()-startTime;

    ##### Make a plot #####
    if(!silentGraph){
        if(interval){
            graphmaker(actuals=otAll, forecast=output$forecast, fitted=output$fitted,
                       lower=output$lower, upper=output$upper,
                       level=level,legend=!silentLegend,main=output$model);
        }
        else{
            graphmaker(actuals=otAll,forecast=output$forecast,fitted=output$fitted,
                       legend=!silentLegend,main=output$model);
        }
    }

    # Produce log likelihood. It's the same for all the models
    pt <- output$fitted;
    if(any(c(1-pt[ot==0]==0,pt[ot==1]==0))){
        ptNew <- pt[(pt!=0) & (pt!=1)];
        otNew <- ot[(pt!=0) & (pt!=1)];
        output$logLik <- sum(log(ptNew[otNew==1])) + sum(log(1-ptNew[otNew==0]));
    }
    else{
        output$logLik <- (sum(log(pt[ot!=0])) + sum(log(1-pt[ot==0])));
    }

    # The occurrence="none" should have likelihood based on pt->1
    if(occurrence=="n"){
        output$logLik <- (obsOnes*log(1-1e-100) + (obsInSample-obsOnes)*log(1e-100));
    }

    # This is needed in order to standardise the output and make plots work
    output$loss <- "likelihood";
    output$distribution <- "plogis";

    return(structure(output,class=c("oes","smooth","occurrence")));
}
