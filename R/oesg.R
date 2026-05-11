#' Occurrence ETS, general model
#'
#' Function returns the general occurrence model of the of iETS model.
#'
#' The function estimates probability of demand occurrence, based on the iETS_G
#' state-space model. It involves the estimation and modelling of the two
#' simultaneous state space equations. Thus two parts for the model type,
#' persistence, initials etc.
#'
#' For the details about the model and its implementation, see the respective
#' vignette: \code{vignette("oes","smooth")}
#'
#' The model is based on:
#'
#' \deqn{o_t \sim Bernoulli(p_t)}
#' \deqn{p_t = \frac{a_t}{a_t+b_t}},
#'
#' where a_t and b_t are the parameters of the Beta distribution and are modelled
#' using separate ETS models.
#'
#' @template ssAuthor
#' @template ssKeywords
#'
#' @param y Either numeric vector or time series vector.
#' @param modelA The type of the ETS for the model A.
#' @param modelB The type of the ETS for the model B.
#' @param persistenceA The persistence vector \eqn{g}, containing smoothing
#' parameters used in the model A. If \code{NULL}, then estimated.
#' @param persistenceB The persistence vector \eqn{g}, containing smoothing
#' parameters used in the model B. If \code{NULL}, then estimated.
#' @param phiA The value of the dampening parameter in the model A. Used only
#' for damped-trend models.
#' @param phiB The value of the dampening parameter in the model B. Used only
#' for damped-trend models.
#' @param initialA Either \code{"o"} - optimal or the vector of initials for the
#' level and / or trend for the model A.
#' @param initialB Either \code{"o"} - optimal or the vector of initials for the
#' level and / or trend for the model B.
#' @param initialSeasonA The vector of the initial seasonal components for the
#' model A. If \code{NULL}, then it is estimated.
#' @param initialSeasonB The vector of the initial seasonal components for the
#' model B. If \code{NULL}, then it is estimated.
#' @param ic Information criteria to use in case of model selection.
#' @param h Forecast horizon.
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
#' @param xregA The vector or the matrix of exogenous variables, explaining some parts
#' of occurrence variable of the model A.
#' @param xregB The vector or the matrix of exogenous variables, explaining some parts
#' of occurrence variable of the model B.
#' @param regressorsA Variable defines what to do with the provided \code{xregA}:
#' \code{"use"} means that all of the data should be used, while
#' \code{"select"} means that a selection using \code{ic} should be done.
#' @param regressorsB Similar to the \code{regressorsA}, but for the part B of the model.
#' @param initialXA The vector of initial parameters for exogenous variables in the model
#' A. Ignored if \code{xregA} is NULL.
#' @param initialXB The vector of initial parameters for exogenous variables in the model
#' B. Ignored if \code{xregB} is NULL.
#' @param ... The parameters passed to the optimiser, such as \code{maxeval},
#' \code{xtol_rel}, \code{algorithm} and \code{print_level}. The description of
#' these is printed out by \code{nloptr.print.options()} function from the \code{nloptr}
#' package. The default values in the oes function are \code{maxeval=500},
#' \code{xtol_rel=1E-8}, \code{algorithm="NLOPT_LN_NELDERMEAD"} and \code{print_level=0}.
#' @return The object of class "occurrence" is returned. It contains following list of
#' values:
#'
#' \itemize{
#' \item \code{modelA} - the model A of the class oes, that contains the output similar
#' to the one from the \code{oes()} function;
#' \item \code{modelB} - the model B of the class oes, that contains the output similar
#' to the one from the \code{oes()} function.
#' \item \code{B} - the vector of all the estimated parameters.
#' }
#' @seealso \code{\link[smooth]{es}, \link[smooth]{oes}}
#' @keywords iss intermittent demand intermittent demand state space model
#' exponential smoothing forecasting
#' @examples
#'
#' y <- rpois(100,0.1)
#' oesg(y, modelA="MNN", modelB="ANN")
#'
#' @export
oesg_old <- function(y, modelA="MNN", modelB="MNN", persistenceA=NULL, persistenceB=NULL,
                 phiA=NULL, phiB=NULL,
                 initialA="optimal", initialB="optimal", initialSeasonA=NULL, initialSeasonB=NULL,
                 ic=c("AICc","AIC","BIC","BICc"), h=10, holdout=FALSE,
                 # interval=c("none","parametric","likelihood","semiparametric","nonparametric"), level=0.95,
                 bounds=c("usual","admissible","none"),
                 silent=c("all","graph","legend","output","none"),
                 xregA=NULL, xregB=NULL, initialXA=NULL, initialXB=NULL,
                 regressorsA=c("use","select"), regressorsB=c("use","select"),
                 ...){
    # Function returns the occurrence part of the intermittent state space model, type G

    # A fix for a weird case of selection, when initial disappears.
    # I don't have time to fix it right now...
    if(is.null(initialA) || is.null(initialB)){
        initialA <- initialB <- "optimal";
    }

# Start measuring the time of calculations
    startTime <- Sys.time();

    # Set the defaults for the parameters that are no longer supported
    interval <- "none";
    level <- 0.95;
    updateXA <- FALSE;
    updateXB <- FALSE;
    transitionXA <- NULL;
    transitionXB <- NULL;
    persistenceXA <- NULL;
    persistenceXB <- NULL;

    ##### Preparations #####
    occurrence <- "g";

    if(is.smooth.sim(y)){
        y <- y$data;
    }

    # Add all the variables in ellipsis to current environment
    # list2env(list(...),environment());
    ellipsis <- list(...);
    ellipsis <- depricator(ellipsis, "xregDo", "regressors");

    # If OES_G was provided as either modelA or modelB, deal with it
    if(is.oesg(modelA)){
        modelB <- modelA$modelB;
        modelA <- modelA$modelA;
    }
    else if(is.oesg(modelB)){
        modelA <- modelB$modelA;
        modelB <- modelB$modelB;
    }

    if(is.oes(modelA)){
        persistenceA <- modelA$persistence;
        phiA <- modelA$phi;
        initialA <- modelA$initial;
        initialSeasonA <- modelA$initialSeason;
        xregA <- modelA$xreg;
        initialXA <- modelA$initialX;
        transitionXA <- modelA$transitionX;
        persistenceXA <- modelA$persistenceX;
        if(any(c(persistenceXA)!=0) | any((transitionXA!=0)&(transitionXA!=1))){
            updateXA <- TRUE;
        }
        modelA <- modelType(modelA);
    }
    if(is.oes(modelB)){
        persistenceB <- modelB$persistence;
        phiB <- modelB$phi;
        initialB <- modelB$initial;
        initialSeasonB <- modelB$initialSeason;
        xregB <- modelB$xreg;
        initialXB <- modelB$initialX;
        transitionXB <- modelB$transitionX;
        persistenceXB <- modelB$persistenceX;
        if(any(c(persistenceXB)!=0) | any((transitionXB!=0)&(transitionXB!=1))){
            updateXB <- TRUE;
        }
        modelB <- modelType(modelB);
    }

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

    #### First call for the environment ####
    ## Set environment for ssInput and make all the checks
    model <- modelA;
    persistence <- persistenceA;
    phi <- phiA;
    initial <- initialA;
    initialSeason <- initialSeasonA;
    xreg <- xregA;
    regressors <- regressorsA;

    environment(ssInput) <- environment();
    ssInput("oes",ParentEnvironment=environment());
    regressorsA <- regressors;

    ### Produce vectors with zeroes and ones, fixed probability and the number of ones.
    ot <- (yInSample!=0)*1;
    otAll <- (y!=0)*1;
    iprob <- mean(ot);
    obsOnes <- sum(ot);

    if(all(ot==ot[1])){
        warning(paste0("There is no variability in the occurrence of the variable in-sample.\n",
                       "Switching to occurrence='none'."),call.=FALSE)
        return(oes(y,occurrence="n"));
    }

    ### Prepare exogenous variables
    xregdata <- ssXreg(y=otAll, Etype="A", xreg=xregA, updateX=updateXA, ot=rep(1,obsInSample),
                       persistenceX=persistenceXA, transitionX=transitionXA, initialX=initialXA,
                       obsInSample=obsInSample, obsAll=obsAll, obsStates=obsStates,
                       lagsModelMax=1, h=h, regressors=regressorsA, silent=silentText,
                       allowMultiplicative=FALSE);

    ### Write down all the values in the model A
    # From the ssInput
    obsStatesA <- obsStates;
    initialValueA <- initialValue;
    initialTypeA <- initialType;
    initialSeasonEstimateA <- initialSeasonEstimate;
    modelIsSeasonalA <- modelIsSeasonal;
    initialSeasonA <- initialSeason;
    modelA <- model;
    modelsPoolA <- modelsPool;
    EtypeA <- Etype;
    TtypeA <- Ttype;
    StypeA <- Stype;
    persistenceA <- persistence;
    persistenceEstimateA <- persistenceEstimate;
    dampedA <- damped;
    phiA <- phi;
    phiEstimateA <- phiEstimate;
    modelDoA <- modelDo;
    regressorsA <- regressors;
    parametersNumberA <- parametersNumber;

    # From the ssXreg
    nExovarsA <- xregdata$nExovars;
    matxtA <- xregdata$matxt;
    matatA <- t(xregdata$matat);
    xregEstimateA <- xregdata$xregEstimate;
    matFXA <- xregdata$matFX;
    vecgXA <- xregdata$vecgX;
    xregNamesA <- colnames(matxtA);
    xregA <- xregdata$xreg;
    initialXEstimateA <- xregdata$initialXEstimate;

    #### Second call for the environment ####
    ## Set environment for ssInput and make all the checks
    model <- modelB;
    persistence <- persistenceB;
    phi <- phiB;
    initial <- initialB;
    initialSeason <- initialSeasonB;
    xreg <- xregB;
    regressors <- regressorsB;

    environment(ssInput) <- environment();
    ssInput("oes",ParentEnvironment=environment());
    regressorsB <- regressors;

    ### Prepare exogenous variables
    xregdata <- ssXreg(y=1-otAll, Etype="A", xreg=xregB, updateX=updateXB, ot=rep(1,obsInSample),
                       persistenceX=persistenceXB, transitionX=transitionXB, initialX=initialXB,
                       obsInSample=obsInSample, obsAll=obsAll, obsStates=obsStates,
                       lagsModelMax=1, h=h, regressors=regressorsB, silent=silentText,
                       allowMultiplicative=FALSE);

    ### Write down all the values in the model B
    # From the ssInput
    obsStatesB <- obsStates;
    initialValueB <- initialValue;
    initialTypeB <- initialType;
    initialSeasonEstimateB <- initialSeasonEstimate;
    modelIsSeasonalB <- modelIsSeasonal;
    initialSeasonB <- initialSeason;
    modelB <- model;
    modelsPoolB <- modelsPool;
    EtypeB <- Etype;
    TtypeB <- Ttype;
    StypeB <- Stype;
    persistenceB <- persistence;
    persistenceEstimateB <- persistenceEstimate;
    dampedB <- damped;
    phiB <- phi;
    phiEstimateB <- phiEstimate;
    modelDoB <- modelDo;
    regressorsB <- regressors;
    parametersNumberB <- parametersNumber;

    # From the ssXreg
    nExovarsB <- xregdata$nExovars;
    matxtB <- xregdata$matxt;
    matatB <- t(xregdata$matat);
    xregEstimateB <- xregdata$xregEstimate;
    matFXB <- xregdata$matFX;
    vecgXB <- xregdata$vecgX;
    xregNamesB <- colnames(matxtB);
    xregB <- xregdata$xreg;
    initialXEstimateB <- xregdata$initialXEstimate;

    #### The functions for the model ####
    ##### Initialiser of oes. This is called separately for each model #####
    # This creates the states, transition, persistence and measurement matrices
    oesInitialiser <- function(Etype, Ttype, Stype, damped,
                               dataFreq, obsInSample, obsAll, obsStates, ot,
                               persistenceEstimate, persistence, initialType, initialValue,
                               initialSeasonEstimate, initialSeason, modelType){
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

        # Persistence vector. The initials are set here!
        if(persistenceEstimate){
            vecg <- matrix(0.1, nComponentsAll, 1);
        }
        else{
            vecg <- matrix(persistence, nComponentsAll, 1);
        }
        rownames(vecg) <- statesNames;

        # Measurement vector
        matw <- matrix(1, 1, nComponentsAll, dimnames=list(NULL, statesNames));

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
            if(modelType=="A"){
                initialStates[1] <- initialStates[1] / (1 - initialStates[1]);
            }
            else{
                initialStates[1] <- (1-initialStates[1]) / initialStates[1];
            }
            # Initials specifically for ETS(A,M,N) and alike
            if(Etype=="A" && Ttype=="M" && (initialStates[1]>1)){
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
                # If there are some boundary values, move them a bit
                if(any(abs(initialSeasonValue)==1)){
                    initialSeasonValue[initialSeasonValue==1] <- 1 - 1E-10;
                    initialSeasonValue[initialSeasonValue==-1] <- -1 + 1E-10;
                }
                # Transform this into the probability scale
                initialSeasonValue <- (initialSeasonValue + 1) / 2;
                if(modelType=="A"){
                    initialSeasonValue <- initialSeasonValue / (1 - initialSeasonValue);
                }
                else{
                    initialSeasonValue <- (1 - initialSeasonValue) / initialSeasonValue;
                }

                # Transform to the adequate scale
                if(Stype=="A"){
                    initialSeasonValue <- log(initialSeasonValue);
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
                i[] <- i + nExovars;
            }
        }

        return(list(vecg=vecg, matF=matF, matw=matw, matvt=matvt,
                    matat=matat, matFX=matFX, vecgX=vecgX, B=B[-c(1:i)], i=i));
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
    CF <- function(B, ot, bounds,
                   # The parameters of the model A
                   lagsModelA, EtypeA, TtypeA, StypeA, dampedA,
                   nComponentsAllA, nComponentsNonSeasonalA, nExovarsA, lagsModelMaxA,
                   persistenceEstimateA, initialTypeA, phiEstimateA, initialSeasonEstimateA,
                   xregEstimateA, initialXEstimateA, updateXA,
                   matvtA, vecgA, matFA, matwA, matatA, matFXA, vecgXA, matxtA,
                   # The parameters of the model B
                   lagsModelB, EtypeB, TtypeB, StypeB, dampedB,
                   nComponentsAllB, nComponentsNonSeasonalB, nExovarsB, lagsModelMaxB,
                   persistenceEstimateB, initialTypeB, phiEstimateB, initialSeasonEstimateB,
                   xregEstimateB, initialXEstimateB, updateXB,
                   matvtB, vecgB, matFB, matwB, matatB, matFXB, vecgXB, matxtB){

        # Extract elements for each model
        elementsA <- oesElements(B, lagsModelA, TtypeA, StypeA, dampedA,
                                 nComponentsAllA, nComponentsNonSeasonalA, nExovarsA, lagsModelMaxA,
                                 persistenceEstimateA, initialTypeA, phiEstimateA,
                                 modelIsSeasonalA, initialSeasonEstimateA,
                                 xregEstimateA, initialXEstimateA, updateXA,
                                 matvtA, vecgA, matFA, matwA, matatA, matFXA, vecgXA);
        elementsB <- oesElements(elementsA$B, lagsModelB, TtypeB, StypeB, dampedB,
                                 nComponentsAllB, nComponentsNonSeasonalB, nExovarsB, lagsModelMaxB,
                                 persistenceEstimateB, initialTypeB, phiEstimateB,
                                 modelIsSeasonalB, initialSeasonEstimateB,
                                 xregEstimateB, initialXEstimateB, updateXB,
                                 matvtB, vecgB, matFB, matwB, matatB, matFXB, vecgXB);

        # Calculate the CF value
        cfRes <- occurrenceGeneralOptimizerWrap(ot, bounds,
                                                lagsModelA, EtypeA, TtypeA, StypeA,
                                                elementsA$matvt, elementsA$matF, elementsA$matw, elementsA$vecg,
                                                matxtA, elementsA$matat, elementsA$matFX, elementsA$vecgX,
                                                lagsModelB, EtypeB, TtypeB, StypeB,
                                                elementsB$matvt, elementsB$matF, elementsB$matw, elementsB$vecg,
                                                matxtB, elementsB$matat, elementsB$matFX, elementsB$vecgX);

        if(is.nan(cfRes) | is.na(cfRes) | is.infinite(cfRes)){
            cfRes <- 1e+500;
        }

        return(cfRes);
    }

    ##### Start the calculations #####
    if(modelDo=="estimate"){
        # Initialise the model
        basicparamsA <- oesInitialiser(EtypeA, TtypeA, StypeA, dampedA,
                                       dataFreq, obsInSample, obsAll, obsStatesA, ot,
                                       persistenceEstimateA, persistenceA, initialTypeA, initialValueA,
                                       initialSeasonEstimateA, initialSeasonA, modelType="A");
        basicparamsB <- oesInitialiser(EtypeB, TtypeB, StypeB, dampedB,
                                       dataFreq, obsInSample, obsAll, obsStatesB, ot,
                                       persistenceEstimateB, persistenceB, initialTypeB, initialValueB,
                                       initialSeasonEstimateB, initialSeasonB, modelType="B");

        # Write down the names of each model
        if(dampedA){
            modelA <- paste0(EtypeA,TtypeA,"d",StypeA);
        }
        else{
            modelA <- paste0(EtypeA,TtypeA,StypeA);
        }
        if(dampedB){
            modelB <- paste0(EtypeB,TtypeB,"d",StypeB);
        }
        else{
            modelB <- paste0(EtypeB,TtypeB,StypeB);
        }

        #### Start the optimisation ####
        if(any(c(persistenceEstimateA,initialTypeA=="o",phiEstimateA,initialSeasonEstimateA,xregEstimateA,initialXEstimateA,
                 persistenceEstimateB,initialTypeB=="o",phiEstimateB,initialSeasonEstimateB,xregEstimateB,initialXEstimateB))){
            # Prepare the parameters of the two models
            BValuesA <- BValues(bounds, TtypeA, StypeA, dampedA, phiEstimateA, persistenceEstimateA,
                                initialTypeA, modelIsSeasonalA, initialSeasonEstimateA,
                                xregEstimateA, initialXEstimateA, updateXA,
                                basicparamsA$lagsModelMax, basicparamsA$nComponentsAll, basicparamsA$nComponentsNonSeasonal,
                                basicparamsA$vecg, basicparamsA$matvt, matatA,
                                matFXA, vecgXA, xregNamesA, nExovarsA);
            BValuesB <- BValues(bounds, TtypeB, StypeB, dampedB, phiEstimateB, persistenceEstimateB,
                                initialTypeB, modelIsSeasonalB, initialSeasonEstimateB,
                                xregEstimateB, initialXEstimateB, updateXB,
                                basicparamsB$lagsModelMax, basicparamsB$nComponentsAll, basicparamsB$nComponentsNonSeasonal,
                                basicparamsB$vecg, basicparamsB$matvt, matatB,
                                matFXB, vecgXB, xregNamesB, nExovarsB);


            # This is needed for the degrees of freedom calculation
            nParamA <- length(BValuesA$B);
            # This is needed for the degrees of freedom calculation
            nParamB <- length(BValuesB$B);

            if(length(BValuesA$B)>0 && length(BValuesB$B)>0){
                nloptrArgs <- list(ot=ot, bounds=bounds,
                                   lagsModelA=basicparamsA$lagsModel,
                                   EtypeA=EtypeA, TtypeA=TtypeA, StypeA=StypeA, dampedA=dampedA,
                                   nComponentsAllA=basicparamsA$nComponentsAll,
                                   nComponentsNonSeasonalA=basicparamsA$nComponentsNonSeasonal,
                                   nExovarsA=nExovarsA, lagsModelMaxA=basicparamsA$lagsModelMax,
                                   persistenceEstimateA=persistenceEstimateA,
                                   initialTypeA=initialTypeA, phiEstimateA=phiEstimateA,
                                   initialSeasonEstimateA=initialSeasonEstimateA,
                                   xregEstimateA=xregEstimateA, initialXEstimateA=initialXEstimateA,
                                   updateXA=updateXA,
                                   matvtA=basicparamsA$matvt, vecgA=basicparamsA$vecg,
                                   matFA=basicparamsA$matF, matwA=basicparamsA$matw,
                                   matatA=matatA, matFXA=matFXA, vecgXA=vecgXA, matxtA=matxtA,
                                   lagsModelB=basicparamsB$lagsModel,
                                   EtypeB=EtypeB, TtypeB=TtypeB, StypeB=StypeB, dampedB=dampedB,
                                   nComponentsAllB=basicparamsB$nComponentsAll,
                                   nComponentsNonSeasonalB=basicparamsB$nComponentsNonSeasonal,
                                   nExovarsB=nExovarsB, lagsModelMaxB=basicparamsB$lagsModelMax,
                                   persistenceEstimateB=persistenceEstimateB,
                                   initialTypeB=initialTypeB, phiEstimateB=phiEstimateB,
                                   initialSeasonEstimateB=initialSeasonEstimateB,
                                   xregEstimateB=xregEstimateB, initialXEstimateB=initialXEstimateB,
                                   updateXB=updateXB,
                                   matvtB=basicparamsB$matvt, vecgB=basicparamsB$vecg,
                                   matFB=basicparamsB$matF, matwB=basicparamsB$matw,
                                   matatB=matatB, matFXB=matFXB, vecgXB=vecgXB, matxtB=matxtB);
                opts <- list(algorithm=algorithm, xtol_rel=xtol_rel,
                             maxeval=maxeval, print_level=print_level);

                # Run the optimisation
                res <- do.call(nloptr,
                               c(list(x0=c(BValuesA$B, BValuesB$B), eval_f=CF,
                                      lb=c(BValuesA$lb, BValuesB$lb),
                                      ub=c(BValuesA$ub, BValuesB$ub), opts=opts),
                                 nloptrArgs));
                res$call <- quote(nloptr(x0=c(BValuesA$B, BValuesB$B), eval_f=CF,
                                         lb=c(BValuesA$lb, BValuesB$lb),
                                         ub=c(BValuesA$ub, BValuesB$ub), opts=opts));

                B <- res$solution;
            }
            else{
                B <- c(BValuesA$B,BValuesB$B);
            }
        }

        ##### Deal with the fitting and the forecasts #####
        elementsA <- oesElements(B, basicparamsA$lagsModel, TtypeA, StypeA, dampedA,
                                 basicparamsA$nComponentsAll, basicparamsA$nComponentsNonSeasonal,
                                 nExovarsA, basicparamsA$lagsModelMax,
                                 persistenceEstimateA, initialTypeA, phiEstimateA,
                                 modelIsSeasonalA, initialSeasonEstimateA,
                                 xregEstimateA, initialXEstimateA, updateXA,
                                 basicparamsA$matvt, basicparamsA$vecg, basicparamsA$matF, basicparamsA$matw,
                                 matatA, matFXA, vecgXA);
        # Write down phi if it was estimated
        if(dampedA && phiEstimateA){
            phi <- B[basicparamsA$nComponentsAll+1];
        }
        parametersNumberA[1,1] <- elementsA$i;
        matvtA <- elementsA$matvt;
        matFA <- elementsA$matF;
        matwA <- elementsA$matw;
        vecgA <- elementsA$vecg;
        matFXA[] <- elementsA$matFX;
        vecgXA[] <- elementsA$vecgX;
        if(dampedB && phiEstimateB){
            phiB <- B[elementsA$i+basicparamsB$nComponentsAll+1];
        }
        elementsB <- oesElements(elementsA$B, basicparamsB$lagsModel, TtypeB, StypeB, dampedB,
                                 basicparamsB$nComponentsAll, basicparamsB$nComponentsNonSeasonal,
                                 nExovarsB, basicparamsB$lagsModelMax,
                                 persistenceEstimateB, initialTypeB, phiEstimateB,
                                 modelIsSeasonalB, initialSeasonEstimateB,
                                 xregEstimateB, initialXEstimateB, updateXB,
                                 basicparamsB$matvt, basicparamsB$vecg, basicparamsB$matF, basicparamsB$matw,
                                 matatB, matFXB, vecgXB);
        parametersNumberB[1,1] <- elementsB$i;
        matvtB <- elementsB$matvt;
        matFB <- elementsB$matF;
        matwB <- elementsB$matw;
        vecgB <- elementsB$vecg;
        matFXB[] <- elementsB$matFX;
        vecgXB[] <- elementsB$vecgX;

        # Produce fitted values
        fitting <- occurenceGeneralFitterWrap(ot,
                                              basicparamsA$lagsModel, EtypeA, TtypeA, StypeA,
                                              elementsA$matvt, matFA, matwA, vecgA,
                                              matxtA, elementsA$matat, matFXA, vecgXA,
                                              basicparamsB$lagsModel, EtypeB, TtypeB, StypeB,
                                              elementsB$matvt, matFB, matwB, vecgB,
                                              matxtB, elementsB$matat, matFXB, vecgXB)
        matvtA[] <- fitting$matvtA;
        matvtB[] <- fitting$matvtB;
        matatA[] <- fitting$matatA;
        matatB[] <- fitting$matatB;
        pFitted <- ts(fitting$pfit,start=dataStart,frequency=dataFreq);
        aFitted <- ts(fitting$afit,start=dataStart,frequency=dataFreq);
        bFitted <- ts(fitting$bfit,start=dataStart,frequency=dataFreq);
        errorsA <- ts(fitting$errorsA,start=dataStart,frequency=dataFreq);
        errorsB <- ts(fitting$errorsB,start=dataStart,frequency=dataFreq);

        if(fitting$warning){
            warning(paste0("Unreasonable values of states were produced in the estimation. ",
                           "So, we substituted them with the previous values.\nThis is because the model exhibited explosive behaviour."),
                    call.=FALSE);
        }

        # Produce forecasts
        if(h>0){
            nParam <- nParamA;
            # aForecast is the underlying forecast of the model A
            aForecast <- as.vector(forecasterwrap(t(matvtA[,(obsInSample+1):(obsInSample+basicparamsA$lagsModelMax),drop=FALSE]),
                                                  matFA, matwA, h, EtypeA, TtypeA, StypeA, basicparamsA$lagsModel,
                                                  matxtA[(obsAll-h+1):(obsAll),,drop=FALSE],
                                                  t(matatA[,(obsAll-h+1):(obsAll),drop=FALSE]), matFXA));

            nParam <- nParamB;
            # bForecast is the underlying forecast of the model B
            bForecast <- as.vector(forecasterwrap(t(matvtB[,(obsInSample+1):(obsInSample+basicparamsB$lagsModelMax),drop=FALSE]),
                                                  matFB, matwB, h, EtypeB, TtypeB, StypeB, basicparamsB$lagsModel,
                                                  matxtB[(obsAll-h+1):(obsAll),,drop=FALSE],
                                                  t(matatB[,(obsAll-h+1):(obsAll),drop=FALSE]), matFXB));

            # pForecast is the final probability forecast
            if(EtypeA=="M" && !any(is.nan(aForecast)) && any(aForecast<0)){
                aForecast[aForecast<=0] <- 1E-10;
                warning(paste0("Negative values were produced in the forecast of the model A. ",
                               "This is unreasonable for the model with the multiplicative error, so we trimmed them out."),
                        call.=FALSE);
            }
            if(any(is.nan(aForecast)) || any(is.infinite(aForecast))){
                aForecast[is.nan(aForecast)] <- aForecast[which(!is.nan(aForecast))[sum(!is.nan(aForecast))]];
                aForecast[is.infinite(aForecast)] <- aForecast[which(is.finite(aForecast))[sum(is.finite(aForecast))]];
                warning("The model A has exploded. We substituted those values with ones.",
                        call.=FALSE);
            }
            if(EtypeB=="M" && !any(is.nan(bForecast)) && any(bForecast<0)){
                bForecast[bForecast<=0] <- 1E-10;
                warning(paste0("Negative values were produced in the forecast of the model B. ",
                               "This is unreasonable for the model with the multiplicative error, so we trimmed them out."),
                        call.=FALSE);
            }
            if(any(is.nan(bForecast)) || any(is.infinite(bForecast))){
                bForecast[is.nan(bForecast)] <- bForecast[which(!is.nan(bForecast))[sum(!is.nan(bForecast))]];
                bForecast[is.infinite(bForecast)] <- bForecast[which(is.finite(bForecast))[sum(is.finite(bForecast))]];
                warning("The model B has exploded. We substituted those values with ones.",
                        call.=FALSE);
            }

            pForecast <- switch(EtypeA,
                                "M" = switch(EtypeB,
                                             "M"=aForecast/(aForecast+bForecast),
                                             "A"=aForecast/(aForecast+exp(bForecast))),
                                "A" = switch(Etype,
                                             "M"=exp(aForecast)/(exp(aForecast)+bForecast),
                                             "A"=exp(aForecast)/(exp(aForecast)+exp(bForecast))));
            # This is usually due to the exp(big number), which corresponds to 1
            if(any(c(EtypeA,EtypeB)=="A") && any(is.nan(pForecast))){
                pForecast[is.nan(pForecast)] <- 1;
            }

            pForecast <- ts(pForecast, start=yForecastStart, frequency=dataFreq);
            aForecast <- ts(aForecast, start=yForecastStart, frequency=dataFreq);
            bForecast <- ts(bForecast, start=yForecastStart, frequency=dataFreq);
        }
        else{
            aForecast <- bForecast <- pForecast <- ts(NA,start=yForecastStart,frequency=dataFreq);
        }

        parametersNumberA[1,4] <- sum(parametersNumberA[1,1:3]);
        parametersNumberA[2,4] <- sum(parametersNumberA[2,1:3]);

        parametersNumberB[1,4] <- sum(parametersNumberB[1,1:3]);
        parametersNumberB[2,4] <- sum(parametersNumberB[2,1:3]);

        if(holdout){
            yHoldout <- ts(otAll[(obsInSample+1):obsAll],start=yForecastStart,frequency=dataFreq);
            errormeasures <- measures(yHoldout,pForecast,ot);
        }
        else{
            yHoldout <- NA;
            errormeasures <- NA;
        }
    }
    else{
        stop("The model selection and combinations are not implemented in oesg() just yet", call.=FALSE);
    }

    # Merge states of vt and at if the xreg was provided
    if(!is.null(xregA)){
        matvtA <- rbind(matvtA,matatA);
        xregA <- matxtA;
    }
    if(!is.null(xregB)){
        matvtB <- rbind(matvtB,matatB);
        xregB <- matxtB;
    }

    # Prepare the initial seasonals
    if(modelIsSeasonalA){
        initialSeasonA <- matvtA[basicparamsA$nComponentsAll,1:basicparamsA$lagsModelMax];
    }
    else{
        initialSeasonA <- NULL;
    }
    if(modelIsSeasonalB){
        initialSeasonB <- matvtB[basicparamsB$nComponentsAll,1:basicparamsB$lagsModelMax];
    }
    else{
        initialSeasonB <- NULL;
    }

    # Occurrence and model name
    if(!is.null(xregA)){
        modelnameA <- "oETSX";
    }
    else{
        modelnameA <- "oETS";
    }
    if(!is.null(xregB)){
        modelnameB <- "oETSX";
    }
    else{
        modelnameB <- "oETS";
    }

    #### Prepare the output ####
    # Prepare two models
    modelA <- list(model=paste0(modelnameA,"[G](",modelA,")_A"), y=aFitted+errorsA,
                   states=ts(t(matvtA), start=(time(y)[1] - deltat(y)*basicparamsA$lagsModelMax),
                             frequency=dataFreq),
                   nParam=parametersNumberA, residuals=errorsA, occurrence="odds-ratio",
                   persistence=vecgA, phi=phiA, initial=matvtA[1:basicparamsA$nComponentsNonSeasonal,1],
                   initialSeason=initialSeasonA, s2=mean(errorsA^2), loss="likelihood",
                   fittedModel=aFitted, forecastModel=aForecast,
                   initialX=matatA[,1], xreg=xregA);
    class(modelA) <- c("oes","smooth");

    modelB <- list(model=paste0(modelnameB,"[G](",modelB,")_B"), y=bFitted+errorsB,
                   states=ts(t(matvtB), start=(time(y)[1] - deltat(y)*basicparamsB$lagsModelMax),
                             frequency=dataFreq),
                   nParam=parametersNumberB, residuals=errorsB, occurrence="inverse-odds-ratio",
                   persistence=vecgB, phi=phiB, initial=matvtB[1:basicparamsB$nComponentsNonSeasonal,1],
                   initialSeason=initialSeasonB, s2=mean(errorsB^2), loss="likelihood",
                   fittedModel=bFitted, forecastModel=bForecast,
                   initialX=matatB[,1], xreg=xregB);
    class(modelB) <- c("oes","smooth");

    # Occurrence and model name
    if(!is.null(xreg)){
        modelname <- "oETSX";
    }
    else{
        modelname <- "oETS";
    }
    # Start forming the output
    output <- list(model=paste0(modelname,"[G](",modelType(modelA),")(",modelType(modelB),")"), occurrence="general", y=otAll,
                   fitted=pFitted, forecast=pForecast, lower=NA, upper=NA,
                   modelA=modelA, modelB=modelB,
                   nParam=parametersNumberA+parametersNumberB);
    output$timeElapsed <- Sys.time()-startTime;

    # If there was a holdout, measure the accuracy
    if(holdout){
        yHoldout <- ts(otAll[(obsInSample+1):obsAll],start=yForecastStart,frequency=dataFreq);
        output$accuracy <- measures(yHoldout,pForecast,ot);
    }
    else{
        yHoldout <- NA;
        output$accuracy <- NA;
    }

    ##### Make a plot #####
    if(!silentGraph){
        # if(interval){
        #     graphmaker(actuals=otAll, forecast=yForecastNew, fitted=pFitted, lower=yLowerNew, upper=yUpperNew,
        #                level=level,legend=!silentLegend,main=output$model);
        # }
        # else{
        graphmaker(actuals=otAll,forecast=pForecast,fitted=pFitted,
                   legend=!silentLegend,main=output$model);
        # }
    }

    # Produce log likelihood
    pt <- output$fitted;
    if(any(c(1-pt[ot==0]==0,pt[ot==1]==0))){
        ptNew <- pt[(pt!=0) & (pt!=1)];
        otNew <- ot[(pt!=0) & (pt!=1)];
        output$logLik <- sum(log(ptNew[otNew==1])) + sum(log(1-ptNew[otNew==0]));
    }
    else{
        output$logLik <- (sum(log(pt[ot!=0])) + sum(log(1-pt[ot==0])));
    }
    output$modelA$logLik <- output$modelB$logLik <- output$logLik;

    # This is needed in order to standardise the output and make plots work
    output$loss <- "likelihood";
    output$distribution <- "plogis";
    output$B <- B;
    return(structure(output,class=c("oesg","oes","occurrence","smooth")));
}

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
                 initial=c("backcasting","optimal","two-stage","complete"),
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
