#' Simulate SSARIMA
#'
#' Function generates data using SSARIMA with Single Source of Error as a data
#' generating process.
#'
#' For the information about the function, see the vignette:
#' \code{vignette("simulate","smooth")}
#'
#' @template ssSimParam
#' @template ssAuthor
#' @template ssKeywords
#'
#' @template ssGeneralRef
#' @template ssARIMARef
#'
#' @param orders List of orders, containing vector variables \code{ar},
#' \code{i} and \code{ma}. Example:
#' \code{orders=list(ar=c(1,2),i=c(1),ma=c(1,1,1))}. If a variable is not
#' provided in the list, then it is assumed to be equal to zero. At least one
#' variable should have the same length as \code{lags}.
#' @param lags Defines lags for the corresponding orders (see examples above).
#' The length of \code{lags} must correspond to the length of \code{orders}.
#' There is no restrictions on the length of \code{lags} vector.
#' It is recommended to order \code{lags} ascending.
#' @param initial Vector of initial values for state matrix. If \code{NULL},
#' then generated using advanced, sophisticated technique - uniform
#' distribution.
#' @param arma List with ar/ma parameters. The order of parameters should
#' be lag-wise. This means that first all the AR parameters of the firs lag
#' should be passed, then for the second etc. AR of another ssarima can be
#' passed here.
#' @param constant If \code{TRUE}, constant term is included in the model. Can
#' also be a number (constant value).
#' @param bounds Type of bounds to use for AR and MA if values are generated.
#' \code{"admissible"} - bounds guaranteeing stability and stationarity of
#' SSARIMA. \code{"none"} - we generate something, but do not guarantee
#' stationarity and stability. Using first letter of the type of bounds also
#' works.
#' @param ...  Additional parameters passed to the chosen randomizer. All the
#' parameters should be passed in the order they are used in chosen randomizer.
#' For example, passing just \code{sd=0.5} to \code{rnorm} function will lead
#' to the call \code{rnorm(obs, mean=0.5, sd=1)}.
#'
#' @return List of the following values is returned:
#' \itemize{
#' \item \code{model} - Name of SSARIMA model.
#' \item \code{arma} - List of AR/MA parameters. If \code{nsim>1}, then this is a
#' list of matrices.
#' \item \code{constant} - Value of constant term. If \code{nsim>1}, then this
#' is a vector.
#' \item \code{initial} - Initial values of SSARIMA. If \code{nsim>1}, then this
#' is a matrix.
#' \item \code{profile} - The final profile produced in the simulation.
#' \item \code{data} - Time series vector (or matrix if \code{nsim>1}) of the
#' generated series.
#' \item \code{states} - Matrix (or array if \code{nsim>1}) of states. States
#' are in columns, time is in rows.
#' \item \code{residuals} - Error terms used in the simulation. Either vector or
#' matrix, depending on \code{nsim}.
#' \item \code{occurrence} - Values of occurrence variable. Once again, can be
#' either a vector or a matrix...
#' \item \code{logLik} - Log-likelihood of the constructed model.
#' }
#'
#' @seealso \code{\link[smooth]{sim.es}, \link[smooth]{ssarima},
#' \link[stats]{Distributions}, \link[smooth]{orders}}
#'
#' @examples
#'
#' # Create 120 observations from ARIMA(1,1,1) with drift. Generate 100 time series of this kind.
#' x <- sim.ssarima(orders=c(1,1,1),obs=120,nsim=100,constant=TRUE)
#'
#' # Generate similar thing for seasonal series of SARIMA(1,1,1)(0,0,2)_4
#' x <- sim.ssarima(orders=list(ar=c(1,0),i=c(1,0),ma=c(1,2)),lags=c(1,4),
#'                  frequency=4,obs=80,nsim=100,constant=FALSE)
#'
#' # Generate 10 series of high frequency data from SARIMA(1,0,2)_1(0,1,1)_7(1,0,1)_30
#' x <- sim.ssarima(orders=list(ar=c(1,0,1),i=c(0,1,0),ma=c(2,1,1)),lags=c(1,7,30),
#'                  obs=360,nsim=10)
#'
#'
#' @export sim.ssarima
sim.ssarima <- function(orders=list(ar=0,i=1,ma=1), lags=1,
                        obs=10, nsim=1,
                        frequency=1, arma=NULL, constant=FALSE,
                        initial=NULL, bounds=c("admissible","none"),
                        randomizer=c("rnorm","rt","rlaplace","rs"),
                        probability=1, ...){
    # Function generates data using SSARIMA in Single Source of Error as a data generating process.
    #    Copyright (C) 2015 - Inf Ivan Svetunkov

    randomizer <- randomizer[1];
    ellipsis <- list(...);
    bounds <- bounds[1];
    # If R decided that by "b" we meant "bounds", fix this!
    if(is.numeric(bounds)){
        ellipsis$b <- bounds;
        bounds <- "u";
    }

    if(all(bounds!=c("n","a","none","admissible"))){
        warning(paste0("Strange type of bounds provided: ",bounds,". Switching to 'admissible'."),
                call.=FALSE);
        bounds <- "a";
    }

    bounds <- substring(bounds[1],1,1);

    if(!is.null(orders)){
        if(is.list(orders)){
            arOrders <- orders$ar;
            iOrders <- orders$i;
            maOrders <- orders$ma;
        }
        else{
            arOrders <- orders[1];
            iOrders <- orders[2];
            maOrders <- orders[3];
        }
    }
    else{
        arOrders <- 0;
        iOrders <- 0;
        maOrders <- 0;
    }

    creator <- function(arimaPolynomials, matF, vecG){
        if(arRequired || any(iOrders>0)){
            # Fill in the transition matrix
            matF[1:length(arimaPolynomials$ariPolynomial[-1]),1] <- -arimaPolynomials$ariPolynomial[-1];
            # Fill in the persistence vector
            vecG[1:length(arimaPolynomials$ariPolynomial[-1]),1] <- -arimaPolynomials$ariPolynomial[-1];
            if(maRequired){
                vecG[1:length(arimaPolynomials$maPolynomial[-1]),1] <- vecG[1:length(arimaPolynomials$maPolynomial[-1]),1] +
                    arimaPolynomials$maPolynomial[-1];
            }
        }
        else{
            if(maRequired){
                vecG[1:length(arimaPolynomials$maPolynomial[-1]),1] <- arimaPolynomials$maPolynomial[-1];
            }
        }

        return(list(matF=matF, vecG=vecG));
    }

    #### Elements Generator for AR and MA ####
    elementsGenerator <- function(arOrders=arOrders, maOrders=maOrders, iOrders=iOrders,
                                  arValue=arValue, maValue=maValue,
                                  arGenerate=FALSE, maGenerate=FALSE,
                                  componentsNumber, adamCpp){
        if(arGenerate){
            arRoots <- 0.5;
            while(any(arRoots<1)){
                arValue <- runif(arNumber,-1,1);

                # This is needed only to get the correct AR polynomials,
                # which is why we set the orders of I and MA to zero
                arimaPolynomials <- lapply(adamCpp$polynomialise(0,
                                                                 arOrders, rep(0, lagsLength), rep(0, lagsLength),
                                                                 FALSE, FALSE, arValue, lags), as.vector);

                if(bounds=="a" && (componentsNumber > 0)){
                    arRoots <- abs(polyroot(arimaPolynomials$arPolynomial));
                }
                else{
                    arRoots <- 1;
                }
            }
        }

        if(maGenerate){
            maRoots <- 0.5;
            while(any(maRoots<1)){
                maValue <- runif(maNumber,-1,1);

                # This is needed only to get the correct MA polynomials,
                # which is why we set the orders of I and AR to zero
                arimaPolynomials <- lapply(adamCpp$polynomialise(0,
                                                                 rep(0, lagsLength), rep(0, lagsLength), maOrders,
                                                                 FALSE, FALSE, maValue, lags), as.vector);

                if(bounds=="a" && (componentsNumber > 0)){
                    maRoots <- abs(polyroot(arimaPolynomials$maPolynomial));
                }
                else{
                    maRoots <- 1;
                }
            }
        }

        return(list(arValue=arValue,maValue=maValue));
    }

    ##### Orders and lags for ssarima #####
    if(any(is.complex(c(arOrders,iOrders,maOrders,lags)))){
        stop("Come on! Be serious! This is ARIMA, not CES!",call.=FALSE);
    }

    if(any(c(arOrders,iOrders,maOrders)<0)){
        stop("Funny guy! How am I gonna construct a model with negative order?",call.=FALSE);
    }

    if(any(c(lags)<0)){
        stop("Right! Why don't you try complex lags then, mister smart guy?",call.=FALSE);
    }

    if(length(lags)!=length(arOrders) & length(lags)!=length(iOrders) & length(lags)!=length(maOrders)){
        stop("Seasonal lags do not correspond to any element of SARIMA",call.=FALSE);
    }

    # If there are zero lags, drop them
    if(any(lags==0)){
        arOrders <- arOrders[lags!=0];
        iOrders <- iOrders[lags!=0];
        maOrders <- maOrders[lags!=0];
        lags <- lags[lags!=0];
    }

    lagsLength <- length(lags);

    # Define maxorder and make all the values look similar (for the polynomials)
    maxorder <- max(length(arOrders),length(iOrders),length(maOrders));
    if(length(arOrders)!=maxorder){
        arOrders <- c(arOrders,rep(0,maxorder-length(arOrders)));
    }
    if(length(iOrders)!=maxorder){
        iOrders <- c(iOrders,rep(0,maxorder-length(iOrders)));
    }
    if(length(maOrders)!=maxorder){
        maOrders <- c(maOrders,rep(0,maxorder-length(maOrders)));
    }

    # If zeroes are defined for some orders, drop them.
    if(any((arOrders + iOrders + maOrders)==0)){
        orders2leave <- (arOrders + iOrders + maOrders)!=0;
        if(all(orders2leave==FALSE)){
            orders2leave <- lags==min(lags);
        }
        arOrders <- arOrders[orders2leave];
        iOrders <- iOrders[orders2leave];
        maOrders <- maOrders[orders2leave];
        lags <- lags[orders2leave];
    }

    # Get rid of duplicates in lags
    if(length(unique(lags))!=length(lags)){
        if(frequency!=1){
            warning(paste0("'lags' variable contains duplicates: (",paste0(lags,collapse=","),
                           "). Getting rid of some of them."),call.=FALSE);
        }
        lags.new <- unique(lags);
        arOrders.new <- iOrders.new <- maOrders.new <- lags.new;
        for(i in 1:length(lags.new)){
            arOrders.new[i] <- max(arOrders[which(lags==lags.new[i])]);
            iOrders.new[i] <- max(iOrders[which(lags==lags.new[i])]);
            maOrders.new[i] <- max(maOrders[which(lags==lags.new[i])]);
        }
        arOrders <- arOrders.new;
        iOrders <- iOrders.new;
        maOrders <- maOrders.new;
        lags <- lags.new;
    }

    arValue <- arma$ar;
    # Check the provided AR matrix / vector
    if(!is.null(arValue)){
        if((!is.numeric(arValue) | !is.vector(arValue)) & !is.matrix(arValue)){
            warning(paste0("AR should be either vector or matrix. You have provided something strange...\n",
                           "AR will be generated."),call.=FALSE);
            arRequired <- arGenerate <- TRUE;
            arValue <- NULL;
        }
        else{
            if(sum(arOrders)!=length(arValue[arValue!=0])){
                warning(paste0("Wrong number of non-zero elements of AR. Should be ",sum(arOrders),
                               " instead of ",length(arValue[arValue!=0]),".\n",
                               "AR will be generated."),call.=FALSE);
                arRequired <- arGenerate <- TRUE;
                arValue <- NULL;
            }
            else{
                if(all(arOrders==0)){
                    arValue <- NULL;
                    arRequired <- arGenerate <- FALSE;
                }
                else{
                    arValue <- as.vector(arValue[arValue!=0]);
                    arGenerate <- FALSE;
                    arRequired <- TRUE;
                }
            }
        }
    }
    else{
        if(all(arOrders==0)){
            arRequired <- arGenerate <- FALSE;
        }
        else{
            arRequired <- arGenerate <- TRUE;
        }
    }
    arNumber <- sum(arOrders);

    maValue <- arma$ma;
    # Check the provided MA matrix / vector
    if(!is.null(maValue)){
        if((!is.numeric(maValue) | !is.vector(maValue)) & !is.matrix(maValue)){
            warning(paste0("MA should be either vector or matrix. You have provided something strange...\n",
                           "MA will be generated."),call.=FALSE);
            maRequired <- maGenerate <- TRUE;
            maValue <- NULL;
        }
        else{
            if(sum(maOrders)!=length(maValue[maValue!=0])){
                warning(paste0("Wrong number of non-zero elements of MA. Should be ",sum(maOrders),
                               " instead of ",length(maValue[maValue!=0]),".\n",
                               "MA will be generated."),call.=FALSE);
                maRequired <- maGenerate <- TRUE;
                maValue <- NULL;
            }
            else{
                if(all(maOrders==0)){
                    maValue <- NULL;
                    maRequired <- maGenerate <- FALSE;
                }
                else{
                    maValue <- as.vector(maValue[maValue!=0]);
                    maGenerate <- FALSE;
                    maRequired <- TRUE;
                }
            }
        }
    }
    else{
        if(all(maOrders==0)){
            maRequired <- maGenerate <- FALSE;
        }
        else{
            maRequired <- maGenerate <- TRUE;
        }
    }
    maNumber <- sum(maOrders);

    #### Constant ####
    # Check the provided constant
    if(is.numeric(constant)){
        constantGenerate <- FALSE;
        constantRequired <- TRUE;
        constantValue <- constant;
    }
    else if(is.logical(constant)){
        constantRequired <- constantGenerate <- constant;
        constantValue <- NULL;
    }

    #### Number of components and observations ####
    # Number of components to use
    componentsNumber <- max(arOrders %*% lags + iOrders %*% lags,maOrders %*% lags);
    componentsNames <- paste0("Component ",1:(componentsNumber+constantRequired));
    lagsModel <- matrix(rep(1,times=componentsNumber),ncol=1);
    if(constantRequired){
        lagsModel <- rbind(lagsModel,1);
    }
    lagsModelMax <- 1;

    #### Variables for the adamCore ####
    xregNumber <- 0;
    adamETS <- FALSE;
    # Create all the necessary matrices and vectors
    componentsNumberARIMA <- componentsNumber;
    componentsNumberETS <- componentsNumberETSNonSeasonal <- componentsNumberETSSeasonal <- 0;

    Etype <- "A";
    Stype <- Ttype <- "N";

    # Create C++ adam class, which will then use fit, forecast etc methods
    adamCpp <- new(adamCore,
                   lagsModel, Etype, Ttype, Stype,
                   componentsNumberETSNonSeasonal,
                   componentsNumberETSSeasonal,
                   componentsNumberETS, componentsNumberARIMA,
                   xregNumber, length(lagsModel),
                   constantRequired, adamETS);

    #### Initials ####
    initialValue <- initial;
    initialGenerate <- FALSE;
    if(!is.null(initialValue)){
        if(!is.numeric(initialValue)){
            warning(paste0("Initial vector is not numeric!\n",
                           "Initial values will be generated."),call.=FALSE);
            initialValue <- NULL;
            initialGenerate <- TRUE;
        }
        else{
            if(length(initialValue) != componentsNumber){
                warning(paste0("Wrong length of initial vector. Should be ",componentsNumber,
                               " instead of ",length(initialValue),".\n",
                               "Initial values will be generated."),call.=FALSE);
                initialValue <- NULL;
                initialGenerate <- TRUE;
            }
        }
    }
    else{
        initialGenerate <- TRUE;
    }

    # Check the vector of probabilities
    if(is.vector(probability)){
        if(any(probability!=probability[1])){
            if(length(probability)!=obs){
                warning("Length of probability does not correspond to number of observations.",call.=FALSE);
                if(length(probability)>obs){
                    warning("We will cut off the excessive ones.",call.=FALSE);
                    probability <- probability[1:obs];
                }
                else{
                    warning("We will duplicate the last one.",call.=FALSE);
                    probability <- c(probability,rep(probability[length(probability)],obs-length(probability)));
                }
            }
        }
    }

    # In the case of wrong nsim, make it natural number. The same is for obs and frequency.
    nsim <- abs(round(nsim,0));
    obs <- abs(round(obs,0));
    obsStates <- obs + 1;
    frequency <- abs(round(frequency,0));

    if(initialGenerate){
        burnInPeriod <- max(lags);
        obs <- obs + burnInPeriod;
        obsStates <- obsStates + burnInPeriod;
    }

    if((componentsNumber==0) & !constantRequired){
        warning("You have not defined any model. So here's series generated from your distribution.", call.=FALSE);
        matYt <- matErrors <- matrix(NA,obs,nsim);
        ellipsis$n <- nsim*obs;
        matErrors[,] <- do.call(randomizer,ellipsis);

        matOt <- matrix(NA,obs,nsim);
        # Generate values for occurence variable
        if(all(probability == 1)){
            matOt[,] <- 1;
        }
        else{
            matOt[,] <- rbinom(obs*nsim,1,probability);
        }

        matOt <- ts(matOt,frequency=frequency);
        matErrors <- ts(matErrors,frequency=frequency);
        matYt <- matErrors;

        veclikelihood <- -obs/2 *(log(2*pi*exp(1)) + log(colMeans(matErrors^2)));
        modelname <- "ARIMA(0,0,0)";
        model <- list(model=modelname,
                      constant=NA, initial=NULL,
                      data=matYt, states=NULL, residuals=matErrors,
                      occurrence=matOt, likelihood=veclikelihood);
        return(structure(model,class="smooth.sim"));
    }

    ##### Preset values of matVt and other matrices and arrays ######
    if(componentsNumber > 0){
        # Transition matrix, measurement vector and persistence vector + state vector
        matF <- rbind(cbind(rep(0,componentsNumber-1),diag(componentsNumber-1)),rep(0,componentsNumber));
        matWt <- matrix(c(1,rep(0,componentsNumber-1)),1,componentsNumber);
        if(constantRequired){
            matF <- cbind(rbind(matF,rep(0,componentsNumber)),c(1,rep(0,componentsNumber-1),1));
            matWt <- cbind(matWt,0);
        }
    }
    else{
        matWt <- matF <- matrix(1,1,1);
    }

    persistenceLength <- componentsNumber + constantRequired;
    matWt <- matrix(matWt, obs, persistenceLength, byrow=TRUE);

    # Matrix with some initials. Used as an interim stuff for the arrVt
    matVt <- matrix(1,persistenceLength,obsStates);

    # Define arrays
    arrVt <- array(NA,c(persistenceLength,obsStates,nsim),
                   dimnames=list(componentsNames,NULL,NULL));
    arrF <- array(0,c(dim(matF),nsim));
    matG <- matrix(0,persistenceLength,nsim);

    matErrors <- matrix(NA,obs,nsim);
    matYt <- matrix(NA,obs,nsim);
    matOt <- matrix(NA,obs,nsim);
    matarValue <- matrix(NA,max(1,arNumber),nsim);
    matmaValue <- matrix(NA,max(1,maNumber),nsim);
    vecConstantValue <- rep(NA,nsim);
    matInitialValue <- matrix(NA,componentsNumber,nsim);

    #### Generate stuff if needed ####
    if(componentsNumber>0){
        if(initialGenerate){
            matInitialValue[1:componentsNumber,] <- runif(componentsNumber*nsim,0,1000);
            arrVt[1:componentsNumber,1,] <- matInitialValue[1:componentsNumber,];
        }
        else{
            matInitialValue[1:componentsNumber,] <- rep(initialValue,nsim);
            arrVt[1:componentsNumber,1,] <- matInitialValue[1:componentsNumber,];
        }
    }

    if(arRequired){
        if(arGenerate){
            for(i in 1:nsim){
                matarValue[,i] <- elementsGenerator(arOrders=arOrders, maOrders=maOrders, iOrders=iOrders,
                                              arValue=NULL, maValue=NULL,
                                              arGenerate=TRUE, maGenerate=FALSE, componentsNumber, adamCpp)$arValue;
            }
        }
        else{
            matarValue[] <- arValue;
        }
    }

    if(maRequired){
        if(maGenerate){
            for(i in 1:nsim){
                matmaValue[,i] <- elementsGenerator(arOrders=arOrders, maOrders=maOrders, iOrders=iOrders,
                                              arValue=NULL, maValue=NULL,
                                              arGenerate=FALSE, maGenerate=TRUE, componentsNumber, adamCpp)$maValue;
            }
        }
        else{
            matmaValue[] <- maValue;
        }
    }

    armaParameters <- matrix(0, arNumber+maNumber, nsim);
    for(l in 1:nsim){
        j <- arIndex <- maIndex <- 0;
        for(i in 1:length(lags)){
            if(arRequired && arOrders[i]>0){
                armaParameters[j+c(1:arOrders[i]),l] <- matarValue[arIndex+c(1:arOrders[i]),l];
                j[] <- j+arOrders[i];
                arIndex[] <- arIndex+arOrders[i];
            }
            if(maRequired && maOrders[i]>0){
                armaParameters[j+c(1:maOrders[i]),l] <- matmaValue[maIndex+c(1:maOrders[i]),l];
                j[] <- j+maOrders[i];
                maIndex[] <- maIndex+maOrders[i];
            }
        }
    }

    if(constantRequired){
        if(constantGenerate){
            if(any(iOrders>0)){
                vecConstantValue <- runif(nsim,-200,200);
            }
            else{
                vecConstantValue <- runif(nsim,100,1000);
            }
        }
        else{
            vecConstantValue[] <- constantValue;
        }
    }
    else{
        vecConstantValue[] <- 0;
    }

    for(i in 1:nsim){
        arimaPolynomials <- lapply(adamCpp$polynomialise(0,
                                                         arOrders, iOrders, maOrders,
                                                         FALSE, FALSE, armaParameters[,i], lags), as.vector);
        elements <- creator(arimaPolynomials, matF, matG[,i,drop=FALSE]);

        arrF[,,i] <- elements$matF;
        matG[,i] <- elements$vecG;

        # A correction in order to make sense out of generated initial components
        if(initialGenerate){
            arrVt[,,i] <- matVt;
            arrVt[,1,i] <- matrixPowerWrap(as.matrix(arrF[,,i]),componentsNumber+1) %*% arrVt[,1,i];
        }

        if(constantRequired){
            arrVt[persistenceLength,1,i] <- matVt[persistenceLength,1];
        }
    }

    # If the chosen randomizer is not default and no parameters are provided, change to rnorm.
    if(all(randomizer!=c("rnorm","rt","rlaplace","rs")) & (length(ellipsis)==0)){
        warning("The chosen randomizer - ", randomizer,
                " - needs some arbitrary parameters! Changing to 'rnorm' now.",
                call.=FALSE);
        randomizer = "rnorm";
    }

    # Check if no argument was passed in dots
    if(length(ellipsis)==0){
        ellipsis$n <- nsim*obs;
        # Create vector of the errors
        if(any(randomizer==c("rnorm","rlaplace","rs"))){
            matErrors[,] <- do.call(randomizer,ellipsis);
        }
        else if(randomizer=="rt"){
            # The degrees of freedom are df = n - k.
            matErrors[,] <- rt(nsim*obs,obs-(persistenceLength + lagsModelMax));
        }

        # Center errors just in case
        matErrors <- matErrors - colMeans(matErrors);
        # Change variance to make some sense. Errors should not be ridiculously high and not too low.
        matErrors <- matErrors * sqrt(abs(colMeans(as.matrix(arrVt[1,1:lagsModelMax,]))));
        if(randomizer=="rs"){
            matErrors <- matErrors / 4;
        }
    }
    # If arguments are passed, use them. WE ASSUME HERE THAT USER KNOWS WHAT HE'S DOING!
    else{
        ellipsis$n <- nsim*obs;
        matErrors[,] <- do.call(randomizer,ellipsis);
        if(randomizer=="rbeta"){
            # Center the errors around 0
            matErrors <- matErrors - 0.5;
            # Make a meaningful variance of data. Something resembling to var=1.
            matErrors <- matErrors / rep(sqrt(colMeans(matErrors^2)) *
                                             sqrt(abs(colMeans(as.matrix(arrVt[1,1:lagsModelMax,])))),each=obs);
        }
        else if(randomizer=="rt"){
            # Make a meaningful variance of data.
            matErrors <- matErrors * rep(sqrt(abs(colMeans(as.matrix(arrVt[1,1:lagsModelMax,])))),each=obs);
        }
    }

    # Generate ones for the possible intermittency
    if(all(probability == 1)){
        matOt[,] <- 1;
    }
    else{
        matOt[,] <- rbinom(obs*nsim,1,probability);
    }

    profiles <- adamProfileCreator(lagsModel, lagsModelMax, obs);
    indexLookupTable <- profiles$lookup;
    profilesRecentArray <- arrVt[,1:lagsModelMax,, drop=FALSE];

    #### Simulate the data ####
    simulateddata <- adamCpp$simulate(matErrors, matOt,
                                      arrVt, matWt,
                                      arrF,
                                      matG,
                                      indexLookupTable, profilesRecentArray,
                                      Etype, TRUE);

    if(all(probability == 1)){
        matYt <- simulateddata$data;
    }
    else{
        matYt <- round(simulateddata$data,0);
    }
    arrVt[] <- simulateddata$states;
    profilesRecentArray[] <- simulateddata$profile;

    if(any(randomizer==c("rnorm","rt"))){
        veclikelihood <- -obs/2 *(log(2*pi*exp(1)) + log(colMeans(matErrors^2)));
    }
    else if(randomizer=="rlaplace"){
        veclikelihood <- -obs*(log(2*exp(1)) + log(colMeans(abs(matErrors))));
    }
    else if(randomizer=="rs"){
        veclikelihood <- -2*obs*(log(2*exp(1)) + log(0.5*colMeans(sqrt(abs(matErrors)))));
    }
    else if(randomizer=="rlnorm"){
        veclikelihood <- -obs/2 *(log(2*pi*exp(1)) + log(colMeans(matErrors^2))) - colSums(log(matYt));
    }
    # If this is something unknown, forget about it
    else{
        veclikelihood <- NA;
    }

    if(constantRequired){
        dimnames(arrVt)[[1]][persistenceLength] <- "Constant";
    }

    if(initialGenerate){
        if(constantRequired){
            matInitialValue[,] <- arrVt[-persistenceLength,burnInPeriod+1,];
        }
        else{
            matInitialValue[,] <- arrVt[,burnInPeriod+1,];
        }
        arrvtDim <- dim(arrVt);
        arrvtDim[2] <- arrvtDim[2] - burnInPeriod;
        arrVt <- array(arrVt[-c(1:burnInPeriod),,],arrvtDim);
        matErrors <- matErrors[-c(1:burnInPeriod),];
        matYt <- matYt[-c(1:burnInPeriod),];
        matOt <- matOt[-c(1:burnInPeriod),];
    }

    if(nsim==1){
        matYt <- ts(matYt,frequency=frequency);
        matErrors <- ts(matErrors,frequency=frequency);
        arrVt <- ts(arrVt[,,1],frequency=frequency,start=c(0,frequency-lagsModelMax+1));
        matOt <- ts(matOt,frequency=frequency);
    }
    else{
        matYt <- ts(matYt,frequency=frequency);
        matErrors <- ts(matErrors,frequency=frequency);
        matOt <- ts(matOt,frequency=frequency);
    }

    # Give model the name
    if((length(arOrders)==1) && all(lags==1)){
        modelname <- paste0("ARIMA(",arOrders,",",iOrders,",",maOrders,")");
    }
    else{
        modelname <- "";
        for(i in 1:length(arOrders)){
            modelname <- paste0(modelname,"(",arOrders[i],",");
            modelname <- paste0(modelname,iOrders[i],",");
            modelname <- paste0(modelname,maOrders[i],")[",lags[i],"]");
        }
        modelname <- paste0("SARIMA",modelname);
    }
    if(any(probability!=1)){
        modelname <- paste0("i",modelname);
    }

    if(constantRequired){
        if(all(iOrders==0)){
            modelname <- paste0(modelname," with constant");
        }
        else{
            modelname <- paste0(modelname," with drift");
        }
        names(vecConstantValue) <- rep("Constant",length(vecConstantValue));
    }
    else{
        const <- FALSE;
        constantValue <- NULL;
    }

    model <- list(model=modelname, arma=list(ar=matarValue, ma=matmaValue),
                  constant=vecConstantValue, initial=matInitialValue,
                  profile=profilesRecentArray,
                  data=matYt, states=arrVt, residuals=matErrors,
                  occurrence=matOt, logLik=veclikelihood);
    return(structure(model,class="smooth.sim"));
}
