#' Simulate Generalised Exponential Smoothing
#'
#' Function generates data using GUM with Single Source of Error as a data
#' generating process.
#'
#' For the information about the function, see the vignette:
#' \code{vignette("simulate","smooth")}
#'
#' @template ssSimParam
#' @template ssAuthor
#' @template ssKeywords
#'
#' @template smoothRef
#'
#' @param orders Order of the model. Specified as vector of number of states
#' with different lags. For example, \code{orders=c(1,1)} means that there are
#' two states: one of the first lag type, the second of the second type.
#' @param lags Defines lags for the corresponding orders. If, for example,
#' \code{orders=c(1,1)} and lags are defined as \code{lags=c(1,12)}, then the
#' model will have two states: the first will have lag 1 and the second will
#' have lag 12. The length of \code{lags} must correspond to the length of
#' \code{orders}.
#' @param persistence Persistence vector \eqn{g}, containing smoothing
#' parameters. If \code{NULL}, then randomly generated.
#' @param transition Transition matrix \eqn{F}. Can be provided as a vector.
#' Matrix will be formed using the default \code{matrix(transition,nc,nc)},
#' where \code{nc} is the number of components in state vector. If \code{NULL},
#' then randomly generated.
#' @param measurement Measurement vector \eqn{w}. If \code{NULL}, then
#' randomly generated (between 0 and 1 for stability).
#' @param initial Vector of initial values for state matrix. If \code{NULL},
#' then generated using advanced, sophisticated technique - uniform
#' distribution.
#' @param ...  Additional parameters passed to the chosen randomizer. All the
#' parameters should be passed in the order they are used in chosen randomizer.
#' For example, passing just \code{sd=0.5} to \code{rnorm} function will lead
#' to the call \code{rnorm(obs, mean=0.5, sd=1)}.
#'
#' @return List of the following values is returned:
#' \itemize{
#' \item \code{model} - Name of GUM model.
#' \item \code{measurement} - Matrix w.
#' \item \code{transition} - Matrix F.
#' \item \code{persistence} - Persistence vector. This is the place, where
#' smoothing parameters live.
#' \item \code{initial} - Initial values of GUM in a form of matrix. If \code{nsim>1},
#' then this is an array.
#' \item \code{profile} - The final profile produced in the simulation.
#' \item \code{data} - Time series vector (or matrix if \code{nsim>1}) of the generated
#' series.
#' \item \code{states} - Matrix (or array if \code{nsim>1}) of states. States are in
#' columns, time is in rows.
#' \item \code{residuals} - Error terms used in the simulation. Either vector or matrix,
#' depending on \code{nsim}.
#' \item \code{occurrence} - Values of occurrence variable. Once again, can be either
#' a vector or a matrix...
#' \item \code{logLik} - Log-likelihood of the constructed model.
#' }
#'
#' @seealso \code{\link[smooth]{sim.es}, \link[smooth]{sim.ssarima},
#' \link[smooth]{sim.ces}, \link[smooth]{gum}, \link[stats]{Distributions}}
#'
#' @examples
#'
#' # Create 120 observations from GUM(1[1]). Generate 100 time series of this kind.
#' x <- sim.gum(orders=c(1),lags=c(1),obs=120,nsim=100)
#'
#' # Generate similar thing for seasonal series of GUM(1[1],1[4]])
#' x <- sim.gum(orders=c(1,1),lags=c(1,4),frequency=4,obs=80,nsim=100,transition=c(1,0,0.9,0.9))
#'
#' # Estimate model and then generate 10 time series from it
#' ourModel <- gum(rnorm(100,100,5))
#' simulate(ourModel,nsim=10)
#'
#' @export sim.gum
sim.gum <- function(orders=c(1), lags=c(1),
                    obs=10, nsim=1,
                    frequency=1, measurement=NULL,
                    transition=NULL, persistence=NULL, initial=NULL,
                    randomizer=c("rnorm","rt","rlaplace","rs"),
                    probability=1, ...){

    randomizer <- randomizer[1];

    ellipsis <- list(...);

    # Function generates values of measurement, transition and persistence
    gumGenerator <- function(nsim=nsim){
        GUMNotStable <- TRUE;
        # Generate something safe for the measurement
        if(measurementGenerate){
            matWt[] <- rep(runif(componentsNumber,0,1), each=obs);
        }
        for(i in 1:nsim){
            GUMNotStable[] <- TRUE;
            while(GUMNotStable){
                if(transitionGenerate){
                    arrF[,,i] <- runif(componentsNumber^2,-1,1);
                }
                if(persistenceGenerate){
                    matG[,i] <- runif(componentsNumber,-1,1);
                }

                # Use smoothEigens to calculate eigenvalues correctly
                eigenValues <- abs(smoothEigens(matrix(matG[,i], componentsNumber, 1),
                                                matrix(arrF[,,i],componentsNumber,componentsNumber),
                                                matWt,
                                                lagsModel, FALSE, obs));
                if(all(eigenValues<=1)){
                    GUMNotStable[] <- FALSE;
                }
            }
        }
        return(list(arrF=arrF,matG=matG,matWt=matWt));
    }

    #### Check values and preset parameters ####
    if(any(is.complex(c(orders,lags)))){
        stop("Complex values? Right! Come on! Be real!",call.=FALSE);
    }
    if(any(c(orders)<0)){
        stop("Funny guy! How am I gonna construct a model with negative orders?",call.=FALSE);
    }
    if(any(c(lags)<0)){
        stop("Right! Why don't you try complex lags then, mister smart guy?",call.=FALSE);
    }
    if(length(orders) != length(lags)){
        stop(paste0("The length of 'lags' (",length(lags),
                    ") differes from the length of 'orders' (",length(orders),")."), call.=FALSE);
    }

    # If there are zero lags, drop them
    if(any(lags==0)){
        orders <- orders[lags!=0];
        lags <- lags[lags!=0];
    }
    # If zeroes are defined for some orders, drop them.
    if(any(orders==0)){
        lags <- lags[orders!=0];
        orders <- orders[orders!=0];
    }

    # Get rid of duplicates in lags
    if(length(unique(lags))!=length(lags)){
        lags.new <- unique(lags);
        orders.new <- lags.new;
        for(i in 1:length(lags.new)){
            orders.new[i] <- max(orders[which(lags==lags.new[i])]);
        }
        orders <- orders.new;
        lags <- lags.new;
    }

    lagsModel <- matrix(rep(lags,times=orders),ncol=1);
    lagsModelMax <- max(lagsModel);
    componentsNumber <- sum(orders);
    componentsNames <- paste0("Component",c(1:length(lagsModel)),", lag",lagsModel);

    # In the case of wrong nsim, make it natural number. The same is for obs and frequency.
    nsim <- abs(round(nsim,0));
    obs <- abs(round(obs,0));
    obsStates <- obs + lagsModelMax;
    frequency <- abs(round(frequency,0));

    # Define arrays
    arrVt <- array(NA,c(componentsNumber,obsStates,nsim),dimnames=list(componentsNames,NULL,NULL));
    arrF <- array(0,c(componentsNumber,componentsNumber,nsim));
    matG <- matrix(0,componentsNumber,nsim);
    matWt <- matrix(0,obs,componentsNumber);

    matErrors <- matrix(NA,obs,nsim);
    matYt <- matrix(NA,obs,nsim);
    matOt <- matrix(NA,obs,nsim);
    matInitialValue <- array(NA,c(componentsNumber,lagsModelMax,nsim));

    # Initial values
    initialValue <- initial;
    if(!is.null(initialValue)){
        if(length(initialValue) != (componentsNumber*lagsModelMax)){
            warning(paste0("Wrong length of initial vector. Should be ",(componentsNumber*lagsModelMax),
                           " instead of ",length(initialValue),".\n",
                           "Values of initial vector will be generated"),call.=FALSE);
            initialValue <- NULL;
            initialGenerate <- TRUE;
        }
        else{
            initialGenerate <- FALSE;
        }
    }
    else{
        initialGenerate <- TRUE;
    }

    # Check measurement vector
    measurementValue <- measurement;
    if(!is.null(measurementValue)){
        if(length(measurementValue) != componentsNumber){
            warning(paste0("Wrong length of measurement vector. Should be ",componentsNumber,
                           " instead of ",length(measurementValue),".\n",
                           "Values of measurement vector will be generated"),call.=FALSE);
            measurementValue <- NULL;
            measurementGenerate <- TRUE;
        }
        else{
            measurementGenerate <- FALSE;
        }
    }
    else{
        measurementGenerate <- TRUE;
    }

    # Check transition matrix
    transitionValue <- transition;
    if(!is.null(transitionValue)){
        if(length(transitionValue) != componentsNumber^2){
            warning(paste0("Wrong dimension of transition matrix. Should be ",componentsNumber^2,
                           " instead of ",length(transitionValue),".\n",
                           "Values of transition matrix will be generated"),call.=FALSE);
            transitionValue <- NULL;
            transitionGenerate <- TRUE;
        }
        else{
            transitionGenerate <- FALSE;
        }
    }
    else{
        transitionGenerate <- TRUE;
    }

    # Check persistence vector
    persistenceValue <- persistence;
    if(!is.null(persistenceValue)){
        if(length(persistenceValue) != componentsNumber){
            warning(paste0("Wrong length of persistence vector. Should be ",componentsNumber,
                           " instead of ",length(persistenceValue),".\n",
                           "Values of persistence vector will be generated"),call.=FALSE);
            persistenceValue <- NULL;
            persistenceGenerate <- TRUE;
        }
        else{
            persistenceGenerate <- FALSE;
        }
    }
    else{
        persistenceGenerate <- TRUE;
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

    #### Generate stuff if needed ####
    # First deal with initials
    if(initialGenerate){
        matInitialValue[,,] <- runif(componentsNumber*nsim*lagsModelMax,0,1000);
    }
    else{
        matInitialValue[,1:lagsModelMax,] <- rep(initialValue,each=nsim);
    }
    arrVt[,1:lagsModelMax,] <- matInitialValue;

    # Now do the other parameters
    if(!measurementGenerate){
        matWt[] <- rep(measurementValue, each=obs);
    }
    if(!transitionGenerate){
        arrF[] <- transitionValue;
    }
    if(!persistenceGenerate){
        matG[] <- persistenceValue;
    }
    if(any(measurementGenerate,transitionGenerate,persistenceGenerate)){
        generatedParameters <- gumGenerator(nsim);
        arrF[] <- generatedParameters$arrF;
        matG[] <- generatedParameters$matG;
        matWt[] <- generatedParameters$matWt;
    }

    # If the chosen randomizer is not default and no parameters are provided, change to rnorm.
    if(all(randomizer!=c("rnorm","rt","rlaplace","rs")) & (length(ellipsis)==0)){
        warning(paste0("The chosen randomizer - ",randomizer," - needs some arbitrary parameters! Changing to 'rnorm' now."),call.=FALSE);
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
            matErrors[,] <- rt(nsim*obs,obs-(componentsNumber + lagsModelMax));
        }

        # Center errors just in case
        matErrors <- matErrors - colMeans(matErrors);
        # Change variance to make some sense. Errors should not be rediculously high and not too low.
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

    #### Variables for the adamCore ####
    xregNumber <- 0;
    constantRequired <- FALSE;
    adamETS <- FALSE;
    # Create all the necessary matrices and vectors
    componentsNumberARIMA <- componentsNumber;

    componentsNumberETS <- componentsNumberETSNonSeasonal <- componentsNumberETSSeasonal <- 0;

    Etype <- "A";
    Stype <- Ttype <- "N";

    profiles <- adamProfileCreator(lagsModel, lagsModelMax, obs);
    indexLookupTable <- profiles$lookup;
    profilesRecentArray <- arrVt[,1:lagsModelMax,, drop=FALSE];

    # Create C++ adam class, which will then use fit, forecast etc methods
    adamCpp <- new(adamCore,
                   lagsModel, Etype, Ttype, Stype,
                   componentsNumberETSNonSeasonal,
                   componentsNumberETSSeasonal,
                   componentsNumberETS, componentsNumberARIMA,
                   xregNumber, length(lagsModel),
                   constantRequired, adamETS);

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

    if(nsim==1){
        matYt <- ts(matYt[,1],frequency=frequency);
        matErrors <- ts(matErrors[,1],frequency=frequency);
        arrVt <- ts(arrVt[,,1],frequency=frequency,start=c(0,frequency-lagsModelMax+1));
        matOt <- ts(matOt[,1],frequency=frequency);
        matInitialValue <- matInitialValue[,,1];
    }
    else{
        matYt <- ts(matYt,frequency=frequency);
        matErrors <- ts(matErrors,frequency=frequency);
        matOt <- ts(matOt,frequency=frequency);
    }

    modelname <- "GUM";
    modelname <- paste0(modelname,"(",paste(orders,"[",lags,"]",collapse=",",sep=""),")");
    if(any(probability!=1)){
        modelname <- paste0("i",modelname);
    }

    if(measurementGenerate){
        measurementValue <- matWt;
    }
    if(transitionGenerate){
        transitionValue <- arrF;
    }
    if(persistenceGenerate){
        persistenceValue <- matG;
    }
    model <- list(model=modelname, measurement=measurementValue, transition=transitionValue,
                  persistence=persistenceValue,initial=matInitialValue,
                  profile=profilesRecentArray,
                  data=matYt, states=arrVt, residuals=matErrors,
                  occurrence=matOt, logLik=veclikelihood);
    return(structure(model,class="smooth.sim"));
}
