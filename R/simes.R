#' Simulate Exponential Smoothing
#'
#' Function generates data using ETS with Single Source of Error as a data
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
#'
#' @param model Type of ETS model according to [Hyndman et. al., 2008]
#' taxonomy. Can consist of 3 or 4 chars: \code{ANN}, \code{AAN}, \code{AAdN},
#' \code{AAA}, \code{AAdA}, \code{MAdM} etc.
#' @param persistence Persistence vector, which includes all the smoothing
#' parameters. Must correspond to the chosen model. The maximum length is 3:
#' level, trend and seasonal smoothing parameters. If \code{NULL}, values are
#' generated.
#' @param phi Value of damping parameter. If trend is not chosen in the model,
#' the parameter is ignored.
#' @param initial Vector of initial states of level and trend. The maximum
#' length is 2. If \code{NULL}, values are generated.
#' @param initialSeason Vector of initial states for seasonal coefficients.
#' Should have length equal to \code{frequency} parameter. If \code{NULL},
#' values are generated.
#' @param bounds Type of bounds to use for persistence vector if values are
#' generated. \code{"usual"} - bounds from p.156 by Hyndman et. al., 2008.
#' \code{"restricted"} - similar to \code{"usual"} but with upper bound equal
#' to 0.3. \code{"admissible"} - bounds from tables 10.1 and 10.2 of Hyndman
#' et. al., 2008. Using first letter of the type of bounds also works. These
#' bounds are also used for multiplicative models, but the models are much
#' more restrictive, so weird results might be obtained. Be careful!
#' @param ...  Additional parameters passed to the chosen randomizer. All the
#' parameters should be passed in the order they are used in chosen randomizer.
#' For example, passing just \code{sd=0.5} to \code{rnorm} function will lead
#' to the call \code{rnorm(obs, mean=0.5, sd=1)}.  ATTENTION! When generating
#' the multiplicative errors some tuning might be needed to obtain meaningful
#' data. \code{sd=0.1} is usually already a high value for such models. ALSO
#' NOTE: In case of multiplicative error model, the randomizer will generate
#' \code{1+e_t} error, not \code{e_t}. This means that the mean should
#' typically be equal to 1, not zero.
#'
#' @return List of the following values is returned:
#' \itemize{
#' \item \code{model} - Name of ETS model.
#' \item \code{data} - Time series vector (or matrix if \code{nsim>1}) of the generated
#' series.
#' \item \code{states} - Matrix (or array if \code{nsim>1}) of states. States are in
#' columns, time is in rows.
#' \item \code{persistence} - Vector (or matrix if \code{nsim>1}) of smoothing
#' parameters used in the simulation.
#' \item \code{phi} - Value of damping parameter used in time series generation.
#' \item \code{initial} - Vector (or matrix) of initial values.
#' \item \code{initialSeason} - Vector (or matrix) of initial seasonal coefficients.
#' \item \code{profile} - The final profile produced in the simulation.
#' \item \code{probability} - vector of probabilities used in the simulation.
#' \item \code{intermittent} - type of the intermittent model used.
#' \item \code{residuals} - Error terms used in the simulation. Either vector or matrix,
#' depending on \code{nsim}.
#' \item \code{occurrence} - Values of occurrence variable. Once again, can be either
#' a vector or a matrix...
#' \item \code{logLik} - Log-likelihood of the constructed model.
#' }
#'
#' @seealso \code{\link[smooth]{es}, \link[stats]{ts}, \link[stats]{Distributions}}
#'
#' @examples
#'
#' # Create 40 observations of quarterly data using AAA model with errors from normal distribution
#' ETSAAA <- sim.es(model="AAA",frequency=4,obs=40,randomizer="rnorm",mean=0,sd=100)
#'
#' # Create 50 series of quarterly data using AAA model
#' # with 40 observations each with errors from normal distribution
#' ETSAAA <- sim.es(model="AAA",frequency=4,obs=40,randomizer="rnorm",mean=0,sd=100,nsim=50)
#'
#' # Create 50 series of quarterly data using AAdA model
#' # with 40 observations each with errors from normal distribution
#' # and smoothing parameters lying in the "admissible" range.
#' ETSAAA <- sim.es(model="AAA",phi=0.9,frequency=4,obs=40,bounds="admissible",
#'                   randomizer="rnorm",mean=0,sd=100,nsim=50)
#'
#' # Create 60 observations of monthly data using ANN model
#' # with errors from beta distribution
#' ETSANN <- sim.es(model="ANN",persistence=c(1.5),frequency=12,obs=60,
#'                   randomizer="rbeta",shape1=1.5,shape2=1.5)
#' plot(ETSANN$states)
#'
#' # Create 60 observations of monthly data using MAM model
#' # with errors from uniform distribution
#' ETSMAM <- sim.es(model="MAdM",persistence=c(0.3,0.2,0.1),initial=c(2000,50),
#'            phi=0.8,frequency=12,obs=60,randomizer="runif",min=-0.5,max=0.5)
#'
#' # Create 80 observations of quarterly data using MMM model
#' # with predefined initial values and errors from the normal distribution
#' ETSMMM <- sim.es(model="MMM",persistence=c(0.1,0.1,0.1),initial=c(2000,1),
#'            initialSeason=c(1.1,1.05,0.9,.95),frequency=4,obs=80,mean=0,sd=0.01)
#'
#' # Generate intermittent data using AAdN
#' iETSAAdN <- sim.es("AAdN",obs=30,frequency=1,probability=0.1,initial=c(3,0),phi=0.8)
#'
#' # Generate iETS(MNN) with TSB style probabilities
#' oETSMNN <- sim.oes("MNN",obs=50,occurrence="d",persistence=0.2,initial=1,
#'                    randomizer="rlnorm",meanlog=0,sdlog=0.3)
#' iETSMNN <- sim.es("MNN",obs=50,frequency=12,persistence=0.2,initial=4,
#'                   probability=oETSMNN$probability)
#'
#' @importFrom stats optim
#' @importFrom greybox rlaplace rs
#' @export sim.es
sim.es <- function(model="ANN", obs=10, nsim=1,
                   frequency=1, persistence=NULL, phi=1,
                   initial=NULL, initialSeason=NULL,
                   bounds=c("usual","admissible","restricted"),
                   randomizer=c("rnorm","rlnorm","rt","rlaplace","rs"),
                   probability=1, ...){
    # Function generates data using ETS with Single Source of Error as a data generating process.
    #    Copyright (C) 2015 - Inf Ivan Svetunkov

    randomizer <- randomizer[1];
    ellipsis <- list(...);
    bounds <- match.arg(bounds);
    # If R decided that by "b" we meant "bounds", fix this!
    if(is.numeric(bounds)){
        ellipsis$b <- bounds;
        bounds <- "usual";
    }

    # If chosen model is "AAdN" or anything like that, we are taking the appropriate values
    if(nchar(model)==4){
        Etype <- substring(model,1,1);
        Ttype <- substring(model,2,2);
        Stype <- substring(model,4,4);
        if(substring(model,3,3)!="d"){
            warning(paste0("You have defined a strange model: ",model),call.=FALSE);
            # if(!silent){
            #     sowhat(model);
            # }
            model <- paste0(Etype,Ttype,"d",Stype);
        }
        if(Ttype!="N" & phi==1){
            model <- paste0(Etype,Ttype,Stype);
            warning(paste0("Damping parameter is set to 1. Changing model to: ",model),call.=FALSE);
        }
    }
    else if(nchar(model)==3){
        Etype <- substring(model,1,1);
        Ttype <- substring(model,2,2);
        Stype <- substring(model,3,3);
        if(phi!=1 & Ttype!="N"){
            model <- paste0(Etype,Ttype,"d",Stype);
            warning(paste0("Damping parameter is set to ",phi,". Changing model to: ",model),call.=FALSE);
        }
    }
    else{
        stop(paste0("You have defined a strange model: ",model,". Cannot proceed"),call.=FALSE);
    }

    # In the case of wrong nsim, make it natural number. The same is for obs and frequency.
    nsim <- abs(round(nsim,0));
    obs <- abs(round(obs,0));
    frequency <- abs(round(frequency,0));

    if(!is.null(persistence) & length(persistence)>3){
        stop("The length of persistence vector is wrong! It should not be greater than 3.",call.=FALSE);
    }

    if(phi<0 | phi>2){
        warning(paste0("Damping parameter should lie in (0, 2) region! You have chosen phi=",phi,
                       ". Be careful!"),call.=FALSE);
    }

    # Check the used model and estimate the length of needed persistence vector.
    if(Etype!="A" & Etype!="M"){
        stop("Wrong error type! Should be 'A' or 'M'.",call.=FALSE);
    }
    else{
        # The number of the smoothing parameters needed
        persistenceLength <- 1;
        # The number initial values of the state vector
        componentsNumber <- 1;
        # The lag of components (needed for the seasonal models)
        lagsModel <- 1;
        # The names of the state vector components
        componentsNames <- "level";
        matWt <- 1;
        # The transition matrix
        matF <- matrix(1,1,1);
    }

    # Check the trend type of the model
    if(Ttype!="N" & Ttype!="A" & Ttype!="M"){
        stop("Wrong trend type! Should be 'N', 'A' or 'M'.",call.=FALSE);
    }
    else if(Ttype!="N"){
        if(is.na(phi) | is.null(phi)){
            phi <- 1;
        }
        persistenceLength <- persistenceLength + 1;
        componentsNumber <- componentsNumber + 1;
        lagsModel <- c(lagsModel,1);
        componentsNames <- c(componentsNames,"trend");
        matWt <- c(matWt,phi);
        matF <- matrix(c(1,0,phi,phi),2,2);
        componentTrend=TRUE;
        if(phi!=1){
            model <- paste0(Etype,Ttype,"d",Stype);
        }
    }
    else{
        componentTrend=FALSE;
    }

    # Check the seasonaity type of the model
    if(Stype!="N" & Stype!="A" & Stype!="M"){
        stop("Wrong seasonality type! Should be 'N', 'A' or 'M'.",call.=FALSE);
    }

    if(Stype!="N" & frequency==1){
        stop("Cannot create the seasonal model with the data frequency 1!",call.=FALSE);
    }

    if(Stype!="N"){
        persistenceLength <- persistenceLength + 1;
        # lagsModelMax is used in the cases of seasonal models.
        #   if lagsModelMax==1 then non-seasonal data will be produced with the defined frequency.
        lagsModel <- c(lagsModel,frequency);
        componentsNames <- c(componentsNames,"seasonality");
        matWt <- c(matWt,1);
        componentSeasonal <- TRUE;

        if(!componentTrend){
            matF <- matrix(c(1,0,0,1),2,2);
        }
        else{
            matF <- matrix(c(1,0,0,phi,phi,0,0,0,1),3,3);
        }
    }
    else{
        componentSeasonal <- FALSE;
    }

    # Make matrices
    lagsModel <- matrix(lagsModel,persistenceLength,1);
    lagsModelMax <- max(lagsModel);
    matWt <- matrix(matWt,obs,persistenceLength, byrow=TRUE);
    arrF <- array(matF,c(dim(matF),nsim));

    # Check the persistence vector length
    if(!is.null(persistence)){
        if(persistenceLength != length(persistence)){
            if(length(persistence)!=1){
                warning(paste0("The length of persistence vector does not correspond to the chosen model!\n",
                               "Falling back to random number generator."),call.=FALSE);
                persistence <- NULL;
            }
            else{
                persistence <- rep(persistence,persistenceLength);
            }
        }
    }

    # Check the inital vector length
    if(!is.null(initial)){
        if(length(initial)>2){
            stop("The length of the initial value is wrong! It should not be greater than 2.",call.=FALSE);
        }
        if(componentsNumber!=length(initial)){
            warning(paste0("The length of initial state vector does not correspond to the chosen model!\n",
                           "Falling back to random number generator."),call.=FALSE);
            initial <- NULL;
        }
        else{
            if(Ttype=="M" & initial[2]<=0){
                warning(paste0("Wrong initial value for multiplicative trend! It should be greater than zero!\n",
                               "Falling back to random number generator."),call.=FALSE);
                initial <- NULL;
            }
        }
    }

    # Check the inital seasonal vector length
    if(!is.null(initialSeason)){
        if(lagsModelMax!=length(initialSeason)){
            warning(paste0("The length of seasonal initial states does not correspond to the chosen frequency!\n",
                           "Falling back to random number generator."),call.=FALSE);
            initialSeason <- NULL;
        }
    }

    # If the chosen randomizer is not default and no parameters are provided, change to rnorm.
    if(all(randomizer!=c("rnorm","rt","rlaplace","rs","rlnorm")) & (length(ellipsis)==0)){
        warning(paste0("The chosen randomizer - ",randomizer," - needs some arbitrary parameters! Changing to 'rnorm' now."),call.=FALSE);
        randomizer = "rnorm";
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
        else{
            probability <- probability[1];
        }
    }

    # Check the probabilities and try to assign the type of intermittent model
    if(length(probability)==1){
        intermittent <- "fixed";
    }
    else{
        # This is a strong assumption!
        intermittent <- "tsb";
    }

    if(all(probability==1)){
        intermittent <- "none";
    }

    ##### Let's make sum fun #####
    matG <- matrix(NA,persistenceLength,nsim);
    arrVt <- array(NA,c(persistenceLength,obs+lagsModelMax,nsim),dimnames=list(componentsNames,NULL,NULL));
    matErrors <- matrix(NA,obs,nsim);
    matYt <- matrix(NA,obs,nsim);
    matOt <- matrix(NA,obs,nsim);

    # If the persistence is NULL or was of the wrong length, generate the values
    if(is.null(persistence)){
        ### For the case of "usual" bounds make restrictions on the generated smoothing parameters so the ETS can be "averaging" model.

        ### First generate the first smoothing parameter.
        if(bounds=="usual"){
            matG[1,] <- runif(nsim,0,1);
        }
        ### These restrictions are even touhger
        else if(bounds=="restricted"){
            matG[1,] <- runif(nsim,0,0.3);
        }

        ### Fill in the other smoothing parameters
        if(bounds!="admissible"){
            if(Ttype!="N"){
                matG[2,] <- runif(nsim,0,matG[1,]);
            }
            if(Stype!="N"){
                matG[persistenceLength,] <- runif(nsim,0,max(0,1-matG[1]));
            }
        }
        ### In case of admissible bounds, do some stuff
        else{
            matG[,] <- runif(persistenceLength*nsim,1-1/phi,1+1/phi);
            if(Ttype!="N"){
                matG[2,] <- runif(nsim,matG[1,]*(phi-1),(2-matG[1,])*(1+phi));
                if(Stype!="N"){
                    Theta.func <- function(Theta){
                        result <- (phi*matG[1,i]+phi+1)/(matG[3,i]) +
                            ((phi-1)*(1+cos(Theta)-cos(lagsModelMax*Theta)) +
                                 cos((lagsModelMax-1)*Theta)-phi*cos((lagsModelMax+1)*Theta))/(2*(1+cos(Theta))*(1-cos(lagsModelMax*Theta)));
                        return(abs(result));
                    }

                    for(i in 1:nsim){
                        matG[3,i] <- runif(1,max(1-1/phi-matG[1,i],0),1+1/phi-matG[1,i]);

                        B <- phi*(4-3*matG[3,i])+matG[3,i]*(1-phi)/lagsModelMax;
                        C <- sqrt(B^2-8*(phi^2*(1-matG[3,i])^2+2*(phi-1)*(1-matG[3,i])-1)+8*matG[3,i]^2*(1-phi)/lagsModelMax);
                        matG[1,i] <- runif(1,1-1/phi-matG[3,i]*(1-lagsModelMax+phi*(1+lagsModelMax))/(2*phi*lagsModelMax),(B+C)/(4*phi));
                        # Solve the equation to get Theta value. Theta

                        Theta <- 0.1;
                        Theta <- optim(Theta,Theta.func,method="Brent",lower=0,upper=1)$par;

                        D <- (phi*(1-matG[1,i])+1)*(1-cos(Theta)) - matG[3,i]*((1+phi)*(1-cos(Theta) - cos(lagsModelMax*Theta)) +
                                                                                   cos((lagsModelMax-1)*Theta)+phi*cos((lagsModelMax+1)*Theta))/
                            (2*(1+cos(Theta))*(1-cos(lagsModelMax*Theta)));
                        matG[2,i] <- runif(1,-(1-phi)*(matG[3,i]/lagsModelMax+matG[1,i]),D+(phi-1)*matG[1,i]);
                    }
                }
            }
            else{
                if(Stype!="N"){
                    matG[1,] <- runif(nsim,-2/(lagsModelMax-1),2);
                    for(i in 1:nsim){
                        matG[2,i] <- runif(1,max(-lagsModelMax*matG[1,i],0),2-matG[1,i]);
                    }
                    matG[1,] <- runif(nsim,-2/(lagsModelMax-1),2-matG[2,]);
                }
            }
        }
    }
    else{
        matG[,] <- rep(persistence,nsim);
    }

    # Generate initial states of level and trend if they were not supplied
    if(is.null(initial)){
        if(Ttype=="N"){
            arrVt[1,1:lagsModelMax,] <- rep(runif(nsim,0,1000), each=lagsModelMax);
        }
        else if(Ttype=="A"){
            arrVt[1,1:lagsModelMax,] <- rep(runif(nsim,0,5000), each=lagsModelMax);
            arrVt[2,1:lagsModelMax,] <- rep(runif(nsim,-100,100), each=lagsModelMax);
        }
        else{
            arrVt[1,1:lagsModelMax,] <- rep(runif(nsim,500,5000), each=lagsModelMax);
            arrVt[2,1:lagsModelMax,] <- 1;
        }
    }
    else{
        arrVt[1:componentsNumber,1:lagsModelMax,] <- rep(rep(initial,lagsModelMax),nsim);
    }
    initial <- matrix(arrVt[1:componentsNumber,1,],nrow=nsim);

    # Generate seasonal states if they were not supplied
    if(componentSeasonal & is.null(initialSeason)){
        # Create and normalize seasonal components. Use geometric mean for multiplicative case
        if(Stype == "A"){
            arrVt[componentsNumber+1,1:lagsModelMax,] <- runif(nsim*lagsModelMax,-500,500);
            for(i in 1:nsim){
                arrVt[componentsNumber+1,1:lagsModelMax,i] <- arrVt[componentsNumber+1,1:lagsModelMax,i] -
                    mean(arrVt[componentsNumber+1,1:lagsModelMax,i]);
            }
        }
        else{
            arrVt[componentsNumber+1,1:lagsModelMax,] <- runif(nsim*lagsModelMax,0.3,1.7);
            for(i in 1:nsim){
                arrVt[componentsNumber+1,1:lagsModelMax,i] <- arrVt[componentsNumber+1,1:lagsModelMax,i] /
                    exp(mean(log(arrVt[componentsNumber+1,1:lagsModelMax,i])));
            }
        }
        initialSeason <- matrix(arrVt[componentsNumber+1,1:lagsModelMax,],nrow=nsim);
        # Count seasonal as a component
        componentsNumber[] <- componentsNumber+1;
    }
    # If the seasonal model is chosen, fill in the first "frequency" values of seasonal component.
    else if(componentSeasonal & !is.null(initialSeason)){
        arrVt[componentsNumber+1,1:lagsModelMax,] <- rep(initialSeason,nsim);
        initialSeason <- matrix(arrVt[componentsNumber+1,1:lagsModelMax,],nrow=nsim);
        # Count seasonal as a component
        componentsNumber[] <- componentsNumber+1;
    }

    # Check if any argument was passed in dots
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
        else if(randomizer=="rlnorm"){
            matErrors[,] <- rlnorm(n=nsim*obs,0,0.01+(1-probability));
            matErrors <- matErrors - 1;
        }

        if(randomizer!="rlnorm"){
            # If the error is multiplicative, scale it!
            if(Etype=="M"){
                # Errors will be lognormal, decrease variance, so it behaves better
                if(any(probability!=1)){
                    matErrors <- matErrors * 0.5;
                }
                else{
                    matErrors <- matErrors * 0.1;
                }
                matErrors <- exp(matErrors) - 1;
            }
            else if(Etype=="A"){
                # Change variance to make some sense. Errors should not be rediculously high and not too low.
                if(all(arrVt[1,1,]!=0)){
                    matErrors <- matErrors * sqrt(abs(arrVt[1,1,]));
                }

                if(randomizer=="rs"){
                    matErrors <- matErrors / 4;
                }
            }
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
            matErrors <- matErrors / rep(sqrt(colMeans(matErrors^2)) * sqrt(abs(arrVt[1,1,])),each=obs);
        }
        else if(randomizer=="rt"){
            # Make a meaningful variance of data.
            matErrors <- matErrors * rep(sqrt(abs(arrVt[1,1,])),each=obs);
        }

        # Substitute 1 to get epsilon_t
        if(Etype=="M"){
            matErrors <- matErrors - 1;
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
    componentsNumberARIMA <- 0;
    componentsNumberETS <- componentsNumber;

    componentsNumberETSSeasonal <- c(0,1)[componentSeasonal+1];
    componentsNumberETSNonSeasonal <- componentsNumberETS - componentsNumberETSSeasonal;

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

    matYt <- simulateddata$data;
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
    else if(randomizer=="rinvgauss"){
        veclikelihood <- -0.5*(obs*(log(colMeans(matErrors^2/(1+matErrors))/(2*pi))-1) +
                                   sum(log(matYt/(1+matErrors))) - 3*sum(log(matYt)));
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
    }
    else{
        matYt <- ts(matYt,frequency=frequency);
        matErrors <- ts(matErrors,frequency=frequency);
        matOt <- ts(matOt,frequency=frequency);
    }

    if(Ttype!="N"){
        rownames(matG) <- c("alpha","beta","gamma")[1:persistenceLength];
    }
    else{
        rownames(matG) <- c("alpha","gamma")[1:persistenceLength];
    }

    model <- paste0("ETS(",model,")");
    if(any(probability!=1)){
        model <- paste0("i",model);
    }

    if(any(is.nan(matYt))){
        warning("NaN values were produced by the simulator.",call.=FALSE);
    }

    model <- list(model=model, data=matYt, states=arrVt, persistence=matG, phi=phi,
                  initial=initial, initialSeason=initialSeason,
                  profile=profilesRecentArray,
                  probability=probability, intermittent=intermittent,
                  residuals=matErrors, occurrence=matOt, logLik=veclikelihood, other=ellipsis);
    return(structure(model,class="smooth.sim"));
}
