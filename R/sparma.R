#' Sparse ARMA Model in State Space Form
#'
#' @description
#' Implements a Sparse ARMA model in the State Space form.
#' Unlike standard ARIMA which expands polynomials,
#' this function directly maps AR and MA orders to specific lags.
#'
#' @param data Vector or ts object with the data
#' @param orders List with vectors for ar and ma, specifying, which AR and MA orders to fit
#' e.g. orders=list(ar=c(1,4), ma=0) will fit ARMA
#' \eqn{y_{t} = phi_1 y_{t-1} + phi_4 y_{t-4} + e_t}
#' @param constant Logical, whether to include a constant term (default: FALSE)
#' @param loss Loss function type.
#' @param h Forecast horizon (default: 0)
#' @param holdout Logical, whether to use holdout sample (default: FALSE)
#' @param arma List with ar and ma parameters if they do not need to be estimated
#' @param initial Initialisation method for states
#' @param bounds Parameter bounds
#' @param silent Logical, whether to suppress output (default: TRUE)
#' @param ... Other parameters passed to god knows what.
#'
#' @return Object of class c("adam", "smooth") containing:
#' \itemize{
#'   \item model - Model name
#'   \item timeElapsed - Computation time
#'   \item data - Input data
#'   \item holdout - Holdout sample (if applicable)
#'   \item fitted - Fitted values
#'   \item residuals - Residuals
#'   \item forecast - Point forecasts if h>0
#'   \item states - State matrix
#'   \item persistence - Persistence vector (g)
#'   \item transition - Transition matrix (F)
#'   \item measurement - Measurement matrix (W)
#'   \item B - Vector of estimated parameters
#'   \item orders - Orders specified by the user
#'   \item constant - Constant value (if included)
#'   \item arma - vector of ARMA parameters
#'   \item initial - Initial state values
#'   \item initialType - Type of initialisation
#'   \item nParam - Number of parameters
#'   \item logLik - Log-likelihood value
#'   \item loss - Loss function used in the estimation
#'   \item lossValue - Value of the loss function
#'   \item accuracy - Accuracy measures
#' }
#'
#' @details
#' The model implements: \deqn{y_t = phi * y_{t-p} + theta * epsilon_{t-q} + epsilon_t}
#' with a possibility of defining several lags for AR/MA.
#'
#' @examples
#' \dontrun{
#' # Fit SpARMA(1,1) model
#' model <- sparma(BJSales, orders=c(2,1), h=12, holdout=TRUE)
#'
#' # Provide fixed parameters
#' model <- sparma(rnorm(100), orders=c(1,1), arma=c(0.7,0.5))
#' }
#'
#' @export
sparma <- function(data, orders=list(ar=c(1), ma=c(1)), constant=FALSE,
                   loss=c("likelihood","MSE","MAE","HAM","LASSO","RIDGE","MSEh","TMSE","GTMSE","MSCE","GPL"),
                   h=0, holdout=FALSE, arma=NULL,
                   initial=c("backcasting","optimal","two-stage","complete","gradient"),
                   bounds=c("none","usual","admissible"), silent=TRUE, ...) {

    # Start timer
    startTime <- Sys.time();
    cl <- match.call();

    # ===== ARGUMENT VALIDATION =====
    loss <- match.arg(loss);
    initial <- match.arg(initial);
    bounds <- match.arg(bounds);

    ellipsis <- list(...);

    if(!is.list(orders)){
        orders <- list(ar=orders[1], ma=orders[2]);
    }

    p <- orders$ar;
    q <- orders$ma;

    if(length(p)>0){
        p <- p[p!=0];
        if(length(p)==0){
            p <- 0;
        }
    }
    else{
        p <- 0;
    }
    if(length(q)>0){
        q <- q[q!=0];
        if(length(q)==0){
            q <- 0;
        }
    }
    else{
        q <- 0;
    }
    # Rerecord in case this was amended
    orders$ar <- p;
    orders$ma <- q;

    if(any(p<0) || any(q<0)) {
        stop("Orders must be non-negative");
    }

    if(all(p==0) && all(q==0) && !constant) {
        stop("At least one of p or q must be greater than 0");
    }

    # State space dimension
    K <- max(p, q);
    lags <- 1;

    # Convert orders to list format
    orders_list <- list(ar=max(p), i=0, ma=max(q));

    # Build model string for parametersChecker
    model <- "NNN";
    yName <- deparse(substitute(data));

    # Default parameters for parametersChecker
    outliers <- NULL;
    level <- 0.95;
    persistence <- NULL;
    phi <- NULL;
    distribution <- "dnorm";
    occurrence <- "none";
    ic <- "AICc";
    regressors <- "use";
    formula <- NULL;
    modelDo <- "";

    # Create a list of dummy parameters to trick the checker
    armaToTrickTheChecker <- list(ar=NULL,ma=NULL);
    if(max(p)>0){
        armaToTrickTheChecker$ar <- rep(0.1,max(p));
    }
    if(max(q)>0){
        armaToTrickTheChecker$ma <- rep(0.1,max(q));
    }

    # Call parametersChecker
    checkerReturn <- parametersChecker(data=data, model=model, lags=lags, formulaToUse=formula,
                                       orders=orders_list, constant=constant, arma=armaToTrickTheChecker,
                                       outliers=outliers, level=level,
                                       persistence=persistence, phi=phi, initial=initial,
                                       distribution=distribution, loss=loss, h=h, holdout=holdout,
                                       occurrence=occurrence, ic=ic, bounds=bounds,
                                       regressors=regressors, yName=yName,
                                       silent=silent, modelDo=modelDo,
                                       ellipsis=ellipsis, fast=FALSE);
    list2env(checkerReturn, envir=environment());

    # Reset the parameters. This is to address the trick to the checker
    armaParameters <- arma;
    arEstimate <- maEstimate <- TRUE;

    #### Hack the outputs of the function to align with sparma ####

    if(obsInSample <= K + 1) {
        stop("Not enough observations for the specified orders");
    }

    # Handle arma parameter input
    if(!is.null(arma)) {
        if(length(arma)==2){
            armaParameters <- arma;

            if(is.null(armaParameters$ar)){
                arEstimate <- FALSE;
            }
            else{
                arEstimate <- TRUE;
            }
            if(is.null(armaParameters$ma)){
                maEstimate <- FALSE;
            }
            else{
                maEstimate <- TRUE;
            }
        }
        else{
            warning("arma needs to be of length 2. I'll ignore it and estimate the parameters.");

            arEstimate <- arRequired;
            maEstimate <- maRequired;
        }
    }
    else{
        arEstimate <- arRequired;
        maEstimate <- maRequired;
    }

    # Fix lags, orders etc
    ordersUnique <- sort(unique(c(p,q)));
    lagsModelARIMA <- lagsModelARIMA[lagsModelARIMA %in% ordersUnique, ,drop=FALSE];

    lagsModelAll <- lagsModelARIMA;
    lagsModelMax <- max(lagsModelAll);
    initialArimaNumber <- componentsNumberARIMA <- length(lagsModelAll);
    componentsNamesARIMA <- componentsNamesARIMA[lagsModelAll];

    # Fix the non-zero ARI/MA to have the sparse ones
    nonZeroARI <- nonZeroARI[nonZeroARI[,2] %in% p,, drop=FALSE];
    nonZeroMA <- nonZeroMA[nonZeroMA[,2] %in% q,, drop=FALSE];

    ordersLeft <- unique(sort(c(nonZeroARI[,2], nonZeroMA[,2])));

    # First column is where the thing is in the states vector
    # The second column is the place in the polynomial
    nonZeroARI[,1] <- which(ordersLeft %in% nonZeroARI[,2]);
    nonZeroMA[,1] <- which(ordersLeft %in% nonZeroMA[,2]);

    pLength <- length(p);
    qLength <- length(q);

    # Initial parameter values
    if(arRequired && arEstimate){
        arValue <- rep(0.1, pLength);
    }
    else if(arRequired){
        arValue <- armaParameters[[1]];
    }
    else{
        arValue <- NULL;
    }

    if(maRequired && maEstimate){
        maValue <- rep(0.1, qLength);
    }
    else if(maRequired){
        maValue <- armaParameters[[2]];
    }
    else{
        maValue <- NULL;
    }

    if(constantRequired && constantEstimate) {
        constantValue <- mean(yInSample);
        lagsModelAll <- matrix(c(lagsModelAll,1), ncol=1);
    }
    else{
        constantValue <- NULL;
    }

    # Create C++ adam class, which will then use fit, forecast etc methods
    adamCpp <- new(adamCore,
                   lagsModelAll, Etype, Ttype, Stype,
                   componentsNumberETSNonSeasonal,
                   componentsNumberETSSeasonal,
                   componentsNumberETS, componentsNumberARIMA,
                   xregNumber, length(lagsModelAll),
                   constantRequired, FALSE);

    # Helper function: Create initial state space matrices ####
    sparmaMatricesCreator <- function(p, q, armaParameters,
                                      arRequired, arEstimate,
                                      maRequired, maEstimate,
                                      obsInSample,
                                      lagsModelAll, lagsModelMax,
                                      nonZeroARI, nonZeroMA,
                                      componentsNumberARIMA,
                                      componentsNamesARIMA,
                                      constantRequired, constantName){

        # Build measurement matrix (rows = observations, cols = states)
        matWt <- matrix(1, nrow=obsInSample, ncol=componentsNumberARIMA+constantRequired,
                        dimnames=list(NULL, c(componentsNamesARIMA, constantName)));

        vecG <- matrix(0, componentsNumberARIMA+constantRequired, 1);

        # Build transition matrix F
        matF <- matrix(0, componentsNumberARIMA+constantRequired, componentsNumberARIMA+constantRequired);

        # Fill in the transition where the AR is present
        if(arRequired && !arEstimate){
            matF[nonZeroARI[,1],] <- armaParameters$ar;
            vecG[nonZeroARI[,1],] <- vecG[nonZeroARI[,1],] + armaParameters$ar;
        }

        # Fill in the transition where the AR is present
        if(maRequired && !maEstimate){
            vecG[nonZeroMA[,1],] <- vecG[nonZeroMA[,1],] + armaParameters$ma;
        }

        if(constantRequired){
            matF[componentsNumberARIMA+constantRequired, componentsNumberARIMA+constantRequired] <- 1;
        }

        # Initialize state matrix
        matVt <- matrix(0, componentsNumberARIMA+constantRequired, obsInSample+lagsModelMax,
                        dimnames=list(c(componentsNamesARIMA, constantName), NULL));

        return(list(matVt=matVt, matWt=matWt, matF=matF, vecG=vecG));
    }


    # Helper function: Fill matrices with parameters from vector B ####
    sparmaMatricesFiller <- function(B, matricesCreated,
                                     arRequired, maRequired, constantRequired,
                                     arEstimate, maEstimate, constantEstimate,
                                     arValue, maValue, constantValue,
                                     lagsModelAll, lagsModelMax,
                                     nonZeroARI, nonZeroMA,
                                     componentsNumberARIMA,
                                     p, q, pLength, qLength,
                                     initialType) {

        idx <- 0

        # Extract AR parameter
        if(arRequired && arEstimate) {
            arValue <- B[idx+1:pLength];
            idx[] <- idx + pLength;
        }

        # Extract MA parameter
        if(maRequired && maEstimate) {
            maValue <- B[idx+1:qLength];
            idx[] <- idx + qLength;
        }

        # Fill in the transition and persistence where the AR is present
        if(arRequired && arEstimate){
            matricesCreated$matF[nonZeroARI[,1],1:componentsNumberARIMA] <- arValue;
            matricesCreated$vecG[nonZeroARI[,1],] <- matricesCreated$vecG[nonZeroARI[,1],] + arValue;
        }
        # Fill in the persistence where the MA is present
        if(maRequired && maEstimate){
            matricesCreated$vecG[nonZeroMA[,1],] <- matricesCreated$vecG[nonZeroMA[,1],] + maValue;
        }

        if(initialType=="optimal"){
            # Fill in the AR components
            matricesCreated$matVt[nonZeroARI[,1], 1:componentsNumberARIMA] <- B[idx+c(1:componentsNumberARIMA)];
            # MA components are zero, so don't bother
            idx[] <- idx + componentsNumberARIMA;
        }

        # Extract constant
        if(constantRequired){
            if(constantEstimate){
                idx[] <- idx + 1;
                constantValue <- B[idx];
            }
            matricesCreated$matVt[length(lagsModelAll), 1:lagsModelMax] <- constantValue;
        }

        return(matricesCreated);
    }

    # Create state space matrices
    matricesCreated <- sparmaMatricesCreator(p, q, armaParameters,
                                             arRequired, arEstimate,
                                             maRequired, maEstimate,
                                             obsInSample,
                                             lagsModelAll, lagsModelMax,
                                             nonZeroARI, nonZeroMA,
                                             componentsNumberARIMA,
                                             componentsNamesARIMA,
                                             constantRequired, constantName);

    matVt <- matricesCreated$matVt;
    matWt <- matricesCreated$matWt;
    matF <- matricesCreated$matF;
    vecG <- matricesCreated$vecG;

    # Create profiles for C++ fitter
    profilesList <- adamProfileCreator(lagsModelAll, lagsModelMax, obsAll);
    indexLookupTable <- profilesList$lookup;
    profilesRecentInitial <- profilesRecentTable <- profilesList$recent;


    ##### Function returns scale parameter for the provided parameters #####
    scaler <- function(errors, obsInSample){
        return(sqrt(sum(errors^2)/obsInSample));
    }

    # Cost function using C++ fitter
    CF <- function(B){
        # Fill matrices with parameters from B
        matricesFilled <- sparmaMatricesFiller(B, matricesCreated,
                                               arRequired, maRequired, constantRequired,
                                               arEstimate, maEstimate, constantEstimate,
                                               arValue, maValue, constantValue,
                                               lagsModelAll, lagsModelMax,
                                               nonZeroARI, nonZeroMA,
                                               componentsNumberARIMA,
                                               p, q, pLength, qLength,
                                               initialType);

        profilesRecentTable[] <- matricesFilled$matVt[,1:lagsModelMax];

        # Fit using C++ function
        adamFitted <- adamCpp$fit(matricesFilled$matVt, matricesFilled$matWt,
                                  matricesFilled$matF, matricesFilled$vecG,
                                  indexLookupTable, profilesRecentTable,
                                  yInSample, ot,
                                  any(initialType==c("complete","backcasting","gradient")), nIterations, "n");

        if(!multisteps){
            if(loss=="likelihood"){
                # Scale for different functions
                scale <- scaler(adamFitted$errors, obsInSample);

                # Calculate the likelihood
                CFValue <- -sum(dnorm(x=yInSample[otLogical],
                                      mean=adamFitted$fitted[otLogical],
                                      sd=scale, log=TRUE));
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
            else if(loss=="custom"){
                CFValue <- lossFunction(actual=yInSample,fitted=adamFitted$fitted,B=B);
            }
        }
        else{
            # Call for the Rcpp function to produce a matrix of multistep errors
            adamErrors <- adamCpp$ferrors(adamFitted$states, matWt,
                                          matricesFilled$matF,
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

    #### Likelihood function ####
    logLikFunction <- function(B){
        return(-CF(B));
    }

    if(is.null(B)){
        # Build initial parameter vector
        B <- vector("numeric", arEstimate*pLength + maEstimate*qLength +
                        (initialType=="optimal")*componentsNumberARIMA +
                        constantEstimate);
        names(B) <- c(paste0("phi",p), paste0("theta",q),
                      paste0("initial",c(1:componentsNumberARIMA)),
                      constantName)[c(rep(arEstimate, pLength), rep(maEstimate, qLength),
                                      rep((initialType=="optimal"),componentsNumberARIMA),
                                      constantEstimate)];

        idx <- 0
        if(arEstimate) {
            pacfValues <- rep(0.1, pLength);
            pacfValues[] <- pacf(yInSample, lag.max=max(p), plot=FALSE)$acf[nonZeroARI[,2]];
            B[idx+1:pLength] <- pacfValues;
            idx[] <- idx + pLength;
        }
        if(maEstimate) {
            acfValues <- rep(-0.1, qLength);
            acfValues[] <- acf(yInSample, lag.max=max(q), plot=FALSE)$acf[1+nonZeroMA[,2]];
            B[idx+1:qLength] <- acfValues;
            idx[] <- idx + qLength;
        }
        if(initialType == "optimal") {
            B[idx + c(1:componentsNumberARIMA)] <- yInSample[c(1:componentsNumberARIMA)];
            idx[] <- idx + componentsNumberARIMA;
        }
        if(constantEstimate) {
            idx <- idx + 1;
            B[idx] <- constantValue;
        }
    }

    #### Parameters of the optimiser ####
    print_level_hidden <- print_level;
    if(print_level==41){
        cat("Initial parameters:", B,"\n");
        print_level[] <- 0;
    }

    maxevalUsed <- maxeval;
    if(is.null(maxeval)){
        maxevalUsed <- length(B) * 200;
    }

    # Optimize if there are parameters to optimise
    if(length(B) > 0){
        opts <- list(algorithm=algorithm, maxeval=maxevalUsed,
                     xtol_rel=xtol_rel, ftol_rel=ftol_rel, print_level=print_level_hidden);
        res <- nloptr(x0=B, eval_f=CF, opts=opts);
        res$call <- quote(nloptr(x0=B, eval_f=CF, opts=opts));

        B[] <- res$solution
        CFValue <- res$objective;

        if(print_level_hidden>0){
            print(res);
        }
    }
    else{
        CFValue <- CF(B);
    }

    nStatesBackcasting <- 0;
    # Calculate the number of degrees of freedom coming from states in case of backcasting
    if(any(initialType==c("backcasting","complete"))){
        # Fill matrices with parameters from B
        matricesFilled <- sparmaMatricesFiller(B, matricesCreated,
                                               arRequired, maRequired, constantRequired,
                                               arEstimate, maEstimate, constantEstimate,
                                               arValue, maValue, constantValue,
                                               lagsModelAll, lagsModelMax,
                                               nonZeroARI, nonZeroMA,
                                               componentsNumberARIMA,
                                               p, q, pLength, qLength,
                                               initialType);

        nStatesBackcasting[] <- calculateBackcastingDF(profilesRecentTable, lagsModelAll,
                                                       FALSE, Stype, componentsNumberETSNonSeasonal,
                                                       componentsNumberETSSeasonal, matricesFilled$vecG, matricesFilled$matF,
                                                       obsInSample, lagsModelMax, indexLookupTable,
                                                       adamCpp);
    }

    # Parameters estimated + variance
    nParamEstimated <- length(B) + nStatesBackcasting;
    parametersNumber[1,1] <- nParamEstimated;

    # Final fit with optimized parameters
    matricesFinal <- sparmaMatricesFiller(B, matricesCreated,
                                          arRequired, maRequired, constantRequired,
                                          arEstimate, maEstimate, constantEstimate,
                                          arValue, maValue, constantValue,
                                          lagsModelAll, lagsModelMax,
                                          nonZeroARI, nonZeroMA,
                                          componentsNumberARIMA,
                                          p, q, pLength, qLength,
                                          initialType);

    profilesRecentInitial[] <- profilesRecentTable[] <- matricesFinal$matVt[,1:lagsModelMax];

    # Fit using C++ function
    adamFitted <- adamCpp$fit(matricesFinal$matVt, matricesFinal$matWt,
                              matricesFinal$matF, matricesFinal$vecG,
                              indexLookupTable, profilesRecentTable,
                              yInSample, ot,
                              any(initialType==c("complete","backcasting","gradient")), nIterations, "n");

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
    # Write down the recent profile for future use
    profilesRecentTable <- adamFitted$profile;
    matVt[] <- adamFitted$states;

    # Calculate final loss and logLik
    scale <- scaler(adamFitted$errors, obsInSample);

    logLikValue <- logLikFunction(B);

    if(any(yClasses=="ts")){
        yForecast <- ts(rep(NA, max(1,h)), start=yForecastStart, frequency=yFrequency);
    }
    else{
        yForecast <- zoo(rep(NA, max(1,h)), order.by=yForecastIndex);
    }

    # Forecasting if h > 0
    if(h>0){
        yForecast[] <- adamCpp$forecast(tail(matricesFinal$matWt,h), matricesFinal$matF,
                                        indexLookupTable[,lagsModelMax+obsInSample+c(1:h),drop=FALSE],
                                        profilesRecentTable,
                                        h)$forecast;
    }
    else{
        yForecast[] <- NA;
    }

    ##### Deal with the holdout sample #####
    if(holdout && h>0){
        errormeasures <- measures(yHoldout,yForecast,yInSample);
    }
    else{
        errormeasures <- NULL;
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


    # Build model name
    modelName <- paste0("SpARMA(", paste0(p, collapse=","), ";", paste0(q, collapse=","), ")");
    if(constantRequired){
        modelName <- paste0(modelName, " with constant");
        constantValue <- B["constant"];
    }

    initialValue <- list(arma=matricesFinal$matVt[,1:lagsModelMax]);

    # Record the ARMA parameters
    if(is.null(arma)){
        arma <- vector("list", 2);
        names(arma) <- c("ar","ma");
        idx <- 0;
        if(any(p>0)){
            arma$ar <- B[1:pLength];
            idx[] <- idx + pLength;
        }
        if(any(q>0)){
            arma$ma <- B[idx+1:qLength];
        }
    }

    parametersNumber[1,4] <- (loss=="likelihood")*1;
    parametersNumber[1,5] <- sum(parametersNumber[1,]);

    ##### Return values #####
    modelReturned <- structure(list(model=modelName, timeElapsed=Sys.time()-startTime,
                                    call=cl, orders=orders, arma=arma, formula=formula,
                                    constant=constantValue,
                                    data=yInSample, holdout=yHoldout, fitted=yFitted, residuals=errors,
                                    forecast=yForecast, states=t(matVt), accuracy=errormeasures,
                                    profile=profilesRecentTable, profileInitial=profilesRecentInitial,
                                    persistence=matricesFinal$vecG[,1], transition=matricesFinal$matF,
                                    measurement=matricesFinal$matWt, initial=initialValue, initialType=initialType,
                                    nParam=parametersNumber,
                                    loss=loss, lossValue=CFValue, lossFunction=lossFunction, logLik=logLikValue,
                                    distribution=distribution, bounds=bounds,
                                    scale=scale, B=B, lags=lags, lagsAll=lagsModelAll, res=res,
                                    adamCpp=adamCpp),
                               class=c("adam","smooth"));

    # Plot if not silent
    if(!silent) {
        plot(modelReturned, which=7)
    }

    return(modelReturned)
}
