#' @param parallel If TRUE, the estimation of ADAM models is done in parallel (used in \code{auto.adam} only).
#' If the number is provided (e.g. \code{parallel=41}), then the specified number of cores is set up.
#' WARNING! Packages \code{foreach} and either \code{doMC} (Linux and Mac only)
#' or \code{doParallel} are needed in order to run the function in parallel.
#' @param outliers Defines what to do with outliers: \code{"ignore"}, so just returning the model,
#' \code{"use"} - detect outliers based on specified \code{level} and include dummies for them in the model,
#' or detect and \code{"select"} those of them that reduce \code{ic} value.
#' @param level What confidence level to use for detection of outliers. The default is 99\%. The specific
#' bounds of confidence interval depend on the distribution used in the model.
#'
#' @examples
#' # Automatic selection of appropriate distribution and orders of ADAM ETS+ARIMA
#' \donttest{ourModel <- auto.adam(rnorm(100,100,10), model="ZZN", lags=c(1,4),
#'                       orders=list(ar=c(2,2),ma=c(2,2),select=TRUE))}
#'
#' @rdname adam
#' @importFrom stats update.formula
#' @export
auto.adam <- function(data, model="ZXZ", lags=c(frequency(data)),
                      orders=list(ar=c(3,3),i=c(2,1),ma=c(3,3),select=TRUE),
                      formula=NULL, regressors=c("use","select","adapt"),
                      occurrence=c("none","auto","fixed","general","odds-ratio","inverse-odds-ratio","direct"),
                      distribution=c("dnorm","dlaplace","ds","dgnorm","dlnorm","dinvgauss","dgamma"),
                      outliers=c("ignore","use","select"), level=0.99,
                      h=0, holdout=FALSE,
                      persistence=NULL, phi=NULL, initial=c("backcasting","optimal","two-stage","complete"), arma=NULL,
                      ic=c("AICc","AIC","BIC","BICc"), bounds=c("usual","admissible","none"),
                      silent=TRUE, parallel=FALSE, ets=c("conventional","adam"), ...){
    # Copyright (C) 2020 - Inf  Ivan Svetunkov

    # Start measuring the time of calculations
    startTime <- Sys.time();
    cl <- match.call();
    env <- environment();

    # paste0() is needed in order to get rid of potential issues with names
    responseName <- paste0(deparse(substitute(data)),collapse="");
    responseName <- make.names(responseName,unique=TRUE);

    #### modelDo, ic ####
    if(any(unlist(strsplit(model,""))=="C")){
        modelDo <- "combine";
    }
    else{
        modelDo <- "select";
    }

    ic <- match.arg(ic,c("AICc","AIC","BIC","BICc"));
    IC <- switch(ic,
                 "AIC"=AIC,
                 "AICc"=AICc,
                 "BIC"=BIC,
                 "BICc"=BICc);

    initial <- match.arg(initial);
    outliers <- match.arg(outliers);

    # Check the provided level value.
    if(length(level)>1){
        warning(paste0("Sorry, but we only support scalar for the level, ",
                       "when constructing in-sample prediction interval. ",
                       "Using the first provided value."),
                call.=FALSE);
        level <- level[1];
    }
    # Fix just in case a silly user used 95 etc instead of 0.95
    if(level>1){
        level[] <- level / 100;
    }

    # The function checks the provided parameters of adam and/or oes
    ##### data #####
    # If this is simulated, extract the actuals
    if(is.adam.sim(data) || is.smooth.sim(data)){
        data <- data$data;
    }
    # If this is Mdata, use all the available stuff
    else if(inherits(data,"Mdata")){
        h <- data$h;
        holdout <- TRUE;
        lags <- frequency(data$x);
        data <- ts(c(data$x,data$xx),start=start(data$x),frequency=frequency(data$x));
    }

    # If this is a vector, use length
    # yInSample is needed for checks only
    if(is.null(dim(data))){
        obsInSample <- length(data) - holdout*h;
        yInSample <- data[1:obsInSample];
    }
    else{
        obsInSample <- nrow(data) - holdout*h;
        if(!is.null(formula)){
            yInSample <- data[1:obsInSample,all.vars(formula)[1]];
        }
        else{
            yInSample <- data[1:obsInSample,1];
        }
        if(is.null(formula)){
            responseName <- colnames(data)[1];
        }
        else{
            responseName <- all.vars(formula)[1];
        }
    }

    #### Create logical, determining, what we are dealing with ####
    # ETS
    etsModel <- all(model!="NNN");
    # These values are needed for number of degrees of freedom check
    Etype <- substr(model,1,1);
    Ttype <- substr(model,2,2);
    Stype <- substr(model,nchar(model),nchar(model));
    damped <- (nchar(model)==4);
    if(length(Etype)>1){
        Etype <- "Z";
    }
    if(length(Ttype)>1){
        Ttype <- "Z";
        damped <- TRUE;
    }
    if(length(Stype)>1){
        Stype <- "Z";
    }

    # ARIMA + ARIMA select
    if(is.list(orders)){
        arimaModel <- any(c(orders$ar,orders$i,orders$ma)>0);
    }
    else{
        arimaModel <- any(orders>0);
    }

    # xreg - either as a separate variable or as a matrix for data
    xregModel <- (!is.null(dim(data)) && ncol(data)>1);
    xregNumber <- 0;
    if(xregModel){
        xregNumber[] <- ncol(data)-1;
    }
    regressors <- match.arg(regressors);

    # If this is non-positive data and positive defined distributions are used, fix this
    if(any(yInSample<=0) && any(c("dlnorm","dllaplace","dls","dinvgauss","dgamma") %in% distribution) &&
       (!is.occurrence(occurrence) && occurrence[1]=="none")){
        distributionToDrop <- c("dlnorm","dllaplace","dls","dinvgauss","dgamma")[
            c("dlnorm","dllaplace","dls","dinvgauss","dgamma") %in% distribution];
        warning(paste0("The data is not strictly positive, so not all distributions make sense. ",
                       "Dropping ",paste0(distributionToDrop,collapse=", "),"."),
                call.=FALSE);
        distribution <- distribution[!(distribution %in% distributionToDrop)];
    }
    nModels <- length(distribution);

    # Amend the default list of distributions for the pure ARIMA
    if(arimaModel && !etsModel && length(distribution) == 7 &&
       all(distribution %in% c("dnorm", "dlaplace", "ds", "dgnorm", "dlnorm", "dinvgauss", "dgamma"))){
        distributionToDrop <- c("dlnorm","dinvgauss","dgamma")[
            c("dlnorm","dinvgauss","dgamma") %in% distribution];
        warning(paste0("Not all distributions make sense for the pure ARIMA. ",
                       "Dropping ",paste0(distributionToDrop,collapse=", "),"."),
                call.=FALSE);
        distribution <- distribution[!(distribution %in% distributionToDrop)];
    }

    #### Checks of provided parameters for ARIMA selection ####
    if(arimaModel){
        arimaModelSelect <- FALSE;
        if(is.list(orders)){
            arimaModelSelect <- orders$select;
            arMax <- orders$ar;
            iMax <- orders$i;
            maMax <- orders$ma;
        }
        else{
            arMax <- orders[1];
            iMax <- orders[2];
            maMax <- orders[3];
        }

        if(is.null(arimaModelSelect)){
            arimaModelSelect <- FALSE;
        }

        if(any(c(arMax,iMax,maMax)<0)){
            stop("Funny guy! How am I gonna construct a model with negative order?",call.=FALSE);
        }

        # If there are no lags for the basic components, correct this.
        if(sum(lags==1)==0){
            lags <- c(1,lags);
        }

        # If there are zero lags, drop them
        if(any(lags==0)){
            arMax <- arMax[lags!=0];
            iMax <- iMax[lags!=0];
            maMax <- maMax[lags!=0];
            lags <- lags[lags!=0];
        }

        # Define maxorder and make all the values look similar (for the polynomials)
        maxorder <- max(length(arMax),length(iMax),length(maMax),length(lags));
        if(length(arMax)!=maxorder){
            arMax <- c(arMax,rep(0,maxorder-length(arMax)));
        }
        if(length(iMax)!=maxorder){
            iMax <- c(iMax,rep(0,maxorder-length(iMax)));
        }
        if(length(maMax)!=maxorder){
            maMax <- c(maMax,rep(0,maxorder-length(maMax)));
        }

        # If zeroes are defined as orders for some lags, drop them.
        if(any((arMax + iMax + maMax)==0)){
            orders2leave <- (arMax + iMax + maMax)!=0;
            if(all(!orders2leave)){
                orders2leave <- lags==min(lags);
            }
            arMax <- arMax[orders2leave];
            iMax <- iMax[orders2leave];
            maMax <- maMax[orders2leave];
            lags <- lags[orders2leave];
        }

        # Get rid of duplicates in lags
        if(length(unique(lags))!=length(lags)){
            lagsNew <- unique(lags);
            arMaxNew <- iMaxNew <- maMaxNew <- lagsNew;
            for(i in 1:length(lagsNew)){
                arMaxNew[i] <- max(arMax[which(lags==lagsNew[i])],na.rm=TRUE);
                iMaxNew[i] <- max(iMax[which(lags==lagsNew[i])],na.rm=TRUE);
                maMaxNew[i] <- max(maMax[which(lags==lagsNew[i])],na.rm=TRUE);
            }
            arMax <- arMaxNew;
            iMax <- iMaxNew;
            maMax <- maMaxNew;
            lags <- lagsNew;
        }

        # Order things, so we would deal with the lowest level of seasonality first
        arMax <- arMax[order(lags,decreasing=FALSE)];
        iMax <- iMax[order(lags,decreasing=FALSE)];
        maMax <- maMax[order(lags,decreasing=FALSE)];
        lags <- sort(lags,decreasing=FALSE);
        initialArimaNumber <- max((arMax + iMax) %*% lags, maMax %*% lags);
    }
    else{
        arMax <- iMax <- maMax <- NULL;
        arimaModelSelect <- FALSE;
        initialArimaNumber <- 0;
        orders <- c(0,0,0);
    }

    #### Maximum number of parameters to estimate ####
    nParamMax <- (1 +
                      # ETS model
                      etsModel*((Etype!="N") + (Ttype!="N") + (Stype!="N")*length(lags) + damped +
                                    any(initial==c("optimal","two-stage")) * ((Etype!="N") + (Ttype!="N") + (Stype!="N")*sum(lags))) +
                      # ARIMA components: initials + parameters
                      arimaModel*(initialArimaNumber + sum(arMax) + sum(maMax)) +
                      # Xreg initials and smoothing parameters
                      xregModel*(xregNumber*(1+(regressors=="adapt"))));

    # Do something in order to make sure that the stuff works
    if((nParamMax > obsInSample) && arimaModelSelect){
        # If this is ARIMA, remove some orders
        if(arimaModel){
            nParamMaxNonARIMA <- nParamMax - (initialArimaNumber + sum(arMax) + sum(maMax));
            if(obsInSample > nParamMaxNonARIMA){
                # Drop out some ARIMA orders, start with seasonal
                # Reduce maximum order of AR
                while(nParamMax > obsInSample){
                    arimaTail <- max(tail(which(arMax!=0),1),tail(which(iMax!=0),1),tail(which(maMax!=0),1));
                    if(arMax[arimaTail]>0){
                        arMax[arimaTail] <- arMax[arimaTail]-1;
                        initialArimaNumber[] <- max((arMax + iMax) %*% lags, maMax %*% lags);
                        nParamMax[] <- (1 +
                                            # ETS model
                                            etsModel*((Etype!="N") + (Ttype!="N") + (Stype!="N")*length(lags) + damped +
                                                          any(initial==c("optimal","two-stage")) *
                                                          ((Etype!="N") + (Ttype!="N") + (Stype!="N")*sum(lags))) +
                                            # ARIMA components: initials + parameters
                                            arimaModel*(initialArimaNumber + sum(arMax) + sum(maMax)) +
                                            # Xreg initials and smoothing parameters
                                            xregModel*(xregNumber*(1+(regressors=="adapt"))));
                    }

                    # Reduce maximum order of I
                    if(nParamMax > obsInSample && iMax[arimaTail]>0){
                        iMax[arimaTail] <- iMax[arimaTail]-1;
                        initialArimaNumber[] <- max((arMax + iMax) %*% lags, maMax %*% lags);
                        nParamMax[] <- (1 +
                                            # ETS model
                                            etsModel*((Etype!="N") + (Ttype!="N") + (Stype!="N")*length(lags) + damped +
                                                          any(initial==c("optimal","two-stage")) *
                                                          ((Etype!="N") + (Ttype!="N") + (Stype!="N")*sum(lags))) +
                                            # ARIMA components: initials + parameters
                                            arimaModel*(initialArimaNumber + sum(arMax) + sum(maMax)) +
                                            # Xreg initials and smoothing parameters
                                            xregModel*(xregNumber*(1+(regressors=="adapt"))));
                    }

                    # Reduce maximum order of MA
                    if(nParamMax > obsInSample && maMax[arimaTail]>0){
                        maMax[arimaTail] <- maMax[arimaTail]-1;
                        initialArimaNumber[] <- max((arMax + iMax) %*% lags, maMax %*% lags);
                        nParamMax[] <- (1 +
                                            # ETS model
                                            etsModel*((Etype!="N") + (Ttype!="N") + (Stype!="N")*length(lags) + damped +
                                                          any(initial==c("optimal","two-stage")) *
                                                          ((Etype!="N") + (Ttype!="N") + (Stype!="N")*sum(lags))) +
                                            # ARIMA components: initials + parameters
                                            arimaModel*(initialArimaNumber + sum(arMax) + sum(maMax)) +
                                            # Xreg initials and smoothing parameters
                                            xregModel*(xregNumber*(1+(regressors=="adapt"))));
                    }
                }
            }
        }
    }

    #### Parallel calculations ####
    setupResult <- adam_setupParallel(parallel, nModels);
    parallel <- setupResult$parallel;
    cluster <- setupResult$cluster;

    if(!silent){
        if(!parallel){
            cat("Evaluating models with different distributions... ");
        }
        else{
            cat("Working... ");
        }
    }

    #### The function that does the loop and returns a list of ETS(X) models ####
    adamReturner <- function(data, model, lags, orders,
                             distribution, h, holdout,
                             persistence, phi, initial, arma,
                             occurrence, ic, bounds,
                             regressors, parallel,
                             arimaModelSelect, arMax, iMax, maMax, ...){
        env <- environment();
        # If we select ARIMA, don't do it in the first step
        if(arimaModelSelect){
            ordersToUse <- c(0,0,0);
        }
        else{
            ordersToUse <- orders;
        }

        if(!parallel){
            # Prepare the list of models
            selectedModels <- vector("list",length(distribution));
            for(i in 1:length(distribution)){
                if(!silent){
                    cat(distribution[i],"\b, ");
                }
                if(etsModel || xregModel || (arimaModel && !arimaModelSelect)){
                    selectedModels[[i]] <- adam(data=data, model=model, lags=lags, orders=ordersToUse,
                                                distribution=distribution[i], formula=formula,
                                                h=h, holdout=holdout,
                                                persistence=persistence, phi=phi, initial=initial, arma=arma,
                                                occurrence=occurrence, ic=ic, bounds=bounds,
                                                regressors=regressors, silent=TRUE, ets=ets, environment=env, ...);
                }

                if(arimaModelSelect){
                    selectedModels[[i]] <- adam_arimaSelector(
                        data=data, model=model,
                        lags=lags, arMax=arMax, iMax=iMax, maMax=maMax,
                        h=h, holdout=holdout,
                        persistence=persistence, phi=phi, initial=initial,
                        ic=ic, bounds=bounds, silent=silent, regressors=regressors,
                        testModelETS=selectedModels[[i]],
                        fitter=adam, fitter_args=list(distribution=distribution[i], occurrence=occurrence),
                        IC=IC, formula=formula, ets=ets,
                        responseName=responseName, obsInSample=obsInSample, ...)
                }
            }
        }
        else{
            selectedModels <- foreach::`%dopar%`(foreach::foreach(i=1:length(distribution)),{
                if(etsModel || xregModel){
                    testModel <- adam(data=data, model=model, lags=lags, orders=ordersToUse,
                                      distribution=distribution[i], formula=formula,
                                      h=h, holdout=holdout,
                                      persistence=persistence, phi=phi, initial=initial, arma=arma,
                                      occurrence=occurrence, ic=ic, bounds=bounds,
                                      regressors=regressors, silent=TRUE, ets=ets, environment=env, ...)
                }
                else{
                    testModel <- NULL;
                }

                if(arimaModelSelect){
                    testModel <- adam_arimaSelector(
                        data=data, model=model,
                        lags=lags, arMax=arMax, iMax=iMax, maMax=maMax,
                        h=h, holdout=holdout,
                        persistence=persistence, phi=phi, initial=initial,
                        ic=ic, bounds=bounds, silent=TRUE, regressors=regressors,
                        testModelETS=testModel,
                        fitter=adam, fitter_args=list(distribution=distribution[i], occurrence=occurrence),
                        IC=IC, formula=formula, ets=ets,
                        responseName=responseName, obsInSample=obsInSample, ...)
                }
                return(testModel);
            })
        }
        return(selectedModels);
    }

    #### ARIMA selection script ####
    if(arimaModelSelect){
        if(!is.null(arma)){
            warning("ARIMA order selection cannot be done with the provided arma parameters. Dropping them.",
                    call.=FALSE);
            arma <- NULL;
        }
    }

    #### The function that extracts outliers and refits the model with the selection ####
    # Outliers types to include: additive outlier (AO) and level shift (LS).
    # Introduce leads and lags of outliers in case of "select"
    outlierDetector <- function(adamModel, outliers="use"){
        if(outliers=="ignore"){
            return(adamModel);
        }
        else{
            outliersModel <- outlierdummy(adamModel, level=level);
            if(length(outliersModel$id)>0){
                if(!silent){
                    cat("\nDealing with outliers...");
                }
                # Create a proper xreg matrix
                if(h>0){
                    outliersXreg <- rbind(outliersModel$outliers, matrix(0,nrow=h,length(outliersModel$id)));
                }
                else{
                    outliersXreg <- outliersModel$outliers;
                }
                # If select outliers, then introduce lags and leads
                if(outliers=="select"){
                    # data.frame is needed to bind the thing with ts() object few lines below
                    outliersXreg <- as.data.frame(xregExpander(outliersXreg,c(-1:1),gaps="zero"));
                }
                outliersXregNames <- colnames(outliersXreg);
                outliersDo <- outliers;
                # Additional variables to preserve the class of the object
                yClasses <- class(data);
                if(is.data.frame(data)){
                    yIndex <- time(data[[responseName]]);
                }
                else{
                    yIndex <- time(data);
                }
                notAMatrix <- (is.null(ncol(data)) || (!is.null(ncol(data)) & ncol(data)==1));
                data <- data.frame(data,outliersXreg);
                # Don't loose the zoo class
                if(any(yClasses=="zoo")){
                    if(notAMatrix){
                        data[[1]] <- zoo(data[[1]], order.by=yIndex);
                        colnames(data)[1] <- responseName;
                    }
                    else{
                        data[[responseName]] <- zoo(data[[responseName]], order.by=yIndex);
                    }
                }
                else{
                    if(notAMatrix){
                        data[[1]] <- ts(data[[1]], start=yIndex[1], deltat=yIndex[2]-yIndex[1]);
                        colnames(data)[1] <- responseName;
                    }
                    else{
                        data[[responseName]] <- ts(data[[responseName]], start=yIndex[1], deltat=yIndex[2]-yIndex[1]);
                    }
                }

                # If the names of xreg are wrong, fix them
                if(!all(outliersXregNames %in% colnames(data))){
                    colnames(data)[substr(colnames(data),1,12)=="outliersXreg"] <- outliersXregNames;
                }

                # Form new xreg matrix (check data and xreg)
                if(xregModel){
                    # Update formula if it is provided
                    if(!is.null(formula)){
                        # If this is not the formula of a type y~., then add outliers.
                        if(!(length(all.vars(formula))==2 && all.vars(formula)[2]==".")){
                            formula <- update.formula(as.formula(formula),
                                                      as.formula(paste0("~.+",paste0(colnames(outliersXreg),collapse="+"))));
                        }
                    }
                    else{
                        formula <- as.formula(paste0(responseName,"~."));
                    }
                    # outliersDo <- regressors;
                }
                else{
                    colnames(data)[1] <- responseName;
                }
                newCall <- cl;
                newCall$data <- data;
                newCall$formula <- formula;
                newCall$silent <- TRUE;
                newCall$regressors <- outliersDo;
                newCall$outliers <- "ignore";
                # These two are needed for cases with Mcomp data
                newCall$holdout <- holdout;
                newCall$h <- h;
                newCall$lags <- lags;
                adamModel <- eval(newCall);
            }
            else{
                if(!silent){
                    cat("No outliers detected.\n");
                }
            }
            return(adamModel);
        }
    }

    #### A simple loop, no ARIMA orders selection ####
    if(!arimaModelSelect){
        selectedModels <- adamReturner(data, model, lags, orders,
                                       distribution, h, holdout,
                                       persistence, phi, initial, arma,
                                       occurrence, ic, bounds,
                                       regressors, parallel,
                                       arimaModelSelect, arMax, iMax, maMax, ...);
    }
    else{
        #### If there is ETS(X), do ARIMA selection on residuals ####
        # Extract residuals from adam for each distribution, fit best ARIMA for each, refit the models.
        if(etsModel || xregModel){
            selectedModels <- adamReturner(data, model, lags, orders,
                                           distribution, h, holdout,
                                           persistence, phi, initial, arma,
                                           occurrence, ic, bounds,
                                           regressors, parallel,
                                           arimaModelSelect, arMax, iMax, maMax, ...);
        }
        #### Otherwise, do the stuff directly ####
        # Do ARIMA selection for each distribution in parallel.
        else{
            if(!parallel){
                # Prepare the list of models
                selectedModels <- vector("list",length(distribution));
                for(i in 1:length(distribution)){
                    if(!silent){
                        cat(distribution[i],"\b: ");
                    }
                    selectedModels[[i]] <- adam_arimaSelector(
                        data=data, model=model,
                        lags=lags, arMax=arMax, iMax=iMax, maMax=maMax,
                        h=h, holdout=holdout,
                        persistence=persistence, phi=phi, initial=initial,
                        ic=ic, bounds=bounds, silent=silent, regressors=regressors,
                        testModelETS=NULL,
                        fitter=adam, fitter_args=list(distribution=distribution[i], occurrence=occurrence),
                        IC=IC, formula=formula, ets=ets,
                        responseName=responseName, obsInSample=obsInSample, ...);
                }
            }
            else{
                selectedModels <- foreach::`%dopar%`(foreach::foreach(i=1:length(distribution)),{
                    testModel <- adam_arimaSelector(
                        data=data, model=model,
                        lags=lags, arMax=arMax, iMax=iMax, maMax=maMax,
                        h=h, holdout=holdout,
                        persistence=persistence, phi=phi, initial=initial,
                        ic=ic, bounds=bounds, silent=TRUE, regressors=regressors,
                        testModelETS=NULL,
                        fitter=adam, fitter_args=list(distribution=distribution[i], occurrence=occurrence),
                        IC=IC, formula=formula, ets=ets,
                        responseName=responseName, obsInSample=obsInSample, ...);
                    return(testModel);
                })
            }
        }
    }

    if(modelDo=="select"){
        ICValues <- sapply(selectedModels, IC);
    }
    else{
        ICValues <- vector("numeric",length(distribution));
        for(i in 1:length(distribution)){
            ICValues[i] <- selectedModels[[i]]$ICs[is.finite(selectedModels[[i]]$ICs)] %*%
                selectedModels[[i]]$ICw[is.finite(selectedModels[[i]]$ICs)];
        }
    }

    selectedModels[[which.min(ICValues)]]$timeElapsed <- Sys.time()-startTime;
    # selectedModels[[which.min(ICValues)]]$formula <- as.formula(do.call("substitute",
    #                                                                     list(expr=selectedModels[[which.min(ICValues)]]$formula,
    #                                                                          env=list(y=as.name(responseName)))));
    # names(ICValues) <- sapply(selectedModels, modelType);
    # selectedModels[[which.min(ICValues)]]$ICValues <- ICValues;

    if(outliers!="ignore"){
        selectedModels[[which.min(ICValues)]] <- outlierDetector(selectedModels[[which.min(ICValues)]], outliers=outliers);
    }

    if(!silent){
        if(outliers=="ignore"){
            cat("Done!\n");
        }
        plot(selectedModels[[which.min(ICValues)]],7);
    }

    selectedModels[[which.min(ICValues)]]$call <- cl;

    adam_teardownParallel(cluster);

    return(selectedModels[[which.min(ICValues)]]);
}
