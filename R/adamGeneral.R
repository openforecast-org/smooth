commonParametersChecker <- function(data, model, lags, formulaToUse, orders, constant=FALSE, arma,
                                    persistence, phi, initial,
                                    distribution=c("default","dnorm","dlaplace","dalaplace","ds","dgnorm",
                                                   "dlnorm","dinvgauss","dgamma","plogis"),
                                    loss, h, holdout, occurrence,
                                    ic=c("AICc","AIC","BIC","BICc"), bounds=c("usual","admissible","none"),
                                    regressors, yName,
                                    silent, modelDo,
                                    ellipsis, fast=FALSE){

    #### Data ####
    list2env(adam_checkData(data, lags, h, holdout, yName, modelDo, formulaToUse),
             envir=environment());

    #### Check what is used for the model ####
    if(!is.character(model)){
        stop(paste0("Something strange is provided instead of character object in model: ",
                    paste0(model,collapse=",")),call.=FALSE);
    }

    # Predefine models pool for a model selection
    modelsPool <- NULL;
    if(!fast){
        # Deal with the list of models. Check what has been provided. Stop if there is a mistake.
        if(length(model)>1){
            if(any(nchar(model)>4) || any(nchar(model)<3)){
                stop(paste0("You have defined strange model(s) in the pool: ",
                            paste0(model[nchar(model)>4 | nchar(model)<3],collapse=",")),call.=FALSE);
            }
            else if(any(substr(model,1,1)!="A" & substr(model,1,1)!="M" & substr(model,1,1)!="C")){
                stop(paste0("You have defined strange model(s) in the pool: ",
                            paste0(model[substr(model,1,1)!="A" & substr(model,1,1)!="M"],collapse=",")),
                     call.=FALSE);
            }
            else if(any(substr(model,2,2)!="N" & substr(model,2,2)!="A" &
                        substr(model,2,2)!="M" & substr(model,2,2)!="C")){
                stop(paste0("You have defined strange model(s) in the pool: ",
                            paste0(model[substr(model,2,2)!="N" & substr(model,2,2)!="A" &
                                             substr(model,2,2)!="M"],collapse=",")),call.=FALSE);
            }
            else if(any(substr(model,3,3)!="N" & substr(model,3,3)!="A" &
                        substr(model,3,3)!="M" & substr(model,3,3)!="d" & substr(model,3,3)!="C")){
                stop(paste0("You have defined strange model(s) in the pool: ",
                            paste0(model[substr(model,3,3)!="N" & substr(model,3,3)!="A" &
                                             substr(model,3,3)!="M" & substr(model,3,3)!="d"],collapse=",")),
                     call.=FALSE);
            }
            else if(any(nchar(model)==4 & substr(model,4,4)!="N" & substr(model,4,4)!="A" &
                        substr(model,4,4)!="M" & substr(model,4,4)!="C")){
                stop(paste0("You have defined strange model(s) in the pool: ",
                            paste0(model[nchar(model)==4 & substr(model,4,4)!="N" &
                                             substr(model,4,4)!="A" & substr(model,4,4)!="M"],collapse=",")),
                     call.=FALSE);
            }
            else{
                modelsPoolCombiner <- (substr(model,1,1)=="C" | substr(model,2,2)=="C" |
                                           substr(model,3,3)=="C" | substr(model,4,4)=="C");
                modelsPool <- model[!modelsPoolCombiner];
                modelsPool <- unique(modelsPool);
                if(any(modelsPoolCombiner)){
                    if(any(substr(model,nchar(model),nchar(model))!="N")){
                        model <- "CCC";
                    }
                    else{
                        model <- "CCN";
                    }
                }
                else{
                    model <- c("Z","Z","Z");
                    if(all(substr(modelsPool,nchar(modelsPool),nchar(modelsPool))=="N")){
                        model[3] <- "N";
                    }
                    if(all(substr(modelsPool,2,2)=="N")){
                        model[2] <- "N";
                    }
                    model <- paste0(model,collapse="");
                }
            }
        }
    }

    # If chosen model is "AAdN" or anything like that, we are taking the appropriate values
    if(nchar(model)==4){
        Etype <- substr(model,1,1);
        Ttype <- substr(model,2,2);
        Stype <- substr(model,4,4);
        damped <- TRUE;
        if(substr(model,3,3)!="d"){
            message(paste0("You have defined a strange model: ", model,
                           ". Switching to ", paste0(Etype,Ttype,"d",Stype)));
            model <- paste0(Etype,Ttype,"d",Stype);
        }
    }
    else if(nchar(model)==3){
        Etype <- substr(model,1,1);
        Ttype <- substr(model,2,2);
        Stype <- substr(model,3,3);
        if(any(Ttype==c("Z","X","Y"))){
            damped <- TRUE;
        }
        else{
            damped <- FALSE;
        }
    }
    else{
        message(paste0("You have defined a strange model: ",model));
        message("Switching to 'ZZZ'");
        model <- "ZZZ";

        Etype <- "Z";
        Ttype <- "Z";
        Stype <- "Z";
        damped <- TRUE;
    }

    # Define if we want to select or combine models... or do none of the above.
    if(is.null(modelsPool)){
        if(any(unlist(strsplit(model,""))=="C")){
            modelDo <- "combine";
            if(Etype=="C"){
                Etype <- "Z";
            }
            if(Ttype=="C"){
                Ttype <- "Z";
            }
            if(Stype=="C"){
                Stype <- "Z";
            }
        }
        else if(any(unlist(strsplit(model,""))=="Z") ||
                any(unlist(strsplit(model,""))=="X") ||
                any(unlist(strsplit(model,""))=="Y") ||
                any(unlist(strsplit(model,""))=="F") ||
                any(unlist(strsplit(model,""))=="P") ||
                any(unlist(strsplit(model,""))=="S")){
            modelDo <- "select";

            # The full test, sidestepping branch and bound
            if(any(unlist(strsplit(model,""))=="F")){
                modelsPool <- c("ANN","AAN","AAdN","AMN","AMdN",
                                "ANA","AAA","AAdA","AMA","AMdA",
                                "ANM","AAM","AAdM","AMM","AMdM",
                                "MNN","MAN","MAdN","MMN","MMdN",
                                "MNA","MAA","MAdA","MMA","MMdA",
                                "MNM","MAM","MAdM","MMM","MMdM");
                # Remove models from pool if specific elements are provided
                if(Etype!="F"){
                    modelsPool <- modelsPool[substr(modelsPool,1,1)==Etype];
                }
                else{
                    Etype[] <- "Z"
                }
                if(Ttype!="F"){
                    modelsPool <- modelsPool[substr(modelsPool,2,2)==Ttype];
                }
                else{
                    Ttype[] <- "Z"
                }
                if(Stype!="F"){
                    modelsPool <- modelsPool[substr(modelsPool,nchar(modelsPool),nchar(modelsPool))==Stype];
                }
                else{
                    Stype[] <- "Z"
                }
                model <- "FFF";
            }

            # The test for pure models only
            if(any(unlist(strsplit(model,""))=="P")){
                modelsPool <- c("ANN","AAN","AAdN","ANA","AAA","AAdA",
                                "MNN","MMN","MMdN","MNM","MMM","MMdM");
                # Remove models from pool if specific elements are provided
                if(Etype!="P"){
                    modelsPool <- modelsPool[substr(modelsPool,1,1)==Etype];
                }
                else{
                    Etype[] <- "Z"
                }
                if(Ttype!="P"){
                    modelsPool <- modelsPool[substr(modelsPool,2,2)==Ttype];
                }
                else{
                    Ttype[] <- "Z"
                }
                if(Stype!="P"){
                    modelsPool <- modelsPool[substr(modelsPool,nchar(modelsPool),nchar(modelsPool))==Stype];
                }
                else{
                    Stype[] <- "Z"
                }
                model <- "PPP";
            }

            # The pool of sensible/standard models, those that have finite variance
            if(any(unlist(strsplit(model,""))=="S")){
                modelsPool <- c("ANN", "AAN", "AAdN", "ANA", "AAA", "AAdA",
                                "MNN", "MAN", "MAdN", "MNA", "MAA", "MAdA",
                                "MNM", "MAM", "MAdM", "MMN", "MMdN", "MMM", "MMdM");
                # Remove models from pool if specific elements are provided
                if(Etype!="S"){
                    # Switch is needed here if people want to restrict the basic pool further
                    modelsPool <- modelsPool[substr(modelsPool,1,1)==switch(Etype,
                                                                            "X"="A",
                                                                            "Y"="M",
                                                                            Etype)];
                }
                else{
                    Etype[] <- "Z"
                }
                if(Ttype!="S"){
                    modelsPool <- modelsPool[substr(modelsPool,2,2) %in% switch(Ttype,
                                                                                "X"=c("A","N"),
                                                                                "Y"=c("M","N"),
                                                                                Ttype)];
                }
                else{
                    Ttype[] <- "Z"
                }
                if(Stype!="S"){
                    modelsPool <- modelsPool[substr(modelsPool,
                                                    nchar(modelsPool),
                                                    nchar(modelsPool))==switch(Stype,
                                                                               "X"=c("A","N"),
                                                                               "Y"=c("M","N"),
                                                                               Stype)];
                }
                else{
                    Stype[] <- "Z"
                }
                model <- "SSS";
            }
        }
        else{
            modelDo <- "estimate";
        }

        if(Etype=="X"){
            Etype <- "A";
        }
        else if(Etype=="Y"){
            Etype <- "M";
        }
    }
    else{
        if(any(unlist(strsplit(model,""))=="C")){
            modelDo <- "combine";
        }
        else{
            modelDo <- "select";
        }
    }

    #### Check the components of model ####
    ### Check error type
    if(all(Etype!=c("Z","X","Y","A","M","C","N"))){
        warning(paste0("Wrong error type: ",Etype,". Should be 'Z', 'X', 'Y', 'C', 'A', 'M' or 'N'. ",
                       "Changing to 'Z'"),call.=FALSE);
        Etype <- "Z";
        modelDo <- "select";
    }
    # If the error is "N", then switch off the ETS model
    if(Etype=="N"){
        componentsNamesETS <- NULL;
        componentsNumberETS <- 0;
        Ttype[] <- "N";
        damped[] <- FALSE;
        Stype[] <- "N";
        modelDo[] <- "estimate";
        modelsPool[] <- NULL;
        etsModel <- FALSE;
    }
    else{
        componentsNamesETS <- "level";
        componentsNumberETS <- 1;
        etsModel <- TRUE;
    }

    if(etsModel){
        ### Check trend type
        if(all(Ttype!=c("Z","X","Y","N","A","M","C"))){
            warning(paste0("Wrong trend type: ",Ttype,". Should be 'Z', 'X', 'Y', 'C' 'N', 'A', or 'M'. ",
                           "Changing to 'Z'"),call.=FALSE);
            Ttype <- "Z";
            modelDo <- "select";
        }
        modelIsTrendy <- (Ttype!="N");
        if(modelIsTrendy){
            componentsNamesETS <- c(componentsNamesETS,"trend");
            componentsNumberETS[] <- componentsNumberETS+1;
        }
    }
    else{
        modelIsTrendy <- FALSE;
    }

    #### Check the lags vector ####
    if(any(c(lags)<0)){
        stop("Right! Why don't you try complex lags then, mister smart guy?",call.=FALSE);
    }

    # If there are zero lags, drop them
    if(any(lags==0)){
        lags <- lags[lags!=0];
    }

    # Form the lags based on the provided stuff. Get rid of ones and leave unique seasonals
    # Add one for the level
    lags <- c(1,unique(lags[lags>1]));

    # Warning if the lags length is higher than the sample size
    if(max(lags) > obsInSample){
        warning("The maximum lags value is ", max(lags),
                ", while the sample size is ", obsInSample,
                ". I cannot fit the seasonal model in this case. ",
                "Dropping the highest lag.",
                call.=FALSE);
        lags <- lags[-which.max(lags)];
    }

    #### ARIMA term ####
    # This should be available for pure models only
    if(is.list(orders)){
        arOrders <- orders$ar;
        iOrders <- orders$i;
        maOrders <- orders$ma;
        select <- orders$select;
        if(is.null(select)){
            select <- FALSE;
        }
    }
    else if(is.vector(orders)){
        arOrders <- orders[1];
        iOrders <- orders[2];
        maOrders <- orders[3];
        select <- FALSE;
    }
    else{
        select <- FALSE;
    }

    # If there is arima, prepare orders
    if(sum(c(arOrders,iOrders,maOrders))>0){
        arimaModel <- TRUE;

        # See if AR is needed
        arRequired <- FALSE;
        if(sum(arOrders)>0){
            arRequired[] <- TRUE;
        }

        # See if I is needed
        iRequired <- FALSE;
        if(sum(iOrders)>0){
            iRequired[] <- TRUE;
        }

        # See if I is needed
        maRequired <- FALSE;
        if(sum(maOrders)>0){
            maRequired[] <- TRUE;
        }

        # Define maxOrder and make all the values look similar (for the polynomials)
        maxOrder <- max(length(arOrders),length(iOrders),length(maOrders),length(lags));
        if(length(arOrders)!=maxOrder){
            arOrders <- c(arOrders,rep(0,maxOrder-length(arOrders)));
        }
        if(length(iOrders)!=maxOrder){
            iOrders <- c(iOrders,rep(0,maxOrder-length(iOrders)));
        }
        if(length(maOrders)!=maxOrder){
            maOrders <- c(maOrders,rep(0,maxOrder-length(maOrders)));
        }
        if(length(lags)!=maxOrder){
            lagsNew <- c(lags,rep(0,maxOrder-length(lags)));
            arOrders <- arOrders[lagsNew!=0];
            iOrders <- iOrders[lagsNew!=0];
            maOrders <- maOrders[lagsNew!=0];
        }

        # Define the non-zero values. This is done via the calculation of orders of polynomials
        ariValues <- list(NA);
        maValues <- list(NA);
        for(i in 1:length(lags)){
            ariValues[[i]] <- c(0,min(1,arOrders[i]):arOrders[i])
            if(iOrders[i]!=0){
                ariValues[[i]] <- c(ariValues[[i]],1:iOrders[i]+arOrders[i]);
            }
            ariValues[[i]] <- unique(ariValues[[i]] * lags[i]);
            maValues[[i]] <- unique(c(0,min(1,maOrders[i]):maOrders[i]) * lags[i]);
        }

        # Produce ARI polynomials
        ariLengths <- unlist(lapply(ariValues,length));
        ariPolynomial <- array(0,ariLengths);
        for(i in 1:length(ariValues)){
            if(i==1){
                ariPolynomial <- ariPolynomial + array(ariValues[[i]], ariLengths);
            }
            else{
                ariPolynomial <- ariPolynomial + array(rep(ariValues[[i]],each=prod(ariLengths[1:(i-1)])),
                                                       ariLengths);
            }
        }

        # Produce MA polynomials
        maLengths <- unlist(lapply(maValues,length));
        maPolynomial <- array(0,maLengths);
        for(i in 1:length(maValues)){
            if(i==1){
                maPolynomial <- maPolynomial + array(maValues[[i]], maLengths);
            }
            else{
                maPolynomial <- maPolynomial + array(rep(maValues[[i]],each=prod(maLengths[1:(i-1)])),
                                                     maLengths);
            }
        }

        # What are the non-zero ARI and MA polynomials?
        ### What are their positions in transition matrix?
        nonZeroARI <- unique(matrix(c(ariPolynomial)[-1],ncol=1));
        nonZeroMA <- unique(matrix(c(maPolynomial)[-1],ncol=1));
        # Lags for the ARIMA components
        lagsModelARIMA <- matrix(sort(unique(c(nonZeroARI,nonZeroMA))),ncol=1);
        nonZeroARI <- cbind(nonZeroARI+1,which(lagsModelARIMA %in% nonZeroARI));
        nonZeroMA <- cbind(nonZeroMA+1,which(lagsModelARIMA %in% nonZeroMA));

        # Number of components
        componentsNumberARIMA <- length(lagsModelARIMA);
        # Their names
        componentsNamesARIMA <- paste0("ARIMAState",c(1:componentsNumberARIMA));

        # If all orders are zero, drop ARIMA part
        if(all(c(arOrders,iOrders,maOrders)==0)){
            arOrders <- NULL;
            iOrders <- NULL;
            maOrders <- NULL;
            arimaModel <- FALSE;
            arRequired <- arEstimate <- FALSE;
            iRequired <- FALSE;
            maRequired <- maEstimate <- FALSE;
            lagsModelARIMA <- initialArimaNumber <- 0;
            componentsNumberARIMA <- 0;
            componentsNamesARIMA <- NULL;
            nonZeroARI <- NULL;
            nonZeroMA <- NULL;
        }
        else{
            # Number of initials needed. This is based on the longest one. The others are just its transformations
            initialArimaNumber <- max(lagsModelARIMA);
        }
    }
    else{
        arOrders <- NULL;
        iOrders <- NULL;
        maOrders <- NULL;
        arimaModel <- FALSE;
        arRequired <- arEstimate <- FALSE;
        iRequired <- FALSE;
        maRequired <- maEstimate <- FALSE;
        lagsModelARIMA <- initialArimaNumber <- 0;
        componentsNumberARIMA <- 0;
        componentsNamesARIMA <- NULL;
        nonZeroARI <- NULL;
        nonZeroMA <- NULL;
        select <- FALSE;
    }

    if(etsModel){
        modelIsSeasonal <- Stype!="N";
    }
    else{
        modelIsSeasonal <- FALSE;
    }

    # Lags of the model used inside the functions
    lagsModel <- matrix(lags,ncol=1);

    # If we have a trend add one more lag
    if(modelIsTrendy){
        lagsModel <- rbind(1,lagsModel);
    }
    # If we don't have seasonality, remove seasonal lag
    if(!modelIsSeasonal && any(lagsModel>1)){
        lagsModel <- lagsModel[lagsModel==1,,drop=FALSE];
    }

    # If this is non-seasonal model and there are no seasonal ARIMA lags, trim the original lags
    if(!modelIsSeasonal && all(c(arOrders[lags>1],iOrders[lags>1],maOrders[lags>1])==0) && any(lags>1)){
        arOrders <- arOrders[lags==1];
        iOrders <- iOrders[lags==1];
        maOrders <- maOrders[lags==1];
        lags <- lags[lags==1];
    }

    # Lags of the model
    lagsModelSeasonal <- lagsModel[lagsModel>1];
    lagsModelMax <- max(lagsModel);
    lagsLength <- length(lagsModel);

    if(etsModel){
        #### Check the seasonal model vs lags ####
        if(all(Stype!=c("Z","X","Y","N","A","M","C"))){
            warning(paste0("Wrong seasonality type: ",Stype,". Should be 'Z', 'X', 'Y', 'C', 'N', 'A' or 'M'. ",
                           "Setting to 'Z'."),call.=FALSE);
            if(lagsModelMax==1){
                Stype <- "N";
                modelIsSeasonal <- FALSE;
            }
            else{
                Stype <- "Z";
                modelDo <- "select";
            }
        }
        if(all(modelIsSeasonal,lagsModelMax==1)){
            if(all(Stype!=c("Z","X","Y"))){
                warning(paste0("Cannot build the seasonal model on data with the unity lags.\n",
                               "Switching to non-seasonal model: ETS(",substr(model,1,nchar(model)-1),"N)"),
                        call.=FALSE);
            }
            Stype <- "N";
            modelIsSeasonal <- FALSE;
            substr(model,nchar(model),nchar(model)) <- "N";
        }

        # Check the pool of models to combine if it was decided that the data is not seasonal
        if(!modelIsSeasonal && !is.null(modelsPool)){
            modelsPool <- modelsPool[substr(modelsPool,nchar(modelsPool),nchar(modelsPool))=="N"];
        }

        # Check the type of seasonal
        if(Stype!="N"){
            componentsNamesETS <- c(componentsNamesETS,"seasonal");
            componentsNumberETS[] <- componentsNumberETS+1;
            componentsNumberETSSeasonal <- 1;
        }
        else{
            componentsNumberETSSeasonal <- 0;
        }

        # Check, whether the number of lags and the number of components are the same
        if(lagsLength>componentsNumberETS){
            if(Stype!="N"){
                componentsNamesETS <- c(componentsNamesETS[-length(componentsNamesETS)],
                                        paste0("seasonal",c(1:(lagsLength-componentsNumberETS-1))));
                componentsNumberETSSeasonal[] <- lagsLength-componentsNumberETS+1;
                componentsNumberETS[] <- lagsLength;
            }
            else{
                lagsModel <- lagsModel[1:componentsNumberETS,,drop=FALSE];
                lagsModelMax <- max(lagsModel);
                lagsLength <- length(lagsModel);
            }
        }
        else if(lagsLength<componentsNumberETS){
            stop("The number of components of the model is smaller than the number of provided lags", call.=FALSE);
        }
    }
    else{
        componentsNumberETSSeasonal <- 0;
    }

    if(!fast){
        #### Distribution selected ####
        distribution <- match.arg(distribution[1], c("default","dnorm","dlaplace","dalaplace","ds","dgnorm",
                                                      "dlnorm","dinvgauss","dgamma","plogis"));
    }

    if(select){
        return(list(select=TRUE, distribution=distribution));
    }

    #### Loss function type ####
    if(is.function(loss)){
        lossFunction <- loss;
        loss <- "custom";
        multisteps <- FALSE;
    }
    else{
        loss <- match.arg(loss[1],c("likelihood","MSE","MAE","HAM","LASSO","RIDGE",
                                    "MSEh","TMSE","GTMSE","MSCE",
                                    "MAEh","TMAE","GTMAE","MACE",
                                    "HAMh","THAM","GTHAM","CHAM","GPL",
                                    "aMSEh","aTMSE","aGTMSE","aMSCE","aGPL",
                                    # "custom" is set upstream by om() / omg()
                                    # when the user passes a callable — accept
                                    # it as a valid string so the rest of the
                                    # checker pipeline doesn't reject the OM
                                    # custom-loss flow.
                                    "custom"));

        if(any(loss==c("MSEh","TMSE","GTMSE","MSCE","MAEh","TMAE","GTMAE","MACE",
                       "HAMh","THAM","GTHAM","CHAM","GPL",
                       "aMSEh","aTMSE","aGTMSE","aMSCE","aGPL"))){
            if(!is.null(h) && h>0){
                multisteps <- TRUE;
            }
            else{
                stop("The horizon \"h\" needs to be specified and be positive in order for the multistep loss to work.",
                     call.=FALSE);
                multisteps <- FALSE;
            }
        }
        else{
            multisteps <- FALSE;
        }
        lossFunction <- NULL;
    }

    #### Explanatory variables: xregModel and regressors ####
    regressors <- match.arg(regressors,c("use","select","adapt"));
    xregModel <- !is.null(xregData);

    #### Persistence provided ####
    # Vectors for persistence of different components
    persistenceLevel <- NULL;
    persistenceTrend <- NULL;
    persistenceSeasonal <- NULL;
    persistenceXreg <- NULL;
    # InitialEstimate vectors, defining what needs to be estimated
    persistenceEstimate <- persistenceLevelEstimate <- persistenceTrendEstimate <-
        persistenceXregEstimate <- TRUE;
    # persistence of seasonal is a vector, not a scalar, because we can have several lags
    persistenceSeasonalEstimate <- rep(TRUE,componentsNumberETSSeasonal);
    if(!is.null(persistence)){
        if(all(modelDo!=c("estimate","use"))){
            warning(paste0("Predefined persistence can only be used with ",
                           "preselected ETS model.\n",
                           "Changing to estimation of persistence values."),call.=FALSE);
            persistence <- NULL;
            persistenceEstimate <- TRUE;
        }
        else{
            # If it is a list
            if(is.list(persistence)){
                # If this is a named list, then extract stuff using names
                if(!is.null(names(persistence))){
                    if(!is.null(persistence$level)){
                        persistenceLevel <- persistence$level;
                    }
                    else if(!is.null(persistence$alpha)){
                        persistenceLevel <- persistence$alpha;
                    }
                    if(!is.null(persistence$trend)){
                        persistenceTrend <- persistence$trend;
                    }
                    else if(!is.null(persistence$beta)){
                        persistenceTrend <- persistence$beta;
                    }
                    if(!is.null(persistence$seasonal)){
                        persistenceSeasonal <- persistence$seasonal;
                    }
                    else if(!is.null(persistence$gamma)){
                        persistenceSeasonal <- persistence$gamma;
                    }
                    if(!is.null(persistence$xreg)){
                        persistenceXreg <- persistence$xreg;
                    }
                    else if(!is.null(persistence$delta)){
                        persistenceXreg <- persistence$delta;
                    }
                }
                else{
                    if(!is.null(persistence[[1]])){
                        persistenceLevel <- persistence[[1]];
                    }
                    if(!is.null(persistence[[2]])){
                        persistenceTrend <- persistence[[2]];
                    }
                    if(!is.null(persistence[[3]])){
                        persistenceSeasonal <- persistence[[3]];
                    }
                    if(!is.null(persistence[[4]])){
                        persistenceXreg <- persistence[[4]];
                    }
                }
                # Define estimate variables
                if(!is.null(persistenceLevel)){
                    persistenceLevelEstimate[] <- FALSE;
                    parametersNumber[2,1] <- parametersNumber[2,1] + 1;
                }
                if(!is.null(persistenceTrend)){
                    persistenceTrendEstimate[] <- FALSE;
                    parametersNumber[2,1] <- parametersNumber[2,1] + 1;
                }
                if(!is.null(persistenceSeasonal)){
                    if(is.list(persistenceSeasonal)){
                        persistenceSeasonalEstimate[] <- length(persistenceSeasonal)==length(lagsModelSeasonal);
                    }
                    else{
                        persistenceSeasonalEstimate[] <- FALSE;
                    }
                    parametersNumber[2,1] <- parametersNumber[2,1] + length(unlist(persistenceSeasonal));
                }
                if(!is.null(persistenceXreg)){
                    persistenceXregEstimate[] <- FALSE;
                    parametersNumber[2,1] <- parametersNumber[2,1] + length(persistenceXreg);
                }
            }
            else if(is.numeric(persistence)){
                # If it is smaller... We don't know the length of xreg yet at this stage
                if(length(persistence)<lagsLength){
                    warning(paste0("Length of persistence vector is wrong! ",
                                   "Changing to estimation of persistence vector values."),
                            call.=FALSE);
                    persistence <- NULL;
                    persistenceEstimate <- TRUE;
                }
                else{
                    # If there ARIMA elements, remove them
                    if(any(substr(names(persistence),1,3)=="psi")){
                        persistence <- persistence[substr(names(persistence),1,3)!="psi"];
                    }
                    j <- 0;
                    if(etsModel){
                        j <- j+1;
                        persistenceLevel <- as.vector(persistence)[1];
                        names(persistenceLevel) <- "alpha";
                        if(modelIsTrendy && length(persistence)>j){
                            j <- j+1;
                            persistenceTrend <- as.vector(persistence)[j];
                            names(persistenceTrend) <- "beta";
                        }
                        if(Stype!="N" && length(persistence)>j){
                            persistenceSeasonal <- as.vector(persistence)[j+c(1:length(lagsModelSeasonal))];
                            names(persistenceSeasonal) <- paste0("gamma",c(1:length(persistenceSeasonal)));
                            j <- j+length(lagsModelSeasonal);
                        }
                    }
                    if(xregModel && length(persistence)>j){
                        if(j>0){
                            persistenceXreg <- persistence[-c(1:j)];
                        }
                        else{
                            persistenceXreg <- persistence;
                        }
                        # If there are names, make sure that only deltas are used
                        if(!is.null(names(persistenceXreg))){
                            persistenceXreg <- persistenceXreg[substr(names(persistenceXreg),1,5)=="delta"];
                        }
                        else{
                            names(persistenceXreg) <- paste0("delta",c(1:length(persistenceXreg)));
                        }
                    }

                    persistenceEstimate[] <- persistenceLevelEstimate[] <- persistenceTrendEstimate[] <-
                        persistenceXregEstimate[] <- persistenceSeasonalEstimate[] <- FALSE;
                    parametersNumber[2,1] <- parametersNumber[2,1] + length(persistence);
                    # bounds <- "none";
                }
            }
            else{
                warning(paste0("Persistence is not a numeric vector!\n",
                               "Changing to estimation of persistence vector values."),call.=FALSE);
                persistence <- NULL;
                persistenceEstimate <- TRUE;
            }
        }
    }
    else{
        persistenceEstimate <- TRUE;
    }

   # Make sure that only important elements are estimated.
    if(!etsModel){
        persistenceLevelEstimate[] <- FALSE;
    }
    if(!etsModel || Ttype=="N"){
        persistenceTrendEstimate[] <- FALSE;
        persistenceTrend <- NULL;
    }
    if(!etsModel || Stype=="N"){
        persistenceSeasonalEstimate[] <- FALSE;
        persistenceSeasonal <- NULL;
    }
    if(!xregModel){
        persistenceXregEstimate[] <- FALSE;
        persistenceXreg <- NULL;
    }

    #### Phi ####
    if(etsModel){
        if(!is.null(phi)){
            if(all(modelDo!=c("estimate","use"))){
                warning(paste0("Predefined phi can only be used with preselected ETS model.\n",
                               "Changing to estimation."),call.=FALSE);
                phi <- 0.95;
                phiEstimate <- TRUE;
            }
            else{
                if(!is.numeric(phi) & (damped)){
                    warning(paste0("Provided value of phi is meaningless. phi will be estimated."),
                            call.=FALSE);
                    phi <- 0.95;
                    phiEstimate <- TRUE;
                }
                else if(is.numeric(phi) & (phi<0 | phi>2)){
                    warning(paste0("Damping parameter should lie in (0, 2) region. ",
                                   "Changing to the estimation of phi."),call.=FALSE);
                    phi[] <- 0.95;
                    phiEstimate <- TRUE;
                }
                else{
                    phiEstimate <- FALSE;
                    if(damped){
                        parametersNumber[2,1] <- parametersNumber[2,1] + 1;
                    }
                }
            }
        }
        else{
            if(damped){
                phiEstimate <- TRUE;
                phi <- 0.95;
            }
            else{
                phiEstimate <- FALSE;
                phi <- 1;
            }
        }
    }
    else{
        phi <- 1;
        phiEstimate <- FALSE;
    }


    #### Lags for ARIMA ####
    if(arimaModel){
        lagsModelAll <- rbind(lagsModel,lagsModelARIMA);
        lagsModelMax <- max(lagsModel);
    }
    else{
        lagsModelAll <- lagsModel;
    }

    #### This needs to be amended after developing the first prototype! ####
    # If we have the zoo class and weird lags, amend profiles
    # Weird lags (dst and fractional): 24, 24*2 (half hour), 52, 24*4 (15 minutes), 7*24, 7*48, 365, 24*52, 24*365
    # if(any(yClasses=="zoo") && any(lags %in% c(24, 48, 52, 96, 168, 336, 365, 1248, 8760))){
        # For hourly, half-hourly and quarter hour data, just amend the profiles for DST.
        # For daily, repeat profile of 28th on 29th February.
        # For weekly, repeat the last week, when we have 53 instead of 52.
    # }

    #### Occurrence variable ####
    if(is.occurrence(occurrence)){
        omModel <- occurrence;
        occurrence <- omModel$occurrence;
        if(occurrence=="provided"){
            occurrenceModelProvided <- FALSE;
        }
        else{
            occurrenceModelProvided <- TRUE;
        }
        pFitted <- matrix(fitted(omModel), obsInSample, 1);
    }
    else{
        occurrenceModelProvided <- FALSE;
        omModel <- NULL;
        pFitted <- matrix(1, obsInSample, 1);
    }
    pForecast <- rep(NA,h);

    # If it is logical, convert to numeric
    if(is.logical(occurrence)){
        occurrence <- occurrence*1;
    }
    if(is.numeric(occurrence)){
        # If it is data, then it should correspond to the in-sample.
        if(all(occurrence==1)){
            occurrence <- "none";
        }
        else{
            if(any(occurrence<0,occurrence>1)){
                warning(paste0("Parameter 'occurrence' should contain values between zero and one.\n",
                               "Converting to appropriate vector."),call.=FALSE);
                occurrence[] <- (occurrence!=0)*1;
            }

            # "provided", meaning that we have been provided the values of p
            pFitted[] <- occurrence[1:obsInSample];
            # Create forecasted values for occurrence
            if(h>0){
                if(length(occurrence)>obsInSample){
                    pForecast <- occurrence[-c(1:obsInSample)];
                }
                else{
                    pForecast <- rep(tail(occurrence,1),h);
                }
                if(length(pForecast)>h){
                    pForecast <- pForecast[1:h];
                }
                else if(length(pForecast)<h){
                    pForecast <- c(pForecast,rep(tail(pForecast,1),h-length(pForecast)));
                }
            }
            else{
                pForecast <- NA;
            }
            occurrence <- "provided";
            omModel <- list(fitted=pFitted,forecast=pForecast,occurrence="provided");
        }
    }
    else if(is.occurrence(occurrence)){
        omModel <- occurrence;
        occurrence <- omModel$occurrence;
    }

    occurrence <- match.arg(occurrence[1],c("none","auto","fixed","general","odds-ratio",
                                            "inverse-odds-ratio","direct","provided"));

    otLogical <- yInSample!=0;

    # If the data is not occurrence, let's assume that the parameter was switched unintentionally.
    if(all(otLogical) & all(occurrence!=c("none","provided"))){
        occurrence <- "none";
        occurrenceModelProvided <- FALSE;
    }

    # If there were NAs and the occurrence was not specified, do something with it
    # In all the other cases, NAs will be sorted out by the model
    if(any(yNAValues) && (occurrence=="none")){
        otLogical <- (!yNAValues)[1:obsInSample];
        occurrence[] <- "provided";
        pFitted[] <- otLogical*1;
        pForecast[] <- 1;
        occurrenceModel <- FALSE;
        omModel <- structure(list(y=matrix((otLogical)*1,ncol=1),fitted=pFitted,forecast=pForecast,
                                   occurrence="provided"),class="occurrence");
    }
    else{
        if(occurrence=="none"){
            occurrenceModel <- FALSE;
            otLogical <- rep(TRUE,obsInSample);
        }
        else if(occurrence=="provided"){
            occurrenceModel <- TRUE;
            omModel$y <- matrix(otLogical*1,ncol=1);
        }
        else{
            occurrenceModel <- TRUE;
        }

        # Include NAs in the zeroes, so that we avoid weird cases
        if(any(yNAValues)){
            otLogical <- !(!otLogical | yNAValues[1:obsInSample]);
        }
    }

    if(any(yClasses=="ts")){
        ot <- ts(matrix(otLogical*1,ncol=1), start=yStart, frequency=yFrequency);
    }
    else{
        ot <- ts(matrix(otLogical*1,ncol=1), start=c(0,0), frequency=lagsModelMax);
    }
    obsNonzero <- sum(ot);
    obsZero <- obsInSample - obsNonzero;

    # If occurrence is provided, use it as is
    if(occurrence=="provided"){
        ot[] <- pFitted;
    }

    # Check if multiplicative models can be fitted
    allowMultiplicative <- !((any(yInSample<=0) && !occurrenceModel) || (occurrenceModel && any(yInSample<0)));

    if(etsModel){
        # Clean the pool of models if only additive are allowed
        if(!allowMultiplicative && !is.null(modelsPool)){
            modelsPoolMultiplicative <- ((substr(modelsPool,1,1)=="M") |
                                             substr(modelsPool,2,2)=="M" |
                                             substr(modelsPool,nchar(modelsPool),nchar(modelsPool))=="M");
            if(any(modelsPoolMultiplicative)){
                modelsPool <- modelsPool[!modelsPoolMultiplicative];

                # This is needed, because PPP and FFF use pool, not Branch and bound
                if(!any(c(any(unlist(strsplit(model,""))=="P"),any(unlist(strsplit(model,""))=="F")))){
                    warning("Only additive models are allowed for your data. Amending the pool.",
                            call.=FALSE);
                }
            }
        }
        if((any(model==c("PPP","FFF","YYY")) || any(unlist(strsplit(model,""))=="Z")) && !allowMultiplicative){
            model <- "XXX";
            Etype[] <- "A";
            Ttype[] <- switch(Ttype,"Y"=,"Z"=,"P"=,"F"="X",Ttype);
            Stype[] <- switch(Stype,"Y"=,"Z"=,"P"=,"F"="X",Stype);
            modelsPool <- NULL;
            warning("Only additive models are allowed for your data. Changing the selection mechanism.",
                    call.=FALSE);
        }
        else if(!allowMultiplicative && any(c(Etype,Ttype,Stype)=="M")){
            warning("Your data contains non-positive values, so the ETS(",model,") might break down.",
                    call.=FALSE);
        }
        else if(any(model==c("PPP","FFF")) && allowMultiplicative){
            model <- "ZZZ";
        }
    }

    #### Initial values ####
    # Vectors for initials of different components
    initialLevel <- NULL;
    initialTrend <- NULL;
    initialSeasonal <- NULL;
    initialArima <- NULL;
    initialXreg <- NULL;
    # InitialEstimate vectors, defining what needs to be estimated
    # NOTE: that initial==c("optimal","complete") means initialEstimate==TRUE!
    initialEstimate <- initialLevelEstimate <- initialTrendEstimate <-
        initialArimaEstimate <- initialXregEstimate <- TRUE;
    # initials of seasonal is a vector, not a scalar, because we can have several lags
    initialSeasonalEstimate <- rep(TRUE,componentsNumberETSSeasonal);

    # This is an initialisation of the variable
    initialType <- "backcasting"
    # initial type can be: "o" - optimal, "b" - backcasting, "p" - provided.
    if(any(is.character(initial))){
        initialType[] <- match.arg(initial, c("backcasting","optimal","two-stage","complete","gradient"));
    }
    else if(is.null(initial)){
        if(!silent){
            message("Initial value is not selected. Switching to backcasting.");
        }
        initialType[] <- "backcasting";
    }
    else if(!is.null(initial)){
        if(all(modelDo!=c("estimate","use"))){
            warning(paste0("Predefined initials vector can only be used with preselected ETS model.\n",
                           "Changing to estimation of initials."),call.=FALSE);
            initialType[] <- "optimal";
            initialEstimate[] <- initialLevelEstimate[] <- initialTrendEstimate[] <-
                initialSeasonalEstimate[] <- initialArimaEstimate[] <- initialXregEstimate[] <- TRUE;
        }
        else{
            # If the list is provided, then check what this is.
            # This should be: level, trend, seasonal[[1]], seasonal[[2]], ..., ARIMA, xreg
            if(is.list(initial)){
                # If this is a named list, then extract stuff using names
                if(!is.null(names(initial))){
                    if(!is.null(initial$level)){
                        initialLevel <- initial$level;
                    }
                    if(!is.null(initial$trend)){
                        initialTrend <- initial$trend;
                    }
                    if(!is.null(initial$seasonal)){
                        initialSeasonal <- initial$seasonal;
                    }
                    if(!is.null(initial$arima)){
                        initialArima <- initial$arima;
                    }
                    if(!is.null(initial$xreg)){
                        initialXreg <- initial$xreg;
                    }
                }
                else{
                    if(!is.null(initial[[1]])){
                        initialLevel <- initial[[1]];
                    }
                    if(!is.null(initial[[2]])){
                        initialTrend <- initial[[2]];
                    }
                    if(!is.null(initial[[3]])){
                        initialSeasonal <- initial[[3]];
                    }
                    if(!is.null(initial[[4]])){
                        initialArima <- initial[[4]];
                    }
                    if(!is.null(initial[[5]])){
                        initialXreg <- initial[[5]];
                    }
                }
            }
            else{
                if(!is.numeric(initial)){
                    warning(paste0("Initial vector is not numeric!\n",
                                   "Values of initial vector will be backcasted."),call.=FALSE);
                    initialType[] <- "backcasting";
                }
                else{
                    # If this is a vector, then it should contain values in the order:
                    # level, trend, seasonal1, seasonal2, ..., ARIMA, xreg
                    # if(length(initial)<(sum(lagsModelAll))){
                    #     warning(paste0("The vector of initials contains only values for several components. ",
                    #                    "We will use what we can."),call.=FALSE);
                    # }
                    # else{
                        j <- 0;
                        if(etsModel){
                            initialLevel <- initial[1];
                            initialLevelEstimate[] <- FALSE;
                            if(modelIsTrendy){
                                j <- 2;
                                # If there is something in the vector, use it
                                if(all(!is.na(initial[j]))){
                                    initialTrend <- initial[j];
                                    initialTrendEstimate[] <- FALSE;
                                }
                            }
                            if(Stype!="N"){
                                # If there is something in the vector, use it
                                if(length(initial[-c(1:j)])>0){
                                    initialSeasonal <- vector("list",componentsNumberETSSeasonal);
                                    m <- 0;
                                    for(i in 1:componentsNumberETSSeasonal){
                                        if(all(!is.na(initial[j+m+1:lagsModelSeasonal[i]]))){
                                            initialSeasonal[[i]] <- initial[j+m+1:lagsModelSeasonal[i]];
                                            m <- m + lagsModelSeasonal[i];
                                        }
                                        else{
                                            break;
                                        }
                                    }
                                    j <- j+m;
                                    initialSeasonalEstimate[] <- FALSE;
                                }
                            }
                        }
                        if(arimaModel){
                            # If there is something else left, this must be ARIMA
                            if(all(!is.na(initial[j+c(1:initialArimaNumber)]))){
                                initialArima <- initial[j+c(1:initialArimaNumber)];
                                j <- j+max(lagsModelARIMA);
                                initialArimaEstimate[] <- FALSE;
                            }
                        }
                        if(xregModel){
                            # Something else? xreg for sure!
                            if(length(initial[-c(1:j)])>0){
                                initialXreg <- initial[-c(1:j)];
                                initialXregEstimate[] <- FALSE
                            }
                        }
                        parametersNumber[2,1] <- parametersNumber[2,1] + j;
                    }
                # }
            }
        }
    }

    #### Check the provided initials and define initialEstimate variables ####
    if(etsModel){
        # Level
        if(!is.null(initialLevel)){
            if(length(initialLevel)>1){
                warning("Initial level contains more than one value! Using the first one.",
                        call.=FALSE);
                initialLevel <- initialLevel[1];
            }
            initialLevelEstimate[] <- FALSE;
            parametersNumber[2,1] <- parametersNumber[2,1] + 1;
        }
        # Trend
        if(!is.null(initialTrend)){
            if(length(initialTrend)>1){
                warning("Initial trend contains more than one value! Using the first one.",
                        call.=FALSE);
                initialTrend <- initialTrend[1];
            }
            initialTrendEstimate[] <- FALSE;
            parametersNumber[2,1] <- parametersNumber[2,1] + 1;
        }
        # Seasonal
        if(!is.null(initialSeasonal)){
            # The list means several seasonal lags
            if(is.list(initialSeasonal)){
                # Is the number of seasonal initials correct? If it is bigger, then remove redundant
                if(length(initialSeasonal)>componentsNumberETSSeasonal){
                    warning("Initial seasonals contained more elements than needed! Removing redundant ones.",
                            call.=FALSE);
                    initialSeasonal <- initialSeasonal[1:componentsNumberETSSeasonal];
                }
                # Is the number of initials in each season correct? Use the correct ones only
                if(any(!(sapply(initialSeasonal,length) %in% lagsModelSeasonal))){
                    warning(paste0("Some of initial seasonals have a wrong length, ",
                                   "not corresponding to the provided lags. I will estimate them."),
                            call.=FALSE);
                    initialSeasonalToUse <- sapply(initialSeasonal,length) %in% lagsModelSeasonal;
                    initialSeasonal <- initialSeasonal[initialSeasonalToUse];
                }
                initialSeasonalEstimate[] <- !(lagsModelSeasonal %in% sapply(initialSeasonal,length));
                # If there are some gaps in what to estimate, reform initialSeason to make sense in the future creator function
                if(!all(initialSeasonalEstimate) && !all(!initialSeasonalEstimate)){
                    initialSeasonalCorrect <- vector("list",componentsNumberETSSeasonal);
                    initialSeasonalCorrect[which(!initialSeasonalEstimate)] <- initialSeasonal;
                    initialSeasonal <- initialSeasonalCorrect;
                }
            }
            # The vector implies only one seasonal
            else{
                if(all(length(initialSeasonal)!=lagsModelSeasonal)){
                    warning(paste0("Wrong length of seasonal initial: ",length(initialSeasonal),
                                   "Instead of ",lagsModelSeasonal,". Switching to estimation."),
                            call.=FALSE)
                    initialSeasonalEstimate[] <- TRUE;
                }
                else{
                    initialSeasonalEstimate[] <- FALSE;
                }
                # Create a list from the vector for consistency purposes
                initialSeasonal <- list(initialSeasonal);
            }
            parametersNumber[2,1] <- parametersNumber[2,1] + length(unlist(initialSeasonal));
        }
    }
    else{
        initialLevel <- initialTrend <- initialSeasonal <- NULL;
        initialLevelEstimate <- initialTrendEstimate <- initialSeasonalEstimate <- FALSE

        # If ETS is switched off, set error to whatever, based on the used distribution
        Etype[] <- switch(distribution,
                          "dinvgauss"=,"dlnorm"=,"dllaplace"=,"dls"=,"dlgnorm"=,"dgamma"="M",
                          "A");
    }

    # ARIMA
    if(!is.null(initialArima)){
        if(length(initialArima)!=initialArimaNumber){
            warning(paste0("The length of ARIMA initials is ",length(initialArima),
                           " instead of ",initialArimaNumber,". Estimating initials instead!"),
                    call.=FALSE);
            initialArimaEstimate[] <- TRUE;
        }
        else{
            initialArimaEstimate[] <- FALSE;
            parametersNumber[2,1] <- parametersNumber[2,1] + length(initialArima);
        }
    }
    # xreg
    if(!is.null(initialXreg)){
        initialXregEstimate[] <- FALSE;
    }

    #### Check ARIMA parameters, if they are provided ####
    if(arimaModel){
        # Check the provided parameters for AR and MA
        if(!is.null(arma)){
            arEstimate <- arRequired;
            maEstimate <- maRequired;
            # If this is a proper list, extract parameters separately
            if(is.list(arma)){
                armaParameters <- vector("numeric",sum(sapply(arma,length)));
                j <- arIndex <- maIndex <- 0;
                # The named list (proper thing)
                if(!is.null(names(arma))){
                    # Check if the length of the provided parameters is correct
                    if(arRequired && !is.null(arma$ar) && length(arma$ar)!=sum(arOrders)){
                        warning(paste0("The number of provided AR parameters is ",length(arma$ar),
                                       "while I need ",sum(arOrders),". ",
                                       "Switching to estimation."),call.=FALSE);
                        arEstimate[] <- TRUE;
                    }
                    if(maRequired && !is.null(arma$ma) && length(arma$ma)!=sum(maOrders)){
                        warning(paste0("The number of provided MA parameters is ",length(arma$ma),
                                       "while I need ",sum(maOrders),". ",
                                       "Switching to estimation."),call.=FALSE);
                        maEstimate[] <- TRUE;
                    }
                    for(i in 1:length(lags)){
                        if(arRequired && !is.null(arma$ar) && arOrders[i]>0){
                            armaParameters[j+c(1:arOrders[i])] <- arma$ar[arIndex+c(1:arOrders[i])];
                            names(armaParameters)[j+c(1:arOrders[i])] <- paste0("phi",1:arOrders[i],"[",lags[i],"]");
                            j[] <- j+arOrders[i];
                            arIndex[] <- arIndex+arOrders[i];
                            arEstimate[] <- FALSE;
                        }
                        if(maRequired && !is.null(arma$ma) && maOrders[i]>0){
                            armaParameters[j+c(1:maOrders[i])] <- arma$ma[maIndex+c(1:maOrders[i])];
                            names(armaParameters)[j+c(1:maOrders[i])] <- paste0("theta",1:maOrders[i],"[",lags[i],"]");
                            j[] <- j+maOrders[i];
                            maIndex[] <- maIndex+maOrders[i];
                            maEstimate[] <- FALSE;
                        }
                    }
                }
                # Just a list. Assume that the first element is ar, and the second is ma
                else{
                    for(i in 1:length(lags)){
                        k <- 1;
                        if(arRequired && !is.null(arma[[1]]) && arOrders[i]>0){
                            armaParameters[j+c(1:arOrders[i])] <- arma[[1]][arIndex+c(1:arOrders[i])];
                            names(armaParameters)[j+c(1:arOrders[i])] <- paste0("phi",1:arOrders[i],"[",lags[i],"]");
                            j[] <- j+arOrders[i];
                            arIndex[] <- arIndex+arOrders[i];
                            arEstimate[] <- FALSE;
                            k[] <- 2;
                        }
                        if(maRequired && !is.null(arma[[k]]) && maOrders[i]>0){
                            armaParameters[j+c(1:maOrders[i])] <- arma[[k]][maIndex+c(1:maOrders[i])];
                            names(armaParameters)[j+c(1:maOrders[i])] <- paste0("theta",1:maOrders[i],"[",lags[i],"]");
                            j[] <- j+maOrders[i];
                            maIndex[] <- maIndex+maOrders[i];
                            maEstimate[] <- FALSE;
                        }
                    }
                    # Check if the length of the provided parameters is correct
                    if(length(armaParameters)!=sum(arOrders)+sum(maOrders)){
                        warning(paste0("The number of provided ARMA parameters is ",length(armaParameters),
                                       "while I need ",sum(arOrders)+sum(maOrders),". ",
                                       "Switching to estimation."),call.=FALSE);
                        maEstimate <- arEstimate[] <- TRUE;
                        armaParameters <- NULL;
                    }
                }
            }
            else if(is.vector(arma)){
                # Check the length of the vector
                if(length(arma)!=sum(arOrders)+sum(maOrders)){
                    warning(paste0("The number of provided ARMA parameters is ",length(arma),
                                   "while I need ",sum(arOrders)+sum(maOrders),". ",
                                   "Switching to estimation."),call.=FALSE);
                    maEstimate <- arEstimate[] <- TRUE;
                    armaParameters <- NULL;
                }
                else{
                    armaParameters <- arma;
                    j <- 0;
                    for(i in 1:length(lags)){
                        if(arOrders[i]>0){
                            names(armaParameters)[j+1:arOrders[i]] <- paste0("phi",1:arOrders[i],"[",lags[i],"]");
                            j <- j + arOrders[i];
                        }
                        if(maOrders[i]>0){
                            names(armaParameters)[j+1:maOrders[i]] <- paste0("theta",1:maOrders[i],"[",lags[i],"]");
                            j <- j + maOrders[i];
                        }
                    }
                    maEstimate <- arEstimate[] <- FALSE;
                }
            }
            parametersNumber[2,1] <- parametersNumber[2,1] + length(armaParameters);
        }
        else{
            arEstimate <- arRequired;
            maEstimate <- maRequired;
            armaParameters <- NULL;
        }
    }
    else{
        armaParameters <- NULL;
    }

    # Make armaParameters a zero vector. Needed for C++ code to work
    if(is.null(armaParameters)){
        armaParameters <- matrix(0,0,0);
    }

    #### xreg preparation ####
    # Check the regressors
    if(!xregModel){
        regressors[] <- "use";
        formulaToUse <- NULL;
    }
    else{
        if(regressors=="select"){
            # If this has not happened by chance, then switch to optimisation
            if(!is.null(initialXreg) && any(initialType==c("backcasting","optimal","two-stage"))){
                warning("Variables selection does not work with the provided initials for explantory variables. I will drop them.",
                        call.=FALSE);
                initialXreg <- NULL;
                initialXregEstimate <- TRUE;
            }
            if(!is.null(persistenceXreg) && any(persistenceXreg!=0)){
                warning(paste0("I cannot do variables selection with the provided smoothing parameters ",
                               "for explantory variables. I will estimate them instead."),
                        call.=FALSE);
                persistenceXreg <- NULL;
            }
        }
    }

    # Use alm() in order to fit the preliminary model for xreg
    if(xregModel){
        # List of initials. The first element is additive error, the second is the multiplicative one
        xregModelInitials <- vector("list",2);

        if(regressors!="select"){
            #### Initial xreg are not provided ####
            # If the initials are not provided, estimate them using ALM.
            if(initialXregEstimate){
                initialXregProvided <- FALSE;
                # The function returns an ALM model
                xregInitialiser <- function(Etype,distribution,formulaToUse,subset,responseName){
                    # Fix the default distribution for ALM
                    if(distribution=="default"){
                        distribution <- switch(Etype,
                                               "A"="dnorm",
                                               "M"="dlnorm");
                    }
                    else if(distribution=="dllaplace"){
                        distribution <- "dlaplace";
                        Etype <- "M";
                    }
                    else if(distribution=="dls"){
                        distribution <- "ds";
                        Etype <- "M";
                    }
                    else if(distribution=="dlgnorm"){
                        distribution <- "dgnorm";
                        Etype <- "M";
                    }
                    # This is needed to see if trend was asked explicitly. If not, we add it to get rid of bias
                    trendIncluded <- any(colnames(xregData)[-1]=="trend");
                    formulaIsAbsent <- is.null(formulaToUse);
                    formulaOriginal <- formulaToUse;
                    # If the formula is not provided, construct one
                    if(formulaIsAbsent){
                        if(Etype=="M" && any(distribution==c("dnorm","dlaplace","ds","dgnorm","dlogis","dt","dalaplace"))){
                            if(trendIncluded){
                                formulaOriginal <- formulaToUse <- as.formula(paste0("log(`",responseName,"`)~."));
                            }
                            else{
                                formulaOriginal <- as.formula(paste0("log(`",responseName,"`)~."));
                                formulaToUse <- as.formula(paste0("log(`",responseName,"`)~.+trend"));
                            }
                        }
                        else{
                            if(trendIncluded){
                                formulaOriginal <- formulaToUse <- as.formula(paste0("`",responseName,"`~."));
                            }
                            else{
                                formulaOriginal <- as.formula(paste0("`",responseName,"`~."));
                                formulaToUse <- as.formula(paste0("`",responseName,"`~.+trend"));
                            }
                        }
                    }
                    else{
                        # If formula only contains ".", then just change it
                        if(length(all.vars(formulaToUse))==2 && all.vars(formulaToUse)[2]=="."){
                            # Take logs if the model requires that
                            if((Etype=="M" && any(distribution==c("dnorm","dlaplace","ds","dgnorm","dlogis","dt","dalaplace")))){
                                if(trendIncluded){
                                    formulaToUse <- as.formula(paste0("log(`",responseName,"`)~."));
                                }
                                else{
                                    formulaToUse <- as.formula(paste0("log(`",responseName,"`)~.+trend"));
                                }
                            }
                            else{
                                if(trendIncluded){
                                    formulaToUse <- as.formula(paste0(responseName,"~."));
                                }
                                else{
                                    formulaToUse <- as.formula(paste0(responseName,"~.+trend"));
                                }
                            }
                        }
                        else{
                            trendIncluded <- any(all.vars(formulaToUse)[-1]=="trend");
                            # If formula contains only one element, or several, but no logs, then change response formula
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
                        }
                    }
                    almModel <- do.call(alm,list(formula=formulaToUse,data=xregData,distribution=distribution,subset=which(subset)));
                    # Remove trend
                    if(!trendIncluded){
                        almModel$coefficients <- almModel$coefficients[names(almModel$coefficients)!="trend"];
                        almModel$data <- almModel$data[,colnames(almModel$data)!="trend",drop=FALSE];
                        if(formulaIsAbsent){
                            almModel$call$formula <- as.formula(paste0("`",responseName,"`~."));
                        }
                        else{
                            almModel$call$formula <- formulaOriginal;
                        }
                        # Reset terms, they are not needed for what comes, but confuse the formula() method
                        almModel$terms <- NULL;
                    }
                    return(almModel);
                }

                # Extract names of variables, fix the formula
                if(!is.null(formulaToUse)){
                    formulaToUse <- as.formula(formulaToUse);
                    responseName <- all.vars(formulaToUse)[1];
                    # If there are spaces in names, give a warning
                    if(any(grepl("[^A-Za-z0-9,;._-]", all.vars(formulaToUse))) ||
                       # If the names only contain numbers
                       any(suppressWarnings(!is.na(as.numeric(all.vars(formulaToUse)))))){
                        warning("The names of your variables contain special characters ",
                                "(such as numbers, spaces, comas, brackets etc). adam() might not work properly. ",
                                "It is recommended to use `make.names()` function to fix the names of variables.",
                                call.=FALSE);
                        formulaToUse <- as.formula(paste0(gsub(paste0("`",all.vars(formulaToUse)[1],"`"),
                                                          make.names(all.vars(formulaToUse)[1]),
                                                          all.vars(formulaToUse)[1]),"~",
                                                     paste0(mapply(gsub, paste0("`",all.vars(formulaToUse)[-1],"`"),
                                                                   make.names(all.vars(formulaToUse)[-1]),
                                                                   labels(terms(formulaToUse))),
                                                            collapse="+")));
                    }
                    formulaProvided <- TRUE;
                }
                else{
                    formulaToUse <- reformulate(make.names(colnames(xregData)[-1]), response=responseName);
                    # Quotes are needed here for the insecure names of variables, such as "1", "2", "3" etc
                    # formulaToUse <- reformulate(setdiff(paste0("`",colnames(xregData),"`"),
                    #                                     paste0("`",responseName,"`")),
                    #                             response=responseName);
                    formulaProvided <- FALSE;
                }

                # Robustify the names of variables
                colnames(xregData) <- make.names(colnames(xregData));
                # This allows to save the original data
                xreg <- xregData;
                obsXreg <- nrow(xregData);

                # Form subset in order to use in-sample only
                subset <- rep(FALSE, obsAll);
                subset[1:obsInSample] <- TRUE;
                # Exclude zeroes if this is an occurrence model (demand-size models need
                # only non-zero demand; for the occurrence-probability ALM we keep all obs)
                if(occurrenceModel){
                    subsetOccurrence <- subset;
                    subset[1:obsInSample][!otLogical] <- FALSE;
                }

                #### Pure regression ####
                #### If this is just a regression ALM
                if((!etsModel && !arimaModel) && regressors!="adapt"){
                    # Return the estimated model based on the provided xreg
                    if(is.null(formulaToUse)){
                        formulaToUse <- reformulate(setdiff(colnames(xregData), responseName), response=responseName);
                        # formulaToUse <- as.formula(paste0("`",responseName,"`~."));
                        formulaProvided <- FALSE;
                    }
                    else{
                        formulaProvided <- TRUE;
                    }
                    if(distribution=="default"){
                        distribution[] <- "dnorm";
                    }

                    # Redefine loss for ALM
                    loss <- switch(loss,
                                   "MSEh"=,"TMSE"=,"GTMSE"=,"MSCE"="MSE",
                                   "MAEh"=,"TMAE"=,"GTMAE"=,"MACE"="MAE",
                                   "HAMh"=,"THAM"=,"GTHAM"=,"CHAM"="HAM",
                                   loss);
                    lossOriginal <- loss;
                    if(loss=="custom"){
                        loss <- lossFunction;
                    }

                    # Either use or select the model via greybox functions
                    # Fisher Information
                    if(is.null(ellipsis$FI)){
                        FI <- FALSE;
                    }
                    else{
                        FI <- ellipsis$FI;
                    }
                    almSubset <- if(occurrenceModel) which(subsetOccurrence) else which(subset);
                    almModel <- do.call("alm", list(formula=formulaToUse, data=xregData,
                                                    distribution=distribution, loss=loss,
                                                    subset=almSubset,
                                                    occurrence=omModel,FI=FI));
                    almModel$call$data <- as.name(yName);
                    return(almModel);
                }

                #### ETSX / ARIMAX ####
                almModel <- NULL;
                if(Etype!="Z"){
                    almModel <- xregInitialiser(Etype,distribution,formulaToUse,subset,responseName);
                    # If Intercept was not included, substitute with zero
                    if(!any(names(almModel$coefficients)=="(Intercept)")){
                        almIntercept <- 0;
                    }
                    else{
                        almIntercept <- almModel$coefficients["(Intercept)"];
                    }
                    if(Etype=="A"){
                        # If this is just a regression, include intercept
                        if(!etsModel && !arimaModel){
                            xregModelInitials[[1]]$initialXreg <- almModel$coefficients;
                        }
                        else{
                            xregModelInitials[[1]]$initialXreg <- almModel$coefficients[-1];
                        }
                        if(is.null(formulaToUse)){
                            xregModelInitials[[1]]$formula <- formulaToUse <- formula(almModel);
                        }
                        xregModelInitials[[1]]$other <- almModel$other;
                        names(xregModelInitials[[1]]$initialXreg) <-
                            make.names(names(xregModelInitials[[1]]$initialXreg));
                    }
                    else{
                        # If this is just a regression, include intercept
                        if(!etsModel && !arimaModel){
                            xregModelInitials[[2]]$initialXreg <- almModel$coefficients;
                        }
                        else{
                            xregModelInitials[[2]]$initialXreg <- almModel$coefficients[-1];
                        }
                        if(is.null(formulaToUse)){
                            xregModelInitials[[2]]$formula <- formulaToUse <- formula(almModel);
                        }
                        xregModelInitials[[2]]$other <- almModel$other;
                        names(xregModelInitials[[2]]$initialXreg) <-
                            make.names(names(xregModelInitials[[2]]$initialXreg));
                    }
                }
                # If we are selecting the appropriate error, produce two models: for "M" and for "A"
                else{
                    # Additive model
                    almModel <- xregInitialiser("A",distribution,formulaToUse,subset,responseName);
                    # If Intercept was not included, substitute with zero
                    if(!any(names(almModel$coefficients)=="(Intercept)")){
                        almIntercept <- 0;
                    }
                    else{
                        almIntercept <- almModel$coefficients["(Intercept)"];
                    }
                    # If this is just a regression, include intercept
                    if(!etsModel && !arimaModel){
                        xregModelInitials[[1]]$initialXreg <- almModel$coefficients;
                    }
                    else{
                        xregModelInitials[[1]]$initialXreg <- almModel$coefficients[-1];
                    }
                    if(is.null(formulaToUse)){
                        xregModelInitials[[1]]$formula <- formula(almModel);
                    }
                    xregModelInitials[[1]]$other <- almModel$other;
                    # Multiplicative model
                    almModel[] <- xregInitialiser("M",distribution,formulaToUse,subset,responseName);
                    # If Intercept was not included, substitute with zero
                    if(!any(names(almModel$coefficients)=="(Intercept)")){
                        almIntercept <- 0;
                    }
                    else{
                        almIntercept <- almModel$coefficients["(Intercept)"];
                    }
                    # If this is just a regression, include intercept
                    if(!etsModel && !arimaModel){
                        xregModelInitials[[2]]$initialXreg <- almModel$coefficients;
                    }
                    else{
                        xregModelInitials[[2]]$initialXreg <- almModel$coefficients[-1];
                    }
                    if(is.null(formulaToUse)){
                        xregModelInitials[[2]]$formula <- formula(almModel);
                    }
                    xregModelInitials[[2]]$other <- almModel$other;
                }

                # If this is just a regression, include intercept
                if(!etsModel && !arimaModel){
                    xregNumber <- ncol(almModel$data);
                    xregNames <- names(coef(almModel));
                }
                else{
                    # Write down the number and names of parameters
                    xregNumber <- ncol(almModel$data)-1;
                    xregNames <- names(coef(almModel))[-1];
                }
                # The original number of obs in xreg
                obsXreg <- nrow(xreg);

                #### Data manipulations for further use ####
                # This formula is needed in order to expand the data
                if(is.null(formulaToUse)){
                    formulaToUse <- as.formula(paste0("`",responseName,"`~",
                                                      paste0(colnames(xreg)[colnames(xreg)!=responseName],
                                                             collapse="+")));
                }

                # Remove variables without variability and amend the formula
                noVariability <- setNames(vector("logical",ncol(xreg)),colnames(xreg));
                noVariability[] <- apply((as.matrix(xreg[1:obsInSample,])==matrix(xreg[1,],obsInSample,ncol(xreg),byrow=TRUE)),2,all);
                noVariabilityNames <- names(noVariability)[noVariability];
                # If there are variables with no variability, their nams are in the formula, and the formula is not "y~.",
                # update the formula
                if(any(noVariability) && any(all.vars(formulaToUse) %in% names(noVariability)) &&
                   all.vars(formulaToUse)[2]!="."){
                    formulaToUse <- update.formula(formulaToUse,
                                                   paste0(".~.-",paste0(noVariabilityNames,collapse="-")));
                }

                # Robustify the names of variables
                colnames(xreg) <- make.names(colnames(xreg),unique=TRUE);
                # The names of the original variables
                xregNamesOriginal <- colnames(xregData)[-1];
                if((is.matrix(xreg) && any(apply(xreg,2,is.character))) ||
                   (!is.matrix(xreg) && any(sapply(xreg,is.character)))){
                    warning("You have character variables in your data. ",
                            "I will treat them as factors, but it is advised to convert them to factors manually",
                            call.=FALSE);
                }
                # If xreg is data frame or formula is provided, do model.frame
                additionalManipulations <- (is.data.frame(xreg) || formulaProvided);
                if(additionalManipulations){
                    # Expand the variables. We cannot use alm, because it is based on obsInSample
                    xregData <- model.frame(formulaToUse,data=as.data.frame(xreg));
                }
                else{
                    # If there was no formula, use the alm variable names
                    if(!formulaProvided){
                        formulaToUse <- as.formula(paste0("`",responseName,"`~",
                                                      paste0(xregNames, collapse="+")));
                    }
                    # Use the matrix with the xregNames - all the others are dropped by alm().
                    xregData <- as.matrix(xreg)[,xregNames,drop=FALSE];
                    xregNumber <- ncol(xregData);
                    xregNames <- colnames(xregData);
                }

                # If the number of rows is different, this might be because of NAs
                if(additionalManipulations && nrow(xregData)!=nrow(xreg)){
                    warning("Some variables contained NAs. This might cause issues in the estimation. ",
                            "I will substitute those values with the first non-NA values",
                            call.=FALSE);
                    # Get indices of NAs and nonNAs
                    xregNAs <- which(is.na(xreg),arr.ind=TRUE);
                    xregNonNAs <- which(!is.na(xreg),arr.ind=TRUE);
                    # Go through variables and substitute values
                    for(i in unique(xregNAs[,2])){
                        # This split on [row, column] is needed, because data.table is funny with cbind(row,column)
                        xreg[xregNAs[xregNAs[,2]==i,]] <- xreg[xregNonNAs[which(xregNonNAs[,2]==i)[1],1],
                                                               xregNonNAs[which(xregNonNAs[,2]==i)[1],2]];
                    }
                    xregData <- model.frame(formulaToUse,data=as.data.frame(xreg));
                }
                # Get the response variable, just in case it was transformed
                if(additionalManipulations && length(formulaToUse[[2]])>1){
                    y <- xregData[,1];
                    yInSample <- matrix(y[1:obsInSample],ncol=1);
                    if(holdout){
                        yHoldout <- matrix(y[-c(1:obsInSample)],ncol=1);
                    }
                }

                #### Drop the variables with no variability and perfectly correlated
                # This part takes care of explanatory variables potentially dropped by ALM
                # This is needed to get the correct xregData
                if(additionalManipulations){
                    # If we have a data.frame and there is a factor, we need to compare variables
                    # differently
                    xregFactors <- FALSE
                    if(is.data.frame(xreg) && any(sapply(xreg, is.factor))){
                        xregFactors[] <- TRUE;
                    }
                    xregExpanded <- colnames(model.matrix(xregData,data=xregData));
                    # Remove intercept
                    if(any(xregExpanded=="(Intercept)")){
                        xregExpanded <- xregExpanded[-1];
                    }
                    # Fix formula in case some variables were dropped by alm()
                    if(any(!(xregExpanded %in% xregNames))){
                        xregNamesRetained <- rep(TRUE,length(xregNamesOriginal));
                        for(i in 1:length(xregNamesOriginal)){
                            if(xregFactors){
                                xregNamesRetained[i] <- any(grepl(xregNamesOriginal[i], xregNames));
                            }
                            else{
                                xregNamesRetained[i] <- any(xregNamesOriginal[i] %in% xregNames);
                            }
                        }

                        # If the dropped variables are in the formula, update the formula
                        if(length(all.vars(formulaToUse))>2 &&
                           any(all.vars(formulaToUse) %in% xregNamesOriginal[!xregNamesRetained])){
                            formulaToUse <- update.formula(formulaToUse,
                                                           paste0(".~.-",
                                                                  paste(xregNamesOriginal[!xregNamesRetained],
                                                                        collapse="-")));
                        }
                        xregNamesOriginal <- c(responseName,xregNamesOriginal[xregNamesRetained]);
                        xregData <- model.frame(formulaToUse,data=as.data.frame(xreg[,xregNamesOriginal,drop=FALSE]));
                        # Remove response variable
                        xregNamesOriginal <- xregNamesOriginal[-1]
                    }

                    # Binary, flagging factors in the data
                    xregFactors <- (attr(terms(xregData),"dataClasses")=="factor")[-1];

                    # Expanded stuff with all levels for factors
                    if(any(xregFactors)){
                        # Levels for the factors
                        xregFactorsLevels <- lapply(xreg,levels);
                        xregFactorsLevels[[responseName]] <- NULL;
                        xregModelMatrix <- model.matrix(xregData,xregData,
                                                        contrasts.arg=lapply(xregData[attr(terms(xregData),"dataClasses")=="factor"],
                                                                             contrasts, contrasts=FALSE));
                        xregNamesModified <- colnames(xregModelMatrix)[-1];
                    }
                    else{
                        # Drop variables that were removed by alm()
                        xregModelMatrix <- model.matrix(xregData,data=xregData);
                        xregNamesModified <- xregNames;
                    }
                    # Drop the unused variables
                    xregData <- as.matrix(xregModelMatrix);

                    # Remove intercept
                    interceptIsPresent <- FALSE;
                    if(any(colnames(xregData)=="(Intercept)")){
                        interceptIsPresent[] <- TRUE;
                        xregData <- xregData[,-1,drop=FALSE];
                    }
                    xregNumber <- ncol(xregData);

                    # If there are more obs in xreg than the obsAll cut the thing
                    if(obsXreg>=obsAll){
                        xregData <- xregData[1:obsAll,,drop=FALSE]
                    }
                    # If there are less xreg observations than obsAll, use Naive
                    else{
                        newnRows <- obsAll-obsXreg;
                        xregData <- rbind(xregData,matrix(rep(tail(xregData,1),each=newnRows),newnRows,xregNumber));
                    }

                    #### Fix parameters for factors ####
                    {
                        # This part of code is terrible! In the nutshell, what it does it keeps
                        # all levels of categorical variables to make sure that the same
                        # categorical variable is treated equally inside ETS. This is mainly needed
                        # for regressors="adapt", so that "colour" changes over time in the same way,
                        # and not that "blue" reacts to error more than "red"... Why did I do that?!!
                        #### Potentially, this part can be removed to save some brain cells from destruction

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
                        if(length(xregParametersPersistence)==0){
                            xregParametersPersistence <- 0;
                        }

                        # If there are factors not in the alm data, create additional initials
                        if(any(xregFactors) && any(!(xregNamesModified %in% xregNames))){
                            xregAbsent <- !(xregNamesModified %in% xregNames);
                            xregParametersNew <- setNames(rep(NA,xregNumber),xregNamesModified);
                            # If there is stuff for additive error model, fix parameters
                            if(!is.null(xregModelInitials[[1]])){
                                xregParametersNew[!xregAbsent] <- xregModelInitials[[1]]$initialXreg;
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
                                xregModelInitials[[1]]$initialXreg <- xregParametersNew;
                            }
                            # If there is stuff for multiplicative error model, fix parameters
                            if(!is.null(xregModelInitials[[2]])){
                                xregParametersNew[!xregAbsent] <- xregModelInitials[[2]]$initialXreg;
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
                                xregModelInitials[[2]]$initialXreg <- xregParametersNew;
                            }
                            xregNames <- xregNamesModified;
                        }
                        # The vector of parameters that should be estimated (numeric + original levels of factors)
                        xregParametersEstimated <- xregParametersIncluded
                        xregParametersEstimated[xregParametersEstimated!=0] <- 1;
                        xregParametersEstimated[xregParametersMissing==0 & xregParametersIncluded==0] <- 1;
                    }
                }
                else{
                    # If there are more obs in xreg than the obsAll cut the thing
                    if(obsXreg>=obsAll){
                        xregData <- xregData[1:obsAll,,drop=FALSE]
                    }
                    # If there are less xreg observations than obsAll, use Naive
                    else{
                        newnRows <- obsAll-obsXreg;
                        xregData <- rbind(xregData,matrix(rep(tail(xregData,1),each=newnRows),newnRows,xregNumber));
                    }
                    xregFactors <- FALSE;
                    xregParametersPersistence <- setNames(c(1:xregNumber),xregNames);
                    xregParametersEstimated <- setNames(rep(1,xregNumber),xregNames);
                    xregParametersMissing <- setNames(c(1:xregNumber),xregNames);
                    xregParametersIncluded <- setNames(c(1:xregNumber),xregNames);
                }

                # Remove xreg, just to preserve some memory
                rm(xreg);
                # Clean memory if have a big object
                if(obsInSample>10000){
                    gc(verbose=FALSE);
                }
            }
            #### Initial xreg are provided ####
            else{
                #### Pure regression ####
                #### If this is just a regression, then this must be a reuse of alm.
                if((!etsModel && !arimaModel) && regressors!="adapt"){
                    # Return the estimated model based on the provided xreg
                    if(is.null(formulaToUse)){
                        formulaToUse <- as.formula(paste0("`",responseName,"`~."));
                    }
                    if(distribution=="default"){
                        distribution[] <- "dnorm";
                    }
                    # Redefine loss for ALM
                    loss <- switch(loss,
                                   "MSEh"=,"TMSE"=,"GTMSE"=,"MSCE"="MSE",
                                   "MAEh"=,"TMAE"=,"GTMAE"=,"MACE"="MAE",
                                   "HAMh"=,"THAM"=,"GTHAM"=,"CHAM"="HAM",
                                   loss);
                    lossOriginal <- loss;
                    if(loss=="custom"){
                        loss <- lossFunction;
                    }

                    # Form subset in order to use in-sample only
                    subset <- rep(FALSE, obsAll);
                    subset[1:obsInSample] <- TRUE;
                    # Exclude zeroes if this is an occurrence model
                    if(occurrenceModel){
                        subset[1:obsInSample][!otLogical] <- FALSE;
                    }
                    # Fisher Information
                    if(is.null(ellipsis$FI)){
                        FI <- FALSE;
                    }
                    else{
                        FI <- ellipsis$FI;
                    }
                    if(is.null(ellipsis$B)){
                        B <- initialXreg;
                    }
                    else{
                        B <- ellipsis$B;
                    }
                    almModel <- do.call("alm", list(formula=formulaToUse, data=xregData,
                                                    distribution=distribution, loss=loss, subset=which(subset),
                                                    occurrence=occurrence,FI=FI,
                                                    parameters=B,fast=TRUE));
                    almModel$call$data <- as.name(yName);
                    return(almModel);
                }
                #### InitialXreg is provided ####
                else{
                    # Save the original data
                    xreg <- xregData;
                    initialXregProvided <- TRUE;
                    # Robustify the names of variables
                    colnames(xreg) <- make.names(colnames(xreg),unique=TRUE);

                    # This formula is needed only to expand xreg
                    if(is.null(formulaToUse)){
                        responseName <- colnames(xreg)[1]
                        formulaToUse <- as.formula(paste0("`",responseName,"`~",
                                                          paste0(colnames(xreg)[colnames(xreg)!=responseName],
                                                                 collapse="+")));
                    }
                    # Extract names and form a proper matrix for the regression
                    else{
                        responseName <- all.vars(formulaToUse)[1];
                    }

                    # Get the names of initials
                    xregNames <- names(initialXreg);

                    # The names of the original variables
                    xregNamesOriginal <- colnames(xregData)[-1];
                    # Expand the variables. We cannot use alm, because it is based on obsInSample
                    xregData <- model.frame(formulaToUse,data=as.data.frame(xreg));
                    # Get the response variable, just in case it was transformed
                    if(length(formulaToUse[[2]])>1){
                        y <- xregData[,1];
                        yInSample <- matrix(y[1:obsInSample],ncol=1);
                        if(holdout){
                            yHoldout <- matrix(y[-c(1:obsInSample)],ncol=1);
                        }
                    }

                    # Binary, flagging factors in the data
                    xregFactors <- (attr(terms(xregData),"dataClasses")=="factor")[-1];
                    # Expanded stuff with all levels for factors
                    if(any(xregFactors)){
                        # Levels for the factors
                        xregFactorsLevels <- lapply(xreg[,-1,drop=FALSE],levels);
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
                    obsXreg <- nrow(xregData);

                    # If there are more obs in xreg than the obsAll cut the thing
                    if(obsXreg>=obsAll){
                        xregData <- xregData[1:obsAll,,drop=FALSE]
                    }
                    # If there are less xreg observations than obsAll, use Naive
                    else{
                        newnRows <- obsAll-obsXreg;
                        xregData <- rbind(xregData,matrix(rep(tail(xregData,1),each=newnRows),newnRows,xregNumber));
                    }

                    # This variable is needed in order to do model.matrix only, when required.
                    xregDataIsDataFrame <- is.data.frame(xregData);
                    if(xregDataIsDataFrame){
                        # Expand xregData if it is a data frame
                        xregData <- model.frame(formulaToUse,data=xregData);
                    }
                    xregNumber[] <- ncol(xregData);

                    #### Fix parameters for factors ####
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
                    if(length(xregParametersPersistence)==0){
                        xregParametersPersistence <- 0;
                    }

                    # If there are factors and the number of initials is not the same as the number of parameters needed
                    # This stuff assumes that the provided xreg parameters are named.
                    if(any(xregFactors)){
                        # Expand the data again in order to find the missing elements
                        xregNames <- colnames(as.matrix(model.matrix(formulaToUse,
                                                                     model.frame(formulaToUse,data=as.data.frame(xreg)))));
                        if(any(xregNames=="(Intercept)")){
                            xregNames <- xregNames[xregNames!="(Intercept)"];
                        }
                        if(length(xregNames)!=length(xregNamesModified)){
                            xregAbsent <- !(xregNamesModified %in% xregNames);
                            # Create the new vector of parameters
                            xregParametersNew <- setNames(rep(NA,xregNumber),xregNamesModified);
                            if(!is.null(names(initialXreg))){
                                xregParametersNew[xregNames] <- initialXreg[xregNames];
                            }
                            else{
                                # Insert values one by one
                                j <- 1;
                                for(i in 1:length(xregNamesModified)){
                                    if(!xregAbsent[i]){
                                        xregParametersNew[i] <- initialXreg[j];
                                        j <- j+1;
                                    }
                                }
                            }
                            # Go through new names and find, where they came from. Then get the missing parameters
                            for(i in which(xregAbsent)){
                                # Find the name of the original variable
                                # Use only the last value... hoping that the names like x and x1 are not used.
                                xregNameFound <- tail(names(unlist(sapply(xregNamesOriginal,grep,xregNamesModified[i]))),1);
                                # Get the indices of all k-1 levels
                                xregParametersIncluded[xregNames[xregNames %in% paste0(xregNameFound,
                                                                                       xregFactorsLevels[[xregNameFound]])]] <- i;
                                # Get the index of the absent one
                                xregParametersMissing[i] <- i;
                                # Fill in the absent one
                                xregParametersNew[i] <- -sum(xregParametersNew[xregParametersIncluded==i],
                                                             na.rm=TRUE);
                            }
                            # Write down the new parameters
                            xregModelInitials[[1]]$initialXreg <- xregParametersNew;

                            # Write down the new parameters to the second part
                            if(any(Etype==c("M","Z"))){
                                xregModelInitials[[2]]$initialXreg <- xregParametersNew;
                            }
                            xregNames <- xregNamesModified;
                        }
                    }
                    else{
                        xregModelInitials[[1]]$initialXreg <- initialXreg;
                        if(any(Etype==c("M","Z"))){
                            xregModelInitials[[2]]$initialXreg <- initialXreg;
                        }
                    }
                    # The vector of parameters that should be estimated (numeric + original levels of factors)
                    xregParametersEstimated <- xregParametersIncluded
                    xregParametersEstimated[xregParametersEstimated!=0] <- 1;
                    xregParametersEstimated[xregParametersMissing==0 & xregParametersIncluded==0] <- 1;

                    parametersNumber[2,2] <- parametersNumber[2,2] + xregNumber;
                }

                # Remove xreg, just to preserve some memory
                rm(xreg);
                # Clean memory if have a big object
                if(obsInSample>10000){
                    gc(verbose=FALSE);
                }
            }
        }
        else{
            #### Pure regression ####
            #### If this is just a regression, use stepwise
            if((!etsModel && !arimaModel) && regressors!="adapt"){
                # Return the estimated model based on the provided xreg
                if(is.null(formulaToUse)){
                    formulaToUse <- reformulate(setdiff(colnames(xregData), responseName), response=responseName);
                    # formulaToUse <- as.formula(paste0("`",responseName,"`~."));
                    formulaProvided <- FALSE;
                }
                else{
                    formulaProvided <- TRUE;
                }
                if(distribution=="default"){
                    distribution[] <- "dnorm";
                }

                # Form subset in order to use in-sample only
                subset <- rep(FALSE, obsAll);
                subset[1:obsInSample] <- TRUE;
                # Exclude zeroes if this is an occurrence model
                if(occurrenceModel){
                    subset[1:obsInSample][!otLogical] <- FALSE;
                }

                almModel <- do.call("stepwise", list(data=xregData, formula=formulaToUse, subset=subset,
                                                     distribution=distribution, occurrence=omModel));
                almModel$call$data <- as.name(yName);
                return(almModel);
            }

            # Include only variables from the formula
            if(is.null(formulaToUse)){
                formulaToUse <- as.formula(paste0("`",responseName,"`~."));
            }
            else{
                # Do model.frame manipulations
                # We do it this way to avoid factors expansion into dummies at this stage
                mf <- as.call(list(quote(stats::model.frame), formula=formulaToUse,
                                   data=xregData, drop.unused.levels=FALSE));

                if(!is.data.frame(xregData)){
                    mf$data <- as.data.frame(xregData);
                }
                # Evaluate data frame to do transformations of variables
                xregData <- eval(mf, parent.frame());

                # Remove variables that have "-x" in the formula
                dataTerms <- terms(xregData);
                xregData <- xregData[,colnames(attr(dataTerms,"factors"))];
            }
            xregNumber <- ncol(xregData);
            xregNames <- colnames(xregData);
            initialXregProvided <- FALSE;
            xregFactors <- FALSE;
            xregParametersPersistence <- setNames(c(1:xregNumber),xregNames);
            xregParametersEstimated <- setNames(rep(1,xregNumber),xregNames);
            xregParametersMissing <- setNames(c(1:xregNumber),xregNames);
            xregParametersIncluded <- setNames(c(1:xregNumber),xregNames);
        }

        #### persistence for xreg ####
        # Process the persistence for xreg
        if(!is.null(persistenceXreg)){
            if(length(persistenceXreg)!=xregNumber && length(persistenceXreg)!=1){
                warning("The length of the provided persistence for the xreg variables is wrong. Reverting to the estimation.",
                        call.=FALSE);
                persistenceXreg <- rep(0.5,xregNumber);
                persistenceXregProvided <- FALSE;
                persistenceXregEstimate <- TRUE;
            }
            else if(length(persistenceXreg)==1){
                persistenceXreg <- rep(persistenceXreg,xregNumber);
                persistenceXregProvided <- TRUE;
                persistenceXregEstimate <- FALSE;
            }
            else{
                persistenceXregProvided <- TRUE;
                persistenceXregEstimate <- FALSE;
            }
        }
        else{
            # The persistenceXregEstimate is provided
            if(regressors=="adapt" && !persistenceXregEstimate){
                persistenceXregProvided <- TRUE;
                persistenceXregEstimate <- FALSE;
            }
            else if(regressors=="adapt" && persistenceXregEstimate){
                persistenceXreg <- rep(0.05,xregNumber);
                persistenceXregProvided <- FALSE;
                persistenceXregEstimate <- TRUE;
            }
            else{
                persistenceXreg <- rep(0,xregNumber);
                persistenceXregProvided <- FALSE;
                persistenceXregEstimate <- FALSE;
            }
        }

        # If this is just a regression, include intercept
        if(!etsModel && !arimaModel){
            lagsModelAll <- matrix(rep(1,xregNumber),ncol=1);
        }
        else{
            lagsModelAll <- matrix(c(lagsModelAll,rep(1,xregNumber)),ncol=1);
        }
        # If there's only one explanatory variable, then there's nothing to select
        if(xregNumber==1){
            regressors[] <- "use";
        }

        # Fix the names of variables
        colnames(xregData) <- make.names(colnames(xregData), unique=TRUE);
        xregNames[] <- make.names(xregNames, unique=TRUE);
        xregData <- as.matrix(xregData);

        # If there are no variables after all of that, then xreg doesn't exist
        if(xregNumber==0){
            warning("It looks like there are no suitable explanatory variables. Check the xreg! I dropped them out.",
                    call.=FALSE);
            xregModel[] <- FALSE;
            xregData <- NULL;
        }
    }
    else{
        initialXregProvided <- FALSE;
        initialXregEstimate <- FALSE;
        persistenceXregProvided <- FALSE;
        persistenceXregEstimate <- FALSE;
        xregModelInitials <- NULL;
        xregData <- NULL;
        xregNumber <- 0;
        xregNames <- NULL;
        if(is.null(formulaToUse)){
            formulaToUse <- as.formula(paste0("`",responseName,"`~."));
        }
        xregParametersMissing <- 0;
        xregParametersIncluded <- 0;
        xregParametersEstimated <- 0;
        xregParametersPersistence <- 0;
    }

    # Fix the occurrenceModel for "provided"
    if(occurrence=="provided"){
        occurrenceModel <- FALSE;
    }

    # Redefine persitenceEstimate value
    persistenceEstimate[] <- any(c(persistenceLevelEstimate,persistenceTrendEstimate,
                                   persistenceSeasonalEstimate,persistenceXregEstimate));

    #### Conclusions about the initials ####
    # Make sure that only important elements are estimated.
    if(!modelIsTrendy){
        initialTrendEstimate <- FALSE;
        initialTrend <- NULL;
    }
    if(!modelIsSeasonal){
        initialSeasonalEstimate <- FALSE;
        initialSeasonal <- NULL;
    }
    if(!arimaModel){
        initialArimaEstimate <- FALSE;
        initialArima <- NULL;
    }
    if(!xregModel){
        initialXregEstimate <- FALSE;
        initialXreg <- NULL;
    }

    # If we don't need to estimate anything, flag initialEstimate
    if(!any(c(etsModel && initialLevelEstimate,
              (etsModel && modelIsTrendy && initialTrendEstimate),
              (etsModel && modelIsSeasonal && all(initialSeasonalEstimate)),
              (arimaModel && initialArimaEstimate),
              (xregModel && initialXregEstimate)))){
        initialEstimate[] <- FALSE;
    }
    else{
        initialEstimate[] <- TRUE;
    }

    # If at least something is provided, flag it as "provided"
    if((etsModel && !initialLevelEstimate) ||
       (etsModel && modelIsTrendy && !initialTrendEstimate) ||
       (etsModel && modelIsSeasonal && any(!initialSeasonalEstimate)) ||
       (arimaModel && !initialArimaEstimate) ||
       (xregModel && !initialXregEstimate)){
        initialType[] <- "provided";
    }

    # Observations in the states matrix
    # Define the number of cols that should be in the matvt
    obsStates <- obsInSample + lagsModelMax;

    if(any(yInSample<=0) && any(distribution==c("dinvgauss","dgamma","dlnorm","dllaplace","dls","dlgnorm")) &&
       !occurrenceModel && (occurrence!="provided")){
        warning(paste0("You have non-positive values in the data. ",
                       "The distribution ",distribution," does not support that. ",
                       "This might lead to problems in the estimation."),
                call.=FALSE);
    }

    # Update the number of parameters
    if(occurrenceModelProvided){
        parametersNumber[2,3] <- nparam(omModel);
        pForecast <- c(forecast(omModel, h=h)$mean);
    }

    #### Information Criteria ####
    ic <- match.arg(ic,c("AICc","AIC","BIC","BICc"));
    icFunction <- switch(ic,
                         "AIC"=AIC,
                         "AICc"=AICc,
                         "BIC"=BIC,
                         "BICc"=BICc);

    #### Bounds for the smoothing parameters ####
    bounds <- match.arg(bounds,c("usual","admissible","none"));


    #### Checks for the potential number of degrees of freedom ####
    # This is needed in order to make the function work on small samples
    # scale parameter, smoothing parameters and phi
    nParamMax <- (1 +
                      # ETS model
                      etsModel*(persistenceLevelEstimate + modelIsTrendy*persistenceTrendEstimate +
                                    modelIsSeasonal*sum(persistenceSeasonalEstimate) +
                                    phiEstimate + any(initialType==c("optimal","two-stage")) *
                                    (initialLevelEstimate + initialTrendEstimate + sum(initialSeasonalEstimate*lagsModelSeasonal))) +
                      # ARIMA components: initials + parameters
                      arimaModel*(any(initialType==c("optimal","two-stage"))*initialArimaNumber +
                                      arRequired*arEstimate*sum(arOrders) + maRequired*maEstimate*sum(maOrders)) +
                      # Xreg initials and smoothing parameters
                      xregModel*(xregNumber*(any(initialType==c("backcasting","optimal","two-stage"))*
                                                 initialXregEstimate+persistenceXregEstimate)));

    # If the sample is smaller than the number of parameters
    if(obsNonzero <= nParamMax){
        # If there is both ETS and ARIMA, remove ARIMA
        if(etsModel && arimaModel && !select){
            warning("I don't have enough observations to fit ETS with ARIMA terms. I will construct the simple ETS.",
                    call.=FALSE);
            lagsModelAll <- lagsModelAll[-c(componentsNumberETS+c(1:componentsNumberARIMA)),,drop=FALSE];
            arRequired <- iRequired <- maRequired <- arimaModel <- FALSE;
            arOrders <- iOrders <- maOrders <- NULL;
            nonZeroARI <- nonZeroMA <- lagsModelARIMA <- NULL;
            componentsNamesARIMA <- NULL;
            initialArimaNumber <- componentsNumberARIMA <- 0;
            lagsModelMax <- max(lagsModelAll);
        }
        else if(arimaModel && !etsModel && !select){
            # If the backacasting helps, switch to it.
            if(any(initialType==c("optimal","two-stage")) &&
               (obsNonzero > (nParamMax - any(initialType==c("optimal","two-stage"))*initialArimaNumber))){
                warning(paste0("The number of parameter to estimate is ",nParamMax,
                            ", while the number of observations is ",obsNonzero,
                            ". Switching initial to 'backcasting' to save some degrees of freedom."), call.=FALSE);
                initialType <- "complete";
            }
            else{
                warning(paste0("The number of parameter to estimate is ",nParamMax,
                               ", while the number of observations is ",obsNonzero,
                               ". Switching to ARIMA(0,1,1) model."), call.=FALSE);
                # Add ARIMA(0,1,1) lags
                lagsModelAll <- matrix(1,1,1);
                arRequired <- FALSE;
                iRequired <- maRequired <- arimaModel <- TRUE;
                arOrders <- 0;
                iOrders <- maOrders <- 1;
                nonZeroARI <- nonZeroMA <- matrix(c(2,1),1,2);
                lagsModelARIMA <- matrix(1,1,1);
                componentsNamesARIMA <- 1;
                initialArimaNumber <- componentsNumberARIMA <- 1;
                lagsModelMax <- max(lagsModelAll);
            }
        }
    }

    if(arimaModel && obsNonzero < any(initialType==c("optimal","two-stage"))*initialArimaNumber && !select){
        warning(paste0("In-sample size is ",obsNonzero,", while number of ARIMA components is ",initialArimaNumber,
                       ". Cannot fit the model."),call.=FALSE)
        stop("Not enough observations for such a complicated model.",call.=FALSE);
    }

    # Recalculate the number of parameters
    nParamMax[] <- (1 +
                        # ETS model
                        etsModel*(persistenceLevelEstimate + modelIsTrendy*persistenceTrendEstimate +
                                      modelIsSeasonal*sum(persistenceSeasonalEstimate) +
                                      phiEstimate + any(initialType==c("optimal","two-stage")) *
                                      (initialLevelEstimate + initialTrendEstimate + sum(initialSeasonalEstimate*lagsModelSeasonal))) +
                        # ARIMA components: initials + parameters
                        arimaModel*(any(initialType==c("optimal","two-stage"))*initialArimaNumber +
                                        arRequired*arEstimate*sum(arOrders) + maRequired*maEstimate*sum(maOrders)) +
                        # Xreg initials and smoothing parameters
                        xregModel*(xregNumber*(any(initialType==c("backcasting","optimal","two-stage"))*initialXregEstimate+persistenceXregEstimate)));

    # If the sample is still smaller than the number of parameters (even after removing ARIMA)
    if(etsModel){
        if(obsNonzero <= nParamMax){
            nParamExo <- xregNumber*(initialXregEstimate+persistenceXregEstimate);
            if(!silent){
                message(paste0("Number of non-zero observations is ",obsNonzero,
                               ", while the maximum number of parameters to estimate is ", nParamMax,".\n",
                               "Updating pool of models."));
            }

            # If the number of observations is still enough for the model selection and the pool is not specified
            if(obsNonzero > (3 + nParamExo) && is.null(modelsPool) && any(modelDo==c("select","combine"))){
                # We have enough observations for local level model
                modelsPool <- c("ANN");
                if(allowMultiplicative){
                    modelsPool <- c(modelsPool,"MNN");
                }
                # We have enough observations for trend model
                if(obsNonzero > (5 + nParamExo)){
                    if(any(Ttype==c("Z","X","A"))){
                        modelsPool <- c(modelsPool,"AAN","MAN");
                    }
                    if(allowMultiplicative && any(Ttype==c("Z","Y","M"))){
                        modelsPool <- c(modelsPool,"AMN","MMN");
                    }
                }
                # We have enough observations for damped trend model
                if(obsNonzero > (6 + nParamExo)){
                    if(any(Ttype==c("Z","X","A"))){
                        modelsPool <- c(modelsPool,"AAdN","MAdN");
                    }
                    if(allowMultiplicative && any(Ttype==c("Z","Y","M"))){
                        modelsPool <- c(modelsPool,"AMdN","MMdN");
                    }
                }
                # We have enough observations for seasonal model
                if((obsNonzero > (lagsModelMax)) && lagsModelMax!=1){
                    if(any(Stype==c("Z","X","A"))){
                        modelsPool <- c(modelsPool,"ANA","MNA");
                    }
                    if(allowMultiplicative && any(Stype==c("Z","Y","M"))){
                        modelsPool <- c(modelsPool,"ANM","MNM");
                    }
                }
                # We have enough observations for seasonal model with trend
                if((obsNonzero > (6 + lagsModelMax + nParamExo)) &&
                   (obsNonzero > 2*lagsModelMax) && lagsModelMax!=1){
                    if(any(Ttype==c("Z","X","A")) && any(Stype==c("Z","X","A"))){
                        modelsPool <- c(modelsPool,"AAA");
                    }
                    if(allowMultiplicative && any(Ttype==c("Z","X","A")) && any(Stype==c("Z","Y","A"))){
                        modelsPool <- c(modelsPool,"MAA");
                    }
                    if(allowMultiplicative && any(Ttype==c("Z","X","A")) && any(Stype==c("Z","Y","M"))){
                        modelsPool <- c(modelsPool,"AAM","MAM");
                    }
                    if(allowMultiplicative && any(Ttype==c("Z","Y","M")) && any(Stype==c("Z","X","A"))){
                        modelsPool <- c(modelsPool,"AMA","MMA");
                    }
                    if(allowMultiplicative && any(Ttype==c("Z","Y","M")) && any(Stype==c("Z","Y","M"))){
                        modelsPool <- c(modelsPool,"AMM","MMM");
                    }
                }

                warning("Not enough of non-zero observations for the fit of ETS(",model,")! Fitting what I can...",
                        call.=FALSE);
                if(modelDo=="combine"){
                    model <- "CNN";
                    if(length(modelsPool)>2){
                        model <- "CCN";
                    }
                    if(length(modelsPool)>10){
                        model <- "CCC";
                    }
                }
                else{
                    modelDo <- "select"
                    model <- "ZZZ";
                }
            }
            # If the pool is provided (so, select / combine), amend it
            else if(obsNonzero > (3 + nParamExo) && !is.null(modelsPool)){
                # We don't have enough observations for seasonal models with damped trend
                if((obsNonzero <= (6 + lagsModelMax + 1 + nParamExo))){
                    modelsPool <- modelsPool[!(nchar(modelsPool)==4 &
                                                   substr(modelsPool,nchar(modelsPool),nchar(modelsPool))=="A")];
                    modelsPool <- modelsPool[!(nchar(modelsPool)==4 &
                                                   substr(modelsPool,nchar(modelsPool),nchar(modelsPool))=="M")];
                }
                # We don't have enough observations for seasonal models with trend
                if((obsNonzero <= (5 + lagsModelMax + 1 + nParamExo))){
                    modelsPool <- modelsPool[!(substr(modelsPool,2,2)!="N" &
                                                   substr(modelsPool,nchar(modelsPool),nchar(modelsPool))!="N")];
                }
                # We don't have enough observations for seasonal models
                if(obsNonzero <= lagsModelMax){
                    modelsPool <- modelsPool[substr(modelsPool,nchar(modelsPool),nchar(modelsPool))=="N"];
                }
                # We don't have enough observations for damped trend
                if(obsNonzero <= (6 + nParamExo)){
                    modelsPool <- modelsPool[nchar(modelsPool)!=4];
                }
                # We don't have enough observations for any trend
                if(obsNonzero <= (5 + nParamExo)){
                    modelsPool <- modelsPool[substr(modelsPool,2,2)=="N"];
                }

                modelsPool <- unique(modelsPool);
                warning("Not enough of non-zero observations for the fit of ETS(",model,")! Fitting what I can...",
                        call.=FALSE);
                if(modelDo=="combine"){
                    model <- "CNN";
                    if(length(modelsPool)>2){
                        model <- "CCN";
                    }
                    if(length(modelsPool)>10){
                        model <- "CCC";
                    }
                }
                else{
                    modelDo <- "select"
                    model <- "ZZZ";
                }
            }
            # If the model needs to be estimated / used, not selected
            else if(obsNonzero > (3 + nParamExo) && any(modelDo==c("estimate","use"))){
                # We don't have enough observations for seasonal models with damped trend
                if((obsNonzero <= (6 + lagsModelMax + 1 + nParamExo))){
                    if(nchar(model)==4){
                        warning("Not enough of non-zero observations for the fit of ETS(",model,")! Fitting what I can...",
                                call.=FALSE);
                        model <- paste0(substr(model,1,2),substr(model,4,4));
                    }
                    # model <- model[!(nchar(model)==4 &
                    #                      substr(model,nchar(model),nchar(model))=="A")];
                    # model <- model[!(nchar(model)==4 &
                    #                      substr(model,nchar(model),nchar(model))=="M")];
                }
                # We don't have enough observations for seasonal models with trend
                if((obsNonzero <= (5 + lagsModelMax + 1 + nParamExo))){
                    warning("Not enough of non-zero observations for the fit of ETS(",model,")! Fitting what I can...",
                            call.=FALSE);
                    model <- paste0(substr(model,1,1),"N",substr(model,3,3));
                    # model <- model[!(substr(model,2,2)!="N" &
                    #                      substr(model,nchar(model),nchar(model))!="N")];
                }
                # We don't have enough observations for seasonal models
                if(obsNonzero <= lagsModelMax){
                    warning("Not enough of non-zero observations for the fit of ETS(",model,")! Fitting what I can...",
                            call.=FALSE);
                    model <- paste0(substr(model,1,2),"N");
                    # model <- model[substr(model,nchar(model),nchar(model))=="N"];
                }
                # We don't have enough observations for damped trend
                if(obsNonzero <= (6 + nParamExo)){
                    if(nchar(model)==4){
                        warning("Not enough of non-zero observations for the fit of ETS(",model,")! Fitting what I can...",
                                call.=FALSE);
                        model <- paste0(substr(model,1,2),substr(model,4,4));
                    }
                    # model <- model[nchar(model)!=4];
                }
                # We don't have enough observations for any trend
                if(obsNonzero <= (5 + nParamExo)){
                    warning("Not enough of non-zero observations for the fit of ETS(",model,")! Fitting what I can...",
                            call.=FALSE);
                    model <- paste0(substr(model,1,1),"N",substr(model,3,3));
                    # model <- model[substr(model,2,2)=="N"];
                }
                # Change E,T,S elements based on the trimmed thingy
                Etype <- substr(model,1,1);
                Ttype <- substr(model,2,2);
                Stype <- substr(model,nchar(model),nchar(model));

                modelIsTrendy <- Ttype!="N";
                modelIsSeasonal <- Stype!="N";

                damped <- modelIsTrendy && (nchar(model)==4);
                phiEstimate <- damped;
            }
            # Extreme cases of small samples
            else if(obsNonzero==4){
                if(any(Etype==c("A","M"))){
                    modelDo <- "estimate";
                    Ttype <- "N";
                    Stype <- "N";
                }
                else{
                    modelsPool <- c("ANN");
                    if(allowMultiplicative){
                        modelsPool <- c(modelsPool,"MNN");
                    }
                    modelDo <- "select";
                    model <- "ZZZ";
                    Etype <- "Z";
                    Ttype <- "N";
                    Stype <- "N";
                    warning("You have a very small sample. The only available model is level model.",
                            call.=FALSE);
                }
                phiEstimate <- FALSE;
            }
            # Even smaller sample
            else if(obsNonzero==3){
                if(any(Etype==c("A","M"))){
                    modelDo <- "estimate";
                    Ttype <- "N";
                    Stype <- "N";
                    model <- paste0(Etype,"NN");
                }
                else{
                    modelsPool <- c("ANN");
                    if(allowMultiplicative){
                        modelsPool <- c(modelsPool,"MNN");
                    }
                    modelDo <- "select";
                    model <- "ZNN";
                    Etype <- "Z";
                    Ttype <- "N";
                    Stype <- "N";
                }
                persistenceLevel <- 0;
                persistenceEstimate <- persistenceLevelEstimate <- FALSE;
                warning("I did not have enough of non-zero observations, so persistence value was set to zero.",
                        call.=FALSE);
                phiEstimate <- FALSE;
            }
            # Can it be even smaller?
            else if(obsNonzero==2){
                modelsPool <- NULL;
                persistenceLevel <- 0;
                persistenceEstimate <- persistenceLevelEstimate <- FALSE;
                initialLevel <- mean(yInSample);
                initialType <- "provided";
                initialEstimate <- initialLevelEstimate <- FALSE;
                warning("I did not have enough of non-zero observations, so persistence value was set to zero and initial was preset.",
                        call.=FALSE);
                modelDo <- "use";
                model <- "ANN";
                Etype <- "A";
                Ttype <- "N";
                Stype <- "N";
                phiEstimate <- FALSE;
                parametersNumber[1,1] <- 0;
                parametersNumber[2,1] <- 2;
            }
            # And how about now?!
            else if(obsNonzero==1){
                modelsPool <- NULL;
                persistenceLevel <- 0;
                persistenceEstimate <- persistenceLevelEstimate <- FALSE;
                initialLevel <- yInSample[yInSample!=0];
                initialType <- "provided";
                initialEstimate <- initialLevelEstimate <- FALSE;
                warning("I did not have enough of non-zero observations, so I used Naive.",call.=FALSE);
                modelDo <- "use"
                model <- "ANN";
                Etype <- "A";
                Ttype <- "N";
                Stype <- "N";
                phiEstimate <- FALSE;
                parametersNumber[1,1] <- 0;
                parametersNumber[2,1] <- 2;
            }
            # Only zeroes in the data...
            else if(obsNonzero==0 && obsInSample>1){
                modelsPool <- NULL;
                persistenceLevel <- 0;
                persistenceEstimate <- persistenceLevelEstimate <- FALSE;
                initialLevel <- 0;
                initialType <- "provided";
                initialEstimate <- initialLevelEstimate <- FALSE;
                occurrenceModelProvided <- occurrenceModel <- FALSE;
                occurrence <- "none";
                warning("You have a sample with zeroes only. Your forecast will be zero.",call.=FALSE);
                modelDo <- "use"
                model <- "ANN";
                Etype <- "A";
                Ttype <- "N";
                Stype <- "N";
                phiEstimate <- FALSE;
                parametersNumber[1,1] <- 0;
                parametersNumber[2,1] <- 2;
            }
            # If you don't have observations, then fuck off!
            else{
                stop("Not enough observations... Even for fitting of ETS('ANN')!",call.=FALSE);
            }
        }
    }
    # Reset the maximum lag. This is in order to take potential changes into account
    lagsModelMax[] <- max(lagsModelAll);

    #### Process ellipsis ####
    list2env(adam_checkOptimizer(ellipsis, loss, distribution, initialType, lags, arimaModel),
             envir=environment());

    # Add constant in the model
    if(is.numeric(constant)){
        constantRequired <- TRUE;
        constantEstimate <- FALSE;
        # This is just in case a vector was provided
        constantValue <- constant[1];
        if(any(iOrders!=0) || etsModel){
            constantName <- "drift";
        }
        else{
            constantName <- "constant";
        }
    }
    else if(is.logical(constant)){
        constantEstimate <- constantRequired <- constant;
        constantValue <- NULL;
        if(constantRequired){
            if(any(iOrders!=0) || etsModel){
                constantName <- "drift";
            }
            else{
                constantName <- "constant";
            }
        }
        else{
            constantName <- NULL;
        }
    }
    else{
        warning("The parameter constant can only be TRUE or FALSE.",
                "You have: ",constant,". Switching to FALSE",call.=FALSE);
        constantEstimate <- constantRequired <- FALSE;
        constantName <- NULL;
    }

    # If there is no model, return a constant level
    if(!etsModel && !arimaModel && !xregModel){
        modelsPool <- NULL;
        if(!constantRequired){
            constantEstimate <- constantRequired <- TRUE;
            constantName <- "constant";
        }

        model <- "NNN";
        if(is.null(B)){
            modelDo <- "estimate";
        }
        Etype <- switch(distribution,
                        "default"=,"dnorm"=,"dlaplace"=,"ds"=,"dgnorm"=,"dlogis"=,"plogis"=,"dt"=,"dalaplace"="A",
                        "dlnorm"=,"dllaplace"=,"dls"=,"dlgnorm"=,"dinvgauss"=,"dgamma"="M",
                        "A");
        Ttype <- "N";
        Stype <- "N";
        phiEstimate <- FALSE;
        parametersNumber[1,1] <- 0;
        parametersNumber[2,1] <- 2;
    }

    # See if the estimation of the model is not needed (do we estimate anything?)
    if(!any(c(etsModel & c(persistenceLevelEstimate, persistenceTrendEstimate,
                           persistenceSeasonalEstimate, phiEstimate,
                           all(initialType!=c("complete","backcasting","gradient")) & c(initialLevelEstimate,
                                                                             initialTrendEstimate,
                                                                             initialSeasonalEstimate)),
              arimaModel & c(arEstimate, maEstimate,
                             all(initialType!=c("complete","backcasting","gradient")) & initialEstimate & initialArimaEstimate),
              xregModel & c(persistenceXregEstimate, (initialType!="complete") & initialXregEstimate),
              constantEstimate,
              otherParameterEstimate))){
        modelDo <- "use";
    }

    # Switch usual bounds to the admissible if there's no ETS - this speeds up ARIMA
    if(!etsModel && bounds=="usual"){
        bounds[] <- "admissible";
    }

    # If we do model selection / combination with non-standard losses, complain
    if(any(modelDo==c("select","combine")) &&
       ((any(loss==c("MSE","MSEh","MSCE","GPL")) && all(distribution!=c("default","dnorm"))) ||
        (any(loss==c("MAE","MAEh","MACE")) && all(distribution!=c("default","dlaplace"))) ||
        (any(loss==c("HAM","HAMh","CHAM")) && all(distribution!=c("default","ds"))) ||
        (any(loss==c("TMSE","GTMSE","TMAE","GTMAE","THAM","GTHAM"))))){
        warning("The model selection only works in case of loss='likelihood'. I hope you know what you are doing.",
                call.=FALSE);
    }

    # Just in case, give names to yHoldout and yInSample
    colnames(yInSample) <- responseName;
    if(holdout){
        colnames(yHoldout) <- responseName;
    }

    #### Return the validated parameters as a named list ####
    return(list(
        # Actuals
        y = y,
        yHoldout = yHoldout,
        yInSample = yInSample,
        yNAValues = yNAValues,
        # Index and structure
        yClasses = yClasses,
        yIndex = yIndex,
        yInSampleIndex = yInSampleIndex,
        yForecastIndex = yForecastIndex,
        yIndexAll = yIndexAll,
        yFrequency = yFrequency,
        yStart = yStart,
        yForecastStart = yForecastStart,
        horizon = h,
        h = h,
        holdout = holdout,
        # Observation counts
        obsInSample = obsInSample,
        obsAll = obsAll,
        obsStates = obsStates,
        obsNonzero = obsNonzero,
        obsZero = obsZero,
        parametersNumber = parametersNumber,
        # Model type
        etsModel = etsModel,
        model = model,
        Etype = Etype,
        Ttype = Ttype,
        Stype = Stype,
        modelIsTrendy = modelIsTrendy,
        modelIsSeasonal = modelIsSeasonal,
        modelsPool = modelsPool,
        damped = damped,
        modelDo = modelDo,
        allowMultiplicative = allowMultiplicative,
        # Component counts and names
        componentsNumberETS = componentsNumberETS,
        componentsNamesETS = componentsNamesETS,
        componentsNumberETSNonSeasonal = componentsNumberETS - componentsNumberETSSeasonal,
        componentsNumberETSSeasonal = componentsNumberETSSeasonal,
        componentsNumberARIMA = componentsNumberARIMA,
        componentsNamesARIMA = componentsNamesARIMA,
        # Lags
        lags = lags,
        lagsModel = lagsModel,
        lagsModelSeasonal = lagsModelSeasonal,
        lagsModelARIMA = lagsModelARIMA,
        lagsModelAll = lagsModelAll,
        lagsModelMax = lagsModelMax,
        # Persistence
        persistence = persistence,
        persistenceEstimate = persistenceEstimate,
        persistenceLevel = persistenceLevel,
        persistenceLevelEstimate = persistenceLevelEstimate,
        persistenceTrend = persistenceTrend,
        persistenceTrendEstimate = persistenceTrendEstimate,
        persistenceSeasonal = persistenceSeasonal,
        persistenceSeasonalEstimate = persistenceSeasonalEstimate,
        persistenceXreg = persistenceXreg,
        persistenceXregEstimate = persistenceXregEstimate,
        persistenceXregProvided = persistenceXregProvided,
        # phi
        phi = phi,
        phiEstimate = phiEstimate,
        # Initials
        initial = initial,
        initialType = initialType,
        initialEstimate = initialEstimate,
        initialLevel = initialLevel,
        initialLevelEstimate = initialLevelEstimate,
        initialTrend = initialTrend,
        initialTrendEstimate = initialTrendEstimate,
        initialSeasonal = initialSeasonal,
        initialSeasonalEstimate = initialSeasonalEstimate,
        initialArima = initialArima,
        initialArimaEstimate = initialArimaEstimate,
        initialArimaNumber = initialArimaNumber,
        initialXregEstimate = initialXregEstimate,
        initialXregProvided = initialXregProvided,
        # Occurrence model
        omModel = omModel,
        occurrenceModel = occurrenceModel,
        occurrenceModelProvided = occurrenceModelProvided,
        occurrence = occurrence,
        pFitted = pFitted,
        pForecast = pForecast,
        ot = ot,
        otLogical = otLogical,
        # Distribution, loss, bounds and IC
        distribution = distribution,
        loss = loss,
        lossFunction = lossFunction,
        multisteps = multisteps,
        ic = ic,
        icFunction = icFunction,
        bounds = bounds,
        # ARIMA components
        arimaModel = arimaModel,
        arOrders = arOrders,
        iOrders = iOrders,
        maOrders = maOrders,
        arRequired = arRequired,
        iRequired = iRequired,
        maRequired = maRequired,
        arEstimate = arEstimate,
        maEstimate = maEstimate,
        armaParameters = armaParameters,
        nonZeroARI = nonZeroARI,
        nonZeroMA = nonZeroMA,
        select = select,
        # Explanatory variables
        xregModel = xregModel,
        regressors = regressors,
        xregModelInitials = xregModelInitials,
        xregData = xregData,
        xregNumber = xregNumber,
        xregNames = xregNames,
        responseName = responseName,
        formula = formulaToUse,
        xregParametersMissing = xregParametersMissing,
        xregParametersIncluded = xregParametersIncluded,
        xregParametersEstimated = xregParametersEstimated,
        xregParametersPersistence = xregParametersPersistence,
        # Constant
        constantRequired = constantRequired,
        constantEstimate = constantEstimate,
        constantValue = constantValue,
        constantName = constantName,
        # Optimisation
        maxeval = maxeval,
        maxtime = maxtime,
        xtol_rel = xtol_rel,
        xtol_abs = xtol_abs,
        ftol_rel = ftol_rel,
        ftol_abs = ftol_abs,
        algorithm = algorithm,
        print_level = print_level,
        B = B,
        lb = lb,
        ub = ub,
        # Distribution shape and regularisation
        other = other,
        otherParameterEstimate = otherParameterEstimate,
        lambda = lambda,
        # Misc
        nIterations = nIterations,
        smoother = smoother,
        FI = FI,
        stepSize = stepSize,
        dfForBack = dfForBack
    ));
}

#### adamSpecificChecker: thin wrapper adding outliers/distribution early-exit ####
#' @keywords internal
adamSpecificChecker <- function(data, model, lags, formulaToUse, orders, constant=FALSE, arma,
                                outliers=c("ignore","use","select"), level=0.99,
                                persistence, phi, initial,
                                distribution=c("default","dnorm","dlaplace","dalaplace",
                                               "ds","dgnorm","dlnorm","dinvgauss","dgamma","plogis"),
                                loss, h, holdout, occurrence,
                                ic=c("AICc","AIC","BIC","BICc"),
                                bounds=c("usual","admissible","none"),
                                regressors, yName,
                                silent, modelDo,
                                ellipsis, fast=FALSE){
    outliers <- match.arg(outliers)
    if(!fast){
        distribution <- match.arg(distribution[1],
                                   c("default","dnorm","dlaplace","dalaplace",
                                     "ds","dgnorm","dlnorm","dinvgauss","dgamma","plogis"))
    }

    if(outliers != "ignore"){
        return(list(select=TRUE, distribution=distribution, outliers=outliers))
    }

    result <- commonParametersChecker(data=data, model=model, lags=lags,
                                      formulaToUse=formulaToUse,
                                      orders=orders, constant=constant, arma=arma,
                                      persistence=persistence, phi=phi, initial=initial,
                                      distribution=distribution, loss=loss,
                                      h=h, holdout=holdout,
                                      occurrence=occurrence, ic=ic, bounds=bounds,
                                      regressors=regressors, yName=yName,
                                      silent=silent, modelDo=modelDo,
                                      ellipsis=ellipsis, fast=fast)

    if(is.alm(result)){
        return(result)
    }

    return(c(result, list(outliers=outliers)))
}

#### Backward-compatible alias ####
parametersChecker <- adamSpecificChecker;
