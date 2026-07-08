##### *Checker of input of basic functions* #####
ssInput <- function(smoothType=c("es","gum","ces","ssarima","smoothC"),...){
    # This is universal function needed in order to check the passed arguments to es(), gum(),
    # ces() and ssarima()

    smoothType <- smoothType[1];

    ellipsis <- list(...);
    ParentEnvironment <- ellipsis[['ParentEnvironment']];

    ##### silent #####
    silent <- silent[1];
    # Fix for cases with TRUE/FALSE.
    if(!is.logical(silent)){
        if(all(silent!=c("none","all","graph","legend","output","debugging","n","a","g","l","o","d"))){
            warning(paste0("Sorry, I have no idea what 'silent=",silent,"' means. Switching to 'none'."),
                    call.=FALSE);
            silent <- "none";
        }
        silent <- substring(silent,1,1);
    }
    silentValue <- silent;

    if(silentValue==FALSE | silentValue=="n"){
        silentText <- FALSE;
        silentGraph <- FALSE;
        silentLegend <- FALSE;
    }
    else if(silentValue==TRUE | silentValue=="a"){
        silentText <- TRUE;
        silentGraph <- TRUE;
        silentLegend <- TRUE;
    }
    else if(silentValue=="g"){
        silentText <- FALSE;
        silentGraph <- TRUE;
        silentLegend <- TRUE;
    }
    else if(silentValue=="l"){
        silentText <- FALSE;
        silentGraph <- FALSE;
        silentLegend <- TRUE;
    }
    else if(silentValue=="o"){
        silentText <- TRUE;
        silentGraph <- FALSE;
        silentLegend <- FALSE;
    }
    else if(silentValue=="d"){
        silentText <- TRUE;
        silentGraph <- FALSE;
        silentLegend <- FALSE;
    }

    # Check what was passed as a horizon
    if(h<0){
        warning("Sorry, I can't do anything with negative horizon, so I'll set it to zero.",
                call.=FALSE);
        h <- 0;
    }

    ##### data #####
    if(any(is.smooth.sim(y))){
        y <- y$data;
    }
    else if(any(class(y)=="Mdata")){
        h <- y$h;
        holdout <- TRUE;
        y <- ts(c(y$x,y$xx),start=start(y$x),frequency=frequency(y$x));
    }

    if(!is.numeric(y)){
        stop("The provided data is not a vector or ts object! Can't construct any model!", call.=FALSE);
    }
    if(!is.null(ncol(y))){
        if(ncol(y)>1){
            stop("The provided data is not a vector! Can't construct any model!", call.=FALSE);
        }
    }
    # Check the data for NAs
    if(any(is.na(y))){
        if(!silentText){
            warning("Data contains NAs. These observations will be substituted by zeroes.",call.=FALSE);
        }
        y[is.na(y)] <- 0;
    }

    # Define obs, the number of observations of in-sample
    obsInSample <- length(y) - holdout*h;

    # Define obsAll, the overal number of observations (in-sample + holdout)
    obsAll <- length(y) + (1 - holdout)*h;

    # If obsInSample is negative, this means that we can't do anything...
    if(obsInSample<=0){
        stop("Not enough observations in sample.",call.=FALSE);
    }
    # Define the actual values
    yInSample <- matrix(y[1:obsInSample],obsInSample,1);
    dataFreq <- frequency(y);
    dataStart <- start(y);
    yForecastStart <- time(y)[obsInSample]+deltat(y);

    # Number of parameters to estimate / provided
    parametersNumber <- matrix(0,2,4,
                               dimnames=list(c("Estimated","Provided"),
                                             c("nParamInternal","nParamXreg","nParamOccurrence","nParamAll")));

    if(any(smoothType==c("es","oes"))){
        ##### model for ES #####
        if(!is.character(model)){
            stop(paste0("Something strange is provided instead of character object in model: ",
                        paste0(model,collapse=",")),call.=FALSE);
        }

        # Predefine models pool for a model selection
        modelsPool <- NULL;
        # Deal with the list of models. Check what has been provided. Stop if there is a mistake.
        if(length(model)>1){
            if(any(nchar(model)>4)){
                stop(paste0("You have defined strange model(s) in the pool: ",
                            paste0(model[nchar(model)>4],collapse=",")),call.=FALSE);
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

        # If chosen model is "AAdN" or anything like that, we are taking the appropriate values
        if(nchar(model)==4){
            Etype <- substring(model,1,1);
            Ttype <- substring(model,2,2);
            Stype <- substring(model,4,4);
            damped <- TRUE;
            if(substring(model,3,3)!="d"){
                message(paste0("You have defined a strange model: ",model));
                sowhat(model);
                model <- paste0(Etype,Ttype,"d",Stype);
            }
        }
        else if(nchar(model)==3){
            Etype <- substring(model,1,1);
            Ttype <- substring(model,2,2);
            Stype <- substring(model,3,3);
            if(any(Ttype==c("Z","X","Y"))){
                damped <- TRUE;
            }
            else{
                damped <- FALSE;
            }
        }
        else{
            message(paste0("You have defined a strange model: ",model));
            sowhat(model);
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
            else if(any(unlist(strsplit(model,""))=="Z") |
                    any(unlist(strsplit(model,""))=="X") |
                    any(unlist(strsplit(model,""))=="Y")){
                modelDo <- "select";
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

        ### Check error type
        if(all(Etype!=c("Z","X","Y","A","M","C"))){
            warning(paste0("Wrong error type: ",Etype,". Should be 'Z', 'X', 'Y', 'A' or 'M'.\n",
                           "Changing to 'Z'"),call.=FALSE);
            Etype <- "Z";
            modelDo <- "select";
        }

        ### Check trend type
        if(all(Ttype!=c("Z","X","Y","N","A","M","C"))){
            warning(paste0("Wrong trend type: ",Ttype,". Should be 'Z', 'X', 'Y', 'N', 'A' or 'M'.\n",
                           "Changing to 'Z'"),call.=FALSE);
            Ttype <- "Z";
            modelDo <- "select";
        }
    }
    else if(any(smoothType==c("ssarima","msarima"))){
        ##### Orders and lags for ssarima #####
        if(any(is.complex(c(ar.orders,i.orders,ma.orders,lags)))){
            stop("Come on! Be serious! This is ARIMA, not CES!",call.=FALSE);
        }

        if(any(c(ar.orders,i.orders,ma.orders)<0)){
            stop("Funny guy! How am I gonna construct a model with negative order?",call.=FALSE);
        }

        if(any(c(lags)<0)){
            stop("Right! Why don't you try complex lags then, mister smart guy?",call.=FALSE);
        }

        # If there are zero lags, drop them
        if(any(lags==0)){
            ar.orders <- ar.orders[lags!=0];
            i.orders <- i.orders[lags!=0];
            ma.orders <- ma.orders[lags!=0];
            lags <- lags[lags!=0];
        }

        if(any(lags>48) & (smoothType=="ssarima")){
            warning(paste0("SSARIMA is quite slow with lags greater than 48. ",
                           "It is recommended to use MSARIMA in this case instead."),
                    call.=FALSE);
        }

        # Define maxorder and make all the values look similar (for the polynomials)
        maxorder <- max(length(ar.orders),length(i.orders),length(ma.orders));
        if(length(ar.orders)!=maxorder){
            ar.orders <- c(ar.orders,rep(0,maxorder-length(ar.orders)));
        }
        if(length(i.orders)!=maxorder){
            i.orders <- c(i.orders,rep(0,maxorder-length(i.orders)));
        }
        if(length(ma.orders)!=maxorder){
            ma.orders <- c(ma.orders,rep(0,maxorder-length(ma.orders)));
        }

        if((length(lags)!=length(ar.orders)) & (length(lags)!=length(i.orders)) & (length(lags)!=length(ma.orders))){
            stop("Seasonal lags do not correspond to any element of SARIMA",call.=FALSE);
        }

        # If zeroes are defined for some orders, drop them.
        if(any((ar.orders + i.orders + ma.orders)==0)){
            orders2leave <- (ar.orders + i.orders + ma.orders)!=0;
            if(all(!orders2leave)){
                orders2leave <- lags==min(lags);
            }
            ar.orders <- ar.orders[orders2leave];
            i.orders <- i.orders[orders2leave];
            ma.orders <- ma.orders[orders2leave];
            lags <- lags[orders2leave];
        }

        # Get rid of duplicates in lags
        if(length(unique(lags))!=length(lags)){
            if(dataFreq!=1){
                warning(paste0("'lags' variable contains duplicates: (",paste0(lags,collapse=","),
                               "). Getting rid of some of them."),call.=FALSE);
            }
            lagsNew <- unique(lags);
            arOrdersNew <- iOrdersNew <- maOrdersNew <- lagsNew;
            for(i in 1:length(lagsNew)){
                arOrdersNew[i] <- max(ar.orders[which(lags==lagsNew[i])]);
                iOrdersNew[i] <- max(i.orders[which(lags==lagsNew[i])]);
                maOrdersNew[i] <- max(ma.orders[which(lags==lagsNew[i])]);
            }
            ar.orders <- arOrdersNew;
            i.orders <- iOrdersNew;
            ma.orders <- maOrdersNew;
            lags <- lagsNew;
        }

        ARValue <- AR;
        # Check the provided AR matrix / vector
        if(!is.null(ARValue)){
            if((!is.numeric(ARValue) | !is.vector(ARValue)) & !is.matrix(ARValue)){
                warning(paste0("AR should be either vector or matrix. You have provided something strange...\n",
                               "AR will be estimated."),call.=FALSE);
                ARRequired <- AREstimate <- TRUE;
                ARValue <- NULL;
            }
            else{
                if(is.matrix(ARValue)){
                    ARLength <- length(ARValue[ARValue!=0]);
                }
                else{
                    ARLength <- length(ARValue);
                }
                if(sum(ar.orders)!=ARLength){
                    warning(paste0("Wrong number of non-zero elements of AR. Should be ",sum(ar.orders),
                                   " instead of ",length(ARValue[ARValue!=0]),".\n",
                                   "AR will be estimated."),call.=FALSE);
                    ARRequired <- AREstimate <- TRUE;
                    ARValue <- NULL;
                }
                else{
                    if(is.matrix(ARValue)){
                        ARValue <- ARValue[ARValue!=0];
                    }
                    AREstimate <- FALSE;
                    ARRequired <- TRUE;
                    parametersNumber[2,1] <- parametersNumber[2,1] + length(ARValue);
                }
            }
        }
        else{
            if(all(ar.orders==0)){
                ARRequired <- AREstimate <- FALSE;
            }
            else{
                ARRequired <- AREstimate <- TRUE;
            }
        }

        MAValue <- MA;
        # Check the provided MA matrix / vector
        if(!is.null(MAValue)){
            if((!is.numeric(MAValue) | !is.vector(MAValue)) & !is.matrix(MAValue)){
                warning(paste0("MA should be either vector or matrix. You have provided something strange...\n",
                               "MA will be estimated."),call.=FALSE);
                MARequired <- MAEstimate <- TRUE;
                MAValue <- NULL;
            }
            else{
                if(is.matrix(MAValue)){
                    MALength <- length(MAValue[MAValue!=0]);
                }
                else{
                    MALength <- length(MAValue);
                }
                if(sum(ma.orders)!=MALength){
                    warning(paste0("Wrong number of non-zero elements of MA. Should be ",sum(ma.orders),
                                   " instead of ",length(MAValue[MAValue!=0]),".\n",
                                   "MA will be estimated."),call.=FALSE);
                    MARequired <- MAEstimate <- TRUE;
                    MAValue <- NULL;
                }
                else{
                    if(is.matrix(MAValue)){
                        MAValue <- MAValue[MAValue!=0];
                    }
                    MAEstimate <- FALSE;
                    MARequired <- TRUE;
                    parametersNumber[2,1] <- parametersNumber[2,1] + length(MAValue);
                }
            }
        }
        else{
            if(all(ma.orders==0)){
                MARequired <- MAEstimate <- FALSE;
            }
            else{
                MARequired <- MAEstimate <- TRUE;
            }
        }

        constantValue <- constant;
        # Check the provided constant
        if(is.numeric(constantValue)){
            constantEstimate <- FALSE;
            constantRequired <- TRUE;
            parametersNumber[2,1] <- parametersNumber[2,1] + 1;
        }
        else if(is.logical(constantValue)){
            constantRequired <- constantEstimate <- constantValue;
            constantValue <- NULL;
        }

        # Number of components to use
        nComponents <- max(ar.orders %*% lags + i.orders %*% lags,ma.orders %*% lags);
        # If there is no constant and there are no orders
        if(nComponents==0 & !constantRequired){
            constantValue <- 0;
            constantRequired[] <- TRUE
            nComponenst <- 1;
        }

        nonZeroARI <- matrix(1,ncol=2);
        nonZeroMA <- matrix(1,ncol=2);
        lagsModel <- matrix(rep(1,nComponents),ncol=1);

        if(constantRequired){
            lagsModel <- rbind(lagsModel,1);
        }
        lagsModelMax <- 1;

        if(obsInSample < nComponents){
            warning(paste0("In-sample size is ",obsInSample,", while number of components is ",nComponents,
                           ". Cannot fit the model."),call.=FALSE)
            stop("Not enough observations for such a complicated model.",call.=FALSE);
        }
    }
    else if(smoothType=="ces"){
        # If the user typed wrong seasonality, use the "Full" instead
        if(all(seasonality!=c("n","s","p","f","none","simple","partial","full"))){
            warning(paste0("Wrong seasonality type: '",seasonality, "'. Changing to 'full'"), call.=FALSE);
            seasonality <- "f";
        }
        seasonality <- substring(seasonality[1],1,1);
    }

    if(any(smoothType==c("es","oes"))){
        modelIsSeasonal <- Stype!="N";
        # Check if the data is ts-object
        if(!is.ts(y) & modelIsSeasonal){
            if(!silentText){
                message("The provided data is not ts object. Only non-seasonal models are available.");
            }
            Stype <- "N";
            modelIsSeasonal <- FALSE;
            substr(model,nchar(model),nchar(model)) <- "N";
        }

        ### Check seasonality type
        if(all(Stype!=c("Z","X","Y","N","A","M","C"))){
            warning(paste0("Wrong seasonality type: ",Stype,". Should be 'Z', 'X', 'Y', 'N', 'A' or 'M'.",
                           "Setting to 'Z'."),call.=FALSE);
            if(dataFreq==1){
                Stype <- "N";
                modelIsSeasonal <- FALSE;
            }
            else{
                Stype <- "Z";
                modelDo <- "select";
            }
        }
        if(all(modelIsSeasonal,dataFreq==1)){
            if(all(Stype!=c("Z","X","Y"))){
                warning(paste0("Cannot build the seasonal model on data with frequency 1.\n",
                               "Switching to non-seasonal model: ETS(",substring(model,1,nchar(model)-1),"N)"));
            }
            Stype <- "N";
            modelIsSeasonal <- FALSE;
        }

        # Check the pool of models to combine if it was decided that the data is not seasonal
        if(!modelIsSeasonal && !is.null(modelsPool)){
            modelsPool <- modelsPool[substr(modelsPool,nchar(modelsPool),nchar(modelsPool))=="N"];
        }
    }
    else if(smoothType=="sma"){
        lagsModelMax <- 1;
        if(is.null(order)){
            nParamMax <- obsInSample;
        }
        else{
            nParamMax <- order;
        }
    }
    else{
        lagsModelMax <- 1;
        nParamMax <- 0;
    }

    # Make sure that y is ts
    if(!is.ts(y)){
        y <- ts(y);
    }

    ##### Lags and components for GUM #####
    if(smoothType=="gum"){
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
            lagsNew <- unique(lags);
            ordersNew <- lagsNew;
            for(i in 1:length(lagsNew)){
                ordersNew[i] <- max(orders[which(lags==lagsNew[i])]);
            }
            orders <- ordersNew;
            lags <- lagsNew;
        }

        lagsModel <- matrix(rep(lags,times=orders),ncol=1);
        lagsModelMax <- max(lagsModel);
        nComponents <- sum(orders);

        type <- substr(type[1],1,1);
        if(type=="m"){
            if(any(yInSample<=0)){
                warning("Multiplicative model can only be used on positive data. Switching to the additive one.",
                        call.=FALSE);
                modelIsMultiplicative <- FALSE;
                type <- "a";
            }
            else{
                yInSample <- log(yInSample);
                modelIsMultiplicative <- TRUE;
            }
        }
        else{
            modelIsMultiplicative <- FALSE;
        }
    }
    else if(any(smoothType==c("es","oes"))){
        lagsModelMax <- dataFreq * modelIsSeasonal + 1 * (!modelIsSeasonal);
    }
    else if(smoothType=="ces"){
        a <- list(value=a);
        b <- list(value=b);

        if(is.null(a$value)){
            a$estimate <- TRUE;
        }
        else{
            a$estimate <- FALSE;
            if(!is.null(a$value)){
                parametersNumber[2,1] <- parametersNumber[2,1] + length(Re(a$value)) + length(Im(a$value));
            }
        }
        if(all(is.null(b$value),any(seasonality==c("p","f")))){
            b$estimate <- TRUE;
        }
        else{
            b$estimate <- FALSE;
            if(!is.null(b$value)){
                parametersNumber[2,1] <- parametersNumber[2,1] + length(Re(b$value)) + length(Im(b$value));
            }
        }

        # Define lags, number of components and number of parameters
        if(seasonality=="n"){
            # No seasonality
            lagsModelMax <- 1;
            lagsModel <- c(1,1);
            # Define the number of all the parameters (smoothing parameters + initial states). Used in AIC mainly!
            nComponents <- 2;
            a$number <- 2;
            b$number <- 0;
        }
        else if(seasonality=="s"){
            # Simple seasonality, lagged CES
            lagsModelMax <- dataFreq;
            lagsModel <- c(lagsModelMax,lagsModelMax);
            nComponents <- 2;
            a$number <- 2;
            b$number <- 0;
        }
        else if(seasonality=="p"){
            # Partial seasonality with a real part only
            lagsModelMax <- dataFreq;
            lagsModel <- c(1,1,lagsModelMax);
            nComponents <- 3;
            a$number <- 2;
            b$number <- 1;
        }
        else if(seasonality=="f"){
            # Full seasonality with both real and imaginary parts
            lagsModelMax <- dataFreq;
            lagsModel <- c(1,1,lagsModelMax,lagsModelMax);
            nComponents <- 4;
            a$number <- 2;
            b$number <- 2;
        }
    }

    #### This is a temporary thing. If the function works, we will do that properly ####
    else if(smoothType=="msarima"){
        # Define the non-zero values. This is done via the calculation of orders of polynomials
        ariValues <- list(NA);
        maValues <- list(NA);
        for(i in 1:length(lags)){
            ariValues[[i]] <- c(0,min(1,ar.orders[i]):ar.orders[i])
            if(i.orders[i]!=0){
                ariValues[[i]] <- c(ariValues[[i]],1:i.orders[i]+ar.orders[i]);
            }
            ariValues[[i]] <- unique(ariValues[[i]] * lags[i]);
            maValues[[i]] <- unique(c(0,min(1,ma.orders[i]):ma.orders[i]) * lags[i]);
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
        nonZeroComponents <- sort(unique(c(nonZeroARI,nonZeroMA)));
        nonZeroARI <- cbind(nonZeroARI,which(nonZeroComponents %in% nonZeroARI)-1);
        nonZeroMA <- cbind(nonZeroMA,which(nonZeroComponents %in% nonZeroMA)-1);

        nComponents <- length(nonZeroComponents);

        if(obsInSample < nComponents){
            warning(paste0("In-sample size is ",obsInSample,", while number of components is ",nComponents,
                           ". Cannot fit the model."),call.=FALSE)
            stop("Not enough observations for such a complicated model.",call.=FALSE);
        }

        lagsModel <- matrix(nonZeroComponents,ncol=1);

        if(constantRequired){
            lagsModel <- rbind(lagsModel,1);
        }
        lagsModelMax <- max(lagsModel);
    }

    ##### obsStates #####
    # Define the number of rows that should be in the matvt
    obsStates <- max(obsAll + lagsModelMax, obsInSample + 2*lagsModelMax);

    ##### bounds #####
    bounds <- substring(bounds[1],1,1);
    if(all(bounds!=c("n","a","r","u"))){
        warning("Strange bounds are defined. Switching to 'admissible'.",call.=FALSE);
        bounds <- "a";
    }

    if(bounds=="n"){
        warning("You have defined bounds='none'. ",
                "This is dangerous and might lead to an unstable model or even break the function. ",
                "Hopefully, you know what you are doing :).",
                call.=FALSE);
    }

    ##### Information Criteria #####
    ic <- ic[1];
    if(all(ic!=c("AICc","AIC","BIC","BICc"))){
        warning(paste0("Strange type of information criteria defined: ",ic,". Switching to 'AICc'."),
                call.=FALSE);
        ic <- "AICc";
    }

    ##### Loss function type #####
    loss <- loss[1];
    if(any(loss==c("MSEh","TMSE","GTMSE","MSCE","MAEh","TMAE","GTMAE","MACE",
                     "HAMh","THAM","GTHAM","CHAM",
                     "GPL","aMSEh","aTMSE","aGTMSE","aGPL"))){
        multisteps <- TRUE;
    }
    else if(any(loss==c("likelihood","MSE","MAE","HAM","Rounded"))){
        multisteps <- FALSE;
    }
    else{
        if(loss=="MSTFE"){
            warning(paste0("This estimator has recently been renamed from \"MSTFE\" to \"TMSE\". ",
                           "Please, use the new name."),call.=FALSE);
            multisteps <- TRUE;
            loss <- "TMSE";
        }
        else if(loss=="GMSTFE"){
            warning(paste0("This estimator has recently been renamed from \"GMSTFE\" to \"GTMSE\". ",
                           "Please, use the new name."),call.=FALSE);
            multisteps <- TRUE;
            loss <- "GTMSE";
        }
        else if(loss=="aMSTFE"){
            warning(paste0("This estimator has recently been renamed from \"aMSTFE\" to \"aTMSE\". ",
                           "Please, use the new name."),call.=FALSE);
            multisteps <- TRUE;
            loss <- "aTMSE";
        }
        else if(loss=="aGMSTFE"){
            warning(paste0("This estimator has recently been renamed from \"aGMSTFE\" to \"aGTMSE\". ",
                           "Please, use the new name."),call.=FALSE);
            multisteps <- TRUE;
            loss <- "aGTMSE";
        }
        else{
            warning(paste0("Strange loss function specified: ",loss,". Switching to 'MSE'."),call.=FALSE);
            loss <- "MSE";
            multisteps <- FALSE;
        }
    }
    lossOriginal <- loss;

    ##### interval, intervalType, level #####
    #intervalType <- substring(intervalType[1],1,1);
    intervalType <- interval[1];
    # Check the provided type of interval

    if(is.logical(intervalType)){
        if(intervalType){
            intervalType <- "p";
        }
        else{
            intervalType <- "none";
        }
    }

    if(all(intervalType!=c("p","l","s","n","a","sp","np","none","parametric","likelihood","semiparametric","nonparametric"))){
        warning(paste0("Wrong type of interval: '",intervalType, "'. Switching to 'parametric'."),call.=FALSE);
        intervalType <- "p";
    }

    if(any(intervalType==c("none","n"))){
        intervalType <- "n";
        interval <- FALSE;
    }
    else if(any(intervalType==c("parametric","p"))){
        intervalType <- "p";
        interval <- TRUE;
    }
    else if(any(intervalType==c("semiparametric","sp"))){
        intervalType <- "sp";
        interval <- TRUE;
    }
    else if(any(intervalType==c("nonparametric","np"))){
        intervalType <- "np";
        interval <- TRUE;
    }
    else if(any(intervalType==c("likelihood","l"))){
        intervalType <- "l";
        interval <- TRUE;
    }
    else{
        interval <- TRUE;
    }

    if(level>1){
        level <- level / 100;
    }

    ##### Occurrence part of the model #####
    if(inherits(occurrence,"oes")){
        occurrenceModel <- occurrence;
        occurrence <- occurrenceModel$occurrence;
        occurrenceModelProvided <- TRUE;
    }
    else if(is.list(occurrence)){
        warning(paste0("occurrence is not of the class oes. ",
                       "We will try to extract the type of model, but cannot promise anything."),
                call.=FALSE);
        occurrenceModel <- modelType(occurrence);
        occurrence <- occurrence$occurrence;
        occurrenceModelProvided <- FALSE;
    }
    else if(is.null(occurrence)){
        occurrence <- "none";
        occurrenceModel <- "MNN";
        occurrenceModelProvided <- FALSE;
    }
    else{
        if(is.null(oesmodel) || is.na(oesmodel)){
            occurrenceModel <- "MNN";
        }
        else{
            occurrenceModel <- oesmodel;
        }
        occurrenceModelProvided <- FALSE;
    }

    if(smoothType!="oes"){
        if(is.numeric(occurrence)){
            # If it is data, then it should either correspond to the whole sample (in-sample + holdout)
            # or be equal to forecating horizon.
            if(any(occurrence!=1) && all(length(c(occurrence))!=c(h,obsAll))){
                warning(paste0("Length of the provided future occurrences is ",length(c(occurrence)),
                               " while length of forecasting horizon is ",h,".\n",
                               "Where should we plug in the future occurences anyway?\n",
                               "Switching to occurrence='fixed'."),call.=FALSE);
                occurrence <- "f";
                ot <- (yInSample!=0)*1;
                obsNonzero <- sum(ot);
                obsZero <- obsInSample - obsNonzero;
                yot <- matrix(yInSample[yInSample!=0],obsNonzero,1);
                pFitted <- matrix(mean(ot),obsInSample,1);
                pForecast <- matrix(1,h,1);
                nParamOccurrence <- 1;
            }
            else if(all(occurrence==1)){
                obsNonzero <- obsInSample;
                obsZero <- 0;
                pFitted <- ot <- rep(1,obsInSample);
                yot <- yInSample;
                pForecast <- matrix(1,h,1);
                nParamOccurrence <- 0;
                occurrence <- "n";
                occurrenceModelProvided <- FALSE;
            }
            else{
                if(any(occurrence<0,occurrence>1)){
                    warning(paste0("Parameter 'occurrence' should contain values between zero and one.\n",
                                   "Converting to appropriate vector."),call.=FALSE);
                    occurrence <- (occurrence!=0)*1;
                }

                ot <- (yInSample!=0)*1;
                obsNonzero <- sum(ot);
                obsZero <- obsInSample - obsNonzero;
                yot <- matrix(yInSample[yInSample!=0],obsNonzero,1);
                if(length(occurrence)==obsAll){
                    pFitted <- occurrence[1:obsInSample];
                    pForecast <- occurrence[(obsInSample+1):(obsInSample+h)];
                }
                else{
                    pFitted <- matrix(ot,obsInSample,1);
                    pForecast <- matrix(occurrence,h,1);
                }

                # "p" stand for "provided", meaning that we have been provided the future data
                occurrence <- "p";
                nParamOccurrence <- 0;
            }
        }
        else{
            occurrence <- occurrence[1];
            if(all(occurrence!=c("n","a","f","g","o","i","d",
                                 "none","auto","fixed","general","odds-ratio","inverse-odds-ratio","direct"))){
                warning(paste0("Strange type of occurrence model defined: '",occurrence,
                               "'. Switching to 'fixed'."),
                        call.=FALSE);
                occurrence <- "f";
            }
            occurrence <- substring(occurrence[1],1,1);

            environment(intermittentParametersSetter) <- environment();
            intermittentParametersSetter(occurrence,ParentEnvironment=environment());
        }

        # If the data is not occurrence, let's assume that the parameter was switched unintentionally.
        if(all(ot==1) & all(occurrence!=c("n","p","provided"))){
            occurrence <- "n";
            occurrenceModelProvided <- FALSE;
        }

        if(occurrenceModelProvided){
            parametersNumber[2,3] <- nparam(occurrenceModel);
        }
    }
    else{
        obsNonzero <- obsInSample;
        obsZero <- 0;
    }

    if(any(smoothType==c("es"))){
        # Check if multiplicative models can be fitted
        allowMultiplicative <- !((any(yInSample<=0) && occurrence=="n") |
                                     (occurrence!="n" && any(yInSample<0)));
        # If non-positive values are present, check if data is intermittent,
        # if negatives are here, switch to additive models
        if(!allowMultiplicative){
            if(any(Etype==c("M","Y"))){
                warning(paste0("Can't apply multiplicative model to non-positive data. ",
                               "Switching error type to 'A'"), call.=FALSE);
                Etype <- ifelse(Etype=="M","A","X");
            }
            if(any(Ttype==c("M","Y"))){
                warning(paste0("Can't apply multiplicative model to non-positive data. ",
                               "Switching trend type to 'A'"), call.=FALSE);
                Ttype <- ifelse(Ttype=="M","A","X");
            }
            if(any(Stype==c("M","Y"))){
                warning(paste0("Can't apply multiplicative model to non-positive data. ",
                               "Switching seasonality type to 'A'"), call.=FALSE);
                Stype <- ifelse(Stype=="M","A","X");
            }

            if(!is.null(modelsPool)){
                if(any(c(substr(modelsPool,1,1),
                         substr(modelsPool,2,2),
                         substr(modelsPool,nchar(modelsPool),nchar(modelsPool)))=="M")){
                    warning(paste0("Can't apply multiplicative model to non-positive data. ",
                                   "Switching to additive."), call.=FALSE);
                    substr(modelsPool,1,1)[substr(modelsPool,1,1)=="M"] <- "A";
                    substr(modelsPool,2,2)[substr(modelsPool,2,2)=="M"] <- "A";
                    substr(modelsPool,nchar(modelsPool),
                           nchar(modelsPool))[substr(modelsPool,nchar(modelsPool),
                                                     nchar(modelsPool))=="M"] <- "A";
                }
            }
        }

        # Check the pool of models. Make sure that there are no duplicates
        if(!is.null(modelsPool)){
            modelsPool <- unique(modelsPool);
            if(is.null(modelsPool)){
                stop(paste0("Cannot combine the models, your pool is empty. ",
                            "Please, check the vector of models you provided."),
                     call.=FALSE);
            }
        }
    }

    if(any(smoothType==c("es","gum","oes"))){
        ##### persistence for ES & GUM #####
        if(!is.null(persistence)){
            if((!is.numeric(persistence) | !is.vector(persistence)) & !is.matrix(persistence)){
                warning(paste0("Persistence is not a numeric vector!\n",
                               "Changing to estimation of persistence vector values."),call.=FALSE);
                persistence <- NULL;
                persistenceEstimate <- TRUE;
            }
            else{
                if(any(smoothType==c("es","oes"))){
                    if(modelDo!="estimate"){
                        warning(paste0("Predefined persistence vector can only be used with ",
                                       "preselected ETS model.\n",
                                       "Changing to estimation of persistence vector values."),call.=FALSE);
                        persistence <- NULL;
                        persistenceEstimate <- TRUE;
                    }
                    else{
                        if(length(persistence)>3){
                            warning(paste0("Length of persistence vector is wrong! ",
                                           "It should not be greater than 3.\n",
                                           "Changing to estimation of persistence vector values."),
                                    call.=FALSE);
                            persistence <- NULL;
                            persistenceEstimate <- TRUE;
                        }
                        else{
                            if(length(persistence)!=(1 + (Ttype!="N") + (modelIsSeasonal))){
                                warning(paste0("Wrong length of persistence vector. ",
                                               "Should be ",(1 + (Ttype!="N") + (modelIsSeasonal)),
                                               " instead of ",length(persistence),".\n",
                                               "Changing to estimation of persistence vector values."),
                                        call.=FALSE);
                                persistence <- NULL;
                                persistenceEstimate <- TRUE;
                            }
                            else{
                                persistence <- as.vector(persistence);
                                persistenceEstimate <- FALSE;
                                parametersNumber[2,1] <- parametersNumber[2,1] + length(persistence);
                                # bounds <- "n";
                            }
                        }
                    }
                }
                else if(smoothType=="gum"){
                    if(length(persistence) != nComponents){
                        warning(paste0("Wrong length of persistence vector. Should be ",nComponents,
                                       " instead of ",length(persistence),".\n",
                                       "Changing to estimation of persistence vector values."),call.=FALSE);
                        persistence <- NULL;
                        persistenceEstimate <- TRUE;
                    }
                    else{
                        persistenceEstimate <- FALSE;
                        parametersNumber[2,1] <- parametersNumber[2,1] + length(persistence);
                    }
                }
            }
        }
        else{
            persistenceEstimate <- TRUE;
        }
    }

    ##### initials ####
    # initial type can be: "o" - optimal, "b" - backcasting, "p" - provided.
    initialValue <- initial;
    if(is.character(initialValue)){
        initialValue <- substring(initialValue[1],1,1);
        if(all(initialValue!=c("o","b","p"))){
            warning("You asked for a strange initial value. We don't do that here. Switching to optimal.",
                    call.=FALSE);
            initialType <- "o";
        }
        else{
            # if(smoothType=="msarima" & initialValue=="o"){
            #     initialValue <- "b";
            #     warning(paste0("We don't support optimisation of the initial states of MSARIMA. ",
            #                    "Switching to 'backcasting'."),
            #             call.=FALSE);
            # }
            initialType <- initialValue;
        }
        initialValue <- NULL;
    }
    else if(is.null(initialValue)){
        if(silentText){
            message("Initial value is not selected. Switching to optimal.");
        }
        initialType <- "o";
    }
    else if(!is.null(initialValue)){
        if(!is.numeric(initialValue)){
            warning(paste0("Initial vector is not numeric!\n",
                           "Values of initial vector will be estimated."),call.=FALSE);
            initialValue <- NULL;
            initialType <- "o";
        }
        else{
            if(any(smoothType==c("es","oes"))){
                if(modelDo!="estimate"){
                    warning(paste0("Predefined initials vector can only be used with preselected ETS model.\n",
                                   "Changing to estimation of initials."),call.=FALSE);
                    initialValue <- NULL;
                    initialType <- "o";
                }
                else{
                    if(length(initialValue)>2){
                        warning(paste0("Length of initial vector is wrong! It should not be greater than 2.\n",
                                       "Values of initial vector will be estimated."),call.=FALSE);
                        initialValue <- NULL;
                        initialType <- "o";
                    }
                    else{
                        if(length(initialValue) != (1*(Ttype!="N") + 1)){
                            warning(paste0("Length of initial vector is wrong! It should be ",
                                           (1*(Ttype!="N") + 1),
                                           " instead of ",length(initialValue),".\n",
                                           "Values of initial vector will be estimated."),call.=FALSE);
                            initialValue <- NULL;
                            initialType <- "o";
                        }
                        else{
                            initialType <- "p";
                            # initialValue <- initial;
                            parametersNumber[2,1] <- parametersNumber[2,1] + length(initial);
                        }
                    }
                }
            }
            else if(smoothType=="gum"){
                if(length(initialValue) != (nComponents*max(lags))){
                    warning(paste0("Wrong length of initial vector. Should be ",(nComponents*max(lags)),
                                   " instead of ",length(initial),".\n",
                                   "Values of initial vector will be estimated."),call.=FALSE);
                    initialValue <- NULL;
                    initialType <- "o";
                }
                else{
                    initialType <- "p";
                    initialValue <- initial;
                    parametersNumber[2,1] <- parametersNumber[2,1] + orders %*% lags;
                }
            }
            else if(smoothType=="ssarima"){
                if(length(initialValue) != nComponents){
                    warning(paste0("Wrong length of initial vector. Should be ",nComponents,
                                   " instead of ",length(initial),".\n",
                                   "Values of initial vector will be estimated."),call.=FALSE);
                    initialValue <- NULL;
                    initialType <- "o";
                }
                else{
                    initialType <- "p";
                    initialValue <- initial;
                    parametersNumber[2,1] <- parametersNumber[2,1] + length(initial);
                }
            }
            else if(smoothType=="msarima"){
                if(length(initialValue) != nComponents*lagsModelMax){
                    warning(paste0("Wrong length of initial vector. Should be ",nComponents,
                                   " instead of ",length(initial),".\n",
                                   "Values of initial vector will be backcasted."),call.=FALSE);
                    initialValue <- NULL;
                    initialType <- "b";
                }
                else{
                    initialType <- "p";
                    # initialValue <- initial;
                    parametersNumber[2,1] <- parametersNumber[2,1] + length(initial);
                }
            }
            else if(smoothType=="ces"){
                if(length(initialValue) != lagsModelMax*nComponents){
                    warning(paste0("Wrong length of initial vector. Should be ",lagsModelMax*nComponents,
                                   " instead of ",length(initial),".\n",
                                   "Values of initial vector will be estimated."),call.=FALSE);
                    initialValue <- NULL;
                    initialType <- "o";
                }
                else{
                    initialType <- "p";
                    # initialValue <- initial;
                    parametersNumber[2,1] <- (parametersNumber[2,1] + 2*(seasonality!="s") +
                                                  lagsModelMax*(seasonality!="n") +
                                                  lagsModelMax*any(seasonality==c("f","s")));
                }
            }
            else if(smoothType=="smoothC"){
                warning("We cannot use the preset initials for the models. Switching to optimal.",
                        call.=FALSE);
                initialType <- "o";
                initialValue <- NULL;
            }
        }
    }

    if(any(smoothType==c("es","oes"))){
        # If model selection is chosen, forget about the initial values and persistence
        if(any(Etype=="Z",any(Ttype==c("X","Y","Z")),Stype=="Z")){
            if(any(!is.null(initialValue),!is.null(initialSeason),!is.null(persistence),!is.null(phi))){
                warning(paste0("Model selection doesn't go well with the predefined values.\n",
                               "Switching to estimation of all the parameters."),call.=FALSE);
                initialValue <- NULL;
                initialType <- "o";
                initialSeason <- NULL;
                persistence <- NULL;
                phi <- NULL;
            }
        }

        ##### initialSeason for ES #####
        if(!is.null(initialSeason)){
            if(!is.numeric(initialSeason)){
                warning(paste0("InitialSeason vector is not numeric!\n",
                               "Values of initialSeason vector will be estimated."),call.=FALSE);
                initialSeason <- NULL;
                initialSeasonEstimate <- TRUE;
            }
            else{
                if(length(initialSeason)!=dataFreq){
                    warning(paste0("The length of initialSeason vector is wrong! ",
                                   "It should correspond to the frequency of the data.\n",
                                   "Values of initialSeason vector will be estimated."),call.=FALSE);
                    initialSeason <- NULL;
                    initialSeasonEstimate <- TRUE
                }
                else{
                    initialSeasonEstimate <- FALSE;
                    parametersNumber[2,1] <- parametersNumber[2,1] + length(initialSeason);
                }
            }
        }
        else{
            initialSeasonEstimate <- TRUE;
        }

        # Check the length of the provided data. Say bad words if:
        # 1. Seasonal model, <2 seasons of data and no initial seasonals.
        # 2. Seasonal model, <1 season of data, no initial seasonals and no persistence.
        if(is.null(modelsPool)){
            if((modelIsSeasonal & (obsInSample < 2*dataFreq) & is.null(initialSeason)) |
               (modelIsSeasonal & (obsInSample < dataFreq) & is.null(initialSeason) & is.null(persistence))){
                if(is.null(initialSeason)){
                    warning(paste0("Sorry, but we don't have enough observations for the seasonal model!\n",
                                   "Switching to non-seasonal."),call.=FALSE);
                    Stype <- "N";
                    modelIsSeasonal <- FALSE;
                    initialSeasonEstimate <- FALSE;
                }
            }
        }

        ##### phi for ES #####
        if(!is.null(phi)){
            if(!is.numeric(phi) & (damped)){
                warning(paste0("Provided value of phi is meaningless. phi will be estimated."),
                        call.=FALSE);
                phi <- NULL;
                phiEstimate <- TRUE;
            }
            else if(is.numeric(phi) & (phi<0 | phi>2)){
                warning(paste0("Damping parameter should lie in (0, 2) region. ",
                               "Changing to the estimation of phi."),call.=FALSE);
                phi <- NULL;
                phiEstimate <- TRUE;
            }
            else{
                phiEstimate <- FALSE;
                if(damped){
                    parametersNumber[2,1] <- parametersNumber[2,1] + 1;
                }
            }
        }
        else{
            if(damped){
                phiEstimate <- TRUE;
            }
            else{
                phiEstimate <- FALSE;
            }
        }
    }

    if(smoothType=="gum"){
        ##### transition for GUM #####
        # Check the provided vector of initials: length and provided values.
        if(!is.null(transition)){
            if((!is.numeric(transition) | !is.vector(transition)) & !is.matrix(transition)){
                warning(paste0("Transition matrix is not numeric!\n",
                               "The matrix will be estimated!"),call.=FALSE);
                transitionEstimate <- TRUE;
            }
            else if(length(transition) != nComponents^2){
                warning(paste0("Wrong length of transition matrix. Should be ",nComponents^2,
                               " instead of ",length(transition),".\n",
                               "The matrix will be estimated!"),call.=FALSE);
                transitionEstimate <- TRUE;
            }
            else{
                transitionEstimate <- FALSE;
                parametersNumber[2,1] <- parametersNumber[2,1] + length(transition);
            }
        }
        else{
            transitionEstimate <- TRUE;
        }

        ##### measurement for GUM #####
        if(!is.null(measurement)){
            if((!is.numeric(measurement) | !is.vector(measurement)) & !is.matrix(measurement)){
                warning(paste0("Measurement vector is not numeric!\n",
                               "The vector will be estimated!"),call.=FALSE);
                measurementEstimate <- TRUE;
            }
            else if(length(measurement) != nComponents){
                warning(paste0("Wrong length of measurement vector. Should be ",nComponents,
                               " instead of ",length(measurement),".\n",
                               "The vector will be estimated!"),call.=FALSE);
                measurementEstimate <- TRUE;
            }
            else{
                measurementEstimate <- FALSE;
                parametersNumber[2,1] <- parametersNumber[2,1] + length(measurement);
            }
        }
        else{
            measurementEstimate <- TRUE;
        }
    }

    if(smoothType=="ssarima"){
        if((nComponents==0) & (!constantRequired)){
            if(!silentText){
                warning("You have not defined any model! Constructing model with zero constant.",
                        call.=FALSE);
            }
            constantRequired <- TRUE;
            constantValue <- 0;
            initialType <- "p";
        }
    }

    ##### Calculate nParamMax for checks #####
    if(any(smoothType==c("es","oes"))){
        # 1: estimation of variance;
        # 1 - 3: persitence vector;
        # 1 - 2: initials;
        # 1 - 1 phi value;
        # dataFreq: dataFreq initials for seasonal component;
        nParamMax <- (1 + (1 + (Ttype!="N") + (modelIsSeasonal))*persistenceEstimate +
                          (1 + (Ttype!="N"))*(initialType=="o") + phiEstimate*damped +
                          dataFreq*(modelIsSeasonal)*initialSeasonEstimate*(initialType!="b"));
    }
    else if(smoothType=="gum"){
        nParamMax <- (1 + nComponents*measurementEstimate + nComponents*persistenceEstimate +
                          (nComponents^2)*transitionEstimate + (orders %*% lags)*(initialType=="o"));
    }
    else if(smoothType=="ssarima"){
        nParamMax <- (1 + nComponents*(initialType=="o") + sum(ar.orders)*ARRequired*AREstimate +
                          sum(ma.orders)*MARequired*MAEstimate + constantRequired*constantEstimate);
    }
    else if(smoothType=="ces"){
        nParamMax <- (1 + sum(lagsModel)*(initialType=="o") + a$number*a$estimate + b$number*b$estimate);
    }

    # Stop if number of observations is less than horizon and multisteps is chosen.
    if((multisteps) & (obsNonzero < h+1) & all(loss!=c("aMSEh","aTMSE","aGTMSE","aGPL"))){
        warning(paste0("Do you seriously think that you can use ",loss,
                       " with h=",h," on ",obsNonzero," non-zero observations?!"),call.=FALSE);
        stop("Not enough observations for multisteps loss function.",call.=FALSE);
    }
    else if((multisteps) & (obsNonzero < 2*h) & all(loss!=c("aMSEh","aTMSE","aGTMSE","aGPL"))){
        warning(paste0("Number of observations is really low for a multisteps loss function! ",
                       "We will, try but cannot guarantee anything..."),call.=FALSE);
    }

    normalizer <- mean(abs(diff(c(yInSample))));

    ##### Define regressors #####
    if(smoothType!="sma"){
        if(!any(regressors==c("use","select","u","s"))){
            warning("Wrong type of regressors parameter. Changing to 'select'.", call.=FALSE);
            regressors <- "select";
        }
        regressors <- substr(regressors[1],1,1);
    }

    if(is.null(xreg)){
        regressors <- "u";
    }

    ##### Fisher Information #####
    if(!exists("FI",envir=ParentEnvironment,inherits=FALSE)){
        FI <- FALSE;
    }
    else{
        if(!is.logical(FI)){
            FI <- FALSE;
        }
        if(!requireNamespace("numDeriv",quietly=TRUE) & FI){
            warning(paste0("Sorry, but you don't have 'numDeriv' package, ",
                           "which is required in order to produce Fisher Information."),
                    call.=FALSE);
            FI <- FALSE;
        }
    }

    ##### Rounded up values #####
    if(!exists("rounded",envir=ParentEnvironment,inherits=FALSE)){
        rounded <- FALSE;
    }
    else{
        if(!is.logical(rounded)){
            rounded <- FALSE;
        }
        else{
            if(rounded){
                loss <- "Rounded";
                lossOriginal <- loss;
            }
        }
    }

    ##### Return values to previous environment #####
    assign("h",h,ParentEnvironment);
    assign("holdout",holdout,ParentEnvironment);
    assign("silentText",silentText,ParentEnvironment);
    assign("silentGraph",silentGraph,ParentEnvironment);
    assign("silentLegend",silentLegend,ParentEnvironment);
    assign("obsInSample",obsInSample,ParentEnvironment);
    assign("obsAll",obsAll,ParentEnvironment);
    assign("obsStates",obsStates,ParentEnvironment);
    assign("obsNonzero",obsNonzero,ParentEnvironment);
    assign("obsZero",obsZero,ParentEnvironment);
    assign("y",y,ParentEnvironment);
    assign("yInSample",yInSample,ParentEnvironment);
    assign("dataFreq",dataFreq,ParentEnvironment);
    assign("dataStart",dataStart,ParentEnvironment);
    assign("yForecastStart",yForecastStart,ParentEnvironment);
    assign("bounds",bounds,ParentEnvironment);
    assign("loss",loss,ParentEnvironment);
    assign("lossOriginal",lossOriginal,ParentEnvironment);
    assign("multisteps",multisteps,ParentEnvironment);
    assign("intervalType",intervalType,ParentEnvironment);
    assign("interval",interval,ParentEnvironment);
    assign("initialValue",initialValue,ParentEnvironment);
    assign("initialType",initialType,ParentEnvironment);
    assign("normalizer",normalizer,ParentEnvironment);
    assign("nParamMax",nParamMax,ParentEnvironment);
    assign("regressors",regressors,ParentEnvironment);
    assign("FI",FI,ParentEnvironment);
    assign("rounded",rounded,ParentEnvironment);
    assign("parametersNumber",parametersNumber,ParentEnvironment);
    assign("ic",ic,ParentEnvironment);

    #### intermittent part of the model... outdated ####
    if(smoothType!="oes"){
        assign("occurrence",occurrence,ParentEnvironment);
        assign("occurrenceModel",occurrenceModel,ParentEnvironment);
        assign("ot",ot,ParentEnvironment);
        assign("yot",yot,ParentEnvironment);
        assign("pFitted",pFitted,ParentEnvironment);
        assign("pForecast",pForecast,ParentEnvironment);
        assign("nParamOccurrence",nParamOccurrence,ParentEnvironment);
        assign("occurrenceModelProvided",occurrenceModelProvided,ParentEnvironment);
    }
    else{
        assign("initialSeasonEstimate",initialSeasonEstimate,ParentEnvironment);
        assign("modelIsSeasonal",modelIsSeasonal,ParentEnvironment);
    }

    if(any(smoothType==c("es","oes"))){
        assign("model",model,ParentEnvironment);
        assign("modelsPool",modelsPool,ParentEnvironment);
        assign("Etype",Etype,ParentEnvironment);
        assign("Ttype",Ttype,ParentEnvironment);
        assign("Stype",Stype,ParentEnvironment);
        assign("damped",damped,ParentEnvironment);
        assign("modelDo",modelDo,ParentEnvironment);
        assign("initialSeason",initialSeason,ParentEnvironment);
        assign("phi",phi,ParentEnvironment);
        assign("phiEstimate",phiEstimate,ParentEnvironment);
        if(smoothType=="es"){
            assign("allowMultiplicative",allowMultiplicative,ParentEnvironment);
        }
    }
    else if(smoothType=="gum"){
        assign("transitionEstimate",transitionEstimate,ParentEnvironment);
        assign("measurementEstimate",measurementEstimate,ParentEnvironment);
        assign("orders",orders,ParentEnvironment);
        assign("lags",lags,ParentEnvironment);
        assign("modelIsMultiplicative",modelIsMultiplicative,ParentEnvironment);
    }
    else if(any(smoothType==c("ssarima","msarima"))){
        assign("ar.orders",ar.orders,ParentEnvironment);
        assign("i.orders",i.orders,ParentEnvironment);
        assign("ma.orders",ma.orders,ParentEnvironment);
        assign("lags",lags,ParentEnvironment);
        assign("ARValue",ARValue,ParentEnvironment);
        assign("ARRequired",ARRequired,ParentEnvironment);
        assign("AREstimate",AREstimate,ParentEnvironment);
        assign("MAValue",MAValue,ParentEnvironment);
        assign("MARequired",MARequired,ParentEnvironment);
        assign("MAEstimate",MAEstimate,ParentEnvironment);
        assign("constantValue",constantValue,ParentEnvironment);
        assign("constantEstimate",constantEstimate,ParentEnvironment);
        assign("constantRequired",constantRequired,ParentEnvironment);
        assign("nonZeroARI",nonZeroARI,ParentEnvironment);
        assign("nonZeroMA",nonZeroMA,ParentEnvironment);
    }
    else if(smoothType=="ces"){
        assign("seasonality",seasonality,ParentEnvironment);
        assign("a",a,ParentEnvironment);
        assign("b",b,ParentEnvironment);
    }

    if(any(smoothType==c("es","oes","gum"))){
        assign("persistence",persistence,ParentEnvironment);
        assign("persistenceEstimate",persistenceEstimate,ParentEnvironment);
    }

    if(any(smoothType==c("gum","ssarima","ces","msarima"))){
        assign("nComponents",nComponents,ParentEnvironment);
        assign("lagsModelMax",lagsModelMax,ParentEnvironment);
        assign("lagsModel",lagsModel,ParentEnvironment);
    }
}

##### *State space interval* #####
ssIntervals <- function(errors, ev=median(errors), level=0.95, intervalType=c("a","p","l","sp","np"), df=NULL,
                        measurement=NULL, transition=NULL, persistence=NULL, s2=NULL,
                        lagsModel=NULL, states=NULL, cumulative=FALSE, loss="MSE",
                        yForecast=rep(0,ncol(errors)), Etype="A", Ttype="N", Stype="N", s2g=NULL,
                        probability=1){
    # Function constructs interval based on the provided random variable.
    # If errors is a matrix, then it is assumed that each column has a variable that needs an interval.
    # based on errors the horison is estimated as ncol(errors)

    # Make a correction for the cases, when likelihood is used for the estimation
    if(loss=="likelihood"){
        loss <- "MSE";
    }

    matrixpower <- function(A,n){
        if(n==0){
            return(diag(nrow(A)));
        }
        else if(n==1){
            return(A);
        }
        else if(n>1){
            return(A %*% matrixpower(A, n-1));
        }
    }

    hsmN <- gamma(0.75)*pi^(-0.5)*2^(-0.75);
    intervalType <- intervalType[1]
    # Check the provided type of interval

    if(is.logical(intervalType)){
        if(intervalType){
            intervalType <- "p";
        }
        else{
            intervalType <- "none";
        }
    }

    if(all(intervalType!=c("a","p","l","s","n","a","sp","np","none","parametric","likelihood",
                           "semiparametric","nonparametric","asymmetric"))){
        stop(paste0("What do you mean by 'intervalType=",intervalType,"'? I can't work with this!"),
             call.=FALSE);
    }

    if(intervalType=="none"){
        intervalType <- "n";
    }
    else if(intervalType=="parametric"){
        intervalType <- "p";
    }
    # If it is likelihood, then it is "parametric", just uses a differen number of degrees of freedom
    else if(any(intervalType==c("likelihood","l"))){
        intervalType <- "p";
    }
    else if(intervalType=="semiparametric"){
        intervalType <- "sp";
    }
    else if(intervalType=="nonparametric"){
        intervalType <- "np";
    }

    if(intervalType=="p"){
        if(any(is.null(measurement),is.null(transition),is.null(persistence),is.null(s2),is.null(lagsModel))){
            stop(paste0("measurement, transition, persistence, s2 and lagsModel ",
                        "need to be provided in order to construct the parametric interval!"),
                 call.=FALSE);
        }

        if(any(!is.matrix(measurement),!is.matrix(transition),!is.matrix(persistence))){
            stop(paste0("measurement, transition and persistence must me matrices. ",
                        "Can't do stuff with what you've provided."),
                 call.=FALSE);
        }
    }

    # Function allows to estimate the coefficients of the simple quantile regression.
    # Used in interval construction.
    quantfunc <- function(A){
        ee[] <- ye - (A[1]*xe^A[2]);
        return((1-quant)*sum(abs(ee[ee<0]))+quant*sum(abs(ee[ee>=0])));
    }

    # Function returns quantiles of Bernoulli-lognormal cumulative distribution for a predefined parameters
    qlnormBin <- function(probability, level=0.95, meanVec=0, sdVec=1, Etype="A"){

        levelResidual <- (level - (1-probability)) / probability

        lowerquant <- upperquant <- rep(0,length(sdVec));

        positiveLevels <- levelResidual>0;

        if(any(positiveLevels)){
            # If this is Laplace or S, then get b values
            if(loss=="MAE"){
                sdVec <- sqrt(sdVec/2);
            }
            else if(loss=="HAM"){
                sdVec <- (sdVec/120)^0.25;
            }

            # Produce lower quantiles if the probability is still lower than the lower P
            if(Etype=="A" | all(Etype=="M",all((1-probability) < (1-level)/2))){
                if(Etype=="M"){
                    if(loss=="MAE"){
                        lowerquant[positiveLevels] <- exp(qlaplace((1-levelResidual[positiveLevels])/2,
                                                                   meanVec,sdVec));
                    }
                    else if(loss=="HAM"){
                        lowerquant[positiveLevels] <- exp(qs((1-levelResidual[positiveLevels])/2,
                                                             meanVec,sdVec));
                    }
                    else{
                        lowerquant[positiveLevels] <- qlnorm((1-levelResidual[positiveLevels])/2,
                                                             meanVec,sdVec);
                    }
                }
                else{
                    if(loss=="MAE"){
                        lowerquant[positiveLevels] <- qlaplace((1-levelResidual[positiveLevels])/2,
                                                               meanVec,sdVec);
                    }
                    else if(loss=="HAM"){
                        lowerquant[positiveLevels] <- qs((1-levelResidual[positiveLevels])/2,
                                                         meanVec,sdVec);
                    }
                    else{
                        lowerquant[positiveLevels] <- qnorm((1-levelResidual[positiveLevels])/2,
                                                            meanVec,sdVec);
                    }
                }

                levelNew <- (1+levelResidual[positiveLevels])/2;
            }
            else{
                levelNew <- levelResidual[positiveLevels];
            }

            # Produce upper quantiles
            if(Etype=="M"){
                if(loss=="MAE"){
                    upperquant[positiveLevels] <- exp(qlaplace(levelNew,meanVec,sdVec));
                }
                else if(loss=="HAM"){
                    upperquant[positiveLevels] <- exp(qs(levelNew,meanVec,sdVec));
                }
                else{
                    upperquant[positiveLevels] <- qlnorm(levelNew,meanVec,sdVec);
                }
            }
            else{
                if(loss=="MAE"){
                    upperquant[positiveLevels] <- qlaplace(levelNew,meanVec,sdVec);
                }
                else if(loss=="HAM"){
                    upperquant[positiveLevels] <- qs(levelNew,meanVec,sdVec);
                }
                else{
                    upperquant[positiveLevels] <- qnorm(levelNew,meanVec,sdVec);
                }
            }
        }

        return(list(lower=lowerquant,upper=upperquant));
    }

    if(loss=="MAE"){
        upperquant <- qlaplace((1+level)/2,0,1);
        lowerquant <- qlaplace((1-level)/2,0,1);
    }
    else if(loss=="HAM"){
        upperquant <- qs((1+level)/2,0,1);
        lowerquant <- qs((1-level)/2,0,1);
    }
    else{
        #if(loss=="MSE")
        # If degrees of freedom are provided, use Student's distribution. Otherwise stick with normal.
        if(is.null(df)){
            upperquant <- qnorm((1+level)/2,0,1);
            lowerquant <- qnorm((1-level)/2,0,1);
        }
        else{
            if(df>0){
                upperquant <- qt((1+level)/2,df=df);
                lowerquant <- qt((1-level)/2,df=df);
            }
            else{
                upperquant <- sqrt(1/((1-level)/2));
                lowerquant <- -upperquant;
            }
        }
    }

    ##### If they want us to produce several steps ahead #####
    if(is.matrix(errors) | is.data.frame(errors)){
        if(!cumulative){
            nVariables <- ncol(errors);
        }
        else{
            nVariables <- 1;
        }
        obs <- nrow(errors);
        if(length(ev)==1){
            ev <- rep(ev,nVariables);
        }

        upper <- rep(NA,nVariables);
        lower <- rep(NA,nVariables);

        #### Asymmetric interval using HM ####
        if(intervalType=="a"){
            if(!cumulative){
                for(i in 1:nVariables){
                    upper[i] <- ev[i] + upperquant / hsmN^2 * Re(hm(errors[,i],ev[i]))^2;
                    lower[i] <- ev[i] + lowerquant / hsmN^2 * Im(hm(errors[,i],ev[i]))^2;
                }
            }
            else{
                upper <- ev + upperquant / hsmN^2 * Re(hm(rowSums(errors),sum(ev)))^2;
                lower <- ev + lowerquant / hsmN^2 * Im(hm(rowSums(errors),sum(ev)))^2;
            }
            if(Etype=="M"){
                upper <- yForecast*(1 + upper);
                lower <- yForecast*(1 + lower);
            }
            varVec <- NULL;
        }

        #### Semiparametric interval using the variance of errors ####
        else if(intervalType=="sp"){
            if(Etype=="M"){
                errors[errors < -1] <- -0.999;
                if(!cumulative){
                    varVec <- colSums(log(1+errors)^2,na.rm=T)/df;
                    if(any(probability!=1)){
                        quants <- qlnormBin(probability, level=level, meanVec=log(yForecast),
                                            sdVec=sqrt(varVec), Etype="M");
                        upper <- quants$upper;
                        lower <- quants$lower;
                    }
                    else{
                        if(loss=="MAE"){
                            varVec <- sqrt(varVec/2);
                            upper <- exp(qlaplace((1+level)/2,0,varVec));
                            lower <- exp(qlaplace((1-level)/2,0,varVec));
                        }
                        else if(loss=="HAM"){
                            varVec <- (varVec/120)^0.25;
                            upper <- exp(qs((1+level)/2,0,varVec));
                            lower <- exp(qs((1-level)/2,0,varVec));
                        }
                        else{
                            upper <- qlnorm((1+level)/2,rep(0,nVariables),sqrt(varVec));
                            lower <- qlnorm((1-level)/2,rep(0,nVariables),sqrt(varVec));
                        }
                    }
                    upper <- yForecast*upper;
                    lower <- yForecast*lower;
                }
                else{
                    #This is wrong. And there's not way to make it right.
                    varVec <- sum(rowSums(log(1+errors))^2,na.rm=T)/df;
                    if(any(probability!=1)){
                        quants <- qlnormBin(probability, level=level, meanVec=log(sum(yForecast)),
                                            sdVec=sqrt(varVec), Etype="M");
                        upper <- quants$upper;
                        lower <- quants$lower;
                    }
                    else{
                        if(loss=="MAE"){
                            varVec <- sqrt(varVec/2);
                            upper <- exp(qlaplace((1+level)/2,0,varVec));
                            lower <- exp(qlaplace((1-level)/2,0,varVec));
                        }
                        else if(loss=="HAM"){
                            varVec <- (varVec/120)^0.25;
                            upper <- exp(qs((1+level)/2,0,varVec));
                            lower <- exp(qs((1-level)/2,0,varVec));
                        }
                        else{
                            upper <- qlnorm((1+level)/2,rep(0,nVariables),sqrt(varVec));
                            lower <- qlnorm((1-level)/2,rep(0,nVariables),sqrt(varVec));
                        }
                    }
                    upper <- sum(yForecast)*upper;
                    lower <- sum(yForecast)*lower;
                }
            }
            else{
                if(!cumulative){
                    errors <- errors - matrix(ev,nrow=obs,ncol=nVariables,byrow=T);
                    varVec <- colSums(errors^2,na.rm=T)/df;
                    if(any(probability!=1)){
                        quants <- qlnormBin(probability, level=level, meanVec=ev, sdVec=sqrt(varVec), Etype="A");
                        upper <- ev + quants$upper;
                        lower <- ev + quants$lower;
                    }
                    else{
                        if(loss=="MAE"){
                            # s^2 = 2 b^2 => b^2 = s^2 / 2
                            varVec <- varVec / 2;
                        }
                        else if(loss=="HAM"){
                            # s^2 = 120 b^4 => b^4 = s^2 / 120
                            # S(mu, b) = S(mu, 1) * 50^2
                            varVec <- varVec/120;
                        }
                        upper <- ev + upperquant * sqrt(varVec);
                        lower <- ev + lowerquant * sqrt(varVec);
                    }
                }
                else{
                    errors <- errors - matrix(ev,nrow=obs,ncol=ncol(errors),byrow=T);
                    varVec <- sum(rowSums(errors,na.rm=T)^2,na.rm=T)/df;
                    if(any(probability!=1)){
                        quants <- qlnormBin(probability, level=level, meanVec=sum(ev), sdVec=sqrt(varVec), Etype="A");
                        upper <- sum(ev) + quants$upper;
                        lower <- sum(ev) + quants$lower;
                    }
                    else{
                        if(loss=="MAE"){
                            # s^2 = 2 b^2 => b^2 = s^2 / 2
                            varVec <- varVec / 2;
                        }
                        else if(loss=="HAM"){
                            # s^2 = 120 b^4 => b^4 = s^2 / 120
                            # S(mu, b) = S(mu, 1) * 50^2
                            varVec <- varVec/120;
                        }
                        upper <- sum(ev) + upperquant * sqrt(varVec);
                        lower <- sum(ev) + lowerquant * sqrt(varVec);
                    }

                }
            }
        }

        #### Nonparametric interval using Taylor and Bunn, 1999 ####
        else if(intervalType=="np"){
            nonNAobs <- apply(!is.na(errors),1,all);
            ye <- errors[nonNAobs,];

            if(Etype=="M"){
                ye <- 1 + ye;
            }

            # Define the correct bounds for the intermittent model
            levelResidual <- (level - (1-probability)) / probability;
            lower <- upper <- rep(0,length(yForecast));

            if(!cumulative){
                ee <- ye;
                xe <- matrix(c(1:nVariables),nrow=sum(nonNAobs),ncol=nVariables,byrow=TRUE);

                if(Etype=="A" | all(Etype=="M",all((1-probability) < (1-level)/2))){
                    A <- rep(1,2);
                    quant <- (1-levelResidual)/2;
                    A <- nlminb(A,quantfunc)$par;
                    lower <- A[1]*c(1:nVariables)^A[2];

                    levelNew <- (1+levelResidual)/2;
                }
                else{
                    levelNew <- levelResidual;
                }

                A <- rep(1,2);
                quant <- levelNew;
                A <- nlminb(A,quantfunc)$par;
                upper <- A[1]*c(1:nVariables)^A[2];

                if(Etype=="M"){
                    upper <- yForecast * upper;
                    lower <- yForecast * lower;
                }
            }
            else{
                if(Etype=="A" | all(Etype=="M",all((1-probability) < (1-level)/2))){
                #This is wrong. And there's no way to make it right.
                    lower <- quantile(rowSums(ye),(1-levelResidual)/2);
                    levelNew <- (1+levelResidual)/2;
                }
                else{
                    levelNew <- levelResidual;
                }
                upper <- quantile(rowSums(ye),levelNew);
            }
            varVec <- NULL;
        }

        #### Parametric interval ####
        else if(intervalType=="p"){
            h <- length(yForecast);

            # Vector of final variances
            varVec <- rep(NA,h);

            #### Pure Multiplicative models ####
            if(Etype=="M"){
                # This is just an approximation of the true interval
                covarMat <- covarAnal(lagsModel, h, measurement, transition, persistence, s2);

                ### Cumulative variance is different.
                if(cumulative){
                    varVec <- sum(covarMat);
                    varVec <- log(exp(varVec / h) * h);

                    if(any(probability!=1)){
                        quants <- qlnormBin(probability, level=level, meanVec=log(sum(yForecast)),
                                            sdVec=sqrt(varVec), Etype="M");
                        upper <- quants$upper;
                        lower <- quants$lower;
                    }
                    else{
                        if(loss=="MAE"){
                            varVec <- sqrt(varVec / 2);
                            upper <- exp(qlaplace((1+level)/2,0,varVec));
                            lower <- exp(qlaplace((1-level)/2,0,varVec));
                        }
                        else if(loss=="HAM"){
                            varVec <- (varVec/120)^0.25;
                            upper <- exp(qs((1+level)/2,0,varVec));
                            lower <- exp(qs((1-level)/2,0,varVec));
                        }
                        else{
                            # Produce quantiles for log-normal dist with the specified variance
                            upper <- qlnorm((1+level)/2,0,sqrt(varVec));
                            lower <- qlnorm((1-level)/2,0,sqrt(varVec));
                        }
                        upper <- sum(yForecast)*upper;
                        lower <- sum(yForecast)*lower;
                    }
                }
                else{
                    varVec <- diag(covarMat);

                    if(any(probability!=1)){
                        quants <- qlnormBin(probability, level=level, meanVec=log(yForecast),
                                            sdVec=sqrt(varVec), Etype="M");
                        upper <- quants$upper;
                        lower <- quants$lower;
                    }
                    else{
                        if(loss=="MAE"){
                            # s^2 = 2 b^2 => b = sqrt(s^2 / 2)
                            varVec <- sqrt(varVec / 2);
                            upper <- exp(qlaplace((1+level)/2,0,varVec));
                            lower <- exp(qlaplace((1-level)/2,0,varVec));
                        }
                        else if(loss=="HAM"){
                            # s^2 = 120 b^4 => b^4 = s^2 / 120
                            # S(mu, b) = S(mu, 1) * 50^2
                            varVec <- (varVec/120)^0.25;
                            upper <- exp(qs((1+level)/2,0,varVec));
                            lower <- exp(qs((1-level)/2,0,varVec));
                        }
                        else{
                            # Produce quantiles for log-normal dist with the specified variance
                            upper <- qlnorm((1+level)/2,0,sqrt(varVec));
                            lower <- qlnorm((1-level)/2,0,sqrt(varVec));
                        }
                        upper <- yForecast*upper;
                        lower <- yForecast*lower;
                    }
                }
            }
            #### Multiplicative error and additive trend / seasonality
            # else if(Etype=="M" & all(c(Ttype,Stype)!="M") & all(c(Ttype,Stype)!="N")){
            # }
            #### Pure Additive models ####
            else{
                covarMat <- covarAnal(lagsModel, h, measurement, transition, persistence, s2);

                ### Cumulative variance is a sum of all the elements of the matrix
                if(cumulative){
                    varVec <- sum(covarMat);
                }
                else{
                    varVec <- diag(covarMat);
                }

                if(any(probability!=1)){
                    # Take intermittent data into account
                    quants <- qlnormBin(probability, level=level, meanVec=rep(0,length(varVec)),
                                        sdVec=sqrt(varVec), Etype="A");
                    upper <- quants$upper;
                    lower <- quants$lower;
                }
                else{
                    if(loss=="MAE"){
                        # s^2 = 2 b^2 => b^2 = s^2 / 2
                        varVec <- varVec / 2;
                    }
                    else if(loss=="HAM"){
                        # s^2 = 120 b^4 => b^4 = s^2 / 120
                        # S(mu, b) = S(mu, 1) * b^2
                        varVec <- varVec/120;
                    }
                    upper <- upperquant * sqrt(varVec);
                    lower <- lowerquant * sqrt(varVec);
                }
            }
        }
    }
    ##### If we were asked to produce 1 value #####
    else if(is.numeric(errors) & length(errors)>1 & !is.array(errors)){
        if(length(ev)>1){
            stop("Provided expected value doesn't correspond to the dimension of errors.", call.=FALSE);
        }

        if(intervalType=="a"){
            upper <- ev + upperquant / hsmN^2 * Re(hm(errors,ev))^2;
            lower <- ev + lowerquant / hsmN^2 * Im(hm(errors,ev))^2;
        }
        else if(any(intervalType==c("sp","p"))){
            if(Etype=="M"){
                if(any(probability!=1)){
                    quants <- qlnormBin(probability, level=level, meanVec=0, sdVec=sqrt(s2), Etype="M");
                    upper <- quants$upper;
                    lower <- quants$lower;
                }
                else{
                    if(loss=="MAE"){
                        # s^2 = 2 b^2 => b = sqrt(s^2 / 2)
                        s2 <- sqrt(s2 / 2);
                        upper <- exp(qlaplace((1+level)/2,0,s2));
                        lower <- exp(qlaplace((1-level)/2,0,s2));
                    }
                    else if(loss=="HAM"){
                        # s^2 = 120 b^4 => b^4 = s^2 / 120
                        # S(mu, b) = S(mu, 1) * 50^2
                        s2 <- (s2/120)^0.25;
                        upper <- exp(qs((1+level)/2,0,s2));
                        lower <- exp(qs((1-level)/2,0,s2));
                    }
                    else{
                        upper <- qlnorm((1+level)/2,0,sqrt(s2));
                        lower <- qlnorm((1-level)/2,0,sqrt(s2));
                    }
                    upper <- yForecast*upper;
                    lower <- yForecast*lower;
                }
            }
            else{
                if(any(probability!=1)){
                    quants <- qlnormBin(probability, level=level, meanVec=ev, sdVec=sqrt(s2), Etype="A");
                    upper <- quants$upper;
                    lower <- quants$lower;
                }
                else{
                    if(loss=="MAE"){
                        # s^2 = 2 b^2 => b^2 = s^2 / 2
                        s2 <- s2 / 2;
                    }
                    else if(loss=="HAM"){
                        # s^2 = 120 b^4 => b^4 = s^2 / 120
                        # S(mu, b) = S(mu, 1) * 50^2
                        s2 <- s2/120;
                    }
                    upper <- ev + upperquant * sqrt(s2);
                    lower <- ev + lowerquant * sqrt(s2);
                }
            }
        }
        else if(intervalType=="np"){
            if(Etype=="M"){
                errors <- errors + 1;
            }
            upper <- quantile(errors,(1+level)/2);
            lower <- quantile(errors,(1-level)/2);
        }
        varVec <- NULL;
    }
    else{
        stop("The provided data is not either vector or matrix. Can't do anything with it!", call.=FALSE);
    }

    return(list(upper=upper,lower=lower));
}

##### *Forecaster of state space functions* #####
ssForecaster <- function(...){
    ellipsis <- list(...);
    ParentEnvironment <- ellipsis[['ParentEnvironment']];

    if(intervalType!="l" && obsInSample > nParam){
        obsDF <- obsInSample - nParam;
    }
    else{
        obsDF <- obsInSample;
    }
    if(!rounded){
        s2 <- as.vector(sum((errors*ot)^2)/obsDF);
        # If error is additive, s2g is just 1
        if(Etype=="A"){
            s2g <- 1;
        }
        else{
            s2g <- log(1 + vecg %*% as.vector(errors*ot)) %*% t(log(1 + vecg %*%
                                                                        as.vector(errors*ot)))/obsDF;
        }
    }

    if(h>0){
        yForecast <- ts(c(forecasterwrap(matvt[(obsInSample+1):(obsInSample+lagsModelMax),,drop=FALSE],
                                         matF, matw, h, Etype, Ttype, Stype, lagsModel,
                                         matxt[(obsAll-h+1):(obsAll),,drop=FALSE],
                                         matat[(obsAll-h+1):(obsAll),,drop=FALSE], matFX)),
                        start=yForecastStart,frequency=dataFreq);

        if(any(is.nan(yForecast))){
            warning(paste0("NaNs were produced in the forecast.\n",
                           "This could happen if you used multiplicative model on non-positive data. ",
                           "Please, use a different model."),
                    call.=FALSE);
            yForecast[] <- 0;
        }
        if(Etype=="M" & any(yForecast<0)){
            warning(paste0("Negative values produced in forecast. ",
                           "This does not make any sense for model with multiplicative error.\n",
                           "Please, use another model."),
                    call.=FALSE);
            if(interval){
                warning("And don't expect anything reasonable from the prediction interval!",
                        call.=FALSE);
            }
        }

        #### This chunk of code is commented out because it is not used anywhere
        # Write down the forecasting interval
        # if(interval){
        #     if(h==1){
        #         errors.x <- as.vector(errors);
        #         ev <- median(errors);
        #     }
        #     else{
        #         errors.x <- errors.mat;
        #         ev <- apply(errors.mat,2,median,na.rm=TRUE);
        #     }
        #     if(intervalType!="a"){
        #         ev <- 0;
        #     }
        #
        #     # We don't simulate pure additive models, pure multiplicative and
        #     # additive models with multiplicative error,
        #     # because they can be approximated by the pure additive ones
        #     if(any(intervalType==c("p","l"))){
        #         if(all(c(Etype,Stype,Ttype)!="M") |
        #            all(c(Etype,Stype,Ttype)!="A") |
        #            (all(Etype=="M",any(Ttype==c("A","N")),any(Stype==c("A","N"))) & s2<0.1)){
        #             simulateIntervals <- FALSE;
        #         }
        #         else{
        #             simulateIntervals <- TRUE;
        #         }
        #     }
        #     else{
        #         simulateIntervals <- FALSE;
        #     }
        #
        #     # It is not possible to produce parametric / semi / non interval for cumulative values
        #     # of multiplicative model. So we use simulations instead.
        #     # if(Etype=="M"){
        #     #     simulateIntervals <- TRUE;
        #     # }
        #
        #     if(simulateIntervals){
        #         nSamples <- 100000;
        #         matg <- matrix(vecg,nComponents,nSamples);
        #         arrvt <- array(NA,c(h+lagsModelMax,nComponents,nSamples));
        #         arrvt[1:lagsModelMax,,] <- rep(matvt[obsInSample+(1:lagsModelMax),],nSamples);
        #         materrors <- matrix(rnorm(h*nSamples,0,sqrt(s2)),h,nSamples);
        #
        #         if(Etype=="M"){
        #             materrors <- exp(materrors) - 1;
        #         }
        #         if(all(occurrence!=c("n","p"))){
        #             matot <- matrix(rbinom(h*nSamples,1,pForecast),h,nSamples);
        #         }
        #         else{
        #             matot <- matrix(1,h,nSamples);
        #         }
        #
        #         ySimulated <- simulatorwrap(arrvt,materrors,matot,array(matF,c(dim(matF),nSamples)),matw,matg,
        #                                     Etype,Ttype,Stype,lagsModel)$matyt;
        #
        #         if(!is.null(xreg)){
        #             yForecastExo <- (c(yForecast) -
        #                                  forecasterwrap(matrix(matvt[(obsInSample+1):(obsInSample+lagsModelMax),],
        #                                                        nrow=lagsModelMax),
        #                                                 matF, matw, h, Etype, Ttype, Stype, lagsModel,
        #                                                 matrix(rep(1,h),ncol=1), matrix(rep(0,h),ncol=1),
        #                                                 matrix(1,1,1)));
        #         }
        #         else{
        #             yForecastExo <- rep(0,h);
        #         }
        #
        #         if(Etype=="M"){
        #             yForecast[] <- apply(ySimulated, 1, mean);
        #         }
        #
        #         if(rounded){
        #             ySimulated <- ceiling(ySimulated + matrix(yForecastExo,nrow=h,ncol=nSamples));
        #             quantileType <- 1;
        #             for(i in 1:h){
        #                 yForecast[i] <- median(ySimulated[i,ySimulated[i,]!=0]);
        #             }
        #             # NA means that there were no non-zero demands
        #             yForecast[is.na(yForecast)] <- 0;
        #         }
        #         else{
        #             ySimulated <- ySimulated + matrix(yForecastExo,nrow=h,ncol=nSamples);
        #             quantileType <- 7;
        #         }
        #
        #         yForecast[] <- yForecast + yForecastExo;
        #         yForecast[] <- pForecast * yForecast;
        #
        #         if(cumulative){
        #             yForecast <- ts(sum(yForecast),start=yForecastStart,frequency=dataFreq);
        #             yLower <- ts(quantile(colSums(ySimulated,na.rm=T),(1-level)/2,type=quantileType),
        #                          start=yForecastStart,frequency=dataFreq);
        #             yUpper <- ts(quantile(colSums(ySimulated,na.rm=T),(1+level)/2,type=quantileType),
        #                          start=yForecastStart,frequency=dataFreq);
        #         }
        #         else{
        #             # yForecast <- ts(yForecast,start=yForecastStart,frequency=dataFreq);
        #             yLower <- ts(apply(ySimulated,1,quantile,(1-level)/2,na.rm=T,type=quantileType) +
        #                              yForecastExo,
        #                          start=yForecastStart,frequency=dataFreq);
        #             yUpper <- ts(apply(ySimulated,1,quantile,(1+level)/2,na.rm=T,type=quantileType) +
        #                              yForecastExo,
        #                          start=yForecastStart,frequency=dataFreq);
        #         }
        #     }
        #     else{
        #         quantvalues <- ssIntervals(errors.x, ev=ev, level=level, intervalType=intervalType,
        #                                    df=obsDF,
        #                                    measurement=matw, transition=matF, persistence=vecg, s2=s2,
        #                                    lagsModel=lagsModel, states=matvt[(obsInSample-lagsModelMax+1):obsInSample,],
        #                                    cumulative=cumulative, loss=loss,
        #                                    yForecast=yForecast, Etype=Etype, Ttype=Ttype, Stype=Stype, s2g=s2g,
        #                                    probability=pForecast);
        #
        #         # if(!(intervalType=="sp" & Etype=="M")){
        #             yForecast[] <- c(pForecast) * c(yForecast);
        #         # }
        #
        #         if(cumulative){
        #             yForecast <- ts(sum(yForecast),start=yForecastStart,frequency=dataFreq);
        #         }
        #
        #         if(Etype=="A"){
        #             yLower <- ts(c(yForecast) + quantvalues$lower,start=yForecastStart,frequency=dataFreq);
        #             yUpper <- ts(c(yForecast) + quantvalues$upper,start=yForecastStart,frequency=dataFreq);
        #         }
        #         else{
        #             # if(any(intervalType==c("np","sp","a"))){
        #             #     quantvalues$upper <- quantvalues$upper * yForecast;
        #             #     quantvalues$lower <- quantvalues$lower * yForecast;
        #             # }
        #             yLower <- ts(quantvalues$lower,start=yForecastStart,frequency=dataFreq);
        #             yUpper <- ts(quantvalues$upper,start=yForecastStart,frequency=dataFreq);
        #         }
        #
        #         if(rounded){
        #             yLower <- ceiling(yLower);
        #             yUpper <- ceiling(yUpper);
        #         }
        #     }
        # }
        # else{
            yLower <- NA;
            yUpper <- NA;
            if(rounded){
                yForecast[] <- ceiling(yForecast);
            }
            yForecast[] <- pForecast*yForecast;
            if(cumulative){
                yForecast <- ts(sum(yForecast),start=yForecastStart,frequency=dataFreq);
            }
            # else{
            #     yForecast <- ts(yForecast,start=yForecastStart,frequency=dataFreq);
            # }
        # }
    }
    else{
        yLower <- NA;
        yUpper <- NA;
        yForecast <- ts(NA,start=yForecastStart,frequency=dataFreq);
    }

    if(any(is.na(yFitted),all(is.na(yForecast),h>0))){
        warning("Something went wrong during the optimisation and NAs were produced!",call.=FALSE);
        warning("Please check the input and report this error to the maintainer if it persists.",call.=FALSE);
    }

    assign("s2",s2,ParentEnvironment);
    assign("yForecast",yForecast,ParentEnvironment);
    assign("yLower",yLower,ParentEnvironment);
    assign("yUpper",yUpper,ParentEnvironment);
}

##### *Check and initialisation of xreg* #####
ssXreg <- function(y, Etype="A", xreg=NULL, updateX=FALSE, ot=NULL,
                   persistenceX=NULL, transitionX=NULL, initialX=NULL,
                   obsInSample, obsAll, obsStates, lagsModelMax=1, h=1, regressors="u", silent=FALSE,
                   allowMultiplicative=FALSE){
    # The function does general checks needed for exogenouse variables and returns the list of
    # necessary parameters

    if(!is.null(xreg)){
        xreg <- as.matrix(xreg);
        if(any(is.na(xreg))){
            warning(paste0("The exogenous variables contain NAs! ",
                           "This may lead to problems during estimation and in forecasting.",
                           "\nSubstituting them with 0."),
                    call.=FALSE);
            xreg[is.na(xreg)] <- 0;
        }
        if(!is.null(dim(xreg))){
            if(ncol(xreg)==1){
                xreg <- as.vector(xreg);
            }
            if(length(dim(xreg))>2){
                stop(paste0("Sorry, but we don't deal with multidimensional arrays as exogenous variables here.\n",
                            "Pleas think of your behaviour and come back with matrix!"),call.=FALSE);
            }
        }
        ##### The case with vectors and ts objects, but not matrices
        if(is.vector(xreg) | (is.ts(xreg) & !is.matrix(xreg))){
            # Check if xreg contains something meaningful
            if(is.null(initialX)){
                if(all(xreg[1:obsInSample]==xreg[1])){
                    warning(paste0("The exogenous variable has no variability. ",
                                   "Cannot do anything with that, so dropping out xreg."),
                            call.=FALSE);
                    xreg <- NULL;
                }
            }

            if(!is.null(xreg)){
                if(length(xreg) < obsAll){
                    warning("xreg did not contain values for the holdout, so we had to predict missing values.",
                            call.=FALSE);
                    # If this is a binary variable, use iss function.
                    if(all((xreg==0) | (xreg==1))){
                        xregForecast <- oes(xreg, model="MNN", h=obsAll-length(xreg),
                                            occurrence="o", ic="AIC")$forecast;
                    }
                    else{
                        xregForecast <- es(xreg, h=obsAll-length(xreg), ic="AICc",silent=TRUE)$forecast;
                    }
                    xreg <- c(as.vector(xreg),as.vector(xregForecast));
                }
                else if(length(xreg) > obsAll){
                    warning("xreg contained too many observations, so we had to cut off some of them.",
                            call.=FALSE);
                    xreg <- xreg[1:obsAll];
                }

                if(all(y[1:obsInSample]==xreg[1:obsInSample])){
                    warning(paste0("The exogenous variable and the forecasted data are exactly the same. ",
                                   "What's the point of such a regression?"),
                            call.=FALSE);
                    xreg <- NULL;
                }

                # Number of exogenous variables
                nExovars <- 1;
                # Define matrix w for exogenous variables
                matxt <- matrix(xreg,ncol=1);
                # Define the second matat to fill in the coefs of the exogenous vars
                matatMultiplicative <- matat <- matrix(NA,obsStates,1);
                # Fill in the initial values for exogenous coefs using OLS
                if(is.null(initialX)){
                    if(Etype=="M"){
                        matat[1:lagsModelMax,] <- cov(log(y[1:obsInSample][ot==1]),
                                                xreg[1:obsInSample][ot==1])/var(xreg[1:obsInSample][ot==1]);
                        matatMultiplicative[1:lagsModelMax,] <- matat[1:lagsModelMax,];
                    }
                    else{
                        matat[1:lagsModelMax,] <- cov(y[1:obsInSample][ot==1],
                                                xreg[1:obsInSample][ot==1])/var(xreg[1:obsInSample][ot==1]);
                    }
                    matat[] <- matat[1,]

                    # If Etype=="Z" or "C", estimate multiplicative stuff.
                    if(allowMultiplicative & all(Etype!=c("M","A"))){
                        matatMultiplicative[1:lagsModelMax,] <- cov(log(y[1:obsInSample][ot==1]),
                                                              xreg[1:obsInSample][ot==1])/var(xreg[1:obsInSample][ot==1]);
                    }
                }
                if(is.null(names(xreg))){
                    colnames(matxt) <- colnames(matat) <- colnames(matatMultiplicative) <- "x";
                }
                else{
                    xregNames <- gsub(" ", "_", names(xreg), fixed = TRUE);
                    colnames(matxt) <- colnames(matat) <- colnames(matatMultiplicative) <- xregNames
                }
            }
            if(!is.null(xreg)){
                xreg <- as.matrix(xreg);
            }
        }
        ##### The case with matrices and data frames
        else if(is.matrix(xreg) | is.data.frame(xreg)){
            if(is.data.frame(xreg)){
                xreg <- as.matrix(xreg);
            }
            nExovars <- ncol(xreg);
            if(nrow(xreg) < obsAll){
                warning("xreg did not contain values for the holdout, so we had to predict missing values.",
                        call.=FALSE);
                xregForecast <- matrix(NA,nrow=obsAll-nrow(xreg),ncol=nExovars);
                if(!silent){
                    message("Producing forecasts for xreg variable...");
                }
                for(j in 1:nExovars){
                    if(!silent){
                        cat(paste0(rep("\b",nchar(round((j-1)/nExovars,2)*100)+1),collapse=""));
                        cat(paste0(round(j/nExovars,2)*100,"%"));
                    }

                    if(all((xreg[,j]==0) | (xreg[,j]==1))){
                        xregForecast[,j] <- oes(xreg[,j], model="MNN", h=obsAll-nrow(xreg), occurrence="o",ic="AIC")$forecast;
                    }
                    else{
                        xregForecast[,j] <- es(xreg[,j], h=obsAll-nrow(xreg), ic="AICc")$forecast;
                    }
                }
                xreg <- rbind(xreg,xregForecast);
                if(!silent){
                    cat("\b\b\b\bDone!\n");
                }
            }
            else if(nrow(xreg) > obsAll){
                warning("xreg contained too many observations, so we had to cut off some of them.",
                        call.=FALSE);
                xreg <- xreg[1:obsAll,];
            }

            xregEqualToData <- apply(xreg[1:obsInSample,]==y[1:obsInSample],2,all);
            if(any(xregEqualToData)){
                warning(paste0("One of exogenous variables and the forecasted data are exactly the same. ",
                               "We have dropped it."),
                        call.=FALSE);
                xreg <- matrix(xreg[,!xregEqualToData],nrow=nrow(xreg),ncol=ncol(xreg)-1,
                               dimnames=list(NULL,colnames(xreg[,!xregEqualToData])));
            }

            nExovars <- ncol(xreg);

            # If initialX is provided, then probably we don't need to check the xreg on variability and multicollinearity
            if(is.null(initialX)){
                checkvariability <- apply(matrix(xreg[1:obsInSample,][ot==1,]==rep(xreg[ot==1,][1,],
                                                                                   each=sum(ot)),sum(ot),
                                                 nExovars),2,all);
                if(any(checkvariability)){
                    if(all(checkvariability)){
                        warning(paste0("None of exogenous variables has variability. ",
                                       "Cannot do anything with that, so dropping out xreg."),
                                call.=FALSE);
                        xreg <- NULL;
                        nExovars <- 0;
                    }
                    else{
                        warning("Some exogenous variables do not have any variability. Dropping them out.",
                                call.=FALSE);
                        xreg <- as.matrix(xreg[,!checkvariability]);
                        nExovars <- ncol(xreg);
                    }
                }

                if(!is.null(xreg)){
                    # Check for multicollinearity and drop something if there is a perfect one
                    corMatrix <- cor(xreg);
                    corCheck <- upper.tri(corMatrix) & abs(corMatrix)>=0.999;
                    if(any(corCheck)){
                        removexreg <- unique(which(corCheck,arr.ind=TRUE)[,1]);
                        if(ncol(xreg)-length(removexreg)>1){
                            xreg <- xreg[,-removexreg];
                        }
                        else{
                            xreg <- matrix(xreg[,-removexreg],ncol=ncol(xreg)-length(removexreg),
                                           dimnames=list(NULL,c(colnames(xreg)[-removexreg])));
                        }
                        nExovars <- ncol(xreg);
                        warning("Some exogenous variables were perfectly correlated. We've dropped them out.",
                                call.=FALSE);
                    }
                    # Check multiple correlations. This is needed for cases with dummy variables.
                    # In case with regressors="select" some of the perfectly correlated things,
                    # will be dropped out automatically.
                    if(nExovars>2 & (regressors=="u")){
                        corMatrix <- cor(xreg);
                        corMulti <- rep(NA,nExovars);
                        if(det(corMatrix)!=0){
                            for(i in 1:nExovars){
                                corMulti[i] <- 1 - det(corMatrix) / det(corMatrix[-i,-i]);
                            }
                            if(any(corMulti>=0.999)){
                                stop(paste0("Some combinations of exogenous variables are perfectly correlated. \n",
                                            "If you use sets of dummy variables, don't forget to drop some of them."),
                                     call.=FALSE);
                            }
                        }
                        else{
                            stop(paste0("Some combinations of exogenous variables are perfectly correlated. \n",
                                        "If you use sets of dummy variables, don't forget to drop some of them."),
                                 call.=FALSE);
                        }
                    }
                }
            }

            if(!is.null(xreg)){
                # mat.x is needed for the initial values of coefs estimation using OLS
                mat.x <- as.matrix(cbind(rep(1,obsAll),xreg));
                # Define the second matat to fill in the coefs of the exogenous vars
                matatMultiplicative <- matat <- matrix(NA,obsStates,nExovars);
                # Define matrix w for exogenous variables
                matxt <- as.matrix(xreg);
                # Fill in the initial values for exogenous coefs using OLS
                if(is.null(initialX)){
                    if(Etype=="M"){
                        matat[1:lagsModelMax,] <- rep(t(solve(t(mat.x[1:obsInSample,][ot==1,]) %*%
                                                            mat.x[1:obsInSample,][ot==1,],tol=1e-50) %*%
                                                      t(mat.x[1:obsInSample,][ot==1,]) %*%
                                                      log(y[1:obsInSample][ot==1]))[2:(nExovars+1)],
                                                each=lagsModelMax);
                        matatMultiplicative[1:lagsModelMax,] <- matat[1:lagsModelMax,];
                    }
                    else{
                        matat[1:lagsModelMax,] <- rep(t(solve(t(mat.x[1:obsInSample,][ot==1,]) %*%
                                                            mat.x[1:obsInSample,][ot==1,],tol=1e-50) %*%
                                                      t(mat.x[1:obsInSample,][ot==1,]) %*%
                                                      y[1:obsInSample][ot==1])[2:(nExovars+1)],
                                                each=lagsModelMax);
                    }
                    matat[-1,] <- rep(matat[1,],each=obsStates-1);

                    # If Etype=="Z" or "C", estimate multiplicative stuff.
                    if(allowMultiplicative & all(Etype!=c("M","A"))){
                        matatMultiplicative[1:lagsModelMax,] <- rep(t(solve(t(mat.x[1:obsInSample,][ot==1,]) %*%
                                                                          mat.x[1:obsInSample,][ot==1,],tol=1e-50) %*%
                                                                    t(mat.x[1:obsInSample,][ot==1,]) %*%
                                                                    log(y[1:obsInSample][ot==1]))[2:(nExovars+1)],
                                                              each=lagsModelMax);
                    }
                }
                if(is.null(colnames(xreg))){
                    colnames(matxt) <- colnames(matat) <- colnames(matatMultiplicative) <- paste0("x",c(1:nExovars));
                }
                else{
                    xregNames <- gsub(" ", "_", colnames(xreg), fixed = TRUE);
                    if(regressors=="s" & any(grepl('[^[:alnum:]]', xregNames))){
                        warning(paste0("There were some special characters in names of ",
                                       "xreg variables. We had to remove them."),call.=FALSE);
                        xregNames <- gsub("[^[:alnum:]]", "", xregNames);
                    }
                    xregDuplicated <- duplicated(colnames(xreg));
                    if(any(xregDuplicated)){
                        warning(paste0("Some names of variables are duplicated. ",
                                       "We had to rename them."),call.=FALSE);
                        xregNames[xregDuplicated] <- paste0("xDuplicated",c(1:sum(xregDuplicated)));
                    }
                    colnames(matxt) <- colnames(matat) <- colnames(matatMultiplicative) <- xregNames;
                }
            }
        }
        else{
            stop("Unknown format of xreg. Should be either vector or matrix. Aborting!",call.=FALSE);
        }
        xregEstimate <- TRUE;

        # Check the provided initialX vector
        if(!is.null(initialX)){
            if(!is.numeric(initialX) & !is.vector(initialX) & !is.matrix(initialX)){
                stop("The initials for exogenous are not a numeric vector or a matrix!", call.=FALSE);
            }
            else{
                if(length(initialX) != nExovars){
                    stop(paste0("The size of initial vector for exogenous is wrong!\n",
                                "It should correspond to the number of exogenous variables."), call.=FALSE);
                }
                else{
                    matat[1:lagsModelMax,] <- as.vector(rep(initialX,each=lagsModelMax));
                    initialXEstimate <- FALSE;
                }
            }
        }
        else{
            initialXEstimate <- TRUE;
        }
    }
    else{
        updateX <- FALSE;
    }

    ##### In case we changed xreg to null or if it was like that...
    if(is.null(xreg)){
        # "1" is needed for the final forecast simplification
        nExovars <- 1;
        matxt <- matrix(1,obsStates,1);
        matatMultiplicative <- matat <- matrix(0,obsStates,1);
        matFX <- matrix(1,1,1);
        vecgX <- matrix(0,1,1);
        xregEstimate <- FALSE;
        FXEstimate <- FALSE;
        gXEstimate <- FALSE;
        initialXEstimate <- FALSE;
    }

    # Now check transition and persistence of exogenous variables
    if(xregEstimate & updateX){
        # First - transition matrix
        if(!is.null(transitionX)){
            if(!is.numeric(transitionX) & !is.vector(transitionX) & !is.matrix(transitionX)){
                stop("Transition matrix for exogenous is not a numeric vector or a matrix!", call.=FALSE);
            }
            else{
                if(length(transitionX) != nExovars^2){
                    stop(paste0("Size of transition matrix for exogenous is wrong!\n",
                                "It should correspond to the number of exogenous variables."), call.=FALSE);
                }
                else{
                    matFX <- matrix(transitionX,nExovars,nExovars);
                    FXEstimate <- FALSE;
                }
            }
        }
        else{
            matFX <- diag(nExovars);
            FXEstimate <- TRUE;
        }
        # Now - persistence vector
        if(!is.null(persistenceX)){
            if(!is.numeric(persistenceX) & !is.vector(persistenceX) & !is.matrix(persistenceX)){
                stop("Persistence vector for exogenous is not numeric!", call.=FALSE);
            }
            else{
                if(length(persistenceX) != nExovars){
                    stop(paste0("Size of persistence vector for exogenous is wrong!\n",
                                "It should correspond to the number of exogenous variables."), call.=FALSE);
                }
                else{
                    vecgX <- matrix(persistenceX,nExovars,1);
                    gXEstimate <- FALSE;
                }
            }
        }
        else{
            vecgX <- matrix(0,nExovars,1);
            gXEstimate <- TRUE;
        }
    }
    else if(xregEstimate & !updateX){
        matFX <- diag(nExovars);
        FXEstimate <- FALSE;

        vecgX <- matrix(0,nExovars,1);
        gXEstimate <- FALSE;
    }

    if(all(!FXEstimate,!gXEstimate,!initialXEstimate)){
        xregEstimate <- FALSE;
    }

    return(list(nExovars=nExovars, matxt=matxt, matat=matat, matFX=matFX, vecgX=vecgX,
                xreg=xreg, xregEstimate=xregEstimate, FXEstimate=FXEstimate,
                gXEstimate=gXEstimate, initialXEstimate=initialXEstimate,
                matatMultiplicative=matatMultiplicative))
}

##### *Likelihood function* #####
likelihoodFunction <- function(B,yFittedSumLog=0){
    #### Concentrated logLikelihood based on B and CF ####
    logLikFromCF <- function(B, loss){
        if(loss=="likelihood"){
            return(-CFValue);
        }

        if(Etype=="M" && any(loss==c("TMSE","GTMSE","TMAE","GTMAE","THAM","GTHAM",
                                       "GPL","aTMSE","aGTMSE","aGPL"))){
            yFittedSumLog <- yFittedSumLog * h;
        }

        if(all(loss!=c("GTMSE","GTMAE","GTHAM","GPL","aGPL","aGTMSE"))){
            CFValue[] <- log(CFValue);
        }

        if(any(loss==c("MAE","MAEh","MACE","TMAE","GTMAE"))){
            return(- (obsInSample*(log(2) + 1 + CFValue) + obsZero) - yFittedSumLog);
        }
        else if(any(loss==c("HAM","HAMh","CHAM","THAM","GTHAM"))){
            #### This is a temporary fix for the oes models... Needs to be done properly!!! ####
            return(- 2*(obsInSample*(log(2) + 1 + CFValue) + obsZero) - yFittedSumLog);
        }
        else if(any(loss==c("GPL","aGPL","aGTMSE"))){
            return(- 0.5 *(obsInSample*(h*log(2*pi) + 1 + CFValue) + obsZero) - yFittedSumLog);
        }
        else{
            #if(loss==c("MSE","MSEh","MSCE")) obsNonzero
            return(- 0.5 *(obsInSample*(log(2*pi) + 1 + CFValue) + obsZero) - yFittedSumLog);
        }
    }
    CFValue <- CF(B);

    if(any(occurrence==c("n","p"))){
        return(logLikFromCF(B, loss));
    }
    else{
        # Failsafe for exceptional cases when the probability is equal to zero / one,
        # when it should not have been.
        if(any(c(1-pFitted[ot==0]==0,pFitted[ot==1]==0))){
            # return(-Inf);
            ptNew <- pFitted[(pFitted!=0) & (pFitted!=1)];
            otNew <- ot[(pFitted!=0) & (pFitted!=1)];
            if(length(ptNew)==0){
                return(logLikFromCF(B, loss));
            }
            else{
                return(sum(log(ptNew[otNew==1])) + sum(log(1-ptNew[otNew==0]))
                       + logLikFromCF(B, loss));
            }
        }
        #Failsafe for cases, when data has no variability when ot==1.
        if(CFValue==0){
            if(loss=="GPL" | loss=="aGPL"){
                return(sum(log(pFitted[ot==1]))*h + sum(log(1-pFitted[ot==0]))*h);
            }
            else{
                return(sum(log(pFitted[ot==1])) + sum(log(1-pFitted[ot==0])));
            }
        }
        if(rounded){
            return(sum(log(pFitted[ot==1])) + sum(log(1-pFitted[ot==0])) - CFValue -
                       obsZero/2*(log(2*pi*B[length(B)]^2)+1));
        }
        if(loss=="GPL" | loss=="aGPL"){
            return(sum(log(pFitted[ot==1]))*h
                   + sum(log(1-pFitted[ot==0]))*h
                   + logLikFromCF(B, loss));
        }
        else{
            return(sum(log(pFitted[ot==1])) + sum(log(1-pFitted[ot==0]))
                   + logLikFromCF(B, loss));
        }
    }
}

##### *Function calculates ICs* #####
ICFunction <- function(nParam=nParam,nParamOccurrence=nParamOccurrence,
                       B,Etype=Etype,yFittedSumLog=0){
    # Information criteria are calculated with the constant part "log(2*pi*exp(1)*h+log(obs))*obs".
    # And it is based on the mean of the sum squared residuals either than sum.
    # Hyndman likelihood is: llikelihood <- obs*log(obs*cfObjective)

    nParamOverall <- nParam + nParamOccurrence;
    llikelihood <- likelihoodFunction(B,yFittedSumLog=yFittedSumLog);

    # max here is needed in order to take into account cases with higher
    ## number of parameters than observations
    ### AICc and BICc are incorrect in case of non-normal residuals!
    if(loss=="GPL"){
        coefAIC <- 2*nParamOverall*h - 2*llikelihood;
        coefBIC <- log(obsInSample)*nParamOverall*h - 2*llikelihood;
        coefAICc <- (2*obsInSample*(nParam*h + (h*(h+1))/2) /
                         max(obsInSample - nParam - 1 - h,0)
                     -2*llikelihood);
        coefBICc <- (((nParam + (h*(h+1))/2)*
                          log(obsInSample*h)*obsInSample*h) /
                         max(obsInSample*h - nParam - (h*(h+1))/2,0)
                     -2*llikelihood);
    }
    else{
        coefAIC <- 2*nParamOverall - 2*llikelihood;
        coefBIC <- log(obsInSample)*nParamOverall - 2*llikelihood;
        coefAICc <- coefAIC + 2*nParam*(nParam+1) / max(obsInSample-nParam-1,0);
        coefBICc <- (nParam * log(obsInSample) * obsInSample) / (obsInSample - nParam - 1) -2*llikelihood;
    }

    ICs <- c(coefAIC, coefAICc, coefBIC, coefBICc);
    names(ICs) <- c("AIC", "AICc", "BIC", "BICc");

    return(list(llikelihood=llikelihood,ICs=ICs));
}

# debugger <- function(){
#     continue <- readline(prompt="Continue? ")
#     if(continue=="n"){
#         stop("The user asked to stop the calculations.");
#     }
# }
