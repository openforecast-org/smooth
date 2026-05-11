#### Helper functions used by adam() and others

#### Functions calculating the efficient number of estimated parameters of ADAM in case of backcasting ####
# This function is used internally by adam() et al. to calculate the fractional df
# This is needed for debiasing sigma in case of adam with backcasted initials
calculateBackcastingDF <- function(profilesRecentTable, lagsModelAll,
                                   etsModel, Stype, componentsNumberETSNonSeasonal,
                                   componentsNumberETSSeasonal, vecG, matF,
                                   obsInSample, lagsModelMax, indexLookupTable,
                                   adamCpp, dfForBack=TRUE){

    # Switch off if this was not required
    if(!dfForBack){
        return(0);
    }

    # The code below creates dummy states with 1 where the value was supposed to be estimated
    # Then it propagates the states to the end of sample and back
    # After that we compare it with the deterministic and get the fraction of the original df
    # that is in the final state.
    # Create a new profile, which has 1 for the initial states
    profilesRecentTableBack <- profilesRecentTable;
    for(i in 1:nrow(profilesRecentTableBack)){
        profilesRecentTableBack[i,1:lagsModelAll[i]] <- 1;
    }
    # For the deterministic, everything should be one
    # This way, the maths with seasonality adds up
    # i.e., we can do (profile/deterministic profile) and get sensible df
    # Otherwise due to fractional seasonal dfs, this would result in 1 more df than needed
    profilesRecentTableBackDeterministic <- profilesRecentTableBack;
    # Seasonality needs to be treated differently, because we estimate m-1 initials
    # We spread m-1 to the m elements to reflect the idea that we estimated only m-1
    # If we estimated all m, we would have 1 in every cell
    if(etsModel && Stype!="N"){
        for(k in 1:componentsNumberETSSeasonal){
            profilesRecentTableBack[componentsNumberETSNonSeasonal+k,
                                    1:lagsModelAll[componentsNumberETSNonSeasonal+k]] <-
                (lagsModelAll[componentsNumberETSNonSeasonal+k]-1)/lagsModelAll[componentsNumberETSNonSeasonal+k];
        }
    }
    # Record the final profile to see how states evolved
    dfs1 <- dfDiscounterFit(vecG, matF, obsInSample, lagsModelMax,
                            indexLookupTable, profilesRecentTableBack,
                            etsModel, adamCpp);
    # Record what would have happened if we had a deterministic stuff
    vecGZero <- vecG;
    vecGZero[] <- 0;
    dfs2 <- dfDiscounterFit(vecGZero, matF, obsInSample, lagsModelMax,
                            indexLookupTable, profilesRecentTableBackDeterministic,
                            etsModel, adamCpp);
    #### Calculate df
    # Record the states that are way off from the (0,1) region. They evolved enough
    discountedStates <- (dfs1$profileRecent>dfs2$profileRecent | dfs1$profileRecent<0);
    # Those states have no impact on the final df
    dfs1$profileRecent[discountedStates] <- 0;
    # For the others, take a proportion from the original ones to see how much they evolved
    dfs1$profileRecent[] <- dfs1$profileRecent/dfs2$profileRecent;
    # Finally, take the sum to get the df estimate
    # na.rm is needed to avoid NaNs due to 0/0
    nStatesBackcasting <- sum(dfs1$profileRecent, na.rm=TRUE);

    return(nStatesBackcasting);
}

# The function calculates the discounted number of degrees of freedom for the model
# This is the same as calculateBackcastingDF, but works for adam objects
dfDiscounter <- function(object){
    lagsModelAll <- modelLags(object);
    lagsModelMax <- max(lagsModelAll);
    obsInSample <- nobs(object);
    components <- componentsDefiner(object);
    vecG <- matrix(object$persistence);
    matF <- object$transition;
    xregNumber <- 0
    if(is.list(object$initial) && !is.null(object$initial$xreg)){
        xregNumber[] <- length(object$initial$xreg);
    }

    etsModel <- etsChecker(object);

    Etype <- errorType(object);
    Ttype <- trendType(object);
    Stype <- seasonType(object);

    componentsNumberETSSeasonal <- components$componentsNumberETSSeasonal;
    componentsNumberETSNonSeasonal <- components$componentsNumberETSNonSeasonal;
    componentsNumberETS <- components$componentsNumberETS;
    componentsNumberARIMA <- components$componentsNumberARIMA;

    constantRequired <- !is.null(object$constant);
    adamETS <- adamETSChecker(object);

    # Create C++ adam class, which will then use fit, forecast etc methods
    adamCpp <- new(adamCore,
                   lagsModelAll, Etype, Ttype, Stype,
                   componentsNumberETSNonSeasonal,
                   componentsNumberETSSeasonal,
                   componentsNumberETS, componentsNumberARIMA,
                   xregNumber, length(lagsModelAll),
                   constantRequired, adamETS);

    adamProfileCreated <- adamProfileCreator(lagsModelAll, lagsModelMax, obsInSample);
    indexLookupTable <- adamProfileCreated$lookup;
    profilesRecentTableBack <- adamProfileCreated$recent;


    # Create a new profile, which has 1 for the initial states
    # profilesRecentTableBack <- matrix(0, nStates, lagsModelMax);
    for(i in 1:nrow(profilesRecentTableBack)){
        profilesRecentTableBack[i,1:lagsModelAll[i]] <- 1;
    }

    # Seasonality needs to be treated differently, because we estimate m-1 initials
    # We spread m-1 to the m elements to reflect the idea that we estimated only m-1
    # If we estimated all m, we would have 1 in every cell
    if(etsModel && Stype!="N"){
        for(k in 1:componentsNumberETSSeasonal){
            profilesRecentTableBack[componentsNumberETSNonSeasonal+k,
                                    1:lagsModelAll[componentsNumberETSNonSeasonal+k]] <-
                (lagsModelAll[componentsNumberETSNonSeasonal+k]-1)/lagsModelAll[componentsNumberETSNonSeasonal+k];
        }
    }

    # Record the final profile to see how states evolved
    dfs1 <- dfDiscounterFit(vecG, matF,
                            obsInSample, lagsModelMax,
                            indexLookupTable, profilesRecentTableBack,
                            etsModel, adamCpp);

    # Record what would have happened if we had a deterministic stuff
    vecG[] <- 0;
    # For the deterministic, everything should be one
    # This way, the maths with seasonality adds up
    # i.e., we can do (profile/deterministic profile) and get sensible df
    # Otherwise due to fractional seasonal dfs, this would result in 1 more df than needed
    profilesRecentTableBackDeterministic <- profilesRecentTableBack;
    profilesRecentTableBackDeterministic[profilesRecentTableBackDeterministic!=0] <- 1;
    dfs2 <- dfDiscounterFit(vecG, matF,
                            obsInSample, lagsModelMax,
                            indexLookupTable, profilesRecentTableBackDeterministic,
                            etsModel, adamCpp);

    #### Calculate df
    # Record the states that are way off from the (0,1) region. They evolved enough
    discountedStates <- (dfs1$profileRecent>dfs2$profileRecent | dfs1$profileRecent<0);
    profileRecent <- dfs1$profileRecent;
    # Those states have no impact on the final df
    profileRecent[discountedStates] <- 0;
    # For the others, take a proportion from the original ones to see how much they evolved
    profileRecent[] <- profileRecent/dfs2$profileRecent;
    # Finally, take the sum to get the df estimate
    # na.rm is needed to avoid 0/0
    df <- sum(profileRecent, na.rm=TRUE);

    return(list(profile1=t(dfs1$profileRecent), profileInitial=t(profilesRecentTableBack),
                profile2=t(dfs2$profileRecent), df=df));
    # return(df);
}

# Te function fits a simple adam with unit states to the zero data to see how they propagate over time
dfDiscounterFit <- function(persistence, transition,
                            obsInSample, lagsModelMax,
                            indexLookupTable, profilesRecentTableBack,
                            etsModel, adamCpp){

    # This is the sample to model to reflect the backcasted period (back and forth)
    # lagsModelMax appears because we have "refineHead"
    obsInSampleBackcasting <- obsInSample*2+lagsModelMax-1;
    nStates <- ncol(transition);
    # State matrix that has columns similar to obsStates
    matVtBack <- matrix(1, nStates, obsInSampleBackcasting + lagsModelMax);
    # Measurement matrix with the new sample
    matWtBack <- matrix(1, obsInSampleBackcasting, nStates);
    # indexLookupTable for the new data. This is similar to doing forth and back pass
    indexLookupTableBack <- cbind(indexLookupTable,indexLookupTable[,(ncol(indexLookupTable)-1):1, drop=FALSE]);

    # The new data. This is just zeroes to see how the df effect evaporates
    # But it's not exactly zero, because otherwise multiplicative models won't work
    yInSampleBack <- matrix(1e-100, obsInSampleBackcasting, 1);
    # New occurrence, which is 1 everywhere
    otBack <- matrix(1, obsInSampleBackcasting, 1);

    # Fit the model to the data
    adamFittedBack <- adamCpp$fit(matVtBack, matWtBack,
                                  transition, persistence,
                                  indexLookupTableBack, profilesRecentTableBack,
                                  yInSampleBack, otBack,
                                  FALSE, 1,
                                  FALSE, "n");

    # Get the final profile. It now contains the discounted df for the start of the data
    return(list(profileRecent=adamFittedBack$profile));
                # states=tail(t(adamFittedBack$states), lagsModelMax)));
}


#### Small technical functions returning types of models and components ####
# Function defines number of components based on the model type
componentsDefiner <- function(object){
    etsModel <- etsChecker(object);
    arimaModel <- arimaChecker(object);
    cesModel <- cesChecker(object);
    gumModel <- gumChecker(object);
    ssarimaModel <- ssarimaChecker(object);
    sparmaModel <- sparmaChecker(object);

    if(cesModel){
        componentsNumberETS <- componentsNumberETSSeasonal <- componentsNumberETSNonSeasonal <- 0;
        componentsNumberARIMA <- length(object$initial$nonseasonal);
        # If seasonal is formed via a matrix, this must be "simple" or a "full" model
        if(!is.null(object$initial$seasonal)){
            # If this is not a matrix then we have only one seasonal component
            if(is.matrix(object$initial$seasonal)){
                componentsNumberARIMA[] <- componentsNumberARIMA + nrow(object$initial$seasonal);
            }
            else{
                componentsNumberARIMA[] <- componentsNumberARIMA + 1;
            }
        }
    }
    else if(gumModel){
        componentsNumberETS <- componentsNumberETSSeasonal <- componentsNumberETSNonSeasonal <- 0;
        componentsNumberARIMA <- sum(orders(object));
    }
    else if(ssarimaModel){
        arimaOrders <- orders(object);
        lags <- lags(object);
        componentsNumberETS <- componentsNumberETSSeasonal <- componentsNumberETSNonSeasonal <- 0;
        componentsNumberARIMA <- max(arimaOrders$ar %*% lags + arimaOrders$i %*% lags, arimaOrders$ma %*% lags);
    }
    else if(sparmaModel){
        componentsNumberETS <- componentsNumberETSSeasonal <- componentsNumberETSNonSeasonal <- 0;
        componentsNumberARIMA <- length(modelLags(object));
    }
    else{
        if(!is.null(object$initial$seasonal)){
            if(is.list(object$initial$seasonal)){
                componentsNumberETSSeasonal <- length(object$initial$seasonal);
            }
            else{
                componentsNumberETSSeasonal <- 1;
            }
        }
        else{
            componentsNumberETSSeasonal <- 0;
        }
        componentsNumberETSNonSeasonal <- length(object$initial$level) + length(object$initial$trend);
        componentsNumberETS <- componentsNumberETSNonSeasonal + componentsNumberETSSeasonal;
        componentsNumberARIMA <- sum(substr(colnames(object$states),1,10)=="ARIMAState");
    }

    # See if constant is required
    constantRequired <- !is.null(object$constant);

    return(list(componentsNumberETS=componentsNumberETS,
                componentsNumberETSNonSeasonal=componentsNumberETSNonSeasonal,
                componentsNumberETSSeasonal=componentsNumberETSSeasonal,
                componentsNumberARIMA=componentsNumberARIMA,
                constantRequired=constantRequired))
}


etsChecker <- function(object){
    return(any(unlist(gregexpr("ETS",object$model))!=-1));
}

arimaChecker <- function(object){
    return(any(unlist(gregexpr("ARIMA",object$model))!=-1));
}

gumChecker <- function(object){
    return(smoothType(object)=="GUM");
}

ssarimaChecker <- function(object){
    return(smoothType(object)=="SSARIMA");
}

cesChecker <- function(object){
    return(smoothType(object)=="CES");
}

sparmaChecker <- function(object){
    return(smoothType(object)=="SpARMA");
}


#### The function that returns the eigen values for specified parameters ADAM ####
smoothEigens <- function(persistence, transition, measurement,
                         lagsModelAll, xregModel, obsInSample){
    persistenceNames <- names(persistence);
    hasDelta <- any(substr(persistenceNames,1,5)=="delta");
    xregNumber <- sum(substr(persistenceNames,1,5)=="delta");
    constantRequired <- any(persistenceNames %in% c("constant","drift"));

    return(smoothEigensR(persistence, transition, measurement,
                         lagsModelAll, xregModel, obsInSample,
                         hasDelta, xregNumber, constantRequired));

    # lagsUnique <- unique(lagsModelAll);
    # lagsUniqueLength <- length(lagsUnique);
    # eigenValues <- vector("numeric", lagsUniqueLength);
    # # Check eigen values per unique component (unique lag)
    # #### !!!! Eigen values checks do not work for xreg. So, check the average condition
    # if(xregModel && any(substr(names(persistence),1,5)=="delta")){
    #     # We check the condition on average
    #     return(eigen((transition -
    #                       diag(as.vector(persistence)) %*%
    #                       t(measurementInverter(measurement[1:obsInSample,,drop=FALSE])) %*%
    #                       measurement[1:obsInSample,,drop=FALSE] / obsInSample),
    #                  symmetric=FALSE, only.values=TRUE)$values);
    # }
    # else{
    #     for(i in 1:lagsUniqueLength){
    #         eigenValues[which(lagsModelAll==lagsUnique[i])] <-
    #             eigen(transition[lagsModelAll==lagsUnique[i], lagsModelAll==lagsUnique[i], drop=FALSE] -
    #                       persistence[lagsModelAll==lagsUnique[i],,drop=FALSE] %*%
    #                       measurement[obsInSample,lagsModelAll==lagsUnique[i],drop=FALSE],
    #                   symmetric=FALSE, only.values=TRUE)$values
    #     }
    # }
    # return(eigenValues);
}

