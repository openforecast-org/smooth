covarAnal <- function(lagsModel, h, measurement, transition, persistence, s2){
    # Function returns analytical conditional h-steps ahead covariance matrix
    # This is used in covar() method and in the construction of parametric prediction intervals
    # The algorithm is in the C++ shared with Python (src/headers/covarAnalCore.h)
    return(covarAnalCpp(lagsModel, h, measurement, transition, as.vector(persistence), s2));
}

adamVarAnal <- function(lagsModel, h, measurement, transition, persistence, s2){
    #### The function returns variances for the multiplicative error ETS models
    # Prepare the necessary parameters
    lagsUnique <- unique(lagsModel);
    steps <- sort(lagsUnique[lagsUnique<=h]);
    stepsNumber <- length(steps);
    nComponents <- nrow(transition);
    k <- length(persistence);

    # Prepare the persistence array and measurement matrix
    arrayPersistenceQ <- array(0,c(nComponents,nComponents,stepsNumber));
    matrixMeasurement <- matrix(measurement,1,nComponents);

    # Form partial matrices for different steps
    for(i in 1:stepsNumber){
        arrayPersistenceQ[,lagsModel==steps[i],i] <- diag(as.vector(persistence),k,k)[,lagsModel==steps[i]];
    }

    ## The matrices that will be used in the loop
    matrixPersistenceQ <- matrix(0,nComponents,nComponents);
    # The interrim value (Q), which accumulates the values before taking logs
    IQ <- vector("numeric",1);
    Ik <- diag(k);

    # The vector of variances
    varMat <- rep(0, h);


    if(h>1){
        # Start the loop for varMat
        for(i in 2:h){
            IQ[] <- 0;
            # Form the correct interrim Q that will be used for variances
            for(k in 1:sum(steps<i)){
                if(k==0){
                    next;
                }
                matrixPersistenceQ[] <- arrayPersistenceQ[,,k];
                IQ[] <- IQ[] + sum(diag((matrixPowerWrap(Ik + matrixPowerWrap(matrixPersistenceQ,2)*s2,
                                                         ceiling(i/lagsUnique[k])-1) - Ik)));
            }
            varMat[i] <- log(IQ);
        }
    }
    varMat[] <- exp(varMat)*(1+s2);
    varMat[1] <- varMat[1] - 1;
    varMat[-1] <- varMat[-1] + s2;

    return(varMat);
}
