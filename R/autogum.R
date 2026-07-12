#' \code{auto.gum()} function selects the order of GUM model based on information criteria,
#' using fancy branch and bound mechanism. The function checks several GUM models
#' (see \link[smooth]{gum}) and selects the best one based on the specified
#' information criterion.
#'
#' @seealso \code{\link[smooth]{gum}, \link[smooth]{es},
#' \link[smooth]{ces}, \link[smooth]{sim.es}, \link[smooth]{ssarima}}
#'
#' @examples
#'
#' x <- rnorm(50,100,3)
#'
#' # The best GUM model for the data
#' ourModel <- auto.gum(x, orders=2, lags=4, h=18, holdout=TRUE)
#'
#' \donttest{summary(ourModel)}
#'
#' @rdname gum
#' @export
auto.gum <- function(y, orders=3, lags=frequency(y), type=c("additive","multiplicative","select"),
                     initial=c("backcasting","optimal","two-stage","complete","gradient"), ic=c("AICc","AIC","BIC","BICc"),
                     loss=c("likelihood","MSE","MAE","HAM","MSEh","TMSE","GTMSE","MSCE","GPL"),
                     h=0, holdout=FALSE, bounds=c("usual","admissible","none"), silent=TRUE,
                     xreg=NULL, regressors=c("use","select","adapt","integrate"), ...){
# Function estimates several GUM models and selects the best one using the selected information criterion.
#
#    Copyright (C) 2017 - Inf  Ivan Svetunkov

# Start measuring the time of calculations
    startTime <- Sys.time();

    ### Depricate the old parameters
    ellipsis <- list(...)

    # Record the parental environment. Needed for optimal initialisation
    env <- environment();

    # If this is Mcomp data, then take the frequency from it
    if(any(class(y)=="Mdata") && lags==frequency(y)){
        lags <- frequency(y$x);
        yInSample <- y$x;
        # Measure the sample size based on what was provided as data
        obsInSample <- length(y$x) - holdout*h;
    }
    else{
        # Measure the sample size based on what was provided as data
        obsInSample <- length(y) - holdout*h;
        yInSample <- y[1:obsInSample];
    }

    if(any(is.complex(c(orders,lags)))){
        stop("Complex numbers? Really? Be serious! This is GUM, not CES!",call.=FALSE);
    }

    if(any(c(orders)<0)){
        stop("Funny guy! How am I gonna construct a model with negative maximum order?",call.=FALSE);
    }

    if(any(c(lags)<0)){
        stop("Right! Why don't you try complex lags then, mister smart guy?",call.=FALSE);
    }

    if(any(c(lags,orders)==0)){
        stop("Sorry, but we cannot construct GUM model with zero lags / orders.",call.=FALSE);
    }

    type <- match.arg(type);
    # Check if the multiplicative model is possible
    if(any(type==c("select","multiplicative"))){
        if(any(yInSample<=0)){
            warning("Multiplicative model can only be used on positive data. Switching to the additive one.",call.=FALSE);
            type <- "additive";
        }
        if(type=="select"){
            type <- c("additive","multiplicative");
        }
    }

    initial <- match.arg(initial);

    ic <- match.arg(ic);
    IC <- switch(ic,
                 "AIC"=AIC,
                 "AICc"=AICc,
                 "BIC"=BIC,
                 "BICc"=BICc);

    ICsFinal <- rep(NA,length(type));
    lagsFinal <- list(NA);
    ordersFinal <- list(NA);
    # List of estimated parameters
    BValues <- list(NA);
    BFinal <- list(NA);

    if(!silent){
        if(lags>12){
            message(paste0("You have large lags: ",lags,". So, the calculation may take some time."));
            if(lags<24){
                message(paste0("Go get some coffee, or tea, or whatever, while we do the work here.\n"));
            }
            else{
                message(paste0("Go for a lunch or something, while we do the work here.\n"));
            }
        }
        if(orders>3){
            message(paste0("Beware that you have specified large orders: ",orders,
                           ". This means that the calculations may take a lot of time.\n"));
        }
    }

    for(t in 1:length(type)){
        ICs <- rep(NA,lags);
        lagsBest <- NULL

        if((!silent) & length(type)!=1){
            cat(paste0("Checking model with a type=\"",type[t],"\".\n"));
        }

    #### Preliminary loop ####
        #Checking all the models with lag from 1 to lags
        if(!silent){
            progressBar <- c("/","\u2014","\\","|");
            cat("Starting preliminary loop: ");
            cat(paste0(rep(" ",9+nchar(lags)),collapse=""));
        }
        for(i in 1:lags){
            gumModel <- gum(y, orders=c(1), lags=c(i), type=type[t],
                            initial=initial, loss=loss,
                            h=h, holdout=holdout,
                            bounds=bounds, silent=TRUE, environment=env,
                            xreg=xreg, regressors=regressors, ...);
            ICs[i] <- IC(gumModel);
            BValues[[i]] <- gumModel$B;
            if(!silent){
                cat(paste0(rep("\b",nchar(paste0(i-1," out of ",lags))),collapse=""));
                cat(paste0(i," out of ",lags));
            }
        }

        ##### Checking all the possible lags ####
        if(!silent){
            cat(". Done.\n");
            cat("Searching for appropriate lags:  ");
        }
        lagsBest <- c(which(ICs==min(ICs))[1],lagsBest);
        ICsBest <- 1E100;
        while(min(ICs)<ICsBest){
            for(i in 1:lags){
                if(!silent){
                    cat("\b");
                    cat(progressBar[(i/4-floor(i/4))*4+1]);
                }
                if(any(i==lagsBest)){
                    next;
                }
                ordersTest <- rep(1,length(lagsBest)+1);
                lagsTest <- c(i,lagsBest);
                nComponents <- sum(ordersTest);
                # We don't estimate measurement by default, so the nParamMax is lower
                # nParamMax <- (1 + nComponents + nComponents + (nComponents^2)
                nParamMax <- (1 + nComponents + (nComponents^2)
                              + (ordersTest %*% lagsTest)*(initial=="optimal"));
                if(obsInSample<=nParamMax){
                    ICs[i] <- 1E100;
                    next;
                }
                gumModel <- gum(y, orders=ordersTest, lags=lagsTest, type=type[t],
                                initial=initial, loss=loss,
                                h=h, holdout=holdout,
                                bounds=bounds, silent=TRUE, environment=env,
                                xreg=xreg, regressors=regressors, ...);
                ICs[i] <- IC(gumModel);
                BValues[[i]] <- gumModel$B;
            }
            iBest <- which.min(ICs)[1];
            if(!any(iBest==lagsBest)){
                lagsBest <- c(iBest, lagsBest);
            }
            ICsBest <- min(ICs);
        }
        BBest <- BValues[[iBest]]

        #### Checking all the possible orders ####
        if(!silent){
            cat("\b");
            cat("We found them!\n");
            cat("Searching for appropriate orders:  ");
        }
        ICsBest <- min(ICs);
        ICs <- array(c(1:(orders^length(lagsBest))),rep(orders,length(lagsBest)));
        ICs[1] <- ICsBest;
        BValues[[1]] <- BBest;
        for(i in 1:length(ICs)){
            if(!silent){
                cat("\b");
                cat(progressBar[(i/4-floor(i/4))*4+1]);
            }
            if(i==1){
                next;
            }
            ordersTest <- which(ICs==ICs[i],arr.ind=TRUE);
            nComponents <- sum(ordersTest);
            nParamMax <- (1 + nComponents + (nComponents^2)
                          + (ordersTest %*% lagsBest)*(initial=="optimal"));
            if(obsInSample<=nParamMax){
                ICs[i] <- NA;
                next;
            }
            gumModel <- gum(y, orders=ordersTest, lags=lagsBest, type=type[t],
                            initial=initial, loss=loss,
                            h=h, holdout=holdout,
                            bounds=bounds, silent=TRUE, environment=env,
                            xreg=xreg, regressors=regressors, ...);
            ICs[i] <- IC(gumModel);
            BValues[[i]] <- gumModel$B;
        }
        ordersBest <- which(ICs==min(ICs,na.rm=TRUE),arr.ind=TRUE);
        BBest <- BValues[[which.min(ICs)]];
        if(!silent){
            cat("\b");
            cat("Orders found.\n");
        }

        ICsFinal[t] <- min(ICs,na.rm=TRUE);
        lagsFinal[[t]] <- lagsBest;
        ordersFinal[[t]] <- ordersBest;
        BFinal[[t]] <- BBest;
    }
    t <- which.min(ICsFinal)[1];

    if(!silent){
        cat("Reestimating the model. ");
    }

    bestModel <- gum(y, orders=ordersFinal[[t]], lags=lagsFinal[[t]], type=type[t],
                     initial=initial, loss=loss,
                     h=h, holdout=holdout,
                     bounds=bounds, silent=TRUE, environment=env,
                     xreg=xreg, regressors=regressors, B=BFinal[[t]], maxeval=1, ...);

    bestModel$timeElapsed <- Sys.time()-startTime;

    if(!silent){
        cat("Done!\n");
    }

##### Make a plot #####
    if(!silent){
        plot(bestModel, 7);
    }

    return(bestModel);
}
