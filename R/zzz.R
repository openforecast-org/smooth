.onAttach <- function(libname, pkgname) {
    startUpMessage <- paste0("This is package \"smooth\", v",packageVersion(pkgname));
    randomNumber <- sample(c(1:100), 1);
    if(randomNumber<=4){
      if(randomNumber==1){
        startUpMessage <- paste0(startUpMessage,"\nBy the way, have you already tried adam() function from smooth?");
      }
      else if(randomNumber==2){
        startUpMessage <- paste0(startUpMessage,"\nIf you want to know more about the smooth package and forecasting, ",
                                 "you can visit my website: https://www.openforecast.org/");
      }
      else if(randomNumber==3){
        startUpMessage <- paste0(startUpMessage,"\nHave you tried adam() yet? If you want to know more about this function, ",
                                 "you can read the online monograph about it: https://www.openforecast.org/adam/");
      }
      else if(randomNumber==4){
        startUpMessage <- paste0(startUpMessage,"\nAny thoughts or suggestions about the package? ",
                                 "Have you found a bug? File an issue on github: https://github.com/openforecast-org/smooth/issues");
      }
    }
    startUpMessage <- paste0(startUpMessage,"\n");
    packageStartupMessage(startUpMessage);
}

.onLoad <- function(libname, pkgname) {
  # Load the C++ module when package loads
  Rcpp::loadModule("adamCore_module", TRUE)
}

.onUnload <- function (libpath) {
  library.dynam.unload("smooth", libpath)
}
