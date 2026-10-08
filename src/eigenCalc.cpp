#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

#include "headers/eigenCalc.h"

// [[Rcpp::export]]
arma::vec smoothEigensR(const arma::mat& persistence,
                        const arma::mat& transition,
                        const arma::mat& measurement,
                        const arma::ivec& lagsModelAll,
                        bool xregModel,
                        int obsInSample,
                        bool hasDelta,
                        int xregNumber = 0,
                        bool constantRequired = false) {
    return smoothEigensCpp(persistence,
                           transition,
                           measurement,
                           lagsModelAll,
                           xregModel,
                           obsInSample,
                           hasDelta,
                           xregNumber,
                           constantRequired);
}

// [[Rcpp::export]]
arma::vec eigenModuliCpp(const arma::mat& A) {
    return eigenModuliCore(A);
}
