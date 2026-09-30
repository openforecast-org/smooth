#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

#include "headers/olsCore.h"

// [[Rcpp::export]]
arma::vec olsCpp(const arma::mat& X, const arma::vec& y, double tol = 1e-7) {
    return olsCore(X, y, tol);
}

#include "headers/arimaInitCore.h"

// [[Rcpp::export]]
arma::vec arimaHRCpp(const arma::vec& y, const arma::uvec& arOrders, const arma::uvec& maOrders,
                     const arma::uvec& lags, bool arEstimate, bool maEstimate,
                     const arma::vec& armaParameters, const arma::uvec& useLevel, bool bounded) {
    return arimaHRCore(y, arOrders, maOrders, lags, arEstimate, maEstimate, armaParameters, useLevel, bounded);
}
