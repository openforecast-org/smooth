#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

#include "headers/olsCore.h"

// [[Rcpp::export]]
arma::vec olsCpp(const arma::mat& X, const arma::vec& y, double tol = 1e-7) {
    return olsCore(X, y, tol);
}
