#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

#include "headers/covarAnalCore.h"

// The analytical covariance matrix of the multistep errors (src/headers/covarAnalCore.h)
// [[Rcpp::export]]
arma::mat covarAnalCpp(const arma::vec& lagsModel, const unsigned int h, const arma::mat& measurement,
                       const arma::mat& transition, const arma::vec& persistence, const double s2) {
    return covarAnalCore(lagsModel, h, measurement.row(0), transition, persistence, s2);
}
