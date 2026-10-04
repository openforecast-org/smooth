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

// [[Rcpp::export]]
List arimaHRSelectCpp(const arma::vec& y, const arma::uvec& arOrders, const arma::uvec& maOrders,
                      const arma::uvec& lags, int screen, int arMax, int maMax, bool bounded) {
    HRSelectResult result = arimaHRSelectCore(y, arOrders, maOrders, lags, screen, arMax, maMax, bounded);
    return List::create(Named("orders") = result.orders, Named("parameters") = result.parameters,
                        Named("innovations") = result.innovations);
}

// [[Rcpp::export]]
arma::vec arimaParameterBoundsCpp(const arma::vec& values, int j, double sign) {
    return arimaParameterBounds(values, j, sign);
}

// The Householder QR of a fixed design and its least squares (src/headers/olsCore.h)
// [[Rcpp::export]]
List householderQRCpp(const arma::mat& X) {
    HouseholderQR d = householderQR(X);
    return List::create(Named("qr") = d.qr, Named("qraux") = d.qraux, Named("rDiag") = d.rDiag);
}

// [[Rcpp::export]]
arma::vec householderCoefCpp(const arma::mat& qr, const arma::vec& qraux, const arma::vec& rDiag,
                             const arma::vec& y) {
    return householderCoef(HouseholderQR{qr, qraux, rDiag}, y);
}

// [[Rcpp::export]]
arma::vec householderResidCpp(const arma::mat& qr, const arma::vec& qraux, const arma::vec& rDiag,
                              const arma::vec& y) {
    return householderResid(HouseholderQR{qr, qraux, rDiag}, y);
}
