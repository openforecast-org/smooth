#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

#include "headers/eigenCalc.h"

// [[Rcpp::export]]
arma::vec eigenModuliCpp(const arma::mat& A) {
    return eigenModuliCore(A);
}

// The admissibility of the model (smoothEigensCpp()), with the cancelled AR and MA polynomials
// of its ARIMA form as attributes
// [[Rcpp::export]]
Rcpp::NumericVector smoothEigensR(const arma::mat& persistence,
                                  const arma::mat& transition,
                                  const arma::mat& measurement,
                                  const arma::ivec& lagsModelAll,
                                  bool xregModel,
                                  int obsInSample,
                                  bool hasDelta,
                                  int xregNumber = 0,
                                  bool constantRequired = false) {
    SmoothEigensResult result = smoothEigensCpp(persistence, transition, measurement, lagsModelAll,
                                                xregModel, obsInSample, hasDelta, xregNumber,
                                                constantRequired);
    Rcpp::NumericVector modulus = Rcpp::NumericVector::create(result.modulus);
    modulus.attr("ariPolynomial") = Rcpp::NumericVector(result.ariPolynomial.begin(), result.ariPolynomial.end());
    modulus.attr("maPolynomial") = Rcpp::NumericVector(result.maPolynomial.begin(), result.maPolynomial.end());
    return modulus;
}
