# Time of each step of smoothEigensPolyCpp() at the C++ level, without the R call overhead
args <- commandArgs(trailingOnly=TRUE)
root <- args[1]
repeats <- as.integer(args[2])
Sys.setenv(PKG_CXXFLAGS=paste0("-I", file.path(root, "src")))
Rcpp::sourceCpp(code='
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <chrono>
#include "headers/eigenCalc.h"
// [[Rcpp::export]]
Rcpp::NumericVector profileCore(arma::mat persistence, arma::mat transition, arma::mat measurement,
                                arma::ivec lags, int repeats) {
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto micro = [&](auto start) {
        return std::chrono::duration<double, std::micro>(now() - start).count() / repeats;
    };
    arma::mat D = transition - persistence * measurement;
    arma::vec theta, phi;
    auto t = now();
    for(int i = 0; i < repeats; i++) { theta = lagPolynomial(D, lags); phi = lagPolynomial(transition, lags); }
    double polynomials = micro(t);
    t = now();
    for(int i = 0; i < repeats; i++) { arma::vec a = theta, b = phi; cancelUnitRoots(a, b, lags); }
    double cancellation = micro(t);
    cancelUnitRoots(theta, phi, lags);
    t = now();
    for(int i = 0; i < repeats; i++) { largestRootModulus(theta); }
    double bisection = micro(t);
    arma::vec c = theta.tail(theta.n_elem - 1) / theta(0);
    t = now();
    for(int i = 0; i < repeats; i++) { arimaReflection(c); }
    double stepDown = micro(t);
    bool xreg = false, delta = false;
    int obs = 1;
    t = now();
    for(int i = 0; i < repeats; i++) { smoothEigensPolyCpp(persistence, transition, measurement, lags, xreg, obs, delta); }
    double total = micro(t);
    t = now();
    for(int i = 0; i < repeats; i++) { smoothEigensCpp(persistence, transition, measurement, lags, xreg, obs, delta); }
    double perLag = micro(t);
    return Rcpp::NumericVector::create(Rcpp::Named("polynomials")=polynomials,
                                       Rcpp::Named("cancellation")=cancellation,
                                       Rcpp::Named("bisection")=bisection,
                                       Rcpp::Named("one step-down")=stepDown,
                                       Rcpp::Named("total")=total,
                                       Rcpp::Named("per-lag check")=perLag);
}')
set.seed(41)
models <- list("ETS(A,N,A), m=12"=list(g=matrix(c(0.3, 0.1)), F=diag(2), w=matrix(1, 1, 2), lags=c(1L, 12L)),
               "ETS(A,N,A,A), m=24 and 168"=list(g=matrix(c(0.3, 0.1, 0.05)), F=diag(3), w=matrix(1, 1, 3),
                                                 lags=c(1L, 24L, 168L)),
               "GUM, orders (1,1,1), lags (1,24,168)"=list(g=matrix(rnorm(3, 0, 0.15)), F=matrix(rnorm(9, 0, 0.3), 3),
                                                           w=matrix(rnorm(3, 1, 0.15), 1), lags=c(1L, 24L, 168L)))
results <- list()
for(name in names(models)) {
    m <- models[[name]]
    times <- profileCore(m$g, m$F, m$w, m$lags, repeats)
    for(step in names(times)) {
        results[[length(results) + 1]] <- list(quantity="mean time at the C++ level, microseconds",
                                               value=times[[step]], conditions=list(step=step, model=name))
        cat(name, step, times[[step]], "\n")
    }
}
path <- Sys.getenv("CLAIMSET_RESULTS")
if(nzchar(path)) {
    jsonlite::write_json(results, path, auto_unbox=TRUE, digits=NA)
}
