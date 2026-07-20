#include <RcppArmadillo.h>
#include <iostream>
#include <cmath>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

#include "headers/adamCore.h"


// ============================================================================
// WRAPPER FUNCTIONS TO CONVERT STRUCTURES TO R LISTS
// ============================================================================

namespace Rcpp {
    // Wrapper for PolyResult
    template <> SEXP wrap(const PolyResult& result) {
        return List::create(
            Named("arPolynomial") = result.arPolynomial,
            Named("iPolynomial") = result.iPolynomial,
            Named("ariPolynomial") = result.ariPolynomial,
            Named("maPolynomial") = result.maPolynomial
        );
    }

    // Wrapper for FitResult
    template <> SEXP wrap(const FitResult& result) {
        return List::create(
            Named("states") = result.states,
            Named("fitted") = result.fitted,
            Named("errors") = result.errors,
            Named("profile") = result.profile
        );
    }

    // Wrapper for OmFitGeneralResult
    template <> SEXP wrap(const OmFitGeneralResult& result) {
        return List::create(
            Named("statesA")  = result.statesA,
            Named("fittedA")  = result.fittedA,
            Named("errorsA")  = result.errorsA,
            Named("profileA") = result.profileA,
            Named("statesB")  = result.statesB,
            Named("fittedB")  = result.fittedB,
            Named("errorsB")  = result.errorsB,
            Named("profileB") = result.profileB
        );
    }

    // Wrapper for GradientSolveGeneralResult
    template <> SEXP wrap(const GradientSolveGeneralResult& result) {
        return List::create(
            Named("profileA") = result.profileA,
            Named("profileB") = result.profileB
        );
    }

    // Wrapper for ForecastResult
    template <> SEXP wrap(const ForecastResult& result) {
        return List::create(
            Named("forecast") = result.forecast
        );
    }

    // Wrapper for ErrorResult
    template <> SEXP wrap(const ErrorResult& result) {
        return List::create(
            Named("errors") = result.errors
        );
    }

    // Wrapper for SimulateResult
    template <> SEXP wrap(const SimulateResult& result) {
        return List::create(
            Named("states") = result.states,
            Named("profile") = result.profile,
            Named("data") = result.data
        );
    }

    // Wrapper for refit/rreapply
    template <> SEXP wrap(const ReapplyResult& result) {
        return List::create(
            Named("states") = result.states,
            Named("fitted") = result.fitted,
            Named("profile") = result.profile
        );
    }

    // Wrapper for reforecast
    template <> SEXP wrap(const ReforecastResult& result) {
        return List::create(
            Named("data") = result.data
        );
    }
}

// ============================================================================
// RCPP MODULES FOR EXPOSING THE CLASS TO R
// ============================================================================

RCPP_MODULE(adamCore_module) {
    class_<adamCore>("adamCore")
    .constructor<arma::uvec, char, char, char, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, bool, bool>()
    .field("flipConstant", &adamCore::flipConstant)
    .method("polynomialise", &adamCore::polynomialise)
    .method("fit", &adamCore::fit)
    .method("omfitGeneral", &adamCore::omfitGeneral)
    .method("forecast", &adamCore::forecast)
    .method("ferrors", &adamCore::ferrors)
    .method("simulate", &adamCore::simulate)
    .method("reapply", &adamCore::reapply)
    .method("gradientSolve", &adamCore::gradientSolve)
    .method("gradientSolveGeneral", &adamCore::gradientSolveGeneral)
    .method("reforecast", &adamCore::reforecast);
}
