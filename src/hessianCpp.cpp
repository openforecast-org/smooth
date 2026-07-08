// Rcpp wrapper around the shared C++ numerical Hessian implementation
// (headers/hessianCore.h). Used by vcov.adam / vcov.om / vcov.omg in R
// instead of pracma::hessian — keeps the algorithm byte-identical between
// R and Python and lets us drop the pracma dependency.

#include <Rcpp.h>
#include "headers/hessianCore.h"

#include <vector>

using namespace Rcpp;

// Hessian by central finite difference. Mirrors pracma::hessian exactly:
// single absolute step size h, no Richardson extrapolation.
//
// Args:
//   f   : R function of one numeric vector argument returning a scalar.
//         Any extra arguments must be captured in a closure.
//   x0  : point at which to evaluate the Hessian.
//   h   : finite-difference step. Defaults to .Machine$double.eps^(1/4).
//
// Returns: n×n symmetric numeric matrix.
//
// [[Rcpp::export]]
NumericMatrix hessianCpp(Function f, NumericVector x0,
                         double h = 1.220703125e-4) {
    const std::size_t n = x0.size();

    // Copy x0 into a std::vector once.
    std::vector<double> base(n);
    for (std::size_t k = 0; k < n; ++k) {
        base[k] = x0[k];
    }

    // Callable that hands a std::vector<double> back through R for evaluation.
    auto eval = [&](const std::vector<double>& v) -> double {
        NumericVector x(n);
        for (std::size_t k = 0; k < n; ++k) {
            x[k] = v[k];
        }
        SEXP res = f(x);
        return as<double>(res);
    };

    std::vector<double> H_flat = smooth_hessian::hessian(eval, base, h);

    NumericMatrix H(static_cast<int>(n), static_cast<int>(n));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            H(static_cast<int>(i), static_cast<int>(j)) = H_flat[i * n + j];
        }
    }
    return H;
}
