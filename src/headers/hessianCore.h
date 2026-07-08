// Shared numerical Hessian implementation used by both the R (Rcpp)
// wrapper in src/hessianCpp.cpp and the Python (pybind11) wrapper in
// src/python/numDeriv.cpp. Single source of truth so both languages get
// bit-identical results given the same inputs.
//
// Algorithm: plain central finite difference (no Richardson extrapolation),
// with per-parameter relative step sizes:
//
//   h_i = h * max(|x0[i]|, 1)
//
//   diagonal:  H[i,i] = (f(x - h_i e_i) - 2 f(x) + f(x + h_i e_i)) / h_i^2
//   off-diag:  H[i,j] = (f(x + h_i e_i + h_j e_j) - f(x + h_i e_i - h_j e_j)
//                       - f(x - h_i e_i + h_j e_j)
//                       + f(x - h_i e_i - h_j e_j)) / (4 h_i h_j)
//
// Why per-parameter relative steps. A pure absolute step h applied to a
// parameter of magnitude |x_i| ≫ 1 yields a relative perturbation h/|x_i|
// that sits below the cost-function precision floor; the four FD evals
// then differ only in floating-point noise and dividing by 4 h^2 produces
// either NaN or random small values for the Hessian entry. This bites
// hardest on ADAM with initial="optimal" or "two-stage", where B carries
// the initial level / trend / seasonal states whose magnitudes mirror the
// data (e.g. level ≈ 280 for AirPassengers). The relative-step rule
// max(|x_i|, 1) leaves O(1)-or-smaller parameters at the original
// absolute step (backcasting / OM / OMG behaviour is bit-equivalent to
// before) and scales up for large-magnitude parameters so the
// perturbation captures the local curvature.
//
// The default base h = .Machine$double.eps^(1/4) is set by the wrappers,
// not here.
//
// f is supplied as a templated callable so the same body works with
// std::function<double(std::vector<double>)> (pybind11) and a small
// callable wrapping an Rcpp::Function (Rcpp).

#ifndef HESSIAN_CORE_H
#define HESSIAN_CORE_H

#include <vector>
#include <cstddef>
#include <cmath>
#include <algorithm>

namespace smooth_hessian {

// Eval f on a vector-typed callable. Returns the n×n symmetric Hessian as a
// row-major std::vector<double> of size n*n so the wrappers can copy it into
// their native matrix types without further allocation.
template <typename F>
std::vector<double> hessian(F f, const std::vector<double>& x0, double h)
{
    const std::size_t n = x0.size();
    std::vector<double> H(n * n, 0.0);

    // Per-parameter step h_i = h * max(|x0[i]|, 1). See the file-level
    // comment for the rationale.
    std::vector<double> hv(n);
    for (std::size_t i = 0; i < n; ++i) {
        hv[i] = h * std::max(std::abs(x0[i]), 1.0);
    }

    const double f0 = f(x0);

    std::vector<double> xp(x0), xm(x0);
    std::vector<double> xpp(x0), xpm(x0), xmp(x0), xmm(x0);

    for (std::size_t i = 0; i < n; ++i) {
        const double hi  = hv[i];
        const double hi2 = hi * hi;

        // Diagonal entry: 2-eval central FD with per-parameter step
        xp = x0; xm = x0;
        xp[i] += hi;
        xm[i] -= hi;
        H[i * n + i] = (f(xm) - 2.0 * f0 + f(xp)) / hi2;

        // Off-diagonal entries: 4-eval central FD, symmetric copy.
        for (std::size_t j = i + 1; j < n; ++j) {
            const double hj    = hv[j];
            const double denom = 4.0 * hi * hj;
            xpp = x0; xpm = x0; xmp = x0; xmm = x0;
            xpp[i] += hi; xpp[j] += hj;
            xpm[i] += hi; xpm[j] -= hj;
            xmp[i] -= hi; xmp[j] += hj;
            xmm[i] -= hi; xmm[j] -= hj;
            const double v = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / denom;
            H[i * n + j] = v;
            H[j * n + i] = v;
        }
    }

    return H;
}

} // namespace smooth_hessian

#endif // HESSIAN_CORE_H
