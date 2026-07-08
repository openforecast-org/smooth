// pybind11 wrapper around the shared numerical Hessian implementation
// (src/headers/hessianCore.h). The same header is used by the Rcpp wrapper
// (src/hessianCpp.cpp) so R and Python compute the Hessian via byte-identical
// code paths — given the same inputs, results are bit-identical.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include <vector>

#include "../headers/hessianCore.h"

namespace py = pybind11;

py::array_t<double> hessian(std::function<double(py::array_t<double>)> f,
                            py::array_t<double> x0, double h) {
    auto x = x0.unchecked<1>();
    const py::ssize_t n = x.shape(0);

    std::vector<double> base(n);
    for (py::ssize_t k = 0; k < n; ++k) {
        base[k] = x(k);
    }

    auto eval = [&](const std::vector<double>& v) -> double {
        py::array_t<double> a(static_cast<py::ssize_t>(v.size()));
        auto ab = a.mutable_unchecked<1>();
        for (py::ssize_t k = 0; k < static_cast<py::ssize_t>(v.size()); ++k) {
            ab(k) = v[k];
        }
        return f(a);
    };

    std::vector<double> H_flat = smooth_hessian::hessian(eval, base, h);

    py::array_t<double> H({n, n});
    auto Hb = H.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < n; ++i) {
        for (py::ssize_t j = 0; j < n; ++j) {
            Hb(i, j) = H_flat[i * n + j];
        }
    }
    return H;
}

PYBIND11_MODULE(_numDeriv, m) {
    m.def("hessian", &hessian, py::arg("f"), py::arg("x0"),
          py::arg("h") = 1.220703125e-04);  // ~ .Machine$double.eps^(1/4)
}
