#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <armadillo>

#ifndef PYTHON_BUILD
#define PYTHON_BUILD 1
#endif
#include <carma>

#include "../headers/olsCore.h"

namespace py = pybind11;

py::array_t<double> ols_wrapper(const arma::mat& X, const arma::vec& y, double tol) {
    arma::vec b = olsCore(X, y, tol);
    py::array_t<double> arr({static_cast<py::ssize_t>(b.n_elem)});
    auto buf = arr.mutable_unchecked<1>();
    for(size_t i = 0; i < b.n_elem; i++) {
        buf(i) = b(i);
    }
    return arr;
}

PYBIND11_MODULE(_ols, m) {
    m.doc() = "Shared C++ OLS solver (pivoted QR with rank cutoff)";
    m.def(
        "ols",
        &ols_wrapper,
        "Least-squares solution to X * b = y via pivoted QR with rank cutoff.",
        py::arg("X"),
        py::arg("y"),
        py::arg("tol") = 1e-7
    );
}
