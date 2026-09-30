#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <armadillo>

#ifndef PYTHON_BUILD
#define PYTHON_BUILD 1
#endif
#include <carma>

#include "../headers/olsCore.h"
#include "../headers/arimaInitCore.h"

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

py::array_t<double> arima_hr_wrapper(const arma::vec& y, const arma::uvec& ar_orders,
                                     const arma::uvec& ma_orders, const arma::uvec& lags,
                                     bool ar_estimate, bool ma_estimate,
                                     const arma::vec& arma_parameters, const arma::uvec& use_level,
                                     bool bounded) {
    arma::vec b = arimaHRCore(y, ar_orders, ma_orders, lags, ar_estimate, ma_estimate,
                              arma_parameters, use_level, bounded);
    py::array_t<double> arr({static_cast<py::ssize_t>(b.n_elem)});
    auto buf = arr.mutable_unchecked<1>();
    for(size_t i = 0; i < b.n_elem; i++) {
        buf(i) = b(i);
    }
    return arr;
}

py::array_t<double> arima_parameter_bounds_wrapper(const arma::vec& values, arma::uword j,
                                                   double sign) {
    arma::vec b = arimaParameterBounds(values, j, sign);
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
    m.def(
        "arima_hr",
        &arima_hr_wrapper,
        "Hannan-Rissanen starting values of the AR / MA parameters (see arimaInitCore.h).",
        py::arg("y"),
        py::arg("ar_orders"),
        py::arg("ma_orders"),
        py::arg("lags"),
        py::arg("ar_estimate"),
        py::arg("ma_estimate"),
        py::arg("arma_parameters"),
        py::arg("use_level"),
        py::arg("bounded")
    );
    m.def(
        "arima_parameter_bounds",
        &arima_parameter_bounds_wrapper,
        "Bounds of one AR / MA parameter within its factor (see arimaBounds.h).",
        py::arg("values"),
        py::arg("j"),
        py::arg("sign")
    );
}
