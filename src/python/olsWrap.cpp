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

py::tuple arima_hr_select_wrapper(const arma::vec& y, const arma::uvec& ar_orders,
                                  const arma::uvec& ma_orders, const arma::uvec& lags,
                                  arma::uword screen, arma::uword ar_max, arma::uword ma_max,
                                  bool bounded) {
    HRSelectResult result = arimaHRSelectCore(y, ar_orders, ma_orders, lags, screen, ar_max,
                                              ma_max, bounded);
    arma::mat orders = arma::conv_to<arma::mat>::from(result.orders);
    return py::make_tuple(carma::to_numpy(orders),
                          carma::to_numpy(result.parameters),
                          carma::to_numpy(result.innovations));
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
        "arima_hr_select",
        &arima_hr_select_wrapper,
        "Hannan-Rissanen screen of the ARMA orders of one level (see arimaInitCore.h): "
        "the orders, the parameters and the innovations of every candidate.",
        py::arg("y"),
        py::arg("ar_orders"),
        py::arg("ma_orders"),
        py::arg("lags"),
        py::arg("screen"),
        py::arg("ar_max"),
        py::arg("ma_max"),
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
