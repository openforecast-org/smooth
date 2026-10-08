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

// A vector as a NumPy array
static py::array_t<double> householder_vector(const arma::vec& v) {
    py::array_t<double> arr({static_cast<py::ssize_t>(v.n_elem)});
    auto buf = arr.mutable_unchecked<1>();
    for(size_t i = 0; i < v.n_elem; i++) {
        buf(i) = v(i);
    }
    return arr;
}

// The Householder QR of a fixed design (see olsCore.h): (qr, qraux, rDiag)
py::tuple householder_qr_wrapper(const arma::mat& X) {
    HouseholderQR d = householderQR(X);
    py::array_t<double> qr({static_cast<py::ssize_t>(d.qr.n_rows),
                            static_cast<py::ssize_t>(d.qr.n_cols)});
    auto buf = qr.mutable_unchecked<2>();
    for(size_t i = 0; i < d.qr.n_rows; i++) {
        for(size_t j = 0; j < d.qr.n_cols; j++) {
            buf(i, j) = d.qr(i, j);
        }
    }
    return py::make_tuple(qr, householder_vector(d.qraux), householder_vector(d.rDiag));
}

py::array_t<double> householder_coef_wrapper(const arma::mat& qr, const arma::vec& qraux,
                                             const arma::vec& r_diag, const arma::vec& y) {
    return householder_vector(householderCoef(HouseholderQR{qr, qraux, r_diag}, y));
}

py::array_t<double> householder_resid_wrapper(const arma::mat& qr, const arma::vec& qraux,
                                              const arma::vec& r_diag, const arma::vec& y) {
    return householder_vector(householderResid(HouseholderQR{qr, qraux, r_diag}, y));
}

py::array_t<double> na_fill_wrapper(const arma::vec& y, unsigned int lag_max) {
    return householder_vector(naFillCore(y, lag_max));
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
    m.def("householder_qr", &householder_qr_wrapper,
          "Householder QR of a fixed design, BLAS-free (see olsCore.h): (qr, qraux, r_diag).",
          py::arg("X"));
    m.def("householder_coef", &householder_coef_wrapper,
          "Least-squares coefficients from householder_qr, as R's qr.coef().",
          py::arg("qr"), py::arg("qraux"), py::arg("r_diag"), py::arg("y"));
    m.def("householder_resid", &householder_resid_wrapper,
          "Least-squares residuals from householder_qr, as R's qr.resid().",
          py::arg("qr"), py::arg("qraux"), py::arg("r_diag"), py::arg("y"));
    m.def("na_fill", &na_fill_wrapper,
          "The missing values of a series filled for the initialisation (see olsCore.h).",
          py::arg("y"), py::arg("lag_max"));
    m.def(
        "arima_parameter_bounds",
        &arima_parameter_bounds_wrapper,
        "Bounds of one AR / MA parameter within its factor (see arimaBounds.h).",
        py::arg("values"),
        py::arg("j"),
        py::arg("sign")
    );
}
