#include "../headers/fpContract.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// Include armadillo and carma BEFORE the core implementation
#include <armadillo>
#include <carma>

#include <iostream>
#include <cmath>

// Include pure C++ implementation AFTER armadillo is available
#include "../headers/matrixPowerCore.h"

namespace py = pybind11;

// Wrapper for matrixPowerCore function
py::array_t<double> matrixPowerWrapper(arma::mat const &matrixA, int power) {
    arma::mat result = matrixPowerCore(matrixA, power);
    return carma::to_numpy(result);
}

PYBIND11_MODULE(_matrix_power, m) {
    m.doc() = "Matrix power functions";
    m.def(
        "matrix_power",
        &matrixPowerWrapper,
        "Computes the power of a matrix",
        py::arg("matrixA"),
        py::arg("power")
    );
}
