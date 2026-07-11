#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <armadillo>

#ifndef PYTHON_BUILD
#define PYTHON_BUILD 1  // Define this before including headers to enable Python-specific code paths
#endif
#include <carma>

#include "../headers/adamCore.h"

namespace py = pybind11;

PYBIND11_MODULE(_adamCore, m) {
    m.doc() = "Python bindings for adamCore class";

    // Bind PolyResult struct
    py::class_<PolyResult>(m, "PolyResult")
        .def_readonly("arPolynomial", &PolyResult::arPolynomial)
        .def_readonly("iPolynomial", &PolyResult::iPolynomial)
        .def_readonly("ariPolynomial", &PolyResult::ariPolynomial)
        .def_readonly("maPolynomial", &PolyResult::maPolynomial);

    // Bind FitResult struct
    py::class_<FitResult>(m, "FitResult")
        .def_readonly("states", &FitResult::states)
        .def_readonly("fitted", &FitResult::fitted)
        .def_readonly("errors", &FitResult::errors)
        .def_readonly("profile", &FitResult::profile);

    // Bind OmFitGeneralResult struct
    py::class_<OmFitGeneralResult>(m, "OmFitGeneralResult")
        .def_readonly("statesA",  &OmFitGeneralResult::statesA)
        .def_readonly("fittedA",  &OmFitGeneralResult::fittedA)
        .def_readonly("errorsA",  &OmFitGeneralResult::errorsA)
        .def_readonly("profileA", &OmFitGeneralResult::profileA)
        .def_readonly("statesB",  &OmFitGeneralResult::statesB)
        .def_readonly("fittedB",  &OmFitGeneralResult::fittedB)
        .def_readonly("errorsB",  &OmFitGeneralResult::errorsB)
        .def_readonly("profileB", &OmFitGeneralResult::profileB);

    // Bind ForecastResult struct
    py::class_<ForecastResult>(m, "ForecastResult")
        .def_readonly("forecast", &ForecastResult::forecast);

    // Bind ErrorResult struct
    py::class_<ErrorResult>(m, "ErrorResult")
        .def_readonly("errors", &ErrorResult::errors);

    // Bind SimulateResult struct
    py::class_<SimulateResult>(m, "SimulateResult")
        .def_readonly("states", &SimulateResult::states)
        .def_readonly("data", &SimulateResult::data);

    // Bind ReapplyResult struct
    py::class_<ReapplyResult>(m, "ReapplyResult")
        .def_readonly("states", &ReapplyResult::states)
        .def_readonly("fitted", &ReapplyResult::fitted)
        .def_readonly("profile", &ReapplyResult::profile);

    // Bind ReforecastResult struct
    py::class_<ReforecastResult>(m, "ReforecastResult")
        .def_readonly("data", &ReforecastResult::data);

    // Bind adamCore class
    py::class_<adamCore>(m, "adamCore")
        .def(py::init<arma::uvec, char, char, char, unsigned int, unsigned int,
            unsigned int, unsigned int, unsigned int, unsigned int, bool, bool>(),
            py::arg("lags"),
            py::arg("E"),
            py::arg("T"),
            py::arg("S"),
            py::arg("nNonSeasonal"),
            py::arg("nSeasonal"),
            py::arg("nETS"),
            py::arg("nArima"),
            py::arg("nXreg"),
            py::arg("nComponents"),
            py::arg("constant"),
            py::arg("adamETS"))
        .def("polynomialise", &adamCore::polynomialise,
            py::arg("B"),
            py::arg("arOrders"),
            py::arg("iOrders"),
            py::arg("maOrders"),
            py::arg("arEstimate"),
            py::arg("maEstimate"),
            py::arg("armaParameters"),
            py::arg("lagsARIMA"))
        .def("fit", &adamCore::fit,
            py::arg("matrixVt"),
            py::arg("matrixWt"),
            py::arg("matrixF"),
            py::arg("vectorG"),
            py::arg("indexLookupTable"),
            py::arg("profilesRecent"),
            py::arg("vectorYt"),
            py::arg("vectorOt"),
            py::arg("backcast"),
            py::arg("nIterations"),
            py::arg("O") = 'n')
        .def("omfitGeneral", &adamCore::omfitGeneral,
            py::arg("matrixVtA"),
            py::arg("matrixWtA"),
            py::arg("matrixFA"),
            py::arg("vectorGA"),
            py::arg("indexLookupTableA"),
            py::arg("profilesRecentA"),
            py::arg("EB"),
            py::arg("TB"),
            py::arg("SB"),
            py::arg("nNonSeasonalB"),
            py::arg("nSeasonalB"),
            py::arg("nETSB"),
            py::arg("nArimaB"),
            py::arg("nXregB"),
            py::arg("nComponentsB"),
            py::arg("constantB"),
            py::arg("adamETSB"),
            py::arg("matrixVtB"),
            py::arg("matrixWtB"),
            py::arg("matrixFB"),
            py::arg("vectorGB"),
            py::arg("indexLookupTableB"),
            py::arg("profilesRecentB"),
            py::arg("vectorOt"),
            py::arg("backcast"),
            py::arg("nIterations"))
        .def("forecast", &adamCore::forecast,
            py::arg("matrixWt"),
            py::arg("matrixF"),
            py::arg("indexLookupTable"),
            py::arg("profilesRecent"),
            py::arg("horizon"))
        .def("ferrors", &adamCore::ferrors,
            py::arg("matrixVt"),
            py::arg("matrixWt"),
            py::arg("matrixF"),
            py::arg("indexLookupTable"),
            py::arg("profilesRecent"),
            py::arg("horizon"),
            py::arg("vectorYt"))
        .def("simulate", &adamCore::simulate,
            py::arg("matrixErrors"),
            py::arg("matrixOt"),
            py::arg("arrayVt"),
            py::arg("matrixWt"),
            py::arg("arrayF"),
            py::arg("matrixG"),
            py::arg("indexLookupTable"),
            py::arg("profilesRecent"),
            py::arg("E"),
            py::arg("refineHead"))
        .def("reapply", &adamCore::reapply,
            py::arg("matrixYt"),
            py::arg("matrixOt"),
            py::arg("arrayVt"),
            py::arg("arrayWt"),
            py::arg("arrayF"),
            py::arg("matrixG"),
            py::arg("indexLookupTable"),
            py::arg("arrayProfilesRecent"),
            py::arg("backcast"))
        .def("reforecast", &adamCore::reforecast,
            py::arg("arrayErrors"),
            py::arg("arrayOt"),
            py::arg("arrayWt"),
            py::arg("arrayF"),
            py::arg("matrixG"),
            py::arg("indexLookupTable"),
            py::arg("arrayProfileRecent"),
            py::arg("E"));
}
