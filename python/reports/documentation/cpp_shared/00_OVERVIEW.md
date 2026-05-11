# C++ Shared Code Overview

## R vs Python Usage

| Component | R | Python |
|-----------|---|--------|
| adamCore (fit, forecast, simulate) | adamGeneral.cpp (Rcpp) | adamPython.cpp (pybind11) |
| forecasterwrap | RcppExports, ssGeneral | N/A (Python uses adamCore.forecast) |
| matrixPowerWrap | RcppExports | matrixPowerWrap.cpp (pybind11) |
| eigenCalc | eigenCalc.cpp (Rcpp) | eigenCalc.cpp (pybind11) |
| lowess | stats::lowess (R) | lowess.cpp (pybind11) |
| ssGeneral | ssGeneral.cpp (R only) | N/A |
| ssOccurrence | ssOccurrence.cpp (R only) | N/A |

## Shared Header: adamCore.h

Both R and Python use `src/headers/adamCore.h` which defines:
- adamCore class
- FitResult, ForecastResult, SimulateResult, etc.

adamCore.h includes ssGeneral.h and adamGeneral.h. The core fitting and forecasting logic is identical.

## Build

- **R**: Uses R CMD build, Rcpp, Makevars. Compiles adamGeneral.cpp, ssGeneral.cpp, ssOccurrence.cpp, eigenCalc.cpp, matrixPowerWrap.cpp.
- **Python**: Uses CMake + scikit-build-core. Builds _adamCore from adamPython.cpp, _lowess, _eigenCalc. Uses carma for Armadillo-NumPy conversion.

## Python-Only Copies

Python has its own copies in src/python/:
- adamPython.cpp
- lowess.cpp
- eigenCalc.cpp
- matrixPowerWrap.cpp

R has these in src/ (Rcpp). The C++ algorithm code (headers) is shared; the bindings differ.
