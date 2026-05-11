# C++ Headers and Sources

## Headers (src/headers/)

| Header | Purpose | Used By |
|--------|---------|---------|
| adamCore.h | adamCore class, FitResult, ForecastResult, etc. | adamGeneral.cpp, adamPython.cpp |
| adamGeneral.h | ADAM general definitions | adamCore.h |
| ssGeneral.h | State-space general | adamCore.h, ssGeneral.cpp, ssOccurrence.cpp |
| eigenCalc.h | Eigenvalue calculations | eigenCalc.cpp (R and Python) |
| lowess.h | LOWESS smoother | lowess.cpp (Python) |
| matrixPowerCore.h | Matrix power | matrixPowerWrap.cpp (R and Python) |

## R Sources (src/)

| Source | Bindings | Purpose |
|--------|----------|---------|
| adamGeneral.cpp | Rcpp | adamCore class, polynomialise, fit, forecast, simulate, reapply, reforecast |
| ssGeneral.cpp | Rcpp | forecasterwrap, state-space ops |
| ssOccurrence.cpp | Rcpp | occurenceFitterWrap, occurrenceOptimizerWrap |
| eigenCalc.cpp | Rcpp | smoothEigensR |
| matrixPowerWrap.cpp | Rcpp | matrixPowerWrap |
| RcppExports.cpp | Generated | Exports all Rcpp functions |

## Python Sources (src/python/)

| Source | Module | Purpose |
|--------|--------|---------|
| adamPython.cpp | _adamCore | adamCore class (pybind11) |
| lowess.cpp | _lowess | LOWESS (pybind11) |
| eigenCalc.cpp | _eigenCalc | Eigenvalues (pybind11) |
| matrixPowerWrap.cpp | (if built) | Matrix power |

Note: Python CMakeLists.txt defines _adamCore, _lowess, _eigenCalc. matrixPowerWrap may be included in _adamCore or separate.

## Include Chain

adamCore.h → ssGeneral.h, adamGeneral.h
adamPython.cpp → adamCore.h
adamGeneral.cpp → adamCore.h
