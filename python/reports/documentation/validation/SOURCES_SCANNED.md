# Sources Scanned

**Scan Date**: 2025-02-12  
**Purpose**: Anti-hallucination proof - lists all files actually scanned for documentation generation.

## R Package (R/)

| File | Lines | Scan Method |
|------|-------|-------------|
| adam.R | 9371 | wc -l |
| adamGeneral.R | 3382 | wc -l |
| adam-ces.R | 1204 | wc -l |
| adam-es.R | 413 | wc -l |
| adam-gum.R | 1006 | wc -l |
| adam-msarima.R | 321 | wc -l |
| adam-sma.R | 341 | wc -l |
| adam-ssarima.R | 1341 | wc -l |
| arimaCompact.R | 89 | wc -l |
| autoadam.R | 982 | wc -l |
| autoces.R | 189 | wc -l |
| autogum.R | 253 | wc -l |
| automsarima.R | 752 | wc -l |
| autossarima.R | 693 | wc -l |
| cma.R | 184 | wc -l |
| depricator.R | 9 | wc -l |
| globals.R | 169 | wc -l |
| helper.R | 321 | wc -l |
| isFunctions.R | 107 | wc -l |
| iss.R | 71 | wc -l |
| methods.R | 2532 | wc -l |
| msdecompose.R | 486 | wc -l |
| oes.R | 1291 | wc -l |
| oesg.R | 1092 | wc -l |
| reapply.R | 1401 | wc -l |
| rmultistep.R | 123 | wc -l |
| RcppExports.R | 31 | wc -l |
| simces.R | 471 | wc -l |
| simes.R | 615 | wc -l |
| simgum.R | 436 | wc -l |
| simoes.R | 152 | wc -l |
| simsma.R | 91 | wc -l |
| simssarima.R | 752 | wc -l |
| sm.R | 421 | wc -l |
| smooth-package.R | 86 | wc -l |
| smoothCombine.R | 348 | wc -l |
| sowhat.R | 38 | wc -l |
| sparma.R | 691 | wc -l |
| ssfunctions.R | 2816 | wc -l |
| variance-covariance.R | 136 | wc -l |
| zzz.R | 32 | wc -l |

**Grep pattern for R functions**: `^[a-zA-Z0-9_.]+ *<- *function`

## Python Package (python/src/smooth/)

| Path | Lines | Scan Method |
|------|-------|-------------|
| __init__.py | 6 | wc -l |
| adam_general/__init__.py | 7 | wc -l |
| adam_general/_adam_general.py | 219 | wc -l |
| adam_general/core/__init__.py | 4 | wc -l |
| adam_general/core/adam.py | 2349 | wc -l |
| adam_general/core/checker/__init__.py | 3 | wc -l |
| adam_general/core/checker/arima_checks.py | 272 | wc -l |
| adam_general/core/checker/data_checks.py | 282 | wc -l |
| adam_general/core/checker/model_checks.py | 658 | wc -l |
| adam_general/core/checker/organizers.py | 411 | wc -l |
| adam_general/core/checker/parameter_checks.py | 596 | wc -l |
| adam_general/core/checker/parameters_checker.py | 1047 | wc -l |
| adam_general/core/checker/sample_size.py | 435 | wc -l |
| adam_general/core/checker/_utils.py | 13 | wc -l |
| adam_general/core/creator/__init__.py | 12 | wc -l |
| adam_general/core/creator/architector.py | 515 | wc -l |
| adam_general/core/creator/creator.py | 703 | wc -l |
| adam_general/core/creator/filler.py | 455 | wc -l |
| adam_general/core/creator/initialization.py | 748 | wc -l |
| adam_general/core/creator/initialiser.py | 1631 | wc -l |
| adam_general/core/es.py | 266 | wc -l |
| adam_general/core/estimator/__init__.py | 7 | wc -l |
| adam_general/core/estimator/estimator.py | 780 | wc -l |
| adam_general/core/estimator/initial_values.py | 179 | wc -l |
| adam_general/core/estimator/optimization.py | 455 | wc -l |
| adam_general/core/estimator/selector.py | 1244 | wc -l |
| adam_general/core/estimator/two_stage.py | 329 | wc -l |
| adam_general/core/forecaster/__init__.py | 4 | wc -l |
| adam_general/core/forecaster/forecaster.py | 881 | wc -l |
| adam_general/core/forecaster/_helpers.py | 117 | wc -l |
| adam_general/core/forecaster/intervals.py | 720 | wc -l |
| adam_general/core/forecaster/preparator.py | 1139 | wc -l |
| adam_general/core/utils/__init__.py | 8 | wc -l |
| adam_general/core/utils/cost_functions.py | 1169 | wc -l |
| adam_general/core/utils/distributions.py | 305 | wc -l |
| adam_general/core/utils/ic.py | 160 | wc -l |
| adam_general/core/utils/likelihood.py | 0 | wc -l (empty stub) |
| adam_general/core/utils/n_param.py | 404 | wc -l |
| adam_general/core/utils/polynomials.py | 93 | wc -l |
| adam_general/core/utils/printing.py | 828 | wc -l |
| adam_general/core/utils/utils.py | 989 | wc -l |
| adam_general/core/utils/var_covar.py | 671 | wc -l |
| lowess.py | 132 | wc -l |

**Grep pattern for Python functions**: `^def [a-zA-Z_][a-zA-Z0-9_]*\(` (includes CF, AIC, AICc, BIC, BICc)

## C++ Sources (src/)

| Path | Purpose |
|------|---------|
| src/headers/adamCore.h | Shared ADAM core (fit, forecast, simulate) |
| src/headers/adamGeneral.h | ADAM general definitions |
| src/headers/ssGeneral.h | State-space general |
| src/headers/eigenCalc.h | Eigenvalue calculations |
| src/headers/lowess.h | LOWESS smoother |
| src/headers/matrixPowerCore.h | Matrix power |
| src/adamGeneral.cpp | Rcpp bindings for adamCore |
| src/ssGeneral.cpp | R: state-space ops |
| src/ssOccurrence.cpp | R: occurrence models |
| src/eigenCalc.cpp | R: eigenvalues |
| src/matrixPowerWrap.cpp | R: matrix power |
| src/python/adamPython.cpp | pybind11 bindings for adamCore |
| src/python/lowess.cpp | Python LOWESS |
| src/python/eigenCalc.cpp | Python eigenvalues |
| src/python/matrixPowerWrap.cpp | Python matrix power |
