# Python Package Overview

## Architecture

The Python smooth package is a translation of the R ADAM framework. It uses a modular, scikit-learn-style design with clear separation: checker, creator, estimator, forecaster, utils.

## Module Layout

```
python/src/smooth/
├── __init__.py              # ADAM, ES, msdecompose, lowess
├── lowess.py                # LOWESS smoother
└── adam_general/
    ├── _adam_general.py     # adam_fitter, adam_forecaster, adam_simulator (C++ wrappers)
    └── core/
        ├── adam.py          # ADAM class (main interface)
        ├── es.py            # ES class (ETS-only wrapper)
        ├── checker/         # parameters_checker and sub-checks
        ├── creator/         # architector, creator, filler, initialiser
        ├── estimator/       # estimator, selector, optimization
        ├── forecaster/      # forecaster, preparator, intervals
        └── utils/           # cost_functions, ic, var_covar, distributions
```

## Entry Points

- **ADAM**: Full model (ETS + ARIMA + xreg). `from smooth import ADAM`
- **ES**: ETS-only, subclasses ADAM with fixed ar_order=[0], i_order=[0], ma_order=[0]
- **msdecompose**: Multiple seasonal decomposition
- **lowess**: LOWESS smoothing

## C++ Integration

- **_adam_general.py**: Wraps `_adamCore` (pybind11) from adamPython.cpp
- **adam_fitter**, **adam_forecaster**: Call adamCore.fit, adamCore.forecast
- Built via CMakeLists.txt, uses carma for Armadillo-NumPy conversion

## Workflow

```
ADAM(model, lags, ...).fit(y, X)
  → parameters_checker
  → architector
  → creator
  → estimator (initialiser, CF cost function, adam_fitter)
  → store results

ADAM.predict(h, X)
  → preparator (filler if needed)
  → forecaster (adam_forecaster for point, intervals for PI)
```
