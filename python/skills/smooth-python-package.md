---
name: smooth-python-package
description: Navigates the Python smooth package for function lookup, file location, and ADAM flow understanding. Use when working with Python code, finding Python functions, understanding Python ADAM flow, or modifying python/src/smooth/ and adam_general.
---

# Python Package Navigation

Quick reference for locating Python smooth package code and understanding its flow.

## When to Read Each Report

| Need | Read |
|------|------|
| Module locations, line counts, key exports | [python/reports/documentation/python_package/01_FILE_INDEX.md](../reports/documentation/python_package/01_FILE_INDEX.md) |
| Function → module lookup (every Python function) | [python/reports/documentation/python_package/02_FUNCTION_REGISTRY.md](../reports/documentation/python_package/02_FUNCTION_REGISTRY.md) |
| Fit/predict call graph | [python/reports/documentation/python_package/03_ADAM_FLOW.md](../reports/documentation/python_package/03_ADAM_FLOW.md) |
| Per-module specifics (checker, creator, estimator, forecaster, utils) | [python/reports/documentation/python_package/04_MODULE_SPECIFICS/](../reports/documentation/python_package/04_MODULE_SPECIFICS/) |
| Package architecture overview | [python/reports/documentation/python_package/00_OVERVIEW.md](../reports/documentation/python_package/00_OVERVIEW.md) |

## Entry Points

- **ADAM**: `smooth.adam_general.core.adam.ADAM` (full model: ETS + ARIMA + xreg)
- **ES**: `smooth.adam_general.core.es.ES` (ETS-only wrapper)
- **msdecompose**: `smooth.adam_general.core.utils.utils.msdecompose`
- **lowess**: `smooth.lowess.lowess`

## Core Modules

- **checker**: parameters_checker, model_checks, arima_checks, data_checks, etc.
- **creator**: architector, creator, filler, initialiser, initialization
- **estimator**: estimator, selector, optimization, two_stage
- **forecaster**: forecaster, preparator, intervals
- **utils**: cost_functions, ic, var_covar, polynomials, distributions

## Fit Flow (ADAM.fit)

```
parameters_checker → architector → creator → estimator
  → initialiser → CF (filler + adam_fitter) → _run_optimization
```

## Predict Flow (ADAM.predict)

```
preparator (filler if needed) → forecaster → adam_forecaster
```
