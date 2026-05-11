---
name: smooth-cpp-shared
description: Understands shared C++ code and how R vs Python use it. Use when modifying C++ code, understanding adamCore, debugging C++ extensions, or addressing build/compilation issues.
---

# C++ Shared Code Navigation

Quick reference for the shared C++ layer used by both R and Python.

## When to Read Each Report

| Need | Read |
|------|------|
| R vs Python usage of adamCore, forecasterwrap, bindings | [python/reports/documentation/cpp_shared/00_OVERVIEW.md](../reports/documentation/cpp_shared/00_OVERVIEW.md) |
| Headers and source files | [python/reports/documentation/cpp_shared/01_HEADERS_AND_SOURCES.md](../reports/documentation/cpp_shared/01_HEADERS_AND_SOURCES.md) |
| API reference | [python/reports/documentation/cpp_shared/02_API_REFERENCE.md](../reports/documentation/cpp_shared/02_API_REFERENCE.md) |

## R vs Python at a Glance

| Component | R | Python |
|-----------|---|--------|
| adamCore (fit, forecast, simulate) | adamGeneral.cpp (Rcpp) | adamPython.cpp (pybind11) |
| forecasterwrap | RcppExports, ssGeneral | N/A (Python uses adamCore.forecast) |
| matrixPowerWrap | RcppExports | matrixPowerWrap.cpp (pybind11) |
| eigenCalc | eigenCalc.cpp (Rcpp) | eigenCalc.cpp (pybind11) |
| ssGeneral, ssOccurrence | R only | N/A |

## Build

- **R**: R CMD build, Rcpp, Makevars
- **Python**: CMake + scikit-build-core, pybind11, carma for Armadillo-NumPy

## Shared Header

Both use `src/headers/adamCore.h` for the core fitting and forecasting logic. Algorithm code is identical; bindings differ (Rcpp vs pybind11).
