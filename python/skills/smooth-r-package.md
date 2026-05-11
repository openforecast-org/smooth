---
name: smooth-r-package
description: Navigates the R smooth package for function lookup, file location, and ADAM flow understanding. Use when working with R code, finding R functions, understanding R ADAM flow, or modifying R/adam.R and related R files.
---

# R Package Navigation

Quick reference for locating R smooth package code and understanding its flow.

## When to Read Each Report

| Need | Read |
|------|------|
| File locations, line counts, key exports | [python/reports/documentation/r_package/01_FILE_INDEX.md](../reports/documentation/r_package/01_FILE_INDEX.md) |
| Function → file lookup (every R function) | [python/reports/documentation/r_package/02_FUNCTION_REGISTRY.md](../reports/documentation/r_package/02_FUNCTION_REGISTRY.md) |
| Call graph: parametersChecker → architector → creator → filler → adamCpp | [python/reports/documentation/r_package/03_ADAM_FLOW.md](../reports/documentation/r_package/03_ADAM_FLOW.md) |
| Per-model specifics (es, ces, gum, ssarima, msarima, oes, sma) | [python/reports/documentation/r_package/04_MODEL_SPECIFICS/](../reports/documentation/r_package/04_MODEL_SPECIFICS/) |
| Package architecture overview | [python/reports/documentation/r_package/00_OVERVIEW.md](../reports/documentation/r_package/00_OVERVIEW.md) |

## Key Entry Points

- **adam**: R/adam.R line 326 (main entry)
- **es**: R/adam-es.R line 224 (wraps adam with ETS defaults)
- **gum**: R/adam-gum.R line 98
- **ces**: R/adam-ces.R line 91
- **ssarima**: R/adam-ssarima.R line 110
- **msarima**: R/adam-msarima.R line 192
- **sma**: R/adam-sma.R line 101
- **oes**: R/oes.R line 107
- **msdecompose**: R/msdecompose.R line 54

## Core Internal Flow (adam.R)

1. `parametersChecker()` (adamGeneral.R) — validates inputs
2. `architector()` (local, ~line 656) — components, lags, profiles
3. `creator()` (local, ~line 750) — matVt, matF, vecg, matw
4. `initialiser()` (local, ~line 1402) — B, Bl, Bu
5. Optimization: `filler()` → adamCpp$fit()
6. Forecast: `filler()` → forecasterwrap (Rcpp)
