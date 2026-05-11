---
name: smooth-translation
description: Translates R features to Python using mapping docs and checklist. Use when translating R features to Python, mapping parameters, checking coverage, or following the translation workflow.
---

# R-to-Python Translation

Quick reference for translating smooth package features from R to Python.

## When to Read Each Report

| Need | Read |
|------|------|
| R function → Python equivalent lookup | [python/reports/documentation/translation/00_TRANSLATION_INDEX.md](../reports/documentation/translation/00_TRANSLATION_INDEX.md) |
| Model params, fitted attributes (model$persistence → persistence_level_, etc.) | [python/reports/documentation/translation/01_PARAMETER_MAPPING.md](../reports/documentation/translation/01_PARAMETER_MAPPING.md) |
| Data structures (matVt → mat_vt, R list → Python dict) | [python/reports/documentation/translation/02_DATA_STRUCTURE_MAPPING.md](../reports/documentation/translation/02_DATA_STRUCTURE_MAPPING.md) |
| Full / partial / not-translated status | [python/reports/documentation/translation/03_COVERAGE_MATRIX.md](../reports/documentation/translation/03_COVERAGE_MATRIX.md) |
| Step-by-step translation workflow | [python/reports/documentation/translation/04_TRANSLATION_CHECKLIST.md](../reports/documentation/translation/04_TRANSLATION_CHECKLIST.md) |

## Key Mappings (Quick Reference)

| R | Python |
|---|--------|
| model | model |
| orders (list: ar, i, ma) | ar_order, i_order, ma_order |
| xreg | X (in fit) |
| model$persistence | persistence_level_, persistence_trend_, persistence_seasonal_ |
| model$phi | phi_ |
| model$scale | scale_ |
| matVt | mat_vt (matrices_dict) |

## Translation Guidelines

1. **Always read the R implementation first** — Check R function registry and the actual R source.
2. **RNG differences** — R and Python use different random algorithms. For tests: generate data in R, save to CSV, read in Python.
3. **Follow the checklist** — Use [04_TRANSLATION_CHECKLIST.md](../reports/documentation/translation/04_TRANSLATION_CHECKLIST.md) for new features.
4. **Update docs** — After translating, update coverage matrix and translation index.
