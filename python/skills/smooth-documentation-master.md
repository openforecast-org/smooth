---
name: smooth-documentation-master
description: Master index for smooth package documentation. Use when new to the codebase, need orientation, or ask where the documentation is or how to navigate smooth.
---

# Smooth Documentation Master Index

Entry point for navigating smooth package documentation. All reports live in [python/reports/documentation/](../reports/documentation/).

## Documentation Structure

| Folder | Contents |
|--------|----------|
| **r_package/** | R package: file index, function registry, ADAM flow, per-model docs |
| **python_package/** | Python package: file index, function registry, ADAM flow, per-module docs |
| **cpp_shared/** | Shared C++: architecture, headers/sources, API reference |
| **translation/** | R-to-Python mapping: function index, parameters, data structures, coverage |
| **validation/** | Scan proof: SOURCES_SCANNED.md, CHANGELOG.md |

## Which Skill to Use

| Task | Skill | Location |
|------|-------|----------|
| R code, R functions, R flow | smooth-r-package | [smooth-r-package.md](smooth-r-package.md) |
| Python code, Python functions, Python flow | smooth-python-package | [smooth-python-package.md](smooth-python-package.md) |
| Translate R → Python, mapping, coverage | smooth-translation | [smooth-translation.md](smooth-translation.md) |
| C++ code, adamCore, build issues | smooth-cpp-shared | [smooth-cpp-shared.md](smooth-cpp-shared.md) |

## Quick Lookup Paths

1. **Find where a function lives**: `r_package/02_FUNCTION_REGISTRY.md` or `python_package/02_FUNCTION_REGISTRY.md`
2. **Understand flow**: `r_package/03_ADAM_FLOW.md` or `python_package/03_ADAM_FLOW.md`
3. **Translate R → Python**: `translation/00_TRANSLATION_INDEX.md` and `translation/01_PARAMETER_MAPPING.md`
4. **Check coverage**: `translation/03_COVERAGE_MATRIX.md`
5. **Translate a new feature**: `translation/04_TRANSLATION_CHECKLIST.md`

## Main README

[python/reports/documentation/README.md](../reports/documentation/README.md) — Full documentation overview and usage for agents.
