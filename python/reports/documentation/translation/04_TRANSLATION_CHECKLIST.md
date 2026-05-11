# Translation Checklist

Step-by-step template for translating a new R feature to Python.

## 1. Locate R Implementation

- [ ] Identify R function(s) in [r_package/02_FUNCTION_REGISTRY.md](r_package/02_FUNCTION_REGISTRY.md)
- [ ] Read R file(s) and note:
  - Parameters and defaults
  - Call flow (what it calls internally)
  - Data structures (lists, matrices)
  - C++ usage if any

## 2. Check Python Equivalent

- [ ] Search [python_package/02_FUNCTION_REGISTRY.md](python_package/02_FUNCTION_REGISTRY.md)
- [ ] Check [translation/00_TRANSLATION_INDEX.md](translation/00_TRANSLATION_INDEX.md)
- [ ] If gap: note where it should slot in (checker, creator, estimator, forecaster, utils)
- [ ] If partial: identify what’s missing

## 3. Parameter Alignment

- [ ] Map R parameters to Python using [translation/01_PARAMETER_MAPPING.md](translation/01_PARAMETER_MAPPING.md)
- [ ] Add new mappings if needed
- [ ] Verify names and types (R list vs Python dict, etc.)

## 4. Data Structure Mapping

- [ ] Map R structures using [translation/02_DATA_STRUCTURE_MAPPING.md](translation/02_DATA_STRUCTURE_MAPPING.md)
- [ ] Note any new structures
- [ ] Ensure matrix order (Fortran/column-major for C++)

## 5. Implementation

- [ ] Add/modify Python code in appropriate module
- [ ] Mirror R logic; preserve algorithm
- [ ] Use existing patterns (e.g. filler, parameters_checker)

## 6. C++ Changes (if any)

- [ ] Check [cpp_shared/](cpp_shared/) for shared C++ usage
- [ ] If R uses Rcpp and Python doesn’t: add Python binding or reimplement in Python
- [ ] If both use adamCore: confirm API match

## 7. Testing

- [ ] Generate identical data (R: write CSV; Python: read CSV) to avoid RNG differences
- [ ] Compare fitted parameters, forecasts, residuals
- [ ] Add test to python/tests/

## 8. Update Documentation

- [ ] Update [translation/03_COVERAGE_MATRIX.md](translation/03_COVERAGE_MATRIX.md)
- [ ] Update [translation/00_TRANSLATION_INDEX.md](translation/00_TRANSLATION_INDEX.md) if new mappings
- [ ] Update [validation/CHANGELOG.md](validation/CHANGELOG.md) if regenerating reports
