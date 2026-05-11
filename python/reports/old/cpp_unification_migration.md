# C++ Core Unification Migration Report

**Date**: 2025-12-18
**Status**: ✅ Migration Complete - Testing Phase
**Objective**: Migrate Python implementation to use unified `adamCore` C++ class shared with R

---

## Executive Summary

Successfully migrated the Python smooth package to use the unified `adamCore` C++ class that your R colleague created. Both R and Python now share the same C++ backend at `/src/headers/adamCore.h`, eliminating code duplication and ensuring consistency.

**Current Status**:
- ✅ C++ module compiles and loads successfully
- ✅ Python code runs without errors
- ⚠️ Results differ from R - requires investigation

---

## Background

### Previous Architecture
- **R**: Used standalone C++ functions exported via `RcppExports.cpp`
- **Python**: Used separate implementation in `/src/python_examples/adamGeneral.cpp`
- **Problem**: Two codebases to maintain, potential for divergence

### New Architecture
- **Unified C++ Core**: Single `adamCore` class in `/src/headers/adamCore.h`
- **R Bindings**: Uses Rcpp modules (`RCPP_MODULE` in `/src/adamGeneral.cpp`)
- **Python Bindings**: Uses pybind11 (`/src/python/adamPython.cpp`)
- **Benefits**: Single source of truth, guaranteed identical algorithms

---

## Changes Made

### 1. C++ Header Modifications

#### File: `/src/headers/adamCore.h`

**Problem**: Header contained R-specific types (`SEXP`, `Rf_isNull`, `as<>`)

**Solution**: Added conditional compilation for cross-platform compatibility

```cpp
#ifdef PYTHON_BUILD
    PolyResult polynomialise(arma::vec const &B,
                              arma::uvec const &arOrders, ...,
                              arma::vec armaParameters,  // Direct vector in Python
                              arma::uvec const &lagsARIMA){
        arma::vec armaParametersValue = armaParameters;
#else
    PolyResult polynomialise(arma::vec const &B,
                              arma::uvec const &arOrders, ...,
                              SEXP armaParameters,  // R type
                              arma::uvec const &lagsARIMA){
        arma::vec armaParametersValue;
        if(!Rf_isNull(armaParameters)){
            armaParametersValue = as<arma::vec>(armaParameters);
        }
#endif
```

**Location**: Lines 83-102

---

#### File: `/src/headers/ssGeneral.h`

**Problem**: Used R-specific constant `R_PosInf` for infinity

**Solution**: Added platform-specific infinity handling

```cpp
else if((yact!=0) & (yfit==0)){
#ifdef PYTHON_BUILD
    return std::numeric_limits<double>::infinity();
#else
    return R_PosInf;
#endif
}
```

**Location**: Lines 24-29

---

### 2. Python C++ Bindings

#### File: `/src/python/adamPython.cpp`

**Changes**:
1. **Added `#define PYTHON_BUILD`** before including headers (line 7)
2. **Fixed method signatures** to match actual C++ class:
   - `fit()`: 11 parameters (was 8)
   - `forecast()`: 5 parameters (was 5 but wrong order)
   - `ferrors()`: 7 parameters (was 6)
   - `simulate()`: 9 parameters (was 10 but wrong order)
   - `reapply()`: 10 parameters (was 10 but wrong order)
   - `polynomialise()`: Uses `arma::vec` instead of `SEXP`

**Key Insight**: The C++ class methods have different signatures than the old procedural functions

---

#### File: `/python/CMakeLists.txt`

**Changes**:
- Added `target_compile_definitions(_adamCore PRIVATE PYTHON_BUILD)` (line 48)
- Ensures `PYTHON_BUILD` macro is defined during compilation

---

### 3. Python Wrapper Layer

#### File: `/python/smooth/adam_general/_adam_general.py` (NEW)

**Purpose**: Backward-compatible wrappers maintaining old API while using new `adamCore` class

**Key Functions**:

```python
def adam_fitter(...) -> dict:
    """Wraps adamCore.fit() method"""
    # Instantiate adamCore
    adam_core = _adamCore.adamCore(lags, E, T, S, ...)

    # Call fit method
    result = adam_core.fit(matrixVt, matrixWt, ...)

    # Convert to old dict format for backward compatibility
    return {
        'matVt': result.states,      # New field name
        'yFitted': result.fitted,    # New field name
        'errors': result.errors,
        'profile': result.profile
    }
```

**Similar wrappers for**:
- `adam_forecaster()` → wraps `adamCore.forecast()`
- `adam_simulator()` → wraps `adamCore.simulate()`

**Type Conversions Applied**:
- Lags: `np.uint64` (arma::uvec is 64-bit)
- E/T/S: Pass as strings `'A'`, `'M'`, `'N'` (not `ord()` integers)
- All numeric parameters: Explicit casting to `int()`, `bool()`

---

#### File: `/python/smooth/adam_general/core/utils/polynomials.py`

**Before**: Stub function with `pass`

**After**: Full implementation wrapping `adamCore.polynomialise()`

```python
def adam_polynomialiser(parameters, ar_orders, i_orders, ma_orders,
                       ar_estimate, ma_estimate, arma_parameters, lags):
    # Create minimal adamCore instance
    adam_core = _adamCore.adamCore(
        lags=lags_arima,
        E='A', T='N', S='N',  # Minimal settings
        nNonSeasonal=0, nSeasonal=0,
        nETS=0, nArima=int(n_arima),
        nXreg=0, constant=False, adamETS=False
    )

    # Call polynomialise method
    result = adam_core.polynomialise(
        parameters, ar_orders, i_orders, ma_orders,
        ar_estimate, ma_estimate, arma_parameters, lags_arima
    )

    # Return as dict
    return {
        'ar_polynomial': np.array(result.arPolynomial),
        'i_polynomial': np.array(result.iPolynomial),
        'ari_polynomial': np.array(result.ariPolynomial),
        'ma_polynomial': np.array(result.maPolynomial)
    }
```

---

### 4. Architecture Updates

#### File: `/python/smooth/adam_general/core/creator.py`

**Function**: `architector()` (lines 2679-2705)

**Added**: `adamCore` instantiation (similar to R's approach at R/adam.R:752)

```python
# Create C++ adamCore object (like R does in architector)
try:
    from smooth import _adamCore

    # Convert lags_model_all to numpy array
    lags_array = np.asarray(lags_dict["lags_model_all"], dtype=np.uint64).ravel()

    adam_cpp = _adamCore.adamCore(
        lags=lags_array,
        E=model_type_dict["error_type"],           # 'A' or 'M'
        T=model_type_dict["trend_type"][0],        # 'A', 'M', or 'N'
        S=model_type_dict["season_type"][0],       # 'A', 'M', or 'N'
        nNonSeasonal=int(components_dict["components_number_ets_non_seasonal"]),
        nSeasonal=int(components_dict["components_number_ets_seasonal"]),
        nETS=int(components_dict["components_number_ets"]),
        nArima=int(components_dict["components_number_arima"]),
        nXreg=int(explanatory_checked.get("xreg_number", 0) if explanatory_checked else 0),
        constant=bool(constants_checked.get("constant_required", False) if constants_checked else False),
        adamETS=bool(model_type_dict.get("adam_ets", False))
    )

    # Store in components_dict for use by other functions
    components_dict["adam_cpp"] = adam_cpp
except ImportError:
    components_dict["adam_cpp"] = None
```

**Why**:
- R creates the `adamCpp` object once in architector
- Reuses it across `fit()`, `forecast()`, `polynomialise()` calls
- More efficient than recreating for each operation
- Python now follows the same pattern

---

## Key Technical Decisions

### 1. Type Mapping: Python ↔ C++

| Python Type | C++ Type | Notes |
|------------|----------|-------|
| `str` ('A', 'M', 'N') | `char` | Pass single-character strings |
| `np.uint64` array | `arma::uvec` | Use uint64, not uint32 |
| `np.float64` array | `arma::vec` | Double precision |
| `int` | `unsigned int` | Explicit casting |
| `bool` | `bool` | Explicit casting |

**Critical Fix**: Initially tried `ord('A')` → 65, but C++ binding expects string `'A'`

---

### 2. Return Value Mapping

| C++ Struct | Python Wrapper Returns |
|-----------|----------------------|
| `FitResult{states, fitted, errors, profile}` | `dict{'matVt': states, 'yFitted': fitted, ...}` |
| `ForecastResult{forecast}` | `np.ndarray` (directly return `.forecast`) |
| `PolyResult{arPolynomial, ...}` | `dict{'ar_polynomial': ..., ...}` |

**Why dict format**: Maintains backward compatibility with existing Python code

---

### 3. Conditional Compilation Strategy

Used `#ifdef PYTHON_BUILD` to handle platform differences:

**When to use**:
- R-specific types (`SEXP`, `Rcpp::List`)
- R-specific functions (`Rf_isNull`, `as<>()`)
- Platform-specific constants (`R_PosInf` vs `std::numeric_limits<double>::infinity()`)

**Where defined**:
- CMakeLists.txt: `target_compile_definitions(_adamCore PRIVATE PYTHON_BUILD)`
- adamPython.cpp: `#define PYTHON_BUILD` before includes

---

## Testing Results

### ✅ Successful

```bash
# Module loads correctly
$ python -c "from smooth import _adamCore; print(dir(_adamCore.adamCore))"
['ferrors', 'fit', 'forecast', 'polynomialise', 'reapply', 'reforecast', 'simulate']

# Python code runs without errors
model = ADAM(model='ANN', lags=[12], initial='optimal')
model.fit(ts_df)  # ✅ Runs successfully
```

### ⚠️ Results Differ from R

**Test**: `python/smooth/adam_general/tests/two_stage/test_two_stage_python.ipynb`

**Observation**: Model fits and produces forecasts, but parameter values differ from R

**Potential Causes** (to investigate):
1. **refineHead logic**: Recently changed to be conditional on trend (lines 62-73 in adamCore.h)
   - Python may have different default or logic

2. **adamETS flag**: New parameter controls ETS formulation
   - Check if Python and R use same default

3. **Initialization differences**:
   - Two-stage initialization may have different starting values
   - Check `initial='optimal'` implementation

4. **Floating point precision**:
   - Check if matrix operations use same order
   - Verify BLAS/LAPACK linking is consistent

5. **Profile/lookup table indexing**:
   - Complex indexing logic for time-varying parameters
   - Verify `index_lookup_table` matches R

---

## Files Modified Summary

### C++ Core (Shared R/Python)
- ✏️ `/src/headers/adamCore.h` - Added `#ifdef PYTHON_BUILD` conditional
- ✏️ `/src/headers/ssGeneral.h` - Platform-specific infinity handling
- ✏️ `/src/python/adamPython.cpp` - Fixed method bindings, added `#define PYTHON_BUILD`

### Build System
- ✏️ `/python/CMakeLists.txt` - Added `PYTHON_BUILD` compilation flag

### Python Implementation
- ✨ `/python/smooth/adam_general/_adam_general.py` - **NEW** Wrapper module
- ✏️ `/python/smooth/adam_general/core/utils/polynomials.py` - Implemented `adam_polynomialiser()`
- ✏️ `/python/smooth/adam_general/core/creator.py` - Added `adamCore` instantiation in `architector()`

Legend: ✨ New file, ✏️ Modified file

---

## Next Steps for Result Alignment

### 1. Compare C++ Method Calls

**R side** (from `R/adam.R`):
```r
adamFitted <- adamCpp$fit(matVt, matWt, matF, vecG,
                          indexLookupTable, profilesRecent,
                          vectorYt, vectorOt, backcast, nIterations, refineHead)
```

**Python side** (from `_adam_general.py:133`):
```python
result = adam_core.fit(matrixVt, matrixWt, matrixF, vectorG,
                       indexLookupTable, profilesRecent,
                       vectorYt, vectorOt, backcast, nIterations, refineHead)
```

**Action**: Verify parameter values are identical between R and Python calls

---

### 2. Check refineHead Logic

**C++ implementation** (adamCore.h:223):
```cpp
if(refineHead && (T!='N')){
    // Only refine head for models with trend
}
```

**Python** (cost_functions.py:414-416):
```python
# refineHead should always be True (fixed backcasting issue)
refine_head = True
```

**Concern**: Comment suggests Python always uses `refine_head=True`, but C++ only applies it when `T != 'N'`

**Action**:
1. Check if Python passes correct trend type to C++
2. Verify R and Python use same `refineHead` value
3. Test model without trend (`ANN`) vs with trend (`AAN`)

---

### 3. Compare Initialization Values

**Two-stage initialization**:
- Stage 1: Optimize initials only
- Stage 2: Optimize all parameters with initials from stage 1

**Action**: Log and compare intermediate values:
```python
# After stage 1
print("Stage 1 initials:", adam_created['mat_vt'][:, :lags_max])

# After stage 2
print("Final parameters:", adam_estimated['B'])
```

Compare with R:
```r
# After stage 1
print(adamCreated$matVt[, 1:lagsModelMax])

# After stage 2
print(adamEstimated$B)
```

---

### 4. Verify Matrix Memory Layout

**Potential issue**: C++ expects Fortran-contiguous (column-major) matrices, Python uses C-contiguous by default

**Check** (forecaster.py:1699):
```python
# Current comment warns about this
# DO NOT change to C-order as it will cause incorrect results in adam_fitter
```

**Action**: Verify all matrices passed to C++ are F-contiguous:
```python
assert matrixVt.flags['F_CONTIGUOUS'], "matrixVt must be F-contiguous"
assert matrixWt.flags['F_CONTIGUOUS'], "matrixWt must be F-contiguous"
```

---

### 5. Debug Print Values

**Add debug output to C++ temporarily**:

```cpp
// In adamCore::fit() at line 204
std::cout << "C++ fit called with:" << std::endl;
std::cout << "  E=" << E << " T=" << T << " S=" << S << std::endl;
std::cout << "  obs=" << obs << " nComponents=" << nComponents << std::endl;
std::cout << "  refineHead=" << refineHead << " backcast=" << backcast << std::endl;
std::cout << "  vectorYt[0:5]=" << vectorYt.rows(0,4).t() << std::endl;
std::cout << "  vectorG[0:5]=" << vectorG.rows(0,std::min(4, (int)vectorG.n_rows-1)).t() << std::endl;
```

**Rebuild** and compare R vs Python outputs

---

### 6. Unit Test Individual Methods

Create isolated tests for each C++ method:

```python
# Test polynomialise
from smooth import _adamCore
adam_core = _adamCore.adamCore(lags=np.array([1], dtype=np.uint64), ...)
result = adam_core.polynomialise(
    np.array([0.5, -0.3]),  # parameters
    np.array([1, 0], dtype=np.uint64),  # arOrders
    np.array([1, 0], dtype=np.uint64),  # iOrders
    np.array([1, 0], dtype=np.uint64),  # maOrders
    True, True,  # arEstimate, maEstimate
    np.array([]),  # armaParameters
    np.array([1, 12], dtype=np.uint64)  # lagsARIMA
)
print("AR polynomial:", result.arPolynomial)
print("MA polynomial:", result.maPolynomial)
```

Compare with R:
```r
adamCpp$polynomialise(c(0.5, -0.3), c(1,0), c(1,0), c(1,0),
                      TRUE, TRUE, NULL, c(1,12))
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     User Code (R / Python)                   │
└────────────────┬──────────────────────┬─────────────────────┘
                 │                      │
                 │ R                    │ Python
                 ▼                      ▼
┌────────────────────────┐  ┌──────────────────────────────┐
│   R/adam.R             │  │  python/adam_general/        │
│   - adam()             │  │  - ADAM class (adam.py)      │
│   - estimator()        │  │  - estimator()               │
│   - creator()          │  │  - creator()                 │
└────────────┬───────────┘  └───────────┬──────────────────┘
             │                          │
             │ Creates                  │ Creates
             │ adamCpp                  │ adam_cpp
             ▼                          ▼
┌────────────────────────┐  ┌──────────────────────────────┐
│ Rcpp Module            │  │ Wrapper Layer                │
│ (src/adamGeneral.cpp)  │  │ (_adam_general.py)           │
│ RCPP_MODULE            │  │ - adam_fitter()              │
│                        │  │ - adam_forecaster()          │
└────────────┬───────────┘  └───────────┬──────────────────┘
             │                          │
             │ Exposes                  │ Calls
             │ adamCore                 │ _adamCore
             ▼                          ▼
┌────────────────────────┐  ┌──────────────────────────────┐
│ Rcpp Bindings          │  │ pybind11 Bindings            │
│ (auto-generated)       │  │ (src/python/adamPython.cpp)  │
└────────────┬───────────┘  └───────────┬──────────────────┘
             │                          │
             └──────────┬───────────────┘
                        │ Both use
                        ▼
         ┌──────────────────────────────────┐
         │  Unified C++ Core                │
         │  src/headers/adamCore.h          │
         │                                  │
         │  class adamCore {                │
         │    - polynomialise()             │
         │    - fit()                       │
         │    - forecast()                  │
         │    - simulate()                  │
         │    - reapply()                   │
         │    - reforecast()                │
         │    - ferrors()                   │
         │  }                               │
         └──────────────────────────────────┘
```

---

## Verification Checklist

For the next developer investigating result differences:

- [ ] Verify `refineHead` value matches between R and Python
- [ ] Check `adamETS` flag is same for both
- [ ] Confirm trend type `T` is passed correctly (not 'Ad' but 'A')
- [ ] Verify all matrices are F-contiguous in Python
- [ ] Compare initial values from stage 1 optimization
- [ ] Test with simple model (ANN) first, then complex (AAA)
- [ ] Add C++ debug prints to log actual parameter values
- [ ] Compare persistence vector `vectorG` values
- [ ] Check if `lags_model_all` ordering is identical
- [ ] Verify `index_lookup_table` is constructed identically

---

## Key Learnings

1. **Type matters**: C++ binding expects `str`, not `int` for single chars
2. **Precision matters**: Use `uint64` not `uint32` for `arma::uvec`
3. **Conditional compilation**: Essential for sharing code between R and Python
4. **Wrapper pattern**: Maintains backward compatibility while using new architecture
5. **One-time instantiation**: Create `adamCore` once in architector, reuse it

---

## References

### R Implementation Files
- `/R/adam.R` - Main ADAM interface, creates `adamCpp` at line 752
- `/src/adamGeneral.cpp` - Rcpp module bindings
- `/src/RcppExports.R` - Auto-generated R exports (deleted in new version)

### Python Implementation Files
- `/python/smooth/adam_general/core/adam.py` - ADAM class
- `/python/smooth/adam_general/core/estimator.py` - Parameter estimation
- `/python/smooth/adam_general/core/creator.py` - Matrix creation
- `/python/smooth/adam_general/core/forecaster.py` - Forecasting

### C++ Core Files
- `/src/headers/adamCore.h` - Unified C++ class (600+ lines)
- `/src/headers/ssGeneral.h` - Shared utility functions
- `/src/headers/adamGeneral.h` - Legacy header (may be deprecated)

### Build Files
- `/python/CMakeLists.txt` - Python build configuration
- `/python/pyproject.toml` - Python package metadata

### Test Files
- `/python/smooth/adam_general/tests/two_stage/test_two_stage_python.ipynb`
- `/python/smooth/adam_general/tests/two_stage/test_two_stage_r.ipynb`

---

## Contact / Handoff

**Completed By**: Claude Code Assistant
**Date**: 2025-12-18
**Next Owner**: [To be assigned]

**Status**: Migration complete, module builds and runs. Results differ from R - needs investigation.

**Recommended Next Action**: Start with refineHead debugging (see Section "Next Steps #2")
