# Two-Stage Initialization Investigation Summary

## Problem Statement
Python and R produce different results for `initial='two-stage'` initialization in the ADAM forecasting model, despite using the same C++ backend.

## Test Case: AAN Model with Shared Data
Using identical test data (test_data_aan.csv, 120 points):

### R Results (Expected)
- **Stage 1 (complete)**: Persistence=[0.1303759, 0], Initial: level=85.51, trend=0.357
- **Stage 2 (two-stage)**: B=[0.1299162, 0, 85.82645, 0.3136136]

### Python Results (Current - WRONG)
- **Stage 1 (complete)**: Persistence=[0.13037586, 0], Initial: level=85.936, trend=0.302
- **Stage 2 (two-stage)**: B=[0.301585, 0, 155.33, 0.000525]

## Key Findings

### 1. Persistence Values Match Perfectly ✓
Both R and Python estimate identical persistence parameters in Stage 1:
- R: 0.1303759
- Python: 0.13037586

This confirms the C++ `adam_fitter()` is working correctly for persistence estimation.

### 2. Initial States Diverge in Stage 1 ❌
**R extracts**: level=85.51, trend=0.357
**Python extracts**: level=85.936, trend=0.302

Difference: ~0.4 for level, ~0.055 for trend

### 3. Both Use Same C++ adamCore Class ✓
- R: `src/adamGeneral.cpp` includes `headers/adamCore.h`
- Python: `src/python/adamPython.cpp` includes `headers/adamCore.h`
- Both receive `FitResult` with `.states` and `.profile` fields

### 4. C++ Returns Two Different Matrices

After C++ `adam_fitter()` returns (Python debug output):
- `adam_fitted['matVt'][:, 0]`: [85.936, 0.302] ← States matrix
- `adam_fitted['profile'][:, 0]`: [123.365, 0.302] ← Profile matrix

**Key Insight**: The `refineHead` block in C++ modifies `profilesRecent` during the backward pass, but does NOT update `matrixVt`. Therefore:
- `matrixVt.col(0)` contains pre-backward-pass values
- `profilesRecent.col(0)` contains post-backward-pass (refined) values

### 5. Python Extraction Logic
Python currently extracts from `matVt[:, 0]`:
```python
# estimator.py line 2232-2238
extracted_val = matrices_dict["mat_vt"][i, : lags_dict["lags_model_max"]][0]
initial_value_ets[i] = extracted_val
```

This gives [85.936, 0.302].

### 6. R Extraction Logic (Hypothesis)
R preparator function also claims to extract from `matVt[i, 1]` (line 3813):
```r
initialValueETS[[i]] <- head(matVt[i,1:lagsModelMax],1);
```

But R gets [85.51, 0.357], which is DIFFERENT from Python's [85.936, 0.302].

**Critical Question**: Why does R extract different values if both extract from the same matVt column?

## Hypotheses for the Divergence

### Hypothesis A: Different nIterations
R and Python might use different `nIterations` values for the backward pass, leading to different final states.

### Hypothesis B: R Extracts from Profile, Not MatVt
Despite what the code says, R might actually extract from `profilesRecent` or a modified `matVt` that was updated from `profile`.

### Hypothesis C: Different Post-Processing
R might apply some transformation or normalization to the extracted values that Python doesn't.

### Hypothesis D: Timing Issue
R might call preparator at a different point in the process (e.g., after additional refinement steps).

## Next Steps

1. **Add R Debug Output** to print actual `matVt[1:2, 1]` values right before preparator extracts them
2. **Compare nIterations** values used by R and Python for complete mode
3. **Check if R Updates matVt** from profile after the C++ fit
4. **Verify C++ Return Values** by enabling C++ debug output (if possible)
5. **Test the "Profile Fix"**: Enable the disabled fix at estimator.py:1088 that extracts from profile instead of matVt

## Code References

### Python
- `/home/filtheo/smooth/python/smooth/adam_general/core/estimator.py`
  - Line 1071: Updates `mat_vt` from C++ returned states
  - Line 1088: Disabled fix to extract from profile
  - Line 2232-2238: Extraction logic

- `/home/filtheo/smooth/python/smooth/adam_general/core/adam.py`
  - Line 1036-1204: Two-stage implementation

### R
- `/home/filtheo/smooth/R/adam.R`
  - Line 2610-2729: Two-stage implementation
  - Line 3692: Updates `matVt` from `adamFitted$states`
  - Line 3813-3820: Extraction logic in preparator

### C++
- `/home/filtheo/smooth/src/headers/adamCore.h`
  - Line 260-296: refineHead block
  - Line 365-370: Backward pass head filling (updates profile only!)
  - Line 388-393: Return FitResult

## Conclusion So Far

The divergence occurs because Python extracts initial states that differ from R's extracted states, even though both claim to use the same extraction logic and both call the same C++ code. The root cause appears to be related to which matrix (matVt vs profile) should be used for extraction after backcasting with refineHead.

**Most Likely Culprit**: The C++ backward pass updates `profilesRecent` but not `matrixVt` (line 365-370 of adamCore.h), and R may be extracting from a matrix that was updated from profile, while Python extracts from the stale matVt.
