# R: oes() - Occurrence ETS

**File**: R/oes.R  
**Entry**: oes(), line 107  
**Python equivalent**: Partial (occurrence in checker, forecaster; not full oes API)

## Description

Occurrence model for intermittent demand. Estimates probability of demand occurrence. Can be used with adam(occurrence=...) for intermittent forecasting.

## Call Flow

oes() is standalone. Uses occurenceFitterWrap, occurrenceOptimizerWrap (Rcpp). adam() can take occurrence="auto" or oes model.

## Special Parameters

- **model**: "MNN" typical for occurrence
- **persistence**, **initial** for occurrence model

## Python Translation

Python has occurrence checks in checker (data_checks._check_occurrence), and occurrence handling in forecaster. Full oes() API not exposed as separate class.
