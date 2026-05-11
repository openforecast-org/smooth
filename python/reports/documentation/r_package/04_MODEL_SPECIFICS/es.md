# R: es() - Exponential Smoothing

**File**: R/adam-es.R  
**Entry**: es(), line 224  
**Python equivalent**: `ES` class (python/src/smooth/adam_general/core/es.py)

## Description

es() is a wrapper around adam() for pure ETS (Error, Trend, Seasonal) models. No ARIMA components.

## Call Flow

```
es(y, model, lags, persistence, phi, initial, ...)
  └─► adam(data=y, model=model, lags=lags, orders=list(ar=c(0),i=c(0),ma=c(0)),
           persistence=persistence, phi=phi, initial=initialValue,
           distribution="dnorm", ets="conventional", ...)
```

Key: es() passes `orders=list(ar=c(0),i=c(0),ma=c(0))` to disable ARIMA, and `distribution="dnorm"`.

## Special Parameters

- **model**: "ZXZ" (default), "ANN", "AAN", etc. Same as adam.
- **initial**: "backcasting" (default), "optimal", "two-stage", "complete"
- **ets**: "conventional" passed to adam
- Accepts previous smooth/ets model as `model` to reuse parameters

## Translated to Python

Python `ES` class subclasses `ADAM` with `ar_order=[0]`, `i_order=[0]`, `ma_order=[0]` fixed. See translation/03_COVERAGE_MATRIX.md.
