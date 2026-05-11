# R: ces() - Complex Exponential Smoothing

**File**: R/adam-ces.R  
**Entry**: ces(), line 91  
**Python equivalent**: Not translated (R only)

## Description

CES uses complex-valued smoothing. Has its own creator, filler, initialiser. Uses parameters a, b.

## Call Flow

ces() does NOT call adam(). Uses ssfunctions and model-specific:
- **creator()**: seasonality, xregModel (line ~379)
- **filler()**: B, matVt, matF, vecG, a, b (line ~269)
- **initialiser()**: (line ~680)

## Special Parameters

- **seasonality**: "none", "simple", "partial", "full"
- **a**, **b**: Complex smoothing parameters
- **lags**: c(frequency(y))

## Status

Python: No CES implementation.
