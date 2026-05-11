# R: gum() - General Univariate Model

**File**: R/adam-gum.R  
**Entry**: gum(), line 98  
**Python equivalent**: Not translated (R only)

## Description

GUM uses orders and lags to define a state-space structure (similar to ARIMA but different formulation). Has its own creator, filler, initialiser.

## Call Flow

gum() does NOT call adam() directly. It uses ssfunctions (ssInput, ssForecaster) and has model-specific:
- **creator()**: Builds matrices for GUM structure (line ~379)
- **filler()**: Populates from B (line ~269)
- **initialiser()**: Parameter bounds (line ~680)

## Special Parameters

- **orders**: c(1,1) default; defines model structure
- **lags**: c(1, frequency(y))
- **type**: "additive" or "multiplicative"
- **transition**, **measurement**, **persistence**: Can be provided

## Status

Python: No GUM implementation.
