# R: msarima() - Multi-Seasonal ARIMA

**File**: R/adam-msarima.R  
**Entry**: msarima(), line 192  
**Python equivalent**: Not fully translated (ARIMA in progress)

## Description

Multi-seasonal ARIMA. Wraps adam() with model="NNN" and multi-seasonal orders. Simpler wrapper than ssarima.

## Call Flow

msarima() -> adam() with appropriate orders and lags for multiple seasons.

## Special Parameters

- **orders**: list(ar, i, ma) per seasonal lag
- **lags**: c(1, freq1, freq2, ...)

## Status

Python: ARIMA support in progress; multi-seasonal not explicitly tested.
