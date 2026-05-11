# R: sma() - Simple Moving Average

**File**: R/adam-sma.R  
**Entry**: sma(), line 101  
**Python equivalent**: Not translated (R only)

## Description

SMA in state-space form. Uses IC to select order if order=NULL. Has distinct logic from adam (no ETS, no ARIMA in conventional sense).

## Call Flow

sma() uses ssInput/ssForecaster or similar. Order selection via AICc/AIC/BIC/BICc if order not provided.

## Special Parameters

- **order**: Moving average order (or NULL for auto)
- **ic**: "AICc", "AIC", "BIC", "BICc" for order selection
- **fast**: TRUE

## Status

Python: No SMA implementation.
