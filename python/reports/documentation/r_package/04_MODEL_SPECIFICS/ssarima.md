# R: ssarima() - State-Space ARIMA

**File**: R/adam-ssarima.R  
**Entry**: ssarima(), line 110  
**Python equivalent**: ADAM with ar_order, i_order, ma_order (partial)

## Description

ssarima() builds pure ARIMA state-space model. Uses model="NNN" (no ETS). Has its own filler and initialiser (different from adam's).

## Call Flow

ssarima() eventually calls adam() with model="NNN" and orders. But it has local:
- **filler()**: B, matVt, matF, vecG, matWt; arRequired, maRequired, etc. (line ~252)
- **initialiser()**: (line ~715)

## Special Parameters

- **orders**: list(ar=c(0), i=c(1), ma=c(1))
- **lags**: c(1, frequency(y)) or multi-seasonal
- **constant**: FALSE
- **arma**: Fixed ARMA parameters

## Python Translation

Python ADAM supports ARIMA via ar_order, i_order, ma_order. See checker/arima_checks.py, creator polynomial setup. Status: partial (in progress on feature/arima branch).
