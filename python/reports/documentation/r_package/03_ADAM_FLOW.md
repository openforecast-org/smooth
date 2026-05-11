# R ADAM Internal Flow

Call graph for `adam()` from R/adam.R. Local functions architector, creator, filler, initialiser are defined inside adam().

## Main Flow

```
adam(data, model, lags, orders, ...)
  │
  ├─► parametersChecker() [adamGeneral.R]
  │     Returns checked params, yInSample, holdout, lags, model type, etc.
  │
  ├─► architector() [local, adam.R ~656]
  │     Inputs: etsModel, Etype, Ttype, Stype, lags, lagsModelSeasonal, ...
  │     Returns: lagsModel, lagsModelAll, lagsModelMax, componentsNumberETS,
  │              indexLookupTable, profilesRecentTable, adamCpp (C++ class)
  │
  ├─► creator() [local, adam.R ~750]
  │     Inputs: etsModel, Etype, Ttype, Stype, obsStates, obsInSample, ...
  │     Returns: matVt, matF, vecg, matw, and other matrices
  │
  ├─► initialiser() [local, adam.R ~1402]
  │     Returns: B (parameter vector), Bl (lower bounds), Bu (upper bounds)
  │
  ├─► Optimization loop
  │     For each B from optimizer:
  │       filler(B, ...) [local, adam.R ~1194]
  │         → Updates matVt, matF, vecg, matw with B
  │       adamCpp$fit(...)  [C++ adamCore]
  │         → Returns matVt, yFitted, errors, profile
  │     Cost: likelihoodFunction(B) or similar
  │
  └─► For forecasting:
        filler(B_final, ...) → filled matrices
        forecasterwrap(matvt, matF, matw, h, ...) [RcppExports.R]
          → Point forecast
```

## Key Line References (adam.R)

| Step | Approx Line |
|------|-------------|
| adam() entry | 326 |
| architector() def | 656 |
| creator() def | 750 |
| filler() def | 1194 |
| initialiser() def | 1402 |
| First architector call | ~2377 |
| First creator call | ~2384 |
| First initialiser call | ~2408 |
| filler in optimization | ~1844, 2316, 2748, 2807, 2831, 3454 |
| adamProfileCreator (used by architector) | 4881 |

## C++ adamCore Usage

Created in architector (line ~730):
```r
adamCpp <- new(adamCore, lagsModelAll, Etype, Ttype, Stype, ...)
```

Used for fitting:
```r
adamCpp$fit(matVt, matWt, matF, vecG, ...)
```

Point forecasting uses `forecasterwrap` (Rcpp) which wraps C++ forecaster, not adamCore.forecast directly in some code paths.

## Data Flow

- **B**: Parameter vector (persistence, phi, initials, ARIMA, xreg)
- **matVt**: State matrix (rows=components, cols=time)
- **matF**: Transition matrix
- **vecg**: Persistence vector (alpha, beta, gamma, ...)
- **matw**: Measurement vector
- **profilesRecentTable**, **indexLookupTable**: For seasonal indexing
