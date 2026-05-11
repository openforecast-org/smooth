# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

The **smooth** package implements forecasting using state space models in R. The centerpiece is ADAM (Augmented Dynamic Adaptive Model), which combines ETS (Error, Trend, Seasonal) models, ARIMA, and regression in a unified Single Source of Error framework. The package is production-ready, available on CRAN, and includes both R and C++ implementations for performance.

A Python implementation is under development in the `python/` subdirectory (see `python/CLAUDE.md` for Python-specific guidance).

## Claude instructions
Do not create summaries of what you do. Do not create additional files/documents if not explicitly asked to.

When coding, use the best practice, focusing on the following principles:

- Write as few lines as possible.
- Use appropriate naming conventions.
- Segment blocks of code in the same section into paragraphs.
- Use indentation to mark the beginning and end of control structures. Specify the code between them.
- Don’t use lengthy functions. Ideally, a single function should carry out a single task.
- Use the DRY (Don’t Repeat Yourself) principle. Automate repetitive tasks whenever necessary. The same piece of code should not be repeated in the script.
- Avoid Deep Nesting. Too many nesting levels make code harder to read and follow.
- Avoid long lines. It is easier for humans to read blocks of lines that are horizontally short and vertically long.

When writing R code specifically:
- Always use explicit `return(object)` at the end of every function.
- Always wrap `if`, `for`, and `while` bodies in `{}`, even for single-line bodies: `if (...) { ... }`.


## R Package Development

### Build and Test Commands

```bash
# Build package from source
R CMD build .

# Check package (CRAN standards)
R CMD check smooth_*.tar.gz

# Install locally for testing
R CMD INSTALL .

# Or use devtools in R session
R -e "devtools::load_all()"
R -e "devtools::test()"
R -e "devtools::check()"

# Run specific tests
R -e "testthat::test_file('tests/testthat/test_adam.R')"
```

### Testing

Test files are in `tests/testthat/`:
- `test_adam.R` - ADAM model tests
- `test_es.R` - Exponential Smoothing tests
- `test_ssarima.R` - State-Space ARIMA tests
- `test_ces.R` - Complex Exponential Smoothing tests
- `test_gum.R` - Generalised Uniform Model tests
- `test_oes.R` - Occurrence model tests
- `test_simulate.R` - Simulation function tests

Run all tests with: `R -e "devtools::test()"`

### CI/CD

GitHub Actions workflows:
- `.github/workflows/test.yml` - R-CMD-check on macOS, Windows, Ubuntu
- `.github/workflows/rhub.yaml` - R-hub checks for CRAN submission
- `.github/workflows/python_ci.yml` - Python linting (Python branch only)

### Documentation

Build documentation: `R -e "devtools::document()"`

Vignettes (in `vignettes/`):
- `adam.Rmd` - ADAM model guide
- `es.Rmd`, `ces.Rmd`, `gum.Rmd` - Specific model guides
- `ssarima.Rmd`, `sma.Rmd` - ARIMA and moving average guides
- `simulate.Rmd` - Simulation functions
- `oes.Rmd` - Occurrence models for intermittent demand

Build vignettes: `R -e "devtools::build_vignettes()"`

## Architecture

### State Space Model Framework

All models use a unified state space representation:

```
Observation equation: y_t = o_t * (w(v_{t-l}) + h(x_t, a_{t-1}) + r(v_{t-l}) * ε_t)
State equation: v_t = f(v_{t-l}, a_{t-1}) + g(v_{t-l}, a_{t-1}, x_t) * ε_t
```

Where:
- `v_t` - State vector (levels, trends, seasonal components, ARIMA states)
- `l` - Vector of lags
- `w(.)` - Measurement function
- `f(.)` - Transition function
- `g(.)` - Persistence function
- `ε_t` - Error term (various distributions supported)

### Core Model Types

1. **ADAM (`R/adam.R`)** - Main unified model combining:
   - ETS components (error, trend, seasonality)
   - ARIMA components (AR, I, MA orders)
   - Regression with exogenous variables
   - Multiple seasonal patterns

2. **ETS models** (`R/es.R`, `R/adam-es.R`) - Exponential Smoothing variants

3. **ARIMA models** (`R/ssarima.R`, `R/adam-ssarima.R`) - State-Space ARIMA

4. **CES** (`R/ces.R`, `R/adam-ces.R`) - Complex Exponential Smoothing

5. **GUM** (`R/gum.R`, `R/adam-gum.R`) - Generalised Uniform Model

6. **Occurrence models** (`R/oes.R`) - For intermittent demand

### R and C++ Integration

Critical performance functions are implemented in C++ (in `src/`):

- `adamGeneral.cpp` - Core ADAM fitting/forecasting (uses `adamCore.h`)
- `ssGeneral.cpp` - General state space operations
- `ssOccurrence.cpp` - Occurrence model operations
- `ssSimulator.cpp` - Simulation functions
- `matrixPowerWrap.cpp` - Matrix power operations for variance calculations

The R code (in `R/`) provides:
- User-facing API and parameter validation
- Model selection logic
- Information criteria calculations
- Plotting and diagnostics
- Integration with R ecosystem (formula interface, time series objects)

### Key Function Flow in ADAM

1. **Input validation** (`R/adamGeneral.R::parametersChecker()`) - Validates all user inputs, processes data, handles formula interface

2. **Model architecture setup** (`architector()` in R code) - Determines components, lags, seasonal structure

3. **Model creation** (`creator()` in R code) - Builds state-space matrices (measurement, transition, persistence)

4. **Parameter estimation** (`estimator()` in R code) - Calls C++ optimization routines:
   - Uses `nloptr` for optimization
   - C++ `adamFitter()` computes likelihood/errors during optimization
   - Returns fitted parameters, states, errors

5. **Model selection** (if `model="ZZZ"` or similar) - Tests multiple models, selects by IC

6. **Forecasting** (`forecaster()` in R code) - Calls C++ `adamForecaster()` for point forecasts and intervals

## Important Implementation Details

### Distribution Support

Error distributions (via `distribution` parameter):
- `dnorm` - Normal (default for additive)
- `dlaplace` - Laplace
- `ds` - S distribution
- `dgnorm` - Generalised Normal
- `dlnorm` - Log-Normal
- `dgamma` - Gamma (default for multiplicative)
- `dinvgauss` - Inverse Gaussian

Distribution selection affects likelihood calculation and prediction intervals.

### Parameter Constraints

ETS smoothing parameters must satisfy:
- 0 ≤ α, β, γ ≤ 1 (persistence parameters)
- β ≤ α (trend constraint)
- γ ≤ 1 - α (seasonal constraint)
- 0 ≤ φ ≤ 1 (damping parameter)

Violations return penalty during optimization (1e100).

### Model Notation

ETS models use three-letter codes:
- First letter: Error type (A=Additive, M=Multiplicative)
- Second: Trend type (N=None, A=Additive, Ad=Additive damped, M=Multiplicative, Md=Multiplicative damped)
- Third: Seasonal type (N=None, A=Additive, M=Multiplicative)

Example: `model="AAdN"` = Additive error, Additive damped trend, No seasonality

ARIMA specified via `orders` parameter: `list(ar=c(p1,p2,...), i=c(d1,d2,...), ma=c(q1,q2,...))`

### Lags and Multiple Seasonality

The `lags` parameter specifies seasonal periods: `lags=c(1,12)` for monthly data with annual seasonality.
- First lag typically = 1 (for level/trend)
- Additional lags for seasonal components
- Multiple seasonal patterns supported (e.g., hourly data with daily/weekly patterns)

### C++ Matrix Requirements

When modifying C++ code, matrices must be in Fortran (column-major) order for Armadillo compatibility. The R code handles conversion between R matrices and C++ via RcppArmadillo.

## Common Development Patterns

### Adding New Distribution

1. Add distribution to list in `R/adam.R` documentation
2. Implement log-likelihood in C++ (`adamCore.h`)
3. Add density/quantile functions for intervals
4. Update distribution selection logic in `parametersChecker()`

### Modifying State Space Structure

1. Update `creator()` for matrix construction
2. Modify C++ `adamFitter()` if transition logic changes
3. Ensure `filler()` correctly populates parameter vector
4. Update `forecaster()` for prediction with new structure

### Adding Model Variants

Follow existing pattern (e.g., `R/adam-es.R`):
1. Create wrapper function with specific defaults
2. Call main `adam()` with appropriate parameters
3. Add tests in `tests/testthat/`
4. Document with roxygen2 comments

## Dependencies

Key R packages:
- **Rcpp, RcppArmadillo** - C++ integration (required)
- **greybox** - Distributions, information criteria, utilities (required)
- **nloptr** - Nonlinear optimization (required)
- **pracma** - Mathematical functions (required)
- **zoo** - Time series handling (required)
- **testthat** - Testing (suggested)
- **numDeriv** - Numerical derivatives for diagnostics (suggested)

C++ dependencies:
- Armadillo library (via RcppArmadillo)
- C++11 or later

## Mac OS Specific Notes

Mac users may need gfortran libraries for Rcpp/RcppArmadillo compilation. See: http://www.thecoatlessprofessor.com/programming/rcpp-rcpparmadillo-and-os-x-mavericks-lgfortran-and-lquadmath-error/

If functions stop working after package upgrade, restart R or completely remove and reinstall the package (C++ functions can be cached).

## Python Translation

A Python implementation is under development in `python/` subdirectory. The Python version:
- Directly translates R algorithms to Python/NumPy
- Uses scikit-learn-style API (fit/predict methods)
- Reimplements core logic in Python with C++ extensions via pybind11
- See `python/CLAUDE.md` for Python-specific development guidelines

When modifying R code that affects algorithms, check if corresponding Python code in `python/smooth/adam_general/core/` needs updates to maintain equivalence.

## Git Workflow

- **Main branch**: `master` (stable, CRAN-ready)
- **Development branch**: `Python` (active Python development)
- Python changes should be made on Python branch
- R changes on master or feature branches
