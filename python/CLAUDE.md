# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Always update `python/NEWS.md` after changing Python code

When you make a user-visible change to the Python package — new feature, bugfix,
dependency change, API change, supported-version change — add an entry under the
current unreleased version in `python/NEWS.md`. If no unreleased section exists
yet, create one above the latest released version. Keep entries terse and grouped
under `Changes:` / `Bugfixes:`, matching the existing style.

Skip the changelog only for: pure refactors with no behaviour change, comment/doc
typo fixes, and edits to test files that don't change tested behaviour. When in
doubt, add the entry.

## Never add a `frequency` parameter

Do not add a `frequency` (or `period`, `seasonality`, or any equivalent) parameter
to any class or function. Seasonal period is always inferred automatically from the
data — from a pandas `DatetimeIndex`, from `lags`, or from the model specification.
Exposing it as a parameter creates redundancy and diverges from the R API.

**Exception:** the standalone `sim_*` simulators (`sim_es`, `sim_gum`, `sim_ces`,
`sim_ssarima`, `sim_sma`, `sim_oes`) accept `frequency=<int>` to match the R API
one-for-one. They are data-generators with no fitted state or input series to
infer seasonality from, so the parameter is load-bearing. This exception does
**not** extend to `ADAM.simulate()` / `ES.simulate()` / etc., which read `lags`
straight off the fitted model.

## Never clip, clamp, or patch around bad numerics

This rule applies across the whole project — `adam`, `OM`, `OMG`, `ES`,
`MSARIMA`, `AutoADAM`, the cost functions, the link functions, the C++
`adamCore` path, and anywhere else a numerical signal flows from the model
into a loss or back into the optimiser.

- **Do not clip or clamp the model output.** Fitted values, states,
  residuals, forecasts, intermediate vectors out of the C++ — leave them
  exactly as the model produced them. If a value lands outside an
  expected range (a probability outside `[0, 1]`, a state diverging,
  a forecast going negative for a multiplicative model), that is a true
  signal of a model–data mismatch, a bad initialisation, or an upstream
  bug. Clipping turns those signals into silently-wrong results.
- **Do not patch the loss function with `pmax` / `np.maximum` floors
  inside `log()` (or analogous "epsilon" safeguards) either.** It looks
  like numerical hygiene but it does the same thing: a `-Inf` log-
  likelihood is the optimiser telling you "the parameters at this point
  are inconsistent with the data". Hiding it loses the diagnostic.
- **There is one correct exception: an infeasibility guard at the top of
  the cost function.** A check like
  `if(any(is.nan(yFitted)) || any(yFitted<0) || any(yFitted>1)) return(1e+300)`
  in `om()`'s cost is *not* a clip — it tells the optimiser "the
  parameters at this point are inconsistent with the model" and returns a
  uniformly large penalty so it steers away. That is fine. The bad
  patterns are silent in-place fixes (`pmax`/`np.maximum` inside `log()`,
  `np.clip` on the fitted vector, "if cost is `Inf` return `1e10`", etc.)
  that change the value flowing forward without flagging anything.
- **When the infeasibility guard fires repeatedly and the optimiser is
  stuck on the penalty plateau, that is also a bug report.** Usually the
  initialiser handed it a broken `x0`, or the model–data combination is
  invalid and should have been rejected at parameter-check time.
- **The right responses, in order of preference:**
  1. Fix the initialiser so the optimiser's starting point produces a
     finite, finite-gradient cost in the feasible region.
  2. Reject incompatible model / data / option combinations at parameter
     check time with an informative error.
  3. Surface (don't hide) the signal when the user is exploring a
     known-fragile combination — warn, don't suppress.

When you are about to write `pmax(x, 1e-15)`, `np.clip(...)`, `if x < 0
return 1e+300`, `if cf is Inf return 1e10`, or anything in that family:
**stop and find the actual root cause first.**

## Distributions come from greybox

`greybox` supplies `dnorm`, `dlaplace`, `ds`, `dgnorm`, `dalaplace`, `dlnorm`,
`dllaplace`, `dls`, `dlgnorm`, `dinvgauss` and `dgamma`, plus their `p`/`q`/`r`
counterparts. These are the same functions R's `smooth` calls, so **do not
re-implement a density or quantile against `scipy`** — import from `greybox` and spell
out only the parameterisation. Re-derivation is how the port ended up with five
distributions whose prediction intervals were wrong.

The `r*` random draws are the one exception: they keep `scipy`, because greybox's take
no `random_state` and seeded reproducibility matters.

## Important Testing Note

**RNG Differences**: R and Python use different random number generation algorithms. Even with the same seed (e.g., `set.seed(33)` in R and `np.random.seed(33)` in Python), they will produce completely different random data.

When comparing Python implementations against R:
1. Generate data in R and save to CSV: `write.csv(data.frame(value=time_series), "test_data.csv", row.names=FALSE)`
2. Read the same data in Python: `pd.read_csv("test_data.csv")`
3. This ensures both implementations work with identical input data for valid comparison

## Repository Overview

The **smooth** repository contains time series forecasting implementations in both R and Python. The R package is the mature, production version available on CRAN. The Python implementation is a direct translation focusing on the ADAM (Augmented Dynamic Adaptive Model) forecasting framework. Both are developed on `master`; the `Python` branch is historical.

**Primary Languages**: R (production), Python (in development)

**Repository Structure**:
- `R/` - R package source code (production)
- `python/` - Python implementation (active development)
- `src/` - C++ source code for R package
- `tests/` - R package tests
- `man/` - R package documentation

## Python Development (Primary Focus)

### Working Directory
All Python development happens in the `python/` subdirectory. Navigate there before running commands:
```bash
cd python/
```

### Build and Installation

The package uses scikit-build-core for building C++ extensions with pybind11:

```bash
# Install in editable mode with development dependencies
pip install -e ".[dev]"

# Or use the Makefile target
make install

```
to run sth activate .venv from the main smooth folder and then do cd python install using python3 -m pip install -e . ...
Update only if you do C++ code changes not python code changes.


**Build System**: Uses CMakeLists.txt for C++ extensions, requires pybind11 and numpy

### Testing

```bash
# Run pytest (from python/)
.venv/bin/python -m pytest tests/ -q

# Or use the Makefile
make test
```

**Test location**: `python/tests/` — a pytest suite of 914 tests (a further 460 are
deselected by default; see below).

- `tests/data/` holds the R-generated reference fixtures.
- `tests/R scripts/` holds the generators that produce them. Regenerate with
  `Rscript "python/tests/R scripts/<name>.R"` **from the repository root**.
- `tests/notebooks/` holds benchmark runners and notebooks. These are *not* part of the
  suite and are not linted — `ruff check tests/` will report a wall of pre-existing
  errors from them, which is why the linting commands target `src/` only.

**Deselected markers**: `pyproject.toml` deselects `r_comparison` and `r_parity` by
default, because they need R with `smooth`'s dependencies, `devtools` and `jsonlite`
installed. Where that is available they run as-is:

```bash
.venv/bin/python -m pytest tests/ -m "r_parity or r_comparison"
```

They shell out to R through `tests/_r_bridge.py`, which loads the *local* R source
with `devtools::load_all()` — so they always compare against the working tree, and an
R-side edit is picked up without reinstalling. The repo root is derived from the
module's own location, so the bridge follows the checkout. Current baseline: **457
passed, 3 xfailed** (the three xfails are all `simulate.om`, reasons recorded in
`tests/test_om_simulate_r_parity.py`). Any other failure is yours.

### Linting and Code Quality

**Always run ruff check, ruff format, and mypy after every code change:**

```bash
# Run ruff linter (must pass with zero errors)
.venv/bin/ruff check src/

# Run ruff formatter
.venv/bin/ruff format src/

# Run mypy type checker (must pass with zero errors)
.venv/bin/mypy src/smooth
```

These three commands must be run together after any Python source edit. Fix all errors reported by `ruff check` and `mypy` before considering a task complete.

**Whenever you make an important change** — changing a function's signature (inputs/outputs), adding or removing parameters, changing return types, adding or removing classes or large blocks of code, touching any base class whose subclasses override the same method — **you MUST run all three checks**. Signature changes on a base class break LSP-compatibility on subclass overrides, and mypy is the only thing that will catch that before CI does. Don't skip mypy because "it's just a small change" — the LSP-break in `OM.predict` from a single added kwarg on `ADAM.predict` is exactly the class of bug it catches.

**Linting Config**: Defined in `pyproject.toml` under `[tool.ruff]`
- Line length: 88 (Black-compatible)
- Notable exceptions: Allows uppercase variable names (N803, N806) for matrices

### Development Environment Setup

```bash
# Create virtual environment and install package with pre-commit hooks
make environment

# This creates .venv, installs the package, sets up Jupyter kernel, and installs pre-commit hooks
```

### CI/CD

Python CI (`.github/workflows/python_ci.yml`) runs on pull requests touching
`python/**` and on workflow_dispatch. The `lint` job runs `ruff check`, `ruff format
--check` and `mypy`; `build-and-test` then uses cibuildwheel to build a wheel and run
the pytest suite against it on Linux (3.11-3.14), macOS and Windows
(`CIBW_TEST_COMMAND: pytest {project}/python/tests`). The `r_parity` /
`r_comparison` markers stay deselected there — CI has no R — so run those locally.

## Python Package Architecture

The Python ADAM implementation follows a modular, scikit-learn-style design:

### Core Module Structure
```
python/src/smooth/adam_general/
├── _adamCore, _eigenCalc, _ols, ...   # compiled C++ extensions (pybind11)
└── core/
    ├── adam.py               # Main ADAM class (user interface)
    ├── es.py, sma.py, msarima.py, auto_msarima.py, auto_adam.py
    ├── om.py, omg.py, auto_om.py      # occurrence models
    ├── ces/, ces_model.py             # complex exponential smoothing
    ├── sm.py                          # scale model (R's sm(); see docs/sm.rst)
    │                                  # exported as smooth.sm; also ADAM.sm()
    ├── plotting.py
    ├── checker/              # parameter validation (split by concern)
    ├── creator/              # architector, creator, filler, initialiser
    ├── estimator/            # estimation, optimisation, model/ARIMA selection
    ├── forecaster/           # preparator, forecaster, intervals
    ├── simulate/
    └── utils/
        ├── cost_functions.py # Optimisation cost functions (CF, log_Lik_ADAM)
        ├── ic.py             # Information criteria (AIC, BIC, etc.)
        ├── polynomials.py    # ARIMA polynomial utilities
        ├── utils.py          # msdecompose, ACF/PACF, likelihoods
        ├── var_covar.py      # sigma, covariance matrices
        └── reforecast.py     # reforecast/reapply helpers
```

The `checker`/`creator`/`estimator`/`forecaster` names are packages now, not single
modules — the flow below still holds, but each step lives in its own subpackage.

### Critical Function Flow

**Model Fitting Pipeline**:
1. `ADAM.__init__()` - Store configuration
2. `ADAM.fit(y, X)` - Main fitting method
   - `parameters_checker()` (checker.py) - Validate all inputs
   - `architector()` (creator.py) - Define model architecture (components, lags, profiles)
   - `creator()` (creator.py) - Build state-space matrices (mat_vt, mat_wt, mat_f, vec_g)
   - `estimator()` (estimator.py) - Optimize parameters using NLopt
     - `initialiser()` (creator.py) - Get initial parameter vector B and bounds
     - `CF()` (cost_functions.py) - Cost function evaluated during optimization
       - `filler()` (creator.py) - Fill matrices with current parameters from B
       - `adam_fitter()` (_adam_general.py) - C++ fitting routine
3. `ADAM.predict(h, X)` - Generate forecasts, returns `ForecastResult`
   - `preparator()` (forecaster.py) - Prepare model for forecasting
   - `forecaster()` (forecaster.py) - Generate point forecasts and intervals, returns `ForecastResult`
     - `adam_forecaster()` (_adam_general.py) - C++ forecasting routine

```
┌───────────┐    ┌─────────┐    ┌─────────┐    ┌───────────┐    ┌────────────┐
│ User input│───►│ checker │───►│ creator │───►│ estimator │───►│ forecaster │
└───────────┘    └─────────┘    └─────────┘    └───────────┘    └────────────┘
      │                                                                │
      └──────────────────────  ADAM (public API)  ─────────────────────┘
```

**Key Functions**:
- `filler()` in creator.py: Central function that populates state-space matrices with parameter values (called repeatedly during optimization and before forecasting)
- `CF()` in cost_functions.py: Main cost function for parameter estimation
- `parameters_checker()` in checker.py: Comprehensive input validation

### State-Space Matrices

The package uses state-space representation with these matrices:
- `mat_vt`: State vector (levels, trends, seasonal components, ARIMA states)
- `mat_wt`: Measurement matrix (maps states to observations)
- `mat_f`: Transition matrix (state evolution)
- `vec_g`: Persistence vector (smoothing parameters: α, β, γ)

**Important**: Matrices must be in Fortran (column-major) order for C++ compatibility

### Parameter Vector (B)

The optimization parameter vector B contains (in order):
1. ETS persistence parameters (α, β, γ)
2. Damping parameter (φ)
3. Initial states
4. ARIMA parameters (AR, MA coefficients)
5. Regression coefficients
6. Constant term
7. Distribution parameters

## R Package Development

### Build and Test
```bash
# From repository root
R CMD build .
R CMD check smooth_*.tar.gz

# Or use devtools in R
devtools::load_all()
devtools::test()
```

### Key R Files
- `R/adam.R` - Main ADAM function (original implementation that Python translates)
- `src/` - C++ code for performance-critical operations

## Translation Philosophy

The Python implementation is a **direct translation** of the R version:
- Maintains same algorithms and mathematical formulations
- Uses object-oriented design (scikit-learn style) vs R's functional approach
- Function names and logic mirror R counterparts
- See the `smooth-translation` skill for the R↔Python map and the parity rules

**When modifying Python code**: Compare with corresponding R implementation in `R/adam.R` to maintain equivalence

## Important Development Notes

### Naming Conventions
- Python uses snake_case for functions: `parameters_checker()`, `adam_fitter()`
- Matrix variables use prefix: `mat_` for matrices, `vec_` for vectors
- Fitted model attributes use trailing underscore (scikit-learn convention): `persistence_level_`, `phi_`

### Performance Considerations
- C++ extensions via pybind11 for performance-critical matrix operations
- NLopt library used for parameter optimization (not scipy)
- Matrix operations should be vectorized with NumPy
- `filler()` function is called many times during optimization - keep it efficient

### Distribution Support
Supported error distributions: dnorm, dlaplace, ds, dgnorm, dlnorm, dgamma, dinvgauss

Default distribution selection:
- Additive errors: dnorm
- Multiplicative errors: dgamma
- MAE/MACE loss: dlaplace
- HAM/CHAM loss: ds

### Common Constraints
- ETS smoothing parameters: 0 ≤ α, β, γ ≤ 1
- Trend constraint: β ≤ α
- Seasonal constraint: γ ≤ 1 - α
- Damping: 0 ≤ φ ≤ 1
- Violations return penalty value (1e100) during optimization

## Usage Example

```python
import numpy as np
from smooth import ADAM

# Sample data
y_data = np.array([10, 12, 15, 13, 16, 18, 20, 19, 22, 25, 28, 30,
                   11, 13, 16, 14, 17, 19, 21, 20, 23, 26, 29, 31])

# Initialize the model
model = ADAM(model="ANN", lags=[1,12])  # Additive error, no trend, no seasonality

# Fit the model to data
model.fit(y_data)

# Generate forecasts (returns ForecastResult)
fc = model.predict(h=10)
fc.mean              # pd.Series of point forecasts

# Generate prediction intervals
fc = model.predict(h=10, interval="prediction", level=[0.8, 0.95])
fc.lower             # pd.DataFrame, columns are quantile values
fc.upper             # pd.DataFrame, columns are quantile values
fc.to_dataframe()    # flat pd.DataFrame with prefixed column names
```

## Reference Documentation

- **Skills** (`.claude/skills/`, invocable by name):
  - `smooth-translation` - the R↔Python name map (arguments, fitted attributes,
    matrices, internal dicts), both call flows side by side, the parity rules,
    and the checklist for landing a translation.
  - `smooth-cpp-shared` - the shared `src/headers/` layer, the Rcpp and pybind11
    bindings, both build systems, and the floating-point rules.
  - `explain-smooth` - explaining fitted models and choosing a function, for
    both languages.
- `docs/` - Sphinx documentation. `docs/sm.rst` is the scale model (R's `sm()`); build
  with `.venv/bin/python -m sphinx -b html docs docs/_build/html` after
  `pip install -e ".[docs]"`.
- `python/ADAM_API_REFERENCE.md` - every `ADAM` constructor argument, property and
  public method, with signatures.
- [Wiki](https://github.com/openforecast-org/smooth/wiki) - user-facing reference for
  both languages. `Roadmap` and `R-Python-differences` are the maintained answers
  on port coverage and numerical parity; do not duplicate them in the repo.
- `README.md` - High-level package overview (R-focused)

## Git Workflow

**Working branch**: `master` — both the R package and the Python port are developed
here. The `Python` branch is historical.

## R / Python API Parity

The Python implementation must match R's public API as closely as possible. **Parameters, defaults, return types, attributes, and output structure should be equivalent** unless a language difference makes strict parity impossible (e.g. R uses `...` / `formula`, Python uses `X=` / keyword args).

When adding or removing parameters from any class (`ADAM`, `OM`, `OMG`, `AutoOM`, `AutoADAM`, etc.), **check the corresponding R function signature first**. If Python has parameters or return-type behaviour that R does not (or vice-versa), flag the discrepancy explicitly before implementing. In particular:

- Functions that return a fitted object in R should have `.fit()` return the same type in Python (not a separate wrapper class). For example, `auto.om()` returns the best `om` object; `AutoOM.fit()` must return the best `OM` or `OMG`.
- Parameter names may differ between R and Python (camelCase → snake_case is acceptable). Extra parameters with no R equivalent are not allowed without approval.
- Fitted attributes should mirror R's `$` access: if R has `m$timeElapsed`, Python should have `m.time_elapsed_`.
