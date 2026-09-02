---
name: smooth-cpp-shared
description: Work on the C++ layer that the R package and the Python port share — the headers under src/headers/, the Rcpp bindings in src/, the pybind11 bindings in src/python/, and the two build systems that compile them. Use when editing or debugging C++ in this repo, when a numeric result differs between the R and Python builds, when adding a new binding, or when a build/compilation step fails.
---

# The shared C++ layer

R and Python compile the **same algorithm sources** and differ only in their
bindings. That is what makes bit-for-bit parity achievable: an edit to a header
changes both languages at once.

## Layout

| Directory | What it holds |
|---|---|
| `src/headers/` | The algorithms. Header-only, binding-agnostic, shared by both builds. |
| `src/*.cpp` | Rcpp bindings, compiled into the R package. |
| `src/python/*.cpp` | pybind11 bindings, compiled into the Python package. |

### Headers

`adamCore.h` (fit / forecast / simulate — the core state-space recursion),
`adamGeneral.h`, `adamGradient.h` (`gradientSolve` for `initial="gradient"`),
`hessianCore.h` (finite-difference Hessian, the single source of truth for both
languages' `vcov`), `olsCore.h` (pivoted QR with a scale-invariant rank cutoff,
behind `msdecompose`'s global smoother), `matrixPowerCore.h`, `eigenCalc.h`,
`ssGeneral.h`, `ssOccurrence.h`.

`ssGeneral.h` is the one header with a `#ifdef PYTHON_BUILD` branch; the rest
compile identically for both.

### Bindings

| Algorithm | R (`src/`) | Python (`src/python/`) → module |
|---|---|---|
| ADAM core | `adamGeneral.cpp` | `adamPython.cpp` → `_adamCore` |
| Eigenvalue bounds | `eigenCalc.cpp` | `eigenCalc.cpp` → `_eigenCalc` |
| FD Hessian | `hessianCpp.cpp` | `numDeriv.cpp` → `_numDeriv` |
| OLS (pivoted QR) | `olsWrap.cpp` | `olsWrap.cpp` → `_ols` |
| Matrix power | `matrixPowerWrap.cpp` | not built — `matrix_power_wrap` in `core/utils/var_covar.py` is pure NumPy |
| State-space / occurrence | `ssGeneral.cpp` | not bound |

The Python modules land in `smooth.adam_general` (`_adamCore`, `_eigenCalc`,
`_numDeriv`, `_ols`). `src/python/matrixPowerWrap.cpp` exists but has no
`pybind11_add_module` entry in `python/CMakeLists.txt`.

## Building

**R** — `R CMD INSTALL .` or `devtools::load_all()`. Rcpp + RcppArmadillo, flags
from `src/Makevars`. `devtools::load_all()` recompiles what changed, which is
what the Python `r_parity` tests drive through `tests/_r_bridge.py`.

**Python** — CMake + scikit-build-core + pybind11, with carma bridging NumPy and
Armadillo. `python/CMakeLists.txt` declares one `pybind11_add_module` per
binding, each compiled with `PYTHON_BUILD` defined and `../src` on the include
path. Rebuild with `cd python && python3 -m pip install -e .`; a pure-Python
edit needs no rebuild.

## Rules

**Matrices are Fortran (column-major) order.** Armadillo requires it. Both
binding layers convert at the boundary; get it wrong and you read transposed
data with no error.

**Never let the compiler contract floating-point operations.** A fused
multiply-add rounds once where the source asks for twice. That is one ULP, and
in an iterative routine it compounds — a whole class of "works on Linux, fails
on macOS" bugs, because every arm64 chip has an FMA instruction and baseline
x86-64 does not. Do not add `-ffast-math`, `-Ofast` or `-march=native` to either
build. If a numeric result differs between platforms and the algorithm is
identical, test the hypothesis directly: rebuild on x86-64 with
`-mfma -ffp-contract=fast` and see whether the difference reproduces.

**Change the header, not one binding.** Anything that alters numbers belongs in
`src/headers/` so both languages move together. A fix applied to only one side
is a parity regression, even when it makes that side more correct.

**Verify against R, not against intuition.** After a header change, run the
Python `r_parity` markers — they load the *local* R source via
`devtools::load_all()`, so they compare the C++ you just edited:

```bash
cd python && .venv/bin/python -m pytest tests/ -m "r_parity or r_comparison"
```

Baseline is 457 passed, 3 xfailed. Anything else is yours.

**Read a difference correctly.** Same kernel, same optimiser, same
initialisation means a materially different optimum cannot be an optimiser
artefact — see the `smooth-translation` skill for the diagnostic.
