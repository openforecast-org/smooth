# Plan — arbitrary loss functions for initial="gradient"

**Scope:** extend the gradient initial-state solve so it profiles the *actual*
estimation loss over the initial states, not the one-step SSE surrogate it uses
now. R + Python via the shared C++ core (`adamCore::gradientSolve`), exact
cross-language parity by construction. Builds directly on the analytic-Jacobian
machinery committed in `9b7da03b`.

---

## 0. Background: the mismatch and the asset

Current state:

- The solve minimises `sum(e_t^2)` of the `errorf` residuals (relative errors
  for E='M'), regardless of the `loss` / `distribution` the outer optimiser
  uses. For `loss="likelihood"` with dnorm this *is* the concentrated MLE of
  the initials; for everything else it is an approximation.
- Measured consequence: ETS(MAM) on AirPassengers with the Gamma likelihood —
  the SSE-profiled initials can produce a *worse* likelihood than backcasting
  until the LM fallback compensates. Multistep losses (MSEh, TMSE, GTMSE,
  MSCE, GPL) are worse: the initials are 1-step-optimal, never h-step-optimal.

The asset: after `9b7da03b`, **one forward pass yields both the residuals e0
and their exact Jacobian J = d(e)/d(theta)** (analytic sensitivities, both
`ets="conventional"` and `ets="adam"` maths). For additive ETS the residuals
are exactly affine: `e(delta) = e0 + J*delta` with constant J. That turns
loss-profiling into a tiny nFree-dimensional problem where candidate residuals
are a matrix-vector product — **no further model passes**.

## 1. Loss family: everything reduces to sum(rho(e_t))

At fixed persistence (and concentrated scale), every supported loss is a
separable function of the errorf residuals:

| loss / distribution | rho(e) | IRLS weight w = rho'(e)/e |
|---|---|---|
| MSE, likelihood+dnorm | e^2 | 1 |
| MAE, likelihood+dlaplace | \|e\| | 1/\|e\| |
| HAM, likelihood+ds | sqrt(\|e\|) | 1/(2\|e\|^{3/2}) |
| likelihood+dgnorm | \|e\|^beta | beta\|e\|^{beta-2} |
| likelihood+dlnorm (E='M') | log(1+e)^2-ish (exact form from logLikADAM) | derive from d(rho)/de |
| likelihood+dgamma (E='M') | (1+e) - log(1+e) scaled by shape | shape-scaled, derive |
| likelihood+dinvgauss (E='M') | (e^2/(1+e)) scaled | derive |

Notes:
- The exact rho for the E='M' distributions must be read off `logLikADAM` /
  the C++ likelihood code (`adamCore`/`scaler`), holding scale concentrated.
  Only rho' (up to a positive factor) matters for the solve — constants and
  scale factors drop out of argmin.
- Scale concentration: refresh the concentrated scale from the current
  residuals between IRLS sweeps using the existing `scaler()` formulas. For
  most rho above the scale cancels in argmin (location-scale families), so
  this only matters for dgamma/dinvgauss shape interplay — check and document
  per distribution.
- dgnorm beta: passed in as a loss parameter (estimated outside the solve).

## 2. Additive models: affine surrogate + IRLS (zero extra passes)

```
build (e0, J) once                      # existing analytic-design pass
delta = SSE solution                    # existing solve = iteration 0
repeat 3..10 times or until delta stabilises:
    e = e0 + J*delta                    # vector op, no pass
    w_t = rho'(e_t)/e_t                 # loss-specific, ~5 lines each
    delta = argmin sum w_t (e0 + J d)^2 # one weighted QR: olsCore(sqrt(w).*J, sqrt(w).*e0)
```

- Monotone (MM algorithm) for convex rho. Each sweep is O(obs*nFree^2) —
  microseconds at m=12.
- **L1-type losses at e=0** (MAE/HAM: w -> inf): per the no-clipping rule, do
  NOT use an epsilon floor. Correct MM treatment: observations with e_t == 0
  exactly are at their subgradient optimum — drop them from that sweep's
  weighted system (zero rows). Document; add a test with ties.
- Convergence guard: cap sweeps (deterministic, uniform), stop on relative
  delta change < 1e-8. Stateless as now.

## 3. Multiplicative/mixed models: weighted GN step (one-line-deep change)

The existing GN loop already computes (e, J) analytically each iteration. The
only change: the step solve becomes weighted —

```
step = -olsCore(sqrt(w) .* J, sqrt(w) .* e, 1e-7)   # w from rho' at current e
```

plus the line-search/LM acceptance criterion switches from SSE to
`sum(rho(e_candidate))` (same backtracking ladder, same LM damping on the
weighted system). Same pass count as today; loss-exact direction. This is
"Gauss-Newton for M-estimators" — the exact fix for the MAM/Gamma anomaly.

## 4. Multistep losses (additive first)

For additive ETS the h-step-ahead errors are also affine in the initials:
`e_{t+h|t} = y_{t+h} - w' F_prod(h) v_t` — linear composition. Implementation:

- Extend the analytic-design pass to also propagate the multistep measurement
  sensitivities: for each t, the h rows of the multistep design. Cost
  O(obs*h*nFree) — one bigger pass, still no probes. Reuse the recursion
  structure of `adamErrorer` (C++ multistep error code) for the value part.
- Then per loss:
  - **MSEh**: quadratic in delta on the h-th column — exact one-shot QR.
  - **TMSE / MSCE**: sums of quadratics — one-shot QR on the stacked design.
  - **GTMSE**: sum of log-quadratics — a few Newton steps on the affine
    surrogate (no passes).
  - **GPL**: log-det of the multistep error covariance — matrix Newton on the
    affine surrogate; start from the TMSE solution.
- Multiplicative multistep: out of scope for the first iteration of this work
  (falls back to the 1-step rho solve); revisit if demanded.

## 5. Interface

C++: `gradientSolve(..., lossType, lossParams)` where `lossType` is a char/enum
(`'S'` SSE (default, current behaviour), `'A'` MAE, `'H'` HAM, `'G'` gnorm,
`'g'` gamma, `'l'` lnorm, `'i'` invgauss, `'h'` MSEh, `'T'` TMSE, ...) and
`lossParams` an arma::vec (beta, shape, h as needed). Wrappers map the R/Python
`loss` + `distribution` pair to the enum in one place each
(`R/adam-gradient.R`, `python .../gradient.py`) — keep the mapping tables
mirror-identical, add a parity test that iterates over the full mapping.

**Custom user losses** (`loss=function(...)` in R / callable in Python): cannot
cross into C++ without callbacks (slow; callback semantics differ between
languages — parity risk). Behaviour: gradient falls back to backcasting with a
one-time message, documented in the `initial` docs. Same for `loss="custom"`
occurrence variants.

## 6. Files

| file | change |
|---|---|
| `src/headers/adamGradient.h` | add `rho(e, lossType, params)` + `irlsWeight(e, ...)` inline helpers |
| `src/headers/adamCore.h` | `gradientSolve` loss params; IRLS sweep (additive); weighted GN step + rho-based acceptance (nonlinear); multistep design builder |
| `src/adamGeneral.cpp` | module registration unchanged (same method, new args) |
| `src/python/adamPython.cpp` | pybind args for lossType/lossParams |
| `R/adam-gradient.R` | loss+distribution -> enum mapping; pass through |
| `python .../core/utils/gradient.py` | same mapping, mirror-identical |
| `tests/testthat/test_gradient.R` | per-loss tests: gradient loss <= backcasting loss on the same spec; MAE/HAM tie case |
| `python/tests/test_gradient.py` | same + mapping parity test |
| `NEWS`, `python/NEWS.md` | entries |

## 7. Validation

1. **Oracle per loss:** for each (loss, model) pair, compare the solved
   initials against a brute-force `nloptr`/`scipy` optimisation of the same
   loss over the initials at fixed persistence (small m to keep it feasible).
   Tolerance 1e-4 on the loss value.
2. **Monotonicity:** IRLS sweeps never increase the loss (assert in tests via
   the `analytic=FALSE`-style debug flag or repeated solves with capped
   sweeps).
3. **The MAM/Gamma case:** gradient's Gamma likelihood on AirPassengers must
   be >= backcasting's (it already is after `9b7da03b`; must not regress and
   should improve further with the exact rho).
4. **Multistep:** ETS(AAA) + MSEh — solved initials beat the current
   (1-step-profiled) initials on the MSEh criterion itself.
5. **Parity:** R vs Python fitted/logLik to machine precision across the loss
   grid (same C++, so this is a smoke test of the two mapping tables).
6. Full suites (R 461+, Python 725+) green; `ruff`/`mypy` clean.

## 8. Order of work

1. rho/weight helpers + SSE-default plumbing (no behaviour change, suites green).
2. Additive IRLS (MAE/HAM/gnorm + M-likelihoods) + tests + oracle validation.
3. Weighted GN step for multiplicative + the MAM/Gamma check.
4. Multistep additive designs (MSEh -> TMSE/MSCE -> GTMSE -> GPL).
5. Docs, NEWS, version bump per package convention.

Out of scope: custom-loss callbacks; multiplicative multistep; occurrence-model
losses (om/omg have their own cost functions — separate plan).
