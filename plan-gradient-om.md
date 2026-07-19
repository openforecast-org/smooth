# Plan — initial="gradient" for the single-model occurrence (om)

**Scope:** make `initial="gradient"` in `om()` (R) / `OM` (Python) solve the
initial states of the occurrence ETS model by profiling the occurrence loss,
via the shared C++ `adamCore::gradientSolve`, with exact cross-language parity.
The two-model `omg()` is out of scope (separate discussion; needs a coupled
solve). Builds on the arbitrary-loss machinery of `c9a09fc6`.

---

## 0. Background: the current state is a silent no-op

`om()` accepts `initial="gradient"` but its cost function and refit gate the
backcast flag on `c("complete","backcasting")` only (`R/om.R:1764`, `:872`;
mirrored in `python .../utils/om_cost.py`, `core/om.py`). The common checker
groups gradient with backcasting (initials NOT in B), so a gradient fit today
neither backcasts nor solves — the initials stay at the creator seed. Measured:
odds-ratio MNN on a 120-obs binary series gives logLik −88.23 (gradient) vs
−80.26 (backcasting) vs −75.00 (optimal). Two deliverables:

1. **Correctness floor:** gradient must at minimum fall back to backcasting
   (add "gradient" to the gates) — same convention as CES/SSARIMA.
2. **The real solve:** profile the occurrence loss over the initials at each
   evaluation, aiming at optimal-quality fits at backcasting-style cost.

## 1. The reduction: occurrence losses are separable in the probability residual

The om cost is a function of the fitted probability `p_t = link(yhat_t)`
(`omLinkFunction`, per occurrence type and error type):

| occurrence | E='M' | E='A' |
|---|---|---|
| direct ('d') | p = clamp(yhat, 0, 1) | p = clamp(yhat, 0, 1) |
| odds-ratio ('o') | p = yhat/(1+yhat) | p = e^yhat/(1+e^yhat) |
| inverse-odds ('i') | p = 1/(1+yhat) | p = 1/(1+e^yhat) |

Define the probability-scale residual `r_t = o_t − p_t` (exactly the `errors`
the om CF uses). Then, because `o_t ∈ {0,1}`:

- **Bernoulli likelihood**: −[o log p + (1−o) log(1−p)] = **−log(1−|r|)** —
  separable, convex, smooth away from |r|=1. psi = sign(r)/(1−|r|),
  psi' = 1/(1−|r|)². New loss code **'B'**.
- **MSE / MAE / HAM** on `r`: the existing 'S'/'A'/'H' rho machinery applies
  to `r` verbatim.
- LASSO/RIDGE → 'S' (their error term is RMSE of r); custom → backcasting
  fallback (same rule and message as adam).

So no occurrence-specific loss mathematics beyond one new rho — the work is in
the **residual channel and its Jacobian**.

## 2. C++: an occurrence channel for the gradient solve

`gradientSolve`, `gradientPass`, `gradientPassJacobian` gain the occurrence
char `O` ('n' = demand path, unchanged; 'd'/'o'/'i' = occurrence):

- **Forward pass** (occurrence mode): mirror `fit()`'s O-path exactly —
  `yhat = adamWvalue(...)`, `e_t = errorf(o_t, yhat, E, o_t, O)`
  (= `occurrenceError`), state update via `adamFvalue + adamGvalue` with that
  `e_t`. The pass ALSO produces `p_t` and `r_t = o_t − p_t` (the solve's
  residual vector — replaces the errorf residuals of the demand path).
- **Jacobian pass**: existing `adamWvalueJac`/`adamFvalueJac`/`adamGvalueJac`
  are reused untouched (the occurrence model is pure ETS). Two new inline
  helpers in `adamGradient.h`:
  - `occurrenceLinkJac(yhat, E, O)` → (p, dp/dyhat). Clamp of 'd' at p∈{0,1}
    has derivative 0 on the flat spots (same convention as the mixed clamp).
  - `occurrenceErrorJac(o, yhat, E, O)` → de/dyhat for the state-update
    channel, differentiating `occurrenceError` branch-by-branch (chain through
    u = (1+o−p)/2, the odds/log transforms, and exp(yhat) for E='A').
  Sensitivity recursion: `dr/dθ = −(dp/dyhat)·dyhat/dθ` (the design row) and
  `de/dθ = (de/dyhat)·dyhat/dθ` feeding `jacGe` in the state update.
- **Solve**: occurrence always uses the Gauss-Newton branch (the link makes
  the residuals nonlinear even for E='A'; no affine shortcut). Loss-aware
  step rows via the existing `gradientLossStepRow` for 'S'/'A'/'H', plus the
  'B' Newton rows (a = sqrt(psi'), b = psi/a — always defined since
  psi' = 1/(1−|r|)² > 0). Line search + LM acceptance on the rho sum, as now.
- **Failure = fallback**: non-finite residuals/Jacobian (e.g. p saturated at
  0/1 on degenerate stretches) return the empty matrix → the caller backcasts.
  No epsilon floors anywhere (kappa=1e-10 lives in `occurrenceError` already
  and is mirrored, not added to).
- Signature: `gradientSolve(..., lossType, lossParams, O)`. Multistep codes
  never reach it from om (mapper never emits them).

## 3. R wiring

- `R/adam-gradient.R`:
  - `adam_gradientOmLossCode(loss)` — tiny mapper: likelihood→'B', MSE→'S',
    MAE→'A', HAM→'H', LASSO/RIDGE→'S', custom→NULL. Mirror-identical to the
    Python one.
  - `adam_fitOrGradient` gains `oType="n"` and, for om calls, takes the
    ready-made lossCode; passes `oType` through to both the gradient fit
    (currently hard-coded `"n"`) and the fallback fit; gradient joins the
    backcast group in the fallback (this alone fixes the no-op bug).
- `R/om.R`: the two fit sites (`omCF_local:1760`, refit at `:872`) route
  through the dispatcher with `oType=occurrenceChar` and the om loss code;
  occurrence `"fixed"` never reaches the solve (no estimated initials);
  ARIMA/xreg om specs fall back automatically (probe basis returns NULL).
  `changeOrigin` (`:1904`) and the model-reuse gate (`:115`) add "gradient"
  to their backcasting groups where semantics require it.

## 4. Python wiring

Mirror exactly: `adam_gradient_om_loss_code` in `core/utils/gradient.py`;
`adam_fit_or_gradient` gains the same `o_type` pass-through and pre-computed
loss-code path; `utils/om_cost.py` and the om preparator in `core/om.py` route
through the dispatcher; backcast groups gain "gradient".

## 5. Validation

1. **No-op bug fixed:** om gradient logLik ≥ om backcasting logLik on the
   measured case (−88.23 → ≤ −80.26), and close to optimal (−75.00).
2. **Oracle:** per (occurrence type × E type × loss), brute-force `nloptr`
   optimisation of the same loss over the initials at fixed persistence;
   gradient loss ≤ oracle + 1e-4.
3. **Analytic vs FD:** the occurrence Jacobian pass against finite
   differences on a fixed fixture (the `analytic=FALSE` path stays as the
   oracle in occurrence mode too).
4. **Parity:** R vs Python over occurrence {direct, odds-ratio,
   inverse-odds-ratio} × E {A, M} × loss {likelihood, MSE, MAE}, fitted and
   loss values to machine precision; direct C++ fixture calls bitwise.
5. **Degenerate series:** near-constant ot (long all-one runs) must either
   solve finitely or fall back — never NaN.
6. Full suites (R 485+, Python 730+) green; ruff/mypy clean; NEWS both sides.

## 6. Files

| file | change |
|---|---|
| `src/headers/adamGradient.h` | `occurrenceLinkJac`, `occurrenceErrorJac`, 'B' rho in the loss helpers |
| `src/headers/adamCore.h` | O channel in `gradientSolve`/`gradientPass`/`gradientPassJacobian` |
| `src/adamGeneral.cpp`, `src/python/adamPython.cpp` | binding arg |
| `R/adam-gradient.R` | om loss mapper; `oType` in the dispatcher |
| `R/om.R` | route both fit sites through the dispatcher; gate fixes |
| `python .../utils/gradient.py` | mirrored mapper + `o_type` |
| `python .../utils/om_cost.py`, `core/om.py` | route fit sites; gate fixes |
| `tests/testthat/test_om.R` (or test_gradient.R), `python/tests/test_gradient.py` | per-type tests, no-op regression test |
| `NEWS`, `python/NEWS.md` | entries |

## 7. Order of work

1. Gate fixes (gradient → backcasting fallback in om, both languages): the
   correctness floor, tested first.
2. C++ occurrence channel + 'B' rho + FD-vs-analytic check.
3. R + Python wiring through the dispatcher; parity grid.
4. Oracle validation, degenerate-series tests, suites, NEWS.

Out of scope: `omg()` (coupled two-model solve — to be discussed), om with
ARIMA/xreg (falls back), occurrence "fixed" (no estimated initials), custom
losses (fall back with the standard message).
