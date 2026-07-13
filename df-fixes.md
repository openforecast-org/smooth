# df-fixes: remove fractional df, implement efficient df for backcasting

**Scope:** adam (ETS / ARIMA / regression / mixtures), CES, GUM, SSARIMA, SPARMA, and the
occurrence models om / omg / oes-family (which currently count backcast initials as zero)
— R side, Python side, and (optionally, for speed) the shared C++ core.
**Author of the plan:** based on the theory developed for the backcasting paper
(Svetunkov & Pritularga, CSDA revision, July 2026). Verified numerically against
`adam()` in `theory/` scripts of the paper project.

---

## 0. Background: why, and what the correct df is

Current state of the package:

- With `initial="optimal"` the initial states are counted in `nParam` (level + trend +
  (m-1) per seasonal block + smoothing + sigma).
- With `initial="backcasting"` the initials are counted as **zero** in `adam()`, and as a
  **fractional heuristic** (`calculateBackcastingDF()` / `dfDiscounter()`) in CES / GUM /
  SSARIMA / SPARMA behind the non-default `dfForBack=TRUE` ellipsis flag.
- Consequence: the same model fitted with the two initialisation methods gets different
  penalties (e.g. monthly ETS(A,A,A): 17 vs 4), so ICs across the methods are not
  comparable, and sigma^2 under backcasting is biased.

What the theory says (proven for pure additive SSOE, verified against `adam()`):

1. Backcasting is equivalent to the per-persistence optimal (profile) initialisation up
   to terms of order `rho(D)^(2T)`, where `D = F - g w'`. Its likelihood flattery
   ("degrees of freedom") is therefore the **same** as under optimisation.
2. That df equals the **number of identifiable initial directions**:
   `df_init = rank(X)`, where `X` is the `T x p` matrix with rows
   `X_t = w_t' * D_{t-1} * ... * D_1` — the sensitivity of the one-step fitted value to
   the initial state. For pure additive ETS this rank equals
   `p - (number of seasonal blocks)`, i.e. exactly the conventional
   `level + trend + (m-1) per seasonal block` count. For ARIMA / CES / GUM / xreg the
   rank handles redundancies automatically without per-model conventions.
3. The fractional, persistence-dependent "discounting" of the df is not supported by the
   theory: the total is an integer up to `O(rho(D)^{2T})` crumbs at all usable sample
   sizes. The fractional heuristic must go.

The fix, in one sentence: **backcasting counts its initials exactly like "optimal" does,
with the count computed generally as `rank(X)`, and the fractional df machinery is
removed.**

---

## 1. Inventory of what exists now (verified 2026-07-13)

Fractional df implementation (to REMOVE):

| Where | What |
|---|---|
| `R/helper.R` (top of file, ~lines 1–200) | `calculateBackcastingDF()`, `dfDiscounter()`, `dfDiscounterFit()` |
| `R/adam-ces.R` (~line 882–896) | `nStatesBackcasting` block calling `calculateBackcastingDF` |
| `R/adam-gum.R` (~line 729–754) | same pattern; note `parametersNumber[1,1]` also uses it |
| `R/adam-ssarima.R` (~line 982–998) | same pattern |
| `R/sparma.R` (~line 531–553) | same pattern |
| `R/adam-sma.R` (~line 243) | commented-out call — delete the comment too |
| `R/utils-adam.R` (~line 307, 328) | `dfForBack` ellipsis extraction and pass-through |
| `R/adamGeneral.R` (~line 3002) | `dfForBack = dfForBack` pass-through |
| `R/globals.R` (~line 9) | `"dfForBack"` in globalVariables |

Not exported: `dfDiscounter` et al. appear in no `NAMESPACE` entry, no `man/`, no
`tests/`, no vignettes (checked by grep). Removal is internal-only; still add a NEWS
line because `dfForBack` was reachable through `...`.

Correct-but-special-cased counting (to KEEP, and use as the template):

- `R/adam.R` (~line 1533–1548): the `initial="gradient"` block counts
  `initialLevelEstimate + trendy*initialTrendEstimate + sum(initialSeasonalEstimate*(lagsModelSeasonal-1))`.
  The comment there explicitly says backcasting keeps "the historical
  zero-initial-df accounting" — that historical accounting is what this plan replaces.

Python: no fractional df exists (nothing to remove). Parameter counting lives in
`python/src/smooth/adam_general/core/utils/n_param.py` (class `NParam`) and is consumed
in `core/adam.py` (`n_param` property, `_adam_estimated["n_param_estimated"]`) and
`core/estimator/estimator.py` / `optimization.py`.

Useful existing machinery (to REUSE — do not reinvent):

- `adamCore$fit(matrixVt, matrixWt, matrixF, vectorG, indexLookupTable, profilesRecent,
  vectorYt, vectorOt, backcast, nIterations, O)` in `src/headers/adamCore.h` — the
  filter. `dfDiscounterFit()` already demonstrates how to drive it from R with
  synthetic inputs; the new probe does the same thing with different inputs.
- `adamProfileCreator(lagsModelAll, lagsModelMax, obsInSample)` — builds
  `$lookup` (indexLookupTable) and `$recent` (profilesRecent template).
- `componentsDefiner()`, `etsChecker()`, `cesChecker()`, `gumChecker()`,
  `ssarimaChecker()` in `R/helper.R` — model-type introspection used by `dfDiscounter`;
  keep these (they are general helpers, only the three df functions go).

---

## 2. Part A — Remove the fractional df

Order matters: do removal AFTER Part B is implemented and wired, in the same commit or
branch, so that `nStatesBackcasting` never silently becomes 0 for CES/GUM/SSARIMA.

A1. In each of `R/adam-ces.R`, `R/adam-gum.R`, `R/adam-ssarima.R`, `R/sparma.R`:
    locate the block

    ```r
    nStatesBackcasting <- 0;
    ...
    nStatesBackcasting[] <- calculateBackcastingDF(profilesRecentTable, lagsModelAll, ...)
    ```

    and replace the `calculateBackcastingDF(...)` call with the new
    `dfInitialsBackcasting(...)` (Part B). Do NOT change the surrounding
    `nParamEstimated <- length(B) + (loss=="likelihood")*1 + nStatesBackcasting;` lines
    — the wiring stays, only the number changes. In `adam-gum.R` also check
    `parametersNumber[1,1] <- length(B) + nStatesBackcasting;` (~line 754) stays
    consistent.

A2. Refactor rather than delete-and-rewrite (see 3.3b for the reuse map):
    - `calculateBackcastingDF()`: KEEP the function and its signature; REPLACE the body
      with the rank probe of Part B and RENAME to `dfInitialsBackcasting()`. The
      signature already carries everything the probe needs, so engine call sites
      change only the function name (and drop the `dfForBack` argument).
    - `dfDiscounter(object)`: KEEP the setup half (introspection, `new(adamCore,...)`,
      `adamProfileCreator()`); REPLACE the discounting half with a call to
      `dfInitialsBackcasting()`; RENAME to `dfInitials(object)`. This becomes the
      post-hoc extractor used by the tests (T1--T3).
    - `dfDiscounterFit()`: DELETE, replacing with the parameterised probe runner
      inside `dfInitialsBackcasting()` (single in-sample forward pass, real `matWt`,
      real `vecG`, `y = 0` -- see 3.3).
    - MUST NOT survive the refactor: the doubled `2T` sample and mirrored
      `indexLookupTable`, the `1e-100` data trick, the all-ones `matWtBack`, and the
      (0,1)-region discounting/ratio logic -- all fractional-specific.
    Do not touch `componentsDefiner()` and the checker functions below them.

A3. Remove the `dfForBack` plumbing:
    - `R/utils-adam.R`: delete the `dfForBack <- if(is.null(ellipsis$dfForBack)) ...`
      line and the `dfForBack = dfForBack` element in the returned list.
    - `R/adamGeneral.R` (~line 3002): delete `dfForBack = dfForBack`.
    - `R/globals.R`: remove `"dfForBack"` from the globalVariables vector.
    - grep the whole `R/` tree for `dfForBack` afterwards; zero hits required.

A4. Delete the commented-out call in `R/adam-sma.R` (~line 243).

A5. NEWS: add an entry under the next version, e.g.
    "Backcasted initial states now count towards the number of estimated parameters
    exactly as optimised ones do (computed as the rank of the initial-state design).
    The experimental fractional df (`dfForBack`) is removed. Information criteria of
    models with `initial='backcasting'` will increase relative to previous versions;
    model *ranking* within one initialisation method is typically unaffected."

---

## 3. Part B — The efficient df: design

### 3.1 The quantity

`df_init = rank(X)`, `X[t, j] =` sensitivity of the one-step fitted value at time `t`
to initial-state slot `j`. "Slot" = one element of the initial profile: a state `i`
with lag `l_i` contributes `l_i` slots (`profilesRecent[i, 1:l_i]`). Total slots
`P = sum(lagsModelAll)`.

### 3.2 The probe: X by running the existing filter — NO new maths in C++

Key identity: run the filter on **zero data** with occurrence 1 and the *estimated*
persistence. Then the recursion collapses to `v_t = (F - g w_t') v_{t-1}` and the
fitted values returned by the filter are exactly `yfit_t = w_t' v_{t-1} = X[t, j]` when
the initial profile is the `j`-th basis vector. So:

```
for j in 1..P:
    profilesRecent <- all zeros; slot j <- 1
    run adamCore$fit( y = zeros(T), ot = ones(T), matrixWt = REAL matWt,
                      matrixF = REAL F, vectorG = REAL estimated g,
                      backcast = FALSE, nIterations = 1 )
    X[, j] <- fitted values from that run
df_init <- qr(X)$rank
```

This reuses the exact code path that already computes the model fit, so it is correct
by construction for every engine (ETS, ARIMA, CES, GUM, SSARIMA, SPARMA, xreg) and for
every lag structure. It is the same trick `dfDiscounterFit()` used, with three crucial
differences: real `matrixWt` (not ones), real `g` (not unit profile propagation), and
basis vectors in the profile (not all-ones).

### 3.3 Implementation: refactored function in `R/helper.R`

`dfInitialsBackcasting()` is `calculateBackcastingDF()` with a new body: keep its
existing argument list (drop `dfForBack`), which minimises the engine-side diffs.
Reference body (adjust argument names to the inherited signature):

```r
# Efficient number of initial-state parameters consumed by backcasting.
# Equals rank of the T x P sensitivity matrix of fitted values to the
# initial profile slots; computed by probing the C++ filter with basis
# initial profiles on zero data. See the backcasting paper for the theory.
dfInitialsBackcasting <- function(adamCpp, matWt, matF, vecG,
                                  indexLookupTable, profilesRecentTable,
                                  lagsModelAll, obsInSample){
    lagsModelMax <- max(lagsModelAll);
    nStates <- nrow(profilesRecentTable);
    slots <- cbind(rep(1:nStates, lagsModelAll),
                   unlist(lapply(lagsModelAll, seq_len)));
    P <- nrow(slots);
    # Zero data: the filter then propagates pure initial-state sensitivities
    yZero <- matrix(0, obsInSample, 1);
    otOnes <- matrix(1, obsInSample, 1);
    matVt <- matrix(0, nStates, obsInSample + lagsModelMax);
    X <- matrix(0, obsInSample, P);
    for(j in 1:P){
        profilesJ <- profilesRecentTable;
        profilesJ[] <- 0;
        profilesJ[slots[j,1], slots[j,2]] <- 1;
        fitJ <- adamCpp$fit(matVt, matWt, matF, vecG,
                            indexLookupTable, profilesJ,
                            yZero, otOnes, FALSE, 1, "n");
        X[, j] <- fitJ$yFitted;
    }
    return(qr(X)$rank);
}
```

Notes for the implementer (read carefully, these are the mistakes to avoid):

1. **`fitJ$yFitted`** — check the actual name of the fitted-values element in
   `FitResult` (see `src/headers/adamCore.h` and how `adam-ces.R` consumes
   `adamCpp$fit(...)`). Use whatever the engines use (`yfit` / `yFitted` / `$fitted`).
2. **Multiplicative components break on zero data.** For any model with a
   multiplicative E/T/S component, construct the probe `adamCore` object with the
   **additive twin**: same `lagsModelAll` and component counts, but characters
   `E='A'`, `T` mapped `M->A`, `Md->Ad`, `S` mapped `M->A`. This is the first-order
   (log-scale) approximation and is the documented behaviour. Do NOT feed `1e-100`
   data to a multiplicative filter and hope — division by near-zero states produces
   garbage ranks. The additive twin needs its own `new(adamCore, ...)` instantiation
   inside the helper when `any(c(Etype,Ttype,Stype) %in% c("M","Md"))`; the F, g, w
   matrices passed in stay the numeric ones from the fitted model.
3. **`matWt` must be the model's real measurement matrix** (`obsInSample` rows) — this
   is what makes xreg/ETSX and time-varying measurement correct. Do not pass ones.
4. **Occurrence:** probe with `ot = 1` always. The occurrence model's own parameters
   are counted by the oes machinery separately; the initials' identifiability is a
   property of the deterministic recursion.
5. **`indexLookupTable`** — pass the in-sample table
   (`adamProfileCreator(...)$lookup`, columns `1:(obsInSample+lagsModelMax)`), same as
   the engines use for fitting. No doubling, no mirroring: the probe is a single
   forward pass.
6. **Rank tolerance:** `qr()$rank` (LAPACK default) is acceptable. If flakiness is
   observed for near-unit-root ARIMA, switch to
   `sum(svd(X)$d > max(dim(X)) * .Machine$double.eps * svd(X)$d[1])`.
7. **Do not probe when initials are NOT backcast**: `initial="provided"` consumes 0 df
   (data-independent); `initial="optimal"`/`"two-stage"` are already counted via
   `length(B)`. The helper is called only under backcasting (and `"complete"`
   backcasting) — mirror the existing `if(initialType=="backcasting")`-style
   conditions where `calculateBackcastingDF` is called today.

### 3.3b Reuse map (what to take from the current helper.R)

| Existing code | Reuse | Detail |
|---|---|---|
| `calculateBackcastingDF()` signature | full | already receives `profilesRecentTable`, `lagsModelAll`, `vecG`, `matF`, `obsInSample`, `lagsModelMax`, `indexLookupTable`, `adamCpp` + ETS metadata; only the body changes |
| `dfDiscounter(object)` setup half | full | lags/components introspection, `new(adamCore, ...)` construction, `adamProfileCreator()` -- becomes the `dfInitials(object)` extractor skeleton |
| ETS seasonal `(m-1)/m` spreading loop | as inputs | its `componentsNumberETSSeasonal` / `lagsModelAll[...]` indexing is exactly what the 3.4 fast-path formula needs |
| `new(adamCore, lagsModelAll, Etype, Ttype, Stype, ...)` call in `dfDiscounter` | full | the additive-twin substitution of 3.3 note 2 happens at precisely this call: pass `'A'`/`'Ad'` chars for `M`/`Md` components |
| `dfDiscounterFit()` internals | shell only | the pattern of building `matVt`/`ot` and calling `adamCpp$fit` with synthetic inputs; every hardcoded input changes (see A2 must-not-survive list) |

### 3.4 Fast path (optional but recommended) — READ THE RESTRICTION

For **pure ETS without xreg** the rank is known analytically, but ONLY with at most
one seasonal block OR pairwise-coprime seasonal periods:

```r
lagsSeas <- lagsModelSeasonal;
coprime <- length(lagsSeas) <= 1 ||
           all(apply(combn(lagsSeas, 2), 2, function(x){ gcd(x[1], x[2]) == 1 }));
if(etsModel && !arimaModel && !xregModel && coprime){
    return(1 + modelIsTrendy + sum(lagsSeas - 1));
}
```

**Why the restriction (verified numerically, 2026-07-13):** for seasonal periods with
a common divisor, any pattern with period `gcd(m1, m2)` can be moved between the two
seasonal blocks without changing fitted values, so the identifiable count drops by
`gcd - 1` per pair beyond the usual per-block `-1`. Examples (slots / rank / naive
`m-1` convention): nested (4, 8): 13 / 8 / 11; (4, 6): 11 / 8 / 9; hourly-weekly
(24, 168): 193 / **168** / 191 — the naive convention overcounts the standard ADAM
double-seasonal model by 23 df (46 AIC units). Multi-seasonal models with non-coprime
periods MUST go through the probe (or a gcd-aware closed form, only after it is
proven and tested against the probe). This is precisely why the probe, not a lookup
table of conventions, is the primary mechanism.

The fast path covers the overwhelming majority of calls at zero cost and makes the
probe the fallback for ARIMA/CES/GUM/SSARIMA/xreg/multiplicative/multi-seasonal.
Keep an internal option (e.g. `options(smooth.dfProbe=TRUE)`) to force the probe for
testing the fast path against it (test T3). Note: R has no base `gcd`; implement the
two-liner Euclid inside the helper.

### 3.5 Cost budget

The probe runs the C++ filter `P` times, once, after estimation (never inside the
optimiser loop). For monthly ETS: 14 runs of an O(T) filter — microseconds to
milliseconds. For high-frequency multi-seasonal (P in the thousands), the probe is
O(T*P^2)-ish in total; the fast path avoids it for pure ETS, and for the rest emit a
once-per-session message if `P > 500` (and consider the C++ batch probe, Part E).
Requirement: default `adam()` fit time regression < 2% (measure in the test suite,
Part F, T6).

---

## 4. Part C — Wiring (R)

C1. `R/adam.R` (~line 1533): the main inconsistency fix. Where the comment currently
    says backcasting keeps "the historical zero-initial-df accounting", add the
    backcasting branch:

    ```r
    if(any(initialType==c("backcasting","complete")) ...){
        nStatesBackcasting[] <- dfInitialsBackcasting(...);
    }
    ```

    keeping the existing `initial="gradient"` branch as is. The objects needed
    (`adamCpp`, `matWt`, `matF`, `vecG`, `indexLookupTable`, `profilesRecentTable`,
    `lagsModelAll`, `obsInSample`) are all in scope at that point of `adam.R` — verify
    names against the local environment (they may be `matWt`/`matrixWt` etc.).

C2. `R/adam-ces.R`, `R/adam-gum.R`, `R/adam-ssarima.R`, `R/sparma.R`: replace the
    `calculateBackcastingDF` call as per A1. The call sites already have every needed
    argument (they currently pass more).

C3. Confirm downstream consumers need no change (they key off `nParam`/logLik df):
    - `R/methods.R` sigma (`df <- obs - nparam(object)`), logLik structure
      (`df=nparam(object)`), AIC/AICc/BIC via standard generics.
    - `parametersNumber` matrix rows in each engine — make sure the initials go into
      the same cell as under `initial="optimal"` (compare with the optimal branch in
      the same file; for `adam()` this is `parametersNumber[1,1]`).

C4. Documentation: `man-roxygen`/roxygen blocks for `adam`, `ces`, `gum`, `ssarima`,
    `auto.*` — wherever `initial` is documented, add one sentence: backcasted initials
    are counted in the number of parameters (rank of the initial-state design), making
    ICs comparable with `initial="optimal"`. Rebuild man pages.

C5. **Occurrence models: om() / omg() / oes-family (R/om.R, R/omg.R, R/om-oes.R,
    R/oesg.R, R/auto-om.R).** Status quo (verified): `nParamEstimated <-
    length(B_used)` (`R/om.R` ~line 727), so with `initial="backcasting"`/`"complete"`
    the occurrence initials cost ZERO — the plain form of the same inconsistency, no
    fractional heuristic involved. Since om defaults to `model="ZXZ"` (a full ETS for
    the occurrence probability), the uncounted df is up to `m+1` per occurrence model.
    Fixes:
    - `R/om.R` (~line 727): after `nParamEstimated <- length(B_used);` add, under
      `if(any(initialType == c("backcasting","complete")))`, the increment by
      `dfInitialsBackcasting(...)`. The needed objects are in the returned/ambient
      structures (`adamArchitect`, `adamCreated` carry the adamCore instance and
      matrices) — reuse them, do not rebuild.
    - Update the same-file consumers: `parNum[1,1] <- res$nParamEstimated` (~line 1003)
      and the IC environment `.icEnv$nP <- res$nParamEstimated` (~line 1125) pick the
      corrected value up automatically — verify, don't edit blindly.
    - `R/omg.R`: TWO underlying ADAM models (A and B; `jointResult$nParamsA` /
      `$nParamsB`, ~lines 1070–1095, each with its own `adamArchitectA/B`,
      `adamCreatedA/B`). Apply the increment to EACH side separately, using each
      side's own matrices and lag structure.
    - `R/oesg.R`, `R/om-oes.R`, `R/auto-om.R`: same pattern; locate
      `nParamEstimated`/`nParams` assignments and treat identically.
    - **Nonlinearity note:** the occurrence measurement passes through a link
      (logistic / odds-ratio), so the probe applies to the *underlying* recursion —
      exactly the additive-twin logic of 3.3 note 2. The occurrence's underlying ETS
      is often multiplicative (Z components may resolve to M): always route through
      the twin substitution.
    - **Propagation for free:** adam's `parametersNumber` has an `nParamOccurrence`
      column (`R/adam.R` ~line 600) and `n_param_estimated += om_model.nparam` on the
      Python side — once om counts correctly, intermittent-demand ADAM ICs become
      consistent with no further changes. Verify with test T8, do not double-count.

---

## 5. Part D — Python mirror

Python has no fractional df; only the ADD side applies.

D1. Port `dfInitialsBackcasting` to
    `python/src/smooth/adam_general/core/utils/n_param.py` (or a sibling
    `df_initials.py`): same probe loop against the pybind11 filter (see
    `src/python/adamPython.cpp` for the exposed fit entry point; mirror how
    `core/estimator/estimator.py` calls it). numpy: `np.linalg.matrix_rank(X)`.
D2. Wire into the `NParam` accounting where backcasting is the initialisation
    (search `initial` handling in `core/estimator/initial_values.py` and
    `core/adam.py` `n_param`), so that `model.n_param` matches R's `nparam()` for the
    same model/initialisation. Same for the occurrence models
    (`core/om.py`, `core/omg.py`, `core/auto_om.py`) mirroring C5; the
    `_adam_estimated["n_param_estimated"] += self._om_model.nparam` line in
    `core/adam.py` (~line 1166) then propagates it — check for double counting.
D3. Fast path identical to 3.4.
D4. Per repo rules: after editing, run from `python/`:
    `.venv/bin/ruff check src/ && .venv/bin/ruff format src/ && .venv/bin/mypy src/smooth`
    — all three must pass.

---

## 6. Part E — C++ core: nothing required, one optional addition

**No C++ change is required for correctness.** The probe reuses `adamCore::fit`.

Optional (only if profiling shows the R-loop overhead matters for large P): add a
batch method to `adamCore`:

```cpp
// Returns obs x P matrix of fitted values from basis initial profiles on zero data.
// P = sum(lagsModelAll). Reuses fit() internals with vectorYt = zeros.
arma::mat probeInitials(arma::mat const &matrixWt, arma::mat &matrixF,
                        arma::vec const &vectorG,
                        arma::umat const &indexLookupTable,
                        arma::mat const &profilesTemplate);
```

Loop over slots inside C++, filling columns; expose via the existing Rcpp module (see
how `fit` is exposed in `src/adamGeneral.cpp` / `RcppExports`) and pybind11
(`src/python/adamPython.cpp`). Keep the R/Python-side loop as the reference
implementation and add the batch path behind the same interface. Do not modify
`fit()` itself — zero risk to existing behaviour, zero slowdown of normal fitting.

---

## 7. Part F — Testing (testthat + python tests)

Create `tests/testthat/test-df-backcasting.R` (and a mirrored
`python/tests/test_df_backcasting.py`) with:

T1. **Parity of counts.** For each of: ETS(AAA) monthly, ETS(AAdN), ETSX(ANN)+2 xreg,
    ARIMA(2,1,2), CES("n"), CES("f"), GUM(orders=c(2),lags=c(1)), SSARIMA(1,1,1)(0,1,1)[12]:
    fit with `initial="optimal"` and `initial="backcasting"` with the SAME fixed
    persistence/parameters where the API allows; assert
    `nparam(fitBackcast) == nparam(fitOptimal)`. (Where optimal counts m-1 per
    seasonal block, the rank must reproduce it — this is the central invariant.)

T2. **Known ranks.** Assert the probe returns: 1 for ANN, 2 for AAN, `m+1` for AAA
    (13 monthly, 5 for m=4), 2 for CES(n), `sum(orders)` checks for GUM/SSARIMA
    computed once and frozen as regression values after manual verification.

T3. **Fast path == probe.** For pure additive ETS across a small grid, assert the
    analytic shortcut equals the probe result (force probe via the option).

T4. **The oracle statistical test** (the one that has already caught two real bugs):
    simulate ETS(A,N,N) and ETS(A,A,A) m=4 with known persistence via `sim.es`,
    fit with `persistence` fixed and `initial="backcasting"`, and check over N=500
    replications that `mean(sum(residuals^2) - sum(innovations^2))/sigma^2` is within
    MC error (±4*se) of `-rank(X)`. Skip on CRAN (`skip_on_cran()`), run in CI.
    This test fails loudly if the backcasting implementation or the df count is wrong.

T5. **Selection regression.** AirPassengers, pool of 30 ETS models, backcasting:
    the AIC ranking with the new count must equal the ranking obtained by adding
    `1 + trendy + (m-1)*seasonal` to each model's old k (frozen expected winner: MAM).

T6. **Performance guard.** `system.time` on `adam(AirPassengers, "MAM",
    initial="backcasting")` before/after: increase < 2%. And one large-P sanity:
    a double-seasonal model probe completes < 1s.

T7. **No dangling references.** `grep -r "dfForBack\|calculateBackcastingDF\|dfDiscounter" R/ src/ python/src man/` returns nothing.

T8. **Occurrence models.** (a) Parity: `oes`/`om` on a binary series with
    `initial="optimal"` vs `initial="backcasting"`, same fixed persistence — equal
    `nparam()`. (b) `omg`: the A- and B-side counts each increase by their own
    initials rank. (c) Intermittent end-to-end: `adam(y, occurrence=omModel)` totals
    include the occurrence initials exactly once (no double counting through
    `nParamOccurrence`). (d) Frozen value: om with monthly seasonal occurrence model
    counts m+1 more under backcasting than before the fix.

Follow the repo rule: no failing test is dismissable. If T4 fails for a model class,
that is a bug in that engine's backcasting (this has happened: AAA m=12 with trend,
and damped trend AAdN — see the paper project's `theory/issue-backcast-aaa12.md`);
report it, do not loosen the tolerance.

---

## 8. Rollout order

1. Part B helper (+ fast path) in R, with T2/T3 tests green.
2. Part C wiring, engine by engine: adam -> ces -> gum -> ssarima -> sparma; T1 green
   after each.
3. Part A removal; T7 green.
4. T4/T5/T6; NEWS; man pages.
5. Part D python mirror with its tests + ruff/mypy.
6. (Optional) Part E batch probe if profiling justifies it.

Do not reorder 2 and 3: removal before replacement silently zeroes the CES/GUM/SSARIMA
counts.

## 9. Out of scope (explicitly)

- The df of the smoothing parameters themselves (kappa-based corrections) — separate
  research line, not part of this fix.
- Fixing the backcasting defects the oracle test may expose (seasonal+trend m=12,
  damped trend) — separate issues; the df count is correct regardless.
- Occurrence-model parameter counting — unchanged.
