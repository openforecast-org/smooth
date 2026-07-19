# df-fixes: principled degrees-of-freedom accounting across smooth

**Scope:** adam (ETS / ARIMA / regression / mixtures), CES, GUM, SSARIMA, SPARMA,
and the occurrence models om / omg — R side, Python side. No C++ changes.

**Goal:** replace the `length(B) + (loss=="likelihood") + nStatesBackcasting`
patchwork (plus the fractional `dfForBack` heuristic) with a **structural**,
**estimate-gated** degrees-of-freedom count that is **identical across
initialisation methods** and correct for **non-coprime multi-seasonal** models.

Everything below is grounded in diagnostics run 2026-07 against the current
package (probe = rank of the initial-state sensitivity matrix `X`, the theory
oracle from the backcasting paper).

---

## 0. Principle

df = the number of **identifiable estimated parameters**, computed from the
model **structure**, not from the optimiser's working vector `length(B)`.

Consequences:

- **Init-method invariance.** The same model has the same df whether the
  initials are optimised, backcast, gradient-solved or complete-backcast. Only
  `initial="provided"` (and per-component provided initials) removes df. This is
  the comparability requirement — it holds *by construction*, not by patching.
- **Estimate-gating.** Every term is gated by its own estimate flag. A user may
  `persistence=list(alpha=0.2)` and estimate β, γ: then α is *not* counted (it is
  provided), β and γ are. Same for φ, initials (per component / per seasonal
  block), ARMA, xreg, constant, distribution shape. Provided quantities go to
  the **Provided** row of the `nParam` table.
- **Scale is a real parameter.** Every continuous-response model reports a
  *concentrated* likelihood (Normal / Laplace / S / Gamma / … with the scale
  concentrated out), so the scale is an estimated parameter for **every loss**,
  not only `loss="likelihood"`. `df_scale = 1` for continuous engines,
  unconditionally. **Exception:** Bernoulli occurrence (om / omg) has **no
  scale** — `df_scale = 0` there (verified: `om$nParam[1,4] == 0`).

df is **not** a single universal formula: each engine has its own parameter
structure (ETS smoothing vs CES complex `a`/`b` vs GUM general matrices vs ARIMA
polynomials). What is shared is the *conventions* (gating, scale, table layout)
and *one* piece of new maths — the ETS seasonal identifiability correction.

---

## 1. The one new piece of maths: ETS seasonal identifiability (inclusion–exclusion)

Verified: for **additive/ETS** seasonal components the identifiable initial df is
below the naive `Σ(m_i − 1)` count when seasonal periods share common divisors,
because a pattern of period `gcd(m_i, m_j)` can move between blocks `i` and `j`
without changing fitted values. An `m`-periodic component spans frequencies
`{0, 1/m, …, (m−1)/m}`; two blocks share the `gcd(m_i, m_j)` frequencies. The
dimension of the sum of these subspaces is inclusion–exclusion over gcd.

Treating the level as a period-1 block, the identifiable **level + seasonal** df is

```
IE(L) = Σ_{∅≠S ⊆ L} (−1)^{|S|+1} · gcd(S)          # L = {1} ∪ {estimated seasonal lags}
```

and the ETS identifiable initial df is `IE(L) + modelIsTrendy·initialTrendEstimate`.

Verified exactly against the probe:

| model | rank(X) | IE formula |
|---|---|---|
| AAA (1,12) | 13 | 13 |
| AAA (1,4,8) gcd 4 | 9 | 9 |
| AAA (1,4,6) gcd 2 | 9 | 9 |
| AAA (1,3,7) coprime | 10 | 10 |
| AAA (1,2,4,8) | 9 | 9 |
| ANA (1,4,8) | 8 | 8 |

`gcd` = 2-line Euclid; k seasonal blocks → 2^k−1 subsets (k ≤ 3–4 in practice).

**This correction is ETS-only.** Verified that GUM(4,8), SSARIMA(1,4,8),
ADAM-ARIMA(1,4,8) have `rank(X) == naive optimal count` — no gcd drop. Their
seasonality is backshift-differencing / general-transition, not the sum-to-zero
periodic structure. So for every non-ETS engine the identifiable initial df is
simply the structural initial-state count the engine already computes for
`optimal`.

---

## 2. Per-engine parameter maps (verified against nParam tables)

Each engine fills its `parametersNumber` table (2×5:
rows Estimated/Provided; cols internal | xreg | occurrence | scale | all) from
these structural, estimate-gated counts. `nparam(object)` (reads `[1,5]`) is
**untouched**.

### 2.1 adam — ETS part
```
df_persist = persistenceLevelEstimate
           + modelIsTrendy   · persistenceTrendEstimate
           + sum(persistenceSeasonalEstimate)          # vector, per block
           + phiEstimate
df_init_ETS = IE({1 if initialLevelEstimate} ∪ {m_k : initialSeasonalEstimate[k]})
            + modelIsTrendy · initialTrendEstimate
```
Both gated. `df_init_ETS` uses the IE formula (fixes non-coprime for optimal AND
backcasting/gradient). The current `initial="gradient"` block (`adam.R:1550`) and
every `nStatesBackcasting` use the naive `Σ(m−1)` — replaced by `df_init_ETS`.

### 2.2 adam — ARIMA part
```
df_arma   = arEstimate·sum(arOrders) + maEstimate·sum(maOrders)
df_init_AR = <number of estimated ARIMA initial states>   # structural, no IE correction
```

### 2.3 adam — xreg part  (INCLUDES the adapt delta — was missing)
```
df_xreg = sum(xregParametersEstimated) · initialXregEstimate          # coefficients
        + max(xregParametersPersistence) · persistenceXregEstimate    # adapt "delta"
```
Verified: ETSX(ANN) `regressors="use"` → xreg cell 2; `regressors="adapt"` → 4
(2 coefficients + 2 delta persistence). The delta term MUST be counted. Goes in
`[1,2]` (nParamXreg), exactly as the current `adam.R:2333` does — keep it.

### 2.4 adam — constant / shape / scale
```
df_constant = constantEstimate                # -> internal cell
df_shape    = otherParameterEstimate          # dgnorm/dlgnorm β etc -> scale cell
df_scale    = 1                               # (b) always, continuous response
```

### 2.5 CES  (complex smoothing, own initial formula)
- Smoothing: `a = a0 + i·a1` (2 params if `seasonality != "simple"`, `a$estimate`)
  plus `b` per seasonal (`2·nSeasonal` for full/simple-complex, `nSeasonal` for
  partial; `b$estimate`). (`adam-ces.R:1052+`.)
- Initials (`adam-ces.R:1041`):
  `2·(seasonality≠"simple") + lagsModelMax·(seasonality≠"none") + lagsModelMax·(seasonality∈{full,simple})`,
  gated by `initialType` (0 when provided).
- xreg: `xregNumber + sum(persistenceXreg)` (`adam-ces.R:620`) — same coef+delta shape.
- **No ETS IE correction** (CES has no multi-seasonal sum-to-zero; single seasonal
  verified `rank == count`). Backcasting df = the same structural initial count.
- `df_scale = 1`.

### 2.6 GUM  (general linear SSOE)
- Estimates elements of `matF`, `matWt`, `vecG` + initials (all in `length(B)`
  via `filler()`), so the non-initial count is `length(B) − <initials>`; initials
  = the structural initial-state count.
- **No IE correction** (verified GUM(4,8) no gcd drop). Backcasting df = the same
  structural initial count.
- xreg `xregNumber + sum(persistenceXreg)`; `df_scale = 1`.

### 2.7 SSARIMA / SPARMA
- ARMA coefficients + structural initial-state count; no IE correction.
- SPARMA currently `parametersNumber[1,4] = (loss=="likelihood")` (`sparma.R:661`)
  — update to always 1.

### 2.8 om  (Bernoulli occurrence = ETS on the probability)
- Inherits the adam-ETS structure: `df_persist` + `df_init_ETS` (IE formula
  applies — occurrence can be seasonal, e.g. MNM), plus oETSX xreg (coef + delta).
- **`df_scale = 0`** (Bernoulli, verified `om$nParam[1,4]==0`).
- Backcasting currently `nParamEstimated = length(B_used)` → initials counted 0.
  Verified: om(MNM m=12) optimal = 14 (2 smoothing + 12 initials), backcasting = 2.
  Fix: add `df_init_ETS` under backcasting/complete/gradient. Goes in `[1,1]`
  (om puts everything in the internal cell; occurrence df for the *demand* side
  propagates via adam's `nParamOccurrence` column — check no double count).

### 2.9 omg  (two coupled occurrence models A + B)
- Two om-like ETS models (`nParamsA = length(B_A)`, `nParamsB = length(B_B)`).
  Apply the om treatment (2.8) to **each** side with its own lags/structure; sum.
- `df_scale = 0` for both.

---

## 3. The shared helper + wiring

- **`dfInitialsETS(seasonalLags, levelEstimated, trendy, seasonalEstimated, trendEstimated)`**
  — R (`R/helper.R`) + Python (`n_param.py`): the IE formula, gated. The ONLY
  identifiability maths. (Euclid `gcd` inline; R has no base gcd.)
- **`dfModel(...)`** per engine: sums the engine's structural terms into the
  `parametersNumber` table (estimated row from estimate flags, provided row from
  the complementary provided quantities), fills scale per §0, `[1,5]=sum`. One
  call replaces `nParamEstimated <- length(B) + (loss=="likelihood") + nStatesBackcasting`.
- The backcasting/complete/gradient initial df = the engine's structural initial
  count (ETS via `dfInitialsETS`; others via their existing optimal-initial
  formula) — the **same number optimal uses**, so ICs match across methods.
- Post-hoc reconstruction from a fitted object (replaces `dfDiscounter`'s role;
  used only by the test oracle).

---

## 4. Remove the fractional df machinery

After §3 is wired (so counts never silently drop to 0):

- Delete `calculateBackcastingDF()`, `dfDiscounter()`, `dfDiscounterFit()`
  (`R/helper.R`). Keep `componentsDefiner()` and the `*Checker()` helpers.
- Replace the `calculateBackcastingDF(...)` call sites in `adam-ces.R`,
  `adam-gum.R`, `adam-ssarima.R`, `sparma.R` with the engine `dfModel` path.
- Remove `dfForBack` plumbing: `utils-adam.R` (ellipsis extract + list element),
  `adamGeneral.R` pass-through, `globals.R` entry, and the commented `adam-sma.R`
  line. grep `dfForBack|calculateBackcastingDF|dfDiscounter` over `R/ src/ python/`
  → zero hits.

---

## 5. Python mirror

- `dfInitialsETS` in `n_param.py`; `NParam`/`n_param` fills the same cell layout.
- Wire into adam (`core/adam.py`, `estimator/`), CES, GUM, SSARIMA (their Python
  homes), and om/omg (`core/om.py`, `core/omg.py`) — mirroring §2. The
  `_adam_estimated["n_param_estimated"] += om.nparam` propagation stays; check no
  double count.
- `df_scale = 1` continuous / `0` occurrence, matching R.
- ruff / ruff format / mypy clean.

---

## 6. Testing (testthat + pytest)

1. **Init-method invariance:** per engine, `nparam(backcast) == nparam(optimal) ==
   nparam(gradient)` for the same model/persistence (single & coprime seasonal).
2. **Non-coprime frozen:** AAA(1,4,8)=… , (1,4,6)=… , (1,2,4,8)=… — optimal AND
   backcasting equal, and equal the IE value (probe as oracle).
3. **Estimate-gating:** `persistence=list(alpha=…)` estimating β,γ → α in Provided
   row, β,γ,initials in Estimated; totals correct. Same for provided initials,
   φ, constant, shape.
4. **xreg adapt:** ETSX `regressors="use"` vs `"adapt"` — the adapt case counts
   the delta persistence (xreg cell larger by the number of adaptive regressors).
5. **Scale:** every continuous model has `[1,4] ≥ 1` including non-likelihood
   losses (df up by 1 vs old); om/omg `[1,4] == 0`.
6. **om/omg:** om(MNM) backcast nparam == optimal nparam (initials now counted);
   omg A/B each gain their initials; no double count via `nParamOccurrence`.
7. **Formula == probe** across the ETS grid (probe test-only).
8. **No dangling refs** (§4 grep).
9. **Selection regression:** AirPassengers ETS pool, backcasting — ranking stable
   after the uniform df shift.

---

## 7. Rollout order

1. `dfInitialsETS` + `dfModel` helper (R), with formula==probe tests green.
2. Wire adam (ETS + ARIMA + xreg-adapt + mixtures); init-invariance + adapt +
   scale tests green.
3. Wire CES, GUM, SSARIMA, SPARMA; per-engine tests green.
4. Wire om, omg (scale=0, initials); occurrence tests green.
5. Remove fractional machinery (§4); dangling-ref grep clean.
6. Python mirror (§5) + its tests + ruff/mypy.
7. NEWS (both), man pages.

---

## 8. NEWS (both packages) — user-facing changes

- Backcasted / gradient / complete initial states now count towards the number of
  parameters exactly as optimised ones do (structural identifiable count), so
  information criteria are comparable across `initial=` methods. The experimental
  fractional df (`dfForBack`) is removed.
- Non-coprime multi-seasonal models (e.g. lags 4 & 8) now use the correct reduced
  seasonal df (`Σ m_i − Σ gcd + …`) for **all** init methods, including
  `initial="optimal"` — their ICs change accordingly.
- The distribution **scale** is now counted as a parameter for every loss (not
  only `loss="likelihood"`), since all reported log-likelihoods are concentrated
  likelihoods. Non-likelihood-loss models' df increases by 1 (AIC by 2).
  Occurrence models (om/omg, Bernoulli) are unaffected — they have no scale.
- ICs of `initial="backcasting"` (the default) models will increase relative to
  previous versions; ranking within one initialisation method is typically
  unaffected.

---

## 9. Out of scope

- The df of the smoothing parameters themselves (kappa-based corrections).
- Any change to `nparam()` accessor semantics (reads `nParam[1,5]`; untouched).
- Backcasting numerical defects (already fixed separately).
