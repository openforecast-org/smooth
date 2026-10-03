# TBATS in smooth: implementation plan

Issue: openforecast-org/smooth#396. `tbats()` in R and `TBATS` in Python, built as a
native linear single-source-of-error model on the shared `adamCore` (the generic path,
`adamETS=FALSE`, as CES and GUM use it). The fitter itself needs no C++ changes.

Updated after openforecast-org/smooth#413 (per-lag backcasting turns) and the scale
convention of master (October 2026).

## A. The model

The whole model lives in the Box-Cox space `y⁽λ⁾`.

**Level and trend.** The ETS slot of `adamCore` with `E='A'`, `T∈{N,A}`, `S='N'`:
ETS(A,N,N), ETS(A,A,N) or ETS(A,Ad,N). Backcasting reuses the existing trend reversal.

**Harmonics.** For each period `m_i` (fractional allowed) and harmonic `j = 1..k_i`,
with `λ = 2πj/m_i`, two states with lags 1 and 2 in the `nArima` slot:

```
u_t     = v_{1,t-1} + v_{2,t-2}                 (contribution to y_t, w = (1, 1))
v_{i,t} = η_i u_t + g_i ε_t,  η = (2cos λ, −1)
g       = (γ₁ᵢ, sin λ·γ₂ᵢ − cos λ·γ₁ᵢ)
```

This is an exact reparameterisation of De Livera's rotation form
`s_t = cos λ·s_{t-1} + sin λ·s*_{t-1} + γ₁ε_t`, `s*_t = −sin λ·s_{t-1} + cos λ·s*_{t-1} + γ₂ε_t`:
both give `(1 − 2cos λ B + B²) u_t = (γ₁B + (sin λ·γ₂ − cos λ·γ₁)B²) ε_t`. Only γ₁ᵢ and
γ₂ᵢ are estimated per period, as in De Livera. The polynomial `1 − 2cos λ B + B²` is
palindromic, so the time-reversed recursion is the same recursion: the cells store
`η_i × value at time t`, which does not depend on the direction, and backcasting needs no
state reversal (the rotation form would need `s*` negated, which the lookup table cannot do).
Rules: `k_i < m_i/2`; a harmonic of a longer period whose frequency coincides with a
harmonic of a shorter one is dropped (the 7th harmonic of 168 is the 1st of 24).

**Where the harmonics live.** In the `nArima` slot of `adamCore`, which for `E='A'` is a
plain linear block: `w'v` in the measurement, `F v` in the transition, `g ε` in the
update (`adamGeneral.h`), with no ARIMA-specific logic in C++. Only the C++ slot is
shared. None of `adam()`'s ARIMA machinery is used for the harmonics: the polynomialiser
and the factor-by-factor stationarity bounds would reject `1 − 2cos λ B + B²` (its roots
are on the unit circle), and the companion-form ARIMA initials, the `phi` / `theta` names
and `arimaChecker()` (which keys on "ARIMA" in the model name) do not apply. The state
order in the slot is the harmonics first, then the ARMA states, so the ARMA initials stay
in the last state as in `adam()`. Checked on the current master: a noise-free harmonic of
period 7.3 in this slot, with lags (1, 2), is fitted with errors below 2e-13 under
backcasting.

The alternatives do not fit the framework:
- the rotation form with two lag-1 states, `F = [[cos λ, sin λ], [−sin λ, cos λ]]`: the
  backward pass applies the same `F`, but time reversal needs `F⁻¹ = F'`, i.e. `s*`
  negated at both turns, which would be a special case in C++;
- the ETS seasonal slot: one state per season, so integer periods only and `m` states
  (168, 365) instead of `2k`.

**The matrices.** With the ETS part `l, b`, harmonics `i = 1..K` (all periods and
harmonics numbered together) and an ADAM ARMA with `r = max(p, q)` states, the model is
`y⁽λ⁾_t = w' v_{t−l} + ε_t`, `v_t = F v_{t−l} + g ε_t`, where `v_{t−l}` takes each state
at its own lag:

```
v_t = (l_t, b_t, v¹_{1,t}, v¹_{2,t}, …, vᴷ_{1,t}, vᴷ_{2,t}, a_{1,t}, …, a_{r,t})'
l   = (1,   1,   1,        2,        …, 1,        2,        1,       …, r)'
w   = (1,   φ,   1,        1,        …, 1,        1,        1,       …, 1)'
F   = diag( [1 φ; 0 φ],  H₁, …, H_K,  A ),   H_i = η_i 1' = [2cos λ_i  2cos λ_i; −1  −1],
                                            A   = η_A 1',  η_A = (φ₁, …, φ_r)'
g   = (α, β,  γ₁₁, sin λ₁ γ₂₁ − cos λ₁ γ₁₁, …,  η_A + θ_A)'
```

with `η_A`, `θ_A` padded with zeros to length `r`; no trend drops `b` (and the first
block is `1`), no damping sets `φ = 1`. A harmonic adds
`u_{i,t} = v^i_{1,t−1} + v^i_{2,t−2}` to the measurement, and its rows of the transition
give `v^i_{j,t} = η_{ij} u_{i,t} + g_{ij} ε_t`, so
`(1 − 2cos λ_i B + B²) u_{i,t} = (g_{i1} B + g_{i2} B²) ε_t`: De Livera's harmonic. The
same block with `g = η + θ` is `adam()`'s ARMA(2,2) with `φ = (2cos λ, −1)`, which is what
the check above fits.

**ARMA.** ADAM's additive form (the ARIMA block of `adam()` with no differencing), not
TBATS's ARMA errors. Multiplicative seasonal ARMA is allowed through `lags` aligned with
`orders`. The ARMA lags are truncated to integers (365.25 → 365, 52.18 → 52); lags that
coincide after truncation are merged. The harmonics keep the fractional periods.

Why not TBATS's ARMA errors: for ANN + AR(1),
- ADAM: `y_t = l_{t-1} + a_{t-1} + ε_t`, `l_t = l_{t-1} + αε_t`, `a_t = φa_{t-1} + φε_t` gives
  `(1−φB)(1−B)y_t = (1 + (α−1)B − αφB²)ε_t`, impulse response `α + φ^j`;
- TBATS: `y_t = l_{t-1} + d_t`, `l_t = l_{t-1} + αd_t`, `d_t = φd_{t-1} + ε_t` gives
  `(1−φB)(1−B)y_t = (1 − (1−α)B)ε_t`, impulse response `φ^j + α(1−φ^j)/(1−φ)`.

Both have two parameters and neither nests the other. In TBATS the AR leaks into every
state, and the long-run response `α/(1−φ)` blows up as φ → 1; in ADAM each state responds to
the innovation only and the ARMA is a separate component, consistent with `adam()`.

**Box-Cox.** Inside `tbats()` only. λ is estimated in B together with the other parameters,
as in `alm()`: starting value λ₀ from the global model (section C), bounds [0, 1]. A numeric
`lambda` fixes it. λ is estimated only with `loss="likelihood"`; with other losses it has to
be provided (otherwise λ = 1 with a message). With `y ≤ 0` anywhere, λ = 1 with a warning.
The Jacobian `(λ−1)Σlog y` is added to the log-likelihood.

**Distributions.** `distribution = dnorm / dlaplace / ds / dgnorm`, in the Box-Cox space,
densities from greybox. The `dgnorm` shape is estimated in B unless provided.

## B. API

R:
```r
tbats(y, lags=c(1, frequency(y)), harmonics=NULL,
      trend=c("auto","none","additive","damped"),
      lambda=NULL, orders=list(ar=3, ma=3, select=TRUE),
      distribution=c("dnorm","dlaplace","ds","dgnorm"),
      loss=c("likelihood","MSE","MAE","HAM","MSEh","TMSE","GTMSE","MSCE","GPL"),
      ic=c("AICc","AIC","BIC","BICc"), h=0, holdout=FALSE,
      initial=c("backcasting","optimal","two-stage","complete"),
      bounds=c("usual","admissible","none"), silent=TRUE, model=NULL, ...)
```

- Harmonics are fitted for every lag above 1; `harmonics` is one integer per such lag.
- `orders` align with `lags` as in `adam()`; `ar=3` / `ma=3` refer to lag 1 only.
- `model=` refits a previous `tbats` object (needed for the Hessian / FI).

Python: `TBATS(lags=..., harmonics=None, trend="auto", lambda_bc=None, orders=...,
distribution="dnorm", loss="likelihood", ic="AICc", h=0, holdout=False,
initial="backcasting", bounds="usual", ...)` with `.fit(y)` / `.predict(h)`.

## C. The global model

OLS of `y⁽λ⁾` on an intercept, a time trend (if the model has one) and the Fourier terms of
the harmonics. The projection matrix is computed once, so re-evaluating it at a new λ is one
matrix-vector product. It provides:

1. λ₀, from its profile likelihood with the Jacobian (1-D search over [0, 1], R's
   `optimize()`, ported as Brent's `fmin`), rounded to 1e-8: the search finds it to
   about 1e-4, and the last bits of the least squares differ between R's LINPACK QR and
   NumPy's LAPACK one (3e-15 on AirPassengers), which was the only difference between
   the two fits;
2. the preselection of `k_i` by IC, one period at a time;
3. the initial states, recomputed at the current λ on every evaluation: level and trend;
   the pre-sample values `s₀`, `s₋₁` of each harmonic in its cells
   (`v_{1,0} = 2cos λ·s₀`, `v_{2,0} = −s₀`, `v_{2,−1} = −s₋₁`); ARMA states at zero
   (the ADAM ETS+ARIMA convention);
4. the residuals for the Hannan-Rissanen starting values of the ARMA parameters when the
   orders are given without selection (as `adam_arimaInitialiser` uses its decomposition
   residuals). Values the cost function would reject are moved inside the boundary
   (`hrFeasible`); non-invertible MA values are reflected (`hrReflect`).

## D. Initialisation

- `backcasting` (default): the first forward pass starts from the global-model states
  (section C). Each state crosses the turns of the backcast by its own lag (#413), so the
  lag-2 harmonic states make two zero-error steps and the lag-1 ones one, and the
  palindromic harmonic polynomial is reversed exactly; the trend is flipped as in ETS and
  there is no constant. With states in the `nArima` slot the head filter is always on
  (`headFlipIsExact` is false); it is cheap, as the default head is `lagsModelMax` (2, or
  the largest ARMA lag). The head of a backcasted fit is the recorded zero-error
  trajectory, so the initials keep their meaning for `two-stage`.
- `optimal`: only the identified initials: level, trend, 2 per harmonic (its Fourier
  coefficients (a, b), mapped to `s₀`, `s₋₁` and then to the cells, the third cell pinned),
  and for ARMA as many initials as the largest ARMA lag, held in the last ARMA state (the
  companion form of the ARIMA initialisation work). Notation: `s_t` is the harmonic's
  contribution to `y_t`, so the cells are `v_{1,0} = 2cos λ·s₀`, `v_{2,0} = −s₀`,
  `v_{2,−1} = −s₋₁` and `u_1 = 2cos λ·s₀ − s₋₁ = s₁`. `refineHeadFwd` walks only the level
  and trend across the head, so the harmonic cells are not touched. B holds deviations from the global
  model at the current λ, so λ and the level do not fight. The collector reads the
  identified initials back, so `initial` of a backcasted model reproduces its fit and
  two-stage hands it over without loss. The degrees of freedom count exactly these.
- `two-stage`, `complete`: as in `gum()`. `gradient` is not supported in the first version.

## E. Bounds

- `usual` (default):
  - α ∈ [0, 1], β ∈ [0, α], φ ∈ [0, 1];
  - the level-plus-seasonal response to a unit error at every horizon of the seasonal
    cycle stays in [0, 1]:
    `R(h) = α + Σ_i Σ_j [γ₁ᵢ cos((h−1)λ_ij) + γ₂ᵢ sin((h−1)λ_ij)] ∈ [0, 1]`,
    `h = 1..⌈max m_i⌉`. For conventional ETS(A,N,A) this gives `α ∈ [0, 1]`,
    `γ ∈ [−α, 1−α]`, i.e. ETS's usual region with γ allowed down to −α. The cos/sin matrix
    is precomputed; the check is a penalty, like β ≤ α;
  - ARMA stationarity / invertibility factor by factor (`arimaBounds.h`);
  - λ ∈ [0, 1], dgnorm shape > 0.
- `admissible`: eigenvalues of `F − g w'` for the ETS and harmonic block on the
  lag-expanded matrix, plus the factor-by-factor ARMA checks. `smoothEigens()` cannot be
  reused: it splits the states by unique lag (`eigenCalc.h`), which separates the two
  coupled states of a harmonic. Lag-expanded, a harmonic is
  `x_t = (v_{1,t}, v_{2,t}, v_{2,t−1})'` with
  `F̃ = [η₁ 0 η₁; η₂ 0 η₂; 0 1 0]`, `w̃ = (1, 0, 1)'`, `g̃ = (g₁, g₂, 0)'`, all lag 1.
- `none`.

## F. Selection (3 or 4 full fits by default)

1. Harmonics: from the global OLS model only (no full fits).
2. Trend: the requested candidates, fitted without ARMA, chosen by IC (`parallel` allowed).
3. ARMA (`orders$select=TRUE`; the orders are the maxima):
   - the residuals of the global model at the λ of the chosen model without ARMA. Not the
     errors of that model: its adaptive level turns an AR into an ARMA with a near-unit MA
     root, on which Hannan-Rissanen screens badly. On 17 series (12 R datasets and 5
     simulated), wherever both screens were fitted the global one was at least as good, and
     it found the gains the other missed (harmonics + AR(1) with φ=0.7: AR(1), AICc
     1470.0, against ARMA(1,2), 1476.9);
   - screen all candidate orders with Hannan-Rissanen and filter to get each candidate's
     innovations (C++, section G). Lags are screened one at a time, largest first, the
     others at the orders chosen so far, so the cost adds over the lags (about 5 ms);
   - approximate IC from the log-likelihood of the innovations under the chosen
     distribution (dgnorm shape from the no-ARMA fit), on a common sample (the first
     `min(sum(p_max s), T/4)` observations dropped). Everything but the `p + q` penalty is
     common to the candidates;
   - fit the winner (one fit), from the same cold start a direct call with those orders
     would use, and keep it only if its true IC beats the models without ARMA. Gating this
     fit on the approximate IC of the global model with the ARMA does not work: it is
     optimistic (AirPassengers: 1074.7 against 1091.2 without ARMA, while the fit gives
     1105.5).
4. All the ICs are stored in `object$ICs`.

## G. C++ (shared)

`arimaHRSelectCore` in `src/headers/arimaInitCore.h`: takes a series, the orders per lag,
the index of the lag to screen with its maximum AR and MA orders, and the bounds flag;
returns, for each candidate, the orders, the coefficients (a row padded with NaN, in the
order of B) and the innovations (via `hrFilterLevels`). `arimaHRCore` and it share
`arimaHRLevels`. Bindings: `arimaHRSelectCpp` (Rcpp, `src/olsWrap.cpp`)
and `_ols.arima_hr_select` (pybind, `src/python/olsWrap.cpp`), next to the existing
Hannan-Rissanen bindings. The log-likelihoods are computed on the R / Python side with
greybox densities.

## H. Forecasts and methods

- Scale, as master's convention (the ADAM monograph): the scaler returns `σ²` for
  `dnorm` and the scale `s` for `dlaplace`, `ds` and `dgnorm`; the likelihood takes the
  square root for `dnorm`. Everything that needs the variance goes through
  `adam_varianceDebiased()` / `adam_dfScale()`: the df are the observations minus every
  estimated parameter but the scale, so λ and the dgnorm shape count.
- Point forecast: inverse Box-Cox of the Box-Cox-scale point, which is the exact median
  (all four distributions are symmetric and the transform is monotone).
- Intervals: Box-Cox-scale quantiles from the linear variance (via `forecast.adam()` and
  the de-biased variance), inverse-transformed. Python hands its matrices to ADAM's
  `forecaster()`, as `CES.predict()` now does, rather than having its own forecast code.
- Fitted values are back-transformed; residuals stay on the Box-Cox scale.
- Class `c("adam","smooth")`, `smoothType()` "TBATS", `tbatsChecker()`, added wherever
  `cesChecker()` / `gumChecker()` already branch (`coefbootstrap`, the OPG `vcov`,
  `reapply`, `print`, `plot`). Printed name `TBATS(λ, {p,q}, φ, <m₁,k₁>, …)`, without
  "ARIMA", so `arimaChecker()` stays false. `pointLik()` / Python `point_lik()` as CES has
  it, for the OPG covariance.
- `vcov`: `covarOPGtbats` (OPG) and the Hessian via a refit with `model=object`;
  `confint` and `summary` through the existing adam methods, with λ and shape listed.
- The components plot sums the harmonics of each period into one seasonal series.
- Python mirrors all of it: `predict`, `summary`, `vcov`, `confint`, `plot`.

## I. Files

- R: `R/adam-tbats.R` (new: `tbats()`, architector, creator, filler, initialiser, cost
  function); `R/helper.R` (`tbatsChecker`, `covarOPGtbats`); `vcov.adam` dispatch; small
  TBATS branches in `forecast` / `print` / `plot`; NAMESPACE, roxygen, `man/tbats.Rd`, NEWS.
- Python: `core/tbats/` (`architector.py`, `creator.py`, `filler.py`, `initialiser.py`,
  `cost_function.py`), `core/tbats_model.py`, export in `__init__.py`, `python/NEWS.md`.
- C++: `arimaInitCore.h`, `src/olsWrap.cpp`, `src/python/olsWrap.cpp`, `RcppExports`.

## J. Tests

R (`tests/testthat/test_tbats.R`):
- the likelihood at fixed parameters equals an independent De Livera rotation-form recursion;
- a noise-free harmonic with a fractional period is reproduced with zero errors under
  backcasting; a series generated from the global model is reproduced from the seed;
- ANN + AR(1) equals the ARIMA(1,1,2) recursion;
- λ = 0 fixed equals the log model; estimated λ within [0, 1]; the λ = 1 fall-backs;
- all four distributions; dgnorm shape estimated;
- selection: harmonics + AR(1) data selects AR(1); the fall-back to no ARMA;
- ARMA lag truncation and merging; lossless two-stage; `R(h)` penalties;
- `vcov` / `confint` / `summary` finite; intervals ordered and non-negative.

Python: `test_tbats.py` mirroring the above; `test_tbats_r_parity.py` with the likelihood at
identical parameters, the same Hannan-Rissanen screen, and the same B, log-likelihood and
forecasts on AirPassengers and the two-period `taylor` data.

## K. Order of work (branch `tbats`)

1. Prototype checks of the assumptions behind "no fitter changes": the generic path with the
   ETS level / trend and the harmonics / ARMA together in the `nArima` slot (the harmonic
   alone is checked, see section A); `refineHeadFwd` with `lagsModelMax = 2` under
   `initial="optimal"`; the linear forecast variance against a simulation. Any failure is
   discussed before C++ changes.
   **Done, no C++ changes needed.** Damped trend, harmonics of 7 (k=2) and 30.4375 (k=3)
   and an ADAM ARMA(1,1), built as in section A and run through `adamCore`:
   - at fixed parameters and initials the errors equal the innovations of De Livera's
     rotation form to 1e-13; `refineHeadFwd` walks only the level and trend (one step,
     H = 2) and leaves the harmonic cells as given;
   - a noise-free series with an undamped trend is reproduced by backcasting from a crude
     seed: max error 0.11 at 2 iterations, 1.3e-3 at 3, 2.7e-7 at 5, 1.4e-13 at 10. With
     a damped trend it stalls at 0.07 after the first 30 observations: the trend flip is
     exact only for φ = 1 (ETS(A,Ad,N) alone stalls at 0.006), and the slowly forgetting
     harmonics carry the junction error further. On the noisy series at the true
     parameters the loss settles by 3 iterations (SSE 1623.06 at 2, 1622.43 from 3);
   - `adamCore$forecast` equals the rotation form to 3.4e-13 over 70 steps; `covarAnal()`
     equals the variance from the rotation form's impulse responses to 7e-16, and a
     Monte Carlo of 20000 paths agrees within 1.5%.
2. R core: fixed-structure fit, global model, initialisation, Box-Cox, distributions, bounds.
   **Done** (admissible bounds by default).
3. R forecasting and methods. **Done**: the adam methods on the model in the Box-Cox space;
   `reapply` / `reforecast` per draw with its own λ; `confint` pulled inside the bounds
   along each parameter; `coefbootstrap` refitting the structure.
4. C++ `arimaHRSelectCore` (both bindings in one commit) and the R selection. **Done**.
5. Python port and parity tests (`ruff check`, `ruff format`, `mypy` after every edit).
   **Done**: the fits, forecasts, covariance and confidence intervals agree with R
   (`tests/test_tbats_r_parity.py`).
6. NEWS, docs, `R CMD check`, full testthat and pytest suites with zero failures.

Later phases:
- `xreg` with `regressors` use / select / adapt / integrate, after the Python port;
- `occurrence` accepting a provided or estimated `om` / `omg` model, with the Box-Cox
  transform and its Jacobian on the non-zero observations only.
