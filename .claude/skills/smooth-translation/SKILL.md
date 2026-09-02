---
name: smooth-translation
description: Port a feature from the R smooth package to the Python port, or check how an R name maps to Python. Covers the R↔Python name map for user arguments, fitted attributes, state-space matrices and internal dicts; the R and Python call flows side by side; the parity rules that govern the port (same optimiser, same initialisation, distributions from greybox, no clipping); and the checklist for landing a translation with a test that proves it. Use when translating an R function or argument to Python, when a Python result disagrees with R, or when looking up what an R name is called on the Python side.
---

# R → Python translation

The Python port is a **numerical** port of the R package. Same C++ kernel, same
optimiser, same initialisation: on the same data the two are expected to return
the same bits, not merely the same answer to a few decimals.

Source of truth is always the code — `R/` and `python/src/smooth/` — never this
file. Read the R implementation before writing the Python one.

## Where things live

| Layer | R | Python |
|---|---|---|
| User entry points | `R/adam.R`, `R/adam-es.R`, `R/adam-ces.R`, `R/adam-msarima.R`, `R/adam-sma.R`, `R/om.R`, `R/omg.R`, `R/om-oes.R`, `R/sm.R`, `R/msdecompose.R` | `smooth.ADAM`, `ES`, `CES`/`AutoCES`, `MSARIMA`/`AutoMSARIMA`, `SMA`, `OM`/`OMG`/`AutoOM`, `sm`, `msdecompose` |
| Validation | `parametersChecker()` in `R/adamGeneral.R` | `core/checker/` (package) |
| Architecture / matrices | `architector()`, `creator()`, `filler()`, `initialiser()` — locals in `R/adam.R` | `core/creator/` (package) |
| Estimation / selection | `estimator()`, `selector()` — locals in `R/adam.R` | `core/estimator/` (package) |
| Forecasting | `preparator()`, `forecaster()` — locals in `R/adam.R` | `core/forecaster/` (package) |
| Cost functions | `CF()`, `logLikADAM()` — locals in `R/adam.R` | `core/utils/cost_functions.py` |
| Information criteria | `ICFunction()` | `core/utils/ic.py` |
| Covariance | `covarAnal()`, `adamVarAnal()` | `core/utils/var_covar.py` |
| Shared C++ | see the `smooth-cpp-shared` skill | — |

`checker`, `creator`, `estimator` and `forecaster` are **packages** (directories)
on the Python side, not single modules.

### Call flow

```
R    adam() → parametersChecker → architector → creator → initialiser
              → CF (filler + adamCpp$fit) → forecaster

Py   ADAM.fit() → parameters_checker → architector → creator → estimator
                  → initialiser → CF (filler + adam_fitter) → ...
     ADAM.predict() → preparator → forecaster → adam_forecaster
```

## Name map

### User arguments

| R | Python |
|---|---|
| `model`, `lags`, `phi`, `persistence`, `initial`, `distribution`, `loss`, `ic`, `bounds`, `h`, `holdout`, `regressors` | same names |
| `orders = list(ar=, i=, ma=)` | `orders={"ar": …}` **or** the scalar trio `ar_order`, `i_order`, `ma_order` |
| `xreg` / `formula` | `X` (positional, on `fit`) |
| `initialSeason`, `initialX` | folded into `initial`, which takes a dict of state values as well as a method name (`MSARIMA` also has `initial_X`) |
| `lambda` (LASSO/RIDGE weight) | `lambda_param`, or `**{"lambda": …}`. `OM` / `OMG` use `reg_lambda` |
| `silent` | `verbose` (inverted) |
| `sm(object, …)` + `implant(model, scale)` | `model.sm(…)` then `model.scale_model = scale`; `= None` detaches |

### Fitted attributes

R exposes `model$x`; Python uses properties, with a trailing underscore only
where the plain name would clash with a constructor argument.

| R | Python |
|---|---|
| `coef(m)` | `m.coef` |
| `logLik(m)` | `m.loglik` |
| `nparam(m)` | `m.nparam` (`m.n_param` on `CES`) |
| `nobs(m)` | `m.nobs` |
| `AIC` / `AICc` / `BIC` / `BICc` | `m.aic` / `m.aicc` / `m.bic` / `m.bicc` |
| `fitted(m)`, `residuals(m)`, `actuals(m)` | `m.fitted`, `m.residuals`, `m.actuals` |
| `m$states` | `m.states` |
| `m$persistence` | `m.persistence_vector` |
| `m$phi` | `m.phi_` |
| `m$initial` | `m.initial_value` |
| `m$scale` (number **or** scale model) | `m.scale` (always a float) and `m.scale_model` (model or `None`) |
| `m$lossValue` | `m.loss_value` |
| `m$loss`, `m$distribution` | `m.loss_`, `m.distribution_` |
| `m$model` | `m.model_name` |
| `m$timeElapsed` | `m.time_elapsed` |
| `sigma(m)` | `m.sigma` |
| `extractScale(m)`, `extractSigma(m)` | `m.extract_scale()`, `m.extract_sigma()` |
| `pointLik(m)` | `m.point_lik()` |
| `forecast(m, h=)` | `m.predict(h=)` → `ForecastResult` with `.mean` / `.lower` / `.upper` |

R stores either a number or a model in the single `$scale` slot and
disambiguates with `is.scale()`. Python keeps the two apart so the return type
is stable.

### State-space matrices and internal structures

| R | Python |
|---|---|
| `matVt` | `mat_vt` |
| `matF` | `mat_f` |
| `matWt` | `mat_wt` |
| `vecg` | `vec_g` |
| `matxt` | `mat_xt` |
| `profilesRecentTable` | `profiles_recent_table` |
| `indexLookupTable` | `index_lookup_table` |
| `lagsModel`, `lagsModelAll` | `lags_dict` |
| `yInSample`, `yHoldout` | `observations_dict` |
| `otLogical` | `observations_dict["ot_logical"]` |
| `Etype`, `Ttype`, `Stype` | `model_type_dict` |
| `initialType`, `initialValue` | `initials_dict` |

R passes state through the calling environment; Python passes explicit dicts.
Matrices must stay Fortran (column-major) order for Armadillo.

### Coverage

Do not keep a coverage table here — it rots. The maintained answers are:

- [Roadmap](https://github.com/openforecast-org/smooth/wiki/Roadmap) — what is R-only (`ssarima()`, `gum()`), partial, or not planned.
- [R vs Python differences](https://github.com/openforecast-org/smooth/wiki/R-Python-differences) — measured numerical status, with the scripts that reproduce it.

## Rules the port is held to

1. **A difference in results is never "optimiser noise".** Both languages run
   NLopt on the same loss from the same initialisation, so a materially
   different optimum is impossible as an optimiser artefact. A real gap means
   either the initialisation differs (seed, `B0`, bounds, profile tables) or the
   two are not fitting the same model (state structure, lags, matrices). The
   diagnostic that settles it in one step: evaluate the likelihood in both
   languages at *identical* parameter values. Equal there means the seed
   differs; different there means the models differ, and the next move is to
   print `lagsModelAll`, the persistence vector and the component counts side by
   side.
2. **Distributions come from greybox, never re-derived.** `greybox` supplies
   `dnorm`/`dlaplace`/`ds`/`dgnorm`/`dalaplace`/`dlnorm`/`dllaplace`/`dls`/
   `dlgnorm`/`dinvgauss`/`dgamma` and their `p`/`q` counterparts — the same
   functions R's `smooth` calls. Spell out the parameterisation, not the
   density. The `r*` draws keep `scipy`, because greybox's take no
   `random_state`.
3. **Never clip, clamp or floor bad numerics.** No `np.clip` on fitted values,
   no `np.maximum(x, 1e-15)` inside `log()`. A `-Inf` log-likelihood is the
   optimiser reporting that those parameters are inconsistent with the data.
   The one legitimate exception is an infeasibility guard at the *top* of a cost
   function returning a uniformly large penalty.
4. **Match R's error behaviour, not just its results.** R stops in
   `match.arg()` on a bad argument; Python should raise, not warn and
   substitute a default.
5. **Summation order matters.** R's `sum()` and `mean()` accumulate in a long
   double register. Where a 1-ulp difference can reorder the optimiser, use the
   `_sum_r` / `_mean_r` helpers in `core/utils/`, not `np.sum`.
6. **No `frequency` parameter.** Seasonal period is inferred from the data,
   `lags`, or the model spec. The standalone `sim_*` generators are the
   documented exception.

## Landing a translation

1. Read the R implementation end to end — arguments, defaults, what it calls,
   what it returns.
2. Find where it belongs on the Python side (checker / creator / estimator /
   forecaster / utils) and whether a partial version already exists.
3. Mirror the algorithm. Match argument names per the map above; flag any
   parameter that exists on one side only rather than inventing one.
4. **Prove it against R.** Generate the data in one language and read it in the
   other — never rely on a shared seed, the RNGs differ. Add the test to
   `python/tests/` with the `r_parity` marker if it shells out to R; those run
   through `tests/_r_bridge.py`, which loads the local R source with
   `devtools::load_all()` so it always compares against the working tree.
5. Run `ruff check src/`, `ruff format src/` and `mypy src/smooth` — all three,
   every time.
6. Add a `python/NEWS.md` entry under the current unreleased version.
7. If the behaviour is user-visible, update the wiki: the function page, plus
   `Roadmap` or `R-Python-differences` if the parity status moved.
