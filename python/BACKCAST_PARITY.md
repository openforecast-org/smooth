# Backcasting head: R / Python parity record

Record of the parity sweep run when the Python port was brought in line with the
R-side backcasting changes on `backcast-warmstart` (`279de4e1`, `1e98f7df`,
`785405d8`, `dd684eb0`, `d198b282`).

## What the R side changed, and where Python mirrors it

| R change | Python |
|---|---|
| `adamCore.h`: the head is filtered over the model's own backcasts | Arrives on rebuild — the Python extension compiles the same header. `headLength` is now exposed on the pybind11 binding (`src/python/adamPython.cpp`); it was Rcpp-only, so the kernel had silently run with `headLength = 0` (filtering off) while R defaulted it on. |
| `adam_headLength()` resolves the option in one place | `adam_head_length()` in `core/creator/architector.py`, with the same rule: absent means one full lag cycle (filtering on), `0` switches it off, larger values give that head length. |
| `headLength` reaches `adamProfileCreator()`, `obsStates` and `adamCpp$headLength` | `adam_profile_creator(head_length=)`, `observations_dict["obs_states"]`, `adam_cpp.headLength`. `head_length=` is a constructor argument on `ADAM` (so `ES`, `MSARIMA`, `SMA`, `AutoADAM`, `AutoMSARIMA` inherit it) and on `CES`. |
| ADAM's ARIMA no longer flips the drift | `adam_cpp.flipConstant = False` unconditionally in `architector()`. |
| `reapply()` takes the head length from the fitted object | `ADAM.reapply` passes `observations_dict["head_length"]` to the profile creator. |
| States trimmed to the last `lagsModelMax` head columns when the head is longer | Same trim in `forecaster/preparator.py`; the two downstream state slices now index from the array's own width, as R does with `nrow(object$states)`. |
| `auto.ces()` refuses a seasonal candidate below one full cycle | Same guard in `AutoCES` for partial / simple / full. |
| `ssarima`'s pure-MA transition fix; `auto.gum()`'s cycle guard; the `omg` / `sparma` forecast slices | No Python equivalent: the port has no `ssarima`, `gum` or `sparma` estimator, and `OMG` does not slice the lookup table itself. `sim_ssarima` builds its companion matrix from zeros, so it never had the `matF[1,1]` defect. |

## Parity sweep

Three series of different frequencies generated in R and read from the same CSV
in both languages (yearly n=40, quarterly n=64, monthly n=120), fitted with
`initial="backcasting"` and otherwise default settings. Compared: fitted values,
residuals, point forecasts, `logLik`, and the selected model where selection
applies. The figure is `max|R - Py| / max|R|` per vector; `logLik` is a plain
relative difference.

| series | spec | class | fitted | residuals | forecast | logLik | model (R \| Py) |
|---|---|---|---|---|---|---|---|
| yearly | ADAM `ZZZ` | ADAM | 0 | 4.3e-16 | 0 | 5.7e-14 | ETS(MAN) \| ETS(MAN) |
| yearly | ADAM `AAdA` | ADAM | 0 | 1.4e-16 | 0 | 0 | ETS(AAdN) \| ETS(AAdN) |
| yearly | ES `ZZZ` | ES | 0 | 1.9e-16 | 0 | 0 | ETS(MAN) \| ETS(MAN) |
| yearly | ES `AAdN` | ES | 0 | 1.4e-16 | 0 | 0 | ETS(AAdN) \| ETS(AAdN) |
| yearly | MSARIMA(0,1,1) | MSARIMA | 0 | 1.1e-16 | 0 | 0 | ARIMA(0,1,1) \| ARIMA(0,1,1) |
| yearly | MSARIMA seasonal | MSARIMA | 0 | 1.1e-16 | 0 | 0 | ARIMA(0,1,1) \| ARIMA(0,1,1) |
| yearly | CES none | CES | 0 | 1.3e-16 | 0 | 0 | CES(none) \| CES(none) |
| quarterly | ADAM `ZZZ` | ADAM | 0 | 1.3e-16 | 0 | 0 | ETS(AAA) \| ETS(AAA) |
| quarterly | ADAM `AAdA` | ADAM | 0 | 1.3e-16 | 0 | 0 | ETS(AAdA) \| ETS(AAdA) |
| quarterly | ES `ZZZ` | ES | 0 | 1.3e-16 | 0 | 0 | ETS(AAA) \| ETS(AAA) |
| quarterly | ES `AAdN` | ES | 0 | 5.4e-17 | 0 | 0 | ETS(AAdN) \| ETS(AAdN) |
| quarterly | MSARIMA(0,1,1) | MSARIMA | 0 | 6.4e-17 | 0 | 0 | ARIMA(0,1,1) \| ARIMA(0,1,1) |
| quarterly | MSARIMA seasonal | MSARIMA | 0 | 2.1e-16 | 0 | 0 | SARIMA(0,1,1)[1](0,1,1)[4] \| same |
| quarterly | CES none | CES | 0 | 4.7e-17 | 0 | 0 | CES(none) \| CES(none) |
| quarterly | CES full | CES | 0 | 1.2e-16 | 0 | 0 | CES(full) \| CES(full) |
| monthly | ADAM `ZZZ` | ADAM | 0 | 1.1e-16 | 0 | 0 | ETS(AAA) \| ETS(AAA) |
| monthly | ADAM `AAdA` | ADAM | 0 | 1.1e-16 | 0 | 0 | ETS(AAdA) \| ETS(AAdA) |
| monthly | ES `ZZZ` | ES | 0 | 1.1e-16 | 0 | 0 | ETS(AAA) \| ETS(AAA) |
| monthly | ES `AAdN` | ES | 0 | 7.3e-17 | 0 | 0 | ETS(AAdN) \| ETS(AAdN) |
| monthly | MSARIMA(0,1,1) | MSARIMA | 0 | 8.8e-17 | 0 | 0 | ARIMA(0,1,1) \| ARIMA(0,1,1) |
| monthly | MSARIMA seasonal | MSARIMA | 0 | 7.9e-17 | 0 | 0 | SARIMA(0,1,1)[1](0,1,1)[12] \| same |
| monthly | CES none | CES | 0 | 7.1e-17 | 0 | 0 | CES(none) \| CES(none) |
| monthly | CES full | CES | 0 | 9.8e-17 | 0 | 0 | CES(full) \| CES(full) |

Worst relative difference per class: **ADAM 5.7e-14, MSARIMA 2.1e-16,
ES 1.9e-16, CES 1.3e-16** — all PASS at the 1e-10 threshold, and every selected
model agrees.

The one cell above pure rounding is `logLik` on the yearly ETS(MAN) selection:
4e-12 absolute on -74.78. Fitted values and forecasts there are bit-identical,
so the gap enters through the last bit of the multiplicative residuals and the
scale computed from them.

## Divergences found and fixed along the way

1. **The compiled extension was stale.** `python/build/_adamCore*.so` predated
   the header by three weeks, and a header-only edit does not force a rebuild.
   The build directory is now cleaned (`ninja -t clean`) before reinstalling; the
   new kernel is confirmed live by `headLength` existing on `adamCore` at all and
   by `head_length=0` / `12` / `24` producing three different fits.

2. **CES landed in a different optimiser basin.** On quarterly data CES(full)
   reached -132.24 against R's -120.96. The cost surfaces were identical (40
   random parameter vectors, worst relative difference 3.2e-16, penalty region
   included), the C++ kernel returned bit-identical output for the same inputs,
   and the trajectories agreed for 40 BOBYQA evaluations before separating. The
   cause was one ULP in the CES level seed: `mean(yInSample[1:lagsModelMax])`
   uses R's two-pass long-double accumulator, `np.mean` a pairwise sum in double
   — 101.90728900000000579 against 101.90728899999999157. `ces_creator` now uses
   `_mean_r`, and the fit matches R exactly.

3. **`ADAM(model="AAdA", lags=[1])` crashed with `StopIteration`.** R warns and
   drops the seasonal component when the maximum lag is 1; Python kept
   `season_type="A"` with zero seasonal components. `_check_ets_model` now
   applies R's "unity lags" guard, which also makes `model="CCC", lags=[1]`
   report `ETS(CCN)` as R does (`tests/test_adam_combination.py` was pinning the
   old Python behaviour and has been corrected against a live R run).

4. **A named `persistence` dict was silently ignored** unless it used
   `level` / `trend` / `seasonal` / `xreg`. R accepts `alpha` / `beta` / `gamma` /
   `delta` as aliases, and `python/CLAUDE.md` documents the Greek form, so
   `persistence={"alpha": 0.3, ...}` fitted with persistence zero and no warning
   — 330 of log-likelihood on AirPassengers. Both spellings now work, with the
   component name winning when both are given.

5. **`tests/R scripts/ces_reference.R` loaded the installed package**
   (`library(smooth)`), not the working tree, so the fixtures encoded whatever
   version happened to be in the user's R library. It now uses
   `devtools::load_all(".")`, matching `tests/_r_bridge.py`, and the fixtures
   were regenerated. This is why `test_ces.py` failed after the rebuild: the
   references were from a pre-backcasting `smooth`.

## Suite status

* `pytest tests/ -q` — 926 passed, 460 deselected.
* `pytest tests/ -m "r_parity or r_comparison" -q` — 460 passed.

`tests/test_backcast_head.py` mirrors R's `tests/testthat/test_backcastHead.R`
and pins the head-length semantics, the kernel flag, and `flipConstant` staying
`False`.
