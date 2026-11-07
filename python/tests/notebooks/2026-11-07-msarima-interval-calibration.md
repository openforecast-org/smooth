# MSARIMA prediction interval calibration

Diagnosis of the under-coverage of `AutoMSARIMA` / `auto.msarima()` on M1, M3 and Tourism
(5,315 series), as asked in `.claude/msarima-interval-calibration-prompt.md`. Nothing in `src/`,
`python/src/` or `R/` was changed.

Every number below comes from a recorded run of the claimset study
`msarima-interval-calibration` (branch `claimset/msarima-interval-calibration`, anchored to a
local bare repository). `RNx` is the run and `PLx` the plan in that record. The code is in
`claimset-code/msarima-interval-calibration/` on the branch `msarima-interval-calibration`.
`2026-11-07-msarima-interval-calibration.ipynb` reproduces the tables from the recorded outputs
and lists the commands that rerun them.

## Verdicts of the planned tests

All plans were approved by the supervisor before they ran. Thresholds are fixed in the plans.

| Plan | Hypothesis | Verdict | Deciding numbers |
|---|---|---|---|
| PL4 | H1: true orders and parameters give nominal coverage at the theoretical width | supported | MCE about 0.002 at T=100 and 300, width 1.002 (RN6) |
| PL6 | H3: analytic and simulated intervals agree | supported | (RN7) |
| PL7 | H4: estimating the parameters costs at least 6 points of cov80 at T=100 | not supported | 0.800 to 0.770, 2.96 points (RN8) |
| PL8 | H5: `reforecast()` improves on `predict()` in simulation | not supported | relative pinball 1.043 to 1.958 with backcasting (RN9) |
| PL9 | H6: a wrong model alone gives a benchmark-sized gap | supported | cov80 0.676, q50 0.428 (RN10) |
| PL5 | H2: selection loses little (cov80 at T=100 at least 0.77) | inconclusive | 0.756 (RN11) |
| PL10 | H7: `reforecast()` closes half of the benchmark gap | not supported | cov80 0.6812 to 0.6827, pinball mean 7e191 (RN12) |
| PL12 | H8: coverage falls with the level, on M4 | supported | 0.813 / 0.699 / 0.554 by level tercile; ETS(MNN) 0.764 / 0.778 / 0.814 (RN41) |
| PL13 | H9: a log-normal error with the same orders removes it, on M4 | supported | tercile spread 0.031, MCE 0.0386 to 0.0177 (RN42) |
| PL11 | H10: 1.3.0 is calibrated no worse than 1.0.8 on known models | supported | MCE at T=100: E back 0.0100 vs 0.0120, S back 0.0139 vs 0.0172, E opt 0.0085 vs 0.0246 (RN43, RN44) |

The plans PL11 to PL13 carry the flag "code changed after plan": the only commit between them
and their runs updated `.claude/skills` to claimset 0.39.0 at the supervisor's request (NT6).

## Q1. Are the intervals right when the model is right?

Yes. `interval="prediction"` dispatches to `approximate` for a pure ARIMA in both languages.
With the true orders and parameters the 99 quantiles are calibrated and the width equals the
theoretical h-step width (PL4). The simulated interval agrees with the analytic one (PL6). R and
Python give the same model, width (ratio 1.000) and coverage (0.677) on 500 benchmark series
(RN37).

Pooled cov80 over 11 Gaussian DGPs, backcasting (RN8, RN11, RN39):

| T | true parameters | estimated | selected |
|---|---|---|---|
| 15 | 0.818 | 0.590 | 0.618 |
| 25 | 0.793 | 0.691 | 0.677 |
| 50 | 0.800 | 0.739 | 0.722 |
| 100 | 0.800 | 0.770 | 0.756 |
| 300 | 0.799 | 0.793 | 0.786 |

Parameter and selection uncertainty is small at T=100 (H4 not supported) but large on short
series. At the benchmark's lengths (median 53, 22% of 25 or fewer) it predicts cov80 0.717 with
selection and 0.725 with known orders (RN40).

Wrong models (PL9) can produce the size of the gap, mainly through location (q50 0.428), but the
pool of mistakes was chosen, not sampled.

## Q2. What moved between 1.0.8 and 1.3.0?

The width did not: its median ratio between the versions is 1.000 (RN20). The apparent narrowing
with fixed parameters (0.987) is an artefact of fixing them, as they leave the df.

- **Backcasting, cov80 0.7002 to 0.6812:**
  - series that changed model: −1.11 points (27% of series);
  - series that kept it: −0.77 points;
  - the drift share fell from 30.9% to 24.8% (RN20).
- **Bisect** (RN21 to RN36), all three correctness fixes by their commit messages:
  - `d308f9bf`, headLength exposed to Python: 0.7002 to 0.7217;
  - `d477bb81`, each state moved across the backcasting turns by its own lag: 0.7168 to 0.6887;
  - `e3b156ed`, ARIMA initial states aligned by lag (optimal): 0.6992 to 0.6519, partly recovered
    later to 0.6845.
- **1.0.8 was too wide on known models:**
  - with the true parameters it gives cov80 0.826 at width 1.110;
  - with optimal initials and estimated parameters, width 3.386 (RN43).
  1.3.0 is calibrated there (H10). So the 1.0.8 coverage was not better calibration.
- **Pinball:** with backcasting 0.8961 (1.0.8) against 0.8991 (1.3.0); with optimal 0.9118
  against 0.9009 (RN16, RN19).

## Q3. Where does it concentrate?

On the benchmark (backcasting, RN14):

| Group | cov80 |
|---|---|
| M1 | 0.631 |
| M3 | 0.718 |
| Tourism | 0.634 |
| yearly | 0.611 |
| quarterly | 0.662 |
| monthly | 0.736 |
| 25 observations or fewer | 0.571 |
| ARIMA(0,1,0) with drift | 0.578 |
| d=1 without drift | 0.742 |

Coverage is flat over the horizon, and the error is already too wide for the interval at h=1:
the sd of z is 1.45 at h=1.

**Location against width:**
- Re-centring on the pooled median leaves cov80 at 0.71, so the q50 bias costs almost nothing
  (RN14).
- Re-centring on each series gives 0.873. Under the true model the same operation gives 0.95 to
  0.98 (RN15), so this is not evidence of location error: it points to width.

**The scale grows with the level.**
- By tercile of the recent level over the mean level, cov80 is 0.786, 0.668 and 0.590 (RN15),
  confirmed on M4 (H8).
- statsforecast (0.789, 0.667, 0.589), pmdarima and skforecast show the same fall (RN15).
- ETS(MNN), whose scale follows the level, is flat.
- A constant error variance, estimated over the whole sample, is too small at the origin of a
  series that has grown.

**Same orders with a log-normal error:**
- On the benchmark (RN38): cov80 0.681 to 0.756, MCE 0.0402 to 0.0124, q50 0.470 to 0.499, and
  the terciles go flat at 0.749, 0.751 and 0.768.
- On M4 (H9, RN42): cov80 0.689 to 0.770.
- It costs pinball: the median goes from 0.496 to 0.551 on the benchmark and from 0.675 to 0.768
  on M4.
- On simulated series with multiplicative errors, AutoMSARIMA reproduces the pattern (cov80
  0.658, terciles 0.813, 0.638, 0.524). The log-normal version covers 0.763 with a lower pinball
  mean (0.808 against 0.878) (RN38).

## Q4. Why is ETS wider than ARIMA?

Not the scale or the variance code.
- Where ETS(ANN) and ARIMA(0,1,1) give the same forecast (114 series), their coverage is equal,
  0.734 (RN15).
- Where they differ, ARIMA's MA term lies outside ETS's bounds on alpha.
- The gap comes from the multiplicative ETS models in the pool. Forced on all series (RN14):

  | Model | cov80 |
  |---|---|
  | ETS(MNN) | 0.764 |
  | ETS(ANN) | 0.668 |
  | ARIMA(0,1,1) | 0.695 |

- Seasonal ETS against seasonal ARIMA: 0.795 against 0.709.
- The q50 of ETS (0.448) was not examined further.

## Q5. Which part is smooth's to fix?

1. **Under-coverage shared by all ARIMA (fact 1):** a shared limitation and a property of the
   data, not a defect.
   - About two thirds of the gap comes from the additive constant-variance error on series whose
     scale grows with the level.
   - Most of the rest comes from estimation and selection on short series.
   - The two overlap, so their shares are not additive.
2. **1.0.8 to 1.3.0 (fact 2):** not a defect. Three correctness fixes removed intervals that were
   too wide on known models.
3. **ARIMA narrower than ETS (fact 3):** the model pool. ETS can have multiplicative errors and
   ARIMA in `AutoMSARIMA` cannot.

**Ruled out:**
- a bug in the analytic variance;
- a difference between the analytic and the simulated intervals;
- an R/Python difference in the intervals;
- the de-biased scale;
- parameter uncertainty alone at T=100;
- q50 bias as a cause of the coverage loss.

**Defects found along the way:**
- `reforecast()` for ARIMA with a constant or AR terms:
  - draws give intervals several times too wide, in both languages but differently (RN5);
  - the benchmark pinball explodes on some series (RN12);
  - on a model with no estimated parameter it raises `ValueError` in Python (NT1; the harness
    substitutes `predict()` there).
- With optimal initials the parameter covariance differs between R and Python (RN5).
- `sar12` with optimal initials estimated AR near zero on one series in both languages (RN5): a
  lead, not a finding.

**Is a fix warranted?**
- A log-normal (multiplicative) error option for `AutoMSARIMA` would raise cov80 by about 7.5
  points and cut MCE by a half to two thirds. But it worsens the median pinball by 11-14%, so it
  is a modelling choice, not a correction.
- `reforecast()` for ARIMA is a defect worth fixing, with no effect on `predict()`.
