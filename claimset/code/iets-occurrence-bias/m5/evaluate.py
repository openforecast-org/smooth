"""Evaluation of the M5 forecasts of forecast_iets.py and forecast_benchmarks.R. Usage:
    evaluate.py <mode> <sales_train> <sales_test> <rows.csv> <iets_forecasts.csv>
                <benchmark_forecasts.csv>
Per series and method (over h = 1-14, in-sample after the leading zeros):
- RMSSE, as in M5: sqrt(mean((y - f)^2) / mean(diff(y_in)^2));
- sCE = sum(y - f) / mean(y_in); sAPIS = |sum_j sum_{i<=j} (f_i - y_i)| / mean(y_in);
- scaled pinball at tau, as M5's SPL: mean pinball_tau(y, q_tau) / mean(|diff(y_in)|);
- coverage at tau: the share of the horizon with y <= q_tau; both also for the quantiles
  rounded up (SB23, eq. 35).
Every run writes series_measures.csv (each measure per series and method) and
summary_measures.csv (per method and measure: n, mean, sd, min, quartiles, max) to
$CLAIMSET_RUN_DIR.
Exclusions: series with fewer than 5 non-zero in-sample observations ("few"), series where
the in-sample scale (mean of diff^2) is zero ("zero scale"), and series where any method
failed or left a forecast it should produce missing ("failed"); each is counted. Tests, by mode:
- direction: the per-series sum of the 14 point forecasts, score minus ratio, for iETS_I and
  iETS_O under conventional and ADAM ETS; predicted: + for I conventional, - for the others;
- rmsse: RMSSE, score minus ratio, for O, I, G, A under conventional and ADAM ETS; predicted -;
- pinball: scaled pinball at the four levels for the same eight pairs; predicted -;
- mcb: rmcb() of greybox (Nemenyi/MCB, 99%) of RMSSE over all methods and of the scaled
  pinball at each level over the methods with quantiles; per measure, the best score-proxy
  model against each benchmark.
Wilcoxon signed-rank tests are two-sided, zero differences dropped (zero_method="wilcox"),
with Holm's correction over the tests of the mode at alpha = 0.01; the direction of a
difference is the sign of its median over the non-zero differences.
"""

import csv
import json
import os
import sys

import numpy as np
from greybox import rmcb
from scipy.stats import wilcoxon

from load import H, read_series

LEVELS = [0.8, 0.9, 0.95, 0.99]
ALPHA = 0.01
FLAVOURS = [(k, e) for e in ("conv", "adam") for k in "OIGA"]
POINT_ONLY = {"Croston", "SBA", "TSB"}
BENCHMARKS = {"RMSSE": ["Croston", "SBA", "TSB", "NegBin", "HSP"],
              **{f"pinball{t}": ["NegBin", "HSP"] for t in LEVELS}}


def read_forecasts(*paths):
    out, status = {}, {}
    for path in paths:
        for r in csv.DictReader(open(path)):
            key = (int(r["row"]), r["method"])
            status.setdefault(key, set()).add(r["status"])
            if r["status"] == "ok":
                out.setdefault(key, np.full((H, 5), np.nan))[int(r["horizon"]) - 1] = [
                    float(r[c]) if r[c] not in ("", "NA") else np.nan
                    for c in ("mean", "q0.8", "q0.9", "q0.95", "q0.99")]
    return out, status


def pinball(y, q, tau):
    return np.mean(np.maximum(tau * (y - q), (tau - 1) * (y - q)))


def measures(y, fc):
    y_in, y_out = y[:-H], y[-H:]
    scale2, scale1, level = np.mean(np.diff(y_in) ** 2), np.mean(np.abs(np.diff(y_in))), np.mean(y_in)
    f = fc[:, 0]
    m = {"RMSSE": np.sqrt(np.mean((y_out - f) ** 2) / scale2),
         "sCE": np.sum(y_out - f) / level,
         "sAPIS": abs(np.sum(np.cumsum(f - y_out))) / level,
         "sum_forecast": np.sum(f)}
    for j, tau in enumerate(LEVELS):
        q = fc[:, j + 1]
        if np.all(np.isfinite(q)):
            for suffix, qq in (("", q), ("_rounded", np.ceil(q))):
                m[f"pinball{tau}{suffix}"] = pinball(y_out, qq, tau) / scale1
                m[f"coverage{tau}{suffix}"] = np.mean(y_out <= qq)
    return m


def holm(p):
    order, adjusted, running = np.argsort(p), np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def paired(table, comparisons):
    # Wilcoxon tests of (score, ratio, measure, predicted sign): one Holm family; counts of
    # the significant results in and against the predicted direction
    diffs = [np.array([table[r][s][m] - table[r][b][m] for r in table])
             for s, b, m, _ in comparisons]
    p = np.array([wilcoxon(d, zero_method="wilcox").pvalue if np.any(d != 0) else 1.0
                  for d in diffs])
    adjusted, results, good, bad = holm(p), [], 0, 0
    for (s, b, m, sign), d, pa in zip(comparisons, diffs, adjusted):
        median = float(np.median(d[d != 0])) if np.any(d != 0) else 0.0
        significant = pa < ALPHA and median != 0
        good += significant and np.sign(median) == sign
        bad += significant and np.sign(median) == -sign
        cond = {"measure": m, "comparison": f"{s} minus {b}"}
        results += [{"quantity": "Wilcoxon p-value, Holm-adjusted", "value": float(pa), "conditions": cond},
                    {"quantity": "median non-zero difference", "value": median, "conditions": cond},
                    {"quantity": "share of zero differences", "value": float(np.mean(d == 0)), "conditions": cond}]
    return results, good, bad


def mcb(table):
    methods = sorted(next(iter(table.values())))
    results, good, bad = [], 0, 0
    for measure, benchmarks in BENCHMARKS.items():
        names = [m for m in methods if measure in table[next(iter(table))][m]]
        data = np.array([[table[r][m][measure] for m in names] for r in table])
        res = rmcb(data, level=1 - ALPHA, outplot="none")
        mean = dict(zip(names, np.asarray(res.mean)))
        lower = dict(zip(names, np.asarray(res.interval)[:, 0]))
        upper = dict(zip(names, np.asarray(res.interval)[:, 1]))
        best = min((m for m in names if "_score_" in m), key=lambda m: mean[m])
        for b in benchmarks:
            good += upper[best] < lower[b]
            bad += upper[b] < lower[best]
        results += [{"quantity": "mean rank, rmcb", "value": float(mean[m]),
                     "interval": {"level": 1 - ALPHA, "lower": float(lower[m]), "upper": float(upper[m])},
                     "conditions": {"measure": measure, "method": m}} for m in names]
        results.append({"quantity": "rmcb p-value", "value": float(res.p_value), "conditions": {"measure": measure}})
    return results, good, bad


def write_measures(table, outdir):
    keys = ["RMSSE", "sCE", "sAPIS"] + [f"{k}{t}{s}" for k in ("pinball", "coverage")
                                         for t in LEVELS for s in ("", "_rounded")]
    methods = sorted(next(iter(table.values())))
    with open(os.path.join(outdir, "series_measures.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "method", *keys])
        for row, by_method in table.items():
            for m in methods:
                writer.writerow([row, m, *[by_method[m].get(k, "") for k in keys]])
    with open(os.path.join(outdir, "summary_measures.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "measure", "n", "mean", "sd", "min", "q25", "median",
                         "q75", "max"])
        for m in methods:
            for k in keys:
                v = np.array([table[r][m][k] for r in table if k in table[r][m]])
                if v.size:
                    writer.writerow([m, k, v.size, v.mean(), v.std(ddof=1), v.min(),
                                     *np.quantile(v, [0.25, 0.5, 0.75]), v.max()])


def main(mode, train, test, rows_file, *forecast_files):
    rows = [int(r["row"]) for r in csv.DictReader(open(rows_file))]
    series = read_series(train, test, rows)
    forecasts, status = read_forecasts(*forecast_files)
    methods = sorted({m for _, m in status})
    excluded = {"few": 0, "zero scale": 0, "failed": 0}
    table = {}
    for row in rows:
        y = np.asarray(series[row][2], float)
        if np.count_nonzero(y[:-H]) < 5:
            excluded["few"] += 1
        elif np.mean(np.diff(y[:-H]) ** 2) == 0:
            excluded["zero scale"] += 1
        elif any((row, m) not in forecasts
                 or not np.all(np.isfinite(forecasts[(row, m)][:, :1 if m in POINT_ONLY else 5]))
                 for m in methods):
            excluded["failed"] += 1
        else:
            table[row] = {m: measures(y, forecasts[(row, m)]) for m in methods}
    write_measures(table, os.environ.get("CLAIMSET_RUN_DIR", "."))
    results = [{"quantity": f"series excluded by {k}", "value": v} for k, v in excluded.items()]
    results.append({"quantity": "series evaluated", "value": len(table)})
    pairs = [(f"iETS_{k}_score_{e}", f"iETS_{k}_ratio_{e}") for k, e in FLAVOURS]
    if mode == "direction":
        signs = {("I", "conv"): 1, ("O", "conv"): -1, ("I", "adam"): -1, ("O", "adam"): -1}
        comparisons = [(f"iETS_{k}_score_{e}", f"iETS_{k}_ratio_{e}", "sum_forecast", sign)
                       for (k, e), sign in signs.items()]
        res, good, bad = paired(table, comparisons)
    elif mode == "rmsse":
        res, good, bad = paired(table, [(s, b, "RMSSE", -1) for s, b in pairs])
    elif mode == "pinball":
        res, good, bad = paired(table, [(s, b, f"pinball{t}", -1) for t in LEVELS for s, b in pairs])
    else:
        res, good, bad = mcb(table)
    for m in methods:
        for key in ("RMSSE", "sCE", "sAPIS", *[f"{k}{t}{s}" for k in ("pinball", "coverage")
                                               for t in LEVELS for s in ("", "_rounded")]):
            vals = [table[r][m][key] for r in table if key in table[r][m]]
            if vals:
                results.append({"quantity": f"mean {key}", "value": float(np.mean(vals)),
                                "conditions": {"method": m}})
    results += res + [{"quantity": "comparisons significant in the predicted direction", "value": int(good)},
                      {"quantity": "comparisons significant against the predicted direction", "value": int(bad)}]
    print({r["quantity"]: r["value"] for r in results if "conditions" not in r})
    path = os.environ.get("CLAIMSET_RESULTS")
    if path:
        json.dump(results, open(path, "w"))


if __name__ == "__main__":
    main(*sys.argv[1:])
