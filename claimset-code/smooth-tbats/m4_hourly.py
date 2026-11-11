"""smooth's TBATS against forecast::tbats, AutoTBATS, ADAM ETS and seasonal naive on the
M4 hourly series (Q2 of the smooth-tbats study).

Each method is fitted once to each fitting sample and forecasts 48 steps ahead
(fixed-origin evaluation). The split is either the official one (``--split test``: the
training samples are fitted and the test samples evaluated) or the validation one
(``--split validation``: the last 48 training observations are held out and never the
official test samples are read).

Methods, all single-threaded (OMP_NUM_THREADS=1), with 95% prediction intervals:
- smooth TBATS: TBATS(lags=[1, 24, 168]) with its default settings;
- forecast::tbats: tbats(msts(y, c(24, 168))) in R (m4_forecast_tbats.R);
- AutoTBATS: statsforecast's AutoTBATS(season_length=[24, 168]);
- ADAM ETS: ADAM(model="ZXZ", lags=[1, 24, 168]) with its default settings;
- seasonal naive 168: the last week repeated (no intervals).
A method that fails on a series (an error, or non-finite forecasts) gets the forecasts
of seasonal naive 168 for that series, and no interval; the failures are counted.

Measures: MASE (scaled by the in-sample mean absolute seasonal difference at lag 24, as
in M4), sMAPE, the coverage of the 95% intervals, the elapsed seconds per series.
Statistics of smooth TBATS against each comparator: the geometric mean over series of
the ratio of MASEs (relative MASE) with its 95% t-interval on the log scale, and
one-sided t-tests on the mean log ratio: non-inferiority with the margin 1.05 against
forecast::tbats and AutoTBATS (H0: relative MASE >= 1.05), superiority against
seasonal naive 168 (H0: relative MASE >= 1), and the elapsed time against
forecast::tbats (H0: ratio of times >= 1); the four p-values are adjusted by Holm.

Run: python m4_hourly.py --train TRAIN.csv --test TEST.csv --split test [--series N]
Writes the per-series measures and the forecasts into $CLAIMSET_RUN_DIR and the
results to $CLAIMSET_RESULTS.
"""

import argparse
import json
import os
import subprocess
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

H = 48
LEVEL = 0.95
HERE = Path(__file__).resolve().parent
PYTHON_METHODS = ["smooth TBATS", "AutoTBATS", "ADAM ETS", "seasonal naive 168"]
COMPARATORS = ["forecast::tbats", "AutoTBATS", "ADAM ETS", "seasonal naive 168"]


def load(train_path, test_path, split, count):
    train = pd.read_csv(train_path, index_col=0)
    names = list(train.index)[: count or None]
    samples = {}
    if split == "test":
        test = pd.read_csv(test_path, index_col=0)
    for name in names:
        y = train.loc[name].dropna().to_numpy(float)
        if split == "test":
            samples[name] = (y, test.loc[name].dropna().to_numpy(float)[:H])
        else:
            samples[name] = (y[:-H], y[-H:])
    return samples


def snaive(y):
    return np.resize(y[-168:], H)


def fit_python(method, y):
    """Forecasts, lower and upper bounds (None without intervals) of a method."""
    if method == "seasonal naive 168":
        return snaive(y), None, None
    if method == "smooth TBATS":
        from smooth import TBATS

        fc = TBATS(lags=[1, 24, 168]).fit(y).predict(
            h=H, interval="prediction", level=LEVEL
        )
    elif method == "ADAM ETS":
        from smooth import ADAM

        fc = ADAM(model="ZXZ", lags=[1, 24, 168]).fit(y).predict(
            h=H, interval="prediction", level=LEVEL
        )
    else:
        from statsforecast.models import AutoTBATS

        out = AutoTBATS(season_length=[24, 168]).forecast(y, h=H, level=[95])
        return out["mean"], out["lo-95"], out["hi-95"]
    return (np.ravel(np.asarray(fc.mean, float)),
            np.ravel(np.asarray(fc.lower, float)),
            np.ravel(np.asarray(fc.upper, float)))


def warm_up():
    """Imports and JIT compilation outside the timings."""
    os.environ["OMP_NUM_THREADS"] = "1"
    rng = np.random.default_rng(0)
    y = 100 + 10 * np.sin(2 * np.pi * np.arange(400) / 24) + rng.normal(0, 1, 400)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for method in ("AutoTBATS",):
            fit_python(method, y)
        from smooth import ADAM, TBATS  # noqa: F401


def run_series(args):
    name, y = args
    rows = []
    for method in PYTHON_METHODS:
        started = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mean, lower, upper = fit_python(method, y)
            status = "ok" if np.all(np.isfinite(mean)) else "non-finite forecasts"
        except Exception as error:  # noqa: BLE001 - a failure is counted, not hidden
            mean = lower = upper = None
            status = f"error: {type(error).__name__}: {error}"[:300]
        rows.append({"series": name, "method": method, "mean": mean, "lower": lower,
                     "upper": upper, "seconds": time.perf_counter() - started,
                     "status": status})
    return rows


def run_r(samples, run_dir, workers):
    data = pd.DataFrame([{"series": k, "y": v} for k, (y, _) in samples.items()
                         for v in y])
    path_in, path_out = run_dir / "tbats_input.csv", run_dir / "forecast_tbats.csv"
    data.to_csv(path_in, index=False)
    subprocess.run(["Rscript", str(HERE / "m4_forecast_tbats.R"), str(path_in),
                    str(path_out), str(workers), str(H)], check=True,
                   env={**os.environ, "OMP_NUM_THREADS": "1"})
    path_in.unlink()
    out = pd.read_csv(path_out)
    rows = []
    for name, group in out.groupby("series", sort=False):
        group = group.sort_values("h")
        mean = group["mean"].to_numpy(float)
        status = group["status"].iloc[0]
        if status == "ok" and not np.all(np.isfinite(mean)):
            status = "non-finite forecasts"
        rows.append({"series": name, "method": "forecast::tbats", "mean": mean,
                     "lower": group["lower"].to_numpy(float),
                     "upper": group["upper"].to_numpy(float),
                     "seconds": float(group["seconds"].iloc[0]), "status": status})
    return rows


def measures(row, y, actual):
    failed = row["status"] != "ok"
    mean = snaive(y) if failed else row["mean"]
    scale = np.mean(np.abs(y[24:] - y[:-24]))
    result = {
        "series": row["series"], "method": row["method"], "failed": failed,
        "status": row["status"], "seconds": row["seconds"],
        "MASE": float(np.mean(np.abs(actual - mean)) / scale),
        "sMAPE": float(np.mean(200 * np.abs(actual - mean)
                               / (np.abs(actual) + np.abs(mean)))),
        "coverage": np.nan,
    }
    if not failed and row["lower"] is not None:
        inside = (actual >= row["lower"]) & (actual <= row["upper"])
        result["coverage"] = float(np.mean(inside))
    return result


def one_sided(log_ratios, null):
    """Mean, its 95% interval and the one-sided t-test p-value of H0: mean >= null."""
    n = len(log_ratios)
    mean, se = np.mean(log_ratios), np.std(log_ratios, ddof=1) / np.sqrt(n)
    half = stats.t.ppf(0.975, n - 1) * se
    p = stats.t.cdf((mean - null) / se, n - 1)
    return mean, mean - half, mean + half, float(p), n


def holm(p_values):
    order = np.argsort(p_values)
    adjusted, running = np.empty(len(p_values)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def statistics(table, split):
    wide = table.pivot(index="series", columns="method", values="MASE")
    times = table.pivot(index="series", columns="method", values="seconds")
    results, tests = [], {}
    base = {"split": split}
    for method, group in table.groupby("method"):
        results += [
            {"variable": "MASE", "value": float(group["MASE"].mean()),
             "conditions": {**base, "method": method, "statistic": "mean"}},
            {"variable": "MASE", "value": float(group["MASE"].median()),
             "conditions": {**base, "method": method, "statistic": "median"}},
            {"variable": "sMAPE", "value": float(group["sMAPE"].mean()),
             "conditions": {**base, "method": method, "statistic": "mean"}},
            {"variable": "elapsed time", "value": float(group["seconds"].median()),
             "unit": "s", "conditions": {**base, "method": method,
                                         "statistic": "median"}},
            {"variable": "series failed", "value": int(group["failed"].sum()),
             "conditions": {**base, "method": method}},
        ]
        if group["coverage"].notna().any():
            results.append({"variable": "coverage",
                            "value": float(group["coverage"].mean()),
                            "conditions": {**base, "method": method,
                                           "nominal level": LEVEL}})
    for comparator in COMPARATORS:
        ratios = np.log(wide["smooth TBATS"] / wide[comparator]).to_numpy()
        excluded = int(np.sum(~np.isfinite(ratios)))
        ratios = ratios[np.isfinite(ratios)]
        null = np.log(1.05) if comparator in ("forecast::tbats", "AutoTBATS") else 0.0
        mean, lower, upper, p, n = one_sided(ratios, null)
        conditions = {**base, "method": "smooth TBATS", "comparator": comparator}
        results += [
            {"variable": "relative MASE", "value": float(np.exp(mean)),
             "interval": {"level": 0.95, "lower": float(np.exp(lower)),
                          "upper": float(np.exp(upper))}, "conditions": conditions},
            {"variable": "series in relative MASE", "value": n,
             "conditions": conditions},
            {"variable": "series excluded by zero MASE", "value": excluded,
             "conditions": conditions},
        ]
        if comparator != "ADAM ETS":
            tests[comparator] = p
    log_times = np.log(times["smooth TBATS"] / times["forecast::tbats"]).to_numpy()
    mean, lower, upper, tests["elapsed time"], n = one_sided(log_times, 0.0)
    results.append({"variable": "relative elapsed time", "value": float(np.exp(mean)),
                    "interval": {"level": 0.95, "lower": float(np.exp(lower)),
                                 "upper": float(np.exp(upper))},
                    "conditions": {**base, "method": "smooth TBATS",
                                   "comparator": "forecast::tbats"}})
    names = list(tests)
    for name, p, adjusted in zip(names, [tests[k] for k in names],
                                 holm(np.array([tests[k] for k in names]))):
        test = "time" if name == "elapsed time" else "MASE"
        for kind, value in (("p-value", p), ("Holm-adjusted p-value", adjusted)):
            results.append({"variable": f"{kind}, one-sided t-test, smooth TBATS vs "
                                        f"{name if test == 'MASE' else 'forecast::tbats'}"
                                        f" ({test})", "value": float(value),
                            "conditions": base})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--series", type=int, default=0, help="first N series; 0: all")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    run_dir = Path(os.environ.get("CLAIMSET_RUN_DIR", "."))
    samples = load(args.train, args.test, args.split, args.series)
    rows = run_r(samples, run_dir, args.workers)
    with ProcessPoolExecutor(args.workers, initializer=warm_up) as pool:
        for series_rows in pool.map(run_series, [(k, v[0]) for k, v in samples.items()]):
            rows += series_rows
    table = pd.DataFrame([measures(r, *samples[r["series"]]) for r in rows])
    table.to_csv(run_dir / "measures.csv", index=False)
    forecasts = pd.DataFrame([
        {"series": r["series"], "method": r["method"], "h": i + 1,
         "mean": None if r["mean"] is None else r["mean"][i],
         "lower": None if r["lower"] is None else r["lower"][i],
         "upper": None if r["upper"] is None else r["upper"][i]}
        for r in rows for i in range(H)])
    forecasts.to_csv(run_dir / "forecasts.csv", index=False)
    results = statistics(table, args.split)
    for r in results:
        print(r["variable"], r.get("conditions", {}), round(r["value"], 5),
              r.get("interval", ""))
    path = os.environ.get("CLAIMSET_RESULTS")
    if path:
        with open(path, "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    main()
