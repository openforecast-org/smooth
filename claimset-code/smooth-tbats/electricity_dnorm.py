"""smooth's TBATS with distribution="dnorm" against its default distribution="auto" on the
hourly electricity data, with the windows, origins, measures and failure rule of
electricity_hourly.py (--split test: the windows of PL7).

At each window both are fitted in the same worker, one after the other, so that their
elapsed times are measured under the same load. Per series (the means over the 7
origins), three one-sided t-tests at ALPHA, adjusted by Holm over the three:
- RMSSE: the mean log ratio of RMSSE, dnorm to auto, H0: <= 0;
- coverage: the mean coverage of the 95% interval of dnorm minus 0.95, H0: >= 0;
- time: the mean log ratio of the elapsed time, dnorm to auto, H0: >= 0.

Not deciding: the Nemenyi test (rmcb) on the RMSSE of all six methods, the four other
methods taken from the per-series measures of PL7's run (--others, RN13's measures.csv),
drawn in nemenyi.png; the tables of the methods, the tests and the Nemenyi test as CSV.

Run: python electricity_dnorm.py --data electricity_nips.tar.gz --split test --workers N
    --others RN13/measures.csv
"""

import argparse
import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from electricity_hourly import load, per_series
from m4_hourly import ALPHA, H, LEVEL, holm, measures, nemenyi, one_sided, warm_up
from greybox import rmcb

matplotlib.use("Agg")

METHODS = {"smooth TBATS": "auto", "smooth TBATS dnorm": "dnorm"}


def run_series(args):
    from smooth import TBATS

    name, y = args
    rows = []
    for method, distribution in METHODS.items():
        started = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fc = TBATS(lags=[1, 24, 168], distribution=distribution).fit(y).predict(
                    h=H, interval="prediction", level=LEVEL
                )
            mean, lower, upper = (np.ravel(np.asarray(v, float)) for v in (fc.mean, fc.lower, fc.upper))
            status = "ok" if np.all(np.isfinite(mean)) else "non-finite forecasts"
        except Exception as error:  # noqa: BLE001 - a failure is counted, not hidden
            mean = lower = upper = None
            status = f"error: {type(error).__name__}: {error}"[:300]
        rows.append({"series": name, "method": method, "mean": mean, "lower": lower,
                     "upper": upper, "seconds": time.perf_counter() - started, "status": status})
    return rows


def test(values, null, name, split):
    """The one-sided test of H0: mean >= null over the finite values (a zero RMSSE or time
    makes a log ratio non-finite), as a result with its 1 - ALPHA interval."""
    finite = np.isfinite(values)
    mean, lower, upper, p, n = one_sided(values[finite], null)
    return p, {"variable": name, "value": float(mean), "conditions": {"split": split},
               "interval": {"level": 1 - ALPHA, "lower": float(lower), "upper": float(upper)}}, n, int((~finite).sum())


def statistics(series, split):
    wide = {v: series.pivot(index="series", columns="method", values=v) for v in ("RMSSE", "coverage", "seconds")}
    auto, dnorm = "smooth TBATS", "smooth TBATS dnorm"
    results = []
    for method, group in series.groupby("method"):
        c = {"split": split, "method": method}
        results += [
            {"variable": "RMSSE", "value": float(group["RMSSE"].mean()), "conditions": {**c, "statistic": "mean"}},
            {"variable": "RMSSE", "value": float(group["RMSSE"].median()), "conditions": {**c, "statistic": "median"}},
            {"variable": "coverage", "value": float(group["coverage"].mean()), "conditions": {**c, "nominal level": LEVEL}},
            {"variable": "elapsed time", "value": float(group["seconds"].median()),
             "conditions": {**c, "statistic": "median"}},
            {"variable": "series failed", "value": int(group["failed"].sum()), "conditions": c},
            {"variable": "fits failed", "value": int(group["fits_failed"].sum()), "conditions": c},
        ]
    tests = {
        # dnorm less accurate: H0 log(RMSSE dnorm / RMSSE auto) <= 0, tested as the negated ratio >= 0
        "RMSSE": test(np.log(wide["RMSSE"][auto] / wide["RMSSE"][dnorm]), 0,
                      "mean log ratio of RMSSE, smooth TBATS to smooth TBATS dnorm", split),
        "coverage": test(wide["coverage"][dnorm] - LEVEL, 0,
                         "mean coverage of smooth TBATS dnorm minus 0.95", split),
        "time": test(np.log(wide["seconds"][dnorm] / wide["seconds"][auto]), 0,
                     "mean log ratio of elapsed time, smooth TBATS dnorm to smooth TBATS", split),
    }
    adjusted = holm(np.array([t[0] for t in tests.values()]))
    for (name, (p, result, n, excluded)), p_holm in zip(tests.items(), adjusted):
        results += [result,
                    {"variable": f"p-value, one-sided t-test, {name}", "value": p, "conditions": {"split": split}},
                    {"variable": f"Holm-adjusted p-value, one-sided t-test, {name}", "value": float(p_holm),
                     "conditions": {"split": split}},
                    {"variable": f"series in the t-test, {name}", "value": n, "conditions": {"split": split}},
                    {"variable": f"series excluded by a non-finite value, {name}", "value": excluded,
                     "conditions": {"split": split}}]
    for name, num, den in (("relative RMSSE", dnorm, auto), ("relative elapsed time", dnorm, auto)):
        v = "RMSSE" if name == "relative RMSSE" else "seconds"
        ratio = np.log(wide[v][num] / wide[v][den])
        mean, lower, upper, _, _ = one_sided(ratio[np.isfinite(ratio)], 0)
        results.append({"variable": name, "value": float(np.exp(mean)),
                        "conditions": {"split": split, "method": num, "comparator": den},
                        "interval": {"level": 1 - ALPHA, "lower": float(np.exp(lower)), "upper": float(np.exp(upper))}})
    return results


def all_methods(series, others, split, run_dir):
    """The Nemenyi test on the RMSSE of the six methods, over the series of this run."""
    other = pd.read_csv(others, dtype={"series": str})
    other = other[~other["method"].isin(METHODS)]
    table = pd.concat([series.assign(series=series["series"].astype(str)), other])
    wide = table.pivot(index="series", columns="method", values="RMSSE").loc[series["series"].astype(str).unique()]
    test = rmcb(wide, level=1 - ALPHA, outplot="none")
    ax = test.plot(select="smooth TBATS", main=f"Nemenyi test, RMSSE, {len(wide)} series, level {1 - ALPHA}", ylab="mean rank")
    ax.figure.set_size_inches(8, 5)
    ax.figure.tight_layout()
    ax.figure.savefig(run_dir / "nemenyi.png", dpi=150)
    table.to_csv(run_dir / "measures_all.csv", index=False)
    return [dict(r, conditions={**r["conditions"], "methods": "all six"}) for r in nemenyi(wide, {"split": split})]


def tables(results, run_dir):
    """The results as three tables: per method, the tests, the Nemenyi test."""
    rows = [{"variable": r["variable"], **r["conditions"], "value": r["value"],
             **({f"interval {k}": v for k, v in r["interval"].items()} if "interval" in r else {})} for r in results]
    frame = pd.DataFrame(rows).drop(columns="split")
    nem = frame["variable"].str.contains("Nemenyi|rmcb")
    per_method = frame["method"].notna() & ~nem & ~frame["variable"].str.startswith("relative")
    frame[per_method].dropna(axis=1, how="all").to_csv(run_dir / "table_methods.csv", index=False)
    frame[~per_method & ~nem].dropna(axis=1, how="all").to_csv(run_dir / "table_tests.csv", index=False)
    frame[nem].dropna(axis=1, how="all").to_csv(run_dir / "table_nemenyi.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--series", type=int, default=0, help="first N series; 0: all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--others", required=True, help="per-series measures of PL7's run")
    args = parser.parse_args()
    run_dir = Path(os.environ.get("CLAIMSET_RUN_DIR", "."))
    samples = load(args.data, args.split, args.series)
    with ProcessPoolExecutor(args.workers, initializer=warm_up) as pool:
        rows = [r for series_rows in pool.map(run_series, [(k, v[0]) for k, v in samples.items()])
                for r in series_rows]
    table = pd.DataFrame([measures(r, *samples[r["series"]]) for r in rows])
    table.to_csv(run_dir / "measures_origins.csv", index=False)
    series = per_series(table)
    series.to_csv(run_dir / "measures.csv", index=False)
    undefined = series.loc[~np.isfinite(series["RMSSE"]), "series"].unique()
    series = series[~series["series"].isin(undefined)]
    results = statistics(series, args.split)
    results += all_methods(series, args.others, args.split, run_dir)
    tables(results, run_dir)
    results.append({"variable": "series excluded by undefined RMSSE", "value": len(undefined),
                    "conditions": {"split": args.split}})
    for r in results:
        print(r["variable"], r.get("conditions", {}), round(r["value"], 5), r.get("interval", ""))
    path = os.environ.get("CLAIMSET_RESULTS")
    if path:
        with open(path, "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    main()
