"""smooth's TBATS against forecast::tbats, AutoTBATS, ADAM ETS and seasonal naive on the
hourly electricity data (GluonTS electricity_nips, 370 series), with the methods, the
RMSSE and the statistics of m4_hourly.py.

Each series is the training series of the archive extended by the last of its test
windows, so that it ends at 2014-09-08 01:00. Rolling origin: 7 daily origins, the
last 48 hours before the end of the series and six more 24, 48, ..., 144 hours earlier
(``--split test``), or the same origins one week earlier (``--split validation``, which
never reads the last week). At each origin each method is fitted to the last 1008 hours
(six weeks; fewer when a series is shorter, which happens only at the validation
origins) and forecasts 48 hours ahead.

Per series and method, the RMSSE (scaled by the root mean squared first difference of
the fitting window) and the interval coverage are averaged over the 7 origins, and the
elapsed time per fit likewise; a series counts as failed for a method when any of its 7
fits failed (each failed fit is replaced by seasonal naive 168, as in m4_hourly.py).
A series whose RMSSE is undefined (a constant fitting window: zero scale) at any origin
is left out of all statistics and counted.
The statistics over the 370 series are those of m4_hourly.py.

Run: python electricity_hourly.py --data electricity_nips.tar.gz --split test
Writes the per-origin and per-series measures and the forecasts into $CLAIMSET_RUN_DIR
and the results to $CLAIMSET_RESULTS.
"""

import argparse
import json
import os
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from m4_hourly import H, measures, run_r, run_series, statistics, warm_up

WINDOW = 1008
ORIGINS = 7


def load(path, split, count):
    """Fitting windows and test values keyed "series@origin" (origin 1 the earliest)."""
    with tarfile.open(path) as archive:

        def rows(part):
            data = archive.extractfile(f"electricity_nips/{part}/data.json")
            return [json.loads(line) for line in data]

        train, test = rows("train"), rows("test")
    ends = {}
    for row in test:
        key = row["feat_static_cat"][0]
        end = pd.Timestamp(row["start"]) + pd.Timedelta(hours=len(row["target"]))
        if key not in ends or end > ends[key][0]:
            ends[key] = (end, row["target"])
    samples = {}
    shift = 168 if split == "validation" else 0
    for row in train[: count or None]:
        key = row["feat_static_cat"][0]
        tail = ends[key][1][-168:]
        y = np.concatenate([row["target"], tail]).astype(float)
        for k in range(ORIGINS):
            origin = len(y) - shift - H - 24 * (ORIGINS - 1 - k)
            fit = y[max(0, origin - WINDOW) : origin]
            samples[f"{key}@{k + 1}"] = (fit, y[origin : origin + H])
    return samples


def per_series(table):
    table = table.assign(series=table["series"].str.split("@").str[0])
    return table.groupby(["series", "method"], as_index=False).agg(
        RMSSE=("RMSSE", "mean"),
        coverage=("coverage", "mean"),
        seconds=("seconds", "mean"),
        failed=("failed", "any"),
        fits_failed=("failed", "sum"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--series", type=int, default=0, help="first N series; 0: all")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    run_dir = Path(os.environ.get("CLAIMSET_RUN_DIR", "."))
    samples = load(args.data, args.split, args.series)
    rows = run_r(samples, run_dir, args.workers)
    jobs = [(k, v[0]) for k, v in samples.items()]
    with ProcessPoolExecutor(args.workers, initializer=warm_up) as pool:
        for series_rows in pool.map(run_series, jobs):
            rows += series_rows
    table = pd.DataFrame([measures(r, *samples[r["series"]]) for r in rows])
    table.to_csv(run_dir / "measures_origins.csv", index=False)
    pd.DataFrame(
        [
            {
                "series": r["series"],
                "method": r["method"],
                "h": i + 1,
                **{
                    p: None if r[p] is None else r[p][i]
                    for p in ("mean", "lower", "upper")
                },
            }
            for r in rows
            for i in range(H)
        ]
    ).to_csv(run_dir / "forecasts.csv", index=False)
    series = per_series(table)
    series.to_csv(run_dir / "measures.csv", index=False)
    undefined = series.loc[~np.isfinite(series["RMSSE"]), "series"].unique()
    series = series[~series["series"].isin(undefined)]
    results = statistics(series, args.split)
    results.append(
        {
            "variable": "series excluded by undefined RMSSE",
            "value": len(undefined),
            "conditions": {"split": args.split},
        }
    )
    results += [
        {
            "variable": "fits failed",
            "value": int(group["fits_failed"].sum()),
            "conditions": {"split": args.split, "method": method},
        }
        for method, group in series.groupby("method")
    ]
    for r in results:
        print(
            r["variable"],
            r.get("conditions", {}),
            round(r["value"], 5),
            r.get("interval", ""),
        )
    path = os.environ.get("CLAIMSET_RESULTS")
    if path:
        with open(path, "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    main()
