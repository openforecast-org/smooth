"""Forecasts of the iETS(M,N,N) models on M5 series, as SB23 section 5: ADAM with
ETS(M,N,N) demand sizes, Gamma distribution, the last 14 days held out, h = 1-14, and the
upper quantiles at 0.80, 0.90, 0.95 and 0.99. Usage:
    forecast_iets.py <sales_train> <sales_test> <rows.csv> <workers>
rows.csv has a column "row" (rows of the sales files). Writes iets_forecasts.csv to
$CLAIMSET_RUN_DIR: row, method, horizon, mean, q0.8, q0.9, q0.95, q0.99, seconds, status.
Series with fewer than 5 non-zero in-sample observations are skipped (status "excluded").
"""

import csv
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np

from load import H, read_series

LEVELS = [0.8, 0.9, 0.95, 0.99]
OCCURRENCE = {"F": "fixed", "D": "direct", "O": "odds-ratio", "I": "inverse-odds-ratio",
              "G": "general", "A": "auto"}
# (method name, occurrence, proxy, ets)
METHODS = [(f"iETS_{k}_conv", OCCURRENCE[k], "ratio", "conventional") for k in "FD"] + [
    (f"iETS_{k}_{proxy}_{e[:4]}", OCCURRENCE[k], proxy, e)
    for e in ("conventional", "adam") for k in "OIGA" for proxy in ("ratio", "score")]


def fit_series(item):
    from smooth import ADAM

    row, y = item
    y = np.asarray(y, float)
    out = []
    if np.count_nonzero(y[:-H]) < 5:
        return [[row, name, 0] + [np.nan] * 5 + [0, "excluded"] for name, *_ in METHODS]
    for k, (name, occ, proxy, ets) in enumerate(METHODS):
        t0 = time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ADAM(model="MNN", occurrence=occ, distribution="dgamma", h=H,
                             holdout=True, proxy_occurrence=proxy, ets=ets).fit(y)
                f = model.predict(h=H, interval="prediction", level=LEVELS, side="upper",
                                  seed=row * 100 + k)
            mean, upper = np.asarray(f.mean, float), np.asarray(f.upper, float)
            status = "ok"
        except Exception as error:  # a failed fit is recorded, not hidden
            mean, upper, status = np.full(H, np.nan), np.full((H, 4), np.nan), repr(error)
        seconds = time.time() - t0
        out += [[row, name, h + 1, mean[h], *upper[h], seconds, status] for h in range(H)]
    return out


def main(train, test, rows_file, workers):
    rows = [int(r["row"]) for r in csv.DictReader(open(rows_file))]
    series = read_series(train, test, rows)
    items = [(row, series[row][2]) for row in rows]
    outdir = os.environ.get("CLAIMSET_RUN_DIR", ".")
    with Pool(int(workers)) as pool, open(os.path.join(outdir, "iets_forecasts.csv"), "w",
                                          newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "method", "horizon", "mean", "q0.8", "q0.9", "q0.95",
                         "q0.99", "seconds", "status"])
        for lines in pool.imap(fit_series, items, chunksize=4):
            writer.writerows(lines)
    print(len(items), "series,", len(METHODS), "methods")


if __name__ == "__main__":
    main(*sys.argv[1:5])
