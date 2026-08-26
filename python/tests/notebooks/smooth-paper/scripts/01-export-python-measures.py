#!/usr/bin/env python3
"""Export per-series error measures for the Python implementations.

Reads the arrays written by `run_full_benchmark.py` (in the smooth repository
under python/tests/notebooks/) and writes one CSV per model class, with one row
per series and one column per method.

  RMSSE   : point accuracy, as reported in Tables 6 and 7 of the paper
  pinball : scaled pinball loss averaged over the 99 quantile levels

Only the competitor columns are taken from Python; the smooth rows of the
paper's tables come from R, where the two implementations agree (Section 5.1).

Usage:  python3 01-export-python-measures.py [<notebooks-dir>] [<output-dir>]
"""
import glob
import os
import sys

import numpy as np

NOTEBOOKS = sys.argv[1] if len(sys.argv) > 1 else (
    "/home/config/Python/Libraries/openforecast-org/smooth/python/tests/notebooks"
)
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "Data"
)

KEEP = {
    "ets": ["statsforecast AutoETS", "sktime AutoETS", "skforecast AutoETS", "aeon AutoETS"],
    # aeon's AutoARIMA has no seasonal period, so it cannot be compared fairly
    # against seasonal searches on this mostly-seasonal collection; pmdarima
    # fails on 42 series and so is scored on a different sample.  Both are
    # excluded from the ARIMA table and from the MCB test.
    "arima": ["statsforecast AutoARIMA", "skforecast Arima"],
}


def latest(pattern):
    hits = sorted(glob.glob(os.path.join(NOTEBOOKS, pattern)))
    if not hits:
        raise SystemExit("no file matching %s in %s" % (pattern, NOTEBOOKS))
    return hits[-1]


def write_csv(path, names, columns):
    n = len(columns[0])
    with open(path, "w") as f:
        f.write(",".join('"%s"' % s for s in names) + "\n")
        for i in range(n):
            f.write(",".join(
                "" if not np.isfinite(c[i]) else repr(float(c[i])) for c in columns
            ) + "\n")
    print("wrote %s  (%d series x %d methods)" % (path, n, len(names)))


# run_full_benchmark.py tags its output with the subset that produced it, so the
# ETS competitors and the ARIMA competitors live in different arrays.
SOURCE = {"ets": "ets", "arima": "arima"}

if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    for tag, keep in KEEP.items():
        src = SOURCE[tag]
        point = np.load(latest("*-benchmark-%s-point.npy" % src))
        dist = np.load(latest("*-benchmark-%s-dist.npy" % src))
        with open(latest("*-benchmark-%s-methods.txt" % src)) as f:
            names = [l.rstrip("\n").split("\t")[0] for l in f if l.strip()]
        missing = [n for n in keep if n not in names]
        if missing:
            raise SystemExit("methods missing from the %s arrays: %s" % (src, missing))
        idx = [names.index(n) for n in keep]
        rmsse = [point[j, :, 0] for j in idx]
        pinball = [np.nanmean(dist[j, :, :, 0], axis=1) for j in idx]
        write_csv(os.path.join(OUTDIR, "python-%s-rmsse.csv" % tag), keep, rmsse)
        write_csv(os.path.join(OUTDIR, "python-%s-pinball.csv" % tag), keep, pinball)
