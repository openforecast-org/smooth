"""Exploration: are the upper quantiles of iETS with ets="adam" simulated? Compares two R fits
with different seeds, two Python fits with different seeds, and R with Python, on series
outside the pre-registered subset."""

import csv
import json
import os
import random
import subprocess
import sys
import tempfile
import warnings

import numpy as np

from load import H, read_series
from timing import LEVELS

R_CODE = r"""
suppressMessages(library(smooth))
args <- commandArgs(TRUE); y <- scan(args[1], quiet=TRUE); out <- list()
for(s in c(1, 2)){
    set.seed(s)
    m <- adam(y, "MNN", occurrence=args[2], distribution="dgamma", h=14, holdout=TRUE,
              proxyOccurrence=args[3], ets=args[4])
    f <- forecast(m, h=14, interval="prediction", level=c(0.8,0.9,0.95,0.99), side="upper")
    out[[s]] <- as.numeric(f$upper)
}
jsonlite::write_json(out, args[5], digits=NA)
"""
CASES = [("inverse-odds-ratio", "score", "adam"), ("inverse-odds-ratio", "ratio", "adam"),
         ("inverse-odds-ratio", "score", "conventional")]


def python_upper(y, occ, proxy, ets, seed):
    from smooth import ADAM
    np.random.seed(seed)
    m = ADAM(model="MNN", occurrence=occ, distribution="dgamma", h=H, holdout=True,
             proxy_occurrence=proxy, ets=ets).fit(np.asarray(y))
    f = m.predict(h=H, interval="prediction", level=LEVELS, side="upper")
    return np.asarray(f.upper, float).T.ravel()


def main(train, test, subset):
    taken = {int(r["row"]) for r in csv.DictReader(open(subset))}
    rows = random.Random(7).sample([i for i in range(30490) if i not in taken], 10)[:3]
    series = read_series(train, test, rows)
    tmp = tempfile.mkdtemp()
    rscript = os.path.join(tmp, "q.R")
    open(rscript, "w").write(R_CODE)
    results = []
    for occ, proxy, ets in CASES:
        d = {"R seed 1 vs R seed 2": 0.0, "Python seed 1 vs Python seed 2": 0.0,
             "Python seed 1 vs R seed 1": 0.0}
        for _, (_, _, y) in series.items():
            ypath, rout = os.path.join(tmp, "y.txt"), os.path.join(tmp, "r.json")
            np.savetxt(ypath, y)
            subprocess.run(["Rscript", rscript, ypath, occ, proxy, ets, rout], check=True)
            r = [np.asarray(v) for v in json.load(open(rout))]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p = [python_upper(y, occ, proxy, ets, s) for s in (1, 2)]
            for key, (a, b) in zip(d, ((r[0], r[1]), (p[0], p[1]), (p[0], r[0]))):
                d[key] = max(d[key], float(np.max(np.abs(a - b))))
        results += [{"quantity": "max abs difference of upper quantiles", "value": v,
                     "conditions": {"occurrence": occ, "proxy": proxy, "ets": ets,
                                    "comparison": k}} for k, v in d.items()]
    for r in results:
        print(r)
    path = os.environ.get("CLAIMSET_RESULTS")
    if path:
        json.dump(results, open(path, "w"))


if __name__ == "__main__":
    main(*sys.argv[1:4])
