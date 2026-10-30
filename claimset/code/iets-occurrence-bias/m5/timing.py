"""Exploration: time the iETS fits and forecasts in R (adam()) and Python (ADAM) on series
outside the pre-registered subset, and compare their point forecasts and upper quantiles."""

import csv
import json
import os
import random
import subprocess
import sys
import tempfile
import time
import warnings

import numpy as np

from load import H, read_series

LEVELS = [0.8, 0.9, 0.95, 0.99]
MODELS = [("odds-ratio", "ratio", "conventional"), ("odds-ratio", "score", "conventional"),
          ("inverse-odds-ratio", "score", "conventional"), ("general", "score", "conventional"),
          ("auto", "score", "conventional"), ("inverse-odds-ratio", "score", "adam")]
R_CODE = r"""
suppressMessages(library(smooth))
args <- commandArgs(TRUE); y <- scan(args[1], quiet=TRUE)
spec <- read.csv(args[2], stringsAsFactors=FALSE); out <- list()
for(i in seq_len(nrow(spec))){
    t0 <- Sys.time()
    m <- adam(y, "MNN", occurrence=spec$occ[i], distribution="dgamma", h=14, holdout=TRUE,
              proxyOccurrence=spec$proxy[i], ets=spec$ets[i])
    f <- forecast(m, h=14, interval="prediction", level=c(0.8,0.9,0.95,0.99), side="upper")
    out[[i]] <- list(time=as.numeric(Sys.time()-t0, units="secs"), mean=as.numeric(f$mean),
                     upper=as.numeric(f$upper))
}
jsonlite::write_json(out, args[3], digits=NA, auto_unbox=TRUE)
"""


def python_fit(y, occ, proxy, ets):
    from smooth import ADAM
    t0 = time.time()
    m = ADAM(model="MNN", occurrence=occ, distribution="dgamma", h=H, holdout=True,
             proxy_occurrence=proxy, ets=ets).fit(np.asarray(y))
    f = m.predict(h=H, interval="prediction", level=LEVELS, side="upper")
    return {"time": time.time() - t0, "mean": np.asarray(f.mean, float),
            "upper": np.asarray(f.upper, float).T.ravel()}


def main(train, test, subset):
    taken = {int(r["row"]) for r in csv.DictReader(open(subset))}
    rows = random.Random(7).sample([i for i in range(30490) if i not in taken], 10)
    series = read_series(train, test, rows)
    tmp = tempfile.mkdtemp()
    spec = os.path.join(tmp, "spec.csv")
    with open(spec, "w") as f:
        f.write("occ,proxy,ets\n" + "".join(f"{o},{p},{e}\n" for o, p, e in MODELS))
    rscript = os.path.join(tmp, "fit.R")
    open(rscript, "w").write(R_CODE)
    times = {"R": np.zeros(len(MODELS)), "Python": np.zeros(len(MODELS))}
    dmean, dupper = np.zeros(len(MODELS)), np.zeros(len(MODELS))
    for row, (_, _, y) in series.items():
        ypath, rout = os.path.join(tmp, "y.txt"), os.path.join(tmp, "r.json")
        np.savetxt(ypath, y[:-H] + y[-H:])
        subprocess.run(["Rscript", rscript, ypath, spec, rout], check=True)
        r = json.load(open(rout))
        for k, (occ, proxy, ets) in enumerate(MODELS):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p = python_fit(y, occ, proxy, ets)
            times["R"][k] += r[k]["time"]
            times["Python"][k] += p["time"]
            dmean[k] = max(dmean[k], np.max(np.abs(p["mean"] - np.asarray(r[k]["mean"]))))
            dupper[k] = max(dupper[k], np.max(np.abs(p["upper"] - np.asarray(r[k]["upper"]))))
    results = []
    for k, (occ, proxy, ets) in enumerate(MODELS):
        cond = {"occurrence": occ, "proxy": proxy, "ets": ets}
        for lang in times:
            results.append({"quantity": "mean seconds per series, fit and forecast",
                            "value": times[lang][k] / len(series), "unit": "s",
                            "conditions": {**cond, "language": lang}})
        results += [{"quantity": "max abs difference of point forecasts, Python minus R",
                     "value": float(dmean[k]), "conditions": cond},
                    {"quantity": "max abs difference of upper quantiles, Python minus R",
                     "value": float(dupper[k]), "conditions": cond}]
    for r in results:
        print(r)
    path = os.environ.get("CLAIMSET_RESULTS")
    if path:
        json.dump(results, open(path, "w"))


if __name__ == "__main__":
    main(*sys.argv[1:4])
