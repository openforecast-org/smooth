"""auto.adam(model="NNN") choosing between the normal and the log-normal error by AICc, against
auto.msarima, in R: Python's AutoADAM drops dlnorm for a pure ARIMA whatever the user asks.
The 99 quantiles of each series are kept, for the calibration curves."""
import argparse
import json
import os
import subprocess
import tempfile

import numpy as np
import pandas as pd

from q1_benchmark_reforecast import load
from q1_simulate import I10, I90, TAUS

ROOT = "/home/config/Misc/Python/Libraries/openforecast-org/smooth"
R = r"""
suppressMessages(pkgload::load_all("%s", quiet=TRUE))
a <- jsonlite::fromJSON("%s", simplifyVector=FALSE)
fit <- function(e, method){
  x <- ts(unlist(e$x), frequency=e$period)
  lags <- if(e$period > 1) c(1, e$period) else 1
  m <- if(method == "msarima") auto.msarima(x, lags=lags, initial="backcasting") else
    auto.adam(x, "NNN", lags=lags, distribution=c("dnorm", "dlnorm"), initial="backcasting")
  f <- forecast(m, h=e$h, interval="prediction", level=seq(0.01, 0.99, 0.01), side="upper")
  list(model=m$model, distribution=m$distribution, q=unname(as.matrix(f$upper)))
}
out <- parallel::mclapply(a, function(e){
  setTimeLimit(elapsed=300, transient=TRUE)
  r <- tryCatch(fit(e, e$method), error=function(err) list(model=paste("ERROR", conditionMessage(err))))
  c(list(i=e$i, method=e$method), r)
}, mc.cores=%d, mc.preschedule=FALSE)
jsonlite::write_json(out, "%s", auto_unbox=TRUE, digits=NA, matrix="rowmajor")
"""


def m4_sample(per_period, seed, exclude_seed):
    # Series of M4 not in the sample of exclude_seed (PL12 and PL13), drawn with seed
    from fcompdata import load_m4

    old, new = np.random.default_rng(exclude_seed), np.random.default_rng(seed)
    chosen = []
    for frequency in ("yearly", "quarterly", "monthly"):
        data = load_m4(frequency)
        keys = list(data.keys())
        rest = np.setdiff1d(np.arange(len(keys)), old.choice(len(keys), per_period, replace=False))
        chosen += [data[keys[k]] for k in np.sort(new.choice(rest, per_period, replace=False))]
    return chosen


def summarise(res, cov, label):
    results = []
    res["cov80"] = cov[:, I90] - cov[:, I10]
    res["tercile"] = pd.qcut(res["level"], 3, labels=["low", "mid", "high"])
    for name, g in res.groupby("method"):
        ok = g["pinball"].notna()
        c = cov[g.index[ok]].mean(axis=0)
        by = g[ok].groupby("tercile", observed=True)["cov80"].mean()
        vals = dict(n=ok.sum(), failed=(~ok).sum(), cov80=c[I90] - c[I10], q10=c[I10], q50=c[49], q90=c[I90],
                    mce=np.abs(c - TAUS).mean(), pinball_mean=g["pinball"].mean(), pinball_median=g["pinball"].median(),
                    cov80_low=by["low"], cov80_mid=by["mid"], cov80_high=by["high"], spread=by.max() - by.min(),
                    share_dlnorm=(g["distribution"] == "dlnorm").mean())
        for dist, gd in g[ok].groupby("distribution"):
            cd = cov[gd.index].mean(axis=0)
            vals[f"cov80_when_{dist}"], vals[f"mce_when_{dist}"] = cd[I90] - cd[I10], np.abs(cd - TAUS).mean()
        print(label, name, {k: round(float(v), 4) for k, v in vals.items()})
        results += [dict(quantity=f"{label} {k}", value=float(v), conditions=dict(method=name)) for k, v in vals.items()]
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=("benchmark", "m4"), required=True)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--per-period", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--exclude-seed", type=int, default=20261105)
    a = ap.parse_args()
    series = load() if a.data == "benchmark" else m4_sample(a.per_period, a.seed, a.exclude_seed)
    series = series[: a.limit] if a.limit else series
    methods = ("msarima", "adam_select")
    inputs = [dict(i=i, method=m, x=np.asarray(s.x, float).tolist(), h=int(s.h), period=int(s.period))
              for m in methods for i, s in enumerate(series)]
    with tempfile.TemporaryDirectory() as tmp:
        fin, fout = os.path.join(tmp, "in.json"), os.path.join(tmp, "out.json")
        json.dump(inputs, open(fin, "w"))
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        run = subprocess.run(["Rscript", "-e", R % (ROOT, fin, a.workers, fout)], capture_output=True, text=True, cwd=tmp, env=env)
        print(run.stderr[-500:])
        rows = json.load(open(fout))

    cov = np.full((len(rows), TAUS.size), np.nan)
    res = []
    for k, e in enumerate(rows):
        s = series[e["i"]]
        x, y = np.asarray(s.x, float), np.asarray(s.xx, float)
        r = dict(i=e["i"], method=e["method"], model=e["model"], distribution=e.get("distribution", ""), pinball=np.nan,
                 level=x[-max(len(x) // 3, 2):].mean() / x.mean(), n=len(x), period=s.period)
        if "q" in e:
            Q = np.array(e["q"], float).reshape(len(y), TAUS.size)
            diff = y[:, None] - Q
            r["pinball"] = np.maximum(TAUS * diff, (TAUS - 1) * diff).mean() / np.mean(np.abs(np.diff(x)))
            cov[k] = (y[:, None] <= Q).mean(axis=0)
        res.append(r)
    res = pd.DataFrame(res)
    print(res[res["model"].str.startswith("ERROR")]["model"].str[:90].value_counts().head(4).to_dict())
    run_dir = os.environ.get("CLAIMSET_RUN_DIR", ".")
    res.to_csv(os.path.join(run_dir, f"q5_{a.data}_series.csv.gz"), index=False)
    np.save(os.path.join(run_dir, f"q5_{a.data}_coverage.npy"), cov)
    results = summarise(res, cov, a.data)
    if os.environ.get("CLAIMSET_RESULTS"):
        json.dump(results, open(os.environ["CLAIMSET_RESULTS"], "w"))


if __name__ == "__main__":
    main()
