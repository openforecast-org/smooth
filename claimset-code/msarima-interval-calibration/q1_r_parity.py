"""The intervals of R and Python on a sample of the benchmark: the same orders in both, the
width and the point forecast; and R's auto.msarima against AutoMSARIMA."""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

from q1_benchmark_reforecast import load
from q2_forecasts import parse

ROOT, FITS, FORECASTS = sys.argv[1:4]
series = load()
fits = json.load(open(FITS))[0]
q = np.load(FORECASTS)["q"][0]
rng = np.random.default_rng(41)
idx = np.sort(rng.choice(len(series), 500, replace=False))
inputs = []
for i in idx:
    s = series[i]
    p, d, m, c = parse(fits[i]["model"])
    inputs.append(dict(i=int(i), x=np.asarray(s.x, float).tolist(), xx=np.asarray(s.xx, float).tolist(), h=int(s.h), period=int(s.period),
                       ar=p, d=d, ma=m, constant=bool(c), model=fits[i]["model"]))
R = r"""
suppressMessages(pkgload::load_all("%s", quiet=TRUE))
a <- jsonlite::fromJSON("%s", simplifyVector=FALSE)
out <- lapply(a, function(e){
  x <- unlist(e$x); xx <- unlist(e$xx); ar <- unlist(e$ar); d <- unlist(e$d); ma <- unlist(e$ma)
  lags <- if(length(ar)>1) c(1, e$period) else 1
  m <- msarima(x, orders=list(ar=ar, i=d, ma=ma), lags=lags, constant=e$constant, initial="backcasting")
  f <- forecast(m, h=e$h, interval="prediction", level=0.8, side="both")
  auto <- auto.msarima(ts(x, frequency=e$period), lags=if(e$period>1) c(1, e$period) else 1, initial="backcasting")
  g <- forecast(auto, h=e$h, interval="prediction", level=0.8, side="both")
  list(i=e$i, interval=f$interval, mean=as.numeric(f$mean), lower=as.numeric(f$lower), upper=as.numeric(f$upper),
       auto=auto$model, cov=mean(xx>=as.numeric(g$lower) & xx<=as.numeric(g$upper)))
})
jsonlite::write_json(out, "%s", auto_unbox=TRUE, digits=NA)
"""
with tempfile.TemporaryDirectory() as tmp:
    a, b = os.path.join(tmp, "in.json"), os.path.join(tmp, "out.json")
    json.dump(inputs, open(a, "w"))
    run = subprocess.run(["Rscript", "-e", R % (ROOT, a, b)], capture_output=True, text=True, cwd=tmp)
    print(run.stderr[-500:])
    rows = json.load(open(b))
res = []
for e in rows:
    i = e["i"]
    h = series[i].h
    w = q[i, : h, 1] - q[i, : h, 0]
    wr = (np.array(e["upper"]) - np.array(e["lower"])) / w
    y = np.asarray(series[i].xx, float)
    res.append(dict(i=i, ratio=np.median(wr), mean_diff=np.median(np.abs(np.array(e["mean"]) - q[i, : h, 2]) / w), interval=e["interval"],
                    same=e["auto"] == fits[i]["model"], cov_r=e["cov"], cov_py=np.mean((y >= q[i, : h, 0]) & (y <= q[i, : h, 1]))))
r = pd.DataFrame(res)
vals = dict(median_width_ratio_R_over_Python=r["ratio"].median(), p05=r["ratio"].quantile(0.05), p95=r["ratio"].quantile(0.95),
            median_forecast_diff=r["mean_diff"].median(), share_same_auto=r["same"].mean(), cov80_R_auto=r["cov_r"].mean(), cov80_Python_auto=r["cov_py"].mean())
print({k: round(float(v), 4) for k, v in vals.items()}, r["interval"].value_counts().to_dict())
if os.environ.get("CLAIMSET_RESULTS"):
    json.dump([dict(quantity=k, value=float(v), conditions=dict(sample=500)) for k, v in vals.items()], open(os.environ["CLAIMSET_RESULTS"], "w"))
