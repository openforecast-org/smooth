"""Does the error scale at the origin exceed the in-sample one? Coverage and the sd of z by
terciles of the series' level and of its change scale at the end of the sample; the same for
the competitors and ETS; the null reference of re-centring on the series; ETS(ANN) against
ARIMA(0,1,1) on the same series."""
import argparse
import json
import os

import numpy as np
import pandas as pd

from q1_benchmark_reforecast import load
from q3_analysis import I10, I90, IM, METHODS, Z80

ap = argparse.ArgumentParser()
ap.add_argument("--forecasts", required=True)
ap.add_argument("--models", required=True)
ap.add_argument("--competitors", required=True)
ap.add_argument("--competitor-names", required=True)
ap.add_argument("--saved-ets", required=True)
a = ap.parse_args()
RESULTS = []


def out(quantity, value, **conditions):
    RESULTS.append(dict(quantity=quantity, value=float(value), conditions=conditions))


series = load()
q = np.load(a.forecasts)["q"]
models = pd.read_csv(a.models)
meta = []
for s in series:
    x = np.asarray(s.x, float)
    lag = s.period
    third = max(len(x) // 3, 2)
    d = np.abs(x[lag:] - x[:-lag]) if len(x) > lag + 2 else np.abs(np.diff(x))
    meta.append(dict(level=x[-third:].mean() / x.mean(), change=d[-third:].mean() / d.mean(), n=len(x), period=lag))
meta = pd.DataFrame(meta)
for c in ("level", "change"):
    meta[c + "_t"] = pd.qcut(meta[c], 3, labels=["low", "mid", "high"])


def per_series(j):
    rows = []
    for i, s in enumerate(series):
        y = np.asarray(s.xx, float)
        v = q[j, i, : y.size]
        sd = (v[:, I90] - v[:, I10]) / Z80
        z = (y - v[:, IM]) / sd
        rows.append(dict(cov=np.mean((y >= v[:, I10]) & (y <= v[:, I90])), z1=z[0], zall=np.median(np.abs(z))))
    return pd.DataFrame(rows)


comp = np.load(a.competitors)
names = open(a.competitor_names).read().split("\n")
names = [x for x in names if x.strip()]
print("competitors", comp.shape, names)
ets = np.load(a.saved_ets)
for j, m in enumerate(METHODS):
    r = per_series(j).join(meta)
    for key in ("level_t", "change_t"):
        g = r.groupby(key, observed=True)
        scale1 = g["z1"].apply(lambda z: np.nanquantile(np.abs(z), 0.8) / (Z80 / 2))
        t = pd.DataFrame(dict(cov80=g["cov"].mean(), scale_h1_for_80=scale1))
        print(m, key, t.round(3).to_dict("index"))
        for k, row in t.iterrows():
            out("cov80 by tercile", row["cov80"], method=m, by=key, tercile=str(k))
            out("scale at h=1 needed for 80 by tercile", row["scale_h1_for_80"], method=m, by=key, tercile=str(k))
for k, name in enumerate(names):
    c = comp[k, ..., 1] if comp.ndim == 4 else comp[k]
    cov = c[:, 89] - c[:, 9]
    for key in ("level_t", "change_t"):
        g = pd.Series(cov).groupby(meta[key].values, observed=True).mean()
        print(name, key, g.round(3).to_dict())
        for tt, v in g.items():
            out("cov80 by tercile", v, method=name, by=key, tercile=str(tt))
g = pd.Series(ets[1, :, 89] - ets[1, :, 9]).groupby(meta["level_t"].values, observed=True).mean()
print("ES Back saved level_t", g.round(3).to_dict())

# ETS(ANN) against ARIMA(0,1,1) selected on the same series
e, f = q[METHODS.index("es_back")], q[METHODS.index("auto_back")]
mask = (models["es_back"].astype(str) == "ETS(ANN)") & (models["auto_back"].astype(str) == "ARIMA(0,1,1)")
rows = []
for i in np.where(mask.values)[0]:
    y = np.asarray(series[i].xx, float)
    rows.append(dict(i=i, w=np.mean((e[i, : y.size, I90] - e[i, : y.size, I10]) / (f[i, : y.size, I90] - f[i, : y.size, I10])),
                     dm=np.mean(np.abs(e[i, : y.size, IM] - f[i, : y.size, IM])) / np.mean(f[i, : y.size, I90] - f[i, : y.size, I10]),
                     ce=np.mean((y >= e[i, : y.size, I10]) & (y <= e[i, : y.size, I90])),
                     ca=np.mean((y >= f[i, : y.size, I10]) & (y <= f[i, : y.size, I90]))))
r = pd.DataFrame(rows)
same = (r["w"].sub(1).abs() < 0.01) & (r["dm"] < 0.01)
print("ANN & 011: same forecast and width on", int(same.sum()), "of", len(r), "; cov there", r.loc[same, ["ce", "ca"]].mean().round(3).to_dict(),
      "; elsewhere", r.loc[~same, ["ce", "ca"]].mean().round(3).to_dict(), "width ratio there", r.loc[~same, "w"].quantile([0.1, 0.5, 0.9]).round(3).to_dict())
out("series with the same forecast and width among ETS(ANN) and ARIMA(0,1,1) selected", same.sum())
for col, lab in (("ce", "ETS"), ("ca", "ARIMA")):
    out("cov80 among ETS(ANN) and ARIMA(0,1,1) selected", r.loc[same, col].mean(), subset="same forecast", method=lab)
    out("cov80 among ETS(ANN) and ARIMA(0,1,1) selected", r.loc[~same, col].mean(), subset="different", method=lab)

# The null reference of re-centring: Gaussian RW and IMA, known model, the real h
rng = np.random.default_rng(5)
for name, theta in (("rw", 0.0), ("ima", -0.5)):
    for h in (6, 8, 18):
        e_ = rng.normal(size=(40000, h))
        psi = np.r_[1.0, np.full(h - 1, 1 + theta)]
        y = np.stack([np.convolve(row, psi)[: h] for row in e_[:4000]])
        sd = np.sqrt(np.cumsum(psi**2))
        z = y / sd
        centred = z - z.mean(axis=1, keepdims=True)
        print("null", name, h, "cov80 raw", round(np.mean(np.abs(z) <= Z80 / 2), 3), "centred on the series", round(np.mean(np.abs(centred) <= Z80 / 2), 3))
        out("cov80 re-centred on the series under the true model", np.mean(np.abs(centred) <= Z80 / 2), dgp=name, h=h)
if os.environ.get("CLAIMSET_RESULTS"):
    json.dump(RESULTS, open(os.environ["CLAIMSET_RESULTS"], "w"))
