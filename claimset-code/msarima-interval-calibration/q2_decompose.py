"""Q2: what differs between the stages of RN16 (1.0.8), RN17 (1.3.0, its parameters), RN18
(1.3.0, its orders) and RN19 (1.3.0). Width, point forecasts and scale, per initial."""
import json
import os
import sys

import numpy as np
import pandas as pd

L = sys.argv[1]
STAGES = dict(A="RN16", D="RN17", B="RN18", C="RN19")
data = {k: np.load(f"{L}/{v}/q2_forecasts.npz") for k, v in STAGES.items()}
fits = {k: json.load(open(f"{L}/{v}/q2_fits.json")) for k, v in STAGES.items()}
results = []
I = {"backcasting": 0, "optimal": 1}


def out(q, v, **c):
    results.append(dict(quantity=q, value=float(v), conditions=c))


def compare(a, b, j):
    qa, qb = data[a]["q"][j], data[b]["q"][j]
    wa, wb = qa[..., 1] - qa[..., 0], qb[..., 1] - qb[..., 0]
    with np.errstate(all="ignore"):
        r1, ra = wb[:, 0] / wa[:, 0], np.nanmean(wb, axis=1) / np.nanmean(wa, axis=1)
    moved = np.nanmean(np.abs(qb[..., 2] - qa[..., 2]), axis=1) / np.nanmean(wa, axis=1)
    sa = np.array([f["sigma"] for f in fits[a][j]])
    sb = np.array([f["sigma"] for f in fits[b][j]])
    with np.errstate(all="ignore"):
        sigma = sb / sa
    ca, cb = data[a]["cov"][j], data[b]["cov"][j]
    dc = (cb[:, 89] - cb[:, 9]) - (ca[:, 89] - ca[:, 9])
    return dict(width_h1=np.nanmedian(r1), width_all=np.nanmedian(ra), share_width_narrower=np.nanmean(ra < 0.995),
                share_width_wider=np.nanmean(ra > 1.005), sigma=np.nanmedian(sigma), forecast_moved=np.nanmedian(moved),
                share_forecast_moved=np.nanmean(moved > 0.01), dcov80=np.nanmean(dc))


for ini, j in I.items():
    for a, b in (("A", "D"), ("D", "B"), ("B", "C"), ("A", "C")):
        v = compare(a, b, j)
        print(ini, f"{a}->{b}", {k: round(float(x), 4) for k, x in v.items()})
        for k, x in v.items():
            out(k, x, initial=ini, stage=f"{a}->{b}")
    names = {k: pd.Series([f["model"] for f in fits[k][j]]) for k in STAGES}
    same = names["A"] == names["C"]
    cov = {k: data[k]["cov"][j][:, 89] - data[k]["cov"][j][:, 9] for k in STAGES}
    print(ini, "same model A and C", round(same.mean(), 3), "cov80 there", round(np.nanmean(cov["A"][same]), 4), round(np.nanmean(cov["C"][same]), 4),
          "elsewhere", round(np.nanmean(cov["A"][~same]), 4), round(np.nanmean(cov["C"][~same]), 4))
    out("share of series selecting the same model", same.mean(), initial=ini)
    for k in ("A", "C"):
        out("cov80 on the series selecting the same model", np.nanmean(cov[k][same]), initial=ini, stage=k)
        out("cov80 on the series selecting another model", np.nanmean(cov[k][~same]), initial=ini, stage=k)
        pure = names[k].str.fullmatch(r"ARIMA\(0,1,0\)( with drift)?|Constant level")
        print(ini, k, "share of random walk or constant", round(pure.mean(), 3), "drift share", round(names[k].str.contains("drift").mean(), 3), names[k].value_counts().head(4).to_dict())
        out("share of ARIMA(0,1,0) or constant level", pure.mean(), initial=ini, stage=k)
        out("share with drift", names[k].str.contains("drift").mean(), initial=ini, stage=k)
    moved = names["C"].str.contains("drift").astype(int) - names["A"].str.contains("drift").astype(int)
    print(ini, "gained drift", int((moved > 0).sum()), "lost", int((moved < 0).sum()), "cov80 change where gained", round(np.nanmean(cov["C"][moved > 0] - cov["A"][moved > 0]), 4),
          "lost", round(np.nanmean(cov["C"][moved < 0] - cov["A"][moved < 0]), 4))
    out("series gaining drift", (moved > 0).sum(), initial=ini)
    out("series losing drift", (moved < 0).sum(), initial=ini)
if os.environ.get("CLAIMSET_RESULTS"):
    json.dump(results, open(os.environ["CLAIMSET_RESULTS"], "w"))
