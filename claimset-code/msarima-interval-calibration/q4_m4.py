"""Confirmatory on data outside the benchmark: M4 series, 1500 each of yearly, quarterly and
monthly, drawn with a fixed seed. Coverage by tercile of the recent level over the mean."""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from q1_simulate import I10, I90, TAUS
from q3_multiplicative import one


def sample(per_period, seed):
    from fcompdata import load_m4

    rng = np.random.default_rng(seed)
    chosen = []
    for frequency in ("yearly", "quarterly", "monthly"):
        data = load_m4(frequency)
        keys = list(data.keys())
        chosen += [data[keys[k]] for k in np.sort(rng.choice(len(keys), per_period, replace=False))]
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--per-period", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=20261105)
    a = ap.parse_args()
    series = sample(a.per_period, a.seed)
    tasks = [("m4", i, m, np.asarray(s.x, float), np.asarray(s.xx, float), s.period, s.h)
             for m in a.methods for i, s in enumerate(series)]
    with Pool(a.workers) as pool:
        res = pd.DataFrame(pool.map(one, tasks, chunksize=8))
    run_dir = os.environ.get("CLAIMSET_RUN_DIR", ".")
    res.drop(columns="cov").to_csv(os.path.join(run_dir, "m4_series.csv.gz"), index=False)
    print(res[res["model"].str.match("ERROR|TIMEOUT")]["model"].str[:80].value_counts().head(4).to_dict())
    results = []
    terciles = pd.qcut(res[res["name"] == a.methods[0]].set_index("i")["level"], 3, labels=["low", "mid", "high"])
    for name, g in res.groupby("name"):
        failed = g["pinball"].isna().sum()
        g = g[g["pinball"].notna()].copy()
        cov = np.stack(g["cov"].values)
        g["cov80"] = cov[:, I90] - cov[:, I10]
        by = g.groupby(terciles.reindex(g["i"]).values, observed=True)["cov80"].mean()
        c = cov.mean(axis=0)
        vals = dict(cov80=c[I90] - c[I10], q50=c[49], mce=np.abs(c - TAUS).mean(), pinball_median=g["pinball"].median(),
                    pinball_mean=g["pinball"].mean(), failed=failed, cov80_low=by["low"], cov80_mid=by["mid"], cov80_high=by["high"],
                    fall_low_to_high=by["low"] - by["high"], spread=by.max() - by.min())
        print(name, {k: round(float(v), 4) for k, v in vals.items()})
        results += [dict(quantity=f"M4 {k}", value=float(v), conditions=dict(method=name)) for k, v in vals.items()]
    if os.environ.get("CLAIMSET_RESULTS"):
        json.dump(results, open(os.environ["CLAIMSET_RESULTS"], "w"))


if __name__ == "__main__":
    main()
