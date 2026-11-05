"""Q3 and Q4: where the under-coverage sits on the benchmark, location against width, and
ETS against ARIMA, from the forecasts of q3_refit.py and the saved coverage of the 99."""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd

from q1_benchmark_reforecast import load

Z80 = 2 * 1.2815515655446004
TAUS = np.round(np.arange(0.01, 1.0, 0.01), 2)
METHODS = ("auto_back", "auto_opt", "es_back", "force_011", "force_ann", "force_mnn")
I10, I90, IM = 2, 5, 8
RESULTS = []


def out(quantity, value, **conditions):
    RESULTS.append(dict(quantity=quantity, value=float(value), conditions=conditions))


def family(name):
    # The class of a selected ARIMA from its name.
    if name == "Constant level":
        return "constant level"
    found = re.search(r"\((\d),(\d),(\d)\)", name)
    if not found:
        return "other"
    p, d, q = map(int, found.groups())
    seasonal = "[" in name and name.count("(") > 1
    drift = "drift" in name or "constant" in name
    return (f"{'seasonal ' if seasonal else ''}d={d}" + (" drift" if drift else "")
            + (" pure I" if p == 0 and q == 0 and not seasonal else ""))


def frame(series, q, models):
    # One row per observation of the holdout.
    rows = []
    for i, s in enumerate(series):
        y = np.asarray(s.xx, float)
        h = np.arange(1, y.size + 1)
        rows.append(pd.DataFrame(dict(i=i, h=h, y=y, mean=q[i, : y.size, IM], q10=q[i, : y.size, I10],
                                      q90=q[i, : y.size, I90], model=models[i])))
    d = pd.concat(rows, ignore_index=True)
    d["sd"] = (d["q90"] - d["q10"]) / Z80
    d["z"] = (d["y"] - d["mean"]) / d["sd"]
    d["hit"] = (d["y"] >= d["q10"]) & (d["y"] <= d["q90"])
    d["below10"], d["below90"] = d["y"] <= d["q10"], d["y"] <= d["q90"]
    return d


def series_cov(d):
    # The benchmark's weights: the mean within a series, then over the series.
    g = d.groupby("i")
    return g[["hit", "below10", "below90"]].mean().assign(below50=g["z"].apply(lambda z: (z <= 0).mean()))


def table(cov, key, label, saved=None):
    t = cov.groupby(key)[["hit", "below10", "below50", "below90"]].mean()
    t.insert(0, "n", cov.groupby(key).size())
    if saved is not None:
        t["mce"] = [np.abs(saved[cov[key].values == g].mean(axis=0) - TAUS).mean() for g in t.index]
    print(f"\n{label}\n{t.round(3).to_string()}")
    for g, r in t.iterrows():
        out("cov80 of a group", r["hit"], by=label, group=str(g), n=int(r["n"]))
        out("q50 of a group", r["below50"], by=label, group=str(g), n=int(r["n"]))
    return t


def splits(d, label):
    # Location against width on the pooled z, and the share of the error between series.
    z = d["z"].to_numpy()
    z = z[np.isfinite(z)]
    w = d.groupby("i")["z"].transform("mean")
    centred = d["z"] - d["z"].median()
    within = d["z"] - w
    scale = np.nanquantile(np.abs(z), 0.8) / (Z80 / 2)
    vals = dict(
        cov80=np.mean(np.abs(z) <= Z80 / 2),
        cov80_centred_global=np.nanmean(np.abs(centred) <= Z80 / 2),
        cov80_centred_series=np.nanmean(np.abs(within) <= Z80 / 2),
        scale_for_80=scale,
        z_median=np.median(z),
        z_mean_trimmed=np.mean(z[np.abs(z) < 10]),
        z_sd_trimmed=np.std(z[np.abs(z) < 10]),
        share_variance_between_series=np.nanvar(w[np.abs(d["z"]) < 10]) / np.nanvar(d["z"][np.abs(d["z"]) < 10]),
    )
    print(label, {k: round(float(v), 3) for k, v in vals.items()})
    for k, v in vals.items():
        out(k, v, method=label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forecasts", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--saved", required=True)
    ap.add_argument("--saved-ets", required=True)
    ap.add_argument("--sim-cells", required=True)
    ap.add_argument("--pinball", required=True)
    a = ap.parse_args()
    series = load()
    q = np.load(a.forecasts)["q"]
    models = pd.read_csv(a.models)
    saved, saved_ets = np.load(a.saved), np.load(a.saved_ets)
    pd.set_option("display.width", 200)

    meta = pd.DataFrame(dict(
        dataset=["M1"] * 1001 + ["M3"] * 3003 + ["Tourism"] * 1311,
        period=[s.period for s in series], n=[len(s.x) for s in series], h=[s.h for s in series]))
    meta["length"] = pd.cut(meta["n"], [0, 25, 50, 75, 100, 150, 1000])
    frames = {m: frame(series, q[j], models[m].astype(str).values) for j, m in enumerate(METHODS)}

    # The refit reproduces the saved coverage
    for j, (m, sv) in enumerate((("auto_back", saved[0]), ("auto_opt", saved[1]))):
        below90 = frames[m].groupby("i")["below90"].mean().values
        out("mean absolute difference of q90 coverage from the saved array", np.nanmean(np.abs(below90 - sv[:, 89])), method=m)
        print(m, "q90 against saved", np.nanmean(np.abs(below90 - sv[:, 89])))

    # Q3: the breakdown
    d = frames["auto_back"]
    cov = series_cov(d).join(meta)
    cov["family"] = [family(x) for x in models["auto_back"].astype(str)]
    cov["seasonal"] = cov["period"] > 1
    sv = saved[0]
    for key in ("dataset", "period", "length", "family"):
        table(cov, key, key, sv)
    t = d.merge(meta[["dataset", "period"]], left_on="i", right_index=True)
    by_h = t.groupby(pd.cut(t["h"], [0, 1, 3, 6, 12, 24]))[["hit", "below10", "below90"]].mean()
    by_h["below50"] = t.groupby(pd.cut(t["h"], [0, 1, 3, 6, 12, 24]))["z"].apply(lambda z: (z <= 0).mean())
    by_h["z_sd"] = t[np.abs(t["z"]) < 10].groupby(pd.cut(t["h"], [0, 1, 3, 6, 12, 24]))["z"].std()
    print("\nhorizon (pooled over observations)\n", by_h.round(3).to_string())
    for g, r in by_h.iterrows():
        out("cov80 by horizon, pooled over observations", r["hit"], h=str(g))
        out("sd of z below 10 by horizon", r["z_sd"], h=str(g))

    # Location and width
    for m in METHODS:
        splits(frames[m], m)
    for name in ("dataset", "family"):
        for g, idx in cov.groupby(name).groups.items():
            part = d[d["i"].isin(idx)]
            if len(idx) > 100:
                splits(part, f"auto_back {name}={g}")

    # The simulation's coverage at the benchmark's lengths: what selection and estimation
    # alone would give, the lengths below 50 and above 300 at the nearest size.
    cells = pd.read_csv(a.sim_cells)
    sim = cells[cells["case"] == "S_backcasting"].groupby("T")["cov80"].mean()
    expected = np.interp(np.log(meta["n"]), np.log(sim.index.values), sim.values)
    out("cov80 expected from selection and estimation at the benchmark lengths", expected.mean())
    print("\nexpected cov80 of the simulation at the benchmark lengths", round(expected.mean(), 4), sim.round(3).to_dict())
    print("median n", meta["n"].median(), "share n<=50", (meta["n"] <= 50).mean())
    out("median series length", meta["n"].median())
    out("share of series of 50 observations or fewer", (meta["n"] <= 50).mean())

    # The reforecast blow-ups, from PL10
    pin = np.load(a.pinball)
    for j, ini in enumerate(("backcasting", "optimal")):
        ratio = pin[j, :, 1] / pin[j, :, 0]
        for k, name in (("predict", 0), ("reforecast", 1)):
            out("median benchmark pinball", np.nanmedian(pin[j, :, name]), initial=ini, method=k)
        out("series with reforecast pinball over twice predict's", np.nansum(ratio > 2), initial=ini)
        out("series with reforecast pinball over ten times predict's", np.nansum(ratio > 10), initial=ini)
        print(ini, "pinball median", np.nanmedian(pin[j], axis=0).round(4), ">2x", int(np.nansum(ratio > 2)), ">10x", int(np.nansum(ratio > 10)))

    # Q4: ETS against ARIMA, on the same series
    es, ar = series_cov(frames["es_back"]), series_cov(frames["auto_back"])
    width = lambda f, h=None: f[f["h"] == h].set_index("i")["sd"] if h else f.groupby("i")["sd"].mean()
    ratio = (width(frames["es_back"]) / width(frames["auto_back"])).replace([np.inf, -np.inf], np.nan)
    ratio1 = (width(frames["es_back"], 1) / width(frames["auto_back"], 1)).replace([np.inf, -np.inf], np.nan)
    print("\nETS over ARIMA, all series: cov80", round(es["hit"].mean(), 3), round(ar["hit"].mean(), 3),
          "q50", round(es["below50"].mean(), 3), round(ar["below50"].mean(), 3),
          "median width ratio h=1", round(ratio1.median(), 3), "all h", round(ratio.median(), 3))
    out("median width ETS over ARIMA, h=1", ratio1.median())
    out("median width ETS over ARIMA, all horizons", ratio.median())
    ets_names = models["es_back"].astype(str)
    arima_names = models["auto_back"].astype(str)
    is011 = arima_names.str.fullmatch(r"ARIMA\(0,1,1\)")
    for label, mask in (("ETS(ANN) and ARIMA(0,1,1) selected", (ets_names == "ETS(ANN)") & is011),
                        ("ETS(MNN) and ARIMA(0,1,1) selected", (ets_names == "ETS(MNN)") & is011),
                        ("ETS with a trend, ARIMA without drift, d=1", ets_names.str.contains(r"ETS\(.A") & arima_names.str.fullmatch(r"ARIMA\(\d,1,\d\)")),
                        ("seasonal ETS and seasonal ARIMA", ets_names.str.contains(r"[AM]\)$") & arima_names.str.startswith("SARIMA")),
                        ("seasonal ETS, non-seasonal ARIMA", ets_names.str.contains(r"[AM]\)$") & ~arima_names.str.startswith("SARIMA")),
                        ("all series", pd.Series(True, index=ets_names.index))):
        idx = mask.values
        r1, rh = ratio1[idx].median(), ratio[idx].median()
        ce, ca = es["hit"][idx].mean(), ar["hit"][idx].mean()
        print(f"{label}: n={idx.sum()} cov80 ETS {ce:.3f} ARIMA {ca:.3f}; q50 {es['below50'][idx].mean():.3f} {ar['below50'][idx].mean():.3f}; width ratio h=1 {r1:.3f} all {rh:.3f}")
        out("number of series", idx.sum(), subset=label)
        out("cov80 ETS", ce, subset=label)
        out("cov80 ARIMA", ca, subset=label)
        out("median width ratio ETS over ARIMA, all horizons", rh, subset=label)
        out("median width ratio ETS over ARIMA, h=1", r1, subset=label)
    for m in ("force_011", "force_ann", "force_mnn"):
        c = series_cov(frames[m])
        print(m, "all series cov80", round(c["hit"].mean(), 3), "q50", round(c["below50"].mean(), 3))
        out("cov80 of the forced model on all series", c["hit"].mean(), method=m)
        out("q50 of the forced model on all series", c["below50"].mean(), method=m)
    r = (width(frames["force_ann"]) / width(frames["force_011"])).replace([np.inf, -np.inf], np.nan)
    print("force ANN over 011: width ratio median", round(r.median(), 4), "p10, p90", np.nanpercentile(r, [10, 90]).round(3))
    out("median width ratio ETS(ANN) over ARIMA(0,1,1) forced on all series", r.median())
    print("saved q90 ES back", round(np.nanmean(saved_ets[1, :, 89]), 3))

    path = os.environ.get("CLAIMSET_RESULTS")
    if path:
        json.dump(RESULTS, open(path, "w"))


if __name__ == "__main__":
    main()
