"""Q1: the predictive distribution of MSARIMA (99 quantiles) when the model is known.

Cases, as <kind>_<initial>[_sim|_ref]:
  T: true orders and parameters.
  E: scenario a, the DGP is known and its parameters are estimated.
  S: scenario b, the DGP is unknown: AutoMSARIMA selects the orders and estimates.
  W: scenario c, the DGP is unknown and the model is wrong (WRONG below).
_sim is interval="simulated" and _ref is reforecast() on the same fit as the plain case.

The data, the oracle's point forecasts and the theoretical h-step variance come from the
polynomials below, independently of smooth. Writes the coverage per cell, case and horizon
to $CLAIMSET_RUN_DIR and the pooled numbers to $CLAIMSET_RESULTS.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import re
import signal
import warnings
import zlib
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student

HMAX, SIGMA, MEAN, BURN = 18, 10.0, 1000.0, 300
TAUS = np.round(np.arange(0.01, 1.0, 0.01), 2)
TWO = [round(1 - 2 * t, 2) for t in TAUS if t < 0.5]
TIMEOUT, NSIM_REF = 120, 300
I10, I50, I90, I01, I99 = 9, 49, 89, 0, 98


def dgp(lag=1, o=(0, 0, 0), s=(0, 0, 0), ar=(), ma=(), sar=(), sma=(), drift=None, df=None):
    return dict(lag=lag, o=o, s=s, ar=list(ar), ma=list(ma), sar=list(sar), sma=list(sma),
                drift=drift, df=df)


# ar, ma: in smooth's signs, y_t = ar y_{t-1} + ... + e_t + ma e_{t-1}. drift: the constant
# of the differenced model in the units of SIGMA. df: Student's t innovations, else normal.
DGPS = {
    "rw": dgp(o=(0, 1, 0)),
    "ima": dgp(o=(0, 1, 1), ma=[-0.5]),
    "ar1": dgp(o=(1, 0, 0), ar=[0.7]),
    "ma1": dgp(o=(0, 0, 1), ma=[0.5]),
    "arima111": dgp(o=(1, 1, 1), ar=[0.5], ma=[-0.3]),
    "ima22": dgp(o=(0, 2, 2), ma=[-1.2, 0.32]),
    "rwd": dgp(o=(0, 1, 0), drift=0.5),
    "imad": dgp(o=(0, 1, 1), ma=[-0.5], drift=0.5),
    "airline12": dgp(12, (0, 1, 1), (0, 1, 1), ma=[-0.4], sma=[-0.6]),
    "airline4": dgp(4, (0, 1, 1), (0, 1, 1), ma=[-0.4], sma=[-0.6]),
    "sar12": dgp(12, (1, 0, 0), (1, 0, 0), ar=[0.5], sar=[0.6]),
    "ima_t4": dgp(o=(0, 1, 1), ma=[-0.5], df=4),
}
CORE = [k for k in DGPS if k != "ima_t4"]

# Scenario c: the wrong model for each DGP as (ar, i, ma, constant), without seasonality.
WRONG = {
    "rw": (1, 0, 0, True),          # not differenced
    "ima": (0, 1, 0, False),        # the MA missing
    "ar1": (0, 1, 0, False),        # over-differenced
    "ma1": (0, 1, 1, False),        # over-differenced
    "arima111": (0, 1, 0, False),   # the ARMA part missing
    "ima22": (0, 1, 1, False),      # one difference missing
    "rwd": (0, 1, 0, False),        # the drift missing
    "imad": (0, 1, 1, False),       # the drift missing
    "airline12": (0, 1, 1, False),  # the seasonality missing
    "airline4": (0, 1, 1, False),   # the seasonality missing
    "sar12": (1, 0, 0, True),       # the seasonality missing
    "ima_t4": (0, 1, 1, False),     # the right orders, normal instead of heavy tails
}


def polys(d):
    # The expanded AR side (with the differences), the MA side and the AR side alone.
    ar = np.array([1.0] + [-a for a in d["ar"]])
    ma = np.array([1.0] + d["ma"])
    if d["sar"]:
        seasonal = np.zeros(len(d["sar"]) * d["lag"] + 1)
        seasonal[0] = 1.0
        seasonal[d["lag"]:: d["lag"]] = [-a for a in d["sar"]]
        ar = np.convolve(ar, seasonal)
    if d["sma"]:
        seasonal = np.zeros(len(d["sma"]) * d["lag"] + 1)
        seasonal[0] = 1.0
        seasonal[d["lag"]:: d["lag"]] = d["sma"]
        ma = np.convolve(ma, seasonal)
    stationary_ar = ar.copy()
    for _ in range(d["o"][1]):
        ar = np.convolve(ar, [1.0, -1.0])
    if d["s"][1]:
        diff = np.zeros(d["lag"] + 1)
        diff[[0, d["lag"]]] = [1.0, -1.0]
        ar = np.convolve(ar, diff)
    return ar, ma, stationary_ar


def psi_weights(ar, ma, n):
    psi = np.zeros(n)
    for j in range(n):
        psi[j] = ma[j] if j < len(ma) else 0.0
        psi[j] -= sum(ar[i] * psi[j - i] for i in range(1, min(j, len(ar) - 1) + 1))
    return psi


def constant_of(d, stationary_ar):
    # The drift when differenced, else the constant of a process with the mean MEAN.
    if d["drift"] is not None:
        return d["drift"] * SIGMA
    if d["o"][1] == 0 and d["s"][1] == 0:
        return MEAN * stationary_ar.sum()
    return 0.0


def innovations(d, size, rng):
    if d["df"] is None:
        return rng.normal(0, SIGMA, size)
    return rng.standard_t(d["df"], size) * SIGMA / np.sqrt(d["df"] / (d["df"] - 2))


def simulate(d, n, rng):
    # y_1..y_n, the future values and the oracle's point forecasts, from the true past.
    ar, ma, stationary_ar = polys(d)
    c = constant_of(d, stationary_ar)
    total, start = BURN + n + HMAX, BURN + n
    e = innovations(d, total, rng)
    y = np.full(total, MEAN)
    p, q = len(ar), len(ma)
    for t in range(max(p, q), total):
        y[t] = c - ar[1:] @ y[t - p + 1 : t][::-1] + ma[1:] @ e[t - q + 1 : t][::-1] + e[t]
    oracle = np.zeros(total)
    for t in range(start, total):
        mean = c
        for i in range(1, p):
            mean -= ar[i] * (y[t - i] if t - i < start else oracle[t - i])
        for j in range(1, q):
            if t - j < start:
                mean += ma[j] * e[t - j]
        oracle[t] = mean
    return y[BURN:start], y[start:], oracle[start:]


def spec_of(d):
    seasonal = d["lag"] > 1
    lags = [1, d["lag"]] if seasonal else [1]
    orders = {"ar": [d["o"][0]] + ([d["s"][0]] if seasonal else []),
              "i": [d["o"][1]] + ([d["s"][1]] if seasonal else []),
              "ma": [d["o"][2]] + ([d["s"][2]] if seasonal else [])}
    arma = {"ar": d["ar"] + d["sar"], "ma": d["ma"] + d["sma"]}
    has_constant = d["drift"] is not None or (d["o"][1] == 0 and d["s"][1] == 0)
    return lags, orders, arma, has_constant


def fit_model(kind, initial, name, x):
    from smooth import AutoMSARIMA, MSARIMA

    d = DGPS[name]
    lags, orders, arma, has_constant = spec_of(d)
    if kind == "S":
        return AutoMSARIMA(lags=lags, initial=initial).fit(x)
    if kind == "W":
        p, i, q, constant = WRONG[name]
        return MSARIMA(orders={"ar": [p], "i": [i], "ma": [q]}, lags=[1], initial=initial,
                       constant=constant).fit(x)
    if kind == "T":
        c = constant_of(d, polys(d)[2])
        return MSARIMA(orders=orders, lags=lags, initial=initial,
                       arma={k: v for k, v in arma.items() if v},
                       constant=float(c) if has_constant else False).fit(x)
    return MSARIMA(orders=orders, lags=lags, initial=initial, constant=has_constant).fit(x)


def column_level(c):
    # Predict names the bounds by their levels, and reforecast by the percentages.
    found = re.search(r"([\d.]+)%", str(c))
    return round(float(found.group(1)) / 100 if found else float(c), 2)


def quantiles(fc):
    # The 99 quantiles from the two-sided intervals, the point forecast being the median.
    mean = np.asarray(fc.mean, float).ravel()
    lower, upper = np.asarray(fc.lower, float), np.asarray(fc.upper, float)
    lo = {column_level(c): i for i, c in enumerate(fc.lower.columns)}
    up = {column_level(c): i for i, c in enumerate(fc.upper.columns)}
    Q = np.empty((mean.size, TAUS.size))
    for k, t in enumerate(TAUS):
        if k == I50:
            Q[:, k] = mean
        else:
            Q[:, k] = lower[:, lo[t]] if t < 0.5 else upper[:, up[t]]
    return mean, np.sort(Q, axis=1)


def predict_case(model, suffix, seed):
    if suffix == "ref":
        try:
            return model.reforecast(h=HMAX, interval="prediction", level=TWO, side="both",
                                    nsim=NSIM_REF, seed=seed)
        except ValueError as exc:
            # No estimated parameter, nothing to draw: reforecast() raises (reported), and
            # its parameter uncertainty is none, so the interval is predict()'s.
            if "size 0" not in str(exc):
                raise
    kind = "simulated" if suffix == "sim" else "prediction"
    return model.predict(h=HMAX, interval=kind, level=TWO, side="both")


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


def record(future, oracle, sd_h, mean, Q, model="", dispatched=""):
    diff = future[:, None] - Q
    pinball = np.maximum(TAUS * diff, (TAUS - 1) * diff).mean() / SIGMA
    return dict(hit=(future[:, None] <= Q).astype(float), pinball=pinball,
                width=(Q[:, I90] - Q[:, I10]) / (2 * norm.ppf(0.9) * sd_h),
                loc=(mean - oracle) / sd_h, model=model, dispatched=dispatched, failed=0)


def one(args):
    # One replicate of one cell, with the fits shared between the cases of one fit.
    warnings.filterwarnings("ignore")
    name, n, rep, seed, cases = args
    d = DGPS[name]
    rng = np.random.default_rng(np.random.SeedSequence([seed, zlib.crc32(name.encode()), n, rep]))
    np.random.seed(int(rng.integers(2**31)))
    ref_seed = int(rng.integers(2**31))
    x, future, oracle = simulate(d, n, rng)
    ar, ma, _ = polys(d)
    sd_h = SIGMA * np.sqrt(np.cumsum(psi_weights(ar, ma, HMAX) ** 2))

    quant = norm.ppf(TAUS)
    out = {"oracle": record(future, oracle, sd_h, oracle, oracle[:, None] + sd_h[:, None] * quant[None, :], "oracle")}
    fits, seconds = {}, {}
    signal.signal(signal.SIGALRM, _alarm)
    for case in cases:
        parts = case.split("_")
        kind, initial, suffix = parts[0], parts[1], parts[2] if len(parts) > 2 else ""
        rec = dict(hit=np.full((HMAX, TAUS.size), np.nan), width=np.full(HMAX, np.nan),
                   loc=np.full(HMAX, np.nan), pinball=np.nan, model="", dispatched="", failed=1)
        signal.alarm(TIMEOUT)
        try:
            if (kind, initial) not in fits:
                fits[(kind, initial)] = fit_model(kind, initial, name, x)
            model = fits[(kind, initial)]
            fc = predict_case(model, suffix, ref_seed)
            mean, Q = quantiles(fc)
            signal.alarm(0)
            rec = record(future, oracle, sd_h, mean, Q, model.model_name, fc.interval)
        except Timeout:
            rec["model"] = "TIMEOUT"
        except Exception as exc:
            rec["model"] = f"ERROR {type(exc).__name__}: {exc}"[:150]
        finally:
            signal.alarm(0)
        out[case] = rec
    return name, n, rep, out


def summaries(res):
    # Per replicate the scalars; per cell, case and horizon the mean coverage of the 99.
    reps, hits = [], {}
    for name, n, rep, out in res:
        for case, r in out.items():
            h = r["hit"]
            reps.append(dict(dgp=name, T=n, rep=rep, case=case, model=r["model"], failed=r["failed"],
                             dispatched=r["dispatched"], pinball=r["pinball"],
                             cov80=np.nanmean(h[:, I90] - h[:, I10]), cov98=np.nanmean(h[:, I99] - h[:, I01]),
                             q10=np.nanmean(h[:, I10]), q50=np.nanmean(h[:, I50]), q90=np.nanmean(h[:, I90]),
                             width=np.nanmean(r["width"]), width1=r["width"][0], widthH=r["width"][-1],
                             loc=np.nanmean(r["loc"])))
            hits.setdefault((name, n, case), []).append(h)
    return pd.DataFrame(reps), {k: np.nanmean(np.stack(v), axis=0) for k, v in hits.items()}


def mce(coverage):
    return float(np.abs(coverage - TAUS).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sizes", type=int, nargs="+", default=[50, 100, 300])
    ap.add_argument("--dgps", nargs="+", default=CORE)
    ap.add_argument("--cases", nargs="+", required=True)
    a = ap.parse_args()
    tasks = [(name, n, rep, a.seed, a.cases) for name in a.dgps for n in a.sizes for rep in range(a.n)]
    with Pool(a.workers) as pool:
        res = pool.map(one, tasks, chunksize=4)
    reps, hits = summaries(res)

    run_dir = os.environ.get("CLAIMSET_RUN_DIR", ".")
    reps.to_csv(os.path.join(run_dir, "q1_replicates.csv.gz"), index=False)
    rows = [dict(dgp=k[0], T=k[1], case=k[2], h=h + 1, **{f"q{t:.2f}": v[h, i] for i, t in enumerate(TAUS)})
            for k, v in hits.items() for h in range(HMAX)]
    pd.DataFrame(rows).to_csv(os.path.join(run_dir, "q1_coverage_by_horizon.csv.gz"), index=False)

    cols = ["pinball", "cov80", "cov98", "q10", "q50", "q90", "width", "width1", "widthH", "loc"]
    cell = reps.groupby(["dgp", "T", "case"])[cols].mean().reset_index()
    cell["mce"] = [mce(hits[(r.dgp, r.T, r.case)].mean(axis=0)) for r in cell.itertuples()]
    cell["se80"] = reps.groupby(["dgp", "T", "case"])["cov80"].sem().values
    pooled = cell.groupby(["T", "case"])[cols].mean().reset_index()
    pooled["mce"] = [mce(np.mean([hits[k].mean(axis=0) for k in hits if k[1] == r.T and k[2] == r.case], axis=0))
                     for r in pooled.itertuples()]
    pooled["maxcellmce"] = cell.groupby(["T", "case"])["mce"].max().values
    oracle = cell[cell["case"] == "oracle"].set_index(["dgp", "T"])["pinball"]
    cell["relpin"] = cell["pinball"] / oracle.reindex(pd.MultiIndex.from_frame(cell[["dgp", "T"]])).values
    pooled["relpin"] = cell.groupby(["T", "case"])["relpin"].mean().values
    cell.to_csv(os.path.join(run_dir, "q1_cells.csv"), index=False)

    results = []
    for r in pooled.to_dict("records"):
        for c in cols + ["mce", "maxcellmce", "relpin"]:
            results.append(dict(quantity=f"pooled {c} over the DGPs", value=float(r[c]),
                                conditions=dict(T=int(r["T"]), case=r["case"])))
    for r in cell[cell["T"] == 100].to_dict("records"):
        for c in ("cov80", "q50", "mce"):
            results.append(dict(quantity=f"{c} of a DGP", value=float(r[c]),
                                conditions=dict(dgp=r["dgp"], T=100, case=r["case"])))
    for case, g in reps.groupby("case"):
        results.append(dict(quantity="replicates failed", value=float(g["failed"].sum()), conditions=dict(case=case)))
    path = os.environ.get("CLAIMSET_RESULTS")
    if path:
        with open(path, "w") as f:
            json.dump(results, f)
    pd.set_option("display.width", 220)
    print(pooled[["T", "case", "mce", "maxcellmce", "cov80", "cov98", "q10", "q50", "q90", "width", "relpin"]]
          .round(3).to_string())
    print(cell.pivot_table(index=["dgp", "T"], columns="case", values="cov80").round(3).to_string())
    print(reps[reps["failed"] == 1]["model"].str[:80].value_counts().head(5).to_string())
    print(reps.groupby("case")["dispatched"].agg(lambda s: s.value_counts().to_dict()).to_string())


if __name__ == "__main__":
    main()
