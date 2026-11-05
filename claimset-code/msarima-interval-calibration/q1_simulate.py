"""Q1: coverage of MSARIMA's intervals when the model is known, for true parameters and
orders (T), true orders with estimated parameters (E) and selected orders (S).

The data, the point forecasts of the oracle and the theoretical h-step variance come from
the polynomials below, independently of smooth. Writes a long CSV of the coverage per cell,
case, horizon and quantile into $CLAIMSET_RUN_DIR and the pooled numbers into
$CLAIMSET_RESULTS.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import signal
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import norm

HMAX, SIGMA, MEAN, BURN = 18, 10.0, 1000.0, 300
TWO = [0.98, 0.95, 0.90, 0.80]
TAUS = np.array([0.01, 0.025, 0.05, 0.10, 0.5, 0.90, 0.95, 0.975, 0.99])
TIMEOUT = 120


# ar, ma: non-seasonal and seasonal coefficients, in smooth's signs: y_t = ar y_{t-1} + ... +
# e_t + ma e_{t-1}. drift: the constant of the differenced model, in the units of SIGMA.
DGPS = {
    "rw": dict(lag=1, o=(0, 1, 0), s=(0, 0, 0), ar=[], ma=[], sar=[], sma=[], drift=None),
    "ima": dict(lag=1, o=(0, 1, 1), s=(0, 0, 0), ar=[], ma=[-0.5], sar=[], sma=[], drift=None),
    "ar1": dict(lag=1, o=(1, 0, 0), s=(0, 0, 0), ar=[0.7], ma=[], sar=[], sma=[], drift=None),
    "ma1": dict(lag=1, o=(0, 0, 1), s=(0, 0, 0), ar=[], ma=[0.5], sar=[], sma=[], drift=None),
    "arima111": dict(lag=1, o=(1, 1, 1), s=(0, 0, 0), ar=[0.5], ma=[-0.3], sar=[], sma=[], drift=None),
    "ima22": dict(lag=1, o=(0, 2, 2), s=(0, 0, 0), ar=[], ma=[-1.2, 0.32], sar=[], sma=[], drift=None),
    "rwd": dict(lag=1, o=(0, 1, 0), s=(0, 0, 0), ar=[], ma=[], sar=[], sma=[], drift=0.5),
    "imad": dict(lag=1, o=(0, 1, 1), s=(0, 0, 0), ar=[], ma=[-0.5], sar=[], sma=[], drift=0.5),
    "airline12": dict(lag=12, o=(0, 1, 1), s=(0, 1, 1), ar=[], ma=[-0.4], sar=[], sma=[-0.6], drift=None),
    "airline4": dict(lag=4, o=(0, 1, 1), s=(0, 1, 1), ar=[], ma=[-0.4], sar=[], sma=[-0.6], drift=None),
    "sar12": dict(lag=12, o=(1, 0, 0), s=(1, 0, 0), ar=[0.5], ma=[], sar=[0.6], sma=[], drift=None),
}


def polys(d):
    # The expanded AR side (with the differences) and MA side, both starting at 1.
    ar = np.array([1.0] + [-a for a in d["ar"]])
    ma = np.array([1.0] + list(d["ma"]))
    if d["sar"] or d["s"][1] or d["sma"]:
        sar = np.zeros(len(d["sar"]) * d["lag"] + 1)
        sar[0] = 1.0
        sar[d["lag"]:: d["lag"]] = [-a for a in d["sar"]] or sar[d["lag"]:: d["lag"]]
        ar = np.convolve(ar, sar)
        sma = np.zeros(len(d["sma"]) * d["lag"] + 1)
        sma[0] = 1.0
        sma[d["lag"]:: d["lag"]] = list(d["sma"]) or sma[d["lag"]:: d["lag"]]
        ma = np.convolve(ma, sma)
    stationary_ar = ar.copy()
    for _ in range(d["o"][1]):
        ar = np.convolve(ar, [1.0, -1.0])
    if d["s"][1]:
        seasonal_diff = np.zeros(d["lag"] + 1)
        seasonal_diff[[0, d["lag"]]] = [1.0, -1.0]
        ar = np.convolve(ar, seasonal_diff)
    return ar, ma, stationary_ar


def psi_weights(ar, ma, n):
    psi = np.zeros(n)
    for j in range(n):
        psi[j] = ma[j] if j < len(ma) else 0.0
        psi[j] -= sum(ar[i] * psi[j - i] for i in range(1, min(j, len(ar) - 1) + 1))
    return psi


def constant_of(d, stationary_ar):
    # The constant in the units of the data: the drift when differenced, else the one of
    # a process with the mean MEAN.
    if d["drift"] is not None:
        return d["drift"] * SIGMA
    if d["o"][1] == 0 and d["s"][1] == 0:
        return MEAN * stationary_ar.sum()
    return 0.0


def simulate(d, n, rng):
    # y_1..y_n and the oracle's point forecasts for the last HMAX, from the true past.
    ar, ma, stationary_ar = polys(d)
    c = constant_of(d, stationary_ar)
    total = BURN + n + HMAX
    e = rng.normal(0, SIGMA, total)
    y = np.zeros(total)
    start = BURN + n
    oracle = np.zeros(total)
    p, q = len(ar), len(ma)
    y[: p] = MEAN
    oracle[: p] = MEAN
    for t in range(p, total):
        mean = c - ar[1:] @ y[t - 1 : t - p : -1] if p > 1 else c
        if q > 1:
            lags = np.arange(1, q)
            idx = t - lags
            mask = idx >= 0
            mean = mean + ma[1:][mask] @ e[idx[mask]]
        y[t] = mean + e[t]
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
    lags = [1, d["lag"]] if d["lag"] > 1 else [1]
    seasonal = d["lag"] > 1
    orders = {"ar": [d["o"][0]] + ([d["s"][0]] if seasonal else []),
              "i": [d["o"][1]] + ([d["s"][1]] if seasonal else []),
              "ma": [d["o"][2]] + ([d["s"][2]] if seasonal else [])}
    arma = {"ar": list(d["ar"]) + list(d["sar"]), "ma": list(d["ma"]) + list(d["sma"])}
    has_constant = d["drift"] is not None or (d["o"][1] == 0 and d["s"][1] == 0)
    return lags, orders, arma, has_constant


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


def fit_case(case, name, x):
    # The model of one case; returns the fitted object.
    from smooth import AutoMSARIMA, MSARIMA

    d = DGPS[name]
    lags, orders, arma, has_constant = spec_of(d)
    kind, initial = case.split("_")[:2]
    if kind == "S":
        return AutoMSARIMA(lags=lags, initial=initial).fit(x)
    if kind == "T":
        c = constant_of(d, polys(d)[2])
        return MSARIMA(orders=orders, lags=lags, initial=initial,
                       arma={k: v for k, v in arma.items() if v},
                       constant=float(c) if has_constant else False).fit(x)
    return MSARIMA(orders=orders, lags=lags, initial=initial, constant=has_constant).fit(x)


def bounds_of(fc):
    # Quantiles at TAUS, the median being the point forecast, with the two-sided intervals.
    mean = np.asarray(fc.mean, float).ravel()
    lower, upper = np.asarray(fc.lower, float), np.asarray(fc.upper, float)
    lo = {round(float(c), 3): i for i, c in enumerate(fc.lower.columns)}
    up = {round(float(c), 3): i for i, c in enumerate(fc.upper.columns)}
    Q = np.empty((mean.size, TAUS.size))
    for k, t in enumerate(TAUS):
        if t == 0.5:
            Q[:, k] = mean
        else:
            Q[:, k] = lower[:, lo[round(t, 3)]] if t < 0.5 else upper[:, up[round(t, 3)]]
    return mean, Q


def one(args):
    # One replicate of one cell: per case, the hits (h, tau), the width against the
    # theoretical one and the point forecast against the oracle's, in units of the h-step
    # standard deviation.
    warnings.filterwarnings("ignore")
    name, n, rep, seed, cases = args
    d = DGPS[name]
    rng = np.random.default_rng(np.random.SeedSequence([seed, zlib_id(name), n, rep]))
    np.random.seed(int(rng.integers(2**31)))
    x, future, oracle = simulate(d, n, rng)
    ar, ma, _ = polys(d)
    sd_h = SIGMA * np.sqrt(np.cumsum(psi_weights(ar, ma, HMAX) ** 2))
    z80 = 2 * norm.ppf(0.9)

    out = {}
    q_o = oracle[:, None] + sd_h[:, None] * norm.ppf(TAUS)[None, :]
    out["oracle"] = dict(hit=(future[:, None] <= q_o).astype(float), width=np.ones(HMAX),
                         loc=np.zeros(HMAX), model="oracle", failed=0)
    signal.signal(signal.SIGALRM, _alarm)
    for case in cases:
        rec = dict(hit=np.full((HMAX, TAUS.size), np.nan), width=np.full(HMAX, np.nan),
                   loc=np.full(HMAX, np.nan), model="", failed=1)
        signal.alarm(TIMEOUT)
        try:
            m = fit_case(case, name, x)
            kind = "simulated" if case.endswith("sim") else "prediction"
            fc = m.predict(h=HMAX, interval=kind, level=TWO, side="both")
            mean, Q = bounds_of(fc)
            signal.alarm(0)
            i10, i90 = np.where(TAUS == 0.1)[0][0], np.where(TAUS == 0.9)[0][0]
            rec = dict(hit=(future[:, None] <= Q).astype(float),
                       width=(Q[:, i90] - Q[:, i10]) / (z80 * sd_h),
                       loc=(mean - oracle) / sd_h, model=m.model_name, failed=0,
                       dispatched=fc.interval)
        except Timeout:
            rec["model"] = "TIMEOUT"
        except Exception as exc:
            rec["model"] = f"ERROR {type(exc).__name__}: {exc}"[:150]
        finally:
            signal.alarm(0)
        out[case] = rec
    return name, n, rep, out


def zlib_id(name):
    import zlib

    return zlib.crc32(name.encode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sizes", type=int, nargs="+", default=[50, 100, 300])
    ap.add_argument("--dgps", nargs="+", default=list(DGPS))
    ap.add_argument("--cases", nargs="+", default=["T_backcasting", "T_optimal", "E_backcasting",
                                                   "E_optimal", "S_backcasting", "S_optimal",
                                                   "E_backcasting_sim"])
    a = ap.parse_args()
    tasks = [(name, n, rep, a.seed, a.cases) for name in a.dgps for n in a.sizes for rep in range(a.n)]
    with Pool(a.workers) as pool:
        res = pool.map(one, tasks, chunksize=1)

    rows, reps = [], []
    for name, n, rep, out in res:
        for case, r in out.items():
            reps.append(dict(dgp=name, T=n, rep=rep, case=case, model=r["model"], failed=r["failed"],
                             dispatched=r.get("dispatched", ""),
                             cov80=np.nanmean(r["hit"][:, 7 - 2] - r["hit"][:, 3]) if not r["failed"] else np.nan,
                             cov95=np.nanmean(r["hit"][:, 7] - r["hit"][:, 1]) if not r["failed"] else np.nan,
                             q10=np.nanmean(r["hit"][:, 3]), q50=np.nanmean(r["hit"][:, 4]),
                             q90=np.nanmean(r["hit"][:, 5]), width=np.nanmean(r["width"]),
                             width1=r["width"][0], widthH=r["width"][-1], loc=np.nanmean(r["loc"])))
            rows.append(dict(dgp=name, T=n, rep=rep, case=case, hit=r["hit"], width=r["width"]))
    reps = pd.DataFrame(reps)
    by_h = []
    for (name, n, case), g in pd.DataFrame(rows).groupby(["dgp", "T", "case"]):
        hit = np.nanmean(np.stack(g["hit"].values), axis=0)
        width = np.nanmean(np.stack(g["width"].values), axis=0)
        for h in range(HMAX):
            by_h.append(dict(dgp=name, T=n, case=case, h=h + 1, width=width[h],
                             **{f"q{t}": hit[h, k] for k, t in enumerate(TAUS)}))
    run_dir = os.environ.get("CLAIMSET_RUN_DIR", ".")
    reps.to_csv(os.path.join(run_dir, "q1_replicates.csv.gz"), index=False)
    pd.DataFrame(by_h).to_csv(os.path.join(run_dir, "q1_by_horizon.csv"), index=False)

    cols = ["cov80", "cov95", "q10", "q50", "q90", "width", "width1", "widthH", "loc"]
    results = []
    cell = reps.groupby(["dgp", "T", "case"])[cols].mean().reset_index()
    se = reps.groupby(["dgp", "T", "case"])["cov80"].sem().reset_index(name="se")
    cell = cell.merge(se)
    pooled = cell.groupby(["T", "case"])[cols].mean().reset_index()
    for _, r in pooled.iterrows():
        for c in cols:
            results.append(dict(quantity=f"pooled mean {c} over the DGPs",
                                value=float(r[c]), conditions=dict(T=int(r["T"]), case=r["case"])))
    for _, r in cell.iterrows():
        results.append(dict(quantity="coverage of the 80% interval, mean over horizons",
                            value=float(r["cov80"]),
                            conditions=dict(dgp=r["dgp"], T=int(r["T"]), case=r["case"], se=float(r["se"]))))
    for (n, case), g in cell.groupby(["T", "case"]):
        for c, level in (("cov80", 0.80), ("cov95", 0.95)):
            results.append(dict(quantity=f"largest absolute deviation of a cell's {c} from {level}",
                                value=float((g[c] - level).abs().max()), conditions=dict(T=int(n), case=case)))
    for case, g in reps.groupby("case"):
        results.append(dict(quantity="replicates failed", value=float(g["failed"].sum()), conditions=dict(case=case)))
    path = os.environ.get("CLAIMSET_RESULTS")
    if path:
        with open(path, "w") as f:
            json.dump(results, f)
    pd.set_option("display.width", 200)
    print(pooled.round(3).to_string())
    print(cell[cell["case"].isin(["oracle", "T_backcasting", "E_backcasting", "S_backcasting"])]
          .pivot_table(index=["dgp", "T"], columns="case", values="cov80").round(3).to_string())
    print(cell[cell["case"].isin(["T_backcasting", "T_optimal"])].pivot_table(
        index=["dgp", "T"], columns="case", values=["loc", "width1", "widthH"]).round(3).to_string())
    print(reps.groupby("case")["failed"].sum().to_string())
    print(reps[reps["failed"] == 1]["model"].value_counts().head(5).to_string())
    print(reps.groupby("case")["dispatched"].agg(lambda s: s.value_counts().to_dict()).to_string())


if __name__ == "__main__":
    main()
