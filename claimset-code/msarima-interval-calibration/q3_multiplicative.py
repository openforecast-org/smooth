"""Does an ARIMA whose error scale follows the level remove the pattern by level? AutoMSARIMA
with distribution="dlnorm" on the benchmark, and on simulated series with multiplicative
errors, against the default and ETS."""
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

from q1_benchmark_reforecast import load, score
from q2_forecasts import parse
from q1_simulate import I10, I90, TAUS, TIMEOUT, TWO, Timeout, _alarm, quantiles


def model(name, x, period):
    from smooth import ADAM, ES, AutoMSARIMA

    lags = [1, period] if period > 1 else [1]
    if name == "auto":
        return AutoMSARIMA(lags=lags, initial="backcasting").fit(x)
    if name == "auto_dlnorm":
        # The orders AutoMSARIMA selects, with the error scale following the level
        p, d, q, constant = parse(AutoMSARIMA(lags=lags, initial="backcasting").fit(x).model_name)
        return ADAM(model="NNN", orders={"ar": p, "i": d, "ma": q}, lags=lags if len(p) > 1 else [1], constant=constant,
                    distribution="dlnorm", initial="backcasting").fit(x)
    return ES(model="ZXZ" if period > 1 else "ZXN", lags=lags, initial="backcasting").fit(x)


def one(args):
    warnings.filterwarnings("ignore")
    kind, i, name, x, y, period, h = args
    out = dict(kind=kind, i=i, name=name, cov=np.full(99, np.nan), pinball=np.nan, model="", level=x[-max(len(x) // 3, 2) :].mean() / x.mean(), n=len(x))
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(TIMEOUT)
    try:
        m = model(name, x, period)
        _, Q = quantiles(m.predict(h=h, interval="prediction", level=TWO, side="both"))
        diff = y[:, None] - Q
        scale = np.mean(np.abs(np.diff(x)))
        out["pinball"] = np.maximum(TAUS * diff, (TAUS - 1) * diff).mean() / scale
        out["cov"] = (y[:, None] <= Q).mean(axis=0)
        out["model"] = m.model_name
    except Timeout:
        out["model"] = "TIMEOUT"
    except Exception as exc:
        out["model"] = f"ERROR {type(exc).__name__}: {exc}"[:120]
    finally:
        signal.alarm(0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--nsim", type=int, default=1000)
    a = ap.parse_args()
    series = load()
    series = series[: a.limit] if a.limit else series
    tasks = []
    for name in ("auto", "auto_dlnorm"):
        for i, s in enumerate(series):
            tasks.append(("benchmark", i, name, np.asarray(s.x, float), np.asarray(s.xx, float), s.period, s.h))
    # Multiplicative errors: log y follows an IMA with a drift
    rng = np.random.default_rng(31)
    for i in range(a.nsim):
        e = rng.normal(0, 0.05, 119)
        drift = rng.choice([0.0, 0.01])
        lx = np.log(1000) + np.cumsum(drift + e[1:] - 0.5 * e[:-1])
        y = np.exp(lx)
        for name in ("auto", "auto_dlnorm", "es"):
            tasks.append(("multiplicative", i, name, y[:100], y[100:], 1, 18))
    with Pool(a.workers) as pool:
        res = pd.DataFrame(pool.map(one, tasks, chunksize=8))
    print(res[res["model"].str.match("ERROR|TIMEOUT")]["model"].str[:90].value_counts().head(4).to_dict())
    results = []
    print(res.groupby(["kind", "name"])["pinball"].apply(lambda v: v.notna().sum()).to_dict())
    for (kind, name), g in res.groupby(["kind", "name"]):
        g = g[g["pinball"].notna()].copy()
        if g.empty:
            continue
        cov = np.stack(g["cov"].values)
        g["cov80"] = cov[:, I90] - cov[:, I10]
        g["t"] = pd.qcut(g["level"], 3, labels=["low", "mid", "high"])
        c = cov.mean(axis=0)
        vals = dict(n=len(g), cov80=c[I90] - c[I10], q10=c[I10], q50=c[49], q90=c[I90], mce=np.abs(c - TAUS).mean(),
                    pinball_mean=g["pinball"].mean(), pinball_median=g["pinball"].median(),
                    **{f"cov80_{k}": v for k, v in g.groupby("t", observed=True)["cov80"].mean().items()})
        print(kind, name, {k: round(float(v), 4) for k, v in vals.items()}, g["model"].value_counts().head(3).to_dict())
        results += [dict(quantity=k, value=float(v), conditions=dict(data=kind, method=name)) for k, v in vals.items()]
    if os.environ.get("CLAIMSET_RESULTS"):
        json.dump(results, open(os.environ["CLAIMSET_RESULTS"], "w"))


if __name__ == "__main__":
    main()
