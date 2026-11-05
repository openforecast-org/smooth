"""Q3 and Q4: per-horizon forecasts on the benchmark for the selected and the forced models,
at the 8 quantiles below, to split the error into location and width."""
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

from q1_benchmark_reforecast import load
from q1_simulate import TIMEOUT, Timeout, _alarm, column_level

TWO = [0.98, 0.90, 0.80, 0.50]
TAU = [0.01, 0.05, 0.10, 0.25, 0.75, 0.90, 0.95, 0.99]
HMAX = 24
METHODS = ("auto_back", "auto_opt", "es_back", "force_011", "force_ann", "force_mnn")


def fit(method, s):
    from smooth import ES, MSARIMA, AutoMSARIMA

    x = np.asarray(s.x, float)
    lags = [1, s.period] if s.period > 1 else [1]
    if method.startswith("auto"):
        return AutoMSARIMA(lags=lags, initial="optimal" if method == "auto_opt" else "backcasting").fit(x)
    if method == "es_back":
        return ES(model="ZXZ" if s.period > 1 else "ZXN", lags=lags, initial="backcasting").fit(x)
    if method == "force_011":
        return MSARIMA(orders={"ar": [0], "i": [1], "ma": [1]}, lags=[1], initial="backcasting").fit(x)
    return ES(model="ANN" if method == "force_ann" else "MNN", lags=[1], initial="backcasting").fit(x)


def one(args):
    warnings.filterwarnings("ignore")
    j, i, s, method = args
    np.random.seed(i)
    h = s.h
    out = dict(j=j, i=i, q=np.full((HMAX, 9), np.nan), model="", interval="")
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(TIMEOUT)
    try:
        m = fit(method, s)
        fc = m.predict(h=h, interval="prediction", level=TWO, side="both")
        lo = {column_level(c): k for k, c in enumerate(fc.lower.columns)}
        up = {column_level(c): k for k, c in enumerate(fc.upper.columns)}
        lower, upper = np.asarray(fc.lower, float), np.asarray(fc.upper, float)
        for k, t in enumerate(TAU):
            out["q"][: h, k] = lower[:, lo[t]] if t < 0.5 else upper[:, up[t]]
        out["q"][: h, 8] = np.asarray(fc.mean, float).ravel()
        out["model"], out["interval"] = m.model_name, fc.interval
    except Timeout:
        out["model"] = "TIMEOUT"
    except Exception as exc:
        out["model"] = f"ERROR {type(exc).__name__}: {exc}"[:150]
    finally:
        signal.alarm(0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    series = load()
    series = series[: a.limit] if a.limit else series
    tasks = [(j, i, s, m) for j, m in enumerate(METHODS) for i, s in enumerate(series)]
    with Pool(a.workers) as pool:
        res = pool.map(one, tasks, chunksize=8)
    q = np.full((len(METHODS), len(series), HMAX, 9), np.nan, dtype=np.float32)
    names = np.empty((len(METHODS), len(series)), dtype=object)
    kinds = np.empty((len(METHODS), len(series)), dtype=object)
    for r in res:
        q[r["j"], r["i"]], names[r["j"], r["i"]], kinds[r["j"], r["i"]] = r["q"], r["model"], r["interval"]
    run_dir = os.environ.get("CLAIMSET_RUN_DIR", ".")
    np.savez_compressed(os.path.join(run_dir, "q3_forecasts.npz"), q=q)
    pd.DataFrame(names, index=METHODS).T.to_csv(os.path.join(run_dir, "q3_models.csv.gz"), index=False)
    results = [dict(quantity="series failed", value=float(pd.Series(names[j]).str.match("ERROR|TIMEOUT").sum()),
                    conditions=dict(method=m)) for j, m in enumerate(METHODS)]
    print({m: pd.Series(kinds[j]).value_counts().to_dict() for j, m in enumerate(METHODS)})
    print(results)
    if os.environ.get("CLAIMSET_RESULTS"):
        json.dump(results, open(os.environ["CLAIMSET_RESULTS"], "w"))


if __name__ == "__main__":
    main()
