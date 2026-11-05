"""Q2: the same harness under any version of smooth. Modes: selected (AutoMSARIMA), orders
(the orders of --source, estimated parameters), params (the orders and the parameters of
--source). Writes the coverage of the 99, the pinball, the bounds and the fits."""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import re
import signal
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

from q1_benchmark_reforecast import load, score
from q1_simulate import I10, I90, TAUS, TIMEOUT, TWO, Timeout, _alarm, quantiles

INITIALS = ("backcasting", "optimal")
HMAX = 24
SOURCE = None


def parse(name):
    # The orders and the constant in a model's name, the seasonal part if it has one.
    if name == "Constant level":
        return [0], [0], [0], True
    parts = re.findall(r"\((\d),(\d),(\d)\)", name)
    p, d, q = (list(map(int, x)) for x in zip(*parts))
    return p, d, q, ("drift" in name or "constant" in name)


def fit(i, s, j, mode):
    from smooth import AutoMSARIMA, MSARIMA

    x = np.asarray(s.x, float)
    initial = INITIALS[j]
    lags = [1, s.period] if s.period > 1 else [1]
    if mode == "selected":
        return AutoMSARIMA(lags=lags, initial=initial).fit(x)
    src = SOURCE[j][i]
    p, d, q, constant = parse(src["model"])
    lags = lags if len(p) > 1 else [1]
    orders = {"ar": p, "i": d, "ma": q}
    if mode == "orders":
        return MSARIMA(orders=orders, lags=lags, initial=initial, constant=constant).fit(x)
    coef = src["coef"]
    n_ar, n_ma = sum(p), sum(q)
    arma = {k: v for k, v in (("ar", coef[:n_ar]), ("ma", coef[n_ar : n_ar + n_ma])) if v}
    return MSARIMA(orders=orders, lags=lags, initial=initial, arma=arma or None,
                   constant=float(coef[-1]) if constant else False).fit(x)


def one(args):
    warnings.filterwarnings("ignore")
    j, i, s, mode = args
    out = dict(j=j, i=i, model="", coef=[], sigma=np.nan, pinball=np.nan, cov=np.full(99, np.nan),
               q=np.full((HMAX, 3), np.nan))
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(TIMEOUT)
    try:
        m = fit(i, s, j, mode)
        fc = m.predict(h=s.h, interval="prediction", level=TWO, side="both")
        mean, Q = quantiles(fc)
        out["pinball"], out["cov"] = score(s, Q)
        out["q"][: s.h] = np.c_[Q[:, I10], Q[:, I90], mean]
        out["model"], out["coef"], out["sigma"] = m.model_name, [float(c) for c in m.coef], float(m.sigma)
    except Timeout:
        out["model"] = "TIMEOUT"
    except Exception as exc:
        out["model"] = f"ERROR {type(exc).__name__}: {exc}"[:150]
    finally:
        signal.alarm(0)
    return out


def main():
    global SOURCE
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--source")
    ap.add_argument("--saved")
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    series = load()
    series = series[: a.limit] if a.limit else series
    if a.source:
        SOURCE = json.load(open(a.source))
    tasks = [(j, i, s, a.mode) for j in range(2) for i, s in enumerate(series)]
    with Pool(a.workers) as pool:
        res = pool.map(one, tasks, chunksize=8)
    n = len(series)
    cov, pin = np.full((2, n, 99), np.nan), np.full((2, n), np.nan)
    q = np.full((2, n, HMAX, 3), np.nan, dtype=np.float32)
    fits = [[None] * n for _ in range(2)]
    for r in res:
        cov[r["j"], r["i"]], pin[r["j"], r["i"]], q[r["j"], r["i"]] = r["cov"], r["pinball"], r["q"]
        fits[r["j"]][r["i"]] = dict(model=r["model"], coef=r["coef"], sigma=r["sigma"])
    run_dir = os.environ.get("CLAIMSET_RUN_DIR", ".")
    np.savez_compressed(os.path.join(run_dir, "q2_forecasts.npz"), cov=cov, pinball=pin, q=q)
    json.dump(fits, open(os.path.join(run_dir, "q2_fits.json"), "w"))
    results = []
    for j, ini in enumerate(INITIALS):
        ok = np.isfinite(pin[j])
        c = cov[j, ok].mean(axis=0)
        vals = dict(cov80=c[I90] - c[I10], q10=c[I10], q50=c[49], q90=c[I90], mce=np.abs(c - TAUS).mean(),
                    pinball=pin[j, ok].mean(), failed=(~ok).sum())
        print(ini, {k: round(float(v), 4) for k, v in vals.items()})
        results += [dict(quantity=f"benchmark {k}", value=float(v), conditions=dict(initial=ini, mode=a.mode)) for k, v in vals.items()]
        if a.saved:
            sv = np.load(a.saved)[j, : n]
            sv = sv[..., 1] if sv.ndim == 3 else sv
            gap = np.nanmean(np.abs(cov[j, ok] - sv[ok]))
            print(ini, "mean abs difference from saved", gap)
            results.append(dict(quantity="mean absolute difference of the coverage from the saved array",
                                value=float(gap), conditions=dict(initial=ini, mode=a.mode)))
        names = pd.Series([f["model"] for f in fits[j]])
        print(ini, "errors", names[names.str.match("ERROR|TIMEOUT")].str[:90].value_counts().head(3).to_dict())
    if os.environ.get("CLAIMSET_RESULTS"):
        json.dump(results, open(os.environ["CLAIMSET_RESULTS"], "w"))


if __name__ == "__main__":
    main()
