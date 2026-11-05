"""The 5,315 benchmark series: predict() and reforecast() on the same AutoMSARIMA fit, with
the harness of 2026-11-05-smooth-1-3-0-benchmark_auto_arima.ipynb. The predict() coverage
is compared with the saved array, to show that the fit reproduces the benchmark.
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

from q1_simulate import I10, I50, I90, I01, I99, TAUS, TWO, NSIM_REF, TIMEOUT, Timeout, _alarm, mce, quantiles

INITIALS = ("backcasting", "optimal")


def load():
    from fcompdata import M1, M3, Tourism

    return [s for d in (M1, M3, Tourism) for s in (d[k] for k in d.keys())]


def score(s, Q):
    x, y = np.asarray(s.x, float), np.asarray(s.xx, float)
    scale = np.mean(np.abs(np.diff(x)))
    diff = y[:, None] - Q[: y.size]
    pinball = np.maximum(TAUS * diff, (TAUS - 1) * diff).mean() / (scale if scale else 1.0)
    return pinball, (y[:, None] <= Q[: y.size]).mean(axis=0)


def one(args):
    from smooth import AutoMSARIMA

    warnings.filterwarnings("ignore")
    i, j, s, initial = args
    np.random.seed(i)
    lags = [1, s.period] if s.period > 1 else [1]
    out = dict(i=i, j=j, pinball=np.full(2, np.nan), cov=np.full((2, TAUS.size), np.nan), model="", no_parameters=0)
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(TIMEOUT)
    try:
        m = AutoMSARIMA(lags=lags, initial=initial).fit(np.asarray(s.x, float))
        predicted = m.predict(h=s.h, interval="prediction", level=TWO, side="both")
        try:
            fcs = (predicted, m.reforecast(h=s.h, interval="prediction", level=TWO, side="both",
                                           nsim=NSIM_REF, seed=i))
        except ValueError as exc:
            # No estimated parameter: reforecast() raises, and the interval is predict()'s.
            if "size 0" not in str(exc):
                raise
            fcs, out["no_parameters"] = (predicted, predicted), 1
        for k, fc in enumerate(fcs):
            out["pinball"][k], out["cov"][k] = score(s, quantiles(fc)[1])
        out["model"] = m.model_name
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
    ap.add_argument("--saved", required=True)
    a = ap.parse_args()
    series = load()
    series = series[: a.limit] if a.limit else series
    tasks = [(i, j, s, ini) for j, ini in enumerate(INITIALS) for i, s in enumerate(series)]
    with Pool(a.workers) as pool:
        res = pool.map(one, tasks, chunksize=8)

    n = len(series)
    pin, cov = np.full((2, n, 2), np.nan), np.full((2, n, 2, TAUS.size), np.nan)
    failed, no_parameters = np.zeros(2), np.zeros(2)
    for r in res:
        pin[r["j"], r["i"]], cov[r["j"], r["i"]] = r["pinball"], r["cov"]
        failed[r["j"]] += not np.isfinite(r["pinball"]).all()
        no_parameters[r["j"]] += r["no_parameters"]
    run_dir = os.environ.get("CLAIMSET_RUN_DIR", ".")
    np.save(os.path.join(run_dir, "benchmark_pinball.npy"), pin)
    np.save(os.path.join(run_dir, "benchmark_coverage.npy"), cov)

    saved = np.load(a.saved)[:, : n]
    results = []
    for j, ini in enumerate(INITIALS):
        ok = np.isfinite(pin[j]).all(axis=1)
        for k, kind in enumerate(("predict", "reforecast")):
            c = cov[j, ok, k].mean(axis=0)
            vals = dict(cov80=c[I90] - c[I10], cov98=c[I99] - c[I01], q10=c[I10], q50=c[I50], q90=c[I90],
                        mce=mce(c), pinball=np.nanmean(pin[j, ok, k]))
            for q, v in vals.items():
                results.append(dict(quantity=f"benchmark {q}", value=float(v), conditions=dict(initial=ini, method=kind)))
            print(ini, kind, {q: round(float(v), 4) for q, v in vals.items()})
        gap = np.nanmean(np.abs(cov[j, ok, 0] - saved[j, ok]))
        results.append(dict(quantity="mean absolute difference of the predict coverage from the saved array",
                            value=float(gap), conditions=dict(initial=ini)))
        results.append(dict(quantity="series failed", value=float(failed[j]), conditions=dict(initial=ini)))
        results.append(dict(quantity="series without parameters", value=float(no_parameters[j]), conditions=dict(initial=ini)))
        print(ini, "diff from saved", round(float(gap), 5), "failed", failed[j])
    path = os.environ.get("CLAIMSET_RESULTS")
    if path:
        with open(path, "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    main()
