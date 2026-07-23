"""Auto-ARIMA benchmark: point (RMSSE/SAME/time) + predictive distribution
(scaled pinball + calibration coverage at levels 0.01..0.99) for every Auto
ARIMA implementation. One fit per (method, series) produces both.

Saves (next to this file):
  {date}-benchmark-arima-point.npy  shape (M, N, 3)  -> rmsse, same, time
  {date}-benchmark-arima-dist.npy   shape (M, N, 99, 2) -> scaled pinball, coverage

Set BENCH_N to subsample (stratified stride across M1+M3+Tourism); default = all.
"""
import os, time, datetime, warnings, multiprocessing, signal
import numpy as np
warnings.filterwarnings("ignore")

LEVELS = np.round(np.arange(0.01, 1.0, 0.01), 2)          # 99
TWO = [round(1 - 2 * t, 2) for t in LEVELS if t < 0.5]    # two-sided coverages

# (name, package, config).  smooth AutoMSARIMA in three initialisations; the
# competitor Auto ARIMA engines.  aeon has no quantile API -> dist = NaN.
METHODS = [
    ("AutoMSARIMA Back", "smooth", {"initial": "backcasting"}),
    ("AutoMSARIMA Opt",  "smooth", {"initial": "optimal"}),
    ("AutoMSARIMA Two",  "smooth", {"initial": "two-stage"}),
    ("statsforecast AutoARIMA", "statsforecast", {}),
    ("skforecast Arima", "skforecast", {}),
    ("aeon AutoARIMA", "aeon", {}),
]

def _lags(s):  return [1, s.period] if s.period > 1 else [1]

def _pt_and_q_smooth(cfg, s):
    from smooth import AutoMSARIMA
    m = AutoMSARIMA(lags=_lags(s), initial=cfg["initial"]); m.fit(np.asarray(s.x, float))
    fc = m.predict(h=s.h, interval="prediction", level=list(TWO), side="both")
    mean = np.asarray(fc.mean).ravel()
    lov = np.asarray(fc.lower); upv = np.asarray(fc.upper)
    loc = {round(float(c), 2): i for i, c in enumerate(fc.lower.columns)}
    upc = {round(float(c), 2): i for i, c in enumerate(fc.upper.columns)}
    Q = np.empty((s.h, 99))
    for k, t in enumerate(LEVELS):
        if abs(t - 0.5) < 1e-9: Q[:, k] = mean
        elif t < 0.5:           Q[:, k] = lov[:, loc[round(t, 2)]]
        else:                   Q[:, k] = upv[:, upc[round(t, 2)]]
    return mean, Q

def _pt_and_q_statsforecast(s):
    from statsforecast.models import AutoARIMA
    m = AutoARIMA(season_length=s.period if s.period > 1 else 1); m.fit(np.asarray(s.x, float))
    fc = m.predict(h=s.h, level=sorted({int(round(100 * l)) for l in TWO}))
    mean = np.asarray(fc["mean"]).ravel(); Q = np.empty((s.h, 99))
    for k, t in enumerate(LEVELS):
        if abs(t - 0.5) < 1e-9: Q[:, k] = mean
        else:
            P = int(round(100 * (1 - 2 * t))) if t < 0.5 else int(round(100 * (2 * t - 1)))
            Q[:, k] = np.asarray(fc[f"lo-{P}" if t < 0.5 else f"hi-{P}"]).ravel()
    return mean, Q

def _pt_and_q_sktime(s):
    import pandas as pd
    from sktime.forecasting.arima import AutoARIMA
    m = AutoARIMA(sp=s.period if s.period > 1 else 1, suppress_warnings=True,
                  max_p=3, max_q=3, maxiter=15, error_action="ignore")
    m.fit(pd.Series(np.asarray(s.x, float)))
    fh = np.arange(1, s.h + 1)
    mean = np.asarray(m.predict(fh)).ravel()
    Q = np.asarray(m.predict_quantiles(fh=fh, alpha=list(LEVELS)))
    return mean, Q

def _pt_and_q_skforecast(s):
    from skforecast.stats import Arima
    m = Arima(m=s.period if s.period > 1 else 1); m.fit(np.asarray(s.x, float))
    fc = m.predict_interval(steps=s.h, level=list(TWO))
    mean = np.asarray(fc["mean"]).ravel(); Q = np.empty((s.h, 99))
    for k, t in enumerate(LEVELS):
        if abs(t - 0.5) < 1e-9: Q[:, k] = mean
        else:
            L = round(1 - 2 * t, 2) if t < 0.5 else round(2 * t - 1, 2)
            Q[:, k] = np.asarray(fc[f"lower_{L}" if t < 0.5 else f"upper_{L}"]).ravel()
    return mean, Q

def _pt_aeon(s):
    from aeon.forecasting.stats import AutoARIMA
    m = AutoARIMA()
    mean = np.asarray(m.iterative_forecast(np.asarray(s.x, float), prediction_horizon=s.h)).ravel()
    return mean, None

def _task(args):
    warnings.filterwarnings("ignore")
    j, i, s, pkg, cfg = args
    pt3 = np.full(3, np.nan); pin = np.full(99, np.nan); cov = np.full(99, np.nan)
    _TL = int(os.environ.get("BENCH_TASK_TIMEOUT", "45"))
    def _alarm(sig, frm): raise TimeoutError("task timeout")
    try:
        signal.signal(signal.SIGALRM, _alarm); signal.alarm(_TL)
        t0 = time.time()
        if pkg == "smooth":          mean, Q = _pt_and_q_smooth(cfg, s)
        elif pkg == "statsforecast": mean, Q = _pt_and_q_statsforecast(s)
        elif pkg == "sktime":        mean, Q = _pt_and_q_sktime(s)
        elif pkg == "skforecast":    mean, Q = _pt_and_q_skforecast(s)
        else:                        mean, Q = _pt_aeon(s)
        signal.alarm(0)
        elapsed = time.time() - t0
        mean = np.asarray(mean, float).ravel()[: s.h]
        xx = np.asarray(s.xx, float)
        scale_r = np.mean(np.diff(np.asarray(s.x, float)) ** 2)
        scale_a = np.mean(np.abs(np.diff(np.asarray(s.x, float))))
        if np.isfinite(mean).all():
            pt3[0] = np.sqrt(np.mean((xx - mean) ** 2) / scale_r) if scale_r > 0 else np.nan
            pt3[1] = np.abs(np.mean(xx - mean)) / scale_a if scale_a > 0 else np.nan
        pt3[2] = elapsed
        if Q is not None:
            Q = np.sort(np.asarray(Q, float), axis=1)
            if np.isfinite(Q).all():
                y = xx[:, None]; diff = y - Q
                sc = scale_a if (np.isfinite(scale_a) and scale_a > 0) else 1.0
                pin = np.maximum(LEVELS * diff, (LEVELS - 1) * diff).mean(axis=0) / sc
                cov = (y <= Q).mean(axis=0)
        return (j, i, pt3, pin, cov)
    except Exception:
        try: signal.alarm(0)
        except Exception: pass
        return (j, i, pt3, pin, cov)

def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from fcompdata import M1, M3, Tourism
    datasets = [M1[k] for k in M1.keys()] + [M3[k] for k in M3.keys()] + [Tourism[k] for k in Tourism.keys()]
    N = int(os.environ.get("BENCH_N", "0"))
    if N and N < len(datasets):
        stride = len(datasets) / N
        idx = sorted({int(i * stride) for i in range(N)})
        datasets = [datasets[i] for i in idx]
    nS = len(datasets); nM = len(METHODS)
    point = np.full((nM, nS, 3), np.nan)
    dist = np.full((nM, nS, 99, 2), np.nan)
    tasks = [(j, i, s, pkg, cfg) for j, (_, pkg, cfg) in enumerate(METHODS) for i, s in enumerate(datasets)]
    ncap = int(os.environ.get("BENCH_WORKERS", min(30, multiprocessing.cpu_count())))
    print(f"ARIMA benchmark: {nM} methods x {nS} series = {len(tasks)} tasks, {ncap} workers", flush=True)
    t0 = time.time(); done = 0
    ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=ncap, mp_context=ctx) as ex:
        for f in as_completed([ex.submit(_task, t) for t in tasks]):
            j, i, p3, pin, cov = f.result()
            point[j, i, :] = p3; dist[j, i, :, 0] = pin; dist[j, i, :, 1] = cov
            done += 1
            if done % 2000 == 0:
                print(f"  {done}/{len(tasks)} ({100*done/len(tasks):.1f}%) {(time.time()-t0)/60:.1f}min", flush=True)
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    d = os.path.dirname(__file__)
    np.save(os.path.join(d, f"{date}-benchmark-arima-point.npy"), point)
    np.save(os.path.join(d, f"{date}-benchmark-arima-dist.npy"), dist)
    print(f"saved point+dist ({nS} series) in {(time.time()-t0)/60:.1f} min", flush=True)
    names = [m[0] for m in METHODS]
    for j, nm in enumerate(names):
        rmsse = np.nanmean(point[j, :, 0]); same = np.nanmean(point[j, :, 1]); tm = np.nanmean(point[j, :, 2])
        pinm = np.nanmedian(np.nanmean(dist[j, :, :, 0], axis=1))
        cvc = np.nanmean(dist[j, :, :, 1], axis=0)
        mce = np.nanmean(np.abs(cvc - LEVELS)) if np.isfinite(cvc).any() else np.nan
        print(f"  {nm:30s} RMSSE={rmsse:.4f} SAME={same:.4f} t={tm:.3f}s medPinball={pinm:.4f} MCE={mce:.4f}", flush=True)

if __name__ == "__main__":
    main()
