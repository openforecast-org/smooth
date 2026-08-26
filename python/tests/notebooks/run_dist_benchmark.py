"""Predictive-distribution benchmark: pinball (scaled) + calibration coverage
at quantile levels 0.01..0.99 for all methods. Saves {date}-benchmark-dist.npy
of shape (n_methods, n_series, 99, 2) -> [:, :, :, 0]=scaled pinball,
[:, :, :, 1]=coverage indicator mean. Mirrors the point benchmark structure."""
import os, time, datetime, warnings, multiprocessing
import numpy as np
warnings.filterwarnings("ignore")

LEVELS = np.round(np.arange(0.01, 1.0, 0.01), 2)          # 99 quantile levels
TWO = [round(1 - 2 * t, 2) for t in LEVELS if t < 0.5]    # two-sided coverages 0.98..0.02

METHODS = [
    ("ADAM ETS Back", "smooth", {"class": "ADAM", "model": "ZXZ", "initial": "backcasting"}),
    ("ADAM ETS Opt",  "smooth", {"class": "ADAM", "model": "ZXZ", "initial": "optimal"}),
    ("ADAM ETS Two",  "smooth", {"class": "ADAM", "model": "ZXZ", "initial": "two-stage"}),
    ("ES Back", "smooth", {"class": "ES", "model": "ZXZ", "initial": "backcasting"}),
    ("ES Opt",  "smooth", {"class": "ES", "model": "ZXZ", "initial": "optimal"}),
    ("ES Two",  "smooth", {"class": "ES", "model": "ZXZ", "initial": "two-stage"}),
    ("ES XXX", "smooth", {"class": "ES", "model": "XXX", "initial": "backcasting"}),
    ("ES ZZZ", "smooth", {"class": "ES", "model": "ZZZ", "initial": "backcasting"}),
    ("ES FFF", "smooth", {"class": "ES", "model": "FFF", "initial": "backcasting"}),
    ("ES SXS", "smooth", {"class": "ES", "model": "SXS", "initial": "backcasting"}),
    ("statsforecast AutoETS", "statsforecast", {}),
    ("sktime AutoETS", "sktime", {}),
    ("skforecast AutoETS", "skforecast", {}),
    ("aeon AutoETS", "aeon", {}),
]

def _q_smooth(cfg, s):
    from smooth import ADAM, ES
    lags = [1, s.period] if s.period > 1 else [1]
    ms = cfg["model"]
    if s.period <= 1 and len(ms) == 3 and ms[2] != "N":
        ms = ms[:2] + "N"
    M = ADAM if cfg["class"] == "ADAM" else ES
    m = M(model=ms, lags=lags, initial=cfg["initial"]); m.fit(s.x)
    fc = m.predict(h=s.h, interval="prediction", level=list(TWO), side="both")
    lov = np.asarray(fc.lower); upv = np.asarray(fc.upper); mean = np.asarray(fc.mean).ravel()
    loc = {round(float(c), 2): i for i, c in enumerate(fc.lower.columns)}
    upc = {round(float(c), 2): i for i, c in enumerate(fc.upper.columns)}
    Q = np.empty((s.h, 99))
    for k, t in enumerate(LEVELS):
        if abs(t - 0.5) < 1e-9: Q[:, k] = mean
        elif t < 0.5:           Q[:, k] = lov[:, loc[round(t, 2)]]
        else:                   Q[:, k] = upv[:, upc[round(t, 2)]]
    return Q

def _q_statsforecast(s):
    from statsforecast.models import AutoETS
    m = AutoETS(season_length=s.period if s.period > 1 else 1); m.fit(np.asarray(s.x, float))
    fc = m.predict(h=s.h, level=sorted({int(round(100 * l)) for l in TWO}))
    mean = np.asarray(fc["mean"]).ravel(); Q = np.empty((s.h, 99))
    for k, t in enumerate(LEVELS):
        if abs(t - 0.5) < 1e-9: Q[:, k] = mean
        else:
            P = int(round(100 * (1 - 2 * t))) if t < 0.5 else int(round(100 * (2 * t - 1)))
            Q[:, k] = np.asarray(fc[f"lo-{P}" if t < 0.5 else f"hi-{P}"]).ravel()
    return Q

def _q_sktime(s):
    import pandas as pd
    from sktime.forecasting.ets import AutoETS
    m = AutoETS(sp=s.period if s.period > 1 else 1, auto=True)
    m.fit(pd.Series(np.asarray(s.x, float)))
    return np.asarray(m.predict_quantiles(fh=np.arange(1, s.h + 1), alpha=list(LEVELS)))

def _q_skforecast(s):
    from skforecast.stats import Ets
    m = Ets(model="ZZZ", m=s.period if s.period > 1 else 1); m.fit(np.asarray(s.x, float))
    fc = m.predict_interval(steps=s.h, level=list(TWO))
    mean = np.asarray(fc["mean"]).ravel(); Q = np.empty((s.h, 99))
    for k, t in enumerate(LEVELS):
        if abs(t - 0.5) < 1e-9: Q[:, k] = mean
        else:
            L = round(1 - 2 * t, 2) if t < 0.5 else round(2 * t - 1, 2)
            Q[:, k] = np.asarray(fc[f"lower_{L}" if t < 0.5 else f"upper_{L}"]).ravel()
    return Q

def _task(args):
    warnings.filterwarnings("ignore")
    j, i, s, pkg, cfg = args
    try:
        if pkg == "smooth":        Q = _q_smooth(cfg, s)
        elif pkg == "statsforecast": Q = _q_statsforecast(s)
        elif pkg == "sktime":      Q = _q_sktime(s)
        elif pkg == "skforecast":  Q = _q_skforecast(s)
        else:                      return (j, i, None, None)   # aeon: no quantiles
        Q = np.sort(np.asarray(Q, float), axis=1)              # enforce monotone quantiles
        y = np.asarray(s.xx, float)[:, None]                   # (h,1)
        scale = np.mean(np.abs(np.diff(np.asarray(s.x, float))))
        if not np.isfinite(scale) or scale == 0: scale = 1.0
        diff = y - Q
        pinball = np.maximum(LEVELS * diff, (LEVELS - 1) * diff).mean(axis=0) / scale  # (99,)
        cover = (y <= Q).mean(axis=0)                                                   # (99,)
        if not np.all(np.isfinite(Q)): return (j, i, None, None)
        return (j, i, pinball, cover)
    except Exception:
        return (j, i, None, None)

def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from fcompdata import M1, M3, Tourism
    datasets = [M1[k] for k in M1.keys()] + [M3[k] for k in M3.keys()] + [Tourism[k] for k in Tourism.keys()]
    _lim = int(os.environ.get('BENCH_LIMIT', '0'))
    if _lim: datasets = datasets[:_lim] + datasets[1001:1001+_lim] + datasets[-_lim:]
    nS = len(datasets); nM = len(METHODS)
    out = np.full((nM, nS, 99, 2), np.nan)
    tasks = [(j, i, s, pkg, cfg) for j, (_, pkg, cfg) in enumerate(METHODS) for i, s in enumerate(datasets)]
    ncap = int(os.environ.get("BENCH_WORKERS", min(30, multiprocessing.cpu_count())))
    print(f"dist benchmark: {nM} methods x {nS} series = {len(tasks)} tasks, {ncap} workers", flush=True)
    t0 = time.time(); done = 0
    ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=ncap, mp_context=ctx) as ex:
        futs = [ex.submit(_task, t) for t in tasks]
        for f in as_completed(futs):
            j, i, pb, cv = f.result()
            if pb is not None:
                out[j, i, :, 0] = pb; out[j, i, :, 1] = cv
            done += 1
            if done % 5000 == 0:
                el = time.time() - t0
                print(f"  {done}/{len(tasks)} ({100*done/len(tasks):.1f}%) {el/60:.1f}min", flush=True)
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    path = os.path.join(os.path.dirname(__file__), f"{date}-benchmark-dist.npy")
    np.save(path, out)
    print(f"saved {path} in {(time.time()-t0)/60:.1f} min", flush=True)
    # quick summary
    names = [m[0] for m in METHODS]
    for j, nm in enumerate(names):
        pb = np.nanmean(out[j, :, :, 0]); cov = np.nanmean(out[j, :, :, 1], axis=0)
        mce = np.nanmean(np.abs(cov - LEVELS)) if np.isfinite(pb) else np.nan
        print(f"  {nm:24s} meanScaledPinball={pb:.4f} MCE={mce:.4f}", flush=True)

if __name__ == "__main__":
    main()
