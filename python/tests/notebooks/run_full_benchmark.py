"""Full competition benchmark: ETS and ARIMA, every implementation.

Runs the M1 + M3 + Tourism series (5,315 in total) through each method in turn.
Methods are evaluated ONE AT A TIME; within a method the series are spread over
a process pool, so the per-series timings are comparable within a method and the
memory footprint stays bounded.

Adds `pmdarima` auto_arima, which earlier runs omitted.  sktime's AutoARIMA is
deliberately absent: it is a thin adapter over pmdarima.arima.AutoARIMA and
returns identical forecasts, so it would only duplicate the pmdarima row.
Each series is given a wall-clock budget (BENCH_TIMEOUT, default 120s); a series
that exceeds it is recorded as a timeout rather than being allowed to stall the
run, so a slow implementation is quantified instead of silently excluded.

Outputs, in this directory:
  {date}-benchmark-full-point.npy   (n_methods, n_series, 3)   rmsse, time, timed_out
  {date}-benchmark-full-dist.npy    (n_methods, n_series, 99, 2) scaled pinball, coverage
  {date}-benchmark-full-methods.txt  one method name per line, in array order

Environment:
  BENCH_WORKERS  process pool size (default 32)
  BENCH_TIMEOUT  per-series seconds (default 120)
  BENCH_LIMIT    evaluate only the first N series of each dataset (smoke test)
  BENCH_ONLY     comma-separated substrings; run only matching methods
  BENCH_KIND     "ets" or "arima"; run only that model class
"""
import datetime
import multiprocessing
import os
import signal
import time
import warnings

# Pin the numerical libraries to a single thread BEFORE numpy is imported: BLAS
# reads these at load time, and with a 32-process pool each worker would
# otherwise spawn its own threads and oversubscribe the machine.  Left unset,
# statsmodels-backed methods (pmdarima especially) slow down by an order of
# magnitude and the per-series times measure contention rather than the method.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

warnings.filterwarnings("ignore")

LEVELS = np.round(np.arange(0.01, 1.0, 0.01), 2)             # 99 quantile levels
TWO = [round(1 - 2 * t, 2) for t in LEVELS if t < 0.5]       # 49 two-sided levels
TIMEOUT = int(os.environ.get("BENCH_TIMEOUT", "120"))

# (name, kind, package, config).  kind selects the ETS or ARIMA table.
METHODS = [
    ("ADAM ETS Back",           "ets",   "smooth",        {"class": "ADAM", "model": "ZXZ", "initial": "backcasting"}),
    ("ES Back",                 "ets",   "smooth",        {"class": "ES",   "model": "ZXZ", "initial": "backcasting"}),
    ("ES Opt",                  "ets",   "smooth",        {"class": "ES",   "model": "ZXZ", "initial": "optimal"}),
    ("statsforecast AutoETS",   "ets",   "sf_ets",        {}),
    ("sktime AutoETS",          "ets",   "skt_ets",       {}),
    ("skforecast AutoETS",      "ets",   "skf_ets",       {}),
    ("aeon AutoETS",            "ets",   "aeon_ets",      {}),
    ("AutoMSARIMA Back",        "arima", "smooth_arima",  {"initial": "backcasting"}),
    ("AutoMSARIMA Opt",         "arima", "smooth_arima",  {"initial": "optimal"}),
    ("statsforecast AutoARIMA", "arima", "sf_arima",      {}),
    ("pmdarima auto_arima",     "arima", "pmd_arima",     {}),
    ("skforecast Arima",        "arima", "skf_arima",     {}),
    ("aeon AutoARIMA",          "arima", "aeon_arima",    {}),
]


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


def _quantiles_from_smooth(fc):
    """smooth labels its interval columns by the quantile level itself:
    `lower` carries 0.01..0.49 and `upper` 0.51..0.99, so index by tau directly."""
    mean = np.asarray(fc.mean).ravel()
    lo, up = np.asarray(fc.lower), np.asarray(fc.upper)
    lc = {round(float(c), 2): i for i, c in enumerate(fc.lower.columns)}
    uc = {round(float(c), 2): i for i, c in enumerate(fc.upper.columns)}
    Q = np.empty((len(mean), 99))
    for k, t in enumerate(LEVELS):
        if abs(t - 0.5) < 1e-9:
            Q[:, k] = mean
        elif t < 0.5:
            Q[:, k] = lo[:, lc[round(float(t), 2)]]
        else:
            Q[:, k] = up[:, uc[round(float(t), 2)]]
    return mean, Q


def _fit(pkg, cfg, s):
    """Return (mean, Q) for one series; Q is (h, 99) or None when unavailable."""
    y = np.asarray(s.x, float)
    sp = s.period if s.period > 1 else 1

    if pkg == "smooth":
        from smooth import ADAM, ES
        ms = cfg["model"]
        if sp <= 1 and len(ms) == 3 and ms[2] != "N":
            ms = ms[:2] + "N"
        M = ADAM if cfg["class"] == "ADAM" else ES
        m = M(model=ms, lags=[1, sp] if sp > 1 else [1], initial=cfg["initial"])
        m.fit(y)
        return _quantiles_from_smooth(
            m.predict(h=s.h, interval="prediction", level=list(TWO), side="both"))

    if pkg == "smooth_arima":
        from smooth import AutoMSARIMA
        m = AutoMSARIMA(lags=[1, sp] if sp > 1 else [1], initial=cfg["initial"])
        m.fit(y)
        return _quantiles_from_smooth(
            m.predict(h=s.h, interval="prediction", level=list(TWO), side="both"))

    if pkg in ("sf_ets", "sf_arima"):
        from statsforecast import models as sfm
        m = (sfm.AutoETS if pkg == "sf_ets" else sfm.AutoARIMA)(season_length=sp)
        m.fit(y)
        pct = sorted({int(round(100 * l)) for l in TWO})
        fc = m.predict(h=s.h, level=pct)
        mean = np.asarray(fc["mean"]).ravel()
        Q = np.empty((s.h, 99))
        for k, t in enumerate(LEVELS):
            if abs(t - 0.5) < 1e-9:
                Q[:, k] = mean
            else:
                P = int(round(100 * (1 - 2 * t))) if t < 0.5 else int(round(100 * (2 * t - 1)))
                Q[:, k] = np.asarray(fc["lo-%d" % P if t < 0.5 else "hi-%d" % P]).ravel()
        return mean, Q

    # skt_arima is retained for reference only: sktime's AutoARIMA subclasses
    # _PmdArimaAdapter and delegates to pmdarima.arima.AutoARIMA, so it returns
    # pmdarima's forecasts exactly.  It is not in METHODS.
    if pkg in ("skt_ets", "skt_arima"):
        import pandas as pd
        if pkg == "skt_ets":
            from sktime.forecasting.ets import AutoETS as M
            m = M(sp=sp, auto=True)
        else:
            from sktime.forecasting.arima import AutoARIMA as M
            m = M(sp=sp, suppress_warnings=True)
        m.fit(pd.Series(y))
        fh = np.arange(1, s.h + 1)
        mean = np.asarray(m.predict(fh)).ravel()
        Q = np.asarray(m.predict_quantiles(fh=fh, alpha=list(LEVELS)))
        return mean, Q

    if pkg == "pmd_arima":
        import pmdarima as pm
        m = pm.auto_arima(y, seasonal=sp > 1, m=sp, suppress_warnings=True,
                          error_action="ignore")
        mean = np.asarray(m.predict(n_periods=s.h)).ravel()
        Q = np.empty((s.h, 99))
        for k, t in enumerate(LEVELS):
            if abs(t - 0.5) < 1e-9:
                Q[:, k] = mean
            else:
                alpha = float(2 * t) if t < 0.5 else float(2 * (1 - t))
                _, ci = m.predict(n_periods=s.h, return_conf_int=True, alpha=alpha)
                Q[:, k] = np.asarray(ci)[:, 0 if t < 0.5 else 1]
        return mean, Q

    if pkg in ("skf_ets", "skf_arima"):
        if pkg == "skf_ets":
            from skforecast.stats import Ets
            m = Ets(model="ZZZ", m=sp)
        else:
            from skforecast.stats import Arima
            # skforecast has no AutoArima class: `Arima` switches to its own
            # Hyndman-Khandakar search (skforecast.stats.arima._auto_arima) only
            # when an order is left unspecified.  The defaults are concrete --
            # order=(1,0,0), seasonal_order=(0,0,0) -- so Arima(m=sp) alone fits
            # a fixed non-seasonal AR(1) and silently ignores m.
            m = Arima(order=None, seasonal_order=None, m=sp)
        m.fit(y)
        fc = m.predict_interval(steps=s.h, level=list(TWO))
        mean = np.asarray(fc["mean"]).ravel()
        Q = np.empty((s.h, 99))
        for k, t in enumerate(LEVELS):
            if abs(t - 0.5) < 1e-9:
                Q[:, k] = mean
            else:
                L = round(1 - 2 * t, 2) if t < 0.5 else round(2 * t - 1, 2)
                Q[:, k] = np.asarray(fc["lower_%s" % L if t < 0.5 else "upper_%s" % L]).ravel()
        return mean, Q

    if pkg in ("aeon_ets", "aeon_arima"):
        if pkg == "aeon_ets":
            from aeon.forecasting.stats import AutoETS
            m = AutoETS(seasonal_period=sp)
        else:
            from aeon.forecasting.stats import AutoARIMA
            m = AutoARIMA()            # aeon's AutoARIMA has no seasonal option
        m.fit(y)
        mean = np.asarray(m.iterative_forecast(y, prediction_horizon=s.h)).ravel()
        return mean, None              # aeon exposes no quantile API

    raise ValueError("unknown package %s" % pkg)


def task(args):
    i, s, pkg, cfg = args
    warnings.filterwarnings("ignore")
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(TIMEOUT)
    try:
        t0 = time.time()
        mean, Q = _fit(pkg, cfg, s)
        elapsed = time.time() - t0
        signal.alarm(0)

        y = np.asarray(s.xx, float)
        tr = np.asarray(s.x, float)
        sc2 = np.mean(np.diff(tr) ** 2)
        rmsse = float(np.sqrt(np.mean((y - mean) ** 2) / sc2)) if sc2 else np.nan

        if Q is None or not np.all(np.isfinite(Q)):
            return (i, rmsse, elapsed, 0.0, None, None)
        Q = np.sort(np.asarray(Q, float), axis=1)
        scale = np.mean(np.abs(np.diff(tr)))
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        diff = y[:, None] - Q
        pinball = np.maximum(LEVELS * diff, (LEVELS - 1) * diff).mean(axis=0) / scale
        coverage = (y[:, None] <= Q).mean(axis=0)
        return (i, rmsse, elapsed, 0.0, pinball, coverage)
    except Timeout:
        signal.alarm(0)
        return (i, np.nan, float(TIMEOUT), 1.0, None, None)
    except Exception:
        signal.alarm(0)
        return (i, np.nan, np.nan, 0.0, None, None)


def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from fcompdata import M1, M3, Tourism

    data = [M1[k] for k in M1.keys()] + [M3[k] for k in M3.keys()] + [Tourism[k] for k in Tourism.keys()]
    lim = int(os.environ.get("BENCH_LIMIT", "0"))
    if lim:
        data = data[:lim] + data[1001:1001 + lim] + data[-lim:]
    only = [s.strip().lower() for s in os.environ.get("BENCH_ONLY", "").split(",") if s.strip()]
    kind_filter = os.environ.get("BENCH_KIND", "").strip().lower()
    methods = [m for m in METHODS if not only or any(o in m[0].lower() for o in only)]
    if kind_filter:
        methods = [m for m in methods if m[1] == kind_filter]

    nS, nM = len(data), len(methods)
    workers = int(os.environ.get("BENCH_WORKERS", "32"))
    point = np.full((nM, nS, 3), np.nan)
    dist = np.full((nM, nS, 99, 2), np.nan)
    print("%d methods x %d series, %d workers, %ds per-series budget"
          % (nM, nS, workers, TIMEOUT), flush=True)

    ctx = multiprocessing.get_context("fork")
    for j, (name, kind, pkg, cfg) in enumerate(methods):
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = [ex.submit(task, (i, s, pkg, cfg)) for i, s in enumerate(data)]
            for f in as_completed(futs):
                i, rmsse, el, to, pb, cv = f.result()
                point[j, i] = (rmsse, el, to)
                if pb is not None:
                    dist[j, i, :, 0] = pb
                    dist[j, i, :, 1] = cv
        r = point[j, :, 0]
        print("  %-24s medRMSSE=%8.4f meanRMSSE=%12.4f meanTime=%7.3f "
              "timeouts=%4d failed=%4d  [%.1f min]"
              % (name, np.nanmedian(r), np.nanmean(r), np.nanmean(point[j, :, 1]),
                 int(np.nansum(point[j, :, 2])), int(np.sum(np.isnan(r))),
                 (time.time() - t0) / 60), flush=True)

    date = datetime.datetime.now().strftime("%Y-%m-%d")
    here = os.path.dirname(os.path.abspath(__file__))
    # Tag the output with the subset that was run, so a partial run (BENCH_KIND
    # or BENCH_ONLY) cannot overwrite the arrays of a different subset saved on
    # the same day.
    tag = kind_filter or ("-".join(sorted(only)) if only else "all")
    tag = "".join(c if c.isalnum() or c in "-_" else "_" for c in tag)
    np.save(os.path.join(here, "%s-benchmark-%s-point.npy" % (date, tag)), point)
    np.save(os.path.join(here, "%s-benchmark-%s-dist.npy" % (date, tag)), dist)
    with open(os.path.join(here, "%s-benchmark-%s-methods.txt" % (date, tag)), "w") as f:
        f.write("\n".join("%s\t%s" % (m[0], m[1]) for m in methods) + "\n")
    print("saved arrays for %d methods" % nM, flush=True)


if __name__ == "__main__":
    main()
