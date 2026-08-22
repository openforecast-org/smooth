"""CES point-forecast benchmark across implementations, on M1 + M3 + Tourism.

Same harness and settings as the ETS benchmark: fit on ``series.x``, forecast
``series.h`` steps, score against ``series.xx``, and record RMSSE, SAME and
wall-clock time. Saves ``{date}-benchmark-ces.npy`` of shape
(n_methods, n_series, 3) -> [..., 0]=RMSSE, [..., 1]=SAME, [..., 2]=seconds.

Only ``smooth`` and ``statsforecast`` implement CES natively; sktime's
``StatsForecastAutoCES`` is a thin wrapper over the statsforecast routine and is
included so the two rows can be read as a harness check rather than as two
independent implementations. skforecast and aeon have no CES at all.
"""

import datetime
import multiprocessing
import os
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

METHODS = [
    # smooth AutoCES: seasonality selected by AICc, one row per initialisation
    ("CES Back", "smooth-auto", {"initial": "backcasting"}),
    ("CES Opt", "smooth-auto", {"initial": "optimal"}),
    ("CES Two", "smooth-auto", {"initial": "two-stage"}),
    # smooth CES with the seasonality fixed -- the members AutoCES chooses among
    ("CES none", "smooth-fixed", {"seasonality": "none", "initial": "backcasting"}),
    ("CES simple", "smooth-fixed", {"seasonality": "simple", "initial": "backcasting"}),
    (
        "CES partial",
        "smooth-fixed",
        {"seasonality": "partial", "initial": "backcasting"},
    ),
    ("CES full", "smooth-fixed", {"seasonality": "full", "initial": "backcasting"}),
    # other packages
    ("statsforecast AutoCES", "statsforecast", {}),
    ("sktime AutoCES", "sktime", {}),
]


def _lags(s):
    return [1, s.period] if s.period > 1 else [1]


def _fc_smooth_auto(cfg, s):
    from smooth import AutoCES

    m = AutoCES(lags=_lags(s), initial=cfg["initial"])
    m.fit(s.x)
    return np.asarray(m.predict(h=s.h)["mean"]).ravel()


def _fc_smooth_fixed(cfg, s):
    from smooth import CES

    seasonality = cfg["seasonality"]
    # A seasonal CES needs a seasonal lag; on yearly data fall back to "none",
    # the same way the ETS benchmark drops the seasonal letter when period == 1.
    if s.period <= 1:
        seasonality = "none"
    m = CES(seasonality=seasonality, lags=_lags(s), initial=cfg["initial"])
    m.fit(s.x)
    return np.asarray(m.predict(h=s.h)["mean"]).ravel()


def _fc_statsforecast(s):
    from statsforecast.models import AutoCES

    m = AutoCES(season_length=s.period if s.period > 1 else 1)
    m.fit(np.asarray(s.x, dtype=float))
    fc = m.predict(h=s.h)
    return np.asarray(fc["mean"] if isinstance(fc, dict) else fc).ravel()


def _fc_sktime(s):
    import pandas as pd
    from sktime.forecasting.statsforecast import StatsForecastAutoCES

    m = StatsForecastAutoCES(season_length=s.period if s.period > 1 else 1)
    m.fit(pd.Series(np.asarray(s.x, dtype=float)))
    return np.asarray(m.predict(np.arange(1, s.h + 1))).ravel()


def _task(args):
    warnings.filterwarnings("ignore")
    j, i, s, pkg, cfg = args
    try:
        t0 = time.time()
        if pkg == "smooth-auto":
            fc = _fc_smooth_auto(cfg, s)
        elif pkg == "smooth-fixed":
            fc = _fc_smooth_fixed(cfg, s)
        elif pkg == "statsforecast":
            fc = _fc_statsforecast(s)
        elif pkg == "sktime":
            fc = _fc_sktime(s)
        else:
            raise ValueError(f"unknown package {pkg}")
        elapsed = time.time() - t0

        fc = np.asarray(fc, dtype=float).ravel()[: s.h]
        y = np.asarray(s.xx, dtype=float)
        x = np.asarray(s.x, dtype=float)
        if fc.shape != y.shape or not np.all(np.isfinite(fc)):
            return (j, i, np.nan, np.nan, np.nan)

        scale_sq = np.mean(np.diff(x) ** 2)
        scale_abs = np.mean(np.abs(np.diff(x)))
        rmsse = (
            float(np.sqrt(np.mean((y - fc) ** 2) / scale_sq)) if scale_sq else np.nan
        )
        same = float(np.abs(np.mean(y - fc)) / scale_abs) if scale_abs else np.nan
        return (j, i, rmsse, same, elapsed)
    except Exception:
        return (j, i, np.nan, np.nan, np.nan)


def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from fcompdata import M1, M3, Tourism

    datasets = (
        [M1[k] for k in M1.keys()]
        + [M3[k] for k in M3.keys()]
        + [Tourism[k] for k in Tourism.keys()]
    )
    limit = int(os.environ.get("BENCH_LIMIT", "0"))
    if limit:
        datasets = datasets[:limit] + datasets[1001 : 1001 + limit] + datasets[-limit:]

    n_series, n_methods = len(datasets), len(METHODS)
    out = np.full((n_methods, n_series, 3), np.nan)
    tasks = [
        (j, i, s, pkg, cfg)
        for j, (_, pkg, cfg) in enumerate(METHODS)
        for i, s in enumerate(datasets)
    ]
    ncap = int(os.environ.get("BENCH_WORKERS", min(30, multiprocessing.cpu_count())))
    print(
        f"CES benchmark: {n_methods} methods x {n_series} series = "
        f"{len(tasks)} tasks, {ncap} workers",
        flush=True,
    )

    t0 = time.time()
    done = 0
    ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=ncap, mp_context=ctx) as ex:
        futures = [ex.submit(_task, t) for t in tasks]
        for f in as_completed(futures):
            j, i, rmsse, same, elapsed = f.result()
            out[j, i, :] = (rmsse, same, elapsed)
            done += 1
            if done % 5000 == 0:
                print(
                    f"  {done}/{len(tasks)} ({100 * done / len(tasks):.1f}%) "
                    f"{(time.time() - t0) / 60:.1f}min",
                    flush=True,
                )

    date = datetime.datetime.now().strftime("%Y-%m-%d")
    path = os.path.join(os.path.dirname(__file__), f"{date}-benchmark-ces.npy")
    np.save(path, out)
    print(f"saved {path} in {(time.time() - t0) / 60:.1f} min", flush=True)

    for j, (name, _, _) in enumerate(METHODS):
        print(
            f"  {name:24s} RMSSE={np.nanmean(out[j, :, 0]):.4f} "
            f"SAME={np.nanmean(out[j, :, 1]):.4f} "
            f"Time={np.nanmean(out[j, :, 2]):.3f}s "
            f"Failed={int(np.sum(np.isnan(out[j, :, 0])))}",
            flush=True,
        )


if __name__ == "__main__":
    main()
