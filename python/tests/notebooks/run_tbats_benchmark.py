"""TBATS benchmark on M4 hourly: data, the Python forecasters and the scores.

    python run_tbats_benchmark.py data                    # download and verify M4 hourly
    python run_tbats_benchmark.py run smooth smooth-dnorm statsforecast [--workers 4]
    python run_tbats_benchmark.py score OUT.csv           # score every method in the cache

The R forecasters (forecast::tbats and smooth's R tbats()) are run by
run_tbats_benchmark.R and write the same files. Each method writes one CSV per series
to CACHE/tbats-m4-hourly/<method>/<series>.csv: the point forecast and the 99
quantiles over the horizon, with the fit time and the model, so an interrupted run
resumes where it stopped and all the methods are scored by the same code.
"""

import hashlib
import os
import signal
import subprocess
import sys
import time
import urllib.request
import warnings
from concurrent.futures import ProcessPoolExecutor

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd

CACHE = os.environ.get("SMOOTH_BENCH_CACHE", os.path.expanduser("~/.cache/smooth-benchmarks"))
DATA = os.path.join(CACHE, "m4")
FORECASTS = os.path.join(CACHE, "tbats-m4-hourly")
URL = "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/{}"
FILES = {
    "Hourly-train.csv": (
        "Train/Hourly-train.csv",
        "ea59b7783573c49077a835ab6465c7d66f1474783360f310988a9a737fbca62f",
    ),
    "Hourly-test.csv": (
        "Test/Hourly-test.csv",
        "71a57fccb15e534d973626ada4fb87febf789e9317715b7cc6dd3b5f90db6f42",
    ),
}
H = 48
LAGS = [1, 24, 168]
LEVELS = np.round(np.arange(0.01, 1.0, 0.01), 2)  # the 99 quantile levels
TWO = [round(1 - 2 * t, 2) for t in LEVELS if t < 0.5]  # two-sided levels 0.98..0.02
TIMEOUT = int(os.environ.get("BENCH_TASK_TIMEOUT", "1800"))
QCOLS = [f"q{int(round(100 * t)):02d}" for t in LEVELS]


def data():
    # The M4 hourly series as {id: (in-sample, holdout)}, downloaded once and verified.
    os.makedirs(DATA, exist_ok=True)
    for name, (path, sha) in FILES.items():
        target = os.path.join(DATA, name)
        if not os.path.exists(target):
            urllib.request.urlretrieve(URL.format(path), target)
        digest = hashlib.sha256(open(target, "rb").read()).hexdigest()
        if digest != sha:
            raise RuntimeError(f"{name}: sha256 {digest} differs from {sha}")
    train = pd.read_csv(os.path.join(DATA, "Hourly-train.csv"), index_col=0)
    test = pd.read_csv(os.path.join(DATA, "Hourly-test.csv"), index_col=0)
    return {
        sid: (train.loc[sid].dropna().to_numpy(float), test.loc[sid].to_numpy(float))
        for sid in train.index
    }


def quantiles(point, lower, upper):
    # The 99 quantiles from the two-sided intervals of the levels TWO, the point at 0.5.
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    return np.column_stack([lower, np.asarray(point, float), upper[:, ::-1]])


def forecast_smooth(x, distribution="auto"):
    from smooth import TBATS

    model = TBATS(lags=LAGS, distribution=distribution).fit(x)
    fc = model.predict(h=H, interval="prediction", level=TWO, side="both")
    point = np.asarray(fc.mean, float).ravel()
    return point, quantiles(point, fc.lower, fc.upper), f"{model.model_name} {model.distribution_}"


def forecast_smooth_dnorm(x):
    return forecast_smooth(x, "dnorm")


def forecast_statsforecast(x):
    from statsforecast.models import AutoTBATS

    model = AutoTBATS(season_length=LAGS[1:]).fit(x)
    percents = [int(round(100 * level)) for level in TWO]
    fc = model.predict(h=H, level=percents)
    lower = np.column_stack([fc[f"lo-{p}"] for p in percents])
    upper = np.column_stack([fc[f"hi-{p}"] for p in percents])
    fit, use = model.model_, model.model_["description"]
    lam = "-" if fit["BoxCox_lambda"] is None else f"{fit['BoxCox_lambda']:.3g}"
    trend = "damped" if use["use_damped_trend"] else ("trend" if use["use_trend"] else "-")
    k = ", ".join(f"<{m},{int(v)}>" for m, v in zip(LAGS[1:], fit["k_vector"]))
    name = f"TBATS({lam}, {{{fit['p']},{fit['q']}}}, {trend}, {k})"
    return fc["mean"], quantiles(fc["mean"], lower, upper), name


FORECASTERS = {
    "smooth": forecast_smooth,
    "smooth-dnorm": forecast_smooth_dnorm,
    "statsforecast": forecast_statsforecast,
}


def _alarm(signum, frame):
    raise TimeoutError(f"no forecast within {TIMEOUT}s")


def write(method, sid, n, seconds, model, Q):
    # The forecasts of one series, written atomically, as run_tbats_benchmark.R does.
    frame = pd.DataFrame(Q, columns=QCOLS)
    frame.insert(0, "point", Q[:, 49])
    frame.insert(0, "h", np.arange(1, Q.shape[0] + 1))
    frame.insert(0, "model", model)
    frame.insert(0, "time", seconds)
    frame.insert(0, "n", n)
    frame.insert(0, "series", sid)
    path = os.path.join(FORECASTS, method, f"{sid}.csv")
    frame.to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)


def task(args):
    # Fit and forecast one series, timing both; a failure is written with NaN quantiles.
    method, sid, x = args
    warnings.filterwarnings("ignore")
    signal.signal(signal.SIGALRM, _alarm)
    start = time.perf_counter()
    try:
        signal.alarm(TIMEOUT)
        point, Q, model = FORECASTERS[method](x)
        signal.alarm(0)
    except Exception as error:
        signal.alarm(0)
        Q, model = np.full((H, 99), np.nan), f"ERROR: {type(error).__name__}: {error}"[:200]
    write(method, sid, len(x), time.perf_counter() - start, model, Q)
    return sid


def warm(method):
    # Compile numba and load the extensions before the timed fits.
    warnings.filterwarnings("ignore")
    t = np.arange(400.0)
    FORECASTERS[method](10 + np.sin(2 * np.pi * t / 24) + np.sin(2 * np.pi * t / 168) + t / 100)


def run(method, workers):
    series = data()
    os.makedirs(os.path.join(FORECASTS, method), exist_ok=True)
    done = {f[:-4] for f in os.listdir(os.path.join(FORECASTS, method)) if f.endswith(".csv")}
    todo = [(method, sid, x) for sid, (x, _) in series.items() if sid not in done]
    print(
        f"{method}: {len(todo)} of {len(series)} series to forecast, {workers} workers", flush=True
    )
    with ProcessPoolExecutor(workers, initializer=warm, initargs=(method,)) as pool:
        for i, _ in enumerate(pool.map(task, todo), 1):
            if i % 50 == 0:
                print(f"  {i}/{len(todo)}", flush=True)


def score_series(x, y, Q):
    # RMSSE, SAME and the scaled pinball and coverage at the 99 levels of one series,
    # all scaled by the in-sample one-step differences.
    scale_sq, scale_abs = np.mean(np.diff(x) ** 2), np.mean(np.abs(np.diff(x)))
    point = Q[:, 49]
    rmsse = np.sqrt(np.mean((y - point) ** 2) / scale_sq)
    same = np.abs(np.mean(y - point)) / scale_abs
    Q = np.sort(Q, axis=1)
    diff = y[:, None] - Q
    pinball = np.maximum(LEVELS * diff, (LEVELS - 1) * diff).mean(axis=0) / scale_abs
    coverage = (y[:, None] <= Q).mean(axis=0)
    return rmsse, same, pinball, coverage


def score(methods=None):
    # One row per method and series: the metrics, the per-level pinball and coverage,
    # the time and the model; NaN metrics for the failed forecasts.
    series = data()
    methods = methods or sorted(os.listdir(FORECASTS))
    rows = []
    for method in methods:
        for sid, (x, y) in series.items():
            path = os.path.join(FORECASTS, method, f"{sid}.csv")
            if not os.path.exists(path):
                continue
            frame = pd.read_csv(path)
            Q = frame[QCOLS].to_numpy(float)
            row = {
                "method": method,
                "series": sid,
                "n": len(x),
                "time": frame.time[0],
                "model": frame.model[0],
            }
            if np.all(np.isfinite(Q)):
                rmsse, same, pinball, coverage = score_series(x, y, Q)
                row.update(rmsse=rmsse, same=same, pinball=pinball.mean())
                row.update({f"pinball_{c[1:]}": v for c, v in zip(QCOLS, pinball)})
                row.update({f"coverage_{c[1:]}": v for c, v in zip(QCOLS, coverage)})
            rows.append(row)
    return pd.DataFrame(rows)


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
R_LIB = os.path.join(CACHE, "rlib")


def _shell(*command, **kwargs):
    return subprocess.run(command, capture_output=True, text=True, check=True, **kwargs).stdout


def versions():
    # The version of each method, with the git commit of the smooth working tree.
    from importlib.metadata import version

    commit = _shell("git", "-C", REPO, "describe", "--always", "--dirty").strip()
    r_version = _shell(
        "Rscript",
        "-e",
        f'cat(as.character(packageVersion("smooth", lib.loc="{R_LIB}")),'
        ' as.character(packageVersion("forecast")),'
        ' paste(R.version$major, R.version$minor, sep="."),'
        ' as.character(packageVersion("greybox")))',
    ).split()
    python = sys.version.split()[0]
    smooth = f"smooth {version('smooth')} (git {commit}), greybox {version('greybox')}"
    return {
        "smooth": f"{smooth}, Python {python}",
        "smooth-dnorm": f"{smooth}, Python {python}",
        "smooth-R": f"smooth {r_version[0]} (git {commit}), greybox {r_version[3]},"
        f" R {r_version[2]}",
        "statsforecast": f"statsforecast {version('statsforecast')}, Python {python}",
        "forecast": f"forecast {r_version[1]}, R {r_version[2]}",
    }


def run_all(output, workers):
    # The whole benchmark: the methods one after another (so their timings do not
    # compete), smooth's R package installed from this working tree, then the scores.
    data()
    for method in FORECASTERS:
        run(method, workers)
    os.makedirs(R_LIB, exist_ok=True)
    _shell("R", "CMD", "INSTALL", "--preclean", "-l", R_LIB, REPO)
    environment = dict(os.environ, SMOOTH_R_LIB=R_LIB, SMOOTH_BENCH_CACHE=CACHE)
    for method in ("smooth-R", "forecast"):
        print(
            _shell(
                "Rscript",
                os.path.join(HERE, "run_tbats_benchmark.R"),
                method,
                "--workers",
                str(workers),
                env=environment,
            ),
            flush=True,
        )
    scores = score(list(FORECASTERS) + ["smooth-R", "forecast"])
    scores.insert(1, "version", scores.method.map(versions()))
    scores.to_csv(output, index=False, float_format="%.6g")
    return scores


if __name__ == "__main__":
    command, arguments = sys.argv[1], sys.argv[2:]
    workers = int(arguments[arguments.index("--workers") + 1]) if "--workers" in arguments else 4
    if command == "data":
        print(f"{len(data())} series in {DATA}")
    elif command == "run":
        for name in [a for a in arguments if a in FORECASTERS]:
            run(name, workers)
    elif command == "score":
        score().to_csv(arguments[0], index=False, float_format="%.6g")
    elif command == "all":
        run_all(arguments[0], workers)
