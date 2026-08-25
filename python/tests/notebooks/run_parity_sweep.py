"""R/Python parity sweep over M1 + M3 + Tourism.

Fits ADAM ETS and AutoMSARIMA to every competition series and records what the
two languages must agree on: the selected model, the log-likelihood, the
parameter count and AICc. This is a *parity* check, not an accuracy benchmark --
see run_full_benchmark.py for the latter.

The R counterpart is run_parity_sweep.R; compare the two outputs with
compare_parity_sweep.py.

Environment:
  PARITY_WORKERS  process pool size (default: min(30, cpu_count))
  PARITY_LIMIT    first N series of each dataset (smoke test)
  PARITY_OUT      output path (default: parity-python.csv)
"""

import os

# Pin BLAS before numpy is imported: each pool worker would otherwise spawn its
# own thread pool on top of the process pool.
for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import multiprocessing  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")


def _lags(period):
    return [1, period] if period > 1 else [1]


def _fit(args):
    """Return one CSV row per method for a single series."""
    name, values, period = args
    from smooth import ADAM, AutoMSARIMA

    y = np.asarray(values, dtype=float)
    rows = []

    # ADAM ETS: drop the seasonal letter when the data is not seasonal, the way
    # the accuracy benchmark does.
    spec = "ZXZ" if period > 1 else "ZXN"
    for label, build in (
        ("ETS", lambda: ADAM(model=spec, lags=_lags(period), initial="backcasting")),
        (
            "ARIMA",
            lambda: AutoMSARIMA(lags=_lags(period), initial="backcasting"),
        ),
    ):
        try:
            t0 = time.time()
            model = build()
            model.fit(y)
            rows.append(
                f"{name}|{label}|{model.model_name}|{model.loglik:.10f}"
                f"|{model.nparam:g}|{model.aicc:.6f}|{time.time() - t0:.3f}"
            )
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            rows.append(f"{name}|{label}|ERROR:{type(exc).__name__}|nan|nan|nan|nan")
    return rows


def main():
    from fcompdata import M1, M3, Tourism

    limit = int(os.environ.get("PARITY_LIMIT", "0"))
    tasks = []
    for dataset in (M1, M3, Tourism):
        keys = list(dataset.keys())
        if limit:
            keys = keys[:limit]
        for key in keys:
            series = dataset[key]
            tasks.append((series.sn, np.asarray(series.x, dtype=float), series.period))

    workers = int(
        os.environ.get("PARITY_WORKERS", min(30, multiprocessing.cpu_count()))
    )
    out_path = os.environ.get("PARITY_OUT", "parity-python.csv")
    print(f"{len(tasks)} series, {workers} workers -> {out_path}", flush=True)

    start = time.time()
    done = 0
    pool = ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("fork")
    )
    with open(out_path, "w") as handle:
        futures = [pool.submit(_fit, t) for t in tasks]
        for future in as_completed(futures):
            for row in future.result():
                handle.write(row + "\n")
            done += 1
            if done % 250 == 0:
                elapsed = (time.time() - start) / 60
                rate = done / elapsed if elapsed else 0
                print(
                    f"  {done}/{len(tasks)} ({100 * done / len(tasks):.1f}%) "
                    f"{elapsed:.1f}min, ~{(len(tasks) - done) / rate:.1f}min left",
                    flush=True,
                )
    pool.shutdown()
    print(f"done in {(time.time() - start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
