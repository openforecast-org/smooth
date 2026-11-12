"""Whether the t-test of relative RMSSE can stand for PL7: on the per-series RMSSE of a
run of m4_hourly.py (its measures.csv), the 99% t-interval and the 99% percentile
bootstrap interval (10000 resamples of series, seed 1) of the mean log ratio of smooth
TBATS to each comparator, with the skewness and excess kurtosis of the log ratios (DC5).

Run: python ttest_check.py --measures RUN_DIR/measures.csv
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from m4_hourly import ALPHA, COMPARATORS, one_sided
from scipy import stats

RESAMPLES = 10000


def check(log_ratios, rng):
    mean, lower, upper, _, n = one_sided(log_ratios, 0.0)
    draws = rng.integers(0, n, (RESAMPLES, n))
    means = log_ratios[draws].mean(axis=1)
    b_lower, b_upper = np.quantile(means, [ALPHA / 2, 1 - ALPHA / 2])
    half = (upper - lower) / 2
    return {
        "skewness of the log ratios": stats.skew(log_ratios),
        "excess kurtosis of the log ratios": stats.kurtosis(log_ratios),
        "lower end, 99% t-interval of the mean log ratio": lower,
        "upper end, 99% t-interval of the mean log ratio": upper,
        "lower end, 99% bootstrap interval of the mean log ratio": b_lower,
        "upper end, 99% bootstrap interval of the mean log ratio": b_upper,
        "largest end difference, bootstrap vs t, in t half-widths": max(
            abs(b_lower - lower), abs(b_upper - upper)
        )
        / half,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--measures", required=True)
    args = parser.parse_args()
    table = pd.read_csv(args.measures)
    wide = table.pivot(index="series", columns="method", values="RMSSE")
    rng = np.random.default_rng(1)
    results = []
    for comparator in COMPARATORS:
        ratios = np.log(wide["smooth TBATS"] / wide[comparator]).to_numpy()
        ratios = ratios[np.isfinite(ratios)]
        conditions = {"method": "smooth TBATS", "comparator": comparator}
        results += [
            {"variable": k, "value": float(v), "conditions": conditions}
            for k, v in check(ratios, rng).items()
        ]
    worst = max(r["value"] for r in results if r["variable"].startswith("largest"))
    results.append(
        {
            "variable": "t-test stays (DC5: every end within 0.1 half-widths)",
            "value": bool(worst < 0.1),
        }
    )
    for r in results:
        print(r["variable"], r.get("conditions", ""), r["value"])
    path = os.environ.get("CLAIMSET_RESULTS")
    if path:
        with open(path, "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    main()
