"""Long-run mean occurrence probability of smooth's occurrence filters and Croston's method.

Compiles filter.cpp against smooth's src/headers/ssOccurrence.h and runs it on a grid of
(model, error, p, alpha). Grid and sample sizes come from a JSON file (first argument).
With "predict": true in the grid, also reports, per (model, error), the largest relative
deviation of the simulated mean from the predicted fixed point (fixed_point()).
Writes the results to $CLAIMSET_RESULTS and a CSV to $CLAIMSET_RUN_DIR.
"""

import csv
import itertools
import json
import math
import os
import subprocess
import sys
import tempfile

NAMES = {"i": "iETS_I", "o": "iETS_O", "c": "Croston"}


def bisect(f, lo=1e-12, hi=1 - 1e-12):
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(lo) * f(mid) > 0 else (lo, mid)
    return (lo + hi) / 2


def fixed_point(model, error, p):
    # p_hat at which the expected state update is zero (alpha -> 0), given o_t ~ Bernoulli(p)
    if model == "c":
        return p
    if error == "M":
        return 1 - math.sqrt(1 - p) if model == "i" else math.sqrt(p)
    # Additive error: E[log(proxy)] = 0; identical for 'i' and 'o' (the proxies are reciprocal)
    return bisect(lambda x: p * math.log((2 - x) / x) + (1 - p) * math.log((1 - x) / (1 + x)))


def simulate(binary, model, error, p, alpha, grid, seed):
    out = subprocess.run([binary, model, error, str(p), str(alpha), str(grid["T"]),
                          str(grid["burnin"]), str(seed)],
                         check=True, capture_output=True, text=True).stdout.split()
    return float(out[0]), int(out[1])


def main(grid_path):
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.environ.get("CLAIMSET_RUN_DIR") or tempfile.mkdtemp()
    binary = os.path.join(outdir, "filter")
    subprocess.run(["g++", "-O2", "-std=c++11", "-o", binary,
                    os.path.join(here, "filter.cpp")], check=True)
    with open(grid_path) as f:
        grid = json.load(f)

    results, rows, worst = [], [], {}
    cells = itertools.product(grid["models"], grid["errors"], grid["p"], grid["alpha"])
    for k, (model, error, p, alpha) in enumerate(cells):
        if model == "c" and error == "A":
            continue
        mean, nonfinite = simulate(binary, model, error, p, alpha, grid, grid["seed"] + k)
        cond = {"model": NAMES[model], "error": error, "p": p, "alpha": alpha}
        results += [{"quantity": "stationary mean probability", "value": mean,
                     "conditions": cond},
                    {"quantity": "ratio of stationary mean probability to p",
                     "value": mean / p, "conditions": cond}]
        row = {**cond, "mean_p_hat": mean, "ratio": mean / p, "nonfinite": nonfinite}
        if grid.get("predict"):
            pred = fixed_point(model, error, p)
            row.update(predicted=pred, rel_dev=abs(mean - pred) / pred)
            key = (NAMES[model], error)
            worst[key] = max(worst.get(key, 0), row["rel_dev"], nonfinite)
        rows.append(row)

    for (name, error), dev in worst.items():
        results.append({"quantity": "max relative deviation from predicted fixed point",
                        "value": dev, "conditions": {"model": name, "error": error}})

    with open(os.path.join(outdir, "stationary_probability.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(r)
    path = os.environ.get("CLAIMSET_RESULTS")
    if path:
        with open(path, "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    main(sys.argv[1])
