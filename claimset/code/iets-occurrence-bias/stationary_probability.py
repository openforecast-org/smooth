"""Long-run mean occurrence probability of smooth's occurrence filters and Croston's method.

Compiles filter.cpp against smooth's src/headers/ssOccurrence.h and runs it on a grid of
(model, error, p, alpha). Grid and sample sizes come from a JSON file (first argument).
Writes the results to $CLAIMSET_RESULTS and a CSV to $CLAIMSET_RUN_DIR.
"""

import csv
import itertools
import json
import os
import subprocess
import sys
import tempfile

here = os.path.dirname(os.path.abspath(__file__))
outdir = os.environ.get("CLAIMSET_RUN_DIR") or tempfile.mkdtemp()
binary = os.path.join(outdir, "filter")
subprocess.run(["g++", "-O2", "-std=c++11", "-o", binary, os.path.join(here, "filter.cpp")], check=True)

with open(sys.argv[1]) as f:
    grid = json.load(f)

results, rows = [], []
for k, (model, error, p, alpha) in enumerate(
        itertools.product(grid["models"], grid["errors"], grid["p"], grid["alpha"])):
    if model == "c" and error == "A":
        continue
    out = subprocess.run([binary, model, error, str(p), str(alpha), str(grid["T"]),
                          str(grid["burnin"]), str(grid["seed"] + k)],
                         check=True, capture_output=True, text=True).stdout.split()
    mean, nonfinite = float(out[0]), int(out[1])
    name = {"i": "iETS_I", "o": "iETS_O", "c": "Croston"}[model]
    cond = {"model": name, "error": error, "p": p, "alpha": alpha}
    results += [{"quantity": "stationary mean probability", "value": mean, "conditions": cond},
                {"quantity": "ratio of stationary mean probability to p", "value": mean / p,
                 "conditions": cond}]
    rows.append({**cond, "mean_p_hat": mean, "ratio": mean / p, "nonfinite": nonfinite})

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
