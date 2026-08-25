"""Compare the R and Python parity sweeps.

    python compare_parity_sweep.py parity-r.csv parity-python.csv

Each row is  series|method|model|logLik|nparam|AICc|seconds.
Reports, per method: how many series agree on the selected model, and how
closely the log-likelihoods match where they do.
"""

import sys
from collections import defaultdict


def load(path):
    out = {}
    with open(path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split("|")
            if len(parts) != 7:
                continue
            out[(parts[0], parts[1])] = parts[2:]
    return out


def main():
    r_rows, py_rows = load(sys.argv[1]), load(sys.argv[2])
    keys = sorted(set(r_rows) & set(py_rows))
    only_r = sorted(set(r_rows) - set(py_rows))
    only_py = sorted(set(py_rows) - set(r_rows))

    stats = defaultdict(
        lambda: {"n": 0, "model": 0, "ll": [], "errors": 0, "worst": []}
    )
    for key in keys:
        method = key[1]
        r_model, r_ll = r_rows[key][0], r_rows[key][1]
        py_model, py_ll = py_rows[key][0], py_rows[key][1]
        entry = stats[method]
        entry["n"] += 1
        if r_model.startswith("ERROR") or py_model.startswith("ERROR"):
            entry["errors"] += 1
            continue
        if r_model == py_model:
            entry["model"] += 1
            try:
                diff = abs(float(r_ll) - float(py_ll))
            except ValueError:
                continue
            entry["ll"].append(diff)
            entry["worst"].append((diff, key[0], r_model))
        else:
            entry["worst"].append((float("inf"), key[0], f"{r_model} != {py_model}"))

    print(f"compared {len(keys)} (series, method) pairs")
    if only_r or only_py:
        print(f"  only in R: {len(only_r)}   only in Python: {len(only_py)}")

    for method in sorted(stats):
        e = stats[method]
        comparable = e["n"] - e["errors"]
        agree = e["model"]
        print(f"\n{method}:")
        print(f"  series compared      : {e['n']}  (errors: {e['errors']})")
        pct = 100 * agree / comparable if comparable else 0.0
        print(f"  same model selected  : {agree}/{comparable}  ({pct:.2f}%)")
        if e["ll"]:
            worst = max(e["ll"])
            over = sum(1 for d in e["ll"] if d > 1e-6)
            print(f"  max |dLogLik|        : {worst:.3e}")
            print(f"  |dLogLik| > 1e-6     : {over}")
        mismatches = sorted(e["worst"], reverse=True)[:10]
        shown = [m for m in mismatches if m[0] > 1e-6]
        if shown:
            print("  largest disagreements:")
            for diff, name, detail in shown:
                label = "model" if diff == float("inf") else f"{diff:.3e}"
                print(f"    {name:10s} {label:>12s}  {detail}")


if __name__ == "__main__":
    main()
