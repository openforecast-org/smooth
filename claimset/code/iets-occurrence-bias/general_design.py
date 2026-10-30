"""Occurrence filters on a design of (p, alpha) drawn at random: limits as alpha -> 0,
direction of the bias and dominance of the score proxy over the old one.

The design file (first argument) sets the mode and everything the run uses:
- "limit": n_p values of p ~ U(p_range); for each, the stationary mean at each of
  `alphas`; least-squares line in alpha extrapolated to alpha = 0, compared with the
  prediction: the zero-drift point for the old proxy (stationary_probability.fixed_point),
  p for the score proxy and for Croston's method.
- "random": n_cells pairs, p ~ U(p_range), log(alpha) ~ U(log(alpha_range)); the bias
  mean - p of every filter in the design at each pair; direction and dominance counts.
A difference counts only if it exceeds `z` Monte Carlo standard errors (batch means).
Writes the results to $CLAIMSET_RESULTS and CSVs to $CLAIMSET_RUN_DIR.
"""

import csv
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

from stationary_probability import NAMES, build, fixed_point


def filters(design):
    for model in design["models"]:
        for error in (["M"] if model == "c" else design["errors"]):
            for proxy in (["old"] if model == "c" else design["proxies"]):
                yield model, error, proxy


def cells(design):
    rng = random.Random(design["seed"])
    lo, hi = design["p_range"]
    if design["mode"] == "limit":
        ps = [rng.uniform(lo, hi) for _ in range(design["n_p"])]
        return [(p, a) for p in ps for a in design["alphas"]]
    alo, ahi = design["alpha_range"]
    if design.get("alpha_scale", "log") == "uniform":
        return [(rng.uniform(lo, hi), rng.uniform(alo, ahi)) for _ in range(design["n_cells"])]
    alo, ahi = math.log(alo), math.log(ahi)
    return [(rng.uniform(lo, hi), math.exp(rng.uniform(alo, ahi)))
            for _ in range(design["n_cells"])]


def run_all(binary, design):
    jobs = [(f, c) for f in filters(design) for c in cells(design)]

    def one(k):
        (model, error, proxy), (p, alpha) = jobs[k]
        out = subprocess.run([binary, model, error, proxy, repr(p), repr(alpha),
                              str(design["T"]), str(design["burnin"]),
                              str(design["seed"] * 100000 + k), str(design["batches"])],
                             check=True, capture_output=True, text=True).stdout.split()
        return {"model": NAMES[model], "error": error, "proxy": proxy, "p": p,
                "alpha": alpha, "mean": float(out[0]), "se": float(out[1]),
                "nonfinite": int(out[2]), "key": (model, error, proxy)}

    with ThreadPoolExecutor(os.cpu_count()) as pool:
        return list(pool.map(one, range(len(jobs))))


def target(key, p):
    # The old proxy under ADAM ETS ('L') is the old additive filter (derivations.md §5)
    model, error, proxy = key
    return fixed_point(model, "A" if error == "L" else error, p) if proxy == "old" else p


def limits(rows, design):
    # Per filter and p: intercept of the least-squares line mean ~ alpha
    out, groups = [], {}
    for r in rows:
        groups.setdefault((r["key"], r["p"]), []).append(r)
    for (key, p), g in groups.items():
        a = [r["alpha"] for r in g]
        m = [r["mean"] for r in g]
        abar, mbar = sum(a) / len(a), sum(m) / len(m)
        slope = sum((x - abar) * (y - mbar) for x, y in zip(a, m)) / sum((x - abar) ** 2 for x in a)
        limit, pred = mbar - slope * abar, target(key, p)
        bad = any(r["nonfinite"] for r in g)
        out.append({**{k: g[0][k] for k in ("model", "error", "proxy")}, "p": p,
                    "limit": limit, "predicted": pred,
                    "rel_dev": 1.0 if bad else abs(limit - pred) / pred, "key": key})
    return out


def wrong_direction(r, z):
    # Old proxy: iETS_I (M) below p, iETS_O (M) above p, additive error between p and 0.5;
    # Croston above p. Wrong only by more than z standard errors.
    model, error, proxy = r["key"]
    d, tol, p = r["mean"] - r["p"], z * r["se"], r["p"]
    if r["nonfinite"]:
        return True
    if model == "c" or (model == "o" and error == "M"):
        return d < -tol
    if model == "i" and error == "M":
        return d > tol
    # Additive old proxy, and the old proxy under ADAM ETS (the same filter): towards 0.5
    toward = 1 if p < 0.5 else -1
    return toward * d < -tol or toward * (r["mean"] - 0.5) > tol


def dominance(rows, z):
    # Pairs of cells that differ only in the proxy: the score proxy is worse if its absolute
    # bias exceeds the old proxy's by more than z combined standard errors.
    old = {(r["model"], r["error"], r["p"], r["alpha"]): r for r in rows if r["proxy"] == "old"}
    out = []
    for s in (r for r in rows if r["proxy"] != "old"):
        o = old[(s["model"], s["error"], s["p"], s["alpha"])]
        gap = abs(s["mean"] - s["p"]) - abs(o["mean"] - o["p"])
        worse = s["nonfinite"] > 0 or gap > z * math.hypot(s["se"], o["se"])
        out.append({"model": s["model"], "error": s["error"], "proxy": s["proxy"], "p": s["p"],
                    "alpha": s["alpha"], "bias_score": s["mean"] - s["p"],
                    "bias_old": o["mean"] - o["p"], "score_worse": worse})
    return out


def count_by(rows, keys, flag):
    counts = {}
    for r in rows:
        k = tuple(r[c] for c in keys)
        counts[k] = counts.get(k, 0) + int(flag(r))
    return counts


def write_csv(outdir, name, rows):
    rows = [{k: v for k, v in r.items() if k != "key"} for r in rows]
    with open(os.path.join(outdir, name), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(design_path):
    outdir = os.environ.get("CLAIMSET_RUN_DIR") or tempfile.mkdtemp()
    with open(design_path) as f:
        design = json.load(f)
    rows = run_all(build(outdir), design)
    write_csv(outdir, "cells.csv", rows)
    results = [{"quantity": "stationary mean probability", "value": r["mean"],
                "interval": {"level": 0.95, "lower": r["mean"] - 1.96 * r["se"],
                             "upper": r["mean"] + 1.96 * r["se"]},
                "conditions": {k: r[k] for k in ("model", "error", "proxy", "p", "alpha")}}
               for r in rows]
    fkeys = ("model", "error", "proxy")

    if design["mode"] == "limit":
        lim = limits(rows, design)
        write_csv(outdir, "limits.csv", lim)
        worst = {}
        for r in lim:
            k = tuple(r[c] for c in fkeys)
            worst[k] = max(worst.get(k, 0.0), r["rel_dev"])
        results += [{"quantity": "max relative deviation of the alpha-to-0 limit from its prediction",
                     "value": v, "conditions": dict(zip(fkeys, k))} for k, v in worst.items()]
    else:
        wrong = count_by([r for r in rows if r["proxy"] == "old"], fkeys,
                         lambda r: wrong_direction(r, design["z"]))
        results += [{"quantity": "cells where the bias has the wrong direction", "value": v,
                     "conditions": dict(zip(fkeys, k))} for k, v in wrong.items()]
        if any(x != "old" for x in design["proxies"]):
            dom = dominance(rows, design["z"])
            write_csv(outdir, "dominance.csv", dom)
            worse = count_by(dom, ("model", "error"), lambda r: r["score_worse"])
            results += [{"quantity": "cells where the score proxy is more biased than the old proxy",
                         "value": v, "conditions": {"model": k[0], "error": k[1]}}
                        for k, v in worse.items()]
            results.append({"quantity": "cells where the score proxy is more biased than the old proxy",
                            "value": sum(worse.values()), "conditions": {"model": "all", "error": "all"}})

    for r in results:
        if r["quantity"] != "stationary mean probability":
            print(r)
    path = os.environ.get("CLAIMSET_RESULTS")
    if path:
        with open(path, "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    main(sys.argv[1])
