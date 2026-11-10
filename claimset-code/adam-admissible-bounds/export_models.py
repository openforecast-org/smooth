"""Write the models of a planned test, with the prototype's values for each, to models.json.

The model sets are drawn exactly as confirm.py (sets perlag, cancel, stepdown) and cpp_check.py (set cpp) draw them
for the same seed and n; set examples holds the two models of explore.py's ARIMA-form example. Each set's results
repeat the counts of the original test, so a run shows that the exported models are the tested ones.
"""

import argparse
import json
import os

import numpy as np

from admissibility import (arima_form, cancel, ets, gum_random, gum_start, max_lambda_roots, pbh_max, per_lag_max,
                           schur_cohn, spectral_radius, trim, write_results)

parser = argparse.ArgumentParser()
parser.add_argument("--set", required=True, choices=["perlag", "cancel", "stepdown", "cpp", "examples"])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n", type=int, default=200)
args = parser.parse_args()
rng = np.random.default_rng(args.seed)
EDGE = 1e-8


# The generators of confirm.py and cpp_check.py, unchanged
def ets_random(seasonalChoices=(4, 7, 12, 24, 52)):
    seasonal = sorted(rng.choice(seasonalChoices, size=rng.integers(0, 3), replace=False).tolist())
    trend = rng.choice(["N", "A", "Ad"])
    beta = None if trend == "N" else (0.0 if rng.random() < 0.1 else rng.uniform(0, 0.5))
    gamma = [0.0 if rng.random() < 0.15 else rng.uniform(0.01, 1.5) for _ in seasonal]
    return ets(rng.uniform(0.01, 1.9), seasonal, gamma, beta,
               phi=rng.uniform(0.8, 0.99) if trend == "Ad" else 1.0, constant=rng.random() < 0.3)


def gum_identity():
    lags = sorted(rng.choice([1, 4, 12], size=rng.integers(2, 5)).tolist())
    k = len(lags)
    return np.eye(k), rng.uniform(-0.3, 0.6, k), rng.normal(1, 0.3, k), lags


def model_set(n):
    return ([("ETS", ets_random()) for _ in range(n)] + [("GUM identity", gum_identity()) for _ in range(n // 2)] +
            [("GUM random", gum_random(sorted(rng.choice([1, 4, 12, 24], size=rng.integers(2, 5)).tolist()), rng,
                                       sd=0.35)) for _ in range(n)])


def high_degree_set(n):
    return ([("ETS high degree", ets_random(seasonalChoices=(24, 168, 365))) for _ in range(n)] +
            [("GUM high degree", gum_random([1, 24, 168], rng, sd=0.3)) for _ in range(n // 2)])


def cpp_set(n):
    families = {
        "ETS": lambda: ets_random((4, 7, 12, 24, 52)),
        "ETS high degree": lambda: ets_random((24, 168, 365)),
        "GUM identity": lambda: gum_start(sorted(rng.choice([1, 4, 12], size=rng.integers(2, 5)).tolist()),
                                          g=rng.uniform(-0.3, 0.6)),
        "GUM random": lambda: gum_random(sorted(rng.choice([1, 4, 12, 24], size=rng.integers(2, 5)).tolist()), rng,
                                         sd=0.35),
    }
    return [(name, make()) for name, make in families.items() for _ in range(n)]


def cancelled(model):
    theta, phi = arima_form(*model)
    thetaStar, phiStar, _ = cancel(theta, phi, model[3])
    return trim(theta), trim(phi), trim(thetaStar), trim(phiStar)


def count(name, value, **conditions):
    return {"quantity": name, "value": float(value), "conditions": conditions}


if args.set == "perlag":
    grid = np.arange(0.05, 2.0, 0.05)
    models = ([("ETS(A,N,A)", ets(a, [m], [gm])) for m in (4, 12) for a in grid for gm in grid] +
              [("GUM", gum_random([1, 12], rng, sd=0.35)) for _ in range(args.n)])
elif args.set == "cancel":
    models = model_set(args.n)
elif args.set == "stepdown":
    models = model_set(args.n) + high_degree_set(args.n // 4)
elif args.set == "cpp":
    models = cpp_set(args.n)
else:
    models = [("ETS(A,N,A,A), alpha=0.3, gamma=(0.1,0.05)", ets(0.3, [24, 168], [0.1, 0.05])),
              ("GUM start, orders (2,2), lags (1,12)", gum_start([1, 1, 12, 12]))]

rows = []
for i, (family, model) in enumerate(models):
    F, g, w, lags = model
    row = {"index": i, "family": family, "F": F.tolist(), "g": g.tolist(), "w": w.tolist(),
           "lags": [int(l) for l in lags], "prototype": {}}
    p = row["prototype"]
    if args.set == "perlag":
        p["per_lag_max"] = float(per_lag_max(*model))
        p["pbh_max"] = float(pbh_max(*model))
    else:
        theta, phi, thetaStar, phiStar = cancelled(model)
        p["degree_theta"], p["degree_phi"] = len(theta) - 1, len(phi) - 1
        p["degree_theta_cancelled"], p["degree_phi_cancelled"] = len(thetaStar) - 1, len(phiStar) - 1
        p["max_lambda_cancelled"] = float(max_lambda_roots(thetaStar))
        if args.set in ("cancel", "examples"):
            p["pbh_max"] = float(pbh_max(*model))
        if args.set == "stepdown":
            p["stepdown_admissible"] = bool(schur_cohn(thetaStar))
            if p["max_lambda_cancelled"] > 1:
                p["bisection_max_lambda"] = float(spectral_radius(thetaStar))
    rows.append(row)

with open(os.path.join(os.environ.get("CLAIMSET_RUN_DIR", "."), "models.json"), "w") as f:
    json.dump({"set": args.set, "seed": args.seed, "n": args.n, "models": rows}, f)

# The counts of the original test, from the exported values
results = []
P = [r["prototype"] for r in rows]
if args.set == "perlag":
    for name in ("ETS(A,N,A)", "GUM"):
        sub = [p for r, p in zip(rows, P) if r["family"] == name]
        results += [count("models accepted by the per-lag check that do not forget their initial states",
                          sum(p["per_lag_max"] < 1 and p["pbh_max"] > 1 + EDGE for p in sub), family=name,
                          models=len(sub)),
                    count("models rejected by the per-lag check that forget their initial states",
                          sum(p["per_lag_max"] >= 1 and p["pbh_max"] < 1 - EDGE for p in sub), family=name,
                          models=len(sub))]
elif args.set in ("cancel", "examples"):
    kept = [p for p in P if abs(p["pbh_max"] - 1) >= EDGE]
    results += [count("admissibility decisions that differ from the PBH reference",
                      sum((p["pbh_max"] < 1) != (p["max_lambda_cancelled"] < 1) for p in kept)),
                count("models within 1e-8 of the unit circle, excluded", len(P) - len(kept))]
elif args.set == "stepdown":
    kept = [p for p in P if abs(p["max_lambda_cancelled"] - 1) >= EDGE]
    results += [count("step-down decisions that differ from the decision by the roots",
                      sum(p["stepdown_admissible"] != (p["max_lambda_cancelled"] < 1) for p in kept)),
                count("max abs difference of max |lambda|, bisection against roots",
                      max([abs(p["bisection_max_lambda"] - p["max_lambda_cancelled"]) for p in kept
                           if "bisection_max_lambda" in p], default=0.0)),
                count("models within 1e-8 of the unit circle, excluded", len(P) - len(kept)),
                count("max degree of theta before cancellation", max(p["degree_theta"] for p in P))]
else:
    results += [count("prototype modulus below 1", sum(p["max_lambda_cancelled"] < 1 for p in P))]
results.append(count("models exported", len(rows)))
write_results(results)
