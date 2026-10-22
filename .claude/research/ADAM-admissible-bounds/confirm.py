"""Tests of the admissibility check on random model sets (one test per call)."""

import argparse

import numpy as np

from admissibility import (arima_form, cancel, ets, gum_random, max_lambda_roots, pbh_max, per_lag_max,
                           schur_cohn, spectral_radius, write_results)

parser = argparse.ArgumentParser()
parser.add_argument("--test", required=True, choices=["perlag", "cancel", "stepdown", "bisection"])
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--n", type=int, default=200)
args = parser.parse_args()
rng = np.random.default_rng(args.seed)
EDGE = 1e-8


# Additive ETS: level, optional (damped) trend, up to two seasonals, optional constant, some persistence fixed at 0
def ets_random(seasonalChoices=(4, 7, 12, 24, 52)):
    seasonal = sorted(rng.choice(seasonalChoices, size=rng.integers(0, 3), replace=False).tolist())
    trend = rng.choice(["N", "A", "Ad"])
    beta = None if trend == "N" else (0.0 if rng.random() < 0.1 else rng.uniform(0, 0.5))
    gamma = [0.0 if rng.random() < 0.15 else rng.uniform(0.01, 1.5) for _ in seasonal]
    return ets(rng.uniform(0.01, 1.9), seasonal, gamma, beta,
               phi=rng.uniform(0.8, 0.99) if trend == "Ad" else 1.0, constant=rng.random() < 0.3)


# GUM at an identity transition (the default starting point), with random lags, persistence and measurement
def gum_identity():
    lags = sorted(rng.choice([1, 4, 12], size=rng.integers(2, 5)).tolist())
    k = len(lags)
    return np.eye(k), rng.uniform(-0.3, 0.6, k), rng.normal(1, 0.3, k), lags


def model_set(n):
    return ([ets_random() for _ in range(n)] + [gum_identity() for _ in range(n // 2)] +
            [gum_random(sorted(rng.choice([1, 4, 12, 24], size=rng.integers(2, 5)).tolist()), rng, sd=0.35)
             for _ in range(n)])


def high_degree_set(n):
    return ([ets_random(seasonalChoices=(24, 168, 365)) for _ in range(n)] +
            [gum_random([1, 24, 168], rng, sd=0.3) for _ in range(n // 2)])


def cancelled_theta(model):
    return cancel(*arima_form(*model), model[3])[0]


def count(name, value, **conditions):
    return {"quantity": name, "value": float(value), "conditions": conditions}


results = []
if args.test == "perlag":
    # ETS(A,N,A) grid and random GUMs with lags (1,12): the per-lag check against the reference
    grid = np.arange(0.05, 2.0, 0.05)
    etsModels = [ets(a, [m], [gm]) for m in (4, 12) for a in grid for gm in grid]
    gumModels = [gum_random([1, 12], rng, sd=0.35) for _ in range(args.n)]
    for name, models in [("ETS(A,N,A)", etsModels), ("GUM", gumModels)]:
        perLag = np.array([per_lag_max(*model) for model in models])
        reference = np.array([pbh_max(*model) for model in models])
        results += [count("models accepted by the per-lag check that do not forget their initial states",
                          np.sum((perLag < 1) & (reference > 1 + EDGE)), family=name, models=len(models)),
                    count("models rejected by the per-lag check that forget their initial states",
                          np.sum((perLag >= 1) & (reference < 1 - EDGE)), family=name, models=len(models)),
                    count("models accepted by the per-lag check", np.sum(perLag < 1), family=name,
                          models=len(models))]
elif args.test == "cancel":
    # Roots of theta after cyclotomic cancellation against the PBH reference
    mismatches, edge, inadmissible, diffs, models = 0, 0, 0, [0.0], model_set(args.n)
    for model in models:
        reference, cancelled = pbh_max(*model), max_lambda_roots(cancelled_theta(model))
        if abs(reference - 1) < EDGE:
            edge += 1
            continue
        mismatches += (reference < 1) != (cancelled < 1)
        if reference >= 1 and cancelled >= 1:
            inadmissible += 1
            diffs.append(abs(reference - cancelled))
    results += [count("admissibility decisions that differ from the PBH reference", mismatches),
                count("max abs difference of max |lambda| from the PBH reference, inadmissible models",
                      max(diffs), inadmissible_models=inadmissible),
                count("models within 1e-8 of the unit circle, excluded", edge),
                count("models compared", len(models) - edge)]
else:
    # Step-down decision and bisection against the roots of the cancelled theta
    mismatches, edge, diffs, models = 0, 0, [0.0], model_set(args.n) + high_degree_set(args.n // 4)
    for model in models:
        theta = cancelled_theta(model)
        rootMax = max_lambda_roots(theta)
        if abs(rootMax - 1) < EDGE:
            edge += 1
            continue
        if args.test == "stepdown":
            mismatches += schur_cohn(theta) != (rootMax < 1)
        elif rootMax > 1:
            diffs.append(abs(spectral_radius(theta) - rootMax))
    results += ([count("step-down decisions that differ from the decision by the roots", mismatches)]
                if args.test == "stepdown" else
                [count("max abs difference of max |lambda|, bisection against roots", max(diffs),
                       inadmissible_models=len(diffs) - 1)])
    results += [count("models within 1e-8 of the unit circle, excluded", edge),
                count("models compared", len(models) - edge)]

write_results(results)
