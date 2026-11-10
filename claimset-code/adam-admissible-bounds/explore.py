"""Exploration: the per-lag check, the ARIMA form, the cyclotomic cancellation and the step-down test."""

import argparse
import time

import numpy as np

from admissibility import (P, arima_form, cancel, ets, expand, gum_random, gum_start, max_lambda_roots,
                           pbh_max, per_lag_max, schur_cohn, spectral_radius, trim, write_results)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
rng = np.random.default_rng(parser.parse_args().seed)
results = []


def add(quantity, value, **conditions):
    results.append({"quantity": quantity, "value": float(value), "conditions": conditions})


def cancelled_max(model):
    theta, phi = arima_form(*model)
    return max_lambda_roots(cancel(theta, phi, model[3])[0])


# ETS(A,N,A), m = 12: the per-lag check against the modes that reach the forecasts
for a, gm in [(0.1, 0.1), (0.5, 1.4), (1.2, 0.9), (1.9, 0.05)]:
    model = ets(a, seasonal=[12], gamma=[gm])
    add("max |lambda|, per-lag check", per_lag_max(*model), model="ETS(A,N,A)", m=12, alpha=a, gamma=gm)
    add("max |lambda|, all modes of the expanded discount matrix",
        np.abs(np.linalg.eigvals(expand(model[0] - np.outer(model[1], model[2]), *model[1:])[0])).max(),
        model="ETS(A,N,A)", m=12, alpha=a, gamma=gm)
    add("max |lambda|, controllable and observable modes (PBH)", pbh_max(*model),
        model="ETS(A,N,A)", m=12, alpha=a, gamma=gm)
    add("max |lambda|, roots of theta after cyclotomic cancellation", cancelled_max(model),
        model="ETS(A,N,A)", m=12, alpha=a, gamma=gm)

# Multiple seasonality and the GUM starting point: the factor shared by theta and phi
for name, model in [("ETS(A,N,A,A), alpha=0.3, gamma=(0.1,0.05)", ets(0.3, [24, 168], [0.1, 0.05])),
                    ("GUM start, orders (2,2), lags (1,12)", gum_start([1, 1, 12, 12]))]:
    theta, phi = arima_form(*model)
    thetaStar, _, h = cancel(theta, phi, model[3])
    add("degree of theta", len(trim(theta)) - 1, model=name)
    add("degree of the shared cyclotomic factor", len(h) - 1, model=name)
    add("max |lambda|, roots of theta before cancellation", max_lambda_roots(theta), model=name)
    add("max |lambda|, roots of theta after cancellation", max_lambda_roots(thetaStar), model=name)
    add("max |lambda|, controllable and observable modes (PBH)", pbh_max(*model), model=name)

# Block-diagonal D: the expanded system against the per-lag check
model = (np.diag([0.9, 0.3]), np.array([0.2, 0.1]), np.array([1.0, 0.0]), [1, 12])
add("max |lambda|, per-lag check", per_lag_max(*model), model="block-diagonal D, lags (1,12)")
add("max |lambda|, all modes of the expanded discount matrix",
    np.abs(np.linalg.eigvals(expand(model[0] - np.outer(model[1], model[2]), *model[1:])[0])).max(),
    model="block-diagonal D, lags (1,12)")

# Matrix determinant lemma: det(I - F L) = det(I - D L) (1 - w' L (I - D L)^{-1} g) at random complex B
errors = []
for _ in range(100):
    F, g, w, lags = gum_random(list(rng.choice([1, 4, 12, 24], size=rng.integers(2, 6))), rng)
    k, B = len(lags), complex(*rng.normal(0, 0.7, 2))
    L, I = np.diag(B ** np.array(lags)), np.eye(k)
    lhs = np.linalg.det(I - F @ L)
    rhs = np.linalg.det(I - (F - np.outer(g, w)) @ L) * (1 - w @ L @ np.linalg.solve(I - (F - np.outer(g, w)) @ L, g))
    errors.append(abs(lhs - rhs) / max(1.0, abs(lhs)))
add("max relative error of the determinant identity", max(errors), models=100)

# Step-down decision and bisection against the roots on random GUMs
agree, bisect = 0, []
for _ in range(300):
    model = gum_random(list(rng.choice([1, 4, 12, 24, 52, 168], size=rng.integers(2, 6))), rng)
    theta = arima_form(*model)[0]
    rootMax = max_lambda_roots(theta)
    agree += schur_cohn(theta) == (rootMax < 1)
    if rootMax >= 1:
        bisect.append(abs(spectral_radius(theta) - rootMax))
add("step-down decisions equal to root decisions", agree, models=300)
add("max abs difference, bisection against roots", max(bisect), inadmissible_models=len(bisect))

# Timings at degree 193
model = gum_random([1, 24, 168], rng)
theta = arima_form(*model)[0]
for name, f in [("coefficients by DFT", lambda: arima_form(*model)), ("step-down test", lambda: schur_cohn(theta)),
                ("all roots, numpy companion eigenvalues", lambda: P.polyroots(theta))]:
    t = time.perf_counter()
    for _ in range(50):
        f()
    add("mean time, ms", (time.perf_counter() - t) / 50 * 1e3, task=name, degree=193)

write_results(results)
