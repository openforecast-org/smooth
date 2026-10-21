"""Timings of the step-down test (full recursion and early exit), the roots and the DFT coefficients."""

import argparse
import time

import numpy as np

from admissibility import (P, arima_form, cancel, ets, gum_random, max_lambda_roots, schur_cohn, trim,
                           write_results)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--repeats", type=int, default=50)
args = parser.parse_args()
rng = np.random.default_rng(args.seed)


# Number of recursion steps the step-down test runs before it stops
def steps(c):
    c = trim(c)
    c = c / c[0]
    for n, j in enumerate(range(len(c) - 1, 0, -1), start=1):
        if abs(c[j]) >= 1:
            return n
        c = (c[:j] - c[j] * c[j:0:-1]) / (1 - c[j] ** 2)
    return len(c) - 1


def mean_ms(f):
    t = time.perf_counter()
    for _ in range(args.repeats):
        f()
    return (time.perf_counter() - t) / args.repeats * 1e3


# The first admissible and the first inadmissible random GUM with lags (1, 24, 168), and ETS(A,N,A,A) 24/168
cases = {}
while len(cases) < 2:
    model = gum_random([1, 24, 168], rng, sd=0.3)
    theta = arima_form(*model)[0]
    cases.setdefault("random GUM, lags (1, 24, 168), " + ("admissible" if schur_cohn(theta) else "inadmissible"),
                     (model, theta))
model = ets(0.3, [24, 168], [0.1, 0.05])
cases["ETS(A,N,A,A), lags 24 and 168, cancelled theta"] = (model, cancel(*arima_form(*model), model[3])[0])

results = []
for name, (model, theta) in cases.items():
    c = {"model": name, "degree": len(trim(theta)) - 1, "repeats": args.repeats}
    results += [{"quantity": "largest root modulus 1/min|B| of the polynomial", "value": max_lambda_roots(theta),
                 "conditions": c},
                {"quantity": "recursion steps run by the step-down test", "value": steps(theta), "conditions": c},
                {"quantity": "mean time of the step-down test, ms", "value": mean_ms(lambda: schur_cohn(theta)),
                 "conditions": c},
                {"quantity": "mean time of all roots by numpy companion eigenvalues, ms",
                 "value": mean_ms(lambda: P.polyroots(trim(theta))), "conditions": c},
                {"quantity": "mean time of the coefficients of theta and phi by inverse DFT, ms",
                 "value": mean_ms(lambda: arima_form(*model)), "conditions": c}]
write_results(results)
