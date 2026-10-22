"""The per-lag condition of the ADAM monograph (eqs 5.20-5.22) against the full discount recursion."""

import argparse

import numpy as np

from admissibility import arima_form, cancel, ets, max_lambda_roots, per_lag_max, write_results

parser = argparse.ArgumentParser()
parser.add_argument("--periods", type=int, default=2400)
args = parser.parse_args()
results = []

# ETS(A,N,A), m = 12: the book's D_1 = [[1 - alpha, 0], [0, 0]] and D_m = [[0, 0], [0, 1 - gamma]] against the
# recursion v_t = D v_{t-l} with the full D = F - g w' (y = 0), which carries the initial states
m = 12
for alpha, gamma in [(0.1, 0.1), (1.2, 0.9)]:
    model = ets(alpha, [m], [gamma])
    D = model[0] - np.outer(model[1], model[2])
    level, seasonal, effect = 1.0, list(np.ones(m)), []
    for t in range(args.periods):
        lagged = np.array([level, seasonal[-m]])
        effect.append(abs(model[2] @ lagged))
        level, newSeasonal = D @ lagged
        seasonal.append(newSeasonal)
    half = args.periods // 2
    conditions = {"model": "ETS(A,N,A)", "m": m, "alpha": alpha, "gamma": gamma}
    results += [{"quantity": "largest nonzero eigenvalue modulus of the book's per-lag discount matrices",
                 "value": per_lag_max(*model), "conditions": conditions},
                {"quantity": "effect of the initial states on the fitted value at the last period",
                 "value": effect[-1], "conditions": {**conditions, "period": args.periods}},
                {"quantity": "growth of the effect of the initial states per period over the second half",
                 "value": (effect[-1] / effect[half - 1]) ** (1 / (args.periods - half)), "conditions": conditions},
                {"quantity": "largest root modulus of the cancelled MA polynomial",
                 "value": max_lambda_roots(cancel(*arima_form(*model), model[3])[0]), "conditions": conditions}]
write_results(results)
