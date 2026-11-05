"""What estimation and selection alone predict at the benchmark's lengths: the simulated
coverage of each case by T, interpolated in log T, averaged over the series."""
import json
import os
import sys

import numpy as np
import pandas as pd

from q1_benchmark_reforecast import load

cells = pd.concat([pd.read_csv(p) for p in sys.argv[1:]])
n = np.array([len(s.x) for s in load()])
results = []
for case in ("T_backcasting", "E_backcasting", "S_backcasting"):
    sim = cells[cells["case"] == case].groupby("T")["cov80"].mean()
    v = np.interp(np.log(n), np.log(sim.index.values), sim.values).mean()
    print(case, sim.round(3).to_dict(), "expected cov80", round(v, 4))
    results.append(dict(quantity="cov80 expected at the benchmark lengths", value=float(v), conditions=dict(case=case)))
print("shares of n", {t: round(float(np.mean(n <= t)), 3) for t in (15, 25, 50, 100)})
if os.environ.get("CLAIMSET_RESULTS"):
    json.dump(results, open(os.environ["CLAIMSET_RESULTS"], "w"))
