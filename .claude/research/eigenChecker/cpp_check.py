"""The C++ core (smoothEigensPolyR through R) against the numpy prototype on random models."""

import argparse
import json
import os
import subprocess
import tempfile

import numpy as np

from admissibility import (arima_form, cancel, ets, gum_random, gum_start, max_lambda_roots, trim,
                           write_results)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--n", type=int, default=100)
args = parser.parse_args()
rng = np.random.default_rng(args.seed)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def ets_random(seasonalChoices):
    seasonal = sorted(rng.choice(seasonalChoices, size=rng.integers(0, 3), replace=False).tolist())
    trend = rng.choice(["N", "A", "Ad"])
    beta = None if trend == "N" else (0.0 if rng.random() < 0.1 else rng.uniform(0, 0.5))
    gamma = [0.0 if rng.random() < 0.15 else rng.uniform(0.01, 1.5) for _ in seasonal]
    return ets(rng.uniform(0.01, 1.9), seasonal, gamma, beta,
               phi=rng.uniform(0.8, 0.99) if trend == "Ad" else 1.0, constant=rng.random() < 0.3)


families = {
    "ETS": lambda: ets_random((4, 7, 12, 24, 52)),
    "ETS high degree": lambda: ets_random((24, 168, 365)),
    "GUM identity": lambda: gum_start(sorted(rng.choice([1, 4, 12], size=rng.integers(2, 5)).tolist()),
                                      g=rng.uniform(-0.3, 0.6)),
    "GUM random": lambda: gum_random(sorted(rng.choice([1, 4, 12, 24], size=rng.integers(2, 5)).tolist()), rng,
                                     sd=0.35),
}
models = [(name, make()) for name, make in families.items() for _ in range(args.n)]

# The prototype
expected = []
for name, (F, g, w, lags) in models:
    theta, phi, _ = cancel(*arima_form(F, g, w, lags), lags)
    expected.append({"family": name, "modulus": max_lambda_roots(theta), "theta": trim(theta).tolist(),
                     "phi": trim(phi).tolist()})

# The C++ core through R
with tempfile.TemporaryDirectory() as tmp:
    inputPath, outputPath = os.path.join(tmp, "models.json"), os.path.join(tmp, "out.json")
    json.dump([{"F": F.tolist(), "g": g.tolist(), "w": w.tolist(), "lags": lags} for _, (F, g, w, lags) in models],
              open(inputPath, "w"))
    script = f"""
    suppressMessages(pkgload::load_all("{ROOT}", quiet=TRUE))
    models <- jsonlite::fromJSON("{inputPath}", simplifyVector=FALSE)
    out <- lapply(models, function(m) {{
        k <- length(m$lags)
        F <- matrix(unlist(m$F), k, k, byrow=TRUE)
        r <- smoothEigensPolyR(matrix(unlist(m$g)), F, matrix(unlist(m$w), 1), as.integer(unlist(m$lags)),
                               FALSE, 1L, FALSE)
        list(modulus=as.numeric(r), theta=attr(r, "maPolynomial"), phi=attr(r, "ariPolynomial"))
    }})
    jsonlite::write_json(out, "{outputPath}", digits=NA, auto_unbox=TRUE)
    """
    subprocess.run(["Rscript", "-e", script], check=True)
    actual = json.load(open(outputPath))


def degree(c):
    return len(trim(np.atleast_1d(c))) - 1


results = []
for name in families:
    pairs = [(e, a) for e, a in zip(expected, actual) if e["family"] == name]
    mismatches = sum((e["modulus"] < 1) != (a["modulus"] < 1) for e, a in pairs)
    modulusDiff = max(abs(e["modulus"] - a["modulus"]) for e, a in pairs)
    degreeDiffs = sum(degree(e["theta"]) != degree(a["theta"]) or degree(e["phi"]) != degree(a["phi"])
                      for e, a in pairs)
    conditions = {"family": name, "models": len(pairs)}
    results += [{"quantity": "admissibility decisions of the C++ core that differ from the prototype",
                 "value": mismatches, "conditions": conditions},
                {"quantity": "max abs difference of the largest root modulus, C++ core against prototype",
                 "value": modulusDiff, "conditions": conditions},
                {"quantity": "models whose cancelled polynomials differ in degree, C++ core against prototype",
                 "value": degreeDiffs, "conditions": conditions}]
write_results(results)
