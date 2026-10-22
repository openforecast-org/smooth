"""Admissibility of ADAM models that combine ETS with ARIMA: the check of the parts apart (the ETS
part by the polynomial check, the ARIMA part by its own polynomials, as adam() does) and the check of
the whole model (the ARIMA form of the whole lagged SSOE with cyclotomic cancellation), against the
PBH reference, with the AR polynomial stationary in all three."""

import argparse
import json
import os
import subprocess
import tempfile

import numpy as np

from admissibility import arima_form, cancel, ets, max_lambda_roots, pbh_max, write_results

P = np.polynomial.polynomial
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--n", type=int, default=300)
parser.add_argument("--grid", action="store_true", help="the ETS(A,N,N)+MA(1) grid")
parser.add_argument("--adam", type=int, default=0, help="models compared with the matrices of adam()")
args = parser.parse_args()
rng = np.random.default_rng(args.seed)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TOL = 1e-6


# Polynomial 1 - sum a_j B^{jm} (sign=-1, AR) or 1 + sum a_j B^{jm} (sign=1, MA) of one factor
def factor(coefs, lag, sign):
    p = np.zeros(len(coefs) * lag + 1)
    p[0] = 1
    p[lag::lag] = sign * np.asarray(coefs, float)
    return p


def product(polys):
    out = np.array([1.0])
    for p in polys:
        out = P.polymul(out, p)
    return out


# ADAM's ARIMA part: one state per lag of a coefficient of the ARI or MA polynomial that the orders
# allow, transition row -ari_l, persistence ma_l - ari_l, measurement 1 (R/adamGeneral.R, creator)
def arima_part(ar, i, ma):
    ari = product([factor(c, m, -1) for m, c in ar] + [factor([1], m, -1) for m, d in i for _ in range(d)])
    maPoly = product([factor(c, m, 1) for m, c in ma])
    support = product([factor(np.ones(len(c)), m, 1) for m, c in ar] +
                      [factor([1], m, 1) for m, d in i for _ in range(d)])
    supportMA = product([factor(np.ones(len(c)), m, 1) for m, c in ma])
    lags = sorted({j for j in range(1, len(support)) if support[j] != 0} |
                  {j for j in range(1, len(supportMA)) if supportMA[j] != 0})
    pad = lambda p, j: p[j] if j < len(p) else 0.0
    k = len(lags)
    F = np.tile(np.array([-pad(ari, j) for j in lags])[:, None], (1, k))
    g = np.array([pad(maPoly, j) - pad(ari, j) for j in lags])
    return F, g, np.ones(k), lags, ari, maPoly


def whole(etsPart, arimaPart):
    (Fe, ge, we, le), (Fa, ga, wa, la) = etsPart, arimaPart[:4]
    F = np.zeros((len(le) + len(la),) * 2)
    F[:len(le), :len(le)], F[len(le):, len(le):] = Fe, Fa
    return F, np.r_[ge, ga], np.r_[we, wa], list(le) + list(la)


# Stationary AR / random MA coefficients of one factor from coefficients of the step-down recursion
def from_reflection(k):
    a = np.array([])
    for kj in k:
        a = np.r_[a - kj * a[::-1], kj]
    return a


def stationary(ar):
    return all(max_lambda_roots(factor(c, m, -1)) < 1 for m, c in ar if len(c))


def decisions(etsPart, arimaPart, ar):
    F, g, w, lags = whole(etsPart, arimaPart)
    arOK = stationary(ar)
    etsModulus = max_lambda_roots(cancel(*arima_form(*etsPart), etsPart[3])[0])
    maModulus = max_lambda_roots(arimaPart[5])
    jointModulus = max_lambda_roots(cancel(*arima_form(F, g, w, lags), lags)[0])
    reference = pbh_max(F, g, w, lags)
    return {"split": bool(arOK and etsModulus < 1 - TOL and maModulus < 1 - TOL),
            "joint": bool(arOK and jointModulus < 1 - TOL),
            "reference": bool(arOK and reference < 1 - TOL),
            "jointModulus": jointModulus, "referenceModulus": reference}


def ets_random(m):
    trend = rng.choice(["N", "A", "Ad"])
    seasonal = [m] if rng.random() < 0.5 else []
    beta = None if trend == "N" else (0.0 if rng.random() < 0.1 else rng.uniform(0, 0.5))
    gamma = [0.0 if rng.random() < 0.15 else rng.uniform(0.01, 1.5) for _ in seasonal]
    alpha = 0.0 if rng.random() < 0.1 else rng.uniform(0.01, 1.9)
    phi = rng.uniform(0.8, 0.99) if trend == "Ad" else 1.0
    return ets(alpha, seasonal, gamma, beta, phi), {"trend": str(trend), "seasonal": seasonal, "alpha": alpha,
                                                     "beta": beta, "gamma": gamma, "phi": phi}


def arima_random(m, seasonal):
    lags = [1, m] if seasonal else [1]
    ar = [(lag, from_reflection(rng.uniform(-0.95, 0.95, rng.integers(0, 3 - (lag > 1))))) for lag in lags]
    ma = [(lag, from_reflection(rng.uniform(-1.5, 1.5, rng.integers(0, 3 - (lag > 1))))) for lag in lags]
    i = [(lag, int(rng.integers(0, 2))) for lag in lags]
    # Over-differenced: the MA factor of a differenced lag has a root at one on the unit circle
    if rng.random() < 0.15:
        lag = int(rng.choice(lags))
        i = [(l, 1 if l == lag else d) for l, d in i]
        ma = [(l, np.array([-1.0]) if l == lag else c) for l, c in ma]
    return [(l, c) for l, c in ar if len(c)], [(l, d) for l, d in i if d], [(l, c) for l, c in ma if len(c)]


results, specs = [], []
families = {"ETS+ARIMA": False, "ETS+SARIMA": True}
for name, seasonal in families.items():
    rows = []
    for _ in range(args.n):
        m = int(rng.choice([4, 12]))
        etsPart, etsSpec = ets_random(m)
        while True:
            ar, i, ma = arima_random(m, seasonal)
            if ar or i or ma:
                break
        arimaPart = arima_part(ar, i, ma)
        rows.append(decisions(etsPart, arimaPart, ar))
        specs.append({"ets": etsSpec, "m": m, "ar": [(l, c.tolist()) for l, c in ar], "i": i,
                      "ma": [(l, c.tolist()) for l, c in ma], "arima": arimaPart, "etsPart": etsPart})
    for method in ("split", "joint"):
        cond = {"family": name, "models": len(rows), "check": method}
        results += [
            {"quantity": "models whose admissibility decision differs from the PBH reference",
             "value": sum(r[method] != r["reference"] for r in rows), "conditions": cond},
            {"quantity": "models accepted by the check and rejected by the PBH reference",
             "value": sum(r[method] and not r["reference"] for r in rows), "conditions": cond},
            {"quantity": "models rejected by the check and accepted by the PBH reference",
             "value": sum(r["reference"] and not r[method] for r in rows), "conditions": cond}]
    both = [r for r in rows if r["jointModulus"] >= 1 - TOL and r["referenceModulus"] >= 1 - TOL]
    results += [{"quantity": "max abs difference of the largest modulus, check of the whole model against the "
                             "PBH reference, on models both find inadmissible",
                 "value": max([abs(r["jointModulus"] - r["referenceModulus"]) for r in both], default=0.0),
                 "conditions": {"family": name, "inadmissible": len(both)}},
                {"quantity": "models found admissible by the PBH reference",
                 "value": sum(r["reference"] for r in rows), "conditions": {"family": name, "models": len(rows)}}]

# The ARIMA part alone has the ARIMA form of its ARI and MA polynomials
diff = 0.0
for s in specs:
    F, g, w, lags, ari, maPoly = s["arima"]
    theta, phi = arima_form(F, g, w, lags)
    n = max(len(theta), len(maPoly))
    diff = max(diff, np.abs(np.pad(theta, (0, n - len(theta))) - np.pad(maPoly, (0, n - len(maPoly)))).max(),
               np.abs(np.pad(phi, (0, max(len(phi), len(ari)) - len(phi))) -
                      np.pad(ari, (0, max(len(phi), len(ari)) - len(ari)))).max())
results.append({"quantity": "max abs difference between the ARIMA form of the ARIMA part and its ARI and MA "
                            "polynomials", "value": float(diff), "conditions": {"models": len(specs)}})

# The ETS(A,N,N)+MA(1) grid
if args.grid:
    rows = []
    for alpha in np.round(np.arange(-1, 3.0001, 0.1), 10):
        for theta in np.round(np.arange(-0.95, 0.95001, 0.05), 10):
            rows.append({"alpha": alpha, "theta": theta,
                         **decisions(ets(alpha), arima_part([], [], [(1, [theta])]), [])})
    cond = {"family": "ETS(A,N,N)+MA(1) grid", "models": len(rows), "alpha": "-1 to 3 by 0.1",
            "theta": "-0.95 to 0.95 by 0.05"}
    for method in ("split", "joint"):
        results += [{"quantity": "models accepted by the check and rejected by the PBH reference",
                     "value": sum(r[method] and not r["reference"] for r in rows), "conditions": {**cond, "check": method}},
                    {"quantity": "models rejected by the check and accepted by the PBH reference",
                     "value": sum(r["reference"] and not r[method] for r in rows), "conditions": {**cond, "check": method}}]
    example = next(r for r in rows if abs(r["alpha"] - 1.9) < 1e-9 and abs(r["theta"] - 0.1) < 1e-9)
    results.append({"quantity": "largest modulus of the whole model, PBH reference", "value": example["referenceModulus"],
                    "conditions": {"model": "ETS(A,N,N)+MA(1)", "alpha": 1.9, "theta": 0.1,
                                   "split check": "admissible" if example["split"] else "inadmissible"}})

# The matrices built here against those of adam() with the same fixed parameters
if args.adam:
    chosen = [specs[j] for j in rng.choice(len(specs), args.adam, replace=False)]
    with tempfile.TemporaryDirectory() as tmp:
        inputPath, outputPath = os.path.join(tmp, "in.json"), os.path.join(tmp, "out.json")
        json.dump([{k: s[k] for k in ("ets", "m", "ar", "i", "ma")} for s in chosen], open(inputPath, "w"))
        script = f"""
        suppressMessages(pkgload::load_all("{ROOT}", quiet=TRUE))
        specs <- jsonlite::fromJSON("{inputPath}", simplifyVector=FALSE)
        orderOf <- function(parts, lag, m) {{
            for(p in parts) if(p[[1]]==lag) return(if(is.list(p[[2]])) length(p[[2]]) else p[[2]])
            return(0)
        }}
        coefsOf <- function(parts) unlist(lapply(parts, function(p) unlist(p[[2]])))
        out <- lapply(specs, function(s) {{
            e <- s$ets
            model <- paste0("A", switch(e$trend, N="N", A="A", Ad="Ad"), if(length(e$seasonal)) "A" else "N")
            persistence <- c(alpha=e$alpha)
            if(e$trend != "N") persistence <- c(persistence, beta=e$beta)
            if(length(e$seasonal)) persistence <- c(persistence, gamma=unlist(e$gamma))
            lags <- c(1, s$m)
            orders <- list(ar=sapply(lags, function(l) orderOf(s$ar, l)), i=sapply(lags, function(l) orderOf(s$i, l)),
                           ma=sapply(lags, function(l) orderOf(s$ma, l)))
            arma <- list(ar=coefsOf(s$ar), ma=coefsOf(s$ma))
            arma <- arma[sapply(arma, length) > 0]
            if(length(arma) == 0) arma <- NULL
            y <- ts(100 + cumsum(rnorm(120)), frequency=s$m)
            m <- suppressWarnings(adam(y, model, orders=orders, lags=lags, persistence=persistence,
                                       phi=if(e$trend=="Ad") e$phi else NULL, arma=arma, h=0))
            list(F=m$transition, g=as.numeric(m$persistence), w=as.numeric(tail(m$measurement, 1)),
                 lags=as.numeric(m$lagsAll))
        }})
        jsonlite::write_json(out, "{outputPath}", digits=NA, auto_unbox=FALSE)
        """
        subprocess.run(["Rscript", "-e", script], check=True)
        adams = json.load(open(outputPath))
    worst, lagsDiffer = 0.0, 0
    for s, a in zip(chosen, adams):
        F, g, w, lags = whole(s["etsPart"], s["arima"])
        Fa = np.array(a["F"], float).reshape(len(lags), -1) if len(a["lags"]) == len(lags) else None
        if Fa is None or list(map(int, a["lags"])) != lags:
            lagsDiffer += 1
            continue
        gap = max(np.abs(Fa - F).max(), np.abs(np.array(a["g"]) - g).max(), np.abs(np.array(a["w"]) - w).max())
        if gap > 1e-10:
            print({k: s[k] for k in ("ets", "m", "ar", "i", "ma")}, "lags", lags)
            print("  F here", np.round(F, 4).tolist(), "\n  F adam", np.round(Fa, 4).tolist())
            print("  g here", np.round(g, 4).tolist(), "\n  g adam", np.round(a["g"], 4).tolist())
            print("  w here", np.round(w, 4).tolist(), "\n  w adam", np.round(a["w"], 4).tolist())
        worst = max(worst, gap)
    results += [{"quantity": "models whose component lags differ from those of adam()", "value": lagsDiffer,
                 "conditions": {"models": len(chosen)}},
                {"quantity": "max abs difference between the matrices built here and those of adam()",
                 "value": float(worst), "conditions": {"models": len(chosen) - lagsDiffer}}]

write_results(results)
