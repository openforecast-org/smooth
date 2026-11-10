"""AC3: ETS+ARIMA models with rational parameters; exact gcd(theta, phi) oracle against the author's cyclotomic
cancellation, the PBH reference and the package's smooth_eigens(). A wider generator than the author's:
constant state, two seasonal lags, AR up to 3, I up to 2, MA up to 3 with cyclotomic and unit-circle factors."""
import argparse
import os
import sys
import time
from fractions import Fraction as Fr

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/home/user/smooth/claimset/adam-admissible-bounds/code")
from lib import *  # noqa
import admissibility as ad  # the author's prototype: arima_form, cancel, pbh_max, max_lambda_roots
from smooth.adam_general._eigenCalc import smooth_eigens

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--n", type=int, default=100)
args = ap.parse_args()
rng = np.random.default_rng(args.seed)
fr = lambda k, d: Fr(int(k), d)
pick = lambda xs: xs[int(rng.integers(len(xs)))]


def ets_rational():
    m = int(pick([4, 6, 12]))
    trend = pick(["N", "A", "Ad"])
    seas = []
    if rng.random() < 0.5:
        seas = [m] if rng.random() < 0.8 or m == 4 else [m // 2 if m % 2 == 0 else m, m]
        seas = sorted(set(seas))
    const = rng.random() < 0.2
    alpha = fr(0 if rng.random() < 0.05 else (rng.integers(1, 21) if rng.random() < 0.8 else rng.integers(21, 40)), 20)
    beta = None if trend == "N" else fr(0 if rng.random() < 0.1 else rng.integers(1, 7), 20)
    gam = [fr(0 if rng.random() < 0.15 else (rng.integers(1, 9) if rng.random() < 0.85 else rng.integers(9, 31)), 20) for _ in seas]
    phi = fr(pick([16, 18, 19]), 20) if trend == "Ad" else Fr(1)
    lags = [1] + ([1] if beta is not None else []) + seas + ([1] if const else [])
    k = len(lags)
    F = [[Fr(int(i == j)) for j in range(k)] for i in range(k)]
    g, w = [Fr(0)] * k, [Fr(1)] * k
    g[0] = alpha
    s = 1
    if beta is not None:
        F[0][1] = F[1][1] = phi
        w[1] = phi
        g[1] = beta
        s = 2
    for q, gg in enumerate(gam):
        g[s + q] = gg
    return (F, g, w, lags), m


MA1 = [[Fr(1, 2)], [Fr(-1, 2)], [Fr(4, 5)], [Fr(-4, 5)], [Fr(3, 2)], [Fr(-3, 2)], [Fr(1)], [Fr(-1)],
       [Fr(1), Fr(1)], [Fr(0), Fr(1)], [Fr(-1), Fr(1)]]  # 1+sB, 1+B+B^2 (Phi_3), 1+B^2 (Phi_4), 1-B+B^2 (Phi_6)


MA1_STABLE = [[Fr(1, 2)], [Fr(-1, 2)], [Fr(4, 5)], [Fr(-4, 5)], [Fr(1, 5)], [Fr(-3, 10)]]
CIRC = [[1], [-1], [1, 1], [0, 1], [-1, 1]]


def arima_rational(m):
    ar = []
    ar = [(1, [fr(rng.integers(-9, 10), 10)]) for _ in range(int(rng.integers(0, 4)))]
    if rng.random() < 0.4:
        ar.append((m, [fr(rng.integers(-9, 10), 10)]))
    i = [(1, int(rng.integers(0, 3)))]
    if rng.random() < 0.4:
        i.append((m, 1))
    ma = [(1, pick(MA1_STABLE) if rng.random() < 0.65 else pick(MA1)) for _ in range(int(rng.integers(0, 4)))]
    if rng.random() < 0.4:
        ma.append((m, [pick([Fr(1, 2), Fr(-1, 2), Fr(1, 5), Fr(1), Fr(-1), Fr(2)])]))
    return ar, [(l, d) for l, d in i if d], ma


def build(ar, i, ma):
    ari = prod([ar_factor(c, l, -1, True) for l, c in ar] +
               [ar_factor([1], l, -1, True) for l, d in i for _ in range(d)], True)
    maPoly = prod([ar_factor(c, l, 1, True) for l, c in ma], True)
    return arima_matrices(ari, maPoly, True), ari, maPoly


def reduce_exact(theta, phi):
    gcd = pgcd(theta, phi)
    return pdivmod(theta, gcd)[0] if len(gcd) > 1 else theta


def has_cyclotomic(p):
    for n in range(1, 61):
        if len(p) > len(cyclo(n)) - 1 and not any(x != 0 for x in pdivmod(p, cyclo(n))[1]):
            return n
    return 0


rows = []
t0 = time.time()
while len(rows) < args.n:
    etsP, m = ets_rational()
    ar, i, ma = arima_rational(m)
    if not (ar or i or ma):
        continue
    (Fa, ga, wa, la), ari, maPoly = build(ar, i, ma)
    if len(la) > 9 or len(la) + len(etsP[3]) > 12:
        continue
    F, g, w, lags = blockdiag([etsP, (Fa, ga, wa, la)])
    D = Dmat(F, g, w)
    theta, phi = detpoly(D, lags, True), detpoly(F, lags, True)
    red = reduce_exact(theta, phi)
    cyc = has_cyclotomic(red)
    mr = minroot_f = float(np.abs(np.roots(np.array([float(x) for x in red])[::-1])).min()) if len(red) > 1 else np.inf
    oracle_ok = (cyc == 0) and mr > 1 + 1e-8
    border = abs(mr - 1) < 1e-5 and cyc == 0
    Ff, gf, wf = np.array(F, float), np.array(g, float), np.array(w, float)
    # the author's cyclotomic-only cancellation and the PBH reference
    th_a, ph_a = ad.arima_form(Ff, gf, wf, lags)
    th_c = ad.cancel(th_a, ph_a, lags)[0]
    author_mod = ad.max_lambda_roots(th_c)
    pbh = ad.pbh_max(Ff, gf, wf, lags)
    pkg = float(np.max(smooth_eigens(persistence=np.asfortranarray(gf.reshape(-1, 1)), transition=np.asfortranarray(Ff),
                                       measurement=np.asfortranarray(wf.reshape(1, -1)),
                                       lags_model_all=np.asarray(lags, dtype=np.int32), xreg_model=False,
                                       obs_in_sample=1, has_delta=False, xreg_number=0, constant_required=False)))
    rows.append({"oracle": oracle_ok, "author": author_mod < 1 - 1e-6, "pbh": pbh < 1 - 1e-6, "pkg": not pkg > 1 + 1e-50,
                 "border": border, "cyc": cyc, "unitrootcancelled": any(f in CIRC for l, f in ma),
                 "oracleMod": 1 / mr if np.isfinite(mr) else 0.0, "authorMod": author_mod, "pbhMod": pbh, "pkgMod": pkg,
                 "k": len(lags), "deg": sum(lags),
                 "spec": {"ets_lags": etsP[3], "g": [str(x) for x in etsP[1]], "F_ets": [[str(x) for x in r] for r in etsP[0]],
                          "ar": [(l, [str(x) for x in c]) for l, c in ar], "i": i,
                          "ma": [(l, [str(x) for x in c]) for l, c in ma], "red": [str(x) for x in red]}})
    if rows[-1]["pbh"] != rows[-1]["oracle"] or rows[-1]["pkg"] != rows[-1]["oracle"] or rows[-1]["author"] != rows[-1]["oracle"]:
        print("DISAGREE", {k: v for k, v in rows[-1].items()})
rows_n = len(rows)
res = []
add = lambda q, v, **c: res.append({"quantity": q, "value": float(v), "conditions": {"models": rows_n, "seed": args.seed, **c}})
nb = sum(r["border"] for r in rows)
use = [r for r in rows if not r["border"]]
add("models whose root modulus after exact gcd cancellation is within 1e-5 of the unit circle, not on a cyclotomic factor (excluded)", nb)
for name in ("author", "pbh", "pkg"):
    add("models whose admissibility decision differs from the exact gcd oracle", sum(r[name] != r["oracle"] for r in use), check=name)
    add("of those, accepted by the check and rejected by the oracle", sum(r[name] and not r["oracle"] for r in use), check=name)
onc = [r for r in use if r["cyc"] > 0]
add("models whose reduced theta still has a cyclotomic factor, i.e. a root exactly on the unit circle (inadmissible)", len(onc))
for name in ("author", "pbh", "pkg"):
    add("of those, accepted by the check", sum(r[name] for r in onc), check=name)
add("models admissible by the exact oracle", sum(r["oracle"] for r in use))
add("models with an MA factor on the unit circle (1+-B, Phi_3, Phi_4, Phi_6 factors at lag 1 or seasonal)", sum(r["unitrootcancelled"] for r in use))
add("of those, admissible by the exact oracle (the root cancelled)", sum(r["oracle"] for r in use if r["unitrootcancelled"]))
both = [r for r in use if not r["oracle"]]
add("max abs difference of the largest modulus, exact oracle against the author's cancelled polynomial, inadmissible models",
    max([abs(r["oracleMod"] - r["authorMod"]) for r in both], default=0.0), inadmissible=len(both))
add("max abs difference of the largest modulus, exact oracle against the package smooth_eigens, inadmissible models",
    max([abs(r["oracleMod"] - r["pkgMod"]) for r in both], default=0.0), inadmissible=len(both))
add("largest number of components in a model", max(r["k"] for r in rows))
add("seconds", time.time() - t0)
results_out(res)
