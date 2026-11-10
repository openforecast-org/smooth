"""Independent exact oracle of admissibility for pure additive lagged SSOE models.

Written from the definitions of the claimset adam-admissible-bounds only (D1, D3,
D6, D10, D11, D29, D30, C17, C19, C20), without the study's prototype.

For a model (F, g, w, lags), every float entry is converted exactly to a rational,
D = F - g w', and

    theta(B) = det(I - D L(B)),  phi(B) = det(I - F L(B)),  L(B) = diag(B^l_j),

are computed exactly over Q[B] by fraction-free (Bareiss) elimination.  Then

* G   = gcd(theta, phi), exact; theta_G = theta / G (C17: its roots are the
        reciprocals of the reachable and observable modes; the counterpart of
        the PBH reference D5);
* C   = the product of the cyclotomic polynomials Phi_n, n dividing a lag, each
        removed from theta and phi for as long as it divides both (D6, in exact
        arithmetic); theta_C = theta / C (the counterpart of the prototype's
        cancelled MA polynomial);
* R   = G / C, the shared factor that is not such a cyclotomic product (C19 says
        it has no root in |B| <= 1).

Roots are enclosed by Arb balls (python-flint's complex_roots, which isolates
every root); a mode is lambda = 1/B, and a polynomial is admissible when every
mode has |lambda| < 1, certified by the enclosures.  When an enclosure cannot
separate |lambda| from 1, the precision is raised; a root that stays ambiguous
is proved to lie on the unit circle when the ball holds a root of unity
exp(2 pi i k/n) and Phi_n divides the polynomial exactly; otherwise the model is
undecided.

Usage: oracle.py MODELS.json [MODELS.json ...] [--jobs N]
Writes results to $CLAIMSET_RESULTS and per-model values to $CLAIMSET_RUN_DIR.
"""

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction

import flint
from flint import acb, arb, ctx, fmpq, fmpq_poly, fmpz_poly

PRECS = (128, 512, 2048)  # bits of the root enclosures, raised while ambiguous
EDGE = 1e-8               # band around 1 left out of the comparisons (as D26)
MAX_ORDER = 10 ** 7       # largest order n of a root of unity tried in a proof


def exact(x):
    """The float x as an exact rational."""
    n, d = Fraction(x).as_integer_ratio()
    return fmpq(n, d)


def monomial(c, power):
    return fmpq_poly([0] * power + [c])


def lag_det(M, lags):
    """det(I - M L(B)) over Q[B] by Bareiss elimination (exact divisions).

    The leading principal minors of I - M L(B) are 1 at B = 0, so no pivot
    is the zero polynomial and no pivoting is needed."""
    k = len(lags)
    A = [[(fmpq_poly([1]) if i == j else fmpq_poly([0])) - monomial(M[i][j], lags[j])
          for j in range(k)] for i in range(k)]
    prev = fmpq_poly([1])
    for p in range(k - 1):
        for i in range(p + 1, k):
            for j in range(p + 1, k):
                num = A[i][j] * A[p][p] - A[i][p] * A[p][j]
                quo, rem = divmod(num, prev)
                assert rem == 0, "inexact Bareiss division"
                A[i][j] = quo
        prev = A[p][p]
    return A[k - 1][k - 1]


def exact_div(a, b):
    quo, rem = divmod(a, b)
    assert rem == 0, "inexact division"
    return quo


def divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def cyclotomic(n):
    return fmpq_poly(fmpz_poly.cyclotomic(n).coeffs())


def cyclotomic_cancel(theta, phi, lags):
    """Divide theta and phi by every Phi_n, n dividing a lag, while it divides both."""
    C = fmpq_poly([1])
    for n in sorted({d for l in lags for d in divisors(l)}):
        cp = cyclotomic(n)
        while True:
            qt, rt = divmod(theta, cp)
            qp, rp = divmod(phi, cp)
            if rt != 0 or rp != 0:
                break
            theta, phi, C = qt, qp, C * cp
    return theta, phi, C


def degree(p):
    return max(p.degree(), 0)


def on_circle(poly, root, balls):
    """True if the isolated root ball holds exp(2 pi i k/n) and Phi_n | poly.

    The balls of complex_roots are disjoint and each holds exactly one root;
    when Phi_n divides poly, zeta = exp(2 pi i k/n) is a root, so a ball of
    zeta that meets this ball and no other proves that this root is zeta."""
    t = float((root.arg() / (2 * arb.pi())).mid())
    fr = Fraction(t).limit_denominator(MAX_ORDER)
    n = fr.denominator
    if divmod(poly, cyclotomic(n))[1] != 0:
        return False
    zeta = (2 * acb.pi() * acb(0, 1) * acb(fr.numerator) / n).exp()
    return root.overlaps(zeta) and not any(b.overlaps(zeta) for b in balls if b is not root)


def modes(p):
    """Certified largest modulus of the modes lambda = 1/B over the roots B of p.

    Returns {status: 'admissible' | 'inadmissible' | 'undecided', lo, hi,
    unit_roots}: the largest modulus lies in [lo, hi]; unit_roots counts the
    roots proved to lie on the unit circle."""
    if p.degree() <= 0:
        return {"status": "admissible", "lo": 0.0, "hi": 0.0, "unit_roots": 0, "prec": 0}
    coeffs = p.coeffs()
    assert coeffs[0] != 0, "p(0) = 0"
    rev = fmpq_poly(list(reversed(coeffs)))  # its roots are lambda = 1/B
    one = arb(1)
    for prec in PRECS:
        ctx.prec = prec
        balls = [r for r, _ in rev.complex_roots()]
        lows = [r.abs_lower() for r in balls]
        highs = [r.abs_upper() for r in balls]
        above = any(lo > one for lo in lows)
        ambiguous = [r for r, lo, hi in zip(balls, lows, highs) if not (lo > one or hi < one)]
        if above or not ambiguous:
            break
    lo = max(lows)
    hi = max(highs)
    out = {"lo": float(lo.mid()), "hi": float(hi.mid()), "prec": prec, "unit_roots": 0}
    # round outwards the float bounds of [lo, hi]
    out["lo"] = math.nextafter(out["lo"], -math.inf)
    out["hi"] = math.nextafter(out["hi"], math.inf)
    if above:
        out["status"] = "inadmissible"
    elif not ambiguous:
        out["status"] = "admissible"
    elif all(on_circle(rev, r, balls) for r in ambiguous):
        out["status"] = "inadmissible"  # |lambda| = 1 exactly: weights do not decay
        out["unit_roots"] = len(ambiguous)
        out["lo"] = out["hi"] = 1.0  # the other roots are certified inside
    else:
        out["status"] = "undecided"
    ctx.prec = 53
    return out


def oracle(model):
    """The exact polynomials and the certified decisions of one model."""
    t0 = time.time()
    k = len(model["lags"])
    lags = [int(l) for l in model["lags"]]
    F = [[exact(x) for x in row] for row in model["F"]]
    g = [exact(x) for x in model["g"]]
    w = [exact(x) for x in model["w"]]
    D = [[F[i][j] - g[i] * w[j] for j in range(k)] for i in range(k)]
    theta = lag_det(D, lags)
    phi = lag_det(F, lags)
    G = theta.gcd(phi)
    theta_G = exact_div(theta, G)
    theta_C, phi_C, C = cyclotomic_cancel(theta, phi, lags)
    R = exact_div(G, C)  # G / C; exact because C | theta and C | phi
    rG = modes(theta_G)
    rC = modes(theta_C)
    rR = modes(R)  # a mode of R with |lambda| >= 1 is a root in |B| <= 1
    if rR["status"] == "admissible":
        shared = "outside"          # no root of R in |B| <= 1: C19 holds here
    elif rR["status"] == "inadmissible":
        shared = "inside"           # C19 fails on this model
    else:
        shared = "undecided"
    return {
        "index": model["index"], "family": model["family"], "lags": lags,
        "degree_theta": degree(theta), "degree_phi": degree(phi),
        "degree_gcd": degree(G), "degree_cyclotomic": degree(C),
        "degree_theta_gcd": degree(theta_G), "degree_theta_cancelled": degree(theta_C),
        "degree_phi_cancelled": degree(phi_C), "degree_noncyclotomic_shared": degree(R),
        "gcd": rG, "cancelled": rC, "shared": shared, "shared_modes": rR,
        "seconds": time.time() - t0,
    }


def safe_oracle(model):
    try:
        return oracle(model)
    except Exception as e:  # recorded as a failure, never dropped
        return {"index": model["index"], "family": model["family"], "error": repr(e)}


def decided(r):
    return r["status"] in ("admissible", "inadmissible")


def near_edge(r):
    return r["hi"] >= 1 - EDGE and r["lo"] <= 1 + EDGE


def largest(r):
    return (r["lo"] + r["hi"]) / 2


def compare(models, outs, setname):
    """Results of one set: exclusions, decisions and moduli against the prototype."""
    res = []

    def put(variable, value, **cond):
        res.append({"variable": variable, "value": float(value),
                    "conditions": dict({"set": setname}, **cond)})

    failed = [o for o in outs if "error" in o]
    ok = [(m, o) for m, o in zip(models, outs) if "error" not in o]
    undecided = [(m, o) for m, o in ok if not (decided(o["gcd"]) and decided(o["cancelled"]))]
    dec = [(m, o) for m, o in ok if decided(o["gcd"]) and decided(o["cancelled"])]
    edge = [(m, o) for m, o in dec if near_edge(o["gcd"]) or near_edge(o["cancelled"])]
    kept = [(m, o) for m, o in dec if not (near_edge(o["gcd"]) or near_edge(o["cancelled"]))]
    put("models given to the oracle", len(models))
    put("models on which the oracle failed", len(failed))
    put("models excluded by undecided", len(undecided))
    put("models excluded by edge", len(edge))
    put("models compared", len(kept))
    put("max degree of theta, exact", max((o["degree_theta"] for _, o in ok), default=0))
    put("max degree of the cancelled MA polynomial, exact",
        max((o["degree_theta_cancelled"] for _, o in ok), default=0))
    put("max precision of the root enclosures, bits",
        max((max(o["gcd"]["prec"], o["cancelled"]["prec"]) for _, o in ok), default=0))
    put("roots proved on the unit circle", sum(o["gcd"]["unit_roots"] + o["cancelled"]["unit_roots"] for _, o in ok))

    proto = models[0]["prototype"] if models else {}
    if "pbh_max" in proto:
        mism, diffs, both = 0, [], 0
        for m, o in kept:
            p = m["prototype"]["pbh_max"]
            if (p < 1) != (o["gcd"]["status"] == "admissible"):
                mism += 1
            elif p >= 1:
                both += 1
                diffs.append(abs(p - largest(o["gcd"])))
        put("admissibility decisions of the PBH reference that differ from the exact oracle", mism)
        put("max abs difference of the largest mode modulus, PBH reference against the exact oracle, "
            "models both find inadmissible", max(diffs, default=0.0))
        put("models both find inadmissible, PBH reference and exact oracle", both)
    if "max_lambda_cancelled" in proto:
        mism, diffs, both, degs = 0, [], 0, 0
        for m, o in kept:
            p = m["prototype"]["max_lambda_cancelled"]
            if (p < 1) != (o["cancelled"]["status"] == "admissible"):
                mism += 1
            elif p >= 1:
                both += 1
                diffs.append(abs(p - largest(o["cancelled"])))
        for m, o in ok:
            if m["prototype"].get("degree_theta_cancelled") != o["degree_theta_cancelled"]:
                degs += 1
        put("admissibility decisions of the prototype's cancellation that differ from the exact oracle", mism)
        put("max abs difference of the largest mode modulus, prototype's roots after cancellation "
            "against the exact oracle, models both find inadmissible", max(diffs, default=0.0))
        put("models both find inadmissible, prototype's cancellation and exact oracle", both)
        put("models whose cancelled MA degree differs from the prototype's", degs)
    if "stepdown_admissible" in proto:
        mism, diffs = 0, []
        for m, o in kept:
            if bool(m["prototype"]["stepdown_admissible"]) != (o["cancelled"]["status"] == "admissible"):
                mism += 1
            b = m["prototype"].get("bisection_max_lambda")
            if b is not None:
                diffs.append(abs(b - largest(o["cancelled"])))
        put("step-down decisions of the prototype that differ from the exact oracle", mism)
        put("max abs difference of the largest mode modulus, prototype's bisection against the exact oracle",
            max(diffs, default=0.0))
        put("models with a bisection value", len(diffs))
    # C19: a shared factor that is not a cyclotomic product (n | lag) with a root in |B| <= 1
    put("models with a shared factor of theta and phi that is not a product of Phi_n, n dividing a lag, "
        "with a root in |B| <= 1", sum(o["shared"] == "inside" for _, o in ok))
    put("models whose shared non-cyclotomic factor could not be located", sum(o["shared"] == "undecided" for _, o in ok))
    put("models with a shared non-cyclotomic factor, all roots in |B| > 1",
        sum(o["degree_noncyclotomic_shared"] > 0 and o["shared"] == "outside" for _, o in ok))
    if "per_lag_max" in proto:
        for fam in sorted({m["family"] for m in models}):
            sub = [(m, o) for m, o in dec if m["family"] == fam]
            acc = [(m, o) for m, o in sub if m["prototype"]["per_lag_max"] < 1]
            bad = sum(o["gcd"]["lo"] > 1 + EDGE for m, o in acc)
            rej = sum(o["gcd"]["hi"] < 1 - EDGE for m, o in sub if m["prototype"]["per_lag_max"] >= 1)
            put("models accepted by the per-lag check that do not forget their initial states, exact oracle",
                bad, family=fam)
            put("models rejected by the per-lag check that forget their initial states, exact oracle",
                rej, family=fam)
            put("models accepted by the per-lag check", len(acc), family=fam)
            put("models of the family", sum(m["family"] == fam for m in models), family=fam)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--families", nargs="*", help="only these families (exploration)")
    args = ap.parse_args()
    print(f"python-flint {flint.__version__}; enclosures {PRECS} bits; edge {EDGE}", flush=True)
    t0 = time.time()
    results, details = [], []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for path in args.files:
            with open(path) as f:
                data = json.load(f)
            models = data["models"]
            if args.families:
                models = [m for m in models if m["family"] in args.families]
            t1 = time.time()
            outs = list(pool.map(safe_oracle, models, chunksize=1))
            res = compare(models, outs, data["set"])
            res.append({"variable": "time of the oracle on the set, s", "value": time.time() - t1,
                        "conditions": {"set": data["set"]}})
            results += res
            details.append({"file": path, "set": data["set"], "seed": data.get("seed"),
                            "models": [dict(o, prototype=m["prototype"]) for m, o in zip(models, outs)]})
            for r in res:
                print(r["variable"], r["conditions"], r["value"], flush=True)
    # totals over all sets, for the decision rules
    totals = {}
    for r in results:
        if set(r["conditions"]) == {"set"} and not r["variable"].startswith(("max ", "time ")):
            totals[r["variable"]] = totals.get(r["variable"], 0) + r["value"]
        elif set(r["conditions"]) == {"set"} and r["variable"].startswith("max "):
            totals[r["variable"]] = max(totals.get(r["variable"], 0), r["value"])
    for v, x in totals.items():
        results.append({"variable": v, "value": x, "conditions": {"set": "all"}})
        print(v, {"set": "all"}, x)
    results.append({"variable": "time of the oracle, s", "value": time.time() - t0, "conditions": {"set": "all"}})
    out = os.environ.get("CLAIMSET_RESULTS")
    if out:
        with open(out, "w") as f:
            json.dump(results, f, indent=1)
    run_dir = os.environ.get("CLAIMSET_RUN_DIR")
    if run_dir:
        with open(os.path.join(run_dir, "oracle_models.json"), "w") as f:
            json.dump({"flint": flint.__version__, "precs": PRECS, "edge": EDGE, "sets": details}, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
