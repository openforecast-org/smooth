"""Checks the derivations in derivations.md: identities symbolically (sympy), inequalities on a
grid evaluated with 50-digit arithmetic. The first argument picks the section:
old-drift (§2, H5), fixed-points (§3, H6), score (§4, H7) or adam-ets (§5, H13).
Results: the number of identities sympy could not reduce to zero, and the number of numerical
counterexamples (an identity off by more than 1e-30, or an inequality that fails).
"""

import json
import os
import sys

import mpmath as mp
import sympy as sp

mp.mp.dps = 50
p, x, a, mu = sp.symbols("p x alpha mu", positive=True)
GRID = [mp.mpf(k) / 2000 for k in range(1, 2000)]  # p in (0, 1), step 0.0005


def identity(lhs, rhs):
    # (sympy could not reduce it, numerical counterexamples on the grid)
    diff = sp.simplify(sp.together(lhs - rhs))
    f = sp.lambdify((p, x), lhs - rhs, "mpmath")
    bad = sum(abs(f(q, r)) > mp.mpf("1e-30") for q in GRID[::50] for r in GRID[::50])
    return int(diff != 0), int(bad)


def inequality(holds):
    # Counterexamples of a predicate on the p grid
    return 0, sum(not holds(q) for q in GRID)


def expect(e_one, e_zero, prob=p):
    # E over o ~ Bernoulli(p) of a quantity worth e_one when o = 1 and e_zero when o = 0
    return prob * e_one + (1 - prob) * e_zero


def old_ratio(o, phat):
    # 1 + e_a = (1 + v)/(1 - v), v = o - phat
    return (1 + o - phat) / (1 - o + phat)


def g(q):
    # E[log(1 + e_a)] of the additive old proxy at phat = p (derivations.md §2)
    return q * mp.log((2 - q) / q) + (1 - q) * mp.log((1 - q) / (1 + q))


def h_log_b(q, r):
    # Drift of the additive old proxy for mu_b at phat = r
    return q * mp.log(r / (2 - r)) + (1 - q) * mp.log((1 + r) / (1 - r))


def root(q):
    return mp.findroot(lambda r: h_log_b(q, r), (mp.mpf("1e-40"), 1 - mp.mpf("1e-40")),
                       solver="anderson")


def old_drift():
    ea = expect(old_ratio(1, p), old_ratio(0, p))
    eb = expect(1 / old_ratio(1, p), 1 / old_ratio(0, p))
    checks = [identity(ea, 1 + (1 - p) + (1 - p) ** 2 / (1 + p)),
              identity(eb, 1 + p + p ** 2 / (2 - p)),
              inequality(lambda q: (1 - q) + (1 - q) ** 2 / (1 + q) > 0),
              inequality(lambda q: q + q ** 2 / (2 - q) > 0),
              inequality(lambda q: abs(g(1 - q) + g(q)) < mp.mpf("1e-40")),
              inequality(lambda q: (g(q) > 0) if q < 0.5 else (g(q) < 0) if q > 0.5 else True)]
    return checks


def fixed_points():
    hb = expect(1 / old_ratio(1, x), 1 / old_ratio(0, x)) - 1
    ha = expect(old_ratio(1, x), old_ratio(0, x)) - 1
    xb, xa = 1 - sp.sqrt(1 - p), sp.sqrt(p)

    def stable(h, star):
        # h changes sign from + to - (a) or - to + (b) across star, and has no other root
        f = sp.lambdify((p, x), h, "mpmath")
        return lambda q: all((f(q, r) > 0) == (r < star(q)) or abs(r - star(q)) < 1e-9
                             for r in GRID[::20])

    def between(q):
        r = root(q)
        lo, hi = sorted((q, mp.mpf("0.5")))
        return lo - mp.mpf("1e-30") <= r <= hi + mp.mpf("1e-30")

    def unique_stable_log(q):
        r = root(q)
        return all((h_log_b(q, s) < 0) == (s < r) or abs(s - r) < 1e-9 for s in GRID[::20])

    checks = [identity(hb, -2 * (x ** 2 - 2 * x + p) / ((2 - x) * (1 - x))),
              identity(ha, 2 * (p - x ** 2) / (x * (1 + x))),
              (int(sp.simplify(hb.subs(x, xb)) != 0), 0),
              (int(sp.simplify(ha.subs(x, xa)) != 0), 0),
              inequality(lambda q: 1 - mp.sqrt(1 - q) < q),
              inequality(lambda q: mp.sqrt(q) > q),
              inequality(stable(-hb, lambda q: 1 - mp.sqrt(1 - q))),
              inequality(stable(ha, mp.sqrt)),
              inequality(between),
              inequality(unique_stable_log)]
    return checks


def score():
    loglik = lambda o, prob: o * sp.log(prob) + (1 - o) * sp.log(1 - prob)
    # (probability as a function of the state, state at which the probability is x, sign)
    models = ((mu / (1 + mu), x / (1 - x), 1), (1 / (1 + mu), (1 - x) / x, -1))
    checks = []
    for prob, at_x, sign in models:
        for o in (0, 1):
            s = sp.diff(loglik(o, prob), mu) * mu  # d loglik / d log(mu)
            checks.append(identity(s.subs(mu, at_x), sign * (o - x)))
    # Expected score at phat = x is +-(p - x); zero only at x = p
    checks.append(identity(expect(1 - x, 0 - x), p - x))
    checks.append(inequality(lambda q: all(1 + al * e > 0 for al in (mp.mpf(1), mp.mpf("0.5"))
                                           for e in (q - 1, q, 1 - q, -q))))
    return checks


def adam_ets():
    # §5: E[log(1 + e)] for the ADAM multiplicative filter at p_hat = x
    log_old = expect(sp.log(old_ratio(1, x)), sp.log(old_ratio(0, x)))
    log_add = expect(sp.log((1 + 1 - x) / (1 - 1 + x)), sp.log((1 + 0 - x) / (1 - 0 + x)))
    log_score_a = expect(sp.log(2 - x), sp.log(1 - x))   # 1 + e_a = 1 + o - x
    log_score_b = expect(sp.log(x), sp.log(1 + x))       # 1 + e_b = 1 + x - o
    log_exp_a = expect(sp.log(sp.exp(1 - x)), sp.log(sp.exp(-x)))

    def root_of(f, q):
        return mp.findroot(lambda r: f(q, r), (mp.mpf("1e-40"), 1 - mp.mpf("1e-40")),
                           solver="anderson")

    fa = sp.lambdify((p, x), log_score_a, "mpmath")
    fb = sp.lambdify((p, x), log_score_b, "mpmath")
    checks = [identity(log_old, log_add),           # old proxy = old additive filter
              identity(log_exp_a, p - x),           # exponential score = additive score
              inequality(lambda q: fa(q, q) < 0),   # 1+s proxy: negative drift at the truth
              inequality(lambda q: fb(q, q) < 0),
              inequality(lambda q: root_of(fa, q) < q),
              inequality(lambda q: root_of(fb, q) > q)]
    return checks


def main(section):
    checks = {"old-drift": old_drift, "fixed-points": fixed_points, "score": score,
              "adam-ets": adam_ets}[section]()
    unverified = sum(c[0] for c in checks)
    counter = sum(c[1] for c in checks)
    print(section, checks)
    results = [{"quantity": "identities sympy could not reduce to zero", "value": unverified,
                "conditions": {"section": section}},
               {"quantity": "numerical counterexamples", "value": counter,
                "conditions": {"section": section}},
               {"quantity": "checks performed", "value": len(checks),
                "conditions": {"section": section}}]
    path = os.environ.get("CLAIMSET_RESULTS")
    if path:
        with open(path, "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    main(sys.argv[1])
