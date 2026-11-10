"""Independent helpers of the checker AC3: exact polynomial determinant (Fractions), gcd, ADAM's ARIMA part."""
import json
import os
from fractions import Fraction as Fr

import numpy as np


def padd(a, b):
    n = max(len(a), len(b))
    return [(a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0) for i in range(n)]


def pmul(a, b):
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x == 0:
            continue
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out


def pscale(a, s):
    return [s * x for x in a]


def ptrim(a):
    a = list(a)
    while len(a) > 1 and a[-1] == 0:
        a.pop()
    return a


def pdivmod(a, b):
    a, b = ptrim(a), ptrim(b)
    if len(a) < len(b):
        return [0], a
    q = [0] * (len(a) - len(b) + 1)
    r = list(a)
    for d in range(len(a) - len(b), -1, -1):
        c = r[d + len(b) - 1] / b[-1]
        q[d] = c
        for i, y in enumerate(b):
            r[i + d] -= c * y
    return ptrim(q), ptrim(r[:len(b) - 1] or [0])


def pgcd(a, b):
    a, b = ptrim(a), ptrim(b)
    while any(x != 0 for x in b):
        a, b = b, pdivmod(a, b)[1]
    return [x / a[-1] for x in a]


# det(I - M L(z)) by expansion over column subsets (rows taken in order); entries are polynomials
def detpoly(M, lags, exact=False):
    k = len(lags)
    one = Fr(1) if exact else 1.0
    zero = Fr(0) if exact else 0.0

    def entry(i, j):
        p = [zero] * (lags[j] + 1)
        if i == j:
            p[0] = one
        p[lags[j]] = p[lags[j]] - M[i][j]
        return ptrim(p)

    memo = {}

    def rec(row, used):
        if row == k:
            return [one]
        key = (row, used)
        if key in memo:
            return memo[key]
        total = [zero]
        sign = 1
        free = [j for j in range(k) if not used >> j & 1]
        for pos, j in enumerate(free):
            e = entry(row, j)
            if all(x == 0 for x in e):
                sign = -sign
                continue
            sub = rec(row + 1, used | 1 << j)
            term = pmul(e, sub)
            total = padd(total, pscale(term, sign))
            sign = -sign
        memo[key] = ptrim(total)
        return memo[key]

    return rec(0, 0)


def ar_factor(coefs, lag, sign, exact=False):
    p = [Fr(0) if exact else 0.0] * (len(coefs) * lag + 1)
    p[0] = Fr(1) if exact else 1.0
    for j, c in enumerate(coefs):
        p[(j + 1) * lag] = sign * c
    return p


def prod(polys, exact=False):
    out = [Fr(1) if exact else 1.0]
    for p in polys:
        out = pmul(out, p)
    return out


# ADAM's ARIMA part from the orders: one state per lag in the support of the ARI and MA polynomials,
# transition row -ari_l in every ARIMA column, persistence ma_l - ari_l, measurement 1
def arima_polys(ar, i, ma, exact=False):
    """ar, ma: lists of (lag, coefs); i: list of (lag, d). Returns ari, maPoly (lists)."""
    f = [ar_factor(c, l, -1, exact) for l, c in ar]
    f += [ar_factor([1], l, -1, exact) for l, d in i for _ in range(d)]
    ari = prod(f, exact)
    maPoly = prod([ar_factor(c, l, 1, exact) for l, c in ma], exact)
    return ari, maPoly


def arima_matrices(ari, maPoly, exact=False):
    lags = sorted({j for j in range(1, len(ari)) if ari[j] != 0} | {j for j in range(1, len(maPoly)) if maPoly[j] != 0})
    pad = lambda p, j: p[j] if j < len(p) else (Fr(0) if exact else 0.0)
    k = len(lags)
    F = [[-pad(ari, lj) for _ in range(k)] for lj in lags]
    g = [pad(maPoly, lj) - pad(ari, lj) for lj in lags]
    return F, g, [Fr(1) if exact else 1.0] * k, lags


def blockdiag(parts):
    n = sum(len(p[3]) for p in parts)
    F = [[0 * parts[0][0][0][0]] * n for _ in range(n)]
    F = [[(Fr(0) if isinstance(parts[0][0][0][0], Fr) else 0.0) for _ in range(n)] for _ in range(n)]
    g, w, lags, o = [], [], [], 0
    for Fi, gi, wi, li in parts:
        k = len(li)
        for a in range(k):
            for b in range(k):
                F[o + a][o + b] = Fi[a][b]
        g += list(gi)
        w += list(wi)
        lags += list(li)
        o += k
    return F, g, w, lags


def Dmat(F, g, w):
    k = len(g)
    return [[F[i][j] - g[i] * w[j] for j in range(k)] for i in range(k)]


def results_out(results):
    path = os.environ.get("CLAIMSET_RESULTS")
    if path:
        with open(path, "w") as f:
            json.dump(results, f, indent=1)
    for r in results:
        print(r["quantity"], r.get("conditions", ""), r["value"])


# n-th cyclotomic polynomials over the integers (Fractions or ints)
def cyclo(n, _c={}):
    if n in _c:
        return _c[n]
    p = [-1] + [0] * (n - 1) + [1]
    for d in range(1, n):
        if n % d == 0:
            p = pdivmod(p, cyclo(d))[0]
    _c[n] = [int(round(x)) for x in p]
    return _c[n]
