"""Admissibility of pure additive lagged SSOE models (ADAM notation), numpy prototype.

y_t = w' v_{t-l} + e_t,  v_t = F v_{t-l} + g e_t,  D = F - g w',  L(B) = diag(B^l_1, ..., B^l_k).
ARIMA form: phi(B) y_t = theta(B) e_t with phi = det(I - F L(B)) and theta = det(I - D L(B)).
Polynomials are coefficient vectors in increasing powers of B.
"""

import json
import os

import numpy as np

P = np.polynomial.polynomial


# Lag-1 expansion of v_t = M v_{t-l}: block j holds v_{j,t}, ..., v_{j,t-l_j+1}
def expand(M, g, w, lags):
    lags = np.asarray(lags)
    start = np.r_[0, np.cumsum(lags)[:-1]]
    last = start + lags - 1
    n = lags.sum()
    A, gt, wt = np.zeros((n, n)), np.zeros(n), np.zeros(n)
    for j, s in enumerate(start):
        A[s, last] = M[j]
        for i in range(1, lags[j]):
            A[s + i, s + i - 1] = 1
        gt[s], wt[last[j]] = g[j], w[j]
    return A, gt, wt


# The current smoothEigens(): eigenvalues of the diagonal blocks of D, one block per unique lag
def per_lag_max(F, g, w, lags):
    D, lags = F - np.outer(g, w), np.asarray(lags)
    moduli = [np.abs(np.linalg.eigvals(D[np.ix_(i, i)])).max()
              for i in (np.where(lags == lag)[0] for lag in np.unique(lags))]
    return max(moduli)


# Reference: max |lambda| over the modes of the expanded D that g reaches and w sees (PBH rank tests),
# checking the eigenvalues from the largest down until one is both controllable and observable
def pbh_max(F, g, w, lags, tol=1e-7):
    A, gt, wt = expand(F - np.outer(g, w), g, w, lags)
    lam = np.linalg.eigvals(A)
    scale = np.linalg.norm(A, 2) + np.linalg.norm(gt) + np.linalg.norm(wt)
    eye = np.eye(len(lam))
    for x in lam[np.argsort(-np.abs(lam))]:
        sObs = np.linalg.svd(np.vstack([x * eye - A, wt]), compute_uv=False)[-1]
        sCtr = np.linalg.svd(np.hstack([x * eye - A, gt[:, None]]), compute_uv=False)[-1]
        if min(sObs, sCtr) > tol * scale:
            return abs(x)
    return 0.0


# Coefficients of det(I - M L(B)) from its values at N roots of unity (inverse DFT): O(N k^3)
def det_poly(M, lags):
    lags = np.asarray(lags)
    d = int(lags.sum())
    N = 1 << int(np.ceil(np.log2(d + 1)))
    z = np.exp(2j * np.pi * np.arange(N) / N)
    eye = np.eye(len(lags))
    values = np.array([np.linalg.det(eye - M * zz ** lags) for zz in z])
    return np.real(np.fft.fft(values) / N)[:d + 1]


def arima_form(F, g, w, lags):
    return det_poly(F - np.outer(g, w), lags), det_poly(F, lags)


# Integer coefficients of the n-th cyclotomic polynomial
def cyclotomic(n):
    p = np.r_[-1.0, np.zeros(n - 1), 1.0]
    for d in range(1, n):
        if n % d == 0:
            p = P.polydiv(p, cyclotomic(d))[0]
    return np.round(p)


def _divides(c, poly, tol):
    q, r = P.polydiv(poly, c)
    return q, np.abs(r).max() <= tol * np.abs(poly).sum()


# Divide theta and phi by every cyclotomic factor Phi_n (n divides a lag) that they share
def cancel(theta, phi, lags, tol=1e-9):
    h = np.array([1.0])
    for n in sorted({n for lag in lags for n in range(1, lag + 1) if lag % n == 0}):
        c = cyclotomic(n)
        while len(theta) > len(c) - 1 and len(phi) > len(c) - 1:
            qt, okt = _divides(c, theta, tol)
            qp, okp = _divides(c, phi, tol)
            if not (okt and okp):
                break
            theta, phi, h = qt, qp, P.polymul(h, c)
    return theta, phi, h


def trim(c):
    c = np.asarray(c, float)
    return c[:np.flatnonzero(np.abs(c) > 1e-14 * np.abs(c).max())[-1] + 1]


# max |lambda| = 1 / min |root| from all the roots (numpy companion eigenvalues)
def max_lambda_roots(c):
    c = trim(c)
    return 0.0 if len(c) < 2 else 1 / np.abs(P.polyroots(c)).min()


# Step-down (Schur-Cohn) test: True if all roots of c lie outside the unit circle
def schur_cohn(c):
    c = trim(c)
    c = c / c[0]
    for j in range(len(c) - 1, 0, -1):
        kappa = c[j]
        if abs(kappa) >= 1:
            return False
        c = (c[:j] - kappa * c[j:0:-1]) / (1 - kappa ** 2)
    return True


# max |lambda| < r iff the step-down passes on c_i r^{-i}; bisection when the model is not admissible
def spectral_radius(c, tol=1e-10):
    c = trim(c)
    scaled = lambda r: schur_cohn(c * r ** -np.arange(len(c)))
    if scaled(1.0):
        return None
    lo, hi = 1.0, 2.0
    while not scaled(hi):
        lo, hi = hi, 2 * hi
    while hi - lo > tol:
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if scaled(mid) else (mid, hi)
    return hi


# ETS with additive components: level, optional (damped) trend, seasonals, optional constant
def ets(alpha, seasonal=(), gamma=(), beta=None, phi=1.0, constant=False):
    lags = [1] + ([1] if beta is not None else []) + list(seasonal) + ([1] if constant else [])
    k = len(lags)
    F, g, w = np.eye(k), np.zeros(k), np.ones(k)
    g[0] = alpha
    if beta is not None:
        F[0, 1] = F[1, 1] = w[1] = phi
        g[1] = beta
    s = 2 if beta is not None else 1
    g[s:s + len(seasonal)] = gamma
    return F, g, w, lags


def gum_random(lags, rng, sd=0.6):
    k = len(lags)
    return rng.normal(0, sd, (k, k)), rng.normal(0, sd / 2, k), rng.normal(1, sd / 2, k), list(lags)


def gum_start(lags, g=0.1):
    k = len(lags)
    return np.eye(k), np.full(k, g), np.ones(k), list(lags)


def write_results(results):
    path = os.environ.get("CLAIMSET_RESULTS")
    if path:
        with open(path, "w") as f:
            json.dump(results, f, indent=1)
    for r in results:
        print(r["quantity"], r.get("conditions", ""), r["value"])
