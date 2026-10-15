"""Recovery of the scale of the Gamma and the inverse Gaussian by the moment equations.

Claim (INTERMITTENT_LIKELIHOOD.tex, Section 5.3 "The scale", the remark after
Proposition "moments", E21): "On simulated errors with the drift of iETS(M,N,N) these
equations recover the scale without bias. Within a fitted model the estimate is still
biased upwards, because the states after an interval are updated with the gain g rather
than with the conditional mean, so that part of the drift leaks into the following
errors." No numbers are given. Targets (qualitative): (i) the bias of the estimate is
zero within Monte Carlo error on simulated errors; (ii) it is positive within a fitted
model whose states follow the recursion.

Model. iETS(M,N,N): sizes z_t = l_{t-1}(1+eps_t), E(1+eps) = 1, Var(1+eps) = s; the
level moves at every period, observed or not:
  conventional ETS (Hyndman et al.; the package's default, ets="conventional"):
      l_t = l_{t-1}(1 + alpha eps_t),  L(eps) = log(1 + alpha eps);
  ETS of ADAM (ets="adam"):
      l_t = l_{t-1}(1+eps_t)^alpha,    L(eps) = alpha log(1+eps).
The paper leaves the form open and the drift differs between them, so both are computed
and labelled (column `ets`). Distributions (greybox parameterisation, as the package):
  dgamma: 1+eps ~ Gamma(shape 1/s, scale s); draws numpy gamma(1/s, s);
  dinvgauss: 1+eps ~ IG(mean 1, dispersion s), variance s; draws numpy wald(1, 1/s).
Intervals j_t (periods since the previous observed size, the first one counted from the
initial level): fixed j in {2, 5, 10} ("j2", "j5", "j10"), or geometric with the
occurrence probability p in {0.5, 0.2} ("geo0.5", "geo0.2"), P(j) = p(1-p)^(j-1).

Moment estimator (Proposition "moments", eq:fixedpoint). Relative errors e_t =
z_t/mu_hat-1. Drift moments (eq:driftmoments, iETS(M,N,N): c_i = 1): m_t = (j_t-1) E L,
w_t = (j_t-1) Var L, with E L and Var L by the 32-node Gauss-Legendre rule on (0,1)
applied to the greybox quantile function of 1+eps (numpy leggauss(32), checked equal to
the package's constants and to its gap_drift() moments). E D = exp(m + w/2), E D^2 =
exp(2m + 2w), E 1/D = exp(-m + w/2).
  Gamma:  s = h(s) = [sum e^2 - sum(E D^2 - 2 E D + 1)] / sum E D^2
  IG:     s = h(s) = [sum e^2/(1+e) - sum(E D - 2 + E 1/D)] / sum E 1/D
Two solvers, reported separately:
  "fixed_point": the paper's iteration s <- h(s) from the one-step estimate, stopping at
      |s_new - s| <= 1e-10 s, at most 100 iterations (as the package's scaler); an
      iterate that is not positive and finite, or no convergence, counts as a failure
      (n_fail);
  "root": the solution of s = h(s) by bisection in log s (64 halvings): g(s) = s - h(s)
      is -s0 < 0 as s -> 0, g(S_MIN) < 0 is checked, and the upper end doubles from the
      one-step estimate until g > 0 (an overflow of the drift moments counts as g > 0);
      NaN when no sign change is found in [S_MIN, S_MAX] = [0.005, 20].
"one_step" is the estimate without the drift (s0 = mean e^2, or mean e^2/(1+e) for the
IG). For speed, E L and Var L at an arbitrary s are interpolated from the 32-node rule
evaluated on 2001 log-spaced s in [S_MIN, S_MAX] (cubic splines of E L/s and Var L/s in
log s); the interpolation error against the rule is checked at 1200 random s
(table_vs_rule_maxrel). Values of s outside [S_MIN, S_MAX] count as failures. Reported
per cell: mean, median, bias, its Monte Carlo s.e. (sd/sqrt(n_ok)), relative bias and
t = bias/s.e., and drift_index = E(j-1) alpha^2 s, the paper's order of the dependence
of h on s, by which the summaries are binned.

Part A (simulated errors, the first sentence): 1 + e_t = (1+eps_t) D_t, D_t = exp(d_t),
d_t = sum of L(eps) over the j_t - 1 unobserved periods, from actual draws (eq:drift;
the states at the previous observed period known). Grid: dist {dgamma, dinvgauss} x ets
{conventional, adam} x s {0.05, 0.2, 0.5, 1} x alpha {0.1, 0.3, 0.6} x pattern {j2, j5,
j10, geo0.5, geo0.2} x T1 (observed sizes per replication) {100, 1000}; R_A = 2000
replications per cell. Diagnostics s_inf / s_inf_quad: the large-sample limit of the
root (the equation with the exact E e^2, or E e^2/(1+e), at the true s, i.e. E D^k =
M(k)^(j-1) with M(k) = E exp(k L(eps)) by scipy quad over the greybox density, and the
normal drift moments on the right-hand side, with E L and Var L by the 32-node rule or
by adaptive quadrature); s_inf - s is the bias that remains as T1 grows, from the normal
approximation of the drift (and the rule). For the geometric intervals the sums over j
are truncated where the terms fall below 1e-15 and s_inf is NaN when they diverge ((1-p)
max M(k) >= 1: E D^2 or E 1/D infinite). h_slope = h'(s_inf) of this limiting equation
(central difference): the fixed-point iteration converges only where |h'| < 1.

Part B (within a fitted model, the second sentence). Series of iETS(M,N,N) with T1 = 100
observed sizes and T = sum of the intervals periods (zeros at the unobserved periods,
the occurrence known), l0 = 100, the same ETS form for simulation and fitting. Grid:
dist x ets x s {0.05, 0.5} x alpha {0.1, 0.3, 0.6} x pattern {j2, j5, j10, geo0.5,
geo0.2}. Methods:
  "recursion_{one_step,fixed_point,root}": the estimator of the remark, computed here
      directly: the states follow the recursion eq:recursion with the true alpha and l0
      (constant over the unobserved periods, updated with the gain alpha at the observed
      ones), the errors against it, and the estimators above with the reset drift
      moments (R_B = 2000). The leak is the difference to Part A's "root" at T1 = 100 in
      the same cell (printed, with the combined s.e.).
  "package_true_params": smooth's Python ADAM(model="MNN", distribution=dist,
      occurrence="fixed", ets=ets, persistence={"alpha": alpha}, initial={"level": l0}),
      attribute `scale` (the estimate divided by T1, no df correction); R_FIX = 400;
  "package_estimated": the same with alpha and the initial level estimated (defaults:
      backcasting, bounds "usual"); R_EST = 100; the mean estimated alpha is
      `alpha_hat`.
The package fits use the first R_FIX / R_EST of the same series. A failed fit (exception
or NaN scale) counts in n_fail. The package no longer implements the recursion of the
remark for these models: on this branch the fit runs the approximate filter of Remark
"approxfilter" (cost_functions.CF, filter_logs=True for the pure multiplicative ETS with
dgamma/dinvgauss), whose drift is N(0, f_t - sigma_u^2) (conventional) or N(mu_u (m_t -
1), sigma_u^2 (f_t - 1)) (ADAM ETS), with the scale from the same fixed-point iteration
(utils.scaler); there is no public switch back to the recursion. So "recursion_*"
reproduce the claim's setting, and "package_*" measure what the package does now. A
sanity check compares the recursion here with the package on a complete series (no
zeros), where both are the SSOE recursion.

Seed: numpy SeedSequence(SEED = 20261015): one child for Part A and one for Part B, then
one grandchild per cell, so the results do not depend on the number of worker processes
(WORKERS = 3). Versions: smooth 1.2.0 (python, editable from the working tree), greybox
1.0.9, numpy 2.4.6, scipy 1.17.1, Python 3.11.15. The commit (git rev-parse HEAD) and
the dirty state are printed at run time and written into the CSV.

Run from the repository root:
  python/.venv/bin/python python/docs/claimset/scripts/scale_recovery.py
Output: results/scale_recovery.csv next to this script (one row per cell and method).

"""

import subprocess
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from itertools import product
from pathlib import Path

import greybox as gb
import numpy as np
import pandas as pd
import scipy
from scipy import integrate
from scipy.interpolate import CubicSpline

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
OUT = HERE / "results" / "scale_recovery.csv"

SEED = 20261015
NGL = 32
TOL, MAXIT, BISECT = 1e-10, 100, 64
S_MIN, S_MAX, TABLE_N = 5e-3, 20.0, 2001
DISTS = ("dgamma", "dinvgauss")
ETS = ("conventional", "adam")
PATTERNS = ("j2", "j5", "j10", "geo0.5", "geo0.2")
A_S, A_ALPHA, A_T1, R_A = (0.05, 0.2, 0.5, 1.0), (0.1, 0.3, 0.6), (100, 1000), 2000
B_S, B_ALPHA, B_T1, L0 = (0.05, 0.5), (0.1, 0.3, 0.6), 100, 100.0
R_B, R_FIX, R_EST = 2000, 400, 100
CHUNK = 250
WORKERS = 3

_x, _w = np.polynomial.legendre.leggauss(NGL)
GL_NODES, GL_WEIGHTS = (_x + 1) / 2, _w / 2


# --- distributions --------------------------------------------------------------------
def quantiles(dist, s):
    """Quantiles of 1+eps at the Gauss-Legendre nodes for a scalar s (greybox)."""
    if dist == "dgamma":
        return np.asarray(gb.qgamma(GL_NODES, shape=1 / s, scale=s))
    return np.asarray(gb.qinvgauss(GL_NODES, 1.0, s))


def density(dist, s):
    if dist == "dgamma":
        return lambda x: gb.dgamma(x, shape=1 / s, scale=s)
    return lambda x: gb.dinvgauss(x, 1.0, s)


def draw(rng, dist, s, size):
    """Draws of 1+eps."""
    if dist == "dgamma":
        return rng.gamma(1 / s, s, size)
    return rng.wald(1.0, 1 / s, size)


def innovation(q, ets, alpha):
    """L(eps) as a function of 1+eps = q."""
    if ets == "adam":
        return alpha * np.log(q)
    return np.log1p(alpha * (q - 1))


def drift_moments(dist, ets, alpha, s):
    """E L and Var L by the 32-node Gauss-Legendre rule on the quantile function."""
    lq = innovation(quantiles(dist, s), ets, alpha)
    mean = lq @ GL_WEIGHTS
    return mean, (lq**2) @ GL_WEIGHTS - mean**2


def moments_quad(dist, ets, alpha, s):
    """E L and Var L by adaptive quadrature over the density (reference)."""
    f = density(dist, s)
    ref = [
        sum(
            integrate.quad(
                lambda x, k=k: innovation(x, ets, alpha) ** k * f(x),
                a,
                b,
                limit=400,
                epsabs=0,
                epsrel=1e-12,
            )[0]
            for a, b in ((0, 1), (1, np.inf))
        )
        for k in (1, 2)
    ]
    return ref[0], ref[1] - ref[0] ** 2


@lru_cache(maxsize=None)
def moment_table(dist, ets, alpha):
    """drift_moments() on TABLE_N log-spaced s in [S_MIN, S_MAX], interpolated by cubic
    splines of E L / s and Var L / s in log s; NaN outside the range."""
    grid = np.geomspace(S_MIN, S_MAX, TABLE_N)
    vals = np.array([drift_moments(dist, ets, alpha, x) for x in grid]) / grid[:, None]
    splines = [CubicSpline(np.log(grid), vals[:, k]) for k in (0, 1)]

    def moments(s):
        s = np.asarray(s, dtype=float)
        inside = (s >= S_MIN) & (s <= S_MAX)
        ls = np.log(np.where(inside, s, 1.0))
        return tuple(np.where(inside, sp(ls) * s, np.nan) for sp in splines)

    return moments


def exact_moment(dist, ets, alpha, s, k):
    """M(k) = E exp(k L(eps)) by adaptive quadrature over the density."""
    f = density(dist, s)

    def integrand(x):
        return np.exp(k * innovation(x, ets, alpha)) * f(x)

    return sum(
        integrate.quad(integrand, a, b, limit=400, epsabs=0, epsrel=1e-12)[0]
        for a, b in ((0, 1), (1, np.inf))
    )


# --- the estimator --------------------------------------------------------------------
def statistic(e, dist):
    return np.sum(e**2 if dist == "dgamma" else e**2 / (1 + e), axis=1)


def h_value(stat, nm1, dist, moments, s):
    """The right-hand side h(s) of eq:fixedpoint (Gamma) or its IG analogue, per row."""
    mu, var = moments(s)
    m, w = nm1 * mu[:, None], nm1 * var[:, None]
    with np.errstate(over="ignore", invalid="ignore"):
        ed, ed2, edi = np.exp(m + w / 2), np.exp(2 * m + 2 * w), np.exp(-m + w / 2)
        if dist == "dgamma":
            return (stat - np.sum(ed2 - 2 * ed + 1, 1)) / np.sum(ed2, 1)
        return (stat - np.sum(ed - 2 + edi, 1)) / np.sum(edi, 1)


def fixed_point(e, nm1, dist, moments):
    """The iteration s <- h(s) from the one-step estimate, per row of e (R x T1), nm1 =
    j_t - 1. Returns s0, s_hat (NaN when an iterate is not positive and finite, or after
    MAXIT iterations) and the number of iterations."""
    stat = statistic(e, dist)
    s0 = stat / e.shape[1]
    s, iters = s0.copy(), np.zeros(len(s0), dtype=int)
    active = np.isfinite(s) & (s > 0)
    s[~active] = np.nan
    for _ in range(MAXIT):
        if not active.any():
            break
        idx = np.flatnonzero(active)
        new = h_value(stat[idx], nm1[idx], dist, moments, s[idx])
        iters[idx] += 1
        bad = ~np.isfinite(new) | (new <= 0)
        done = bad | (np.abs(new - s[idx]) <= TOL * s[idx])
        s[idx] = np.where(bad, np.nan, new)
        active[idx[done]] = False
    s[active] = np.nan
    return s0, s, iters


def root(e, nm1, dist, moments):
    """The solution of s = h(s) by bisection (in log s), independent of the iteration:
    g(s) = s - h(s) < 0 at S_MIN (checked) and the upper end doubles from the one-step
    estimate until g > 0 (an overflow of the drift moments counts as g > 0); NaN when no
    sign change is found in [S_MIN, S_MAX]."""
    stat = statistic(e, dist)
    s0 = stat / e.shape[1]

    def positive(s):
        g = s - h_value(stat, nm1, dist, moments, s)
        return ~(g <= 0)

    lo, hi = np.full(len(s0), S_MIN), np.clip(s0, S_MIN, S_MAX)
    ok = ~positive(lo)
    for _ in range(40):
        up = positive(hi)
        grow = ~up & (hi < S_MAX)
        if not grow.any():
            break
        hi[grow] = np.minimum(2 * hi[grow], S_MAX)
    ok &= positive(hi)
    for _ in range(BISECT):
        mid = np.sqrt(lo * hi)
        up = positive(mid)
        hi, lo = np.where(up, mid, hi), np.where(up, lo, mid)
    return np.where(ok, np.sqrt(lo * hi), np.nan)


# --- simulation helpers ---------------------------------------------------------------
def intervals(rng, pattern, size):
    if pattern.startswith("geo"):
        return rng.geometric(float(pattern[3:]), size)
    return np.full(size, int(pattern[1:]))


def segment_sums(values, counts):
    """Sums of consecutive segments of values with lengths counts (zeros allowed)."""
    out = np.zeros(counts.size)
    pos = counts > 0
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])[pos]
    if starts.size:
        out[pos] = np.add.reduceat(values, starts)
    return out


def simulate_errors(rng, dist, ets, s, alpha, pattern, t1, reps):
    """Part A: e (reps x t1) and j - 1, with the drift from the actual innovations."""
    nm1 = intervals(rng, pattern, (reps, t1)) - 1
    eps1 = draw(rng, dist, s, (reps, t1))
    drift = innovation(draw(rng, dist, s, int(nm1.sum())), ets, alpha)
    d = segment_sums(drift, nm1.ravel()).reshape(reps, t1)
    return eps1 * np.exp(d) - 1, nm1


def simulate_series(rng, dist, ets, s, alpha, pattern, t1):
    """Part B: an iETS(M,N,N) series with t1 observed sizes; y, the observed mask, j."""
    j = intervals(rng, pattern, t1)
    t = int(j.sum())
    q = draw(rng, dist, s, t)
    log_level = np.log(L0) + np.concatenate([[0], np.cumsum(innovation(q, ets, alpha))])
    z = np.exp(log_level[:-1]) * q
    observed = np.zeros(t, dtype=bool)
    observed[np.cumsum(j) - 1] = True
    return np.where(observed, z, 0.0), observed, j


def recursion_errors(y, observed, ets, alpha):
    """eq:recursion at the true parameters: the level is constant at the unobserved
    periods and updated with the gain alpha at the observed ones."""
    level, errors = L0, []
    for value in y[observed]:
        e = value / level - 1
        errors.append(e)
        level *= (1 + e) ** alpha if ets == "adam" else 1 + alpha * e
    return np.array(errors)


# --- Part A ---------------------------------------------------------------------------
def s_infinity(dist, ets, alpha, s, pattern, rule):
    """Large-sample limit of the root: exact E e^2 (or E e^2/(1+e)) at s, normal drift
    moments on the right-hand side, with E L and Var L by the 32-node rule (rule="gl32")
    or by adaptive quadrature (rule="quad"); solved by bisection as root(). Returns it
    and the slope h'(s) there (central difference): the iteration s <- h(s) converges
    locally only when |h'| < 1."""
    mk = {k: exact_moment(dist, ets, alpha, s, k) for k in (1, 2, -1)}
    if pattern.startswith("geo"):
        p = float(pattern[3:])
        rate = (1 - p) * max(mk[2], mk[-1], mk[1], 1.0)
        if rate >= 1:
            return np.nan, np.nan
        jmax = int(np.ceil(np.log(1e-15) / np.log(rate))) + 2
        js = np.arange(1, jmax + 1)
        pj = p * (1 - p) ** (js - 1)
    else:
        js, pj = np.array([int(pattern[1:])]), np.array([1.0])
    n = js - 1
    if dist == "dgamma":
        target = np.sum(pj * ((1 + s) * mk[2] ** n - 2 * mk[1] ** n + 1))
    else:
        target = np.sum(pj * (mk[1] ** n - 2 + (1 + s) * mk[-1] ** n))
    moments = drift_moments if rule == "gl32" else moments_quad

    def h(x):
        mu, var = moments(dist, ets, alpha, x)
        m, w = n * mu, n * var
        ed, ed2, edi = np.exp(m + w / 2), np.exp(2 * m + 2 * w), np.exp(-m + w / 2)
        if dist == "dgamma":
            return (target - np.sum(pj * (ed2 - 2 * ed + 1))) / np.sum(pj * ed2)
        return (target - np.sum(pj * (ed - 2 + edi))) / np.sum(pj * edi)

    def positive(x):
        return not (x - h(x) <= 0)

    lo, hi = S_MIN, s
    if positive(lo):
        return np.nan, np.nan
    while not positive(hi):
        if hi >= S_MAX:
            return np.nan, np.nan
        hi = min(2 * hi, S_MAX)
    for _ in range(BISECT):
        mid = np.sqrt(lo * hi)
        lo, hi = (lo, mid) if positive(mid) else (mid, hi)
    x = np.sqrt(lo * hi)
    return x, (h(x * (1 + 1e-5)) - h(x * (1 - 1e-5))) / (2e-5 * x)


def summarise(est, s, **keys):
    ok = np.isfinite(est)
    n = int(ok.sum())
    mean = float(np.mean(est[ok])) if n else np.nan
    se = float(np.std(est[ok], ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    median = float(np.median(est[ok])) if n else np.nan
    return dict(
        keys,
        s=s,
        n_ok=n,
        n_fail=int(est.size - n),
        mean=mean,
        bias=mean - s,
        mc_se=se,
        rel_bias=(mean - s) / s,
        t_stat=(mean - s) / se,
        median=median,
    )


def estimates(e, nm1, s, **keys):
    """Rows of the one-step estimate, the fixed-point iteration and the root."""
    dist = keys["dist"]
    moments = moment_table(dist, keys["ets"], keys["alpha"])
    s0, fp, its = fixed_point(e, nm1, dist, moments)
    rt = root(e, nm1, dist, moments)
    both = np.isfinite(fp) & np.isfinite(rt)
    agree = float(np.max(np.abs(fp[both] / rt[both] - 1))) if both.any() else np.nan
    return [
        summarise(s0, s, method="one_step", **keys),
        summarise(
            fp,
            s,
            method="fixed_point",
            iter_mean=float(its.mean()),
            iter_max=int(its.max()),
            fp_vs_root_maxrel=agree,
            **keys,
        ),
        summarise(rt, s, method="root", **keys),
    ]


def part_a_cell(args):
    (dist, ets, s, alpha, pattern), ss = args
    rows = []
    for t1, ss_t1 in zip(A_T1, ss.spawn(len(A_T1))):
        rng = np.random.default_rng(ss_t1)
        e, nm1 = zip(
            *[
                simulate_errors(
                    rng, dist, ets, s, alpha, pattern, t1, min(CHUNK, R_A - k)
                )
                for k in range(0, R_A, CHUNK)
            ]
        )
        rows += estimates(
            np.vstack(e),
            np.vstack(nm1),
            s,
            part="A",
            dist=dist,
            ets=ets,
            alpha=alpha,
            pattern=pattern,
            T1=t1,
            R=R_A,
        )
    sinf = [s_infinity(dist, ets, alpha, s, pattern, r) for r in ("gl32", "quad")]
    for row in rows:
        row.update(s_inf=sinf[0][0], s_inf_quad=sinf[1][0], h_slope=sinf[0][1])
    return rows


# --- Part B ---------------------------------------------------------------------------
def package_fit(dist, ets, alpha, y, fixed):
    from smooth import ADAM

    kw = dict(persistence={"alpha": alpha}, initial={"level": L0}) if fixed else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            m = ADAM(
                model="MNN", distribution=dist, occurrence="fixed", ets=ets, **kw
            ).fit(y)
            return m.scale, m.persistence_vector["alpha"]
        except Exception:  # noqa: BLE001 - a failed fit is counted, not hidden
            return np.nan, np.nan


def part_b_cell(args):
    (dist, ets, s, alpha, pattern), ss = args
    rng = np.random.default_rng(ss)
    t0 = time.time()
    series = [
        simulate_series(rng, dist, ets, s, alpha, pattern, B_T1) for _ in range(R_B)
    ]
    e = np.array([recursion_errors(y, o, ets, alpha) for y, o, _ in series])
    nm1 = np.array([j - 1 for _, _, j in series])
    keys = dict(part="B", dist=dist, ets=ets, alpha=alpha, pattern=pattern, T1=B_T1)
    rows = estimates(e, nm1, s, R=R_B, **keys)
    for row in rows:
        row["method"] = "recursion_" + row["method"]
    fixed = np.array(
        [package_fit(dist, ets, alpha, y, True) for y, _, _ in series[:R_FIX]]
    )
    est = np.array(
        [package_fit(dist, ets, alpha, y, False) for y, _, _ in series[:R_EST]]
    )
    rows += [
        summarise(fixed[:, 0], s, method="package_true_params", R=R_FIX, **keys),
        summarise(
            est[:, 0],
            s,
            method="package_estimated",
            R=R_EST,
            alpha_hat=float(np.nanmean(est[:, 1])),
            **keys,
        ),
    ]
    rows[-1]["seconds"] = time.time() - t0
    return rows


def run_cells(func, grid, seed_seq, label):
    cells = list(product(*grid))
    tasks = list(zip(cells, seed_seq.spawn(len(cells))))
    rows = []
    with ProcessPoolExecutor(WORKERS) as pool:
        for done, cell_rows in enumerate(pool.map(func, tasks), 1):
            rows += cell_rows
            if done % 20 == 0 or done == len(cells):
                print(f"  {label}: {done}/{len(cells)} cells", flush=True)
    return rows


# --- checks ---------------------------------------------------------------------------
def checks():
    from smooth.adam_general.core.utils import var_covar

    out = {}
    out["gl_nodes_vs_package"] = float(
        max(
            np.max(np.abs(GL_NODES - var_covar._GAUSS_LEGENDRE_NODES)),
            np.max(np.abs(GL_WEIGHTS - var_covar._GAUSS_LEGENDRE_WEIGHTS)),
        )
    )
    # densities: mass, mean and variance of 1+eps
    worst = 0.0
    for dist, s in product(DISTS, A_S):
        f = density(dist, s)
        mom = [
            sum(
                integrate.quad(lambda x, k=k: x**k * f(x), a, b, limit=400)[0]
                for a, b in ((0, 1), (1, np.inf))
            )
            for k in (0, 1, 2)
        ]
        worst = max(worst, abs(mom[0] - 1), abs(mom[1] - 1), abs(mom[2] - 1 - s) / s)
    out["density_mass_mean_var_maxerr"] = worst
    # 32-node rule against adaptive quadrature for E L and Var L; package's moments
    worst_rel, worst_pkg = 0.0, 0.0
    for dist, ets, s, alpha in product(DISTS, ETS, A_S, A_ALPHA):
        ref = moments_quad(dist, ets, alpha, s)
        mu, var = drift_moments(dist, ets, alpha, s)
        worst_rel = max(worst_rel, abs(mu / ref[0] - 1), abs(var / ref[1] - 1))
        noise = var_covar.gap_drift([alpha], [True], dist, ets == "adam", s)[2]
        worst_pkg = max(worst_pkg, abs(noise[0][1] - mu), abs(noise[1][1, 1] - var))
    # the interpolation of the table against the rule at random s (log-uniform)
    rng = np.random.default_rng(SEED)
    worst_tab = 0.0
    for dist, ets, alpha in product(DISTS, ETS, A_ALPHA):
        table = moment_table(dist, ets, alpha)
        for x in np.exp(rng.uniform(np.log(S_MIN), np.log(S_MAX), 100)):
            direct = drift_moments(dist, ets, alpha, x)
            tab = table(np.array([x]))
            worst_tab = max(
                worst_tab, *(abs(tab[k][0] / direct[k] - 1) for k in (0, 1))
            )
    out["table_vs_rule_maxrel"] = worst_tab
    out["gl32_vs_quad_maxrel"] = worst_rel
    out["drift_moments_vs_package_maxabs"] = worst_pkg
    # exact moments against closed forms (Gamma, ADAM ETS; conventional k = 1, 2)
    from scipy.special import gammaln

    worst = 0.0
    for s, alpha, k in product(A_S, A_ALPHA, (1, 2, -1)):
        a = 1 / s
        closed = np.exp(gammaln(a + k * alpha) - gammaln(a) + k * alpha * np.log(s))
        worst = max(
            worst, abs(exact_moment("dgamma", "adam", alpha, s, k) / closed - 1)
        )
        if k > 0:
            for dist in DISTS:
                closed = 1 + (k == 2) * alpha**2 * s
                worst = max(
                    worst,
                    abs(exact_moment(dist, "conventional", alpha, s, k) / closed - 1),
                )
    out["exact_moments_vs_closed_maxrel"] = worst
    # the recursion here against the package on a complete series (both are SSOE)
    rng = np.random.default_rng(SEED)
    worst = 0.0
    for dist, ets in product(DISTS, ETS):
        y, o, _ = simulate_series(rng, dist, ets, 0.3, 0.3, "j10", 100)
        y = y[o]  # complete data
        e = recursion_errors(y, np.ones(y.size, dtype=bool), ets, 0.3)
        scale = package_fit(dist, ets, 0.3, y, True)[0]
        worst = max(worst, abs(scale / (statistic(e[None, :], dist)[0] / e.size) - 1))
    out["recursion_vs_package_complete_data_rel"] = worst
    return out


def git_state():
    def run(*a):
        return subprocess.run(
            ["git", "-C", str(REPO), *a], capture_output=True, text=True
        ).stdout.strip()

    return run("rev-parse", "HEAD"), bool(run("status", "--porcelain"))


def main():
    import smooth

    commit, dirty = git_state()
    print(f"commit {commit} dirty={dirty}")
    print(
        f"smooth {smooth.__version__}, greybox {gb.__version__}, "
        f"numpy {np.__version__}, scipy {scipy.__version__}, "
        f"Python {sys.version.split()[0]}"
    )
    t0 = time.time()
    chk = checks()
    print("Accuracy checks:")
    for k, v in chk.items():
        print(f"  {k:42s} {v:.3e}")

    root = np.random.SeedSequence(SEED)
    seq_a, seq_b = root.spawn(2)
    ta = time.time()
    rows_a = run_cells(
        part_a_cell, (DISTS, ETS, A_S, A_ALPHA, PATTERNS), seq_a, "part A"
    )
    print(f"Part A: {time.time() - ta:.0f} s")
    tb = time.time()
    rows_b = run_cells(
        part_b_cell, (DISTS, ETS, B_S, B_ALPHA, PATTERNS), seq_b, "part B"
    )
    print(f"Part B: {time.time() - tb:.0f} s")
    df = pd.DataFrame(rows_a + rows_b)
    df["drift_index"] = df.apply(drift_index, axis=1)
    df = df.assign(commit=commit, dirty=dirty, seed=SEED, ngl=NGL, **chk)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    report(df)
    print(f"Total {time.time() - t0:.0f} s; CSV {OUT}")


def drift_index(row):
    """E(j - 1) alpha^2 s: the paper's order of the dependence of h on s."""
    p = row["pattern"]
    mean_nm1 = (
        (1 - float(p[3:])) / float(p[3:]) if p.startswith("geo") else int(p[1:]) - 1
    )
    return mean_nm1 * row["alpha"] ** 2 * row["s"]


BINS = [0, 0.02, 0.1, 0.3, np.inf]


def summary(sub):
    return pd.Series(
        {
            "cells": len(sub),
            "|t|>3": int((sub.t_stat.abs() > 3).sum()),
            "t>3": int((sub.t_stat > 3).sum()),
            "t<-3": int((sub.t_stat < -3).sum()),
            "med_rel_bias": sub.rel_bias.median(),
            "min_rel_bias": sub.rel_bias.min(),
            "max_rel_bias": sub.rel_bias.max(),
            "cells_with_fail": int((sub.n_fail > 0).sum()),
            "fail_share": sub.n_fail.sum() / (sub.n_fail + sub.n_ok).sum(),
        }
    )


def report(df):
    pd.set_option(
        "display.width", 220, "display.max_rows", 600, "display.max_columns", 30
    )
    df = df.assign(bin=pd.cut(df.drift_index, BINS))
    a = df[df.part == "A"]
    print(
        "\n=== Part A: simulated errors with the drift (relative bias = bias / s) ==="
    )
    print("By method, T1 and the drift index E(j-1) alpha^2 s:")
    print(
        a.groupby(["method", "T1", "bin"], observed=True)
        .apply(summary, include_groups=False)
        .round(4)
        .to_string()
    )
    root = a[a.method == "root"].assign(
        inf_rel=lambda d: d.s_inf / d.s - 1,
        inf_quad_rel=lambda d: d.s_inf_quad / d.s - 1,
    )
    print("\nRoot: relative bias at T1 = 100 and 1000, and of the large-sample limit:")
    wide = root.pivot_table(
        index=["dist", "ets", "s", "alpha", "pattern"],
        columns="T1",
        values=["rel_bias", "t_stat"],
    )
    wide.columns = [f"{v}_{t}" for v, t in wide.columns]
    lim = root[root["T1"] == A_T1[0]].set_index(
        ["dist", "ets", "s", "alpha", "pattern"]
    )
    wide = wide.join(lim[["inf_rel", "inf_quad_rel", "drift_index"]])
    fpf = a[(a.method == "fixed_point")].pivot_table(
        index=["dist", "ets", "s", "alpha", "pattern"], columns="T1", values="n_fail"
    )
    wide["fp_fail_share_1000"] = fpf[A_T1[1]] / R_A
    print(wide.round(4).to_string())
    fp = a[a.method == "fixed_point"].assign(
        slope=lambda d: pd.cut(d.h_slope.abs(), [0, 0.5, 0.9, 1.1, np.inf])
    )
    print("\nFixed-point iteration: failures by |h'(s_inf)| (NaN slope: no limit):")
    print(
        fp.groupby(["T1", "slope"], observed=True)
        .apply(summary, include_groups=False)
        .round(4)
        .to_string()
    )

    b = df[df.part == "B"]
    print("\n=== Part B: within a fitted iETS(M,N,N), T1 = 100 ===")
    print(
        b.groupby(["method", "bin"], observed=True)
        .apply(summary, include_groups=False)
        .round(4)
        .to_string()
    )
    keys = ["dist", "ets", "s", "alpha", "pattern"]
    wide = b.pivot_table(index=keys, columns="method", values="rel_bias")
    ref = a[(a.method == "root") & (a["T1"] == B_T1)].set_index(keys)
    rec = b[b.method == "recursion_root"].set_index(keys)
    leak = (rec["mean"] - ref["mean"]).dropna()
    wide["leak_rel"] = leak / leak.index.get_level_values("s")
    wide["leak_t"] = leak / np.hypot(rec["mc_se"], ref["mc_se"])
    fails = b.pivot_table(index=keys, columns="method", values="n_fail")
    wide["pkg_true_fail"] = fails["package_true_params"] / R_FIX
    wide["pkg_est_fail"] = fails["package_estimated"] / R_EST
    est = b[b.method == "package_estimated"].set_index(keys)
    wide["alpha_hat"] = est["alpha_hat"]
    print("Relative bias by method; leak = recursion_root (B) - root (A, T1 = 100):")
    print(wide.round(4).to_string())
    print(
        f"leak: t > 3 in {(wide.leak_t > 3).sum()}, "
        f"t < -3 in {(wide.leak_t < -3).sum()}"
        f" of {wide.leak_t.notna().sum()} cells; median leak_rel "
        f"{wide.leak_rel.median():+.4f}"
    )


if __name__ == "__main__":
    main()
