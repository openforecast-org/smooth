"""KL divergence of the normal-drift mixture and of the one-step density against
the exact density of the size after an interval, iETS(M,N,N).

Claim (INTERMITTENT_LIKELIHOOD.tex, Section 5.2 "The mixture density", after
eq:mixture): "Against the exact density of the size after an interval, computed by
numerical convolution, the largest Kullback-Leibler divergence per observation over
a grid of s in [0.05,1], alpha in [0.1,0.6] and j in [2,20] for iETS(M,N,N) was
0.0007 for the Gamma, 0.0033 for the log-normal and 0.0038 for the inverse Gaussian
distribution, against 2.36, 5.05 and 31.5 for the one-step density."
Targets (maxima over the grid), Gamma / log-normal / inverse Gaussian:
mixture 0.0007 / 0.0033 / 0.0038; one-step density 2.36 / 5.05 / 31.5.

Model. iETS(M,N,N) with the states known at the previous observed period (the
reset approximation of the paper) and mu_hat = 1 (the KL does not depend on it).
After j-1 unobserved periods the size is z = D (1+e_t), log D = sum_{i<j} L(e_i),
e_i iid and independent of e_t (eq:drift with c_i = 1). The paper leaves the form
of L open; both are computed and labelled in the column `ets`, neither is chosen:
  ets="adam":         L(e) = alpha log(1+e)  (ETS of ADAM, l_t = l_{t-1}(1+e)^alpha)
  ets="conventional": L(e) = log(1+alpha e)  (conventional, l_t = l_{t-1}(1+alpha e))
Distributions of 1+e (greybox parameterisation, as smooth's adam_gapDrift):
  Gamma:            shape 1/s, scale s           (mean 1, variance s)
  inverse Gaussian: mean 1, dispersion s         (mean 1, variance s)
  log-normal:       meanlog -s/2, sdlog sqrt(s)  (mean 1, Var log(1+e) = s)

Exact density. Computed for y = log z: the KL is invariant to the monotone map
z -> log z, so the KL on y is the KL on z. On a uniform grid of step H = 0.001:
  1. the masses of L in bins of width H: differences of the CDF of 1+e (greybox
     p*) at the bin edges mapped back through L^{-1}; L is truncated at the
     quantiles EPS and 1-EPS of 1+e (EPS = 1e-12 per innovation);
  2. the masses of log D: the masses of L convolved j-1 times
     (scipy.signal.fftconvolve; rounding negatives of order 1e-17 set to zero;
     the ends with a cumulative mass below TRIM = 1e-15 dropped after each
     convolution);
  3. the density of y = log D + u, u = log(1+e_t), by integrating over d:
     f(y_m) = sum_k P(log D = d_k) f_u(y_m - d_k), with the exact density
     f_u(u) = f_{1+e}(e^u) e^u (greybox d*) on the same grid, truncated at the
     quantiles EPS and 1-EPS of 1+e.
Approximations, at the same y grid:
  - mixture (eq:mixture): the drift d ~ N(m, w), m = (j-1) E L, w = (j-1) Var L,
    with E L and Var L from the package's 32-node Gauss-Legendre rule on the
    quantile function (smooth's gap_drift). The density is the package's
    gap_log_density (16-node Gauss-Hermite: 16 evaluations of the one-step
    density) for the Gamma and the inverse Gaussian, and the paper's closed form
    log z ~ N(-s/2 + m, s + w) (no quadrature) for the log-normal. The column
    kl_mixture_gh16 is the same mixture by this script's own 16-node Gauss-Hermite
    rule (numpy hermgauss) for all three (a check of the package's function, and
    the quadrature version for the log-normal);
  - the one-step density (no drift): f_z(z | mu_hat = 1).
KL(exact || approximation) = int f log(f/g) dy, the expected excess of the
log-score per observation under the true density. The integral is the rectangle
rule on the y grid over the points with f > REL_THRESHOLD * max f
(REL_THRESHOLD = 1e-13); the mass f integrates to there is reported. The reverse
direction is not computed.

Grid: s = 0.05, 0.10, ..., 1.00 (step 0.05, 20 values); alpha = 0.10, 0.15, ...,
0.60 (step 0.05, 11 values); j = 2, 3, ..., 20 (step 1, 19 values): 4180 cells
per distribution and ETS variant, 25080 in all.

Accuracy checks. Columns `mass` (int f dy over the integration region, 1 up to
the truncation) and `mean_error` (|int y f dy - ((j-1) E L + E u)|, with E L and
E u by adaptive quadrature, scipy.integrate.quad, on the density of 1+e). A
refinement run on a sub-grid (s in {0.05, 0.5, 1}, alpha in {0.1, 0.35, 0.6},
j in {2, 10, 20}) and the cells of the maxima, with the step H/2, EPS = 1e-14,
TRIM = 1e-17 and REL_THRESHOLD = 1e-15, saved to results/kl_mixture_refine.csv
with the absolute and relative changes of the KLs. The refinement run also splits
the error of the mixture: kl_mixture_gh100 (the same moments, 100 Gauss-Hermite
nodes: without the error of the 16-node rule) and kl_normal_drift (the moments by
adaptive quadrature and 100 nodes: the error of the normal drift alone).

No random numbers are drawn; the seed (SEED = 20261015, np.random.default_rng)
is set for the record only. Versions at writing: smooth 1.2.0 (editable, working
tree), greybox 1.0.9, numpy 2.4.6, scipy 1.17.1, pandas, Python 3.11.15. The run
prints the versions, the commit of the repository (git rev-parse HEAD) and its
dirty state, and writes them into the CSVs.

Run from the repository root, with the number of worker processes (default 2):
  python/.venv/bin/python claimset/intermittent-likelihood/code/kl_mixture.py [workers]
Outputs: results/kl_mixture.csv (one row per grid cell) and
results/kl_mixture_refine.csv, next to this script.
"""

import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import greybox as gb
import numpy as np
import pandas as pd
import scipy
from scipy import integrate, signal
from scipy.special import logsumexp

import smooth
from smooth.adam_general.core.utils.utils import gap_log_density
from smooth.adam_general.core.utils.var_covar import gap_drift

SEED = 20261015
RNG = np.random.default_rng(SEED)
MAIN = dict(h=1e-3, eps=1e-12, trim=1e-15, threshold=1e-13)
FINE = dict(h=5e-4, eps=1e-14, trim=1e-17, threshold=1e-15)
S_GRID = np.round(np.arange(0.05, 1.0001, 0.05), 2)
ALPHA_GRID = np.round(np.arange(0.10, 0.6001, 0.05), 2)
J_GRID = np.arange(2, 21)
DISTRIBUTIONS = ["dgamma", "dlnorm", "dinvgauss"]
VARIANTS = ["adam", "conventional"]
TARGETS = {
    "dgamma": (0.0007, 2.36),
    "dlnorm": (0.0033, 5.05),
    "dinvgauss": (0.0038, 31.5),
}
KLS = ["kl_mixture", "kl_onestep", "kl_mixture_gh16"]
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def x_functions(dist, s):
    """CDF, quantile function and log-density of 1+e (greybox)."""
    if dist == "dgamma":
        args = (1 / s, s)
        cdf, qf, logd = gb.pgamma, gb.qgamma, gb.dgamma
    elif dist == "dinvgauss":
        args = (1.0, s)
        cdf, qf, logd = gb.pinvgauss, gb.qinvgauss, gb.dinvgauss
    else:
        args = (-s / 2, np.sqrt(s))
        cdf, qf, logd = gb.plnorm, gb.qlnorm, gb.dlnorm
    return (
        lambda x: np.asarray(cdf(x, *args)),
        lambda p: np.asarray(qf(p, *args)),
        lambda x: np.asarray(logd(x, *args, log=True)),
    )


def innovation_map(variant, alpha):
    """L(x) and its inverse, with x = 1+e."""
    if variant == "adam":
        return lambda x: alpha * np.log(x), lambda v: np.exp(v / alpha)
    return (
        lambda x: np.log(1 - alpha + alpha * x),
        lambda v: (np.exp(v) - 1 + alpha) / alpha,
    )


def l_masses(dist, s, alpha, variant, h, eps):
    """The first bin centre and the masses of L in the bins of width h."""
    cdf, qf, _ = x_functions(dist, s)
    forward, inverse = innovation_map(variant, alpha)
    lower, upper = forward(qf(np.array([eps, 1 - eps])))
    edges = lower + h * np.arange(int(np.ceil((upper - lower) / h)) + 1)
    masses = np.clip(np.diff(cdf(inverse(edges))), 0, None)
    return lower + h / 2, masses


def u_density(dist, s, h, eps):
    """The first grid point and the density of u = log(1+e) on the grid of step h."""
    _, qf, logd = x_functions(dist, s)
    lower, upper = np.log(qf(np.array([eps, 1 - eps])))
    grid = lower + h * np.arange(int(np.ceil((upper - lower) / h)) + 1)
    return lower, np.exp(logd(np.exp(grid)) + grid)


def trim(start, masses, h, tol):
    """Drop the ends of a mass vector whose cumulative mass is below tol."""
    first = np.searchsorted(np.cumsum(masses), tol)
    last = len(masses) - np.searchsorted(np.cumsum(masses[::-1]), tol)
    return start + first * h, masses[first:last]


def exact_moments(dist, s, alpha, variant):
    """E L, Var L and E u by adaptive quadrature on the density of 1+e."""
    _, _, logd = x_functions(dist, s)
    forward, _ = innovation_map(variant, alpha)

    def expect(fun):
        def integrand(x):
            return fun(x) * np.exp(logd(np.array([x]))[0])

        parts = [(0, 1), (1, np.inf)]
        return sum(integrate.quad(integrand, a, b, limit=200)[0] for a, b in parts)

    mean = expect(forward)
    return mean, expect(lambda x: forward(x) ** 2) - mean**2, expect(np.log)


def one_step_log(dist, s, y):
    """Log-density of y = log z under the one-step density f_z(z | 1)."""
    _, _, logd = x_functions(dist, s)
    return logd(np.exp(y)) + y


def mixture_gh(dist, s, y, mean, variance, nodes_number):
    """Log-density of y under the mixture by this script's Gauss-Hermite rule:
    f_z(e^y | e^d) e^y = f_u(y - d) at d = m + sqrt(2w) x_q."""
    nodes, weights = np.polynomial.hermite.hermgauss(nodes_number)
    drift = mean + np.sqrt(2 * variance) * nodes
    logs = one_step_log(dist, s, y[:, None] - drift[None, :])
    return logsumexp(logs + np.log(weights / np.sqrt(np.pi))[None, :], axis=1)


def mixture_log(dist, s, y, mean, variance):
    """Log-density of y under the mixture eq:mixture: the package's
    gap_log_density for the Gamma and the inverse Gaussian, the closed form
    for the log-normal."""
    if dist == "dlnorm":
        total = s + variance
        return -0.5 * np.log(2 * np.pi * total) - (y + s / 2 - mean) ** 2 / (2 * total)
    ones = np.ones_like(y)
    return gap_log_density(np.exp(y), ones, s, dist, mean * ones, variance * ones) + y


def exact_density(d_start, d_mass, u_start, u_dens, h, threshold):
    """The density of y on its grid, over the points above the threshold."""
    density = np.clip(signal.fftconvolve(d_mass, u_dens), 0, None)
    y = d_start + u_start + h * np.arange(len(density))
    keep = density > threshold * density.max()
    return y[keep], density[keep]


def run_pair(task):
    """All the cells j of one (distribution, ETS variant, s, alpha)."""
    dist, variant, s, alpha, settings, js, split = task
    h = settings["h"]
    noise = gap_drift([alpha], np.zeros(0, bool), dist, variant == "adam", s)[2]
    mean_l, var_l = noise[0][1], noise[1][1, 1]
    exact_mean_l, exact_var_l, exact_mean_u = exact_moments(dist, s, alpha, variant)
    l_start, l_mass = l_masses(dist, s, alpha, variant, h, settings["eps"])
    u_start, u_dens = u_density(dist, s, h, settings["eps"])
    d_start, d_mass = l_start, l_mass
    rows = []
    for j in range(2, max(js) + 1):
        if j > 2:
            d_mass = np.clip(signal.fftconvolve(d_mass, l_mass), 0, None)
            d_start, d_mass = trim(d_start + l_start, d_mass, h, settings["trim"])
        if j not in js:
            continue
        y, f = exact_density(d_start, d_mass, u_start, u_dens, h, settings["threshold"])
        log_f = np.log(f)

        def kl(log_g):
            return h * np.sum(f * (log_f - log_g))

        m, w = (j - 1) * mean_l, (j - 1) * var_l
        row = dict(
            distribution=dist,
            ets=variant,
            s=s,
            alpha=alpha,
            j=j,
            kl_mixture=kl(mixture_log(dist, s, y, m, w)),
            kl_onestep=kl(one_step_log(dist, s, y)),
            kl_mixture_gh16=kl(mixture_gh(dist, s, y, m, w, 16)),
        )
        if split:
            exact_m, exact_w = (j - 1) * exact_mean_l, (j - 1) * exact_var_l
            row["kl_mixture_gh100"] = kl(mixture_gh(dist, s, y, m, w, 100))
            row["kl_normal_drift"] = kl(mixture_gh(dist, s, y, exact_m, exact_w, 100))
        mean_y = (j - 1) * exact_mean_l + exact_mean_u
        row.update(
            drift_mean=m,
            drift_variance=w,
            drift_mean_exact=(j - 1) * exact_mean_l,
            drift_variance_exact=(j - 1) * exact_var_l,
            mass=h * f.sum(),
            mean_error=abs(h * np.sum(y * f) - mean_y * h * f.sum()),
            grid_points=len(f),
        )
        rows.append(row)
    return rows


def run(tasks, workers):
    with ProcessPoolExecutor(workers) as pool:
        return pd.DataFrame([row for rows in pool.map(run_pair, tasks) for row in rows])


def provenance():
    def git(*args):
        command = ["git", "-C", str(REPO), *args]
        return subprocess.run(command, capture_output=True, text=True).stdout.strip()

    return dict(
        commit=git("rev-parse", "HEAD"),
        dirty=bool(git("status", "--porcelain")),
        smooth=smooth.__version__,
        greybox=gb.__version__,
        numpy=np.__version__,
        scipy=scipy.__version__,
        pandas=pd.__version__,
        python=sys.version.split()[0],
    )


def refinement_tasks(table):
    """The sub-grid and the cells of the maxima, by (dist, variant, s, alpha)."""
    cells = {
        (d, v, s, a, j)
        for d in DISTRIBUTIONS
        for v in VARIANTS
        for s in (0.05, 0.5, 1.0)
        for a in (0.1, 0.35, 0.6)
        for j in (2, 10, 20)
    }
    for (d, v), group in table.groupby(["distribution", "ets"]):
        for column in KLS:
            best = group.loc[group[column].idxmax()]
            cells.add((d, v, best.s, best.alpha, int(best.j)))
    pairs: dict = {}
    for d, v, s, a, j in cells:
        pairs.setdefault((d, v, float(s), float(a)), set()).add(int(j))
    return [(*key, FINE, tuple(sorted(j)), True) for key, j in pairs.items()]


def report(table, refine, workers, elapsed):
    print("\nMaximum KL(exact || approximation) per observation over the grid")
    for (d, v), group in table.groupby(["distribution", "ets"], sort=False):
        target_mix, target_one = TARGETS[d]
        mix = group.loc[group.kl_mixture.idxmax()]
        one = group.loc[group.kl_onestep.idxmax()]
        print(
            f"{d:9s} {v:12s} mixture {mix.kl_mixture:.5f} (target {target_mix}) "
            f"at s={mix.s}, alpha={mix.alpha}, j={mix.j}; "
            f"own GH16 {group.kl_mixture_gh16.max():.5f}\n{'':22s} one-step "
            f"{one.kl_onestep:.3f} (target {target_one}) at s={one.s}, "
            f"alpha={one.alpha}, j={one.j}"
        )
    pd.set_option("display.width", 200)
    for column, digits in (("kl_mixture", 5), ("kl_onestep", 3)):
        print(f"\nMaximum {column} over s and alpha, by j")
        pivot = table.pivot_table(column, "j", ["distribution", "ets"], "max")
        print(pivot.round(digits).to_string())
    print("\nAccuracy checks over the main grid")
    print(f"  mass: min {table.mass.min():.12f}, max {table.mass.max():.12f}")
    print(f"  error of the mean of log z: max {table.mean_error.max():.2e}")
    print(f"  refinement, {len(refine)} cells (h={FINE['h']}, eps={FINE['eps']}):")
    for column in KLS:
        print(
            f"    {column}: max abs change {refine[column + '_absdiff'].max():.2e}, "
            f"max rel change {refine[column + '_reldiff'].max():.2e}"
        )
    print("\nSplit of the mixture error at the cells of the refinement run (max):")
    columns = ["kl_mixture_fine", "kl_mixture_gh100", "kl_normal_drift"]
    print(refine.groupby(["distribution", "ets"])[columns].max().to_string())
    print(f"\nrun time {elapsed:.0f}s with {workers} workers")


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    info = provenance()
    print(" ".join(f"{k}={v}" for k, v in info.items()), f"seed={SEED}")
    started = time.time()
    js = tuple(int(j) for j in J_GRID)
    tasks = [
        (d, v, float(s), float(a), MAIN, js, False)
        for d in DISTRIBUTIONS
        for v in VARIANTS
        for s in S_GRID
        for a in ALPHA_GRID
    ]
    table = run(tasks, workers)
    print(f"main grid: {len(table)} cells in {time.time() - started:.0f}s")

    fine = run(refinement_tasks(table), workers)
    keys = ["distribution", "ets", "s", "alpha", "j"]
    columns = keys + KLS + ["kl_mixture_gh100", "kl_normal_drift", "mass", "mean_error"]
    refine = fine[columns].merge(table[keys + KLS], on=keys, suffixes=("_fine", ""))
    for column in KLS:
        diff = (refine[column + "_fine"] - refine[column]).abs()
        refine[column + "_absdiff"] = diff
        refine[column + "_reldiff"] = diff / refine[column].abs()
    elapsed = time.time() - started

    for frame, settings in ((table, MAIN), (refine, FINE)):
        for key, value in {**settings, **info, "seed": SEED}.items():
            frame[key] = value
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    table.to_csv(out / "kl_mixture.csv", index=False)
    refine.to_csv(out / "kl_mixture_refine.csv", index=False)
    report(table, refine, workers, elapsed)


if __name__ == "__main__":
    main()
