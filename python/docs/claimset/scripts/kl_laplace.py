"""KL of the scaled-family approximation: Laplace, S and Generalised Normal (E24).

Claim (INTERMITTENT_LIKELIHOOD.tex, Section 7 "Implementation", E24): "The pure
additive models with the Laplace, the S and the Generalised Normal distributions take
the variances of the filter, keeping the error in its family with the scale s f_t^{1/p}
(p such that the variance is proportional to s^p): the convolution of these kinked
densities with the normal uncertainty of the states is poorly integrated by the
Gauss-Hermite rule, and the error of the approximation is small (a Kullback-Leibler
divergence below 0.05 for the Laplace at f_t = 4)." Target: KL < 0.05 for the Laplace
at f_t = 4.

Definitions (Proposition "kalman", eq:kalman): f_t = w'P_t w + 1 and the predictive
variance of z_t is sigma^2 f_t, with sigma^2 the variance of the error and sigma^2 P_t
the covariance of the states. So the state uncertainty contributes the variance sigma^2
(f_t - 1) to the observation, and the size is z = mu + e + u with e in the family
(scale s, variance V(s) = sigma^2) and u ~ N(0, (f_t - 1) V(s)), independent. The
normality of u is the paper's premise (with a non-normal error the drift sum_i c_i e_i
is not exactly normal); it is kept here.

  exact          g(x)   = int f(x - u; s) phi(u; 0, (f_t - 1) V(s)) du
  approximation  h(x)   = f(x; s f_t^{1/p})   (variance V(s) f_t = Var g)
  Gauss-Hermite  h_Q(x) = sum_q w_q/sqrt(pi) f(x - sqrt(2 (f_t - 1) V(s)) x_q; s)
  one-step       h_1(x) = f(x; s)   (no correction, for context)

Families (greybox densities, loc 0): dlaplace (scale s, V = 2 s^2, p = 2); ds (density
exp(-sqrt|x|/s)/(4 s^2), V = 120 s^4, p = 4); dgnorm with shape beta in {0.5, 1, 1.5,
2, 3} (V = s^2 Gamma(3/beta)/Gamma(1/beta), p = 2). This is the parameterisation of the
package (python/src/smooth/adam_general/core/utils/utils.py, scale *
gap_variance^(1/p)).

KL direction: the main column is KL(exact || approximation) = int g log(g/h), as asked;
the reverse KL(h || g) is reported too, labelled. s = 1 throughout; the invariance in s
is checked at f_t = 4 with s = 0.5 and 2 (all densities are scale families and sqrt(V)
is proportional to the scale of the density in x, s for Laplace/dgnorm and s^2 for ds,
so the KL cannot depend on s).

Grid: f_t in geomspace(1.01, 100, 21) plus f_t = 4 (22 values). Gauss-Hermite: Q = 16
(the rule of the package, numpy hermgauss(16), checked equal to smooth's constants) for
every family; Q in {8, 16, 32, 64} for the Laplace.

Numerics (no randomness; the seed below is set for the record only):
- Laplace-normal convolution in closed form: with b the Laplace scale and sd sigma_u,
  g(x) = 1/(2b) exp(sigma_u^2/(2 b^2)) [e^{-x/b} Phi(x/sigma_u - sigma_u/b) + e^{x/b}
  Phi(-x/sigma_u - sigma_u/b)], computed in logs (scipy log_ndtr).
- Other convolutions (x >= 0): g(x) = int phi(z) f(x - sigma_u z) dz on z in
  [-10, max(10, k + 1)], k = x/sigma_u the kink, so that the kink is always inside
  (the far tail of g is driven by z near k when the family is lighter-tailed than the
  normal shifted by x); lost normal mass below 1.5e-23. The range is split at k and
  each piece substituted z = k -/+ w^2, which makes the integrand smooth (the densities
  are smooth in |x|^(1/2)), and integrated by composite Gauss-Legendre in w with
  IN_PANELS panels of 8 nodes per piece.
- All densities enter in logs (greybox log=True; the Gauss-Hermite mixture by
  logsumexp), so that a light-tailed approximation that underflows in the tails still
  gives a finite KL.
- KL and moments: g and h are symmetric, so int over x >= 0 times 2, on [0, X].
  Breakpoints at 0, X and (for the Gauss-Hermite mixture) its kinks sqrt(2) sigma_u
  x_q; each interval is split in halves and each half substituted x = a + w^2 (or b -
  w^2) at its end, composite Gauss-Legendre in w with OUT_PANELS panels of 8 nodes per
  half.
- Truncation X: the largest of the points where the two-sided tail of the approximation
  and of the error is 1e-14, the latter plus max(9, sqrt(2) max x_q) sigma_u (normal
  tail 2e-19).
- Refinement check: everything is computed at (OUT_PANELS, IN_PANELS) = (40, 60) and at
  twice that (80, 120); the reported values are the fine ones, |fine - coarse| is
  reported.
- Accuracy checks reported: mass and variance ratio Var/(V f_t) of g and h on the grid;
  the numerical Laplace convolution against the closed form (max relative error);
  dgnorm with beta = 1 against the closed-form Laplace (same density); dgnorm with beta
  = 2 (normal, where the approximation is exact, KL = 0, and the one-step KL is
  (f_t - 1 - log f_t)/2 in closed form). dgnorm with beta = 0.5 and ds are the same
  family (exp(-sqrt(|x|/s)) against exp(-sqrt|x|/s)), so their KLs coincide.

Versions: smooth 1.2.0 (python, editable from the working tree), greybox 1.0.9, numpy
2.4.6, scipy 1.17.1, Python 3.11.15. The commit (git rev-parse HEAD of the repository)
and its dirty state are printed at run time and written into the CSV.

Run from the repository root:
  python/.venv/bin/python python/docs/claimset/scripts/kl_laplace.py
It writes results/kl_laplace.csv next to this script.
"""

import subprocess
import time
from pathlib import Path

import greybox as gb
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import brentq
from scipy.special import gamma, gammaincc, log_ndtr, logsumexp

SEED = 20261015
RNG = np.random.default_rng(SEED)  # not used: the computation is deterministic
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
F_GRID = np.unique(np.r_[np.geomspace(1.01, 100, 21), 4.0])
SHAPES = [0.5, 1.0, 1.5, 2.0, 3.0]
Q_LAPLACE = [8, 16, 32, 64]
Q_OTHER = [16]
EPS_TAIL = 1e-14
Z_MAX = 10.0
NGL = 8
RESOLUTIONS = {"coarse": (40, 60), "fine": (80, 120)}
TARGET = 0.05


def family(name, shape=None):
    """Density f(x; scale), variance V(scale), power p and two-sided tail P(|e| > x)."""
    if name == "dlaplace":
        return dict(
            dens=lambda x, sc: gb.dlaplace(x, 0, sc),
            ldens=lambda x, sc: gb.dlaplace(x, 0, sc, log=True),
            var=lambda sc: 2 * sc**2,
            p=2,
            tail=lambda x, sc: np.exp(-x / sc),
        )
    if name == "ds":
        return dict(
            dens=lambda x, sc: gb.ds(x, 0, sc),
            ldens=lambda x, sc: gb.ds(x, 0, sc, log=True),
            var=lambda sc: 120 * sc**4,
            p=4,
            tail=lambda x, sc: (1 + np.sqrt(x) / sc) * np.exp(-np.sqrt(x) / sc),
        )
    return dict(
        dens=lambda x, sc: gb.dgnorm(x, 0, sc, shape),
        ldens=lambda x, sc: gb.dgnorm(x, 0, sc, shape, log=True),
        var=lambda sc: sc**2 * gamma(3 / shape) / gamma(1 / shape),
        p=2,
        tail=lambda x, sc: gammaincc(1 / shape, (x / sc) ** shape),
    )


def gl01(panels):
    """Composite Gauss-Legendre nodes and weights on [0, 1]."""
    xi, wi = np.polynomial.legendre.leggauss(NGL)
    left = np.arange(panels)[:, None] / panels
    return (left + (xi + 1) / (2 * panels)).ravel(), np.tile(wi / (2 * panels), panels)


def outer_grid(breaks, panels):
    """Nodes, weights on [breaks[0], breaks[-1]], halves substituted at their ends."""
    u, wu = gl01(panels)
    xs, ws = [], []
    for a, b in zip(breaks[:-1], breaks[1:]):
        m = (a + b) / 2
        w, ww = u * np.sqrt(m - a), wu * np.sqrt(m - a)
        xs += [a + w**2, b - w**2]
        ws += [2 * w * ww, 2 * w * ww]
    return np.concatenate(xs), np.concatenate(ws)


def conv_numeric(x, fam, scale, sd, panels):
    """log int phi(z) f(x - sd z) dz for x >= 0, z on [-Z, max(Z, k + 1)], split at
    the kink k = x/sd and substituted z = k -/+ w^2 on each piece."""
    u, wu = gl01(panels)
    k = (x / sd)[:, None]
    total = np.zeros(len(x))
    for sign, length in ((-1, k + Z_MAX), (1, np.maximum(Z_MAX - k, 1.0))):
        w = np.sqrt(length) * u[None, :]
        z = k + sign * w**2
        dens = fam["dens"]((x[:, None] - sd * z).ravel(), scale).reshape(z.shape)
        phi = np.exp(-(z**2) / 2) / np.sqrt(2 * np.pi)
        total += np.sum(phi * dens * 2 * w * wu[None, :] * np.sqrt(length), axis=1)
    return np.log(total)


def conv_laplace(x, b, sd):
    """Closed-form Laplace(0, b) convolved with N(0, sd^2), in logs."""
    c = sd**2 / (2 * b**2) - np.log(2 * b)
    t1 = c - x / b + log_ndtr(x / sd - sd / b)
    t2 = c + x / b + log_ndtr(-x / sd - sd / b)
    return np.logaddexp(t1, t2)


def kl_pair(w, lg, lh):
    """KL(g||h) and KL(h||g) on x >= 0 of symmetric densities (times 2), from logs."""
    return 2 * np.sum(w * np.exp(lg) * (lg - lh)), 2 * np.sum(
        w * np.exp(lh) * (lh - lg)
    )


def truncation(fam, scale, ft, sd, qmax):
    """X where the tails of the approximation and of the exact density are below EPS."""

    def root(sc):
        with np.errstate(divide="ignore"):
            return brentq(
                lambda x: np.log(fam["tail"](x, sc)) - np.log(EPS_TAIL), 1e-12, 1e8
            )

    x_h = root(scale * ft ** (1 / fam["p"]))
    x_e = root(scale) + max(9.0, np.sqrt(2) * qmax) * sd
    return max(x_h, x_e)


def run_case(name, shape, scale, ft, panels_out, panels_in, qs):
    """All approximations at one (family, shape, scale, f_t) and resolution."""
    fam = family(name, shape)
    sd = np.sqrt((ft - 1) * fam["var"](scale))
    nodes = {q: np.polynomial.hermite.hermgauss(q) for q in qs}
    x_max = truncation(fam, scale, ft, sd, max(nodes[q][0].max() for q in qs))

    def exact(x):
        if name == "dlaplace":
            return conv_laplace(x, scale, sd)
        return conv_numeric(x, fam, scale, sd, panels_in)

    var_target = fam["var"](scale) * ft
    rows, x0, w0 = [], *outer_grid(np.array([0.0, x_max]), panels_out)
    g0 = exact(x0)
    approx = {
        "family_scaled": fam["ldens"](x0, scale * ft ** (1 / fam["p"])),
        "one_step": fam["ldens"](x0, scale),
    }
    common = dict(
        distribution=name, shape=shape, scale=scale, f_t=ft, truncation_X=x_max
    )
    for label, h in approx.items():
        rows.append(
            dict(
                common,
                approximation=label,
                Q=np.nan,
                **measures(w0, x0, g0, h, var_target),
            )
        )
    if name == "dlaplace":
        num = conv_numeric(x0, fam, scale, sd, panels_in)
        rel = np.abs(np.expm1(num - g0))
        rows[0]["laplace_numeric_vs_closed_relerr"] = np.max(rel)
    for q, (xq, wq) in nodes.items():
        cusps = np.sqrt(2) * sd * xq[xq > 0]
        x1, w1 = outer_grid(np.unique(np.r_[0.0, cusps, x_max]), panels_out)
        g1 = exact(x1)
        h1 = logsumexp(
            [fam["ldens"](x1 - np.sqrt(2) * sd * xi, scale) for xi in xq],
            b=(wq / np.sqrt(np.pi))[:, None],
            axis=0,
        )
        rows.append(
            dict(
                common,
                approximation="gauss_hermite",
                Q=q,
                **measures(w1, x1, g1, h1, var_target),
            )
        )
    return rows


def measures(w, x, lg, lh, var_target):
    kl_gh, kl_hg = kl_pair(w, lg, lh)
    g, h = np.exp(lg), np.exp(lh)
    return dict(
        kl_exact_approx=kl_gh,
        kl_approx_exact=kl_hg,
        mass_exact=2 * np.sum(w * g),
        mass_approx=2 * np.sum(w * h),
        var_ratio_exact=2 * np.sum(w * x**2 * g) / var_target,
        var_ratio_approx=2 * np.sum(w * x**2 * h) / var_target,
    )


def cases():
    out = [("dlaplace", None, 1.0, f, Q_LAPLACE) for f in F_GRID]
    out += [("dlaplace", None, s, 4.0, Q_LAPLACE) for s in (0.5, 2.0)]
    for name, shape in [("ds", None)] + [("dgnorm", b) for b in SHAPES]:
        out += [(name, shape, 1.0, f, Q_OTHER) for f in F_GRID]
        out += [(name, shape, s, 4.0, Q_OTHER) for s in (0.5, 2.0)]
    return out


def git_state():
    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    dirty = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain"],
        capture_output=True,
        text=True,
    )
    return head.stdout.strip(), bool(dirty.stdout.strip())


def main():
    start = time.time()
    commit, dirty = git_state()
    print(f"commit {commit} dirty={dirty}; greybox {gb.__version__}, ", end="")
    print(f"numpy {np.__version__}, scipy {scipy.__version__}")
    tables = {}
    for res, (po, pi) in RESOLUTIONS.items():
        rows = [r for c in cases() for r in run_case(*c[:4], po, pi, c[4])]
        tables[res] = pd.DataFrame(rows)
    fine, coarse = tables["fine"], tables["coarse"]
    for col in ("kl_exact_approx", "kl_approx_exact"):
        fine[col + "_refine_diff"] = np.abs(
            fine[col].to_numpy() - coarse[col].to_numpy()
        )
    fine["shape"] = fine["shape"].astype(float)
    fine = fine.assign(
        commit=commit, dirty=dirty, seed=SEED, ngl=NGL, z_max=Z_MAX, eps_tail=EPS_TAIL
    )
    fine = fine.assign(
        out_panels=RESOLUTIONS["fine"][0], in_panels=RESOLUTIONS["fine"][1]
    )
    (HERE / "results").mkdir(exist_ok=True)
    fine.to_csv(HERE / "results" / "kl_laplace.csv", index=False)
    report(fine, start)


def _shape_label(shape):
    return "" if np.isnan(shape) else f"({shape:g})"


def report(df, start):
    pd.set_option("display.width", 200, "display.max_rows", 500)
    df = df.assign(family=df.distribution + df["shape"].map(_shape_label))
    base = df[df.scale == 1.0]
    wide = base.pivot_table(
        index=["family", "f_t"],
        columns=["approximation", "Q"],
        values="kl_exact_approx",
        dropna=False,
    )
    print("\nKL(exact || approximation), s = 1 (rows: f_t; Q only for gauss_hermite)")
    print(wide.dropna(axis=1, how="all").to_string(float_format=lambda v: f"{v:.4g}"))
    scaled = df[df.approximation == "family_scaled"]
    print("\nfamily_scaled, both directions:")
    cols = [
        "family",
        "scale",
        "f_t",
        "kl_exact_approx",
        "kl_approx_exact",
    ]
    print(scaled[cols].to_string(index=False, float_format=lambda v: f"{v:.5g}"))
    lap = scaled[(scaled.distribution == "dlaplace") & (scaled.f_t == 4.0)]
    print(
        "\nHEADLINE Laplace f_t = 4: KL(exact||approx) =",
        ", ".join(f"{v:.5f} (s={s})" for s, v in zip(lap.scale, lap.kl_exact_approx)),
        f"| target < {TARGET}:",
        "reproduced" if (lap.kl_exact_approx < TARGET).all() else "NOT reproduced",
    )
    print(
        "  reverse KL(approx||exact) =",
        ", ".join(f"{v:.5f}" for v in lap.kl_approx_exact),
    )
    print("\nChecks (all cases):")
    print(f"  max |mass_exact - 1|  = {np.max(np.abs(df.mass_exact - 1)):.2e}")
    print(f"  max |mass_approx - 1| = {np.max(np.abs(df.mass_approx - 1)):.2e}")
    print(f"  max |var_ratio_exact - 1| = {np.max(np.abs(df.var_ratio_exact - 1)):.2e}")
    print(
        f"  max |var_ratio_approx - 1| (family_scaled) = "
        f"{np.max(np.abs(scaled.var_ratio_approx - 1)):.2e}"
    )
    print(
        f"  max refinement diff KL(e||a) = {df.kl_exact_approx_refine_diff.max():.2e}, "
        f"KL(a||e) = {df.kl_approx_exact_refine_diff.max():.2e}"
    )
    print(
        f"  Laplace numeric vs closed-form convolution, max rel. error = "
        f"{df.laplace_numeric_vs_closed_relerr.max():.2e}"
    )
    a = (
        scaled[(scaled.distribution == "dlaplace")]
        .set_index(["scale", "f_t"])
        .kl_exact_approx
    )
    b = (
        scaled[(scaled.distribution == "dgnorm") & (scaled["shape"] == 1.0)]
        .set_index(["scale", "f_t"])
        .kl_exact_approx
    )
    print(
        f"  dgnorm(beta=1) vs closed-form Laplace, max |dKL| = {np.max(abs(a - b)):.2e}"
    )
    n = scaled[
        (scaled.distribution == "dgnorm") & (scaled["shape"] == 2.0)
    ].kl_exact_approx
    print(f"  dgnorm(beta=2) (normal, exact): max |KL| = {np.max(np.abs(n)):.2e}")
    one = df[(df.family == "dgnorm(2)") & (df.approximation == "one_step")]
    err = one.kl_exact_approx - (one.f_t - 1 - np.log(one.f_t)) / 2
    print(f"  dgnorm(2) one-step vs (f-1-log f)/2: max |diff| = {err.abs().max():.2e}")
    s4 = scaled[scaled.f_t == 4.0].groupby("family").kl_exact_approx
    print(f"  invariance in s at f_t=4: max spread = {np.max(s4.max() - s4.min()):.2e}")
    print(f"\nrun time {time.time() - start:.1f} s")


if __name__ == "__main__":
    main()
