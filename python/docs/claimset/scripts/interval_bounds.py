"""Approximate versus simulated upper bound of the 90% interval of ETS(M,N,N).

Claim (INTERMITTENT_LIKELIHOOD.tex, Section 6 "Prediction intervals", E22): "The
quantiles drift down in the same way, which the approximate intervals of the
conventional ETS ignore: with s = 0.32 and alpha = 0.21 the upper bound of the 90%
interval twelve steps ahead was 0.45 against 0.32 from the simulated paths, and 0.59
against 0.30 after six unobserved periods at the end of the sample." Targets: 0.45 vs
0.32 (h = 12), 0.59 vs 0.30 (h = 12 after six unobserved periods, i.e. 18 steps from the
last observed period, Section 6).

Model: ETS(M,N,N), z_t = l_{t-1} (1 + e_t), E(1 + e_t) = 1, alpha = 0.21, with the scale
s = 0.32 of the package for each distribution:
  dgamma    1 + e ~ Gamma(shape 1/s, scale s), Var(1 + e) = s;
  dinvgauss 1 + e ~ IG(mean 1, dispersion s),  Var(1 + e) = s;
  dlnorm    log(1 + e) ~ N(-s/2, s),           Var(1 + e) = exp(s) - 1 = 0.377
(python/src/smooth/adam_general/core/utils/distributions.py, generate_errors). The
paragraph gives the Gamma mean formula, so the Gamma is the main case; the other two are
reported, labelled.
Two forms of the level update (a choice the paper leaves open; both computed):
  "adam"         l_t = l_{t-1} (1 + e_t)^alpha  (ETS of ADAM, ets="adam");
  "conventional" l_t = l_{t-1} (1 + alpha e_t)  (ets="conventional").
The point forecast mu_hat is the forecast with zero errors (the skeleton), l_T, or the
level after the last observed period with the six unobserved periods at the end.

The approximate interval (the package's, generate_prediction_interval in
python/src/smooth/adam_general/core/forecaster/intervals.py, with var_anal in
utils/var_covar.py), for both forms: V_h = (1 + v)(1 + alpha^2 v)^(h-1) - 1 with
v = Var(1 + e) (v = exp(s) - 1 for dlnorm), the variance of z_{T+h}/mu_hat of the
conventional ETS; the bound is mu_hat q_0.95(V_h) with q the quantile of the
distribution with mean one and variance V_h: greybox qgamma(p, shape 1/V, scale V),
qinvgauss(p, 1, V), qlnorm(p, -log(1+V)/2, sqrt(log(1+V))). Exception, labelled: the
package's approximate interval of the ETS of ADAM with dlnorm is exact in logs,
log(z/mu_hat) ~ N(-s a_h/2, s k_h), a_h = 1 + (h-1) alpha, k_h = 1 + (h-1) alpha^2.
The script computes both the package's interval and this independent implementation.

The unobserved periods at the end are missing values (NaN): the last K = 6 of the
T = 100 observations. The package then forecasts the pure multiplicative ETS from the
last observed period as h + K steps ahead and keeps the last h (forecaster.py, 6b), so
the bound 12 steps after the end is the 18-step one from the last observed period. (The
zeros of an occurrence model would also scale the levels by the probability; the claim
is about the sizes, so the missing values are used.)

The scale: the package has no argument that fixes it, it is the moment estimate from the
in-sample errors. The script fixes it at s by replacing, in the namespace of the
preparator module, its two calls of the scale estimator (_calculate_scale_parameter and
scaler, the latter used with the drift after unobserved periods) by s; nothing in the
package files is changed. The check: model.scale == s after predict, and the package's
1-step bound equals mu_hat q_0.95(s). The fitted model has every other parameter fixed:
model="MNN", persistence=0.21, initial={"level": 10}, so the degrees of freedom equal
the observed periods and the scale is not de-biased (checked: _df_scale = observed).
The in-sample data (T = 100) are simulated from the model with the same form and
distribution; they matter only through mu_hat.

Simulated bounds:
- package: predict(interval="simulated", nsim=NSIM_PKG, seed), the empirical quantile;
- own: N_OWN independent paths of z_h/mu_hat from the level 1, h = 1..18, the empirical
  quantile at 0.95 at h = 12 and h = 18.
MC standard error of a quantile: sqrt(p (1-p) / N) / f(q), with the density at the
quantile estimated from the own paths by 2 d / (Q(p + d) - Q(p - d)), d = 0.005.
The mean of z/mu_hat is reported for context: for the ETS of ADAM with the Gamma the
paper gives E = [Gamma(1/s + alpha) s^alpha / Gamma(1/s)]^(h-1).

Scale of the paper's numbers: the bounds are computed on the scale of z/mu_hat (the
bound divided by the point forecast) and on the scale of the data with the realised
mu_hat; log(z/mu_hat) and z/mu_hat - 1 are written to the CSV too. The ratio of the
approximate to the simulated bound does not depend on the scale and is compared with the
paper's 0.45/0.32 = 1.41 and 0.59/0.30 = 1.97. The lower bounds (the 0.05 quantile, the
only bounds below one on the scale of z/mu_hat) are written to the CSV as well.

Settings: S = 0.32, ALPHA = 0.21, LEVEL = 0.9 (two-sided: the upper bound is the 0.95
quantile), H = 12, K = 6, T = 100, initial level 10, N_OWN = 1,000,000,
NSIM_PKG = 1,000,000, SEED = 20261015 (numpy default_rng; the package simulations take
seed = SEED + case index).
Versions: smooth 1.2.0 (python, editable from the working tree), greybox 1.0.9, numpy
2.4.6, scipy 1.17.1, Python 3.11.15. The commit (git rev-parse HEAD of the repository)
and its dirty state are printed at run time and written into the CSV.

Run: /home/user/smooth/python/.venv/bin/python interval_bounds.py  (from any
directory); writes results/interval_bounds.csv next to this script."""

import importlib
import subprocess
import sys
import time
import warnings
from importlib.metadata import version
from pathlib import Path

import greybox as gb
import numpy as np
import pandas as pd
import scipy
from scipy.special import gammaln

from smooth import ADAM

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
S, ALPHA, LEVEL, H, K, T, L0 = 0.32, 0.21, 0.9, 12, 6, 100, 10.0
P = (1 + LEVEL) / 2
N_OWN, NSIM_PKG, SEED = 1_000_000, 1_000_000, 20261015
DISTRIBUTIONS = ("dgamma", "dinvgauss", "dlnorm")
FORMS = ("adam", "conventional")
TARGETS = {0: (0.45, 0.32), K: (0.59, 0.30)}


def draw(rng, distribution, size):
    """Draws of 1 + e with the mean one at the scale S."""
    if distribution == "dgamma":
        return rng.gamma(1 / S, S, size)
    if distribution == "dinvgauss":
        return rng.wald(1.0, 1 / S, size)
    return np.exp(rng.normal(-S / 2, np.sqrt(S), size))


def paths(rng, distribution, form, n, steps):
    """z_h / mu_hat for h = 1..steps from the level one, (n, steps)."""
    level = np.ones(n)
    ratio = np.empty((n, steps))
    for i in range(steps):
        one_e = draw(rng, distribution, n)
        ratio[:, i] = level * one_e
        level = level * (one_e**ALPHA if form == "adam" else 1 + ALPHA * (one_e - 1))
    return ratio


def variance_one(distribution):
    return np.exp(S) - 1 if distribution == "dlnorm" else S


def approximate(distribution, h, p=P):
    """The approximate bound of the conventional ETS on z/mu_hat (var_anal)."""
    v = variance_one(distribution)
    var_h = (1 + v) * (1 + ALPHA**2 * v) ** (h - 1) - 1
    if distribution == "dgamma":
        return float(gb.qgamma(p, shape=1 / var_h, scale=var_h))
    if distribution == "dinvgauss":
        return float(gb.qinvgauss(p, 1.0, var_h))
    sdlog = np.sqrt(np.log(1 + var_h))
    return float(gb.qlnorm(p, -(sdlog**2) / 2, sdlog))


def exact_lognormal_adam(h, p=P):
    """The exact bound of the ETS of ADAM with dlnorm on z/mu_hat."""
    meanlog = -S * (1 + (h - 1) * ALPHA) / 2
    return float(gb.qlnorm(p, meanlog, np.sqrt(S * (1 + (h - 1) * ALPHA**2))))


def quantile_se(sample, n):
    """MC standard error of the P quantile with n draws, the density from sample."""
    low, up = np.quantile(sample, [P - 0.005, P + 0.005])
    return np.sqrt(P * (1 - P) / n) * (up - low) / 0.01


def series(rng, distribution, form):
    """In-sample data from the model, with the last K missing for the gap case."""
    ratio = paths(rng, distribution, form, 1, T)[0]
    return L0 * ratio


def fixed_scale(*args, **kwargs):
    return S


def package_bounds(y, distribution, form, seed):
    """The package's point forecast, approximate and simulated upper bounds (h=1..H)."""
    model = ADAM(
        model="MNN",
        persistence=ALPHA,
        initial={"level": L0},
        distribution=distribution,
        ets=form,
    ).fit(y)
    out = {}
    for interval in ("approximate", "simulated"):
        forecast = model.predict(
            h=H, interval=interval, level=LEVEL, nsim=NSIM_PKG, seed=seed
        )
        out[interval] = np.asarray(forecast.upper, dtype=float).ravel()
        out[interval + "_lower"] = np.asarray(forecast.lower, dtype=float).ravel()
        out["mean"] = np.asarray(forecast.mean, dtype=float).ravel()
    observed = int(model._nobs_observed())
    out["checks"] = (model.scale, model._df_scale, observed)
    return out


def git_state():
    commit = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return commit, bool(status)


def main():
    start = time.time()
    commit, dirty = git_state()
    print(f"commit {commit} dirty={dirty}")
    print(
        ", ".join(f"{name} {version(name)}" for name in ("smooth", "greybox")) + ",",
        f"numpy {np.__version__}, scipy {scipy.__version__}, Python {sys.version}",
    )
    preparator = importlib.import_module(
        "smooth.adam_general.core.forecaster.preparator"
    )
    preparator._calculate_scale_parameter = fixed_scale
    preparator.scaler = fixed_scale
    warnings.simplefilter("ignore")
    rng = np.random.default_rng(SEED)

    rows = []
    case = 0
    for distribution in DISTRIBUTIONS:
        for form in FORMS:
            own = paths(rng, distribution, form, N_OWN, H + K)
            y = series(rng, distribution, form)
            for gap in (0, K):
                case += 1
                y_case = y.copy()
                y_case[T - gap :] = np.nan
                package = package_bounds(y_case, distribution, form, SEED + case)
                scale, df, observed = package["checks"]
                mu_hat = package["mean"][-1]
                steps = H + gap
                sample = own[:, steps - 1]
                approx_formula = approximate(distribution, steps)
                exact = (
                    exact_lognormal_adam(steps)
                    if distribution == "dlnorm" and form == "adam"
                    else np.nan
                )
                sim_own = float(np.quantile(sample, P))
                se_own = quantile_se(sample, N_OWN)
                approx_pkg = package["approximate"][-1] / mu_hat
                sim_pkg = package["simulated"][-1] / mu_hat
                one_step = package["approximate"][0] / package["mean"][0]
                one_step_ref = (
                    exact_lognormal_adam(1 + gap)
                    if distribution == "dlnorm" and form == "adam"
                    else approximate(distribution, 1 + gap)
                )
                mean_adam_gamma = (
                    np.exp(gammaln(1 / S + ALPHA) + ALPHA * np.log(S) - gammaln(1 / S))
                    ** (steps - 1)
                    if distribution == "dgamma" and form == "adam"
                    else np.nan
                )
                rows.append(
                    {
                        "distribution": distribution,
                        "ets": form,
                        "unobserved_at_end": gap,
                        "h": H,
                        "steps_from_last_observed": steps,
                        "s": S,
                        "alpha": ALPHA,
                        "level": LEVEL,
                        "quantile": P,
                        "n_own": N_OWN,
                        "nsim_package": NSIM_PKG,
                        "seed": SEED,
                        "package_scale": scale,
                        "package_df_scale": df,
                        "package_observed": observed,
                        "mu_hat": mu_hat,
                        "package_1step_bound_ratio": one_step,
                        "reference_1step_bound_ratio": one_step_ref,
                        "approx_package_ratio": approx_pkg,
                        "approx_formula_ratio": approx_formula,
                        "exact_lnorm_adam_ratio": exact,
                        "sim_package_ratio": sim_pkg,
                        "sim_package_se": se_own * np.sqrt(N_OWN / NSIM_PKG),
                        "sim_own_ratio": sim_own,
                        "sim_own_se": se_own,
                        "package_minus_own_z": (sim_pkg - sim_own)
                        / (se_own * np.sqrt(1 + N_OWN / NSIM_PKG)),
                        "approx_lower_package_ratio": package["approximate_lower"][-1]
                        / mu_hat,
                        "approx_lower_formula_ratio": approximate(
                            distribution, steps, 1 - P
                        ),
                        "sim_lower_package_ratio": package["simulated_lower"][-1]
                        / mu_hat,
                        "sim_lower_own_ratio": float(np.quantile(sample, 1 - P)),
                        "approx_over_sim": approx_pkg / sim_own,
                        "approx_package_data": package["approximate"][-1],
                        "sim_package_data": package["simulated"][-1],
                        "approx_log_ratio": np.log(approx_pkg),
                        "sim_own_log_ratio": np.log(sim_own),
                        "approx_relative": approx_pkg - 1,
                        "sim_own_relative": sim_own - 1,
                        "mean_own_ratio": float(np.mean(sample)),
                        "mean_formula_adam_gamma": mean_adam_gamma,
                        "variance_own_ratio": float(np.var(sample)),
                        "target_approx": TARGETS[gap][0],
                        "target_sim": TARGETS[gap][1],
                        "target_ratio": TARGETS[gap][0] / TARGETS[gap][1],
                        "commit": commit,
                        "dirty": dirty,
                    }
                )
            del own

    table = pd.DataFrame(rows)
    (HERE / "results").mkdir(exist_ok=True)
    table.to_csv(HERE / "results" / "interval_bounds.csv", index=False)

    show = table[
        [
            "distribution",
            "ets",
            "unobserved_at_end",
            "approx_package_ratio",
            "approx_formula_ratio",
            "exact_lnorm_adam_ratio",
            "sim_package_ratio",
            "sim_own_ratio",
            "sim_own_se",
            "approx_over_sim",
            "mu_hat",
            "approx_package_data",
            "sim_package_data",
        ]
    ]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print("\nUpper bound of the 90% interval, 12 steps after the end (z/mu_hat):")
        print(show.round(4).to_string(index=False))
        print("\nChecks (package scale, df, observed; 1-step bound vs reference):")
        checks = table[
            [
                "distribution",
                "ets",
                "unobserved_at_end",
                "package_scale",
                "package_df_scale",
                "package_observed",
                "package_1step_bound_ratio",
                "reference_1step_bound_ratio",
                "mean_own_ratio",
                "mean_formula_adam_gamma",
                "variance_own_ratio",
            ]
        ]
        print(checks.round(5).to_string(index=False))

    print("\nHeadline (Gamma, z/mu_hat), against the paper:")
    for form in FORMS:
        for gap in (0, K):
            row = table[
                (table.distribution == "dgamma")
                & (table.ets == form)
                & (table.unobserved_at_end == gap)
            ].iloc[0]
            print(
                f"  ets={form:12s} unobserved={gap}:"
                f" approx {row.approx_package_ratio:.3f}"
                f" vs simulated {row.sim_own_ratio:.3f} (se {row.sim_own_se:.4f}),"
                f" ratio {row.approx_over_sim:.3f};  paper {row.target_approx} vs"
                f" {row.target_sim}, ratio {row.target_ratio:.3f}"
            )
    print(f"\nrun time {time.time() - start:.1f} s")


if __name__ == "__main__":
    main()
