"""Checks of the formulation of smooth's TBATS (H1, H2, H3 of the smooth-tbats study).

H1: with the same parameters and initial states, smooth's TBATS (no ARMA) gives the
    one-step fitted values of an independent implementation of De Livera et al.'s
    rotation-form recursion, in the Box-Cox space.
H2: backcasting through the harmonic block is exact on a noise-free series.
H3: the impulse response of ADAM's ARMA in smooth's matrices is alpha + phi^j, and
    differs from the ARMA errors of De Livera et al.

Run: python formulation_checks.py --seeds 1 2 [--n 300]
Writes its results to $CLAIMSET_RESULTS (JSON) when set.
"""

import argparse
import itertools
import json
import os
import warnings

import numpy as np
from smooth import TBATS

PERIODS = {"24": [24.0], "7.3": [7.3], "24+168": [24.0, 168.0], "7.3+30.4": [7.3, 30.4]}
TRENDS = ["none", "additive", "damped"]
LAMBDAS = [1.0, 0.5, 0.0]


def box_cox(y, lam):
    return np.log(y) if lam == 0 else (y**lam - 1) / lam


def box_cox_inverse(z, lam):
    return np.exp(z) if lam == 0 else (lam * z + 1) ** (1 / lam)


def de_livera(z, periods, harmonics, trend, alpha, beta, phi, gammas, init):
    """One-step fitted values of De Livera et al.'s TBATS (rotation form, no ARMA)
    for the series z in the Box-Cox space, from the states at t=0: the level, the
    trend and, per harmonic and period, s*_0 (the first k values of the period's
    array) and s_0 (the last k)."""
    freq = np.concatenate(
        [2 * np.pi * np.arange(1, k + 1) / m for m, k in zip(periods, harmonics)]
    )
    g1 = np.concatenate([np.full(k, g[0]) for k, g in zip(harmonics, gammas)])
    g2 = np.concatenate([np.full(k, g[1]) for k, g in zip(harmonics, gammas)])
    coefficients = [np.asarray(c, float) for c in init["seasonal"]]
    s_star = np.concatenate([c[: k] for c, k in zip(coefficients, harmonics)])
    s = np.concatenate([c[k:] for c, k in zip(coefficients, harmonics)])
    level = init["level"]
    slope = init["trend"] if trend != "none" else 0.0
    damping = phi if trend == "damped" else 1.0
    cos, sin = np.cos(freq), np.sin(freq)
    fitted = np.empty(len(z))
    for t in range(len(z)):
        fitted[t] = level + damping * slope + s.sum()
        d = z[t] - fitted[t]
        level = level + damping * slope + alpha * d
        slope = damping * slope + beta * d
        s, s_star = cos * s + sin * s_star + g1 * d, -sin * s + cos * s_star + g2 * d
    return fitted


def smooth_coefficients(periods, harmonics, seasonal):
    """smooth's initial Fourier coefficients (sines, then cosines per period) from De
    Livera's states at t=0. smooth's coefficients define s(t) = a sin(lambda t) +
    b cos(lambda t) with s(1) entering y_1, while De Livera's s_0 enters y_1: so
    s_0 = a sin(lambda) + b cos(lambda) and s*_0 = a cos(lambda) - b sin(lambda), i.e.
    a = s_0 sin(lambda) + s*_0 cos(lambda), b = s_0 cos(lambda) - s*_0 sin(lambda)."""
    result = []
    for m, k, c in zip(periods, harmonics, seasonal):
        f = 2 * np.pi * np.arange(1, k + 1) / m
        s_star, s = c[:k], c[k:]
        a = s * np.sin(f) + s_star * np.cos(f)
        b = s * np.cos(f) - s_star * np.sin(f)
        result.append(np.concatenate([a, b]))
    return result


def draw_case(rng, periods, trend, lam, n):
    """Random parameters, initial states and a series simulated by De Livera's
    recursion (normal errors) in the Box-Cox space, returned in the data space."""
    harmonics = [int(rng.integers(1, 4)) for _ in periods]
    alpha = rng.uniform(0.05, 0.3)
    beta = rng.uniform(0.0, 0.05) * alpha if trend != "none" else 0.0
    phi = rng.uniform(0.85, 0.98) if trend == "damped" else 1.0
    gammas = [rng.uniform(-0.01, 0.01, 2) for _ in periods]
    level0 = box_cox(100.0, lam)
    amplitude = 0.03 * abs(level0) + 0.1
    init = {
        "level": level0,
        "trend": rng.uniform(-0.002, 0.002) * abs(level0) if trend != "none" else 0.0,
        "seasonal": [rng.normal(0, amplitude, 2 * k) for k in harmonics],
    }
    sigma = 0.01 * abs(level0) + 0.01
    # simulate: the recursion with normal errors, fitted value + error
    z = np.zeros(n)
    errors = rng.normal(0, sigma, n)
    state = dict(init)
    for t in range(n):
        z_t = de_livera(np.zeros(1), periods, harmonics, trend, alpha, beta, phi,
                        gammas, state)[0]
        z[t] = z_t + errors[t]
        state = _advance(state, periods, harmonics, trend, alpha, beta, phi, gammas,
                         errors[t])
    y = box_cox_inverse(z, lam)
    return y, harmonics, alpha, beta, phi, gammas, init


def _advance(state, periods, harmonics, trend, alpha, beta, phi, gammas, d):
    """The states at t+1 from those at t and the error d (De Livera's recursion)."""
    freq = [2 * np.pi * np.arange(1, k + 1) / m for m, k in zip(periods, harmonics)]
    damping = phi if trend == "damped" else 1.0
    slope = state["trend"]
    seasonal = []
    for f, c, g in zip(freq, state["seasonal"], gammas):
        k = len(f)
        s_star, s = c[:k], c[k:]
        s_new = np.cos(f) * s + np.sin(f) * s_star + g[0] * d
        s_star_new = -np.sin(f) * s + np.cos(f) * s_star + g[1] * d
        seasonal.append(np.concatenate([s_star_new, s_new]))
    return {
        "level": state["level"] + damping * slope + alpha * d,
        "trend": damping * slope + beta * d,
        "seasonal": seasonal,
    }


def check_h1(seeds, n):
    results, worst = [], 0.0
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for (label, periods), trend, lam in itertools.product(
            PERIODS.items(), TRENDS, LAMBDAS
        ):
            y, harmonics, alpha, beta, phi, gammas, init = draw_case(
                rng, periods, trend, lam, n
            )
            model = TBATS(
                lags=[1.0] + periods,
                harmonics=harmonics,
                trend=trend,
                lambda_bc=lam,
                orders={"ar": 0, "ma": 0, "select": False},
                persistence={"level": alpha, "trend": beta,
                             "seasonal": [list(g) for g in gammas]},
                phi=phi if trend == "damped" else None,
                initial={"level": init["level"], "trend": init["trend"],
                         "seasonal": smooth_coefficients(periods, harmonics,
                                                         init["seasonal"])},
                distribution="dnorm",
                bounds="none",
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(y)
            z = box_cox(y, lam)
            fitted_smooth = z - np.asarray(model.residuals, float)
            fitted_dl = de_livera(z, periods, harmonics, trend, alpha, beta, phi,
                                  gammas, init)
            diff = float(np.max(np.abs(fitted_smooth - fitted_dl)
                                / np.maximum(np.abs(fitted_dl), 1.0)))
            worst = max(worst, diff)
            results.append({
                "variable": "max relative difference of fitted values, smooth TBATS vs "
                            "De Livera recursion",
                "value": diff,
                "conditions": {"seed": seed, "periods": label, "trend": trend,
                               "lambda": lam, "harmonics": str(harmonics)},
            })
    results.append({
        "variable": "max relative difference of fitted values over all cases, smooth "
                    "TBATS vs De Livera recursion",
        "value": worst,
        "conditions": {"seeds": str(list(seeds)), "cases": len(results)},
    })
    return results


def check_h2(n):
    t = np.arange(1, n + 1)
    z = 50.0 + 2.0 * np.sin(2 * np.pi * t / 7.3) + 1.0 * np.cos(4 * np.pi * t / 7.3)
    z += 3.0 * np.sin(2 * np.pi * t / 24) - 1.5 * np.cos(2 * np.pi * t / 24)
    z += 0.5 * np.sin(6 * np.pi * t / 24)
    model = TBATS(
        lags=[1.0, 7.3, 24.0],
        harmonics=[2, 3],
        trend="none",
        lambda_bc=1.0,
        orders={"ar": 0, "ma": 0, "select": False},
        persistence={"level": 0.1, "seasonal": [[0.01, 0.005], [0.02, -0.01]]},
        initial="backcasting",
        distribution="dnorm",
        bounds="none",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(z)
    error = float(np.nanmax(np.abs(np.asarray(model.residuals, float))))
    return [{
        "variable": "max absolute in-sample error, noise-free harmonic series, "
                    "backcasting",
        "value": error,
        "conditions": {"periods": "7.3+24", "harmonics": "[2, 3]", "n": n},
    }]


def check_h3():
    results, worst_smooth, closest_errors = [], 0.0, np.inf
    y = 100 + np.random.default_rng(0).normal(0, 1, 200)
    for alpha, phi in itertools.product([0.1, 0.3, 0.7], [-0.5, 0.3, 0.8]):
        model = TBATS(
            lags=[1.0],
            trend="none",
            lambda_bc=1.0,
            orders={"ar": 1, "ma": 0, "select": False},
            persistence={"level": alpha},
            arma={"ar": [phi]},
            initial="backcasting",
            distribution="dnorm",
            bounds="none",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(y)
        F = np.asarray(model.transition, float)
        w = np.asarray(model.measurement, float)[-1]
        g = np.array(list(model.persistence_vector.values()), float)
        j = np.arange(1, 21)
        response = np.array([w @ np.linalg.matrix_power(F, k - 1) @ g for k in j])
        adam_formula = alpha + phi**j
        errors_formula = phi**j + alpha * (1 - phi**j) / (1 - phi)
        d_smooth = float(np.max(np.abs(response - adam_formula)))
        d_errors = float(np.max(np.abs(response - errors_formula)))
        worst_smooth = max(worst_smooth, d_smooth)
        closest_errors = min(closest_errors, d_errors)
        for name, value in (("alpha + phi^j", d_smooth),
                            ("ARMA-errors response", d_errors)):
            results.append({
                "variable": f"max absolute difference of the impulse response of smooth "
                            f"TBATS from {name}, lags 1-20",
                "value": value,
                "conditions": {"alpha": alpha, "phi": phi},
            })
    results += [
        {"variable": "max absolute difference of smooth's impulse response from "
                     "alpha + phi^j over all cases", "value": worst_smooth},
        {"variable": "min over cases of the max absolute difference of smooth's impulse "
                     "response from the ARMA-errors response", "value": closest_errors},
    ]
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--n", type=int, default=300)
    args = parser.parse_args()
    results = check_h1(args.seeds, args.n) + check_h2(args.n) + check_h3()
    for r in results:
        if "seed" not in r.get("conditions", {}):
            print(r["variable"], r.get("conditions", ""), r["value"])
    path = os.environ.get("CLAIMSET_RESULTS")
    if path:
        with open(path, "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    main()
