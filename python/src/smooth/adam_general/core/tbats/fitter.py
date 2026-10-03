"""The parameters and the fit of a TBATS structure (R/adam-tbats.R: tbats_B,
tbats_filler, tbats_gammaStart, tbats_eigens and tbats_fit)."""

import math
import re
from typing import Any, Callable, Dict, List, Optional

import nlopt
import numpy as np
from numpy.typing import NDArray

from smooth.adam_general import _adamCore, _ols  # type: ignore[attr-defined]
from smooth.adam_general._numDeriv import hessian as _hessian_cpp
from smooth.adam_general.core.creator.architector import (
    adam_head_length,
    adam_profile_creator,
)
from smooth.adam_general.core.tbats import structure as st
from smooth.adam_general.core.utils.utils import _sum_r, calculate_likelihood

PENALTY = 1e100
MULTISTEP_LOSSES = ("MSEh", "TMSE", "GTMSE", "MSCE", "GPL")


def _periods_used(struct: Dict[str, Any]) -> List[float]:
    """The periods with at least one harmonic, in the order of the periods."""
    used = set(struct["table"]["period"].tolist())
    return [p for p in struct["periods"] if p in used]


def parameters(
    struct: Dict[str, Any],
    spec: Dict[str, Any],
    arma_start: NDArray,
    lam_spec: Dict[str, Any],
    lam_start: float,
    distribution: str,
    other_estimate: bool,
    initial_estimate: bool,
    bounds: str,
) -> Dict[str, Any]:
    """The names, starting values and bounds of the parameter vector."""
    names: List[str] = ["alpha"]
    values: List[float] = [0.1]
    if struct["trend_in"]:
        names.append("beta")
        values.append(0.05)
    if struct["damped"]:
        names.append("phi")
        values.append(0.95)
    # No seasonal smoothing: on the boundary of the admissible region
    for period in _periods_used(struct):
        label = st._period_label(period)
        names += [f"gamma1[{label}]", f"gamma2[{label}]"]
        values += [0.0, 0.0]
    names += spec["names"]
    values += list(arma_start)
    if initial_estimate:
        labels = st.harmonic_labels(struct["table"])
        names.append("level")
        values.append(0.0)
        if struct["trend_in"]:
            names.append("trend")
            values.append(0.0)
        names += [f"sin{label}" for label in labels] + [
            f"cos{label}" for label in labels
        ]
        values += [0.0] * (2 * len(labels))
        names += [f"ARMAState{k}" for k in range(1, struct["arma_lag_max"] + 1)]
        values += [0.0] * struct["arma_lag_max"]
    if lam_spec["estimate"]:
        names.append("lambda")
        values.append(lam_start)
    if distribution == "dgnorm" and other_estimate:
        names.append("shape")
        values.append(2.0)
    lb = np.full(len(names), -np.inf)
    ub = np.full(len(names), np.inf)
    for i, name in enumerate(names):
        if bounds == "usual" and name in ("alpha", "beta", "phi"):
            lb[i], ub[i] = 0.0, 1.0
        if name == "lambda":
            lb[i], ub[i] = 0.0, 1.0
        if name == "shape":
            lb[i] = 0.0
    return {"names": names, "B": np.asarray(values, dtype=float), "lb": lb, "ub": ub}


def eigens(
    mat_f: NDArray, vec_g: NDArray, w: NDArray, struct: Dict[str, Any]
) -> NDArray:
    """The moduli of the eigenvalues of the discount matrix of the level, trend and
    harmonics, on the lag-expanded form: a harmonic is (v1_t, v2_t, v2_{t-1})."""
    n_ets = struct["n_ets"]
    n_h = struct["n_harmonics"]
    k = n_ets + 3 * n_h
    fe = np.zeros((k, k))
    ge = np.zeros(k)
    we = np.zeros(k)
    fe[:n_ets, :n_ets] = mat_f[:n_ets, :n_ets]
    ge[:n_ets] = vec_g[:n_ets]
    we[:n_ets] = w[:n_ets]
    for i in range(n_h):
        old = struct["harmonic_rows"][i]
        rows = n_ets + 3 * i + np.arange(3)
        eta = mat_f[old : old + 2, old]
        fe[np.ix_(rows, rows)] = [[eta[0], 0, eta[0]], [eta[1], 0, eta[1]], [0, 1, 0]]
        ge[rows[:2]] = vec_g[old : old + 2]
        we[rows] = [1, 0, 1]
    return np.abs(np.linalg.eigvals(fe - np.outer(ge, we)))


class Filler:
    """The elements of the model for a parameter vector: the matrices, the initial
    deviations, lambda, the shape and the penalty of the bounds."""

    def __init__(
        self,
        names: List[str],
        struct: Dict[str, Any],
        spec: Dict[str, Any],
        lam_spec: Dict[str, Any],
        other: Optional[float],
        initial_estimate: bool,
        adam_cpp: Any,
    ):
        self.index = {name: i for i, name in enumerate(names)}
        self.struct = struct
        self.spec = spec
        self.lam_spec = lam_spec
        self.other = other
        self.initial_estimate = initial_estimate
        self.adam_cpp = adam_cpp
        used = _periods_used(struct)
        gamma1 = [self.index[f"gamma1[{st._period_label(p)}]"] for p in used]
        gamma2 = [self.index[f"gamma2[{st._period_label(p)}]"] for p in used]
        position = [used.index(p) for p in struct["table"]["period"]]
        self.gamma1 = np.asarray([gamma1[k] for k in position], dtype=int)
        self.gamma2 = np.asarray([gamma2[k] for k in position], dtype=int)
        self.arma = np.asarray([self.index[n] for n in spec["names"]], dtype=int)
        labels = st.harmonic_labels(struct["table"])
        if initial_estimate:
            self.sin = np.asarray([self.index[f"sin{x}"] for x in labels], dtype=int)
            self.cos = np.asarray([self.index[f"cos{x}"] for x in labels], dtype=int)
            self.arma_states = np.asarray(
                [
                    self.index[f"ARMAState{k}"]
                    for k in range(1, struct["arma_lag_max"] + 1)
                ],
                dtype=int,
            )

    def _get(self, B: NDArray, name: str, default: Any) -> Any:
        return B[self.index[name]] if name in self.index else default

    def __call__(self, B: NDArray, bounds: str) -> Dict[str, Any]:
        struct = self.struct
        alpha = B[self.index["alpha"]]
        beta = self._get(B, "beta", 0.0)
        phi = self._get(B, "phi", 1.0)
        gamma1 = B[self.gamma1]
        gamma2 = B[self.gamma2]
        lam = self._get(B, "lambda", self.lam_spec["value"])
        shape = self._get(B, "shape", 2.0 if self.other is None else self.other)
        penalty = 0.0

        mat_f = struct["mat_f"].copy()
        vec_g = np.zeros(struct["n_components"])
        w = np.ones(struct["n_components"])
        vec_g[0] = alpha
        if struct["trend_in"]:
            mat_f[0:2, 1] = phi
            w[1] = phi
            vec_g[1] = beta
        if struct["n_harmonics"] > 0:
            frequency = struct["table"]["frequency"]
            rows = struct["harmonic_rows"]
            vec_g[rows] = gamma1
            vec_g[rows + 1] = np.sin(frequency) * gamma2 - np.cos(frequency) * gamma1
        if struct["n_arma"] > 0:
            spec = self.spec
            polynomials = self.adam_cpp.polynomialise(
                B[self.arma],
                spec["ar_orders"].astype(np.uint64),
                np.zeros(len(spec["lags"]), dtype=np.uint64),
                spec["ma_orders"].astype(np.uint64),
                True,
                True,
                np.zeros(0),
                spec["lags"].astype(np.uint64),
            )
            lags = np.asarray(spec["state_lags"], dtype=int)
            ari = np.ravel(polynomials.ariPolynomial)
            ma_poly = np.ravel(polynomials.maPolynomial)
            ar = np.where(lags < len(ari), -ari[np.minimum(lags, len(ari) - 1)], 0.0)
            ma = np.where(
                lags < len(ma_poly), ma_poly[np.minimum(lags, len(ma_poly) - 1)], 0.0
            )
            ar = np.where(np.isfinite(ar), ar, 0.0)
            ma = np.where(np.isfinite(ma), ma, 0.0)
            rows = struct["arma_rows"]
            mat_f[np.ix_(rows, rows)] = np.outer(ar, np.ones(struct["n_arma"]))
            vec_g[rows] = ar + ma
            if bounds != "none":
                reflection = max(
                    float(polynomials.arReflection) * (spec["ar_orders"].sum() > 0),
                    float(polynomials.maReflection) * (spec["ma_orders"].sum() > 0),
                )
                if reflection >= 1:
                    penalty += PENALTY * reflection

        if lam < 0 or lam > 1 or shape <= 0:
            penalty += PENALTY
        if bounds == "usual":
            response = np.atleast_1d(alpha)
            if struct["n_harmonics"] > 0:
                response = (
                    alpha
                    + struct["response_cos"] @ gamma1
                    + struct["response_sin"] @ gamma2
                )
            if (
                min(alpha, phi) < 0
                or max(alpha, phi) > 1
                or beta < 0
                or beta > alpha
                or np.any(response < 0)
                or np.any(response > 1)
            ):
                penalty += PENALTY
        elif bounds == "admissible":
            values = eigens(mat_f, vec_g, w, struct)
            if np.any(values > 1 + 1e-10):
                penalty += PENALTY * values.max()

        deviations = None
        if self.initial_estimate:
            deviations = {
                "level": B[self.index["level"]],
                "trend": self._get(B, "trend", 0.0),
                "sin": B[self.sin],
                "cos": B[self.cos],
                "arma": B[self.arma_states],
            }
        return {
            "mat_f": mat_f,
            "vec_g": vec_g,
            "w": w,
            "phi": phi,
            "lambda": float(lam),
            "shape": shape,
            "deviations": deviations,
            "penalty": penalty,
        }


def gamma_start(B: NDArray, names: List[str], filler: Filler) -> NDArray:
    """The starting seasonal smoothing parameters for the admissible bounds: the
    point of a small grid, shared by the periods, that is the furthest inside."""
    is1 = np.array([n.startswith("gamma1") for n in names])
    is2 = np.array([n.startswith("gamma2") for n in names])
    grid = [-0.01, -0.001, 0.0, 0.001, 0.01]
    # R's expand.grid: gamma1 varies fastest
    points = [(g1, g2) for g2 in grid for g1 in grid]
    eigen_max = []
    for g1, g2 in points:
        test = B.copy()
        test[is1] = g1
        test[is2] = g2
        elements = filler(test, "none")
        eigen_max.append(
            eigens(
                elements["mat_f"], elements["vec_g"], elements["w"], filler.struct
            ).max()
        )
    best = int(np.argmin(eigen_max))
    if eigen_max[best] < 1:
        B = B.copy()
        B[is1], B[is2] = points[best]
    return B


def _optimise(
    cf: Callable[[NDArray], float],
    B: NDArray,
    lb: NDArray,
    ub: NDArray,
    s: Dict[str, Any],
) -> Dict[str, Any]:
    """NLopt with R's nloptr settings; the best point seen is returned."""
    best_x = [B.copy()]
    best_f = [math.inf]

    def objective(x: NDArray, grad: NDArray) -> float:
        value = cf(x)
        if value < best_f[0]:
            best_f[0], best_x[0] = value, x.copy()
        return value

    algorithm = getattr(
        nlopt, s["algorithm"].replace("NLOPT_", ""), nlopt.LN_NELDERMEAD
    )
    opt = nlopt.opt(algorithm, len(B))
    opt.set_min_objective(objective)
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)
    opt.set_maxeval(int(s["maxeval_used"]))
    opt.set_maxtime(s["maxtime"])
    opt.set_xtol_rel(s["xtol_rel"])
    opt.set_xtol_abs(s["xtol_abs"])
    opt.set_ftol_rel(s["ftol_rel"])
    opt.set_ftol_abs(s["ftol_abs"])
    try:
        opt.optimize(B.copy())
    except (nlopt.RoundoffLimited, ValueError, RuntimeError):
        pass
    return {"solution": best_x[0], "objective": best_f[0]}


def fit(
    y: NDArray,
    trend_type: str,
    table: Dict[str, NDArray],
    spec: Dict[str, Any],
    lam_spec: Dict[str, Any],
    distribution: str,
    initial: str,
    s: Dict[str, Any],
) -> Dict[str, Any]:
    """One fit of a fixed structure in the space of the Box-Cox transformed data."""
    obs = len(y)
    periods = sorted(set(table["period"].tolist()))
    struct = st.structure(trend_type, table, spec, periods)
    X = st.design(obs, struct["trend_in"], table)
    qr_x = st.QR(X)
    lam_start = st.lambda_start(y, X, lam_spec)
    y_bc_start = st.box_cox(y, lam_start)
    log_y = _sum_r(np.log(y)) if (lam_spec["estimate"] or lam_start != 1) else 0.0

    lags_all = struct["lags_model_all"]
    n_ets = struct["n_ets"]
    adam_cpp = _adamCore.adamCore(
        lags=np.asarray(lags_all, dtype=np.uint64),
        E="A",
        T="A" if struct["trend_in"] else "N",
        S="N",
        nNonSeasonal=n_ets,
        nSeasonal=0,
        nETS=n_ets,
        nArima=struct["n_components"] - n_ets,
        nXreg=0,
        nComponents=struct["n_components"],
        constant=False,
        adamETS=False,
    )
    head = adam_head_length(s["head_length"], struct["lags_model_max"], obs)
    adam_cpp.headLength = head["flag"]
    lookup = np.asfortranarray(
        adam_profile_creator(
            lags_all,
            struct["lags_model_max"],
            obs + max(s["h"], 1),
            head_length=head["geometry"],
        )["index_lookup_table"],
        dtype=np.uint64,
    )
    mat_vt = np.zeros((struct["n_components"], obs + head["geometry"]), order="F")
    ot = np.ones(obs)

    backcast = initial in ("backcasting", "complete")
    initial_estimate = initial in ("optimal", "two-stage")

    # The starting values of the ARMA from Hannan-Rissanen on the global residuals
    arma_start = np.zeros(0)
    if spec["n_param"] > 0:
        arma_start = np.asarray(
            _ols.arima_hr(
                qr_x.resid(y_bc_start),
                spec["ar_orders"].astype(np.uint64),
                spec["ma_orders"].astype(np.uint64),
                spec["lags"].astype(np.uint64),
                True,
                True,
                np.zeros(0),
                np.ones(len(spec["lags"]), dtype=np.uint64),
                s["bounds"] != "none",
            ),
            dtype=float,
        )

    other = s["shape"]
    other_estimate = distribution == "dgnorm" and s["shape_estimate"]
    b_list = parameters(
        struct,
        spec,
        arma_start,
        lam_spec,
        lam_start,
        distribution,
        other_estimate,
        initial_estimate,
        s["bounds"],
    )
    names = b_list["names"]
    filler = Filler(names, struct, spec, lam_spec, other, initial_estimate, adam_cpp)
    if s["bounds"] == "admissible" and struct["n_harmonics"] > 0:
        b_list["B"] = gamma_start(b_list["B"], names, filler)

    # The cost function
    def fit_states(elements: Dict[str, Any]) -> Dict[str, Any]:
        y_bc = st.box_cox(y, elements["lambda"]) if lam_spec["estimate"] else y_bc_start
        states = st.global_states(qr_x.coef(y_bc), struct)
        arma_initial = np.zeros(struct["arma_lag_max"])
        deviations = elements["deviations"]
        if deviations is not None:
            for key in ("level", "trend", "sin", "cos"):
                states[key] = states[key] + deviations[key]
            arma_initial = deviations["arma"]
        profile = st.profile(states, arma_initial, struct, elements["phi"])
        fitted = adam_cpp.fit(
            mat_vt.copy(order="F"),
            np.asfortranarray(np.tile(elements["w"], (obs, 1))),
            np.asfortranarray(elements["mat_f"]),
            np.asarray(elements["vec_g"], dtype=float),
            lookup,
            np.array(profile, order="F"),
            np.asarray(y_bc, dtype=float),
            ot,
            backcast,
            int(s["n_iterations"]),
            "n",
        )
        return {
            "states": np.asarray(fitted.states),
            "fitted": np.ravel(fitted.fitted),
            "errors": np.ravel(fitted.errors),
            "profile": np.asarray(fitted.profile),
            "y_bc": y_bc,
            "profile_initial": profile,
        }

    def loss_value(B: NDArray, loss: str) -> float:
        elements = filler(B, s["bounds"])
        if elements["penalty"] > 0:
            return float(elements["penalty"])
        fitted = fit_states(elements)
        errors = fitted["errors"]
        if loss in MULTISTEP_LOSSES:
            horizon = s["h"]
            adam_errors = np.asarray(
                adam_cpp.ferrors(
                    np.asfortranarray(fitted["states"]),
                    np.asfortranarray(np.tile(elements["w"], (obs, 1))),
                    np.asfortranarray(elements["mat_f"]),
                    lookup,
                    np.array(fitted["profile_initial"], order="F"),
                    int(horizon),
                    np.asarray(fitted["y_bc"], dtype=float),
                ).errors
            )
            n = obs - horizon
            squares = _sum_r(adam_errors**2, axis=0) / n
            if loss == "MSEh":
                value = _sum_r(adam_errors[:, horizon - 1] ** 2) / n
            elif loss == "TMSE":
                value = _sum_r(squares)
            elif loss == "GTMSE":
                value = _sum_r(np.log(squares))
            elif loss == "MSCE":
                value = _sum_r(_sum_r(adam_errors, axis=1) ** 2) / n
            else:
                value = math.log(np.linalg.det(adam_errors.T @ adam_errors / n))
        elif loss == "likelihood":
            value = (
                -st.loglik_value(errors, distribution, elements["shape"])
                - (elements["lambda"] - 1) * log_y
            )
        elif loss == "MSE":
            value = _sum_r(errors**2) / obs
        elif loss == "MAE":
            value = _sum_r(np.abs(errors)) / obs
        else:
            value = _sum_r(np.sqrt(np.abs(errors))) / obs
        return float(value) if np.isfinite(value) else 1e300

    def cf(B: NDArray) -> float:
        return loss_value(B, s["loss"])

    def point_lik(B: NDArray) -> NDArray:
        """The log-densities of the data at any parameters, as a refit with the
        model fixed: the final fit does not look at the bounds."""
        elements = filler(np.asarray(B, dtype=float), s["bounds"])
        errors = fit_states(elements)["errors"]
        scale = st.scale_value(errors, distribution, elements["shape"])
        values = calculate_likelihood(
            distribution, "A", errors, np.zeros((obs, 1)), scale, elements["shape"]
        )
        return np.ravel(values) + (elements["lambda"] - 1) * np.log(y)

    def fitter(B: NDArray) -> Optional[Dict[str, Any]]:
        """The fit at any parameters, for reapply: None outside the bounds."""
        elements = filler(np.asarray(B, dtype=float), s["bounds"])
        if elements["penalty"] > 0:
            return None
        fitted = fit_states(elements)
        return {**fitted, **elements}

    # Estimation
    B = b_list["B"].copy()
    if initial == "two-stage":
        backcast_fit = fit(
            y, trend_type, table, spec, lam_spec, distribution, "complete", s
        )
        common = [n for n in names if n in backcast_fit["names"]]
        for name in common:
            B[names.index(name)] = backcast_fit["B"][backcast_fit["names"].index(name)]
        B = deviations_from(B, names, backcast_fit, struct, qr_x, y)
    if s["B"] is not None:
        provided = np.asarray(s["B"], dtype=float)
        if len(provided) != len(B):
            raise ValueError(
                f"The provided B has {len(provided)} values, while the model needs "
                f"{len(B)}: {', '.join(names)}."
            )
        B = provided.copy()
    lb = b_list["lb"] if s["lb"] is None else np.asarray(s["lb"], dtype=float)
    ub = b_list["ub"] if s["ub"] is None else np.asarray(s["ub"], dtype=float)
    res = None
    if s["model_do"] == "estimate" and len(B) > 0:
        settings = {**s, "maxeval_used": s["maxeval"] or len(B) * 200}
        res = _optimise(cf, B, lb, ub, settings)
        # Stuck on a penalty: restart from no smoothing, unless B was provided
        if s["B"] is None and (
            not np.isfinite(res["objective"]) or res["objective"] >= 1e100
        ):
            restart = B.copy()
            restart[[bool(re.match(r"^(alpha|beta|gamma)", n)) for n in names]] = 0
            res = _optimise(cf, restart, lb, ub, settings)
        B = np.asarray(res["solution"], dtype=float)
    loss_final = cf(B)

    # The final fit
    elements = filler(B, s["bounds"])
    fitted = fit_states(elements)
    states = fitted["states"]
    if head["geometry"] > struct["lags_model_max"]:
        states = states[:, head["geometry"] - struct["lags_model_max"] :]
    initial_read = st.initials_read(states, struct)
    # The profile of the identified initials (the backcast ones with backcasting),
    # where the simulations start
    fitted["profile_initial"] = st.profile(
        initial_read["states"], initial_read["arma"], struct, elements["phi"]
    )

    # The identified initials are counted whether they are optimised or backcast
    n_initials = (
        1 + int(struct["trend_in"]) + 2 * struct["n_harmonics"] + struct["arma_lag_max"]
    )
    n_param_estimated = (
        len(B) * (s["model_do"] == "estimate") + 1 + n_initials * backcast
    )
    loglik = -loss_value(B, "likelihood")

    fi = None
    if s["fi"] and len(B) > 0:
        fi = -np.asarray(
            _hessian_cpp(
                lambda b: -loss_value(np.asarray(b), "likelihood"), B, s["step_size"]
            )
        )

    scale = st.scale_value(fitted["errors"], distribution, elements["shape"])
    forecast_bc = None
    if s["h"] > 0:
        columns = head["geometry"] + obs + np.arange(s["h"])
        forecast_bc = np.ravel(
            adam_cpp.forecast(
                np.asfortranarray(np.tile(elements["w"], (s["h"], 1))),
                np.asfortranarray(elements["mat_f"]),
                np.asfortranarray(lookup[:, columns]),
                np.array(fitted["profile"], order="F"),
                int(s["h"]),
            ).forecast
        )

    return {
        "names": names,
        "B": B,
        "res": res,
        "loss_value": loss_final,
        "loglik": loglik,
        "n_param_estimated": n_param_estimated,
        "n_initials": n_initials * backcast,
        "struct": struct,
        "spec": spec,
        "elements": elements,
        "fitted": fitted,
        "states": states,
        "initial_read": initial_read,
        "scale": scale,
        "forecast_bc": forecast_bc,
        "fi": fi,
        "trend_type": trend_type,
        "initial_type": initial,
        "distribution": distribution,
        "adam_cpp": adam_cpp,
        "lookup": lookup,
        "head": head,
        "fitter": fitter,
        "loss_function": loss_value,
        "point_lik": point_lik,
        "y": y,
    }


def deviations_from(
    B: NDArray,
    names: List[str],
    backcast_fit: Dict[str, Any],
    struct: Dict[str, Any],
    qr_x: st.QR,
    y: NDArray,
) -> NDArray:
    """The initial deviations from the global model that reproduce the initials of
    a fit."""
    B = B.copy()
    lam = backcast_fit["elements"]["lambda"]
    states = st.global_states(qr_x.coef(st.box_cox(y, lam)), struct)
    read = backcast_fit["initial_read"]
    index = {name: i for i, name in enumerate(names)}
    B[index["level"]] = read["states"]["level"] - states["level"]
    if struct["trend_in"]:
        B[index["trend"]] = read["states"]["trend"] - states["trend"]
    labels = st.harmonic_labels(struct["table"])
    for k, label in enumerate(labels):
        B[index[f"sin{label}"]] = read["states"]["sin"][k] - states["sin"][k]
        B[index[f"cos{label}"]] = read["states"]["cos"][k] - states["cos"][k]
    for k in range(struct["arma_lag_max"]):
        B[index[f"ARMAState{k + 1}"]] = read["arma"][k]
    return B
