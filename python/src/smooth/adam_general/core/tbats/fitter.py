"""The parameters and the fit of a TBATS structure (R/adam-tbats.R: tbats_B,
tbats_filler, tbats_gammaStart, tbats_eigens and tbats_fit)."""

import math
import re
from typing import Any, Callable, Dict, List, Optional

import nlopt
import numpy as np
from numpy.typing import NDArray

from smooth.adam_general import _adamCore, _ols  # type: ignore[attr-defined]
from smooth.adam_general._eigenCalc import eigen_moduli, smooth_eigens
from smooth.adam_general._numDeriv import hessian as _hessian_cpp
from smooth.adam_general.core.creator.architector import (
    adam_head_length,
    adam_profile_creator,
)
from smooth.adam_general.core.tbats import structure as st
from smooth.adam_general.core.utils.gradient import adam_fit_or_gradient
from smooth.adam_general.core.utils.utils import (
    _log_r,
    _sum_r,
    calculate_likelihood,
    complete_windows,
)
from smooth.adam_general.core.utils.var_covar import gap_variance

PENALTY = 1e100
MULTISTEP_LOSSES = ("MSEh", "TMSE", "GTMSE", "MSCE", "GPL")


# No occurrence model: all the observations are sizes
_NO_OCCURRENCE: Dict[str, Any] = {
    "model": None,
    "ot_logical": None,
    "loglik": 0.0,
    "n_param": 0,
    "p_fitted": None,
}


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
    xreg_estimate: bool = False,
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
    if struct["xreg_adapt"]:
        names += [f"delta{k}" for k in range(1, struct["n_xreg"] + 1)]
        values += [0.01] * struct["n_xreg"]
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
    # The regressors: deviations from the global model
    if xreg_estimate:
        names += list(struct["xreg"]["names"])
        values += [0.0] * struct["n_xreg"]
    if lam_spec["estimate"]:
        names.append("lambda")
        values.append(lam_start)
    if distribution == "dgnorm" and other_estimate:
        names.append("shape")
        values.append(2.0)
    lb = np.full(len(names), -np.inf)
    ub = np.full(len(names), np.inf)
    for i, name in enumerate(names):
        persistence = name in ("alpha", "beta", "phi") or name.startswith("delta")
        if bounds == "usual" and persistence:
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
    harmonics. A harmonic is reduced to (s_t, v2_t) with s_t = v1_t + v2_{t-1}: the
    lag-expanded form (v1_t, v2_t, v2_{t-1}) only adds a zero eigenvalue, as the
    discount matrix sends v1_t - v2_{t-1} to zero. The eigenvalues come from the
    LAPACK-free routine shared with R (``eigenModuliCore``): the optima often lie on the
    boundary, where two LAPACK builds disagreed in the last bit on which parameters
    were admissible."""
    n_ets = struct["n_ets"]
    n_h = struct["n_harmonics"]
    k = n_ets + 2 * n_h
    fe = np.zeros((k, k))
    ge = np.zeros(k)
    we = np.zeros(k)
    fe[:n_ets, :n_ets] = mat_f[:n_ets, :n_ets]
    ge[:n_ets] = vec_g[:n_ets]
    we[:n_ets] = w[:n_ets]
    for i in range(n_h):
        old = struct["harmonic_rows"][i]
        row = n_ets + 2 * i
        fe[row : row + 2, row] = mat_f[old : old + 2, old]
        fe[row, row + 1] = 1
        ge[row : row + 2] = vec_g[old : old + 2]
        we[row] = 1
    return np.ravel(eigen_moduli(np.asfortranarray(fe - np.outer(ge, we))))


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
        xreg_estimate: bool = False,
    ):
        self.index = {name: i for i, name in enumerate(names)}
        self.struct = struct
        self.spec = spec
        self.lam_spec = lam_spec
        self.other = other
        self.initial_estimate = initial_estimate
        self.adam_cpp = adam_cpp
        self.xreg_estimate = xreg_estimate
        n_xreg = struct["n_xreg"]
        self.deltas = np.asarray(
            [self.index[f"delta{k}"] for k in range(1, n_xreg + 1)]
            if struct["xreg_adapt"]
            else [],
            dtype=int,
        )
        self.xreg = np.asarray(
            [self.index[n] for n in struct["xreg"]["names"]] if xreg_estimate else [],
            dtype=int,
        )
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

        if struct["xreg_adapt"]:
            penalty += self._deltas(B[self.deltas], vec_g, bounds)

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
        xreg_deviations = (
            B[self.xreg] if self.xreg_estimate else np.zeros(struct["n_xreg"])
        )
        return {
            "mat_f": mat_f,
            "vec_g": vec_g,
            "w": w,
            "phi": phi,
            "lambda": float(lam),
            "shape": shape,
            "deviations": deviations,
            "xreg_deviations": xreg_deviations,
            "penalty": penalty,
        }

    def _deltas(self, deltas: NDArray, vec_g: NDArray, bounds: str) -> float:
        """The smoothing parameters of the regressors into the persistence vector,
        and the penalty of their bounds: the averaged condition of ADAM for the
        regressors, separately from the rest."""
        struct = self.struct
        vec_g[struct["xreg_rows"]] = deltas
        if bounds == "usual" and (np.any(deltas < 0) or np.any(deltas > 1)):
            return PENALTY
        if bounds != "admissible":
            return 0.0
        n_xreg = struct["n_xreg"]
        # over the rows of the observations the fit takes
        data = struct["xreg"]["data"]
        data = data[np.all(np.isfinite(data), axis=1)]
        values = np.abs(
            smooth_eigens(
                persistence=np.asfortranarray(deltas.reshape(-1, 1), dtype=float),
                transition=np.asfortranarray(np.eye(n_xreg)),
                measurement=np.asfortranarray(data, dtype=float),
                lags_model_all=np.ones(n_xreg, dtype=np.int32),
                xreg_model=True,
                obs_in_sample=data.shape[0],
                has_delta=True,
                xreg_number=n_xreg,
                constant_required=False,
            )
        )
        return PENALTY * values.max() if np.any(values > 1 + 1e-10) else 0.0


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


def _onto_bounds(B: NDArray, lb: NDArray, ub: NDArray, tol: float = 1e-10) -> NDArray:
    """The starting values within a hair of a finite bound, moved onto it, as R's
    tbats_ontoBounds: NLopt's default initial step shrinks to the distance to the
    bound, so the simplex of Nelder-Mead collapses within ~1e-14 of it (NLopt then
    fails), while on the bound it steps inwards. The values outside the bounds are
    left as they are."""
    B = np.array(B, dtype=float)
    lb, ub = np.asarray(lb, dtype=float), np.asarray(ub, dtype=float)
    with np.errstate(invalid="ignore"):
        to_upper, to_lower = ub - B, B - lb
        hair_upper = tol * np.maximum(1, abs(ub))
        hair_lower = tol * np.maximum(1, abs(lb))
        near_upper = np.isfinite(ub) & (to_upper >= 0) & (to_upper < hair_upper)
        near_lower = np.isfinite(lb) & (to_lower >= 0) & (to_lower < hair_lower)
    B[near_upper] = ub[near_upper]
    B[near_lower] = lb[near_lower]
    return B


def _optimise(
    cf: Callable[[NDArray], float],
    B: NDArray,
    lb: NDArray,
    ub: NDArray,
    s: Dict[str, Any],
) -> Dict[str, Any]:
    """NLopt with R's nloptr settings; the best point seen is returned."""
    B = _onto_bounds(B, lb, ub)
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
    xreg: Optional[Dict[str, Any]] = None,
    b_start: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """One fit of a fixed structure in the space of the Box-Cox transformed data,
    warm started from the parameters of a related fit in ``b_start``."""
    obs = len(y)
    periods = sorted(set(table["period"].tolist()))
    struct = st.structure(trend_type, table, spec, periods, xreg)
    xreg_data = None if xreg is None else xreg["data"]
    # With an occurrence model, the sizes: the global model, the transform and its
    # Jacobian on the non-zero observations, which keep their time index
    occurrence = s.get("occurrence") or _NO_OCCURRENCE
    ot_logical = (
        np.ones(obs, dtype=bool)
        if occurrence["ot_logical"] is None
        else occurrence["ot_logical"]
    )
    obs_nonzero = int(ot_logical.sum())
    # The sizes of the zeros are not observed: the zeros have only the likelihood
    # of the occurrence, and the scale is divided by the observed sizes
    observed = ~np.isnan(y)
    X = st.design(obs, struct["trend_in"], table, xreg_data)[ot_logical]
    qr_x = st.QR(X)
    lam_start = st.lambda_start(y[ot_logical], X, lam_spec)
    y_bc_start = st.box_cox_sizes(y, lam_start, ot_logical)
    log_y = (
        _sum_r(_log_r(y[ot_logical]))
        if (lam_spec["estimate"] or lam_start != 1)
        else 0.0
    )

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
        nArima=struct["n_components"] - n_ets - struct["n_xreg"],
        nXreg=struct["n_xreg"],
        nComponents=struct["n_components"],
        constant=False,
        adamETS=False,
    )
    # The harmonics make F large and mostly zeros
    adam_cpp.sparseTransition = True
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
    ot = ot_logical * 1.0
    # The structure as ADAM's gradient solve reads it: the level and trend as ETS, the
    # harmonics and the ARMA in the slot of ARIMA; the regressors stay in B
    n_arima = struct["n_components"] - n_ets - struct["n_xreg"]
    gradient_model = {
        "ets_model": True,
        "arima_model": n_arima > 0,
        "xreg_model": False,
        "error_type": "A",
        "trend_type": "A" if struct["trend_in"] else "N",
        "season_type": "N",
    }
    gradient_components = {
        "components_number_ets": n_ets,
        "components_number_ets_seasonal": 0,
        "components_number_ets_non_seasonal": n_ets,
        "components_number_arima": n_arima,
    }
    gradient_lags = {
        "lags_model": list(lags_all),
        "lags_model_all": list(lags_all),
        "lags_model_max": struct["lags_model_max"],
    }

    backcast = initial in ("backcasting", "complete")
    # The initials of the states that the fit determines (backcast or solved)
    initials_profiled = backcast or initial == "gradient"
    initial_estimate = initial in ("optimal", "two-stage")
    # The coefficients of the regressors are estimated unless all is backcast
    xreg_estimate = struct["n_xreg"] > 0 and initial != "complete"

    # The starting values of the ARMA from Hannan-Rissanen on the global residuals
    arma_start = np.zeros(0)
    if spec["n_param"] > 0:
        arma_start = np.asarray(
            _ols.arima_hr(
                st.gapped(qr_x.resid(y_bc_start[ot_logical]), ot_logical),
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
        xreg_estimate,
    )
    names_all = b_list["names"]
    filler = Filler(
        names_all,
        struct,
        spec,
        lam_spec,
        other,
        initial_estimate,
        adam_cpp,
        xreg_estimate,
    )
    # The provided values: in the full vector (the starting gammas around them), out
    # of B
    fixed = st.provided_values(s.get("provided") or {}, struct, spec)
    positions = [names_all.index(name) for name in fixed["B"]]
    b_all = b_list["B"].copy()
    b_all[positions] = list(fixed["B"].values())
    if s["bounds"] == "admissible" and struct["n_harmonics"] > 0:
        b_all = gamma_start(b_all, names_all, filler)
        b_all[positions] = list(fixed["B"].values())
    estimated = np.array(
        [n not in fixed["B"] and n not in fixed["drop"] for n in names_all],
        dtype=bool,
    )
    names = [n for n, e in zip(names_all, estimated) if e]
    b_list = {
        "B": b_all[estimated],
        "lb": b_list["lb"][estimated],
        "ub": b_list["ub"][estimated],
    }

    def b_full(B: NDArray) -> NDArray:
        """The full vector for the parameters of B, the provided ones included."""
        full = b_all.copy()
        full[estimated] = B
        return full

    # The cost function
    def fit_inputs(elements: Dict[str, Any]) -> Dict[str, Any]:
        """The transformed data, the initial profile and the measurement of the
        elements."""
        y_bc = (
            st.box_cox_sizes(y, elements["lambda"], ot_logical)
            if lam_spec["estimate"]
            else y_bc_start
        )
        states = st.global_states(qr_x.coef(y_bc[ot_logical]), struct)
        arma_initial = np.zeros(struct["arma_lag_max"])
        deviations = elements["deviations"]
        if deviations is not None:
            for key in ("level", "trend", "sin", "cos"):
                states[key] = states[key] + deviations[key]
            arma_initial = deviations["arma"]
        states["xreg"] = states["xreg"] + elements["xreg_deviations"]
        for key, value in fixed["states"].items():
            known = ~np.isnan(value)
            if np.ndim(value) == 0:
                states[key] = value if known else states[key]
            else:
                states[key] = np.where(known, value, states[key])
        if fixed["arma"] is not None:
            arma_initial = fixed["arma"]
        return {
            "y_bc": y_bc,
            "profile": st.profile(states, arma_initial, struct, elements["phi"]),
            "mat_wt": st.mat_wt(elements["w"], struct, obs, xreg_data),
        }

    def fit_states(elements: Dict[str, Any]) -> Dict[str, Any]:
        inputs = fit_inputs(elements)
        y_bc, profile, w_t = inputs["y_bc"], inputs["profile"], inputs["mat_wt"]
        # "gradient" solves for the initials of the states by least squares (the
        # model is additive in the transformed space), the regressors staying in B
        fitted = adam_fit_or_gradient(
            adam_cpp,
            mat_vt.copy(order="F"),
            w_t.copy(order="F"),
            np.asfortranarray(elements["mat_f"]),
            np.asarray(elements["vec_g"], dtype=float),
            lookup,
            np.array(profile, order="F"),
            np.asarray(y_bc, dtype=float),
            ot,
            initial,
            int(s["n_iterations"]),
            backcast,
            gradient_model,
            gradient_components,
            gradient_lags,
            obs,
            loss=s["loss"],
            distribution=distribution,
            other=elements["shape"],
            horizon=s["h"],
            multisteps=s["loss"] in MULTISTEP_LOSSES,
        )
        return {
            "states": np.asarray(fitted.states),
            "fitted": np.ravel(fitted.fitted),
            "errors": np.ravel(fitted.errors),
            "profile": np.asarray(fitted.profile),
            "y_bc": y_bc,
            "profile_initial": np.asarray(fitted.profileInitial),
            "mat_wt": w_t,
        }

    def loss_value(B: NDArray, loss: str) -> float:
        elements = filler(b_full(B), s["bounds"])
        if elements["penalty"] > 0:
            return float(elements["penalty"])
        fitted = fit_states(elements)
        errors = fitted["errors"][ot_logical]
        if loss in MULTISTEP_LOSSES:
            horizon = s["h"]
            adam_errors = np.asarray(
                adam_cpp.ferrors(
                    np.asfortranarray(fitted["states"]),
                    fitted["mat_wt"].copy(order="F"),
                    np.asfortranarray(elements["mat_f"]),
                    lookup,
                    np.array(fitted["profile_initial"], order="F"),
                    int(horizon),
                    np.asarray(fitted["y_bc"], dtype=float),
                ).errors
            )
            # The windows with all their targets observed, as R's
            adam_errors = adam_errors[complete_windows(~np.isnan(y), horizon)]
            n = len(adam_errors)
            squares = _sum_r(adam_errors**2, axis=0) / n
            if loss == "MSEh":
                value = _sum_r(adam_errors[:, horizon - 1] ** 2) / n
            elif loss == "TMSE":
                value = _sum_r(squares)
            elif loss == "GTMSE":
                value = _sum_r(_log_r(squares))
            elif loss == "MSCE":
                value = _sum_r(_sum_r(adam_errors, axis=1) ** 2) / n
            else:
                value = math.log(np.linalg.det(adam_errors.T @ adam_errors / n))
        elif loss == "likelihood":
            # The variance of the errors after the periods without an observed size,
            # with the normal distribution, as R
            gap = (
                gap_variance(
                    lags_all,
                    fitted["mat_wt"],
                    elements["mat_f"],
                    elements["vec_g"],
                    ot_logical,
                )[ot_logical]
                if distribution == "dnorm"
                else np.ones(obs_nonzero)
            )
            value = (
                -st.loglik_value(
                    errors / np.sqrt(gap), distribution, elements["shape"], obs_nonzero
                )
                + _sum_r(_log_r(gap)) / 2
                - (elements["lambda"] - 1) * log_y
            )
        elif loss == "MSE":
            value = _sum_r(errors**2) / obs_nonzero
        elif loss == "MAE":
            value = _sum_r(np.abs(errors)) / obs_nonzero
        elif loss == "custom":
            # On the observed values in the transformed space (zero where there is
            # no demand), as ADAM's
            value = s["loss_function"](
                actual=fitted["y_bc"][observed],
                fitted=fitted["fitted"][observed],
                B=B,
            )
        else:
            value = _sum_r(np.sqrt(np.abs(errors))) / obs_nonzero
        return float(value) if np.isfinite(value) else 1e300

    def cf(B: NDArray) -> float:
        return loss_value(B, s["loss"])

    def point_lik(B: NDArray) -> NDArray:
        """The log-densities of the data at any parameters, as a refit with the
        model fixed: the final fit does not look at the bounds."""
        elements = filler(b_full(np.asarray(B, dtype=float)), s["bounds"])
        fitted = fit_states(elements)
        gap = (
            gap_variance(
                lags_all,
                fitted["mat_wt"],
                elements["mat_f"],
                elements["vec_g"],
                ot_logical,
            )[ot_logical]
            if distribution == "dnorm"
            else np.ones(obs_nonzero)
        )
        errors = fitted["errors"][ot_logical] / np.sqrt(gap)
        scale = st.scale_value(errors, distribution, elements["shape"], obs_nonzero)
        values = calculate_likelihood(
            distribution,
            "A",
            errors,
            np.zeros((obs_nonzero, 1)),
            scale,
            elements["shape"],
        )
        sizes = (
            np.ravel(values)
            - _log_r(gap) / 2
            + (elements["lambda"] - 1) * _log_r(y[ot_logical])
        )
        # The occurrence, and the sizes where there is a demand; the missing values
        # are not in the likelihood, so theirs stay zero
        result = np.zeros(obs)
        if occurrence["model"] is not None:
            p_fitted = occurrence["p_fitted"]
            result[observed] = _log_r(1 - p_fitted[observed])
            result[ot_logical] = _log_r(p_fitted[ot_logical])
        result[ot_logical] += sizes
        return result

    def in_bounds(B: NDArray) -> bool:
        """Whether the parameters satisfy the bounds, for the draws of reapply."""
        return bool(
            filler(b_full(np.asarray(B, dtype=float)), s["bounds"])["penalty"] == 0
        )

    # The refits at the draws of the parameters (rows), for reapply: the C++ refitter
    # over all of them at once, each with its own matrices, initial profile and, with
    # lambda, data. The solved initials of "gradient" are kept as their deviations
    # from the global model (profile_offset, set after the estimation)
    profile_offset: Any = 0.0
    refit_backcast = backcast

    def refitter(draws: NDArray) -> Dict[str, Any]:
        refits = []
        for draw in np.atleast_2d(draws):
            elements = filler(b_full(np.asarray(draw, dtype=float)), s["bounds"])
            refits.append({**elements, **fit_inputs(elements)})
        nsim = len(refits)
        # The missing values are skipped, as in the fit
        y_bc = np.column_stack([r["y_bc"] for r in refits])
        missing = np.isnan(y_bc[:, 0]) | np.isnan(ot)
        y_bc[missing] = 0.0
        ot_refit = np.where(missing, 0.0, ot)
        arr_f = np.stack([r["mat_f"] for r in refits], axis=2)
        arr_wt = np.stack([r["mat_wt"] for r in refits], axis=2)
        mat_g = np.column_stack([np.ravel(r["vec_g"]) for r in refits])
        profiles = np.stack([r["profile"] + profile_offset for r in refits], axis=2)
        refitted = adam_cpp.reapply(
            matrixYt=np.asfortranarray(y_bc),
            matrixOt=np.asfortranarray(ot_refit[:, None]),
            arrayVt=np.zeros((*mat_vt.shape, nsim), order="F"),
            arrayWt=np.asfortranarray(arr_wt),
            arrayF=np.asfortranarray(arr_f),
            matrixG=np.asfortranarray(mat_g),
            indexLookupTable=lookup,
            arrayProfilesRecent=np.asfortranarray(profiles),
            backcast=refit_backcast,
        )
        return {
            "fitted": np.asarray(refitted.fitted),
            "states": np.asarray(refitted.states),
            "profile": np.asarray(refitted.profile),
            "mat_f": arr_f,
            "mat_wt": arr_wt,
            "vec_g": mat_g,
            "lambda": np.array([r["lambda"] for r in refits]),
        }

    # Estimation
    B = b_list["B"].copy()
    # The warm start from the parameters of a related fit, kept if it beats the default
    if b_start is not None:
        warm = np.array([b_start.get(n, b) for n, b in zip(names, B)])
        if cf(warm) < cf(B):
            B = warm
    # Backcast first, then optimise all from its parameters and initials, unless B is
    # provided
    if initial == "two-stage" and s["B"] is None:
        backcast_fit = fit(
            y,
            trend_type,
            table,
            spec,
            lam_spec,
            distribution,
            "complete",
            s,
            xreg,
            b_start,
        )
        common = [n for n in names if n in backcast_fit["names"]]
        for name in common:
            B[names.index(name)] = backcast_fit["B"][backcast_fit["names"].index(name)]
        B = deviations_from(B, names, backcast_fit, struct, qr_x, y[ot_logical])
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
    elements = filler(b_full(B), s["bounds"])
    fitted = fit_states(elements)
    states = fitted["states"]
    if head["geometry"] > struct["lags_model_max"]:
        states = states[:, head["geometry"] - struct["lags_model_max"] :]
    initial_read = st.initials_read(states, struct)
    # The solved initials of "gradient" for the refits; a failed solve falls back to
    # the backcast fit
    if initial == "gradient":
        profile_offset = fitted["profile_initial"] - fit_inputs(elements)["profile"]
        refit_backcast = bool(np.all(profile_offset == 0))
    # The profile of the identified initials (the backcast ones with backcasting),
    # where the simulations start
    fitted["profile_initial"] = st.profile(
        initial_read["states"], initial_read["arma"], struct, elements["phi"]
    )

    # The identified initials are counted whether they are optimised, backcast or
    # solved
    n_initials = (
        1 + int(struct["trend_in"]) + 2 * struct["n_harmonics"] + struct["arma_lag_max"]
    )
    n_param_estimated = (
        len(B) * (s["model_do"] == "estimate")
        + 1
        + n_initials * initials_profiled
        + struct["n_xreg"] * (not xreg_estimate)
    )
    # The likelihood of the occurrence model is added, as its parameters are. A custom
    # loss is minus the log-likelihood, as in ADAM
    loss_likelihood = "custom" if s["loss"] == "custom" else "likelihood"
    loglik = -loss_value(B, loss_likelihood) + occurrence["loglik"]

    fi = None
    if s["fi"] and len(B) > 0:
        fi = -np.asarray(
            _hessian_cpp(
                lambda b: -loss_value(np.asarray(b), loss_likelihood), B, s["step_size"]
            )
        )

    gap = (
        gap_variance(
            lags_all, fitted["mat_wt"], elements["mat_f"], elements["vec_g"], ot_logical
        )[ot_logical]
        if distribution == "dnorm"
        else np.ones(obs_nonzero)
    )
    scale = st.scale_value(
        fitted["errors"][ot_logical] / np.sqrt(gap),
        distribution,
        elements["shape"],
        obs_nonzero,
    )
    forecast_bc = None
    if s["h"] > 0:
        columns = head["geometry"] + obs + np.arange(s["h"])
        forecast_bc = np.ravel(
            adam_cpp.forecast(
                st.mat_wt(
                    elements["w"],
                    struct,
                    s["h"],
                    None if xreg is None else xreg["future"],
                ),
                np.asfortranarray(elements["mat_f"]),
                np.asfortranarray(lookup[:, columns]),
                np.array(fitted["profile"], order="F"),
                int(s["h"]),
            ).forecast
        )

    return {
        "names": names,
        "B": B,
        "B_full": dict(zip(names_all, b_full(B))),
        "n_param_provided": fixed["number"],
        "initial_provided": fixed["initial"],
        "res": res,
        "loss_value": loss_final,
        "loglik": loglik,
        "n_param_estimated": n_param_estimated,
        "n_param_occurrence": occurrence["n_param"],
        "n_initials": n_initials * initials_profiled,
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
        "in_bounds": in_bounds,
        "refitter": refitter,
        "loss_function": loss_value,
        "point_lik": point_lik,
        "y": y,
        "xreg_estimate": xreg_estimate,
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
    for k, name in enumerate(struct["xreg"]["names"] if struct["n_xreg"] else []):
        B[index[name]] = read["states"]["xreg"][k] - states["xreg"][k]
    return B
