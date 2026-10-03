"""The structure of TBATS: the Box-Cox transform, the harmonics, the global model,
the ARMA specification and the state-space layout (R/adam-tbats.R)."""

import math
import warnings
from typing import Any, Callable, Dict, List, Optional

import greybox as gb
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_triangular

from smooth.adam_general import _ols  # type: ignore[attr-defined]
from smooth.adam_general.core.utils.ic import AIC, BIC, AICc, BICc
from smooth.adam_general.core.utils.utils import _sum_r, scaler


# The Box-Cox transform and its inverse
def box_cox(y: NDArray, lam: float) -> NDArray:
    y = np.asarray(y, dtype=float)
    if lam == 0:
        return np.log(y)
    return (y**lam - 1) / lam


def box_cox_inverse(z: NDArray, lam: float) -> NDArray:
    z = np.asarray(z, dtype=float)
    if lam == 0:
        return np.exp(z)
    return np.maximum(lam * z + 1, 0) ** (1 / lam)


def lambda_spec(lam: Optional[float], y: NDArray, loss: str) -> Dict[str, Any]:
    """How lambda is treated: estimated in [0, 1] only with the likelihood and
    positive data; otherwise fixed (provided, or 1)."""
    if lam is not None:
        if lam < 0 or lam > 1:
            raise ValueError("lambda should lie in [0, 1].")
        if np.any(y <= 0) and lam != 1:
            warnings.warn(
                "The Box-Cox transform needs positive data. Setting lambda=1.",
                stacklevel=3,
            )
            lam = 1.0
        return {"estimate": False, "value": float(lam)}
    if np.any(y <= 0):
        warnings.warn(
            "The Box-Cox transform needs positive data. Setting lambda=1.",
            stacklevel=3,
        )
        return {"estimate": False, "value": 1.0}
    if loss != "likelihood":
        warnings.warn(
            'lambda is estimated with loss="likelihood" only. Setting lambda=1.',
            stacklevel=3,
        )
        return {"estimate": False, "value": 1.0}
    return {"estimate": True, "value": None}


# The harmonics and the global model
def harmonics_table(periods: List[float], harmonics: List[int]) -> Dict[str, NDArray]:
    """The harmonics of the periods, without those of a longer period whose
    frequency coincides with a harmonic of a shorter one."""
    period: List[float] = []
    j: List[int] = []
    for p, k in zip(periods, harmonics):
        period.extend([float(p)] * int(k))
        j.extend(range(1, int(k) + 1))
    period_arr = np.asarray(period, dtype=float)
    j_arr = np.asarray(j, dtype=np.int64)
    frequency = 2 * np.pi * j_arr / period_arr if len(j) else np.zeros(0)
    _, first = np.unique(np.round(frequency, 10), return_index=True)
    keep = np.sort(first)
    return {"period": period_arr[keep], "j": j_arr[keep], "frequency": frequency[keep]}


def design(obs: int, trend_in: bool, table: Dict[str, NDArray]) -> NDArray:
    """An intercept, a trend and the Fourier terms."""
    times = np.arange(1, obs + 1, dtype=float)
    columns = [np.ones(obs)]
    if trend_in:
        columns.append(times)
    if len(table["frequency"]) > 0:
        angles = np.outer(times, table["frequency"])
        columns.extend(np.sin(angles).T)
        columns.extend(np.cos(angles).T)
    return np.column_stack(columns)


class QR:
    """The least squares of a fixed design, as R's ``qr.coef`` / ``qr.resid``."""

    def __init__(self, X: NDArray):
        self.q, self.r = np.linalg.qr(X)

    def coef(self, y: NDArray) -> NDArray:
        return solve_triangular(self.r, self.q.T @ y)

    def resid(self, y: NDArray) -> NDArray:
        return y - self.q @ (self.q.T @ y)


def r_optimize(f: Callable[[float], float], lower: float, upper: float) -> float:
    """R's ``optimize()``: Brent's minimisation (``Brent_fmin`` in R's
    ``optimize.c``) with ``tol=.Machine$double.eps^0.25``."""
    tol = np.finfo(float).eps ** 0.25
    c = (3.0 - math.sqrt(5.0)) * 0.5
    eps = math.sqrt(np.finfo(float).eps)
    a, b = lower, upper
    v = w = x = a + c * (b - a)
    d = e = 0.0
    fx = fv = fw = f(x)
    tol3 = tol / 3.0
    while True:
        xm = (a + b) * 0.5
        tol1 = eps * abs(x) + tol3
        t2 = tol1 * 2.0
        if abs(x - xm) <= t2 - (b - a) * 0.5:
            break
        p = q = r = 0.0
        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = (q - r) * 2.0
            if q > 0.0:
                p = -p
            else:
                q = -q
            r = e
            e = d
        if abs(p) >= abs(q * 0.5 * r) or p <= q * (a - x) or p >= q * (b - x):
            e = (b - x) if x < xm else (a - x)
            d = c * e
        else:
            d = p / q
            u = x + d
            if u - a < t2 or b - u < t2:
                d = tol1 if x < xm else -tol1
        if abs(d) >= tol1:
            u = x + d
        elif d > 0.0:
            u = x + tol1
        else:
            u = x - tol1
        fu = f(u)
        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, fv = w, fw
                w, fw = u, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu
    return x


def lambda_profile(y: NDArray, qr_x: QR) -> float:
    """The maximum of the profile log-likelihood of the global model in lambda,
    with the Jacobian."""
    obs = len(y)
    log_y = _sum_r(np.log(y))

    def negative(lam: float) -> float:
        rss = _sum_r(qr_x.resid(box_cox(y, lam)) ** 2)
        return -(-obs / 2 * math.log(rss / obs) + (lam - 1) * log_y)

    # Rounded: Brent's search finds it to about 1e-4, and the last bits of the
    # least squares differ between linear algebra libraries (R's LINPACK QR)
    return round(r_optimize(negative, 0.0, 1.0), 8)


def lambda_start(y: NDArray, X: NDArray, spec: Dict[str, Any]) -> float:
    if not spec["estimate"]:
        return float(spec["value"])
    return lambda_profile(y, QR(X))


def ic_value(loglik: float, nobs: int, df: float, ic: str) -> float:
    function = {"AIC": AIC, "AICc": AICc, "BIC": BIC, "BICc": BICc}[ic]
    return float(function(loglik, nobs, df))


def harmonics_select(
    y: NDArray, periods: List[float], trend_in: bool, spec: Dict[str, Any], ic: str
) -> List[int]:
    """The number of harmonics of each period by the IC of the global model, one
    period at a time, stopping after two harmonics without improvement."""
    harmonics = [0] * len(periods)
    if len(periods) == 0:
        return harmonics
    k_max = [max(math.ceil(p / 2) - 1, 0) for p in periods]
    obs = len(y)
    table = harmonics_table(periods, [min(k, 3) for k in k_max])
    y_bc = box_cox(y, lambda_start(y, design(obs, trend_in, table), spec))

    def value(test: List[int]) -> float:
        X = design(obs, trend_in, harmonics_table(periods, test))
        if X.shape[1] >= obs - 1:
            return math.inf
        rss = _sum_r(QR(X).resid(y_bc) ** 2)
        loglik = -obs / 2 * (math.log(2 * math.pi * rss / obs) + 1)
        return ic_value(loglik, obs, X.shape[1] + 1, ic)

    best = value(harmonics)
    for i in range(len(periods)):
        failures = 0
        test = list(harmonics)
        while test[i] < k_max[i] and failures < 2:
            test[i] += 1
            current = value(test)
            if current < best:
                best = current
                harmonics[i] = test[i]
                failures = 0
            else:
                failures += 1
    return harmonics


# The ARMA
def arma_spec(orders: Dict[str, Any], lags: List[float]) -> Dict[str, Any]:
    """The orders of the ARMA aligned with the lags, truncated to integers."""
    select = bool(orders.get("select", False))
    ar = np.atleast_1d(orders.get("ar", 0) or 0).astype(int)
    ma = np.atleast_1d(orders.get("ma", 0) or 0).astype(int)
    if len(ar) == 1 and len(ma) == 1:
        arma_lags = np.array([1])
    else:
        arma_lags = np.trunc(np.asarray(lags, dtype=float)).astype(int)
    ar = np.resize(ar, len(arma_lags))
    ma = np.resize(ma, len(arma_lags))
    return arma_build(ar, ma, arma_lags, select)


def arma_build(ar, ma, arma_lags, select: bool = False) -> Dict[str, Any]:
    """The specification of the ARMA from its orders per lag."""
    ar = np.atleast_1d(np.asarray(ar, dtype=int))
    ma = np.atleast_1d(np.asarray(ma, dtype=int))
    arma_lags = np.atleast_1d(np.asarray(arma_lags, dtype=int))
    lags_unique = np.unique(arma_lags)
    ar_orders = np.array([ar[arma_lags == lag].max() for lag in lags_unique], dtype=int)
    ma_orders = np.array([ma[arma_lags == lag].max() for lag in lags_unique], dtype=int)
    keep = (ar_orders + ma_orders) > 0
    ar_orders, ma_orders, lags_unique = (
        ar_orders[keep],
        ma_orders[keep],
        lags_unique[keep],
    )

    def powers(orders: NDArray) -> set:
        if len(orders) == 0:
            return set()
        grid = np.array(
            np.meshgrid(
                *[np.arange(o + 1) * lag for o, lag in zip(orders, lags_unique)]
            )
        ).reshape(len(orders), -1)
        return set(grid.sum(axis=0).tolist()) - {0}

    state_lags = sorted(powers(ar_orders) | powers(ma_orders))
    names: List[str] = []
    for p, q, lag in zip(ar_orders, ma_orders, lags_unique):
        names += [f"phi{k}[{lag}]" for k in range(1, p + 1)]
        names += [f"theta{k}[{lag}]" for k in range(1, q + 1)]
    return {
        "select": select,
        "ar_orders": ar_orders,
        "ma_orders": ma_orders,
        "lags": lags_unique,
        "state_lags": [int(s) for s in state_lags],
        "n_param": int((ar_orders + ma_orders).sum()),
        "names": names,
    }


def scale_value(errors: NDArray, distribution: str, shape: Optional[float]) -> float:
    """sigma^2 for dnorm, s for the others (the ADAM monograph)."""
    return float(scaler(distribution, "A", errors, None, len(errors), shape))


def loglik_value(errors: NDArray, distribution: str, shape: Optional[float]) -> float:
    """The log-likelihood of the errors in the space of the transformed data."""
    scale = scale_value(errors, distribution, shape)
    if distribution == "dnorm":
        values = gb.dnorm(errors, 0, math.sqrt(scale), log=True)
    elif distribution == "dlaplace":
        values = gb.dlaplace(errors, 0, scale, log=True)
    elif distribution == "ds":
        values = gb.ds(errors, 0, scale, log=True)
    else:
        values = gb.dgnorm(errors, 0, scale, shape, log=True)
    return float(_sum_r(np.asarray(values, dtype=float)))


def arma_select(
    residuals: NDArray,
    spec: Dict[str, Any],
    distribution: str,
    shape: Optional[float],
    n_param_base: float,
    ic: str,
) -> Dict[str, Any]:
    """The ARMA orders screened with Hannan-Rissanen on the residuals of a model
    without ARMA, one lag at a time from the largest."""
    lags = spec["lags"]
    ar_orders = np.zeros(len(lags), dtype=int)
    ma_orders = np.zeros(len(lags), dtype=int)
    obs = len(residuals)
    n_drop = min(int((spec["ar_orders"] * lags).sum()), obs // 4)
    obs_used = obs - n_drop
    for i in np.argsort(-lags, kind="stable"):
        orders, _, innovations = _ols.arima_hr_select(
            np.asarray(residuals, dtype=float),
            ar_orders.astype(np.uint64),
            ma_orders.astype(np.uint64),
            lags.astype(np.uint64),
            int(i),
            int(spec["ar_orders"][i]),
            int(spec["ma_orders"][i]),
            True,
        )
        others = np.delete(ar_orders + ma_orders, i).sum()
        values = [
            ic_value(
                loglik_value(innovations[n_drop:, j], distribution, shape),
                obs_used,
                n_param_base + others + orders[j].sum(),
                ic,
            )
            for j in range(orders.shape[0])
        ]
        winner = int(np.argmin(values))
        ar_orders[i] = int(orders[winner, 0])
        ma_orders[i] = int(orders[winner, 1])
    return arma_build(ar_orders, ma_orders, lags)


# The state-space layout
def structure(
    trend_type: str,
    table: Dict[str, NDArray],
    spec: Dict[str, Any],
    periods: List[float],
) -> Dict[str, Any]:
    """The parts of the model that do not depend on the parameters."""
    trend_in = trend_type != "none"
    n_ets = 1 + int(trend_in)
    n_h = len(table["frequency"])
    n_arma = len(spec["state_lags"])
    lags_model_all = [1] * n_ets + [1, 2] * n_h + list(spec["state_lags"])
    n_components = len(lags_model_all)
    harmonic_rows = n_ets + 2 * np.arange(n_h)
    arma_rows = n_ets + 2 * n_h + np.arange(n_arma)
    mat_f = np.zeros((n_components, n_components))
    mat_f[0, 0] = 1
    for i in range(n_h):
        rows = harmonic_rows[i] + np.arange(2)
        mat_f[np.ix_(rows, rows)] = np.outer(
            [2 * np.cos(table["frequency"][i]), -1], [1, 1]
        )
    horizons = np.arange(math.ceil(max([1.0] + list(periods))))
    angles = np.outer(horizons, table["frequency"])
    labels = harmonic_labels(table)
    names = ["level"] + (["trend"] if trend_in else [])
    for label in labels:
        names += [f"s{label}", f"s*{label}"]
    names += [f"ARMAState{k}" for k in range(1, n_arma + 1)]
    return {
        "trend_type": trend_type,
        "trend_in": trend_in,
        "damped": trend_type == "damped",
        "n_ets": n_ets,
        "n_harmonics": n_h,
        "n_arma": n_arma,
        "n_components": n_components,
        "lags_model_all": lags_model_all,
        "lags_model_max": max(lags_model_all),
        "harmonic_rows": harmonic_rows,
        "arma_rows": arma_rows,
        "mat_f": mat_f,
        "table": table,
        "periods": list(periods),
        "response_cos": np.cos(angles),
        "response_sin": np.sin(angles),
        "arma_lag_max": max(spec["state_lags"]) if n_arma > 0 else 0,
        "component_names": names,
    }


def harmonic_labels(table: Dict[str, NDArray]) -> List[str]:
    """The labels of the harmonics, "j[period]"."""
    return [f"{j}[{_period_label(p)}]" for j, p in zip(table["j"], table["period"])]


def _period_label(period: float) -> str:
    """A period as R prints round(period, 4)."""
    value = round(float(period), 4)
    return str(int(value)) if value == int(value) else repr(value)


# The initial states
def global_states(beta: NDArray, struct: Dict[str, Any]) -> Dict[str, Any]:
    """The level and trend at t=0 and the Fourier coefficients of the global model."""
    beta = np.where(np.isfinite(beta), beta, 0.0)
    n_h = struct["n_harmonics"]
    k = 1 + int(struct["trend_in"])
    return {
        "level": beta[0],
        "trend": beta[1] if struct["trend_in"] else 0.0,
        "sin": beta[k : k + n_h],
        "cos": beta[k + n_h : k + 2 * n_h],
    }


def profile(
    states: Dict[str, Any],
    arma_initial: NDArray,
    struct: Dict[str, Any],
    phi: float = 1,
) -> NDArray:
    """The recent profile from the initial states (see R's ``tbats_profile``)."""
    lag_max = struct["lags_model_max"]
    result = np.zeros((struct["n_components"], lag_max))
    # phi=0 gives no finite walk back: the loss is then not finite, as in R
    with np.errstate(divide="ignore", invalid="ignore"):
        trends = states["trend"] / phi ** np.arange(lag_max)
    result[0, 0] = states["level"] - _sum_r(trends[:-1])
    if struct["trend_in"]:
        result[1, 0] = trends[-1]
    if struct["n_harmonics"] > 0:
        frequency = struct["table"]["frequency"]
        s0 = states["cos"]
        sm1 = -states["sin"] * np.sin(frequency) + states["cos"] * np.cos(frequency)
        rows = struct["harmonic_rows"]
        result[rows, 0] = 2 * np.cos(frequency) * s0
        result[rows + 1, 0] = -sm1
        result[rows + 1, 1] = -s0
    if struct["n_arma"] > 0:
        result[-1, : struct["arma_lag_max"]] = arma_initial
    return result


def initials_read(mat_vt: NDArray, struct: Dict[str, Any]) -> Dict[str, Any]:
    """The identified initials read back from the states (columns up to t=0)."""
    lag_max = struct["lags_model_max"]
    last = lag_max - 1
    states: Dict[str, Any] = {
        "level": mat_vt[0, last],
        "trend": mat_vt[1, last] if struct["trend_in"] else 0.0,
        "sin": np.zeros(0),
        "cos": np.zeros(0),
    }
    if struct["n_harmonics"] > 0:
        frequency = struct["table"]["frequency"]
        rows = struct["harmonic_rows"]
        s0 = -mat_vt[rows + 1, last]
        s1 = mat_vt[rows, last] + mat_vt[rows + 1, last - 1]
        sm1 = 2 * np.cos(frequency) * s0 - s1
        states["cos"] = s0
        states["sin"] = (s0 * np.cos(frequency) - sm1) / np.sin(frequency)
    arma = np.zeros(0)
    if struct["n_arma"] > 0:
        arma = mat_vt[-1, lag_max - struct["arma_lag_max"] : lag_max]
    return {"states": states, "arma": arma}
