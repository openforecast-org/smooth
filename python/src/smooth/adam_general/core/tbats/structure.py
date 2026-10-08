"""The structure of TBATS: the Box-Cox transform, the harmonics, the global model,
the ARMA specification and the state-space layout (R/adam-tbats.R)."""

import math
import warnings
from typing import Any, Callable, Dict, List, Optional

import greybox as gb
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from smooth.adam_general import _ols  # type: ignore[attr-defined]
from smooth.adam_general.core.utils.ic import AIC, BIC, AICc, BICc
from smooth.adam_general.core.utils.utils import (
    _exp_r,
    _log_r,
    _pow_r,
    _sum_r,
    make_names,
    scale_debias,
    scaler,
    xreg_selector,
)


# The Box-Cox transform and its inverse
# Through libm, as R's log(), exp() and ^: NumPy's kernels round differently
def box_cox(y: NDArray, lam: float) -> NDArray:
    y = np.asarray(y, dtype=float)
    if lam == 0:
        return _log_r(y)
    return (_pow_r(y, lam) - 1) / lam


def box_cox_inverse(z: NDArray, lam: float) -> NDArray:
    z = np.asarray(z, dtype=float)
    if lam == 0:
        return _exp_r(z)
    return _pow_r(np.maximum(lam * z + 1, 0), 1 / lam)


def box_cox_sizes(y: NDArray, lam: float, ot_logical: NDArray) -> NDArray:
    """The transformed sizes, zero where there is no demand (the fitter skips them)."""
    y_bc = np.zeros(len(y))
    y_bc[ot_logical] = box_cox(np.asarray(y, dtype=float)[ot_logical], lam)
    return y_bc


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


# The explanatory variables
def xreg_spec(
    X: Any, obs_in_sample: int, h: int, regressors: str
) -> Optional[Dict[str, Any]]:
    """The explanatory variables (R's ``tbats_xreg``): numeric, named, the in-sample
    rows and those of the horizon, the last row repeated when they do not reach it."""
    if X is None:
        return None
    if isinstance(X, pd.Series):
        X = X.to_frame()
    if isinstance(X, pd.DataFrame):
        if not all(pd.api.types.is_numeric_dtype(t) for t in X.dtypes):
            raise ValueError(
                "X should contain numeric variables only: convert the factors into "
                "dummy variables."
            )
        names = [str(c) for c in X.columns]
        values = X.to_numpy(dtype=float)
    else:
        values = np.asarray(X, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        names = [f"x{k}" for k in range(1, values.shape[1] + 1)]
    names = make_names(names)
    if values.shape[0] < obs_in_sample:
        raise ValueError("X has fewer rows than the in-sample data.")
    # The observations with a missing regressor are dropped: they are gaps of the
    # response
    missing = ~np.all(np.isfinite(values[:obs_in_sample]), axis=1)
    if np.any(missing):
        warnings.warn(
            "X has missing values: the observations of their rows are dropped.",
            stacklevel=3,
        )
    if values.shape[0] < obs_in_sample + h:
        warnings.warn(
            "X does not cover the horizon h. Repeating its last row.", stacklevel=3
        )
        pad = np.repeat(values[-1:], obs_in_sample + h - values.shape[0], axis=0)
        values = np.vstack([values, pad])
    if h > 0 and not np.all(np.isfinite(values[obs_in_sample : obs_in_sample + h])):
        raise ValueError(
            "X has missing values in the horizon: the forecasts need them."
        )
    return {
        "data": values[:obs_in_sample],
        "future": values[obs_in_sample : obs_in_sample + h] if h > 0 else None,
        "names": names,
        "number": len(names),
        "regressors": regressors,
        "missing": missing,
    }


def xreg_subset(spec: Dict[str, Any], names: List[str]) -> Optional[Dict[str, Any]]:
    """Some of the regressors, used as they are (None if none)."""
    if len(names) == 0:
        return None
    index = [spec["names"].index(name) for name in names]
    future = spec["future"]
    return {
        "data": spec["data"][:, index],
        "future": None if future is None else future[:, index],
        "names": list(names),
        "number": len(names),
        "regressors": "use",
    }


def mat_wt(
    w: NDArray, struct: Dict[str, Any], rows: int, xreg: Optional[NDArray] = None
) -> NDArray:
    """The measurement matrix for the rows of the regressors (or a number of rows).
    Their missing values are placeholders: the fit skips those observations."""
    result = np.asfortranarray(np.tile(w, (rows, 1)))
    if struct["n_xreg"] > 0:
        result[:, struct["xreg_rows"]] = xreg
        result[np.isnan(result)] = 0
    return result


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


def design(
    obs: int,
    trend_in: bool,
    table: Dict[str, NDArray],
    xreg: Optional[NDArray] = None,
) -> NDArray:
    """An intercept, a trend, the Fourier terms and the regressors."""
    times = np.arange(1, obs + 1, dtype=float)
    columns = [np.ones(obs)]
    if trend_in:
        columns.append(times)
    if len(table["frequency"]) > 0:
        angles = np.outer(times, table["frequency"])
        columns.extend(np.sin(angles).T)
        columns.extend(np.cos(angles).T)
    if xreg is not None:
        columns.extend(np.asarray(xreg, dtype=float).T)
    return np.column_stack(columns)


class QR:
    """The least squares of a fixed design, as R's ``tbats_qrCoef`` /
    ``tbats_qrResid``: the Householder QR shared with R (src/headers/olsCore.h,
    BLAS-free), so that the two agree to the last bit."""

    def __init__(self, X: NDArray):
        X = np.asarray(X, dtype=np.float64)
        self.qr, self.qraux, self.r_diag = _ols.householder_qr(np.asfortranarray(X))

    def _args(self, y: NDArray) -> tuple:
        return (
            np.asfortranarray(self.qr),
            self.qraux,
            self.r_diag,
            np.asarray(y, dtype=np.float64).ravel(),
        )

    def coef(self, y: NDArray) -> NDArray:
        return np.asarray(_ols.householder_coef(*self._args(y)))

    def resid(self, y: NDArray) -> NDArray:
        return np.asarray(_ols.householder_resid(*self._args(y)))


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
    log_y = _sum_r(_log_r(y))

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
    y: NDArray,
    periods: List[float],
    trend_in: bool,
    spec: Dict[str, Any],
    ic: str,
    xreg: Optional[NDArray] = None,
    ot_logical: Optional[NDArray] = None,
) -> List[int]:
    """The number of harmonics of each period by the IC of the global model, one
    period at a time, stopping after two harmonics without improvement. With an
    occurrence model, on the non-zero observations, which keep their time index."""
    harmonics = [0] * len(periods)
    if len(periods) == 0:
        return harmonics
    k_max = [max(math.ceil(p / 2) - 1, 0) for p in periods]
    obs_all = len(y)
    rows = np.ones(obs_all, dtype=bool) if ot_logical is None else ot_logical

    def design_used(test: List[int]) -> NDArray:
        return design(obs_all, trend_in, harmonics_table(periods, test), xreg)[rows]

    y = np.asarray(y, dtype=float)[rows]
    obs = len(y)
    y_bc = box_cox(y, lambda_start(y, design_used([min(k, 3) for k in k_max]), spec))

    def value(test: List[int]) -> float:
        X = design_used(test)
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


def gapped(residuals: NDArray, ot_logical: NDArray) -> NDArray:
    """The residuals of the observed values at their places and zeros at the gaps
    (the zeros of the occurrence), so that the lags of the ARMA stay aligned. The
    residuals of the global model are centred, and so is the series in
    Hannan-Rissanen."""
    result = np.zeros(len(ot_logical))
    result[ot_logical] = residuals
    return result


def outlier_dummies(
    y: NDArray,
    ot_logical: NDArray,
    trend_in: bool,
    table: Dict[str, NDArray],
    lam_spec: Dict[str, Any],
    xreg_data: Optional[NDArray],
    distribution: str,
    level: float,
    outliers: str,
    ic: str,
    h: int,
) -> Optional[Dict[str, Any]]:
    """The dummies of the outliers of the global model, as R's ``tbats_outliers``:
    the observations whose residuals, at the starting value of lambda and on the
    non-zero observations, standardised as ``rstandard()`` does, lie outside the
    ``level`` quantiles of the distribution (dgnorm with the shape of ``ALM`` on
    the residuals). With ``"select"``, ``stepwise()`` chooses among them and their
    leads and lags, on the residuals. ``{"data", "names"}`` with zeros over the
    horizon, or None."""
    X = design(len(y), trend_in, table, xreg_data)[ot_logical]
    lam = lambda_start(y[ot_logical], X, lam_spec)
    residuals = QR(X).resid(box_cox(y[ot_logical], lam))
    obs = len(residuals)
    shape = None
    if distribution == "dgnorm":
        fitted = gb.ALM(distribution="dgnorm").fit(np.ones((obs, 1)), residuals)
        shape = float(fitted.other_)
    scale = scale_debias(
        scale_value(residuals, distribution, shape), distribution, obs, obs - X.shape[1]
    )
    divisor = {"dnorm": math.sqrt(scale), "ds": scale**2}.get(distribution, scale)
    errors = residuals / divisor
    probabilities = [(1 - level) / 2, (1 + level) / 2]
    if distribution == "dlaplace":
        statistic = gb.qlaplace(probabilities, 0, 1)
    elif distribution == "ds":
        statistic = gb.qs(probabilities, 0, 1)
    elif distribution == "dgnorm":
        statistic = gb.qgnorm(probabilities, 0, 1, shape)
    else:
        statistic = gb.qnorm(probabilities, 0, 1)
    statistic = np.asarray(statistic, dtype=float)
    ids = np.flatnonzero(ot_logical)[(errors < statistic[0]) | (errors > statistic[1])]
    if len(ids) == 0:
        return None
    data = np.zeros((len(y) + h, len(ids)))
    data[ids, np.arange(len(ids))] = 1
    names = [f"outlier{i + 1}" for i in range(len(ids))]
    if outliers == "select":
        # Each dummy with its lag and lead, in R's order
        expanded = [
            gb.xreg_expander(data[:, [k]], [-1, 0, 1], gaps="zero")
            for k in range(data.shape[1])
        ]
        data = np.column_stack(expanded)
        suffixes = ("", "Lag1", "Lead1")
        names = [f"{name}{suffix}" for name in names for suffix in suffixes]
        selected = xreg_selector(
            residuals,
            data[: len(y)][ot_logical],
            names,
            ic,
            X.shape[1] + 1,
            distribution,
            shape,
        )
        if not selected:
            return None
        data = data[:, [names.index(name) for name in selected]]
        names = list(selected)
    return {"data": data, "names": names}


def arma_select(
    residuals: NDArray,
    spec: Dict[str, Any],
    distribution: str,
    shape: Optional[float],
    n_param_base: float,
    ic: str,
    observed: Optional[NDArray] = None,
) -> Dict[str, Any]:
    """The ARMA orders screened with Hannan-Rissanen on the residuals of a model
    without ARMA, one lag at a time from the largest. The IC of a candidate comes
    from the likelihood of its innovations on a common sample of the observed
    values."""
    lags = spec["lags"]
    ar_orders = np.zeros(len(lags), dtype=int)
    ma_orders = np.zeros(len(lags), dtype=int)
    obs = len(residuals)
    n_drop = min(int((spec["ar_orders"] * lags).sum()), obs // 4)
    used = np.arange(obs) >= n_drop
    if observed is not None:
        used &= observed
    obs_used = int(used.sum())
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
                loglik_value(innovations[used, j], distribution, shape),
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
    xreg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The parts of the model that do not depend on the parameters."""
    trend_in = trend_type != "none"
    n_ets = 1 + int(trend_in)
    n_h = len(table["frequency"])
    n_arma = len(spec["state_lags"])
    n_xreg = 0 if xreg is None else xreg["number"]
    lags_model_all = (
        [1] * n_ets + [1, 2] * n_h + list(spec["state_lags"]) + [1] * n_xreg
    )
    n_components = len(lags_model_all)
    harmonic_rows = n_ets + 2 * np.arange(n_h)
    arma_rows = n_ets + 2 * n_h + np.arange(n_arma)
    xreg_rows = n_ets + 2 * n_h + n_arma + np.arange(n_xreg)
    mat_f = np.zeros((n_components, n_components))
    mat_f[0, 0] = 1
    mat_f[xreg_rows, xreg_rows] = 1
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
    names += [] if xreg is None else list(xreg["names"])
    return {
        "trend_type": trend_type,
        "trend_in": trend_in,
        "damped": trend_type == "damped",
        "n_ets": n_ets,
        "n_harmonics": n_h,
        "n_arma": n_arma,
        "n_xreg": n_xreg,
        "n_components": n_components,
        "lags_model_all": lags_model_all,
        "lags_model_max": max(lags_model_all),
        "harmonic_rows": harmonic_rows,
        "arma_rows": arma_rows,
        "xreg_rows": xreg_rows,
        "xreg": xreg,
        "xreg_adapt": n_xreg > 0 and xreg is not None and xreg["regressors"] == "adapt",
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
        "xreg": beta[k + 2 * n_h : k + 2 * n_h + struct["n_xreg"]],
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
        result[struct["arma_rows"][-1], : struct["arma_lag_max"]] = arma_initial
    if struct["n_xreg"] > 0:
        result[struct["xreg_rows"], 0] = states["xreg"]
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
        "xreg": mat_vt[struct["xreg_rows"], last],
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
        arma = mat_vt[
            struct["arma_rows"][-1], lag_max - struct["arma_lag_max"] : lag_max
        ]
    return {"states": states, "arma": arma}


# The provided parameters
_PERSISTENCE_ALIASES = {"alpha": "level", "beta": "trend", "gamma": "seasonal"}


def provided_values(
    provided: Dict[str, Any], struct: Dict[str, Any], spec: Dict[str, Any]
) -> Dict[str, Any]:
    """The parameters and initial states provided by the user for a structure, as in
    ADAM (see R's ``tbats_provided``): persistence, phi and the ARMA on the names of
    B, the initials as the states in the space of the transformed data, which replace
    the global ones (and the B entries they make redundant)."""
    periods = struct["periods"]
    rows_period = [np.flatnonzero(struct["table"]["period"] == p) for p in periods]
    k = 1 + int(struct["trend_in"])
    n_xreg = struct["n_xreg"]
    xreg_names = list(struct["xreg"]["names"]) if n_xreg > 0 else []

    def split_periods(values: NDArray, sizes: List[int]) -> List[NDArray]:
        """The values per period after the level and trend of an unnamed vector."""
        ends = k + np.cumsum(sizes, dtype=int)
        return [values[end - size : end] for end, size in zip(ends, sizes)]

    values: Dict[str, float] = {}
    persistence = provided.get("persistence")
    if persistence is not None and not isinstance(persistence, dict):
        vector = np.atleast_1d(np.asarray(persistence, dtype=float))
        persistence = {
            "level": vector[0],
            "trend": vector[1] if struct["trend_in"] else None,
            "seasonal": split_periods(vector, [2] * len(periods)),
            "xreg": vector[k + 2 * len(periods) :],
        }
    persistence = {
        _PERSISTENCE_ALIASES.get(key, key): value
        for key, value in (persistence or {}).items()
    }
    if persistence.get("level") is not None:
        values["alpha"] = float(persistence["level"])
    if persistence.get("trend") is not None and struct["trend_in"]:
        values["beta"] = float(persistence["trend"])
    for period, rows, pair in zip(
        periods, rows_period, persistence.get("seasonal") or []
    ):
        if len(rows) > 0:
            label = _period_label(period)
            values[f"gamma1[{label}]"], values[f"gamma2[{label}]"] = map(float, pair)
    deltas = np.atleast_1d(np.asarray(persistence.get("xreg", []), dtype=float))
    if len(deltas) > 0 and struct["xreg_adapt"]:
        values.update({f"delta{j + 1}": float(v) for j, v in enumerate(deltas)})
    if provided.get("phi") is not None and struct["damped"]:
        values["phi"] = float(provided["phi"])
    arma = provided.get("arma")
    if arma is not None and spec["n_param"] > 0:
        kinds = arma if isinstance(arma, dict) else {"arma": arma}
        for kind, prefix in (("ar", "phi"), ("ma", "theta"), ("arma", "")):
            if kinds.get(kind) is not None:
                names = [n for n in spec["names"] if n.startswith(prefix)]
                parameters = np.atleast_1d(np.asarray(kinds[kind], dtype=float))
                values.update(zip(names, parameters.tolist()))

    # The initials, NaN where they are estimated
    initial = provided.get("initial")
    sizes = [2 * len(rows) for rows in rows_period]
    if initial is not None and not isinstance(initial, dict):
        vector = np.atleast_1d(np.asarray(initial, dtype=float))
        start = k + sum(sizes)
        initial = {
            "level": vector[0],
            "trend": vector[1] if struct["trend_in"] else None,
            "seasonal": split_periods(vector, sizes),
            "arma": vector[start : start + struct["arma_lag_max"]],
            "xreg": vector[
                start + struct["arma_lag_max"] : start + struct["arma_lag_max"] + n_xreg
            ],
        }
    initial = initial or {}
    n_h = struct["n_harmonics"]
    states: Dict[str, Any] = {
        "level": np.nan,
        "trend": np.nan,
        "sin": np.full(n_h, np.nan),
        "cos": np.full(n_h, np.nan),
        "xreg": np.full(n_xreg, np.nan),
    }
    if initial.get("level") is not None:
        states["level"] = float(initial["level"])
    if initial.get("trend") is not None and struct["trend_in"]:
        states["trend"] = float(initial["trend"])
    for period, rows, size, coefficients in zip(
        periods, rows_period, sizes, initial.get("seasonal") or []
    ):
        coefficients = np.asarray(coefficients, dtype=float)
        if len(coefficients) != size:
            raise ValueError(
                f"The initial seasonal coefficients of the period {period:g} should be "
                f"{size} values: the sines and then the cosines of its harmonics."
            )
        states["sin"][rows] = coefficients[: len(rows)]
        states["cos"][rows] = coefficients[len(rows) :]
    xreg = initial.get("xreg")
    if isinstance(xreg, dict):
        xreg = [xreg[name] for name in xreg_names]
    xreg = np.atleast_1d(np.asarray([] if xreg is None else xreg, dtype=float))
    if len(xreg) > 0:
        states["xreg"][:] = xreg
    arma_states = np.atleast_1d(np.asarray(initial.get("arma", []), dtype=float))
    arma_initial = (
        arma_states if len(arma_states) > 0 and struct["arma_lag_max"] > 0 else None
    )
    labels = harmonic_labels(struct["table"])
    drop = (
        (["level"] if not np.isnan(states["level"]) else [])
        + (["trend"] if not np.isnan(states["trend"]) else [])
        + [f"sin{x}" for x, v in zip(labels, states["sin"]) if not np.isnan(v)]
        + [f"cos{x}" for x, v in zip(labels, states["cos"]) if not np.isnan(v)]
        + (
            [f"ARMAState{j}" for j in range(1, struct["arma_lag_max"] + 1)]
            if arma_initial is not None
            else []
        )
        + [name for name, v in zip(xreg_names, states["xreg"]) if not np.isnan(v)]
    )
    known = sum(
        int(np.sum(~np.isnan(np.atleast_1d(value)))) for value in states.values()
    )
    return {
        "B": values,
        "states": states,
        "arma": arma_initial,
        "drop": drop,
        "number": len(values)
        + known
        + (0 if arma_initial is None else len(arma_initial)),
        "initial": len(drop) > 0,
    }
