"""TBATS in the Single Source of Error framework of ADAM (R's ``tbats()``)."""

import math
import time
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from smooth.adam_general.core.forecaster.forecaster import forecaster
from smooth.adam_general.core.forecaster.result import ForecastResult
from smooth.adam_general.core.tbats import fitter as ft
from smooth.adam_general.core.tbats import structure as st

TREND_OPTIONS = ("auto", "none", "additive", "damped")
DISTRIBUTION_OPTIONS = ("dnorm", "dlaplace", "ds", "dgnorm")
LOSS_OPTIONS = (
    "likelihood",
    "MSE",
    "MAE",
    "HAM",
    "MSEh",
    "TMSE",
    "GTMSE",
    "MSCE",
    "GPL",
)
IC_OPTIONS = ("AICc", "AIC", "BIC", "BICc")
INITIAL_OPTIONS = ("backcasting", "optimal", "two-stage", "complete")
BOUNDS_OPTIONS = ("admissible", "usual", "none")


def _match(value: str, options: tuple, name: str) -> str:
    """R's ``match.arg``: the value has to be one of the options."""
    if value not in options:
        raise ValueError(
            f"{name} should be one of {', '.join(options)}, not {value!r}."
        )
    return value


class TBATS:
    """
    Trigonometric Box-Cox ARMA Trend Seasonal model (De Livera et al., 2011) in the
    Single Source of Error framework of ADAM: the level and trend of ETS, a
    trigonometric seasonality for each period (fractional periods are allowed) and
    an ARMA in ADAM's form, all in the space of the Box-Cox transformed data.

    The point forecasts are the medians: the inverse Box-Cox transform of the point
    forecasts of the transformed data. See R's ``tbats()`` for the details.

    Parameters
    ----------
    lags : list of float or None
        The lags of the model: 1 and the seasonal periods, which can be fractional
        (e.g. ``[1, 7, 365.25]``). The harmonics are fitted for every lag above 1.
        None means ``[1]``.
    harmonics : list of int or None
        The number of harmonics for each lag above 1. None selects them by the
        information criterion on the global model.
    trend : str, default="auto"
        ``"none"``, ``"additive"``, ``"damped"`` or ``"auto"`` (selected by IC).
    lambda_bc : float or None
        The Box-Cox parameter (R's ``lambda``). None estimates it in [0, 1] with
        ``loss="likelihood"`` and sets it to 1 otherwise.
    orders : dict or None
        ``{"ar": ..., "ma": ..., "select": ...}`` aligned with ``lags``; a single
        value refers to the lag 1. None means ``{"ar": 3, "ma": 3, "select": True}``:
        the orders up to these are screened with Hannan-Rissanen on the residuals of
        the global model and the winner is kept if it improves the IC.
    distribution : str, default="dnorm"
        ``"dnorm"``, ``"dlaplace"``, ``"ds"`` or ``"dgnorm"``, in the space of the
        transformed data.
    loss, ic, h, holdout, initial, bounds
        As in R's ``tbats()``; ``bounds="admissible"`` keeps the model stable.
    verbose : int, default=0
        Not used yet (R's ``silent``).
    B, lb, ub, maxeval, maxtime, algorithm, xtol_rel, xtol_abs, ftol_rel, ftol_abs,
    print_level, n_iterations, head_length, fi, step_size, shape
        The optimiser settings and other parameters of R's ellipsis.
    """

    def __init__(
        self,
        lags: Optional[List[float]] = None,
        harmonics: Optional[List[int]] = None,
        trend: str = "auto",
        lambda_bc: Optional[float] = None,
        orders: Optional[Dict[str, Any]] = None,
        distribution: str = "dnorm",
        loss: str = "likelihood",
        ic: str = "AICc",
        h: int = 0,
        holdout: bool = False,
        initial: str = "backcasting",
        bounds: str = "admissible",
        verbose: int = 0,
        B: Optional[NDArray] = None,
        lb: Optional[NDArray] = None,
        ub: Optional[NDArray] = None,
        maxeval: Optional[int] = None,
        maxtime: float = -1,
        algorithm: str = "NLOPT_LN_NELDERMEAD",
        xtol_rel: float = 1e-6,
        xtol_abs: float = 1e-8,
        ftol_rel: float = 1e-8,
        ftol_abs: float = 0,
        print_level: int = 0,
        n_iterations: Optional[int] = None,
        head_length: Optional[int] = None,
        fi: bool = False,
        step_size: float = float(np.finfo(float).eps ** 0.25),
        shape: Optional[float] = None,
    ) -> None:
        self.lags = [1.0] if lags is None else [float(lag) for lag in lags]
        self.harmonics = None if harmonics is None else [int(k) for k in harmonics]
        self.trend = _match(trend, TREND_OPTIONS, "trend")
        self.lambda_bc = lambda_bc
        self.orders = {"ar": 3, "ma": 3, "select": True} if orders is None else orders
        self.distribution = _match(distribution, DISTRIBUTION_OPTIONS, "distribution")
        self.loss = _match(loss, LOSS_OPTIONS, "loss")
        self.ic = _match(ic, IC_OPTIONS, "ic")
        self.h = int(h)
        self.holdout = holdout
        self.initial = _match(initial, INITIAL_OPTIONS, "initial")
        self.bounds = _match(bounds, BOUNDS_OPTIONS, "bounds")
        self.verbose = verbose
        self.B = B
        self.lb = lb
        self.ub = ub
        self.maxeval = maxeval
        self.maxtime = maxtime
        self.algorithm = algorithm
        self.xtol_rel = xtol_rel
        self.xtol_abs = xtol_abs
        self.ftol_rel = ftol_rel
        self.ftol_abs = ftol_abs
        self.print_level = print_level
        self.n_iterations = n_iterations
        self.head_length = head_length
        self.fi = fi
        self.step_size = step_size
        self.shape = shape

    # Set by fit()
    _best: Dict[str, Any]
    harmonics_: List[int]
    trend_type_: str
    lambda_: float

    # Fitting
    def _settings(self) -> Dict[str, Any]:
        """R's ``checked``: the settings of the fitter."""
        n_iterations = self.n_iterations
        if n_iterations is None:
            n_iterations = 2 if self.initial in ("backcasting", "complete") else 1
        return {
            "h": self.h,
            "loss": self.loss,
            "bounds": self.bounds,
            "model_do": "estimate",
            "B": self.B,
            "lb": self.lb,
            "ub": self.ub,
            "maxeval": self.maxeval,
            "maxtime": self.maxtime,
            "algorithm": self.algorithm,
            "xtol_rel": self.xtol_rel,
            "xtol_abs": self.xtol_abs,
            "ftol_rel": self.ftol_rel,
            "ftol_abs": self.ftol_abs,
            "print_level": self.print_level,
            "n_iterations": n_iterations,
            "head_length": self.head_length,
            "fi": self.fi,
            "step_size": self.step_size,
            "shape": 2.0 if self.shape is None else float(self.shape),
            "shape_estimate": self.shape is None,
        }

    def fit(self, y: Union[NDArray, pd.Series]) -> "TBATS":
        """Fit the model to the series ``y``."""
        start_time = time.time()
        index = y.index if isinstance(y, pd.Series) else None
        values = np.asarray(y, dtype=float).ravel()
        h = self.h
        obs_in_sample = len(values) - h if (self.holdout and h > 0) else len(values)
        y_in_sample = values[:obs_in_sample]
        self._y_holdout = values[obs_in_sample:] if (self.holdout and h > 0) else None
        if not np.all(np.isfinite(y_in_sample)):
            raise ValueError("TBATS does not support missing values yet.")
        self._index = index

        lags, harmonics, trend = self.lags, self.harmonics, self.trend

        periods = sorted({lag for lag in lags if lag > 1})
        lam_spec = st.lambda_spec(self.lambda_bc, y_in_sample, self.loss)
        spec = st.arma_spec(self.orders, lags)
        spec_fit = st.arma_build([0], [0], [1]) if spec["select"] else spec
        trend_types = ["none", "additive", "damped"] if trend == "auto" else [trend]
        settings = self._settings()

        if harmonics is None:
            harmonics = st.harmonics_select(
                y_in_sample,
                periods,
                any(t != "none" for t in trend_types),
                lam_spec,
                self.ic,
            )
        else:
            if len(harmonics) != len(periods):
                raise ValueError("harmonics should have one value per lag above 1.")
            k_max = [math.ceil(p / 2) - 1 for p in periods]
            if any(k > m for k, m in zip(harmonics, k_max)):
                import warnings

                warnings.warn(
                    "The number of harmonics has to be below half of the period. "
                    "Reducing it.",
                    stacklevel=2,
                )
                harmonics = [min(k, m) for k, m in zip(harmonics, k_max)]
        table = st.harmonics_table(periods, harmonics)

        # Fit the candidates and select
        candidates = [
            ft.fit(
                y_in_sample,
                t,
                table,
                spec_fit,
                lam_spec,
                self.distribution,
                self.initial,
                settings,
            )
            for t in trend_types
        ]
        ics: Dict[str, float] = {
            t: self._ic(c) for t, c in zip(trend_types, candidates)
        }
        best = candidates[int(np.argmin(list(ics.values())))]

        # The ARMA orders, screened on the residuals of the global model
        if spec["select"] and spec["n_param"] > 0:
            X = st.design(len(y_in_sample), best["trend_type"] != "none", table)
            residuals = st.QR(X).resid(
                st.box_cox(y_in_sample, best["elements"]["lambda"])
            )
            spec_best = st.arma_select(
                residuals,
                spec,
                self.distribution,
                best["elements"]["shape"],
                best["n_param_estimated"],
                self.ic,
            )
            if spec_best["n_param"] > 0:
                candidate = ft.fit(
                    y_in_sample,
                    best["trend_type"],
                    table,
                    spec_best,
                    lam_spec,
                    self.distribution,
                    self.initial,
                    settings,
                )
                ic_candidate = self._ic(candidate)
                if ic_candidate < min(ics.values()):
                    best = candidate
                ar = ",".join(str(o) for o in spec_best["ar_orders"])
                ma = ",".join(str(o) for o in spec_best["ma_orders"])
                ics[f"{candidate['trend_type']}+ARMA({ar};{ma})"] = ic_candidate

        self._best = best
        self._settings_used = settings
        self.ics = ics
        self.harmonics_ = list(harmonics)
        self.periods_ = periods
        self.trend_type_ = best["trend_type"]
        self.lambda_ = best["elements"]["lambda"]
        self._y_in_sample = y_in_sample
        self.time_elapsed = time.time() - start_time
        return self

    def _ic(self, fitted: Dict[str, Any]) -> float:
        return st.ic_value(
            fitted["loglik"], len(fitted["y"]), fitted["n_param_estimated"], self.ic
        )

    # Fitted attributes
    def _check_fitted(self) -> None:
        if not hasattr(self, "_best"):
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

    @property
    def coef(self) -> NDArray:
        """The parameter vector B."""
        self._check_fitted()
        return self._best["B"].copy()

    @property
    def coef_names(self) -> List[str]:
        """The names of the parameters in B."""
        self._check_fitted()
        return list(self._best["names"])

    @property
    def loglik(self) -> float:
        """The log-likelihood of the data, with the Jacobian of the transform."""
        self._check_fitted()
        return float(self._best["loglik"])

    @property
    def nobs(self) -> int:
        self._check_fitted()
        return len(self._y_in_sample)

    @property
    def nparam(self) -> int:
        """The number of estimated parameters, the scale and the identified
        initials included."""
        self._check_fitted()
        return int(self._best["n_param_estimated"])

    @property
    def aic(self) -> float:
        return st.ic_value(self.loglik, self.nobs, self.nparam, "AIC")

    @property
    def aicc(self) -> float:
        return st.ic_value(self.loglik, self.nobs, self.nparam, "AICc")

    @property
    def bic(self) -> float:
        return st.ic_value(self.loglik, self.nobs, self.nparam, "BIC")

    @property
    def bicc(self) -> float:
        return st.ic_value(self.loglik, self.nobs, self.nparam, "BICc")

    @property
    def fitted(self) -> NDArray:
        """The fitted values in the space of the data (the medians)."""
        self._check_fitted()
        return st.box_cox_inverse(self._best["fitted"]["fitted"], self.lambda_)

    @property
    def residuals(self) -> NDArray:
        """The errors in the space of the transformed data."""
        self._check_fitted()
        return self._best["fitted"]["errors"].copy()

    @property
    def actuals(self) -> NDArray:
        self._check_fitted()
        return self._y_in_sample.copy()

    @property
    def states(self) -> NDArray:
        """The states, one row per component, from the start of the profile."""
        self._check_fitted()
        return self._best["states"].copy()

    @property
    def component_names(self) -> List[str]:
        self._check_fitted()
        return list(self._best["struct"]["component_names"])

    @property
    def persistence_vector(self) -> Dict[str, float]:
        self._check_fitted()
        return dict(zip(self.component_names, self._best["elements"]["vec_g"]))

    @property
    def transition(self) -> NDArray:
        self._check_fitted()
        return self._best["elements"]["mat_f"].copy()

    @property
    def measurement(self) -> NDArray:
        self._check_fitted()
        return np.tile(self._best["elements"]["w"], (self.nobs, 1))

    @property
    def phi_(self) -> float:
        self._check_fitted()
        return (
            float(self._best["elements"]["phi"])
            if self._best["struct"]["damped"]
            else 1.0
        )

    @property
    def scale(self) -> float:
        """sigma^2 for dnorm, s for the other distributions."""
        self._check_fitted()
        return float(self._best["scale"])

    @property
    def loss_value(self) -> float:
        self._check_fitted()
        return float(self._best["loss_value"])

    @property
    def distribution_(self) -> str:
        return self.distribution

    @property
    def loss_(self) -> str:
        return self.loss

    @property
    def orders_(self) -> Dict[str, Any]:
        """The ARMA orders and their lags."""
        self._check_fitted()
        spec = self._best["spec"]
        return {
            "ar": spec["ar_orders"].tolist(),
            "ma": spec["ma_orders"].tolist(),
            "lags": spec["lags"].tolist(),
        }

    @property
    def initial_value(self) -> Dict[str, Any]:
        """The level, trend, the Fourier coefficients of the harmonics and the ARMA
        initials at t=0."""
        self._check_fitted()
        struct = self._best["struct"]
        read = self._best["initial_read"]
        result: Dict[str, Any] = {"level": float(read["states"]["level"])}
        if struct["trend_in"]:
            result["trend"] = float(read["states"]["trend"])
        if struct["n_harmonics"] > 0:
            result["seasonal"] = pd.DataFrame(
                {
                    "period": struct["table"]["period"],
                    "j": struct["table"]["j"],
                    "sin": read["states"]["sin"],
                    "cos": read["states"]["cos"],
                }
            )
        if struct["n_arma"] > 0:
            result["arma"] = np.asarray(read["arma"])
        return result

    @property
    def model_name(self) -> str:
        """TBATS(lambda, {p,q}, phi, <m1,k1>, ...)."""
        self._check_fitted()
        struct = self._best["struct"]
        spec = self._best["spec"]
        seasonal = ""
        if struct["n_harmonics"] > 0:
            counts = [
                int(np.sum(struct["table"]["period"] == p)) for p in self.periods_
            ]
            seasonal = "".join(
                f", <{st._period_label(p)},{k}>"
                for p, k in zip(self.periods_, counts)
                if k > 0
            )
        phi = _r_round(self.phi_, 3) if struct["damped"] else "-"
        return (
            f"TBATS({_r_round(self.lambda_, 3)}, {{{int(spec['ar_orders'].sum())},"
            f"{int(spec['ma_orders'].sum())}}}, {phi}{seasonal})"
        )

    @property
    def fi_(self) -> Optional[NDArray]:
        """The observed Fisher Information (with ``fi=True``)."""
        self._check_fitted()
        return self._best["fi"]

    @property
    def forecast_(self) -> Optional[NDArray]:
        """The forecasts of the fit for ``h`` steps ahead."""
        self._check_fitted()
        if self._best["forecast_bc"] is None:
            return None
        return st.box_cox_inverse(self._best["forecast_bc"], self.lambda_)

    # Methods
    def point_lik(self, log: bool = True) -> NDArray:
        """The log-densities of the data: those of the transformed data and the
        Jacobian, which sum to the log-likelihood."""
        self._check_fitted()
        values = self._best["point_lik"](self._best["B"])
        return values if log else np.exp(values)

    def vcov(
        self,
        type: Optional[str] = None,  # noqa: A002
        heuristics: Optional[float] = None,
        step_size: Optional[float] = None,
    ) -> pd.DataFrame:
        """The covariance matrix of the parameters, as R's ``vcov.adam``:
        ``"opg"`` (the default) from the scores of the log-densities, or
        ``"hessian"`` from the observed Fisher Information."""
        import warnings

        from smooth.adam_general.core.utils.var_covar import (
            covar_opg,
            invert_fisher_information,
            resolve_covar_type,
        )

        self._check_fitted()
        names = self.coef_names
        B = self.coef
        if heuristics is not None:
            return pd.DataFrame(
                np.diag(np.abs(B) * heuristics), index=names, columns=names
            )
        covariance_type = resolve_covar_type(type)
        if covariance_type == "bootstrap":
            raise ValueError("The bootstrap covariance is not available for TBATS yet.")
        if covariance_type == "opg":
            covariance = None
            if self.loss == "likelihood":
                covariance = covar_opg(
                    B, self._best["point_lik"], self.nobs, self.loglik, step_size
                )
            if covariance is not None:
                return pd.DataFrame(covariance, index=names, columns=names)
            warnings.warn(
                "The OPG covariance could not be computed for this model; "
                "falling back to the observed Fisher Information.",
                RuntimeWarning,
                stacklevel=2,
            )
        fi = self._fisher_information(step_size)
        broken = np.all(fi == 0, axis=1) | np.any(np.isnan(fi), axis=1)
        if np.any(broken) and step_size is None:
            fi = self._fisher_information(float(np.finfo(float).eps ** (1 / 6)))
        return pd.DataFrame(invert_fisher_information(fi), index=names, columns=names)

    def _fisher_information(self, step_size: Optional[float]) -> NDArray:
        """The observed Fisher Information at the estimates."""
        from smooth.adam_general._numDeriv import hessian

        loss_function = self._best["loss_function"]
        step = self.step_size if step_size is None else step_size
        return -np.asarray(
            hessian(
                lambda b: -loss_function(np.asarray(b), "likelihood"), self.coef, step
            )
        )

    def predict(
        self,
        h: Optional[int] = None,
        interval: Literal[
            "none",
            "prediction",
            "simulated",
            "approximate",
            "semiparametric",
            "nonparametric",
            "empirical",
        ] = "none",
        level: Union[float, List[float]] = 0.95,
        side: Literal["both", "upper", "lower"] = "both",
        cumulative: bool = False,
        nsim: int = 10000,
    ) -> ForecastResult:
        """The forecasts of ADAM's forecaster in the space of the transformed data,
        transformed back: the point forecasts are the medians and the quantiles map
        onto those of the data."""
        self._check_fitted()
        if cumulative and self.lambda_ != 1:
            raise ValueError(
                "Cumulative forecasts of TBATS are only available for lambda=1: the "
                "sums of the transformed values do not transform back."
            )
        if h is None:
            h = self.h if self.h > 0 else 10
        best = self._best
        struct = best["struct"]
        n_ets = struct["n_ets"]
        n_param = self.nparam
        n_scale = int(self.loss == "likelihood")
        y_bc = st.box_cox(self._y_in_sample, self.lambda_)
        result = forecaster(
            model_prepared={
                "states": best["states"],
                "mat_vt": best["fitted"]["profile_initial"],
                "measurement": self.measurement,
                "transition": best["elements"]["mat_f"],
                "persistence": best["elements"]["vec_g"],
                "profiles_recent_table": best["fitted"]["profile"],
                "residuals": pd.Series(self.residuals),
                "y_fitted": y_bc - self.residuals,
                "scale": self.scale,
            },
            observations_dict={
                "obs_in_sample": self.nobs,
                "y_in_sample": y_bc,
                "y_forecast_start": self._forecast_start(),
                "frequency": self._frequency(),
            },
            general_dict={
                "h": int(h),
                "cumulative": cumulative,
                "nsim": nsim,
                "scenarios": False,
                "distribution": self.distribution,
                "loss": self.loss,
                "other": {"shape": best["elements"]["shape"]},
                "n_param": None,
                "scale_forecast": None,
            },
            occurrence_dict={"occurrence_model": False, "occurrence": "none"},
            lags_dict={
                "lags_model_all": struct["lags_model_all"],
                "lags_model_max": struct["lags_model_max"],
                "lags_model_min": 2 if struct["lags_model_max"] > 1 else np.inf,
                "lags": struct["lags_model_all"],
            },
            model_type_dict={
                "ets_model": True,
                "error_type": "A",
                "trend_type": "A" if struct["trend_in"] else "N",
                "season_type": "N",
                "damped": struct["damped"],
            },
            explanatory_checked={
                "xreg_model": False,
                "xreg_number": 0,
                "new_xreg": None,
            },
            components_dict={
                "components_number_ets": n_ets,
                "components_number_ets_seasonal": 0,
                "components_number_arima": struct["n_components"] - n_ets,
            },
            constants_checked={"constant_required": False},
            params_info=[[n_param - n_scale, n_scale, n_param]],
            adam_cpp=best["adam_cpp"],
            interval=interval,
            level=level,
            side=side,
        )
        result.mean = _inverse_like(result.mean, self.lambda_)
        if result.lower is not None:
            result.lower = _inverse_like(result.lower, self.lambda_)
        if result.upper is not None:
            result.upper = _inverse_like(result.upper, self.lambda_)
        return result

    def _forecast_start(self) -> Any:
        index = self._index
        if isinstance(index, pd.DatetimeIndex) and index.freq is not None:
            return index[self.nobs - 1] + index.freq
        return self.nobs

    def _frequency(self) -> Any:
        index = self._index
        if isinstance(index, pd.DatetimeIndex) and index.freq is not None:
            return index.freq
        return "1"

    def summary(self) -> Dict[str, Any]:
        """The main elements of the fitted model."""
        self._check_fitted()
        return {
            "model": self.model_name,
            "coef": dict(zip(self.coef_names, self.coef)),
            "loglik": self.loglik,
            "aicc": self.aicc,
            "scale": self.scale,
            "ics": self.ics,
        }

    def __repr__(self) -> str:
        if hasattr(self, "_best"):
            return f"{self.model_name}, AICc {self.aicc:.3f}"
        return "TBATS (not fitted)"


def _r_round(value: float, digits: int) -> str:
    """A number as R prints round(value, digits)."""
    rounded = round(float(value), digits)
    return str(int(rounded)) if rounded == int(rounded) else repr(rounded)


def _inverse_like(values: Any, lam: float) -> Any:
    """The inverse Box-Cox transform keeping the pandas container."""
    if isinstance(values, (pd.Series, pd.DataFrame)):
        result = values.copy()
        result[:] = st.box_cox_inverse(values.to_numpy(), lam).reshape(values.shape)
        return result
    return st.box_cox_inverse(values, lam)
