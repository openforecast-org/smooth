"""TBATS in the Single Source of Error framework of ADAM (R's ``tbats()``)."""

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from smooth.adam_general.core.forecaster.forecaster import forecaster
from smooth.adam_general.core.forecaster.result import ForecastResult
from smooth.adam_general.core.simulate.result import SimulateResult
from smooth.adam_general.core.tbats import fitter as ft
from smooth.adam_general.core.tbats import structure as st
from smooth.adam_general.core.utils.reapply import ReapplyResult
from smooth.adam_general.core.utils.reforecast import ReforecastResult

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


@dataclass
class TBATSReapplyResult(ReapplyResult):
    """``ReapplyResult`` with the lambda of each draw and its errors, in the space
    of its own transform."""

    lambdas: NDArray
    errors: NDArray


def _refit_one_replicate(
    actuals: NDArray, indices: List[NDArray], kwargs: Dict[str, Any], k: int, i: int
) -> Optional[NDArray]:
    """One bootstrap replicate: the parameters of the refit, or None. Module-level,
    so that joblib can pickle it."""
    try:
        coef = TBATS(**kwargs).fit(actuals[indices[i]]).coef
    except Exception:
        return None
    if coef.shape[0] != k or not np.all(np.isfinite(coef)):
        return None
    return coef


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
    B, lb, ub, maxeval, maxtime, algorithm : optional
        The starting values, bounds and NLopt settings, as in R's ellipsis.
    xtol_rel, xtol_abs, ftol_rel, ftol_abs, print_level : optional
        The tolerances and the print level of NLopt.
    n_iterations, head_length, fi, step_size, shape : optional
        The iterations and head of backcasting, the Fisher Information with its
        step, and the shape of ``dgnorm`` (estimated if None).
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
        **boot_kwargs: Any,
    ) -> pd.DataFrame:
        """The covariance matrix of the parameters, as R's ``vcov.adam``:
        ``"opg"`` (the default) from the scores of the log-densities,
        ``"hessian"`` from the observed Fisher Information, or ``"bootstrap"``
        from :meth:`coefbootstrap` (``boot_kwargs`` go there)."""
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
            return self.coefbootstrap(**boot_kwargs).vcov
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
            "confidence",
            "complete",
        ] = "none",
        level: Union[float, List[float]] = 0.95,
        side: Literal["both", "upper", "lower"] = "both",
        cumulative: bool = False,
        nsim: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> ForecastResult:
        """The forecasts of ADAM's forecaster in the space of the transformed data,
        transformed back: the point forecasts are the medians and the quantiles map
        onto those of the data. ``"confidence"`` and ``"complete"`` take the
        uncertainty of the parameters from :meth:`reforecast` (``nsim`` draws, 100
        by default; ``"simulated"`` uses 10000 paths by default)."""
        self._check_fitted()
        if cumulative and self.lambda_ != 1:
            raise ValueError(
                "Cumulative forecasts of TBATS are only available for lambda=1: the "
                "sums of the transformed values do not transform back."
            )
        if h is None:
            h = self.h if self.h > 0 else 10
        if interval in ("confidence", "complete"):
            return self.reforecast(
                h=h,
                interval="confidence" if interval == "confidence" else "prediction",
                level=level,
                side=side,
                cumulative=cumulative,
                nsim=100 if nsim is None else nsim,
                seed=seed,
            ).to_forecast_result()
        if nsim is None:
            nsim = 10000
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

    def _pull_back(self, parameters: NDArray, point: NDArray) -> NDArray:
        """The point of the segment from the estimates to ``point`` that is the
        furthest from them and satisfies the bounds, by bisection (R's
        ``tbats_pullBack``)."""
        fitter = self._best["fitter"]
        if fitter(point) is not None:
            return point
        inside, outside = 0.0, 1.0
        for _ in range(20):
            share = (inside + outside) / 2
            if fitter(parameters + share * (point - parameters)) is None:
                outside = share
            else:
                inside = share
        return parameters + inside * (point - parameters)

    def reapply(
        self,
        nsim: int = 1000,
        type: Optional[str] = None,  # noqa: A002
        heuristics: Optional[float] = None,
        seed: Optional[int] = None,
        **vcov_kwargs: Any,
    ) -> "TBATSReapplyResult":
        """The refits at parameters drawn from their distribution (R's
        ``reapply.tbats``). Each draw has its own lambda; a draw outside the bounds
        is pulled towards the estimates."""
        from smooth.adam_general.core.utils.reapply import sampling_vcov
        from smooth.adam_general.core.utils.var_covar import resolve_covar_type

        self._check_fitted()
        start_time = time.time()
        covariance_type = resolve_covar_type(type)
        if covariance_type == "bootstrap":
            vcov_kwargs.setdefault("nsim", nsim)
        covariance = sampling_vcov(
            self.vcov(type=covariance_type, heuristics=heuristics, **vcov_kwargs)
        )
        parameters = self.coef
        rng = np.random.default_rng(seed)
        draws = rng.multivariate_normal(parameters, covariance, size=nsim)
        refits = []
        for i in range(nsim):
            draws[i] = self._pull_back(parameters, draws[i])
            refits.append(self._best["fitter"](draws[i]))
        obs = self.nobs
        lag_max = self._best["struct"]["lags_model_max"]
        names = self.component_names
        lambdas = np.array([refit["lambda"] for refit in refits])
        columns = [f"nsim{i}" for i in range(1, nsim + 1)]
        refitted = np.column_stack(
            [st.box_cox_inverse(r["fitted"], lam) for r, lam in zip(refits, lambdas)]
        )
        return TBATSReapplyResult(
            time_elapsed=time.time() - start_time,
            y=pd.Series(self.actuals),
            states=np.stack(
                [r["states"][:, -(obs + lag_max) :] for r in refits], axis=2
            ),
            refitted=pd.DataFrame(refitted, columns=columns),
            fitted=pd.Series(self.fitted),
            model=self.model_name,
            transition=np.stack([r["mat_f"] for r in refits], axis=2),
            measurement=np.stack([np.tile(r["w"], (obs, 1)) for r in refits], axis=2),
            persistence=pd.DataFrame(
                np.column_stack([r["vec_g"] for r in refits]),
                index=names,
                columns=columns,
            ),
            profile=np.stack([r["profile"] for r in refits], axis=2),
            random_parameters=pd.DataFrame(draws, columns=self.coef_names),
            nsim=nsim,
            lambdas=lambdas,
            errors=np.column_stack(
                [
                    st.box_cox(self._y_in_sample, lam) - r["fitted"]
                    for r, lam in zip(refits, lambdas)
                ]
            ),
        )

    def reforecast(
        self,
        h: int = 10,
        interval: Literal["prediction", "confidence", "none"] = "prediction",
        level: Union[float, List[float]] = 0.95,
        side: Literal["both", "upper", "lower"] = "both",
        cumulative: bool = False,
        nsim: int = 100,
        type: Optional[str] = None,  # noqa: A002
        heuristics: Optional[float] = None,
        seed: Optional[int] = None,
        **vcov_kwargs: Any,
    ) -> ReforecastResult:
        """The forecasts with the uncertainty of the parameters (R's
        ``reforecast.tbats``): for each draw of :meth:`reapply`, its point forecasts
        (``"confidence"``) or its paths simulated with the scale of its own errors
        (``"prediction"``), in the space of its own transform and transformed back.
        The point forecast stays the median of the model."""
        from smooth.adam_general.core.adam import (
            _column_names_for_levels,
            _level_bounds,
        )
        from smooth.adam_general.core.creator.architector import adam_profile_creator
        from smooth.adam_general.core.utils.distributions import generate_errors
        from smooth.adam_general.core.utils.utils import scale_debias

        self._check_fitted()
        if cumulative and self.lambda_ != 1:
            raise ValueError(
                "Cumulative forecasts of TBATS are only available for lambda=1."
            )
        levels = list(np.atleast_1d(level).astype(float))
        rng = np.random.default_rng(seed)
        refitted = self.reapply(
            nsim=nsim,
            type=type,
            heuristics=heuristics,
            seed=int(rng.integers(2**31)),
            **vcov_kwargs,
        )
        best = self._best
        struct = best["struct"]
        obs = self.nobs
        lag_max = struct["lags_model_max"]
        n_components = struct["n_components"]
        lookup = adam_profile_creator(struct["lags_model_all"], lag_max, obs + h)[
            "index_lookup_table"
        ][:, obs + lag_max :]
        lookup = np.asfortranarray(lookup, dtype=np.uint64)
        n_scale = int(self.loss == "likelihood")
        df_scale = obs - (self.nparam - n_scale)
        if df_scale <= 0:
            df_scale = obs
        draws = refitted.random_parameters
        adam_cpp = best["adam_cpp"]
        paths = []
        for j in range(nsim):
            mat_wt = np.asfortranarray(np.tile(refitted.measurement[0, :, j], (h, 1)))
            mat_f = np.asfortranarray(refitted.transition[:, :, j])
            profile = np.array(refitted.profile[:, :, j], order="F")
            lam = refitted.lambdas[j]
            if interval == "prediction":
                shape = (
                    draws["shape"].iloc[j]
                    if "shape" in draws
                    else best["elements"]["shape"]
                )
                scale = scale_debias(
                    st.scale_value(refitted.errors[:, j], self.distribution, shape),
                    self.distribution,
                    obs,
                    df_scale,
                )
                errors = generate_errors(
                    self.distribution,
                    h * nsim,
                    scale,
                    obs_in_sample=obs,
                    n_param=obs - df_scale,
                    shape=shape,
                    random_state=rng,
                )
                simulated = adam_cpp.reforecast(
                    np.asfortranarray(np.reshape(errors, (h, nsim, 1), order="F")),
                    np.ones((h, nsim, 1), order="F"),
                    np.asfortranarray(mat_wt.reshape(h, n_components, 1)),
                    np.asfortranarray(mat_f.reshape(n_components, n_components, 1)),
                    np.asfortranarray(refitted.persistence.iloc[:, [j]].to_numpy()),
                    lookup,
                    np.asfortranarray(profile.reshape(n_components, lag_max, 1)),
                    "A",
                ).data
                paths.append(st.box_cox_inverse(np.asarray(simulated)[:, :, 0], lam))
            else:
                point = adam_cpp.forecast(mat_wt, mat_f, lookup, profile, h).forecast
                paths.append(st.box_cox_inverse(np.ravel(point), lam).reshape(h, 1))
        path_matrix = np.column_stack(paths)
        if cumulative:
            path_matrix = path_matrix.sum(axis=0, keepdims=True)

        point_forecast = self.predict(h=h).mean
        mean = point_forecast
        if cumulative:
            mean = pd.Series([point_forecast.sum()], index=point_forecast.index[:1])
        if interval == "none":
            lower = upper = None
        else:
            level_low, level_up = _level_bounds(levels, side, path_matrix.shape[0])
            lower_cols, upper_cols = _column_names_for_levels(levels, side)
            lower_values = np.array(
                [np.nanquantile(row, q) for row, q in zip(path_matrix, level_low)]
            )
            upper_values = np.array(
                [np.nanquantile(row, q) for row, q in zip(path_matrix, level_up)]
            )
            lower_values[level_low == 0] = 0
            upper_values[level_up == 1] = np.inf
            lower = pd.DataFrame(lower_values, index=mean.index, columns=lower_cols)
            upper = pd.DataFrame(upper_values, index=mean.index, columns=upper_cols)
        return ReforecastResult(
            mean=mean,
            lower=lower,
            upper=upper,
            level=levels,
            interval=interval,
            side=side,
            cumulative=cumulative,
            h=h,
            paths=path_matrix,
            model=self.model_name,
        )

    def simulate(
        self, nsim: int = 1, seed: Optional[int] = None, obs: Optional[int] = None
    ) -> SimulateResult:
        """Series simulated from the model in the space of the transformed data,
        transformed back (R's ``simulate.tbats``), starting from its initials."""
        from smooth.adam_general._adam_general import adam_simulator
        from smooth.adam_general.core.creator.architector import adam_profile_creator
        from smooth.adam_general.core.utils.distributions import generate_errors
        from smooth.adam_general.core.utils.utils import scale_debias

        self._check_fitted()
        best = self._best
        struct = best["struct"]
        obs = self.nobs if obs is None else int(obs)
        lag_max = struct["lags_model_max"]
        n_ets = struct["n_ets"]
        n_scale = int(self.loss == "likelihood")
        df_scale = max(self.nobs - (self.nparam - n_scale), 1)
        rng = np.random.default_rng(seed)
        scale = scale_debias(self.scale, self.distribution, self.nobs, df_scale)
        errors = np.reshape(
            generate_errors(
                self.distribution,
                obs * nsim,
                scale,
                obs_in_sample=self.nobs,
                n_param=self.nobs - df_scale,
                shape=best["elements"]["shape"],
                random_state=rng,
            ),
            (obs, nsim),
            order="F",
        )
        profile = best["fitted"]["profile_initial"]
        array_vt = np.zeros((struct["n_components"], obs + lag_max, nsim), order="F")
        array_vt[:, :lag_max, :] = profile[:, :, None]
        lookup = adam_profile_creator(struct["lags_model_all"], lag_max, obs)[
            "index_lookup_table"
        ]
        result = adam_simulator(
            matrixErrors=errors,
            matrixOt=np.ones((obs, nsim)),
            arrayVt=array_vt,
            matrixWt=np.tile(best["elements"]["w"], (obs, 1)),
            arrayF=np.repeat(best["elements"]["mat_f"][:, :, None], nsim, axis=2),
            matrixG=np.repeat(best["elements"]["vec_g"][:, None], nsim, axis=1),
            lags=np.asarray(struct["lags_model_all"], dtype=np.uint64),
            indexLookupTable=lookup,
            profilesRecent=np.repeat(profile[:, :, None], nsim, axis=2),
            E="A",
            T="A" if struct["trend_in"] else "N",
            S="N",
            nNonSeasonal=n_ets,
            nSeasonal=0,
            nArima=struct["n_components"] - n_ets,
            nXreg=0,
            constant=False,
        )
        data = st.box_cox_inverse(np.asarray(result["matrixYt"]), self.lambda_)
        data = data.reshape(obs, nsim)
        return SimulateResult(
            model=self.model_name,
            data=pd.Series(data[:, 0]) if nsim == 1 else pd.DataFrame(data),
            states=np.asarray(result["arrayVt"]).reshape(array_vt.shape, order="F"),
            residuals=pd.Series(errors[:, 0]) if nsim == 1 else pd.DataFrame(errors),
            persistence=best["elements"]["vec_g"].reshape(-1, 1),
            measurement=np.tile(best["elements"]["w"], (obs, 1)),
            transition=best["elements"]["mat_f"].copy(),
            initial=profile.copy(),
            probability=np.ones(obs),
            occurrence=None,
            profile=profile.copy(),
            other={"shape": best["elements"]["shape"]}
            if self.distribution == "dgnorm"
            else {},
        )

    def confint(
        self,
        parm: Optional[Any] = None,
        level: float = 0.95,
        type: Optional[str] = None,  # noqa: A002
        step_size: Optional[float] = None,
        **boot_kwargs: Any,
    ) -> pd.DataFrame:
        """The confidence intervals of the parameters (R's ``confint.adam``): the
        bounds from :meth:`vcov` moved inside the bounds of the model, one
        parameter at a time; the quantiles of :meth:`coefbootstrap` with
        ``type="bootstrap"``."""
        from scipy import stats as scipy_stats

        from smooth.adam_general.core.utils.bootstrap import bootstrap_confint_frame
        from smooth.adam_general.core.utils.var_covar import resolve_covar_type

        self._check_fitted()
        names = self.coef_names
        parameters = self.coef
        covariance_type = resolve_covar_type(type)
        if covariance_type == "bootstrap":
            boot = self.coefbootstrap(**boot_kwargs)
            return bootstrap_confint_frame(boot, names, parameters, level, parm)
        se = np.sqrt(
            np.abs(np.diag(self.vcov(type=covariance_type, step_size=step_size)))
        )
        bounds = np.column_stack(
            [
                scipy_stats.t.ppf((1 - level) / 2, df=self.nobs - self.nparam) * se,
                scipy_stats.t.ppf((1 + level) / 2, df=self.nobs + self.nparam) * se,
            ]
        )
        for j in range(len(parameters)):
            for side in range(2):
                point = parameters.copy()
                point[j] += bounds[j, side]
                bounds[j, side] = self._pull_back(parameters, point)[j] - parameters[j]
        result = pd.DataFrame(
            np.column_stack([se, bounds + parameters[:, None]]),
            index=names,
            columns=[
                "S.E.",
                f"{(1 - level) / 2 * 100:g}%",
                f"{(1 + level) / 2 * 100:g}%",
            ],
        )
        if parm is not None:
            result = result.loc[parm if isinstance(parm, (list, tuple)) else [parm]]
        return result

    def _refit_kwargs(self) -> Dict[str, Any]:
        """The arguments refitting the model with its structure, starting from its
        parameters without bounds (R's ``tbats_refitCall``)."""
        spec = self._best["spec"]
        lags = [int(math.trunc(lag)) for lag in self.lags]
        position = {lag: k for k, lag in enumerate(spec["lags"].tolist())}
        B = self.coef
        kwargs: Dict[str, Any] = {
            "lags": self.lags,
            "harmonics": self.harmonics_,
            "trend": self.trend_type_,
            "lambda_bc": None if "lambda" in self.coef_names else self.lambda_,
            "orders": {
                "ar": [
                    int(spec["ar_orders"][position[lag]]) if lag in position else 0
                    for lag in lags
                ],
                "ma": [
                    int(spec["ma_orders"][position[lag]]) if lag in position else 0
                    for lag in lags
                ],
                "select": False,
            },
            "distribution": self.distribution,
            "loss": self.loss,
            "initial": self.initial,
            "bounds": self.bounds,
            "B": B,
            "lb": np.full(len(B), -np.inf),
            "ub": np.full(len(B), np.inf),
        }
        if self.distribution == "dgnorm" and "shape" not in self.coef_names:
            kwargs["shape"] = self._best["elements"]["shape"]
        return kwargs

    def coefbootstrap(
        self,
        nsim: int = 1000,
        parallel: Union[bool, int] = False,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> Any:
        """The bootstrap of the parameters (R's ``coefbootstrap.adam``): refits of
        the model with its structure on contiguous subsamples of random length,
        starting at random origins with backcasting."""
        from functools import partial

        from smooth.adam_general.core.utils.bootstrap import (
            _build_result,
            run_replicates,
            time_series_sample_indices,
        )

        self._check_fitted()
        names = self.coef_names
        obs_minimum = int(max(max(self.lags), len(names))) + 2
        indices = time_series_sample_indices(
            self.nobs,
            nsim,
            obs_minimum,
            self.initial in ("backcasting", "complete"),
            np.random.default_rng(seed),
        )
        worker = partial(
            _refit_one_replicate,
            self.actuals,
            indices,
            self._refit_kwargs(),
            len(names),
        )
        start_time = time.time()
        coefficients, parallel_used = run_replicates(
            worker, nsim=nsim, parallel=parallel, verbose=verbose, label="coefbootstrap"
        )
        return _build_result(
            coefficients,
            names,
            method="cr",
            nsim=nsim,
            size=0,
            replace=False,
            prob=None,
            parallel=parallel_used,
            model=self.model_name,
            time_elapsed=time.time() - start_time,
        )

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
