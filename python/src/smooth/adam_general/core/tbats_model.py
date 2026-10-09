"""TBATS in the Single Source of Error framework of ADAM (R's ``tbats()``)."""

import math
import time
import warnings
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import special

from smooth.adam_general.core.adam import ADAM
from smooth.adam_general.core.checker.data_checks import _warn_missing
from smooth.adam_general.core.forecaster.forecaster import forecaster
from smooth.adam_general.core.forecaster.result import ForecastResult
from smooth.adam_general.core.simulate.result import SimulateResult
from smooth.adam_general.core.tbats import fitter as ft
from smooth.adam_general.core.tbats import structure as st
from smooth.adam_general.core.utils.n_param import NParam
from smooth.adam_general.core.utils.reapply import ReapplyResult
from smooth.adam_general.core.utils.reforecast import ReforecastResult
from smooth.adam_general.core.utils.utils import (
    OUTLIER_NAMES,
    _exp_r,
    _log_r,
    _sum_r,
    scale_debias,
    xreg_selector,
)

TREND_OPTIONS = ("auto", "none", "additive", "damped")
DISTRIBUTION_OPTIONS = ("auto", "dnorm", "dlaplace", "ds", "dgnorm")
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
INITIAL_OPTIONS = ("backcasting", "optimal", "two-stage", "complete", "gradient")
# The settings of the optimiser, the keys of ADAM's nlopt_kwargs
_NLOPT_DEFAULTS: Dict[str, Any] = {
    "B": None,
    "lb": None,
    "ub": None,
    "maxeval": None,
    "maxtime": -1,
    "algorithm": "NLOPT_LN_NELDERMEAD",
    "xtol_rel": 1e-6,
    "xtol_abs": 1e-8,
    "ftol_rel": 1e-8,
    "ftol_abs": 0,
    "print_level": 0,
}
BOUNDS_OPTIONS = ("admissible", "usual", "none")
REGRESSORS_OPTIONS = ("use", "select", "adapt")
OUTLIERS_OPTIONS = ("ignore", "use", "select")
OCCURRENCE_OPTIONS = (
    "none",
    "auto",
    "fixed",
    "general",
    "odds-ratio",
    "inverse-odds-ratio",
    "direct",
)
MULTISTEP_LOSSES = ("MSEh", "TMSE", "GTMSE", "MSCE", "GPL")


@dataclass
class TBATSReapplyResult(ReapplyResult):
    """``ReapplyResult`` with the lambda of each draw and its errors, in the space
    of its own transform."""

    lambdas: NDArray
    errors: NDArray


def _named_b(fitted: Dict[str, Any]) -> Dict[str, float]:
    """The parameters of a fit by name, the warm start of a related one."""
    return dict(zip(fitted["names"], fitted["B"]))


def _refit_one_replicate(
    actuals: NDArray,
    X: Optional[NDArray],
    indices: List[NDArray],
    kwargs: Dict[str, Any],
    k: int,
    i: int,
) -> Optional[NDArray]:
    """One bootstrap replicate: the parameters of the refit, or None. Module-level,
    so that joblib can pickle it."""
    try:
        X_i = None if X is None else X[indices[i]]
        coef = TBATS(**kwargs).fit(actuals[indices[i]], X_i).coef
    except Exception:
        return None
    if coef.shape[0] != k or not np.all(np.isfinite(coef)):
        return None
    return coef


def _occurrence_spec(occurrence: Any, y: NDArray, loss: str) -> Dict[str, Any]:
    """The occurrence of an intermittent demand (R's ``tbats_occurrence``): a fitted
    ``OM`` / ``OMG``, the provided probabilities (or 0/1), or an ``OM`` with the level
    only (and the trend selected) for the occurrence type, as its seasonal pattern is
    hard to find in zeros and ones. The non-zero observations, and the log-likelihood
    and the parameters of the occurrence, which are added to those of the sizes."""
    from smooth.adam_general.core.om import OM
    from smooth.adam_general.core.omg import OMG

    obs = len(y)
    # The missing values are neither zeros nor demand: ot_logical marks the observed
    # (and non-zero) values that the sizes are fitted to
    observed = ~np.isnan(y)
    ot_logical = observed & (y != 0)
    none = {
        "model": None,
        "ot_logical": None if np.all(observed) else observed,
        "loglik": 0.0,
        "n_param": 0,
        "p_fitted": np.ones(obs),
    }
    if isinstance(occurrence, (OM, OMG)):
        model = occurrence
        if len(np.ravel(model.fitted)) != obs:
            raise ValueError(
                "The occurrence model should be fitted to the in-sample data."
            )
    elif not isinstance(occurrence, str):
        probabilities = np.asarray(occurrence, dtype=float).ravel()
        if len(probabilities) < obs or np.any(
            (probabilities < 0) | (probabilities > 1)
        ):
            raise ValueError(
                "The provided occurrence should have the probabilities (in [0, 1]) of "
                "the in-sample observations, and possibly of the horizon."
            )
        p_fitted = probabilities[:obs]
        zero = ~ot_logical & observed
        if np.any(p_fitted[ot_logical] == 0) or np.any(p_fitted[zero] == 1):
            raise ValueError("The provided occurrence contradicts the data.")
        loglik = _sum_r(_log_r(p_fitted[ot_logical])) + _sum_r(
            _log_r(1 - p_fitted[zero])
        )
        provided = {
            "occurrence": "provided",
            "fitted": p_fitted,
            "forecast": probabilities[obs:],
        }
        return {
            "model": provided,
            "ot_logical": ot_logical,
            "loglik": loglik,
            "n_param": 0,
            "p_fitted": p_fitted,
        }
    else:
        _match(occurrence, OCCURRENCE_OPTIONS, "occurrence")
        if occurrence == "none":
            if np.any(y == 0):
                warnings.warn(
                    "The data has zeros, which are fitted as values. For an "
                    "intermittent demand, use the occurrence argument.",
                    stacklevel=3,
                )
            return none
        # No zeros: nothing to model
        if np.all(ot_logical[observed]):
            return none
        # fit() returns the selected model for occurrence="auto"
        model = OM(model="ZXN", lags=[1], occurrence=occurrence).fit(y)
    if loss in MULTISTEP_LOSSES:
        raise ValueError(
            "The multistep losses are not available with an occurrence model."
        )
    return {
        "model": model,
        "ot_logical": ot_logical,
        "loglik": float(model.loglik),
        "n_param": model.nparam,
        "p_fitted": np.asarray(model.fitted, dtype=float).ravel(),
    }


def _p_forecast(model: Any, h: int) -> NDArray:
    """The probabilities of occurrence for the horizon (R's ``tbats_pForecast``):
    forecasts of the occurrence model, or the provided ones (the last of them
    repeated)."""
    if model is None or h <= 0:
        return np.ones(max(h, 0))
    if not isinstance(model, dict):
        return np.asarray(model.predict(h=h).mean, dtype=float)
    known = np.concatenate([model["fitted"], model["forecast"]])
    return np.concatenate([model["forecast"], np.repeat(known[-1], h)])[:h]


def _occurrence_draws(paths: NDArray, p_forecast: NDArray, rng: Any) -> NDArray:
    """The paths with the occurrence drawn with its probabilities (R's
    ``tbats_occurrenceDraws``), unchanged without occurrence."""
    if np.any(p_forecast < 1):
        paths = paths * rng.binomial(
            1, np.repeat(p_forecast[:, None], paths.shape[1], axis=1)
        )
    return paths


def _paths_bounds(paths: NDArray, level: Any, side: str) -> Tuple[NDArray, NDArray]:
    """The bounds of the paths, a row per horizon (R's ``tbats_pathsBounds``):
    their quantiles at the levels of the side, zero and Inf at the levels 0 and
    1."""
    from smooth.adam_general.core.adam import _level_bounds

    levels = np.atleast_1d(np.asarray(level, dtype=float))
    levels = np.where(levels > 1, levels / 100, levels)
    level_low, level_up = _level_bounds(levels, side, paths.shape[0])
    lower = np.array([np.nanquantile(row, q) for row, q in zip(paths, level_low)])
    upper = np.array([np.nanquantile(row, q) for row, q in zip(paths, level_up)])
    lower[level_low == 0] = 0
    upper[level_up == 1] = np.inf
    return lower, upper


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
    regressors : str, default="use"
        How to treat the explanatory variables ``X`` of :meth:`fit`: ``"use"`` them
        as they are (constant coefficients), ``"select"`` them as ADAM does
        (``stepwise()`` on the errors of the model chosen without them, which is
        refitted with the selected ones and kept if it improves the IC), or
        ``"adapt"`` their coefficients over time (``delta1``, ...).
    occurrence : str, OM, OMG or array, default="none"
        The occurrence of an intermittent demand. ``"none"`` fits the zeros as values,
        with a warning. A fitted ``OM`` / ``OMG`` is used as it is, and an array gives
        the probabilities of occurrence (or 0/1) of the in-sample observations, and
        possibly of the horizon. ``"fixed"``, ``"auto"``, ``"odds-ratio"``,
        ``"inverse-odds-ratio"``, ``"direct"`` and ``"general"`` fit
        ``OM(model="ZXN", lags=[1])`` of that type: a level-only occurrence with the
        trend selected, as the seasonal pattern of the probability is hard to find in
        zeros and ones (a seasonal one can be provided as a fitted ``OM``). The sizes
        are then modelled on the non-zero observations: the Box-Cox transform, its
        Jacobian and the global model use them only, and the states evolve through
        the zeros. The log-likelihood and the number of parameters include those of
        the occurrence model, and the fitted values and forecasts are the probability
        times those of the sizes (see ``point`` in :meth:`predict`).
    distribution : str, default="auto"
        ``"auto"``, ``"dnorm"``, ``"dlaplace"``, ``"ds"`` or ``"dgnorm"``, in the
        space of the transformed data. With ``"auto"``, the model is selected with
        ``"dgnorm"``, whose shape nests the others (2 is the normal, 1 the Laplace and
        0.5 the S distribution), and the named distribution closest to the estimated
        shape on the log scale is then fitted on the selected structure (from the
        default starting values and from the estimates of ``"dgnorm"``, the higher
        likelihood kept); its information criterion is added to ``ICs``. With a loss
        other than the likelihood, ``"auto"`` is ``"dlaplace"`` for ``"MAE"``,
        ``"ds"`` for ``"HAM"`` and ``"dnorm"`` otherwise, as in ``ADAM``. The
        distribution used is ``distribution_``.
    outliers : str, default="ignore"
        What to do with the outliers, as ``ADAM``: ``"ignore"`` them, ``"use"`` a
        dummy variable for each, or ``"select"`` among the dummies and their leads and
        lags. They are found on the residuals of the global model (the regression on
        the trend, the harmonics and the regressors, at the starting value of lambda
        and on the non-zero observations), outside the ``outliers_level`` quantiles of
        the distribution (of ``"dgnorm"`` with the shape estimated on the residuals for
        ``"auto"``), so no fit is added. With ``"select"``, ``stepwise()`` chooses the
        dummies on these residuals. The dummies join the regressors in all the fits,
        with zeros over the horizon, and ``regressors`` becomes ``"use"`` or
        ``"select"`` (for the regressors of ``X``), as in R's ``auto.adam()``.
    outliers_level : float, default=0.99
        The confidence level of the detection of the outliers (R's ``level``).
    loss, ic, h, holdout, bounds
        As in R's ``tbats()``; ``bounds="admissible"`` keeps the model stable.
    persistence : dict, array or None
        The smoothing parameters, as in ``ADAM``: a vector (level, trend, a pair
        gamma1, gamma2 per period of ``lags`` above 1, the smoothing parameters of the
        regressors with ``regressors="adapt"``), used only when the structure is not
        selected, or a dict with ``"level"``, ``"trend"``, ``"seasonal"`` (a list
        with one pair ``[gamma1, gamma2]`` per period) and ``"xreg"`` (or ``"alpha"``,
        ``"beta"``, ``"gamma"``), where only the provided elements are fixed. The
        values for the components that the model does not have are ignored.
    phi : float or None
        The damping parameter, used with ``trend="damped"``. With ``trend="auto"``,
        it is estimated with a warning, as in ``ADAM``.
    initial : str, dict or array, default="backcasting"
        The initialisation, as in R's ``tbats()``, or the initial states in the space
        of the Box-Cox transformed data (meaningful with a provided ``lambda_bc``): a
        vector (level, trend, then for each period the sine coefficients of its
        harmonics followed by their cosine coefficients, the ARMA states, the
        coefficients of the regressors), used only when the structure is not
        selected, or a dict with ``"level"``, ``"trend"``, ``"seasonal"`` (a list
        with one array of the sine and then the cosine coefficients per period, which
        needs ``harmonics``), ``"arma"`` and ``"xreg"`` (an array, or a dict by
        name). The states not provided are estimated, with ``"optimal"``
        initialisation, and ``initial_type`` is ``"provided"``.
    arma : dict, array or None
        The parameters of the ARMA, as in ``ADAM``: a dict with ``"ar"`` and
        ``"ma"`` (either can be left out to estimate it), or a vector with the AR and
        then the MA parameters of each lag, lag by lag. If provided, the orders are
        not selected.
    verbose : int, default=0
        Not used yet (R's ``silent``).
    nlopt_kwargs : dict or None
        The settings of the optimiser, as ``ADAM``'s: ``"B"``, ``"lb"``, ``"ub"``
        (the starting values and bounds of the parameters, as R's ellipsis),
        ``"maxeval"``, ``"maxtime"``, ``"algorithm"``, ``"xtol_rel"``,
        ``"xtol_abs"``, ``"ftol_rel"``, ``"ftol_abs"`` and ``"print_level"``.
    n_iterations, head_length, fi, step_size : optional
        The iterations and head of backcasting, the Fisher Information with its
        step, as ``ADAM``.
    gnorm_shape : float or None
        The shape of ``dgnorm``, as ``ADAM``; estimated if None.
    """

    def __init__(
        self,
        lags: Optional[List[float]] = None,
        harmonics: Optional[List[int]] = None,
        trend: str = "auto",
        lambda_bc: Optional[float] = None,
        orders: Optional[Dict[str, Any]] = None,
        regressors: str = "use",
        occurrence: Any = "none",
        distribution: str = "auto",
        loss: Union[str, Callable[..., float]] = "likelihood",
        outliers: str = "ignore",
        outliers_level: float = 0.99,
        ic: str = "AICc",
        h: int = 0,
        holdout: bool = False,
        persistence: Optional[Union[Dict[str, Any], List[float], NDArray]] = None,
        phi: Optional[float] = None,
        initial: Union[str, Dict[str, Any], List[float], NDArray] = "backcasting",
        arma: Optional[Union[Dict[str, Any], List[float], NDArray]] = None,
        bounds: str = "admissible",
        verbose: int = 0,
        nlopt_kwargs: Optional[Dict[str, Any]] = None,
        n_iterations: Optional[int] = None,
        head_length: Optional[int] = None,
        fi: bool = False,
        step_size: Optional[float] = None,
        gnorm_shape: Optional[float] = None,
    ) -> None:
        self.lags = [1.0] if lags is None else [float(lag) for lag in lags]
        self.harmonics = None if harmonics is None else [int(k) for k in harmonics]
        self.trend = _match(trend, TREND_OPTIONS, "trend")
        self.lambda_bc = lambda_bc
        default_orders = {"ar": 3, "ma": 3, "select": True}
        self._init_orders = default_orders if orders is None else orders
        self.regressors = _match(regressors, REGRESSORS_OPTIONS, "regressors")
        self.outliers = _match(outliers, OUTLIERS_OPTIONS, "outliers")
        self.outliers_level = outliers_level
        if isinstance(occurrence, str):
            _match(occurrence, OCCURRENCE_OPTIONS, "occurrence")
        self.occurrence = occurrence
        self.distribution = _match(distribution, DISTRIBUTION_OPTIONS, "distribution")
        # A function is a custom loss of actual, fitted and B, as in ADAM
        self.loss_function = loss if callable(loss) else None
        self.loss = "custom" if callable(loss) else _match(loss, LOSS_OPTIONS, "loss")
        self.ic = _match(ic, IC_OPTIONS, "ic")
        self.h = int(h)
        self.holdout = holdout
        self.persistence = persistence
        self.phi = phi
        self.initial = initial
        self.arma = arma
        # The provided initials are estimated where they are missing, as in ADAM
        self._initial_method = (
            _match(initial, INITIAL_OPTIONS, "initial")
            if isinstance(initial, str)
            else "optimal"
        )
        # The gradient solve cannot take a custom loss: the initials are backcast
        if self._initial_method == "gradient" and self.loss == "custom":
            warnings.warn(
                'initial="gradient" is not available for custom loss functions. '
                'Switching to initial="backcasting".',
                stacklevel=2,
            )
            self._initial_method = "backcasting"
            if isinstance(self.initial, str):
                self.initial = "backcasting"
        self.bounds = _match(bounds, BOUNDS_OPTIONS, "bounds")
        self.verbose = verbose
        self.nlopt_kwargs = nlopt_kwargs
        unknown = set(nlopt_kwargs or {}) - set(_NLOPT_DEFAULTS)
        if unknown:
            raise ValueError(
                f"Unknown nlopt_kwargs of TBATS: {', '.join(sorted(unknown))}. "
                f"Accepted: {', '.join(_NLOPT_DEFAULTS)}."
            )
        self.n_iterations = n_iterations
        self.head_length = head_length
        self.fi = fi
        self.step_size = (
            float(np.finfo(float).eps ** 0.25) if step_size is None else step_size
        )
        self.gnorm_shape = gnorm_shape

    # Set by fit()
    _best: Dict[str, Any]
    # Set by sm() on a scale model, as ADAM's
    is_scale_: bool
    loglik_sm_: float
    df_sm_: int
    location_: "TBATS"
    harmonics_: List[int]
    trend_type_: str
    lambda_: float

    # Fitting
    def _settings(self) -> Dict[str, Any]:
        """R's ``checked``: the settings of the fitter."""
        n_iterations = self.n_iterations
        if n_iterations is None:
            n_iterations = (
                2
                if self._initial_method in ("backcasting", "complete", "gradient")
                else 1
            )
        return {
            "h": self.h,
            "loss": self.loss,
            "loss_function": self.loss_function,
            "bounds": self.bounds,
            "model_do": "estimate",
            **_NLOPT_DEFAULTS,
            **(self.nlopt_kwargs or {}),
            "n_iterations": n_iterations,
            "head_length": self.head_length,
            "fi": self.fi,
            "step_size": self.step_size,
            "shape": 2.0 if self.gnorm_shape is None else float(self.gnorm_shape),
            "shape_estimate": self.gnorm_shape is None,
        }

    def fit(self, y: Union[NDArray, pd.Series], X: Optional[Any] = None) -> "TBATS":
        """Fit the model to the series ``y``, with the explanatory variables ``X``
        (a numeric array or data frame with the rows of ``y``, and of the horizon
        ``h`` for its forecasts) in the space of the transformed data.

        The missing values (NaN) of ``y`` are gaps, as in R: the global model is
        fitted to the observed values, the states move through the transition
        without an update at the gaps, the likelihood and the information criteria
        count the observed values only, and the ARMA screen takes zeros at the gaps
        of the residuals. The residuals are NaN there, and the fitted values are the
        predictions of the model. The observations with a missing value of a
        regressor are dropped (gaps of ``y``, with a warning), and their fitted values
        are NaN; the future values of the regressors cannot be missing."""
        start_time = time.time()
        index = y.index if isinstance(y, pd.Series) else None
        values = np.asarray(y, dtype=float).ravel()
        h = self.h
        obs_in_sample = len(values) - h if (self.holdout and h > 0) else len(values)
        y_in_sample = values[:obs_in_sample]
        self._y_holdout = values[obs_in_sample:] if (self.holdout and h > 0) else None
        # The missing values are gaps: the global model, the fit and the likelihood
        # use the observed values only
        missing = np.isnan(values)
        if np.any(missing):
            _warn_missing(missing, obs_in_sample, stacklevel=3)
        self._index = index

        # The outliers make the regressors used or selected, as in R's auto.adam()
        regressors = self.regressors if self.outliers == "ignore" else self.outliers
        xreg = st.xreg_spec(X, obs_in_sample, h, regressors)
        # The regressors with the names of the dummies of the outliers are renamed
        if self.outliers != "ignore" and xreg is not None:
            clashes = [n for n in xreg["names"] if OUTLIER_NAMES.match(n)]
            if clashes:
                warnings.warn(
                    f"The names of the regressors {', '.join(clashes)} are those of "
                    "the dummies of the outliers. Renaming them to "
                    f"{', '.join('x.' + c for c in clashes)}.",
                    stacklevel=2,
                )
                xreg["names"] = [f"x.{n}" if n in clashes else n for n in xreg["names"]]
        if xreg is not None:
            y_in_sample = np.where(xreg["missing"], np.nan, y_in_sample)

        lags, harmonics, trend = self.lags, self.harmonics, self.trend
        occurrence = _occurrence_spec(self.occurrence, y_in_sample, self.loss)
        ot = occurrence["ot_logical"]
        if ot is None:
            ot = np.ones(obs_in_sample, dtype=bool)

        periods = sorted({lag for lag in lags if lag > 1})
        lam_spec = st.lambda_spec(self.lambda_bc, y_in_sample[ot], self.loss)
        # The provided values, as ADAM takes them: phi needs a preselected trend, and
        # an unnamed vector needs a preselected structure to be matched with it
        orders = dict(self._init_orders)
        persistence, phi, arma = self.persistence, self.phi, self.arma
        initial = None if isinstance(self.initial, str) else self.initial
        selection = (
            trend == "auto"
            or harmonics is None
            or bool(orders.get("select"))
            or regressors == "select"
        )
        if phi is not None and trend == "auto":
            warnings.warn(
                "Predefined phi can only be used with a preselected trend. Changing "
                "to estimation.",
                stacklevel=2,
            )
            phi = None
        if persistence is not None and not isinstance(persistence, dict) and selection:
            warnings.warn(
                "Predefined persistence vector can only be used with a preselected "
                "structure.\nChanging to estimation of persistence values.",
                stacklevel=2,
            )
            persistence = None
        if initial is not None and not isinstance(initial, dict) and selection:
            warnings.warn(
                "Predefined initials vector can only be used with a preselected "
                "structure.\nChanging to estimation of initials.",
                stacklevel=2,
            )
            initial = None
        if isinstance(initial, dict):
            initial = dict(initial)
            for key, unknown, message in (
                (
                    "seasonal",
                    harmonics is None,
                    "Initial seasonal coefficients need the harmonics to be provided.",
                ),
                (
                    "arma",
                    bool(orders.get("select")) and arma is None,
                    "Initial ARMA states need the orders to be preselected.",
                ),
                (
                    "xreg",
                    regressors == "select",
                    "Initial values of the regressors cannot be used with their "
                    "selection.",
                ),
            ):
                if initial.get(key) is not None and unknown:
                    warnings.warn(f"{message} Estimating them.", stacklevel=2)
                    initial.pop(key)
        # The ARMA parameters fix its orders
        if arma is not None:
            orders["select"] = False
        spec = st.arma_spec(orders, lags)
        # The provided AR or MA parameters of a wrong number are estimated, as in ADAM
        if arma is not None:
            kinds: Dict[str, Any] = (
                dict(arma) if isinstance(arma, dict) else {"arma": arma}
            )
            needed = {
                "ar": int(np.sum(spec["ar_orders"])),
                "ma": int(np.sum(spec["ma_orders"])),
                "arma": spec["n_param"],
            }
            for kind in [k for k in needed if kinds.get(k) is not None]:
                count = len(np.atleast_1d(kinds[kind]))
                if count != needed[kind]:
                    warnings.warn(
                        f"The number of provided {kind.upper()} parameters is {count}, "
                        f"while the orders imply {needed[kind]}. Estimating them.",
                        stacklevel=2,
                    )
                    kinds[kind] = None
            arma = kinds if isinstance(arma, dict) else kinds["arma"]
        self._provided = {
            "persistence": persistence,
            "phi": phi,
            "arma": arma,
            "initial": initial,
        }
        spec_fit = st.arma_build([0], [0], [1]) if spec["select"] else spec
        trend_types = ["none", "additive", "damped"] if trend == "auto" else [trend]
        settings = {
            **self._settings(),
            "occurrence": occurrence,
            "provided": self._provided,
        }
        distribution = self._distribution_selection()
        # The regressors are selected on the errors of the model without them
        xreg_fit = None if regressors == "select" else xreg

        if harmonics is None:
            harmonics = st.harmonics_select(
                y_in_sample,
                periods,
                any(t != "none" for t in trend_types),
                lam_spec,
                self.ic,
                None if xreg_fit is None else xreg_fit["data"],
                ot,
            )
        else:
            if len(harmonics) != len(periods):
                raise ValueError("harmonics should have one value per lag above 1.")
            k_max = [math.ceil(p / 2) - 1 for p in periods]
            if any(k > m for k, m in zip(harmonics, k_max)):
                warnings.warn(
                    "The number of harmonics has to be below half of the period. "
                    "Reducing it.",
                    stacklevel=2,
                )
                harmonics = [min(k, m) for k, m in zip(harmonics, k_max)]
        table = st.harmonics_table(periods, harmonics)

        # The dummies of the outliers of the global model join the regressors, in all
        # the fits
        outlier_names: List[str] = []
        if self.outliers != "ignore":
            dummies = st.outlier_dummies(
                y_in_sample,
                ot,
                any(t != "none" for t in trend_types),
                table,
                lam_spec,
                None if xreg_fit is None else xreg_fit["data"],
                distribution,
                self.outliers_level,
                self.outliers,
                self.ic,
                h,
            )
            if dummies is not None:
                outlier_names = dummies["names"]
                data = dummies["data"]
                n = len(y_in_sample)
                user = [] if xreg is None else [xreg]
                future = [u["future"] for u in user] + [data[n:]]
                names = [name for u in user for name in u["names"]] + outlier_names
                xreg = {
                    "data": np.column_stack([u["data"] for u in user] + [data[:n]]),
                    "future": np.column_stack(future) if h > 0 else None,
                    "names": names,
                    "number": len(names),
                    "regressors": regressors,
                }
                xreg_fit = st.xreg_subset(
                    xreg, outlier_names if regressors == "select" else xreg["names"]
                )

        # Fit the candidates and select. The trends are warm started from the model
        # without it, with no trend smoothing
        candidates: List[Dict[str, Any]] = []
        for t in trend_types:
            candidates.append(
                ft.fit(
                    y_in_sample,
                    t,
                    table,
                    spec_fit,
                    lam_spec,
                    distribution,
                    self._initial_method,
                    settings,
                    xreg_fit,
                    {**_named_b(candidates[0]), "beta": 0.0} if candidates else None,
                )
            )
        ics: Dict[str, float] = {
            t: self._ic(c) for t, c in zip(trend_types, candidates)
        }
        best = candidates[int(np.argmin(list(ics.values())))]

        if regressors == "select" and xreg is not None:
            best, xreg_fit = self._select_xreg(
                best,
                xreg,
                outlier_names,
                xreg_fit,
                ics,
                y_in_sample,
                table,
                spec_fit,
                lam_spec,
                settings,
                ot,
                distribution,
            )

        # The ARMA orders, screened on the residuals of the global model
        if spec["select"] and spec["n_param"] > 0:
            design = st.design(
                len(y_in_sample),
                best["trend_type"] != "none",
                table,
                None if xreg_fit is None else xreg_fit["data"],
            )[ot]
            residuals = st.QR(design).resid(
                st.box_cox(y_in_sample[ot], best["elements"]["lambda"])
            )
            spec_best = st.arma_select(
                st.gapped(residuals, ot),
                spec,
                distribution,
                best["elements"]["shape"],
                best["n_param_estimated"],
                self.ic,
                ot,
            )
            if spec_best["n_param"] > 0:
                # Warm started from the best model, with no ARMA
                no_arma = dict.fromkeys(spec_best["names"], 0.0)
                candidate = ft.fit(
                    y_in_sample,
                    best["trend_type"],
                    table,
                    spec_best,
                    lam_spec,
                    distribution,
                    self._initial_method,
                    settings,
                    xreg_fit,
                    {**_named_b(best), **no_arma},
                )
                ic_candidate = self._ic(candidate)
                if ic_candidate < min(ics.values()):
                    best = candidate
                ar = ",".join(str(o) for o in spec_best["ar_orders"])
                ma = ",".join(str(o) for o in spec_best["ma_orders"])
                ics[f"{candidate['trend_type']}+ARMA({ar};{ma})"] = ic_candidate

        if self.distribution == "auto" and distribution == "dgnorm":
            best = self._closest(best, y_in_sample, table, lam_spec, settings, xreg_fit)
            ics[best["distribution"]] = self._ic(best)

        self._best = best
        self._settings_used = settings
        self.ICs = ics
        self.harmonics_ = list(harmonics)
        self.periods_ = periods
        self.trend_type_ = best["trend_type"]
        self.lambda_ = best["elements"]["lambda"]
        self._y_in_sample = y_in_sample
        self._occurrence = occurrence
        self._ot = ot
        self.time_elapsed_ = time.time() - start_time
        return self

    def _select_xreg(
        self,
        best: Dict[str, Any],
        xreg: Dict[str, Any],
        kept: List[str],
        xreg_fit: Optional[Dict[str, Any]],
        ics: Dict[str, float],
        y: NDArray,
        table: Dict[str, NDArray],
        spec: Dict[str, Any],
        lam_spec: Dict[str, Any],
        settings: Dict[str, Any],
        ot: NDArray,
        distribution: str,
    ) -> Any:
        """The regressors selected by ``stepwise()`` on the errors of the best model
        without them, as R's ``adam_xreg_selector``: the model refitted with them is
        kept if it improves the IC. The regressors in ``kept`` (the dummies of the
        outliers) are in the model already, in ``xreg_fit``."""
        names = [name for name in xreg["names"] if name not in kept]
        if not names:
            return best, xreg_fit
        shape_estimated = int("shape" in best["names"])
        selected = xreg_selector(
            best["fitted"]["errors"][ot],
            xreg["data"][ot][:, [xreg["names"].index(name) for name in names]],
            names,
            self.ic,
            len(best["B"]) + 1 - shape_estimated,
            distribution,
            best["elements"]["shape"],
        )
        if not selected:
            return best, xreg_fit
        subset = st.xreg_subset(xreg, kept + list(selected))
        candidate = ft.fit(
            y,
            best["trend_type"],
            table,
            spec,
            lam_spec,
            distribution,
            self._initial_method,
            settings,
            subset,
            _named_b(best),
        )
        ic_candidate = self._ic(candidate)
        improves = ic_candidate < min(ics.values())
        ics[f"{candidate['trend_type']}+X({','.join(selected)})"] = ic_candidate
        return (candidate, subset) if improves else (best, xreg_fit)

    def _distribution_selection(self) -> str:
        """The distribution of the selection: ``"auto"`` selects with dgnorm, unless
        the loss implies one, as in ADAM."""
        if self.distribution != "auto":
            return self.distribution
        implied = {"likelihood": "dgnorm", "MAE": "dlaplace", "HAM": "ds"}
        return implied.get(self.loss, "dnorm")

    def _closest(
        self,
        best: Dict[str, Any],
        y: NDArray,
        table: Dict[str, NDArray],
        lam_spec: Dict[str, Any],
        settings: Dict[str, Any],
        xreg: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """The named distribution closest to the shape of dgnorm on the log scale
        (S 0.5, Laplace 1, normal 2), fitted on the structure selected with dgnorm
        from the default start and from its estimates without the shape, the higher
        likelihood kept (R's ``tbats_closest``)."""
        shapes = {"ds": 0.5, "dlaplace": 1.0, "dnorm": 2.0}
        log_shape = np.log(best["elements"]["shape"])
        distribution = min(shapes, key=lambda d: abs(log_shape - np.log(shapes[d])))
        keep = np.array([name != "shape" for name in best["names"]])
        bounds = {
            key: None if settings[key] is None else np.asarray(settings[key])[keep]
            for key in ("lb", "ub")
        }
        fits = [
            ft.fit(
                y,
                best["trend_type"],
                table,
                best["spec"],
                lam_spec,
                distribution,
                self._initial_method,
                {**settings, **bounds, "B": start},
                xreg,
            )
            for start in (None, best["B"][keep])
        ]
        return max(fits, key=lambda fit: fit["loglik"])

    def _ic(self, fitted: Dict[str, Any]) -> float:
        return st.ic_value(
            fitted["loglik"],
            # The observed values, as R's nobs of the log-likelihood
            int(np.sum(~np.isnan(fitted["y"]))),
            fitted["n_param_estimated"] + fitted["n_param_occurrence"],
            self.ic,
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
        """The log-likelihood of the data, with the Jacobian of the transform: that of
        the scale model when one is attached (R's ``implant()``)."""
        self._check_fitted()
        if self.scale_model is not None:
            return float(self.scale_model.loglik_sm_)
        return float(self._best["loglik"])

    @property
    def nobs(self) -> int:
        self._check_fitted()
        return len(self._y_in_sample)

    def _nobs_observed(self) -> int:
        """The observed values, which the likelihood and the information criteria
        count: the missing ones are not (R's nobs attribute of logLik)."""
        return int(np.sum(~np.isnan(self._y_in_sample)))

    @property
    def nparam(self) -> int:
        """The number of estimated parameters, the scale, the identified initials
        and the parameters of the occurrence model included."""
        self._check_fitted()
        best = self._best
        n_param = int(best["n_param_estimated"] + best["n_param_occurrence"])
        # The parameters of an attached scale model replace the scale (R's implant())
        if self.scale_model is not None:
            return n_param - 1 + int(self.scale_model.nparam)
        return n_param

    @property
    def n_param(self) -> NParam:
        """The table of the numbers of parameters, as ADAM's (R's ``$nParam``): the
        scale apart from the internal ones."""
        self._check_fitted()
        n_param = NParam.from_dict(
            {
                "estimated": {
                    "internal": int(self._best["n_param_estimated"]) - 1,
                    "occurrence": int(self._best["n_param_occurrence"]),
                    "scale": 1,
                },
                "provided": {"internal": int(self._best["n_param_provided"])},
            }
        )
        n_param.update_totals()
        return n_param

    @property
    def profile(self) -> NDArray:
        """The profile of the states at the end of the sample, from which the
        forecasts start (R's ``$profile``)."""
        self._check_fitted()
        return np.asarray(self._best["fitted"]["profile"])

    @property
    def initial_type(self) -> str:
        """The initialisation used (R's ``$initialType``): ``"provided"`` when some
        initial states are."""
        self._check_fitted()
        if self._best["initial_provided"]:
            return "provided"
        return str(self._best["initial_type"])

    @property
    def time_elapsed(self) -> float:
        """The time of the fit in seconds (R's ``$timeElapsed``)."""
        self._check_fitted()
        return self.time_elapsed_

    @property
    def aic(self) -> float:
        return st.ic_value(self.loglik, self._nobs_observed(), self.nparam, "AIC")

    def _ic_sizes(self) -> Optional[Tuple[float, float, int]]:
        """``(n_param_all, n_param_sizes, obs)`` of a mixture, or None: R's
        ``AICc.smooth`` / ``BICc.smooth`` correct the small sample over the
        parameters of the sizes and the non-zero fitted observed values."""
        if self._occurrence["model"] is None:
            return None
        observed = ~np.isnan(self._y_in_sample)
        obs = int(np.count_nonzero(np.asarray(self.fitted, dtype=float)[observed]))
        n_all = float(self.nparam)
        return n_all, n_all - float(self._best["n_param_occurrence"]), obs

    @property
    def aicc(self) -> float:
        terms = self._ic_sizes()
        if terms is None:
            return st.ic_value(self.loglik, self._nobs_observed(), self.nparam, "AICc")
        n_all, n_sizes, obs = terms
        correction = 2 * n_sizes * (n_sizes + 1) / (obs - n_sizes - 1)
        return float(2 * n_all - 2 * self.loglik + correction)

    @property
    def bic(self) -> float:
        return st.ic_value(self.loglik, self._nobs_observed(), self.nparam, "BIC")

    @property
    def bicc(self) -> float:
        terms = self._ic_sizes()
        if terms is None:
            return st.ic_value(self.loglik, self._nobs_observed(), self.nparam, "BICc")
        _, n_sizes, obs = terms
        return float(
            -2 * self.loglik + n_sizes * math.log(obs) * obs / (obs - n_sizes - 1)
        )

    @property
    def fitted(self) -> NDArray:
        """The fitted values in the space of the data (the medians), times the
        probabilities of occurrence."""
        self._check_fitted()
        sizes = st.box_cox_inverse(self._best["fitted"]["fitted"], self.lambda_)
        fitted = sizes * self._occurrence["p_fitted"]
        # No fitted value without the regressors
        xreg = self._best["struct"]["xreg"]
        if xreg is not None:
            fitted = np.where(np.any(np.isnan(xreg["data"]), axis=1), np.nan, fitted)
        return fitted

    @property
    def residuals(self) -> NDArray:
        """The errors in the space of the transformed data, NaN at the missing
        values; those of the location model standardised by the scale for a scale
        model (``sm()``), as R."""
        self._check_fitted()
        errors = self._best["fitted"]["errors"].copy()
        errors[np.isnan(self._y_in_sample)] = np.nan
        if getattr(self, "is_scale_", False):
            from smooth.adam_general.core.sm import _standardise_residuals

            return _standardise_residuals(
                self.location_.residuals,
                np.asarray(self.fitted, dtype=float),
                self.distribution_,
                None,
            )
        return errors

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
    def _component_names(self) -> List[str]:
        self._check_fitted()
        return list(self._best["struct"]["component_names"])

    @property
    def persistence_vector(self) -> Dict[str, float]:
        """The persistence vector (R's ``$persistence``) under ADAM's names:
        ``alpha``, ``beta``, ``gamma1_j[m]`` and ``gamma2_j[m]`` for the two states of
        the harmonic ``j`` of the period ``m``, ``psi`` for the ARMA states and
        ``delta`` for the regressors."""
        self._check_fitted()
        struct = self._best["struct"]
        table = struct["table"]
        n_arma = struct["n_arma"]
        names = ["alpha"] + ["beta"] * struct["trend_in"]
        names += [
            f"gamma{k}_{j}[{st._period_label(p)}]"
            for j, p in zip(table["j"], table["period"])
            for k in (1, 2)
        ]
        names += ["psi"] if n_arma == 1 else [f"psi{i}" for i in range(1, n_arma + 1)]
        names += [f"delta{i}" for i in range(1, struct["n_xreg"] + 1)]
        values = np.ravel(self._best["elements"]["vec_g"]).astype(float).tolist()
        return dict(zip(names, values))

    @property
    def persistence_level_(self) -> float:
        """The smoothing parameter of the level, alpha, as ADAM's."""
        self._check_fitted()
        return float(self._best["B_full"]["alpha"])

    @property
    def persistence_trend_(self) -> Optional[float]:
        """The smoothing parameter of the trend, beta, or None."""
        self._check_fitted()
        beta = self._best["B_full"].get("beta")
        return None if beta is None else float(beta)

    @property
    def persistence_seasonal_(self) -> List[List[float]]:
        """The smoothing parameters of the harmonics, ``[gamma1, gamma2]`` per period
        with harmonics, as ``persistence["seasonal"]`` takes them."""
        self._check_fitted()
        full = self._best["B_full"]
        return [
            [float(full[f"gamma{k}[{st._period_label(p)}]"]) for k in (1, 2)]
            for p in self.periods_
            if f"gamma1[{st._period_label(p)}]" in full
        ]

    @property
    def persistence_xreg_(self) -> Optional[List[float]]:
        """The smoothing parameters of the regressors, delta, or None."""
        self._check_fitted()
        deltas = [float(v) for k, v in self._best["B_full"].items() if k[:5] == "delta"]
        return deltas or None

    @property
    def b_value(self) -> NDArray:
        """The parameter vector B (R's ``$B``), as ``coef``."""
        return self.coef

    @property
    def om_model(self) -> Any:
        """The fitted occurrence model (OM / OMG), as ADAM's, or None."""
        self._check_fitted()
        model = self._occurrence["model"]
        return None if isinstance(model, dict) else model

    @property
    def transition(self) -> NDArray:
        self._check_fitted()
        return self._best["elements"]["mat_f"].copy()

    @property
    def measurement(self) -> NDArray:
        """The measurement matrix, with the regressors in their columns."""
        self._check_fitted()
        return self._best["fitted"]["mat_wt"].copy()

    @property
    def _xreg_names(self) -> List[str]:
        """The names of the regressors in the model (after the selection)."""
        self._check_fitted()
        xreg = self._best["struct"]["xreg"]
        return [] if xreg is None else list(xreg["names"])

    @property
    def phi_(self) -> float:
        self._check_fitted()
        return (
            float(self._best["elements"]["phi"])
            if self._best["struct"]["damped"]
            else 1.0
        )

    @property
    def arma_parameters_(self) -> Optional[Dict[str, Dict[str, float]]]:
        """The AR and MA parameters, estimated or provided, as ADAM's (R's
        ``$arma``): the parts the model has, or None without ARMA."""
        self._check_fitted()
        result: Dict[str, Dict[str, float]] = {}
        for name in self._best["spec"]["names"]:
            kind = "ar" if name.startswith("phi") else "ma"
            result.setdefault(kind, {})[name] = float(self._best["B_full"][name])
        return {kind: result[kind] for kind in ("ar", "ma") if kind in result} or None

    @property
    def scale(self) -> float:
        """sigma^2 for dnorm, s for the other distributions."""
        self._check_fitted()
        return float(self._best["scale"])

    # The diagnostics of ADAM, which need only the residuals, the scale and the
    # distribution: the residuals are those of the transformed data
    _check_is_fitted = _check_fitted
    rstandard = ADAM.rstandard
    rstudent = ADAM.rstudent
    outlierdummy = ADAM.outlierdummy
    multicov = ADAM.multicov
    _multicov_empirical = ADAM._multicov_empirical
    _variance_debiased = ADAM._variance_debiased
    plot = ADAM.plot
    is_combined = False

    @property
    def scale_model(self) -> Any:
        """The attached scale model (``sm()``), or None, as ``ADAM``'s."""
        return getattr(self, "_scale_model", None)

    @scale_model.setter
    def scale_model(self, value: Any) -> None:
        """Attach a scale model, as ``ADAM``'s (R's ``implant()``)."""
        ADAM.scale_model.fset(self, value)  # type: ignore[attr-defined]

    @property
    def _df_scale(self) -> float:
        """The degrees of freedom of the scale (R's ``adam_dfScale``): the non-zero
        observations minus the parameters, without the scale under likelihood."""
        obs_nonzero = int(self._ot.sum())
        df = obs_nonzero - (self.nparam - int(self.loss == "likelihood"))
        return float(df if df > 0 else obs_nonzero)

    def _gnorm_shape(self) -> Optional[float]:
        """The dgnorm shape, provided or estimated."""
        return float(self._best["elements"]["shape"])

    # The scale, or the fitted scale of an attached scale model (R's extractScale)
    extract_scale = ADAM.extract_scale

    @property
    def sigma(self) -> float:
        """The standard deviation of the residuals, de-biased (R's ``sigma``)."""
        self._check_fitted()
        return float(np.sqrt(np.nansum(self.residuals**2) / self._df_scale))

    def extract_sigma(self) -> float:
        """The standard deviation of the residuals (R's ``extractSigma``)."""
        return self.sigma

    @property
    def error_type(self) -> str:
        """The type of the error, additive in the transformed space."""
        return "A"

    @property
    def lags_used(self) -> List[float]:
        """The lags of the model: 1 and the seasonal periods."""
        self._check_fitted()
        return [1.0, *self.periods_]

    @property
    def data(self) -> NDArray:
        """The in-sample data."""
        self._check_fitted()
        return self._y_in_sample.copy()

    @property
    def holdout_data(self) -> Optional[NDArray]:
        """The holdout, when it was requested."""
        return self._y_holdout

    @property
    def _auto_forecast(self) -> Any:
        """The forecast of the fit, for the plot of the series."""
        forecast = self._forecast
        return None if forecast is None else SimpleNamespace(mean=forecast)

    def rmultistep(self, h: int = 10) -> pd.DataFrame:
        """The in-sample multistep forecast errors of the transformed data (R's
        ``rmultistep``), a row per origin and a column per horizon."""
        from smooth.adam_general.core.creator.architector import adam_profile_creator

        self._check_fitted()
        best, struct = self._best, self._best["struct"]
        lookup = adam_profile_creator(
            struct["lags_model_all"], struct["lags_model_max"], self.nobs
        )["index_lookup_table"]
        y_bc = st.box_cox_sizes(self._y_in_sample, self.lambda_, self._ot)
        y_bc[np.isnan(self._y_in_sample)] = np.nan
        errors = (
            best["adam_cpp"]
            .ferrors(
                np.asfortranarray(self.states),
                np.asfortranarray(self.measurement),
                np.asfortranarray(best["elements"]["mat_f"]),
                np.asfortranarray(lookup, dtype=np.uint64),
                np.asfortranarray(best["fitted"]["profile_initial"]),
                int(h),
                np.asarray(y_bc, dtype=float),
            )
            .errors
        )
        return pd.DataFrame(errors, columns=[f"h={i + 1}" for i in range(int(h))])

    def _multicov_analytical(
        self, h: int, covar_anal_fn: Any, var_anal_fn: Any
    ) -> NDArray:
        """The covariance of the transformed data from the matrices (R's
        ``multicov`` analytical), with the measurement of the last ``h`` rows."""
        struct = self._best["struct"]
        measurement = np.asarray(self.measurement, dtype=float)
        if measurement.shape[0] < h:
            mat_wt = np.tile(measurement[-1], (h, 1))
        else:
            mat_wt = measurement[-h:]
        return np.asarray(
            covar_anal_fn(
                np.asarray(struct["lags_model_all"]).flatten(),
                h,
                mat_wt,
                np.asarray(self._best["elements"]["mat_f"], dtype=float),
                np.asarray(self._best["elements"]["vec_g"], dtype=float).flatten(),
                # ADAM's method, which needs only the scale and the distribution
                self._variance_debiased(),  # type: ignore[misc]
            ),
            dtype=float,
        )

    def _multicov_simulated(self, h: int, nsim: int) -> NDArray:
        """The covariance of the simulated paths of the transformed data (R's
        ``multicov`` simulated)."""
        run = self._forecaster_run(h, None, False, nsim, None)
        _, general = run("simulated", 0.5, "both", scenarios=True, cumulative=False)
        paths = np.asarray(general["_scenarios_matrix"], dtype=float)
        centred = paths - paths.mean(axis=1, keepdims=True)
        return np.asarray((centred @ centred.T) / nsim)

    @property
    def loss_value(self) -> float:
        self._check_fitted()
        return float(self._best["loss_value"])

    @property
    def distribution_(self) -> str:
        """The distribution of the model, the one chosen with ``"auto"``."""
        self._check_fitted()
        return str(self._best["distribution"])

    @property
    def loss_(self) -> str:
        return self.loss

    @property
    def orders(self) -> Dict[str, List[int]]:
        """The ARMA orders as ADAM's, with no integration (R's ``$orders``)."""
        self._check_fitted()
        spec = self._best["spec"]
        return {
            "ar": spec["ar_orders"].tolist(),
            "i": [0] * len(spec["lags"]),
            "ma": spec["ma_orders"].tolist(),
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
        if struct["n_xreg"] > 0:
            result["xreg"] = dict(
                zip(self._xreg_names, np.asarray(read["states"]["xreg"], dtype=float))
            )
        return result

    @property
    def model_name(self) -> str:
        """TBATS(lambda, {p,q}, phi, <m1,k1>, ...), TBATSX with the regressors and
        {D} when they adapt."""
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
            f"TBATS{'X' if struct['n_xreg'] > 0 else ''}("
            f"{_r_round(self.lambda_, 3)}, {{{int(spec['ar_orders'].sum())},"
            f"{int(spec['ma_orders'].sum())}}}, {phi}{seasonal})"
            f"{'{D}' if struct['xreg_adapt'] else ''}"
        )

    @property
    def fisher_information_(self) -> Optional[NDArray]:
        """The observed Fisher Information (with ``fi=True``)."""
        self._check_fitted()
        return self._best["fi"]

    @property
    def _forecast(self) -> Optional[NDArray]:
        """The forecasts of the fit for ``h`` steps ahead."""
        self._check_fitted()
        if self._best["forecast_bc"] is None:
            return None
        sizes = st.box_cox_inverse(self._best["forecast_bc"], self.lambda_)
        return sizes * _p_forecast(self._occurrence["model"], len(sizes))

    # Methods
    def point_lik(self, log: bool = True) -> NDArray:
        """The log-densities of the data: those of the transformed data and the
        Jacobian, which sum to the log-likelihood. With a scale model, those of its
        likelihood (R's ``pointLik.sm.adam``)."""
        self._check_fitted()
        if self.scale_model is not None:
            return self.scale_model.point_lik(log=log)
        if getattr(self, "is_scale_", False):
            from smooth.adam_general.core.sm import _log_density

            location = self.location_
            lam = location.lambda_
            shape = location._best["elements"]["shape"]
            y = location._y_in_sample
            observed = ~np.isnan(y)
            ot = location._ot & observed
            zero = observed & ~ot
            y_bc = st.box_cox_sizes(y, lam, ot)
            scale = np.asarray(self.fitted, dtype=float)
            values = np.zeros(len(y))
            values[ot] = _log_density(
                self.distribution_,
                "A",
                y_bc[ot],
                y_bc[ot] - location.residuals[ot],
                scale[ot],
                shape,
            ) + (lam - 1) * _log_r(y[ot])
            # The occurrence model: the zeros have only its likelihood
            if location._occurrence["model"] is not None:
                p_fitted = location._occurrence["p_fitted"]
                values[zero] = _log_r(1 - p_fitted[zero])
                values[ot] += _log_r(p_fitted[ot])
        else:
            values = self._best["point_lik"](self._best["B"])
        return values if log else _exp_r(values)

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
        X: Optional[Any] = None,
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
        point: Literal["skeleton", "mean", "median"] = "skeleton",
        occurrence: Optional[Any] = None,
    ) -> ForecastResult:
        """The forecasts of ADAM's forecaster in the space of the transformed data,
        transformed back: the quantiles map onto those of the data. The point
        forecast (``point``, see :meth:`ADAM.predict`) is by default the skeleton, the
        inverse transform of the point forecast of the transformed data and the
        median of the forecast distribution, as is ``"median"``; ``"mean"`` is the
        mean, by Gauss-Hermite quadrature over the normal forecast distribution of
        the transformed data with ``dnorm`` (exp(mu + sigma^2/2) for lambda=0), from
        ``nsim`` simulated paths transformed back otherwise. ``X`` holds the future
        values of the regressors (the holdout, else their forecasts, when it is
        None). ``"confidence"`` and ``"complete"`` take the uncertainty of the
        parameters from :meth:`reforecast` (``nsim`` draws, 100 by default;
        ``"simulated"`` uses 10000 paths by default). With an occurrence model, the
        forecasts are those of the mixture of no demand and the sizes: the skeleton
        and the mean are multiplied by the probability of occurrence (``occurrence``
        provides it for the horizon, else the occurrence model forecasts it), and the
        median and the bounds are the quantiles of the mixture. The cumulative
        forecasts with lambda other than 1 or an occurrence model are the sum of the
        skeletons (times the probabilities), or the mean or the median of the sums of
        the ``nsim`` paths transformed back (``point``), whose quantiles give the
        bounds."""
        self._check_fitted()
        if point not in ("skeleton", "mean", "median"):
            raise ValueError(
                f'point should be "skeleton", "mean" or "median", not {point!r}.'
            )
        if h is None:
            h = self.h if self.h > 0 else 10
        # The probabilities of occurrence: provided, or forecast by the occurrence
        # model
        p_forecast = (
            _p_forecast(self._occurrence["model"], h)
            if occurrence is None
            else np.resize(np.asarray(occurrence, dtype=float), max(h, 0))
        )
        intermittent = bool(np.any(p_forecast < 1))
        if interval in ("confidence", "complete"):
            return self.reforecast(
                h=h,
                X=X,
                interval="confidence" if interval == "confidence" else "prediction",
                level=level,
                side=side,
                cumulative=cumulative,
                nsim=100 if nsim is None else nsim,
                seed=seed,
                point=point,
            ).to_forecast_result()
        if nsim is None:
            nsim = 10000
        run = self._forecaster_run(h, X, cumulative, nsim, seed)

        # The sums of the transformed values do not transform back: the cumulative
        # forecasts come from the paths of the data
        if cumulative and h > 0 and (self.lambda_ != 1 or intermittent):
            return self._cumulative(run, interval, level, side, point, p_forecast, seed)
        # The median of the transformed data is its skeleton, and transforms back into
        # the median of the data
        result, _ = run(interval, level, side)
        result.mean = _inverse_like(result.mean, self.lambda_)
        if point == "mean" and self.lambda_ != 1:
            result.mean[:] = self._mean(run, nsim)
        if result.lower is not None:
            result.lower = _inverse_like(result.lower, self.lambda_)
        if result.upper is not None:
            result.upper = _inverse_like(result.upper, self.lambda_)
        if intermittent and h > 0:
            self._mixture(result, run, p_forecast, interval, level, side, point)
        return result

    def _forecaster_run(
        self, h: int, X: Optional[Any], cumulative: bool, nsim: int, seed: Any
    ) -> Callable[..., Any]:
        """ADAM's forecaster in the space of the transformed data, as a function of
        the interval, the level, the side and the other general settings."""
        best = self._best
        struct = best["struct"]
        n_ets = struct["n_ets"]
        n_xreg = struct["n_xreg"]
        future = self._future_x(h, X)
        n_param = self.nparam
        # The parameters of the scale: those of an attached scale model, which
        # replace it (R's implant() puts them in the scale column)
        n_scale = int(self.loss == "likelihood") * (
            1 if self.scale_model is None else int(self.scale_model.nparam)
        )
        # The sizes: zero where there is no demand, and the scale divided by all the
        # observations, as ADAM's of an occurrence model, which the forecaster
        # de-biases by the non-zero ones
        ot = self._ot
        # The missing values stay missing, as in R's tbats_boxCoxObject
        y_bc = st.box_cox_sizes(self._y_in_sample, self.lambda_, ot)
        y_bc[np.isnan(self._y_in_sample)] = np.nan
        # The scale, divided by all the observed values, as ADAM's
        scale = self.scale
        errors = self._best["fitted"]["errors"].copy()
        errors[np.isnan(self._y_in_sample)] = np.nan
        # The scale of each horizon from an attached scale model (R's sm())
        scale_forecast = (
            np.asarray(self.scale_model.predict(h=h).mean, dtype=np.float64)
            if self.scale_model is not None and h > 0
            else None
        )

        def run(interval: Any, level: Any, side: Any, **general: Any) -> Any:
            """ADAM's forecaster in the space of the transformed data."""
            general_dict = {
                "h": int(h),
                "cumulative": cumulative,
                "nsim": nsim,
                "scenarios": False,
                "distribution": self.distribution_,
                "loss": self.loss,
                "other": {"shape": best["elements"]["shape"]},
                "n_param": None,
                "scale_forecast": scale_forecast,
                "seed": seed,
                **general,
            }
            result = forecaster(
                model_prepared={
                    "states": best["states"],
                    "mat_vt": best["fitted"]["profile_initial"],
                    "measurement": self.measurement,
                    "transition": best["elements"]["mat_f"],
                    "persistence": best["elements"]["vec_g"],
                    "profiles_recent_table": best["fitted"]["profile"],
                    "residuals": pd.Series(errors),
                    # The prediction of the model at the missing values
                    "y_fitted": np.where(
                        np.isnan(y_bc),
                        np.ravel(best["fitted"]["fitted"]),
                        y_bc - errors,
                    ),
                    "scale": scale,
                },
                observations_dict={
                    "obs_in_sample": self.nobs,
                    "obs_nonzero": int(ot.sum()),
                    "ot_logical": ot & ~np.isnan(self._y_in_sample),
                    "y_na_values": np.isnan(self._y_in_sample),
                    "y_in_sample": y_bc,
                    "y_forecast_start": self._forecast_start(),
                    "frequency": self._frequency(),
                },
                general_dict=general_dict,
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
                    "xreg_model": n_xreg > 0,
                    "xreg_number": n_xreg,
                    "new_xreg": future,
                },
                components_dict={
                    "components_number_ets": n_ets,
                    "components_number_ets_seasonal": 0,
                    "components_number_arima": struct["n_components"] - n_ets - n_xreg,
                },
                constants_checked={"constant_required": False},
                params_info=[[n_param - n_scale, n_scale, n_param]],
                adam_cpp=best["adam_cpp"],
                interval=interval,
                level=level,
                side=side,
            )
            return result, general_dict

        return run

    def _mixture(
        self,
        result: ForecastResult,
        run: Any,
        p_forecast: NDArray,
        interval: str,
        level: Any,
        side: str,
        point: str,
    ) -> None:
        """The mixture of no demand and the sizes: the skeleton and the mean
        multiplied by the probability, the median and the bounds its quantiles."""
        if point == "median":
            interval_used = "prediction" if interval == "none" else interval
            result.mean[:] = self._mixture_quantiles(
                run, interval_used, np.array([0.5]), p_forecast
            )[:, 0]
        else:
            result.mean[:] = np.asarray(result.mean, dtype=float) * p_forecast
        if interval == "none":
            return
        levels = np.atleast_1d(np.asarray(level, dtype=float))
        levels = np.where(levels > 1, levels / 100, levels)
        probs = {
            "both": ((1 - levels) / 2, (1 + levels) / 2),
            "upper": (np.zeros(len(levels)), levels),
            "lower": (1 - levels, np.ones(len(levels))),
        }[side]
        for bounds, prob in zip((result.lower, result.upper), probs):
            if bounds is not None:
                bounds.iloc[:, :] = self._mixture_quantiles(
                    run, interval, prob, p_forecast
                )

    def _mixture_quantiles(
        self, run: Any, interval: str, probs: NDArray, p_forecast: NDArray
    ) -> NDArray:
        """The quantiles of the mixture of no demand and the sizes (R's
        ``tbats_mixtureQuantiles``): zero below the probability of no demand,
        otherwise the quantile (q-(1-p))/p of the sizes, by the method of the
        interval."""
        quantiles = np.zeros((len(p_forecast), len(probs)))
        for p in np.unique(p_forecast):
            rows = p_forecast == p
            size_levels = (probs - (1 - p)) / p
            quantiles[np.ix_(rows, size_levels >= 1)] = np.inf
            columns = np.flatnonzero((size_levels > 0) & (size_levels < 1))
            if len(columns) > 0:
                sizes, _ = run(interval, list(size_levels[columns]), "upper")
                upper = st.box_cox_inverse(
                    np.asarray(sizes.upper, dtype=float), self.lambda_
                )
                quantiles[np.ix_(rows, columns)] = upper.reshape(len(p_forecast), -1)[
                    rows
                ]
        return quantiles

    def _mean(self, run: Any, nsim: int) -> NDArray:
        """The mean of the forecast distribution of the data (R's ``tbats_mean``):
        Gauss-Hermite quadrature over the normal forecast distribution of the
        transformed data, with the variance of the approximate interval (the closed
        form for lambda=0); for the other distributions the mean of the simulated
        paths transformed back."""
        from scipy.stats import norm

        lam = self.lambda_
        if self.distribution_ == "dnorm":
            # The bound at the level 2*pnorm(1)-1 is one standard deviation away
            bounds, _ = run("approximate", 2 * norm.cdf(1) - 1, "both")
            mu = np.asarray(bounds.mean, dtype=float)
            sigma = np.ravel(np.asarray(bounds.upper, dtype=float)) - mu
            if lam == 0:
                return _exp_r(mu + sigma**2 / 2)
            nodes, weights = np.polynomial.hermite.hermgauss(50)
            values = st.box_cox_inverse(
                mu[:, None] + np.sqrt(2) * sigma[:, None] * nodes, lam
            )
            return np.asarray(values @ weights / np.sqrt(np.pi))
        shape = self._best["elements"]["shape"]
        if lam == 0 and (
            self.distribution_ == "ds" or (self.distribution_ == "dgnorm" and shape < 1)
        ):
            warnings.warn(
                f"With lambda=0 and the {self.distribution_} distribution, the mean of "
                "the forecast distribution does not exist: the simulated one is "
                "unstable and grows with nsim.",
                stacklevel=3,
            )
        return np.asarray(self._paths(run, None, None).mean(axis=1))

    def _paths(self, run: Any, p_forecast: Optional[NDArray], rng: Any) -> NDArray:
        """The simulated paths of the data (R's ``tbats_paths``, h x nsim): those of
        the transformed data from the forecaster, transformed back, with the
        occurrence drawn with its probabilities."""
        _, general = run("simulated", 0.95, "both", scenarios=True, cumulative=False)
        paths = np.asarray(general["_scenarios_matrix"], dtype=float)
        paths = st.box_cox_inverse(paths, self.lambda_).reshape(paths.shape)
        if p_forecast is None:
            return paths
        return _occurrence_draws(paths, p_forecast, rng)

    def _cumulative(
        self,
        run: Any,
        interval: str,
        level: Any,
        side: str,
        point: str,
        p_forecast: NDArray,
        seed: Optional[int],
    ) -> ForecastResult:
        """The cumulative forecast from the paths of the data (R's
        ``tbats_cumulative``): the sum of the skeletons (times the probabilities), or
        the mean or the median of the sums of the paths, and their quantiles."""
        rng = np.random.default_rng(seed)
        totals = self._paths(run, p_forecast, rng).sum(axis=0, keepdims=True)
        # The structure of the cumulative forecast of the transformed data
        result, _ = run(
            "none" if interval == "none" else "simulated",
            level,
            side,
            cumulative=True,
            nsim=10,
        )
        if point == "skeleton":
            skeleton, _ = run("none", level, side, cumulative=False)
            skeleton = np.asarray(skeleton.mean, dtype=float)
            sizes = st.box_cox_inverse(skeleton, self.lambda_)
            result.mean[:] = np.sum(sizes * p_forecast)
        else:
            result.mean[:] = (np.mean if point == "mean" else np.median)(totals)
        if interval != "none":
            lower, upper = _paths_bounds(totals, level, side)
            if result.lower is not None:
                result.lower.iloc[:, :] = lower
            if result.upper is not None:
                result.upper.iloc[:, :] = upper
            result.interval = interval
        return result

    def _future_x(self, h: int, X: Optional[Any]) -> Optional[NDArray]:
        """The future values of the regressors (R's ``adam_xregNewdata``): ``X``,
        else the holdout, else the regressors forecast by ADAM with a warning. The
        dummies of the outliers are zero: not forecast, and added to ``X`` when it
        lacks them."""
        xreg = self._best["struct"]["xreg"]
        if xreg is None or h <= 0:
            return None
        names = xreg["names"]
        dummies = [i for i, name in enumerate(names) if OUTLIER_NAMES.match(name)]
        others = [i for i in range(len(names)) if i not in dummies]
        if X is not None:
            if isinstance(X, pd.DataFrame):
                values = X.set_axis(st.make_names([str(c) for c in X.columns]), axis=1)
                for i in dummies:
                    if names[i] not in values:
                        values[names[i]] = 0.0
                values = values[names].to_numpy(dtype=float)
            else:
                values = np.asarray(X, dtype=float)
                if values.ndim == 1:
                    values = values.reshape(-1, len(others) if dummies else len(names))
                if dummies and values.shape[1] == len(others):
                    full = np.zeros((values.shape[0], len(names)))
                    full[:, others] = values
                    values = full
            if values.shape[0] < h:
                warnings.warn(
                    f"X has {values.shape[0]} observations, while {h} are needed. "
                    "Using the last available values as future ones.",
                    stacklevel=3,
                )
                pad = np.repeat(values[-1:], h - values.shape[0], axis=0)
                values = np.vstack([values, pad])
            elif values.shape[0] > h:
                warnings.warn(
                    f"X has {values.shape[0]} observations, while only {h} are "
                    f"needed. Using the last {h} of them.",
                    stacklevel=3,
                )
                values = values[-h:]
            return values
        holdout = xreg["future"] if self.holdout else None
        if holdout is not None and holdout.shape[0] >= h:
            return holdout[:h].copy()
        from smooth.adam_general.core.adam import ADAM

        if others:
            warnings.warn(
                "X is not provided. Predicting the explanatory variables based on "
                "what I have in-sample.",
                stacklevel=3,
            )
        known = np.zeros((0, len(names))) if holdout is None else holdout
        h_needed = h - known.shape[0]
        forecasts = np.zeros((h_needed, len(names)))
        for i in others:
            forecasts[:, i] = np.asarray(
                ADAM().fit(xreg["data"][:, i]).predict(h=h_needed).mean, dtype=float
            )
        return np.vstack([known, forecasts])

    def _pull_back(self, parameters: NDArray, point: NDArray) -> NDArray:
        """The point of the segment from the estimates to ``point`` that is the
        furthest from them and satisfies the bounds, by bisection (R's
        ``tbats_pullBack``)."""
        in_bounds = self._best["in_bounds"]
        if in_bounds(point):
            return point
        inside, outside = 0.0, 1.0
        for _ in range(20):
            share = (inside + outside) / 2
            if not in_bounds(parameters + share * (point - parameters)):
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
        ``reapply.adamTBATS``). Each draw has its own lambda; a draw outside the bounds
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
        for i in range(nsim):
            draws[i] = self._pull_back(parameters, draws[i])
        refits = self._best["refitter"](draws)
        obs = self.nobs
        lag_max = self._best["struct"]["lags_model_max"]
        names = self._component_names
        lambdas = refits["lambda"]
        fitted_bc = refits["fitted"]
        columns = [f"nsim{i}" for i in range(1, nsim + 1)]
        refitted = (
            np.column_stack(
                [st.box_cox_inverse(fitted_bc[:, i], lambdas[i]) for i in range(nsim)]
            )
            * self._occurrence["p_fitted"][:, None]
        )
        # The sizes: no error where there is no demand
        ot = self._ot
        y = self._y_in_sample
        return TBATSReapplyResult(
            time_elapsed=time.time() - start_time,
            y=pd.Series(self.actuals),
            states=refits["states"][:, -(obs + lag_max) :, :],
            refitted=pd.DataFrame(refitted, columns=columns),
            fitted=pd.Series(self.fitted),
            model=self.model_name,
            transition=refits["mat_f"],
            measurement=refits["mat_wt"],
            persistence=pd.DataFrame(refits["vec_g"], index=names, columns=columns),
            profile=refits["profile"],
            random_parameters=pd.DataFrame(draws, columns=self.coef_names),
            nsim=nsim,
            lambdas=lambdas,
            # No error where there is no demand, NaN where the value is missing
            errors=np.where(
                np.isnan(y)[:, None],
                np.nan,
                np.column_stack(
                    [
                        (st.box_cox_sizes(y, lambdas[i], ot) - fitted_bc[:, i]) * ot
                        for i in range(nsim)
                    ]
                ),
            ),
        )

    def reforecast(
        self,
        h: int = 10,
        X: Optional[Any] = None,
        interval: Literal["prediction", "confidence", "none"] = "prediction",
        level: Union[float, List[float]] = 0.95,
        side: Literal["both", "upper", "lower"] = "both",
        cumulative: bool = False,
        nsim: int = 100,
        type: Optional[str] = None,  # noqa: A002
        heuristics: Optional[float] = None,
        seed: Optional[int] = None,
        point: Literal["skeleton", "mean", "median"] = "skeleton",
        **vcov_kwargs: Any,
    ) -> ReforecastResult:
        """The forecasts with the uncertainty of the parameters (R's
        ``reforecast.adamTBATS``): for each draw of :meth:`reapply`, its point forecasts
        (``"confidence"``) or its paths simulated with the scale of its own errors
        (``"prediction"``), in the space of its own transform and transformed back.
        The point forecast is the skeleton of the model (its median), or the mean or
        the median of the paths (``point``)."""
        from smooth.adam_general.core.adam import _column_names_for_levels
        from smooth.adam_general.core.creator.architector import adam_profile_creator
        from smooth.adam_general.core.utils.distributions import generate_errors

        self._check_fitted()
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
        # The scale of the sizes, on the non-zero observations (R's adam_dfScale)
        ot = self._ot
        obs_nonzero = int(self._ot.sum())
        df_scale = self._df_scale
        draws = refitted.random_parameters
        adam_cpp = best["adam_cpp"]
        # The future values of the regressors, as predict() takes them
        future = self._future_x(h, X)
        paths = []
        for j in range(nsim):
            mat_wt = np.tile(refitted.measurement[0, :, j], (h, 1))
            if future is not None:
                mat_wt[:, struct["xreg_rows"]] = future
            mat_wt = np.asfortranarray(mat_wt)
            mat_f = np.asfortranarray(refitted.transition[:, :, j])
            profile = np.array(refitted.profile[:, :, j], order="F")
            lam = refitted.lambdas[j]
            if interval == "prediction":
                shape = (
                    draws["shape"].iloc[j]
                    if "shape" in draws
                    else best["elements"]["shape"]
                )
                errors_sizes = refitted.errors[ot, j]
                scale = scale_debias(
                    st.scale_value(errors_sizes, self.distribution_, shape),
                    self.distribution_,
                    len(errors_sizes),
                    df_scale,
                )
                errors = generate_errors(
                    self.distribution_,
                    h * nsim,
                    scale,
                    obs_in_sample=obs_nonzero,
                    n_param=obs_nonzero - df_scale,
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
                skeleton = adam_cpp.forecast(mat_wt, mat_f, lookup, profile, h).forecast
                paths.append(st.box_cox_inverse(np.ravel(skeleton), lam).reshape(h, 1))
        path_matrix = np.column_stack(paths)
        # The occurrence drawn with its probabilities; the cumulative values are the
        # sums of the paths in the space of the data
        p_forecast = _p_forecast(self._occurrence["model"], h)
        path_matrix = _occurrence_draws(path_matrix, p_forecast, rng)
        if cumulative:
            path_matrix = path_matrix.sum(axis=0, keepdims=True)

        # The skeleton, times the probability with an occurrence model
        point_forecast = self.predict(h=h, X=future).mean
        mean = point_forecast
        if cumulative:
            mean = pd.Series([point_forecast.sum()], index=point_forecast.index[:1])
        if point != "skeleton":
            statistic = np.nanmean if point == "mean" else np.nanmedian
            mean = pd.Series(statistic(path_matrix, axis=1), index=mean.index)
        if interval == "none":
            lower = upper = None
        else:
            lower_cols, upper_cols = _column_names_for_levels(levels, side)
            lower_values, upper_values = _paths_bounds(path_matrix, levels, side)
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
        transformed back (R's ``simulate.adamTBATS``), starting from its initials."""
        from smooth.adam_general._adam_general import adam_simulator
        from smooth.adam_general.core.creator.architector import adam_profile_creator
        from smooth.adam_general.core.utils.distributions import generate_errors

        self._check_fitted()
        best = self._best
        struct = best["struct"]
        obs = self.nobs if obs is None else int(obs)
        lag_max = struct["lags_model_max"]
        n_ets = struct["n_ets"]
        n_scale = int(self.loss == "likelihood")
        # The scale of the sizes, de-biased on the non-zero observations
        obs_nonzero = int(self._ot.sum())
        df_scale = max(obs_nonzero - (self.nparam - n_scale), 1)
        rng = np.random.default_rng(seed)
        # The scale is divided by the observed sizes
        scale = scale_debias(self.scale, self.distribution_, obs_nonzero, df_scale)
        errors = np.reshape(
            generate_errors(
                self.distribution_,
                obs * nsim,
                scale,
                obs_in_sample=obs_nonzero,
                n_param=obs_nonzero - df_scale,
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
        # The in-sample measurement, its last row repeated beyond the sample
        measurement = self.measurement
        if measurement.shape[0] < obs:
            pad = np.repeat(measurement[-1:], obs - measurement.shape[0], axis=0)
            measurement = np.vstack([measurement, pad])
        measurement = measurement[:obs]
        result = adam_simulator(
            matrixErrors=errors,
            matrixOt=np.ones((obs, nsim)),
            arrayVt=array_vt,
            matrixWt=np.asfortranarray(measurement),
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
            nArima=struct["n_components"] - n_ets - struct["n_xreg"],
            nXreg=struct["n_xreg"],
            constant=False,
        )
        data = st.box_cox_inverse(np.asarray(result["matrixYt"]), self.lambda_)
        # The sizes, and the occurrence drawn with the fitted probabilities
        probability = np.resize(self._occurrence["p_fitted"], obs)
        occurrence = rng.binomial(1, np.repeat(probability[:, None], nsim, axis=1))
        data = data.reshape(obs, nsim) * occurrence
        return SimulateResult(
            model=self.model_name,
            data=pd.Series(data[:, 0]) if nsim == 1 else pd.DataFrame(data),
            states=np.asarray(result["arrayVt"]).reshape(array_vt.shape, order="F"),
            residuals=pd.Series(errors[:, 0]) if nsim == 1 else pd.DataFrame(errors),
            persistence=best["elements"]["vec_g"].reshape(-1, 1),
            measurement=measurement,
            transition=best["elements"]["mat_f"].copy(),
            initial=profile.copy(),
            probability=probability,
            occurrence=None if self._occurrence["model"] is None else occurrence,
            profile=profile.copy(),
            other={"shape": best["elements"]["shape"]}
            if self.distribution_ == "dgnorm"
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

    def sm(self, X: Optional[Any] = None, **kwargs: Any) -> "TBATS":
        """The model of the scale of the error term (R's ``sm()``): TBATS on the
        transformed errors in the space of the Box-Cox transformed data (the squares
        for ``"dnorm"``, the absolute values for ``"dlaplace"``, ...), with lambda 0,
        estimated by the joint likelihood of this model's data.

        The transformed errors are divided by the exponent of the mean of their
        logarithm at a unit scale, so that their logarithms are unbiased for the
        log-scale. It takes the arguments and defaults of ``TBATS`` (``X`` its
        regressors), but the lags of this model. Attach the
        result to :attr:`scale_model` (R's ``implant()``): the forecasts then have the
        scale of each horizon. With an occurrence model, the zeros, whose sizes are
        not observed, are gaps of the scale model and have only the likelihood of
        the occurrence."""
        from smooth.adam_general.core.sm import _log_density, _residual_transform

        self._check_fitted()
        if self.loss != "likelihood":
            raise ValueError(
                "sm() only works with models estimated via maximisation of "
                f"likelihood. Yours was estimated via {self.loss}. Cannot proceed."
            )
        distribution = self.distribution_
        shape = self._best["elements"]["shape"]
        lam = self.lambda_
        # The sizes: the zeros of an occurrence model are missing values of the
        # response, gaps of the scale model, as the missing ones
        y = self._y_in_sample
        observed = ~np.isnan(y)
        ot = self._ot & observed
        zero = observed & ~ot
        errors = self.residuals
        y_bc = st.box_cox_sizes(y, lam, ot)
        response = np.full(len(errors), np.nan)
        response[ot] = _residual_transform(errors[ot], distribution, shape)
        # The logarithm of the transformed error is biased for the log-scale by the
        # mean of its logarithm at a unit scale (that of chi-squared with one degree of
        # freedom for dnorm, -1.27): removed, so that the scale model in logs follows
        # the scale, and its states (backcast, or updated by the errors in logs) are
        # not off by a factor
        log_bias = {
            "dnorm": special.digamma(0.5) + math.log(2),
            "dlaplace": special.digamma(1),
            "ds": special.digamma(2) - math.log(2),
        }.get(distribution)
        if log_bias is None:
            log_bias = (math.log(shape) + special.digamma(1 / shape)) / shape
        response[ot] = response[ot] * _exp_r(np.array([-log_bias]))[0]

        # The joint log-likelihood of the observed values given the scale, the
        # exponent of the fitted values of the scale model in logs: the densities of
        # the sizes (TBATS hands the custom loss the sizes; the zeros of an
        # occurrence model have only its likelihood)
        y_sm = y_bc[ot]
        mu_sm = y_sm - errors[ot]

        def loss(actual: Any = None, fitted: Any = None, B: Any = None) -> float:
            scale = _exp_r(np.asarray(fitted))
            densities = _log_density(distribution, "A", y_sm, mu_sm, scale, shape)
            return -float(_sum_r(densities))

        arguments: Dict[str, Any] = {
            "lags": self.lags,
            "distribution": distribution,
            "gnorm_shape": shape if distribution == "dgnorm" else None,
            **kwargs,
            "lambda_bc": 0,
            "loss": loss,
        }
        # The zeros of an occurrence model are the missing values of the response
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Data contains NAs")
            warnings.filterwarnings("ignore", message="More than half")
            scale_model = TBATS(**arguments).fit(response, X)

        # The log-likelihood of the data: the sizes with the Jacobian of their
        # transform, the occurrence, and the parameters of both models (one scale)
        jacobian = (lam - 1) * _log_r(y[ot])
        scale_model.loglik_sm_ = scale_model.loglik + float(_sum_r(jacobian))
        if self._occurrence["model"] is not None:
            p_fitted = self._occurrence["p_fitted"]
            scale_model.loglik_sm_ += float(
                _sum_r(_log_r(p_fitted[ot])) + _sum_r(_log_r(1 - p_fitted[zero]))
            )
        scale_model.df_sm_ = int(scale_model.nparam) + int(self.nparam) - 1
        scale_model.is_scale_ = True
        scale_model.location_ = self
        return scale_model

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
            "distribution": self.distribution_,
            "loss": self.loss_function if self.loss == "custom" else self.loss,
            # The provided values, as they were applied
            "persistence": self._provided["persistence"],
            "phi": self._provided["phi"],
            "initial": (
                self._initial_method
                if self._provided["initial"] is None
                else self._provided["initial"]
            ),
            "arma": self._provided["arma"],
            "bounds": self.bounds,
            "nlopt_kwargs": {
                **(self.nlopt_kwargs or {}),
                "B": B,
                "lb": np.full(len(B), -np.inf),
                "ub": np.full(len(B), np.inf),
            },
        }
        if self._best["struct"]["n_xreg"] > 0:
            kwargs["regressors"] = (
                "adapt" if self._best["struct"]["xreg_adapt"] else "use"
            )
        # The occurrence model is refitted on the sample, with its type
        model = self._occurrence["model"]
        if model is not None:
            if isinstance(model, dict):
                raise ValueError(
                    "The refits need an occurrence model, not the provided "
                    "probabilities."
                )
            kwargs["occurrence"] = model.occurrence
        if self.distribution_ == "dgnorm" and "shape" not in self.coef_names:
            kwargs["gnorm_shape"] = self._best["elements"]["shape"]
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
            self._initial_method in ("backcasting", "complete", "gradient"),
            np.random.default_rng(seed),
        )
        xreg = self._best["struct"]["xreg"]
        worker = partial(
            _refit_one_replicate,
            self.actuals,
            None if xreg is None else xreg["data"],
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
            "ICs": self.ICs,
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
