"""
Complex Exponential Smoothing (CES) model.

Translates R/adam-ces.R ces() and R/autoces.R auto.ces().
Self-contained module with its own fit pipeline, reusing adamCore C++ for
state-space filtering and forecasting.
"""

import re
import time
import warnings
from typing import Any, Dict, List, Literal, Optional, Union, cast

import nlopt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from smooth.adam_general import _adamCore
from smooth.adam_general._numDeriv import hessian as _hessian_cpp
from smooth.adam_general.core.adam import ADAM
from smooth.adam_general.core.ces.cost_function import ces_cf
from smooth.adam_general.core.ces.creator import ces_creator
from smooth.adam_general.core.ces.filler import ces_filler
from smooth.adam_general.core.ces.initialiser import ces_initialiser
from smooth.adam_general.core.checker.data_checks import _fill_missing
from smooth.adam_general.core.creator.architector import (
    adam_head_length,
)
from smooth.adam_general.core.forecaster.forecaster import forecaster
from smooth.adam_general.core.forecaster.result import ForecastResult
from smooth.adam_general.core.utils.ic import AIC, BIC, AICc, BICc
from smooth.adam_general.core.utils.n_param import NParam
from smooth.adam_general.core.utils.utils import _sum_r, multistep_log_lik

SEASONALITY_OPTIONS = Literal["none", "simple", "partial", "full"]
LOSS_OPTIONS = Literal[
    "likelihood", "MSE", "MAE", "HAM", "MSEh", "TMSE", "GTMSE", "MSCE", "GPL"
]
_VALID_LOSSES = (
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
_CES_NLOPT_WARNING_SHOWN = False

# The settings of the two optimisers, as R's ces() takes them in the ellipsis: the
# first (BOBYQA) with the suffix 0, the second (Nelder-Mead) without
_NLOPT_DEFAULTS: Dict[str, Any] = {
    "algorithm0": "NLOPT_LN_BOBYQA",
    "algorithm": "NLOPT_LN_NELDERMEAD",
    "maxeval0": None,
    "maxeval": None,
    "maxtime0": -1,
    "maxtime": -1,
    "xtol_rel0": 1e-8,
    "xtol_abs0": 0,
    "ftol_rel0": 0,
    "ftol_abs0": 0,
    "xtol_rel": 1e-6,
    "xtol_abs": 1e-8,
    "ftol_rel": 1e-8,
    "ftol_abs": 0,
}


def _check_loss(loss):
    """Reject a loss ces() would not accept.

    R's ces() runs match.arg() over exactly this set, so a callable or one of
    ADAM's absolute / half-moment multistep losses is an error there. Without
    this check the cost function falls through to its MSE branch and returns a
    fit under a loss that was never requested.
    """
    if loss not in _VALID_LOSSES:
        raise ValueError(
            f"Unknown loss function: {loss!r}. CES accepts one of "
            f"{', '.join(repr(v) for v in _VALID_LOSSES)}."
        )
    return


def _pristine(kwargs):
    """Deep-enough copy of the cost-function arguments.

    ``ces_cf`` writes through ``mat_vt``, ``mat_f``, ``vec_g`` and the profile
    tables, and the final fit leaves the *fitted* head in them rather than the
    creator seed, so anything re-evaluated afterwards would profile a different
    surface than the one that was optimised. R gets this for free from
    copy-on-modify.
    """
    out = {}
    for key, value in kwargs.items():
        if isinstance(value, np.ndarray):
            out[key] = value.copy()
        elif isinstance(value, dict):
            out[key] = {
                k: (v.copy() if isinstance(v, np.ndarray) else v)
                for k, v in value.items()
            }
        else:
            out[key] = value
    return out


def _validate_b(b, seasonality):
    """Check the provided ``b`` against the seasonality it belongs to.

    "partial" carries one real coefficient, "full" a complex pair, and
    "none"/"simple" have no ``b`` at all. Letting a wrong-shaped ``b`` through
    only surfaces later as a 1e+300 penalty masquerading as a fit, and silently
    taking ``Re(b)`` would hide a user error rather than report it. Mirrors the
    same guard in R's ``ces()`` (``R/adam-ces.R``).
    """
    if b is None:
        return None
    if seasonality in ("none", "simple"):
        warnings.warn(
            f"CES({seasonality}) has no second smoothing parameter, so the "
            "provided b is not used. Dropping it.",
            UserWarning,
            stacklevel=3,
        )
        return None
    if seasonality == "partial" and isinstance(b, complex):
        raise ValueError(
            "CES(partial) has a real second smoothing parameter, but b is "
            'complex. Provide a real value, or use seasonality="full" for a '
            "complex b."
        )
    if seasonality == "full" and not isinstance(b, complex):
        raise ValueError(
            "CES(full) has a complex second smoothing parameter, but b is "
            "real. Provide a complex value (e.g. complex(re, im)), or use "
            'seasonality="partial" for a real b.'
        )
    return b


class CES:
    """
    Complex Exponential Smoothing in state space form.

    CES uses complex-valued smoothing parameters to capture level and
    "potential" (rate of change) dynamics. It supports four seasonality modes.

    Parameters
    ----------
    seasonality : str, default="none"
        Seasonality type: "none", "simple", "partial", or "full".
    lags : list of int or None
        Seasonal period(s). If None, defaults to [1].
    initial : str, default="backcasting"
        Initialization method: "backcasting", "optimal", "two-stage", "complete".
    a : complex or None
        First complex smoothing parameter. None = estimate.
    b : complex, float, or None
        Second smoothing parameter. Real for partial, complex for full.
        None = estimate (for partial/full only).
    loss : str, default="likelihood"
        Loss function for parameter estimation.
    h : int or None
        Forecast horizon.
    holdout : bool, default=False
        Whether to use holdout sample for validation.
    bounds : str, default="admissible"
        "admissible" (eigenvalue stability) or "none".
    ic : str, default="AICc"
        Information criterion for model comparison.
    verbose : int, default=0
        Verbosity level.
    regressors : str, default="use"
        How to handle external regressors.
    fi : bool, default=False
        Whether to compute the Fisher Information of the parameters, as ``ADAM``
        (R's ``FI=TRUE``), in ``fisher_information_``.
    step_size : float or None
        The step of the Hessian behind the Fisher Information, as ``ADAM``.
    nlopt_kwargs : dict or None
        The settings of the two optimisers, as ``ADAM``'s ``nlopt_kwargs`` and R's
        ellipsis of ``ces()``: the first one (``"algorithm0"``, default
        ``"NLOPT_LN_BOBYQA"``; ``"maxeval0"``, default as ``"maxeval"``;
        ``"maxtime0"``, ``"xtol_rel0"`` 1e-8, ``"xtol_abs0"``, ``"ftol_rel0"``,
        ``"ftol_abs0"`` 0) and the second one (``"algorithm"``, default
        ``"NLOPT_LN_NELDERMEAD"``; ``"maxeval"``, default 40 per parameter;
        ``"maxtime"``, ``"xtol_rel"`` 1e-6, ``"xtol_abs"`` 1e-8, ``"ftol_rel"``
        1e-8, ``"ftol_abs"`` 0).
    """

    # Attributes set during .fit() (declared here so mypy can resolve them
    # when one CES instance reads another's fitted state, e.g. two-stage init).
    coef: NDArray
    initial_value: Dict[str, NDArray]
    # Set by AutoCES on the model it selects, as R's auto.ces()
    ICs: Dict[str, float]

    def __init__(
        self,
        seasonality: SEASONALITY_OPTIONS = "none",
        lags: Optional[List[int]] = None,
        initial: str = "backcasting",
        a: Optional[complex] = None,
        b: Optional[Union[complex, float]] = None,
        loss: LOSS_OPTIONS = "likelihood",
        h: Optional[int] = None,
        holdout: bool = False,
        bounds: Literal["admissible", "none"] = "admissible",
        ic: Literal["AIC", "AICc", "BIC", "BICc"] = "AICc",
        verbose: int = 0,
        regressors: Literal["use", "select"] = "use",
        fi: bool = False,
        step_size: Optional[float] = None,
        nlopt_kwargs: Optional[Dict[str, Any]] = None,
        head_length: Optional[int] = None,
    ) -> None:
        # Validate seasonality
        valid = {"none", "simple", "partial", "full"}
        abbrev = {"n": "none", "s": "simple", "p": "partial", "f": "full"}
        seasonality_resolved: str = abbrev.get(seasonality, seasonality)
        if seasonality_resolved not in valid:
            raise ValueError(
                f"seasonality must be one of {valid}, got '{seasonality_resolved}'"
            )

        self.seasonality = cast(SEASONALITY_OPTIONS, seasonality_resolved)
        self.lags = lags
        self.initial = initial
        self.head_length = head_length
        self._a_provided = a
        self._b_provided = _validate_b(b, seasonality)
        self.loss = loss
        self.h = h
        self.holdout = holdout
        self.bounds = bounds
        self.ic = ic
        self.verbose = verbose
        self.regressors = regressors
        self.fi = fi
        self.step_size = step_size
        self.nlopt_kwargs = nlopt_kwargs
        unknown = set(nlopt_kwargs or {}) - set(_NLOPT_DEFAULTS)
        if unknown:
            raise ValueError(
                f"Unknown nlopt_kwargs of CES: {', '.join(sorted(unknown))}. "
                f"Accepted: {', '.join(_NLOPT_DEFAULTS)}."
            )
        self._nlopt = {**_NLOPT_DEFAULTS, **(nlopt_kwargs or {})}

    def fit(self, y: NDArray, X: Optional[NDArray] = None) -> "CES":
        """
        Fit the CES model to time series data.

        Parameters
        ----------
        y : array-like
            Time series values.
        X : array-like or None
            Exogenous regressors matrix.

        Returns
        -------
        self
        """
        _check_loss(self.loss)
        start_time = time.time()
        y = np.asarray(y, dtype=np.float64).ravel()
        nlopt_settings = self._nlopt

        # CES parity with R depends on the stage-1 BOBYQA trajectory.
        # ETS / ARIMA do not use this two-stage CES path, so keep the check local.
        if nlopt_settings["algorithm0"] == "NLOPT_LN_BOBYQA":
            version_match = re.findall(r"\d+", getattr(nlopt, "__version__", ""))
            version_tuple = tuple(int(part) for part in version_match[:3])
            if version_tuple and version_tuple < (2, 10, 0):
                global _CES_NLOPT_WARNING_SHOWN
                if not _CES_NLOPT_WARNING_SHOWN:
                    warnings.warn(
                        "CES strict parity with R requires nlopt>=2.10.0 for "
                        "the stage-1 BOBYQA path; current Python nlopt is "
                        f"{nlopt.__version__}.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    _CES_NLOPT_WARNING_SHOWN = True

        # Handle holdout — R lines 65-66 of autoces
        h = self.h if self.h is not None else 0
        if self.holdout and h > 0:
            obs_in_sample = len(y) - h
            y_holdout = y[obs_in_sample:]
            y_in_sample = y[:obs_in_sample]
        else:
            obs_in_sample = len(y)
            y_holdout = None
            y_in_sample = y

        # Determine frequency from lags
        if self.lags is None or len(self.lags) == 0:
            lags = [1]
        else:
            lags = list(self.lags)
        y_frequency = max(lags)

        # The missing values are gaps: filled for the initialisation only, skipped by
        # the fit and not in the loss, as in R's ces()
        y_actuals = y[:obs_in_sample].copy()
        _, y_filled, y_na_values = _fill_missing(y, y, lags, obs_in_sample)
        y_in_sample = np.asarray(y_filled, dtype=float)[:obs_in_sample]
        observed = ~np.asarray(y_na_values, dtype=bool)[:obs_in_sample]
        obs_observed = int(np.sum(observed))

        # Set up a and b parameter dicts — R lines 157-181
        a: Dict[str, Any] = {
            "value": self._a_provided,
            "estimate": self._a_provided is None,
        }
        b: Dict[str, Any]
        if self._b_provided is None and self.seasonality in ("partial", "full"):
            b = {"value": None, "estimate": True}
        else:
            b = {"value": self._b_provided, "estimate": False}

        if self.seasonality == "partial":
            b["number"] = 1
        elif self.seasonality == "full":
            b["number"] = 2
        else:
            b["number"] = 0

        # Seasonal lags — R line 574
        lags_model_seasonal = (
            [lag for lag in lags if lag > 1] if y_frequency > 1 else lags
        )
        if not lags_model_seasonal:
            lags_model_seasonal = lags
        n_seasonal = len(lags_model_seasonal)

        # Component count — R lines 576-580
        if self.seasonality == "none":
            components_number = 2
        elif self.seasonality == "simple":
            components_number = 2 * n_seasonal
        elif self.seasonality == "partial":
            components_number = 2 + n_seasonal
        elif self.seasonality == "full":
            components_number = 2 + 2 * n_seasonal

        # Xreg setup
        xreg_model = X is not None and X.shape[1] > 0
        xreg_number = X.shape[1] if X is not None and xreg_model else 0
        xreg_data = X[:obs_in_sample] if X is not None and xreg_model else None
        xreg_names = [f"x{i + 1}" for i in range(xreg_number)]

        # Build lags_model_all — R lines 584-590
        if self.seasonality == "none":
            ces_lags = [1, 1]
        elif self.seasonality == "simple":
            ces_lags = []
            for lag in lags_model_seasonal:
                ces_lags.extend([lag, lag])
        elif self.seasonality == "partial":
            ces_lags = [1, 1] + lags_model_seasonal
        elif self.seasonality == "full":
            ces_lags = [1, 1]
            for lag in lags_model_seasonal:
                ces_lags.extend([lag, lag])

        # Add xreg lags (all 1)
        lags_model_all = ces_lags + [1] * xreg_number
        lags_model_max = max(lags_model_all)

        # The profile lookup table has to span the forecast horizon too, otherwise
        # the multistep losses walk off the end of it (R/utils-adam.R:107).
        obs_all = len(y) + (0 if self.holdout else h)
        # Backcasting head: one full lag cycle by default (filtering on), 0 switches
        # it off, larger values give that head length (R/adam-ces.R).
        head_resolved = adam_head_length(
            self.head_length, lags_model_max, obs_in_sample
        )
        head_length = head_resolved["geometry"]
        obs_states = obs_in_sample + head_length

        # Occurrence (CES doesn't support occurrence — R line 618)
        # The fit skips the missing values
        ot = observed.astype(np.float64)
        ot_logical = observed.copy()

        # Determine initial type
        initial_type = self.initial
        if isinstance(initial_type, dict):
            initial_type = "provided"

        # Match R/adamGeneral.R: backcasting / complete use two iterations,
        # optimal / provided paths use one.
        if initial_type in ("backcasting", "complete", "gradient"):
            n_iterations = 2
        else:
            n_iterations = 1

        # Create adamCore C++ instance — R lines 597-603
        adam_cpp = _adamCore.adamCore(
            lags=np.array(lags_model_all, dtype=np.uint64),
            E="A",
            T="N",
            S="N",
            nNonSeasonal=0,
            nSeasonal=0,
            nETS=0,
            nArima=components_number,
            nXreg=xreg_number,
            nComponents=len(lags_model_all),
            constant=False,
            adamETS=False,
        )
        adam_cpp.headLength = head_resolved["flag"]

        # Create matrices — R line creator() call
        created = ces_creator(
            seasonality=self.seasonality,
            n_seasonal=n_seasonal,
            lags_model_seasonal=lags_model_seasonal,
            lags_model_all=lags_model_all,
            lags_model_max=lags_model_max,
            components_number=components_number,
            xreg_number=xreg_number,
            obs_in_sample=obs_in_sample,
            obs_states=obs_states,
            obs_all=obs_all,
            y_in_sample=y_in_sample,
            y_frequency=y_frequency,
            lags=lags,
            xreg_data=xreg_data,
            xreg_names=xreg_names,
            head_length=head_length,
        )

        mat_vt = created["mat_vt"]
        mat_wt = created["mat_wt"]
        mat_f = created["mat_f"]
        vec_g = created["vec_g"]
        profiles_recent_table = created["profiles_recent_table"]
        index_lookup_table = created["index_lookup_table"]

        # Multistep detection
        multisteps = self.loss in (
            "MSEh",
            "TMSE",
            "GTMSE",
            "MSCE",
            "GPL",
            "MAEh",
            "TMAE",
            "GTMAE",
            "MACE",
            "HAMh",
            "THAM",
            "GTHAM",
            "CHAM",
        )

        # Two-stage initialization — R lines 750-786
        B: Optional[NDArray] = None
        if initial_type == "two-stage" and B is None:
            ces_back = CES(
                seasonality=self.seasonality,
                lags=self.lags,
                initial="complete",
                a=self._a_provided,
                b=self._b_provided,
                loss=self.loss,
                h=h,
                holdout=self.holdout,
                bounds=self.bounds,
                verbose=0,
                nlopt_kwargs=self.nlopt_kwargs,
            )
            ces_back.fit(y, X=X)
            B = ces_back.coef.copy()

            # Append initial state estimates — R lines 769-785
            if self.seasonality != "simple":
                B = np.concatenate([B, ces_back.initial_value["nonseasonal"]])
            if self.seasonality != "none":
                seasonal_init = ces_back.initial_value.get("seasonal")
                if seasonal_init is not None:
                    # Time by time, the components of each, as R's as.vector() of
                    # its components-by-time matrix
                    B = np.concatenate([B, seasonal_init.ravel(order="C")])
            if xreg_model and "xreg" in ces_back.initial_value:
                B = np.concatenate([B, ces_back.initial_value["xreg"]])

        # The initial B, unless from two-stage, and the names of its elements --
        # R line 788-789
        B_initial, b_names = ces_initialiser(
            a=a,
            b=b,
            seasonality=self.seasonality,
            n_seasonal=n_seasonal,
            lags_model_seasonal=lags_model_seasonal,
            lags_model_max=lags_model_max,
            mat_vt=mat_vt,
            initial_type=initial_type,
            components_number=components_number,
            xreg_model=xreg_model,
            xreg_number=xreg_number,
            xreg_names=xreg_names,
        )
        if B is None:
            B = B_initial

        # Maxeval — R lines 800-807
        maxeval_used = nlopt_settings["maxeval"]
        if maxeval_used is None:
            maxeval_used = len(B) * 40
            if xreg_model:
                maxeval_used = max(1000, len(B) * 100)
        maxeval0_used = nlopt_settings["maxeval0"]
        if maxeval0_used is None:
            maxeval0_used = maxeval_used

        # CF arguments shared by both optimizer stages
        cf_kwargs = dict(
            mat_vt=mat_vt,
            mat_wt=mat_wt,
            mat_f=mat_f,
            vec_g=vec_g,
            a=a,
            b=b,
            seasonality=self.seasonality,
            n_seasonal=n_seasonal,
            lags_model_seasonal=lags_model_seasonal,
            lags_model_max=lags_model_max,
            initial_type=initial_type,
            xreg_model=xreg_model,
            xreg_number=xreg_number,
            initial_xreg_estimate=xreg_model,
            components_number=components_number,
            lags_model_all=lags_model_all,
            index_lookup_table=index_lookup_table,
            profiles_recent_table=profiles_recent_table,
            y_in_sample=y_in_sample,
            ot=ot,
            ot_logical=ot_logical,
            obs_in_sample=obs_in_sample,
            n_iterations=n_iterations,
            bounds=self.bounds,
            loss=self.loss,
            h=h,
            multisteps=multisteps,
            adam_cpp=adam_cpp,
        )

        # Snapshot for the likelihood re-evaluation, taken before the optimiser
        # (and the final fit) write through the shared matrices.
        ll_kwargs = {**_pristine(cf_kwargs), "loss": "likelihood", "bounds": "none"}

        def objective(x, grad):
            return ces_cf(B=x, **cf_kwargs)

        # Stage 1: BOBYQA — R lines 853-857
        algo_map = {
            "NLOPT_LN_BOBYQA": nlopt.LN_BOBYQA,
            "NLOPT_LN_NELDERMEAD": nlopt.LN_NELDERMEAD,
            "NLOPT_LN_SBPLX": nlopt.LN_SBPLX,
            "NLOPT_LN_COBYLA": nlopt.LN_COBYLA,
        }

        if len(B) == 0:
            # Nothing left to estimate: `a` (and `b`) supplied together with
            # backcast/complete initials and no xreg leaves an empty parameter
            # vector. R switches ``modelDo`` to "use" there and evaluates the
            # cost once instead of optimising (R/adam-ces.R:633-636). nlopt
            # cannot be constructed with zero dimensions -- it raises
            # ``invalid_argument`` -- so the optimiser is skipped entirely.
            cf_value = float(ces_cf(B=B, **cf_kwargs))
        else:
            opt1 = nlopt.opt(
                algo_map.get(nlopt_settings["algorithm0"], nlopt.LN_BOBYQA), len(B)
            )
            opt1.set_min_objective(objective)
            opt1.set_lower_bounds(np.full(len(B), -np.inf))
            opt1.set_upper_bounds(np.full(len(B), np.inf))
            opt1.set_maxeval(maxeval0_used)
            opt1.set_xtol_rel(nlopt_settings["xtol_rel0"])
            opt1.set_xtol_abs(nlopt_settings["xtol_abs0"])
            opt1.set_ftol_rel(nlopt_settings["ftol_rel0"])
            opt1.set_ftol_abs(nlopt_settings["ftol_abs0"])
            opt1.set_maxtime(nlopt_settings["maxtime0"])
            try:
                B = opt1.optimize(B)
            except nlopt.RoundoffLimited:
                B = B.copy()

            # Stage 2: Nelder-Mead — R lines 866-870
            opt2 = nlopt.opt(
                algo_map.get(nlopt_settings["algorithm"], nlopt.LN_NELDERMEAD), len(B)
            )
            opt2.set_min_objective(objective)
            opt2.set_lower_bounds(np.full(len(B), -np.inf))
            opt2.set_upper_bounds(np.full(len(B), np.inf))
            opt2.set_maxeval(maxeval_used)
            opt2.set_xtol_rel(nlopt_settings["xtol_rel"])
            opt2.set_xtol_abs(nlopt_settings["xtol_abs"])
            opt2.set_ftol_rel(nlopt_settings["ftol_rel"])
            opt2.set_ftol_abs(nlopt_settings["ftol_abs"])
            opt2.set_maxtime(nlopt_settings["maxtime"])
            try:
                B = opt2.optimize(B)
            except nlopt.RoundoffLimited:
                B = B.copy()

            cf_value = opt2.last_optimum_value()

        # --- Final fit with optimized B --- R lines 931-996

        # Fill matrices one final time
        elements = ces_filler(
            B=B,
            mat_vt=mat_vt,
            mat_f=mat_f,
            vec_g=vec_g,
            a=a,
            b=b,
            seasonality=self.seasonality,
            n_seasonal=n_seasonal,
            lags_model_seasonal=lags_model_seasonal,
            lags_model_max=lags_model_max,
            initial_type=initial_type,
            xreg_model=xreg_model,
            xreg_number=xreg_number,
            initial_xreg_estimate=xreg_model,
            components_number=components_number,
        )
        mat_f = elements["mat_f"]
        vec_g = elements["vec_g"]
        mat_vt[:, :lags_model_max] = elements["vt"]
        profiles_recent_table[:] = elements["vt"]
        profiles_recent_initial = elements["vt"].copy()

        # Final fit — additive SSOE gradient solve for initial="gradient".
        from smooth.adam_general.core.utils.gradient import adam_fit_or_gradient

        adam_fitted = adam_fit_or_gradient(
            adam_cpp=adam_cpp,
            mat_vt=np.asfortranarray(mat_vt, dtype=np.float64),
            mat_wt=np.asfortranarray(mat_wt, dtype=np.float64),
            mat_f=np.asfortranarray(mat_f, dtype=np.float64),
            vec_g=np.asfortranarray(vec_g.ravel(), dtype=np.float64),
            index_lookup_table=np.asfortranarray(index_lookup_table, dtype=np.uint64),
            profiles_recent_table=np.asfortranarray(
                profiles_recent_table, dtype=np.float64
            ),
            y_in_sample=np.asfortranarray(y_in_sample, dtype=np.float64).ravel(),
            ot=np.asfortranarray(ot, dtype=np.float64).ravel(),
            initial_type=initial_type,
            n_iterations=int(n_iterations),
            backcast_value=initial_type in ("complete", "backcasting", "gradient"),
            model_type_dict={
                "ets_model": False,
                "arima_model": True,
                "xreg_model": bool(xreg_model),
                "error_type": "A",
                "trend_type": "N",
                "season_type": "N",
                "model_is_trendy": False,
                "model_is_seasonal": False,
            },
            components_dict={
                "components_number_ets": 0,
                "components_number_ets_seasonal": 0,
                "components_number_ets_non_seasonal": 0,
                "components_number_arima": int(components_number),
            },
            lags_dict={
                "lags_model_max": int(lags_model_max),
                "lags_model": lags_model_all,
                "lags_model_all": lags_model_all,
                "lags_model_seasonal": lags_model_seasonal,
            },
            obs_in_sample=obs_in_sample,
            o_type="n",
            loss=self.loss,
            distribution="dnorm",
        )

        errors = np.array(adam_fitted.errors).ravel()
        y_fitted = np.array(adam_fitted.fitted).ravel()
        profiles_recent_table = np.array(adam_fitted.profile)
        mat_vt = np.array(adam_fitted.states).T  # C++ returns (components, time)

        # Scale, sigma^2 as in the ADAM monograph -- R's scaler() in ces()
        scale = _sum_r(errors[ot_logical] ** 2) / obs_observed
        # No errors at the missing values, where the fitted values are the predictions
        errors[~observed] = np.nan

        # Reconstruct complex a and b from B — R lines 1048-1093
        n_coefficients = 0
        if a["estimate"]:
            if self.seasonality != "simple":
                a["value"] = complex(B[0], B[1])
                n_coefficients = 2
            else:
                a_vals = []
                for i in range(n_seasonal):
                    a_vals.append(
                        complex(
                            B[n_coefficients + 2 * i], B[n_coefficients + 2 * i + 1]
                        )
                    )
                a["value"] = a_vals if n_seasonal > 1 else a_vals[0]
                n_coefficients += 2 * n_seasonal

        if b["estimate"]:
            if self.seasonality == "partial":
                b_vals = B[n_coefficients : n_coefficients + n_seasonal]
                b["value"] = b_vals.tolist() if n_seasonal > 1 else float(b_vals[0])
                n_coefficients += n_seasonal
            elif self.seasonality == "full":
                b_vals = []
                for i in range(n_seasonal):
                    b_vals.append(
                        complex(
                            B[n_coefficients + 2 * i],
                            B[n_coefficients + 2 * i + 1],
                        )
                    )
                b["value"] = b_vals if n_seasonal > 1 else b_vals[0]
                n_coefficients += 2 * n_seasonal

        # Initial values — R lines 1021-1044
        # mat_vt is (time, components) after transpose
        initial_states = {}
        if self.seasonality == "none":
            initial_states["nonseasonal"] = mat_vt[0, 0:2]
        elif self.seasonality == "simple":
            initial_states["seasonal"] = mat_vt[:lags_model_max, : n_seasonal * 2]
        else:
            initial_states["nonseasonal"] = mat_vt[0, 0:2]
            seasonal_mask = np.array(lags_model_all[:components_number]) != 1
            initial_states["seasonal"] = mat_vt[:lags_model_max, :components_number][
                :, seasonal_mask
            ]
        if xreg_model:
            initial_states["xreg"] = mat_vt[
                0, components_number : components_number + xreg_number
            ]

        # Model name — R lines 1096-1104
        model_name = "CES"
        if xreg_model:
            model_name += "X"
        model_name += f"({self.seasonality})"

        # Log-likelihood. The reported value is a concentrated likelihood, never
        # -loss. For a fit-only loss it is the *Normal* likelihood re-evaluated
        # at the fitted parameters: CES is a Normal-error model throughout --
        # its scale, its density and its prediction intervals are all Normal --
        # so there is no loss-implied distribution to switch to. This is what
        # ES does, pinning distribution="dnorm" for the same reason
        # (R/adam-es.R:417); ADAM follows the loss only because it has a
        # distribution argument to follow. A multistep loss instead reports the
        # predictive likelihood of the GPL paper. Mirrors R/adam-ces.R.
        if multisteps:
            log_lik_value = multistep_log_lik(cf_value, self.loss, h, observed)
        else:
            log_lik_value = -float(ces_cf(B=B, **_pristine(ll_kwargs)))

        # Initials count in the df however obtained (optimised or backcast/
        # complete/gradient) -- the same identifiable count either way. CES has
        # no multi-seasonal shared-frequency redundancy, so it is the plain
        # structural count. The scale is always estimated (concentrated
        # likelihood). Mirrors R (R/adam-ces.R).
        ces_initial_count = (
            2 * (self.seasonality != "simple")
            + lags_model_max * (self.seasonality != "none")
            + lags_model_max * (self.seasonality in ("full", "simple"))
        )
        n_states_backcasting = (
            ces_initial_count
            if initial_type in ("backcasting", "complete", "gradient")
            else 0
        )
        n_param_estimated = len(B) + 1 + n_states_backcasting

        # Information criteria (reuse existing utilities)
        self.aic = AIC(log_lik_value, nobs=obs_observed, df=n_param_estimated)
        self.aicc = AICc(log_lik_value, nobs=obs_observed, df=n_param_estimated)
        self.bic = BIC(log_lik_value, nobs=obs_observed, df=n_param_estimated)
        self.bicc = BICc(log_lik_value, nobs=obs_observed, df=n_param_estimated)

        # Store all results, under ADAM's names
        self.fitted = y_fitted
        self.residuals = errors
        # Components in rows, without the extra backcasting head, as ADAM's
        self.states = mat_vt.T[:, -(obs_in_sample + lags_model_max) :]
        self.model_name = model_name
        self.a_ = a["value"]
        self.b_ = b["value"]
        self.coef = np.array(B)
        self.loglik = log_lik_value
        self.loss_value = cf_value
        self.scale = scale
        self.initial_value = initial_states
        self.initial_type = initial_type
        self.persistence_vector = vec_g[:, 0]
        self.transition = mat_f
        self.measurement = mat_wt
        self.nparam = n_param_estimated
        self._ll_kwargs = ll_kwargs
        self.fisher_information_ = None
        if self.fi and len(B) > 0:
            self.fisher_information_ = self._fisher_information_matrix(
                self.step_size or float(np.finfo(float).eps ** 0.25)
            )
        self.time_elapsed_ = time.time() - start_time
        self._b_names = b_names
        self._y_actuals = y_actuals
        self._lags = lags
        self._obs_observed = obs_observed

        # Store internals for predict
        self._mat_vt = mat_vt
        self._mat_f = mat_f
        self._mat_wt = mat_wt
        self._vec_g = vec_g
        self._profiles_recent_table = profiles_recent_table
        self._profiles_recent_initial = profiles_recent_initial
        self._index_lookup_table = index_lookup_table
        self._adam_cpp = adam_cpp
        self._lags_model_all = lags_model_all
        self._lags_model_max = lags_model_max
        self._head_geometry = head_length
        self._components_number = components_number
        self._xreg_number = xreg_number
        self._obs_in_sample = obs_in_sample
        self._y_in_sample = y_in_sample
        self._y_na_values = y_na_values
        self._y_holdout = y_holdout
        self._y_frequency = y_frequency
        self._h = h

        # The internals of ADAM that its methods read, as R's methods of adam() read
        # the elements of the ces() models
        seasonal_lags = [lag for lag in lags_model_all if lag != 1]
        n_scale = int(self.loss == "likelihood")
        self._prepared = {
            # States without the extra backcasting head, as R stores them
            "states": self.states,
            "mat_vt": profiles_recent_initial,
            "measurement": mat_wt,
            "transition": mat_f,
            "persistence": vec_g,
            "profiles_recent_table": profiles_recent_table,
            "residuals": pd.Series(np.asarray(errors, dtype=float)),
            "y_fitted": np.asarray(y_fitted, dtype=float).copy(),
            "scale": scale,
        }
        self._observations = {
            "obs_in_sample": obs_in_sample,
            "y_in_sample": np.asarray(y_in_sample, dtype=float),
            "y_holdout": y_holdout,
            "y_na_values": y_na_values,
            "ot_logical": ot_logical,
            "y_forecast_start": 1,
            "frequency": y_frequency,
        }
        self._general = {
            "h": h,
            "holdout": self.holdout,
            "distribution": "dnorm",
            "distribution_new": "dnorm",
            "loss": self.loss,
            "ic": self.ic,
            "bounds": self.bounds,
            "other": {},
            "n_param": None,
            "scale_forecast": None,
            "parameters_number": [
                [n_param_estimated - n_scale, n_scale, n_param_estimated]
            ],
        }
        self._lags_model = {
            "lags": lags_model_all,
            "lags_model_all": lags_model_all,
            "lags_model_max": lags_model_max,
            "lags_model_min": min(seasonal_lags) if seasonal_lags else np.inf,
        }
        self._model_type = {
            "ets_model": False,
            "arima_model": False,
            "xreg_model": xreg_model,
            "error_type": "A",
            "trend_type": "N",
            "season_type": "N",
            "damped": False,
            "model": model_name,
            "model_do": "estimate",
        }
        self._components = {
            "components_number_ets": 0,
            "components_number_ets_seasonal": 0,
            "components_number_ets_non_seasonal": 0,
            "components_number_arima": components_number,
        }
        self._explanatory = {"xreg_model": xreg_model, "xreg_number": xreg_number}
        self._constant = {"constant_required": False}
        self._occurrence = {"occurrence_model": False, "occurrence": "none"}
        self._adam_created = {"mat_wt": mat_wt, "mat_f": mat_f, "vec_g": vec_g}

        return self

    def predict(
        self,
        h: Optional[int] = None,
        X: Optional[NDArray] = None,
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
        scenarios: bool = False,
    ) -> ForecastResult:
        """
        Generate point forecasts and prediction intervals from the fitted CES model.

        CES is forecast by ADAM's forecaster, as R's ``ces()`` returns an ``adam``
        object forecast by ``forecast.adam()``. CES is a pure additive model, so
        ``interval="prediction"`` resolves to the analytical (``"approximate"``)
        interval with Normal quantiles.

        Parameters
        ----------
        h : int or None
            Forecast horizon. If None, uses the h from fit.
        X : array-like or None
            Future exogenous regressors.
        interval : str, default="none"
            ``"none"``, ``"prediction"``, ``"approximate"``, ``"simulated"``,
            ``"semiparametric"``, ``"nonparametric"`` or ``"empirical"``, as in
            :meth:`ADAM.predict`.
        level : float or list of float, default=0.95
            Confidence level(s) for intervals.
        side : str, default="both"
            ``"both"``, ``"upper"`` or ``"lower"``.
        cumulative : bool, default=False
            If True, forecast the sum over the horizon.
        nsim : int, default=10000
            Number of paths for ``interval="simulated"``.
        scenarios : bool, default=False
            Keep the simulated paths, as ``ADAM.predict``.

        Returns
        -------
        ForecastResult
            Forecast result with ``.mean``, ``.lower`` and ``.upper``.
        """
        if not hasattr(self, "_adam_cpp"):
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        if h is None:
            h = self._h if self._h > 0 else 1
        self._general.update(
            h=int(h),
            cumulative=cumulative,
            nsim=nsim,
            scenarios=scenarios,
            interval=interval,
            level=level,
        )

        new_xreg = None
        if X is not None and self._xreg_number > 0:
            new_xreg = np.asarray(X, dtype=np.float64)[:h].reshape(h, -1)

        return forecaster(
            model_prepared=dict(self._prepared),
            observations_dict=self._observations,
            # Updated in place, as ADAM's: the simulated paths are kept there
            general_dict=self._general,
            occurrence_dict={"occurrence_model": False, "occurrence": "none"},
            lags_dict=self._lags_model,
            model_type_dict=self._model_type,
            explanatory_checked={**self._explanatory, "new_xreg": new_xreg},
            components_dict=self._components,
            constants_checked={"constant_required": False},
            params_info=self._general["parameters_number"],
            adam_cpp=self._adam_cpp,
            interval=interval,
            level=level,
            side=side,
        )

    def point_lik(self, log: bool = True) -> NDArray:
        """Per-observation log-likelihood of the fitted model.

        R's ``pointLik.adam`` for a ``ces()`` model: the Normal log-density of
        each in-sample observation around its fitted value with the variance
        ``scale``, zero at the missing values. With ``log=False`` the densities
        themselves are returned.
        """
        from smooth.adam_general.core.utils.utils import calculate_likelihood

        if not hasattr(self, "model_name"):
            raise RuntimeError("Model has not been fitted yet.")
        y = self.fitted + self.residuals
        lik_values = np.ravel(
            calculate_likelihood(
                "dnorm", "A", y, self.fitted.reshape(-1, 1), self.scale, None
            )
        )
        lik_values[np.isnan(self.residuals)] = 0
        return lik_values if log else np.exp(lik_values)

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the fitted model."""
        if not hasattr(self, "model_name"):
            raise RuntimeError("Model has not been fitted yet.")
        return {
            "model": self.model_name,
            "a": self.a_,
            "b": self.b_,
            "loss": self.loss,
            "loss_value": self.loss_value,
            "logLik": self.loglik,
            "AIC": self.aic,
            "AICc": self.aicc,
            "BIC": self.bic,
            "BICc": self.bicc,
            "nParam": self.nparam,
            "scale": self.scale,
        }

    # The properties of ADAM that CES has
    def _check_fitted(self) -> None:
        if not hasattr(self, "model_name"):
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

    # The methods of ADAM, as R's methods of adam() take the ces() models: these need
    # only the residuals, the scale and the distribution
    _check_is_fitted = _check_fitted
    rstandard = ADAM.rstandard
    rstudent = ADAM.rstudent
    outlierdummy = ADAM.outlierdummy
    extract_scale = ADAM.extract_scale
    extract_sigma = ADAM.extract_sigma
    multicov = ADAM.multicov
    rmultistep = ADAM.rmultistep
    _multicov_analytical = ADAM._multicov_analytical
    _multicov_simulated = ADAM._multicov_simulated
    _multicov_empirical = ADAM._multicov_empirical
    vcov = ADAM.vcov
    confint = ADAM.confint
    simulate = ADAM.simulate

    def _simulate_state_head(self, lags_model_max):
        """The profile the fit started from, R's ``$profileInitial`` of ces()."""
        return self._profiles_recent_initial

    reapply = ADAM.reapply
    reforecast = ADAM.reforecast
    coefbootstrap = ADAM.coefbootstrap
    _variance_debiased = ADAM._variance_debiased
    plot = ADAM.plot
    scale_model = None
    is_combined = False

    def _nobs_observed(self) -> int:
        """The observed values, which the likelihood counts."""
        return int(self._obs_observed)

    # The covariance of ADAM's vcov() for CES, as R's vcov.adam() does it for ces()
    def _fisher_information_matrix(self, step_size=None):
        """The Hessian of minus the log-likelihood at the estimates. R's vcov() takes
        the step 1e-8 for CES, which is sensitive."""
        step = 1e-8 if step_size is None else step_size
        return np.asarray(
            _hessian_cpp(
                lambda b: float(ces_cf(B=np.asarray(b), **_pristine(self._ll_kwargs))),
                np.asarray(self.coef, dtype=float),
                step,
            )
        )

    def _opg_covariance(self, step_size=None):
        """The OPG covariance, as R's covarOPGces(): the log-densities of the data
        at the perturbed parameters, with the scale of each refit."""
        from smooth.adam_general.core.utils.utils import calculate_likelihood
        from smooth.adam_general.core.utils.var_covar import covar_opg

        observed = np.asarray(self._observations["ot_logical"], dtype=bool)

        def point_lik_at(b):
            result = ces_cf(
                B=np.asarray(b, dtype=float),
                **_pristine(self._ll_kwargs),
                return_fitted=True,
            )
            if not isinstance(result, tuple):
                return np.full(len(observed), np.nan)
            fitted, errors = result
            scale = _sum_r(errors[observed] ** 2) / np.sum(observed)
            values = np.ravel(
                calculate_likelihood(
                    "dnorm", "A", fitted + errors, fitted.reshape(-1, 1), scale, None
                )
            )
            values[~observed] = 0
            return values

        return covar_opg(
            np.asarray(self.coef, dtype=float),
            point_lik_at,
            self.nobs,
            self.loglik,
            step_size,
        )

    def _clamp_confint_offsets(self, names, params, lo, hi):
        """No bounds on the intervals of CES, as in R's confint.adam()."""
        return None

    @property
    def _df_scale(self) -> float:
        """The degrees of freedom of the scale (R's ``adam_dfScale``): the observed
        values minus the parameters, without the scale under the likelihood."""
        return float(self._obs_observed - self.nparam + int(self.loss == "likelihood"))

    @property
    def coef_names(self) -> List[str]:
        """The names of the parameters in ``coef``, as R's ``names(B)``."""
        self._check_fitted()
        return list(self._b_names)

    @property
    def b_value(self) -> NDArray:
        """The parameter vector B (R's ``$B``), as ``coef``."""
        return self.coef

    @property
    def n_param(self) -> NParam:
        """The table of the numbers of parameters (R's ``$nParam``)."""
        self._check_fitted()
        xreg = self._xreg_number
        n_param = NParam.from_dict(
            {
                "estimated": {
                    "internal": int(self.nparam) - 1 - xreg,
                    "xreg": xreg,
                    "scale": 1,
                }
            }
        )
        n_param.update_totals()
        return n_param

    @property
    def nobs(self) -> int:
        """The number of in-sample observations."""
        self._check_fitted()
        return int(self._obs_in_sample)

    @property
    def loss_(self) -> str:
        """The loss used (R's ``$loss``)."""
        return self.loss

    @property
    def distribution_(self) -> str:
        """The distribution of the error term (R's ``$distribution``): normal."""
        return "dnorm"

    @property
    def error_type(self) -> str:
        """The type of the error term: additive."""
        return "A"

    @property
    def sigma(self) -> float:
        """The standard error of the residuals, as R's ``sigma()``: over the observed
        values minus the parameters, without the scale under the likelihood."""
        self._check_fitted()
        residuals = np.asarray(self.residuals, dtype=float)
        residuals = residuals[np.isfinite(residuals)]
        return float(np.sqrt(np.sum(residuals**2) / self._df_scale))

    @property
    def profile(self) -> NDArray:
        """The profile of the states at the end of the sample (R's ``$profile``)."""
        self._check_fitted()
        return self._profiles_recent_table

    @property
    def time_elapsed(self) -> float:
        """The time of the fit in seconds (R's ``$timeElapsed``)."""
        self._check_fitted()
        return self.time_elapsed_

    @property
    def actuals(self) -> NDArray:
        """The in-sample data, with their missing values."""
        self._check_fitted()
        return self._y_actuals

    @property
    def data(self) -> NDArray:
        """The in-sample data (R's ``$data``), as ``actuals``."""
        return self.actuals

    @property
    def holdout_data(self) -> Optional[NDArray]:
        """The holdout data (R's ``$holdout``), or None."""
        self._check_fitted()
        return self._y_holdout

    @property
    def lags_used(self) -> List[int]:
        """The lags of the model."""
        self._check_fitted()
        return list(self._lags)


class AutoCES:
    """
    Automatic CES model selection across seasonality types.

    Translates R/autoces.R. Fits candidate seasonality types and selects
    the best by information criterion, with sample size validation.

    Parameters
    ----------
    seasonality : list of str or None
        Pool of seasonality types to try. None = all four.
    lags : list of int or None
        Seasonal period(s).
    initial : str, default="backcasting"
        Initialization method.
    ic : str, default="AICc"
        Information criterion for selection.
    loss : str, default="likelihood"
        Loss function.
    h : int or None
        Forecast horizon.
    holdout : bool, default=False
        Whether to use holdout.
    bounds : str, default="admissible"
        Parameter bounds.
    verbose : int, default=0
        Verbosity level.
    **kwargs
        Additional arguments passed to CES.
    """

    def __init__(
        self,
        seasonality: Optional[List[str]] = None,
        lags: Optional[List[int]] = None,
        initial: str = "backcasting",
        ic: Literal["AIC", "AICc", "BIC", "BICc"] = "AICc",
        loss: LOSS_OPTIONS = "likelihood",
        h: Optional[int] = None,
        holdout: bool = False,
        bounds: Literal["admissible", "none"] = "admissible",
        verbose: int = 0,
        **kwargs,
    ) -> None:
        self.seasonality = seasonality
        self.lags = lags
        self.initial = initial
        self.ic = ic
        self.loss = loss
        self.h = h
        self.holdout = holdout
        self.bounds = bounds
        self.verbose = verbose
        self._kwargs = kwargs

    def fit(self, y: NDArray, X: Optional[NDArray] = None) -> CES:
        """
        Fit CES models for each seasonality type and return the best, as R's
        ``auto.ces()`` returns a ``ces`` object.

        Parameters
        ----------
        y : array-like
            Time series data.
        X : array-like or None
            Exogenous regressors.

        Returns
        -------
        CES
            The selected model, with the information criteria of all the
            candidates in ``ICs`` and the time of the selection in
            ``time_elapsed``.
        """
        _check_loss(self.loss)
        start_time = time.time()
        y = np.asarray(y, dtype=np.float64).ravel()

        # Determine lags and frequency
        lags = self.lags if self.lags is not None else [1]
        y_frequency = max(lags)

        # Validate and normalize seasonality pool — R lines 68-76
        valid_full = {"none", "simple", "partial", "full"}
        abbrev = {"n": "none", "s": "simple", "p": "partial", "f": "full"}

        if self.seasonality is None:
            pool = ["none", "simple", "partial", "full"]
        else:
            pool = []
            all_ok = True
            for s in self.seasonality:
                s_norm = abbrev.get(s, s)
                if s_norm in valid_full:
                    pool.append(s_norm)
                else:
                    all_ok = False
            if not all_ok:
                warnings.warn(
                    "The pool of models includes a strange type of model! "
                    "Reverting to default pool."
                )
                pool = ["none", "simple", "partial", "full"]

        h = self.h if self.h is not None else 0
        obs_in_sample = len(y) - (self.holdout and h > 0) * h
        initial = self.initial

        # Frequency=1 shortcut — R lines 126-132
        if y_frequency == 1:
            if self.verbose > 0:
                print("The data is not seasonal. Simple CES was the only solution.")
            pool = ["none"]

        # Sample size pruning — R lines 88-146
        pruned = []
        for s in pool:
            if s == "none":
                n_param_max = 3
                if initial in ("optimal", "two-stage"):
                    n_param_max += 2
            elif s == "partial":
                n_param_max = 4
                if initial in ("optimal", "two-stage"):
                    n_param_max += 2 + y_frequency
                # A seasonal candidate needs at least one full cycle of
                # observations, otherwise the head of the lookup table cannot
                # carry a full set of profile cells (R/autoces.R).
                if obs_in_sample <= n_param_max or obs_in_sample < y_frequency:
                    warnings.warn(
                        "The sample is too small. Cannot use partial seasonal model."
                    )
                    continue
                if obs_in_sample <= y_frequency + 2 + 3 + 1:
                    warnings.warn("Not enough observations for CES(partial).")
                    continue
            elif s == "simple":
                n_param_max = 3
                if initial in ("optimal", "two-stage"):
                    n_param_max += 2 * y_frequency
                # A seasonal candidate needs at least one full cycle of
                # observations, otherwise the head of the lookup table cannot
                # carry a full set of profile cells (R/autoces.R).
                if obs_in_sample <= n_param_max or obs_in_sample < y_frequency:
                    warnings.warn(
                        "The sample is too small. Cannot use simple seasonal model."
                    )
                    continue
                if obs_in_sample <= y_frequency * 2 + 2 + 1:
                    warnings.warn("Not enough observations for CES(simple).")
                    continue
            elif s == "full":
                n_param_max = 5
                if initial in ("optimal", "two-stage"):
                    n_param_max += 2 + 2 * y_frequency
                # A seasonal candidate needs at least one full cycle of
                # observations, otherwise the head of the lookup table cannot
                # carry a full set of profile cells (R/autoces.R).
                if obs_in_sample <= n_param_max or obs_in_sample < y_frequency:
                    warnings.warn(
                        "The sample is too small. Cannot use full seasonal model."
                    )
                    continue
                if obs_in_sample <= y_frequency * 2 + 2 + 4 + 1:
                    warnings.warn("Not enough observations for CES(full).")
                    continue
            pruned.append(s)

        pool = pruned
        if not pool:
            pool = ["none"]

        # Fit each candidate — R lines 160-167
        ic_func = {"AIC": AIC, "AICc": AICc, "BIC": BIC, "BICc": BICc}[self.ic]

        models = {}
        ics = {}
        for s in pool:
            if self.verbose > 0:
                print(f'Estimating CES with seasonality: "{s}" ', end="")
            model = CES(
                seasonality=cast(SEASONALITY_OPTIONS, s),
                lags=self.lags,
                initial=self.initial,
                loss=self.loss,
                h=self.h,
                holdout=self.holdout,
                bounds=self.bounds,
                verbose=0,
                **self._kwargs,
            )
            try:
                model.fit(y, X=X)
                models[s] = model
                # Over the observed values, as the likelihood
                ics[s] = ic_func(
                    model.loglik,
                    nobs=int(np.sum(~model._y_na_values[:obs_in_sample])),
                    df=model.nparam,
                )
            except Exception as e:
                if self.verbose > 0:
                    print(f"[failed: {e}]")
                continue

        if not models:
            raise RuntimeError("All CES models failed to fit.")

        # Select best — R lines 170-186
        best_key = min(ics, key=lambda k: ics[k])
        best = models[best_key]
        best.ICs = ics
        best.time_elapsed_ = time.time() - start_time

        if self.verbose > 0:
            print(f'\nThe best model is with seasonality = "{best_key}"')

        return best
