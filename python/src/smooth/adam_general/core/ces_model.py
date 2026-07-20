"""
Complex Exponential Smoothing (CES) model.

Translates R/adam-ces.R ces() and R/autoces.R auto.ces().
Self-contained module with its own fit pipeline, reusing adamCore C++ for
state-space filtering and forecasting.
"""

import re
import warnings
from typing import Any, Dict, List, Literal, Optional, Union, cast

import nlopt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from smooth.adam_general import _adamCore
from smooth.adam_general.core.ces.cost_function import ces_cf
from smooth.adam_general.core.ces.creator import ces_creator
from smooth.adam_general.core.ces.filler import ces_filler
from smooth.adam_general.core.ces.initialiser import ces_initialiser
from smooth.adam_general.core.forecaster.result import ForecastResult
from smooth.adam_general.core.utils.ic import AIC, BIC, AICc, BICc

SEASONALITY_OPTIONS = Literal["none", "simple", "partial", "full"]
LOSS_OPTIONS = Literal[
    "likelihood", "MSE", "MAE", "HAM", "MSEh", "TMSE", "GTMSE", "MSCE", "GPL"
]
_CES_NLOPT_WARNING_SHOWN = False


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
    algorithm0 : str, default="NLOPT_LN_BOBYQA"
        First-stage optimizer algorithm.
    algorithm : str, default="NLOPT_LN_NELDERMEAD"
        Second-stage optimizer algorithm.
    maxeval : int or None
        Maximum second-stage optimizer evaluations. None = 40*len(B).
    maxeval0 : int or None
        Maximum first-stage optimizer evaluations. None = maxeval default.
    xtol_rel0 : float, default=1e-8
        Relative tolerance for first-stage optimizer.
    xtol_abs0 : float, default=0
        Absolute tolerance for first-stage optimizer.
    ftol_rel0 : float, default=0
        Relative objective tolerance for first-stage optimizer.
    ftol_abs0 : float, default=0
        Absolute objective tolerance for first-stage optimizer.
    xtol_rel : float, default=1e-6
        Relative tolerance for second-stage optimizer.
    xtol_abs : float, default=1e-8
        Absolute tolerance for second-stage optimizer.
    ftol_rel : float, default=1e-8
        Relative objective tolerance for second-stage optimizer.
    ftol_abs : float, default=0
        Absolute objective tolerance for second-stage optimizer.
    """

    # Attributes set during .fit() (declared here so mypy can resolve them
    # when one CES instance reads another's fitted state, e.g. two-stage init).
    B: NDArray
    initial_states_: Dict[str, NDArray]

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
        algorithm0: str = "NLOPT_LN_BOBYQA",
        algorithm: str = "NLOPT_LN_NELDERMEAD",
        maxeval: Optional[int] = None,
        maxeval0: Optional[int] = None,
        maxtime: float = -1,
        maxtime0: float = -1,
        xtol_rel0: float = 1e-8,
        xtol_abs0: float = 0,
        ftol_rel0: float = 0,
        ftol_abs0: float = 0,
        xtol_rel: float = 1e-6,
        xtol_abs: float = 1e-8,
        ftol_rel: float = 1e-8,
        ftol_abs: float = 0,
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
        self._a_provided = a
        self._b_provided = b
        self.loss = loss
        self.h = h
        self.holdout = holdout
        self.bounds = bounds
        self.ic = ic
        self.verbose = verbose
        self.regressors = regressors
        self.algorithm0 = algorithm0
        self.algorithm = algorithm
        self.maxeval = maxeval
        self.maxeval0 = maxeval0
        self.maxtime = maxtime
        self.maxtime0 = maxtime0
        self.xtol_rel0 = xtol_rel0
        self.xtol_abs0 = xtol_abs0
        self.ftol_rel0 = ftol_rel0
        self.ftol_abs0 = ftol_abs0
        self.xtol_rel = xtol_rel
        self.xtol_abs = xtol_abs
        self.ftol_rel = ftol_rel
        self.ftol_abs = ftol_abs

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
        y = np.asarray(y, dtype=np.float64).ravel()

        # CES parity with R depends on the stage-1 BOBYQA trajectory.
        # ETS / ARIMA do not use this two-stage CES path, so keep the check local.
        if self.algorithm0 == "NLOPT_LN_BOBYQA":
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

        obs_all = len(y)
        obs_states = obs_in_sample + lags_model_max

        # Occurrence (CES doesn't support occurrence — R line 618)
        ot = np.ones(obs_in_sample, dtype=np.float64)
        ot_logical = np.ones(obs_in_sample, dtype=bool)

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
                algorithm0=self.algorithm0,
                algorithm=self.algorithm,
                maxeval=self.maxeval,
                maxeval0=self.maxeval0,
                maxtime=self.maxtime,
                maxtime0=self.maxtime0,
                xtol_rel0=self.xtol_rel0,
                xtol_abs0=self.xtol_abs0,
                ftol_rel0=self.ftol_rel0,
                ftol_abs0=self.ftol_abs0,
                xtol_rel=self.xtol_rel,
                xtol_abs=self.xtol_abs,
                ftol_rel=self.ftol_rel,
                ftol_abs=self.ftol_abs,
            )
            ces_back.fit(y, X=X)
            B = ces_back.B.copy()

            # Append initial state estimates — R lines 769-785
            if self.seasonality != "simple":
                B = np.concatenate([B, ces_back.initial_states_["nonseasonal"]])
            if self.seasonality != "none":
                seasonal_init = ces_back.initial_states_.get("seasonal")
                if seasonal_init is not None:
                    B = np.concatenate([B, seasonal_init.ravel(order="F")])
            if xreg_model and "xreg" in ces_back.initial_states_:
                B = np.concatenate([B, ces_back.initial_states_["xreg"]])

        # Build initial B if not from two-stage — R line 788-789
        if B is None:
            B = ces_initialiser(
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

        # Maxeval — R lines 800-807
        maxeval_used = self.maxeval
        if maxeval_used is None:
            maxeval_used = len(B) * 40
            if xreg_model:
                maxeval_used = max(1000, len(B) * 100)
        maxeval0_used = self.maxeval0 if self.maxeval0 is not None else maxeval_used

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

        def objective(x, grad):
            return ces_cf(B=x, **cf_kwargs)

        # Stage 1: BOBYQA — R lines 853-857
        algo_map = {
            "NLOPT_LN_BOBYQA": nlopt.LN_BOBYQA,
            "NLOPT_LN_NELDERMEAD": nlopt.LN_NELDERMEAD,
            "NLOPT_LN_SBPLX": nlopt.LN_SBPLX,
            "NLOPT_LN_COBYLA": nlopt.LN_COBYLA,
        }

        opt1 = nlopt.opt(algo_map.get(self.algorithm0, nlopt.LN_BOBYQA), len(B))
        opt1.set_min_objective(objective)
        opt1.set_lower_bounds(np.full(len(B), -np.inf))
        opt1.set_upper_bounds(np.full(len(B), np.inf))
        opt1.set_maxeval(maxeval0_used)
        opt1.set_xtol_rel(self.xtol_rel0)
        opt1.set_xtol_abs(self.xtol_abs0)
        opt1.set_ftol_rel(self.ftol_rel0)
        opt1.set_ftol_abs(self.ftol_abs0)
        opt1.set_maxtime(self.maxtime0)
        try:
            B = opt1.optimize(B)
        except nlopt.RoundoffLimited:
            B = B.copy()

        # Stage 2: Nelder-Mead — R lines 866-870
        opt2 = nlopt.opt(algo_map.get(self.algorithm, nlopt.LN_NELDERMEAD), len(B))
        opt2.set_min_objective(objective)
        opt2.set_lower_bounds(np.full(len(B), -np.inf))
        opt2.set_upper_bounds(np.full(len(B), np.inf))
        opt2.set_maxeval(maxeval_used)
        opt2.set_xtol_rel(self.xtol_rel)
        opt2.set_xtol_abs(self.xtol_abs)
        opt2.set_ftol_rel(self.ftol_rel)
        opt2.set_ftol_abs(self.ftol_abs)
        opt2.set_maxtime(self.maxtime)
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

        # Scale — R line 1001
        scale = np.sqrt(np.sum(errors[ot_logical] ** 2) / obs_in_sample)

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

        # Point forecasts — R lines 1009-1017
        if h > 0:
            mat_wt_forecast = mat_wt[-h:]
            if mat_wt_forecast.shape[0] < h:
                mat_wt_forecast = np.tile(mat_wt[-1:], (h, 1))

            # Build forecast index lookup
            idx_start = lags_model_max + obs_in_sample
            idx_end = idx_start + h
            ilt_forecast = index_lookup_table[:, idx_start:idx_end]

            forecast_profiles = profiles_recent_table.copy()
            y_forecast = adam_cpp.forecast(
                matrixWt=np.asfortranarray(mat_wt_forecast, dtype=np.float64),
                matrixF=np.asfortranarray(mat_f, dtype=np.float64),
                indexLookupTable=np.asfortranarray(ilt_forecast, dtype=np.uint64),
                profilesRecent=np.asfortranarray(forecast_profiles, dtype=np.float64),
                horizon=int(h),
            ).forecast
            y_forecast = np.array(y_forecast).ravel()
        else:
            y_forecast = np.array([np.nan])

        # Log-likelihood — R stores the final optimizer objective as CFValue
        # and then defines logLik as -CFValue before the final fit.
        log_lik_value = -cf_value

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
        self.aic = AIC(log_lik_value, nobs=obs_in_sample, df=n_param_estimated)
        self.aicc = AICc(log_lik_value, nobs=obs_in_sample, df=n_param_estimated)
        self.bic = BIC(log_lik_value, nobs=obs_in_sample, df=n_param_estimated)
        self.bicc = BICc(log_lik_value, nobs=obs_in_sample, df=n_param_estimated)

        # Store all results
        self.B = np.array(B)
        self.fitted = y_fitted
        self.residuals = errors
        self.states = mat_vt
        self.forecast_ = y_forecast
        self.model_name = model_name
        self.a_ = a["value"]
        self.b_ = b["value"]
        self.coef = B
        self.loglik = log_lik_value
        self.loss_value = cf_value
        self.scale_ = scale
        self.initial_states_ = initial_states
        self.initial_type_ = initial_type
        self.persistence_vector = vec_g[:, 0]
        self.transition_matrix = mat_f
        self.measurement_matrix = mat_wt
        self.n_param = n_param_estimated

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
        self._components_number = components_number
        self._xreg_number = xreg_number
        self._obs_in_sample = obs_in_sample
        self._y_in_sample = y_in_sample
        self._y_holdout = y_holdout
        self._y_frequency = y_frequency
        self._h = h

        return self

    def predict(
        self,
        h: Optional[int] = None,
        X: Optional[NDArray] = None,
        interval: Optional[str] = None,
        level: Union[float, List[float]] = 0.95,
    ) -> ForecastResult:
        """
        Generate point forecasts from the fitted CES model.

        Parameters
        ----------
        h : int or None
            Forecast horizon. If None, uses the h from fit.
        X : array-like or None
            Future exogenous regressors.
        interval : str or None
            Not yet implemented for CES. Use None or "none".
        level : float or list of float
            Confidence level(s) for intervals.

        Returns
        -------
        ForecastResult
            Forecast result with .mean attribute.
        """
        if not hasattr(self, "_adam_cpp"):
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        if h is None:
            h = self._h if self._h > 0 else 1

        mat_wt = self._mat_wt
        mat_f = self._mat_f
        profiles_recent = self._profiles_recent_table
        index_lookup = self._index_lookup_table

        # Prepare forecast measurement matrix
        if h <= mat_wt.shape[0]:
            mat_wt_forecast = mat_wt[-h:]
        else:
            mat_wt_forecast = np.tile(mat_wt[-1:], (h, 1))

        # Handle xreg for forecast period
        if X is not None and self._xreg_number > 0:
            X_future = np.asarray(X, dtype=np.float64)[:h]
            mat_wt_forecast = mat_wt_forecast.copy()
            mat_wt_forecast[
                :, self._components_number : self._components_number + self._xreg_number
            ] = X_future

        # Forecast index lookup
        idx_start = self._lags_model_max + self._obs_in_sample
        idx_end = idx_start + h
        if idx_end <= index_lookup.shape[1]:
            ilt_forecast = index_lookup[:, idx_start:idx_end]
        else:
            # Extend by tiling
            available = index_lookup[:, idx_start:]
            needed = h - available.shape[1]
            if needed > 0:
                tile_src = index_lookup[
                    :,
                    self._lags_model_max : self._lags_model_max + self._obs_in_sample,
                ]
                extension = np.tile(tile_src, (1, (needed // tile_src.shape[1]) + 1))[
                    :, :needed
                ]
                ilt_forecast = np.hstack([available, extension])
            else:
                ilt_forecast = available[:, :h]

        y_forecast = self._adam_cpp.forecast(
            matrixWt=np.asfortranarray(mat_wt_forecast, dtype=np.float64),
            matrixF=np.asfortranarray(mat_f, dtype=np.float64),
            indexLookupTable=np.asfortranarray(ilt_forecast, dtype=np.uint64),
            profilesRecent=np.asfortranarray(profiles_recent, dtype=np.float64),
            horizon=int(h),
        ).forecast
        y_forecast = np.array(y_forecast).ravel()

        forecast_index = pd.RangeIndex(start=1, stop=h + 1, name="h")
        mean_series = pd.Series(y_forecast, index=forecast_index, name="forecast")

        return ForecastResult(
            mean=mean_series,
            lower=None,
            upper=None,
            level=level,
            side="both",
            interval=interval if interval else "none",
        )

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
            "nParam": self.n_param,
            "scale": self.scale_,
        }


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

    def fit(self, y: NDArray, X: Optional[NDArray] = None) -> "AutoCES":
        """
        Fit CES models for each seasonality type and select the best.

        Parameters
        ----------
        y : array-like
            Time series data.
        X : array-like or None
            Exogenous regressors.

        Returns
        -------
        self
        """
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
                if obs_in_sample <= n_param_max:
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
                if obs_in_sample <= n_param_max:
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
                if obs_in_sample <= n_param_max:
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
                ics[s] = ic_func(model.loglik, nobs=obs_in_sample, df=model.n_param)
            except Exception as e:
                if self.verbose > 0:
                    print(f"[failed: {e}]")
                continue

        if not models:
            raise RuntimeError("All CES models failed to fit.")

        # Select best — R line 170
        best_key = min(ics, key=lambda k: ics[k])
        self.best_model_ = models[best_key]
        self.ICs = ics
        self.models_ = models
        self.model_name = self.best_model_.model_name

        if self.verbose > 0:
            print(f'\nThe best model is with seasonality = "{best_key}"')

        return self

    def predict(
        self,
        h: Optional[int] = None,
        X: Optional[NDArray] = None,
        **kwargs,
    ) -> ForecastResult:
        """Delegate prediction to the best model."""
        if not hasattr(self, "best_model_"):
            raise RuntimeError("AutoCES has not been fitted yet.")
        return self.best_model_.predict(h=h, X=X, **kwargs)

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the best model."""
        if not hasattr(self, "best_model_"):
            raise RuntimeError("AutoCES has not been fitted yet.")
        result = self.best_model_.summary()
        result["ICs"] = self.ICs
        return result
