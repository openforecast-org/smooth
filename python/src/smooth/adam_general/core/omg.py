"""General Occurrence Model (OMG): port of R's ``omg()`` to Python.

Fits two parallel ETS / ARIMA occurrence sub-models — A (odds-ratio) and
B (inverse-odds-ratio) — jointly, against a shared Bernoulli log-likelihood
on the combined probability ``p = aFit / (aFit + bFit)``.

Mirrors R/omg.R. The single C++ entry point ``adamCore.omfitGeneral`` advances
both sub-models simultaneously inside the optimiser.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Dict, List, Literal, Optional, Union

import nlopt
import numpy as np
from numpy.typing import NDArray

from smooth.adam_general.core.creator import initialiser
from smooth.adam_general.core.estimator.optimization import (
    _configure_optimizer,
    _setup_arima_polynomials,
)
from smooth.adam_general.core.om import (
    OM,
    om_preparator,
)
from smooth.adam_general.core.utils.ic import ic_function
from smooth.adam_general.core.utils.omg_cost import omg_cf, omg_link_function


class OMG:
    """General occurrence model — two parallel ETS sub-models combined.

    Public API:
        ``fit(y, X=None)`` → self
        ``predict(h, X=None)`` → ``ForecastResult``

    Attributes after fit:
        ``model_a`` : :class:`OM` — odds-ratio sub-model
        ``model_b`` : :class:`OM` — inverse-odds-ratio sub-model
        ``fitted``  : combined probabilities ∈ (0, 1)
        ``residuals`` : ``ot - fitted``
        ``loss_value``, ``loglik``, ``aic``/``aicc``/``bic``/``bicc``
        ``coef`` : joint parameter vector ``concat(B_A, B_B)``
        ``model_name`` : ``"oETS[G](MNN)(MNN)"``-style string
    """

    def __init__(
        self,
        model_a: str = "MNN",
        model_b: Optional[str] = None,
        lags: Optional[List[int]] = None,
        orders_a: Optional[Dict[str, Any]] = None,
        orders_b: Optional[Dict[str, Any]] = None,
        constant_a: bool = False,
        constant_b: Optional[bool] = None,
        formula_a: Optional[str] = None,
        formula_b: Optional[str] = None,
        regressors_a: Literal["use", "select", "adapt"] = "use",
        regressors_b: Optional[str] = None,
        persistence_a: Optional[Dict[str, float]] = None,
        persistence_b: Optional[Dict[str, float]] = None,
        phi_a: Optional[float] = None,
        phi_b: Optional[float] = None,
        arma_a: Optional[Dict[str, Any]] = None,
        arma_b: Optional[Dict[str, Any]] = None,
        h: int = 0,
        holdout: bool = False,
        initial: Union[str, Dict[str, Any]] = "backcasting",
        loss: Literal["likelihood"] = "likelihood",
        ic: Literal["AIC", "AICc", "BIC", "BICc"] = "AICc",
        bounds: Literal["usual", "admissible", "none"] = "usual",
        verbose: int = 0,
        nlopt_kargs: Optional[Dict[str, Any]] = None,
        ets: Literal["conventional", "adam"] = "conventional",
    ) -> None:
        if loss != "likelihood":
            # R/omg.R only ever uses Bernoulli likelihood for the joint cost
            raise ValueError(
                "OMG only supports loss='likelihood' (Bernoulli log-likelihood "
                "on the combined probability)."
            )

        self.model_a_spec = model_a
        self.model_b_spec = model_b if model_b is not None else model_a
        self.lags = lags
        self.orders_a = orders_a
        self.orders_b = orders_b if orders_b is not None else orders_a
        self.constant_a = constant_a
        self.constant_b = constant_b if constant_b is not None else constant_a
        self.formula_a = formula_a
        self.formula_b = formula_b if formula_b is not None else formula_a
        self.regressors_a = regressors_a
        self.regressors_b = regressors_b if regressors_b is not None else regressors_a
        self.persistence_a = persistence_a
        self.persistence_b = (
            persistence_b if persistence_b is not None else persistence_a
        )
        self.phi_a = phi_a
        self.phi_b = phi_b if phi_b is not None else phi_a
        self.arma_a = arma_a
        self.arma_b = arma_b if arma_b is not None else arma_a
        self.h = h
        self.holdout = holdout
        self.initial = initial
        self.loss = loss
        self.ic = ic
        self.bounds = bounds
        self.verbose = verbose
        self.nlopt_kargs = nlopt_kargs
        if ets not in ("conventional", "adam"):
            raise ValueError(f"Invalid ets: {ets!r}. Must be 'conventional' or 'adam'.")
        self.ets = ets

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def fit(self, y: NDArray, X: Optional[NDArray] = None) -> "OMG":
        self._start_time = time.time()

        # Build OM-style scaffolding for each side without doing the inner
        # estimation (we replace it with the joint nlopt run below). We
        # leverage OM internals so that `_check_parameters`, the
        # MNN-restoration, the binary-ot replacement, and the architector /
        # creator + om_initial_transform path are all consistent with stage 1.
        side_a = self._build_side(
            spec=self.model_a_spec,
            occurrence="odds-ratio",
            orders=self.orders_a,
            constant=self.constant_a,
            regressors=self.regressors_a,
            persistence=self.persistence_a,
            phi=self.phi_a,
            arma=self.arma_a,
            y=y,
            X=X,
        )
        side_b = self._build_side(
            spec=self.model_b_spec,
            occurrence="inverse-odds-ratio",
            orders=self.orders_b,
            constant=self.constant_b,
            regressors=self.regressors_b,
            persistence=self.persistence_b,
            phi=self.phi_b,
            arma=self.arma_b,
            y=y,
            X=X,
        )

        self._side_a = side_a
        self._side_b = side_b

        # Pure-regression early exit: both sides are ALM.
        if side_a.get("is_alm") and side_b.get("is_alm"):
            self._fit_alm_omg(side_a, side_b, y, X)
            self.time_elapsed_ = time.time() - self._start_time
            return self

        B_used, lb, ub, n_params_a = self._initial_B(side_a, side_b)

        if len(B_used) == 0:
            # Nothing to estimate — evaluate the cost once at the empty B
            cf_value = omg_cf(
                B=B_used,
                side_a=side_a,
                side_b=side_b,
                n_params_a=n_params_a,
                observations_dict=side_a["observations_dict"],
                bounds=self.bounds,
                adam_ets=(self.ets == "adam"),
            )
        else:
            cf_value = self._optimise(B_used, lb, ub, side_a, side_b, n_params_a)

        self._cf_value = float(cf_value)
        self._B_joint = np.array(B_used, dtype=float)
        self._n_params_a = n_params_a
        self._loglik = -self._cf_value

        # Information-criterion bookkeeping mirrors OM (must be set before
        # building sub-models — _om_from_side reads _log_lik_dict and
        # _ic_value).
        nobs = side_a["observations_dict"]["obs_in_sample"]
        df = len(B_used)
        self._log_lik_dict = {"value": self._loglik, "nobs": nobs, "df": df}
        self._ic_value = ic_function(self.ic, self._log_lik_dict)

        # Build the two OM sub-objects from the joint solution; they expose
        # the standard OM property surface for diagnostics.
        self.model_a = self._om_from_side(side_a, B_used[:n_params_a], "odds-ratio")
        self.model_b = self._om_from_side(
            side_b, B_used[n_params_a:], "inverse-odds-ratio"
        )

        # Combined probability via individual sub-model raw fitted values —
        # matches R's omgFinalFitA/B + omgLinkFunction approach.  Fall back to
        # the joint omfitGeneral result when individual re-fits produce NaN
        # (can happen for certain ARIMA configurations during backcasting).
        raw_a = np.asarray(self.model_a._prepared["y_fitted_raw"]).ravel()
        raw_b = np.asarray(self.model_b._prepared["y_fitted_raw"]).ravel()
        e_a = side_a["model_type_dict"]["error_type"]
        e_b = side_b["model_type_dict"]["error_type"]
        if np.any(np.isnan(raw_a)) or np.any(np.isnan(raw_b)):
            self._fitted_combined = self._joint_fitted(
                B_used, side_a, side_b, n_params_a
            )
        else:
            self._fitted_combined = omg_link_function(raw_a, raw_b, e_a, e_b)
        ot = np.asarray(side_a["observations_dict"]["ot"], dtype=np.float64)
        self._residuals_combined = ot - self._fitted_combined
        self._ot = ot
        self._observations_dict = side_a["observations_dict"]

        # Auto-forecast if h > 0
        if self.h and self.h > 0:
            self._auto_forecast = self._forecast_combined(self.h, X_future=None)
        else:
            self._auto_forecast = None

        self.time_elapsed_ = time.time() - self._start_time
        return self

    def predict(
        self,
        h: int,
        X: Optional[NDArray] = None,
        interval: Literal["none"] = "none",
        level: Optional[float] = 0.95,
        side: str = "both",
    ):
        if interval != "none":
            warnings.warn(
                "Intervals on the probability scale are not implemented for OMG; "
                "returning point forecasts only (matches R's forecast.omg)."
            )
        return self._forecast_combined(h, X_future=X)

    # ---------------------------------------------------------------------
    # Properties
    # ---------------------------------------------------------------------

    @property
    def fitted(self) -> NDArray:
        return self._fitted_combined

    @property
    def residuals(self) -> NDArray:
        return self._residuals_combined

    @property
    def actuals(self) -> NDArray:
        return self._ot.copy()

    @property
    def coef(self) -> NDArray:
        return self._B_joint

    @property
    def b_value(self) -> NDArray:
        return self.coef

    @property
    def loss_value(self) -> float:
        return self._cf_value

    @property
    def loglik(self) -> float:
        return self._loglik

    @property
    def occurrence(self) -> str:
        return "general"

    @property
    def distribution_(self) -> str:
        return "plogis"

    @property
    def loss_(self) -> str:
        return "likelihood"

    @property
    def scale(self) -> float:
        return float("nan")

    @property
    def sigma(self) -> float:
        return float("nan")

    @property
    def model_name(self) -> str:
        return f"oETS[G]({self.model_a.model_type})({self.model_b.model_type})"

    @property
    def model(self) -> str:
        return self.model_name

    @property
    def nobs(self) -> int:
        return int(self._observations_dict["obs_in_sample"])

    @property
    def nparam(self) -> int:
        return int(len(self._B_joint))

    @property
    def aic(self) -> float:
        from smooth.adam_general.core.utils.ic import AIC

        d = self._log_lik_dict
        return AIC(d["value"], d["nobs"], d["df"])

    @property
    def aicc(self) -> float:
        from smooth.adam_general.core.utils.ic import AICc

        d = self._log_lik_dict
        return AICc(d["value"], d["nobs"], d["df"])

    @property
    def bic(self) -> float:
        from smooth.adam_general.core.utils.ic import BIC

        d = self._log_lik_dict
        return BIC(d["value"], d["nobs"], d["df"])

    @property
    def bicc(self) -> float:
        from smooth.adam_general.core.utils.ic import BICc

        d = self._log_lik_dict
        return BICc(d["value"], d["nobs"], d["df"])

    @property
    def time_elapsed(self) -> float:
        return self.time_elapsed_

    @property
    def lags_used(self) -> List[int]:
        return list(self._side_a["lags_dict"]["lags"])

    @property
    def holdout_data(self) -> Optional[NDArray]:
        if not self.holdout:
            return None
        y = np.asarray(self._observations_dict.get("y_holdout"), dtype=float)
        return (y != 0).astype(float)

    def rstandard(self) -> NDArray:
        """Pearson standardised residuals for the general occurrence model.

        Formula: ``(ot - p) / sqrt(p*(1-p)) * sqrt(n/df)``
        where ``df = n - k``.  Mirrors R's ``rstandard.omg()``.
        """
        obs = self.nobs
        df = obs - self.nparam
        p = self.fitted
        e = self.actuals - p
        return e / np.sqrt(p * (1 - p)) * np.sqrt(obs / df)

    def rstudent(self) -> NDArray:
        """Pearson studentised residuals for the general occurrence model.

        Formula: ``(ot - p) / sqrt(p*(1-p)) * sqrt(n/df)``
        where ``df = n - k - 1``.  Mirrors R's ``rstudent.omg()``.
        """
        obs = self.nobs
        df = obs - self.nparam - 1
        p = self.fitted
        e = self.actuals - p
        return e / np.sqrt(p * (1 - p)) * np.sqrt(obs / df)

    # ---------------------------------------------------------------------
    # Internals — building the per-side scaffolding
    # ---------------------------------------------------------------------

    def _build_side(
        self,
        *,
        spec: str,
        occurrence: str,
        orders,
        constant,
        regressors,
        persistence,
        phi,
        arma,
        y,
        X,
    ) -> Dict[str, Any]:
        """Assemble all the per-side artefacts the joint cost needs.

        Reuses :class:`OM` internals (parameters_checker, restore-user-spec,
        architector, creator, om_initial_transform) so the per-side state is
        produced exactly the same way as a standalone ``OM(...)`` would.
        """
        scaffold = OM(
            model=spec,
            occurrence=occurrence,
            lags=self.lags,
            orders=orders,
            constant=constant,
            regressors=regressors,
            persistence=persistence,
            phi=phi,
            initial=self.initial,
            arma=arma,
            ic=self.ic,
            bounds=self.bounds,
            verbose=self.verbose,
            holdout=self.holdout,
            h=self.h,
            nlopt_kargs=self.nlopt_kargs,
            ets=self.ets,
        )
        scaffold._start_time = time.time()
        requested = (
            scaffold.model
            if isinstance(scaffold.model, str) and len(scaffold.model) in (3, 4)
            else None
        )
        scaffold._check_parameters(y, X)

        # Pure-regression early exit: return a minimal marker dict so OMG.fit()
        # can detect and handle both sides without running the ETS machinery.
        if getattr(scaffold, "_alm_model", None) is not None:
            return {
                "scaffold": scaffold,
                "is_alm": True,
                "occurrence_str": occurrence,
            }

        scaffold._restore_user_model_spec(requested)
        ot = np.asarray(scaffold._observations["ot"], dtype=np.float64)
        scaffold._observations["y_in_sample"] = ot
        scaffold._observations["obs_zero"] = int(
            np.sum(~scaffold._observations["ot_logical"])
        )

        adam_cpp, adam_created, profile_dict = scaffold._build_om_artifacts()
        ar_pm, ma_pm = _setup_arima_polynomials(
            scaffold._model_type, scaffold._arima, scaffold._lags_model
        )

        return {
            "scaffold": scaffold,
            "model_type_dict": scaffold._model_type,
            "components_dict": scaffold._components,
            "lags_dict": scaffold._lags_model,
            "matrices_dict": adam_created,
            "persistence": scaffold._persistence,
            "initials": scaffold._initials,
            "arima": scaffold._arima,
            "explanatory": scaffold._explanatory,
            "phi": scaffold._phi_internal,
            "constant": scaffold._constant,
            "observations_dict": scaffold._observations,
            "profile": profile_dict,
            "adam_cpp": adam_cpp,
            "ar_polynomial_matrix": ar_pm,
            "ma_polynomial_matrix": ma_pm,
            "occurrence_str": occurrence,
            "occurrence_char": scaffold._occurrence_char,
        }

    def _initial_B(self, side_a, side_b):  # noqa: N802
        b_a = self._initial_B_side(side_a)
        b_b = self._initial_B_side(side_b)
        n_params_a = len(b_a["B"])
        B_used = np.concatenate([b_a["B"], b_b["B"]])
        lb = np.concatenate([b_a["Bl"], b_b["Bl"]])
        ub = np.concatenate([b_a["Bu"], b_b["Bu"]])
        return B_used, lb, ub, n_params_a

    def _initial_B_side(self, side):  # noqa: N802
        return initialiser(
            model_type_dict=side["model_type_dict"],
            components_dict=side["components_dict"],
            lags_dict=side["lags_dict"],
            adam_created=side["matrices_dict"],
            persistence_checked=side["persistence"],
            initials_checked=side["initials"],
            arima_checked=side["arima"],
            constants_checked=side["constant"],
            explanatory_checked=side["explanatory"],
            observations_dict=side["observations_dict"],
            bounds=self.bounds,
            phi_dict=side["phi"],
            profile_dict=side["profile"],
            adam_cpp=side["adam_cpp"],
            other_parameter_estimate=False,
            other_value=2.0,
        )

    def _optimise(self, B_used, lb, ub, side_a, side_b, n_params_a):
        kargs = self.nlopt_kargs or {}
        algorithm = kargs.get("algorithm", "NLOPT_LN_NELDERMEAD")
        xtol_rel = kargs.get("xtol_rel", 1e-6)
        xtol_abs = kargs.get("xtol_abs", 1e-8)
        ftol_rel = kargs.get("ftol_rel", 1e-8)
        ftol_abs = kargs.get("ftol_abs", 0)
        maxeval = kargs.get("maxeval", len(B_used) * 40)
        nlopt_algorithm = getattr(
            nlopt, algorithm.replace("NLOPT_", ""), nlopt.LN_NELDERMEAD
        )

        _adam_ets = self.ets == "adam"

        def _objective(x, _grad):
            try:
                cf = omg_cf(
                    B=x,
                    side_a=side_a,
                    side_b=side_b,
                    n_params_a=n_params_a,
                    observations_dict=side_a["observations_dict"],
                    bounds=self.bounds,
                    adam_ets=_adam_ets,
                )
            except Exception:
                cf = 1e100
            return float(cf) if np.isfinite(cf) else 1e300

        opt = nlopt.opt(nlopt_algorithm, len(B_used))
        opt = _configure_optimizer(
            opt,
            lb,
            ub,
            maxeval,
            None,
            xtol_rel=xtol_rel,
            xtol_abs=xtol_abs,
            ftol_rel=ftol_rel,
            ftol_abs=ftol_abs,
        )
        opt.set_min_objective(_objective)
        try:
            B_used[:] = opt.optimize(B_used)
        except Exception:
            pass
        return opt.last_optimum_value()

    # ---------------------------------------------------------------------
    # Build per-sub-model OM objects from the joint estimate
    # ---------------------------------------------------------------------

    def _joint_fitted(self, B_used, side_a, side_b, n_params_a):
        """Run omfitGeneral with the final optimised B and return p_combined.

        Uses the same joint C++ path as the cost function, which is numerically
        stable even when individual sub-model re-fits (om_preparator) would
        diverge (e.g. certain ARIMA configurations during backcasting).
        """

        def _f(x, dtype=np.float64):
            return np.asfortranarray(x, dtype=dtype)

        B_A = B_used[:n_params_a]
        B_B = B_used[n_params_a:]

        from smooth.adam_general.core.creator import filler

        elem_a = filler(
            B_A,
            model_type_dict=side_a["model_type_dict"],
            components_dict=side_a["components_dict"],
            lags_dict=side_a["lags_dict"],
            matrices_dict=side_a["matrices_dict"],
            persistence_checked=side_a["persistence"],
            initials_checked=side_a["initials"],
            arima_checked=side_a["arima"],
            explanatory_checked=side_a["explanatory"],
            phi_dict=side_a["phi"],
            constants_checked=side_a["constant"],
            adam_cpp=side_a["adam_cpp"],
        )
        elem_b = filler(
            B_B,
            model_type_dict=side_b["model_type_dict"],
            components_dict=side_b["components_dict"],
            lags_dict=side_b["lags_dict"],
            matrices_dict=side_b["matrices_dict"],
            persistence_checked=side_b["persistence"],
            initials_checked=side_b["initials"],
            arima_checked=side_b["arima"],
            explanatory_checked=side_b["explanatory"],
            phi_dict=side_b["phi"],
            constants_checked=side_b["constant"],
            adam_cpp=side_b["adam_cpp"],
        )
        side_a["profile"]["profiles_recent_table"][:] = elem_a["mat_vt"][
            :, : side_a["lags_dict"]["lags_model_max"]
        ]
        side_b["profile"]["profiles_recent_table"][:] = elem_b["mat_vt"][
            :, : side_b["lags_dict"]["lags_model_max"]
        ]

        initials_a = side_a["initials"]
        if isinstance(initials_a["initial_type"], list):
            backcast = any(
                t in ("complete", "backcasting") for t in initials_a["initial_type"]
            )
        else:
            backcast = initials_a["initial_type"] in ("complete", "backcasting")

        ot = np.asarray(side_a["observations_dict"]["ot"], dtype=np.float64)
        res = side_a["adam_cpp"].omfitGeneral(
            matrixVtA=_f(elem_a["mat_vt"]),
            matrixWtA=_f(elem_a["mat_wt"]),
            matrixFA=_f(elem_a["mat_f"]),
            vectorGA=_f(elem_a["vec_g"]),
            indexLookupTableA=_f(side_a["profile"]["index_lookup_table"], np.uint64),
            profilesRecentA=_f(side_a["profile"]["profiles_recent_table"]),
            EB=side_b["model_type_dict"]["error_type"],
            TB=side_b["model_type_dict"]["trend_type"],
            SB=side_b["model_type_dict"]["season_type"],
            nNonSeasonalB=int(
                side_b["components_dict"]["components_number_ets_non_seasonal"]
            ),
            nSeasonalB=int(side_b["components_dict"]["components_number_ets_seasonal"]),
            nETSB=int(side_b["components_dict"]["components_number_ets"]),
            nArimaB=int(side_b["components_dict"].get("components_number_arima", 0)),
            nXregB=int(side_b["explanatory"].get("xreg_number", 0)),
            nComponentsB=int(side_b["components_dict"]["components_number_all"]),
            constantB=bool(side_b["constant"].get("constant_required", False)),
            adamETSB=False,
            matrixVtB=_f(elem_b["mat_vt"]),
            matrixWtB=_f(elem_b["mat_wt"]),
            matrixFB=_f(elem_b["mat_f"]),
            vectorGB=_f(elem_b["vec_g"]),
            indexLookupTableB=_f(side_b["profile"]["index_lookup_table"], np.uint64),
            profilesRecentB=_f(side_b["profile"]["profiles_recent_table"]),
            vectorOt=ot,
            backcast=backcast,
            nIterations=int(initials_a["n_iterations"]),
            refineHead=True,
        )
        e_a = side_a["model_type_dict"]["error_type"]
        e_b = side_b["model_type_dict"]["error_type"]
        return omg_link_function(
            np.asarray(res.fittedA).ravel(),
            np.asarray(res.fittedB).ravel(),
            e_a,
            e_b,
        )

    def _om_from_side(self, side, B, occurrence_str) -> OM:
        scaffold: OM = side["scaffold"]
        # Inject the joint estimate into the scaffold so its post-fit
        # plumbing (om_preparator, model_name, etc.) reflects the joint
        # solution rather than re-running its own optimiser.
        scaffold._adam_estimated = {
            "B": np.asarray(B, dtype=float),
            "CF_value": self._cf_value,
            "n_param_estimated": int(len(B)),
            "log_lik_adam_value": dict(self._log_lik_dict),
            "arima_polynomials": side["matrices_dict"].get("arima_polynomials"),
            "adam_cpp": side["adam_cpp"],
        }
        scaffold._adam_cpp = side["adam_cpp"]
        scaffold._profile = side["profile"]
        scaffold._ic_selection = self._ic_value
        scaffold._select_distribution()

        # Build fresh matrices with the ORIGINAL error/trend/season types —
        # mirrors R's omgFinalFitA/B which calls adam_creator with checkerA/B's
        # actual Etype (not the forced-additive used during optimization).
        # The difference matters for seasonal initial states that seed backcasting.
        adam_created_final = scaffold._build_final_fit_adam_created(side["profile"])
        scaffold._adam_created = adam_created_final

        scaffold._prepared = om_preparator(
            model_type_dict=scaffold._model_type,
            components_dict=scaffold._components,
            lags_dict=scaffold._lags_model,
            matrices_dict=scaffold._adam_created,
            persistence_checked=scaffold._persistence,
            initials_checked=scaffold._initials,
            arima_checked=scaffold._arima,
            explanatory_checked=scaffold._explanatory,
            phi_dict=scaffold._phi_internal,
            constants_checked=scaffold._constant,
            observations_dict=scaffold._observations,
            profiles_dict=scaffold._profile,
            adam_estimated=scaffold._adam_estimated,
            adam_cpp=scaffold._adam_cpp,
            occurrence=occurrence_str,
            occurrence_char=scaffold._occurrence_char,
        )
        scaffold._set_om_fitted_attributes()
        return scaffold

    # ---------------------------------------------------------------------
    # Pure-regression path (both sub-models are greybox.ALM)
    # ---------------------------------------------------------------------

    def _fit_alm_omg(self, side_a, side_b, y, X):
        """Handle OMG when both sub-models reduce to logistic regression.

        Side A is fitted on occurrence (ot); side B is refitted on non-occurrence
        (1-ot) — the inverse-odds-ratio complement.  The combined probability
        p = pA / (pA + pB) simplifies to the side-A probability because
        symmetry of logistic regression guarantees pA + pB = 1.
        """
        from greybox import ALM

        scaffold_a = side_a["scaffold"]
        alm_a = scaffold_a._alm_model

        y_inv = 1.0 - np.asarray(alm_a._y_train_, dtype=float)
        alm_b = ALM(distribution=alm_a.distribution)
        alm_b.fit(np.asarray(alm_a._X_train_), y_inv)

        scaffold_b = side_b["scaffold"]
        scaffold_b._alm_model = alm_b

        scaffold_a._populate_from_alm(y, X)
        scaffold_b._populate_from_alm(y, X)

        self.model_a = scaffold_a
        self.model_b = scaffold_b

        pA = np.asarray(alm_a.fitted_values_, dtype=float)
        pB = np.asarray(alm_b.fitted_values_, dtype=float)
        self._fitted_combined = pA / (pA + pB)

        ot = np.asarray(alm_a._y_train_, dtype=float)
        self._residuals_combined = ot - self._fitted_combined
        self._ot = ot
        self._observations_dict = scaffold_a._observations

        nobs = int(alm_a.nobs)
        df = int(alm_a.nparam) + int(alm_b.nparam)
        loglik = float(alm_a.loglik) + float(alm_b.loglik)
        self._loglik = loglik
        self._cf_value = -loglik
        self._log_lik_dict = {"value": loglik, "nobs": nobs, "df": df}
        self._ic_value = None
        self._B_joint = np.concatenate(
            [np.asarray(alm_a.coefficients), np.asarray(alm_b.coefficients)]
        )
        self._n_params_a = int(alm_a.nparam)
        self._auto_forecast = None

    # ---------------------------------------------------------------------
    # Combined forecast
    # ---------------------------------------------------------------------

    def _forecast_combined(self, h: int, X_future=None):
        # Combine raw (pre-link) forecasts from both sub-models via omg_link_function,
        # mirroring R's omg() outer block: adamCppA$forecast + adamCppB$forecast → link.
        fc_a, fc_a_raw = self.model_a._run_forecaster(h, X_future=X_future)
        fc_b_raw = self.model_b._raw_forecast_direct(h, X_future=X_future)
        e_a = self._side_a["model_type_dict"]["error_type"]
        e_b = self._side_b["model_type_dict"]["error_type"]
        p_combined = omg_link_function(fc_a_raw, fc_b_raw, e_a, e_b)
        p_combined = np.where(np.isnan(p_combined), 1.0, p_combined)
        fc_a.mean[:] = p_combined
        return fc_a


def _build_omg_from_om_kwargs(**om_kwargs) -> OMG:
    """Construct an OMG from the kwargs supplied to OM(occurrence='general', ...).

    Mirrors R's om(occurrence='general'): forwards model/lags/orders as both
    modelA and modelB and delegates to omg().
    """
    model = om_kwargs.pop("model", "MNN")
    return OMG(
        model_a=model,
        model_b=model,
        lags=om_kwargs.pop("lags", None),
        orders_a=om_kwargs.pop("orders", None),
        orders_b=None,
        constant_a=om_kwargs.pop("constant", False),
        formula_a=om_kwargs.pop("formula", None),
        regressors_a=om_kwargs.pop("regressors", "use"),
        persistence_a=om_kwargs.pop("persistence", None),
        phi_a=om_kwargs.pop("phi", None),
        arma_a=om_kwargs.pop("arma", None),
        h=om_kwargs.pop("h", 0),
        holdout=om_kwargs.pop("holdout", False),
        initial=om_kwargs.pop("initial", "backcasting"),
        ic=om_kwargs.pop("ic", "AICc"),
        bounds=om_kwargs.pop("bounds", "usual"),
        verbose=om_kwargs.pop("verbose", 0),
        nlopt_kargs=om_kwargs.pop("nlopt_kargs", None),
    )
