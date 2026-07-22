"""General Occurrence Model (OMG).

Fits two parallel ETS / ARIMA occurrence sub-models — A (odds-ratio) and
B (inverse-odds-ratio) — jointly, against a shared Bernoulli log-likelihood
on the combined probability ``p = aFit / (aFit + bFit)``.

The single C++ entry point ``adamCore.omfitGeneral`` advances both
sub-models simultaneously inside the optimiser.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import nlopt
import numpy as np
import pandas as pd
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
from smooth.adam_general.core.utils.utils import SMOOTHER_DEFAULT, SmootherType


def _omg_refit_one_replicate(
    y_raw: NDArray,
    idx_matrix: Union[NDArray, list[NDArray]],
    clone_kwargs: Dict[str, Any],
    k: int,
    i: int,
) -> Optional[NDArray]:
    """Refit one OMG bootstrap replicate; return the joint coef vector or ``None``.

    Module-level so it can be pickled by ``joblib.Parallel`` workers. ``i``
    is the last positional argument so callers can bind everything else
    with :func:`functools.partial`. The OMG class is resolved locally to
    avoid pickling the class object itself.
    """
    y_boot = y_raw[idx_matrix[i]]
    # Pathological draw (all zeros or all ones) → OMG cannot fit; skip.
    if np.all(y_boot == 0) or np.all(y_boot != 0):
        return None
    try:
        boot_model = OMG(**clone_kwargs).fit(y_boot)
        boot_coef = np.asarray(boot_model.coef, dtype=float)
    except Exception:
        return None
    if boot_coef.shape[0] != k or not np.all(np.isfinite(boot_coef)):
        return None
    return boot_coef


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
        loss: Union[
            Literal["likelihood", "MSE", "MAE", "HAM", "LASSO", "RIDGE"],
            Callable,
        ] = "likelihood",
        reg_lambda: Optional[float] = None,
        ic: Literal["AIC", "AICc", "BIC", "BICc"] = "AICc",
        bounds: Literal["usual", "admissible", "none"] = "usual",
        verbose: int = 0,
        nlopt_kwargs: Optional[Dict[str, Any]] = None,
        ets: Literal["conventional", "adam"] = "conventional",
        smoother: SmootherType = SMOOTHER_DEFAULT,
    ) -> None:
        # Accept callable for user-defined custom loss (same API as ADAM,
        # mirrors R/adamGeneral.R:574-602): signature
        # ``(actual, fitted, B) -> scalar``.
        loss_function: Optional[Callable] = None
        if callable(loss):
            loss_function = loss
            loss = "custom"  # type: ignore[assignment]
        elif loss not in ("likelihood", "MSE", "MAE", "HAM", "LASSO", "RIDGE"):
            raise ValueError(
                f"Invalid loss={loss!r}; expected one of "
                "'likelihood', 'MSE', 'MAE', 'HAM', 'LASSO', 'RIDGE', "
                "or a callable returning a scalar."
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
        self.loss_function = loss_function
        self.reg_lambda = reg_lambda
        self.ic = ic
        self.bounds = bounds
        self.verbose = verbose
        self.nlopt_kwargs = nlopt_kwargs
        if ets not in ("conventional", "adam"):
            raise ValueError(f"Invalid ets: {ets!r}. Must be 'conventional' or 'adam'.")
        self.ets = ets
        if smoother not in ("default", "ma", "lowess", "supsmu", "global"):
            raise ValueError(
                f"Invalid smoother: {smoother!r}. Must be one of "
                "'default', 'ma', 'lowess', 'supsmu', 'global'."
            )
        # The scaffold OM built in _build_side resolves "default" per side using
        # the shared initialisation, so both sub-models get the same smoother.
        self.smoother = smoother

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def fit(self, y: NDArray, X: Optional[NDArray] = None) -> "OMG":
        self._start_time = time.time()
        self._fi_cache = None

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
                loss=self.loss,  # type: ignore[arg-type]
                loss_function=self.loss_function,
                reg_lambda=self.reg_lambda,
            )
        else:
            cf_value = self._optimise(B_used, lb, ub, side_a, side_b, n_params_a)

        self._cf_value = float(cf_value)
        self._B_joint = np.array(B_used, dtype=float)
        self._n_params_a = n_params_a

        # Coupled fitted probability, from the same recursion the optimiser
        # minimised (re-run omg_cf at the final B). Its Bernoulli is the logLik,
        # so fitted and logLik stay mutually consistent for every loss — the
        # standalone per-side refits (model_a/model_b below) only approximate it
        # and are kept for sub-model diagnostics. Mirrors R's omgCF_local.
        ot = np.asarray(side_a["observations_dict"]["ot"], dtype=np.float64)
        self._fitted_combined = np.asarray(
            omg_cf(
                B_used,
                side_a=side_a,
                side_b=side_b,
                n_params_a=n_params_a,
                observations_dict=side_a["observations_dict"],
                bounds=self.bounds,
                adam_ets=(self.ets == "adam"),
                loss=self.loss,  # type: ignore[arg-type]
                loss_function=self.loss_function,
                reg_lambda=self.reg_lambda,
                return_fitted=True,
            ),
            dtype=np.float64,
        ).ravel()
        self._residuals_combined = ot - self._fitted_combined
        self._ot = ot
        self._loglik = float(
            np.sum(
                ot * np.log(self._fitted_combined)
                + (1.0 - ot) * np.log(1.0 - self._fitted_combined)
            )
        )

        # Information-criterion bookkeeping mirrors OM (must be set before
        # building sub-models — _om_from_side reads _log_lik_dict and
        # _ic_value).
        nobs = side_a["observations_dict"]["obs_in_sample"]
        # Both sides' initials count towards df like a single OM, so the
        # OM-vs-OMG selection is fair (mirrors the om initial-df fix).
        df = (
            len(B_used)
            + side_a["scaffold"]._om_initials_df()
            + side_b["scaffold"]._om_initials_df()
        )
        self._log_lik_dict = {"value": self._loglik, "nobs": nobs, "df": df}
        self._ic_value = ic_function(self.ic, self._log_lik_dict)

        # Build the two OM sub-objects from the joint solution; they expose
        # the standard OM property surface for diagnostics.
        self.model_a = self._om_from_side(side_a, B_used[:n_params_a], "odds-ratio")
        self.model_b = self._om_from_side(
            side_b, B_used[n_params_a:], "inverse-odds-ratio"
        )
        # Preserve the raw input y for ``actuals`` — equivalent to R's
        # storing the original series on the omg object so that ``actuals(omg)``
        # returns the same type and content as ``actuals(om)`` would.
        self._y_raw = np.asarray(y, dtype=float)
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
                "returning point forecasts only."
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
        """Binary 0/1 indicator built from the raw input series.

        Mirrors R's ``actuals.omg``: returns ``(y != 0) * 1`` using the
        original ``y`` (with its original dtype/shape preserved), not the
        ``ot`` vector held in ``_observations_dict``.
        """
        y = getattr(self, "_y_raw", None)
        if y is None:
            return self._ot.copy()
        return (np.asarray(y, dtype=float) != 0).astype(float)

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

    def point_lik(self, log: bool = True) -> NDArray:
        """Per-observation log-likelihood of the coupled occurrence model.

        Bernoulli contribution of each observation against the coupled fitted
        probability: ``log(p_t)`` when ``o_t = 1`` and ``log(1 - p_t)``
        otherwise. Mirrors R's ``pointLik.om`` dispatched on an ``omg`` object
        (R/methods.R:598), so ``sum(point_lik())`` equals :attr:`loglik`.
        """
        ot = np.asarray(self.actuals, dtype=float).ravel()
        p = np.asarray(self.fitted, dtype=float).ravel()
        ot_logical = ot == 1
        lik_values = np.empty(len(ot), dtype=float)
        lik_values[ot_logical] = np.log(p[ot_logical])
        lik_values[~ot_logical] = np.log(1.0 - p[~ot_logical])
        if not log:
            lik_values = np.exp(lik_values)
        return lik_values

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
        where ``df = n - k``.
        """
        obs = self.nobs
        df = obs - self.nparam
        p = self.fitted
        e = self.actuals - p
        return e / np.sqrt(p * (1 - p)) * np.sqrt(obs / df)

    def rstudent(self) -> NDArray:
        """Pearson studentised residuals for the general occurrence model.

        Formula: ``(ot - p) / sqrt(p*(1-p)) * sqrt(n/df)``
        where ``df = n - k - 1``.
        """
        obs = self.nobs
        df = obs - self.nparam - 1
        p = self.fitted
        e = self.actuals - p
        return e / np.sqrt(p * (1 - p)) * np.sqrt(obs / df)

    # ---------------------------------------------------------------------
    # Inference — vcov / confint / summary (R: vcov.omg / confint.omg /
    # summary.omg). Both sub-models share a single joint Fisher Information
    # matrix; rows are prefixed ``A:`` / ``B:``.
    # ---------------------------------------------------------------------

    @property
    def coef_names(self) -> List[str]:
        """Joint parameter names with ``A:`` / ``B:`` prefixes.

        Mirrors R's ``names(coef(omg))``. Falls back to ``b1, b2, …`` if a
        sub-model's own names are unavailable.
        """
        if not hasattr(self, "model_a") or not hasattr(self, "model_b"):
            return [f"b{i + 1}" for i in range(int(len(self._B_joint)))]
        names_a = list(self.model_a.coef_names)
        names_b = list(self.model_b.coef_names)
        return [f"A:{n}" for n in names_a] + [f"B:{n}" for n in names_b]

    def _fisher_information_matrix(self, step_size=None):
        """Observed FI of the joint OMG cost at the estimated ``_B_joint``.

        ``omg_cf`` returns the negative log-likelihood, so its Hessian *is* the
        observed Fisher Information — no sign flip.

        Bounds are disabled (``bounds="none"``) during the Hessian call to
        match R's ``vcov.adam`` / ``vcov.omg`` (R/adam.R:2797 —
        ``boundsFI <- "none"``). With the user's usual/admissible bounds left
        on, FD perturbations ``B ± h`` near a boundary parameter (e.g.
        ``alpha=0``) would cross into the penalty region and return 1e300,
        making the diagonal FI explode and the inverse covariance collapse
        to ~0. Disabling bounds lets the underlying C++ propagation evaluate
        the actual log-likelihood at the perturbed B, even when the
        smoothing parameter is briefly out of range.
        """
        from smooth.adam_general.core.utils.var_covar import numerical_hessian

        side_a = self._side_a
        side_b = self._side_b
        n_params_a = int(self._n_params_a)
        observations = side_a["observations_dict"]
        adam_ets = self.ets == "adam"

        def _cost(b):
            return omg_cf(
                B=b,
                side_a=side_a,
                side_b=side_b,
                n_params_a=n_params_a,
                observations_dict=observations,
                bounds="none",
                adam_ets=adam_ets,
                loss=self.loss,  # type: ignore[arg-type]
                loss_function=self.loss_function,
                reg_lambda=self.reg_lambda,
            )

        return numerical_hessian(_cost, self._B_joint, step_size=step_size)

    @property
    def fisher_information_(self) -> Optional[NDArray]:
        """Cached observed Fisher Information at the estimated joint B.

        Computed lazily on first access via :meth:`_fisher_information_matrix`;
        cleared whenever :meth:`fit` is called again.
        """
        cached = getattr(self, "_fi_cache", None)
        if cached is not None:
            return cached
        self._fi_cache = self._fisher_information_matrix()
        return self._fi_cache

    def _opg_covariance(self, step_size=None):
        """OPG / BHHH covariance of the joint OMG coefficients, or None.

        The per-observation score is that of the coupled Bernoulli likelihood:
        ``omg_cf(b, return_fitted=True)`` gives the coupled fitted probability at
        a perturbed ``b``, whose Bernoulli log-density is differenced. Mirrors
        R's ``covarOPGomg``.
        """
        from smooth.adam_general.core.utils.var_covar import covar_opg

        side_a = self._side_a
        side_b = self._side_b
        n_params_a = int(self._n_params_a)
        observations = side_a["observations_dict"]
        adam_ets = self.ets == "adam"
        ot = np.asarray(self.actuals, dtype=float).ravel()

        def point_lik_at(b):
            p = omg_cf(
                B=np.asarray(b, dtype=float),
                side_a=side_a,
                side_b=side_b,
                n_params_a=n_params_a,
                observations_dict=observations,
                bounds="none",
                adam_ets=adam_ets,
                loss=self.loss,  # type: ignore[arg-type]
                loss_function=self.loss_function,
                reg_lambda=self.reg_lambda,
                return_fitted=True,
            )
            p = np.asarray(p, dtype=float).ravel()
            if (
                p.shape[0] != ot.shape[0]
                or not np.all(np.isfinite(p))
                or np.any(p <= 0.0)
                or np.any(p >= 1.0)
            ):
                return None
            return np.where(ot == 1, np.log(p), np.log(1.0 - p))

        return covar_opg(
            self._B_joint, point_lik_at, ot.shape[0], float(self.loglik), step_size
        )

    def vcov(self, type=None, bootstrap: bool = False, step_size=None, **boot_kwargs):  # noqa: A002
        """Joint variance–covariance matrix for both OMG sub-models.

        Mirrors R's ``vcov.omg``. ``type`` selects the estimator: ``"opg"`` (the
        default, the OPG/BHHH covariance of the coupled Bernoulli score, PSD by
        construction), ``"hessian"`` (the observed Fisher Information) or
        ``"bootstrap"``. Rows/cols are prefixed ``A:`` / ``B:``.

        Parameters
        ----------
        type : {"opg", "hessian", "bootstrap"}, optional
            Covariance estimator; defaults to ``"opg"``.
        bootstrap : bool, default=False
            Deprecated alias for ``type="bootstrap"`` (warns).
        step_size : float, optional
            Finite-difference step.
        **boot_kwargs
            Forwarded to :meth:`coefbootstrap` for the bootstrap type.

        Returns
        -------
        pandas.DataFrame
            Joint covariance matrix with prefixed row/col names.
        """
        import warnings

        import pandas as pd

        from smooth.adam_general.core.utils.var_covar import (
            invert_fisher_information,
            resolve_covar_type,
        )

        cov_type = resolve_covar_type(type, bootstrap)
        names = self.coef_names

        if cov_type == "bootstrap":
            return self.coefbootstrap(**boot_kwargs).vcov

        if cov_type == "opg":
            cov_opg = self._opg_covariance(step_size=step_size)
            if cov_opg is not None:
                return pd.DataFrame(cov_opg, index=names, columns=names)
            warnings.warn(
                "The OPG covariance could not be computed for this omg model; "
                "falling back to the observed Fisher Information.",
                RuntimeWarning,
                stacklevel=2,
            )

        FI = np.asarray(self._fisher_information_matrix(step_size=step_size))  # noqa: N806
        broken = np.all(FI == 0, axis=1) | np.any(np.isnan(FI), axis=1)
        if np.any(broken) and step_size is None:
            FI = np.asarray(  # noqa: N806
                self._fisher_information_matrix(
                    step_size=float(np.finfo(float).eps ** (1 / 6))
                )
            )

        cov = invert_fisher_information(FI)
        names = self.coef_names
        return pd.DataFrame(cov, index=names, columns=names)

    def confint(
        self,
        parm=None,
        level: float = 0.95,
        type=None,  # noqa: A002
        bootstrap: bool = False,
        step_size=None,
        **boot_kwargs,
    ):
        """Confidence intervals for the joint OMG parameters.

        Mirrors R's ``confint.omg``: SEs from the joint :meth:`vcov`, t-quantile
        half-widths with R's asymmetric degrees of freedom, then per-sub-model
        clamping to each side's admissible region (via the sub-model's own
        :meth:`~ADAM._clamp_confint_offsets`). Rows are prefixed ``A:`` /
        ``B:`` to identify which sub-model each parameter belongs to.

        Parameters
        ----------
        parm : str or sequence of str, optional
            Subset of (prefixed) names to return.
        level : float, default=0.95
            Confidence level for the interval.
        bootstrap : bool, default=False
            Not implemented for OMG yet.
        step_size : float, optional
            Finite-difference step forwarded to :meth:`vcov`.

        Returns
        -------
        pandas.DataFrame
            Columns ``["S.E.", "<lo>%", "<hi>%"]`` indexed by prefixed names.
        """
        import pandas as pd
        from scipy import stats as scipy_stats

        from smooth.adam_general.core.utils.var_covar import resolve_covar_type

        cov_type = resolve_covar_type(type, bootstrap)
        names = self.coef_names
        params = np.asarray(self._B_joint, dtype=float)

        if cov_type == "bootstrap":
            from smooth.adam_general.core.utils.bootstrap import (
                bootstrap_confint_frame,
            )

            boot = self.coefbootstrap(**boot_kwargs)
            return bootstrap_confint_frame(boot, names, params, level, parm)
        n_a = int(self._n_params_a)

        V = self.vcov(type=cov_type, step_size=step_size).to_numpy()  # noqa: N806
        se = np.sqrt(np.abs(np.diag(V)))

        nobs = int(self.nobs)
        nparam = int(self.nparam)
        # R: asymmetric df for the two tails (omg.R:1611–1612).
        lo = scipy_stats.t.ppf((1 - level) / 2, df=nobs - nparam) * se
        hi = scipy_stats.t.ppf((1 + level) / 2, df=nobs + nparam) * se

        # Per-sub-model clamping using each sub-model's own bounds knowledge.
        # Slicing returns views; in-place mutations propagate back.
        params_a = np.asarray(self.model_a.coef, dtype=float)
        params_b = np.asarray(self.model_b.coef, dtype=float)
        self.model_a._clamp_confint_offsets(
            self.model_a.coef_names, params_a, lo[:n_a], hi[:n_a]
        )
        self.model_b._clamp_confint_offsets(
            self.model_b.coef_names, params_b, lo[n_a:], hi[n_a:]
        )

        lo = lo + params
        hi = hi + params

        lo_name = f"{(1 - level) / 2 * 100:g}%"
        hi_name = f"{(1 + level) / 2 * 100:g}%"
        out = pd.DataFrame(
            np.column_stack([se, lo, hi]),
            index=names,
            columns=["S.E.", lo_name, hi_name],
        )
        if parm is not None:
            out = out.loc[parm if isinstance(parm, (list, tuple)) else [parm]]
        return out

    def summary(self, level: float = 0.95, digits: int = 4, type=None):  # noqa: A002
        """Coefficient-table summary for the joint occurrence model.

        Mirrors R's ``summary.omg``: two coefficient tables (one per
        sub-model), each with significance indicators based on whether zero
        falls inside the CI, followed by a shared footer (sample size, total
        estimated parameters, loss value, AICc/BICc). ``type`` selects the
        covariance estimator for the standard errors (default ``"opg"``).
        """
        from smooth.adam_general.core.utils.printing import OMGSummary

        return OMGSummary(self, level=level, digits=digits, type=type)

    def _bootstrap_clone_kwargs(self) -> Dict[str, Any]:
        """Snapshot the init kwargs needed to instantiate a fresh OMG for refit.

        OMG stores its init args directly as instance attributes (no
        ``_config`` dict), so we collect them explicitly. Kept private so the
        bootstrap doesn't depend on the precise attribute names of OMG's
        :meth:`__init__`.
        """
        return dict(
            model_a=self.model_a_spec,
            model_b=self.model_b_spec,
            lags=self.lags,
            orders_a=self.orders_a,
            orders_b=self.orders_b,
            constant_a=self.constant_a,
            constant_b=self.constant_b,
            formula_a=self.formula_a,
            formula_b=self.formula_b,
            regressors_a=self.regressors_a,
            regressors_b=self.regressors_b,
            persistence_a=self.persistence_a,
            persistence_b=self.persistence_b,
            phi_a=self.phi_a,
            phi_b=self.phi_b,
            arma_a=self.arma_a,
            arma_b=self.arma_b,
            h=0,
            holdout=False,
            initial=self.initial,
            # Carry the callable through so bootstrap refits replay the
            # same custom loss; otherwise the resolved string flag.
            loss=self.loss_function if self.loss_function is not None else self.loss,
            reg_lambda=self.reg_lambda,
            ic=self.ic,
            bounds=self.bounds,
            verbose=0,
            nlopt_kwargs=self.nlopt_kwargs,
            ets=self.ets,
        )

    def coefbootstrap(
        self,
        nsim: int = 1000,
        size: Optional[int] = None,
        replace: bool = False,
        prob: Optional[NDArray] = None,
        parallel: Union[bool, int] = False,
        method: str = "cr",
        seed: Optional[int] = None,
        verbose: bool = False,
    ):
        """Bootstrap the joint OMG coefficient sampling distribution.

        Mirrors R's ``coefbootstrap.omg`` (R/omg.R:1401-1600): case-resamples
        the raw input series, refits the joint OMG on each replicate, and
        stacks the prefixed ``A:`` / ``B:`` coefficient vectors.

        ``parallel=True`` (or ``parallel=<int>``) runs replicates in
        parallel via ``joblib.Parallel`` (``cpu_count - 1`` workers by
        default, or the supplied integer). Requires the optional
        ``joblib`` dependency; if unavailable, a warning is emitted and
        the call falls back to serial. See
        :meth:`smooth.adam_general.core.adam.ADAM.coefbootstrap` for the
        full parameter documentation; the OMG version takes the same
        signature and returns the same
        :class:`~smooth.adam_general.core.utils.bootstrap.BootstrapResult`.
        """
        import time as _time
        from functools import partial

        from smooth.adam_general.core.utils.bootstrap import (
            _build_result,
            case_resample_indices,
            run_replicates,
            time_series_sample_indices,
        )

        if method not in ("cr", "dsr"):
            raise ValueError(f"method must be 'cr' or 'dsr', got {method!r}.")
        if method == "dsr":
            raise NotImplementedError(
                "method='dsr' requires greybox.dsrboot which is not yet "
                "available in Python; use method='cr'."
            )
        any_regressors = self.regressors_a or self.regressors_b
        if any_regressors and (self.formula_a or self.formula_b):
            raise NotImplementedError(
                "coefbootstrap does not yet support OMG models with external "
                "regressors. File an issue if you need this."
            )

        y_raw = np.asarray(self._y_raw, dtype=float)
        nobs = int(y_raw.size)
        if size is None:
            size = max(int(np.floor(0.75 * nobs)), 1)
        size = int(size)
        if size < 2:
            raise ValueError(f"size={size} too small; need at least 2 observations.")

        rng = np.random.default_rng(seed)
        original_coef_names = list(self.coef_names)
        k = len(original_coef_names)
        clone_kwargs = self._bootstrap_clone_kwargs()

        # Match R's coefbootstrap.omg sampler (omg.R:1496-1508): variable-
        # length contiguous-window resampling. obs_minimum uses max(lags)
        # across both sub-models — for OMG we use ``self.lags`` if set
        # else 1.
        lags = self.lags if self.lags else [1]
        max_lag = int(np.max(lags)) if len(lags) else 1
        obs_minimum = max(max_lag, k) + 2
        if obs_minimum >= nobs:
            raise ValueError(
                f"Not enough observations to do case-resampling bootstrap "
                f"(obs_minimum={obs_minimum}, nobs={nobs})."
            )
        initial_is_window = isinstance(self.initial, str) and self.initial in (
            "backcasting",
            "complete",
        )
        change_origin = initial_is_window
        if replace or prob is not None:
            idx_list = list(case_resample_indices(nobs, size, nsim, replace, prob, rng))
        else:
            idx_list = time_series_sample_indices(
                nobs, nsim, obs_minimum, change_origin, rng
            )

        worker = partial(_omg_refit_one_replicate, y_raw, idx_list, clone_kwargs, k)

        t0 = _time.time()
        replicate_coefs, parallel_used = run_replicates(
            worker,
            nsim=nsim,
            parallel=parallel,
            verbose=verbose,
            label="coefbootstrap[omg]",
        )
        elapsed = _time.time() - t0

        return _build_result(
            replicate_coefs,
            original_coef_names,
            method=method,
            nsim=nsim,
            size=size,
            replace=replace,
            prob=prob,
            parallel=parallel_used,
            model="omg",
            time_elapsed=elapsed,
        )

    def multicov(
        self,
        type: str = "analytical",
        h: int = 10,
        nsim: int = 1000,
    ):
        """Multi-step forecast-error covariance — not defined for OMG.

        R's ``multicov`` is dispatched on ``adam`` / ``smooth`` classes via
        the state-space matrices ``(F, W, g, σ²)``. OMG has **two** parallel
        sub-models joined by the non-linear link
        ``p = pA / (pA + pB)``; the resulting joint multi-step distribution
        has no closed-form covariance from those matrices and R does not
        define a ``multicov.omg`` method either (it would dispatch to
        ``multicov.smooth`` and crash on the missing ``$persistence``).

        Call :meth:`smooth.adam_general.core.adam.ADAM.multicov` on each
        sub-model — ``model.model_a.multicov(...)`` /
        ``model.model_b.multicov(...)`` — for per-side multi-step
        covariance instead. Raises :class:`NotImplementedError`.
        """
        raise NotImplementedError(
            "multicov() is not defined for OMG — the joint occurrence "
            "model has no closed-form multi-step covariance in terms of "
            "the per-sub-model state-space matrices. Call "
            "model.model_a.multicov(...) and model.model_b.multicov(...) "
            "for per-side covariances."
        )

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
            nlopt_kwargs=self.nlopt_kwargs,
            ets=self.ets,
            smoother=self.smoother,
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
        B_used = np.concatenate([b_a["B"], b_b["B"]])  # noqa: N806
        lb = np.concatenate([b_a["Bl"], b_b["Bl"]])
        ub = np.concatenate([b_a["Bu"], b_b["Bu"]])

        # Cache the per-side parameter names so ``_om_from_side`` can set
        # ``B_names`` on each sub-model. Without this, ``OM.coef_names`` falls
        # back to ``b1, b2, …`` and ``_clamp_confint_offsets`` silently fails
        # to clamp bounds on persistence parameters (cf. ``confint.omg``).
        self._names_a = list(b_a.get("names") or [])
        self._names_b = list(b_b.get("names") or [])

        # Respect user-supplied B / lb / ub from nlopt_kwargs. B is the JOINT
        # vector spanning A-side then B-side. dict-keyed names are matched
        # against suffixed names ("alpha_A", "alpha_B", ...); array-like B is
        # assigned positionally. lb / ub override positionally.
        kwargs = self.nlopt_kwargs or {}
        user_B = kwargs.get("B")  # noqa: N806
        user_lb = kwargs.get("lb")
        user_ub = kwargs.get("ub")

        names_a = b_a.get("names") or []
        names_b = b_b.get("names") or []
        joint_names = [f"{n}_A" for n in names_a] + [f"{n}_B" for n in names_b]

        if user_B is not None:
            if isinstance(user_B, dict):
                for k, v in user_B.items():
                    if k in joint_names:
                        B_used[joint_names.index(k)] = float(v)
            else:
                B_used[:] = np.asarray(user_B, dtype=float)
        if user_lb is not None:
            lb[:] = np.asarray(user_lb, dtype=float)
        if user_ub is not None:
            ub[:] = np.asarray(user_ub, dtype=float)

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
        kwargs = self.nlopt_kwargs or {}
        algorithm = kwargs.get("algorithm", "NLOPT_LN_NELDERMEAD")
        xtol_rel = kwargs.get("xtol_rel", 1e-6)
        xtol_abs = kwargs.get("xtol_abs", 1e-8)
        ftol_rel = kwargs.get("ftol_rel", 1e-8)
        ftol_abs = kwargs.get("ftol_abs", 0)
        maxeval = kwargs.get("maxeval", len(B_used) * 40)
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
                    loss=self.loss,  # type: ignore[arg-type]
                    loss_function=self.loss_function,
                    reg_lambda=self.reg_lambda,
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
        cf_value = opt.last_optimum_value()

        # Retry from a small-persistence safe point if the first run hit the
        # infeasibility plateau, but ONLY when the user did NOT supply their
        # own B — otherwise their B is the authoritative starting point.
        # Mirrors the failsafe in R/omg.R: all params 0.001 with the two
        # leading alphas (A-side and B-side) bumped to 0.01.
        user_B_supplied = kwargs.get("B") is not None  # noqa: N806
        if not user_B_supplied and (not np.isfinite(cf_value) or cf_value >= 1e300):
            B_used[:] = 0.001
            if len(B_used) > 0:
                B_used[0] = 0.01  # alpha for A-side
            if n_params_a < len(B_used):
                B_used[n_params_a] = 0.01  # alpha for B-side
            opt2 = nlopt.opt(nlopt_algorithm, len(B_used))
            opt2 = _configure_optimizer(
                opt2,
                lb,
                ub,
                maxeval,
                None,
                xtol_rel=xtol_rel,
                xtol_abs=xtol_abs,
                ftol_rel=ftol_rel,
                ftol_abs=ftol_abs,
            )
            opt2.set_min_objective(_objective)
            try:
                B_used[:] = opt2.optimize(B_used)
            except Exception:
                pass
            cf_value = opt2.last_optimum_value()

        return cf_value

    # ---------------------------------------------------------------------
    # Build per-sub-model OM objects from the joint estimate
    # ---------------------------------------------------------------------

    def _om_from_side(self, side, B, occurrence_str) -> OM:
        scaffold: OM = side["scaffold"]
        # Mark this sub-model so ``OM.actuals`` returns the latent
        # (unobservable) reconstruction rather than the binary indicator.
        # Mirrors R's ``omg_submodel`` S3 class tag on ``omg$modelA/B``.
        scaffold._is_omg_submodel = True
        # Inject the joint estimate into the scaffold so its post-fit
        # plumbing (om_preparator, model_name, etc.) reflects the joint
        # solution rather than re-running its own optimiser.
        side_names = self._names_a if occurrence_str == "odds-ratio" else self._names_b
        scaffold._adam_estimated = {
            "B": np.asarray(B, dtype=float),
            "B_names": side_names if len(side_names) == len(B) else None,
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

        # Mirror R's ``omgFinalFit`` (R/omg.R:864-906): reuse the matrices the
        # joint optimiser saw — built with ``Etype="A"`` for numerical
        # stability — rather than rebuilding with the user-requested
        # multiplicative types. Without this, the post-fit sub-model fitted
        # values diverge from R's reference (the optimum agrees but the link
        # function then takes ``log(fittedB)`` of values produced under a
        # different state-space structure). Standalone Python ``OM`` still
        # rebuilds via ``_build_final_fit_adam_created`` (matching standalone
        # R ``om()``); only the OMG post-fit shares the joint matrices.
        scaffold._adam_created = side["matrices_dict"]

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
        # Combine raw (pre-link) forecasts from both sub-models via
        # omg_link_function to produce the combined probability forecast.
        fc_a, fc_a_raw = self.model_a._run_forecaster(h, X_future=X_future)
        fc_b_raw = self.model_b._raw_forecast_direct(h, X_future=X_future)
        e_a = self._side_a["model_type_dict"]["error_type"]
        e_b = self._side_b["model_type_dict"]["error_type"]
        p_combined = omg_link_function(fc_a_raw, fc_b_raw, e_a, e_b)
        p_combined = np.where(np.isnan(p_combined), 1.0, p_combined)
        fc_a.mean[:] = p_combined
        return fc_a

    def simulate(
        self,
        nsim: int = 1,
        seed: Optional[int] = None,
        obs: Optional[int] = None,
        **kwargs,
    ):
        """Re-simulate the combined OMG occurrence series.

        Python port of R's ``simulate.omg`` (R/omg.R:1945-1997).
        Simulates the two sub-models independently via
        :meth:`OM.simulate`, combines their **latent** series via
        :func:`omg_link_function`, then draws 0/1 occurrence
        indicators with the resulting probability.

        Parameters
        ----------
        nsim, obs, **kwargs
            Forwarded to each sub-model's :meth:`OM.simulate`.
        seed : int, optional
            Master seed. Derived seeds ``seed`` and ``seed + 1`` are
            passed to the two sub-model calls so the joint result is
            reproducible from a single integer.

        Returns
        -------
        SimulateResult
            ``probability`` carries the combined probability series;
            ``occurrence`` carries the 0/1 indicators; ``model_a`` /
            ``model_b`` carry the sub-model :class:`SimulateResult`
            instances (with their ``latent`` fields exposed for any
            downstream inspection).
        """
        from smooth.adam_general.core.simulate.result import SimulateResult

        if self.model_a is None or self.model_b is None:
            raise RuntimeError("OMG must be fitted before ``simulate`` is called.")

        # Derived sub-model seeds so the joint result is reproducible
        # from one master seed.
        seed_a = seed
        seed_b = None if seed is None else seed + 1

        sim_a = self.model_a.simulate(nsim=nsim, seed=seed_a, obs=obs, **kwargs)
        sim_b = self.model_b.simulate(nsim=nsim, seed=seed_b, obs=obs, **kwargs)

        e_type_a = self.model_a._model_type.get("error_type", "A")
        e_type_b = self.model_b._model_type.get("error_type", "A")

        prob = omg_link_function(sim_a.latent, sim_b.latent, e_type_a, e_type_b)
        prob = np.asarray(prob, dtype=np.float64)
        if prob.ndim == 1:
            prob = prob.reshape(-1, 1)
        prob = np.nan_to_num(prob, nan=0.5, posinf=1.0, neginf=0.0)
        prob = np.clip(prob, 0.0, 1.0)

        rng = np.random.default_rng(seed)
        occurrence_data = rng.binomial(1, prob)

        obs_out, nsim_out = prob.shape
        if nsim_out == 1:
            data_out = pd.Series(prob[:, 0])
            residuals_out = pd.Series(np.zeros(obs_out))
        else:
            data_out = pd.DataFrame(prob)
            residuals_out = pd.DataFrame(np.zeros((obs_out, nsim_out)))

        model_label = f"{self.model} (occurrence simulated)"

        return SimulateResult(
            model=model_label,
            data=data_out,
            states=np.empty((0, 0, nsim_out)),
            residuals=residuals_out,
            probability=prob if nsim_out > 1 else prob[:, 0],
            occurrence=occurrence_data,
            model_a=sim_a,
            model_b=sim_b,
            occurrence_type="general",
            other={"binary_data": occurrence_data},
        )


def _build_omg_from_om_kwargs(**om_kwargs) -> OMG:
    """Construct an OMG from the kwargs supplied to OM(occurrence='general', ...).

    Forwards ``model`` / ``lags`` / ``orders`` as both ``model_a`` and
    ``model_b`` and delegates to :class:`OMG`.
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
        nlopt_kwargs=om_kwargs.pop("nlopt_kwargs", None),
    )
