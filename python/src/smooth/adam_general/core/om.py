"""Occurrence Model (OM).

This module implements the :class:`OM` class — a state-space model for the
*probability of occurrence* of demand. Useful for intermittent-demand data
where many observations are zero and the size of the non-zero demands is
modelled separately.

Supported ``occurrence`` values:

* ``"fixed"``, ``"odds-ratio"``, ``"inverse-odds-ratio"``, ``"direct"`` —
  handled by :class:`OM` itself (Stage 1).
* ``"general"`` — :class:`OM`'s ``__new__`` transparently returns an
  :class:`OMG` instance (Stage 2).
* ``"auto"`` — raises :class:`NotImplementedError`; coming in Stage 4.

Design: ``OM(ADAM)`` reuses ADAM's architector → creator → forecaster
machinery. It overrides exactly the surfaces that differ between regular ADAM
and an occurrence model — the cost function (Bernoulli likelihood on the link
function applied to raw fitted values), the post-estimation transformation of
fitted values into [0, 1] probabilities, and the model name format.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Callable, Dict, List, Literal, Optional, Union, cast

import nlopt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from smooth.adam_general.core.adam import ADAM
from smooth.adam_general.core.creator import (
    architector,
    creator,
    filler,
    initialiser,
)
from smooth.adam_general.core.estimator.optimization import (
    _configure_optimizer,
    _setup_arima_polynomials,
)
from smooth.adam_general.core.forecaster import forecaster
from smooth.adam_general.core.utils.gradient import adam_fit_or_gradient
from smooth.adam_general.core.utils.ic import ic_function
from smooth.adam_general.core.utils.om_cost import om_cf, om_link_function

# Map user-facing occurrence string to the single-character flag the C++
# adamCore.fit() expects via its ``O=`` parameter (see src/headers/adamCore.h).
_OCC_TO_CHAR = {
    "fixed": "f",
    "odds-ratio": "o",
    "inverse-odds-ratio": "i",
    "direct": "d",
}

# Uppercase letter shown in the printed model name, e.g. "oETS(MNN)[O]".
_OCC_TO_BRACKET = {
    "fixed": "F",
    "odds-ratio": "O",
    "inverse-odds-ratio": "I",
    "direct": "D",
}

OM_OCCURRENCE_OPTIONS = Literal["fixed", "odds-ratio", "inverse-odds-ratio", "direct"]


def om_initial_transform(
    mat_vt: np.ndarray,
    occurrence: str,
    error_type: str,
    trend_type: str,
    season_type: str,
    ets_model: bool,
    model_is_trendy: bool,
    model_is_seasonal: bool,
    initial_level_estimate: bool,
    initial_trend_estimate: bool,
    initial_seasonal_estimate,
    components_number_ets: int,
    components_number_ets_non_seasonal: int,
    components_number_ets_seasonal: int,
    lags_model: list,
    lags_model_max: int,
    lags_model_seasonal: list,
    obs_in_sample: int,
    ot: np.ndarray,
    arima_model: bool,
    components_number_arima: int,
    initial_arima_estimate: bool,
    initial_arima_number: int,
    xreg_model: bool,
    xreg_number: int,
    initial_xreg_estimate: bool,
    constant_required: bool,
    constant_estimate: bool,
) -> np.ndarray:
    """Transform initial state values from probability space onto the
    model-native scale before optimisation begins.

    Mutates ``mat_vt`` in place and also returns it.
    """

    def _transform(value):
        v = np.asarray(value, dtype=np.float64).copy()
        if occurrence == "odds-ratio":
            v = v / (1.0 - v)
        elif occurrence == "inverse-odds-ratio":
            v = (1.0 - v) / v
        # else: "fixed"/"direct" → identity
        if error_type == "A" and occurrence in ("odds-ratio", "inverse-odds-ratio"):
            v = np.log(v)
        return v

    # j tracks the number of ETS rows already consumed in mat_vt.
    j = 0

    if ets_model:
        # ---- Level
        if initial_level_estimate:
            if mat_vt[0, 0] < 0 or mat_vt[0, 0] > 1:
                level_original = float(np.mean(ot))
                mat_vt[0, :lags_model_max] = level_original
            mat_vt[0, :lags_model_max] = _transform(mat_vt[0, :lags_model_max])
        j = 1

        # ---- Trend (zero out for additive, set to 1 for multiplicative)
        if model_is_trendy:
            if initial_trend_estimate:
                if trend_type == "A":
                    mat_vt[j, :lags_model_max] = 0.0
                else:
                    mat_vt[j, :lags_model_max] = 1.0
            j += 1

        # ---- Seasonal
        if model_is_seasonal:
            estimates = initial_seasonal_estimate
            if isinstance(estimates, bool):
                estimates_iter = [estimates] * components_number_ets_seasonal
            else:
                estimates_iter = list(estimates)
            if any(estimates_iter):
                for i in range(components_number_ets_seasonal):
                    n = lags_model_seasonal[i]
                    seasonal_occ = np.asarray(mat_vt[j + i, :n], dtype=np.float64)
                    if season_type == "M":
                        # We pull the level off mat_vt[0,0] but it has just been
                        # transformed; for the multiplicative-seasonal-ratio R
                        # uses the ORIGINAL untransformed level. Recover it via
                        # the inverse of the level transform below if needed.
                        level_original = (
                            np.exp(mat_vt[0, 0])
                            if error_type == "A"
                            and occurrence in ("odds-ratio", "inverse-odds-ratio")
                            else mat_vt[0, 0]
                        )
                        if occurrence == "odds-ratio":
                            level_original = level_original / (1.0 + level_original)
                        elif occurrence == "inverse-odds-ratio":
                            level_original = 1.0 / (1.0 + level_original)
                        seasonal_occ = seasonal_occ / level_original + 1.0
                    else:
                        seasonal_occ = seasonal_occ - np.mean(seasonal_occ)
                    mat_vt[j + i, :n] = seasonal_occ
            j += components_number_ets_seasonal

    # ---- ARIMA
    # ARIMA initial states live in the same raw state-space as the level
    # *after* the level has been mapped onto the model-native scale; they are
    # not probabilities. Running ``_transform`` on them turns the default
    # seed of 0 into log(0) = -Inf for Etype="A", which corrupts the initial
    # parameter vector handed to the optimiser.
    if arima_model:
        j += components_number_arima

    # ---- xreg
    # xreg coefficients are regression weights, not probabilities. They can
    # be negative; running log(value/(1-value)) on a negative coefficient
    # gives NaN, which corrupts the initial parameter vector.
    if xreg_model:
        j += xreg_number

    # ---- constant
    # The constant lives on the same scale as the (already-transformed)
    # level. Transforming it again is a category error.

    return mat_vt


def om_preparator(
    *,
    model_type_dict,
    components_dict,
    lags_dict,
    matrices_dict,
    persistence_checked,
    initials_checked,
    arima_checked,
    explanatory_checked,
    phi_dict,
    constants_checked,
    observations_dict,
    profiles_dict,
    adam_estimated,
    adam_cpp,
    occurrence,
    occurrence_char,
    loss="likelihood",
):
    """OM-specific ``preparator``: runs the fit dispatcher with the ``O=`` flag
    (gradient solve for ``initial="gradient"``, otherwise the ordinary fit) and
    applies the occurrence link function to obtain probabilities.

    Returns a dict shaped exactly like the regular ADAM preparator output so
    the inherited :class:`OM` properties (``fitted``, ``residuals``,
    ``states``, ``transition`` …) work without further glue.
    """
    # 1. Fill matrices with optimised B
    matrices_dict = filler(
        adam_estimated["B"],
        model_type_dict=model_type_dict,
        components_dict=components_dict,
        lags_dict=lags_dict,
        matrices_dict=matrices_dict,
        persistence_checked=persistence_checked,
        initials_checked=initials_checked,
        arima_checked=arima_checked,
        explanatory_checked=explanatory_checked,
        phi_dict=phi_dict,
        constants_checked=constants_checked,
        adam_cpp=adam_cpp,
    )

    # 2. Profile setup
    profiles_recent_table = matrices_dict["mat_vt"][:, : lags_dict["lags_model_max"]]

    # 3. Run adam_cpp.fit() with vectorYt = vectorOt = ot, O = occurrence_char
    ot = np.asarray(observations_dict["ot"], dtype=np.float64)
    mat_vt = np.asfortranarray(matrices_dict["mat_vt"], dtype=np.float64)
    mat_wt = np.asfortranarray(matrices_dict["mat_wt"], dtype=np.float64)
    mat_f = np.asfortranarray(matrices_dict["mat_f"], dtype=np.float64)
    vec_g = np.asfortranarray(matrices_dict["vec_g"], dtype=np.float64)
    index_lookup_table = np.asfortranarray(
        profiles_dict["index_lookup_table"], dtype=np.uint64
    )
    profiles_recent_fortran = np.asfortranarray(profiles_recent_table, dtype=np.float64)

    if isinstance(initials_checked["initial_type"], list):
        backcast_value = any(
            t in ("complete", "backcasting") for t in initials_checked["initial_type"]
        )
    else:
        backcast_value = initials_checked["initial_type"] in (
            "complete",
            "backcasting",
        )

    adam_fitted = adam_fit_or_gradient(
        adam_cpp=adam_cpp,
        mat_vt=mat_vt,
        mat_wt=mat_wt,
        mat_f=mat_f,
        vec_g=vec_g,
        index_lookup_table=index_lookup_table,
        profiles_recent_table=profiles_recent_fortran,
        y_in_sample=ot,
        ot=ot,
        initial_type=initials_checked["initial_type"],
        n_iterations=initials_checked["n_iterations"],
        backcast_value=backcast_value,
        model_type_dict=model_type_dict,
        components_dict=components_dict,
        lags_dict=lags_dict,
        obs_in_sample=observations_dict["obs_in_sample"],
        o_type=occurrence_char,
        loss=loss,
    )

    # Fitted on probability scale
    error_type = model_type_dict["error_type"]
    raw_fitted = np.asarray(adam_fitted.fitted).ravel()
    p_fitted = om_link_function(raw_fitted, error_type, occurrence)
    p_fitted = np.where(np.isnan(p_fitted), 0.0, p_fitted)
    residuals = ot - p_fitted

    # 4. Initial values dict (use ADAM's _process_initial_values)
    from smooth.adam_general.core.forecaster.preparator import _process_initial_values

    matrices_for_initials = {
        "mat_vt": np.asarray(adam_fitted.states).copy(),
        "mat_wt": matrices_dict["mat_wt"],
        "mat_f": matrices_dict["mat_f"],
        "vec_g": matrices_dict["vec_g"],
    }
    initial_value, _, initial_estimated = _process_initial_values(
        model_type_dict,
        lags_dict,
        matrices_for_initials,
        components_dict,
        arima_checked,
        explanatory_checked,
        initials_checked,
    )

    # 5. Persistence vector (named)
    n_ets = components_dict.get("components_number_ets", 0)
    persistence_named: Dict[str, Any] = {}
    if n_ets > 0:
        names = ["alpha"]
        if model_type_dict.get("model_is_trendy"):
            names.append("beta")
        if model_type_dict.get("model_is_seasonal"):
            n_seas = components_dict.get("components_number_ets_seasonal", 1)
            if n_seas > 1:
                names += [f"gamma{i + 1}" for i in range(n_seas)]
            else:
                names += ["gamma"]
        for i, nm in enumerate(names[:n_ets]):
            persistence_named[nm] = float(matrices_dict["vec_g"][i, 0])

    return {
        "model": model_type_dict.get("model"),
        "time_elapsed": None,
        "holdout": False,
        "y_fitted_raw": raw_fitted,
        "y_fitted": p_fitted,
        "residuals": residuals,
        "states": adam_fitted.states,
        "profiles_recent_table": adam_fitted.profile,
        "persistence": persistence_named,
        "transition": matrices_dict["mat_f"],
        "measurement": matrices_dict["mat_wt"],
        "mat_vt": matrices_dict["mat_vt"],
        "mat_f": matrices_dict["mat_f"],
        "mat_wt": matrices_dict["mat_wt"],
        "phi": phi_dict.get("phi"),
        "initial_value": initial_value,
        "initial": initial_value,
        "initial_type": initials_checked["initial_type"],
        "initial_estimated": initial_estimated,
        "orders": None,
        "arma": None,
        "constant": None,
        "n_param": None,
        "occurrence": occurrence,
        "formula": (explanatory_checked or {}).get("formula"),
        "regressors": (explanatory_checked or {}).get("regressors"),
        "loss": "likelihood",
        "loss_value": adam_estimated["CF_value"],
        "log_lik": adam_estimated["log_lik_adam_value"],
        "distribution": "plogis",
        "scale": float("nan"),
        "other": None,
        "B": adam_estimated["B"],
        "lags": lags_dict["lags"],
        "lags_all": lags_dict["lags_model_all"],
        "FI": None,
    }


class OM(ADAM):
    """Occurrence model — state-space model for the probability of demand occurrence.

    Inherits the ADAM API surface and overrides the bits that differ for an
    occurrence model: cost function (Bernoulli likelihood applied to the
    link-transformed fitted values), distribution (always ``"plogis"``),
    scale (``nan``), and model-name format (``"oETS(...)[F|O|I|D]"``).
    """

    _OM_DEFAULT_LOSS = "likelihood"

    def __new__(cls, *args, **kwargs):
        occ = kwargs.get("occurrence")
        if occ == "general":
            from smooth.adam_general.core.omg import _build_omg_from_om_kwargs

            kwargs.pop("loss", None)
            kwargs.pop("n_iterations", None)
            return _build_omg_from_om_kwargs(**kwargs)
        if occ == "auto":
            from smooth.adam_general.core.auto_om import AutoOM

            kwargs.pop("occurrence", None)
            return AutoOM(
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k
                    in (
                        "model",
                        "lags",
                        "orders",
                        "h",
                        "holdout",
                        "persistence",
                        "phi",
                        "initial",
                        "constant",
                        "arma",
                        "regressors",
                        "ic",
                        "bounds",
                        "verbose",
                        "nlopt_kargs",
                    )
                }
            )
        return super().__new__(cls)

    def __init__(
        self,
        model: Union[str, List[str]] = "ZXZ",
        lags: Optional[Union[List[int], NDArray]] = None,
        ar_order: Union[int, List[int]] = 0,
        i_order: Union[int, List[int]] = 0,
        ma_order: Union[int, List[int]] = 0,
        orders: Optional[Dict[str, Any]] = None,
        constant: bool = False,
        formula: Optional[str] = None,
        regressors: Literal["use", "select", "adapt"] = "use",
        # ``str`` (not just the OM_OCCURRENCE_OPTIONS literals) because ``__new__``
        # also routes "general"/"auto", and wrappers pass runtime-validated strings.
        occurrence: str = "odds-ratio",
        loss: Union[
            Literal["likelihood", "MSE", "MAE", "HAM", "LASSO", "RIDGE", "custom"],
            Callable,
        ] = "likelihood",
        reg_lambda: Optional[float] = None,
        h: int = 0,
        holdout: bool = False,
        # ``float`` permitted so the "fixed" occurrence path can set persistence=0.
        persistence: Optional[Union[Dict[str, float], float]] = None,
        phi: Optional[float] = None,
        initial: Union[Dict[str, Any], str] = "backcasting",
        n_iterations: Optional[int] = None,
        arma: Optional[Dict[str, Any]] = None,
        ic: Literal["AIC", "AICc", "BIC", "BICc"] = "AICc",
        bounds: Literal["usual", "admissible", "none"] = "usual",
        verbose: int = 0,
        nlopt_kargs: Optional[Dict[str, Any]] = None,
        ets: Literal["conventional", "adam"] = "conventional",
        **kwargs,
    ) -> None:
        if occurrence in ("auto", "general"):
            # __new__ handled the redirect to OMG; __init__ on the OMG
            # instance has already run.  This guard is just a safety net for
            # any code path that constructs OM directly via __init__.
            return
        if occurrence not in _OCC_TO_CHAR:
            raise ValueError(
                f"Invalid occurrence={occurrence!r}; expected one of "
                f"{list(_OCC_TO_CHAR)}."
            )
        # Accept callable for a user-defined custom loss (signature
        # ``(actual, fitted, B) -> scalar`` — same as ADAM, mirrors R's
        # ``adamGeneral.R:574-602``). The callable flows through to
        # ADAM's ``parameter_checks._check_distribution_and_loss`` which
        # detects ``callable(loss)`` and sets the internal flag to
        # ``"custom"`` while populating ``_general["loss_function"]``.
        if not callable(loss) and loss not in (
            "likelihood",
            "MSE",
            "MAE",
            "HAM",
            "LASSO",
            "RIDGE",
        ):
            raise ValueError(
                f"Invalid loss={loss!r}; expected one of "
                "'likelihood', 'MSE', 'MAE', 'HAM', 'LASSO', 'RIDGE', "
                "or a callable returning a scalar."
            )

        # For "fixed" occurrence the model is forced to ANN with persistence
        # disabled and an analytic initial level.
        if occurrence == "fixed":
            model = "ANN"
            persistence = 0
            initial = "optimal"

        super().__init__(
            model=model,
            lags=lags,
            ar_order=ar_order,
            i_order=i_order,
            ma_order=ma_order,
            orders=orders,
            constant=constant,
            regressors=regressors,
            distribution="dnorm",
            # ADAM.__init__ types `loss` as a string Literal; the callable
            # path is handled by ADAM's parameter_checks (which natively
            # accepts callables and flips the internal flag to "custom").
            loss=loss,  # type: ignore[arg-type]
            ic=ic,
            bounds=bounds,
            occurrence=occurrence,
            persistence=persistence,
            phi=phi,
            initial=initial,
            n_iterations=n_iterations,
            arma=arma,
            verbose=verbose,
            h=h,
            holdout=holdout,
            nlopt_kargs=nlopt_kargs,
            ets=ets,
            reg_lambda=reg_lambda,
            # ``reg_lambda`` is the user-facing name on OM (mirrors ADAM's
            # public surface), but the cost function reads
            # ``_general["lambda"]`` which is populated from
            # ``lambda_param`` in ``parameters_checker``. Forward both so
            # LASSO / RIDGE actually see the chosen weight.
            lambda_param=reg_lambda,
            **kwargs,
        )

        self._is_om = True
        # Set to True by OMG._om_from_side when this OM instance is a
        # sub-model of an OMG; flips ``actuals`` from binary-indicator to
        # the latent reconstruction. See OM.actuals.
        self._is_omg_submodel: bool = False
        self._om_occurrence = occurrence
        self._occurrence_char = _OCC_TO_CHAR[occurrence]
        self._formula = formula

    # ------------------------------------------------------------------
    # Hook overrides
    # ------------------------------------------------------------------

    def _select_distribution(self):
        """Always ``"plogis"`` for occurrence models."""
        self._general["distribution_new"] = "plogis"
        self._general["distribution"] = "plogis"

    def _restore_user_model_spec(self, requested_model_spec: Optional[str]) -> None:
        """Re-apply the user's explicit ETS code after parameters_checker.

        For OM with binary 0/1 data, parameters_checker (via
        ``model_checks.py:572``) auto-downgrades any ``M`` components to
        ``A``. R's equivalent does not — multiplicative components are valid
        for occurrence models because the link function maps the raw fitted
        values into [0, 1]. This restores the user-requested codes.
        Wildcards ``Z``/``X``/``Y`` are left as parameters_checker resolved
        them.
        """
        if not requested_model_spec:
            return
        spec = requested_model_spec
        e_req = spec[0]
        # Trend can be 1 or 2 chars (e.g. "Ad")
        if len(spec) == 4:
            t_letter = spec[1]
            damped_req = spec[2] == "d"
            s_req = spec[3]
        else:
            t_letter = spec[1]
            damped_req = False
            s_req = spec[2] if len(spec) >= 3 else "N"

        changed = False
        if e_req in ("A", "M") and self._model_type.get("error_type") != e_req:
            self._model_type["error_type"] = e_req
            changed = True
        if t_letter in ("A", "M") and self._model_type.get("trend_type") != t_letter:
            self._model_type["trend_type"] = t_letter
            self._model_type["model_is_trendy"] = True
            changed = True
        if t_letter == "N" and self._model_type.get("trend_type") != "N":
            # User explicitly asked for no trend — respect it
            self._model_type["trend_type"] = "N"
            self._model_type["model_is_trendy"] = False
            changed = True
        if damped_req != self._model_type.get("damped", False) and t_letter in (
            "A",
            "M",
        ):
            self._model_type["damped"] = damped_req
            self._phi_internal["phi_estimate"] = damped_req
            changed = True
        if s_req in ("A", "M") and self._model_type.get("season_type") != s_req:
            self._model_type["season_type"] = s_req
            self._model_type["model_is_seasonal"] = True
            changed = True
        if s_req == "N" and self._model_type.get("season_type") != "N":
            self._model_type["season_type"] = "N"
            self._model_type["model_is_seasonal"] = False
            changed = True

        if changed:
            new_model = self._model_type["error_type"] + self._model_type["trend_type"]
            if self._model_type.get("damped", False) and self._model_type[
                "trend_type"
            ] not in ("N", ""):
                new_model += "d"
            new_model += self._model_type["season_type"]
            self._model_type["model"] = new_model

    # ------------------------------------------------------------------
    # fit / predict
    # ------------------------------------------------------------------

    def fit(self, y: NDArray, X: Optional[NDArray] = None):
        if getattr(self, "arima_select", False):
            from smooth.adam_general.core.auto_om import AutoOM

            ar = getattr(self, "ar_order", 0) or 3
            i_ = getattr(self, "i_order", 0) or 2
            ma = getattr(self, "ma_order", 0) or 3
            return AutoOM(
                model=cast(str, self.model),  # OM always carries a string spec
                lags=getattr(self, "lags", None),
                occurrence=[self._om_occurrence],
                arima_select=True,
                ar_order=ar,
                i_order=i_,
                ma_order=ma,
                h=getattr(self, "h", 0),
                holdout=getattr(self, "holdout", False),
                persistence=getattr(self, "persistence", None),
                phi=getattr(self, "phi", None),
                initial=getattr(self, "initial", "backcasting"),
                ic=getattr(self, "ic", "AICc"),
                bounds=getattr(self, "bounds", "usual"),
                ets=self.ets,
                constant=getattr(self, "constant", False),
                arma=getattr(self, "arma", None),
                regressors=getattr(self, "regressors", "use"),
                verbose=getattr(self, "verbose", 0),
                nlopt_kargs=getattr(self, "nlopt_kargs", None),
            ).fit(y, X)

        self._start_time = time.time()
        # Stash the user's requested model spec so we can restore "M*"
        # components after parameters_checker (which auto-downgrades them when
        # the data contains zeros — see model_checks.py:572). For OM, binary
        # data always has zeros yet multiplicative components are valid via
        # the link function.
        requested_model_spec = (
            self.model
            if isinstance(self.model, str) and len(self.model) in (3, 4)
            else None
        )
        self._check_parameters(y, X)

        # Pure-regression early exit: when the user supplied an explicit
        # ALM regression model, refit it under the plogis link and return.
        if getattr(self, "_alm_model", None) is not None:
            from greybox import ALM as _ALM

            alm_orig = self._alm_model
            alm_plogis = _ALM(distribution="plogis")
            alm_plogis.fit(
                np.asarray(alm_orig._X_train_),
                np.asarray(alm_orig._y_train_),
            )
            self._alm_model = alm_plogis
            self._populate_from_alm(y, X)
            self.time_elapsed_ = time.time() - self._start_time
            return self

        self._restore_user_model_spec(requested_model_spec)

        # Replace y_in_sample with the binary occurrence indicator so the
        # inherited forecaster/preparator paths see the 0/1 series.
        ot = np.asarray(self._observations["ot"], dtype=np.float64)
        self._observations["y_in_sample"] = ot
        self._observations["obs_zero"] = int(np.sum(~self._observations["ot_logical"]))
        # Binarise the holdout as well.
        y_hld_raw = self._observations.get("y_holdout")
        if self._general.get("holdout") and y_hld_raw is not None:
            self._observations["y_holdout"] = (np.asarray(y_hld_raw) != 0).astype(
                np.float64
            )

        if self._om_occurrence == "fixed":
            self._fit_fixed()
        else:
            self._fit_occurrence()

        self._prepare_results_om()
        self._set_om_fitted_attributes()

        # Auto-forecast if h > 0 — mirrors ADAM._auto_predict() but applies the
        # occurrence link to the raw forecast.
        self._auto_predict_om()

        self.time_elapsed_ = time.time() - self._start_time

        # Consolidate init params into _config (mirrors ADAM.fit())
        self._config = {
            "model": self.model,
            "lags": self.lags,
            "ar_order": self.ar_order,
            "i_order": self.i_order,
            "ma_order": self.ma_order,
            "orders": self._init_orders,
            "constant": self.constant,
            "regressors": self.regressors,
            # Carry the callable through ``_config`` so ``coefbootstrap``
            # refits replay the same custom loss; otherwise the resolved
            # string flag.
            "loss": (
                self._general.get("loss_function")
                if self._general.get("loss") == "custom"
                else self.loss
            ),
            "reg_lambda": self.reg_lambda,
            "ic": self.ic,
            "bounds": self.bounds,
            "occurrence": self._om_occurrence,
            "persistence": self.persistence,
            "phi": self.phi,
            "initial": self.initial,
            "n_iterations": self.n_iterations,
            "arma": self.arma,
            "h": self.h,
            "holdout": self.holdout,
            "formula": self._formula,
        }
        for key in (
            "lags",
            "ar_order",
            "i_order",
            "ma_order",
            "constant",
            "regressors",
            "loss",
            "ic",
            "bounds",
            "persistence",
            "phi",
            "initial",
            "n_iterations",
            "arma",
            "holdout",
        ):
            try:
                delattr(self, key)
            except AttributeError:
                pass

        return self

    # ------------------------------------------------------------------
    # Internal: estimation paths
    # ------------------------------------------------------------------

    def _build_om_artifacts(self):
        """Run architector → creator → om_initial_transform; return artifacts.

        Mirrors the in-place R block in om.R (lines ~317-375): the creator is
        called with ``Etype="A"`` to keep the state-space decomposition
        well-defined for binary data.
        """
        (
            model_type_dict,
            components_dict,
            lags_dict,
            observations_dict,
            profile_dict,
            adam_cpp,
        ) = architector(
            self._model_type,
            self._lags_model,
            self._observations,
            self._arima,
            self._explanatory,
            self._constant,
            self.profiles_recent_table,
            self.profiles_recent_provided,
            self.ets == "adam",
        )

        # R: adam_creator() called with Etype="A" so the matrix layout assumes
        # additive errors regardless of the modeled error_type. We flip the
        # creator-only Etype/Ttype/Stype temporarily.
        original_error_type = model_type_dict["error_type"]
        original_trend_type = model_type_dict.get("trend_type", "N")
        original_season_type = model_type_dict.get("season_type", "N")
        creator_model_type = dict(model_type_dict)
        creator_model_type["error_type"] = "A"
        if original_trend_type != "N":
            creator_model_type["trend_type"] = "A"
        if original_season_type != "N":
            creator_model_type["season_type"] = "A"

        # R: otLogicalInternal[] <- TRUE  (om.R:198)
        # The creator's initialiser reads ot_logical to subset observations
        # for the level mean; on binary data this would degenerate to
        # mean(ot[ot==1]) = 1. Pretend every observation is "active" so the
        # level is initialised to mean(ot) — the actual occurrence rate.
        original_ot_logical = observations_dict["ot_logical"]
        observations_dict_for_creator = dict(observations_dict)
        observations_dict_for_creator["ot_logical"] = np.ones_like(
            original_ot_logical, dtype=bool
        )

        adam_created = creator(
            creator_model_type,
            lags_dict,
            profile_dict,
            observations_dict_for_creator,
            self._persistence,
            self._initials,
            self._arima,
            self._constant,
            self._phi_internal,
            components_dict,
            self._explanatory,
            smoother=self.smoother,
        )

        # Apply occurrence-specific transform of the initial state vector
        adam_created["mat_vt"] = om_initial_transform(
            mat_vt=adam_created["mat_vt"],
            occurrence=self._om_occurrence,
            error_type=original_error_type,
            trend_type=original_trend_type,
            season_type=original_season_type,
            ets_model=model_type_dict.get("ets_model", False),
            model_is_trendy=model_type_dict.get("model_is_trendy", False),
            model_is_seasonal=model_type_dict.get("model_is_seasonal", False),
            initial_level_estimate=self._initials.get("initial_level_estimate", True),
            initial_trend_estimate=self._initials.get("initial_trend_estimate", True),
            initial_seasonal_estimate=self._initials.get(
                "initial_seasonal_estimate", True
            ),
            components_number_ets=components_dict.get("components_number_ets", 0),
            components_number_ets_non_seasonal=components_dict.get(
                "components_number_ets_non_seasonal", 0
            ),
            components_number_ets_seasonal=components_dict.get(
                "components_number_ets_seasonal", 0
            ),
            lags_model=lags_dict["lags_model"],
            lags_model_max=lags_dict["lags_model_max"],
            lags_model_seasonal=lags_dict.get("lags_model_seasonal", []),
            obs_in_sample=observations_dict["obs_in_sample"],
            ot=observations_dict["ot"],
            arima_model=self._arima.get("arima_model", False),
            components_number_arima=components_dict.get("components_number_arima", 0),
            initial_arima_estimate=self._initials.get("initial_arima_estimate", False),
            initial_arima_number=self._initials.get("initial_arima_number", 0),
            xreg_model=self._explanatory.get("xreg_model", False),
            xreg_number=self._explanatory.get("xreg_number", 0),
            initial_xreg_estimate=self._initials.get("initial_xreg_estimate", False),
            constant_required=self._constant.get("constant_required", False),
            constant_estimate=self._constant.get("constant_estimate", False),
        )

        # Persist updated state on self for downstream code
        self._model_type = model_type_dict
        self._components = components_dict
        self._lags_model = lags_dict
        self._observations = observations_dict
        self._profile = profile_dict
        self._adam_created = adam_created
        self._adam_cpp = adam_cpp

        return adam_cpp, adam_created, profile_dict

    def _build_final_fit_adam_created(self, profile_dict):
        """Rebuild matrices using the ORIGINAL error/trend/season types.

        During optimisation the model is forced to use additive error/trend/
        season types for numerical stability. For the final standalone fit
        we rebuild the matrices using the user-requested Etype/Ttype/Stype,
        because the difference matters for the seasonal initial-state values
        in ``mat_vt`` that seed the backcasting inside ``adamCpp$fit``.
        """
        model_type_dict = self._model_type
        original_ot_logical = self._observations["ot_logical"]
        observations_dict_for_creator = dict(self._observations)
        observations_dict_for_creator["ot_logical"] = np.ones_like(
            original_ot_logical, dtype=bool
        )

        adam_created = creator(
            model_type_dict,  # original types — no forcing to "A"
            self._lags_model,
            profile_dict,
            observations_dict_for_creator,
            self._persistence,
            self._initials,
            self._arima,
            self._constant,
            self._phi_internal,
            self._components,
            self._explanatory,
            smoother=self.smoother,
        )

        adam_created["mat_vt"] = om_initial_transform(
            mat_vt=adam_created["mat_vt"],
            occurrence=self._om_occurrence,
            error_type=model_type_dict["error_type"],
            trend_type=model_type_dict.get("trend_type", "N"),
            season_type=model_type_dict.get("season_type", "N"),
            ets_model=model_type_dict.get("ets_model", False),
            model_is_trendy=model_type_dict.get("model_is_trendy", False),
            model_is_seasonal=model_type_dict.get("model_is_seasonal", False),
            initial_level_estimate=self._initials.get("initial_level_estimate", True),
            initial_trend_estimate=self._initials.get("initial_trend_estimate", True),
            initial_seasonal_estimate=self._initials.get(
                "initial_seasonal_estimate", True
            ),
            components_number_ets=self._components.get("components_number_ets", 0),
            components_number_ets_non_seasonal=self._components.get(
                "components_number_ets_non_seasonal", 0
            ),
            components_number_ets_seasonal=self._components.get(
                "components_number_ets_seasonal", 0
            ),
            lags_model=self._lags_model["lags_model"],
            lags_model_max=self._lags_model["lags_model_max"],
            lags_model_seasonal=self._lags_model.get("lags_model_seasonal", []),
            obs_in_sample=self._observations["obs_in_sample"],
            ot=self._observations["ot"],
            arima_model=self._arima.get("arima_model", False),
            components_number_arima=self._components.get("components_number_arima", 0),
            initial_arima_estimate=self._initials.get("initial_arima_estimate", False),
            initial_arima_number=self._initials.get("initial_arima_number", 0),
            xreg_model=self._explanatory.get("xreg_model", False),
            xreg_number=self._explanatory.get("xreg_number", 0),
            initial_xreg_estimate=self._initials.get("initial_xreg_estimate", False),
            constant_required=self._constant.get("constant_required", False),
            constant_estimate=self._constant.get("constant_estimate", False),
        )

        return adam_created

    def _fit_occurrence(self):
        """Estimate persistence + initials for non-fixed occurrence types."""
        adam_cpp, adam_created, profile_dict = self._build_om_artifacts()

        b_values = initialiser(
            model_type_dict=self._model_type,
            components_dict=self._components,
            lags_dict=self._lags_model,
            adam_created=adam_created,
            persistence_checked=self._persistence,
            initials_checked=self._initials,
            arima_checked=self._arima,
            constants_checked=self._constant,
            explanatory_checked=self._explanatory,
            observations_dict=self._observations,
            bounds=self._general["bounds"],
            phi_dict=self._phi_internal,
            profile_dict=profile_dict,
            adam_cpp=adam_cpp,
            other_parameter_estimate=False,
            other_value=2.0,
        )

        # Optimisation knobs (subset of nlopt_kargs we care about for OM)
        kargs = self.nlopt_kargs or {}

        # Respect user-supplied B / lb / ub from nlopt_kargs (mirrors R's
        # adam_checkOptimizer + adam.R:1229-1361 pattern). Named B is supplied
        # as a dict (parameter-name -> value); array-like B is assigned
        # positionally. lb / ub fall back to b_values only when not provided.
        user_B = kargs.get("B")  # noqa: N806
        user_lb = kargs.get("lb")
        user_ub = kargs.get("ub")

        B = np.asarray(b_values["B"], dtype=float)  # noqa: N806
        names = b_values.get("names")
        if user_B is not None:
            if isinstance(user_B, dict):
                if names is None:
                    raise ValueError(
                        "Initialiser did not return parameter names; "
                        "cannot apply dict-keyed user B."
                    )
                for k, v in user_B.items():
                    if k in names:
                        B[names.index(k)] = float(v)
            else:
                B[:] = np.asarray(user_B, dtype=float)

        lb = (
            np.asarray(user_lb, dtype=float)
            if user_lb is not None
            else np.asarray(b_values["Bl"], dtype=float)
        )
        ub = (
            np.asarray(user_ub, dtype=float)
            if user_ub is not None
            else np.asarray(b_values["Bu"], dtype=float)
        )

        ar_polynomial_matrix, ma_polynomial_matrix = _setup_arima_polynomials(
            self._model_type, self._arima, self._lags_model
        )
        algorithm = kargs.get("algorithm", "NLOPT_LN_NELDERMEAD")
        xtol_rel = kargs.get("xtol_rel", 1e-6)
        xtol_abs = kargs.get("xtol_abs", 1e-8)
        ftol_rel = kargs.get("ftol_rel", 1e-8)
        ftol_abs = kargs.get("ftol_abs", 0)
        maxeval = kargs.get("maxeval", len(B) * 40)
        if self._explanatory.get("xreg_model"):
            maxeval = max(maxeval, max(1000, len(B) * 100))

        nlopt_algorithm = getattr(
            nlopt, algorithm.replace("NLOPT_", ""), nlopt.LN_NELDERMEAD
        )

        regressors = self._explanatory.get("regressors")
        # Mirror R: omCF receives `loss` separately; we store it in a dict for
        # om_cf to read. ``loss_function`` is already in ``_general`` from
        # ADAM's checker when the user passes a callable.
        general_for_cf = dict(self._general)
        general_for_cf["loss"] = self._general.get("loss", "likelihood")

        # Snapshot the *pristine* mat_vt and profiles_recent_table BEFORE
        # NLopt mutates them.  The C++ kernel's head-fill / backcasting
        # passes overwrite ``adam_created["mat_vt"]`` and
        # ``profile_dict["profiles_recent_table"]`` in place every CF
        # evaluation; without this snapshot the post-fit Hessian (used by
        # vcov / FI) would be computed off a CF that disagrees with R's
        # by ~0.002 — R re-creates fresh matrices for FI via
        # adam_creator + om_initial_transform on every Hessian probe
        # (R/om.R:830-870).  We do the equivalent here by stashing the
        # initial bytes and restoring them in _fisher_information_matrix.
        self._fi_pristine = {
            "mat_vt": adam_created["mat_vt"].copy(),
            "profiles_recent_table": profile_dict["profiles_recent_table"].copy(),
        }

        def _objective(x, _grad):
            try:
                cf_value = om_cf(
                    B=x,
                    model_type_dict=self._model_type,
                    components_dict=self._components,
                    lags_dict=self._lags_model,
                    matrices_dict=adam_created,
                    persistence_checked=self._persistence,
                    initials_checked=self._initials,
                    arima_checked=self._arima,
                    explanatory_checked=self._explanatory,
                    phi_dict=self._phi_internal,
                    constants_checked=self._constant,
                    observations_dict=self._observations,
                    profile_dict=profile_dict,
                    general=general_for_cf,
                    adam_cpp=adam_cpp,
                    occurrence=self._om_occurrence,
                    occurrence_char=self._occurrence_char,
                    bounds=self._general["bounds"],
                    arPolynomialMatrix=ar_polynomial_matrix,
                    maPolynomialMatrix=ma_polynomial_matrix,
                    regressors=regressors,
                )
            except Exception:
                cf_value = 1e100
            return float(cf_value) if np.isfinite(cf_value) else 1e300

        # Nothing to estimate (e.g. fixed persistence with backcasting /
        # gradient initials, which are solved inside the fit and never enter
        # B): skip nlopt — a zero-dimension problem never evaluates the
        # objective and last_optimum_value() would return +inf. Evaluate the
        # cost once at the empty B instead. Mirrors R/om.R's length(B)==0 guard.
        if len(B) == 0:
            cf_value = _objective(B, None)
        else:
            opt = nlopt.opt(nlopt_algorithm, len(B))
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
                B[:] = opt.optimize(B)
            except Exception:
                pass
            cf_value = opt.last_optimum_value()

        # Retry from a small-persistence safe point if the first run hit the
        # infeasibility plateau, but ONLY when the user did NOT supply their
        # own B — otherwise their B is the authoritative starting point.
        # Failsafe mirrors R/om.R: all params 0.001 with alpha bumped to 0.01.
        if user_B is None and (not np.isfinite(cf_value) or cf_value >= 1e300):
            B[:] = 0.001
            if len(B) > 0:
                B[0] = 0.01
            opt2 = nlopt.opt(nlopt_algorithm, len(B))
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
                B[:] = opt2.optimize(B)
            except Exception:
                pass
            cf_value = opt2.last_optimum_value()

        n_param_estimated = len(B)
        log_lik_value = -float(cf_value)

        self._adam_estimated = {
            "B": B,
            "B_names": list(names) if names is not None else None,
            "CF_value": float(cf_value),
            "n_param_estimated": n_param_estimated,
            "log_lik_adam_value": {
                "value": log_lik_value,
                "nobs": self._observations["obs_in_sample"],
                "df": n_param_estimated,
            },
            "arima_polynomials": adam_created.get("arima_polynomials"),
            "adam_cpp": adam_cpp,
        }
        self._ic_selection = ic_function(
            self._general["ic"],
            self._adam_estimated["log_lik_adam_value"],
        )

    def _fit_fixed(self):
        """Fixed occurrence: no optimisation. Initial level = mean(ot)."""
        # Force model_type to ANN and persistence_level=0 — parameters_checker
        # has already received model="ANN" so this should already hold. We
        # additionally hard-set the persistence flag so the initialiser sees a
        # zero-length B.
        self._persistence["persistence_estimate"] = False
        self._persistence["persistence_level_estimate"] = False
        self._persistence["persistence_level"] = 0.0
        self._initials["initial_type"] = "provided"
        self._initials["initial_level_estimate"] = False
        self._initials["initial_estimate"] = False
        self._initials["initial_level"] = float(np.mean(self._observations["ot"]))

        adam_cpp, adam_created, profile_dict = self._build_om_artifacts()

        # Manually overwrite the level row of mat_vt with the analytical mean
        # (om_initial_transform leaves it untouched because *_estimate=False)
        lmm = self._lags_model["lags_model_max"]
        adam_created["mat_vt"][0, :lmm] = self._initials["initial_level"]

        # Compute Bernoulli log-lik analytically with constant probability p=mean(ot)
        p = self._initials["initial_level"]
        ot_logical = self._observations["ot_logical"]
        eps = 1e-15
        ll = float(
            np.sum(np.log(max(p, eps)) * ot_logical.sum())
            + np.sum(np.log(max(1.0 - p, eps)) * (~ot_logical).sum())
        )

        n_param_estimated = 1  # The level
        self._adam_estimated = {
            "B": np.array([], dtype=float),
            "CF_value": -ll,
            "n_param_estimated": n_param_estimated,
            "log_lik_adam_value": {
                "value": ll,
                "nobs": self._observations["obs_in_sample"],
                "df": n_param_estimated,
            },
            "arima_polynomials": adam_created.get("arima_polynomials"),
            "adam_cpp": adam_cpp,
        }
        self._ic_selection = ic_function(
            self._general["ic"],
            self._adam_estimated["log_lik_adam_value"],
        )

    # ------------------------------------------------------------------
    # Post-fit packaging
    # ------------------------------------------------------------------

    def _prepare_results_om(self):
        # Bypass ADAM's _format_time_series_data (we just store y as np arrays)
        self._select_distribution()
        self._prepared = om_preparator(
            model_type_dict=self._model_type,
            components_dict=self._components,
            lags_dict=self._lags_model,
            matrices_dict=self._adam_created,
            persistence_checked=self._persistence,
            initials_checked=self._initials,
            arima_checked=self._arima,
            explanatory_checked=self._explanatory,
            phi_dict=self._phi_internal,
            constants_checked=self._constant,
            observations_dict=self._observations,
            profiles_dict=self._profile,
            adam_estimated=self._adam_estimated,
            adam_cpp=self._adam_cpp,
            occurrence=self._om_occurrence,
            occurrence_char=self._occurrence_char,
            loss=self._general.get("loss", "likelihood"),
        )
        self._prepared["holdout"] = self._general.get("holdout", False)

    def _set_om_fitted_attributes(self):
        # Persistence trailing-underscore attrs (alpha/beta/gamma)
        persistence = self._prepared.get("persistence", {}) or {}
        if "alpha" in persistence:
            self.persistence_level_ = persistence["alpha"]
        if "beta" in persistence:
            self.persistence_trend_ = persistence["beta"]
        if "gamma" in persistence:
            self.persistence_seasonal_ = persistence["gamma"]

        # Build the model name in oETS(...)[X] form
        e = self._model_type.get("error_type", "")
        t = self._model_type.get("trend_type", "")
        s = self._model_type.get("season_type", "")
        if self._model_type.get("damped", False) and t not in ("", "N"):
            t = t + "d"
        ets_str = f"{e}{t}{s}"
        bracket = _OCC_TO_BRACKET[self._om_occurrence]
        parts = [f"oETS({ets_str})[{bracket}]"]
        if self._arima.get("arima_model"):
            ar = (self._arima.get("ar_orders") or [0])[0]
            i = (self._arima.get("i_orders") or [0])[0]
            ma = (self._arima.get("ma_orders") or [0])[0]
            parts.append(f"ARIMA({ar},{i},{ma})")
        self.model = "+".join(parts)

    def _auto_predict_om(self):
        h = getattr(self, "h", None)
        if not h or h <= 0:
            return
        self._general["h"] = h
        # Reuse the inherited prediction-data preparation, then forecaster.
        # The forecaster reads y_fitted from self._prepared which we have set
        # to probabilities; for the raw forecast call we need to re-run the
        # C++ forecaster on the optimal states then apply the link.
        self._auto_forecast = self._om_forecast(h, X_future=None)

    def _run_forecaster(self, h: int, X_future=None):
        """Call the C++ forecaster; return ``(fc, raw_array)`` (before link function).

        We must NOT call ADAM's :func:`preparator`: it would re-run the
        C++ fitter with ``O='n'`` (additive recursion) and overwrite the
        OM states. Instead, feed the OM-computed states (already in
        :attr:`_prepared`) directly into the C++ forecaster via a minimal
        ``model_prepared`` dict.
        """
        general_dict = dict(self._general)
        general_dict["h"] = h
        general_dict["interval"] = "none"
        general_dict["distribution"] = "dnorm"
        general_dict["distribution_new"] = "dnorm"

        prep = self._prepared
        model_prepared = {
            "model": self._model_type.get("model"),
            "y_fitted": prep["y_fitted"],
            "residuals": prep["residuals"],
            "states": prep["states"],
            "profiles_recent_table": prep["profiles_recent_table"],
            "persistence": prep["persistence"],
            "transition": prep["transition"],
            "measurement": prep["measurement"],
            "mat_vt": prep["mat_vt"],
            "mat_f": prep["mat_f"],
            "mat_wt": prep["mat_wt"],
            "phi": prep["phi"],
            "scale": 1.0,
            "initial_value": prep["initial_value"],
            "initial_type": prep["initial_type"],
            "occurrence": "none",
            "distribution": "dnorm",
            "loss": "likelihood",
            "loss_value": prep["loss_value"],
            "log_lik": prep["log_lik"],
            "B": prep["B"],
            "lags": prep["lags"],
            "lags_all": prep["lags_all"],
            "regressors": prep.get("regressors"),
            "constant": prep.get("constant"),
            "n_param": [[0], [0]],
            "FI": None,
        }

        if X_future is not None and self._explanatory.get("xreg_model"):
            self._explanatory["new_xreg"] = np.asarray(X_future, dtype=float)

        fc = forecaster(
            model_prepared=model_prepared,
            observations_dict=self._observations,
            general_dict=general_dict,
            occurrence_dict=self._occurrence_dict_for_forecaster(),
            lags_dict=self._lags_model,
            model_type_dict=self._model_type,
            explanatory_checked=self._explanatory,
            components_dict=self._components,
            constants_checked=self._constant,
            params_info=self._params_info,
            adam_cpp=self._adam_cpp,
            interval="none",
            level=0.95,
            side="both",
        )
        return fc, np.asarray(fc.mean.values, dtype=float)

    def _raw_forecast_direct(self, h: int, X_future=None):
        """Raw (pre-link) state-space forecast; used by OMG to combine sub-models."""
        _, raw = self._run_forecaster(h, X_future)
        return raw

    def _om_forecast(self, h: int, X_future=None):
        """Generate an h-step probability forecast."""
        fc, raw = self._run_forecaster(h, X_future)
        e_type = self._model_type["error_type"]
        p_forecast = om_link_function(raw, e_type, self._om_occurrence)
        is_logit = (
            self._om_occurrence in ("odds-ratio", "inverse-odds-ratio")
            and e_type == "A"
        )
        if is_logit:
            p_forecast = np.where(np.isnan(p_forecast), 1.0, p_forecast)
        fc.mean[:] = p_forecast
        return fc

    def _occurrence_dict_for_forecaster(self):
        """Forecaster expects an occurrence_dict with these keys; we feed
        ``occurrence_model=False`` so it does not try to apply a separate
        occurrence layer on top of probabilities we are already producing."""
        return {
            "occurrence": "none",
            "occurrence_model": False,
            "p_fitted": np.ones_like(np.asarray(self._observations["ot"], dtype=float)),
            "ot_logical": self._observations["ot_logical"],
            "oes_model": "none",
        }

    def predict(
        self,
        h: int,
        X: Optional[NDArray] = None,
        # Wider than ADAM's interval literals (LSP-safe); only "none" is supported
        # and other values warn and fall back to point forecasts.
        interval: str = "none",
        level: Optional[Union[float, List[float]]] = 0.95,
        side: Literal["both", "upper", "lower"] = "both",
        cumulative: bool = False,
        nsim: int = 10000,
        occurrence: Optional[NDArray] = None,
        scenarios: bool = False,
        seed: Optional[int] = None,
    ):
        """Probability forecast for the occurrence model.

        Currently only ``interval="none"`` is supported; intervals on the
        probability scale are not implemented yet.
        """
        self._check_is_fitted()
        if interval != "none":
            warnings.warn(
                "Intervals on the probability scale are not implemented for OM "
                "models; returning point forecasts only."
            )
        return self._om_forecast(h, X_future=X)

    # ------------------------------------------------------------------
    # Property overrides (return OM-specific values)
    # ------------------------------------------------------------------

    @property
    def occurrence(self) -> str:
        """Occurrence type used for fitting (e.g. ``"odds-ratio"``)."""
        return self._om_occurrence

    @occurrence.setter
    def occurrence(self, value):
        # ADAM.__init__ assigns to self.occurrence; route into the private
        # backing attribute so OM keeps a stable, read-only public API.
        self._om_occurrence = value

    @property
    def occurrence_char(self) -> str:
        """Single-letter occurrence flag handed to the C++ fitter."""
        return self._occurrence_char

    @property
    def scale(self) -> float:
        """Link-scale residual std-dev — alias for :attr:`sigma`.

        Mirrors :attr:`ADAM.scale`, which is an alias for ``sigma`` (the
        Python convention is that ``scale`` and ``sigma`` both report the
        std-dev, not the variance — even though R calls the variance
        ``s2`` and ``sigma`` the std-dev). See :attr:`sigma` for the
        underlying formula and R reference.
        """
        return self.sigma

    @property
    def sigma(self) -> float:
        """Link-scale residual std-dev: ``sqrt(mean(residuals²))``.

        Matches R's ``sigma.om`` (R/om.R): the OM residuals live on the
        link-transformed scale (logit / log-odds), so their second moment
        is a meaningful scale parameter for the underlying ETS, even
        though there's no equivalent on the probability axis.

        Returns NaN only if the model has no residuals yet (e.g.
        constructed but not fitted).
        """
        residuals = getattr(self, "_prepared", {}).get("residuals")
        if residuals is None:
            return float("nan")
        return float(np.sqrt(np.mean(np.asarray(residuals, dtype=float) ** 2)))

    @property
    def distribution_(self) -> str:
        """Always ``"plogis"`` for occurrence models."""
        return "plogis"

    @property
    def loss_(self) -> str:
        return self._general.get("loss", self._OM_DEFAULT_LOSS)

    @property
    def model_name(self) -> str:
        """Model name in ``oETS(...)[X]`` form (uppercase X = F/O/I/D)."""
        self._check_is_fitted()
        return self.model if isinstance(self.model, str) else str(self.model)

    @property
    def actuals(self) -> NDArray:
        """In-sample actuals.

        For a standalone :class:`OM`, returns the binary 0/1 occurrence
        indicator. For an OM that has been built as a sub-model of an
        :class:`~smooth.adam_general.core.omg.OMG` (flag
        ``_is_omg_submodel`` set by ``OMG._om_from_side``), returns the
        latent unobservable value the sub-model was implicitly fitting:
        ``fitted + residuals`` — mirrors R's ``actuals.omg_submodel``
        (R/omg.R:1748-1756). OM stores residuals additively
        (``ot - fitted``) regardless of error type, so the same formula
        recovers the latent value for both ``Etype="A"`` and ``Etype="M"``.
        """
        self._check_is_fitted()
        if getattr(self, "_is_omg_submodel", False):
            fitted = np.asarray(self.fitted, dtype=float)
            residuals = np.asarray(self.residuals, dtype=float)
            return fitted + residuals
        return np.asarray(self._observations["ot"], dtype=float)

    @property
    def fitted(self) -> NDArray:
        """In-sample fitted *probabilities* in [0, 1]."""
        self._check_is_fitted()
        return np.asarray(self._prepared["y_fitted"], dtype=float)

    @property
    def residuals(self) -> NDArray:
        """``ot - p_fitted`` (binary indicator minus fitted probability)."""
        self._check_is_fitted()
        return np.asarray(self._prepared["residuals"], dtype=float)

    def rstandard(self) -> NDArray:
        """Pearson standardised residuals for the occurrence model.

        Formula: ``(ot - p) / sqrt(p*(1-p)) * sqrt(n/df)``
        where ``df = n - k`` (n observations, k estimated parameters).
        """
        self._check_is_fitted()
        obs = self.nobs
        df = obs - self.nparam
        p = self.fitted
        e = self.residuals
        return e / np.sqrt(p * (1 - p)) * np.sqrt(obs / df)

    def rstudent(self) -> NDArray:
        """Pearson studentised (leave-one-out) residuals for the occurrence model.

        Formula: ``(ot - p) / sqrt(p*(1-p)) * sqrt(n/df)``
        where ``df = n - k - 1``.
        """
        self._check_is_fitted()
        obs = self.nobs
        df = obs - self.nparam - 1
        p = self.fitted
        e = self.residuals
        return e / np.sqrt(p * (1 - p)) * np.sqrt(obs / df)

    @property
    def loglik(self) -> float:
        self._check_is_fitted()
        return float(self._adam_estimated["log_lik_adam_value"]["value"])

    # ------------------------------------------------------------------
    # Inference (overrides ADAM's FI path because om_cf — not the ADAM
    # likelihood — is the cost the OM optimiser minimised)
    # ------------------------------------------------------------------

    def _fisher_information_matrix(self, step_size=None):
        """Observed FI for OM (Hessian of ``om_cf`` at the estimated B).

        ``om_cf`` already returns the negative log-likelihood, so its Hessian
        is the observed Fisher Information directly — no sign flip.

        Bounds are disabled (``bounds="none"``) during the Hessian call to
        match R's ``vcov.adam`` (R/adam.R:2797 — ``boundsFI <- "none"``). With
        the user's usual/admissible bounds left on, FD perturbations ``B ± h``
        that cross the feasibility boundary would return the 1e300 penalty
        instead of the actual log-likelihood — the resulting second
        derivative blows up and the inverse FI collapses to ~0. ``bounds=
        "none"`` lets the underlying ``adam_cpp.fit`` evaluate cleanly even
        when smoothing parameters are nominally out of range.
        """
        from smooth.adam_general.core.utils.var_covar import numerical_hessian

        self._check_is_fitted()

        ar_polynomial_matrix, ma_polynomial_matrix = _setup_arima_polynomials(
            self._model_type, self._arima, self._lags_model
        )
        general_for_cf = dict(self._general)
        general_for_cf["loss"] = self._general.get("loss", "likelihood")

        # R rebuilds adam_created fresh every Hessian probe via
        # adam_creator + om_initial_transform (R/om.R:830-870).  We do the
        # equivalent by restoring the pristine pre-NLopt snapshots of
        # mat_vt and profiles_recent_table before each _cost call — the
        # kernel will mutate them again inside om_cf via the
        # backcasting / head-fill passes, so each call gets the same
        # starting state R sees.
        pristine = getattr(self, "_fi_pristine", None)

        def _cost(b):
            if pristine is not None:
                self._adam_created["mat_vt"][...] = pristine["mat_vt"]
                self._profile["profiles_recent_table"][...] = pristine[
                    "profiles_recent_table"
                ]
            return om_cf(
                B=b,
                model_type_dict=self._model_type,
                components_dict=self._components,
                lags_dict=self._lags_model,
                matrices_dict=self._adam_created,
                persistence_checked=self._persistence,
                initials_checked=self._initials,
                arima_checked=self._arima,
                explanatory_checked=self._explanatory,
                phi_dict=self._phi_internal,
                constants_checked=self._constant,
                observations_dict=self._observations,
                profile_dict=self._profile,
                general=general_for_cf,
                adam_cpp=self._adam_cpp,
                occurrence=self._om_occurrence,
                occurrence_char=self._occurrence_char,
                bounds="none",
                arPolynomialMatrix=ar_polynomial_matrix,
                maPolynomialMatrix=ma_polynomial_matrix,
                regressors=self._explanatory.get("regressors"),
            )

        return numerical_hessian(_cost, self.coef, step_size=step_size)

    def summary(self, level: float = 0.95, digits: int = 4):
        """Coefficient-table summary, mirroring R's ``summary.om``.

        Identical content to :meth:`ADAM.summary`, but the printed report is
        prefixed by an ``"Occurrence model"`` header (R's ``summary.om``
        prepends the same line before delegating to ``summary.adam``).
        """
        from smooth.adam_general.core.utils.printing import OMSummary

        self._check_is_fitted()
        return OMSummary(self, level=level, digits=digits)

    def simulate(
        self,
        nsim: int = 1,
        seed: Optional[int] = None,
        obs: Optional[int] = None,
        randomizer: Optional[Any] = None,
        **randomizer_kwargs,
    ):
        """Re-simulate probabilities + 0/1 occurrence indicators.

        Python port of R's ``simulate.om`` (R/om.R:2272-2342). Calls
        :meth:`ADAM.simulate` to obtain the latent ETS series via
        ``super().simulate(...)``, maps it to a probability via
        :func:`om_link_function`, and draws 0/1 occurrence indicators
        via ``rng.binomial(1, prob)``.

        Parameters
        ----------
        nsim : int, default 1
            Number of simulated series.
        seed : int, optional
            RNG seed. Pins both the latent-error draw and the
            binomial occurrence draw.
        obs : int, optional
            Observations per simulated series. Defaults to the
            in-sample length.
        randomizer : str | callable, optional
            Forwarded to :meth:`ADAM.simulate`. When ``None`` (the
            default), substitute a Gaussian-noise sampler with the
            empirical residual std as the scale — needed because OM
            stores ``distribution="plogis"`` for which
            ``generate_errors()`` has no closed-form branch and
            ``self.scale`` is ``NaN``.
        **randomizer_kwargs
            Forwarded to the randomizer.

        Returns
        -------
        SimulateResult
            ``data`` and ``probability`` carry the probability
            series; ``occurrence`` carries the 0/1 indicators;
            ``latent`` carries the pre-link state-space output (used
            by :meth:`OMG.simulate` to combine sub-models).
        """
        from smooth.adam_general.core.simulate.result import SimulateResult

        self._check_is_fitted()

        # 1. Latent simulation via ADAM.simulate. Override the
        # default ``randomizer`` with a Gaussian sampler on the
        # empirical residual scale — OM uses ``distribution="plogis"``
        # which has no branch in ``generate_errors()``.
        effective_randomizer = randomizer
        if effective_randomizer is None:
            effective_randomizer = _om_default_randomizer(self, seed=seed)

        latent_sim = super().simulate(
            nsim=nsim,
            seed=seed,
            obs=obs,
            randomizer=effective_randomizer,
            **randomizer_kwargs,
        )

        # 2. Extract the latent matrix shape from latent_sim.data.
        latent_arr = (
            latent_sim.data.to_numpy()
            if isinstance(latent_sim.data, pd.Series)
            else latent_sim.data.to_numpy()
        )
        if latent_arr.ndim == 1:
            latent_arr = latent_arr.reshape(-1, 1)
        n_obs_out = latent_arr.shape[0]
        nsim_out = latent_arr.shape[1]

        # 3. latent → probability via the link function.
        e_type = self._model_type.get("error_type", "A")
        occurrence = self._om_occurrence
        probability = om_link_function(latent_arr.ravel(), e_type, occurrence)
        probability = np.asarray(probability, dtype=np.float64).reshape(
            n_obs_out, nsim_out
        )
        # ``om_link_function``'s odds-ratio paths can overshoot under
        # extreme latent noise; clip uniformly as a numerical guard.
        # NaN can appear on multiplicative-ETS paths where the latent
        # state collapses to zero — replace with the neutral 0.5 so
        # the downstream binomial draw is well-defined.
        probability = np.nan_to_num(probability, nan=0.5, posinf=1.0, neginf=0.0)
        probability = np.clip(probability, 0.0, 1.0)

        # 4. 0/1 indicator draw, seeded from the master seed.
        rng = np.random.default_rng(seed)
        occurrence_data = rng.binomial(1, probability)

        # 5. Output assembly.
        if nsim_out == 1:
            data_out = pd.Series(probability[:, 0])
        else:
            data_out = pd.DataFrame(probability)

        base_model = self._model_type.get("model", "oETS")
        model_label = f"{base_model} (occurrence simulated)"

        return SimulateResult(
            model=model_label,
            data=data_out,
            states=latent_sim.states,
            residuals=latent_sim.residuals,
            latent=latent_arr,
            probability=probability if nsim_out > 1 else probability[:, 0],
            occurrence=occurrence_data,
            occurrence_type=occurrence,
            other={"binary_data": occurrence_data},
        )


def _om_default_randomizer(om_object, seed=None):
    """Gaussian-noise randomizer keyed on the empirical residual std.

    OM stores ``distribution="plogis"`` (the logit link) for which
    :func:`generate_errors` has no closed-form branch, and
    ``OM.scale`` returns ``NaN``. Substituting ``rnorm(0, sigma)``
    with ``sigma = sqrt(mean(residuals**2))`` is the equivalent of
    the R helper's fallback in ``simulateADAMCore`` for distributions
    without a closed-form scale (R/adam.R, the post-refactor
    ``simulateADAMCore`` body).

    The ``seed`` argument is threaded through so the returned
    callable is reproducible — same seed in, same draws out.
    """
    res = np.asarray(om_object.residuals, dtype=np.float64).ravel()
    if res.size:
        sigma = float(np.sqrt(np.nanmean(res**2)))
    else:
        sigma = 1.0
    if not np.isfinite(sigma) or sigma == 0.0:
        sigma = 1.0
    rng = np.random.default_rng(seed)

    def _draw(n):
        return rng.normal(0.0, sigma, n)

    return _draw
