"""Scale model for ADAM -- the Python port of R's ``sm.adam`` (``R/sm.R``).

A scale model is a second ADAM fitted to a transform of the first model's
residuals, so the location model's variance becomes time-varying instead of a
constant. The scale model is fitted with a custom loss: it is *scored* by the
**location** model's log-likelihood, with the scale model's own fitted values
supplying that distribution's scale at each observation.

Attach the result to the location model with ``location.scale_model = s`` --
R needs a separate ``implant()`` verb because it cannot mutate the fitted
object in place; the property setter does the same job here.
"""

import warnings
from typing import Any, Callable, Dict, Optional

import numpy as np
from numpy.typing import NDArray
from scipy import special

from smooth.adam_general.core.utils.cost_functions import _sum_r

# Distributions whose likelihood ``sm()`` knows how to score. R's switch has
# entries for dalaplace/dlogis and the log-variants commented out, so they are
# genuinely unsupported rather than merely absent here.
SUPPORTED_DISTRIBUTIONS = (
    "dnorm",
    "dlaplace",
    "ds",
    "dgnorm",
    "dlnorm",
    "dgamma",
    "dinvgauss",
)

# Distributions for which an additive-error scale model has to be fitted in
# logarithms (R/sm.R: the logModelSM branch).
_LOG_TRICK_DISTRIBUTIONS = ("dnorm", "dlaplace", "ds", "dgnorm")


def _residual_transform(
    residuals: NDArray, distribution: str, other: Optional[float]
) -> NDArray:
    """The scale response: residuals mapped to the quantity the scale measures.

    Mirrors the ``et`` switch in ``R/sm.R``. Each transform is the per-observation
    contribution to that distribution's scale estimator, so fitting a model to it
    is fitting a model to the scale itself.
    """
    e = np.asarray(residuals, dtype=np.float64)
    if distribution == "dnorm":
        return e**2
    if distribution in ("dlaplace", "dalaplace"):
        return np.abs(e)
    if distribution == "ds":
        return 0.5 * np.abs(e) ** 0.5
    if distribution == "dgnorm":
        if other is None:
            raise ValueError("dgnorm needs a shape; the model carries none.")
        return (other * np.abs(e) ** other) ** (1.0 / other)
    if distribution == "dlnorm":
        return np.log(e) ** 2
    if distribution == "dgamma":
        return (e - 1.0) ** 2
    if distribution == "dinvgauss":
        return (e - 1.0) ** 2 / e
    raise ValueError(f"sm() does not support distribution {distribution!r}.")


def _holdout_transform(
    e_forecast: NDArray,
    distribution: str,
    error_type: str,
    other: Optional[float],
    forecast: NDArray,
) -> NDArray:
    """Same mapping for the holdout errors (R/sm.R, the ``newCall$data`` block).

    The log-domain and multiplicative distributions differ from the in-sample
    transform here: the holdout error is on the original scale, so it is
    converted to a relative error first.
    """
    ef = np.asarray(e_forecast, dtype=np.float64)
    fc = np.asarray(forecast, dtype=np.float64)
    if distribution == "dnorm":
        return ef**2
    if distribution in ("dlaplace", "dalaplace"):
        return np.abs(ef)
    if distribution == "ds":
        return 0.5 * np.abs(ef) ** 0.5
    if distribution == "dgnorm":
        if other is None:
            raise ValueError("dgnorm needs a shape; the model carries none.")
        return (other * np.abs(ef) ** other) ** (1.0 / other)
    if distribution == "dlnorm":
        return np.log(1.0 + (ef / fc if error_type == "A" else ef)) ** 2
    if distribution == "dgamma":
        return (ef / fc if error_type == "A" else ef) ** 2
    if distribution == "dinvgauss":
        rel = ef / fc if error_type == "A" else ef
        return rel**2 / (rel + 1.0)
    raise ValueError(f"sm() does not support distribution {distribution!r}.")


def _standardise_residuals(
    location_residuals: NDArray,
    scale_fitted: NDArray,
    distribution: str,
    other: Optional[float],
) -> NDArray:
    """Residuals rescaled by the fitted scale, so they follow the unit distribution.

    Mirrors the final ``adamModel$residuals`` switch in ``R/sm.R``: N(0,1),
    Laplace(0,1), S(0,1), GN(0,1,beta), logN(-1/2,1), Gamma(sigma^-2,1).
    ``dinvgauss`` keeps the scale model's own residuals, as R does.
    """
    e = np.asarray(location_residuals, dtype=np.float64)
    f = np.asarray(scale_fitted, dtype=np.float64)
    if distribution == "dnorm":
        return e / np.sqrt(f)
    if distribution == "dlaplace":
        return e / f
    if distribution == "ds":
        return e / f**2
    if distribution == "dgnorm":
        if other is None:
            raise ValueError("dgnorm needs a shape; the model carries none.")
        return e / f ** (1.0 / other)
    if distribution == "dlnorm":
        return np.exp((np.log(e) + f**2 / 2.0 - 0.5) / f) - 1.0
    if distribution == "dgamma":
        return e / np.sqrt(f) - 1.0
    # dinvgauss is handled by the caller (it keeps the scale model's residuals)
    return e / f


def _log_density(
    distribution: str,
    error_type: str,
    y: NDArray,
    mu: NDArray,
    scale: NDArray,
    other: Optional[float],
) -> NDArray:
    """Log-density of the *location* model at a per-observation scale.

    ``scale`` is the scale model's fitted vector. The ``error_type`` branches
    reproduce how R scales a multiplicative-error model's density by the fitted
    value (``R/sm.R``, the ``lossFunction`` switch).
    """
    if distribution == "dnorm":
        sd = np.sqrt(scale) * (mu if error_type == "M" else 1.0)
        return -0.5 * np.log(2.0 * np.pi) - np.log(sd) - 0.5 * ((y - mu) / sd) ** 2

    if distribution == "dlaplace":
        b = scale * (mu if error_type == "M" else 1.0)
        return -np.log(2.0 * b) - np.abs(y - mu) / b

    if distribution == "ds":
        b = scale * (np.sqrt(mu) if error_type == "M" else 1.0)
        return -np.log(4.0 * b**2) - np.sqrt(np.abs(y - mu)) / b

    if distribution == "dgnorm":
        if other is None:
            raise ValueError("dgnorm needs a shape; the model carries none.")
        b = scale * (mu**other if error_type == "M" else 1.0)
        return (
            np.log(other)
            - np.log(2.0 * b)
            - special.gammaln(1.0 / other)
            - (np.abs(y - mu) / b) ** other
        )

    if distribution == "dlnorm":
        # R uses sdlog=sqrt(f) but meanlog=log(mu)-f^2/2 -- f plays the role of
        # a variance in one and a standard deviation in the other. Reproduced as
        # written; changing it would move every dlnorm scale model off R.
        sdlog = np.sqrt(scale)
        meanlog = np.log(np.abs(mu)) - scale**2 / 2.0
        return -np.log(y * sdlog * np.sqrt(2.0 * np.pi)) - (
            np.log(y) - meanlog
        ) ** 2 / (2.0 * sdlog**2)

    if distribution == "dinvgauss":
        mean = np.abs(mu)
        disp = np.abs(scale / mu)
        return -0.5 * (np.log(np.pi * 2.0 * disp) + 3.0 * np.log(y)) - (
            y - mean
        ) ** 2 / (2.0 * disp * y * mean**2)

    if distribution == "dgamma":
        shape = 1.0 / scale
        scl = scale * mu
        return (
            (shape - 1.0) * np.log(y)
            - y / scl
            - special.gammaln(shape)
            - shape * np.log(scl)
        )

    raise ValueError(f"sm() does not support distribution {distribution!r}.")


def _differential_entropy(
    distribution: str, scale_zero: NDArray, other: Optional[float]
) -> float:
    """Entropy contributed by the zero observations of an occurrence model.

    ``scale_zero`` is the scale model's fitted values at the zero observations,
    so the sum already runs over them -- unlike ``adam()``, where the scale is a
    scalar and the term carries an explicit ``obsZero`` multiplier.
    """
    s = np.asarray(scale_zero, dtype=np.float64)
    if s.size == 0:
        return 0.0
    if distribution in ("dnorm", "dlnorm"):
        return float(_sum_r(np.log(np.sqrt(2.0 * np.pi) * s) + 0.5))
    if distribution == "dgnorm":
        if other is None:
            raise ValueError("dgnorm needs a shape; the model carries none.")
        return float(
            _sum_r(1.0 / other - np.log(other / (2.0 * s * special.gamma(1.0 / other))))
        )
    if distribution == "dinvgauss":
        return float(_sum_r(0.5 * (np.log(np.pi / 2.0) + 1.0 + np.log(s))))
    if distribution == "dgamma":
        return float(
            _sum_r(
                1.0 / s
                + np.log(s)
                + special.gammaln(1.0 / s)
                + (1.0 - 1.0 / s) * special.digamma(1.0 / s)
            )
        )
    if distribution == "dlaplace":
        return float(_sum_r(1.0 + np.log(2.0 * s)))
    if distribution == "ds":
        return float(_sum_r(2.0 + 2.0 * np.log(2.0 * s)))
    return 0.0


def make_scale_loss(
    distribution: str,
    error_type: str,
    y_in_sample: NDArray,
    y_fitted: NDArray,
    ot_logical: NDArray,
    other: Optional[float],
    log_model: bool,
    occurrence_model: bool,
) -> Callable[..., float]:
    """Build the custom loss ``sm()`` hands to ``ADAM``.

    The closure is called as ``loss(actual=, fitted=, B=)`` where ``fitted`` is
    the *scale* model's fitted vector. ``actual`` is ignored: the objective is
    the location model's negative log-likelihood, evaluated at the scale the
    scale model currently proposes.
    """
    y = np.asarray(y_in_sample, dtype=np.float64).ravel()
    mu = np.asarray(y_fitted, dtype=np.float64).ravel()
    ot = np.asarray(ot_logical, dtype=bool).ravel()

    def loss(actual: Any = None, fitted: Any = None, B: Any = None, **_: Any) -> float:
        scale = np.asarray(fitted, dtype=np.float64).ravel()
        if log_model:
            scale = np.exp(scale)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            ll = _log_density(distribution, error_type, y[ot], mu[ot], scale[ot], other)
            cf = -float(_sum_r(ll))
            if occurrence_model:
                cf += _differential_entropy(distribution, scale[~ot], other)
        return cf

    return loss


def _resolve_log_model(sm_model: str, distribution: str) -> bool:
    """Whether the scale model has to be fitted in logs (``R/sm.R``).

    An additive-error scale model cannot keep a strictly-positive scale
    positive, so R takes logs of the response. A multiplicative or
    multiplicative-selection (``M``/``Y``) first letter does not need it.

    R's guard also tests ``!is.null(orders) || !is.null(formula) || first in
    ("A","X")``, but ``orders`` is a formal with the default
    ``list(ar=0,i=0,ma=0,select=FALSE)`` and so is never NULL -- that clause is
    a tautology, and only the letter and the distribution actually decide.
    """
    first = sm_model[0] if sm_model else ""
    return first not in ("M", "Y") and distribution in _LOG_TRICK_DISTRIBUTIONS


def build_scale_response(
    location: Any,
    distribution: str,
    error_type: str,
    other: Optional[float],
) -> Dict[str, Any]:
    """Assemble the scale model's response from the location model's residuals.

    Returns the in-sample response, the holdout response (or ``None``), the
    occurrence mask and the occurrence flag.
    """
    e = np.asarray(location.residuals, dtype=np.float64).ravel()
    y_in = np.asarray(location.actuals, dtype=np.float64).ravel()

    occ = getattr(location, "_occurrence", {}) or {}
    occurrence_model = bool(occ.get("occurrence_model"))
    if occurrence_model:
        ot = y_in != 0
    else:
        ot = np.ones(e.size, dtype=bool)

    # Assign into the non-zero positions only. Transforming the whole vector
    # would recycle a shorter right-hand side across the sample for occurrence
    # models -- the defect fixed in R/sm.R alongside this port.
    response = e.copy()
    response[ot] = _residual_transform(e[ot], distribution, other)

    return {
        "response": response,
        "ot_logical": ot,
        "occurrence_model": occurrence_model,
        "y_in_sample": y_in,
        "y_fitted": np.asarray(location.fitted, dtype=np.float64).ravel(),
    }


def sm(
    location: Any,
    model: Optional[str] = None,
    lags: Optional[Any] = None,
    orders: Optional[Any] = None,
    constant: Optional[bool] = None,
    regressors: Optional[str] = None,
    X: Optional[Any] = None,
    **kwargs: Any,
) -> Any:
    """Fit a scale model for an already-estimated ADAM (R: ``sm.adam``).

    The returned model's fitted values are the location model's time-varying
    scale. Attach it with ``location.scale_model = s``.

    Only arguments passed explicitly are forwarded to :class:`ADAM`, mirroring
    R, where ``newCall`` is the *matched* call: ``sm()``'s own signature
    defaults never reach ``adam()``, so ``adam()``'s defaults apply (notably
    ``initial="backcasting"``, not the ``"optimal"`` that heads ``sm()``'s
    formal).

    Parameters
    ----------
    location : ADAM
        A fitted model estimated with ``loss="likelihood"``.
    model : str, optional
        Scale model specification; ``"YYY"`` (multiplicative selection) as in R.
    lags, orders, constant, regressors, X
        Passed through to :class:`ADAM`; ``lags`` defaults to the location
        model's and ``constant`` to ``False``.
    **kwargs
        Any other :class:`ADAM` argument (``initial``, ``ic``, ``bounds``, ...).

    Returns
    -------
    ADAM
        The fitted scale model, with ``is_scale_`` set.
    """
    from smooth.adam_general.core.adam import ADAM

    if getattr(location, "loss_", None) != "likelihood":
        raise ValueError(
            "sm() only works with models estimated via maximisation of "
            f"likelihood. Yours was estimated via {location.loss_}. "
            "Cannot proceed."
        )

    distribution = location.distribution_
    if distribution not in SUPPORTED_DISTRIBUTIONS:
        raise ValueError(
            f"sm() does not support distribution {distribution!r}. "
            f"Supported: {', '.join(SUPPORTED_DISTRIBUTIONS)}."
        )

    error_type = location.error_type
    other_dict = getattr(location, "other", None)
    other = None
    if isinstance(other_dict, dict):
        other = other_dict.get("shape", other_dict.get("alpha"))

    sm_model = "YYY" if model is None else model
    if any(c in ("Z", "F", "P") for c in (sm_model[0], sm_model[1], sm_model[-1])):
        warnings.warn(
            "This type of model selection is not supported by the sm() function.",
            stacklevel=2,
        )

    info = build_scale_response(location, distribution, error_type, other)
    response = info["response"]

    # Holdout: append the transformed forecast errors, so the scale model is
    # fitted on the same sample the location model was and scored on the rest.
    h = int(getattr(location, "h", 0) or 0)
    holdout = bool(getattr(location, "holdout", False)) and h > 0
    if holdout:
        y_holdout = np.asarray(location.holdout_data, dtype=np.float64).ravel()
        forecast = np.asarray(location.predict(h=h).mean, dtype=np.float64).ravel()
        e_fc = (
            y_holdout - forecast
            if error_type == "A"
            else (y_holdout - forecast) / forecast
        )
        response = np.concatenate(
            [
                response,
                _holdout_transform(e_fc, distribution, error_type, other, forecast),
            ]
        )

    log_model = _resolve_log_model(sm_model, distribution)
    if log_model:
        warnings.warn(
            "This type of model can only be applied to the data in logarithms. "
            "Amending the data",
            stacklevel=2,
        )
        response = np.log(response)

    loss = make_scale_loss(
        distribution,
        error_type,
        info["y_in_sample"],
        info["y_fitted"],
        info["ot_logical"],
        other,
        log_model,
        info["occurrence_model"],
    )

    args: Dict[str, Any] = {
        "model": sm_model,
        "lags": location.lags_used if lags is None else lags,
        "constant": False if constant is None else constant,
        "regressors": "use" if regressors is None else regressors,
        "distribution": distribution,
        "loss": loss,
        "outliers": "ignore",
        "h": h,
        "holdout": holdout,
    }
    if orders is not None:
        args["orders"] = orders
    if distribution in ("dgnorm", "dlgnorm") and other is not None:
        args["gnorm_shape"] = other
    # Reuse the location model's *fitted* occurrence model, as R does
    # (newCall$occurrence <- object$occurrence): the scale model shares that
    # occurrence rather than estimating a second one on the transformed
    # residuals, which are a different series with different zeroes.
    if info["occurrence_model"] and getattr(location, "om_model", None) is not None:
        args["occurrence"] = location.om_model
    args.update(kwargs)

    scale_model = ADAM(**args)
    scale_model.fit(response, X)

    _finalise(scale_model, location, distribution, other, log_model)
    return scale_model


def _finalise(
    scale_model: Any,
    location: Any,
    distribution: str,
    other: Optional[float],
    log_model: bool,
) -> None:
    """Post-fit corrections R applies before returning the scale model."""
    if log_model:
        # Undo the log trick on everything the user reads back.
        scale_model._prepared["fitted"] = np.exp(
            np.asarray(scale_model.fitted, dtype=np.float64)
        )
        scale_model.model_name_sm_ = f"{scale_model.model_name} in logs"

    # logLik is the loss (the location likelihood), not the scale model's own.
    scale_model.loglik_sm_ = -float(scale_model.loss_value)
    # -1 removes the scale from the location model's parameter count.
    scale_model.df_sm_ = int(scale_model.nparam) + int(location.nparam) - 1

    if distribution != "dinvgauss":
        scale_model._prepared["residuals"] = _standardise_residuals(
            np.asarray(location.residuals, dtype=np.float64).ravel(),
            np.asarray(scale_model.fitted, dtype=np.float64).ravel(),
            distribution,
            other,
        )

    scale_model.is_scale_ = True
    scale_model.location_ = location
