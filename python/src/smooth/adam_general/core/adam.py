import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats as scipy_stats

from smooth.adam_general._adam_general import adam_simulator
from smooth.adam_general.core.checker import parameters_checker
from smooth.adam_general.core.creator import architector, creator
from smooth.adam_general.core.estimator import (
    estimator,
    selector,
)
from smooth.adam_general.core.forecaster import forecaster, preparator
from smooth.adam_general.core.utils.ic import calculate_ic_weights, ic_function
from smooth.adam_general.core.utils.n_param import NParam
from smooth.adam_general.core.utils.utils import SMOOTHER_DEFAULT

# Note: adam_cpp instance is stored in self and passed to forecasting functions


# Type hint groups
DISTRIBUTION_OPTIONS = Literal[
    "default", "dnorm", "dlaplace", "ds", "dgnorm", "dlnorm", "dinvgauss", "dgamma"
]

LOSS_OPTIONS = Literal[
    "likelihood",
    "GPL",
    "MSE",
    "MAE",
    "HAM",
    "MSEh",
    "MAEh",
    "HAMh",
    "MSCE",
    "MACE",
    "CHAM",
    "TMSE",
    "TMAE",
    "THAM",
    "GTMSE",
    "GTAME",
    "GTHAM",
    "LASSO",
    "RIDGE",
]

OCCURRENCE_OPTIONS = Literal[
    "none", "auto", "fixed", "general", "odds-ratio", "inverse-odds-ratio", "direct"
]

# ``str`` is included alongside the documented literals because wrapper classes
# (ES, MSARIMA, AutoADAM, …) accept arbitrary user strings and the value is
# validated at runtime by ``parameters_checker``.
INITIAL_OPTIONS = Optional[
    Union[
        Dict[str, Any],
        str,
        Tuple[str, ...],
    ]
]


@dataclass
class OutlierDummy:
    """Container returned by ADAM.outlierdummy()."""

    outliers: Optional[NDArray]  # (n, m) 0/1 matrix, one column per outlier, or None
    id: NDArray  # 0-based indices of outlier observations
    statistic: NDArray  # [lower, upper] quantile bounds
    level: float
    type: str  # "rstandard" or "rstudent"


def _adam_refit_one_replicate(
    actuals: NDArray,
    idx_matrix: Union[NDArray, list[NDArray]],
    refit_cls_name: str,
    model_spec: Any,
    refit_kwargs: Dict[str, Any],
    k: int,
    include_model_kwarg: bool,
    i: int,
) -> Optional[NDArray]:
    """Refit one bootstrap replicate; return the coef vector or ``None``.

    Module-level so it can be pickled and sent to ``joblib.Parallel``
    workers. The class is resolved by name inside the worker (a local
    import) instead of pickling the class object itself — keeps the
    pickled payload small and avoids any class-identity issues across
    interpreters. ``i`` is the *last* positional argument so callers can
    bind everything else with :func:`functools.partial` and pass the
    integer index as the call site.
    """
    from smooth.adam_general.core.om import OM as _OM

    refit_cls = _OM if refit_cls_name == "OM" else ADAM
    y_boot = actuals[idx_matrix[i]]
    try:
        if include_model_kwarg:
            boot_model = refit_cls(model=model_spec, **refit_kwargs)
        else:
            boot_model = refit_cls(**refit_kwargs)
        boot_model.fit(y_boot)
        boot_coef = np.asarray(boot_model.coef, dtype=float)
    except Exception:
        return None
    if boot_coef.shape[0] != k or not np.all(np.isfinite(boot_coef)):
        return None
    return boot_coef


def _psd_correct(vcov: NDArray) -> NDArray:
    """Ensure ``vcov`` is positive semi-definite for the MVN sampler.

    Mirrors R's ``reapply.adam`` lines 96-115: if the smallest eigenvalue
    is negative, shift the diagonal by ``|min_eig| + 1e-10`` when the
    shift is small (PSD repair). When the eigenvalue is below ``-1`` the
    repair is too aggressive — fall back to the diagonal-only matrix
    which is always PSD.
    """
    vcov = np.asarray(vcov, dtype=float)
    if vcov.size == 0:
        return vcov
    try:
        eig_min = float(np.min(np.linalg.eigvalsh(vcov)))
    except np.linalg.LinAlgError:
        # ``Eigenvalues did not converge`` can fire on platform-specific
        # LAPACK iteration noise (seen on Windows wheels under
        # ``bounds="admissible"`` reapply) even when the input is a
        # well-formed covariance matrix. The diagonal-only matrix is
        # PSD by construction, so use it as the safe fallback for the
        # MVN sampler — same response as the very-negative-eigenvalue
        # branch below.
        warnings.warn(
            "Eigendecomposition of the covariance matrix did not converge; "
            "falling back to the diagonal-only matrix for MVN sampling.",
            stacklevel=3,
        )
        return np.diag(np.diag(vcov))
    if eig_min < 0:
        if eig_min > -1:
            warnings.warn(
                "The covariance matrix of parameters is not positive "
                "semi-definite; shifting the diagonal to repair it. "
                "Consider re-estimating the model with a different "
                "optimiser configuration.",
                stacklevel=3,
            )
            return vcov + (-eig_min + 1e-10) * np.eye(vcov.shape[0])
        warnings.warn(
            "The covariance matrix of parameters has a large negative "
            "eigenvalue; falling back to the diagonal-only matrix for "
            "MVN sampling. It is worth re-estimating the model.",
            stacklevel=3,
        )
        return np.diag(np.diag(vcov))
    return vcov


def _clip_ets_usual_smoothing(random_parameters: NDArray, idx: dict) -> None:
    """Closed-form ETS smoothing-parameter clipping (R/reapply.R:254-282).

    Enforces ``alpha ∈ [0, 1]``, ``beta ∈ [0, alpha]``, ``gamma ∈ [0,
    1 - alpha]`` (per seasonal index) and ``phi ∈ [0, 1]``. Mutates
    ``random_parameters`` in place. Separate from
    :func:`_clip_ets_multiplicative_states` so the admissible-bounds
    path can swap in eigen-based bounds while reusing the
    positivity check.
    """
    if "alpha" in idx:
        np.clip(
            random_parameters[:, idx["alpha"]],
            0.0,
            1.0,
            out=random_parameters[:, idx["alpha"]],
        )
    if "beta" in idx and "alpha" in idx:
        a = random_parameters[:, idx["alpha"]]
        b_col = idx["beta"]
        random_parameters[:, b_col] = np.clip(random_parameters[:, b_col], 0.0, a)
    gamma_cols = [v for k, v in idx.items() if k.startswith("gamma")]
    if gamma_cols and "alpha" in idx:
        a = random_parameters[:, idx["alpha"]]
        for c in gamma_cols:
            random_parameters[:, c] = np.clip(random_parameters[:, c], 0.0, 1.0 - a)
    if "phi" in idx:
        np.clip(
            random_parameters[:, idx["phi"]],
            0.0,
            1.0,
            out=random_parameters[:, idx["phi"]],
        )


def _clip_ets_multiplicative_states(
    random_parameters: NDArray, idx: dict, model_type: dict
) -> None:
    """Replace negative initial-state draws for multiplicative trend / season.

    Mirrors R/reapply.R:324-333. Required for **any** bounds mode
    because multiplicative states must be strictly positive — clipping
    them is a model-consistency check, not a bounds restriction.
    """
    if model_type.get("trend_type") == "M" and "trend" in idx:
        col = random_parameters[:, idx["trend"]]
        col[col < 0] = 1e-6
    if model_type.get("season_type") == "M":
        for k, v in idx.items():
            if k.startswith("seasonal"):
                col = random_parameters[:, v]
                col[col < 0] = 1e-6


def _clip_ets_usual(random_parameters: NDArray, idx: dict, model_type: dict) -> None:
    """Combined ``bounds="usual"`` ETS clipper (kept for backwards compat).

    Calls :func:`_clip_ets_usual_smoothing` then
    :func:`_clip_ets_multiplicative_states`. Mirrors R/reapply.R:254-333.
    """
    _clip_ets_usual_smoothing(random_parameters, idx)
    _clip_ets_multiplicative_states(random_parameters, idx, model_type)


def _clip_deltas(random_parameters: NDArray, idx: dict) -> None:
    """Clip xreg ``delta`` smoothing parameters to ``[0, 1]`` in place.

    Mirrors R/reapply.R:438-443. No-op when the model has no deltas.
    """
    for k, v in idx.items():
        if k.startswith("delta"):
            np.clip(
                random_parameters[:, v],
                0.0,
                1.0,
                out=random_parameters[:, v],
            )


def _level_bounds(levels, side, h):
    """Build per-horizon lower / upper quantile matrices for an interval.

    Mirrors R's ``levelLow`` / ``levelUp`` block in ``reforecast.adam``
    (R/reapply.R:1080-1098). Returns two ``(h, n_levels)`` numpy arrays.
    """
    n_levels = len(levels)
    level_arr = np.tile(np.asarray(levels, dtype=float), (h, 1))
    low = np.zeros((h, n_levels))
    high = np.zeros((h, n_levels))
    if side == "both":
        low[:] = (1.0 - level_arr) / 2.0
        high[:] = (1.0 + level_arr) / 2.0
    elif side == "upper":
        low[:] = 0.0
        high[:] = level_arr
    else:  # "lower"
        low[:] = 1.0 - level_arr
        high[:] = 1.0
    np.clip(low, 0.0, 1.0, out=low)
    np.clip(high, 0.0, 1.0, out=high)
    return low, high


def _column_names_for_levels(levels, side):
    """R-style bound-column labels for the lower / upper DataFrames."""
    if side == "both":
        lower = [f"Lower bound ({(1 - lvl) / 2 * 100:g}%)" for lvl in levels]
        upper = [f"Upper bound ({(1 + lvl) / 2 * 100:g}%)" for lvl in levels]
    elif side == "upper":
        lower = ["Lower 0%"] * len(levels)
        upper = [f"Upper bound ({lvl * 100:g}%)" for lvl in levels]
    else:  # "lower"
        lower = [f"Lower bound ({(1 - lvl) * 100:g}%)" for lvl in levels]
        upper = ["Upper 100%"] * len(levels)
    return lower, upper


def _forecast_index(fitted, h):
    """Build a ``h``-element pandas Index extrapolated from ``fitted.index``.

    Falls back to a ``RangeIndex`` if the in-sample index isn't a
    ``DatetimeIndex``-style monotonic step.
    """
    import pandas as _pd

    if not isinstance(fitted, _pd.Series) or len(fitted.index) < 2:
        return _pd.RangeIndex(start=0, stop=h)
    idx = fitted.index
    if isinstance(idx, _pd.DatetimeIndex):
        freq = idx.freq if idx.freq is not None else (idx[-1] - idx[-2])
        return _pd.date_range(start=idx[-1] + freq, periods=h, freq=freq)
    # Numeric index — extrapolate by the last step.
    try:
        step = idx[-1] - idx[-2]
        return _pd.Index([idx[-1] + step * (i + 1) for i in range(h)])
    except (TypeError, ValueError):
        return _pd.RangeIndex(start=len(idx), stop=len(idx) + h)


def _trim_mean(arr, trim):
    """Trimmed mean (R's ``mean(x, trim=...)``), NaN-safe.

    R trims the same proportion from each tail; ``scipy.stats.trim_mean``
    does the same, but ignores NaNs only when fed a filtered array.
    """
    a = np.asarray(arr, dtype=float).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    if trim <= 0:
        return float(a.mean())
    return float(scipy_stats.trim_mean(a, trim))


def _build_profiles_array(arr_vt_seed: NDArray, L: int, nsim: int) -> NDArray:
    """Stack the first-``L``-columns initial profile across ``nsim`` slices.

    Returns shape ``(n_components, L, nsim)``, F-ordered. The initial
    profile is the prefix of the fitted state matrix — R uses
    ``object$profileInitial`` for the same purpose (R/reapply.R:639).
    Built with ``np.zeros + assignment`` rather than ``np.repeat`` to
    match the layout the C++ kernel + carma expect across sequential
    reapply calls (see ``ADAM.reapply`` notes about heap corruption).
    """
    profile_seed = np.ascontiguousarray(arr_vt_seed[:, :L], dtype=np.float64)
    c = profile_seed.shape[0]
    profile = np.zeros((c, L, nsim), order="F")
    for i in range(nsim):
        profile[:, :, i] = profile_seed
    return profile


class ADAM:
    """
    ADAM: Augmented Dynamic Adaptive Model for Time Series Forecasting.

    ADAM is an advanced state-space modeling framework that combines
    **ETS** (Error, Trend, Seasonal) and **ARIMA** components into a unified
    Single Source of Error (SSOE) model. It provides a flexible, data-driven
    approach to time series forecasting with automatic model selection,
    parameter estimation, and prediction intervals.

    **Mathematical Form**:

    The ADAM model is specified in state-space form as:

    .. math::

        y_t &= o_t(w(v_{t-l}) + h(x_t, a_{t-1}) + r(v_{t-l})\\epsilon_t)

        v_t &= f(v_{t-l}, a_{t-1}) + g(v_{t-l}, a_{t-1}, x_t)\\epsilon_t

    where:

    - :math:`y_t`: Observed value at time t
    - :math:`o_t`: Occurrence indicator (Bernoulli variable for intermittent
      data, 1 otherwise)
    - :math:`v_t`: State vector (level, trend, seasonal, ARIMA components)
    - :math:`l`: Vector of lags
    - :math:`x_t`: Vector of exogenous variables
    - :math:`a_t`: Parameters for exogenous variables
    - :math:`w(\\cdot)`: Measurement function
    - :math:`r(\\cdot)`: Error function (additive or multiplicative)
    - :math:`f(\\cdot)`: Transition function (state evolution)
    - :math:`g(\\cdot)`: Persistence function (smoothing parameters)
    - :math:`\\epsilon_t`: Error term (various distributions supported)

    **Key Features**:

    1. **Unified Framework**: Seamlessly combines ETS and ARIMA in a single model
    2. **Multiple Seasonality**: Supports multiple seasonal periods
       (e.g., daily + weekly)
    3. **Automatic Selection**: Branch & Bound algorithm for efficient model selection
    4. **Flexible Distributions**: Normal, Laplace, Gamma, Log-Normal, and more
    5. **Intermittent Demand**: Built-in occurrence models for sparse data
    6. **External Regressors**: Include covariates with adaptive or fixed coefficients
    7. **Scikit-learn Compatible**: Familiar `.fit()` and `.predict()` API

    **Model Specification**:

    Models are specified using a string notation:

    - **ETS Models**: "ETS" where E=Error, T=Trend, S=Seasonal

      * E (Error): "A" (Additive), "M" (Multiplicative)
      * T (Trend): "N" (None), "A" (Additive), "Ad" (Additive Damped),
        "M" (Multiplicative), "Md" (Multiplicative Damped)
      * S (Seasonal): "N" (None), "A" (Additive), "M" (Multiplicative)

      Examples: "ANN" (Simple Exponential Smoothing), "AAN" (Holt's Linear),
      "AAA" (Holt-Winters Additive)

    - **Automatic Selection**:

      * "ZZZ": Select best model using Branch & Bound
      * "XXX": Select only additive components
      * "YYY": Select only multiplicative components
      * "ZXZ": Auto-select error and seasonal, additive trend only (**default**, safer)
      * "FFF": Full search across all 30 ETS model types

    - **ARIMA Models**: Specified via `ar_order`, `i_order`, `ma_order` parameters

      * Supports seasonal ARIMA: SARIMA(p,d,q)(P,D,Q)m
      * Multiple seasonality: e.g., hourly data with daily (24) and
        weekly (168) patterns

    **Supported Error Distributions**:

    - **Normal** (``distribution="dnorm"``): Default for additive errors
    - **Gamma** (``distribution="dgamma"``): Default for multiplicative errors
    - **Laplace** (``distribution="dlaplace"``): For heavy-tailed errors (MAE loss)
    - **Log-Normal** (``distribution="dlnorm"``): For positive-only data
    - **Inverse Gaussian** (``distribution="dinvgauss"``): For skewed positive data
    - **S distribution** (``distribution="ds"``): For extremely heavy-tailed data
    - **Generalized Normal** (``distribution="dgnorm"``): Flexible shape parameter

    Distribution is auto-selected based on loss function if ``distribution=None``.

    **Loss Functions**:

    - ``loss="likelihood"``: Maximum likelihood estimation (**default**)
    - ``loss="MSE"``: Mean Squared Error
    - ``loss="MAE"``: Mean Absolute Error
    - ``loss="HAM"``: Half-Absolute Moment
    - ``loss="MSEh"``: Multi-step MSE (h-step ahead)
    - ``loss="LASSO"``: L1 regularization for variable selection
    - ``loss="RIDGE"``: L2 regularization for shrinkage

    **Multistep loss functions**:

    - ``loss="MSEh"``: Mean Squared Error for specific h-steps ahead
    - ``loss="TMSE"``: Trace Mean Squared Error (sum of MSEh from 1 to h)
    - ``loss="GTMSE"``: Geometric Trace Mean Squared Error (sum of logs of MSEh
      from 1 to h)
    - ``loss="MSCE"``: Mean Squared Cumulative Error (sum of MSEh from 1 to h and
      covariances between them)
    - ``loss="GPL"``: Generalised Predictive Likelihood (minimum of the determinant of
      the covariance matrix of multistep errors)

    **Initialization Methods**:

    - ``initial="optimal"``: Optimize all initial states (default)
    - ``initial="backcasting"``: Use backcasting to initialize states
    - ``initial="two-stage"``: Backcast then optimize
    - ``initial="complete"``: Pure backcasting without optimization
    - ``initial={"level": 100, ...}``: Provide custom initial states

    **Workflow Example**:

    .. code-block:: python

        from smooth import ADAM
        import numpy as np

        # Generate sample data
        y = np.array([112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118] * 3)

        # Automatic model selection
        model = ADAM(model="ZZZ", lags=[12], ic="AICc")
        ADAM_fit = model.fit(y)
        print(ADAM_fit)

        # Generate 12-step ahead forecasts with intervals
        forecasts = model.predict(h=12, level=0.95)
        print(forecasts)

        # Access fitted parameters
        print(f"Model: {model.model_name}")
        print(f"Alpha: {model.persistence_level_:.3f}")
        print(f"AICc: {model.aicc}")

    **Attributes (After Fitting)**:

    The model stores fitted results as attributes with trailing underscores
    (scikit-learn convention):

    - ``persistence_level_``: Level smoothing parameter (α)
    - ``persistence_trend_``: Trend smoothing parameter (β)
    - ``persistence_seasonal_``: Seasonal smoothing parameter(s) (γ)
    - ``phi_``: Damping parameter (φ)
    - ``initial_states_``: Estimated initial states
    - ``arma_parameters_``: AR/MA coefficients (if ARIMA)

    Additional fitted attributes:

    - ``model_name``: Full model specification string
    - ``coef``: Estimated parameter vector B
    - ``states``: State matrix over time
    - ``persistence_vector``: Named persistence parameters
    - ``transition``: Transition matrix

    **Performance Considerations**:

    - **Small Data** (T < 100): Use "backcasting" initialization,
      it's faster
    - **Large Data** (T > 1000): "optimal" initialization works well
    - **Multiple Seasonality**: Can be slow; consider simpler models
      first
    - **Model Selection**: "ZZZ" with Branch & Bound is much faster than
      "FFF" exhaustive search

    **Common Use Cases**:

    1. **Automatic Forecasting**: ``ADAM(model="ZXZ", lags=[12])`` -
       Let the model choose
    2. **Intermittent Demand**: ``ADAM(model="ANN", occurrence="auto")`` -
       For sparse data
    3. **External Regressors**: ``ADAM(model="AAN").fit(y, X=regressors)`` -
       Include covariates
    4. **Multiple Seasonality**: ``ADAM(model="AAA", lags=[24, 168])`` -
       Hourly data with daily/weekly patterns
    5. **ARIMA**: ``ADAM(model="NNN", ar_order=1, i_order=1, ma_order=1)`` -
       Pure ARIMA(1,1,1)
    6. **Custom Model**: ``ADAM(model="MAM", persistence={"alpha": 0.3})`` -
       Fix some parameters

    **Comparison to R's smooth::adam**:

    This Python implementation is a direct translation of the R smooth
    package's ``adam()`` function, maintaining mathematical equivalence
    while adapting to scikit-learn conventions:

    - R: ``adam(data, model="ZZZ", h=10)`` →
      Python: ``ADAM(model="ZZZ").fit(y).predict(h=10)``
    - R: ``persistence=list(alpha=0.3)`` → Python: ``persistence={"alpha": 0.3}``
    - R: ``orders=list(ar=c(1,1))`` → Python: ``ar_order=[1, 1]``

    **References**:

    - Svetunkov, I. (2023). Forecasting and Analytics with the Augmented Dynamic
      Adaptive Model. https://openforecast.org/adam/
    - Hyndman, R.J., et al. (2008). "Forecasting with Exponential Smoothing"
    - Svetunkov, I. & Boylan, J.E. (2017). "State-space ARIMA for
      supply-chain forecasting"
    - Svetunkov, I. & Kourentzes, N. & Killick, R. (2023). "Multi-step
      estimators and shrinkage effect in time series models".
      DOI: 10.1007/s00180-023-01377-x

    See Also
    --------
    adam.fit : Fit the ADAM model to data
    adam.predict : Generate point forecasts
    adam.predict_intervals : Generate prediction intervals
    selector : Automatic model selection function
    estimator : Parameter estimation function
    forecaster : Forecasting function
    print : Print the outputs of the ADAM class

    Examples
    --------
    Simple exponential smoothing::

        >>> from smooth import ADAM
        >>> import numpy as np
        >>> y = np.array([112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118])
        >>> model = ADAM(model="ANN", lags=[1])
        >>> model.fit(y)
        >>> forecasts = model.predict(h=6)
        >>> print(forecasts)

    Automatic model selection with multiple seasonality::

        >>> model = ADAM(model="ZZZ", lags=[12], ic="AICc")
        >>> model.fit(y)
        >>> print(f"Selected model: {model.model_name}")

    SARIMA(1,1,1)(1,1,1)₁₂::

        >>> model = ADAM(
        ...     model="NNN",  # Pure ARIMA
        ...     ar_order=[1, 1],
        ...     i_order=[1, 1],
        ...     ma_order=[1, 1],
        ...     lags=[1, 12]
        ... )
        >>> model.fit(y)
        >>> forecasts = model.predict(h=12)

    With external regressors::

        >>> X = np.random.randn(len(y), 2)  # Two regressors
        >>> X_future = np.random.randn(6, 2)  # Regressors for forecast period
        >>> model = ADAM(model="AAN", regressors="use")
        >>> model.fit(y, X=X)
        >>> forecasts = model.predict(h=6, X=X_future)

    Fix some parameters, estimate others::

        >>> model = ADAM(
        ...     model="AAA",
        ...     lags=[12],
        ...     persistence={"alpha": 0.3},  # Fix alpha, estimate beta and gamma
        ...     initial="backcasting"
        ... )
        >>> model.fit(y)

    Intermittent demand forecasting::

        >>> sparse_data = np.array([0, 0, 15, 0, 0, 23, 0, 0, 0, 18, 0, 0])
        >>> model = ADAM(model="ANN", occurrence="auto")
        >>> model.fit(sparse_data)
        >>> forecasts = model.predict(h=6)  # Accounts for zero-demand probability
    """

    def __init__(
        self,
        model: Union[str, List[str]] = "ZXZ",
        lags: Optional[Union[List[int], NDArray]] = None,
        # ARIMA specific parameters
        ar_order: Union[int, List[int]] = 0,
        i_order: Union[int, List[int]] = 0,
        ma_order: Union[int, List[int]] = 0,
        orders: Optional[Dict[str, Any]] = None,
        arima_select: bool = False,
        # end of ARIMA specific parameters
        constant: Union[bool, float] = False,
        regressors: Literal["use", "select", "adapt"] = "use",
        # ``str`` accepted alongside DISTRIBUTION_OPTIONS for wrapper/selector
        # call sites that hold runtime-validated strings.
        distribution: Optional[Union[DISTRIBUTION_OPTIONS, str]] = None,
        loss: LOSS_OPTIONS = "likelihood",
        loss_horizon: Optional[int] = None,
        # outlier detection
        outliers: Literal["ignore", "use", "select"] = "ignore",
        outliers_level: float = 0.99,
        # end of outlier detection
        ic: Literal["AIC", "AICc", "BIC", "BICc"] = "AICc",
        bounds: Literal["usual", "admissible", "none"] = "usual",
        # ``str`` (not just OCCURRENCE_OPTIONS) so OM/wrappers can forward
        # runtime-validated strings and OM may override it as a property.
        occurrence: Union[OCCURRENCE_OPTIONS, str] = "none",
        # ---- These are the estimated parameters that we can choose to fix ----
        # Dictionary of terms e.g. {"alpha": 0.5, "beta": 0.5}
        persistence: Optional[Union[Dict[str, float], float]] = None,
        phi: Optional[float] = None,
        initial: INITIAL_OPTIONS = "backcasting",
        # Number of iterations for backcasting (default 2 for backcasting, 1 otherwise)
        n_iterations: Optional[int] = None,
        # TODO: enforce the structure of this
        arma: Optional[Dict[str, Any]] = None,
        # ----- End of parameters----
        verbose: int = 0,
        # Parameters moved from fit
        h: Optional[int] = None,
        holdout: bool = False,
        fast: bool = False,
        lambda_param: Optional[float] = None,
        # Profile parameters
        profiles_recent_provided: bool = False,
        profiles_recent_table: Optional[Any] = None,
        # Fisher information matrix: when True, the observed FI is computed at the
        # estimated parameters (mirrors R's adam(..., FI=TRUE)). step_size is the
        # absolute finite-difference step (default .Machine$double.eps^(1/4)).
        fi: bool = False,
        step_size: Optional[float] = None,
        # initial values for optimization parameters:
        nlopt_initial: Optional[Dict[str, Any]] = None,
        nlopt_upper: Optional[Dict[str, Any]] = None,
        nlopt_lower: Optional[Dict[str, Any]] = None,
        nlopt_kwargs: Optional[Dict[str, Any]] = None,
        # specific to losses or distributions
        reg_lambda: Optional[float] = None,
        gnorm_shape: Optional[float] = None,
        smoother: Literal["lowess", "ma", "global"] = SMOOTHER_DEFAULT,
        ets: Literal["conventional", "adam"] = "conventional",
        **kwargs,
    ) -> None:
        """
        Initialize the ADAM model with specified parameters.

        Parameters
        ----------
        model : Union[str, List[str]], default="ZXZ"
            Model specification string (e.g., "ANN" for ETS) or
            list of model strings.
        lags : Optional[Union[int, List[int]]], default=None
            Seasonal period(s). Can be a single integer or a list of integers.
            E.g., ``lags=12`` is equivalent to ``lags=[12]``.
        ar_order : Union[int, List[int]], default=0
            Autoregressive order(s) for ARIMA components.
        i_order : Union[int, List[int]], default=0
            Integration order(s) for ARIMA components.
        ma_order : Union[int, List[int]], default=0
            Moving average order(s) for ARIMA components.
        orders : Optional[Dict[str, Any]], default=None
            Dict-style alternative to ``ar_order``/``i_order``/``ma_order``.
            A dict with keys ``"ar"``, ``"i"``, ``"ma"`` (each an int or list
            of ints) and optionally ``"select"`` (bool).  Example::

                orders={"ar": [1, 1], "i": [1, 1], "ma": [2, 2]}

            If ``ar_order``, ``i_order``, or ``ma_order`` are non-zero they
            take priority over ``orders``.
        arima_select : bool, default=False
            Whether to perform automatic ARIMA order selection.
        constant : Union[bool, float], default=False
            Whether to include a constant (intercept/drift) term.

            - ``True``: estimate the constant as a free parameter.
            - ``False``: no constant.
            - A numeric value: fix the constant at that value (not estimated).

            The model name reflects the role of the constant:

            - **"with drift"** — when the model is ETS *or* ARIMA with any
              integration order > 0 (e.g. ``ETS(ANN) with drift``,
              ``ARIMA(1,1,1) with drift``).
            - **"with constant"** — when the model is a pure non-integrated
              ARIMA (all ``i_order = 0``), e.g. ``ARIMA(1,0,1) with constant``.

            The fitted value is accessible via ``model.constant_value``.
        regressors : Literal["use", "select", "adapt"], default="use"
            How to handle external regressors.
        distribution : Optional[DISTRIBUTION_OPTIONS], default=None
            Error distribution. If None, it is selected automatically based
            on the loss function.
        loss : LOSS_OPTIONS, default="likelihood"
            Loss function for parameter estimation.
        loss_horizon : Optional[int], default=None
            Number of steps for multi-step loss functions (e.g., MSEh).
        outliers : Literal["ignore", "use", "select"], default="ignore"
            Outlier handling: ``"ignore"`` skips detection; ``"use"`` detects
            outliers and includes their dummies as fixed regressors; ``"select"``
            also expands each dummy with lag/lead columns and lets the regressor
            selection mechanism choose which matter.
        outliers_level : float, default=0.99
            Confidence level for outlier detection.
        ic : Literal["AIC", "AICc", "BIC", "BICc"], default="AICc"
            Information criterion for model selection.
        bounds : Literal["usual", "admissible", "none"], default="usual"
            Parameter bounds specification during optimization.
        occurrence : OCCURRENCE_OPTIONS, default="none"
            Occurrence model type for intermittent data.
        persistence : Optional[Dict[str, float]], default=None
            Fixed persistence parameters (e.g., {"alpha": 0.5, "beta": 0.5}).
            If None, parameters are estimated.
        phi : Optional[float], default="backcasting"
            Fixed damping parameter for damped trend models. If None,
            estimated if applicable.
        initial : INITIAL_OPTIONS, default=None
            Method for initializing states or fixed initial states. Can be a string
            (e.g., 'optimal', 'backcasting'), a dictionary of initial state values,
            or a tuple of state names to initialize.
        arma : Optional[Dict[str, Any]], default=None
            Fixed ARMA parameters specification. If None, estimated if applicable.
        verbose : int, default=0
            Verbosity level (0=silent, higher values indicate more output).
        h : Optional[int], default=None
            Forecast horizon. If None during initialization, can be set in `predict`.
        holdout : bool, default=False
            Whether to use a holdout sample for validation during the fit process.
        fast : bool, default=False
            Whether to use faster, possibly less accurate, estimation methods.
        lambda_param : Optional[float], default=None
            Lambda parameter for Box-Cox transformation or regularization.
        profiles_recent_provided : bool, default=False
            Whether recent profiles (e.g., for exogenous variables) are provided.
        profiles_recent_table : Optional[Any], default=None
            Table containing recent profiles data.
        nlopt_initial : Optional[Dict[str, Any]], default=None
            Initial values for optimization parameters for NLopt solver.
        nlopt_upper : Optional[Dict[str, Any]], default=None
            Upper bounds for optimization parameters for NLopt solver.
        nlopt_lower : Optional[Dict[str, Any]], default=None
            Lower bounds for optimization parameters for NLopt solver.
        nlopt_kwargs : Optional[Dict[str, Any]], default=None
            Additional keyword arguments for optimization. Supported keys:

            - ``print_level`` (int): Verbosity level for optimization progress
              (default=0). When >0, prints parameter vector B and cost function
              value on every iteration.
              Output format: ``Iter N: B=[val1, val2, ...] -> CF=value``
            - ``xtol_rel`` (float): Relative tolerance on optimization parameters
              (default=1e-6). Optimization stops when parameter changes are
              smaller than xtol_rel * |params|.
            - ``xtol_abs`` (float): Absolute tolerance on optimization parameters
              (default=1e-8). Optimization stops when parameter changes are
              smaller than xtol_abs.
            - ``ftol_rel`` (float): Relative tolerance on function value
              (default=1e-8). Optimization stops when CF changes are smaller
              than ftol_rel * |CF|.
            - ``ftol_abs`` (float): Absolute tolerance on function value (default=0).
              Optimization stops when CF changes are smaller than ftol_abs.
            - ``algorithm`` (str): NLopt algorithm name
              (default="NLOPT_LN_NELDERMEAD"). Common alternatives:
              "NLOPT_LN_SBPLX" (Subplex), "NLOPT_LN_COBYLA" (COBYLA),
              "NLOPT_LN_BOBYQA" (BOBYQA). Use "LN_" prefix for derivative-free
              algorithms.

            Example::

                model = ADAM(model="AAN", nlopt_kwargs={
                    "print_level": 1,
                    "xtol_rel": 1e-8,
                    "algorithm": "NLOPT_LN_SBPLX"
                })
        reg_lambda : Optional[float], default=None
            Regularization parameter specifically for LASSO/RIDGE losses.
        gnorm_shape : Optional[float], default=None
            Shape parameter 's' for the generalized normal distribution.
        smoother : Literal["lowess", "ma", "global"], default="global"
            Smoother type for time series decomposition in initial state estimation.

            - "lowess": Uses LOWESS (Locally Weighted Scatterplot Smoothing) for both
              trend and seasonal pattern extraction. This is the default.
            - "ma": Uses simple moving average for both trend and seasonality.
            - "global": Uses lowess for trend and "ma" (moving average) for seasonality.
              Provides robust trend estimation while avoiding lowess instability for
              small seasonal samples.
        """
        # Start time for measuring computation duration
        self._start_time = time.time()

        # Store initialization parameters
        self.model = model
        self.lags = lags
        self.ar_order = ar_order
        self.i_order = i_order
        self.ma_order = ma_order
        self._init_orders = orders
        self.arima_select = arima_select
        self.constant = constant
        self.regressors = regressors
        self.distribution = distribution
        self.loss = loss
        self.loss_horizon = loss_horizon
        self.outliers = outliers
        self.outliers_level = outliers_level
        self.ic = ic
        self.bounds = bounds
        self.occurrence = occurrence
        self.persistence = persistence
        self.phi = phi
        self.initial = initial
        self.n_iterations = n_iterations
        self.arma = arma
        self.verbose = verbose
        self.nlopt_initial = nlopt_initial
        self.nlopt_upper = nlopt_upper
        self.nlopt_lower = nlopt_lower
        self.nlopt_kwargs = nlopt_kwargs
        self.reg_lambda = reg_lambda
        self.gnorm_shape = gnorm_shape
        self.smoother = smoother
        if ets not in ("conventional", "adam"):
            raise ValueError(f"Invalid ets: {ets!r}. Must be 'conventional' or 'adam'.")
        self.ets = ets

        # Store parameters that were moved from fit
        self.h = h
        self.holdout = holdout
        self.fast = fast
        # Handle 'lambda' from kwargs (since 'lambda' is a reserved word in Python)
        # Users can pass either lambda_param=0.5 or **{'lambda': 0.5}
        if "lambda" in kwargs:
            self.lambda_param = kwargs["lambda"]
        else:
            self.lambda_param = lambda_param

        # Handle 'print_level' from kwargs for convenience
        # Users can pass print_level=1 directly or via nlopt_kwargs={'print_level': 1}
        if "print_level" in kwargs:
            if self.nlopt_kwargs is None:
                self.nlopt_kwargs = {}
            self.nlopt_kwargs["print_level"] = kwargs["print_level"]

        # Store profile parameters
        self.profiles_recent_provided = profiles_recent_provided
        self.profiles_recent_table = profiles_recent_table

        # Fisher Information settings
        self.fi = fi
        self.step_size = step_size

    def fit(
        self,
        y: NDArray,
        X: Optional[NDArray] = None,
    ):
        """
        Fit the ADAM model to time series data.

        This method estimates the model parameters, selects the best model
        (if automatic selection is enabled), and prepares the model for
        forecasting. It implements the complete ADAM estimation pipeline:
        parameter checking, model architecture creation, state-space matrix
        construction, parameter optimization, and model preparation.

        **Estimation Process**:

        1. **Parameter Validation**: Check all inputs via ``parameters_checker()``
        2. **Model Selection** (if ``model_do="select"``): Use Branch & Bound algorithm
        3. **Model Architecture**: Define components and lags via ``architector()``
        4. **Matrix Creation**: Build state-space matrices via ``creator()``
        5. **Parameter Estimation**: Optimize using NLopt via ``estimator()``
        6. **Model Preparation**: Compute fitted values and final states
           via ``preparator()``

        After fitting, the model stores all results as attributes:

        - Estimated parameters (``persistence_level_``, ``phi_``, etc.)
        - State-space matrices (``states``, ``transition``, ``measurement``)
        - Fitted values and residuals (``fitted``, ``residuals``)
        - Information criteria (``aic``, ``aicc``, ``bic``, ``bicc``)
        - Model specification (``model_name``, ``model_type``)

        Parameters
        ----------
        y : array-like, shape (T,)
            Time series data to fit. Can be:

            - ``numpy.ndarray``: Shape (T,) for univariate time series
            - ``pandas.Series``: Will use index for time information if DatetimeIndex

            Data requirements:

            - **Minimum length**: Depends on model complexity. Rule of thumb:
              T ≥ 3 × (number of parameters)
            - **Multiplicative models**: Require strictly positive data (y > 0)
            - **Missing values**: Currently not supported in Python version
            - **Frequency**: Auto-detected from pandas Series with DatetimeIndex

        X : array-like, shape (T, n_features), optional
            External regressors (explanatory variables). If provided:

            - Must have same length as ``y`` (T observations)
            - Each column is a separate regressor
            - Used only if ``regressors`` parameter was set in ``__init__``
            - Can be adaptive (with persistence) or fixed coefficients

            Example::

                >>> X = np.column_stack([trend, holidays, temperature])
                >>> model.fit(y, X=X)

        Returns
        -------
        self : ADAM
            The fitted model instance with populated attributes:

            **Fitted Parameters** (scikit-learn style with trailing underscores):

            - ``persistence_level_``: α (level smoothing), range [0, 1]
            - ``persistence_trend_``: β (trend smoothing), range [0, α]
            - ``persistence_seasonal_``: γ (seasonal smoothing), list if multiple
            - ``phi_``: Damping parameter, range [0, 1]
            - ``arma_parameters_``: AR/MA coefficients (if ARIMA)
            - ``initial_states_``: Initial state values

            **Model Components** (via properties):

            - ``model_name``: Full model specification string
            - ``coef`` / ``B``: Estimated parameter vector
            - ``states``: State matrix over time
            - ``transition``: Transition matrix
            - ``measurement``: Measurement matrix
            - ``aic``, ``aicc``, ``bic``, ``bicc``: Information criteria

        Raises
        ------
        ValueError
            If data validation fails:

            - y contains NaN values
            - y has insufficient length for model complexity
            - Multiplicative model specified for non-positive data
            - X and y have mismatched lengths

        RuntimeError
            If optimization fails to converge. Check:

            - Model specification is appropriate for data
            - Initial values are reasonable (try different ``initial`` method)
            - Bounds are not too restrictive

        Notes
        -----
        **Optimization Algorithm**:

        Uses NLopt's Nelder-Mead simplex algorithm by default (derivative-free, robust).
        For large models or difficult optimization, consider:

        - Changing ``initial`` method
        - Adjusting bounds (``bounds="usual"`` vs ``bounds="admissible"``)
        - Providing custom starting values via ``nlopt_initial`` parameter

        **Computational Complexity**:

        Fitting time depends on:

        - Sample size T: O(T) per function evaluation
        - Number of parameters k: ~40k function evaluations
        - Model selection: Estimates ~10-15 models for "ZZZ"

        Typical fitting times:

        - T=100, simple ETS: ~0.1 seconds
        - T=1000, ETS with 2 seasonalities: ~1-2 seconds
        - T=1000, automatic selection: ~10-20 seconds

        **Model Selection Details**:

        When ``model_do="select"``:

        1. Branch & Bound explores model space efficiently
        2. Each candidate model is fully estimated
        3. Best model selected based on ``ic`` criterion
        4. Selected model is re-estimated with full optimization

        To see which models were tested::

            >>> model.fit(y)
            >>> print(model.ic_selection)  # Dict of model names -> IC values

        **Holdout Validation**:

        If ``holdout=True`` in ``__init__``:

        - Last ``h`` observations are withheld for validation
        - Model estimated on first T-h observations
        - Can use holdout for out-of-sample accuracy assessment

        **Two-Stage Initialization**:

        When ``initial="two-stage"``:

        1. **Stage 1**: Quick backcasting estimation for initial states
        2. **Stage 2**: Refined optimization starting from stage 1 results

        Often provides better results than pure ``initial="optimal"`` for
        complex models.

        **Memory Usage**:

        - State matrix: O(n_components × (T + max_lag)) floats
        - Modest for typical models (~1-10 MB)
        - Multiple seasonality can increase memory usage

        See Also
        --------
        predict : Generate forecasts from fitted model
        predict_intervals : Generate prediction intervals
        estimator : Underlying estimation function
        parameters_checker : Input validation function

        Examples
        --------
        Basic fitting::

            >>> from smooth import ADAM
            >>> import numpy as np
            >>> y = np.array([
            ...     112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118
            ... ])
            >>> model = ADAM(model="ANN", lags=[1])
            >>> model.fit(y)
            >>> print(f"Alpha: {model.persistence_level_:.3f}")

        With external regressors::

            >>> X = np.random.randn(len(y), 2)
            >>> model = ADAM(model="AAN", regressors="use")
            >>> model.fit(y, X=X)
            >>> print(f"Coefficients: {model.coef}")

        Automatic model selection::

            >>> model = ADAM(model="ZZZ", lags=[12], ic="AICc")
            >>> model.fit(y)
            >>> print(f"Selected: {model.model_name}")
            >>> print(f"AICc: {model.aicc}")

        Access fitted values and residuals::

            >>> model.fit(y)
            >>> fitted = model.fitted
            >>> residuals = model.residuals
            >>> print(f"RMSE: {np.sqrt(np.mean(residuals**2)):.3f}")

        Using pandas Series with datetime index::

            >>> import pandas as pd
            >>> dates = pd.date_range(
            ...     '2020-01-01', periods=len(y), freq='M'
            ... )
            >>> y_series = pd.Series(y, index=dates)
            >>> model.fit(y_series)
            >>> # Frequency auto-detected from index
        """
        # Store fit parameters - these are now set in __init__
        # No need to call _setup_parameters as those parameters are now
        # instance attributes

        # Check parameters and prepare data
        self._check_parameters(y, X)

        # Pure-regression early exit: when model="NNN" with regressors but
        # no ETS/ARIMA, the parameter checker has wrapped a greybox.ALM fit
        # as ``_alm_model`` — populate from it and return.
        if getattr(self, "_alm_model", None) is not None:
            self._populate_from_alm(y, X)
            return self

        # Fit the occurrence model first if requested. The fitted occurrence
        # probability p_fitted is held constant while the demand-sizes model
        # is estimated.
        self._om_model = None
        if self._occurrence.get("occurrence_model"):
            ot_logical = self._observations["ot_logical"]
            self._observations["obs_zero"] = int(np.sum(~ot_logical))
            self._om_model = self._fit_occurrence_model(y)
            self._occurrence["p_fitted"] = self._om_model.fitted
            self._occurrence["oes_model"] = self._occurrence["occurrence"]

        # Execute model estimation or selection based on model_do
        if self._model_type["model_do"] == "estimate":
            self._execute_estimation()
        elif self._model_type["model_do"] == "select":
            # get the best model
            self._execute_selection()
            # Execute estimation for the selected model
            # Note: estimator() handles two-stage initialization internally,
            # so all models in the pool use consistent initialization
            self._execute_estimation(estimation=True)

        elif self._model_type["model_do"] == "combine":
            # Store original model specification for display (e.g., "CCC")
            self._original_model_spec = self._model_type.get("model", self.model)
            # Run model selection first to get all candidate models
            self._execute_selection()
            # Combine models using IC weights
            self._execute_combination()
            # Execute estimation for the best model (for state-space matrices)
            self._execute_estimation(estimation=True)
        else:
            model_do = self._model_type["model_do"]
            warnings.warn(
                f"Unknown model_do value: {model_do}. "
                "Expected one of: 'estimate', 'select', 'combine'"
            )

        # Prepare final results and format output data
        self._prepare_results()

        # Scale the demand-sizes fitted values by the occurrence probability
        # so the resulting ``fitted`` series is on the original (mixed) scale.
        if self._om_model is not None:
            import pandas as pd

            p = self._occurrence["p_fitted"]
            yf = self._prepared["y_fitted"]
            if isinstance(yf, pd.Series):
                yf = yf * p
            else:
                yf = np.asarray(yf) * np.asarray(p)
            self._prepared["y_fitted"] = yf
            self._prepared["residuals"] = np.asarray(
                self._observations["y_in_sample"]
            ) - np.asarray(yf)
            # Include occurrence model params in the total count
            self._adam_estimated["n_param_estimated"] += self._om_model.nparam

        # Store fitted parameters with trailing underscores
        self._set_fitted_attributes()

        # Auto-forecast if h > 0 (before _config consolidation so holdout is accessible)
        self._auto_predict()

        # Compute elapsed time
        self.time_elapsed_ = time.time() - self._start_time

        # Outlier detection and refitting
        # "ignore" → skip; "use" → include dummies as fixed regressors;
        # "select" → expand dummies with lag/lead and let selection prune them.
        _outliers = self.outliers
        if _outliers in ("use", "select"):
            _level = self.outliers_level
            od = self.outlierdummy(level=_level)
            if len(od.id) > 0:
                dummies = od.outliers
                assert dummies is not None  # non-empty od.id guarantees outliers
                D = (
                    self._expand_outlier_dummies(dummies)
                    if _outliers == "select"
                    else dummies
                )
                h_eff = len(y) - self.nobs
                if h_eff > 0:
                    D = np.vstack([D, np.zeros((h_eff, D.shape[1]))])
                X_new = np.hstack([X, D]) if X is not None else D
                self.outliers = "ignore"
                self.regressors = "select" if _outliers == "select" else "use"
                self.fit(y, X_new)
                self._config["outliers"] = _outliers
                return self  # _config already built by the recursive fit()

        # Consolidate init params into _config and remove individual attributes
        self._config: Dict[str, Any] = {
            "lags": self.lags,
            "ar_order": self.ar_order,
            "i_order": self.i_order,
            "ma_order": self.ma_order,
            "orders": self._init_orders,
            "arima_select": self.arima_select,
            "constant": self.constant,
            "regressors": self.regressors,
            "distribution": self.distribution,
            "loss": self.loss,
            "loss_horizon": self.loss_horizon,
            "outliers": self.outliers,
            "outliers_level": self.outliers_level,
            "ic": self.ic,
            "bounds": self.bounds,
            "occurrence": self.occurrence,
            "persistence": self.persistence,
            "phi": self.phi,
            "initial": self.initial,
            "n_iterations": self.n_iterations,
            "arma": self.arma,
            "reg_lambda": self.reg_lambda,
            "gnorm_shape": self.gnorm_shape,
            "lambda_param": self.lambda_param,
            "fast": self.fast,
            "holdout": self.holdout,
        }
        for key in self._config:
            try:
                delattr(self, key)
            except AttributeError:
                pass

        return self

    def _set_fitted_attributes(self):
        """
        Set fitted parameters as attributes with trailing underscores.

        This follows scikit-learn conventions for fitted attributes.
        """
        # Distribution-specific extra parameter — mirrors R's ``m$other``.
        # For ``dgnorm`` / ``dlgnorm`` the estimated shape lives on
        # ``self.gnorm_shape`` (set in ``_execute_estimation`` at the line
        # ``self.gnorm_shape = float(abs(self._adam_estimated["B"][-1]))``).
        # Exposing it under the same key as R lets ``_format_distribution``
        # render ``"Generalised Normal with shape=2.4791"`` in ``print(m)``
        # and ``m.summary()`` instead of ``shape=?``.
        dist = self._general.get("distribution", "dnorm") if self._general else "dnorm"
        gnorm_shape = getattr(self, "gnorm_shape", None)
        if dist in ("dgnorm", "dlgnorm") and gnorm_shape is not None:
            self.other = {"shape": float(gnorm_shape)}

        # Set persistence parameters (pre-estimation values for provided params)
        if self._persistence:
            if "persistence_level" in self._persistence:
                self.persistence_level_ = self._persistence["persistence_level"]
            if "persistence_trend" in self._persistence:
                self.persistence_trend_ = self._persistence["persistence_trend"]
            if "persistence_seasonal" in self._persistence:
                self.persistence_seasonal_ = self._persistence["persistence_seasonal"]
            if "persistence_xreg" in self._persistence:
                self.persistence_xreg_ = self._persistence["persistence_xreg"]

        # For combined models, preserve original model specification with ETS prefix
        if getattr(self, "_is_combined", False):
            self.model = f"ETS({self._original_model_spec})"
            return

        # Update self.model with the selected/estimated model name
        if self._model_type:
            if hasattr(self, "_best_model") and self._best_model:
                ets_str = self._best_model
            else:
                e = self._model_type.get("error_type", "")
                t = self._model_type.get("trend_type", "")
                s = self._model_type.get("season_type", "")
                damped = self._model_type.get("damped", False)
                if damped and t not in ["N", ""]:
                    t = t + "d"
                ets_str = e + t + s if (e or t or s) else self.model

            model_parts = []
            is_ets = self._model_type.get("ets_model", False)
            has_xreg = self._explanatory.get("xreg_model", False)
            if is_ets:
                ets_prefix = "ETSX" if has_xreg else "ETS"
                model_parts.append(f"{ets_prefix}({ets_str})")

            is_arima = self._model_type.get("arima_model", False)
            if is_arima and self._arima:
                ar_orders = self._arima.get("ar_orders", [0]) or [0]
                i_orders = self._arima.get("i_orders", [0]) or [0]
                ma_orders = self._arima.get("ma_orders", [0]) or [0]
                lags = self._lags_model.get("lags_original", [1]) or [1]
                has_xreg_arima = (
                    self._explanatory.get("xreg_model", False) and not is_ets
                )
                seasonal_have_orders = any(
                    (ar_orders[j] if j < len(ar_orders) else 0) != 0
                    or (i_orders[j] if j < len(i_orders) else 0) != 0
                    or (ma_orders[j] if j < len(ma_orders) else 0) != 0
                    for j, lag in enumerate(lags)
                    if lag > 1
                )
                if all(lag == 1 for lag in lags) or not seasonal_have_orders:
                    prefix = "ARIMAX" if has_xreg_arima else "ARIMA"
                    model_parts.append(
                        f"{prefix}({ar_orders[0]},{i_orders[0]},{ma_orders[0]})"
                    )
                else:
                    prefix = "SARIMAX" if has_xreg_arima else "SARIMA"
                    arima_str = prefix
                    for j, lag in enumerate(lags):
                        p = ar_orders[j] if j < len(ar_orders) else 0
                        d = i_orders[j] if j < len(i_orders) else 0
                        q = ma_orders[j] if j < len(ma_orders) else 0
                        if p == 0 and d == 0 and q == 0:
                            continue
                        arima_str += f"({p},{d},{q})[{lag}]"
                    model_parts.append(arima_str)

            if model_parts:
                self.model = "+".join(model_parts)
            else:
                self.model = ets_str

            if self._constant and self._constant.get("constant_required", False):
                i_orders = (self._arima or {}).get("i_orders") or []
                has_drift = is_ets or any(d != 0 for d in i_orders)
                self.model += " with " + ("drift" if has_drift else "constant")

    # =========================================================================
    # Extraction properties — convenience accessors over the fitted state.
    # =========================================================================

    def _check_is_fitted(self):
        """Check if model has been fitted."""
        if not hasattr(self, "_prepared") or self._prepared is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

    # =========================================================================
    # R-Equivalent Properties
    # =========================================================================

    @property
    def states(self) -> NDArray:
        """
        State matrix containing component values over time (R: $states).

        The state matrix stores the evolution of all model components including
        level, trend (if present), seasonal components (if present), and ARIMA
        states (if present). Each column represents a different state component,
        and each row represents a time point.

        Returns
        -------
        NDArray
            2D array of shape (n_states, obs_in_sample + 1). Columns represent
            level, trend (if model has trend), and seasonal components (if model
            has seasonality, one column per lag).

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAA", lags=12)
        >>> model.fit(y)
        >>> states = model.states
        >>> level = states[0, :]  # Level component over time
        """
        self._check_is_fitted()
        return self._prepared["states"]

    @property
    def persistence_vector(self) -> Dict[str, Any]:
        """
        Estimated smoothing parameters (R: $persistence).

        Returns a dictionary containing the smoothing/persistence parameters
        that control how quickly the model adapts to new observations. Higher
        values mean faster adaptation (more weight on recent observations).

        Returns
        -------
        Dict[str, Any]
            Dictionary with keys ``persistence_level`` (alpha, 0-1),
            ``persistence_trend`` (beta, 0-alpha, only if model has trend),
            and ``persistence_seasonal`` (gamma, 0 to 1-alpha, only if model
            has seasonality).

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        See Also
        --------
        phi_ : Damping parameter for trend

        Examples
        --------
        >>> model = ADAM(model="AAA", lags=12)
        >>> model.fit(y)
        >>> alpha = model.persistence_vector['persistence_level']
        >>> gamma = model.persistence_vector['persistence_seasonal']
        """
        self._check_is_fitted()
        return self._prepared.get("persistence", {})

    @property
    def phi_(self) -> Optional[float]:
        """
        Damping parameter for trend component (R: $phi).

        The damping parameter (phi) controls how quickly the trend dampens
        toward zero over the forecast horizon. A value of 1.0 means no damping
        (linear trend), while values less than 1.0 cause the trend to
        gradually flatten.

        Returns
        -------
        Optional[float]
            Damping parameter value between 0 and 1, or None if the model
            does not include a damped trend component.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Notes
        -----
        Uses trailing underscore following scikit-learn convention for fitted
        parameters.

        Examples
        --------
        >>> model = ADAM(model="AAdN")  # Damped trend model
        >>> model.fit(y)
        >>> print(f"Damping: {model.phi_:.3f}")
        """
        self._check_is_fitted()
        if self._model_type.get("damped", False):
            return self._prepared.get("phi", 1.0)
        return None

    @property
    def transition(self) -> NDArray:
        """
        State transition matrix F (R: $transition).

        The transition matrix governs how states evolve from one time period
        to the next in the state-space formulation: v_t = F @ v_{t-1} + g * e_t

        Returns
        -------
        NDArray
            Square matrix of shape (n_states, n_states) defining state
            transitions. Structure depends on model components (ETS, ARIMA).

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        See Also
        --------
        measurement : Measurement matrix W
        states : State values over time

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> F = model.transition
        """
        self._check_is_fitted()
        return self._prepared["mat_f"]

    @property
    def measurement(self) -> NDArray:
        """
        Measurement matrix W (R: $measurement).

        The measurement matrix maps the state vector to the observation
        equation: y_t = W @ v_t + e_t (for additive errors).

        Returns
        -------
        NDArray
            Matrix of shape (obs_in_sample, n_states) that maps states to
            observations. For time-invariant models, all rows are identical.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        See Also
        --------
        transition : State transition matrix F
        states : State values over time

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> W = model.measurement
        """
        self._check_is_fitted()
        return self._prepared["mat_wt"]

    @property
    def initial_value(self) -> Dict[str, Any]:
        """
        Initial state values used for model fitting (R: $initial).

        Contains the starting values for each state component at time t=0,
        which serve as the foundation for the state evolution.

        Returns
        -------
        Dict[str, Any]
            Dictionary with keys ``level`` (initial level value), ``trend``
            (initial trend value, only if model has trend), and ``seasonal``
            (initial seasonal values as array, only if model has seasonality).

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        See Also
        --------
        initial_type : Method used for initialization

        Examples
        --------
        >>> model = ADAM(model="AAA", lags=12)
        >>> model.fit(y)
        >>> init_level = model.initial_value['level']
        """
        self._check_is_fitted()
        return self._prepared.get("initial_value", {})

    @property
    def initial_type(self) -> str:
        """
        Initialization method used for initial states (R: $initialType).

        Returns
        -------
        str
            One of:
            - ``"optimal"``: Initial states optimized during fitting
            - ``"backcasting"``: Initial states estimated via backcasting
            - ``"two-stage"``: Backcast then optimize
            - ``"complete"``: Pure backcasting without optimization
            - ``"provided"``: User-supplied initial values

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        See Also
        --------
        initial_value : The actual initial state values

        Examples
        --------
        >>> model = ADAM(model="AAN", initial="backcasting")
        >>> model.fit(y)
        >>> print(model.initial_type)
        'backcasting'
        """
        self._check_is_fitted()
        return self._initials.get("initial_type", "optimal")

    @property
    def loss_value(self) -> float:
        """
        Optimized loss/cost function value (R: $lossValue).

        The final value of the loss function after optimization. Lower values
        indicate better fit (for most loss functions).

        Returns
        -------
        float
            The minimized loss function value from the optimization process.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        See Also
        --------
        loss : The loss function type used
        loglik : Log-likelihood value

        Examples
        --------
        >>> model = ADAM(model="AAN", loss="MSE")
        >>> model.fit(y)
        >>> print(f"MSE: {model.loss_value:.4f}")
        """
        self._check_is_fitted()
        return self._adam_estimated["CF_value"]

    @property
    def time_elapsed(self) -> float:
        """
        Time taken to fit the model in seconds.

        Returns
        -------
        float
            Elapsed time in seconds from start to end of the fit() call.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="ZXZ", lags=12)
        >>> model.fit(y)
        >>> print(f"Fitting took {model.time_elapsed:.2f} seconds")
        """
        self._check_is_fitted()
        return self.time_elapsed_

    @property
    def data(self) -> NDArray:
        """
        In-sample training data (R: $data).

        Alias for ``actuals`` property. Returns the original time series
        data used for model fitting.

        Returns
        -------
        NDArray
            1D array of in-sample observations.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> training_data = model.data
        """
        return self.actuals

    @property
    def holdout_data(self) -> Optional[NDArray]:
        """
        Holdout validation data (R: $holdout).

        If ``holdout=True`` was specified during fitting, returns the portion
        of data withheld for validation. Otherwise returns None.

        Returns
        -------
        Optional[NDArray]
            1D array of holdout observations, or None if no holdout was used.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAN", holdout=True, h=12)
        >>> model.fit(y)
        >>> if model.holdout_data is not None:
        ...     print(f"Holdout size: {len(model.holdout_data)}")
        """
        self._check_is_fitted()
        if self._general.get("holdout"):
            return np.array(self._observations.get("y_holdout", []))
        return None

    @property
    def b_value(self) -> NDArray:
        """
        Full parameter vector B (R: $B).

        Alias for ``coef`` property. Contains all estimated parameters in a
        single vector, including persistence parameters, initial states,
        ARIMA coefficients, and other model parameters.

        Returns
        -------
        NDArray
            1D array of all model parameters.

        See Also
        --------
        coef : Primary property for parameter vector

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> all_params = model.b_value
        """
        return self.coef

    @property
    def scale(self) -> float:
        """Internal optimisation scale of the error distribution
        (R: ``adam_obj$scale``).

        This is the scale parameter the cost function uses inside the
        density (e.g. for ``dgnorm`` it's the ``α`` in
        ``f(x; β, σ) = β/(2σ Γ(1/β)) exp(-(|x|/σ)^β)``). It coincides
        with the empirical residual std (:attr:`sigma`) only for the
        Normal distribution; for non-normal distributions the two differ
        — :attr:`sigma` is then the variance-adjusted residual std-dev
        (matches R's ``sigma()`` generic), while ``scale`` is the
        density-parameterising scalar used by the cost function.

        Use :attr:`sigma` for reporting "error standard deviation" /
        constructing intervals; use ``scale`` for re-evaluating the
        likelihood or other quantities defined in terms of the density.
        """
        self._check_is_fitted()
        return float(self._prepared.get("scale", float("nan")))

    @property
    def profile(self) -> Optional[Any]:
        """
        Profile information for seasonal patterns (R: $profile).

        Contains profile data used for forecasting with multiple seasonality.
        The profile captures the seasonal pattern structure.

        Returns
        -------
        Optional[Any]
            Profile table for recent observations, or None if not applicable.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAA", lags=[24, 168])
        >>> model.fit(y)
        >>> seasonal_profile = model.profile
        """
        self._check_is_fitted()
        return self._prepared.get("profiles_recent_table")

    @property
    def n_param(self) -> Any:
        """
        Parameter count information (R: $nParam).

        Provides information about the number of parameters in the model,
        distinguishing between estimated and provided parameters.

        Returns
        -------
        Any
            Parameter count information (structure may vary).

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAA", lags=12)
        >>> model.fit(y)
        >>> print(model.n_param)
        """
        self._check_is_fitted()
        return self._general.get("n_param")

    @property
    def constant_value(self) -> Optional[float]:
        """
        Constant/intercept term value (R: $constant).

        The estimated constant term in the model, if one was included.

        Returns
        -------
        Optional[float]
            Constant term value, or None if no constant was estimated.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAN", constant=True)
        >>> model.fit(y)
        >>> if model.constant_value is not None:
        ...     print(f"Constant: {model.constant_value:.4f}")
        """
        self._check_is_fitted()
        return self._prepared.get("constant")

    @property
    def distribution_(self) -> str:
        """
        Error distribution used for fitting (R: $distribution).

        The probability distribution assumed for the error term in the model.
        Uses trailing underscore to distinguish from the input parameter and
        follow scikit-learn convention for fitted attributes.

        Returns
        -------
        str
            Distribution name, one of:
            - ``"dnorm"``: Normal distribution (default for additive errors)
            - ``"dgamma"``: Gamma distribution (default for multiplicative)
            - ``"dlaplace"``: Laplace distribution
            - ``"dlnorm"``: Log-Normal distribution
            - ``"dinvgauss"``: Inverse Gaussian distribution
            - ``"ds"``: S distribution
            - ``"dgnorm"``: Generalized Normal distribution

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        See Also
        --------
        loss_ : Loss function used for estimation

        Examples
        --------
        >>> model = ADAM(model="AAN", distribution="dlaplace")
        >>> model.fit(y)
        >>> print(model.distribution_)
        'dlaplace'
        """
        self._check_is_fitted()
        return self._general.get("distribution_new", self._general.get("distribution"))

    @property
    def loss_(self) -> str:
        """
        Loss function used for parameter estimation (R: $loss).

        The objective function minimized during model fitting. Uses trailing
        underscore to distinguish from the input parameter and follow
        scikit-learn convention for fitted attributes.

        Returns
        -------
        str
            Loss function name, one of:
            - ``"likelihood"``: Maximum likelihood (default)
            - ``"MSE"``: Mean Squared Error
            - ``"MAE"``: Mean Absolute Error
            - ``"HAM"``: Half Absolute Moment
            - ``"LASSO"``: L1 regularization
            - ``"RIDGE"``: L2 regularization
            - ``"GTMSE"``: Geometric Trace MSE
            - ``"GPL"``: Generalized Predictive Likelihood

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        See Also
        --------
        loss_value : The optimized loss function value
        distribution_ : Error distribution used

        Examples
        --------
        >>> model = ADAM(model="AAN", loss="MAE")
        >>> model.fit(y)
        >>> print(model.loss_)
        'MAE'
        """
        self._check_is_fitted()
        return self._general.get("loss")

    # =========================================================================
    # Combined Model Properties
    # =========================================================================

    @property
    def is_combined(self) -> bool:
        """
        Return True if model is a combination of multiple models.

        Combined models are created using "C" in the model string
        (e.g., "CCC", "CCN", "ACA"). The combination uses IC weights to
        combine fitted values and forecasts from all models in the pool.

        Returns
        -------
        bool
            True if the model is a weighted combination of multiple models,
            False otherwise.

        Examples
        --------
        >>> model = ADAM(model="CCC", lags=[1])
        >>> model.fit(y)
        >>> model.is_combined  # True
        """
        return getattr(self, "_is_combined", False)

    @property
    def ic_weights(self) -> Dict[str, float]:
        """
        Return IC weights for combined models (R: $ICw).

        Akaike weights represent the relative likelihood of each model being
        the best model given the data. They are used to combine forecasts from
        multiple models.

        Returns
        -------
        Dict[str, float]
            Dictionary mapping model names to their IC weights. Weights sum to 1.0.

        Raises
        ------
        ValueError
            If the model has not been fitted or was not fitted with combination.

        Examples
        --------
        >>> model = ADAM(model="CCC", lags=[1])
        >>> model.fit(y)
        >>> weights = model.ic_weights
        >>> print(f"ANN weight: {weights.get('ANN', 0):.3f}")
        """
        self._check_is_fitted()
        if not getattr(self, "_is_combined", False):
            raise ValueError("Model was not fitted with combination. Use model='CCC'.")
        return self._ic_weights

    @property
    def models(self) -> List[Dict]:
        """
        Return list of individual models in the combination (R: $models).

        Each entry contains the model name, its IC weight, and the prepared
        model data for forecasting.

        Returns
        -------
        List[Dict]
            List of dictionaries containing:
            - "name": Model name string (e.g., "ANN")
            - "weight": IC weight for this model
            - "prepared": Prepared model data for forecasting

        Raises
        ------
        ValueError
            If the model has not been fitted or was not fitted with combination.

        Examples
        --------
        >>> model = ADAM(model="CCC", lags=[1])
        >>> model.fit(y)
        >>> for m in model.models:
        ...     print(f"{m['name']}: {m['weight']:.3f}")
        """
        self._check_is_fitted()
        if not getattr(self, "_is_combined", False):
            raise ValueError("Model was not fitted with combination. Use model='CCC'.")
        return self._prepared_models

    # =========================================================================
    # Existing Properties
    # =========================================================================

    @property
    def fitted(self) -> NDArray:
        """
        Return in-sample fitted values.

        For combined models (e.g., model="CCC"), returns IC-weighted combination
        of fitted values from all models in the combination.

        Returns
        -------
        NDArray
            Array of fitted values for the in-sample period.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> fitted_values = model.fitted
        """
        self._check_is_fitted()
        if getattr(self, "_is_combined", False):
            return self._combined_fitted
        return self._prepared["y_fitted"]

    @property
    def actuals(self) -> NDArray:
        """
        Return original in-sample data.

        Returns
        -------
        NDArray
            Array of original observations used for fitting.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> original_data = model.actuals
        """
        self._check_is_fitted()
        return np.array(self._observations["y_in_sample"])

    @property
    def coef(self) -> NDArray:
        """
        Return estimated coefficients (parameter vector B).

        The parameter vector B contains all optimized parameters in order:
        1. ETS persistence parameters (α, β, γ)
        2. Damping parameter (φ)
        3. Initial states
        4. ARIMA parameters (AR, MA coefficients)
        5. Regression coefficients
        6. Constant term
        7. Distribution parameters

        Returns
        -------
        NDArray
            Parameter vector B with all estimated coefficients.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> coefficients = model.coef
        """
        self._check_is_fitted()
        return self._adam_estimated["B"]

    @property
    def coef_names(self) -> list:
        """Parameter names aligned with :attr:`coef` (the B vector).

        Mirrors ``names(coef(object))`` in R — e.g. ``alpha``, ``beta``,
        ``gamma``/``gamma1``, ``phi``, ``level``, ``trend``, ``seasonal_2`` …,
        ARIMA ``phi1``/``theta1``, regressor names, ``constant``. Falls back to
        ``b1, b2, …`` if names are unavailable.
        """
        self._check_is_fitted()
        names = self._adam_estimated.get("B_names")
        if names is None or len(names) != len(self.coef):
            return [f"b{i + 1}" for i in range(len(self.coef))]
        return list(names)

    @property
    def fisher_information_(self) -> Optional[NDArray]:
        """Observed Fisher Information matrix at the estimated parameters.

        Computed only when the model was constructed with ``fi=True`` (mirrors
        R's ``adam(..., FI=TRUE)$FI``); otherwise ``None``. The matrix is the
        negative Hessian of the log-likelihood evaluated at the estimated
        coefficient vector :attr:`coef`, with rows/columns in the same order.

        Returns
        -------
        NDArray or None
            ``len(coef) × len(coef)`` symmetric matrix, or ``None`` if
            ``fi=False``.
        """
        self._check_is_fitted()
        return self._adam_estimated.get("FI")

    @property
    def residuals(self) -> NDArray:
        """
        Return model residuals (errors from fitting).

        For additive error models, residuals are y_t - fitted_t.
        For multiplicative error models, residuals are y_t / fitted_t - 1.
        For combined models, returns residuals computed from the IC-weighted
        combined fitted values: y_t - combined_fitted_t.

        Returns
        -------
        NDArray
            Array of residuals from the fitted model.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> errors = model.residuals
        >>> rmse = np.sqrt(np.mean(errors**2))
        """
        self._check_is_fitted()
        if getattr(self, "_is_combined", False):
            return self._combined_residuals
        return self._prepared["residuals"]

    def rstandard(self) -> NDArray:
        """
        Return standardised residuals.

        Residuals are scaled by distribution-specific estimates of their
        scale, corrected for degrees of freedom ``df = nobs - nparam``.
        The standardisation formula depends on the fitted distribution:

        - **dnorm**: ``(e - ē) / (σ √(n/df))``
        - **dlaplace**: ``e / σ · n/df``
        - **ds**: ``(e - ē) / (σ · n/df)²``
        - **dgnorm**: ``(e - ē) / (σ^β · n/df)^(1/β)``, β = shape
        - **dlnorm**: ``exp((log(e) + σ²/2 - mean(·)) / (σ √(n/df)))``
        - **dinvgauss / dgamma**: ``e / ē``

        For a correctly specified model the result should be approximately
        distributed as the standardised version of the fitted distribution.

        Returns
        -------
        NDArray
            Standardised residuals, length ``nobs``.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        See Also
        --------
        rstudent : Leave-one-out studentised residuals.
        outlierdummy : Outlier detection based on standardised residuals.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> std_res = model.rstandard()
        >>> # For a well-specified normal model, std_res ≈ N(0, 1)
        >>> abs(std_res.mean()) < 0.1
        True
        """
        self._check_is_fitted()
        obs = self.nobs
        df = obs - self.nparam
        errors = self.residuals.copy().astype(float)
        dist = self.distribution_
        scale = self.sigma

        if dist == "dnorm":
            mean_e = np.mean(errors)
            return (errors - mean_e) / (scale * np.sqrt(obs / df))
        elif dist == "ds":
            mean_e = np.mean(errors)
            return (errors - mean_e) / (scale * obs / df) ** 2
        elif dist == "dgnorm":
            beta = self.gnorm_shape if self.gnorm_shape is not None else 2.0
            mean_e = np.mean(errors)
            return (errors - mean_e) / (scale**beta * obs / df) ** (1.0 / beta)
        elif dist in ("dinvgauss", "dgamma"):
            return errors / np.mean(errors)
        elif dist == "dlnorm":
            log_e = np.log(errors) + scale**2 / 2
            return np.exp((log_e - np.mean(log_e)) / (scale * np.sqrt(obs / df)))
        else:  # dlaplace and other additive distributions
            return errors / scale * obs / df

    def rstudent(self) -> NDArray:
        """
        Return studentised (leave-one-out) residuals.

        Each residual is scaled by the distribution-specific scale estimate
        recomputed *without that observation* (leave-one-out). Compared to
        ``rstandard()``, the result is more sensitive to individual outliers
        because no single point inflates the global scale estimate it is
        judged against.

        The leave-one-out sums are computed in O(n) using identities such as
        ``Σ e[-i]² = Σ e² − e[i]²``, avoiding an explicit loop.
        Degrees of freedom: ``df = nobs - nparam - 1``.

        Standardisation by distribution:

        - **dnorm**: ``(e[i] - ē) / √(Σe[-i]² / df)``
        - **dlaplace**: ``(e[i] - ē) / (Σ|e[-i]| / df)``
        - **ds**: ``(e[i] - ē) / (Σ√|e[-i]| / (2 df))²``
        - **dgnorm**: ``(e[i] - ē) / (Σ|e[-i]|^β · β/df)^(1/β)``
        - **dlnorm**: ``exp(log_e[i] / √(Σlog_e[-i]² / df))``,
          where ``log_e = log(e) - mean(log e) - σ²/2``
        - **dinvgauss / dgamma**: ``e[i] / mean(e[-i])``

        Returns
        -------
        NDArray
            Studentised residuals, length ``nobs``.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        See Also
        --------
        rstandard : Simpler standardised residuals (faster).
        outlierdummy : Outlier detection based on studentised residuals.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> stu_res = model.rstudent()
        >>> # Studentised residuals are slightly more spread than rstandard
        >>> stu_res.std() >= model.rstandard().std()
        True
        """
        self._check_is_fitted()
        obs = self.nobs
        df = obs - self.nparam - 1
        errors = self.residuals.copy().astype(float)
        dist = self.distribution_

        if dist == "dnorm":
            errors -= np.mean(errors)
            total_sq = np.sum(errors**2)
            denom = np.sqrt((total_sq - errors**2) / df)
            return errors / denom

        elif dist == "dlaplace":
            errors -= np.mean(errors)
            total_abs = np.sum(np.abs(errors))
            denom = (total_abs - np.abs(errors)) / df
            return errors / denom

        elif dist == "ds":
            errors -= np.mean(errors)
            total_sqrt_abs = np.sum(np.sqrt(np.abs(errors)))
            denom = ((total_sqrt_abs - np.sqrt(np.abs(errors))) / (2 * df)) ** 2
            return errors / denom

        elif dist == "dgnorm":
            beta = self.gnorm_shape if self.gnorm_shape is not None else 2.0
            errors -= np.mean(errors)
            total_pow = np.sum(np.abs(errors) ** beta)
            denom = ((total_pow - np.abs(errors) ** beta) * (beta / df)) ** (1.0 / beta)
            return errors / denom

        elif dist in ("dinvgauss", "dgamma"):
            total = np.sum(errors)
            mean_loo = (total - errors) / (obs - 1)
            return errors / mean_loo

        elif dist == "dlnorm":
            scale = self.sigma
            log_e = np.log(errors) - np.mean(np.log(errors)) - scale**2 / 2
            total_sq = np.sum(log_e**2)
            denom = np.sqrt((total_sq - log_e**2) / df)
            return np.exp(log_e / denom)

        else:  # default: treat like normal
            errors -= np.mean(errors)
            total_sq = np.sum(errors**2)
            denom = np.sqrt((total_sq - errors**2) / df)
            return errors / denom

    def outlierdummy(
        self,
        level: float = 0.999,
        type: str = "rstandard",  # noqa: A002
    ) -> OutlierDummy:
        """
        Detect outliers and return a matrix of indicator dummy variables.

        Computes standardised residuals (via ``rstandard()`` or ``rstudent()``),
        then derives two-sided quantile bounds ``[q_lo, q_hi]`` for the fitted
        distribution at the given confidence ``level``. Observations whose
        standardised residual falls outside these bounds are labelled outliers.

        The quantile bounds are distribution-specific:

        - **dnorm / dlnorm** (log-space): ``scipy.stats.norm.ppf``
        - **dlaplace**: ``scipy.stats.laplace.ppf``
        - **ds**: ``scipy.stats.gennorm.ppf(..., beta=0.5)``
        - **dgnorm**: ``scipy.stats.gennorm.ppf(..., beta=shape)``
        - **dgamma**: ``scipy.stats.gamma.ppf(..., a=1/σ, scale=σ)``
        - **dinvgauss**: ``scipy.stats.invgauss.ppf`` with df-corrected dispersion

        Parameters
        ----------
        level : float, default 0.999
            Two-sided confidence level for outlier detection.
            0.99 flags observations in the outer 1 % of the distribution.
        type : {"rstandard", "rstudent"}, default "rstandard"
            Which standardised residuals to use. ``"rstudent"`` is more
            sensitive to individual outliers in small samples.

        Returns
        -------
        OutlierDummy
            Dataclass with fields:

            ``outliers`` : ndarray of shape (n, m) or None
                Binary dummy matrix, one column per detected outlier.
                ``None`` when no outliers are found.
            ``id`` : ndarray of int
                0-based indices of outlier observations.
            ``statistic`` : ndarray of shape (2,)
                ``[lower, upper]`` quantile bounds used for detection.
            ``level`` : float
                The confidence level used.
            ``type`` : str
                The residual type used (``"rstandard"`` or ``"rstudent"``).

        Raises
        ------
        ValueError
            If ``type`` is not ``"rstandard"`` or ``"rstudent"``.
        ValueError
            If the model has not been fitted yet.

        See Also
        --------
        rstandard : Standardised residuals.
        rstudent : Studentised residuals.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> od = model.outlierdummy(level=0.99)
        >>> od.id          # 0-based indices of detected outliers
        >>> od.statistic   # [lower, upper] bounds, e.g. [-2.576, 2.576]
        >>> if od.outliers is not None:
        ...     # Refit with outlier dummies as exogenous regressors
        ...     model2 = ADAM(model="AAN")
        ...     model2.fit(y, X=od.outliers)
        """
        self._check_is_fitted()
        if type not in ("rstandard", "rstudent"):
            raise ValueError("type must be 'rstandard' or 'rstudent'")

        errors = self.rstandard() if type == "rstandard" else self.rstudent()
        dist = self.distribution_
        p = np.array([(1 - level) / 2, (1 + level) / 2])

        if dist == "dnorm":
            stat = scipy_stats.norm.ppf(p)
        elif dist == "dlaplace":
            stat = scipy_stats.laplace.ppf(p)
        elif dist == "ds":
            stat = scipy_stats.gennorm.ppf(p, beta=0.5)
        elif dist == "dgnorm":
            beta = self.gnorm_shape if self.gnorm_shape is not None else 2.0
            stat = scipy_stats.gennorm.ppf(p, beta=beta)
        elif dist == "dlnorm":
            errors = np.log(errors)
            stat = scipy_stats.norm.ppf(p)
        elif dist == "dgamma":
            scale = self.sigma
            stat = scipy_stats.gamma.ppf(p, a=1.0 / scale, scale=scale)
        elif dist == "dinvgauss":
            nobs, npar = self.nobs, self.nparam
            disp = float(self.sigma) * nobs / (nobs - npar)
            stat = scipy_stats.invgauss.ppf(p, mu=disp, scale=1.0 / disp)
        else:
            stat = scipy_stats.norm.ppf(p)

        outlier_ids = np.where((errors > stat[1]) | (errors < stat[0]))[0]
        n_out = len(outlier_ids)

        if n_out > 0:
            dummy_mat = np.zeros((self.nobs, n_out), dtype=float)
            dummy_mat[outlier_ids, np.arange(n_out)] = 1.0
        else:
            dummy_mat = None

        return OutlierDummy(
            outliers=dummy_mat,
            id=outlier_ids,
            statistic=stat,
            level=level,
            type=type,
        )

    @staticmethod
    def _expand_outlier_dummies(D: NDArray) -> NDArray:
        """
        Expand each outlier dummy column into three columns: lag-1, t, lead+1.

        Used by ``outliers="select"`` to allow the regressor selection
        mechanism to choose which temporal offset carries the outlier effect.

        Parameters
        ----------
        D : NDArray of shape (n, m)
            Binary dummy matrix from ``outlierdummy()``.

        Returns
        -------
        NDArray of shape (n, 3*m)
            Expanded matrix with columns ordered as
            ``[lag_1, t, lead_1, lag_2, t_2, lead_2, ...]``.
        """
        n, m = D.shape
        cols = []
        for j in range(m):
            col = D[:, j]
            cols.append(np.concatenate([[0.0], col[:-1]]))  # lag -1
            cols.append(col.copy())  # t
            cols.append(np.concatenate([col[1:], [0.0]]))  # lead +1
        return np.column_stack(cols)

    @property
    def nobs(self) -> int:
        """
        Return number of observations used for fitting.

        Returns
        -------
        int
            Number of in-sample observations.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> n = model.nobs
        """
        self._check_is_fitted()
        return len(self._observations["y_in_sample"])

    @property
    def nparam(self) -> int:
        """
        Return number of estimated parameters.

        Returns
        -------
        int
            Number of parameters estimated during optimization.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> k = model.nparam
        """
        self._check_is_fitted()
        return self._adam_estimated["n_param_estimated"]

    @property
    def sigma(self) -> float:
        """Empirical residual standard deviation (R: ``sigma(adam_obj)``).

        Mirrors R's ``sigma.adam`` (R/adam.R:4625-4658): df-adjusted sum
        of squared residuals (with a distribution-specific transformation
        for log-domain and multiplicative distributions). For likelihood
        loss the scale parameter is subtracted from ``nparam`` to match
        R's convention.

        For the common case (``dnorm`` / ``dlaplace`` / ``ds`` / ``dgnorm``
        / ``dt`` / ``dlogis`` / ``dalaplace``) this is
        ``sqrt(sum(residuals²) / (nobs − nparam + 1))`` when loss is
        likelihood, or ``sqrt(sum(residuals²) / (nobs − nparam))``
        otherwise.

        Use :attr:`scale` if you want the model's internal optimisation
        scale (R: ``adam_obj$scale``) — they coincide only for ``dnorm``
        but differ for non-normal distributions where the optimisation
        scale parameterises the density and the empirical residual std
        is a separate scalar.
        """
        self._check_is_fitted()

        residuals = np.asarray(self.residuals, dtype=float)
        residuals = residuals[np.isfinite(residuals)]
        n_obs = int(self.nobs)
        # R's ``sigma.adam`` formula is ``df = nobs − (nparam − 1)`` for
        # likelihood loss (subtract one for the scale parameter R counts
        # but treats specially). Python's ``self.nparam`` already excludes
        # the scale parameter, so the equivalent is just ``nobs - nparam``
        # — no further subtraction needed. For non-likelihood losses R
        # uses ``nobs - nparam`` directly; same here.
        n_param = int(self.nparam)
        df = n_obs - n_param
        if df <= 0:
            df = n_obs

        distribution = (
            (
                self._general.get("distribution_new")
                or self._general.get("distribution", "dnorm")
            )
            if self._general
            else "dnorm"
        )

        # Log-domain distributions: R's residuals.adam returns 1+epsilon
        # for these; the log of that recovers epsilon-on-log-scale. Use
        # complex arithmetic so log(non-positive) doesn't NaN out — same
        # pattern as ``adam_scaler``'s inline ``complex_log``.
        def _complex_log_abs(x):
            return np.abs(np.log(np.asarray(x, dtype=np.complex128)))

        if distribution in ("dlnorm", "dllaplace", "dls"):
            ss = float(np.sum(_complex_log_abs(1.0 + residuals) ** 2))
        elif distribution == "dlgnorm":
            opt_scale = float(self._prepared.get("scale", 0.0))
            ss = float(
                np.sum(_complex_log_abs(1.0 + residuals - opt_scale**2 / 2.0) ** 2)
            )
        elif distribution in ("dinvgauss", "dgamma"):
            # R: sum((residuals − 1)²) where residuals is 1+epsilon — i.e.
            # sum(epsilon²) in Python's additive convention.
            ss = float(np.sum(residuals**2))
        else:
            # dnorm, dlaplace, ds, dgnorm, dt, dlogis, dalaplace — plain
            # sum of squared residuals.
            ss = float(np.sum(residuals**2))

        return float(np.sqrt(ss / df))

    @property
    def loglik(self) -> float:
        """
        Return log-likelihood of the fitted model.

        Returns
        -------
        float
            Log-likelihood value.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> ll = model.loglik
        """
        self._check_is_fitted()
        return self._adam_estimated["log_lik_adam_value"]["value"]

    @property
    def aic(self) -> float:
        """
        Return Akaike Information Criterion.

        Returns
        -------
        float
            AIC value.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> aic_val = model.aic
        """
        self._check_is_fitted()
        from smooth.adam_general.core.utils.ic import AIC

        log_lik = self._adam_estimated["log_lik_adam_value"]
        return AIC(log_lik["value"], log_lik["nobs"], log_lik["df"])

    @property
    def aicc(self) -> float:
        """
        Return corrected Akaike Information Criterion.

        Returns
        -------
        float
            AICc value.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> aicc_val = model.aicc
        """
        self._check_is_fitted()
        from smooth.adam_general.core.utils.ic import AICc

        log_lik = self._adam_estimated["log_lik_adam_value"]
        return AICc(log_lik["value"], log_lik["nobs"], log_lik["df"])

    @property
    def bic(self) -> float:
        """
        Return Bayesian Information Criterion.

        Returns
        -------
        float
            BIC value.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> bic_val = model.bic
        """
        self._check_is_fitted()
        from smooth.adam_general.core.utils.ic import BIC

        log_lik = self._adam_estimated["log_lik_adam_value"]
        return BIC(log_lik["value"], log_lik["nobs"], log_lik["df"])

    @property
    def bicc(self) -> float:
        """
        Return corrected Bayesian Information Criterion.

        Returns
        -------
        float
            BICc value.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> bicc_val = model.bicc
        """
        self._check_is_fitted()
        from smooth.adam_general.core.utils.ic import BICc

        log_lik = self._adam_estimated["log_lik_adam_value"]
        return BICc(log_lik["value"], log_lik["nobs"], log_lik["df"])

    @property
    def error_type(self) -> str:
        """
        Return error type: 'A' (additive) or 'M' (multiplicative).

        Returns
        -------
        str
            'A' for additive errors, 'M' for multiplicative errors.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> err_type = model.error_type  # Returns 'A'
        """
        self._check_is_fitted()
        return self._model_type["error_type"]

    @property
    def model_type(self) -> str:
        """
        Return ETS model type code (e.g., 'AAN', 'AAA', 'MAdM').

        Returns just the three/four-letter ETS code, not the full model name.
        For the full model name including ARIMA orders and X indicator,
        use ``model_name``.

        Returns
        -------
        str
            ETS model code where:
            - First letter: Error type (A/M)
            - Second letter: Trend type (N/A/Ad/M/Md)
            - Third letter: Seasonal type (N/A/M)

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="ZZZ")  # Auto-selection
        >>> model.fit(y)
        >>> selected_type = model.model_type  # e.g., 'AAN'
        """
        self._check_is_fitted()
        return self._model_type.get("model", "")

    @property
    def orders(self) -> Dict[str, List[int]]:
        """
        Return ARIMA orders as dict with 'ar', 'i', 'ma' keys.

        Returns
        -------
        Dict[str, List[int]]
            Dictionary with keys 'ar', 'i', 'ma' containing lists of orders
            for each lag. For pure ETS models, returns [0] for each.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="NNN", ar_order=1, i_order=1, ma_order=1)
        >>> model.fit(y)
        >>> arima_orders = model.orders
        >>> print(arima_orders)  # {'ar': [1], 'i': [1], 'ma': [1]}
        """
        self._check_is_fitted()
        ar = self._arima.get("ar_orders")
        i = self._arima.get("i_orders")
        ma = self._arima.get("ma_orders")
        return {
            "ar": ar if ar is not None else [0],
            "i": i if i is not None else [0],
            "ma": ma if ma is not None else [0],
        }

    @property
    def model_name(self) -> str:
        """
        Return full model name string (R: modelName()).

        Returns the complete model specification including ETS type,
        ARIMA orders, and X indicator for regressors.

        Returns
        -------
        str
            Full model name, e.g., 'ETS(AAN)', 'ETS(AAA)+ARIMA(1,1,1)',
            or 'ETSX(MAN)' when regressors are included.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> name = model.model_name  # 'ETS(AAN)'
        """
        self._check_is_fitted()
        return self.model if isinstance(self.model, str) else str(self.model)

    @property
    def lags_used(self) -> List[int]:
        """
        Return the vector of lags used in the model.

        Returns
        -------
        List[int]
            List of lag values (seasonal periods) used in the model.
            For example, [1, 12] for monthly data with annual seasonality.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.

        Examples
        --------
        >>> model = ADAM(model="AAA", lags=[1, 12])
        >>> model.fit(y)
        >>> model.lags_used  # [1, 12]
        """
        self._check_is_fitted()
        return list(self._lags_model.get("lags", [1]))

    @property
    def om_model(self):
        """Fitted occurrence model (OM / OMG / AutoOM), or None."""
        self._check_is_fitted()
        return getattr(self, "_om_model", None)

    def rmultistep(self, h: int = 10) -> pd.DataFrame:
        """Return the (T-h) × h matrix of rolling in-sample multistep forecast errors.

        For each origin ``t``, computes the ``h``-step-ahead forecast and the
        corresponding errors against the realised observations. Must be called
        after fit().

        Parameters
        ----------
        h : int
            Forecast horizon (number of steps ahead). Default 10.

        Returns
        -------
        pd.DataFrame
            Shape (T-h, h) where T is obs_in_sample.
        """
        from smooth.adam_general.core.forecaster._helpers import (
            _compute_multistep_errors,
        )

        self._check_is_fitted()
        gen = {**self._general, "h": h}
        mat_wt = np.asfortranarray(self._prepared["measurement"], dtype=np.float64)
        mat_f = np.asfortranarray(self._prepared["transition"], dtype=np.float64)
        errors = _compute_multistep_errors(
            self._adam_cpp,
            self._prepared,
            self._observations,
            self._lags_model,
            gen,
            mat_wt,
            mat_f,
        )
        return pd.DataFrame(errors, columns=[f"h={i + 1}" for i in range(h)])

    def predict(
        self,
        h: int,
        X: Optional[NDArray] = None,
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
        level: Optional[Union[float, List[float]]] = 0.95,
        side: Literal["both", "upper", "lower"] = "both",
        cumulative: bool = False,
        nsim: int = 10000,
        occurrence: Optional[NDArray] = None,
        scenarios: bool = False,
        seed: Optional[int] = None,
    ) -> NDArray:
        """
        Generate forecasts using the fitted ADAM model.

        Parameters
        ----------
        h : int
            Forecast horizon (number of steps to forecast).
        X : Optional[NDArray], default=None
            Exogenous variables for the forecast period.
            Ensure that X covers the entire forecast horizon ``h``.
        interval : str, default="none"
            Type of prediction interval to construct:

            - ``"none"``: No intervals, point forecasts only.
            - ``"prediction"``: Automatically selects ``"simulated"`` or
              ``"approximate"`` depending on the model type.
            - ``"simulated"``: Simulation-based intervals (Monte Carlo).
            - ``"approximate"``: Analytical (parametric) intervals.
            - ``"semiparametric"``: Not yet implemented.
            - ``"nonparametric"``: Not yet implemented.
            - ``"empirical"``: Not yet implemented.
            - ``"confidence"``: Not yet implemented.
            - ``"complete"``: Not yet implemented.
        level : float or list of float, default=0.95
            Confidence level(s) for prediction intervals. Accepts a single
            value (e.g. ``0.95``) or a list for multiple simultaneous levels
            (e.g. ``[0.9, 0.95, 0.99]``). Values above 1 are treated as
            percentages and divided by 100.

            Each level produces a pair of ``lower_X`` / ``upper_X`` columns
            in the output, where X is the corresponding quantile. For example,
            ``level=0.95`` with ``side="both"`` yields columns
            ``"lower_0.025"`` and ``"upper_0.975"``.
        side : str, default="both"
            Which side(s) of the intervals to compute:

            - ``"both"``: Both lower and upper bounds (default).
            - ``"upper"``: Upper bound only. Column named ``"upper_<level>"``.
            - ``"lower"``: Lower bound only. Column named ``"lower_<1-level>"``.
        cumulative : bool, default=False
            If True, return cumulative (summed) forecasts over the horizon.
        nsim : int, default=10000
            Number of simulations for simulation-based intervals.
        occurrence : Optional[NDArray], default=None
            External occurrence probabilities for the forecast period.
            Overrides the fitted model's occurrence for forecasting.
        scenarios : bool, default=False
            If True and ``interval="simulated"``, store the raw simulation
            matrix in ``self._general["_scenarios_matrix"]``.
        seed : int, optional
            Seed forwarded to :meth:`reforecast` when ``interval`` is
            ``"complete"`` or ``"confidence"``. Pins the Monte-Carlo
            paths so the interval is reproducible across runs and
            platforms. Ignored for the other ``interval`` modes.

        Returns
        -------
        ForecastResult
            Structured result with ``.mean`` (pd.Series), ``.lower`` and
            ``.upper`` (pd.DataFrame or None), ``.level``, ``.side``, and
            ``.interval`` attributes.  Use ``.to_dataframe()`` for a flat
            pd.DataFrame.

        Raises
        ------
        ValueError
            If the model has not been fitted yet or ``h`` is not set.
        """
        # Set forecast horizon
        if h is not None:
            self.h = h
            self._general["h"] = self.h
        else:
            if self._general["h"] is None:
                raise ValueError("Forecast horizon is not set.")

        self._general["interval"] = interval
        self._general["nsim"] = nsim
        self._general["cumulative"] = cumulative
        self._general["scenarios"] = scenarios

        if occurrence is not None:
            self._occurrence["occurrence"] = occurrence

        # ``interval="complete"`` / ``"confidence"`` delegate to
        # :meth:`reforecast` — they need per-parameter-draw refitting of
        # the in-sample data, which the existing forecaster path doesn't
        # do. R's ``forecast.adam`` defaults ``nsim=100`` for both modes;
        # honour that whenever the caller hasn't overridden the
        # ``predict`` default of 10000.
        if interval in ("complete", "confidence"):
            reforecast_nsim = 100 if nsim == 10000 else int(nsim)
            sub_interval: Literal["prediction", "confidence"] = (
                "prediction" if interval == "complete" else "confidence"
            )
            reforecast_result = self.reforecast(
                h=h,
                X=X,
                occurrence=occurrence,
                interval=sub_interval,
                level=level if level is not None else 0.95,
                side=side,
                cumulative=cumulative,
                nsim=reforecast_nsim,
                seed=seed,
            )
            return reforecast_result.to_forecast_result()

        # Store new_xreg for forecast period (used in _generate_point_forecasts)
        if X is not None and self._explanatory.get("xreg_model"):
            new_xreg = np.asarray(X)
            if new_xreg.dtype.names is not None:
                new_xreg = np.column_stack(
                    [new_xreg[f].astype(float) for f in new_xreg.dtype.names]
                )
            else:
                new_xreg = new_xreg.astype(float)
            if new_xreg.ndim == 1:
                new_xreg = new_xreg.reshape(-1, 1)
            self._explanatory["new_xreg"] = new_xreg
        else:
            self._explanatory.pop("new_xreg", None)

        # Validate prediction inputs and prepare data for forecasting
        self._validate_prediction_inputs()
        self._prepare_prediction_data()

        # Pre-compute p_forecast so forecaster can use it for both point forecasts
        # and interval adjustment (R: forecast.adam lines 6134-6172, 6204-6210).
        p_forecast_arr = None
        om_model = getattr(self, "_om_model", None)
        if om_model is not None:
            occ_fc = om_model.predict(h=h)
            p_forecast_arr = np.asarray(occ_fc.mean.values, dtype=float)
            # Store in occurrence_dict so _process_occurrence_forecast can use it
            self._occurrence["p_forecast"] = p_forecast_arr

        # Execute the prediction (forecaster handles p scaling + interval adjustment)
        predictions = self._execute_prediction(
            interval=interval,
            level=level,
            side=side,
        )

        # Post-interval NaN patch: when an occurrence forecast is present
        # and the upper bound came back as NaN (sparse occurrence at long
        # horizons), substitute mean / p so the band stays usable.
        if p_forecast_arr is not None:
            if predictions.upper is not None:
                upper_vals = predictions.upper.values
                mask = np.isnan(upper_vals)
                if np.any(mask):
                    fill = (predictions.mean.values / p_forecast_arr).reshape(-1, 1)
                    predictions.upper = pd.DataFrame(
                        np.where(mask, fill, upper_vals),
                        index=predictions.upper.index,
                        columns=predictions.upper.columns,
                    )
            if predictions.lower is not None:
                lower_vals = predictions.lower.values
                if np.any(np.isnan(lower_vals)):
                    predictions.lower = pd.DataFrame(
                        np.where(np.isnan(lower_vals), 0.0, lower_vals),
                        index=predictions.lower.index,
                        columns=predictions.lower.columns,
                    )

        # Recompute accuracy measures against holdout if available
        if self._general.get("holdout", False):
            y_holdout = self._observations.get("y_holdout")
            y_in_sample = self._observations.get("y_in_sample")
            if y_holdout is not None and len(y_holdout) > 0:
                from smooth.adam_general.core.utils.printing import (
                    _compute_forecast_errors,
                )

                fc_values = np.asarray(predictions.mean, dtype=float).ravel()
                y_holdout_arr = np.asarray(y_holdout, dtype=float).ravel()
                n = min(len(fc_values), len(y_holdout_arr))
                # ``lags`` can be empty (e.g. non-seasonal SMA); fall back to 1
                period = (
                    max(self._lags_model.get("lags") or [1]) if self._lags_model else 1
                )
                self.accuracy = _compute_forecast_errors(
                    y_holdout_arr[:n],
                    fc_values[:n],
                    np.asarray(y_in_sample, dtype=float),
                    period,
                )

        return predictions

    def predict_intervals(
        self,
        h: int,
        X: Optional[NDArray] = None,
        levels: List[float] = [0.8, 0.95],
        side: Literal["both", "upper", "lower"] = "both",
        nsim: int = 10000,
    ):
        """
        Generate prediction intervals using the fitted ADAM model.

        Convenience wrapper around ``predict()`` that defaults to
        ``interval="prediction"`` and accepts multiple confidence levels.

        Parameters
        ----------
        h : int
            Forecast horizon (number of steps to forecast).
        X : Optional[NDArray], default=None
            Exogenous variables for the forecast period.
        levels : List[float], default=[0.8, 0.95]
            Confidence levels for prediction intervals. Each level produces
            a pair of lower/upper columns in the output DataFrame. For
            example, ``levels=[0.8, 0.95]`` with ``side="both"`` yields
            columns ``"lower_0.1"``, ``"lower_0.025"``, ``"upper_0.9"``,
            ``"upper_0.975"``.
        side : Literal["both", "upper", "lower"], default="both"
            Which side(s) of the intervals to return.
        nsim : int, default=10000
            Number of simulations for simulation-based intervals.

        Returns
        -------
        ForecastResult
            Structured result with ``.mean``, ``.lower``, ``.upper`` attributes.
        """
        return self.predict(
            h=h,
            X=X,
            interval="prediction",
            level=levels,
            side=side,
            nsim=nsim,
        )

    def _check_parameters(self, ts, X=None):
        """
        Check parameters using parameters_checker and store results.

        Parameters
        ----------
        ts : NDArray or pd.Series
            Time series data (numpy array or pandas Series).
        X : NDArray or pd.DataFrame, optional
            External regressors, shape (len(ts), n_features).
        """
        # Build orders dict for parameters_checker via the shared precedence
        # resolver in ``checker/arima_checks.py``:
        # ``orders`` dict → wins (scalars ignored, warns if both supplied).
        # ``ar_order``/``i_order``/``ma_order`` (any non-zero) → fixed orders.
        # Otherwise → no ARIMA.
        from smooth.adam_general.core.checker.arima_checks import resolve_arima_orders

        orders, _ = resolve_arima_orders(
            self._init_orders,
            self.ar_order,
            self.i_order,
            self.ma_order,
            arima_select=self.arima_select,
        )

        result = parameters_checker(
            ts,
            model=self.model,
            lags=self.lags,
            orders=orders,
            constant=self.constant,
            outliers=self.outliers,
            level=self.outliers_level,
            persistence=self.persistence,
            phi=self.phi,
            initial=self.initial,
            n_iterations=self.n_iterations,
            distribution=self.distribution,
            loss=self.loss,
            h=self.h,
            holdout=self.holdout,
            occurrence=self.occurrence,
            ic=self.ic,
            bounds=self.bounds,
            silent=(self.verbose == 0),
            fast=self.fast,
            lambda_param=self.lambda_param,
            X=X,
            regressors=self.regressors,
            arma=self.arma,
        )
        if not isinstance(result, tuple):
            self._alm_model = result
            return
        self._alm_model = None
        (
            self._general,
            self._observations,
            self._persistence,
            self._initials,
            self._arima,
            self._constant,
            self._model_type,
            self._components,
            self._lags_model,
            self._occurrence,
            self._phi_internal,
            self._explanatory,
            self._params_info,
        ) = result

    def _populate_from_alm(self, y, X):
        """Populate model attributes from the greybox.ALM early-exit object.

        Used when ``model="NNN"`` with regressors but no ETS/ARIMA — in that
        case the parameter checker fits a greybox.ALM and we wrap its
        outputs into the ADAM attribute surface.
        """
        alm = self._alm_model
        n = int(alm.nobs)
        y_in_sample = np.asarray(y[:n], dtype=float)
        fitted = np.asarray(alm.fitted_values_, dtype=float)

        self._observations = {
            "y_in_sample": y_in_sample,
            "ot": (y_in_sample != 0).astype(float),
        }
        self._model_type = {
            "model": "NNN",
            "ets_model": False,
            "arima_model": False,
            "xreg_model": True,
        }
        self._arima = {"ar_orders": [0], "i_orders": [0], "ma_orders": [0]}
        self._explanatory = {"xreg_model": True}
        self._general = {
            "loss": "likelihood",
            "h": getattr(self, "h", 0),
            "holdout": getattr(self, "holdout", False),
        }
        self._prepared = {
            "y_fitted": fitted,
            "residuals": y_in_sample - fitted,
        }
        self._adam_estimated = {
            "B": np.asarray(alm.coefficients),
            "n_param_estimated": int(alm.nparam),
            "log_lik_adam_value": {
                "value": float(alm.loglik),
                "nobs": n,
                "df": int(alm.nparam),
            },
        }
        self.model = "Regression"
        self.time_elapsed_ = time.time() - self._start_time

    def _handle_lasso_ridge_special_case(self):
        """
        Handle special case for LASSO/RIDGE with lambda=1.

        Sets appropriate parameter values. This is a special case where we use
        MSE to estimate initials only and disable other parameter estimation.
        """
        if self._general["loss"] in ["LASSO", "RIDGE"] and self._general["lambda"] == 1:
            if self._model_type["ets_model"]:
                # Pre-set ETS parameters
                self._persistence["persistence_estimate"] = False
                self._persistence["persistence_level_estimate"] = False
                self._persistence["persistence_trend_estimate"] = False
                self._persistence["persistence_seasonal_estimate"] = [False]
                self._persistence["persistence_level"] = 0
                self._persistence["persistence_trend"] = 0
                self._persistence["persistence_seasonal"] = [0]
                # Phi
                self._phi_internal["phi_estimate"] = False
                self._phi_internal["phi"] = 1

            if self._model_type["xreg_model"]:
                # ETSX parameters
                self._persistence["persistence_xreg_estimate"] = False
                self._persistence["persistence_xreg"] = 0

            if self._model_type["arima_model"]:
                # Pre-set ARMA parameters
                self._arima["ar_estimate"] = [False]
                self._arima["ma_estimate"] = [False]
                self._preset_arima_parameters()

            self._general["lambda"] = 0

    def _fit_occurrence_model(self, y):
        """Fit an occurrence model on ``y`` and return it.

        The occurrence type is taken from ``self._occurrence["occurrence"]``.
        """
        from smooth.adam_general.core.auto_om import AutoOM
        from smooth.adam_general.core.om import OM

        occ = self._occurrence["occurrence"]
        lags = list(self._lags_model.get("lags", [1]))
        adam_model = self._model_type.get("model", "MNN")
        common = dict(
            lags=lags,
            h=self._general.get("h", 0),
            holdout=self._general.get("holdout", False),
            ic=self._general.get("ic", "AICc"),
            bounds=self._general.get("bounds", "usual"),
            initial=self._initials.get("initial_type", "backcasting"),
        )
        if occ == "auto":
            m = AutoOM(model=adam_model, **common)
        elif occ == "general":
            from smooth.adam_general.core.omg import OMG

            m = OMG(**common)
        else:
            m = OM(model=adam_model, occurrence=occ, **common)
        m.fit(y)
        return m

    def _preset_arima_parameters(self):
        """Set up ARIMA parameters for special cases where estimation is disabled."""
        arma_parameters = []
        for i, lag in enumerate(self._lags_model["lags"]):
            if self._arima["ar_orders"][i] > 0:
                arma_parameters.extend([1] * self._arima["ar_orders"][i])
            if self._arima["ma_orders"][i] > 0:
                arma_parameters.extend([0] * self._arima["ma_orders"][i])
        self._arima["arma_parameters"] = arma_parameters

    def _execute_estimation(self, estimation=True):
        """
        Execute model estimation when model_do is 'estimate'.

        This handles parameter estimation and model creation.
        """
        # Handle special case for LASSO/RIDGE with lambda=1
        self._handle_lasso_ridge_special_case()

        # Estimate the model
        # Note: estimator() handles two-stage initialization internally
        if estimation:
            # gnorm shape: estimate it from the data when the user did not
            # supply ``gnorm_shape``.
            dist = self._general.get("distribution_new") or self._general.get(
                "distribution", "dnorm"
            )
            other_parameter_estimate = dist == "dgnorm" and self.gnorm_shape is None
            if dist != "dgnorm":
                other_value = None
            elif self.gnorm_shape is None:
                other_value = 2.0
            else:
                other_value = float(self.gnorm_shape)

            nlopt_params = self.nlopt_kwargs if self.nlopt_kwargs else {}
            self._adam_estimated = estimator(
                general_dict=self._general,
                model_type_dict=self._model_type,
                lags_dict=self._lags_model,
                observations_dict=self._observations,
                arima_dict=self._arima,
                constant_dict=self._constant,
                explanatory_dict=self._explanatory,
                profiles_recent_table=self.profiles_recent_table,
                profiles_recent_provided=self.profiles_recent_provided,
                persistence_dict=self._persistence,
                initials_dict=self._initials,
                occurrence_dict=self._occurrence,
                phi_dict=self._phi_internal,
                components_dict=self._components,
                multisteps=self._general.get("multisteps", False),
                smoother=self.smoother,
                adam_ets=(self.ets == "adam"),
                other=other_value,
                other_parameter_estimate=other_parameter_estimate,
                fi=self.fi,
                step_size=self.step_size,
                **nlopt_params,
            )
            # Extract adam_cpp from estimation results
            self._adam_cpp = self._adam_estimated["adam_cpp"]

            # Store back estimated gnorm shape
            if other_parameter_estimate and "B" in self._adam_estimated:
                self.gnorm_shape = float(abs(self._adam_estimated["B"][-1]))

        # Build the model structure - architector() returns 6 values including
        # adam_cpp, but we already have adam_cpp from estimation
        (
            self._model_type,
            self._components,
            self._lags_model,
            self._observations,
            self._profile,
            _,  # adam_cpp - already stored from estimation result
        ) = architector(
            model_type_dict=self._model_type,
            lags_dict=self._lags_model,
            observations_dict=self._observations,
            arima_checked=self._arima,
            constants_checked=self._constant,
            explanatory_checked=self._explanatory,
            profiles_recent_table=self.profiles_recent_table,
            profiles_recent_provided=self.profiles_recent_provided,
            adam_ets=(self.ets == "adam"),
        )
        # print(self._components)
        # Create the model matrices
        self._adam_created = creator(
            model_type_dict=self._model_type,
            lags_dict=self._lags_model,
            profiles_dict=self._profile,
            observations_dict=self._observations,
            persistence_checked=self._persistence,
            initials_checked=self._initials,
            arima_checked=self._arima,
            constants_checked=self._constant,
            phi_dict=self._phi_internal,
            components_dict=self._components,
            explanatory_checked=self._explanatory,
            smoother=self.smoother,
        )

        # Calculate information criterion
        if estimation:
            self._ic_selection = ic_function(
                self._general["ic"], self._adam_estimated["log_lik_adam_value"]
            )

        # Update parameters number
        self._update_parameters_number(self._adam_estimated["n_param_estimated"])

    def _update_parameters_number(self, n_param_estimated):
        """
        Update the parameters number in the general dictionary.

        Parameters
        ----------
        n_param_estimated : int
            Number of estimated parameters from optimization
        """
        # Store number of estimated parameters
        self._n_param_estimated = n_param_estimated

        # Skip updating n_param for combined models
        # (already set in _execute_combination)
        if getattr(self, "_is_combined", False):
            # Legacy format for backward compatibility
            if "parameters_number" not in self._general:
                self._general["parameters_number"] = self._params_info.get(
                    "parameters_number", [[0], [0]]
                )
            return

        # Update the n_param table
        if "n_param" in self._general:
            n_param = self._general["n_param"]
            # n_param_estimated = len(B) = internal_in_B + xreg_in_B
            # Subtract xreg (correctly set by build_n_param_table) to get internal
            n_param.estimated["internal"] = (
                n_param_estimated - n_param.estimated["xreg"]
            )

            # Handle likelihood loss case - scale parameter is estimated
            if self._general["loss"] == "likelihood":
                n_param.estimated["scale"] = 1
            else:
                n_param.estimated["scale"] = 0

            # Update totals
            n_param.update_totals()

            # Store reference for easy access
            self._n_param = n_param

        # Legacy format for backward compatibility
        if "parameters_number" not in self._general:
            self._general["parameters_number"] = self._params_info.get(
                "parameters_number", [[0], [0]]
            )
        self._general["parameters_number"][0][0] = n_param_estimated

        # Handle likelihood loss case in legacy format
        if self._general["loss"] == "likelihood":
            if len(self._general["parameters_number"][0]) <= 3:
                self._general["parameters_number"][0].append(1)
            else:
                self._general["parameters_number"][0][3] = 1

        # Calculate row sums in legacy format
        if len(self._general["parameters_number"][0]) <= 4:
            self._general["parameters_number"][0].append(
                sum(self._general["parameters_number"][0][0:4])
            )
            self._general["parameters_number"][1].append(
                sum(self._general["parameters_number"][1][0:4])
            )
        else:
            self._general["parameters_number"][0][4] = sum(
                self._general["parameters_number"][0][0:4]
            )
            self._general["parameters_number"][1][4] = sum(
                self._general["parameters_number"][1][0:4]
            )

    def _execute_selection(self):
        """
        Execute model selection when model_do is 'select'.

        This handles model selection and creation of the selected model.
        """
        # Run model selection
        self._adam_selected = selector(
            model_type_dict=self._model_type,
            phi_dict=self._phi_internal,
            general_dict=self._general,
            lags_dict=self._lags_model,
            observations_dict=self._observations,
            arima_dict=self._arima,
            constant_dict=self._constant,
            explanatory_dict=self._explanatory,
            occurrence_dict=self._occurrence,
            components_dict=self._components,
            profiles_recent_table=self.profiles_recent_table,
            profiles_recent_provided=self.profiles_recent_provided,
            persistence_results=self._persistence,
            initials_results=self._initials,
            criterion=self._general["ic"],
            silent=self.verbose == 0,
            nlopt_kwargs=self.nlopt_kwargs,
            smoother=self.smoother,
        )
        # print(self._adam_selected)
        # print(self._adam_selected["ic_selection"])

        # Updates parametes with the selected model and updates adam_estimated
        self.select_best_model()

        # print(self._adam_selected["ic_selection"])
        # Process each selected model
        # The following commented-out loop and its associated helper method
        # calls (_update_model_from_selection, _create_matrices_for_selected_model,
        # _update_parameters_for_selected_model) appear to be placeholders
        # or remnants of a "combine" functionality that is not fully implemented
        # yet, as indicated by the NotImplementedError in the fit method for
        # model_do="combine". These will be kept for now as they might be
        # relevant for future development.
        # for i, result in enumerate(self._adam_selected["results"]):
        #     # Update model parameters with the selected model
        #     self._update_model_from_selection(i, result)
        #
        #     # Create matrices for this model
        #     self._create_matrices_for_selected_model(i)
        #
        #     # Update parameters number for this model
        #     self._update_parameters_for_selected_model(i, result)

    def select_best_model(self):
        """
        Select the best model based on information criteria and update model parameters.
        """
        self._ic_selection = self._adam_selected["ic_selection"]
        results = self._adam_selected["results"]
        # Find best model
        self._best_model = min(self._ic_selection.items(), key=lambda x: x[1])[0]
        best_id = next(
            i for i, result in enumerate(results) if result["model"] == self._best_model
        )
        # Update dictionaries with best model results
        self._model_type = results[best_id]["model_type_dict"]
        self._phi_internal = results[best_id]["phi_dict"]
        self._adam_estimated = results[best_id]["adam_estimated"]
        self._adam_cpp = self._adam_estimated["adam_cpp"]

    def _execute_combination(self):
        """
        Execute model combination using IC weights.

        Combines fitted values from multiple models using Akaike weights derived from
        information criterion values. Each model's contribution is proportional to
        its relative likelihood of being the best model.

        The combined fitted values are IC-weighted averages across all models.
        """
        import copy

        # Get IC weights from selection results
        self._ic_weights = calculate_ic_weights(self._ic_selection)
        results = self._adam_selected["results"]

        # Compute filtered weights (>= 0.01) for fitted values calculation only
        # Full model set is stored; filtering happens at predict-time
        filtered_weights = {k: v for k, v in self._ic_weights.items() if v >= 0.01}
        total_filtered = sum(filtered_weights.values())
        if total_filtered > 0:
            filtered_weights = {
                k: v / total_filtered for k, v in filtered_weights.items()
            }

        # Initialize combined fitted values
        obs_in_sample = self._observations["obs_in_sample"]
        y_fitted_combined = np.zeros(obs_in_sample)
        n_param_weighted = 0.0

        # Store ALL models for later forecasting (filtering happens at predict-time)
        self._prepared_models = []

        for result in results:
            model_name = result["model"]
            original_weight = self._ic_weights.get(model_name, 0)
            filtered_weight = filtered_weights.get(model_name, 0)

            # Make copies of dicts that get modified by architector
            lags_dict_copy = copy.deepcopy(self._lags_model)
            observations_dict_copy = copy.deepcopy(self._observations)
            model_type_dict = result["model_type_dict"].copy()
            phi_dict = result["phi_dict"].copy()

            # Call architector to get components for this model
            (
                model_type_dict,
                components_dict,
                lags_dict_copy,
                observations_dict_copy,
                profile_dict,
                _,
            ) = architector(
                model_type_dict=model_type_dict,
                lags_dict=lags_dict_copy,
                observations_dict=observations_dict_copy,
                arima_checked=self._arima,
                constants_checked=self._constant,
                explanatory_checked=self._explanatory,
                profiles_recent_table=self.profiles_recent_table,
                profiles_recent_provided=self.profiles_recent_provided,
                adam_ets=(self.ets == "adam"),
            )

            # Call creator to build matrices
            adam_created = creator(
                model_type_dict=model_type_dict,
                lags_dict=lags_dict_copy,
                profiles_dict=profile_dict,
                observations_dict=observations_dict_copy,
                persistence_checked=self._persistence,
                initials_checked=self._initials,
                arima_checked=self._arima,
                constants_checked=self._constant,
                phi_dict=phi_dict,
                components_dict=components_dict,
                explanatory_checked=self._explanatory,
                smoother=self.smoother,
            )

            # Make copy of general_dict for preparator
            general_dict_copy = copy.deepcopy(self._general)

            # Ensure distribution_new is set (needed by preparator/scaler)
            if "distribution_new" not in general_dict_copy:
                if general_dict_copy.get("distribution") == "default":
                    if general_dict_copy.get("loss") == "likelihood":
                        if model_type_dict.get("error_type") == "M":
                            general_dict_copy["distribution_new"] = "dgamma"
                        else:
                            general_dict_copy["distribution_new"] = "dnorm"
                    else:
                        general_dict_copy["distribution_new"] = "dnorm"
                else:
                    general_dict_copy["distribution_new"] = general_dict_copy.get(
                        "distribution", "dnorm"
                    )

            # Call preparator to get fitted values
            prepared = preparator(
                model_type_dict=model_type_dict,
                components_dict=components_dict,
                lags_dict=lags_dict_copy,
                matrices_dict=adam_created,
                persistence_checked=self._persistence,
                initials_checked=self._initials,
                arima_checked=self._arima,
                explanatory_checked=self._explanatory,
                phi_dict=phi_dict,
                constants_checked=self._constant,
                observations_dict=observations_dict_copy,
                occurrence_dict=self._occurrence,
                general_dict=general_dict_copy,
                profiles_dict=profile_dict,
                adam_estimated=result["adam_estimated"],
                adam_cpp=result["adam_estimated"]["adam_cpp"],
            )

            # Get fitted values and handle NaN
            fitted_values = np.asarray(prepared["y_fitted"]).flatten()
            fitted_values = np.nan_to_num(fitted_values, nan=0.0)

            # Add IC-weighted contribution (using filtered weight for fitted values)
            y_fitted_combined += fitted_values * filtered_weight
            n_param_weighted += (
                result["adam_estimated"]["n_param_estimated"] * filtered_weight
            )

            # Store for later forecasting
            # (using ORIGINAL weight - filtering at predict-time)
            self._prepared_models.append(
                {
                    "name": model_name,
                    "weight": original_weight,
                    "result": result,
                    "model_type_dict": model_type_dict,
                    "components_dict": components_dict,
                    "lags_dict": lags_dict_copy,
                    "observations_dict": observations_dict_copy,
                    "profile_dict": profile_dict,
                    "phi_dict": phi_dict,
                    "adam_created": adam_created,
                    "prepared": prepared,
                    "explanatory_dict": self._explanatory,
                    "constants_dict": self._constant,
                    "n_param_estimated": result["adam_estimated"]["n_param_estimated"],
                }
            )

        # Store combined results
        self._combined_fitted = y_fitted_combined
        self._combined_residuals = (
            np.array(self._observations["y_in_sample"]) - y_fitted_combined
        )

        # Create NParam with weighted average internal params + scale
        n_param = NParam()
        n_param.estimated["internal"] = n_param_weighted
        if self._general.get("loss") == "likelihood":
            n_param.estimated["scale"] = 1
        n_param.update_totals()
        self._n_param = n_param
        self._general["n_param"] = n_param
        self._n_param_combined = n_param.estimated["all"]  # Backward compat

        # Mark as combined model
        self._is_combined = True

        # Use the best model as the primary model for state-space matrices
        self.select_best_model()

    def _update_model_from_selection(self, index, result):
        """
        Update model parameters with the selected model parameters.

        Parameters
        ----------
        index : int
            Index of the selected model in the results list from model selection.
        result : dict
            The dictionary containing all parameters and estimation results
            for the selected model.
        """
        # Update global dictionaries with the selected model info
        self._general.update(result["general"])
        self._model_type.update(result["model_type_dict"])
        self._arima.update(result["arima_dict"])
        self._constant.update(result["constant_dict"])
        self._persistence.update(result["persistence_dict"])
        self._initials.update(result["initials_dict"])
        self._phi_internal.update(result["phi_dict"])
        self._components.update(result["components_dict"])
        self._lags_model.update(result["lags_dict"])
        self._observations.update(result["observations_dict"])
        self._profile = result.get(
            "profile_dict",
            {
                "profiles_recent_provided": self.profiles_recent_provided,
                "profiles_recent_table": self.profiles_recent_table,
            },
        )

        # Store the estimated model and adam_cpp
        self._adam_estimated = result["adam_estimated"]
        self._adam_cpp = self._adam_estimated["adam_cpp"]

    def _create_matrices_for_selected_model(self, index):
        """
        Create matrices for a selected model. This is typically used when iterating
        through models in a selection process, particularly for a "combine" feature.

        Parameters
        ----------
        index : int
            Index of the selected model.
        """
        # Create matrices for this model
        self._adam_created = creator(
            model_type_dict=self._model_type,
            lags_dict=self._lags_model,
            profiles_dict=self._profile,
            observations_dict=self._observations,
            persistence_checked=self._persistence,
            initials_checked=self._initials,
            arima_checked=self._arima,
            constants_checked=self._constant,
            phi_dict=self._phi_internal,
            components_dict=self._components,
            explanatory_checked=self._explanatory,
            smoother=self.smoother,
        )

        # Store created matrices
        self._adam_selected["results"][index]["adam_created"] = self._adam_created

    def _update_parameters_for_selected_model(self, index, result):
        """
        Update parameters number for a selected model.

        This is typically used when iterating through models in a selection
        process, particularly for a "combine" feature.

        Parameters
        ----------
        index : int
            Index of the selected model.
        result : dict
            Selected model result containing `adam_estimated` which has
            `n_param_estimated`.
        """
        # Update parameters number
        n_param_estimated = result["adam_estimated"]["n_param_estimated"]

        # Update n_param table if available
        if "n_param" in self._general and self._general["n_param"] is not None:
            n_param = self._general["n_param"]
            n_param.estimated["internal"] = n_param_estimated

            if self._general["loss"] == "likelihood":
                n_param.estimated["scale"] = 1
            else:
                n_param.estimated["scale"] = 0

            n_param.update_totals()
            self._n_param = n_param

        # Legacy format
        self._general["parameters_number"] = self._params_info.get(
            "parameters_number", [[0], [0]]
        )
        self._general["parameters_number"][0][0] = n_param_estimated

        # Handle likelihood loss case
        if self._general["loss"] == "likelihood":
            if len(self._general["parameters_number"][0]) <= 3:
                self._general["parameters_number"][0].append(1)
            else:
                self._general["parameters_number"][0][3] = 1

        # Calculate row sums
        if len(self._general["parameters_number"][0]) <= 4:
            self._general["parameters_number"][0].append(
                sum(self._general["parameters_number"][0][0:4])
            )
            self._general["parameters_number"][1].append(
                sum(self._general["parameters_number"][1][0:4])
            )
        else:
            self._general["parameters_number"][0][4] = sum(
                self._general["parameters_number"][0][0:4]
            )
            self._general["parameters_number"][1][4] = sum(
                self._general["parameters_number"][1][0:4]
            )

        # Store parameters number
        self._adam_selected["results"][index]["parameters_number"] = self._general[
            "parameters_number"
        ]

    def _prepare_results(self):
        """
        Prepare final results and format output data.

        This transforms data into appropriate formats and handles distribution
        selection.
        """
        # Transform data into appropriate classes
        self._format_time_series_data()

        # Handle distribution selection
        self._select_distribution()

        # Compute fitted values and residuals by calling preparator
        self._compute_fitted_values()

    def _format_time_series_data(self):
        """
        Format time series data into pandas Series with appropriate indexes.
        """
        if isinstance(self._observations["y_in_sample"], np.ndarray):
            # Check if frequency is a valid pandas frequency (not just "1" string)
            freq = self._observations.get("frequency", "1")
            try:
                # Try to use date_range if frequency looks valid
                if freq != "1" and isinstance(
                    self._observations.get("y_start"), (pd.Timestamp, str)
                ):
                    self._y_in_sample = pd.Series(
                        self._observations["y_in_sample"],
                        index=pd.date_range(
                            start=self._observations["y_start"],
                            periods=len(self._observations["y_in_sample"]),
                            freq=freq,
                        ),
                    )
                else:
                    # Use simple range index for non-datetime data
                    self._y_in_sample = pd.Series(self._observations["y_in_sample"])
            except (ValueError, TypeError):
                # Fallback to simple range index if date_range fails
                self._y_in_sample = pd.Series(self._observations["y_in_sample"])

            if self._general["holdout"]:
                try:
                    if freq != "1" and isinstance(
                        self._observations.get("y_forecast_start"),
                        (pd.Timestamp, str),
                    ):
                        self._y_holdout = pd.Series(
                            self._observations["y_holdout"],
                            index=pd.date_range(
                                start=self._observations["y_forecast_start"],
                                periods=len(self._observations["y_holdout"]),
                                freq=freq,
                            ),
                        )
                    else:
                        self._y_holdout = pd.Series(self._observations["y_holdout"])
                except (ValueError, TypeError):
                    self._y_holdout = pd.Series(self._observations["y_holdout"])
        else:
            self._y_in_sample = self._observations["y_in_sample"].copy()
            if self._general["holdout"]:
                self._y_holdout = pd.Series(
                    self._observations["y_holdout"],
                    index=self._observations["y_forecast_index"],
                )

    def _select_distribution(self):
        """
        Select appropriate distribution based on model and loss function.

        Distribution mapping:
        - likelihood: dnorm (additive) or dgamma (multiplicative)
        - MAE-based losses: dlaplace
        - HAM-based losses: ds
        - MSE-based and other losses: dnorm
        """
        if self._general["distribution"] == "default":
            loss = self._general["loss"]
            if loss == "likelihood":
                if self._model_type["error_type"] == "A":
                    self._general["distribution_new"] = "dnorm"
                elif self._model_type["error_type"] == "M":
                    self._general["distribution_new"] = "dgamma"
            elif loss in [
                "MAE",
                "MAEh",
                "TMAE",
                "GTMAE",
                "MACE",
                "aTMAE",
                "aGTMAE",
                "aMACE",
            ]:
                self._general["distribution_new"] = "dlaplace"
            elif loss in [
                "HAM",
                "HAMh",
                "THAM",
                "GTHAM",
                "CHAM",
                "aTHAM",
                "aGTHAM",
                "aCHAM",
            ]:
                self._general["distribution_new"] = "ds"
            elif loss in [
                "MSE",
                "MSEh",
                "TMSE",
                "GTMSE",
                "MSCE",
                "GPL",
                "aMSEh",
                "aTMSE",
                "aGTMSE",
                "aMSCE",
                "aGPL",
                "LASSO",
                "RIDGE",
                "custom",
            ]:
                self._general["distribution_new"] = "dnorm"
            else:
                # Fallback to dnorm for any unrecognized loss
                self._general["distribution_new"] = "dnorm"
        else:
            self._general["distribution_new"] = self._general["distribution"]

    def _compute_fitted_values(self):
        """
        Compute fitted values and residuals after model estimation.

        This calls preparator() to run the fitter with final parameters
        and extract fitted values, residuals, and scale.
        """
        # Set a default h if not provided (needed for preparator)
        if self.h is None:
            if self._lags_model and len(self._lags_model["lags"]) > 0:
                self._general["h"] = max(self._lags_model["lags"])
            else:
                self._general["h"] = 10

        # Call preparator to compute fitted values and residuals
        self._prepared = preparator(
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
            occurrence_dict=self._occurrence,
            general_dict=self._general,
            profiles_dict=self._profile,
            adam_estimated=self._adam_estimated,
            adam_cpp=self._adam_cpp,
            bounds="usual",
            # The current dgnorm shape, so the gradient initial-state solve in
            # preparator() profiles the same loss the estimation used
            other=getattr(self, "gnorm_shape", None),
        )

    def _auto_predict(self):
        """Run point forecasts automatically if h > 0; compute accuracy if holdout."""
        h = getattr(self, "h", None)
        if not h or h <= 0:
            return

        self._general["h"] = h
        self._prepare_prediction_data()

        if getattr(self, "_om_model", None) is not None:
            occ_fc = self._om_model.predict(h=h)
            self._occurrence["p_forecast"] = np.asarray(occ_fc.mean.values, dtype=float)

        auto_fc = forecaster(
            model_prepared=self._prepared,
            observations_dict=self._observations,
            general_dict=self._general,
            occurrence_dict=self._occurrence,
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
        self._auto_forecast = auto_fc

        if not getattr(self, "holdout", False):
            return
        y_holdout = self._observations.get("y_holdout")
        y_in_sample = self._observations.get("y_in_sample")
        if y_holdout is None or len(y_holdout) == 0:
            return

        from smooth.adam_general.core.utils.printing import _compute_forecast_errors

        fc_values = np.asarray(auto_fc.mean, dtype=float).ravel()
        y_holdout_arr = np.asarray(y_holdout, dtype=float).ravel()
        n = min(len(fc_values), len(y_holdout_arr))
        # ``lags`` can be an empty list (e.g. non-seasonal SMA), so fall back to 1
        period = max(self._lags_model.get("lags") or [1]) if self._lags_model else 1
        self.accuracy = _compute_forecast_errors(
            y_holdout_arr[:n],
            fc_values[:n],
            np.asarray(y_in_sample, dtype=float),
            period,
        )

    def _validate_prediction_inputs(self):
        """
        Validate that the model is properly fitted before prediction.

        Raises
        ------
        ValueError
            If the model has not been fitted yet or is missing required components.
        """
        # Verify that the model has been fitted
        if not hasattr(self, "_model_type"):
            raise ValueError("Model must be fitted before prediction.")

        # Check if we have the necessary components based on the model type
        if self._model_type["model_do"] == "estimate" and not hasattr(
            self, "_adam_estimated"
        ):
            raise ValueError("Model estimation results not found.")
        elif self._model_type["model_do"] == "select" and not hasattr(
            self, "_adam_selected"
        ):
            raise ValueError("Model selection results not found.")

    def _prepare_prediction_data(self):
        """
        Prepare data for prediction by setting up necessary matrices and parameters.
        """
        # If h wasn't provided, use default h
        if self.h is None:
            if self._lags_model and len(self._lags_model["lags"]) > 0:
                self.h = max(self._lags_model["lags"])

            else:
                self.h = 10
                self._general["h"] = self.h

        # Prepare necessary data for forecasting
        self._prepared = preparator(
            # Model info
            model_type_dict=self._model_type,
            # Components info
            components_dict=self._components,
            # Lags info
            lags_dict=self._lags_model,
            # Matrices from creator
            matrices_dict=self._adam_created,
            # Parameter dictionaries
            persistence_checked=self._persistence,
            initials_checked=self._initials,
            arima_checked=self._arima,
            explanatory_checked=self._explanatory,
            phi_dict=self._phi_internal,
            constants_checked=self._constant,
            # Other parameters
            observations_dict=self._observations,
            occurrence_dict=self._occurrence,
            general_dict=self._general,
            profiles_dict=self._profile,
            # The parameter vector
            adam_estimated=self._adam_estimated,
            # adamCore C++ object
            adam_cpp=self._adam_cpp,
            # Optional parameters
            bounds="usual",
            # The current dgnorm shape, so the gradient initial-state solve in
            # preparator() profiles the same loss the estimation used
            other=getattr(self, "gnorm_shape", None),
        )

    def _execute_prediction(
        self,
        interval="prediction",
        level=0.95,
        side="both",
    ):
        """
        Execute the forecasting process based on the prepared data.

        For combined models, runs forecaster for each model with non-zero
        IC weight and combines the results.

        Returns
        -------
        ForecastResult
            Forecast results including point forecasts and prediction intervals.
        """
        # Handle combined models
        if getattr(self, "_is_combined", False):
            return self._execute_prediction_combined(
                interval=interval,
                level=level,
                side=side,
            )

        # Standard single-model prediction
        self._forecast_results = forecaster(
            model_prepared=self._prepared,
            observations_dict=self._observations,
            general_dict=self._general,
            occurrence_dict=self._occurrence,
            lags_dict=self._lags_model,
            model_type_dict=self._model_type,
            explanatory_checked=self._explanatory,
            components_dict=self._components,
            constants_checked=self._constant,
            params_info=self._params_info,
            adam_cpp=self._adam_cpp,
            interval=interval,
            level=level,
            side=side,
        )
        return self._forecast_results

    def _execute_prediction_combined(
        self,
        interval="prediction",
        level=0.95,
        side="both",
    ):
        """
        Execute combined forecasting using IC-weighted model averaging.

        Generates forecasts from all models with non-zero IC weights and
        combines them using Akaike weights.

        Returns
        -------
        ForecastResult
            IC-weighted combined forecast results.
        """
        from smooth.adam_general.core.forecaster import forecaster_combined

        self._forecast_results = forecaster_combined(
            prepared_models=self._prepared_models,
            ic_weights=self._ic_weights,
            observations_dict=self._observations,
            general_dict=self._general,
            occurrence_dict=self._occurrence,
            params_info=self._params_info,
            interval=interval,
            level=level,
            side=side,
        )
        return self._forecast_results

    def _format_prediction_results(self):
        """
        Format the prediction results into a more user-friendly structure.

        Note: This method is defined but not explicitly called within the ADAM
        class's current public interface (fit, predict, predict_intervals).

        Returns
        -------
        ForecastResult
            Formatted prediction results including point forecasts and intervals.
        """
        return self._forecast_results

    def __str__(self) -> str:
        """
        Return a formatted string representation of the fitted model.

        Returns a comprehensive summary including model type, parameters,
        information criteria, and forecast errors (if holdout was used).

        Returns
        -------
        str
            Formatted model summary
        """
        from smooth.adam_general.core.utils.printing import model_summary

        # Check if model has been fitted
        if not hasattr(self, "_model_type"):
            return f"ADAM(model={self.model}) - not fitted"

        return model_summary(self)

    def __repr__(self) -> str:
        """
        Return a string representation of the ADAM model.

        Returns
        -------
        str
            Brief model representation
        """
        if hasattr(self, "_model_type") and self._model_type:
            model_str = self._model_type.get("model", self.model)
            if self._model_type.get("ets_model", False):
                return f"ADAM(ETS({model_str}), fitted=True)"
            return f"ADAM({model_str}, fitted=True)"
        return f"ADAM(model={self.model}, fitted=False)"

    def _fisher_information_matrix(self, step_size=None):
        """Observed FI at the estimated coefficients (computed if not cached)."""
        from smooth.adam_general.core.utils.var_covar import fisher_information

        if step_size is None and self.fisher_information_ is not None:
            return self.fisher_information_
        return fisher_information(
            self.coef,
            self._model_type,
            self._components,
            self._lags_model,
            self._adam_created,
            self._persistence,
            self._initials,
            self._arima,
            self._explanatory,
            self._phi_internal,
            self._constant,
            self._observations,
            self._occurrence,
            self._general,
            self._profile,
            self._adam_cpp,
            step_size=step_size,
            other_parameter_estimate=self._adam_estimated.get(
                "other_parameter_estimate", False
            ),
        )

    def vcov(self, bootstrap=False, heuristics=None, step_size=None, **boot_kwargs):
        """Variance-covariance matrix of the estimated parameters.

        Mirrors R's ``vcov.adam``: inverts the observed Fisher Information,
        excluding non-informative ("broken") parameters and retrying with a
        larger finite-difference step if any are detected.

        Parameters
        ----------
        bootstrap : bool, default=False
            If True, delegate to :meth:`coefbootstrap` and return the
            empirical replicate covariance instead of the Fisher-based one.
        heuristics : float, optional
            If given, returns ``diag(abs(coef) * heuristics)``.
        step_size : float, optional
            Finite-difference step for the Fisher Information.
        **boot_kwargs
            Forwarded to :meth:`coefbootstrap` when ``bootstrap=True``
            (``nsim``, ``size``, ``replace``, ``method``, ``seed``, …).

        Returns
        -------
        pandas.DataFrame
            Covariance matrix indexed/columned by :attr:`coef_names`.
        """
        import pandas as pd

        from smooth.adam_general.core.utils.var_covar import invert_fisher_information

        self._check_is_fitted()
        names = self.coef_names

        if bootstrap:
            return self.coefbootstrap(**boot_kwargs).vcov

        if heuristics is not None:
            cov = np.diag(np.abs(np.asarray(self.coef, dtype=float)) * heuristics)
            return pd.DataFrame(cov, index=names, columns=names)

        FI = np.asarray(self._fisher_information_matrix(step_size=step_size))
        # R retries with a coarser step when variables look "broken".
        broken = np.all(FI == 0, axis=1) | np.any(np.isnan(FI), axis=1)
        if np.any(broken) and step_size is None:
            FI = np.asarray(
                self._fisher_information_matrix(
                    step_size=float(np.finfo(float).eps ** (1 / 6))
                )
            )

        cov = invert_fisher_information(FI)
        return pd.DataFrame(cov, index=names, columns=names)

    def _persistence_index(self, name):
        """vec_g row for a named persistence parameter (alpha/beta/gamma*/delta*)."""
        n_ets = self._components["components_number_ets"]
        n_nonseas = self._components["components_number_ets_non_seasonal"]
        n_arima = self._components.get("components_number_arima", 0)
        if name == "alpha":
            return 0
        if name == "beta":
            return 1
        if name.startswith("gamma"):
            digits = "".join(ch for ch in name if ch.isdigit())
            i = int(digits) - 1 if digits else 0
            return n_nonseas + i
        if name.startswith("delta"):
            digits = "".join(ch for ch in name if ch.isdigit())
            i = int(digits) - 1 if digits else 0
            return n_ets + n_arima + i
        raise ValueError(f"Unknown persistence parameter: {name}")

    def confint(
        self, parm=None, level=0.95, bootstrap=False, step_size=None, **boot_kwargs
    ):
        """Confidence intervals for the estimated parameters.

        Mirrors R's ``confint.adam``: standard errors from :meth:`vcov`,
        t-interval half-widths (with R's asymmetric degrees of freedom), then
        clamping to the admissible region for ETS smoothing parameters
        (``bounds="usual"`` or ``"admissible"``), multiplicative initial states,
        and ARIMA AR/MA parameters. With ``bootstrap=True`` the intervals are
        the empirical quantiles of the replicate matrix returned by
        :meth:`coefbootstrap` (no clamping / no t-quantile).

        Parameters
        ----------
        parm : str or sequence of str, optional
            Subset of names to return.
        level : float, default=0.95
            Confidence level.
        bootstrap : bool, default=False
            Switch to empirical-quantile intervals via :meth:`coefbootstrap`.
        step_size : float, optional
            Finite-difference step forwarded to :meth:`vcov` for the
            Fisher-based path. Ignored when ``bootstrap=True``.
        **boot_kwargs
            Forwarded to :meth:`coefbootstrap` (``nsim``, ``size``, …).

        Returns
        -------
        pandas.DataFrame
            Columns ``["S.E.", "<lo>%", "<hi>%"]`` indexed by :attr:`coef_names`.
        """
        import pandas as pd

        self._check_is_fitted()
        names = self.coef_names
        params = np.asarray(self.coef, dtype=float)

        if bootstrap:
            from smooth.adam_general.core.utils.bootstrap import (
                bootstrap_confint_frame,
            )

            boot = self.coefbootstrap(**boot_kwargs)
            return bootstrap_confint_frame(boot, names, params, level, parm)

        V = self.vcov(step_size=step_size).to_numpy()
        se = np.sqrt(np.abs(np.diag(V)))

        nobs = self.nobs
        nparam = self.nparam
        # R uses asymmetric degrees of freedom for the two tails.
        lo = scipy_stats.t.ppf((1 - level) / 2, df=nobs - nparam) * se
        hi = scipy_stats.t.ppf((1 + level) / 2, df=nobs + nparam) * se

        self._clamp_confint_offsets(names, params, lo, hi)

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

    def _eigen_static_args(self):
        """Static arguments forwarded to ``eigen_bounds``/``eigen_values``."""
        regressors = self._explanatory.get("regressors") or self._general.get(
            "regressors", "use"
        )
        xreg_model = self._explanatory.get("xreg_model", False)
        transition = self._prepared.get("transition", self._adam_created["mat_f"])
        measurement = self._prepared.get("measurement", self._adam_created["mat_wt"])
        return dict(
            transition=np.asarray(transition, dtype=float),
            measurement=np.asarray(measurement, dtype=float),
            lags_model_all=self._lags_model["lags_model_all"],
            xreg_model=xreg_model,
            obs_in_sample=self._observations["obs_in_sample"],
            has_delta=bool(xreg_model and regressors == "adapt"),
            xreg_number=self._explanatory.get("xreg_number", 0),
            constant_required=self._constant.get("constant_required", False),
        )

    def _clamp_confint_offsets(self, names, params, lo, hi):
        """Clamp confint half-width offsets to the admissible region in place.

        Mutates the ``lo`` / ``hi`` arrays so that ``param + lo`` and
        ``param + hi`` stay inside the feasible ETS/ARIMA region (mirrors the
        clamping block of R's ``confint.adam``). Extracted out of
        :meth:`confint` so :class:`~smooth.adam_general.core.omg.OMG` can apply
        the same per-sub-model clamping to slices of a joint CI half-width.
        """
        from smooth.adam_general.core.utils.bounds import (
            ar_polynomial_bounds,
            eigen_bounds,
        )

        idx = {name: i for i, name in enumerate(names)}
        bounds_type = self._general.get("bounds", "usual")
        ets_model = self._model_type.get("ets_model", False)
        arima_model = self._arima.get("arima_model", False)
        model_code = self.model_type

        if ets_model:
            vec_g = np.asarray(self._adam_created["vec_g"], dtype=float).ravel()
            static_args = self._eigen_static_args()

            def _eig(name):
                return eigen_bounds(vec_g, self._persistence_index(name), **static_args)

            if bounds_type == "usual":
                if "alpha" in idx:
                    a = idx["alpha"]
                    lo[a] = max(-params[a], lo[a])
                    hi[a] = min(1 - params[a], hi[a])
                if "beta" in idx:
                    b = idx["beta"]
                    lo[b] = max(-params[b], lo[b])
                    alpha_val = params[idx["alpha"]] if "alpha" in idx else vec_g[0]
                    hi[b] = min(alpha_val - params[b], hi[b])
                for name in [g for g in names if g.startswith("gamma")]:
                    gi = idx[name]
                    lo[gi] = max(-params[gi], lo[gi])
                    alpha_val = params[idx["alpha"]] if "alpha" in idx else vec_g[0]
                    hi[gi] = min((1 - alpha_val) - params[gi], hi[gi])
                for name in [d for d in names if d.startswith("delta")]:
                    di = idx[name]
                    lo[di] = max(-params[di], lo[di])
                    hi[di] = min(1 - params[di], hi[di])
                if "phi" in idx:
                    p = idx["phi"]
                    lo[p] = max(-params[p], lo[p])
                    hi[p] = min(1 - params[p], hi[p])
            elif bounds_type == "admissible":
                for name in names:
                    if (
                        name == "alpha"
                        or name == "beta"
                        or name.startswith(("gamma", "delta"))
                    ):
                        b1, b2 = _eig(name)
                        k = idx[name]
                        lo[k] = max(b1 - params[k], lo[k])
                        hi[k] = min(b2 - params[k], hi[k])

            # Multiplicative initial-state restrictions (both bounds >= -param).
            trend_mult = len(model_code) >= 2 and model_code[1] == "M"
            season_mult = len(model_code) >= 1 and model_code[-1] == "M"
            if trend_mult and "trend" in idx:
                t = idx["trend"]
                lo[t] = max(-params[t], lo[t])
                hi[t] = max(-params[t], hi[t])
            if season_mult:
                for name in [s for s in names if s.startswith("seasonal")]:
                    s = idx[name]
                    lo[s] = max(-params[s], lo[s])
                    hi[s] = max(-params[s], hi[s])

        if arima_model:
            self._clamp_arima_bounds(
                names, params, lo, hi, idx, eigen_bounds, ar_polynomial_bounds
            )

    def _clamp_arima_bounds(
        self, names, params, lo, hi, idx, eigen_bounds, ar_polynomial_bounds
    ):
        """Clamp ARIMA AR (phi*) and MA (theta*) CIs (R confint.adam:4544-4590)."""
        other = self._prepared.get("other", {})
        poly = other.get("polynomial", {})
        ari_polynomial = np.asarray(poly.get("ariPolynomial", []), dtype=float).ravel()
        ar_polynomial = np.asarray(poly.get("arPolynomial", []), dtype=float).ravel()
        non_zero_ari = np.atleast_2d(np.asarray(self._arima.get("non_zero_ari", [])))
        non_zero_ma = np.atleast_2d(np.asarray(self._arima.get("non_zero_ma", [])))
        ar_poly_matrix = other.get("ar_polynomial_matrix")
        n_ets = self._components["components_number_ets"]

        vec_g = np.asarray(self._adam_created["vec_g"], dtype=float).ravel()
        static_args = self._eigen_static_args()

        thetas = [nm for nm in names if nm.startswith("theta")]
        for i, nm in enumerate(thetas):
            k = idx[nm]
            psi_row = n_ets + int(non_zero_ma[i, 1])
            b1, b2 = eigen_bounds(vec_g, psi_row, **static_args)
            adj = 0.0
            if non_zero_ari.size and np.any(non_zero_ari[:, 1] == i):
                ari_index = np.where(non_zero_ari[:, 1] == i)[0][0]
                adj = ari_polynomial[int(non_zero_ari[ari_index, 0])]
            lo[k] = max(b1 - params[k] + adj, lo[k])
            hi[k] = min(b2 - params[k] + adj, hi[k])

        if ar_poly_matrix is not None and len(ar_polynomial) > 0:
            ar_mat = np.asarray(ar_poly_matrix, dtype=float)
            nonzero_pos = [j for j in range(len(ar_polynomial)) if ar_polynomial[j]]
            ar_positions = nonzero_pos[1:]  # drop the leading 1
            phis = [nm for nm in names if nm.startswith("phi") and len(nm) > 3]
            for i, nm in enumerate(phis):
                k = idx[nm]
                b1, b2 = ar_polynomial_bounds(ar_mat, ar_polynomial, ar_positions[i])
                lo[k] = max(b1 - params[k], lo[k])
                hi[k] = min(b2 - params[k], hi[k])

    def summary(self, level: float = 0.95, digits: int = 4):
        """
        Generate a coefficient-table summary of the fitted model.

        Mirrors R's ``summary.adam``: returns an object whose printed form is a
        table of estimates with standard errors, confidence intervals and
        significance marks, plus the error scale, sample size, parameter counts
        and information criteria. This is distinct from ``print(model)`` /
        ``str(model)``, which give the concise ``print.adam``-style report.

        Parameters
        ----------
        level : float, default=0.95
            Confidence level for the coefficient intervals.
        digits : int, default=4
            Number of decimal places for numeric output.

        Returns
        -------
        ADAMSummary
            Printable summary object (``print(model.summary())``).

        Examples
        --------
        >>> model = ADAM(model="AAN")
        >>> model.fit(y)
        >>> print(model.summary())
        """
        from smooth.adam_general.core.utils.printing import ADAMSummary

        self._check_is_fitted()
        return ADAMSummary(self, level=level, digits=digits)

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
        """Bootstrap the coefficient sampling distribution by refitting subsamples.

        Mirrors R's ``coefbootstrap.adam`` (R/adam.R:4850-5113). Draws ``nsim``
        case-resamples of the in-sample series, refits the same model on each,
        and returns the empirical covariance / replicate matrix. The dispatch
        in :meth:`vcov` and :meth:`confint` forwards ``bootstrap=True`` here.

        Parameters
        ----------
        nsim : int, default=1000
            Number of bootstrap replicates.
        size : int, optional
            Subsample size per replicate. Defaults to ``floor(0.75 * nobs)``,
            matching R.
        replace : bool, default=False
            Resample with replacement.
        prob : array-like, optional
            Sampling probabilities per observation (uniform if ``None``).
        parallel : bool or int, default=False
            ``True`` runs replicates in parallel via ``joblib.Parallel``
            (``cpu_count - 1`` workers). An integer specifies the exact
            worker count. Requires the optional ``joblib`` dependency
            (``pip install joblib`` or ``pip install "smooth[parallel]"``);
            if ``joblib`` is not importable, a one-line warning is emitted
            and the call falls back to a serial loop.
        method : {"cr", "dsr"}, default="cr"
            ``"cr"`` is case resampling (the implemented path). ``"dsr"``
            (data-shape replication, R's ``greybox::dsrboot``) raises
            ``NotImplementedError``.
        seed : int, optional
            Seed for the :class:`numpy.random.Generator` used to draw indices.
            Makes the result reproducible (same indices in serial and
            parallel modes — the optimiser is deterministic given a fixed
            sample).
        verbose : bool, default=False
            Print a one-line progress message every 10% of replicates (in
            parallel mode, forwards ``verbose=10`` to ``joblib.Parallel``).

        Returns
        -------
        smooth.adam_general.core.utils.bootstrap.BootstrapResult
            Container with ``.vcov``, ``.coefficients``, ``.method``,
            ``.nsim``, ``.nsim_effective``, ``.size``, ``.time_elapsed``, …
            Mirrors R's ``"bootstrap"`` S3 class. ``.parallel`` reflects
            whether parallel execution actually ran (``False`` when we
            fell back due to a missing ``joblib``).

        Notes
        -----
        Replicates whose refit fails (non-convergence, mismatched parameter
        count) are dropped silently; ``result.nsim_effective`` reports the
        count that contributed to the variance estimate.

        Bootstrap on a model with external regressors (``X``) is not yet
        supported and will raise.

        Examples
        --------
        >>> m = ADAM(model="ANN").fit(y)
        >>> b = m.coefbootstrap(nsim=50, seed=42)
        >>> b.vcov.shape
        (1, 1)
        """
        import time as _time
        from functools import partial

        from smooth.adam_general.core.utils.bootstrap import (
            _build_result,
            case_resample_indices,
            run_replicates,
            time_series_sample_indices,
        )

        self._check_is_fitted()

        if method not in ("cr", "dsr"):
            raise ValueError(f"method must be 'cr' or 'dsr', got {method!r}.")
        if method == "dsr":
            raise NotImplementedError(
                "method='dsr' requires greybox.dsrboot which is not yet "
                "available in Python; use method='cr'."
            )
        if self._explanatory.get("xreg_model", False):
            raise NotImplementedError(
                "coefbootstrap does not yet support models with external "
                "regressors (X). File an issue if you need this."
            )

        nobs = int(self.nobs)
        if size is None:
            size = max(int(np.floor(0.75 * nobs)), 1)
        size = int(size)
        if size < 2:
            raise ValueError(f"size={size} too small; need at least 2 observations.")

        rng = np.random.default_rng(seed)
        initial_type = (
            (self._initials or {}).get("initial_type")
            if hasattr(self, "_initials")
            else None
        )
        original_coef_names = list(self.coef_names)
        k = len(original_coef_names)
        model_spec = self._model_type.get("model", self.model)

        # R's sampler picks variable-length contiguous windows for
        # time-series models. ``size`` is honoured only on the
        # ``regressionPure`` path (none of the supported Python models
        # currently exercise that path — pure-regression ADAM is a future
        # feature). For now, always use the time-series sampler.
        lags = (
            self._lags_model.get("lags_model_all", [1])
            if hasattr(self, "_lags_model")
            else [1]
        )
        max_lag = int(np.max(lags)) if len(lags) else 1
        # Match R: obsMinimum = max(lags, nVariables) + 2.
        obs_minimum = max(max_lag, k) + 2
        if obs_minimum >= nobs:
            raise ValueError(
                f"Not enough observations to do case-resampling bootstrap "
                f"(obs_minimum={obs_minimum}, nobs={nobs}). R warns and "
                "falls back to method='dsr' here, which is not yet ported."
            )
        change_origin = initial_type in ("backcasting", "complete")
        if replace or prob is not None:
            # User explicitly asked for iid-style sampling; honour it.
            idx_list = list(case_resample_indices(nobs, size, nsim, replace, prob, rng))
        else:
            idx_list = time_series_sample_indices(
                nobs, nsim, obs_minimum, change_origin, rng
            )

        # OM has its own ``_config`` shape (already includes ``model`` and
        # the occurrence flag) and must round-trip through ``OM.fit`` to
        # rebuild the occurrence machinery — refitting as a plain ADAM
        # would ignore the occurrence model. ADAM (and ES/MSARIMA, which
        # inherit) use the plain ADAM path.
        from smooth.adam_general.core.om import OM

        if isinstance(self, OM):
            refit_cls_name = "OM"
            refit_kwargs = {key: value for key, value in self._config.items()}
            include_model_kwarg = False  # already inside ``_config``
        else:
            refit_cls_name = "ADAM"
            refit_kwargs = {key: value for key, value in self._config.items()}
            include_model_kwarg = True
        refit_kwargs["holdout"] = False
        refit_kwargs.setdefault("verbose", 0)

        actuals = np.asarray(self.actuals, dtype=float)

        # ``functools.partial`` over the top-level worker is picklable —
        # joblib needs that. Same callable works in serial mode too.
        worker = partial(
            _adam_refit_one_replicate,
            actuals,
            idx_list,
            refit_cls_name,
            model_spec,
            refit_kwargs,
            k,
            include_model_kwarg,
        )

        t0 = _time.time()
        replicate_coefs, parallel_used = run_replicates(
            worker,
            nsim=nsim,
            parallel=parallel,
            verbose=verbose,
            label="coefbootstrap",
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
            model=str(model_spec),
            time_elapsed=elapsed,
        )

    def reapply(
        self,
        nsim: int = 1000,
        bootstrap: bool = False,
        heuristics: Optional[float] = None,
        seed: Optional[int] = None,
        **vcov_kwargs,
    ):
        """Re-run the model on the in-sample data for ``nsim`` parameter draws.

        Python port of R's ``reapply.adam`` (R/reapply.R:87-778). Samples
        ``nsim`` parameter vectors from a multivariate normal centred on
        :attr:`coef` with covariance from :meth:`vcov`, clips each draw to
        the admissible region, then re-runs the shared C++ ADAM kernel
        (``adamCore::reapply``) once per draw. The resulting per-draw
        fitted paths, states, transition / measurement matrices,
        persistence vectors, and final profiles are returned in a
        :class:`~smooth.adam_general.core.utils.reapply.ReapplyResult`
        with array shapes matching R exactly.

        Parameters
        ----------
        nsim : int, default=1000
            Number of parameter draws.
        bootstrap : bool, default=False
            Forwarded to :meth:`vcov`. ``True`` uses the empirical
            covariance from :meth:`coefbootstrap` instead of the analytical
            inverse-Fisher matrix.
        heuristics : float, optional
            Forwarded to :meth:`vcov` — heuristic diagonal proportion
            (``vcov = diag(|coef| * heuristics)``) when set.
        seed : int, optional
            Seed for the MVN sampler. Makes the draw reproducible.
        **vcov_kwargs
            Forwarded to :meth:`vcov` (``step_size``, bootstrap kwargs).

        Returns
        -------
        ReapplyResult
            Container with ``time_elapsed``, ``y``, ``states`` ``(c, n+L,
            nsim)``, ``refitted`` ``(n, nsim)``, ``fitted``, ``model``,
            ``transition`` ``(c, c, nsim)``, ``measurement`` ``(n, c,
            nsim)``, ``persistence`` ``(c, nsim)``, ``profile`` ``(c, L,
            nsim)``, ``random_parameters`` ``(nsim, k)`` and ``nsim``.

        Notes
        -----
        Covers ETS (with ``bounds="usual"``, ``"admissible"`` or
        ``"none"``) and pure / mixed ARIMA models. External regressors
        (``X``) are still rejected — that branch arrives in a follow-up.
        """
        import time as _time

        from smooth.adam_general.core.utils.reapply import ReapplyResult

        self._check_is_fitted()
        t0 = _time.time()

        # 1. Sampling covariance + PSD repair (R/reapply.R:95-115).
        # R forwards ``nsim=nsim`` to ``vcov`` when ``bootstrap=TRUE`` so
        # the bootstrap covariance uses the same replicate count as the
        # reapply MVN draw (R/reapply.R:95).
        vcov_call_kwargs = dict(vcov_kwargs)
        if bootstrap and "nsim" not in vcov_call_kwargs:
            vcov_call_kwargs["nsim"] = nsim
        vcov_df = self.vcov(
            bootstrap=bootstrap, heuristics=heuristics, **vcov_call_kwargs
        )
        coef = np.asarray(self.coef, dtype=float)
        coef_names = list(self.coef_names)
        vcov_arr = np.asarray(vcov_df, dtype=float)
        vcov_arr = _psd_correct(vcov_arr)

        # 2. MVN sample (R/reapply.R:251)
        rng = np.random.default_rng(seed)
        random_parameters = rng.multivariate_normal(mean=coef, cov=vcov_arr, size=nsim)

        # 3. ETS smoothing-parameter clipping (R/reapply.R:254-333).
        # ``bounds="usual"`` uses closed-form clamps; ``bounds="admissible"``
        # uses eigenvalue-derived bounds via the same ``eigen_bounds()``
        # helper that ``confint`` already relies on.
        idx = {nm: i for i, nm in enumerate(coef_names)}
        bounds_mode = self._general.get("bounds", "usual")
        ets_model = self._model_type.get("ets_model", True)
        if ets_model and bounds_mode == "usual":
            _clip_ets_usual_smoothing(random_parameters, idx)
        elif ets_model and bounds_mode == "admissible":
            from smooth.adam_general.core.utils.bounds import eigen_bounds

            vec_g_eig = np.asarray(self._adam_created["vec_g"], dtype=float).ravel()
            static_args = self._eigen_static_args()
            for nm in ("alpha", "beta"):
                if nm in idx:
                    lo, hi = eigen_bounds(
                        vec_g_eig, self._persistence_index(nm), **static_args
                    )
                    np.clip(
                        random_parameters[:, idx[nm]],
                        lo,
                        hi,
                        out=random_parameters[:, idx[nm]],
                    )
            for nm in [k for k in idx if k.startswith("gamma")]:
                lo, hi = eigen_bounds(
                    vec_g_eig, self._persistence_index(nm), **static_args
                )
                np.clip(
                    random_parameters[:, idx[nm]],
                    lo,
                    hi,
                    out=random_parameters[:, idx[nm]],
                )
            # phi (damping) stays in [0, 1] regardless of bounds mode.
            if "phi" in idx:
                np.clip(
                    random_parameters[:, idx["phi"]],
                    0.0,
                    1.0,
                    out=random_parameters[:, idx["phi"]],
                )
        # Multiplicative-state positivity check runs unconditionally
        # (R/reapply.R:324-333 — outside the bounds-mode switch).
        if ets_model:
            _clip_ets_multiplicative_states(random_parameters, idx, self._model_type)
        _clip_deltas(random_parameters, idx)

        # 3b. ARIMA parameter clipping (R/reapply.R:391-436).
        # ``theta`` (MA) bounds come from ``eigen_bounds`` on the psi row;
        # ``phi`` (AR) bounds come from ``ar_polynomial_bounds`` on the
        # companion matrix. When an ARI element is present for a given
        # theta, the bounds shift by ``ariPolynomial[nonZeroARI]`` so the
        # net ``theta - ariPolynomial`` lies inside the psi region.
        arima_model = self._model_type.get("arima_model", False)
        other_dict = (self._prepared or {}).get("other") or {}
        if arima_model:
            from smooth.adam_general.core.utils.bounds import (
                ar_polynomial_bounds,
                eigen_bounds,
            )

            poly = other_dict.get("polynomial", {}) or {}
            ari_polynomial = np.asarray(
                poly.get("ariPolynomial", poly.get("ari_polynomial", [])),
                dtype=float,
            ).ravel()
            ar_polynomial = np.asarray(
                poly.get("arPolynomial", poly.get("ar_polynomial", [])),
                dtype=float,
            ).ravel()
            non_zero_ari = np.atleast_2d(
                np.asarray(self._arima.get("non_zero_ari", []))
            )
            non_zero_ma = np.atleast_2d(np.asarray(self._arima.get("non_zero_ma", [])))
            ar_poly_matrix = other_dict.get("ar_polynomial_matrix")
            n_ets_arima_clip = self._components["components_number_ets"]
            vec_g_eig = np.asarray(self._adam_created["vec_g"], dtype=float).ravel()
            static_args = self._eigen_static_args()

            thetas = [nm for nm in coef_names if nm.startswith("theta")]
            for i, nm in enumerate(thetas):
                col = idx[nm]
                psi_row = n_ets_arima_clip + int(non_zero_ma[i, 1])
                lo, hi = eigen_bounds(vec_g_eig, psi_row, **static_args)
                adj = 0.0
                if non_zero_ari.size and np.any(non_zero_ari[:, 1] == i):
                    ari_index = np.where(non_zero_ari[:, 1] == i)[0][0]
                    adj = ari_polynomial[int(non_zero_ari[ari_index, 0])]
                np.clip(
                    random_parameters[:, col],
                    lo + adj,
                    hi + adj,
                    out=random_parameters[:, col],
                )

            if ar_poly_matrix is not None and len(ar_polynomial) > 0:
                ar_mat = np.asarray(ar_poly_matrix, dtype=float)
                nonzero_pos = [
                    pos for pos in range(len(ar_polynomial)) if ar_polynomial[pos]
                ]
                ar_positions = nonzero_pos[1:]  # drop the leading 1
                phis = [nm for nm in coef_names if nm.startswith("phi") and len(nm) > 3]
                for i, nm in enumerate(phis):
                    if i >= len(ar_positions):
                        break
                    col = idx[nm]
                    lo, hi = ar_polynomial_bounds(
                        ar_mat, ar_polynomial, ar_positions[i]
                    )
                    np.clip(
                        random_parameters[:, col],
                        lo,
                        hi,
                        out=random_parameters[:, col],
                    )

        # 4. Build the per-draw cubes (R/reapply.R:447-469)
        n = int(self.nobs)
        states_2d = np.asarray(self.states, dtype=np.float64)
        n_components, n_states_time = states_2d.shape
        L = int(self._lags_model["lags_model_max"])
        # R's $states is [obsInSample + lagsModelMax, nComponents]; Python's
        # ``self.states`` is its transpose, so we already have the cube-
        # friendly (c, n+L) layout.
        if n_states_time != n + L:
            # Older fits may have stored ``n + 1`` instead of ``n + L``;
            # pad / truncate so the C++ kernel sees the full grid.
            arr_vt_seed = np.zeros((n_components, n + L))
            cols = min(n_states_time, n + L)
            arr_vt_seed[:, :cols] = states_2d[:, :cols]
        else:
            arr_vt_seed = states_2d

        # Build cubes via ``np.zeros + slice assignment``. ``np.repeat`` +
        # ``.astype(order="F")`` produces an array whose stride metadata
        # carma's arma::cube view interprets in a way that triggers heap
        # corruption when chained across multiple reapply calls (sequential
        # ADAM models in one process). The zeros-then-fill pattern matches
        # the proven simulator path (``intervals.py:493-507``).
        arr_vt = np.zeros((n_components, n + L, nsim), order="F")
        for i in range(nsim):
            arr_vt[:, :, i] = arr_vt_seed
        transition_seed = np.asarray(self.transition, dtype=np.float64)
        arr_f = np.zeros(transition_seed.shape + (nsim,), order="F")
        for i in range(nsim):
            arr_f[:, :, i] = transition_seed
        measurement_seed = np.asarray(self.measurement, dtype=np.float64)
        arr_wt = np.zeros(measurement_seed.shape + (nsim,), order="F")
        for i in range(nsim):
            arr_wt[:, :, i] = measurement_seed

        vec_g_seed = np.asarray(
            self._prepared.get("vec_g", self._adam_created["vec_g"]),
            dtype=np.float64,
        ).ravel()
        if len(vec_g_seed) != n_components:
            # Pad with zeros (failsafe for xreg-with-no-deltas case, R 132-135)
            vec_g_seed = np.concatenate(
                [vec_g_seed, np.zeros(n_components - len(vec_g_seed))]
            )
        mat_g = np.zeros((n_components, nsim), order="F")
        for i in range(nsim):
            mat_g[:, i] = vec_g_seed

        # 5. Fill cubes from random_parameters (R/reapply.R:471-743).
        #    Phase 1: ETS-only. ``k`` mirrors R's column-stride counter.
        k = 0
        ets_model = self._model_type.get("ets_model", True)
        if ets_model:
            if "alpha" in idx:
                mat_g[0, :] = random_parameters[:, idx["alpha"]]
                k += 1
            if "beta" in idx:
                mat_g[1, :] = random_parameters[:, idx["beta"]]
                k += 1
            gamma_names = [nm for nm in coef_names if nm.startswith("gamma")]
            if gamma_names:
                n_nonseas = self._components["components_number_ets_non_seasonal"]
                for j, nm in enumerate(gamma_names):
                    mat_g[n_nonseas + j, :] = random_parameters[:, idx[nm]]
                k += len(gamma_names)
            if "phi" in idx:
                phi_vals = random_parameters[:, idx["phi"]]
                arr_f[0, 1, :] = phi_vals
                arr_f[1, 1, :] = phi_vals
                arr_wt[:, 1, :] = phi_vals[np.newaxis, :]
                k += 1

        # 5b. ARIMA polynomial fill into arr_f and mat_g (R/reapply.R:554-634).
        # For each parameter draw, call ``polynomialise`` to expand the
        # sampled phi / theta vector into AR / I / ARI / MA polynomials,
        # then write the relevant entries into ``arr_f`` / ``mat_g``.
        ar_orders_padded: list = []
        i_orders_padded: list = []
        ma_orders_padded: list = []
        lags_arima: list = []
        arma_params_arr: NDArray = np.zeros(0, dtype=float)
        ar_estimate = False
        ma_estimate = False
        poly_index = -1
        n_arma = 0
        if arima_model:
            ar_orders_padded = list(self._arima["ar_orders"])
            i_orders_padded = list(self._arima["i_orders"])
            ma_orders_padded = list(self._arima["ma_orders"])
            # Mirror filler.py's lookup: ``lags_original`` is the truth for
            # the polynomialise call; ``lags`` in ``_lags_model`` may be
            # the empty / expanded form depending on model layout.
            lags_arima = list(
                self._lags_model.get("lags_original")
                or self._lags_model.get("lags")
                or [1]
            )
            arma_params_arr = np.asarray(
                self._arima.get("arma_parameters") or [], dtype=float
            ).ravel()

            max_order = max(
                len(ar_orders_padded),
                len(i_orders_padded),
                len(ma_orders_padded),
                len(lags_arima),
            )
            ar_orders_padded += [0] * (max_order - len(ar_orders_padded))
            i_orders_padded += [0] * (max_order - len(i_orders_padded))
            ma_orders_padded += [0] * (max_order - len(ma_orders_padded))
            if len(lags_arima) != max_order:
                lags_new = lags_arima + [0] * (max_order - len(lags_arima))
                ar_orders_padded = [
                    a for a, lv in zip(ar_orders_padded, lags_new) if lv != 0
                ]
                i_orders_padded = [
                    a for a, lv in zip(i_orders_padded, lags_new) if lv != 0
                ]
                ma_orders_padded = [
                    a for a, lv in zip(ma_orders_padded, lags_new) if lv != 0
                ]

            ar_estimate = any(nm.startswith("phi") and len(nm) > 3 for nm in coef_names)
            ma_estimate = any(nm.startswith("theta") for nm in coef_names)
            n_arma = sum(o for o in ar_orders_padded) * int(ar_estimate) + sum(
                o for o in ma_orders_padded
            ) * int(ma_estimate)

            if ar_estimate or ma_estimate:
                phi_pos = [
                    i
                    for i, nm in enumerate(coef_names)
                    if nm.startswith("phi") and len(nm) > 3
                ]
                theta_pos = [
                    i for i, nm in enumerate(coef_names) if nm.startswith("theta")
                ]
                candidates = phi_pos + theta_pos
                poly_index = min(candidates) - 1 if candidates else -1

            from smooth.adam_general.core.utils.polynomials import (
                adam_polynomialiser,
            )

            n_ets_arima = self._components["components_number_ets"]
            n_arima_comp = self._components["components_number_arima"]

            for s in range(nsim):
                b_slice = random_parameters[s, poly_index + 1 : poly_index + 1 + n_arma]
                polys = adam_polynomialiser(
                    adam_cpp=self._adam_cpp,
                    B=b_slice,
                    ar_orders=ar_orders_padded,
                    i_orders=i_orders_padded,
                    ma_orders=ma_orders_padded,
                    ar_estimate=bool(ar_estimate),
                    ma_estimate=bool(ma_estimate),
                    arma_parameters=arma_params_arr,
                    lags=lags_arima,
                )
                ari_poly = polys["ari_polynomial"]
                ma_poly = polys["ma_polynomial"]
                if non_zero_ari.size:
                    for row in non_zero_ari:
                        arr_f[
                            n_ets_arima + int(row[1]),
                            n_ets_arima : n_ets_arima + n_arima_comp,
                            s,
                        ] = -ari_poly[int(row[0])]
                        mat_g[n_ets_arima + int(row[1]), s] = -ari_poly[int(row[0])]
                if non_zero_ma.size:
                    for row in non_zero_ma:
                        mat_g[n_ets_arima + int(row[1]), s] += ma_poly[int(row[0])]
            k += n_arma

        # 5c. xreg delta fill (R/reapply.R:548-552).
        # Only relevant when ``regressors="adapt"`` — that's the mode
        # that gives each xreg a persistence parameter. For ``"use"``
        # there are no delta names in ``coef_names`` and this is a
        # no-op.
        xreg_model = self._explanatory.get("xreg_model", False)
        delta_names = [nm for nm in coef_names if nm.startswith("delta")]
        if xreg_model and delta_names:
            n_ets_for_xreg = self._components["components_number_ets"]
            n_arima_for_xreg = self._components["components_number_arima"]
            for i, nm in enumerate(delta_names):
                mat_g[n_ets_for_xreg + n_arima_for_xreg + i, :] = random_parameters[
                    :, idx[nm]
                ]
            k += len(delta_names)

        # 6. Fill the profile array (R/reapply.R:637-674).
        profiles_recent_array = _build_profiles_array(arr_vt_seed, L, nsim)
        j = 0
        if ets_model:
            j += 1
            if "level" in idx:
                profiles_recent_array[j - 1, 0, :] = random_parameters[:, idx["level"]]
                k += 1
            if "trend" in idx:
                profiles_recent_array[j, 0, :] = random_parameters[:, idx["trend"]]
                j += 1
                k += 1
            seasonal_names = [nm for nm in coef_names if nm.startswith("seasonal")]
            if seasonal_names:
                # Two layouts mirror R's: ``seasonal_i`` (single seasonality)
                # and ``seasonalX_i`` (multiple). Group by the prefix.
                groups: dict[str, list[str]] = {}
                for nm in seasonal_names:
                    prefix = nm.rsplit("_", 1)[0]
                    groups.setdefault(prefix, []).append(nm)
                lags_seasonal = self._lags_model.get("lags_model_seasonal", [])
                stype = self._model_type.get("season_type", "N")
                for s_idx, (_prefix, members) in enumerate(groups.items()):
                    lag_s = int(lags_seasonal[s_idx])
                    # First ``lag_s - 1`` seasonal slots are free parameters;
                    # the lag-th is closed by sum-to-zero (A) or product-to-1 (M).
                    cols = [random_parameters[:, idx[nm]] for nm in members]
                    arr = np.column_stack(cols)  # (nsim, lag_s - 1)
                    profiles_recent_array[j, : lag_s - 1, :] = arr.T
                    if stype == "A":
                        profiles_recent_array[j, lag_s - 1, :] = -arr.sum(axis=1)
                    elif stype == "M":
                        prod = np.prod(arr, axis=1)
                        # Guard against zero — fall back to 1 (R would emit NaN).
                        profiles_recent_array[j, lag_s - 1, :] = np.where(
                            prod != 0, 1.0 / prod, 1.0
                        )
                    j += 1
                k += sum(len(v) for v in groups.values())

        # 6b. ARIMA profile fill (R/reapply.R:680-729).
        # Optimal / two-stage initials propagate the sampled ARIMAState
        # entries through the AR or MA polynomial onto the per-component
        # rows of the profile cube. Backcasting / complete don't fit
        # ARIMA initials explicitly, so this block is a no-op.
        if arima_model:
            initial_arima_number = self._arima.get("initial_arima_number")
            if initial_arima_number is None:
                initial_arima_number = sum(
                    1 for nm in coef_names if nm.startswith("ARIMAState")
                )
            initial_arima_number = int(initial_arima_number)
            initial_type = self.initial_type
            n_ets_arima = self._components["components_number_ets"]
            n_arima_comp = self._components["components_number_arima"]
            if (
                initial_type in ("optimal", "two-stage")
                and (ar_estimate or ma_estimate)
                and initial_arima_number > 0
            ):
                from smooth.adam_general.core.utils.polynomials import (
                    adam_polynomialiser,
                )

                e_type = self._model_type.get("error_type", "A")
                ari_dominant = non_zero_ari.size > 0 and (
                    not non_zero_ma.size
                    or non_zero_ari.shape[0] >= non_zero_ma.shape[0]
                )
                for s in range(nsim):
                    b_slice = random_parameters[
                        s, poly_index + 1 : poly_index + 1 + n_arma
                    ]
                    polys = adam_polynomialiser(
                        adam_cpp=self._adam_cpp,
                        B=b_slice,
                        ar_orders=ar_orders_padded,
                        i_orders=i_orders_padded,
                        ma_orders=ma_orders_padded,
                        ar_estimate=bool(ar_estimate),
                        ma_estimate=bool(ma_estimate),
                        arma_parameters=arma_params_arr,
                        lags=lags_arima,
                    )
                    ari_poly = polys["ari_polynomial"]
                    ma_poly = polys["ma_polynomial"]
                    sampled = random_parameters[s, k : k + initial_arima_number]
                    if ari_dominant:
                        mother_row = j + n_arima_comp - 1
                        profiles_recent_array[mother_row, :initial_arima_number, s] = (
                            sampled
                        )
                        for row in non_zero_ari:
                            target = j + int(row[1])
                            coeff = ari_poly[int(row[0])]
                            if e_type == "A":
                                profiles_recent_array[
                                    target, :initial_arima_number, s
                                ] = coeff * sampled
                            else:
                                profiles_recent_array[
                                    target, :initial_arima_number, s
                                ] = np.exp(coeff * np.log(np.abs(sampled) + 1e-300))
                    else:
                        mother_row = n_ets_arima + n_arima_comp - 1
                        profiles_recent_array[mother_row, :initial_arima_number, s] = (
                            sampled
                        )
                        for row in non_zero_ma:
                            target = j + int(row[1])
                            coeff = ma_poly[int(row[0])]
                            if e_type == "A":
                                profiles_recent_array[
                                    target, :initial_arima_number, s
                                ] = coeff * sampled
                            else:
                                profiles_recent_array[
                                    target, :initial_arima_number, s
                                ] = np.exp(coeff * np.log(np.abs(sampled) + 1e-300))
            j += initial_arima_number
            k += initial_arima_number

        # 6b. xreg profile fill (R/reapply.R:730-740).
        # Each xreg parameter named ``xreg1, xreg2, …`` carries its
        # initial coefficient on the profile row that follows the ETS +
        # ARIMA blocks. For numeric xreg ``estimated`` is all-ones and
        # ``missing`` is all-zeros; factor levels with one missing
        # category get the negative-sum normalisation per R lines
        # 735-737. ``xreg_parameters_missing`` / ``_included`` are
        # populated by the checker for factor inputs; numeric-only
        # fits leave them ``None`` and we default to "all included".
        if xreg_model:
            estimated_raw = self._explanatory.get("xreg_parameters_estimated")
            if estimated_raw is None:
                xreg_number = int(self._explanatory.get("xreg_number", 0))
                estimated = np.ones(xreg_number, dtype=int)
            else:
                estimated = np.asarray(estimated_raw, dtype=int)
            missing_raw = self._explanatory.get("xreg_parameters_missing")
            if missing_raw is None:
                missing = np.zeros(len(estimated), dtype=int)
            else:
                missing = np.asarray(missing_raw, dtype=int)
            n_to_estimate = int(estimated.sum())
            estimated_idx = np.where(estimated == 1)[0]
            xreg_coef_names = [nm for nm in coef_names if nm.startswith("xreg")]
            for slot, comp_offset in enumerate(estimated_idx):
                if slot >= len(xreg_coef_names):
                    break
                profiles_recent_array[j + int(comp_offset), 0, :] = random_parameters[
                    :, idx[xreg_coef_names[slot]]
                ]
            absent_indices = np.where(missing != 0)[0]
            if absent_indices.size > 0:
                est_sum = profiles_recent_array[
                    [j + int(c) for c in estimated_idx], 0, :
                ].sum(axis=0)
                for absent in absent_indices:
                    profiles_recent_array[j + int(absent), 0, :] = -est_sum
            j += n_to_estimate
            k += n_to_estimate

        # 7. yt and ot (R/reapply.R:745-754)
        y_in_sample = np.asarray(self.actuals, dtype=np.float64).reshape(-1, 1)
        occurrence_dict = getattr(self, "_occurrence", {}) or {}
        occurrence_model = occurrence_dict.get("occurrence_model", False)
        if occurrence_model:
            occ = occurrence_dict.get("occurrence_object")
            if occ is not None:
                ot = np.asarray(occ.actuals, dtype=np.float64).reshape(-1, 1)
                pt = np.asarray(occ.fitted, dtype=np.float64).reshape(-1)
            else:
                ot = np.ones((n, 1), dtype=np.float64)
                pt = np.ones(n, dtype=np.float64)
        else:
            ot = np.ones((n, 1), dtype=np.float64)
            pt = np.ones(n, dtype=np.float64)

        # 8. Build the index lookup table and call C++ (R/reapply.R:239, 757-761)
        from smooth.adam_general.core.creator import adam_profile_creator

        profiles = adam_profile_creator(
            lags_model_all=self._lags_model["lags_model_all"],
            lags_model_max=L,
            obs_all=n,
            lags=self._lags_model.get("lags"),
        )
        index_lookup_table = np.asfortranarray(
            profiles["index_lookup_table"], dtype=np.uint64
        )
        profiles_recent_array_f = np.asfortranarray(
            profiles_recent_array, dtype=np.float64
        )

        initial_type = self.initial_type
        backcast = initial_type in ("backcasting", "complete")

        result = self._adam_cpp.reapply(
            matrixYt=np.asfortranarray(y_in_sample),
            matrixOt=np.asfortranarray(ot),
            arrayVt=np.asfortranarray(arr_vt),
            arrayWt=np.asfortranarray(arr_wt),
            arrayF=np.asfortranarray(arr_f),
            matrixG=np.asfortranarray(mat_g),
            indexLookupTable=index_lookup_table,
            arrayProfilesRecent=profiles_recent_array_f,
            backcast=backcast,
        )

        # 9. Unpack + scale fitted by occurrence probabilities (R/reapply.R:763-770).
        # Take full ``copy()`` ownership of all data leaving the C++ kernel.
        # pybind11/carma returns buffers whose lifetime is tied to the
        # ``result`` struct on the C++ stack; pandas DataFrames built from
        # those views can outlive the buffer and segfault at GC time.
        new_states = np.array(result.states, copy=True, order="C")
        new_fitted = np.array(result.fitted, copy=True, order="C") * pt.reshape(-1, 1)
        new_fitted = np.ascontiguousarray(new_fitted).copy()
        new_profile = np.array(result.profile, copy=True, order="C")

        # 10. Build the result object
        fitted_series = self.fitted
        if not isinstance(fitted_series, pd.Series):
            fitted_series = pd.Series(np.asarray(fitted_series).ravel())
        index = fitted_series.index
        y_series = pd.Series(
            np.asarray(self.actuals).ravel().copy(),
            index=index,
        )

        col_names = [f"nsim{i + 1}" for i in range(nsim)]
        # Pandas 3.x is always copy-on-write — building a DataFrame from a
        # numpy buffer takes a view, not a copy. The view is patched at the
        # first write but never if the DataFrame is read-only. Stashing the
        # data through a per-column ``dict`` forces pandas to own the
        # memory immediately and avoids a shutdown-time dealloc crash when
        # the original numpy buffer's refcount underflows.
        refitted_df = pd.DataFrame(
            {col: new_fitted[:, i].copy() for i, col in enumerate(col_names)},
            index=index,
        )
        random_params_df = pd.DataFrame(
            {nm: random_parameters[:, i].copy() for i, nm in enumerate(coef_names)},
            index=pd.RangeIndex(start=1, stop=nsim + 1, name="nsim"),
        )
        component_names = self._component_names_for_states()
        mat_g_owned = np.ascontiguousarray(mat_g).copy()
        persistence_df = pd.DataFrame(
            {col: mat_g_owned[:, i].copy() for i, col in enumerate(col_names)},
            index=component_names,
        )

        # Convert the cubes to C-order copies before returning — keeps
        # downstream numpy operations predictable and severs ownership ties
        # to F-ordered scratch buffers built inside this method.
        arr_f_out = np.array(arr_f, copy=True, order="C")
        arr_wt_out = np.array(arr_wt, copy=True, order="C")

        return ReapplyResult(
            time_elapsed=_time.time() - t0,
            y=y_series,
            states=new_states,
            refitted=refitted_df,
            fitted=fitted_series,
            model=str(self._prepared.get("model", self.model)),
            transition=arr_f_out,
            measurement=arr_wt_out,
            persistence=persistence_df,
            profile=new_profile,
            random_parameters=random_params_df,
            nsim=nsim,
        )

    def reforecast(
        self,
        h: int = 10,
        X: Optional[NDArray] = None,
        occurrence: Optional[NDArray] = None,
        interval: Literal["prediction", "confidence", "none"] = "prediction",
        level: Union[float, list] = 0.95,
        side: Literal["both", "upper", "lower"] = "both",
        cumulative: bool = False,
        nsim: int = 100,
        bootstrap: bool = False,
        heuristics: Optional[float] = None,
        seed: Optional[int] = None,
        trim: float = 0.01,
        **vcov_kwargs,
    ):
        """Produce ``h``-step-ahead forecasts via Monte-Carlo reforecasting.

        Python port of R's ``reforecast.adam`` (R/reapply.R:941-1402).
        Internally calls :meth:`reapply` to obtain per-draw refitted
        states, samples per-distribution errors and occurrence draws,
        then runs the shared C++ ``adamCore::reforecast`` kernel to
        produce an ``(h, nsim, nsim)`` cube of trajectories. The cube
        is reduced to a point forecast (trimmed mean across all paths)
        and one of two interval flavours:

        * ``interval="prediction"`` — quantile across all
          ``nsim*nsim`` paths per horizon step (parameter + prediction
          uncertainty mixed).
        * ``interval="confidence"`` — for each error sample, average
          across parameter sets first, then quantile across error
          samples (parameter uncertainty only, conditional on a
          marginalised error path).

        Parameters
        ----------
        h : int, default=10
            Forecast horizon. ``h<=0`` returns fitted-period CIs from
            :meth:`reapply`.
        X : array-like, optional
            Future exogenous regressors (xreg). Phase 2 raises
            ``NotImplementedError`` if the fitted model has xreg.
        occurrence : array-like, optional
            Future occurrence probabilities. Phase 2 raises
            ``NotImplementedError`` if the fitted model has an
            occurrence model.
        interval : {"prediction", "confidence", "none"}
            Interval type. ``"none"`` returns only the point forecast.
        level : float or list of float, default=0.95
            Confidence level(s). Values above 1 are interpreted as
            percentages (e.g. ``95`` -> ``0.95``).
        side : {"both", "upper", "lower"}, default="both"
            Which side(s) of the interval to compute.
        cumulative : bool, default=False
            If True, sum trajectories over the horizon and return a
            length-1 point + interval.
        nsim : int, default=100
            Number of parameter draws (the cube is ``(h, nsim, nsim)``).
        bootstrap, heuristics, **vcov_kwargs
            Forwarded to :meth:`reapply` (which forwards them to
            :meth:`vcov`).
        seed : int, optional
            Forwarded to :meth:`reapply` for reproducible draws.
        trim : float, default=0.01
            Trim proportion for the point-forecast mean (R uses 1% by
            default).

        Returns
        -------
        ReforecastResult
            Container with ``mean``, ``lower``, ``upper``, ``level``,
            ``interval``, ``side``, ``cumulative``, ``h``, ``paths`` and
            ``model``.
        """
        from smooth.adam_general.core.creator import adam_profile_creator
        from smooth.adam_general.core.utils.reforecast import (
            ReforecastResult,
            sample_reforecast_errors,
        )

        self._check_is_fitted()

        if interval not in ("prediction", "confidence", "none"):
            interval = "prediction"
        # R accepts numeric arrays mistakenly used with percent units.
        if np.isscalar(level):
            level_iter: list = [float(level)]  # type: ignore[arg-type]
        else:
            level_iter = [float(lvl) for lvl in level]  # type: ignore[union-attr]
        levels: list[float] = [lvl / 100.0 if lvl > 1.0 else lvl for lvl in level_iter]
        n_levels = len(levels)

        # 1. Reapply to get per-draw states / profile / measurement / etc.
        refitted = self.reapply(
            nsim=nsim,
            bootstrap=bootstrap,
            heuristics=heuristics,
            seed=seed,
            **vcov_kwargs,
        )

        n_obs = int(self.nobs)
        n_components = int(self.transition.shape[0])
        L = int(self._lags_model["lags_model_max"])
        e_type = self._model_type.get("error_type", "A")
        model_name = str(refitted.model)

        # 2. h <= 0 — short-circuit on the refitted matrix (R lines 1101-1125).
        if h <= 0:
            h_final = n_obs
            mean_arr = np.asarray(refitted.refitted, dtype=float).mean(axis=1)
            mean_series = pd.Series(mean_arr, index=refitted.refitted.index)
            if interval == "confidence":
                level_low, level_up = _level_bounds(levels, side, h_final)
                lower_cols, upper_cols = _column_names_for_levels(levels, side)
                lower_df = pd.DataFrame(
                    np.zeros((h_final, n_levels)),
                    index=refitted.refitted.index,
                    columns=lower_cols,
                )
                upper_df = pd.DataFrame(
                    np.zeros((h_final, n_levels)),
                    index=refitted.refitted.index,
                    columns=upper_cols,
                )
                paths_in = np.asarray(refitted.refitted, dtype=float)
                for i in range(h_final):
                    lower_df.iloc[i, :] = np.quantile(paths_in[i, :], level_low[i, :])
                    upper_df.iloc[i, :] = np.quantile(paths_in[i, :], level_up[i, :])
                return ReforecastResult(
                    mean=mean_series,
                    lower=lower_df,
                    upper=upper_df,
                    level=levels,
                    interval=interval,
                    side=side,
                    cumulative=cumulative,
                    h=h,
                    paths=None,
                    model=model_name,
                )
            return ReforecastResult(
                mean=mean_series,
                lower=None,
                upper=None,
                level=levels,
                interval="none",
                side=side,
                cumulative=cumulative,
                h=h,
                paths=None,
                model=model_name,
            )

        # 3. Build the horizon arrays (R lines 1127-1132 + 1287-1291).
        meas_cube = np.asarray(refitted.measurement, dtype=np.float64)
        arr_wt = np.zeros((h, n_components, nsim), order="F")
        if meas_cube.shape[0] >= h:
            arr_wt[:, :, :] = meas_cube[-h:, :, :]
        else:
            # Pad by repeating the last in-sample measurement.
            for hi in range(h):
                src = meas_cube[min(hi, meas_cube.shape[0] - 1), :, :]
                arr_wt[hi, :, :] = src

        # 3b. xreg newdata expansion (R/reapply.R:1135-1226).
        # The in-sample ``meas_cube`` carries the historic xreg columns;
        # for ``h``-step forecasting they must be overwritten with future
        # values from ``X`` (or the last in-sample slice as the R-style
        # fallback). Only numeric xreg is wired up here — formula-based
        # factor expansion is a follow-up.
        if self._explanatory.get("xreg_model", False):
            xreg_number = int(self._explanatory.get("xreg_number", 0))
            n_ets_fc = self._components["components_number_ets"]
            n_arima_fc = self._components["components_number_arima"]
            xreg_col_lo = n_ets_fc + n_arima_fc
            xreg_col_hi = xreg_col_lo + xreg_number
            if X is None:
                xreg_in = np.asarray(self._explanatory["xreg_data"], dtype=float)
                if xreg_in.shape[0] >= h:
                    new_xreg = xreg_in[-h:, :]
                else:
                    new_xreg = np.tile(xreg_in[-1:], (h, 1))
                warnings.warn(
                    "newdata (X) not provided to reforecast for an xreg "
                    "model; using the last h in-sample xreg rows as a "
                    "fallback (matches R's behaviour).",
                    stacklevel=2,
                )
            else:
                new_xreg = np.asarray(X, dtype=float)
                if new_xreg.dtype.kind not in ("f", "i", "u"):
                    raise NotImplementedError(
                        "ADAM.reforecast for xreg models with non-numeric "
                        "newdata (factor expansion) is not yet implemented."
                    )
                if new_xreg.ndim == 1:
                    new_xreg = new_xreg.reshape(-1, 1)
                if new_xreg.shape[0] < h:
                    pad = np.tile(new_xreg[-1:], (h - new_xreg.shape[0], 1))
                    new_xreg = np.vstack([new_xreg, pad])
                elif new_xreg.shape[0] > h:
                    new_xreg = new_xreg[:h, :]
            arr_wt[:, xreg_col_lo:xreg_col_hi, :] = new_xreg[:, :, np.newaxis]

        # Forecast-step index lookup (R: ``adamProfileCreator(..., obsInSample+h)``)
        profiles = adam_profile_creator(
            lags_model_all=self._lags_model["lags_model_all"],
            lags_model_max=L,
            obs_all=n_obs + h,
            lags=self._lags_model.get("lags"),
        )
        ilt_full = profiles["index_lookup_table"]
        index_lookup_table = np.asfortranarray(
            ilt_full[:, n_obs + L :], dtype=np.uint64
        )

        # 4. Error sampling per R's switch (reapply.R:1254-1273)
        distribution = (
            (self._general.get("distribution_new") if self._general else None)
            or (self._general.get("distribution") if self._general else "dnorm")
            or "dnorm"
        )
        other_dict = (self._prepared or {}).get("other") or {}
        rng = np.random.default_rng(seed if seed is None else seed + 1)
        arr_errors = sample_reforecast_errors(
            distribution=distribution,
            h=h,
            nsim=nsim,
            sigma=float(self.sigma),
            n_obs=n_obs,
            n_param=int(self.nparam),
            opt_scale=float(self.scale),
            shape=other_dict.get("shape", getattr(self, "gnorm_shape", None)),
            alpha=other_dict.get("alpha"),
            rng=rng,
        )

        # R normalises the errors when nsim <= 500 to remove MC bias.
        if nsim <= 500:
            if e_type == "A":
                arr_errors -= arr_errors.mean(axis=(1, 2), keepdims=True)
            else:
                shifted = 1.0 + arr_errors
                arr_errors = shifted / shifted.mean(axis=(1, 2), keepdims=True) - 1.0
            arr_errors = np.asfortranarray(arr_errors, dtype=np.float64)

        # 5. Occurrence draws (no occurrence model → all-ones).
        occurrence_dict = getattr(self, "_occurrence", {}) or {}
        occurrence_model = occurrence_dict.get("occurrence_model", False)
        if occurrence is not None:
            p_forecast = np.asarray(occurrence, dtype=float).ravel()
            if p_forecast.size < h:
                p_forecast = np.concatenate(
                    [p_forecast, np.full(h - p_forecast.size, p_forecast[-1])]
                )
            elif p_forecast.size > h:
                p_forecast = p_forecast[:h]
        elif occurrence_model:
            raise NotImplementedError(
                "ADAM.reforecast for occurrence models (mixture demand) "
                "requires forecasting the occurrence probabilities and "
                "is not yet implemented."
            )
        else:
            p_forecast = np.ones(h, dtype=float)

        rng_occ = np.random.default_rng(seed if seed is None else seed + 2)
        # ``(h, nsim, nsim)`` Bernoulli draws with R's per-horizon p_forecast.
        arr_ot = rng_occ.binomial(
            1, p_forecast.reshape(-1, 1, 1), size=(h, nsim, nsim)
        ).astype(np.float64)
        arr_ot = np.asfortranarray(arr_ot)

        # 6. C++ call (R: ``adamCpp$reforecast``).
        # ``e_type_modified`` mirrors R lines 1248-1252: additive errors
        # combined with a positive-support distribution route through the
        # multiplicative branch of the kernel.
        e_type_modified = e_type
        if e_type == "A" and distribution in {
            "dlnorm",
            "dinvgauss",
            "dgamma",
            "dls",
            "dllaplace",
            "dlgnorm",
        }:
            e_type_modified = "M"

        # Make per-call fresh F-ordered copies (mirrors the reapply hardening).
        arr_f_call = np.array(
            refitted.transition, copy=True, order="F", dtype=np.float64
        )
        mat_g_call = np.array(
            refitted.persistence.to_numpy(), copy=True, order="F", dtype=np.float64
        )
        profile_call = np.array(
            refitted.profile, copy=True, order="F", dtype=np.float64
        )

        result = self._adam_cpp.reforecast(
            arrayErrors=arr_errors,
            arrayOt=arr_ot,
            arrayWt=arr_wt,
            arrayF=arr_f_call,
            matrixG=mat_g_call,
            indexLookupTable=index_lookup_table,
            arrayProfileRecent=profile_call,
            E=e_type_modified,
        )
        paths = np.array(result.data, copy=True, order="C")  # (h, nsim, nsim)

        # 7. Reduce to mean + intervals (R lines 1294-1316).
        mean_index = _forecast_index(self.fitted, h)
        if cumulative:
            cumsums = np.nansum(paths, axis=0)  # (nsim, nsim)
            mean_scalar = float(_trim_mean(cumsums.ravel(), trim))
            mean_series = pd.Series([mean_scalar], index=mean_index[:1])
        else:
            # Trimmed mean per horizon step over all (k, j) pairs.
            mean_arr = np.array(
                [_trim_mean(paths[i, :, :].ravel(), trim) for i in range(h)],
                dtype=float,
            )
            mean_series = pd.Series(mean_arr, index=mean_index)

        if interval == "none":
            return ReforecastResult(
                mean=mean_series,
                lower=None,
                upper=None,
                level=levels,
                interval="none",
                side=side,
                cumulative=cumulative,
                h=h,
                paths=paths,
                model=model_name,
            )

        level_low, level_up = _level_bounds(levels, side, 1 if cumulative else h)
        lower_cols, upper_cols = _column_names_for_levels(levels, side)
        idx = mean_index[:1] if cumulative else mean_index
        lower_df = pd.DataFrame(
            np.zeros(level_low.shape), index=idx, columns=lower_cols
        )
        upper_df = pd.DataFrame(np.zeros(level_up.shape), index=idx, columns=upper_cols)

        if cumulative:
            cumsums = np.nansum(paths, axis=0).ravel()
            lower_df.iloc[0, :] = np.nanquantile(cumsums, level_low[0, :])
            upper_df.iloc[0, :] = np.nanquantile(cumsums, level_up[0, :])
        elif interval == "prediction":
            for i in range(h):
                flat = paths[i, :, :].ravel()
                lower_df.iloc[i, :] = np.nanquantile(flat, level_low[i, :])
                upper_df.iloc[i, :] = np.nanquantile(flat, level_up[i, :])
        else:  # confidence
            for i in range(h):
                # Average across parameter sets (axis 0 of the (nsim, nsim)
                # slice) first, then quantile across error samples.
                col_means = np.array(
                    [_trim_mean(paths[i, :, j], trim) for j in range(nsim)],
                    dtype=float,
                )
                lower_df.iloc[i, :] = np.nanquantile(col_means, level_low[i, :])
                upper_df.iloc[i, :] = np.nanquantile(col_means, level_up[i, :])

        # Inf / NaN substitution (R lines 1320-1364).
        if not cumulative:
            inf_low_val = -np.inf if e_type == "A" else 0.0
            lower_df = lower_df.where(level_low != 0.0, inf_low_val)
            upper_df = upper_df.where(level_up != 1.0, np.inf)
        else:
            if e_type == "A" and np.any(level_low == 0.0):
                lower_df[:] = -np.inf
            elif e_type == "M" and np.any(level_low == 0.0):
                lower_df[:] = 0.0
            if np.any(level_up == 1.0):
                upper_df[:] = np.inf

        fill_val = 0.0 if e_type == "A" else 1.0
        lower_df = lower_df.fillna(fill_val)
        upper_df = upper_df.fillna(fill_val)

        return ReforecastResult(
            mean=mean_series,
            lower=lower_df,
            upper=upper_df,
            level=levels,
            interval=interval,
            side=side,
            cumulative=cumulative,
            h=h,
            paths=paths,
            model=model_name,
        )

    def simulate(
        self,
        nsim: int = 1,
        seed: Optional[int] = None,
        obs: Optional[int] = None,
        randomizer: Optional[Any] = None,
        **randomizer_kwargs,
    ):
        """Re-simulate ``obs`` observations from the fitted model.

        Python port of R's ``simulate.adam`` (R/adam.R:7365-7611). Pulls
        the fitted state matrices off ``self`` and feeds them through
        the shared C++ ``adamCore::simulate`` kernel. The default error
        distribution / scale match what the model was fitted under, so
        the simulated series statistically resembles the in-sample fit.

        Parameters
        ----------
        nsim : int, default ``1``
            Number of simulated series.
        seed : int, optional
            Seed for the RNG used by the error sampler. Set this to
            make the result deterministic across runs.
        obs : int, optional
            Number of observations per simulated series. Defaults to
            the in-sample length.
        randomizer : str | callable, optional
            Override the model's fitted distribution. Accepts the same
            R-style names / callables as :func:`sim_es`. When ``None``
            (the default), R's ``simulate.adam`` distribution dispatch
            is used: ``self.distribution`` selects the per-distribution
            sampler from ``sample_reforecast_errors`` (same dispatch as
            ``reforecast``), scaled by ``self.scale``.
        **randomizer_kwargs
            Forwarded to the randomizer when overridden.

        Returns
        -------
        SimulateResult
            Container with ``data``, ``states``, ``residuals``,
            ``persistence``, ``measurement``, ``transition``,
            ``initial``, ``probability``, ``occurrence`` — matches R's
            ``"adam.sim"`` S3 list field-for-field. The ``model``
            string is suffixed with ``" estimated via adam()"`` so
            ``print()`` reproduces R's ``print.adam.sim`` output.
        """
        from smooth.adam_general.core.creator import adam_profile_creator
        from smooth.adam_general.core.simulate.randomizer import resolve_randomizer
        from smooth.adam_general.core.simulate.result import SimulateResult
        from smooth.adam_general.core.utils.distributions import generate_errors

        self._check_is_fitted()

        obs_in_sample = int(self._observations["obs_in_sample"])
        if obs is None:
            obs = obs_in_sample
        obs = int(obs)
        nsim = int(nsim)

        e_type = self._model_type.get("error_type", "A")
        t_type = self._model_type.get("trend_type", "N")
        s_type = self._model_type.get("season_type", "N")
        lags_model_all = list(self._lags_model["lags_model_all"])
        lags_model_max = int(
            self._lags_model.get("lags_model_max", max(lags_model_all))
        )
        n_ets = int(self._components.get("components_number_ets", 0))
        n_ets_seas = int(self._components.get("components_number_ets_seasonal", 0))
        n_ets_nonseas = int(
            self._components.get("components_number_ets_non_seasonal", n_ets)
        )
        n_arima = int(self._components.get("components_number_arima", 0))
        xreg_number = int(self._explanatory.get("xreg_number", 0))
        constant_required = bool(self._constant.get("constant_required", False))

        # Pull fitted matrices: prefer ``_prepared`` (post-fit) over
        # ``_adam_created`` (pre-optimisation defaults).
        transition = np.asarray(
            self._prepared.get("transition", self._adam_created["mat_f"]),
            dtype=np.float64,
        )
        measurement = np.asarray(
            self._prepared.get("measurement", self._adam_created["mat_wt"]),
            dtype=np.float64,
        )
        # ``self._prepared["persistence"]`` is sometimes a flat array
        # and sometimes a structured dict (e.g. OM stores it as
        # ``{"alpha": 0.37}``); fall back to ``vec_g`` from
        # ``_adam_created`` in the dict case since that's always the
        # raw column-vector the C++ kernel wants.
        persistence_raw = self._prepared.get("persistence")
        if persistence_raw is None or isinstance(persistence_raw, dict):
            persistence_raw = self._adam_created["vec_g"]
        persistence = np.asarray(persistence_raw, dtype=np.float64).reshape(-1)
        states_fit = np.asarray(self._prepared["states"], dtype=np.float64)

        # ``states_fit`` has shape ``(n_components, obs_in_sample + lag_max)``.
        # Build the simulation state cube as a per-sim copy of those states,
        # extended (or truncated) to the requested ``obs``.
        n_components_states = states_fit.shape[0]
        n_state_cols = obs + lags_model_max
        arr_vt = np.full(
            (n_components_states, n_state_cols, nsim), np.nan, dtype=np.float64
        )
        cols_to_copy = min(states_fit.shape[1], n_state_cols)
        for s in range(nsim):
            arr_vt[:, :cols_to_copy, s] = states_fit[:, :cols_to_copy]
            if cols_to_copy < n_state_cols:
                # Pad with last in-sample column (matches R's repeated tail).
                arr_vt[:, cols_to_copy:, s] = states_fit[:, -1:].repeat(
                    n_state_cols - cols_to_copy, axis=1
                )

        # Build the per-step measurement matrix (R/adam.R:7515-7519).
        if measurement.shape[0] < obs:
            pad_rows = obs - measurement.shape[0]
            measurement = np.vstack(
                [measurement, np.tile(measurement[-1:], (pad_rows, 1))]
            )
        elif measurement.shape[0] > obs:
            measurement = measurement[:obs, :]

        arr_f = np.repeat(transition[:, :, None], nsim, axis=2)
        mat_g = np.repeat(persistence[:, None], nsim, axis=1)

        # ---------- error sampling ----------------------------------------
        rng = np.random.default_rng(seed)
        n_errors = obs * nsim
        if randomizer is None:
            # R's distribution-aware default. ``self.scale`` is the
            # optimisation-scale parameter that ``generate_errors`` consumes.
            distribution = self._general.get("distribution", "dnorm") or "dnorm"
            n_param = int(self.df_used) if hasattr(self, "df_used") else 0
            df = max(obs_in_sample - n_param, 1)
            scale_val = float(self.scale) * obs_in_sample / df
            shape = (self.other or {}).get("shape") if hasattr(self, "other") else None
            alpha = (self.other or {}).get("alpha") if hasattr(self, "other") else None
            errors_flat = generate_errors(
                distribution=distribution,
                n=n_errors,
                scale=scale_val,
                obs_in_sample=obs_in_sample,
                n_param=n_param,
                shape=shape,
                alpha=alpha,
                random_state=rng,
            )
        else:
            errors_flat = resolve_randomizer(randomizer, rng, **randomizer_kwargs)(
                n_errors
            )
        mat_errors = np.asarray(errors_flat, dtype=np.float64).reshape(
            (obs, nsim), order="F"
        )

        # ---------- occurrence mask --------------------------------------
        occurrence_model = getattr(self, "occurrence", None)
        if occurrence_model in (None, "none", "fixed"):
            pt = np.ones(obs, dtype=np.float64)
            mat_ot = np.ones((obs, nsim), dtype=np.float64)
        else:
            fitted_occ = getattr(self, "occurrence_fitted", None)
            pt = (
                np.asarray(fitted_occ, dtype=np.float64).ravel()
                if fitted_occ is not None
                else np.ones(obs, dtype=np.float64)
            )
            if pt.size < obs:
                pt = np.concatenate(
                    [pt, np.full(obs - pt.size, pt[-1] if pt.size else 1.0)]
                )
            else:
                pt = pt[:obs]
            mat_ot = rng.binomial(1, pt[:, None], size=(obs, nsim)).astype(np.float64)

        # ---------- profile lookup ----------------------------------------
        profiles = adam_profile_creator(
            lags_model_all=lags_model_all,
            lags_model_max=lags_model_max,
            obs_all=obs,
        )
        index_lookup_table = profiles["index_lookup_table"]
        profiles_recent_array = np.ascontiguousarray(
            arr_vt[:, :lags_model_max, :], dtype=np.float64
        )

        # ---------- error-type modifier (R/adam.R:7573-7576) --------------
        e_type_modified = e_type
        if e_type == "A" and (
            self._general.get("distribution")
            in {"dlnorm", "dinvgauss", "dgamma", "dls", "dllaplace"}
        ):
            e_type_modified = "M"

        # ---------- drive the C++ kernel ----------------------------------
        result = adam_simulator(
            matrixErrors=mat_errors,
            matrixOt=mat_ot,
            arrayVt=arr_vt,
            matrixWt=measurement,
            arrayF=arr_f,
            matrixG=mat_g,
            lags=np.asarray(lags_model_all, dtype=np.uint64),
            indexLookupTable=index_lookup_table,
            profilesRecent=profiles_recent_array,
            E=e_type_modified,
            T=t_type,
            S=s_type,
            nNonSeasonal=n_ets_nonseas,
            nSeasonal=n_ets_seas,
            nArima=n_arima,
            nXreg=xreg_number,
            constant=constant_required,
        )
        mat_yt = np.asarray(result["matrixYt"], dtype=np.float64)
        arr_vt_out = np.asarray(result["arrayVt"], dtype=np.float64).reshape(
            arr_vt.shape, order="F"
        )

        # ---------- wrap output ------------------------------------------
        if nsim == 1:
            data_out: Union[pd.Series, pd.DataFrame] = pd.Series(mat_yt[:, 0])
            residuals_out: Union[pd.Series, pd.DataFrame] = pd.Series(mat_errors[:, 0])
        else:
            data_out = pd.DataFrame(mat_yt)
            residuals_out = pd.DataFrame(mat_errors)

        model_label = (
            f"{self._model_type.get('model', '')} estimated via adam()"
        ).strip()

        initial_arr = np.asarray(states_fit[:, :lags_model_max], dtype=np.float64)

        return SimulateResult(
            model=model_label,
            data=data_out,
            states=arr_vt_out,
            residuals=residuals_out,
            persistence=np.asarray(persistence, dtype=np.float64).reshape(-1, 1),
            measurement=np.asarray(measurement, dtype=np.float64),
            transition=np.asarray(transition, dtype=np.float64),
            initial=initial_arr,
            probability=pt,
            occurrence=mat_ot if not np.all(mat_ot == 1.0) else None,
            profile=profiles_recent_array,
            other=dict(randomizer_kwargs),
        )

    def _component_names_for_states(self) -> list:
        """Component-axis labels for state-space matrices.

        Mirrors the column names of R's ``object$states`` / row names of
        ``object$persistence``: ``level``, ``trend``, ``seasonal``/
        ``seasonal1``..., ARIMA states, regressors, ``constant``. Falls
        back to ``c1, c2, …`` if a definitive layout cannot be derived.
        """
        comp = self._components
        n_total = int(self.transition.shape[0])
        names: list[str] = []
        ets = self._model_type.get("ets_model", False)
        if ets:
            ttype = self._model_type.get("trend_type", "N")
            if ttype != "N":
                names.append("level")
                names.append("trend")
            else:
                names.append("level")
            n_seas = int(comp.get("components_number_ets_seasonal", 0))
            for i in range(n_seas):
                names.append(f"seasonal{i + 1}" if n_seas > 1 else "seasonal")
        n_arima = int(comp.get("components_number_arima", 0))
        for i in range(n_arima):
            names.append(f"ARIMAState{i + 1}")
        xreg_names = self._explanatory.get("xreg_names") or []
        for nm in xreg_names:
            names.append(str(nm))
        if (
            self._constant
            and self._constant.get("constant_required", False)
            and len(names) < n_total
        ):
            names.append("constant")
        # Pad with generic component names if anything is left over.
        while len(names) < n_total:
            names.append(f"c{len(names) + 1}")
        return names[:n_total]

    def multicov(
        self,
        type: str = "analytical",
        h: int = 10,
        nsim: int = 1000,
    ) -> "pd.DataFrame":
        """Covariance matrix of multi-step-ahead forecast errors.

        Mirrors R's ``multicov.adam`` (R/adam.R:7051-7236). For an ``h``-step
        horizon, returns a symmetric ``(h, h)`` matrix where the ``(i, j)``
        entry is the covariance between the ``i``-step and ``j``-step
        forecast errors. Useful for cumulative-forecast variance, joint
        prediction-interval construction, and multi-step diagnostics.

        Parameters
        ----------
        type : {"analytical", "empirical", "simulated"}, default="analytical"
            * ``"analytical"`` — closed-form from the state-space matrices
              ``(F, W, g, σ²)``. Uses the existing
              :func:`~smooth.adam_general.core.utils.var_covar.covar_anal`
              for additive errors; falls back to a diagonal built from
              :func:`~smooth.adam_general.core.utils.var_covar.var_anal`
              for multiplicative-error models on log/positive
              distributions (matches the dispatch in
              :mod:`~smooth.adam_general.core.forecaster.intervals`).
            * ``"simulated"`` — averages the empirical covariance across
              ``nsim`` simulator paths. Reuses the existing
              ``predict(interval="simulated", scenarios=True)`` machinery
              so distribution-specific error generation, scale de-biasing,
              and occurrence handling are consistent with the
              prediction-interval path.
            * ``"empirical"`` — rolling-origin cross-product:
              ``(errorsᵀ errors) / (nobs - h)`` where ``errors`` is
              :meth:`rmultistep`'s ``(T-h, h)`` output. Mirrors R's
              ``multicov.adam`` empirical branch (R/adam.R:7090-7092);
              both languages call the same C++ ``adamCore::ferrors``
              backend so the per-cell residuals are bit-equivalent.
        h : int, default=10
            Forecast horizon. The returned matrix is ``(h, h)``.
        nsim : int, default=1000
            Number of simulator paths when ``type="simulated"``. Ignored
            otherwise.

        Returns
        -------
        pandas.DataFrame
            Symmetric ``(h, h)`` covariance, indexed and columned by
            ``["h1", "h2", ..., "hh"]``.

        Notes
        -----
        Standalone :class:`OM` inherits this method and **produces a
        link-scale covariance** with ``type="analytical"``. The OM's
        ``sigma`` is ``sqrt(mean(residuals²))`` (mirroring R's
        ``sigma.om`` in R/om.R), so the returned matrix is the covariance
        of multi-step forecast errors on the link-transformed (logit /
        log-odds) scale, not on the probability axis.
        ``type="simulated"`` is not yet supported
        on OM because the occurrence-aware predict route does not
        populate the scenarios matrix the simulated branch relies on.

        :class:`~smooth.adam_general.core.omg.OMG` overrides this method to
        raise — the joint occurrence model's multi-step distribution does
        not have a closed-form covariance in terms of the per-sub-model
        state-space matrices; call ``model.model_a.multicov()`` and
        ``model.model_b.multicov()`` instead.
        """
        import pandas as pd

        from smooth.adam_general.core.utils.var_covar import covar_anal, var_anal

        self._check_is_fitted()

        if type not in ("analytical", "empirical", "simulated"):
            raise ValueError(
                "type must be one of 'analytical', 'empirical', 'simulated'; "
                f"got {type!r}."
            )
        if int(h) < 1:
            raise ValueError(f"h must be >= 1, got {h!r}.")
        if int(nsim) < 1:
            raise ValueError(f"nsim must be >= 1, got {nsim!r}.")
        h = int(h)
        nsim = int(nsim)

        h_labels = [f"h{i + 1}" for i in range(h)]

        if type == "empirical":
            cov = self._multicov_empirical(h)
        elif type == "analytical":
            cov = self._multicov_analytical(h, covar_anal, var_anal)
        else:
            cov = self._multicov_simulated(h, nsim)

        return pd.DataFrame(cov, index=h_labels, columns=h_labels)

    def _multicov_empirical(self, h: int) -> NDArray:
        """Rolling-origin empirical covariance — mirrors R/adam.R:7090-7092.

        Calls :meth:`rmultistep` to get the ``(T-h, h)`` matrix of
        rolling-origin multi-step forecast errors, then forms
        ``(errorsᵀ errors) / (nobs - h)`` (R's ``multicov.adam`` empirical
        formula). ``rmultistep`` itself routes through the same C++
        ``adamCore::ferrors`` backend R uses, so the per-cell errors are
        bit-equivalent between languages.
        """
        errors = self.rmultistep(h=h).to_numpy()
        n_obs = int(self.nobs)
        # Guard the denominator against pathological tiny samples
        # (matches the spirit of R/methods.R:215 — ``df[df<=0] <-
        # obs[df<=0]``).
        df = max(n_obs - h, 1)
        return (errors.T @ errors) / df

    def _multicov_analytical(self, h: int, covar_anal_fn, var_anal_fn) -> NDArray:
        """Closed-form covariance — mirrors R/adam.R:7087-7088."""
        # Pull the matrices the same way intervals.generate_prediction_interval
        # does (intervals.py:67-103): pad/truncate measurement to h rows,
        # transition + persistence as-is, σ² from the fitted scale.
        mat_f = np.asarray(self._prepared["transition"], dtype=float)
        # OM stores persistence as a dict {param_name: value}; ADAM stores it
        # as an ndarray. Coerce both forms to a flat array.
        persistence_raw = self._prepared["persistence"]
        if isinstance(persistence_raw, dict):
            vec_g = np.asarray(list(persistence_raw.values()), dtype=float).flatten()
        else:
            vec_g = np.asarray(persistence_raw, dtype=float).flatten()
        meas_raw = np.asarray(self._prepared["measurement"], dtype=float)
        if meas_raw.shape[0] < h:
            mat_wt = np.tile(meas_raw[-1], (h, 1))
        else:
            mat_wt = meas_raw[-h:]
        lags_all = np.asarray(self._lags_model["lags_model_all"]).flatten()
        sigma_val = float(self.sigma) if self.sigma is not None else float("nan")
        s2 = sigma_val * sigma_val

        # Dispatch on error_type / distribution to mirror the
        # intervals.py branch (line 81-104). Multiplicative-error models on
        # log/positive distributions don't have a meaningful off-diagonal
        # closed form — return a diagonal of per-horizon variances.
        e_type = self._model_type.get("error_type", "A")
        distribution = self._general.get("distribution", "dnorm")
        log_or_pos = distribution in (
            "dinvgauss",
            "dgamma",
            "dlnorm",
            "dllaplace",
            "dls",
            "dlgnorm",
        )
        if self._model_type.get("ets_model") and e_type == "M" and log_or_pos:
            diag = var_anal_fn(lags_all, h, mat_wt[0], mat_f, vec_g, s2)
            diag = np.asarray(diag, dtype=float).flatten()
            if distribution in ("dlnorm", "dls", "dllaplace", "dlgnorm"):
                diag = np.log(1.0 + diag)
            return np.diag(diag)

        cov = covar_anal_fn(lags_all, h, mat_wt, mat_f, vec_g, s2)
        return np.asarray(cov, dtype=float)

    def _multicov_simulated(self, h: int, nsim: int) -> NDArray:
        """Simulator-based covariance — mirrors R/adam.R:7203-7231.

        Drives the existing ``predict(interval='simulated', scenarios=True)``
        path so distribution sampling, scale de-biasing, and occurrence
        handling all match the prediction-interval code. After the
        simulator returns the ``(h, nsim)`` matrix of path values, we
        subtract the per-horizon mean (and divide by the mean for
        multiplicative errors) and form the empirical covariance.
        """
        prev_h = self._general.get("h")
        prev_interval = self._general.get("interval")
        prev_nsim = self._general.get("nsim")
        prev_scenarios = self._general.get("scenarios")
        prev_level = self._general.get("level")
        try:
            self.predict(
                h=h,
                interval="simulated",
                nsim=nsim,
                scenarios=True,
                level=0.5,
            )
        finally:
            self._general["h"] = prev_h
            self._general["interval"] = prev_interval
            self._general["nsim"] = prev_nsim
            self._general["scenarios"] = prev_scenarios
            if prev_level is not None:
                self._general["level"] = prev_level

        sim = self._general.get("_scenarios_matrix")
        if sim is None:
            raise RuntimeError(
                "Simulated path did not return scenarios matrix — internal "
                "forecaster did not populate `_scenarios_matrix`."
            )
        y_sim = np.asarray(sim, dtype=float)  # (h, nsim)

        y_forecast = y_sim.mean(axis=1, keepdims=True)
        y_centred = y_sim - y_forecast
        if self._model_type.get("error_type", "A") == "M":
            # Convert level-residuals to relative residuals: ε_t = (y - ŷ)/ŷ
            # (R/adam.R:7227). Guard against zero forecast values.
            safe = np.where(np.abs(y_forecast) < 1e-15, np.nan, y_forecast)
            y_centred = y_centred / safe
        return (y_centred @ y_centred.T) / nsim

    def plot(  # noqa: B006
        self,
        which=[1, 2, 4, 6],
        level: float = 0.95,
        legend: bool = False,
        lowess: bool = True,
        **kwargs,
    ):
        """
        Diagnostic plots for the fitted ADAM model (R: ``plot.adam``).

        Parameters
        ----------
        which : int or list of int, optional
            Plot type(s) to produce. Default ``[1, 2, 4, 6]``.

            1  — Actuals vs Fitted

            2  — Standardised Residuals vs Fitted

            3  — Studentised Residuals vs Fitted

            4  — |Residuals| vs Fitted

            5  — Residuals² vs Fitted

            6  — Q-Q plot (distribution-specific)

            7  — Actuals and Fitted over time

            8  — Standardised Residuals vs Time

            9  — Studentised Residuals vs Time

            10 — ACF of Residuals

            11 — PACF of Residuals

            12 — Model states over time

            13 — |Standardised Residuals| vs Fitted

            14 — Standardised Residuals² vs Fitted

            15 — ACF of Squared Residuals

            16 — PACF of Squared Residuals

        level : float, optional
            Confidence level for bounds and bands. Default 0.95.
        legend : bool, optional
            Show legend on applicable plots (2, 3, 7, 8, 9). Default False.
        lowess : bool, optional
            Add LOWESS smoothing line to scatter plots. Default True.
        **kwargs
            Passed to matplotlib (e.g. ``figsize``).

        Returns
        -------
        matplotlib.figure.Figure or list[matplotlib.figure.Figure]
            Single figure when ``which`` has one element, list otherwise.

        Examples
        --------
        >>> model = ADAM(model="AAA", lags=[1, 12])
        >>> model.fit(y)
        >>> figs = model.plot()                   # default: which=[1,2,4,6]
        >>> fig  = model.plot(which=7)            # single time-series plot
        >>> figs = model.plot(which=[10, 11])     # ACF and PACF
        """
        from smooth.adam_general.core.plotting import plot_adam

        return plot_adam(
            self, which=which, level=level, legend=legend, lowess=lowess, **kwargs
        )
