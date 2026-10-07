import math
import re
from typing import Literal

import greybox as gb
import numpy as np
import pandas as pd
from greybox import lowess as _greybox_lowess
from scipy.special import beta, digamma, gamma

from smooth.adam_general import _ols  # type: ignore[attr-defined]


def _fsum_mean(x):
    # Shewchuk exact summation, matches R's LDOUBLE mean() to ULP.
    n = len(x)
    return math.fsum(x) / n if n else float("nan")


def _mean_r(x):
    """``mean()`` with R's accumulator.

    R's ``mean.default`` drops the NAs, sums into a long double register,
    divides by n, then makes a second long double pass over ``x[i] - s`` to
    correct the rounding, and only rounds to double at the very end.
    ``math.fsum(...) / n`` rounds twice -- once for the sum, once for the
    divide -- and lands a ULP away wherever the correction pass bites. That
    ULP reaches the ADAM initials: msdecompose's trend seed is the mean of the
    smoothed first differences, so it moves the whole backcast.

    ``np.longdouble`` is 80-bit on x86-64 Linux; where the platform aliases it
    to double this degrades to the plain two-pass mean, which is exactly what
    R does when built without long double support.
    """
    arr = np.asarray(x, dtype=np.float64).ravel()
    arr = arr[~np.isnan(arr)]
    n = arr.size
    if not n:
        return float("nan")

    wide = arr.astype(np.longdouble)
    s = wide.sum() / n
    if not np.isfinite(np.float64(s)):
        return float(s)
    return float(s + (wide - s).sum() / n)


def _r_filter_mean(x):
    # Mirror R's stats::filter(weights=1/N) summation order byte-for-byte:
    # walk the array from the last element down to the first, accumulating
    # `value * (1/N)` in IEEE-double. Without this exact order the seasonal
    # init seeds drift by ≤1 ULP, which the undamped multiplicative ETS(M,M,M)
    # recursion amplifies into a different NLopt basin on chaotic configs
    # like taylor at lag=48.
    n = len(x)
    if not n:
        return float("nan")
    inv_n = 1.0 / n
    arr = np.asarray(x, dtype=np.float64).ravel()
    s = 0.0
    for i in range(n - 1, -1, -1):
        s += float(arr[i]) * inv_n
    return s


# Smoother choices for ADAM/ES/OM/OMG initialisation. "default" resolves to "ma"
# for initial="optimal" and to "global" for every other initialisation method.
# (msdecompose itself keeps its own "lowess" default.)
SmootherType = Literal["default", "ma", "lowess", "supsmu", "global"]
SMOOTHER_DEFAULT: SmootherType = "default"


def resolve_smoother(smoother: str, initial_type: str) -> str:
    """Resolve smoother="default" to the initialisation-specific smoother.

    Mirrors R's adam_checkOptimizer(): "default" becomes "ma" (centred moving
    average) for the optimal initialisation and "global" (a global model fitted
    to the data) for every other initialisation method. Any explicit smoother is
    returned unchanged.
    """
    if smoother != "default":
        return smoother
    return "ma" if initial_type == "optimal" else "global"


def msdecompose(y, lags=[12], type="additive", smoother="lowess"):
    """
    Multiple seasonal decomposition of time series with multiple frequencies.

    This function performs **classical seasonal decomposition** for time series with
    multiple
    seasonal patterns (e.g., hourly data with daily and weekly seasonality, or daily
    data
    with weekly and yearly patterns). It extends the standard STL decomposition to
    handle
    multiple seasonal periods simultaneously.

    The decomposition separates the time series into:

    - **Trend**: Long-term movement (captured via smoothing)
    - **Seasonal components**: One for each seasonal period in `lags`
    - **Remainder** (not explicitly returned but implied): y - trend - seasonals

    **Decomposition Method**:

    For **additive** decomposition:

    .. math::

        y_t = \\text{Trend}_t + \\sum_i \\text{Seasonal}_i(t) + \\epsilon_t

    For **multiplicative** decomposition:

    .. math::

        y_t = \\text{Trend}_t \\times \\prod_i \\text{Seasonal}_i(t) \\times \\epsilon_t

    **Algorithm Steps**:

    1. **Log Transform** (if multiplicative): Apply log to convert to additive form.
    2. **Missing Value Imputation**: Fill NaN using polynomial + Fourier regression.
    3. **Iterative Smoothing**: For each lag period (sorted ascending), apply smoother
       with window = lag period, extract seasonal pattern, remove seasonal mean.
    4. **Trend Extraction**: Final smoothed series is the trend.
    5. **Initial States**: Compute level and slope from trend for model initialization.

    **Smoother Types**:

    - **"ma"**: Moving average with window = lag period. Fast but less flexible.
    - **"lowess"** (default): LOWESS smoothing. Robust to outliers.
    - **"supsmu"**: Friedman's super smoother (uses LOWESS in Python).
    - **"global"**: Global linear regression with intercept and deterministic trend.

    Parameters
    ----------
    y : array-like
        Time series data to decompose. Can contain NaN values (will be imputed).
        Shape: (T,) where T is the number of observations.

    lags : list or array, default=[12]
        Seasonal periods to extract. Examples:

        - [12]: Monthly data with yearly seasonality
        - [24]: Hourly data with daily seasonality
        - [7, 365.25]: Daily data with weekly and yearly seasonality
        - [24, 168]: Hourly data with daily (24h) and weekly (7×24=168h) patterns

        Must contain positive integers. Lags are sorted automatically.

    type : str, default="additive"
        Decomposition type:

        - **"additive"**: Components are summed (for stable seasonality)
        - **"multiplicative"**: Components are multiplied (for proportional seasonality,
          requires y > 0)

    smoother : str, default="lowess"
        Smoothing method for trend and seasonal extraction:

        - **"lowess"**: LOWESS with adaptive span (recommended, **default**)
        - **"supsmu"**: Super smoother (uses LOWESS in Python)
        - **"ma"**: Simple moving average (faster but less robust)
        - **"global"**: Global linear regression (straight line fit)

    Returns
    -------
    dict
        Dictionary containing decomposition results with keys: ``'states'``
        (ndarray of shape (T, n_states) with level, trend, seasonals),
        ``'initial'`` (dict with 'nonseasonal' and 'seasonal' initial values),
        ``'trend'`` (ndarray of shape (T,) with trend component),
        ``'seasonal'`` (list of ndarrays, one per lag, each centered at 0),
        ``'component'`` (list of component descriptions),
        ``'lags'`` (ndarray of sorted unique lag periods),
        ``'type'`` (str, 'additive' or 'multiplicative').

    Raises
    ------
    ValueError
        If type not in ['additive', 'multiplicative']
        If smoother not in ['ma', 'lowess', 'supsmu']
    ImportError
        If smoother='lowess' or 'supsmu' but statsmodels is not installed

    Notes
    -----
    **Missing Values**:

    NaN values are automatically imputed using a regression model:

    .. math::

        \\hat{y}_t = \\sum_{k=0}^d \\beta_k t^k + \\sum_{j=1}^m \\alpha_j \\sin(\\pi t j
        / m)

    where d is polynomial degree (up to 5) and m is the maximum lag.
    This preserves trend and seasonal structure during imputation.

    **Multiplicative Decomposition**:

    Requires strictly positive data. If y ≤ 0, those values are treated as missing.
    Internally works on log(y), then exponentiates results.

    **Smoother Span Selection**:

    For LOWESS, span (bandwidth) is automatically selected based on lag period:

    - For lag = 1: span = 2/3 (R's default)
    - For lag = T: span = 2/3
    - Otherwise: span = 1 / lag
    - Minimum span: 3 / T (ensures smoothness)

    **Seasonal Centering**:

    Each seasonal pattern is centered to have mean zero. This ensures identifiability:
    trend captures the level, seasonals capture deviations.

    **Performance**:

    - Moving average: Very fast (~1ms for T=1000)
    - LOWESS: Moderate (~10-50ms depending on T)
    - Multiple lags: Time scales linearly with number of lags

    **Use in ADAM**:

    The decomposition is used for initial state estimation when initial="backcasting"
    or when the model includes seasonal components. The extracted states provide
    reasonable starting values for the level, trend, and seasonal components.

    **Comparison to STL**:

    Unlike STL (Seasonal-Trend decomposition using Loess), which handles only one
    seasonal period, msdecompose handles **multiple** seasonal periods by iteratively
    removing each seasonal component.

    See Also
    --------
    creator : Uses msdecompose results for initial state estimation
    initialiser : May use decomposition results for parameter initialization

    Examples
    --------
    Decompose monthly data with yearly seasonality::

        >>> y = np.array([112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
        ...               115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140])
        >>> result = msdecompose(y, lags=[12], type='additive', smoother='lowess')
        >>> print(result['trend'])  # Trend component
        >>> print(result['seasonal'][0])  # Yearly seasonal pattern
        >>> print(result['initial']['nonseasonal']['level'])  # Initial level
        >>> print(result['initial']['nonseasonal']['trend'])  # Initial trend
        >>> print(result['initial']['seasonal'][0])  # First 12 seasonal values

    Decompose hourly data with daily and weekly seasonality::

        >>> hourly_data = np.random.randn(24 * 7 * 4)  # 4 weeks of hourly data
        >>> result = msdecompose(hourly_data, lags=[24, 168],  # 24h and 7*24h
        ...                      type='additive', smoother='lowess')
        >>> daily_pattern = result['seasonal'][0]  # 24-hour pattern
        >>> weekly_pattern = result['seasonal'][1]  # Weekly pattern

    Multiplicative decomposition for positive data::

        >>> sales = np.array([100, 120, 150, 140, 130, 160, 200, 210, 180, 140, 110,
        130])
        >>> result = msdecompose(sales, lags=[12], type='multiplicative')
        >>> # Seasonality proportional to level

    Use decomposition for ADAM initialization::

        >>> result = msdecompose(y, lags=[12], type='additive')
        >>> initial_level = result['initial']['nonseasonal']['level']
        >>> initial_trend = result['initial']['nonseasonal']['trend']
        >>> initial_seasonal = result['initial']['seasonal'][0]  # First 12 values
        >>> # Pass to ADAM's initials parameter
    """
    # Argument validation
    if type not in ["additive", "multiplicative"]:
        raise ValueError("type must be 'additive' or 'multiplicative'")
    if smoother not in ["ma", "lowess", "supsmu", "global"]:
        raise ValueError("smoother must be 'ma', 'lowess', 'supsmu', or 'global'")

    # Note: lowess/supsmu use greybox's lowess implementation.

    # Variable name handling
    y_name = "y"

    # Data preparation
    y = np.asarray(y)
    obs_in_sample = len(y)

    # Handle empty lags case — treat as lags=[1]. The decomposition entry
    # point filters out lag=1 before reaching this branch, which can leave
    # an empty list when the only requested lag was 1. Falling back to
    # lags=[1] keeps the smoothing path consistent.
    if len(lags) == 0:
        lags = [1]

    seasonal_lags = any(lag > 1 for lag in lags)

    # Smoothing function definition
    def smoothing_function_ma(y, order):
        """Moving average smoother"""
        # Convert y to float to avoid integer overflow
        y = y.astype(float)
        if order == np.sum(~np.isnan(y)) or order % 2 != 0:
            # Odd order or order equals non-NA count: simple moving average
            k = order
            weights = np.ones(k) / order
        else:
            # Even order: use filter of length order + 1
            k = order + 1
            weights = np.array([0.5] + [1] * (order - 1) + [0.5]) / order
        half_k = (k - 1) // 2  # e.g., for k=13, half_k=6
        trend = np.full_like(y, np.nan)
        n = len(y)
        if n < k:
            return trend

        # R's stats::filter accumulates `z += weights[j] * y[i + nshift - j]`
        # with j ascending -- sequentially, walking *backwards* across the
        # window. np.sum is pairwise and lands on a different double, and the
        # trend feeds the seasonal indices that seed the ARIMA states, where a
        # few ulps are enough to send NLopt down a different path. cumsum is
        # the one NumPy reduction that keeps R's order, and it stays
        # vectorised. (`_r_filter_mean` already does this for the seasonal
        # indices; k is always odd here, so nshift == half_k.)
        idx = np.arange(half_k, n - half_k)[:, None] + half_k - np.arange(k)[None, :]
        trend[half_k : n - half_k] = np.cumsum(y[idx] * weights[None, :], axis=1)[:, -1]
        return trend

    def smoothing_function_lowess(y, order):
        """LOWESS smoother matching R's stats::lowess exactly."""
        y = y.astype(float)
        n = len(y)
        x = np.arange(1, n + 1, dtype=float)

        # Match R's span calculation
        if order is None or order == 1 or order == lags[-1] or order == obs_in_sample:
            span = 2 / 3
        else:
            span = 1 / order

        # Handle missing values
        valid_mask = ~np.isnan(y)
        if not np.any(valid_mask):
            return np.full_like(y, np.nan)

        x_valid = x[valid_mask]
        y_valid = y[valid_mask]

        # R's delta default: 0.01 * diff(range(x))
        x_range = x_valid.max() - x_valid.min()
        delta = 0.01 * x_range if x_range > 0 else 0.0

        # greybox's lowess (x_valid is ascending, so its internal sort is a no-op)
        smoothed_y = np.asarray(
            _greybox_lowess(x_valid, y_valid, f=span, iter=3, delta=delta)["y"]
        )

        # Map back to original indices
        result = np.full_like(y, np.nan)
        result[valid_mask] = smoothed_y

        return result

    def smoothing_function_global(y, order=None):
        """Global linear regression smoother with block dummies"""
        y = y.astype(float)
        n = len(y)
        if order is None or order <= 1:
            X = np.column_stack([np.ones(n), np.arange(1, n + 1)])
        else:
            n_groups = int(np.ceil(int(lags[-1]) / order))
            if n_groups <= 1:
                X = np.column_stack([np.ones(n), np.arange(1, n + 1)])
            else:
                block_idx = np.resize(np.repeat(np.arange(n_groups), order), n)
                dummies = (
                    block_idx[:, None] == np.arange(n_groups - 1)[None, :]
                ).astype(float)
                X = np.column_stack([np.ones(n), dummies, np.arange(1, n + 1)])
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        coef = _ols.ols(X, y)
        return X @ coef

    # Initial data processing
    # obs_in_sample is already defined above

    # Select smoothing function based on smoother type
    if smoother == "ma":
        smoothing_function = smoothing_function_ma
    elif smoother == "global":
        smoothing_function = smoothing_function_global
    else:  # lowess or supsmu
        smoothing_function = smoothing_function_lowess

    # Check if MA smoother works with the given sample size
    if smoother == "ma" and obs_in_sample <= min(lags):
        import warnings

        warnings.warn(
            "The minimum lag is larger than the sample size. "
            "Moving average does not work in this case. "
            "Switching smoother to LOWESS.",
            stacklevel=2,
        )
        smoother = "lowess"
        smoothing_function = smoothing_function_lowess

    y_na_values = np.isnan(y)
    if type == "multiplicative":
        if np.any(y[~y_na_values] <= 0):
            y_na_values = y_na_values | (y <= 0)
        # The non-positive values are imputed below, as the missing ones
        with np.errstate(divide="ignore", invalid="ignore"):
            y_insample = np.log(y)
    else:
        y_insample = y.copy()

    # Missing value imputation
    if np.any(y_na_values):
        degree = min(max(int(np.floor(obs_in_sample / 10)), 1), 5)
        t = np.arange(1, obs_in_sample + 1)
        X_poly = np.vander(t, degree + 1, increasing=True)
        max_lag = np.max(lags)
        X_sin = np.column_stack(
            [np.sin(np.pi * t * k / max_lag) for k in range(1, max_lag + 1)]
        )
        X = np.ascontiguousarray(np.column_stack((X_poly, X_sin)), dtype=np.float64)
        y_fit = np.ascontiguousarray(y_insample[~y_na_values], dtype=np.float64)
        coef = _ols.ols(X[~y_na_values], y_fit)
        y_insample[y_na_values] = X[y_na_values] @ coef

    # Smoothing and trend extraction
    lags = np.sort(np.unique(lags))

    lags_length = len(lags)
    y_smooth = [None] * (lags_length + 1)
    y_smooth[0] = y_insample
    for i in range(lags_length):
        y_smooth[i + 1] = smoothing_function(y_insample, order=lags[i])
    trend = y_smooth[lags_length]

    # Cleared series
    if seasonal_lags:
        y_clear = [None] * lags_length
        for i in range(lags_length):
            y_clear[i] = y_smooth[i] - y_smooth[i + 1]

    # Seasonal patterns
    # Use "ma" smoother for seasonality when original smoother is "global"
    smoother_second = "ma" if smoother == "global" else smoother

    if seasonal_lags:
        patterns = []
        for i in range(lags_length):
            pattern_i = np.zeros(obs_in_sample)
            for j in range(lags[i]):
                indices = np.arange(j, obs_in_sample, lags[i])
                y_seasonal = y_clear[i][indices]
                y_seasonal_non_na = y_seasonal[~np.isnan(y_seasonal)]

                if len(y_seasonal_non_na) > 0:
                    if smoother_second == "ma":
                        y_seasonal_smooth = _r_filter_mean(y_seasonal_non_na)
                        pattern_i[indices] = y_seasonal_smooth
                    else:
                        y_seasonal_smooth = smoothing_function(
                            y_seasonal_non_na, order=obs_in_sample
                        )
                        new_indices = np.arange(len(y_seasonal_smooth)) * lags[i] + j
                        pattern_i[new_indices] = y_seasonal_smooth

            # Truncate to obs_in_sample and normalize (matching R lines 186-189)
            pattern_i = pattern_i[:obs_in_sample]
            # Use only complete seasonal cycles for mean calculation
            obs_in_sample_lags = int(np.floor(obs_in_sample / lags[i]) * lags[i])
            if obs_in_sample_lags > 0:
                pattern_i -= _mean_r(pattern_i[:obs_in_sample_lags])
            patterns.append(pattern_i)
    else:
        patterns = None

    # Initial level and trend
    # Create initial as a dict with nonseasonal and seasonal components
    initial = {"nonseasonal": {}, "seasonal": []}

    # Calculate nonseasonal initial values (level and trend) from the
    # smoothed series at the largest seasonal lag.
    data_for_initial = y_smooth[lags_length]
    valid_data_for_initial = data_for_initial[~np.isnan(data_for_initial)]
    if len(valid_data_for_initial) == 0:
        init_level = 0.0
        init_trend = 0.0
    else:
        # Level: first non-NA value
        init_level = valid_data_for_initial[0]
        # Trend: NaN-skipping mean of first differences of the full series.
        diffs = np.diff(data_for_initial)
        init_trend = _mean_r(diffs) if len(diffs) > 0 else 0.0

    lags_max = max(lags)

    # Centre-correct the initial level when using the moving-average smoother
    if smoother == "ma":
        init_level -= init_trend * np.floor(lags_max / 2)

    # Lag things back to get values useful for ADAM
    init_level -= init_trend * lags_max

    # Store in nonseasonal dict
    initial["nonseasonal"] = {"level": init_level, "trend": init_trend}

    # Return to the original scale
    if type == "multiplicative":
        # Transform nonseasonal initial values back to exponential scale
        initial["nonseasonal"]["level"] = np.exp(initial["nonseasonal"]["level"])
        initial["nonseasonal"]["trend"] = np.exp(initial["nonseasonal"]["trend"])
        trend = np.exp(trend)
        if seasonal_lags:
            patterns = [np.exp(pattern) for pattern in patterns]

    # Extract seasonal initial values (first lags[i] values from each pattern)
    # Lines 256-258 in R
    if seasonal_lags:
        for i in range(lags_length):
            initial["seasonal"].append(patterns[i][: lags[i]])

    # Fitted values and states
    y_fitted = trend.copy()
    if seasonal_lags:
        states = np.column_stack(
            (
                trend,
                np.concatenate(([np.nan], np.diff(trend))),
                np.column_stack(patterns),
            )
        )
        if type == "additive":
            for i in range(lags_length):
                pattern_rep = np.tile(
                    patterns[i], int(np.ceil(obs_in_sample / lags[i]))
                )[:obs_in_sample]
                y_fitted += pattern_rep
        else:
            for i in range(lags_length):
                pattern_rep = np.tile(
                    patterns[i], int(np.ceil(obs_in_sample / lags[i]))
                )[:obs_in_sample]
                y_fitted *= pattern_rep
    else:
        states = np.column_stack((trend, np.concatenate(([np.nan], np.diff(trend)))))

    # Fix for the "NA" in trend in case of global trend (lines 266-268 in R)
    if smoother == "global":
        states[:, 1] = np.nanmean(states[:, 1])

    # Return structure
    result = {
        "y": y,
        "states": states,
        "initial": initial,
        "seasonal": patterns,
        "fitted": y_fitted,
        "loss": "MSE",
        "lags": lags,
        "type": type,
        "yName": y_name,
        "smoother": smoother,
    }
    return result


def _cumsum_r(first, terms):
    """Sequential double-precision accumulation of ``first + sum(terms)``.

    R's C loops accumulate one term at a time in a plain ``double``. ``np.sum``
    is pairwise and ``_sum_r`` is long-double, so neither reproduces that;
    ``np.cumsum`` is the one NumPy reduction that stays strictly sequential, so
    its last element is the running total R would have computed.
    """
    if terms.size == 0:
        return float(first)
    return float(np.cumsum(np.concatenate(([first], terms)))[-1])


def _acf_r(x, nlags):
    """``stats::acf`` (type="correlation"), bit-for-bit.

    R demeans with ``colMeans`` (a long-double accumulator), forms each
    autocovariance as a sequential sum of lagged products over ``n``, and then
    divides by ``sqrt(c0) * sqrt(c0)`` rather than by ``c0`` -- a different
    rounding that shows up in the last bit. The result is clamped to [-1, 1] as
    R does.

    Bit-exactness matters because these values seed the ARIMA parameters in
    ``initialiser()``: Nelder-Mead ranks its simplex by comparison, so a
    one-ulp difference in the starting point flips a tie and sends the two
    languages to different optima within the same evaluation budget.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    nlags = min(int(nlags), n - 1)
    xo = x - _sum_r(x) / n

    c = np.empty(nlags + 1)
    for lag in range(nlags + 1):
        c[lag] = _cumsum_r(0.0, xo[lag:] * xo[: n - lag]) / n

    se = np.sqrt(c[0])
    return np.clip(c / (se * se), -1.0, 1.0)


def _pacf_r(x, nlags):
    """``stats::pacf``, bit-for-bit: Durbin-Levinson over R's own ACF.

    Returns lags ``1..nlags`` -- R carries no lag-0 entry for a partial
    autocorrelation, unlike ``statsmodels``. Note that R centres ``x`` with
    ``scale()`` and then ``acf()`` centres the result a second time; that
    second, near-zero shift is why ``pacf(x)[1]`` and ``acf(x)[2]`` can differ
    in the last bit, and it has to be reproduced here.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    nlags = min(int(nlags), x.size - 1)
    cor = _acf_r(x - _sum_r(x) / x.size, nlags)[1:]

    p = np.zeros(nlags)
    v = np.zeros(nlags)
    w = np.zeros(nlags)
    w[0] = p[0] = cor[0]
    for ll in range(1, nlags):
        a = _cumsum_r(cor[ll], -w[:ll] * cor[ll - 1 :: -1])
        b = _cumsum_r(1.0, -w[:ll] * cor[:ll])
        p[ll] = c = a / b
        if ll + 1 == nlags:
            break
        w[ll] = c
        v[:ll] = w[ll - 1 :: -1]
        w[:ll] -= c * v[:ll]

    return p


def calculate_acf(data, nlags=40):
    """
    Calculate Autocorrelation Function for numpy array or pandas Series.

    Parameters:
    data (np.array or pd.Series): Input time series data
    nlags (int): Number of lags to calculate ACF for

    Returns:
    np.array: ACF values
    """
    if isinstance(data, pd.Series):
        data = data.values

    return _acf_r(data, nlags)


def calculate_pacf(data, nlags=40):
    """
    Calculate Partial Autocorrelation Function for numpy array or pandas Series.

    Parameters:
    data (np.array or pd.Series): Input time series data
    nlags (int): Number of lags to calculate PACF for

    Returns:
    np.array: PACF values
    """
    if isinstance(data, pd.Series):
        data = data.values

    return _pacf_r(data, nlags)


def _real_log(x):
    """``Re(log(as.complex(x)))`` -- log|x|, finite for a negative fitted value.

    R takes the real part of the complex logarithm so that a negative fitted
    value produced mid-optimisation gives a finite number rather than NaN.
    """
    return np.log(np.abs(np.asarray(x, dtype=np.float64)))


def calculate_likelihood(distribution, Etype, y, y_fitted, scale, other):
    """Log-density of one observation, mirroring R's cost function.

    The densities come from greybox, so both languages evaluate the same code;
    only the parameterisation is spelled out here, following the switch in
    ``R/adam.R:790-862``. A multiplicative error scales the density's scale by
    the fitted value.
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    mult = y_fitted if Etype == "M" else 1.0

    if distribution == "dnorm":
        return gb.dnorm(y, y_fitted, np.sqrt(scale) * mult, log=True)
    if distribution == "dlaplace":
        return gb.dlaplace(y, y_fitted, scale * mult, log=True)
    if distribution == "ds":
        # R scales by sqrt(fitted) here, not by fitted.
        return gb.ds(
            y,
            y_fitted,
            scale * (np.sqrt(np.abs(y_fitted)) if Etype == "M" else 1.0),
            log=True,
        )
    if distribution == "dgnorm":
        beta = other if other is not None else 2.0
        return gb.dgnorm(y, y_fitted, scale * mult, beta, log=True)
    if distribution == "dalaplace":
        return gb.dalaplace(y, y_fitted, scale * mult, other, log=True)
    if distribution == "dlogis":
        return gb.dlogis(y, y_fitted, scale * mult, log=True)
    if distribution == "dt":
        errors = (y - y_fitted) * (y_fitted if Etype == "M" else 1.0)
        return gb.dt(errors, abs(other), log=True)

    # Log-domain: the density on the log scale plus the Jacobian -log(y).
    if distribution == "dlnorm":
        return gb.dlnorm(y, _real_log(y_fitted) - scale / 2, np.sqrt(scale), log=True)
    if distribution == "dllaplace":
        return gb.dlaplace(np.log(y), _real_log(y_fitted), scale, log=True) - np.log(y)
    if distribution == "dls":
        return gb.ds(np.log(y), _real_log(y_fitted), scale, log=True) - np.log(y)
    if distribution == "dlgnorm":
        return gb.dgnorm(
            np.log(y), _real_log(y_fitted), scale, other, log=True
        ) - np.log(y)

    if distribution == "dinvgauss":
        return gb.dinvgauss(y, np.abs(y_fitted), np.abs(scale / y_fitted), log=True)
    if distribution == "dgamma":
        return gb.dgamma(y, shape=1 / scale, scale=scale * np.abs(y_fitted), log=True)

    raise ValueError(f"Unsupported distribution {distribution!r}.")


def calculate_entropy(distribution, scale, other, obsZero, y_fitted):
    if distribution == "dnorm":
        return obsZero * (np.log(np.sqrt(2 * np.pi * scale)) + 0.5)
    elif distribution == "dlnorm":
        return obsZero * (np.log(np.sqrt(2 * np.pi * scale)) + 0.5) - scale / 2
    elif distribution == "dlogis":
        return obsZero * 2
    elif distribution in ["dlaplace", "dllaplace", "dalaplace"]:
        return obsZero * (1 + np.log(2 * scale))
    elif distribution in ["ds", "dls"]:
        return obsZero * (2 + 2 * np.log(2 * scale))
    elif distribution in ["dgnorm", "dlgnorm"]:
        return obsZero * (1 / other - np.log(other / (2 * scale * gamma(1 / other))))
    elif distribution == "dt":
        return obsZero * (
            (scale + 1) / 2 * (digamma((scale + 1) / 2) - digamma(scale / 2))
            + np.log(np.sqrt(scale) * beta(scale / 2, 0.5))
        )
    elif distribution == "dinvgauss":
        return 0.5 * (
            obsZero * (np.log(np.pi / 2) + 1 + np.log(scale)) - np.sum(np.log(y_fitted))
        )
    elif distribution == "dgamma":
        return obsZero * (
            1 / scale + np.log(gamma(1 / scale)) + (1 - 1 / scale) * digamma(1 / scale)
        ) + np.sum(np.log(scale * y_fitted))


def observed_mask(observations_dict):
    """The observed in-sample values: the missing ones (``y_na_values``) are not."""
    n = observations_dict["obs_in_sample"]
    missing = observations_dict.get("y_na_values")
    if missing is None:
        return np.ones(n, dtype=bool)
    return ~np.asarray(missing, dtype=bool)[:n]


def complete_windows(observed, h):
    """The windows of the multistep errors (row i of ferrors has the targets
    i..i+h-1) whose targets are all observed: the losses over the missing values are
    not taken (R's ``adam_completeWindows``)."""
    missing_count = np.concatenate([[0], np.cumsum(~np.asarray(observed, dtype=bool))])
    rows = np.arange(max(len(observed) - h + 1, 0))
    return missing_count[rows + h] - missing_count[rows] == 0


def multistep_log_lik(loss_value, loss, h, observed):
    """The concentrated log-likelihood of a multistep loss over the windows with all
    their targets observed, rescaled to the observed values to be comparable with
    the one-step likelihoods (R's ``adam_multistepLogLik``)."""
    n_windows = int(np.sum(complete_windows(observed, h)))
    log_2pi = math.log(2 * math.pi)
    if loss in ("MSEh", "aMSEh", "TMSE", "aTMSE", "MSCE", "aMSCE"):
        value = -n_windows / 2 * (log_2pi + 1 + float(_log_r(loss_value)))
    elif loss in ("GTMSE", "aGTMSE"):
        value = -n_windows / 2 * (log_2pi + 1 + loss_value)
    elif loss in ("MAEh", "TMAE", "GTMAE", "MACE"):
        value = -n_windows * (math.log(2) + 1 + float(_log_r(loss_value)))
    elif loss in ("HAMh", "THAM", "GTHAM", "CHAM"):
        value = -n_windows * (math.log(4) + 2 + 2 * float(_log_r(loss_value)))
    elif loss in ("GPL", "aGPL"):
        # Divided by h to make it comparable with the univariate ones
        value = -n_windows / 2 * (h * log_2pi + h + loss_value) / h
    else:
        value = loss_value
    return value / n_windows * int(np.sum(observed))


def calculate_multistep_loss(loss, adam_errors, obs_in_sample, h):
    """Multistep loss over the matrix of h-steps-ahead errors.

    Every reduction goes through ``_sum_r`` (``colSums``/``rowSums`` for the
    per-column and per-row ones), because R accumulates all three in a long
    double register and a 1-ulp gap here reorders the Nelder-Mead simplex at a
    near-tie and sends the two optimisers to different optima. The squared
    sums are written as ``sum(x**2)`` rather than ``norm(x)**2``: the latter
    takes a square root and squares it back, rounding twice.
    """
    denom = obs_in_sample - h
    last = adam_errors[:, h - 1]
    if loss == "MSEh":
        return _sum_r(last**2) / denom
    elif loss == "TMSE":
        return _sum_r(_sum_r(adam_errors**2, axis=0) / denom)
    elif loss == "GTMSE":
        return _sum_r(np.log(_sum_r(adam_errors**2, axis=0) / denom))
    elif loss == "MSCE":
        return _sum_r(_sum_r(adam_errors, axis=1) ** 2) / denom
    elif loss == "MAEh":
        return _sum_r(np.abs(last)) / denom
    elif loss == "TMAE":
        return _sum_r(_sum_r(np.abs(adam_errors), axis=0) / denom)
    elif loss == "GTMAE":
        return _sum_r(np.log(_sum_r(np.abs(adam_errors), axis=0) / denom))
    elif loss == "MACE":
        return _sum_r(np.abs(_sum_r(adam_errors, axis=1))) / denom
    elif loss == "HAMh":
        return _sum_r(np.sqrt(np.abs(last))) / denom
    elif loss == "THAM":
        return _sum_r(_sum_r(np.sqrt(np.abs(adam_errors)), axis=0) / denom)
    elif loss == "GTHAM":
        return _sum_r(np.log(_sum_r(np.sqrt(np.abs(adam_errors)), axis=0) / denom))
    elif loss == "CHAM":
        return _sum_r(np.sqrt(np.abs(_sum_r(adam_errors, axis=1)))) / denom
    elif loss == "GPL":
        return np.log(np.linalg.det(adam_errors.T @ adam_errors / denom))
    else:
        return 0


def _libm(fun, np_fun, x):
    # Elementwise through the C library, as R's exp() / log() are. NumPy's SIMD
    # kernels round differently in the last bit (exp on ~5% of inputs, log on
    # ~0.4%), and one ulp in the OMG probability flipped a Nelder-Mead step.
    # The non-finite results keep NumPy's value: math raises where R returns
    # Inf / -Inf / NaN, and those are exact anyway.
    arr = np.asarray(x, dtype=np.float64)
    # R returns these silently; the values flow on unchanged
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        out = np.array(np_fun(arr), dtype=np.float64)
    finite = np.isfinite(out)
    values = arr[finite].tolist()
    out[finite] = np.fromiter(map(fun, values), np.float64, len(values))
    return out


def _exp_r(x):
    """``exp()`` as R computes it, elementwise through libm."""
    return _libm(math.exp, np.exp, x)


def _log_r(x):
    """``log()`` as R computes it, elementwise through libm."""
    return _libm(math.log, np.log, x)


def _pow_r(x, power):
    """``x^power`` as R computes it, elementwise through libm's ``pow`` (NumPy's
    power rounds differently on ~5% of the inputs)."""
    p = float(power)
    return _libm(lambda v: math.pow(v, p), lambda a: np.power(a, p), x)


# The Chebyshev coefficients of R's gammafn() (the first ngam=22 of gamcs) and
# lgammacor() (the first nalgm=5 of algmcs), nmath/gamma.c and lgammacor.c
_GAMCS = (
    +0.8571195590989331421920062399942e-2,
    +0.4415381324841006757191315771652e-2,
    +0.5685043681599363378632664588789e-1,
    -0.4219835396418560501012500186624e-2,
    +0.1326808181212460220584006796352e-2,
    -0.1893024529798880432523947023886e-3,
    +0.3606925327441245256578082217225e-4,
    -0.6056761904460864218485548290365e-5,
    +0.1055829546302283344731823509093e-5,
    -0.1811967365542384048291855891166e-6,
    +0.3117724964715322277790254593169e-7,
    -0.5354219639019687140874081024347e-8,
    +0.9193275519859588946887786825940e-9,
    -0.1577941280288339761767423273953e-9,
    +0.2707980622934954543266540433089e-10,
    -0.4646818653825730144081661058933e-11,
    +0.7973350192007419656460767175359e-12,
    -0.1368078209830916025799499172309e-12,
    +0.2347319486563800657233471771688e-13,
    -0.4027432614949066932766570534699e-14,
    +0.6910051747372100912138336975257e-15,
    -0.1185584500221992907052387126192e-15,
)
_ALGMCS = (
    +0.1666389480451863247205729650822e0,
    -0.1384948176067563840732986059135e-4,
    +0.9810825646924729426157171547487e-8,
    -0.1809129475572494194263306266719e-10,
    +0.6221098041892605227126015543416e-13,
)
# stirlerr() at 0, 0.5, ..., 15
_SFERR_HALVES = (
    0.0,
    0.1534264097200273452913848,
    0.0810614667953272582196702,
    0.0548141210519176538961390,
    0.0413406959554092940938221,
    0.03316287351993628748511048,
    0.02767792568499833914878929,
    0.02374616365629749597132920,
    0.02079067210376509311152277,
    0.01848845053267318523077934,
    0.01664469118982119216319487,
    0.01513497322191737887351255,
    0.01387612882307074799874573,
    0.01281046524292022692424986,
    0.01189670994589177009505572,
    0.01110455975820691732662991,
    0.010411265261972096497478567,
    0.009799416126158803298389475,
    0.009255462182712732917728637,
    0.008768700134139385462952823,
    0.008330563433362871256469318,
    0.007934114564314020547248100,
    0.007573675487951840794972024,
    0.007244554301320383179543912,
    0.006942840107209529865664152,
    0.006665247032707682442354394,
    0.006408994188004207068439631,
    0.006171712263039457647532867,
    0.005951370112758847735624416,
    0.005746216513010115682023589,
    0.005554733551962801371038690,
)
_LN_SQRT_2PI = 0.918938533204672741780329736406


def _chebyshev_r(x, coefficients):
    """R's chebyshev_eval()."""
    twox = x * 2
    b0 = b1 = b2 = 0.0
    for a in reversed(coefficients):
        b2 = b1
        b1 = b0
        b0 = twox * b1 - b2 + a
    return (b0 - b2) * 0.5


def _lgammacor_r(x):
    """R's lgammacor() for 10 <= x < 2^26.5."""
    tmp = 10 / x
    return _chebyshev_r(tmp * tmp * 2 - 1, _ALGMCS) / x


def _stirlerr_r(n):
    """R's stirlerr() for n > 10."""
    if n <= 15.0:
        nn = n + n
        if nn == int(nn):
            return _SFERR_HALVES[int(nn)]
        # lgammafn(n + 1)
        x = n + 1.0
        lgamma = _LN_SQRT_2PI + (x - 0.5) * math.log(x) - x + _lgammacor_r(x)
        return lgamma - (n + 0.5) * math.log(n) + n - _LN_SQRT_2PI
    nn = n * n
    s0, s1, s2, s3, s4 = 1 / 12, 1 / 360, 1 / 1260, 1 / 1680, 1 / 1188
    if n > 500:
        return (s0 - s1 / nn) / n
    if n > 80:
        return (s0 - (s1 - s2 / nn) / nn) / n
    if n > 35:
        return (s0 - (s1 - (s2 - s3 / nn) / nn) / nn) / n
    return (s0 - (s1 - (s2 - (s3 - s4 / nn) / nn) / nn) / nn) / n


def _gamma_r(x):
    """``gamma()`` as R computes it (nmath/gamma.c) for x > 0: math.gamma and
    SciPy's round differently in the last bit on most inputs, which moves the
    likelihood of dgnorm through gamma(1/shape)."""
    if x <= 10:
        n = int(x)
        y = x - n
        n -= 1
        value = _chebyshev_r(y * 2 - 1, _GAMCS) + 0.9375
        if n < 0:
            return value / x
        for i in range(1, n + 1):
            value *= y + i
        return value
    if x > 171.61447887182298:
        return math.inf
    if x <= 50 and x == int(x):
        value = 1.0
        for i in range(2, int(x)):
            value *= i
        return value
    # R tests 2*y == (int)2*y, which casts the 2 only: stirlerr() is always taken
    return math.exp((x - 0.5) * math.log(x) - x + _LN_SQRT_2PI + _stirlerr_r(x))


def _log_density_r(distribution, q, scale, shape=None):
    """The log-densities of greybox's dlaplace(), ds() and dgnorm() at location
    zero as R evaluates them, the log of the density, rather than the analytical
    logs of the Python greybox: the two differ in the last bit, enough to move the
    estimates on flat surfaces."""
    if distribution == "dlaplace":
        density = 1 / (2 * scale) * _exp_r(-np.abs(q) / scale)
    elif distribution == "ds":
        density = 1 / (4 * (scale * scale)) * _exp_r(-np.sqrt(np.abs(q)) / scale)
    else:
        density = _exp_r(-_pow_r(np.abs(q) / scale, shape)) * shape
        density = density / (2 * scale * _gamma_r(1 / shape))
    return _log_r(density)


def _sum_r(values, axis=None):
    """``sum()`` with R's accumulator.

    R accumulates ``sum()`` over doubles in a long double register and rounds
    the result back to double; NumPy's pairwise reduction stays in double and
    loses the last bits. That is enough to turn an exact likelihood tie between
    two parameterisations of the same fit -- ANN and MNN coincide when
    alpha = 0, the multiplicative errors being the additive ones rescaled by a
    constant level -- into a 1-ulp difference, which then flips the selected
    model. Squaring still happens in double, as in R.

    The accumulation is sequential, as R's: ``np.sum`` adds pairwise even in
    long double, which rounds differently often enough to send the optimiser
    elsewhere (one ulp in 154 CES likelihoods on Tourism Q375 moved the fit
    from -802.66 to -799.98). ``np.cumsum`` adds in order.

    ``np.longdouble`` is 80-bit on x86-64 Linux; where the platform makes it an
    alias of double this degrades to the plain sequential sum.

    ``axis`` gives R's ``colSums()`` / ``rowSums()``, which use the same long
    double accumulator and also round back to double.
    """
    values = np.asarray(values, dtype=float)
    if axis is None:
        values = values.ravel()
        axis = 0
    if values.shape[axis] == 0:
        total = np.sum(values, axis=axis)
    else:
        running = np.cumsum(values, axis=axis, dtype=np.longdouble)
        total = np.take(running, -1, axis=axis)
    return float(total) if np.ndim(total) == 0 else np.asarray(total, dtype=float)


# Overflow here is the infeasibility signal, not a defect. During optimisation
# NLopt probes parameter vectors that make the state recursion diverge -- on a
# 336-lag MSARIMA one probe reached |e| ~ 2e165 -- and squaring those errors
# overflows to inf. That inf flows into the likelihood, CF() turns a non-finite
# cost into the 1e300 penalty, and the optimiser steers away, which is exactly
# what should happen. R computes the same thing and reports nothing, because R
# does not warn on double overflow. Suppress the warning, not the value: the
# result is unchanged, so the fit stays bit-comparable with R.
@np.errstate(over="ignore")
def scaler(distribution, Etype, errors, y_fitted, obs_in_sample, other):
    """
    Calculate scale parameter for the provided parameters.

    Parameters:
    - distribution (str): The distribution type
    - Etype (str): Error type ('A' for additive, 'M' for multiplicative)
    - errors (np.array): Array of errors
    - y_fitted (np.array): Array of fitted values
    - obs_in_sample (int): Number of observations in sample
    - other (float): Additional parameter for some distributions

    Returns:
    float: The calculated scale parameter
    """

    # Helper: take ``log`` of a possibly-negative input via complex extension.
    # ``log(as.complex(z))`` for z < 0 yields ``log|z| + iπ``; downstream the
    # modulus ``abs(...)`` is taken so the result is finite and continuous.
    # Mirrors R's ``log(as.complex(...))`` pattern used in ``adam_scaler``.
    def complex_log(x):
        return np.log(np.asarray(x, dtype=np.complex128))

    if distribution == "dnorm":
        # sigma^2 = sum(e^2)/n; the likelihood takes sqrt() of it, which is the
        # same double as sqrt(sum(e^2)/n). Not norm(e)^2/n: that rounds twice and
        # lands on a different double, enough to break an exact likelihood tie
        # between two parameterisations of the same fit -- ANN and MNN coincide
        # when alpha = 0 -- and flip the selected model. Mirrors R's
        # ``adam_scaler`` (R/utils-adam.R).
        return _sum_r(errors**2) / obs_in_sample

    elif distribution == "dlaplace":
        return _sum_r(np.abs(errors)) / obs_in_sample

    elif distribution == "ds":
        return _sum_r(np.sqrt(np.abs(errors))) / (obs_in_sample * 2)

    elif distribution == "dgnorm":
        beta = other if other is not None else 2.0
        return (beta * _sum_r(_pow_r(np.abs(errors), beta)) / obs_in_sample) ** (
            1 / beta
        )

    elif distribution == "dalaplace":
        return _sum_r(errors * (other - (errors <= 0) * 1)) / obs_in_sample

    elif distribution == "dlnorm":
        # Cast 1+errors (or 1+errors/yFitted) to complex so log() of negative
        # arguments stays finite; the outer modulus turns the complex log
        # into a real number. Mirrors R's ``log(as.complex(...))`` pattern.
        if Etype == "A":
            log_term = np.abs(complex_log(1 + errors / y_fitted))
        else:  # "M"
            log_term = np.abs(complex_log(1 + errors))
        temp = 1 - np.sqrt(np.abs(1 - _sum_r(log_term**2) / obs_in_sample))
        return 2 * np.abs(temp)

    elif distribution == "dllaplace":
        if Etype == "A":
            return _sum_r(np.abs(complex_log(1 + errors / y_fitted))) / obs_in_sample
        else:  # "M"
            return _sum_r(np.abs(complex_log(1 + errors))) / obs_in_sample

    elif distribution == "dls":
        if Etype == "A":
            return (
                _sum_r(np.sqrt(np.abs(complex_log(1 + errors / y_fitted))))
                / obs_in_sample
            )
        else:  # "M"
            return _sum_r(np.sqrt(np.abs(complex_log(1 + errors)))) / obs_in_sample

    elif distribution == "dlgnorm":
        if Etype == "A":
            return (
                other
                * _sum_r(np.abs(complex_log(1 + errors / y_fitted)) ** other)
                / obs_in_sample
            ) ** (1 / other)
        else:  # "M"
            return (
                other * _sum_r(np.abs(complex_log(1 + errors)) ** other) / obs_in_sample
            ) ** (1 / other)

    elif distribution == "dinvgauss":
        if Etype == "A":
            return (
                _sum_r((errors / y_fitted) ** 2 / (1 + errors / y_fitted))
                / obs_in_sample
            )
        else:  # "M"
            return _sum_r(errors**2 / (1 + errors)) / obs_in_sample

    elif distribution == "dgamma":
        if Etype == "A":
            return _sum_r((errors / y_fitted) ** 2) / obs_in_sample
        else:  # "M"
            return _sum_r(errors**2) / obs_in_sample

    else:
        raise ValueError(f"Unknown distribution: {distribution}")


# The scale is the parameter of the distribution as written in the ADAM monograph
# (Tables 11.1-11.2): sigma^2 for dnorm, dlnorm, dinvgauss and dgamma, s for the
# others. The variance of the error is proportional to scale^p.
def scale_power(distribution):
    """The power p of the scale in the variance of the error term."""
    if distribution in ("ds", "dls"):
        return 4
    if distribution in (
        "dlaplace",
        "dalaplace",
        "dgnorm",
        "dllaplace",
        "dlgnorm",
        "dlogis",
    ):
        return 2
    return 1


def scale_debias(scale, distribution, obs, df):
    """De-bias the scale in the variance space: the variance times obs / df."""
    return scale * (obs / df) ** (1 / scale_power(distribution))


def scale_variance(scale, distribution, other=None):
    """The variance of the error term implied by the scale."""
    other = other or {}
    if distribution in ("dlaplace", "dllaplace"):
        return 2 * scale**2
    if distribution in ("ds", "dls"):
        return 120 * scale**4
    if distribution in ("dgnorm", "dlgnorm"):
        shape = other["shape"]
        return scale**2 * gamma(3 / shape) / gamma(1 / shape)
    if distribution == "dalaplace":
        alpha = other["alpha"]
        return scale**2 / (alpha**2 * (1 - alpha) ** 2 / (alpha**2 + (1 - alpha) ** 2))
    if distribution == "dlogis":
        return scale**2 * np.pi**2 / 3
    return scale


def xreg_selector(errors, xreg_data, names, ic, df, distribution, other=None):
    """R's ``adam_xreg_selector``: ``stepwise()`` on the errors of the model without
    the regressors, with its degrees of freedom added. The names of the selected
    regressors."""
    import warnings

    data = pd.DataFrame(np.asarray(xreg_data, dtype=float), columns=list(names))
    data.insert(0, "errorsIvan41", np.asarray(errors, dtype=float))
    kwargs = {"shape": other} if distribution in ("dgnorm", "dlgnorm") else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = gb.stepwise(
            data, ic=ic, df=df, distribution=distribution, silent=True, **kwargs
        )
    return list(model._feature_names or [])


def make_names(names: list) -> list:
    """R's ``make.names(unique=TRUE)``: the syntactic, unique names of the
    regressors, as R gives them to the variables of the model."""
    result: list = []
    for name in names:
        name = re.sub(r"[^0-9A-Za-z._]", ".", str(name))
        if not re.match(r"^([A-Za-z]|\.(?![0-9]))", name):
            name = "X" + name
        candidate, k = name, 1
        while candidate in result:
            candidate, k = f"{name}.{k}", k + 1
        result.append(candidate)
    return result
