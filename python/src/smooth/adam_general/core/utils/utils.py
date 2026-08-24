import math
from typing import Literal

import numpy as np
import pandas as pd
from greybox import lowess as _greybox_lowess
from scipy import stats
from scipy.special import beta, digamma, gamma

from smooth.adam_general import _ols  # type: ignore[attr-defined]


def _fsum_mean(x):
    # Shewchuk exact summation, matches R's LDOUBLE mean() to ULP.
    n = len(x)
    return math.fsum(x) / n if n else float("nan")


def _fsum_nanmean(x):
    arr = np.asarray(x, dtype=np.float64).ravel()
    mask = ~np.isnan(arr)
    n = int(mask.sum())
    return math.fsum(arr[mask]) / n if n else float("nan")


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
                pattern_i -= _fsum_nanmean(pattern_i[:obs_in_sample_lags])
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
        init_trend = _fsum_nanmean(diffs) if len(diffs) > 0 else 0.0

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


def calculate_likelihood(distribution, Etype, y, y_fitted, scale, other):
    # Fixes the output dimension
    y = y.reshape(-1, 1)

    if distribution == "dnorm":
        if Etype == "A":
            return stats.norm.logpdf(y, loc=y_fitted, scale=scale)
        else:  # "M"
            return stats.norm.logpdf(y, loc=y_fitted, scale=scale * y_fitted)
    elif distribution == "dlaplace":
        if Etype == "A":
            return stats.laplace.logpdf(y, loc=y_fitted, scale=scale)
        else:  # "M"
            return stats.laplace.logpdf(y, loc=y_fitted, scale=scale * y_fitted)
    elif distribution == "ds":
        # The S distribution (greybox::ds), NOT Student's t: its log-density is
        # -log(4 s^2) - sqrt(|x - mu|) / s. scipy has no S distribution.
        if Etype == "A":
            s = scale
            return -np.log(4 * s**2) - np.sqrt(np.abs(y - y_fitted)) / s
        else:  # "M"
            s = scale * np.sqrt(y_fitted)
            return -np.log(4 * s**2) - np.sqrt(np.abs(y - y_fitted)) / s
    elif distribution == "dgnorm":
        beta = other if other is not None else 2.0
        if Etype == "A":
            return stats.gennorm.logpdf(y, beta, loc=y_fitted, scale=scale)
        else:  # "M"
            return stats.gennorm.logpdf(y, beta, loc=y_fitted, scale=scale * y_fitted)
    elif distribution == "dalaplace":
        # Implement asymmetric Laplace distribution
        pass
    elif distribution == "dlnorm":
        # Use the real part of the complex logarithm so that negative
        # y_fitted values during optimisation produce a finite log instead
        # of NaN (the imaginary part is discarded).
        meanlog = np.real(np.log(y_fitted.astype(complex))) - scale**2 / 2
        return stats.lognorm.logpdf(y, s=scale, scale=np.exp(meanlog))
    elif distribution == "dllaplace":
        return stats.laplace.logpdf(
            np.log(y), loc=np.log(y_fitted), scale=scale
        ) - np.log(y)
    elif distribution == "dls":
        # Log-S: the S log-density on the log scale, minus log(y) for the
        # Jacobian. Mirrors R: ds(log(y), log(fitted), scale) - log(y).
        return (
            -np.log(4 * scale**2)
            - np.sqrt(np.abs(np.log(y) - np.log(y_fitted))) / scale
            - np.log(y)
        )
    elif distribution == "dlgnorm":
        # Implement log-generalized normal distribution
        pass
    elif distribution == "dinvgauss":
        # scipy's invgauss(mu, scale=s) is IG(mean=mu*s, lambda=s); statmod's
        # (mean, dispersion) parameterisation maps to mu = mean*dispersion,
        # s = 1/dispersion. Mirrors R: dinvgauss(y, mean=|fitted|,
        # dispersion=|scale/fitted|).
        dispersion = np.abs(scale / y_fitted)
        return stats.invgauss.logpdf(
            y, mu=np.abs(y_fitted) * dispersion, scale=1 / dispersion
        )
    elif distribution == "dgamma":
        return stats.gamma.logpdf(y, a=1 / scale, scale=scale * np.abs(y_fitted))


def calculate_entropy(distribution, scale, other, obsZero, y_fitted):
    if distribution == "dnorm":
        return obsZero * (np.log(np.sqrt(2 * np.pi) * scale) + 0.5)
    elif distribution == "dlnorm":
        return obsZero * (np.log(np.sqrt(2 * np.pi) * scale) + 0.5) - scale**2 / 2
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


def _sum_r(values, axis=None):
    """``sum()`` with R's accumulator.

    R accumulates ``sum()`` over doubles in a long double register and rounds
    the result back to double; NumPy's pairwise reduction stays in double and
    loses the last bits. That is enough to turn an exact likelihood tie between
    two parameterisations of the same fit -- ANN and MNN coincide when
    alpha = 0, the multiplicative errors being the additive ones rescaled by a
    constant level -- into a 1-ulp difference, which then flips the selected
    model. Squaring still happens in double, as in R.

    ``np.longdouble`` is 80-bit on x86-64 Linux; where the platform makes it an
    alias of double this degrades to the plain pairwise sum.

    ``axis`` gives R's ``colSums()`` / ``rowSums()``, which use the same long
    double accumulator and also round back to double.
    """
    total = np.sum(np.asarray(values, dtype=float), axis=axis, dtype=np.longdouble)
    if axis is None:
        return float(total)
    return np.asarray(total, dtype=float)


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
        # sqrt(sum(e^2)/n), not norm(e)/sqrt(n): the second form rounds twice
        # (once in the irrational sqrt(n), once in the divide) and lands on a
        # different double. That is enough to break an exact likelihood tie
        # between two parameterisations of the same fit -- ANN and MNN coincide
        # when alpha = 0 -- and flip the selected model. Mirrors R's
        # ``adam_scaler`` (R/utils-adam.R).
        return np.sqrt(_sum_r(errors**2) / obs_in_sample)

    elif distribution == "dlaplace":
        return _sum_r(np.abs(errors)) / obs_in_sample

    elif distribution == "ds":
        return _sum_r(np.sqrt(np.abs(errors))) / (obs_in_sample * 2)

    elif distribution == "dgnorm":
        beta = other if other is not None else 2.0
        return (beta * _sum_r(np.abs(errors) ** beta) / obs_in_sample) ** (1 / beta)

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
        return np.sqrt(2 * np.abs(temp))

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
