import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import beta, digamma, gamma
from statsmodels.tsa.stattools import acf, pacf

# Import C++ lowess for performance (with Python fallback)
try:
    from smooth.adam_general import lowess_cpp as _lowess_cpp

    _USE_CPP_LOWESS = True
except ImportError:
    _USE_CPP_LOWESS = False

# Note: Custom lowess_r function is kept as reference/fallback implementation

# Default smoother for ADAM/ES model initialisation (msdecompose keeps "lowess")
SMOOTHER_DEFAULT = "global"


def lowess_r(x, y, f=2 / 3, nsteps=3, delta=None):
    """
    LOWESS smoother that exactly matches R's stats::lowess function.

    This is a pure Python/NumPy implementation of Cleveland's LOWESS algorithm
    as implemented in R's stats package (clowess C function).

    Parameters
    ----------
    x : array-like
        X values (will be converted to float)
    y : array-like
        Y values (will be converted to float)
    f : float
        Smoother span (fraction of points), default 2/3
    nsteps : int
        Number of robustifying iterations, default 3
    delta : float or None
        Distance threshold for interpolation. If None, uses 0.01 * range(x)

    Returns
    -------
    ndarray
        Smoothed y values

    References
    ----------
    Cleveland, W.S. (1979) "Robust Locally Weighted Regression and
    Smoothing Scatterplots". JASA 74(368): 829-836.
    R source: https://github.com/SurajGupta/r-source/blob/master/src/library/stats/src/lowess.c
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)

    if n < 2:
        return y.copy()

    if delta is None:
        delta = 0.01 * (x.max() - x.min())

    # Number of points in local window - at least 2, at most n
    ns = max(2, min(n, int(f * n + 1e-7)))

    # Sort by x if not already sorted
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    ys = np.zeros(n)  # smoothed values
    rw = np.ones(n)  # robustness weights
    res = np.zeros(n)  # residuals

    # Compute range for stability checks
    x_range = x_sorted[n - 1] - x_sorted[0]

    def lowest(xs, nleft, nright, rw_iter):
        """
        Compute weighted local regression at point xs.
        This matches R's lowest() function exactly.
        """
        # Compute bandwidth h
        h = max(xs - x_sorted[nleft], x_sorted[nright] - xs)

        # Thresholds for weight calculation (R's h9 and h1)
        h9 = 0.999 * h
        h1 = 0.001 * h

        # Sum of weights
        a = 0.0
        # Store weights - allocate enough space for potential ties beyond nright
        w = np.zeros(n)

        # Loop through points - continue past nright to pick up ties
        j = nleft
        nrt = nright  # will track rightmost point with non-zero weight

        while j < n:
            # Compute absolute distance
            r = abs(x_sorted[j] - xs)

            # Check if within bandwidth (using h9 threshold)
            if r <= h9:
                if r <= h1:
                    # Very close - weight = 1
                    w[j] = 1.0
                else:
                    # Tricube weight: (1 - (r/h)^3)^3
                    w[j] = (1.0 - (r / h) ** 3) ** 3

                # Apply robustness weight from previous iteration
                w[j] *= rw_iter[j]
                a += w[j]
                nrt = j
            elif x_sorted[j] > xs:
                # Past the point, no more ties possible
                break

            j += 1

        # Check if we have any non-zero weights
        if a <= 0.0:
            return 0.0, False

        # Normalize weights to sum to 1
        for j in range(nleft, nrt + 1):
            w[j] /= a

        # Check if we can fit a line (h > 0)
        if h > 0.0:
            # Compute weighted center of x values
            a = 0.0
            for j in range(nleft, nrt + 1):
                a += w[j] * x_sorted[j]

            # Compute slope if points are spread out enough
            b = xs - a
            c = 0.0
            for j in range(nleft, nrt + 1):
                c += w[j] * (x_sorted[j] - a) ** 2

            # Stability check - only use slope if points are spread out
            # (R checks: sqrt(c) > 0.001 * range)
            if np.sqrt(c) > 0.001 * x_range:
                b /= c
                # Adjust weights for linear fit
                for j in range(nleft, nrt + 1):
                    w[j] *= b * (x_sorted[j] - a) + 1.0

        # Compute fitted value as weighted sum
        ys_out = 0.0
        for j in range(nleft, nrt + 1):
            ys_out += w[j] * y_sorted[j]

        return ys_out, True

    # Main robustness iterations
    iteration = 0
    while iteration <= nsteps:
        nleft = 0
        nright = ns - 1
        last = -1  # index of previous estimated point
        i = 0  # index of current point

        while True:
            # Move window right if it decreases radius
            if nright < n - 1:
                d1 = x_sorted[i] - x_sorted[nleft]
                d2 = x_sorted[nright + 1] - x_sorted[i]

                if d1 > d2:
                    # Radius decreases by moving right
                    nleft += 1
                    nright += 1
                    continue

            # Compute fitted value at x[i]
            ys[i], ok = lowest(x_sorted[i], nleft, nright, rw)

            if not ok:
                # All weights zero - copy over value
                ys[i] = y_sorted[i]

            # Interpolate skipped points
            if last < i - 1:
                denom = x_sorted[i] - x_sorted[last]
                # Should be non-zero by construction
                if denom > 0:
                    for j in range(last + 1, i):
                        alpha = (x_sorted[j] - x_sorted[last]) / denom
                        ys[j] = alpha * ys[i] + (1.0 - alpha) * ys[last]

            # Update last estimated point
            last = i

            # Skip ahead using delta - find next point beyond delta
            cut = x_sorted[last] + delta
            i += 1
            while i < n:
                if x_sorted[i] > cut:
                    break
                # Special case: exact ties get same value
                if x_sorted[i] == x_sorted[last]:
                    ys[i] = ys[last]
                    last = i
                i += 1

            # Adjust i (R's: i = max(last+1, i-1))
            i = max(last + 1, i - 1)

            if last >= n - 1:
                break

        # Compute residuals
        for i in range(n):
            res[i] = y_sorted[i] - ys[i]

        # Overall scale estimate (mean absolute residual)
        sc = np.sum(np.abs(res)) / n

        # Compute robustness weights (except on last iteration)
        if iteration >= nsteps:
            break

        # Compute median absolute deviation (cmad = 6 * median)
        abs_res = np.abs(res)
        m1 = n // 2

        # Partial sort to find median
        abs_res_sorted = np.partition(abs_res, m1)
        if n % 2 == 0:
            m2 = n - m1 - 1
            abs_res_sorted = np.partition(abs_res_sorted, m2)
            cmad = 3.0 * (abs_res_sorted[m1] + abs_res_sorted[m2])
        else:
            cmad = 6.0 * abs_res_sorted[m1]

        # Check if effectively zero (R's threshold: 1e-7 * sc)
        if cmad < 1e-7 * sc:
            break

        # Compute biweight robustness weights
        c9 = 0.999 * cmad
        c1 = 0.001 * cmad
        for i in range(n):
            r = abs(res[i])
            if r <= c1:
                rw[i] = 1.0
            elif r <= c9:
                rw[i] = (1.0 - (r / cmad) ** 2) ** 2
            else:
                rw[i] = 0.0

        iteration += 1

    # Restore original order
    result = np.zeros(n)
    result[order] = ys

    return result


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

    # Note: lowess/supsmu now use custom lowess_r implementation that matches R exactly
    # No external dependencies required

    # Variable name handling
    y_name = "y"

    # Data preparation
    y = np.asarray(y)
    obs_in_sample = len(y)

    # Handle empty lags case - treat as lags=[1] to match R behavior
    # In R, msdecompose is never called with empty lags, but the Python code
    # filters lags to remove lag=1, which can result in empty lags.
    # We treat empty lags as lags=[1] to ensure consistent smoothing behavior.
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
        for i in range(half_k, len(y) - half_k):
            trend[i] = np.sum(y[i - half_k : i + half_k + 1] * weights)
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

        # Use C++ lowess for performance, fall back to Python if unavailable
        if _USE_CPP_LOWESS:
            smoothed_y = _lowess_cpp(x_valid, y_valid, f=span, nsteps=3, delta=delta)
        else:
            smoothed_y = lowess_r(x_valid, y_valid, f=span, nsteps=3, delta=delta)

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
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
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
        X = np.column_stack((X_poly, X_sin))
        coef = np.linalg.lstsq(X[~y_na_values], y_insample[~y_na_values], rcond=None)[0]
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
                        y_seasonal_smooth = np.mean(y_seasonal_non_na)
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
                pattern_i -= np.nanmean(pattern_i[:obs_in_sample_lags])
            patterns.append(pattern_i)
    else:
        patterns = None

    # Initial level and trend
    # Create initial as a dict with nonseasonal and seasonal components
    initial = {"nonseasonal": {}, "seasonal": []}

    # Calculate nonseasonal initial values (level and trend)
    #  R: initial$nonseasonal <-
    # c(ySmooth[[ySmoothLength]][!is.na(ySmooth[[ySmoothLength]])][1],
    #                             mean(diff(ySmooth[[ySmoothLength]]),na.rm=T));
    data_for_initial = y_smooth[lags_length]  # Matches R's ySmooth[[ySmoothLength]]
    valid_data_for_initial = data_for_initial[~np.isnan(data_for_initial)]
    if len(valid_data_for_initial) == 0:
        init_level = 0.0
        init_trend = 0.0
    else:
        # Level: first non-NA value
        init_level = valid_data_for_initial[0]
        # Trend: mean of diffs of full series (with NAs), then remove NAs from mean
        # This matches R's mean(diff(x), na.rm=T) behavior
        diffs = np.diff(data_for_initial)  # Diff full series, NAs propagate
        init_trend = np.nanmean(diffs) if len(diffs) > 0 else 0.0

    lags_max = max(lags)

    # Fix the initial for MA smoother (lines 200-202 in R)
    if smoother == "ma":
        init_level -= init_trend * np.floor(lags_max / 2)

    # Lag things back to get values useful for ADAM (lines 204-206 in R)
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

    return acf(data, nlags=nlags, fft=False)


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

    return pacf(data, nlags=nlags, method="ywmle")


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
        if Etype == "A":
            return stats.t.logpdf(y, df=2, loc=y_fitted, scale=scale)
        else:  # "M"
            return stats.t.logpdf(
                y, df=2, loc=y_fitted, scale=scale * np.sqrt(y_fitted)
            )
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
        # Use complex log to handle negative y_fitted during optimization,
        # mirroring R's Re(log(as.complex(y_fitted))) failsafe.
        meanlog = np.real(np.log(y_fitted.astype(complex))) - scale**2 / 2
        return stats.lognorm.logpdf(y, s=scale, scale=np.exp(meanlog))
    elif distribution == "dllaplace":
        return stats.laplace.logpdf(
            np.log(y), loc=np.log(y_fitted), scale=scale
        ) - np.log(y)
    elif distribution == "dls":
        return stats.t.logpdf(
            np.log(y), df=2, loc=np.log(y_fitted), scale=scale
        ) - np.log(y)
    elif distribution == "dlgnorm":
        # Implement log-generalized normal distribution
        pass
    elif distribution == "dinvgauss":
        return stats.invgauss.logpdf(
            y, mu=np.abs(y_fitted), scale=np.abs(scale / y_fitted)
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
    if loss == "MSEh":
        return np.linalg.norm(adam_errors[:, h - 1]) ** 2 / (obs_in_sample - h)
    elif loss == "TMSE":
        return np.sum(np.linalg.norm(adam_errors, axis=0) ** 2 / (obs_in_sample - h))
    elif loss == "GTMSE":
        return np.sum(
            np.log(np.linalg.norm(adam_errors, axis=0) ** 2 / (obs_in_sample - h))
        )
    elif loss == "MSCE":
        return np.sum(np.sum(adam_errors, axis=1) ** 2) / (obs_in_sample - h)
    elif loss == "MAEh":
        return np.sum(np.abs(adam_errors[:, h - 1])) / (obs_in_sample - h)
    elif loss == "TMAE":
        return np.sum(np.sum(np.abs(adam_errors), axis=0) / (obs_in_sample - h))
    elif loss == "GTMAE":
        return np.sum(np.log(np.sum(np.abs(adam_errors), axis=0) / (obs_in_sample - h)))
    elif loss == "MACE":
        return np.sum(np.abs(np.sum(adam_errors, axis=1))) / (obs_in_sample - h)
    elif loss == "HAMh":
        return np.sum(np.sqrt(np.abs(adam_errors[:, h - 1]))) / (obs_in_sample - h)
    elif loss == "THAM":
        return np.sum(
            np.sum(np.sqrt(np.abs(adam_errors)), axis=0) / (obs_in_sample - h)
        )
    elif loss == "GTHAM":
        return np.sum(
            np.log(np.sum(np.sqrt(np.abs(adam_errors)), axis=0) / (obs_in_sample - h))
        )
    elif loss == "CHAM":
        return np.sum(np.sqrt(np.abs(np.sum(adam_errors, axis=1)))) / (
            obs_in_sample - h
        )
    elif loss == "GPL":
        return np.log(np.linalg.det(adam_errors.T @ adam_errors / (obs_in_sample - h)))
    else:
        return 0


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

    # Helper function to safely compute complex logarithm
    def safe_log(x):
        return np.log(np.abs(x) + 1j * (x.imag - x.real))

    if distribution == "dnorm":
        return np.linalg.norm(errors) / np.sqrt(obs_in_sample)

    elif distribution == "dlaplace":
        return np.sum(np.abs(errors)) / obs_in_sample

    elif distribution == "ds":
        return np.sum(np.sqrt(np.abs(errors))) / (obs_in_sample * 2)

    elif distribution == "dgnorm":
        beta = other if other is not None else 2.0
        return (beta * np.sum(np.abs(errors) ** beta) / obs_in_sample) ** (1 / beta)

    elif distribution == "dalaplace":
        return np.sum(errors * (other - (errors <= 0) * 1)) / obs_in_sample

    elif distribution == "dlnorm":
        if Etype == "A":
            temp = 1 - np.sqrt(
                np.abs(
                    1
                    - np.linalg.norm(np.log(np.abs(1 + errors / y_fitted))) ** 2
                    / obs_in_sample
                )
            )
        else:  # "M"
            temp = 1 - np.sqrt(
                np.abs(1 - np.linalg.norm(np.log(1 + errors)) ** 2 / obs_in_sample)
            )
        return np.sqrt(2 * np.abs(temp))

    elif distribution == "dllaplace":
        if Etype == "A":
            return np.real(
                np.sum(np.abs(safe_log(1 + errors / y_fitted))) / obs_in_sample
            )
        else:  # "M"
            return np.sum(np.abs(np.log(1 + errors))) / obs_in_sample

    elif distribution == "dls":
        if Etype == "A":
            return np.real(
                np.sum(np.sqrt(np.abs(safe_log(1 + errors / y_fitted)))) / obs_in_sample
            )
        else:  # "M"
            return np.sum(np.sqrt(np.abs(np.log(1 + errors)))) / obs_in_sample

    elif distribution == "dlgnorm":
        if Etype == "A":
            return np.real(
                (
                    other
                    * np.sum(np.abs(safe_log(1 + errors / y_fitted)) ** other)
                    / obs_in_sample
                )
                ** (1 / other)
            )
        else:  # "M"
            return (
                other * np.sum(np.abs(safe_log(1 + errors)) ** other) / obs_in_sample
            ) ** (1 / other)

    elif distribution == "dinvgauss":
        if Etype == "A":
            return (
                np.sum((errors / y_fitted) ** 2 / (1 + errors / y_fitted))
                / obs_in_sample
            )
        else:  # "M"
            return np.sum(errors**2 / (1 + errors)) / obs_in_sample

    elif distribution == "dgamma":
        if Etype == "A":
            return np.linalg.norm(errors / y_fitted) ** 2 / obs_in_sample
        else:  # "M"
            return np.linalg.norm(errors) ** 2 / obs_in_sample

    else:
        raise ValueError(f"Unknown distribution: {distribution}")
