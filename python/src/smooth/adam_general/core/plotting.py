"""Diagnostic plots for fitted ADAM models — Python equivalent of R's plot.adam()."""

import math

import numpy as np
import scipy.stats as sp_stats
from statsmodels.tsa.stattools import acf as _acf
from statsmodels.tsa.stattools import pacf as _pacf

_LOG_DISTS = {"dinvgauss", "dgamma", "dlnorm", "dllaplace", "dls", "dlgnorm"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _lazy_mpl():
    try:
        import matplotlib.figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: F401

        return matplotlib.figure
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plot(). Install with: pip install matplotlib"
        ) from exc


def _attach_canvas(fig):
    """Attach an Agg canvas and add _repr_png_ so Jupyter renders the figure."""
    import io

    from matplotlib.backends.backend_agg import FigureCanvasAgg

    FigureCanvasAgg(fig)

    def _repr_png_(self=fig):
        buf = io.BytesIO()
        self.canvas.print_png(buf)
        return buf.getvalue()

    fig._repr_png_ = _repr_png_
    return fig


class PlotCollection:
    """Container for multiple diagnostic figures with Jupyter display support."""

    def __init__(self, figs):
        self._figs = figs

    def __getitem__(self, i):
        return self._figs[i]

    def __len__(self):
        return len(self._figs)

    def __iter__(self):
        return iter(self._figs)

    def __repr__(self):
        return f"[{len(self._figs)} diagnostic plot(s)]"

    def _ipython_display_(self):
        import io

        try:
            from IPython.display import Image, display
        except ImportError:
            return
        for fig in self._figs:
            buf = io.BytesIO()
            fig.canvas.print_png(buf)
            display(Image(data=buf.getvalue()))


def _is_log_dist(dist):
    return dist in _LOG_DISTS


def _log_transform(y, dist):
    """Log-transform residuals for multiplicative / log distributions."""
    y = np.asarray(y, dtype=float).copy()
    if dist in {"dinvgauss", "dgamma"}:
        y = np.log(y)
    elif dist in {"dlnorm", "dllaplace", "dls", "dlgnorm"}:
        y = np.log(y)
    return y


def _lowess_line(ax, x, y):
    """Draw a red LOWESS smoothing line on ax."""
    from smooth.lowess import lowess

    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 4:  # noqa: PLR2004
        return
    result = lowess(x[mask], y[mask])
    ax.plot(result["x"], result["y"], color="red", lw=1.5)


def _residual_bounds(ax, statistic, x_range):
    """Draw grey hatched band and red dashed bounds on ax (scatter plots)."""
    lo, hi = statistic
    ax.axhline(0, color="grey", linestyle="--", lw=1)
    ax.axhline(lo, color="red", linestyle="--", lw=1)
    ax.axhline(hi, color="red", linestyle="--", lw=1)
    ax.fill_between(
        x_range, lo, hi, color="lightgrey", alpha=0.5, hatch="///", edgecolor="none"
    )


def _acf_pacf_bounds(nobs, level):
    z = sp_stats.norm.ppf((1 + level) / 2)
    bound = z / math.sqrt(nobs)
    return -bound, bound


def _make_fig(mpl_figure, **kwargs):
    figsize = kwargs.pop("figsize", (7, 5))
    return _attach_canvas(mpl_figure.Figure(figsize=figsize)), kwargs


def _state_labels(model):
    """Derive human-readable labels for each row of model.states."""
    comp = model._components
    is_trendy = comp.get("model_is_trendy", False)
    n_seasonal = comp.get("components_number_ets_seasonal", 0)
    n_arima = comp.get("components_number_arima", 0)

    labels = ["Level"]
    if is_trendy:
        labels.append("Trend")
    labels += [f"Seasonal {i + 1}" for i in range(n_seasonal)]
    labels += [f"ARIMA {i + 1}" for i in range(n_arima)]
    return labels


# ---------------------------------------------------------------------------
# which=1 — Actuals vs Fitted
# ---------------------------------------------------------------------------


def _plot1(model, ax, lowess, **kw):
    fitted = np.asarray(model.fitted, dtype=float)
    actuals = np.asarray(model.actuals, dtype=float)
    mask = ~(np.isnan(fitted) | np.isnan(actuals))
    x, y = fitted[mask], actuals[mask]

    ax.scatter(x, y, s=10, color="black")
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lims, lims, color="grey", linestyle="--", lw=1.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    if lowess:
        _lowess_line(ax, x, y)
    ax.set_title(kw.get("main", "Actuals vs Fitted"))
    ax.set_xlabel(kw.get("xlab", "Fitted"))
    ax.set_ylabel(kw.get("ylab", "Actuals"))


# ---------------------------------------------------------------------------
# which=2/3 — Standardised / Studentised Residuals vs Fitted
# ---------------------------------------------------------------------------


def _plot2(model, ax, resid_type, level, lowess, legend, **kw):
    fitted = np.asarray(model.fitted, dtype=float)
    resid = model.rstandard() if resid_type == "rstandard" else model.rstudent()
    resid = np.asarray(resid, dtype=float)
    dist = model.distribution_

    mask = ~(np.isnan(fitted) | np.isnan(resid))
    x, y = fitted[mask], resid[mask]

    if _is_log_dist(dist):
        y = np.log(y)

    outlier_result = model.outlierdummy(level=level, type=resid_type)
    statistic = outlier_result.statistic.copy()
    if dist in {"dinvgauss", "dgamma"}:
        statistic = np.log(statistic)

    outlier_idx = np.where((y > statistic[1]) | (y < statistic[0]))[0]

    y_pad = max(abs(y)) * 1.2 if len(y) else 1
    ax.set_ylim(-y_pad, y_pad)

    x_range = np.array([x.min() - x.std(), x.max() + x.std()])
    _residual_bounds(ax, statistic, x_range)
    ax.scatter(x, y, s=8, color="black")
    if len(outlier_idx):
        ax.scatter(x[outlier_idx], y[outlier_idx], s=20, color="black", zorder=5)
        for i in outlier_idx:
            va = "bottom" if y[i] > 0 else "top"
            ax.annotate(str(i + 1), (x[i], y[i]), fontsize=7, va=va)
    if lowess:
        _lowess_line(ax, x, y)

    y_name = "Standardised" if resid_type == "rstandard" else "Studentised"
    log_prefix = "log(" if _is_log_dist(dist) else ""
    log_suffix = ")" if _is_log_dist(dist) else ""
    default_title = f"{log_prefix}{y_name} Residuals{log_suffix} vs Fitted"
    ax.set_title(kw.get("main", default_title))
    ax.set_xlabel(kw.get("xlab", "Fitted"))
    ax.set_ylabel(kw.get("ylab", f"{y_name} Residuals"))

    if legend:
        from matplotlib.lines import Line2D

        pct = f"{level * 100:.0f}% bounds"
        handles = [
            Line2D([0], [0], color="red", linestyle="--", label=pct),
            Line2D([0], [0], color="black", marker="o", ls="None", label="Outliers"),
        ]
        if lowess:
            handles.append(Line2D([0], [0], color="red", label="LOWESS"))
        ax.legend(handles=handles, fontsize=8)


# ---------------------------------------------------------------------------
# which=4/5 — |Residuals| or Residuals² vs Fitted
# ---------------------------------------------------------------------------


def _plot3(model, ax, type_, lowess, **kw):
    fitted = np.asarray(model.fitted, dtype=float)
    resid = np.asarray(model.residuals, dtype=float)
    dist = model.distribution_

    if _is_log_dist(dist):
        resid = np.log(resid)
    if type_ == "abs":
        y = np.abs(resid)
        default_title = "|Residuals| vs Fitted"
        default_ylab = "|Residuals|"
    else:
        y = resid**2
        default_title = "Residuals² vs Fitted"
        default_ylab = "Residuals²"

    mask = ~(np.isnan(fitted) | np.isnan(y))
    x, y = fitted[mask], y[mask]

    ax.scatter(x, y, s=8, color="black")
    if lowess:
        _lowess_line(ax, x, y)
    ax.set_title(kw.get("main", default_title))
    ax.set_xlabel(kw.get("xlab", "Fitted"))
    ax.set_ylabel(kw.get("ylab", default_ylab))


# ---------------------------------------------------------------------------
# which=6 — Q-Q Plot
# ---------------------------------------------------------------------------


def _qqplot_dist(model):
    """Return (scipy_dist_frozen, title) for the model's distribution."""
    dist = model.distribution_
    scale = model.scale

    if dist == "dnorm":
        return sp_stats.norm(), "QQ plot of Normal distribution"

    if dist == "dlnorm":
        s = scale
        return (
            sp_stats.lognorm(s=s, scale=math.exp(-(s**2) / 2)),
            "QQ plot of Log-Normal distribution",
        )

    if dist == "dlaplace":
        return sp_stats.laplace(scale=scale), "QQ-plot of Laplace distribution"

    if dist == "ds":
        # S distribution not in scipy; fall back to normal
        return sp_stats.norm(), "QQ-plot of S distribution (Normal approx.)"

    if dist == "dgnorm":
        shape = model._config.get("gnorm_shape", 2.0)
        return (
            sp_stats.gennorm(beta=shape, scale=scale),
            f"QQ-plot of Generalised Normal (shape={shape:.3f})",
        )

    if dist == "dinvgauss":
        return (
            sp_stats.invgauss(mu=1, scale=scale),
            "QQ-plot of Inverse Gaussian distribution",
        )

    if dist == "dgamma":
        a = 1.0 / scale if scale > 0 else 1.0
        return sp_stats.gamma(a=a, scale=scale), "QQ-plot of Gamma distribution"

    # Fallback
    return sp_stats.norm(), f"QQ plot ({dist})"


def _plot4(model, ax, **kw):
    resid = np.asarray(model.residuals, dtype=float)
    resid = resid[~np.isnan(resid)]

    dist_frozen, default_title = _qqplot_dist(model)
    (osm, osr), (slope, intercept, _) = sp_stats.probplot(resid, dist=dist_frozen)

    ax.scatter(osm, osr, s=8, color="black")
    line_x = np.array([osm[0], osm[-1]])
    ax.plot(line_x, slope * line_x + intercept, color="red", lw=1.5)
    ax.set_title(kw.get("main", default_title))
    ax.set_xlabel(kw.get("xlab", "Theoretical Quantiles"))
    ax.set_ylabel(kw.get("ylab", "Sample Quantiles"))


# ---------------------------------------------------------------------------
# which=7 — Fitted over time (replaces graphmaker)
# ---------------------------------------------------------------------------


def _plot5(model, ax, legend, **kw):
    actuals = np.asarray(model.actuals, dtype=float)
    fitted = np.asarray(model.fitted, dtype=float)
    n = len(actuals)
    t_in = np.arange(n)

    # Continuous actuals line: training + holdout joined
    holdout = model.holdout_data
    if holdout is not None:
        holdout = np.asarray(holdout, dtype=float).ravel()
        t_h = np.arange(n, n + len(holdout))
        full_t = np.concatenate([t_in, t_h])
        full_y = np.concatenate([actuals, holdout])
    else:
        full_t, full_y = t_in, actuals
    ax.plot(full_t, full_y, color="#000000", lw=1.5, label="Actuals")

    # Fitted values — dashed purple on top (#A020F0 matches R's "purple")
    ax.plot(t_in, fitted, color="#A020F0", lw=1.5, linestyle="--", label="Fitted")

    # Horizontal line at the last in-sample observation
    ax.axvline(n - 1, color="#FF0000", lw=0.8)

    # Forecast mean: prefer manual predict() result, fall back to auto-forecast
    fc = getattr(model, "_forecast_results", None) or getattr(
        model, "_auto_forecast", None
    )
    if fc is not None and hasattr(fc, "mean") and fc.mean is not None:
        fc_mean = np.asarray(fc.mean, dtype=float).ravel()
        t_f = np.arange(n, n + len(fc_mean))
        ax.plot(t_f, fc_mean, color="#0000FF", lw=1.5, label="Forecast")

        lower = getattr(fc, "lower", None)
        upper = getattr(fc, "upper", None)
        if lower is not None or upper is not None:
            lower_arr = np.asarray(lower, dtype=float) if lower is not None else None
            upper_arr = np.asarray(upper, dtype=float) if upper is not None else None
            ref = lower_arr if lower_arr is not None else upper_arr
            k = ref.shape[1] if ref.ndim > 1 else 1
            shades = np.linspace(0.35, 0.70, k)
            for i in range(k):
                grey = str(shades[i])
                lbl = "Bounds" if i == 0 else None
                if lower_arr is not None:
                    col = lower_arr[:, i] if lower_arr.ndim > 1 else lower_arr.ravel()
                    ax.plot(t_f, col, color=grey, lw=1, linestyle="--", label=lbl)
                    lbl = None
                if upper_arr is not None:
                    j = -(i + 1)
                    col = upper_arr[:, j] if upper_arr.ndim > 1 else upper_arr.ravel()
                    ax.plot(t_f, col, color=grey, lw=1, linestyle="--", label=lbl)

    ax.set_title(kw.get("main", model.model_name))
    ax.set_xlabel(kw.get("xlab", "Time"))
    ax.set_ylabel(kw.get("ylab", "Value"))
    if legend:
        ax.legend(fontsize=8)


# ---------------------------------------------------------------------------
# which=8/9 — Standardised / Studentised Residuals vs Time
# ---------------------------------------------------------------------------


def _plot6(model, ax, resid_type, level, lowess, legend, **kw):
    resid = model.rstandard() if resid_type == "rstandard" else model.rstudent()
    resid = np.asarray(resid, dtype=float)
    dist = model.distribution_

    if _is_log_dist(dist):
        resid = np.log(resid)

    outlier_result = model.outlierdummy(level=level, type=resid_type)
    statistic = outlier_result.statistic.copy()
    if dist in {"dinvgauss", "dgamma"}:
        statistic = np.log(statistic)

    t = np.arange(len(resid))
    y_pad = max(abs(resid[~np.isnan(resid)])) * 1.2 if np.any(~np.isnan(resid)) else 1
    ax.set_ylim(-y_pad, y_pad)

    ax.fill_between(
        t,
        statistic[0],
        statistic[1],
        color="lightgrey",
        alpha=0.5,
        hatch="///",
        edgecolor="none",
    )
    ax.plot(t, resid, color="black", lw=1)
    ax.axhline(0, color="grey", linestyle="--", lw=1)
    ax.axhline(statistic[0], color="red", linestyle="--", lw=1)
    ax.axhline(statistic[1], color="red", linestyle="--", lw=1)

    outside = (resid > statistic[1]) | (resid < statistic[0])
    outlier_idx = np.where(~np.isnan(resid) & outside)[0]
    if len(outlier_idx):
        ax.scatter(t[outlier_idx], resid[outlier_idx], s=20, color="black", zorder=5)
        for i in outlier_idx:
            va = "bottom" if resid[i] > 0 else "top"
            ax.annotate(str(i + 1), (t[i], resid[i]), fontsize=7, va=va)

    if lowess:
        y_fill = resid.copy()
        if np.any(np.isnan(y_fill)):
            y_fill[np.isnan(y_fill)] = np.nanmean(y_fill)
        _lowess_line(ax, t.astype(float), y_fill)

    y_name = "Standardised" if resid_type == "rstandard" else "Studentised"
    log_prefix = "log(" if _is_log_dist(dist) else ""
    log_suffix = ")" if _is_log_dist(dist) else ""
    default_title = f"{log_prefix}{y_name} Residuals{log_suffix} vs Time"
    ax.set_title(kw.get("main", default_title))
    ax.set_xlabel(kw.get("xlab", "Time"))
    ax.set_ylabel(kw.get("ylab", f"{y_name} Residuals"))

    if legend:
        from matplotlib.lines import Line2D

        pct = f"{level * 100:.0f}% bounds"
        handles = [
            Line2D([0], [0], color="black", label="Residuals"),
            Line2D([0], [0], color="red", linestyle="--", label=pct),
        ]
        ax.legend(handles=handles, fontsize=8)


# ---------------------------------------------------------------------------
# which=10/11/15/16 — ACF / PACF (residuals or squared)
# ---------------------------------------------------------------------------


def _plot7(model, ax, type_, squared, level, **kw):
    resid = np.asarray(model.residuals, dtype=float)
    resid = resid[~np.isnan(resid)]

    if squared:
        resid = resid**2

    nobs = len(resid)
    nlags = min(40, nobs // 2)

    if type_ == "acf":
        values = _acf(resid, nlags=nlags, fft=True)[1:]  # skip lag 0
        default_title = "ACF of Squared Residuals" if squared else "ACF of Residuals"
        default_ylab = "ACF"
    else:
        values = _pacf(resid, nlags=nlags)
        default_title = "PACF of Squared Residuals" if squared else "PACF of Residuals"
        default_ylab = "PACF"

    lags = np.arange(1, len(values) + 1)
    lo, hi = _acf_pacf_bounds(nobs, level)

    ax.vlines(lags, 0, values, colors="black", lw=1.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(lo, color="red", linestyle="--", lw=1)
    ax.axhline(hi, color="red", linestyle="--", lw=1)
    ax.set_ylim(-1, 1)

    sig_idx = np.where((values > hi) | (values < lo))[0]
    if len(sig_idx):
        ax.scatter(lags[sig_idx], values[sig_idx], s=20, color="black", zorder=5)
        for i in sig_idx:
            va = "bottom" if values[i] > 0 else "top"
            ax.annotate(str(lags[i]), (lags[i], values[i]), fontsize=7, va=va)

    ax.set_title(kw.get("main", default_title))
    ax.set_xlabel(kw.get("xlab", "Lags"))
    ax.set_ylabel(kw.get("ylab", default_ylab))


# ---------------------------------------------------------------------------
# which=12 — States over time
# ---------------------------------------------------------------------------


def _plot8(model, mpl_figure, **kw):
    """Returns list[Figure] — one figure per batch of 10 states."""
    if model.is_combined:
        print("Combination of models was done. Nothing to plot for states.")
        return []

    states = model.states  # shape (n_states, T+1); col 0 is initial state
    actuals = np.asarray(model.actuals, dtype=float).ravel()
    resid = np.asarray(model.residuals, dtype=float).ravel()
    n_obs = len(actuals)

    # Drop initial state column so states align with actuals (both length T)
    states_aligned = states[:, 1 : n_obs + 1]

    # Prepend actuals, append residuals (matching R's plot8)
    all_series = np.vstack(
        [actuals[np.newaxis, :], states_aligned, resid[np.newaxis, :]]
    )

    state_labels = _state_labels(model)
    all_labels = ["Actuals"] + state_labels + ["Residuals"]

    # Trim to actual length
    n_series = all_series.shape[0]
    all_labels = all_labels[:n_series]

    batch_size = 10
    n_batches = math.ceil(n_series / batch_size)
    figs = []

    for b in range(n_batches):
        batch = all_series[b * batch_size : (b + 1) * batch_size]
        labels = all_labels[b * batch_size : (b + 1) * batch_size]
        n = len(labels)

        fig = _attach_canvas(mpl_figure.Figure(figsize=kw.get("figsize", (9, 2 * n))))
        axes = fig.subplots(n, 1, sharex=True)
        if n == 1:
            axes = [axes]

        if n_batches > 1:
            title = kw.get("main", f"States of {model.model_name}, part {b + 1}")
        else:
            title = kw.get("main", f"States of {model.model_name}")

        fig.suptitle(title, fontsize=11)

        for i, (series, label) in enumerate(zip(batch, labels)):
            t = np.arange(series.shape[0])
            axes[i].plot(t, series, color="black", lw=1)
            axes[i].set_ylabel(label, fontsize=8)
            axes[i].tick_params(labelsize=7)

        axes[-1].set_xlabel("Time")
        fig.tight_layout()
        figs.append(fig)

    return figs


# ---------------------------------------------------------------------------
# which=13/14 — |Standardised Residuals| or Standardised Residuals² vs Fitted
# ---------------------------------------------------------------------------


def _plot9(model, ax, type_, lowess, **kw):
    fitted = np.asarray(model.fitted, dtype=float)
    resid = np.asarray(model.rstandard(), dtype=float)
    dist = model.distribution_

    if dist in {"dinvgauss", "dgamma"}:
        resid = np.log(resid)

    if type_ == "abs":
        y = np.abs(resid)
        if _is_log_dist(dist):
            default_title = "|log(Standardised Residuals)| vs Fitted"
            default_ylab = "|log(Standardised Residuals)|"
        else:
            default_title = "|Standardised Residuals| vs Fitted"
            default_ylab = "|Standardised Residuals|"
    else:
        y = resid**2
        if _is_log_dist(dist):
            default_title = "log(Standardised Residuals)² vs Fitted"
            default_ylab = "log(Standardised Residuals)²"
        else:
            default_title = "Standardised Residuals² vs Fitted"
            default_ylab = "Standardised Residuals²"

    mask = ~(np.isnan(fitted) | np.isnan(y))
    x, y = fitted[mask], y[mask]

    ax.scatter(x, y, s=8, color="black")
    ax.axhline(0, color="grey", linestyle="--", lw=1)
    if lowess:
        _lowess_line(ax, x, y)
    ax.set_title(kw.get("main", default_title))
    ax.set_xlabel(kw.get("xlab", "Fitted"))
    ax.set_ylabel(kw.get("ylab", default_ylab))


# ---------------------------------------------------------------------------
# Dispatch table and main entry point
# ---------------------------------------------------------------------------

_WHICH_MAP = {
    1: (_plot1, {}),
    2: (_plot2, {"resid_type": "rstandard"}),
    3: (_plot2, {"resid_type": "rstudent"}),
    4: (_plot3, {"type_": "abs"}),
    5: (_plot3, {"type_": "squared"}),
    6: (_plot4, {}),
    7: (_plot5, {}),
    8: (_plot6, {"resid_type": "rstandard"}),
    9: (_plot6, {"resid_type": "rstudent"}),
    10: (_plot7, {"type_": "acf", "squared": False}),
    11: (_plot7, {"type_": "pacf", "squared": False}),
    12: None,  # special — _plot8 returns list[Figure]
    13: (_plot9, {"type_": "abs"}),
    14: (_plot9, {"type_": "squared"}),
    15: (_plot7, {"type_": "acf", "squared": True}),
    16: (_plot7, {"type_": "pacf", "squared": True}),
}


def plot_adam(model, which, level, legend, lowess, **kwargs):
    """Create diagnostic plots for a fitted ADAM model.

    Parameters
    ----------
    model : ADAM
        Fitted ADAM instance.
    which : int or list of int
        Plot indices (1–16).
    level : float
        Confidence level for bounds.
    legend : bool
        Show legends.
    lowess : bool
        Draw LOWESS lines on scatter plots.
    **kwargs
        Passed to matplotlib (e.g. figsize).

    Returns
    -------
    matplotlib.figure.Figure or list[matplotlib.figure.Figure]
    """
    mpl_figure = _lazy_mpl()

    if isinstance(which, (int, np.integer)):
        which = [int(which)]
    else:
        which = [int(w) for w in which]

    figs = []
    for w in which:
        if w not in _WHICH_MAP:
            raise ValueError(f"which={w} is not valid. Choose 1–16.")

        if w == 12:
            figs.extend(_plot8(model, mpl_figure, **kwargs))
            continue

        fn, extra = _WHICH_MAP[w]
        fig, kw = _make_fig(mpl_figure, **kwargs)
        ax = fig.add_subplot(111)

        call_kw = {**extra}
        # Thread common params where needed
        if "level" in fn.__code__.co_varnames:
            call_kw["level"] = level
        if "legend" in fn.__code__.co_varnames:
            call_kw["legend"] = legend
        if "lowess" in fn.__code__.co_varnames:
            call_kw["lowess"] = lowess

        fn(model, ax, **call_kw, **kw)
        fig.tight_layout()
        figs.append(fig)

    return figs[0] if len(figs) == 1 else PlotCollection(figs)
