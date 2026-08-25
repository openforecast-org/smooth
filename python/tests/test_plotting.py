"""Unit tests for figure sizing in ``ADAM.plot()``.

Regression guard for the case where the plotting helpers hardcoded a figure
size and therefore silently ignored both ``plt.rcParams["figure.figsize"]``
and an explicit ``dpi``.
"""

from __future__ import annotations

import numpy as np
import pytest

from smooth import ADAM

matplotlib = pytest.importorskip("matplotlib")
plt = pytest.importorskip("matplotlib.pyplot")


@pytest.fixture(scope="module")
def fitted_model():
    """A cheap fitted ADAM to plot."""
    np.random.seed(42)
    y = 10 + 0.5 * np.arange(60) + np.random.randn(60) * 2
    return ADAM(model="AAN", lags=[1]).fit(y)


@pytest.fixture
def rc_figsize():
    """Set a non-default figsize for the duration of one test."""
    with plt.rc_context({"figure.figsize": (12, 6), "figure.dpi": 100}):
        yield (12, 6)


def test_rcparams_figsize_is_honoured(fitted_model, rc_figsize):
    fig = fitted_model.plot(7)
    assert tuple(fig.get_size_inches()) == rc_figsize


def test_explicit_figsize_overrides_rcparams(fitted_model, rc_figsize):
    fig = fitted_model.plot(7, figsize=(12.8, 4.8))
    assert tuple(fig.get_size_inches()) == (12.8, 4.8)


def test_figsize_and_dpi_give_requested_pixels(fitted_model):
    """figsize (inches) x dpi is the pixel size of the rendered PNG."""
    fig = fitted_model.plot(7, figsize=(12.8, 4.8), dpi=100)
    assert fig.dpi == 100
    assert fig.canvas.get_width_height() == (1280, 480)


def test_states_plot_keeps_panel_dependent_height(fitted_model, rc_figsize):
    """which=12 scales its height with the panel count, ignoring rcParams."""
    fig = fitted_model.plot(12)
    width, height = fig.get_size_inches()
    assert width == 9
    assert height == 2 * len(fig.axes)


def test_states_plot_accepts_explicit_figsize(fitted_model):
    fig = fitted_model.plot(12, figsize=(14, 7))
    assert tuple(fig.get_size_inches()) == (14, 7)


# --------------------------------------------------------------------------
# Scale-model diagnostics (R: plot.adam's `if(is.scale(x$scale)) x <- x$scale`)
# --------------------------------------------------------------------------

# The panels R hands to the scale model. 2-6, 8, 9, 13, 14 come from R's own
# gate; 15 and 16 were added in both languages together, because the squared-
# residual ACF/PACF exist to detect heteroscedasticity the scale model is
# supposed to have removed, so showing raw residuals there made a working scale
# model look like a failed one.
R_SCALE_MODEL_PANELS = {2, 3, 4, 5, 6, 8, 9, 13, 14, 15, 16}


def _heteroscedastic_fit():
    """A series whose variance itself cycles, so panel 15 has something to show."""
    from pathlib import Path

    import pandas as pd

    y = pd.read_csv(Path(__file__).parent / "data" / "sm_heteroscedastic.csv")
    model = ADAM(model="ANN", lags=[1], distribution="dnorm")
    model.fit(y["y"].to_numpy(float))
    return model


def _acf_stems(fig, n=4):
    """The ACF/PACF stem heights a panel actually drew."""
    from matplotlib.collections import LineCollection

    for collection in fig.axes[0].collections:
        if isinstance(collection, LineCollection):
            segments = collection.get_segments()
            if len(segments) > 5:
                heights = np.array([s[1][1] for s in segments])
                return heights if n is None else heights[:n]
    raise AssertionError("no ACF stems found on the figure")


def test_scale_model_panel_set_matches_r():
    from smooth.adam_general.core.plotting import _SCALE_MODEL_PANELS

    assert set(_SCALE_MODEL_PANELS) == R_SCALE_MODEL_PANELS


def test_squared_acf_uses_the_scale_models_residuals():
    """which=15 must diagnose the scale model, not the raw residuals."""
    from statsmodels.tsa.stattools import acf

    model = _heteroscedastic_fit()
    raw = np.asarray(model.residuals, dtype=float)
    scale_model = model.sm()
    model.scale_model = scale_model
    standardised = np.asarray(scale_model.residuals, dtype=float)

    with pytest.warns(UserWarning, match="scale model"):
        drawn = _acf_stems(model.plot(which=15))

    expected = acf(standardised**2, nlags=4, fft=True)[1:5]
    np.testing.assert_allclose(drawn, expected, atol=1e-8)

    # and it must *not* be the raw residuals, which still look autocorrelated
    raw_acf = acf(raw**2, nlags=4, fft=True)[1:5]
    assert abs(raw_acf[0]) > 0.2
    assert not np.allclose(drawn, raw_acf, atol=1e-3)


def test_plain_acf_stays_with_the_location_model():
    """which=10 is not a heteroscedasticity check, so R leaves it alone."""
    from statsmodels.tsa.stattools import acf

    model = _heteroscedastic_fit()
    raw = np.asarray(model.residuals, dtype=float)
    model.scale_model = model.sm()

    # No note either: R's gate fires only when a scale-diagnostic panel is asked
    # for, and which=10 is not one.
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        drawn = _acf_stems(model.plot(which=10))

    np.testing.assert_allclose(drawn, acf(raw, nlags=4, fft=True)[1:5], atol=1e-8)


@pytest.mark.parametrize("which", sorted(set(range(1, 17))))
def test_every_panel_renders_with_a_scale_model(which):
    """All sixteen panels must still draw once a scale model is attached."""
    import contextlib

    model = _heteroscedastic_fit()
    model.scale_model = model.sm()
    expects_note = which in R_SCALE_MODEL_PANELS
    context = (
        pytest.warns(UserWarning, match="scale model")
        if expects_note
        else contextlib.nullcontext()
    )
    with context:
        assert model.plot(which=which) is not None


def test_no_note_without_a_scale_model():
    import warnings as _warnings

    model = _heteroscedastic_fit()
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        assert model.plot(which=2) is not None


def test_standardised_residuals_are_actually_standardised():
    """rstandard() divides by extract_scale(), which is 1 for a scale model.

    Dividing by ``scale`` instead shrank them by that factor, so nothing ever
    reached +-1.96 and panels 2, 8 and 14 looked far tighter than the data.
    """
    model = _heteroscedastic_fit()
    scale_model = model.sm()
    standardised = np.asarray(scale_model.rstandard(), dtype=float)

    assert scale_model.extract_scale() == pytest.approx(1.0)
    # R: sd 1.0711, 8.0% outside +-1.96, range [-4.49, 3.20]
    assert standardised.std(ddof=1) == pytest.approx(1.0711, abs=1e-3)
    assert np.mean(np.abs(standardised) > 1.96) == pytest.approx(0.08, abs=0.005)
    assert standardised.min() == pytest.approx(-4.49, abs=0.01)


def test_qqline_uses_quartiles_not_least_squares():
    """R's qqline goes through Q1 and Q3; a regression line is tilted by the
    very outliers the plot exists to reveal."""
    model = _heteroscedastic_fit()
    model.scale_model = model.sm()
    with pytest.warns(UserWarning, match="scale model"):
        fig = model.plot(which=6)

    line = fig.axes[0].lines[0]
    x, y = line.get_xdata(), line.get_ydata()
    slope = (y[1] - y[0]) / (x[1] - x[0])
    intercept = y[0] - slope * x[0]
    # R: slope 0.821242, intercept 0.337407
    assert slope == pytest.approx(0.821242, abs=1e-6)
    assert intercept == pytest.approx(0.337407, abs=1e-6)


def test_acf_lag_count_matches_r():
    """R's acf()/pacf() default to floor(10 * log10(n)) lags."""
    model = _heteroscedastic_fit()
    expected = int(np.floor(10 * np.log10(model.nobs)))
    for which in (10, 11, 15, 16):
        stems = _acf_stems(model.plot(which=which), n=None)
        assert len(stems) == expected, f"which={which}"


def test_no_divider_line_without_a_forecast():
    """The red divider marks where the forecast starts, so h=0 has none."""
    from pathlib import Path

    import pandas as pd

    y = pd.read_csv(Path(__file__).parent / "data" / "sm_heteroscedastic.csv")
    y = y["y"].to_numpy(float)

    def _has_vline(fig):
        return any(
            len(set(line.get_xdata())) == 1 and line.get_color() == "#FF0000"
            for line in fig.axes[0].lines
        )

    no_forecast = ADAM(model="ANN", lags=[1], distribution="dnorm")
    no_forecast.fit(y)
    assert not _has_vline(no_forecast.plot(which=7))

    with_forecast = ADAM(
        model="ANN", lags=[1], distribution="dnorm", h=12, holdout=True
    )
    with_forecast.fit(y)
    assert _has_vline(with_forecast.plot(which=7))


def test_states_plot_drops_the_initial_state():
    """The states matrix carries lags_model_max leading columns that hold the
    initialiser's seed, not anything the fitter used. They must not be drawn."""
    model = _heteroscedastic_fit()
    states = np.asarray(model.states, dtype=float)
    assert states.shape[1] == model.nobs + 1

    figs = model.plot(which=12)
    figs = figs if isinstance(figs, list) else getattr(figs, "figures", [figs])
    drawn = [
        line.get_ydata()
        for fig in figs
        for ax in fig.axes
        for line in ax.get_lines()
        if len(line.get_ydata()) == model.nobs
    ]
    assert drawn, "no series of length nobs was drawn"
    # the stale seed must not appear anywhere
    assert not any(np.isclose(series[0], states[0, 0]) for series in drawn)
