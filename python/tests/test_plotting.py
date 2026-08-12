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
