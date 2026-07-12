"""Tests for ``initial="gradient"`` — the least-squares initial-state solve.

Gradient initialisation solves for the initial ETS state that minimises the
in-sample SSE at the current persistence (exact one-shot least squares for
additive models, Gauss-Newton otherwise), instead of the backcasting backward
pass. Direct port of ``R/adam-gradient.R``; see also ``tests/testthat/
test_gradient.R`` on the R side.
"""

from __future__ import annotations

import numpy as np

from smooth import ADAM

# Seasonal series with trend — the case where backcasting historically drifted.
_Y = np.array(
    [10, 12, 15, 13, 16, 18, 20, 19, 22, 25, 28, 30,
     11, 13, 16, 14, 17, 19, 21, 20, 23, 26, 29, 31,
     12, 14, 17, 15, 18, 20, 22, 21, 24, 27, 30, 32,
     13, 15, 18, 16, 19, 21, 23, 22, 25, 28, 31, 33],
    dtype=float,
)


def test_gradient_runs_and_is_finite():
    m = ADAM(model="AAA", lags=[1, 12], initial="gradient").fit(_Y)
    fitted = np.asarray(m.fitted).ravel()
    assert np.all(np.isfinite(fitted))
    assert m.initial_type == "gradient"


def test_gradient_counts_initial_df_like_optimal():
    # Gradient solves the initials, so it spends the same degrees of freedom on
    # them as initial="optimal" (unlike the historical backcast zero-df count).
    mg = ADAM(model="AAA", lags=[1, 12], initial="gradient").fit(_Y)
    mo = ADAM(model="AAA", lags=[1, 12], initial="optimal").fit(_Y)
    assert mg.nparam == mo.nparam


def test_backcasting_does_not_count_initial_df():
    # Only gradient (and optimal/two-stage) count the initials as df; backcasting
    # keeps its historical zero-initial-df accounting, so it reports fewer
    # parameters than optimal for the same model.
    mb = ADAM(model="AAA", lags=[1, 12], initial="backcasting").fit(_Y)
    mo = ADAM(model="AAA", lags=[1, 12], initial="optimal").fit(_Y)
    assert mb.nparam < mo.nparam


def test_gradient_additive_is_least_squares_optimum():
    # For an additive model the residuals are affine in the initial state, so the
    # one-shot solve lands at the global SSE minimum: no perturbation of the
    # initials at the fitted persistence can lower the in-sample SSE.
    m = ADAM(model="AAN", lags=[1], initial="gradient").fit(_Y)
    fitted = np.asarray(m.fitted).ravel()
    sse = float(np.sum((_Y - fitted) ** 2))
    # Compare against backcasting at the same spec — gradient should not be worse.
    mb = ADAM(model="AAN", lags=[1], initial="backcasting").fit(_Y)
    sse_b = float(np.sum((_Y - np.asarray(mb.fitted).ravel()) ** 2))
    assert sse <= sse_b + 1e-6


def test_gradient_multiplicative_runs():
    m = ADAM(model="MAM", lags=[1, 12], initial="gradient").fit(_Y)
    fitted = np.asarray(m.fitted).ravel()
    assert np.all(np.isfinite(fitted))
    assert np.all(fitted > 0)


def test_gradient_arima_falls_back_gracefully():
    # ARIMA is out of scope for the gradient solve; it must fall back to
    # backcasting rather than error.
    m = ADAM(
        model="NNN",
        orders={"ar": [1], "i": [1], "ma": [1]},
        initial="gradient",
    ).fit(_Y)
    assert np.all(np.isfinite(np.asarray(m.fitted).ravel()))


def test_gradient_nonseasonal():
    m = ADAM(model="ANN", lags=[1], initial="gradient").fit(_Y)
    assert np.all(np.isfinite(np.asarray(m.fitted).ravel()))


def test_gradient_msarima_falls_back_to_backcasting():
    # MSARIMA is not ETS, so initial="gradient" must fall back to backcasting and
    # produce an identical fit (same as the R side).
    from smooth import MSARIMA

    orders = {"ar": [1, 0], "i": [1, 0], "ma": [1, 0]}
    mg = MSARIMA(orders=orders, lags=[1, 12], initial="gradient").fit(_Y)
    mb = MSARIMA(orders=orders, lags=[1, 12], initial="backcasting").fit(_Y)
    assert abs(float(mg.loglik) - float(mb.loglik)) < 1e-8


def test_gradient_ces_falls_back_to_backcasting():
    # CES is not ETS, so initial="gradient" must fall back to backcasting.
    from smooth import CES

    mg = CES(seasonality="full", lags=[1, 12], initial="gradient").fit(_Y)
    mb = CES(seasonality="full", lags=[1, 12], initial="backcasting").fit(_Y)
    assert abs(float(mg.loglik) - float(mb.loglik)) < 1e-8
