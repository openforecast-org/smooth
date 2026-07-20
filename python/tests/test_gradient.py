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
    [
        10,
        12,
        15,
        13,
        16,
        18,
        20,
        19,
        22,
        25,
        28,
        30,
        11,
        13,
        16,
        14,
        17,
        19,
        21,
        20,
        23,
        26,
        29,
        31,
        12,
        14,
        17,
        15,
        18,
        20,
        22,
        21,
        24,
        27,
        30,
        32,
        13,
        15,
        18,
        16,
        19,
        21,
        23,
        22,
        25,
        28,
        31,
        33,
    ],
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


def test_backcasting_counts_initial_df_like_optimal():
    # Initial states count towards df identically however they are obtained:
    # backcasting, gradient and optimal all report the same number of parameters
    # for the same model (structural df).
    mb = ADAM(model="AAA", lags=[1, 12], initial="backcasting").fit(_Y)
    mo = ADAM(model="AAA", lags=[1, 12], initial="optimal").fit(_Y)
    mg = ADAM(model="AAA", lags=[1, 12], initial="gradient").fit(_Y)
    assert mb.nparam == mo.nparam
    assert mg.nparam == mo.nparam


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


def test_gradient_msarima_solves_additive_arima_initials():
    # MSARIMA is an additive SSOE model, so initial="gradient" profiles the
    # ARIMA initials by least squares (matching optimal at fixed dynamics and
    # differing from backcasting), same as the R side.
    from smooth import MSARIMA

    def sse(m):
        return float(np.sum(np.asarray(m.residuals) ** 2))

    af = {"ar": [0.4], "ma": [0.3]}
    kw = dict(orders={"ar": [1], "i": [1], "ma": [1]}, lags=[1], arma=af, loss="MSE")
    mo = MSARIMA(initial="optimal", **kw).fit(_Y)
    mb = MSARIMA(initial="backcasting", **kw).fit(_Y)
    mg = MSARIMA(initial="gradient", **kw).fit(_Y)
    assert abs(sse(mg) - sse(mo)) < 1e-2 * max(1.0, sse(mo))
    assert abs(sse(mg) - sse(mb)) > 1e-5


def test_gradient_ces_solves_additive_ssoe_initials():
    # CES is an additive SSOE model, so initial="gradient" profiles the initials
    # by least squares (residuals are affine in the initial profile): it must run
    # (not fall back), differ from backcasting, and not lose on in-sample SSE.
    from smooth import CES

    def sse(m):
        return float(np.sum(np.asarray(m.residuals) ** 2))

    mg = CES(seasonality="none", initial="gradient", loss="MSE").fit(_Y)
    mb = CES(seasonality="none", initial="backcasting", loss="MSE").fit(_Y)
    assert np.isfinite(sse(mg))
    assert abs(sse(mg) - sse(mb)) > 1e-6  # genuinely solved, not a fallback
    assert sse(mg) <= sse(mb) * 1.001 + 1e-6  # no worse than backcasting


# --- Loss-aware solve -------------------------------------------------------

_RNG = np.random.default_rng(41)
_TREND = 100 + np.cumsum(_RNG.normal(0, 2, 72))
_SEAS = np.tile([5, -3, 2, -4, 6, -6, 3, -1, 2, -2, 4, -6], 6)
_YL = _TREND + _SEAS
_YLM = _TREND * np.tile(
    [1.05, 0.97, 1.02, 0.96, 1.06, 0.94, 1.03, 0.99, 1.02, 0.98, 1.04, 0.94], 6
)


def _loss_value(model):
    return float(model._adam_estimated["CF_value"])


def test_gradient_profiles_robust_one_step_losses():
    for loss in ["MAE", "HAM"]:
        mb = ADAM(model="AAA", lags=[12], initial="backcasting", loss=loss).fit(_YL)
        mg = ADAM(model="AAA", lags=[12], initial="gradient", loss=loss).fit(_YL)
        assert _loss_value(mg) < _loss_value(mb)


def test_gradient_profiles_multiplicative_likelihoods():
    for dist in [None, "dlnorm", "dinvgauss"]:
        kw = {} if dist is None else {"distribution": dist}
        mb = ADAM(model="MAM", lags=[12], initial="backcasting", **kw).fit(_YLM)
        mg = ADAM(model="MAM", lags=[12], initial="gradient", **kw).fit(_YLM)
        assert _loss_value(mg) < _loss_value(mb) + 1e-8


def test_gradient_solves_additive_multistep_losses():
    for loss in ["MSEh", "TMSE", "GTMSE", "MSCE", "GPL"]:
        mb = ADAM(model="AAA", lags=[12], initial="backcasting", loss=loss, h=6).fit(
            _YL
        )
        mg = ADAM(model="AAA", lags=[12], initial="gradient", loss=loss, h=6).fit(_YL)
        assert _loss_value(mg) < _loss_value(mb)


def test_gradient_loss_code_mapping_matches_r():
    # The mapping table must stay mirror-identical to adam_gradientLossCode in
    # R/adam-gradient.R -- this pins the full grid.
    from smooth.adam_general.core.utils.gradient import adam_gradient_loss_code

    grid = [
        (("MSE", "default", "A", None, 0, False), ("S", 0.0)),
        (("MAE", "default", "A", None, 0, False), ("A", 0.0)),
        (("HAM", "default", "A", None, 0, False), ("H", 0.0)),
        (("likelihood", "default", "A", None, 0, False), ("S", 0.0)),
        (("likelihood", "default", "M", None, 0, False), ("g", 0.0)),
        (("likelihood", "dlaplace", "A", None, 0, False), ("A", 0.0)),
        (("likelihood", "ds", "M", None, 0, False), ("H", 0.0)),
        (("likelihood", "dgnorm", "A", 1.5, 0, False), ("G", 1.5)),
        (("likelihood", "dgnorm", "A", None, 0, False), ("S", 0.0)),
        (("likelihood", "dlnorm", "M", None, 0, False), ("l", 0.0)),
        (("likelihood", "dlnorm", "A", None, 0, False), ("S", 0.0)),
        (("likelihood", "dgamma", "M", None, 0, False), ("g", 0.0)),
        (("likelihood", "dinvgauss", "M", None, 0, False), ("i", 0.0)),
        (("LASSO", "default", "A", None, 0, False), ("S", 0.0)),
        (("MSEh", "default", "A", None, 6, True), ("h", 6.0)),
        (("TMSE", "default", "A", None, 6, True), ("T", 6.0)),
        (("GTMSE", "default", "A", None, 6, True), ("t", 6.0)),
        (("MSCE", "default", "A", None, 6, True), ("C", 6.0)),
        (("GPL", "default", "A", None, 6, True), ("P", 6.0)),
        (("MAEh", "default", "A", None, 6, True), ("S", 0.0)),
    ]
    for args, expected in grid:
        assert adam_gradient_loss_code(*args) == expected
    assert adam_gradient_loss_code("custom", "default", "A", None, 0, False) is None
    assert (
        adam_gradient_loss_code(lambda a, f, b: 0.0, "default", "A", 0, 0, False)
        is None
    )


def test_gradient_custom_loss_falls_back_to_backcasting():
    def loss_fn(actual, fitted, B):  # noqa: N803
        return float(np.sum(np.abs(actual - fitted) ** 1.5))

    mb = ADAM(model="AAA", lags=[12], initial="backcasting", loss=loss_fn).fit(_YL)
    mg = ADAM(model="AAA", lags=[12], initial="gradient", loss=loss_fn).fit(_YL)
    assert abs(_loss_value(mg) - _loss_value(mb)) < 1e-8


# --- Occurrence models (om) -------------------------------------------------

_OT = np.array(
    [
        (1 if v > 0.5 else 0)
        for v in (0.5 + 0.4 * np.sin(np.arange(120) / 7) + 0.003 * np.arange(120))
    ],
    dtype=float,
)


def test_gradient_om_reaches_optimal_quality():
    from smooth import OM

    for occ in ["odds-ratio", "inverse-odds-ratio", "direct"]:
        mb = OM(model="MNN", occurrence=occ, initial="backcasting").fit(_OT)
        mo = OM(model="MNN", occurrence=occ, initial="optimal").fit(_OT)
        mg = OM(model="MNN", occurrence=occ, initial="gradient").fit(_OT)
        assert np.isfinite(float(mg.loglik))
        # Gradient profiles the same likelihood over the initials: at least
        # backcasting, and within 1% of the fully optimal fit.
        assert float(mg.loglik) >= float(mb.loglik) - 1e-4
        assert abs(float(mg.loglik) - float(mo.loglik)) < 0.01 * abs(float(mo.loglik))


def test_gradient_om_loss_code_mapping_matches_r():
    from smooth.adam_general.core.utils.gradient import adam_gradient_om_loss_code

    assert adam_gradient_om_loss_code("likelihood") == ("B", 0.0)
    assert adam_gradient_om_loss_code("MSE") == ("S", 0.0)
    assert adam_gradient_om_loss_code("MAE") == ("A", 0.0)
    assert adam_gradient_om_loss_code("HAM") == ("H", 0.0)
    assert adam_gradient_om_loss_code("LASSO") == ("S", 0.0)
    assert adam_gradient_om_loss_code("custom") is None
    assert adam_gradient_om_loss_code(lambda a, f, b: 0.0) is None


def test_gradient_om_fixed_persistence_loglik_is_finite():
    # Fixed persistence + gradient (or backcasting) leaves B empty: the
    # estimator must still evaluate the cost once (mirroring R's length(B)==0
    # guard) instead of returning nlopt's uninitialised +inf. Regression for
    # the -inf logLik bug.
    from smooth import OM

    for ini in ["backcasting", "gradient"]:
        m = OM(model="MNN", occurrence="odds-ratio", initial=ini, persistence=0.1).fit(
            _OT
        )
        assert len(m._adam_estimated["B"]) == 0
        ll = float(m.loglik)
        assert np.isfinite(ll)
        # logLik must equal the Bernoulli likelihood of the fitted probabilities
        f = np.asarray(m.fitted).ravel()
        on, off = f[_OT == 1], f[_OT == 0]
        expected = float(np.sum(np.log(on)) + np.sum(np.log(1 - off)))
        assert abs(ll - expected) < 1e-6


def test_om_loglik_is_bernoulli_not_cost_for_non_likelihood_loss():
    # For a fit-only loss (MSE), logLik must be the Bernoulli likelihood of the
    # fitted probabilities, NOT -CF (the MSE). lossValue keeps the MSE. No
    # flooring of the probabilities.
    from smooth import OM

    m = OM(model="MNN", occurrence="odds-ratio", initial="optimal", loss="MSE").fit(_OT)
    f = np.asarray(m.fitted).ravel()
    with np.errstate(invalid="ignore", divide="ignore"):
        bernoulli = float(np.sum(_OT * np.log(f) + (1 - _OT) * np.log(1 - f)))
    assert abs(float(m.loglik) - bernoulli) < 1e-9
    # lossValue is the MSE, a small number, not the Bernoulli magnitude
    assert m._adam_estimated["CF_value"] < 1.0


def test_om_lossvalue_is_the_fitting_loss_even_when_infeasible():
    # lossValue must be the actual fitting loss on the fitted probabilities,
    # never the optimiser's infeasibility penalty (1e300) — even when the loss
    # optimum leaves [0, 1]. Constructed data where odds-ratio MAE goes
    # infeasible (p < 0), so the guard would otherwise fire.
    from smooth import OM

    ot = np.tile([0.0, 0.0, 1.0, 0.0, 1.0, 0.0], 20)
    m = OM(
        model="MNN",
        occurrence="odds-ratio",
        initial="gradient",
        persistence=0.1,
        loss="MAE",
    ).fit(ot)
    f = np.asarray(m.fitted).ravel()
    assert abs(float(m.loss_value) - float(np.mean(np.abs(ot - f)))) < 1e-9
    assert float(m.loss_value) < 1.0  # the MAE, not 1e300
    # If the fit is infeasible (some p outside [0, 1]) the Bernoulli logLik is
    # NaN/-inf — surfaced, not clipped; if feasible it is finite. Either way it
    # is decoupled from lossValue.
    if f.min() < 0 or f.max() > 1:
        assert not np.isfinite(float(m.loglik))
