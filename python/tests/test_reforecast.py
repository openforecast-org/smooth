"""Unit tests for ``ADAM.reforecast()``.

Phase 2 of the R ``reforecast.adam`` port (R/reapply.R:941). Verifies
the public surface: shape and finiteness, lower <= mean <= upper, the
``confidence`` vs ``prediction`` width relationship, the ``h<=0``
short-circuit on the refitted matrix, ``cumulative`` mode, and the
multi-level interval shape.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smooth import ADAM
from smooth.adam_general.core.utils.reforecast import ReforecastResult

AIRPASSENGERS = np.array(
    [
        112,
        118,
        132,
        129,
        121,
        135,
        148,
        148,
        136,
        119,
        104,
        118,
        115,
        126,
        141,
        135,
        125,
        149,
        170,
        170,
        158,
        133,
        114,
        140,
        145,
        150,
        178,
        163,
        172,
        178,
        199,
        199,
        184,
        162,
        146,
        166,
        171,
        180,
        193,
        181,
        183,
        218,
        230,
        242,
        209,
        191,
        172,
        194,
        196,
        196,
        236,
        235,
        229,
        243,
        264,
        272,
        237,
        211,
        180,
        201,
        204,
        188,
        235,
        227,
        234,
        264,
        302,
        293,
        259,
        229,
        203,
        229,
        242,
        233,
        267,
        269,
        270,
        315,
        364,
        347,
        312,
        274,
        237,
        278,
        284,
        277,
        317,
        313,
        318,
        374,
        413,
        405,
        355,
        306,
        271,
        306,
        315,
        301,
        356,
        348,
        355,
        422,
        465,
        467,
        404,
        347,
        305,
        336,
        340,
        318,
        362,
        348,
        363,
        435,
        491,
        505,
        404,
        359,
        310,
        337,
        360,
        342,
        406,
        396,
        420,
        472,
        548,
        559,
        463,
        407,
        362,
        405,
        417,
        391,
        419,
        461,
        472,
        535,
        622,
        606,
        508,
        461,
        390,
        432,
    ],
    dtype=float,
)


def _fit_mam(initial="backcasting"):
    return ADAM(model="MAM", lags=[12], initial=initial).fit(AIRPASSENGERS)


def test_reforecast_prediction_shapes_and_bounds():
    """``interval='prediction'`` returns finite mean/lower/upper with the
    documented shapes; lower <= mean <= upper everywhere."""
    m = _fit_mam()
    h, nsim = 12, 30
    fc = m.reforecast(h=h, nsim=nsim, interval="prediction", seed=42)

    assert isinstance(fc, ReforecastResult)
    assert fc.h == h
    assert fc.interval == "prediction"
    assert fc.mean.shape == (h,)
    assert fc.lower.shape == (h, 1)
    assert fc.upper.shape == (h, 1)
    assert fc.paths.shape == (h, nsim, nsim)
    assert np.all(np.isfinite(fc.mean.to_numpy()))
    lower = fc.lower.iloc[:, 0].to_numpy()
    upper = fc.upper.iloc[:, 0].to_numpy()
    mean = fc.mean.to_numpy()
    assert np.all(lower <= mean + 1e-9), "lower must not exceed mean"
    assert np.all(mean <= upper + 1e-9), "mean must not exceed upper"


def test_reforecast_confidence_narrower_than_prediction():
    """For the same fit, ``confidence`` intervals are strictly narrower
    than ``prediction`` intervals on average."""
    m = _fit_mam()
    fc_pred = m.reforecast(h=12, nsim=30, interval="prediction", seed=42)
    fc_conf = m.reforecast(h=12, nsim=30, interval="confidence", seed=42)

    pred_widths = (fc_pred.upper.iloc[:, 0] - fc_pred.lower.iloc[:, 0]).to_numpy()
    conf_widths = (fc_conf.upper.iloc[:, 0] - fc_conf.lower.iloc[:, 0]).to_numpy()
    # Average confidence width must be less than average prediction
    # width — confidence marginalises over the error draws and so loses
    # the prediction-uncertainty component.
    assert conf_widths.mean() < pred_widths.mean(), (
        f"conf mean={conf_widths.mean():.2f} >= pred mean={pred_widths.mean():.2f}"
    )


def test_reforecast_h_zero_returns_fitted_period_ci():
    """``h<=0`` short-circuits and returns fitted-period results."""
    m = _fit_mam()
    fc = m.reforecast(h=0, nsim=20, interval="confidence", seed=42)
    assert fc.mean.shape == (m.nobs,)
    assert fc.lower is not None
    assert fc.lower.shape == (m.nobs, 1)
    assert fc.paths is None  # No forward sim was run


def test_reforecast_none_interval_skips_quantiles():
    """``interval='none'`` returns mean only — lower/upper are ``None``."""
    m = _fit_mam()
    fc = m.reforecast(h=12, nsim=20, interval="none", seed=42)
    assert fc.lower is None
    assert fc.upper is None
    assert fc.mean.shape == (12,)
    assert fc.paths.shape == (12, 20, 20)


def test_reforecast_cumulative_single_step():
    """``cumulative=True`` collapses the horizon to a single value."""
    m = _fit_mam()
    fc = m.reforecast(h=12, nsim=30, interval="prediction", cumulative=True, seed=42)
    assert fc.cumulative is True
    assert fc.mean.shape == (1,)
    assert fc.lower.shape == (1, 1)
    assert fc.upper.shape == (1, 1)
    lower = float(fc.lower.iloc[0, 0])
    upper = float(fc.upper.iloc[0, 0])
    assert lower <= float(fc.mean.iloc[0]) <= upper


def test_reforecast_multi_level_intervals():
    """A list of levels produces one column per level on each side."""
    m = _fit_mam()
    fc = m.reforecast(h=12, nsim=30, interval="prediction", level=[0.8, 0.95], seed=42)
    assert fc.lower.shape == (12, 2)
    assert fc.upper.shape == (12, 2)
    # 80% interval must be narrower than 95% at every step.
    width80 = (fc.upper.iloc[:, 0] - fc.lower.iloc[:, 0]).to_numpy()
    width95 = (fc.upper.iloc[:, 1] - fc.lower.iloc[:, 1]).to_numpy()
    assert np.all(width80 <= width95 + 1e-6)


def test_reforecast_to_forecast_result_projection():
    """The ``to_forecast_result`` helper produces a stock
    ``ForecastResult`` with the same data."""
    m = _fit_mam()
    fc = m.reforecast(h=12, nsim=20, interval="prediction", seed=42)
    proj = fc.to_forecast_result()
    pd.testing.assert_series_equal(proj.mean, fc.mean)
    pd.testing.assert_frame_equal(proj.lower, fc.lower)
    pd.testing.assert_frame_equal(proj.upper, fc.upper)


def test_reforecast_arima_runs_and_returns_finite():
    """Phase 5: ARIMA models now go through ``reforecast()``.

    Uses a seasonal ARIMA(1,0,1)x(1,0,1)[12] spec for the same reason
    as ``test_reapply_arima_runs_and_returns_finite`` — keeps
    ``L > 1`` so the carma allocator stays out of the heap-corruption
    regime documented in test_reapply.
    """
    m = ADAM(
        model="NNN",
        orders={"ar": [1, 1], "i": [0, 0], "ma": [1, 1]},
        lags=[1, 12],
        initial="backcasting",
    ).fit(AIRPASSENGERS)
    fc = m.reforecast(h=6, nsim=15, interval="prediction", seed=0)
    assert fc.mean.shape == (6,)
    assert fc.lower is not None and fc.lower.shape == (6, 1)
    assert fc.upper is not None and fc.upper.shape == (6, 1)
    assert np.all(np.isfinite(fc.mean.to_numpy()))


def test_reforecast_xreg_runs_with_newdata():
    """Phase 5: numeric xreg + future X goes through reforecast()."""
    n = len(AIRPASSENGERS)
    t = np.arange(n, dtype=float)
    X = np.column_stack([t / n, np.cos(2 * np.pi * t / 12), np.sin(2 * np.pi * t / 12)])
    m = ADAM(model="ANN", lags=[12]).fit(AIRPASSENGERS, X=X)
    h = 12
    t_future = np.arange(n, n + h, dtype=float)
    X_future = np.column_stack(
        [
            t_future / n,
            np.cos(2 * np.pi * t_future / 12),
            np.sin(2 * np.pi * t_future / 12),
        ]
    )
    fc = m.reforecast(h=h, X=X_future, nsim=15, interval="prediction", seed=0)
    assert fc.mean.shape == (h,)
    assert fc.lower is not None and fc.lower.shape == (h, 1)
    assert fc.upper is not None and fc.upper.shape == (h, 1)
    assert np.all(np.isfinite(fc.mean.to_numpy()))


def test_reforecast_xreg_warns_when_x_missing():
    """Without future X, the in-sample tail is reused with a warning."""
    n = len(AIRPASSENGERS)
    X = np.column_stack(
        [np.arange(n, dtype=float) / n, np.cos(2 * np.pi * np.arange(n) / 12)]
    )
    m = ADAM(model="ANN", lags=[12]).fit(AIRPASSENGERS, X=X)
    with pytest.warns(UserWarning, match="newdata.*fallback"):
        fc = m.reforecast(h=6, nsim=10, interval="prediction", seed=0)
    assert fc.mean.shape == (6,)
