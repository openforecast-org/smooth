"""Parameter validation for :class:`smooth.CES`.

Kept out of ``test_ces.py`` because that module is marked ``r_parity`` as a
whole; these check the Python API's own guards and need no R installation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smooth import CES

_DATA_DIR = Path(__file__).parent / "data"


class TestCESProvidedB:
    """The provided ``b`` must match the seasonality it belongs to."""

    y = pd.read_csv(_DATA_DIR / "ces_airpassengers.csv")["y"].values

    def test_full_rejects_a_real_b(self):
        with pytest.raises(ValueError, match="complex second smoothing parameter"):
            CES(seasonality="full", lags=[12], b=0.3).fit(self.y)

    def test_partial_rejects_a_complex_b(self):
        with pytest.raises(ValueError, match="real second smoothing parameter"):
            CES(seasonality="partial", lags=[12], b=complex(0.3, 0.1)).fit(self.y)

    @pytest.mark.parametrize("seasonality", ["none", "simple"])
    def test_b_is_dropped_where_there_is_none(self, seasonality):
        with pytest.warns(UserWarning, match="no second smoothing parameter"):
            m = CES(seasonality=seasonality, lags=[12], b=0.3)
        m.fit(self.y)
        assert m.b_ is None
        free = CES(seasonality=seasonality, lags=[12])
        free.fit(self.y)
        assert m.loglik == free.loglik

    def test_valid_b_is_honoured(self):
        free = CES(seasonality="full", lags=[12])
        free.fit(self.y)
        held = CES(seasonality="full", lags=[12], b=free.b_)
        held.fit(self.y)
        assert held.b_ == free.b_

        partial = CES(seasonality="partial", lags=[12], b=0.3)
        partial.fit(self.y)
        assert partial.b_ == 0.3


def test_ces_predict_keeps_the_seasonal_phase_past_one_cycle():
    """predict() beyond one cycle reads the seasonal cells in phase, as the
    forecast made at fit time does, when the sample ends mid-cycle."""
    t = np.arange(138)
    y = (
        100
        + t
        + 10 * np.sin(2 * np.pi * t / 12)
        + np.random.default_rng(3).normal(0, 2, t.size)
    )
    fitted_with_h = CES(seasonality="full", lags=[12], h=30).fit(y)
    fitted_alone = CES(seasonality="full", lags=[12]).fit(y)
    # The forecast of the C++ core from the end of the sample, with the lookup
    # table of the fit, which spans the horizon
    m = fitted_with_h
    start = m._head_geometry + m._obs_in_sample
    forecast = m._adam_cpp.forecast(
        matrixWt=np.asfortranarray(np.tile(m._mat_wt[-1:], (30, 1))),
        matrixF=np.asfortranarray(m._mat_f),
        indexLookupTable=np.asfortranarray(m._index_lookup_table[:, start : start + 30]),
        profilesRecent=np.asfortranarray(m._profiles_recent_table.copy()),
        horizon=30,
    ).forecast
    np.testing.assert_allclose(
        np.asarray(fitted_alone.predict(h=30).mean), np.ravel(forecast), rtol=1e-12
    )


def test_sum_r_adds_in_order_as_r():
    """R's sum() accumulates sequentially in long double: each 1 is lost
    against 2^64, so the total is 0 (R: sum(c(2^64, rep(1,7), -2^64, rep(0,7))))
    where NumPy's pairwise long double sum gives 7."""
    from smooth.adam_general.core.utils.utils import _sum_r

    values = np.array([2.0**64] + [1.0] * 7 + [-(2.0**64)] + [0.0] * 7)
    assert _sum_r(values) == 0.0
    assert np.array_equal(_sum_r(np.vstack([values, values]), axis=1), [0.0, 0.0])


def test_ces_takes_the_missing_values_for_gaps():
    """The missing values are skipped by the fit and are not in the loss: the
    likelihood is over the observed values, as its per-observation terms."""
    y = pd.read_csv(_DATA_DIR / "ces_airpassengers.csv")["y"].values.astype(float)
    y[[9, 49, 50, 89]] = np.nan
    with pytest.warns(UserWarning, match="NAs"):
        model = CES(seasonality="full", lags=[12]).fit(y)
    assert np.isfinite(model.loglik)
    assert np.sum(model.point_lik()) == pytest.approx(model.loglik, rel=1e-12)
    np.testing.assert_array_equal(np.isnan(model.residuals), np.isnan(y))
    assert np.all(np.isfinite(model.predict(h=12, interval="prediction").upper))
