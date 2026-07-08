"""Tests for ADAM.predict() method options: intervals, levels, cumulative, scenarios."""

import warnings

import numpy as np
import pytest

from smooth.adam_general.core.adam import ADAM
from smooth.adam_general.core.forecaster.result import ForecastResult

# ---------------------------------------------------------------------------
# Module-level data and pre-fitted models (created once, shared across tests)
# ---------------------------------------------------------------------------


def _make_simple():
    np.random.seed(42)
    return 10 + 0.5 * np.arange(50) + np.random.randn(50) * 2


def _make_seasonal():
    np.random.seed(42)
    n = 120
    t = np.arange(n)
    return 100 + 0.5 * t + 10 * np.sin(2 * np.pi * t / 12) + np.random.randn(n) * 3


def _make_multiplicative():
    np.random.seed(42)
    n = 120
    t = np.arange(n)
    trend = 100 + 0.5 * t
    seasonal = 1 + 0.2 * np.sin(2 * np.pi * t / 12)
    noise = 1 + np.random.randn(n) * 0.05
    return trend * seasonal * noise


# Pre-fit models at module load time
_simple_data = _make_simple()
_seasonal_data = _make_seasonal()
_mult_data = _make_multiplicative()

_simple_model = ADAM(model="ANN", lags=[1])
_simple_model.fit(_simple_data)

_additive_model = ADAM(model="ANA", lags=[1, 12])
_additive_model.fit(_seasonal_data)

_mult_model = ADAM(model="MAM", lags=[1, 12])
_mult_model.fit(_mult_data)

# Real AirPassengers (Box & Jenkins): a well-conditioned series we use
# for tests that compare prediction-vs-confidence interval widths. The
# synthetic ``_mult_data`` above sometimes yields an indefinite vcov,
# which forces a PSD repair and distorts parameter draws.
_AIRPASSENGERS = np.array(
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


# ---------------------------------------------------------------------------
# 1. Interval types
# ---------------------------------------------------------------------------


class TestPredictIntervalTypes:
    """Tests for each interval type with additive and multiplicative models."""

    def test_interval_none(self):
        r = _simple_model.predict(h=5, interval="none")
        assert isinstance(r, ForecastResult)
        assert len(r.mean) == 5
        assert r.lower is None
        assert r.upper is None

    def test_interval_approximate(self):
        r = _additive_model.predict(h=5, interval="approximate")
        assert len(r.mean) == 5
        assert r.lower is not None and r.lower.shape == (5, 1)
        assert r.upper is not None and r.upper.shape == (5, 1)
        assert (r.lower.iloc[:, 0].values <= r.mean.values).all()
        assert (r.mean.values <= r.upper.iloc[:, 0].values).all()

    def test_interval_simulated(self):
        r = _additive_model.predict(h=5, interval="simulated", nsim=1000)
        assert len(r.mean) == 5
        assert r.lower is not None and r.upper is not None
        assert (r.lower.iloc[:, 0].values <= r.mean.values).all()
        assert (r.mean.values <= r.upper.iloc[:, 0].values).all()

    def test_interval_prediction_additive_resolves_approximate(self):
        """For additive models, interval='prediction' auto-resolves to 'approximate'."""
        r_pred = _additive_model.predict(h=5, interval="prediction")
        r_approx = _additive_model.predict(h=5, interval="approximate")
        np.testing.assert_array_almost_equal(
            r_pred.mean.values, r_approx.mean.values, decimal=10
        )
        np.testing.assert_array_almost_equal(
            r_pred.lower.values, r_approx.lower.values, decimal=10
        )
        np.testing.assert_array_almost_equal(
            r_pred.upper.values, r_approx.upper.values, decimal=10
        )

    def test_interval_prediction_multiplicative_has_intervals(self):
        """For multiplicative models, interval='prediction' resolves to 'simulated'."""
        r = _mult_model.predict(h=5, interval="prediction", nsim=1000)
        assert len(r.mean) == 5
        assert r.lower is not None and r.upper is not None
        assert (r.lower.iloc[:, 0].values <= r.mean.values).all()
        assert (r.mean.values <= r.upper.iloc[:, 0].values).all()

    def test_interval_simulated_multiplicative(self):
        r = _mult_model.predict(h=5, interval="simulated", nsim=1000)
        assert len(r.mean) == 5
        assert r.lower is not None and r.upper is not None

    def test_interval_complete_returns_forecast_result(self):
        """``interval='complete'`` dispatches through ``reforecast`` and
        returns the stock :class:`ForecastResult` shape."""
        r = _mult_model.predict(h=6, interval="complete", nsim=40)
        assert isinstance(r, ForecastResult)
        assert len(r.mean) == 6
        assert r.lower is not None and r.lower.shape == (6, 1)
        assert r.upper is not None and r.upper.shape == (6, 1)
        # ``complete`` is an alias for ``prediction`` on reforecast — the
        # returned ``ForecastResult.interval`` reflects the underlying
        # reforecast mode, matching R's behaviour.
        assert r.interval == "prediction"
        assert (r.lower.iloc[:, 0].values <= r.mean.values + 1e-9).all()
        assert (r.mean.values <= r.upper.iloc[:, 0].values + 1e-9).all()

    def test_interval_confidence_returns_forecast_result(self):
        r = _mult_model.predict(h=6, interval="confidence", nsim=40)
        assert isinstance(r, ForecastResult)
        assert len(r.mean) == 6
        assert r.interval == "confidence"
        assert r.lower is not None and r.upper is not None

    def test_interval_confidence_narrower_than_complete(self):
        """For the same fit + nsim, confidence intervals are strictly
        narrower than prediction (= ``complete``) intervals on average.

        Uses AirPassengers + MAM (same well-conditioned setup as
        ``test_reforecast_confidence_narrower_than_prediction``) and a
        fixed ``seed`` so the inequality is deterministic and far
        outside MC noise across platforms. The synthetic multiplicative
        series shared by other tests in this class has an indefinite
        vcov for some platform RNG seeds — the resulting PSD repair
        distorts parameter draws enough that the confidence-vs-
        prediction inequality is no longer a clean unit test of the
        reforecast reduction.
        """
        m = ADAM(model="MAM", lags=[12], initial="backcasting").fit(_AIRPASSENGERS)
        r_pred = m.predict(h=12, interval="complete", nsim=100, seed=42)
        r_conf = m.predict(h=12, interval="confidence", nsim=100, seed=42)
        pred_width = (
            (r_pred.upper.iloc[:, 0] - r_pred.lower.iloc[:, 0]).to_numpy().mean()
        )
        conf_width = (
            (r_conf.upper.iloc[:, 0] - r_conf.lower.iloc[:, 0]).to_numpy().mean()
        )
        assert conf_width < pred_width, (
            f"confidence mean width {conf_width:.2f} not < prediction "
            f"mean width {pred_width:.2f}"
        )

    def test_interval_complete_default_nsim_is_100(self):
        """When the caller leaves ``nsim`` at the predict default, the
        reforecast path uses R's ``forecast.adam`` default of 100. We
        can't observe ``nsim`` directly from the ``ForecastResult``, but
        a successful call confirms the dispatch routes through
        reforecast (which has nsim=100 default) rather than failing on
        a 10000-sample reapply that would be very slow."""
        r = _additive_model.predict(h=4, interval="complete")
        assert isinstance(r, ForecastResult)
        assert len(r.mean) == 4

    def test_interval_complete_multi_level(self):
        r = _mult_model.predict(h=6, interval="complete", level=[0.8, 0.95], nsim=40)
        assert r.lower.shape == (6, 2)
        assert r.upper.shape == (6, 2)
        # Tighter level must give narrower interval.
        w80 = (r.upper.iloc[:, 0] - r.lower.iloc[:, 0]).to_numpy()
        w95 = (r.upper.iloc[:, 1] - r.lower.iloc[:, 1]).to_numpy()
        assert (w80 <= w95 + 1e-6).all()

    def test_interval_complete_arima_runs(self):
        """Phase 5: ARIMA models go through ``predict(interval='complete')``
        without raising. Uses a seasonal ARIMA spec to stay out of the
        ``L=1`` carma heap-corruption regime (see ``test_reapply`` for
        the same workaround)."""
        m = ADAM(
            model="NNN",
            orders={"ar": [1, 1], "i": [0, 0], "ma": [1, 1]},
            lags=[1, 12],
            initial="backcasting",
        ).fit(_seasonal_data)
        r = m.predict(h=4, interval="complete", nsim=10)
        assert isinstance(r, ForecastResult)
        assert r.lower is not None and r.lower.shape == (4, 1)
        assert r.upper is not None and r.upper.shape == (4, 1)


# ---------------------------------------------------------------------------
# 2. Confidence levels
# ---------------------------------------------------------------------------


class TestPredictLevels:
    """Tests for different confidence level values."""

    def test_level_80_column_names(self):
        r = _simple_model.predict(h=5, interval="approximate", level=0.80)
        assert 0.1 in r.lower.columns
        assert 0.9 in r.upper.columns

    def test_level_99_column_names(self):
        r = _simple_model.predict(h=5, interval="approximate", level=0.99)
        assert 0.005 in r.lower.columns
        assert 0.995 in r.upper.columns

    def test_wider_level_has_wider_interval(self):
        r80 = _simple_model.predict(h=5, interval="approximate", level=0.80)
        r99 = _simple_model.predict(h=5, interval="approximate", level=0.99)
        width80 = (r80.upper.iloc[:, 0] - r80.lower.iloc[:, 0]).values
        width99 = (r99.upper.iloc[:, 0] - r99.lower.iloc[:, 0]).values
        assert (width99 > width80).all()

    def test_level_as_percentage_auto_converts(self):
        """level=95 should be auto-converted to 0.95."""
        r = _simple_model.predict(h=5, interval="approximate", level=95)
        assert 0.025 in r.lower.columns
        assert 0.975 in r.upper.columns

    def test_default_level_95(self):
        r = _simple_model.predict(h=5, interval="approximate")
        assert 0.025 in r.lower.columns
        assert 0.975 in r.upper.columns


# ---------------------------------------------------------------------------
# 3. Side parameter
# ---------------------------------------------------------------------------


class TestPredictSides:
    """Tests for the side parameter (both, upper, lower)."""

    def test_side_both(self):
        r = _simple_model.predict(h=5, interval="approximate", side="both")
        assert r.lower is not None and r.lower.shape[1] == 1
        assert r.upper is not None and r.upper.shape[1] == 1

    def test_side_upper(self):
        """side='upper' returns only upper interval."""
        r = _simple_model.predict(h=5, interval="approximate", level=0.9, side="upper")
        assert r.lower is None
        assert r.upper is not None and r.upper.shape[1] == 1
        assert 0.9 in r.upper.columns
        assert (r.upper.iloc[:, 0].values >= r.mean.values).all()

    def test_side_lower(self):
        """side='lower' returns only lower interval."""
        r = _simple_model.predict(h=5, interval="approximate", level=0.9, side="lower")
        assert r.lower is not None and r.lower.shape[1] == 1
        assert r.upper is None
        assert 0.1 in r.lower.columns
        assert (r.lower.iloc[:, 0].values <= r.mean.values).all()


# ---------------------------------------------------------------------------
# 4. Cumulative forecasts
# ---------------------------------------------------------------------------


class TestPredictCumulative:
    """Tests for cumulative forecast output."""

    def test_cumulative_mean_equals_sum(self):
        """Cumulative mean should equal sum of individual step means."""
        r_cum = _simple_model.predict(h=5, interval="none", cumulative=True)
        r_ind = _simple_model.predict(h=5, interval="none", cumulative=False)
        np.testing.assert_almost_equal(
            r_cum.mean.values[0], r_ind.mean.sum(), decimal=8
        )

    def test_cumulative_returns_single_row(self):
        """Cumulative mode returns exactly one row."""
        r = _simple_model.predict(h=5, interval="none", cumulative=True)
        assert len(r.mean) == 1

    def test_cumulative_with_simulated_intervals(self):
        """Cumulative with simulated intervals produces single-row output."""
        r = _simple_model.predict(h=5, interval="simulated", cumulative=True, nsim=1000)
        assert len(r.mean) == 1
        assert r.lower is not None and r.upper is not None
        assert r.lower.iloc[0, 0] <= r.mean.values[0]
        assert r.mean.values[0] <= r.upper.iloc[0, 0]

    def test_cumulative_with_approximate_intervals(self):
        """Cumulative with approximate intervals produces single-row output."""
        r = _additive_model.predict(h=12, interval="approximate", cumulative=True)
        assert len(r.mean) == 1
        assert r.lower is not None and r.upper is not None
        assert r.lower.iloc[0, 0] <= r.mean.values[0]
        assert r.mean.values[0] <= r.upper.iloc[0, 0]

    def test_cumulative_with_approximate_side_upper(self):
        """Cumulative + side='upper' + approximate intervals works."""
        r = _additive_model.predict(
            h=18, interval="approximate", side="upper", cumulative=True
        )
        assert len(r.mean) == 1
        assert r.lower is None
        assert r.upper is not None
        assert r.mean.values[0] <= r.upper.iloc[0, 0]


# ---------------------------------------------------------------------------
# 5. Scenarios
# ---------------------------------------------------------------------------


class TestPredictScenarios:
    """Tests for scenario output storage."""

    def test_scenarios_stored(self):
        _simple_model.predict(h=5, interval="simulated", nsim=1000, scenarios=True)
        sc = _simple_model._general.get("_scenarios_matrix")
        assert sc is not None

    def test_scenarios_shape(self):
        _simple_model.predict(h=5, interval="simulated", nsim=1000, scenarios=True)
        sc = _simple_model._general["_scenarios_matrix"]
        assert sc.shape == (5, 1000)

    def test_scenarios_requires_simulated_warns(self):
        """scenarios=True with interval='approximate' issues a warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _simple_model.predict(h=5, interval="approximate", scenarios=True)
            scenario_warnings = [x for x in w if "scenarios=True" in str(x.message)]
            assert len(scenario_warnings) >= 1


# ---------------------------------------------------------------------------
# 6. Multiplicative model specifics
# ---------------------------------------------------------------------------


class TestPredictMultiplicative:
    """Tests specific to multiplicative models and simulation-based means."""

    def test_additive_same_across_intervals(self):
        """Additive model point forecasts are identical for none/approximate."""
        r_none = _additive_model.predict(h=5, interval="none")
        r_approx = _additive_model.predict(h=5, interval="approximate")
        np.testing.assert_array_almost_equal(
            r_none.mean.values, r_approx.mean.values, decimal=10
        )

    def test_multiplicative_approximate_produces_forecasts(self):
        """MAM with interval='approximate' produces valid finite forecasts."""
        r = _mult_model.predict(h=5, interval="approximate")
        assert np.all(np.isfinite(r.mean.values))
        assert (r.mean.values > 0).all()

    def test_multiplicative_simulated_produces_forecasts(self):
        """MAM with interval='simulated' produces valid finite forecasts."""
        r = _mult_model.predict(h=5, interval="simulated", nsim=1000)
        assert np.all(np.isfinite(r.mean.values))
        assert (r.mean.values > 0).all()

    def test_multiplicative_cumulative_simulated_not_inflated(self):
        """Cumulative simulated mean for MAM should equal sum of step means."""
        h = 12
        r_steps = _mult_model.predict(h=h, interval="simulated", nsim=5000)
        r_cum = _mult_model.predict(
            h=h, interval="simulated", cumulative=True, nsim=5000
        )
        step_sum = r_steps.mean.sum()
        cum_mean = r_cum.mean.values[0]
        # Both are simulation-based so allow 15% tolerance
        np.testing.assert_allclose(cum_mean, step_sum, rtol=0.15)


# ---------------------------------------------------------------------------
# 7. Combined model prediction
# ---------------------------------------------------------------------------


class TestPredictCombined:
    """Tests for predict on combined (CXC) models."""

    @pytest.fixture(scope="class")
    def combined_model(self):
        m = ADAM(model="CXC", lags=[1, 12])
        m.fit(_seasonal_data)
        return m

    def test_combined_predict_with_intervals(self, combined_model):
        r = combined_model.predict(h=10, interval="prediction")
        assert isinstance(r, ForecastResult)
        assert len(r.mean) == 10
        assert r.lower is not None and r.upper is not None
        assert (r.lower.iloc[:, 0].values <= r.mean.values).all()
        assert (r.mean.values <= r.upper.iloc[:, 0].values).all()

    def test_combined_predict_none(self, combined_model):
        r = combined_model.predict(h=10, interval="none")
        assert len(r.mean) == 10
        assert r.lower is None and r.upper is None
        assert np.all(np.isfinite(r.mean.values))


# ---------------------------------------------------------------------------
# 8. Multi-level prediction intervals
# ---------------------------------------------------------------------------


class TestPredictMultiLevel:
    """Tests for multi-level prediction intervals."""

    def test_multi_level_column_count(self):
        """Two levels produce 2 lower + 2 upper columns."""
        r = _simple_model.predict(h=5, interval="approximate", level=[0.9, 0.95])
        assert len(r.mean) == 5
        assert r.lower.shape == (5, 2)
        assert r.upper.shape == (5, 2)

    def test_multi_level_column_names(self):
        r = _simple_model.predict(h=5, interval="approximate", level=[0.9, 0.95])
        assert 0.05 in r.lower.columns
        assert 0.025 in r.lower.columns
        assert 0.95 in r.upper.columns
        assert 0.975 in r.upper.columns

    def test_multi_level_nesting(self):
        """Wider level contains narrower level."""
        r = _simple_model.predict(h=5, interval="approximate", level=[0.8, 0.99])
        lower80 = r.lower[0.1].values
        lower99 = r.lower[0.005].values
        upper80 = r.upper[0.9].values
        upper99 = r.upper[0.995].values
        assert (lower99 <= lower80).all()
        assert (upper99 >= upper80).all()

    def test_multi_level_simulated(self):
        """Multi-level works with simulated intervals."""
        r = _additive_model.predict(
            h=5, interval="simulated", level=[0.8, 0.95], nsim=1000
        )
        assert len(r.mean) == 5
        assert r.lower.shape == (5, 2)
        assert r.upper.shape == (5, 2)

    def test_multi_level_side_upper(self):
        """Multi-level with side='upper' gives only upper columns."""
        r = _simple_model.predict(
            h=5, interval="approximate", level=[0.9, 0.95], side="upper"
        )
        assert r.lower is None
        assert r.upper is not None and r.upper.shape == (5, 2)

    def test_multi_level_side_lower(self):
        """Multi-level with side='lower' gives only lower columns."""
        r = _simple_model.predict(
            h=5, interval="approximate", level=[0.9, 0.95], side="lower"
        )
        assert r.lower is not None and r.lower.shape == (5, 2)
        assert r.upper is None

    def test_three_levels(self):
        """Three levels produce 3 lower + 3 upper columns."""
        r = _simple_model.predict(h=5, interval="approximate", level=[0.8, 0.9, 0.99])
        assert r.lower.shape == (5, 3)
        assert r.upper.shape == (5, 3)

    def test_predict_intervals_method(self):
        """predict_intervals() returns ForecastResult with multi-level intervals."""
        r = _simple_model.predict_intervals(h=5, levels=[0.8, 0.95])
        assert isinstance(r, ForecastResult)
        assert r.lower is not None and r.lower.shape[1] == 2
        assert r.upper is not None and r.upper.shape[1] == 2

    def test_to_dataframe(self):
        """to_dataframe() returns a flat DataFrame with prefixed column names."""
        r = _simple_model.predict(h=5, interval="approximate", level=[0.9, 0.95])
        df = r.to_dataframe()
        assert "mean" in df.columns
        assert "lower_0.05" in df.columns
        assert "upper_0.975" in df.columns
        assert df.shape == (5, 5)
