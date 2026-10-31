"""Unit tests for the SMA (Simple Moving Average) class."""

import numpy as np
import pytest

from smooth import SMA


@pytest.fixture
def y60():
    rng = np.random.default_rng(seed=7)
    return rng.normal(loc=100, scale=5, size=60)


class TestSMAInit:
    def test_import(self):
        from smooth import SMA

        assert SMA is not None

    def test_basic_init(self):
        assert SMA(order=3) is not None

    def test_auto_init(self):
        assert SMA() is not None

    def test_blocked_model_raises(self):
        with pytest.raises(ValueError, match="does not support"):
            SMA(model="ANN")

    def test_blocked_arma_raises(self):
        with pytest.raises(ValueError, match="does not support"):
            SMA(arma={"ar": [0.5]})

    def test_blocked_ar_order_raises(self):
        with pytest.raises(ValueError, match="does not support"):
            SMA(ar_order=3)

    def test_blocked_distribution_raises(self):
        with pytest.raises(ValueError, match="does not support"):
            SMA(distribution="dgamma")


class TestSMAFit:
    def test_fit_returns_self(self, y60):
        m = SMA(order=3)
        assert m.fit(y60) is m

    def test_model_name_fixed_order(self, y60):
        m = SMA(order=4).fit(y60)
        assert m.model == "SMA(4)"

    def test_model_name_auto(self, y60):
        m = SMA().fit(y60)
        assert m.model.startswith("SMA(")

    def test_fitted_shape_no_holdout(self, y60):
        m = SMA(order=3).fit(y60)
        assert len(m.fitted) == len(y60)

    def test_fitted_not_nan(self, y60):
        m = SMA(order=3).fit(y60)
        assert not np.any(np.isnan(m.fitted))

    def test_ics_present_when_auto(self, y60):
        m = SMA().fit(y60)
        assert hasattr(m, "ICs")
        assert isinstance(m.ICs, dict)
        assert len(m.ICs) > 0

    def test_ics_of_the_model_when_fixed(self, y60):
        # A fixed order gives the criterion of the model, as R's sma()$ICs
        m = SMA(order=3).fit(y60)
        assert m.ICs == pytest.approx(m.aicc)

    def test_fast_false_auto(self, y60):
        m = SMA(fast=False).fit(y60)
        assert m.model.startswith("SMA(")
        assert hasattr(m, "ICs")

    def test_order_too_large_raises(self, y60):
        with pytest.raises((ValueError, RuntimeError, Exception)):
            SMA(order=1000).fit(y60)

    def test_ics_keys_are_integers(self, y60):
        m = SMA().fit(y60)
        assert all(isinstance(k, int) for k in m.ICs.keys())

    def test_ics_selected_order_is_best(self, y60):
        m = SMA(fast=False).fit(y60)
        selected = int(m.model.split("(")[1].rstrip(")"))
        best_key = min(m.ICs, key=m.ICs.get)
        assert selected == best_key

    def test_ic_choices(self, y60):
        for ic in ("AIC", "AICc", "BIC", "BICc"):
            m = SMA(ic=ic).fit(y60)
            assert m.model.startswith("SMA(")


class TestSMAPredict:
    def test_predict_shape(self, y60):
        m = SMA(order=3).fit(y60)
        fc = m.predict(h=10)
        assert len(fc.mean) == 10

    def test_predict_not_nan(self, y60):
        m = SMA(order=3).fit(y60)
        fc = m.predict(h=10)
        assert not fc.mean.isna().any()

    def test_predict_before_fit_raises(self):
        m = SMA(order=3)
        with pytest.raises((AttributeError, ValueError, RuntimeError, KeyError)):
            m.predict(h=5)

    def test_predict_finite(self, y60):
        """All SMA forecasts should be finite (no inf/nan)."""
        m = SMA(order=3).fit(y60)
        fc = m.predict(h=5)
        assert np.all(np.isfinite(fc.mean.values))


class TestSMAHoldout:
    """Holdout path (regression guard for the empty-``lags`` fix)."""

    def test_fit_with_holdout(self, y60):
        m = SMA(order=4, h=12, holdout=True).fit(y60)
        assert m.model == "SMA(4)"

    def test_auto_with_holdout(self, y60):
        m = SMA(h=12, holdout=True).fit(y60)
        assert m.model.startswith("SMA(")

    def test_predict_with_holdout(self, y60):
        m = SMA(order=4, h=12, holdout=True).fit(y60)
        fc = m.predict(h=12)
        assert len(fc.mean) == 12
        assert np.all(np.isfinite(fc.mean.values))

    def test_prediction_interval_with_holdout(self, y60):
        m = SMA(order=4, h=12, holdout=True).fit(y60)
        fc = m.predict(h=12, interval="prediction")
        assert len(fc.mean) == 12


class TestSMAInferFrequency:
    """The frequency of the data, R's frequency(y), evaluated next to the ternary
    search of the order."""

    def test_no_frequency_without_a_dated_index(self):
        import pandas as pd

        from smooth.adam_general.core.sma import _infer_frequency

        assert _infer_frequency(np.arange(24.0)) is None
        assert _infer_frequency(pd.Series(np.arange(24.0))) is None
        irregular = pd.DatetimeIndex(["2020-01-01", "2020-01-03", "2020-01-10"])
        assert irregular.freq is None
        assert _infer_frequency(pd.Series([1.0, 2.0, 3.0], index=irregular)) is None

    @pytest.mark.parametrize(
        "freq, expected", [("MS", 12), ("QS", 4), ("W", 52), ("D", 7), ("h", 24)]
    )
    def test_the_frequency_of_a_dated_index(self, freq, expected):
        import pandas as pd

        from smooth.adam_general.core.sma import _infer_frequency

        index = pd.date_range("2020-01-01", periods=30, freq=freq)
        assert _infer_frequency(pd.Series(np.arange(30.0), index=index)) == expected

    def test_the_order_of_the_frequency_is_evaluated_as_r(self):
        import pandas as pd

        from tests.test_outliers import AIRPASSENGERS

        y = pd.Series(
            AIRPASSENGERS, index=pd.date_range("1949-01-01", periods=144, freq="MS")
        )
        model = SMA().fit(y)
        # R: which(!is.na(sma(AirPassengers)$ICs)) and sma(AirPassengers)$ICs[12]
        assert sorted(model.ICs) == [1, 2, 3, 5, 9, 12, 18, 36, 71, 141]
        assert model.ICs[12] == pytest.approx(1550.957584, rel=1e-8)
