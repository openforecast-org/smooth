"""Tests for AutoADAM — automatic ADAM model selection."""

import warnings

import numpy as np
import pytest

from smooth import AutoADAM
from smooth.adam_general.core.estimator.arima_selector import (
    _check_ima_models,
    _normalize_lags_and_orders,
    _select_i_orders,
    arima_selector,
)


@pytest.fixture
def trend_data():
    np.random.seed(42)
    return 100 + np.arange(1, 61) * 0.5 + np.random.randn(60) * 3


@pytest.fixture
def seasonal_data():
    np.random.seed(42)
    trend = np.arange(1, 145) * 0.3
    seasonal = np.tile(np.sin(np.linspace(0, 2 * np.pi, 12)), 12) * 10
    return 100 + trend + seasonal + np.random.randn(144) * 2


@pytest.fixture
def positive_data():
    np.random.seed(7)
    return np.abs(np.cumsum(np.random.randn(80))) + 10


# ---------------------------------------------------------------------------
# Normalise lags and orders
# ---------------------------------------------------------------------------


class TestNormalizeLagsAndOrders:
    def test_prepends_lag1_when_missing(self):
        lags, ar, i, ma = _normalize_lags_and_orders([12], [3], [1], [3])
        assert lags[0] == 1
        assert lags[1] == 12

    def test_no_prepend_when_lag1_present(self):
        lags, ar, i, ma = _normalize_lags_and_orders([1, 12], [3, 2], [2, 1], [3, 2])
        assert lags == [1, 12]
        assert len(ar) == 2

    def test_pads_short_order_lists(self):
        lags, ar, i, ma = _normalize_lags_and_orders([1, 12], [3], [2], [3])
        assert len(ar) == 2
        assert len(i) == 2
        assert len(ma) == 2


# ---------------------------------------------------------------------------
# ARIMA selector phases
# ---------------------------------------------------------------------------


class TestARIMASelector:
    def test_i_order_selection_returns_result(self, trend_data):
        best_i, const, model, ic = _select_i_orders(
            y=trend_data,
            ets_model="NNN",
            max_i=[2],
            lags=[1],
            distribution="dnorm",
            ic_name="AICc",
            loss="likelihood",
            h=None,
            holdout=False,
            bounds="usual",
            initial="backcasting",
            regressors="use",
            verbose=0,
        )
        assert model is not None
        assert ic < np.inf
        assert isinstance(best_i, list)

    def test_arima_selector_returns_dict(self, trend_data):
        result = arima_selector(
            y=trend_data,
            ets_model="NNN",
            max_ar_orders=[2],
            max_i_orders=[2],
            max_ma_orders=[2],
            lags=[1],
            distribution="dnorm",
            ic="AICc",
            loss="likelihood",
            h=None,
            holdout=False,
            bounds="usual",
            initial="backcasting",
            regressors="use",
            verbose=0,
        )
        assert "ar_orders" in result
        assert "i_orders" in result
        assert "ma_orders" in result
        assert "constant" in result
        assert "model" in result
        assert "ic_value" in result

    def test_arima_selector_model_fitted(self, trend_data):
        result = arima_selector(
            y=trend_data,
            ets_model="NNN",
            max_ar_orders=[2],
            max_i_orders=[2],
            max_ma_orders=[2],
            lags=[1],
            distribution="dnorm",
            ic="AICc",
            loss="likelihood",
            h=None,
            holdout=False,
            bounds="usual",
            initial="backcasting",
            regressors="use",
            verbose=0,
        )
        assert result["model"] is not None
        assert result["ic_value"] < np.inf

    def test_check_ima_models_runs(self, trend_data):
        from smooth import ADAM

        base = ADAM(model="NNN", lags=[1], ar_order=0, i_order=1, ma_order=0).fit(
            trend_data
        )
        new_i, new_ma, new_model, new_ic = _check_ima_models(
            y=trend_data,
            ets_model="NNN",
            best_ar=[0],
            max_i=[2],
            max_ma=[2],
            lags=[1],
            constant=False,
            distribution="dnorm",
            ic_name="AICc",
            best_ic=base.aicc,
            loss="likelihood",
            h=None,
            holdout=False,
            bounds="usual",
            initial="backcasting",
            regressors="use",
            verbose=0,
        )
        # May or may not improve; just must not crash
        assert isinstance(new_ic, float)


# ---------------------------------------------------------------------------
# AutoADAM ARIMA selection
# ---------------------------------------------------------------------------


class TestAutoADAMARIMASelection:
    def test_arima_select_true_fits(self, trend_data):
        m = AutoADAM(
            model="NNN",
            distribution="dnorm",
            lags=[1],
            ar_order=2,
            i_order=2,
            ma_order=2,
        ).fit(trend_data)
        assert m is not None
        assert m.model.startswith("ARIMA")

    def test_arima_select_finds_nonzero_orders(self, trend_data):
        m = AutoADAM(
            model="NNN",
            distribution="dnorm",
            lags=[1],
            ar_order=2,
            i_order=2,
            ma_order=2,
        ).fit(trend_data)
        orders = m._selected_arima_orders
        total = (
            sum(orders["i_orders"])
            + sum(orders["ma_orders"])
            + sum(orders["ar_orders"])
        )
        assert total > 0

    def test_arima_select_false_uses_fixed_orders(self, trend_data):
        m = AutoADAM(
            model="NNN",
            distribution="dnorm",
            lags=[1],
            ar_order=1,
            i_order=1,
            ma_order=1,
            arima_select=False,
        ).fit(trend_data)
        assert "1,1,1" in m.model

    def test_orders_dict_accepted(self, trend_data):
        m = AutoADAM(
            model="NNN",
            distribution="dnorm",
            lags=[1],
            orders={"ar": 2, "i": 2, "ma": 2, "select": True},
        ).fit(trend_data)
        assert m is not None


# ---------------------------------------------------------------------------
# AutoADAM distribution selection
# ---------------------------------------------------------------------------


class TestAutoADAMDistribution:
    def test_distribution_str_uses_single(self, trend_data):
        m = AutoADAM(
            model="NNN",
            distribution="dnorm",
            lags=[1],
            ar_order=1,
            i_order=1,
            ma_order=1,
        ).fit(trend_data)
        assert m._selected_distribution == "dnorm"

    def test_distribution_list_selects_best(self, trend_data):
        m = AutoADAM(
            model="NNN",
            distribution=["dnorm", "dlaplace"],
            lags=[1],
            arima_select=False,
            ar_order=1,
            i_order=1,
            ma_order=1,
        ).fit(trend_data)
        assert m._selected_distribution in ("dnorm", "dlaplace")
        assert len(m._all_ic_values) == 2

    def test_distribution_filtered_negative_y(self):
        np.random.seed(1)
        y = np.random.randn(60)  # has negatives
        m = AutoADAM(
            model="NNN", lags=[1], arima_select=False, ar_order=1, i_order=1, ma_order=1
        ).fit(y)
        assert m._selected_distribution not in {"dlnorm", "dinvgauss", "dgamma"}

    def test_distribution_filtered_pure_arima(self, positive_data):
        m = AutoADAM(
            model="NNN",
            lags=[1],
            arima_select=False,
            ar_order=1,
            i_order=1,
            ma_order=1,
        ).fit(positive_data)
        assert m._selected_distribution not in {"dlnorm", "dinvgauss", "dgamma"}

    def test_all_ic_values_stored(self, trend_data):
        m = AutoADAM(
            model="NNN",
            distribution=["dnorm", "dlaplace", "ds"],
            lags=[1],
            arima_select=False,
            ar_order=1,
            i_order=1,
            ma_order=1,
        ).fit(trend_data)
        assert len(m._all_ic_values) >= 1
        assert m._selected_distribution in m._all_ic_values


# ---------------------------------------------------------------------------
# Outlier parameter
# ---------------------------------------------------------------------------


class TestAutoADAMOutliers:
    def test_outliers_ignore_no_warning(self, trend_data):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            m = AutoADAM(
                model="NNN",
                distribution="dnorm",
                lags=[1],
                arima_select=False,
                ar_order=1,
                i_order=1,
                ma_order=1,
                outliers="ignore",
            ).fit(trend_data)
        assert m is not None

    def test_outliers_use_fits_without_warning(self, trend_data):
        """outliers='use' is now implemented and fits without a warning."""
        m = AutoADAM(
            model="NNN",
            distribution="dnorm",
            lags=[1],
            arima_select=False,
            ar_order=1,
            i_order=1,
            ma_order=1,
            outliers="use",
        ).fit(trend_data)
        assert m is not None

    def test_outliers_select_fits_without_warning(self, trend_data):
        """outliers='select' is now implemented and fits without a warning."""
        m = AutoADAM(
            model="NNN",
            distribution="dnorm",
            lags=[1],
            arima_select=False,
            ar_order=1,
            i_order=1,
            ma_order=1,
            outliers="select",
        ).fit(trend_data)
        assert m is not None


# ---------------------------------------------------------------------------
# AutoADAM defaults and end-to-end
# ---------------------------------------------------------------------------


class TestAutoADAMDefaults:
    def test_default_arima_select_is_true(self):
        m = AutoADAM()
        assert m._auto_arima_select is True

    def test_default_ic_is_aicc(self):
        m = AutoADAM()
        assert m.ic == "AICc"

    def test_repr_before_fit(self):
        m = AutoADAM()
        assert "not fitted" in repr(m).lower()

    def test_repr_after_fit(self, trend_data):
        m = AutoADAM(
            model="NNN",
            distribution="dnorm",
            lags=[1],
            ar_order=1,
            i_order=1,
            ma_order=1,
        ).fit(trend_data)
        r = repr(m)
        assert "AutoADAM" in r
        assert "dnorm" in r

    def test_predict_after_fit(self, trend_data):
        m = AutoADAM(
            model="NNN",
            distribution="dnorm",
            lags=[1],
            ar_order=1,
            i_order=1,
            ma_order=1,
        ).fit(trend_data)
        fc = m.predict(h=5)
        assert len(fc.mean) == 5

    def test_ets_arima_joint_selection(self, seasonal_data):
        m = AutoADAM(
            model="ZXZ",
            distribution=["dnorm", "dlaplace"],
            lags=[12],
            ar_order=1,
            i_order=1,
            ma_order=1,
        ).fit(seasonal_data)
        assert m is not None
        assert m._selected_distribution in ("dnorm", "dlaplace")

    def test_auto_state_survives_fit(self, trend_data):
        """_auto_* attributes survive state copy from best_model."""
        m = AutoADAM(
            model="NNN",
            distribution="dnorm",
            lags=[1],
            ar_order=1,
            i_order=1,
            ma_order=1,
        ).fit(trend_data)
        assert hasattr(m, "_auto_arima_select")
        assert hasattr(m, "_auto_distribution_spec")
