"""Tests for semiparametric, empirical, and nonparametric intervals, and rmultistep."""

import numpy as np
import pytest

from smooth import ADAM

AIRPASSENGERS = np.array(
    [
        112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
        115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
        145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
        171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
        196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
        204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
        242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
        284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
        315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
        340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
        360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
        417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432,
    ],
    dtype=float,
)
H = 4


@pytest.fixture(scope="module")
def ann_model():
    m = ADAM(model="ANN", lags=[1])
    m.fit(AIRPASSENGERS)
    return m


@pytest.fixture(scope="module")
def aan_model():
    m = ADAM(model="AAN", lags=[1, 12])
    m.fit(AIRPASSENGERS)
    return m


class TestRmultistep:
    def test_shape(self, ann_model):
        df = ann_model.rmultistep(h=H)
        t = len(AIRPASSENGERS)
        assert df.shape == (t - H, H)

    def test_column_names(self, ann_model):
        df = ann_model.rmultistep(h=H)
        assert list(df.columns) == [f"h={i+1}" for i in range(H)]

    def test_finite(self, ann_model):
        df = ann_model.rmultistep(h=H)
        assert np.all(np.isfinite(df.values))

    def test_h1_fallback(self, ann_model):
        df = ann_model.rmultistep(h=1)
        assert df.shape == (len(AIRPASSENGERS) - 1, 1)

    def test_requires_fit(self):
        m = ADAM(model="ANN", lags=[1])
        with pytest.raises(Exception):
            m.rmultistep(h=4)


class TestSemiparametric:
    @pytest.fixture(scope="class")
    def fc(self, ann_model):
        return ann_model.predict(h=H, interval="semiparametric", level=0.95)

    def test_shape(self, fc):
        assert fc.mean.shape == (H,)
        assert fc.upper.shape == (H, 1)
        assert fc.lower.shape == (H, 1)

    def test_finite(self, fc):
        assert np.all(np.isfinite(fc.mean.values))
        assert np.all(np.isfinite(fc.upper.values))
        assert np.all(np.isfinite(fc.lower.values))

    def test_monotonicity(self, fc):
        mean = fc.mean.values
        assert np.all(fc.lower.values.ravel() <= mean + 1e-6)
        assert np.all(mean <= fc.upper.values.ravel() + 1e-6)

    def test_label(self, fc):
        assert fc.interval == "semiparametric"

    def test_multi_level(self, ann_model):
        fc = ann_model.predict(h=H, interval="semiparametric", level=[0.8, 0.95])
        assert fc.upper.shape == (H, 2)
        assert fc.lower.shape == (H, 2)
        # Outer interval must be wider
        assert np.all(fc.lower.values[:, 1] <= fc.lower.values[:, 0] + 1e-6)
        assert np.all(fc.upper.values[:, 0] <= fc.upper.values[:, 1] + 1e-6)

    def test_seasonal_model(self, aan_model):
        fc = aan_model.predict(h=H, interval="semiparametric", level=0.95)
        assert fc.mean.shape == (H,)
        assert np.all(np.isfinite(fc.upper.values))
        assert np.all(np.isfinite(fc.lower.values))


class TestEmpirical:
    @pytest.fixture(scope="class")
    def fc(self, ann_model):
        return ann_model.predict(h=H, interval="empirical", level=0.95)

    def test_shape(self, fc):
        assert fc.mean.shape == (H,)
        assert fc.upper.shape == (H, 1)
        assert fc.lower.shape == (H, 1)

    def test_finite(self, fc):
        assert np.all(np.isfinite(fc.mean.values))
        assert np.all(np.isfinite(fc.upper.values))
        assert np.all(np.isfinite(fc.lower.values))

    def test_monotonicity(self, fc):
        mean = fc.mean.values
        assert np.all(fc.lower.values.ravel() <= mean + 1e-6)
        assert np.all(mean <= fc.upper.values.ravel() + 1e-6)

    def test_label(self, fc):
        assert fc.interval == "empirical"

    def test_h1_fallback(self, ann_model):
        fc = ann_model.predict(h=1, interval="empirical", level=0.95)
        assert fc.mean.shape == (1,)
        assert np.all(np.isfinite(fc.upper.values))

    def test_multi_level(self, ann_model):
        fc = ann_model.predict(h=H, interval="empirical", level=[0.8, 0.95])
        assert fc.upper.shape == (H, 2)
        assert np.all(fc.upper.values[:, 0] <= fc.upper.values[:, 1] + 1e-6)


class TestNonparametric:
    @pytest.fixture(scope="class")
    def fc(self, ann_model):
        return ann_model.predict(h=H, interval="nonparametric", level=0.95)

    def test_shape(self, fc):
        assert fc.mean.shape == (H,)
        assert fc.upper.shape == (H, 1)
        assert fc.lower.shape == (H, 1)

    def test_finite(self, fc):
        assert np.all(np.isfinite(fc.mean.values))
        assert np.all(np.isfinite(fc.upper.values))
        assert np.all(np.isfinite(fc.lower.values))

    def test_monotonicity(self, fc):
        mean = fc.mean.values
        assert np.all(fc.lower.values.ravel() <= mean + 1e-6)
        assert np.all(mean <= fc.upper.values.ravel() + 1e-6)

    def test_label(self, fc):
        assert fc.interval == "nonparametric"

    def test_h1_fallback(self, ann_model):
        fc = ann_model.predict(h=1, interval="nonparametric", level=0.95)
        assert fc.mean.shape == (1,)
        assert np.all(np.isfinite(fc.upper.values))

    def test_power_law_grows_with_horizon(self, fc):
        """Upper bounds from power-law fit should be non-decreasing (ANN model)."""
        upper = fc.upper.values.ravel()
        assert np.all(np.diff(upper) >= -1e-3), (
            "Nonparametric upper bounds should grow with horizon for ANN"
        )
