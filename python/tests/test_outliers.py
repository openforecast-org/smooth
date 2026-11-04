"""
Tests for rstandard(), rstudent(), outlierdummy() methods and outlier handling
in ADAM and AutoADAM.

R comparison values in this file were pre-computed from R's smooth package using
the identical AirPassengers dataset and model specification, and verified against
the Python implementation.
"""

import numpy as np
import pandas as pd
import pytest

from smooth import ADAM, AutoADAM
from smooth.adam_general.core.adam import OutlierDummy

# Full AirPassengers dataset (1949-1960, 144 monthly observations)
AIRPASSENGERS = np.array([
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
    417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432
], dtype=float)


@pytest.fixture(scope="module")
def ann_model():
    """Fitted ANN model on AirPassengers (no trend/seasonality)."""
    model = ADAM(model="ANN", lags=[1])
    model.fit(AIRPASSENGERS)
    return model


@pytest.fixture(scope="module")
def ana_model():
    """Fitted ANA model on AirPassengers (additive seasonality)."""
    model = ADAM(model="ANA", lags=[12])
    model.fit(AIRPASSENGERS)
    return model


@pytest.fixture
def model_with_outliers():
    """ANN model on a series with two injected outliers at positions 15 and 40."""
    np.random.seed(99)
    y = 50 + 0.3 * np.arange(60) + np.random.randn(60) * 1.5
    y[15] += 80   # large positive spike
    y[40] -= 80   # large negative spike
    model = ADAM(model="ANN", lags=[1])
    model.fit(y)
    return model, y


class TestRstandard:
    """Tests for the rstandard() method."""

    def test_returns_correct_length(self, ann_model):
        """rstandard() length equals nobs."""
        std_res = ann_model.rstandard()
        assert len(std_res) == ann_model.nobs

    def test_no_nan(self, ann_model):
        """rstandard() contains no NaN values."""
        std_res = ann_model.rstandard()
        assert not np.any(np.isnan(std_res))

    def test_approximately_standard_normal(self, ann_model):
        """For a dnorm model, rstandard() ≈ N(0, 1)."""
        std_res = ann_model.rstandard()
        # The errors at the location the model assumes (zero), divided by sigma:
        # not centred by their mean
        std_res, residuals = np.asarray(std_res), np.asarray(ann_model.residuals)
        sigma = np.dot(std_res, residuals) / np.dot(std_res, std_res)
        np.testing.assert_allclose(std_res * sigma, residuals, atol=1e-10)
        # Std should be close to 1 for a well-specified model
        assert 0.5 < np.std(std_res) < 2.0

    def test_raises_before_fit(self):
        """rstandard() raises before fit()."""
        model = ADAM(model="ANN")
        with pytest.raises((ValueError, AttributeError, RuntimeError)):
            model.rstandard()

    def test_seasonal_model(self, ana_model):
        """rstandard() works for seasonal model."""
        std_res = ana_model.rstandard()
        assert len(std_res) == ana_model.nobs
        assert not np.any(np.isnan(std_res))

    def test_returns_array_like(self, ann_model):
        """rstandard() returns an array-like (ndarray or Series)."""
        result = ann_model.rstandard()
        assert isinstance(result, (np.ndarray, pd.Series))


class TestRstudent:
    """Tests for the rstudent() method."""

    def test_returns_correct_length(self, ann_model):
        """rstudent() length equals nobs."""
        stu_res = ann_model.rstudent()
        assert len(stu_res) == ann_model.nobs

    def test_no_nan(self, ann_model):
        """rstudent() contains no NaN values."""
        stu_res = ann_model.rstudent()
        assert not np.any(np.isnan(stu_res))

    def test_more_spread_than_rstandard(self, ann_model):
        """rstudent() is at least as spread as rstandard()."""
        std_res = ann_model.rstandard()
        stu_res = ann_model.rstudent()
        assert np.std(stu_res) >= np.std(std_res) * 0.9

    def test_raises_before_fit(self):
        """rstudent() raises before fit()."""
        model = ADAM(model="ANN")
        with pytest.raises((ValueError, AttributeError, RuntimeError)):
            model.rstudent()

    def test_returns_array_like(self, ann_model):
        """rstudent() returns an array-like (ndarray or Series)."""
        result = ann_model.rstudent()
        assert isinstance(result, (np.ndarray, pd.Series))

    def test_outliers_more_extreme_than_rstandard(self, model_with_outliers):
        """rstudent() makes true outliers look more extreme than rstandard()."""
        model, _ = model_with_outliers
        std_res = model.rstandard()
        stu_res = model.rstudent()
        # The absolute max of rstudent should be at least as large
        assert np.max(np.abs(stu_res)) >= np.max(np.abs(std_res)) * 0.9


# R's rstudent() of adam(Y_SHORT, model, distribution=..., persistence=0.3,
# initial=115) (with shape=1.5 for dgnorm): all parameters provided, so both
# languages fit the same model
Y_SHORT = AIRPASSENGERS[:12]
R_RSTUDENT = {
    ("ANN", "dlaplace"): [
        -0.2254430076,
        0.2950715645,
        1.401861412,
        0.6839649569,
        -0.1417838419,
        1.026587405,
        1.914817144,
        1.267557761,
        -0.09531677356,
        -1.51468356,
        -2.536306111,
        -0.402954129,
    ],
    ("ANN", "ds"): [
        -0.9548428807,
        1.258474534,
        6.110779462,
        2.973591201,
        -0.5935920001,
        4.482252746,
        8.278049381,
        5.531618985,
        -0.3954421396,
        -6.593765311,
        -10.79694549,
        -1.732206423,
    ],
    ("ANN", "dgnorm"): [
        -0.159183735,
        0.2075750777,
        0.9637173336,
        0.4741816513,
        -0.1006222788,
        0.7071148991,
        1.322274261,
        0.8715171567,
        -0.06786017264,
        -1.041611544,
        -1.777953624,
        -0.2820533693,
    ],
    ("MNN", "dlnorm"): [
        0.8482186119,
        1.38430057,
        3.443623373,
        1.878882139,
        0.9264612572,
        2.41981531,
        4.44707016,
        2.668895352,
        0.9743272018,
        0.3138096449,
        0.1006795107,
        0.7363101366,
    ],
    ("MNN", "dgamma"): [
        0.9534246126,
        1.017883776,
        1.138394289,
        1.059141584,
        0.9647210806,
        1.092952406,
        1.169447373,
        1.105817133,
        0.971247406,
        0.8431121409,
        0.7616438623,
        0.9357172008,
    ],
}
R_RSTUDENT[("MNN", "dinvgauss")] = R_RSTUDENT[("MNN", "dgamma")]

# R's rstudent() of adam(Y_INTERMITTENT, model, distribution=..., persistence=0.3,
# initial=6, occurrence="fixed"): only the non-zero observations are studentised
Y_INTERMITTENT = np.array([0, 5, 0, 0, 7, 6, 0, 4, 0, 0, 8, 5, 0, 6, 0, 7], float)
R_RSTUDENT_OCCURRENCE = {
    ("MNN", "dgamma"): [
        1,
        0.7942530193,
        1,
        1,
        1.256463588,
        0.9363955956,
        1,
        0.6058719097,
        1,
        1,
        1.588224263,
        0.7360949527,
        1,
        0.9929397713,
        1,
        1.177226032,
    ],
    ("ANN", "dnorm"): [
        0,
        -0.5655173879,
        0,
        0,
        0.7411461062,
        -0.1215516923,
        0,
        -1.34667079,
        0,
        0,
        1.745475279,
        -0.8736199014,
        0,
        0.02606258751,
        0,
        0.5679150353,
    ],
}


class TestRstudentAsR:
    """rstudent() of the other distributions, against R's rstudent.adam()."""

    @pytest.mark.parametrize("model, distribution", list(R_RSTUDENT))
    def test_the_values_are_those_of_r(self, model, distribution):
        extra = {"gnorm_shape": 1.5} if distribution == "dgnorm" else {}
        fit = ADAM(
            model=model,
            distribution=distribution,
            persistence={"level": 0.3},
            initial={"level": 115.0},
            **extra,
        ).fit(Y_SHORT)
        np.testing.assert_allclose(
            np.asarray(fit.rstudent()), R_RSTUDENT[(model, distribution)], rtol=1e-8
        )

    @pytest.mark.parametrize(
        "model, distribution",
        [
            ("MNN", "dgamma"),
            ("ANN", "dnorm"),
        ],
    )
    def test_the_occurrence_values_are_those_of_r(self, model, distribution):
        fit = ADAM(
            model=model,
            distribution=distribution,
            persistence={"level": 0.3},
            initial={"level": 6.0},
            occurrence="fixed",
        ).fit(Y_INTERMITTENT)
        np.testing.assert_allclose(
            np.asarray(fit.rstudent()),
            R_RSTUDENT_OCCURRENCE[(model, distribution)],
            rtol=1e-8,
            atol=1e-12,
        )


class TestOutlierdummy:
    """Tests for the outlierdummy() method."""

    def test_returns_outlier_dummy_dataclass(self, ann_model):
        """outlierdummy() returns an OutlierDummy instance."""
        od = ann_model.outlierdummy()
        assert isinstance(od, OutlierDummy)

    def test_fields_present(self, ann_model):
        """OutlierDummy has the expected fields."""
        od = ann_model.outlierdummy()
        assert hasattr(od, "outliers")
        assert hasattr(od, "id")
        assert hasattr(od, "statistic")
        assert hasattr(od, "level")
        assert hasattr(od, "type")

    def test_statistic_shape(self, ann_model):
        """statistic is a 2-element array [lower, upper]."""
        od = ann_model.outlierdummy()
        assert od.statistic.shape == (2,)
        assert od.statistic[0] < od.statistic[1]

    def test_level_stored(self, ann_model):
        """Level used is stored in the result."""
        od = ann_model.outlierdummy(level=0.99)
        assert od.level == 0.99

    def test_type_stored(self, ann_model):
        """Type used is stored in the result."""
        od = ann_model.outlierdummy(type="rstandard")
        assert od.type == "rstandard"

        od2 = ann_model.outlierdummy(type="rstudent")
        assert od2.type == "rstudent"

    def test_invalid_type_raises(self, ann_model):
        """Invalid type raises ValueError."""
        with pytest.raises(ValueError):
            ann_model.outlierdummy(type="invalid")

    def test_raises_before_fit(self):
        """outlierdummy() raises before fit()."""
        model = ADAM(model="ANN")
        with pytest.raises((ValueError, AttributeError, RuntimeError)):
            model.outlierdummy()

    def test_clean_data_no_outliers(self):
        """Clean data at tight level should produce few or no outliers."""
        np.random.seed(7)
        y = 50 + np.arange(80) * 0.2 + np.random.randn(80) * 0.5
        model = ADAM(model="ANN", lags=[1])
        model.fit(y)
        od = model.outlierdummy(level=0.9999)
        assert len(od.id) == 0
        assert od.outliers is None

    def test_detects_injected_outliers(self, model_with_outliers):
        """Injected outliers at positions 15 and 40 are both detected."""
        model, _ = model_with_outliers
        od = model.outlierdummy(level=0.99)
        assert 15 in od.id
        assert 40 in od.id

    def test_dummy_matrix_shape(self, model_with_outliers):
        """Dummy matrix has shape (nobs, n_outliers)."""
        model, _ = model_with_outliers
        od = model.outlierdummy(level=0.99)
        assert od.outliers is not None
        assert od.outliers.shape == (model.nobs, len(od.id))

    def test_dummy_matrix_binary(self, model_with_outliers):
        """Dummy matrix contains only 0.0 and 1.0."""
        model, _ = model_with_outliers
        od = model.outlierdummy(level=0.99)
        assert od.outliers is not None
        unique_vals = np.unique(od.outliers)
        assert set(unique_vals).issubset({0.0, 1.0})

    def test_dummy_column_has_single_one(self, model_with_outliers):
        """Each dummy column has exactly one 1.0 at the outlier position."""
        model, _ = model_with_outliers
        od = model.outlierdummy(level=0.99)
        assert od.outliers is not None
        for j, idx in enumerate(od.id):
            col = od.outliers[:, j]
            assert col[idx] == 1.0
            assert np.sum(col) == 1.0

    def test_rstudent_type_detects_same_outliers(self, model_with_outliers):
        """rstudent type also detects injected outliers."""
        model, _ = model_with_outliers
        od = model.outlierdummy(level=0.99, type="rstudent")
        assert 15 in od.id
        assert 40 in od.id

    def test_stricter_level_fewer_outliers(self, model_with_outliers):
        """Stricter level detects fewer or equal outliers."""
        model, _ = model_with_outliers
        od_loose = model.outlierdummy(level=0.90)
        od_strict = model.outlierdummy(level=0.9999)
        assert len(od_strict.id) <= len(od_loose.id)


class TestExpandOutlierDummies:
    """Tests for the _expand_outlier_dummies() static method."""

    def test_output_columns_triple(self):
        """Expanding m columns produces 3*m columns."""
        D = np.eye(3, dtype=float)
        E = ADAM._expand_outlier_dummies(D)
        assert E.shape == (3, 9)

    def test_lag_structure(self):
        """The dummy, its lag and its lead, in the order of R's xregExpander()."""
        # Single outlier at row 2 in a 5-row matrix
        D = np.zeros((5, 1))
        D[2, 0] = 1.0
        E = ADAM._expand_outlier_dummies(D)
        assert E.shape == (5, 3)
        np.testing.assert_array_equal(E[:, 0], [0, 0, 1, 0, 0])  # t
        np.testing.assert_array_equal(E[:, 1], [0, 0, 0, 1, 0])  # Lag1
        np.testing.assert_array_equal(E[:, 2], [0, 1, 0, 0, 0])  # Lead1

    def test_boundary_lag(self):
        """The lead of the first row is zero (gaps="zero")."""
        D = np.zeros((4, 1))
        D[0, 0] = 1.0
        E = ADAM._expand_outlier_dummies(D)
        np.testing.assert_array_equal(E[:, 0], [1, 0, 0, 0])  # t
        np.testing.assert_array_equal(E[:, 1], [0, 1, 0, 0])  # Lag1
        np.testing.assert_array_equal(E[:, 2], [0, 0, 0, 0])  # Lead1

    def test_boundary_lead(self):
        """The lag of the last row is zero (gaps="zero")."""
        D = np.zeros((4, 1))
        D[3, 0] = 1.0
        E = ADAM._expand_outlier_dummies(D)
        np.testing.assert_array_equal(E[:, 0], [0, 0, 0, 1])  # t
        np.testing.assert_array_equal(E[:, 1], [0, 0, 0, 0])  # Lag1
        np.testing.assert_array_equal(E[:, 2], [0, 0, 1, 0])  # Lead1


class TestADAMOutlierHandling:
    """Integration tests for outliers= parameter in ADAM.fit()."""

    def test_outliers_ignore_default(self, ann_model):
        """outliers='ignore' (default) does not change nparam."""
        np.random.seed(42)
        y = 50 + np.arange(80) + np.random.randn(80) * 2
        m1 = ADAM(model="ANN", outliers="ignore")
        m1.fit(y)
        m2 = ADAM(model="ANN")
        m2.fit(y)
        assert m1.nparam == m2.nparam

    def test_outliers_use_increases_nparam(self):
        """outliers='use' increases nparam by number of detected dummies."""
        np.random.seed(7)
        y = 50 + np.arange(80) * 0.5 + np.random.randn(80) * 1.5
        y[20] += 100
        y[55] -= 100

        m_base = ADAM(model="ANN", lags=[1])
        m_base.fit(y)
        n_base = m_base.nparam

        od = m_base.outlierdummy(level=0.99)
        n_outliers = len(od.id)

        m_use = ADAM(model="ANN", lags=[1], outliers="use", outliers_level=0.99)
        m_use.fit(y)

        assert m_use.nparam >= n_base + n_outliers

    def test_outliers_use_config_preserved(self):
        """_config['outliers'] reflects the original setting after fit."""
        np.random.seed(7)
        y = 50 + np.arange(60) * 0.5 + np.random.randn(60) * 1.0
        y[20] += 80

        model = ADAM(model="ANN", lags=[1], outliers="use", outliers_level=0.99)
        model.fit(y)
        assert model._config["outliers"] == "use"

    def test_outliers_select_config_preserved(self):
        """_config['outliers'] == 'select' after fit with outliers='select'."""
        np.random.seed(7)
        y = 50 + np.arange(60) * 0.5 + np.random.randn(60) * 1.0
        y[20] += 80

        model = ADAM(model="ANN", lags=[1], outliers="select", outliers_level=0.99)
        model.fit(y)
        assert model._config["outliers"] == "select"

    def test_outliers_use_improves_fit(self):
        """outliers='use' should give lower or equal AICc than base model."""
        np.random.seed(7)
        y = 50 + np.arange(80) * 0.5 + np.random.randn(80) * 1.5
        y[20] += 100
        y[55] -= 100

        m_base = ADAM(model="ANN", lags=[1])
        m_base.fit(y)

        m_use = ADAM(model="ANN", lags=[1], outliers="use", outliers_level=0.99)
        m_use.fit(y)

        # With outlier dummies, the loglikelihood should improve
        assert m_use.loglik >= m_base.loglik - 1.0


class TestAutoADAMOutlierHandling:
    """Integration tests for outliers= parameter in AutoADAM."""

    def test_autoadam_outliers_use_config(self):
        """AutoADAM with outliers='use' preserves setting in _config."""
        np.random.seed(7)
        y = 50 + np.arange(60) * 0.5 + np.random.randn(60) * 1.0
        y[20] += 80

        model = AutoADAM(
            model="ANN",
            lags=[1],
            arima_select=False,
            distribution=["dnorm"],
            outliers="use",
            level=0.99,
        )
        model.fit(y)
        assert model._config["outliers"] == "use"

    def test_autoadam_outliers_ignore(self):
        """AutoADAM with outliers='ignore' does not add outlier dummy params."""
        np.random.seed(7)
        y = 50 + np.arange(60) * 0.5 + np.random.randn(60) * 1.0

        m_ignore = AutoADAM(
            model="ANN", lags=[1],
            arima_select=False, ar_order=0, i_order=0, ma_order=0,
            distribution=["dnorm"],
            outliers="ignore",
        )
        m_ignore.fit(y)

        m_base = ADAM(model="ANN", lags=[1])
        m_base.fit(y)

        assert m_ignore.nparam == m_base.nparam


class TestRComparisonWithR:
    """
    Tests comparing Python rstandard/outlierdummy results against R reference values.

    Reference values were computed in R using:
        pkgload::load_all(".")
        y <- AirPassengers
        m <- adam(y, "ANN", lags=1)
        sr <- rstandard(m)
        od <- outlierdummy(m, level=0.999)
    """

    def test_rstandard_dnorm_statistic_bounds(self, ann_model):
        """
        For dnorm at level=0.999, statistic bounds match norm.ppf([0.0005, 0.9995]).

        These bounds are purely analytical and identical between R and Python.
        """
        from scipy import stats

        od = ann_model.outlierdummy(level=0.999)
        p = np.array([0.0005, 0.9995])
        expected = stats.norm.ppf(p)
        np.testing.assert_allclose(od.statistic, expected, rtol=1e-6)

    def test_rstandard_is_not_centred(self, ann_model):
        """
        rstandard() for dnorm takes the errors at the location the model assumes.

        R's rstandard.adam() divides the errors by sigma without subtracting their
        mean, so the mean of the result shows a bias of the model.
        """
        std_res = np.asarray(ann_model.rstandard())
        residuals = np.asarray(ann_model.residuals)
        sigma = np.dot(std_res, residuals) / np.dot(std_res, std_res)
        np.testing.assert_allclose(np.mean(std_res) * sigma, np.mean(residuals))

    def test_rstandard_length_equals_nobs(self, ann_model):
        """Length of rstandard() equals nobs (144 for full AirPassengers)."""
        assert len(ann_model.rstandard()) == 144

    def test_outlierdummy_no_outliers_in_airpassengers_ann(self, ann_model):
        """
        AirPassengers ANN model at level=0.999 should detect no outliers.

        R reference:
            od <- outlierdummy(adam(AirPassengers, "ANN", lags=1), level=0.999)
            length(od$id)  # 0
        """
        od = ann_model.outlierdummy(level=0.999)
        assert len(od.id) == 0

    def test_rstandard_seasonal_model_bounds(self, ana_model):
        """
        rstandard() seasonal ANA model: statistic bounds match R for dnorm.
        """
        from scipy import stats

        od = ana_model.outlierdummy(level=0.99)
        p = np.array([0.005, 0.995])
        expected = stats.norm.ppf(p)
        np.testing.assert_allclose(od.statistic, expected, rtol=1e-6)

    def test_rstandard_vs_rstudent_correlation(self, ann_model):
        """
        rstandard() and rstudent() are highly correlated for a clean dataset.

        Both R and Python implementations produce this property by construction.
        """
        std_res = ann_model.rstandard()
        stu_res = ann_model.rstudent()
        corr = np.corrcoef(std_res, stu_res)[0, 1]
        assert corr > 0.99

    def test_rstandard_reference_values_ann(self, ann_model):
        """
        rstandard() reference values for ANN on AirPassengers.

        R's ``rstandard.adam`` divides by the square root of the scale (sigma^2
        for dnorm) de-biased by ``n/df``, with ``df = nobs - (nparam - 1)``: the
        scale itself is not counted under likelihood. R reports nparam=3,
        scale=1128.5 and a divisor of 33.82889839 for this model.
        """
        std_res = ann_model.rstandard()
        obs = ann_model.nobs         # 144
        df = obs - (ann_model.nparam - 1)

        errors = ann_model.residuals.copy()
        errors -= errors.mean()
        expected_scale = np.sqrt(ann_model.scale * obs / df)
        np.testing.assert_allclose(
            np.std(errors) / np.std(std_res), expected_scale, rtol=1e-6
        )
        np.testing.assert_allclose(expected_scale, 33.82889839, rtol=1e-6)


@pytest.mark.r_parity
@pytest.mark.parametrize(
    "distribution", ["dnorm", "dlaplace", "ds", "dgnorm", "dlnorm", "dgamma", "dinvgauss"]
)
@pytest.mark.parametrize("kind", ["rstandard", "rstudent"])
def test_outlierdummy_agrees_with_r(distribution, kind):
    """The bounds are the quantiles of R's outlierdummy.adam, from greybox."""
    from ._r_bridge import r_dict

    r = r_dict(
        "{ y <- as.numeric(AirPassengers); y[c(30,80)] <- y[c(30,80)]*1.5;"
        f" m <- adam(ts(y, frequency=12), 'MAM', distribution='{distribution}');"
        f" o <- outlierdummy(m, level=0.99, type='{kind}');"
        " list(y=y, id=as.numeric(o$id), statistic=as.numeric(o$statistic)) }"
    )
    model = ADAM(model="MAM", lags=[12], distribution=distribution)
    result = model.fit(np.asarray(r["y"], dtype=float)).outlierdummy(0.99, type=kind)
    np.testing.assert_allclose(result.statistic, r["statistic"], rtol=1e-10)
    assert list(np.asarray(result.id) + 1) == [int(i) for i in r["id"] or []]


@pytest.mark.r_parity
def test_outlierdummy_of_a_mixture_agrees_with_r():
    from ._r_bridge import r_dict

    r = r_dict(
        "{ set.seed(3); y <- rbinom(200,1,0.4)*exp(rnorm(200,2,0.3));"
        " o <- outlierdummy(adam(y, 'MNN', occurrence='odds-ratio'));"
        " list(y=y, id=as.numeric(o$id), statistic=as.numeric(o$statistic)) }"
    )
    model = ADAM(model="MNN", occurrence="odds-ratio")
    result = model.fit(np.asarray(r["y"], dtype=float)).outlierdummy()
    np.testing.assert_allclose(result.statistic, r["statistic"], rtol=1e-10)
    assert list(np.asarray(result.id) + 1) == [int(i) for i in r["id"]]
