"""Parity tests for ``sm()`` -- the scale model -- against R's ``sm.adam``.

Reference values come from ``python/tests/R scripts/sm_reference.R``; regenerate
with::

    Rscript "python/tests/R scripts/sm_reference.R"
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smooth import ADAM
from smooth.adam_general.core.sm import (
    SUPPORTED_DISTRIBUTIONS,
    _differential_entropy,
    _log_density,
    sm,
)

DATA = Path(__file__).parent / "data"
REFERENCE = json.loads((DATA / "sm_reference.json").read_text())

# R's sm() reuses the location model's already-fitted occurrence model rather
# than estimating a second one on the transformed residuals.
OCCURRENCE_CASES = {"occ_dnorm", "occ_MNN_dnorm"}


def _series(name):
    return pd.read_csv(DATA / f"sm_{name}.csv")["y"].to_numpy(float)


def _location(case_id, ref):
    kwargs = {
        "model": ref["loc_model"],
        "lags": [1, 12] if "seasonal" in case_id else [1],
        "distribution": ref["distribution"],
    }
    if "holdout" in case_id:
        kwargs.update(h=12, holdout=True)
    if case_id in OCCURRENCE_CASES:
        kwargs["occurrence"] = "odds-ratio"
    model = ADAM(**kwargs)
    model.fit(_series(ref["data"]))
    return model


def _scale_model(case_id, ref):
    kwargs = {"model": "MNN"} if case_id.endswith("smMNN") else {}
    return sm(_location(case_id, ref), **kwargs)


@pytest.mark.parametrize("case_id", sorted(REFERENCE))
def test_location_matches_r(case_id):
    """The location model must match R before its scale model can."""
    ref = REFERENCE[case_id]
    model = _location(case_id, ref)
    assert model.loglik == pytest.approx(ref["loc_logLik"], abs=1e-8)
    assert model.scale == pytest.approx(ref["loc_scale"], abs=1e-8)


@pytest.mark.parametrize("case_id", sorted(REFERENCE))
def test_scale_model_matches_r(case_id):
    ref = REFERENCE[case_id]
    scale_model = _scale_model(case_id, ref)

    assert scale_model.model_name == ref["sm_model"]
    assert scale_model.loglik_sm_ == pytest.approx(ref["sm_logLik"], abs=1e-8)
    assert scale_model.nparam == ref["sm_nparam"]
    assert scale_model.df_sm_ == ref["sm_df_stored"]
    assert scale_model.is_scale_ is True

    fitted = np.asarray(scale_model.fitted, dtype=float).ravel()
    np.testing.assert_allclose(fitted, np.array(ref["sm_fitted"], float), atol=1e-8)

    residuals = np.asarray(scale_model.residuals, dtype=float).ravel()
    np.testing.assert_allclose(
        residuals, np.array(ref["sm_residuals"], float), atol=1e-8
    )


def test_rejects_non_likelihood_loss():
    y = _series("positive")
    model = ADAM(model="ANN", lags=[1], loss="MSE")
    model.fit(y)
    with pytest.raises(ValueError, match="maximisation of likelihood"):
        sm(model)


def test_rejects_unsupported_distribution():
    """R's loss switch has no branch for dalaplace, so sm() cannot score it."""
    assert "dalaplace" not in SUPPORTED_DISTRIBUTIONS


def test_log_trick_warns_and_exponentiates():
    """An additive scale model on an additive distribution is fitted in logs."""
    y = _series("positive")
    model = ADAM(model="ANN", lags=[1], distribution="dnorm")
    model.fit(y)
    with pytest.warns(UserWarning, match="logarithms"):
        scale_model = sm(model, model="ANN")
    # The exponentiated fitted scale must be positive everywhere.
    assert np.all(np.asarray(scale_model.fitted, dtype=float) > 0)


def test_selection_letters_warn():
    y = _series("positive")
    model = ADAM(model="ANN", lags=[1], distribution="dnorm")
    model.fit(y)
    with pytest.warns(UserWarning, match="not supported by the sm"):
        sm(model, model="ZZZ")


@pytest.mark.parametrize("distribution", SUPPORTED_DISTRIBUTIONS)
def test_log_density_is_finite_and_shaped(distribution):
    y = np.array([3.0, 5.5, 1.2, 9.9, 4.4])
    mu = np.array([3.4, 5.0, 1.5, 9.0, 4.0])
    scale = np.array([0.30, 0.55, 0.12, 0.80, 0.44])
    out = _log_density(distribution, "A", y, mu, scale, 1.7)
    assert out.shape == y.shape
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize("distribution", SUPPORTED_DISTRIBUTIONS)
def test_entropy_is_zero_without_zero_observations(distribution):
    """No zero observations means no entropy contribution."""
    assert _differential_entropy(distribution, np.array([]), 1.7) == 0.0


IMPLANT_REFERENCE = json.loads((DATA / "sm_implant_reference.json").read_text())


@pytest.mark.parametrize("distribution", sorted(IMPLANT_REFERENCE))
def test_implant_matches_r(distribution):
    """Attaching a scale model reproduces R's implant() + extractScale/Sigma."""
    ref = IMPLANT_REFERENCE[distribution]
    location = ADAM(model="ANN", lags=[1], distribution=distribution)
    location.fit(_series("positive"))
    scale_model = location.sm()

    assert location.scale_model is None
    location.scale_model = scale_model
    assert location.scale_model is scale_model

    np.testing.assert_allclose(
        np.asarray(location.extract_scale(), float).ravel()[:5],
        np.array(ref["scale_head"], float),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(location.extract_sigma(), float).ravel()[:5],
        np.array(ref["sigma_head"], float),
        atol=1e-8,
    )
    assert location.loglik == pytest.approx(ref["logLik"], abs=1e-8)
    assert location.nparam == ref["nparam"]
    # A scale model asked for its own scale returns 1: it is the scale.
    assert scale_model.extract_scale() == pytest.approx(ref["sm_scale_own"])


def test_extract_scale_without_scale_model_is_the_scalar():
    location = ADAM(model="ANN", lags=[1], distribution="dnorm")
    location.fit(_series("positive"))
    assert location.extract_scale() == location.scale
    assert location.extract_sigma() == location.sigma


def test_scale_model_setter_rejects_a_plain_model():
    y = _series("positive")
    location = ADAM(model="ANN", lags=[1], distribution="dnorm")
    location.fit(y)
    other = ADAM(model="ANN", lags=[1], distribution="dnorm")
    other.fit(y)
    with pytest.raises(ValueError, match="Not a scale model"):
        location.scale_model = other


def test_scale_model_can_be_detached():
    location = ADAM(model="ANN", lags=[1], distribution="dnorm")
    location.fit(_series("positive"))
    plain_loglik, plain_nparam = location.loglik, location.nparam
    location.scale_model = location.sm()
    assert location.nparam != plain_nparam
    location.scale_model = None
    assert location.loglik == plain_loglik
    assert location.nparam == plain_nparam
