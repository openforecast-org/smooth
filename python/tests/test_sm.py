"""Parity tests for ``sm()`` -- the scale model -- against R's ``sm.adam``.

Reference values come from ``python/tests/R scripts/sm_reference.R``; regenerate
with::

    Rscript "python/tests/R scripts/sm_reference.R"
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smooth import ADAM
from smooth.adam_general.core.sm import (
    SUPPORTED_DISTRIBUTIONS,
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
        residuals,
        # No residual where no size is observed: R writes NA
        np.array([np.nan if v == "NA" else v for v in ref["sm_residuals"]], float),
        atol=1e-8,
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


FORECAST_REFERENCE = json.loads((DATA / "sm_forecast_reference.json").read_text())
SIMULATED_REFERENCE = json.loads((DATA / "sm_simulated_reference.json").read_text())


def _with_scale_model(distribution):
    location = ADAM(model="ANN", lags=[1], distribution=distribution)
    location.fit(_series("positive"))
    location.scale_model = location.sm()
    return location


@pytest.mark.parametrize("distribution", sorted(FORECAST_REFERENCE))
def test_analytical_interval_uses_the_scale_model(distribution):
    """A time-varying scale must reach the analytical prediction interval."""
    ref = FORECAST_REFERENCE[distribution]
    location = _with_scale_model(distribution)
    forecast = location.predict(h=12, interval="prediction", level=0.95)

    np.testing.assert_allclose(
        np.asarray(location.scale_model.predict(h=12).mean, dtype=float).ravel(),
        np.array(ref["scale_forecast"], dtype=float),
        rtol=1e-9,
    )
    for name, got in (("lower", forecast.lower), ("upper", forecast.upper)):
        np.testing.assert_allclose(
            np.asarray(got, dtype=float).ravel(),
            np.array(ref[name], dtype=float),
            rtol=1e-8,
            err_msg=f"{distribution}: {name} differs from R",
        )


@pytest.mark.parametrize("distribution", sorted(FORECAST_REFERENCE))
def test_scale_model_widens_or_narrows_against_the_constant_scale(distribution):
    """The interval must change once a time-varying scale model is attached.

    Without this the previous two assertions would still pass if the scale
    forecast were computed and then quietly dropped. A flat scale model sitting
    on the estimated scale must instead reproduce the constant-scale interval,
    as both are then the same model.
    """
    plain = ADAM(model="ANN", lags=[1], distribution=distribution)
    plain.fit(_series("positive"))
    flat = plain.predict(h=12, interval="prediction", level=0.95)
    location = _with_scale_model(distribution)
    varying = location.predict(h=12, interval="prediction", level=0.95)
    flat_width = np.asarray(flat.upper, float) - np.asarray(flat.lower, float)
    varying_width = np.asarray(varying.upper, float) - np.asarray(varying.lower, float)
    scale_fitted = np.asarray(location.scale_model.fitted, float)
    if np.allclose(scale_fitted, plain.scale, rtol=1e-10):
        np.testing.assert_allclose(flat_width, varying_width, rtol=1e-10)
    else:
        assert not np.allclose(flat_width, varying_width)


@pytest.mark.parametrize("distribution", sorted(SIMULATED_REFERENCE))
def test_simulated_interval_scale_matches_r(distribution):
    """The simulator's per-horizon scale, which R de-biases and roots."""
    import smooth.adam_general.core.forecaster.intervals as intervals_module

    ref = SIMULATED_REFERENCE[distribution]
    captured = {}
    original = intervals_module.generate_errors

    def _capture(distribution, n, scale, **kwargs):
        captured["scale"] = np.asarray(scale, dtype=float).ravel()
        return original(distribution, n, scale, **kwargs)

    intervals_module.generate_errors = _capture
    try:
        _with_scale_model(distribution).predict(
            h=12, interval="simulated", level=0.95, nsim=50
        )
    finally:
        intervals_module.generate_errors = original

    np.testing.assert_allclose(
        captured["scale"][:12], np.array(ref["sim_scale"], dtype=float), rtol=1e-9
    )


def test_reforecast_accepts_a_per_horizon_sigma():
    """``reforecast`` must take the scale model's forecast as its sigma."""
    location = _with_scale_model("dnorm")
    result = location.reforecast(h=12, nsim=20, interval="prediction", seed=7)
    assert result.paths.shape == (12, 20, 20)
    assert np.all(np.isfinite(np.asarray(result.mean, dtype=float)))


def test_the_scale_model_takes_the_missing_values_for_gaps():
    """The gaps of the location model are skipped by the scale model too, and are
    not an occurrence of it."""
    y = _series("positive").copy()
    y[[9, 49, 50]] = np.nan
    location = ADAM(model="MAM", lags=[1, 12]).fit(y)
    scale = sm(location)
    assert np.isfinite(scale.loglik)
    assert "iETS" not in str(scale.model_name)


def test_the_scale_model_in_logs_sums_in_the_space_of_the_data():
    """The cumulative forecasts of a scale model fitted in logs are the sums of the
    scale, simulated for the intervals, and its refitted values and reforecasts are
    the scale, as R."""
    scale = sm(ADAM(model="ANN", lags=[1]).fit(_series("positive")), model="ANN")
    assert scale.model_name_sm_.endswith(" in logs")
    # Its data are the scale, and its standardised residuals stay after forecasting
    assert np.all(np.asarray(scale.actuals) > 0)
    residuals = np.asarray(scale.residuals).copy()
    forecasts = np.asarray(scale.predict(h=4).mean)
    np.testing.assert_allclose(scale.predict(h=4, cumulative=True).mean, forecasts.sum())
    cumulative = scale.predict(h=4, interval="approximate", cumulative=True, seed=41)
    assert cumulative.interval == "simulated"
    assert 0 < float(cumulative.lower.iloc[0, 0]) < forecasts.sum()
    np.testing.assert_allclose(
        np.median(np.asarray(scale.reapply(nsim=100, seed=41).refitted), axis=1),
        np.asarray(scale.fitted),
        rtol=0.05,
    )
    # The mean of the scale is above its skeleton, the median of a model in logs
    mean = np.asarray(scale.predict(h=4, point="mean", seed=41).mean)
    assert np.all(mean > forecasts)
    simulated = scale.predict(h=2, interval="simulated", nsim=20000, seed=41)
    reforecast = scale.reforecast(h=2, nsim=200, seed=41)
    np.testing.assert_allclose(
        np.ravel(reforecast.lower), np.ravel(simulated.lower), rtol=0.2
    )
    np.testing.assert_array_equal(np.asarray(scale.residuals), residuals)


# The scale model in logs exponentiates its fitted values and forecasts, and the
# analytical variance of a multiplicative model averages the de-biased fitted scale,
# as R
SM_PARITY_CASES = {
    "logs": ("AAN", "dnorm", "ANN"),
    "multiplicative": ("MNM", "dgamma", "MNN"),
}


@pytest.mark.r_parity
@pytest.mark.parametrize("case", sorted(SM_PARITY_CASES))
def test_the_scale_model_forecasts_agree(case):
    sys.path.insert(0, str(Path(__file__).parent))
    from _r_bridge import r_dict

    model, distribution, scale_model = SM_PARITY_CASES[case]
    r = r_dict(
        "{ y <- as.numeric(AirPassengers);"
        f" m <- adam(ts(y, frequency=12), '{model}', distribution='{distribution}');"
        f" s <- suppressWarnings(sm(m, model='{scale_model}')); m <- implant(m, s);"
        " f <- forecast(m, h=6, interval='approximate', level=0.9);"
        " list(y=y, fitted=as.numeric(fitted(s)),"
        " scale=as.numeric(forecast(s, h=6)$mean),"
        " lower=as.numeric(f$lower), upper=as.numeric(f$upper)) }"
    )
    location = ADAM(model=model, lags=[12], distribution=distribution)
    location.fit(np.array(r["y"], float))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        location.scale_model = location.sm(model=scale_model)
    forecast = location.predict(h=6, interval="approximate", level=0.9)
    np.testing.assert_allclose(np.ravel(location.scale_model.fitted), r["fitted"])
    np.testing.assert_allclose(
        np.ravel(location.scale_model.predict(h=6).mean), r["scale"]
    )
    np.testing.assert_allclose(np.ravel(forecast.lower), r["lower"], rtol=1e-8)
    np.testing.assert_allclose(np.ravel(forecast.upper), r["upper"], rtol=1e-8)
