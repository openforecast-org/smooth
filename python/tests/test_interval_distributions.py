"""Prediction-interval parity with R for every error distribution.

Reference values come from
``python/tests/R scripts/interval_distributions_reference.R``.

The quantile functions themselves are greybox's, so both languages call the
same code; these tests pin the parameterisation each distribution is called
with, which is where the two used to part company.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smooth import ADAM

DATA = Path(__file__).parent / "data"
REFERENCE = json.loads((DATA / "interval_distributions_reference.json").read_text())


@pytest.mark.parametrize("case_id", sorted(REFERENCE))
def test_interval_matches_r(case_id):
    ref = REFERENCE[case_id]
    model_spec, distribution = case_id.split("_", 1)
    y = pd.read_csv(DATA / "sm_positive.csv")["y"].to_numpy(float)

    model = ADAM(model=model_spec, lags=[1], distribution=distribution)
    model.fit(y)
    forecast = model.predict(h=8, interval="prediction", level=0.95)

    assert model.sigma == pytest.approx(ref["sigma"], rel=1e-9)
    for name, got in (
        ("mean", forecast.mean),
        ("lower", forecast.lower),
        ("upper", forecast.upper),
    ):
        np.testing.assert_allclose(
            np.asarray(got, dtype=float).ravel(),
            np.array(ref[name], dtype=float),
            rtol=1e-8,
            err_msg=f"{case_id}: {name} differs from R",
        )


@pytest.mark.parametrize("case_id", sorted(REFERENCE))
def test_interval_is_not_degenerate(case_id):
    """An interval that collapses onto the point forecast is a broken one.

    Every distribution here previously either collapsed (``ds``, ``dgnorm``,
    ``dlnorm``) or came back orders of magnitude out (``dgamma``,
    ``dinvgauss``), so this guards the failure mode directly rather than only
    through the reference values.
    """
    ref = REFERENCE[case_id]
    lower = np.array(ref["lower"], dtype=float)
    upper = np.array(ref["upper"], dtype=float)
    mean = np.array(ref["mean"], dtype=float)
    assert np.all(upper - lower > 1e-6)
    assert np.all(lower < mean) and np.all(mean < upper)
