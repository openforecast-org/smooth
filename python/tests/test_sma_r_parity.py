"""R parity tests for SMA: Python vs R smooth::sma().

Run locally with:  pytest tests/test_sma_r_parity.py -m r_parity -v
Skipped in CI by default (requires R + smooth + devtools).
"""

from __future__ import annotations

import json
import subprocess

import numpy as np
import pytest

from smooth import SMA

pytestmark = pytest.mark.r_parity

# Hardcoded 60-obs deterministic series — identical values in R and Python.
Y = np.array(
    [
        102.3, 98.1, 104.5, 99.8, 101.2, 103.7, 97.6, 100.9, 105.1, 98.4,
        101.8, 103.2, 99.5, 102.7, 100.3, 97.9, 104.1, 101.5, 98.7, 103.4,
        100.6, 102.1, 98.9, 101.3, 104.8, 99.2, 102.5, 100.7, 98.3, 103.9,
        101.1, 99.6, 104.3, 100.4, 102.8, 98.6, 101.7, 103.1, 99.3, 102.6,
        100.2, 98.8, 104.6, 101.4, 99.7, 103.5, 100.8, 102.4, 98.2, 101.6,
        104.0, 99.9, 102.2, 100.5, 98.5, 103.8, 101.0, 99.4, 104.7, 100.1,
    ],
    dtype=float,
)

_R_Y = "c(" + ",".join(str(v) for v in Y) + ")"


def _r_eval(expr: str) -> np.ndarray:
    script = (
        "suppressMessages(devtools::load_all('.', quiet=TRUE));"
        f"cat(jsonlite::toJSON({expr}, digits=15))"
    )
    out = subprocess.check_output(
        ["Rscript", "--vanilla", "-e", script],
        text=True,
        cwd="/home/config/Misc/Python/Libraries/smooth",
    )
    return np.array(json.loads(out))


def test_sma_fixed_order3_fitted():
    """Fitted values from SMA(3) match R's sma(order=3)."""
    py = np.asarray(SMA(order=3, h=0).fit(Y).fitted, dtype=float)
    r = _r_eval(
        f"as.numeric(sma({_R_Y}, order=3, h=0, holdout=FALSE, silent=TRUE)$fitted)"
    )
    np.testing.assert_allclose(py, r, rtol=1e-5, atol=1e-6)


def test_sma_fixed_order5_fitted():
    """Fitted values from SMA(5) match R's sma(order=5)."""
    py = np.asarray(SMA(order=5, h=0).fit(Y).fitted, dtype=float)
    r = _r_eval(
        f"as.numeric(sma({_R_Y}, order=5, h=0, holdout=FALSE, silent=TRUE)$fitted)"
    )
    np.testing.assert_allclose(py, r, rtol=1e-5, atol=1e-6)


def test_sma_fixed_order4_forecast():
    """h-step-ahead point forecasts from SMA(4) match R."""
    py_fc = SMA(order=4, h=5).fit(Y).predict(h=5).mean.values
    r = _r_eval(
        f"as.numeric(sma({_R_Y}, order=4, h=5, holdout=FALSE, silent=TRUE)$forecast)"
    )
    np.testing.assert_allclose(py_fc, r, rtol=1e-5, atol=1e-6)
