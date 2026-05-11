"""Cross-language parity tests: Python ALM pure-regression path vs R smooth.

These tests verify that Python's ADAM/OM/OMG with model='NNN' and external
regressors produces results numerically identical to R's smooth::adam() /
om() / omg() which delegate to greybox::alm().

The reference data is hardcoded (no random seeds) so R and Python always
operate on identical inputs.

Run R-dependent tests with:  pytest -m r_parity
Skip them (Python-only CI) with:  pytest -m "not r_parity"
"""

from __future__ import annotations

import json
import subprocess

import numpy as np
import pytest

from smooth import ADAM, OM

pytestmark = pytest.mark.r_parity

# Hardcoded 20-obs deterministic dataset — same values used in both languages.
Y = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0], dtype=float)
X = np.arange(20, dtype=float).reshape(20, 1)

_R_Y = "(1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0)"
_R_X = "0:19"


def _r_eval(expr: str) -> np.ndarray:
    """Run a one-liner R expression that prints a JSON array; return as ndarray."""
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


def test_adam_pure_regression_coef():
    py = ADAM(model="NNN").fit(Y, X).coef
    r = _r_eval(
        f"adam(cbind(y=c{_R_Y},x={_R_X}),model='NNN',regressors='use',"
        "distribution='dnorm',silent=TRUE)$B"
    )
    np.testing.assert_allclose(py, r, rtol=1e-6, atol=1e-8)


def test_adam_pure_regression_fitted():
    py = ADAM(model="NNN").fit(Y, X).fitted
    r = _r_eval(
        f"as.numeric(adam(cbind(y=c{_R_Y},x={_R_X}),model='NNN',regressors='use',"
        "distribution='dnorm',silent=TRUE)$fitted)"
    )
    np.testing.assert_allclose(py, r, rtol=1e-6, atol=1e-8)


def test_om_pure_regression_coef():
    py = OM(model="NNN", occurrence="odds-ratio").fit(Y, X).coef
    r = _r_eval(
        f"om(cbind(y=c{_R_Y},x={_R_X}),model='NNN',occurrence='odds-ratio',"
        "regressors='use',silent=TRUE)$B"
    )
    np.testing.assert_allclose(py, r, rtol=1e-6, atol=1e-8)


def test_om_pure_regression_fitted():
    py = OM(model="NNN", occurrence="odds-ratio").fit(Y, X).fitted
    r = _r_eval(
        f"as.numeric(om(cbind(y=c{_R_Y},x={_R_X}),model='NNN',occurrence='odds-ratio',"
        "regressors='use',silent=TRUE)$fitted)"
    )
    np.testing.assert_allclose(py, r, rtol=1e-6, atol=1e-8)
