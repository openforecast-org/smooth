"""The OPG covariance of ADAM against R's vcov(type="opg").

Both differentiate the per-observation log-likelihood at the estimates, the
distribution, its shape and the constant included, so the standard errors agree
to the last digits. Skipped by default (``r_parity`` marker).
"""

import numpy as np
import pytest

from smooth import ADAM

from ._r_bridge import r_dict

pytestmark = pytest.mark.r_parity

CASES = {
    "dlaplace": (
        "AirPassengers",
        "'MMM', lags=c(1,12), distribution='dlaplace'",
        {"model": "MMM", "lags": [1, 12], "distribution": "dlaplace"},
    ),
    "dgnorm shape": (
        "AirPassengers",
        "'AAN', distribution='dgnorm'",
        {"model": "AAN", "distribution": "dgnorm"},
    ),
    "ARIMA drift": (
        "BJsales",
        "'NNN', orders=list(ar=1,i=1,ma=1), constant=TRUE",
        {"model": "NNN", "orders": {"ar": 1, "i": 1, "ma": 1}, "constant": True},
    ),
    "optimal": (
        "AirPassengers",
        "'MMM', lags=12, initial='optimal'",
        {"model": "MMM", "lags": [12], "initial": "optimal"},
    ),
    "ets adam": ("AirPassengers", "'MAN', ets='adam'", {"model": "MAN", "ets": "adam"}),
}


@pytest.mark.parametrize("case", list(CASES))
def test_the_opg_covariance_agrees(case):
    data, r_args, py_args = CASES[case]
    r = r_dict(
        f"{{ m <- adam({data}, {r_args}); list(y=as.numeric({data}),"
        " B=unname(coef(m)), vcov=as.vector(vcov(m, type='opg'))) }"
    )
    fit = ADAM(**py_args).fit(np.asarray(r["y"], dtype=float))
    np.testing.assert_allclose(fit.coef, r["B"], rtol=1e-8, atol=1e-10)
    covariance = np.asarray(fit.vcov(type="opg"), dtype=float)
    # Near-zero covariances are compared against the scale of the matrix
    scale = np.max(np.abs(r["vcov"]))
    np.testing.assert_allclose(
        covariance.ravel(order="F"), r["vcov"], rtol=1e-8, atol=1e-10 * scale
    )
