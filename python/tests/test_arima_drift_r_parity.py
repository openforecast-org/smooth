"""ARIMA with a drift against R's adam(): the drift starts and is bounded on the
series differenced as the model differences it, and a multiplicative drift is a
ratio, inverted by backcasting and entering the ARIMA states in logs. Skipped by
default (``r_parity`` marker)."""

import numpy as np
import pytest

from smooth import ADAM

from ._r_bridge import r_dict

pytestmark = pytest.mark.r_parity

SEASONAL = {"ar": [0, 0], "i": [0, 1], "ma": [0, 0]}
R_SEASONAL = "lags=c(1,12), orders=list(ar=c(0,0), i=c(0,1), ma=c(0,0))"
AIR = "as.numeric(AirPassengers)"

CASES = {
    "seasonal dlnorm": (
        AIR,
        f"'NNN', {R_SEASONAL}, constant=TRUE, distribution='dlnorm'",
        {"model": "NNN", "lags": [1, 12], "orders": SEASONAL, "distribution": "dlnorm"},
    ),
    "airline": (
        AIR,
        "'NNN', lags=c(1,12), orders=list(ar=c(0,0), i=c(1,1), ma=c(1,1)),"
        " constant=TRUE",
        {
            "model": "NNN",
            "lags": [1, 12],
            "orders": {"ar": [0, 0], "i": [1, 1], "ma": [1, 1]},
        },
    ),
    "seasonal AR dgamma": (
        AIR,
        "'NNN', lags=c(1,12), orders=list(ar=c(1,0), i=c(0,1), ma=c(0,0)),"
        " constant=TRUE, distribution='dgamma'",
        {
            "model": "NNN",
            "lags": [1, 12],
            "orders": {"ar": [1, 0], "i": [0, 1], "ma": [0, 0]},
            "distribution": "dgamma",
        },
    ),
    "ETS(M,M,N) with seasonal ARIMA": (
        AIR,
        f"'MMN', {R_SEASONAL}, constant=TRUE",
        {"model": "MMN", "lags": [1, 12], "orders": SEASONAL},
    ),
    "BJsales ARIMA(1,1,1)": (
        "as.numeric(BJsales)",
        "'NNN', orders=list(ar=1, i=1, ma=1), constant=TRUE",
        {"model": "NNN", "orders": {"ar": 1, "i": 1, "ma": 1}},
    ),
}


@pytest.mark.parametrize("case", list(CASES))
def test_the_drift_agrees(case):
    data, r_args, py_args = CASES[case]
    r = r_dict(
        f"{{ y <- {data}; m <- adam(y, {r_args});"
        " list(y=y, logLik=as.numeric(logLik(m)), B=unname(m$B)) }"
    )
    fit = ADAM(constant=True, **py_args).fit(np.asarray(r["y"], dtype=float))
    np.testing.assert_allclose(fit.coef, r["B"], rtol=1e-8, atol=1e-10)
    assert fit.loglik == pytest.approx(r["logLik"][0], rel=1e-10)
