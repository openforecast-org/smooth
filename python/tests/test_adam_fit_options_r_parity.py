"""R-Python parity of options of ADAM's fit: the selection of the ARIMA orders
by orders={"select": True}, as R's adam() hands it to auto.adam()."""

import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

from smooth import ADAM, MSARIMA
from tests._r_bridge import r_dict

pytestmark = pytest.mark.r_parity

Y = pd.read_csv(pathlib.Path(__file__).parent / "data" / "ces_airpassengers.csv")
Y = Y["y"].to_numpy(dtype=float)


@pytest.mark.parametrize("cls", [ADAM, MSARIMA])
def test_the_arima_orders_are_selected(cls):
    r = r_dict(
        "{ m <- adam(AirPassengers, 'NNN', lags=c(1,12),"
        " orders=list(ar=c(2,1), i=c(1,1), ma=c(2,1), select=TRUE));"
        " list(orders=m$orders, logLik=as.numeric(logLik(m))) }"
    )
    orders = {"ar": [2, 1], "i": [1, 1], "ma": [2, 1], "select": True}
    kwargs = dict(model="NNN") if cls is ADAM else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = cls(lags=[1, 12], orders=orders, **kwargs).fit(Y)
    for key in ("ar", "i", "ma"):
        assert list(model.orders[key]) == list(r["orders"][key])
    np.testing.assert_allclose(model.loglik, r["logLik"], rtol=1e-6)


# (R model, Python model): the pools whose model was reported as asked
POOLS = {
    "PPP": ("'PPP'", "PPP"),
    "FFF": ("'FFF'", "FFF"),
    "list": ("c('ANN','MAM')", ["ANN", "MAM"]),
    "CCN": ("'CCN'", "CCN"),
    "ZXZ": ("'ZXZ'", "ZXZ"),
}


@pytest.mark.parametrize("pool", list(POOLS))
def test_the_model_type_is_the_selected_one(pool):
    r_model, python_model = POOLS[pool]
    r = r_dict(
        f"{{ m <- adam(AirPassengers, {r_model}, lags=12);"
        " list(type=modelType(m)) }"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ADAM(model=python_model, lags=[12]).fit(Y)
    assert model.model_type == r["type"][0]


# (R call, Python arguments, with the regressor x1)
INITIALS = {
    "MAM": ("AirPassengers, 'MAM', lags=12", dict(model="MAM", lags=[12]), False),
    "ARIMA": (
        "AirPassengers, 'ANN', lags=c(1,12), initial='optimal',"
        " orders=list(ar=c(1,1), i=c(1,0), ma=c(0,0))",
        dict(
            model="ANN",
            lags=[1, 12],
            initial="optimal",
            orders={"ar": [1, 1], "i": [1, 0], "ma": [0, 0]},
        ),
        False,
    ),
    "ETSX": ("d, 'MNN', lags=1", dict(model="MNN", lags=[1]), True),
}


@pytest.mark.parametrize("case", list(INITIALS))
def test_the_initial_values_are_those_of_r(case):
    r_call, python_arguments, xreg = INITIALS[case]
    r = r_dict(
        "{ d <- data.frame(y=as.numeric(AirPassengers), x1=(0:143) %% 5);"
        f" m <- adam({r_call}); i <- m$initial;"
        " if(!is.null(i$xreg)) i$xreg <- as.list(i$xreg); list(initial=i) }"
    )
    X = pd.DataFrame({"x1": np.arange(144) % 5.0}) if xreg else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ADAM(**python_arguments).fit(Y, X)
    initial = model.initial_value
    assert list(initial) == list(r["initial"])
    for key, value in r["initial"].items():
        if key == "xreg":
            assert list(initial[key]) == list(value)
            np.testing.assert_allclose(
                list(initial[key].values()), np.ravel(list(value.values())), rtol=1e-5
            )
        else:
            np.testing.assert_allclose(np.ravel(initial[key]), value, rtol=1e-5)
