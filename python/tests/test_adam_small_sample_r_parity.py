"""R-Python parity of ADAM on small samples: the sample-size check of R's adam()
(R/adamGeneral.R) simplifies the model, restricts the pool or fixes the parameters
as the sample shrinks, and Python's checker/sample_size.py does the same."""

import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

from smooth import ADAM, OM
from tests._r_bridge import r_dict

pytestmark = pytest.mark.r_parity

Y = pd.read_csv(
    pathlib.Path(__file__).parent / "data" / "ces_airpassengers.csv",
    float_precision="round_trip",
)
Y = Y["y"].to_numpy(dtype=float)
X = np.array([[1.0], [3.0], [2.0], [5.0]])
INTERMITTENT = np.array([0, 0, 5, 0, 7, 0, 0, 0], dtype=float)

# (R call, Python arguments, observations, with the regressor x); zero observations
# stand for the intermittent series
CASES = {
    "ETS+ARIMA": (
        "'AAN', orders=list(ar=2,i=1,ma=2), initial='optimal'",
        dict(model="AAN", orders={"ar": [2], "i": [1], "ma": [2]}, initial="optimal"),
        10,
        False,
    ),
    "ARIMA backcasting": (
        "'NNN', orders=list(ar=2,i=1,ma=2), initial='optimal'",
        dict(model="NNN", orders={"ar": [2], "i": [1], "ma": [2]}, initial="optimal"),
        7,
        False,
    ),
    "ARIMA(0,1,1)": (
        "'NNN', orders=list(ar=3,i=2,ma=3), initial='optimal'",
        dict(model="NNN", orders={"ar": [3], "i": [2], "ma": [3]}, initial="optimal"),
        5,
        False,
    ),
    "pool": ("'ZZZ', initial='optimal'", dict(model="ZZZ", initial="optimal"), 6, False),
    "damped": ("'AAdN', initial='optimal'", dict(model="AAdN", initial="optimal"), 6, False),
    "trend": ("'AAdN', initial='optimal'", dict(model="AAdN", initial="optimal"), 5, False),
    "seasonal": (
        "'AAA', lags=12, initial='optimal'",
        dict(model="AAA", lags=[12], initial="optimal"),
        14,
        False,
    ),
    "seasonal 18": (
        "'AAA', lags=12, initial='optimal'",
        dict(model="AAA", lags=[12], initial="optimal"),
        18,
        False,
    ),
    "four with x": ("'AAN', initial='optimal'", dict(model="AAN", initial="optimal"), 4, True),
    "three": ("'ANN', initial='optimal'", dict(model="ANN", initial="optimal"), 3, False),
    "two": ("'ANN'", dict(model="ANN"), 2, False),
    "one": ("'ANN'", dict(model="ANN"), 1, False),
    "intermittent": (
        "'ANN', occurrence='fixed'",
        dict(model="ANN", occurrence="fixed"),
        0,
        False,
    ),
}


@pytest.mark.parametrize("case", list(CASES))
def test_the_small_sample_model_is_that_of_r(case):
    r_call, python_arguments, obs, xreg = CASES[case]
    data = (
        "c(0,0,5,0,7,0,0,0)"
        if obs == 0
        else f"data.frame(y=AirPassengers[1:{obs}], x=c(1,3,2,5))"
        if xreg
        else f"AirPassengers[1:{obs}]"
    )
    r = r_dict(
        f"{{ m <- suppressWarnings(adam({data}, {r_call}));"
        " list(name=m$model, type=modelType(m), nparam=nparam(m),"
        " logLik=as.numeric(logLik(m)), initialType=m$initialType,"
        " orders=orders(m)[c('ar','i','ma')]) }"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ADAM(**python_arguments).fit(
            INTERMITTENT if obs == 0 else Y[:obs], X=X if xreg else None
        )
    assert model.model_name == r["name"][0]
    assert model.model_type == r["type"][0]
    assert model.initial_type == r["initialType"][0]
    assert model.nparam == r["nparam"][0]
    for key in ("ar", "i", "ma"):
        assert list(model.orders[key]) == list(r["orders"][key])
    np.testing.assert_allclose(model.loglik, float(r["logLik"][0]), rtol=1e-6)


@pytest.mark.parametrize(
    "occurrence", ["fixed", "odds-ratio", "inverse-odds-ratio", "direct", "general"]
)
def test_the_occurrence_of_two_demands_is_that_of_r(occurrence):
    r = r_dict(
        f"{{ m <- suppressWarnings(om(c(0,0,5,0,7,0,0,0), occurrence='{occurrence}'));"
        " list(logLik=as.numeric(logLik(m)), fitted=as.numeric(fitted(m))) }"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = OM(occurrence=occurrence).fit(INTERMITTENT)
    np.testing.assert_allclose(model.loglik, r["logLik"][0], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(model.fitted), r["fitted"], rtol=1e-6)
