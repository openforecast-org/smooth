"""R-Python parity of ADAM's branch-and-bound selection of the ETS model: the
same pool of models is estimated and the same model is selected as by R's adam(),
for the Z, X and Y placeholders."""

import warnings

import numpy as np
import pytest

from smooth import ADAM
from tests._r_bridge import r_dict

pytestmark = pytest.mark.r_parity

SERIES = {"AirPassengers": 12, "UKgas": 4, "Nile": 1}
MODELS = ["ZZZ", "YYY", "XXX", "ZXZ", "ZZN", "ZYM"]


@pytest.mark.parametrize("series", list(SERIES))
@pytest.mark.parametrize("model", MODELS)
def test_the_selection_agrees(series, model):
    lag = SERIES[series]
    r = r_dict(
        f"{{ m <- suppressWarnings(adam({series}, '{model}', lags={lag}));"
        f" list(y=as.numeric({series}), model=modelType(m), pool=names(m$ICs),"
        " ICs=unname(m$ICs)) }"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ADAM(model=model, lags=[lag]).fit(np.asarray(r["y"], dtype=float))
    pool = fit._adam_selected["ic_selection"]
    assert fit.model_name == f"ETS({np.atleast_1d(r['model'])[0]})"
    assert list(pool) == list(np.atleast_1d(r["pool"]))
    assert np.allclose(list(pool.values()), np.atleast_1d(r["ICs"]), rtol=1e-6)
