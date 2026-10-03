"""R-Python parity of ADAM's regressors="select": the regressors are selected by
stepwise() on the errors of the model estimated without them, which is then
estimated with them from its parameters, as R's adam()."""

import warnings

import numpy as np
import pandas as pd
import pytest

from smooth import ADAM
from tests._r_bridge import r_dict

pytestmark = pytest.mark.r_parity

# The regressors of etsx_data.csv and three of noise
DATA = (
    "d <- read.csv('python/tests/data/etsx_data.csv'); set.seed(7);"
    " d$n1 <- rnorm(120); d$n2 <- rnorm(120); d$n3 <- rnorm(120); d$y <- d$y + 20;"
)

# (R arguments, Python arguments)
CASES = {
    "ANN": ("'ANN'", dict(model="ANN")),
    "ANN-optimal": ("'ANN', initial='optimal'", dict(model="ANN", initial="optimal")),
    "AAdN": ("'AAdN'", dict(model="AAdN")),
    "MNN": ("'MNN'", dict(model="MNN")),
    "ZZN": ("'ZZN'", dict(model="ZZN")),
    "two-stage": ("'AAN', initial='two-stage'", dict(model="AAN", initial="two-stage")),
    "dgnorm": (
        "'ANN', distribution='dgnorm'",
        dict(model="ANN", distribution="dgnorm"),
    ),
    "MSE": ("'ANN', loss='MSE'", dict(model="ANN", loss="MSE")),
    "ARIMAX": (
        "'NNN', orders=list(ar=1, i=0, ma=1)",
        dict(model="NNN", orders={"ar": [1], "i": [0], "ma": [1]}),
    ),
}


@pytest.mark.parametrize("case", list(CASES))
def test_the_selection_agrees(case):
    r_arguments, python_arguments = CASES[case]
    r = r_dict(
        f"{{ {DATA} m <- suppressWarnings(adam(d[1:108,], {r_arguments},"
        " regressors='select'));"
        " f <- forecast(m, h=12, newdata=d[109:120,], interval='prediction');"
        " list(y=d$y, X=as.matrix(d[,-1]), names=colnames(d)[-1],"
        " xreg=names(m$initial$xreg), B=unname(m$B), logLik=as.numeric(logLik(m)),"
        " mean=as.numeric(f$mean), lower=as.numeric(f$lower)) }"
    )
    y = np.asarray(r["y"], dtype=float)
    X = pd.DataFrame(np.asarray(r["X"], dtype=float), columns=r["names"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ADAM(regressors="select", **python_arguments).fit(y[:108], X[:108])
        # The forecasts take all the regressors given to fit()
        forecast = model.predict(h=12, X=X[108:].to_numpy(), interval="prediction")
    assert model._explanatory["xreg_names"] == r["xreg"]
    np.testing.assert_allclose(model.coef, r["B"], rtol=1e-6, atol=1e-8)
    assert model.loglik == pytest.approx(r["logLik"][0], rel=1e-10)
    np.testing.assert_allclose(np.asarray(forecast.mean), r["mean"], rtol=1e-10)
    np.testing.assert_allclose(np.ravel(forecast.lower), r["lower"], rtol=1e-9)
