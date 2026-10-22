"""R-Python parity of regressors="select" in CES, OM, OMG and ADAM with an occurrence
model, on a series with one true regressor and two noise ones: CES selects them by
stepwise() on the errors of the model without them, as R's adam_xreg_selectRefit(),
OM and OMG by stepwise(plogis) on the occurrence indicator, as R's om(), ADAM by
stepwise() on the errors of its non-zero values, and each refits with the selected
ones."""

import warnings

import numpy as np
import pandas as pd
import pytest

from smooth import ADAM, CES, OM, OMG
from tests._r_bridge import r_dict

pytestmark = pytest.mark.r_parity

NAMES = ["a", "n1", "n2"]
DATA = (
    "set.seed(41); d <- data.frame(y=0, a=rnorm(300), n1=rnorm(300), n2=rnorm(300));"
    " d$y <- 100 + cumsum(rnorm(300)) + 3*d$a + rnorm(300);"
    " d$o <- rbinom(300, 1, plogis(2*d$a)); d$i <- d$o * (10 + 3*d$a + rnorm(300));"
    " dc <- d[, c('y', 'a', 'n1', 'n2')]; do <- d[, c('o', 'a', 'n1', 'n2')];"
    " di <- d[, c('i', 'a', 'n1', 'n2')];"
)
CASES = {
    "CES": (
        "ces(dc[1:280,], regressors='select')",
        lambda: CES(regressors="select"),
        "y",
    ),
    "OM": (
        "om(do[1:280,], 'ANN', occurrence='odds-ratio', regressors='select')",
        lambda: OM(model="ANN", occurrence="odds-ratio", regressors="select"),
        "o",
    ),
    "OMG": (
        "omg(do[1:280,], 'ANN', regressorsA='select', regressorsB='use')",
        lambda: OMG(model_a="ANN", regressors_a="select", regressors_b="use"),
        "o",
    ),
    "ADAM": (
        "adam(di[1:280,], 'ANN', occurrence='auto', regressors='select')",
        lambda: ADAM(model="ANN", occurrence="auto", regressors="select"),
        "i",
    ),
}


def _selected(model):
    columns = model._explanatory.get("xreg_columns")
    if columns is None:
        return list(model._explanatory.get("xreg_names") or [])
    return [NAMES[i] for i in columns]


@pytest.mark.parametrize("case", list(CASES))
def test_the_selection_agrees(case):
    r_call, model, response = CASES[case]
    r = r_dict(
        f"{{ {DATA} m <- {r_call}; mA <- if(is.omg(m)) m$modelA else m;"
        " list(y=d[[" + repr(response) + "]], a=d$a, n1=d$n1, n2=d$n2,"
        " names=setdiff(names(coef(mA)), c('alpha', 'alpha_0', 'alpha_1')),"
        " namesB=if(is.omg(m)) names(coef(m$modelB)) else list(),"
        " logLik=as.numeric(logLik(m)),"
        " forecast=as.numeric(forecast(m, h=20, newdata=d[281:300,])$mean)) }"
    )
    y = np.asarray(r["y"], dtype=float)
    X = pd.DataFrame({name: r[name] for name in NAMES})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = model().fit(y[:280], X.iloc[:280])
    side = fit.model_a if case == "OMG" else fit
    assert _selected(side) == list(r["names"])
    if case == "OMG":
        assert len(r["namesB"]) == 1 + len(NAMES)
        assert list(side._explanatory["xreg_names"]) == list(r["names"])
    # The forecasts take all the regressors given to fit()
    forecast = np.asarray(fit.predict(h=20, X=X.iloc[280:]).mean, dtype=float)
    assert np.all(np.isfinite(forecast))
    assert fit.loglik == pytest.approx(r["logLik"][0], rel=1e-6)
    np.testing.assert_allclose(forecast, r["forecast"], rtol=1e-5)
