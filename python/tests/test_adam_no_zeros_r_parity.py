"""R-Python parity of adam() on data without zeros: R switches the occurrence model
off (R/adamGeneral.R, otLogical), and a zero scale of the Gamma gives a NaN density,
which the cost function turns into the 1e+300 penalty instead of an error."""

import warnings

import numpy as np
import pandas as pd
import pytest

from smooth import ADAM
from tests._r_bridge import r_dict

pytestmark = pytest.mark.r_parity

OCCURRENCES = ["auto", "fixed", "general", "odds-ratio", "inverse-odds-ratio", "direct"]

SERIES = {
    "trend": "y <- 1:10;",
    "noise": "set.seed(41); y <- round(rnorm(40, 100, 10), 2);",
    "missing": "set.seed(41); y <- round(rnorm(40, 100, 10), 2); y[c(7, 20)] <- NA;",
}


def _fit(y, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ADAM(**kwargs).fit(y)


@pytest.mark.parametrize("series", list(SERIES))
@pytest.mark.parametrize("occurrence", OCCURRENCES)
def test_the_occurrence_is_switched_off_without_zeros(series, occurrence):
    r = r_dict(
        f"{{ {SERIES[series]} m <- suppressWarnings(adam(y, 'ZZN',"
        f" occurrence='{occurrence}'));"
        " list(y=as.numeric(y), model=modelType(m), B=unname(m$B),"
        " logLik=as.numeric(logLik(m)), occurrence=if(is.occurrence(m$occurrence))"
        " m$occurrence$occurrence else 'none') }"
    )
    y = pd.to_numeric(pd.Series(r["y"]), errors="coerce").to_numpy()
    model = _fit(y, model="ZZN", occurrence=occurrence)
    # None at all, or that of the missing values alone (R's "provided")
    assert np.atleast_1d(r["occurrence"])[0] in ("none", "provided")
    assert model._occurrence["occurrence"] in ("none", "provided")
    assert model.model_name == f"ETS({np.atleast_1d(r['model'])[0]})"
    assert np.allclose(model.b_value, r["B"], rtol=1e-6)
    assert model.loglik == pytest.approx(float(np.atleast_1d(r["logLik"])[0]), rel=1e-8)


def test_a_zero_scale_gives_the_penalty():
    r = r_dict(
        "{ m <- suppressWarnings(adam(10:1, 'MAN', persistence=c(1, 0.8535)));"
        " list(logLik=as.numeric(logLik(m)), scale=m$scale) }"
    )
    model = _fit(np.arange(10.0, 0, -1), model="MAN", persistence=[1, 0.8535])
    assert np.atleast_1d(r["scale"])[0] == 0
    assert model.loglik == float(np.atleast_1d(r["logLik"])[0]) == -1e300
