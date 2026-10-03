"""R-Python parity of the point forecasts (R's and Python's ``point``) where they
are deterministic: the skeleton, the mean where it is the skeleton, and the median
from the analytical intervals."""

import warnings

import numpy as np
import pytest

from smooth import ADAM
from tests._r_bridge import r_dict

pytestmark = pytest.mark.r_parity

# (data, model, distribution, lags, horizon): up to the first seasonal lag the mean of
# ETS(M,A,M) is its skeleton
CASES = [
    ("BJsales", "ANN", "dnorm", [1], 12),
    ("BJsales", "MNN", "dgamma", [1], 12),
    ("BJsales", "MNN", "dlnorm", [1], 12),
    ("AirPassengers", "MAM", "dgamma", [12], 12),
]


@pytest.mark.parametrize("data, model, distribution, lags, h", CASES)
def test_the_point_forecasts_agree(data, model, distribution, lags, h):
    r = r_dict(
        f"{{ m <- adam({data}, '{model}', distribution='{distribution}');"
        f" f <- function(point) as.numeric(forecast(m, h={h}, point=point)$mean);"
        f" list(y=as.numeric({data}), skeleton=f('skeleton'), mean=f('mean'),"
        " median=f('median')) }"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ADAM(model=model, distribution=distribution, lags=lags).fit(
            np.asarray(r["y"], dtype=float)
        )
    for point in ("skeleton", "mean", "median"):
        np.testing.assert_allclose(
            fit.predict(h=h, point=point).mean, r[point], rtol=1e-10
        )
