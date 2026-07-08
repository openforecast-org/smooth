"""R-Python parity tests for msdecompose's OLS-driven initial estimation.

Both languages now share the C++ olsCore (pivoted QR with rank cutoff) for
the global smoother and NA imputation paths. The remaining R-Python
discrepancy is bounded by:

* the LINPACK -> LAPACK shift in R (pivoted QR is now LAPACK geqp3 instead
  of LINPACK dqrls — a few ULPs);
* downstream summation-order differences (R ``mean(diff(...))`` vs NumPy
  ``np.nanmean(np.diff(...))``);
* design-matrix construction (R ``poly()`` vs Python ``np.vander``) for the
  NA-imputation path. These span the same column space, so the OLS fitted
  values are within ULPs in well-conditioned cases but can diverge more on
  ill-conditioned raw Vandermonde input.

So we use a tighter bound on the global smoother and a looser one on NA
imputation. Both bounds are still far below the pre-shared-OLS gap.
"""

from __future__ import annotations

import numpy as np
import pytest

from smooth.adam_general.core.utils.utils import msdecompose
from tests._r_bridge import r_dict

pytestmark = pytest.mark.r_parity


AIR_PASSENGERS_EXPR = "as.numeric(AirPassengers)"


def _r_msdecompose(y_values, smoother):
    tokens = ("NA_real_" if np.isnan(v) else repr(float(v)) for v in y_values)
    expr = (
        f"{{ y <- c({','.join(tokens)}); "
        f"d <- msdecompose(y, lags=c(12), type='additive', smoother='{smoother}'); "
        "list(level=unname(d$initial$nonseasonal[1]),"
        "     trend=unname(d$initial$nonseasonal[2]),"
        "     seasonal=unname(d$initial$seasonal[[1]])) }"
    )
    return r_dict(expr)


def _air_passengers():
    return np.asarray(r_dict(f"list(y={AIR_PASSENGERS_EXPR})")["y"], dtype=float)


@pytest.mark.parametrize("smoother", ["lowess", "global"])
def test_msdecompose_clean_initials(smoother):
    y = _air_passengers()
    py = msdecompose(y, lags=[12], type="additive", smoother=smoother)
    r = _r_msdecompose(y, smoother)

    # Global smoother is the user-reported case; expect ULP-level parity.
    # Lowess goes through a different smoother but still shares the OLS
    # implementation in the NA-imputation branch (not exercised here).
    rtol = 1e-13 if smoother == "global" else 1e-10
    np.testing.assert_allclose(
        py["initial"]["nonseasonal"]["level"], r["level"], rtol=rtol,
        err_msg=f"level mismatch under smoother={smoother}",
    )
    np.testing.assert_allclose(
        py["initial"]["nonseasonal"]["trend"], r["trend"], rtol=rtol,
        err_msg=f"trend mismatch under smoother={smoother}",
    )
    np.testing.assert_allclose(
        np.asarray(py["initial"]["seasonal"][0]),
        np.asarray(r["seasonal"]),
        rtol=rtol, atol=1e-12,
        err_msg=f"seasonal pattern mismatch under smoother={smoother}",
    )


def test_msdecompose_na_imputation():
    """NAs force msdecompose to go through olsCore's NA-imputation path."""
    y = _air_passengers().copy()
    y[[19, 39, 59]] = np.nan

    py = msdecompose(y, lags=[12], type="additive", smoother="lowess")
    r = _r_msdecompose(y, "lowess")

    # The raw Vandermonde basis (Python) vs orthogonal poly basis (R) span
    # the same column space, but their QR rounding paths differ, so we
    # don't expect ULP-level parity here -- only "close enough that the
    # ADAM initialiser sees consistent state seeds".
    np.testing.assert_allclose(
        py["initial"]["nonseasonal"]["level"], r["level"], rtol=1e-11,
    )
    np.testing.assert_allclose(
        py["initial"]["nonseasonal"]["trend"], r["trend"], rtol=1e-11,
    )
