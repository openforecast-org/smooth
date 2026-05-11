"""Cost function for general occurrence (OMG) models.

Mirrors ``omgCF_local`` in ``R/omg.R``: fills two parallel sets of state-space
matrices from a joint parameter vector ``B = [B_A | B_B]``, runs the C++
``adamCore.omfitGeneral`` to advance both sub-models simultaneously, applies
``omg_link_function`` to combine the two raw fitted vectors into a probability,
and returns the negative Bernoulli log-likelihood.

The single-model OM cost lives in :mod:`om_cost`; this module is the
two-model analogue.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import eigvals

from smooth.adam_general.core.creator import filler


def omg_link_function(fitted_a, fitted_b, error_type_a, error_type_b):
    """Translation of ``omgLinkFunction`` (R/omg.R:1).

    Combines the raw (state-space) fitted output of model A and model B into a
    probability.  All four branches are numerically stable reformulations of
    ``aFit / (aFit + bFit)`` that avoid exp-overflow by dividing through by the
    larger of the two exponentials before returning.

    A+A: 1/(1+exp(fb-fa))          M+M: 1/(1+fb/fa)
    M+A: 1/(1+exp(fb-log(fa)))     A+M: 1/(1+exp(log(fb)-fa))
    """
    fa = np.asarray(fitted_a, dtype=np.float64)
    fb = np.asarray(fitted_b, dtype=np.float64)
    if error_type_a == "A" and error_type_b == "A":
        return 1.0 / (1.0 + np.exp(fb - fa))
    if error_type_a == "M" and error_type_b == "M":
        return 1.0 / (1.0 + fb / fa)
    if error_type_a == "M" and error_type_b == "A":
        return 1.0 / (1.0 + np.exp(fb - np.log(fa)))
    # error_type_a == "A", error_type_b == "M"
    return 1.0 / (1.0 + np.exp(np.log(fb) - fa))


def _ets_bounds_check(model_type_dict, components_dict, vec_g, mat_f, phi_dict):
    """Mirror of the ETS "usual" bounds branch in ``CF`` and ``om_cf``."""
    if not model_type_dict["ets_model"]:
        return 0.0
    n_ets = components_dict["components_number_ets"]
    if any(vec_g[:n_ets] > 1) or any(vec_g[:n_ets] < 0):
        return 1e300
    if model_type_dict["model_is_trendy"]:
        if vec_g[1] > vec_g[0]:
            return 1e300
        n_ns = components_dict["components_number_ets_non_seasonal"]
        n_seas = components_dict["components_number_ets_seasonal"]
        if model_type_dict["model_is_seasonal"] and any(
            vec_g[n_ns : n_ns + n_seas] > (1 - vec_g[0])
        ):
            return 1e300
    elif model_type_dict["model_is_seasonal"]:
        n_ns = components_dict["components_number_ets_non_seasonal"]
        n_seas = components_dict["components_number_ets_seasonal"]
        if any(vec_g[n_ns : n_ns + n_seas] > (1 - vec_g[0])):
            return 1e300
    if phi_dict["phi_estimate"] and (mat_f[1, 1] > 1 or mat_f[1, 1] < 0):
        return 1e300
    return 0.0


def _arima_bounds_check(arima_checked, arima_polynomials, ar_pm, ma_pm):
    """Mirror of the ARIMA "usual" bounds branch in ``CF`` and ``om_cf``."""
    if not arima_checked["arima_model"]:
        return 0.0
    if not (arima_checked["ar_estimate"] or arima_checked["ma_estimate"]):
        return 0.0
    if (
        arima_checked["ar_estimate"]
        and np.all(-arima_polynomials["arPolynomial"][1:] > 0)
        and sum(-arima_polynomials["arPolynomial"][1:]) >= 1
    ):
        ar_pm[:, 0] = -arima_polynomials["arPolynomial"][1:]
        roots = np.abs(eigvals(ar_pm))
        if any(roots > 1):
            return 1e100 * max(roots)
    if arima_checked["ma_estimate"] and sum(arima_polynomials["maPolynomial"][1:]) >= 1:
        ma_pm[:, 0] = arima_polynomials["maPolynomial"][1:]
        roots = np.abs(eigvals(ma_pm))
        if any(roots > 1):
            return 1e100 * max(abs(roots))
    return 0.0


def omg_cf(  # noqa: N802
    B,
    *,
    side_a,
    side_b,
    n_params_a,
    observations_dict,
    bounds,
    adam_ets: bool = False,
):
    """OMG cost function — joint Bernoulli likelihood on combined probability.

    ``side_a`` and ``side_b`` are dicts collecting everything ``filler`` and
    ``adam_cpp.omfitGeneral`` need for the two sub-models. ``n_params_a``
    splits the concatenated parameter vector. Mirrors R/omg.R:omgCF_local.
    """
    B_A = B[:n_params_a]
    B_B = B[n_params_a:]

    elem_a = filler(
        B_A,
        model_type_dict=side_a["model_type_dict"],
        components_dict=side_a["components_dict"],
        lags_dict=side_a["lags_dict"],
        matrices_dict=side_a["matrices_dict"],
        persistence_checked=side_a["persistence"],
        initials_checked=side_a["initials"],
        arima_checked=side_a["arima"],
        explanatory_checked=side_a["explanatory"],
        phi_dict=side_a["phi"],
        constants_checked=side_a["constant"],
        adam_cpp=side_a["adam_cpp"],
    )
    elem_b = filler(
        B_B,
        model_type_dict=side_b["model_type_dict"],
        components_dict=side_b["components_dict"],
        lags_dict=side_b["lags_dict"],
        matrices_dict=side_b["matrices_dict"],
        persistence_checked=side_b["persistence"],
        initials_checked=side_b["initials"],
        arima_checked=side_b["arima"],
        explanatory_checked=side_b["explanatory"],
        phi_dict=side_b["phi"],
        constants_checked=side_b["constant"],
        adam_cpp=side_b["adam_cpp"],
    )

    if bounds == "usual":
        penalty_a = _ets_bounds_check(
            side_a["model_type_dict"],
            side_a["components_dict"],
            elem_a["vec_g"],
            elem_a["mat_f"],
            side_a["phi"],
        )
        if penalty_a > 0:
            return float(penalty_a)
        penalty_a = _arima_bounds_check(
            side_a["arima"],
            elem_a.get("arima_polynomials", {}),
            side_a.get("ar_polynomial_matrix"),
            side_a.get("ma_polynomial_matrix"),
        )
        if penalty_a > 0:
            return float(penalty_a)

        penalty_b = _ets_bounds_check(
            side_b["model_type_dict"],
            side_b["components_dict"],
            elem_b["vec_g"],
            elem_b["mat_f"],
            side_b["phi"],
        )
        if penalty_b > 0:
            return float(penalty_b)
        penalty_b = _arima_bounds_check(
            side_b["arima"],
            elem_b.get("arima_polynomials", {}),
            side_b.get("ar_polynomial_matrix"),
            side_b.get("ma_polynomial_matrix"),
        )
        if penalty_b > 0:
            return float(penalty_b)

    # Refresh the profile seed from the freshly-filled mat_vt
    side_a["profile"]["profiles_recent_table"][:] = elem_a["mat_vt"][
        :, : side_a["lags_dict"]["lags_model_max"]
    ]
    side_b["profile"]["profiles_recent_table"][:] = elem_b["mat_vt"][
        :, : side_b["lags_dict"]["lags_model_max"]
    ]

    ot = np.asarray(observations_dict["ot"], dtype=np.float64)

    # Build Fortran-ordered copies for the C++ call
    def _f(x, dtype=np.float64):
        return np.asfortranarray(x, dtype=dtype)

    initials_a = side_a["initials"]
    if isinstance(initials_a["initial_type"], list):
        backcast = any(
            t in ("complete", "backcasting") for t in initials_a["initial_type"]
        )
    else:
        backcast = initials_a["initial_type"] in ("complete", "backcasting")

    res = side_a["adam_cpp"].omfitGeneral(
        matrixVtA=_f(elem_a["mat_vt"]),
        matrixWtA=_f(elem_a["mat_wt"]),
        matrixFA=_f(elem_a["mat_f"]),
        vectorGA=_f(elem_a["vec_g"]),
        indexLookupTableA=_f(side_a["profile"]["index_lookup_table"], np.uint64),
        profilesRecentA=_f(side_a["profile"]["profiles_recent_table"]),
        EB=side_b["model_type_dict"]["error_type"],
        TB=side_b["model_type_dict"]["trend_type"],
        SB=side_b["model_type_dict"]["season_type"],
        nNonSeasonalB=int(
            side_b["components_dict"]["components_number_ets_non_seasonal"]
        ),
        nSeasonalB=int(side_b["components_dict"]["components_number_ets_seasonal"]),
        nETSB=int(side_b["components_dict"]["components_number_ets"]),
        nArimaB=int(side_b["components_dict"].get("components_number_arima", 0)),
        nXregB=int(side_b["explanatory"].get("xreg_number", 0)),
        nComponentsB=int(side_b["components_dict"]["components_number_all"]),
        constantB=bool(side_b["constant"].get("constant_required", False)),
        adamETSB=adam_ets,
        matrixVtB=_f(elem_b["mat_vt"]),
        matrixWtB=_f(elem_b["mat_wt"]),
        matrixFB=_f(elem_b["mat_f"]),
        vectorGB=_f(elem_b["vec_g"]),
        indexLookupTableB=_f(side_b["profile"]["index_lookup_table"], np.uint64),
        profilesRecentB=_f(side_b["profile"]["profiles_recent_table"]),
        vectorOt=ot,
        backcast=backcast,
        nIterations=int(initials_a["n_iterations"]),
        refineHead=True,
    )

    e_a = side_a["model_type_dict"]["error_type"]
    e_b = side_b["model_type_dict"]["error_type"]
    p_combined = omg_link_function(
        np.asarray(res.fittedA).ravel(),
        np.asarray(res.fittedB).ravel(),
        e_a,
        e_b,
    )

    # Infeasibility guard, mirroring R/omg.R:336 (NaN or boundary p means
    # the parameters are inconsistent with the data — uniform large penalty
    # so the optimiser steers away). NOT a clip on the model output.
    if (
        np.any(np.isnan(p_combined))
        or np.any(p_combined <= 0)
        or np.any(p_combined >= 1)
    ):
        return 1e300

    ot_logical = observations_dict["ot_logical"]
    cf_value = -(
        np.sum(np.log(p_combined[ot_logical]))
        + np.sum(np.log(1.0 - p_combined[~ot_logical]))
    )
    return float(cf_value)
