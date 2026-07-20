"""Cost function for general occurrence (OMG) models.

Fills two parallel sets of state-space matrices from a joint parameter
vector ``B = [B_A | B_B]``, runs the C++ ``adamCore.omfitGeneral`` to
advance both sub-models simultaneously, applies ``omg_link_function`` to
combine the two raw fitted vectors into a probability, and returns the
negative Bernoulli log-likelihood.

The single-model OM cost lives in :mod:`om_cost`; this module is the
two-model analogue.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.linalg import eigvals

from smooth.adam_general.core.creator import filler


def _side_probe_basis(side, elem, o_type="g"):
    """Per-side probe basis for the coupled occurrence gradient solve.

    Marks each side's free initial cells (ETS / ARIMA / xreg) exactly as the
    single-model dispatcher does; ``o_type="g"`` opens the occurrence path
    (ARIMA + xreg solvable via the finite-difference Jacobian). Mirrors the
    per-side ``adam_gradientProbeBasis`` calls in R's ``omgCF_local``.
    """
    from smooth.adam_general.core.utils.gradient import adam_gradient_probe_basis

    mt = side["model_type_dict"]
    cd = side["components_dict"]
    ld = side["lags_dict"]
    return adam_gradient_probe_basis(
        bool(mt.get("ets_model", False)),
        bool(mt.get("arima_model", False)),
        bool(mt.get("xreg_model", False)),
        mt.get("trend_type", "N"),
        mt.get("season_type", "N"),
        int(cd.get("components_number_ets_seasonal", 0) or 0),
        int(cd.get("components_number_ets_non_seasonal", 0) or 0),
        int(elem["mat_vt"].shape[0]),
        ld["lags_model"],
        int(ld["lags_model_max"]),
        mt.get("error_type", "A"),
        int(cd.get("components_number_ets", 0) or 0),
        int(cd.get("components_number_arima", 0) or 0),
        ld.get("lags_model_all", ld["lags_model"]),
        int(side["explanatory"].get("xreg_number", 0) or 0),
        o_type,
    )


def _omg_gradient_profiles(side_a, side_b, elem_a, elem_b, ot, loss, adam_ets=False):
    """Coupled gradient solve of the two occurrence initial profiles.

    Returns ``(profileA, profileB)`` from ``adamCore.gradientSolveGeneral`` when
    the model is in scope (a mappable loss and at least one solvable side), or
    ``None`` to fall back to the ordinary coupled fit. Mirrors the gradient
    branch of R's ``omgCF_local``.
    """
    from smooth.adam_general.core.utils.gradient import adam_gradient_om_loss_code

    loss_code = adam_gradient_om_loss_code(loss)
    if loss_code is None:
        return None
    pb_a = _side_probe_basis(side_a, elem_a)
    pb_b = _side_probe_basis(side_b, elem_b)
    if pb_a is None and pb_b is None:
        return None

    def _f(x, dtype=np.float64):
        return np.asfortranarray(x, dtype=dtype)

    n_a = int(elem_a["mat_vt"].shape[0])
    n_b = int(elem_b["mat_vt"].shape[0])
    if pb_a is None:
        pb_a = np.zeros((n_a, 0), order="F")
    if pb_b is None:
        pb_b = np.zeros((n_b, 0), order="F")

    solved = side_a["adam_cpp"].gradientSolveGeneral(
        matrixVtA=_f(elem_a["mat_vt"]),
        matrixWtA=_f(elem_a["mat_wt"]),
        matrixFA=_f(elem_a["mat_f"]),
        vectorGA=_f(elem_a["vec_g"]),
        indexLookupTableA=_f(side_a["profile"]["index_lookup_table"], np.uint64),
        profileA=_f(side_a["profile"]["profiles_recent_table"]),
        probeBasisA=_f(pb_a),
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
        adamETSB=bool(adam_ets),
        matrixVtB=_f(elem_b["mat_vt"]),
        matrixWtB=_f(elem_b["mat_wt"]),
        matrixFB=_f(elem_b["mat_f"]),
        vectorGB=_f(elem_b["vec_g"]),
        indexLookupTableB=_f(side_b["profile"]["index_lookup_table"], np.uint64),
        profileB=_f(side_b["profile"]["profiles_recent_table"]),
        probeBasisB=_f(pb_b),
        vectorOt=_f(ot),
        nIterations=15,
        lossType=loss_code[0],
    )
    prof_a = np.asarray(solved.profileA, dtype=np.float64)
    prof_b = np.asarray(solved.profileB, dtype=np.float64)
    if prof_a.size == 0 or prof_b.size == 0:
        return None
    return prof_a, prof_b


def omg_link_function(fitted_a, fitted_b, error_type_a, error_type_b):
    """Combine the raw fitted outputs of two sub-models into a probability.

    All four branches are numerically stable reformulations of
    ``aFit / (aFit + bFit)`` that avoid exp-overflow by dividing through by
    the larger of the two exponentials before returning.

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
    loss: str = "likelihood",
    loss_function=None,
    reg_lambda: Optional[float] = None,
    return_fitted: bool = False,
):
    """OMG cost function — joint Bernoulli likelihood on combined probability
    (or a single-step / regularised / custom loss).

    With ``return_fitted=True`` the coupled probability vector is returned
    instead of the scalar loss, so the final object's fitted values (and the
    Bernoulli logLik computed from them) come from the same coupled recursion
    the optimiser minimised — not the standalone per-side refit. Mirrors
    ``omgCF_local``'s ``returnFitted``.

    ``side_a`` and ``side_b`` are dicts collecting everything ``filler`` and
    ``adam_cpp.omfitGeneral`` need for the two sub-models. ``n_params_a``
    splits the concatenated parameter vector.

    The C++ joint state-space step always runs (returns the combined
    probability via the link function), then ``loss`` decides what to
    return as the scalar objective. ``"likelihood"`` is the joint
    Bernoulli; ``"MSE" / "MAE" / "HAM"`` use the probability-scale
    residual ``ot - p_combined``; ``"LASSO" / "RIDGE"`` add the standard
    penalty on the joint ``B = [B_A | B_B]``; ``"custom"`` delegates to
    ``loss_function(actual, fitted, B)``. Mirrors :func:`om_cf`.
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
    init_type_a = initials_a["initial_type"]
    if isinstance(init_type_a, list):
        backcast = any(t in ("complete", "backcasting") for t in init_type_a)
        is_gradient = "gradient" in init_type_a
    else:
        backcast = init_type_a in ("complete", "backcasting")
        is_gradient = init_type_a == "gradient"

    # initial="gradient": solve both occurrence initials jointly over the shared
    # probability residual (coupled Gauss-Newton in C++), then run one forward
    # pass from the solved profiles. Gradient joins the backcasting group as a
    # fall-back when out of scope. Mirrors R's omgCF_local.
    prof_a_used = side_a["profile"]["profiles_recent_table"]
    prof_b_used = side_b["profile"]["profiles_recent_table"]
    grad_backcast = backcast or is_gradient
    grad_n_iter = int(initials_a["n_iterations"])
    if is_gradient:
        solved = _omg_gradient_profiles(
            side_a, side_b, elem_a, elem_b, ot, loss, adam_ets
        )
        if solved is not None:
            prof_a_used, prof_b_used = solved
            grad_backcast = False
            grad_n_iter = 1

    res = side_a["adam_cpp"].omfitGeneral(
        matrixVtA=_f(elem_a["mat_vt"]),
        matrixWtA=_f(elem_a["mat_wt"]),
        matrixFA=_f(elem_a["mat_f"]),
        vectorGA=_f(elem_a["vec_g"]),
        indexLookupTableA=_f(side_a["profile"]["index_lookup_table"], np.uint64),
        profilesRecentA=_f(prof_a_used),
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
        profilesRecentB=_f(prof_b_used),
        vectorOt=ot,
        backcast=grad_backcast,
        nIterations=grad_n_iter,
    )

    e_a = side_a["model_type_dict"]["error_type"]
    e_b = side_b["model_type_dict"]["error_type"]
    p_combined = omg_link_function(
        np.asarray(res.fittedA).ravel(),
        np.asarray(res.fittedB).ravel(),
        e_a,
        e_b,
    )

    # The coupled fitted probability, returned directly for the final object so
    # its fitted values and Bernoulli logLik come from the same recursion the
    # optimiser minimised (mirrors omgCF_local's returnFitted).
    if return_fitted:
        return p_combined

    # Infeasibility guard: NaN or boundary p means the parameters are
    # inconsistent with the data — return a uniform large penalty so the
    # optimiser steers away. NOT a clip on the model output.
    if (
        np.any(np.isnan(p_combined))
        or np.any(p_combined <= 0)
        or np.any(p_combined >= 1)
    ):
        return 1e300

    ot_logical = observations_dict["ot_logical"]
    residual = ot - p_combined

    if loss == "custom":
        if loss_function is None:
            raise ValueError(
                "loss='custom' requires `loss_function`; OMG.__init__ should "
                "have captured the callable and passed it through."
            )
        return float(loss_function(actual=ot, fitted=p_combined, B=np.asarray(B)))
    if loss == "likelihood":
        return float(
            -(
                np.sum(np.log(p_combined[ot_logical]))
                + np.sum(np.log(1.0 - p_combined[~ot_logical]))
            )
        )
    if loss == "MSE":
        return float(np.mean(residual**2))
    if loss == "MAE":
        return float(np.mean(np.abs(residual)))
    if loss == "HAM":
        return float(np.mean(np.sqrt(np.abs(residual))))
    if loss in ("LASSO", "RIDGE"):
        from smooth.adam_general.core.utils.cost_functions import trim_b_for_penalty

        lam = float(reg_lambda if reg_lambda is not None else 0.0)
        # Trim each side separately (each has its own component layout)
        # and concatenate for the joint penalty.
        B_pen_a = trim_b_for_penalty(  # noqa: N806
            B[:n_params_a],
            side_a["components_dict"],
            side_a["persistence"],
            side_a["explanatory"],
            side_a["phi"],
            side_a["arima"],
            side_a["initials"],
            side_a["model_type_dict"],
            side_a["lags_dict"],
            {},
        )
        B_pen_b = trim_b_for_penalty(  # noqa: N806
            B[n_params_a:],
            side_b["components_dict"],
            side_b["persistence"],
            side_b["explanatory"],
            side_b["phi"],
            side_b["arima"],
            side_b["initials"],
            side_b["model_type_dict"],
            side_b["lags_dict"],
            {},
        )
        B_penalty = np.concatenate([B_pen_a, B_pen_b])  # noqa: N806

        obs_in_sample = int(observations_dict.get("obs_in_sample", len(residual)))
        error_term = (
            (1.0 - lam)
            * float(np.linalg.norm(residual))
            / float(np.sqrt(obs_in_sample))
        )
        if loss == "LASSO":
            return error_term + lam * float(np.sum(np.abs(B_penalty)))
        return error_term + lam * float(np.linalg.norm(B_penalty))
    raise ValueError(f"Unsupported OMG loss={loss!r}.")
