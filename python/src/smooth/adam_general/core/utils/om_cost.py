"""Cost function for occurrence (OM) models.

Mirrors ``omCF_local`` in ``R/om.R``: fills the state-space matrices from B,
runs the C++ fitter with the occurrence flag, applies the link function to map
raw fitted values into [0, 1] probabilities, and returns the negative
Bernoulli log-likelihood (or MSE on the binary indicators).
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import eigvals

from smooth.adam_general.core.creator import filler


def om_link_function(x, error_type, occurrence):
    """Translation of ``omLinkFunction`` (R/om.R:1258).

    Maps raw state-space fitted values onto a probability scale.
    """
    x = np.asarray(x, dtype=np.float64)
    if occurrence == "odds-ratio":
        if error_type == "A":
            ex = np.exp(x)
            return ex / (1.0 + ex)
        return x / (1.0 + x)
    if occurrence == "inverse-odds-ratio":
        if error_type == "A":
            return 1.0 / (1.0 + np.exp(x))
        return 1.0 / (1.0 + x)
    if occurrence in ("fixed", "direct"):
        return np.clip(x, 0.0, 1.0)
    return x


def om_cf(  # noqa: N802
    B,
    model_type_dict,
    components_dict,
    lags_dict,
    matrices_dict,
    persistence_checked,
    initials_checked,
    arima_checked,
    explanatory_checked,
    phi_dict,
    constants_checked,
    observations_dict,
    profile_dict,
    general,
    adam_cpp,
    occurrence,
    occurrence_char,
    bounds="usual",
    arPolynomialMatrix=None,  # noqa: N803
    maPolynomialMatrix=None,  # noqa: N803
    regressors=None,
):
    """OM cost function (Bernoulli log-likelihood or MSE on binary indicators).

    Differences from :func:`smooth.adam_general.core.utils.cost_functions.CF`:

    * always passes ``vectorYt = vectorOt = ot`` (binary) to the C++ fitter,
    * passes ``O = occurrence_char`` to enable the C++ occurrence error path,
    * applies ``om_link_function`` to the raw fitted output to obtain
      probabilities,
    * computes Bernoulli log-likelihood or MSE on (ot - p_fitted) instead of
      the additive/multiplicative ADAM likelihood.
    """
    # 1. Fill matrices with current parameter values
    adam_elements = filler(
        B,
        model_type_dict,
        components_dict,
        lags_dict,
        matrices_dict,
        persistence_checked,
        initials_checked,
        arima_checked,
        explanatory_checked,
        phi_dict,
        constants_checked,
        adam_cpp=adam_cpp,
    )

    # Capture the filler-modified state for C++ and profiles
    profile_dict["profiles_recent_table"][:] = adam_elements["mat_vt"][
        :, : lags_dict["lags_model_max"]
    ]
    mat_vt = np.asfortranarray(adam_elements["mat_vt"], dtype=np.float64)

    # 2. Bounds checking (mirror CF's "usual" branch — admissible/none not
    #    used by om() but support graceful pass-through)
    if bounds == "usual":
        if arima_checked["arima_model"] and any(
            [arima_checked["ar_estimate"], arima_checked["ma_estimate"]]
        ):
            if (
                arima_checked["ar_estimate"]
                and np.all(-adam_elements["arima_polynomials"]["arPolynomial"][1:] > 0)
                and sum(-adam_elements["arima_polynomials"]["arPolynomial"][1:]) >= 1
            ):
                arPolynomialMatrix[:, 0] = -adam_elements["arima_polynomials"][
                    "arPolynomial"
                ][1:]
                roots = np.abs(eigvals(arPolynomialMatrix))
                if any(roots > 1):
                    return 1e100 * max(roots)
            if (
                arima_checked["ma_estimate"]
                and sum(adam_elements["arima_polynomials"]["maPolynomial"][1:]) >= 1
            ):
                maPolynomialMatrix[:, 0] = adam_elements["arima_polynomials"][
                    "maPolynomial"
                ][1:]
                roots = np.abs(eigvals(maPolynomialMatrix))
                if any(roots > 1):
                    return 1e100 * max(abs(roots))

        if model_type_dict["ets_model"]:
            n_ets = components_dict["components_number_ets"]
            vec_g = adam_elements["vec_g"]
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
            if phi_dict["phi_estimate"] and (
                adam_elements["mat_f"][1, 1] > 1 or adam_elements["mat_f"][1, 1] < 0
            ):
                return 1e300

    # 3. Run the C++ fitter with O=occurrence_char and y=ot=binary indicators
    ot = np.asarray(observations_dict["ot"], dtype=np.float64)

    mat_wt = np.asfortranarray(adam_elements["mat_wt"], dtype=np.float64)
    mat_f = np.asfortranarray(adam_elements["mat_f"], dtype=np.float64)
    vec_g = np.asfortranarray(adam_elements["vec_g"], dtype=np.float64)
    index_lookup_table = np.asfortranarray(
        profile_dict["index_lookup_table"], dtype=np.uint64
    )
    profiles_recent_table = np.asfortranarray(
        profile_dict["profiles_recent_table"], dtype=np.float64
    )

    if isinstance(initials_checked["initial_type"], list):
        backcast_value = any(
            t in ("complete", "backcasting") for t in initials_checked["initial_type"]
        )
    else:
        backcast_value = initials_checked["initial_type"] in (
            "complete",
            "backcasting",
        )

    adam_fitted = adam_cpp.fit(
        matrixVt=mat_vt,
        matrixWt=mat_wt,
        matrixF=mat_f,
        vectorG=vec_g,
        indexLookupTable=index_lookup_table,
        profilesRecent=profiles_recent_table,
        vectorYt=ot,
        vectorOt=ot,
        backcast=backcast_value,
        nIterations=initials_checked["n_iterations"],
        refineHead=True,
        O=occurrence_char,
    )

    # 4. Apply link function to raw fitted output
    error_type = model_type_dict["error_type"]
    p_fitted = om_link_function(
        np.asarray(adam_fitted.fitted).ravel(), error_type, occurrence
    )

    # Infeasibility guard (NOT a clipping hack): if the link function
    # produced NaN or values outside [0, 1], the parameters at this point
    # are infeasible for the model — return a uniformly large penalty so
    # the optimiser steers away. Mirrors R/om.R:omCF_local.
    if np.any(np.isnan(p_fitted)) or np.any(p_fitted < 0) or np.any(p_fitted > 1):
        return 1e300

    # 5. Compute loss
    ot_logical = observations_dict["ot_logical"]
    loss = general.get("loss", "likelihood")
    if loss == "likelihood":
        # Bernoulli log-likelihood: -(sum log p[ot=1] + sum log(1-p)[ot=0]).
        # No epsilon floor inside log() — see CLAUDE.md "never clip" rule.
        # If p hits 0 or 1 exactly, the resulting -Inf is a true signal that
        # the initialiser handed the optimiser a parameter region the model
        # cannot represent, and that should surface, not be hidden.
        cf_value = -(
            np.sum(np.log(p_fitted[ot_logical]))
            + np.sum(np.log(1.0 - p_fitted[~ot_logical]))
        )
    else:
        cf_value = float(np.mean((ot - p_fitted) ** 2))

    return cf_value
