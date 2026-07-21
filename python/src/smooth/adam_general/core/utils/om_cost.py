"""Cost function for occurrence (OM) models.

Fills the state-space matrices from the optimisation vector ``B``, runs the
C++ fitter with the occurrence flag, applies the link function to map the
raw fitted values into ``[0, 1]`` probabilities, and returns the negative
Bernoulli log-likelihood (or MSE on the binary indicators).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.linalg import eigvals

from smooth.adam_general.core.creator import filler
from smooth.adam_general.core.utils.gradient import adam_fit_or_gradient


def om_link_function(x, error_type, occurrence):
    """Map raw state-space fitted values onto a probability scale.

    The exact transformation depends on the ``occurrence`` type
    (odds-ratio, inverse-odds-ratio, direct or fixed) and on the
    underlying error type (additive or multiplicative).
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
    return_fitted=False,
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
    # An explicit copy: the gradient dispatcher overwrites the head of mat_vt
    # in place, and np.asfortranarray would alias an already-Fortran-ordered
    # array, leaking the solved initials into the shared creator matrices that
    # the om preparator later re-seeds from (R's copy-on-modify never leaks it).
    mat_vt = np.array(adam_elements["mat_vt"], dtype=np.float64, order="F")

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

    # Fit dispatcher: for initial="gradient" this solves the initials by
    # profiling the occurrence loss over the probability residuals (in C++);
    # otherwise the ordinary occurrence fit, with gradient joining the
    # backcasting group as a fall-back. Mirrors R's omCF_local routing.
    adam_fitted = adam_fit_or_gradient(
        adam_cpp=adam_cpp,
        mat_vt=mat_vt,
        mat_wt=mat_wt,
        mat_f=mat_f,
        vec_g=vec_g,
        index_lookup_table=index_lookup_table,
        profiles_recent_table=profiles_recent_table,
        y_in_sample=ot,
        ot=ot,
        initial_type=initials_checked["initial_type"],
        n_iterations=initials_checked["n_iterations"],
        backcast_value=backcast_value,
        model_type_dict=model_type_dict,
        components_dict=components_dict,
        lags_dict=lags_dict,
        obs_in_sample=observations_dict["obs_in_sample"],
        o_type=occurrence_char,
        loss=general.get("loss", "likelihood"),
        xreg_number=int(explanatory_checked.get("xreg_number", 0) or 0),
    )

    # 4. Apply link function to raw fitted output
    error_type = model_type_dict["error_type"]
    p_fitted = om_link_function(
        np.asarray(adam_fitted.fitted).ravel(), error_type, occurrence
    )

    # The fitted probability at this B, returned directly for the OPG covariance
    # (its per-observation Bernoulli score); the caller applies its own
    # feasibility check. Mirrors omg_cf's return_fitted.
    if return_fitted:
        return p_fitted

    # Infeasibility guard (NOT a clipping hack): if the link function
    # produced NaN or values outside [0, 1], the parameters at this point
    # are infeasible for the model — return a uniformly large penalty so
    # the optimiser steers away.
    if np.any(np.isnan(p_fitted)) or np.any(p_fitted < 0) or np.any(p_fitted > 1):
        return 1e300

    # 5. Compute loss — mirrors ADAM's CF dispatch
    # (`cost_functions.py:544-685`). The same single-step loss menu plus
    # LASSO/RIDGE and custom-callable, but errors are
    # ``residual = ot - p_fitted`` on the probability scale instead of the
    # ADAM additive/multiplicative errors.
    ot_logical = observations_dict["ot_logical"]
    loss = general.get("loss", "likelihood")
    loss_function = general.get("loss_function")
    residual = ot - p_fitted

    if loss == "custom":
        if loss_function is None:
            raise ValueError(
                "loss='custom' requires `loss_function` in `general` dict; "
                "om.OM.__init__ should have spliced it in."
            )
        cf_value = float(loss_function(actual=ot, fitted=p_fitted, B=np.asarray(B)))
    elif loss == "likelihood":
        # Bernoulli log-likelihood: -(sum log p[ot=1] + sum log(1-p)[ot=0]).
        # No epsilon floor inside log() — see CLAUDE.md "never clip" rule.
        # If p hits 0 or 1 exactly, the resulting -Inf is a true signal that
        # the initialiser handed the optimiser a parameter region the model
        # cannot represent, and that should surface, not be hidden.
        #
        # Aggregate via math.fsum (Shewchuk exact summation) instead of
        # np.sum (pairwise IEEE-double) to match R's LDOUBLE-accumulator
        # sum() byte-for-byte at feasible points.  Same ULP-level
        # difference can otherwise push NLopt's deterministic simplex
        # into a different basin on flat-loss seasonal OM surfaces.
        p_on = p_fitted[ot_logical]
        p_off = p_fitted[~ot_logical]
        cf_value = -(
            math.fsum(np.log(p_on).tolist()) + math.fsum(np.log(1.0 - p_off).tolist())
        )
    elif loss == "MSE":
        cf_value = float(np.mean(residual**2))
    elif loss == "MAE":
        cf_value = float(np.mean(np.abs(residual)))
    elif loss == "HAM":
        cf_value = float(np.mean(np.sqrt(np.abs(residual))))
    elif loss in ("LASSO", "RIDGE"):
        from smooth.adam_general.core.utils.cost_functions import trim_b_for_penalty

        lam = float(general.get("lambda", 0) or 0.0)
        B_penalty = trim_b_for_penalty(  # noqa: N806
            B,
            components_dict,
            persistence_checked,
            explanatory_checked,
            phi_dict,
            arima_checked,
            initials_checked,
            model_type_dict,
            lags_dict,
            general,
        )
        obs_in_sample = observations_dict.get("obs_in_sample", len(residual))
        # Error term identical in shape to ADAM's additive branch but
        # measured on the probability-scale residual (the OM "error").
        error_term = (
            (1.0 - lam)
            * float(np.linalg.norm(residual))
            / float(np.sqrt(obs_in_sample))
        )
        if loss == "LASSO":
            cf_value = error_term + lam * float(np.sum(np.abs(B_penalty)))
        else:  # RIDGE
            cf_value = error_term + lam * float(np.linalg.norm(B_penalty))
    else:
        raise ValueError(f"Unsupported OM loss={loss!r}.")

    return cf_value
