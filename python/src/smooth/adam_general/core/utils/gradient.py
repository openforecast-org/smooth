"""Least-squares initial-state solve for ETS (``initial="gradient"``).

Direct port of ``R/adam-gradient.R``. Given fixed persistence (the current
``mat_f`` / ``vec_g``) and a seeded recent profile, this solves for the initial
state that minimises the in-sample SSE. It never runs the backcasting backward
pass.

All the numerical work happens in C++ (``adam_cpp.gradientSolve``, shared with
the R build for exact parity): for additive ETS the residuals are affine in the
initial state, so the design is propagated analytically alongside a single
forward pass and solved by one pivoted-QR least squares; otherwise
finite-difference Gauss-Newton with a Levenberg-Marquardt fallback is used.
The solve is stateless — a deterministic function of its inputs.
"""

from __future__ import annotations

import numpy as np


def adam_gradient_probe_basis(
    ets_model,
    arima_model,
    xreg_model,
    t_type,
    s_type,
    components_number_ets_seasonal,
    components_number_ets_non_seasonal,
    n_components,
    lags_model,
    lags_model_max,
):
    """Build the probe basis for gradient initialisation of an ETS model.

    A ``(n_components * lags_model_max, n_free)`` 0/1 matrix whose column j marks
    the profile cells (in column-major order, matching the C++ lookup indices)
    that move together as one free initial parameter. Level and trend span all
    head columns; each seasonal cell is its own parameter. Returns ``None`` when
    the model is out of scope (ARIMA / xreg / non-ETS), so the caller falls back
    to backcasting. Mirrors ``adam_gradientProbeBasis``.
    """
    if not ets_model or arima_model or xreg_model:
        return None

    def cells_of(row, cols):
        """Column-major cell indices of profile[row, cols]."""
        return [row + c * n_components for c in cols]

    probes = [cells_of(0, range(lags_model_max))]
    if t_type != "N":
        probes.append(cells_of(1, range(lags_model_max)))
    if s_type != "N" and components_number_ets_seasonal > 0:
        for i in range(components_number_ets_seasonal):
            r = components_number_ets_non_seasonal + i
            for j in range(int(lags_model[r])):
                probes.append(cells_of(r, [j]))

    probe_basis = np.zeros((n_components * lags_model_max, len(probes)), order="F")
    for j, cells in enumerate(probes):
        probe_basis[cells, j] = 1.0
    return probe_basis


def adam_gradient_solve(
    adam_cpp,
    mat_wt,
    mat_f,
    vec_g,
    index_lookup_table,
    profile,
    y_in_sample,
    ot,
    probe_basis,
    lags_model_max,
    obs_in_sample,
):
    """Solve for the initial recent profile that minimises in-sample SSE.

    Thin wrapper around the shared C++ ``gradientSolve``. Returns the solved
    recent profile (same shape as ``profile``), or ``None`` on failure.
    Mirrors ``adam_gradientSolve``.
    """
    solved = adam_cpp.gradientSolve(
        matrixYt=np.asfortranarray(
            np.asarray(y_in_sample, dtype=np.float64).reshape(obs_in_sample, 1)
        ),
        matrixOt=np.asfortranarray(
            np.asarray(ot, dtype=np.float64).reshape(obs_in_sample, 1)
        ),
        matrixWt=np.asfortranarray(mat_wt, dtype=np.float64),
        matrixF=np.asfortranarray(mat_f, dtype=np.float64),
        vectorG=np.asfortranarray(vec_g, dtype=np.float64).ravel(),
        indexLookupTable=np.asfortranarray(index_lookup_table, dtype=np.uint64),
        profile=np.asfortranarray(profile, dtype=np.float64),
        probeBasis=np.asfortranarray(probe_basis, dtype=np.float64),
        nIterations=15,
        analytic=True,
    )
    solved = np.asarray(solved, dtype=np.float64)
    if solved.size == 0:
        return None
    return solved


def adam_fit_or_gradient(
    adam_cpp,
    mat_vt,
    mat_wt,
    mat_f,
    vec_g,
    index_lookup_table,
    profiles_recent_table,
    y_in_sample,
    ot,
    initial_type,
    n_iterations,
    backcast_value,
    model_type_dict,
    components_dict,
    lags_dict,
    obs_in_sample,
    o_type="n",
):
    """Fit dispatcher mirroring ``adam_fitOrGradient``.

    Runs the gradient initial-state solve when ``initial_type == "gradient"`` and
    the model is in scope (ETS, no ARIMA/xreg), otherwise the ordinary
    ``adam_cpp.fit`` (with the backcast flag). Gradient always fits with
    ``backcast=False`` from the solved initials and ``n_iterations=1``; it never
    runs the backward pass. Out-of-scope gradient models fall back to backcasting.

    ``mat_vt`` is mutated in place (its head columns are overwritten with the
    solved profile) and the C++ fit result is returned.
    """
    is_gradient = initial_type == "gradient" or (
        isinstance(initial_type, (list, tuple)) and "gradient" in initial_type
    )
    if is_gradient:
        lags_model_max = int(lags_dict["lags_model_max"])
        probe_basis = adam_gradient_probe_basis(
            bool(model_type_dict.get("ets_model", False)),
            bool(model_type_dict.get("arima_model", False)),
            bool(model_type_dict.get("xreg_model", False)),
            model_type_dict.get("trend_type", "N"),
            model_type_dict.get("season_type", "N"),
            int(components_dict.get("components_number_ets_seasonal", 0) or 0),
            int(components_dict.get("components_number_ets_non_seasonal", 0) or 0),
            int(profiles_recent_table.shape[0]),
            lags_dict["lags_model"],
            lags_model_max,
        )
        if probe_basis is not None:
            solved = adam_gradient_solve(
                adam_cpp,
                mat_wt,
                mat_f,
                vec_g,
                index_lookup_table,
                profiles_recent_table,
                y_in_sample,
                ot,
                probe_basis,
                lags_model_max,
                obs_in_sample,
            )
            if solved is not None:
                mat_vt[:, :lags_model_max] = solved
                # A single forward pass from the solved initials (backcast=False).
                # n_iterations must be 1 here: with no backward pass, a second
                # iteration would re-run the head fill on the profile already
                # mutated by the first forward pass and diverge.
                return adam_cpp.fit(
                    matrixVt=mat_vt,
                    matrixWt=mat_wt,
                    matrixF=mat_f,
                    vectorG=vec_g,
                    indexLookupTable=index_lookup_table,
                    profilesRecent=np.asfortranarray(solved),
                    vectorYt=y_in_sample,
                    vectorOt=ot,
                    backcast=False,
                    nIterations=1,
                    O=o_type,
                )
    # Fallback (non-gradient, or gradient out of scope / solve failed). R joins
    # gradient to the backcast group here: any(c("complete","backcasting","gradient")).
    return adam_cpp.fit(
        matrixVt=mat_vt,
        matrixWt=mat_wt,
        matrixF=mat_f,
        vectorG=vec_g,
        indexLookupTable=index_lookup_table,
        profilesRecent=profiles_recent_table,
        vectorYt=y_in_sample,
        vectorOt=ot,
        backcast=True if is_gradient else backcast_value,
        nIterations=n_iterations,
        O=o_type,
    )
