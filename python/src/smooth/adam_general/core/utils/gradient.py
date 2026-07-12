"""Least-squares initial-state solve for ETS (``initial="gradient"``).

Direct port of ``R/adam-gradient.R``. Given fixed persistence (the current
``mat_f`` / ``vec_g``) and a seeded recent profile, this solves for the initial
state that minimises the in-sample SSE. In a Single Source of Error model the
residuals are, at fixed persistence, an affine function of the initial state for
additive models (exact one-shot least squares) and a smooth nonlinear function
otherwise (Gauss-Newton). It never runs the backcasting backward pass.

The forward-pass oracle is ``adam_cpp.reapply(backcast=False)``: each slice of
the profile cube is one probe, run through the same forward machinery the final
fit uses, in a single C++ call. The least-squares solve is the shared C++
``_ols.ols`` (pivoted QR with rank cutoff) — identical to R's ``olsCpp`` — so the
two languages return bit-for-bit the same result.
"""

from __future__ import annotations

import numpy as np

from smooth.adam_general import _ols  # type: ignore[attr-defined]


def adam_gradient_layout(
    ets_model,
    arima_model,
    xreg_model,
    e_type,
    t_type,
    s_type,
    components_number_ets_seasonal,
    components_number_ets_non_seasonal,
    lags_model,
    lags_model_max,
):
    """Build the free-initial probe map for gradient initialisation of an ETS model.

    Returns ``None`` when the model is out of scope (ARIMA / xreg / non-ETS), so
    the caller falls back to backcasting. Mirrors ``adam_gradientLayout``.
    """
    if not ets_model or arima_model or xreg_model:
        return None

    probes = []
    # level: row 0, all head columns move together
    probes.append({"row": 0, "cols": list(range(lags_model_max))})
    # trend: row 1
    if t_type != "N":
        probes.append({"row": 1, "cols": list(range(lags_model_max))})
    # seasonal: each cell of each seasonal row is a free parameter
    if s_type != "N" and components_number_ets_seasonal > 0:
        for i in range(components_number_ets_seasonal):
            r = components_number_ets_non_seasonal + i
            for c in range(int(lags_model[r])):
                probes.append({"row": r, "cols": [c]})

    additive = (e_type == "A") and (t_type in ("N", "A")) and (s_type in ("N", "A"))
    return {"probes": probes, "e_type": e_type, "additive": additive}


def adam_gradient_solve(
    adam_cpp,
    mat_wt,
    mat_f,
    vec_g,
    index_lookup_table,
    profile,
    y_in_sample,
    ot,
    layout,
    lags_model_max,
    obs_in_sample,
):
    """Solve for the initial recent profile that minimises in-sample SSE.

    Returns the solved recent profile (same shape as ``profile``), or ``None`` on
    failure. Mirrors ``adam_gradientSolve``.
    """
    e_type = layout["e_type"]
    additive = layout["additive"]
    probes = layout["probes"]
    n_free = len(probes)
    if n_free == 0:
        return None

    # y_in_sample and ot arrive as length-obs_in_sample vectors; reapply() needs
    # them as (obs_in_sample, 1) matrices, so reshape once. The same y matrix
    # broadcasts against the (obs, n_slices) fitted values in the residuals, so it
    # doubles as the residual operand. vec_g is an (n_components, 1) persistence
    # column, used as-is. All of these are loop-invariant.
    profile = np.array(profile, dtype=np.float64)
    n_components = profile.shape[0]
    y_matrix = np.asfortranarray(
        np.asarray(y_in_sample, dtype=np.float64).reshape(obs_in_sample, 1)
    )
    ot_matrix = np.asfortranarray(
        np.asarray(ot, dtype=np.float64).reshape(obs_in_sample, 1)
    )
    mat_wt = np.asarray(mat_wt, dtype=np.float64)
    mat_f = np.asarray(mat_f, dtype=np.float64)
    vec_g = np.asarray(vec_g, dtype=np.float64).reshape(n_components, 1)
    index_lookup_table = np.asfortranarray(index_lookup_table, dtype=np.uint64)

    def residuals_for(prof_list):
        """Residuals of one forward pass for candidate profiles (slices).

        Returns an ``(obs_in_sample, n_slices)`` array of residuals, or ``None``.
        """
        n_slices = len(prof_list)
        arr_prof = np.zeros((n_components, lags_model_max, n_slices), order="F")
        for i in range(n_slices):
            arr_prof[:, :, i] = prof_list[i]
        # Slice-count varies per call, so these cubes are genuinely per-call.
        arr_vt = np.zeros(
            (n_components, obs_in_sample + lags_model_max, n_slices), order="F"
        )
        arr_wt = np.repeat(mat_wt[:, :, None], n_slices, axis=2)
        arr_f = np.repeat(mat_f[:, :, None], n_slices, axis=2)
        mat_g = np.repeat(vec_g, n_slices, axis=1)
        try:
            res = adam_cpp.reapply(
                matrixYt=y_matrix,
                matrixOt=ot_matrix,
                arrayVt=np.asfortranarray(arr_vt),
                arrayWt=np.asfortranarray(arr_wt),
                arrayF=np.asfortranarray(arr_f),
                matrixG=np.asfortranarray(mat_g),
                indexLookupTable=index_lookup_table,
                arrayProfilesRecent=np.asfortranarray(arr_prof),
                backcast=False,
            )
        except Exception:
            return None
        y_fitted = np.array(res.fitted, dtype=np.float64).reshape(
            obs_in_sample, n_slices
        )
        if not np.all(np.isfinite(y_fitted)):
            return None
        if e_type == "M":
            e = y_matrix / y_fitted - 1.0
        else:
            e = y_matrix - y_fitted
        if not np.all(np.isfinite(e)):
            return None
        return e

    def apply_step(prof, step):
        prof = np.array(prof, dtype=np.float64)
        for j in range(n_free):
            p = probes[j]
            prof[p["row"], p["cols"]] += step[j]
        return prof

    if additive:
        # Affine: build the design with unit probes, solve once.
        slices = [profile]
        for j in range(n_free):
            pj = np.array(profile, dtype=np.float64)
            pp = probes[j]
            pj[pp["row"], pp["cols"]] += 1.0
            slices.append(pj)
        e = residuals_for(slices)
        if e is None:
            return None
        e0 = e[:, 0]
        design_a = np.zeros((obs_in_sample, n_free), dtype=np.float64)
        for j in range(n_free):
            design_a[:, j] = e0 - e[:, j + 1]
        # Solve min ||e0 - A d|| in C++ (pivoted QR, rank-deficiency aware) so the
        # result is identical to the R engine that shares the same core.
        d = np.asarray(_ols.ols(design_a, e0, 1e-7), dtype=np.float64).ravel()
        d[~np.isfinite(d)] = 0.0
        return apply_step(profile, d)

    # Nonlinear: Gauss-Newton with finite-difference Jacobian and line search.
    prof = np.array(profile, dtype=np.float64)
    e0m = residuals_for([prof])
    if e0m is None:
        return None
    e0 = e0m[:, 0]
    f_cur = float(np.sum(e0**2))
    for _ in range(15):
        slices = []
        hs = np.zeros(n_free, dtype=np.float64)
        for j in range(n_free):
            pp = probes[j]
            h = 1e-4 * max(1.0, abs(prof[pp["row"], pp["cols"][0]]))
            pj = np.array(prof, dtype=np.float64)
            pj[pp["row"], pp["cols"]] += h
            slices.append(pj)
            hs[j] = h
        e = residuals_for(slices)
        if e is None:
            break
        jm = np.zeros((obs_in_sample, n_free), dtype=np.float64)
        for j in range(n_free):
            jm[:, j] = (e[:, j] - e0) / hs[j]
        try:
            step = -np.asarray(_ols.ols(jm, e0, 1e-7), dtype=np.float64).ravel()
        except Exception:
            step = np.full(n_free, np.nan)
        step[~np.isfinite(step)] = 0.0
        if np.sqrt(np.sum(step**2)) < 1e-8:
            break
        improved = False
        for tt in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
            et = residuals_for([apply_step(prof, tt * step)])
            if et is not None and float(np.sum(et[:, 0] ** 2)) < f_cur:
                prof = apply_step(prof, tt * step)
                e0 = et[:, 0]
                f_cur = float(np.sum(e0**2))
                improved = True
                break
        if not improved:
            break
    return prof


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
        layout = adam_gradient_layout(
            bool(model_type_dict.get("ets_model", False)),
            bool(model_type_dict.get("arima_model", False)),
            bool(model_type_dict.get("xreg_model", False)),
            model_type_dict.get("error_type", "A"),
            model_type_dict.get("trend_type", "N"),
            model_type_dict.get("season_type", "N"),
            int(components_dict.get("components_number_ets_seasonal", 0) or 0),
            int(components_dict.get("components_number_ets_non_seasonal", 0) or 0),
            lags_dict["lags_model"],
            lags_model_max,
        )
        if layout is not None:
            solved = adam_gradient_solve(
                adam_cpp,
                mat_wt,
                mat_f,
                vec_g,
                index_lookup_table,
                profiles_recent_table,
                y_in_sample,
                ot,
                layout,
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
                    profilesRecent=solved,
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
