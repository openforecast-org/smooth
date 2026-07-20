"""Initial-state solve for ETS (``initial="gradient"``).

Direct port of ``R/adam-gradient.R``. Given fixed persistence (the current
``mat_f`` / ``vec_g``) and a seeded recent profile, this solves for the initial
state that minimises the estimation loss. It never runs the backcasting
backward pass.

All the numerical work happens in C++ (``adam_cpp.gradientSolve``, shared with
the R build for exact parity): for additive ETS the residuals are affine in the
initial state, so the design is propagated analytically alongside a single
forward pass and solved by pivoted-QR least squares (with IRLS sweeps or the
affine multistep designs for the non-SSE losses); otherwise loss-aware
Gauss-Newton with an analytic Jacobian and a Levenberg-Marquardt fallback is
used. The solve is stateless — a deterministic function of its inputs.
"""

from __future__ import annotations

import numpy as np


def adam_gradient_loss_code(loss, distribution, e_type, other, horizon, multisteps):
    """Map the estimation loss + distribution pair to the C++ loss code.

    Returns ``(code, params)`` for ``gradientSolve`` (codes documented in
    ``src/headers/adamGradient.h``), or ``None`` for custom loss functions
    (which cannot cross into C++), making the caller fall back to backcasting.
    This table must stay mirror-identical to ``adam_gradientLossCode`` in
    ``R/adam-gradient.R``. Losses without a matching rho keep the SSE default
    ``"S"``; the multiplicative likelihood codes (``l``/``g``/``i``) are mapped
    for ``e_type == "M"`` only.
    """
    if loss == "custom" or callable(loss):
        return None
    if distribution == "default":
        if loss == "likelihood":
            distribution = "dnorm" if e_type == "A" else "dgamma"
        elif loss in ("MAEh", "MACE", "MAE"):
            distribution = "dlaplace"
        elif loss in ("HAMh", "CHAM", "HAM"):
            distribution = "ds"
        else:
            distribution = "dnorm"
    if multisteps:
        code = {"MSEh": "h", "TMSE": "T", "GTMSE": "t", "MSCE": "C", "GPL": "P"}.get(
            loss, "S"
        )
        if code != "S" and horizon is not None and horizon >= 1:
            return code, float(horizon)
        return "S", 0.0
    if loss == "MAE":
        return "A", 0.0
    if loss == "HAM":
        return "H", 0.0
    if loss == "likelihood":
        code = {
            "dlaplace": "A",
            "ds": "H",
            "dgnorm": "G",
            "dlnorm": "l" if e_type == "M" else "S",
            "dgamma": "g" if e_type == "M" else "S",
            "dinvgauss": "i" if e_type == "M" else "S",
        }.get(distribution, "S")
        if code == "G":
            if other is None or not np.isfinite(other):
                return "S", 0.0
            return "G", float(other)
        return code, 0.0
    return "S", 0.0


def adam_gradient_om_loss_code(loss):
    """Map an occurrence-model (om) loss to the C++ gradientSolve loss code.

    The om losses act on the probability residual ``r = ot - p``: the Bernoulli
    likelihood is exactly ``-log(1-|r|)`` (code ``"B"``), and MSE/MAE/HAM reuse
    the standard rho codes on ``r``. Must stay mirror-identical to
    ``adam_gradientOmLossCode`` in ``R/adam-gradient.R``. Returns ``None`` for
    custom loss callables (backcasting fallback).
    """
    if loss == "custom" or callable(loss):
        return None
    return {"likelihood": ("B", 0.0), "MAE": ("A", 0.0), "HAM": ("H", 0.0)}.get(
        loss, ("S", 0.0)
    )


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
    e_type="A",
    components_number_ets=0,
    components_number_arima=0,
    lags_model_all=None,
):
    """Build the probe basis for gradient initialisation.

    A ``(n_components * lags_model_max, n_free)`` 0/1 matrix whose column j marks
    the profile cells (column-major, matching the C++ lookup indices) that move
    together as one free initial parameter. ETS: level and trend span all head
    columns, each seasonal cell is its own parameter. ARIMA: each state's lag
    slots are free parameters (the rank-revealing solve drops redundant
    directions) -- included only for ADDITIVE models (the exact affine
    least-squares branch). Multiplicative ARIMA and xreg fall back (``None``).
    Mirrors ``adam_gradientProbeBasis``.
    """
    if xreg_model:
        return None
    if not ets_model and not arima_model:
        return None
    additive = e_type == "A" and t_type not in ("M", "Md") and s_type != "M"
    if arima_model and not additive:
        return None
    if lags_model_all is None:
        lags_model_all = lags_model

    def cells_of(row, cols):
        """Column-major cell indices of profile[row, cols]."""
        return [row + c * n_components for c in cols]

    probes = []
    if ets_model:
        probes.append(cells_of(0, range(lags_model_max)))
        if t_type != "N":
            probes.append(cells_of(1, range(lags_model_max)))
        if s_type != "N" and components_number_ets_seasonal > 0:
            for i in range(components_number_ets_seasonal):
                r = components_number_ets_non_seasonal + i
                for j in range(int(lags_model_all[r])):
                    probes.append(cells_of(r, [j]))
    if arima_model and components_number_arima > 0:
        for i in range(components_number_arima):
            r = components_number_ets + i
            for j in range(int(lags_model_all[r])):
                probes.append(cells_of(r, [j]))
    if not probes:
        return None

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
    loss_code,
    o_type="n",
):
    """Solve for the initial recent profile that minimises the estimation loss.

    Thin wrapper around the shared C++ ``gradientSolve``; ``loss_code`` is the
    ``(code, params)`` pair from :func:`adam_gradient_loss_code`. Returns the
    solved recent profile (same shape as ``profile``), or ``None`` on failure.
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
        lossType=loss_code[0],
        lossParams=np.asarray([loss_code[1]], dtype=np.float64),
        O=o_type,
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
    loss="MSE",
    distribution="dnorm",
    other=None,
    horizon=0,
    multisteps=False,
):
    """Fit dispatcher mirroring ``adam_fitOrGradient``.

    Runs the gradient initial-state solve when ``initial_type == "gradient"`` and
    the model is in scope (ETS, no ARIMA/xreg, a loss that maps onto a C++ loss
    code), otherwise the ordinary ``adam_cpp.fit`` (with the backcast flag).
    Gradient always fits with ``backcast=False`` from the solved initials and
    ``n_iterations=1``; it never runs the backward pass. Out-of-scope gradient
    models (including custom loss functions) fall back to backcasting.

    ``mat_vt`` is mutated in place (its head columns are overwritten with the
    solved profile) and the C++ fit result is returned.
    """
    is_gradient = initial_type == "gradient" or (
        isinstance(initial_type, (list, tuple)) and "gradient" in initial_type
    )
    if is_gradient:
        # Occurrence models profile their own losses over the probability
        # residuals; occurrence "fixed" ('f') has no estimated initials and
        # falls straight through to the backcasting fit.
        if o_type == "n":
            loss_code = adam_gradient_loss_code(
                loss,
                distribution,
                model_type_dict.get("error_type", "A"),
                other,
                horizon,
                multisteps,
            )
        elif o_type in ("d", "o", "i"):
            loss_code = adam_gradient_om_loss_code(loss)
        else:
            loss_code = None
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
            model_type_dict.get("error_type", "A"),
            int(components_dict.get("components_number_ets", 0) or 0),
            int(components_dict.get("components_number_arima", 0) or 0),
            lags_dict.get("lags_model_all", lags_dict["lags_model"]),
        )
        if probe_basis is not None and loss_code is not None:
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
                loss_code,
                o_type,
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
