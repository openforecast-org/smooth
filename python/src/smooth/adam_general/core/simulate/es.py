"""``sim_es`` — Python port of R's ``sim.es`` (R/simes.R).

Generates ETS series from a model specification, using the same C++
``adamCore::simulate`` kernel R does. Mirrors R parameter-for-parameter
(snake_cased) and keeps the R API's ``frequency`` parameter — the
no-``frequency`` rule in ``python/CLAUDE.md`` has an explicit carve-out
for the standalone ``sim_*`` simulators because they have no fitted
state to infer seasonality from.
"""

from __future__ import annotations

import warnings
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd

from smooth.adam_general._adam_general import adam_simulator
from smooth.adam_general.core.creator.architector import adam_profile_creator
from smooth.adam_general.core.simulate.randomizer import (
    is_default_randomizer,
    resolve_randomizer,
)
from smooth.adam_general.core.simulate.result import SimulateResult


def sim_es(
    model: str = "ANN",
    obs: int = 10,
    nsim: int = 1,
    frequency: int = 1,
    persistence: Optional[Union[List[float], np.ndarray]] = None,
    phi: float = 1.0,
    initial: Optional[Union[List[float], np.ndarray]] = None,
    initial_season: Optional[Union[List[float], np.ndarray]] = None,
    bounds: str = "usual",
    randomizer: Union[str, Callable] = "rnorm",
    probability: Union[float, np.ndarray] = 1.0,
    seed: Optional[int] = None,
    **randomizer_kwargs,
) -> SimulateResult:
    """Simulate one or more ETS series.

    Python port of R's ``sim.es`` (``R/simes.R:113-615``). Matches R
    parameter-for-parameter (snake_cased).

    Parameters
    ----------
    model : str, default ``"ANN"``
        ETS taxonomy code: ``E``-error, ``T``-trend, ``S``-seasonality.
        Three letters (e.g. ``"MAM"``) or four with a ``d`` in slot 3 for
        damped trend (e.g. ``"AAdN"``).
    obs : int, default ``10``
        Observations per simulated series.
    nsim : int, default ``1``
        Number of simulated series.
    frequency : int, default ``1``
        Seasonal period of the *generated* series. ``frequency=1`` means
        non-seasonal; seasonal models (``S != "N"``) require
        ``frequency > 1``. Carve-out from the no-``frequency`` rule in
        ``python/CLAUDE.md`` — ``sim_*`` is the only place this
        parameter is allowed.
    persistence : array-like, optional
        Smoothing parameters in the order ``(alpha, beta, gamma)``,
        truncated to the model's component count. ``None`` → R-style
        random draw under ``bounds``.
    phi : float, default ``1.0``
        Damping parameter (only used when ``T != "N"``).
    initial : array-like, optional
        Initial states ``(level, trend)`` truncated to the model's
        component count. ``None`` → random draw.
    initial_season : array-like, optional
        Initial seasonal states (length ``frequency``). ``None`` →
        random draw.
    bounds : ``"usual"`` | ``"admissible"`` | ``"restricted"``
        Bound type used when generating persistence draws (only
        consulted when ``persistence is None``).
    randomizer : str | callable, default ``"rnorm"``
        Error randomizer — see :func:`resolve_randomizer`. Accepts R
        names (``"rnorm"``, ``"rlnorm"``, ``"rt"``, ``"rlaplace"``,
        ``"rs"``, etc.) or a Python callable ``f(n) -> ndarray``.
    probability : float | array-like, default ``1.0``
        Occurrence probability (scalar or length-``obs`` vector). Below
        ``1`` produces an intermittent series.
    seed : int, optional
        Seed for the RNG used by string-named randomizers and by the
        occurrence mask draw. Ignored for callable randomizers (which
        own their randomness).
    **randomizer_kwargs
        Forwarded to the randomizer as R's ``...`` ellipsis.

    Returns
    -------
    SimulateResult
        Class-``"smooth.sim"``-equivalent container.
    """
    # ---------- 1. parse model code ------------------------------------
    if len(model) == 4:
        e_type, t_type, _, s_type = model[0], model[1], model[2], model[3]
        if model[2] != "d":
            warnings.warn(f"You have defined a strange model: {model}", stacklevel=2)
            model = e_type + t_type + "d" + s_type
        if t_type != "N" and phi == 1:
            model = e_type + t_type + s_type
            warnings.warn(
                f"Damping parameter is set to 1. Changing model to: {model}",
                stacklevel=2,
            )
    elif len(model) == 3:
        e_type, t_type, s_type = model[0], model[1], model[2]
        if phi != 1 and t_type != "N":
            model = e_type + t_type + "d" + s_type
            warnings.warn(
                f"Damping parameter is set to {phi}. Changing model to: {model}",
                stacklevel=2,
            )
    else:
        raise ValueError(f"You have defined a strange model: {model}. Cannot proceed")

    if e_type not in ("A", "M"):
        raise ValueError("Wrong error type! Should be 'A' or 'M'.")
    if t_type not in ("N", "A", "M"):
        raise ValueError("Wrong trend type! Should be 'N', 'A' or 'M'.")
    if s_type not in ("N", "A", "M"):
        raise ValueError("Wrong seasonality type! Should be 'N', 'A' or 'M'.")
    if s_type != "N" and frequency == 1:
        raise ValueError("Cannot create the seasonal model with the data frequency 1!")

    nsim = abs(int(round(nsim)))
    obs = abs(int(round(obs)))
    frequency = abs(int(round(frequency)))
    if not 0 <= phi <= 2:
        warnings.warn(
            f"Damping parameter should lie in (0, 2) region! You have chosen "
            f"phi={phi}. Be careful!",
            stacklevel=2,
        )

    # ---------- 2. assemble per-model state structure -------------------
    component_trend = t_type != "N"
    component_seasonal = s_type != "N"

    components_number = 1
    lags_model = [1]
    components_names = ["level"]
    mat_wt_row = [1.0]
    mat_f_blocks = [[1.0]]
    persistence_length = 1

    if component_trend:
        components_number += 1
        persistence_length += 1
        lags_model.append(1)
        components_names.append("trend")
        mat_wt_row.append(phi)
        mat_f_blocks = [[1.0, phi], [0.0, phi]]
    if component_seasonal:
        persistence_length += 1
        lags_model.append(frequency)
        components_names.append("seasonality")
        mat_wt_row.append(1.0)
        if not component_trend:
            mat_f_blocks = [[1.0, 0.0], [0.0, 1.0]]
        else:
            mat_f_blocks = [
                [1.0, phi, 0.0],
                [0.0, phi, 0.0],
                [0.0, 0.0, 1.0],
            ]

    mat_f = np.asarray(mat_f_blocks, dtype=np.float64)
    lags_model_max = max(lags_model)
    mat_wt = np.tile(np.asarray(mat_wt_row, dtype=np.float64), (obs, 1))

    # ---------- 3. validate / default persistence ----------------------
    persistence_arr: Optional[np.ndarray] = None
    if persistence is not None:
        persistence_arr = np.asarray(persistence, dtype=float).reshape(-1)
        if persistence_arr.shape[0] not in (1, persistence_length):
            warnings.warn(
                "The length of persistence vector does not correspond to the "
                "chosen model! Falling back to random number generator.",
                stacklevel=2,
            )
            persistence_arr = None
        elif persistence_arr.shape[0] == 1 and persistence_length > 1:
            persistence_arr = np.repeat(persistence_arr, persistence_length)

    # ---------- 4. validate / default initials -------------------------
    initial_arr: Optional[np.ndarray] = None
    if initial is not None:
        initial_arr = np.asarray(initial, dtype=float).reshape(-1)
        if initial_arr.shape[0] > 2:
            raise ValueError(
                "The length of the initial value is wrong! "
                "It should not be greater than 2."
            )
        if components_number != initial_arr.shape[0]:
            warnings.warn(
                "The length of initial state vector does not correspond to the "
                "chosen model! Falling back to random number generator.",
                stacklevel=2,
            )
            initial_arr = None
        elif t_type == "M" and initial_arr.shape[0] >= 2 and initial_arr[1] <= 0:
            warnings.warn(
                "Wrong initial value for multiplicative trend! "
                "It should be greater than zero! Falling back to random.",
                stacklevel=2,
            )
            initial_arr = None

    initial_season_arr: Optional[np.ndarray] = None
    if initial_season is not None:
        initial_season_arr = np.asarray(initial_season, dtype=float).reshape(-1)
        if lags_model_max != initial_season_arr.shape[0]:
            warnings.warn(
                "The length of seasonal initial states does not correspond "
                "to the chosen frequency! Falling back to random.",
                stacklevel=2,
            )
            initial_season_arr = None

    # ---------- 5. randomizer guard (R/simes.R:294-297) -----------------
    if (
        isinstance(randomizer, str)
        and randomizer not in {"rnorm", "rt", "rlaplace", "rs", "rlnorm"}
        and not randomizer_kwargs
    ):
        warnings.warn(
            f"The chosen randomizer - {randomizer} - needs some arbitrary "
            f"parameters! Changing to 'rnorm' now.",
            stacklevel=2,
        )
        randomizer = "rnorm"

    # ---------- 6. validate / coerce probability vector ----------------
    probability_arr = np.atleast_1d(np.asarray(probability, dtype=float)).reshape(-1)
    if probability_arr.size > 1:
        if not np.all(probability_arr == probability_arr[0]):
            if probability_arr.size != obs:
                warnings.warn(
                    "Length of probability does not correspond to number of "
                    "observations.",
                    stacklevel=2,
                )
                if probability_arr.size > obs:
                    warnings.warn("We will cut off the excessive ones.", stacklevel=2)
                    probability_arr = probability_arr[:obs]
                else:
                    warnings.warn("We will duplicate the last one.", stacklevel=2)
                    probability_arr = np.concatenate(
                        [
                            probability_arr,
                            np.full(obs - probability_arr.size, probability_arr[-1]),
                        ]
                    )
        else:
            probability_arr = probability_arr[:1]

    if probability_arr.size == 1:
        intermittent = "fixed"
    else:
        intermittent = "tsb"
    if np.all(probability_arr == 1.0):
        intermittent = "none"

    # ---------- 7. build state buffers ---------------------------------
    arr_vt = np.full(
        (persistence_length, obs + lags_model_max, nsim),
        np.nan,
        dtype=np.float64,
    )
    mat_g = np.full((persistence_length, nsim), np.nan, dtype=np.float64)
    mat_errors = np.full((obs, nsim), np.nan, dtype=np.float64)
    mat_ot = np.full((obs, nsim), np.nan, dtype=np.float64)

    rng = np.random.default_rng(seed)

    # ---------- 8. fill persistence -------------------------------------
    if persistence_arr is None:
        if bounds == "usual":
            mat_g[0, :] = rng.uniform(0.0, 1.0, nsim)
        elif bounds == "restricted":
            mat_g[0, :] = rng.uniform(0.0, 0.3, nsim)
        # admissible bounds: full ETS admissibility logic at R lines 362-401.
        # For Phase 1 we cover the non-admissible draw path used by 99% of
        # callers; admissible-bounds users typically pass an explicit
        # persistence vector.
        if bounds != "admissible":
            if component_trend:
                mat_g[1, :] = rng.uniform(0.0, mat_g[0, :], nsim)
            if component_seasonal:
                mat_g[persistence_length - 1, :] = rng.uniform(
                    0.0, np.maximum(0.0, 1.0 - mat_g[0, :]), nsim
                )
        else:
            mat_g[:, :] = rng.uniform(
                1.0 - 1.0 / phi, 1.0 + 1.0 / phi, (persistence_length, nsim)
            )
    else:
        mat_g[:, :] = np.repeat(persistence_arr[:, None], nsim, axis=1)

    # ---------- 9. fill initials ----------------------------------------
    if initial_arr is None:
        if t_type == "N":
            level_draw = rng.uniform(0.0, 1000.0, nsim)
            for s in range(nsim):
                arr_vt[0, :lags_model_max, s] = level_draw[s]
        elif t_type == "A":
            level_draw = rng.uniform(0.0, 5000.0, nsim)
            trend_draw = rng.uniform(-100.0, 100.0, nsim)
            for s in range(nsim):
                arr_vt[0, :lags_model_max, s] = level_draw[s]
                arr_vt[1, :lags_model_max, s] = trend_draw[s]
        else:  # multiplicative trend
            level_draw = rng.uniform(500.0, 5000.0, nsim)
            for s in range(nsim):
                arr_vt[0, :lags_model_max, s] = level_draw[s]
                arr_vt[1, :lags_model_max, s] = 1.0
    else:
        for s in range(nsim):
            for j in range(components_number):
                arr_vt[j, :lags_model_max, s] = initial_arr[j]
    initial_out = np.asarray(arr_vt[:components_number, 0, :].T)

    # ---------- 10. fill seasonal initials ------------------------------
    initial_season_out = None
    if component_seasonal and initial_season_arr is None:
        if s_type == "A":
            draws = rng.uniform(-500.0, 500.0, (lags_model_max, nsim))
            draws -= draws.mean(axis=0, keepdims=True)
            arr_vt[components_number, :lags_model_max, :] = draws
        else:
            draws = rng.uniform(0.3, 1.7, (lags_model_max, nsim))
            log_geomean = np.exp(np.log(draws).mean(axis=0, keepdims=True))
            arr_vt[components_number, :lags_model_max, :] = draws / log_geomean
        initial_season_out = np.asarray(arr_vt[components_number, :lags_model_max, :].T)
        components_number_plus = components_number + 1
    elif component_seasonal and initial_season_arr is not None:
        for s in range(nsim):
            arr_vt[components_number, :lags_model_max, s] = initial_season_arr
        initial_season_out = np.asarray(arr_vt[components_number, :lags_model_max, :].T)
        components_number_plus = components_number + 1
    else:
        components_number_plus = components_number

    # ---------- 11. draw errors -----------------------------------------
    is_default = is_default_randomizer(randomizer)
    n_errors = obs * nsim
    if is_default and not randomizer_kwargs:
        # R/simes.R:459-469 — default randomizers use ``rng(n)`` with the
        # R defaults; ``rlnorm`` and ``rt`` have model-aware shifts.
        rstr = randomizer
        if rstr in ("rnorm", "rlaplace", "rs"):
            errors_flat = resolve_randomizer(rstr, rng)(n_errors)
        elif rstr == "rt":
            df = max(obs - (persistence_length + lags_model_max), 1)
            errors_flat = rng.standard_t(df, n_errors)
        elif rstr == "rlnorm":
            sd = 0.01 + float(
                (1.0 - probability_arr.mean()) if probability_arr.size > 0 else 0.0
            )
            errors_flat = rng.lognormal(0.0, sd, n_errors) - 1.0
        else:
            errors_flat = resolve_randomizer(rstr, rng)(n_errors)
    else:
        errors_flat = resolve_randomizer(randomizer, rng, **randomizer_kwargs)(n_errors)

    mat_errors[:, :] = errors_flat.reshape((obs, nsim), order="F")

    # ---------- 12. apply R's error-type scaling ------------------------
    if is_default and not randomizer_kwargs and randomizer != "rlnorm":
        if e_type == "M":
            scale = 0.5 if np.any(probability_arr != 1.0) else 0.1
            mat_errors *= scale
            mat_errors = np.exp(mat_errors) - 1.0
        else:  # additive errors get a level-aware scaling
            if np.all(arr_vt[0, 0, :] != 0.0):
                mat_errors *= np.sqrt(np.abs(arr_vt[0, 0, :]))
            if randomizer == "rs":
                mat_errors /= 4.0
    elif randomizer_kwargs:
        if randomizer == "rbeta":
            mat_errors -= 0.5
            col_rms = np.sqrt((mat_errors**2).mean(axis=0))
            mat_errors /= (col_rms * np.sqrt(np.abs(arr_vt[0, 0, :])))[None, :]
        elif randomizer == "rt":
            mat_errors *= np.sqrt(np.abs(arr_vt[0, 0, :]))[None, :]
        if e_type == "M":
            mat_errors -= 1.0

    # ---------- 13. occurrence mask -------------------------------------
    if np.all(probability_arr == 1.0):
        mat_ot[:, :] = 1.0
    else:
        if probability_arr.size == 1:
            p = np.full(obs, probability_arr[0])
        else:
            p = probability_arr
        mat_ot[:, :] = rng.binomial(1, p[:, None], size=(obs, nsim)).astype(float)

    # ---------- 14. drive the C++ kernel --------------------------------
    components_number_arima = 0
    components_number_ets = components_number_plus
    components_number_ets_seasonal = 1 if component_seasonal else 0
    components_number_ets_non_seasonal = (
        components_number_ets - components_number_ets_seasonal
    )
    xreg_number = 0
    arr_f = np.repeat(mat_f[:, :, None], nsim, axis=2)

    lags_array = np.asarray(lags_model, dtype=np.uint64)
    profiles = adam_profile_creator(
        lags_model_all=lags_array.tolist(),
        lags_model_max=lags_model_max,
        obs_all=obs,
    )
    index_lookup_table = profiles["index_lookup_table"]
    profiles_recent_array = np.ascontiguousarray(
        arr_vt[:, :lags_model_max, :], dtype=np.float64
    )

    result = adam_simulator(
        matrixErrors=mat_errors,
        matrixOt=mat_ot,
        arrayVt=arr_vt,
        matrixWt=mat_wt,
        arrayF=arr_f,
        matrixG=mat_g,
        lags=lags_array,
        indexLookupTable=index_lookup_table,
        profilesRecent=profiles_recent_array,
        E=e_type,
        T=t_type,
        S=s_type,
        nNonSeasonal=components_number_ets_non_seasonal,
        nSeasonal=components_number_ets_seasonal,
        nArima=components_number_arima,
        nXreg=xreg_number,
        constant=False,
    )
    mat_yt = np.asarray(result["matrixYt"])
    arr_vt = np.asarray(result["arrayVt"]).reshape(arr_vt.shape, order="F")

    # ---------- 15. likelihood (R/simes.R:560-580) ----------------------
    log_lik = _true_log_lik(
        randomizer if isinstance(randomizer, str) else None,
        mat_errors,
        mat_yt,
        obs,
    )

    # ---------- 16. wrap output -----------------------------------------
    if nsim == 1:
        data_out = pd.Series(mat_yt[:, 0])
        residuals_out: Union[pd.Series, pd.DataFrame] = pd.Series(mat_errors[:, 0])
    else:
        data_out = pd.DataFrame(mat_yt)
        residuals_out = pd.DataFrame(mat_errors)

    damp_letter = "d" if phi != 1 and t_type != "N" else ""
    model_label = f"ETS({e_type}{t_type}{damp_letter}{s_type})"
    if np.any(probability_arr != 1.0):
        model_label = "i" + model_label

    # Name persistence rows R-style for round-trippable __repr__.
    persistence_out = np.asarray(mat_g)

    return SimulateResult(
        model=model_label,
        data=data_out,
        states=arr_vt,
        residuals=residuals_out,
        occurrence=mat_ot if intermittent != "none" else None,
        probability=probability_arr if intermittent != "none" else None,
        persistence=persistence_out,
        phi=float(phi),
        initial=initial_out,
        initial_season=initial_season_out,
        profile=profiles_recent_array,
        intermittent=intermittent,
        log_lik=log_lik,
        other=dict(randomizer_kwargs),
    )


def _true_log_lik(
    randomizer_name: Optional[str],
    mat_errors: np.ndarray,
    mat_yt: np.ndarray,
    obs: int,
) -> Optional[np.ndarray]:
    """Per-series true log-likelihood, matching R/simes.R:560-580."""
    if randomizer_name is None:
        return None
    log2pi_p1 = np.log(2 * np.pi * np.e)
    if randomizer_name in ("rnorm", "rt"):
        col_mse = (mat_errors**2).mean(axis=0)
        return -obs / 2.0 * (log2pi_p1 + np.log(col_mse))
    if randomizer_name == "rlaplace":
        return -obs * (np.log(2 * np.e) + np.log(np.abs(mat_errors).mean(axis=0)))
    if randomizer_name == "rs":
        return (
            -2
            * obs
            * (
                np.log(2 * np.e)
                + np.log(0.5 * np.sqrt(np.abs(mat_errors)).mean(axis=0))
            )
        )
    if randomizer_name == "rlnorm":
        col_mse = (mat_errors**2).mean(axis=0)
        return -obs / 2.0 * (log2pi_p1 + np.log(col_mse)) - np.log(
            np.maximum(mat_yt, 1e-300)
        ).sum(axis=0)
    if randomizer_name == "rinvgauss":
        return -0.5 * (
            obs
            * (
                np.log((mat_errors**2 / (1.0 + mat_errors)).mean(axis=0) / (2 * np.pi))
                - 1
            )
            + np.log(np.maximum(mat_yt / (1.0 + mat_errors), 1e-300)).sum(axis=0)
            - 3 * np.log(np.maximum(mat_yt, 1e-300)).sum(axis=0)
        )
    return None
