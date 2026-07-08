"""``sim_oes`` — Python port of R's ``sim.oes`` (R/simoes.R).

Generates occurrence ETS series. Internally simulates two ETS
sub-models (A and B) via :func:`sim_es` and combines their fitted
data into a probability series via one of four occurrence schemes:
``"odds-ratio"``, ``"inverse-odds-ratio"``, ``"direct"``, ``"general"``.

The combination formulas mirror R/simoes.R:118-134. The R stub for
the unused sub-model is a list with ``data=matrix(0, …)``;
``errorType.default`` on such a stub returns ``"A"``, so R evaluates
``exp(0) = 1`` for the missing side. We make that "missing side = 1"
explicit in Python rather than carry the stub trick across the bridge.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd

from smooth.adam_general.core.simulate.es import sim_es
from smooth.adam_general.core.simulate.result import SimulateResult


def sim_oes(
    model: str = "MNN",
    obs: int = 10,
    nsim: int = 1,
    frequency: int = 1,
    occurrence: str = "odds-ratio",
    bounds: str = "usual",
    randomizer: Union[str, Callable] = "rlnorm",
    persistence: Optional[Union[List[float], np.ndarray]] = None,
    phi: float = 1.0,
    initial: Optional[Union[List[float], np.ndarray]] = None,
    initial_season: Optional[Union[List[float], np.ndarray]] = None,
    model_b: Optional[str] = None,
    persistence_b: Optional[Union[List[float], np.ndarray]] = None,
    phi_b: Optional[float] = None,
    initial_b: Optional[Union[List[float], np.ndarray]] = None,
    initial_season_b: Optional[Union[List[float], np.ndarray]] = None,
    seed: Optional[int] = None,
    **randomizer_kwargs,
) -> SimulateResult:
    """Simulate one or more occurrence ETS series.

    Python port of R's ``sim.oes`` (``R/simoes.R:84-152``). Wraps
    :func:`sim_es` twice (models A and B) and combines their data into
    an occurrence probability series.

    Parameters
    ----------
    model : str, default ``"MNN"``
        ETS taxonomy code for the **A** sub-model. Multiplicative-error
        ETS is the natural default — occurrence-rate magnitudes are
        positive.
    obs, nsim, frequency : int
        Standard ``sim_*`` shape arguments.
    occurrence : str
        ``"odds-ratio"`` | ``"inverse-odds-ratio"`` | ``"direct"`` | ``"general"``.
        Combination scheme:

        * ``"odds-ratio"``: only A simulated.
          ``prob = a / (1 + a)`` where ``a = data_A`` if A has
          multiplicative errors, else ``exp(data_A)``.
        * ``"inverse-odds-ratio"``: only B simulated.
          ``prob = 1 / (1 + b)`` (same transform rule for B).
        * ``"direct"``: only A simulated. ``prob = clip(data_A, 0, 1)``.
        * ``"general"``: both simulated. ``prob = 1 / (1 + b / a)``.
    bounds : ``"usual"`` | ``"admissible"`` | ``"restricted"``
        Forwarded to both sub-:func:`sim_es` calls.
    randomizer : str | callable, default ``"rlnorm"``
        Error randomizer, forwarded to both sub-models. The R default
        is ``"rlnorm"`` because occurrence-rate ETSes naturally use
        log-normal multiplicative noise.
    persistence, phi, initial, initial_season
        Parameters for the **A** sub-model. See :func:`sim_es`.
    model_b, persistence_b, phi_b, initial_b, initial_season_b
        Parameters for the **B** sub-model. ``None`` defaults to the
        corresponding A-side value (R/simoes.R:90).
    seed : int, optional
        Master seed. Sub-models receive deterministic derived seeds
        (``seed`` and ``seed + 1``) so the result is reproducible.
    **randomizer_kwargs
        Forwarded to both sub-:func:`sim_es` calls.

    Returns
    -------
    SimulateResult
        ``data`` holds the occurrence probability series (R's
        ``$probability``). ``model_a`` / ``model_b`` carry the sub-model
        :class:`SimulateResult` instances. ``occurrence_type`` echoes
        the chosen scheme. ``log_lik`` is the true log-likelihood
        (R/simoes.R:138-143).
    """
    valid_occ = {"odds-ratio", "inverse-odds-ratio", "direct", "general"}
    if occurrence not in valid_occ:
        raise ValueError(
            f"Unknown occurrence type {occurrence!r}. Must be one of "
            f"{sorted(valid_occ)}."
        )

    if model_b is None:
        model_b = model
    if persistence_b is None:
        persistence_b = persistence
    if phi_b is None:
        phi_b = phi
    if initial_b is None:
        initial_b = initial
    if initial_season_b is None:
        initial_season_b = initial_season

    # ---------- 1. simulate the relevant sub-models --------------------
    need_a = occurrence in {"odds-ratio", "direct", "general"}
    need_b = occurrence in {"inverse-odds-ratio", "general"}
    seed_a = seed
    seed_b = None if seed is None else seed + 1

    model_a_result: Optional[SimulateResult] = None
    model_b_result: Optional[SimulateResult] = None
    if need_a:
        model_a_result = sim_es(
            model=model,
            obs=obs,
            nsim=nsim,
            frequency=frequency,
            persistence=persistence,
            phi=phi,
            initial=initial,
            initial_season=initial_season,
            bounds=bounds,
            randomizer=randomizer,
            seed=seed_a,
            **randomizer_kwargs,
        )
    if need_b:
        model_b_result = sim_es(
            model=model_b,
            obs=obs,
            nsim=nsim,
            frequency=frequency,
            persistence=persistence_b,
            phi=phi_b,
            initial=initial_b,
            initial_season=initial_season_b,
            bounds=bounds,
            randomizer=randomizer,
            seed=seed_b,
            **randomizer_kwargs,
        )

    # ---------- 2. extract data matrices as (obs, nsim) -----------------
    def _data_matrix(sim: SimulateResult) -> np.ndarray:
        arr = sim.data.to_numpy()
        return arr.reshape(-1, 1) if arr.ndim == 1 else arr

    # ---------- 3. combine into probability -----------------------------
    if occurrence == "direct":
        assert model_a_result is not None
        a_data = _data_matrix(model_a_result)
        probability = np.clip(a_data, 0.0, 1.0)
    else:
        # ``a`` / ``b`` are the *positive* odds, transformed per error
        # type (M: identity; A: exp).
        a_pos: np.ndarray
        b_pos: np.ndarray
        if model_a_result is not None:
            a_data = _data_matrix(model_a_result)
            a_etype = model_a_result.model.split("(")[-1].rstrip(")")[0]
            a_pos = a_data if a_etype == "M" else np.exp(a_data)
        else:
            a_pos = np.ones((obs, nsim))
        if model_b_result is not None:
            b_data = _data_matrix(model_b_result)
            b_etype = model_b_result.model.split("(")[-1].rstrip(")")[0]
            b_pos = b_data if b_etype == "M" else np.exp(b_data)
        else:
            b_pos = np.ones((obs, nsim))
        # Guard against division by zero — when ``a_pos`` underflows the
        # ratio explodes and ``1 / (1 + huge)`` → 0, which is the right
        # answer. Clamp ``a_pos`` to ``eps`` only to avoid divide warnings.
        a_safe = np.where(a_pos == 0.0, np.finfo(float).tiny, a_pos)
        probability = 1.0 / (1.0 + b_pos / a_safe)

    # ---------- 4. log-likelihood (R/simoes.R:138-143) ------------------
    safe_prob = np.clip(probability, np.finfo(float).tiny, 1.0)
    log_lik = np.log(safe_prob).sum(axis=0)

    # ---------- 5. occurrence draw --------------------------------------
    # sim.oes does NOT draw 0/1 occurrence indicators in R — the
    # returned object carries the *probability* series. We mirror that:
    # ``data`` holds the probability, and any 0/1 mask is the caller's
    # responsibility (or a downstream ``oes`` fit's responsibility).

    # ---------- 6. label ------------------------------------------------
    a_name = model
    b_name = model_b
    if occurrence == "odds-ratio":
        model_label = f"oETS[O]({a_name})"
    elif occurrence == "inverse-odds-ratio":
        model_label = f"oETS[I]({b_name})"
    elif occurrence == "direct":
        model_label = f"oETS[D]({a_name})"
    else:  # general
        model_label = f"oETS[G]({a_name})({b_name})"

    if nsim == 1:
        data_out: Union[pd.Series, pd.DataFrame] = pd.Series(probability[:, 0])
    else:
        data_out = pd.DataFrame(probability)

    return SimulateResult(
        model=model_label,
        data=data_out,
        states=np.empty((0, 0, nsim)),  # OES doesn't carry a single state cube
        residuals=(
            pd.Series(np.zeros(obs))
            if nsim == 1
            else pd.DataFrame(np.zeros((obs, nsim)))
        ),
        probability=probability if nsim > 1 else probability[:, 0],
        log_lik=log_lik,
        model_a=model_a_result,
        model_b=model_b_result,
        occurrence_type=occurrence,
        other=dict(randomizer_kwargs),
    )
