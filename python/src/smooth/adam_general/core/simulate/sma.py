"""``sim_sma`` — Python port of R's ``sim.sma`` (R/simsma.R).

A thin wrapper that delegates to :func:`sim_ssarima` with
``orders={"ar": order, "i": 0, "ma": 0}`` and ``arma={"ar":
[1/order]*order}`` — the SMA(``order``) representation in
state-space form.
"""

from __future__ import annotations

import warnings
from typing import Callable, List, Optional, Union

import numpy as np

from smooth.adam_general.core.simulate.result import SimulateResult
from smooth.adam_general.core.simulate.ssarima import sim_ssarima


def sim_sma(
    order: Optional[int] = None,
    obs: int = 10,
    nsim: int = 1,
    frequency: int = 1,
    initial: Optional[Union[List[float], np.ndarray]] = None,
    randomizer: Union[str, Callable] = "rnorm",
    probability: Union[float, np.ndarray] = 1.0,
    seed: Optional[int] = None,
    **randomizer_kwargs,
) -> SimulateResult:
    """Simulate one or more SMA series.

    Python port of R's ``sim.sma`` (``R/simsma.R:48-91``).

    Parameters
    ----------
    order : int, optional
        SMA order. ``None`` → R-style random draw in ``1..100``.
    obs, nsim, frequency, initial, randomizer, probability, seed
        See :func:`sim_es`.
    **randomizer_kwargs
        Forwarded to the randomizer.

    Returns
    -------
    SimulateResult
        ``model`` is set to ``"SMA(<order>)"`` (or ``"iSMA(<order>)"``
        when ``probability < 1``); ``arma`` and ``constant`` fields
        cleared per R's post-processing.
    """
    rng = np.random.default_rng(seed)
    if order is None:
        order = int(np.ceil(rng.uniform(0.0, 100.0)))
    order = int(abs(round(order)))
    if order < 1:
        raise ValueError("SMA order must be >= 1.")

    if initial is not None:
        initial_arr = np.asarray(initial, dtype=float).ravel()
        if initial_arr.size != order:
            warnings.warn(
                "The length of initial state vector does not correspond "
                "to the chosen SMA order! Falling back to random.",
                stacklevel=2,
            )
            initial_arr_for_sim: Optional[np.ndarray] = None
        else:
            initial_arr_for_sim = initial_arr
    else:
        initial_arr_for_sim = None

    sma_result = sim_ssarima(
        orders={"ar": [order], "i": [0], "ma": [0]},
        lags=[1],
        obs=obs,
        nsim=nsim,
        frequency=frequency,
        arma={"ar": [1.0 / order] * order, "ma": []},
        constant=False,
        initial=initial_arr_for_sim,
        bounds="none",
        randomizer=randomizer,
        probability=probability,
        seed=seed,
        **randomizer_kwargs,
    )

    # R's post-processing: strip arma / constant, re-label model.
    label = f"SMA({order})"
    if isinstance(probability, (int, float)) and probability != 1.0:
        label = "i" + label
    elif isinstance(probability, (list, np.ndarray)) and np.any(
        np.asarray(probability) != 1.0
    ):
        label = "i" + label
    sma_result.model = label
    sma_result.arma = None
    sma_result.constant = None
    return sma_result
