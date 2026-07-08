"""``sim_gum`` — Python port of R's ``sim.gum`` (R/simgum.R).

Generates Generalised Unobserved-components series. The state-space
structure is arbitrary — the caller can supply ``measurement``,
``transition`` and ``persistence`` matrices or let them be drawn
randomly from a stability-rejecting sampler. Stability is enforced via
``smooth_eigens``: the discount-matrix eigenvalues must all lie inside
the unit circle (R/simgum.R:104-111).
"""

from __future__ import annotations

import warnings
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd

from smooth.adam_general._adam_general import adam_simulator
from smooth.adam_general._eigenCalc import smooth_eigens
from smooth.adam_general.core.creator.architector import adam_profile_creator
from smooth.adam_general.core.simulate.randomizer import (
    is_default_randomizer,
    resolve_randomizer,
)
from smooth.adam_general.core.simulate.result import SimulateResult


def sim_gum(
    orders: Union[int, List[int]] = 1,
    lags: Union[int, List[int]] = 1,
    obs: int = 10,
    nsim: int = 1,
    frequency: int = 1,
    measurement: Optional[Union[List[float], np.ndarray]] = None,
    transition: Optional[Union[List[float], np.ndarray]] = None,
    persistence: Optional[Union[List[float], np.ndarray]] = None,
    initial: Optional[Union[List[float], np.ndarray]] = None,
    randomizer: Union[str, Callable] = "rnorm",
    probability: Union[float, np.ndarray] = 1.0,
    seed: Optional[int] = None,
    **randomizer_kwargs,
) -> SimulateResult:
    """Simulate one or more GUM series.

    Python port of R's ``sim.gum`` (``R/simgum.R:76-436``).

    Parameters
    ----------
    orders, lags : sequence of int
        ``orders[k]`` copies of state at lag ``lags[k]``. The total
        number of states is ``sum(orders)``.
    obs, nsim, frequency : int
        Standard ``sim_*`` shape arguments.
    measurement : array-like, optional
        Length ``components_number`` measurement vector. ``None`` →
        random ``Uniform(0, 1)`` draw.
    transition : array-like, optional
        ``components_number × components_number`` transition matrix.
        ``None`` → random ``Uniform(-1, 1)`` draw, rejected until the
        discount-matrix eigenvalues fall inside the unit circle.
    persistence : array-like, optional
        Length ``components_number`` persistence vector. ``None`` →
        random ``Uniform(-1, 1)`` draw under the same stability check.
    initial : array-like, optional
        Length ``components_number × lag_max`` initial state vector.
        ``None`` → random ``Uniform(0, 1000)``.
    randomizer, probability, seed, **randomizer_kwargs
        See :func:`sim_es`.

    Returns
    -------
    SimulateResult
        ``measurement`` / ``transition`` / ``persistence`` / ``initial``
        carry the matrices actually used. ``model`` is
        ``"GUM(o1[L1],o2[L2],...)"`` (``"i"``-prefixed when
        ``probability < 1``).
    """
    rng = np.random.default_rng(seed)

    orders_arr = np.atleast_1d(np.asarray(orders)).ravel().astype(int)
    lags_arr = np.atleast_1d(np.asarray(lags)).ravel().astype(int)
    if np.any(orders_arr < 0):
        raise ValueError("Negative orders are not allowed.")
    if np.any(lags_arr < 0):
        raise ValueError("Negative lags are not allowed.")
    if orders_arr.size != lags_arr.size:
        raise ValueError(
            f"length(orders)={orders_arr.size} != length(lags)={lags_arr.size}."
        )

    # Drop zero entries (R/simgum.R:132-141).
    keep = lags_arr != 0
    orders_arr = orders_arr[keep]
    lags_arr = lags_arr[keep]
    keep = orders_arr != 0
    orders_arr = orders_arr[keep]
    lags_arr = lags_arr[keep]

    # De-duplicate lags by taking max order.
    if np.unique(lags_arr).size != lags_arr.size:
        unique_lags = np.unique(lags_arr)
        new_orders = np.array(
            [orders_arr[lags_arr == lag].max() for lag in unique_lags], dtype=int
        )
        orders_arr = new_orders
        lags_arr = unique_lags

    # Expand the lags vector: ``orders[k]`` copies of ``lags[k]``.
    lags_model = np.repeat(lags_arr, orders_arr).astype(int)
    lags_model_max = int(lags_model.max())
    components_number = int(orders_arr.sum())

    nsim = abs(int(round(nsim)))
    obs = abs(int(round(obs)))
    frequency = abs(int(round(frequency)))
    obs_states = obs + lags_model_max

    # ---------- validate supplied matrices -----------------------------
    m_arr: Optional[np.ndarray] = None
    t_arr: Optional[np.ndarray] = None
    p_arr: Optional[np.ndarray] = None
    init_arr: Optional[np.ndarray] = None

    measurement_generate = measurement is None
    if measurement is not None:
        m_arr = np.asarray(measurement, dtype=float).ravel()
        if m_arr.size != components_number:
            warnings.warn(
                f"Wrong length of measurement vector ({m_arr.size}); "
                f"expected {components_number}. Regenerating.",
                stacklevel=2,
            )
            measurement_generate = True
            m_arr = None

    transition_generate = transition is None
    if transition is not None:
        t_flat = np.asarray(transition, dtype=float).ravel()
        if t_flat.size != components_number * components_number:
            warnings.warn(
                f"Wrong size of transition matrix ({t_flat.size}); "
                f"expected {components_number**2}. Regenerating.",
                stacklevel=2,
            )
            transition_generate = True
        else:
            # R's ``arrF[] <- transitionValue`` is column-major (Fortran
            # order); preserve that for caller-supplied flat vectors.
            t_arr = t_flat.reshape(components_number, components_number, order="F")

    persistence_generate = persistence is None
    if persistence is not None:
        p_arr = np.asarray(persistence, dtype=float).ravel()
        if p_arr.size != components_number:
            warnings.warn(
                f"Wrong length of persistence vector ({p_arr.size}); "
                f"expected {components_number}. Regenerating.",
                stacklevel=2,
            )
            persistence_generate = True
            p_arr = None

    initial_generate = initial is None
    if initial is not None:
        init_arr = np.asarray(initial, dtype=float).ravel()
        expected = components_number * lags_model_max
        if init_arr.size != expected:
            warnings.warn(
                f"Wrong length of initial vector ({init_arr.size}); "
                f"expected {expected}. Regenerating.",
                stacklevel=2,
            )
            initial_generate = True
            init_arr = None

    # ---------- build per-sim matrices ----------------------------------
    arr_vt = np.zeros((components_number, obs_states, nsim), dtype=np.float64)
    arr_f = np.zeros((components_number, components_number, nsim), dtype=np.float64)
    mat_g = np.zeros((components_number, nsim), dtype=np.float64)
    mat_initial = np.zeros((components_number, lags_model_max, nsim), dtype=np.float64)

    if initial_generate:
        mat_initial = rng.uniform(
            0.0,
            1000.0,
            (components_number, lags_model_max, nsim),
        )
    else:
        # R's column-major fill (R/simgum.R:271-272); each chunk of
        # ``components_number`` values is one lag's snapshot.
        mat_initial[:, :, :] = init_arr.reshape(  # type: ignore[union-attr]
            components_number, lags_model_max, order="F"
        )[:, :, None]
    arr_vt[:, :lags_model_max, :] = mat_initial

    # measurement is shared across sims unless we're generating per-sim
    # (R draws a single matWt per call when measurement_generate=True,
    # see R/simgum.R:92).
    mat_wt = np.zeros((obs, components_number), dtype=np.float64)
    if measurement_generate:
        mat_wt[:] = rng.uniform(0.0, 1.0, components_number)
    else:
        assert m_arr is not None
        mat_wt[:] = m_arr

    if not transition_generate:
        assert t_arr is not None
        arr_f[:] = t_arr[:, :, None]
    if not persistence_generate:
        assert p_arr is not None
        mat_g[:] = p_arr[:, None]

    if transition_generate or persistence_generate or measurement_generate:
        # Rejection-sample until the discount-matrix eigenvalues fall
        # inside the unit circle. R/simgum.R:88-115.
        for s in range(nsim):
            for _try in range(2000):
                if transition_generate:
                    arr_f[:, :, s] = rng.uniform(
                        -1.0, 1.0, (components_number, components_number)
                    )
                if persistence_generate:
                    mat_g[:, s] = rng.uniform(-1.0, 1.0, components_number)
                eig = smooth_eigens(
                    persistence=np.asfortranarray(
                        mat_g[:, s].reshape(-1, 1), dtype=np.float64
                    ),
                    transition=np.asfortranarray(arr_f[:, :, s], dtype=np.float64),
                    measurement=np.asfortranarray(mat_wt, dtype=np.float64),
                    lags_model_all=np.asarray(lags_model, dtype=np.int32),
                    xreg_model=False,
                    obs_in_sample=int(obs),
                    has_delta=False,
                    xreg_number=0,
                    constant_required=False,
                )
                if np.all(np.abs(np.asarray(eig)) <= 1.0):
                    break
            else:
                raise RuntimeError(
                    f"Failed to draw a stable GUM transition/persistence "
                    f"after 2000 tries (sim {s})."
                )

    # ---------- errors --------------------------------------------------
    is_default = is_default_randomizer(randomizer)
    n_errors = obs * nsim
    if is_default and not randomizer_kwargs:
        rstr = randomizer
        if rstr in ("rnorm", "rlaplace", "rs"):
            errors_flat = resolve_randomizer(rstr, rng)(n_errors)
        elif rstr == "rt":
            df = max(obs - (components_number + lags_model_max), 1)
            errors_flat = rng.standard_t(df, n_errors)
        else:
            errors_flat = resolve_randomizer(rstr, rng)(n_errors)
    else:
        errors_flat = resolve_randomizer(randomizer, rng, **randomizer_kwargs)(n_errors)
    mat_errors = np.asarray(errors_flat, dtype=np.float64).reshape(
        (obs, nsim), order="F"
    )

    if is_default and not randomizer_kwargs:
        mat_errors -= mat_errors.mean(axis=0, keepdims=True)
        scale = np.sqrt(np.abs(arr_vt[0, :lags_model_max, :].mean(axis=0)))
        mat_errors *= scale[None, :]
        if randomizer == "rs":
            mat_errors /= 4.0
    elif randomizer_kwargs:
        if randomizer == "rbeta":
            mat_errors -= 0.5
            col_rms = np.sqrt((mat_errors**2).mean(axis=0))
            col_scale = np.sqrt(np.abs(arr_vt[0, :lags_model_max, :].mean(axis=0)))
            mat_errors /= (col_rms * col_scale)[None, :]
        elif randomizer == "rt":
            mat_errors *= np.sqrt(np.abs(arr_vt[0, :lags_model_max, :].mean(axis=0)))[
                None, :
            ]

    # ---------- occurrence ----------------------------------------------
    probability_arr = np.atleast_1d(np.asarray(probability, dtype=float)).ravel()
    if probability_arr.size == 1 and probability_arr[0] == 1.0:
        mat_ot = np.ones((obs, nsim), dtype=np.float64)
    else:
        if probability_arr.size == 1:
            p_vec = np.full(obs, probability_arr[0])
        else:
            p_vec = probability_arr[:obs]
        mat_ot = rng.binomial(1, p_vec[:, None], size=(obs, nsim)).astype(np.float64)

    # ---------- drive C++ ------------------------------------------------
    lags_model_arr = np.asarray(lags_model, dtype=np.uint64)
    profiles = adam_profile_creator(
        lags_model_all=lags_model_arr.tolist(),
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
        lags=lags_model_arr,
        indexLookupTable=index_lookup_table,
        profilesRecent=profiles_recent_array,
        E="A",
        T="N",
        S="N",
        nNonSeasonal=0,
        nSeasonal=0,
        nArima=components_number,
        nXreg=0,
        constant=False,
    )
    mat_yt = np.asarray(result["matrixYt"], dtype=np.float64)
    arr_vt = np.asarray(result["arrayVt"], dtype=np.float64).reshape(
        arr_vt.shape, order="F"
    )

    # ---------- wrap output ---------------------------------------------
    if nsim == 1:
        data_out: Union[pd.Series, pd.DataFrame] = pd.Series(mat_yt[:, 0])
        residuals_out: Union[pd.Series, pd.DataFrame] = pd.Series(mat_errors[:, 0])
    else:
        data_out = pd.DataFrame(mat_yt)
        residuals_out = pd.DataFrame(mat_errors)

    model_label = (
        "GUM("
        + ",".join(f"{int(o)}[{int(L)}]" for o, L in zip(orders_arr, lags_arr))
        + ")"
    )
    if np.any(probability_arr != 1.0):
        model_label = "i" + model_label

    return SimulateResult(
        model=model_label,
        data=data_out,
        states=arr_vt,
        residuals=residuals_out,
        measurement=mat_wt,
        transition=arr_f if nsim > 1 else arr_f[:, :, 0],
        persistence=mat_g,
        initial=mat_initial.reshape(-1, nsim),
        profile=profiles_recent_array,
        occurrence=mat_ot if not np.all(mat_ot == 1.0) else None,
        probability=probability_arr if probability_arr.size > 1 else None,
        intermittent=("none" if np.all(probability_arr == 1.0) else "tsb"),
        other=dict(randomizer_kwargs),
    )
