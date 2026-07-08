"""``sim_ces`` — Python port of R's ``sim.ces`` (R/simces.R).

Generates Complex Exponential Smoothing series. The CES smoothing
parameters ``a`` and ``b`` are complex-valued; the simulator draws
``Re`` / ``Im`` uniformly and rejects values that fall outside the
"banana"-shaped stability region (R/simces.R:108-110) — three circle
conditions on the real / imaginary plane.

The CES state-space layout matches R/simces.R:160-197:

* ``none`` → 2 states (level, potential), lag 1
* ``simple`` → 2 states (seasonal level / potential), lag ``frequency``
* ``partial`` → 3 states (level, potential, seasonal)
* ``full`` → 4 states (level, potential, seasonal level / potential)

The ``E="A"`` simulator is used regardless of ``seasonality`` because
CES is an additive-error model.
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


def sim_ces(
    seasonality: str = "none",
    obs: int = 10,
    nsim: int = 1,
    frequency: int = 1,
    a: Optional[complex] = None,
    b: Optional[Union[complex, float]] = None,
    initial: Optional[Union[List[float], np.ndarray]] = None,
    randomizer: Union[str, Callable] = "rnorm",
    probability: Union[float, np.ndarray] = 1.0,
    seed: Optional[int] = None,
    **randomizer_kwargs,
) -> SimulateResult:
    """Simulate one or more CES series.

    Python port of R's ``sim.ces`` (``R/simces.R:78-471``).

    Parameters
    ----------
    seasonality : ``"none"`` | ``"simple"`` | ``"partial"`` | ``"full"``
        CES seasonality flavour (R/simces.R:160-197).
    obs, nsim, frequency : int
        Standard ``sim_*`` shape arguments.
    a : complex, optional
        Complex smoothing parameter for the level / potential pair.
        ``None`` → random draw from the stability region.
    b : complex or float, optional
        Smoothing parameter for the seasonal pair:

        * ``"partial"``: a single real value in ``(0, 1)``;
        * ``"full"``: a complex value drawn from the same stability
          region as ``a``;
        * ``"none"`` / ``"simple"``: ignored.
    initial : array-like, optional
        Length-``components_number × lag_max`` initial state vector.
        ``None`` → random ``Uniform(0, 1000)``.
    randomizer, probability, seed, **randomizer_kwargs
        See :func:`sim_es`.

    Returns
    -------
    SimulateResult
        ``a`` / ``b`` slots carry the complex smoothing parameters
        (vectors of length ``nsim``). ``model`` is ``"CES(seasonality)"``
        (prefixed with ``"i"`` for ``probability < 1``).
    """
    rng = np.random.default_rng(seed)

    valid = {"none", "simple", "partial", "full"}
    if seasonality not in valid:
        raise ValueError(
            f"Unknown seasonality {seasonality!r}. Must be one of {sorted(valid)}."
        )
    if seasonality != "none" and frequency == 1:
        raise ValueError("Cannot simulate seasonal CES with frequency=1.")

    nsim = abs(int(round(nsim)))
    obs = abs(int(round(obs)))
    frequency = abs(int(round(frequency)))

    # ---------- state-space layout (R/simces.R:160-197) -----------------
    if seasonality == "none":
        lags_model_max = 1
        lags_model = [1, 1]
        components_number = 2
        b_number = 0
        mat_wt_row = [1.0, 0.0]
    elif seasonality == "simple":
        lags_model_max = frequency
        lags_model = [lags_model_max, lags_model_max]
        components_number = 2
        b_number = 0
        mat_wt_row = [1.0, 0.0]
    elif seasonality == "partial":
        lags_model_max = frequency
        lags_model = [1, 1, lags_model_max]
        components_number = 3
        b_number = 1
        mat_wt_row = [1.0, 0.0, 1.0]
    else:  # full
        lags_model_max = frequency
        lags_model = [1, 1, lags_model_max, lags_model_max]
        components_number = 4
        b_number = 2
        mat_wt_row = [1.0, 0.0, 1.0, 0.0]

    mat_wt = np.tile(np.asarray(mat_wt_row, dtype=np.float64), (obs, 1))

    # ---------- complex smoothing parameter draws ------------------------
    a_generate = a is None
    b_generate = b is None and b_number > 0
    if a is not None and not _is_stable_complex(complex(a)):
        warnings.warn(
            "The provided complex smoothing parameter ``a`` leads to a "
            "non-stable CES model.",
            stacklevel=2,
        )
    if b is not None and seasonality == "full" and not _is_stable_complex(complex(b)):
        warnings.warn(
            "The provided complex smoothing parameter ``b`` leads to a "
            "non-stable CES model.",
            stacklevel=2,
        )
    if b is not None and seasonality == "partial":
        b_val = float(np.real(b))
        if not 0.0 <= b_val <= 1.0:
            warnings.warn(
                "Be careful with the provided ``b`` — value outside "
                "[0, 1] can cause an unstable CES model.",
                stacklevel=2,
            )

    a_mat = np.zeros((2, nsim), dtype=np.float64)
    b_mat = np.zeros((max(1, b_number), nsim), dtype=np.float64)

    if a_generate:
        for s in range(nsim):
            a_mat[:, s] = _draw_stable_complex(rng)
    else:
        assert a is not None
        a_c = complex(a)
        a_mat[0, :] = a_c.real
        a_mat[1, :] = a_c.imag

    if b_number > 0:
        if b_generate:
            if seasonality == "full":
                for s in range(nsim):
                    b_mat[:, s] = _draw_stable_complex(rng)
            else:  # partial
                b_mat[0, :] = rng.uniform(0.0, 1.0, nsim)
        else:
            assert b is not None
            if seasonality == "full":
                b_c = complex(b)
                b_mat[0, :] = b_c.real
                b_mat[1, :] = b_c.imag
            else:
                b_mat[0, :] = float(np.real(b))

    # ---------- transition + persistence (R/simces.R:295-311) ------------
    arr_f = np.zeros((components_number, components_number, nsim), dtype=np.float64)
    mat_g = np.zeros((components_number, nsim), dtype=np.float64)
    arr_f[0:2, 0, :] = 1.0
    for s in range(nsim):
        arr_f[0, 1, s] = a_mat[1, s] - 1.0
        arr_f[1, 1, s] = 1.0 - a_mat[0, s]
        mat_g[0, s] = a_mat[0, s] - a_mat[1, s]
        mat_g[1, s] = a_mat[0, s] + a_mat[1, s]
    if seasonality == "partial":
        arr_f[2, 2, :] = 1.0
        mat_g[2, :] = b_mat[0, :]
    elif seasonality == "full":
        arr_f[2:4, 2, :] = 1.0
        for s in range(nsim):
            arr_f[2, 3, s] = b_mat[1, s] - 1.0
            arr_f[3, 3, s] = 1.0 - b_mat[0, s]
            mat_g[2, s] = b_mat[0, s] - b_mat[1, s]
            mat_g[3, s] = b_mat[0, s] + b_mat[1, s]

    # ---------- initial states ------------------------------------------
    obs_states = obs + lags_model_max
    arr_vt = np.zeros((components_number, obs_states, nsim), dtype=np.float64)
    mat_initial = np.zeros((components_number, lags_model_max, nsim), dtype=np.float64)

    if initial is None:
        mat_initial = rng.uniform(
            0.0, 1000.0, (components_number, lags_model_max, nsim)
        )
        # For non-"none"/"simple" cases R repeats the level/potential
        # across the lag head (R/simces.R:257-259).
        if seasonality in ("partial", "full"):
            head_block = mat_initial[:2, -1:, :].repeat(lags_model_max, axis=1)
            mat_initial[:2, :, :] = head_block
    else:
        init_arr = np.asarray(initial, dtype=float).ravel()
        expected = components_number * lags_model_max
        if init_arr.size != expected:
            warnings.warn(
                f"Wrong length of initial vector ({init_arr.size}); "
                f"expected {expected}. Regenerating.",
                stacklevel=2,
            )
            mat_initial = rng.uniform(
                0.0, 1000.0, (components_number, lags_model_max, nsim)
            )
        else:
            # R fills ``matInitialValue[,1:lagsModelMax,]`` column-major
            # (R/simces.R:262), so each chunk of ``components_number``
            # values is one lag's snapshot across all components.
            block = init_arr.reshape(components_number, lags_model_max, order="F")
            mat_initial[:, :, :] = block[:, :, None]

    arr_vt[:components_number, :lags_model_max, :] = mat_initial

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

    # R-style centring + variance scaling for default randomizers
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

    # ---------- occurrence mask ------------------------------------------
    probability_arr = np.atleast_1d(np.asarray(probability, dtype=float)).ravel()
    if probability_arr.size == 1 and probability_arr[0] == 1.0:
        mat_ot = np.ones((obs, nsim), dtype=np.float64)
    else:
        if probability_arr.size == 1:
            p_vec = np.full(obs, probability_arr[0])
        else:
            p_vec = probability_arr[:obs]
        mat_ot = rng.binomial(1, p_vec[:, None], size=(obs, nsim)).astype(np.float64)

    # ---------- run the C++ kernel --------------------------------------
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
        nArima=components_number,  # treated as ARIMA-style components
        nXreg=0,
        constant=False,
    )
    mat_yt = np.asarray(result["matrixYt"], dtype=np.float64)
    arr_vt = np.asarray(result["arrayVt"], dtype=np.float64).reshape(
        arr_vt.shape, order="F"
    )

    # ---------- wrap output ---------------------------------------------
    a_complex = a_mat[0] + 1j * a_mat[1]
    b_complex: Optional[Union[complex, np.ndarray]]
    if seasonality in ("none", "simple"):
        b_complex = None
    elif seasonality == "full":
        b_complex = b_mat[0] + 1j * b_mat[1]
    else:  # partial
        b_complex = b_mat[0].astype(float)

    if nsim == 1:
        data_out: Union[pd.Series, pd.DataFrame] = pd.Series(mat_yt[:, 0])
        residuals_out: Union[pd.Series, pd.DataFrame] = pd.Series(mat_errors[:, 0])
        a_out: Union[complex, np.ndarray] = complex(np.asarray(a_complex)[0])
        b_out: Optional[Union[complex, np.ndarray, float]]
        if b_complex is not None and seasonality == "full":
            b_out = complex(np.asarray(b_complex)[0])
        elif b_complex is not None:  # partial
            b_out = float(np.asarray(b_complex)[0])
        else:
            b_out = None
    else:
        data_out = pd.DataFrame(mat_yt)
        residuals_out = pd.DataFrame(mat_errors)
        a_out = a_complex
        b_out = b_complex

    model_label = f"CES({seasonality})"
    if np.any(probability_arr != 1.0):
        model_label = "i" + model_label

    return SimulateResult(
        model=model_label,
        data=data_out,
        states=arr_vt,
        residuals=residuals_out,
        a=a_out,
        b=b_out,
        initial=mat_initial.reshape(-1, nsim),
        profile=profiles_recent_array,
        occurrence=mat_ot if not np.all(mat_ot == 1.0) else None,
        probability=probability_arr if probability_arr.size > 1 else None,
        intermittent=("none" if np.all(probability_arr == 1.0) else "tsb"),
        other=dict(randomizer_kwargs),
    )


def _is_stable_complex(z: complex) -> bool:
    """Stability "banana" region (R/simces.R:108-110)."""
    re, im = z.real, z.imag
    return (
        ((re - 2.5) ** 2 + im**2 > 1.25)
        and ((re - 0.5) ** 2 + (im - 1.0) ** 2 > 0.25)
        and ((re - 1.5) ** 2 + (im - 0.5) ** 2 < 1.5)
    )


def _draw_stable_complex(rng: np.random.Generator, max_tries: int = 1000) -> np.ndarray:
    """Reject-sample until the draw lands in the stability region.

    R/simces.R:100-115 draws ``Re ~ U(0.9, 2.5)`` and
    ``Im ~ U(0.9, 1.1)`` and retries until the three circle conditions
    are satisfied.
    """
    for _ in range(max_tries):
        re = rng.uniform(0.9, 2.5)
        im = rng.uniform(0.9, 1.1)
        if _is_stable_complex(complex(re, im)):
            return np.array([re, im])
    raise RuntimeError(
        f"Failed to draw a stable CES coefficient after {max_tries} tries."
    )
