"""``sim_ssarima`` — Python port of R's ``sim.ssarima`` (R/simssarima.R).

Generates state-space SARIMA series via the shared C++
``adamCore::simulate`` kernel. The AR / MA polynomial fill mirrors the
inner ``creator()`` closure in R: ARI coefficients go into the first
column of the transition matrix, MA coefficients into the persistence
vector. Stability constraints on randomly-drawn AR / MA coefficients
are enforced via the polynomial roots (``|root| > 1``), matching R's
``polyroot`` check.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from smooth.adam_general import _adamCore
from smooth.adam_general._adam_general import adam_simulator
from smooth.adam_general.core.creator.architector import adam_profile_creator
from smooth.adam_general.core.simulate.randomizer import (
    is_default_randomizer,
    resolve_randomizer,
)
from smooth.adam_general.core.simulate.result import SimulateResult
from smooth.adam_general.core.utils.polynomials import adam_polynomialiser


def sim_ssarima(
    orders: Optional[Union[Dict[str, Any], List[int]]] = None,
    lags: Union[int, List[int]] = 1,
    obs: int = 10,
    nsim: int = 1,
    frequency: int = 1,
    arma: Optional[Dict[str, Union[List[float], np.ndarray]]] = None,
    constant: Union[bool, float] = False,
    initial: Optional[Union[List[float], np.ndarray]] = None,
    bounds: str = "admissible",
    randomizer: Union[str, Callable] = "rnorm",
    probability: Union[float, np.ndarray] = 1.0,
    seed: Optional[int] = None,
    **randomizer_kwargs,
) -> SimulateResult:
    """Simulate one or more SARIMA series.

    Python port of R's ``sim.ssarima`` (``R/simssarima.R:83-752``).
    Matches R parameter-for-parameter (snake_cased).

    Parameters
    ----------
    orders : dict | list-like, optional
        ARIMA orders. A dict ``{"ar": ..., "i": ..., "ma": ...}`` (each
        entry is a list aligned with ``lags``) or a length-3 sequence
        ``(p, d, q)`` for the non-seasonal case. Defaults to
        ``{"ar": 0, "i": 1, "ma": 1}`` (the IMA(1,1) "random walk plus
        noise" of R's default).
    lags : int | list of int, default ``1``
        Lag vector. ``[1]`` for non-seasonal; ``[1, 12]`` for
        non-seasonal + monthly seasonal SARIMA.
    obs : int, default ``10``
        Observations per simulated series.
    nsim : int, default ``1``
        Number of simulated series.
    frequency : int, default ``1``
        Seasonal period of the *generated* series. See the
        ``sim_es`` carve-out in ``python/CLAUDE.md``.
    arma : dict, optional
        Pre-specified AR / MA parameters as
        ``{"ar": [...], "ma": [...]}``. ``None`` triggers random
        admissible draws.
    constant : bool | float, default ``False``
        ``False``: no constant. ``True``: draw a random constant.
        Numeric: use the given value.
    initial : array-like, optional
        Initial state vector (length ``components_number``). ``None``
        triggers a random draw with a burn-in period of ``max(lags)``.
    bounds : ``"admissible"`` | ``"none"``, default ``"admissible"``
        Stability constraint on randomly-drawn AR / MA coefficients
        (``"admissible"`` rejects draws whose polynomial roots fall
        inside the unit circle).
    randomizer, probability, seed, **randomizer_kwargs
        See :func:`sim_es`.

    Returns
    -------
    SimulateResult
    """
    rng = np.random.default_rng(seed)

    # ---------- 1. parse orders + lags ----------------------------------
    if orders is None:
        orders = {"ar": 0, "i": 1, "ma": 1}
    if isinstance(orders, dict):
        ar_orders = np.atleast_1d(np.asarray(orders.get("ar", 0))).ravel()
        i_orders = np.atleast_1d(np.asarray(orders.get("i", 0))).ravel()
        ma_orders = np.atleast_1d(np.asarray(orders.get("ma", 0))).ravel()
    else:
        seq = list(orders)  # type: ignore[arg-type]
        if len(seq) < 3:
            raise ValueError("orders sequence must have length 3 (p, d, q).")
        ar_orders = np.atleast_1d(seq[0]).ravel()
        i_orders = np.atleast_1d(seq[1]).ravel()
        ma_orders = np.atleast_1d(seq[2]).ravel()

    lags_arr = np.atleast_1d(np.asarray(lags)).ravel().astype(int)

    if np.any(ar_orders < 0) or np.any(i_orders < 0) or np.any(ma_orders < 0):
        raise ValueError("Negative ARIMA orders are not allowed.")
    if np.any(lags_arr < 0):
        raise ValueError("Negative lags are not allowed.")

    n_lags_inputs = {len(ar_orders), len(i_orders), len(ma_orders)}
    if len(lags_arr) not in n_lags_inputs:
        raise ValueError("Lag length does not match any of the order vectors.")

    # Drop zero lags
    if np.any(lags_arr == 0):
        keep = lags_arr != 0
        ar_orders = ar_orders[keep] if len(ar_orders) == len(lags_arr) else ar_orders
        i_orders = i_orders[keep] if len(i_orders) == len(lags_arr) else i_orders
        ma_orders = ma_orders[keep] if len(ma_orders) == len(lags_arr) else ma_orders
        lags_arr = lags_arr[keep]

    # Pad order vectors to a common length
    max_order_len = max(len(ar_orders), len(i_orders), len(ma_orders))
    if len(ar_orders) < max_order_len:
        ar_orders = np.concatenate(
            [ar_orders, np.zeros(max_order_len - len(ar_orders), dtype=int)]
        )
    if len(i_orders) < max_order_len:
        i_orders = np.concatenate(
            [i_orders, np.zeros(max_order_len - len(i_orders), dtype=int)]
        )
    if len(ma_orders) < max_order_len:
        ma_orders = np.concatenate(
            [ma_orders, np.zeros(max_order_len - len(ma_orders), dtype=int)]
        )

    # Drop entries where all three orders are zero
    nonzero_mask = (ar_orders + i_orders + ma_orders) != 0
    if not np.any(nonzero_mask):
        # Edge case: nothing to model, fall back to white noise (R behavior).
        nonzero_mask = lags_arr == lags_arr.min()
    ar_orders = ar_orders[nonzero_mask]
    i_orders = i_orders[nonzero_mask]
    ma_orders = ma_orders[nonzero_mask]
    lags_arr = lags_arr[nonzero_mask]

    # De-duplicate lags by taking max order per lag
    if len(np.unique(lags_arr)) != len(lags_arr):
        if frequency != 1:
            warnings.warn(
                "'lags' contains duplicates; collapsing by max order.", stacklevel=2
            )
        unique_lags = np.unique(lags_arr)
        new_ar = np.zeros(len(unique_lags), dtype=int)
        new_i = np.zeros(len(unique_lags), dtype=int)
        new_ma = np.zeros(len(unique_lags), dtype=int)
        for k, lag in enumerate(unique_lags):
            new_ar[k] = ar_orders[lags_arr == lag].max()
            new_i[k] = i_orders[lags_arr == lag].max()
            new_ma[k] = ma_orders[lags_arr == lag].max()
        ar_orders, i_orders, ma_orders, lags_arr = new_ar, new_i, new_ma, unique_lags

    # ---------- 2. derive component count -------------------------------
    components_number = int(
        max(
            int((ar_orders * lags_arr).sum() + (i_orders * lags_arr).sum()),
            int((ma_orders * lags_arr).sum()),
        )
    )
    ar_required = int(ar_orders.sum()) > 0
    ma_required = int(ma_orders.sum()) > 0
    ar_number = int(ar_orders.sum())
    ma_number = int(ma_orders.sum())

    # ---------- 3. constant flag ---------------------------------------
    constant_value = None
    if isinstance(constant, bool):
        constant_required = constant
        constant_generate = constant
    else:
        constant_required = True
        constant_generate = False
        constant_value = float(constant)

    persistence_length = components_number + int(constant_required)
    if persistence_length == 0:
        raise ValueError(
            "All orders are zero; nothing to simulate. Provide non-zero "
            "ar / i / ma orders or set ``constant=True``."
        )

    # ---------- 4. instantiate adamCore for polynomialise ---------------
    lags_model = np.ones(persistence_length, dtype=np.uint64)
    adam_cpp = _adamCore.adamCore(
        lags=lags_model,
        E="A",
        T="N",
        S="N",
        nNonSeasonal=0,
        nSeasonal=0,
        nETS=0,
        nArima=components_number,
        nXreg=0,
        nComponents=persistence_length,
        constant=bool(constant_required),
        adamETS=False,
    )

    # ---------- 5. validate / draw AR & MA --------------------------------
    if arma is None:
        ar_value: Optional[np.ndarray] = None
        ma_value: Optional[np.ndarray] = None
    else:
        ar_value_raw = arma.get("ar")
        ma_value_raw = arma.get("ma")
        ar_value = (
            np.asarray(ar_value_raw, dtype=float).ravel()
            if ar_value_raw is not None
            else None
        )
        ma_value = (
            np.asarray(ma_value_raw, dtype=float).ravel()
            if ma_value_raw is not None
            else None
        )
        if ar_value is not None and (
            ar_number != int((ar_value != 0).sum()) and ar_number != ar_value.size
        ):
            warnings.warn(
                f"Wrong number of AR coefficients (expected {ar_number}). "
                f"AR will be generated.",
                stacklevel=2,
            )
            ar_value = None
        if ma_value is not None and (
            ma_number != int((ma_value != 0).sum()) and ma_number != ma_value.size
        ):
            warnings.warn(
                f"Wrong number of MA coefficients (expected {ma_number}). "
                f"MA will be generated.",
                stacklevel=2,
            )
            ma_value = None

    ar_generate = ar_required and (ar_value is None)
    ma_generate = ma_required and (ma_value is None)

    # ---------- 6. random AR / MA draws with stability check ------------
    mat_ar = np.zeros((max(1, ar_number), nsim), dtype=np.float64)
    mat_ma = np.zeros((max(1, ma_number), nsim), dtype=np.float64)
    if ar_required:
        if ar_generate:
            for s in range(nsim):
                mat_ar[:ar_number, s] = _draw_stable_polynomial_roots(
                    rng,
                    adam_cpp=adam_cpp,
                    n_coefs=ar_number,
                    orders=ar_orders,
                    lags=lags_arr,
                    poly_kind="ar",
                    require_stable=(bounds == "admissible") and (components_number > 0),
                )
        else:
            # ar_value is set above (ar_required and not ar_generate ⇒ supplied).
            assert ar_value is not None
            mat_ar[:ar_number, :] = ar_value[ar_value != 0][:, None]
    if ma_required:
        if ma_generate:
            for s in range(nsim):
                mat_ma[:ma_number, s] = _draw_stable_polynomial_roots(
                    rng,
                    adam_cpp=adam_cpp,
                    n_coefs=ma_number,
                    orders=ma_orders,
                    lags=lags_arr,
                    poly_kind="ma",
                    require_stable=(bounds == "admissible") and (components_number > 0),
                )
        else:
            assert ma_value is not None
            mat_ma[:ma_number, :] = ma_value[ma_value != 0][:, None]

    # ---------- 7. assemble joint arma_parameters per sim ----------------
    arma_parameters = np.zeros((ar_number + ma_number, nsim), dtype=np.float64)
    for ell in range(nsim):
        j = ar_idx = ma_idx = 0
        for k in range(len(lags_arr)):
            if ar_required and ar_orders[k] > 0:
                p = int(ar_orders[k])
                arma_parameters[j : j + p, ell] = mat_ar[ar_idx : ar_idx + p, ell]
                j += p
                ar_idx += p
            if ma_required and ma_orders[k] > 0:
                q = int(ma_orders[k])
                arma_parameters[j : j + q, ell] = mat_ma[ma_idx : ma_idx + q, ell]
                j += q
                ma_idx += q

    # ---------- 8. constant draw ---------------------------------------
    vec_constant = np.zeros(nsim, dtype=np.float64)
    if constant_required:
        if constant_generate:
            if np.any(i_orders > 0):
                vec_constant = rng.uniform(-200.0, 200.0, nsim)
            else:
                vec_constant = rng.uniform(100.0, 1000.0, nsim)
        else:
            vec_constant[:] = constant_value

    # ---------- 9. base matrices ----------------------------------------
    # R uses a companion-form transition matrix: shift-1 sub-diagonal,
    # zero last row. Persistence vector starts as zeros; AR/I polynomial
    # coefficients fill its head.
    if components_number > 0:
        mat_f_base = np.zeros((components_number, components_number), dtype=np.float64)
        if components_number > 1:
            mat_f_base[: components_number - 1, 1:] = np.eye(components_number - 1)
        mat_wt_row = np.zeros(components_number, dtype=np.float64)
        mat_wt_row[0] = 1.0
    else:
        mat_f_base = np.ones((1, 1), dtype=np.float64)
        mat_wt_row = np.ones(1, dtype=np.float64)
    if constant_required:
        # Append the constant column to the transition matrix and a 0
        # to the measurement row (R/simssarima.R:473).
        n = mat_f_base.shape[0]
        mat_f_new = np.zeros((n + 1, n + 1), dtype=np.float64)
        mat_f_new[:n, :n] = mat_f_base
        mat_f_new[0, n] = 1.0
        mat_f_new[n, n] = 1.0
        mat_f_base = mat_f_new
        mat_wt_row = np.concatenate([mat_wt_row, [0.0]])

    obs_in = int(abs(round(obs)))
    nsim = int(abs(round(nsim)))
    frequency = int(abs(round(frequency)))
    initial_generate = initial is None

    burn_in = int(lags_arr.max()) if initial_generate else 0
    obs_with_burn = obs_in + burn_in
    obs_states = obs_with_burn + 1

    mat_wt = np.tile(mat_wt_row, (obs_with_burn, 1))

    arr_vt = np.zeros((persistence_length, obs_states, nsim), dtype=np.float64)
    arr_f = np.zeros((mat_f_base.shape[0], mat_f_base.shape[1], nsim), dtype=np.float64)
    mat_g = np.zeros((persistence_length, nsim), dtype=np.float64)
    mat_initial_value = np.zeros((max(1, components_number), nsim), dtype=np.float64)

    # ---------- 10. initials --------------------------------------------
    if components_number > 0:
        if initial_generate:
            mat_initial_value = rng.uniform(0.0, 1000.0, (components_number, nsim))
            arr_vt[:components_number, 0, :] = mat_initial_value
        else:
            init_arr = np.asarray(initial, dtype=float).ravel()
            if init_arr.size != components_number:
                warnings.warn(
                    f"Wrong length of initial vector (got {init_arr.size}, "
                    f"expected {components_number}). Regenerating.",
                    stacklevel=2,
                )
                init_arr = rng.uniform(0.0, 1000.0, components_number)
                initial_generate = True
            mat_initial_value[:components_number, :] = init_arr[:, None]
            arr_vt[:components_number, 0, :] = mat_initial_value

    # ---------- 11. per-sim polynomial fill ----------------------------
    for s in range(nsim):
        polys = adam_polynomialiser(
            adam_cpp=adam_cpp,
            B=np.zeros(0, dtype=np.float64),
            ar_orders=ar_orders.astype(np.uint64),
            i_orders=i_orders.astype(np.uint64),
            ma_orders=ma_orders.astype(np.uint64),
            ar_estimate=False,
            ma_estimate=False,
            arma_parameters=arma_parameters[:, s],
            lags=lags_arr.astype(np.uint64),
        )
        ari_poly = polys["ari_polynomial"]
        ma_poly = polys["ma_polynomial"]

        mat_f_s = mat_f_base.copy()
        vec_g_s = np.zeros(persistence_length, dtype=np.float64)
        if components_number > 0:
            # ARI coefficients (excluding leading 1) fill first column of F
            ari_tail = -ari_poly[1:]
            n_fill_f = min(len(ari_tail), components_number)
            mat_f_s[:n_fill_f, 0] = ari_tail[:n_fill_f]
            # Same ARI tail also drives the persistence vector
            vec_g_s[:n_fill_f] = ari_tail[:n_fill_f]
            if ma_required:
                ma_tail = ma_poly[1:]
                n_fill_g = min(len(ma_tail), components_number)
                vec_g_s[:n_fill_g] += ma_tail[:n_fill_g]
        arr_f[:, :, s] = mat_f_s
        mat_g[:, s] = vec_g_s

        # R: initialise burn-in state via F^(k+1) * vt to spread the
        # initial draw across the lag head.
        if initial_generate and components_number > 0:
            arr_vt[:, 0, s] = (
                np.linalg.matrix_power(mat_f_s, components_number + 1) @ arr_vt[:, 0, s]
            )
        if constant_required:
            arr_vt[persistence_length - 1, 0, s] = vec_constant[s]

    # ---------- 12. errors ---------------------------------------------
    is_default = is_default_randomizer(randomizer)
    n_errors = obs_with_burn * nsim
    if is_default and not randomizer_kwargs:
        if isinstance(randomizer, str) and randomizer in (
            "rnorm",
            "rlaplace",
            "rs",
        ):
            errors_flat = resolve_randomizer(randomizer, rng)(n_errors)
        elif randomizer == "rt":
            df = max(obs_with_burn - (persistence_length + 1), 1)
            errors_flat = rng.standard_t(df, n_errors)
        else:
            errors_flat = resolve_randomizer(randomizer, rng)(n_errors)
    else:
        errors_flat = resolve_randomizer(randomizer, rng, **randomizer_kwargs)(n_errors)
    mat_errors = np.asarray(errors_flat, dtype=np.float64).reshape(
        (obs_with_burn, nsim), order="F"
    )

    # R-style centring + variance scaling for default randomizers
    if is_default and not randomizer_kwargs:
        mat_errors -= mat_errors.mean(axis=0, keepdims=True)
        scale = np.sqrt(np.abs(arr_vt[0, 0, :]))
        mat_errors *= scale[None, :]
        if randomizer == "rs":
            mat_errors /= 4.0
    elif randomizer_kwargs:
        if randomizer == "rbeta":
            mat_errors -= 0.5
            col_rms = np.sqrt((mat_errors**2).mean(axis=0))
            mat_errors /= (col_rms * np.sqrt(np.abs(arr_vt[0, 0, :])))[None, :]
        elif randomizer == "rt":
            mat_errors *= np.sqrt(np.abs(arr_vt[0, 0, :]))[None, :]

    # ---------- 13. occurrence mask ------------------------------------
    probability_arr = np.atleast_1d(np.asarray(probability, dtype=float)).ravel()
    if probability_arr.size == 1 and probability_arr[0] == 1.0:
        mat_ot = np.ones((obs_with_burn, nsim), dtype=np.float64)
    else:
        if probability_arr.size == 1:
            p_vec = np.full(obs_with_burn, probability_arr[0])
        elif probability_arr.size == obs_in:
            # Re-align around the burn-in head: use the first probability
            # value for the burn-in section, the original for the rest.
            p_vec = np.concatenate(
                [np.full(burn_in, probability_arr[0]), probability_arr]
            )
        else:
            p_vec = probability_arr[:obs_with_burn]
        mat_ot = rng.binomial(1, p_vec[:, None], size=(obs_with_burn, nsim)).astype(
            np.float64
        )

    # ---------- 14. drive the C++ kernel --------------------------------
    profiles = adam_profile_creator(
        lags_model_all=lags_model.tolist(),
        lags_model_max=1,
        obs_all=obs_with_burn,
    )
    index_lookup_table = profiles["index_lookup_table"]
    profiles_recent_array = np.ascontiguousarray(arr_vt[:, :1, :], dtype=np.float64)

    result = adam_simulator(
        matrixErrors=mat_errors,
        matrixOt=mat_ot,
        arrayVt=arr_vt,
        matrixWt=mat_wt,
        arrayF=arr_f,
        matrixG=mat_g,
        lags=lags_model,
        indexLookupTable=index_lookup_table,
        profilesRecent=profiles_recent_array,
        E="A",
        T="N",
        S="N",
        nNonSeasonal=0,
        nSeasonal=0,
        nArima=components_number,
        nXreg=0,
        constant=bool(constant_required),
    )
    mat_yt = np.asarray(result["matrixYt"], dtype=np.float64)
    arr_vt = np.asarray(result["arrayVt"], dtype=np.float64).reshape(
        arr_vt.shape, order="F"
    )

    # ---------- 15. strip the burn-in ----------------------------------
    if initial_generate:
        mat_yt = mat_yt[burn_in:, :]
        mat_errors = mat_errors[burn_in:, :]
        mat_ot = mat_ot[burn_in:, :]
        arr_vt = arr_vt[:, burn_in:, :]
        if components_number > 0:
            target_col = burn_in if constant_required else burn_in
            sliced = arr_vt[: persistence_length - int(constant_required), 0, :]
            mat_initial_value = sliced if sliced.ndim == 2 else sliced[:, None]
            del target_col

    # ---------- 16. wrap output ----------------------------------------
    if nsim == 1:
        data_out: Union[pd.Series, pd.DataFrame] = pd.Series(mat_yt[:, 0])
        residuals_out: Union[pd.Series, pd.DataFrame] = pd.Series(mat_errors[:, 0])
    else:
        data_out = pd.DataFrame(mat_yt)
        residuals_out = pd.DataFrame(mat_errors)

    model_label = _build_model_label(
        ar_orders,
        i_orders,
        ma_orders,
        lags_arr,
        constant_required=constant_required,
        any_i_order=bool(np.any(i_orders > 0)),
        intermittent=bool(np.any(probability_arr != 1.0)),
    )

    return SimulateResult(
        model=model_label,
        data=data_out,
        states=arr_vt,
        residuals=residuals_out,
        arma={"ar": mat_ar, "ma": mat_ma},
        constant=float(vec_constant[0]) if constant_required and nsim == 1 else None,
        initial=mat_initial_value,
        profile=profiles_recent_array,
        occurrence=mat_ot if not np.all(mat_ot == 1.0) else None,
        probability=probability_arr if probability_arr.size > 1 else None,
        intermittent=("none" if np.all(probability_arr == 1.0) else "tsb"),
        other=dict(randomizer_kwargs),
    )


def _draw_stable_polynomial_roots(
    rng: np.random.Generator,
    *,
    adam_cpp: Any,
    n_coefs: int,
    orders: np.ndarray,
    lags: np.ndarray,
    poly_kind: str,
    require_stable: bool,
    max_tries: int = 1000,
) -> np.ndarray:
    """Draw AR or MA coefficients with stability rejection.

    Mirrors R's ``elementsGenerator`` (R/simssarima.R:148-193): draw
    uniform-(-1, 1) coefficients and reject draws whose ``polyroot``
    falls inside the unit circle. Caps at ``max_tries`` to avoid
    pathological models.
    """
    lags_u = lags.astype(np.uint64)
    zero_orders = np.zeros_like(orders, dtype=np.uint64)
    orders_u = orders.astype(np.uint64)
    poly_key = "ar_polynomial" if poly_kind == "ar" else "ma_polynomial"
    for _ in range(max_tries):
        candidate = rng.uniform(-1.0, 1.0, n_coefs)
        if not require_stable:
            return candidate
        polys = adam_polynomialiser(
            adam_cpp=adam_cpp,
            B=np.zeros(0, dtype=np.float64),
            ar_orders=orders_u if poly_kind == "ar" else zero_orders,
            i_orders=zero_orders,
            ma_orders=zero_orders if poly_kind == "ar" else orders_u,
            ar_estimate=False,
            ma_estimate=False,
            arma_parameters=candidate,
            lags=lags_u,
        )
        poly = polys[poly_key]
        if poly.size <= 1:
            return candidate
        roots = np.roots(poly[::-1])  # numpy's roots takes highest-power-first
        if np.all(np.abs(roots) > 1.0):
            return candidate
    raise RuntimeError(
        f"Failed to draw stable {poly_kind.upper()} coefficients after "
        f"{max_tries} tries; try ``bounds='none'`` or provide an explicit "
        f"``arma=...`` argument."
    )


def _build_model_label(
    ar_orders: np.ndarray,
    i_orders: np.ndarray,
    ma_orders: np.ndarray,
    lags: np.ndarray,
    *,
    constant_required: bool,
    any_i_order: bool,
    intermittent: bool,
) -> str:
    """R-style model label string. Matches R/simssarima.R:716-740."""
    if len(ar_orders) == 1 and np.all(lags == 1):
        label = f"ARIMA({int(ar_orders[0])},{int(i_orders[0])},{int(ma_orders[0])})"
    else:
        parts = []
        for k in range(len(ar_orders)):
            parts.append(
                f"({int(ar_orders[k])},{int(i_orders[k])},{int(ma_orders[k])})"
                f"[{int(lags[k])}]"
            )
        label = "SARIMA" + "".join(parts)
    if intermittent:
        label = "i" + label
    if constant_required:
        label += " with drift" if any_i_order else " with constant"
    return label
