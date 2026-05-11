"""
ARIMA order selection for ADAM.

Mirrors R's three-phase arimaSelector() algorithm from autoadam.R.
Phase 1: Select differencing (I) orders exhaustively.
Phase 2: Greedy MA order selection using ACF of residuals (while loop).
Phase 3: Greedy AR order selection using PACF of residuals (while loop).
Phase 4: Check IMA(d,q) special models (MA = I).
"""

import itertools
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _get_ic(model, ic_name: str) -> float:
    """Extract IC value from fitted ADAM model."""
    return float(getattr(model, ic_name.lower()))


def _fit_arima_model(
    y,
    ets_model: str,
    ar_orders: List[int],
    i_orders: List[int],
    ma_orders: List[int],
    lags: List[int],
    constant: bool,
    distribution: str,
    ic_name: str,
    X=None,
    **adam_kwargs,
) -> Tuple[Any, float]:
    """Fit ADAM with given ARIMA orders; return (model, IC).

    Returns (None, inf) on failure.
    """
    from smooth.adam_general.core.adam import ADAM

    try:
        kw = {k: v for k, v in adam_kwargs.items() if k != "ic"}
        kw["verbose"] = 0
        m = ADAM(
            model=ets_model,
            lags=lags,
            ar_order=ar_orders,
            i_order=i_orders,
            ma_order=ma_orders,
            constant=constant,
            distribution=distribution,
            arima_select=False,
            **kw,
        ).fit(y, X)
        return m, _get_ic(m, ic_name)
    except Exception:
        return None, np.inf


def _normalize_lags_and_orders(
    lags: List[int],
    max_ar: List[int],
    max_i: List[int],
    max_ma: List[int],
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """Ensure lags starts with 1; pad/truncate order lists to match lags length."""
    lags = list(lags)
    if not lags:
        lags = [1]
    if lags[0] != 1:
        lags = [1] + lags
        max_ar = [3] + list(max_ar)
        max_i = [2] + list(max_i)
        max_ma = [3] + list(max_ma)

    n = len(lags)
    max_ar = (list(max_ar) + [0] * n)[:n]
    max_i = (list(max_i) + [0] * n)[:n]
    max_ma = (list(max_ma) + [0] * n)[:n]
    return lags, max_ar, max_i, max_ma


def _prune_orders_for_ets(
    ets_model: str,
    lags: List[int],
    max_ar: List[int],
    max_i: List[int],
    max_ma: List[int],
) -> Tuple[List[int], List[int], List[int]]:
    """Reduce ARIMA search space based on ETS components.

    Mirrors autoadam.R lines 498-510:
    - ETS has non-N error (A/M):  zero out non-seasonal I and MA
    - ETS has non-N seasonal (A/M): zero out seasonal I and MA
    - ETS has damped trend: zero out non-seasonal AR
    """
    if not ets_model or ets_model == "NNN":
        return list(max_ar), list(max_i), list(max_ma)

    max_ar = list(max_ar)
    max_i = list(max_i)
    max_ma = list(max_ma)

    error_type = ets_model[0]
    seasonal_type = ets_model[-1] if len(ets_model) >= 3 else "N"
    trend_part = ets_model[1:-1] if len(ets_model) >= 3 else ""
    damped = "d" in trend_part.lower()

    for k, lag in enumerate(lags):
        if error_type in ("A", "M") and lag == 1:
            max_i[k] = 0
            max_ma[k] = 0
        if seasonal_type in ("A", "M") and lag != 1:
            max_i[k] = 0
            max_ma[k] = 0
        if damped and lag == 1:
            max_ar[k] = 0

    return max_ar, max_i, max_ma


def _select_i_orders(
    y,
    ets_model: str,
    max_i: List[int],
    lags: List[int],
    distribution: str,
    ic_name: str,
    X=None,
    initial_ic: float = np.inf,
    initial_model: Any = None,
    **adam_kwargs,
) -> Tuple[List[int], bool, Any, float]:
    """Phase 1: exhaustive search over all I-order combinations × constant.

    When an ETS baseline is provided via ``initial_ic`` / ``initial_model``,
    they serve as the starting best (mirrors R's ICOriginal).
    """
    n = len(lags)
    zeros = [0] * n
    best_ic = initial_ic
    best_i = zeros[:]
    best_constant = False
    best_model = initial_model

    for combo in itertools.product(*[range(m + 1) for m in max_i]):
        i_orders = list(combo)
        for constant in (False, True):
            # Skip (0,..,0) + const=False when an ETS baseline already covers it
            if (
                all(d == 0 for d in i_orders)
                and not constant
                and initial_model is not None
            ):
                continue
            model, ic = _fit_arima_model(
                y,
                ets_model,
                zeros,
                i_orders,
                zeros,
                lags,
                constant,
                distribution,
                ic_name,
                X,
                **adam_kwargs,
            )
            if ic < best_ic:
                best_ic, best_i, best_constant, best_model = (
                    ic,
                    i_orders,
                    constant,
                    model,
                )

    return best_i, best_constant, best_model, best_ic


def _select_ma_orders(
    y,
    ets_model: str,
    best_i: List[int],
    max_ma: List[int],
    lags: List[int],
    constant: bool,
    distribution: str,
    ic_name: str,
    best_model: Any,
    best_ic: float,
    X=None,
    **adam_kwargs,
) -> Tuple[List[int], Any, float]:
    """Phase 2: greedy MA selection using ACF of residuals (while loop).

    Mirrors R's while loop: after accepting an MA order, updates residuals
    and tries again until no further IC improvement.
    """
    from statsmodels.tsa.stattools import acf

    n = len(lags)
    best_ma = [0] * n

    try:
        resids = np.array(best_model.residuals, dtype=float)
    except Exception:
        return best_ma, best_model, best_ic

    for idx in sorted(range(n), key=lambda i: lags[i], reverse=True):
        lag, max_q = lags[idx], max_ma[idx]
        if max_q == 0:
            continue

        while True:
            try:
                n_lags_acf = max(max_q * lag * 2, len(resids) // 2) + 1
                acf_vals = acf(resids, nlags=n_lags_acf, fft=True)[1:]
            except Exception:
                break

            multiples = [
                k * lag - 1 for k in range(1, max_q + 1) if k * lag - 1 < len(acf_vals)
            ]
            if not multiples:
                break

            best_k = int(np.argmax([abs(acf_vals[m]) for m in multiples])) + 1
            trial_ma = best_ma[:]
            trial_ma[idx] = best_k

            model, ic = _fit_arima_model(
                y,
                ets_model,
                [0] * n,
                best_i,
                trial_ma,
                lags,
                constant,
                distribution,
                ic_name,
                X,
                **adam_kwargs,
            )
            if ic < best_ic:
                best_ic, best_ma, best_model = ic, trial_ma, model
                try:
                    resids = np.array(model.residuals, dtype=float)
                except Exception:
                    pass
                # Loop again with updated residuals and accepted order
            else:
                break

    return best_ma, best_model, best_ic


def _select_ar_orders(
    y,
    ets_model: str,
    best_i: List[int],
    best_ma: List[int],
    max_ar: List[int],
    lags: List[int],
    constant: bool,
    distribution: str,
    ic_name: str,
    best_model: Any,
    best_ic: float,
    X=None,
    **adam_kwargs,
) -> Tuple[List[int], Any, float]:
    """Phase 3: greedy AR selection using PACF of residuals (while loop).

    Mirrors R's while loop: after accepting an AR order, updates residuals
    and tries again until no further IC improvement.
    """
    from statsmodels.tsa.stattools import pacf

    n = len(lags)
    best_ar = [0] * n

    try:
        resids = np.array(best_model.residuals, dtype=float)
    except Exception:
        return best_ar, best_model, best_ic

    for idx in sorted(range(n), key=lambda i: lags[i], reverse=True):
        lag, max_p = lags[idx], max_ar[idx]
        if max_p == 0:
            continue

        while True:
            try:
                # statsmodels requires nlags < len(resids)//2
                n_lags_pacf = min(
                    max(max_p * lag * 2, len(resids) // 2),
                    len(resids) // 2 - 1,
                )
                pacf_vals = pacf(resids, nlags=n_lags_pacf)[1:]
            except Exception:
                break

            multiples = [
                k * lag - 1 for k in range(1, max_p + 1) if k * lag - 1 < len(pacf_vals)
            ]
            if not multiples:
                break

            best_k = int(np.argmax([abs(pacf_vals[m]) for m in multiples])) + 1
            trial_ar = best_ar[:]
            trial_ar[idx] = best_k

            model, ic = _fit_arima_model(
                y,
                ets_model,
                trial_ar,
                best_i,
                best_ma,
                lags,
                constant,
                distribution,
                ic_name,
                X,
                **adam_kwargs,
            )
            if ic < best_ic:
                best_ic, best_ar, best_model = ic, trial_ar, model
                try:
                    resids = np.array(model.residuals, dtype=float)
                except Exception:
                    pass
                # Loop again with updated residuals and accepted order
            else:
                break

    return best_ar, best_model, best_ic


def _check_ima_models(
    y,
    ets_model: str,
    best_ar: List[int],
    max_i: List[int],
    max_ma: List[int],
    lags: List[int],
    constant: bool,
    distribution: str,
    ic_name: str,
    best_ic: float,
    X=None,
    **adam_kwargs,
) -> Tuple[Optional[List[int]], Optional[List[int]], Any, float]:
    """Phase 4: test IMA(d,q) models where MA order = I order."""
    new_i = new_ma = new_model = None
    new_ic = best_ic

    for combo in itertools.product(*[range(m + 1) for m in max_i]):
        i_orders = list(combo)
        ma_orders = [min(d, max_ma[k]) for k, d in enumerate(i_orders)]

        if all(m == 0 for m in ma_orders):
            continue

        model, ic = _fit_arima_model(
            y,
            ets_model,
            best_ar,
            i_orders,
            ma_orders,
            lags,
            False,  # R always uses constant=FALSE for IMA models
            distribution,
            ic_name,
            X,
            **adam_kwargs,
        )
        if ic < new_ic:
            new_ic, new_i, new_ma, new_model = ic, i_orders, ma_orders, model

    return new_i, new_ma, new_model, new_ic


def arima_selector(
    y,
    ets_model: str = "NNN",
    max_ar_orders: Optional[List[int]] = None,
    max_i_orders: Optional[List[int]] = None,
    max_ma_orders: Optional[List[int]] = None,
    lags: Optional[List[int]] = None,
    distribution: str = "dnorm",
    ic: str = "AICc",
    X=None,
    ets_baseline=None,
    **adam_kwargs,
) -> Dict:
    """
    Three-phase ARIMA order selection.

    Mirrors R's ``arimaSelector()`` from autoadam.R lines 455-791.

    Parameters
    ----------
    y : array-like
        Time series data.
    ets_model : str, default="NNN"
        ETS model string passed to each internal ADAM fit.
    max_ar_orders : list of int, optional
        Maximum AR order per lag level. Defaults to ``[3]`` for each lag.
    max_i_orders : list of int, optional
        Maximum I order per lag level. Defaults to ``[2]`` for lag 1,
        ``[1]`` for seasonal lags.
    max_ma_orders : list of int, optional
        Maximum MA order per lag level. Defaults to ``[3]`` for each lag.
    lags : list of int, optional
        Seasonal periods. Lag 1 is prepended automatically if absent.
    distribution : str, default="dnorm"
        Error distribution used for all internal ADAM fits.
    ic : str, default="AICc"
        Information criterion used for model comparison.
    X : array-like, optional
        External regressor matrix passed to each internal fit.
    ets_baseline : fitted ADAM, optional
        Pre-fitted ETS-only model for this distribution. When provided, its
        IC serves as the Phase-1 baseline (mirrors R's ICOriginal), and the
        final result is compared against it — pure ETS is returned if
        ETS+ARIMA is not an improvement.
    **adam_kwargs
        Additional keyword arguments forwarded to every internal ADAM fit.

    Returns
    -------
    dict
        Keys: ``"ar_orders"``, ``"i_orders"``, ``"ma_orders"``,
        ``"constant"`` (bool), ``"model"`` (fitted ADAM), ``"ic_value"``.
    """
    lags = list(lags) if lags is not None else [1]
    n_lags = len(lags)

    default_max_ar = [3] * n_lags
    default_max_i = [2] + [1] * (n_lags - 1)
    default_max_ma = [3] * n_lags

    max_ar = list(max_ar_orders) if max_ar_orders is not None else default_max_ar
    max_i = list(max_i_orders) if max_i_orders is not None else default_max_i
    max_ma = list(max_ma_orders) if max_ma_orders is not None else default_max_ma

    lags, max_ar, max_i, max_ma = _normalize_lags_and_orders(
        lags, max_ar, max_i, max_ma
    )

    # Resolve the actual selected ETS type from the fitted baseline (mirrors
    # R's: model <- etsModelType <- modelType(testModelETS)).
    # When the input is "ZXZ"/"ZZZ" etc., the baseline already has the
    # auto-selected type (e.g. "ETS(AAA)"). Extract it so pruning and
    # internal fits use the concrete model string.
    resolved_ets = ets_model
    if ets_baseline is not None and ets_model != "NNN":
        import re

        raw = getattr(ets_baseline, "model", "")
        m = re.search(r"ETS\(([^)]+)\)", raw)
        if m:
            resolved_ets = m.group(1)  # e.g. "AAA"

    # Prune ARIMA search space based on ETS model type (autoadam.R lines 498-510)
    if resolved_ets != "NNN":
        max_ar, max_i, max_ma = _prune_orders_for_ets(
            resolved_ets, lags, max_ar, max_i, max_ma
        )

    # ETS baseline: use its IC and residuals as starting point for Phase 1
    initial_ic = _get_ic(ets_baseline, ic) if ets_baseline is not None else np.inf
    initial_model = ets_baseline

    # Phase 1 — differencing orders
    best_i, best_constant, best_model, best_ic = _select_i_orders(
        y,
        resolved_ets,
        max_i,
        lags,
        distribution,
        ic,
        X,
        initial_ic=initial_ic,
        initial_model=initial_model,
        **adam_kwargs,
    )

    if best_model is None:
        warnings.warn(
            "ARIMA order selection failed at phase 1 (I selection). "
            "Returning default ARIMA(0,1,1).",
            UserWarning,
            stacklevel=2,
        )
        n = len(lags)
        return {
            "ar_orders": [0] * n,
            "i_orders": [1] + [0] * (n - 1),
            "ma_orders": [1] + [0] * (n - 1),
            "constant": False,
            "model": None,
            "ic_value": np.inf,
        }

    # Phase 2 — MA orders
    best_ma, best_model, best_ic = _select_ma_orders(
        y,
        resolved_ets,
        best_i,
        max_ma,
        lags,
        best_constant,
        distribution,
        ic,
        best_model,
        best_ic,
        X,
        **adam_kwargs,
    )

    # Phase 3 — AR orders
    best_ar, best_model, best_ic = _select_ar_orders(
        y,
        resolved_ets,
        best_i,
        best_ma,
        max_ar,
        lags,
        best_constant,
        distribution,
        ic,
        best_model,
        best_ic,
        X,
        **adam_kwargs,
    )

    # Phase 4 — IMA(d,q) special models
    ima_i, ima_ma, ima_model, ima_ic = _check_ima_models(
        y,
        resolved_ets,
        best_ar,
        max_i,
        max_ma,
        lags,
        best_constant,
        distribution,
        ic,
        best_ic,
        X,
        **adam_kwargs,
    )
    if ima_model is not None:
        best_i, best_ma, best_model, best_ic = ima_i, ima_ma, ima_model, ima_ic

    # Final check: if ETS baseline is better or equal, return it
    # (autoadam.R lines 760-774)
    if ets_baseline is not None and best_ic >= initial_ic:
        best_model = ets_baseline
        best_ic = initial_ic
        n = len(lags)
        best_ar = [0] * n
        best_i = [0] * n
        best_ma = [0] * n
        best_constant = False

    # Final re-estimation: refit the selected ETS+ARIMA from scratch.
    # Mirrors autoadam.R lines 762-774: after greedy phase selection,
    # R always refits the complete model one more time (fresh backcasting
    # start), giving the optimizer an independent run that can find a
    # better local optimum (especially for distributions like dlnorm).
    if ets_baseline is not None and best_model is not ets_baseline:
        refit_model, refit_ic = _fit_arima_model(
            y,
            resolved_ets,
            best_ar,
            best_i,
            best_ma,
            lags,
            best_constant,
            distribution,
            ic,
            X,
            **adam_kwargs,
        )
        if refit_ic < best_ic:
            best_model, best_ic = refit_model, refit_ic
        elif refit_ic >= initial_ic:
            # Refit worse than ETS-only baseline — fall back to ETS
            best_model = ets_baseline
            best_ic = initial_ic
            n = len(lags)
            best_ar = [0] * n
            best_i = [0] * n
            best_ma = [0] * n
            best_constant = False

    return {
        "ar_orders": best_ar,
        "i_orders": best_i,
        "ma_orders": best_ma,
        "constant": best_constant,
        "model": best_model,
        "ic_value": best_ic,
    }
