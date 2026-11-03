"""
The estimation of CES: the two stages of the optimiser from the starting values.

Translates ces_estimator() of R/adam-ces.R.
"""

import nlopt
import numpy as np

from smooth.adam_general.core.ces.cost_function import ces_cf


def ces_estimator(B, cf_kwargs, settings, xreg_model):
    """The estimates of CES from the starting B: BOBYQA, then the algorithm of the
    optimiser (Nelder-Mead by default), with the settings of the first stage in
    ``settings`` (algorithm0, maxeval0, xtol_rel0, ...). Returns B and the loss (R's
    ces_estimator)."""

    def objective(x, grad):
        return ces_cf(B=x, **cf_kwargs)

    # Maxeval — R lines 800-807
    maxeval_used = settings["maxeval"]
    if maxeval_used is None:
        maxeval_used = len(B) * 40
        if xreg_model:
            maxeval_used = max(1000, len(B) * 100)
    maxeval0_used = settings["maxeval0"]
    if maxeval0_used is None:
        maxeval0_used = maxeval_used

    # Stage 1: BOBYQA — R lines 853-857
    algo_map = {
        "NLOPT_LN_BOBYQA": nlopt.LN_BOBYQA,
        "NLOPT_LN_NELDERMEAD": nlopt.LN_NELDERMEAD,
        "NLOPT_LN_SBPLX": nlopt.LN_SBPLX,
        "NLOPT_LN_COBYLA": nlopt.LN_COBYLA,
    }

    lower = np.full(len(B), -np.inf)
    upper = np.full(len(B), np.inf)
    if settings["lb"] is not None:
        lower = np.asarray(settings["lb"], dtype=float)
    if settings["ub"] is not None:
        upper = np.asarray(settings["ub"], dtype=float)
    if len(B) == 0:
        # Nothing left to estimate: `a` (and `b`) supplied together with
        # backcast/complete initials and no xreg leaves an empty parameter
        # vector. R switches ``modelDo`` to "use" there and evaluates the
        # cost once instead of optimising (R/adam-ces.R). nlopt cannot be
        # constructed with zero dimensions -- it raises ``invalid_argument`` --
        # so the optimiser is skipped entirely.
        return B, float(ces_cf(B=B, **cf_kwargs))
    # First run of BOBYQA to get better values of B, tuned then by the main
    # optimiser -- R's two nloptr() calls
    for suffix, default, maxeval in (
        ("0", nlopt.LN_BOBYQA, maxeval0_used),
        ("", nlopt.LN_NELDERMEAD, maxeval_used),
    ):
        opt = nlopt.opt(algo_map.get(settings["algorithm" + suffix], default), len(B))
        opt.set_min_objective(objective)
        opt.set_lower_bounds(lower)
        opt.set_upper_bounds(upper)
        opt.set_maxeval(maxeval)
        opt.set_xtol_rel(settings["xtol_rel" + suffix])
        opt.set_xtol_abs(settings["xtol_abs" + suffix])
        opt.set_ftol_rel(settings["ftol_rel" + suffix])
        opt.set_ftol_abs(settings["ftol_abs" + suffix])
        opt.set_maxtime(settings["maxtime" + suffix])
        try:
            B = opt.optimize(B)
        except nlopt.RoundoffLimited:
            B = B.copy()
    return B, opt.last_optimum_value()
