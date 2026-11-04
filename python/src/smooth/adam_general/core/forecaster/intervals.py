import greybox as gb
import nlopt
import numpy as np
from scipy.special import gamma

from smooth.adam_general.core.utils.distributions import (
    generate_errors,
    normalize_errors,
)
from smooth.adam_general.core.utils.utils import (
    _sum_r,
    gap_quantile,
    observed_mask,
    scale_debias,
    scale_variance,
)
from smooth.adam_general.core.utils.var_covar import (
    covar_anal,
    gap_drift,
    var_anal,
)

from ._helpers import (
    _compute_multistep_errors,
    _prepare_lookup_table,
    _prepare_matrices_for_forecast,
)


def ensure_level_format(level, side):
    """Convert level scalar/list to numpy arrays of lower and upper quantiles.

    Parameters
    ----------
    level : float or list of float
        Confidence level(s), e.g. 0.95 or [0.9, 0.95, 0.99].
    side : str
        "both", "upper", or "lower".

    Returns
    -------
    level_low, level_up : numpy arrays of shape (n_levels,)
    """
    if isinstance(level, (int, float)):
        level = [level]
    level = np.array([lv / 100 if lv > 1 else lv for lv in level])

    if side == "both":
        level_low = (1 - level) / 2
        level_up = (1 + level) / 2
    elif side == "upper":
        level_low = np.zeros_like(level)
        level_up = level
    else:  # "lower"
        level_low = 1 - level
        level_up = np.ones_like(level)

    # Exact, as R: only the names of the columns are rounded
    return level_low, level_up


def _obs_observed(observations_dict):
    """The observed sizes, which the scale is divided by: the missing values are
    not, and neither are the zeros of an occurrence model (R: ``adam_nobsObserved``)."""
    return int(np.sum(observations_dict["ot_logical"]))


def _df_scale(general, observations_dict, params_info):
    """Degrees of freedom for de-biasing the scale (R: ``adam_dfScale``).

    The model passes its own as ``general["df_scale"]``. Otherwise: the observed
    sizes minus the parameters, without the scale ones under likelihood,
    with ``params_info[0]`` holding ``[..., n_scale, n_all]`` as ``sigma()`` reads it.
    """
    if general.get("df_scale") is not None:
        return general["df_scale"]
    obs_df = _obs_observed(observations_dict)
    info = params_info[0]
    n_param = info[-1]
    if general.get("loss") == "likelihood" and len(info) > 1:
        n_param = n_param - info[1]
    df = obs_df - n_param
    return df if df > 0 else obs_df


def _scale_model_variance(general, observations_dict, params_info):
    """Per-horizon variance implied by an implanted scale model, or ``None``.

    R replaces the constant ``s2`` with the scale model's forecast when one is
    implanted (``R/adam.R:6485-6530``). The forecast is the scale itself, so it
    is mapped onto a variance and de-biased by ``obsInSample / df``.
    """
    scale_forecast = general.get("scale_forecast")
    if scale_forecast is None:
        return None

    sf = np.asarray(scale_forecast, dtype=np.float64).ravel()[-general["h"] :]
    variance = scale_variance(sf, general["distribution"], general.get("other"))
    obs = _obs_observed(observations_dict)
    return variance * obs / _df_scale(general, observations_dict, params_info)


def generate_prediction_interval(
    predictions,
    prepared_model,
    general,
    observations_dict,
    model_type_dict,
    lags_dict,
    params_info,
    level_low,
    level_up,
    p_forecast=None,
    scale_2d_override=None,
):
    """The approximate interval: the variances of the horizons implied by the model,
    and the quantiles of its distribution with them, relative to the forecast, which
    ``_interval_bounds`` puts around it (R's adam_generatePredictionInterval)."""
    mat_vt, mat_wt, vec_g, mat_f = _prepare_matrices_for_forecast(
        prepared_model, observations_dict, lags_dict, general
    )

    # Estimate sigma. The error type decides how residuals(object) is
    # reconstructed for the ratio-domain distributions.
    e_type = model_type_dict["error_type"]  # "A" or "M"
    # The variance implied by the scale, de-biased (R: adam_varianceDebiased), so the
    # analytical intervals use the same estimate as the likelihood and simulations
    scale = general.get("scale_fitted")
    s2 = float(
        np.mean(
            scale_variance(
                prepared_model["scale"] if scale is None else scale,
                general["distribution"],
                general.get("other"),
            )
        )
        * _obs_observed(observations_dict)
        / _df_scale(general, observations_dict, params_info)
    )
    s2_forecast = _scale_model_variance(general, observations_dict, params_info)
    # The cumulative forecast sums the h steps after the sample, with the origin k
    # periods before its end
    shift = int(general.get("_gap_shift", 0) or 0)

    # lines 8015 to 8022
    # line 8404 -> I dont get the (is.scale(object$scale))
    # Skipping for now.
    # Will ask Ivan what this is

    # The mean of log(1+e) times -2 for the log-normal, when it is not the variance,
    # and the moments of the log of the drift of the states for the mixtures
    v_mean = drift = None
    if scale_2d_override is not None:
        v_voc_multi = np.asarray(scale_2d_override, dtype=float).ravel()
    elif (
        general["distribution"] == "dlnorm"
        and general.get("ets") == "adam"
        and model_type_dict["ets_model"]
        and not model_type_dict.get("arima_model", False)
        and e_type == "M"
        and model_type_dict["trend_type"] != "A"
        and model_type_dict["season_type"] != "A"
    ):
        # A pure multiplicative ADAM ETS with the log-normal is additive in logs:
        # log(1+e) at the horizon j has the mean -s2*a_j/2 and the variance s2*k_j,
        # as over the gaps
        covar = covar_anal(
            lags_dict["lags_model_all"], general["h"], mat_wt, mat_f, vec_g, 1.0
        )
        s2_horizon = s2 if s2_forecast is None else s2_forecast
        v_voc_multi = s2_horizon * np.diag(covar)
        v_mean = s2_horizon * np.cumsum(covar[0])
        # After the periods without an observed size, those of the filter, with the
        # uncertainty of the states at the end of the sample
        if general.get("_filter_variances") is not None:
            v_voc_multi = s2_horizon * general["_filter_variances"]
            v_mean = s2_horizon * general["_filter_mean"]
        if general.get("cumulative", False):
            v_voc_multi = np.sum(v_voc_multi[shift:])
    elif (
        general["distribution"] in ("dgamma", "dinvgauss")
        and general.get("ets") == "adam"
        and model_type_dict["ets_model"]
        and not model_type_dict.get("arima_model", False)
        and e_type == "M"
        and model_type_dict["trend_type"] != "A"
        and model_type_dict["season_type"] != "A"
        and s2_forecast is None
        and not general.get("cumulative", False)
    ):
        # A pure multiplicative ADAM ETS with the Gamma or the Inverse Gaussian: z/mu
        # at the horizon j is D(1+e), log(D) ~ N(mu (a_j - 1), v (k_j - 1)) with the
        # moments mu and v of log(1+e) and the multipliers a_j and k_j of the
        # log-normal above, the quantiles of their mixture
        multiplier_mean = general.get("_filter_mean")
        multiplier_variance = general.get("_filter_variances")
        if multiplier_variance is None:
            covar = covar_anal(
                lags_dict["lags_model_all"], general["h"], mat_wt, mat_f, vec_g, 1.0
            )
            multiplier_mean = np.cumsum(covar[0])
            multiplier_variance = np.diag(covar)
        noise = gap_drift(vec_g, [], general["distribution"], True, s2)[2]
        v_voc_multi = s2
        drift = (
            noise[0][0] * (np.ravel(multiplier_mean) - 1),
            noise[1][0, 0] * (np.ravel(multiplier_variance) - 1),
        )
    elif (
        model_type_dict["ets_model"]
        and general["distribution"]
        in ["dinvgauss", "dgamma", "dlnorm", "dllaplace", "dls", "dlgnorm"]
        and model_type_dict["error_type"] == "M"
    ):
        # Multiplicative-error variance under one of the log/positive
        # distributions: compute the per-horizon analytic variance. The log-normal
        # takes the variance of 1+e, s2 being the one of log(1+e)
        if general["distribution"] == "dlnorm":
            s2 = np.exp(s2) - 1
            if s2_forecast is not None:
                s2_forecast = np.exp(s2_forecast) - 1
        v_voc_multi = var_anal(
            lags_dict["lags_model_all"], general["h"], mat_wt[0], mat_f, vec_g, s2
        )

        if s2_forecast is not None:
            # R rescales the whole matrix by sqrt(s2F) outer sqrt(s2F); on the
            # diagonal that is just v / s2 * s2F.
            v_voc_multi = v_voc_multi / s2 * s2_forecast

        # For log-based distributions, transform the variance through log(1+v)
        if general["distribution"] in ["dlnorm", "dls", "dllaplace", "dlgnorm"]:
            v_voc_multi = np.log(1 + v_voc_multi)

        # Cumulative forecasts in this branch are not strictly correct —
        # we fall back to summing the per-horizon variances.
        if general.get("cumulative", False):
            v_voc_multi = np.sum(v_voc_multi[shift:])
    else:
        v_voc_multi = covar_anal(
            lags_dict["lags_model_all"], general["h"], mat_wt, mat_f, vec_g, s2
        )

        if s2_forecast is not None:
            # Time-varying variance: rescale the covariance matrix itself, so a
            # cumulative forecast sums the rescaled off-diagonals too.
            root = np.sqrt(s2_forecast)
            v_voc_multi = v_voc_multi / s2 * np.outer(root, root)

        # Variance of the cumulative sum vs. per-horizon diagonal variances: after
        # the periods without an observed value, those of the filter, with the
        # uncertainty of the states at the end of the sample
        if general.get("cumulative", False):
            v_voc_multi = np.sum(v_voc_multi[shift:, shift:])
        elif general.get("_filter_variances") is not None:
            v_voc_multi = (s2 if s2_forecast is None else s2_forecast) * general[
                "_filter_variances"
            ]
        else:
            v_voc_multi = np.diag(v_voc_multi)

    # Build prediction intervals per distribution.
    y_forecast = np.atleast_1d(predictions)
    v_voc_multi = np.atleast_1d(v_voc_multi)

    # Reshape for broadcasting: scale (h,1), levels (1,n_levels) or (h,n_levels)
    scale_2d = v_voc_multi.reshape(-1, 1) if v_voc_multi.ndim == 1 else v_voc_multi
    ll = level_low.reshape(1, -1)  # (1, n_levels)
    lu = level_up.reshape(1, -1)  # (1, n_levels)

    # Occurrence-aware level adjustment. Adjust the confidence LEVEL (e.g.
    # 0.95), not the quantiles, then derive symmetric quantile pairs from
    # the adjusted level:
    #     level_adj = max(0, (level - (1 - p)) / p)
    # ``level = level_up - level_low`` regardless of side (both / upper / lower).
    if p_forecast is not None:
        p_col = np.asarray(p_forecast, dtype=float).reshape(-1, 1)  # (h, 1)
        conf_levels = (level_up - level_low).reshape(1, -1)  # (1, n_levels)
        conf_levels_adj = np.maximum(  # (h, n_levels)
            0.0, (conf_levels - (1.0 - p_col)) / p_col
        )
        # Reconstruct quantile pairs using the same side logic as ensure_level_format
        is_upper_side = level_low == 0  # (n_levels,)
        is_lower_side = level_up == 1  # (n_levels,)
        ll = np.where(
            is_upper_side,
            0.0,
            np.where(is_lower_side, 1.0 - conf_levels_adj, (1.0 - conf_levels_adj) / 2),
        )
        lu = np.where(
            is_upper_side,
            conf_levels_adj,
            np.where(is_lower_side, 1.0, (1.0 + conf_levels_adj) / 2),
        )

    # Quantiles of the *error* distribution, relative to the forecast (R's
    # adam_intervalQuantiles). `scale_2d` is R's `vcovMulti`.
    variance = s2 if drift is not None else scale_2d
    variance_mean = None if v_mean is None else v_mean.reshape(-1, 1)
    y_lower = _interval_quantiles(
        ll, general, e_type, variance, y_forecast, variance_mean, drift
    )
    y_upper = _interval_quantiles(
        lu, general, e_type, variance, y_forecast, variance_mean, drift
    )
    return y_lower, y_upper


def _interval_quantiles(
    levels, general, e_type, variance, centre, variance_mean=None, drift=None
):
    """The quantiles of the error term at the levels, of the distribution with the
    variances (variance_mean is the mean of log(1+e) times -2 for the log-normal,
    drift the moments of the drift of the states after the periods without an
    observed size): around zero for the additive error and one for the
    multiplicative one; the distributions of 1+e of an additive error are relative to
    the centre (R's adam_intervalQuantiles). From greybox, parameterised as R."""
    distribution = general["distribution"]
    rows = len(np.atleast_1d(centre))
    columns = np.shape(levels)[-1]
    if drift is not None:
        return gap_quantile(
            np.broadcast_to(levels, (rows, columns)), distribution, variance, *drift
        )
    other = general.get("other", {}) or {}
    shape = other.get("shape")
    loc = 1.0 if e_type == "M" else 0.0
    gnorm_scale = (
        np.sqrt(variance * (gamma(1 / shape) / gamma(3 / shape)))
        if shape is not None
        else None
    )
    quantiles = {
        "dnorm": lambda: gb.qnorm(levels, loc, np.sqrt(variance)),
        "dlaplace": lambda: gb.qlaplace(levels, loc, np.sqrt(variance / 2)),
        "ds": lambda: gb.qs(levels, loc, (variance / 120) ** 0.25),
        "dgnorm": lambda: gb.qgnorm(levels, loc, gnorm_scale, shape),
        "dlogis": lambda: gb.qlogis(levels, loc, np.sqrt(variance * 3) / np.pi),
        # R scales the standard t rather than passing a scale argument
        "dt": lambda: (
            loc
            + np.sqrt(variance) * gb.qt(levels, general.get("df_t", other.get("nu")))
        ),
        # 1+e with the mean of one: variance is the one of log(1+e)
        "dlnorm": lambda: gb.qlnorm(
            levels,
            -(variance if variance_mean is None else variance_mean) / 2,
            np.sqrt(variance),
        ),
        "dllaplace": lambda: np.exp(gb.qlaplace(levels, 0.0, np.sqrt(variance / 2))),
        "dls": lambda: np.exp(gb.qs(levels, 0.0, (variance / 120) ** 0.25)),
        "dlgnorm": lambda: np.exp(gb.qgnorm(levels, 0.0, gnorm_scale, shape)),
        # mean 1, dispersion vcovMulti: greybox's (loc, scale) are statmod's
        # (mean, dispersion)
        "dinvgauss": lambda: gb.qinvgauss(levels, 1.0, variance),
        "dgamma": lambda: gb.qgamma(levels, shape=1 / variance, scale=variance),
    }.get(distribution, lambda: 0 * levels)()
    quantiles = np.array(np.broadcast_to(quantiles, (rows, columns)), dtype=float)
    if e_type == "A" and distribution in _LOG_DISTS:
        quantiles = (quantiles - 1) * np.reshape(centre, (-1, 1))
    return quantiles


def _interval_bounds(y_lower, y_upper, centre, level_low, level_up, e_type, cumulative):
    """The bounds of the interval around the forecasts from the quantiles relative to
    them: the levels of zero and one at the ends of the support, no value as zero
    (R's adam_intervalBounds)."""
    y_lower = np.array(y_lower, dtype=float)
    y_upper = np.array(y_upper, dtype=float)
    # Make sensible values out of those weird quantiles
    if cumulative:
        if np.any(level_low == 0):
            y_lower[:] = -np.inf if e_type == "A" else 0.0
        if np.any(level_up == 1):
            y_upper[:] = np.inf
    else:
        y_lower[:, level_low == 0] = -np.inf if e_type == "A" else 0.0
        y_upper[:, level_up == 1] = np.inf
    # Substitute NAs and NaNs with zeroes
    no_value = 0.0 if e_type == "A" else 1.0
    y_lower[np.isnan(y_lower)] = no_value
    y_upper[np.isnan(y_upper)] = no_value
    # Do intervals around the forecasts
    centre = np.reshape(centre, (-1, 1))
    if e_type == "A":
        return centre + y_lower, centre + y_upper
    return centre * y_lower, centre * y_upper


def generate_simulation_interval(
    predictions,
    prepared_model,
    general_dict,
    observations_dict,
    model_type_dict,
    lags_dict,
    components_dict,
    explanatory_checked,
    constants_checked,
    params_info,
    adam_cpp,
    level_low,
    level_up,
    nsim=10000,
    external_errors=None,
    p_forecast=None,
):
    """
    Generate prediction intervals using simulation.

    Simulates ``nsim`` forecast trajectories from the fitted state-space
    model and takes empirical quantiles at the requested levels.

    Parameters
    ----------
    predictions : np.ndarray
        Point forecasts.
    prepared_model : dict
        Dictionary with the prepared model.
    general_dict : dict
        Dictionary with general model parameters.
    observations_dict : dict
        Dictionary with observation data.
    model_type_dict : dict
        Dictionary with model type information.
    lags_dict : dict
        Dictionary with lag-related information.
    components_dict : dict
        Dictionary with model components information.
    explanatory_checked : dict
        Dictionary with external regressors information.
    constants_checked : dict
        Dictionary with information about constants.
    params_info : dict
        Dictionary with parameter information.
    level_low : numpy.ndarray
        Lower quantile levels, shape (n_levels,).
    level_up : numpy.ndarray
        Upper quantile levels, shape (n_levels,).
    nsim : int
        Number of simulations to run.
    external_errors : np.ndarray, optional
        Pre-generated error matrix of shape (h, nsim) for deterministic testing.
        If provided, these errors are used instead of generating new ones.
        This allows 100% reproducibility between R and Python by using
        the same random errors.

    Returns
    -------
    tuple
        (y_lower, y_upper, y_simulated, y_mean): the quantiles relative to the
        forecast (differences for the additive error, ratios for the multiplicative
        one), which ``_interval_bounds`` puts around it; y_simulated is the raw
        (h, nsim) simulation matrix when ``general_dict["scenarios"]`` is True, else
        None; y_mean the mean of the paths.
    """
    h = general_dict["h"]
    lags_model_max = lags_dict["lags_model_max"]
    # The errors and the occurrence draws, reproducible with a seed
    rng = np.random.default_rng(general_dict.get("seed"))

    # Get number of components
    n_components = (
        components_dict["components_number_ets"]
        + components_dict.get("components_number_arima", 0)
        + explanatory_checked["xreg_number"]
        + int(constants_checked["constant_required"])
    )

    # 1. Create 3D state array: [components, h+lags_max, nsim]
    arr_vt = np.zeros((n_components, h + lags_model_max, nsim), order="F")

    # Initialize with current states (replicated across nsim)
    # The states at the origin of the forecast (k periods before the end with
    # "_gap_shift")
    origin = prepared_model["states"].shape[1] - int(
        general_dict.get("_gap_shift", 0) or 0
    )
    mat_vt = prepared_model["states"][:, origin - lags_model_max : origin]
    for i in range(nsim):
        arr_vt[:, :lags_model_max, i] = mat_vt[:, :lags_model_max]

    # 2. The scale, or the scale model's forecasts, de-biased in the variance space
    obs_in_sample = _obs_observed(observations_dict)
    df = _df_scale(general_dict, observations_dict, params_info)
    scale_forecast = general_dict.get("scale_forecast")
    if scale_forecast is None:
        scale_value = prepared_model["scale"]
    else:
        # One draw per (horizon, replication); the h scales tile across nsim so
        # that reshaping column-major puts the right scale on the right horizon.
        scale_value = np.tile(
            np.asarray(scale_forecast, dtype=np.float64).ravel()[-h:], nsim
        )
    scale_value = scale_debias(
        scale_value, general_dict["distribution"], obs_in_sample, df
    )

    # 4. Generate random errors or use external errors
    if external_errors is not None:
        # Use externally provided errors for deterministic testing
        mat_errors = external_errors
        if mat_errors.shape != (h, nsim):
            raise ValueError(
                f"external_errors shape {mat_errors.shape} "
                f"does not match (h={h}, nsim={nsim})"
            )
        distribution = general_dict["distribution"]
    else:
        distribution = general_dict["distribution"]
        other_params = general_dict.get("other", {})

        # Generate h*nsim errors and reshape to (h, nsim)
        errors_flat = generate_errors(
            distribution=distribution,
            n=h * nsim,
            scale=scale_value,
            obs_in_sample=obs_in_sample,
            n_param=obs_in_sample - df,
            shape=other_params.get("shape"),
            random_state=rng,
        )
        mat_errors = errors_flat.reshape((h, nsim), order="F")

    # 5. Normalize errors if nsim <= 500
    e_type = model_type_dict["error_type"]
    if nsim <= 500:
        mat_errors = normalize_errors(mat_errors, e_type)

    # 6. Determine modified error type for additive models with log-distributions
    e_type_modified = e_type
    if e_type == "A" and distribution in [
        "dlnorm",
        "dinvgauss",
        "dgamma",
        "dls",
        "dllaplace",
        "dlgnorm",
    ]:
        e_type_modified = "M"

    # 7. Prepare matrices for simulator
    mat_vt_prep, mat_wt, vec_g, mat_f = _prepare_matrices_for_forecast(
        prepared_model, observations_dict, lags_dict, general_dict
    )

    # Prepare lookup table
    lookup = _prepare_lookup_table(lags_dict, observations_dict, general_dict)

    # Create 3D arrays for F and G (replicated for each simulation)
    arr_f = np.zeros((mat_f.shape[0], mat_f.shape[1], nsim), order="F")
    for i in range(nsim):
        arr_f[:, :, i] = mat_f

    # G matrix: [n_components, nsim]
    mat_g = np.zeros((n_components, nsim), order="F")
    for i in range(nsim):
        mat_g[:, i] = vec_g.flatten()

    # Occurrence matrix: Bernoulli draws when occurrence model is active
    if p_forecast is not None:
        mat_ot = rng.binomial(
            1, np.asarray(p_forecast, dtype=float).reshape(-1, 1), (h, nsim)
        ).astype(float)
        mat_ot = np.asfortranarray(mat_ot)
    else:
        mat_ot = np.ones((h, nsim), order="F")

    # Profiles recent table - expand to 3D cube (nComponents, lagsModelMax, nsim)
    profiles_recent_2d = prepared_model["profiles_recent_table"]
    profiles_recent = np.zeros(
        (profiles_recent_2d.shape[0], profiles_recent_2d.shape[1], nsim), order="F"
    )
    for i in range(nsim):
        profiles_recent[:, :, i] = profiles_recent_2d
    profiles_recent = np.asfortranarray(profiles_recent, dtype=np.float64)

    # Prepare inputs for C++ simulator
    arr_vt_f = np.asfortranarray(arr_vt, dtype=np.float64)
    mat_errors_f = np.asfortranarray(mat_errors, dtype=np.float64)
    mat_ot_f = np.asfortranarray(mat_ot, dtype=np.float64)
    arr_f_f = np.asfortranarray(arr_f, dtype=np.float64)
    mat_wt_f = np.asfortranarray(mat_wt, dtype=np.float64)
    mat_g_f = np.asfortranarray(mat_g, dtype=np.float64)
    lookup_f = np.asfortranarray(lookup, dtype=np.uint64)

    # 8. Call adam_cpp.simulate() with the prepared inputs
    # Note: E, T, S, nNonSeasonal, nSeasonal, nArima, nXreg, constant are set
    # during adamCore construction
    # refineHead=False: this is the forecast-interval path; ``arr_vt``'s first
    # ``lagsModelMax`` columns hold the fitted tail of the state matrix (not
    # a raw initialiser output), so no head walk is required — and the sliced
    # ``lookup`` has only ``h`` columns, so calling refineHeadFwd on it would
    # read out of bounds when ``lagsModelMax > h``.
    sim_result = adam_cpp.simulate(
        matrixErrors=mat_errors_f,
        matrixOt=mat_ot_f,
        arrayVt=arr_vt_f,
        matrixWt=mat_wt_f,
        arrayF=arr_f_f,
        matrixG=mat_g_f,
        indexLookupTable=lookup_f,
        profilesRecent=profiles_recent,
        E=e_type_modified,
        refineHead=False,
    )

    y_simulated = sim_result.data  # Shape: (h, nsim)

    # 9. Handle cumulative forecasts
    n_levels = len(level_low)
    if general_dict.get("cumulative", False):
        # The h steps after the sample, with the origin k periods before its end, in
        # the space of the data for a model in logs
        paths = y_simulated[int(general_dict.get("_gap_shift", 0) or 0) :]
        cum_sums = np.sum(
            np.exp(paths) if general_dict.get("_log_model") else paths, axis=0
        )
        y_forecast_sim = np.mean(cum_sums)
        y_lower = np.nanquantile(cum_sums, level_low).reshape(1, n_levels)
        y_upper = np.nanquantile(cum_sums, level_up).reshape(1, n_levels)
    else:
        # 10. Calculate quantiles for each horizon
        y_lower = np.zeros((h, n_levels))
        y_upper = np.zeros((h, n_levels))
        y_forecast_sim = np.zeros(h)

        for i in range(h):
            # Point forecasts come from adam_cpp.forecast(); the sim mean is
            # kept here only for diagnostics / cross-checks.
            y_forecast_sim[i] = np.mean(y_simulated[i, :])

            y_lower[i, :] = np.nanquantile(y_simulated[i, :], level_low)
            y_upper[i, :] = np.nanquantile(y_simulated[i, :], level_up)

    # 11. Convert to relative form (like parametric intervals)
    # Use (h, 1) broadcasting for 2D y_lower/y_upper
    if e_type == "A":
        pred_col = predictions.reshape(-1, 1)
        y_lower = y_lower - pred_col
        y_upper = y_upper - pred_col
    else:
        pred_col = predictions.reshape(-1, 1)
        y_lower = np.where(pred_col != 0, y_lower / pred_col, 0)
        y_upper = np.where(pred_col != 0, y_upper / pred_col, 0)

    # No value as zero, for both the additive and the multiplicative error, as R
    y_lower = np.where(np.isnan(y_lower), 0.0, y_lower)
    y_upper = np.where(np.isnan(y_upper), 0.0, y_upper)

    scenarios_out = y_simulated if general_dict.get("scenarios", False) else None
    return y_lower, y_upper, scenarios_out, y_forecast_sim


# Distributions where errors are normalised by fitted values for semiparametric/
# empirical/nonparametric intervals when Etype=="A".
_LOG_DISTS = frozenset({"dinvgauss", "dgamma", "dlnorm", "dls", "dllaplace", "dlgnorm"})


def _fit_power_quantile(errors, level):
    """Power-law quantile ``A1 * j^A2``, ``j = 1..h``, of the multistep errors.

    Mirrors R's ``adam_quantilePower`` (Taylor & Bunn): for a given ``A2`` the
    pinball loss is minimised by the weighted quantile of ``e / j^A2`` with
    weights ``j^A2``, so only ``A2`` is optimised, with NLopt's Nelder-Mead.

    Parameters
    ----------
    errors : np.ndarray, shape (T-h, h)
    level : float
        Quantile level in [0, 1].

    Returns
    -------
    np.ndarray, shape (h,)
    """
    errors = np.asarray(errors, dtype=np.float64)
    h = errors.shape[1]
    # Column-major, as R's errors[!is.na(errors)] and col(errors)
    values = errors.ravel(order="F")
    horizons = np.repeat(np.arange(1, h + 1, dtype=np.float64), errors.shape[0])
    keep = ~np.isnan(values)
    values, horizons = values[keep], horizons[keep]

    def quantile_a1(power):
        weights = horizons**power
        ratios = values / weights
        ordering = np.argsort(ratios, kind="stable")
        # R's cumsum() accumulates in long double and stores doubles
        cumulative = np.cumsum(weights[ordering].astype(np.longdouble)).astype(
            np.float64
        )
        return ratios[ordering][np.argmax(cumulative >= level * cumulative[-1])]

    def pinball(x, grad):
        residuals = values - quantile_a1(x[0]) * horizons ** x[0]
        return float(
            (1 - level) * _sum_r(np.abs(residuals[residuals < 0]))
            + level * _sum_r(np.abs(residuals[residuals >= 0]))
        )

    opt = nlopt.opt(nlopt.LN_NELDERMEAD, 1)
    opt.set_min_objective(pinball)
    opt.set_lower_bounds([-2.0])
    opt.set_upper_bounds([4.0])
    # Explicitly, as R's nloptr would otherwise default xtol_rel to 1e-4
    opt.set_xtol_rel(1e-8)
    opt.set_xtol_abs(1e-6)
    opt.set_maxeval(500)
    power = opt.optimize([0.5])[0]
    return quantile_a1(power) * np.arange(1, h + 1, dtype=np.float64) ** power


def generate_multistep_interval(
    predictions,
    prepared_model,
    general_dict,
    observations_dict,
    model_type_dict,
    lags_dict,
    params_info,
    adam_cpp,
    level_low,
    level_up,
    interval_type,
):
    """Generate semiparametric, empirical, or nonparametric prediction intervals.

    All three interval types use the ``(T - h + 1) × h`` multistep in-sample
    error matrix produced by ``ferrors``.

    Returns
    -------
    tuple
        (y_lower, y_upper) each of shape (h, n_levels), relative to the forecast,
        which ``_interval_bounds`` puts around it.
    """
    h = general_dict["h"]
    obs = observations_dict["obs_in_sample"]
    e_type = model_type_dict["error_type"]
    distribution = general_dict["distribution"]
    cumulative = general_dict.get("cumulative", False)
    # The cumulative forecast sums the h steps after the sample, with the origin k
    # periods before its end
    shift = int(general_dict.get("_gap_shift", 0) or 0)

    mat_wt = np.asfortranarray(prepared_model["measurement"], dtype=np.float64)
    mat_f = np.asfortranarray(prepared_model["transition"], dtype=np.float64)

    if h > 1:
        adam_errors = _compute_multistep_errors(
            adam_cpp,
            prepared_model,
            observations_dict,
            lags_dict,
            general_dict,
            mat_wt,
            mat_f,
        )
    else:
        # A copy: under pandas' copy-on-write the array of the residuals is read-only
        adam_errors = np.array(prepared_model["residuals"], dtype=float).reshape(-1, 1)
        # No residual at the missing values
        adam_errors[~observed_mask(observations_dict)] = np.nan

    # The errors relative to the fitted values for the ratio distributions with an
    # additive error, at h=1 as for the multistep ones
    if distribution in _LOG_DISTS and e_type == "A":
        y_fitted = np.asarray(prepared_model["y_fitted"], dtype=float)
        if h > 1:
            fitted_matrix = np.column_stack(
                [y_fitted[i - 1 : obs - h + i] for i in range(1, h + 1)]
            )
        else:
            fitted_matrix = y_fitted.reshape(-1, 1)
        adam_errors = adam_errors / fitted_matrix
    # The windows with all their targets observed
    adam_errors = adam_errors[~np.any(np.isnan(adam_errors), axis=1)]
    n = len(adam_errors)

    if interval_type == "semiparametric":
        if cumulative:
            vcov = float(np.sum((adam_errors.T @ adam_errors / n)[shift:, shift:]))
        else:
            vcov = np.diag(adam_errors.T @ adam_errors / n)
        # The log-normal takes the variance of log(1+e), as at h=1
        if distribution == "dlnorm":
            vcov = np.log(1 + vcov)
        return generate_prediction_interval(
            predictions,
            prepared_model,
            general_dict,
            observations_dict,
            model_type_dict,
            lags_dict,
            params_info,
            level_low,
            level_up,
            # At h=1, R uses the variance of the model, as the approximate interval
            scale_2d_override=np.atleast_1d(vcov).reshape(-1, 1) if h > 1 else None,
        )

    n_levels = len(level_low)
    yf = np.atleast_1d(np.asarray(predictions, dtype=float))

    if cumulative:
        cum_errors = np.sum(adam_errors[:, shift:], axis=1)
        y_lower = np.array([[np.quantile(cum_errors, q) for q in level_low]])
        y_upper = np.array([[np.quantile(cum_errors, q) for q in level_up]])
    elif interval_type == "empirical" or h == 1:
        y_lower = np.zeros((h, n_levels))
        y_upper = np.zeros((h, n_levels))
        for i in range(h):
            col = adam_errors[:, i]
            y_lower[i] = [np.quantile(col, q) for q in level_low]
            y_upper[i] = [np.quantile(col, q) for q in level_up]
    else:  # nonparametric, h > 1
        y_lower = np.column_stack(
            [_fit_power_quantile(adam_errors, q) for q in level_low]
        )
        y_upper = np.column_stack(
            [_fit_power_quantile(adam_errors, q) for q in level_up]
        )

    # The errors relative to the forecast: those of a multiplicative error, and of
    # the ratio distributions with an additive one (R: the quantile times the
    # forecast, which the forecast is added to)
    if e_type == "M":
        return 1.0 + y_lower, 1.0 + y_upper
    if distribution in _LOG_DISTS:
        return y_lower * yf.reshape(-1, 1), y_upper * yf.reshape(-1, 1)
    return y_lower, y_upper
