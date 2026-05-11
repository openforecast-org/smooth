import numpy as np
import pandas as pd

from smooth.adam_general.core.creator import adam_profile_creator


def _compute_multistep_errors(
    adam_cpp,
    prepared_model,
    observations_dict,
    lags_dict,
    general_dict,
    mat_wt,
    mat_f,
):
    """Call adam_cpp.ferrors() and return the (T-h) × h in-sample error matrix.

    Uses the INITIAL profile and an in-sample lookup table — same as R's
    rmultistep.adam() which calls adamCpp$ferrors with object$profileInitial
    and adamProfileCreator(lagsModelAll, lagsModelMax, obsInSample, ...).
    """
    h = general_dict["h"]
    obs = observations_dict["obs_in_sample"]
    lags_model_max = lags_dict["lags_model_max"]

    profile_result = adam_profile_creator(
        lags_model_all=lags_dict["lags_model_all"],
        lags_model_max=lags_model_max,
        obs_all=obs,
        lags=lags_dict["lags"],
    )
    idx_table = np.asfortranarray(
        profile_result["index_lookup_table"][:, lags_model_max:], dtype=np.uint64
    )

    profiles_initial = np.asfortranarray(
        prepared_model["mat_vt"][:, :lags_model_max], dtype=np.float64
    )

    y_in_sample = np.asarray(
        observations_dict["y_in_sample"], dtype=np.float64
    ).reshape(-1, 1)

    result = adam_cpp.ferrors(
        matrixVt=prepared_model["states"],
        matrixWt=mat_wt,
        matrixF=mat_f,
        indexLookupTable=idx_table,
        profilesRecent=profiles_initial,
        horizon=h,
        vectorYt=y_in_sample,
    )
    return np.asarray(result.errors)


def _safe_create_index(start, periods, freq):
    """
    Safely create a pandas index, handling both datetime and numeric cases.

    Parameters
    ----------
    start : any
        Start of the index (can be timestamp, int, etc.)
    periods : int
        Number of periods
    freq : any
        Frequency (can be pandas freq string or numeric)

    Returns
    -------
    pandas.Index
        Either DatetimeIndex or RangeIndex depending on input types
    """
    # If start is numeric, use RangeIndex
    if isinstance(start, (int, float, np.integer, np.floating)):
        return pd.RangeIndex(start=int(start), stop=int(start) + periods)

    # If freq is numeric (not a valid pandas frequency string), use RangeIndex
    if isinstance(freq, (int, float, np.integer, np.floating)):
        # Try to infer start as numeric
        if hasattr(start, "__len__") or start is None:
            return pd.RangeIndex(start=0, stop=periods)
        try:
            start_int = int(start)
            return pd.RangeIndex(start=start_int, stop=start_int + periods)
        except (TypeError, ValueError):
            pass

    # Try to create DatetimeIndex
    try:
        return pd.date_range(start=start, periods=periods, freq=freq)
    except (TypeError, ValueError):
        # Fallback to RangeIndex
        return pd.RangeIndex(start=0, stop=periods)


def _prepare_lookup_table(lags_dict, observations_dict, general_dict):
    """
    Prepare lookup table for forecasting.

    Parameters
    ----------
    lags_dict : dict
        Dictionary with lag-related information
    observations_dict : dict
        Dictionary with observation data and related information
    general_dict : dict
        Dictionary with general model parameters

    Returns
    -------
    numpy.ndarray
        Lookup table for forecasting
    """
    lookup_result = adam_profile_creator(
        lags_model_all=lags_dict["lags_model_all"],
        lags_model_max=lags_dict["lags_model_max"],
        obs_all=observations_dict["obs_in_sample"] + general_dict["h"],
        lags=lags_dict["lags"],
    )
    lookup = lookup_result["index_lookup_table"][
        :, (observations_dict["obs_in_sample"] + lags_dict["lags_model_max"]) :
    ]
    return lookup


def _prepare_matrices_for_forecast(
    model_prepared, observations_dict, lags_dict, general_dict
):
    """
    Prepare matrices required for forecasting.

    Parameters
    ----------
    model_prepared : dict
        Dictionary with the prepared model including matrices
    observations_dict : dict
        Dictionary with observation data and related information
    lags_dict : dict
        Dictionary with lag-related information
    general_dict : dict
        Dictionary with general model parameters

    Returns
    -------
    tuple
        Tuple containing (mat_vt, mat_wt, vec_g, mat_f)
    """
    # Get state matrix
    mat_vt = model_prepared["states"][
        :,
        observations_dict["obs_states"]
        - lags_dict["lags_model_max"] : observations_dict["obs_states"] + 1,
    ]

    # Get measurement matrix
    if model_prepared["measurement"].shape[0] < general_dict["h"]:
        mat_wt = np.tile(model_prepared["measurement"][-1], (general_dict["h"], 1))
    else:
        mat_wt = model_prepared["measurement"][-general_dict["h"] :]

    # Get other matrices
    vec_g = model_prepared["persistence"]
    mat_f = model_prepared["transition"]

    return mat_vt, mat_wt, vec_g, mat_f
