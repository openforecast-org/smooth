import numpy as np

from ._utils import _warn


def _check_occurrence(data, occurrence, silent=False, holdout=False, h=0):
    """
    Check and handle 'occurrence' parameter for intermittent demand data.

    Parameters
    ----------
    data : array-like
        Input time series data
    occurrence : str
        Occurrence type ('none', 'auto', 'fixed', etc.)
    silent : bool, optional
        Whether to suppress warnings
    holdout : bool, optional
        Whether to use holdout
    h : int, optional
        Forecast horizon

    Returns
    -------
    dict
        Dictionary with occurrence details and nonzero counts
    """
    data_list = list(data) if not isinstance(data, list) else data
    obs_in_sample = len(data_list)
    obs_all = obs_in_sample + (h if holdout else 0)
    # Identify non-zero observations
    nonzero_indices = [
        i for i, val in enumerate(data_list) if val is not None and val != 0
    ]
    obs_nonzero = len(nonzero_indices)

    # If all zeroes, fallback
    if all(val == 0 for val in data):
        _warn("You have a sample with zeroes only. Your forecast will be zero.", silent)
        return {
            "occurrence": "none",
            "occurrence_model": False,
            "obs_in_sample": obs_in_sample,
            "obs_nonzero": 0,
            "obs_all": obs_all,
        }

    # Validate the occurrence choice
    valid_occ = [
        "none",
        "auto",
        "fixed",
        "general",
        "odds-ratio",
        "inverse-odds-ratio",
        "direct",
        "provided",
    ]
    if occurrence not in valid_occ:
        _warn(f"Invalid occurrence: {occurrence}. Switching to 'none'.", silent)
        occurrence = "none"

    occurrence_model = occurrence not in ["none", "provided"]
    return {
        "occurrence": occurrence,
        "occurrence_model": occurrence_model,
        "obs_in_sample": obs_in_sample,
        "obs_nonzero": obs_nonzero,
        "obs_all": obs_all,
    }


def _check_lags(lags, obs_in_sample, silent=False):
    """
    Validate or adjust the set of lags.

    Parameters
    ----------
    lags : list
        List of lag values
    obs_in_sample : int
        Number of in-sample observations
    silent : bool, optional
        Whether to suppress warnings

    Returns
    -------
    dict
        Dictionary with lags information including seasonal lags
    """
    # Handle None or empty lags - default to [1]
    if lags is None:
        lags = [1]
    # Handle scalar lags - wrap in list for convenience
    elif isinstance(lags, (int, float, np.integer, np.floating)):
        lags = [int(lags)]

    # Remove any zero-lags
    lags = [lg for lg in lags if lg != 0]

    # Force 1 in lags (for level)
    if 1 not in lags:
        lags.insert(0, 1)

    # Must be positive
    if any(lg <= 0 for lg in lags):
        raise ValueError(
            "Right! Why don't you try complex lags then, "
            "mister smart guy? (Lag <= 0 given)"
        )

    # Create lagsModel (matrix in R, list here)
    lags_model = sorted(set(lags))

    # Get seasonal lags (all lags > 1)
    lags_model_seasonal = [lag for lag in lags_model if lag > 1]
    max_lag = max(lags) if lags else 1

    if max_lag >= obs_in_sample:
        msg = (
            f"The maximum lags value is {max_lag}, "
            f"while sample size is {obs_in_sample}. "
            f"I cannot guarantee that I'll be able to fit the model."
        )
        _warn(msg, silent)

    return {
        "lags": sorted(set(lags)),
        "lags_model": lags_model,
        "lags_model_seasonal": lags_model_seasonal,
        "lags_length": len(lags_model),
        "max_lag": max_lag,
    }


def _calculate_ot_logical(
    data,
    occurrence,
    occurrence_model,
    obs_in_sample,
    h=0,
    holdout=False,
):
    """
    Calculate logical observation vector and observation time indices.

    Parameters
    ----------
    data : array-like
        Input time series data
    occurrence : str
        Occurrence type
    occurrence_model : bool
        Whether occurrence model is used
    obs_in_sample : int
        Number of in-sample observations
    h : int, optional
        Forecast horizon
    holdout : bool, optional
        Whether to use holdout data

    Returns
    -------
    dict
        Dictionary with observation information
    """
    # Convert data to numpy array if needed
    if hasattr(data, "values"):
        y_in_sample = (
            data.values.flatten() if hasattr(data.values, "flatten") else data.values
        )
    else:
        y_in_sample = np.asarray(data).flatten()

    # Handle holdout if requested and possible
    y_holdout = None
    if holdout and h > 0 and len(y_in_sample) > h:
        # Split the data
        y_holdout = y_in_sample[-h:]
        y_in_sample = y_in_sample[:-h]

    # Initial calculation - data != 0
    ot_logical = y_in_sample != 0

    # If occurrence is "none" and all values are non-zero, set all to True
    if occurrence == "none" and all(ot_logical):
        ot_logical = np.ones_like(ot_logical, dtype=bool)

    # If occurrence model is not used and occurrence is not "provided"
    if not occurrence_model and occurrence != "provided":
        ot_logical = np.ones_like(ot_logical, dtype=bool)

    # Determine frequency from pandas index if available
    freq = "1"
    if (
        hasattr(data, "index")
        and hasattr(data.index, "freq")
        and data.index.freq is not None
    ):
        freq = data.index.freq

    # Get start time if available
    y_start = 0  # Default
    if hasattr(data, "index") and len(data.index) > 0:
        y_start = data.index[0]

    # Handle forecast start time
    if hasattr(data, "index") and len(data.index) > 0:
        if holdout and h > 0:
            y_forecast_start = data.index[-h]
        else:
            # Last data point + 1 period
            try:
                # For DatetimeIndex
                from pandas import DatetimeIndex

                if isinstance(data.index, DatetimeIndex):
                    # Get the last index and add one frequency unit
                    from pandas import Timedelta

                    last_idx = data.index[-1]
                    freq_delta = Timedelta(freq)
                    y_forecast_start = last_idx + freq_delta
                else:
                    # For numeric index
                    if hasattr(data.index, "freq") and data.index.freq is not None:
                        y_forecast_start = data.index[-1] + data.index.freq
                    else:
                        # Fallback for numeric index without freq - use integer
                        y_forecast_start = int(data.index[-1]) + 1
            except (ImportError, AttributeError, ValueError):
                # Fallback: use the last index + freq
                if hasattr(data.index, "freq") and data.index.freq is not None:
                    y_forecast_start = data.index[-1] + data.index.freq
                else:
                    # Ultimate fallback for numeric index - use integer
                    y_forecast_start = int(data.index[-1]) + 1
    else:
        # For non-indexed data, just use the total length
        y_forecast_start = len(y_in_sample)

    # Create basic result
    result = {
        "ot_logical": ot_logical,
        "ot": np.where(ot_logical, 1, 0),
        "y_in_sample": y_in_sample,
        "y_holdout": y_holdout,
        "frequency": freq,
        "y_start": y_start,
        "y_forecast_start": y_forecast_start,
    }

    # Add index information if available
    if hasattr(data, "index"):
        if holdout and h > 0:
            result["y_in_sample_index"] = data.index[:-h]
            result["y_forecast_index"] = data.index[-h:]
        else:
            result["y_in_sample_index"] = data.index
            # Create forecast index
            try:
                import pandas as pd

                result["y_forecast_index"] = pd.date_range(
                    start=y_forecast_start, periods=h, freq=freq
                )
            except (ImportError, ValueError, TypeError):
                # Fallback for non-time data
                result["y_forecast_index"] = None

    return result
