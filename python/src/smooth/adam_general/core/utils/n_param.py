"""
Parameter counting utilities for ADAM models.

This module provides the NParam class for tracking estimated and provided parameters
across different model components (internal, xreg, occurrence, scale).
"""

from typing import Dict


class NParam:
    """
    Parameter count table for ADAM models.

    Tracks the number of estimated and provided parameters across different
    categories: internal (ETS+ARIMA+constant), xreg (exogenous regressors),
    occurrence (intermittent demand models), and scale (distribution parameters).

    Attributes
    ----------
    estimated : dict
        Dictionary with keys 'internal', 'xreg', 'occurrence', 'scale', 'all'
        containing counts of estimated parameters.
    provided : dict
        Dictionary with keys 'internal', 'xreg', 'occurrence', 'scale', 'all'
        containing counts of provided (fixed) parameters.

    Examples
    --------
    >>> n_param = NParam()
    >>> n_param.estimated['internal'] = 3
    >>> n_param.estimated['scale'] = 1
    >>> n_param.update_totals()
    >>> print(n_param.estimated['all'])
    4
    >>> print(n_param)
                      internal  xreg  occurrence  scale  all
    estimated                3     0           0      1    4
    provided                 0     0           0      0    0
    """

    def __init__(self):
        """Initialize empty parameter count table."""
        self.estimated = {
            "internal": 0,
            "xreg": 0,
            "occurrence": 0,
            "scale": 0,
            "all": 0,
        }
        self.provided = {
            "internal": 0,
            "xreg": 0,
            "occurrence": 0,
            "scale": 0,
            "all": 0,
        }

    def update_totals(self):
        """Update the 'all' column by summing other columns."""
        self.estimated["all"] = (
            self.estimated["internal"]
            + self.estimated["xreg"]
            + self.estimated["occurrence"]
            + self.estimated["scale"]
        )
        self.provided["all"] = (
            self.provided["internal"]
            + self.provided["xreg"]
            + self.provided["occurrence"]
            + self.provided["scale"]
        )

    @property
    def n_param_estimated(self) -> int:
        """Total number of estimated parameters (for degrees of freedom)."""
        return self.estimated["all"]

    @property
    def n_param_provided(self) -> int:
        """Total number of provided parameters."""
        return self.provided["all"]

    @property
    def n_param_for_variance(self) -> int:
        """Number of parameters for variance calculation (all - scale)."""
        return self.estimated["all"] - self.estimated["scale"]

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {"estimated": self.estimated.copy(), "provided": self.provided.copy()}

    @classmethod
    def from_dict(cls, data: Dict) -> "NParam":
        """Create NParam from dictionary representation."""
        n_param = cls()
        if "estimated" in data:
            n_param.estimated.update(data["estimated"])
        if "provided" in data:
            n_param.provided.update(data["provided"])
        return n_param

    def __str__(self) -> str:
        """Return formatted string representation like R's output."""
        # Column headers
        cols = ["internal", "xreg", "occurrence", "scale", "all"]
        col_width = 10

        # Build header
        header = " " * 10  # Row label space
        for col in cols:
            header += f"{col:>{col_width}}"

        # Build rows
        est_row = "estimated "
        for col in cols:
            val = self.estimated[col]
            est_row += f"{float(val):>{col_width}.3f}"

        prov_row = "provided  "
        for col in cols:
            val = self.provided[col]
            prov_row += f"{float(val):>{col_width}.3f}"

        return f"{header}\n{est_row}\n{prov_row}"

    def __repr__(self) -> str:
        return f"NParam(estimated={self.estimated}, provided={self.provided})"


def create_n_param(
    n_param_internal_estimated: int = 0,
    n_param_internal_provided: int = 0,
    n_param_xreg_estimated: int = 0,
    n_param_xreg_provided: int = 0,
    n_param_occurrence_estimated: int = 0,
    n_param_occurrence_provided: int = 0,
    n_param_scale_estimated: int = 0,
    n_param_scale_provided: int = 0,
) -> NParam:
    """
    Create an NParam table with specified values.

    Parameters
    ----------
    n_param_internal_estimated : int
        Number of estimated internal parameters (ETS + ARIMA + constant)
    n_param_internal_provided : int
        Number of provided internal parameters
    n_param_xreg_estimated : int
        Number of estimated exogenous regressor parameters
    n_param_xreg_provided : int
        Number of provided exogenous regressor parameters
    n_param_occurrence_estimated : int
        Number of estimated occurrence model parameters
    n_param_occurrence_provided : int
        Number of provided occurrence model parameters
    n_param_scale_estimated : int
        Number of estimated scale parameters (1 if likelihood used)
    n_param_scale_provided : int
        Number of provided scale parameters

    Returns
    -------
    NParam
        Initialized parameter count table
    """
    n_param = NParam()
    n_param.estimated["internal"] = n_param_internal_estimated
    n_param.estimated["xreg"] = n_param_xreg_estimated
    n_param.estimated["occurrence"] = n_param_occurrence_estimated
    n_param.estimated["scale"] = n_param_scale_estimated

    n_param.provided["internal"] = n_param_internal_provided
    n_param.provided["xreg"] = n_param_xreg_provided
    n_param.provided["occurrence"] = n_param_occurrence_provided
    n_param.provided["scale"] = n_param_scale_provided

    n_param.update_totals()
    return n_param


def count_internal_params(
    model_type_dict: Dict,
    persistence_checked: Dict,
    initials_checked: Dict,
    arima_checked: Dict,
    phi_dict: Dict,
    constants_checked: Dict,
) -> tuple:
    """
    Count internal parameters (ETS + ARIMA + constant).

    Parameters
    ----------
    model_type_dict : dict
        Model type information
    persistence_checked : dict
        Persistence parameter information
    initials_checked : dict
        Initial state information
    arima_checked : dict
        ARIMA parameter information
    phi_dict : dict
        Damping parameter information
    constants_checked : dict
        Constant/drift parameter information

    Returns
    -------
    tuple
        (n_estimated, n_provided) counts of internal parameters
    """
    n_estimated = 0
    n_provided = 0

    # ETS persistence parameters
    if model_type_dict.get("ets_model", False):
        # Alpha (level)
        if persistence_checked.get("persistence_level_estimate", True):
            n_estimated += 1
        else:
            n_provided += 1

        # Beta (trend)
        if model_type_dict.get("trend_type", "N") != "N":
            if persistence_checked.get("persistence_trend_estimate", True):
                n_estimated += 1
            else:
                n_provided += 1

        # Gamma (seasonal) - can be multiple
        n_seasonal = persistence_checked.get("persistence_seasonal_number", 0)
        if n_seasonal > 0:
            if persistence_checked.get("persistence_seasonal_estimate", True):
                n_estimated += n_seasonal
            else:
                n_provided += n_seasonal

    # Phi (damping parameter)
    if phi_dict.get("phi_estimate", False):
        n_estimated += 1
    elif phi_dict.get("phi_required", False):
        n_provided += 1

    # Initial states — only counted when optimised (optimal/two-stage),
    # not for backcasting/complete
    initial_type = initials_checked.get("initial_type", "optimal")
    initial_optimised = initial_type in ["optimal", "two-stage"]

    if initial_optimised and initials_checked.get("initial_estimate", True):
        # Level initial
        if initials_checked.get("initial_level_estimate", True):
            n_estimated += 1
        else:
            n_provided += 1

        # Trend initial
        if initials_checked.get("initial_trend_estimate", False):
            n_estimated += 1
        elif model_type_dict.get("trend_type", "N") != "N":
            n_provided += 1

        # Seasonal initials
        n_seasonal_init = initials_checked.get("initial_seasonal_number", 0)
        if n_seasonal_init > 0:
            if initials_checked.get("initial_seasonal_estimate", True):
                n_estimated += n_seasonal_init
            else:
                n_provided += n_seasonal_init
    elif not initials_checked.get("initial_estimate", True):
        # All initials explicitly provided (fixed)
        if model_type_dict.get("ets_model", False):
            n_provided += 1  # level
            if model_type_dict.get("trend_type", "N") != "N":
                n_provided += 1
            n_provided += initials_checked.get("initial_seasonal_number", 0)

    # ARIMA parameters
    if arima_checked.get("arima_model", False):
        ar_orders = arima_checked.get("ar_orders", [0])
        ma_orders = arima_checked.get("ma_orders", [0])
        n_ar = sum(ar_orders) if ar_orders else 0
        n_ma = sum(ma_orders) if ma_orders else 0

        if arima_checked.get("ar_estimate", True):
            n_estimated += n_ar
        else:
            n_provided += n_ar

        if arima_checked.get("ma_estimate", True):
            n_estimated += n_ma
        else:
            n_provided += n_ma

    # Constant/drift
    if constants_checked.get("constant_estimate", False):
        n_estimated += 1
    elif constants_checked.get("constant_required", False):
        n_provided += 1

    return n_estimated, n_provided


def count_xreg_params(explanatory_checked: Dict) -> tuple:
    """
    Count exogenous regressor parameters.

    Parameters
    ----------
    explanatory_checked : dict
        Explanatory variable information

    Returns
    -------
    tuple
        (n_estimated, n_provided) counts of xreg parameters
    """
    n_estimated = 0
    n_provided = 0

    if explanatory_checked.get("xreg_model", False):
        n_xreg = explanatory_checked.get("xreg_number", 0)
        if explanatory_checked.get("xreg_estimate", True):
            n_estimated += n_xreg
        else:
            n_provided += n_xreg

        # Xreg persistence (if dynamic regressors)
        n_xreg_persistence = explanatory_checked.get("xreg_persistence_number", 0)
        if n_xreg_persistence > 0:
            if explanatory_checked.get("xreg_persistence_estimate", True):
                n_estimated += n_xreg_persistence
            else:
                n_provided += n_xreg_persistence

    return n_estimated, n_provided


def build_n_param_table(
    model_type_dict: Dict,
    persistence_checked: Dict,
    initials_checked: Dict,
    arima_checked: Dict,
    phi_dict: Dict,
    constants_checked: Dict,
    explanatory_checked: Dict,
    general: Dict,
) -> NParam:
    """
    Build complete parameter count table.

    Parameters
    ----------
    model_type_dict : dict
        Model type information
    persistence_checked : dict
        Persistence parameter information
    initials_checked : dict
        Initial state information
    arima_checked : dict
        ARIMA parameter information
    phi_dict : dict
        Damping parameter information
    constants_checked : dict
        Constant/drift parameter information
    explanatory_checked : dict
        Explanatory variable information
    general : dict
        General settings including loss function

    Returns
    -------
    NParam
        Complete parameter count table
    """
    n_param = NParam()

    # Count internal parameters
    n_internal_est, n_internal_prov = count_internal_params(
        model_type_dict,
        persistence_checked,
        initials_checked,
        arima_checked,
        phi_dict,
        constants_checked,
    )
    n_param.estimated["internal"] = n_internal_est
    n_param.provided["internal"] = n_internal_prov

    # Count xreg parameters — for "complete" initial type, xreg is estimated via
    # backcasting (not in B), so R does not count it as a free parameter
    n_xreg_est, n_xreg_prov = count_xreg_params(explanatory_checked)
    if initials_checked.get("initial_type") == "complete":
        n_xreg_prov += n_xreg_est
        n_xreg_est = 0
    n_param.estimated["xreg"] = n_xreg_est
    n_param.provided["xreg"] = n_xreg_prov

    # Occurrence parameters (not yet implemented)
    n_param.estimated["occurrence"] = 0
    n_param.provided["occurrence"] = 0

    # Scale parameter (1 if likelihood was used)
    if general.get("loss", "likelihood") == "likelihood":
        n_param.estimated["scale"] = 1
    else:
        n_param.estimated["scale"] = 0
    n_param.provided["scale"] = 0

    # Update totals
    n_param.update_totals()

    return n_param
