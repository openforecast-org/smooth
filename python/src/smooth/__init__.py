from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from smooth.adam_general.core.adam import ADAM
from smooth.adam_general.core.auto_adam import AutoADAM
from smooth.adam_general.core.auto_msarima import AutoMSARIMA
from smooth.adam_general.core.auto_om import AutoOM
from smooth.adam_general.core.ces_model import CES, AutoCES
from smooth.adam_general.core.es import ES
from smooth.adam_general.core.msarima import MSARIMA
from smooth.adam_general.core.om import OM
from smooth.adam_general.core.omg import OMG
from smooth.adam_general.core.simulate import (
    SimulateResult,
    sim_ces,
    sim_es,
    sim_gum,
    sim_oes,
    sim_sma,
    sim_ssarima,
)
from smooth.adam_general.core.sma import SMA
from smooth.adam_general.core.utils.utils import msdecompose
from smooth.utils import show_versions

try:
    __version__ = _pkg_version("smooth")
except PackageNotFoundError:  # editable install before metadata is built
    __version__ = "0.0.0+unknown"

__all__ = [
    "ADAM",
    "AutoADAM",
    "AutoMSARIMA",
    "AutoOM",
    "CES",
    "AutoCES",
    "ES",
    "MSARIMA",
    "OM",
    "OMG",
    "SMA",
    "SimulateResult",
    "__version__",
    "msdecompose",
    "show_versions",
    "sim_ces",
    "sim_es",
    "sim_gum",
    "sim_oes",
    "sim_sma",
    "sim_ssarima",
]
