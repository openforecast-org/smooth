"""Simulation entry points — Python ports of R's ``sim.*`` family.

* :func:`sim_es` — port of R's ``sim.es``
* :class:`SimulateResult` — container mirroring R's ``smooth.sim`` /
  ``oes.sim`` S3 list.
* :func:`resolve_randomizer` — R-style randomizer dispatch (accepts
  callables for the parity-test plug-in-numbers trick).

The remaining ``sim_gum``, ``sim_ces``, ``sim_ssarima``, ``sim_sma``,
``sim_oes`` ports arrive in later phases.
"""

from smooth.adam_general.core.simulate.ces import sim_ces
from smooth.adam_general.core.simulate.es import sim_es
from smooth.adam_general.core.simulate.gum import sim_gum
from smooth.adam_general.core.simulate.oes import sim_oes
from smooth.adam_general.core.simulate.randomizer import (
    is_default_randomizer,
    resolve_randomizer,
)
from smooth.adam_general.core.simulate.result import SimulateResult
from smooth.adam_general.core.simulate.sma import sim_sma
from smooth.adam_general.core.simulate.ssarima import sim_ssarima

__all__ = [
    "SimulateResult",
    "is_default_randomizer",
    "resolve_randomizer",
    "sim_ces",
    "sim_es",
    "sim_gum",
    "sim_oes",
    "sim_sma",
    "sim_ssarima",
]
