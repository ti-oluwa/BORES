"""
Hysteresis models for relative permeability.

The old module path `bores.blackoil.satfunc.relperm.hysteresis` still provides everything it used to.

**References**:

- Killough, J.E. (1976). "Reservoir Simulation With History-Dependent Saturation Functions".
  SPE 5106.
- Land, C.S. (1968). "Calculation of Imbibition Relative Permeability for Two- and Three-Phase
  Flow from Rock Properties". SPE 1942.
- Carlson, F.M. (1981). "Simulation of Relative Permeability Hysteresis to the Non-Wetting
  Phase". SPE 10157.
"""

from bores.blackoil.satfunc.relperm.hysteresis.base import HysteresisRelPermTable
from bores.blackoil.satfunc.relperm.hysteresis.carlson import CarlsonRelPermTable
from bores.blackoil.satfunc.relperm.hysteresis.killough.blended import BlendedKilloughRelPermTable
from bores.blackoil.satfunc.relperm.hysteresis.killough.scaled import KilloughRelPermTable
from bores.blackoil.satfunc.relperm.hysteresis.land import (
    LandTrapping,
    compute_land_residual_saturation,
    compute_land_trapping_coefficient,
)

__all__ = [
    "BlendedKilloughRelPermTable",
    "CarlsonRelPermTable",
    "HysteresisRelPermTable",
    "KilloughRelPermTable",
    "LandTrapping",
    "compute_land_residual_saturation",
    "compute_land_trapping_coefficient",
]
