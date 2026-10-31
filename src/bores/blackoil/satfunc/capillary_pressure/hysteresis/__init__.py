"""
Hysteresis models for capillary pressure.

The old module path `bores.blackoil.satfunc.capillary_pressure.hysteresis` still provides everything
it used to.

**References**:

- Killough, J.E. (1976). "Reservoir Simulation With History-Dependent Saturation Functions".
  SPE 5106.
"""

from bores.blackoil.satfunc.capillary_pressure.hysteresis.base import (
    HysteresisCapillaryPressureTable,
)
from bores.blackoil.satfunc.capillary_pressure.hysteresis.killough import (
    KilloughCapillaryPressureTable,
)

__all__ = ["HysteresisCapillaryPressureTable", "KilloughCapillaryPressureTable"]
