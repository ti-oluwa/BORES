"""
Killough (1976) relative permeability hysteresis, in two forms.

- `KilloughRelPermTable` (`scaled.py`): Killough's scheme. The non-wetting phase scans along the
  imbibition curve with its saturation axis stretched to the trapped saturation, and the wetting
  phase has a scheme of its own.
- `BlendedKilloughRelPermTable` (`blended.py`): BORES's own scheme, where every phase is a weighted
  blend of its drainage and imbibition curves.

- Killough, J.E. (1976). "Reservoir Simulation With History-Dependent Saturation Functions".
  SPE 5106.
"""

from bores.blackoil.satfunc.relperm.hysteresis.killough.blended import (
    BlendedKilloughRelPermTable,
)
from bores.blackoil.satfunc.relperm.hysteresis.killough.scaled import KilloughRelPermTable

__all__ = ["BlendedKilloughRelPermTable", "KilloughRelPermTable"]
