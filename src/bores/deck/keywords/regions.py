"""
REGIONS section keyword implementations.

The REGIONS section assigns each grid cell to one or more numbered
regions that control which PVT, saturation-function, equilibration, or
rock-compaction table applies to that cell. Every keyword here is a
per-cell integer array (dtype int32) whose value selects the 1-based
table index for the corresponding cell.

Keywords:

- `SATNUM`: saturation-function region (selects `SWOF`/`SGOF`/…
  table).
- `PVTNUM`: PVT region (selects `PVTW`/`PVDO`/`PVTO`/… table).
- `EQLNUM`: equilibration region (selects `EQUIL` record).
- `FIPNUM`: fluid-in-place region (for `ROIP`/`RGIP`/`RWIP`
  reporting).
- `ROCKNUM`: rock-compaction region (selects `ROCK`/`ROCKTAB`
  table).
- `IMBNUM`: imbibition saturation-function region for hysteresis
  modelling.

All keywords default to `1` (first table) when absent.
"""

import numpy as np

from bores.deck.keywords.base import ArrayKeyword

__all__ = ["EQLNUM", "FIPNUM", "IMBNUM", "PVTNUM", "ROCKNUM", "SATNUM"]


SATNUM = ArrayKeyword("SATNUM", dtype=np.int32, default_value=1)
"""`SATNUM` - saturation function region number (1-based int)."""

PVTNUM = ArrayKeyword("PVTNUM", dtype=np.int32, default_value=1)
"""`PVTNUM` - PVT region number (1-based int)."""

EQLNUM = ArrayKeyword("EQLNUM", dtype=np.int32, default_value=1)
"""`EQLNUM` - equilibration region number (1-based int)."""

FIPNUM = ArrayKeyword("FIPNUM", dtype=np.int32, default_value=1)
"""`FIPNUM` - fluid-in-place region number (1-based int)."""

ROCKNUM = ArrayKeyword("ROCKNUM", dtype=np.int32, default_value=1)
"""
`ROCKNUM` - rock-compaction region number (1-based int).
 
Selects which `ROCKTAB` (or single `ROCK` compressibility record) table
applies to each cell, when more than one rock region is defined. Default
is `1` (the first/only rock region) when absent.
"""

IMBNUM = ArrayKeyword("IMBNUM", dtype=np.int32, default_value=1)
"""
`IMBNUM` - imbibition saturation-function region number (1-based int).
 
Selects the imbibition (re-saturation) relative-permeability/capillary-
pressure table for hysteresis modelling, paired with the drainage table
selected by `SATNUM`. Default is `1` when absent.
"""
