"""
PVT (Pressure-Volume-Temperature) property tables for reservoir simulation.

Architecture overview
---------------------
This module provides three layers:

**Data layer** - `PVTData`, `PVTDataSet`
    Raw tabulated arrays exactly as read from a deck or built by a correlation
    builder. No interpolators. Fully serialisable. Units are always FIELD
    (psi, °F, bbl/STB, ft³/SCF, cP, lbm/ft³, SCF/STB) unless documented
    otherwise.

**Table layer** - `PVTTable`, `PVTTables`, `PVT`
    Wraps `PVTData` with fast scipy / PCHIP interpolators. Every primary
    property (Bo, μo, Rs, Bg, μg, z, Bw, μw) has a matching `*_dp` method
    that returns ∂/∂P via the derivative of the same interpolator - no
    second table lookup, no finite differences. Derived properties (ρ, c)
    are pre-built into their own tables at construction time so simulation
    evaluation is a single interpolator call.

**Region layer** - `PVT`
    `Dict[int, PVTTables]` keyed by 1-based `PVTNUM` region index.
    `from_deck` builds all regions from a parsed `DeckFile`.

Deck keyword support
--------------------
`pvt_regions_from_data_file` detects which Eclipse PVT keywords are present
(`PVTO` > `PVDO` > `PVCO` for oil; `PVTG` > `PVDG` for gas;
`PVTW` for water - always analytical) and builds one `PVTTables` per
`PVTNUM` region.

Property construction rule
--------------------------
*Only the minimum deck columns are interpolated directly.*  Everything else
is built into a table at construction time using the standard formulas:

    ρo,res = (ρo,SC + Rs · ρg,SC) / Bo
    ρg,res = (ρg,SC + Rv · ρo,SC) / Bg          [wet gas]
    ρg,res = ρg,SC / Bg                          [dry gas]
    ρw,res = ρw,SC / Bw
    co     = -(1/Bo) · (∂Bo/∂P)
    cg     = 1/P - (1/z) · (∂z/∂P)
    cw     = -(1/Bw) · (∂Bw/∂P)  or  PVTW constant

This means that at simulation time every property - primary or derived - is
evaluated by a single interpolator call with no per-step arithmetic.
"""

from .data import *
from .factories import *
from .regions import *
from .static import *
from .tables import *
