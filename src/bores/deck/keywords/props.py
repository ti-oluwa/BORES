"""
PROPS section keyword implementations.

The PROPS section contains fluid and rock property tables used by the
simulator: PVT data (formation volume factors, viscosities, compressibilities)
and relative permeability / capillary pressure curves.

**PVT keywords** (return `List[List[Dict]]` - one inner list per PVT region):

- `DENSITY`: stock-tank densities (oil, water, gas) [one record per region]
- `GRAVITY`: stock-tank gravities (oil API, water and gas specific gravity) [one record per region]
- `EHYSTR`: hysteresis model controls
- `PVTW`: water PVT table (ref pressure, Bw, Cw, viscosity, Cv)
- `PVDO`: dead-oil PVT (pressure, Bo, viscosity) tabulated
- `PVTO`: live-oil PVT (Rs-bracketed) tabulated
- `PVCO`: live-oil PVT with constant compressibility / viscosibility (row per bubble point)
- `PVCDO`: dead-oil PVT with constant compressibility / viscosibility (single record)
- `PVDG`: dry-gas PVT (pressure, Bg, viscosity) tabulated
- `PVTG`: wet-gas PVT (Rv-bracketed) tabulated
- `ROCK`: rock compressibility (ref pressure, compressibility)
- `ROCKTAB`: tabulated pore-volume vs. pressure compaction table
- `AQUTAB`: dimensionless influence function tables for `AQUCT` aquifers

**Relative permeability / capillary pressure keywords**:

- `SWOF`: Sw-indexed: Krw, Krow, Pcow
- `SGOF`: Sg-indexed: Krg, Krog, Pcog
- `SWFN`: Sw-indexed: Krw, Pcow (two-function family)
- `SGFN`: Sg-indexed: Krg, Pcog (two-function family)
- `SOF2`: So-indexed: Kro (two-phase oil)
- `SOF3`: So-indexed: Krow, Krog (three-phase oil)

**End-point scaling keywords** (enabled by `ENDSCALE`; each returns a `(n_cells,)` `float64` array):

- `SWL`, `SWCR`, `SWU`: connate, critical and maximum water saturation.
- `SGL`, `SGCR`, `SGU`: connate, critical and maximum gas saturation.
- `SOWCR`, `SOGCR`: critical oil saturation in water, and in gas.
- `ISGCR`: critical gas saturation of the imbibition curves (used with hysteresis).
- `SCALECRS`: how the critical saturations are scaled.

A cell that is never given a value by one of these arrays is `NaN`, not `0.0`: Eclipse then takes
that cell's end-point from the saturation tables, and `0.0` would be a different (and wrong)
end-point. Operators leave an unset cell unset, since `NaN` plus, times, or limited by a number is
still `NaN`.

**Tracer keywords**:

- `TRACER`: declare tracers (name, carrying fluid, ...).

Saturation and relative permeability columns of these tables must lie in `[0, 1]`
(capillary pressure columns are unrestricted); a value outside that range is
rejected with a `DeckParseError` when the table is parsed.
"""

import numpy as np

from bores.datastructures import GridDimensions
from bores.deck.core import Deck, split_segments, tokenize
from bores.deck.keywords.base import (
    ArrayKeyword,
    Field,
    RecordKeyword,
    RepeatedRecordKeyword,
    Table,
    TableKeyword,
)
from bores.deck.keywords.parsers import fraction, uppercase, yes_no
from bores.deck.operators import Operation
from bores.types import Number

__all__ = [
    "AQUTAB",
    "DENSITY",
    "EHYSTR",
    "GRAVITY",
    "ISGCR",
    "PVCDO",
    "PVCO",
    "PVDG",
    "PVDO",
    "PVTG",
    "PVTO",
    "PVTW",
    "ROCK",
    "ROCKTAB",
    "SCALECRS",
    "SGCR",
    "SGFN",
    "SGL",
    "SGOF",
    "SGU",
    "SOF2",
    "SOF3",
    "SOGCR",
    "SOWCR",
    "SWCON",
    "SWCR",
    "SWCRIT",
    "SWFN",
    "SWL",
    "SWOF",
    "SWU",
    "TRACER",
]

DENSITY = TableKeyword(
    "DENSITY",
    columns=[
        Field("oil", np.float64),
        Field("water", np.float64),
        Field("gas", np.float64),
    ],
)
"""
`DENSITY  OIL_DENSITY  WATER_DENSITY  GAS_DENSITY /`
- stock-tank fluid densities at standard conditions.

One record per PVT region. Units depend on the deck's declared unit
system:

- FIELD: lb/ft³
- METRIC: kg/m³
- LAB: g/cc

Columns:

- `oil`: oil density at standard conditions.
- `water`: water density at standard conditions.
- `gas`: gas density at standard conditions.

`parse` returns `List[List[Dict]]` where each inner list
contains a single-row table (one density record per PVT region).
"""

GRAVITY = RepeatedRecordKeyword[float](
    "GRAVITY",
    fields=[
        Field("api_gravity", np.float64),
        Field("water_specific_gravity", np.float64, required=False, default=1.0),
        Field("gas_specific_gravity", np.float64, required=False, default=0.7773),
    ],
)
"""
`GRAVITY  OIL_API  WATER_SG  GAS_SG /`
- stock-tank fluid gravities at standard conditions, an alternative to `DENSITY`.

One record per PVT region, each ended by its own `/`. `parse` returns one
`Dict` per region, in region order.

Fields:

- `api_gravity`: oil gravity in degrees API.
- `water_specific_gravity`: water specific gravity relative to pure water.
  `1.0` if omitted.
- `gas_specific_gravity`: gas specific gravity relative to air. `0.7773` if
  omitted.
"""

EHYSTR = RecordKeyword[float | int | str](
    "EHYSTR",
    fields=[
        Field("capillary_pressure_curvature", np.float64, required=False, default=0.1),
        Field("relative_permeability_model", int, required=False, default=0),
        Field("killough_wetting_curvature", np.float64, required=False, default=1.0),
        Field("killough_trapped_modifier", np.float64, required=False, default=0.1),
        Field(
            "limiting_curves",
            uppercase,
            required=False,
            default="BOTH",
            options={"BOTH", "KR", "PC"},
        ),
        Field(
            "capillary_pressure_shape",
            uppercase,
            required=False,
            default="RETR",
            options={"RETR", "NEW"},
        ),
        Field(
            "initial_mobility_curve",
            uppercase,
            required=False,
            default="DRAIN",
            options={"DRAIN", "IMBIB"},
        ),
        Field(
            "wetting_phase",
            uppercase,
            required=False,
            default="OIL",
            options={"OIL", "GAS", "WATER"},
        ),
        Field(
            "baker_oil",
            uppercase,
            required=False,
            default="NO",
            options={"YES", "NO"},
        ),
        Field(
            "baker_gas",
            uppercase,
            required=False,
            default="NO",
            options={"YES", "NO"},
        ),
        Field(
            "baker_water",
            uppercase,
            required=False,
            default="NO",
            options={"YES", "NO"},
        ),
        Field("threshold_saturation", np.float64, required=False, default=0.0),
        Field("fix_killough_wetting_phase", int, required=False, default=0),
        Field("scale_capillary_pressure", int, required=False, default=0),
    ],
)
"""
`EHYSTR [pc_curvature [kr_model [...]]] /` - hysteresis model controls.

Fields:

- `capillary_pressure_curvature`: curvature parameter of the capillary
  pressure hysteresis model. `0.1` if omitted.
- `relative_permeability_model`: relative permeability hysteresis model. `-1`
  disables it, `0` uses the Carlson model for the non-wetting phase and the
  drainage curve for the wetting phase, `1` uses the imbibition curve for the
  wetting phase instead. `0` if omitted.
- `killough_wetting_curvature`: curvature parameter of the Killough model for
  the wetting phase. `1.0` if omitted.
- `killough_trapped_modifier`: modification parameter of the Killough model for
  the trapped non-wetting saturation. `0.1` if omitted.
- `limiting_curves`: which curves hysteresis applies to, `BOTH`, `KR` (relative
  permeability only) or `PC` (capillary pressure only). `BOTH` if omitted.
- `capillary_pressure_shape`: shape of the capillary pressure scanning
  curves, `RETR` (retraced) or `NEW`. `RETR` if omitted.
- `initial_mobility_curve`: curve the fluid mobility starts on, `DRAIN` or
  `IMBIB`. `DRAIN` if omitted.
- `wetting_phase`: the wetting phase, `OIL`, `GAS` or `WATER`. `OIL` if omitted.
- `baker_oil`: whether Baker's interpolation is used for oil relative
  permeability, `YES` or `NO`. `NO` if omitted.
- `baker_gas`: the same for gas. `NO` if omitted.
- `baker_water`: the same for water. `NO` if omitted.
- `threshold_saturation`: saturation change below which hysteresis is not
  triggered. `0.0` if omitted.
- `fix_killough_wetting_phase`: flag controlling the Killough wetting phase
  correction. `0` if omitted.
- `scale_capillary_pressure`: flag controlling capillary pressure scaling.
  `0` if omitted.
"""

PVTW = TableKeyword(
    "PVTW",
    columns=[
        Field("reference_pressure", np.float64),
        Field("fvf", np.float64),
        Field("compressibility", np.float64),
        Field("viscosity", np.float64),
        Field("viscosibility", np.float64, required=False, default=0.0),
    ],
)
"""
`PVTW  P_REF  BW  CW  MU_W  CV /` - water PVT table (one record per PVT region).

Columns:

- `reference_pressure`: reference pressure (psi / bar).
- `fvf`: water formation volume factor at `reference_pressure` (rb/stb / rm³/sm³).
- `compressibility`: water compressibility (1/psi / 1/bar).
- `viscosity`: water viscosity at `reference_pressure` (cP).
- `viscosibility`: water viscosibility (1/psi / 1/bar), optional (default 0).
"""


PVDO = TableKeyword(
    "PVDO",
    columns=[
        Field("pressure", np.float64),
        Field("fvf", np.float64),
        Field("viscosity", np.float64),
    ],
)
"""
`PVDO` - dead-oil (no dissolved gas) PVT table.

Each table (one per PVT region) is a sequence of rows terminated
by `/`, with columns:

- `pressure`: oil pressure (psi / bar).
- `fvf`: oil formation volume factor (rb/stb / rm³/sm³).
- `viscosity`: oil viscosity (cP).

Rows must be in ascending pressure order.
"""

PVTO = TableKeyword(
    "PVTO",
    columns=[
        Field("pressure", np.float64),
        Field("fvf", np.float64),
        Field("viscosity", np.float64),
    ],
    primary_key="solution_gor",
)
"""
`PVTO` - live-oil (dissolved-gas, Rs-bracketed) PVT table.

The Eclipse format uses a primary Rs value on a line by itself,
followed by rows of `(pressure, Bo, viscosity)` at that Rs.
Each Rs group is terminated by `/`; the table block ends with `/`.

Columns in each data row (the `solution_gor` field is injected from the
bracketing primary-key line):

- `solution_gor`: solution gas-oil ratio (scf/stb / sm³/sm³).
- `pressure`: oil pressure (psi / bar).
- `fvf`: oil FVF (rb/stb / rm³/sm³).
- `viscosity`: oil viscosity (cP).
"""


PVCDO = TableKeyword(
    "PVCDO",
    columns=[
        Field("reference_pressure", np.float64),
        Field("fvf", np.float64),
        Field("compressibility", np.float64),
        Field("viscosity", np.float64),
        Field("viscosibility", np.float64, required=False, default=0.0),
    ],
)
"""
`PVCDO` - dead oil (no dissolved gas) with constant compressibility and viscosibility.

One record per PVT region (single-row format), evaluated like `PVTW` is for water.

Columns:

- `reference_pressure`: reference pressure.
- `fvf`: FVF at `reference_pressure`.
- `compressibility`: oil compressibility (1/psi / 1/bar).
- `viscosity`: oil viscosity at `reference_pressure`.
- `viscosibility`: viscosibility (optional, default 0).
"""


PVCO = TableKeyword(
    "PVCO",
    columns=[
        Field("bubble_point_pressure", np.float64),
        Field("solution_gor", np.float64),
        Field("fvf", np.float64),
        Field("viscosity", np.float64),
        Field("compressibility", np.float64),
        Field("viscosibility", np.float64, required=False, default=0.0),
    ],
)
"""
`PVCO` - live-oil (dissolved-gas) PVT table with constant compressibility / viscosibility.

Each table (one per PVT region) is a sequence of rows, one per bubble point, terminated
by `/`. Every row is a saturated state. Unlike `PVTO`, the undersaturated behaviour at that
Rs is not tabulated: it follows from the row's own compressibility and viscosibility.

Columns:

- `bubble_point_pressure`: bubble-point pressure (psi / bar).
- `solution_gor`: solution gas-oil ratio at that bubble point (Mscf/stb / sm³/sm³).
- `fvf`: oil FVF at the bubble point (rb/stb / rm³/sm³).
- `viscosity`: oil viscosity at the bubble point (cP).
- `compressibility`: undersaturated oil compressibility `-(1/Bo)·(dBo/dP)` (1/psi / 1/bar).
- `viscosibility`: undersaturated oil viscosibility `(1/μ)·(dμ/dP)` (1/psi / 1/bar),
    optional (default 0).

Rows must be in ascending bubble-point-pressure order. For the single-record dead-oil layout
(reference pressure, Bo, compressibility, viscosity, viscosibility) use `PVCDO`.
"""


PVDG = TableKeyword(
    "PVDG",
    columns=[
        Field("pressure", np.float64),
        Field("fvf", np.float64),
        Field("viscosity", np.float64),
    ],
)
"""
`PVDG` - dry-gas (no vaporized oil) PVT table.

Each table (one per PVT region) contains rows:

- `pressure`: gas pressure (psi / bar).
- `fvf`: gas FVF (rb/Mscf / rm³/sm³).
- `viscosity`: gas viscosity (cP).

Rows must be in ascending pressure order.
"""


PVTG = TableKeyword(
    "PVTG",
    columns=[
        Field("vaporized_ogr", np.float64),
        Field("fvf", np.float64),
        Field("viscosity", np.float64),
    ],
    primary_key="pressure",
)
"""
`PVTG` - wet-gas (vaporized-oil, Rv-bracketed) PVT table.

Mirrors the structure of `PVTO` but with pressure as the outer
(primary) key and Rv as the inner column.

Columns in each data row (the `pressure` field is injected from the
primary-key line):

- `pressure`: gas pressure (psi / bar).
- `vaporized_ogr`: vaporized oil-gas ratio (stb/Mscf / sm³/sm³).
- `fvf`: gas FVF (rb/Mscf / rm³/sm³).
- `viscosity`: gas viscosity (cP).
"""


ROCK = TableKeyword(
    "ROCK",
    columns=[Field("reference_pressure", np.float64), Field("compressibility", np.float64)],
)
"""
`ROCK  P_REF  CR /`
- rock compressibility data (one record per rock region).

Columns:

- `reference_pressure`: reference pressure at which the pore volume equals the
    geometrically calculated value (psi / bar).
- `compressibility`: rock compressibility (1/psi / 1/bar).
"""

ROCKTAB = TableKeyword(
    "ROCKTAB",
    columns=[
        Field("pressure", np.float64),
        Field("pore_volume_multiplier", np.float64),
        Field("transmissibility_multiplier", np.float64, required=False, default=1.0),
    ],
)
"""
`ROCKTAB` - tabulated pore-volume multiplier vs. pressure.

Used when rock compaction cannot be described by a single
compressibility value. Each table (one per rock region) contains rows:

- `pressure`: pressure (psi / bar).
- `pore_volume_multiplier`: pore-volume multiplier at that pressure (dimensionless).
- `transmissibility_multiplier`: transmissibility multiplier (optional, default 1.0).
"""

SWOF = TableKeyword(
    "SWOF",
    columns=[
        Field("sw", fraction),
        Field("krw", fraction),
        Field("krow", fraction),
        Field("pcow", np.float64),
    ],
)
"""
`SWOF` - water saturation function table (first relperm family).

Each table (one per saturation region) contains rows:

- `sw`: water saturation [-].
- `krw`: water relative permeability [-].
- `krow`: oil relative permeability in presence of water [-].
- `pcow`: oil-water capillary pressure Pcow = Po - Pw (psi / bar).

Rows must be in ascending `sw` order.
"""

SGOF = TableKeyword(
    "SGOF",
    columns=[
        Field("sg", fraction),
        Field("krg", fraction),
        Field("krog", fraction),
        Field("pcog", np.float64),
    ],
)
"""
`SGOF` - gas saturation function table (first relperm family).

Each table (one per saturation region) contains rows:

- `sg`: gas saturation [-].
- `krg`: gas relative permeability [-].
- `krog`: oil relative permeability in presence of gas [-].
- `pcog`: gas-oil capillary pressure Pcog = Pg - Po (psi / bar).

Rows must be in ascending `sg` order.
"""

SWFN = TableKeyword(
    "SWFN",
    columns=[
        Field("sw", fraction),
        Field("krw", fraction),
        Field("pcow", np.float64),
    ],
)
"""
`SWFN` - water saturation function table (second relperm family).

Each table contains rows:

- `sw`: water saturation [-].
- `krw`: water relative permeability [-].
- `pcow`: oil-water capillary pressure (psi / bar).

Used alongside `SOF2` or `SOF3` in the second saturation-function
family (as opposed to the `SWOF`/`SGOF` first family).
"""

SGFN = TableKeyword(
    "SGFN",
    columns=[
        Field("sg", fraction),
        Field("krg", fraction),
        Field("pcog", np.float64),
    ],
)
"""
`SGFN` - gas saturation function table (second relperm family).

Each table contains rows:

- `sg`: gas saturation [-].
- `krg`: gas relative permeability [-].
- `pcog`: gas-oil capillary pressure (psi / bar).
"""


SOF2 = TableKeyword(
    "SOF2",
    columns=[
        Field("so", fraction),
        Field("kro", fraction),
    ],
)
"""
`SOF2` - oil relative permeability vs. oil saturation (two-phase).

Each table contains rows:

- `so`: oil saturation [-].
- `kro`: oil relative permeability [-].

Used in two-phase (oil-water or oil-gas) runs with the second
saturation-function family.
"""

SOF3 = TableKeyword(
    "SOF3",
    columns=[
        Field("so", fraction),
        Field("krow", fraction),
        Field("krog", fraction),
    ],
)
"""
`SOF3` - oil relative permeability vs. oil saturation (three-phase).

Each table contains rows:

- `so`: oil saturation [-].
- `krow`: oil relative permeability (water-oil system) [-].
- `krog`: oil relative permeability (gas-oil system) [-].

Used in three-phase runs with the second saturation-function family
alongside `SWFN` and `SGFN`.
"""


class InfluenceTableKeyword(TableKeyword):
    """
    A keyword holding several tables in one block, each terminated by `/`.

    The tables are numbered in the order they appear.
    """

    def parse(
        self,
        deck: Deck,
        dims: GridDimensions | None,
        *,
        operations: list[Operation] | None = None,
        schedule_times: dict[int, float] | None = None,
    ) -> list[Table[Number]] | None:
        records = deck.get_records_for(self.name)
        if not records:
            return None

        tables: list[Table[Number]] = []
        for record in records:
            for segment in split_segments(record.body):
                if tokenize(segment):
                    tables.extend(self._parse_flat_tables(segment))
        return tables or None


AQUTAB = InfluenceTableKeyword(
    "AQUTAB",
    columns=[
        Field("dimensionless_time", np.float64),
        Field("dimensionless_pressure", np.float64),
    ],
)
"""
`AQUTAB  TD  PD /` - aquifer influence tables for `AQUCT` aquifers.

Each table is a list of rows terminated by `/`. Tables are numbered from 2 in the
order they appear, since table 1 is the built-in influence function that an `AQUCT`
aquifer uses by default. An `AQUCT` record picks a table with its
`aquifer_influence_table_number`.

Columns:

- `dimensionless_time`: dimensionless time `tD`. Must increase down the table.
- `dimensionless_pressure`: dimensionless pressure influence function `pD(tD)`.
"""

SCALECRS = RecordKeyword[bool](
    "SCALECRS",
    fields=[Field("enabled", yes_no, required=False, default=False)],
)
"""
`SCALECRS YES|NO /` - select the alternative end-point scaling, which also honours the critical
saturations.

`parse` returns `{"enabled": bool}`; `enabled` is `False` (`NO`) when the item is left out.
"""

TRACER = RepeatedRecordKeyword[str | int](
    "TRACER",
    fields=[
        Field("name", uppercase),
        Field("fluid", uppercase, options={"WAT", "OIL", "GAS"}),
        Field("unit", str, required=False, default=""),
        Field("solution_phase", uppercase, required=False, default=""),
        Field("num_part_table", int, required=False, default=0),
        Field("adsorb_phase", uppercase, required=False, default=""),
    ],
)
"""
`TRACER NAME FLUID [UNIT [SOLUTION_PHASE [NUM_PART_TABLE [ADSORB_PHASE]]]] /` - declare passive
tracers, one record per tracer, closed by a final `/`.

The name is also what the tracer's concentration keywords carry as a suffix, e.g. `TVDPFSEA` for
a tracer called `SEA`.

Fields:

- `name`: tracer name, upper-cased.
- `fluid`: the phase that carries the tracer, one of `WAT`, `OIL` or `GAS`.
- `unit`: unit of the tracer concentration (default empty).
- `solution_phase`: the phase the tracer also dissolves in, if any (default empty).
- `num_part_table`: number of the tracer partition table (default 0, none).
- `adsorb_phase`: the phase whose adsorption the tracer follows, if any (default empty).
"""


def endpoint_array(name: str) -> ArrayKeyword:
    """
    Build the array keyword for a saturation end-point scaling array.

    A cell that nothing assigns is `NaN` rather than `0.0`: it has no end-point of its own, and
    Eclipse takes the end-point from the saturation table there.

    :param name: Keyword name, e.g. `"SWL"`.
    :returns: The array keyword.
    """
    return ArrayKeyword(name, default_value=float("nan"))


SWL = endpoint_array("SWL")
"""`SWL` - connate (irreducible) water saturation, per cell. `NaN` where never set."""

SWCR = endpoint_array("SWCR")
"""`SWCR` - critical water saturation, per cell. `NaN` where never set."""

SWCON = endpoint_array("SWCON")
"""`SWCON` - BORES alias of `SWL`, read only when `SWL` is absent. `NaN` where never set."""

SWCRIT = endpoint_array("SWCRIT")
"""`SWCRIT` - BORES alias of `SWCR`, read only when `SWCR` is absent. `NaN` where never set."""

SWU = endpoint_array("SWU")
"""`SWU` - maximum water saturation, per cell. `NaN` where never set."""

SGL = endpoint_array("SGL")
"""`SGL` - connate gas saturation, per cell. `NaN` where never set."""

SGCR = endpoint_array("SGCR")
"""`SGCR` - critical gas saturation, per cell. `NaN` where never set."""

SGU = endpoint_array("SGU")
"""`SGU` - maximum gas saturation, per cell. `NaN` where never set."""

SOWCR = endpoint_array("SOWCR")
"""`SOWCR` - critical oil-in-water saturation, per cell. `NaN` where never set."""

SOGCR = endpoint_array("SOGCR")
"""`SOGCR` - critical oil-in-gas saturation, per cell. `NaN` where never set."""

ISGCR = endpoint_array("ISGCR")
"""
`ISGCR` - critical gas saturation of the imbibition curves, per cell (used with hysteresis).
`NaN` where never set.
"""
