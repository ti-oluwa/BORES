"""
SOLUTION section keyword implementations.

The SOLUTION section specifies initial reservoir conditions: either as
explicit per-cell arrays (`PRESSURE`, `SWAT`, `SGAS`, `RS`,
`RV`) or via gravity/capillary equilibration (`EQUIL`), or by
restarting from a previously saved state (`RESTART`).

**Explicit initialisation keywords** (return shape `(n_cells,)`
float64 arrays):

- `PRESSURE` - initial reservoir pressure (psi in FIELD, bar in
  METRIC).
- `SWAT` / `SW` - initial water saturation.
- `SOIL` / `SO` - initial oil saturation.
- `SGAS` / `SG` - initial gas saturation.
- `RS` - initial solution gas-oil ratio.
- `RV` - initial vaporized oil-gas ratio.

**Equilibration keyword**:

- `EQUIL` - one record per equilibration region specifying datum
  depth/pressure, contact depths, and capillary pressure at contacts.

**Restart keyword**:

- `RESTART` - resumes from a previously written restart file instead
  of explicit initial conditions.
"""

import typing

import numpy as np

from bores.datastructures import GridDimensions
from bores.deck.core import Deck, DeckParseError, tokenize
from bores.deck.keywords.base import (
    ArrayKeyword,
    Field,
    Keyword,
    RepeatedRecordKeyword,
    TableKeyword,
)
from bores.deck.operators import Operation

__all__ = [
    "EQUIL",
    "PRESSURE",
    "RESTART",
    "RS",
    "RSVD",
    "RTEMP",
    "RTEMPVD",
    "RV",
    "RVVD",
    "SG",
    "SGAS",
    "SO",
    "SOIL",
    "SW",
    "SWAT",
    "TEMPVD",
]


SWAT = SW = ArrayKeyword("SWAT", dtype=np.float64, default_value=0.0)
"""`SWAT` / `SW` - initial water saturation `[0, 1]`."""

SOIL = SO = ArrayKeyword("SOIL", dtype=np.float64, default_value=0.0)
"""`SOIL` / `SO` - initial oil saturation `[0, 1]`."""

SGAS = SG = ArrayKeyword("SGAS", dtype=np.float64, default_value=0.0)
"""`SGAS` / `SG` - initial gas saturation `[0, 1]`."""

PRESSURE = ArrayKeyword("PRESSURE", dtype=np.float64, default_value=0.0)
"""`PRESSURE` - initial reservoir pressure (psi in FIELD, barsa in METRIC)."""

RS = ArrayKeyword("RS", dtype=np.float64, default_value=0.0)
"""`RS` - initial solution gas-oil ratio (scf/stb in FIELD)."""

RV = ArrayKeyword("RV", dtype=np.float64, default_value=0.0)
"""`RV` - initial vaporized oil-gas ratio (stb/scf in FIELD)."""

EQUIL = RepeatedRecordKeyword[float](
    "EQUIL",
    fields=[
        Field("datum_depth", np.float64),
        Field("datum_pressure", np.float64),
        Field("woc_depth", np.float64, required=False, default=0.0),
        Field("pcow_at_woc", np.float64, required=False, default=0.0),
        Field("goc_depth", np.float64, required=False, default=0.0),
        Field("pcgo_at_goc", np.float64, required=False, default=0.0),
        Field("rsvd_table", int, required=False, default=0),
        Field("rvvd_table", int, required=False, default=0),
        Field("accuracy_flag", int, required=False, default=0),
    ],
)
"""
`EQUIL  DATUM_DEPTH  DATUM_PRESSURE  WOC  PCOW_WOC  GOC  PCOG_GOC
RSVD_TABLE  RVVD_TABLE  ACCURACY_FLAG /` (one record per equilibration
region) - gravity/capillary equilibration data, used to derive initial
pressure and saturation distributions instead of supplying explicit
`PRESSURE` / `SWAT` / `SGAS` arrays.
 
One record per `EQLNUM` region; multiple records (one per region) are
separated by `/` within a single `EQUIL` block, and multiple `EQUIL`
blocks in the same deck are concatenated in file order.
 
Fields:
 
- `datum_depth`    - depth of the datum point at which `datum_pressure`
  applies.
- `datum_pressure` - reservoir pressure at `datum_depth`.
- `woc_depth`      - water-oil contact depth.
- `pcow_at_woc`       - oil-water capillary pressure at the WOC (usually
  `0`).
- `goc_depth`      - gas-oil contact depth.
- `pcgo_at_goc`       - gas-oil capillary pressure at the GOC (usually
  `0`).
- `rsvd_table`      - `RSVD`-table number for depth-dependent solution
  GOR below the bubble point (`0` = use the `PVTO` bubble-point Rs
  instead, i.e. assume saturated oil).
- `rvvd_table`      - `RVVD`-table number for depth-dependent
  vaporized-oil ratio above the dew point (`0` = use the `PVTG`
  dew-point Rv instead, i.e. assume saturated gas).
- `accuracy_flag`   - initialization accuracy/option switch (e.g.
  selecting compositional-vs-black-oil equilibration nuances); `0` is
  the simulator default behaviour.
 
Note:
    The exact semantics of `accuracy_flag` (item 9) vary slightly
    between Eclipse 100 and Eclipse 300 / compositional runs; verify
    against your simulator's manual if this run uses anything beyond
    plain black-oil equilibration.
"""

RTEMP = TableKeyword(
    "RTEMP",
    columns=[Field("temperature", np.float64)],
)
"""
`RTEMP  TEMPERATURE /`
- single reservoir temperature applied uniformly to all cells.

One record per run. Used when the reservoir temperature is spatially
constant and does not vary with depth. When both `RTEMP` and `TEMPVD`
are present, `TEMPVD` takes precedence.

Columns:

- `temperature` - reservoir temperature (°F in FIELD, °C in METRIC / LAB).
"""

TEMPVD = TableKeyword(
    "TEMPVD",
    columns=[
        Field("depth", np.float64),
        Field("temperature", np.float64),
    ],
)
"""
`TEMPVD` - temperature versus depth table.

Specifies how reservoir temperature varies with depth, one table per
equilibration region (matched by `EQLNUM`). Cell temperatures are
interpolated linearly from this table at each cell centroid depth.
Values above the shallowest entry or below the deepest entry are
clamped to the endpoint value (no extrapolation).

Each table (one per equilibration region) contains rows:

- `depth`       - true vertical depth (ft in FIELD, m in METRIC).
- `temperature` - reservoir temperature at that depth (°F in FIELD,
  °C in METRIC / LAB).

Rows must be in ascending depth order.
"""

RTEMPVD = TableKeyword(
    "RTEMPVD",
    columns=[
        Field("depth", np.float64),
        Field("temperature", np.float64),
    ],
)
"""
`RTEMPVD` - temperature versus depth table.

Same as `TEMPVD` but used in more recent Eclipse versions

Specifies how reservoir temperature varies with depth, one table per
equilibration region (matched by `EQLNUM`). Cell temperatures are
interpolated linearly from this table at each cell centroid depth.
Values above the shallowest entry or below the deepest entry are
clamped to the endpoint value (no extrapolation).

Each table (one per equilibration region) contains rows:

- `depth`       - true vertical depth (ft in FIELD, m in METRIC).
- `temperature` - reservoir temperature at that depth (°F in FIELD,
  °C in METRIC / LAB).

Rows must be in ascending depth order.
"""


RSVD = TableKeyword(
    "RSVD",
    columns=[
        Field("depth", np.float64),
        Field("solution_gor", np.float64),
    ],
)
"""
`RSVD` - solution GOR versus depth table.

One table per equilibration region (matched by `EQLNUM`). Gives Rs
below the bubble point as a function of depth for saturated-oil
columns; cells are linearly interpolated from this table at their
centroid depth instead of using the PVT table's bubble-point Rs
directly. Values outside the table range are clamped to the endpoint
(no extrapolation), consistent with Eclipse.

Columns:

- `depth` - true vertical depth (ft in FIELD, m in METRIC).
- `solution_gor`    - solution gas-oil ratio at that depth (scf/stb in FIELD).
"""

RVVD = TableKeyword(
    "RVVD",
    columns=[
        Field("depth", np.float64),
        Field("vaporized_ogr", np.float64),
    ],
)
"""
`RVVD` - vaporized oil-gas ratio versus depth table.

Same convention as `RSVD` but for Rv above the dew point in
gas-condensate columns.

Columns:

- `depth` - true vertical depth (ft in FIELD, m in METRIC).
- `vaporized_ogr`    - vaporized oil-gas ratio at that depth (stb/scf in FIELD).
"""


class RestartKeyword(Keyword[dict[str, typing.Any]]):
    """
    The `RESTART` keyword: resume a run from a previously written
    restart file instead of the explicit `PRESSURE` / `SWAT` / `SGAS` /
    `EQUIL` initial-conditions keywords.

    `parse` returns `{"root_name": str, "report_step": int}`, or
    `None` if absent.

    Example deck fragment:

        RESTART
         CASE1 100 /
    """

    def __init__(self) -> None:
        super().__init__("RESTART")

    def parse(
        self,
        deck: Deck,
        dims: GridDimensions | None,
        *,
        operations: list[Operation] | None = None,
        schedule_times: dict[int, float] | None = None,
    ) -> dict[str, typing.Any] | None:
        record = deck.get_first_record_for(self.name)
        if record is None:
            return None

        body = record.body.split("/", 1)[0]
        tokens = tokenize(body)
        if len(tokens) < 2:
            raise DeckParseError(
                f"RESTART: expected 2 tokens (ROOT_NAME REPORT_STEP); "
                f"got {len(tokens)}: {list(tokens)!r}."
            )

        try:
            report_step = int(tokens[1])
        except ValueError as exc:
            raise DeckParseError(
                f"RESTART: report step token {tokens[1]!r} is not an integer."
            ) from exc

        return {"root_name": tokens[0], "report_step": report_step}


RESTART = RestartKeyword()
"""
`RESTART  ROOT_NAME  REPORT_STEP /` - resume from a saved restart file.
 
Fields (returned as a dict rather than via `Field`, since the second
token's "is this required" rule depends only on simple positional
arity, not Eclipse's usual optional/`1*` trailing-default convention):
 
- `root_name`   - root filename (without extension) of the restart
  case to resume from.
- `report_step` - report step number within that case to resume from.
 
`parse` returns `{"root_name": str, "report_step": int}`, or `None`
if absent.
"""
