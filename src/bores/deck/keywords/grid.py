"""
GRID section keyword implementations.

Geometry and coordinate-system keyword implementations.
"""

import typing

import numpy as np

from bores.datastructures import GridDimensions
from bores.deck.core import Deck, DeckParseError, tokenize
from bores.deck.keywords.base import (
    ArrayKeyword,
    Field,
    FlagKeyword,
    Keyword,
    RecordKeyword,
    RepeatedRecordKeyword,
)
from bores.deck.operators import Operation
from bores.types import FloatArray, Integer, ThreeDimensions

__all__ = [
    "ACTNUM",
    "COORD",
    "DIMENS",
    "DX",
    "DXV",
    "DY",
    "DYV",
    "DZ",
    "DZV",
    "FAULTS",
    "GRIDUNIT",
    "MAPAXES",
    "MAPUNITS",
    "MULTFLT",
    "MULTX",
    "MULTX_MINUS",
    "MULTY",
    "MULTY_MINUS",
    "MULTZ",
    "MULTZ_MINUS",
    "NNC",
    "NTG",
    "PERMX",
    "PERMY",
    "PERMZ",
    "PINCH",
    "PINCHOUT",
    "PORO",
    "PORV",
    "SPECGRID",
    "TOPS",
    "ZCORN",
]


SPECGRID = RecordKeyword[str | int](
    "SPECGRID",
    fields=[
        Field("nx", int),
        Field("ny", int),
        Field("nz", int),
        Field("numres", int, required=False, default=1),
        Field("gflag", str, required=False, default="F"),
    ],
)
"""`SPECGRID NX NY NZ [NUMRES [GFLAG]] /` - structured grid dimensions."""

DIMENS = RecordKeyword(
    "DIMENS",
    fields=[
        Field("nx", int),
        Field("ny", int),
        Field("nz", int),
    ],
)
"""`DIMENS NX NY NZ /` - structured grid dimensions (non-corner-point)."""


class CoordKeyword(Keyword[FloatArray[ThreeDimensions]]):
    """
    `COORD` - corner-point pillar array.

    :returns: Shape `(ny+1, nx+1, 6)` float64 array in Eclipse pillar order
        (j outermost, i innermost). Each pillar stores
        `[x_top, y_top, z_top, x_bottom, y_bottom, z_bottom]`.
        Access pattern: `coord[j, i, :]` for cell `(i, j, :)`.
    """

    def __init__(self) -> None:
        super().__init__("COORD")

    def parse(
        self,
        deck: Deck,
        dims: GridDimensions | None,
        *,
        operations: list[Operation] | None = None,
        schedule_times: dict[int, float] | None = None,
    ) -> FloatArray[ThreeDimensions] | None:
        if dims is None:
            raise DeckParseError("`COORD` requires grid dimensions (`SPECGRID`/`DIMENS`).")

        record = deck.get_first_record_for(self.name)
        if record is None:
            return None

        # `record.body` spans up to the next keyword line (some
        # keywords need multiple internal `/` segments); `COORD` is a
        # single `/`-terminated block, so take only the first one.
        body = record.body.split("/", 1)[0]
        tokens = tokenize(body)
        expected = (dims.nx + 1) * (dims.ny + 1) * 6
        if len(tokens) != expected:
            raise DeckParseError(
                f"`COORD` expected {expected} values for a {dims.nx}x{dims.ny} "
                f"grid; got {len(tokens)}."
            )

        # Eclipse writes the coord array (pillars) with coord6 varying fastest
        # then i, and j (slowest), hence the corresponding C-order shape is (ny + 1, nx +1, 6)
        # Access: coord[j, i, :] == [x_top, y_top, z_top, x_bottom, y_bottom, z_bottom]
        return np.array(tokens, dtype=np.float64).reshape(dims.ny + 1, dims.nx + 1, 6)


class ZCornKeyword(Keyword[FloatArray[ThreeDimensions]]):
    """
    `ZCORN` - corner-point depth array.

    :returns: Shape `(nz*2, ny*2, nx*2)` float64 array matching Eclipse
        Fortran write order (i fastest, k slowest).
        Access pattern: `zcorn[2*k, 2*j, 2*i]` for cell `(i, j, k)`.
    """

    def __init__(self) -> None:
        super().__init__("ZCORN")

    def parse(
        self,
        deck: Deck,
        dims: GridDimensions | None,
        *,
        operations: list[Operation] | None = None,
        schedule_times: dict[int, float] | None = None,
    ) -> FloatArray[ThreeDimensions] | None:
        if dims is None:
            raise DeckParseError("`ZCORN` requires grid dimensions (`SPECGRID`/`DIMENS`).")

        record = deck.get_first_record_for(self.name)
        if record is None:
            return None

        body = record.body.split("/", 1)[0]
        tokens = tokenize(body)
        expected = dims.nx * dims.ny * dims.nz * 8
        if len(tokens) != expected:
            raise DeckParseError(
                f"`ZCORN` expected {expected} values for a "
                f"{dims.nx}x{dims.ny}x{dims.nz} grid; got {len(tokens)}."
            )

        # Eclipse writes `ZCORN` in Fortran order with i varying fastest, then j, then k varying slowest,
        # hence in the corresponding shape in C-order with i-fastest is (nz*2, ny*2, nx*2)
        # Access: zcorn[2*k, 2*j, 2*i] for cell (i, j, k)
        return np.array(tokens, dtype=np.float64).reshape(dims.nz * 2, dims.ny * 2, dims.nx * 2)


COORD = CoordKeyword()
"""`COORD` - corner-point pillar array."""

ZCORN = ZCornKeyword()
"""`ZCORN` - corner-point depth array."""

MAPAXES = RecordKeyword[float](
    "MAPAXES",
    fields=[
        Field("y_axis_x", np.float64),
        Field("y_axis_y", np.float64),
        Field("origin_x", np.float64),
        Field("origin_y", np.float64),
        Field("x_axis_x", np.float64),
        Field("x_axis_y", np.float64),
    ],
)
"""
`MAPAXES X_YAXIS Y_YAXIS X_ORIGIN Y_ORIGIN X_XAXIS Y_XAXIS /`
- map coordinate system axes.

Eclipse stores six floats in the order:
`(Y-axis point X, Y-axis point Y, origin X, origin Y, X-axis point X, X-axis point Y)`.

The parsed dict uses those field names verbatim so callers can decide
how to assemble their own `MapAxes` object.
"""

GRIDUNIT = RecordKeyword[str](
    "GRIDUNIT",
    fields=[
        Field("unit", str),
        Field("type", str, required=False, default=""),
    ],
)
"""`GRIDUNIT 'UNIT' ['TYPE'] /` - geometry length unit declaration."""

MAPUNITS = RecordKeyword[str]("MAPUNITS", fields=[Field("unit", str)])
"""
`MAPUNITS 'UNIT' /` - map coordinate unit declaration.
"""

PINCHOUT = FlagKeyword("PINCHOUT")
"""
`PINCHOUT` - enable pinchout transmissibility across inactive cells
sandwiched between two active cells in the same column.
 
Bare/nullary keyword (no data section). `parse` returns `True` if
present in the deck, `False` otherwise.
"""


PINCH = RecordKeyword[str | float](
    "PINCH",
    fields=[
        Field("thickness", np.float64, required=False, default=1e-6),
        Field("gap_mode", str, required=False, default="GAP"),
        Field("pinchout_option", str, required=False, default="TOPBOT"),
        Field("multz_option", str, required=False, default="TOP"),
        Field("composition_option", str, required=False, default="COMPZ"),
    ],
)
"""
`PINCH thickness [gap_mode [pinchout_option [multz_option [comp_option]]]] /`
    - pinchout handling parameters.
"""

TOPS = ArrayKeyword(
    "TOPS",
    dtype=np.float64,
    default_value=0.0,
    column_shape=("nx", "ny"),  # accepts NX*NY column tops; tiled across NZ
)
"""
`TOPS` - depth to the top face of each cell in the first layer
(one value per column `nx * ny`, or all `nx * ny * nz` cells for
non-uniform decks).

In practice Eclipse accepts either `nx * ny` values (applying the same
top to every layer) or `nx * ny * nz` values. The array length
reported by `.ArrayKeyword.parse` matches `n_cells`;
callers should slice `[:nx*ny]` to get the top-layer tops.
"""

DX = ArrayKeyword("DX", dtype=np.float64, default_value=0.0)
"""`DX` - cell size in the x direction (one value per cell)."""

DY = ArrayKeyword("DY", dtype=np.float64, default_value=0.0)
"""`DY` - cell size in the y direction (one value per cell)."""

DZ = ArrayKeyword("DZ", dtype=np.float64, default_value=0.0)
"""`DZ` - cell size in the z direction (one value per cell)."""


class VectorDimsKeyword(Keyword[list[np.float64]]):
    """
    Shared implementation for `DXV` / `DYV` / `DZV`: a flat list of
    per-column or per-layer cell sizes along one structured axis, given
    once rather than once-per-cell.

    Eclipse expects exactly `nx` (`DXV`), `ny` (`DYV`), or `nz` (`DZV`)
    values; every cell in that row/column/layer shares the corresponding
    entry. This is a non-uniform-Cartesian-grid shorthand for `DX` /
    `DY` / `DZ`, used instead of (never alongside) the full per-cell
    array keyword for the same axis.

    :param name: Keyword name (`"DXV"`, `"DYV"`, or `"DZV"`).
    :param axis_extent: Callable extracting the expected length
        (`nx`, `ny`, or `nz`) from `GridDimensions`.
    """

    __slots__ = ("_axis_extent",)

    def __init__(
        self,
        name: str,
        axis_extent: typing.Callable[[GridDimensions], Integer],
    ) -> None:
        super().__init__(name)
        self._axis_extent = axis_extent

    def parse(
        self,
        deck: Deck,
        dims: GridDimensions | None,
        *,
        operations: list[Operation] | None = None,
        schedule_times: dict[int, float] | None = None,
    ) -> list[np.float64] | None:
        if dims is None:
            raise DeckParseError(
                f"Cannot parse {self.name!r} without resolved grid "
                "dimensions (`SPECGRID`/`DIMENS` not found)."
            )

        record = deck.get_first_record_for(self.name)
        if record is None:
            return None

        body = record.body.split("/", 1)[0]
        tokens = tokenize(body)
        expected = self._axis_extent(dims)
        try:
            values = [np.float64(token) for token in tokens]
        except ValueError as exc:
            raise DeckParseError(f"{self.name} contains a non-numeric value: {exc}") from exc

        if len(values) != expected:
            raise DeckParseError(f"{self.name} expected {expected} value(s); got {len(values)}.")
        return values


DXV = VectorDimsKeyword("DXV", axis_extent=lambda dims: dims.nx)
"""
`DXV` - x-direction cell sizes for a non-uniform Cartesian grid, one
value per column (`nx` values total).
 
Used instead of the full per-cell `DX` array when every cell in a given
i-column shares the same x-extent. `parse` returns a length-`nx`
`List[float]`, or `None` if the keyword is absent.
 
:raises DeckParseError: If grid dimensions are unresolved, or the
    token count doesn't match `nx`.
"""

DYV = VectorDimsKeyword("DYV", axis_extent=lambda dims: dims.ny)
"""
`DYV` - y-direction cell sizes for a non-uniform Cartesian grid, one
value per row (`ny` values total). See `DXV`.
"""

DZV = VectorDimsKeyword("DZV", axis_extent=lambda dims: dims.nz)
"""
`DZV` - z-direction cell sizes for a non-uniform Cartesian grid, one
value per layer (`nz` values total). See `DXV`.
"""

ACTNUM = ArrayKeyword("ACTNUM", dtype=np.int32, default_value=1)
"""
`ACTNUM` - active-cell mask.

Values are `1` (active) or `0` (inactive). Stored as int32.
Default is `1` (all cells active) when the keyword is absent.

Note:
    A missing `ACTNUM` keyword means all cells are active in Eclipse,
    so `.ArrayKeyword.parse` returns `None` (keyword
    absent) rather than an all-ones array. Callers should treat
    `None` as "all active".
"""

MULTX = ArrayKeyword("MULTX", is_multiplier=True)
"""
`MULTX` - transmissibility multiplier for the positive-x face
of each cell.
"""

MULTY = ArrayKeyword("MULTY", is_multiplier=True)
"""
`MULTY` - transmissibility multiplier for the positive-y face
of each cell.
"""

MULTZ = ArrayKeyword("MULTZ", is_multiplier=True)
"""
`MULTZ` - transmissibility multiplier for the positive-z face
of each cell.
"""

MULTX_MINUS = ArrayKeyword("MULTX-", is_multiplier=True)
"""
`MULTX-` - transmissibility multiplier for the negative-x face
of each cell.
"""

MULTY_MINUS = ArrayKeyword("MULTY-", is_multiplier=True)
"""
`MULTY-` - transmissibility multiplier for the negative-y face
of each cell.
"""

MULTZ_MINUS = ArrayKeyword("MULTZ-", is_multiplier=True)
"""
`MULTZ-` - transmissibility multiplier for the negative-z face
of each cell.
"""

PORO = ArrayKeyword("PORO", dtype=np.float64, default_value=0.0)
"""
`PORO` - porosity fraction `[0, 1]`.

A missing `PORO` keyword returns `None`; the simulator should treat
that as zero porosity (dead rock).
"""

PERMX = ArrayKeyword("PERMX", dtype=np.float64, default_value=0.0)
"""`PERMX` - permeability in the x direction (mD)."""

PERMY = ArrayKeyword("PERMY", dtype=np.float64, default_value=0.0)
"""`PERMY` - permeability in the y direction (mD)."""

PERMZ = ArrayKeyword("PERMZ", dtype=np.float64, default_value=0.0)
"""`PERMZ` - permeability in the z direction (mD)."""


class MultFLTKeyword(RepeatedRecordKeyword[np.float64 | str]):
    """
    `MULTFLT 'NAME' MULTIPLIER / ... /` - per-fault transmissibility multiplier.

    Eclipse semantics: when a fault name appears in multiple records
    across one or more `MULTFLT` blocks, the *last* value wins.
    `parse` enforces this automatically.
    """

    def __init__(self) -> None:
        super().__init__(
            "MULTFLT",
            fields=[
                Field("name", str),
                Field("multiplier", np.float64),
            ],
        )

    def parse(
        self,
        deck: Deck,
        dims: GridDimensions | None,
        *,
        operations: list[Operation] | None = None,
        schedule_times: dict[int, float] | None = None,
    ) -> list[dict[str, typing.Any]] | None:
        records = super().parse(deck, dims)
        if records is None:
            return None

        # Last value for each fault name wins.
        by_name: dict[typing.Hashable, dict[str, typing.Any]] = {}
        for record in records:
            if record is None:
                continue
            by_name[record["name"]] = record
        return list(by_name.values())


FAULTS = RepeatedRecordKeyword[str | int](
    "FAULTS",
    fields=[
        Field("name", str),
        Field("i1", int),
        Field("i2", int),
        Field("j1", int),
        Field("j2", int),
        Field("k1", int),
        Field("k2", int),
        Field(
            "face",
            lambda v: str(v).removesuffix("+").upper(),
            options={"X", "X-", "Y", "Y-", "Z", "Z-"},
        ),
    ],
)
"""
`FAULTS 'NAME' I1 I2 J1 J2 K1 K2 FACE / ... /` - named fault planes.

Multiple `FAULTS` keyword blocks in the same deck are concatenated in
file order.  Indices `I1`/`I2`/`J1`/`J2`/`K1`/`K2` are
*1-based* Eclipse IJK indices, passed through unchanged to the caller.

:raises DeckParseError: If `FACE` is not one of `X`, `X-`, `Y`,
    `Y-`, `Z`, `Z-`.
"""

MULTFLT = MultFLTKeyword()
"""
`MULTFLT 'NAME' MULTIPLIER / ... /` - per-fault transmissibility multiplier.

Eclipse semantics: when a fault name appears in multiple records
across one or more `MULTFLT` blocks, the *last* value wins.
`parse` enforces this automatically.
"""

NNC = RepeatedRecordKeyword[float](
    "NNC",
    fields=[
        Field("i1", int),
        Field("j1", int),
        Field("k1", int),
        Field("i2", int),
        Field("j2", int),
        Field("k2", int),
        Field("transmissibility", np.float64),
    ],
)
"""
`NNC I1 J1 K1 I2 J2 K2 T / ... /` - explicit non-neighbour connections.

Indices `I1/J1/K1` and `I2/J2/K2` are *1-based* Eclipse IJK
structured cell indices. `T` is the transmissibility value in the
grid's declared unit system.

Multiple `NNC` keyword blocks in the same deck are concatenated.
"""

PORV = ArrayKeyword("PORV", dtype=np.float64, default_value=0.0)
"""
`PORV` - pore volume per cell (bbl in FIELD, m³ in METRIC).

When present, the simulator should use this directly rather than
computing pore volume from geometry and porosity.
"""

NTG = ArrayKeyword("NTG", dtype=np.float64, default_value=1.0)
"""
`NTG` - net-to-gross ratio `[0, 1]`.

Effective pore volume: `PORV = NTG * PORO * bulk_volume`.
Default is `1.0` (100 % net) when absent.
"""
