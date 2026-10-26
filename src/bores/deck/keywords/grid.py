"""
GRID section keyword implementations.

Geometry and coordinate-system keyword implementations.
"""

import typing

import numpy as np

from bores.datastructures import GridDimensions
from bores.deck.core import Deck, DeckParseError, first_segment, parse_numeric_runs, tokenize
from bores.deck.keywords.base import (
    ArrayKeyword,
    Field,
    FlagKeyword,
    Keyword,
    RecordKeyword,
    RepeatedRecordKeyword,
    fill_from_runs,
)
from bores.deck.keywords.parsers import fault_face, uppercase
from bores.deck.operators import Operation
from bores.types import FloatArray, Integer, OneDimension, ThreeDimensions

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
    "GDORIENT",
    "GRIDFILE",
    "GRIDUNIT",
    "INIT",
    "MAPAXES",
    "MAPUNITS",
    "MINPV",
    "MULTFLT",
    "MULTX",
    "MULTX_MINUS",
    "MULTY",
    "MULTY_MINUS",
    "MULTZ",
    "MULTZ_MINUS",
    "NEWTRAN",
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
        Field("gflag", uppercase, required=False, default="F", options={"F", "T"}),
    ],
)
"""
`SPECGRID NX NY NZ [NUMRES [GFLAG]] /` - structured grid dimensions.

Fields:

- `nx`: number of cells in the x (i) direction.
- `ny`: number of cells in the y (j) direction.
- `nz`: number of cells in the z (k) direction.
- `numres`: number of reservoirs in the grid. `1` if omitted.
- `gflag`: coordinate type of the grid, `F` for Cartesian or `T` for
    radial (cylindrical). `F` if omitted.
"""

DIMENS = RecordKeyword(
    "DIMENS",
    fields=[
        Field("nx", int),
        Field("ny", int),
        Field("nz", int),
    ],
)
"""
`DIMENS NX NY NZ /` - structured grid dimensions (non-corner-point).

Fields:

- `nx`: number of cells in the x (i) direction.
- `ny`: number of cells in the y (j) direction.
- `nz`: number of cells in the z (k) direction.
"""


def read_exact_values(
    body: str, *, name: str, expected: int, describe: str
) -> FloatArray[OneDimension]:
    """
    Read the numbers of a record body (`N*value` groups included) into one flat array of exactly
    `expected` values.

    The numbers are parsed in bulk, with no list of token strings in between, so a large
    `COORD` or `ZCORN` costs little more than the array itself.

    :param body: Record body up to its terminating `/`.
    :param name: Keyword name, for error messages.
    :param expected: How many values the record must hold.
    :param describe: What `expected` follows from, for the count mismatch message.
    :returns: A writable `float64` array of `expected` values.
    :raises DeckParseError: If a value is not a number or defaulted (`N*`), or the count is wrong.
    """
    runs = parse_numeric_runs(body, name=name)
    if any(run.values is None for run in runs):
        raise DeckParseError(f"{name} contains a defaulted ('N*') entry, which it cannot take.")

    total = sum(run.entry_count for run in runs)
    if total != expected:
        raise DeckParseError(f"{name} expected {expected} values for {describe}; got {total}.")

    if len(runs) == 1 and isinstance(runs[0].values, np.ndarray):
        return runs[0].values
    values = np.empty(expected, dtype=np.float64)
    fill_from_runs(values, runs)
    return values


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

        record = deck.get_single_record_for(self.name)
        if record is None:
            return None

        # `record.body` spans up to the next keyword line (some
        # keywords need multiple internal `/` segments); `COORD` is a
        # single `/`-terminated block, so take only the first one.
        values = read_exact_values(
            first_segment(record.body),
            name="`COORD`",
            expected=(dims.nx + 1) * (dims.ny + 1) * 6,
            describe=f"a {dims.nx}x{dims.ny} grid",
        )

        # Eclipse writes the coord array (pillars) with coord6 varying fastest
        # then i, and j (slowest), hence the corresponding C-order shape is (ny + 1, nx +1, 6)
        # Access: coord[j, i, :] == [x_top, y_top, z_top, x_bottom, y_bottom, z_bottom]
        return values.reshape(dims.ny + 1, dims.nx + 1, 6)


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

        record = deck.get_single_record_for(self.name)
        if record is None:
            return None

        values = read_exact_values(
            first_segment(record.body),
            name="`ZCORN`",
            expected=dims.nx * dims.ny * dims.nz * 8,
            describe=f"a {dims.nx}x{dims.ny}x{dims.nz} grid",
        )

        # Eclipse writes `ZCORN` in Fortran order with i varying fastest, then j, then k varying slowest,
        # hence in the corresponding shape in C-order with i-fastest is (nz*2, ny*2, nx*2)
        # Access: zcorn[2*k, 2*j, 2*i] for cell (i, j, k)
        return values.reshape(dims.nz * 2, dims.ny * 2, dims.nx * 2)


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

The parsed dict is returned as is, so callers can decide how to assemble
their own `MapAxes` object.

Fields:

- `y_axis_x`: map x coordinate of a point on the grid's y axis.
- `y_axis_y`: map y coordinate of a point on the grid's y axis.
- `origin_x`: map x coordinate of the grid origin.
- `origin_y`: map y coordinate of the grid origin.
- `x_axis_x`: map x coordinate of a point on the grid's x axis.
- `x_axis_y`: map y coordinate of a point on the grid's x axis.
"""

GRIDUNIT = RecordKeyword[str](
    "GRIDUNIT",
    fields=[
        Field("unit", uppercase, options={"METRES", "FEET", "CM"}),
        Field("type", uppercase, required=False, default="", options={"MAP", ""}),
    ],
)
"""
`GRIDUNIT 'UNIT' ['TYPE'] /` - geometry length unit declaration.

Fields:

- `unit`: length unit of the grid geometry, one of `METRES`, `FEET` or `CM`.
- `type`: `MAP` if the grid coordinates are map coordinates, otherwise empty
    (the grid is in local coordinates). Empty if omitted.
"""

MAPUNITS = RecordKeyword[str](
    "MAPUNITS", fields=[Field("unit", uppercase, options={"METRES", "FEET", "CM"})]
)
"""
`MAPUNITS 'UNIT' /` - map coordinate unit declaration.

Fields:

- `unit`: length unit of the `MAPAXES` map coordinates, one of `METRES`,
    `FEET` or `CM`.
"""

INIT = FlagKeyword("INIT")
"""
`INIT` - request the initial grid properties file (static cell properties such as
porosity, permeability, and transmissibility multipliers) as simulation output.

Bare/nullary keyword (no data section). `parse` returns `True` if present in the
deck, `False` otherwise. It does not change the simulation itself.
"""

GDORIENT = RecordKeyword[str](
    "GDORIENT",
    fields=[
        Field(
            "i_direction",
            uppercase,
            required=False,
            default="INC",
            options={"INC", "DEC"},
        ),
        Field(
            "j_direction",
            uppercase,
            required=False,
            default="INC",
            options={"INC", "DEC"},
        ),
        Field(
            "k_direction",
            uppercase,
            required=False,
            default="INC",
            options={"INC", "DEC"},
        ),
        Field(
            "z_direction",
            uppercase,
            required=False,
            default="DOWN",
            options={"DOWN", "UP"},
        ),
        Field(
            "handedness",
            uppercase,
            required=False,
            default="RIGHT",
            options={"RIGHT", "LEFT"},
        ),
    ],
)
"""
`GDORIENT I J K Z HAND /` - orientation of the grid axes.

Says which way the cell indices run and how the depth axis points. Every item is optional and
takes the default shown when left out.

Fields:

- `i_direction`, `j_direction`, `k_direction`: whether the `I`, `J` and `K` index increases
  (`"INC"`, default) or decreases (`"DEC"`) along its coordinate axis.
- `z_direction`: whether depth, the `Z` axis, points `"DOWN"` (default) or `"UP"`.
- `handedness`: whether the axes form a `"RIGHT"` (default) or a `"LEFT"` handed system.

`GDORIENT INC INC INC DOWN LEFT /` is therefore a left-handed grid whose indices all increase
along their axes, with depth increasing downwards.
"""

GRIDFILE = RecordKeyword[int](
    "GRIDFILE",
    fields=[
        Field("grid", int, required=False, default=0),
        Field("egrid", int, required=False, default=1),
    ],
)
"""
`GRIDFILE GRID EGRID /` - which grid geometry files the run writes.

Fields:

- `grid`: `1` writes the `.GRID` file, `0` (default) does not.
- `egrid`: `1` (default) writes the `.EGRID` file, `0` does not.

`GRIDFILE 0 1 /` asks for an `.EGRID` file and no `.GRID` file. It only affects output, not the
simulation.
"""

MINPV = RecordKeyword[float](
    "MINPV",
    fields=[Field("minpv", np.float64, required=False, default=1e-6)],
)
"""
`MINPV VALUE /` - minimum pore volume for a cell to be active.

A cell whose pore volume is below `minpv` is made inactive. The value is a reservoir volume in
the deck's unit system, and defaults to `1e-6`.
"""

NEWTRAN = FlagKeyword("NEWTRAN")
"""
`NEWTRAN` - calculate transmissibilities with the newer method rather than the older one
(`OLDTRAN`).

Bare/nullary keyword (no data section). `parse` returns `True` if present in the deck, `False`
otherwise.
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
        Field("gap_mode", uppercase, required=False, default="GAP", options={"GAP", "NOGAP"}),
        Field("max_empty_gap", np.float64, required=False, default=None),
        Field(
            "pinchout_option",
            uppercase,
            required=False,
            default="TOPBOT",
            options={"TOPBOT", "ALL"},
        ),
        Field("multz_option", uppercase, required=False, default="TOP", options={"TOP", "ALL"}),
    ],
)
"""
`PINCH thickness [gap_mode [max_empty_gap [pinchout_option [multz_option]]]] /`
- pinchout handling parameters.

Fields:

- `thickness`: cells thinner than this are treated as pinched out.
- `gap_mode`: `GAP` lets a pinched-out cell sit between two active cells and
    still connect them across the gap, `NOGAP` does not. `GAP` if omitted.
- `max_empty_gap`: largest gap of pinched-out cells that may still be connected
    across. `None` (no limit) if omitted.
- `pinchout_option`: `TOPBOT` connects only across a pinched-out cell at the top
    or bottom of a column of active cells, `ALL` connects across any. `TOPBOT`
    if omitted.
- `multz_option`: `TOP` takes the `MULTZ` multiplier of the upper cell across a
    pinchout connection, `ALL` applies the `MULTZ` of every cell crossed. `TOP`
    if omitted.
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
reported by `ArrayKeyword.parse` matches `n_cells`;
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

        record = deck.get_single_record_for(self.name)
        if record is None:
            return None

        body = first_segment(record.body)
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
    so `ArrayKeyword.parse` returns `None` (keyword
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

    Fields:

    - `name`: name of the fault, as declared by `FAULTS`.
    - `multiplier`: transmissibility multiplier applied to the fault.
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
        Field("face", fault_face, options={"X", "X-", "Y", "Y-", "Z", "Z-"}),
    ],
)
"""
`FAULTS 'NAME' I1 I2 J1 J2 K1 K2 FACE / ... /` - named fault planes.

Multiple `FAULTS` keyword blocks in the same deck are concatenated in
file order. Indices `I1`/`I2`/`J1`/`J2`/`K1`/`K2` are
*1-based* Eclipse IJK indices, passed through unchanged to the caller.

Fields:

- `name`: name of the fault.
- `i1` / `i2`: first and last cell of the fault in the i direction.
- `j1` / `j2`: first and last cell of the fault in the j direction.
- `k1` / `k2`: first and last cell of the fault in the k direction.
- `face`: the face of the cells the fault lies on, one of `X`, `X-`, `Y`, `Y-`,
    `Z`, `Z-`. The Eclipse spellings `I`, `J`, `K` (and a trailing `+`) are
    accepted and returned as the matching `X`, `Y`, `Z` form.

:raises DeckParseError: If `FACE` is not a valid face.
"""

MULTFLT = MultFLTKeyword()
"""
`MULTFLT 'NAME' MULTIPLIER / ... /` - per-fault transmissibility multiplier.

Eclipse semantics: when a fault name appears in multiple records
across one or more `MULTFLT` blocks, the *last* value wins.
`parse` enforces this automatically.

Fields:

- `name`: name of the fault, as declared by `FAULTS`.
- `multiplier`: transmissibility multiplier applied to the fault.
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

Multiple `NNC` keyword blocks in the same deck are concatenated.

Fields:

- `i1` / `j1` / `k1`: 1-based IJK index of the first cell of the connection.
- `i2` / `j2` / `k2`: 1-based IJK index of the second cell of the connection.
- `transmissibility`: transmissibility of the connection, in the grid's
    declared unit system.
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
