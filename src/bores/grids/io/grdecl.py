"""
GRDECL text-format reader and writer.

GRDECL is the ASCII keyword-based format used by Eclipse, ResInsight, and
most other reservoir simulators to describe/declare corner-point pillar grids.

**Supported keywords (read)**:

`SPECGRID`, `DIMENS`, `COORD`, `ZCORN`, `ACTNUM`, `GRIDUNIT`,
`TOPS`, `DX`, `DY`, `DZ`, `DXV`, `DYV`, `DZV`,
`MAPAXES`, `MAPUNITS`, `PINCH`, `PINCHOUT`, `NNC`, `FAULTS`, `MULTFLT`,
`MULTX`, `MULTY`, `MULTZ`, `MULTX-`, `MULTY-`, `MULTZ-`.

**Supported keywords (write)**:

`SPECGRID`, `GRIDUNIT`, `MAPAXES`, `MAPUNITS`,
`COORD`, `ZCORN`, `ACTNUM` (corner-point grids);
`SPECGRID`, `GRIDUNIT`, `MAPAXES`, `MAPUNITS`,
`TOPS`, `DXV`, `DYV`, `DZV`, `ACTNUM` (Cartesian grids).

Both grid types also emit (when present on the grid):
`MULTX`, `MULTY`, `MULTZ`, `MULTX-`, `MULTY-`, `MULTZ-`,
`FAULTS`, `MULTFLT`, `NNC`, `PINCH`.
"""

import typing
import warnings
from pathlib import Path

import numba
import numpy as np
import numpy.typing as npt

from bores.datastructures import MapAxes
from bores.deck.core import DeckParseError
from bores.deck.file import DeckFile
from bores.errors import GridExportError, GridImportError
from bores.grids.base import ConnectionType, Grid
from bores.grids.factories.cartesian import make_cartesian_grid
from bores.grids.factories.corner_point import (
    ActNumArray,
    FaultRecord,
    get_map_axes_xy_inverse,
    make_corner_point_grid,
    rederive_corner_point_arrays,
)
from bores.types import (
    IntArray,
    Integer,
    Number,
    NumberArray,
    OneDimension,
    PathOrStr,
    TextOrPath,
    TwoDimensions,
    UnitSystem,
)

__all__ = ["dump_grdecl", "load_grdecl"]


UNITS_MAP: dict[str, UnitSystem] = {
    "METRES": UnitSystem.METRIC,
    "METER": UnitSystem.METRIC,
    "M": UnitSystem.METRIC,
    "FEET": UnitSystem.FIELD,
    "FT": UnitSystem.FIELD,
    "CM": UnitSystem.LAB,
    "CENTIMETRES": UnitSystem.LAB,
    "CENTIMETERS": UnitSystem.LAB,
}


US_TO_GRIDUNIT: dict[UnitSystem, str] = {
    UnitSystem.FIELD: "FEET",
    UnitSystem.METRIC: "METRES",
    UnitSystem.LAB: "CM",
    UnitSystem.SI: "METRES",
}

US_TO_MAPUNITS: dict[UnitSystem, str] = {
    UnitSystem.FIELD: "FEET",
    UnitSystem.METRIC: "METRES",
    UnitSystem.LAB: "CM",
    UnitSystem.SI: "METRES",
}


def detect_unit_system(deck_file: DeckFile) -> UnitSystem:
    """
    Determine the grid geometry unit system from a parsed `DeckFile`.

    :param deck_file: Already-constructed `DeckFile`.
    :returns: The declared `bores.typing.UnitSystem`.
    """
    gridunit = deck_file.get("GRIDUNIT")
    if gridunit is not None:
        unit_str = str(gridunit.get("unit", "")).strip().upper()
        unit_system = UNITS_MAP.get(unit_str)
        if unit_system is not None:
            return unit_system

    return deck_file.unit_system


def build_map_axes(deck_file: DeckFile) -> MapAxes | None:
    """
    Construct a `MapAxes` from parsed `MAPAXES` / `MAPUNITS` keyword dicts.

    Eclipse `MAPAXES` field order:
    `(Y-axis X, Y-axis Y, origin X, origin Y, X-axis X, X-axis Y)`.

    :param deck_file: Parsed deck.
    :returns: `MapAxes` or `None` if `MAPAXES` is absent.
    """
    mapaxes = deck_file.get("MAPAXES")
    if mapaxes is None:
        return None

    mapunits = deck_file.get("MAPUNITS")
    map_unit_str = str(mapunits.get("unit", "")).strip().upper() if mapunits else ""
    map_unit = UNITS_MAP.get(map_unit_str, UnitSystem.FIELD)
    return MapAxes(  # type: ignore[arg-type]
        origin=np.array([mapaxes["origin_x"], mapaxes["origin_y"]], dtype=np.float64),  # type: ignore[arg-type]
        map_x_axis_point=np.array(  # type: ignore[arg-type]
            [mapaxes["x_axis_x"], mapaxes["x_axis_y"]], dtype=np.float64
        ),
        map_y_axis_point=np.array(  # type: ignore[arg-type]
            [mapaxes["y_axis_x"], mapaxes["y_axis_y"]], dtype=np.float64
        ),
        unit_system=map_unit,
    )


def build_nnc_arrays(
    deck_file: DeckFile,
    nx: Integer,
    ny: Integer,
    nz: Integer,
) -> tuple[
    IntArray[TwoDimensions] | None,
    NumberArray[OneDimension] | None,
]:
    """
    Convert parsed `NNC` keyword records to flat cell-index arrays.

    :param deck_file: Parsed deck (`NNC` keyword already registered).
    :param nx: Grid extent in x.
    :param ny: Grid extent in y.
    :param nz: Grid extent in z.
    :returns: `(pairs, transmissibilities)` - shape `(n_nnc, 2)` int32
        and shape `(n_nnc,)` float64 arrays, or `(None, None)` if the
        keyword is absent.
    :raises GridImportError: If a record has out-of-bounds cell indices.
    """
    nnc_records = deck_file.get("NNC")
    if not nnc_records:
        return None, None

    pairs: list[tuple[int, int]] = []
    transmissibilities: list[float] = []

    for idx, record in enumerate(nnc_records):
        i1, j1, k1 = record["i1"], record["j1"], record["k1"]
        i2, j2, k2 = record["i2"], record["j2"], record["k2"]
        transmissibility = record["transmissibility"]

        for label, i, j, k in [("first", i1, j1, k1), ("second", i2, j2, k2)]:
            if not (1 <= i <= nx and 1 <= j <= ny and 1 <= k <= nz):
                raise GridImportError(
                    f"NNC record {idx} {label} cell ({i},{j},{k}) is outside "
                    f"grid bounds ({nx}x{ny}x{nz})."
                )

        # Eclipse flat index: i fastest -> (i-1) + (j-1)*nx + (k-1)*nx*ny
        c1 = (i1 - 1) + (j1 - 1) * nx + (k1 - 1) * nx * ny
        c2 = (i2 - 1) + (j2 - 1) * nx + (k2 - 1) * nx * ny
        pairs.append((c1, c2))
        transmissibilities.append(transmissibility)

    if not pairs:
        return None, None

    return (
        np.asarray(pairs, dtype=np.int32).reshape(-1, 2),
        np.asarray(transmissibilities, dtype=np.float64),  # type: ignore[arg-type]
    )


def build_fault_records(deck_file: DeckFile) -> list[FaultRecord]:
    """
    Convert parsed `FAULTS` keyword records to `FaultRecord` objects.

    :param deck_file: Parsed deck.
    :returns: List of `FaultRecord` in file order.
    """
    faults = deck_file.get("FAULTS")
    if not faults:
        return []

    return [
        FaultRecord(
            name=record["name"],
            i1=record["i1"],
            i2=record["i2"],
            j1=record["j1"],
            j2=record["j2"],
            k1=record["k1"],
            k2=record["k2"],
            face_direction=record["face"],
        )
        for record in faults
    ]


def build_multflt(deck_file: DeckFile) -> dict[str, Number] | None:
    """
    Convert parsed `MULTFLT` records to a `{name: multiplier}` dict.

    :param deck_file: Parsed deck.
    :returns: Dict or `None` if `MULTFLT` is absent.
    """
    multflt_records = deck_file.get("MULTFLT")
    if not multflt_records:
        return None
    return {record["name"]: record["multiplier"] for record in multflt_records}


def resolve_vector_spacing(
    deck_file: DeckFile,
    vector_key: str,
    per_cell_key: str,
    count: Integer,
    axis: str,
    nz: Integer,
    ny: Integer,
    nx: Integer,
) -> npt.NDArray[np.float64] | None:
    """
    Try to obtain a 1-D spacing vector for one axis.

    Resolution order:

    1. `DXV` / `DYV` / `DZV` - single-valued vectors of length
       `nx`, `ny`, `nz` respectively, already 1-D in the deck.
    2. `DX` / `DY` / `DZ` - per-cell arrays of length
       `nx*ny*nz`.  Reshaped to `(nz, ny, nx)` and the first
       row / column / layer is extracted.
    3. Returns `None` if neither keyword is present.

    :param deck_file: Parsed deck.
    :param vector_key: Vector keyword name (`"DXV"`, `"DYV"`, `"DZV"`).
    :param per_cell_key: Per-cell keyword name (`"DX"`, `"DY"`, `"DZ"`).
    :param count: Expected length of the vector (`nx`, `ny`, or `nz`).
    :param axis: Axis label for error messages (`"x"`, `"y"`, `"z"`).
    :param nz: Grid depth count.
    :param ny: Grid lateral count (y).
    :param nx: Grid lateral count (x).
    :returns: 1-D float64 spacing array or `None`.
    :raises GridImportError: If the vector keyword has the wrong length.
    """
    # Prefer the vector form (DXV / DYV / DZV).
    vec = deck_file.get(vector_key)
    if vec is not None:
        array = np.asarray(vec, dtype=np.float64).ravel()
        if len(array) != count:
            raise GridImportError(
                f"{vector_key} has {len(array)} values but expected {count} "
                f"(grid has n{axis}={count})."
            )
        return array

    # Fall back to the per-cell array.
    per_cell = deck_file.get(per_cell_key)
    if per_cell is None:
        return None

    flat = np.asarray(per_cell, dtype=np.float64, copy=False).ravel()
    reshaped = flat.reshape(nz, ny, nx)
    if axis == "x":
        return reshaped[0, 0, :]  # varies in x; take first y-row, first z-layer
    if axis == "y":
        return reshaped[0, :, 0]  # varies in y
    return reshaped[:, 0, 0]  # varies in z


def load_grdecl(
    source: TextOrPath | DeckFile,
    *,
    encoding: str = "ascii",
    unit_system: UnitSystem | None = None,
    metadata: typing.Mapping[str, typing.Any] | None = None,
) -> Grid:
    """
    Load a GRDECL corner-point (or Cartesian) grid from a `DeckFile`, file path, raw
    string, or bytes.

    Automatically detects whether `source` is a filesystem path or raw
    GRDECL text / bytes, or a `DeckFile`. Recursively resolves `INCLUDE` directives when
    loading from a file path.

    Parsed keyword coverage:

    * Grid dimensions:   `SPECGRID`, `DIMENS`
    * Geometry:          `COORD`, `ZCORN`, `TOPS`, `DX`, `DY`,
                         `DZ`, `DXV`, `DYV`, `DZV`
    * Activity:          `ACTNUM`
    * Units:             `GRIDUNIT`, `FIELD`, `METRIC`, `LAB`, `SI`
    * Map CRS:           `MAPAXES`, `MAPUNITS`
    * Pinchouts:         `PINCH`, `PINCHOUT`
    * Connections:       `NNC` (with transmissibilities)
    * Faults:            `FAULTS`, `MULTFLT`
    * Multipliers:       `MULTX`, `MULTY`, `MULTZ` and `-` variants

    :param source: One of:

        * `pathlib.Path` - path to a `.grdecl` file.
        * `str` - filesystem path *or* raw GRDECL text content.
        * `bytes` - raw GRDECL content encoded as `encoding`.

    :param encoding: Text encoding used when decoding `bytes` / file
        input (default `"ascii"`).
    :param unit_system: If provided, the returned
        `bores.grids.base.Grid` is converted to this unit system
        after loading.
    :param metadata: Optional extra key/value pairs merged into the
        `Grid.metadata` dict.
    :returns: A fully initialised `bores.grids.base.Grid`.
    :raises GridImportError: If required keywords are missing or malformed.
    """
    if isinstance(source, DeckFile):
        deck_file = source
    else:
        try:
            deck_file = DeckFile(source, encoding=encoding, unit_system=unit_system)
        except DeckParseError as exc:
            raise GridImportError(f"Failed to parse GRDECL deck: {exc}") from exc

    grid = assemble_grid(deck_file, metadata=metadata)
    return grid.convert(unit_system) if unit_system is not None else grid


def dump_grdecl(
    grid: Grid,
    destination: PathOrStr | None = None,
    *,
    actnum: ActNumArray | None = None,
    encoding: str = "ascii",
) -> str | None:
    """
    Serialise a `Grid` to GRDECL text format.

    Emits all keywords that can be reconstructed from the grid object.
    The full set that may appear in the output is:

    `SPECGRID`, `GRIDUNIT`, `MAPAXES`, `MAPUNITS`,
    `COORD` / `ZCORN` (corner-point) or `TOPS` / `DXV` / `DYV` /
    `DZV` (Cartesian), `ACTNUM`, `MULTX`, `MULTY`, `MULTZ`,
    `MULTX-`, `MULTY-`, `MULTZ-`, `FAULTS`, `MULTFLT`,
    `NNC`, `PINCH`.

    :param grid: The grid to serialise.
    :param destination: One of:

        * `pathlib.Path` or `str` path - write to file, return `None`.
        * `None` - return the GRDECL text as a `str`.

    :param actnum: Optional shape `(n_cells,)` integer array of
        active-cell flags (1 = active, 0 = inactive).  If `None`, the
        value stored in `grid.metadata["actnum"]` is used; if that is
        also absent, all cells are written as active.
    :param encoding: Encoding used when writing to a file (default
        `"ascii"`).
    :returns: GRDECL text as a `str` when `destination` is `None`;
        `None` otherwise.
    :raises GridExportError: If the grid cannot be serialised.
    """
    try:
        text = build_grdecl_text(grid, actnum=actnum)
    except Exception as exc:
        raise GridExportError(f"Failed to serialise grid to GRDECL: {exc}") from exc

    if destination is None:
        return text

    path = Path(destination)
    path.write_text(text, encoding=encoding)
    return None


def assemble_grid(
    deck_file: DeckFile,
    metadata: typing.Mapping[str, typing.Any] | None = None,
) -> Grid:
    """
    Assemble a `bores.grids.base.Grid` from a parsed `bores.deck.file`.

    :param deck_file: Parsed deck with grid dimensions resolved.
    :param metadata: Optional extra metadata for the returned grid.
    :returns: A fully initialised `bores.grids.base.Grid`.
    :raises GridImportError: If required keywords are missing or the deck
        has no grid-dimension keyword.
    """
    if deck_file.dimensions is None:
        raise GridImportError("GRDECL file is missing the required `SPECGRID` keyword.")

    dims = deck_file.dimensions
    nx, ny, nz = dims

    unit_system = detect_unit_system(deck_file)
    map_axes = build_map_axes(deck_file)

    pinch_rec = deck_file.get("PINCH")
    pinch = pinch_rec["thickness"] if pinch_rec is not None else None
    if pinch is None and deck_file.has("PINCHOUT"):
        pinch = 1e-6

    meta: dict[str, typing.Any] = dict(metadata or {})
    meta.update(
        map_axes=map_axes,
        pinch=pinch,
        nx=nx,
        ny=ny,
        nz=nz,
        dimensions=dims,
    )

    has_coord = deck_file.has("COORD")
    has_tops = deck_file.has("TOPS") or deck_file.has("DX") or deck_file.has("DXV")

    try:
        if has_coord:
            return assemble_corner_point(deck_file, nx, ny, nz, unit_system, meta)
        if has_tops:
            return assemble_cartesian(deck_file, nx, ny, nz, unit_system, meta)
    except GridImportError:
        raise
    except Exception as exc:
        raise GridImportError(
            f"Failed to construct grid from parsed GRDECL arrays: {exc}"
        ) from exc

    raise GridImportError(
        "GRDECL file contains neither `COORD` (corner-point) nor `TOPS` / `DX` / `DXV` "
        "(Cartesian) geometry keywords."
    )


def assemble_corner_point(
    deck_file: DeckFile,
    nx: Integer,
    ny: Integer,
    nz: Integer,
    unit_system: UnitSystem,
    meta: dict[str, typing.Any],
) -> Grid:
    """
    Build a corner-point `bores.grids.base.Grid` from a parsed deck.

    :param deck_file: Parsed deck.
    :param nx: Grid extent in x.
    :param ny: Grid extent in y.
    :param nz: Grid extent in z.
    :param unit_system: Detected unit system.
    :param meta: Metadata dict (augmented in-place and passed to factory).
    :returns: Fully initialised `bores.grids.base.Grid`.
    :raises GridImportError: If `COORD` or `ZCORN` are missing or
        malformed.
    """
    coord = deck_file.get("COORD")
    if coord is None:
        raise GridImportError("GRDECL file is missing the required `COORD` keyword.")

    zcorn = deck_file.get("ZCORN")
    if zcorn is None:
        raise GridImportError("GRDECL file is missing the required `ZCORN` keyword.")

    actnum_flat = deck_file.get("ACTNUM")
    if actnum_flat is not None:
        actnum = actnum_flat.astype(np.int32, copy=False).reshape(nz, ny, nx)
    else:
        actnum = None

    meta["source_format"] = "grdecl_corner_point"
    meta["actnum"] = actnum

    nnc_pairs, nnc_transmissibilities = build_nnc_arrays(deck_file, nx, ny, nz)
    fault_records = build_fault_records(deck_file)
    multflt = build_multflt(deck_file)
    return make_corner_point_grid(
        coord=coord,
        zcorn=zcorn,
        actnum=actnum,
        unit_system=unit_system,
        metadata=meta,
        nnc_cell_indices=nnc_pairs,
        nnc_transmissibilities=nnc_transmissibilities,
        fault_records=fault_records,
        fault_transmissibility_multipliers=multflt,
        positive_x_transmissibility_multipliers=deck_file.get("MULTX"),
        negative_x_transmissibility_multipliers=deck_file.get("MULTX-"),
        positive_y_transmissibility_multipliers=deck_file.get("MULTY"),
        negative_y_transmissibility_multipliers=deck_file.get("MULTY-"),
        positive_z_transmissibility_multipliers=deck_file.get("MULTZ"),
        negative_z_transmissibility_multipliers=deck_file.get("MULTZ-"),
    )


def assemble_cartesian(
    deck_file: DeckFile,
    nx: Integer,
    ny: Integer,
    nz: Integer,
    unit_system: UnitSystem,
    meta: dict[str, typing.Any],
) -> Grid:
    """
    Build a Cartesian `bores.grids.base.Grid` from `TOPS` /
    `DX` / `DY` / `DZ` (per-cell) or `DXV` / `DYV` / `DZV`
    (vector) keywords.

    Vector forms (`DXV` / `DYV` / `DZV`) are preferred when present
    and specify spacing directly as 1-D arrays of length `nx`, `ny`,
    and `nz`.  Per-cell forms are reshaped and sliced to 1-D.

    Also reads `FAULTS`, `MULTFLT`, `NNC`, and all six `MULT*`
    arrays, passing them on to :func:`make_cartesian_grid`.

    :param deck_file: Parsed deck.
    :param nx: Grid extent in x.
    :param ny: Grid extent in y.
    :param nz: Grid extent in z.
    :param unit_system: Detected unit system.
    :param meta: Metadata dict passed to the factory.
    :returns: Fully initialised `bores.grids.base.Grid`.
    :raises GridImportError: If required keywords are missing or malformed.
    """
    # Z origin from TOPS
    tops_flat = deck_file.get("TOPS")
    if tops_flat is not None:
        tops_flat = np.asarray(tops_flat, dtype=np.float64, copy=False)
        n_columns = nx * ny
        tops_col = tops_flat[:n_columns]
        z_top = tops_col.min()
        if tops_col.max() - tops_col.min() > 1.0:
            warnings.warn(
                "GRDECL `TOPS` values vary by more than 1 unit; the Cartesian factory "
                "uses a flat top surface at the minimum `TOPS` value. Geometry may be "
                "approximate for dipping grids.",
                stacklevel=6,
            )
    else:
        z_top = 0.0

    # Spacing vectors
    dx_1d = resolve_vector_spacing(deck_file, "DXV", "DX", nx, "x", nz, ny, nx)
    dy_1d = resolve_vector_spacing(deck_file, "DYV", "DY", ny, "y", nz, ny, nx)
    dz_1d = resolve_vector_spacing(deck_file, "DZV", "DZ", nz, "z", nz, ny, nx)

    if dx_1d is None:
        raise GridImportError(
            "Cartesian GRDECL grid is missing required spacing keyword `DX` or `DXV`."
        )
    if dy_1d is None:
        raise GridImportError(
            "Cartesian GRDECL grid is missing required spacing keyword `DY` or `DYV`."
        )
    if dz_1d is None:
        raise GridImportError(
            "Cartesian GRDECL grid is missing required spacing keyword `DZ` or `DZV`."
        )

    actnum_flat = deck_file.get("ACTNUM")
    meta["source_format"] = "grdecl_cartesian"
    if actnum_flat is not None:
        meta["actnum"] = actnum_flat.astype(np.int32).reshape(nz, ny, nx)

    nnc_pairs, nnc_transmissibilities = build_nnc_arrays(deck_file, nx, ny, nz)
    fault_records = build_fault_records(deck_file)
    multflt = build_multflt(deck_file)

    # Store pinch in metadata so dump_grdecl can re-emit it.
    if meta.get("pinch") is not None:
        meta["pinch"] = meta["pinch"]
    return make_cartesian_grid(
        nx=nx,
        ny=ny,
        nz=nz,
        dx=dx_1d,  # type: ignore[arg-type]
        dy=dy_1d,  # type: ignore[arg-type]
        dz=dz_1d,  # type: ignore[arg-type]
        origin=(0.0, 0.0, z_top),
        unit_system=unit_system,
        metadata=meta,
        fault_records=fault_records,
        fault_transmissibility_multipliers=multflt,
        nnc_cell_indices=nnc_pairs,
        nnc_transmissibilities=nnc_transmissibilities,
        positive_x_transmissibility_multipliers=deck_file.get("MULTX"),
        negative_x_transmissibility_multipliers=deck_file.get("MULTX-"),
        positive_y_transmissibility_multipliers=deck_file.get("MULTY"),
        negative_y_transmissibility_multipliers=deck_file.get("MULTY-"),
        positive_z_transmissibility_multipliers=deck_file.get("MULTZ"),
        negative_z_transmissibility_multipliers=deck_file.get("MULTZ-"),
    )


GRDECL_SOURCES = frozenset({"grdecl_corner_point", "grdecl_cartesian"})


def build_grdecl_text(grid: Grid, *, actnum: ActNumArray | None = None) -> str:
    meta: typing.Mapping[str, typing.Any] = getattr(grid, "metadata", {}) or {}
    source_format: str = meta.get("source_format", "")
    if source_format not in GRDECL_SOURCES:
        raise GridExportError(
            f"Cannot export a `Grid` with `source_format={source_format!r}` to GRDECL. "
            "Only grids originally loaded by `load_grdecl()` support GRDECL export.  "
            f"Supported source formats: {sorted(GRDECL_SOURCES)}."
        )
    if source_format == "grdecl_cartesian":
        return build_grdecl_cartesian_text(grid, actnum=actnum)
    return build_grdecl_corner_point_text(grid, actnum=actnum)


def _emit_specgrid(lines: list[str], nx: Integer, ny: Integer, nz: Integer) -> None:
    """Append a `SPECGRID` block."""
    lines.append("SPECGRID")
    lines.append(f"  {nx}  {ny}  {nz}  1  F /")
    lines.append("")


def _emit_gridunit(lines: list[str], unit_system: UnitSystem) -> None:
    """Append a `GRIDUNIT` block."""
    unit_str = US_TO_GRIDUNIT.get(unit_system, "FEET")
    lines.append("GRIDUNIT")
    lines.append(f"  '{unit_str}  ' '        ' /")
    lines.append("")


def _emit_mapaxes(lines: list[str], map_axes: MapAxes) -> None:
    """Append `MAPUNITS` and `MAPAXES` blocks."""
    # Eclipse order for MAPAXES: Y-axis point, origin, X-axis point
    my = map_axes.map_y_axis_point
    o = map_axes.origin
    mx = map_axes.map_x_axis_point

    unit_str = US_TO_MAPUNITS.get(map_axes.unit_system, "FEET")
    lines.append("MAPUNITS")
    lines.append(f"  '{unit_str}' /")
    lines.append("")

    lines.append("MAPAXES")
    lines.append(
        f"  {my[0]:.6f}  {my[1]:.6f}  {o[0]:.6f}  {o[1]:.6f}  {mx[0]:.6f}  {mx[1]:.6f}  /"
    )
    lines.append("")


def _emit_actnum(
    lines: list[str],
    actnum: ActNumArray,
    n_cells: int,
    nx: Integer,
    ny: Integer,
    nz: Integer,
) -> None:
    """Append an `ACTNUM` block in Eclipse Fortran order (i fastest)."""
    actnum_arr = np.asarray(actnum, dtype=np.int32, copy=False)
    if len(actnum_arr) != n_cells:
        raise GridExportError(
            f"`actnum` length {len(actnum_arr)} does not match `n_cells` {n_cells}."
        )
    lines.append("")
    lines.append("ACTNUM")
    flat = actnum_arr.reshape(nz, ny, nx).transpose(2, 1, 0).ravel(order="F")
    for i in range(0, len(flat), 20):
        chunk = flat[i : i + 20]
        lines.append("  " + "  ".join(str(int(v)) for v in chunk))
    lines.append("/")


def _emit_mult_array(
    lines: list[str],
    keyword: str,
    array: NumberArray[OneDimension],
    nx: Integer,
    ny: Integer,
    nz: Integer,
) -> None:
    """Append a `MULT*` array block in Eclipse Fortran order (i fastest)."""
    lines.append("")
    lines.append(keyword)
    flat = (
        np
        .asarray(array, dtype=np.float64, copy=False)
        .reshape(nz, ny, nx)
        .transpose(2, 1, 0)
        .ravel(order="F")
    )
    for i in range(0, len(flat), 6):
        chunk = flat[i : i + 6]
        lines.append("  " + "  ".join(f"{v:.6f}" for v in chunk))
    lines.append("/")


def _emit_mult_arrays(
    lines: list[str],
    grid: Grid,
    nx: Integer,
    ny: Integer,
    nz: Integer,
) -> None:
    """Emit all non-`None` `MULT*` arrays from `grid`."""
    pairs = [
        ("MULTX", grid.positive_x_transmissibility_multipliers),
        ("MULTX-", grid.negative_x_transmissibility_multipliers),
        ("MULTY", grid.positive_y_transmissibility_multipliers),
        ("MULTY-", grid.negative_y_transmissibility_multipliers),
        ("MULTZ", grid.positive_z_transmissibility_multipliers),
        ("MULTZ-", grid.negative_z_transmissibility_multipliers),
    ]
    for kw, array in pairs:
        if array is not None:
            _emit_mult_array(lines, kw, array, nx, ny, nz)


def _emit_faults(lines: list[str], grid: Grid, nx: Integer, ny: Integer) -> None:
    """
    Append a `FAULTS` block to `lines` covering all named faults on the grid.

    Named faults can manifest in two ways on the grid and both are handled here:

    - **Face-based faults** (`grid.fault_face_indices`): cell pairs that share
      a geometric face. The face index is looked up in `grid.face_cell_indices`
      to recover the two adjacent cells, which are then converted to 1-based IJK
      and compared to infer the face direction.

    - **NNC-based faults** (`grid.nnc_fault_indices`): cell pairs from the
      original `FAULTS` records that had no shared geometric face (e.g. across
      a pinched-out layer) and were instead stored as `FAULT_NNC` connections.
      Their cell indices are read directly from `grid.nnc_cell_indices` and
      converted to IJK in the same way.

    A fault that appears in both maps (some faces resolved geometrically, others
    as NNCs) will have records emitted from both passes under the same fault name,
    which is correct Eclipse behaviour.

    Faces or NNC pairs whose IJK coordinates are identical in all three directions
    (degenerate connections) are skipped with a warning.

    Does nothing if both `grid.fault_face_indices` and `grid.nnc_fault_indices`
    are absent or empty.

    :param lines: Output text lines list, mutated in-place.
    :param grid: Source grid.
    :param nx: Grid dimension in x.
    :param ny: Grid dimension in y.
    :param nz: Grid dimension in z.
    """
    has_face_faults = bool(grid.fault_face_indices)
    has_nnc_faults = bool(grid.nnc_fault_indices)
    if not has_face_faults and not has_nnc_faults:
        return

    def _flat_to_ijk(flat: int) -> tuple[int, int, int]:
        i = flat % nx
        j = (flat // nx) % ny
        k = flat // (nx * ny)
        # Add 1, to move from 0-based to 1-based indexing which Eclipse uses
        return (i + 1, j + 1, k + 1)

    lines.append("")
    lines.append("FAULTS")

    # Face-based faults
    if has_face_faults:
        for fault_name, face_indices in sorted(grid.fault_face_indices.items()):  # type: ignore
            for face_idx in face_indices:
                owner = grid.face_cell_indices[face_idx, 0]
                neighbour = grid.face_cell_indices[face_idx, 1]
                if owner < 0 or neighbour < 0:
                    continue

                oi, oj, ok = _flat_to_ijk(owner)
                ni, nj, nk = _flat_to_ijk(neighbour)

                if ni != oi:
                    face_dir = "I" if ni > oi else "I-"
                    if ni < oi:
                        oi, oj, ok = ni, nj, nk
                elif nj != oj:
                    face_dir = "J" if nj > oj else "J-"
                    if nj < oj:
                        oi, oj, ok = ni, nj, nk
                elif nk != ok:
                    face_dir = "K" if nk > ok else "K-"
                    if nk < ok:
                        oi, oj, ok = ni, nj, nk
                else:
                    warnings.warn(
                        f"Fault {fault_name!r}: face {face_idx} connects cells with "
                        f"identical IJK ({oi},{oj},{ok}) - cannot determine direction. "
                        f"Skipping.",
                        stacklevel=4,
                    )
                    continue

                lines.append(
                    f"  '{fault_name}'  {oi}  {oi}  {oj}  {oj}  {ok}  {ok}  '{face_dir}'  /"
                )

    # Fault NNCs (cell pairs that had no shared geometric face)
    if has_nnc_faults:
        assert grid.nnc_cell_indices is not None
        for fault_name, nnc_indices in sorted(grid.nnc_fault_indices.items()):  # type: ignore
            for nnc_idx in nnc_indices:
                c1 = grid.nnc_cell_indices[nnc_idx, 0]
                c2 = grid.nnc_cell_indices[nnc_idx, 1]
                oi, oj, ok = _flat_to_ijk(c1)
                ni, nj, nk = _flat_to_ijk(c2)

                if ni != oi:
                    face_dir = "I" if ni > oi else "I-"
                    if ni < oi:
                        oi, oj, ok = ni, nj, nk
                elif nj != oj:
                    face_dir = "J" if nj > oj else "J-"
                    if nj < oj:
                        oi, oj, ok = ni, nj, nk
                elif nk != ok:
                    face_dir = "K" if nk > ok else "K-"
                    if nk < ok:
                        oi, oj, ok = ni, nj, nk
                else:
                    warnings.warn(
                        f"Fault {fault_name!r}: NNC {nnc_idx} connects cells with "
                        f"identical IJK ({oi},{oj},{ok}) - cannot determine direction. "
                        f"Skipping.",
                        stacklevel=4,
                    )
                    continue

                lines.append(
                    f"  '{fault_name}'  {oi}  {oi}  {oj}  {oj}  {ok}  {ok}  '{face_dir}'  /"
                )

    lines.append("/")
    lines.append("")


def _emit_multflt(lines: list[str], grid: Grid) -> None:
    """Append a `MULTFLT` block from `grid.fault_transmissibility_multipliers`."""
    if not grid.fault_transmissibility_multipliers:
        return
    lines.append("")
    lines.append("MULTFLT")
    for name, mult in sorted(grid.fault_transmissibility_multipliers.items()):
        lines.append(f"  '{name}'  {mult:.6f}  /")
    lines.append("/")
    lines.append("")


def _emit_nnc(lines: list[str], grid: Grid, nx: Integer, ny: Integer) -> None:
    """
    Emit a `NNC` block containing only explicitly user-defined NNCs
    (`ConnectionType.USER_NNC`).

    Fault-derived NNCs are emitted by `_emit_faults` via
    `grid.nnc_fault_indices`. Pinchout NNCs are implicitly
    reconstructed by the simulator from the `PINCH` keyword and must
    not be listed here.
    """
    if grid.nnc_cell_indices is None or grid.nnc_connection_types is None:
        return

    user_type = int(ConnectionType.USER_NNC)
    has_transmissibility = grid.nnc_transmissibilities is not None and len(
        grid.nnc_transmissibilities
    ) == len(grid.nnc_cell_indices)

    def _flat_to_ijk(flat: int) -> tuple[int, int, int]:
        i = flat % nx
        j = (flat // nx) % ny
        k = flat // (nx * ny)
        return i + 1, j + 1, k + 1

    user_nnc_lines: list[str] = []
    for idx, (c1, c2) in enumerate(grid.nnc_cell_indices):
        if int(grid.nnc_connection_types[idx]) != user_type:
            continue

        i1, j1, k1 = _flat_to_ijk(int(c1))
        i2, j2, k2 = _flat_to_ijk(int(c2))
        if has_transmissibility:
            transmissibility = grid.nnc_transmissibilities[idx]  # type: ignore
            transmissibility_str = (
                f"{transmissibility:.6e}" if not np.isnan(transmissibility) else "0.0 -- T unknown"
            )
        else:
            transmissibility_str = "0.0 -- T unknown"
        user_nnc_lines.append(f"  {i1}  {j1}  {k1}  {i2}  {j2}  {k2}  {transmissibility_str}  /")

    if not user_nnc_lines:
        return

    lines.append("")
    lines.append("NNC")
    lines.extend(user_nnc_lines)
    lines.append("/")
    lines.append("")


def _emit_pinch(
    lines: list[str],
    meta: typing.Mapping[str, typing.Any],
) -> None:
    """
    Emit a `PINCH` block when the pinch tolerance is stored in metadata.

    :param lines: Output text lines list.
    :param meta: Grid metadata mapping; reads `meta["pinch"]`.
    """
    pinch = meta.get("pinch")
    if pinch is None:
        return
    lines.append("")
    lines.append("PINCH")
    lines.append(f"  {float(pinch):.6g}  /")
    lines.append("")


@numba.njit(cache=True, parallel=True)
def compute_cell_bounds_from_vertices(
    local_vertices: NumberArray[TwoDimensions],
    cell_face_offsets: IntArray[OneDimension],
    cell_face_indices: IntArray[OneDimension],
    face_vertex_offsets: IntArray[OneDimension],
    face_vertex_indices: IntArray[OneDimension],
    n_cells: int,
) -> tuple[NumberArray[TwoDimensions], NumberArray[TwoDimensions]]:
    """
    Per-cell (min, max) over each cell's own vertices, walked via
    cell -> face -> vertex CSR connectivity.

    :param local_vertices: Shape `(n_vertices, 3)`.
    :param cell_face_offsets: Shape `(n_cells + 1,)` CSR offsets.
    :param cell_face_indices: Shape `(n_cell_face,)` face index per
        (cell, face) occurrence.
    :param face_vertex_offsets: Shape `(n_faces + 1,)` CSR offsets.
    :param face_vertex_indices: Shape `(n_face_vertex,)` vertex index per
        (face, vertex) occurrence.
    :param n_cells: Number of cells.
    :returns: `(cell_min, cell_max)`, each shape `(n_cells, 3)`.
    """
    cell_min = np.empty((n_cells, 3), dtype=local_vertices.dtype)
    cell_max = np.empty((n_cells, 3), dtype=local_vertices.dtype)

    for cell_idx in numba.prange(n_cells):
        xmin = ymin = zmin = np.inf
        xmax = ymax = zmax = -np.inf
        face_start = cell_face_offsets[cell_idx]
        face_end = cell_face_offsets[cell_idx + 1]

        for f in range(face_start, face_end):
            face_idx = cell_face_indices[f]
            vertex_start = face_vertex_offsets[face_idx]
            vertex_end = face_vertex_offsets[face_idx + 1]

            for v in range(vertex_start, vertex_end):
                vertex_idx = face_vertex_indices[v]
                x = local_vertices[vertex_idx, 0]
                y = local_vertices[vertex_idx, 1]
                z = local_vertices[vertex_idx, 2]

                if x < xmin:
                    xmin = x
                if x > xmax:
                    xmax = x

                if y < ymin:
                    ymin = y
                if y > ymax:
                    ymax = y

                if z < zmin:
                    zmin = z
                if z > zmax:
                    zmax = z

        cell_min[cell_idx, 0] = xmin
        cell_min[cell_idx, 1] = ymin
        cell_min[cell_idx, 2] = zmin
        cell_max[cell_idx, 0] = xmax
        cell_max[cell_idx, 1] = ymax
        cell_max[cell_idx, 2] = zmax

    return cell_min, cell_max  # type: ignore[return-value]


def compute_local_cartesian_cell_bounds(
    grid: Grid, map_axes: MapAxes
) -> tuple[NumberArray[TwoDimensions], NumberArray[TwoDimensions]]:
    """
    Recompute each cell's axis-aligned bounding box in local (pre-`MAPAXES`)
    space, from the grid's (map-space) vertex positions.

    `grid.cell_min_xyz`/`grid.cell_max_xyz` can't be inverse-rotated
    directly to get this: an axis-aligned bounding box computed in map
    space doesn't correspond to the local-space bounding box of the same
    cell once you rotate it back - a rotated box's AABB is, in general,
    larger than the box itself along the new axes. Unlike the corner-point
    writer (`rederive_corner_point_arrays`), Cartesian cells have no "z is
    a free axis" shortcut to fall back on, since `MAPAXES` generally
    rotates their edges away from the map-space X/Y axes entirely. So this
    re-derives the bound from each cell's actual vertices, individually
    inverse-transformed, rather than from the already-computed (map-space)
    per-cell AABB.

    :param grid: A Cartesian `Grid` whose `vertex_coordinates` are in map
        space (i.e. built with `apply_map_axes=True`, the default).
    :param map_axes: The `MAPAXES` that was applied when building `grid`.
    :returns: `(cell_min_xyz, cell_max_xyz)`, each shape `(n_cells, 3)`, in
        local space.
    """
    local_xy = get_map_axes_xy_inverse(
        xy=grid.vertex_coordinates[:, :2],  # type: ignore[arg-type]
        map_axes=map_axes,
    )
    local_vertices = np.column_stack([local_xy, grid.vertex_coordinates[:, 2]])
    return compute_cell_bounds_from_vertices(
        local_vertices=local_vertices,  # type: ignore[arg-type]
        cell_face_offsets=grid.cell_face_offsets,
        cell_face_indices=grid.cell_face_indices,
        face_vertex_offsets=grid.face_vertex_offsets,
        face_vertex_indices=grid.face_vertex_indices,
        n_cells=grid.n_cells,
    )


def build_grdecl_cartesian_text(grid: Grid, *, actnum: ActNumArray | None = None) -> str:
    """
    Build a GRDECL text representation for a Cartesian grid.

    Emits (in order): `SPECGRID`, `GRIDUNIT`, `MAPAXES` (if present),
    `TOPS`, `DXV`, `DYV`, `DZV`, `ACTNUM` (if present),
    `MULTX` / `MULTY` / `MULTZ` / `-` variants (if present),
    `FAULTS` (if present), `MULTFLT` (if present),
    `NNC` (if present), `PINCH` (if present).

    :param grid: Cartesian grid with `source_format = "grdecl_cartesian"`
        in metadata.
    :param actnum: Optional active-cell mask; falls back to
        `grid.metadata["actnum"]`.
    :returns: GRDECL text string.
    """
    meta: typing.Mapping[str, typing.Any] = getattr(grid, "metadata", {}) or {}

    if grid.dimensions is not None:
        nx, ny, nz = grid.dimensions
    else:
        nx = meta.get("nx")
        ny = meta.get("ny")
        nz = meta.get("nz")
        if nx is None or ny is None or nz is None:
            nx, ny, nz = 1, 1, grid.n_cells

    n_cells = nx * ny * nz
    if n_cells != grid.n_cells:
        raise GridExportError(
            f"Stored dimensions ({nx}x{ny}x{nz}={n_cells}) do not match "
            f"`grid.n_cells={grid.n_cells}`."
        )

    lines: list[str] = []

    _emit_specgrid(lines, nx, ny, nz)
    _emit_gridunit(lines, grid.unit_system)

    map_axes: MapAxes | None = meta.get("map_axes")
    if map_axes is not None:
        _emit_mapaxes(lines, map_axes)
        # grid.cell_min_xyz/cell_max_xyz are in map space; TOPS/DXV/DYV/DZV
        # need local-space bounds so they stay consistent with the `MAPAXES`
        # card just emitted above (see `compute_local_cartesian_cell_bounds`).
        cell_min_xyz, cell_max_xyz = compute_local_cartesian_cell_bounds(grid, map_axes)
    else:
        cell_min_xyz, cell_max_xyz = grid.cell_min_xyz, grid.cell_max_xyz

    # TOPS - use the top-layer cell minimum z values.
    # Order: i fastest (Eclipse Fortran / C with transposed axes).
    top_layer_indices = [i + j * nx for j in range(ny) for i in range(nx)]
    tops_vals = cell_min_xyz[top_layer_indices, 2]
    lines.append("TOPS")
    for i in range(0, len(tops_vals), 6):
        chunk = tops_vals[i : i + 6]
        lines.append("  " + "  ".join(f"{v:.6f}" for v in chunk))
    lines.append("/")
    lines.append("")

    # DXV / DYV / DZV - one representative value per cell count.
    # We extract from cell bounding boxes: first j=0, k=0 row for DXV; etc.
    dx_vals = cell_max_xyz[:nx, 0] - cell_min_xyz[:nx, 0]
    lines.append("DXV")
    lines.append("  " + "  ".join(f"{v:.6f}" for v in dx_vals))
    lines.append("/")
    lines.append("")

    # DYV: cells at i=0, k=0 -> indices 0, nx, 2*nx, …
    dy_indices = np.arange(ny) * nx
    dy_vals = cell_max_xyz[dy_indices, 1] - cell_min_xyz[dy_indices, 1]
    lines.append("DYV")
    lines.append("  " + "  ".join(f"{v:.6f}" for v in dy_vals))
    lines.append("/")
    lines.append("")

    # DZV: cells at i=0, j=0, k=0..nz-1 -> indices 0, nx*ny, 2*nx*ny, …
    dz_indices = np.arange(nz) * nx * ny
    dz_vals = cell_max_xyz[dz_indices, 2] - cell_min_xyz[dz_indices, 2]
    lines.append("DZV")
    lines.append("  " + "  ".join(f"{v:.6f}" for v in dz_vals))
    lines.append("/")
    lines.append("")

    effective_actnum = actnum if actnum is not None else meta.get("actnum")
    if effective_actnum is not None:
        _emit_actnum(lines, effective_actnum, grid.n_cells, nx, ny, nz)

    _emit_mult_arrays(lines, grid, nx, ny, nz)
    _emit_faults(lines, grid, nx, ny)
    _emit_multflt(lines, grid)
    _emit_nnc(lines, grid, nx, ny)
    _emit_pinch(lines, meta)
    return "\n".join(lines)


def build_grdecl_corner_point_text(grid: Grid, *, actnum: ActNumArray | None = None) -> str:
    """
    Build a GRDECL text representation for a corner-point grid.

    Emits (in order): `SPECGRID`, `GRIDUNIT`, `MAPAXES` (if present),
    `COORD`, `ZCORN`, `ACTNUM` (if present),
    `MULTX` / `MULTY` / `MULTZ` / `-` variants (if present),
    `FAULTS` (if present), `MULTFLT` (if present),
    `NNC` (if present), `PINCH` (if present).

    :param grid: Corner-point grid with `source_format = "grdecl_corner_point"`
        in metadata.
    :param actnum: Optional active-cell mask.
    :returns: GRDECL text string.
    """
    coord, zcorn, nx, ny, nz = rederive_corner_point_arrays(grid)
    meta: typing.Mapping[str, typing.Any] = getattr(grid, "metadata", {}) or {}

    lines: list[str] = []

    _emit_specgrid(lines, nx, ny, nz)
    _emit_gridunit(lines, grid.unit_system)

    map_axes: MapAxes | None = meta.get("map_axes")
    if map_axes is not None:
        _emit_mapaxes(lines, map_axes)

    # COORD (ny + 1, nx + 1, 6)
    # Already in Eclipse COORD order, so we flatten and write as is
    lines.append("COORD")
    flat_coord = coord.reshape(-1, 6)
    for row in flat_coord:
        x1, y1, z1, x2, y2, z2 = row
        lines.append(f"  {x1:.6f}  {y1:.6f}  {z1:.6f}  {x2:.6f}  {y2:.6f}  {z2:.6f}")
    lines.append("/")
    lines.append("")

    # ZCORN (nz*2, ny*2, nx*2)
    # We flatten the 3D array; x vary fastest, and z slowest just as Eclipse writes it
    lines.append("ZCORN")
    flat_zcorn = zcorn.ravel()
    for i in range(0, len(flat_zcorn), 6):
        chunk = flat_zcorn[i : i + 6]
        lines.append("  " + "  ".join(f"{v:.6f}" for v in chunk))
    lines.append("/")
    lines.append("")

    effective_actnum = actnum if actnum is not None else meta.get("actnum")
    if effective_actnum is not None:
        _emit_actnum(lines, effective_actnum, grid.n_cells, nx, ny, nz)

    _emit_mult_arrays(lines, grid, nx, ny, nz)
    _emit_faults(lines, grid, nx, ny)
    _emit_multflt(lines, grid)
    _emit_nnc(lines, grid, nx, ny)
    _emit_pinch(lines, meta)
    return "\n".join(lines)
