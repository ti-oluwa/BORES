"""Fault definition and application API for reservoir grids."""

import typing
import warnings

import attrs
import numba
import numpy as np
from typing_extensions import Self

from bores.deck.file import DeckFile
from bores.errors import ValidationError
from bores.grids.base import ConnectionType, Grid
from bores.grids.factories.base import VALID_FAULT_FACE_DIRECTIONS
from bores.types import BooleanArray, IntArray, Integer, Number, OneDimension, TwoDimensions

__all__ = ["Fault", "apply_faults", "remove_faults"]


@attrs.frozen(slots=True, unsafe_hash=True)
class Fault:
    """
    Definition of a single named fault and its optional transmissibility multiplier.

    Combines the information from the Eclipse `FAULTS` and `MULTFLT` keywords
    into a single object. A fault without a transmissibility multiplier is valid
    (`transmissibility_multiplier=None`); it will still classify faces but
    will not modify their transmissibilities.

    All IJK indices are 1-based, following Eclipse convention.

    :param name: Fault name as declared in the `FAULTS` keyword.
    :param i1: Start of the fault's cell range in the I direction (1-based).
    :param i2: End of the fault's cell range in the I direction (1-based).
    :param j1: Start of the fault's cell range in the J direction (1-based).
    :param j2: End of the fault's cell range in the J direction (1-based).
    :param k1: Start of the fault's cell range in the K direction (1-based).
    :param k2: End of the fault's cell range in the K direction (1-based).
    :param face_direction: Face direction of the fault interface.
        Must be one of `"X"`, `"X-"`, `"Y"`, `"Y-"`, `"Z"`, `"Z-"`. A direction without a
        minus sign selects the face on the positive side of each cell in the range.
    :param transmissibility_multiplier: MULTFLT value for this fault.
        `None` when no multiplier is declared.
    """

    name: str
    i1: int
    i2: int
    j1: int
    j2: int
    k1: int
    k2: int
    face_direction: str
    transmissibility_multiplier: Number | None = attrs.field(default=None)

    def __attrs_post_init__(self) -> None:
        if not self.name:
            raise ValidationError("`name` must be a non-empty string.")
        if self.face_direction not in VALID_FAULT_FACE_DIRECTIONS:
            raise ValidationError(
                f"`face_direction` must be one of {sorted(VALID_FAULT_FACE_DIRECTIONS)}; "
                f"got {self.face_direction!r}."
            )
        if self.i1 < 1 or self.i2 < self.i1:
            raise ValidationError(
                f"Invalid I range: i1={self.i1}, i2={self.i2}. Expected 1 <= i1 <= i2."
            )
        if self.j1 < 1 or self.j2 < self.j1:
            raise ValidationError(
                f"Invalid J range: j1={self.j1}, j2={self.j2}. Expected 1 <= j1 <= j2."
            )
        if self.k1 < 1 or self.k2 < self.k1:
            raise ValidationError(
                f"Invalid K range: k1={self.k1}, k2={self.k2}. Expected 1 <= k1 <= k2."
            )
        if self.transmissibility_multiplier is not None and (
            self.transmissibility_multiplier < 0.0
        ):
            raise ValidationError(
                f"`transmissibility_multiplier` must be non-negative; "
                f"got {self.transmissibility_multiplier}."
            )

    @typing.overload
    @classmethod
    def from_deck(cls, deck_file: DeckFile, *, name: str) -> Self: ...
    @typing.overload
    @classmethod
    def from_deck(cls, deck_file: DeckFile, *, name: None) -> list[Self]: ...
    @typing.overload
    @classmethod
    def from_deck(cls, deck_file: DeckFile) -> list[Self]: ...

    @classmethod
    def from_deck(
        cls,
        deck_file: DeckFile,
        *,
        name: str | None = None,
    ) -> Self | list[Self]:
        """
        Construct one or all `Fault` objects from a parsed `DeckFile`.

        Reads the `FAULTS` keyword for geometry and `MULTFLT` for multipliers.
        When `name` is given, returns a single `Fault` for that fault.
        When `name` is `None`, returns a list of all faults in the deck.

        :param deck_file: Parsed `bores.deck.file.DeckFile`.
        :param name: Name of a specific fault to extract, or `None` for all.
        :returns: A single `Fault` if `name` is given; a
            `List[Fault]` otherwise.
        :raises ValidationError: If `name` is given but not found in the deck.
        :raises ValidationError: If the deck contains no `FAULTS` keyword.
        """
        fault_records = deck_file.get("FAULTS")
        if not fault_records:
            raise ValidationError("No FAULTS keyword found in the provided data file.")

        multflt_records = deck_file.get("MULTFLT") or []
        multflt_map: dict[str, Number] = {
            record["name"]: record["multiplier"] for record in multflt_records
        }
        if name is not None:
            matching = [record for record in fault_records if record["name"] == name]
            if not matching:
                available = sorted({record["name"] for record in fault_records})
                raise ValidationError(
                    f"Fault {name!r} not found in FAULTS keyword. Available faults: {available}."
                )
            record = matching[0]
            return cls(
                name=record["name"],
                i1=record["i1"],
                i2=record["i2"],
                j1=record["j1"],
                j2=record["j2"],
                k1=record["k1"],
                k2=record["k2"],
                face_direction=record["face"],
                transmissibility_multiplier=multflt_map.get(record["name"]),
            )

        return [
            cls(
                name=record["name"],
                i1=record["i1"],
                i2=record["i2"],
                j1=record["j1"],
                j2=record["j2"],
                k1=record["k1"],
                k2=record["k2"],
                face_direction=record["face"],
                transmissibility_multiplier=multflt_map.get(record["name"]),
            )
            for record in fault_records
        ]

    def __repr__(self) -> str:
        mult_str = (
            f", multiplier={self.transmissibility_multiplier}"
            if self.transmissibility_multiplier is not None
            else ""
        )
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"i=({self.i1}..{self.i2}), "
            f"j=({self.j1}..{self.j2}), "
            f"k=({self.k1}..{self.k2}), "
            f"face={self.face_direction!r}"
            f"{mult_str}"
            f")"
        )


def apply_faults(grid: Grid, *faults: Fault) -> Grid:
    """
    Apply one or more `Fault` definitions to a grid and return a new grid.

    Useful for introducing or updating fault definitions at runtime, e.g,
    during history matching, sensitivity studies, or when
    a fault is not present in the original deck.

    **What changes on the returned grid**:

    - `grid.fault_face_indices` - updated with the new fault's face memberships.
    - `grid.face_connection_types` - faces that now belong to a fault are
      reclassified from `INTERIOR_FACE`/`BOUNDARY_FACE` to
      `INTERIOR_FAULT_FACE`/`BOUNDARY_FAULT_FACE`.
    - Fault cell pairs that share no geometric face are not connected and are skipped.
    - `grid.fault_transmissibility_multipliers` - updated from the
      `transmissibility_multiplier` on each `Fault`. A fault with no multiplier ends up with none,
      including one that replaces an earlier fault of the same name.

    Faults already present on the grid with the same name are replaced. Faults
    with new names are merged in. To remove all existing faults first, call
    `remove_faults` before this function.

    :param grid: The `Grid` to apply faults to.
    :param faults: One or more `Fault` objects.
    :returns: A new `Grid` with the faults applied.
    :raises ValidationError: If no faults are supplied, or if any fault record
        references cells outside the grid bounds.
    """
    if not faults:
        raise ValidationError("At least one `Fault` must be supplied to `apply_faults`.")

    if grid.dimensions is not None:
        nx, ny, nz = grid.dimensions
    else:
        meta: typing.Mapping[str, typing.Any] = getattr(grid, "metadata", {}) or {}
        nx = meta.get("nx", 0)
        ny = meta.get("ny", 0)
        nz = meta.get("nz", 0)

    if nx == 0 or ny == 0 or nz == 0:
        raise ValidationError(
            "Grid metadata does not contain `nx`, `ny`, `nz`. "
            "Cannot resolve IJK cell indices for fault application. "
            "Only grids loaded from a GRDECL source support `apply_faults`."
        )

    validate_fault_bounds(faults, nx, ny, nz)
    return apply_faults_to_grid(grid, faults, nx, ny, nz)


def remove_faults(grid: Grid, *names: str) -> Grid:
    """
    Remove named faults from a grid and return a new grid.

    Strips the specified faults from all grid fault data structures:
    face classifications are reverted to plain `INTERIOR_FACE`/`BOUNDARY_FACE`,
    and their transmissibility multipliers are dropped.

    When `names` is empty, **all** faults are removed.

    :param grid: The `Grid` to remove faults from.
    :param names: Names of faults to remove. If none are given, all faults
        are removed.
    :returns: A new `Grid` with the specified faults removed.
    :raises ValidationError: If a named fault does not exist on the grid.
    """
    existing_names: set[str] = set()
    if grid.fault_face_indices:
        existing_names.update(grid.fault_face_indices.keys())

    names_to_remove: frozenset[str] = frozenset(names) if names else frozenset(existing_names)

    unknown = names_to_remove - existing_names
    if unknown:
        raise ValidationError(
            f"Faults not found on grid: {sorted(unknown)}. "
            f"Available faults: {sorted(existing_names)}."
        )
    return remove_faults_from_grid(grid, names_to_remove)


def validate_fault_bounds(
    faults: tuple[Fault, ...], nx: Integer, ny: Integer, nz: Integer
) -> None:
    """
    Validate that all fault IJK ranges are within the grid bounds.

    :param faults: Fault records to validate.
    :param nx: Grid extent in x.
    :param ny: Grid extent in y.
    :param nz: Grid extent in z.
    :raises ValidationError: On any out-of-bounds IJK index.
    """
    for fault in faults:
        if fault.i2 > nx:
            raise ValidationError(f"Fault {fault.name!r}: i2={fault.i2} exceeds grid nx={nx}.")
        if fault.j2 > ny:
            raise ValidationError(f"Fault {fault.name!r}: j2={fault.j2} exceeds grid ny={ny}.")
        if fault.k2 > nz:
            raise ValidationError(f"Fault {fault.name!r}: k2={fault.k2} exceeds grid nz={nz}.")


FACE_DIRECTION_AXIS_STEP: dict[str, tuple[int, int]] = {
    "X": (0, 1),
    "X-": (0, -1),
    "Y": (1, 1),
    "Y-": (1, -1),
    "Z": (2, 1),
    "Z-": (2, -1),
}
"""Axis (`0` for I, `1` for J, `2` for K) and step (`+1` or `-1`) that each fault face direction crosses."""


class FaultCellPairs(typing.NamedTuple):
    """Cell pairs on either side of a fault interface."""

    lower: IntArray[OneDimension]
    """Flat index of the lower-numbered cell of each pair."""

    upper: IntArray[OneDimension]
    """Flat index of the higher-numbered cell of each pair."""


class FaceLookup(typing.NamedTuple):
    """Interior faces of a grid, searchable by the pair of cells they separate."""

    keys: IntArray[OneDimension]
    """Ascending `lower_cell * n_cells + upper_cell` key of every interior face."""

    faces: IntArray[OneDimension]
    """Face index belonging to each entry of `keys`."""

    n_cells: int
    """Number of cells the keys were built with."""


@numba.njit(parallel=True, cache=True)
def fault_pair(
    i1: Integer,
    j1: Integer,
    k1: Integer,
    ni: Integer,
    nj: Integer,
    nk: Integer,
    axis: Integer,
    step: Integer,
    nx: Integer,
    ny: Integer,
    nz: Integer,
    lower: IntArray[OneDimension],
    upper: IntArray[OneDimension],
    in_bounds: BooleanArray[OneDimension],
) -> None:
    """
    Fill the cell pair across a fault interface for every cell of a fault's IJK box.

    Position `p` of the box is cell `(i1 + p % ni, j1 + (p // ni) % nj, k1 + p // (ni * nj))`, so
    `i` varies fastest and `k` slowest. Every position writes only its own slot of the outputs.

    :param i1: First I index of the box (1-based).
    :param j1: First J index of the box (1-based).
    :param k1: First K index of the box (1-based).
    :param ni: Number of cells of the box in I.
    :param nj: Number of cells of the box in J.
    :param nk: Number of cells of the box in K.
    :param axis: Axis the fault face crosses, `0` for I, `1` for J and `2` for K.
    :param step: `+1` if the neighbour is the next cell along `axis`, `-1` if it is the previous one.
    :param nx: Grid extent in x.
    :param ny: Grid extent in y.
    :param nz: Grid extent in z.
    :param lower: Output, the lower flat index of each pair. Meaningful where `in_bounds` is set.
    :param upper: Output, the higher flat index of each pair. Meaningful where `in_bounds` is set.
    :param in_bounds: Output, whether the neighbour lies inside the grid.
    """
    total = ni * nj * nk
    plane = ni * nj
    for position in numba.prange(total):  # type: ignore[attr-defined]
        i = i1 + position % ni
        j = j1 + (position // ni) % nj
        k = k1 + position // plane
        neighbour_i = i + step if axis == 0 else i
        neighbour_j = j + step if axis == 1 else j
        neighbour_k = k + step if axis == 2 else k
        if (
            neighbour_i < 1
            or neighbour_i > nx
            or neighbour_j < 1
            or neighbour_j > ny
            or neighbour_k < 1
            or neighbour_k > nz
        ):
            in_bounds[position] = False
            lower[position] = 0
            upper[position] = 0
            continue
        cell = (i - 1) + (j - 1) * nx + (k - 1) * nx * ny
        neighbour = (neighbour_i - 1) + (neighbour_j - 1) * nx + (neighbour_k - 1) * nx * ny
        in_bounds[position] = True
        lower[position] = min(cell, neighbour)
        upper[position] = max(cell, neighbour)


@numba.njit(parallel=True, cache=True)
def find_faces(
    keys: IntArray[OneDimension],
    faces: IntArray[OneDimension],
    n_cells: Integer,
    lower: IntArray[OneDimension],
    upper: IntArray[OneDimension],
    out: IntArray[OneDimension],
) -> None:
    """
    Look up the interior face shared by each cell pair, `-1` where the pair shares none.

    If several faces separate the same two cells, the one with the highest position in `keys` is
    returned. Every pair writes only its own slot of `out`.

    :param keys: Ascending face keys from `FaceLookup`.
    :param faces: Face index of each entry of `keys`.
    :param n_cells: Number of cells the keys were built with.
    :param lower: Lower flat index of each pair.
    :param upper: Higher flat index of each pair.
    :param out: Output, the face index of each pair or `-1`.
    """
    for pair in numba.prange(lower.shape[0]):  # type: ignore[attr-defined]
        target = lower[pair] * n_cells + upper[pair]
        low = 0
        high = keys.shape[0]
        while low < high:
            middle = (low + high) // 2
            if keys[middle] <= target:
                low = middle + 1
            else:
                high = middle
        if low > 0 and keys[low - 1] == target:
            out[pair] = faces[low - 1]
        else:
            out[pair] = -1


@numba.njit(cache=True)
def reclassify_faces(
    connection_types: IntArray[OneDimension],
    face_cell_indices: IntArray[TwoDimensions],
    face_indices: IntArray[OneDimension],
    interior_type: Integer,
    boundary_type: Integer,
) -> None:
    """
    Set the connection type of each listed face by whether it has a neighbouring cell.

    :param connection_types: Connection type of every face, updated in place.
    :param face_cell_indices: Shape `(n_faces, 2)` owner and neighbour of every face.
    :param face_indices: Faces to reclassify.
    :param interior_type: Type given to a face with a cell on both sides.
    :param boundary_type: Type given to a face with no cell on one side.
    """
    for position in range(face_indices.shape[0]):
        face = face_indices[position]
        if face_cell_indices[face, 0] < 0 or face_cell_indices[face, 1] < 0:
            connection_types[face] = boundary_type
        else:
            connection_types[face] = interior_type


def resolve_fault_cell_pairs(
    fault: Fault, nx: Integer, ny: Integer, nz: Integer
) -> FaultCellPairs:
    """
    Enumerate all cell pairs across a fault interface.

    Iterates the IJK box defined by the fault record and, for each cell in the
    box, finds its neighbour in the face direction. Pairs are returned with the lower
    flat index first, in the order of the box with `i` varying fastest and `k` slowest.

    Pairs whose neighbour is outside the grid bounds are skipped with a warning.

    :param fault: The fault record to resolve.
    :param nx: Grid extent in x.
    :param ny: Grid extent in y.
    :param nz: Grid extent in z.
    :returns: The lower and higher flat cell index of each pair.
    """
    axis, step = FACE_DIRECTION_AXIS_STEP[fault.face_direction]
    ni = fault.i2 - fault.i1 + 1
    nj = fault.j2 - fault.j1 + 1
    nk = fault.k2 - fault.k1 + 1
    total = ni * nj * nk
    lower = np.empty(total, dtype=np.int64)
    upper = np.empty(total, dtype=np.int64)
    in_bounds = np.empty(total, dtype=np.bool_)
    fault_pair(
        i1=fault.i1,
        j1=fault.j1,
        k1=fault.k1,
        ni=ni,
        nj=nj,
        nk=nk,
        axis=axis,
        step=step,
        nx=nx,
        ny=ny,
        nz=nz,
        lower=lower,
        upper=upper,
        in_bounds=in_bounds,
    )
    if in_bounds.all():
        return FaultCellPairs(lower=lower, upper=upper)

    for position in np.flatnonzero(~in_bounds):
        i = fault.i1 + position % ni
        j = fault.j1 + (position // ni) % nj
        k = fault.k1 + position // (ni * nj)
        neighbour = [i, j, k]
        neighbour[axis] += step
        warnings.warn(
            f"Fault {fault.name!r}: neighbour cell ({neighbour[0]},{neighbour[1]},{neighbour[2]}) "
            f"is outside grid bounds ({nx}x{ny}x{nz}). Skipping pair.",
            stacklevel=5,
        )
    return FaultCellPairs(
        lower=typing.cast("IntArray[OneDimension]", lower[in_bounds]),
        upper=typing.cast("IntArray[OneDimension]", upper[in_bounds]),
    )


def build_face_lookup(grid: Grid) -> FaceLookup:
    """
    Index the interior faces of `grid` by the pair of cells each one separates.

    Only faces with a cell on both sides are included. The index is used to find whether a
    fault's cell pair shares a geometric face.

    :param grid: The source grid.
    :returns: The searchable index.
    """
    n_cells = int(grid.n_cells)
    face_cells = grid.face_cell_indices
    interior_faces = np.flatnonzero((face_cells[:, 0] >= 0) & (face_cells[:, 1] >= 0))
    owners = face_cells[interior_faces, 0].astype(np.int64)
    neighbours = face_cells[interior_faces, 1].astype(np.int64)
    keys = np.minimum(owners, neighbours)
    keys *= n_cells
    keys += np.maximum(owners, neighbours)
    del owners, neighbours
    order = np.argsort(keys, kind="stable")
    return FaceLookup(
        keys=keys[order],
        faces=typing.cast("IntArray[OneDimension]", interior_faces[order]),
        n_cells=n_cells,
    )


def apply_faults_to_grid(
    grid: Grid, faults: tuple[Fault, ...], nx: Integer, ny: Integer, nz: Integer
) -> Grid:
    """
    Produce a new `Grid` with the given faults applied.

    For each fault, cell pairs are resolved from the IJK range. Each pair is
    checked against the face lookup:

    - If a shared face exists -> the face is classified as a fault face.
    - If no shared face exists -> the pair is not connected and is skipped.

    Existing faults with the same name are replaced; new fault names are merged.

    :param grid: The source grid.
    :param faults: Fault records to apply.
    :param nx: Grid extent in x.
    :param ny: Grid extent in y.
    :param nz: Grid extent in z.
    :returns: A new `Grid` with updated fault connectivity.
    """
    assert grid.face_connection_types is not None
    lookup = build_face_lookup(grid)
    face_cell_indices = np.ascontiguousarray(grid.face_cell_indices)
    fault_face_indices: dict[str, IntArray[OneDimension]] = dict(grid.fault_face_indices or {})
    connection_types = np.array(grid.face_connection_types, dtype=np.int8)
    for fault in faults:
        # A replaced fault gives up its old faces before the new ones are classified.
        replaced = fault_face_indices.pop(fault.name, None)
        if replaced is not None:
            reclassify_faces(
                connection_types=typing.cast("IntArray[OneDimension]", connection_types),
                face_cell_indices=typing.cast("IntArray[TwoDimensions]", face_cell_indices),
                face_indices=typing.cast("IntArray[OneDimension]", np.ascontiguousarray(replaced)),
                interior_type=int(ConnectionType.INTERIOR_FACE),
                boundary_type=int(ConnectionType.BOUNDARY_FACE),
            )

    for fault in faults:
        pairs = resolve_fault_cell_pairs(fault, nx, ny, nz)
        found = np.empty(pairs.lower.shape[0], dtype=np.int64)
        find_faces(
            keys=lookup.keys,
            faces=lookup.faces,
            n_cells=lookup.n_cells,
            lower=pairs.lower,
            upper=pairs.upper,
            out=found,
        )
        found = found[found >= 0]
        if found.shape[0] == 0:
            continue
        reclassify_faces(
            connection_types=typing.cast("IntArray[OneDimension]", connection_types),
            face_cell_indices=typing.cast("IntArray[TwoDimensions]", face_cell_indices),
            face_indices=typing.cast("IntArray[OneDimension]", found),
            interior_type=int(ConnectionType.INTERIOR_FAULT_FACE),
            boundary_type=int(ConnectionType.BOUNDARY_FAULT_FACE),
        )
        fault_face_indices[fault.name] = typing.cast(
            "IntArray[OneDimension]", found.astype(np.int32)
        )

    updated_multipliers: dict[str, Number] = dict(grid.fault_transmissibility_multipliers or {})
    for fault in faults:
        # A replaced fault takes only the multiplier it now carries, never the old one.
        updated_multipliers.pop(fault.name, None)
        if fault.transmissibility_multiplier is not None:
            updated_multipliers[fault.name] = fault.transmissibility_multiplier

    return attrs.evolve(
        grid,
        face_connection_types=connection_types,
        fault_face_indices=fault_face_indices or None,
        fault_transmissibility_multipliers=updated_multipliers or None,
    )


def remove_faults_from_grid(grid: Grid, names: frozenset[str]) -> Grid:
    """
    Produce a new `Grid` with the specified faults stripped out.

    Fault face classifications are reverted to plain `INTERIOR_FACE` or
    `BOUNDARY_FACE`. Transmissibility multipliers for the removed faults are dropped.

    :param grid: The source grid.
    :param names: Names of faults to remove.
    :returns: A new `Grid` with those faults absent.
    """
    assert grid.face_connection_types is not None
    connection_types = np.array(grid.face_connection_types, dtype=np.int8)

    if grid.fault_face_indices:
        face_cell_indices = np.ascontiguousarray(grid.face_cell_indices)
        for name in names:
            face_indices = grid.fault_face_indices.get(name)
            if face_indices is None:
                continue
            reclassify_faces(
                connection_types=typing.cast("IntArray[OneDimension]", connection_types),
                face_cell_indices=typing.cast("IntArray[TwoDimensions]", face_cell_indices),
                face_indices=typing.cast(
                    "IntArray[OneDimension]", np.ascontiguousarray(face_indices)
                ),
                interior_type=int(ConnectionType.INTERIOR_FACE),
                boundary_type=int(ConnectionType.BOUNDARY_FACE),
            )

    new_fault_face_indices = (
        {name: indices for name, indices in grid.fault_face_indices.items() if name not in names}
        if grid.fault_face_indices
        else None
    )
    new_multipliers = (
        {
            name: multiplier
            for name, multiplier in grid.fault_transmissibility_multipliers.items()
            if name not in names
        }
        if grid.fault_transmissibility_multipliers
        else None
    )
    return attrs.evolve(
        grid,
        face_connection_types=connection_types,
        fault_face_indices=new_fault_face_indices or None,
        fault_transmissibility_multipliers=new_multipliers or None,
    )
