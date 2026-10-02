"""Fault definition and application API for reservoir grids."""

import typing
import warnings

import attrs
import numpy as np
from typing_extensions import Self

from bores.deck.file import DeckFile
from bores.errors import ValidationError
from bores.grids.base import ConnectionType, Grid
from bores.grids.factories.base import VALID_FAULT_FACE_DIRECTIONS
from bores.serde.base import Serializable
from bores.types import Integer, Number

__all__ = ["Fault", "apply_faults", "remove_faults"]


@attrs.frozen(slots=True, unsafe_hash=True)
class Fault(Serializable):
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
        Must be one of `"I"`, `"I-"`, `"J"`, `"J-"`, `"K"`, `"K-"`.
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
      `transmissibility_multiplier` on each `Fault` (when not `None`).

    Faults already present on the grid with the same name are replaced. Faults
    with new names are merged in. To remove all existing faults first, call
    `remove_faults` before this function.

    :param model: The `Grid` to apply faults to.
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
    return _apply_faults_to_grid(grid, faults, nx, ny, nz)


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
    return _remove_faults_from_grid(grid, names_to_remove)


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


def _ijk_to_flat(i: Integer, j: Integer, k: Integer, nx: Integer, ny: Integer) -> int:
    """
    Convert 1-based IJK to a 0-based flat cell index.

    Uses Eclipse ordering: I varies fastest, K slowest.
    `cell_idx = (i-1) + (j-1)*nx + (k-1)*nx*ny`.

    :param i: 1-based I index.
    :param j: 1-based J index.
    :param k: 1-based K index.
    :param nx: Grid extent in x.
    :param ny: Grid extent in y.
    :returns: 0-based flat cell index.
    """
    return (i - 1) + (j - 1) * nx + (k - 1) * nx * ny


def resolve_fault_cell_pairs(
    fault: Fault, nx: Integer, ny: Integer, nz: Integer
) -> list[tuple[int, int]]:
    """
    Enumerate all cell pairs across a fault interface.

    Iterates the IJK box defined by the fault record and, for each cell in the
    box, finds its neighbour in the face direction. Returns pairs as
    `(owner_flat, neighbour_flat)` in ascending order.

    Pairs where either cell is outside the grid bounds are skipped with a
    warning.

    :param fault: The fault record to resolve.
    :param nx: Grid extent in x.
    :param ny: Grid extent in y.
    :param nz: Grid extent in z.
    :returns: List of `(cell_a_flat, cell_b_flat)` pairs.
    """
    direction = fault.face_direction
    pairs: list[tuple[int, int]] = []

    for k in range(fault.k1, fault.k2 + 1):
        for j in range(fault.j1, fault.j2 + 1):
            for i in range(fault.i1, fault.i2 + 1):
                if direction in ("I", "I-"):
                    ni, nj, nk = (i + 1, j, k) if direction == "I" else (i - 1, j, k)
                elif direction in ("J", "J-"):
                    ni, nj, nk = (i, j + 1, k) if direction == "J" else (i, j - 1, k)
                else:  # K or K-
                    ni, nj, nk = (i, j, k + 1) if direction == "K" else (i, j, k - 1)

                if not (1 <= ni <= nx and 1 <= nj <= ny and 1 <= nk <= nz):
                    warnings.warn(
                        f"Fault {fault.name!r}: neighbour cell ({ni},{nj},{nk}) is "
                        f"outside grid bounds ({nx}x{ny}x{nz}). Skipping pair.",
                        stacklevel=5,
                    )
                    continue

                cell_a = _ijk_to_flat(i, j, k, nx, ny)
                cell_b = _ijk_to_flat(ni, nj, nk, nx, ny)
                # Normalise so lower index is always first
                pairs.append((min(cell_a, cell_b), max(cell_a, cell_b)))
    return pairs


def _build_face_lookup(grid: Grid) -> dict[tuple[Integer, Integer], Integer]:
    """
    Build a `{(min_cell, max_cell): face_index}` lookup from `grid.face_cell_indices`.

    Only interior faces (where both owner and neighbour are >= 0) are included.
    This is used to determine whether a fault cell pair has a shared geometric
    face.

    :param grid: The source grid.
    :returns: Dict mapping sorted cell pairs to face indices.
    """
    lookup: dict[tuple[Integer, Integer], Integer] = {}
    for face_idx in range(grid.n_faces):
        owner = grid.face_cell_indices[face_idx, 0]
        neighbour = grid.face_cell_indices[face_idx, 1]
        if owner >= 0 and neighbour >= 0:
            lookup[min(owner, neighbour), max(owner, neighbour)] = face_idx
    return lookup


def _apply_faults_to_grid(
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
    face_lookup = _build_face_lookup(grid)
    # Start from copies of existing fault data structures
    fault_face_indices = typing.cast(
        dict[str, list[Integer]],
        {name: list(indices) for name, indices in (grid.fault_face_indices or {}).items()},
    )

    names_being_replaced: set[str] = {fault.name for fault in faults}
    for name in names_being_replaced:
        fault_face_indices.pop(name, None)

    # New face_connection_types: start from grid's current array
    assert grid.face_connection_types is not None
    new_face_connection_types = np.array(grid.face_connection_types, dtype=np.int8)

    # Process each incoming fault
    for fault in faults:
        cell_pairs = resolve_fault_cell_pairs(fault, nx, ny, nz)
        face_list: list[Integer] = []

        for cell_a, cell_b in cell_pairs:
            face_idx = face_lookup.get((cell_a, cell_b))
            if face_idx is not None:
                face_list.append(face_idx)
                # Reclassify face
                owner = grid.face_cell_indices[face_idx, 0]
                neighbour = grid.face_cell_indices[face_idx, 1]
                is_boundary = owner < 0 or neighbour < 0
                new_face_connection_types[face_idx] = int(
                    ConnectionType.BOUNDARY_FAULT_FACE
                    if is_boundary
                    else ConnectionType.INTERIOR_FAULT_FACE
                )
        if face_list:
            fault_face_indices[fault.name] = face_list

    # Build updated fault_transmissibility_multipliers
    updated_multipliers: dict[str, Number] = dict(grid.fault_transmissibility_multipliers or {})
    for fault in faults:
        if fault.transmissibility_multiplier is not None:
            updated_multipliers[fault.name] = fault.transmissibility_multiplier
        else:
            # If the old multiplier existed and user supplied None, preserve old value
            pass

    merged_fault_face_indices = (
        {name: np.asarray(idxs, dtype=np.int32) for name, idxs in fault_face_indices.items()}
        if fault_face_indices
        else None
    )
    return attrs.evolve(
        grid,
        face_connection_types=new_face_connection_types,
        fault_face_indices=merged_fault_face_indices,
        fault_transmissibility_multipliers=updated_multipliers or None,
    )


def _remove_faults_from_grid(grid: Grid, names: frozenset[str]) -> Grid:
    """
    Produce a new `Grid` with the specified faults stripped out.

    Fault face classifications are reverted to plain `INTERIOR_FACE` or
    `BOUNDARY_FACE`. Transmissibility multipliers for the removed faults are dropped.

    :param grid: The source grid.
    :param names: Names of faults to remove.
    :returns: A new `Grid` with those faults absent.
    """
    assert grid.face_connection_types is not None
    new_face_connection_types = np.array(grid.face_connection_types, dtype=np.int8)

    # Revert face classifications for removed faults
    if grid.fault_face_indices:
        for name in names:
            face_indices = grid.fault_face_indices.get(name)
            if face_indices is None:
                continue
            for face_idx in face_indices:
                owner = grid.face_cell_indices[face_idx, 0]
                neighbour = grid.face_cell_indices[face_idx, 1]
                is_boundary = owner < 0 or neighbour < 0
                new_face_connection_types[face_idx] = int(
                    ConnectionType.BOUNDARY_FACE if is_boundary else ConnectionType.INTERIOR_FACE
                )

    # Updated fault maps (minus removed faults)
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
        face_connection_types=new_face_connection_types,
        fault_face_indices=new_fault_face_indices or None,
        fault_transmissibility_multipliers=new_multipliers or None,
    )
