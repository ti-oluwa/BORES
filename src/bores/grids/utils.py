"""Gridding utilities"""

import math
import typing
import warnings
from collections.abc import Mapping

import numba
import numpy as np

from bores.errors import ValidationError
from bores.grids.base import Grid
from bores.types import (
    Boolean,
    BooleanArray,
    CellArray,
    IntArray,
    Integer,
    NumberArray,
    OneDimension,
    Side,
    TwoDimensions,
)

__all__ = ["make_pyvista_grid", "resolve_boundary_faces_for_box"]


SIDE_ALIASES: dict[str, Side] = {
    "left": Side.WEST,
    "right": Side.EAST,
    "front": Side.SOUTH,
    "back": Side.NORTH,
    "up": Side.TOP,
    "down": Side.BOTTOM,
}

AXIS_SIDES: dict[int, tuple[Side, Side]] = {
    0: (Side.WEST, Side.EAST),
    1: (Side.SOUTH, Side.NORTH),
    2: (Side.TOP, Side.BOTTOM),
}

# Eclipse's own `I+`/`I-`/`J+`/`J-`/`K+`/`K-` face-direction convention (as
# used by `AQUANCON`), each mapped to (dominant outward-normal axis, sign) -
# axis 0=X/I, 1=Y/J, 2=Z/K, matching `AXIS_SIDES` above.
FACE_DIRECTION_AXIS_SIGN: dict[str, tuple[int, float]] = {
    "I+": (0, 1.0),
    "I-": (0, -1.0),
    "J+": (1, 1.0),
    "J-": (1, -1.0),
    "K+": (2, 1.0),
    "K-": (2, -1.0),
}


def resolve_side(side: Side | str) -> Side:
    """
    Resolve a `Side`, its `.value` string, or a common alias
    ('left'/'right'/'front'/'back'/'up'/'down') to a `Side` member.

    :param side: `Side` member, value string, or alias.
    :returns: Resolved `Side`.
    :raises ValidationError: If `side` doesn't resolve to a known `Side`.
    """
    if isinstance(side, Side):
        return side

    key = side.strip().lower()
    if key in SIDE_ALIASES:
        return SIDE_ALIASES[key]
    try:
        return Side(key)
    except ValueError as exc:
        valid = sorted({member.value for member in Side} | set(SIDE_ALIASES))
        raise ValidationError(f"Unknown side {side!r}. Valid values: {valid}.") from exc


def classify_boundary_faces(grid: Grid) -> dict[Side, IntArray[OneDimension]]:
    """
    Classify every boundary face of `grid` into one of the six `Side`s.

    For each boundary face, its outward normal is re-derived defensively as
    `sign(dot(face_unit_normals[face], face_centroid - owner_cell_centroid))`
    rather than trusting `Grid.face_unit_normals`'s stored sign directly.
    The axis with the largest-magnitude (re-oriented) normal component
    decides X/Y/Z; its sign decides which of the two `Side`s on that axis.

    Every boundary face is genuinely exterior by definition (its
    `face_cell_indices` neighbour is `-1`), so classification is purely by
    outward-normal direction - no bounding-box-extremity check. A face on
    an irregular flank (partway along a fault-bounded edge, or the inner
    corner of an L-shaped field) is still genuinely facing one direction
    and belongs on that side; requiring proximity to the domain's
    bounding-box corner would drop real boundary faces on any non-box-shaped
    field. Every boundary face is assigned to exactly one `Side` - nothing
    is dropped.

    Classification happens in whatever frame `grid`'s stored coordinates
    are actually in. For a grid built with the default
    `apply_map_axes=True` (corner-point/Cartesian factories), that's true
    map/geographic space, so `Side.WEST` means true west. For a grid built
    with `apply_map_axes=False`, it means the grid's own local axes -
    consistent with every other computation on such a grid.

    :param grid: Grid to classify. Must have resolved face geometry
        (`face_unit_normals`, `face_centroids`, `cell_centroids`) - true
        for any `Grid` built through a grid factory.
    :returns: Mapping from every `Side` to the *positions* (0-based, into
        `Grid.boundary_face_indices` - not global face indices) of the
        faces assigned to that side, as an `int32` array. A `Side` with no
        matching faces maps to an empty array rather than being omitted.
    :raises ValidationError: If `grid`'s face geometry hasn't been resolved.
    """
    if grid.cell_centroids is None:
        raise ValidationError(
            "`grid` has no `cell_centroids`; cannot re-derive outward face orientation."
        )

    boundary = grid.boundary_face_indices
    if len(boundary) == 0:
        return {side: np.empty(0, dtype=np.int32) for side in Side}

    owners = grid.face_cell_indices[boundary, 0]
    n_boundary = len(boundary)

    # Everything below works one coordinate column at a time, so no `(n_boundary, 3)` temporary
    # is ever built. The outward sign of each face comes from its stored normal against the
    # vector from its owner's centroid to its own centroid. It does not change which axis
    # dominates, only the sign along it.
    normal_columns = [grid.face_unit_normals[boundary, axis] for axis in range(3)]
    alignment = np.zeros(n_boundary, dtype=np.float64)
    for axis in range(3):
        outward_reference = grid.face_centroids[boundary, axis] - grid.cell_centroids[owners, axis]
        outward_reference *= normal_columns[axis]
        alignment += outward_reference
    flip = np.sign(alignment)
    flip[flip == 0] = 1.0
    del alignment

    magnitudes = [np.abs(column) for column in normal_columns]
    dominant_axis = np.zeros(n_boundary, dtype=np.int8)
    dominant_magnitude = magnitudes[0]
    for axis in (1, 2):
        wins = magnitudes[axis] > dominant_magnitude
        dominant_axis[wins] = axis
        dominant_magnitude = np.where(wins, magnitudes[axis], dominant_magnitude)
    del magnitudes, dominant_magnitude

    dominant_normal = np.choose(dominant_axis, normal_columns)
    dominant_sign = np.sign(dominant_normal) * flip
    # A stored/rederived unit normal should never be exactly zero on its own
    # dominant axis. We guard against it anyway rather than silently dropping.
    dominant_sign[dominant_sign == 0] = 1.0

    positions = np.arange(n_boundary, dtype=np.int32)
    result: dict[Side, IntArray[OneDimension]] = {}
    for axis, (negative_side, positive_side) in AXIS_SIDES.items():
        on_axis = dominant_axis == axis
        result[negative_side] = positions[on_axis & (dominant_sign < 0.0)]  # type: ignore[assignment]
        result[positive_side] = positions[on_axis & (dominant_sign > 0.0)]  # type: ignore[assignment]
    return result


def resolve_boundary_faces_for_box(
    grid: Grid,
    *,
    i1: Integer,
    i2: Integer,
    j1: Integer,
    j2: Integer,
    k1: Integer,
    k2: Integer,
    face_direction: str,
    label: str = "box",
) -> IntArray[OneDimension]:
    """
    Resolve an Eclipse-style IJK box + face direction (as `AQUANCON` gives
    it) to boundary face positions (into `Grid.boundary_face_indices`).

    For each cell in the 1-based, inclusive box, it finds the face in
    `face_direction` by dominant outward-normal axis/sign (same convention
    `classify_boundary_faces` uses) via `Grid.get_face_normal_for_cell`,
    and includes it only when that face is a genuine grid-boundary face, i.e,
    the domain's true edge, not just the edge of an active region within a
    larger box. A cell is skipped, with a warning naming it, when:

    - it is outside `grid`'s bounds,
    - it is inactive,
    - its face in `face_direction` is an interior face. Even one
      adjoining an inactive neighbour. `BoundaryRegion` can only carry
      genuine `Grid.boundary_face_indices` positions, so a boundary
      condition can't be attached to an interior active/inactive junction
      through this function; that needs a different mechanism.

    When a cell has more than one face sharing `face_direction`'s
    dominant axis/sign (possible on irregular corner-point geometry), the
    first one found is used.

    :param grid: Grid to resolve against. Must have `dimensions` set.
    :param i1: 1-based start I index.
    :param i2: 1-based end I index (inclusive).
    :param j1: 1-based start J index.
    :param j2: 1-based end J index (inclusive).
    :param k1: 1-based start K index.
    :param k2: 1-based end K index (inclusive).
    :param face_direction: One of `'I+'`, `'I-'`, `'J+'`, `'J-'`, `'K+'`, `'K-'`.
    :param label: Identifies the source record in warning messages.
    :returns: Sorted, de-duplicated `int32` array of boundary face positions.
    :raises ValidationError: If `face_direction` is invalid, or `grid` has
        no resolvable `dimensions`.
    """
    if face_direction not in FACE_DIRECTION_AXIS_SIGN:
        raise ValidationError(
            f"`face_direction` must be one of {sorted(FACE_DIRECTION_AXIS_SIGN)}; "
            f"got {face_direction!r}."
        )
    axis, sign = FACE_DIRECTION_AXIS_SIGN[face_direction]

    dims = grid.dimensions
    if dims is None:
        raise ValidationError(f"{label}: `grid` has no `dimensions`; cannot resolve an IJK box.")
    nx, ny, nz = dims

    # Every cell of the box, in the order the records list them: `i` fastest, `k` slowest.
    # Decoded from one running index instead of building three full meshgrids.
    n_i = max(int(i2) - int(i1) + 1, 0)
    n_j = max(int(j2) - int(j1) + 1, 0)
    n_k = max(int(k2) - int(k1) + 1, 0)
    n_box = n_i * n_j * n_k
    running = np.arange(n_box, dtype=np.int64)
    box_i = running % n_i + i1 if n_i else running
    box_j = (running // n_i) % n_j + j1 if n_i and n_j else running
    box_k = running // (n_i * n_j) + k1 if n_i and n_j else running
    del running

    # Why a cell is skipped, per cell of the box. `None` means it is used.
    skip_reason: list[str | None] = [None] * n_box

    inside = (
        (box_i >= 1) & (box_i <= nx) & (box_j >= 1) & (box_j <= ny) & (box_k >= 1) & (box_k <= nz)
    )
    inside_rows = np.flatnonzero(inside)
    for row in np.flatnonzero(~inside):
        skip_reason[row] = (
            f"{label}: cell ({box_i[row]},{box_j[row]},{box_k[row]}) is outside grid bounds "
            f"({nx}x{ny}x{nz}). Skipping."
        )

    cells = grid.flat_indices(
        box_i[inside_rows] - 1, box_j[inside_rows] - 1, box_k[inside_rows] - 1
    )
    active = grid.are_cells_active(cells)
    for row in inside_rows[~active]:
        skip_reason[row] = (
            f"{label}: cell ({box_i[row]},{box_j[row]},{box_k[row]}) is inactive. "
            "Skipping connection."
        )
    candidate_rows = inside_rows[active]
    candidate_cells = cells[active]

    # The first face of each cell whose outward normal points mostly along `axis` in the
    # direction of `sign`. All the cells' faces are examined in one pass.
    faces_of_cells = grid.get_cells_face_indices(candidate_cells)
    faces_per_cell = np.diff(faces_of_cells.offsets)
    face_owner_position = np.repeat(np.arange(len(candidate_cells)), faces_per_cell)
    normals = grid.get_face_normals_for_cells(
        faces_of_cells.values,  # type: ignore[arg-type]
        candidate_cells[face_owner_position],
    )
    dominant_axis = np.argmax(np.abs(normals), axis=1)
    dominant_sign = np.sign(normals[np.arange(len(normals)), dominant_axis])
    faces_match = (dominant_axis == axis) & (dominant_sign == sign)
    matching = np.flatnonzero(faces_match)
    cells_with_a_match, first_match = np.unique(face_owner_position[matching], return_index=True)
    matched_face = np.full(len(candidate_cells), -1, dtype=np.int64)
    matched_face[cells_with_a_match] = faces_of_cells.values[matching[first_match]]

    boundary_position_of_face = np.full(grid.n_faces, -1, dtype=np.int64)
    boundary_position_of_face[grid.boundary_face_indices] = np.arange(
        len(grid.boundary_face_indices)
    )
    position = np.where(matched_face >= 0, boundary_position_of_face[matched_face], -1)

    for row in candidate_rows[matched_face < 0]:
        skip_reason[row] = (
            f"{label}: cell ({box_i[row]},{box_j[row]},{box_k[row]}) has no "
            f"{face_direction!r} face. Skipping."
        )
    for row in candidate_rows[(matched_face >= 0) & (position < 0)]:
        skip_reason[row] = (
            f"{label}: cell ({box_i[row]},{box_j[row]},{box_k[row]})'s {face_direction!r} "
            "face is not a grid-boundary face. It has a real interior neighbour, active or "
            "not. Boundary conditions can only attach to genuine grid-edge faces. Skipping."
        )

    # Warn in the order the cells are listed, as one cell at a time would.
    for reason in skip_reason:
        if reason is not None:
            warnings.warn(reason, stacklevel=2)

    return typing.cast(IntArray[OneDimension], np.unique(position[position >= 0]).astype(np.int32))


def classify_boundary_cells(grid: Grid) -> dict[Side, IntArray[OneDimension]]:
    """
    Classify every boundary-adjacent cell of `grid` into one of the six
    `Side`s, derived from `classify_boundary_faces`.

    A cell is included on a `Side` if it owns at least one boundary face
    classified there. A corner cell genuinely touching two flanks (e.g.
    both the west and south edges) appears in both `Side`s' arrays - that's
    correct, not a bug: it really is exposed on both.

    :param grid: Grid to classify.
    :returns: Mapping from every `Side` to the *global cell indices*
        (into `grid`'s own cell numbering) of cells with at least one
        boundary face on that side, deduplicated and sorted (`np.unique`).
        A `Side` with no matching cells maps to an empty array.
    """
    faces_by_side = classify_boundary_faces(grid)
    boundary = grid.boundary_face_indices
    result: dict[Side, IntArray[OneDimension]] = {}
    for side, face_positions in faces_by_side.items():
        if len(face_positions) == 0:
            result[side] = np.empty(0, dtype=np.int32)
            continue
        global_face_indices = boundary[face_positions]
        owners = grid.face_cell_indices[global_face_indices, 0]
        result[side] = np.unique(owners).astype(np.int32)
    return result


def get_cells_on_side(
    grid: Grid,
    side: Side | str,
    *,
    classified: typing.Mapping[Side, IntArray[OneDimension]] | None = None,
) -> IntArray[OneDimension]:
    """
    Global cell indices of every boundary-adjacent cell on one flank of `grid`.

    ```python
    west_cells = get_cells_on_side(grid, "west")
    ```

    :param grid: Grid to classify. Ignored if `classified` is supplied.
    :param side: Which flank - a `Side` member, its `.value` string, or a
        common alias ('left'/'right'/'front'/'back'/'up'/'down').
    :param classified: Pre-computed `classify_boundary_cells(grid)` result.
        Pass this when calling for several sides on the same grid, to avoid
        re-classifying every boundary face once per side.
    :returns: Sorted, deduplicated global cell indices on that side. Empty
        array if the grid has no boundary cells on that flank.
    :raises ValidationError: If `side` doesn't resolve to a known `Side`.
    """
    resolved = resolve_side(side)
    cells = classified if classified is not None else classify_boundary_cells(grid)
    return cells[resolved]


VTK_HEXAHEDRON = 12
"""VTK cell type of an eight-corner hexahedron."""

VTK_POLYHEDRON = 42
"""VTK cell type of a polyhedron described by its faces."""

CLOSED_CELL_TOLERANCE = 1e-6
"""
Relative size of the summed outward face-area vectors of a cell, against the sum of its face
areas, below which the cell's faces count as closed.
"""


@numba.njit(cache=True)
def face_has_vertex(
    face: Integer,
    vertex: Integer,
    face_vertex_offsets: IntArray[OneDimension],
    face_vertex_indices: IntArray[OneDimension],
) -> Boolean:
    """Check whether a face has a vertex among its corners."""
    for slot in range(face_vertex_offsets[face], face_vertex_offsets[face + 1]):
        if face_vertex_indices[slot] == vertex:
            return True
    return False


@numba.njit(cache=True)
def get_side_top_vertices(
    first_slot: Integer,
    top: Integer,
    corner: Integer,
    next_corner: Integer,
    base0: Integer,
    base1: Integer,
    base2: Integer,
    base3: Integer,
    cell_face_indices: IntArray[OneDimension],
    face_vertex_offsets: IntArray[OneDimension],
    face_vertex_indices: IntArray[OneDimension],
) -> tuple[Boolean, Integer, Integer]:
    """
    Compute and return the two top corners of the side face that holds two neighbouring base corners.

    :param first_slot: Position of the cell's first face in `cell_face_indices`.
    :param top: The face opposite the base.
    :param corner: A base corner.
    :param next_corner: The base corner that follows `corner`.
    :param base0: First base corner.
    :param base1: Second base corner.
    :param base2: Third base corner.
    :param base3: Fourth base corner.
    :param cell_face_indices: The faces of every cell, one cell after another.
    :param face_vertex_offsets: Offsets into `face_vertex_indices` for each face.
    :param face_vertex_indices: The vertices of every face, one face after another.
    :returns: Whether exactly one such side face with two top corners was found, then those
        two corners.
    """
    for slot in range(first_slot + 1, first_slot + 6):
        face = cell_face_indices[slot]
        if face == top:
            continue
        if not face_has_vertex(
            face=face,
            vertex=corner,
            face_vertex_offsets=face_vertex_offsets,
            face_vertex_indices=face_vertex_indices,
        ):
            continue
        if not face_has_vertex(
            face=face,
            vertex=next_corner,
            face_vertex_offsets=face_vertex_offsets,
            face_vertex_indices=face_vertex_indices,
        ):
            continue
        first = np.int64(-1)
        second = np.int64(-1)
        count = 0
        start = face_vertex_offsets[face]
        for position in range(4):
            vertex = np.int64(face_vertex_indices[start + position])
            if vertex != base0 and vertex != base1 and vertex != base2 and vertex != base3:
                if count == 0:
                    first = vertex
                else:
                    second = vertex
                count += 1
        if count == 2:
            return True, first, second
        return False, np.int64(-1), np.int64(-1)
    return False, np.int64(-1), np.int64(-1)


@numba.njit(cache=True)
def get_common_vertex(
    first_a: Integer, first_b: Integer, second_a: Integer, second_b: Integer
) -> tuple[Boolean, Integer]:
    """The one vertex that two pairs of vertices have in common, if there is exactly one."""
    found = np.int64(-1)
    matches = 0
    if first_a == second_a or first_a == second_b:
        found = np.int64(first_a)
        matches += 1
    if first_b == second_a or first_b == second_b:
        found = np.int64(first_b)
        matches += 1
    return matches == 1, found


@numba.njit(cache=True)
def compute_hexahedron_corners(
    cell: Integer,
    cell_face_offsets: IntArray[OneDimension],
    cell_face_indices: IntArray[OneDimension],
    face_vertex_offsets: IntArray[OneDimension],
    face_vertex_indices: IntArray[OneDimension],
    vertex_coordinates: NumberArray[TwoDimensions],
) -> tuple[Boolean, Integer, Integer, Integer, Integer, Integer, Integer, Integer, Integer]:
    """
    Compute and return the eight corners of a cell that is a hexahedron, in VTK hexahedron order.

    A cell is a hexahedron if it has six faces, each with four vertices, and the faces pair up
    into a base, a top that shares no vertex with it, and four sides. The base corners come
    first and the top corner above each of them follows in the same order, wound so that the base
    normal points towards the top as VTK requires.

    :param cell: Index of the cell.
    :param cell_face_offsets: Offsets into `cell_face_indices` for each cell.
    :param cell_face_indices: The faces of every cell, one cell after another.
    :param face_vertex_offsets: Offsets into `face_vertex_indices` for each face.
    :param face_vertex_indices: The vertices of every face, one face after another.
    :param vertex_coordinates: Shape `(n_vertices, 3)` vertex positions.
    :returns: Whether the cell is a hexahedron, then its eight corner vertex indices (all `-1` if
        it is not).
    """
    minus = np.int64(-1)
    none = (False, minus, minus, minus, minus, minus, minus, minus, minus)
    first_slot = cell_face_offsets[cell]
    if cell_face_offsets[cell + 1] - first_slot != 6:
        return none
    for slot in range(first_slot, first_slot + 6):
        face = cell_face_indices[slot]
        if face_vertex_offsets[face + 1] - face_vertex_offsets[face] != 4:
            return none

    base_face = cell_face_indices[first_slot]
    base_start = face_vertex_offsets[base_face]
    base0 = np.int64(face_vertex_indices[base_start])
    base1 = np.int64(face_vertex_indices[base_start + 1])
    base2 = np.int64(face_vertex_indices[base_start + 2])
    base3 = np.int64(face_vertex_indices[base_start + 3])

    if (
        base0 == base1
        or base0 == base2
        or base0 == base3
        or base1 == base2
        or base1 == base3
        or base2 == base3
    ):
        return none

    # The top is the one other face with no vertex in common with the base.
    top = -1
    for slot in range(first_slot + 1, first_slot + 6):
        face = cell_face_indices[slot]
        shares = (
            face_has_vertex(
                face=face,
                vertex=base0,
                face_vertex_offsets=face_vertex_offsets,
                face_vertex_indices=face_vertex_indices,
            )
            or face_has_vertex(
                face=face,
                vertex=base1,
                face_vertex_offsets=face_vertex_offsets,
                face_vertex_indices=face_vertex_indices,
            )
            or face_has_vertex(
                face=face,
                vertex=base2,
                face_vertex_offsets=face_vertex_offsets,
                face_vertex_indices=face_vertex_indices,
            )
            or face_has_vertex(
                face=face,
                vertex=base3,
                face_vertex_offsets=face_vertex_offsets,
                face_vertex_indices=face_vertex_indices,
            )
        )
        if not shares:
            if top >= 0:
                return none
            top = face
    if top < 0:
        return none

    # The side holding base corners i and i + 1 also holds the two top corners above them. The top
    # corner above base corner i is the one that side i - 1 and side i have in common.
    ok0, s0a, s0b = get_side_top_vertices(
        first_slot=first_slot,
        top=top,
        corner=base0,
        next_corner=base1,
        base0=base0,
        base1=base1,
        base2=base2,
        base3=base3,
        cell_face_indices=cell_face_indices,
        face_vertex_offsets=face_vertex_offsets,
        face_vertex_indices=face_vertex_indices,
    )
    ok1, s1a, s1b = get_side_top_vertices(
        first_slot=first_slot,
        top=top,
        corner=base1,
        next_corner=base2,
        base0=base0,
        base1=base1,
        base2=base2,
        base3=base3,
        cell_face_indices=cell_face_indices,
        face_vertex_offsets=face_vertex_offsets,
        face_vertex_indices=face_vertex_indices,
    )
    ok2, s2a, s2b = get_side_top_vertices(
        first_slot=first_slot,
        top=top,
        corner=base2,
        next_corner=base3,
        base0=base0,
        base1=base1,
        base2=base2,
        base3=base3,
        cell_face_indices=cell_face_indices,
        face_vertex_offsets=face_vertex_offsets,
        face_vertex_indices=face_vertex_indices,
    )
    ok3, s3a, s3b = get_side_top_vertices(
        first_slot=first_slot,
        top=top,
        corner=base3,
        next_corner=base0,
        base0=base0,
        base1=base1,
        base2=base2,
        base3=base3,
        cell_face_indices=cell_face_indices,
        face_vertex_offsets=face_vertex_offsets,
        face_vertex_indices=face_vertex_indices,
    )
    if not (ok0 and ok1 and ok2 and ok3):
        return none
    found0, top0 = get_common_vertex(first_a=s3a, first_b=s3b, second_a=s0a, second_b=s0b)
    found1, top1 = get_common_vertex(first_a=s0a, first_b=s0b, second_a=s1a, second_b=s1b)
    found2, top2 = get_common_vertex(first_a=s1a, first_b=s1b, second_a=s2a, second_b=s2b)
    found3, top3 = get_common_vertex(first_a=s2a, first_b=s2b, second_a=s3a, second_b=s3b)
    if not (found0 and found1 and found2 and found3):
        return none
    if (
        top0 == top1
        or top0 == top2
        or top0 == top3
        or top1 == top2
        or top1 == top3
        or top2 == top3
    ):
        return none

    # The corners found through the sides must be the corners of the top face itself.
    for corner in (top0, top1, top2, top3):
        if not face_has_vertex(
            face=top,
            vertex=corner,
            face_vertex_offsets=face_vertex_offsets,
            face_vertex_indices=face_vertex_indices,
        ):
            return none

    # The base must wind towards the top: ((p1 - p0) x (p3 - p0)) . (top0 - p0) > 0.
    e1x = vertex_coordinates[base1, 0] - vertex_coordinates[base0, 0]
    e1y = vertex_coordinates[base1, 1] - vertex_coordinates[base0, 1]
    e1z = vertex_coordinates[base1, 2] - vertex_coordinates[base0, 2]
    e2x = vertex_coordinates[base3, 0] - vertex_coordinates[base0, 0]
    e2y = vertex_coordinates[base3, 1] - vertex_coordinates[base0, 1]
    e2z = vertex_coordinates[base3, 2] - vertex_coordinates[base0, 2]
    e3x = vertex_coordinates[top0, 0] - vertex_coordinates[base0, 0]
    e3y = vertex_coordinates[top0, 1] - vertex_coordinates[base0, 1]
    e3z = vertex_coordinates[top0, 2] - vertex_coordinates[base0, 2]
    volume = (
        (e1y * e2z - e1z * e2y) * e3x
        + (e1z * e2x - e1x * e2z) * e3y
        + (e1x * e2y - e1y * e2x) * e3z
    )
    if volume > 0.0:
        return (True, base0, base1, base2, base3, top0, top1, top2, top3)
    if volume < 0.0:
        return (True, base0, base3, base2, base1, top0, top3, top2, top1)
    return none


@numba.njit(parallel=True, cache=True)
def find_exposed_cells(
    cell_face_offsets: IntArray[OneDimension],
    cell_face_indices: IntArray[OneDimension],
    face_cell_indices: IntArray[TwoDimensions],
    face_unit_normals: NumberArray[TwoDimensions],
    face_areas: NumberArray[OneDimension],
    rendered: BooleanArray[OneDimension],
) -> BooleanArray[OneDimension]:
    """
    Find the cells that can be seen from outside the grid.

    A cell is exposed if one of its faces has no cell on the other side, if the cell on the other
    side is not rendered, or if its faces do not close up on their own. The last case covers a cell
    whose neighbour is inactive, since the face towards an inactive cell is removed from the grid.

    :param cell_face_offsets: Offsets into `cell_face_indices` for each cell.
    :param cell_face_indices: The faces of every cell, one cell after another.
    :param face_cell_indices: Shape `(n_faces, 2)` owner and neighbour of every face.
    :param face_unit_normals: Shape `(n_faces, 3)` unit normals pointing out of the owner.
    :param face_areas: Area of every face.
    :param rendered: Whether each cell is part of the mesh at all.
    :returns: Whether each cell is exposed. Cells that are not rendered are never exposed.
    """
    n_cells = cell_face_offsets.shape[0] - 1
    exposed = np.zeros(n_cells, dtype=np.bool_)
    for cell in numba.prange(n_cells):  # type: ignore[attr-defined]
        if not rendered[cell]:
            continue
        sum_x = 0.0
        sum_y = 0.0
        sum_z = 0.0
        total_area = 0.0
        is_exposed = False
        for slot in range(cell_face_offsets[cell], cell_face_offsets[cell + 1]):
            face = cell_face_indices[slot]
            owner = face_cell_indices[face, 0]
            neighbour = face_cell_indices[face, 1]
            if owner == cell:
                other = neighbour
                outward = 1.0
            else:
                other = owner
                outward = -1.0
            if other < 0 or not rendered[other]:
                is_exposed = True
                break
            area = face_areas[face]
            sum_x += outward * face_unit_normals[face, 0] * area
            sum_y += outward * face_unit_normals[face, 1] * area
            sum_z += outward * face_unit_normals[face, 2] * area
            total_area += area
        if not is_exposed:
            gap = math.sqrt(sum_x * sum_x + sum_y * sum_y + sum_z * sum_z)
            is_exposed = gap > CLOSED_CELL_TOLERANCE * total_area
        exposed[cell] = is_exposed
    return exposed


@numba.njit(parallel=True, cache=True)
def count_cell_entries(
    cell_face_offsets: IntArray[OneDimension],
    cell_face_indices: IntArray[OneDimension],
    face_vertex_offsets: IntArray[OneDimension],
    face_vertex_indices: IntArray[OneDimension],
    vertex_coordinates: NumberArray[TwoDimensions],
    render: BooleanArray[OneDimension],
    use_hexahedra: Boolean,
) -> IntArray[OneDimension]:
    """
    Count the entries each cell contributes to the VTK cell stream.

    A polyhedron cell is laid out as

        [total_count, n_faces, nv_f0, v0, v1, ..., nv_f1, v0, ...]

    so a cell with *F* faces of *V0, ..., V(F-1)* vertices takes `2 + F + sum(Vi)` entries. A
    hexahedron cell is `[8, v0, ..., v7]`, which takes 9. A cell that is not rendered, or has no
    faces (a pinched-out cell), takes none.

    :param cell_face_offsets: Offsets into `cell_face_indices` for each cell.
    :param cell_face_indices: The faces of every cell, one cell after another.
    :param face_vertex_offsets: Offsets into `face_vertex_indices` for each face.
    :param face_vertex_indices: The vertices of every face, one face after another.
    :param vertex_coordinates: Shape `(n_vertices, 3)` vertex positions.
    :param render: Whether each cell is to be included.
    :param use_hexahedra: Whether cells that are hexahedra use the compact hexahedron layout.
    :returns: Shape `(n_cells,)` int64 entry count of each cell.
    """
    n_cells = cell_face_offsets.shape[0] - 1
    counts = np.zeros(n_cells, dtype=np.int64)

    for cell in numba.prange(n_cells):  # type: ignore[attr-defined]
        if not render[cell]:
            continue
        face_start = cell_face_offsets[cell]
        face_end = cell_face_offsets[cell + 1]
        n_faces = face_end - face_start
        if n_faces == 0:
            continue
        if use_hexahedra:
            is_hexahedron, _, _, _, _, _, _, _, _ = compute_hexahedron_corners(
                cell=cell,
                cell_face_offsets=cell_face_offsets,
                cell_face_indices=cell_face_indices,
                face_vertex_offsets=face_vertex_offsets,
                face_vertex_indices=face_vertex_indices,
                vertex_coordinates=vertex_coordinates,
            )
            if is_hexahedron:
                counts[cell] = 9
                continue
        n_verts_total = np.int64(0)
        for face_slot in range(face_start, face_end):
            face = cell_face_indices[face_slot]
            n_verts_total += face_vertex_offsets[face + 1] - face_vertex_offsets[face]
        counts[cell] = np.int64(2) + np.int64(n_faces) + n_verts_total
    return counts


@numba.njit(parallel=True, cache=True)
def fill_cell_entries(
    cell_face_offsets: IntArray[OneDimension],
    cell_face_indices: IntArray[OneDimension],
    face_vertex_offsets: IntArray[OneDimension],
    face_vertex_indices: IntArray[OneDimension],
    vertex_coordinates: NumberArray[TwoDimensions],
    counts: IntArray[OneDimension],
    cell_starts: IntArray[OneDimension],
    use_hexahedra: Boolean,
    out: IntArray[OneDimension],
) -> None:
    """
    Fill the cell stream with the cells that are rendered.

    Each cell owns the slice of `out` that starts at its entry of `cell_starts` and is as long as
    its entry of `counts`, so cells write independently. A polyhedron is written as

        out[start]     = entries that follow
        out[start + 1] = n_faces
        then, for each face, its vertex count followed by its vertex indices

    and a hexahedron as `[8, v0, ..., v7]`.

    :param cell_face_offsets: Offsets into `cell_face_indices` for each cell.
    :param cell_face_indices: The faces of every cell, one cell after another.
    :param face_vertex_offsets: Offsets into `face_vertex_indices` for each face.
    :param face_vertex_indices: The vertices of every face, one face after another.
    :param vertex_coordinates: Shape `(n_vertices, 3)` vertex positions.
    :param counts: Entry count of each cell from `count_cell_entries`. Cells with `0` are skipped.
    :param cell_starts: Where each cell's entries start in `out`.
    :param use_hexahedra: Whether cells that are hexahedra use the compact hexahedron layout. Must
        match the value given to `count_cell_entries`.
    :param out: Output cell stream.
    """
    n_cells = cell_face_offsets.shape[0] - 1
    for cell in numba.prange(n_cells):  # type: ignore[attr-defined]
        if counts[cell] == 0:
            continue
        position = cell_starts[cell]
        if use_hexahedra:
            is_hexahedron, c0, c1, c2, c3, c4, c5, c6, c7 = compute_hexahedron_corners(
                cell=cell,
                cell_face_offsets=cell_face_offsets,
                cell_face_indices=cell_face_indices,
                face_vertex_offsets=face_vertex_offsets,
                face_vertex_indices=face_vertex_indices,
                vertex_coordinates=vertex_coordinates,
            )
            if is_hexahedron:
                out[position] = 8
                out[position + 1] = c0
                out[position + 2] = c1
                out[position + 3] = c2
                out[position + 4] = c3
                out[position + 5] = c4
                out[position + 6] = c5
                out[position + 7] = c6
                out[position + 8] = c7
                continue

        face_start = cell_face_offsets[cell]
        face_end = cell_face_offsets[cell + 1]
        first = position
        position += 1
        out[position] = face_end - face_start
        position += 1
        for face_slot in range(face_start, face_end):
            face = cell_face_indices[face_slot]
            vertex_start = face_vertex_offsets[face]
            vertex_end = face_vertex_offsets[face + 1]
            out[position] = vertex_end - vertex_start
            position += 1
            for vertex_slot in range(vertex_start, vertex_end):
                out[position] = face_vertex_indices[vertex_slot]
                position += 1
        out[first] = position - first - np.int64(1)


def make_pyvista_grid(
    grid: Grid,
    *,
    cell_data: Mapping[str, CellArray] | None = None,
    hollow: bool = False,
    use_hexahedra: bool = False,
) -> typing.Any:
    """
    Convert a `bores.grids.base.Grid` to a `pyvista.UnstructuredGrid`.

    Each grid cell is represented as a **VTK polyhedron (type 42)** built
    from the cell's actual face and vertex geometry. This is geometrically
    exact for all grid types; corner-point, Voronoi, Cartesian, and general
    polyhedral.

    Pinched-out cells (cells whose faces were all suppressed during
    corner-point construction) have no VTK representation and are silently
    omitted from the output mesh. Their data is also omitted from any
    attached cell-data arrays so that array lengths always match the number
    of rendered cells. The cells remain present in the source `Grid`
    object so that physics code is unaffected.

    Two options make large models cheaper to build and draw. Both are off by default, and with both
    off the result is exactly the polyhedron mesh described above.

    - `use_hexahedra` stores every cell that is a hexahedron (six four-vertex faces) as a VTK
      hexahedron (type 12) instead of a polyhedron. A hexahedron takes 9 entries in the cell
      stream against 32 for the same cell as a polyhedron, and VTK draws and clips it faster. Cells
      of any other shape stay polyhedra. VTK draws the faces of a hexahedron as smooth surfaces
      between its corners, so a strongly warped cell, whose faces are not flat, differs slightly in
      shape and computed volume from its polyhedron. Cells with flat faces are identical.
    - `hollow` leaves out every cell that cannot be seen from outside the grid, meaning
      cells whose faces all have an active, rendered neighbour and close up on their own. The
      outside of the model looks the same, but cutting into it (clipping, slicing or thresholding)
      reveals a hollow model.

    **`MapAxes`**:

    No rotation happens here. `grid.vertex_coordinates` is already in map
    space by the time it reaches this function as the corner-point and
    Cartesian grid factories apply `MAPAXES` when `grid.metadata["map_axes"]`
    is present, unless they were explicitly built with `apply_map_axes=False`,
    in which case this renders the same local-space geometry every other
    computation on the grid uses.

    :param grid: Source `Grid`.
    :param cell_data: Optional mapping of scalar field name to a shape
        `(n_cells,)` NumPy array. Each entry is attached as a PyVista
        cell-data array and can be visualised with
        `pv_grid.plot(scalars="<scalar>")`. Arrays must have length
        `grid.n_cells`; they are automatically filtered to the cells that are rendered
        before attachment.
    :param hollow: Whether to leave out the cells that cannot be seen from outside the grid.
    :param use_hexahedra: Whether to store hexahedral cells in VTK's compact hexahedron form.
    :returns: A `pyvista.UnstructuredGrid` ready for rendering or
        further PyVista processing.
    :raises ImportError: If `pyvista` is not installed.
    :raises ValueError: If a `cell_data` array has length != `grid.n_cells`.

    Example:

    ```python
    from bores.grids.utils import make_pyvista_grid
    import pyvista as pv

    pressure = np.zeros((n_cells,))
    pv_grid = make_pyvista_grid(
        grid,
        cell_data={"pressure": pressure},
        hollow=True,
        use_hexahedra=True,
    )

    pl = pv.Plotter()
    pl.add_mesh(pv_grid, scalars="pressure", show_edges=True)
    pl.show()
    ```
    """
    try:
        import pyvista as pv  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "The 'pyvista' library is required for PyVista conversion. "
            "Install it with: pip install pyvista"
        ) from exc

    n_cells = grid.n_cells
    assert grid.cell_volumes is not None

    # Copy as this function flips Z in-place below for PyVista's up
    # convention, which must not mutate `grid.vertex_coordinates` itself.
    # Data uses Z positive downward (depth) convention but PyVista uses Z positive upward
    # (3D graphics or coordinate system) convention. The flip comes before the cell stream is
    # built so a hexahedron is wound for the coordinates PyVista will actually see.
    all_points = grid.vertex_coordinates.copy()
    all_points[:, 2] *= -1

    # Cells with faces are the ones that can be drawn at all.
    render = typing.cast(BooleanArray[OneDimension], np.diff(grid.cell_face_offsets) > 0)
    if hollow:
        render = find_exposed_cells(
            cell_face_offsets=grid.cell_face_offsets,
            cell_face_indices=grid.cell_face_indices,
            face_cell_indices=grid.face_cell_indices,
            face_unit_normals=grid.face_unit_normals,
            face_areas=grid.face_areas,
            rendered=render,
        )

    counts = count_cell_entries(
        cell_face_offsets=grid.cell_face_offsets,
        cell_face_indices=grid.cell_face_indices,
        face_vertex_offsets=grid.face_vertex_offsets,
        face_vertex_indices=grid.face_vertex_indices,
        vertex_coordinates=all_points,
        render=render,
        use_hexahedra=use_hexahedra,
    )
    valid_mask = counts > 0

    # Where each cell's entries start: the running total of the entries before it. A cell with no
    # entries adds nothing, so unrendered cells do not move the start of the ones after them.
    cell_starts = np.cumsum(counts)
    cell_starts -= counts
    total_entries = int(counts.sum())

    flat_cells = np.empty(total_entries, dtype=np.int64)
    fill_cell_entries(
        cell_face_offsets=grid.cell_face_offsets,
        cell_face_indices=grid.cell_face_indices,
        face_vertex_offsets=grid.face_vertex_offsets,
        face_vertex_indices=grid.face_vertex_indices,
        vertex_coordinates=all_points,
        counts=counts,
        cell_starts=typing.cast(IntArray[OneDimension], cell_starts),
        use_hexahedra=use_hexahedra,
        out=flat_cells,
    )

    # A hexahedron is the only cell that takes 9 entries, since the smallest polyhedron, a
    # tetrahedron, takes 18.
    rendered_counts = counts[valid_mask]
    cell_types = np.where(rendered_counts == 9, VTK_HEXAHEDRON, VTK_POLYHEDRON).astype(np.uint8)
    del rendered_counts, cell_starts, counts
    pv_grid = pv.UnstructuredGrid(flat_cells, cell_types, all_points)

    # Attach built-in geometric arrays (filtered to rendered cells)
    pv_grid.cell_data["cell_volume"] = grid.cell_volumes[valid_mask]
    pv_grid.cell_data["cell_depth"] = grid.cell_center_depths[valid_mask]
    pv_grid.cell_data["cell_thickness"] = grid.cell_thickness[valid_mask]

    # Attach caller-supplied arrays
    if cell_data:
        for name, array in cell_data.items():
            array = np.asarray(array)
            if array.shape[0] != n_cells:
                raise ValueError(
                    f"cell_data[{name!r}] has {array.shape[0]} entries but grid has {n_cells} cells."
                )
            pv_grid.cell_data[name] = array[valid_mask]
    return pv_grid
