import itertools
import typing
from typing import TypeAlias

import numba
import numpy as np

from bores.datastructures import MapAxes
from bores.errors import InvalidFaceConnectivityError, ValidationError
from bores.types import IntArray, Integer, NumberArray, OneDimension, TwoDimensions

VertexCoordinates: TypeAlias = NumberArray[TwoDimensions]
"""Shape `(n_points, 3)` - 3-D (x, y, z) vertex coordinates."""

FaceVertexIndices: TypeAlias = list[Integer]
"""Ordered list of vertex indices for a single face (CCW from owner)."""

FaceKey: TypeAlias = tuple[Integer, ...]
"""Sorted tuple of vertex indices used as a face deduplication key."""

ElementFaces: TypeAlias = list[FaceVertexIndices]
"""
Per-element-type local face definitions; each entry is a list of local
vertex indices wound CCW from outside (outward normal).
"""


# Element face tables (VTK convention)

#: Outward-pointing face definitions for standard element types.
#: Each value is a list of faces; each face is a list of *local* vertex
#: indices wound counter-clockwise when viewed from outside the cell.
ELEMENTS_FACES: dict[str, ElementFaces] = {
    # #####################################################################
    # Tetrahedron (4 vertices: v0 v1 v2 = base CCW from below, v3 = apex)
    # #####################################################################
    "tetra": [
        [0, 2, 1],  # base - outward normal points downward (away from apex)
        [0, 1, 3],  # front face opposite v2
        [1, 2, 3],  # right face opposite v0
        [0, 3, 2],  # left face opposite v1
    ],
    # #####################################################################
    # Hexahedron / brick (8 vertices, VTK ordering)
    # Bottom face: v0 v1 v2 v3 (z = z_bottom); top face: v4 v5 v6 v7 (z = z_top)
    # v0=(x0,y0,z_bottom) v1=(x1,y0,z_bottom) v2=(x1,y1,z_bottom) v3=(x0,y1,z_bottom)
    # v4=(x0,y0,z_top) v5=(x1,y0,z_top) v6=(x1,y1,z_top) v7=(x0,y1,z_top)
    # #####################################################################
    "hexahedron": [
        [0, 3, 2, 1],  # bottom  - outward = −z
        [4, 5, 6, 7],  # top     - outward = +z
        [0, 1, 5, 4],  # front   - outward = −y
        [2, 3, 7, 6],  # back    - outward = +y
        [0, 4, 7, 3],  # left    - outward = −x
        [1, 2, 6, 5],  # right   - outward = +x
    ],
    # #####################################################################
    # Wedge / triangular prism (6 vertices)
    # Bottom triangle: v0 v1 v2 (CCW from below); top: v3 v4 v5
    # #####################################################################
    "wedge": [
        [0, 2, 1],  # bottom triangle - outward = −z
        [3, 4, 5],  # top triangle    - outward = +z
        [0, 1, 4, 3],  # lateral face 01 - outward away from opposite edge
        [1, 2, 5, 4],  # lateral face 12
        [0, 3, 5, 2],  # lateral face 20
    ],
    # #####################################################################
    # Pyramid (5 vertices: v0 v1 v2 v3 = base CCW from below, v4 = apex)
    # #####################################################################
    "pyramid": [
        [0, 3, 2, 1],  # base   - outward = −z
        [0, 1, 4],  # front  face
        [1, 2, 4],  # right  face
        [2, 3, 4],  # back   face
        [0, 4, 3],  # left   face
    ],
}

#: Mapping from VTK integer cell-type codes to element type names.
VTK_CELL_TYPE_NAMES: dict[int, str] = {
    10: "tetra",
    12: "hexahedron",
    13: "wedge",
    14: "pyramid",
}


class FaceRecord:
    """Mutable record accumulating owner/neighbour information for one face."""

    __slots__ = ("face_vertex_indices", "neighbour_cell_index", "owner_cell_index")

    def __init__(self, owner_cell_index: Integer, face_vertex_indices: FaceVertexIndices) -> None:
        """
        Initialise with only an owner; neighbour is set later if face is interior.

        :param owner_cell_index: The cell that first registered this face.
        :param face_vertex_indices: Face vertex indices winding from the owner's perspective.
        """
        self.owner_cell_index: Integer = owner_cell_index
        self.neighbour_cell_index: Integer = -1
        self.face_vertex_indices: FaceVertexIndices = face_vertex_indices


@numba.njit(cache=True)
def deduplicate_faces(
    entry_cells: IntArray[OneDimension],
    entry_offsets: IntArray[OneDimension],
    entry_vertices: IntArray[OneDimension],
) -> tuple[
    Integer,
    Integer,
    Integer,
    IntArray[OneDimension],
    IntArray[OneDimension],
    IntArray[OneDimension],
]:
    """
    Match the face entries of every cell so a face shared by two cells is stored once.

    An entry is one face of one cell, as the cell sees it: its vertices wound outward from that
    cell. Entries must be ordered by cell, and each cell's entries in the order its faces were
    listed. Two entries are the same face when they have the same vertices, in any order. The
    first cell to list a face owns it and keeps its winding, the second becomes the neighbour.
    Faces are numbered in the order they were first listed.

    :param entry_cells: Shape `(n_entries,)` cell of each entry.
    :param entry_offsets: Shape `(n_entries + 1,)` offsets into `entry_vertices`.
    :param entry_vertices: Vertex indices of every entry, one entry after another.
    :returns: `(status, bad_entry, bad_face, face_entries, owners, neighbours)`. `face_entries[f]`
        is the entry that first listed face `f`, with the face's owner and neighbour (`-1` if
        none). `status` is `0` on success, and `1` when a face turned up in a third cell, in which
        case `bad_entry` is that entry, `bad_face` the face it matched, and the arrays hold the
        faces found before it.
    """
    n_entries = entry_cells.shape[0]
    sorted_vertices = entry_vertices.copy()
    for entry in range(n_entries):
        begin = entry_offsets[entry]
        end = entry_offsets[entry + 1]
        if end - begin <= 32:
            for position in range(begin + 1, end):
                value = sorted_vertices[position]
                hole = position - 1
                while hole >= begin and sorted_vertices[hole] > value:
                    sorted_vertices[hole + 1] = sorted_vertices[hole]
                    hole -= 1
                sorted_vertices[hole + 1] = value
        else:
            sorted_vertices[begin:end].sort()

    capacity = 16
    while capacity < 2 * n_entries:
        capacity *= 2
    mask = capacity - 1
    table = np.full(capacity, -1, dtype=np.int64)
    face_entries = np.empty(n_entries, dtype=np.int64)
    owners = np.empty(n_entries, dtype=np.int64)
    neighbours = np.empty(n_entries, dtype=np.int64)
    n_faces = 0

    for entry in range(n_entries):
        begin = entry_offsets[entry]
        end = entry_offsets[entry + 1]
        h = np.int64(1469598103934665603) ^ np.int64(end - begin)
        for position in range(begin, end):
            h = (h ^ sorted_vertices[position]) * np.int64(1099511628211)
        h = h ^ (h >> 29)
        slot = h & mask
        while True:
            face = table[slot]
            if face < 0:
                table[slot] = n_faces
                face_entries[n_faces] = entry
                owners[n_faces] = entry_cells[entry]
                neighbours[n_faces] = -1
                n_faces += 1
                break
            other = face_entries[face]
            other_begin = entry_offsets[other]
            if entry_offsets[other + 1] - other_begin == end - begin:
                same = True
                for offset in range(end - begin):
                    if sorted_vertices[other_begin + offset] != sorted_vertices[begin + offset]:
                        same = False
                        break
                if same:
                    if neighbours[face] != -1:
                        return (  # type: ignore[return-value]
                            1,
                            entry,
                            face,
                            face_entries[:n_faces],
                            owners[:n_faces],
                            neighbours[:n_faces],
                        )
                    neighbours[face] = entry_cells[entry]
                    break
            slot = (slot + 1) & mask

    return 0, 0, 0, face_entries[:n_faces], owners[:n_faces], neighbours[:n_faces]  # type: ignore[return-value]


def build_csr_face_arrays_from_entries(
    vertex_coordinates: VertexCoordinates,
    entry_cells: IntArray[OneDimension],
    entry_offsets: IntArray[OneDimension],
    entry_vertices: IntArray[OneDimension],
) -> tuple[
    VertexCoordinates,
    IntArray[OneDimension],
    IntArray[OneDimension],
    IntArray[TwoDimensions],
]:
    """
    Deduplicate the faces of every cell and build CSR face arrays, from flat arrays.

    The array form of `build_csr_face_arrays`; see `deduplicate_faces` for what an entry is and
    how faces are matched and numbered.

    :param vertex_coordinates: Shape `(n_vertices, 3)` coordinate array.
    :param entry_cells: Shape `(n_entries,)` cell of each face entry, in ascending cell order.
    :param entry_offsets: Shape `(n_entries + 1,)` offsets into `entry_vertices`.
    :param entry_vertices: Vertex indices of every entry, one entry after another.
    :returns: Tuple `(vertex_coordinates, face_vertex_indices, face_vertex_offsets,
        face_cell_indices)` ready to be passed to `bores.grids.base.Grid`.
    :raises InvalidFaceConnectivityError: If any face is shared by more than two cells.
    """
    entry_cells = typing.cast(
        "IntArray[OneDimension]", np.ascontiguousarray(entry_cells, dtype=np.int64)
    )
    entry_offsets = typing.cast(
        "IntArray[OneDimension]", np.ascontiguousarray(entry_offsets, dtype=np.int64)
    )
    entry_vertices = typing.cast(
        "IntArray[OneDimension]", np.ascontiguousarray(entry_vertices, dtype=np.int64)
    )
    status, bad_entry, bad_face, face_entries, owners, neighbours = deduplicate_faces(
        entry_cells=entry_cells,
        entry_offsets=entry_offsets,
        entry_vertices=entry_vertices,
    )
    if status != 0:
        key = tuple(
            sorted(
                int(v)
                for v in entry_vertices[entry_offsets[bad_entry] : entry_offsets[bad_entry + 1]]
            )
        )
        raise InvalidFaceConnectivityError(
            f"Face with canonical key {key} is shared by "
            f"more than two cells (already has owner "
            f"{int(owners[bad_face])} and neighbour "
            f"{int(neighbours[bad_face])}; tried to add cell {int(entry_cells[bad_entry])})."
        )

    lengths = entry_offsets[face_entries + 1] - entry_offsets[face_entries]
    face_offsets = np.concatenate([[0], np.cumsum(lengths)])
    total = int(face_offsets[-1])
    source = np.repeat(entry_offsets[face_entries] - face_offsets[:-1], lengths) + np.arange(
        total, dtype=np.int64
    )
    return (
        vertex_coordinates,
        typing.cast("IntArray[OneDimension]", entry_vertices[source].astype(np.int32)),
        typing.cast("IntArray[OneDimension]", face_offsets.astype(np.int32)),
        typing.cast(
            "IntArray[TwoDimensions]", np.column_stack([owners, neighbours]).astype(np.int32)
        ),
    )


def build_csr_face_arrays(
    vertex_coordinates: VertexCoordinates,
    per_cell_face_vertex_lists: list[list[FaceVertexIndices]],
) -> tuple[
    VertexCoordinates,
    IntArray[OneDimension],
    IntArray[OneDimension],
    IntArray[TwoDimensions],
]:
    """
    Deduplicate faces across cells and build CSR face arrays.

    Each cell provides a list of faces; each face is a list of vertex
    indices wound CCW from **outside** that cell (outward-normal convention).
    Interior faces (shared by exactly two cells) appear once in each cell's
    list; the first cell to register a canonical key becomes the owner.

    :param vertex_coordinates: Shape `(n_vertices, 3)` coordinate array.
    :param per_cell_face_vertex_lists: Outer list indexed by cell index;
        each element is a list of face vertex lists for that cell.
    :returns: Tuple `(vertex_coordinates, face_vertex_indices,
        face_vertex_offsets, face_cell_indices)` ready to be passed to
        `bores.grids.base.Grid`.
    :raises InvalidFaceConnectivityError: If any face is shared by more
        than two cells.
    """
    faces_per_cell = np.fromiter(map(len, per_cell_face_vertex_lists), dtype=np.int64)
    faces = list(itertools.chain.from_iterable(per_cell_face_vertex_lists))
    face_sizes = np.fromiter(map(len, faces), dtype=np.int64, count=len(faces))
    vertices = np.fromiter(
        itertools.chain.from_iterable(faces), dtype=np.int64, count=int(face_sizes.sum())
    )
    return build_csr_face_arrays_from_entries(
        vertex_coordinates,
        entry_cells=typing.cast(
            "IntArray[OneDimension]",
            np.repeat(np.arange(len(faces_per_cell), dtype=np.int64), faces_per_cell),
        ),
        entry_offsets=typing.cast(
            "IntArray[OneDimension]", np.concatenate([[0], np.cumsum(face_sizes)])
        ),
        entry_vertices=typing.cast("IntArray[OneDimension]", vertices),
    )


class FaultRecord(typing.NamedTuple):
    """
    One record from the GRDECL `FAULTS` keyword.

    Each record declares a named fault plane over a rectangular IJK range
    and a face direction.

    :param name: Fault name (case-sensitive as written in the GRDECL file).
    :param i1: 1-based start index in the I (x) direction (inclusive).
    :param i2: 1-based end index in the I direction (inclusive).
    :param j1: 1-based start index in the J (y) direction (inclusive).
    :param j2: 1-based end index in the J direction (inclusive).
    :param k1: 1-based start index in the K (z) direction (inclusive).
    :param k2: 1-based end index in the K direction (inclusive).
    :param face_direction: Eclipse face direction string, one of
        `'X'`, `'X-'`, `'Y'`, `'Y-'`, `'Z'`, `'Z-'`.
    """

    name: str
    i1: int
    i2: int
    j1: int
    j2: int
    k1: int
    k2: int
    face_direction: str


VALID_FAULT_FACE_DIRECTIONS: frozenset[str] = frozenset({
    "X",
    "X-",
    "Y",
    "Y-",
    "Z",
    "Z-",
})


MAP_AXES_PERPENDICULAR_TOLERANCE = 1e-3
"""Largest accepted cosine of the angle between the two `MAPAXES` axes."""


def validate_map_axes(map_axes: MapAxes) -> None:
    """
    Check that the two `MAPAXES` axes are perpendicular, as a map placement requires.

    Axes that are not perpendicular shear the grid and change cell volumes, so they are
    refused instead of applied.

    :param map_axes: Map axes to check.
    :raises ValidationError: If the axes are not perpendicular.
    """
    x_vector = map_axes.map_x_axis_point - map_axes.origin
    y_vector = map_axes.map_y_axis_point - map_axes.origin
    x_length = np.linalg.norm(x_vector)
    y_length = np.linalg.norm(y_vector)
    if x_length < 1e-14 or y_length < 1e-14:
        return
    cosine = x_vector @ y_vector / (x_length * y_length)
    if abs(cosine) > MAP_AXES_PERPENDICULAR_TOLERANCE:
        angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
        raise ValidationError(
            f"The `MAPAXES` axes are {angle:.1f} degrees apart but must be perpendicular. "
            f"The six values are the Y-axis point, the origin, then the X-axis point."
        )


def map_xy_to_map_space(
    xy: NumberArray[TwoDimensions], map_axes: MapAxes
) -> NumberArray[TwoDimensions]:
    """
    Map local `(x, y)` pairs into map space: `origin + rotation_matrix @ xy`.

    :param xy: Shape `(n, 2)` local-space points.
    :param map_axes: Map axes to apply.
    :returns: Shape `(n, 2)` map-space points.
    """
    return typing.cast(
        "NumberArray[TwoDimensions]", map_axes.origin + xy @ map_axes.rotation_matrix.T
    )
