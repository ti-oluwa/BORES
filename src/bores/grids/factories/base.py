import typing
from typing import TypeAlias

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
    face_registry: dict[FaceKey, FaceRecord] = {}
    for cell_index, cell_faces in enumerate(per_cell_face_vertex_lists):
        for face_vertex_indices in cell_faces:
            key: FaceKey = tuple(sorted(face_vertex_indices))
            if key not in face_registry:
                # First time seen: this cell becomes the owner.
                # Store the winding as-is (CCW from owner's outside = outward from owner).
                face_registry[key] = FaceRecord(
                    owner_cell_index=cell_index,
                    face_vertex_indices=face_vertex_indices,
                )
            else:
                record = face_registry[key]
                if record.neighbour_cell_index != -1:
                    raise InvalidFaceConnectivityError(
                        f"Face with canonical key {key} is shared by "
                        f"more than two cells (already has owner "
                        f"{record.owner_cell_index} and neighbour "
                        f"{record.neighbour_cell_index}; tried to add cell "
                        f"{cell_index})."
                    )
                record.neighbour_cell_index = cell_index

    # Flatten registry to CSR arrays
    flat_face_vertex_indices: list[Integer] = []
    face_vertex_offsets: list[int] = [0]
    face_cell_pairs: list[tuple[Integer, Integer]] = []

    for record in face_registry.values():
        flat_face_vertex_indices.extend(record.face_vertex_indices)
        face_vertex_offsets.append(len(flat_face_vertex_indices))
        face_cell_pairs.append((record.owner_cell_index, record.neighbour_cell_index))

    return (
        vertex_coordinates,
        typing.cast(IntArray[OneDimension], np.asarray(flat_face_vertex_indices, dtype=np.int32)),
        typing.cast(IntArray[OneDimension], np.asarray(face_vertex_offsets, dtype=np.int32)),
        typing.cast(IntArray[TwoDimensions], np.asarray(face_cell_pairs, dtype=np.int32)),
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
        NumberArray[TwoDimensions], map_axes.origin + xy @ map_axes.rotation_matrix.T
    )
