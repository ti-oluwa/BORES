"""Face-based unstructured polyhedral grid for reservoir simulation."""

import enum
import typing

import attrs
import numba
import numpy as np
from scipy.spatial import cKDTree
from typing_extensions import Self

from bores.constants import UnitConversionTable, get_conversion_factors
from bores.datastructures import GridDimensions, MapAxes
from bores.deck.file import DeckFile
from bores.errors import (
    CellNotFoundError,
    InvalidFaceAreaError,
    InvalidFaceConnectivityError,
    InvalidNormalVectorError,
    InvalidPointArrayError,
    InvalidVolumeError,
    ValidationError,
)
from bores.serde.base import Serializable
from bores.types import (
    Float,
    IntArray,
    Integer,
    Number,
    NumberArray,
    NumberOrArray,
    OneDimension,
    TwoDimensions,
    UnitSystem,
)

__all__ = ["CellStatus", "ConnectionType", "Grid"]


class ConnectionType(enum.IntEnum):
    """
    Classification of flow connections between reservoir cells.

    Face-based connections originate from the grid topology and correspond
    to geometric faces in the mesh. NNC-based connections represent
    additional cell-to-cell flow paths that do not correspond to a shared
    geometric face.

    Connection types determine how transmissibilities and transmissibility
    multipliers are applied during simulation.

    **Face connections**:

    `INTERIOR_FACE`
        Standard internal face shared by two active cells.

    `BOUNDARY_FACE`
        Boundary face connecting an active cell to the exterior domain.
        The neighbour cell index is `-1`.

    `INTERIOR_FAULT_FACE`
        Internal face shared by two active cells and belonging to a named
        fault. Directional transmissibility multipliers and fault
        transmissibility multipliers may both apply.

    `BOUNDARY_FAULT_FACE`
        Boundary face belonging to a named fault.

    """

    INTERIOR_FACE = 0
    """Standard interior face between two active cells."""

    BOUNDARY_FACE = 1
    """Boundary face between an active cell and the exterior domain."""

    INTERIOR_FAULT_FACE = 2
    """
    Interior face belonging to a fault.

    Both directional transmissibility multipliers and fault
    transmissibility multipliers may apply.
    """

    BOUNDARY_FAULT_FACE = 3
    """
    Boundary face belonging to a fault.
    """


class CellStatus(enum.IntEnum):
    """
    Activation status of a grid cell.

    `ACTIVE`
        The cell participates in flow simulation.

    `INACTIVE`
        The cell is excluded from flow simulation (e.g. ACTNUM == 0).
    """

    ACTIVE = 1
    INACTIVE = 0


GEOMETRY_TOLERANCE: float = 1e-14


@numba.njit(parallel=True, cache=True)
def compute_face_geometry(
    face_vertex_indices: IntArray[OneDimension],
    face_vertex_offsets: IntArray[OneDimension],
    vertex_coordinates: NumberArray[TwoDimensions],
) -> tuple[
    NumberArray[TwoDimensions],
    NumberArray[OneDimension],
    NumberArray[TwoDimensions],
]:
    """
    Compute area-weighted face centroids, areas, and unit normals (Newell's method).

    :param face_vertex_indices: Flat CSR data array of vertex indices.
    :param face_vertex_offsets: CSR offset array of length `n_faces + 1`.
    :param vertex_coordinates: Shape `(n_vertices, 3)` coordinate array.
    :returns: Tuple `(face_centroids, face_areas, face_unit_normals)`.
    """
    n_faces = face_vertex_offsets.shape[0] - 1
    face_centroids = np.zeros((n_faces, 3), dtype=np.float64)
    face_unit_normals = np.zeros((n_faces, 3), dtype=np.float64)
    face_areas = np.zeros(n_faces, dtype=np.float64)

    for face_idx in numba.prange(n_faces):  # type: ignore
        start = face_vertex_offsets[face_idx]
        end = face_vertex_offsets[face_idx + 1]
        n_verts = end - start

        cx = 0.0
        cy = 0.0
        cz = 0.0
        for local_idx in range(n_verts):
            vert_idx = face_vertex_indices[start + local_idx]
            cx += vertex_coordinates[vert_idx, 0]
            cy += vertex_coordinates[vert_idx, 1]
            cz += vertex_coordinates[vert_idx, 2]

        cx /= n_verts
        cy /= n_verts
        cz /= n_verts

        # Area-weighted centroid: triangle fan about the vertex mean, offsets taken
        # relative to that mean to keep precision at large coordinates.
        weighted_x = 0.0
        weighted_y = 0.0
        weighted_z = 0.0
        total_triangle_area = 0.0
        for local_idx in range(n_verts):
            a_idx = face_vertex_indices[start + local_idx]
            b_idx = face_vertex_indices[start + (local_idx + 1) % n_verts]
            ax = vertex_coordinates[a_idx, 0] - cx
            ay = vertex_coordinates[a_idx, 1] - cy
            az = vertex_coordinates[a_idx, 2] - cz
            bx = vertex_coordinates[b_idx, 0] - cx
            by = vertex_coordinates[b_idx, 1] - cy
            bz = vertex_coordinates[b_idx, 2] - cz
            cross_x = ay * bz - az * by
            cross_y = az * bx - ax * bz
            cross_z = ax * by - ay * bx
            triangle_area = 0.5 * np.sqrt(cross_x**2 + cross_y**2 + cross_z**2)
            total_triangle_area += triangle_area
            weighted_x += triangle_area * (ax + bx) / 3.0
            weighted_y += triangle_area * (ay + by) / 3.0
            weighted_z += triangle_area * (az + bz) / 3.0

        if total_triangle_area > 0.0:
            cx += weighted_x / total_triangle_area
            cy += weighted_y / total_triangle_area
            cz += weighted_z / total_triangle_area
        face_centroids[face_idx, 0] = cx
        face_centroids[face_idx, 1] = cy
        face_centroids[face_idx, 2] = cz

        nx = 0.0
        ny = 0.0
        nz = 0.0
        for local_idx in range(n_verts):
            a_idx = face_vertex_indices[start + local_idx]
            b_idx = face_vertex_indices[start + (local_idx + 1) % n_verts]

            ax = vertex_coordinates[a_idx, 0]
            ay = vertex_coordinates[a_idx, 1]
            az = vertex_coordinates[a_idx, 2]
            bx = vertex_coordinates[b_idx, 0]
            by = vertex_coordinates[b_idx, 1]
            bz = vertex_coordinates[b_idx, 2]

            nx += (ay - by) * (az + bz)
            ny += (az - bz) * (ax + bx)
            nz += (ax - bx) * (ay + by)

        normal_magnitude = np.sqrt(nx * nx + ny * ny + nz * nz)
        if normal_magnitude > 0.0:
            face_unit_normals[face_idx, 0] = nx / normal_magnitude
            face_unit_normals[face_idx, 1] = ny / normal_magnitude
            face_unit_normals[face_idx, 2] = nz / normal_magnitude
            face_areas[face_idx] = normal_magnitude * 0.5

    return face_centroids, face_areas, face_unit_normals


@numba.njit(cache=True)
def compute_cell_volumes_and_centroids(
    face_cell_indices: IntArray[TwoDimensions],
    face_vertex_indices: IntArray[OneDimension],
    face_vertex_offsets: IntArray[OneDimension],
    vertex_coordinates: NumberArray[TwoDimensions],
    n_cells: int,
) -> tuple[NumberArray[OneDimension], NumberArray[TwoDimensions]]:
    """
    Compute cell volumes and centroids via the divergence theorem.

    :param face_cell_indices: Shape `(n_faces, 2)`.
    :param face_vertex_indices: Flat CSR vertex index data array.
    :param face_vertex_offsets: CSR offset array of length `n_faces + 1`.
    :param vertex_coordinates: Shape `(n_vertices, 3)`.
    :param n_cells: Total number of cells.
    :returns: Tuple `(cell_volumes, cell_centroids)`.
    """
    n_faces = face_cell_indices.shape[0]
    cell_volumes = np.zeros(n_cells, dtype=np.float64)
    centroid_accumulators = np.zeros((n_cells, 3), dtype=np.float64)

    for face_idx in range(n_faces):
        owner_cell = face_cell_indices[face_idx, 0]
        neighbour_cell = face_cell_indices[face_idx, 1]
        start = face_vertex_offsets[face_idx]
        end = face_vertex_offsets[face_idx + 1]
        apex = vertex_coordinates[face_vertex_indices[start]]

        for iteration in range(2):
            if iteration == 0:
                cell_idx = owner_cell
                sign = 1.0
            else:
                cell_idx = neighbour_cell
                sign = -1.0

            if cell_idx < 0:
                continue

            for fan_idx in range(start + 1, end - 1):
                v1 = vertex_coordinates[face_vertex_indices[fan_idx]]
                v2 = vertex_coordinates[face_vertex_indices[fan_idx + 1]]

                ax = apex[0]
                ay = apex[1]
                az = apex[2]
                bx = v1[0]
                by = v1[1]
                bz = v1[2]
                cx = v2[0]
                cy = v2[1]
                cz = v2[2]

                signed_tetrahedron_volume = (
                    ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) + az * (bx * cy - by * cx)
                ) / 6.0
                cell_volumes[cell_idx] += sign * signed_tetrahedron_volume

                x_bar = (ax + bx + cx) / 4.0
                y_bar = (ay + by + cy) / 4.0
                z_bar = (az + bz + cz) / 4.0

                w = sign * signed_tetrahedron_volume
                centroid_accumulators[cell_idx, 0] += w * x_bar
                centroid_accumulators[cell_idx, 1] += w * y_bar
                centroid_accumulators[cell_idx, 2] += w * z_bar

    cell_centroids = np.zeros((n_cells, 3), dtype=np.float64)
    for cell_idx in range(n_cells):
        vol = cell_volumes[cell_idx]
        if abs(vol) > 0.0:
            cell_centroids[cell_idx, 0] = centroid_accumulators[cell_idx, 0] / vol
            cell_centroids[cell_idx, 1] = centroid_accumulators[cell_idx, 1] / vol
            cell_centroids[cell_idx, 2] = centroid_accumulators[cell_idx, 2] / vol

    return cell_volumes, cell_centroids


@numba.njit(cache=True)
def compute_cell_bounding_boxes(
    face_cell_indices: IntArray[TwoDimensions],
    face_vertex_indices: IntArray[OneDimension],
    face_vertex_offsets: IntArray[OneDimension],
    vertex_coordinates: NumberArray[TwoDimensions],
    n_cells: int,
) -> tuple[NumberArray[TwoDimensions], NumberArray[TwoDimensions]]:
    """
    Compute per-cell axis-aligned bounding boxes.

    :param face_cell_indices: Shape `(n_faces, 2)`.
    :param face_vertex_indices: Flat CSR vertex index data array.
    :param face_vertex_offsets: CSR offset array.
    :param vertex_coordinates: Shape `(n_vertices, 3)`.
    :param n_cells: Total number of cells.
    :returns: Tuple `(cell_min_xyz, cell_max_xyz)` each of shape `(n_cells, 3)`.
    """
    cell_min = np.full((n_cells, 3), np.inf, dtype=np.float64)
    cell_max = np.full((n_cells, 3), -np.inf, dtype=np.float64)
    n_faces = face_cell_indices.shape[0]

    for face_idx in range(n_faces):
        owner = face_cell_indices[face_idx, 0]
        neighbour = face_cell_indices[face_idx, 1]

        start = face_vertex_offsets[face_idx]
        end = face_vertex_offsets[face_idx + 1]

        for i in range(start, end):
            vid = face_vertex_indices[i]
            vx = vertex_coordinates[vid, 0]
            vy = vertex_coordinates[vid, 1]
            vz = vertex_coordinates[vid, 2]

            if owner >= 0:
                if vx < cell_min[owner, 0]:
                    cell_min[owner, 0] = vx
                if vy < cell_min[owner, 1]:
                    cell_min[owner, 1] = vy
                if vz < cell_min[owner, 2]:
                    cell_min[owner, 2] = vz
                if vx > cell_max[owner, 0]:
                    cell_max[owner, 0] = vx
                if vy > cell_max[owner, 1]:
                    cell_max[owner, 1] = vy
                if vz > cell_max[owner, 2]:
                    cell_max[owner, 2] = vz

            if neighbour >= 0:
                if vx < cell_min[neighbour, 0]:
                    cell_min[neighbour, 0] = vx
                if vy < cell_min[neighbour, 1]:
                    cell_min[neighbour, 1] = vy
                if vz < cell_min[neighbour, 2]:
                    cell_min[neighbour, 2] = vz
                if vx > cell_max[neighbour, 0]:
                    cell_max[neighbour, 0] = vx
                if vy > cell_max[neighbour, 1]:
                    cell_max[neighbour, 1] = vy
                if vz > cell_max[neighbour, 2]:
                    cell_max[neighbour, 2] = vz

    return cell_min, cell_max


@typing.final
@attrs.frozen(slots=True, kw_only=True)
class Grid(
    Serializable,
    fields={
        "vertex_coordinates": NumberArray[TwoDimensions],
        "face_vertex_indices": IntArray[OneDimension],
        "face_vertex_offsets": IntArray[OneDimension],
        "face_cell_indices": IntArray[TwoDimensions],
        "unit_system": UnitSystem,
        "dimensions": GridDimensions | None,
        "metadata": typing.Mapping[str, typing.Any] | None,
        "cell_statuses": IntArray[OneDimension] | None,
        "face_connection_types": IntArray[OneDimension] | None,
        "cell_volumes": NumberArray[OneDimension] | None,
        "cell_centroids": NumberArray[TwoDimensions] | None,
        "cell_min_xyz": NumberArray[TwoDimensions] | None,
        "cell_max_xyz": NumberArray[TwoDimensions] | None,
        "nnc_cell_indices": IntArray[TwoDimensions] | None,
        "nnc_transmissibilities": NumberArray[OneDimension] | None,
        "fault_face_indices": typing.Mapping[str, IntArray[OneDimension]] | None,
        "fault_transmissibility_multipliers": typing.Mapping[str, Number] | None,
        "positive_x_transmissibility_multipliers": NumberArray[OneDimension] | None,
        "negative_x_transmissibility_multipliers": NumberArray[OneDimension] | None,
        "positive_y_transmissibility_multipliers": NumberArray[OneDimension] | None,
        "negative_y_transmissibility_multipliers": NumberArray[OneDimension] | None,
        "positive_z_transmissibility_multipliers": NumberArray[OneDimension] | None,
        "negative_z_transmissibility_multipliers": NumberArray[OneDimension] | None,
    },
):
    """
    Immutable face-based unstructured polyhedral grid.

    All topology and geometry is computed once during construction and stored as
    read-only NumPy arrays. All index arrays use int32 and all floating-point
    arrays use float64.

    **Connection model**

    Two layers of connection data are maintained:

    `face_connection_types` (shape `(n_faces,)`)
        Per-face type: `BOUNDARY_FACE`, `INTERIOR_FACE`, `INTERIOR_FAULT_FACE`,
        `BOUNDARY_FAULT_FACE`. This covers every geometric face in the grid.

    **Raises**:

    `InvalidPointArrayError`
        If `vertex_coordinates` is not a 2-D `(n_vertices, 3)` array.
    `InvalidFaceConnectivityError`
        If face connectivity arrays are malformed.
    `InvalidVolumeError`
        If any cell has a non-positive volume after construction.
    """

    vertex_coordinates: NumberArray[TwoDimensions]
    """
    Shape `(n_vertices, 3)` - world (x, y, z) coordinates.
    z-axis is positive downward (reservoir depth convention).
    """

    face_vertex_indices: IntArray[OneDimension]
    """
    Flat CSR data array: concatenated vertex index lists for all faces.
    Face *f* uses
    `face_vertex_indices[face_vertex_offsets[f]:face_vertex_offsets[f+1]]`.
    """

    face_vertex_offsets: IntArray[OneDimension]
    """CSR offset array of length `n_faces + 1`."""

    face_cell_indices: IntArray[TwoDimensions]
    """
    Shape `(n_faces, 2)` - `(owner_cell_index, neighbour_cell_index)`.
    Boundary faces have `neighbour_cell_index == -1`.
    """

    unit_system: UnitSystem = attrs.field(default=UnitSystem.FIELD)
    """Declared unit system for all coordinate and geometry arrays."""

    dimensions: GridDimensions | None = attrs.field(default=None)
    """Dimension of the unstructure grid."""

    metadata: typing.Mapping[str, typing.Any] | None = attrs.field(default=None)
    """Optional free-form metadata mapping."""

    cell_statuses: IntArray[OneDimension] | None = attrs.field(default=None)
    """
    Shape `(n_cells,)` - per-cell `CellStatus` flags.
    Auto-populated to all `CellStatus.ACTIVE` when `None`.
    """

    face_connection_types: IntArray[OneDimension] | None = attrs.field(default=None)
    """
    Shape `(n_faces,)` - per-face `ConnectionType`.
    Auto-populated from topology (`BOUNDARY` / `INTERIOR`) when `None`.
    Factories that know about faults or pinchouts supply an explicit array.
    """

    cell_volumes: NumberArray[OneDimension] | None = attrs.field(default=None)
    """
    Shape `(n_cells,)` pre-computed cell volumes from the factory.
    When provided, the divergence-theorem computation is skipped.
    """

    cell_centroids: NumberArray[TwoDimensions] | None = attrs.field(default=None)
    """
    Shape `(n_cells, 3)` pre-computed cell centroids from the factory.
    Must be provided together with `cell_volumes`.
    """

    nnc_cell_indices: IntArray[TwoDimensions] | None = attrs.field(default=None)
    """
    Shape `(n_nnc, 2)` - non-neighbour connection cell index pairs.
    `None` when no NNCs are present.
    """

    nnc_transmissibilities: NumberArray[OneDimension] | None = attrs.field(default=None)
    """
    Shape `(n_nnc,)` - explicitly supplied flow transmissibility of each NNC pair: volumetric
    rate x viscosity / pressure, in the grid's unit system (ft3.cP/day/psi in FIELD units).
    It already includes the Darcy unit conversion constant, so it multiplies mobility and
    pressure difference directly. Required whenever `nnc_cell_indices` is given: every value
    must be finite and non-negative.
    """

    fault_face_indices: typing.Mapping[str, IntArray[OneDimension]] | None = attrs.field(
        default=None
    )
    """
    Mapping from fault name to 1-D array of face indices belonging to that fault.
    Populated from the GRDECL `FAULTS` keyword. `None` when absent.
    """

    fault_transmissibility_multipliers: typing.Mapping[str, Number] | None = attrs.field(
        default=None
    )
    """
    Mapping from fault name to its transmissibility multiplier (from `MULTFLT`).
    `None` when absent.
    """

    positive_x_transmissibility_multipliers: NumberArray[OneDimension] | None = attrs.field(
        default=None
    )
    """Shape `(n_cells,)` MULTX multipliers. `None` when not supplied."""

    negative_x_transmissibility_multipliers: NumberArray[OneDimension] | None = attrs.field(
        default=None
    )
    """Shape `(n_cells,)` MULTX- multipliers. `None` when not supplied."""

    positive_y_transmissibility_multipliers: NumberArray[OneDimension] | None = attrs.field(
        default=None
    )
    """Shape `(n_cells,)` MULTY multipliers. `None` when not supplied."""

    negative_y_transmissibility_multipliers: NumberArray[OneDimension] | None = attrs.field(
        default=None
    )
    """Shape `(n_cells,)` MULTY- multipliers. `None` when not supplied."""

    positive_z_transmissibility_multipliers: NumberArray[OneDimension] | None = attrs.field(
        default=None
    )
    """Shape `(n_cells,)` MULTZ multipliers. `None` when not supplied."""

    negative_z_transmissibility_multipliers: NumberArray[OneDimension] | None = attrs.field(
        default=None
    )
    """Shape `(n_cells,)` MULTZ- multipliers. `None` when not supplied."""

    cell_face_indices: IntArray[OneDimension] = attrs.field(init=False)
    """Flat CSR data array: face indices per cell."""

    cell_face_offsets: IntArray[OneDimension] = attrs.field(init=False)
    """CSR offset array of length `n_cells + 1` for the cell-to-face map."""

    cell_neighbor_indices: IntArray[OneDimension] = attrs.field(init=False)
    """
    Flat CSR data array: neighbour cell indices per cell (interior faces only).
    """

    cell_neighbor_offsets: IntArray[OneDimension] = attrs.field(init=False)
    """CSR offset array of length `n_cells + 1` for the cell-to-neighbour map."""

    boundary_face_indices: IntArray[OneDimension] = attrs.field(init=False)
    """Indices of all boundary faces."""

    interior_face_indices: IntArray[OneDimension] = attrs.field(init=False)
    """Indices of all interior faces."""

    face_centroids: NumberArray[TwoDimensions] = attrs.field(init=False)
    """Shape `(n_faces, 3)` - centroid of each face polygon."""

    face_areas: NumberArray[OneDimension] = attrs.field(init=False)
    """Shape `(n_faces,)` - geometric area of each face."""

    face_unit_normals: NumberArray[TwoDimensions] = attrs.field(init=False)
    """Shape `(n_faces, 3)` - unit outward normal from the owner cell."""

    cell_min_xyz: NumberArray[TwoDimensions] = attrs.field(default=None)  # type: ignore[assignment]
    """
    Shape `(n_cells, 3)` - AABB minimum corner per cell. A factory can pass the exact boxes
    (together with `cell_max_xyz`); otherwise they are taken from the cell's face vertices.
    """

    cell_max_xyz: NumberArray[TwoDimensions] = attrs.field(default=None)  # type: ignore[assignment]
    """
    Shape `(n_cells, 3)` - AABB maximum corner per cell. Must be provided together with
    `cell_min_xyz`.
    """

    bounding_box: tuple[Float, Float, Float, Float, Float, Float] = attrs.field(init=False)
    """Global AABB: `(x_min, x_max, y_min, y_max, z_min, z_max)`."""

    cell_length_x: NumberArray[OneDimension] = attrs.field(init=False)
    """Shape `(n_cells,)` - AABB extent in x direction."""

    cell_length_y: NumberArray[OneDimension] = attrs.field(init=False)
    """Shape `(n_cells,)` - AABB extent in y direction."""

    cell_length_z: NumberArray[OneDimension] = attrs.field(init=False)
    """Shape `(n_cells,)` - bounding-box extent in z direction."""

    cell_thickness: NumberArray[OneDimension] = attrs.field(init=False)
    """
    Shape `(n_cells,)` - mean vertical thickness (cell volume divided by its horizontally
    projected area). Equals `cell_length_z` for flat-topped cells and is smaller for dipping ones.
    """

    cell_center_depths: NumberArray[OneDimension] = attrs.field(init=False)
    """Shape `(n_cells,)` - depth of cell centroid (positive downward)."""

    cell_center_elevations: NumberArray[OneDimension] = attrs.field(init=False)
    """Shape `(n_cells,)` - elevation of cell centroid (negation of depth)."""

    _spatial_index: cKDTree | None = attrs.field(init=False, default=None)
    """KD-tree on cell centroids for fast nearest-cell queries."""

    def __attrs_post_init__(self) -> None:
        if self.dimensions is None and self.metadata is not None:
            dims = self.metadata.get("dimensions", None)
            if isinstance(dims, (tuple, list)) and dims:
                size = len(dims)
                if size == 3:
                    dimensions = GridDimensions(*dims)
                elif size == 2:
                    dimensions = GridDimensions(dims[0], dims[1], 0)
                else:
                    dimensions = GridDimensions(dims[0], 0, 0)
                object.__setattr__(self, "dimensions", dimensions)

        self._validate_inputs()
        self._canonicalize_boundary_faces()
        self._classify_faces()
        self._populate_defaults()
        self._validate_consistency()
        self._build_cell_face_connectivity()
        self._build_cell_neighbor_connectivity()
        self._compute_face_geometry()
        self._compute_cell_geometry()
        self._compute_bounding_boxes()
        self._compute_derived_dimensions()
        self._build_spatial_index()

    def _validate_inputs(self) -> None:
        if self.vertex_coordinates.ndim != 2 or self.vertex_coordinates.shape[1] != 3:
            raise InvalidPointArrayError(
                f"`vertex_coordinates` must be shape (n_vertices, 3); "
                f"got {self.vertex_coordinates.shape!r}."
            )
        if not np.isfinite(self.vertex_coordinates).all():
            bad = np.where(~np.isfinite(self.vertex_coordinates).all(axis=1))[0]
            raise InvalidPointArrayError(
                f"`vertex_coordinates` contains non-finite values at vertices "
                f"{bad[:5].tolist()}{'...' if len(bad) > 5 else ''}."
            )
        if self.face_cell_indices.ndim != 2 or self.face_cell_indices.shape[1] != 2:
            raise InvalidFaceConnectivityError(
                f"`face_cell_indices` must be shape (n_faces, 2); "
                f"got {self.face_cell_indices.shape!r}."
            )
        if self.face_vertex_offsets.ndim != 1 or self.face_vertex_offsets[0] != 0:
            raise InvalidFaceConnectivityError(
                "`face_vertex_offsets` must be a 1-D array starting at 0."
            )
        expected_n_faces = self.face_cell_indices.shape[0]
        same_cell = (self.face_cell_indices[:, 0] == self.face_cell_indices[:, 1]) & (
            self.face_cell_indices[:, 0] >= 0
        )
        if same_cell.any():
            bad = np.where(same_cell)[0]
            raise InvalidFaceConnectivityError(
                f"{len(bad)} face(s) have the same cell as owner and neighbour: "
                f"{bad[:5].tolist()}{'...' if len(bad) > 5 else ''}."
            )
        if self.face_vertex_offsets.shape[0] != expected_n_faces + 1:
            raise InvalidFaceConnectivityError(
                f"`face_vertex_offsets` length must be n_faces + 1 = "
                f"{expected_n_faces + 1}; got {self.face_vertex_offsets.shape[0]}."
            )
        if np.any(np.diff(self.face_vertex_offsets) < 0):
            raise InvalidFaceConnectivityError("`face_vertex_offsets` must be non-decreasing.")
        if int(self.face_vertex_offsets[-1]) != len(self.face_vertex_indices):
            raise InvalidFaceConnectivityError(
                f"face_vertex_offsets[-1] = {self.face_vertex_offsets[-1]} does not "
                f"match len(face_vertex_indices) = {len(self.face_vertex_indices)}."
            )
        max_valid_vertex = self.vertex_coordinates.shape[0] - 1
        if self.face_vertex_indices.size > 0:
            if int(self.face_vertex_indices.max()) > max_valid_vertex:
                raise InvalidFaceConnectivityError(
                    f"`face_vertex_indices` contains index "
                    f"{int(self.face_vertex_indices.max())} which exceeds "
                    f"max valid index {max_valid_vertex}."
                )
            if int(self.face_vertex_indices.min()) < 0:
                raise InvalidFaceConnectivityError(
                    f"`face_vertex_indices` contains negative index "
                    f"{int(self.face_vertex_indices.min())}."
                )

        if self.face_cell_indices.shape[0] == 0:
            raise InvalidFaceConnectivityError(
                "`face_cell_indices` must contain at least one face."
            )

        min_cell_index = int(self.face_cell_indices.min())
        if min_cell_index < -1:
            raise InvalidFaceConnectivityError(
                f"`face_cell_indices` contains negative cell index {min_cell_index}; "
                "only -1 is allowed (boundary sentinel)."
            )
        if self.nnc_cell_indices is not None:
            if self.nnc_transmissibilities is not None and len(self.nnc_transmissibilities) != len(
                self.nnc_cell_indices
            ):
                raise InvalidFaceConnectivityError(
                    f"`nnc_transmissibilities` length {len(self.nnc_transmissibilities)} "
                    f"does not match `nnc_cell_indices` length {len(self.nnc_cell_indices)}."
                )
            if len(self.nnc_cell_indices) > 0:
                n_cells_declared = self._get_cell_count()
                if self.nnc_cell_indices.ndim != 2 or self.nnc_cell_indices.shape[1] != 2:
                    raise InvalidFaceConnectivityError(
                        f"`nnc_cell_indices` must be shape (n_nnc, 2); "
                        f"got {self.nnc_cell_indices.shape!r}."
                    )
                lowest = int(self.nnc_cell_indices.min())
                highest = int(self.nnc_cell_indices.max())
                if lowest < 0 or highest >= n_cells_declared:
                    raise InvalidFaceConnectivityError(
                        f"`nnc_cell_indices` contains cell index "
                        f"{lowest if lowest < 0 else highest} outside the valid range "
                        f"[0, {n_cells_declared - 1}]."
                    )

    def _get_cell_count(self) -> int:
        """
        Number of cells implied by the face connectivity and any per-cell arrays supplied.

        Cells with no faces (fully pinched out) carry no entries in `face_cell_indices`, so
        the count is taken from the per-cell arrays when they are longer.
        """
        from_faces = int(self.face_cell_indices.max()) + 1
        supplied = {
            len(array)
            for array in (self.cell_volumes, self.cell_centroids, self.cell_statuses)
            if array is not None
        }
        if len(supplied) > 1:
            raise InvalidFaceConnectivityError(
                "`cell_volumes`, `cell_centroids` and `cell_statuses` must have the same "
                f"length; got lengths {sorted(supplied)}."
            )
        if supplied:
            count = supplied.pop()
            if count < from_faces:
                raise InvalidFaceConnectivityError(
                    f"Per-cell arrays have {count} entries but `face_cell_indices` "
                    f"references cell {from_faces - 1}."
                )
            return count
        return from_faces

    def _validate_consistency(self) -> None:
        """Check that per-cell, per-face and per-NNC arrays agree with the grid topology."""
        n_cells = self._get_cell_count()
        n_faces = self.face_cell_indices.shape[0]

        assert self.cell_statuses is not None
        if (
            self.cell_statuses.shape != (n_cells,)
            or not np.isin(
                self.cell_statuses, [int(CellStatus.ACTIVE), int(CellStatus.INACTIVE)]
            ).all()
        ):
            raise ValidationError(
                f"`cell_statuses` must have shape ({n_cells},) with values 0 (inactive) or 1 (active)."
            )

        assert self.face_connection_types is not None
        valid_face_types = [int(member) for member in ConnectionType]
        if (
            self.face_connection_types.shape != (n_faces,)
            or not np.isin(self.face_connection_types, valid_face_types).all()
        ):
            raise ValidationError(
                f"`face_connection_types` must have shape ({n_faces},) and hold valid `ConnectionType` values."
            )

        if self.dimensions is not None:
            nx, ny, nz = self.dimensions
            if nx * ny * nz != n_cells:
                raise ValidationError(
                    f"`dimensions` {tuple(self.dimensions)} imply {nx * ny * nz} cells but the "
                    f"grid has {n_cells}."
                )

        for name in (
            "positive_x_transmissibility_multipliers",
            "negative_x_transmissibility_multipliers",
            "positive_y_transmissibility_multipliers",
            "negative_y_transmissibility_multipliers",
            "positive_z_transmissibility_multipliers",
            "negative_z_transmissibility_multipliers",
        ):
            multipliers = getattr(self, name)
            if multipliers is None:
                continue
            if multipliers.shape != (n_cells,):
                raise ValidationError(
                    f"`{name}` must have shape ({n_cells},); got {multipliers.shape!r}."
                )
            if not (np.isfinite(multipliers).all() and (multipliers >= 0.0).all()):
                raise ValidationError(f"`{name}` must be finite and non-negative.")

        if self.fault_transmissibility_multipliers:
            for fault_name, multiplier in self.fault_transmissibility_multipliers.items():
                if not (np.isfinite(multiplier) and multiplier >= 0.0):
                    raise ValidationError(
                        f"Transmissibility multiplier for fault {fault_name!r} must be finite "
                        f"and non-negative; got {multiplier}."
                    )
        if self.fault_face_indices:
            for fault_name, face_indices in self.fault_face_indices.items():
                if len(face_indices) and (face_indices.min() < 0 or face_indices.max() >= n_faces):
                    raise ValidationError(
                        f"Fault {fault_name!r} references a face outside [0, {n_faces - 1}]."
                    )

        if self.cell_volumes is not None:
            if self.cell_volumes.shape != (n_cells,) or not (
                np.isfinite(self.cell_volumes).all() and (self.cell_volumes >= 0.0).all()
            ):
                raise InvalidVolumeError(
                    f"`cell_volumes` must have shape ({n_cells},) and be finite and non-negative."
                )
        if (self.cell_min_xyz is None) != (self.cell_max_xyz is None):
            raise InvalidPointArrayError(
                "`cell_min_xyz` and `cell_max_xyz` must be provided together."
            )
        if self.cell_min_xyz is not None and self.cell_max_xyz is not None:
            if (
                self.cell_min_xyz.shape != (n_cells, 3)
                or self.cell_max_xyz.shape != (n_cells, 3)
                or not np.isfinite(self.cell_min_xyz).all()
                or not np.isfinite(self.cell_max_xyz).all()
                or (self.cell_min_xyz > self.cell_max_xyz).any()
            ):
                raise InvalidPointArrayError(
                    f"`cell_min_xyz` and `cell_max_xyz` must have shape ({n_cells}, 3), be finite "
                    "and satisfy min <= max."
                )
        if self.cell_centroids is not None:
            if (
                self.cell_centroids.shape != (n_cells, 3)
                or not np.isfinite(self.cell_centroids).all()
            ):
                raise InvalidPointArrayError(
                    f"`cell_centroids` must have shape ({n_cells}, 3) and be finite."
                )

        if self.nnc_cell_indices is not None and len(self.nnc_cell_indices) > 0:
            n_nnc = len(self.nnc_cell_indices)
            pairs = self.nnc_cell_indices
            if (pairs[:, 0] == pairs[:, 1]).any():
                raise InvalidFaceConnectivityError("An NNC connects a cell to itself.")
            if not (
                (self.cell_statuses[pairs[:, 0]] == int(CellStatus.ACTIVE)).all()
                and (self.cell_statuses[pairs[:, 1]] == int(CellStatus.ACTIVE)).all()
            ):
                raise InvalidFaceConnectivityError("An NNC connects to an inactive cell.")
            if self.nnc_transmissibilities is None:
                raise ValidationError(
                    "`nnc_transmissibilities` is required with `nnc_cell_indices`: an NNC has no "
                    "shared surface, so its flow transmissibility cannot be computed."
                )
            if self.nnc_transmissibilities.shape != (n_nnc,):
                raise InvalidFaceConnectivityError(
                    f"`nnc_transmissibilities` must have {n_nnc} entries; "
                    f"got {self.nnc_transmissibilities.shape!r}."
                )
            if not (
                np.isfinite(self.nnc_transmissibilities).all()
                and (self.nnc_transmissibilities >= 0.0).all()
            ):
                raise ValidationError("`nnc_transmissibilities` must be finite and non-negative.")

    def _canonicalize_boundary_faces(self) -> None:
        """
        Ensure every boundary face has a real owner cell and a `-1` neighbour.

        A boundary face supplied with the `-1` in the owner column is swapped so the
        owner is the real cell, and its vertex winding is reversed so the face normal
        still points out of its (new) owner.
        """
        owner_cells = self.face_cell_indices[:, 0]
        neighbour_cells = self.face_cell_indices[:, 1]
        if np.any((owner_cells < 0) & (neighbour_cells < 0)):
            bad = np.where((owner_cells < 0) & (neighbour_cells < 0))[0]
            raise InvalidFaceConnectivityError(
                f"{len(bad)} face(s) have no adjacent cell: {bad[:5].tolist()}"
                f"{'...' if len(bad) > 5 else ''}."
            )

        flip = np.where((owner_cells < 0) & (neighbour_cells >= 0))[0]
        if flip.size == 0:
            return

        face_cell_indices = self.face_cell_indices.copy()
        face_cell_indices[flip] = face_cell_indices[flip][:, ::-1]

        offsets = self.face_vertex_offsets.astype(np.int64)
        starts = offsets[flip]
        lengths = offsets[flip + 1] - starts
        face_position = np.repeat(np.arange(flip.size), lengths)
        local = np.arange(int(lengths.sum())) - np.repeat(np.cumsum(lengths) - lengths, lengths)
        source = starts[face_position] + local
        destination = starts[face_position] + (lengths[face_position] - 1 - local)
        face_vertex_indices = self.face_vertex_indices.copy()
        face_vertex_indices[destination] = self.face_vertex_indices[source]

        object.__setattr__(self, "face_cell_indices", face_cell_indices)
        object.__setattr__(self, "face_vertex_indices", face_vertex_indices)

    def _classify_faces(self) -> None:
        owner_cells = self.face_cell_indices[:, 0]
        neighbour_cells = self.face_cell_indices[:, 1]
        boundary_mask = (owner_cells < 0) | (neighbour_cells < 0)
        interior_mask = ~boundary_mask
        object.__setattr__(
            self,
            "boundary_face_indices",
            np.where(boundary_mask)[0].astype(np.int32),
        )
        object.__setattr__(
            self,
            "interior_face_indices",
            np.where(interior_mask)[0].astype(np.int32),
        )

    def _populate_defaults(self) -> None:
        n_faces = self.face_cell_indices.shape[0]
        n_cells = self._get_cell_count()

        if self.face_connection_types is None:
            face_connection_types = np.full(n_faces, ConnectionType.INTERIOR_FACE, dtype=np.int8)
            face_connection_types[self.boundary_face_indices] = int(ConnectionType.BOUNDARY_FACE)

            if self.fault_face_indices is not None:
                for face_indices in self.fault_face_indices.values():
                    boundary_fault_mask = (self.face_cell_indices[face_indices, 0] < 0) | (
                        self.face_cell_indices[face_indices, 1] < 0
                    )
                    boundary_fault_faces = face_indices[boundary_fault_mask]
                    interior_fault_faces = face_indices[~boundary_fault_mask]

                    face_connection_types[interior_fault_faces] = int(
                        ConnectionType.INTERIOR_FAULT_FACE
                    )
                    face_connection_types[boundary_fault_faces] = int(
                        ConnectionType.BOUNDARY_FAULT_FACE
                    )
            object.__setattr__(self, "face_connection_types", face_connection_types)

        if self.cell_statuses is None:
            cell_statuses = np.full(n_cells, CellStatus.ACTIVE, dtype=np.int8)
            object.__setattr__(self, "cell_statuses", cell_statuses)

    def _build_cell_face_connectivity(self) -> None:
        n_cells = self._get_cell_count()
        n_faces = self.face_cell_indices.shape[0]
        cells = self.face_cell_indices.reshape(-1).astype(np.int64)
        faces = np.repeat(np.arange(n_faces, dtype=np.int64), 2)
        present = cells >= 0
        cells = cells[present]
        faces = faces[present]
        order = np.argsort(cells, kind="stable")
        counts = np.bincount(cells, minlength=n_cells)
        offsets = np.concatenate([[0], np.cumsum(counts)])
        object.__setattr__(self, "cell_face_indices", faces[order].astype(np.int32))
        object.__setattr__(self, "cell_face_offsets", offsets.astype(np.int32))

    def _build_cell_neighbor_connectivity(self) -> None:
        n_cells = self._get_cell_count()
        interior = (self.face_cell_indices[:, 0] >= 0) & (self.face_cell_indices[:, 1] >= 0)
        owners = self.face_cell_indices[interior, 0].astype(np.int64)
        neighbours = self.face_cell_indices[interior, 1].astype(np.int64)
        sources = np.concatenate([owners, neighbours])
        targets = np.concatenate([neighbours, owners])
        keys = np.unique(sources * n_cells + targets)
        source_cells = keys // n_cells
        counts = np.bincount(source_cells, minlength=n_cells)
        offsets = np.concatenate([[0], np.cumsum(counts)])
        object.__setattr__(self, "cell_neighbor_indices", (keys % n_cells).astype(np.int32))
        object.__setattr__(self, "cell_neighbor_offsets", offsets.astype(np.int32))

    def _compute_face_geometry(self) -> None:
        face_centroids, face_areas, face_unit_normals = compute_face_geometry(
            face_vertex_indices=self.face_vertex_indices,
            face_vertex_offsets=self.face_vertex_offsets,
            vertex_coordinates=self.vertex_coordinates,
        )
        object.__setattr__(self, "face_centroids", face_centroids)
        object.__setattr__(self, "face_areas", face_areas)
        object.__setattr__(self, "face_unit_normals", face_unit_normals)

    def _compute_cell_geometry(self) -> None:
        if self.cell_volumes is not None and self.cell_centroids is not None:
            return

        n_cells = self._get_cell_count()
        origin = self.vertex_coordinates.mean(axis=0)
        cell_volumes, cell_centroids = compute_cell_volumes_and_centroids(
            face_cell_indices=self.face_cell_indices,
            face_vertex_indices=self.face_vertex_indices,
            face_vertex_offsets=self.face_vertex_offsets,
            vertex_coordinates=self.vertex_coordinates - origin,
            n_cells=n_cells,
        )
        cell_centroids += origin
        invalid_mask = ~(cell_volumes > 0.0) & (self.cell_statuses == int(CellStatus.ACTIVE))
        if invalid_mask.any():
            bad = np.where(invalid_mask)[0].tolist()
            raise InvalidVolumeError(
                f"Cells {bad[:20]}{'...' if len(bad) > 20 else ''} "
                f"have non-positive volumes. Check face winding order."
            )
        object.__setattr__(self, "cell_volumes", cell_volumes)
        object.__setattr__(self, "cell_centroids", cell_centroids)

    def _compute_bounding_boxes(self) -> None:
        n_cells = self._get_cell_count()
        if self.cell_min_xyz is not None and self.cell_max_xyz is not None:
            cell_min, cell_max = self.cell_min_xyz, self.cell_max_xyz
        else:
            cell_min, cell_max = compute_cell_bounding_boxes(
                face_cell_indices=self.face_cell_indices,
                face_vertex_indices=self.face_vertex_indices,
                face_vertex_offsets=self.face_vertex_offsets,
                vertex_coordinates=self.vertex_coordinates,
                n_cells=n_cells,
            )
            if self.cell_centroids is not None:
                no_face_mask = ~np.isfinite(cell_min).all(axis=1)
                if no_face_mask.any():
                    cell_min[no_face_mask] = self.cell_centroids[no_face_mask]
                    cell_max[no_face_mask] = self.cell_centroids[no_face_mask]

        bounding_box = (
            cell_min[:, 0].min(),
            cell_max[:, 0].max(),
            cell_min[:, 1].min(),
            cell_max[:, 1].max(),
            cell_min[:, 2].min(),
            cell_max[:, 2].max(),
        )
        object.__setattr__(self, "cell_min_xyz", cell_min)
        object.__setattr__(self, "cell_max_xyz", cell_max)
        object.__setattr__(self, "bounding_box", bounding_box)

    def _compute_derived_dimensions(self) -> None:
        delta = self.cell_max_xyz - self.cell_min_xyz
        object.__setattr__(self, "cell_length_x", np.ascontiguousarray(delta[:, 0]))
        object.__setattr__(self, "cell_length_y", np.ascontiguousarray(delta[:, 1]))
        object.__setattr__(self, "cell_length_z", np.ascontiguousarray(delta[:, 2]))
        object.__setattr__(self, "cell_thickness", self._compute_cell_thickness())

        assert self.cell_centroids is not None
        depths = self.cell_centroids[:, 2].copy()
        object.__setattr__(self, "cell_center_depths", depths)
        object.__setattr__(self, "cell_center_elevations", -depths)

    def _compute_cell_thickness(self) -> NumberArray[OneDimension]:
        """
        Mean vertical thickness of each cell: volume divided by horizontally projected area.

        For a cell with sloping or dipping top and bottom surfaces this is the true vertical
        thickness, unlike the bounding-box height, which also includes the dip.
        """
        assert self.cell_volumes is not None
        n_cells = self.cell_volumes.shape[0]
        vertical_projection = self.face_areas * np.abs(self.face_unit_normals[:, 2])
        projected_area = np.zeros(n_cells, dtype=np.float64)
        for column in (0, 1):
            cells = self.face_cell_indices[:, column]
            present = cells >= 0
            projected_area += np.bincount(
                cells[present], weights=vertical_projection[present], minlength=n_cells
            )
        projected_area *= 0.5
        thickness = np.zeros(n_cells, dtype=np.float64)
        np.divide(self.cell_volumes, projected_area, out=thickness, where=projected_area > 0.0)
        return thickness

    def _build_spatial_index(self) -> None:
        assert self.cell_centroids is not None
        object.__setattr__(self, "_spatial_index", cKDTree(self.cell_centroids))

    @property
    def n_cells(self) -> int:
        """Total number of cells."""
        assert self.cell_centroids is not None
        return self.cell_centroids.shape[0]

    @property
    def n_faces(self) -> int:
        """Total number of faces (boundary + interior)."""
        return self.face_cell_indices.shape[0]

    @property
    def n_vertices(self) -> int:
        """Total number of vertex points."""
        return self.vertex_coordinates.shape[0]

    @property
    def n_boundary_faces(self) -> int:
        """Number of boundary faces."""
        return len(self.boundary_face_indices)

    @property
    def n_interior_faces(self) -> int:
        """Number of interior faces."""
        return len(self.interior_face_indices)

    @property
    def n_nnc(self) -> int:
        """Number of non-neighbour connections. 0 when `nnc_cell_indices` is `None`."""
        if self.nnc_cell_indices is None:
            return 0
        return self.nnc_cell_indices.shape[0]

    @property
    def n_connections(self) -> int:
        """Total connections: `n_faces + n_nnc`."""
        return self.n_faces + self.n_nnc

    @property
    def n_faults(self) -> int:
        """Number of named faults. 0 when no fault data."""
        return len(self.fault_face_indices) if self.fault_face_indices is not None else 0

    @property
    def has_transmissibility_multipliers(self) -> bool:
        """`True` if any directional `MULT*` array is present."""
        return any(
            array is not None
            for array in (
                self.positive_x_transmissibility_multipliers,
                self.negative_x_transmissibility_multipliers,
                self.positive_y_transmissibility_multipliers,
                self.negative_y_transmissibility_multipliers,
                self.positive_z_transmissibility_multipliers,
                self.negative_z_transmissibility_multipliers,
            )
        )

    @property
    def map_axes(self) -> MapAxes | None:
        if self.metadata is not None:
            return self.metadata.get("map_axes", None)
        return None

    def flat_index(self, i: Integer, j: Integer, k: Integer) -> Integer:
        """
        Convert 0-based `(i, j, k)` to a flat index (`i` fastest, `k` slowest):

        `index = i + j*nx + k*nx*ny`.

        :param i: 0-based x index.
        :param j: 0-based y index.
        :param k: 0-based z index.
        :returns: Flat cell index.
        """
        dims = self.dimensions
        if dims is None:
            raise ValidationError(
                "Cannot compute flat index. Grid dimensions cannot be determined."
            )
        return dims.flat_index(i, j, k)

    def ijk_index(self, flat: Integer) -> tuple[int, int, int]:
        """
        Convert a flat index to  0-based `(i, j, k)`.

        Given the flat index was generated as `i` fastest, `k` slowest:

        `index = i + j*nx + k*nx*ny`.

        :param flat: Flat cell index.
        :returns: 0-based `(i, j, k)` cell index.
        """
        dims = self.dimensions
        if dims is None:
            raise ValidationError(
                "Cannot compute IJK index. Grid dimensions cannot be determined."
            )
        return dims.ijk_index(flat)

    def is_cell_active(self, cell_index: Integer) -> bool:
        """
        Return whether a given cell is active.

        :param cell_index: 0-based cell index.
        :returns: `True` if the cell has `CellStatus.ACTIVE`.
        :raises CellNotFoundError: If `cell_index` is out of range.
        """
        if cell_index < 0 or cell_index >= self.n_cells:
            raise CellNotFoundError(
                f"Cell index {cell_index} is out of range [0, {self.n_cells - 1}]."
            )

        assert self.cell_statuses is not None
        return bool(self.cell_statuses[cell_index])

    def get_face_type(self, face_index: Integer) -> ConnectionType:
        """
        Return the `ConnectionType` for a given face.

        :param face_index: 0-based face index.
        :returns: `ConnectionType` enum value.
        :raises IndexError: If `face_index` is out of range.
        """
        assert self.face_connection_types is not None
        if face_index < 0 or face_index >= self.n_faces:
            raise IndexError(f"Face index {face_index} is out of range [0, {self.n_faces - 1}].")
        return ConnectionType(int(self.face_connection_types[face_index]))

    def get_cell_face_indices(self, cell_index: Integer) -> IntArray[OneDimension]:
        """
        Return the indices of all faces belonging to a given cell.

        :param cell_index: 0-based cell index.
        :returns: 1-D array of face indices.
        :raises CellNotFoundError: If `cell_index` is out of range.
        """
        if cell_index < 0 or cell_index >= self.n_cells:
            raise CellNotFoundError(
                f"Cell index {cell_index} is out of range [0, {self.n_cells - 1}]."
            )

        start = self.cell_face_offsets[cell_index]
        end = self.cell_face_offsets[cell_index + 1]
        return typing.cast(IntArray[OneDimension], self.cell_face_indices[start:end])

    def get_cell_neighbor_indices(self, cell_index: Integer) -> IntArray[OneDimension]:
        """
        Return the indices of all face-adjacent neighbours of a given cell.

        :param cell_index: 0-based cell index.
        :returns: 1-D array of neighbouring cell indices.
        :raises CellNotFoundError: If `cell_index` is out of range.
        """
        if cell_index < 0 or cell_index >= self.n_cells:
            raise CellNotFoundError(
                f"Cell index {cell_index} is out of range [0, {self.n_cells - 1}]."
            )

        start = self.cell_neighbor_offsets[cell_index]
        end = self.cell_neighbor_offsets[cell_index + 1]
        return typing.cast(IntArray[OneDimension], self.cell_neighbor_indices[start:end])

    def get_face_vertex_coordinates(self, face_index: Integer) -> NumberArray[TwoDimensions]:
        """
        Return the vertex coordinates of a given face.

        :param face_index: 0-based face index.
        :returns: Shape `(n_verts_for_face, 3)` coordinate array.
        """
        if face_index < 0 or face_index >= self.n_faces:
            raise IndexError(f"Face index {face_index} is out of range [0, {self.n_faces - 1}].")

        start = self.face_vertex_offsets[face_index]
        end = self.face_vertex_offsets[face_index + 1]
        return typing.cast(
            NumberArray[TwoDimensions],
            self.vertex_coordinates[self.face_vertex_indices[start:end]],
        )

    def get_face_cell_indices(self, face_index: Integer) -> IntArray[OneDimension]:
        """
        Return the indices of all cells that share a given face.

        :param face_index: 0-based face index.
        :returns: 1-D array of cell indices (usually of length=2).
        :raises IndexError: If `face_index` is out of range.
        """
        if face_index < 0 or face_index >= self.n_faces:
            raise IndexError(f"Face index {face_index} is out of range [0, {self.n_faces - 1}].")
        return typing.cast(IntArray[OneDimension], self.face_cell_indices[face_index])

    def get_face_normal_for_cell(
        self, face_index: Integer, cell_index: Integer
    ) -> NumberArray[OneDimension]:
        """
        Return the outward unit normal of a face relative to a specific cell.

        :param face_index: 0-based face index.
        :param cell_index: Must be owner or neighbour of `face_index`.
        :returns: Shape `(3,)` unit normal pointing outward from `cell_index`.
        :raises ValidationError: If `cell_index` is not connected to `face_index`.
        """
        if face_index < 0 or face_index >= self.n_faces:
            raise IndexError(f"Face index {face_index} is out of range [0, {self.n_faces - 1}].")

        owner = self.face_cell_indices[face_index, 0]
        neighbour = self.face_cell_indices[face_index, 1]
        if cell_index < 0:
            raise ValidationError(f"Cell {cell_index} is not a valid cell index.")
        if cell_index == owner:
            return self.face_unit_normals[face_index]
        elif cell_index == neighbour:
            return -self.face_unit_normals[face_index]
        raise ValidationError(
            f"Cell {cell_index} is not connected to face {face_index} "
            f"(owner={owner}, neighbour={neighbour})."
        )

    def get_boundary_cell_indices(self) -> IntArray[OneDimension]:
        """Return sorted indices of all cells that touch at least one boundary face."""
        owners = self.face_cell_indices[self.boundary_face_indices, 0]
        neighbours = self.face_cell_indices[self.boundary_face_indices, 1]
        all_boundary = np.concatenate([
            owners[owners >= 0],
            neighbours[neighbours >= 0],
        ])
        return typing.cast(IntArray[OneDimension], np.unique(all_boundary).astype(np.int32))

    def get_interior_cell_indices(self) -> IntArray[OneDimension]:
        """Return sorted indices of all cells that have no boundary faces."""
        boundary_cells = self.get_boundary_cell_indices()
        all_cells = np.arange(self.n_cells, dtype=np.int32)
        return typing.cast(IntArray[OneDimension], np.setdiff1d(all_cells, boundary_cells))

    def is_boundary_cell(self, cell_index: Integer) -> bool:
        """
        Return whether a given cell is adjacent to at least one boundary face.

        :param cell_index: 0-based cell index.
        :raises CellNotFoundError: If `cell_index` is out of range.
        """
        if cell_index < 0 or cell_index >= self.n_cells:
            raise CellNotFoundError(
                f"Cell index {cell_index} is out of range [0, {self.n_cells - 1}]."
            )

        face_indices = self.get_cell_face_indices(cell_index)
        for face_idx in face_indices:
            if self.face_cell_indices[face_idx, 0] < 0 or self.face_cell_indices[face_idx, 1] < 0:
                return True
        return False

    def is_boundary_face(self, face_index: Integer) -> bool:
        """
        Return whether a given face is adjacent to at least one boundary cell.

        :param face_index: 0-based face index.
        :raises IndexError: If `face_index` is out of range.
        """
        face_type = self.get_face_type(face_index)
        return face_type in (
            ConnectionType.BOUNDARY_FACE,
            ConnectionType.BOUNDARY_FAULT_FACE,
        )

    def is_fault_face(self, face_index: Integer) -> bool:
        """
        Return whether a given face belongs to fault.

        :param face_index: 0-based face index.
        :raises IndexError: If `face_index` is out of range.
        """
        face_type = self.get_face_type(face_index)
        return face_type in (
            ConnectionType.INTERIOR_FAULT_FACE,
            ConnectionType.BOUNDARY_FAULT_FACE,
        )

    def get_fault_face_indices(self, fault_name: str) -> IntArray[OneDimension]:
        """
        Return face indices for a named fault.

        :param fault_name: Fault name as declared in `FAULTS`.
        :raises KeyError: If `fault_name` is not found.
        :raises ValidationError: If no fault data is available.
        """
        if self.fault_face_indices is None:
            raise ValidationError(
                "No fault data available on this grid (`fault_face_indices` is None)."
            )

        if fault_name not in self.fault_face_indices:
            available = sorted(self.fault_face_indices.keys())
            raise KeyError(f"Fault {fault_name!r} not found. Available faults: {available}.")
        return self.fault_face_indices[fault_name]

    def get_fault_transmissibility_multiplier(self, fault_name: str) -> Number:
        """
        Return the transmissibility multiplier for a named fault.

        :param fault_name: Fault name as declared in `MULTFLT`.
        :raises KeyError: If `fault_name` is not found.
        :raises ValidationError: If no multiplier data is available.
        """
        if self.fault_transmissibility_multipliers is None:
            raise ValidationError("No fault transmissibility multipliers available on this grid.")
        if fault_name not in self.fault_transmissibility_multipliers:
            available = sorted(self.fault_transmissibility_multipliers.keys())
            raise KeyError(
                f"Fault {fault_name!r} not found in `MULTFLT` data. Available: {available}."
            )
        return self.fault_transmissibility_multipliers[fault_name]

    def get_cell_center_at(
        self, i: Integer, j: Integer, k: Integer
    ) -> tuple[Number, Number, Number]:
        """Return the center coordinates `(x, y, z)`, of cell 0-based index `(i, j, k)`."""
        assert self.cell_centroids is not None
        cell_index = self.flat_index(i, j, k)
        return tuple(self.cell_centroids[cell_index])

    def find_cell_at_position(
        self,
        x: Number,
        y: Number,
        z: Number,
        *,
        max_distance: Number | None = None,
    ) -> tuple[Integer, Integer, Integer]:
        """
        Return the `(i, j, k)` indices of the cell whose center is nearest to
        `(x, y, z)`.

        :param x: Query x-coordinate.
        :param y: Query y-coordinate.
        :param z: Query z-coordinate (positive downward).
        :param max_distance: Maximum allowed distance between the nearest cell
            and the query coordinates.
        :returns: 0-based index of the cell nearest to that position.
        """
        cell_index = self.find_nearest_cell(x, y, z, max_distance=max_distance)
        return self.ijk_index(cell_index)

    find_cell_at_location = find_cell_at_position  # alias

    def find_nearest_cell(
        self,
        x: Number,
        y: Number,
        z: Number,
        *,
        max_distance: Number | None = None,
    ) -> int:
        """
        Find the cell whose centroid is nearest to `(x, y, z)`.

        O(log n) via the pre-built KD-tree.

        :param x: Query x-coordinate.
        :param y: Query y-coordinate.
        :param z: Query z-coordinate (positive downward).
        :param max_distance: Maximum allowed distance between the nearest cell
            and the query coordinates.
        :returns: 0-based index of the nearest cell.
        """
        distance, cell_index = self._spatial_index.query([x, y, z])  # type: ignore
        if max_distance is not None and distance > max_distance:
            raise ValidationError(
                f"No cell center lies within the requested distance ({max_distance})."
            )
        return int(cell_index)

    def find_cells_in_radius(
        self, x: Number, y: Number, z: Number, radius: Number
    ) -> IntArray[OneDimension]:
        """
        Return all cell indices whose centroids fall within `radius` of a point.

        :param x: Query x-coordinate.
        :param y: Query y-coordinate.
        :param z: Query z-coordinate (positive downward).
        :param radius: Search radius in grid length units.
        :returns: 1-D array of matching cell indices.
        """
        raw = self._spatial_index.query_ball_point([x, y, z], r=radius)  # type: ignore
        return typing.cast(IntArray[OneDimension], np.asarray(raw, dtype=np.int32))

    def compute_pore_volume(
        self,
        porosity: NumberOrArray[OneDimension],
        net_to_gross: NumberOrArray[OneDimension],
    ) -> NumberArray[OneDimension]:
        """
        Compute the pore volume for each cell.

        :param porosity: Scalar or shape `(n_cells,)` porosity values in `[0, 1]`.
        :param net_to_gross: Scalar or shape `(n_cells,)` NTG values.
        :returns: Pore volumes in the same units³ as `cell_volumes`.
        """
        assert self.cell_volumes is not None
        return typing.cast(NumberArray[OneDimension], porosity * net_to_gross * self.cell_volumes)

    def validate_geometry(self) -> None:
        """
        Validate that all computed geometry values are physically reasonable.

        :raises InvalidVolumeError: If any active cell volume is <= 0.
        :raises InvalidFaceAreaError: If any face area is negative.
        :raises InvalidNormalVectorError: If any face normal deviates from unit length.
        """
        assert self.cell_volumes is not None
        assert self.cell_statuses is not None
        invalid_volume = ~(self.cell_volumes > 0.0) & (
            self.cell_statuses == int(CellStatus.ACTIVE)
        )
        if invalid_volume.any():
            bad = np.where(invalid_volume)[0]
            raise InvalidVolumeError(
                f"{len(bad)} active cell(s) have non-positive volume: {bad[:5].tolist()}..."
            )
        if (~(self.face_areas >= 0.0)).any():
            bad = np.where(~(self.face_areas >= 0.0))[0]
            raise InvalidFaceAreaError(
                f"{len(bad)} face(s) have negative area: {bad[:5].tolist()}..."
            )

        normal_magnitudes = np.linalg.norm(self.face_unit_normals, axis=1)
        active_mask = self.face_areas > GEOMETRY_TOLERANCE
        if active_mask.any():
            deviation = np.abs(normal_magnitudes[active_mask] - 1.0)
            if (deviation > 1e-10).any():
                raise InvalidNormalVectorError(
                    "One or more face unit normals do not have unit magnitude "
                    f"(max deviation = {deviation.max():.3e})."
                )

    @classmethod
    def from_deck(
        cls,
        deck_file: DeckFile,
        *,
        unit_system: UnitSystem | None = None,
        metadata: typing.Mapping[str, typing.Any] | None = None,
    ) -> Self:
        """
        Build a `Grid` from a parsed `DeckFile`.

        Detects the grid format (corner-point via `COORD`/`ZCORN` or Cartesian
        via `DX`/`DY`/`DZ`) and delegates to the appropriate factory.

        Corner-point grids are built via `make_corner_point_grid`; Cartesian
        grids via `make_cartesian_grid`. Both factories consume the GRDECL
        keywords already parsed in the deck.

        Delegates directly to `load_grdecl`, which handles both corner-point
        and Cartesian grids, fault processing, NNC resolution, and unit
        system detection from the deck.

        :param deck_file: Parsed `DeckFile` containing GRID-section keywords.
        :param unit_system: If provided, convert the grid to this unit system
            after loading. When `None`, the unit system declared in the deck
            is used as-is.
        :param metadata: Optional extra key/value pairs merged into
            `Grid.metadata`.
        :returns: `Grid` for the deck.
        :raises ValidationError: If no recognisable grid keywords are found.
        """
        from bores.grids.io.grdecl import load_grdecl

        return typing.cast(
            Self,
            load_grdecl(source=deck_file, unit_system=unit_system, metadata=metadata),
        )

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Return a new `Grid` expressed in the target unit system.

        Coordinates, volumes and explicit NNC transmissibilities are rescaled. If the grid is
        already in the target unit system, the same object is returned.

        :param target: Target `bores.types.UnitSystem`.
        :param table: Optional precomputed unit conversion table.
        :returns: A new `Grid` in the target unit system, or this grid if already there.

        Example:

        ```python
        from bores.grids.factories.cartesian import make_cartesian_grid
        from bores.types import UnitSystem

        # Build a grid in field units (feet)
        grid_ft = make_cartesian_grid(
            nx=10,
            ny=10,
            nz=5,
            dx=328.084,
            dy=328.084,
            dz=16.4042,  # ≈ 100 m cells
            unit_system=UnitSystem.FIELD,
        )

        # Convert to metric (metres)
        grid_m = grid_ft.convert(UnitSystem.METRIC)
        assert grid_m.unit_system == UnitSystem.METRIC
        # cell volume should now be ≈ 100 * 100 * 5 = 50,000 m³
        ```
        """
        if self.unit_system == target:
            return self

        factors = get_conversion_factors(self.unit_system, target, table=table)
        length_factor = factors["length"]
        volume_factor = length_factor**3
        # Rescale vertex coordinates only.
        # All other geometry is derived and will be recomputed on Grid initialization.
        vertex_coordinates = self.vertex_coordinates * length_factor
        cell_volumes = self.cell_volumes * volume_factor if self.cell_volumes is not None else None
        cell_centroids = (
            self.cell_centroids * length_factor if self.cell_centroids is not None else None
        )
        # Flow transmissibility has units of volumetric rate x viscosity / pressure.
        nnc_transmissibilities = (
            self.nnc_transmissibilities
            * (factors["reservoir_rate"] * factors["viscosity"] / factors["pressure"])
            if self.nnc_transmissibilities is not None
            else None
        )
        metadata = self.metadata
        if metadata:
            metadata = {
                key: (np.asarray(value) * length_factor if key in ("coord", "zcorn") else value)
                for key, value in metadata.items()
            }
        return attrs.evolve(
            self,
            vertex_coordinates=vertex_coordinates,
            cell_volumes=cell_volumes,
            cell_centroids=cell_centroids,
            nnc_transmissibilities=nnc_transmissibilities,
            metadata=metadata,
            unit_system=target,
        )

    def __repr__(self) -> str:
        bbox = self.bounding_box
        fault_info = f", n_faults={self.n_faults}" if self.fault_face_indices else ""
        nnc_info = f", n_nnc={self.n_nnc}" if self.n_nnc > 0 else ""
        return (
            f"{self.__class__.__name__}("
            f"n_cells={self.n_cells}, "
            f"n_faces={self.n_faces}, "
            f"n_interior={self.n_interior_faces}, "
            f"n_boundary={self.n_boundary_faces}"
            f"{nnc_info}"
            f"{fault_info}, "
            f"unit_system={self.unit_system.value!r}, "
            f"bbox=({bbox[0]:.2f}..{bbox[1]:.2f}, {bbox[2]:.2f}..{bbox[3]:.2f}, {bbox[4]:.2f}..{bbox[5]:.2f})"
            f")"
        )
