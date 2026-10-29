"""
Corner-point (pillar) grid factory.

Builds a `bores.grids.base.Grid` from ECLIPSE-style `COORD` / `ZCORN` / `ACTNUM` arrays.

**Coordinate convention**: z-axis positive downward.

**Pinchout handling**: cells whose average thickness is at or below
`pinch_tolerance` have their top/bottom faces suppressed so that adjacent
active cells share those face keys. A face key claimed by a third cell is ignored for
that cell.

**Fault handling**: named faults from `fault_records` are first resolved to
shared face indices. Cell pairs in the fault IJK range that share no geometric
face are not connected and are skipped.
"""

import enum
import logging
import typing
import warnings

import numba
import numpy as np
from typing_extensions import Self

from bores.datastructures import GridDimensions, MapAxes
from bores.errors import GridExportError, InvalidGridError, ValidationError
from bores.grids.base import CellStatus, ConnectionType, Grid
from bores.grids.factories.base import (
    FaultRecord,
    VertexCoordinates,
    map_xy_to_map_space,
    validate_map_axes,
)
from bores.types import (
    Boolean,
    BooleanArray,
    Float,
    IntArray,
    Integer,
    Number,
    NumberArray,
    OneDimension,
    ThreeDimensions,
    TwoDimensions,
    UnitSystem,
)

logger = logging.getLogger(__name__)

__all__ = [
    "InvertedCellPolicy",
    "make_corner_point_grid",
    "rederive_corner_point_arrays",
]


class InvertedCellPolicy(enum.Enum):
    """
    What to do when active cells have inverted (negative volume) geometry.

    `RAISE`
        Refuse to build the grid and report how many cells are inverted.

    `DEACTIVATE`
        Mark the inverted cells inactive, as if `ACTNUM` were zero for them, and warn
        with the number of cells removed.

    `ABSOLUTE`
        Keep the inverted cells active and use the magnitude of their computed volume,
        and warn with the number of cells affected. The face areas, face normals and
        centroids of these cells are not repaired, so transmissibilities across a folded
        cell are only approximate.
    """

    RAISE = "raise"
    DEACTIVATE = "deactivate"
    ABSOLUTE = "absolute"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def _missing_(cls, value: object) -> Self | None:
        lowered = str(value).lower()
        for member in cls:
            if member.value == lowered:
                return member
        return None


CoordArray: typing.TypeAlias = NumberArray[ThreeDimensions]
"""Corner-point COORD array, shape `(NY+1, NX+1, 6)`."""

ZCornArray: typing.TypeAlias = NumberArray[ThreeDimensions]
"""Corner-point ZCORN array, shape `(NZ*2, NY*2, NX*2)`."""

ActNumArray: typing.TypeAlias = IntArray[ThreeDimensions]
"""Corner-point ACTNUM array, shape `(NZ, NY, NX)`; 1 = active."""


HEXAHEDRON_FACES_ZDOWN: list[list[Integer]] = [
    [0, 3, 2, 1],  # top    - outward normal = -z
    [4, 5, 6, 7],  # bottom - outward normal = +z
    [0, 1, 5, 4],  # -y face
    [3, 7, 6, 2],  # +y face
    [0, 4, 7, 3],  # -x face
    [1, 2, 6, 5],  # +x face
]

TOP_FACE_LOCAL: Integer = 0
BOTTOM_FACE_LOCAL: Integer = 1

FACE_DIRECTION_TO_LOCAL: dict[str, Integer] = {
    "X": 5,
    "X-": 4,
    "Y": 3,
    "Y-": 2,
    "Z": 1,
    "Z-": 0,
}


def make_corner_point_grid(
    *,
    coord: CoordArray,
    zcorn: ZCornArray,
    actnum: ActNumArray | None = None,
    vertex_tolerance: Number = 1e-8,
    pinch_tolerance: Number | None = None,
    unit_system: UnitSystem = UnitSystem.FIELD,
    metadata: typing.Mapping[str, typing.Any] | None = None,
    map_axes: MapAxes | None = None,
    apply_map_axes: bool = True,
    on_inverted_cells: InvertedCellPolicy | str = InvertedCellPolicy.RAISE,
    nnc_cell_indices: IntArray[TwoDimensions] | None = None,
    nnc_transmissibilities: NumberArray[OneDimension] | None = None,
    fault_records: typing.Sequence[FaultRecord] | None = None,
    fault_transmissibility_multipliers: typing.Mapping[str, Number] | None = None,
    positive_x_transmissibility_multipliers: NumberArray[OneDimension] | None = None,
    negative_x_transmissibility_multipliers: NumberArray[OneDimension] | None = None,
    positive_y_transmissibility_multipliers: NumberArray[OneDimension] | None = None,
    negative_y_transmissibility_multipliers: NumberArray[OneDimension] | None = None,
    positive_z_transmissibility_multipliers: NumberArray[OneDimension] | None = None,
    negative_z_transmissibility_multipliers: NumberArray[OneDimension] | None = None,
) -> Grid:
    """
    Build a corner-point (pillar) grid from ECLIPSE-style COORD / ZCORN / ACTNUM arrays.

    :param coord: Shape `(NY+1, NX+1, 6)` pillar array.
    :param zcorn: Shape `(NZ*2, NY*2, NX*2)` depth array.
    :param actnum: Shape `(NZ, NY, NX)` integer mask (1=active). All active if `None`.
    :param vertex_tolerance: Merge distance for coincident corner points.
    :param pinch_tolerance: Average thickness threshold for pinch detection.
        Defaults to `metadata['pinch']` or 0.0 (no detection).
    :param unit_system: Declared unit system for coordinate arrays.
    :param metadata: Optional free-form metadata attached to the returned `Grid`.
    :param map_axes: `MAPAXES` to apply to `coord` before deriving geometry.
        Falls back to `metadata['map_axes']` when `None` - a `load_grdecl`
        deck's parsed `MAPAXES` already ends up there, so callers that only
        set it via `metadata` don't need to also pass it here explicitly.
    :param apply_map_axes: When `True` (the default) and a `map_axes` is
        resolved (from this parameter or `metadata`), `coord` is rotated/
        translated into map space before any geometry is derived, so the
        returned `Grid`'s coordinates are already correctly positioned.
        Set `False` to keep the grid in local (pre-`MAPAXES`) space - the
        resolved `map_axes` is still stored on `grid.metadata` either way.
    :param on_inverted_cells: What to do with active cells whose corners fold the cell
        inside out. `"raise"` (the default) refuses to build the grid, `"deactivate"`
        marks those cells inactive and `"absolute"` keeps them with the magnitude of their
        volume. Both of the latter warn with how many cells were affected.
    :param nnc_cell_indices: Shape `(n_nnc, 2)` user-declared NNC cell pairs.
    :param nnc_transmissibilities: Shape `(n_nnc,)` flow transmissibility of each NNC pair.
        Required with `nnc_cell_indices`.
    :param fault_records: `FaultRecord` objects from the GRDECL `FAULTS` keyword.
    :param fault_transmissibility_multipliers: `{name: multiplier}` from `MULTFLT`.
    :param positive_x_transmissibility_multipliers: Per-cell MULTX. `None` if absent.
    :param negative_x_transmissibility_multipliers: Per-cell MULTX-. `None` if absent.
    :param positive_y_transmissibility_multipliers: Per-cell MULTY. `None` if absent.
    :param negative_y_transmissibility_multipliers: Per-cell MULTY-. `None` if absent.
    :param positive_z_transmissibility_multipliers: Per-cell MULTZ. `None` if absent.
    :param negative_z_transmissibility_multipliers: Per-cell MULTZ-. `None` if absent.
    :returns: Fully initialised `Grid`.
    :raises ValidationError: On array shape mismatches, inconsistent NNC lengths, or `MAPAXES`
        axes that are not perpendicular.
    :raises InvalidGridError: If no active cells are found, or if active cells are inverted
        and `on_inverted_cells` is `"raise"`.
    """
    if (
        nnc_cell_indices is not None
        and len(nnc_cell_indices) > 0
        and nnc_transmissibilities is None
    ):
        raise ValidationError("`nnc_cell_indices` was given without `nnc_transmissibilities`.")
    if nnc_cell_indices is not None and nnc_transmissibilities is not None:
        if len(nnc_cell_indices) != len(nnc_transmissibilities):
            raise ValidationError(
                f"`nnc_cell_indices` has {len(nnc_cell_indices)} rows but "
                f"`nnc_transmissibilities` has {len(nnc_transmissibilities)} entries."
            )

    coord_array = np.asarray(coord, dtype=np.float64)
    zcorn_array = np.asarray(zcorn, dtype=np.float64)

    resolved_map_axes = map_axes if map_axes is not None else (metadata or {}).get("map_axes")
    if resolved_map_axes is not None:
        # `MAPUNITS` (map_axes' own unit_system) can differ from GRIDUNIT
        # (this grid's unit_system) so we normalise once, upfront, so both the
        # applied transform and the stored metadata are self-consistent.
        resolved_map_axes = resolved_map_axes.convert(unit_system)

    if resolved_map_axes is not None and apply_map_axes:
        validate_map_axes(resolved_map_axes)
        coord_array = apply_map_axes_to_coord(coord_array, map_axes=resolved_map_axes)  # type: ignore[arg-type]

    if resolved_map_axes is not None:
        # Keep grid.metadata['map_axes'] consistent with whatever was
        # actually resolved above, even if an explicit `map_axes` argument
        # differed from (or `metadata` didn't yet have) one.
        metadata = {**(metadata or {}), "map_axes": resolved_map_axes}

    if coord_array.ndim != 3 or coord_array.shape[2] != 6:
        raise ValidationError(
            f"`coord` must have shape (NY+1, NX+1, 6); got {coord_array.shape!r}."
        )
    if zcorn_array.ndim != 3:
        raise ValidationError(f"`zcorn` must be a 3-D array; got ndim={zcorn_array.ndim}.")

    ny_plus1, nx_plus1 = coord_array.shape[:2]
    nx = nx_plus1 - 1
    ny = ny_plus1 - 1
    nz = zcorn_array.shape[0] // 2

    if zcorn_array.shape != (nz * 2, ny * 2, nx * 2):
        raise ValidationError(
            f"`zcorn` shape {zcorn_array.shape!r} is inconsistent with "
            f"`coord`-derived grid dimensions ({nx} x {ny} x {nz})."
        )

    if actnum is None:
        actnum_array = typing.cast(ActNumArray, np.ones((nz, ny, nx), dtype=np.int32))
    else:
        actnum_array = typing.cast(ActNumArray, np.asarray(actnum, dtype=np.int32))
        if actnum_array.shape != (nz, ny, nx):
            raise ValidationError(
                f"`actnum` shape {actnum_array.shape!r} does not match "
                f"grid dimensions ({nx} x {ny} x {nz})."
            )

    if pinch_tolerance is None:
        pinch_tolerance = float((metadata or {}).get("pinch", None) or 0.0)

    inverted_cell_policy = InvertedCellPolicy(on_inverted_cells)
    # A `MAPAXES` whose Y axis points against the right-handed direction (the usual
    # Eclipse convention, with Y pointing south) reflects the grid when it is applied.
    mirrored = (
        resolved_map_axes is not None
        and apply_map_axes
        and np.linalg.det(resolved_map_axes.rotation_matrix) < 0
    )

    (
        vertex_coordinates,
        face_vertex_indices,
        face_vertex_offsets,
        face_cell_indices,
        face_connection_types,
        cell_statuses,
        cell_volumes,
        cell_centroids,
        cell_min_xyz,
        cell_max_xyz,
    ) = compute_corner_point_geometry(
        coord=coord_array,  # type: ignore[arg-type]
        zcorn=zcorn_array,  # type: ignore[arg-type]
        actnum=actnum_array,
        vertex_tolerance=vertex_tolerance,
        pinch_tolerance=pinch_tolerance,
        mirrored=mirrored,
        on_inverted_cells=inverted_cell_policy,
    )
    if inverted_cell_policy is InvertedCellPolicy.DEACTIVATE:
        actnum_array = typing.cast(
            ActNumArray,
            (cell_statuses == int(CellStatus.ACTIVE)).reshape(nz, ny, nx).astype(np.int32),
        )

    # Resolve fault face indices; cell pairs with no shared face are skipped.
    fault_face_indices: dict[str, IntArray[OneDimension]] | None = None
    if fault_records:
        fault_face_indices = resolve_fault_face_indices(
            fault_records=fault_records,
            dimensions=(nx, ny, nz),
            actnum=actnum_array,
            face_cell_indices=face_cell_indices,
        )
        for face_indices in fault_face_indices.values():
            boundary_fault_mask = (face_cell_indices[face_indices, 0] < 0) | (
                face_cell_indices[face_indices, 1] < 0
            )
            boundary_fault_faces = face_indices[boundary_fault_mask]
            interior_fault_faces = face_indices[~boundary_fault_mask]

            face_connection_types[interior_fault_faces] = int(ConnectionType.INTERIOR_FAULT_FACE)
            face_connection_types[boundary_fault_faces] = int(ConnectionType.BOUNDARY_FAULT_FACE)

    nnc_pairs: IntArray[OneDimension] | None = None
    nnc_flow_transmissibilities: NumberArray[OneDimension] | None = None
    if nnc_cell_indices is not None and cell_statuses is not None and len(nnc_cell_indices) > 0:
        pairs = typing.cast(
            IntArray[OneDimension], np.asarray(nnc_cell_indices, dtype=np.int32).reshape(-1, 2)
        )
        flow_transmissibilities = np.asarray(nnc_transmissibilities, dtype=np.float64)
        if len(flow_transmissibilities) != len(pairs):
            raise ValidationError(
                f"`nnc_cell_indices` has {len(pairs)} pairs but `nnc_transmissibilities` "
                f"has {len(flow_transmissibilities)} values."
            )

        if pairs.min() < 0 or pairs.max() >= cell_statuses.shape[0]:
            raise ValidationError(
                f"`nnc_cell_indices` must lie in [0, {cell_statuses.shape[0] - 1}] "
                f"(flat index i + j * nx + k * nx * ny)."
            )

        keep = (cell_statuses[pairs[:, 0]] == int(CellStatus.ACTIVE)) & (
            cell_statuses[pairs[:, 1]] == int(CellStatus.ACTIVE)
        )
        if not keep.all():
            warnings.warn(
                f"Dropped {int((~keep).sum())} user NNC(s) connected to inactive cells.",
                stacklevel=3,
            )
        if keep.any():
            nnc_pairs = typing.cast(IntArray[OneDimension], pairs[keep])
            nnc_flow_transmissibilities = typing.cast(
                NumberArray[OneDimension], flow_transmissibilities[keep]
            )

    return Grid(
        vertex_coordinates=vertex_coordinates,
        face_vertex_indices=face_vertex_indices,
        face_vertex_offsets=face_vertex_offsets,
        face_cell_indices=face_cell_indices,
        cell_volumes=cell_volumes,
        cell_centroids=cell_centroids,
        cell_min_xyz=cell_min_xyz,
        cell_max_xyz=cell_max_xyz,
        unit_system=unit_system,
        dimensions=GridDimensions(nx, ny, nz),
        metadata=metadata,
        cell_statuses=cell_statuses,  # type: ignore[arg-type]
        face_connection_types=face_connection_types,  # type: ignore[arg-type]
        nnc_cell_indices=nnc_pairs,  # type: ignore[arg-type]
        nnc_transmissibilities=nnc_flow_transmissibilities,  # type: ignore[arg-type]
        fault_face_indices=fault_face_indices,
        fault_transmissibility_multipliers=(
            dict(fault_transmissibility_multipliers)
            if fault_transmissibility_multipliers is not None
            else None
        ),
        positive_x_transmissibility_multipliers=positive_x_transmissibility_multipliers,
        negative_x_transmissibility_multipliers=negative_x_transmissibility_multipliers,
        positive_y_transmissibility_multipliers=positive_y_transmissibility_multipliers,
        negative_y_transmissibility_multipliers=negative_y_transmissibility_multipliers,
        positive_z_transmissibility_multipliers=positive_z_transmissibility_multipliers,
        negative_z_transmissibility_multipliers=negative_z_transmissibility_multipliers,
    )


def get_map_axes_xy_inverse(
    xy: NumberArray[TwoDimensions], map_axes: MapAxes
) -> NumberArray[TwoDimensions]:
    """
    Map `(x, y)` pairs from map space back to local (pre-`MAPAXES`) space.

    Uses `numpy.linalg.inv` rather than assuming `rotation_matrix` is
    orthonormal (it's built from two independently-normalised axis
    direction vectors, so a deck with non-perpendicular `MAPAXES` axes -
    unusual, but not rejected at parse time - would make the transpose an
    incorrect inverse).

    :param xy: Shape `(n, 2)` map-space points.
    :param map_axes: Map axes to invert.
    :returns: Shape `(n, 2)` local-space points.
    """
    inverse_rotation = np.linalg.inv(map_axes.rotation_matrix)
    return typing.cast(NumberArray[TwoDimensions], (xy - map_axes.origin) @ inverse_rotation.T)


def apply_map_axes_to_coord(coord: CoordArray, map_axes: MapAxes) -> CoordArray:
    """
    Rotate, translate and, when the Y axis points the other way, reflect a COORD pillar
    array's `(x, y)` pairs into map space.

    Applied once, before pillar interpolation, so the vertex coordinates, cell centroids
    and face geometry all come out positioned in map space. Pillar depth (columns 2 and
    5) is untouched. A reflection turns the grid into a mirror image, which
    `compute_corner_point_geometry` accounts for through its `mirrored` argument so that
    cell volumes stay positive.

    :param coord: Shape `(NY+1, NX+1, 6)` - `[x_top, y_top, z_top,
        x_bottom, y_bottom, z_bottom]` per pillar, in local (pre-`MAPAXES`)
        space.
    :param map_axes: Map axes to apply.
    :returns: New array of the same shape, with `x`/`y` columns mapped.
    """
    rotated = coord.copy()
    shape_xy = (*coord.shape[:-1], 2)
    top_xy = coord[..., 0:2].reshape(-1, 2)
    bottom_xy = coord[..., 3:5].reshape(-1, 2)
    rotated[..., 0:2] = map_xy_to_map_space(top_xy, map_axes).reshape(shape_xy)
    rotated[..., 3:5] = map_xy_to_map_space(bottom_xy, map_axes).reshape(shape_xy)
    return rotated


@numba.njit(cache=True)
def _interpolate_pillar_point(
    pillar_top: NumberArray[OneDimension],
    pillar_bottom: NumberArray[OneDimension],
    z: Number,
) -> NumberArray[OneDimension]:
    """
    Interpolate an (x, y, z) position along a pillar at depth `z`, extrapolating along the
    pillar line when `z` lies outside its stored extent.

    :param pillar_top: Shape `(3,)` - `[x, y, z]` of pillar top.
    :param pillar_bottom: Shape `(3,)` - `[x, y, z]` of pillar bottom.
    :param z: Target depth.
    :returns: Shape `(3,)` point on the pillar.
    """
    xyz = np.empty(3, dtype=np.float64)
    dz = pillar_bottom[2] - pillar_top[2]
    if abs(dz) < 1e-14:
        xyz[0] = pillar_top[0]
        xyz[1] = pillar_top[1]
        xyz[2] = z
        return xyz
    t = (z - pillar_top[2]) / dz
    xyz[0] = pillar_top[0] + t * (pillar_bottom[0] - pillar_top[0])
    xyz[1] = pillar_top[1] + t * (pillar_bottom[1] - pillar_top[1])
    xyz[2] = z
    return xyz


@numba.njit(parallel=True, cache=True)
def compute_cell_corner_coordinates(
    cells: IntArray[TwoDimensions],
    coord: CoordArray,
    zcorn: ZCornArray,
) -> NumberArray[ThreeDimensions]:
    """
    Compute the eight corner coordinates of each given cell.

    Corner layout (index 0..7):

    ```md
    ==  =========  ========================
    0   (j,  i  )  zcorn[2k,   2j,   2i  ]
    1   (j,  i+1)  zcorn[2k,   2j,   2i+1]
    2   (j+1,i  )  zcorn[2k,   2j+1, 2i  ]
    3   (j+1,i+1)  zcorn[2k,   2j+1, 2i+1]
    4   (j,  i  )  zcorn[2k+1, 2j,   2i  ]
    5   (j,  i+1)  zcorn[2k+1, 2j,   2i+1]
    6   (j+1,i  )  zcorn[2k+1, 2j+1, 2i  ]
    7   (j+1,i+1)  zcorn[2k+1, 2j+1, 2i+1]
    ==  =========  ========================
    ```

    :param cells: Shape `(n_cells, 3)` - `(k, j, i)` index of each cell.
    :param coord: Shape `(NY+1, NX+1, 6)` pillar array.
    :param zcorn: Shape `(NZ*2, NY*2, NX*2)` depth array.
    :returns: Shape `(n_cells, 8, 3)` corner coordinate array.
    """
    n_cells = cells.shape[0]
    corners = np.empty((n_cells, 8, 3), dtype=np.float64)
    pillar_order = [0, 1, 2, 3, 0, 1, 2, 3]

    for cell_idx in numba.prange(n_cells):  # type: ignore
        k = cells[cell_idx, 0]
        j = cells[cell_idx, 1]
        i = cells[cell_idx, 2]

        pt = np.empty((4, 3), dtype=np.float64)
        pb = np.empty((4, 3), dtype=np.float64)
        pt[0] = coord[j, i, :3]
        pb[0] = coord[j, i, 3:]
        pt[1] = coord[j, i + 1, :3]
        pb[1] = coord[j, i + 1, 3:]
        pt[2] = coord[j + 1, i, :3]
        pb[2] = coord[j + 1, i, 3:]
        pt[3] = coord[j + 1, i + 1, :3]
        pb[3] = coord[j + 1, i + 1, 3:]

        zv = np.empty(8, dtype=np.float64)
        zv[0] = zcorn[2 * k, 2 * j, 2 * i]
        zv[1] = zcorn[2 * k, 2 * j, 2 * i + 1]
        zv[2] = zcorn[2 * k, 2 * j + 1, 2 * i]
        zv[3] = zcorn[2 * k, 2 * j + 1, 2 * i + 1]
        zv[4] = zcorn[2 * k + 1, 2 * j, 2 * i]
        zv[5] = zcorn[2 * k + 1, 2 * j, 2 * i + 1]
        zv[6] = zcorn[2 * k + 1, 2 * j + 1, 2 * i]
        zv[7] = zcorn[2 * k + 1, 2 * j + 1, 2 * i + 1]

        for c in range(8):
            p = pillar_order[c]
            xyz = _interpolate_pillar_point(pt[p], pb[p], zv[c])
            corners[cell_idx, c, 0] = xyz[0]
            corners[cell_idx, c, 1] = xyz[1]
            corners[cell_idx, c, 2] = xyz[2]

    return corners


FACE_CORNER_TABLE: IntArray[TwoDimensions] = typing.cast(
    IntArray[TwoDimensions], np.array(HEXAHEDRON_FACES_ZDOWN, dtype=np.int64)
)
"""Shape `(6, 4)` local hexahedron corner indices (VTK order) of each local face."""

VTK_TO_CORNER_ORDER: IntArray[OneDimension] = np.array([0, 1, 3, 2, 4, 5, 7, 6], dtype=np.int64)  # type: ignore[assignment]
"""Maps VTK hexahedron corner order to the corner order of `compute_cell_corner_coordinates`."""

LATERAL_FACE_CORNERS: IntArray[TwoDimensions] = np.array(  # type: ignore[assignment]
    [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 4, 5],  # -y
        [2, 3, 6, 7],  # +y
        [0, 2, 4, 6],  # -x
        [1, 3, 5, 7],  # +x
    ],
    dtype=np.int64,
)
"""Shape `(6, 4)` corner indices (`compute_cell_corner_coordinates` order) spanning each lateral face."""


@numba.njit(cache=True)
def is_cell_pinched(
    vtk_vertices: IntArray[OneDimension],
    vertex_coordinates: VertexCoordinates,
    pinch_tolerance: Number,
) -> Boolean:
    """
    Return `True` if the cell's average thickness is at or below `pinch_tolerance`, or if
    its top and bottom share the same four vertices.

    :param vtk_vertices: Shape `(8,)` global vertex indices in VTK hexahedron order.
    :param vertex_coordinates: Shape `(n_verts, 3)` coordinate array.
    :param pinch_tolerance: Thickness threshold.
    :returns: `True` if the cell should be treated as pinched out.
    """
    same_vertices = True
    for top in range(4):
        found = False
        for bottom in range(4, 8):
            if vtk_vertices[top] == vtk_vertices[bottom]:
                found = True
                break
        if not found:
            same_vertices = False
            break
    if same_vertices:
        for bottom in range(4, 8):
            found = False
            for top in range(4):
                if vtk_vertices[top] == vtk_vertices[bottom]:
                    found = True
                    break
            if not found:
                same_vertices = False
                break
    if same_vertices:
        return True
    if pinch_tolerance <= 0.0:
        return False

    total_dz = 0.0
    for k in range(4):
        z_top = vertex_coordinates[vtk_vertices[k], 2]
        z_bottom = vertex_coordinates[vtk_vertices[k + 4], 2]
        total_dz += abs(z_bottom - z_top)
    return (total_dz / 4.0) <= pinch_tolerance


MINIMUM_RELATIVE_OVERLAP_AREA: Number = 1e-6
"""Overlaps smaller than this fraction of the smaller of the two facing faces are ignored."""


@numba.njit(cache=True)
def compute_polygon_signed_area(polygon: NumberArray[TwoDimensions]) -> Float:
    """
    Signed area of a 2-D polygon (positive when the vertices run counter-clockwise).

    :param polygon: Shape `(n, 2)` polygon vertices.
    :returns: Signed area.
    """
    total = 0.0
    n_vertices = polygon.shape[0]
    for index in range(n_vertices):
        following = (index + 1) % n_vertices
        total += polygon[index, 0] * polygon[following, 1]
        total -= polygon[following, 0] * polygon[index, 1]
    return 0.5 * total


@numba.njit(cache=True)
def clip_half_plane(
    polygon: NumberArray[TwoDimensions],
    a: NumberArray[OneDimension],
    b: NumberArray[OneDimension],
    keep_inside: Boolean,
) -> NumberArray[TwoDimensions]:
    """
    Clip a convex 2-D polygon against the half-plane to the left (or right) of the line `a -> b`.

    :param polygon: Shape `(n, 2)` polygon vertices.
    :param a: Shape `(2,)` first point of the line.
    :param b: Shape `(2,)` second point of the line.
    :param keep_inside: Keep the part to the left of `a -> b` when `True`, otherwise the part
        to its right.
    :returns: Shape `(m, 2)` vertices of the clipped polygon, with `m = 0` if nothing remains.
    """
    n_vertices = polygon.shape[0]
    output = np.empty((2 * n_vertices + 2, 2), dtype=np.float64)
    if n_vertices == 0:
        return output[:0]  # type: ignore[return-value]

    sign = 1.0 if keep_inside else -1.0
    count = 0
    previous = n_vertices - 1
    previous_distance = sign * (
        (b[0] - a[0]) * (polygon[previous, 1] - a[1])
        - (b[1] - a[1]) * (polygon[previous, 0] - a[0])
    )
    for current in range(n_vertices):
        current_distance = sign * (
            (b[0] - a[0]) * (polygon[current, 1] - a[1])
            - (b[1] - a[1]) * (polygon[current, 0] - a[0])
        )
        crosses = (current_distance >= 0.0) != (previous_distance >= 0.0)
        if crosses:
            t = previous_distance / (previous_distance - current_distance)
            output[count, 0] = polygon[previous, 0] + t * (
                polygon[current, 0] - polygon[previous, 0]
            )
            output[count, 1] = polygon[previous, 1] + t * (
                polygon[current, 1] - polygon[previous, 1]
            )
            count += 1
        if current_distance >= 0.0:
            output[count, 0] = polygon[current, 0]
            output[count, 1] = polygon[current, 1]
            count += 1
        previous = current
        previous_distance = current_distance
    return output[:count]  # type: ignore[return-value]


@numba.njit(cache=True)
def clip_convex_polygon(
    subject: NumberArray[TwoDimensions],
    clip: NumberArray[TwoDimensions],
) -> NumberArray[TwoDimensions]:
    """
    Intersect two convex 2-D polygons (Sutherland-Hodgman clipping).

    :param subject: Shape `(n, 2)` polygon to clip, vertices counter-clockwise.
    :param clip: Shape `(m, 2)` convex clipping polygon, vertices counter-clockwise.
    :returns: Vertices of the intersection, with zero rows if the polygons do not overlap.
    """
    output = subject
    n_edges = clip.shape[0]
    for index in range(n_edges):
        output = clip_half_plane(
            polygon=output,
            a=clip[index],
            b=clip[(index + 1) % n_edges],
            keep_inside=True,
        )
        if output.shape[0] == 0:
            break
    return output


@numba.njit(cache=True)
def subtract_convex_polygon_pieces(
    subject: NumberArray[TwoDimensions],
    clip: NumberArray[TwoDimensions],
) -> list[NumberArray[TwoDimensions]]:
    """
    Remove a convex polygon from a convex 2-D polygon.

    :param subject: Shape `(n, 2)` polygon to cut, vertices counter-clockwise.
    :param clip: Shape `(m, 2)` convex polygon to remove, vertices counter-clockwise.
    :returns: Disjoint pieces of `subject` that lie outside `clip`.
    """
    pieces = []
    remaining = subject
    n_edges = clip.shape[0]
    for index in range(n_edges):
        a = clip[index]
        b = clip[(index + 1) % n_edges]
        outside = clip_half_plane(polygon=remaining, a=a, b=b, keep_inside=False)
        if outside.shape[0] >= 3:
            pieces.append(outside)
        remaining = clip_half_plane(polygon=remaining, a=a, b=b, keep_inside=True)
        if remaining.shape[0] < 3:
            break
    return pieces


def subtract_convex_polygon(
    subject: NumberArray[TwoDimensions],
    clip: NumberArray[TwoDimensions],
) -> list[NumberArray[TwoDimensions]]:
    """
    Remove a convex polygon from a convex 2-D polygon.

    :param subject: Shape `(n, 2)` polygon to cut, vertices counter-clockwise.
    :param clip: Shape `(m, 2)` convex polygon to remove, vertices counter-clockwise.
    :returns: Disjoint pieces of `subject` that lie outside `clip`.
    """
    pieces = subtract_convex_polygon_pieces(
        subject=typing.cast(
            NumberArray[TwoDimensions], np.ascontiguousarray(subject, dtype=np.float64)
        ),
        clip=typing.cast(NumberArray[TwoDimensions], np.ascontiguousarray(clip, dtype=np.float64)),
    )
    return list(pieces)


@numba.njit(cache=True)
def _get_point_on_pillar_pair(
    pillar_1: NumberArray[OneDimension],
    pillar_2: NumberArray[OneDimension],
    s_value: Number,
    z_value: Number,
) -> NumberArray[OneDimension]:
    """
    Point at fractional position `s_value` between two pillars, at depth `z_value`.

    :param pillar_1: Shape `(6,)` top and bottom points of the first pillar.
    :param pillar_2: Shape `(6,)` top and bottom points of the second pillar.
    :param s_value: Fraction of the way from the first pillar to the second.
    :param z_value: Depth of the point.
    :returns: Shape `(3,)` coordinates.
    """
    p1 = _interpolate_pillar_point(pillar_1[:3], pillar_1[3:], z_value)  # type: ignore[arg-type]
    p2 = _interpolate_pillar_point(pillar_2[:3], pillar_2[3:], z_value)  # type: ignore[arg-type]
    return p1 + s_value * (p2 - p1)  # type: ignore[return-value]


class FaceRegistry(typing.NamedTuple):
    """Faces of a corner-point grid that were matched by exact vertex ids, stored as arrays."""

    n_faces: Integer
    """Number of registered faces."""

    vertices: IntArray[TwoDimensions]
    """Shape `(n_faces, 4)` vertex ids of each face, wound outward from its owner."""

    owner_cells: IntArray[OneDimension]
    """Shape `(n_faces,)` flat index of the cell that registered the face first."""

    neighbour_cells: IntArray[OneDimension]
    """Shape `(n_faces,)` flat index of the second cell on the face, or `-1` for a boundary face."""

    local_indices: IntArray[OneDimension]
    """Shape `(n_faces,)` local face index (see `HEXAHEDRON_FACES_ZDOWN`) of the face on its owner."""

    degenerate_cells: IntArray[OneDimension]
    """Flat indices of cells owning a lateral or vertical face that collapsed to fewer than four vertices."""

    degenerate_local_indices: IntArray[OneDimension]
    """Local face index of each collapsed face, parallel to `degenerate_cells`."""

    pinched: BooleanArray[OneDimension]
    """Shape `(n_cells,)` flag of the cells that are pinched out."""

    n_ignored_third_claims: Integer
    """Number of faces claimed by a third cell and ignored."""


class NewFaces(typing.NamedTuple):
    """Faces created by `connect_unmatched_faces`, stored as arrays."""

    n_faces: Integer
    """Number of new faces."""

    points: NumberArray[TwoDimensions]
    """Shape `(n_points, 3)` coordinates of the new vertices; every new face owns its own vertices."""

    point_offsets: IntArray[OneDimension]
    """Shape `(n_faces + 1,)` offsets into `points` delimiting the vertices of each new face."""

    owner_cells: IntArray[OneDimension]
    """Shape `(n_faces,)` flat index of the owner cell of each new face."""

    neighbour_cells: IntArray[OneDimension]
    """Shape `(n_faces,)` flat index of the neighbour of each new face, `-1` for a boundary face."""

    replaced: BooleanArray[OneDimension]
    """Shape `(registry.n_faces,)` flag of the registered faces that the new faces supersede."""


@numba.njit(cache=True)
def register_cell_faces(
    corner_global: IntArray[TwoDimensions],
    active_cells: IntArray[OneDimension],
    vertex_coordinates: VertexCoordinates,
    pinch_tolerance: Number,
    mirrored: Boolean,
    face_corner_table: IntArray[TwoDimensions],
    vtk_to_corner: IntArray[OneDimension],
) -> tuple[
    Integer,
    IntArray[TwoDimensions],
    IntArray[OneDimension],
    IntArray[OneDimension],
    IntArray[OneDimension],
    IntArray[OneDimension],
    IntArray[OneDimension],
    BooleanArray[OneDimension],
    Integer,
]:
    """
    Register the six faces of every active cell, matching faces shared by two cells through
    their sorted vertex ids with an open-addressing hash table.

    The first cell to claim a face owns it, the second becomes its neighbour and any later
    claim is ignored. Cells are visited in the order of `active_cells`, and faces in local
    face order, so face numbering is deterministic. Collapsed (degenerate) faces are
    recorded separately and top and bottom faces of pinched cells are not registered.

    :param corner_global: Shape `(n_cells, 8)` global vertex id of each cell corner.
    :param active_cells: Shape `(n_active,)` ascending flat indices of the active cells.
    :param vertex_coordinates: Shape `(n_verts, 3)` merged vertex coordinates.
    :param pinch_tolerance: Average-thickness threshold for pinch detection.
    :param mirrored: Whether to reverse face winding for a mirrored coordinate frame.
    :param face_corner_table: Shape `(6, 4)` local corners of each hexahedron face.
    :param vtk_to_corner: Shape `(8,)` map from VTK corner order to `corner_global` order.
    :returns: `(n_faces, vertices, owners, neighbours, local_indices, degenerate_cells,
        degenerate_local_indices, pinched, n_ignored_third_claims)`, the first five trimmed
        to the registered faces.
    """
    n_total_cells = corner_global.shape[0]
    n_active = active_cells.shape[0]
    max_faces = 6 * n_active

    vertices = np.empty((max_faces, 4), dtype=np.int32)
    keys = np.empty((max_faces, 4), dtype=np.int32)
    owners = np.empty(max_faces, dtype=np.int32)
    neighbours = np.empty(max_faces, dtype=np.int32)
    local_indices = np.empty(max_faces, dtype=np.int8)
    degenerate_cells = np.empty(max_faces, dtype=np.int32)
    degenerate_locals = np.empty(max_faces, dtype=np.int8)
    pinched = np.zeros(n_total_cells, dtype=np.bool_)

    capacity = 16
    while capacity < 2 * max_faces:
        capacity *= 2
    mask = capacity - 1
    table = np.full(capacity, -1, dtype=np.int32)

    n_faces = 0
    n_degenerate = 0
    n_third_claims = 0
    vtk = np.empty(8, dtype=np.int64)
    face = np.empty(4, dtype=np.int64)
    key = np.empty(4, dtype=np.int64)

    for position in range(n_active):
        cell = active_cells[position]
        for v in range(8):
            vtk[v] = corner_global[cell, vtk_to_corner[v]]
        cell_pinched = is_cell_pinched(
            vtk_vertices=vtk,
            vertex_coordinates=vertex_coordinates,
            pinch_tolerance=pinch_tolerance,
        )
        if cell_pinched:
            pinched[cell] = True

        for local in range(6):
            for q in range(4):
                face[q] = vtk[face_corner_table[local, q]]
            if mirrored:
                face[0], face[3] = face[3], face[0]
                face[1], face[2] = face[2], face[1]

            vertical = local == TOP_FACE_LOCAL or local == BOTTOM_FACE_LOCAL
            collapsed = (
                face[0] == face[1]
                or face[0] == face[2]
                or face[0] == face[3]
                or face[1] == face[2]
                or face[1] == face[3]
                or face[2] == face[3]
            )
            if collapsed:
                if not vertical or not cell_pinched:
                    degenerate_cells[n_degenerate] = cell
                    degenerate_locals[n_degenerate] = local
                    n_degenerate += 1
                continue

            if cell_pinched and vertical:
                continue

            for q in range(4):
                key[q] = face[q]
            for q in range(1, 4):
                value = key[q]
                p = q - 1
                while p >= 0 and key[p] > value:
                    key[p + 1] = key[p]
                    p -= 1
                key[p + 1] = value

            h = np.int64(1469598103934665603)
            for q in range(4):
                h = (h ^ key[q]) * np.int64(1099511628211)
            h = h ^ (h >> 29)
            slot = h & mask
            while True:
                existing = table[slot]
                if existing < 0:
                    table[slot] = n_faces
                    for q in range(4):
                        vertices[n_faces, q] = face[q]
                        keys[n_faces, q] = key[q]
                    owners[n_faces] = cell
                    neighbours[n_faces] = -1
                    local_indices[n_faces] = local
                    n_faces += 1
                    break
                if (
                    keys[existing, 0] == key[0]
                    and keys[existing, 1] == key[1]
                    and keys[existing, 2] == key[2]
                    and keys[existing, 3] == key[3]
                ):
                    if neighbours[existing] == -1:
                        neighbours[existing] = cell
                    else:
                        n_third_claims += 1
                    break
                slot = (slot + 1) & mask

    return (  # type: ignore[return-value]
        n_faces,
        vertices[:n_faces],
        owners[:n_faces],
        neighbours[:n_faces],
        local_indices[:n_faces],
        degenerate_cells[:n_degenerate],
        degenerate_locals[:n_degenerate],
        pinched,
        n_third_claims,
    )


INITIAL_POLYGON_CAPACITY = 64
"""Initial rows of each polygon scratch buffer used by the overlap kernel; doubled on overflow."""

INITIAL_POLYGON_PIECES = 256
"""Initial number of pieces one lateral face may be split into; doubled on overflow."""

MAX_POLYGON_CAPACITY = 8192
"""Polygon rows beyond which an overlap is considered too intricate to resolve."""


@numba.njit(cache=True, inline="always")
def _kernel_failure(
    status: Integer,
    new_points: NumberArray[TwoDimensions],
    new_point_offsets: IntArray[OneDimension],
    new_owners: IntArray[OneDimension],
    new_neighbours: IntArray[OneDimension],
    replaced: BooleanArray[OneDimension],
) -> tuple[
    Integer,
    Integer,
    Integer,
    NumberArray[TwoDimensions],
    IntArray[OneDimension],
    IntArray[OneDimension],
    IntArray[OneDimension],
    BooleanArray[OneDimension],
]:
    """Empty kernel result carrying a non-zero `status`."""
    return (  # type: ignore[return-value]
        status,
        0,
        0,
        new_points[:0],
        new_point_offsets[:1],
        new_owners[:0],
        new_neighbours[:0],
        replaced,
    )


@numba.njit(cache=True)
def build_lateral_polygon(
    corner_coordinates: NumberArray[ThreeDimensions],
    cell: Integer,
    local_index: Integer,
    lateral_face_corners: IntArray[TwoDimensions],
    polygon: NumberArray[TwoDimensions],
) -> None:
    """
    Write the polygon of a lateral face in the `(s, depth)` plane spanned by its two pillars,
    wound counter-clockwise, into `polygon`.

    :param corner_coordinates: Shape `(n_cells, 8, 3)` corner coordinates of every cell.
    :param cell: Flat index of the cell.
    :param local_index: Local index of the lateral face (2 to 5).
    :param lateral_face_corners: Shape `(6, 4)` corners spanning each lateral face.
    :param polygon: Shape `(>= 4, 2)` output buffer; only the first four rows are written.
    """
    c0 = lateral_face_corners[local_index, 0]
    c1 = lateral_face_corners[local_index, 1]
    c2 = lateral_face_corners[local_index, 2]
    c3 = lateral_face_corners[local_index, 3]
    polygon[0, 0] = 0.0
    polygon[0, 1] = corner_coordinates[cell, c0, 2]
    polygon[1, 0] = 1.0
    polygon[1, 1] = corner_coordinates[cell, c1, 2]
    polygon[2, 0] = 1.0
    polygon[2, 1] = corner_coordinates[cell, c3, 2]
    polygon[3, 0] = 0.0
    polygon[3, 1] = corner_coordinates[cell, c2, 2]
    if compute_polygon_signed_area(polygon[:4]) < 0.0:  # type: ignore[arg-type]
        for axis in range(2):
            polygon[0, axis], polygon[3, axis] = polygon[3, axis], polygon[0, axis]
            polygon[1, axis], polygon[2, axis] = polygon[2, axis], polygon[1, axis]


@numba.njit(cache=True)
def clip_half_plane_into(
    polygon: NumberArray[TwoDimensions],
    n_vertices: Integer,
    a: NumberArray[OneDimension],
    b: NumberArray[OneDimension],
    keep_inside: Boolean,
    output: NumberArray[TwoDimensions],
) -> Integer:
    """
    Allocation-free `clip_half_plane`: clip the first `n_vertices` rows of `polygon` against
    the half-plane to the left (or right) of `a -> b` and write the result to `output`.

    :param polygon: Shape `(>= n_vertices, 2)` polygon vertices.
    :param n_vertices: Number of valid rows of `polygon`.
    :param a: Shape `(2,)` first point of the line.
    :param b: Shape `(2,)` second point of the line.
    :param keep_inside: Keep the part to the left of `a -> b` when `True`, else the right.
    :param output: Shape `(>= 2 * n_vertices + 2, 2)` output buffer, distinct from `polygon`.
    :returns: Number of vertices written to `output`, or `-1` if `output` is too small.
    """
    if 2 * n_vertices + 2 > output.shape[0]:
        return -1
    if n_vertices == 0:
        return 0

    sign = 1.0 if keep_inside else -1.0
    count = 0
    previous = n_vertices - 1
    previous_distance = sign * (
        (b[0] - a[0]) * (polygon[previous, 1] - a[1])
        - (b[1] - a[1]) * (polygon[previous, 0] - a[0])
    )
    for current in range(n_vertices):
        current_distance = sign * (
            (b[0] - a[0]) * (polygon[current, 1] - a[1])
            - (b[1] - a[1]) * (polygon[current, 0] - a[0])
        )
        crosses = (current_distance >= 0.0) != (previous_distance >= 0.0)
        if crosses:
            t = previous_distance / (previous_distance - current_distance)
            output[count, 0] = polygon[previous, 0] + t * (
                polygon[current, 0] - polygon[previous, 0]
            )
            output[count, 1] = polygon[previous, 1] + t * (
                polygon[current, 1] - polygon[previous, 1]
            )
            count += 1
        if current_distance >= 0.0:
            output[count, 0] = polygon[current, 0]
            output[count, 1] = polygon[current, 1]
            count += 1
        previous = current
        previous_distance = current_distance
    return count


@numba.njit(cache=True)
def clip_convex_polygon_into(
    subject: NumberArray[TwoDimensions],
    n_subject: Integer,
    clip: NumberArray[TwoDimensions],
    n_clip: Integer,
    work: NumberArray[ThreeDimensions],
) -> tuple[Integer, Integer]:
    """
    Allocation-free `clip_convex_polygon` (Sutherland-Hodgman clipping).

    :param subject: Shape `(>= n_subject, 2)` polygon to clip, vertices counter-clockwise.
    :param n_subject: Number of valid rows of `subject`.
    :param clip: Shape `(>= n_clip, 2)` convex clipping polygon, vertices counter-clockwise.
    :param n_clip: Number of valid rows of `clip`.
    :param work: Shape `(2, capacity, 2)` scratch buffers, `capacity >= 2 * n + 2` for the
        largest intermediate polygon.
    :returns: `(buffer, n_vertices)`; the intersection is `work[buffer, :n_vertices]`.
        `n_vertices` is `-1` if `work` was too small.
    """
    if n_subject + 2 > work.shape[1]:
        return 0, -1
    for row in range(n_subject):
        work[0, row, 0] = subject[row, 0]
        work[0, row, 1] = subject[row, 1]
    current = 0
    n_vertices = n_subject
    for index in range(n_clip):
        n_vertices = clip_half_plane_into(
            polygon=work[current],
            n_vertices=n_vertices,
            a=clip[index],
            b=clip[(index + 1) % n_clip],
            keep_inside=True,
            output=work[1 - current],
        )
        if n_vertices < 0:
            return current, -1
        current = 1 - current
        if n_vertices == 0:
            break
    return current, n_vertices


@numba.njit(cache=True)
def subtract_overlap_from_pieces(
    pieces: NumberArray[ThreeDimensions],
    n_pieces: IntArray[OneDimension],
    count: Integer,
    overlap: NumberArray[TwoDimensions],
    n_overlap: Integer,
    next_pieces: NumberArray[ThreeDimensions],
    next_n_pieces: IntArray[OneDimension],
    work: NumberArray[ThreeDimensions],
    minimum_area: Number,
) -> Integer:
    """
    Allocation-free subtraction of a convex polygon from a set of convex pieces; the buffer
    form of calling `subtract_convex_polygon` on every piece and concatenating the results.

    :param pieces: Shape `(max_pieces, capacity, 2)` input polygons.
    :param n_pieces: Shape `(max_pieces,)` vertex count of each input polygon.
    :param count: Number of input polygons.
    :param overlap: Shape `(>= n_overlap, 2)` convex polygon to remove, counter-clockwise.
    :param n_overlap: Number of valid rows of `overlap`.
    :param next_pieces: Output polygons, same shape as `pieces`.
    :param next_n_pieces: Output vertex counts.
    :param work: Shape `(2, capacity, 2)` scratch buffers.
    :param minimum_area: Output polygons whose area is at most this are dropped. Slivers that
        can never become a face would otherwise keep intersecting later overlaps and multiply.
    :returns: Number of output polygons, or `-1` if a buffer was too small.
    """
    max_pieces = pieces.shape[0]
    capacity = pieces.shape[1]
    n_out = 0
    for piece in range(count):
        n_vertices = n_pieces[piece]
        if n_vertices + 2 > work.shape[1]:
            return -1
        for row in range(n_vertices):
            work[0, row, 0] = pieces[piece, row, 0]
            work[0, row, 1] = pieces[piece, row, 1]
        current = 0
        for index in range(n_overlap):
            a = overlap[index]
            b = overlap[(index + 1) % n_overlap]
            # The part of the remaining polygon outside this clip edge is a finished piece.
            if n_out >= max_pieces:
                return -1
            n_outside = clip_half_plane_into(
                polygon=work[current],
                n_vertices=n_vertices,
                a=a,
                b=b,
                keep_inside=False,
                output=next_pieces[n_out],
            )
            if n_outside < 0 or n_outside > capacity:
                return -1
            if n_outside >= 3 and (
                abs(compute_polygon_signed_area(next_pieces[n_out, :n_outside])) > minimum_area  # type: ignore[arg-type]
            ):
                next_n_pieces[n_out] = n_outside
                n_out += 1
            n_vertices = clip_half_plane_into(
                polygon=work[current],
                n_vertices=n_vertices,
                a=a,
                b=b,
                keep_inside=True,
                output=work[1 - current],
            )
            if n_vertices < 0:
                return -1
            current = 1 - current
            if n_vertices < 3:
                break
    return n_out


@numba.njit(cache=True)
def polygon_to_pillar_points(
    polygon: NumberArray[TwoDimensions],
    coord: CoordArray,
    pillars: IntArray[OneDimension],
    points: NumberArray[TwoDimensions],
) -> Integer:
    """
    Lift a polygon in the `(s, depth)` plane of a pillar pair to 3-D points.

    :param polygon: Shape `(n, 2)` polygon.
    :param coord: Shape `(NY+1, NX+1, 6)` pillar array.
    :param pillars: Shape `(4,)` `(j, i)` of the first pillar followed by `(j, i)` of the second.
    :param points: Shape `(>= n, 3)` output buffer.
    :returns: Number of points written, `n`.
    """
    pillar_1 = coord[pillars[0], pillars[1]]
    pillar_2 = coord[pillars[2], pillars[3]]
    for row in range(polygon.shape[0]):
        point = _get_point_on_pillar_pair(  # type: ignore
            pillar_1,  # type: ignore[arg-type]
            pillar_2,  # type: ignore[arg-type]
            polygon[row, 0],
            polygon[row, 1],  # type: ignore[arg-type]
        )
        points[row, 0] = point[0]
        points[row, 1] = point[1]
        points[row, 2] = point[2]
    return polygon.shape[0]


@numba.njit(cache=True)
def append_face(
    points: NumberArray[TwoDimensions],
    n_points: Integer,
    owner: Integer,
    neighbour: Integer,
    cell_centers: NumberArray[TwoDimensions],
    new_points: NumberArray[TwoDimensions],
    new_point_offsets: IntArray[OneDimension],
    new_owners: IntArray[OneDimension],
    new_neighbours: IntArray[OneDimension],
    n_faces: Integer,
    n_stored_points: Integer,
) -> tuple[Integer, Integer]:
    """
    Append a face to the new-face buffers, winding its points so that the normal points out
    of the owner cell.

    :param points: Shape `(>= n_points, 3)` face points.
    :param n_points: Number of points of the face.
    :param owner: Flat index of the owner cell.
    :param neighbour: Flat index of the neighbour cell, `-1` for a boundary face.
    :param cell_centers: Shape `(n_cells, 3)` mean of each cell's corners.
    :param new_points: Output point buffer.
    :param new_point_offsets: Output offsets buffer.
    :param new_owners: Output owner buffer.
    :param new_neighbours: Output neighbour buffer.
    :param n_faces: Faces stored so far.
    :param n_stored_points: Points stored so far.
    :returns: Updated `(n_faces, n_stored_points)`, or `(-1, -1)` if a buffer is full.
    """
    if n_faces >= new_owners.shape[0] or n_stored_points + n_points > new_points.shape[0]:
        return -1, -1

    normal_x = 0.0
    normal_y = 0.0
    normal_z = 0.0
    center_x = 0.0
    center_y = 0.0
    center_z = 0.0
    for index in range(n_points):
        following = (index + 1) % n_points
        ax = points[index, 0]
        ay = points[index, 1]
        az = points[index, 2]
        bx = points[following, 0]
        by = points[following, 1]
        bz = points[following, 2]
        normal_x += ay * bz - az * by
        normal_y += az * bx - ax * bz
        normal_z += ax * by - ay * bx
        center_x += ax
        center_y += ay
        center_z += az
    center_x /= n_points
    center_y /= n_points
    center_z /= n_points

    alignment = (
        normal_x * (center_x - cell_centers[owner, 0])
        + normal_y * (center_y - cell_centers[owner, 1])
        + normal_z * (center_z - cell_centers[owner, 2])
    )
    reverse = alignment < 0.0
    for index in range(n_points):
        source = n_points - 1 - index if reverse else index
        new_points[n_stored_points + index, 0] = points[source, 0]
        new_points[n_stored_points + index, 1] = points[source, 1]
        new_points[n_stored_points + index, 2] = points[source, 2]

    new_point_offsets[n_faces] = n_stored_points
    new_point_offsets[n_faces + 1] = n_stored_points + n_points
    new_owners[n_faces] = owner
    new_neighbours[n_faces] = neighbour
    return n_faces + 1, n_stored_points + n_points


@numba.njit(cache=True)
def connect_unmatched_faces_kernel(
    registry_owners: IntArray[OneDimension],
    registry_neighbours: IntArray[OneDimension],
    registry_local_indices: IntArray[OneDimension],
    n_registered: Integer,
    degenerate_cells: IntArray[OneDimension],
    degenerate_locals: IntArray[OneDimension],
    pinched: BooleanArray[OneDimension],
    active_mask: BooleanArray[OneDimension],
    corner_coordinates: NumberArray[ThreeDimensions],
    cell_centers: NumberArray[TwoDimensions],
    coord: CoordArray,
    nx: Integer,
    ny: Integer,
    nz: Integer,
    lateral_face_corners: IntArray[TwoDimensions],
    minimum_relative_overlap_area: Number,
    max_new_faces: Integer,
    max_new_points: Integer,
    polygon_capacity: Integer,
    max_polygon_pieces: Integer,
) -> tuple[
    Integer,
    Integer,
    Integer,
    NumberArray[TwoDimensions],
    IntArray[OneDimension],
    IntArray[OneDimension],
    IntArray[OneDimension],
    BooleanArray[OneDimension],
]:
    """
    Compiled implementation of `connect_unmatched_faces`.

    :param registry_owners: Owner cell of each registered face.
    :param registry_neighbours: Neighbour cell of each registered face, `-1` when unmatched.
    :param registry_local_indices: Local face index of each registered face on its owner.
    :param n_registered: Number of registered faces.
    :param degenerate_cells: Cells owning a collapsed lateral or vertical face.
    :param degenerate_locals: Local index of each collapsed face.
    :param pinched: Shape `(n_cells,)` pinched-out flag.
    :param active_mask: Shape `(n_cells,)` activation mask.
    :param corner_coordinates: Shape `(n_cells, 8, 3)` corner coordinates.
    :param cell_centers: Shape `(n_cells, 3)` mean of each cell's corners.
    :param coord: Shape `(ny + 1, nx + 1, 6)` pillar array.
    :param nx: Grid extent in x.
    :param ny: Grid extent in y.
    :param nz: Grid extent in z.
    :param lateral_face_corners: Shape `(6, 4)` corners spanning each lateral face.
    :param minimum_relative_overlap_area: Overlaps smaller than this fraction of the smaller
        facing face are ignored.
    :param max_new_faces: Capacity of the new-face buffers.
    :param max_new_points: Capacity of the new-point buffer.
    :param polygon_capacity: Rows of each polygon scratch buffer.
    :param max_polygon_pieces: Number of pieces one lateral face may be split into.
    :returns: `(status, n_faces, n_points, points, point_offsets, owners, neighbours, replaced)`.
        `status` is `0` on success, `1` when a face buffer was too small and `2` when a polygon
        buffer was too small; the call should then be repeated with larger capacities.
    """
    n_cells = nx * ny * nz
    layer_size = nx * ny

    n_degenerate = degenerate_cells.shape[0]
    lookup = np.full(n_cells * 6, -1, dtype=np.int32)
    unmatched_cells = np.empty(n_registered + n_degenerate, dtype=np.int64)
    unmatched_locals = np.empty(n_registered + n_degenerate, dtype=np.int64)
    unmatched_faces = np.empty(n_registered + n_degenerate, dtype=np.int64)
    n_unmatched = 0
    for face in range(n_registered):
        if registry_neighbours[face] == -1:
            cell = registry_owners[face]
            local = registry_local_indices[face]
            unmatched_cells[n_unmatched] = cell
            unmatched_locals[n_unmatched] = local
            unmatched_faces[n_unmatched] = face
            lookup[cell * 6 + local] = face
            n_unmatched += 1
    for index in range(n_degenerate):
        cell = degenerate_cells[index]
        local = degenerate_locals[index]
        unmatched_cells[n_unmatched] = cell
        unmatched_locals[n_unmatched] = local
        unmatched_faces[n_unmatched] = -2
        lookup[cell * 6 + local] = -2
        n_unmatched += 1

    replaced = np.zeros(n_registered, dtype=np.bool_)
    new_points = np.empty((max_new_points, 3), dtype=np.float64)
    new_point_offsets = np.zeros(max_new_faces + 1, dtype=np.int64)
    new_owners = np.empty(max_new_faces, dtype=np.int32)
    new_neighbours = np.empty(max_new_faces, dtype=np.int32)
    n_new_faces = 0
    n_new_points = 0
    if n_unmatched == 0:
        return (  # type: ignore[return-value]
            0,
            0,
            0,
            new_points[:0],
            new_point_offsets[:1],
            new_owners[:0],
            new_neighbours[:0],
            replaced,
        )

    # Overlap polygons found between facing lateral faces, and for each lateral face the
    # linked list of the overlaps that cover it.
    max_covers = 2 * max_new_faces + 2
    overlap_points = np.empty((max_new_points, 2), dtype=np.float64)
    overlap_offsets = np.zeros(max_new_faces + 2, dtype=np.int64)
    entry_overlaps = np.empty(max_covers, dtype=np.int64)
    entry_next = np.full(max_covers, -1, dtype=np.int64)
    n_entries = 0
    n_overlaps = 0
    cover_of_slot = np.full(n_cells * 6, -1, dtype=np.int64)
    cover_slots = np.empty(max_covers, dtype=np.int64)
    cover_head = np.full(max_covers, -1, dtype=np.int64)
    cover_tail = np.full(max_covers, -1, dtype=np.int64)
    cover_pillars = np.empty((max_covers, 4), dtype=np.int64)
    n_covers = 0
    n_overlap_points = 0

    scratch = np.empty((32, 3), dtype=np.float64)
    pillars = np.empty(4, dtype=np.int64)
    face_pillars = np.empty(4, dtype=np.int64)
    polygon_a = np.empty((4, 2), dtype=np.float64)
    polygon_b = np.empty((4, 2), dtype=np.float64)
    polygon_work = np.empty((2, polygon_capacity, 2), dtype=np.float64)

    for direction in range(2):
        if direction == 0:
            plus_local = 5
            minus_local = 4
            step_i = 1
            step_j = 0
            pillars[0] = 0
            pillars[1] = 1
            pillars[2] = 1
            pillars[3] = 1
        else:
            plus_local = 3
            minus_local = 2
            step_i = 0
            step_j = 1
            pillars[0] = 1
            pillars[1] = 0
            pillars[2] = 1
            pillars[3] = 1

        for entry in range(n_unmatched):
            if unmatched_locals[entry] != plus_local:
                continue
            cell_a = unmatched_cells[entry]
            rest = cell_a % layer_size
            j_a = rest // nx
            i_a = rest % nx
            if i_a + step_i >= nx or j_a + step_j >= ny:
                continue

            build_lateral_polygon(
                corner_coordinates=corner_coordinates,
                cell=cell_a,
                local_index=plus_local,
                lateral_face_corners=lateral_face_corners,
                polygon=polygon_a,
            )
            area_a = abs(compute_polygon_signed_area(polygon_a))
            if area_a <= 0.0:
                continue
            a_z_min = min(polygon_a[0, 1], polygon_a[1, 1], polygon_a[2, 1], polygon_a[3, 1])
            a_z_max = max(polygon_a[0, 1], polygon_a[1, 1], polygon_a[2, 1], polygon_a[3, 1])

            face_pillars[0] = j_a + pillars[0]
            face_pillars[1] = i_a + pillars[1]
            face_pillars[2] = j_a + pillars[2]
            face_pillars[3] = i_a + pillars[3]

            for k_b in range(nz):
                cell_b = (i_a + step_i) + (j_a + step_j) * nx + k_b * layer_size
                if lookup[cell_b * 6 + minus_local] == -1:
                    continue
                build_lateral_polygon(
                    corner_coordinates=corner_coordinates,
                    cell=cell_b,
                    local_index=minus_local,
                    lateral_face_corners=lateral_face_corners,
                    polygon=polygon_b,
                )
                area_b = abs(compute_polygon_signed_area(polygon_b))
                if area_b <= 0.0:
                    continue
                b_z_min = min(polygon_b[0, 1], polygon_b[1, 1], polygon_b[2, 1], polygon_b[3, 1])
                b_z_max = max(polygon_b[0, 1], polygon_b[1, 1], polygon_b[2, 1], polygon_b[3, 1])
                if a_z_max <= b_z_min or b_z_max <= a_z_min:
                    continue

                work_index, n_overlap_vertices = clip_convex_polygon_into(
                    subject=polygon_a,
                    n_subject=4,
                    clip=polygon_b,
                    n_clip=4,
                    work=polygon_work,  # type: ignore[arg-type]
                )
                if n_overlap_vertices < 0:
                    return _kernel_failure(
                        2,
                        new_points,  # type: ignore[arg-type]
                        new_point_offsets,
                        new_owners,
                        new_neighbours,
                        replaced,
                    )
                overlap = polygon_work[work_index, :n_overlap_vertices]
                if n_overlap_vertices < 3 or abs(compute_polygon_signed_area(overlap)) <= (  # type: ignore[arg-type]
                    minimum_relative_overlap_area * min(area_a, area_b)
                ):
                    continue

                n_points = polygon_to_pillar_points(
                    polygon=overlap,  # type: ignore[arg-type]
                    coord=coord,
                    pillars=face_pillars,
                    points=scratch,
                )
                n_new_faces, n_new_points = append_face(  # type: ignore
                    points=scratch,
                    n_points=n_points,
                    owner=cell_a,
                    neighbour=cell_b,
                    cell_centers=cell_centers,
                    new_points=new_points,  # type: ignore[arg-type]
                    new_point_offsets=new_point_offsets,
                    new_owners=new_owners,
                    new_neighbours=new_neighbours,
                    n_faces=n_new_faces,
                    n_stored_points=n_new_points,
                )
                if n_new_faces < 0 or n_overlaps >= max_new_faces:
                    return (  # type: ignore[return-value]
                        1,
                        0,
                        0,
                        new_points[:0],
                        new_point_offsets[:1],
                        new_owners[:0],
                        new_neighbours[:0],
                        replaced,
                    )
                if n_overlap_points + overlap.shape[0] > max_new_points:
                    return (  # type: ignore[return-value]
                        1,
                        0,
                        0,
                        new_points[:0],
                        new_point_offsets[:1],
                        new_owners[:0],
                        new_neighbours[:0],
                        replaced,
                    )

                overlap_offsets[n_overlaps] = n_overlap_points
                for row in range(overlap.shape[0]):
                    overlap_points[n_overlap_points + row, 0] = overlap[row, 0]
                    overlap_points[n_overlap_points + row, 1] = overlap[row, 1]
                n_overlap_points += overlap.shape[0]
                overlap_offsets[n_overlaps + 1] = n_overlap_points

                for side in range(2):
                    slot = cell_a * 6 + plus_local if side == 0 else cell_b * 6 + minus_local
                    cover = cover_of_slot[slot]
                    if cover < 0:
                        cover = n_covers
                        n_covers += 1
                        cover_of_slot[slot] = cover
                        cover_slots[cover] = slot
                        for q in range(4):
                            cover_pillars[cover, q] = face_pillars[q]
                    entry_overlaps[n_entries] = n_overlaps
                    if cover_head[cover] < 0:
                        cover_head[cover] = n_entries
                    else:
                        entry_next[cover_tail[cover]] = n_entries
                    cover_tail[cover] = n_entries
                    n_entries += 1
                n_overlaps += 1

    pieces = np.empty((max_polygon_pieces, polygon_capacity, 2), dtype=np.float64)
    n_piece_vertices = np.zeros(max_polygon_pieces, dtype=np.int64)
    next_pieces = np.empty((max_polygon_pieces, polygon_capacity, 2), dtype=np.float64)
    next_piece_vertices = np.zeros(max_polygon_pieces, dtype=np.int64)
    polygon_c = np.empty((4, 2), dtype=np.float64)
    for cover in range(n_covers):
        slot = cover_slots[cover]
        cell = slot // 6
        local = slot % 6
        build_lateral_polygon(
            corner_coordinates=corner_coordinates,
            cell=cell,
            local_index=local,
            lateral_face_corners=lateral_face_corners,
            polygon=polygon_c,
        )
        full_area = abs(compute_polygon_signed_area(polygon_c))
        for row in range(4):
            pieces[0, row, 0] = polygon_c[row, 0]
            pieces[0, row, 1] = polygon_c[row, 1]
        n_piece_vertices[0] = 4
        n_pieces = 1
        entry = cover_head[cover]
        while entry >= 0:
            overlap_index = entry_overlaps[entry]
            start = overlap_offsets[overlap_index]
            stop = overlap_offsets[overlap_index + 1]
            n_next = subtract_overlap_from_pieces(
                pieces=pieces,  # type: ignore[arg-type]
                n_pieces=n_piece_vertices,
                count=n_pieces,
                overlap=overlap_points[start:stop],  # type: ignore[arg-type]
                n_overlap=stop - start,  # type: ignore[arg-type]
                next_pieces=next_pieces,  # type: ignore[arg-type]
                next_n_pieces=next_piece_vertices,
                work=polygon_work,  # type: ignore[arg-type]
                minimum_area=0.5 * minimum_relative_overlap_area * full_area,
            )
            if n_next < 0:
                return _kernel_failure(
                    2,
                    new_points,  # type: ignore[arg-type]
                    new_point_offsets,
                    new_owners,
                    new_neighbours,
                    replaced,
                )
            for piece in range(n_next):
                n_vertices = next_piece_vertices[piece]
                n_piece_vertices[piece] = n_vertices
                for row in range(n_vertices):
                    pieces[piece, row, 0] = next_pieces[piece, row, 0]
                    pieces[piece, row, 1] = next_pieces[piece, row, 1]
            n_pieces = n_next
            entry = entry_next[entry]

        for piece in range(n_pieces):
            piece_polygon = pieces[piece, : n_piece_vertices[piece]]
            if (
                abs(compute_polygon_signed_area(piece_polygon))  # type: ignore[arg-type]
                > minimum_relative_overlap_area * full_area
            ):
                n_points = polygon_to_pillar_points(
                    polygon=piece_polygon,  # type: ignore[arg-type]
                    coord=coord,
                    pillars=cover_pillars[cover],
                    points=scratch,
                )
                n_new_faces, n_new_points = append_face(  # type: ignore
                    points=scratch,
                    n_points=n_points,
                    owner=cell,
                    neighbour=-1,
                    cell_centers=cell_centers,
                    new_points=new_points,  # type: ignore[arg-type]
                    new_point_offsets=new_point_offsets,
                    new_owners=new_owners,
                    new_neighbours=new_neighbours,
                    n_faces=n_new_faces,
                    n_stored_points=n_new_points,
                )
                if n_new_faces < 0:
                    return _kernel_failure(
                        1,
                        new_points,  # type: ignore[arg-type]
                        new_point_offsets,
                        new_owners,
                        new_neighbours,
                        replaced,
                    )
        face = lookup[slot]
        if face >= 0:
            replaced[face] = True

    used_tops = np.zeros(n_cells, dtype=np.bool_)
    footprint = np.empty((4, 3), dtype=np.float64)
    for entry in range(n_unmatched):
        if unmatched_locals[entry] != BOTTOM_FACE_LOCAL:
            continue
        cell_a = unmatched_cells[entry]
        face_a = unmatched_faces[entry]
        cell_b = cell_a + layer_size
        while cell_b < nz * layer_size and active_mask[cell_b] and pinched[cell_b]:
            cell_b += layer_size
        if (
            cell_b >= nz * layer_size
            or not active_mask[cell_b]
            or used_tops[cell_b]
            or lookup[cell_b * 6 + TOP_FACE_LOCAL] == -1
        ):
            continue

        n_points = 0
        for corner in (0, 1, 3, 2):
            mx = 0.5 * (
                corner_coordinates[cell_a, 4 + corner, 0] + corner_coordinates[cell_b, corner, 0]
            )
            my = 0.5 * (
                corner_coordinates[cell_a, 4 + corner, 1] + corner_coordinates[cell_b, corner, 1]
            )
            mz = 0.5 * (
                corner_coordinates[cell_a, 4 + corner, 2] + corner_coordinates[cell_b, corner, 2]
            )
            keep = n_points == 0
            if not keep:
                dx = mx - footprint[n_points - 1, 0]
                dy = my - footprint[n_points - 1, 1]
                dz = mz - footprint[n_points - 1, 2]
                keep = np.sqrt(dx * dx + dy * dy + dz * dz) > 1e-9
            if keep:
                footprint[n_points, 0] = mx
                footprint[n_points, 1] = my
                footprint[n_points, 2] = mz
                n_points += 1

        if n_points > 1:
            dx = footprint[0, 0] - footprint[n_points - 1, 0]
            dy = footprint[0, 1] - footprint[n_points - 1, 1]
            dz = footprint[0, 2] - footprint[n_points - 1, 2]
            if np.sqrt(dx * dx + dy * dy + dz * dz) <= 1e-9:
                n_points -= 1
        if n_points < 3:
            continue

        normal_x = 0.0
        normal_y = 0.0
        normal_z = 0.0
        for index in range(n_points):
            following = (index + 1) % n_points
            ax = footprint[index, 0]
            ay = footprint[index, 1]
            az = footprint[index, 2]
            bx = footprint[following, 0]
            by = footprint[following, 1]
            bz = footprint[following, 2]
            normal_x += ay * bz - az * by
            normal_y += az * bx - ax * bz
            normal_z += ax * by - ay * bx
        if np.sqrt(normal_x * normal_x + normal_y * normal_y + normal_z * normal_z) <= 0.0:
            continue

        n_new_faces, n_new_points = append_face(  # type: ignore
            points=footprint,
            n_points=n_points,
            owner=cell_a,
            neighbour=cell_b,
            cell_centers=cell_centers,
            new_points=new_points,  # type: ignore[arg-type]
            new_point_offsets=new_point_offsets,
            new_owners=new_owners,
            new_neighbours=new_neighbours,
            n_faces=n_new_faces,
            n_stored_points=n_new_points,
        )
        if n_new_faces < 0:
            return (  # type: ignore[return-value]
                1,
                0,
                0,
                new_points[:0],
                new_point_offsets[:1],
                new_owners[:0],
                new_neighbours[:0],
                replaced,
            )
        used_tops[cell_b] = True
        if face_a >= 0:
            replaced[face_a] = True
        face_b = lookup[cell_b * 6 + TOP_FACE_LOCAL]
        if face_b >= 0:
            replaced[face_b] = True

    return (  # type: ignore[return-value]
        0,
        n_new_faces,
        n_new_points,
        new_points[:n_new_points],
        new_point_offsets[: n_new_faces + 1],
        new_owners[:n_new_faces],
        new_neighbours[:n_new_faces],
        replaced,
    )


def connect_unmatched_faces(
    *,
    registry: FaceRegistry,
    active_mask: BooleanArray[OneDimension],
    corner_coordinates: NumberArray[ThreeDimensions],
    coord: CoordArray,
    dimensions: tuple[Integer, Integer, Integer],
) -> NewFaces:
    """
    Connect cells whose facing faces do not share all four vertices.

    **Lateral faces.** Across a fault the neighbouring column's cells may sit at a different
    depth, so one lateral face can face several cells of the next column, each only partly. The
    overlap with every facing cell of the adjacent column (any `k`) is computed as a polygon in
    the surface spanned by the two shared pillars and becomes an interior face between the two
    cells. The part of a face that no neighbour covers is kept as a boundary face.

    **Vertical faces.** The bottom face of a cell and the top face of the next active,
    non-pinched cell below it in the same column share a footprint but not necessarily their
    corner depths (layer gaps or overlaps). They are replaced by one interior face halfway
    between the two surfaces. An inactive cell between them blocks the connection.

    :param registry: Faces registered by exact vertex matches.
    :param active_mask: Shape `(n_cells,)` activation mask.
    :param corner_coordinates: Shape `(n_cells, 8, 3)` corner coordinates of every cell.
    :param coord: Pillar array of shape `(ny + 1, nx + 1, 6)`.
    :param dimensions: Grid extents `(nx, ny, nz)`.
    :returns: The faces to add and the registered faces they replace. The new faces
        hold their own vertex coordinates; the caller appends them to the grid's vertices.
    """
    nx, ny, nz = dimensions
    n_unmatched_estimate = int(np.count_nonzero(registry.neighbour_cells == -1)) + len(
        registry.degenerate_cells
    )
    cell_centers = corner_coordinates.mean(axis=1)
    max_new_faces = 2 * n_unmatched_estimate + 64
    max_new_points = 6 * max_new_faces
    polygon_capacity = INITIAL_POLYGON_CAPACITY
    max_polygon_pieces = INITIAL_POLYGON_PIECES
    for _ in range(24):
        (
            status,
            n_faces,
            _,
            points,
            point_offsets,
            owners,
            neighbours,
            replaced,
        ) = connect_unmatched_faces_kernel(
            registry_owners=registry.owner_cells,
            registry_neighbours=registry.neighbour_cells,
            registry_local_indices=registry.local_indices,
            n_registered=registry.n_faces,
            degenerate_cells=registry.degenerate_cells,
            degenerate_locals=registry.degenerate_local_indices,
            pinched=registry.pinched,
            active_mask=active_mask,
            corner_coordinates=corner_coordinates,
            cell_centers=cell_centers,
            coord=coord,
            nx=nx,
            ny=ny,
            nz=nz,
            lateral_face_corners=LATERAL_FACE_CORNERS,
            minimum_relative_overlap_area=MINIMUM_RELATIVE_OVERLAP_AREA,
            max_new_faces=max_new_faces,
            max_new_points=max_new_points,
            polygon_capacity=polygon_capacity,
            max_polygon_pieces=max_polygon_pieces,
        )
        if status == 0:
            return NewFaces(
                n_faces=n_faces,
                points=points,
                point_offsets=point_offsets,
                owner_cells=owners,
                neighbour_cells=neighbours,
                replaced=replaced,
            )
        if status == 2:
            if polygon_capacity >= MAX_POLYGON_CAPACITY:
                raise InvalidGridError(
                    "A lateral face of the corner-point grid is overlapped too intricately to resolve."
                )
            polygon_capacity *= 2
            max_polygon_pieces *= 2
        else:
            max_new_faces *= 4
            max_new_points *= 4
    raise InvalidGridError("Could not connect the unmatched faces of the corner-point grid.")


@numba.njit(cache=True)
def find_distinct_points(
    points: NumberArray[TwoDimensions],
    tolerance: Number,
) -> tuple[
    Integer,
    IntArray[OneDimension],
    IntArray[TwoDimensions],
    IntArray[OneDimension],
]:
    """
    Group points that fall on the same `tolerance`-sized lattice cell, using a hash table.

    Points are quantised as `round(coordinate / tolerance)` (half to even) and two points
    belong to the same group when all three quantised coordinates are equal.

    :param points: Shape `(n_points, 3)` coordinates.
    :param tolerance: Lattice spacing.
    :returns: `(n_groups, first_index, lattice, group_of_point)`. Groups are numbered in the
        order of their first point; `first_index[g]` is that point and `lattice[g]` its
        quantised coordinates, both of length `n_groups`.
    """
    n_points = points.shape[0]
    capacity = 16
    while capacity < 2 * n_points:
        capacity *= 2
    mask = capacity - 1
    table = np.full(capacity, -1, dtype=np.int64)
    first_index = np.empty(n_points, dtype=np.int64)
    lattice = np.empty((n_points, 3), dtype=np.int64)
    group_of_point = np.empty(n_points, dtype=np.int64)
    n_groups = 0
    for point in range(n_points):
        q0 = np.int64(np.rint(points[point, 0] / tolerance))
        q1 = np.int64(np.rint(points[point, 1] / tolerance))
        q2 = np.int64(np.rint(points[point, 2] / tolerance))
        h = np.int64(1469598103934665603)
        h = (h ^ q0) * np.int64(1099511628211)
        h = (h ^ q1) * np.int64(1099511628211)
        h = (h ^ q2) * np.int64(1099511628211)
        h = h ^ (h >> 29)
        slot = h & mask
        while True:
            group = table[slot]
            if group < 0:
                table[slot] = n_groups
                first_index[n_groups] = point
                lattice[n_groups, 0] = q0
                lattice[n_groups, 1] = q1
                lattice[n_groups, 2] = q2
                group_of_point[point] = n_groups
                n_groups += 1
                break
            if lattice[group, 0] == q0 and lattice[group, 1] == q1 and lattice[group, 2] == q2:
                group_of_point[point] = group
                break
            slot = (slot + 1) & mask
    return n_groups, first_index[:n_groups], lattice[:n_groups], group_of_point  # type: ignore[return-value]


def merge_coincident_points(
    points: NumberArray[TwoDimensions],
    tolerance: Number,
) -> tuple[IntArray[OneDimension], IntArray[OneDimension]]:
    """
    Merge points that quantise to the same `tolerance` lattice cell.

    Equivalent to `np.unique(np.round(points / tolerance).astype(np.int64), axis=0,
    return_index=True, return_inverse=True)` but without sorting every point: points are
    grouped with a hash table and only the distinct ones are sorted, so merged vertices are
    numbered in the same lexicographic order.

    :param points: Shape `(n_points, 3)` coordinates.
    :param tolerance: Lattice spacing.
    :returns: `(first_index, merged_index)` where `points[first_index]` are the merged
        points in lexicographic order of their lattice coordinates and `merged_index[p]` is
        the position of point `p`'s merged point.
    """
    _, first_index, lattice, group_of_point = find_distinct_points(
        points=typing.cast(
            NumberArray[TwoDimensions], np.ascontiguousarray(points, dtype=np.float64)
        ),
        tolerance=float(tolerance),
    )
    order = np.lexsort((lattice[:, 2], lattice[:, 1], lattice[:, 0]))
    rank = np.empty(len(order), dtype=np.intp)
    rank[order] = np.arange(len(order), dtype=np.intp)
    return typing.cast(IntArray[OneDimension], first_index[order]), typing.cast(
        IntArray[OneDimension], rank[group_of_point]
    )


def compute_corner_point_geometry(
    coord: CoordArray,
    zcorn: ZCornArray,
    actnum: ActNumArray,
    vertex_tolerance: Number = 1e-8,
    pinch_tolerance: Number = 0.0,
    mirrored: bool = False,
    on_inverted_cells: InvertedCellPolicy = InvertedCellPolicy.RAISE,
) -> tuple[
    VertexCoordinates,
    IntArray[OneDimension],
    IntArray[OneDimension],
    IntArray[TwoDimensions],
    IntArray[OneDimension],
    IntArray[OneDimension],
    NumberArray[OneDimension],
    NumberArray[TwoDimensions],
    NumberArray[TwoDimensions],
    NumberArray[TwoDimensions],
]:
    """
    Compute 3-D corner coordinates and build face arrays for a corner-point grid.

    :param coord: Shape `(NY+1, NX+1, 6)`.
    :param zcorn: Shape `(NZ*2, NY*2, NX*2)`.
    :param actnum: Shape `(NZ, NY, NX)`.
    :param vertex_tolerance: Vertex merge distance.
    :param pinch_tolerance: Average thickness threshold for pinch detection.
    :param mirrored: Whether `coord` is a mirror image of a right-handed `(x, y, depth)`
        frame, as when a `MAPAXES` with a south-pointing Y axis has been applied. Cell
        volumes and face winding are reported as they would be for the unmirrored grid.
    :param on_inverted_cells: Whether inverted active cells raise, are deactivated, or are
        kept with the magnitude of their volume.
    :returns: 10-tuple `(vertex_coordinates, face_vertex_indices,
        face_vertex_offsets, face_cell_indices, face_connection_types, cell_statuses,
        cell_volumes, cell_centroids, cell_min_xyz, cell_max_xyz)`. The last two are the
        bounding box of each cell's own corners. Cells are
        numbered by their full-grid flat index `i + j * nx + k * nx * ny`. Inactive cells
        keep their index, have status `INACTIVE`, zero volume and no faces.
    :raises InvalidGridError: If no active cells are found, or if active cells are inverted
        and `on_inverted_cells` is `RAISE`.
    """
    active_mask = (actnum > 0).ravel()
    if not active_mask.any():
        raise InvalidGridError(
            "No active cells found in the corner-point grid (`ACTNUM` is all zeros)."
        )

    all_cells = np.argwhere(np.ones(actnum.shape, dtype=bool)).astype(np.int32)
    corner_coordinates = compute_cell_corner_coordinates(
        cells=all_cells,  # type: ignore[arg-type]
        coord=coord,
        zcorn=zcorn,
    )

    flat_corners = corner_coordinates.reshape(-1, 3)
    unique_indices, inverse = merge_coincident_points(
        points=flat_corners,
        tolerance=vertex_tolerance,
    )
    vertex_coordinates = typing.cast(VertexCoordinates, flat_corners[unique_indices])
    n_cells = len(all_cells)
    corner_global = inverse.reshape(n_cells, 8)

    vtk_to_corner = [0, 1, 3, 2, 4, 5, 7, 6]

    if on_inverted_cells is InvertedCellPolicy.DEACTIVATE:
        trial_volumes = compute_oriented_cell_volumes(
            corner_global=corner_global,
            vtk_to_corner=vtk_to_corner,
            vertex_coordinates=vertex_coordinates,
            corner_coordinates=corner_coordinates,
            mirrored=mirrored,
        )[0]
        trial_inverted = active_mask & (trial_volumes < 0.0)
        if trial_inverted.any():
            warnings.warn(
                f"{int(trial_inverted.sum())} active cell(s) have inverted geometry and "
                f"were deactivated.",
                stacklevel=4,
            )
            active_mask = active_mask & ~trial_inverted
            if not active_mask.any():
                raise InvalidGridError("Every active cell has inverted geometry.")

    registry_arrays = register_cell_faces(
        corner_global=corner_global,
        active_cells=typing.cast(
            IntArray[OneDimension], np.flatnonzero(active_mask).astype(np.int32, copy=False)
        ),
        vertex_coordinates=vertex_coordinates,  # type: ignore[arg-type]
        pinch_tolerance=pinch_tolerance,
        mirrored=mirrored,
        face_corner_table=FACE_CORNER_TABLE,
        vtk_to_corner=VTK_TO_CORNER_ORDER,
    )
    registry = FaceRegistry(
        n_faces=registry_arrays[0],
        vertices=registry_arrays[1],
        owner_cells=registry_arrays[2],
        neighbour_cells=registry_arrays[3],
        local_indices=registry_arrays[4],
        degenerate_cells=registry_arrays[5],
        degenerate_local_indices=registry_arrays[6],
        pinched=registry_arrays[7],
        n_ignored_third_claims=registry_arrays[8],
    )
    n_pinched = registry.pinched.sum()

    nz_cells, ny_cells, nx_cells = actnum.shape
    new_faces = connect_unmatched_faces(
        registry=registry,
        active_mask=typing.cast(BooleanArray[OneDimension], active_mask),
        corner_coordinates=corner_coordinates,
        coord=coord,
        dimensions=(nx_cells, ny_cells, nz_cells),
    )
    first_new_vertex = len(vertex_coordinates)
    if len(new_faces.points) > 0:
        vertex_coordinates = typing.cast(
            VertexCoordinates, np.vstack([vertex_coordinates, new_faces.points])
        )

    if n_pinched > 0:
        warnings.warn(
            f"{n_pinched} pinched-out cell(s) detected "
            f"(pinch_tolerance={pinch_tolerance:.3g}). "
            f"{registry.n_ignored_third_claims} face(s) claimed by a third cell were ignored.",
            stacklevel=4,
        )

    kept = ~new_faces.replaced
    kept_vertices = registry.vertices[kept]
    flat_face_vertex_indices = np.concatenate([
        kept_vertices.ravel(),
        first_new_vertex + np.arange(len(new_faces.points)),
    ]).astype(np.int32)
    face_lengths = np.concatenate([
        np.full(len(kept_vertices), 4, dtype=np.int64),
        np.diff(new_faces.point_offsets),
    ])
    face_vertex_offsets = np.concatenate([[0], np.cumsum(face_lengths)]).astype(np.int32)
    face_cell_pairs = np.concatenate([
        np.column_stack([registry.owner_cells[kept], registry.neighbour_cells[kept]]),
        np.column_stack([new_faces.owner_cells, new_faces.neighbour_cells]).reshape(-1, 2),
    ]).astype(np.int32)
    face_connection_types = np.where(
        face_cell_pairs[:, 1] < 0,
        int(ConnectionType.BOUNDARY_FACE),
        int(ConnectionType.INTERIOR_FACE),
    ).astype(np.int8)

    cell_volumes, cell_centroids = compute_oriented_cell_volumes(
        corner_global=corner_global,
        vtk_to_corner=vtk_to_corner,
        vertex_coordinates=vertex_coordinates,
        corner_coordinates=corner_coordinates,
        mirrored=mirrored,
    )
    inverted = active_mask & (cell_volumes < 0.0)
    if inverted.any() and on_inverted_cells is InvertedCellPolicy.ABSOLUTE:
        warnings.warn(
            f"{int(inverted.sum())} active cell(s) have inverted geometry; their volumes "
            f"were set to the magnitude of the computed volume.",
            stacklevel=4,
        )
        cell_volumes = np.where(inverted, -cell_volumes, cell_volumes)
    elif inverted.any():
        bad = np.flatnonzero(inverted)
        raise InvalidGridError(
            f"{len(bad)} active cell(s) have negative volume (inverted geometry): "
            f"{bad[:5].tolist()}{'...' if len(bad) > 5 else ''}."
        )

    cell_volumes = typing.cast(NumberArray[OneDimension], np.where(active_mask, cell_volumes, 0.0))
    cell_statuses = typing.cast(
        IntArray[OneDimension],
        np.where(active_mask, int(CellStatus.ACTIVE), int(CellStatus.INACTIVE)).astype(np.int8),
    )
    return (
        vertex_coordinates,
        typing.cast(IntArray[OneDimension], flat_face_vertex_indices),
        typing.cast(IntArray[OneDimension], face_vertex_offsets),
        typing.cast(IntArray[TwoDimensions], face_cell_pairs),
        typing.cast(IntArray[OneDimension], face_connection_types),
        cell_statuses,
        cell_volumes,
        cell_centroids,
        corner_coordinates.min(axis=1),
        corner_coordinates.max(axis=1),
    )


def resolve_fault_face_indices(
    fault_records: typing.Sequence[FaultRecord],
    dimensions: tuple[Integer, Integer, Integer],
    actnum: ActNumArray,
    face_cell_indices: IntArray[TwoDimensions],
) -> dict[str, IntArray[OneDimension]]:
    """
    Resolve `FaultRecord` IJK ranges to unstructured face index arrays.

    For `X` and `Y` records every face between a cell and the cells of the adjacent column
    is part of the fault, since an offset fault faces several cells across it. Cell pairs that
    share no geometric face are not connected and are skipped, as are pairs involving an inactive
    cell. A record that resolves to no face at all is valid (a fault with more throw than the
    reservoir is thick seals by juxtaposition), so it is only logged at debug level.

    :param fault_records: Sequence of `FaultRecord` objects.
    :param dimensions: Grid extents `(nx, ny, nz)`.
    :param actnum: Shape `(nz, ny, nx)` activation mask.
    :param face_cell_indices: Shape `(n_faces, 2)`.
    :returns: Mapping from fault name to the indices of its faces.
    """
    nx, ny, nz = dimensions
    layer_size = nx * ny
    n_cells = layer_size * nz
    active = np.asarray(actnum > 0, dtype=bool).reshape(nz, ny, nx)

    # Sorted keys of the cell pairs that share an interior face, with the index of the face
    # connecting each pair; when several faces connect one pair the last one wins.
    interior_faces = np.flatnonzero(
        (face_cell_indices[:, 0] >= 0) & (face_cell_indices[:, 1] >= 0)
    )
    owners = face_cell_indices[interior_faces, 0].astype(np.int64)
    neighbours = face_cell_indices[interior_faces, 1].astype(np.int64)
    pair_keys = np.minimum(owners, neighbours) * n_cells + np.maximum(owners, neighbours)
    order = np.argsort(pair_keys, kind="stable")
    sorted_keys = pair_keys[order]
    sorted_faces = interior_faces[order]
    last_of_key = np.ones(len(sorted_keys), dtype=bool)
    last_of_key[:-1] = sorted_keys[1:] != sorted_keys[:-1]
    unique_keys = sorted_keys[last_of_key]
    unique_faces = sorted_faces[last_of_key]

    def is_active(k: IntArray[OneDimension], j: IntArray[OneDimension], i: IntArray[OneDimension]):
        inside = (k >= 0) & (k < nz) & (j >= 0) & (j < ny) & (i >= 0) & (i < nx)
        result = np.zeros(inside.shape, dtype=bool)
        result[inside] = active[k[inside], j[inside], i[inside]]
        return result

    def face_of_pairs(cells_a: IntArray[OneDimension], cells_b: IntArray[OneDimension]):
        """Face index of each cell pair, `-1` where the pair shares no face."""
        keys = np.minimum(cells_a, cells_b) * n_cells + np.maximum(cells_a, cells_b)
        position = np.searchsorted(unique_keys, keys)
        position[position >= len(unique_keys)] = 0
        found = (unique_keys[position] == keys) if len(unique_keys) else np.zeros_like(keys, bool)
        return np.where(found, unique_faces[position] if len(unique_keys) else -1, -1)

    result: dict[str, list[IntArray[OneDimension]]] = {}

    for record in fault_records:
        face_direction = record.face_direction.upper()
        if face_direction not in FACE_DIRECTION_TO_LOCAL:
            warnings.warn(
                f"Fault {record.name!r}: unrecognised face direction "
                f"{record.face_direction!r}. "
                f"Valid: {sorted(FACE_DIRECTION_TO_LOCAL)}. Skipping.",
                stacklevel=4,
            )
            continue

        step = -1 if face_direction.endswith("-") else 1
        if face_direction.startswith("X"):
            di, dj, dk = step, 0, 0
        elif face_direction.startswith("Y"):
            di, dj, dk = 0, step, 0
        else:
            di, dj, dk = 0, 0, step

        k_range = np.arange(record.k1 - 1, record.k2, dtype=np.int64)
        j_range = np.arange(record.j1 - 1, record.j2, dtype=np.int64)
        i_range = np.arange(record.i1 - 1, record.i2, dtype=np.int64)
        if k_range.size == 0 or j_range.size == 0 or i_range.size == 0:
            continue
        kk, jj, ii = (
            axis.ravel() for axis in np.meshgrid(k_range, j_range, i_range, indexing="ij")
        )

        cell_a_active = is_active(kk, jj, ii)
        cells_a = ii + jj * nx + kk * layer_size
        lateral = di != 0 or dj != 0
        has_neighbour = np.zeros(len(kk), dtype=bool)
        found_faces: list[IntArray[OneDimension]] = []
        for neighbour_offset in range(nz) if lateral else (None,):
            neighbour_k = (
                np.full(len(kk), neighbour_offset, dtype=np.int64)
                if neighbour_offset is not None
                else kk + dk
            )
            neighbour_active = is_active(neighbour_k, jj + dj, ii + di)
            has_neighbour |= neighbour_active
            connected = cell_a_active & neighbour_active
            if not connected.any():
                continue
            cells_b = (ii + di) + (jj + dj) * nx + neighbour_k * layer_size
            faces = face_of_pairs(cells_a[connected], cells_b[connected])  # type: ignore[arg-type]
            found_faces.append(faces[faces >= 0])  # type: ignore[arg-type]

        # A fault touching inactive cells is valid: those pairs simply have no flow connection,
        # so they are dropped silently. A record whose active cells share no face is valid too:
        # a fault with more throw than the reservoir is thick seals by juxtaposition, so its
        # multiplier has nothing to act on. That is expected geometry, not a mistake, and is
        # only logged for anyone debugging a fault multiplier that seems to do nothing.
        n_resolved = sum(len(faces) for faces in found_faces)
        n_active_cells = int((cell_a_active & has_neighbour).sum())
        if n_active_cells > 0 and n_resolved == 0:
            logger.debug(
                "Fault %r: none of the %d active cell(s) in record (I %d-%d, J %d-%d, K %d-%d, %s) "
                "share a face with an active neighbour, so it has no effect on flow.",
                record.name,
                n_active_cells,
                record.i1,
                record.i2,
                record.j1,
                record.j2,
                record.k1,
                record.k2,
                record.face_direction,
            )

        if found_faces and sum(len(faces) for faces in found_faces) > 0:
            result.setdefault(record.name, []).extend(found_faces)

    return {
        name: typing.cast(
            IntArray[OneDimension], np.unique(np.concatenate(chunks).astype(np.int32))
        )
        for name, chunks in result.items()
    }


@numba.njit(cache=True)
def _accumulate_pillars(
    cell_min_xyz: NumberArray[TwoDimensions],
    cell_max_xyz: NumberArray[TwoDimensions],
    nx: Integer,
    ny: Integer,
    nz: Integer,
    pillar_x: NumberArray[TwoDimensions],
    pillar_y: NumberArray[TwoDimensions],
    pillar_z_top: NumberArray[TwoDimensions],
    pillar_z_bottom: NumberArray[TwoDimensions],
    pillar_count: IntArray[TwoDimensions],
) -> None:
    """
    Accumulate per-pillar XY positions and Z extents from cell bounding boxes.

    :param cell_min_xyz: Shape `(n_cells, 3)` bounding-box minima.
    :param cell_max_xyz: Shape `(n_cells, 3)` bounding-box maxima.
    :param nx: Number of cells in x.
    :param ny: Number of cells in y.
    :param nz: Number of cells in z.
    :param pillar_x: Accumulator for pillar X (zeroed on entry).
    :param pillar_y: Accumulator for pillar Y (zeroed on entry).
    :param pillar_z_top: Accumulator for minimum pillar Z (`+inf` on entry).
    :param pillar_z_bottom: Accumulator for maximum pillar Z (`-inf` on entry).
    :param pillar_count: Contribution counter per pillar (zeroed on entry).
    """
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                cell_idx = i + j * nx + k * nx * ny
                min_x = cell_min_xyz[cell_idx, 0]
                min_y = cell_min_xyz[cell_idx, 1]
                min_z = cell_min_xyz[cell_idx, 2]
                max_x = cell_max_xyz[cell_idx, 0]
                max_y = cell_max_xyz[cell_idx, 1]
                max_z = cell_max_xyz[cell_idx, 2]

                for corner in range(4):
                    if corner == 0:
                        pj, pi, px, py = j, i, min_x, min_y
                    elif corner == 1:
                        pj, pi, px, py = j, i + 1, max_x, min_y
                    elif corner == 2:
                        pj, pi, px, py = j + 1, i, min_x, max_y
                    else:
                        pj, pi, px, py = j + 1, i + 1, max_x, max_y

                    pillar_x[pj, pi] += px
                    pillar_y[pj, pi] += py
                    if min_z < pillar_z_top[pj, pi]:
                        pillar_z_top[pj, pi] = min_z
                    if max_z > pillar_z_bottom[pj, pi]:
                        pillar_z_bottom[pj, pi] = max_z
                    pillar_count[pj, pi] += 1


@numba.njit(parallel=True, cache=True)
def _fill_zcorn(
    cell_min_xyz: NumberArray[TwoDimensions],
    cell_max_xyz: NumberArray[TwoDimensions],
    nx: Integer,
    ny: Integer,
    nz: Integer,
    zcorn: ZCornArray,
) -> None:
    """
    Fill `ZCORN` array from per-cell Z bounding-box extents.

    :param cell_min_xyz: Shape `(n_cells, 3)` bounding-box minima.
    :param cell_max_xyz: Shape `(n_cells, 3)` bounding-box maxima.
    :param nx: Number of cells in x.
    :param ny: Number of cells in y.
    :param nz: Number of cells in z.
    :param zcorn: Output array, pre-allocated as `(nz*2, ny*2, nx*2)`.
    """
    for k in numba.prange(nz):  # type: ignore
        for j in range(ny):
            for i in range(nx):
                cell_idx = i + j * nx + k * nx * ny
                z_top = cell_min_xyz[cell_idx, 2]
                z_bottom = cell_max_xyz[cell_idx, 2]
                zcorn[2 * k, 2 * j, 2 * i] = z_top
                zcorn[2 * k, 2 * j, 2 * i + 1] = z_top
                zcorn[2 * k, 2 * j + 1, 2 * i] = z_top
                zcorn[2 * k, 2 * j + 1, 2 * i + 1] = z_top
                zcorn[2 * k + 1, 2 * j, 2 * i] = z_bottom
                zcorn[2 * k + 1, 2 * j, 2 * i + 1] = z_bottom
                zcorn[2 * k + 1, 2 * j + 1, 2 * i] = z_bottom
                zcorn[2 * k + 1, 2 * j + 1, 2 * i + 1] = z_bottom


def compute_oriented_cell_volumes(
    *,
    corner_global: IntArray[TwoDimensions],
    vtk_to_corner: typing.Sequence[int],
    vertex_coordinates: VertexCoordinates,
    corner_coordinates: NumberArray[ThreeDimensions],
    mirrored: bool,
) -> tuple[NumberArray[OneDimension], NumberArray[TwoDimensions]]:
    """
    Cell volumes and centroids with the sign of the unmirrored `(x, y, depth)` frame.

    A cell is inverted when its volume is negative. Volumes within round-off of zero are
    set to exactly zero.

    :param corner_global: Shape `(n_cells, 8)` global vertex index of each cell corner.
    :param vtk_to_corner: Corner order that maps VTK hexahedron order to `corner_global`.
    :param vertex_coordinates: Shape `(n_verts, 3)` world coordinates.
    :param corner_coordinates: Shape `(n_cells, 8, 3)` coordinates of each cell's corners.
    :param mirrored: Whether the coordinates are a mirror image of the right-handed frame.
    :returns: `(cell_volumes, cell_centroids)`.
    """
    vtk_corner_indices = typing.cast(
        IntArray[TwoDimensions], corner_global[:, list(vtk_to_corner)].astype(np.int32)
    )
    cell_volumes, cell_centroids = compute_hex_volumes_and_centroids(
        vtk_corner_indices=vtk_corner_indices,
        vertex_coordinates=vertex_coordinates,  # type: ignore[arg-type]
    )
    if mirrored:
        cell_volumes = -cell_volumes
    box_volume = np.prod(corner_coordinates.max(axis=1) - corner_coordinates.min(axis=1), axis=1)
    round_off = (cell_volumes < 0.0) & (cell_volumes >= -1e-9 * box_volume)
    cell_volumes = typing.cast(NumberArray[OneDimension], np.where(round_off, 0.0, cell_volumes))
    return cell_volumes, cell_centroids


@numba.njit(parallel=True, cache=True)
def compute_hex_volumes_and_centroids(
    vtk_corner_indices: IntArray[TwoDimensions],
    vertex_coordinates: NumberArray[TwoDimensions],
) -> tuple[NumberArray[OneDimension], NumberArray[TwoDimensions]]:
    """
    Compute signed hexahedral cell volumes and centroids from the trilinear cell map.

    A corner-point cell has bilinear faces, so its volume is the integral of the Jacobian
    determinant of the trilinear map from the unit cube. The determinant is quadratic in each
    local coordinate (and the centroid integrand cubic), so 2 x 2 x 2 Gauss quadrature is
    exact. Unlike a tetrahedral split, the result does not depend on how a warped face is
    divided, and an inverted cell gets a negative volume.

    VTK hexahedron corner ordering, with `x` along `0 -> 1`, `y` along `0 -> 3` and depth along
    `0 -> 4` (so a regular cell has a positive volume):

        0=(x0,y0,zt)  1=(x1,y0,zt)  2=(x1,y1,zt)  3=(x0,y1,zt)
        4=(x0,y0,zb)  5=(x1,y0,zb)  6=(x1,y1,zb)  7=(x0,y1,zb)

    :param vtk_corner_indices: Shape `(n_cells, 8)` global vertex indices.
    :param vertex_coordinates: Shape `(n_verts, 3)` world coordinates.
    :returns: `(cell_volumes, cell_centroids)`.
    """
    n_cells = vtk_corner_indices.shape[0]
    cell_volumes = np.zeros(n_cells, dtype=np.float64)
    cell_centroids = np.zeros((n_cells, 3), dtype=np.float64)
    gauss_point = 0.5773502691896257
    node_signs = np.array([
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ])

    for cell_idx in numba.prange(n_cells):  # type: ignore
        # Coordinates relative to the first corner keep the arithmetic well conditioned.
        corners = np.empty((8, 3), dtype=np.float64)
        for node in range(8):
            for axis in range(3):
                corners[node, axis] = (
                    vertex_coordinates[vtk_corner_indices[cell_idx, node], axis]
                    - vertex_coordinates[vtk_corner_indices[cell_idx, 0], axis]
                )

        total_volume = 0.0
        weighted = np.zeros(3, dtype=np.float64)
        for i in range(2):
            xi = (2 * i - 1) * gauss_point
            for j in range(2):
                eta = (2 * j - 1) * gauss_point
                for k in range(2):
                    zeta = (2 * k - 1) * gauss_point
                    jacobian = np.zeros((3, 3), dtype=np.float64)
                    position = np.zeros(3, dtype=np.float64)
                    for node in range(8):
                        sx = node_signs[node, 0]
                        sy = node_signs[node, 1]
                        sz = node_signs[node, 2]
                        shape = (1 + sx * xi) * (1 + sy * eta) * (1 + sz * zeta) / 8.0
                        d_xi = sx * (1 + sy * eta) * (1 + sz * zeta) / 8.0
                        d_eta = sy * (1 + sx * xi) * (1 + sz * zeta) / 8.0
                        d_zeta = sz * (1 + sx * xi) * (1 + sy * eta) / 8.0
                        for axis in range(3):
                            jacobian[axis, 0] += d_xi * corners[node, axis]
                            jacobian[axis, 1] += d_eta * corners[node, axis]
                            jacobian[axis, 2] += d_zeta * corners[node, axis]
                            position[axis] += shape * corners[node, axis]
                    determinant = (
                        jacobian[0, 0]
                        * (jacobian[1, 1] * jacobian[2, 2] - jacobian[1, 2] * jacobian[2, 1])
                        - jacobian[0, 1]
                        * (jacobian[1, 0] * jacobian[2, 2] - jacobian[1, 2] * jacobian[2, 0])
                        + jacobian[0, 2]
                        * (jacobian[1, 0] * jacobian[2, 1] - jacobian[1, 1] * jacobian[2, 0])
                    )
                    total_volume += determinant
                    for axis in range(3):
                        weighted[axis] += determinant * position[axis]

        cell_volumes[cell_idx] = total_volume
        # A cell whose volume is almost, but not exactly, zero (a collapsed cell) gives a volume
        # weighted centroid that is a ratio of two rounding-level numbers and can land far
        # outside the cell. A genuine centroid always lies within the bounds of the corners, so
        # a centroid outside them is replaced by the mean of the corners.
        use_volume_weighted_centroid = abs(total_volume) > 0.0
        if use_volume_weighted_centroid:
            for axis in range(3):
                lowest = corners[0, axis]
                highest = corners[0, axis]
                for node in range(1, 8):
                    lowest = min(lowest, corners[node, axis])
                    highest = max(highest, corners[node, axis])
                slack = 1e-9 * (highest - lowest)
                candidate = weighted[axis] / total_volume
                if not (lowest - slack <= candidate <= highest + slack):
                    use_volume_weighted_centroid = False
                    break

        for axis in range(3):
            origin = vertex_coordinates[vtk_corner_indices[cell_idx, 0], axis]
            if use_volume_weighted_centroid:
                cell_centroids[cell_idx, axis] = origin + weighted[axis] / total_volume
            else:
                mean = 0.0
                for node in range(8):
                    mean += corners[node, axis]
                cell_centroids[cell_idx, axis] = origin + mean / 8.0

    return cell_volumes, cell_centroids


def rederive_corner_point_arrays(
    grid: Grid,
) -> tuple[CoordArray, ZCornArray, Integer, Integer, Integer]:
    """
    Reconstruct approximate `COORD` and `ZCORN` arrays from a `Grid`.

    The reconstruction uses each cell's AABB. Pillars are assumed straight
    and vertical, so this is lossy for grids with lateral pillar displacement.

    `(nx, ny, nz)` are taken from `grid.dimensions` when set, falling back
    to `grid.metadata['nx'/'ny'/'nz']`, and finally to factorising
    `grid.n_cells` if neither is available.

    If `grid.metadata['map_axes']` is set, the reconstructed pillars are
    transformed back to local (pre-`MAPAXES`) space before being packed
    into `coord`, so the result stays consistent with that same
    `MAPAXES` card being re-emitted alongside it (see `get_map_axes_xy_inverse`).

    :param grid: A `Grid` whose cells are stored in k-major, j-middle, i-minor order.
    :returns: Tuple `(coord_array, zcorn_array, nx, ny, nz)`.
    :raises GridExportError: If the cell count cannot be factored.
    """
    n_cells = grid.n_cells
    meta = getattr(grid, "metadata", {}) or {}

    if grid.dimensions is not None:
        nx, ny, nz = grid.dimensions
    else:
        nx = meta.get("nx")
        ny = meta.get("ny")
        nz = meta.get("nz")

    if nx is None or ny is None or nz is None:
        found = False
        for nz_try in range(1, n_cells + 1):
            if n_cells % nz_try != 0:
                continue
            nxy = n_cells // nz_try
            for nx_try in range(1, int(nxy**0.5) + 1):
                if nxy % nx_try == 0:
                    nx, ny, nz = nx_try, nxy // nx_try, nz_try
                    found = True
            if found:
                break

        if not found or (nx * ny * nz) != n_cells:  # type: ignore
            raise GridExportError(
                f"Cannot determine (nx, ny, nz) factorisation for "
                f"`n_cells={n_cells}`. Store 'nx', 'ny', 'nz' in "
                "`grid.metadata` to enable GRDECL export."
            )

    if int(nx) * int(ny) * int(nz) != n_cells:  # type: ignore[arg-type]
        raise GridExportError(
            f"Cannot reconstruct `COORD`/`ZCORN`: the grid has {n_cells} cells but its dimensions "
            f"({nx} x {ny} x {nz}) imply {int(nx) * int(ny) * int(nz)}."  # type: ignore[arg-type]
        )

    stored_coord = meta.get("coord")
    stored_zcorn = meta.get("zcorn")
    if stored_coord is not None and stored_zcorn is not None:
        return (
            np.asarray(stored_coord, dtype=np.float64).reshape(int(ny) + 1, int(nx) + 1, 6),  # type: ignore[arg-type]
            np.asarray(stored_zcorn, dtype=np.float64).reshape(
                int(nz) * 2,  # type: ignore[arg-type]
                int(ny) * 2,  # type: ignore[arg-type]
                int(nx) * 2,  # type: ignore[arg-type]
            ),
            int(nx),  # type: ignore[arg-type]
            int(ny),  # type: ignore[arg-type]
            int(nz),  # type: ignore[arg-type]
        )
    if grid.cell_statuses is not None and bool(
        (grid.cell_statuses == int(CellStatus.INACTIVE)).any()
    ):
        raise GridExportError(
            "Cannot reconstruct `COORD`/`ZCORN` for a grid with inactive cells: the geometry of "
            "inactive cells is not retained."
        )

    warnings.warn(
        "Exporting a corner-point Grid to GRDECL without stored `COORD`/`ZCORN` "
        "arrays. Pillars are reconstructed as straight vertical lines from "
        "cell bounding boxes. This is lossy for grids with lateral pillar "
        "displacement (faults, dipping layers).",
        stacklevel=3,
    )
    assert nx is not None and ny is not None and nz is not None

    pillar_x = np.zeros((ny + 1, nx + 1), dtype=np.float64)
    pillar_y = np.zeros((ny + 1, nx + 1), dtype=np.float64)
    pillar_z_top = np.full((ny + 1, nx + 1), np.inf, dtype=np.float64)
    pillar_z_bottom = np.full((ny + 1, nx + 1), -np.inf, dtype=np.float64)
    pillar_count = np.zeros((ny + 1, nx + 1), dtype=np.int32)

    _accumulate_pillars(
        cell_min_xyz=grid.cell_min_xyz,
        cell_max_xyz=grid.cell_max_xyz,
        nx=nx,
        ny=ny,
        nz=nz,
        pillar_x=pillar_x,
        pillar_y=pillar_y,
        pillar_z_top=pillar_z_top,
        pillar_z_bottom=pillar_z_bottom,
        pillar_count=pillar_count,
    )
    nonzero = pillar_count > 0
    pillar_x[nonzero] /= pillar_count[nonzero]
    pillar_y[nonzero] /= pillar_count[nonzero]

    map_axes: MapAxes | None = meta.get("map_axes")
    if map_axes is not None:
        # grid.cell_min_xyz/cell_max_xyz (and so pillar_x/pillar_y above)
        # are in map space whenever this grid was built with `MAPAXES`
        # applied - GRDECL's COORD is defined in local (pre-MAPAXES) space,
        # with the MAPAXES card re-emitted separately, so undo the areal
        # transform here before packing into `coord`. This is exact for
        # vertical pillars (the case this whole reconstruction already
        # assumes): MAPAXES doesn't depend on z, so a pillar that was
        # vertical - constant (x, y) across z - in local space is still
        # vertical, at a rotated/translated (x, y), in map space; nothing
        # extra is lost by inverting on the already-reduced pillar_x/
        # pillar_y rather than on the full per-cell vertex set.
        pillar_xy_local = get_map_axes_xy_inverse(
            xy=np.column_stack([pillar_x.ravel(), pillar_y.ravel()]),  # type: ignore[arg-type]
            map_axes=map_axes,
        )
        pillar_x = pillar_xy_local[:, 0].reshape(pillar_x.shape)
        pillar_y = pillar_xy_local[:, 1].reshape(pillar_y.shape)

    coord = np.empty((ny + 1, nx + 1, 6), dtype=np.float64)
    coord[:, :, 0] = pillar_x
    coord[:, :, 1] = pillar_y
    coord[:, :, 2] = pillar_z_top
    coord[:, :, 3] = pillar_x
    coord[:, :, 4] = pillar_y
    coord[:, :, 5] = pillar_z_bottom

    zcorn = np.empty((nz * 2, ny * 2, nx * 2), dtype=np.float64)
    _fill_zcorn(
        cell_min_xyz=grid.cell_min_xyz,
        cell_max_xyz=grid.cell_max_xyz,
        nx=nx,
        ny=ny,
        nz=nz,
        zcorn=zcorn,
    )
    return coord, zcorn, nx, ny, nz
