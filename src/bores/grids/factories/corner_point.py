"""
Corner-point (pillar) grid factory.

Builds a `bores.grids.base.Grid` from ECLIPSE-style COORD / ZCORN / ACTNUM arrays.

**Coordinate convention**: z-axis positive downward.

**Pinchout handling**: cells whose average thickness is at or below
`pinch_tolerance` have their top/bottom faces suppressed so that adjacent
active cells share those face keys. A face key claimed by a third cell is ignored for
that cell.

**Fault handling**: named faults from `fault_records` are first resolved to
shared face indices. Cell pairs in the fault IJK range that share no geometric
face are not connected and are skipped.
"""

import typing
import warnings

import numba
import numpy as np

from bores.datastructures import GridDimensions, MapAxes
from bores.errors import GridExportError, InvalidGridError, ValidationError
from bores.grids.base import CellStatus, ConnectionType, Grid
from bores.grids.factories.base import (
    FaceKey,
    FaceRecord,
    FaceVertexIndices,
    FaultRecord,
    VertexCoordinates,
    map_xy_to_map_space,
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

__all__ = ["make_corner_point_grid", "rederive_corner_point_arrays"]

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
    :raises ValidationError: On array shape mismatches or inconsistent NNC lengths.
    :raises InvalidGridError: If no active cells are found.
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
    Rotate and translate a COORD pillar array's `(x, y)` pairs into map space.

    Applied once, upstream of pillar interpolation (`coord` is the only
    array `compute_cell_corner_coordinates` reads for areal
    position), so every derived quantity - `vertex_coordinates`,
    `cell_centroids`, face geometry, comes out already correctly
    positioned; `cell_volumes` are unaffected, being invariant under
    rotation/translation. `z` (pillar depth, columns 2 and 5) is untouched,
    since `MAPAXES` is a purely areal transform.

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


@numba.njit(cache=True)
def _is_cell_pinched(
    vtk_vertices: list[Integer],
    vertex_coordinates: VertexCoordinates,
    pinch_tolerance: Number,
) -> Boolean:
    """
    Return `True` if the cell's average thickness is at or below `pinch_tolerance`.

    :param vtk_vertices: 8 global vertex indices in VTK hex order.
    :param vertex_coordinates: Shape `(n_verts, 3)` coordinate array.
    :param pinch_tolerance: Thickness threshold.
    :returns: `True` if the cell should be treated as pinched out.
    """
    top_set = set(vtk_vertices[:4])
    bottom_set = set(vtk_vertices[4:])
    if top_set == bottom_set:
        return True
    if pinch_tolerance <= 0.0:
        return False

    total_dz = 0.0
    for k in range(4):
        z_top = vertex_coordinates[vtk_vertices[k], 2]
        z_bottom = vertex_coordinates[vtk_vertices[k + 4], 2]
        total_dz += abs(z_bottom - z_top)
    return (total_dz / 4.0) <= pinch_tolerance


LATERAL_FACE_LOCAL_INDICES: tuple[Integer, ...] = (2, 3, 4, 5)
"""Local face indices of the four lateral faces (`-y`, `+y`, `-x`, `+x`)."""

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
    pieces: list[NumberArray[TwoDimensions]] = []
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


def connect_unmatched_faces(
    *,
    face_registry: dict[FaceKey, FaceRecord],
    face_local_index: dict[FaceKey, Integer],
    degenerate_faces: list[tuple[Integer, Integer]],
    pinched_cells: set[Integer],
    active_mask: BooleanArray[OneDimension],
    corner_coordinates: NumberArray[ThreeDimensions],
    coord: CoordArray,
    dimensions: tuple[Integer, Integer, Integer],
    vertex_coordinates: VertexCoordinates,
) -> VertexCoordinates:
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

    :param face_registry: Face registry built from exact vertex matches; modified in place.
    :param face_local_index: Local face index of each registered lateral or vertical face.
    :param degenerate_faces: `(cell, local face index)` of lateral or vertical faces that collapse
        to fewer than four distinct vertices and were not registered.
    :param pinched_cells: Flat indices of pinched-out cells.
    :param active_mask: Shape `(n_cells,)` activation mask.
    :param corner_coordinates: Shape `(n_cells, 8, 3)` corner coordinates of every cell.
    :param coord: Pillar array of shape `(ny + 1, nx + 1, 6)`.
    :param dimensions: Grid extents `(nx, ny, nz)`.
    :param vertex_coordinates: Existing merged vertex coordinates.
    :returns: Vertex coordinates with the vertices of the new faces appended.
    """
    nx, ny, nz = dimensions
    layer_size = nx * ny

    unmatched: dict[tuple[Integer, Integer], FaceKey | None] = {}
    for key, local_index in face_local_index.items():
        record = face_registry[key]
        if record.neighbour_cell_index == -1:
            unmatched[record.owner_cell_index, local_index] = key
    for cell, local_index in degenerate_faces:
        unmatched[cell, local_index] = None
    if not unmatched:
        return vertex_coordinates

    new_points: list[NumberArray[OneDimension]] = []
    next_vertex = len(vertex_coordinates)
    replaced: set[FaceKey] = set()
    cell_centers = corner_coordinates.mean(axis=1)

    def add_face(
        points: list[NumberArray[OneDimension]], owner: Integer, neighbour: Integer
    ) -> None:
        nonlocal next_vertex
        normal = np.zeros(3)
        for index, point in enumerate(points):
            normal += np.cross(point, points[(index + 1) % len(points)])
        face_center = np.mean(points, axis=0)
        if float(np.dot(normal, face_center - cell_centers[owner])) < 0.0:
            points = points[::-1]

        indices = typing.cast(
            FaceVertexIndices, list(range(next_vertex, next_vertex + len(points)))
        )
        next_vertex += len(points)
        new_points.extend(points)
        record = FaceRecord(owner_cell_index=owner, face_vertex_indices=indices)
        record.neighbour_cell_index = neighbour
        face_registry[tuple(sorted(indices))] = record

    # (plus local face, minus local face, plus corners, minus corners, pillar 1, pillar 2)
    directions = (
        (5, 4, (1, 3, 5, 7), (0, 2, 4, 6), (0, 1), (1, 1)),
        (3, 2, (2, 3, 6, 7), (0, 1, 4, 5), (1, 0), (1, 1)),
    )
    covers: dict[tuple[Integer, Integer], list[NumberArray[TwoDimensions]]] = {}
    face_geometry: dict[
        tuple[Integer, Integer],
        tuple[NumberArray[TwoDimensions], NumberArray[OneDimension], NumberArray[OneDimension]],
    ] = {}

    def lateral_polygon(cell: Integer, corners: tuple[Integer, ...]) -> NumberArray[TwoDimensions]:
        z = corner_coordinates[cell, :, 2]
        polygon = typing.cast(
            NumberArray[TwoDimensions],
            np.array(
                [
                    (0.0, z[corners[0]]),
                    (1.0, z[corners[1]]),
                    (1.0, z[corners[3]]),
                    (0.0, z[corners[2]]),
                ],
                dtype=np.float64,
            ),
        )
        if compute_polygon_signed_area(polygon) < 0.0:
            polygon = polygon[::-1].copy()
        return typing.cast(VertexCoordinates, polygon)

    for plus_local, minus_local, a_corners, b_corners, offset_1, offset_2 in directions:
        step_i, step_j = (1, 0) if plus_local == 5 else (0, 1)
        for cell_a, local_index in list(unmatched):
            if local_index != plus_local:
                continue

            _k_a, rest = divmod(cell_a, layer_size)  # type: ignore[arg-type]
            j_a, i_a = divmod(rest, nx)
            if i_a + step_i >= nx or j_a + step_j >= ny:
                continue

            pillar_1 = typing.cast(
                NumberArray[OneDimension], coord[j_a + offset_1[0], i_a + offset_1[1]]
            )
            pillar_2 = typing.cast(
                NumberArray[OneDimension], coord[j_a + offset_2[0], i_a + offset_2[1]]
            )
            polygon_a = lateral_polygon(cell_a, a_corners)
            area_a = abs(compute_polygon_signed_area(polygon_a))
            if area_a <= 0.0:
                continue
            face_geometry[cell_a, plus_local] = (polygon_a, pillar_1, pillar_2)

            for k_b in range(nz):
                cell_b = (i_a + step_i) + (j_a + step_j) * nx + k_b * layer_size
                if (cell_b, minus_local) not in unmatched:
                    continue
                polygon_b = lateral_polygon(cell_b, b_corners)
                area_b = abs(compute_polygon_signed_area(polygon_b))
                if area_b <= 0.0:
                    continue

                overlap = clip_convex_polygon(polygon_a, polygon_b)
                if overlap.shape[0] < 3 or abs(compute_polygon_signed_area(overlap)) <= (
                    MINIMUM_RELATIVE_OVERLAP_AREA * min(area_a, area_b)
                ):
                    continue

                face_geometry[cell_b, minus_local] = (polygon_b, pillar_1, pillar_2)
                add_face(
                    [
                        _get_point_on_pillar_pair(pillar_1, pillar_2, s_value, z_value)
                        for s_value, z_value in overlap
                    ],
                    owner=cell_a,
                    neighbour=cell_b,
                )
                covers.setdefault((cell_a, plus_local), []).append(overlap)
                covers.setdefault((cell_b, minus_local), []).append(overlap)

    for (cell, local_index), overlaps in covers.items():
        polygon, pillar_1, pillar_2 = face_geometry[cell, local_index]
        full_area = abs(compute_polygon_signed_area(polygon))
        pieces = [polygon]
        for overlap in overlaps:
            pieces = [rest for piece in pieces for rest in subtract_convex_polygon(piece, overlap)]
        for piece in pieces:
            if abs(compute_polygon_signed_area(piece)) > MINIMUM_RELATIVE_OVERLAP_AREA * full_area:
                add_face(
                    [
                        _get_point_on_pillar_pair(pillar_1, pillar_2, s_value, z_value)
                        for s_value, z_value in piece
                    ],
                    owner=cell,
                    neighbour=-1,
                )
        key = unmatched[cell, local_index]
        if key is not None:
            replaced.add(key)

    used_tops: set[Integer] = set()
    for (cell_a, local_index), key_a in list(unmatched.items()):
        if local_index != BOTTOM_FACE_LOCAL:
            continue
        cell_b = cell_a + layer_size
        while cell_b < nz * layer_size and active_mask[cell_b] and cell_b in pinched_cells:
            cell_b += layer_size
        if (
            cell_b >= nz * layer_size
            or not active_mask[cell_b]
            or cell_b in used_tops
            or (cell_b, TOP_FACE_LOCAL) not in unmatched
        ):
            continue
        bottom_corners = corner_coordinates[cell_a, 4:8]
        top_corners = corner_coordinates[cell_b, 0:4]
        middle = 0.5 * (bottom_corners + top_corners)
        footprint: list[NumberArray[OneDimension]] = []
        for corner in (0, 1, 3, 2):
            if not footprint or float(np.linalg.norm(middle[corner] - footprint[-1])) > 1e-9:
                footprint.append(middle[corner])
        if len(footprint) > 1 and float(np.linalg.norm(footprint[0] - footprint[-1])) <= 1e-9:
            footprint.pop()
        if len(footprint) < 3:
            continue
        normal = np.zeros(3)
        for index, point in enumerate(footprint):
            normal += np.cross(point, footprint[(index + 1) % len(footprint)])
        if float(np.linalg.norm(normal)) <= 0.0:
            continue
        add_face(footprint, owner=cell_a, neighbour=cell_b)
        used_tops.add(cell_b)
        for key in (key_a, unmatched[cell_b, TOP_FACE_LOCAL]):
            if key is not None:
                replaced.add(key)

    for key in replaced:
        face_registry.pop(key, None)
    if not new_points:
        return vertex_coordinates
    return typing.cast(
        VertexCoordinates,
        np.vstack([vertex_coordinates, np.asarray(new_points, dtype=np.float64)]),
    )


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


def compute_corner_point_geometry(
    coord: CoordArray,
    zcorn: ZCornArray,
    actnum: ActNumArray,
    vertex_tolerance: Number = 1e-8,
    pinch_tolerance: Number = 0.0,
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
    :returns: 10-tuple `(vertex_coordinates, face_vertex_indices,
        face_vertex_offsets, face_cell_indices, face_connection_types, cell_statuses,
        cell_volumes, cell_centroids, cell_min_xyz, cell_max_xyz)`. The last two are the
        bounding box of each cell's own corners. Cells are
        numbered by their full-grid flat index `i + j * nx + k * nx * ny`. Inactive cells
        keep their index, have status `INACTIVE`, zero volume and no faces.
    :raises InvalidGridError: If no active cells are found.
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
    quantized = np.round(flat_corners / vertex_tolerance).astype(np.int64, copy=False)
    _, unique_indices, inverse = np.unique(
        quantized, axis=0, return_index=True, return_inverse=True
    )
    vertex_coordinates = typing.cast(VertexCoordinates, flat_corners[unique_indices])
    n_cells = len(all_cells)
    corner_global = inverse.reshape(n_cells, 8)

    vtk_to_corner = [0, 1, 3, 2, 4, 5, 7, 6]

    face_registry: dict[FaceKey, FaceRecord] = {}
    n_ignored_third_claims = 0

    n_pinched = 0
    n_degenerate = 0
    face_local_index: dict[FaceKey, Integer] = {}
    degenerate_faces: list[tuple[Integer, Integer]] = []
    pinched_cells: set[Integer] = set()

    for cell_idx in np.flatnonzero(active_mask):
        vtk_vertices = [corner_global[cell_idx, vtk_to_corner[v]] for v in range(8)]
        pinched = _is_cell_pinched(
            vtk_vertices,
            vertex_coordinates,  # type: ignore[arg-type]
            pinch_tolerance,
        )
        if pinched:
            n_pinched += 1
            pinched_cells.add(int(cell_idx))

        for local_idx, local_face in enumerate(HEXAHEDRON_FACES_ZDOWN):
            face_vertex_indices = [vtk_vertices[v] for v in local_face]

            if len(set(face_vertex_indices)) < len(face_vertex_indices):
                n_degenerate += 1
                if local_idx in LATERAL_FACE_LOCAL_INDICES or not pinched:
                    degenerate_faces.append((int(cell_idx), local_idx))
                continue

            if pinched and local_idx in (TOP_FACE_LOCAL, BOTTOM_FACE_LOCAL):
                continue

            key: FaceKey = tuple(sorted(face_vertex_indices))
            if key not in face_registry:
                face_registry[key] = FaceRecord(
                    owner_cell_index=cell_idx,
                    face_vertex_indices=face_vertex_indices,
                )
                face_local_index[key] = local_idx
            elif face_registry[key].neighbour_cell_index == -1:
                face_registry[key].neighbour_cell_index = cell_idx
            else:
                n_ignored_third_claims += 1

    nz_cells, ny_cells, nx_cells = actnum.shape
    vertex_coordinates = connect_unmatched_faces(
        face_registry=face_registry,
        face_local_index=face_local_index,
        degenerate_faces=degenerate_faces,
        pinched_cells=pinched_cells,
        active_mask=active_mask,
        corner_coordinates=corner_coordinates,
        coord=coord,
        dimensions=(nx_cells, ny_cells, nz_cells),
        vertex_coordinates=vertex_coordinates,
    )

    if n_pinched > 0:
        warnings.warn(
            f"{n_pinched} pinched-out cell(s) detected "
            f"(pinch_tolerance={pinch_tolerance:.3g}). "
            f"{n_ignored_third_claims} face(s) claimed by a third cell were ignored.",
            stacklevel=4,
        )

    flat_face_vertex_indices: list[Integer] = []
    face_vertex_offsets: list[int] = [0]
    face_cell_pairs: list[tuple[Integer, Integer]] = []
    face_connection_types: list[int] = []

    for record in face_registry.values():
        flat_face_vertex_indices.extend(record.face_vertex_indices)
        face_vertex_offsets.append(len(flat_face_vertex_indices))
        face_cell_pairs.append((record.owner_cell_index, record.neighbour_cell_index))

        if record.neighbour_cell_index < 0:
            face_connection_types.append(int(ConnectionType.BOUNDARY_FACE))
        else:
            face_connection_types.append(int(ConnectionType.INTERIOR_FACE))

    vtk_corner_indices = np.empty((n_cells, 8), dtype=np.int32)
    for cell_idx in range(n_cells):
        for vertex in range(8):
            vtk_corner_indices[cell_idx, vertex] = corner_global[cell_idx, vtk_to_corner[vertex]]

    cell_volumes, cell_centroids = _compute_hex_volumes_and_centroids(
        vtk_corner_indices=vtk_corner_indices,
        vertex_coordinates=vertex_coordinates,  # type: ignore[arg-type]
    )
    box_volume = np.prod(corner_coordinates.max(axis=1) - corner_coordinates.min(axis=1), axis=1)
    round_off = (cell_volumes < 0.0) & (cell_volumes >= -1e-9 * box_volume)
    cell_volumes = typing.cast(NumberArray[OneDimension], np.where(round_off, 0.0, cell_volumes))
    inverted = active_mask & (cell_volumes < 0.0)
    if inverted.any():
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
        typing.cast(IntArray[OneDimension], np.asarray(flat_face_vertex_indices, dtype=np.int32)),
        typing.cast(IntArray[OneDimension], np.asarray(face_vertex_offsets, dtype=np.int32)),
        typing.cast(IntArray[TwoDimensions], np.asarray(face_cell_pairs, dtype=np.int32)),
        typing.cast(IntArray[OneDimension], np.asarray(face_connection_types, dtype=np.int8)),
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
    share no geometric face are not connected and are skipped.

    :param fault_records: Sequence of `FaultRecord` objects.
    :param dimensions: Grid extents `(nx, ny, nz)`.
    :param actnum: Shape `(nz, ny, nx)` activation mask.
    :param face_cell_indices: Shape `(n_faces, 2)`.
    :returns: Mapping from fault name to the indices of its faces.
    """
    nx, ny, nz = dimensions
    kji_to_cell: dict[tuple[int, int, int], int] = {}
    for k, j, i in np.argwhere(actnum > 0):
        kji_to_cell[int(k), int(j), int(i)] = int(i) + int(j) * nx + int(k) * nx * ny

    cell_pair_to_face: dict[frozenset[int], int] = {}
    for face_idx, (owner, neighbour) in enumerate(face_cell_indices):
        if owner >= 0 and neighbour >= 0:
            cell_pair_to_face[frozenset((int(owner), int(neighbour)))] = face_idx

    result: dict[str, list[Integer]] = {}

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

        face_indices: list[Integer] = []
        n_inactive = 0

        for k in range(record.k1 - 1, record.k2):
            for j in range(record.j1 - 1, record.j2):
                for i in range(record.i1 - 1, record.i2):
                    cell_a = kji_to_cell.get((k, j, i))
                    if cell_a is None:
                        n_inactive += 1
                        continue

                    lateral = di != 0 or dj != 0
                    neighbour_ks = range(nz) if lateral else (k + dk,)
                    found_neighbour = False
                    for neighbour_k in neighbour_ks:
                        cell_b = kji_to_cell.get((neighbour_k, j + dj, i + di))
                        if cell_b is None:
                            continue
                        found_neighbour = True
                        face_idx = cell_pair_to_face.get(frozenset((cell_a, cell_b)))
                        if face_idx is not None:
                            face_indices.append(face_idx)
                    if not found_neighbour:
                        n_inactive += 1

        if n_inactive > 0:
            warnings.warn(
                f"Fault {record.name!r}: {n_inactive} cell pair(s) skipped "
                f"(one or both cells inactive).",
                stacklevel=4,
            )

        if face_indices:
            existing = result.get(record.name)
            if existing is not None:
                existing.extend(face_indices)
            else:
                result[record.name] = face_indices

    return {
        name: typing.cast(IntArray[OneDimension], np.unique(np.asarray(idxs, dtype=np.int32)))
        for name, idxs in result.items()
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


@numba.njit(parallel=True, cache=True)
def _compute_hex_volumes_and_centroids(
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
        for axis in range(3):
            origin = vertex_coordinates[vtk_corner_indices[cell_idx, 0], axis]
            if abs(total_volume) > 0.0:
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
    Reconstruct approximate COORD and ZCORN arrays from a `Grid`.

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
