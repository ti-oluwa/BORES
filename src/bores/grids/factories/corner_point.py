"""
Corner-point (pillar) grid factory.

Builds a `bores.grids.base.Grid` from ECLIPSE-style COORD / ZCORN / ACTNUM arrays.

**Coordinate convention**: z-axis positive downward.

**Pinchout handling**: cells whose average thickness is at or below
`pinch_tolerance` have their top/bottom faces suppressed so that adjacent
active cells share those face keys. Third claimants on any face key are
recorded as NNCs of type `PINCHOUT_NNC`.

**Fault handling**: named faults from `fault_records` are first resolved to
shared face indices. Cell pairs in the fault IJK range that share no geometric
face are recorded as NNCs of type `FAULT` rather than silently discarded.
"""

import typing
import warnings

import numba
import numpy as np
import numpy.typing as npt

from bores.datastructures import GridDimensions, MapAxes
from bores.errors import GridExportError, InvalidGridError, ValidationError
from bores.grids.base import CellStatus, ConnectionType, Grid
from bores.grids.factories.base import (
    FaceKey,
    FaceRecord,
    FaultRecord,
    VertexCoordinates,
    map_xy_to_map_space,
)
from bores.types import (
    Boolean,
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


HEXAHEDRON_FACES_ZDOWN: list[list[int]] = [
    [0, 3, 2, 1],  # top    - outward normal = -z
    [4, 5, 6, 7],  # bottom - outward normal = +z
    [0, 1, 5, 4],  # -y face
    [3, 7, 6, 2],  # +y face
    [0, 4, 7, 3],  # -x face
    [1, 2, 6, 5],  # +x face
]

TOP_FACE_LOCAL: int = 0
BOTTOM_FACE_LOCAL: int = 1

FACE_DIRECTION_TO_LOCAL: dict[str, int] = {
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
    :param nnc_transmissibilities: Shape `(n_nnc,)` user-declared NNC transmissibilities.
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
        geometric_nnc_pairs,
        geometric_nnc_connection_types,
        cell_statuses,
        cell_volumes,
        cell_centroids,
    ) = compute_corner_point_geometry(
        coord=coord_array,  # type: ignore[arg-type]
        zcorn=zcorn_array,  # type: ignore[arg-type]
        actnum=actnum_array,
        vertex_tolerance=vertex_tolerance,
        pinch_tolerance=pinch_tolerance,
    )

    # Resolve fault face indices; cell pairs with no shared face become fault NNCs.
    fault_nnc_pairs: list[tuple[int, int, str]] = []
    fault_face_indices: dict[str, IntArray[OneDimension]] | None = None
    if fault_records:
        fault_face_indices, fault_nnc_pairs = resolve_fault_face_indices(
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

    # Merge all NNC sources: [geometry pinchouts] + [geometry fault NNCs] + [user NNCs]
    # Each source contributes a pairs array (n, 2), a types array (n,), and
    # optionally a transmissibilities array (n,).
    all_nnc_parts: list[
        tuple[
            npt.NDArray[np.int32],
            npt.NDArray[np.int8],
            npt.NDArray[np.float64],
        ]
    ] = []

    if geometric_nnc_pairs is not None and len(geometric_nnc_pairs) > 0:
        geometric_transmissibilities = np.full(len(geometric_nnc_pairs), np.nan, dtype=np.float64)
        all_nnc_parts.append((
            np.asarray(geometric_nnc_pairs, dtype=np.int32),
            geometric_nnc_connection_types,
            geometric_transmissibilities,
        ))

    fault_nnc_indices: dict[str, list[int]] = {}
    if fault_nnc_pairs:
        fault_pairs = np.asarray([(a, b) for a, b, _ in fault_nnc_pairs], dtype=np.int32).reshape(
            -1, 2
        )
        fault_connection_types = np.full(
            len(fault_nnc_pairs), int(ConnectionType.FAULT_NNC), dtype=np.int8
        )
        fault_transmissibilities = np.full(len(fault_nnc_pairs), np.nan, dtype=np.float64)
        all_nnc_parts.append((
            fault_pairs,
            fault_connection_types,
            fault_transmissibilities,
        ))
        # Build nnc_fault_indices: fault name -> positions into the merged NNC array.
        # The offset is the total NNC count already accumulated before this block.
        fault_nnc_offset = sum(len(p) for p, _, _ in all_nnc_parts[:-1])
        for local_idx, (_, _, name) in enumerate(fault_nnc_pairs):
            fault_nnc_indices.setdefault(name, []).append(fault_nnc_offset + local_idx)

    if nnc_cell_indices is not None and len(nnc_cell_indices) > 0:
        user_nnc_pairs = np.asarray(nnc_cell_indices, dtype=np.int32).reshape(-1, 2)
        user_nnc_transmissibilities = (
            np.asarray(nnc_transmissibilities, dtype=np.float64)
            if nnc_transmissibilities is not None
            else np.full(len(user_nnc_pairs), np.nan, dtype=np.float64)
        )
        if len(user_nnc_transmissibilities) != len(user_nnc_pairs):
            raise ValidationError(
                f"`nnc_cell_indices` has {len(user_nnc_pairs)} pairs but `nnc_transmissibilities` "
                f"has {len(user_nnc_transmissibilities)} values."
            )
        if user_nnc_pairs.min() < 0 or user_nnc_pairs.max() >= cell_statuses.shape[0]:
            raise ValidationError(
                f"`nnc_cell_indices` must lie in [0, {cell_statuses.shape[0] - 1}] "
                f"(flat index i + j * nx + k * nx * ny)."
            )

        keep = (cell_statuses[user_nnc_pairs[:, 0]] == int(CellStatus.ACTIVE)) & (
            cell_statuses[user_nnc_pairs[:, 1]] == int(CellStatus.ACTIVE)
        )
        if not keep.all():
            warnings.warn(
                f"Dropped {int((~keep).sum())} user NNC(s) connected to inactive cells.",
                stacklevel=3,
            )
            user_nnc_pairs = user_nnc_pairs[keep]
            user_nnc_transmissibilities = user_nnc_transmissibilities[keep]

        user_nnc_connection_types = np.full(
            len(user_nnc_pairs), int(ConnectionType.USER_NNC), dtype=np.int8
        )
        all_nnc_parts.append((
            user_nnc_pairs,
            user_nnc_connection_types,
            user_nnc_transmissibilities,
        ))

    merged_nnc_pairs: npt.NDArray[np.int32] | None = None
    merged_nnc_connection_types: npt.NDArray[np.int8] | None = None
    merged_nnc_transmissibilities: npt.NDArray[np.float64] | None = None
    merged_nnc_fault_indices: dict[str, IntArray[OneDimension]] | None = None

    if all_nnc_parts:
        merged_nnc_pairs = np.vstack([p for p, _, _ in all_nnc_parts]).astype(np.int32)
        merged_nnc_connection_types = np.concatenate([t for _, t, _ in all_nnc_parts]).astype(
            np.int8, copy=False
        )
        merged_transmissibilities = np.concatenate([t for _, _, t in all_nnc_parts])
        # Only store if at least one value is finite (avoids all-NaN array)
        merged_nnc_transmissibilities = (
            merged_transmissibilities if np.any(np.isfinite(merged_transmissibilities)) else None
        )
        if fault_nnc_pairs:
            merged_nnc_fault_indices = {  # type: ignore[arg-type]
                name: np.asarray(idxs, dtype=np.int32) for name, idxs in fault_nnc_indices.items()
            }

    return Grid(
        vertex_coordinates=vertex_coordinates,
        face_vertex_indices=face_vertex_indices,
        face_vertex_offsets=face_vertex_offsets,
        face_cell_indices=face_cell_indices,
        cell_volumes=cell_volumes,
        cell_centroids=cell_centroids,
        unit_system=unit_system,
        dimensions=GridDimensions(nx, ny, nz),
        metadata=metadata,
        cell_statuses=cell_statuses,  # type: ignore[arg-type]
        face_connection_types=face_connection_types,  # type: ignore[arg-type]
        nnc_cell_indices=merged_nnc_pairs,  # type: ignore[arg-type]
        nnc_connection_types=merged_nnc_connection_types,  # type: ignore[arg-type]
        nnc_transmissibilities=merged_nnc_transmissibilities,  # type: ignore[arg-type]
        nnc_fault_indices=merged_nnc_fault_indices,
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


LATERAL_FACE_LOCAL_INDICES: tuple[int, ...] = (2, 3, 4, 5)
"""Local face indices of the four lateral faces (`-y`, `+y`, `-x`, `+x`)."""

MINIMUM_RELATIVE_OVERLAP_AREA: float = 1e-6
"""Overlaps smaller than this fraction of the smaller of the two facing faces are ignored."""


def polygon_signed_area(polygon: list[tuple[float, float]]) -> float:
    """
    Signed area of a 2-D polygon (positive when the vertices run counter-clockwise).

    :param polygon: Polygon vertices.
    :returns: Signed area.
    """
    total = 0.0
    for index, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(index + 1) % len(polygon)]
        total += x1 * y2 - x2 * y1
    return 0.5 * total


def clip_convex_polygon(
    subject: list[tuple[float, float]],
    clip: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """
    Intersect two convex 2-D polygons (Sutherland-Hodgman clipping).

    :param subject: Polygon to clip, vertices counter-clockwise.
    :param clip: Convex clipping polygon, vertices counter-clockwise.
    :returns: Vertices of the intersection, empty if the polygons do not overlap.
    """

    def is_inside(
        point: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
    ) -> bool:
        return (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0]) >= 0.0

    def crossing(
        p1: tuple[float, float],
        p2: tuple[float, float],
        a: tuple[float, float],
        b: tuple[float, float],
    ) -> tuple[float, float]:
        d1 = (p2[0] - p1[0], p2[1] - p1[1])
        d2 = (b[0] - a[0], b[1] - a[1])
        denominator = d1[0] * d2[1] - d1[1] * d2[0]
        if denominator == 0.0:
            return p2
        t = ((a[0] - p1[0]) * d2[1] - (a[1] - p1[1]) * d2[0]) / denominator
        return (p1[0] + t * d1[0], p1[1] + t * d1[1])

    output = list(subject)
    for index in range(len(clip)):
        a = clip[index]
        b = clip[(index + 1) % len(clip)]
        incoming = output
        output = []
        if not incoming:
            break
        previous = incoming[-1]
        for current in incoming:
            if is_inside(current, a, b):
                if not is_inside(previous, a, b):
                    output.append(crossing(previous, current, a, b))
                output.append(current)
            elif is_inside(previous, a, b):
                output.append(crossing(previous, current, a, b))
            previous = current
    return output


def connect_overlapping_lateral_faces(
    *,
    face_registry: dict[FaceKey, FaceRecord],
    lateral_face_local_index: dict[FaceKey, int],
    degenerate_lateral_faces: list[tuple[int, int]],
    corner_coordinates: NumberArray[ThreeDimensions],
    coord: CoordArray,
    dimensions: tuple[int, int, int],
    vertex_coordinates: VertexCoordinates,
) -> VertexCoordinates:
    """
    Connect cells whose lateral faces overlap without sharing all four vertices.

    Across a fault, the neighbouring column's cells may sit at a different depth, so a lateral
    face can face several cells of the next column, each only partly. For every unmatched
    lateral face the overlap with each facing cell of the adjacent column (any `k`) is
    computed as a polygon in the surface spanned by the two shared pillars. Each overlap
    becomes an interior face between the two cells, and the unmatched faces it replaces are
    removed from `face_registry`.

    :param face_registry: Face registry built from exact vertex matches; modified in place.
    :param lateral_face_local_index: Local face index of each registered lateral face.
    :param degenerate_lateral_faces: `(cell, local face index)` of lateral faces that collapse to
        a triangle or line and were not registered.
    :param corner_coordinates: Shape `(n_cells, 8, 3)` corner coordinates of every cell.
    :param coord: Pillar array of shape `(ny + 1, nx + 1, 6)`.
    :param dimensions: Grid extents `(nx, ny, nz)`.
    :param vertex_coordinates: Existing merged vertex coordinates.
    :returns: Vertex coordinates with the vertices of the new overlap faces appended.
    """
    nx, ny, nz = dimensions
    unmatched: dict[tuple[int, int], FaceKey | None] = {}
    for key, local_index in lateral_face_local_index.items():
        record = face_registry[key]
        if record.neighbour_cell_index == -1:
            unmatched[record.owner_cell_index, local_index] = key
    for cell, local_index in degenerate_lateral_faces:
        unmatched[cell, local_index] = None
    if not unmatched:
        return vertex_coordinates

    # (plus-side local face, minus-side local face, A corners, B corners, pillar offsets)
    directions = (
        (5, 4, (1, 3, 5, 7), (0, 2, 4, 6), (0, 1), (1, 1)),
        (3, 2, (2, 3, 6, 7), (0, 1, 4, 5), (1, 0), (1, 1)),
    )
    new_points: list[NumberArray[OneDimension]] = []
    next_vertex = len(vertex_coordinates)
    replaced: set[FaceKey] = set()

    for (
        plus_local,
        minus_local,
        a_corners,
        b_corners,
        pillar_1_offset,
        pillar_2_offset,
    ) in directions:
        step_i, step_j = (1, 0) if plus_local == 5 else (0, 1)
        for (cell_a, local_index), key_a in list(unmatched.items()):
            if local_index != plus_local:
                continue
            _k_a, rest = divmod(cell_a, nx * ny)
            j_a, i_a = divmod(rest, nx)
            if i_a + step_i >= nx or j_a + step_j >= ny:
                continue

            pillar_1 = coord[j_a + pillar_1_offset[0], i_a + pillar_1_offset[1]]
            pillar_2 = coord[j_a + pillar_2_offset[0], i_a + pillar_2_offset[1]]
            z_a = corner_coordinates[cell_a, :, 2]
            polygon_a = [
                (0.0, z_a[a_corners[0]]),
                (1.0, z_a[a_corners[1]]),
                (1.0, z_a[a_corners[3]]),
                (0.0, z_a[a_corners[2]]),
            ]
            area_a = abs(polygon_signed_area(polygon_a))
            if area_a <= 0.0:
                continue
            if polygon_signed_area(polygon_a) < 0.0:
                polygon_a.reverse()

            outward_corners = [
                corner_coordinates[cell_a, c]
                for c in (a_corners[0], a_corners[1], a_corners[3], a_corners[2])
            ]
            reference_normal = np.cross(
                outward_corners[2] - outward_corners[0], outward_corners[3] - outward_corners[1]
            )

            for k_b in range(nz):
                cell_b = (i_a + step_i) + (j_a + step_j) * nx + k_b * nx * ny
                if (cell_b, minus_local) not in unmatched:
                    continue
                z_b = corner_coordinates[cell_b, :, 2]
                polygon_b = [
                    (0.0, z_b[b_corners[0]]),
                    (1.0, z_b[b_corners[1]]),
                    (1.0, z_b[b_corners[3]]),
                    (0.0, z_b[b_corners[2]]),
                ]
                area_b = abs(polygon_signed_area(polygon_b))
                if area_b <= 0.0:
                    continue
                if polygon_signed_area(polygon_b) < 0.0:
                    polygon_b.reverse()

                overlap = clip_convex_polygon(polygon_a, polygon_b)
                if len(overlap) < 3:
                    continue
                if abs(polygon_signed_area(overlap)) <= MINIMUM_RELATIVE_OVERLAP_AREA * min(
                    area_a, area_b
                ):
                    continue

                points: list[NumberArray[OneDimension]] = []
                for s_value, z_value in overlap:
                    p1 = _interpolate_pillar_point(pillar_1[:3], pillar_1[3:], z_value)
                    p2 = _interpolate_pillar_point(pillar_2[:3], pillar_2[3:], z_value)
                    points.append(p1 + s_value * (p2 - p1))
                normal = np.zeros(3)
                for index, point in enumerate(points):
                    following = points[(index + 1) % len(points)]
                    normal += np.cross(point, following)
                if float(np.dot(normal, reference_normal)) < 0.0:
                    points.reverse()

                indices = list(range(next_vertex, next_vertex + len(points)))
                next_vertex += len(points)
                new_points.extend(points)
                new_key: FaceKey = tuple(sorted(indices))
                record = FaceRecord(owner_cell_index=cell_a, face_vertex_indices=indices)
                record.neighbour_cell_index = cell_b
                face_registry[new_key] = record
                key_b = unmatched[cell_b, minus_local]
                if key_a is not None:
                    replaced.add(key_a)
                if key_b is not None:
                    replaced.add(key_b)

    for key in replaced:
        face_registry.pop(key, None)
    if not new_points:
        return vertex_coordinates
    return np.vstack([vertex_coordinates, np.asarray(new_points, dtype=np.float64)])


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
    npt.NDArray[np.int8],
    npt.NDArray[np.int32] | None,
    npt.NDArray[np.int8],
    npt.NDArray[np.int8],
    NumberArray[OneDimension],
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
        face_vertex_offsets, face_cell_indices, face_connection_types, geometric_nnc_pairs,
        geometric_nnc_connection_types, cell_statuses, cell_volumes, cell_centroids)`. Cells are
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
    vertex_coordinates = flat_corners[unique_indices]
    n_cells = len(all_cells)
    corner_global = inverse.reshape(n_cells, 8)

    vtk_to_corner = [0, 1, 3, 2, 4, 5, 7, 6]

    face_registry: dict[FaceKey, FaceRecord] = {}
    nnc_pairs: list[tuple[Integer, Integer]] = []
    nnc_pair_types: list[int] = []
    nnc_face_keys: set[FaceKey] = set()

    n_pinched = 0
    n_degenerate = 0
    lateral_face_local_index: dict[FaceKey, int] = {}
    degenerate_lateral_faces: list[tuple[int, int]] = []

    for cell_idx in np.flatnonzero(active_mask):
        vtk_vertices = [corner_global[cell_idx, vtk_to_corner[v]] for v in range(8)]
        pinched = _is_cell_pinched(
            vtk_vertices,
            vertex_coordinates,  # type: ignore[arg-type]
            pinch_tolerance,
        )
        if pinched:
            n_pinched += 1

        for local_idx, local_face in enumerate(HEXAHEDRON_FACES_ZDOWN):
            face_vertex_indices = [vtk_vertices[v] for v in local_face]

            if len(set(face_vertex_indices)) < len(face_vertex_indices):
                n_degenerate += 1
                if local_idx in LATERAL_FACE_LOCAL_INDICES:
                    degenerate_lateral_faces.append((int(cell_idx), local_idx))
                continue

            if pinched and local_idx in (TOP_FACE_LOCAL, BOTTOM_FACE_LOCAL):
                continue

            key: FaceKey = tuple(sorted(face_vertex_indices))
            if key not in face_registry:
                face_registry[key] = FaceRecord(
                    owner_cell_index=cell_idx,
                    face_vertex_indices=face_vertex_indices,
                )
                if local_idx in LATERAL_FACE_LOCAL_INDICES:
                    lateral_face_local_index[key] = local_idx
            elif face_registry[key].neighbour_cell_index == -1:
                face_registry[key].neighbour_cell_index = cell_idx
            else:
                existing = face_registry[key]
                nnc_pairs.append((existing.owner_cell_index, cell_idx))
                nnc_pair_types.append(int(ConnectionType.PINCHOUT_NNC))
                nnc_face_keys.add(key)

    nz_cells, ny_cells, nx_cells = actnum.shape
    vertex_coordinates = connect_overlapping_lateral_faces(
        face_registry=face_registry,
        lateral_face_local_index=lateral_face_local_index,
        degenerate_lateral_faces=degenerate_lateral_faces,
        corner_coordinates=corner_coordinates,
        coord=coord,
        dimensions=(nx_cells, ny_cells, nz_cells),
        vertex_coordinates=vertex_coordinates,
    )

    if n_pinched > 0:
        n_pinchout_nncs = sum(
            1 for typ in nnc_pair_types if typ == int(ConnectionType.PINCHOUT_NNC)
        )
        warnings.warn(
            f"{n_pinched} pinched-out cell(s) detected "
            f"(pinch_tolerance={pinch_tolerance:.3g}). "
            f"{n_pinchout_nncs} PINCHOUT NNC pair(s) recorded.",
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

    nnc_array: npt.NDArray[np.int32] | None = None
    nnc_connection_types_array: npt.NDArray[np.int8] = np.empty(0, dtype=np.int8)
    if nnc_pairs:
        nnc_array = np.asarray(nnc_pairs, dtype=np.int32).reshape(-1, 2)
        nnc_connection_types_array = np.asarray(nnc_pair_types, dtype=np.int8)

    vtk_corner_indices = np.empty((n_cells, 8), dtype=np.int32)
    for cell_idx in range(n_cells):
        for vertex in range(8):
            vtk_corner_indices[cell_idx, vertex] = corner_global[cell_idx, vtk_to_corner[vertex]]

    cell_volumes, cell_centroids = _compute_hex_volumes_and_centroids(
        vtk_corner_indices=vtk_corner_indices,
        vertex_coordinates=vertex_coordinates,  # type: ignore[arg-type]
    )
    cell_volumes = np.where(active_mask, cell_volumes, 0.0)
    cell_statuses = np.where(active_mask, int(CellStatus.ACTIVE), int(CellStatus.INACTIVE)).astype(
        np.int8
    )
    return (  # type: ignore[return-value]
        vertex_coordinates,
        np.asarray(flat_face_vertex_indices, dtype=np.int32),
        np.asarray(face_vertex_offsets, dtype=np.int32),
        np.asarray(face_cell_pairs, dtype=np.int32),
        np.asarray(face_connection_types, dtype=np.int8),
        nnc_array,
        nnc_connection_types_array,
        cell_statuses,
        cell_volumes,
        cell_centroids,
    )


def resolve_fault_face_indices(
    fault_records: typing.Sequence[FaultRecord],
    dimensions: tuple[Integer, Integer, Integer],
    actnum: ActNumArray,
    face_cell_indices: IntArray[TwoDimensions],
) -> tuple[
    dict[str, IntArray[OneDimension]],
    list[tuple[int, int, str]],
]:
    """
    Resolve `FaultRecord` IJK ranges to unstructured face index arrays.

    Cell pairs in the fault range that have no shared geometric face are returned
    as NNC pairs of type `FAULT` instead of being silently skipped.

    :param fault_records: Sequence of `FaultRecord` objects.
    :param dimensions: Grid extents `(nx, ny, nz)`.
    :param actnum: Shape `(nz, ny, nx)` activation mask.
    :param face_cell_indices: Shape `(n_faces, 2)`.
    :returns: Tuple `(fault_face_index_dict, fault_nnc_pairs)`.
    """
    nx, ny, _ = dimensions
    kji_to_cell: dict[tuple[int, int, int], int] = {}
    for k, j, i in np.argwhere(actnum > 0):
        kji_to_cell[int(k), int(j), int(i)] = int(i) + int(j) * nx + int(k) * nx * ny

    cell_pair_to_face: dict[frozenset[int], int] = {}
    for face_idx, (owner, neighbour) in enumerate(face_cell_indices):
        if owner >= 0 and neighbour >= 0:
            cell_pair_to_face[frozenset((int(owner), int(neighbour)))] = face_idx

    result: dict[str, list[int]] = {}
    fault_nnc_pairs: list[tuple[int, int, str]] = []

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

        face_indices: list[int] = []
        n_inactive = 0

        for k in range(record.k1 - 1, record.k2):
            for j in range(record.j1 - 1, record.j2):
                for i in range(record.i1 - 1, record.i2):
                    cell_a = kji_to_cell.get((k, j, i))
                    cell_b = kji_to_cell.get((k + dk, j + dj, i + di))

                    if cell_a is None or cell_b is None:
                        n_inactive += 1
                        continue

                    face_idx = cell_pair_to_face.get(frozenset((cell_a, cell_b)))
                    if face_idx is not None:
                        face_indices.append(face_idx)
                    else:
                        # No shared geometric face -> record as fault NNC
                        fault_nnc_pairs.append((cell_a, cell_b, record.name))

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

    return (  # type: ignore[return-value]
        {name: np.unique(np.asarray(idxs, dtype=np.int32)) for name, idxs in result.items()},
        fault_nnc_pairs,
    )


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
    Compute hexahedral cell volumes and centroids via 5-tetrahedron decomposition.

    VTK hexahedron corner ordering:

        0=(x0,y0,zt)  1=(x1,y0,zt)  2=(x1,y1,zt)  3=(x0,y1,zt)
        4=(x0,y0,zb)  5=(x1,y0,zb)  6=(x1,y1,zb)  7=(x0,y1,zb)

    5-tet decomposition:

        T0: [0,1,3,4]  T1: [1,4,5,6]  T2: [1,3,4,6]
        T3: [1,2,3,6]  T4: [3,4,6,7]

    Uses absolute scalar triple product - robust for distorted/inverted cells.

    :param vtk_corner_indices: Shape `(n_cells, 8)` global vertex indices.
    :param vertex_coordinates: Shape `(n_verts, 3)` world coordinates.
    :returns: `(cell_volumes, cell_centroids)`.
    """
    n_cells = vtk_corner_indices.shape[0]
    cell_volumes = np.zeros(n_cells, dtype=np.float64)
    cell_centroids = np.zeros((n_cells, 3), dtype=np.float64)

    for cell_idx in numba.prange(n_cells):  # type: ignore
        total_volume = 0.0
        wcx = 0.0
        wcy = 0.0
        wcz = 0.0

        for t in range(5):
            if t == 0:
                l0, l1, l2, l3 = 0, 1, 3, 4
            elif t == 1:
                l0, l1, l2, l3 = 1, 4, 5, 6
            elif t == 2:
                l0, l1, l2, l3 = 1, 3, 4, 6
            elif t == 3:
                l0, l1, l2, l3 = 1, 2, 3, 6
            else:
                l0, l1, l2, l3 = 3, 4, 6, 7

            g0 = vtk_corner_indices[cell_idx, l0]
            g1 = vtk_corner_indices[cell_idx, l1]
            g2 = vtk_corner_indices[cell_idx, l2]
            g3 = vtk_corner_indices[cell_idx, l3]

            x0 = vertex_coordinates[g0, 0]
            y0 = vertex_coordinates[g0, 1]
            z0 = vertex_coordinates[g0, 2]

            ax = vertex_coordinates[g1, 0] - x0
            ay = vertex_coordinates[g1, 1] - y0
            az = vertex_coordinates[g1, 2] - z0

            bx = vertex_coordinates[g2, 0] - x0
            by = vertex_coordinates[g2, 1] - y0
            bz = vertex_coordinates[g2, 2] - z0

            cx = vertex_coordinates[g3, 0] - x0
            cy = vertex_coordinates[g3, 1] - y0
            cz = vertex_coordinates[g3, 2] - z0

            cross_x = by * cz - bz * cy
            cross_y = bz * cx - bx * cz
            cross_z = bx * cy - by * cx
            tetrahedron_volume = abs(ax * cross_x + ay * cross_y + az * cross_z) / 6.0

            tetrahedron_cx = (
                x0
                + vertex_coordinates[g1, 0]
                + vertex_coordinates[g2, 0]
                + vertex_coordinates[g3, 0]
            ) * 0.25
            tetrahedron_cy = (
                y0
                + vertex_coordinates[g1, 1]
                + vertex_coordinates[g2, 1]
                + vertex_coordinates[g3, 1]
            ) * 0.25
            tetrahedron_cz = (
                z0
                + vertex_coordinates[g1, 2]
                + vertex_coordinates[g2, 2]
                + vertex_coordinates[g3, 2]
            ) * 0.25

            total_volume += tetrahedron_volume
            wcx += tetrahedron_volume * tetrahedron_cx
            wcy += tetrahedron_volume * tetrahedron_cy
            wcz += tetrahedron_volume * tetrahedron_cz

        cell_volumes[cell_idx] = total_volume
        if total_volume > 0.0:
            cell_centroids[cell_idx, 0] = wcx / total_volume
            cell_centroids[cell_idx, 1] = wcy / total_volume
            cell_centroids[cell_idx, 2] = wcz / total_volume

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
