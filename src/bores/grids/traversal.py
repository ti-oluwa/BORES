"""Walking a straight segment through a grid, cell by cell, using each cell's own faces."""

import math
import typing

import numba
import numpy as np

from bores.grids.base import Grid
from bores.types import (
    Boolean,
    IntArray,
    Integer,
    Number,
    NumberArray,
    OneDimension,
    TwoDimensions,
)

__all__ = [
    "check_point_in_polygon_2d",
    "check_point_in_polygon_3d",
    "check_point_on_segment",
    "get_segment_face_intersection",
    "walk_segment_cells",
]

PARALLEL_TOLERANCE = 1e-12
"""
A segment whose direction has a component along a face's normal smaller than this is treated as
parallel to the face and never crosses it.
"""

CROSSING_TOLERANCE = 1e-9
"""
Fraction of the segment's length within which a crossing counts as the current position, or as
the segment's end.
"""

INITIAL_CELL_CAPACITY = 32
"""Number of cells the walk reserves room for before it has to grow its results."""


@numba.njit(cache=True)
def check_point_on_segment(
    px: Number,
    py: Number,
    x1: Number,
    y1: Number,
    x2: Number,
    y2: Number,
    tolerance: Number,
) -> Boolean:
    """
    Whether `(px, py)` lies on the segment `(x1, y1)` to `(x2, y2)`, within `tolerance`.

    :param px: x coordinate of the point.
    :param py: y coordinate of the point.
    :param x1: x coordinate of the segment's start.
    :param y1: y coordinate of the segment's start.
    :param x2: x coordinate of the segment's end.
    :param y2: y coordinate of the segment's end.
    :param tolerance: Distance from the segment within which the point counts as on it.
    :returns: `True` if the point is on the segment.
    """
    segment_length = math.hypot(x2 - x1, y2 - y1)
    if segment_length < tolerance:
        return math.hypot(px - x1, py - y1) < tolerance

    cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    if abs(cross) / segment_length > tolerance:
        return False
    dot = (px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)
    return -tolerance <= dot <= (segment_length**2 + tolerance)


@numba.njit(cache=True)
def point_in_indexed_polygon(
    x: Number,
    y: Number,
    u_axis: Integer,
    v_axis: Integer,
    vertex_indices: IntArray[OneDimension],
    vertex_coordinates: NumberArray[TwoDimensions],
    tolerance: Number,
) -> Boolean:
    """
    Even-odd point-in-polygon test on two chosen coordinate axes, counting a point on the
    polygon's boundary as inside.

    :param x: Coordinate of the point along `u_axis`.
    :param y: Coordinate of the point along `v_axis`.
    :param u_axis: Index of the coordinate that plays the part of `x`.
    :param v_axis: Index of the coordinate that plays the part of `y`.
    :param vertex_indices: Rows of `vertex_coordinates` that make up the polygon, in order.
    :param vertex_coordinates: Shape `(n_vertices, n_axes)` vertex positions.
    :param tolerance: Distance from the boundary within which a point counts as on it.
    :returns: `True` if the point is inside or on the boundary of the polygon.
    """
    count = vertex_indices.shape[0]
    inside = False
    previous = count - 1
    for current in range(count):
        xi = vertex_coordinates[vertex_indices[current], u_axis]
        yi = vertex_coordinates[vertex_indices[current], v_axis]
        xj = vertex_coordinates[vertex_indices[previous], u_axis]
        yj = vertex_coordinates[vertex_indices[previous], v_axis]
        if check_point_on_segment(x, y, xi, yi, xj, yj, tolerance):
            return True
        if (yi > y) != (yj > y):
            x_intersect = xi + (y - yi) * (xj - xi) / (yj - yi)
            if x < x_intersect:
                inside = not inside
        previous = current
    return inside


@numba.njit(cache=True)
def dominant_axis_of(nx: Number, ny: Number, nz: Number) -> Integer:
    """The axis a normal is most aligned with, the lowest axis if several tie."""
    axis = 0
    largest = abs(nx)
    if abs(ny) > largest:
        axis = 1
        largest = abs(ny)
    if abs(nz) > largest:
        axis = 2
    return axis


@numba.njit(cache=True)
def check_point_in_polygon_2d(
    x: Number,
    y: Number,
    polygon_xy: NumberArray[TwoDimensions],
    tolerance: Number = 1e-9,
) -> Boolean:
    """
    Even-odd ray-casting point-in-polygon test, with a boundary tolerance so a point falling
    exactly on a shared cell face is counted as inside rather than dropped by roundoff.

    :param x: Query x-coordinate.
    :param y: Query y-coordinate.
    :param polygon_xy: Shape `(n, 2)` polygon vertices, in order.
    :param tolerance: Distance tolerance for the boundary case.
    :returns: `True` if `(x, y)` is inside or on the boundary of the polygon.
    """
    return point_in_indexed_polygon(
        x=x,
        y=y,
        u_axis=0,
        v_axis=1,
        vertex_indices=np.arange(polygon_xy.shape[0]),
        vertex_coordinates=polygon_xy,
        tolerance=tolerance,
    )


@numba.njit(cache=True)
def check_point_in_polygon_3d(
    point: tuple[Number, Number, Number],
    polygon_xyz: NumberArray[TwoDimensions],
    normal: NumberArray[OneDimension],
    tolerance: Number = 1e-9,
) -> Boolean:
    """
    Point-in-polygon test for a planar polygon in 3-D, given a point already known to lie on (or
    very near) the polygon's plane.

    Both are projected onto the pair of axes the normal is least aligned with, so a steeply tilted
    face keeps a usable area, and the 2-D test is run on that projection.

    :param point: 3-D point on the polygon's plane.
    :param polygon_xyz: Shape `(n, 3)` polygon vertices, in order.
    :param normal: The polygon's unit normal.
    :param tolerance: Distance tolerance for the boundary case.
    :returns: `True` if `point` is inside or on the boundary of the polygon.
    """
    dominant = dominant_axis_of(nx=normal[0], ny=normal[1], nz=normal[2])
    u_axis = 1 if dominant == 0 else 0
    v_axis = 1 if dominant == 2 else 2
    return point_in_indexed_polygon(
        x=point[u_axis],
        y=point[v_axis],
        u_axis=u_axis,
        v_axis=v_axis,
        vertex_indices=np.arange(polygon_xyz.shape[0]),
        vertex_coordinates=polygon_xyz,
        tolerance=tolerance,
    )


@numba.njit(cache=True)
def face_plane_crossing(
    sx: Number,
    sy: Number,
    sz: Number,
    dx: Number,
    dy: Number,
    dz: Number,
    face: Integer,
    cell: Integer,
    face_cell_indices: IntArray[TwoDimensions],
    face_unit_normals: NumberArray[TwoDimensions],
    face_centroids: NumberArray[TwoDimensions],
) -> Number:
    """
    Where a segment crosses a face's plane.

    :param sx: x coordinate of the segment's start.
    :param sy: y coordinate of the segment's start.
    :param sz: z coordinate of the segment's start.
    :param dx: x component of the segment's direction.
    :param dy: y component of the segment's direction.
    :param dz: z component of the segment's direction.
    :param face: Face whose plane is crossed.
    :param cell: Cell the face belongs to, which sets the face's outward normal.
    :param face_cell_indices: Shape `(n_faces, 2)` owner and neighbour of every face.
    :param face_unit_normals: Shape `(n_faces, 3)` unit normals pointing out of the owner.
    :param face_centroids: Shape `(n_faces, 3)` face centres.
    :returns: The parameter `t` of the crossing (`start + t * direction`), or `nan` if the
        segment is parallel to the face.
    """
    outward = 1.0 if face_cell_indices[face, 0] == cell else -1.0
    nx = outward * face_unit_normals[face, 0]
    ny = outward * face_unit_normals[face, 1]
    nz = outward * face_unit_normals[face, 2]
    denominator = nx * dx + ny * dy + nz * dz
    if abs(denominator) < PARALLEL_TOLERANCE:
        return np.nan
    numerator = (
        nx * (face_centroids[face, 0] - sx)
        + ny * (face_centroids[face, 1] - sy)
        + nz * (face_centroids[face, 2] - sz)
    )
    return numerator / denominator


@numba.njit(cache=True)
def crossing_is_inside_face(
    sx: Number,
    sy: Number,
    sz: Number,
    dx: Number,
    dy: Number,
    dz: Number,
    t: Number,
    face: Integer,
    face_vertex_offsets: IntArray[OneDimension],
    face_vertex_indices: IntArray[OneDimension],
    vertex_coordinates: NumberArray[TwoDimensions],
    face_unit_normals: NumberArray[TwoDimensions],
) -> Boolean:
    """
    Whether the point a segment reaches at `t` lies within a face's polygon.

    :param sx: x coordinate of the segment's start.
    :param sy: y coordinate of the segment's start.
    :param sz: z coordinate of the segment's start.
    :param dx: x component of the segment's direction.
    :param dy: y component of the segment's direction.
    :param dz: z component of the segment's direction.
    :param t: Parameter of the crossing of the face's plane (`start + t * direction`).
    :param face: The face.
    :param face_vertex_offsets: Offsets into `face_vertex_indices` for each face.
    :param face_vertex_indices: The vertices of every face, one face after another.
    :param vertex_coordinates: Shape `(n_vertices, 3)` vertex positions.
    :param face_unit_normals: Shape `(n_faces, 3)` unit normals.
    :returns: `True` if the crossing is inside or on the boundary of the face.
    """
    dominant = dominant_axis_of(
        nx=face_unit_normals[face, 0],
        ny=face_unit_normals[face, 1],
        nz=face_unit_normals[face, 2],
    )
    u_axis = 1 if dominant == 0 else 0
    v_axis = 1 if dominant == 2 else 2
    start = (sx, sy, sz)
    direction = (dx, dy, dz)
    return point_in_indexed_polygon(
        x=start[u_axis] + t * direction[u_axis],
        y=start[v_axis] + t * direction[v_axis],
        u_axis=u_axis,
        v_axis=v_axis,
        vertex_indices=face_vertex_indices[  # type: ignore[arg-type]
            face_vertex_offsets[face] : face_vertex_offsets[face + 1]
        ],
        vertex_coordinates=vertex_coordinates,
        tolerance=1e-9,
    )


@numba.njit(cache=True)
def walk_cells(
    sx: Number,
    sy: Number,
    sz: Number,
    dx: Number,
    dy: Number,
    dz: Number,
    first_cell: Integer,
    max_visits: Integer,
    cell_face_offsets: IntArray[OneDimension],
    cell_face_indices: IntArray[OneDimension],
    face_vertex_offsets: IntArray[OneDimension],
    face_vertex_indices: IntArray[OneDimension],
    vertex_coordinates: NumberArray[TwoDimensions],
    face_cell_indices: IntArray[TwoDimensions],
    face_unit_normals: NumberArray[TwoDimensions],
    face_centroids: NumberArray[TwoDimensions],
) -> tuple[IntArray[OneDimension], NumberArray[OneDimension], NumberArray[OneDimension]]:
    """
    Walk a segment through the grid from `first_cell`, crossing from cell to cell through the face
    it leaves each one by, until it ends or leaves the active grid.

    A face is only checked against its polygon if the segment crosses its plane ahead of the
    current position and no later than the segment's end, since no other crossing can be where the
    segment leaves the cell.

    :param sx: x coordinate of the segment's start.
    :param sy: y coordinate of the segment's start.
    :param sz: z coordinate of the segment's start.
    :param dx: x component of the segment's direction (end minus start).
    :param dy: y component of the segment's direction (end minus start).
    :param dz: z component of the segment's direction (end minus start).
    :param first_cell: Cell containing the segment's start.
    :param max_visits: Most cells the walk may visit before it gives up.
    :param cell_face_offsets: Offsets into `cell_face_indices` for each cell.
    :param cell_face_indices: The faces of every cell, one cell after another.
    :param face_vertex_offsets: Offsets into `face_vertex_indices` for each face.
    :param face_vertex_indices: The vertices of every face, one face after another.
    :param vertex_coordinates: Shape `(n_vertices, 3)` vertex positions.
    :param face_cell_indices: Shape `(n_faces, 2)` owner and neighbour of every face.
    :param face_unit_normals: Shape `(n_faces, 3)` unit normals pointing out of the owner.
    :param face_centroids: Shape `(n_faces, 3)` face centres.
    :returns: The cells visited, in order, with the parameter `t` (`start + t * direction`) at
        which the segment enters and leaves each of them.
    """
    capacity = INITIAL_CELL_CAPACITY
    cells = np.empty(capacity, dtype=np.int64)
    entry_t = np.empty(capacity, dtype=np.float64)
    exit_t = np.empty(capacity, dtype=np.float64)
    count = 0

    current_cell = first_cell
    current_t = 0.0
    for _ in range(max_visits):
        best_t = np.nan
        best_face = -1
        for slot in range(cell_face_offsets[current_cell], cell_face_offsets[current_cell + 1]):
            face = cell_face_indices[slot]
            t = face_plane_crossing(
                sx=sx,
                sy=sy,
                sz=sz,
                dx=dx,
                dy=dy,
                dz=dz,
                face=face,
                cell=current_cell,
                face_cell_indices=face_cell_indices,
                face_unit_normals=face_unit_normals,
                face_centroids=face_centroids,
            )
            if np.isnan(t) or t <= current_t + CROSSING_TOLERANCE or t > 1.0 + CROSSING_TOLERANCE:
                continue
            if best_face >= 0 and t >= best_t:
                continue
            if crossing_is_inside_face(
                sx=sx,
                sy=sy,
                sz=sz,
                dx=dx,
                dy=dy,
                dz=dz,
                t=t,
                face=face,
                face_vertex_offsets=face_vertex_offsets,
                face_vertex_indices=face_vertex_indices,
                vertex_coordinates=vertex_coordinates,
                face_unit_normals=face_unit_normals,
            ):
                best_t = t
                best_face = face

        if count == capacity:
            capacity *= 2
            grown_cells = np.empty(capacity, dtype=np.int64)
            grown_entry = np.empty(capacity, dtype=np.float64)
            grown_exit = np.empty(capacity, dtype=np.float64)
            grown_cells[:count] = cells[:count]
            grown_entry[:count] = entry_t[:count]
            grown_exit[:count] = exit_t[:count]
            cells, entry_t, exit_t = grown_cells, grown_entry, grown_exit

        cells[count] = current_cell
        entry_t[count] = current_t
        if best_face < 0:
            # The segment ends inside this cell without crossing another face.
            exit_t[count] = 1.0
            count += 1
            break

        exit_t[count] = min(best_t, 1.0)
        count += 1
        if best_t >= 1.0 - CROSSING_TOLERANCE:
            break

        owner = face_cell_indices[best_face, 0]
        neighbour = face_cell_indices[best_face, 1]
        next_cell = neighbour if owner == current_cell else owner
        if next_cell < 0:
            break  # a boundary face, so the segment leaves the active grid
        current_cell = next_cell
        current_t = best_t

    return cells[:count].copy(), entry_t[:count].copy(), exit_t[:count].copy()  # type: ignore[return-value]


def traversal_arrays(grid: Grid) -> dict[str, IntArray | NumberArray]:
    """The grid arrays the walk reads."""
    return {
        "cell_face_offsets": grid.cell_face_offsets,
        "cell_face_indices": grid.cell_face_indices,
        "face_vertex_offsets": grid.face_vertex_offsets,
        "face_vertex_indices": grid.face_vertex_indices,
        "vertex_coordinates": grid.vertex_coordinates,
        "face_cell_indices": grid.face_cell_indices,
        "face_unit_normals": grid.face_unit_normals,
        "face_centroids": grid.face_centroids,
    }


def walk_segment_cells(
    grid: Grid,
    start: tuple[Number, Number, Number],
    end: tuple[Number, Number, Number],
    first_cell: Integer,
) -> tuple[IntArray[OneDimension], NumberArray[OneDimension], NumberArray[OneDimension]]:
    """
    Walk the straight segment from `start` to `end` through `grid`, cell by cell.

    The walk crosses from each cell to its neighbour through the face the segment leaves it by,
    until it reaches `end` or leaves the active grid through a boundary face. It stops there
    rather than guessing where the segment re-enters.

    :param grid: Grid to walk.
    :param start: Segment start point.
    :param end: Segment end point.
    :param first_cell: Cell containing `start`.
    :returns: The cells visited, in order, with the fraction of the segment (`0` at `start`, `1`
        at `end`) at which it enters and leaves each of them. Empty if the segment has no length.
    """
    dx, dy, dz = end[0] - start[0], end[1] - start[1], end[2] - start[2]
    if math.sqrt(dx * dx + dy * dy + dz * dz) == 0:
        empty = np.empty(0, dtype=np.float64)
        return np.empty(0, dtype=np.int64), empty, empty.copy()
    return walk_cells(
        sx=start[0],
        sy=start[1],
        sz=start[2],
        dx=dx,
        dy=dy,
        dz=dz,
        first_cell=first_cell,
        max_visits=4 * grid.n_cells + 64,
        **traversal_arrays(grid),  # type: ignore[arg-type]
    )


def get_segment_face_intersection(
    start: tuple[Number, Number, Number],
    direction: tuple[Number, Number, Number],
    grid: Grid,
    face_index: Integer,
    cell_index: Integer,
) -> Number | None:
    """
    Where a ray from `start` along `direction` crosses a face and lands within its polygon.

    :param start: Ray origin.
    :param direction: Ray direction, of any length. `t` is in units of its length, so `t = 1`
        reaches `start + direction`.
    :param grid: Grid providing face geometry.
    :param face_index: Face to test.
    :param cell_index: Cell the face belongs to, which sets the face's outward normal.
    :returns: The parameter `t` (`point = start + t * direction`), or `None` if the ray is parallel
        to the face's plane or crosses it outside the face's polygon.
    """
    arrays = traversal_arrays(grid)
    t = face_plane_crossing(
        sx=start[0],
        sy=start[1],
        sz=start[2],
        dx=direction[0],
        dy=direction[1],
        dz=direction[2],
        face=face_index,
        cell=cell_index,
        face_cell_indices=typing.cast(IntArray[TwoDimensions], arrays["face_cell_indices"]),
        face_unit_normals=arrays["face_unit_normals"],
        face_centroids=arrays["face_centroids"],
    )
    if np.isnan(t):
        return None
    inside = crossing_is_inside_face(
        sx=start[0],
        sy=start[1],
        sz=start[2],
        dx=direction[0],
        dy=direction[1],
        dz=direction[2],
        t=t,
        face=face_index,
        face_vertex_offsets=typing.cast(IntArray[OneDimension], arrays["face_vertex_offsets"]),
        face_vertex_indices=typing.cast(IntArray[OneDimension], arrays["face_vertex_indices"]),
        vertex_coordinates=arrays["vertex_coordinates"],
        face_unit_normals=arrays["face_unit_normals"],
    )
    return t if inside else None
