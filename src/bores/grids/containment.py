"""Tests for whether a point lies inside a grid cell, using the cell's own face polygons."""

import math
import typing

import numba
import numpy as np

from bores.grids.base import Grid
from bores.types import IntArray, Integer, Number, OneDimension

__all__ = [
    "INSIDE_FRACTION",
    "cell_contains_point",
    "first_cell_containing_point",
]

INSIDE_FRACTION = 0.75
"""
Fraction of a full turn of solid angle above which a point is inside a cell. A point inside a
closed cell sees exactly one full turn and a point outside sees none, so this only has to separate
the two. A point on the cell's surface sees half a turn, which is why it is decided by its distance
to the surface instead.
"""


@numba.njit(cache=True)
def solid_angle_fraction(
    x: float,
    y: float,
    z: float,
    cell: int,
    cell_face_offsets: np.ndarray,
    cell_face_indices: np.ndarray,
    face_vertex_offsets: np.ndarray,
    face_vertex_indices: np.ndarray,
    vertex_coordinates: np.ndarray,
    face_cell_indices: np.ndarray,
    face_unit_normals: np.ndarray,
) -> float:
    """
    Fraction of a full turn of solid angle that the surface of a cell subtends at a point.

    The surface is every face polygon of the cell, split into triangles that fan out from the
    face's first vertex, which is also how the grid computes cell volumes. The result is `1` for a
    point inside a closed cell, `0` for a point outside it, and close to `0.5` on a flat face.
    Because it follows the faces rather than their average planes, it stays correct for cells
    that are warped or not convex.

    :param x: x coordinate of the point.
    :param y: y coordinate of the point.
    :param z: z coordinate of the point.
    :param cell: Index of the cell.
    :param cell_face_offsets: Offsets into `cell_face_indices` for each cell.
    :param cell_face_indices: The faces of every cell, one cell after another.
    :param face_vertex_offsets: Offsets into `face_vertex_indices` for each face.
    :param face_vertex_indices: The vertices of every face, one face after another.
    :param vertex_coordinates: Shape `(n_vertices, 3)` vertex positions.
    :param face_cell_indices: Shape `(n_faces, 2)` owner and neighbour of every face.
    :param face_unit_normals: Shape `(n_faces, 3)` unit normals pointing out of the owner.
    :returns: The fraction of a full turn.
    """
    total = 0.0
    for slot in range(cell_face_offsets[cell], cell_face_offsets[cell + 1]):
        face = cell_face_indices[slot]
        start = face_vertex_offsets[face]
        count = face_vertex_offsets[face + 1] - start
        if count < 3:
            continue

        # The face's winding is made to agree with the normal pointing out of this cell.
        outward = 1.0 if face_cell_indices[face, 0] == cell else -1.0
        area_x = 0.0
        area_y = 0.0
        area_z = 0.0
        for position in range(count):
            first = face_vertex_indices[start + position]
            second = face_vertex_indices[start + (position + 1) % count]
            bx = vertex_coordinates[first, 0] - x
            by = vertex_coordinates[first, 1] - y
            bz = vertex_coordinates[first, 2] - z
            cx = vertex_coordinates[second, 0] - x
            cy = vertex_coordinates[second, 1] - y
            cz = vertex_coordinates[second, 2] - z
            area_x += by * cz - bz * cy
            area_y += bz * cx - bx * cz
            area_z += bx * cy - by * cx
        alignment = (
            area_x * face_unit_normals[face, 0]
            + area_y * face_unit_normals[face, 1]
            + area_z * face_unit_normals[face, 2]
        ) * outward
        orientation = 1.0 if alignment >= 0.0 else -1.0

        apex = face_vertex_indices[start]
        ax = vertex_coordinates[apex, 0] - x
        ay = vertex_coordinates[apex, 1] - y
        az = vertex_coordinates[apex, 2] - z
        length_a = math.sqrt(ax * ax + ay * ay + az * az)
        for position in range(1, count - 1):
            second = face_vertex_indices[start + position]
            third = face_vertex_indices[start + position + 1]
            bx = vertex_coordinates[second, 0] - x
            by = vertex_coordinates[second, 1] - y
            bz = vertex_coordinates[second, 2] - z
            cx = vertex_coordinates[third, 0] - x
            cy = vertex_coordinates[third, 1] - y
            cz = vertex_coordinates[third, 2] - z
            length_b = math.sqrt(bx * bx + by * by + bz * bz)
            length_c = math.sqrt(cx * cx + cy * cy + cz * cz)
            triple_product = (
                ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) + az * (bx * cy - by * cx)
            )
            denominator = (
                length_a * length_b * length_c
                + (ax * bx + ay * by + az * bz) * length_c
                + (ax * cx + ay * cy + az * cz) * length_b
                + (bx * cx + by * cy + bz * cz) * length_a
            )
            total += orientation * 2.0 * math.atan2(triple_product, denominator)
    return total / (4.0 * math.pi)


@numba.njit(cache=True)
def triangle_distance_squared(
    px: float,
    py: float,
    pz: float,
    ax: float,
    ay: float,
    az: float,
    bx: float,
    by: float,
    bz: float,
    cx: float,
    cy: float,
    cz: float,
) -> float:
    """
    Squared distance from a point to a triangle.

    :param px: x coordinate of the point.
    :param py: y coordinate of the point.
    :param pz: z coordinate of the point.
    :param ax: x coordinate of the first corner.
    :param ay: y coordinate of the first corner.
    :param az: z coordinate of the first corner.
    :param bx: x coordinate of the second corner.
    :param by: y coordinate of the second corner.
    :param bz: z coordinate of the second corner.
    :param cx: x coordinate of the third corner.
    :param cy: y coordinate of the third corner.
    :param cz: z coordinate of the third corner.
    :returns: The squared distance from the point to the nearest point of the triangle.
    """
    abx = bx - ax
    aby = by - ay
    abz = bz - az
    acx = cx - ax
    acy = cy - ay
    acz = cz - az
    apx = px - ax
    apy = py - ay
    apz = pz - az
    d1 = abx * apx + aby * apy + abz * apz
    d2 = acx * apx + acy * apy + acz * apz
    if d1 <= 0.0 and d2 <= 0.0:
        return apx * apx + apy * apy + apz * apz

    bpx = px - bx
    bpy = py - by
    bpz = pz - bz
    d3 = abx * bpx + aby * bpy + abz * bpz
    d4 = acx * bpx + acy * bpy + acz * bpz
    if d3 >= 0.0 and d4 <= d3:
        return bpx * bpx + bpy * bpy + bpz * bpz

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        gap = d1 - d3
        t = d1 / gap if abs(gap) > 0.0 else 0.0
        qx = px - (ax + t * abx)
        qy = py - (ay + t * aby)
        qz = pz - (az + t * abz)
        return qx * qx + qy * qy + qz * qz

    cpx = px - cx
    cpy = py - cy
    cpz = pz - cz
    d5 = abx * cpx + aby * cpy + abz * cpz
    d6 = acx * cpx + acy * cpy + acz * cpz
    if d6 >= 0.0 and d5 <= d6:
        return cpx * cpx + cpy * cpy + cpz * cpz

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        gap = d2 - d6
        t = d2 / gap if abs(gap) > 0.0 else 0.0
        qx = px - (ax + t * acx)
        qy = py - (ay + t * acy)
        qz = pz - (az + t * acz)
        return qx * qx + qy * qy + qz * qz

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        gap = (d4 - d3) + (d5 - d6)
        t = (d4 - d3) / gap if abs(gap) > 0.0 else 0.0
        qx = px - (bx + t * (cx - bx))
        qy = py - (by + t * (cy - by))
        qz = pz - (bz + t * (cz - bz))
        return qx * qx + qy * qy + qz * qz

    total = va + vb + vc
    if not abs(total) > 0.0:
        # A triangle with no area: its nearest point is one of its corners.
        return min(
            apx * apx + apy * apy + apz * apz,
            min(
                bpx * bpx + bpy * bpy + bpz * bpz,
                cpx * cpx + cpy * cpy + cpz * cpz,
            ),
        )
    v = vb / total
    w = vc / total
    qx = px - (ax + abx * v + acx * w)
    qy = py - (ay + aby * v + acy * w)
    qz = pz - (az + abz * v + acz * w)
    return qx * qx + qy * qy + qz * qz


@numba.njit(cache=True)
def is_within_of_surface(
    x: float,
    y: float,
    z: float,
    cell: int,
    distance: float,
    cell_face_offsets: np.ndarray,
    cell_face_indices: np.ndarray,
    face_vertex_offsets: np.ndarray,
    face_vertex_indices: np.ndarray,
    vertex_coordinates: np.ndarray,
) -> bool:
    """
    Whether a point is no further than `distance` from the surface of a cell.

    The surface is the same fan of triangles that `solid_angle_fraction` uses.

    :param x: x coordinate of the point.
    :param y: y coordinate of the point.
    :param z: z coordinate of the point.
    :param cell: Index of the cell.
    :param distance: Greatest distance from the surface that still counts.
    :param cell_face_offsets: Offsets into `cell_face_indices` for each cell.
    :param cell_face_indices: The faces of every cell, one cell after another.
    :param face_vertex_offsets: Offsets into `face_vertex_indices` for each face.
    :param face_vertex_indices: The vertices of every face, one face after another.
    :param vertex_coordinates: Shape `(n_vertices, 3)` vertex positions.
    :returns: `True` if some part of the surface is within `distance` of the point.
    """
    limit = distance * distance
    for slot in range(cell_face_offsets[cell], cell_face_offsets[cell + 1]):
        face = cell_face_indices[slot]
        start = face_vertex_offsets[face]
        count = face_vertex_offsets[face + 1] - start
        if count < 3:
            continue
        apex = face_vertex_indices[start]
        for position in range(1, count - 1):
            second = face_vertex_indices[start + position]
            third = face_vertex_indices[start + position + 1]
            if (
                triangle_distance_squared(
                    px=x,
                    py=y,
                    pz=z,
                    ax=vertex_coordinates[apex, 0],
                    ay=vertex_coordinates[apex, 1],
                    az=vertex_coordinates[apex, 2],
                    bx=vertex_coordinates[second, 0],
                    by=vertex_coordinates[second, 1],
                    bz=vertex_coordinates[second, 2],
                    cx=vertex_coordinates[third, 0],
                    cy=vertex_coordinates[third, 1],
                    cz=vertex_coordinates[third, 2],
                )
                <= limit
            ):
                return True
    return False


@numba.njit(cache=True)
def contains_point(
    x: float,
    y: float,
    z: float,
    cell: int,
    tolerance: float,
    cell_face_offsets: np.ndarray,
    cell_face_indices: np.ndarray,
    face_vertex_offsets: np.ndarray,
    face_vertex_indices: np.ndarray,
    vertex_coordinates: np.ndarray,
    face_cell_indices: np.ndarray,
    face_unit_normals: np.ndarray,
    cell_min_xyz: np.ndarray,
    cell_max_xyz: np.ndarray,
) -> bool:
    """
    Whether a cell contains a point.

    A cell with no faces (an inactive or pinched-out cell) contains nothing. Otherwise the point
    is inside if the cell's faces surround it, or if it lies within `tolerance` of the surface.

    :param x: x coordinate of the point.
    :param y: y coordinate of the point.
    :param z: z coordinate of the point.
    :param cell: Index of the cell.
    :param tolerance: Distance from the surface up to which a point still counts as inside.
    :param cell_face_offsets: Offsets into `cell_face_indices` for each cell.
    :param cell_face_indices: The faces of every cell, one cell after another.
    :param face_vertex_offsets: Offsets into `face_vertex_indices` for each face.
    :param face_vertex_indices: The vertices of every face, one face after another.
    :param vertex_coordinates: Shape `(n_vertices, 3)` vertex positions.
    :param face_cell_indices: Shape `(n_faces, 2)` owner and neighbour of every face.
    :param face_unit_normals: Shape `(n_faces, 3)` unit normals pointing out of the owner.
    :param cell_min_xyz: Shape `(n_cells, 3)` lower corner of each cell's bounding box.
    :param cell_max_xyz: Shape `(n_cells, 3)` upper corner of each cell's bounding box.
    :returns: `True` if the point is inside the cell.
    """
    if cell_face_offsets[cell + 1] == cell_face_offsets[cell]:
        return False

    # A point further than the tolerance from the cell's bounding box cannot be inside it.
    if (
        x < cell_min_xyz[cell, 0] - tolerance
        or x > cell_max_xyz[cell, 0] + tolerance
        or y < cell_min_xyz[cell, 1] - tolerance
        or y > cell_max_xyz[cell, 1] + tolerance
        or z < cell_min_xyz[cell, 2] - tolerance
        or z > cell_max_xyz[cell, 2] + tolerance
    ):
        return False

    fraction = solid_angle_fraction(
        x=x,
        y=y,
        z=z,
        cell=cell,
        cell_face_offsets=cell_face_offsets,
        cell_face_indices=cell_face_indices,
        face_vertex_offsets=face_vertex_offsets,
        face_vertex_indices=face_vertex_indices,
        vertex_coordinates=vertex_coordinates,
        face_cell_indices=face_cell_indices,
        face_unit_normals=face_unit_normals,
    )
    if fraction >= INSIDE_FRACTION:
        return True
    return is_within_of_surface(
        x=x,
        y=y,
        z=z,
        cell=cell,
        distance=tolerance,
        cell_face_offsets=cell_face_offsets,
        cell_face_indices=cell_face_indices,
        face_vertex_offsets=face_vertex_offsets,
        face_vertex_indices=face_vertex_indices,
        vertex_coordinates=vertex_coordinates,
    )


@numba.njit(cache=True)
def first_containing_cell(
    x: float,
    y: float,
    z: float,
    candidates: np.ndarray,
    tolerance: float,
    cell_face_offsets: np.ndarray,
    cell_face_indices: np.ndarray,
    face_vertex_offsets: np.ndarray,
    face_vertex_indices: np.ndarray,
    vertex_coordinates: np.ndarray,
    face_cell_indices: np.ndarray,
    face_unit_normals: np.ndarray,
    cell_min_xyz: np.ndarray,
    cell_max_xyz: np.ndarray,
) -> int:
    """
    Position, in `candidates`, of the first cell that contains the point.

    :param x: x coordinate of the point.
    :param y: y coordinate of the point.
    :param z: z coordinate of the point.
    :param candidates: Cell indices to test, in priority order.
    :param tolerance: Distance from the surface up to which a point still counts as inside.
    :param cell_face_offsets: Offsets into `cell_face_indices` for each cell.
    :param cell_face_indices: The faces of every cell, one cell after another.
    :param face_vertex_offsets: Offsets into `face_vertex_indices` for each face.
    :param face_vertex_indices: The vertices of every face, one face after another.
    :param vertex_coordinates: Shape `(n_vertices, 3)` vertex positions.
    :param face_cell_indices: Shape `(n_faces, 2)` owner and neighbour of every face.
    :param face_unit_normals: Shape `(n_faces, 3)` unit normals pointing out of the owner.
    :param cell_min_xyz: Shape `(n_cells, 3)` lower corner of each cell's bounding box.
    :param cell_max_xyz: Shape `(n_cells, 3)` upper corner of each cell's bounding box.
    :returns: The position of the first containing candidate, or `-1` if there is none.
    """
    for position in range(candidates.shape[0]):
        if contains_point(
            x=x,
            y=y,
            z=z,
            cell=candidates[position],
            tolerance=tolerance,
            cell_face_offsets=cell_face_offsets,
            cell_face_indices=cell_face_indices,
            face_vertex_offsets=face_vertex_offsets,
            face_vertex_indices=face_vertex_indices,
            vertex_coordinates=vertex_coordinates,
            face_cell_indices=face_cell_indices,
            face_unit_normals=face_unit_normals,
            cell_min_xyz=cell_min_xyz,
            cell_max_xyz=cell_max_xyz,
        ):
            return position
    return -1


def geometry_arrays(grid: Grid) -> dict[str, np.ndarray]:
    """The grid arrays the containment kernels read."""
    return {
        "cell_face_offsets": grid.cell_face_offsets,
        "cell_face_indices": grid.cell_face_indices,
        "face_vertex_offsets": grid.face_vertex_offsets,
        "face_vertex_indices": grid.face_vertex_indices,
        "vertex_coordinates": grid.vertex_coordinates,
        "face_cell_indices": grid.face_cell_indices,
        "face_unit_normals": grid.face_unit_normals,
        "cell_min_xyz": grid.cell_min_xyz,
        "cell_max_xyz": grid.cell_max_xyz,
    }


def cell_contains_point(
    grid: Grid,
    cell_index: Integer,
    point: tuple[Number, Number, Number],
    tolerance: Number = 1e-6,
) -> bool:
    """
    Whether a cell contains a point, judged from the cell's own faces.

    This follows warped and non-convex cells correctly, and a cell with no faces (an inactive
    or pinched-out cell) contains nothing.

    :param grid: Grid providing the cell geometry.
    :param cell_index: Cell to test.
    :param point: Point to test.
    :param tolerance: Distance from the cell's surface up to which a point still counts as inside.
    :returns: `True` if the point is inside the cell or on its boundary.
    """
    return bool(
        contains_point(
            x=float(point[0]),
            y=float(point[1]),
            z=float(point[2]),
            cell=int(cell_index),
            tolerance=float(tolerance),
            **geometry_arrays(grid),
        )
    )


def first_cell_containing_point(
    grid: Grid,
    candidates: IntArray[OneDimension],
    point: tuple[Number, Number, Number],
    tolerance: Number = 1e-6,
) -> typing.Any:
    """
    The first of `candidates` that contains a point.

    :param grid: Grid providing the cell geometry.
    :param candidates: Cell indices to test, in priority order.
    :param point: Point to locate.
    :param tolerance: Distance from a cell's surface up to which a point still counts as inside.
    :returns: The first containing cell, or `None` if none of them contains the point.
    """
    position = first_containing_cell(
        x=float(point[0]),
        y=float(point[1]),
        z=float(point[2]),
        candidates=np.ascontiguousarray(candidates),
        tolerance=float(tolerance),
        **geometry_arrays(grid),
    )
    return candidates[position] if position >= 0 else None
