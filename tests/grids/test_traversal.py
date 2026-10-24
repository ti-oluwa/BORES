"""The compiled segment walk follows the same path as a plain cell-by-cell reference."""

import math
import pathlib
import warnings

import numpy as np
import pytest

from bores.grids.factories.cartesian import make_cartesian_grid
from bores.grids.io.grdecl import load_grdecl
from bores.grids.traversal import (
    check_point_in_polygon_2d,
    check_point_in_polygon_3d,
    check_point_on_segment,
    get_segment_face_intersection,
    walk_segment_cells,
)
from bores.types import UnitSystem
from bores.wells.indices import perforations

DATA = pathlib.Path(__file__).parents[2] / "data"


def load(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_grdecl(DATA / f"{name}.grdecl", unit_system=UnitSystem.FIELD)


def reference_point_on_segment(px, py, x1, y1, x2, y2, tolerance):
    length = math.hypot(x2 - x1, y2 - y1)
    if length < tolerance:
        return math.hypot(px - x1, py - y1) < tolerance
    cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    if abs(cross) / length > tolerance:
        return False
    dot = (px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)
    return -tolerance <= dot <= (length**2 + tolerance)


def reference_point_in_polygon(x, y, polygon, tolerance=1e-9):
    inside = False
    previous = len(polygon) - 1
    for current in range(len(polygon)):
        xi, yi = polygon[current]
        xj, yj = polygon[previous]
        if reference_point_on_segment(x, y, xi, yi, xj, yj, tolerance):
            return True
        if (yi > y) != (yj > y) and x < xi + (y - yi) * (xj - xi) / (yj - yi):
            inside = not inside
        previous = current
    return inside


def reference_walk(grid, first_cell, start, end):
    """The walk written out in plain Python, one face at a time."""
    direction = tuple(e - s for s, e in zip(start, end, strict=True))
    cells, entries, exits = [], [], []
    current_cell, current_t = first_cell, 0.0
    for _ in range(4 * grid.n_cells + 64):
        best_t, best_face = None, None
        for face in grid.get_cell_face_indices(current_cell):
            t = get_segment_face_intersection(start, direction, grid, int(face), int(current_cell))
            if t is None or t <= current_t + 1e-9 or t > 1.0 + 1e-9:
                continue
            if best_t is None or t < best_t:
                best_t, best_face = t, int(face)
        cells.append(int(current_cell))
        entries.append(current_t)
        if best_face is None:
            exits.append(1.0)
            break
        exits.append(min(best_t, 1.0))
        if best_t >= 1.0 - 1e-9:
            break
        owner, neighbour = grid.face_cell_indices[best_face]
        next_cell = int(neighbour if owner == current_cell else owner)
        if next_cell < 0:
            break
        current_cell, current_t = next_cell, best_t
    return cells, entries, exits


def test_point_on_segment_matches_the_plain_formula():
    rng = np.random.default_rng(0)
    for _ in range(500):
        x1, y1, x2, y2 = rng.normal(size=4) * 5.0
        fraction = rng.uniform(-0.2, 1.2)
        px = x1 + fraction * (x2 - x1) + rng.choice([0.0, 1e-12, 1e-6, 0.3])
        py = y1 + fraction * (y2 - y1)
        for tolerance in (1e-9, 1e-3):
            assert check_point_on_segment(px, py, x1, y1, x2, y2, tolerance) == (
                reference_point_on_segment(px, py, x1, y1, x2, y2, tolerance)
            )
    assert check_point_on_segment(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1e-9)  # a segment of no length


def test_point_in_polygon_matches_the_plain_ray_cast():
    rng = np.random.default_rng(1)
    polygons = [
        np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]]),
        np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 1.0], [4.0, 3.0], [0.0, 3.0]]),  # not convex
        np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]]),
    ]
    for polygon in polygons:
        points = list(rng.uniform(-1.0, 6.0, size=(300, 2)))
        points += [vertex for vertex in polygon] + [
            (a + b) / 2 for a, b in zip(polygon, np.roll(polygon, 1, axis=0), strict=True)
        ]
        for x, y in points:
            assert check_point_in_polygon_2d(x, y, polygon) == reference_point_in_polygon(
                x, y, polygon
            )


def test_point_in_a_tilted_polygon_uses_the_least_aligned_axes():
    # A steeply tilted square: dropping the wrong axis would collapse it to a line.
    corners = np.array([[0.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.2, 4.0, 4.0], [0.2, 0.0, 4.0]])
    normal = np.cross(corners[1] - corners[0], corners[3] - corners[0])
    normal /= np.linalg.norm(normal)
    assert check_point_in_polygon_3d((0.1, 2.0, 2.0), corners, normal)
    assert not check_point_in_polygon_3d((0.1, 5.0, 2.0), corners, normal)
    assert not check_point_in_polygon_3d((0.1, 2.0, 5.0), corners, normal)


def test_a_segment_through_a_box_visits_the_cells_it_passes_through():
    grid = make_cartesian_grid(nx=4, ny=3, nz=2, dx=10.0, dy=10.0, dz=10.0)
    first = int(grid.flat_index(0, 0, 0))
    start = tuple(grid.cell_centroids[first])
    end = (start[0] + 30.0, start[1], start[2])
    cells, entry, exit_ = walk_segment_cells(grid, start, end, first)
    assert cells.tolist() == [int(grid.flat_index(i, 0, 0)) for i in range(4)]
    assert entry.tolist() == pytest.approx([0.0, 1 / 6, 0.5, 5 / 6][: len(cells)])
    assert exit_[:-1].tolist() == pytest.approx(entry[1:].tolist())
    assert exit_[-1] == pytest.approx(1.0)  # the segment ends inside the last cell


def test_a_segment_leaving_the_grid_stops_at_the_boundary():
    grid = make_cartesian_grid(nx=3, ny=1, nz=1, dx=10.0, dy=10.0, dz=10.0)
    start = tuple(grid.cell_centroids[2])
    cells, _, exit_ = walk_segment_cells(grid, start, (start[0] + 100.0, start[1], start[2]), 2)
    assert cells.tolist() == [2]
    assert exit_[0] == pytest.approx(5.0 / 100.0)


def test_a_segment_with_no_length_visits_nothing():
    grid = make_cartesian_grid(nx=2, ny=2, nz=2, dx=1.0, dy=1.0, dz=1.0)
    point = tuple(grid.cell_centroids[0])
    cells, entry, exit_ = walk_segment_cells(grid, point, point, 0)
    assert len(cells) == len(entry) == len(exit_) == 0


@pytest.mark.parametrize("name", ["dome", "snarkgrid", "40X48X13_faults", "Johansen", "cp"])
def test_the_walk_matches_the_reference_on_warped_grids(name):
    grid = load(name)
    rng = np.random.default_rng(7)
    with_faces = np.flatnonzero(np.diff(grid.cell_face_offsets) > 0)
    compared = 0
    for cell in rng.choice(with_faces, min(60, len(with_faces)), replace=False):
        face = int(rng.choice(grid.get_cell_face_indices(int(cell))))
        start = grid.cell_centroids[cell] + rng.uniform(0.0, 0.9) * (
            grid.face_centroids[face] - grid.cell_centroids[cell]
        )
        scale = float(np.cbrt(grid.cell_volumes[cell]))
        direction = rng.normal(size=3)
        ends = [start + direction / np.linalg.norm(direction) * rng.uniform(0.5, 20.0) * scale]
        ends += [start + np.eye(3)[axis] * rng.uniform(1.0, 10.0) * scale for axis in range(3)]
        ends.append(grid.face_centroids[face])  # ends exactly on a face
        for end in ends:
            first = perforations.locate_cell(grid, tuple(start), scale)
            if first is None:
                continue
            expected = reference_walk(grid, int(first), tuple(start), tuple(end))
            cells, entry, exit_ = walk_segment_cells(grid, tuple(start), tuple(end), int(first))
            assert cells.tolist() == expected[0]
            assert entry.tolist() == expected[1]
            assert exit_.tolist() == expected[2]
            compared += 1
    assert compared > 100


def test_a_long_walk_grows_past_the_initial_room_for_cells():
    grid = make_cartesian_grid(nx=200, ny=1, nz=1, dx=1.0, dy=1.0, dz=1.0)
    start = tuple(grid.cell_centroids[0])
    cells, entry, exit_ = walk_segment_cells(
        grid, start, (start[0] + 199.0, start[1], start[2]), 0
    )
    assert cells.tolist() == list(range(200))
    assert np.all(np.diff(entry) > 0) and exit_[-1] == pytest.approx(1.0)


def test_the_intersection_helper_reports_a_crossing_or_none():
    grid = make_cartesian_grid(nx=2, ny=1, nz=1, dx=10.0, dy=10.0, dz=10.0)
    start = tuple(grid.cell_centroids[0])
    faces = [int(face) for face in grid.get_cell_face_indices(0)]
    direction = (10.0, 0.0, 0.0)
    crossings = [get_segment_face_intersection(start, direction, grid, face, 0) for face in faces]
    assert sorted(t for t in crossings if t is not None and t > 0) == pytest.approx([0.5])
    # a direction parallel to every face but one finds nothing on the others
    assert sum(t is None for t in crossings) == len(faces) - 2
