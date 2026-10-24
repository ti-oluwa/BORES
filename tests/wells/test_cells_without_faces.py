"""Inactive and pinched-out cells have no faces, so they contain no point and no leg starts in one."""

import math
import pathlib
import warnings

import numpy as np
import pytest

from bores.grids.io.grdecl import load_grdecl
from bores.types import UnitSystem
from bores.wells.indices import perforations

DATA = pathlib.Path(__file__).parents[2] / "data"


def load(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_grdecl(DATA / f"{name}.grdecl", unit_system=UnitSystem.FIELD)


@pytest.fixture(scope="module", params=["dome", "Johansen", "cartesian"])
def grid(request):
    return load(request.param)


def search_radius(grid, point):
    nearest = grid.find_nearest_cell(*point)
    return 0.5 * math.sqrt(grid.cell_length_x[nearest] ** 2 + grid.cell_length_y[nearest] ** 2)


def test_a_cell_without_faces_contains_no_point(grid):
    cells = np.flatnonzero(np.diff(grid.cell_face_offsets) == 0)
    assert len(cells) > 0
    for cell in cells[:25]:
        centre = tuple(grid.cell_centroids[cell])
        assert not perforations.check_cell_contains_point(grid, int(cell), centre)
        assert not perforations.check_cell_contains_point(grid, int(cell), (0.0, 0.0, -1e9))


def test_locate_cell_never_returns_a_cell_without_faces(grid):
    rng = np.random.default_rng(5)
    without_faces = np.flatnonzero(np.diff(grid.cell_face_offsets) == 0)
    with_faces = np.flatnonzero(np.diff(grid.cell_face_offsets) > 0)
    points = [tuple(grid.cell_centroids[c]) for c in rng.choice(without_faces, 40)]
    for cell in rng.choice(with_faces, 60):
        face = int(rng.choice(grid.get_cell_face_indices(int(cell))))
        points.append(
            tuple(
                grid.cell_centroids[cell]
                + 0.9 * (grid.face_centroids[face] - grid.cell_centroids[cell])
            )
        )
    for point in points:
        found = perforations.locate_cell(grid, point, search_radius(grid, point))
        assert found is None or len(grid.get_cell_face_indices(int(found))) > 0
        if found is not None:
            assert perforations.check_cell_contains_point(grid, int(found), point)


def test_a_point_inside_an_active_cell_is_not_claimed_by_a_neighbouring_cell_without_faces(grid):
    # Whichever cells the fallback search lists first, the active cell holding the point wins.
    # A few distorted cells next to pinch-outs fail the containment test for their own
    # centroid, so no search could place a point in them; those are left out.
    rng = np.random.default_rng(11)
    with_faces = np.flatnonzero(np.diff(grid.cell_face_offsets) > 0)
    checked = 0
    for cell in rng.choice(with_faces, 300):
        point = tuple(grid.cell_centroids[cell])
        if not perforations.check_cell_contains_point(grid, int(cell), point):
            continue
        checked += 1
        assert perforations.locate_cell(grid, point, search_radius(grid, point)) == cell
    assert checked > 250


def test_a_leg_starting_in_an_inactive_cell_gets_no_connections():
    grid = load("Johansen")
    cell = int(np.flatnonzero(grid.cell_statuses == 0)[10000])
    x, y, z = grid.cell_centroids[cell]
    legs = perforations.walk_segment_through_grid(
        grid, (x, y, z), (x, y, z + 400.0), 0.0, 400.0, search_radius(grid, (x, y, z))
    )
    assert legs == []


def test_legs_never_contain_a_cell_without_faces(grid):
    rng = np.random.default_rng(2)
    with_faces = np.flatnonzero(np.diff(grid.cell_face_offsets) > 0)
    for cell in rng.choice(with_faces, 120):
        face = int(rng.choice(grid.get_cell_face_indices(int(cell))))
        start = grid.cell_centroids[cell] + 0.8 * (
            grid.face_centroids[face] - grid.cell_centroids[cell]
        )
        direction = rng.normal(size=3)
        end = start + direction / np.linalg.norm(direction) * 4.0 * float(
            np.cbrt(grid.cell_volumes[cell])
        )
        legs = perforations.walk_segment_through_grid(
            grid, tuple(start), tuple(end), 0.0, 100.0, search_radius(grid, start)
        )
        previous_exit = 0.0
        for leg_cell, entry_md, exit_md in legs:
            assert len(grid.get_cell_face_indices(int(leg_cell))) > 0
            assert entry_md == pytest.approx(previous_exit) and exit_md >= entry_md
            previous_exit = exit_md
        if legs:
            assert perforations.check_cell_contains_point(grid, int(legs[0][0]), tuple(start))
