"""`find_first_containing_cell` and `locate_cell` on warped and faulted grids."""

import math
import pathlib
import warnings

import numpy as np
import pytest

from bores.grids.io.grdecl import load_grdecl
from bores.types import UnitSystem
from bores.wells.indices import perforations

DATA = pathlib.Path(__file__).parents[2] / "data"
GRIDS = ["dome", "cp", "40X48X13_faults", "Simple10x10x3_FlatCross_Fault"]


@pytest.fixture(scope="module", params=GRIDS)
def grid(request):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_grdecl(DATA / f"{request.param}.grdecl", unit_system=UnitSystem.FIELD)


def points_near_cell_faces(grid, count, seed=0):
    """Points just inside cells that do contain them, with the cell that holds each one."""
    rng = np.random.default_rng(seed)
    with_faces = np.flatnonzero((grid.cell_statuses != 0) & (np.diff(grid.cell_face_offsets) > 0))
    found = []
    for cell in rng.choice(with_faces, count * 2):
        face = int(rng.choice(grid.get_cell_face_indices(int(cell))))
        point = tuple(
            grid.cell_centroids[cell]
            + 0.95 * (grid.face_centroids[face] - grid.cell_centroids[cell])
        )
        if perforations.check_cell_contains_point(grid, int(cell), point):
            found.append((point, int(cell)))
    return found[:count]


def search_radius(grid, point):
    nearest = grid.find_nearest_cell(*point)
    return 0.5 * math.sqrt(grid.cell_length_x[nearest] ** 2 + grid.cell_length_y[nearest] ** 2)


def one_at_a_time(grid, candidates, point):
    for candidate in candidates:
        if perforations.check_cell_contains_point(grid, candidate, point):
            return candidate
    return None


def test_first_containing_cell_matches_testing_one_at_a_time(grid):
    for point, _ in points_near_cell_faces(grid, 60):
        radius = search_radius(grid, point)
        for scale in (1.0, 4.0, 16.0):
            candidates = grid.find_cells_in_radius(*point, radius * scale)
            assert perforations.find_first_containing_cell(
                grid, candidates, point
            ) == one_at_a_time(grid, candidates, point)


def test_the_cell_holding_a_point_is_among_the_candidates_it_is_found_in(grid):
    for point, _cell in points_near_cell_faces(grid, 60, seed=1):
        candidates = grid.find_cells_in_radius(*point, 4.0 * search_radius(grid, point))
        found = perforations.find_first_containing_cell(grid, candidates, point)
        assert found is not None
        assert perforations.check_cell_contains_point(grid, int(found), point)


def test_a_point_in_none_of_the_candidates_gives_none(grid):
    centre = grid.cell_centroids[0]
    candidates = grid.find_cells_in_radius(*centre, 1e4)
    far = (centre[0], centre[1], centre[2] - 1e6)
    assert perforations.find_first_containing_cell(grid, candidates, far) is None
    assert perforations.find_first_containing_cell(grid, candidates[:0], tuple(centre)) is None


def test_a_cell_without_faces_is_never_returned():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid = load_grdecl(DATA / "dome.grdecl", unit_system=UnitSystem.FIELD)
    without_faces = np.flatnonzero(np.diff(grid.cell_face_offsets) == 0)
    with_faces = np.flatnonzero(np.diff(grid.cell_face_offsets) > 0)[:40]
    candidates = np.concatenate([without_faces[:1], with_faces]).astype(np.int32)
    assert perforations.find_first_containing_cell(grid, candidates, (0.0, 0.0, -1e9)) is None


def test_candidate_order_decides_between_two_containing_cells(grid):
    # A point on a shared face lies in both neighbours; the first listed one wins either way.
    interior = np.flatnonzero((grid.face_cell_indices >= 0).all(axis=1))
    face = int(interior[len(interior) // 2])
    first, second = (int(cell) for cell in grid.face_cell_indices[face])
    point = tuple(grid.face_centroids[face])
    for order in ([first, second], [second, first]):
        candidates = np.array(order, dtype=np.int32)
        found = perforations.find_first_containing_cell(grid, candidates, point)
        assert found in order
        if perforations.check_cell_contains_point(grid, order[0], point):
            assert found == order[0]


def test_locate_cell_finds_a_cell_that_really_contains_the_point(grid):
    for point, _ in points_near_cell_faces(grid, 80, seed=3):
        found = perforations.locate_cell(grid, point, search_radius(grid, point))
        assert found is not None
        assert perforations.check_cell_contains_point(grid, int(found), point)
