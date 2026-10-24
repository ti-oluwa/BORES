"""`find_first_containing_cell` and `locate_cell` give one answer however many cells are tested."""

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
    rng = np.random.default_rng(seed)
    with_faces = np.flatnonzero((grid.cell_statuses != 0) & (np.diff(grid.cell_face_offsets) > 0))
    points = []
    for cell in rng.choice(with_faces, count):
        face = int(rng.choice(grid.get_cell_face_indices(int(cell))))
        points.append(
            tuple(
                grid.cell_centroids[cell]
                + 0.95 * (grid.face_centroids[face] - grid.cell_centroids[cell])
            )
        )
    return points


def one_at_a_time(grid, candidates, point):
    for candidate in candidates:
        if perforations.check_cell_contains_point(grid, candidate, point):
            return candidate
    return None


@pytest.mark.parametrize("threshold", [0, 10**9], ids=["batched", "one-at-a-time"])
def test_first_containing_cell_matches_testing_one_at_a_time(grid, monkeypatch, threshold):
    monkeypatch.setattr(perforations, "BATCHED_CONTAINMENT_MIN_CANDIDATES", threshold)
    for point in points_near_cell_faces(grid, 60):
        nearest = grid.find_nearest_cell(*point)
        radius = 0.5 * math.sqrt(
            grid.cell_length_x[nearest] ** 2 + grid.cell_length_y[nearest] ** 2
        )
        for scale in (1.0, 4.0, 16.0):
            candidates = grid.find_cells_in_radius(*point, radius * scale)
            assert perforations.find_first_containing_cell(
                grid, candidates, point
            ) == one_at_a_time(grid, candidates, point)


@pytest.mark.parametrize("threshold", [0, 10**9], ids=["batched", "one-at-a-time"])
def test_a_point_in_none_of_the_candidates_gives_none(grid, monkeypatch, threshold):
    monkeypatch.setattr(perforations, "BATCHED_CONTAINMENT_MIN_CANDIDATES", threshold)
    centre = grid.cell_centroids[0]
    candidates = grid.find_cells_in_radius(*centre, 1e4)
    candidates = candidates[np.diff(grid.cell_face_offsets)[candidates] > 0]
    far = (centre[0], centre[1], centre[2] - 1e6)
    assert perforations.find_first_containing_cell(grid, candidates, far) is None


@pytest.mark.parametrize("threshold", [0, 10**9], ids=["batched", "one-at-a-time"])
def test_a_cell_without_faces_contains_nothing_in_either_path(monkeypatch, threshold):
    monkeypatch.setattr(perforations, "BATCHED_CONTAINMENT_MIN_CANDIDATES", threshold)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid = load_grdecl(DATA / "dome.grdecl", unit_system=UnitSystem.FIELD)
    without_faces = np.flatnonzero(np.diff(grid.cell_face_offsets) == 0)
    with_faces = np.flatnonzero(np.diff(grid.cell_face_offsets) > 0)[:40]
    candidates = np.concatenate([with_faces, without_faces[:1]]).astype(np.int32)
    far = (0.0, 0.0, -1e9)
    assert perforations.find_first_containing_cell(grid, candidates, far) is None
    assert one_at_a_time(grid, candidates, far) is None


def test_candidate_order_decides_between_two_containing_cells(grid, monkeypatch):
    # A point on a shared face lies in both neighbours; the first listed one wins either way.
    face = int(grid.interior_face_indices[0]) if hasattr(grid, "interior_face_indices") else None
    if face is None:
        pytest.skip("grid exposes no interior face list")
    first, second = (int(c) for c in grid.get_face_cell_indices(face)[:2])
    point = tuple(grid.face_centroids[face])
    for threshold in (0, 10**9):
        monkeypatch.setattr(perforations, "BATCHED_CONTAINMENT_MIN_CANDIDATES", threshold)
        for order in ([first, second], [second, first]):
            candidates = np.array(order, dtype=np.int32)
            assert perforations.find_first_containing_cell(grid, candidates, point) == order[0]


def test_locate_cell_is_the_same_with_and_without_batching(grid, monkeypatch):
    points = points_near_cell_faces(grid, 80, seed=3)
    answers = {}
    for threshold in (0, 10**9):
        monkeypatch.setattr(perforations, "BATCHED_CONTAINMENT_MIN_CANDIDATES", threshold)
        answers[threshold] = []
        for point in points:
            nearest = grid.find_nearest_cell(*point)
            radius = 0.5 * math.sqrt(
                grid.cell_length_x[nearest] ** 2 + grid.cell_length_y[nearest] ** 2
            )
            answers[threshold].append(perforations.locate_cell(grid, point, radius))
    assert answers[0] == answers[10**9]
    assert all(cell is not None for cell in answers[0])
