import warnings

import numpy as np
import pytest

from bores.grids.factories.base import FaultRecord
from bores.grids.factories.corner_point import (
    clip_convex_polygon,
    make_corner_point_grid,
    polygon_signed_area,
)

warnings.simplefilter("ignore")


def build_grid(top, thickness, widths_x, widths_y, **kwargs):
    nz, ny, nx = top.shape
    x_edges = np.r_[0.0, np.cumsum(widths_x)]
    y_edges = np.r_[0.0, np.cumsum(widths_y)]
    coord = np.zeros((ny + 1, nx + 1, 6))
    for j in range(ny + 1):
        for i in range(nx + 1):
            coord[j, i] = [x_edges[i], y_edges[j], -100, x_edges[i], y_edges[j], 200]
    zcorn = np.zeros((2 * nz, 2 * ny, 2 * nx))
    for k in range(nz):
        for dj in (0, 1):
            for di in (0, 1):
                zcorn[2 * k, dj::2, di::2] = top[k]
                zcorn[2 * k + 1, dj::2, di::2] = top[k] + thickness[k]
    return make_corner_point_grid(
        coord=coord, zcorn=zcorn, metadata={"dimensions": (nx, ny, nz)}, **kwargs
    )


def lateral_connection_areas(grid):
    areas = {}
    for face in range(grid.n_faces):
        owner, neighbour = (int(c) for c in grid.face_cell_indices[face])
        if owner < 0 or neighbour < 0:
            continue
        normal = grid.face_unit_normals[face]
        if max(abs(normal[0]), abs(normal[1])) > 0.9:
            key = tuple(sorted((owner, neighbour)))
            areas[key] = areas.get(key, 0.0) + float(grid.face_areas[face])
    return areas


def test_polygon_clipping_returns_the_intersection():
    square = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
    shifted = [(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)]
    assert polygon_signed_area(clip_convex_polygon(square, shifted)) == pytest.approx(1.0)
    far = [(5.0, 5.0), (6.0, 5.0), (6.0, 6.0), (5.0, 6.0)]
    assert clip_convex_polygon(square, far) == []


def test_offset_fault_connects_cells_across_the_fault_with_partial_areas():
    top = np.zeros((2, 1, 2))
    top[0, 0] = [0.0, 5.0]
    top[1, 0] = [10.0, 15.0]
    grid = build_grid(top, np.full((2, 1, 2), 10.0), [10.0, 10.0], [10.0])
    areas = lateral_connection_areas(grid)
    assert set(areas) == {(0, 1), (1, 2), (2, 3)}
    assert all(area == pytest.approx(50.0) for area in areas.values())
    assert grid.cell_volumes.sum() == pytest.approx(4000.0)
    for face in range(grid.n_faces):
        owner = grid.face_cell_indices[face, 0]
        outward = grid.face_centroids[face] - grid.cell_centroids[owner]
        assert np.dot(grid.face_unit_normals[face], outward) > 0


@pytest.mark.parametrize("seed", range(8))
def test_random_faulted_grids_match_the_analytic_overlap(seed):
    rng = np.random.default_rng(seed)
    nx, ny, nz = (int(rng.integers(2, 5)), int(rng.integers(2, 4)), int(rng.integers(2, 5)))
    widths_x, widths_y = rng.uniform(5, 15, nx), rng.uniform(5, 15, ny)
    thickness = rng.uniform(3, 9, (nz, ny, nx))
    shift = np.round(rng.uniform(-6, 6, (ny, nx)), 1)
    top = np.stack([shift + (thickness[:k].sum(axis=0) if k else 0.0) for k in range(nz)])
    bottom = top + thickness
    grid = build_grid(top, thickness, widths_x, widths_y)

    def flat(i, j, k):
        return i + j * nx + k * nx * ny

    expected = {}
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                for di, dj, width in ((1, 0, widths_y[j]), (0, 1, widths_x[i])):
                    ii, jj = i + di, j + dj
                    if ii >= nx or jj >= ny:
                        continue
                    for k2 in range(nz):
                        overlap = min(bottom[k, j, i], bottom[k2, jj, ii]) - max(
                            top[k, j, i], top[k2, jj, ii]
                        )
                        if overlap > 1e-6:
                            expected[tuple(sorted((flat(i, j, k), flat(ii, jj, k2))))] = (
                                overlap * width
                            )
    actual = lateral_connection_areas(grid)
    assert set(actual) == set(expected)
    for key, area in expected.items():
        assert actual[key] == pytest.approx(area, rel=1e-9)


def test_named_fault_over_an_offset_uses_the_overlap_faces():
    top = np.zeros((2, 1, 2))
    top[0, 0] = [0.0, 5.0]
    top[1, 0] = [10.0, 15.0]
    grid = build_grid(
        top,
        np.full((2, 1, 2), 10.0),
        [10.0, 10.0],
        [10.0],
        fault_records=[FaultRecord("F", 1, 1, 1, 1, 1, 2, "X")],
        fault_transmissibility_multipliers={"F": 0.1},
    )
    assert len(grid.fault_face_indices["F"]) == 2


def incident_area(grid, cell, axis):
    total = 0.0
    for face in range(grid.n_faces):
        owner, neighbour = (int(c) for c in grid.face_cell_indices[face])
        if cell in (owner, neighbour) and abs(grid.face_unit_normals[face][axis]) > 0.9:
            total += float(grid.face_areas[face])
    return total


@pytest.mark.parametrize("seed", range(6))
def test_overlaps_and_uncovered_remainders_add_back_up_to_the_full_face(seed):
    rng = np.random.default_rng(seed)
    nx, ny, nz = (int(rng.integers(2, 4)), int(rng.integers(2, 4)), int(rng.integers(2, 4)))
    widths_x, widths_y = rng.uniform(5, 15, nx), rng.uniform(5, 15, ny)
    thickness = rng.uniform(3, 9, (nz, ny, nx))
    shift = np.round(rng.uniform(-6, 6, (ny, nx)), 1)
    top = np.stack([shift + (thickness[:k].sum(axis=0) if k else 0.0) for k in range(nz)])
    grid = build_grid(top, thickness, widths_x, widths_y)
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                cell = i + j * nx + k * nx * ny
                assert incident_area(grid, cell, 0) == pytest.approx(
                    2 * widths_y[j] * thickness[k, j, i], rel=1e-9
                )
                assert incident_area(grid, cell, 1) == pytest.approx(
                    2 * widths_x[i] * thickness[k, j, i], rel=1e-9
                )


def test_uncovered_part_of_an_offset_face_stays_a_boundary_face():
    top = np.zeros((2, 1, 2))
    top[0, 0] = [0.0, 5.0]
    top[1, 0] = [10.0, 15.0]
    grid = build_grid(top, np.full((2, 1, 2), 10.0), [10.0, 10.0], [10.0])
    boundary_areas = sorted(
        round(float(grid.face_areas[face]), 6)
        for face in grid.boundary_face_indices
        if grid.face_cell_indices[face, 0] == 0 and grid.face_unit_normals[face][0] > 0.9
    )
    assert boundary_areas == [50.0]


def vertical_pairs(grid):
    pairs = {}
    for face in range(grid.n_faces):
        owner, neighbour = (int(c) for c in grid.face_cell_indices[face])
        if owner >= 0 and neighbour >= 0 and abs(grid.face_unit_normals[face][2]) > 0.9:
            pairs[tuple(sorted((owner, neighbour)))] = float(grid.face_areas[face])
    return pairs


@pytest.mark.parametrize("second_top", [12.0, 8.0])
def test_layers_with_a_gap_or_overlap_are_still_connected_vertically(second_top):
    top = np.array([[[0.0]], [[second_top]]])
    grid = build_grid(top, np.full((2, 1, 1), 10.0), [10.0], [10.0])
    assert vertical_pairs(grid) == {(0, 1): pytest.approx(100.0)}
    boundary_vertical = [
        face for face in grid.boundary_face_indices if abs(grid.face_unit_normals[face][2]) > 0.9
    ]
    assert len(boundary_vertical) == 2  # only the grid's own top and bottom


def test_an_inactive_cell_blocks_the_vertical_connection():
    top = np.array([[[0.0]], [[12.0]], [[24.0]]])
    actnum = np.array([[[1]], [[0]], [[1]]], dtype=np.int32)
    grid = build_grid(top, np.full((3, 1, 1), 10.0), [10.0], [10.0], actnum=actnum)
    assert vertical_pairs(grid) == {}


def test_a_zero_thickness_cell_is_connected_across():
    top = np.array([[[0.0]], [[10.0]], [[12.0]]])
    thickness = np.array([[[10.0]], [[0.0]], [[10.0]]])
    grid = build_grid(top, thickness, [10.0], [10.0])
    assert (0, 2) in vertical_pairs(grid)
