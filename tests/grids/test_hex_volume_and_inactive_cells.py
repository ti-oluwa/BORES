import itertools
import warnings

import attrs
import numpy as np
import pytest

from bores.errors import InvalidGridError
from bores.grids import Grid
from bores.grids.factories.base import FaultRecord
from bores.grids.factories.cartesian import make_cartesian_grid
from bores.grids.factories.corner_point import (
    compute_hex_volumes_and_centroids,
    make_corner_point_grid,
)
from bores.grids.factories.polyhedral import make_polyhedral_grid
from bores.grids.factories.voronoi import make_voronoi_grid
from bores.serde.base import register_ndarray_serializers

warnings.simplefilter("ignore")

BOX = np.array(
    [
        [0, 0, 0],
        [10, 0, 0],
        [10, 10, 0],
        [0, 10, 0],
        [0, 0, 10],
        [10, 0, 10],
        [10, 10, 10],
        [0, 10, 10],
    ],
    dtype=float,
)
NODE_SIGNS = np.array(
    [
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ],
    dtype=float,
)


def reference_volume_and_centroid(corners):
    """Exact trilinear volume and centroid by 2x2x2 Gauss quadrature, written independently."""
    points = np.array([-1.0, 1.0]) / np.sqrt(3.0)
    volume, weighted = 0.0, np.zeros(3)
    for a, b, g in itertools.product(points, repeat=3):
        derivatives = np.array([
            [
                sx * (1 + sy * b) * (1 + sz * g) / 8,
                sy * (1 + sx * a) * (1 + sz * g) / 8,
                sz * (1 + sx * a) * (1 + sy * b) / 8,
            ]
            for sx, sy, sz in NODE_SIGNS
        ])
        shape = np.array([
            (1 + sx * a) * (1 + sy * b) * (1 + sz * g) / 8 for sx, sy, sz in NODE_SIGNS
        ])
        determinant = np.linalg.det(derivatives.T @ corners)
        volume += determinant
        weighted += determinant * (shape @ corners)
    return volume, weighted / volume


@pytest.mark.parametrize("seed", range(6))
def test_hex_kernel_matches_the_exact_trilinear_integral_for_warped_cells(seed):
    rng = np.random.default_rng(seed)
    offset = np.array([4.5e5, 7.2e6, 2000.0])
    shifted = BOX + rng.uniform(-2.5, 2.5, (8, 3)) + offset
    # The reference sees the same (rounded) coordinates the kernel receives.
    expected_volume, expected_centroid = reference_volume_and_centroid(shifted - offset)
    volumes, centroids = compute_hex_volumes_and_centroids(
        np.arange(8, dtype=np.int32)[None, :], shifted
    )
    assert volumes[0] == pytest.approx(expected_volume, rel=1e-12)
    assert np.allclose(centroids[0], expected_centroid + offset, atol=1e-8)


def test_hex_kernel_reports_an_inverted_cell_as_negative():
    volumes, _ = compute_hex_volumes_and_centroids(
        np.arange(8, dtype=np.int32)[None, :], BOX[[4, 5, 6, 7, 0, 1, 2, 3]]
    )
    assert volumes[0] == pytest.approx(-1000.0)


def test_an_inverted_active_cell_is_rejected():
    coord = np.zeros((2, 2, 6))
    for j in range(2):
        for i in range(2):
            coord[j, i] = [i * 10, j * 10, 0, i * 10, j * 10, 100]
    zcorn = np.zeros((2, 2, 2))
    zcorn[0], zcorn[1] = 10.0, 0.0  # bottom above top
    with pytest.raises(InvalidGridError):
        make_corner_point_grid(coord=coord, zcorn=zcorn, metadata={"dimensions": (1, 1, 1)})


def voronoi_grid(statuses=None):
    rng = np.random.default_rng(3)
    return make_voronoi_grid(
        rng.random((15, 3)) * np.array([100, 100, 50]),
        bounding_box=(0.0, 100.0, 0.0, 100.0, 0.0, 50.0),
        cell_statuses=statuses,
    )


def assert_inactive_cells_are_isolated(grid, inactive):
    assert np.allclose(grid.cell_volumes[inactive], 0.0)
    assert np.isfinite(grid.cell_centroids).all() and np.isfinite(grid.cell_min_xyz).all()
    for cell in inactive:
        assert len(grid.get_cell_face_indices(cell)) == 0
        assert len(grid.get_cell_neighbor_indices(cell)) == 0
    for cell in range(grid.n_cells):
        assert not set(grid.get_cell_neighbor_indices(cell)) & set(inactive.tolist())


def test_voronoi_inactive_cells_have_no_faces_and_zero_volume():
    full = voronoi_grid()
    statuses = np.ones(full.n_cells, dtype=np.int8)
    statuses[[4, 9]] = 0
    grid = voronoi_grid(statuses)
    assert grid.n_cells == full.n_cells
    assert_inactive_cells_are_isolated(grid, np.array([4, 9]))
    assert grid.cell_volumes.sum() == pytest.approx(
        full.cell_volumes.sum() - full.cell_volumes[[4, 9]].sum()
    )
    assert grid.n_faces < full.n_faces


def test_marking_cells_inactive_with_evolve_gives_the_same_result():
    full = voronoi_grid()
    statuses = np.ones(full.n_cells, dtype=np.int8)
    statuses[4] = 0
    via_evolve = attrs.evolve(full, cell_statuses=statuses)
    direct = voronoi_grid(statuses)
    assert via_evolve.n_faces == direct.n_faces
    assert np.array_equal(via_evolve.face_cell_indices, direct.face_cell_indices)
    assert_inactive_cells_are_isolated(via_evolve, np.array([4]))


def test_polyhedral_inactive_cells_have_no_faces():
    points = np.array([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1, 2)], dtype=float)
    cells = [[0, 1, 4, 3, 6, 7, 10, 9], [1, 2, 5, 4, 7, 8, 11, 10]]
    grid = make_polyhedral_grid(
        vertex_coordinates=points,
        cell_blocks=[{"cell_type": "hexahedron", "connectivity": cells}],
        cell_statuses=np.array([1, 0], dtype=np.int8),
    )
    assert grid.n_cells == 2
    assert_inactive_cells_are_isolated(grid, np.array([1]))
    assert grid.cell_volumes[0] == pytest.approx(1.0)


def test_removing_faces_renumbers_named_faults():
    base = make_cartesian_grid(
        nx=5,
        ny=1,
        nz=1,
        dx=10.0,
        dy=10.0,
        dz=5.0,
        fault_records=[FaultRecord("F", 4, 4, 1, 1, 1, 1, "X")],
    )
    statuses = np.ones(5, dtype=np.int8)
    statuses[0] = 0
    grid = attrs.evolve(base, cell_statuses=statuses)
    (face,) = grid.fault_face_indices["F"]
    assert sorted(int(c) for c in grid.face_cell_indices[face]) == [3, 4]


def test_inactive_cells_survive_a_serde_round_trip():
    register_ndarray_serializers()
    statuses = np.ones(voronoi_grid().n_cells, dtype=np.int8)
    statuses[[2, 7]] = 0
    grid = voronoi_grid(statuses)
    loaded = Grid.load(grid.dump())
    assert np.array_equal(loaded.cell_statuses, statuses)
    assert loaded.n_faces == grid.n_faces
    assert np.allclose(loaded.cell_volumes, grid.cell_volumes)
    assert_inactive_cells_are_isolated(loaded, np.array([2, 7]))
