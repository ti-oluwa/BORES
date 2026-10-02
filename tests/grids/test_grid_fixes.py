import warnings

import numpy as np
import pytest

from bores.grids import Grid
from bores.grids.factories.base import FaultRecord
from bores.grids.factories.cartesian import make_cartesian_grid
from bores.grids.factories.polyhedral import make_polyhedral_grid
from bores.grids.factories.voronoi import make_voronoi_grid
from bores.grids.utils import classify_boundary_cells
from bores.serde.base import register_ndarray_serializers
from bores.types import Side

warnings.simplefilter("ignore")


def test_polyhedral_accepts_plain_lists_and_wide_dtypes():
    points = [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ]
    grid = make_polyhedral_grid(
        vertex_coordinates=points,
        cell_blocks=[{"cell_type": "hexahedron", "connectivity": [list(range(8))]}],
    )
    assert grid.cell_volumes[0] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "cell_type, points, volume",
    [
        ("tetra", [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], 1 / 6),
        ("wedge", [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]], 0.5),
        ("pyramid", [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 1]], 1 / 3),
    ],
)
def test_element_face_tables_have_outward_normals(cell_type, points, volume):
    grid = make_polyhedral_grid(
        vertex_coordinates=np.array(points, float),
        cell_blocks=[{"cell_type": cell_type, "connectivity": [list(range(len(points)))]}],
    )
    assert grid.cell_volumes[0] == pytest.approx(volume)
    for face in grid.get_cell_face_indices(0):
        outward = grid.face_centroids[face] - grid.cell_centroids[0]
        assert np.dot(grid.get_face_normal_for_cell(face, 0), outward) > 0


@pytest.mark.parametrize("seed", range(10))
def test_voronoi_2d_fills_the_box(seed):
    rng = np.random.default_rng(seed)
    thickness = rng.random(int(rng.integers(1, 4))) * 10 + 1
    grid = make_voronoi_grid(
        rng.random((int(rng.integers(8, 60)), 2)) * 1000,
        bounding_box=(0.0, 1000.0, 0.0, 1000.0),
        z_top=2000.0,
        layer_thicknesses=thickness,
    )
    assert grid.cell_volumes.sum() == pytest.approx(1e6 * thickness.sum(), rel=1e-10)


def test_volume_precision_at_large_coordinate_offsets():
    ox, oy, oz = 4.5e5, 7.2e6, 2000.0
    rng = np.random.default_rng(5)
    seeds = rng.random((60, 3)) * np.array([1000, 800, 100]) + np.array([ox, oy, oz])
    grid = make_voronoi_grid(seeds, bounding_box=(ox, ox + 1000, oy, oy + 800, oz, oz + 100))
    assert grid.cell_volumes.sum() == pytest.approx(1000 * 800 * 100, rel=1e-12)


def test_boundary_faces_have_real_owner_and_minus_one_neighbour():
    grid = make_cartesian_grid(nx=4, ny=3, nz=2, dx=10.0, dy=10.0, dz=1.0)
    boundary = grid.face_cell_indices[grid.boundary_face_indices]
    assert (boundary[:, 0] >= 0).all()
    assert (boundary[:, 1] == -1).all()
    cells = classify_boundary_cells(grid)
    assert sorted(cells[Side.WEST].tolist()) == [i for i in range(24) if i % 4 == 0]
    assert all(len(cells[side]) > 0 and (cells[side] >= 0).all() for side in Side)


def test_cartesian_grid_has_dimensions():
    grid = make_cartesian_grid(nx=4, ny=3, nz=2, dx=10.0, dy=10.0, dz=1.0)
    assert tuple(grid.dimensions) == (4, 3, 2)
    assert grid.flat_index(1, 0, 0) == 1


def _fault_cells(direction, i):
    grid = make_cartesian_grid(
        nx=5,
        ny=4,
        nz=3,
        dx=10.0,
        dy=10.0,
        dz=1.0,
        fault_records=[FaultRecord("F", i, i, 1, 1, 1, 1, direction)],
    )
    faces = (grid.fault_face_indices or {}).get("F", [])
    return [tuple(int(c) for c in grid.face_cell_indices[f]) for f in faces]


def test_negative_fault_directions_select_the_negative_face():
    assert _fault_cells("X", 3) == [(2, 3)]
    assert _fault_cells("X-", 3) == [(1, 2)]
    assert _fault_cells("X-", 1) == []
    assert _fault_cells("X", 5) == []


def test_serde_round_trip_keeps_statuses_and_dimensions():
    import attrs

    register_ndarray_serializers()
    grid = make_cartesian_grid(nx=4, ny=3, nz=2, dx=10.0, dy=10.0, dz=1.0)
    statuses = np.ones(24, dtype=np.int8)
    statuses[5] = 0
    grid = attrs.evolve(grid, cell_statuses=statuses)
    loaded = Grid.load(grid.dump())
    assert np.array_equal(loaded.cell_statuses, statuses)
    assert tuple(loaded.dimensions) == (4, 3, 2)
