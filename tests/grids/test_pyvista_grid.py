import warnings
from pathlib import Path

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from bores.grids.factories.cartesian import (
    make_cartesian_grid,
)
from bores.grids.io.grdecl import load_grdecl
from bores.grids.utils import make_pyvista_grid
from bores.types import UnitSystem

DATA = Path(__file__).resolve().parents[2] / "data"
VTK_HEXAHEDRON = 12
VTK_POLYHEDRON = 42


@pytest.fixture(scope="module")
def cartesian_grid():
    return make_cartesian_grid(nx=6, ny=5, nz=4, dx=10.0, dy=12.0, dz=3.0)


@pytest.fixture(scope="module")
def corner_point_grid():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_grdecl(DATA / "40X48X13_faults.grdecl", unit_system=UnitSystem.FIELD)


def test_default_cells_are_all_polyhedra(cartesian_grid):
    mesh = make_pyvista_grid(cartesian_grid)
    assert mesh.n_cells == cartesian_grid.n_cells
    assert (mesh.celltypes == VTK_POLYHEDRON).all()


def test_hexahedra_keep_cell_count_data_and_volumes_on_flat_cells(cartesian_grid):
    data = {"index": np.arange(cartesian_grid.n_cells, dtype=float)}
    polyhedra = make_pyvista_grid(cartesian_grid, cell_data=data)
    hexahedra = make_pyvista_grid(cartesian_grid, cell_data=data, use_hexahedra=True)
    assert (hexahedra.celltypes == VTK_HEXAHEDRON).all()
    assert np.array_equal(hexahedra.cell_data["index"], polyhedra.cell_data["index"])
    volumes = hexahedra.compute_cell_sizes(length=False, area=False)["Volume"]
    assert np.allclose(volumes, polyhedra.compute_cell_sizes(length=False, area=False)["Volume"])
    assert (volumes > 0).all()


def test_hexahedron_corners_follow_the_cell_faces(corner_point_grid):
    from bores.grids.utils import compute_hexahedron_corners

    grid = corner_point_grid
    points = grid.vertex_coordinates.copy()
    points[:, 2] *= -1
    quads = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    found = 0
    for cell in range(grid.n_cells):
        is_hexahedron, *corners = compute_hexahedron_corners(
            cell,
            grid.cell_face_offsets,
            grid.cell_face_indices,
            grid.face_vertex_offsets,
            grid.face_vertex_indices,
            points,
        )
        if not is_hexahedron:
            continue
        found += 1
        expected = {frozenset(corners[i] for i in quad) for quad in quads}
        faces = grid.cell_face_indices[
            grid.cell_face_offsets[cell] : grid.cell_face_offsets[cell + 1]
        ]
        actual = {
            frozenset(
                grid.face_vertex_indices[
                    grid.face_vertex_offsets[face] : grid.face_vertex_offsets[face + 1]
                ].tolist()
            )
            for face in faces
        }
        assert expected == actual
    assert found > 0


def test_hexahedra_on_corner_point_grid_are_positive_and_keep_data(corner_point_grid):
    data = {"index": np.arange(corner_point_grid.n_cells, dtype=float)}
    plain = make_pyvista_grid(corner_point_grid, cell_data=data)
    hexahedra = make_pyvista_grid(corner_point_grid, cell_data=data, use_hexahedra=True)
    assert hexahedra.n_cells == plain.n_cells
    assert (hexahedra.celltypes == VTK_HEXAHEDRON).any()
    assert np.array_equal(hexahedra.cell_data["index"], plain.cell_data["index"])
    volumes = hexahedra.compute_cell_sizes(length=False, area=False)["Volume"]
    assert (volumes[hexahedra.celltypes == VTK_HEXAHEDRON] > 0).all()


def test_exterior_only_keeps_exactly_the_shell_of_a_box(cartesian_grid):
    data = {"index": np.arange(cartesian_grid.n_cells, dtype=float)}
    shell = make_pyvista_grid(cartesian_grid, cell_data=data, exterior_only=True)
    nx, ny, nz = cartesian_grid.dimensions
    assert shell.n_cells == nx * ny * nz - (nx - 2) * (ny - 2) * (nz - 2)
    index = np.arange(cartesian_grid.n_cells).reshape(nz, ny, nx)
    interior = index[1:-1, 1:-1, 1:-1].ravel()
    assert not np.isin(shell.cell_data["index"], interior).any()


def test_exterior_only_with_hexahedra_matches_either_option_alone(cartesian_grid):
    both = make_pyvista_grid(cartesian_grid, exterior_only=True, use_hexahedra=True)
    shell = make_pyvista_grid(cartesian_grid, exterior_only=True)
    assert both.n_cells == shell.n_cells
    assert (both.celltypes == VTK_HEXAHEDRON).all()


def test_wrong_cell_data_length_raises(cartesian_grid):
    with pytest.raises(ValueError):
        make_pyvista_grid(cartesian_grid, cell_data={"bad": np.zeros(3)})
