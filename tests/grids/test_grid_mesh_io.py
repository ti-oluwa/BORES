import attrs
import numpy as np
import pytest

from bores.errors import GridImportError
from bores.grids.factories.cartesian import make_cartesian_grid
from bores.grids.io.gmsh import load_msh
from bores.types import UnitSystem

TETRA10_MESH = """$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
10
1 0 0 0
2 1 0 0
3 0 1 0
4 0 0 1
5 .5 0 0
6 .5 .5 0
7 0 .5 0
8 0 0 .5
9 .5 0 .5
10 0 .5 .5
$EndNodes
$Elements
2
1 11 0 1 2 3 4 5 6 7 8 9 10
2 2 0 1 2 3
$EndElements
"""


def test_gmsh_reads_raw_text_and_reduces_higher_order_elements():
    grid = load_msh(TETRA10_MESH)
    assert grid.n_cells == 1
    assert grid.cell_volumes[0] == pytest.approx(1 / 6)


def test_gmsh_unit_system_declares_the_mesh_units_and_does_not_rescale():
    grid = load_msh(TETRA10_MESH, unit_system=UnitSystem.METRIC)
    assert grid.unit_system == UnitSystem.METRIC
    assert grid.cell_volumes[0] == pytest.approx(1 / 6)


def test_gmsh_malformed_numbers_raise_grid_import_error():
    with pytest.raises(GridImportError):
        load_msh(TETRA10_MESH.replace("1 0 0 0", "1 0 abc 0"))


@pytest.mark.parametrize("file_format", ["vtk", "vtu"])
def test_meshio_bytes_round_trip_and_inactive_cells(file_format):
    pytest.importorskip("meshio")
    from bores.grids.io import meshio as mesh_io

    grid = make_cartesian_grid(nx=3, ny=2, nz=2, dx=10.0, dy=10.0, dz=5.0)
    statuses = np.ones(12, dtype=np.int8)
    statuses[4] = 0
    grid = attrs.evolve(grid, cell_statuses=statuses)
    payload = mesh_io.dump_mesh(grid, file_format=file_format)
    loaded = mesh_io.load_mesh(payload, file_format=file_format)
    assert loaded.n_cells == 11
    assert loaded.cell_volumes.sum() == pytest.approx(11 * 500.0)


def test_meshio_unit_system_declares_the_mesh_units():
    pytest.importorskip("meshio")
    from bores.grids.io import meshio as mesh_io

    grid = make_cartesian_grid(
        nx=1, ny=1, nz=1, dx=1.0, dy=1.0, dz=1.0, unit_system=UnitSystem.METRIC
    )
    payload = mesh_io.dump_mesh(grid, file_format="vtu")
    loaded = mesh_io.load_mesh(payload, file_format="vtu", unit_system=UnitSystem.METRIC)
    assert loaded.unit_system == UnitSystem.METRIC
    assert loaded.cell_volumes[0] == pytest.approx(1.0)


def test_corner_point_depth_beyond_a_tilted_pillar_extrapolates_along_the_pillar():
    from bores.grids.factories.corner_point import _interpolate_pillar_point

    top = np.array([0.0, 0.0, 0.0])
    bottom = np.array([10.0, 0.0, 100.0])
    assert _interpolate_pillar_point(top, bottom, 150.0)[0] == pytest.approx(15.0)
    assert _interpolate_pillar_point(top, bottom, -50.0)[0] == pytest.approx(-5.0)
    assert _interpolate_pillar_point(top, bottom, 50.0)[0] == pytest.approx(5.0)
