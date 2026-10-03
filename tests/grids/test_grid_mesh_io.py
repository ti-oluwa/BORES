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


def skewed_corner_point_grid():
    nx, ny, nz = 3, 3, 2
    coord = np.zeros((ny + 1, nx + 1, 6))
    for j in range(ny + 1):
        for i in range(nx + 1):
            coord[j, i] = [i * 10, j * 10, 0, i * 10, j * 10, 100]
    zcorn = np.zeros((2 * nz, 2 * ny, 2 * nx))
    for k in range(nz):
        zcorn[2 * k] = k * 10
        zcorn[2 * k + 1] = (k + 1) * 10
    pillar_x = (np.arange(2 * nx) + 1) // 2
    pillar_y = (np.arange(2 * ny) + 1) // 2
    zcorn += 4.0 * pillar_x[None, None, :] + 2.5 * pillar_y[None, :, None]
    from bores.grids.factories.corner_point import make_corner_point_grid

    return make_corner_point_grid(coord=coord, zcorn=zcorn, metadata={"dimensions": (nx, ny, nz)})


@pytest.mark.parametrize("file_format", ["vtk", "vtu"])
def test_meshio_exports_skewed_hexahedra_exactly(file_format, recwarn):
    pytest.importorskip("meshio")
    from bores.grids.io import meshio as mesh_io

    grid = skewed_corner_point_grid()
    assert grid.cell_length_z[4] > grid.cell_thickness[4] + 1.0
    loaded = mesh_io.load_mesh(
        mesh_io.dump_mesh(grid, file_format=file_format), file_format=file_format
    )
    assert loaded.n_cells == grid.n_cells
    assert loaded.n_faces == grid.n_faces
    assert np.allclose(np.sort(loaded.cell_volumes), np.sort(grid.cell_volumes))
    assert not [w for w in recwarn if "bounding box" in str(w.message)]


def test_meshio_warns_when_a_cell_has_to_be_approximated():
    pytest.importorskip("meshio")
    from bores.grids.factories.voronoi import make_voronoi_grid
    from bores.grids.io import meshio as mesh_io

    rng = np.random.default_rng(3)
    grid = make_voronoi_grid(
        rng.random((12, 3)) * np.array([100, 100, 50]),
        bounding_box=(0.0, 100.0, 0.0, 100.0, 0.0, 50.0),
    )
    with pytest.warns(UserWarning, match="bounding box"):
        mesh_io.dump_mesh(grid, file_format="vtk")


def test_cell_bounding_boxes_ignore_the_connection_across_a_layer_gap():
    from bores.grids.factories.corner_point import make_corner_point_grid

    coord = np.zeros((2, 2, 6))
    for j in range(2):
        for i in range(2):
            coord[j, i] = [i * 10, j * 10, 0, i * 10, j * 10, 100]
    zcorn = np.zeros((4, 2, 2))
    zcorn[0], zcorn[1], zcorn[2], zcorn[3] = 0.0, 10.0, 14.0, 24.0  # a 4-unit gap between layers
    grid = make_corner_point_grid(coord=coord, zcorn=zcorn, metadata={"dimensions": (1, 1, 2)})
    assert grid.cell_min_xyz[0, 2] == pytest.approx(0.0) and grid.cell_max_xyz[
        0, 2
    ] == pytest.approx(10.0)
    assert grid.cell_min_xyz[1, 2] == pytest.approx(14.0) and grid.cell_max_xyz[
        1, 2
    ] == pytest.approx(24.0)
    assert any(abs(grid.face_unit_normals[f][2]) > 0.9 for f in grid.interior_face_indices)
