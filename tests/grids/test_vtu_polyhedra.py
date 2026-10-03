import warnings

import numpy as np
import pytest

from bores.errors import GridExportError, GridImportError
from bores.grids.factories.cartesian import make_cartesian_grid
from bores.grids.factories.corner_point import make_corner_point_grid
from bores.grids.factories.voronoi import make_voronoi_grid
from bores.grids.io.vtu import dump_polyhedral, load_polyhedral

warnings.simplefilter("ignore")


def face_set(grid):
    faces = {}
    for face in range(grid.n_faces):
        start, end = grid.face_vertex_offsets[face], grid.face_vertex_offsets[face + 1]
        key = tuple(
            sorted(
                map(
                    tuple,
                    np.round(grid.vertex_coordinates[grid.face_vertex_indices[start:end]], 9),
                )
            )
        )
        faces[key] = tuple(sorted(int(cell) for cell in grid.face_cell_indices[face]))
    return faces


def voronoi_grid(statuses=None):
    rng = np.random.default_rng(3)
    return make_voronoi_grid(
        rng.random((20, 3)) * np.array([100, 100, 50]),
        bounding_box=(0.0, 100.0, 0.0, 100.0, 0.0, 50.0),
        cell_statuses=statuses,
    )


def faulted_hex_grid():
    coord = np.zeros((2, 3, 6))
    for j in range(2):
        for i in range(3):
            coord[j, i] = [i * 10, j * 10, -50, i * 10, j * 10, 200]
    zcorn = np.zeros((4, 2, 4))
    zcorn[0], zcorn[1], zcorn[2], zcorn[3] = 0.0, 10.0, 10.0, 20.0
    zcorn[:, :, 2:] += 5.0  # the right column drops by half a cell: a fault at the first pillar
    return make_corner_point_grid(coord=coord, zcorn=zcorn, metadata={"dimensions": (2, 1, 2)})


@pytest.mark.parametrize("make_grid", [voronoi_grid, faulted_hex_grid])
def test_polyhedral_vtu_round_trip_is_exact(make_grid):
    grid = make_grid()
    loaded = load_polyhedral(dump_polyhedral(grid))
    assert loaded is not None
    assert loaded.n_cells == grid.n_cells
    assert face_set(loaded) == face_set(grid)
    assert np.allclose(loaded.cell_volumes, grid.cell_volumes)
    assert np.allclose(loaded.cell_centroids, grid.cell_centroids)


def test_cells_next_to_inactive_cells_keep_their_stored_geometry():
    statuses = np.where(np.arange(20) % 7 == 3, 0, 1).astype(np.int8)
    grid = voronoi_grid(statuses)
    active = statuses == 1
    loaded = load_polyhedral(dump_polyhedral(grid))
    assert loaded.n_cells == int(active.sum())
    assert np.allclose(loaded.cell_volumes, grid.cell_volumes[active])
    assert np.allclose(loaded.cell_min_xyz, grid.cell_min_xyz[active])
    assert np.allclose(loaded.cell_max_xyz, grid.cell_max_xyz[active])


def test_cell_data_is_written_and_validated():
    grid = voronoi_grid()
    payload = dump_polyhedral(grid, cell_data={"pressure": np.arange(grid.n_cells, dtype=float)})
    assert b'Name="pressure"' in payload and b'Name="cell_index"' in payload
    with pytest.raises(GridExportError):
        dump_polyhedral(grid, cell_data={"bad": np.zeros(3)})


def test_documents_that_are_not_polyhedral_are_left_to_a_generic_reader():
    assert load_polyhedral(b"not xml at all") is None
    plain = (
        b'<VTKFile type="UnstructuredGrid"><UnstructuredGrid><Piece NumberOfPoints="0" '
        b'NumberOfCells="1"><Cells><DataArray type="UInt8" Name="types" format="ascii">12'
        b"</DataArray></Cells></Piece></UnstructuredGrid></VTKFile>"
    )
    assert load_polyhedral(plain) is None


def test_binary_polyhedron_arrays_are_rejected_clearly():
    grid = voronoi_grid()
    payload = dump_polyhedral(grid).replace(
        b'format="ascii" Name="faces"', b'format="binary" Name="faces"'
    )
    with pytest.raises(GridImportError):
        load_polyhedral(payload)


def test_meshio_uses_exact_polyhedra_for_vtu_and_standard_cells_for_hexahedral_grids():
    pytest.importorskip("meshio")
    from bores.grids.io import meshio as mesh_io

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        voronoi = voronoi_grid()
        loaded = mesh_io.load_mesh(
            mesh_io.dump_mesh(voronoi, file_format="vtu"), file_format="vtu"
        )
    assert face_set(loaded) == face_set(voronoi)

    cartesian = make_cartesian_grid(nx=3, ny=2, nz=2, dx=10.0, dy=10.0, dz=5.0)
    payload = mesh_io.dump_mesh(cartesian, file_format="vtu")
    assert b'<DataArray type="UInt8" Name="types"' not in payload or b">42" not in payload
    assert face_set(mesh_io.load_mesh(payload, file_format="vtu")) == face_set(cartesian)


def test_vtk_legacy_still_warns_when_it_has_to_approximate():
    pytest.importorskip("meshio")
    from bores.grids.io import meshio as mesh_io

    with pytest.warns(UserWarning, match="bounding box"):
        mesh_io.dump_mesh(voronoi_grid(), file_format="vtk")
