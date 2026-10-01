import warnings

import numpy as np
import pytest

from bores.errors import InvalidFaceConnectivityError, ValidationError
from bores.grids import Grid
from bores.grids.factories.cartesian import make_cartesian_grid
from bores.grids.factories.corner_point import make_corner_point_grid
from bores.grids.factories.polyhedral import make_polyhedral_grid
from bores.grids.io.grdecl import build_grdecl_text, load_grdecl
from bores.grids.io.grdecl import GridImportError
from bores.types import UnitSystem

warnings.simplefilter("ignore")


def corner_point(nx, ny, nz, actnum=None, zcorn_edit=None, **kwargs):
    coord = np.zeros((ny + 1, nx + 1, 6))
    for j in range(ny + 1):
        for i in range(nx + 1):
            coord[j, i] = [i * 10, j * 10, 0, i * 10, j * 10, 100]
    zcorn = np.zeros((nz * 2, ny * 2, nx * 2))
    for k in range(nz):
        zcorn[2 * k] = k * 10
        zcorn[2 * k + 1] = (k + 1) * 10
    if zcorn_edit is not None:
        zcorn = zcorn_edit(zcorn)
    return make_corner_point_grid(
        coord=coord, zcorn=zcorn, actnum=actnum, metadata={"dimensions": (nx, ny, nz)}, **kwargs
    )


def test_inactive_cells_keep_their_flat_index():
    actnum = np.ones((2, 2, 3), dtype=np.int32)
    actnum[0, 0, 0] = 0
    grid = corner_point(3, 2, 2, actnum)
    assert grid.n_cells == 12
    assert grid.cell_statuses[0] == 0 and grid.cell_volumes[0] == 0.0
    assert len(grid.get_cell_face_indices(0)) == 0
    for i, j, k in [(1, 0, 0), (2, 1, 1), (0, 1, 0)]:
        flat = grid.flat_index(i, j, k)
        assert flat == i + j * 3 + k * 6
        assert np.allclose(grid.cell_centroids[flat], [i * 10 + 5, j * 10 + 5, k * 10 + 5])
    assert grid.cell_volumes.sum() == pytest.approx(11 * 1000.0)


def test_user_nnc_on_inactive_cell_is_dropped():
    actnum = np.ones((2, 2, 3), dtype=np.int32)
    actnum[0, 0, 0] = 0
    grid = corner_point(
        3, 2, 2, actnum,
        nnc_cell_indices=np.array([[0, 11], [1, 11]]),
        nnc_transmissibilities=np.array([1.0, 2.0]),
    )
    assert grid.nnc_cell_indices.tolist() == [[1, 11]]


def test_face_centroid_is_area_weighted():
    # Trapezoid in the z=0 plane with parallel sides 4 and 1 (height 3); the area-weighted
    # centroid sits at y = 3 * (4 + 2 * 1) / (3 * (4 + 1)) = 1.2, not at the vertex mean 1.5.
    points = np.array([[0, 0, 0], [4, 0, 0], [3, 3, 0], [2, 3, 0], [0, 0, 1], [4, 0, 1], [3, 3, 1], [2, 3, 1]], float)
    grid = make_polyhedral_grid(
        vertex_coordinates=points,
        cell_blocks=[{"cell_type": "hexahedron", "connectivity": [list(range(8))]}],
    )
    bottom = next(
        f for f in grid.get_cell_face_indices(0) if np.allclose(grid.face_centroids[f][2], 0.0)
    )
    assert grid.face_centroids[bottom][1] == pytest.approx(1.2)
    assert grid.face_areas[bottom] == pytest.approx(7.5)


def test_cell_thickness_is_vertical_not_bounding_box_height():
    def dip(zcorn):
        zcorn = zcorn.copy()
        zcorn += (np.arange(zcorn.shape[2]) * 4.0)[None, None, :]  # dips 4 per half-cell in x
        return zcorn

    grid = corner_point(3, 2, 2, zcorn_edit=dip)
    assert np.allclose(grid.cell_thickness, 10.0)
    assert (grid.cell_length_z > 10.0).all()


def test_convert_scales_nnc_transmissibility_by_permeability_and_length():
    grid = make_cartesian_grid(
        nx=4, ny=3, nz=2, dx=10.0, dy=10.0, dz=1.0,
        nnc_cell_indices=np.array([[0, 23], [1, 22]]),
        nnc_transmissibilities=np.array([5.0, np.nan]),
    )
    metric = grid.convert(UnitSystem.METRIC)
    assert metric.nnc_transmissibilities[0] == pytest.approx(5.0 * 0.3048)
    assert np.isnan(metric.nnc_transmissibilities[1])


def write_deck(tmp_path, body):
    path = tmp_path / "grid.grdecl"
    path.write_text(body)
    return path


def keyword(name, values):
    return f"{name}\n" + " ".join(map(str, values)) + " /\n"


NX, NY, NZ = 4, 3, 3
N = NX * NY * NZ


def cartesian_deck(dx=None, dz=None, tops=None, actnum=None, extra=""):
    text = f"SPECGRID\n{NX} {NY} {NZ} 1 F /\nGRIDUNIT\nFEET /\n"
    text += keyword("DX", dx or [10] * N) + keyword("DY", [12] * N) + keyword("DZ", dz or [2] * N)
    text += keyword("TOPS", tops or [2000] * (NX * NY))
    if actnum is not None:
        text += keyword("ACTNUM", actnum)
    return text + extra


def test_uniform_deck_stays_a_cartesian_grid(tmp_path):
    grid = load_grdecl(write_deck(tmp_path, cartesian_deck()), unit_system=UnitSystem.FIELD)
    assert grid.metadata["source_format"] == "grdecl_cartesian"
    assert grid.cell_volumes.sum() == pytest.approx(NX * 10 * NY * 12 * NZ * 2)


def test_dipping_tops_and_varying_dz_are_kept_and_connected(tmp_path):
    tops = [2000 + 5 * i + 3 * j for j in range(NY) for i in range(NX)]
    dz = [1 + 0.2 * i + 0.1 * k for k in range(NZ) for j in range(NY) for i in range(NX)]
    grid = load_grdecl(write_deck(tmp_path, cartesian_deck(dz=dz, tops=tops)), unit_system=UnitSystem.FIELD)
    expected_interior = (NX - 1) * NY * NZ + NX * (NY - 1) * NZ + NX * NY * (NZ - 1)
    assert len(grid.interior_face_indices) == expected_interior
    assert grid.cell_volumes.sum() == pytest.approx(sum(dz) * 10 * 12, rel=0.02)
    assert grid.cell_center_depths.min() >= min(tops) and grid.cell_center_depths.max() <= max(tops) + sum(dz)
    assert grid.cell_thickness.min() >= min(dz) - 1e-9 and grid.cell_thickness.max() <= max(dz) + 1e-9


def test_actnum_in_a_cartesian_deck_is_applied(tmp_path):
    actnum = [1] * N
    actnum[4] = 0
    grid = load_grdecl(write_deck(tmp_path, cartesian_deck(actnum=actnum)), unit_system=UnitSystem.FIELD)
    assert grid.cell_statuses[grid.flat_index(0, 1, 0)] == 0
    assert grid.cell_statuses.sum() == N - 1


def test_dx_varying_across_rows_is_rejected(tmp_path):
    dx = [10 + j for _ in range(NZ) for j in range(NY) for _ in range(NX)]
    with pytest.raises(GridImportError):
        load_grdecl(write_deck(tmp_path, cartesian_deck(dx=dx)), unit_system=UnitSystem.FIELD)


def test_deck_nnc_transmissibility_is_converted_to_geometric_and_back(tmp_path):
    deck = cartesian_deck(extra="NNC\n1 1 1 4 3 3 0.5 /\n/\n")
    grid = load_grdecl(write_deck(tmp_path, deck), unit_system=UnitSystem.FIELD)
    assert grid.nnc_transmissibilities[-1] == pytest.approx(0.5 / 0.00112712)
    path = tmp_path / "out.grdecl"
    path.write_text(build_grdecl_text(grid))
    reloaded = load_grdecl(path, unit_system=UnitSystem.FIELD)
    assert reloaded.nnc_transmissibilities[-1] == pytest.approx(grid.nnc_transmissibilities[-1], rel=1e-5)


def test_export_round_trip_with_inactive_cells(tmp_path):
    actnum = [1] * N
    actnum[4] = actnum[20] = 0
    dz = [1 + 0.1 * k + 0.05 * i for k in range(NZ) for j in range(NY) for i in range(NX)]
    grid = load_grdecl(write_deck(tmp_path, cartesian_deck(dz=dz, actnum=actnum)), unit_system=UnitSystem.FIELD)
    path = tmp_path / "out.grdecl"
    path.write_text(build_grdecl_text(grid))
    reloaded = load_grdecl(path, unit_system=UnitSystem.FIELD)
    assert np.array_equal(reloaded.cell_statuses, grid.cell_statuses)
    assert np.allclose(reloaded.cell_volumes, grid.cell_volumes)


def test_validation_rejects_inconsistent_inputs():
    good = make_cartesian_grid(nx=3, ny=2, nz=2, dx=1.0, dy=1.0, dz=1.0)
    import attrs

    with pytest.raises(ValidationError):
        attrs.evolve(good, positive_x_transmissibility_multipliers=np.ones(5))
    with pytest.raises(ValidationError):
        attrs.evolve(good, positive_x_transmissibility_multipliers=-np.ones(12))
    with pytest.raises(ValidationError):
        attrs.evolve(good, cell_statuses=np.full(12, 7, dtype=np.int8))
    with pytest.raises(InvalidFaceConnectivityError):
        attrs.evolve(good, nnc_cell_indices=np.array([[0, 99]]))
    with pytest.raises(InvalidFaceConnectivityError):
        attrs.evolve(good, nnc_cell_indices=np.array([[2, 2]]))
    with pytest.raises(ValidationError):
        attrs.evolve(good, dimensions=(5, 5, 5))
    with pytest.raises(Exception):
        bad = good.vertex_coordinates.copy()
        bad[0, 0] = np.nan
        attrs.evolve(good, vertex_coordinates=bad)


def test_boundary_transmissibility_uses_the_adjacent_cell_permeability():
    import bores.reservoir.rock as rock_module
    from bores.reservoir.transmissibility import compute_connection_transmissibilities

    nx, ny, nz = 4, 3, 2
    n = nx * ny * nz
    grid = make_cartesian_grid(nx=nx, ny=ny, nz=nz, dx=10.0, dy=10.0, dz=5.0)
    kx = np.arange(1, n + 1, dtype=float) * 10
    permeability_type = next(getattr(rock_module, name) for name in dir(rock_module) if "Permeab" in name)
    ones = np.ones(n)
    rock = rock_module.Rock(
        porosity=ones * 0.2,
        absolute_permeability=permeability_type(x=kx, y=kx, z=kx),
        net_to_gross=ones,
        connate_water_saturation=ones * 0.2,
        irreducible_water_saturation=ones * 0.2,
        residual_oil_saturation_water=ones * 0.2,
        residual_oil_saturation_gas=ones * 0.1,
        residual_gas_saturation=ones * 0.05,
    )
    transmissibilities = compute_connection_transmissibilities(grid, rock)
    assert transmissibilities.nnc is None
    for position, face in enumerate(grid.boundary_face_indices):
        cell = grid.face_cell_indices[face, 0]
        distance = abs(np.dot(grid.face_centroids[face] - grid.cell_centroids[cell], grid.face_unit_normals[face]))
        expected = kx[cell] * grid.face_areas[face] / distance
        assert transmissibilities.boundary[position] == pytest.approx(expected)
