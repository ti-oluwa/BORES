import pathlib
import warnings

import numpy as np
import pytest

from bores.datastructures import MapAxes
from bores.errors import GridImportError, InvalidGridError, ValidationError
from bores.grids.factories.corner_point import InvertedCellPolicy, make_corner_point_grid
from bores.grids.io.grdecl import load_grdecl
from bores.types import UnitSystem

DATA_DIRECTORY = pathlib.Path(__file__).resolve().parents[2] / "data"


def build_arrays(nx=4, ny=2, nz=2, fault_offset=0.0):
    x_edges = np.arange(nx + 1) * 100.0
    y_edges = np.arange(ny + 1) * 80.0
    coord = np.zeros((ny + 1, nx + 1, 6))
    for j in range(ny + 1):
        for i in range(nx + 1):
            coord[j, i] = [x_edges[i], y_edges[j], 0.0, x_edges[i], y_edges[j], 500.0]
    zcorn = np.zeros((2 * nz, 2 * ny, 2 * nx))
    for k in range(nz):
        zcorn[2 * k] = 1000.0 + 20.0 * k
        zcorn[2 * k + 1] = 1020.0 + 20.0 * k
    if fault_offset:
        zcorn[:, :, nx:] += fault_offset
    return coord, zcorn


def make_axes(angle_degrees, y_points_south):
    """Map axes whose X axis is rotated by `angle_degrees` from east."""
    origin = np.array([1000.0, 5000.0])
    angle = np.radians(angle_degrees)
    x_direction = np.array([np.cos(angle), np.sin(angle)])
    y_direction = np.array([x_direction[1], -x_direction[0]])
    if not y_points_south:
        y_direction = -y_direction
    return MapAxes(
        origin=origin,
        map_x_axis_point=origin + 100.0 * x_direction,
        map_y_axis_point=origin + 100.0 * y_direction,
        unit_system=UnitSystem.METRIC,
    )


def inward_face_count(grid):
    owner = grid.face_cell_indices[:, 0]
    has_owner = owner >= 0
    outward = np.einsum(
        "ij,ij->i",
        grid.face_unit_normals[has_owner],
        grid.face_centroids[has_owner] - grid.cell_centroids[owner[has_owner]],
    )
    return int((outward < 0.0).sum())


@pytest.mark.parametrize("fault_offset", [0.0, 7.0])
@pytest.mark.parametrize(
    ("angle_degrees", "y_points_south"),
    [(0.0, True), (0.0, False), (30.0, True), (-30.0, False)],
)
def test_map_axes_do_not_change_volumes_or_face_orientation(
    angle_degrees, y_points_south, fault_offset
):
    coord, zcorn = build_arrays(fault_offset=fault_offset)
    local = make_corner_point_grid(coord=coord, zcorn=zcorn, unit_system=UnitSystem.METRIC)
    placed = make_corner_point_grid(
        coord=coord,
        zcorn=zcorn,
        unit_system=UnitSystem.METRIC,
        map_axes=make_axes(angle_degrees, y_points_south),
    )

    assert (placed.cell_volumes > 0.0).all()
    np.testing.assert_allclose(placed.cell_volumes, local.cell_volumes, rtol=1e-9)
    assert placed.n_faces == local.n_faces
    assert inward_face_count(placed) == inward_face_count(local) == 0
    np.testing.assert_allclose(placed.face_areas, local.face_areas, rtol=1e-9)


def test_map_axes_that_are_not_perpendicular_are_refused():
    coord, zcorn = build_arrays()
    shear = MapAxes(
        origin=np.array([1000.0, 5000.0]),
        map_x_axis_point=np.array([1100.0, 5000.0]),
        map_y_axis_point=np.array([1100.0, 4900.0]),
        unit_system=UnitSystem.METRIC,
    )
    with pytest.raises(ValidationError, match="perpendicular"):
        make_corner_point_grid(
            coord=coord, zcorn=zcorn, unit_system=UnitSystem.METRIC, map_axes=shear
        )


def build_grid_with_inverted_cell(**kwargs):
    coord, zcorn = build_arrays(nx=3, ny=1, nz=1)
    # Lift the bottom of the middle cell above its own top.
    zcorn[1, :, 2:4] = zcorn[0, :, 2:4] - 5.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return make_corner_point_grid(
            coord=coord, zcorn=zcorn, unit_system=UnitSystem.METRIC, **kwargs
        )


def test_inverted_cell_is_refused_by_default():
    with pytest.raises(InvalidGridError, match="negative volume"):
        build_grid_with_inverted_cell()


def test_inverted_cell_can_be_deactivated():
    coord, zcorn = build_arrays(nx=3, ny=1, nz=1)
    zcorn[1, :, 2:4] = zcorn[0, :, 2:4] - 5.0
    with pytest.warns(UserWarning, match="1 active cell.*deactivated"):
        grid = make_corner_point_grid(
            coord=coord,
            zcorn=zcorn,
            unit_system=UnitSystem.METRIC,
            on_inverted_cells="deactivate",
        )
    assert grid.cell_statuses.tolist() == [1, 0, 1]
    assert (grid.cell_volumes[[0, 2]] > 0.0).all()
    assert grid.cell_volumes[1] == pytest.approx(0.0)


def test_inverted_cell_can_keep_the_magnitude_of_its_volume():
    coord, zcorn = build_arrays(nx=3, ny=1, nz=1)
    zcorn[1, :, 2:4] = zcorn[0, :, 2:4] - 5.0
    with pytest.warns(UserWarning, match="1 active cell.*magnitude"):
        grid = make_corner_point_grid(
            coord=coord,
            zcorn=zcorn,
            unit_system=UnitSystem.METRIC,
            on_inverted_cells="absolute",
        )
    assert grid.cell_statuses.tolist() == [1, 1, 1]
    assert (grid.cell_volumes > 0.0).all()
    assert grid.cell_volumes[1] == pytest.approx(100.0 * 80.0 * 5.0)


def test_absolute_policy_leaves_valid_grids_untouched():
    coord, zcorn = build_arrays()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        kept = make_corner_point_grid(
            coord=coord,
            zcorn=zcorn,
            unit_system=UnitSystem.METRIC,
            on_inverted_cells="absolute",
        )
    default = make_corner_point_grid(coord=coord, zcorn=zcorn, unit_system=UnitSystem.METRIC)
    np.testing.assert_array_equal(kept.cell_volumes, default.cell_volumes)


def test_inverted_cell_policy_accepts_any_letter_case_and_rejects_unknown_values():
    assert InvertedCellPolicy("DEACTIVATE") is InvertedCellPolicy.DEACTIVATE
    assert InvertedCellPolicy("Absolute") is InvertedCellPolicy.ABSOLUTE
    with pytest.raises(ValueError):
        InvertedCellPolicy("ignore")


@pytest.mark.parametrize(
    ("name", "unit_system"),
    [
        ("cartesian", UnitSystem.METRIC),
        ("dome", UnitSystem.FIELD),
        ("snarkgrid", UnitSystem.FIELD),
        ("40X48X13_faults", UnitSystem.FIELD),
    ],
)
def test_bundled_decks_with_map_axes_load(name, unit_system):
    grid = load_grdecl(DATA_DIRECTORY / f"{name}.grdecl", unit_system=unit_system)
    active = grid.cell_statuses == 1
    assert active.any()
    assert (grid.cell_volumes[active] >= 0.0).all()


def test_deck_with_swapped_map_axes_values_reports_map_axes(tmp_path):
    deck = (DATA_DIRECTORY / "cartesian.grdecl").read_text()
    broken = deck.replace(
        "450000.000000  6299900.000000    450000.000000  6300000.000000    450100.000000  6300000.000000",
        "450000.000000  6300000.000000    450100.000000  6300000.000000    450000.000000  6299900.000000",
    )
    assert broken != deck
    broken_path = tmp_path / "broken.grdecl"
    broken_path.write_text(broken)
    with pytest.raises(GridImportError, match="MAPAXES"):
        load_grdecl(broken_path, unit_system=UnitSystem.METRIC)
