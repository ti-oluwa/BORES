import types
import warnings
from pathlib import Path

import numpy as np
import pytest

from bores.errors import ValidationError
from bores.grids import Grid
from bores.grids.io.grdecl import load_grdecl
from bores.types import UnitSystem

DATA = Path(__file__).resolve().parents[2] / "data"


def reference_top_bottom(grid, cell_index):
    """One cell at a time, straight from the definition."""
    faces = grid.cell_face_indices[
        grid.cell_face_offsets[cell_index] : grid.cell_face_offsets[cell_index + 1]
    ]
    candidates = [
        (grid.face_centroids[f, 2], grid.face_areas[f])
        for f in faces
        if abs(grid.face_unit_normals[f, 2]) >= np.cos(np.deg2rad(45.0))
    ]
    if len(candidates) < 2:
        return grid.cell_min_xyz[cell_index, 2], grid.cell_max_xyz[cell_index, 2], 1.0, 1.0
    depths = np.array([c[0] for c in candidates])
    areas = np.array([c[1] for c in candidates])
    top = depths <= 0.5 * (depths.min() + depths.max())
    bottom = ~top
    if not bottom.any():
        bottom = top
    return (
        np.average(depths[top], weights=areas[top]),
        np.average(depths[bottom], weights=areas[bottom]),
        areas[top].sum(),
        areas[bottom].sum(),
    )


@pytest.mark.parametrize("name", ["cp", "dome", "pillar_36x48", "40X48X13_faults", "cartesian"])
def test_matches_the_per_cell_definition_on_real_grids(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid = load_grdecl(DATA / f"{name}.grdecl", unit_system=UnitSystem.FIELD)
    cells = np.random.default_rng(0).permutation(grid.n_cells)[:300]  # arbitrary order
    result = grid.get_cell_top_bottom_faces(cell_indices=cells)
    expected = np.array([reference_top_bottom(grid, int(c)) for c in cells])
    np.testing.assert_allclose(np.column_stack(result), expected, rtol=1e-9)


def test_defaults_to_every_cell():
    grid = load_grdecl(DATA / "cp.grdecl", unit_system=UnitSystem.FIELD)
    everything = grid.get_cell_top_bottom_faces()
    assert everything.top_depth.shape == (grid.n_cells,)
    assert np.all(everything.bottom_depth >= everything.top_depth - 1e-9)


def make_mock_grid():
    # Cell 0 has a top and a bottom, cell 1 has two faces at one depth (zero thickness),
    # cell 2 has no faces, cell 3 has a tilted-enough top and bottom plus a side face
    face_cells = np.array([[0, -1], [0, 1], [0, -1], [1, -1], [1, -1], [3, -1], [3, -1], [3, -1]])
    normal_z = np.array([-1.0, 1.0, 0.0, -1.0, 1.0, 0.99, -0.99, 0.0])
    grid = types.SimpleNamespace(
        face_cell_indices=face_cells,
        face_unit_normals=np.stack([np.zeros(8), np.zeros(8), normal_z], axis=1),
        face_centroids=np.stack(
            [
                np.zeros(8),
                np.zeros(8),
                np.array([100.0, 110.0, 105.0, 200.0, 200.0, 300.0, 310.0, 305.0]),
            ],
            axis=1,
        ),
        face_areas=np.array([4.0, 5.0, 2.0, 3.0, 3.0, 1.0, 1.0, 9.0]),
        cell_face_indices=np.arange(8),
        cell_face_offsets=np.array([0, 3, 5, 5, 8]),
        cell_min_xyz=np.array([[0, 0, 95.0]] * 4),
        cell_max_xyz=np.array([[0, 0, 115.0]] * 4),
        n_cells=4,
    )
    return grid


def test_degenerate_cells():
    grid = make_mock_grid()
    result = Grid.get_cell_top_bottom_faces(grid, cell_indices=np.array([1, 2]))
    # Zero thickness: the same depth and area at both ends instead of an error
    assert result.top_depth[0] == result.bottom_depth[0] == pytest.approx(200.0)
    assert result.top_area[0] == result.bottom_area[0] == pytest.approx(6.0)
    # No faces: bounding box range and unit areas
    assert (result.top_depth[1], result.bottom_depth[1]) == (95.0, 115.0)
    assert (result.top_area[1], result.bottom_area[1]) == (1.0, 1.0)


def test_repeated_cells_and_order_are_kept():
    grid = make_mock_grid()
    result = Grid.get_cell_top_bottom_faces(grid, cell_indices=np.array([3, 0, 3]))
    assert result.top_depth[0] == pytest.approx(300.0) == pytest.approx(result.top_depth[2])
    assert result.top_depth[1] == pytest.approx(100.0)
    assert result.bottom_depth[1] == pytest.approx(110.0)


def test_a_face_not_connected_to_its_cell_is_rejected():
    grid = make_mock_grid()
    grid.face_cell_indices = grid.face_cell_indices.copy()
    grid.face_cell_indices[0] = [9, -1]
    with pytest.raises(ValidationError):
        Grid.get_cell_top_bottom_faces(grid, cell_indices=np.array([0]))
