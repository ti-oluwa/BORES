"""Converting a grid's units leaves every length-based array consistent with its vertices."""

import pathlib
import warnings

import numpy as np
import pytest

from bores.grids.base import compute_cell_bounding_boxes
from bores.grids.factories.cartesian import make_cartesian_grid
from bores.grids.io.grdecl import load_grdecl
from bores.types import UnitSystem

DATA = pathlib.Path(__file__).parents[2] / "data"


def assert_lengths_follow_vertices(grid):
    low, high = compute_cell_bounding_boxes(
        face_cell_indices=grid.face_cell_indices,
        face_vertex_indices=grid.face_vertex_indices,
        face_vertex_offsets=grid.face_vertex_offsets,
        vertex_coordinates=grid.vertex_coordinates,
        n_cells=grid.n_cells,
    )
    has_faces = np.isfinite(low).all(axis=1)
    assert np.allclose(grid.cell_min_xyz[has_faces], low[has_faces])
    assert np.allclose(grid.cell_max_xyz[has_faces], high[has_faces])
    assert np.allclose(grid.cell_length_x[has_faces], (high - low)[has_faces, 0])
    assert np.allclose(grid.cell_length_y[has_faces], (high - low)[has_faces, 1])
    assert np.allclose(grid.cell_length_z[has_faces], (high - low)[has_faces, 2])
    assert grid.bounding_box[1] == pytest.approx(high[has_faces, 0].max())


def test_a_converted_cartesian_grid_has_the_converted_cell_sizes():
    metric = make_cartesian_grid(
        nx=4, ny=3, nz=2, dx=100.0, dy=50.0, dz=10.0, unit_system=UnitSystem.METRIC
    )
    field = metric.convert(UnitSystem.FIELD)
    assert np.allclose(field.cell_length_x, 100.0 / 0.3048)
    assert np.allclose(field.cell_length_y, 50.0 / 0.3048)
    assert np.allclose(field.cell_length_z, 10.0 / 0.3048)
    assert_lengths_follow_vertices(field)
    assert_lengths_follow_vertices(field.convert(UnitSystem.METRIC))


@pytest.mark.parametrize("name", ["dome", "Johansen", "40X48X13_faults"])
def test_a_loaded_corner_point_grid_is_consistent(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid = load_grdecl(DATA / f"{name}.grdecl", unit_system=UnitSystem.FIELD)
    assert_lengths_follow_vertices(grid)
