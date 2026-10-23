import warnings
from pathlib import Path

import numpy as np
import pytest

from bores.errors import CellNotFoundError, ValidationError
from bores.grids import Grid
from bores.grids.io.grdecl import load_grdecl
from bores.types import UnitSystem

DATA = Path(__file__).resolve().parents[2] / "data"
GRID_NAMES = ["cp", "dome", "pillar_36x48", "40X48X13_faults", "cartesian"]


@pytest.fixture(scope="module", params=GRID_NAMES)
def grid(request) -> Grid:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_grdecl(DATA / f"{request.param}.grdecl", unit_system=UnitSystem.FIELD)


def sample(grid, count, upper, seed=0):
    return np.random.default_rng(seed).integers(0, upper, size=count)


def split_rows(ragged):
    return [
        ragged.values[a:b] for a, b in zip(ragged.offsets[:-1], ragged.offsets[1:], strict=False)
    ]


class TestIndexConversion:
    def test_ijk_and_flat_round_trip_and_match_the_scalar_methods(self, grid):
        cells = sample(grid, 200, grid.n_cells)
        i, j, k = grid.ijk_indices(cells)
        expected = np.array([grid.ijk_index(int(c)) for c in cells])
        np.testing.assert_array_equal(np.column_stack([i, j, k]), expected)
        np.testing.assert_array_equal(grid.flat_indices(i, j, k), cells)
        np.testing.assert_array_equal(
            grid.flat_indices(i, j, k), [grid.flat_index(*map(int, ijk)) for ijk in expected]
        )

    def test_multidimensional_shapes_are_kept_and_inputs_broadcast(self, grid):
        cells = sample(grid, 24, grid.n_cells).reshape(2, 3, 4)
        i, j, k = grid.ijk_indices(cells)
        assert i.shape == j.shape == k.shape == (2, 3, 4)
        np.testing.assert_array_equal(grid.flat_indices(i, j, k), cells)
        # One k for a whole (i, j) plane
        plane_i, plane_j = np.meshgrid(np.arange(2), np.arange(3), indexing="ij")
        assert grid.flat_indices(plane_i, plane_j, 0).shape == (2, 3)

    @pytest.mark.parametrize("small", [np.int8, np.int16, np.uint8, np.uint16])
    def test_small_integer_inputs_do_not_overflow(self, grid, small):
        dims = grid.dimensions
        largest = np.iinfo(small).max
        i = np.array([min(dims.nx - 1, largest)], dtype=small)
        j = np.array([min(dims.ny - 1, largest)], dtype=small)
        k = np.array([min(dims.nz - 1, largest)], dtype=small)
        expected = int(i[0]) + int(j[0]) * dims.nx + int(k[0]) * dims.nx * dims.ny
        assert int(grid.flat_indices(i, j, k)[0]) == expected

    def test_out_of_range_indices_are_rejected(self, grid):
        for bad in (-1, grid.n_cells):
            with pytest.raises(CellNotFoundError):
                grid.ijk_indices(np.array([0, bad]))
        with pytest.raises(CellNotFoundError):
            grid.flat_indices(grid.dimensions.nx, 0, 0)
        with pytest.raises(ValidationError):
            grid.ijk_indices(np.array([0.5]))

    def test_cell_centers_match_the_scalar_method(self, grid):
        cells = sample(grid, 50, grid.n_cells)
        i, j, k = grid.ijk_indices(cells)
        centers = grid.get_cell_centers_at(i, j, k)
        assert centers.shape == (50, 3)
        expected = np.array([
            grid.get_cell_center_at(int(a), int(b), int(c))
            for a, b, c in zip(i, j, k, strict=False)
        ])
        np.testing.assert_allclose(centers, expected)


class TestFlags:
    def test_cell_flags_match_the_scalar_methods(self, grid):
        cells = sample(grid, 200, grid.n_cells)
        np.testing.assert_array_equal(
            grid.are_cells_active(cells), [grid.is_cell_active(int(c)) for c in cells]
        )
        np.testing.assert_array_equal(
            grid.are_boundary_cells(cells), [grid.is_boundary_cell(int(c)) for c in cells]
        )

    def test_boundary_cells_agree_with_the_boundary_cell_listing(self, grid):
        every_cell = np.arange(grid.n_cells)
        np.testing.assert_array_equal(
            np.flatnonzero(grid.are_boundary_cells(every_cell)), grid.get_boundary_cell_indices()
        )

    def test_face_flags_match_the_scalar_methods(self, grid):
        faces = sample(grid, 300, grid.n_faces)
        np.testing.assert_array_equal(
            grid.get_face_types(faces), [int(grid.get_face_type(int(f))) for f in faces]
        )
        np.testing.assert_array_equal(
            grid.are_boundary_faces(faces), [grid.is_boundary_face(int(f)) for f in faces]
        )
        np.testing.assert_array_equal(
            grid.are_fault_faces(faces), [grid.is_fault_face(int(f)) for f in faces]
        )

    def test_flags_keep_the_shape_of_the_input(self, grid):
        faces = sample(grid, 12, grid.n_faces).reshape(3, 4)
        assert grid.are_boundary_faces(faces).shape == (3, 4)
        assert grid.are_boundary_cells(sample(grid, 12, grid.n_cells).reshape(3, 4)).shape == (
            3,
            4,
        )

    def test_out_of_range_is_rejected(self, grid):
        with pytest.raises(CellNotFoundError):
            grid.are_cells_active(np.array([-1]))
        with pytest.raises(CellNotFoundError):
            grid.are_boundary_cells(np.array([grid.n_cells]))
        with pytest.raises(IndexError):
            grid.get_face_types(np.array([grid.n_faces]))


class TestRaggedLookups:
    def test_cell_faces_and_neighbors_match_the_scalar_methods(self, grid):
        cells = sample(grid, 100, grid.n_cells)
        for bulk, scalar in (
            (grid.get_cells_face_indices, grid.get_cell_face_indices),
            (grid.get_cells_neighbor_indices, grid.get_cell_neighbor_indices),
        ):
            rows = split_rows(bulk(cells))
            assert len(rows) == len(cells)
            for row, cell in zip(rows, cells, strict=False):
                np.testing.assert_array_equal(row, scalar(int(cell)))

    def test_face_vertices_match_the_scalar_method(self, grid):
        faces = sample(grid, 100, grid.n_faces)
        ragged = grid.get_faces_vertex_coordinates(faces)
        assert ragged.values.ndim == 2 and ragged.values.shape[1] == 3
        for row, face in zip(split_rows(ragged), faces, strict=False):
            np.testing.assert_allclose(row, grid.get_face_vertex_coordinates(int(face)))

    def test_empty_requests(self, grid):
        empty = grid.get_cells_face_indices(np.array([], dtype=int))
        assert empty.offsets.tolist() == [0] and empty.values.size == 0

    def test_face_owner_and_neighbor_cells_keep_the_shape(self, grid):
        faces = sample(grid, 12, grid.n_faces).reshape(3, 4)
        result = grid.get_faces_cell_indices(faces)
        assert result.shape == (3, 4, 2)
        np.testing.assert_array_equal(result[1, 2], grid.get_face_cell_indices(int(faces[1, 2])))


class TestFaceNormals:
    def test_match_the_scalar_method_for_owners_and_neighbours(self, grid):
        faces = sample(grid, 300, grid.n_faces)
        owners = grid.face_cell_indices[faces]
        for side in (0, 1):
            keep = owners[:, side] >= 0
            f, c = faces[keep], owners[keep, side]
            normals = grid.get_face_normals_for_cells(f, c)
            expected = np.array([
                grid.get_face_normal_for_cell(int(a), int(b)) for a, b in zip(f, c, strict=False)
            ])
            np.testing.assert_allclose(normals, expected)

    def test_broadcasts_to_a_multidimensional_result(self, grid):
        face = int(np.flatnonzero(grid.face_cell_indices[:, 1] >= 0)[0])
        cells = grid.face_cell_indices[face]
        normals = grid.get_face_normals_for_cells(face, cells.reshape(1, 2))
        assert normals.shape == (1, 2, 3)
        np.testing.assert_allclose(normals[0, 0], -normals[0, 1])

    def test_rejects_unconnected_and_invalid_cells(self, grid):
        face = int(np.flatnonzero(grid.face_cell_indices[:, 1] >= 0)[0])
        unconnected = int(np.setdiff1d(np.arange(grid.n_cells), grid.face_cell_indices[face])[0])
        with pytest.raises(ValidationError, match="not connected"):
            grid.get_face_normals_for_cells(
                np.array([face, face]), np.array([grid.face_cell_indices[face, 0], unconnected])
            )
        with pytest.raises(ValidationError):
            grid.get_face_normals_for_cells(face, -1)
        with pytest.raises(IndexError):
            grid.get_face_normals_for_cells(grid.n_faces, 0)


class TestSpatialQueries:
    def test_nearest_cells_match_the_scalar_method(self, grid):
        rng = np.random.default_rng(1)
        low, high = grid.cell_centroids.min(axis=0), grid.cell_centroids.max(axis=0)
        points = rng.uniform(low, high, size=(150, 3))
        bulk = grid.find_nearest_cells(points)
        assert bulk.shape == (150,)
        np.testing.assert_array_equal(bulk, [grid.find_nearest_cell(*p) for p in points])
        i, j, k = grid.find_cells_at_positions(points)
        np.testing.assert_array_equal(
            np.column_stack([i, j, k]), [grid.find_cell_at_position(*p) for p in points]
        )

    def test_multidimensional_points(self, grid):
        points = grid.cell_centroids[:12].reshape(3, 4, 3)
        np.testing.assert_array_equal(grid.find_nearest_cells(points), np.arange(12).reshape(3, 4))

    def test_max_distance_is_enforced(self, grid):
        far = grid.cell_centroids.max(axis=0) + 1e6
        with pytest.raises(ValidationError, match="no cell center"):
            grid.find_nearest_cells(np.array([far, grid.cell_centroids[0]]), max_distance=1.0)
        with pytest.raises(ValidationError):
            grid.find_nearest_cells(np.array([1.0, 2.0]))

    def test_cells_in_radius_match_the_scalar_method(self, grid):
        points = grid.cell_centroids[:20]
        radius = float(np.linalg.norm(np.ptp(grid.cell_centroids, axis=0))) / 10
        ragged = grid.find_cells_in_radii(points, radius)
        for row, point in zip(split_rows(ragged), points, strict=False):
            np.testing.assert_array_equal(row, np.sort(grid.find_cells_in_radius(*point, radius)))
        # One radius per point
        varied = grid.find_cells_in_radii(points, np.linspace(0.0, radius, 20))
        assert len(varied.offsets) == 21
        assert (
            len(split_rows(varied)[0]) <= 1
        )  # radius 0 finds at most the cell centred on the point
