import numpy as np

from bores.grids.base import build_cell_face_csr, build_cell_neighbor_csr


def reference_csr(face_cell_indices, n_cells):
    cells = face_cell_indices.reshape(-1).astype(np.int64)
    faces = np.repeat(np.arange(len(face_cell_indices), dtype=np.int64), 2)
    present = cells >= 0
    cells, faces = cells[present], faces[present]
    order = np.argsort(cells, kind="stable")
    face_offsets = np.concatenate([[0], np.cumsum(np.bincount(cells, minlength=n_cells))])

    interior = (face_cell_indices[:, 0] >= 0) & (face_cell_indices[:, 1] >= 0)
    owners = face_cell_indices[interior, 0].astype(np.int64)
    neighbours = face_cell_indices[interior, 1].astype(np.int64)
    keys = np.unique(
        np.concatenate([owners, neighbours]) * n_cells + np.concatenate([neighbours, owners])
    )
    neighbour_offsets = np.concatenate([
        [0],
        np.cumsum(np.bincount(keys // n_cells, minlength=n_cells)),
    ])
    return (
        faces[order].astype(np.int32),
        face_offsets.astype(np.int32),
        (keys % n_cells).astype(np.int32),
        neighbour_offsets.astype(np.int32),
    )


def test_csr_builders_match_the_numpy_reference_with_duplicate_and_boundary_faces():
    rng = np.random.default_rng(3)
    n_cells = 200
    pairs = rng.integers(0, n_cells, size=(1500, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    pairs = np.concatenate([pairs, pairs[:300]])  # duplicate connections
    boundary = np.column_stack([rng.integers(0, n_cells, 100), np.full(100, -1)])
    face_cell_indices = np.concatenate([pairs, boundary]).astype(np.int32)
    rng.shuffle(face_cell_indices)

    expected = reference_csr(face_cell_indices, n_cells)
    face_indices, face_offsets = build_cell_face_csr(
        face_cell_indices=face_cell_indices, n_cells=n_cells
    )
    neighbour_indices, neighbour_offsets = build_cell_neighbor_csr(
        face_cell_indices=face_cell_indices, n_cells=n_cells
    )

    for result, reference in zip(
        (face_indices, face_offsets, neighbour_indices, neighbour_offsets), expected, strict=True
    ):
        assert result.dtype == reference.dtype
        assert np.array_equal(result, reference)


def test_csr_builders_handle_cells_without_faces():
    face_cell_indices = np.array([[0, 2]], dtype=np.int32)
    face_indices, face_offsets = build_cell_face_csr(
        face_cell_indices=face_cell_indices, n_cells=5
    )
    neighbour_indices, neighbour_offsets = build_cell_neighbor_csr(
        face_cell_indices=face_cell_indices, n_cells=5
    )
    assert face_offsets.tolist() == [0, 1, 1, 2, 2, 2]
    assert neighbour_indices.tolist() == [2, 0]
    assert neighbour_offsets.tolist() == [0, 1, 1, 2, 2, 2]
    assert face_indices.tolist() == [0, 0]


def test_per_cell_geometry_kernels_match_the_face_scatter_kernels():
    from bores.grids.base import (
        compute_cell_bounding_boxes,
        compute_cell_bounding_boxes_by_cell,
        compute_cell_volumes_and_centroids,
        compute_cell_volumes_and_centroids_by_cell,
    )
    from bores.grids.factories.voronoi import make_voronoi_grid

    seeds = np.random.default_rng(2).random((60, 2)) * 500.0
    grid = make_voronoi_grid(seeds, z_top=100.0, layer_thicknesses=np.array([4.0, 6.0]))
    arguments = (
        grid.face_cell_indices,
        grid.face_vertex_indices,
        grid.face_vertex_offsets,
        grid.vertex_coordinates,
        grid.n_cells,
    )
    volumes, centroids = compute_cell_volumes_and_centroids(*arguments)
    by_cell_volumes, by_cell_centroids = compute_cell_volumes_and_centroids_by_cell(
        grid.cell_face_indices,
        grid.cell_face_offsets,
        grid.face_cell_indices,
        grid.face_vertex_indices,
        grid.face_vertex_offsets,
        grid.vertex_coordinates,
        grid.n_cells,
    )
    assert np.array_equal(volumes, by_cell_volumes)
    assert np.array_equal(centroids, by_cell_centroids)

    low, high = compute_cell_bounding_boxes(*arguments)
    by_cell_low, by_cell_high = compute_cell_bounding_boxes_by_cell(
        grid.cell_face_indices,
        grid.cell_face_offsets,
        grid.face_vertex_indices,
        grid.face_vertex_offsets,
        grid.vertex_coordinates,
        grid.n_cells,
    )
    assert np.array_equal(low, by_cell_low)
    assert np.array_equal(high, by_cell_high)


def test_face_deduplication_numbers_faces_by_first_listing_and_rejects_a_third_cell():
    import pytest

    from bores.errors import InvalidFaceConnectivityError
    from bores.grids.factories.base import build_csr_face_arrays

    vertices = np.zeros((6, 3))
    _, face_vertices, offsets, cells = build_csr_face_arrays(
        vertices, [[[0, 1, 2], [3, 4, 5]], [[2, 1, 0]], []]
    )
    assert face_vertices.tolist() == [0, 1, 2, 3, 4, 5]  # the owner's winding is kept
    assert offsets.tolist() == [0, 3, 6]
    assert cells.tolist() == [[0, 1], [0, -1]]

    with pytest.raises(
        InvalidFaceConnectivityError, match="owner 0 and neighbour 1; tried to add cell 2"
    ):
        build_csr_face_arrays(vertices, [[[0, 1, 2]], [[2, 1, 0]], [[1, 0, 2]]])
