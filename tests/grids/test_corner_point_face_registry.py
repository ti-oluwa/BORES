import warnings

import numpy as np

from bores.grids.factories.corner_point import (
    FACE_CORNER_TABLE,
    LATERAL_FACE_CORNERS,
    MINIMUM_RELATIVE_OVERLAP_AREA,
    VTK_TO_CORNER_ORDER,
    compute_cell_corner_coordinates,
    connect_unmatched_faces_kernel,
    make_corner_point_grid,
    register_cell_faces,
)

warnings.simplefilter("ignore")


def faulted_pillar_grid(nx=3, ny=2, nz=3, fault_throw=0.5):
    """Vertical-pillar grid whose columns with `i >= 1` are shifted down by `fault_throw`."""
    coord = np.zeros((ny + 1, nx + 1, 6))
    for j in range(ny + 1):
        for i in range(nx + 1):
            coord[j, i] = [float(i), float(j), -10.0, float(i), float(j), 10.0]
    zcorn = np.zeros((2 * nz, 2 * ny, 2 * nx))
    for k in range(nz):
        zcorn[2 * k] = float(k)
        zcorn[2 * k + 1] = float(k + 1)
    zcorn[:, :, 2:] += fault_throw
    return coord, zcorn


def merged_corners(coord, zcorn):
    nz, ny, nx = zcorn.shape[0] // 2, zcorn.shape[1] // 2, zcorn.shape[2] // 2
    cells = np.argwhere(np.ones((nz, ny, nx), dtype=bool)).astype(np.int32)
    corners = compute_cell_corner_coordinates(cells=cells, coord=coord, zcorn=zcorn)
    flat = corners.reshape(-1, 3)
    _, first, inverse = np.unique(
        np.round(flat / 1e-8).astype(np.int64), axis=0, return_index=True, return_inverse=True
    )
    return corners, flat[first], inverse.reshape(-1, 8), (nx, ny, nz)


def test_registry_matches_shared_faces_of_a_regular_grid():
    coord, zcorn = faulted_pillar_grid(fault_throw=0.0)
    _corners, vertices, corner_global, (nx, ny, nz) = merged_corners(coord, zcorn)
    n_cells = nx * ny * nz

    n_faces, _, owners, neighbours, _, degenerate, _, pinched, ignored = register_cell_faces(
        corner_global=corner_global,
        active_cells=np.arange(n_cells),
        vertex_coordinates=vertices,
        pinch_tolerance=0.0,
        mirrored=False,
        face_corner_table=FACE_CORNER_TABLE,
        vtk_to_corner=VTK_TO_CORNER_ORDER,
    )
    interior = (nx - 1) * ny * nz + nx * (ny - 1) * nz + nx * ny * (nz - 1)
    boundary = 2 * (ny * nz + nx * nz + nx * ny)
    assert n_faces == interior + boundary
    assert int((neighbours >= 0).sum()) == interior
    assert len(degenerate) == 0 and not pinched.any() and ignored == 0
    assert len(owners) == n_faces


def test_connect_kernel_reports_overflow_so_the_caller_can_retry():
    coord, zcorn = faulted_pillar_grid()
    corners, vertices, corner_global, (nx, ny, nz) = merged_corners(coord, zcorn)
    n_cells = nx * ny * nz
    n_faces, _, owners, neighbours, locals_, degenerate, degenerate_locals, pinched, _ = (
        register_cell_faces(
            corner_global=corner_global,
            active_cells=np.arange(n_cells),
            vertex_coordinates=vertices,
            pinch_tolerance=0.0,
            mirrored=False,
            face_corner_table=FACE_CORNER_TABLE,
            vtk_to_corner=VTK_TO_CORNER_ORDER,
        )
    )
    arguments = {
        "registry_owners": owners,
        "registry_neighbours": neighbours,
        "registry_local_indices": locals_,
        "n_registered": n_faces,
        "degenerate_cells": degenerate,
        "degenerate_locals": degenerate_locals,
        "pinched": pinched,
        "active_mask": np.ones(n_cells, dtype=bool),
        "corner_coordinates": corners,
        "cell_centers": corners.mean(axis=1),
        "coord": coord,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "lateral_face_corners": LATERAL_FACE_CORNERS,
        "minimum_relative_overlap_area": MINIMUM_RELATIVE_OVERLAP_AREA,
    }
    assert connect_unmatched_faces_kernel(**arguments, max_new_faces=1, max_new_points=6)[0] == 1
    status, n_new, *_ = connect_unmatched_faces_kernel(
        **arguments, max_new_faces=500, max_new_points=3000
    )
    assert status == 0 and n_new > 0


def test_faulted_grid_connects_each_column_pair_across_the_throw():
    coord, zcorn = faulted_pillar_grid(nx=2, ny=1, nz=3, fault_throw=0.5)
    grid = make_corner_point_grid(coord=coord, zcorn=zcorn, metadata={"dimensions": (2, 1, 3)})
    cross = {}
    for face in range(grid.n_faces):
        owner, neighbour = (int(c) for c in grid.face_cell_indices[face])
        if owner >= 0 and neighbour >= 0 and abs(grid.face_unit_normals[face, 0]) > 0.9:
            key = tuple(sorted((owner, neighbour)))
            cross[key] = cross.get(key, 0.0) + float(grid.face_areas[face])
    # Left cell k (flat 2k) overlaps right cells k-1 and k (flat 2k - 1 and 2k + 1) by half
    # a cell each, because the right column is thrown down by half a cell.
    for pair in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5)):
        assert abs(cross[pair] - 0.5) < 1e-12
