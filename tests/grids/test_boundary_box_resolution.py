"""`resolve_boundary_faces_for_box` against a plain one-cell-at-a-time reference."""

import pathlib
import warnings

import numpy as np
import pytest

from bores.grids.factories.cartesian import make_cartesian_grid
from bores.grids.io.grdecl import load_grdecl
from bores.grids.utils import FACE_DIRECTION_AXIS_SIGN, resolve_boundary_faces_for_box
from bores.types import UnitSystem

DATA = pathlib.Path(__file__).parents[2] / "data"


def reference(grid, *, i1, i2, j1, j2, k1, k2, face_direction, label):
    """Resolves the box one cell at a time, using the scalar grid methods."""
    axis, sign = FACE_DIRECTION_AXIS_SIGN[face_direction]
    nx, ny, nz = grid.dimensions
    position_of_face = {int(face): n for n, face in enumerate(grid.boundary_face_indices)}
    positions: set[int] = set()
    messages: list[str] = []
    for k in range(k1, k2 + 1):
        for j in range(j1, j2 + 1):
            for i in range(i1, i2 + 1):
                cell_label = f"{label}: cell ({i},{j},{k})"
                if not (1 <= i <= nx and 1 <= j <= ny and 1 <= k <= nz):
                    messages.append(
                        f"{cell_label} is outside grid bounds ({nx}x{ny}x{nz}). Skipping."
                    )
                    continue
                cell = int(grid.flat_index(i - 1, j - 1, k - 1))
                if not grid.is_cell_active(cell):
                    messages.append(f"{cell_label} is inactive. Skipping connection.")
                    continue
                matched = None
                for face in grid.get_cell_face_indices(cell):
                    normal = grid.get_face_normal_for_cell(int(face), cell)
                    dominant = int(np.argmax(np.abs(normal)))
                    if dominant == axis and np.sign(normal[dominant]) == sign:
                        matched = int(face)
                        break
                if matched is None:
                    messages.append(f"{cell_label} has no {face_direction!r} face. Skipping.")
                elif matched not in position_of_face:
                    messages.append(
                        f"{cell_label}'s {face_direction!r} face is not a grid-boundary face. "
                        "It has a real interior neighbour, active or not. Boundary conditions "
                        "can only attach to genuine grid-edge faces. Skipping."
                    )
                else:
                    positions.add(position_of_face[matched])
    return np.asarray(sorted(positions), dtype=np.int32), messages


def resolve(function, grid, **box):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = function(grid, **box)
    return result, [str(item.message) for item in caught]


def boxes(nx, ny, nz):
    return [
        (1, nx, 1, ny, 1, nz),
        (1, 1, 1, ny, 1, nz),
        (nx, nx, 1, ny, 1, nz),
        (1, nx, 1, 1, 1, nz),
        (1, nx, ny, ny, 1, nz),
        (1, nx, 1, ny, 1, 1),
        (1, nx, 1, ny, nz, nz),
        (0, nx + 2, 0, ny + 2, 0, nz + 2),
        (3, 2, 1, 1, 1, 1),
        (2, max(2, nx - 1), 2, max(2, ny - 1), 1, nz),
    ]


GRIDS = {
    "cartesian": lambda: make_cartesian_grid(nx=7, ny=5, nz=4, dx=10.0, dy=10.0, dz=5.0),
    "inactive cells": lambda: load_grdecl(DATA / "cartesian.grdecl", unit_system=UnitSystem.FIELD),
    "corner point": lambda: load_grdecl(DATA / "cp.grdecl", unit_system=UnitSystem.FIELD),
    "faulted": lambda: load_grdecl(
        DATA / "Simple10x10x3_FlatCross_Fault.grdecl", unit_system=UnitSystem.FIELD
    ),
}


@pytest.mark.parametrize("name", sorted(GRIDS))
def test_matches_the_one_cell_at_a_time_reference(name):
    grid = GRIDS[name]()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for direction in FACE_DIRECTION_AXIS_SIGN:
            for i1, i2, j1, j2, k1, k2 in boxes(*grid.dimensions):
                box = {
                    "i1": i1, "i2": i2, "j1": j1, "j2": j2, "k1": k1, "k2": k2,
                    "face_direction": direction, "label": "REC",
                }  # fmt: skip
                expected, expected_messages = reference(grid, **box)
                found, messages = resolve(resolve_boundary_faces_for_box, grid, **box)
                assert np.array_equal(found, expected), (name, box)
                assert found.dtype == expected.dtype
                assert messages == expected_messages, (name, box)
