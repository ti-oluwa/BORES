import numpy as np
import pytest

from bores.errors import ValidationError
from bores.grids.base import ConnectionType
from bores.grids.factories.cartesian import make_cartesian_grid
from bores.reservoir.faults import (
    Fault,
    apply_faults,
    build_face_lookup,
    remove_faults,
    resolve_fault_cell_pairs,
)

NX, NY, NZ = 8, 6, 4


@pytest.fixture(scope="module")
def grid():
    return make_cartesian_grid(nx=NX, ny=NY, nz=NZ, dx=10.0, dy=10.0, dz=5.0)


def flat(i, j, k):
    return (i - 1) + (j - 1) * NX + (k - 1) * NX * NY


def test_pairs_follow_the_box_order_and_put_the_lower_cell_first():
    fault = Fault(name="A", i1=2, i2=3, j1=1, j2=2, k1=1, k2=2, face_direction="X-")
    pairs = resolve_fault_cell_pairs(fault, NX, NY, NZ)
    expected = [
        tuple(sorted((flat(i, j, k), flat(i - 1, j, k))))
        for k in (1, 2)
        for j in (1, 2)
        for i in (2, 3)
    ]
    assert list(zip(pairs.lower.tolist(), pairs.upper.tolist(), strict=True)) == expected


@pytest.mark.parametrize("direction", ["X", "Y", "Z"])
def test_pairs_leaving_the_grid_are_skipped_with_a_warning(direction):
    fault = Fault(name="edge", i1=NX, i2=NX, j1=NY, j2=NY, k1=NZ, k2=NZ, face_direction=direction)
    with pytest.warns(UserWarning, match="outside grid bounds"):
        pairs = resolve_fault_cell_pairs(fault, NX, NY, NZ)
    assert pairs.lower.shape == (0,)


def test_apply_classifies_the_shared_faces_and_stores_the_multiplier(grid):
    fault = Fault(
        name="A",
        i1=3,
        i2=3,
        j1=1,
        j2=NY,
        k1=1,
        k2=NZ,
        face_direction="X",
        transmissibility_multiplier=0.1,
    )
    faulted = apply_faults(grid, fault)
    faces = faulted.fault_face_indices["A"]
    assert len(faces) == NY * NZ
    assert (faulted.face_connection_types[faces] == int(ConnectionType.INTERIOR_FAULT_FACE)).all()
    assert faulted.fault_transmissibility_multipliers == {"A": 0.1}
    changed = np.flatnonzero(faulted.face_connection_types != grid.face_connection_types)
    assert np.array_equal(np.sort(changed), np.sort(faces))
    for face in faces:
        owner, neighbour = grid.face_cell_indices[face]
        assert {int(owner), int(neighbour)} <= {
            flat(3, j, k) for j in range(1, NY + 1) for k in range(1, NZ + 1)
        } | {flat(4, j, k) for j in range(1, NY + 1) for k in range(1, NZ + 1)}
    assert grid.fault_face_indices is None or "A" not in grid.fault_face_indices


def test_reapplying_a_name_replaces_the_fault_and_drops_the_old_multiplier(grid):
    first = apply_faults(
        grid,
        Fault(
            name="A",
            i1=3,
            i2=3,
            j1=1,
            j2=2,
            k1=1,
            k2=1,
            face_direction="X",
            transmissibility_multiplier=0.5,
        ),
    )
    second = apply_faults(
        first, Fault(name="A", i1=5, i2=5, j1=1, j2=3, k1=1, k2=1, face_direction="X")
    )
    assert len(second.fault_face_indices["A"]) == 3
    assert not second.fault_transmissibility_multipliers
    assert (second.face_connection_types == int(ConnectionType.INTERIOR_FAULT_FACE)).sum() == 3


def test_remove_restores_the_original_classification(grid):
    faults = [
        Fault(name="A", i1=3, i2=3, j1=1, j2=NY, k1=1, k2=NZ, face_direction="X"),
        Fault(
            name="B",
            i1=1,
            i2=NX,
            j1=2,
            j2=2,
            k1=1,
            k2=NZ,
            face_direction="Y",
            transmissibility_multiplier=2.0,
        ),
    ]
    faulted = apply_faults(grid, *faults)
    only_b = remove_faults(faulted, "A")
    assert list(only_b.fault_face_indices) == ["B"]
    assert only_b.fault_transmissibility_multipliers == {"B": 2.0}
    cleaned = remove_faults(faulted)
    assert np.array_equal(cleaned.face_connection_types, grid.face_connection_types)
    assert cleaned.fault_face_indices is None
    with pytest.raises(ValidationError):
        remove_faults(faulted, "missing")


def test_face_lookup_is_sorted_and_only_has_interior_faces(grid):
    lookup = build_face_lookup(grid)
    assert (np.diff(lookup.keys) >= 0).all()
    cells = grid.face_cell_indices[lookup.faces]
    assert (cells >= 0).all()
    assert len(lookup.faces) == int(((grid.face_cell_indices >= 0).all(axis=1)).sum())
