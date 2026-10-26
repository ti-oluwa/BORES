"""Point-in-cell tests follow each cell's own faces, however warped the cell is."""

import pathlib
import warnings

import numpy as np
import pytest

from bores.grids.containment import cell_contains_point, get_first_cell_containing_point
from bores.grids.factories.cartesian import make_cartesian_grid
from bores.grids.io.grdecl import load_grdecl
from bores.types import UnitSystem

DATA = pathlib.Path(__file__).parents[2] / "data"
WARPED_GRIDS = ["dome", "snarkgrid", "40X48X13_faults", "Johansen"]


def load(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_grdecl(DATA / f"{name}.grdecl", unit_system=UnitSystem.FIELD)


def compute_solid_angle_fraction(grid, cell, point):
    """Independent reference: the solid angle the cell's faces subtend at the point."""
    total = 0.0
    for face in grid.get_cell_face_indices(cell):
        vertices = grid.get_face_vertex_coordinates(int(face)) - np.asarray(point)
        outward = grid.get_face_normal_for_cell(int(face), cell)
        area = sum(
            np.cross(vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices))
        )
        orientation = 1.0 if np.dot(area, outward) >= 0.0 else -1.0
        a = vertices[0]
        for i in range(1, len(vertices) - 1):
            b, c = vertices[i], vertices[i + 1]
            la, lb, lc = np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(c)
            denominator = la * lb * lc + np.dot(a, b) * lc + np.dot(a, c) * lb + np.dot(b, c) * la
            total += orientation * 2.0 * np.arctan2(np.dot(a, np.cross(b, c)), denominator)
    return total / (4.0 * np.pi)


@pytest.fixture(scope="module")
def box():
    return make_cartesian_grid(nx=3, ny=3, nz=3, dx=10.0, dy=20.0, dz=5.0)


def test_a_box_cell_contains_exactly_its_own_volume(box):
    cell = int(box.flat_index(1, 1, 1))
    low, high = box.cell_min_xyz[cell], box.cell_max_xyz[cell]
    centre = (low + high) / 2
    assert cell_contains_point(box, cell, tuple(centre))
    assert cell_contains_point(box, cell, tuple(low + 1e-3))
    assert not cell_contains_point(box, cell, tuple(centre + np.array([0.0, 0.0, 10.0])))
    assert not cell_contains_point(box, cell, tuple(high + 0.5))
    for axis in range(3):
        outside = centre.copy()
        outside[axis] = high[axis] + 0.01
        assert not cell_contains_point(box, cell, tuple(outside))


def test_points_on_a_face_or_within_the_tolerance_count_as_inside(box):
    cell = int(box.flat_index(1, 1, 1))
    low, high = box.cell_min_xyz[cell], box.cell_max_xyz[cell]
    on_face = (low + high) / 2
    on_face[0] = high[0]
    assert cell_contains_point(box, cell, tuple(on_face))
    just_outside = on_face + np.array([1e-4, 0.0, 0.0])
    assert not cell_contains_point(box, cell, tuple(just_outside))
    assert cell_contains_point(box, cell, tuple(just_outside), tolerance=1e-3)
    assert not cell_contains_point(
        box, cell, tuple(on_face + np.array([0.5, 0.0, 0.0])), tolerance=1e-3
    )


@pytest.mark.parametrize("name", WARPED_GRIDS)
def test_agrees_with_the_independent_reference_on_warped_cells(name):
    grid = load(name)
    rng = np.random.default_rng(0)
    with_faces = np.flatnonzero(np.diff(grid.cell_face_offsets) > 0)
    inside = outside = 0
    for cell in rng.choice(with_faces, min(300, len(with_faces)), replace=False):
        low, high = grid.cell_min_xyz[cell], grid.cell_max_xyz[cell]
        for _ in range(10):
            point = low + rng.random(3) * (high - low)
            fraction = compute_solid_angle_fraction(grid, int(cell), point)
            if abs(fraction - 0.5) < 0.01:
                continue  # on the surface, where either answer is acceptable
            expected = fraction > 0.5
            assert cell_contains_point(grid, int(cell), tuple(point)) == expected, (name, cell)
            inside += expected
            outside += not expected
    assert inside > 50 and outside > 50


@pytest.mark.parametrize("name", ["dome", "Johansen"])
def test_a_cell_without_faces_contains_nothing(name):
    grid = load(name)
    for cell in np.flatnonzero(np.diff(grid.cell_face_offsets) == 0)[:20]:
        assert not cell_contains_point(grid, int(cell), tuple(grid.cell_centroids[cell]))


def test_the_first_containing_candidate_in_order_wins(box):
    shared_face = (
        int(box.interior_face_indices[0]) if hasattr(box, "interior_face_indices") else None
    )
    cells = [int(box.flat_index(1, 1, 1)), int(box.flat_index(2, 1, 1))]
    point = tuple((box.cell_max_xyz[cells[0]] + box.cell_min_xyz[cells[0]]) / 2)
    assert get_first_cell_containing_point(box, np.array(cells, dtype=np.int32), point) == cells[0]
    assert shared_face is None or shared_face >= 0
    # the point also lies on the shared face of cell[0] and the next cell along x
    on_shared = list(point)
    on_shared[0] = box.cell_max_xyz[cells[0]][0]
    for order in (cells, cells[::-1]):
        found = get_first_cell_containing_point(
            box, np.array(order, dtype=np.int32), tuple(on_shared)
        )
        assert found == order[0]


def test_no_candidate_containing_the_point_gives_none(box):
    candidates = np.arange(box.n_cells, dtype=np.int32)
    assert get_first_cell_containing_point(box, candidates, (1e6, 1e6, 1e6)) is None
    assert get_first_cell_containing_point(box, candidates[:0], (5.0, 5.0, 2.0)) is None


def test_candidates_may_be_any_integer_array(box):
    point = tuple((box.cell_min_xyz[7] + box.cell_max_xyz[7]) / 2)
    candidates = np.arange(box.n_cells, dtype=np.int64)
    assert get_first_cell_containing_point(box, candidates[::-1], point) == 7
    assert get_first_cell_containing_point(box, np.asarray(candidates[::2]), point) in (None, 7)


def brute_force_triangle_distance_squared(point, a, b, c, samples=400):
    """Distance to a triangle found by testing a fine lattice of points on it."""
    u = np.linspace(0.0, 1.0, samples)
    uu, vv = np.meshgrid(u, u)
    keep = (uu + vv) <= 1.0
    lattice = a + np.outer(uu[keep], b - a) + np.outer(vv[keep], c - a)
    return float(np.min(np.sum((lattice - point) ** 2, axis=1)))


def test_triangle_distance_matches_a_brute_force_search():
    from bores.grids.containment import compute_triangle_distance_squared

    rng = np.random.default_rng(3)
    for _ in range(200):
        a, b, c = rng.normal(size=(3, 3)) * 5.0
        point = rng.normal(size=3) * 8.0
        exact = compute_triangle_distance_squared(
            px=point[0], py=point[1], pz=point[2],
            ax=a[0], ay=a[1], az=a[2], bx=b[0], by=b[1], bz=b[2], cx=c[0], cy=c[1], cz=c[2],
        )  # fmt: skip
        approximate = brute_force_triangle_distance_squared(point, a, b, c)
        assert exact <= approximate + 1e-9
        assert exact == pytest.approx(approximate, abs=0.05)


def test_triangle_distance_copes_with_a_triangle_that_has_no_area():
    from bores.grids.containment import compute_triangle_distance_squared

    corner = np.array([1.0, 2.0, 3.0])
    for corners in (
        [corner, corner, corner],
        [corner, corner, corner + np.array([1.0, 0.0, 0.0])],
    ):
        (a, b, c) = corners
        distance = compute_triangle_distance_squared(
            px=0.0, py=0.0, pz=0.0,
            ax=a[0], ay=a[1], az=a[2], bx=b[0], by=b[1], bz=b[2], cx=c[0], cy=c[1], cz=c[2],
        )  # fmt: skip
        assert np.isfinite(distance) and distance >= 0.0
