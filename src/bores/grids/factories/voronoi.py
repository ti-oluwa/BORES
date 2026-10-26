"""Voronoi (PEBI) grid factory for both 2-D extruded and 3-D native tessellations."""

import itertools
import typing
from typing import TypeAlias

import numba
import numpy as np
import numpy.typing as npt
from scipy.spatial import Voronoi

from bores.errors import InvalidGridError, ValidationError
from bores.grids.base import Grid
from bores.grids.factories.base import FaceVertexIndices, build_csr_face_arrays_from_entries
from bores.types import (
    BooleanArray,
    IntArray,
    Integer,
    Number,
    NumberArray,
    OneDimension,
    TwoDimensions,
    UnitSystem,
)

__all__ = ["make_voronoi_grid"]

SeedCoordinates2D: TypeAlias = NumberArray[TwoDimensions]
"""Shape `(n_seeds, 2)` - (x, y) Voronoi generator / seed coordinates."""

SeedCoordinates3D: TypeAlias = NumberArray[TwoDimensions]
"""Shape `(n_seeds, 3)` - (x, y, z) Voronoi generator / seed coordinates."""

BoundingBox2D: TypeAlias = tuple[Number, Number, Number, Number]
"""`(x_min, x_max, y_min, y_max)` axis-aligned bounding rectangle."""

BoundingBox3D: TypeAlias = tuple[Number, Number, Number, Number, Number, Number]
"""`(x_min, x_max, y_min, y_max, z_min, z_max)` axis-aligned bounding box."""

LayerThicknessArray: TypeAlias = NumberArray[OneDimension]
"""Shape `(n_layers,)` - positive thickness of each vertical layer."""

PerCellFaceLists: TypeAlias = list[list[FaceVertexIndices]]
"""Outer list indexed by cell; inner list contains that cell's face vertex lists."""

# Index sentinel for boundary (exterior) pseudo-cell.
BOUNDARY_CELL: int = -1


def make_voronoi_grid(
    seed_coordinates: npt.ArrayLike,
    *,
    bounding_box: BoundingBox2D | BoundingBox3D | None = None,
    # 2-D extruded parameters (ignored for 3-D seeds)
    z_top: Number = 0.0,
    layer_thicknesses: Number | npt.ArrayLike = 1.0,
    unit_system: UnitSystem = UnitSystem.FIELD,
    metadata: typing.Mapping[str, typing.Any] | None = None,
    cell_statuses: IntArray[OneDimension] | None = None,
    nnc_cell_indices: IntArray[TwoDimensions] | None = None,
    nnc_transmissibilities: NumberArray[OneDimension] | None = None,
    positive_x_transmissibility_multipliers: NumberArray[OneDimension] | None = None,
    negative_x_transmissibility_multipliers: NumberArray[OneDimension] | None = None,
    positive_y_transmissibility_multipliers: NumberArray[OneDimension] | None = None,
    negative_y_transmissibility_multipliers: NumberArray[OneDimension] | None = None,
    positive_z_transmissibility_multipliers: NumberArray[OneDimension] | None = None,
    negative_z_transmissibility_multipliers: NumberArray[OneDimension] | None = None,
) -> Grid:
    """
    Build a Voronoi (PEBI) grid from seed points.

    Dispatches on seed dimensionality:

    - `seeds.shape == (N, 2)` -> 2-D Voronoi extruded uniformly in depth.
    - `seeds.shape == (N, 3)` -> 3-D native polyhedral Voronoi.

    Boundary handling uses the **mirror-seed strategy**,
    which is robust against all degenerate configurations and requires no
    additional dependencies beyond `scipy`.

    :param seed_coordinates: Shape `(N, 2)` or `(N, 3)` array of Voronoi
        generator points. At least 4 seeds are required (3 are needed for a
        non-degenerate 2-D tessellation, but 4 ensures numerical stability).
    :param bounding_box: Axis-aligned bounding box for the grid domain.
        For 2-D seeds: `(x_min, x_max, y_min, y_max)`.
        For 3-D seeds: `(x_min, x_max, y_min, y_max, z_min, z_max)`.
        If `None`, inferred from seed extents padded by 5 % in each direction.
    :param z_top: Depth (positive downward) of the top surface of the first
        layer. **2-D extruded mode only** - ignored for 3-D seeds.
    :param layer_thicknesses: Thickness of each vertical layer, in depth
        units (positive). May be a scalar (single layer) or a 1-D array
        (multiple layers stacked downward from `z_top`).
        **2-D extruded mode only** - ignored for 3-D seeds.
    :param metadata: Optional free-form metadata dictionary attached to the grid.
    :param cell_statuses: Optional shape `(n_cells,)` activation flags (1 active, 0 inactive).
        Faces that touch an inactive cell are dropped.
    :returns: A fully initialised, immutable `bores.grids.base.Grid`.
    :raises ValidationError: If fewer than 4 seeds are provided, any thickness
        is non-positive, bounding box is degenerate, or the seed array has the
        wrong shape.
    :raises InvalidGridError: If no valid Voronoi cells survive within the
        bounding box after mirror filtering.

    **Examples**:

    2-D extruded, single layer:
    ```python
    import numpy as np
    from bores.grids.factories.voronoi import make_voronoi_grid

    seeds = np.random.rand(30, 2) * 1000.0
    grid = make_voronoi_grid(
        seeds,
        bounding_box=(0.0, 1000.0, 0.0, 1000.0),
        z_top=2000.0,
        layer_thicknesses=10.0,
    )
    ```

    2-D extruded, multiple layers:
    ```python
    grid = make_voronoi_grid(
        seeds,
        bounding_box=(0.0, 1000.0, 0.0, 1000.0),
        z_top=2000.0,
        layer_thicknesses=np.array([5.0, 10.0, 5.0]),
    )
    ```

    3-D native polyhedral:
    ```python
    seeds = np.random.rand(50, 3) * np.array([1000.0, 1000.0, 100.0])
    grid = make_voronoi_grid(
        seeds,
        bounding_box=(0.0, 1000.0, 0.0, 1000.0, 0.0, 100.0),
    )
    ```
    """
    seeds = np.asarray(seed_coordinates, dtype=np.float64)
    if seeds.ndim != 2 or seeds.shape[1] not in (2, 3):
        raise ValidationError(
            f"`seed_coordinates` must be shape (N, 2) or (N, 3); got {seeds.shape!r}."
        )
    if seeds.shape[0] < 4:
        raise ValidationError(f"At least 4 seed points are required; got {seeds.shape[0]}.")

    n_dims = seeds.shape[1]
    if n_dims == 2:
        bounding_box = typing.cast(BoundingBox2D, bounding_box)
        (
            vertex_coordinates,
            face_vertex_indices,
            face_vertex_offsets,
            face_cell_indices,
        ) = make_2d_voronoi_grid(
            seeds=seeds,  # type: ignore[arg-type]
            bounding_box=resolve_2d_bounding_box(seeds, bounding_box),  # type: ignore[arg-type]
            z_top=z_top,
            layer_thicknesses=resolve_layer_thicknesses(layer_thicknesses),
        )
    else:
        bounding_box = typing.cast(BoundingBox3D, bounding_box)
        (
            vertex_coordinates,
            face_vertex_indices,
            face_vertex_offsets,
            face_cell_indices,
        ) = make_3d_voronoi_grid(
            seeds=seeds,  # type: ignore[arg-type]
            bounding_box=resolve_3d_bounding_box(seeds, bounding_box),  # type: ignore[arg-type]
        )

    return Grid(
        vertex_coordinates=vertex_coordinates,
        face_vertex_indices=face_vertex_indices,
        face_vertex_offsets=face_vertex_offsets,
        face_cell_indices=face_cell_indices,
        unit_system=unit_system,
        metadata=metadata,
        cell_statuses=cell_statuses,
        nnc_cell_indices=nnc_cell_indices,
        nnc_transmissibilities=nnc_transmissibilities,
        positive_x_transmissibility_multipliers=positive_x_transmissibility_multipliers,
        negative_x_transmissibility_multipliers=negative_x_transmissibility_multipliers,
        positive_y_transmissibility_multipliers=positive_y_transmissibility_multipliers,
        negative_y_transmissibility_multipliers=negative_y_transmissibility_multipliers,
        positive_z_transmissibility_multipliers=positive_z_transmissibility_multipliers,
        negative_z_transmissibility_multipliers=negative_z_transmissibility_multipliers,
    )


def flatten_ragged(
    rows: typing.Sequence[typing.Sequence[int]],
) -> tuple[IntArray[OneDimension], IntArray[OneDimension]]:
    """
    Pack a list of integer lists into one flat array and its offsets.

    :param rows: The lists, such as `scipy.spatial.Voronoi.regions`.
    :returns: `(flat, offsets)` where `flat[offsets[i]:offsets[i + 1]]` is `rows[i]`.
    """
    lengths = np.fromiter(map(len, rows), dtype=np.int64, count=len(rows))
    flat = np.fromiter(
        itertools.chain.from_iterable(rows), dtype=np.int64, count=int(lengths.sum())
    )
    return flat, np.concatenate([[0], np.cumsum(lengths)])  # type: ignore[return-value]


def resolve_ridge_owners(
    ridge_points: IntArray[TwoDimensions],
    n_seeds: int,
) -> tuple[BooleanArray[OneDimension], IntArray[OneDimension], IntArray[OneDimension]]:
    """
    Decide, for every Voronoi ridge, which original seed owns it and which one it neighbours.

    A ridge between two mirror seeds is dropped. A ridge between an original seed and a mirror
    seed is a boundary face of the original seed; one between two original seeds is interior,
    owned by the first of the pair.

    :param ridge_points: Shape `(n_ridges, 2)` seed pair of each ridge.
    :param n_seeds: Number of original seeds (mirror seeds come after them).
    :returns: `(keep, owners, neighbours)` for every ridge; `neighbours` is `BOUNDARY_CELL` for a
        boundary face, and `owners`/`neighbours` are only meaningful where `keep` is set.
    """
    first = ridge_points[:, 0].astype(np.int64)
    second = ridge_points[:, 1].astype(np.int64)
    first_original = first < n_seeds
    second_original = second < n_seeds
    keep = first_original | second_original
    owners = np.where(first_original, first, second)
    neighbours = np.where(first_original & second_original, second, BOUNDARY_CELL)
    return (
        typing.cast(BooleanArray[OneDimension], keep),
        typing.cast(IntArray[OneDimension], owners),
        typing.cast(IntArray[OneDimension], neighbours),
    )


@numba.njit(cache=True)
def build_2d_prism_entries(
    region_vertices: IntArray[OneDimension],
    region_offsets: IntArray[OneDimension],
    point_region: IntArray[OneDimension],
    voronoi_vertices: NumberArray[TwoDimensions],
    seeds: SeedCoordinates2D,
    ridge_a: IntArray[OneDimension],
    ridge_b: IntArray[OneDimension],
    ridge_owner: IntArray[OneDimension],
    ridge_neighbour: IntArray[OneDimension],
    n_layers: Integer,
    n_z_levels: Integer,
) -> tuple[
    Integer,
    IntArray[OneDimension],
    IntArray[OneDimension],
    IntArray[OneDimension],
    IntArray[OneDimension],
]:
    """
    List the faces of every prism cell of an extruded 2-D Voronoi grid, cell by cell.

    Cells are numbered column by column and, within a column, layer by layer. A cell lists the
    lateral faces of the ridges of its column (in ridge order), then its top and bottom faces.
    Each face is wound outward from the cell. Global vertex `(v, level)` of the extruded grid is
    `v * n_z_levels + level`.

    :param region_vertices: Vertex indices of every Voronoi region, one region after another.
    :param region_offsets: Offsets into `region_vertices` for each region.
    :param point_region: Region of each seed (original seeds first).
    :param voronoi_vertices: Shape `(n_vertices, 2)` Voronoi vertex coordinates.
    :param seeds: Shape `(n_seeds, 2)` original seeds.
    :param ridge_a: First vertex of each kept ridge.
    :param ridge_b: Second vertex of each kept ridge.
    :param ridge_owner: Original seed owning each kept ridge.
    :param ridge_neighbour: Original seed across each kept ridge, or `-1`.
    :param n_layers: Number of layers.
    :param n_z_levels: Number of depth levels (`n_layers + 1`).
    :returns: `(n_columns, valid_seeds, entry_cells, entry_offsets, entry_vertices)`; see
        `bores.grids.factories.base.deduplicate_faces` for the entry arrays.
    """
    n_seeds = seeds.shape[0]
    column_of_seed = np.full(n_seeds, -1, dtype=np.int64)
    valid_seeds = np.empty(n_seeds, dtype=np.int64)
    n_columns = 0
    for seed in range(n_seeds):
        region = point_region[seed]
        begin = region_offsets[region]
        end = region_offsets[region + 1]
        valid = end - begin >= 3
        for position in range(begin, end):
            if region_vertices[position] == -1:
                valid = False
                break
        if valid:
            column_of_seed[seed] = n_columns
            valid_seeds[n_columns] = seed
            n_columns += 1

    n_cells = n_columns * n_layers
    lateral_count = np.zeros(n_cells, dtype=np.int64)
    n_ridges = ridge_a.shape[0]
    for ridge in range(n_ridges):
        owner_column = column_of_seed[ridge_owner[ridge]]
        if owner_column < 0:
            continue
        neighbour_column = -1
        if ridge_neighbour[ridge] != -1:
            neighbour_column = column_of_seed[ridge_neighbour[ridge]]
        for layer in range(n_layers):
            lateral_count[owner_column * n_layers + layer] += 1
            if neighbour_column >= 0:
                lateral_count[neighbour_column * n_layers + layer] += 1

    cell_start = np.zeros(n_cells + 1, dtype=np.int64)
    for cell in range(n_cells):
        cell_start[cell + 1] = cell_start[cell] + lateral_count[cell] + 2
    n_entries = cell_start[n_cells]

    # What each entry is: 0 lateral face seen from the ridge owner, 1 the same face seen from the
    # neighbour, 2 top face, 3 bottom face; `reference` is its ridge (or column), `layer_of` its layer.
    kind = np.empty(n_entries, dtype=np.int8)
    reference = np.empty(n_entries, dtype=np.int64)
    layer_of = np.empty(n_entries, dtype=np.int64)
    cursor = cell_start[:-1].copy()
    for ridge in range(n_ridges):
        owner_column = column_of_seed[ridge_owner[ridge]]
        if owner_column < 0:
            continue
        neighbour_column = -1
        if ridge_neighbour[ridge] != -1:
            neighbour_column = column_of_seed[ridge_neighbour[ridge]]
        for layer in range(n_layers):
            cell = owner_column * n_layers + layer
            slot = cursor[cell]
            kind[slot] = 0
            reference[slot] = ridge
            layer_of[slot] = layer
            cursor[cell] += 1
            if neighbour_column >= 0:
                cell = neighbour_column * n_layers + layer
                slot = cursor[cell]
                kind[slot] = 1
                reference[slot] = ridge
                layer_of[slot] = layer
                cursor[cell] += 1
    for column in range(n_columns):
        for layer in range(n_layers):
            cell = column * n_layers + layer
            slot = cursor[cell]
            kind[slot] = 2
            reference[slot] = column
            layer_of[slot] = layer
            kind[slot + 1] = 3
            reference[slot + 1] = column
            layer_of[slot + 1] = layer
            cursor[cell] += 2

    entry_cells = np.empty(n_entries, dtype=np.int64)
    entry_offsets = np.zeros(n_entries + 1, dtype=np.int64)
    for cell in range(n_cells):
        for slot in range(cell_start[cell], cell_start[cell + 1]):
            entry_cells[slot] = cell
    for slot in range(n_entries):
        size = 4
        if kind[slot] >= 2:
            region = point_region[valid_seeds[reference[slot]]]
            size = region_offsets[region + 1] - region_offsets[region]
        entry_offsets[slot + 1] = entry_offsets[slot] + size
    entry_vertices = np.empty(entry_offsets[n_entries], dtype=np.int64)

    for slot in range(n_entries):
        begin = entry_offsets[slot]
        layer = layer_of[slot]
        if kind[slot] < 2:
            ridge = reference[slot]
            a_top = ridge_a[ridge] * n_z_levels + layer
            a_bottom = a_top + 1
            b_top = ridge_b[ridge] * n_z_levels + layer
            b_bottom = b_top + 1
            signed_area = compute_2d_signed_area(
                a=voronoi_vertices[ridge_a[ridge]],  # type: ignore[arg-type]
                b=voronoi_vertices[ridge_b[ridge]],  # type: ignore[arg-type]
                point=seeds[ridge_owner[ridge]],  # type: ignore[arg-type]
            )
            if signed_area > 0:
                lateral = (a_top, b_top, b_bottom, a_bottom)
            else:
                lateral = (a_top, a_bottom, b_bottom, b_top)
            for corner in range(4):
                if kind[slot] == 0:
                    entry_vertices[begin + corner] = lateral[corner]
                else:
                    entry_vertices[begin + corner] = lateral[3 - corner]
        else:
            region = point_region[valid_seeds[reference[slot]]]
            region_begin = region_offsets[region]
            n_polygon_vertices = region_offsets[region + 1] - region_begin
            forward = 0.0
            backward = 0.0
            for k in range(n_polygon_vertices):
                current = region_vertices[region_begin + k]
                following = region_vertices[region_begin + (k + 1) % n_polygon_vertices]
                forward += voronoi_vertices[current, 0] * voronoi_vertices[following, 1]
                backward += voronoi_vertices[following, 0] * voronoi_vertices[current, 1]
            counter_clockwise = forward - backward >= 0.0
            level = layer if kind[slot] == 2 else layer + 1
            for k in range(n_polygon_vertices):
                # Oriented polygon `o`: the region order when counter-clockwise, else reversed.
                oriented = k if counter_clockwise else n_polygon_vertices - 1 - k
                vertex = region_vertices[region_begin + oriented]
                if kind[slot] == 2:
                    # Top face: outward normal is up, so wind it the other way round.
                    entry_vertices[begin + n_polygon_vertices - 1 - k] = (
                        vertex * n_z_levels + level
                    )
                else:
                    entry_vertices[begin + k] = vertex * n_z_levels + level

    return (
        n_columns,
        valid_seeds[:n_columns],  # type: ignore[return-value]
        entry_cells,
        entry_offsets,
        entry_vertices,
    )


@numba.njit(cache=True)
def build_3d_cell_entries(
    ridge_vertices: IntArray[OneDimension],
    ridge_offsets: IntArray[OneDimension],
    ridge_owner: IntArray[OneDimension],
    ridge_neighbour: IntArray[OneDimension],
    ridge_direction: NumberArray[TwoDimensions],
    keep: npt.NDArray[np.bool_],
    voronoi_vertices: NumberArray[TwoDimensions],
    n_seeds: int,
) -> tuple[
    IntArray[OneDimension],
    IntArray[OneDimension],
    IntArray[OneDimension],
    IntArray[OneDimension],
]:
    """
    List the faces of every cell of a 3-D Voronoi grid, cell by cell.

    A cell lists the faces of its ridges in ridge order. Each face is wound so that its normal
    points from the ridge owner toward the neighbour (or mirror seed), and the neighbour lists
    it reversed.

    :param ridge_vertices: Vertex indices of every ridge, one ridge after another.
    :param ridge_offsets: Offsets into `ridge_vertices` for each ridge.
    :param ridge_owner: Original seed owning each ridge.
    :param ridge_neighbour: Original seed across each ridge, or `-1`.
    :param ridge_direction: Shape `(n_ridges, 3)` direction from the owner toward the neighbour.
    :param keep: Whether each ridge is used.
    :param voronoi_vertices: Shape `(n_vertices, 3)` Voronoi vertex coordinates.
    :param n_seeds: Number of original seeds, which are the cells.
    :returns: `(faces_per_cell, entry_cells, entry_offsets, entry_vertices)`.
    """
    n_ridges = ridge_offsets.shape[0] - 1
    use = np.zeros(n_ridges, dtype=np.bool_)
    faces_per_cell = np.zeros(n_seeds, dtype=np.int64)
    for ridge in range(n_ridges):
        if not keep[ridge]:
            continue
        has_infinite_vertex = False
        for position in range(ridge_offsets[ridge], ridge_offsets[ridge + 1]):
            if ridge_vertices[position] == -1:
                has_infinite_vertex = True
                break
        if has_infinite_vertex:
            continue
        use[ridge] = True
        faces_per_cell[ridge_owner[ridge]] += 1
        if ridge_neighbour[ridge] != -1:
            faces_per_cell[ridge_neighbour[ridge]] += 1

    cell_start = np.zeros(n_seeds + 1, dtype=np.int64)
    for cell in range(n_seeds):
        cell_start[cell + 1] = cell_start[cell] + faces_per_cell[cell]
    n_entries = cell_start[n_seeds]

    entry_ridge = np.empty(n_entries, dtype=np.int64)
    entry_reversed = np.empty(n_entries, dtype=np.bool_)
    entry_cells = np.empty(n_entries, dtype=np.int64)
    cursor = cell_start[:-1].copy()
    for ridge in range(n_ridges):
        if not use[ridge]:
            continue
        begin = ridge_offsets[ridge]
        end = ridge_offsets[ridge + 1]
        polygon = np.empty((end - begin, 3), dtype=np.float64)
        for k in range(end - begin):
            for axis in range(3):
                polygon[k, axis] = voronoi_vertices[ridge_vertices[begin + k], axis]
        normal = compute_newell_normal(polygon)
        alignment = (
            normal[0] * ridge_direction[ridge, 0]
            + normal[1] * ridge_direction[ridge, 1]
            + normal[2] * ridge_direction[ridge, 2]
        )
        reverse = alignment < 0.0

        owner = ridge_owner[ridge]
        slot = cursor[owner]
        entry_ridge[slot] = ridge
        entry_reversed[slot] = reverse
        entry_cells[slot] = owner
        cursor[owner] += 1
        if ridge_neighbour[ridge] != -1:
            neighbour = ridge_neighbour[ridge]
            slot = cursor[neighbour]
            entry_ridge[slot] = ridge
            entry_reversed[slot] = not reverse
            entry_cells[slot] = neighbour
            cursor[neighbour] += 1

    entry_offsets = np.zeros(n_entries + 1, dtype=np.int64)
    for slot in range(n_entries):
        ridge = entry_ridge[slot]
        entry_offsets[slot + 1] = (
            entry_offsets[slot] + ridge_offsets[ridge + 1] - ridge_offsets[ridge]
        )
    entry_vertices = np.empty(entry_offsets[n_entries], dtype=np.int64)
    for slot in range(n_entries):
        ridge = entry_ridge[slot]
        begin = ridge_offsets[ridge]
        size = ridge_offsets[ridge + 1] - begin
        for k in range(size):
            source = size - 1 - k if entry_reversed[slot] else k
            entry_vertices[entry_offsets[slot] + k] = ridge_vertices[begin + source]
    return faces_per_cell, entry_cells, entry_offsets, entry_vertices


def make_2d_voronoi_grid(
    seeds: SeedCoordinates2D,
    bounding_box: BoundingBox2D,
    z_top: Number,
    layer_thicknesses: LayerThicknessArray,
) -> tuple[
    NumberArray[TwoDimensions],
    IntArray[OneDimension],
    IntArray[OneDimension],
    IntArray[TwoDimensions],
]:
    """
    Build a 2-D Voronoi grid **extruded** through one or more depth layers.

    Each 2-D Voronoi cell column is extruded into `n_layers` prism cells
    stacked downward from `z_top`. Interior vertical faces are shared
    between laterally adjacent prisms in the same layer; horizontal top/bottom
    faces are shared between prisms in adjacent layers.

    :param seeds: Shape `(n_seeds, 2)` 2-D seed array.
    :param bounding_box: `(x_min, x_max, y_min, y_max)` bounding rectangle.
    :param z_top: Depth of the top node of the first layer.
    :param layer_thicknesses: Shape `(n_layers,)` positive thickness per layer.
    :param metadata: Optional metadata dictionary.
    :returns: Tuple `(vertex_coordinates, face_vertex_indices,
        face_vertex_offsets, face_cell_indices)` ready to be passed to
        `bores.grids.base.Grid`.
    """
    n_seeds = len(seeds)
    n_layers = len(layer_thicknesses)

    # Compute mirror seeds and run Voronoi. With the mirror trick, the regions of the original
    # seeds are finite.
    mirror_seeds = build_2d_mirror_seeds(seeds, bounding_box)
    all_seeds = np.vstack([seeds, mirror_seeds])
    voronoi = Voronoi(all_seeds)

    # A ridge becomes one vertical face per layer. Ridges between two mirror seeds, and any ridge
    # reaching infinity, take no part.
    ridge_vertices = np.asarray(voronoi.ridge_vertices, dtype=np.int64).reshape(-1, 2)
    keep, owners, neighbours = resolve_ridge_owners(voronoi.ridge_points, n_seeds)
    keep &= ~(ridge_vertices == -1).any(axis=1)

    # Vertex `v` at depth level `lev` of the extruded grid is `v * n_z_levels + lev`.
    z_nodes = compute_depth_nodes(z_top, layer_thicknesses)
    n_z_levels = len(z_nodes)
    n_voronoi_verts = len(voronoi.vertices)
    vertex_coordinates_3d = np.empty((n_voronoi_verts * n_z_levels, 3), dtype=np.float64)
    vertex_coordinates_3d[:, :2] = np.repeat(voronoi.vertices, n_z_levels, axis=0)
    vertex_coordinates_3d[:, 2] = np.tile(z_nodes, n_voronoi_verts)

    region_vertices, region_offsets = flatten_ragged(voronoi.regions)
    n_columns, _, entry_cells, entry_offsets, entry_vertices = build_2d_prism_entries(
        region_vertices=region_vertices,
        region_offsets=region_offsets,
        point_region=typing.cast(
            IntArray[OneDimension], np.ascontiguousarray(voronoi.point_region, dtype=np.int64)
        ),
        voronoi_vertices=typing.cast(
            NumberArray[TwoDimensions], np.ascontiguousarray(voronoi.vertices, dtype=np.float64)
        ),
        seeds=typing.cast(SeedCoordinates2D, np.ascontiguousarray(seeds, dtype=np.float64)),
        ridge_a=typing.cast(IntArray[OneDimension], np.ascontiguousarray(ridge_vertices[keep, 0])),
        ridge_b=typing.cast(IntArray[OneDimension], np.ascontiguousarray(ridge_vertices[keep, 1])),
        ridge_owner=typing.cast(IntArray[OneDimension], np.ascontiguousarray(owners[keep])),
        ridge_neighbour=typing.cast(
            IntArray[OneDimension], np.ascontiguousarray(neighbours[keep])
        ),
        n_layers=n_layers,
        n_z_levels=n_z_levels,
    )
    if n_columns == 0:
        raise InvalidGridError(
            "No valid Voronoi columns found within the bounding box. "
            "Verify that seed points lie within the bounding box."
        )

    return build_csr_face_arrays_from_entries(
        vertex_coordinates_3d,
        entry_cells=entry_cells,
        entry_offsets=entry_offsets,
        entry_vertices=entry_vertices,
    )


def make_3d_voronoi_grid(
    seeds: SeedCoordinates3D, bounding_box: BoundingBox3D
) -> tuple[
    NumberArray[TwoDimensions],
    IntArray[OneDimension],
    IntArray[OneDimension],
    IntArray[TwoDimensions],
]:
    """
    Build a 3-D native polyhedral Voronoi grid.

    Each ridge in the 3-D Voronoi tessellation is a polygonal face (with 3+
    vertices) shared between two cells. The mirror-seed strategy converts all
    domain-interior ridges to finite faces and all boundary ridges to faces
    between an original seed and its mirror.

    :param seeds: Shape `(n_seeds, 3)` 3-D seed array.
    :param bounding_box: `(x_min, x_max, y_min, y_max, z_min, z_max)`.
    :param metadata: Optional metadata dictionary.
    :returns: Tuple `(vertex_coordinates, face_vertex_indices,
        face_vertex_offsets, face_cell_indices)` ready to be passed to
        `bores.grids.base.Grid`.
    """
    n_seeds = len(seeds)

    # Add mirror seeds and run Voronoi
    mirror_seeds = build_3d_mirror_seeds(seeds, bounding_box)
    all_seeds = np.vstack([seeds, mirror_seeds])
    voronoi = Voronoi(all_seeds)

    # In 3-D, each "ridge" is a polygonal face between two seed regions. Ridges between two
    # mirror seeds take no part; the others are interior faces between two original seeds or
    # boundary faces between an original seed and its mirror.
    keep, owners, neighbours = resolve_ridge_owners(voronoi.ridge_points, n_seeds)
    ridge_points = voronoi.ridge_points.astype(np.int64)
    other = np.where(owners == ridge_points[:, 0], ridge_points[:, 1], ridge_points[:, 0])
    direction = np.zeros((len(owners), 3), dtype=np.float64)
    direction[keep] = all_seeds[other[keep]] - seeds[owners[keep]]

    ridge_vertices, ridge_offsets = flatten_ragged(voronoi.ridge_vertices)
    faces_per_cell, entry_cells, entry_offsets, entry_vertices = build_3d_cell_entries(
        ridge_vertices=ridge_vertices,
        ridge_offsets=ridge_offsets,
        ridge_owner=typing.cast(IntArray[OneDimension], np.ascontiguousarray(owners)),
        ridge_neighbour=typing.cast(IntArray[OneDimension], np.ascontiguousarray(neighbours)),
        ridge_direction=direction,
        keep=keep,
        voronoi_vertices=typing.cast(
            NumberArray[TwoDimensions], np.ascontiguousarray(voronoi.vertices, dtype=np.float64)
        ),
        n_seeds=n_seeds,
    )

    # Every cell must have at least 4 faces (minimum for a 3-D cell)
    empty_cells = np.flatnonzero(faces_per_cell < 4).tolist()
    if empty_cells:
        raise InvalidGridError(
            f"Cells {empty_cells[:5]} have fewer than 4 faces after Voronoi construction. "
            f"This may indicate seeds are coplanar, collinear, or outside the bounding box."
        )
    return build_csr_face_arrays_from_entries(
        voronoi.vertices,
        entry_cells=entry_cells,
        entry_offsets=entry_offsets,
        entry_vertices=entry_vertices,
    )


@numba.njit(cache=True)
def build_2d_mirror_seeds(
    seeds: SeedCoordinates2D, bounding_box: BoundingBox2D
) -> npt.NDArray[np.float64]:
    """
    Build mirror seeds reflected across each edge of a 2-D bounding rectangle.

    For each original seed, 4 mirrors are created (one per bounding box edge).
    These ensure the Voronoi diagram for the original seeds has no infinite
    ridges involving original seeds.

    :param seeds: Shape `(n_seeds, 2)` original seed coordinates.
    :param bounding_box: `(x_min, x_max, y_min, y_max)`.
    :returns: Shape `(4 * n_seeds, 2)` float64 mirror seed array.
    """
    x_min, x_max, y_min, y_max = bounding_box
    mirrors = np.empty((4 * len(seeds), 2), dtype=np.float64)
    for k, s in enumerate(seeds):
        base = 4 * k
        mirrors[base + 0] = [2 * x_min - s[0], s[1]]  # left reflection
        mirrors[base + 1] = [2 * x_max - s[0], s[1]]  # right reflection
        mirrors[base + 2] = [s[0], 2 * y_min - s[1]]  # bottom reflection
        mirrors[base + 3] = [s[0], 2 * y_max - s[1]]  # top reflection
    return mirrors


@numba.njit(cache=True)
def build_3d_mirror_seeds(
    seeds: SeedCoordinates3D, bounding_box: BoundingBox3D
) -> NumberArray[TwoDimensions]:
    """
    Build mirror seeds reflected across each face of a 3-D bounding box.

    For each original seed, 6 mirrors are created (one per bounding box face).

    :param seeds: Shape `(n_seeds, 3)` original seed coordinates.
    :param bounding_box: `(x_min, x_max, y_min, y_max, z_min, z_max)`.
    :returns: Shape `(6 * n_seeds, 3)` float64 mirror seed array.
    """
    x_min, x_max, y_min, y_max, z_min, z_max = bounding_box
    mirrors = np.empty((6 * len(seeds), 3), dtype=np.float64)
    for k, s in enumerate(seeds):
        base = 6 * k
        mirrors[base + 0] = [2 * x_min - s[0], s[1], s[2]]  # x_min face
        mirrors[base + 1] = [2 * x_max - s[0], s[1], s[2]]  # x_max face
        mirrors[base + 2] = [s[0], 2 * y_min - s[1], s[2]]  # y_min face
        mirrors[base + 3] = [s[0], 2 * y_max - s[1], s[2]]  # y_max face
        mirrors[base + 4] = [s[0], s[1], 2 * z_min - s[2]]  # z_min face
        mirrors[base + 5] = [s[0], s[1], 2 * z_max - s[2]]  # z_max face
    return mirrors


def _orient_face_vertices(
    vertex_indices: FaceVertexIndices,
    face_vertices_3d: NumberArray[TwoDimensions],
    direction_to_neighbour: NumberArray[TwoDimensions],
) -> FaceVertexIndices:
    """
    Return `vertex_indices`, possibly reversed, so Newell's normal aligns
    with `direction_to_neighbour` (outward from owner toward neighbour).

    :param vertex_indices: Ordered vertex indices as returned by scipy Voronoi.
    :param face_vertices_3d: Shape `(n_verts, 3)` world coordinates of those vertices.
    :param direction_to_neighbour: Unit-direction vector from owner seed toward
        the neighbour seed (or mirror seed for boundary faces).
    :returns: `vertex_indices` or its reverse, whichever yields a Newell normal
        with a positive dot product against `direction_to_neighbour`.
    """
    newell_normal = compute_newell_normal(face_vertices_3d)
    dot = float(np.dot(newell_normal, direction_to_neighbour))
    # dot > 0: normal already points toward neighbour (outward from owner) -> correct
    # dot < 0: normal points toward owner -> reverse the winding
    if dot >= 0.0:
        return vertex_indices
    return list(reversed(vertex_indices))


@numba.njit(cache=True)
def compute_newell_normal(vertices: NumberArray[TwoDimensions]) -> NumberArray[OneDimension]:
    """
    Compute the (unnormalised) Newell normal for a planar polygon.

    :param vertices: Shape `(n_verts, 3)` polygon vertex array.
    :returns: Shape `(3,)` normal vector (magnitude = 2 x face area).
    """
    n = np.zeros(3, dtype=np.float64)
    n_verts = len(vertices)
    for k in range(n_verts):
        v1 = vertices[k]
        v2 = vertices[(k + 1) % n_verts]
        n[0] += (v1[1] - v2[1]) * (v1[2] + v2[2])
        n[1] += (v1[2] - v2[2]) * (v1[0] + v2[0])
        n[2] += (v1[0] - v2[0]) * (v1[1] + v2[1])
    magnitude = np.linalg.norm(n)
    return n / magnitude if magnitude > 0.0 else n  # type: ignore[return-value]


@numba.njit(cache=True)
def compute_2d_signed_area(
    a: NumberArray[OneDimension], b: NumberArray[OneDimension], point: NumberArray[OneDimension]
) -> float:
    """
    Compute the 2-D signed area of triangle `(a, b, point)`.

    A positive value means `point` is to the **left** of the directed
    segment `a -> b` (counter-clockwise orientation).

    :param a: First edge vertex, shape `(2,)`.
    :param b: Second edge vertex, shape `(2,)`.
    :param point: Test point, shape `(2,)`.
    :returns: Signed area (positive if `point` is left of `a -> b`).
    """
    return 0.5 * ((b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0]))


def compute_depth_nodes(
    z_top: Number, layer_thicknesses: LayerThicknessArray
) -> NumberArray[OneDimension]:
    """
    Compute the depth of every node surface from `z_top` downward.

    :param z_top: Depth of the topmost node surface (positive downward).
    :param layer_thicknesses: Shape `(n_layers,)` positive thickness per layer.
    :returns: Shape `(n_layers + 1,)` depth node array starting at `z_top`.
    """
    return np.concatenate([[z_top], z_top + np.cumsum(layer_thicknesses)])  # type: ignore[return-value]


def resolve_2d_bounding_box(
    seeds: NumberArray[TwoDimensions], bounding_box: BoundingBox2D | None
) -> BoundingBox2D:
    """
    Resolve or infer a 2-D bounding box from seed extents.

    :param seeds: Shape `(n_seeds, 2)` seed array.
    :param bounding_box: Caller-supplied box or `None`.
    :returns: `(x_min, x_max, y_min, y_max)` tuple.
    :raises ValidationError: If a supplied box has zero extent in any dimension.
    """
    if bounding_box is not None:
        if len(bounding_box) not in (4, 6):
            raise ValidationError(
                f"For 2-D seeds, `bounding_box` must be length 4 or 6; "
                f"got length {len(bounding_box)}."
            )
        x_min, x_max, y_min, y_max = bounding_box[:4]
    else:
        pad_x = max((seeds[:, 0].max() - seeds[:, 0].min()) * 0.05, 1.0)
        pad_y = max((seeds[:, 1].max() - seeds[:, 1].min()) * 0.05, 1.0)
        x_min = seeds[:, 0].min() - pad_x
        x_max = seeds[:, 0].max() + pad_x
        y_min = seeds[:, 1].min() - pad_y
        y_max = seeds[:, 1].max() + pad_y

    if x_max <= x_min or y_max <= y_min:
        raise ValidationError(
            f"Bounding box has zero or negative extent: "
            f"x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]."
        )
    return x_min, x_max, y_min, y_max


def resolve_3d_bounding_box(
    seeds: NumberArray[TwoDimensions], bounding_box: BoundingBox3D | None
) -> BoundingBox3D:
    """
    Resolve or infer a 3-D bounding box from seed extents.

    :param seeds: Shape `(n_seeds, 3)` seed array.
    :param bounding_box: Caller-supplied box or `None`.
    :returns: `(x_min, x_max, y_min, y_max, z_min, z_max)` tuple.
    :raises ValidationError: If a supplied box has wrong length or zero extent.
    """
    if bounding_box is not None:
        if len(bounding_box) != 6:
            raise ValidationError(
                f"For 3-D seeds, `bounding_box` must be length 6 (x_min, x_max, "
                f"y_min, y_max, z_min, z_max); got length {len(bounding_box)}."
            )
        x_min, x_max, y_min, y_max, z_min, z_max = bounding_box
    else:
        pads = [max((seeds[:, dim].max() - seeds[:, dim].min()) * 0.05, 1.0) for dim in range(3)]
        x_min = seeds[:, 0].min() - pads[0]
        x_max = seeds[:, 0].max() + pads[0]
        y_min = seeds[:, 1].min() - pads[1]
        y_max = seeds[:, 1].max() + pads[1]
        z_min = seeds[:, 2].min() - pads[2]
        z_max = seeds[:, 2].max() + pads[2]

    if x_max <= x_min or y_max <= y_min or z_max <= z_min:
        raise ValidationError(
            f"Bounding box has zero or negative extent in at least one dimension: "
            f"x=[{x_min}, {x_max}], y=[{y_min}, {y_max}], z=[{z_min}, {z_max}]."
        )
    return (x_min, x_max, y_min, y_max, z_min, z_max)


def resolve_layer_thicknesses(
    layer_thicknesses: Number | npt.ArrayLike,
) -> LayerThicknessArray:
    """
    Convert a scalar or array layer thickness specification to a 1-D array.

    :param layer_thicknesses: Scalar (single uniform layer) or 1-D array.
    :returns: Shape `(n_layers,)` positive float64 array.
    :raises ValidationError: If any thickness is non-positive.
    """
    array = np.atleast_1d(layer_thicknesses).astype(np.float64, copy=False).ravel()
    if np.any(array <= 0.0):
        raise ValidationError(
            f"All `layer_thicknesses` must be strictly positive; got min={array.min():.6g}."
        )
    return array
