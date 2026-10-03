"""Voronoi (PEBI) grid factory for both 2-D extruded and 3-D native tessellations."""

import typing
from typing import TypeAlias

import numba
import numpy as np
import numpy.typing as npt
from scipy.spatial import Voronoi

from bores.errors import InvalidGridError, ValidationError
from bores.grids.base import Grid
from bores.grids.factories.base import FaceVertexIndices, build_csr_face_arrays
from bores.types import (
    IntArray,
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
        generator points.  At least 4 seeds are required (3 are needed for a
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

    # Compute mirror seeds and run Voronoi
    mirror_seeds = build_2d_mirror_seeds(seeds, bounding_box)
    all_seeds = np.vstack([seeds, mirror_seeds])
    voronoi = Voronoi(all_seeds)

    # Extract per-column 2-D polygons from Voronoi regions.
    # With the mirror trick, regions for original seeds are guaranteed finite.
    column_polygons: list[npt.NDArray[np.float64] | None] = []
    for seed_idx in range(n_seeds):
        region_idx = voronoi.point_region[seed_idx]
        vertex_indices = voronoi.regions[region_idx]
        if len(vertex_indices) < 3 or -1 in vertex_indices:
            # Should not happen with mirror seeds, but be defensive
            column_polygons.append(None)
        else:
            column_polygons.append(voronoi.vertices[vertex_indices])

    valid_column_indices = [
        column_idx for column_idx, polygon in enumerate(column_polygons) if polygon is not None
    ]
    if not valid_column_indices:
        raise InvalidGridError(
            "No valid Voronoi columns found within the bounding box. "
            "Verify that seed points lie within the bounding box."
        )

    # Collect all valid 2-D ridges (line segments).
    # Each ridge -> one vertical face per layer.
    valid_ridges: list[
        tuple[
            int,  # a_vertex_idx  (index into voronoi.vertices)
            int,  # b_vertex_idx
            npt.NDArray[np.float64],  # a_xy  (for signed-area winding test only)
            npt.NDArray[np.float64],  # b_xy
            int,  # seed_owner
            int,  # seed_neighbour  (-1 = boundary)
        ]
    ] = []

    for ridge_vertex_indices, seed_pair in zip(
        voronoi.ridge_vertices, voronoi.ridge_points, strict=False
    ):
        sa, sb = int(seed_pair[0]), int(seed_pair[1])
        sa_original = sa < n_seeds
        sb_original = sb < n_seeds

        if (not sa_original and not sb_original) or (-1 in ridge_vertex_indices):
            continue

        # Store integer indices directly — no coordinate lookup needed later.
        a_vi = int(ridge_vertex_indices[0])
        b_vi = int(ridge_vertex_indices[1])
        a_xy = voronoi.vertices[a_vi]
        b_xy = voronoi.vertices[b_vi]

        if sa_original and sb_original:
            valid_ridges.append((a_vi, b_vi, a_xy, b_xy, sa, sb))
        elif sa_original:
            valid_ridges.append((a_vi, b_vi, a_xy, b_xy, sa, BOUNDARY_CELL))
        else:
            valid_ridges.append((a_vi, b_vi, a_xy, b_xy, sb, BOUNDARY_CELL))

    # Build 3-D vertex array and CSR face arrays.
    #
    # Vertex indexing scheme for layered extrusion:
    #   z_nodes[layer] = z_top + cumsum(layer_thicknesses)[layer]
    #   There are n_layers + 1 depth levels (z_nodes has n_layers+1 entries).
    #
    # For each 2-D Voronoi vertex v, its 3-D counterpart at depth level lev is:
    #   global_vertex_index = v * (n_layers + 1) + lev
    #
    # Column polygon vertices for top/bottom faces index into the same table.
    z_nodes = compute_depth_nodes(z_top, layer_thicknesses)  # length n_layers + 1
    n_z_levels = len(z_nodes)  # n_layers + 1

    # All 3-D vertex coordinates: (n_voronoi_verts * n_z_levels, 3)
    n_voronoi_verts = len(voronoi.vertices)
    vertex_coordinates_3d = np.empty((n_voronoi_verts * n_z_levels, 3), dtype=np.float64)
    for vert_idx in range(n_voronoi_verts):
        for level in range(n_z_levels):
            row = vert_idx * n_z_levels + level
            vertex_coordinates_3d[row, 0] = voronoi.vertices[vert_idx, 0]
            vertex_coordinates_3d[row, 1] = voronoi.vertices[vert_idx, 1]
            vertex_coordinates_3d[row, 2] = z_nodes[level]

    def get_global_vertex_index(voronoi_vertex_idx: int, z_level: int) -> int:
        """
        Map a 2-D Voronoi vertex index and depth level to a global 3-D index.

        :param voronoi_vertex_idx: Index into `voronoi.vertices`.
        :param z_level: Depth level index in `[0, n_layers]`.
        :returns: Global vertex index into `vertex_coordinates_3d`.
        """
        return voronoi_vertex_idx * n_z_levels + z_level

    # Cell index: cell_index(column_idx, layer) = column_idx * n_layers + layer
    # where column_idx is the index into valid_column_indices
    column_to_cell_column: dict[int, int] = {
        original_column: cell_column
        for cell_column, original_column in enumerate(valid_column_indices)
    }
    n_columns = len(valid_column_indices)
    n_cells = n_columns * n_layers

    # Build per-cell face lists.
    # Cell ordering: for each valid column (outer), for each layer (inner).
    per_cell_face_vertex_lists: PerCellFaceLists = [[] for _ in range(n_cells)]

    def get_global_cell_index(column_idx: int, layer: int) -> int:
        """
        Global cell index from column and layer indices.

        :param column_idx: Index into `valid_column_indices` (0-based).
        :param layer: Layer index in `[0, n_layers - 1]`.
        :returns: Flat cell index.
        """
        return column_idx * n_layers + layer

    # Vertical (lateral) faces: one per 2-D ridge x n_layers
    for a_vertex_idx, b_vertex_idx, a_xy, b_xy, seed_owner, seed_neighbour in valid_ridges:
        # a_vertex_idx / b_vertex_idx are already the correct indices into
        # voronoi.vertices — no coordinate search needed.
        owner_seed_xy = seeds[seed_owner]

        for layer in range(n_layers):
            top_level = layer
            bottom_level = layer + 1

            a_top = get_global_vertex_index(a_vertex_idx, top_level)
            a_bottom = get_global_vertex_index(a_vertex_idx, bottom_level)
            b_top = get_global_vertex_index(b_vertex_idx, top_level)
            b_bottom = get_global_vertex_index(b_vertex_idx, bottom_level)

            signed_area = compute_2d_signed_area(a_xy, b_xy, owner_seed_xy)  # type: ignore[arg-type]
            if signed_area > 0:
                lateral_face: FaceVertexIndices = [a_top, b_top, b_bottom, a_bottom]
            else:
                lateral_face = [a_top, a_bottom, b_bottom, b_top]

            owner_cell_column = column_to_cell_column.get(seed_owner)
            if owner_cell_column is None:
                continue
            owner_cell_idx = get_global_cell_index(owner_cell_column, layer)
            per_cell_face_vertex_lists[owner_cell_idx].append(lateral_face)

            if seed_neighbour != BOUNDARY_CELL:
                neighbour_cell_column = column_to_cell_column.get(seed_neighbour)
                if neighbour_cell_column is not None:
                    neighbour_cell_idx = get_global_cell_index(neighbour_cell_column, layer)
                    per_cell_face_vertex_lists[neighbour_cell_idx].append(
                        list(reversed(lateral_face))
                    )

    # Horizontal faces: top and bottom of each prism cell.
    # Top face (normal = −z = upward): polygon CCW from above = reversed in xy.
    # Bottom face (normal = +z = downward): polygon CCW from below = xy order.
    for cell_column_idx, original_column_idx in enumerate(valid_column_indices):
        polygon_2d = column_polygons[original_column_idx]
        if polygon_2d is None:
            continue
        n_polygon_vertices = len(polygon_2d)

        # Recover Voronoi vertex indices for each polygon vertex
        region_idx = voronoi.point_region[original_column_idx]
        region_vertex_indices = list(voronoi.regions[region_idx])
        assert len(region_vertex_indices) == n_polygon_vertices
        polygon_x = polygon_2d[:, 0]
        polygon_y = polygon_2d[:, 1]
        shoelace = np.dot(polygon_x, np.roll(polygon_y, -1)) - np.dot(
            np.roll(polygon_x, -1), polygon_y
        )
        if shoelace < 0.0:
            region_vertex_indices.reverse()

        for layer in range(n_layers):
            top_level = layer
            bottom_level = layer + 1
            cell_idx = get_global_cell_index(cell_column_idx, layer)

            top_polygon_3d_indices = [
                get_global_vertex_index(region_vertex_indices[k], top_level)
                for k in range(n_polygon_vertices)
            ]
            bottom_polygon_3d_indices = [
                get_global_vertex_index(region_vertex_indices[k], bottom_level)
                for k in range(n_polygon_vertices)
            ]

            # Top face: outward normal = −z -> CCW from above = reversed xy order
            top_face: FaceVertexIndices = list(reversed(top_polygon_3d_indices))
            # Bottom face: outward normal = +z -> CCW from below = xy order
            bottom_face: FaceVertexIndices = list(bottom_polygon_3d_indices)

            per_cell_face_vertex_lists[cell_idx].append(top_face)
            per_cell_face_vertex_lists[cell_idx].append(bottom_face)

    return build_csr_face_arrays(vertex_coordinates_3d, per_cell_face_vertex_lists)


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
    vertices) shared between two cells.  The mirror-seed strategy converts all
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

    # Build per-cell face lists from Voronoi ridges.
    # In 3-D, each "ridge" is a polygonal face between two seed regions.
    per_cell_face_vertex_lists: PerCellFaceLists = [[] for _ in range(n_seeds)]

    for ridge_vertex_indices, seed_pair in zip(
        voronoi.ridge_vertices, voronoi.ridge_points, strict=False
    ):
        sa, sb = int(seed_pair[0]), int(seed_pair[1])
        sa_original = sa < n_seeds
        sb_original = sb < n_seeds

        # Skip: both mirrors, or residual infinite ridge (should be zero with mirror trick)
        if (not sa_original and not sb_original) or (-1 in ridge_vertex_indices):
            continue

        face_vertices_3d = voronoi.vertices[ridge_vertex_indices]

        # Determine owner and the direction toward the neighbour
        if sa_original and sb_original:
            # Interior face: owner = sa, neighbour = sb
            owner_idx = sa
            neighbour_idx = sb
            direction_to_neighbour = seeds[sb] - seeds[sa]
        elif sa_original:
            # Boundary face: owner = sa, neighbour = exterior (-1)
            owner_idx = sa
            neighbour_idx = BOUNDARY_CELL
            direction_to_neighbour = all_seeds[sb] - seeds[sa]
        else:
            # Boundary face: owner = sb, neighbour = exterior (-1)
            owner_idx = sb
            neighbour_idx = BOUNDARY_CELL
            direction_to_neighbour = all_seeds[sa] - seeds[sb]

        # Orient face vertices CCW from owner (normal pointing toward neighbour)
        oriented_vertex_indices = _orient_face_vertices(
            vertex_indices=list(ridge_vertex_indices),
            face_vertices_3d=face_vertices_3d,  # type: ignore[arg-type]
            direction_to_neighbour=direction_to_neighbour,
        )
        per_cell_face_vertex_lists[owner_idx].append(oriented_vertex_indices)

        if neighbour_idx != BOUNDARY_CELL:
            # Neighbour also needs the face with reversed winding
            per_cell_face_vertex_lists[neighbour_idx].append(
                list(reversed(oriented_vertex_indices))
            )

    # Validate: every cell must have at least 4 faces (minimum for a 3-D cell)
    empty_cells = [i for i, faces in enumerate(per_cell_face_vertex_lists) if len(faces) < 4]
    if empty_cells:
        raise InvalidGridError(
            f"Cells {empty_cells[:5]} have fewer than 4 faces after Voronoi construction. "
            f"This may indicate seeds are coplanar, collinear, or outside the bounding box."
        )
    return build_csr_face_arrays(voronoi.vertices, per_cell_face_vertex_lists)


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
