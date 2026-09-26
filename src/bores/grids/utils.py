"""Gridding utilities"""

import typing
import warnings
from collections.abc import Mapping

import numba
import numpy as np
import numpy.typing as npt

from bores.errors import ValidationError
from bores.grids.base import Grid
from bores.types import CellArray, IntArray, Integer, OneDimension, Side

__all__ = ["make_pyvista_grid", "resolve_boundary_faces_for_box"]


SIDE_ALIASES: dict[str, Side] = {
    "left": Side.WEST,
    "right": Side.EAST,
    "front": Side.SOUTH,
    "back": Side.NORTH,
    "up": Side.TOP,
    "down": Side.BOTTOM,
}

AXIS_SIDES: dict[int, tuple[Side, Side]] = {
    0: (Side.WEST, Side.EAST),
    1: (Side.SOUTH, Side.NORTH),
    2: (Side.TOP, Side.BOTTOM),
}

# Eclipse's own `I+`/`I-`/`J+`/`J-`/`K+`/`K-` face-direction convention (as
# used by `AQUANCON`), each mapped to (dominant outward-normal axis, sign) -
# axis 0=X/I, 1=Y/J, 2=Z/K, matching `AXIS_SIDES` above.
FACE_DIRECTION_AXIS_SIGN: dict[str, tuple[int, float]] = {
    "I+": (0, 1.0),
    "I-": (0, -1.0),
    "J+": (1, 1.0),
    "J-": (1, -1.0),
    "K+": (2, 1.0),
    "K-": (2, -1.0),
}


def resolve_side(side: Side | str) -> Side:
    """
    Resolve a `Side`, its `.value` string, or a common alias
    ('left'/'right'/'front'/'back'/'up'/'down') to a `Side` member.

    :param side: `Side` member, value string, or alias.
    :returns: Resolved `Side`.
    :raises ValidationError: If `side` doesn't resolve to a known `Side`.
    """
    if isinstance(side, Side):
        return side

    key = side.strip().lower()
    if key in SIDE_ALIASES:
        return SIDE_ALIASES[key]
    try:
        return Side(key)
    except ValueError as exc:
        valid = sorted({member.value for member in Side} | set(SIDE_ALIASES))
        raise ValidationError(f"Unknown side {side!r}. Valid values: {valid}.") from exc


def classify_boundary_faces(grid: Grid) -> dict[Side, IntArray[OneDimension]]:
    """
    Classify every boundary face of `grid` into one of the six `Side`s.

    For each boundary face, its outward normal is re-derived defensively as
    `sign(dot(face_unit_normals[face], face_centroid - owner_cell_centroid))`
    rather than trusting `Grid.face_unit_normals`'s stored sign directly.
    The axis with the largest-magnitude (re-oriented) normal component
    decides X/Y/Z; its sign decides which of the two `Side`s on that axis.

    Every boundary face is genuinely exterior by definition (its
    `face_cell_indices` neighbour is `-1`), so classification is purely by
    outward-normal direction - no bounding-box-extremity check. A face on
    an irregular flank (partway along a fault-bounded edge, or the inner
    corner of an L-shaped field) is still genuinely facing one direction
    and belongs on that side; requiring proximity to the domain's
    bounding-box corner would drop real boundary faces on any non-box-shaped
    field. Every boundary face is assigned to exactly one `Side` - nothing
    is dropped.

    Classification happens in whatever frame `grid`'s stored coordinates
    are actually in. For a grid built with the default
    `apply_map_axes=True` (corner-point/Cartesian factories), that's true
    map/geographic space, so `Side.WEST` means true west. For a grid built
    with `apply_map_axes=False`, it means the grid's own local axes -
    consistent with every other computation on such a grid.

    :param grid: Grid to classify. Must have resolved face geometry
        (`face_unit_normals`, `face_centroids`, `cell_centroids`) - true
        for any `Grid` built through a grid factory.
    :returns: Mapping from every `Side` to the *positions* (0-based, into
        `Grid.boundary_face_indices` - not global face indices) of the
        faces assigned to that side, as an `int32` array. A `Side` with no
        matching faces maps to an empty array rather than being omitted.
    :raises ValidationError: If `grid`'s face geometry hasn't been resolved.
    """
    if grid.cell_centroids is None:
        raise ValidationError(
            "`grid` has no `cell_centroids`; cannot re-derive outward face orientation."
        )

    boundary = grid.boundary_face_indices
    if len(boundary) == 0:
        return {side: np.empty(0, dtype=np.int32) for side in Side}

    owners = grid.face_cell_indices[boundary, 0]
    face_centroids = grid.face_centroids[boundary]
    cell_centroids = grid.cell_centroids[owners]

    raw_normals = grid.face_unit_normals[boundary]
    outward_ref = face_centroids - cell_centroids
    flip = np.sign(np.einsum("ij,ij->i", raw_normals, outward_ref))
    flip[flip == 0] = 1.0
    normals = raw_normals * flip[:, None]

    dominant_axis = np.argmax(np.abs(normals), axis=1)
    dominant_sign = np.sign(normals[np.arange(len(boundary)), dominant_axis])
    # A stored/rederived unit normal should never be exactly zero on its own
    # dominant axis. We guard against it anyway rather than silently dropping.
    dominant_sign[dominant_sign == 0] = 1.0

    positions = np.arange(len(boundary))
    result: dict[Side, IntArray[OneDimension]] = {}
    for axis, (negative_side, positive_side) in AXIS_SIDES.items():
        on_axis = dominant_axis == axis
        result[negative_side] = positions[on_axis & (dominant_sign < 0.0)].astype(  # type: ignore[assignment]
            np.int32
        )
        result[positive_side] = positions[on_axis & (dominant_sign > 0.0)].astype(  # type: ignore[assignment]
            np.int32
        )
    return result


def resolve_boundary_faces_for_box(
    grid: Grid,
    *,
    i1: Integer,
    i2: Integer,
    j1: Integer,
    j2: Integer,
    k1: Integer,
    k2: Integer,
    face_direction: str,
    label: str = "box",
) -> IntArray[OneDimension]:
    """
    Resolve an Eclipse-style IJK box + face direction (as `AQUANCON` gives
    it) to boundary face positions (into `Grid.boundary_face_indices`).

    For each cell in the 1-based, inclusive box, it finds the face in
    `face_direction` by dominant outward-normal axis/sign (same convention
    `classify_boundary_faces` uses) via `Grid.get_face_normal_for_cell`,
    and includes it only when that face is a genuine grid-boundary face, i.e,
    the domain's true edge, not just the edge of an active region within a
    larger box. A cell is skipped, with a warning naming it, when:

    - it is outside `grid`'s bounds,
    - it is inactive,
    - its face in `face_direction` is an interior face. Even one
      adjoining an inactive neighbour. `BoundaryRegion` can only carry
      genuine `Grid.boundary_face_indices` positions, so a boundary
      condition can't be attached to an interior active/inactive junction
      through this function; that needs a different mechanism.

    When a cell has more than one face sharing `face_direction`'s
    dominant axis/sign (possible on irregular corner-point geometry), the
    first one found is used.

    :param grid: Grid to resolve against. Must have `dimensions` set.
    :param i1: 1-based start I index.
    :param i2: 1-based end I index (inclusive).
    :param j1: 1-based start J index.
    :param j2: 1-based end J index (inclusive).
    :param k1: 1-based start K index.
    :param k2: 1-based end K index (inclusive).
    :param face_direction: One of `'I+'`, `'I-'`, `'J+'`, `'J-'`, `'K+'`, `'K-'`.
    :param label: Identifies the source record in warning messages.
    :returns: Sorted, de-duplicated `int32` array of boundary face positions.
    :raises ValidationError: If `face_direction` is invalid, or `grid` has
        no resolvable `dimensions`.
    """
    if face_direction not in FACE_DIRECTION_AXIS_SIGN:
        raise ValidationError(
            f"`face_direction` must be one of {sorted(FACE_DIRECTION_AXIS_SIGN)}; "
            f"got {face_direction!r}."
        )
    axis, sign = FACE_DIRECTION_AXIS_SIGN[face_direction]

    dims = grid.dimensions
    if dims is None:
        raise ValidationError(f"{label}: `grid` has no `dimensions`; cannot resolve an IJK box.")
    nx, ny, nz = dims

    boundary_position_by_face: dict[int, int] = {
        int(face_idx): position for position, face_idx in enumerate(grid.boundary_face_indices)
    }

    positions: set[int] = set()
    for k in range(k1, k2 + 1):
        for j in range(j1, j2 + 1):
            for i in range(i1, i2 + 1):
                if not (1 <= i <= nx and 1 <= j <= ny and 1 <= k <= nz):
                    warnings.warn(
                        f"{label}: cell ({i},{j},{k}) is outside grid bounds "
                        f"({nx}x{ny}x{nz}). Skipping.",
                        stacklevel=2,
                    )
                    continue

                cell_flat = int(grid.flat_index(i - 1, j - 1, k - 1))
                if not grid.is_cell_active(cell_flat):
                    warnings.warn(
                        f"{label}: cell ({i},{j},{k}) is inactive. Skipping connection.",
                        stacklevel=2,
                    )
                    continue

                matched_face: int | None = None
                for face_idx in grid.get_cell_face_indices(cell_flat):
                    normal = grid.get_face_normal_for_cell(int(face_idx), cell_flat)
                    dominant_axis = int(np.argmax(np.abs(normal)))
                    dominant_sign = np.sign(normal[dominant_axis])
                    if dominant_axis == axis and dominant_sign == sign:
                        matched_face = int(face_idx)
                        break

                if matched_face is None:
                    warnings.warn(
                        f"{label}: cell ({i},{j},{k}) has no {face_direction!r} face. Skipping.",
                        stacklevel=2,
                    )
                    continue

                position = boundary_position_by_face.get(matched_face)
                if position is None:
                    warnings.warn(
                        f"{label}: cell ({i},{j},{k})'s {face_direction!r} face is not a "
                        "grid-boundary face. It has a real interior neighbour, active or "
                        "not. Boundary conditions can only attach to genuine grid-edge "
                        "faces. Skipping.",
                        stacklevel=2,
                    )
                    continue
                positions.add(position)

    return typing.cast(IntArray[OneDimension], np.asarray(sorted(positions), dtype=np.int32))


def classify_boundary_cells(grid: Grid) -> dict[Side, IntArray[OneDimension]]:
    """
    Classify every boundary-adjacent cell of `grid` into one of the six
    `Side`s, derived from `classify_boundary_faces`.

    A cell is included on a `Side` if it owns at least one boundary face
    classified there. A corner cell genuinely touching two flanks (e.g.
    both the west and south edges) appears in both `Side`s' arrays - that's
    correct, not a bug: it really is exposed on both.

    :param grid: Grid to classify.
    :returns: Mapping from every `Side` to the *global cell indices*
        (into `grid`'s own cell numbering) of cells with at least one
        boundary face on that side, deduplicated and sorted (`np.unique`).
        A `Side` with no matching cells maps to an empty array.
    """
    faces_by_side = classify_boundary_faces(grid)
    boundary = grid.boundary_face_indices
    result: dict[Side, IntArray[OneDimension]] = {}
    for side, face_positions in faces_by_side.items():
        if len(face_positions) == 0:
            result[side] = np.empty(0, dtype=np.int32)
            continue
        global_face_indices = boundary[face_positions]
        owners = grid.face_cell_indices[global_face_indices, 0]
        result[side] = np.unique(owners).astype(np.int32)
    return result


def get_cells_on_side(
    grid: Grid,
    side: Side | str,
    *,
    classified: typing.Mapping[Side, IntArray[OneDimension]] | None = None,
) -> IntArray[OneDimension]:
    """
    Global cell indices of every boundary-adjacent cell on one flank of `grid`.

    ```python
    west_cells = get_cells_on_side(grid, "west")
    ```

    :param grid: Grid to classify. Ignored if `classified` is supplied.
    :param side: Which flank - a `Side` member, its `.value` string, or a
        common alias ('left'/'right'/'front'/'back'/'up'/'down').
    :param classified: Pre-computed `classify_boundary_cells(grid)` result.
        Pass this when calling for several sides on the same grid, to avoid
        re-classifying every boundary face once per side.
    :returns: Sorted, deduplicated global cell indices on that side. Empty
        array if the grid has no boundary cells on that flank.
    :raises ValidationError: If `side` doesn't resolve to a known `Side`.
    """
    resolved = resolve_side(side)
    cells = classified if classified is not None else classify_boundary_cells(grid)
    return cells[resolved]


@numba.njit(parallel=True, cache=True)
def _count_cell_entries(
    cell_face_offsets: npt.NDArray,
    cell_face_indices: npt.NDArray,
    face_vertex_offsets: npt.NDArray,
) -> npt.NDArray:
    """
    Count the flat-buffer entries each cell contributes to the VTK polyhedron
    face stream (type 42).

    VTK polyhedron per-cell layout:

        [total_count, n_faces, nv_f0, v0, v1, …, nv_f1, v0, …]

    So the entry count for a cell with *F* faces of vertex counts
    *V0, V1, …, V_{F-1}* is:

        1  (total_count)
      + 1  (n_faces)
      + F  (one nv_fi per face)
      + ΣVi  (all vertex indices)

    Cells with 0 faces (fully suppressed pinchouts) get count 0 and are
    excluded from the output buffer.

    :param cell_face_offsets: CSR offsets array, length `n_cells + 1`.
    :param cell_face_indices: CSR face-index data array.
    :param face_vertex_offsets: CSR offsets into vertex indices, length
        `n_faces + 1`.
    :returns: Shape `(n_cells,)` int64 entry-count array.
    """
    n_cells = cell_face_offsets.shape[0] - 1
    counts = np.zeros(n_cells, dtype=np.int64)

    for cell_idx in numba.prange(n_cells):  # type: ignore
        face_start = cell_face_offsets[cell_idx]
        face_end = cell_face_offsets[cell_idx + 1]
        n_faces = face_end - face_start
        if n_faces == 0:
            continue
        n_verts_total = np.int64(0)
        for face_idx_local in range(face_start, face_end):
            face_idx = cell_face_indices[face_idx_local]
            n_verts_total += face_vertex_offsets[face_idx + 1] - face_vertex_offsets[face_idx]
        # 1 (total_count) + 1 (n_faces) + n_faces (per-face counts) + n_verts_total
        counts[cell_idx] = np.int64(2) + np.int64(n_faces) + n_verts_total
    return counts


@numba.njit(parallel=True, cache=True)
def _fill_cell_entries(
    cell_face_offsets: npt.NDArray,
    cell_face_indices: npt.NDArray,
    face_vertex_offsets: npt.NDArray,
    face_vertex_indices: npt.NDArray,
    cell_starts: npt.NDArray,
    out: npt.NDArray,
) -> None:
    """
    Fill the pre-allocated VTK polyhedron face-stream buffer in parallel.

    Each cell owns the slice `out[cell_starts[cell_idx] : cell_starts[cell_idx] + count[cell_idx]]`
    and writes to it independently.

    Per-cell output layout:

        out[cell_starts[cell_idx]]     = total_count   (entries that follow)
        out[cell_starts[cell_idx] + 1] = n_faces
        then for each face face_idx:
            out[position]   = n_verts_fi
            out[position+1 … position+n_verts_fi] = global vertex indices

    :param cell_face_offsets: CSR offsets, length `n_cells + 1`.
    :param cell_face_indices: CSR face-index data.
    :param face_vertex_offsets: CSR offsets into vertex data, length `n_faces + 1`.
    :param face_vertex_indices: Flat vertex index data.
    :param cell_starts: Start position in `out` for each cell, derived
        from the exclusive prefix sum of `_count_cell_entries`.
        Cells with count 0 are skipped (their start value is arbitrary).
    :param out: Pre-allocated int64 output buffer of length
        `sum(_count_cell_entries(…))`.
    """
    n_cells = cell_face_offsets.shape[0] - 1

    for cell_idx in numba.prange(n_cells):  # type: ignore
        face_start = cell_face_offsets[cell_idx]
        face_end = cell_face_offsets[cell_idx + 1]
        n_faces = face_end - face_start
        if n_faces == 0:
            continue

        position = cell_starts[cell_idx]
        total_count_pos = position  # filled last
        position += 1
        out[position] = n_faces
        position += 1

        for face_idx_local in range(face_start, face_end):
            face_idx = cell_face_indices[face_idx_local]
            vertex_start = face_vertex_offsets[face_idx]
            vertex_end = face_vertex_offsets[face_idx + 1]
            n_verts = vertex_end - vertex_start
            out[position] = n_verts
            position += 1
            for vertex_idx in range(vertex_start, vertex_end):
                out[position] = face_vertex_indices[vertex_idx]
                position += 1

        # total_count = everything written after the leading integer
        out[total_count_pos] = position - total_count_pos - np.int64(1)


def make_pyvista_grid(
    grid: Grid,
    *,
    cell_data: Mapping[str, CellArray | npt.NDArray] | None = None,
) -> typing.Any:
    """
    Convert a `bores.grids.base.Grid` to a `pyvista.UnstructuredGrid`.

    Each grid cell is represented as a **VTK polyhedron (type 42)** built
    from the cell's actual face and vertex geometry. This is geometrically
    exact for all grid types; corner-point, Voronoi, Cartesian, and general
    polyhedral.

    Pinched-out cells (cells whose faces were all suppressed during
    corner-point construction) have no VTK representation and are silently
    omitted from the output mesh. Their data is also omitted from any
    attached cell-data arrays so that array lengths always match the number
    of rendered cells. The cells remain present in the source `Grid`
    object so that physics code is unaffected.


    **`MapAxes`**:

    No rotation happens here. `grid.vertex_coordinates` is already in map
    space by the time it reaches this function as the corner-point and
    Cartesian grid factories apply `MAPAXES` when `grid.metadata["map_axes"]`
    is present, unless they were explicitly built with `apply_map_axes=False`,
    in which case this renders the same local-space geometry every other
    computation on the grid uses.

    :param grid: Source `Grid`.
    :param cell_data: Optional mapping of scalar field name to a shape
        `(n_cells,)` NumPy array. Each entry is attached as a PyVista
        cell-data array and can be visualised with
        `pv_grid.plot(scalars="<scalar>")`. Arrays must have length
        `grid.n_cells`; they are automatically filtered to the valid
        (non-pinched) cells before attachment.
    :returns: A `pyvista.UnstructuredGrid` ready for rendering or
        further PyVista processing.
    :raises ImportError: If `pyvista` is not installed.
    :raises ValueError: If a `cell_data` array has length != `grid.n_cells`.

    Example:

    ```python
    from bores.grids.utils import make_pyvista_grid
    import pyvista as pv

    pressure = np.zeros((n_cells,))
    pv_grid = make_pyvista_grid(grid, cell_data={"pressure": pressure})

    pl = pv.Plotter()
    pl.add_mesh(pv_grid, scalars="pressure", show_edges=True)
    pl.show()
    ```
    """
    try:
        import pyvista as pv  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "The 'pyvista' library is required for PyVista conversion. "
            "Install it with: pip install pyvista"
        ) from exc

    n_cells = grid.n_cells

    # Copy as this function flips Z in-place below for PyVista's up
    # convention, which must not mutate `grid.vertex_coordinates` itself.
    all_points = grid.vertex_coordinates.copy()

    # Build VTK polyhedron face stream
    # First, count entries per cell
    counts = _count_cell_entries(
        cell_face_offsets=grid.cell_face_offsets.astype(np.int64),
        cell_face_indices=grid.cell_face_indices.astype(np.int64),
        face_vertex_offsets=grid.face_vertex_offsets.astype(np.int64),
    )

    # valid_cell_mask: cells that will appear in the PyVista mesh
    valid_cell_mask = counts > 0

    # Next, exclusive prefix sum -> start positions for each cell
    cell_starts = np.zeros(n_cells, dtype=np.int64)
    cell_starts[valid_cell_mask] = np.concatenate([
        [0],
        np.cumsum(counts[valid_cell_mask])[:-1],
    ])
    total_entries = int(counts.sum())

    # Lastly, fill buffer
    flat_cells = np.empty(total_entries, dtype=np.int64)
    _fill_cell_entries(
        cell_face_offsets=grid.cell_face_offsets.astype(np.int64),
        cell_face_indices=grid.cell_face_indices.astype(np.int64),
        face_vertex_offsets=grid.face_vertex_offsets.astype(np.int64),
        face_vertex_indices=grid.face_vertex_indices.astype(np.int64),
        cell_starts=cell_starts,
        out=flat_cells,
    )

    # Assemble PyVista `UnstructuredGrid`
    n_valid = int(valid_cell_mask.sum())
    cell_types = np.full(n_valid, 42, dtype=np.uint8)  # VTK_POLYHEDRON = 42
    # Negate the z-coordinate before creating the mesh. so it can be shown right side up
    # Data uses Z positive downward (depth) convention but PyVista uses Z positive upward
    # (3D graphics or coordinate system) convention
    all_points[:, 2] *= -1
    pv_grid = pv.UnstructuredGrid(flat_cells, cell_types, all_points)

    # Attach built-in geometric arrays (filtered to valid cells)
    assert grid.cell_volumes is not None
    pv_grid.cell_data["cell_volume"] = grid.cell_volumes[valid_cell_mask]
    pv_grid.cell_data["cell_depth"] = grid.cell_center_depths[valid_cell_mask]
    pv_grid.cell_data["cell_thickness"] = grid.cell_thickness[valid_cell_mask]

    # Attach caller-supplied arrays
    if cell_data:
        for name, array in cell_data.items():
            array = np.asarray(array)
            if array.shape[0] != n_cells:
                raise ValueError(
                    f"cell_data[{name!r}] has {array.shape[0]} entries but grid has {n_cells} cells."
                )
            pv_grid.cell_data[name] = array[valid_cell_mask]
    return pv_grid
