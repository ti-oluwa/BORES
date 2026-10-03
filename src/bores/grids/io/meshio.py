"""
`meshio` supported file formats' reader and writer.

The reader delegates cell-block assembly to `bores.grids.factories.polyhedral.make_polyhedral_grid` after
converting each `meshio` cell block to a `{"cell_type": ..., "connectivity": ...}` dict.

**Dependencies**:

`meshio` must be installed (`pip install meshio`).
"""

import tempfile
import typing
import warnings
from pathlib import Path

import numpy as np
import numpy.typing as npt

from bores.types import IntArray, Integer, NumberArray, OneDimension, UnitSystem

try:
    import meshio  # type: ignore[import-untyped]

except ImportError as exc:
    raise ImportError(
        "The 'meshio' library is required for VTK / generic mesh IO. "
        "Install it with: pip install meshio"
    ) from exc


from bores.errors import GridExportError, GridImportError
from bores.grids.base import CellStatus, Grid
from bores.grids.factories.polyhedral import make_polyhedral_grid
from bores.grids.io.vtu import dump_polyhedral, get_outward_face_loops, load_polyhedral
from bores.types import PathOrStr, TextOrPath

__all__ = ["dump_mesh", "load_mesh"]


# `meshio` cell type names that map to 3-D volumetric elements.
# 2-D surface elements (triangle, quad, …) are discarded during import.
VOLUMETRIC_CELL_TYPES: frozenset[str] = frozenset({
    "tetra",
    "hexahedron",
    "wedge",
    "pyramid",
    "tetra10",  # quadratic - treated as linear (first 4 nodes)
    "hexahedron20",  # quadratic - treated as linear (first 8 nodes)
})

# Map from `meshio` quadratic type to the linear equivalent and node count.
QUADRATIC_TO_LINEAR: dict[str, tuple[str, int]] = {
    "tetra10": ("tetra", 4),
    "hexahedron20": ("hexahedron", 8),
}


@typing.overload
def load_mesh(
    source: Path,
    *,
    file_format: str | None = ...,
    unit_system: UnitSystem | None = ...,
    metadata: typing.Mapping[str, typing.Any] | None = ...,
) -> Grid: ...
@typing.overload
def load_mesh(
    source: str,
    *,
    file_format: str | None = ...,
    unit_system: UnitSystem | None = ...,
    metadata: typing.Mapping[str, typing.Any] | None = ...,
) -> Grid: ...
@typing.overload
def load_mesh(
    source: bytes,
    *,
    file_format: str | None = ...,
    unit_system: UnitSystem | None = ...,
    metadata: typing.Mapping[str, typing.Any] | None = ...,
) -> Grid: ...


def load_mesh(
    source: TextOrPath,
    *,
    file_format: str | None = None,
    unit_system: UnitSystem | None = None,
    metadata: typing.Mapping[str, typing.Any] | None = None,
) -> Grid:
    """
    Load any mesh format supported by `meshio` from a path or bytes.

    This is the most general import path. `meshio` supports over 20
    formats including Abaqus, Ansys, OpenFOAM, MEDIT, etc.

    :param source: Filesystem path (`pathlib.Path` or `str`)
        or raw file bytes (`bytes`).
    :param file_format: Explicit `meshio` format string (e.g.
        `"abaqus"`, `"medit"`). If `None`, `meshio` auto-detects
        from the file extension.
    :param unit_system: Unit system the mesh coordinates are expressed in (default `FIELD`).
    :returns: A fully initialised `bores.grids.base.Grid`.
    :raises GridImportError: If the mesh cannot be read or contains no
        supported 3-D cells.
    :raises UnsupportedGridFormatError: If `meshio` is not installed or
        the format is not recognised.
    """
    return _load(source, file_format=file_format, metadata=metadata, unit_system=unit_system)


@typing.overload
def dump_mesh(
    grid: Grid,
    destination: Path,
    *,
    file_format: str,
    cell_data: dict[str, np.ndarray] | None = ...,
) -> None: ...
@typing.overload
def dump_mesh(
    grid: Grid,
    destination: None = None,
    *,
    file_format: str,
    cell_data: dict[str, np.ndarray] | None = ...,
) -> bytes: ...
@typing.overload
def dump_mesh(
    grid: Grid,
    destination: str,
    *,
    file_format: str,
    cell_data: dict[str, np.ndarray] | None = ...,
) -> None: ...


def dump_mesh(
    grid: Grid,
    destination: PathOrStr | None = None,
    *,
    file_format: str,
    cell_data: dict[str, np.ndarray] | None = None,
) -> bytes | None:
    """
    Write a `bores.grids.base.Grid` to any format supported by `meshio`.

    This is the most general export path.  `meshio` supports over 20
    formats including Abaqus, Ansys, MEDIT, OpenFOAM, etc.

    Note:
        `file_format` is mandatory here (unlike `load_mesh`) because
        when `destination` is `None` there is no file extension for
        `meshio` to infer from.  Being explicit also avoids surprises when
        writing to a path with an unusual or missing extension.

    :param grid: The grid to serialise.
    :param destination: One of:

        - `pathlib.Path` or `str` path - write to file and
          return `None`.
        - `None` - return the serialised content as `bytes`.

    :param file_format: Explicit `meshio` format string (e.g. `"abaqus"`,
        `"medit"`, `"vtk"`, `"vtu"`). Always required.
    :param cell_data: Optional mapping of scalar field name to shape
        `(n_cells,)` array.
    :returns: `bytes` when `destination` is `None`; `None` otherwise.
    :raises GridExportError: If serialisation fails.
    :raises UnsupportedGridFormatError: If `meshio` is not installed or
        the format is not recognised.
    """
    return _dump(grid, destination=destination, file_format=file_format, cell_data=cell_data)


def _load(
    source: TextOrPath,
    *,
    file_format: str | None = None,
    metadata: typing.Mapping[str, typing.Any] | None = None,
    unit_system: UnitSystem | None = None,
) -> Grid:
    """
    Load any `meshio`-supported mesh and convert to a
    `bores.grids.base.Grid`.

    :param source: Path or bytes source.
    :param file_format: Explicit `meshio` format string or `None` for
        auto-detection.
    :returns: A fully initialised `bores.grids.base.Grid`.
    :raises GridImportError: If the mesh contains no supported 3-D cells.
    """
    if isinstance(source, bytes):
        if file_format is None:
            raise GridImportError(
                "`file_format` must be specified when loading from raw bytes "
                "(e.g. file_format='vtk' or 'vtu')."
            )

        if file_format == "vtu":
            polyhedral = load_polyhedral(source, metadata=metadata, unit_system=unit_system)
            if polyhedral is not None:
                return polyhedral

        # `meshio` readers need a real path; they cannot read from an in-memory buffer.
        with tempfile.TemporaryDirectory() as directory:
            temporary_path = Path(directory) / f"grid.{file_format}"
            temporary_path.write_bytes(source)
            try:
                mesh = meshio.read(str(temporary_path), file_format=file_format)
            except Exception as exc:
                raise GridImportError(f"`meshio` failed to read bytes: {exc}") from exc
    else:
        path = Path(source)  # type: ignore[arg-type]
        if not path.is_file():
            raise GridImportError(f"Mesh file not found: {path!r}")
        if file_format == "vtu" or (file_format is None and path.suffix.lower() == ".vtu"):
            polyhedral = load_polyhedral(
                path.read_bytes(), metadata=metadata, unit_system=unit_system
            )
            if polyhedral is not None:
                return polyhedral
        try:
            mesh = meshio.read(str(path), file_format=file_format)
        except Exception as exc:
            raise GridImportError(f"`meshio` failed to read {path!r}: {exc}") from exc

    return _mesh_to_grid(mesh, metadata=metadata, unit_system=unit_system)


def _mesh_to_grid(
    mesh: meshio.Mesh,
    metadata: typing.Mapping[str, typing.Any] | None = None,
    unit_system: UnitSystem | None = None,
) -> Grid:
    """
    Convert a `meshio.Mesh` object to a `bores.grids.base.Grid`.

    Only volumetric (3-D) cell types are retained.  Quadratic elements are
    reduced to their linear counterparts by discarding mid-side nodes.

    :param mesh: A `meshio.Mesh` instance.
    :param unit_system: Unit system the coordinates are expressed in (default `FIELD`).
    :returns: A fully initialised `bores.grids.base.Grid`.
    :raises GridImportError: If no supported 3-D cell blocks are found.
    """
    points = np.asarray(mesh.points, dtype=np.float64)
    if points.shape[1] == 2:
        # 2-D mesh: promote to 3-D with z = 0
        points = np.column_stack([points, np.zeros(len(points))])

    cell_blocks = []
    for block in mesh.cells:
        cell_type = block.type
        connectivity = np.asarray(block.data, dtype=np.int32)

        if cell_type in QUADRATIC_TO_LINEAR:
            linear_type, n_linear_verts = QUADRATIC_TO_LINEAR[cell_type]
            connectivity = connectivity[:, :n_linear_verts]
            cell_type = linear_type

        if cell_type not in VOLUMETRIC_CELL_TYPES:
            continue  # skip surface / line elements

        cell_blocks.append({"cell_type": cell_type, "connectivity": connectivity})

    if not cell_blocks:
        raise GridImportError(
            "Mesh contains no supported 3-D cell types "
            f"(supported: {sorted(VOLUMETRIC_CELL_TYPES)})."
        )

    meta = {"source_format": "meshio"}
    if metadata:
        meta.update(metadata)
    try:
        return make_polyhedral_grid(
            vertex_coordinates=points,  # type: ignore[arg-type]
            cell_blocks=cell_blocks,
            metadata=meta,
            unit_system=unit_system if unit_system is not None else UnitSystem.FIELD,
        )
    except Exception as exc:
        raise GridImportError(f"Failed to build Grid from `meshio` cell blocks: {exc}") from exc


def _get_vtk_cell_nodes(grid: Grid, cell: Integer) -> tuple[str, IntArray[OneDimension]] | None:
    """
    Get vertex indices of a cell in VTK node order, if the cell is a hexahedron or a tetrahedron.

    A hexahedron has six quadrilateral faces over eight vertices, each shared by three faces;
    a tetrahedron has four triangular faces over four vertices.

    :param grid: Source grid.
    :param cell: Cell index.
    :returns: `(`meshio` cell type, node indices)`, or `None` for any other cell shape.
    """
    loops = get_outward_face_loops(grid, cell)
    unique = np.unique(np.concatenate(loops)) if loops else np.empty(0, dtype=np.int32)

    if len(loops) == 4 and all(len(loop) == 3 for loop in loops) and len(unique) == 4:
        base = loops[0]
        apex = next(vertex for vertex in unique if vertex not in base)
        return "tetra", typing.cast(
            IntArray[OneDimension], np.array([base[2], base[1], base[0], apex], dtype=np.int32)
        )

    if len(loops) == 6 and all(len(loop) == 4 for loop in loops) and len(unique) == 8:
        faces_per_vertex = np.bincount(np.concatenate(loops), minlength=int(unique.max()) + 1)
        if not (faces_per_vertex[unique] == 3).all():
            return None

        base = loops[0]
        opposite = next((loop for loop in loops[1:] if not set(loop) & set(base)), None)
        if opposite is None:
            return None

        edges: set[tuple[int, int]] = set()
        for loop in loops:
            for index, vertex in enumerate(loop):
                following = loop[(index + 1) % 4]
                edges.add((int(vertex), int(following)))
                edges.add((int(following), int(vertex)))

        top = []
        for vertex in base:
            partners = [w for w in opposite if (int(vertex), int(w)) in edges]
            if len(partners) != 1:
                return None
            top.append(partners[0])

        # The outward winding of the base points away from the top; VTK wants it toward the top.
        order = [3, 2, 1, 0]
        nodes = [base[i] for i in order] + [top[i] for i in order]
        return "hexahedron", typing.cast(IntArray[OneDimension], np.array(nodes, dtype=np.int32))

    return None


def _has_cells_needing_polyhedra(grid: Grid) -> bool:
    """
    Whether any active cell is neither a hexahedron nor a tetrahedron.

    :param grid: Source grid.
    :returns: `True` if the grid cannot be written exactly with standard VTK cell types.
    """
    assert grid.cell_statuses is not None
    return any(
        _get_vtk_cell_nodes(grid, int(cell)) is None
        for cell in np.flatnonzero(grid.cell_statuses == int(CellStatus.ACTIVE))
    )


def _grid_to_mesh(grid: Grid, *, cell_data: dict[str, npt.NDArray] | None) -> typing.Any:
    """
    Convert a `bores.grids.base.Grid` to a `meshio.Mesh`.

    Cells whose faces form a hexahedron or a tetrahedron are exported exactly, sharing the
    grid's own vertices. Any other cell (for example a Voronoi polyhedron, or a hexahedron cut
    by a fault into more than six faces) is exported as the hexahedron of its bounding box and
    a warning reports how many there were. Writing such a grid as `vtu` bypasses this and
    exports every cell exactly as a polyhedron. Only active cells are exported; the `cell_index`
    cell field holds each exported cell's index in the grid.

    :param grid: Source grid.
    :param cell_data: Optional per-cell data fields.
    :returns: A `meshio.Mesh` instance.
    """
    n_cells = grid.n_cells
    assert grid.cell_statuses is not None
    exported_cells = np.flatnonzero(grid.cell_statuses == int(CellStatus.ACTIVE))

    blocks: dict[str, tuple[list[Integer], list[IntArray[OneDimension]]]] = {
        "hexahedron": ([], []),
        "tetra": ([], []),
    }
    extra_points: list[NumberArray[OneDimension]] = []
    next_point = len(grid.vertex_coordinates)
    n_approximated = 0
    for cell in exported_cells:
        exact = _get_vtk_cell_nodes(grid, int(cell))
        if exact is not None:
            cell_type, nodes = exact
        else:
            n_approximated += 1
            low, high = grid.cell_min_xyz[cell], grid.cell_max_xyz[cell]
            corners = [
                (low[0], low[1], low[2]),
                (high[0], low[1], low[2]),
                (high[0], high[1], low[2]),
                (low[0], high[1], low[2]),
                (low[0], low[1], high[2]),
                (high[0], low[1], high[2]),
                (high[0], high[1], high[2]),
                (low[0], high[1], high[2]),
            ]
            extra_points.extend(
                typing.cast(NumberArray[OneDimension], np.array(corner, dtype=np.float64))
                for corner in corners
            )
            nodes = np.arange(next_point, next_point + 8, dtype=np.int32)
            next_point += 8
            cell_type = "hexahedron"

        blocks[cell_type][0].append(int(cell))
        blocks[cell_type][1].append(nodes)

    if n_approximated:
        warnings.warn(
            f"{n_approximated} cell(s) are not hexahedra or tetrahedra and were exported as "
            "the hexahedron of their bounding box.",
            stacklevel=3,
        )

    points = grid.vertex_coordinates
    if extra_points:
        points = np.vstack([points, np.asarray(extra_points)])

    fields: dict[str, npt.NDArray] = {}
    if cell_data:
        for field_name, field_array in cell_data.items():
            array = np.asarray(field_array, dtype=np.float64)
            if array.shape[0] != n_cells:
                raise GridExportError(
                    f"cell_data[{field_name!r}] has {array.shape[0]} entries "
                    f"but grid has {n_cells} cells."
                )
            fields[field_name] = array

    fields["cell_index"] = np.arange(n_cells, dtype=np.int64)

    cells = []
    meshio_cell_data: dict[str, list[npt.NDArray]] = {name: [] for name in fields}
    for cell_type, (indices, nodes_list) in blocks.items():
        if not indices:
            continue
        cells.append((cell_type, np.asarray(nodes_list, dtype=np.int32)))
        for name, array in fields.items():
            meshio_cell_data[name].append(array[np.asarray(indices)])

    return meshio.Mesh(
        points=points,
        cells=cells,
        cell_data=meshio_cell_data,  # type: ignore
    )


def _dump(
    grid: Grid,
    *,
    destination: PathOrStr | None,
    file_format: str,
    cell_data: dict[str, npt.NDArray] | None,
) -> bytes | None:
    """
    Write a `bores.grids.base.Grid` using `meshio`.

    :param grid: Source grid.
    :param destination: File path or `None` to return bytes.
    :param file_format: `meshio` format string (`"vtk"` or `"vtu"`).
    :param cell_data: Optional per-cell data mapping.
    :returns: `bytes` if `destination` is `None`; `None` otherwise.
    :raises GridExportError: If serialisation fails.
    """
    if file_format == "vtu" and _has_cells_needing_polyhedra(grid):
        payload = dump_polyhedral(grid, cell_data=cell_data)
        if destination is None:
            return payload
        Path(destination).write_bytes(payload)
        return None

    try:
        mesh = _grid_to_mesh(grid, cell_data=cell_data)
    except GridExportError:
        raise
    except Exception as exc:
        raise GridExportError(f"Failed to convert grid to `meshio.Mesh`: {exc}") from exc

    if destination is None:
        # `meshio` writers need a real path; they cannot write to an in-memory buffer.
        with tempfile.TemporaryDirectory() as directory:
            temporary_path = Path(directory) / f"grid.{file_format}"
            try:
                meshio.write(str(temporary_path), mesh, file_format=file_format)
            except Exception as exc:
                raise GridExportError(f"`meshio` failed to write {file_format!r}: {exc}") from exc
            return temporary_path.read_bytes()

    path = Path(destination)
    try:
        meshio.write(str(path), mesh, file_format=file_format)
    except Exception as exc:
        raise GridExportError(
            f"`meshio` failed to write {file_format!r} to {path!r}: {exc}"
        ) from exc
    return None
