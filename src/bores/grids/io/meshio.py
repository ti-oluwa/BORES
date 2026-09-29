"""
`meshio` supported file formats' reader and writer.

The reader delegates cell-block assembly to `bores.grids.factories.polyhedral.make_polyhedral_grid` after
converting each `meshio` cell block to a `{"cell_type": ..., "connectivity": ...}` dict.

**Dependencies**:

`meshio` must be installed (`pip install meshio`).
"""

import io
import typing
from pathlib import Path

import numpy as np
import numpy.typing as npt

from bores.types import UnitSystem

try:
    import meshio  # type: ignore[import-untyped]

except ImportError as exc:
    raise ImportError(
        "The 'meshio' library is required for VTK / generic mesh IO. "
        "Install it with: pip install meshio"
    ) from exc


from bores.errors import GridExportError, GridImportError
from bores.grids.base import Grid
from bores.grids.factories.polyhedral import make_polyhedral_grid
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
    :returns: A fully initialised `bores.grids.base.Grid`.
    :raises GridImportError: If the mesh cannot be read or contains no
        supported 3-D cells.
    :raises UnsupportedGridFormatError: If `meshio` is not installed or
        the format is not recognised.
    """
    grid = _load(source, file_format=file_format, metadata=metadata)
    return grid.convert(unit_system) if unit_system is not None else grid


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
                "file_format must be specified when loading from raw bytes "
                "(e.g. file_format='vtk' or 'vtu')."
            )

        buffer = io.BytesIO(source)
        try:
            mesh = meshio.read(buffer, file_format=file_format)
        except Exception as exc:
            raise GridImportError(f"meshio failed to read bytes: {exc}") from exc
    else:
        path = Path(source)  # type: ignore[arg-type]
        if not path.is_file():
            raise GridImportError(f"Mesh file not found: {path!r}")
        try:
            mesh = meshio.read(str(path), file_format=file_format)
        except Exception as exc:
            raise GridImportError(f"meshio failed to read {path!r}: {exc}") from exc

    return _mesh_to_grid(mesh, metadata=metadata)


def _mesh_to_grid(
    mesh: meshio.Mesh, metadata: typing.Mapping[str, typing.Any] | None = None
) -> Grid:
    """
    Convert a `meshio.Mesh` object to a `bores.grids.base.Grid`.

    Only volumetric (3-D) cell types are retained.  Quadratic elements are
    reduced to their linear counterparts by discarding mid-side nodes.

    :param mesh: A `meshio.Mesh` instance.
    :returns: A fully initialised `bores.grids.base.Grid`.
    :raises GridImportError: If no supported 3-D cell blocks are found.
    """
    points = np.asarray(mesh.points, dtype=np.float64, copy=False)
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
        )
    except Exception as exc:
        raise GridImportError(f"Failed to build Grid from meshio cell blocks: {exc}") from exc


def _grid_to_mesh(grid: Grid, *, cell_data: dict[str, npt.NDArray] | None) -> typing.Any:
    """
    Convert a `bores.grids.base.Grid` to a `meshio.Mesh`.

    Since `bores.grids.base.Grid` stores faces rather than cells
    directly, we reconstruct an approximate hexahedral cell mesh where each
    grid cell is represented by a single hex cell derived from its bounding
    box.  This preserves cell count and approximate geometry for
    visualisation purposes.

    For grids produced by the Cartesian factory, the bounding-box hex will
    exactly match the original cell.  For Voronoi / polyhedral grids, the hex
    is an axis-aligned approximation.

    :param grid: Source grid.
    :param cell_data: Optional per-cell data fields.
    :returns: A `meshio.Mesh` instance.
    """
    n_cells = grid.n_cells

    # Build 8 bounding-box vertices per cell: (n_cells * 8, 3)
    vertices_per_cell = 8
    all_vertices = np.empty((n_cells * vertices_per_cell, 3), dtype=np.float64)
    connectivity = np.empty((n_cells, vertices_per_cell), dtype=np.int32)

    for cell_idx in range(n_cells):
        low = grid.cell_min_xyz[cell_idx]
        high = grid.cell_max_xyz[cell_idx]
        base = cell_idx * vertices_per_cell
        # VTK hex vertex order: bottom face CCW then top face CCW
        all_vertices[base + 0] = [low[0], low[1], low[2]]
        all_vertices[base + 1] = [high[0], low[1], low[2]]
        all_vertices[base + 2] = [high[0], high[1], low[2]]
        all_vertices[base + 3] = [low[0], high[1], low[2]]
        all_vertices[base + 4] = [low[0], low[1], high[2]]
        all_vertices[base + 5] = [high[0], low[1], high[2]]
        all_vertices[base + 6] = [high[0], high[1], high[2]]
        all_vertices[base + 7] = [low[0], high[1], high[2]]
        connectivity[cell_idx] = np.arange(base, base + vertices_per_cell, dtype=np.int32)

    meshio_cell_data: dict[str, list[np.ndarray]] = {}
    if cell_data:
        for field_name, field_array in cell_data.items():
            array = np.asarray(field_array, dtype=np.float64, copy=False)
            if array.shape[0] != n_cells:
                raise GridExportError(
                    f"cell_data[{field_name!r}] has {array.shape[0]} entries "
                    f"but grid has {n_cells} cells."
                )
            meshio_cell_data[field_name] = [array]

    return meshio.Mesh(
        points=all_vertices,
        cells=[("hexahedron", connectivity)],
        cell_data=meshio_cell_data if meshio_cell_data else {},  # type: ignore
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
    try:
        mesh = _grid_to_mesh(grid, cell_data=cell_data)
    except GridExportError:
        raise
    except Exception as exc:
        raise GridExportError(f"Failed to convert grid to meshio.Mesh: {exc}") from exc

    if destination is None:
        buf = io.BytesIO()
        try:
            meshio.write(buf, mesh, file_format=file_format)
        except Exception as exc:
            raise GridExportError(
                f"meshio failed to write {file_format!r} to buffer: {exc}"
            ) from exc
        return buf.getvalue()

    path = Path(destination)
    try:
        meshio.write(str(path), mesh, file_format=file_format)
    except Exception as exc:
        raise GridExportError(
            f"meshio failed to write {file_format!r} to {path!r}: {exc}"
        ) from exc
    return None
