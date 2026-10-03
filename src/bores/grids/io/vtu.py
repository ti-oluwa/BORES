"""
VTK XML unstructured grid (`.vtu`) polyhedron reader and writer.

A grid whose cells are not all hexahedra or tetrahedra (Voronoi polyhedra, or hexahedra cut by a
fault into more than six faces) cannot be written exactly as standard VTK cell types. VTK has a
native polyhedron cell (type 42) that lists the faces of each cell, and this module writes and
reads it directly, with ASCII data arrays, so the grid's faces survive a round trip exactly.
"""

import typing
import xml.etree.ElementTree as ElementTree

import numpy as np
import numpy.typing as npt

from bores.errors import GridExportError, GridImportError
from bores.grids.base import CellStatus, Grid
from bores.grids.factories.base import build_csr_face_arrays
from bores.types import IntArray, Integer, NumberArray, OneDimension, UnitSystem

__all__ = ["dump_polyhedral", "get_outward_face_loops", "load_polyhedral"]

VTK_POLYHEDRON = 42
"""VTK cell type identifier of a polyhedron."""


def get_outward_face_loops(grid: Grid, cell: Integer) -> list[IntArray[OneDimension]]:
    """
    Vertex loops of every face of a cell, each wound so the face normal points out of the cell.

    :param grid: Source grid.
    :param cell: Cell index.
    :returns: One vertex index array per face.
    """
    loops = []
    for face in grid.get_cell_face_indices(cell):
        start = grid.face_vertex_offsets[face]
        end = grid.face_vertex_offsets[face + 1]
        loop = grid.face_vertex_indices[start:end]
        loops.append(loop if grid.face_cell_indices[face, 0] == cell else loop[::-1])
    return typing.cast(list[IntArray[OneDimension]], loops)


def format_array(values: npt.ArrayLike, *, kind: typing.Literal["int", "float"]) -> str:
    """
    Text of a flat array for a VTK `DataArray` element.

    :param values: Values to write.
    :param kind: `"int"` for integers, `"float"` for floating-point values at full precision.
    :returns: Whitespace separated values.
    """
    flat = np.asarray(values).ravel()
    if kind == "int":
        return " ".join(str(int(value)) for value in flat)
    return " ".join(f"{float(value):.17g}" for value in flat)


def dump_polyhedral(
    grid: Grid,
    *,
    cell_data: typing.Mapping[str, npt.NDArray] | None = None,
) -> bytes:
    """
    Write the active cells of a grid as VTK polyhedra.

    Every face of every active cell is written, so the grid is reproduced exactly. A
    `cell_index` cell field holds each written cell's index in the grid. Each cell's volume,
    centroid and bounding box are written as well (`bores_cell_volume`, `bores_cell_centroid`,
    `bores_cell_min`, `bores_cell_max`), because a cell next to an inactive cell has an open
    surface and its geometry cannot be recomputed from its faces alone.

    :param grid: Source grid.
    :param cell_data: Optional mapping of field name to a `(n_cells,)` array.
    :returns: The XML document as bytes.
    :raises GridExportError: If a cell data field has the wrong length or the grid has no
        active cells.
    """
    assert grid.cell_statuses is not None
    cells = np.flatnonzero(grid.cell_statuses == int(CellStatus.ACTIVE))
    if len(cells) == 0:
        raise GridExportError("The grid has no active cells to write.")

    loops_per_cell = [get_outward_face_loops(grid, int(cell)) for cell in cells]
    used = np.unique(np.concatenate([np.concatenate(loops) for loops in loops_per_cell]))
    renumber = np.full(len(grid.vertex_coordinates), -1, dtype=np.int64)
    renumber[used] = np.arange(len(used))

    connectivity: list[int] = []
    offsets: list[int] = []
    faces: list[int] = []
    face_offsets: list[int] = []
    for loops in loops_per_cell:
        points = np.unique(np.concatenate(loops))
        connectivity.extend(renumber[points].tolist())
        offsets.append(len(connectivity))
        faces.append(len(loops))
        for loop in loops:
            faces.append(len(loop))
            faces.extend(renumber[loop].tolist())
        face_offsets.append(len(faces))

    fields: dict[str, npt.NDArray] = {}
    for name, values in (cell_data or {}).items():
        array = np.asarray(values, dtype=np.float64)
        if array.shape[0] != grid.n_cells:
            raise GridExportError(
                f"cell_data[{name!r}] has {array.shape[0]} entries but grid has {grid.n_cells} cells."
            )
        fields[name] = array[cells]

    fields["cell_index"] = cells.astype(np.int64)
    assert grid.cell_volumes is not None and grid.cell_centroids is not None
    geometry = {
        "bores_cell_volume": grid.cell_volumes[cells],
        "bores_cell_centroid": grid.cell_centroids[cells],
        "bores_cell_min": grid.cell_min_xyz[cells],
        "bores_cell_max": grid.cell_max_xyz[cells],
    }

    def data_array(
        name: str | None,
        dtype: str,
        values: npt.ArrayLike,
        kind: typing.Literal["int", "float"],
        *,
        components: int = 1,
    ) -> ElementTree.Element:
        element = ElementTree.Element(
            "DataArray", type=dtype, NumberOfComponents=str(components), format="ascii"
        )
        if name is not None:
            element.set("Name", name)
        element.text = format_array(values, kind=kind)
        return element

    root = ElementTree.Element(
        "VTKFile", type="UnstructuredGrid", version="1.0", byte_order="LittleEndian"
    )
    piece = ElementTree.SubElement(
        ElementTree.SubElement(root, "UnstructuredGrid"),
        "Piece",
        NumberOfPoints=str(len(used)),
        NumberOfCells=str(len(cells)),
    )
    ElementTree.SubElement(piece, "Points").append(
        data_array(None, "Float64", grid.vertex_coordinates[used], "float", components=3)
    )
    cell_element = ElementTree.SubElement(piece, "Cells")
    cell_element.append(data_array("connectivity", "Int64", connectivity, "int"))
    cell_element.append(data_array("offsets", "Int64", offsets, "int"))
    cell_element.append(data_array("types", "UInt8", [VTK_POLYHEDRON] * len(cells), "int"))
    cell_element.append(data_array("faces", "Int64", faces, "int"))
    cell_element.append(data_array("faceoffsets", "Int64", face_offsets, "int"))
    data_element = ElementTree.SubElement(piece, "CellData")
    for name, array in fields.items():
        is_integer = np.issubdtype(array.dtype, np.integer)
        data_element.append(
            data_array(
                name,
                dtype="Int64" if is_integer else "Float64",
                values=array,
                kind="int" if is_integer else "float",
            )
        )

    for name, array in geometry.items():
        data_element.append(
            data_array(
                name,
                dtype="Float64",
                values=array,
                kind="float",
                components=1 if array.ndim == 1 else 3,
            )
        )
    return ElementTree.tostring(root, encoding="utf-8", xml_declaration=True)


def read_stored_geometry(
    piece: ElementTree.Element, *, n_cells: Integer
) -> dict[str, NumberArray[typing.Any]]:
    """
    Per-cell geometry stored by `dump_polyhedral`, as keyword arguments for `Grid`.

    :param piece: The `Piece` element of the document.
    :param n_cells: Number of cells in the piece.
    :returns: Mapping with `cell_volumes`, `cell_centroids`, `cell_min_xyz` and `cell_max_xyz`
        when all four are present and consistent, otherwise empty.
    """
    names = {
        "bores_cell_volume": ("cell_volumes", 1),
        "bores_cell_centroid": ("cell_centroids", 3),
        "bores_cell_min": ("cell_min_xyz", 3),
        "bores_cell_max": ("cell_max_xyz", 3),
    }
    found: dict[str, NumberArray[typing.Any]] = {}
    for element in piece.findall("./CellData/DataArray"):
        entry = names.get(element.get("Name", ""))
        if entry is None or element.get("format", "ascii") != "ascii":
            continue

        keyword, components = entry
        values = np.array((element.text or "").split(), dtype=np.float64)
        if values.size != n_cells * components:
            return {}
        found[keyword] = values if components == 1 else values.reshape(n_cells, components)
    return found if len(found) == len(names) else {}


def load_polyhedral(
    payload: bytes,
    *,
    metadata: typing.Mapping[str, typing.Any] | None = None,
    unit_system: UnitSystem | None = None,
) -> Grid | None:
    """
    Read a `.vtu` document made of polyhedron cells into a grid.

    :param payload: The XML document.
    :param metadata: Optional metadata attached to the grid.
    :param unit_system: Unit system the coordinates are expressed in (default `FIELD`).
    :returns: The grid, or `None` if the document does not consist of polyhedron cells (so a
        generic reader should handle it).
    :raises GridImportError: If the document is malformed or uses binary data arrays.
    """
    try:
        root = ElementTree.fromstring(payload)
    except ElementTree.ParseError:
        return None  # not an XML document, so not a `.vtu` file

    piece = root.find("./UnstructuredGrid/Piece")
    cells_element = piece.find("Cells") if piece is not None else None
    if piece is None or cells_element is None:
        return None

    arrays: dict[str, ElementTree.Element] = {
        element.get("Name", ""): element for element in cells_element.findall("DataArray")
    }
    types_element = arrays.get("types")
    if (
        types_element is None
        or types_element.get("format", "ascii") != "ascii"
        or not (types_element.text or "").strip()
    ):
        return None  # binary documents are written by generic tools, not by this module

    cell_types = np.array((types_element.text or "").split(), dtype=np.int64)
    if not (cell_types == VTK_POLYHEDRON).all():
        return None
    if "faces" not in arrays or "faceoffsets" not in arrays:
        raise GridImportError("Polyhedron cells need `faces` and `faceoffsets` arrays.")

    def read_ints(name: str) -> npt.NDArray[np.int64]:
        element = arrays[name]
        if element.get("format", "ascii") != "ascii":
            raise GridImportError(
                f"Only ASCII data arrays are supported for polyhedron cells ({name!r} is "
                f"{element.get('format')!r})."
            )
        return np.array((element.text or "").split(), dtype=np.int64)

    points_element = piece.find("./Points/DataArray")
    if points_element is None or points_element.get("format", "ascii") != "ascii":
        raise GridImportError("The `Points` array must be present and use ASCII data.")

    points = np.array((points_element.text or "").split(), dtype=np.float64).reshape(-1, 3)
    faces = read_ints("faces")
    face_offsets = read_ints("faceoffsets")
    per_cell_faces: list[list[IntArray[OneDimension]]] = []
    start = 0
    for end in face_offsets:
        position = start
        n_faces = int(faces[position])
        position += 1
        cell_faces: list[IntArray[OneDimension]] = []
        for _ in range(n_faces):
            length = int(faces[position])
            cell_faces.append(
                typing.cast(
                    IntArray[OneDimension],
                    faces[position + 1 : position + 1 + length].astype(np.int32),
                )
            )
            position += 1 + length

        if position != end:
            raise GridImportError("The `faces` and `faceoffsets` arrays are inconsistent.")
        per_cell_faces.append(cell_faces)
        start = int(end)

    try:
        vertex_coordinates, face_vertex_indices, face_vertex_offsets, face_cell_indices = (
            build_csr_face_arrays(
                vertex_coordinates=typing.cast(NumberArray[OneDimension], points),  # type: ignore[arg-type]
                per_cell_face_vertex_lists=per_cell_faces,  # type: ignore[arg-type]
            )
        )
        grid_metadata = {"source_format": "vtu_polyhedra"}
        if metadata:
            grid_metadata.update(metadata)

        stored = read_stored_geometry(piece, n_cells=len(per_cell_faces))
        return Grid(
            vertex_coordinates=vertex_coordinates,
            face_vertex_indices=face_vertex_indices,
            face_vertex_offsets=face_vertex_offsets,
            face_cell_indices=face_cell_indices,
            unit_system=unit_system if unit_system is not None else UnitSystem.FIELD,
            metadata=grid_metadata,
            **stored,  # type: ignore[arg-type]
        )
    except Exception as exc:
        raise GridImportError(f"Failed to build a grid from the polyhedron cells: {exc}") from exc
