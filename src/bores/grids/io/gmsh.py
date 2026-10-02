"""
Gmsh `.msh` mesh reader.

Reads Gmsh ASCII mesh files (format version 2.2) and converts 3-D volumetric
elements to a `bores.grids.base.Grid` via the
`bores.grids.factories.polyhedral.make_polyhedral_grid` factory.

**Supported element types**:

=======  ========  =========
Gmsh ID  Name       Vertices
=======  ========  =========
4        tetra      4
5        hexahedron 8
6        prism      6
7        pyramid    5
=======  ========  =========

2-D surface elements (triangles, quads, lines) are silently skipped.

**References**:

Gmsh reference manual - MSH file format, v2.
"""

import re
import typing
from pathlib import Path

import numpy as np

from bores.errors import GridImportError, UnsupportedGridFormatError
from bores.grids.base import Grid
from bores.grids.factories.polyhedral import make_polyhedral_grid
from bores.types import TextOrPath, UnitSystem

__all__ = ["load_msh"]


# Gmsh element type ID -> (n_vertices, cell_type_name)
# Only 3-D volumetric elements supported for simulation grids.
GMSH_ELEMENT_TYPES: dict[int, tuple[int, str]] = {
    4: (4, "tetra"),
    5: (8, "hexahedron"),
    6: (6, "wedge"),
    7: (5, "pyramid"),
}

# Higher-order 3-D element type ID -> (n_corner_vertices, linear_cell_type_name). Gmsh lists the
# corner nodes first, so a higher-order element reduces to its linear counterpart by keeping
# only those nodes.
GMSH_HIGHER_ORDER_ELEMENT_TYPES: dict[int, tuple[int, int, str]] = {
    11: (10, 4, "tetra"),
    12: (27, 8, "hexahedron"),
    13: (18, 6, "wedge"),
    14: (14, 5, "pyramid"),
    17: (20, 8, "hexahedron"),
    18: (15, 6, "wedge"),
    19: (13, 5, "pyramid"),
}


@typing.overload
def load_msh(
    source: Path,
    *,
    encoding: str = ...,
    unit_system: UnitSystem | None = ...,
    metadata: typing.Mapping[str, typing.Any] | None = ...,
) -> Grid: ...
@typing.overload
def load_msh(
    source: str,
    *,
    encoding: str = ...,
    unit_system: UnitSystem | None = ...,
    metadata: typing.Mapping[str, typing.Any] | None = ...,
) -> Grid: ...
@typing.overload
def load_msh(
    source: bytes,
    *,
    encoding: str = ...,
    unit_system: UnitSystem | None = ...,
    metadata: typing.Mapping[str, typing.Any] | None = ...,
) -> Grid: ...


def load_msh(
    source: TextOrPath,
    *,
    encoding: str = "utf-8",
    unit_system: UnitSystem | None = None,
    metadata: typing.Mapping[str, typing.Any] | None = None,
) -> Grid:
    """
    Load a Gmsh `.msh` file (ASCII format 2.2) from a path, string, or bytes.

    :param source: One of:

        - `pathlib.Path` - path to a `.msh` file.
        - `str` - filesystem path or raw `.msh` text content.
        - `bytes` - raw `.msh` content.

    :param encoding: Text encoding for `bytes` / file input (default
        `"utf-8"`).
    :param unit_system: Unit system the mesh coordinates are expressed in (default `FIELD`).
    :returns: A fully initialised `bores.grids.base.Grid` containing
        only the 3-D volumetric elements.
    :raises GridImportError: If the file is malformed or contains no
        supported 3-D elements.
    :raises UnsupportedGridFormatError: If the Gmsh format version is not
        2.2.
    """
    text = resolve_source(source, encoding=encoding)
    return parse_msh(text, metadata=metadata, unit_system=unit_system)


def resolve_source(source: TextOrPath, *, encoding: str) -> str:
    """
    Coerce `source` to a plain text string.

    :param source: Path, raw string, or bytes.
    :param encoding: Byte decoding encoding.
    :returns: Raw `.msh` text.
    :raises GridImportError: If a path cannot be read.
    """
    if isinstance(source, bytes):
        return source.decode(encoding)
    if isinstance(source, Path):
        try:
            return source.read_text(encoding=encoding)
        except OSError as exc:
            raise GridImportError(f"Cannot read .msh file {source!r}: {exc}") from exc

    assert isinstance(source, str)
    if source.lstrip().startswith("$MeshFormat"):
        return source

    candidate = Path(source)  # type: ignore[arg-type]
    if candidate.is_file():
        try:
            return candidate.read_text(encoding=encoding)
        except OSError as exc:
            raise GridImportError(f"Cannot read .msh file {source!r}: {exc}") from exc
    raise GridImportError(f"Invalid source: {source!r}")


def extract_section(text: str, section_name: str) -> str | None:
    """
    Extract the body of a `$SectionName … $EndSectionName` block.

    :param text: Full `.msh` text.
    :param section_name: Section name without the `$` prefix (e.g.
        `"Nodes"`).
    :returns: Section body text, or `None` if the section is absent.
    """
    pattern = re.compile(
        r"\$" + re.escape(section_name) + r"\s*(.*?)\s*\$End" + re.escape(section_name),
        re.DOTALL,
    )
    m = pattern.search(text)
    return m.group(1).strip() if m else None


def parse_msh(
    text: str,
    metadata: typing.Mapping[str, typing.Any] | None = None,
    unit_system: UnitSystem | None = None,
) -> Grid:
    """
    Parse a Gmsh MSH v2.2 ASCII text blob into a `bores.grids.base.Grid`.

    :param text: Raw `.msh` text.
    :param unit_system: Unit system the coordinates are expressed in (default `FIELD`).
    :returns: A fully initialised `bores.grids.base.Grid`.
    :raises UnsupportedGridFormatError: If the mesh format is not version 2.
    :raises GridImportError: If the `$Nodes` or `$Elements` sections are
        malformed or no 3-D elements are found.
    """
    # $MeshFormat
    mesh_format_section = extract_section(text, "MeshFormat")
    if mesh_format_section is not None:
        version_str = mesh_format_section.split()[0]
        major = int(version_str.split(".")[0])
        if major != 2:
            raise UnsupportedGridFormatError(
                f"Only Gmsh MSH format version 2 is supported; got version {version_str!r}."
            )

    # $Nodes
    nodes_section = extract_section(text, "Nodes")
    if nodes_section is None:
        raise GridImportError("Gmsh .msh file is missing the $Nodes section.")
    node_lines = nodes_section.splitlines()
    try:
        n_nodes = int(node_lines[0].strip())
    except (IndexError, ValueError) as exc:
        raise GridImportError(f"Cannot read node count from $Nodes section: {exc}") from exc

    # Gmsh node IDs are 1-based; we map them to 0-based indices.
    node_id_to_index: dict[int, int] = {}
    coords: list[tuple[float, float, float]] = []
    for line in node_lines[1 : n_nodes + 1]:
        parts = line.split()
        if len(parts) < 4:
            raise GridImportError(f"Malformed node line: {line!r}")
        try:
            node_id = int(parts[0])
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError as exc:
            raise GridImportError(f"Malformed node line: {line!r}") from exc
        node_id_to_index[node_id] = len(coords)
        coords.append((x, y, z))

    if not coords:
        raise GridImportError("No nodes found in the $Nodes section.")
    vertex_coordinates = np.array(coords, dtype=np.float64)

    # $Elements
    elements_section = extract_section(text, "Elements")
    if elements_section is None:
        raise GridImportError("Gmsh .msh file is missing the $Elements section.")

    elements_lines = elements_section.splitlines()
    try:
        n_elements = int(elements_lines[0].strip())
    except (IndexError, ValueError) as exc:
        raise GridImportError(f"Cannot read element count from $Elements section: {exc}") from exc

    # Accumulate cells grouped by type
    cell_blocks_by_type: dict[str, list[list[int]]] = {}
    for line in elements_lines[1 : n_elements + 1]:
        parts = line.split()
        if len(parts) < 3:
            continue

        try:
            element_type_id = int(parts[1])
            n_tags = int(parts[2])
        except ValueError as exc:
            raise GridImportError(f"Malformed element line: {line!r}") from exc
        if element_type_id in GMSH_ELEMENT_TYPES:
            n_vertices, type_name = GMSH_ELEMENT_TYPES[element_type_id]
            n_listed = n_vertices
        elif element_type_id in GMSH_HIGHER_ORDER_ELEMENT_TYPES:
            n_listed, n_vertices, type_name = GMSH_HIGHER_ORDER_ELEMENT_TYPES[element_type_id]
        else:
            continue  # points, lines and surface elements are not part of the grid

        node_start = 3 + n_tags
        try:
            node_ids = [int(p) for p in parts[node_start : node_start + n_listed]]
        except ValueError as exc:
            raise GridImportError(f"Malformed element line: {line!r}") from exc
        if len(node_ids) != n_listed:
            raise GridImportError(
                f"Element has {len(node_ids)} node IDs but expected {n_listed} "
                f"for type {type_name!r}."
            )
        node_ids = node_ids[:n_vertices]
        # Convert to 0-based indices
        try:
            vertex_indices = [node_id_to_index[node_id] for node_id in node_ids]
        except KeyError as exc:
            raise GridImportError(f"Element references unknown node ID {exc}.") from exc

        cell_blocks_by_type.setdefault(type_name, []).append(vertex_indices)

    if not cell_blocks_by_type:
        raise GridImportError(
            "No supported 3-D elements (tetra, hexahedron, wedge, pyramid) "
            "found in the $Elements section."
        )

    cell_blocks = [
        {"cell_type": ctype, "connectivity": np.array(cells, dtype=np.int32)}
        for ctype, cells in cell_blocks_by_type.items()
    ]
    meta = {"source_format": "gmsh_msh"}
    if metadata:
        meta.update(metadata)
    try:
        return make_polyhedral_grid(
            vertex_coordinates=vertex_coordinates,  # type: ignore[arg-type]
            cell_blocks=cell_blocks,
            metadata=meta,
            unit_system=unit_system if unit_system is not None else UnitSystem.FIELD,
        )
    except Exception as exc:
        raise GridImportError(f"Failed to construct grid from Gmsh elements: {exc}") from exc
