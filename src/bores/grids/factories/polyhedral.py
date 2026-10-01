import typing

import numpy as np

from bores.errors import InvalidPointArrayError, ValidationError
from bores.grids.base import Grid
from bores.grids.factories.base import (
    ELEMENTS_FACES,
    VTK_CELL_TYPE_NAMES,
    ElementFaces,
    FaceVertexIndices,
    VertexCoordinates,
    build_csr_face_arrays,
)
from bores.types import (
    IntArray,
    Integer,
    NumberArray,
    OneDimension,
    TwoDimensions,
    UnitSystem,
)

__all__ = ["make_polyhedral_grid"]


def make_polyhedral_grid(
    *,
    vertex_coordinates: VertexCoordinates,
    cell_blocks: typing.Sequence[dict[str, typing.Any]],
    custom_cell_faces: dict[str, ElementFaces] | None = None,
    unit_system: UnitSystem = UnitSystem.FIELD,
    metadata: typing.Mapping[str, typing.Any] | None = None,
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
    Factory for general mixed-element polyhedral meshes.

    This is the **most general** factory and is used internally by the IO
    layer to convert VTK, meshio, and other mesh formats into a
    `bores.grids.base.Grid`. It supports any combination of
    tetrahedra, hexahedra, wedges, pyramids, and custom polyhedral cells.

    For custom cell types (not in `ELEMENTS_FACES`), callers must
    supply the face definitions explicitly via `custom_cell_faces`.

    Example usage:

    ```python
    # From meshio-style cell blocks:
    grid = make_polyhedral_grid(
        vertex_coordinates=mesh.points,
        cell_blocks=[
            {"cell_type": "hexahedron", "connectivity": hex_cells},
            {"cell_type": "tetra", "connectivity": tet_cells},
        ],
    )

    # From VTK cell arrays:
    grid = make_polyhedral_grid(
        vertex_coordinates=points,
        cell_blocks=[
            {"vtk_type": 12, "connectivity": vtk_hex_cells},
            {"vtk_type": 10, "connectivity": vtk_tet_cells},
        ],
    )
    ```

    Builds a grid from a sequence of mixed-element cell blocks.

    :param vertex_coordinates: Shape `(n_vertices, 3)` float64 point array.
    :param cell_blocks: List of dictionaries, one per element block.
        Each dict must contain a `"connectivity"` key whose value is an
        array-like of shape `(n_cells_in_block, n_verts_per_cell)`, plus one of:

        - `"cell_type"`: string name matching a key in
            `ELEMENTS_FACES` or `custom_cell_faces` (e.g.
            `"hexahedron"`, `"tetra"`).
        - `"vtk_type"`: integer VTK cell type code (e.g. `12` for hexahedron (hex)).

    :param custom_cell_faces: Optional mapping from cell-type name to face
        table, extending or overriding `ELEMENTS_FACES`.
        Use this for non-standard polyhedral element types.
    :param metadata: Optional metadata dictionary.
    :returns: A fully initialised `bores.grids.base.Grid`.
    :raises ValidationError: If a cell block contains an unrecognised element type.
    :raises InvalidPointArrayError: If `vertex_coordinates` is not `(N, 3)`.
    """
    points = np.asarray(vertex_coordinates, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise InvalidPointArrayError(
            f"vertex_coordinates must be shape (n_vertices, 3); got {points.shape!r}."
        )

    elements_faces = dict(ELEMENTS_FACES)
    if custom_cell_faces:
        elements_faces.update(custom_cell_faces)

    all_per_cell_faces: list[list[FaceVertexIndices]] = []

    for block_idx, block in enumerate(cell_blocks):
        element_name = resolve_cell_type_name(block, elements_faces, block_idx)
        element_faces = elements_faces[element_name]
        connectivity = np.asarray(block["connectivity"], dtype=np.int32)

        if connectivity.ndim != 2:
            raise ValidationError(
                f"Block {block_idx}: connectivity must be 2-D; got ndim={connectivity.ndim}."
            )

        expected_n_verts = max(max(face) for face in element_faces) + 1
        if connectivity.shape[1] < expected_n_verts:
            raise ValidationError(
                f"Block {block_idx} (type '{element_name}'): "
                f"connectivity has {connectivity.shape[1]} vertices per cell but "
                f"face table requires at least {expected_n_verts}."
            )

        for global_vertex_indices in connectivity:
            cell_faces: list[FaceVertexIndices] = [
                [global_vertex_indices[local_vertex] for local_vertex in element_face]
                for element_face in element_faces
            ]
            all_per_cell_faces.append(cell_faces)

    if not all_per_cell_faces:
        raise ValidationError("No cells found across all provided cell blocks.")

    points, face_vertex_indices, face_vertex_offsets, face_cell_indices = build_csr_face_arrays(
        points,  # type: ignore[arg-type]
        all_per_cell_faces,
    )
    return Grid(
        vertex_coordinates=points,
        face_vertex_indices=face_vertex_indices,
        face_vertex_offsets=face_vertex_offsets,
        face_cell_indices=face_cell_indices,
        unit_system=unit_system,
        metadata=metadata,
        nnc_cell_indices=nnc_cell_indices,
        nnc_transmissibilities=nnc_transmissibilities,
        positive_x_transmissibility_multipliers=positive_x_transmissibility_multipliers,
        negative_x_transmissibility_multipliers=negative_x_transmissibility_multipliers,
        positive_y_transmissibility_multipliers=positive_y_transmissibility_multipliers,
        negative_y_transmissibility_multipliers=negative_y_transmissibility_multipliers,
        positive_z_transmissibility_multipliers=positive_z_transmissibility_multipliers,
        negative_z_transmissibility_multipliers=negative_z_transmissibility_multipliers,
    )


def resolve_cell_type_name(
    block: dict[str, typing.Any],
    elements_faces: dict[str, ElementFaces],
    block_idx: Integer,
) -> str:
    """
    Resolve a cell block's element type to a string name.

    Accepts either `"cell_type"` (string) or `"vtk_type"` (integer)
    keys in the block dictionary.

    :param block: Cell block dictionary from the caller.
    :param elements_faces: Merged face table (built-in + custom).
    :param block_idx: Index of this block (for error messages).
    :returns: Resolved element type name string.
    :raises ValidationError: If neither key is present or the type is
        unrecognised.
    """
    if "cell_type" in block:
        name = block["cell_type"]
        if name not in elements_faces:
            raise ValidationError(
                f"Block {block_idx}: unrecognised cell_type '{name}'. "
                f"Known types: {sorted(elements_faces.keys())}."
            )
        return name
    elif "vtk_type" in block:
        vtk_code = int(block["vtk_type"])
        if vtk_code not in VTK_CELL_TYPE_NAMES:
            raise ValidationError(
                f"Block {block_idx}: unrecognised vtk_type {vtk_code}. "
                f"Supported codes: {sorted(VTK_CELL_TYPE_NAMES.keys())}."
            )
        name = VTK_CELL_TYPE_NAMES[vtk_code]
        if name not in elements_faces:
            raise ValidationError(
                f"Block {block_idx}: VTK type {vtk_code} maps to '{name}' "
                f"but no face table is defined for it."
            )
        return name

    raise ValidationError(
        f"Block {block_idx} must contain either 'cell_type' (str) "
        f"or 'vtk_type' (int). Got keys: {list(block.keys())}."
    )
