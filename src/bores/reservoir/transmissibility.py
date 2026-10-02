"""Connection transmissibilities for unstructured polyhedral grids."""

import typing

import numba
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from bores.constants import build_unit_conversion_table, get_conversion_factors
from bores.grids.base import Grid
from bores.precision import get_dtype
from bores.reservoir.rock import Rock
from bores.types import (
    IntArray,
    Integer,
    Number,
    NumberArray,
    NumberOrArray,
    OneDimension,
    TwoDimensions,
    UnitConversionTable,
    UnitSystem,
)
from bores.utils import scale

__all__ = ["ConnectionTransmissibilities", "compute_connection_transmissibilities"]


class ConnectionTransmissibilities(typing.NamedTuple):
    """Precomputed transmissibilities for all connections in a reservoir."""

    interior: NumberArray[OneDimension]
    """
    Shape `(n_interior_faces,)` float64 - TPFA transmissibility for
    every interior face (mD·ft in FIELD units).
    """
    boundary: NumberArray[OneDimension]
    """
    Shape `(n_boundary_faces,)` float64 - half-transmissibility for
    every boundary face.
    """
    nnc: NumberArray[OneDimension] | None
    """
    Shape `(n_nnc,)` float64 or `None` - flow transmissibility of each non-neighbour connection,
    exactly as supplied on the grid. Unlike `interior` and `boundary`, which are geometric
    (permeability x area / length), it already includes the Darcy unit conversion constant, so
    it multiplies mobility and pressure difference directly. Same order as
    `Grid.nnc_cell_indices`; `None` when the grid has no NNCs.
    """
    unit_system: UnitSystem

    def convert(
        self,
        target: UnitSystem,
        /,
        *,
        table: UnitConversionTable | None = None,
    ) -> Self:
        """
        Return a new `ConnectionTransmissibilities` with transmissibilities
        rescaled to *target*.

        :param target: Target `UnitSystem`.
        :returns: New `ConnectionTransmissibilities` in *target* units.
        """
        if target == self.unit_system:
            return self

        factors = get_conversion_factors(self.unit_system, target, table=table)
        # Transmissibility is K * A/d
        transmissibility_factor = factors["permeability"] * factors["area"] / factors["length"]
        return self._replace(
            interior=scale(self.interior, transmissibility_factor),
            boundary=scale(self.boundary, transmissibility_factor),
            nnc=(
                None
                if self.nnc is None
                else scale(
                    self.nnc,
                    factors["reservoir_rate"] * factors["viscosity"] / factors["pressure"],
                )
            ),
            unit_system=target,
        )


def get_face_transmissibility_map(
    grid: Grid, transmissibilities: ConnectionTransmissibilities
) -> dict[int, Number]:
    """
    Build a {global_face_index: transmissibility} dict for single-face lookups.

    Interior faces map to their full harmonic-mean T.
    Boundary faces map to their owner half-T.

    :param grid: The grid whose face indices define the mapping.
    :param transmissibilities: Precomputed transmissibilities for that grid.
    :returns: Dict mapping global face index to transmissibility value.
    """
    result: dict[int, Number] = {}
    for position, global_face_idx in enumerate(grid.interior_face_indices):
        result[int(global_face_idx)] = transmissibilities.interior[position]
    for position, global_face_idx in enumerate(grid.boundary_face_indices):
        result[int(global_face_idx)] = transmissibilities.boundary[position]
    return result


def compute_connection_transmissibilities(
    grid: Grid,
    rock: Rock,
    *,
    net_to_gross: NumberOrArray[OneDimension] | None = None,
    unit_system: UnitSystem | None = None,
    dtype: npt.DTypeLike = None,
) -> ConnectionTransmissibilities:
    """
    Compute TPFA transmissibilities for all connections in an unstructured grid.

    Permeability is projected onto each face normal:

    ```text
    K_proj = |nx|·Kx + |ny|·Ky + |nz|·Kz
    ```

    **Interior / boundary faces** use the standard harmonic-mean / half-T formulas.

    **NNC transmissibilities** are returned as given on the grid. They are flow
    transmissibilities (the Darcy constant is already included) and, since an NNC has no shared
    surface, nothing is computed from rock properties.

    **Multiplier application**:

    - Directional MULT arrays (MULTX, MULTX-, MULTY, MULTY-, MULTZ, MULTZ-) are
      applied only to regular face-based connections (interior, boundary, and fault).
      NNCs are not directional and are not affected.
    - `MULTFLT` is applied to face-based fault connections only.

    Note: On construction the transmissibilities are normalised to the
        declared `unit_system` (defaults to the grid's own unit system).
        However, it is advised that both grid and rock are in the same unit
        system.

    :param grid: Fully constructed `bores.grids.base.Grid`.
    :param rock: `Rock` with `absolute_permeability` and `net_to_gross`.
    :param net_to_gross: Optional override for the NTG array.
    :param dtype: NumPy floating dtype for output arrays. Defaults to `bores.get_dtype()`.
    :returns: `ConnectionTransmissibilities` named tuple.
    :raises ValueError: If permeability or NTG array lengths do not match `grid.n_cells`.
    """
    dtype = np.dtype(dtype) if dtype is not None else get_dtype()
    target_unit_system = unit_system if unit_system is not None else grid.unit_system
    unit_conversion_table: UnitConversionTable | None = None
    if target_unit_system != grid.unit_system:
        unit_conversion_table = build_unit_conversion_table()
        # Normalise grid to the target unit system.
        grid = grid.convert(target_unit_system, table=unit_conversion_table)

    # Normalise rock to the target unit system (if needed).
    rock = rock.convert(target_unit_system, table=unit_conversion_table)

    kx = rock.absolute_permeability.x.astype(dtype, copy=False)
    ky = rock.absolute_permeability.y.astype(dtype, copy=False)
    kz = rock.absolute_permeability.z.astype(dtype, copy=False)
    ntg = np.asarray(rock.net_to_gross if net_to_gross is None else net_to_gross, dtype=dtype)

    n_cells = grid.n_cells
    for name, array in (("Kx", kx), ("Ky", ky), ("Kz", kz), ("NTG", ntg)):
        if array.shape != (n_cells,):
            raise ValueError(f"{name} array has shape {array.shape}; expected ({n_cells},).")

    effective_kx = kx * ntg
    effective_ky = ky * ntg
    effective_kz = kz * ntg

    interior_face_indices = grid.interior_face_indices
    boundary_face_indices = grid.boundary_face_indices

    interior_transmissibilities = compute_interior_tpfa_transmissibilities(
        interior_face_indices=interior_face_indices,
        face_cell_indices=grid.face_cell_indices,
        face_centroids=grid.face_centroids,
        face_areas=grid.face_areas,
        face_unit_normals=grid.face_unit_normals,
        cell_centroids=grid.cell_centroids,  # type: ignore[arg-type]
        effective_kx=effective_kx,  # type: ignore[arg-type]
        effective_ky=effective_ky,  # type: ignore[arg-type]
        effective_kz=effective_kz,  # type: ignore[arg-type]
        dtype=dtype,
    )

    boundary_transmissibilities = compute_boundary_half_transmissibilities(
        boundary_face_indices=boundary_face_indices,
        face_cell_indices=grid.face_cell_indices,
        face_centroids=grid.face_centroids,
        face_areas=grid.face_areas,
        face_unit_normals=grid.face_unit_normals,
        cell_centroids=grid.cell_centroids,  # type: ignore[arg-type]
        effective_kx=effective_kx,  # type: ignore[arg-type]
        effective_ky=effective_ky,  # type: ignore[arg-type]
        effective_kz=effective_kz,  # type: ignore[arg-type]
        dtype=dtype,
    )

    if grid.has_transmissibility_multipliers:
        interior_transmissibilities, boundary_transmissibilities = apply_directional_multipliers(
            interior_transmissibilities=interior_transmissibilities,
            boundary_transmissibilities=boundary_transmissibilities,
            interior_face_indices=interior_face_indices,
            boundary_face_indices=boundary_face_indices,
            face_cell_indices=grid.face_cell_indices,
            face_unit_normals=grid.face_unit_normals,
            positive_x_multipliers=grid.positive_x_transmissibility_multipliers,
            negative_x_multipliers=grid.negative_x_transmissibility_multipliers,
            positive_y_multipliers=grid.positive_y_transmissibility_multipliers,
            negative_y_multipliers=grid.negative_y_transmissibility_multipliers,
            positive_z_multipliers=grid.positive_z_transmissibility_multipliers,
            negative_z_multipliers=grid.negative_z_transmissibility_multipliers,
        )

    nnc_transmissibilities: NumberArray[OneDimension] | None = None
    if grid.n_nnc > 0 and grid.nnc_transmissibilities is not None:
        nnc_transmissibilities = typing.cast(
            NumberArray[OneDimension], grid.nnc_transmissibilities.astype(dtype, copy=True)
        )

    # Apply `MULTFLT` to face-based connections
    if grid.fault_face_indices is not None and grid.fault_transmissibility_multipliers is not None:
        interior_transmissibilities, boundary_transmissibilities = apply_fault_face_multipliers(
            interior_transmissibilities=interior_transmissibilities,
            boundary_transmissibilities=boundary_transmissibilities,
            interior_face_indices=interior_face_indices,
            boundary_face_indices=boundary_face_indices,
            fault_face_indices=grid.fault_face_indices,
            fault_transmissibility_multipliers=grid.fault_transmissibility_multipliers,
        )

    return ConnectionTransmissibilities(
        interior=interior_transmissibilities.astype(dtype, copy=False),
        boundary=boundary_transmissibilities.astype(dtype, copy=False),
        nnc=nnc_transmissibilities,
        unit_system=target_unit_system,
    )


@numba.njit(parallel=True, cache=True)
def compute_interior_tpfa_transmissibilities(
    interior_face_indices: IntArray[OneDimension],
    face_cell_indices: IntArray[TwoDimensions],
    face_centroids: NumberArray[TwoDimensions],
    face_areas: NumberArray[OneDimension],
    face_unit_normals: NumberArray[TwoDimensions],
    cell_centroids: NumberArray[TwoDimensions],
    effective_kx: NumberArray[OneDimension],
    effective_ky: NumberArray[OneDimension],
    effective_kz: NumberArray[OneDimension],
    dtype: npt.DTypeLike,
) -> NumberArray[OneDimension]:
    """
    Harmonic-mean TPFA transmissibilities for all interior faces.

    For each interior face the half-T of cell *c* is:

    ```text
    K_c = |nx|·Kx_c + |ny|·Ky_c + |nz|·Kz_c
    T_c = K_c · area / d_c
    ```

    Full harmonic T: `T_A · T_B / (T_A + T_B)`.

    :param interior_face_indices: Shape `(n_interior,)`.
    :param face_cell_indices: Shape `(n_faces, 2)`.
    :param face_centroids: Shape `(n_faces, 3)`.
    :param face_areas: Shape `(n_faces,)`.
    :param face_unit_normals: Shape `(n_faces, 3)`.
    :param cell_centroids: Shape `(n_cells, 3)`.
    :param effective_kx: Shape `(n_cells,)`.
    :param effective_ky: Shape `(n_cells,)`.
    :param effective_kz: Shape `(n_cells,)`.
    :param dtype: Output dtype.
    :returns: Shape `(n_interior,)` transmissibility array.
    """
    n_interior = interior_face_indices.shape[0]
    transmissibilities = np.zeros(n_interior, dtype=dtype)

    for idx in numba.prange(n_interior):  # type: ignore
        face_idx = interior_face_indices[idx]
        owner = face_cell_indices[face_idx, 0]
        neighbour = face_cell_indices[face_idx, 1]

        nx = abs(face_unit_normals[face_idx, 0])
        ny = abs(face_unit_normals[face_idx, 1])
        nz = abs(face_unit_normals[face_idx, 2])
        area = face_areas[face_idx]

        fx = face_centroids[face_idx, 0]
        fy = face_centroids[face_idx, 1]
        fz = face_centroids[face_idx, 2]

        dx_a = fx - cell_centroids[owner, 0]
        dy_a = fy - cell_centroids[owner, 1]
        dz_a = fz - cell_centroids[owner, 2]
        d_a = (dx_a * dx_a + dy_a * dy_a + dz_a * dz_a) ** 0.5
        k_a = nx * effective_kx[owner] + ny * effective_ky[owner] + nz * effective_kz[owner]
        T_a = k_a * area / d_a if d_a > 0.0 else 0.0

        dx_b = fx - cell_centroids[neighbour, 0]
        dy_b = fy - cell_centroids[neighbour, 1]
        dz_b = fz - cell_centroids[neighbour, 2]
        d_b = (dx_b * dx_b + dy_b * dy_b + dz_b * dz_b) ** 0.5
        k_b = (
            nx * effective_kx[neighbour]
            + ny * effective_ky[neighbour]
            + nz * effective_kz[neighbour]
        )
        T_b = k_b * area / d_b if d_b > 0.0 else 0.0

        if T_a + T_b > 0.0:
            transmissibilities[idx] = (T_a * T_b) / (T_a + T_b)

    return transmissibilities


@numba.njit(parallel=True, cache=True)
def compute_boundary_half_transmissibilities(
    boundary_face_indices: IntArray[OneDimension],
    face_cell_indices: IntArray[TwoDimensions],
    face_centroids: NumberArray[TwoDimensions],
    face_areas: NumberArray[OneDimension],
    face_unit_normals: NumberArray[TwoDimensions],
    cell_centroids: NumberArray[TwoDimensions],
    effective_kx: NumberArray[OneDimension],
    effective_ky: NumberArray[OneDimension],
    effective_kz: NumberArray[OneDimension],
    dtype: npt.DTypeLike,
) -> NumberArray[OneDimension]:
    """
    Owner half-transmissibilities for all boundary faces.

    Formula: `T_half = K_owner · area / d_owner`.

    :param boundary_face_indices: Shape `(n_boundary,)`.
    :param face_cell_indices: Shape `(n_faces, 2)`.
    :param face_centroids: Shape `(n_faces, 3)`.
    :param face_areas: Shape `(n_faces,)`.
    :param face_unit_normals: Shape `(n_faces, 3)`.
    :param cell_centroids: Shape `(n_cells, 3)`.
    :param effective_kx: Shape `(n_cells,)`.
    :param effective_ky: Shape `(n_cells,)`.
    :param effective_kz: Shape `(n_cells,)`.
    :param dtype: Output dtype.
    :returns: Shape `(n_boundary,)` half-transmissibility array.
    """
    n_boundary = boundary_face_indices.shape[0]
    transmissibilities = np.zeros(n_boundary, dtype=dtype)

    for idx in numba.prange(n_boundary):  # type: ignore
        face_idx = boundary_face_indices[idx]
        owner = face_cell_indices[face_idx, 0]

        nx = abs(face_unit_normals[face_idx, 0])
        ny = abs(face_unit_normals[face_idx, 1])
        nz = abs(face_unit_normals[face_idx, 2])
        area = face_areas[face_idx]

        fx = face_centroids[face_idx, 0]
        fy = face_centroids[face_idx, 1]
        fz = face_centroids[face_idx, 2]

        dx = fx - cell_centroids[owner, 0]
        dy = fy - cell_centroids[owner, 1]
        dz = fz - cell_centroids[owner, 2]
        d = (dx * dx + dy * dy + dz * dz) ** 0.5

        k = nx * effective_kx[owner] + ny * effective_ky[owner] + nz * effective_kz[owner]
        if d > 0.0:
            transmissibilities[idx] = k * area / d

    return transmissibilities


@numba.njit(cache=True)
def apply_directional_multipliers(
    interior_transmissibilities: NumberArray[OneDimension],
    boundary_transmissibilities: NumberArray[OneDimension],
    interior_face_indices: IntArray[OneDimension],
    boundary_face_indices: IntArray[OneDimension],
    face_cell_indices: IntArray[TwoDimensions],
    face_unit_normals: NumberArray,
    positive_x_multipliers: NumberArray[OneDimension] | None,
    negative_x_multipliers: NumberArray[OneDimension] | None,
    positive_y_multipliers: NumberArray[OneDimension] | None,
    negative_y_multipliers: NumberArray[OneDimension] | None,
    positive_z_multipliers: NumberArray[OneDimension] | None,
    negative_z_multipliers: NumberArray[OneDimension] | None,
) -> tuple[NumberArray[OneDimension], NumberArray[OneDimension]]:
    """
    Scale face transmissibilities by per-cell directional MULT arrays (in-place).

    For interior faces: `multiplier = MULT_forward(owner) x MULT_backward(neighbour)`.
    For boundary faces: `multiplier = MULT_forward(owner)` (no neighbour).
    Direction is the dominant component of the face unit normal.

    NNCs are not affected.

    :param interior_transmissibilities: Shape `(n_interior,)`.
    :param boundary_transmissibilities: Shape `(n_boundary,)`.
    :param interior_face_indices: Global face indices for interior faces.
    :param boundary_face_indices: Global face indices for boundary faces.
    :param face_cell_indices: Shape `(n_faces, 2)`.
    :param face_unit_normals: Shape `(n_faces, 3)`.
    :param positive_x_multipliers: MULTX or `None`.
    :param negative_x_multipliers: MULTX- or `None`.
    :param positive_y_multipliers: MULTY or `None`.
    :param negative_y_multipliers: MULTY- or `None`.
    :param positive_z_multipliers: MULTZ or `None`.
    :param negative_z_multipliers: MULTZ- or `None`.
    :returns: Updated `(interior_transmissibilities, boundary_transmissibilities)`.
    """
    n_interior = len(interior_face_indices)
    n_boundary = len(boundary_face_indices)

    for idx in range(n_interior):
        face_idx = interior_face_indices[idx]
        owner = face_cell_indices[face_idx, 0]
        neighbour = face_cell_indices[face_idx, 1]

        nx = abs(face_unit_normals[face_idx, 0])
        ny = abs(face_unit_normals[face_idx, 1])
        nz = abs(face_unit_normals[face_idx, 2])

        multiplier = 1.0
        if nx >= ny and nx >= nz:
            if positive_x_multipliers is not None:
                multiplier *= positive_x_multipliers[owner]
            if negative_x_multipliers is not None:
                multiplier *= negative_x_multipliers[neighbour]
        elif ny >= nx and ny >= nz:
            if positive_y_multipliers is not None:
                multiplier *= positive_y_multipliers[owner]
            if negative_y_multipliers is not None:
                multiplier *= negative_y_multipliers[neighbour]
        else:
            if positive_z_multipliers is not None:
                multiplier *= positive_z_multipliers[owner]
            if negative_z_multipliers is not None:
                multiplier *= negative_z_multipliers[neighbour]

        interior_transmissibilities[idx] *= multiplier

    for idx in range(n_boundary):
        face_idx = boundary_face_indices[idx]
        owner = face_cell_indices[face_idx, 0]

        nx = abs(face_unit_normals[face_idx, 0])
        ny = abs(face_unit_normals[face_idx, 1])
        nz = abs(face_unit_normals[face_idx, 2])

        multiplier = 1.0
        if nx >= ny and nx >= nz:
            if positive_x_multipliers is not None:
                multiplier *= positive_x_multipliers[owner]
        elif ny >= nx and ny >= nz:
            if positive_y_multipliers is not None:
                multiplier *= positive_y_multipliers[owner]
        else:
            if positive_z_multipliers is not None:
                multiplier *= positive_z_multipliers[owner]

        boundary_transmissibilities[idx] *= multiplier

    return interior_transmissibilities, boundary_transmissibilities


@numba.njit(cache=True)
def apply_fault_face_multipliers(
    interior_transmissibilities: NumberArray[OneDimension],
    boundary_transmissibilities: NumberArray[OneDimension],
    interior_face_indices: IntArray[OneDimension],
    boundary_face_indices: IntArray[OneDimension],
    fault_face_indices: typing.Mapping[str, IntArray[OneDimension]],
    fault_transmissibility_multipliers: typing.Mapping[str, Number],
) -> tuple[NumberArray[OneDimension], NumberArray[OneDimension]]:
    """
    Apply `MULTFLT` multipliers to face-based fault connections (in-place).

    Only faces tagged with `ConnectionType.*_FAULT_FACE` (i.e. present in
    `fault_face_indices`) are affected. NNCs are handled separately.

    :param interior_transmissibilities: Shape `(n_interior,)`.
    :param boundary_transmissibilities: Shape `(n_boundary,)`.
    :param interior_face_indices: Global face indices for interior faces.
    :param boundary_face_indices: Global face indices for boundary faces.
    :param fault_face_indices: `{name: face_indices}`.
    :param fault_transmissibility_multipliers: `{name: multiplier}`.
    :returns: Updated transmissibility arrays.
    """
    global_to_interior: dict[Integer, int] = {
        global_idx: position for position, global_idx in enumerate(interior_face_indices)
    }
    global_to_boundary: dict[Integer, int] = {
        global_idx: position for position, global_idx in enumerate(boundary_face_indices)
    }

    for fault_name, face_indices in fault_face_indices.items():
        multiplier = fault_transmissibility_multipliers.get(fault_name, 1.0)
        if multiplier == 1:
            continue
        for global_idx in face_indices:
            interior_position = global_to_interior.get(global_idx)
            if interior_position is not None:
                interior_transmissibilities[interior_position] *= multiplier
                continue
            boundary_position = global_to_boundary.get(global_idx)
            if boundary_position is not None:
                boundary_transmissibilities[boundary_position] *= multiplier

    return interior_transmissibilities, boundary_transmissibilities
