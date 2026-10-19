"""Connection transmissibilities for unstructured polyhedral grids."""

import typing

import numba
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from bores.grids.base import Grid
from bores.precision import get_dtype
from bores.reservoir.rock import Rock
from bores.types import (
    IntArray,
    Number,
    NumberArray,
    NumberOrArray,
    OneDimension,
    TwoDimensions,
    UnitConversionTable,
    UnitSystem,
)
from bores.units import build_unit_conversion_table, get_conversion_factors
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
    it multiplies mobility and pressure difference directly. 
    
    Same order as `Grid.nnc_cell_indices`; `None` when the grid has no NNCs.
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

    Each cell's half-transmissibility toward a face is
    `A · |(K n) · c| / |c|²`, with `c` the vector from the cell centroid to the face centroid
    and `K = diag(Kx, Ky, Kz)`. This is exact for orthogonal grids and stays consistent for
    tilted and skewed faces.

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
    :param rock: `Rock` with `absolute_permeability` and `net_to_gross`. Net-to-gross scales
        Kx and Ky only, not Kz.
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

    # Net-to-gross reduces horizontal flow only. Vertical flow crosses the whole gross
    # thickness of the cell, so Kz is not scaled (Eclipse convention).
    effective_kx = kx * ntg
    effective_ky = ky * ntg
    effective_kz = kz

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


@numba.njit(cache=True, inline="always")
def compute_half_transmissibility(
    kx: Number,
    ky: Number,
    kz: Number,
    normal_x: Number,
    normal_y: Number,
    normal_z: Number,
    area: Number,
    offset_x: Number,
    offset_y: Number,
    offset_z: Number,
) -> Number:
    """
    One cell's half-transmissibility toward a face.

    `T = A · |(K n) · c| / |c|²`, with `c` the vector from the cell centroid to the face
    centroid and `n` the face normal. For a diagonal permeability tensor `K`. For a face
    perpendicular to an axis through the cell centroid this reduces to `K A / d`; on a skewed
    or tilted face it keeps the flow driven by the permeability along `c` and the distance
    along `c`, rather than treating the normal as the flow direction.

    :param kx: Effective permeability along x.
    :param ky: Effective permeability along y.
    :param kz: Effective permeability along z.
    :param normal_x: x component of the face unit normal.
    :param normal_y: y component of the face unit normal.
    :param normal_z: z component of the face unit normal.
    :param area: Face area.
    :param offset_x: x of the vector from the cell centroid to the face centroid.
    :param offset_y: y of that vector.
    :param offset_z: z of that vector.
    :returns: The half-transmissibility, zero for a cell centroid on its own face.
    """
    squared_distance = offset_x * offset_x + offset_y * offset_y + offset_z * offset_z
    if squared_distance <= 0.0:
        return 0.0
    projected = kx * normal_x * offset_x + ky * normal_y * offset_y + kz * normal_z * offset_z
    return area * abs(projected) / squared_distance


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

    The half-T of each cell comes from `compute_half_transmissibility`. Full harmonic T: `T_A · T_B / (T_A + T_B)`.

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

        nx = face_unit_normals[face_idx, 0]
        ny = face_unit_normals[face_idx, 1]
        nz = face_unit_normals[face_idx, 2]
        area = face_areas[face_idx]

        fx = face_centroids[face_idx, 0]
        fy = face_centroids[face_idx, 1]
        fz = face_centroids[face_idx, 2]

        T_a = compute_half_transmissibility(
            kx=effective_kx[owner],  # type: ignore[arg-type]
            ky=effective_ky[owner],  # type: ignore[arg-type]
            kz=effective_kz[owner],  # type: ignore[arg-type]
            normal_x=nx,
            normal_y=ny,
            normal_z=nz,
            area=area,  # type: ignore[arg-type]
            offset_x=fx - cell_centroids[owner, 0],
            offset_y=fy - cell_centroids[owner, 1],
            offset_z=fz - cell_centroids[owner, 2],
        )
        T_b = compute_half_transmissibility(
            kx=effective_kx[neighbour],  # type: ignore[arg-type]
            ky=effective_ky[neighbour],  # type: ignore[arg-type]
            kz=effective_kz[neighbour],  # type: ignore[arg-type]
            normal_x=nx,
            normal_y=ny,
            normal_z=nz,
            area=area,  # type: ignore[arg-type]
            offset_x=fx - cell_centroids[neighbour, 0],
            offset_y=fy - cell_centroids[neighbour, 1],
            offset_z=fz - cell_centroids[neighbour, 2],
        )

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

    The half-T comes from `compute_half_transmissibility`.

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

        transmissibilities[idx] = compute_half_transmissibility(
            kx=effective_kx[owner],  # type: ignore[arg-type]
            ky=effective_ky[owner],  # type: ignore[arg-type]
            kz=effective_kz[owner],  # type: ignore[arg-type]
            normal_x=face_unit_normals[face_idx, 0],
            normal_y=face_unit_normals[face_idx, 1],
            normal_z=face_unit_normals[face_idx, 2],
            area=face_areas[face_idx],  # type: ignore[arg-type]
            offset_x=face_centroids[face_idx, 0] - cell_centroids[owner, 0],
            offset_y=face_centroids[face_idx, 1] - cell_centroids[owner, 1],
            offset_z=face_centroids[face_idx, 2] - cell_centroids[owner, 2],
        )

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

    The direction of a face is the dominant component of its unit normal, and the
    normal points from the owner cell to the neighbour cell (outward from the owner for a
    boundary face). When that component is positive, the owner sits on the negative side of
    the face, so the owner's positive multiplier (`MULTX`) and the neighbour's negative
    multiplier (`MULTX-`) apply. When it is negative the roles are swapped. A boundary
    face uses the owner's positive multiplier when its normal points in the positive
    direction and the owner's negative multiplier otherwise.

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

        nx = face_unit_normals[face_idx, 0]
        ny = face_unit_normals[face_idx, 1]
        nz = face_unit_normals[face_idx, 2]
        ax = abs(nx)
        ay = abs(ny)
        az = abs(nz)

        multiplier = 1.0
        if ax >= ay and ax >= az:
            owner_side = positive_x_multipliers if nx >= 0.0 else negative_x_multipliers
            neighbour_side = negative_x_multipliers if nx >= 0.0 else positive_x_multipliers
        elif ay >= ax and ay >= az:
            owner_side = positive_y_multipliers if ny >= 0.0 else negative_y_multipliers
            neighbour_side = negative_y_multipliers if ny >= 0.0 else positive_y_multipliers
        else:
            owner_side = positive_z_multipliers if nz >= 0.0 else negative_z_multipliers
            neighbour_side = negative_z_multipliers if nz >= 0.0 else positive_z_multipliers

        if owner_side is not None:
            multiplier *= owner_side[owner]
        if neighbour_side is not None:
            multiplier *= neighbour_side[neighbour]
        interior_transmissibilities[idx] *= multiplier

    for idx in range(n_boundary):
        face_idx = boundary_face_indices[idx]
        owner = face_cell_indices[face_idx, 0]

        nx = face_unit_normals[face_idx, 0]
        ny = face_unit_normals[face_idx, 1]
        nz = face_unit_normals[face_idx, 2]
        ax = abs(nx)
        ay = abs(ny)
        az = abs(nz)

        if ax >= ay and ax >= az:
            owner_side = positive_x_multipliers if nx >= 0.0 else negative_x_multipliers
        elif ay >= ax and ay >= az:
            owner_side = positive_y_multipliers if ny >= 0.0 else negative_y_multipliers
        else:
            owner_side = positive_z_multipliers if nz >= 0.0 else negative_z_multipliers

        if owner_side is not None:
            boundary_transmissibilities[idx] *= owner_side[owner]

    return interior_transmissibilities, boundary_transmissibilities


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
    `fault_face_indices`) are affected. NNCs are handled separately. A face that belongs
    to more than one fault is scaled by each fault's multiplier.

    :param interior_transmissibilities: Shape `(n_interior,)`.
    :param boundary_transmissibilities: Shape `(n_boundary,)`.
    :param interior_face_indices: Global face indices for interior faces.
    :param boundary_face_indices: Global face indices for boundary faces.
    :param fault_face_indices: `{name: face_indices}`.
    :param fault_transmissibility_multipliers: `{name: multiplier}`.
    :returns: Updated transmissibility arrays.
    """
    # interior_face_indices = np.asarray(interior_face_indices)
    # boundary_face_indices = np.asarray(boundary_face_indices)
    interior_order = np.argsort(interior_face_indices, kind="stable")
    boundary_order = np.argsort(boundary_face_indices, kind="stable")
    sorted_interior = interior_face_indices[interior_order]
    sorted_boundary = boundary_face_indices[boundary_order]

    for fault_name, face_indices in fault_face_indices.items():
        multiplier = fault_transmissibility_multipliers.get(fault_name, 1.0)
        if multiplier == 1:
            continue
        faces = np.asarray(face_indices)
        if faces.size == 0:
            continue

        if sorted_interior.size > 0:
            found = np.searchsorted(sorted_interior, faces)
            found = np.minimum(found, sorted_interior.size - 1)
            is_interior = sorted_interior[found] == faces
            np.multiply.at(
                interior_transmissibilities, interior_order[found[is_interior]], multiplier
            )
        else:
            is_interior = np.zeros(faces.shape, dtype=np.bool_)

        remaining = faces[~is_interior]
        if remaining.size > 0 and sorted_boundary.size > 0:
            found = np.searchsorted(sorted_boundary, remaining)
            found = np.minimum(found, sorted_boundary.size - 1)
            is_boundary = sorted_boundary[found] == remaining
            np.multiply.at(
                boundary_transmissibilities, boundary_order[found[is_boundary]], multiplier
            )

    return interior_transmissibilities, boundary_transmissibilities
