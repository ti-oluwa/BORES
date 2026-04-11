import logging
import typing

import attrs
import numba
import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_matrix, csr_matrix

from bores.config import Config
from bores.constants import c
from bores.datastructures import PhaseTensorsProxy
from bores.grids.base import CapillaryPressureGrids, RelativeMobilityGrids
from bores.grids.rock_fluid import build_rock_fluid_properties_grids
from bores.models import FluidProperties, RockProperties
from bores.solvers.base import (
    EvolutionResult,
    compute_mobility_grids,
    from_1D_index,
    solve_linear_system,
    to_1D_index,
)
from bores.solvers.explicit.saturation.immiscible import (
    compute_fluxes_from_neighbour,
    compute_well_rate_grids,
)
from bores.tables.rock_fluid import RockFluidTables
from bores.transmissibility import FaceTransmissibilities
from bores.types import ThreeDimensionalGrid, ThreeDimensions
from bores.wells.indices import WellIndicesCache

logger = logging.getLogger(__name__)

__all__ = ["evolve_saturation"]


@attrs.frozen
class ImplicitSaturationSolution:
    """Result of an implicit saturation solve."""

    water_saturation_grid: ThreeDimensionalGrid
    oil_saturation_grid: ThreeDimensionalGrid
    gas_saturation_grid: ThreeDimensionalGrid
    newton_iterations: int
    final_residual_norm: float
    maximum_water_saturation_change: float
    maximum_oil_saturation_change: float
    maximum_gas_saturation_change: float


@attrs.frozen
class NewtonConvergenceInfo:
    """Per-iteration convergence record."""

    iteration: int
    residual_norm: float
    relative_residual_norm: float
    max_saturation_update: float
    line_search_factor: float


@numba.njit(cache=True, parallel=True)
def saturation_grids_to_vector(
    water_saturation_grid: ThreeDimensionalGrid,
    gas_saturation_grid: ThreeDimensionalGrid,
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
) -> npt.NDArray:
    """
    Pack Sw and Sg from 3-D grids into a 1-D vector.

    Layout: ``[Sw_0, Sg_0, Sw_1, Sg_1, ..., Sw_{N-1}, Sg_{N-1}]``
    where ``N = cell_count_x * cell_count_y * cell_count_z``.

    :param water_saturation_grid: Water saturation grid, shape ``(nx, ny, nz)``.
    :param gas_saturation_grid: Gas saturation grid, shape ``(nx, ny, nz)``.
    :param cell_count_x: Number of cells in x-direction.
    :param cell_count_y: Number of cells in y-direction.
    :param cell_count_z: Number of cells in z-direction.
    :return: 1-D array of length ``2 * nx * ny * nz``.
    """
    total_cell_count = cell_count_x * cell_count_y * cell_count_z
    saturation_vector = np.empty(2 * total_cell_count)
    for i in numba.prange(cell_count_x):  # type: ignore[attr-defined]
        for j in range(cell_count_y):
            for k in range(cell_count_z):
                cell_1d_index = to_1D_index(
                    i=i,
                    j=j,
                    k=k,
                    cell_count_x=cell_count_x,
                    cell_count_y=cell_count_y,
                    cell_count_z=cell_count_z,
                )
                saturation_vector[2 * cell_1d_index] = water_saturation_grid[i, j, k]
                saturation_vector[2 * cell_1d_index + 1] = gas_saturation_grid[i, j, k]
    return saturation_vector


@numba.njit(cache=True, parallel=True)
def vector_to_saturation_grids(
    saturation_vector: npt.NDArray,
    water_saturation_grid: ThreeDimensionalGrid,
    oil_saturation_grid: ThreeDimensionalGrid,
    gas_saturation_grid: ThreeDimensionalGrid,
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
) -> None:
    """
    Unpack a 1-D saturation vector back into 3-D grids in-place.

    Computes oil saturation as ``So = 1 - Sw - Sg``.

    :param saturation_vector: 1-D array of length ``2 * nx * ny * nz``.
    :param water_saturation_grid: Output water saturation grid (modified in-place).
    :param oil_saturation_grid: Output oil saturation grid (modified in-place).
    :param gas_saturation_grid: Output gas saturation grid (modified in-place).
    :param cell_count_x: Number of cells in x-direction.
    :param cell_count_y: Number of cells in y-direction.
    :param cell_count_z: Number of cells in z-direction.
    """
    for i in numba.prange(cell_count_x):  # type: ignore[attr-defined]
        for j in range(cell_count_y):
            for k in range(cell_count_z):
                cell_1d_index = to_1D_index(
                    i=i,
                    j=j,
                    k=k,
                    cell_count_x=cell_count_x,
                    cell_count_y=cell_count_y,
                    cell_count_z=cell_count_z,
                )
                water_saturation = saturation_vector[2 * cell_1d_index]
                gas_saturation = saturation_vector[2 * cell_1d_index + 1]
                water_saturation_grid[i, j, k] = water_saturation
                gas_saturation_grid[i, j, k] = gas_saturation
                oil_saturation_grid[i, j, k] = max(
                    0.0, 1.0 - water_saturation - gas_saturation
                )


@numba.njit(cache=True, parallel=True)
def project_to_feasible(saturation_vector: npt.NDArray) -> npt.NDArray:
    """
    Project a saturation vector onto the feasible set.

    Enforces ``Sw >= 0``, ``Sg >= 0``, ``Sw + Sg <= 1`` by proportional
    scaling when the sum exceeds unity.

    :param saturation_vector: 1-D vector ``[Sw_0, Sg_0, ...]`` modified in-place.
    :return: The same vector after projection.
    """
    total_cell_count = len(saturation_vector) // 2
    for cell_1d_index in numba.prange(total_cell_count):  # type: ignore[attr-defined]
        water_saturation = max(0.0, saturation_vector[2 * cell_1d_index])
        gas_saturation = max(0.0, saturation_vector[2 * cell_1d_index + 1])
        total_saturation = water_saturation + gas_saturation
        if total_saturation > 1.0:
            water_saturation = water_saturation / total_saturation
            gas_saturation = gas_saturation / total_saturation
        saturation_vector[2 * cell_1d_index] = water_saturation
        saturation_vector[2 * cell_1d_index + 1] = gas_saturation
    return saturation_vector


@numba.njit(cache=True, parallel=True)
def interleave_residuals(
    water_residual: npt.NDArray,
    gas_residual: npt.NDArray,
) -> npt.NDArray:
    """
    Interleave water and gas residual arrays into a single vector.

    Layout: ``[R_w_0, R_g_0, R_w_1, R_g_1, ..., R_w_{N-1}, R_g_{N-1}]``.

    :param water_residual: 1-D water residual array of length N.
    :param gas_residual: 1-D gas residual array of length N.
    :return: Interleaved 1-D array of length 2 * N.
    """
    total_cell_count = len(water_residual)
    result = np.empty(2 * total_cell_count)
    for cell_1d_index in numba.prange(total_cell_count):  # type: ignore[attr-defined]
        result[2 * cell_1d_index] = water_residual[cell_1d_index]
        result[2 * cell_1d_index + 1] = gas_residual[cell_1d_index]
    return result


@numba.njit(parallel=True, cache=True)
def _compute_saturation_residual(
    oil_pressure_grid: ThreeDimensionalGrid,
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
    thickness_grid: ThreeDimensionalGrid,
    cell_size_x: float,
    cell_size_y: float,
    water_relative_mobility_grid: ThreeDimensionalGrid,
    oil_relative_mobility_grid: ThreeDimensionalGrid,
    gas_relative_mobility_grid: ThreeDimensionalGrid,
    face_transmissibilities_x: ThreeDimensionalGrid,
    face_transmissibilities_y: ThreeDimensionalGrid,
    face_transmissibilities_z: ThreeDimensionalGrid,
    oil_water_capillary_pressure_grid: ThreeDimensionalGrid,
    gas_oil_capillary_pressure_grid: ThreeDimensionalGrid,
    oil_density_grid: ThreeDimensionalGrid,
    water_density_grid: ThreeDimensionalGrid,
    gas_density_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    gravitational_constant: float,
    water_saturation_grid: ThreeDimensionalGrid,
    gas_saturation_grid: ThreeDimensionalGrid,
    old_water_saturation_grid: ThreeDimensionalGrid,
    old_gas_saturation_grid: ThreeDimensionalGrid,
    porosity_grid: ThreeDimensionalGrid,
    net_to_gross_grid: ThreeDimensionalGrid,
    time_step_in_days: float,
    net_water_well_rate_grid: ThreeDimensionalGrid,
    net_gas_well_rate_grid: ThreeDimensionalGrid,
    pressure_change_grid: ThreeDimensionalGrid,
    water_compressibility_grid: ThreeDimensionalGrid,
    gas_compressibility_grid: ThreeDimensionalGrid,
    rock_compressibility: float,
    pressure_boundaries: ThreeDimensionalGrid,
    flux_boundaries: ThreeDimensionalGrid,
    md_per_cp_to_ft2_per_psi_per_day: float,
) -> typing.Tuple[npt.NDArray, npt.NDArray]:
    """
    Compute the residual vector R(S) for the implicit saturation equations.

    For each cell ``(i, j, k)`` the residual equations are:

    ```
    R_w[i,j,k] = (phi * V / dt) * (Sw_new - Sw_old)
                    - sum_faces(F_w_face)
                    - qw_well
                    - PVT_correction_w

    R_g[i,j,k] = (phi * V / dt) * (Sg_new - Sg_old)
                    - sum_faces(F_g_face)
                    - qg_well
                    - PVT_correction_g
    ```

    All six faces are examined per cell.  Interior faces use
    `compute_fluxes_from_neighbour` with full upwinding.
    Boundary faces read from the padded ``pressure_boundaries`` /
    ``flux_boundaries`` arrays using the ``i_ghost = i_oob + 1`` offset
    convention, identical to the explicit saturation solver.

    :param oil_pressure_grid: Current oil pressure grid (psi), shape ``(nx, ny, nz)``.
    :param cell_count_x: Number of cells in x-direction.
    :param cell_count_y: Number of cells in y-direction.
    :param cell_count_z: Number of cells in z-direction.
    :param thickness_grid: Cell thickness grid (ft), shape ``(nx, ny, nz)``.
    :param cell_size_x: Cell size in x-direction (ft).
    :param cell_size_y: Cell size in y-direction (ft).
    :param water_relative_mobility_grid: Water relative mobility (ft²/psi·day).
    :param oil_relative_mobility_grid: Oil relative mobility (ft²/psi·day).
    :param gas_relative_mobility_grid: Gas relative mobility (ft²/psi·day).
    :param face_transmissibilities_x: x-direction face transmissibilities (mD·ft).
    :param face_transmissibilities_y: y-direction face transmissibilities (mD·ft).
    :param face_transmissibilities_z: z-direction face transmissibilities (mD·ft).
    :param oil_water_capillary_pressure_grid: Oil-water capillary pressure (psi).
    :param gas_oil_capillary_pressure_grid: Gas-oil capillary pressure (psi).
    :param oil_density_grid: Oil density (lb/ft³).
    :param water_density_grid: Water density (lb/ft³).
    :param gas_density_grid: Gas density (lb/ft³).
    :param elevation_grid: Cell elevation (ft).
    :param gravitational_constant: Gravitational constant conversion factor (lbf/lbm).
    :param water_saturation_grid: Current (Newton iterate) water saturation.
    :param gas_saturation_grid: Current (Newton iterate) gas saturation.
    :param old_water_saturation_grid: Water saturation at start of time step.
    :param old_gas_saturation_grid: Gas saturation at start of time step.
    :param porosity_grid: Porosity (fraction).
    :param net_to_gross_grid: Net-to-gross ratio (fraction).
    :param time_step_in_days: Time step size (days).
    :param net_water_well_rate_grid: Net water well rate per cell (ft³/day).
    :param net_gas_well_rate_grid: Net gas well rate per cell (ft³/day).
    :param pressure_change_grid: ``P_new - P_old`` (psi) for PVT volume correction.
    :param water_compressibility_grid: Water compressibility (psi⁻¹).
    :param gas_compressibility_grid: Gas compressibility (psi⁻¹).
    :param rock_compressibility: Scalar rock compressibility (psi⁻¹).
    :param pressure_boundaries: Padded pressure boundary grid, shape ``(nx+2, ny+2, nz+2)``.
        Ghost cell for out-of-bounds neighbour at ``(i_oob, j, k)`` is accessed at
        ``pressure_boundaries[i_oob + 1, j + 1, k + 1]``.  NaN → Neumann BC.
    :param flux_boundaries: Padded flux boundary grid, shape ``(nx+2, ny+2, nz+2)``.
        Read when ``pressure_boundaries[...]`` is NaN.
    :param md_per_cp_to_ft2_per_psi_per_day: Unit conversion factor.
    :return: Tuple ``(water_residual, gas_residual)``, each a 1-D array of length
        ``nx * ny * nz``.
    """
    total_cell_count = cell_count_x * cell_count_y * cell_count_z
    water_residual = np.zeros(total_cell_count)
    gas_residual = np.zeros(total_cell_count)

    for i in numba.prange(cell_count_x):  # type: ignore[attr-defined]
        for j in range(cell_count_y):
            for k in range(cell_count_z):
                cell_1d_index = to_1D_index(
                    i=i,
                    j=j,
                    k=k,
                    cell_count_x=cell_count_x,
                    cell_count_y=cell_count_y,
                    cell_count_z=cell_count_z,
                )

                cell_total_volume = (
                    cell_size_x
                    * cell_size_y
                    * thickness_grid[i, j, k]
                    * net_to_gross_grid[i, j, k]
                )
                cell_porosity = porosity_grid[i, j, k]
                cell_pore_volume = cell_total_volume * cell_porosity
                accumulation_coefficient = cell_pore_volume / time_step_in_days

                # Accumulation
                water_accumulation = accumulation_coefficient * (
                    water_saturation_grid[i, j, k] - old_water_saturation_grid[i, j, k]
                )
                gas_accumulation = accumulation_coefficient * (
                    gas_saturation_grid[i, j, k] - old_gas_saturation_grid[i, j, k]
                )

                net_water_flux = 0.0
                net_gas_flux = 0.0

                cell_water_mobility = water_relative_mobility_grid[i, j, k]
                cell_oil_mobility = oil_relative_mobility_grid[i, j, k]
                cell_gas_mobility = gas_relative_mobility_grid[i, j, k]
                cell_total_mobility = (
                    cell_water_mobility + cell_oil_mobility + cell_gas_mobility
                )
                cell_pressure = oil_pressure_grid[i, j, k]

                # EAST (i+1, j, k)
                east_i = i + 1
                if east_i < cell_count_x:
                    water_flux, _, gas_flux = compute_fluxes_from_neighbour(
                        cell_indices=(i, j, k),
                        neighbour_indices=(east_i, j, k),
                        oil_pressure_grid=oil_pressure_grid,
                        face_transmissibility=face_transmissibilities_x[i, j, k],
                        water_relative_mobility_grid=water_relative_mobility_grid,
                        oil_relative_mobility_grid=oil_relative_mobility_grid,
                        gas_relative_mobility_grid=gas_relative_mobility_grid,
                        oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                        gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                        oil_density_grid=oil_density_grid,
                        water_density_grid=water_density_grid,
                        gas_density_grid=gas_density_grid,
                        elevation_grid=elevation_grid,
                        gravitational_constant=gravitational_constant,
                        md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
                    )
                    net_water_flux += water_flux
                    net_gas_flux += gas_flux
                else:
                    # Boundary: ghost at padded index (east_i+1, j+1, k+1)
                    ghost_i, ghost_j, ghost_k = east_i + 1, j + 1, k + 1
                    pressure_boundary = pressure_boundaries[ghost_i, ghost_j, ghost_k]
                    t_conv = (
                        face_transmissibilities_x[i, j, k]
                        * md_per_cp_to_ft2_per_psi_per_day
                    )
                    if not np.isnan(pressure_boundary):
                        pressure_diff = pressure_boundary - cell_pressure
                        net_water_flux += cell_water_mobility * t_conv * pressure_diff
                        net_gas_flux += cell_gas_mobility * t_conv * pressure_diff
                    else:
                        flux_boundary = flux_boundaries[ghost_i, ghost_j, ghost_k]
                        if cell_total_mobility > 0.0:
                            net_water_flux += flux_boundary * (
                                cell_water_mobility / cell_total_mobility
                            )
                            net_gas_flux += flux_boundary * (
                                cell_gas_mobility / cell_total_mobility
                            )

                # WEST (i-1, j, k)
                west_i = i - 1
                if west_i >= 0:
                    water_flux, _, gas_flux = compute_fluxes_from_neighbour(
                        cell_indices=(i, j, k),
                        neighbour_indices=(west_i, j, k),
                        oil_pressure_grid=oil_pressure_grid,
                        face_transmissibility=face_transmissibilities_x[west_i, j, k],
                        water_relative_mobility_grid=water_relative_mobility_grid,
                        oil_relative_mobility_grid=oil_relative_mobility_grid,
                        gas_relative_mobility_grid=gas_relative_mobility_grid,
                        oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                        gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                        oil_density_grid=oil_density_grid,
                        water_density_grid=water_density_grid,
                        gas_density_grid=gas_density_grid,
                        elevation_grid=elevation_grid,
                        gravitational_constant=gravitational_constant,
                        md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
                    )
                    net_water_flux += water_flux
                    net_gas_flux += gas_flux
                else:
                    # Boundary: ghost at padded index (west_i+1, j+1, k+1) = (0, j+1, k+1)
                    ghost_i, ghost_j, ghost_k = west_i + 1, j + 1, k + 1
                    pressure_boundary = pressure_boundaries[ghost_i, ghost_j, ghost_k]
                    t_conv = (
                        face_transmissibilities_x[i, j, k]
                        * md_per_cp_to_ft2_per_psi_per_day
                    )
                    if not np.isnan(pressure_boundary):
                        pressure_diff = pressure_boundary - cell_pressure
                        net_water_flux += cell_water_mobility * t_conv * pressure_diff
                        net_gas_flux += cell_gas_mobility * t_conv * pressure_diff
                    else:
                        flux_boundary = flux_boundaries[ghost_i, ghost_j, ghost_k]
                        if cell_total_mobility > 0.0:
                            net_water_flux += flux_boundary * (
                                cell_water_mobility / cell_total_mobility
                            )
                            net_gas_flux += flux_boundary * (
                                cell_gas_mobility / cell_total_mobility
                            )

                # SOUTH (i, j+1, k)
                south_j = j + 1
                if south_j < cell_count_y:
                    water_flux, _, gas_flux = compute_fluxes_from_neighbour(
                        cell_indices=(i, j, k),
                        neighbour_indices=(i, south_j, k),
                        oil_pressure_grid=oil_pressure_grid,
                        face_transmissibility=face_transmissibilities_y[i, j, k],
                        water_relative_mobility_grid=water_relative_mobility_grid,
                        oil_relative_mobility_grid=oil_relative_mobility_grid,
                        gas_relative_mobility_grid=gas_relative_mobility_grid,
                        oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                        gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                        oil_density_grid=oil_density_grid,
                        water_density_grid=water_density_grid,
                        gas_density_grid=gas_density_grid,
                        elevation_grid=elevation_grid,
                        gravitational_constant=gravitational_constant,
                        md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
                    )
                    net_water_flux += water_flux
                    net_gas_flux += gas_flux
                else:
                    ghost_i, ghost_j, ghost_k = i + 1, south_j + 1, k + 1
                    pressure_boundary = pressure_boundaries[ghost_i, ghost_j, ghost_k]
                    t_conv = (
                        face_transmissibilities_y[i, j, k]
                        * md_per_cp_to_ft2_per_psi_per_day
                    )
                    if not np.isnan(pressure_boundary):
                        pressure_diff = pressure_boundary - cell_pressure
                        net_water_flux += cell_water_mobility * t_conv * pressure_diff
                        net_gas_flux += cell_gas_mobility * t_conv * pressure_diff
                    else:
                        flux_boundary = flux_boundaries[ghost_i, ghost_j, ghost_k]
                        if cell_total_mobility > 0.0:
                            net_water_flux += flux_boundary * (
                                cell_water_mobility / cell_total_mobility
                            )
                            net_gas_flux += flux_boundary * (
                                cell_gas_mobility / cell_total_mobility
                            )

                # NORTH (i, j-1, k)
                north_j = j - 1
                if north_j >= 0:
                    water_flux, _, gas_flux = compute_fluxes_from_neighbour(
                        cell_indices=(i, j, k),
                        neighbour_indices=(i, north_j, k),
                        oil_pressure_grid=oil_pressure_grid,
                        face_transmissibility=face_transmissibilities_y[i, north_j, k],
                        water_relative_mobility_grid=water_relative_mobility_grid,
                        oil_relative_mobility_grid=oil_relative_mobility_grid,
                        gas_relative_mobility_grid=gas_relative_mobility_grid,
                        oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                        gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                        oil_density_grid=oil_density_grid,
                        water_density_grid=water_density_grid,
                        gas_density_grid=gas_density_grid,
                        elevation_grid=elevation_grid,
                        gravitational_constant=gravitational_constant,
                        md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
                    )
                    net_water_flux += water_flux
                    net_gas_flux += gas_flux
                else:
                    ghost_i, ghost_j, ghost_k = i + 1, north_j + 1, k + 1
                    pressure_boundary = pressure_boundaries[ghost_i, ghost_j, ghost_k]
                    t_conv = (
                        face_transmissibilities_y[i, j, k]
                        * md_per_cp_to_ft2_per_psi_per_day
                    )
                    if not np.isnan(pressure_boundary):
                        pressure_diff = pressure_boundary - cell_pressure
                        net_water_flux += cell_water_mobility * t_conv * pressure_diff
                        net_gas_flux += cell_gas_mobility * t_conv * pressure_diff
                    else:
                        flux_boundary = flux_boundaries[ghost_i, ghost_j, ghost_k]
                        if cell_total_mobility > 0.0:
                            net_water_flux += flux_boundary * (
                                cell_water_mobility / cell_total_mobility
                            )
                            net_gas_flux += flux_boundary * (
                                cell_gas_mobility / cell_total_mobility
                            )

                # BOTTOM (i, j, k+1)
                bottom_k = k + 1
                if bottom_k < cell_count_z:
                    water_flux, _, gas_flux = compute_fluxes_from_neighbour(
                        cell_indices=(i, j, k),
                        neighbour_indices=(i, j, bottom_k),
                        oil_pressure_grid=oil_pressure_grid,
                        face_transmissibility=face_transmissibilities_z[i, j, k],
                        water_relative_mobility_grid=water_relative_mobility_grid,
                        oil_relative_mobility_grid=oil_relative_mobility_grid,
                        gas_relative_mobility_grid=gas_relative_mobility_grid,
                        oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                        gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                        oil_density_grid=oil_density_grid,
                        water_density_grid=water_density_grid,
                        gas_density_grid=gas_density_grid,
                        elevation_grid=elevation_grid,
                        gravitational_constant=gravitational_constant,
                        md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
                    )
                    net_water_flux += water_flux
                    net_gas_flux += gas_flux
                else:
                    ghost_i, ghost_j, ghost_k = i + 1, j + 1, bottom_k + 1
                    pressure_boundary = pressure_boundaries[ghost_i, ghost_j, ghost_k]
                    t_conv = (
                        face_transmissibilities_z[i, j, k]
                        * md_per_cp_to_ft2_per_psi_per_day
                    )
                    if not np.isnan(pressure_boundary):
                        pressure_diff = pressure_boundary - cell_pressure
                        net_water_flux += cell_water_mobility * t_conv * pressure_diff
                        net_gas_flux += cell_gas_mobility * t_conv * pressure_diff
                    else:
                        flux_boundary = flux_boundaries[ghost_i, ghost_j, ghost_k]
                        if cell_total_mobility > 0.0:
                            net_water_flux += flux_boundary * (
                                cell_water_mobility / cell_total_mobility
                            )
                            net_gas_flux += flux_boundary * (
                                cell_gas_mobility / cell_total_mobility
                            )

                # TOP (i, j, k-1)
                top_k = k - 1
                if top_k >= 0:
                    water_flux, _, gas_flux = compute_fluxes_from_neighbour(
                        cell_indices=(i, j, k),
                        neighbour_indices=(i, j, top_k),
                        oil_pressure_grid=oil_pressure_grid,
                        face_transmissibility=face_transmissibilities_z[i, j, top_k],
                        water_relative_mobility_grid=water_relative_mobility_grid,
                        oil_relative_mobility_grid=oil_relative_mobility_grid,
                        gas_relative_mobility_grid=gas_relative_mobility_grid,
                        oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                        gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                        oil_density_grid=oil_density_grid,
                        water_density_grid=water_density_grid,
                        gas_density_grid=gas_density_grid,
                        elevation_grid=elevation_grid,
                        gravitational_constant=gravitational_constant,
                        md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
                    )
                    net_water_flux += water_flux
                    net_gas_flux += gas_flux
                else:
                    ghost_i, ghost_j, ghost_k = i + 1, j + 1, top_k + 1
                    pressure_boundary = pressure_boundaries[ghost_i, ghost_j, ghost_k]
                    t_conv = (
                        face_transmissibilities_z[i, j, k]
                        * md_per_cp_to_ft2_per_psi_per_day
                    )
                    if not np.isnan(pressure_boundary):
                        pressure_diff = pressure_boundary - cell_pressure
                        net_water_flux += cell_water_mobility * t_conv * pressure_diff
                        net_gas_flux += cell_gas_mobility * t_conv * pressure_diff
                    else:
                        flux_boundary = flux_boundaries[ghost_i, ghost_j, ghost_k]
                        if cell_total_mobility > 0.0:
                            net_water_flux += flux_boundary * (
                                cell_water_mobility / cell_total_mobility
                            )
                            net_gas_flux += flux_boundary * (
                                cell_gas_mobility / cell_total_mobility
                            )

                # PVT volume correction
                delta_pressure = pressure_change_grid[i, j, k]
                negative_delta_pressure = -delta_pressure

                water_pvt_correction = (
                    old_water_saturation_grid[i, j, k]
                    * (water_compressibility_grid[i, j, k] + rock_compressibility)
                    * negative_delta_pressure
                    * accumulation_coefficient
                )
                gas_pvt_correction = (
                    old_gas_saturation_grid[i, j, k]
                    * (gas_compressibility_grid[i, j, k] + rock_compressibility)
                    * negative_delta_pressure
                    * accumulation_coefficient
                )

                water_residual[cell_1d_index] = (
                    water_accumulation
                    - net_water_flux
                    - net_water_well_rate_grid[i, j, k]
                    - water_pvt_correction
                )
                gas_residual[cell_1d_index] = (
                    gas_accumulation
                    - net_gas_flux
                    - net_gas_well_rate_grid[i, j, k]
                    - gas_pvt_correction
                )

    return water_residual, gas_residual


def compute_rock_fluid_properties(
    water_saturation_grid: ThreeDimensionalGrid,
    oil_saturation_grid: ThreeDimensionalGrid,
    gas_saturation_grid: ThreeDimensionalGrid,
    rock_properties: RockProperties[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
    config: Config,
    normalize_saturations: bool = False,
) -> typing.Tuple[
    RelativeMobilityGrids[ThreeDimensions],
    CapillaryPressureGrids[ThreeDimensions],
    typing.Any,
]:
    """
    Compute all saturation-dependent rock-fluid quantities at fixed pressure.

    Called at each Newton iteration. Returns updated relative mobilities,
    capillary pressures, and directional mobility grids.

    :param water_saturation_grid: Current water saturation grid.
    :param oil_saturation_grid: Current oil saturation grid.
    :param gas_saturation_grid: Current gas saturation grid.
    :param rock_properties: Rock properties (permeability, residual saturations, etc.).
    :param fluid_properties: Fluid properties (viscosities, etc.).
    :param config: Simulation configuration.
    :param normalize_saturations: If *True*, saturations are clamped and normalised before
        computing rock-fluid properties (used for numerical Jacobian perturbations).
    :return: 3-tuple of ``(relative_mobility_grids, capillary_pressure_grids, mobility_grids)``.
    """
    if normalize_saturations:
        water_sat_grid = np.clip(water_saturation_grid, 0.0, 1.0)
        gas_sat_grid = np.clip(gas_saturation_grid, 0.0, 1.0)
        oil_sat_grid = np.clip(oil_saturation_grid, 0.0, 1.0)
        total_sat_grid = water_sat_grid + oil_sat_grid + gas_sat_grid
        total_sat_grid = np.where(total_sat_grid > 0.0, total_sat_grid, 1.0)
        water_sat_grid = water_sat_grid / total_sat_grid
        oil_sat_grid = oil_sat_grid / total_sat_grid
        gas_sat_grid = gas_sat_grid / total_sat_grid
    else:
        water_sat_grid = water_saturation_grid
        oil_sat_grid = oil_saturation_grid
        gas_sat_grid = gas_saturation_grid

    _, relative_mobility_grids, capillary_pressure_grids = (
        build_rock_fluid_properties_grids(
            water_saturation_grid=water_sat_grid,  # type: ignore[arg-type]
            oil_saturation_grid=oil_sat_grid,  # type: ignore[arg-type]
            gas_saturation_grid=gas_sat_grid,  # type: ignore[arg-type]
            irreducible_water_saturation_grid=rock_properties.irreducible_water_saturation_grid,
            residual_oil_saturation_water_grid=rock_properties.residual_oil_saturation_water_grid,
            residual_oil_saturation_gas_grid=rock_properties.residual_oil_saturation_gas_grid,
            residual_gas_saturation_grid=rock_properties.residual_gas_saturation_grid,
            water_viscosity_grid=fluid_properties.water_viscosity_grid,
            oil_viscosity_grid=fluid_properties.oil_effective_viscosity_grid,
            gas_viscosity_grid=fluid_properties.gas_viscosity_grid,
            porosity_grid=rock_properties.porosity_grid,
            permeability_grid=rock_properties.absolute_permeability.mean,
            relative_permeability_table=config.rock_fluid_tables.relative_permeability_table,
            capillary_pressure_table=config.rock_fluid_tables.capillary_pressure_table,
            disable_capillary_effects=config.disable_capillary_effects,
            capillary_strength_factor=config.capillary_strength_factor,
            phase_appearance_tolerance=config.phase_appearance_tolerance,
        )
    )

    absolute_permeability = rock_properties.absolute_permeability
    (
        water_relative_mobility_grid,
        oil_relative_mobility_grid,
        gas_relative_mobility_grid,
    ) = relative_mobility_grids

    mobility_grids = compute_mobility_grids(
        absolute_permeability_x=absolute_permeability.x,
        absolute_permeability_y=absolute_permeability.y,
        absolute_permeability_z=absolute_permeability.z,
        water_relative_mobility_grid=water_relative_mobility_grid,
        oil_relative_mobility_grid=oil_relative_mobility_grid,
        gas_relative_mobility_grid=gas_relative_mobility_grid,
        md_per_cp_to_ft2_per_psi_per_day=c.MILLIDARCIES_PER_CENTIPOISE_TO_SQUARE_FEET_PER_PSI_PER_DAY,
    )
    return (relative_mobility_grids, capillary_pressure_grids, mobility_grids)


def _compute_residual(
    water_saturation_grid: ThreeDimensionalGrid,
    gas_saturation_grid: ThreeDimensionalGrid,
    old_water_saturation_grid: ThreeDimensionalGrid,
    old_gas_saturation_grid: ThreeDimensionalGrid,
    oil_pressure_grid: ThreeDimensionalGrid,
    pressure_change_grid: ThreeDimensionalGrid,
    face_transmissibilities: FaceTransmissibilities,
    capillary_pressure_grids: CapillaryPressureGrids[ThreeDimensions],
    relative_mobility_grids: RelativeMobilityGrids[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
    thickness_grid: ThreeDimensionalGrid,
    cell_size_x: float,
    cell_size_y: float,
    elevation_grid: ThreeDimensionalGrid,
    porosity_grid: ThreeDimensionalGrid,
    net_to_gross_grid: ThreeDimensionalGrid,
    time_step_in_days: float,
    gravitational_constant: float,
    water_compressibility_grid: ThreeDimensionalGrid,
    gas_compressibility_grid: ThreeDimensionalGrid,
    rock_compressibility: float,
    well_indices_cache: WellIndicesCache,
    injection_rates: PhaseTensorsProxy[float, ThreeDimensions],
    production_rates: PhaseTensorsProxy[float, ThreeDimensions],
    pressure_boundaries: ThreeDimensionalGrid,
    flux_boundaries: ThreeDimensionalGrid,
    dtype: npt.DTypeLike,
    md_per_cp_to_ft2_per_psi_per_day: float,
) -> typing.Tuple[npt.NDArray, npt.NDArray]:
    """
    Compute the residual from pre-computed saturation-dependent rock-fluid properties.

    :param water_saturation_grid: Current (Newton iterate) water saturation grid.
    :param gas_saturation_grid: Current (Newton iterate) gas saturation grid.
    :param old_water_saturation_grid: Water saturation at start of time step.
    :param old_gas_saturation_grid: Gas saturation at start of time step.
    :param oil_pressure_grid: Oil pressure grid (psi), fixed during Newton loop.
    :param pressure_change_grid: ``P_new - P_old`` (psi) for PVT correction.
    :param face_transmissibilities: Precomputed geometric face transmissibilities.
    :param capillary_pressure_grids: ``(Pcow, Pcgo)`` at current iterate.
    :param relative_mobility_grids: ``(lam_w, lam_o, lam_g)`` at current iterate.
    :param fluid_properties: Fluid properties (density, compressibility grids, etc.).
    :param cell_count_x: Number of cells in x-direction.
    :param cell_count_y: Number of cells in y-direction.
    :param cell_count_z: Number of cells in z-direction.
    :param thickness_grid: Cell thickness grid (ft).
    :param cell_size_x: Cell size in x-direction (ft).
    :param cell_size_y: Cell size in y-direction (ft).
    :param elevation_grid: Cell elevation grid (ft).
    :param porosity_grid: Porosity (fraction).
    :param net_to_gross_grid: Net-to-gross ratio (fraction).
    :param time_step_in_days: Time step size (days).
    :param gravitational_constant: Gravitational constant conversion factor (lbf/lbm).
    :param water_compressibility_grid: Water compressibility (psi⁻¹).
    :param gas_compressibility_grid: Gas compressibility (psi⁻¹).
    :param rock_compressibility: Scalar rock compressibility (psi⁻¹).
    :param well_indices_cache: Cache of well indices.
    :param injection_rates: Injection rates proxy.
    :param production_rates: Production rates proxy.
    :param pressure_boundaries: Padded pressure boundary grid, shape ``(nx+2, ny+2, nz+2)``.
    :param flux_boundaries: Padded flux boundary grid, shape ``(nx+2, ny+2, nz+2)``.
    :param dtype: NumPy dtype for array allocation.
    :param md_per_cp_to_ft2_per_psi_per_day: Unit conversion factor.
    :return: ``(water_residual, gas_residual)`` as 1-D arrays of length ``nx*ny*nz``.
    """
    oil_water_capillary_pressure_grid, gas_oil_capillary_pressure_grid = (
        capillary_pressure_grids
    )
    (
        water_relative_mobility_grid,
        oil_relative_mobility_grid,
        gas_relative_mobility_grid,
    ) = relative_mobility_grids

    net_water_well_rate_grid, _, net_gas_well_rate_grid = compute_well_rate_grids(
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
        well_indices_cache=well_indices_cache,
        injection_rates=injection_rates,
        production_rates=production_rates,
        dtype=dtype,
    )

    return _compute_saturation_residual(
        oil_pressure_grid=oil_pressure_grid,
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
        thickness_grid=thickness_grid,
        cell_size_x=cell_size_x,
        cell_size_y=cell_size_y,
        water_relative_mobility_grid=water_relative_mobility_grid,
        oil_relative_mobility_grid=oil_relative_mobility_grid,
        gas_relative_mobility_grid=gas_relative_mobility_grid,
        face_transmissibilities_x=face_transmissibilities.x,
        face_transmissibilities_y=face_transmissibilities.y,
        face_transmissibilities_z=face_transmissibilities.z,
        oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
        gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
        oil_density_grid=fluid_properties.oil_effective_density_grid,
        water_density_grid=fluid_properties.water_density_grid,
        gas_density_grid=fluid_properties.gas_density_grid,
        elevation_grid=elevation_grid,
        gravitational_constant=gravitational_constant,
        water_saturation_grid=water_saturation_grid,
        gas_saturation_grid=gas_saturation_grid,
        old_water_saturation_grid=old_water_saturation_grid,
        old_gas_saturation_grid=old_gas_saturation_grid,
        porosity_grid=porosity_grid,
        net_to_gross_grid=net_to_gross_grid,
        time_step_in_days=time_step_in_days,
        net_water_well_rate_grid=net_water_well_rate_grid,
        net_gas_well_rate_grid=net_gas_well_rate_grid,
        pressure_change_grid=pressure_change_grid,
        water_compressibility_grid=water_compressibility_grid,
        gas_compressibility_grid=gas_compressibility_grid,
        rock_compressibility=rock_compressibility,
        pressure_boundaries=pressure_boundaries,
        flux_boundaries=flux_boundaries,
        md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
    )


def compute_residual(
    water_saturation_grid: ThreeDimensionalGrid,
    oil_saturation_grid: ThreeDimensionalGrid,
    gas_saturation_grid: ThreeDimensionalGrid,
    old_water_saturation_grid: ThreeDimensionalGrid,
    old_gas_saturation_grid: ThreeDimensionalGrid,
    oil_pressure_grid: ThreeDimensionalGrid,
    pressure_change_grid: ThreeDimensionalGrid,
    rock_properties: RockProperties[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
    face_transmissibilities: FaceTransmissibilities,
    config: Config,
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
    thickness_grid: ThreeDimensionalGrid,
    cell_size_x: float,
    cell_size_y: float,
    elevation_grid: ThreeDimensionalGrid,
    time_step_in_days: float,
    gravitational_constant: float,
    water_compressibility_grid: ThreeDimensionalGrid,
    gas_compressibility_grid: ThreeDimensionalGrid,
    well_indices_cache: WellIndicesCache,
    injection_rates: PhaseTensorsProxy[float, ThreeDimensions],
    production_rates: PhaseTensorsProxy[float, ThreeDimensions],
    pressure_boundaries: ThreeDimensionalGrid,
    flux_boundaries: ThreeDimensionalGrid,
    dtype: npt.DTypeLike,
    md_per_cp_to_ft2_per_psi_per_day: float,
) -> typing.Tuple[npt.NDArray, npt.NDArray]:
    """
    Compute the full residual, (re-)building saturation-dependent properties first.

    :param water_saturation_grid: Current (Newton iterate) water saturation grid.
    :param oil_saturation_grid: Current (Newton iterate) oil saturation grid.
    :param gas_saturation_grid: Current (Newton iterate) gas saturation grid.
    :param old_water_saturation_grid: Water saturation at start of time step.
    :param old_gas_saturation_grid: Gas saturation at start of time step.
    :param oil_pressure_grid: Oil pressure grid (psi), fixed during Newton loop.
    :param pressure_change_grid: ``P_new - P_old`` (psi) for PVT correction.
    :param rock_properties: Rock properties.
    :param fluid_properties: Fluid properties.
    :param face_transmissibilities: Precomputed geometric face transmissibilities.
    :param config: Simulation configuration.
    :param cell_count_x: Number of cells in x-direction.
    :param cell_count_y: Number of cells in y-direction.
    :param cell_count_z: Number of cells in z-direction.
    :param thickness_grid: Cell thickness grid (ft).
    :param cell_size_x: Cell size in x-direction (ft).
    :param cell_size_y: Cell size in y-direction (ft).
    :param elevation_grid: Cell elevation grid (ft).
    :param time_step_in_days: Time step size (days).
    :param gravitational_constant: Gravitational constant conversion factor (lbf/lbm).
    :param water_compressibility_grid: Water compressibility (psi⁻¹).
    :param gas_compressibility_grid: Gas compressibility (psi⁻¹).
    :param well_indices_cache: Cache of well indices.
    :param injection_rates: Injection rates proxy.
    :param production_rates: Production rates proxy.
    :param pressure_boundaries: Padded pressure boundary grid, shape ``(nx+2, ny+2, nz+2)``.
    :param flux_boundaries: Padded flux boundary grid, shape ``(nx+2, ny+2, nz+2)``.
    :param dtype: NumPy dtype for array allocation.
    :param md_per_cp_to_ft2_per_psi_per_day: Unit conversion factor.
    :return: ``(water_residual, gas_residual)`` as 1-D arrays of length ``nx*ny*nz``.
    """
    (
        relative_mobility_grids,
        capillary_pressure_grids,
        _,
    ) = compute_rock_fluid_properties(
        water_saturation_grid=water_saturation_grid,
        oil_saturation_grid=oil_saturation_grid,
        gas_saturation_grid=gas_saturation_grid,
        rock_properties=rock_properties,
        fluid_properties=fluid_properties,
        config=config,
    )
    return _compute_residual(
        water_saturation_grid=water_saturation_grid,
        gas_saturation_grid=gas_saturation_grid,
        old_water_saturation_grid=old_water_saturation_grid,
        old_gas_saturation_grid=old_gas_saturation_grid,
        oil_pressure_grid=oil_pressure_grid,
        pressure_change_grid=pressure_change_grid,
        face_transmissibilities=face_transmissibilities,
        capillary_pressure_grids=capillary_pressure_grids,
        relative_mobility_grids=relative_mobility_grids,
        fluid_properties=fluid_properties,
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
        thickness_grid=thickness_grid,
        cell_size_x=cell_size_x,
        cell_size_y=cell_size_y,
        elevation_grid=elevation_grid,
        porosity_grid=rock_properties.porosity_grid,
        net_to_gross_grid=rock_properties.net_to_gross_ratio_grid,
        time_step_in_days=time_step_in_days,
        gravitational_constant=gravitational_constant,
        water_compressibility_grid=water_compressibility_grid,
        gas_compressibility_grid=gas_compressibility_grid,
        rock_compressibility=rock_properties.compressibility,
        well_indices_cache=well_indices_cache,
        injection_rates=injection_rates,
        production_rates=production_rates,
        pressure_boundaries=pressure_boundaries,
        flux_boundaries=flux_boundaries,
        dtype=dtype,
        md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
    )


def assemble_numerical_jacobian(
    saturation_vector: npt.NDArray,
    residual_base: npt.NDArray,
    total_cell_count: int,
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
    water_saturation_grid: ThreeDimensionalGrid,
    oil_saturation_grid: ThreeDimensionalGrid,
    gas_saturation_grid: ThreeDimensionalGrid,
    old_water_saturation_grid: ThreeDimensionalGrid,
    old_gas_saturation_grid: ThreeDimensionalGrid,
    oil_pressure_grid: ThreeDimensionalGrid,
    pressure_change_grid: ThreeDimensionalGrid,
    face_transmissibilities: FaceTransmissibilities,
    rock_properties: RockProperties[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
    config: Config,
    thickness_grid: ThreeDimensionalGrid,
    cell_size_x: float,
    cell_size_y: float,
    elevation_grid: ThreeDimensionalGrid,
    time_step_in_days: float,
    gravitational_constant: float,
    water_compressibility_grid: ThreeDimensionalGrid,
    gas_compressibility_grid: ThreeDimensionalGrid,
    well_indices_cache: WellIndicesCache,
    injection_rates: PhaseTensorsProxy[float, ThreeDimensions],
    production_rates: PhaseTensorsProxy[float, ThreeDimensions],
    pressure_boundaries: ThreeDimensionalGrid,
    flux_boundaries: ThreeDimensionalGrid,
    dtype: npt.DTypeLike,
    md_per_cp_to_ft2_per_psi_per_day: float,
) -> csr_matrix:
    """
    Assemble the saturation Jacobian using column-wise forward finite differences.

    :param saturation_vector: Current saturation vector ``[Sw_0, Sg_0, ...]``.
    :param residual_base: Base residual ``R(S)`` at current iterate.
    :param total_cell_count: Total number of cells ``nx * ny * nz``.
    :param cell_count_x: Number of cells in x-direction.
    :param cell_count_y: Number of cells in y-direction.
    :param cell_count_z: Number of cells in z-direction.
    :param water_saturation_grid: Current water saturation grid.
    :param oil_saturation_grid: Current oil saturation grid.
    :param gas_saturation_grid: Current gas saturation grid.
    :param old_water_saturation_grid: Water saturation at start of time step.
    :param old_gas_saturation_grid: Gas saturation at start of time step.
    :param oil_pressure_grid: Fixed oil pressure grid (psi).
    :param pressure_change_grid: ``P_new - P_old`` (psi).
    :param face_transmissibilities: Precomputed geometric face transmissibilities.
    :param rock_properties: Rock properties.
    :param fluid_properties: Fluid properties.
    :param config: Simulation configuration.
    :param thickness_grid: Cell thickness grid (ft).
    :param cell_size_x: Cell size in x-direction (ft).
    :param cell_size_y: Cell size in y-direction (ft).
    :param elevation_grid: Cell elevation grid (ft).
    :param time_step_in_days: Time step size (days).
    :param gravitational_constant: Gravitational constant conversion factor (lbf/lbm).
    :param water_compressibility_grid: Water compressibility (psi⁻¹).
    :param gas_compressibility_grid: Gas compressibility (psi⁻¹).
    :param well_indices_cache: Cache of well indices.
    :param injection_rates: Injection rates proxy.
    :param production_rates: Production rates proxy.
    :param pressure_boundaries: Padded pressure boundary grid, shape ``(nx+2, ny+2, nz+2)``.
    :param flux_boundaries: Padded flux boundary grid, shape ``(nx+2, ny+2, nz+2)``.
    :param dtype: NumPy dtype.
    :param md_per_cp_to_ft2_per_psi_per_day: Unit conversion factor.
    :return: Sparse Jacobian of shape ``(2N, 2N)`` in CSR format, where ``N = nx*ny*nz``.
    """
    rows = []
    cols = []
    vals = []

    machine_eps = np.finfo(dtype).eps  # type: ignore
    base_epsilon = float(np.sqrt(machine_eps))

    water_saturation_grid_f64 = water_saturation_grid.astype(np.float64, copy=True)
    oil_saturation_grid_f64 = oil_saturation_grid.astype(np.float64, copy=True)
    gas_saturation_grid_f64 = gas_saturation_grid.astype(np.float64, copy=True)

    residual_kwargs = dict(
        old_water_saturation_grid=old_water_saturation_grid,
        old_gas_saturation_grid=old_gas_saturation_grid,
        oil_pressure_grid=oil_pressure_grid,
        pressure_change_grid=pressure_change_grid,
        face_transmissibilities=face_transmissibilities,
        rock_properties=rock_properties,
        fluid_properties=fluid_properties,
        config=config,
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
        thickness_grid=thickness_grid,
        cell_size_x=cell_size_x,
        cell_size_y=cell_size_y,
        elevation_grid=elevation_grid,
        porosity_grid=rock_properties.porosity_grid,
        net_to_gross_grid=rock_properties.net_to_gross_ratio_grid,
        time_step_in_days=time_step_in_days,
        gravitational_constant=gravitational_constant,
        water_compressibility_grid=water_compressibility_grid,
        gas_compressibility_grid=gas_compressibility_grid,
        rock_compressibility=rock_properties.compressibility,
        well_indices_cache=well_indices_cache,
        injection_rates=injection_rates,
        production_rates=production_rates,
        pressure_boundaries=pressure_boundaries,
        flux_boundaries=flux_boundaries,
        dtype=dtype,
        md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
    )

    for cell_1d_index in range(total_cell_count):
        i, j, k = from_1D_index(
            cell_1d_index,
            cell_count_x,
            cell_count_y,
            cell_count_z,
        )

        # Affected cells: this cell + all face neighbours (for sparsity)
        affected_cell_indices = [cell_1d_index]
        for delta_i, delta_j, delta_k in (
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ):
            ni, nj, nk = i + delta_i, j + delta_j, k + delta_k
            if (
                0 <= ni < cell_count_x
                and 0 <= nj < cell_count_y
                and 0 <= nk < cell_count_z
            ):
                affected_cell_indices.append(
                    to_1D_index(
                        i=ni,
                        j=nj,
                        k=nk,
                        cell_count_x=cell_count_x,
                        cell_count_y=cell_count_y,
                        cell_count_z=cell_count_z,
                    )
                )

        cell_water_saturation = water_saturation_grid_f64[i, j, k]
        cell_gas_saturation = gas_saturation_grid_f64[i, j, k]
        cell_oil_saturation = 1.0 - cell_water_saturation - cell_gas_saturation

        for var_offset in range(2):
            column = 2 * cell_1d_index + var_offset
            s_j = saturation_vector[column]

            epsilon = base_epsilon * max(abs(s_j), 1.0)
            if epsilon > cell_oil_saturation:
                epsilon = -epsilon
                if s_j + epsilon < 0.0:
                    epsilon = (-s_j * 0.5) if s_j > 1e-15 else (base_epsilon * 0.01)

            original_water_saturation = water_saturation_grid_f64[i, j, k]
            original_gas_saturation = gas_saturation_grid_f64[i, j, k]
            original_oil_saturation = oil_saturation_grid_f64[i, j, k]

            if var_offset == 0:
                perturbed_water_saturation = original_water_saturation + epsilon
                perturbed_gas_saturation = original_gas_saturation
            else:
                perturbed_water_saturation = original_water_saturation
                perturbed_gas_saturation = original_gas_saturation + epsilon
            perturbed_oil_saturation = max(
                0.0, 1.0 - perturbed_water_saturation - perturbed_gas_saturation
            )

            water_saturation_grid_f64[i, j, k] = perturbed_water_saturation
            gas_saturation_grid_f64[i, j, k] = perturbed_gas_saturation
            oil_saturation_grid_f64[i, j, k] = perturbed_oil_saturation

            perturbed_water_residual, perturbed_gas_residual = compute_residual(
                water_saturation_grid=water_saturation_grid_f64,
                oil_saturation_grid=oil_saturation_grid_f64,
                gas_saturation_grid=gas_saturation_grid_f64,
                **residual_kwargs,  # type: ignore[arg-type]
            )
            residual_perturbed = interleave_residuals(
                water_residual=perturbed_water_residual,
                gas_residual=perturbed_gas_residual,
            )

            water_saturation_grid_f64[i, j, k] = original_water_saturation
            gas_saturation_grid_f64[i, j, k] = original_gas_saturation
            oil_saturation_grid_f64[i, j, k] = original_oil_saturation

            for affected_idx in affected_cell_indices:
                row_water = 2 * affected_idx
                dR_water = (
                    residual_perturbed[row_water] - residual_base[row_water]
                ) / epsilon
                if abs(dR_water) > 1e-30:
                    rows.append(row_water)
                    cols.append(column)
                    vals.append(dR_water)

                row_gas = 2 * affected_idx + 1
                dR_gas = (
                    residual_perturbed[row_gas] - residual_base[row_gas]
                ) / epsilon
                if abs(dR_gas) > 1e-30:
                    rows.append(row_gas)
                    cols.append(column)
                    vals.append(dR_gas)

    system_size = 2 * total_cell_count
    return coo_matrix(
        (
            np.array(vals, dtype=np.float64),
            (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32)),
        ),
        shape=(system_size, system_size),
        dtype=np.float64,
    ).tocsr()


def compute_relperm_and_capillary_pressure_derivative_grids(
    water_saturation_grid: ThreeDimensionalGrid,
    oil_saturation_grid: ThreeDimensionalGrid,
    gas_saturation_grid: ThreeDimensionalGrid,
    rock_properties: RockProperties[ThreeDimensions],
    rock_fluid_tables: RockFluidTables,
    disable_capillary_effects: bool = False,
    capillary_strength_factor: float = 1.0,
) -> typing.Tuple[
    ThreeDimensionalGrid,  # dkrw_dSw
    ThreeDimensionalGrid,  # dkrw_dSo
    ThreeDimensionalGrid,  # dkrw_dSg
    ThreeDimensionalGrid,  # dkro_dSw
    ThreeDimensionalGrid,  # dkro_dSo
    ThreeDimensionalGrid,  # dkro_dSg
    ThreeDimensionalGrid,  # dkrg_dSw
    ThreeDimensionalGrid,  # dkrg_dSo
    ThreeDimensionalGrid,  # dkrg_dSg
    ThreeDimensionalGrid,  # dPcow_dSw_eff
    ThreeDimensionalGrid,  # dPcow_dSg_eff
    ThreeDimensionalGrid,  # dPcgo_dSw_eff
    ThreeDimensionalGrid,  # dPcgo_dSg_eff
]:
    """
    Compute all relperm and capillary-pressure derivative grids for the analytical Jacobian.

    **Relperm derivatives** — all nine raw partials ``d kr_alpha / d S_beta``
    (with ``beta`` in ``{Sw, So, Sg}``) are returned unchanged.

    **Capillary-pressure derivatives projected onto (Sw, Sg) basis** — the
    constraint ``So = 1 - Sw - Sg`` gives ``dSo/dSw = dSo/dSg = -1``, so

        dPcow_dSw_eff = dPcow_dSw + dPcow_dSo * (-1)
        dPcow_dSg_eff =              dPcow_dSo * (-1)
        dPcgo_dSw_eff =              dPcgo_dSo * (-1)
        dPcgo_dSg_eff = dPcgo_dSg + dPcgo_dSo * (-1)

    :param water_saturation_grid: Current water saturation grid.
    :param oil_saturation_grid: Current oil saturation grid.
    :param gas_saturation_grid: Current gas saturation grid.
    :param rock_properties: Rock properties (residual saturations).
    :param rock_fluid_tables: Rock-fluid property tables.
    :param disable_capillary_effects: If *True*, all capillary derivatives are zero.
    :param capillary_strength_factor: Scale factor applied to capillary derivatives.
    :return: 13-tuple of derivative grids in the order listed above.
    """
    relperm_table = rock_fluid_tables.relative_permeability_table
    capillary_table = rock_fluid_tables.capillary_pressure_table

    irreducible_water_saturation_grid = (
        rock_properties.irreducible_water_saturation_grid
    )
    residual_oil_saturation_water_grid = (
        rock_properties.residual_oil_saturation_water_grid
    )
    residual_oil_saturation_gas_grid = rock_properties.residual_oil_saturation_gas_grid
    residual_gas_saturation_grid = rock_properties.residual_gas_saturation_grid

    relperm_derivatives = relperm_table.derivatives(
        water_saturation=water_saturation_grid,
        oil_saturation=oil_saturation_grid,
        gas_saturation=gas_saturation_grid,
        irreducible_water_saturation=irreducible_water_saturation_grid,
        residual_oil_saturation_water=residual_oil_saturation_water_grid,
        residual_oil_saturation_gas=residual_oil_saturation_gas_grid,
        residual_gas_saturation=residual_gas_saturation_grid,
    )

    dkrw_dSw = relperm_derivatives["dKrw_dSw"]
    dkrw_dSo = relperm_derivatives["dKrw_dSo"]
    dkrw_dSg = relperm_derivatives["dKrw_dSg"]
    dkro_dSw = relperm_derivatives["dKro_dSw"]
    dkro_dSo = relperm_derivatives["dKro_dSo"]
    dkro_dSg = relperm_derivatives["dKro_dSg"]
    dkrg_dSw = relperm_derivatives["dKrg_dSw"]
    dkrg_dSo = relperm_derivatives["dKrg_dSo"]
    dkrg_dSg = relperm_derivatives["dKrg_dSg"]

    if capillary_table is not None and not disable_capillary_effects:
        capillary_pressure_derivatives = capillary_table.derivatives(
            water_saturation=water_saturation_grid,
            oil_saturation=oil_saturation_grid,
            gas_saturation=gas_saturation_grid,
            irreducible_water_saturation=irreducible_water_saturation_grid,
            residual_oil_saturation_water=residual_oil_saturation_water_grid,
            residual_oil_saturation_gas=residual_oil_saturation_gas_grid,
            residual_gas_saturation=residual_gas_saturation_grid,
            porosity=rock_properties.porosity_grid,
            permeability=rock_properties.absolute_permeability.mean,
        )
        raw_dPcow_dSw = (
            capillary_pressure_derivatives["dPcow_dSw"] * capillary_strength_factor
        )
        raw_dPcow_dSo = (
            capillary_pressure_derivatives["dPcow_dSo"] * capillary_strength_factor
        )
        raw_dPcgo_dSo = (
            capillary_pressure_derivatives["dPcgo_dSo"] * capillary_strength_factor
        )
        raw_dPcgo_dSg = (
            capillary_pressure_derivatives["dPcgo_dSg"] * capillary_strength_factor
        )

        dPcow_dSw_eff = raw_dPcow_dSw - raw_dPcow_dSo
        dPcow_dSg_eff = -raw_dPcow_dSo
        dPcgo_dSw_eff = -raw_dPcgo_dSo
        dPcgo_dSg_eff = raw_dPcgo_dSg - raw_dPcgo_dSo
    else:
        zeros = np.zeros_like(water_saturation_grid, dtype=np.float64)
        dPcow_dSw_eff = zeros
        dPcow_dSg_eff = zeros.copy()
        dPcgo_dSw_eff = zeros.copy()
        dPcgo_dSg_eff = zeros.copy()

    return (  # type: ignore[return-value]
        dkrw_dSw,
        dkrw_dSo,
        dkrw_dSg,
        dkro_dSw,
        dkro_dSo,
        dkro_dSg,
        dkrg_dSw,
        dkrg_dSo,
        dkrg_dSg,
        dPcow_dSw_eff,
        dPcow_dSg_eff,
        dPcgo_dSw_eff,
        dPcgo_dSg_eff,
    )


@numba.njit(parallel=True, cache=True)
def _assemble_analytical_jacobian(
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
    thickness_grid: ThreeDimensionalGrid,
    cell_size_x: float,
    cell_size_y: float,
    oil_pressure_grid: ThreeDimensionalGrid,
    water_density_grid: ThreeDimensionalGrid,
    gas_density_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    gravitational_constant: float,
    face_transmissibilities_x: ThreeDimensionalGrid,
    face_transmissibilities_y: ThreeDimensionalGrid,
    face_transmissibilities_z: ThreeDimensionalGrid,
    water_relative_mobility_grid: ThreeDimensionalGrid,
    gas_relative_mobility_grid: ThreeDimensionalGrid,
    oil_water_capillary_pressure_grid: ThreeDimensionalGrid,
    gas_oil_capillary_pressure_grid: ThreeDimensionalGrid,
    dkrw_dSw_grid: ThreeDimensionalGrid,
    dkrw_dSo_grid: ThreeDimensionalGrid,
    dkrw_dSg_grid: ThreeDimensionalGrid,
    dkrg_dSw_grid: ThreeDimensionalGrid,
    dkrg_dSo_grid: ThreeDimensionalGrid,
    dkrg_dSg_grid: ThreeDimensionalGrid,
    dPcow_dSw_eff_grid: ThreeDimensionalGrid,
    dPcow_dSg_eff_grid: ThreeDimensionalGrid,
    dPcgo_dSw_eff_grid: ThreeDimensionalGrid,
    dPcgo_dSg_eff_grid: ThreeDimensionalGrid,
    water_viscosity_grid: ThreeDimensionalGrid,
    gas_viscosity_grid: ThreeDimensionalGrid,
    porosity_grid: ThreeDimensionalGrid,
    net_to_gross_grid: ThreeDimensionalGrid,
    time_step_in_days: float,
    md_per_cp_to_ft2_per_psi_per_day: float,
) -> typing.Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Assemble the inter-cell flux part of the saturation Jacobian analytically.

    Iterates over all real cells. For each cell and each of its six face
    neighbours the function emits Jacobian entries for both the water and gas
    residual equations with respect to `Sw` and `Sg` of both the current
    cell and the (interior) neighbour. Boundary neighbours (out-of-bounds
    indices) are skipped because their contribution to the Jacobian is zero because
    the boundary flux is prescribed (Dirichlet flux from `pressure_boundaries`)
    or constant (Neumann flux from `flux_boundaries`).

    The assembly follows a thread-safe prange-over-i-slices strategy identical
    to the implicit pressure solver: each i-slice writes into its own row of
    2-D thread-local buffers, so no races occur.  A sequential compaction pass
    at the end flattens those buffers into COO triplets.

    :param cell_count_x: Number of cells in x-direction.
    :param cell_count_y: Number of cells in y-direction.
    :param cell_count_z: Number of cells in z-direction.
    :param thickness_grid: Cell thickness grid (ft).
    :param cell_size_x: Cell size in x-direction (ft).
    :param cell_size_y: Cell size in y-direction (ft).
    :param oil_pressure_grid: Oil pressure grid (psi), shape `(nx, ny, nz)`.
    :param water_density_grid: Water density grid (lb/ft³).
    :param gas_density_grid: Gas density grid (lb/ft³).
    :param elevation_grid: Cell elevation grid (ft).
    :param gravitational_constant: Gravitational constant conversion factor (lbf/lbm).
    :param face_transmissibilities_x: x-direction face transmissibilities (mD·ft).
    :param face_transmissibilities_y: y-direction face transmissibilities (mD·ft).
    :param face_transmissibilities_z: z-direction face transmissibilities (mD·ft).
    :param water_relative_mobility_grid: Water relative mobility (ft²/psi·day).
    :param gas_relative_mobility_grid: Gas relative mobility (ft²/psi·day).
    :param oil_water_capillary_pressure_grid: Oil-water capillary pressure (psi).
    :param gas_oil_capillary_pressure_grid: Gas-oil capillary pressure (psi).
    :param dkrw_dSw_grid: `∂krw/∂Sw` grid.
    :param dkrw_dSo_grid: `∂krw/∂So` grid.
    :param dkrw_dSg_grid: `∂krw/∂Sg` grid.
    :param dkrg_dSw_grid: `∂krg/∂Sw` grid.
    :param dkrg_dSo_grid: `∂krg/∂So` grid.
    :param dkrg_dSg_grid: `∂krg/∂Sg` grid.
    :param dPcow_dSw_eff_grid: Effective `∂Pcow/∂Sw` (So eliminated) grid.
    :param dPcow_dSg_eff_grid: Effective `∂Pcow/∂Sg` (So eliminated) grid.
    :param dPcgo_dSw_eff_grid: Effective `∂Pcgo/∂Sw` (So eliminated) grid.
    :param dPcgo_dSg_eff_grid: Effective `∂Pcgo/∂Sg` (So eliminated) grid.
    :param water_viscosity_grid: Water viscosity (cP).
    :param gas_viscosity_grid: Gas viscosity (cP).
    :param porosity_grid: Porosity (fraction).
    :param net_to_gross_grid: Net-to-gross ratio (fraction).
    :param time_step_in_days: Time step size (days).
    :param md_per_cp_to_ft2_per_psi_per_day: Unit conversion factor.
    :return: COO triplet `(rows, cols, vals)` for the inter-cell Jacobian entries.
    """
    # Upper bound per i-slice: each cell has 6 faces, each face can produce up to
    # 8 entries (4 dof combos x 2 cells).  Only interior neighbours contribute
    # off-diagonal entries; the diagonal accumulation term is also included.
    cells_per_slice = cell_count_y * cell_count_z
    max_nnz_per_slice = cells_per_slice * (2 + 6 * 8)

    all_rows = np.empty((cell_count_x, max_nnz_per_slice), dtype=np.int32)
    all_cols = np.empty((cell_count_x, max_nnz_per_slice), dtype=np.int32)
    all_vals = np.empty((cell_count_x, max_nnz_per_slice), dtype=np.float64)
    slice_fill = np.zeros(cell_count_x, dtype=np.int64)

    for i in numba.prange(cell_count_x):  # type: ignore[attr-defined]
        local_ptr = 0

        for j in range(cell_count_y):
            for k in range(cell_count_z):
                cell_1d_index = to_1D_index(
                    i, j, k, cell_count_x, cell_count_y, cell_count_z
                )
                water_row = 2 * cell_1d_index
                gas_row = 2 * cell_1d_index + 1
                cell_water_saturation_column = 2 * cell_1d_index
                cell_gas_saturation_column = 2 * cell_1d_index + 1

                cell_volume = (
                    cell_size_x
                    * cell_size_y
                    * thickness_grid[i, j, k]
                    * net_to_gross_grid[i, j, k]
                )
                accumulation_coefficient = (
                    porosity_grid[i, j, k] * cell_volume / time_step_in_days
                )

                # Accumulation diagonal (dR_w/dSw and dR_g/dSg = phi*V/dt)
                all_rows[i, local_ptr] = water_row
                all_cols[i, local_ptr] = cell_water_saturation_column
                all_vals[i, local_ptr] = accumulation_coefficient
                local_ptr += 1

                all_rows[i, local_ptr] = gas_row
                all_cols[i, local_ptr] = cell_gas_saturation_column
                all_vals[i, local_ptr] = accumulation_coefficient
                local_ptr += 1

                # Projected relperm derivatives (So eliminated via dSo/dSw = dSo/dSg = -1)
                dkrw_dSw_i_eff = dkrw_dSw_grid[i, j, k] - dkrw_dSo_grid[i, j, k]
                dkrw_dSg_i_eff = dkrw_dSg_grid[i, j, k] - dkrw_dSo_grid[i, j, k]
                dkrg_dSw_i_eff = dkrg_dSw_grid[i, j, k] - dkrg_dSo_grid[i, j, k]
                dkrg_dSg_i_eff = dkrg_dSg_grid[i, j, k] - dkrg_dSo_grid[i, j, k]

                dPcow_dSw_i = dPcow_dSw_eff_grid[i, j, k]
                dPcow_dSg_i = dPcow_dSg_eff_grid[i, j, k]
                dPcgo_dSw_i = dPcgo_dSw_eff_grid[i, j, k]
                dPcgo_dSg_i = dPcgo_dSg_eff_grid[i, j, k]

                # Six face neighbours — identify (ni, nj, nk) and transmissibility
                for face in range(6):
                    if face == 0:  # East
                        ni, nj, nk = np.int64(i + 1), np.int64(j), np.int64(k)
                        transmissibility = face_transmissibilities_x[i, j, k]
                    elif face == 1:  # West
                        ni, nj, nk = np.int64(i - 1), np.int64(j), np.int64(k)
                        transmissibility = (
                            face_transmissibilities_x[i - 1, j, k]
                            if i > 0
                            else face_transmissibilities_x[i, j, k]
                        )
                    elif face == 2:  # South
                        ni, nj, nk = np.int64(i), np.int64(j + 1), np.int64(k)
                        transmissibility = face_transmissibilities_y[i, j, k]
                    elif face == 3:  # North
                        ni, nj, nk = np.int64(i), np.int64(j - 1), np.int64(k)
                        transmissibility = (
                            face_transmissibilities_y[i, j - 1, k]
                            if j > 0
                            else face_transmissibilities_y[i, j, k]
                        )
                    elif face == 4:  # Bottom
                        ni, nj, nk = np.int64(i), np.int64(j), np.int64(k + 1)
                        transmissibility = face_transmissibilities_z[i, j, k]
                    else:  # Top (face == 5)
                        ni, nj, nk = np.int64(i), np.int64(j), np.int64(k - 1)
                        transmissibility = (
                            face_transmissibilities_z[i, j, k - 1]
                            if k > 0
                            else face_transmissibilities_z[i, j, k]
                        )

                    # Skip boundary faces — prescribed BC flux has zero Jacobian w.r.t. S
                    if (
                        ni < 0
                        or ni >= cell_count_x
                        or nj < 0
                        or nj >= cell_count_y
                        or nk < 0
                        or nk >= cell_count_z
                    ):
                        continue

                    # Potential differences (matching compute_fluxes_from_neighbour)
                    oil_pressure_difference = (
                        oil_pressure_grid[ni, nj, nk] - oil_pressure_grid[i, j, k]
                    )
                    elevation_difference = (
                        elevation_grid[ni, nj, nk] - elevation_grid[i, j, k]
                    )
                    oil_water_capillary_pressure_difference = (
                        oil_water_capillary_pressure_grid[ni, nj, nk]
                        - oil_water_capillary_pressure_grid[i, j, k]
                    )
                    gas_oil_capillary_pressure_difference = (
                        gas_oil_capillary_pressure_grid[ni, nj, nk]
                        - gas_oil_capillary_pressure_grid[i, j, k]
                    )
                    water_pressure_difference = (
                        oil_pressure_difference
                        - oil_water_capillary_pressure_difference
                    )
                    gas_pressure_difference = (
                        oil_pressure_difference + gas_oil_capillary_pressure_difference
                    )

                    # Density upwinding (matching compute_fluxes_from_neighbour exactly)
                    upwind_water_density = max(
                        water_density_grid[ni, nj, nk], water_density_grid[i, j, k]
                    )
                    upwind_gas_density = min(
                        gas_density_grid[ni, nj, nk], gas_density_grid[i, j, k]
                    )

                    water_gravity_potential = (
                        upwind_water_density
                        * gravitational_constant
                        * elevation_difference
                        / 144.0
                    )
                    gas_gravity_potential = (
                        upwind_gas_density
                        * gravitational_constant
                        * elevation_difference
                        / 144.0
                    )

                    water_potential = (
                        water_pressure_difference + water_gravity_potential
                    )
                    gas_potential = gas_pressure_difference + gas_gravity_potential

                    water_neighbour_is_upwind = water_potential > 0.0
                    gas_neighbour_is_upwind = gas_potential > 0.0

                    neighbour_1d_index = to_1D_index(
                        i=ni,  # type: ignore[arg-type]
                        j=nj,  # type: ignore[arg-type]
                        k=nk,  # type: ignore[arg-type]
                        cell_count_x=cell_count_x,
                        cell_count_y=cell_count_y,
                        cell_count_z=cell_count_z,
                    )
                    neighbour_water_saturation_column = 2 * neighbour_1d_index
                    neighbour_gas_saturation_column = 2 * neighbour_1d_index + 1

                    # Projected relperm derivatives at neighbour
                    dkrw_dSw_n_eff = (
                        dkrw_dSw_grid[ni, nj, nk] - dkrw_dSo_grid[ni, nj, nk]
                    )
                    dkrw_dSg_n_eff = (
                        dkrw_dSg_grid[ni, nj, nk] - dkrw_dSo_grid[ni, nj, nk]
                    )
                    dkrg_dSw_n_eff = (
                        dkrg_dSw_grid[ni, nj, nk] - dkrg_dSo_grid[ni, nj, nk]
                    )
                    dkrg_dSg_n_eff = (
                        dkrg_dSg_grid[ni, nj, nk] - dkrg_dSo_grid[ni, nj, nk]
                    )

                    dPcow_dSw_n = dPcow_dSw_eff_grid[ni, nj, nk]
                    dPcow_dSg_n = dPcow_dSg_eff_grid[ni, nj, nk]
                    dPcgo_dSw_n = dPcgo_dSw_eff_grid[ni, nj, nk]
                    dPcgo_dSg_n = dPcgo_dSg_eff_grid[ni, nj, nk]

                    #  WATER Jacobian contributions
                    upwind_water_relative_mobility = (
                        water_relative_mobility_grid[ni, nj, nk]
                        if water_neighbour_is_upwind
                        else water_relative_mobility_grid[i, j, k]
                    )

                    if not water_neighbour_is_upwind:
                        inv_mu_w = (
                            1.0 / water_viscosity_grid[i, j, k]
                            if water_viscosity_grid[i, j, k] > 0.0
                            else 0.0
                        )
                        dFw_mob_dSw_i = (
                            dkrw_dSw_i_eff
                            * md_per_cp_to_ft2_per_psi_per_day
                            * inv_mu_w
                            * water_potential
                            * transmissibility
                        )
                        dFw_mob_dSg_i = (
                            dkrw_dSg_i_eff
                            * md_per_cp_to_ft2_per_psi_per_day
                            * inv_mu_w
                            * water_potential
                            * transmissibility
                        )
                        dFw_mob_dSw_n = 0.0
                        dFw_mob_dSg_n = 0.0
                    else:
                        inv_mu_w = (
                            1.0 / water_viscosity_grid[ni, nj, nk]
                            if water_viscosity_grid[ni, nj, nk] > 0.0
                            else 0.0
                        )
                        dFw_mob_dSw_i = 0.0
                        dFw_mob_dSg_i = 0.0
                        dFw_mob_dSw_n = (
                            dkrw_dSw_n_eff
                            * md_per_cp_to_ft2_per_psi_per_day
                            * inv_mu_w
                            * water_potential
                            * transmissibility
                        )
                        dFw_mob_dSg_n = (
                            dkrw_dSg_n_eff
                            * md_per_cp_to_ft2_per_psi_per_day
                            * inv_mu_w
                            * water_potential
                            * transmissibility
                        )

                    dFw_cap_dSw_i = (
                        upwind_water_relative_mobility
                        * dPcow_dSw_i
                        * transmissibility
                        * md_per_cp_to_ft2_per_psi_per_day
                    )
                    dFw_cap_dSg_i = (
                        upwind_water_relative_mobility
                        * dPcow_dSg_i
                        * transmissibility
                        * md_per_cp_to_ft2_per_psi_per_day
                    )
                    dFw_cap_dSw_n = (
                        upwind_water_relative_mobility
                        * (-dPcow_dSw_n)
                        * transmissibility
                        * md_per_cp_to_ft2_per_psi_per_day
                    )
                    dFw_cap_dSg_n = (
                        upwind_water_relative_mobility
                        * (-dPcow_dSg_n)
                        * transmissibility
                        * md_per_cp_to_ft2_per_psi_per_day
                    )

                    # dR_w/dS = -dF_w/dS
                    dRw_dSw_i = -(dFw_mob_dSw_i + dFw_cap_dSw_i)
                    dRw_dSg_i = -(dFw_mob_dSg_i + dFw_cap_dSg_i)
                    dRw_dSw_n = -(dFw_mob_dSw_n + dFw_cap_dSw_n)
                    dRw_dSg_n = -(dFw_mob_dSg_n + dFw_cap_dSg_n)

                    #  GAS Jacobian contributions
                    upwind_gas_relative_mobility = (
                        gas_relative_mobility_grid[ni, nj, nk]
                        if gas_neighbour_is_upwind
                        else gas_relative_mobility_grid[i, j, k]
                    )

                    if not gas_neighbour_is_upwind:
                        inv_mu_g = (
                            1.0 / gas_viscosity_grid[i, j, k]
                            if gas_viscosity_grid[i, j, k] > 0.0
                            else 0.0
                        )
                        dFg_mob_dSw_i = (
                            dkrg_dSw_i_eff
                            * md_per_cp_to_ft2_per_psi_per_day
                            * inv_mu_g
                            * gas_potential
                            * transmissibility
                        )
                        dFg_mob_dSg_i = (
                            dkrg_dSg_i_eff
                            * md_per_cp_to_ft2_per_psi_per_day
                            * inv_mu_g
                            * gas_potential
                            * transmissibility
                        )
                        dFg_mob_dSw_n = 0.0
                        dFg_mob_dSg_n = 0.0
                    else:
                        inv_mu_g = (
                            1.0 / gas_viscosity_grid[ni, nj, nk]
                            if gas_viscosity_grid[ni, nj, nk] > 0.0
                            else 0.0
                        )
                        dFg_mob_dSw_i = 0.0
                        dFg_mob_dSg_i = 0.0
                        dFg_mob_dSw_n = (
                            dkrg_dSw_n_eff
                            * md_per_cp_to_ft2_per_psi_per_day
                            * inv_mu_g
                            * gas_potential
                            * transmissibility
                        )
                        dFg_mob_dSg_n = (
                            dkrg_dSg_n_eff
                            * md_per_cp_to_ft2_per_psi_per_day
                            * inv_mu_g
                            * gas_potential
                            * transmissibility
                        )

                    dFg_cap_dSw_i = (
                        upwind_gas_relative_mobility
                        * dPcgo_dSw_i
                        * transmissibility
                        * md_per_cp_to_ft2_per_psi_per_day
                    )
                    dFg_cap_dSg_i = (
                        upwind_gas_relative_mobility
                        * dPcgo_dSg_i
                        * transmissibility
                        * md_per_cp_to_ft2_per_psi_per_day
                    )
                    dFg_cap_dSw_n = (
                        upwind_gas_relative_mobility
                        * (-dPcgo_dSw_n)
                        * transmissibility
                        * md_per_cp_to_ft2_per_psi_per_day
                    )
                    dFg_cap_dSg_n = (
                        upwind_gas_relative_mobility
                        * (-dPcgo_dSg_n)
                        * transmissibility
                        * md_per_cp_to_ft2_per_psi_per_day
                    )

                    dRg_dSw_i = -(dFg_mob_dSw_i + dFg_cap_dSw_i)
                    dRg_dSg_i = -(dFg_mob_dSg_i + dFg_cap_dSg_i)
                    dRg_dSw_n = -(dFg_mob_dSw_n + dFg_cap_dSw_n)
                    dRg_dSg_n = -(dFg_mob_dSg_n + dFg_cap_dSg_n)

                    # Write diagonal contributions (w.r.t. cell i's unknowns)
                    if dRw_dSw_i != 0.0:
                        all_rows[i, local_ptr] = water_row
                        all_cols[i, local_ptr] = cell_water_saturation_column
                        all_vals[i, local_ptr] = dRw_dSw_i
                        local_ptr += 1
                    if dRw_dSg_i != 0.0:
                        all_rows[i, local_ptr] = water_row
                        all_cols[i, local_ptr] = cell_gas_saturation_column
                        all_vals[i, local_ptr] = dRw_dSg_i
                        local_ptr += 1
                    if dRg_dSw_i != 0.0:
                        all_rows[i, local_ptr] = gas_row
                        all_cols[i, local_ptr] = cell_water_saturation_column
                        all_vals[i, local_ptr] = dRg_dSw_i
                        local_ptr += 1
                    if dRg_dSg_i != 0.0:
                        all_rows[i, local_ptr] = gas_row
                        all_cols[i, local_ptr] = cell_gas_saturation_column
                        all_vals[i, local_ptr] = dRg_dSg_i
                        local_ptr += 1

                    # Write off-diagonal contributions (w.r.t. neighbour's unknowns)
                    if dRw_dSw_n != 0.0:
                        all_rows[i, local_ptr] = water_row
                        all_cols[i, local_ptr] = neighbour_water_saturation_column
                        all_vals[i, local_ptr] = dRw_dSw_n
                        local_ptr += 1
                    if dRw_dSg_n != 0.0:
                        all_rows[i, local_ptr] = water_row
                        all_cols[i, local_ptr] = neighbour_gas_saturation_column
                        all_vals[i, local_ptr] = dRw_dSg_n
                        local_ptr += 1
                    if dRg_dSw_n != 0.0:
                        all_rows[i, local_ptr] = gas_row
                        all_cols[i, local_ptr] = neighbour_water_saturation_column
                        all_vals[i, local_ptr] = dRg_dSw_n
                        local_ptr += 1
                    if dRg_dSg_n != 0.0:
                        all_rows[i, local_ptr] = gas_row
                        all_cols[i, local_ptr] = neighbour_gas_saturation_column
                        all_vals[i, local_ptr] = dRg_dSg_n
                        local_ptr += 1

        slice_fill[i] = local_ptr

    # Sequential compaction
    total_nnz = 0
    for s in range(cell_count_x):
        total_nnz += slice_fill[s]

    out_rows = np.empty(total_nnz, dtype=np.int32)
    out_cols = np.empty(total_nnz, dtype=np.int32)
    out_vals = np.empty(total_nnz, dtype=np.float64)
    write_ptr = 0
    for s in range(cell_count_x):
        count = slice_fill[s]
        out_rows[write_ptr : write_ptr + count] = all_rows[s, :count]
        out_cols[write_ptr : write_ptr + count] = all_cols[s, :count]
        out_vals[write_ptr : write_ptr + count] = all_vals[s, :count]
        write_ptr += count

    return out_rows, out_cols, out_vals


def _assemble_jacobian_well_contributions(
    oil_pressure_grid: ThreeDimensionalGrid,
    water_viscosity_grid: ThreeDimensionalGrid,
    gas_viscosity_grid: ThreeDimensionalGrid,
    dkrw_dSw_grid: ThreeDimensionalGrid,
    dkrw_dSo_grid: ThreeDimensionalGrid,
    dkrw_dSg_grid: ThreeDimensionalGrid,
    dkrg_dSw_grid: ThreeDimensionalGrid,
    dkrg_dSo_grid: ThreeDimensionalGrid,
    dkrg_dSg_grid: ThreeDimensionalGrid,
    well_indices_cache: WellIndicesCache,
    injection_bhps: PhaseTensorsProxy[float, ThreeDimensions],
    production_bhps: PhaseTensorsProxy[float, ThreeDimensions],
    md_per_cp_to_ft2_per_psi_per_day: float,
) -> typing.Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Compute well contributions to the saturation Jacobian.

    Wells contribute only to the diagonal (each perforated cell couples only
    to itself). For BHP-controlled wells the phase well rate depends on
    relative permeability, so the saturation Jacobian has non-zero entries at
    those cells. Rate-controlled injection wells have a fixed rate, so their
    contribution is zero.

    For each BHP-controlled perforation the derivative is:

        d(q_alpha)/dSw_eff = WI * conv / mu_alpha * dkr_alpha/dSw_eff * (P_cell - BHP)
        dR_alpha/dSw       = -d(q_alpha)/dSw_eff

    :param oil_pressure_grid: Oil pressure grid (psi).
    :param water_viscosity_grid: Water viscosity grid (cP).
    :param gas_viscosity_grid: Gas viscosity grid (cP).
    :param dkrw_dSw_grid: ``∂krw/∂Sw`` grid.
    :param dkrw_dSo_grid: ``∂krw/∂So`` grid.
    :param dkrw_dSg_grid: ``∂krw/∂Sg`` grid.
    :param dkrg_dSw_grid: ``∂krg/∂Sw`` grid.
    :param dkrg_dSo_grid: ``∂krg/∂So`` grid.
    :param dkrg_dSg_grid: ``∂krg/∂Sg`` grid.
    :param well_indices_cache: Cache of well indices.
    :param injection_bhps: Injection bottom-hole pressures proxy.
    :param production_bhps: Production bottom-hole pressures proxy.
    :param md_per_cp_to_ft2_per_psi_per_day: Unit conversion factor.
    :return: COO triplet ``(rows, cols, vals)`` for the well Jacobian entries.
    """
    rows: typing.List[int] = []
    cols: typing.List[int] = []
    vals: typing.List[float] = []

    def _add_diagonal_entry(
        cell_1d_index: int,
        row_offset: int,
        col_offset: int,
        derivative_value: float,
    ) -> None:
        """Append a single (row, col, val) triplet for a diagonal well entry.

        :param cell_1d_index: Flat 1-D cell index.
        :param row_offset: 0 for water residual row, 1 for gas residual row.
        :param col_offset: 0 for Sw column, 1 for Sg column.
        :param derivative_value: Jacobian entry value.
        """
        if derivative_value == 0.0:
            return
        rows.append(2 * cell_1d_index + row_offset)
        cols.append(2 * cell_1d_index + col_offset)
        vals.append(derivative_value)

    for well_indices in well_indices_cache.injection.values():
        for perforation_index in well_indices:
            i, j, k = perforation_index.cell
            well_index = perforation_index.well_index
            cell_1d_index = perforation_index.cell_1d_index
            cell_pressure = typing.cast(float, oil_pressure_grid[i, j, k])
            water_bhp, _, gas_bhp = injection_bhps[i, j, k]

            if gas_bhp:
                drawdown = cell_pressure - gas_bhp
                gas_viscosity = typing.cast(float, gas_viscosity_grid[i, j, k])
                inverse_gas_viscosity = (
                    1.0 / gas_viscosity if gas_viscosity > 0.0 else 0.0
                )
                dkrg_dSw_eff = dkrg_dSw_grid[i, j, k] - dkrg_dSo_grid[i, j, k]
                dkrg_dSg_eff = dkrg_dSg_grid[i, j, k] - dkrg_dSo_grid[i, j, k]
                dqg_dSw = (
                    well_index
                    * md_per_cp_to_ft2_per_psi_per_day
                    * inverse_gas_viscosity
                    * dkrg_dSw_eff
                    * drawdown
                )
                dqg_dSg = (
                    well_index
                    * md_per_cp_to_ft2_per_psi_per_day
                    * inverse_gas_viscosity
                    * dkrg_dSg_eff
                    * drawdown
                )
                _add_diagonal_entry(cell_1d_index, 1, 0, -dqg_dSw)
                _add_diagonal_entry(cell_1d_index, 1, 1, -dqg_dSg)

            elif water_bhp:
                drawdown = cell_pressure - water_bhp
                water_viscosity = typing.cast(float, water_viscosity_grid[i, j, k])
                inverse_water_viscosity = (
                    1.0 / water_viscosity if water_viscosity > 0.0 else 0.0
                )
                dkrw_dSw_eff = dkrw_dSw_grid[i, j, k] - dkrw_dSo_grid[i, j, k]
                dkrw_dSg_eff = dkrw_dSg_grid[i, j, k] - dkrw_dSo_grid[i, j, k]
                dqw_dSw = (
                    well_index
                    * md_per_cp_to_ft2_per_psi_per_day
                    * inverse_water_viscosity
                    * dkrw_dSw_eff
                    * drawdown
                )
                dqw_dSg = (
                    well_index
                    * md_per_cp_to_ft2_per_psi_per_day
                    * inverse_water_viscosity
                    * dkrw_dSg_eff
                    * drawdown
                )
                _add_diagonal_entry(cell_1d_index, 0, 0, -dqw_dSw)
                _add_diagonal_entry(cell_1d_index, 0, 1, -dqw_dSg)

    for well_indices in well_indices_cache.production.values():
        for perforation_index in well_indices:
            i, j, k = perforation_index.cell
            cell_1d_index = perforation_index.cell_1d_index
            well_index = perforation_index.well_index
            cell_pressure = typing.cast(float, oil_pressure_grid[i, j, k])
            water_bhp, _, gas_bhp = production_bhps[i, j, k]

            if water_bhp:
                drawdown = cell_pressure - water_bhp
                water_viscosity = typing.cast(float, water_viscosity_grid[i, j, k])
                inverse_water_viscosity = (
                    1.0 / water_viscosity if water_viscosity > 0.0 else 0.0
                )
                dkrw_dSw_eff = dkrw_dSw_grid[i, j, k] - dkrw_dSo_grid[i, j, k]
                dkrw_dSg_eff = dkrw_dSg_grid[i, j, k] - dkrw_dSo_grid[i, j, k]
                dqw_dSw = (
                    well_index
                    * md_per_cp_to_ft2_per_psi_per_day
                    * inverse_water_viscosity
                    * dkrw_dSw_eff
                    * drawdown
                )
                dqw_dSg = (
                    well_index
                    * md_per_cp_to_ft2_per_psi_per_day
                    * inverse_water_viscosity
                    * dkrw_dSg_eff
                    * drawdown
                )
                _add_diagonal_entry(cell_1d_index, 0, 0, -dqw_dSw)
                _add_diagonal_entry(cell_1d_index, 0, 1, -dqw_dSg)

            if gas_bhp:
                drawdown = cell_pressure - gas_bhp
                gas_viscosity = typing.cast(float, gas_viscosity_grid[i, j, k])
                inverse_gas_viscosity = (
                    1.0 / gas_viscosity if gas_viscosity > 0.0 else 0.0
                )
                dkrg_dSw_eff = dkrg_dSw_grid[i, j, k] - dkrg_dSo_grid[i, j, k]
                dkrg_dSg_eff = dkrg_dSg_grid[i, j, k] - dkrg_dSo_grid[i, j, k]
                dqg_dSw = (
                    well_index
                    * md_per_cp_to_ft2_per_psi_per_day
                    * inverse_gas_viscosity
                    * dkrg_dSw_eff
                    * drawdown
                )
                dqg_dSg = (
                    well_index
                    * md_per_cp_to_ft2_per_psi_per_day
                    * inverse_gas_viscosity
                    * dkrg_dSg_eff
                    * drawdown
                )
                _add_diagonal_entry(cell_1d_index, 1, 0, -dqg_dSw)
                _add_diagonal_entry(cell_1d_index, 1, 1, -dqg_dSg)

    return (
        np.array(rows, dtype=np.int32),
        np.array(cols, dtype=np.int32),
        np.array(vals, dtype=np.float64),
    )


def assemble_analytical_jacobian(
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
    total_cell_count: int,
    thickness_grid: ThreeDimensionalGrid,
    cell_size_x: float,
    cell_size_y: float,
    water_saturation_grid: ThreeDimensionalGrid,
    oil_saturation_grid: ThreeDimensionalGrid,
    gas_saturation_grid: ThreeDimensionalGrid,
    oil_pressure_grid: ThreeDimensionalGrid,
    water_density_grid: ThreeDimensionalGrid,
    gas_density_grid: ThreeDimensionalGrid,
    water_viscosity_grid: ThreeDimensionalGrid,
    gas_viscosity_grid: ThreeDimensionalGrid,
    face_transmissibilities: FaceTransmissibilities,
    rock_properties: RockProperties[ThreeDimensions],
    injection_bhps: PhaseTensorsProxy[float, ThreeDimensions],
    production_bhps: PhaseTensorsProxy[float, ThreeDimensions],
    capillary_pressure_grids: CapillaryPressureGrids[ThreeDimensions],
    relative_mobility_grids: RelativeMobilityGrids[ThreeDimensions],
    elevation_grid: ThreeDimensionalGrid,
    gravitational_constant: float,
    time_step_in_days: float,
    config: Config,
    well_indices_cache: WellIndicesCache,
    md_per_cp_to_ft2_per_psi_per_day: float,
) -> csr_matrix:
    """
    Assemble the full analytical saturation Jacobian.

    Combines inter-cell flux derivatives and well-rate derivatives, both
    assembled as COO triplets and merged into a single CSR matrix.
    Duplicate (row, col) entries are summed automatically by the COO→CSR
    conversion, so diagonal contributions from multiple faces and wells
    accumulate correctly.

    :param cell_count_x: Number of cells in x-direction.
    :param cell_count_y: Number of cells in y-direction.
    :param cell_count_z: Number of cells in z-direction.
    :param total_cell_count: ``nx * ny * nz``.
    :param thickness_grid: Cell thickness grid (ft).
    :param cell_size_x: Cell size in x-direction (ft).
    :param cell_size_y: Cell size in y-direction (ft).
    :param water_saturation_grid: Current water saturation grid.
    :param oil_saturation_grid: Current oil saturation grid.
    :param gas_saturation_grid: Current gas saturation grid.
    :param oil_pressure_grid: Oil pressure grid (psi).
    :param water_density_grid: Water density grid (lb/ft³).
    :param gas_density_grid: Gas density grid (lb/ft³).
    :param water_viscosity_grid: Water viscosity grid (cP).
    :param gas_viscosity_grid: Gas viscosity grid (cP).
    :param face_transmissibilities: Precomputed geometric face transmissibilities.
    :param rock_properties: Rock properties.
    :param injection_bhps: Injection bottom-hole pressures proxy.
    :param production_bhps: Production bottom-hole pressures proxy.
    :param capillary_pressure_grids: ``(Pcow, Pcgo)`` at current iterate.
    :param relative_mobility_grids: ``(lam_w, lam_o, lam_g)`` at current iterate.
    :param elevation_grid: Cell elevation grid (ft).
    :param gravitational_constant: Gravitational constant conversion factor (lbf/lbm).
    :param time_step_in_days: Time step size (days).
    :param config: Simulation configuration.
    :param well_indices_cache: Cache of well indices.
    :param md_per_cp_to_ft2_per_psi_per_day: Unit conversion factor.
    :return: Sparse Jacobian of shape ``(2N, 2N)`` in CSR format.
    """
    (
        dkrw_dSw_grid,
        dkrw_dSo_grid,
        dkrw_dSg_grid,
        _dkro_dSw,
        _dkro_dSo,
        _dkro_dSg,
        dkrg_dSw_grid,
        dkrg_dSo_grid,
        dkrg_dSg_grid,
        dPcow_dSw_eff_grid,
        dPcow_dSg_eff_grid,
        dPcgo_dSw_eff_grid,
        dPcgo_dSg_eff_grid,
    ) = compute_relperm_and_capillary_pressure_derivative_grids(
        water_saturation_grid=water_saturation_grid,
        oil_saturation_grid=oil_saturation_grid,
        gas_saturation_grid=gas_saturation_grid,
        rock_properties=rock_properties,
        rock_fluid_tables=config.rock_fluid_tables,
        disable_capillary_effects=config.disable_capillary_effects,
        capillary_strength_factor=config.capillary_strength_factor,
    )

    oil_water_capillary_pressure_grid, gas_oil_capillary_pressure_grid = (
        capillary_pressure_grids
    )
    (
        water_relative_mobility_grid,
        _,
        gas_relative_mobility_grid,
    ) = relative_mobility_grids

    flux_rows, flux_cols, flux_vals = _assemble_analytical_jacobian(
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
        thickness_grid=thickness_grid,
        cell_size_x=cell_size_x,
        cell_size_y=cell_size_y,
        oil_pressure_grid=oil_pressure_grid,
        water_density_grid=water_density_grid,
        gas_density_grid=gas_density_grid,
        elevation_grid=elevation_grid,
        gravitational_constant=gravitational_constant,
        water_relative_mobility_grid=water_relative_mobility_grid,
        gas_relative_mobility_grid=gas_relative_mobility_grid,
        face_transmissibilities_x=face_transmissibilities.x,
        face_transmissibilities_y=face_transmissibilities.y,
        face_transmissibilities_z=face_transmissibilities.z,
        oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
        gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
        dkrw_dSw_grid=dkrw_dSw_grid,
        dkrw_dSo_grid=dkrw_dSo_grid,
        dkrw_dSg_grid=dkrw_dSg_grid,
        dkrg_dSw_grid=dkrg_dSw_grid,
        dkrg_dSo_grid=dkrg_dSo_grid,
        dkrg_dSg_grid=dkrg_dSg_grid,
        dPcow_dSw_eff_grid=dPcow_dSw_eff_grid,
        dPcow_dSg_eff_grid=dPcow_dSg_eff_grid,
        dPcgo_dSw_eff_grid=dPcgo_dSw_eff_grid,
        dPcgo_dSg_eff_grid=dPcgo_dSg_eff_grid,
        water_viscosity_grid=water_viscosity_grid,
        gas_viscosity_grid=gas_viscosity_grid,
        porosity_grid=rock_properties.porosity_grid,
        net_to_gross_grid=rock_properties.net_to_gross_ratio_grid,
        time_step_in_days=time_step_in_days,
        md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
    )

    well_rows, well_cols, well_vals = _assemble_jacobian_well_contributions(
        oil_pressure_grid=oil_pressure_grid,
        water_viscosity_grid=water_viscosity_grid,
        gas_viscosity_grid=gas_viscosity_grid,
        dkrw_dSw_grid=dkrw_dSw_grid,
        dkrw_dSo_grid=dkrw_dSo_grid,
        dkrw_dSg_grid=dkrw_dSg_grid,
        dkrg_dSw_grid=dkrg_dSw_grid,
        dkrg_dSo_grid=dkrg_dSo_grid,
        dkrg_dSg_grid=dkrg_dSg_grid,
        well_indices_cache=well_indices_cache,
        injection_bhps=injection_bhps,
        production_bhps=production_bhps,
        md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
    )

    system_size = 2 * total_cell_count
    combined_rows = np.concatenate([flux_rows, well_rows])
    combined_cols = np.concatenate([flux_cols, well_cols])
    combined_vals = np.concatenate([flux_vals, well_vals])
    return coo_matrix(
        (combined_vals, (combined_rows, combined_cols)),
        shape=(system_size, system_size),
        dtype=np.float64,
    ).tocsr()


def assemble_jacobian(
    config: Config,
    saturation_vector: npt.NDArray,
    residual_base: npt.NDArray,
    total_cell_count: int,
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
    water_saturation_grid: ThreeDimensionalGrid,
    oil_saturation_grid: ThreeDimensionalGrid,
    gas_saturation_grid: ThreeDimensionalGrid,
    old_water_saturation_grid: ThreeDimensionalGrid,
    old_gas_saturation_grid: ThreeDimensionalGrid,
    oil_pressure_grid: ThreeDimensionalGrid,
    pressure_change_grid: ThreeDimensionalGrid,
    face_transmissibilities: FaceTransmissibilities,
    rock_properties: RockProperties[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
    thickness_grid: ThreeDimensionalGrid,
    cell_size_x: float,
    cell_size_y: float,
    elevation_grid: ThreeDimensionalGrid,
    time_step_in_days: float,
    gravitational_constant: float,
    water_compressibility_grid: ThreeDimensionalGrid,
    gas_compressibility_grid: ThreeDimensionalGrid,
    well_indices_cache: WellIndicesCache,
    injection_bhps: PhaseTensorsProxy[float, ThreeDimensions],
    production_bhps: PhaseTensorsProxy[float, ThreeDimensions],
    injection_rates: PhaseTensorsProxy[float, ThreeDimensions],
    production_rates: PhaseTensorsProxy[float, ThreeDimensions],
    pressure_boundaries: ThreeDimensionalGrid,
    flux_boundaries: ThreeDimensionalGrid,
    dtype: npt.DTypeLike,
    capillary_pressure_grids: CapillaryPressureGrids[ThreeDimensions],
    relative_mobility_grids: RelativeMobilityGrids[ThreeDimensions],
    md_per_cp_to_ft2_per_psi_per_day: float,
) -> csr_matrix:
    """
    Dispatch Jacobian assembly to the numerical or analytical path.

    :param config: Simulation configuration.  ``config.jacobian_assembly_method``
        selects ``"analytical"`` or ``"numerical"``.
    :param saturation_vector: Current saturation vector (numerical path only).
    :param residual_base: Base residual at current iterate (numerical path only).
    :param total_cell_count: ``nx * ny * nz``.
    :param cell_count_x: Number of cells in x-direction.
    :param cell_count_y: Number of cells in y-direction.
    :param cell_count_z: Number of cells in z-direction.
    :param water_saturation_grid: Current water saturation grid.
    :param oil_saturation_grid: Current oil saturation grid.
    :param gas_saturation_grid: Current gas saturation grid.
    :param old_water_saturation_grid: Water saturation at start of time step.
    :param old_gas_saturation_grid: Gas saturation at start of time step.
    :param oil_pressure_grid: Fixed oil pressure grid (psi).
    :param pressure_change_grid: ``P_new - P_old`` (psi).
    :param face_transmissibilities: Precomputed geometric face transmissibilities.
    :param rock_properties: Rock properties.
    :param fluid_properties: Fluid properties.
    :param thickness_grid: Cell thickness grid (ft).
    :param cell_size_x: Cell size in x-direction (ft).
    :param cell_size_y: Cell size in y-direction (ft).
    :param elevation_grid: Cell elevation grid (ft).
    :param time_step_in_days: Time step size (days).
    :param gravitational_constant: Gravitational constant conversion factor (lbf/lbm).
    :param water_compressibility_grid: Water compressibility (psi⁻¹).
    :param gas_compressibility_grid: Gas compressibility (psi⁻¹).
    :param well_indices_cache: Cache of well indices.
    :param injection_bhps: Injection bottom-hole pressures proxy.
    :param production_bhps: Production bottom-hole pressures proxy.
    :param injection_rates: Injection rates proxy.
    :param production_rates: Production rates proxy.
    :param pressure_boundaries: Padded pressure boundary grid, shape ``(nx+2, ny+2, nz+2)``.
    :param flux_boundaries: Padded flux boundary grid, shape ``(nx+2, ny+2, nz+2)``.
    :param dtype: NumPy dtype.
    :param capillary_pressure_grids: ``(Pcow, Pcgo)`` at current iterate.
    :param relative_mobility_grids: ``(lam_w, lam_o, lam_g)`` at current iterate.
    :param md_per_cp_to_ft2_per_psi_per_day: Unit conversion factor.
    :return: Jacobian as a ``(2N x 2N)`` CSR sparse matrix.
    """
    if config.jacobian_assembly_method == "analytical":
        return assemble_analytical_jacobian(
            total_cell_count=total_cell_count,
            cell_count_x=cell_count_x,
            cell_count_y=cell_count_y,
            cell_count_z=cell_count_z,
            thickness_grid=thickness_grid,
            cell_size_x=cell_size_x,
            cell_size_y=cell_size_y,
            oil_pressure_grid=oil_pressure_grid,
            water_saturation_grid=water_saturation_grid,
            oil_saturation_grid=oil_saturation_grid,
            gas_saturation_grid=gas_saturation_grid,
            water_density_grid=fluid_properties.water_density_grid,
            gas_density_grid=fluid_properties.gas_density_grid,
            elevation_grid=elevation_grid,
            rock_properties=rock_properties,
            face_transmissibilities=face_transmissibilities,
            gravitational_constant=gravitational_constant,
            water_viscosity_grid=fluid_properties.water_viscosity_grid,
            gas_viscosity_grid=fluid_properties.gas_viscosity_grid,
            capillary_pressure_grids=capillary_pressure_grids,
            relative_mobility_grids=relative_mobility_grids,
            time_step_in_days=time_step_in_days,
            config=config,
            well_indices_cache=well_indices_cache,
            injection_bhps=injection_bhps,
            production_bhps=production_bhps,
            md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
        )

    return assemble_numerical_jacobian(
        saturation_vector=saturation_vector,
        residual_base=residual_base,
        total_cell_count=total_cell_count,
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
        water_saturation_grid=water_saturation_grid,
        oil_saturation_grid=oil_saturation_grid,
        gas_saturation_grid=gas_saturation_grid,
        old_water_saturation_grid=old_water_saturation_grid,
        old_gas_saturation_grid=old_gas_saturation_grid,
        oil_pressure_grid=oil_pressure_grid,
        pressure_change_grid=pressure_change_grid,
        face_transmissibilities=face_transmissibilities,
        rock_properties=rock_properties,
        fluid_properties=fluid_properties,
        config=config,
        thickness_grid=thickness_grid,
        cell_size_x=cell_size_x,
        cell_size_y=cell_size_y,
        elevation_grid=elevation_grid,
        time_step_in_days=time_step_in_days,
        gravitational_constant=gravitational_constant,
        water_compressibility_grid=water_compressibility_grid,
        gas_compressibility_grid=gas_compressibility_grid,
        well_indices_cache=well_indices_cache,
        injection_rates=injection_rates,
        production_rates=production_rates,
        pressure_boundaries=pressure_boundaries,
        flux_boundaries=flux_boundaries,
        dtype=dtype,
        md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
    )


def evolve_saturation(
    grid_shape: ThreeDimensions,
    cell_dimension: typing.Tuple[float, float],
    thickness_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    time_step_size: float,
    time: float,
    rock_properties: RockProperties[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
    face_transmissibilities: FaceTransmissibilities,
    pressure_boundaries: ThreeDimensionalGrid,
    flux_boundaries: ThreeDimensionalGrid,
    config: Config,
    well_indices_cache: WellIndicesCache,
    pressure_change_grid: ThreeDimensionalGrid,
    injection_rates: PhaseTensorsProxy[float, ThreeDimensions],
    production_rates: PhaseTensorsProxy[float, ThreeDimensions],
    injection_bhps: PhaseTensorsProxy[float, ThreeDimensions],
    production_bhps: PhaseTensorsProxy[float, ThreeDimensions],
    dtype: npt.DTypeLike = np.float64,
) -> EvolutionResult[ImplicitSaturationSolution, typing.List[NewtonConvergenceInfo]]:
    """
    Solve the implicit saturation equations using Newton-Raphson iteration.

    Operates on the `(nx, ny, nz)` grid.  Boundary conditions
    are communicated through the `pressure_boundaries` and `flux_boundaries`
    arrays of shape `(nx+2, ny+2, nz+2)`, which follow the same ghost-cell
    indexing convention as the explicit saturation and implicit pressure solvers:
    a boundary cell at out-of-bounds index `(i_oob, j, k)` is looked up at
    `pressure_boundaries[i_oob + 1, j + 1, k + 1]`.

    The Newton loop iterates on all `nx * ny * nz` cells simultaneously.
    Convergence is declared when the relative residual norm drops below
    `config.newton_tolerance`, or when the maximum saturation change per
    iteration is sufficiently small (quasi-equilibrium acceptance).

    :param grid_shape: Grid shape `(nx, ny, nz)`.
    :param cell_dimension: `(cell_size_x, cell_size_y)` in feet.
    :param thickness_grid: Cell thickness grid (ft), shape `(nx, ny, nz)`.
    :param elevation_grid: Cell elevation grid (ft), shape `(nx, ny, nz)`.
    :param time_step_size: Time step size in seconds.
    :param time: Total simulation time elapsed, this time step inclusive (seconds).
    :param rock_properties: Rock properties (permeability, porosity, residual saturations, etc.).
    :param fluid_properties: Fluid properties at the new pressure level.
    :param face_transmissibilities: Precomputed geometric face transmissibilities,
        shape `(nx, ny, nz)` each.
    :param pressure_boundaries: Padded Dirichlet BC array, shape `(nx+2, ny+2, nz+2)`.
        NaN indicates a Neumann face.
    :param flux_boundaries: Padded Neumann BC array, shape `(nx+2, ny+2, nz+2)`.
        Read when the corresponding `pressure_boundaries` entry is NaN.
    :param config: Simulation configuration.
    :param well_indices_cache: Cache of well indices for rate look-up.
    :param pressure_change_grid: `P_new - P_old` (psi) for PVT volume correction.
    :param injection_rates: Injection rates proxy (ft³/day per phase per cell).
    :param production_rates: Production rates proxy (ft³/day per phase per cell).
    :param injection_bhps: Injection bottom-hole pressures proxy (psi per phase per cell).
    :param production_bhps: Production bottom-hole pressures proxy (psi per phase per cell).
    :param dtype: NumPy dtype for numerical arrays.
    :return: `EvolutionResult` containing an `ImplicitSaturationSolution` and a list of
        `NewtonConvergenceInfo` records.
    """
    oil_pressure_grid = fluid_properties.pressure_grid
    porosity_grid = rock_properties.porosity_grid
    net_to_gross_grid = rock_properties.net_to_gross_ratio_grid
    rock_compressibility = rock_properties.compressibility
    cell_count_x, cell_count_y, cell_count_z = oil_pressure_grid.shape
    cell_size_x, cell_size_y = cell_dimension

    old_water_saturation_grid = fluid_properties.water_saturation_grid
    old_oil_saturation_grid = fluid_properties.oil_saturation_grid
    old_gas_saturation_grid = fluid_properties.gas_saturation_grid
    water_compressibility_grid = fluid_properties.water_compressibility_grid
    gas_compressibility_grid = fluid_properties.gas_compressibility_grid

    time_step_in_days = time_step_size * c.DAYS_PER_SECOND
    total_cell_count = cell_count_x * cell_count_y * cell_count_z

    gravitational_constant = (
        c.ACCELERATION_DUE_TO_GRAVITY_FEET_PER_SECONDS_SQUARE
        / c.GRAVITATIONAL_CONSTANT_LBM_FT_PER_LBF_S2
    )
    md_per_cp_to_ft2_per_psi_per_day = (
        c.MILLIDARCIES_PER_CENTIPOISE_TO_SQUARE_FEET_PER_PSI_PER_DAY
    )

    water_saturation_grid = old_water_saturation_grid.copy()
    oil_saturation_grid = old_oil_saturation_grid.copy()
    gas_saturation_grid = old_gas_saturation_grid.copy()

    saturation_vector = saturation_grids_to_vector(
        water_saturation_grid=water_saturation_grid,
        gas_saturation_grid=gas_saturation_grid,
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
    )

    # Shared kwargs passed to residual / Jacobian helpers
    shared_residual_kwargs = dict(
        old_water_saturation_grid=old_water_saturation_grid,
        old_gas_saturation_grid=old_gas_saturation_grid,
        oil_pressure_grid=oil_pressure_grid,
        pressure_change_grid=pressure_change_grid,
        face_transmissibilities=face_transmissibilities,
        fluid_properties=fluid_properties,
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
        thickness_grid=thickness_grid,
        cell_size_x=cell_size_x,
        cell_size_y=cell_size_y,
        elevation_grid=elevation_grid,
        time_step_in_days=time_step_in_days,
        gravitational_constant=gravitational_constant,
        water_compressibility_grid=water_compressibility_grid,
        gas_compressibility_grid=gas_compressibility_grid,
        well_indices_cache=well_indices_cache,
        injection_rates=injection_rates,
        production_rates=production_rates,
        pressure_boundaries=pressure_boundaries,
        flux_boundaries=flux_boundaries,
        dtype=dtype,
        md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
    )

    # Newton iteration state
    convergence_history: typing.List[NewtonConvergenceInfo] = []
    initial_residual_norm = 0.0
    converged = False
    final_iteration = 0
    final_residual_norm = 0.0
    best_residual_norm = float("inf")
    stagnation_count = 0

    stagnation_patience = config.newton_stagnation_patience
    stagnation_improvement_threshold = config.newton_stagnation_improvement_threshold
    weak_problem_saturation_threshold = config.newton_weak_problem_saturation_threshold
    minimum_step_size = float(np.sqrt(np.finfo(dtype).eps))  # type: ignore
    maximum_newton_iterations = config.maximum_newton_iterations
    newton_tolerance = config.newton_tolerance
    maximum_line_search_cuts = config.maximum_line_search_cuts
    maximum_saturation_change = config.maximum_saturation_change
    saturation_convergence_tolerance = config.saturation_convergence_tolerance

    for iteration in range(maximum_newton_iterations):
        # Recompute saturation-dependent rock-fluid properties at current iterate
        relative_mobility_grids, capillary_pressure_grids, _ = (
            compute_rock_fluid_properties(
                water_saturation_grid=water_saturation_grid,
                oil_saturation_grid=oil_saturation_grid,
                gas_saturation_grid=gas_saturation_grid,
                rock_properties=rock_properties,
                fluid_properties=fluid_properties,
                config=config,
                normalize_saturations=(config.jacobian_assembly_method == "numerical"),
            )
        )

        # Evaluate residual
        water_residual, gas_residual = _compute_residual(
            water_saturation_grid=water_saturation_grid,
            gas_saturation_grid=gas_saturation_grid,
            capillary_pressure_grids=capillary_pressure_grids,
            relative_mobility_grids=relative_mobility_grids,
            porosity_grid=porosity_grid,
            net_to_gross_grid=net_to_gross_grid,
            rock_compressibility=rock_compressibility,
            **shared_residual_kwargs,  # type: ignore[arg-type]
        )
        residual_vector = interleave_residuals(water_residual, gas_residual)
        residual_norm = float(np.linalg.norm(residual_vector))

        if iteration == 0:
            initial_residual_norm = max(residual_norm, 1e-30)

        relative_residual_norm = residual_norm / initial_residual_norm
        last_max_ds = (
            convergence_history[-1].max_saturation_update
            if convergence_history
            else float("inf")
        )

        # Convergence checks
        residual_converged = relative_residual_norm < newton_tolerance and iteration > 0
        saturation_converged = (
            (
                last_max_ds < saturation_convergence_tolerance
                and relative_residual_norm < 1e-3
            )
            or (
                last_max_ds < weak_problem_saturation_threshold
                and iteration >= 2
                and residual_norm <= best_residual_norm * 1.5
            )
        ) and iteration > 0

        if residual_converged or saturation_converged:
            converged = True
            final_iteration = iteration
            final_residual_norm = residual_norm
            reason = "residual" if residual_converged else "saturation change"
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Newton converged at iteration %d (%s): "
                    "||R||/||R0|| = %.2e, "
                    "max |ΔS| = %.2e",
                    iteration,
                    reason,
                    relative_residual_norm,
                    last_max_ds,
                )
            convergence_history.append(
                NewtonConvergenceInfo(
                    iteration=iteration,
                    residual_norm=residual_norm,
                    relative_residual_norm=relative_residual_norm,
                    max_saturation_update=0.0,
                    line_search_factor=1.0,
                )
            )
            break

        # Assemble Jacobian
        jacobian = assemble_jacobian(
            config=config,
            saturation_vector=saturation_vector,
            residual_base=residual_vector,
            total_cell_count=total_cell_count,
            cell_count_x=cell_count_x,
            cell_count_y=cell_count_y,
            cell_count_z=cell_count_z,
            water_saturation_grid=water_saturation_grid,
            oil_saturation_grid=oil_saturation_grid,
            gas_saturation_grid=gas_saturation_grid,
            old_water_saturation_grid=old_water_saturation_grid,
            old_gas_saturation_grid=old_gas_saturation_grid,
            oil_pressure_grid=oil_pressure_grid,
            pressure_change_grid=pressure_change_grid,
            face_transmissibilities=face_transmissibilities,
            rock_properties=rock_properties,
            fluid_properties=fluid_properties,
            thickness_grid=thickness_grid,
            cell_size_x=cell_size_x,
            cell_size_y=cell_size_y,
            elevation_grid=elevation_grid,
            time_step_in_days=time_step_in_days,
            gravitational_constant=gravitational_constant,
            water_compressibility_grid=water_compressibility_grid,
            gas_compressibility_grid=gas_compressibility_grid,
            well_indices_cache=well_indices_cache,
            injection_bhps=injection_bhps,
            production_bhps=production_bhps,
            injection_rates=injection_rates,
            production_rates=production_rates,
            pressure_boundaries=pressure_boundaries,
            flux_boundaries=flux_boundaries,
            dtype=dtype,
            md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
            capillary_pressure_grids=capillary_pressure_grids,
            relative_mobility_grids=relative_mobility_grids,
        )

        # Solve the linear system J * dS = -R
        saturation_change, _ = solve_linear_system(
            A_csr=jacobian,
            b=-residual_vector,
            solver=config.saturation_solver,
            preconditioner=config.saturation_preconditioner,
            rtol=config.saturation_convergence_tolerance,
            maximum_iterations=config.maximum_solver_iterations,
            fallback_to_direct=True,
        )

        linear_residual_norm = float(
            np.linalg.norm(jacobian @ saturation_change + residual_vector)
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Newton iteration %d: ||R||/||R0|| = %.2e, linear residual = %.2e",
                iteration,
                relative_residual_norm,
                linear_residual_norm,
            )

        # Damp Newton step if it exceeds the configured maximum saturation change
        max_raw_change = float(np.max(np.abs(saturation_change)))
        if max_raw_change > maximum_saturation_change:
            damping_factor = maximum_saturation_change / max_raw_change
            saturation_change = saturation_change * damping_factor
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "  Step damped by %.3f (max |ΔS| = %.4f > %s)",
                    damping_factor,
                    max_raw_change,
                    maximum_saturation_change,
                )

        # Backtracking line search
        line_search_factor = 1.0
        saturation_vector_trial = saturation_vector + saturation_change
        project_to_feasible(saturation_vector_trial)

        water_saturation_grid_trial = water_saturation_grid.copy()
        oil_saturation_grid_trial = oil_saturation_grid.copy()
        gas_saturation_grid_trial = gas_saturation_grid.copy()
        vector_to_saturation_grids(
            saturation_vector=saturation_vector_trial,
            water_saturation_grid=water_saturation_grid_trial,
            oil_saturation_grid=oil_saturation_grid_trial,
            gas_saturation_grid=gas_saturation_grid_trial,
            cell_count_x=cell_count_x,
            cell_count_y=cell_count_y,
            cell_count_z=cell_count_z,
        )

        for ls_iteration in range(maximum_line_search_cuts):
            water_residual_trial, gas_residual_trial = compute_residual(
                water_saturation_grid=water_saturation_grid_trial,
                oil_saturation_grid=oil_saturation_grid_trial,
                gas_saturation_grid=gas_saturation_grid_trial,
                rock_properties=rock_properties,
                config=config,
                **shared_residual_kwargs,  # type: ignore[arg-type]
            )
            residual_trial = interleave_residuals(
                water_residual=water_residual_trial,
                gas_residual=gas_residual_trial,
            )
            residual_trial_norm = float(np.linalg.norm(residual_trial))
            if residual_trial_norm < residual_norm:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "  Line search accepted at alpha = %.4f, ||R_trial|| = %.4e",
                        line_search_factor,
                        residual_trial_norm,
                    )
                break

            line_search_factor *= 0.5
            if (line_search_factor * max_raw_change) < minimum_step_size:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "  Line search hit precision floor at alpha = %.4e",
                        line_search_factor,
                    )
                break

            saturation_vector_trial = (
                saturation_vector + line_search_factor * saturation_change
            )
            project_to_feasible(saturation_vector_trial)
            water_saturation_grid_trial = water_saturation_grid.copy()
            oil_saturation_grid_trial = oil_saturation_grid.copy()
            gas_saturation_grid_trial = gas_saturation_grid.copy()
            vector_to_saturation_grids(
                saturation_vector=saturation_vector_trial,
                water_saturation_grid=water_saturation_grid_trial,
                oil_saturation_grid=oil_saturation_grid_trial,
                gas_saturation_grid=gas_saturation_grid_trial,
                cell_count_x=cell_count_x,
                cell_count_y=cell_count_y,
                cell_count_z=cell_count_z,
            )

        max_saturation_update = float(
            np.max(np.abs(saturation_vector_trial - saturation_vector))
        )
        saturation_vector = saturation_vector_trial
        water_saturation_grid = water_saturation_grid_trial
        oil_saturation_grid = oil_saturation_grid_trial
        gas_saturation_grid = gas_saturation_grid_trial

        convergence_history.append(
            NewtonConvergenceInfo(
                iteration=iteration,
                residual_norm=residual_norm,
                relative_residual_norm=relative_residual_norm,
                max_saturation_update=max_saturation_update,
                line_search_factor=line_search_factor,
            )
        )

        final_iteration = iteration + 1
        final_residual_norm = residual_norm

        # Negligible-update stagnation check
        if max_saturation_update < 1e-10:
            if relative_residual_norm < 1e-3:
                converged = True
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Newton converged (saturation stagnation) at iteration %d: "
                        "max |ΔS| = %.2e, "
                        "||R||/||R0|| = %.2e",
                        iteration,
                        max_saturation_update,
                        relative_residual_norm,
                    )
            elif residual_norm <= best_residual_norm * 1.05 and iteration >= 3:
                converged = True
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Newton converged (weak problem) at iteration %d: "
                        "max |ΔS| = %.2e, "
                        "||R||/||R0|| = %.2e",
                        iteration,
                        max_saturation_update,
                        relative_residual_norm,
                    )
            else:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Newton stagnated (negligible dS, residual not converged) "
                        "at iteration %d: "
                        "max |ΔS| = %.2e, "
                        "||R||/||R0|| = %.2e",
                        iteration,
                        max_saturation_update,
                        relative_residual_norm,
                    )
            break

        # Residual-plateau stagnation check
        if residual_norm < best_residual_norm * (
            1.0 - stagnation_improvement_threshold
        ):
            best_residual_norm = residual_norm
            stagnation_count = 0
        else:
            stagnation_count += 1

        if stagnation_count >= stagnation_patience and iteration >= stagnation_patience:
            if relative_residual_norm < 1e-3:
                converged = True
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Newton converged (residual plateau) at iteration %d: "
                        "||R||/||R0|| = %.2e",
                        iteration,
                        relative_residual_norm,
                    )
            else:
                logger.warning(
                    "Newton stagnated (residual flat) at iteration %d: "
                    "||R||/||R0|| = %.2e, "
                    "no improvement for %d iterations",
                    iteration,
                    relative_residual_norm,
                    stagnation_count,
                )
            break

    # Post-loop: accept weak-problem solutions that hit max iterations
    if (
        not converged
        and maximum_newton_iterations > 0
        and final_iteration >= maximum_newton_iterations
    ):
        last_max_ds = (
            convergence_history[-1].max_saturation_update
            if convergence_history
            else float("inf")
        )
        final_relative_residual = final_residual_norm / initial_residual_norm
        if (
            last_max_ds < weak_problem_saturation_threshold
            and final_relative_residual < 1.0
            and final_residual_norm <= best_residual_norm * 1.5
        ):
            converged = True
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Newton max iterations reached; accepting weak-problem solution: "
                    "max |ΔS| = %.2e, "
                    "||R||/||R0|| = %.2e",
                    last_max_ds,
                    final_relative_residual,
                )

    maximum_water_saturation_change = float(
        np.max(np.abs(water_saturation_grid - old_water_saturation_grid))
    )
    maximum_oil_saturation_change = float(
        np.max(np.abs(oil_saturation_grid - old_oil_saturation_grid))
    )
    maximum_gas_saturation_change = float(
        np.max(np.abs(gas_saturation_grid - old_gas_saturation_grid))
    )

    solution = ImplicitSaturationSolution(
        water_saturation_grid=water_saturation_grid.astype(dtype, copy=False),
        oil_saturation_grid=oil_saturation_grid.astype(dtype, copy=False),
        gas_saturation_grid=gas_saturation_grid.astype(dtype, copy=False),
        newton_iterations=final_iteration,
        final_residual_norm=final_residual_norm,
        maximum_water_saturation_change=maximum_water_saturation_change,
        maximum_oil_saturation_change=maximum_oil_saturation_change,
        maximum_gas_saturation_change=maximum_gas_saturation_change,
    )

    if converged:
        return EvolutionResult(
            value=solution,
            scheme="implicit",
            success=True,
            message=(
                f"Implicit saturation converged in {final_iteration} Newton iterations."
            ),
            metadata=convergence_history,
        )
    return EvolutionResult(
        value=solution,
        scheme="implicit",
        success=False,
        message=(
            f"Newton did not converge after {maximum_newton_iterations} iterations. "
            f"Final relative residual: "
            f"{final_residual_norm / initial_residual_norm:.2e}"
        ),
        metadata=convergence_history,
    )
