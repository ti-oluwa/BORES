import logging
import typing

import attrs
import numba
import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_matrix

from bores.blackoil.satfunc import SatFuncTables
from bores.config import Config
from bores.constants import c
from bores.datastructures import BottomHolePressures
from bores.grids.base import CapillaryPressureGrids, RelativeMobilityGrids
from bores.grids.rock_fluid import build_rock_fluid_properties_grids
from bores.reservoir import FluidProperties, Hysteresis, RockProperties
from bores.solvers.base import (
    Solution,
    compute_mobility_grids,
    from_1D_index,
    scale_linear_system,
    solve_linear_system,
    to_1D_index,
)
from bores.solvers.explicit.transport import compute_face_fluxes
from bores.solvers.implicit.linesearch import line_search
from bores.solvers.rates import WellRates
from bores.transmissibility import FaceTransmissibilities
from bores.types import ThreeDimensionalGrid, ThreeDimensions
from bores.wells.indices import WellsIndices

logger = logging.getLogger(__name__)

__all__ = ["solve_transport"]


@attrs.frozen
class ImplicitSaturationSolution:
    """Result of an implicit saturation solve."""

    water_saturation_grid: ThreeDimensionalGrid
    oil_saturation_grid: ThreeDimensionalGrid
    gas_saturation_grid: ThreeDimensionalGrid
    water_mass_grid: ThreeDimensionalGrid
    oil_mass_grid: ThreeDimensionalGrid
    free_gas_mass_grid: ThreeDimensionalGrid
    dissolved_gas_mass_in_oil_grid: ThreeDimensionalGrid
    dissolved_gas_mass_in_water_grid: ThreeDimensionalGrid
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
def pack_grids(
    water_saturation_grid: ThreeDimensionalGrid,
    gas_saturation_grid: ThreeDimensionalGrid,
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
) -> npt.NDArray:
    """
    Pack Sw and Sg from 3-D grids into a 1-D vector.

    Layout: `[Sw_0, Sg_0, Sw_1, Sg_1, ..., Sw_{N-1}, Sg_{N-1}]`
    where `N = cell_count_x * cell_count_y * cell_count_z`.

    :param water_saturation_grid: Water saturation grid, shape `(nx, ny, nz)`.
    :param gas_saturation_grid: Gas saturation grid, shape `(nx, ny, nz)`.
    :param cell_count_x: Number of cells in x-direction.
    :param cell_count_y: Number of cells in y-direction.
    :param cell_count_z: Number of cells in z-direction.
    :return: 1-D array of length `2 * nx * ny * nz`.
    """
    total_cell_count = cell_count_x * cell_count_y * cell_count_z
    saturation_vector = np.empty(2 * total_cell_count)
    for i in numba.prange(cell_count_x):  # type: ignore
        for j in range(cell_count_y):
            for k in range(cell_count_z):
                cell_idx = to_1D_index(
                    i=i,
                    j=j,
                    k=k,
                    cell_count_x=cell_count_x,
                    cell_count_y=cell_count_y,
                    cell_count_z=cell_count_z,
                )
                saturation_vector[2 * cell_idx] = water_saturation_grid[i, j, k]
                saturation_vector[2 * cell_idx + 1] = gas_saturation_grid[i, j, k]
    return saturation_vector


@numba.njit(cache=True, parallel=True)
def unpack_vector(
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

    Computes oil saturation as `So = 1 - Sw - Sg`.

    :param saturation_vector: 1-D array of length `2 * nx * ny * nz`.
    :param water_saturation_grid: Output water saturation grid (modified in-place).
    :param oil_saturation_grid: Output oil saturation grid (modified in-place).
    :param gas_saturation_grid: Output gas saturation grid (modified in-place).
    :param cell_count_x: Number of cells in x-direction.
    :param cell_count_y: Number of cells in y-direction.
    :param cell_count_z: Number of cells in z-direction.
    """
    for i in numba.prange(cell_count_x):  # type: ignore
        for j in range(cell_count_y):
            for k in range(cell_count_z):
                cell_idx = to_1D_index(
                    i=i,
                    j=j,
                    k=k,
                    cell_count_x=cell_count_x,
                    cell_count_y=cell_count_y,
                    cell_count_z=cell_count_z,
                )
                water_saturation = saturation_vector[2 * cell_idx]
                gas_saturation = saturation_vector[2 * cell_idx + 1]
                water_saturation_grid[i, j, k] = water_saturation
                gas_saturation_grid[i, j, k] = gas_saturation
                oil_saturation_grid[i, j, k] = max(0.0, 1.0 - water_saturation - gas_saturation)


@numba.njit(cache=True, parallel=True)
def project_to_feasible(saturation_vector: npt.NDArray) -> npt.NDArray:
    """
    Project a saturation vector onto the feasible set.

    Enforces `Sw >= 0`, `Sg >= 0`, `Sw + Sg <= 1` by proportional
    scaling when the sum exceeds unity.

    :param saturation_vector: 1-D vector `[Sw_0, Sg_0, ...]` modified in-place.
    :return: The same vector after projection.
    """
    total_cell_count = len(saturation_vector) // 2
    for cell_idx in numba.prange(total_cell_count):  # type: ignore
        water_saturation = max(0.0, saturation_vector[2 * cell_idx])
        gas_saturation = max(0.0, saturation_vector[2 * cell_idx + 1])
        total_saturation = water_saturation + gas_saturation
        if total_saturation > 1.0:
            water_saturation = water_saturation / total_saturation
            gas_saturation = gas_saturation / total_saturation
        saturation_vector[2 * cell_idx] = water_saturation
        saturation_vector[2 * cell_idx + 1] = gas_saturation
    return saturation_vector


@numba.njit(cache=True, parallel=True)
def interleave_residuals(
    water_residual: npt.NDArray,
    gas_residual: npt.NDArray,
) -> npt.NDArray:
    """
    Interleave water and gas residual arrays into a single vector.

    Layout: `[R_w_0, R_g_0, R_w_1, R_g_1, ..., R_w_{N-1}, R_g_{N-1}]`.

    :param water_residual: 1-D water residual array of length N.
    :param gas_residual: 1-D gas residual array of length N.
    :return: Interleaved 1-D array of length 2 * N.
    """
    total_cell_count = len(water_residual)
    result = np.empty(2 * total_cell_count)
    for cell_idx in numba.prange(total_cell_count):  # type: ignore
        result[2 * cell_idx] = water_residual[cell_idx]
        result[2 * cell_idx + 1] = gas_residual[cell_idx]
    return result


@numba.njit(parallel=True, cache=True)
def _assemble_saturation_residuals(
    pressure_grid: ThreeDimensionalGrid,
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
    pore_volume_grid: ThreeDimensionalGrid,
    water_relative_mobility_grid: ThreeDimensionalGrid,
    oil_relative_mobility_grid: ThreeDimensionalGrid,
    gas_relative_mobility_grid: ThreeDimensionalGrid,
    face_transmissibilities_x: ThreeDimensionalGrid,
    face_transmissibilities_y: ThreeDimensionalGrid,
    face_transmissibilities_z: ThreeDimensionalGrid,
    oil_water_capillary_pressure_grid: ThreeDimensionalGrid,
    gas_oil_capillary_pressure_grid: ThreeDimensionalGrid,
    water_density_grid: ThreeDimensionalGrid,
    oil_density_grid: ThreeDimensionalGrid,
    gas_density_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    gravitational_constant: float,
    water_saturation_grid: ThreeDimensionalGrid,
    gas_saturation_grid: ThreeDimensionalGrid,
    old_water_mass_grid: ThreeDimensionalGrid,
    old_free_gas_mass_grid: ThreeDimensionalGrid,
    old_dissolved_gas_mass_in_oil_grid: ThreeDimensionalGrid,
    old_dissolved_gas_mass_in_water_grid: ThreeDimensionalGrid,
    solution_gas_to_oil_ratio_grid: ThreeDimensionalGrid,
    gas_solubility_in_water_grid: ThreeDimensionalGrid,
    gas_formation_volume_factor_grid: ThreeDimensionalGrid,
    oil_formation_volume_factor_grid: ThreeDimensionalGrid,
    water_formation_volume_factor_grid: ThreeDimensionalGrid,
    time_step_in_days: float,
    net_water_well_rate_grid: ThreeDimensionalGrid,
    net_oil_well_rate_grid: ThreeDimensionalGrid,
    net_gas_well_rate_grid: ThreeDimensionalGrid,
    pressure_boundaries: ThreeDimensionalGrid,
    flux_boundaries: ThreeDimensionalGrid,
    md_per_cp_to_ft2_per_psi_per_day: float,
    bbl_to_ft3: float,
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Assemble the mass-based residual vector R(S) for the implicit saturation equations.

    For each cell `(i, j, k)` the residual equations are:

    Water:

        R_w = (phi*V/dt) * (current_water_density*Sw_new - old_water_density*Sw_old)
              - sum_faces(upwind_water_density * F_w_face)
              - water_density_cell * q_w_well

    Gas (free + dissolved in oil + dissolved in water):

        M_g_new = current_gas_density*Sg_new + current_oil_density*alpha_Rs_new*So_new + current_water_density*alpha_Rsw_new*Sw_new
        M_g_old = old_gas_density*Sg_old + old_oil_density*alpha_Rs_old*So_old + old_water_density*alpha_Rsw_old*Sw_old

        R_g = (phi*V/dt) * (M_g_new - M_g_old)
              - sum_faces(upwind_gas_density*F_g + upwind_oil_density*alpha_Rs_upwind*F_o + upwind_water_density*alpha_Rsw_upwind*F_w)
              - (gas_density_cell*q_g + oil_density_cell*alpha_Rs_cell*q_o + water_density_cell*alpha_Rsw_cell*q_w)

    where `alpha_Rs = Rs * Bg / Bo` and `alpha_Rsw = Rsw * Bg / Bw` (both dimensionless).

    Densities and Rs/Rsw grids are **frozen during Newton** (they depend only on
    pressure, which is fixed in the sequential implicit scheme). They appear as
    fixed coefficients: `∂rho/∂S = 0` and `∂Rs/∂S = 0` exactly.

    Boundary faces use the interior cell's density for mass weighting, consistent
    with the explicit solver convention.

    :param pressure_grid: Current oil pressure grid (psi), shape `(nx, ny, nz)`.
    :param cell_count_x: Number of cells in x-direction.
    :param cell_count_y: Number of cells in y-direction.
    :param cell_count_z: Number of cells in z-direction.
    :param water_relative_mobility_grid: Water relative mobility (ft²/psi·day).
    :param oil_relative_mobility_grid: Oil relative mobility (ft²/psi·day).
    :param gas_relative_mobility_grid: Gas relative mobility (ft²/psi·day).
    :param face_transmissibilities_x: x-direction face transmissibilities (mD·ft).
    :param face_transmissibilities_y: y-direction face transmissibilities (mD·ft).
    :param face_transmissibilities_z: z-direction face transmissibilities (mD·ft).
    :param oil_water_capillary_pressure_grid: Oil-water capillary pressure (psi).
    :param gas_oil_capillary_pressure_grid: Gas-oil capillary pressure (psi).
    :param water_density_grid: Water density at new (current) pressure (lb/ft³).
    :param oil_density_grid: Oil effective density at new pressure (lb/ft³).
    :param gas_density_grid: Gas density at new pressure (lb/ft³).
    :param elevation_grid: Cell elevation (ft).
    :param gravitational_constant: g/gc conversion factor (lbf/lbm).
    :param water_saturation_grid: Current Newton-iterate water saturation.
    :param gas_saturation_grid: Current Newton-iterate gas saturation.
    :param old_water_saturation_grid: Water saturation at start of time step.
    :param old_oil_saturation_grid: Oil saturation at start of time step.
    :param old_gas_saturation_grid: Gas saturation at start of time step.
    :param old_water_density_grid: Water density at start-of-step pressure (lb/ft³).
    :param old_oil_density_grid: Oil effective density at start-of-step pressure (lb/ft³).
    :param old_gas_density_grid: Gas density at start-of-step pressure (lb/ft³).
    :param solution_gas_to_oil_ratio_grid: Rs at current (new) pressure (SCF/STB).
    :param gas_solubility_in_water_grid: Rsw at current pressure (SCF/STB).
    :param gas_formation_volume_factor_grid: Bg at current pressure (bbl/SCF).
    :param oil_formation_volume_factor_grid: Bo at current pressure (bbl/STB).
    :param water_formation_volume_factor_grid: Bw at current pressure (bbl/STB).
    :param time_step_in_days: Time step size (days).
    :param net_water_well_rate_grid: Net water well mass rate per cell (lbm/day).
    :param net_oil_well_rate_grid: Net oil well mass rate per cell (lbm/day).
    :param net_gas_well_rate_grid: Net gas well mass rate per cell (lbm/day).
    :param pressure_boundaries: Padded boundary pressure grid, shape `(nx+2, ny+2, nz+2)`.
        Ghost cell for out-of-bounds neighbour at `(i_oob, j, k)` is accessed at
        `pressure_boundaries[i_oob + 1, j + 1, k + 1]`. NaN indicates Neumann BC.
    :param flux_boundaries: Padded flux boundary grid, shape `(nx+2, ny+2, nz+2)`.
    :param md_per_cp_to_ft2_per_psi_per_day: Unit conversion factor.
    :return: Tuple `(water_residual, gas_residual)`, each a 1-D array of length
        `nx * ny * nz`.
    """
    total_cell_count = cell_count_x * cell_count_y * cell_count_z
    water_residual = np.zeros(total_cell_count, dtype=np.float64)
    gas_residual = np.zeros(total_cell_count, dtype=np.float64)

    for i in numba.prange(cell_count_x):  # type: ignore
        for j in range(cell_count_y):
            for k in range(cell_count_z):
                cell_idx = to_1D_index(
                    i=i,
                    j=j,
                    k=k,
                    cell_count_x=cell_count_x,
                    cell_count_y=cell_count_y,
                    cell_count_z=cell_count_z,
                )
                cell_pore_volume = pore_volume_grid[i, j, k]
                accumulation_coefficient = cell_pore_volume / time_step_in_days

                # Current-pressure densities
                current_water_density = water_density_grid[i, j, k]
                current_gas_density = gas_density_grid[i, j, k]

                # Guard against zero density
                if current_gas_density < 1e-30:
                    current_gas_density = 1e-30
                if current_water_density < 1e-30:
                    current_water_density = 1e-30

                safe_oil_fvf = oil_formation_volume_factor_grid[i, j, k]
                safe_water_fvf = water_formation_volume_factor_grid[i, j, k]
                safe_gas_fvf = gas_formation_volume_factor_grid[i, j, k]
                if safe_oil_fvf < 1e-30:
                    safe_oil_fvf = 1e-30
                if safe_water_fvf < 1e-30:
                    safe_water_fvf = 1e-30
                if safe_gas_fvf < 1e-30:
                    safe_gas_fvf = 1e-30

                current_alpha_solution_gor = (
                    solution_gas_to_oil_ratio_grid[i, j, k]
                    * safe_gas_fvf
                    / (safe_oil_fvf * bbl_to_ft3)
                )
                current_alpha_gas_solubility_in_water = (
                    gas_solubility_in_water_grid[i, j, k]
                    * safe_gas_fvf
                    / (safe_water_fvf * bbl_to_ft3)
                )

                # Current Newton-iterate saturations
                current_water_saturation = water_saturation_grid[i, j, k]
                current_gas_saturation = gas_saturation_grid[i, j, k]
                current_oil_saturation = 1.0 - current_water_saturation - current_gas_saturation

                # Water mass accumulation terms
                water_accumulation = accumulation_coefficient * (
                    (current_water_density * current_water_saturation)
                    - (old_water_mass_grid[i, j, k] / cell_pore_volume)
                )
                # Water mass accumulation terms
                current_gas_total_mass_per_unit_volume = (
                    current_gas_density * current_gas_saturation
                    + current_gas_density * current_alpha_solution_gor * current_oil_saturation
                    + current_gas_density
                    * current_alpha_gas_solubility_in_water
                    * current_water_saturation
                )
                old_total_gas_mass_per_unit_volume = (
                    old_free_gas_mass_grid[i, j, k]
                    + old_dissolved_gas_mass_in_oil_grid[i, j, k]
                    + old_dissolved_gas_mass_in_water_grid[i, j, k]
                ) / cell_pore_volume
                gas_accumulation = accumulation_coefficient * (
                    current_gas_total_mass_per_unit_volume - old_total_gas_mass_per_unit_volume
                )

                # Face flux contributions
                net_water_mass_flux = 0.0
                net_gas_mass_flux = 0.0

                cell_water_mobility = water_relative_mobility_grid[i, j, k]
                cell_oil_mobility = oil_relative_mobility_grid[i, j, k]
                cell_gas_mobility = gas_relative_mobility_grid[i, j, k]
                cell_total_mobility = cell_water_mobility + cell_oil_mobility + cell_gas_mobility
                cell_pressure = pressure_grid[i, j, k]

                # Alpha factors for boundary cells (use interior cell values)
                cell_alpha_solution_gor = current_alpha_solution_gor
                cell_alpha_gas_solubility_in_water = current_alpha_gas_solubility_in_water

                # EAST (i+1, j, k)
                east_i = i + 1
                if east_i < cell_count_x:
                    (
                        water_flux,
                        oil_flux,
                        gas_flux,
                        upwind_water_density,
                        _,
                        upwind_gas_density,
                    ) = compute_face_fluxes(
                        cell_indices=(i, j, k),
                        neighbour_indices=(east_i, j, k),
                        pressure_grid=pressure_grid,
                        face_transmissibility=face_transmissibilities_x[i + 1, j + 1, k + 1],
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
                    if oil_flux > 0.0:
                        face_alpha_solution_gor = (
                            solution_gas_to_oil_ratio_grid[east_i, j, k]
                            * gas_formation_volume_factor_grid[east_i, j, k]
                            / (
                                max(
                                    oil_formation_volume_factor_grid[east_i, j, k],
                                    1e-30,
                                )
                                * bbl_to_ft3
                            )
                        )
                    else:
                        face_alpha_solution_gor = cell_alpha_solution_gor

                    if water_flux > 0.0:
                        face_alpha_gas_solubility_in_water = (
                            gas_solubility_in_water_grid[east_i, j, k]
                            * gas_formation_volume_factor_grid[east_i, j, k]
                            / (
                                max(
                                    water_formation_volume_factor_grid[east_i, j, k],
                                    1e-30,
                                )
                                * bbl_to_ft3
                            )
                        )
                    else:
                        face_alpha_gas_solubility_in_water = cell_alpha_gas_solubility_in_water

                    net_water_mass_flux += upwind_water_density * water_flux
                    net_gas_mass_flux += (
                        upwind_gas_density * gas_flux
                        + upwind_gas_density * face_alpha_solution_gor * oil_flux
                        + upwind_gas_density * face_alpha_gas_solubility_in_water * water_flux
                    )
                else:
                    pei, pej, pek = east_i + 1, j + 1, k + 1
                    pressure_boundary = pressure_boundaries[pei, pej, pek]
                    if not np.isnan(pressure_boundary):
                        pressure_difference = pressure_boundary - cell_pressure
                        T = (
                            face_transmissibilities_x[pei, pej, pek]
                            * md_per_cp_to_ft2_per_psi_per_day
                        )
                        water_flux = cell_water_mobility * T * pressure_difference
                        oil_flux = cell_oil_mobility * T * pressure_difference
                        gas_flux = cell_gas_mobility * T * pressure_difference
                        net_water_mass_flux += current_water_density * water_flux
                        net_gas_mass_flux += (
                            current_gas_density * gas_flux
                            + current_gas_density * cell_alpha_solution_gor * oil_flux
                            + current_gas_density * cell_alpha_gas_solubility_in_water * water_flux
                        )
                    else:
                        flux_boundary = flux_boundaries[pei, pej, pek]
                        if flux_boundary != 0 and cell_total_mobility > 0.0:
                            water_fraction = cell_water_mobility / cell_total_mobility
                            oil_fraction = cell_oil_mobility / cell_total_mobility
                            gas_fraction = cell_gas_mobility / cell_total_mobility
                            net_water_mass_flux += (
                                current_water_density * flux_boundary * water_fraction
                            )
                            net_gas_mass_flux += (
                                current_gas_density * flux_boundary * gas_fraction
                                + current_gas_density
                                * cell_alpha_solution_gor
                                * flux_boundary
                                * oil_fraction
                                + current_gas_density
                                * cell_alpha_gas_solubility_in_water
                                * flux_boundary
                                * water_fraction
                            )

                # WEST (i-1, j, k)
                west_i = i - 1
                pwi, pwj, pwk = west_i + 1, j + 1, k + 1
                if west_i >= 0:
                    (
                        water_flux,
                        oil_flux,
                        gas_flux,
                        upwind_water_density,
                        _,
                        upwind_gas_density,
                    ) = compute_face_fluxes(
                        cell_indices=(i, j, k),
                        neighbour_indices=(west_i, j, k),
                        pressure_grid=pressure_grid,
                        face_transmissibility=face_transmissibilities_x[pwi, pwj, pwk],
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
                    if oil_flux > 0.0:
                        face_alpha_solution_gor = (
                            solution_gas_to_oil_ratio_grid[west_i, j, k]
                            * gas_formation_volume_factor_grid[west_i, j, k]
                            / (
                                max(
                                    oil_formation_volume_factor_grid[west_i, j, k],
                                    1e-30,
                                )
                                * bbl_to_ft3
                            )
                        )
                    else:
                        face_alpha_solution_gor = cell_alpha_solution_gor

                    if water_flux > 0.0:
                        face_alpha_gas_solubility_in_water = (
                            gas_solubility_in_water_grid[west_i, j, k]
                            * gas_formation_volume_factor_grid[west_i, j, k]
                            / (
                                max(
                                    water_formation_volume_factor_grid[west_i, j, k],
                                    1e-30,
                                )
                                * bbl_to_ft3
                            )
                        )
                    else:
                        face_alpha_gas_solubility_in_water = cell_alpha_gas_solubility_in_water

                    net_water_mass_flux += upwind_water_density * water_flux
                    net_gas_mass_flux += (
                        upwind_gas_density * gas_flux
                        + upwind_gas_density * face_alpha_solution_gor * oil_flux
                        + upwind_gas_density * face_alpha_gas_solubility_in_water * water_flux
                    )
                else:
                    pressure_boundary = pressure_boundaries[pwi, pwj, pwk]
                    if not np.isnan(pressure_boundary):
                        pressure_difference = pressure_boundary - cell_pressure
                        T = (
                            face_transmissibilities_x[pwi, pwj, pwk]
                            * md_per_cp_to_ft2_per_psi_per_day
                        )
                        water_flux = cell_water_mobility * T * pressure_difference
                        oil_flux = cell_oil_mobility * T * pressure_difference
                        gas_flux = cell_gas_mobility * T * pressure_difference
                        net_water_mass_flux += current_water_density * water_flux
                        net_gas_mass_flux += (
                            current_gas_density * gas_flux
                            + current_gas_density * cell_alpha_solution_gor * oil_flux
                            + current_gas_density * cell_alpha_gas_solubility_in_water * water_flux
                        )
                    else:
                        flux_boundary = flux_boundaries[pwi, pwj, pwk]
                        if flux_boundary != 0 and cell_total_mobility > 0.0:
                            water_fraction = cell_water_mobility / cell_total_mobility
                            oil_fraction = cell_oil_mobility / cell_total_mobility
                            gas_fraction = cell_gas_mobility / cell_total_mobility
                            net_water_mass_flux += (
                                current_water_density * flux_boundary * water_fraction
                            )
                            net_gas_mass_flux += (
                                current_gas_density * flux_boundary * gas_fraction
                                + current_gas_density
                                * cell_alpha_solution_gor
                                * flux_boundary
                                * oil_fraction
                                + current_gas_density
                                * cell_alpha_gas_solubility_in_water
                                * flux_boundary
                                * water_fraction
                            )

                # SOUTH (i, j+1, k)
                south_j = j + 1
                if south_j < cell_count_y:
                    (
                        water_flux,
                        oil_flux,
                        gas_flux,
                        upwind_water_density,
                        _,
                        upwind_gas_density,
                    ) = compute_face_fluxes(
                        cell_indices=(i, j, k),
                        neighbour_indices=(i, south_j, k),
                        pressure_grid=pressure_grid,
                        face_transmissibility=face_transmissibilities_y[i + 1, j + 1, k + 1],
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
                    if oil_flux > 0.0:
                        face_alpha_solution_gor = (
                            solution_gas_to_oil_ratio_grid[i, south_j, k]
                            * gas_formation_volume_factor_grid[i, south_j, k]
                            / (
                                max(
                                    oil_formation_volume_factor_grid[i, south_j, k],
                                    1e-30,
                                )
                                * bbl_to_ft3
                            )
                        )
                    else:
                        face_alpha_solution_gor = cell_alpha_solution_gor

                    if water_flux > 0.0:
                        face_alpha_gas_solubility_in_water = (
                            gas_solubility_in_water_grid[i, south_j, k]
                            * gas_formation_volume_factor_grid[i, south_j, k]
                            / (
                                max(
                                    water_formation_volume_factor_grid[i, south_j, k],
                                    1e-30,
                                )
                                * bbl_to_ft3
                            )
                        )
                    else:
                        face_alpha_gas_solubility_in_water = cell_alpha_gas_solubility_in_water

                    net_water_mass_flux += upwind_water_density * water_flux
                    net_gas_mass_flux += (
                        upwind_gas_density * gas_flux
                        + upwind_gas_density * face_alpha_solution_gor * oil_flux
                        + upwind_gas_density * face_alpha_gas_solubility_in_water * water_flux
                    )
                else:
                    psi, psj, psk = i + 1, south_j + 1, k + 1
                    pressure_boundary = pressure_boundaries[psi, psj, psk]
                    if not np.isnan(pressure_boundary):
                        pressure_difference = pressure_boundary - cell_pressure
                        T = (
                            face_transmissibilities_y[psi, psj, psk]
                            * md_per_cp_to_ft2_per_psi_per_day
                        )
                        water_flux = cell_water_mobility * T * pressure_difference
                        oil_flux = cell_oil_mobility * T * pressure_difference
                        gas_flux = cell_gas_mobility * T * pressure_difference
                        net_water_mass_flux += current_water_density * water_flux
                        net_gas_mass_flux += (
                            current_gas_density * gas_flux
                            + current_gas_density * cell_alpha_solution_gor * oil_flux
                            + current_gas_density * cell_alpha_gas_solubility_in_water * water_flux
                        )
                    else:
                        flux_boundary = flux_boundaries[psi, psj, psk]
                        if flux_boundary != 0 and cell_total_mobility > 0.0:
                            water_fraction = cell_water_mobility / cell_total_mobility
                            oil_fraction = cell_oil_mobility / cell_total_mobility
                            gas_fraction = cell_gas_mobility / cell_total_mobility
                            net_water_mass_flux += (
                                current_water_density * flux_boundary * water_fraction
                            )
                            net_gas_mass_flux += (
                                current_gas_density * flux_boundary * gas_fraction
                                + current_gas_density
                                * cell_alpha_solution_gor
                                * flux_boundary
                                * oil_fraction
                                + current_gas_density
                                * cell_alpha_gas_solubility_in_water
                                * flux_boundary
                                * water_fraction
                            )

                # NORTH (i, j-1, k)
                north_j = j - 1
                pni, pnj, pnk = i + 1, north_j + 1, k + 1
                if north_j >= 0:
                    (
                        water_flux,
                        oil_flux,
                        gas_flux,
                        upwind_water_density,
                        _,
                        upwind_gas_density,
                    ) = compute_face_fluxes(
                        cell_indices=(i, j, k),
                        neighbour_indices=(i, north_j, k),
                        pressure_grid=pressure_grid,
                        face_transmissibility=face_transmissibilities_y[pni, pnj, pnk],
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
                    if oil_flux > 0.0:
                        face_alpha_solution_gor = (
                            solution_gas_to_oil_ratio_grid[i, north_j, k]
                            * gas_formation_volume_factor_grid[i, north_j, k]
                            / (
                                max(
                                    oil_formation_volume_factor_grid[i, north_j, k],
                                    1e-30,
                                )
                                * bbl_to_ft3
                            )
                        )
                    else:
                        face_alpha_solution_gor = cell_alpha_solution_gor

                    if water_flux > 0.0:
                        face_alpha_gas_solubility_in_water = (
                            gas_solubility_in_water_grid[i, north_j, k]
                            * gas_formation_volume_factor_grid[i, north_j, k]
                            / (
                                max(
                                    water_formation_volume_factor_grid[i, north_j, k],
                                    1e-30,
                                )
                                * bbl_to_ft3
                            )
                        )
                    else:
                        face_alpha_gas_solubility_in_water = cell_alpha_gas_solubility_in_water

                    net_water_mass_flux += upwind_water_density * water_flux
                    net_gas_mass_flux += (
                        upwind_gas_density * gas_flux
                        + upwind_gas_density * face_alpha_solution_gor * oil_flux
                        + upwind_gas_density * face_alpha_gas_solubility_in_water * water_flux
                    )
                else:
                    pressure_boundary = pressure_boundaries[pni, pnj, pnk]
                    if not np.isnan(pressure_boundary):
                        pressure_difference = pressure_boundary - cell_pressure
                        T = (
                            face_transmissibilities_y[pni, pnj, pnk]
                            * md_per_cp_to_ft2_per_psi_per_day
                        )
                        water_flux = cell_water_mobility * T * pressure_difference
                        oil_flux = cell_oil_mobility * T * pressure_difference
                        gas_flux = cell_gas_mobility * T * pressure_difference
                        net_water_mass_flux += current_water_density * water_flux
                        net_gas_mass_flux += (
                            current_gas_density * gas_flux
                            + current_gas_density * cell_alpha_solution_gor * oil_flux
                            + current_gas_density * cell_alpha_gas_solubility_in_water * water_flux
                        )
                    else:
                        flux_boundary = flux_boundaries[pni, pnj, pnk]
                        if flux_boundary != 0 and cell_total_mobility > 0.0:
                            water_fraction = cell_water_mobility / cell_total_mobility
                            oil_fraction = cell_oil_mobility / cell_total_mobility
                            gas_fraction = cell_gas_mobility / cell_total_mobility
                            net_water_mass_flux += (
                                current_water_density * flux_boundary * water_fraction
                            )
                            net_gas_mass_flux += (
                                current_gas_density * flux_boundary * gas_fraction
                                + current_gas_density
                                * cell_alpha_solution_gor
                                * flux_boundary
                                * oil_fraction
                                + current_gas_density
                                * cell_alpha_gas_solubility_in_water
                                * flux_boundary
                                * water_fraction
                            )

                # BOTTOM (i, j, k+1)
                bottom_k = k + 1
                if bottom_k < cell_count_z:
                    (
                        water_flux,
                        oil_flux,
                        gas_flux,
                        upwind_water_density,
                        _,
                        upwind_gas_density,
                    ) = compute_face_fluxes(
                        cell_indices=(i, j, k),
                        neighbour_indices=(i, j, bottom_k),
                        pressure_grid=pressure_grid,
                        face_transmissibility=face_transmissibilities_z[i + 1, j + 1, k + 1],
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
                    if oil_flux > 0.0:
                        face_alpha_solution_gor = (
                            solution_gas_to_oil_ratio_grid[i, j, bottom_k]
                            * gas_formation_volume_factor_grid[i, j, bottom_k]
                            / (
                                max(
                                    oil_formation_volume_factor_grid[i, j, bottom_k],
                                    1e-30,
                                )
                                * bbl_to_ft3
                            )
                        )
                    else:
                        face_alpha_solution_gor = cell_alpha_solution_gor

                    if water_flux > 0.0:
                        face_alpha_gas_solubility_in_water = (
                            gas_solubility_in_water_grid[i, j, bottom_k]
                            * gas_formation_volume_factor_grid[i, j, bottom_k]
                            / (
                                max(
                                    water_formation_volume_factor_grid[i, j, bottom_k],
                                    1e-30,
                                )
                                * bbl_to_ft3
                            )
                        )
                    else:
                        face_alpha_gas_solubility_in_water = cell_alpha_gas_solubility_in_water

                    net_water_mass_flux += upwind_water_density * water_flux
                    net_gas_mass_flux += (
                        upwind_gas_density * gas_flux
                        + upwind_gas_density * face_alpha_solution_gor * oil_flux
                        + upwind_gas_density * face_alpha_gas_solubility_in_water * water_flux
                    )
                else:
                    pbi, pbj, pbk = i + 1, j + 1, bottom_k + 1
                    pressure_boundary = pressure_boundaries[pbi, pbj, pbk]
                    if not np.isnan(pressure_boundary):
                        pressure_difference = pressure_boundary - cell_pressure
                        T = (
                            face_transmissibilities_z[pbi, pbj, pbk]
                            * md_per_cp_to_ft2_per_psi_per_day
                        )
                        water_flux = cell_water_mobility * T * pressure_difference
                        oil_flux = cell_oil_mobility * T * pressure_difference
                        gas_flux = cell_gas_mobility * T * pressure_difference
                        net_water_mass_flux += current_water_density * water_flux
                        net_gas_mass_flux += (
                            current_gas_density * gas_flux
                            + current_gas_density * cell_alpha_solution_gor * oil_flux
                            + current_gas_density * cell_alpha_gas_solubility_in_water * water_flux
                        )
                    else:
                        flux_boundary = flux_boundaries[pbi, pbj, pbk]
                        if flux_boundary != 0 and cell_total_mobility > 0.0:
                            water_fraction = cell_water_mobility / cell_total_mobility
                            oil_fraction = cell_oil_mobility / cell_total_mobility
                            gas_fraction = cell_gas_mobility / cell_total_mobility
                            net_water_mass_flux += (
                                current_water_density * flux_boundary * water_fraction
                            )
                            net_gas_mass_flux += (
                                current_gas_density * flux_boundary * gas_fraction
                                + current_gas_density
                                * cell_alpha_solution_gor
                                * flux_boundary
                                * oil_fraction
                                + current_gas_density
                                * cell_alpha_gas_solubility_in_water
                                * flux_boundary
                                * water_fraction
                            )

                # TOP (i, j, k-1)
                top_k = k - 1
                pti, ptj, ptk = i + 1, j + 1, top_k + 1
                if top_k >= 0:
                    (
                        water_flux,
                        oil_flux,
                        gas_flux,
                        upwind_water_density,
                        _,
                        upwind_gas_density,
                    ) = compute_face_fluxes(
                        cell_indices=(i, j, k),
                        neighbour_indices=(i, j, top_k),
                        pressure_grid=pressure_grid,
                        face_transmissibility=face_transmissibilities_z[pti, ptj, ptk],
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
                    if oil_flux > 0.0:
                        face_alpha_solution_gor = (
                            solution_gas_to_oil_ratio_grid[i, j, top_k]
                            * gas_formation_volume_factor_grid[i, j, top_k]
                            / (
                                max(oil_formation_volume_factor_grid[i, j, top_k], 1e-30)
                                * bbl_to_ft3
                            )
                        )
                    else:
                        face_alpha_solution_gor = cell_alpha_solution_gor

                    if water_flux > 0.0:
                        face_alpha_gas_solubility_in_water = (
                            gas_solubility_in_water_grid[i, j, top_k]
                            * gas_formation_volume_factor_grid[i, j, top_k]
                            / (
                                max(
                                    water_formation_volume_factor_grid[i, j, top_k],
                                    1e-30,
                                )
                                * bbl_to_ft3
                            )
                        )
                    else:
                        face_alpha_gas_solubility_in_water = cell_alpha_gas_solubility_in_water

                    net_water_mass_flux += upwind_water_density * water_flux
                    net_gas_mass_flux += (
                        upwind_gas_density * gas_flux
                        + upwind_gas_density * face_alpha_solution_gor * oil_flux
                        + upwind_gas_density * face_alpha_gas_solubility_in_water * water_flux
                    )
                else:
                    pressure_boundary = pressure_boundaries[pti, ptj, ptk]
                    if not np.isnan(pressure_boundary):
                        pressure_difference = pressure_boundary - cell_pressure
                        T = (
                            face_transmissibilities_z[pti, ptj, ptk]
                            * md_per_cp_to_ft2_per_psi_per_day
                        )
                        water_flux = cell_water_mobility * T * pressure_difference
                        oil_flux = cell_oil_mobility * T * pressure_difference
                        gas_flux = cell_gas_mobility * T * pressure_difference
                        net_water_mass_flux += current_water_density * water_flux
                        net_gas_mass_flux += (
                            current_gas_density * gas_flux
                            + current_gas_density * cell_alpha_solution_gor * oil_flux
                            + current_gas_density * cell_alpha_gas_solubility_in_water * water_flux
                        )
                    else:
                        flux_boundary = flux_boundaries[pti, ptj, ptk]
                        if flux_boundary != 0 and cell_total_mobility > 0.0:
                            water_fraction = cell_water_mobility / cell_total_mobility
                            oil_fraction = cell_oil_mobility / cell_total_mobility
                            gas_fraction = cell_gas_mobility / cell_total_mobility
                            net_water_mass_flux += (
                                current_water_density * flux_boundary * water_fraction
                            )
                            net_gas_mass_flux += (
                                current_gas_density * flux_boundary * gas_fraction
                                + current_gas_density
                                * cell_alpha_solution_gor
                                * flux_boundary
                                * oil_fraction
                                + current_gas_density
                                * cell_alpha_gas_solubility_in_water
                                * flux_boundary
                                * water_fraction
                            )

                # Well rates
                net_oil_well_rate = net_oil_well_rate_grid[i, j, k]
                net_water_well_rate = net_water_well_rate_grid[i, j, k]
                net_gas_well_rate = net_gas_well_rate_grid[i, j, k]

                net_water_mass_rate = net_water_well_rate * current_water_density
                free_gas_mass_rate = net_gas_well_rate * current_gas_density
                # Dissolved gas only applies to produced fluids (negative rates)
                # Injected fluids are at surface conditions, so no dissolved gas
                dissolved_gas_mass_rate = current_gas_density * cell_alpha_solution_gor * min(
                    net_oil_well_rate, 0.0
                ) + current_gas_density * cell_alpha_gas_solubility_in_water * min(
                    net_water_well_rate, 0.0
                )
                net_gas_mass_rate = free_gas_mass_rate + dissolved_gas_mass_rate

                water_residual[cell_idx] = (
                    water_accumulation - net_water_mass_flux - net_water_mass_rate
                )
                gas_residual[cell_idx] = gas_accumulation - net_gas_mass_flux - net_gas_mass_rate

    return water_residual, gas_residual


def compute_rock_fluid_properties(
    water_saturation_grid: ThreeDimensionalGrid,
    oil_saturation_grid: ThreeDimensionalGrid,
    gas_saturation_grid: ThreeDimensionalGrid,
    rock_properties: RockProperties[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
    hysteresis_state: Hysteresis[ThreeDimensions] | None,
    config: Config,
) -> tuple[
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
    :return: 3-tuple of `(relative_mobility_grids, capillary_pressure_grids, mobility_grids)`.
    """
    _, relative_mobility_grids, capillary_pressure_grids = build_rock_fluid_properties_grids(
        water_saturation_grid=water_saturation_grid,  # type: ignore[arg-type]
        oil_saturation_grid=oil_saturation_grid,  # type: ignore[arg-type]
        gas_saturation_grid=gas_saturation_grid,  # type: ignore[arg-type]
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
        hysteresis_state=hysteresis_state,
        disable_capillary_effects=config.disable_capillary_effects,
        capillary_strength_factor=config.capillary_strength_factor,
        phase_appearance_tolerance=config.phase_appearance_tolerance,
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


def _assemble_residuals(
    water_saturation_grid: ThreeDimensionalGrid,
    gas_saturation_grid: ThreeDimensionalGrid,
    pressure_grid: ThreeDimensionalGrid,
    old_water_mass_grid: ThreeDimensionalGrid,
    old_free_gas_mass_grid: ThreeDimensionalGrid,
    old_dissolved_gas_mass_in_oil_grid: ThreeDimensionalGrid,
    old_dissolved_gas_mass_in_water_grid: ThreeDimensionalGrid,
    face_transmissibilities: FaceTransmissibilities,
    capillary_pressure_grids: CapillaryPressureGrids[ThreeDimensions],
    relative_mobility_grids: RelativeMobilityGrids[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
    pore_volume_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    time_step_in_days: float,
    gravitational_constant: float,
    net_water_well_rate_grid: ThreeDimensionalGrid,
    net_oil_well_rate_grid: ThreeDimensionalGrid,
    net_gas_well_rate_grid: ThreeDimensionalGrid,
    pressure_boundaries: ThreeDimensionalGrid,
    flux_boundaries: ThreeDimensionalGrid,
    md_per_cp_to_ft2_per_psi_per_day: float,
    bbl_to_ft3: float,
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Assemble the mass-based residual from pre-computed saturation-dependent properties.

    :param water_saturation_grid: Current (Newton iterate) water saturation grid.
    :param gas_saturation_grid: Current (Newton iterate) gas saturation grid.
    :param pressure_grid: Oil pressure grid (psi), fixed during Newton loop.
    :param face_transmissibilities: Precomputed geometric face transmissibilities.
    :param capillary_pressure_grids: `(Pcow, Pcgo)` at current iterate.
    :param relative_mobility_grids: `(lam_w, lam_o, lam_g)` at current iterate.
    :param fluid_properties: Fluid properties (density, Rs, FVF grids at new pressure).
    :param cell_count_x: Number of cells in x-direction.
    :param cell_count_y: Number of cells in y-direction.
    :param cell_count_z: Number of cells in z-direction.
    :param elevation_grid: Cell elevation grid (ft).
    :param time_step_in_days: Time step size (days).
    :param gravitational_constant: Gravitational constant conversion factor (lbf/lbm).
    :param pressure_boundaries: Padded pressure boundary grid, shape `(nx+2, ny+2, nz+2)`.
    :param flux_boundaries: Padded flux boundary grid, shape `(nx+2, ny+2, nz+2)`.
    :param md_per_cp_to_ft2_per_psi_per_day: Unit conversion factor.
    :return: `(water_residual, gas_residual)` as 1-D arrays of length `nx*ny*nz`.
    """
    oil_water_capillary_pressure_grid, gas_oil_capillary_pressure_grid = capillary_pressure_grids
    (
        water_relative_mobility_grid,
        oil_relative_mobility_grid,
        gas_relative_mobility_grid,
    ) = relative_mobility_grids
    return _assemble_saturation_residuals(
        pressure_grid=pressure_grid,
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
        pore_volume_grid=pore_volume_grid,
        water_relative_mobility_grid=water_relative_mobility_grid,
        oil_relative_mobility_grid=oil_relative_mobility_grid,
        gas_relative_mobility_grid=gas_relative_mobility_grid,
        face_transmissibilities_x=face_transmissibilities.x,
        face_transmissibilities_y=face_transmissibilities.y,
        face_transmissibilities_z=face_transmissibilities.z,
        oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
        gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
        water_density_grid=fluid_properties.water_density_grid,
        oil_density_grid=fluid_properties.oil_effective_density_grid,
        gas_density_grid=fluid_properties.gas_density_grid,
        elevation_grid=elevation_grid,
        gravitational_constant=gravitational_constant,
        water_saturation_grid=water_saturation_grid,
        gas_saturation_grid=gas_saturation_grid,
        solution_gas_to_oil_ratio_grid=fluid_properties.solution_gas_to_oil_ratio_grid,
        gas_solubility_in_water_grid=fluid_properties.gas_solubility_in_water_grid,
        gas_formation_volume_factor_grid=fluid_properties.gas_formation_volume_factor_grid,
        oil_formation_volume_factor_grid=fluid_properties.oil_formation_volume_factor_grid,
        water_formation_volume_factor_grid=fluid_properties.water_formation_volume_factor_grid,
        old_water_mass_grid=old_water_mass_grid,
        old_free_gas_mass_grid=old_free_gas_mass_grid,
        old_dissolved_gas_mass_in_oil_grid=old_dissolved_gas_mass_in_oil_grid,
        old_dissolved_gas_mass_in_water_grid=old_dissolved_gas_mass_in_water_grid,
        time_step_in_days=time_step_in_days,
        net_water_well_rate_grid=net_water_well_rate_grid,
        net_oil_well_rate_grid=net_oil_well_rate_grid,
        net_gas_well_rate_grid=net_gas_well_rate_grid,
        pressure_boundaries=pressure_boundaries,
        flux_boundaries=flux_boundaries,
        md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
        bbl_to_ft3=bbl_to_ft3,
    )


def assemble_residuals(
    water_saturation_grid: ThreeDimensionalGrid,
    oil_saturation_grid: ThreeDimensionalGrid,
    gas_saturation_grid: ThreeDimensionalGrid,
    pressure_grid: ThreeDimensionalGrid,
    old_water_mass_grid: ThreeDimensionalGrid,
    old_free_gas_mass_grid: ThreeDimensionalGrid,
    old_dissolved_gas_mass_in_oil_grid: ThreeDimensionalGrid,
    old_dissolved_gas_mass_in_water_grid: ThreeDimensionalGrid,
    rock_properties: RockProperties[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
    hysteresis_state: Hysteresis[ThreeDimensions] | None,
    face_transmissibilities: FaceTransmissibilities,
    config: Config,
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
    pore_volume_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    time_step_in_days: float,
    gravitational_constant: float,
    pressure_boundaries: ThreeDimensionalGrid,
    flux_boundaries: ThreeDimensionalGrid,
    net_water_well_rate_grid: ThreeDimensionalGrid,
    net_oil_well_rate_grid: ThreeDimensionalGrid,
    net_gas_well_rate_grid: ThreeDimensionalGrid,
    md_per_cp_to_ft2_per_psi_per_day: float,
    bbl_to_ft3: float,
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Assemble the mass-based residuals, rebuilding saturation-dependent properties first.

    :param water_saturation_grid: Current (Newton iterate) water saturation grid.
    :param oil_saturation_grid: Current (Newton iterate) oil saturation grid.
    :param gas_saturation_grid: Current (Newton iterate) gas saturation grid.
    :param pressure_grid: Oil pressure grid (psi), fixed during Newton loop.
    :param rock_properties: Rock properties.
    :param fluid_properties: Fluid properties (at new pressure level).
    :param face_transmissibilities: Precomputed geometric face transmissibilities.
    :param config: Simulation configuration.
    :param cell_count_x: Number of cells in x-direction.
    :param cell_count_y: Number of cells in y-direction.
    :param cell_count_z: Number of cells in z-direction.
    :param elevation_grid: Cell elevation grid (ft).
    :param time_step_in_days: Time step size (days).
    :param gravitational_constant: Gravitational constant conversion factor (lbf/lbm).
    :param wells_indices: Cache of well indices.
    :param injection_mass_rates: Injection rates.
    :param production_mass_rates: Production rates.
    :param pressure_boundaries: Padded pressure boundary grid, shape `(nx+2, ny+2, nz+2)`.
    :param flux_boundaries: Padded flux boundary grid, shape `(nx+2, ny+2, nz+2)`.
    :param md_per_cp_to_ft2_per_psi_per_day: Unit conversion factor.
    :return: `(water_residual, gas_residual)` as 1-D arrays of length `nx*ny*nz`.
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
        hysteresis_state=hysteresis_state,
    )
    return _assemble_residuals(
        water_saturation_grid=water_saturation_grid,
        gas_saturation_grid=gas_saturation_grid,
        old_water_mass_grid=old_water_mass_grid,
        old_free_gas_mass_grid=old_free_gas_mass_grid,
        old_dissolved_gas_mass_in_oil_grid=old_dissolved_gas_mass_in_oil_grid,
        old_dissolved_gas_mass_in_water_grid=old_dissolved_gas_mass_in_water_grid,
        pressure_grid=pressure_grid,
        face_transmissibilities=face_transmissibilities,
        capillary_pressure_grids=capillary_pressure_grids,
        relative_mobility_grids=relative_mobility_grids,
        fluid_properties=fluid_properties,
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
        pore_volume_grid=pore_volume_grid,
        elevation_grid=elevation_grid,
        time_step_in_days=time_step_in_days,
        gravitational_constant=gravitational_constant,
        pressure_boundaries=pressure_boundaries,
        flux_boundaries=flux_boundaries,
        net_water_well_rate_grid=net_water_well_rate_grid,
        net_oil_well_rate_grid=net_oil_well_rate_grid,
        net_gas_well_rate_grid=net_gas_well_rate_grid,
        md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
        bbl_to_ft3=bbl_to_ft3,
    )


def assemble_numerical_jacobian(
    saturation_vector: npt.NDArray,
    base_residual: npt.NDArray,
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
    pore_volume_grid: ThreeDimensionalGrid,
    water_saturation_grid: ThreeDimensionalGrid,
    oil_saturation_grid: ThreeDimensionalGrid,
    gas_saturation_grid: ThreeDimensionalGrid,
    old_water_mass_grid: ThreeDimensionalGrid,
    old_free_gas_mass_grid: ThreeDimensionalGrid,
    old_dissolved_gas_mass_in_oil_grid: ThreeDimensionalGrid,
    old_dissolved_gas_mass_in_water_grid: ThreeDimensionalGrid,
    pressure_grid: ThreeDimensionalGrid,
    face_transmissibilities: FaceTransmissibilities,
    rock_properties: RockProperties[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
    hysteresis_state: Hysteresis[ThreeDimensions] | None,
    config: Config,
    elevation_grid: ThreeDimensionalGrid,
    time_step_in_days: float,
    gravitational_constant: float,
    pressure_boundaries: ThreeDimensionalGrid,
    flux_boundaries: ThreeDimensionalGrid,
    net_water_well_rate_grid: ThreeDimensionalGrid,
    net_oil_well_rate_grid: ThreeDimensionalGrid,
    net_gas_well_rate_grid: ThreeDimensionalGrid,
    md_per_cp_to_ft2_per_psi_per_day: float,
    bbl_to_ft3: float,
) -> coo_matrix:
    """
    Assemble the mass-based saturation Jacobian using column-wise forward finite differences.

    The numerical Jacobian is correct for the mass formulation automatically because
    it calls `assemble_residuals`, which evaluates the mass-based residual at each
    perturbed saturation. Densities and Rs/Rsw are frozen during Newton (function of
    pressure only), so the numerical Jacobian is exact for the sequential implicit
    scheme.

    :param saturation_vector: Current saturation vector `[Sw_0, Sg_0, ...]`.
    :param base_residual: Base residual `R(S)` at current iterate.
    :param total_cell_count: Total number of cells `nx * ny * nz`.
    :param cell_count_x: Number of cells in x-direction.
    :param cell_count_y: Number of cells in y-direction.
    :param cell_count_z: Number of cells in z-direction.
    :param water_saturation_grid: Current water saturation grid.
    :param oil_saturation_grid: Current oil saturation grid.
    :param gas_saturation_grid: Current gas saturation grid.
    :param pressure_grid: Fixed oil pressure grid (psi).
    :param face_transmissibilities: Precomputed geometric face transmissibilities.
    :param rock_properties: Rock properties.
    :param fluid_properties: Fluid properties (at new pressure level).
    :param config: Simulation configuration.
    :param elevation_grid: Cell elevation grid (ft).
    :param time_step_in_days: Time step size (days).
    :param gravitational_constant: Gravitational constant conversion factor (lbf/lbm).
    :param pressure_boundaries: Padded pressure boundary grid, shape `(nx+2, ny+2, nz+2)`.
    :param flux_boundaries: Padded flux boundary grid, shape `(nx+2, ny+2, nz+2)`.
    :param md_per_cp_to_ft2_per_psi_per_day: Unit conversion factor.
    :return: Sparse Jacobian of shape `(2N, 2N)` in COO format, where `N = nx*ny*nz`.
    """
    rows = []
    cols = []
    vals = []

    machine_eps = np.finfo(np.float64).eps  # type: ignore
    base_epsilon = float(np.sqrt(machine_eps))

    water_saturation_grid = water_saturation_grid.astype(np.float64, copy=True)
    oil_saturation_grid = oil_saturation_grid.astype(np.float64, copy=True)
    gas_saturation_grid = gas_saturation_grid.astype(np.float64, copy=True)

    residual_kwargs = dict(
        old_water_mass_grid=old_water_mass_grid,
        old_free_gas_mass_grid=old_free_gas_mass_grid,
        old_dissolved_gas_mass_in_oil_grid=old_dissolved_gas_mass_in_oil_grid,
        old_dissolved_gas_mass_in_water_grid=old_dissolved_gas_mass_in_water_grid,
        pressure_grid=pressure_grid,
        face_transmissibilities=face_transmissibilities,
        rock_properties=rock_properties,
        fluid_properties=fluid_properties,
        hysteresis_state=hysteresis_state,
        config=config,
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
        pore_volume_grid=pore_volume_grid,
        elevation_grid=elevation_grid,
        time_step_in_days=time_step_in_days,
        gravitational_constant=gravitational_constant,
        net_water_well_rate_grid=net_water_well_rate_grid,
        net_oil_well_rate_grid=net_oil_well_rate_grid,
        net_gas_well_rate_grid=net_gas_well_rate_grid,
        pressure_boundaries=pressure_boundaries,
        flux_boundaries=flux_boundaries,
        md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
        bbl_to_ft3=bbl_to_ft3,
    )

    total_cell_count = cell_count_x * cell_count_y * cell_count_z
    for cell_idx in range(total_cell_count):
        i, j, k = from_1D_index(
            idx=cell_idx,
            cell_count_x=cell_count_x,
            cell_count_y=cell_count_y,
            cell_count_z=cell_count_z,
        )

        # Affected cells: this cell + all face neighbours (for sparsity)
        affected_cell_indices = [cell_idx]
        for delta_i, delta_j, delta_k in (
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ):
            ni, nj, nk = i + delta_i, j + delta_j, k + delta_k
            if 0 <= ni < cell_count_x and 0 <= nj < cell_count_y and 0 <= nk < cell_count_z:
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

        cell_water_saturation = water_saturation_grid[i, j, k]
        cell_gas_saturation = gas_saturation_grid[i, j, k]
        cell_oil_saturation = 1.0 - cell_water_saturation - cell_gas_saturation

        for var_offset in range(2):
            column = 2 * cell_idx + var_offset
            s_j = saturation_vector[column]

            epsilon = base_epsilon * max(abs(s_j), 1.0)
            if epsilon > cell_oil_saturation:
                epsilon = -epsilon
                if s_j + epsilon < 0.0:
                    epsilon = (-s_j * 0.5) if s_j > 1e-15 else (base_epsilon * 0.01)

            original_water_saturation = water_saturation_grid[i, j, k]
            original_gas_saturation = gas_saturation_grid[i, j, k]
            original_oil_saturation = oil_saturation_grid[i, j, k]

            if var_offset == 0:
                perturbed_water_saturation = original_water_saturation + epsilon
                perturbed_gas_saturation = original_gas_saturation
            else:
                perturbed_water_saturation = original_water_saturation
                perturbed_gas_saturation = original_gas_saturation + epsilon
            perturbed_oil_saturation = max(
                0.0, 1.0 - perturbed_water_saturation - perturbed_gas_saturation
            )

            water_saturation_grid[i, j, k] = perturbed_water_saturation
            gas_saturation_grid[i, j, k] = perturbed_gas_saturation
            oil_saturation_grid[i, j, k] = perturbed_oil_saturation

            perturbed_water_residual, perturbed_gas_residual = assemble_residuals(
                water_saturation_grid=water_saturation_grid,
                oil_saturation_grid=oil_saturation_grid,
                gas_saturation_grid=gas_saturation_grid,
                **residual_kwargs,  # type: ignore
            )
            perturbed_residual = interleave_residuals(
                water_residual=perturbed_water_residual,
                gas_residual=perturbed_gas_residual,
            )

            water_saturation_grid[i, j, k] = original_water_saturation
            gas_saturation_grid[i, j, k] = original_gas_saturation
            oil_saturation_grid[i, j, k] = original_oil_saturation

            for affected_idx in affected_cell_indices:
                water_row = 2 * affected_idx
                dR_water = (perturbed_residual[water_row] - base_residual[water_row]) / epsilon
                if abs(dR_water) > 1e-30:
                    rows.append(water_row)
                    cols.append(column)
                    vals.append(dR_water)

                gas_row = 2 * affected_idx + 1
                dR_gas = (perturbed_residual[gas_row] - base_residual[gas_row]) / epsilon
                if abs(dR_gas) > 1e-30:
                    rows.append(gas_row)
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
    )


def compute_rock_fluid_derivatives(
    water_saturation_grid: ThreeDimensionalGrid,
    oil_saturation_grid: ThreeDimensionalGrid,
    gas_saturation_grid: ThreeDimensionalGrid,
    rock_properties: RockProperties[ThreeDimensions],
    hysteresis_state: Hysteresis[ThreeDimensions] | None,
    rock_fluid_tables: SatFuncTables,
    disable_capillary_effects: bool = False,
    capillary_strength_factor: float = 1.0,
) -> tuple[
    ThreeDimensionalGrid,  # dkrw_dsw
    ThreeDimensionalGrid,  # dkrw_dso
    ThreeDimensionalGrid,  # dkrw_dsg
    ThreeDimensionalGrid,  # dkro_dsw
    ThreeDimensionalGrid,  # dkro_dso
    ThreeDimensionalGrid,  # dkro_dsg
    ThreeDimensionalGrid,  # dkrg_dsw
    ThreeDimensionalGrid,  # dkrg_dso
    ThreeDimensionalGrid,  # dkrg_dsg
    ThreeDimensionalGrid,  # dpcow_dsw_eff
    ThreeDimensionalGrid,  # dpcow_dsg_eff
    ThreeDimensionalGrid,  # dpcgo_dsw_eff
    ThreeDimensionalGrid,  # dpcgo_dsg_eff
]:
    """
    Compute all relperm and capillary-pressure derivative grids for the analytical Jacobian.

    **Relperm derivatives** — all nine raw partials `d kr_alpha / d S_beta`
    (with `beta` in `{Sw, So, Sg}`) are returned unchanged.

    **Capillary-pressure derivatives projected onto (Sw, Sg) basis** — the
    constraint `So = 1 - Sw - Sg` gives `dso/dsw = dso/dsg = -1`, so:

        dpcow_dsw_eff = dpcow_dsw + dpcow_dso * (-1)
        dpcow_dsg_eff =              dpcow_dso * (-1)
        dpcgo_dsw_eff =              dpcgo_dso * (-1)
        dpcgo_dsg_eff = dpcgo_dsg + dpcgo_dso * (-1)

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

    water_saturation_grid = water_saturation_grid.astype(np.float64, copy=False)
    oil_saturation_grid = oil_saturation_grid.astype(np.float64, copy=False)
    gas_saturation_grid = gas_saturation_grid.astype(np.float64, copy=False)
    irreducible_water_saturation_grid = rock_properties.irreducible_water_saturation_grid.astype(
        np.float64, copy=False
    )
    residual_oil_saturation_water_grid = rock_properties.residual_oil_saturation_water_grid.astype(
        np.float64, copy=False
    )
    residual_oil_saturation_gas_grid = rock_properties.residual_oil_saturation_gas_grid.astype(
        np.float64, copy=False
    )
    residual_gas_saturation_grid = rock_properties.residual_gas_saturation_grid.astype(
        np.float64, copy=False
    )
    hysteresis_kwargs = {}
    if hysteresis_state is not None:
        hysteresis_kwargs = dict(
            max_water_saturation=hysteresis_state.max_water_saturation_grid.astype(
                np.float64, copy=False
            ),
            max_gas_saturation=hysteresis_state.max_gas_saturation_grid.astype(
                np.float64, copy=False
            ),
            water_imbibition_flag=hysteresis_state.water_imbibition_flag_grid.astype(
                np.float64, copy=False
            ),
            gas_imbibition_flag=hysteresis_state.gas_imbibition_flag_grid.astype(
                np.float64, copy=False
            ),
            water_reversal_saturation=hysteresis_state.water_reversal_saturation_grid.astype(
                np.float64, copy=False
            ),
            gas_reversal_saturation=hysteresis_state.gas_reversal_saturation_grid.astype(
                np.float64, copy=False
            ),
        )

    relperm_derivatives = relperm_table.derivatives(
        water_saturation=water_saturation_grid,
        oil_saturation=oil_saturation_grid,
        gas_saturation=gas_saturation_grid,
        irreducible_water_saturation=irreducible_water_saturation_grid,
        residual_oil_saturation_water=residual_oil_saturation_water_grid,
        residual_oil_saturation_gas=residual_oil_saturation_gas_grid,
        residual_gas_saturation=residual_gas_saturation_grid,
        **hysteresis_kwargs,
    )

    dkrw_dsw = relperm_derivatives["dkrw_dsw"]
    dkrw_dso = relperm_derivatives["dkrw_dso"]
    dkrw_dsg = relperm_derivatives["dkrw_dsg"]
    dkro_dsw = relperm_derivatives["dkro_dsw"]
    dkro_dso = relperm_derivatives["dkro_dso"]
    dkro_dsg = relperm_derivatives["dkro_dsg"]
    dkrg_dsw = relperm_derivatives["dkrg_dsw"]
    dkrg_dso = relperm_derivatives["dkrg_dso"]
    dkrg_dsg = relperm_derivatives["dkrg_dsg"]

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
            **hysteresis_kwargs,
        )
        dpcow_dsw = capillary_pressure_derivatives["dpcow_dsw"] * capillary_strength_factor
        dpcow_dso = capillary_pressure_derivatives["dpcow_dso"] * capillary_strength_factor
        dpcgo_dso = capillary_pressure_derivatives["dpcgo_dso"] * capillary_strength_factor
        dpcgo_dsg = capillary_pressure_derivatives["dpcgo_dsg"] * capillary_strength_factor
        dpcow_dsw_eff = dpcow_dsw - dpcow_dso
        dpcow_dsg_eff = -dpcow_dso
        dpcgo_dsw_eff = -dpcgo_dso
        dpcgo_dsg_eff = dpcgo_dsg - dpcgo_dso
    else:
        zeros = np.zeros_like(water_saturation_grid)
        dpcow_dsw_eff = zeros
        dpcow_dsg_eff = zeros.copy()
        dpcgo_dsw_eff = zeros.copy()
        dpcgo_dsg_eff = zeros.copy()

    return (  # type: ignore[return-value]
        dkrw_dsw,
        dkrw_dso,
        dkrw_dsg,
        dkro_dsw,
        dkro_dso,
        dkro_dsg,
        dkrg_dsw,
        dkrg_dso,
        dkrg_dsg,
        dpcow_dsw_eff,
        dpcow_dsg_eff,
        dpcgo_dsw_eff,
        dpcgo_dsg_eff,
    )  # ty:ignore[invalid-return-type]


@numba.njit(parallel=True, cache=True)
def assemble_flux_contributions(
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
    pore_volume_grid: ThreeDimensionalGrid,
    pressure_grid: ThreeDimensionalGrid,
    water_density_grid: ThreeDimensionalGrid,
    oil_density_grid: ThreeDimensionalGrid,
    gas_density_grid: ThreeDimensionalGrid,
    solution_gas_to_oil_ratio_grid: ThreeDimensionalGrid,
    gas_solubility_in_water_grid: ThreeDimensionalGrid,
    gas_formation_volume_factor_grid: ThreeDimensionalGrid,
    oil_formation_volume_factor_grid: ThreeDimensionalGrid,
    water_formation_volume_factor_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    gravitational_constant: float,
    face_transmissibilities_x: ThreeDimensionalGrid,
    face_transmissibilities_y: ThreeDimensionalGrid,
    face_transmissibilities_z: ThreeDimensionalGrid,
    water_relative_mobility_grid: ThreeDimensionalGrid,
    gas_relative_mobility_grid: ThreeDimensionalGrid,
    oil_water_capillary_pressure_grid: ThreeDimensionalGrid,
    gas_oil_capillary_pressure_grid: ThreeDimensionalGrid,
    dkrw_dsw_grid: ThreeDimensionalGrid,
    dkrw_dso_grid: ThreeDimensionalGrid,
    dkrw_dsg_grid: ThreeDimensionalGrid,
    dkro_dsw_grid: ThreeDimensionalGrid,
    dkro_dso_grid: ThreeDimensionalGrid,
    dkro_dsg_grid: ThreeDimensionalGrid,
    dkrg_dsw_grid: ThreeDimensionalGrid,
    dkrg_dso_grid: ThreeDimensionalGrid,
    dkrg_dsg_grid: ThreeDimensionalGrid,
    dpcow_dsw_eff_grid: ThreeDimensionalGrid,
    dpcow_dsg_eff_grid: ThreeDimensionalGrid,
    dpcgo_dsw_eff_grid: ThreeDimensionalGrid,
    dpcgo_dsg_eff_grid: ThreeDimensionalGrid,
    water_viscosity_grid: ThreeDimensionalGrid,
    oil_viscosity_grid: ThreeDimensionalGrid,
    gas_viscosity_grid: ThreeDimensionalGrid,
    time_step_in_days: float,
    md_per_cp_to_ft2_per_psi_per_day: float,
    bbl_to_ft3: float,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Assemble the inter-cell flux and accumulation parts of the mass-based saturation Jacobian.

    The accumulation terms in the mass formulation differ from the volumetric formulation.
    Since densities are frozen during Newton iterations (they are functions of pressure only
    in the sequential implicit scheme), the accumulation Jacobian terms are simply scaled by
    the frozen cell density:

        dR_w/dsw_i (accumulation) = cell_water_density * (phi*V_i/dt)
        dR_g/dsg_i (accumulation) = (cell_gas_density - cell_gas_density*alpha_Rs_i) * (phi*V_i/dt)
        dR_g/dsw_i (accumulation) = (cell_gas_density*alpha_Rsw_i - cell_gas_density*alpha_Rs_i) * (phi*V_i/dt)
            [from So = 1 - Sw - Sg substitution into M_g]

    Note: the gas accumulation captures all three saturation dependencies:
        M_g = gas_density*Sg + gas_density*alpha_Rs*(1-Sw-Sg) + gas_density*alpha_Rsw*Sw
        dM_g/dsw = -gas_density*alpha_Rs + gas_density*alpha_Rsw
        dM_g/dsg = gas_density - gas_density*alpha_Rs

    The flux Jacobian has a 2x2 block per (cell, neighbour) pair covering
    dRw/dsw, dRw/dsg, dRg/dsw, dRg/dsg. Every flux stream that appears in
    the residual contributes to all four entries:

      - Water residual R_w = -water_density * F_w  ->  dRw/dS terms
      - Gas residual R_g   = -gas_density * F_g  (free gas)
                           - gas_density * alpha_Rs * F_o  (dissolved gas in oil)
                           - gas_density * alpha_Rsw * F_w  (dissolved gas in water)
                           ->  dRg/dS terms from all three streams

    Alpha coefficients (alpha_Rs, alpha_Rsw) and densities are arithmetic-averaged
    across the face to eliminate O(1) jumps when the upwind cell switches at a front,
    making the Jacobian consistent with a smooth approximation of the hard upwind.

    Boundary faces have zero Jacobian w.r.t. saturation (prescribed flux).

    :param cell_count_x: Number of cells in x-direction.
    :param cell_count_y: Number of cells in y-direction.
    :param cell_count_z: Number of cells in z-direction.
    :param pressure_grid: Oil pressure grid (psi), shape `(nx, ny, nz)`.
    :param water_density_grid: Water density at new pressure (lb/ft³).
    :param oil_density_grid: Oil effective density at new pressure (lb/ft³).
    :param gas_density_grid: Gas density at new pressure (lb/ft³).
    :param solution_gas_to_oil_ratio_grid: Solution GOR grid (scf/STB).
    :param gas_solubility_in_water_grid: Gas solubility in water grid (scf/STB).
    :param gas_formation_volume_factor_grid: Gas FVF grid (res bbl/scf).
    :param oil_formation_volume_factor_grid: Oil FVF grid (res bbl/STB).
    :param water_formation_volume_factor_grid: Water FVF grid (res bbl/STB).
    :param elevation_grid: Cell elevation grid (ft).
    :param gravitational_constant: Gravitational constant conversion factor (lbf/lbm).
    :param face_transmissibilities_x: x-direction face transmissibilities (mD·ft).
    :param face_transmissibilities_y: y-direction face transmissibilities (mD·ft).
    :param face_transmissibilities_z: z-direction face transmissibilities (mD·ft).
    :param water_relative_mobility_grid: Water relative mobility (ft²/psi·day).
    :param gas_relative_mobility_grid: Gas relative mobility (ft²/psi·day).
    :param oil_water_capillary_pressure_grid: Oil-water capillary pressure (psi).
    :param gas_oil_capillary_pressure_grid: Gas-oil capillary pressure (psi).
    :param dkrw_dsw_grid: `∂krw/∂Sw` grid.
    :param dkrw_dso_grid: `∂krw/∂So` grid.
    :param dkrw_dsg_grid: `∂krw/∂Sg` grid.
    :param dkro_dsw_grid: `∂kro/∂Sw` grid.
    :param dkro_dso_grid: `∂kro/∂So` grid.
    :param dkro_dsg_grid: `∂kro/∂Sg` grid.
    :param dkrg_dsw_grid: `∂krg/∂Sw` grid.
    :param dkrg_dso_grid: `∂krg/∂So` grid.
    :param dkrg_dsg_grid: `∂krg/∂Sg` grid.
    :param dpcow_dsw_eff_grid: Effective `∂Pcow/∂Sw` (So eliminated) grid.
    :param dpcow_dsg_eff_grid: Effective `∂Pcow/∂Sg` (So eliminated) grid.
    :param dpcgo_dsw_eff_grid: Effective `∂Pcgo/∂Sw` (So eliminated) grid.
    :param dpcgo_dsg_eff_grid: Effective `∂Pcgo/∂Sg` (So eliminated) grid.
    :param water_viscosity_grid: Water viscosity (cP).
    :param oil_viscosity_grid: Oil viscosity (cP).
    :param gas_viscosity_grid: Gas viscosity (cP).
    :param time_step_in_days: Time step size (days).
    :param md_per_cp_to_ft2_per_psi_per_day: Unit conversion factor.
    :return: COO triplet `(rows, cols, vals)` for the inter-cell Jacobian entries.
    """
    cells_per_slice = cell_count_y * cell_count_z
    # 3 accumulation entries per cell (dRw/dsw, dRg/dsg, dRg/dsw coupling)
    # + up to 8 flux entries per face × 6 faces
    max_nnz_per_slice = cells_per_slice * (3 + 6 * 16)

    all_rows = np.empty((cell_count_x, max_nnz_per_slice), dtype=np.int32)
    all_cols = np.empty((cell_count_x, max_nnz_per_slice), dtype=np.int32)
    all_vals = np.empty((cell_count_x, max_nnz_per_slice), dtype=np.float64)
    slice_fill = np.zeros(cell_count_x, dtype=np.int64)

    for i in numba.prange(cell_count_x):  # type: ignore
        local_ptr = 0

        for j in range(cell_count_y):
            for k in range(cell_count_z):
                cell_idx = to_1D_index(
                    i=i,
                    j=j,
                    k=k,
                    cell_count_x=cell_count_x,
                    cell_count_y=cell_count_y,
                    cell_count_z=cell_count_z,
                )
                water_row = 2 * cell_idx
                gas_row = 2 * cell_idx + 1
                cell_water_column = 2 * cell_idx
                cell_gas_column = 2 * cell_idx + 1

                accumulation_coefficient = pore_volume_grid[i, j, k] / time_step_in_days
                cell_water_density = water_density_grid[i, j, k]
                cell_gas_density = gas_density_grid[i, j, k]

                # FVF guards for cell i
                cell_safe_oil_fvf = oil_formation_volume_factor_grid[i, j, k]
                cell_safe_water_fvf = water_formation_volume_factor_grid[i, j, k]
                cell_safe_gas_fvf = gas_formation_volume_factor_grid[i, j, k]
                if cell_safe_oil_fvf < 1e-30:
                    cell_safe_oil_fvf = 1e-30
                if cell_safe_water_fvf < 1e-30:
                    cell_safe_water_fvf = 1e-30
                if cell_safe_gas_fvf < 1e-30:
                    cell_safe_gas_fvf = 1e-30

                cell_alpha_solution_gor = (
                    solution_gas_to_oil_ratio_grid[i, j, k]
                    * cell_safe_gas_fvf
                    / (cell_safe_oil_fvf * bbl_to_ft3)
                )
                cell_alpha_gas_solubility_in_water = (
                    gas_solubility_in_water_grid[i, j, k]
                    * cell_safe_gas_fvf
                    / (cell_safe_water_fvf * bbl_to_ft3)
                )

                # Water accumulation diagonal: dR_w/dsw = water_density * phi*V/dt
                water_accumulation_diagonal = cell_water_density * accumulation_coefficient
                all_rows[i, local_ptr] = water_row
                all_cols[i, local_ptr] = cell_water_column
                all_vals[i, local_ptr] = water_accumulation_diagonal
                local_ptr += 1

                # Gas accumulation diagonal: dR_g/dsg = (gas_density - gas_density*alpha_Rs) * phi*V/dt
                # Clamp to a small positive value to prevent negative diagonal
                # which destabilises iterative solvers when dissolved gas dominates
                gas_accumulation_diagonal = max(
                    (cell_gas_density - cell_gas_density * cell_alpha_solution_gor)
                    * accumulation_coefficient,
                    1e-6 * water_accumulation_diagonal,  # floor at fraction of water diagonal
                )

                # The diagonal must remain positive for solver stability; if dissolved-gas
                # term dominates (near/above bubble point) we floor at a small positive value
                # proportional to the water accumulation term so the scaling stays consistent
                all_rows[i, local_ptr] = gas_row
                all_cols[i, local_ptr] = cell_gas_column
                all_vals[i, local_ptr] = gas_accumulation_diagonal
                local_ptr += 1

                # Gas-water coupling accumulation:
                # dR_g/dsw = (gas_density*alpha_Rsw - gas_density*alpha_Rs) * phi*V/dt
                # From dM_g/dsw with So = 1 - Sw - Sg substituted:
                #   M_g = gas_density*Sg + gas_density*alpha_Rs*(1-Sw-Sg) + gas_density*alpha_Rsw*Sw
                #   dM_g/dsw = -gas_density*alpha_Rs + gas_density*alpha_Rsw
                coupling_value = (
                    cell_gas_density * cell_alpha_gas_solubility_in_water
                    - cell_gas_density * cell_alpha_solution_gor
                ) * accumulation_coefficient
                if coupling_value != 0:
                    all_rows[i, local_ptr] = gas_row
                    all_cols[i, local_ptr] = cell_water_column
                    all_vals[i, local_ptr] = coupling_value
                    local_ptr += 1

                for face in range(6):
                    if face == 0:  # East
                        ni, nj, nk = np.int64(i + 1), np.int64(j), np.int64(k)
                        T = (
                            face_transmissibilities_x[i + 1, j + 1, k + 1]
                            * md_per_cp_to_ft2_per_psi_per_day
                        )
                    elif face == 1:  # West
                        ni, nj, nk = np.int64(i - 1), np.int64(j), np.int64(k)
                        T = (
                            face_transmissibilities_x[ni + 1, nj + 1, nk + 1]
                            * md_per_cp_to_ft2_per_psi_per_day
                        )
                    elif face == 2:  # South
                        ni, nj, nk = np.int64(i), np.int64(j + 1), np.int64(k)
                        T = (
                            face_transmissibilities_y[i + 1, j + 1, k + 1]
                            * md_per_cp_to_ft2_per_psi_per_day
                        )
                    elif face == 3:  # North
                        ni, nj, nk = np.int64(i), np.int64(j - 1), np.int64(k)
                        T = (
                            face_transmissibilities_y[ni + 1, nj + 1, nk + 1]
                            * md_per_cp_to_ft2_per_psi_per_day
                        )
                    elif face == 4:  # Bottom
                        ni, nj, nk = np.int64(i), np.int64(j), np.int64(k + 1)
                        T = (
                            face_transmissibilities_z[i + 1, j + 1, k + 1]
                            * md_per_cp_to_ft2_per_psi_per_day
                        )
                    else:  # Top (face == 5)
                        ni, nj, nk = np.int64(i), np.int64(j), np.int64(k - 1)
                        T = (
                            face_transmissibilities_z[ni + 1, nj + 1, nk + 1]
                            * md_per_cp_to_ft2_per_psi_per_day
                        )

                    if (
                        ni < 0
                        or ni >= cell_count_x
                        or nj < 0
                        or nj >= cell_count_y
                        or nk < 0
                        or nk >= cell_count_z
                    ):
                        continue

                    neighbour_idx = to_1D_index(
                        i=ni,  # type: ignore[arg-type]
                        j=nj,  # type: ignore[arg-type]
                        k=nk,  # type: ignore[arg-type]
                        cell_count_x=cell_count_x,
                        cell_count_y=cell_count_y,
                        cell_count_z=cell_count_z,
                    )
                    neighbour_water_column = 2 * neighbour_idx
                    neighbour_gas_column = 2 * neighbour_idx + 1

                    # Phase potentials
                    oil_pressure_difference = pressure_grid[ni, nj, nk] - pressure_grid[i, j, k]
                    elevation_difference = elevation_grid[ni, nj, nk] - elevation_grid[i, j, k]
                    oil_water_capillary_pressure_difference = (
                        oil_water_capillary_pressure_grid[ni, nj, nk]
                        - oil_water_capillary_pressure_grid[i, j, k]
                    )
                    gas_oil_capillary_pressure_difference = (
                        gas_oil_capillary_pressure_grid[ni, nj, nk]
                        - gas_oil_capillary_pressure_grid[i, j, k]
                    )
                    water_pressure_difference = (
                        oil_pressure_difference - oil_water_capillary_pressure_difference
                    )
                    gas_pressure_difference = (
                        oil_pressure_difference + gas_oil_capillary_pressure_difference
                    )

                    # Gravity potentials: density upwinded on pressure gradient
                    upwind_water_density_for_gravity = (
                        water_density_grid[ni, nj, nk]
                        if water_pressure_difference > 0
                        else water_density_grid[i, j, k]
                    )
                    upwind_oil_density_for_gravity = (
                        oil_density_grid[ni, nj, nk]
                        if oil_pressure_difference > 0
                        else oil_density_grid[i, j, k]
                    )
                    upwind_gas_density_for_gravity = (
                        gas_density_grid[ni, nj, nk]
                        if gas_pressure_difference > 0
                        else gas_density_grid[i, j, k]
                    )

                    water_potential = (
                        water_pressure_difference
                        + upwind_water_density_for_gravity
                        * gravitational_constant
                        * elevation_difference
                        / 144.0
                    )
                    oil_potential = (
                        oil_pressure_difference
                        + upwind_oil_density_for_gravity
                        * gravitational_constant
                        * elevation_difference
                        / 144.0
                    )
                    gas_potential = (
                        gas_pressure_difference
                        + upwind_gas_density_for_gravity
                        * gravitational_constant
                        * elevation_difference
                        / 144.0
                    )

                    water_neighbour_is_upwind = water_potential > 0.0
                    oil_neighbour_is_upwind = oil_potential > 0.0
                    gas_neighbour_is_upwind = gas_potential > 0.0

                    # So-eliminated relperm derivatives (dso/dsw = dso/dsg = -1)
                    dkrw_dsw_i_eff = dkrw_dsw_grid[i, j, k] - dkrw_dso_grid[i, j, k]
                    dkrw_dsg_i_eff = dkrw_dsg_grid[i, j, k] - dkrw_dso_grid[i, j, k]
                    dkro_dsw_i_eff = dkro_dsw_grid[i, j, k] - dkro_dso_grid[i, j, k]
                    dkro_dsg_i_eff = dkro_dsg_grid[i, j, k] - dkro_dso_grid[i, j, k]
                    dkrg_dsw_i_eff = dkrg_dsw_grid[i, j, k] - dkrg_dso_grid[i, j, k]
                    dkrg_dsg_i_eff = dkrg_dsg_grid[i, j, k] - dkrg_dso_grid[i, j, k]

                    dkrw_dsw_n_eff = dkrw_dsw_grid[ni, nj, nk] - dkrw_dso_grid[ni, nj, nk]
                    dkrw_dsg_n_eff = dkrw_dsg_grid[ni, nj, nk] - dkrw_dso_grid[ni, nj, nk]
                    dkro_dsw_n_eff = dkro_dsw_grid[ni, nj, nk] - dkro_dso_grid[ni, nj, nk]
                    dkro_dsg_n_eff = dkro_dsg_grid[ni, nj, nk] - dkro_dso_grid[ni, nj, nk]
                    dkrg_dsw_n_eff = dkrg_dsw_grid[ni, nj, nk] - dkrg_dso_grid[ni, nj, nk]
                    dkrg_dsg_n_eff = dkrg_dsg_grid[ni, nj, nk] - dkrg_dso_grid[ni, nj, nk]

                    # FVF guards for neighbour; compute alpha on each side then average
                    # Arithmetic averaging eliminates O(1) jumps when upwind cell
                    # switches at a saturation front, keeping the Jacobian consistent.
                    neighbour_safe_oil_fvf = oil_formation_volume_factor_grid[ni, nj, nk]
                    neighbour_safe_gas_fvf = gas_formation_volume_factor_grid[ni, nj, nk]
                    neighbour_safe_water_fvf = water_formation_volume_factor_grid[ni, nj, nk]
                    if neighbour_safe_oil_fvf < 1e-30:
                        neighbour_safe_oil_fvf = 1e-30
                    if neighbour_safe_gas_fvf < 1e-30:
                        neighbour_safe_gas_fvf = 1e-30
                    if neighbour_safe_water_fvf < 1e-30:
                        neighbour_safe_water_fvf = 1e-30

                    neighbour_alpha_solution_gor = (
                        solution_gas_to_oil_ratio_grid[ni, nj, nk]
                        * neighbour_safe_gas_fvf
                        / (neighbour_safe_oil_fvf * bbl_to_ft3)
                    )
                    neighbour_alpha_gas_solubility_in_water = (
                        gas_solubility_in_water_grid[ni, nj, nk]
                        * neighbour_safe_gas_fvf
                        / (neighbour_safe_water_fvf * bbl_to_ft3)
                    )

                    # Inverse viscosities (upwind cell only; zero for non-upwind)
                    if not water_neighbour_is_upwind:
                        cell_inverse_water_viscosity = (
                            1.0 / water_viscosity_grid[i, j, k]
                            if water_viscosity_grid[i, j, k] > 0.0
                            else 0.0
                        )
                        neighbour_inverse_water_viscosity = 0.0
                        face_alpha_gas_solubility_in_water = cell_alpha_gas_solubility_in_water
                        face_water_density = water_density_grid[i, j, k]
                    else:
                        cell_inverse_water_viscosity = 0.0
                        neighbour_inverse_water_viscosity = (
                            1.0 / water_viscosity_grid[ni, nj, nk]
                            if water_viscosity_grid[ni, nj, nk] > 0.0
                            else 0.0
                        )
                        face_alpha_gas_solubility_in_water = (
                            neighbour_alpha_gas_solubility_in_water
                        )
                        face_water_density = water_density_grid[ni, nj, nk]

                    if not oil_neighbour_is_upwind:
                        cell_inverse_oil_viscosity = (
                            1.0 / oil_viscosity_grid[i, j, k]
                            if oil_viscosity_grid[i, j, k] > 0.0
                            else 0.0
                        )
                        neighbour_inverse_oil_viscosity = 0.0
                        face_alpha_solution_gor = cell_alpha_solution_gor
                    else:
                        cell_inverse_oil_viscosity = 0.0
                        neighbour_inverse_oil_viscosity = (
                            1.0 / oil_viscosity_grid[ni, nj, nk]
                            if oil_viscosity_grid[ni, nj, nk] > 0.0
                            else 0.0
                        )
                        face_alpha_solution_gor = neighbour_alpha_solution_gor

                    if not gas_neighbour_is_upwind:
                        cell_inverse_gas_viscosity = (
                            1.0 / gas_viscosity_grid[i, j, k]
                            if gas_viscosity_grid[i, j, k] > 0.0
                            else 0.0
                        )
                        neighbour_inverse_gas_viscosity = 0.0
                        face_gas_density = gas_density_grid[i, j, k]
                    else:
                        cell_inverse_gas_viscosity = 0.0
                        neighbour_inverse_gas_viscosity = (
                            1.0 / gas_viscosity_grid[ni, nj, nk]
                            if gas_viscosity_grid[ni, nj, nk] > 0.0
                            else 0.0
                        )
                        face_gas_density = gas_density_grid[ni, nj, nk]

                    # Mobility flux derivatives: dF_alpha/dS from relperm (upwind only)
                    # dF = (dkr/dS / mu) * potential * T

                    # Water
                    dFw_mob_dsw_i = (
                        dkrw_dsw_i_eff * cell_inverse_water_viscosity * water_potential * T
                    )
                    dFw_mob_dsg_i = (
                        dkrw_dsg_i_eff * cell_inverse_water_viscosity * water_potential * T
                    )
                    dFw_mob_dsw_n = (
                        dkrw_dsw_n_eff * neighbour_inverse_water_viscosity * water_potential * T
                    )
                    dFw_mob_dsg_n = (
                        dkrw_dsg_n_eff * neighbour_inverse_water_viscosity * water_potential * T
                    )

                    # Oil
                    dFo_mob_dsw_i = dkro_dsw_i_eff * cell_inverse_oil_viscosity * oil_potential * T
                    dFo_mob_dsg_i = dkro_dsg_i_eff * cell_inverse_oil_viscosity * oil_potential * T
                    dFo_mob_dsw_n = (
                        dkro_dsw_n_eff * neighbour_inverse_oil_viscosity * oil_potential * T
                    )
                    dFo_mob_dsg_n = (
                        dkro_dsg_n_eff * neighbour_inverse_oil_viscosity * oil_potential * T
                    )

                    # Gas
                    dFg_mob_dsw_i = dkrg_dsw_i_eff * cell_inverse_gas_viscosity * gas_potential * T
                    dFg_mob_dsg_i = dkrg_dsg_i_eff * cell_inverse_gas_viscosity * gas_potential * T
                    dFg_mob_dsw_n = (
                        dkrg_dsw_n_eff * neighbour_inverse_gas_viscosity * gas_potential * T
                    )
                    dFg_mob_dsg_n = (
                        dkrg_dsg_n_eff * neighbour_inverse_gas_viscosity * gas_potential * T
                    )

                    # Capillary flux derivatives: dF_alpha/dS from capillary pressure
                    # The upwind mobility is frozen; the Pc gradient changes with S.
                    # Sign: increasing Pc at cell i reduces potential seen by neighbour,
                    # so the owner derivative is negated vs. the neighbour derivative.
                    # Oil potential has no capillary correction, so dFo_cap = 0.
                    upwind_water_relative_mobility = (
                        water_relative_mobility_grid[ni, nj, nk]
                        if water_neighbour_is_upwind
                        else water_relative_mobility_grid[i, j, k]
                    )
                    upwind_gas_relative_mobility = (
                        gas_relative_mobility_grid[ni, nj, nk]
                        if gas_neighbour_is_upwind
                        else gas_relative_mobility_grid[i, j, k]
                    )

                    # Water capillary: d(Phi_w)/dS = -d(Pcow)/dS at owner, +d(Pcow)/dS at neighbour
                    dFw_cap_dsw_i = (
                        upwind_water_relative_mobility * (dpcow_dsw_eff_grid[i, j, k]) * T
                    )
                    dFw_cap_dsg_i = (
                        upwind_water_relative_mobility * (dpcow_dsg_eff_grid[i, j, k]) * T
                    )
                    dFw_cap_dsw_n = (
                        upwind_water_relative_mobility * (-dpcow_dsw_eff_grid[ni, nj, nk]) * T
                    )
                    dFw_cap_dsg_n = (
                        upwind_water_relative_mobility * (-dpcow_dsg_eff_grid[ni, nj, nk]) * T
                    )

                    # Gas capillary: d(Phi_g)/dS = +d(Pcgo)/dS at owner, -d(Pcgo)/dS at neighbour
                    dFg_cap_dsw_i = (
                        upwind_gas_relative_mobility * (dpcgo_dsw_eff_grid[i, j, k]) * T
                    )
                    dFg_cap_dsg_i = (
                        upwind_gas_relative_mobility * (dpcgo_dsg_eff_grid[i, j, k]) * T
                    )
                    dFg_cap_dsw_n = (
                        upwind_gas_relative_mobility * (-dpcgo_dsw_eff_grid[ni, nj, nk]) * T
                    )
                    dFg_cap_dsg_n = (
                        upwind_gas_relative_mobility * (-dpcgo_dsg_eff_grid[ni, nj, nk]) * T
                    )

                    # Water residual Jacobian: dR_w/dS = -water_density_face * dF_w/dS
                    dRw_dsw_i = -face_water_density * (dFw_mob_dsw_i + dFw_cap_dsw_i)
                    dRw_dsg_i = -face_water_density * (dFw_mob_dsg_i + dFw_cap_dsg_i)
                    dRw_dsw_n = -face_water_density * (dFw_mob_dsw_n + dFw_cap_dsw_n)
                    dRw_dsg_n = -face_water_density * (dFw_mob_dsg_n + dFw_cap_dsg_n)

                    # Gas residual Jacobian from three streams:
                    #   Stream 1 — free gas: -gas_density_face * dF_g/dS
                    #   Stream 2 — dissolved in oil: -gas_density_face * alpha_Rs_face * dF_o/dS
                    #   Stream 3 — dissolved in water: -gas_density_face * alpha_Rsw_face * dF_w/dS

                    # Stream 1: free gas
                    dRg_gas_dsw_i = -face_gas_density * (dFg_mob_dsw_i + dFg_cap_dsw_i)
                    dRg_gas_dsg_i = -face_gas_density * (dFg_mob_dsg_i + dFg_cap_dsg_i)
                    dRg_gas_dsw_n = -face_gas_density * (dFg_mob_dsw_n + dFg_cap_dsw_n)
                    dRg_gas_dsg_n = -face_gas_density * (dFg_mob_dsg_n + dFg_cap_dsg_n)

                    # Stream 2: dissolved gas in oil (oil potential has no Pc correction)
                    dRg_oil_dsw_i = -face_gas_density * face_alpha_solution_gor * dFo_mob_dsw_i
                    dRg_oil_dsg_i = -face_gas_density * face_alpha_solution_gor * dFo_mob_dsg_i
                    dRg_oil_dsw_n = -face_gas_density * face_alpha_solution_gor * dFo_mob_dsw_n
                    dRg_oil_dsg_n = -face_gas_density * face_alpha_solution_gor * dFo_mob_dsg_n

                    # Stream 3: dissolved gas in water (water flux, same mobility+cap derivatives)
                    dRg_water_dsw_i = (
                        -face_gas_density
                        * face_alpha_gas_solubility_in_water
                        * (dFw_mob_dsw_i + dFw_cap_dsw_i)
                    )
                    dRg_water_dsg_i = (
                        -face_gas_density
                        * face_alpha_gas_solubility_in_water
                        * (dFw_mob_dsg_i + dFw_cap_dsg_i)
                    )
                    dRg_water_dsw_n = (
                        -face_gas_density
                        * face_alpha_gas_solubility_in_water
                        * (dFw_mob_dsw_n + dFw_cap_dsw_n)
                    )
                    dRg_water_dsg_n = (
                        -face_gas_density
                        * face_alpha_gas_solubility_in_water
                        * (dFw_mob_dsg_n + dFw_cap_dsg_n)
                    )

                    # Total gas Jacobian entries
                    dRg_dsw_i = dRg_gas_dsw_i + dRg_oil_dsw_i + dRg_water_dsw_i
                    dRg_dsg_i = dRg_gas_dsg_i + dRg_oil_dsg_i + dRg_water_dsg_i
                    dRg_dsw_n = dRg_gas_dsw_n + dRg_oil_dsw_n + dRg_water_dsw_n
                    dRg_dsg_n = dRg_gas_dsg_n + dRg_oil_dsg_n + dRg_water_dsg_n

                    # Write non-zero entries to thread-local COO buffer
                    if dRw_dsw_i != 0:
                        all_rows[i, local_ptr] = water_row
                        all_cols[i, local_ptr] = cell_water_column
                        all_vals[i, local_ptr] = dRw_dsw_i
                        local_ptr += 1
                    if dRw_dsg_i != 0:
                        all_rows[i, local_ptr] = water_row
                        all_cols[i, local_ptr] = cell_gas_column
                        all_vals[i, local_ptr] = dRw_dsg_i
                        local_ptr += 1
                    if dRg_dsw_i != 0:
                        all_rows[i, local_ptr] = gas_row
                        all_cols[i, local_ptr] = cell_water_column
                        all_vals[i, local_ptr] = dRg_dsw_i
                        local_ptr += 1
                    if dRg_dsg_i != 0:
                        all_rows[i, local_ptr] = gas_row
                        all_cols[i, local_ptr] = cell_gas_column
                        all_vals[i, local_ptr] = dRg_dsg_i
                        local_ptr += 1
                    if dRw_dsw_n != 0:
                        all_rows[i, local_ptr] = water_row
                        all_cols[i, local_ptr] = neighbour_water_column
                        all_vals[i, local_ptr] = dRw_dsw_n
                        local_ptr += 1
                    if dRw_dsg_n != 0:
                        all_rows[i, local_ptr] = water_row
                        all_cols[i, local_ptr] = neighbour_gas_column
                        all_vals[i, local_ptr] = dRw_dsg_n
                        local_ptr += 1
                    if dRg_dsw_n != 0:
                        all_rows[i, local_ptr] = gas_row
                        all_cols[i, local_ptr] = neighbour_water_column
                        all_vals[i, local_ptr] = dRg_dsw_n
                        local_ptr += 1
                    if dRg_dsg_n != 0:
                        all_rows[i, local_ptr] = gas_row
                        all_cols[i, local_ptr] = neighbour_gas_column
                        all_vals[i, local_ptr] = dRg_dsg_n
                        local_ptr += 1

        slice_fill[i] = local_ptr

    # Sequential compaction of per-slice buffers into flat COO arrays
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


def assemble_well_contributions(
    pressure_grid: ThreeDimensionalGrid,
    water_density_grid: ThreeDimensionalGrid,
    gas_density_grid: ThreeDimensionalGrid,
    solution_gas_to_oil_ratio_grid: ThreeDimensionalGrid,
    gas_solubility_in_water_grid: ThreeDimensionalGrid,
    gas_formation_volume_factor_grid: ThreeDimensionalGrid,
    oil_formation_volume_factor_grid: ThreeDimensionalGrid,
    water_formation_volume_factor_grid: ThreeDimensionalGrid,
    water_viscosity_grid: ThreeDimensionalGrid,
    oil_viscosity_grid: ThreeDimensionalGrid,
    gas_viscosity_grid: ThreeDimensionalGrid,
    dkrw_dsw_grid: ThreeDimensionalGrid,
    dkrw_dso_grid: ThreeDimensionalGrid,
    dkrw_dsg_grid: ThreeDimensionalGrid,
    dkro_dsw_grid: ThreeDimensionalGrid,
    dkro_dso_grid: ThreeDimensionalGrid,
    dkro_dsg_grid: ThreeDimensionalGrid,
    dkrg_dsw_grid: ThreeDimensionalGrid,
    dkrg_dso_grid: ThreeDimensionalGrid,
    dkrg_dsg_grid: ThreeDimensionalGrid,
    wells_indices: WellsIndices,
    injection_bhps: BottomHolePressures[float, ThreeDimensions],
    production_bhps: BottomHolePressures[float, ThreeDimensions],
    md_per_cp_to_ft2_per_psi_per_day: float,
    bbl_to_ft3: float,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Assemble well contributions to the mass-based saturation Jacobian.

    For BHP-controlled wells, phase mass rates depend on saturation through
    relative permeability, giving non-zero Jacobian entries. Rate-controlled
    injection wells have fixed rates so their Jacobian is zero.

    The mass residual for water is:
        R_w = accum - water_density * F_w - water_density * q_w_vol
    So dR_w/dS_well = -water_density * WI * (dkr_w/dS / water_viscosity) * drawdown * T

    The mass residual for gas includes dissolved components:
        R_g includes: -gas_density*q_g - gas_density*alpha_Rs*q_o - gas_density*alpha_Rsw*q_w
    So dR_g/dS_well = -gas_density * WI * (dkrg/dS / gas_viscosity) * drawdown_g * T
                    - gas_density * alpha_Rs * WI * (dkro/dS / oil_viscosity) * drawdown_o * T
                    - gas_density * alpha_Rsw * WI * (dkrw/dS / water_viscosity) * drawdown_w * T
    """
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    def _add_contribution(cell_idx: int, row_offset: int, col_offset: int, value: float) -> None:
        if value == 0:
            return
        rows.append(2 * cell_idx + row_offset)
        cols.append(2 * cell_idx + col_offset)
        vals.append(value)

    def _alpha_solution_gor(i: int, j: int, k: int) -> float:
        gas_fvf = max(gas_formation_volume_factor_grid[i, j, k], 1e-30)
        oil_fvf = max(oil_formation_volume_factor_grid[i, j, k], 1e-30) * bbl_to_ft3
        return solution_gas_to_oil_ratio_grid[i, j, k] * gas_fvf / oil_fvf

    def _alpha_gas_solubility_in_water(i: int, j: int, k: int) -> float:
        gas_fvf = max(gas_formation_volume_factor_grid[i, j, k], 1e-30)
        water_fvf = max(water_formation_volume_factor_grid[i, j, k], 1e-30) * bbl_to_ft3
        return gas_solubility_in_water_grid[i, j, k] * gas_fvf / water_fvf

    def _effective_derivative(
        dkr_dsw: npt.NDArray,
        dkr_dso: npt.NDArray,
        dkr_dsg: npt.NDArray,
        i: int,
        j: int,
        k: int,
    ):
        """So-eliminated effective derivatives."""
        dkr_dsw_eff = dkr_dsw[i, j, k] - dkr_dso[i, j, k]
        dkr_dsg_eff = dkr_dsg[i, j, k] - dkr_dso[i, j, k]
        return dkr_dsw_eff, dkr_dsg_eff

    for well_indices in wells_indices.injection.values():
        for perf in well_indices:
            i, j, k = perf.cell
            cell_idx = perf.cell_idx
            well_index = perf.well_index
            cell_pressure = float(pressure_grid[i, j, k])
            water_bhp, _, gas_bhp = injection_bhps[i, j, k]

            T = well_index * md_per_cp_to_ft2_per_psi_per_day * 2 * np.pi

            if np.isfinite(water_bhp) and water_bhp != 0:
                drawdown = water_bhp - cell_pressure
                water_viscosity = float(water_viscosity_grid[i, j, k])
                inverse_water_viscosity = 1.0 / water_viscosity if water_viscosity > 0.0 else 0.0
                water_density = float(water_density_grid[i, j, k])
                dkrw_dsw_eff, dkrw_dsg_eff = _effective_derivative(
                    dkrw_dsw_grid, dkrw_dso_grid, dkrw_dsg_grid, i, j, k
                )

                # dR_w/dsw, dR_w/dsg from water well
                dqw_dsw = T * inverse_water_viscosity * dkrw_dsw_eff * drawdown
                dqw_dsg = T * inverse_water_viscosity * dkrw_dsg_eff * drawdown
                _add_contribution(cell_idx, 0, 0, -water_density * dqw_dsw)
                _add_contribution(cell_idx, 0, 1, -water_density * dqw_dsg)

            if np.isfinite(gas_bhp) and gas_bhp != 0:
                drawdown = gas_bhp - cell_pressure
                gas_viscosity = float(gas_viscosity_grid[i, j, k])
                inverse_gas_viscosity = 1.0 / gas_viscosity if gas_viscosity > 0.0 else 0.0
                gas_density = float(gas_density_grid[i, j, k])
                dkrg_dsw_eff, dkrg_dsg_eff = _effective_derivative(
                    dkrg_dsw_grid, dkrg_dso_grid, dkrg_dsg_grid, i, j, k
                )

                dqg_dsw = T * inverse_gas_viscosity * dkrg_dsw_eff * drawdown
                dqg_dsg = T * inverse_gas_viscosity * dkrg_dsg_eff * drawdown
                _add_contribution(cell_idx, 1, 0, -gas_density * dqg_dsw)
                _add_contribution(cell_idx, 1, 1, -gas_density * dqg_dsg)

    for well_indices in wells_indices.production.values():
        for perf in well_indices:
            i, j, k = perf.cell
            cell_idx = perf.cell_idx
            well_index = perf.well_index
            cell_pressure = float(pressure_grid[i, j, k])
            water_bhp, oil_bhp, gas_bhp = production_bhps[i, j, k]

            T = well_index * md_per_cp_to_ft2_per_psi_per_day * 2 * np.pi
            alpha_solution_gor = _alpha_solution_gor(i, j, k)
            alpha_gas_solubility_in_water = _alpha_gas_solubility_in_water(i, j, k)
            gas_density = float(gas_density_grid[i, j, k])

            if np.isfinite(water_bhp) and water_bhp != 0:
                drawdown = water_bhp - cell_pressure
                water_viscosity = float(water_viscosity_grid[i, j, k])
                inverse_water_viscosity = 1.0 / water_viscosity if water_viscosity > 0.0 else 0.0
                water_density = float(water_density_grid[i, j, k])
                dkrw_dsw_eff, dkrw_dsg_eff = _effective_derivative(
                    dkrw_dsw_grid, dkrw_dso_grid, dkrw_dsg_grid, i, j, k
                )

                dqw_dsw = T * inverse_water_viscosity * dkrw_dsw_eff * drawdown
                dqw_dsg = T * inverse_water_viscosity * dkrw_dsg_eff * drawdown
                # dR_w/dS from water production
                _add_contribution(cell_idx, 0, 0, -water_density * dqw_dsw)
                _add_contribution(cell_idx, 0, 1, -water_density * dqw_dsg)
                # dR_g/dS from dissolved gas in produced water
                _add_contribution(
                    cell_idx,
                    1,
                    0,
                    -gas_density * alpha_gas_solubility_in_water * dqw_dsw,
                )
                _add_contribution(
                    cell_idx,
                    1,
                    1,
                    -gas_density * alpha_gas_solubility_in_water * dqw_dsg,
                )

            if np.isfinite(oil_bhp) and oil_bhp != 0:
                drawdown = oil_bhp - cell_pressure
                oil_viscosity = float(oil_viscosity_grid[i, j, k])
                inverse_oil_viscosity = 1.0 / oil_viscosity if oil_viscosity > 0.0 else 0.0
                dkro_dsw_eff, dkro_dsg_eff = _effective_derivative(
                    dkro_dsw_grid, dkro_dso_grid, dkro_dsg_grid, i, j, k
                )

                dqo_dsw = T * inverse_oil_viscosity * dkro_dsw_eff * drawdown
                dqo_dsg = T * inverse_oil_viscosity * dkro_dsg_eff * drawdown
                # dR_g/dS from dissolved gas in produced oil
                _add_contribution(cell_idx, 1, 0, -gas_density * alpha_solution_gor * dqo_dsw)
                _add_contribution(cell_idx, 1, 1, -gas_density * alpha_solution_gor * dqo_dsg)

            if np.isfinite(gas_bhp) and gas_bhp != 0:
                drawdown = gas_bhp - cell_pressure
                gas_viscosity = float(gas_viscosity_grid[i, j, k])
                inverse_gas_viscosity = 1.0 / gas_viscosity if gas_viscosity > 0.0 else 0.0
                dkrg_dsw_eff, dkrg_dsg_eff = _effective_derivative(
                    dkrg_dsw_grid, dkrg_dso_grid, dkrg_dsg_grid, i, j, k
                )

                dqg_dsw = T * inverse_gas_viscosity * dkrg_dsw_eff * drawdown
                dqg_dsg = T * inverse_gas_viscosity * dkrg_dsg_eff * drawdown
                _add_contribution(cell_idx, 1, 0, -gas_density * dqg_dsw)
                _add_contribution(cell_idx, 1, 1, -gas_density * dqg_dsg)

    return (
        np.array(rows, dtype=np.int32),
        np.array(cols, dtype=np.int32),
        np.array(vals, dtype=np.float64),
    )


def assemble_analytical_jacobian(
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
    pore_volume_grid: ThreeDimensionalGrid,
    water_saturation_grid: ThreeDimensionalGrid,
    oil_saturation_grid: ThreeDimensionalGrid,
    gas_saturation_grid: ThreeDimensionalGrid,
    pressure_grid: ThreeDimensionalGrid,
    water_density_grid: ThreeDimensionalGrid,
    oil_density_grid: ThreeDimensionalGrid,
    gas_density_grid: ThreeDimensionalGrid,
    solution_gas_to_oil_ratio_grid: ThreeDimensionalGrid,
    gas_solubility_in_water_grid: ThreeDimensionalGrid,
    gas_formation_volume_factor_grid: ThreeDimensionalGrid,
    oil_formation_volume_factor_grid: ThreeDimensionalGrid,
    water_formation_volume_factor_grid: ThreeDimensionalGrid,
    water_viscosity_grid: ThreeDimensionalGrid,
    oil_viscosity_grid: ThreeDimensionalGrid,
    gas_viscosity_grid: ThreeDimensionalGrid,
    face_transmissibilities: FaceTransmissibilities,
    rock_properties: RockProperties[ThreeDimensions],
    hysteresis_state: Hysteresis[ThreeDimensions] | None,
    capillary_pressure_grids: CapillaryPressureGrids[ThreeDimensions],
    relative_mobility_grids: RelativeMobilityGrids[ThreeDimensions],
    elevation_grid: ThreeDimensionalGrid,
    gravitational_constant: float,
    time_step_in_days: float,
    config: Config,
    md_per_cp_to_ft2_per_psi_per_day: float,
    bbl_to_ft3: float,
    wells_indices: WellsIndices,
    injection_bhps: BottomHolePressures[float, ThreeDimensions] | None = None,
    production_bhps: BottomHolePressures[float, ThreeDimensions] | None = None,
) -> coo_matrix:
    """
    Assemble the full mass-based analytical saturation Jacobian.

    :param cell_count_x: Number of cells in x-direction.
    :param cell_count_y: Number of cells in y-direction.
    :param cell_count_z: Number of cells in z-direction.
    :param water_saturation_grid: Current water saturation grid.
    :param oil_saturation_grid: Current oil saturation grid.
    :param gas_saturation_grid: Current gas saturation grid.
    :param pressure_grid: Oil pressure grid (psi).
    :param water_density_grid: Water density at new pressure (lb/ft³).
    :param oil_density_grid: Oil effective density at new pressure (lb/ft³).
    :param gas_density_grid: Gas density at new pressure (lb/ft³).
    :param water_viscosity_grid: Water viscosity grid (cP).
    :param gas_viscosity_grid: Gas viscosity grid (cP).
    :param face_transmissibilities: Precomputed geometric face transmissibilities.
    :param rock_properties: Rock properties.
    :param injection_bhps: Injection bottom-hole pressures proxy.
    :param production_bhps: Production bottom-hole pressures proxy.
    :param capillary_pressure_grids: `(Pcow, Pcgo)` at current iterate.
    :param relative_mobility_grids: `(lam_w, lam_o, lam_g)` at current iterate.
    :param elevation_grid: Cell elevation grid (ft).
    :param gravitational_constant: Gravitational constant conversion factor (lbf/lbm).
    :param time_step_in_days: Time step size (days).
    :param config: Simulation configuration.
    :param wells_indices: Cache of well indices.
    :param md_per_cp_to_ft2_per_psi_per_day: Unit conversion factor.
    :return: Sparse Jacobian of shape `(2N, 2N)` in COO format.
    """
    (
        dkrw_dsw_grid,
        dkrw_dso_grid,
        dkrw_dsg_grid,
        dkro_dsw_grid,
        dkro_dso_grid,
        dkro_dsg_grid,
        dkrg_dsw_grid,
        dkrg_dso_grid,
        dkrg_dsg_grid,
        dpcow_dsw_eff_grid,
        dpcow_dsg_eff_grid,
        dpcgo_dsw_eff_grid,
        dpcgo_dsg_eff_grid,
    ) = compute_rock_fluid_derivatives(
        water_saturation_grid=water_saturation_grid,
        oil_saturation_grid=oil_saturation_grid,
        gas_saturation_grid=gas_saturation_grid,
        rock_properties=rock_properties,
        hysteresis_state=hysteresis_state,
        rock_fluid_tables=config.rock_fluid_tables,
        disable_capillary_effects=config.disable_capillary_effects,
        capillary_strength_factor=config.capillary_strength_factor,
    )

    oil_water_capillary_pressure_grid, gas_oil_capillary_pressure_grid = capillary_pressure_grids
    (
        water_relative_mobility_grid,
        _,
        gas_relative_mobility_grid,
    ) = relative_mobility_grids

    flux_rows, flux_cols, flux_vals = assemble_flux_contributions(
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
        pore_volume_grid=pore_volume_grid,
        pressure_grid=pressure_grid,
        water_density_grid=water_density_grid,
        oil_density_grid=oil_density_grid,
        gas_density_grid=gas_density_grid,
        solution_gas_to_oil_ratio_grid=solution_gas_to_oil_ratio_grid,
        gas_solubility_in_water_grid=gas_solubility_in_water_grid,
        gas_formation_volume_factor_grid=gas_formation_volume_factor_grid,
        oil_formation_volume_factor_grid=oil_formation_volume_factor_grid,
        water_formation_volume_factor_grid=water_formation_volume_factor_grid,
        elevation_grid=elevation_grid,
        gravitational_constant=gravitational_constant,
        water_relative_mobility_grid=water_relative_mobility_grid,
        gas_relative_mobility_grid=gas_relative_mobility_grid,
        face_transmissibilities_x=face_transmissibilities.x,
        face_transmissibilities_y=face_transmissibilities.y,
        face_transmissibilities_z=face_transmissibilities.z,
        oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
        gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
        dkrw_dsw_grid=dkrw_dsw_grid,
        dkrw_dso_grid=dkrw_dso_grid,
        dkrw_dsg_grid=dkrw_dsg_grid,
        dkro_dsw_grid=dkro_dsw_grid,
        dkro_dso_grid=dkro_dso_grid,
        dkro_dsg_grid=dkro_dsg_grid,
        dkrg_dsw_grid=dkrg_dsw_grid,
        dkrg_dso_grid=dkrg_dso_grid,
        dkrg_dsg_grid=dkrg_dsg_grid,
        dpcow_dsw_eff_grid=dpcow_dsw_eff_grid,
        dpcow_dsg_eff_grid=dpcow_dsg_eff_grid,
        dpcgo_dsw_eff_grid=dpcgo_dsw_eff_grid,
        dpcgo_dsg_eff_grid=dpcgo_dsg_eff_grid,
        water_viscosity_grid=water_viscosity_grid,
        oil_viscosity_grid=oil_viscosity_grid,
        gas_viscosity_grid=gas_viscosity_grid,
        time_step_in_days=time_step_in_days,
        md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
        bbl_to_ft3=bbl_to_ft3,
    )
    total_cell_count = cell_count_x * cell_count_y * cell_count_z
    system_size = 2 * total_cell_count

    if injection_bhps is None or production_bhps is None:
        return coo_matrix(
            (flux_vals, (flux_rows, flux_cols)),
            shape=(system_size, system_size),
            dtype=np.float64,
        )

    well_rows, well_cols, well_vals = assemble_well_contributions(
        pressure_grid=pressure_grid,
        water_density_grid=water_density_grid,
        gas_density_grid=gas_density_grid,
        solution_gas_to_oil_ratio_grid=solution_gas_to_oil_ratio_grid,
        gas_solubility_in_water_grid=gas_solubility_in_water_grid,
        gas_formation_volume_factor_grid=gas_formation_volume_factor_grid,
        oil_formation_volume_factor_grid=oil_formation_volume_factor_grid,
        water_formation_volume_factor_grid=water_formation_volume_factor_grid,
        water_viscosity_grid=water_viscosity_grid,
        oil_viscosity_grid=oil_viscosity_grid,
        gas_viscosity_grid=gas_viscosity_grid,
        dkrw_dsw_grid=dkrw_dsw_grid,
        dkrw_dso_grid=dkrw_dso_grid,
        dkrw_dsg_grid=dkrw_dsg_grid,
        dkro_dsw_grid=dkro_dsw_grid,
        dkro_dso_grid=dkro_dso_grid,
        dkro_dsg_grid=dkro_dsg_grid,
        dkrg_dsw_grid=dkrg_dsw_grid,
        dkrg_dso_grid=dkrg_dso_grid,
        dkrg_dsg_grid=dkrg_dsg_grid,
        wells_indices=wells_indices,
        injection_bhps=injection_bhps,
        production_bhps=production_bhps,
        md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
        bbl_to_ft3=bbl_to_ft3,
    )
    all_rows = np.concatenate([flux_rows, well_rows])
    all_cols = np.concatenate([flux_cols, well_cols])
    all_vals = np.concatenate([flux_vals, well_vals])
    return coo_matrix(
        (all_vals, (all_rows, all_cols)),
        shape=(system_size, system_size),
        dtype=np.float64,
    )


def assemble_jacobian(
    config: Config,
    saturation_vector: npt.NDArray,
    base_residual: npt.NDArray,
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
    pore_volume_grid: ThreeDimensionalGrid,
    water_saturation_grid: ThreeDimensionalGrid,
    oil_saturation_grid: ThreeDimensionalGrid,
    gas_saturation_grid: ThreeDimensionalGrid,
    old_water_mass_grid: ThreeDimensionalGrid,
    old_free_gas_mass_grid: ThreeDimensionalGrid,
    old_dissolved_gas_mass_in_oil_grid: ThreeDimensionalGrid,
    old_dissolved_gas_mass_in_water_grid: ThreeDimensionalGrid,
    pressure_grid: ThreeDimensionalGrid,
    face_transmissibilities: FaceTransmissibilities,
    rock_properties: RockProperties[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
    hysteresis_state: Hysteresis[ThreeDimensions] | None,
    elevation_grid: ThreeDimensionalGrid,
    time_step_in_days: float,
    gravitational_constant: float,
    net_water_well_rate_grid: ThreeDimensionalGrid,
    net_oil_well_rate_grid: ThreeDimensionalGrid,
    net_gas_well_rate_grid: ThreeDimensionalGrid,
    pressure_boundaries: ThreeDimensionalGrid,
    flux_boundaries: ThreeDimensionalGrid,
    capillary_pressure_grids: CapillaryPressureGrids[ThreeDimensions],
    relative_mobility_grids: RelativeMobilityGrids[ThreeDimensions],
    md_per_cp_to_ft2_per_psi_per_day: float,
    bbl_to_ft3: float,
    wells_indices: WellsIndices,
    injection_bhps: BottomHolePressures[float, ThreeDimensions] | None = None,
    production_bhps: BottomHolePressures[float, ThreeDimensions] | None = None,
) -> coo_matrix:
    """
    Dispatch Jacobian assembly to the numerical or analytical path.

    :param config: Simulation configuration.
    :param saturation_vector: Current saturation vector (numerical path only).
    :param base_residual: Base residual at current iterate (numerical path only).
    :param cell_count_x: Number of cells in x-direction.
    :param cell_count_y: Number of cells in y-direction.
    :param cell_count_z: Number of cells in z-direction.
    :param water_saturation_grid: Current water saturation grid.
    :param oil_saturation_grid: Current oil saturation grid.
    :param gas_saturation_grid: Current gas saturation grid.
    :param pressure_grid: Fixed oil pressure grid (psi).
    :param face_transmissibilities: Precomputed geometric face transmissibilities.
    :param rock_properties: Rock properties.
    :param fluid_properties: Fluid properties (at new pressure level).
    :param elevation_grid: Cell elevation grid (ft).
    :param time_step_in_days: Time step size (days).
    :param gravitational_constant: Gravitational constant conversion factor (lbf/lbm).
    :param wells_indices: Cache of well indices.
    :param injection_bhps: Injection bottom-hole pressures.
    :param production_bhps: Production bottom-hole pressures.
    :param pressure_boundaries: Padded pressure boundary grid, shape `(nx+2, ny+2, nz+2)`.
    :param flux_boundaries: Padded flux boundary grid, shape `(nx+2, ny+2, nz+2)`.
    :param capillary_pressure_grids: `(Pcow, Pcgo)` at current iterate.
    :param relative_mobility_grids: `(lam_w, lam_o, lam_g)` at current iterate.
    :param md_per_cp_to_ft2_per_psi_per_day: Unit conversion factor.
    :return: Jacobian as a `(2N x 2N)` COO sparse matrix.
    """
    if config.saturation_jacobian_assembly_method == "analytical":
        return assemble_analytical_jacobian(
            cell_count_x=cell_count_x,
            cell_count_y=cell_count_y,
            cell_count_z=cell_count_z,
            pore_volume_grid=pore_volume_grid,
            pressure_grid=pressure_grid,
            water_saturation_grid=water_saturation_grid,
            oil_saturation_grid=oil_saturation_grid,
            gas_saturation_grid=gas_saturation_grid,
            water_density_grid=fluid_properties.water_density_grid,
            oil_density_grid=fluid_properties.oil_effective_density_grid,
            gas_density_grid=fluid_properties.gas_density_grid,
            solution_gas_to_oil_ratio_grid=fluid_properties.solution_gas_to_oil_ratio_grid,
            gas_solubility_in_water_grid=fluid_properties.gas_solubility_in_water_grid,
            gas_formation_volume_factor_grid=fluid_properties.gas_formation_volume_factor_grid,
            oil_formation_volume_factor_grid=fluid_properties.oil_formation_volume_factor_grid,
            water_formation_volume_factor_grid=fluid_properties.water_formation_volume_factor_grid,
            elevation_grid=elevation_grid,
            rock_properties=rock_properties,
            hysteresis_state=hysteresis_state,
            face_transmissibilities=face_transmissibilities,
            gravitational_constant=gravitational_constant,
            water_viscosity_grid=fluid_properties.water_viscosity_grid,
            oil_viscosity_grid=fluid_properties.oil_effective_viscosity_grid,
            gas_viscosity_grid=fluid_properties.gas_viscosity_grid,
            capillary_pressure_grids=capillary_pressure_grids,
            relative_mobility_grids=relative_mobility_grids,
            time_step_in_days=time_step_in_days,
            config=config,
            md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
            bbl_to_ft3=bbl_to_ft3,
            wells_indices=wells_indices,
            injection_bhps=injection_bhps,
            production_bhps=production_bhps,
        )

    return assemble_numerical_jacobian(
        saturation_vector=saturation_vector,
        base_residual=base_residual,
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
        pore_volume_grid=pore_volume_grid,
        water_saturation_grid=water_saturation_grid,
        oil_saturation_grid=oil_saturation_grid,
        gas_saturation_grid=gas_saturation_grid,
        old_water_mass_grid=old_water_mass_grid,
        old_free_gas_mass_grid=old_free_gas_mass_grid,
        old_dissolved_gas_mass_in_oil_grid=old_dissolved_gas_mass_in_oil_grid,
        old_dissolved_gas_mass_in_water_grid=old_dissolved_gas_mass_in_water_grid,
        pressure_grid=pressure_grid,
        face_transmissibilities=face_transmissibilities,
        rock_properties=rock_properties,
        fluid_properties=fluid_properties,
        hysteresis_state=hysteresis_state,
        config=config,
        elevation_grid=elevation_grid,
        time_step_in_days=time_step_in_days,
        gravitational_constant=gravitational_constant,
        net_water_well_rate_grid=net_water_well_rate_grid,
        net_oil_well_rate_grid=net_oil_well_rate_grid,
        net_gas_well_rate_grid=net_gas_well_rate_grid,
        pressure_boundaries=pressure_boundaries,
        flux_boundaries=flux_boundaries,
        md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
        bbl_to_ft3=bbl_to_ft3,
    )


def solve_transport(
    elevation_grid: ThreeDimensionalGrid,
    pore_volume_grid: ThreeDimensionalGrid,
    time_step_size: float,
    rock_properties: RockProperties[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
    face_transmissibilities: FaceTransmissibilities,
    pressure_boundaries: ThreeDimensionalGrid,
    flux_boundaries: ThreeDimensionalGrid,
    config: Config,
    wells_indices: WellsIndices,
    rates: WellRates[ThreeDimensions] | None = None,
    hysteresis_state: Hysteresis[ThreeDimensions] | None = None,
    dtype: npt.DTypeLike = np.float64,
) -> Solution[ImplicitSaturationSolution, list[NewtonConvergenceInfo]]:
    """
    Solve the mass-based implicit transport equations using Newton-Raphson iteration.

    The residual equations conserve fluid mass rather than reservoir-condition volume.
    For water:

        R_w = (phi*V/dt) * (current_water_density * Sw_new - old_water_density * Sw_old)
              - sum_faces(upwind_water_density * F_w_face)
              - water_density_cell * q_w_vol

    For total gas (free + dissolved in oil + water):

        R_g = (phi*V/dt) * (M_g_new - M_g_old)
              - sum_faces(upwind_gas_density * F_g + upwind_oil_density * alpha_Rs_upwind * F_o + upwind_water_density * alpha_Rsw_upwind*F_w)
              - (gas_density_cell * q_g + oil_density_cell * alpha_Rs_cell * q_o + water_density_cell * alpha_Rsw_cell * q_w)

    :param elevation_grid: Cell elevation grid (ft), shape `(nx, ny, nz)`.
    :param time_step_size: Time step size in seconds.
    :param rock_properties: Rock properties.
    :param fluid_properties: Fluid properties at the new (post-pressure-solve) pressure level.
    :param face_transmissibilities: Precomputed geometric face transmissibilities.
    :param pressure_boundaries: Padded Dirichlet BC array, shape `(nx+2, ny+2, nz+2)`.
        NaN indicates a Neumann face.
    :param flux_boundaries: Padded Neumann BC array, shape `(nx+2, ny+2, nz+2)`.
    :param config: Simulation configuration.
    :param wells_indices: Cache of well indices.
    :param dtype: NumPy dtype for output saturation grids.
    :return: `Solution` containing an `ImplicitSaturationSolution` and a list of
        `NewtonConvergenceInfo` records.
    """
    pressure_grid = fluid_properties.pressure_grid
    cell_count_x, cell_count_y, cell_count_z = pressure_grid.shape

    if rates is not None:
        net_water_well_rate_grid = rates.net_water_well_rate_grid
        net_oil_well_rate_grid = rates.net_oil_well_rate_grid
        net_gas_well_rate_grid = rates.net_gas_well_rate_grid
        injection_bhps = rates.injection_bhps
        production_bhps = rates.production_bhps
    else:
        zeros_grid = np.zeros_like(pressure_grid, dtype=dtype)
        net_water_well_rate_grid = zeros_grid
        net_oil_well_rate_grid = zeros_grid
        net_gas_well_rate_grid = zeros_grid
        injection_bhps = None
        production_bhps = None

    old_water_saturation_grid = fluid_properties.water_saturation_grid
    old_oil_saturation_grid = fluid_properties.oil_saturation_grid
    old_gas_saturation_grid = fluid_properties.gas_saturation_grid
    old_water_mass_grid = fluid_properties.water_mass_grid
    old_free_gas_mass_grid = fluid_properties.free_gas_mass_grid
    old_dissolved_gas_mass_in_oil_grid = fluid_properties.dissolved_gas_mass_in_oil_grid
    old_dissolved_gas_mass_in_water_grid = fluid_properties.dissolved_gas_mass_in_water_grid

    time_step_in_days = time_step_size * c.DAYS_PER_SECOND
    gravitational_constant = (
        c.ACCELERATION_DUE_TO_GRAVITY_FEET_PER_SECONDS_SQUARE
        / c.GRAVITATIONAL_FACTOR_LBM_FT_PER_LBF_S2
    )
    md_per_cp_to_ft2_per_psi_per_day = c.MILLIDARCIES_PER_CENTIPOISE_TO_SQUARE_FEET_PER_PSI_PER_DAY
    bbl_to_ft3 = c.BARRELS_TO_CUBIC_FEET

    water_saturation_grid = old_water_saturation_grid.copy()
    oil_saturation_grid = old_oil_saturation_grid.copy()
    gas_saturation_grid = old_gas_saturation_grid.copy()

    saturation_vector = pack_grids(
        water_saturation_grid=water_saturation_grid,
        gas_saturation_grid=gas_saturation_grid,
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
    )

    # Shared kwargs passed to residual functions
    residual_kwargs = dict(
        old_water_mass_grid=old_water_mass_grid,
        old_free_gas_mass_grid=old_free_gas_mass_grid,
        old_dissolved_gas_mass_in_oil_grid=old_dissolved_gas_mass_in_oil_grid,
        old_dissolved_gas_mass_in_water_grid=old_dissolved_gas_mass_in_water_grid,
        pressure_grid=pressure_grid,
        face_transmissibilities=face_transmissibilities,
        fluid_properties=fluid_properties,
        hysteresis_state=hysteresis_state,
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
        pore_volume_grid=pore_volume_grid,
        elevation_grid=elevation_grid,
        time_step_in_days=time_step_in_days,
        gravitational_constant=gravitational_constant,
        net_water_well_rate_grid=net_water_well_rate_grid,
        net_oil_well_rate_grid=net_oil_well_rate_grid,
        net_gas_well_rate_grid=net_gas_well_rate_grid,
        pressure_boundaries=pressure_boundaries,
        flux_boundaries=flux_boundaries,
        md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
        bbl_to_ft3=bbl_to_ft3,
    )

    convergence_history: list[NewtonConvergenceInfo] = []
    initial_residual_norm = 0.0
    converged = False
    final_iteration = 0
    final_residual_norm = 0.0
    best_residual_norm = float("inf")
    stagnation_count = 0

    stagnation_patience = config.newton_stagnation_patience
    stagnation_improvement_threshold = config.newton_stagnation_improvement_threshold
    minimum_step_size = float(np.sqrt(np.finfo(dtype).eps))  # type: ignore
    maximum_newton_iterations = config.maximum_newton_iterations
    newton_tolerance = config.newton_tolerance
    maximum_line_search_cuts = config.maximum_line_search_cuts
    maximum_saturation_change = config.maximum_newton_saturation_change
    transport_convergence_tolerance = config.transport_convergence_tolerance

    for iteration in range(maximum_newton_iterations):
        relative_mobility_grids, capillary_pressure_grids, _ = compute_rock_fluid_properties(
            water_saturation_grid=water_saturation_grid,
            oil_saturation_grid=oil_saturation_grid,
            gas_saturation_grid=gas_saturation_grid,
            rock_properties=rock_properties,
            fluid_properties=fluid_properties,
            hysteresis_state=hysteresis_state,
            config=config,
        )

        water_residual, gas_residual = _assemble_residuals(
            water_saturation_grid=water_saturation_grid,
            gas_saturation_grid=gas_saturation_grid,
            old_water_mass_grid=old_water_mass_grid,
            old_free_gas_mass_grid=old_free_gas_mass_grid,
            old_dissolved_gas_mass_in_oil_grid=old_dissolved_gas_mass_in_oil_grid,
            old_dissolved_gas_mass_in_water_grid=old_dissolved_gas_mass_in_water_grid,
            pressure_grid=pressure_grid,
            face_transmissibilities=face_transmissibilities,
            capillary_pressure_grids=capillary_pressure_grids,
            relative_mobility_grids=relative_mobility_grids,
            fluid_properties=fluid_properties,
            cell_count_x=cell_count_x,
            cell_count_y=cell_count_y,
            cell_count_z=cell_count_z,
            pore_volume_grid=pore_volume_grid,
            elevation_grid=elevation_grid,
            time_step_in_days=time_step_in_days,
            gravitational_constant=gravitational_constant,
            net_water_well_rate_grid=net_water_well_rate_grid,
            net_oil_well_rate_grid=net_oil_well_rate_grid,
            net_gas_well_rate_grid=net_gas_well_rate_grid,
            pressure_boundaries=pressure_boundaries,
            flux_boundaries=flux_boundaries,
            md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
            bbl_to_ft3=bbl_to_ft3,
        )
        residual_vector = interleave_residuals(water_residual, gas_residual)
        residual_norm = float(np.linalg.norm(residual_vector))

        if iteration == 0:
            logger.warning("=== NEWTON RESIDUAL DIAGNOSIS ===")
            logger.warning("||R0|| = %.6e", residual_norm)
            logger.warning(
                "||water_R|| = %.6e, max = %.6e at idx %d",
                float(np.linalg.norm(water_residual)),
                float(np.max(np.abs(water_residual))),
                int(np.argmax(np.abs(water_residual))),
            )
            logger.warning(
                "||gas_R|| = %.6e, max = %.6e at idx %d",
                float(np.linalg.norm(gas_residual)),
                float(np.max(np.abs(gas_residual))),
                int(np.argmax(np.abs(gas_residual))),
            )

            # Decompose the residual at the injector cell (0,0,0)
            inj_i, inj_j, inj_k = 0, 0, 0
            inj_idx = inj_i * cell_count_y * cell_count_z + inj_j * cell_count_z + inj_k
            logger.warning(
                "Injector cell (0,0,0) idx=%d: water_R=%.6e, gas_R=%.6e",
                inj_idx,
                water_residual[inj_idx],
                gas_residual[inj_idx],
            )
            logger.warning(
                "Injector Sg=%.6e, So=%.6e, Sw=%.6e",
                float(gas_saturation_grid[inj_i, inj_j, inj_k]),
                float(oil_saturation_grid[inj_i, inj_j, inj_k]),
                float(water_saturation_grid[inj_i, inj_j, inj_k]),
            )
            logger.warning(
                "Injector gas_rate=%.6e, water_rate=%.6e, oil_rate=%.6e",
                float(net_gas_well_rate_grid[inj_i, inj_j, inj_k]),
                float(net_water_well_rate_grid[inj_i, inj_j, inj_k]),
                float(net_oil_well_rate_grid[inj_i, inj_j, inj_k]),
            )

        if iteration == 0:
            initial_residual_norm = max(residual_norm, 1e-30)

        relative_residual_norm = residual_norm / initial_residual_norm
        max_saturation_update = (
            convergence_history[-1].max_saturation_update if convergence_history else float("inf")
        )

        residual_converged = relative_residual_norm < newton_tolerance and iteration > 0
        saturation_converged = (
            max_saturation_update < transport_convergence_tolerance
            and relative_residual_norm < 1e-3
            and iteration > 1  # require at least 2 real Newton steps
        )

        if residual_converged or saturation_converged:
            converged = True
            final_iteration = iteration
            final_residual_norm = residual_norm
            reason = "residual" if residual_converged else "saturation change"
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Newton converged at iteration %d (%s): ||R||/||R0|| = %.2e, max |ΔS| = %.2e",
                    iteration,
                    reason,
                    relative_residual_norm,
                    max_saturation_update,
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

        jacobian = assemble_jacobian(
            config=config,
            saturation_vector=saturation_vector,
            base_residual=residual_vector,
            cell_count_x=cell_count_x,
            cell_count_y=cell_count_y,
            cell_count_z=cell_count_z,
            pore_volume_grid=pore_volume_grid,
            water_saturation_grid=water_saturation_grid,
            oil_saturation_grid=oil_saturation_grid,
            gas_saturation_grid=gas_saturation_grid,
            old_water_mass_grid=old_water_mass_grid,
            old_free_gas_mass_grid=old_free_gas_mass_grid,
            old_dissolved_gas_mass_in_oil_grid=old_dissolved_gas_mass_in_oil_grid,
            old_dissolved_gas_mass_in_water_grid=old_dissolved_gas_mass_in_water_grid,
            pressure_grid=pressure_grid,
            face_transmissibilities=face_transmissibilities,
            rock_properties=rock_properties,
            fluid_properties=fluid_properties,
            hysteresis_state=hysteresis_state,
            elevation_grid=elevation_grid,
            time_step_in_days=time_step_in_days,
            gravitational_constant=gravitational_constant,
            net_water_well_rate_grid=net_water_well_rate_grid,
            net_oil_well_rate_grid=net_oil_well_rate_grid,
            net_gas_well_rate_grid=net_gas_well_rate_grid,
            pressure_boundaries=pressure_boundaries,
            flux_boundaries=flux_boundaries,
            md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
            bbl_to_ft3=bbl_to_ft3,
            wells_indices=wells_indices,
            injection_bhps=injection_bhps,
            production_bhps=production_bhps,
            capillary_pressure_grids=capillary_pressure_grids,
            relative_mobility_grids=relative_mobility_grids,
        )

        J_csr, residual_vector, column_scaling_vector = scale_linear_system(
            jacobian_csr=jacobian.tocsr(),
            residual_vector=residual_vector,
        )
        saturation_change, _ = solve_linear_system(
            A_csr=J_csr,
            b=-residual_vector,
            solver=config.transport_solver,
            preconditioner=config.transport_preconditioner,
            rtol=config.transport_convergence_tolerance,
            maximum_iterations=config.maximum_solver_iterations,
            fallback_to_direct=True,
        )
        if column_scaling_vector is not None:
            saturation_change *= column_scaling_vector

        if logger.isEnabledFor(logging.DEBUG):
            linear_residual_norm = float(
                np.linalg.norm(jacobian @ saturation_change + residual_vector)
            )
            logger.debug(
                "Newton iteration %d: ||R||/||R0|| = %.2e, linear residual = %.2e",
                iteration,
                relative_residual_norm,
                linear_residual_norm,
            )

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

        def _residual_norm(trial_vec: npt.NDArray) -> float:
            """Unpack trial vector, compute residual, return norm."""
            water_sat_grid = water_saturation_grid.copy()
            oil_sat_grid = oil_saturation_grid.copy()
            gas_sat_grid = gas_saturation_grid.copy()
            unpack_vector(
                saturation_vector=trial_vec,
                water_saturation_grid=water_sat_grid,
                oil_saturation_grid=oil_sat_grid,
                gas_saturation_grid=gas_sat_grid,
                cell_count_x=cell_count_x,
                cell_count_y=cell_count_y,
                cell_count_z=cell_count_z,
            )
            rw, rg = assemble_residuals(
                water_saturation_grid=water_sat_grid,
                oil_saturation_grid=oil_sat_grid,
                gas_saturation_grid=gas_sat_grid,
                rock_properties=rock_properties,
                config=config,
                **residual_kwargs,  # type: ignore
            )
            return float(np.linalg.norm(interleave_residuals(rw, rg)))

        trial_saturation_vector, line_search_factor, _ = line_search(
            saturation_vector=saturation_vector,
            saturation_change=saturation_change,
            current_residual_norm=residual_norm,
            residual_norm_fn=_residual_norm,
            project_fn=lambda v: project_to_feasible(v.copy()),
            maximum_cuts=maximum_line_search_cuts,
            sufficient_decrease=1e-4,
            min_step=minimum_step_size,
        )

        trial_water_saturation_grid = water_saturation_grid.copy()
        trial_oil_saturation_grid = oil_saturation_grid.copy()
        trial_gas_saturation_grid = gas_saturation_grid.copy()
        unpack_vector(
            saturation_vector=trial_saturation_vector,
            water_saturation_grid=trial_water_saturation_grid,
            oil_saturation_grid=trial_oil_saturation_grid,
            gas_saturation_grid=trial_gas_saturation_grid,
            cell_count_x=cell_count_x,
            cell_count_y=cell_count_y,
            cell_count_z=cell_count_z,
        )

        max_saturation_update = float(np.max(np.abs(trial_saturation_vector - saturation_vector)))
        saturation_vector = trial_saturation_vector
        water_saturation_grid = trial_water_saturation_grid
        oil_saturation_grid = trial_oil_saturation_grid
        gas_saturation_grid = trial_gas_saturation_grid

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

        if max_saturation_update < 1e-10:
            if relative_residual_norm < newton_tolerance:
                converged = True
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Newton converged (saturation stagnation) at iteration %d: "
                        "max |ΔS| = %.2e, ||R||/||R0|| = %.2e",
                        iteration,
                        max_saturation_update,
                        relative_residual_norm,
                    )
            else:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Newton stagnated (negligible dS, residual not converged) "
                        "at iteration %d: max |ΔS| = %.2e, ||R||/||R0|| = %.2e",
                        iteration,
                        max_saturation_update,
                        relative_residual_norm,
                    )
            break

        if residual_norm < best_residual_norm * (1.0 - stagnation_improvement_threshold):
            best_residual_norm = residual_norm
            stagnation_count = 0
        else:
            stagnation_count += 1

        if stagnation_count >= stagnation_patience and iteration >= stagnation_patience:
            if relative_residual_norm < 1e-3:
                converged = True
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Newton converged (residual plateau) at iteration %d: ||R||/||R0|| = %.2e",
                        iteration,
                        relative_residual_norm,
                    )
            else:
                logger.warning(
                    "Newton stagnated (residual flat) at iteration %d: "
                    "||R||/||R0|| = %.2e, no improvement for %d iterations",
                    iteration,
                    relative_residual_norm,
                    stagnation_count,
                )
            break

    # After Newton has converged, reconstruct component masses from converged saturations
    new_water_mass_grid = (
        water_saturation_grid * fluid_properties.water_density_grid * pore_volume_grid
    )
    new_oil_mass_grid = (
        oil_saturation_grid * fluid_properties.oil_effective_density_grid * pore_volume_grid
    )
    gas_density = fluid_properties.gas_density_grid
    oil_fvf = np.maximum(fluid_properties.oil_formation_volume_factor_grid, 1e-30)
    water_fvf = np.maximum(fluid_properties.water_formation_volume_factor_grid, 1e-30)
    gas_fvf = np.maximum(fluid_properties.gas_formation_volume_factor_grid, 1e-30)
    alpha_solution_gor = (
        fluid_properties.solution_gas_to_oil_ratio_grid * gas_fvf / (oil_fvf * bbl_to_ft3)
    )
    alpha_gas_solubility_in_water = (
        fluid_properties.gas_solubility_in_water_grid * gas_fvf / (water_fvf * bbl_to_ft3)
    )

    new_dissolved_gas_mass_in_oil_grid = (
        gas_density * alpha_solution_gor * oil_saturation_grid * pore_volume_grid
    )
    new_dissolved_gas_mass_in_water_grid = (
        gas_density * alpha_gas_solubility_in_water * water_saturation_grid * pore_volume_grid
    )
    new_free_gas_mass_grid = gas_density * gas_saturation_grid * pore_volume_grid

    # Compute maximum saturation changes
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
        water_mass_grid=new_water_mass_grid.astype(dtype, copy=False),
        oil_mass_grid=new_oil_mass_grid.astype(dtype, copy=False),
        free_gas_mass_grid=new_free_gas_mass_grid.astype(dtype, copy=False),
        dissolved_gas_mass_in_oil_grid=new_dissolved_gas_mass_in_oil_grid.astype(
            dtype, copy=False
        ),
        dissolved_gas_mass_in_water_grid=new_dissolved_gas_mass_in_water_grid.astype(
            dtype, copy=False
        ),
        newton_iterations=final_iteration,
        final_residual_norm=final_residual_norm,
        maximum_water_saturation_change=maximum_water_saturation_change,
        maximum_oil_saturation_change=maximum_oil_saturation_change,
        maximum_gas_saturation_change=maximum_gas_saturation_change,
    )
    if converged:
        return Solution(
            value=solution,
            scheme="implicit",
            success=True,
            message=(f"Implicit saturation converged in {final_iteration} Newton iterations."),
            metadata=convergence_history,
        )
    return Solution(
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
