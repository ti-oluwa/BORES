import logging
import typing

import attrs
import numba
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from bores._precision import get_dtype
from bores.boundary_conditions import BoundaryConditions
from bores.config import Config
from bores.constants import c
from bores.correlations.core import compute_harmonic_mean
from bores.grids.base import (
    CapillaryPressureGrids,
    RelativeMobilityGrids,
    RelPermGrids,
)
from bores.grids.rock_fluid import build_rock_fluid_properties_grids
from bores.models import FluidProperties, RockProperties
from bores.solvers.base import (
    EvolutionResult,
    compute_mobility_grids,
    from_1D_index_interior_only,
    solve_linear_system,
    to_1D_index_interior_only,
)
from bores.solvers.explicit.saturation import (
    compute_phase_fluxes_from_neighbour,
    compute_well_rate_grids,
)
from bores.types import (
    SupportsSetItem,
    ThreeDimensionalGrid,
    ThreeDimensions,
)
from bores.wells.base import Wells

logger = logging.getLogger(__name__)


@attrs.frozen
class ImplicitSaturationSolution:
    """Result of an implicit saturation solve."""

    water_saturation_grid: ThreeDimensionalGrid
    oil_saturation_grid: ThreeDimensionalGrid
    gas_saturation_grid: ThreeDimensionalGrid
    newton_iterations: int
    final_residual_norm: float
    max_water_saturation_change: float
    max_oil_saturation_change: float
    max_gas_saturation_change: float


@attrs.frozen
class NewtonConvergenceInfo:
    """Per-iteration convergence record."""

    iteration: int
    residual_norm: float
    relative_residual_norm: float
    max_saturation_update: float
    line_search_factor: float


@numba.njit(cache=True)
def saturations_to_vector(
    water_saturation_grid: ThreeDimensionalGrid,
    gas_saturation_grid: ThreeDimensionalGrid,
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
) -> np.typing.NDArray:
    """
    Pack Sw and Sg from 3D grids into a 1D vector.

    Layout: [Sw_0, Sg_0, Sw_1, Sg_1, ..., Sw_{N-1}, Sg_{N-1}]
    where N is the number of interior cells.
    """
    interior_count = (cell_count_x - 2) * (cell_count_y - 2) * (cell_count_z - 2)
    vector = np.empty(2 * interior_count)
    for idx in range(interior_count):
        i, j, k = from_1D_index_interior_only(
            idx, cell_count_x, cell_count_y, cell_count_z
        )
        vector[2 * idx] = water_saturation_grid[i, j, k]
        vector[2 * idx + 1] = gas_saturation_grid[i, j, k]
    return vector


@numba.njit(cache=True)
def vector_to_saturation_grids(
    saturation_vector: np.typing.NDArray,
    water_saturation_grid: ThreeDimensionalGrid,
    oil_saturation_grid: ThreeDimensionalGrid,
    gas_saturation_grid: ThreeDimensionalGrid,
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
) -> None:
    """
    Unpack 1D saturation vector back into 3D grids (in-place).

    Computes oil saturation as So = 1 - Sw - Sg.
    """
    interior_count = (cell_count_x - 2) * (cell_count_y - 2) * (cell_count_z - 2)
    for idx in range(interior_count):
        i, j, k = from_1D_index_interior_only(
            idx, cell_count_x, cell_count_y, cell_count_z
        )
        sw = saturation_vector[2 * idx]
        sg = saturation_vector[2 * idx + 1]
        water_saturation_grid[i, j, k] = sw
        gas_saturation_grid[i, j, k] = sg
        oil_saturation_grid[i, j, k] = max(0.0, 1.0 - sw - sg)


@numba.njit(cache=True)
def project_to_feasible(saturation_vector: np.typing.NDArray) -> np.typing.NDArray:
    """
    Project saturation vector onto the feasible set:
        0 <= Sw, 0 <= Sg, and Sw + Sg <= 1.

    Uses proportional scaling if Sw + Sg > 1.
    """
    interior_count = len(saturation_vector) // 2
    for idx in range(interior_count):
        sw = max(0.0, saturation_vector[2 * idx])
        sg = max(0.0, saturation_vector[2 * idx + 1])
        total = sw + sg
        if total > 1.0:
            sw = sw / total
            sg = sg / total
        saturation_vector[2 * idx] = sw
        saturation_vector[2 * idx + 1] = sg
    return saturation_vector


@numba.njit(cache=True)
def interleave_residuals(
    residual_water: np.typing.NDArray,
    residual_gas: np.typing.NDArray,
) -> np.typing.NDArray:
    """
    Interleave water and gas residual arrays into a single vector.

    Layout: [R_w_0, R_g_0, R_w_1, R_g_1, ..., R_w_{N-1}, R_g_{N-1}]
    """
    interior_count = len(residual_water)
    result = np.empty(2 * interior_count)
    for idx in range(interior_count):
        result[2 * idx] = residual_water[idx]
        result[2 * idx + 1] = residual_gas[idx]
    return result


@numba.njit(parallel=True, cache=True)
def compute_saturation_residual(
    oil_pressure_grid: ThreeDimensionalGrid,
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
    thickness_grid: ThreeDimensionalGrid,
    cell_size_x: float,
    cell_size_y: float,
    water_mobility_grid_x: ThreeDimensionalGrid,
    water_mobility_grid_y: ThreeDimensionalGrid,
    water_mobility_grid_z: ThreeDimensionalGrid,
    oil_mobility_grid_x: ThreeDimensionalGrid,
    oil_mobility_grid_y: ThreeDimensionalGrid,
    oil_mobility_grid_z: ThreeDimensionalGrid,
    gas_mobility_grid_x: ThreeDimensionalGrid,
    gas_mobility_grid_y: ThreeDimensionalGrid,
    gas_mobility_grid_z: ThreeDimensionalGrid,
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
    time_step_in_days: float,
    net_water_well_rate_grid: ThreeDimensionalGrid,
    net_gas_well_rate_grid: ThreeDimensionalGrid,
    pressure_change_grid: ThreeDimensionalGrid,
    water_compressibility_grid: ThreeDimensionalGrid,
    gas_compressibility_grid: ThreeDimensionalGrid,
    rock_compressibility: float,
) -> typing.Tuple[np.typing.NDArray, np.typing.NDArray]:
    """
    Compute the residual vector R(S) for the implicit saturation equations.

    For each interior cell i, the residual equations are:

        R_w[i] = (phi * V / dt) * (Sw_new - Sw_old)
                 - sum_faces(F_w_face)
                 - qw_well * V
                 - PVT_correction_w

        R_g[i] = (phi * V / dt) * (Sg_new - Sg_old)
                 - sum_faces(F_g_face)
                 - qg_well * V
                 - PVT_correction_g

    Pressure is fixed from the implicit pressure solve. Only saturations
    (and hence kr, Pc, mobility) change during Newton iteration.

    Returns two 1D arrays: (residual_water, residual_gas), each of
    length interior_cell_count.
    """
    interior_count = (cell_count_x - 2) * (cell_count_y - 2) * (cell_count_z - 2)
    residual_water = np.zeros(interior_count)
    residual_gas = np.zeros(interior_count)

    for i in numba.prange(1, cell_count_x - 1):  # type: ignore
        for j in range(1, cell_count_y - 1):
            for k in range(1, cell_count_z - 1):
                cell_thickness = thickness_grid[i, j, k]
                cell_total_volume = cell_size_x * cell_size_y * cell_thickness
                cell_porosity = porosity_grid[i, j, k]
                cell_pore_volume = cell_total_volume * cell_porosity

                # Accumulation term: (phi * V / dt) * (S_new - S_old)
                accumulation_coefficient = cell_pore_volume / time_step_in_days
                water_accumulation = accumulation_coefficient * (
                    water_saturation_grid[i, j, k] - old_water_saturation_grid[i, j, k]
                )
                gas_accumulation = accumulation_coefficient * (
                    gas_saturation_grid[i, j, k] - old_gas_saturation_grid[i, j, k]
                )

                # Inter-cell flux terms — reuse the existing per-face flux function
                # which reads from the mobility/Pc grids we pass in (evaluated at
                # the current Newton iterate).
                net_water_flux = 0.0
                net_gas_flux = 0.0

                # X-direction: East neighbor (i+1, j, k)
                flow_length_x = cell_size_x
                east_neighbour_thickness = thickness_grid[i + 1, j, k]
                harmonic_thickness = compute_harmonic_mean(
                    cell_thickness, east_neighbour_thickness
                )
                east_flow_area = cell_size_y * harmonic_thickness
                water_flux, _, gas_flux = compute_phase_fluxes_from_neighbour(
                    cell_indices=(i, j, k),
                    neighbour_indices=(i + 1, j, k),
                    flow_area=east_flow_area,
                    flow_length=flow_length_x,
                    oil_pressure_grid=oil_pressure_grid,
                    water_mobility_grid=water_mobility_grid_x,
                    oil_mobility_grid=oil_mobility_grid_x,
                    gas_mobility_grid=gas_mobility_grid_x,
                    oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                    gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                    oil_density_grid=oil_density_grid,
                    water_density_grid=water_density_grid,
                    gas_density_grid=gas_density_grid,
                    elevation_grid=elevation_grid,
                    gravitational_constant=gravitational_constant,
                )
                net_water_flux += water_flux
                net_gas_flux += gas_flux

                # X-direction: West neighbor (i-1, j, k)
                west_neighbour_thickness = thickness_grid[i - 1, j, k]
                harmonic_thickness = compute_harmonic_mean(
                    cell_thickness, west_neighbour_thickness
                )
                west_flow_area = cell_size_y * harmonic_thickness
                water_flux, _, gas_flux = compute_phase_fluxes_from_neighbour(
                    cell_indices=(i, j, k),
                    neighbour_indices=(i - 1, j, k),
                    flow_area=west_flow_area,
                    flow_length=flow_length_x,
                    oil_pressure_grid=oil_pressure_grid,
                    water_mobility_grid=water_mobility_grid_x,
                    oil_mobility_grid=oil_mobility_grid_x,
                    gas_mobility_grid=gas_mobility_grid_x,
                    oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                    gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                    oil_density_grid=oil_density_grid,
                    water_density_grid=water_density_grid,
                    gas_density_grid=gas_density_grid,
                    elevation_grid=elevation_grid,
                    gravitational_constant=gravitational_constant,
                )
                net_water_flux += water_flux
                net_gas_flux += gas_flux

                # Y-direction: North neighbor (i, j-1, k)
                flow_length_y = cell_size_y
                north_neighbour_thickness = thickness_grid[i, j - 1, k]
                harmonic_thickness = compute_harmonic_mean(
                    cell_thickness, north_neighbour_thickness
                )
                north_flow_area = cell_size_x * harmonic_thickness
                water_flux, _, gas_flux = compute_phase_fluxes_from_neighbour(
                    cell_indices=(i, j, k),
                    neighbour_indices=(i, j - 1, k),
                    flow_area=north_flow_area,
                    flow_length=flow_length_y,
                    oil_pressure_grid=oil_pressure_grid,
                    water_mobility_grid=water_mobility_grid_y,
                    oil_mobility_grid=oil_mobility_grid_y,
                    gas_mobility_grid=gas_mobility_grid_y,
                    oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                    gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                    oil_density_grid=oil_density_grid,
                    water_density_grid=water_density_grid,
                    gas_density_grid=gas_density_grid,
                    elevation_grid=elevation_grid,
                    gravitational_constant=gravitational_constant,
                )
                net_water_flux += water_flux
                net_gas_flux += gas_flux

                # Y-direction: South neighbor (i, j+1, k)
                south_neighbour_thickness = thickness_grid[i, j + 1, k]
                harmonic_thickness = compute_harmonic_mean(
                    cell_thickness, south_neighbour_thickness
                )
                south_flow_area = cell_size_x * harmonic_thickness
                water_flux, _, gas_flux = compute_phase_fluxes_from_neighbour(
                    cell_indices=(i, j, k),
                    neighbour_indices=(i, j + 1, k),
                    flow_area=south_flow_area,
                    flow_length=flow_length_y,
                    oil_pressure_grid=oil_pressure_grid,
                    water_mobility_grid=water_mobility_grid_y,
                    oil_mobility_grid=oil_mobility_grid_y,
                    gas_mobility_grid=gas_mobility_grid_y,
                    oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                    gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                    oil_density_grid=oil_density_grid,
                    water_density_grid=water_density_grid,
                    gas_density_grid=gas_density_grid,
                    elevation_grid=elevation_grid,
                    gravitational_constant=gravitational_constant,
                )
                net_water_flux += water_flux
                net_gas_flux += gas_flux

                # Z-direction: Top neighbor (i, j, k-1)
                flow_area_z = cell_size_x * cell_size_y
                top_neighbour_thickness = thickness_grid[i, j, k - 1]
                harmonic_thickness = compute_harmonic_mean(
                    cell_thickness, top_neighbour_thickness
                )
                top_flow_length = harmonic_thickness
                water_flux, _, gas_flux = compute_phase_fluxes_from_neighbour(
                    cell_indices=(i, j, k),
                    neighbour_indices=(i, j, k - 1),
                    flow_area=flow_area_z,
                    flow_length=top_flow_length,
                    oil_pressure_grid=oil_pressure_grid,
                    water_mobility_grid=water_mobility_grid_z,
                    oil_mobility_grid=oil_mobility_grid_z,
                    gas_mobility_grid=gas_mobility_grid_z,
                    oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                    gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                    oil_density_grid=oil_density_grid,
                    water_density_grid=water_density_grid,
                    gas_density_grid=gas_density_grid,
                    elevation_grid=elevation_grid,
                    gravitational_constant=gravitational_constant,
                )
                net_water_flux += water_flux
                net_gas_flux += gas_flux

                # Z-direction: Bottom neighbor (i, j, k+1)
                bottom_neighbour_thickness = thickness_grid[i, j, k + 1]
                harmonic_thickness = compute_harmonic_mean(
                    cell_thickness, bottom_neighbour_thickness
                )
                bottom_flow_length = harmonic_thickness
                water_flux, _, gas_flux = compute_phase_fluxes_from_neighbour(
                    cell_indices=(i, j, k),
                    neighbour_indices=(i, j, k + 1),
                    flow_area=flow_area_z,
                    flow_length=bottom_flow_length,
                    oil_pressure_grid=oil_pressure_grid,
                    water_mobility_grid=water_mobility_grid_z,
                    oil_mobility_grid=oil_mobility_grid_z,
                    gas_mobility_grid=gas_mobility_grid_z,
                    oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                    gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                    oil_density_grid=oil_density_grid,
                    water_density_grid=water_density_grid,
                    gas_density_grid=gas_density_grid,
                    elevation_grid=elevation_grid,
                    gravitational_constant=gravitational_constant,
                )
                net_water_flux += water_flux
                net_gas_flux += gas_flux

                # Well source/sink contribution (already in ft³/day)
                water_well_source = net_water_well_rate_grid[i, j, k]
                gas_well_source = net_gas_well_rate_grid[i, j, k]

                # PVT volume correction: ΔSα_pvt = -Sα * (cα + cf) * ΔP
                # Expressed as a rate (divided by dt and multiplied by pore volume)
                # to match the accumulation and flux units.
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

                # Residual = Accumulation - Fluxes - Wells - PVT correction
                idx = to_1D_index_interior_only(
                    i, j, k, cell_count_x, cell_count_y, cell_count_z
                )
                residual_water[idx] = (
                    water_accumulation
                    - net_water_flux
                    - water_well_source
                    - water_pvt_correction
                )
                residual_gas[idx] = (
                    gas_accumulation
                    - net_gas_flux
                    - gas_well_source
                    - gas_pvt_correction
                )

    return residual_water, residual_gas


def recompute_saturation_dependent_properties(
    water_saturation_grid: ThreeDimensionalGrid,
    oil_saturation_grid: ThreeDimensionalGrid,
    gas_saturation_grid: ThreeDimensionalGrid,
    rock_properties: RockProperties[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
    config: Config,
) -> typing.Tuple[
    RelPermGrids[ThreeDimensions],
    RelativeMobilityGrids[ThreeDimensions],
    CapillaryPressureGrids[ThreeDimensions],
    typing.Tuple[
        typing.Tuple[ThreeDimensionalGrid, ThreeDimensionalGrid, ThreeDimensionalGrid],
        typing.Tuple[ThreeDimensionalGrid, ThreeDimensionalGrid, ThreeDimensionalGrid],
        typing.Tuple[ThreeDimensionalGrid, ThreeDimensionalGrid, ThreeDimensionalGrid],
    ],
]:
    """
    Recompute all saturation-dependent quantities at fixed pressure.

    Called at each Newton iteration. Returns updated relative permeabilities,
    relative mobilities, capillary pressures, and directional mobility grids.
    """
    # Clamp to valid range before relperm evaluation — the Newton line search
    # can produce trial states with tiny out-of-bounds values due to floating
    # point residuals in project_to_feasible.
    water_saturation_clamped_grid = np.clip(water_saturation_grid, 0.0, 1.0)
    gas_saturation_clamped_grid = np.clip(gas_saturation_grid, 0.0, 1.0)
    oil_saturation_clamped_grid = np.clip(oil_saturation_grid, 0.0, 1.0)

    # Re-normalize so they sum to exactly 1 at each cell
    total_saturation_grid = (
        water_saturation_clamped_grid
        + oil_saturation_clamped_grid
        + gas_saturation_clamped_grid
    )
    # Avoid division by zero in degenerate cells
    total_saturation_grid = np.where(
        total_saturation_grid > 0.0, total_saturation_grid, 1.0
    )
    water_saturation_clamped_grid = (
        water_saturation_clamped_grid / total_saturation_grid
    )
    oil_saturation_clamped_grid = oil_saturation_clamped_grid / total_saturation_grid
    gas_saturation_clamped_grid = gas_saturation_clamped_grid / total_saturation_grid

    relperm_grids, relative_mobility_grids, capillary_pressure_grids = (
        build_rock_fluid_properties_grids(
            water_saturation_grid=water_saturation_clamped_grid,  # type: ignore
            oil_saturation_grid=oil_saturation_clamped_grid,  # type: ignore
            gas_saturation_grid=gas_saturation_clamped_grid,  # type: ignore
            irreducible_water_saturation_grid=rock_properties.irreducible_water_saturation_grid,
            residual_oil_saturation_water_grid=rock_properties.residual_oil_saturation_water_grid,
            residual_oil_saturation_gas_grid=rock_properties.residual_oil_saturation_gas_grid,
            residual_gas_saturation_grid=rock_properties.residual_gas_saturation_grid,
            water_viscosity_grid=fluid_properties.water_viscosity_grid,
            oil_viscosity_grid=fluid_properties.oil_effective_viscosity_grid,
            gas_viscosity_grid=fluid_properties.gas_viscosity_grid,
            relative_permeability_table=config.rock_fluid_tables.relative_permeability_table,
            capillary_pressure_table=config.rock_fluid_tables.capillary_pressure_table,
            disable_capillary_effects=config.disable_capillary_effects,
            capillary_strength_factor=config.capillary_strength_factor,
            relative_mobility_range=config.relative_mobility_range,
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
    return (
        relperm_grids,
        relative_mobility_grids,
        capillary_pressure_grids,
        mobility_grids,
    )


def evaluate_residual(
    water_saturation_grid: ThreeDimensionalGrid,
    oil_saturation_grid: ThreeDimensionalGrid,
    gas_saturation_grid: ThreeDimensionalGrid,
    old_water_saturation_grid: ThreeDimensionalGrid,
    old_gas_saturation_grid: ThreeDimensionalGrid,
    oil_pressure_grid: ThreeDimensionalGrid,
    pressure_change_grid: ThreeDimensionalGrid,
    rock_properties: RockProperties[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
    wells: Wells[ThreeDimensions],
    config: Config,
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
    thickness_grid: ThreeDimensionalGrid,
    cell_size_x: float,
    cell_size_y: float,
    elevation_grid: ThreeDimensionalGrid,
    porosity_grid: ThreeDimensionalGrid,
    time: float,
    time_step_in_days: float,
    gravitational_constant: float,
    water_compressibility_grid: ThreeDimensionalGrid,
    gas_compressibility_grid: ThreeDimensionalGrid,
    rock_compressibility: float,
    boundary_conditions: BoundaryConditions[ThreeDimensions],
    injection_grid: typing.Optional[
        SupportsSetItem[ThreeDimensions, typing.Tuple[float, float, float]]
    ],
    production_grid: typing.Optional[
        SupportsSetItem[ThreeDimensions, typing.Tuple[float, float, float]]
    ],
    pad_width: int,
    dtype: np.typing.DTypeLike,
) -> typing.Tuple[np.typing.NDArray, np.typing.NDArray]:
    """
    Full residual evaluation: recomputes saturation-dependent properties,
    well rates, and then calls the Numba residual kernel.

    Returns (residual_water, residual_gas) as 1D arrays.
    """
    # Recompute kr, mobility, Pc at current saturation iterate
    (
        _,
        relative_mobility_grids,
        capillary_pressure_grids,
        mobility_grids,
    ) = recompute_saturation_dependent_properties(
        water_saturation_grid=water_saturation_grid,
        oil_saturation_grid=oil_saturation_grid,
        gas_saturation_grid=gas_saturation_grid,
        rock_properties=rock_properties,
        fluid_properties=fluid_properties,
        config=config,
    )

    (
        water_relative_mobility_grid,
        oil_relative_mobility_grid,
        gas_relative_mobility_grid,
    ) = relative_mobility_grids
    oil_water_capillary_pressure_grid, gas_oil_capillary_pressure_grid = (
        capillary_pressure_grids
    )

    (
        (water_mobility_grid_x, oil_mobility_grid_x, gas_mobility_grid_x),
        (water_mobility_grid_y, oil_mobility_grid_y, gas_mobility_grid_y),
        (water_mobility_grid_z, oil_mobility_grid_z, gas_mobility_grid_z),
    ) = mobility_grids

    # Recompute well rates at current iteration mobilities
    net_water_well_rate_grid, _, net_gas_well_rate_grid = compute_well_rate_grids(
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
        wells=wells,
        oil_pressure_grid=oil_pressure_grid,
        temperature_grid=fluid_properties.temperature_grid,
        absolute_permeability=rock_properties.absolute_permeability,
        water_relative_mobility_grid=water_relative_mobility_grid,
        oil_relative_mobility_grid=oil_relative_mobility_grid,
        gas_relative_mobility_grid=gas_relative_mobility_grid,
        water_compressibility_grid=water_compressibility_grid,
        oil_compressibility_grid=fluid_properties.oil_compressibility_grid,
        gas_compressibility_grid=gas_compressibility_grid,
        fluid_properties=fluid_properties,
        thickness_grid=thickness_grid,
        cell_size_x=cell_size_x,
        cell_size_y=cell_size_y,
        time=time,
        config=config,
        boundary_conditions=boundary_conditions,
        pad_width=pad_width,
        injection_grid=injection_grid,
        production_grid=production_grid,
        dtype=dtype,
    )

    oil_density_grid = fluid_properties.oil_effective_density_grid
    water_density_grid = fluid_properties.water_density_grid
    gas_density_grid = fluid_properties.gas_density_grid

    return compute_saturation_residual(
        oil_pressure_grid=oil_pressure_grid,
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
        thickness_grid=thickness_grid,
        cell_size_x=cell_size_x,
        cell_size_y=cell_size_y,
        water_mobility_grid_x=water_mobility_grid_x,
        water_mobility_grid_y=water_mobility_grid_y,
        water_mobility_grid_z=water_mobility_grid_z,
        oil_mobility_grid_x=oil_mobility_grid_x,
        oil_mobility_grid_y=oil_mobility_grid_y,
        oil_mobility_grid_z=oil_mobility_grid_z,
        gas_mobility_grid_x=gas_mobility_grid_x,
        gas_mobility_grid_y=gas_mobility_grid_y,
        gas_mobility_grid_z=gas_mobility_grid_z,
        oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
        gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
        oil_density_grid=oil_density_grid,
        water_density_grid=water_density_grid,
        gas_density_grid=gas_density_grid,
        elevation_grid=elevation_grid,
        gravitational_constant=gravitational_constant,
        water_saturation_grid=water_saturation_grid,
        gas_saturation_grid=gas_saturation_grid,
        old_water_saturation_grid=old_water_saturation_grid,
        old_gas_saturation_grid=old_gas_saturation_grid,
        porosity_grid=porosity_grid,
        time_step_in_days=time_step_in_days,
        net_water_well_rate_grid=net_water_well_rate_grid,
        net_gas_well_rate_grid=net_gas_well_rate_grid,
        pressure_change_grid=pressure_change_grid,
        water_compressibility_grid=water_compressibility_grid,
        gas_compressibility_grid=gas_compressibility_grid,
        rock_compressibility=rock_compressibility,
    )


def assemble_numerical_jacobian(
    saturation_vector: np.typing.NDArray,
    residual_base: np.typing.NDArray,
    interior_cell_count: int,
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
    rock_properties: RockProperties[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
    wells: Wells[ThreeDimensions],
    config: Config,
    thickness_grid: ThreeDimensionalGrid,
    cell_size_x: float,
    cell_size_y: float,
    elevation_grid: ThreeDimensionalGrid,
    porosity_grid: ThreeDimensionalGrid,
    time: float,
    time_step_in_days: float,
    gravitational_constant: float,
    water_compressibility_grid: ThreeDimensionalGrid,
    gas_compressibility_grid: ThreeDimensionalGrid,
    rock_compressibility: float,
    boundary_conditions: BoundaryConditions[ThreeDimensions],
    injection_grid: typing.Optional[
        SupportsSetItem[ThreeDimensions, typing.Tuple[float, float, float]]
    ],
    production_grid: typing.Optional[
        SupportsSetItem[ThreeDimensions, typing.Tuple[float, float, float]]
    ],
    pad_width: int,
    dtype: np.typing.DTypeLike,
) -> csr_matrix:
    """
    Assemble the saturation Jacobian matrix using column-wise forward finite differences.

    Exploits the known 7-point stencil sparsity pattern of the residual: perturbing
    Sw or Sg at cell `(i, j, k)` can only affect the residuals of that cell and its
    six immediate face-neighbours. Only those rows are evaluated, so the cost scales
    as `O(N)` residual evaluations rather than `O(N²)`.

    All arithmetic is performed in `float64` regardless of the working precision
    (`dtype`) to avoid round-off artifacts in the finite-difference quotients.
    Perturbation is applied **in-place** on the shared float64 grids and immediately
    restored after each column evaluation, eliminating the `6N` full-grid copies
    that a copy-based approach would require.

    The perturbation sign is chosen adaptively:

    * **Forward** (`+epsilon`) by default.
    * **Backward** (`-epsilon`) when a forward step would make `So = 1 - Sw - Sg`
      negative (i.e. the cell is near the `So = 0` boundary).
    * A further safety clamp is applied when even the backward step would drive
      the perturbed variable below zero (simplex vertex case).

    :param saturation_vector: Current saturation vector of length `2N`, interleaved
        as `[Sw_0, Sg_0, Sw_1, Sg_1, ..., Sw_{N-1}, Sg_{N-1}]`.
    :param residual_base: Pre-computed residual vector at the current iterate,
        interleaved as `[Rw_0, Rg_0, ..., Rw_{N-1}, Rg_{N-1}]`. Length `2N`.
        Must be evaluated at the same saturation state as `saturation_vector`.
    :param interior_cell_count: Number of interior (non-ghost) cells `N`.
    :param cell_count_x: Total number of cells in the x-direction including ghost cells.
    :param cell_count_y: Total number of cells in the y-direction including ghost cells.
    :param cell_count_z: Total number of cells in the z-direction including ghost cells.
    :param water_saturation_grid: 3D grid of current water saturations (fraction).
        Modified in-place during Jacobian assembly and fully restored on exit.
    :param oil_saturation_grid: 3D grid of current oil saturations (fraction).
        Modified in-place during Jacobian assembly and fully restored on exit.
    :param gas_saturation_grid: 3D grid of current gas saturations (fraction).
        Modified in-place during Jacobian assembly and fully restored on exit.
    :param old_water_saturation_grid: 3D grid of water saturations at the start of
        the time step (fraction). Used in the residual accumulation term.
    :param old_gas_saturation_grid: 3D grid of gas saturations at the start of
        the time step (fraction). Used in the residual accumulation term.
    :param oil_pressure_grid: 3D grid of oil-phase pressures fixed from the implicit
        pressure solve (psi). Not modified.
    :param pressure_change_grid: 3D grid of `P_new - P_old` (psi) used for the
        PVT volume correction term in the residual.
    :param rock_properties: Rock properties container (permeability, porosity,
        residual saturations, compressibility).
    :param fluid_properties: Fluid properties container at the current pressure level
        (viscosities, FVFs, compressibilities, densities).
    :param wells: Well definitions including perforations and controls.
    :param config: Simulation configuration (relperm tables, solver settings, etc.).
    :param thickness_grid: 3D grid of cell thicknesses (ft).
    :param cell_size_x: Cell size in the x-direction (ft).
    :param cell_size_y: Cell size in the y-direction (ft).
    :param elevation_grid: 3D grid of cell centre elevations (ft).
    :param porosity_grid: 3D grid of cell porosities (fraction).
    :param time: Elapsed simulation time at the current time step (s).
    :param time_step_in_days: Current time step size (days).
    :param gravitational_constant: `g / g_c` conversion factor (lbf/lbm),
        equal to 1.0 in consistent imperial units on Earth.
    :param water_compressibility_grid: 3D grid of water compressibilities (1/psi).
    :param gas_compressibility_grid: 3D grid of gas compressibilities (1/psi).
    :param rock_compressibility: Scalar pore (rock) compressibility (1/psi).
    :param boundary_conditions: Pressure and saturation boundary conditions.
    :param injection_grid: Optional grid to accumulate injection rates by phase
        (oil, water, gas) in ft³/day. Written on each residual evaluation so the
        final state reflects the last perturbed column; callers that need
        physically meaningful rates should re-evaluate after Jacobian assembly.
        Pass `None` to skip tracking.
    :param production_grid: Optional grid to accumulate production rates by phase
        (oil, water, gas) in ft³/day. Same semantics as `injection_grid`.
        Pass `None` to skip tracking.
    :param pad_width: Number of ghost-cell layers surrounding the active grid.
        Well coordinates are offset by this amount internally.
    :param dtype: Working floating-point dtype (`np.float32` or `np.float64`).
        The finite-difference arithmetic is always performed in `float64`
        regardless of this value.
    :return: Sparse Jacobian matrix of shape `(2N, 2N)` in CSR format. Entry
        `J[r, c]` approximates `∂R_r / ∂S_c` at the current Newton iterate.
    """
    system_size = 2 * interior_cell_count
    rows_list = []
    cols_list = []
    vals_list = []

    # Perturbation size: sqrt(machine_eps) gives the optimal balance between
    # truncation error (too large) and cancellation error (too small) for
    # forward finite differences. The saturation scale is 1.0 since Sw, Sg ∈ [0, 1].
    machine_eps = np.finfo(dtype).eps  # type: ignore
    base_epsilon = float(np.sqrt(machine_eps))

    # Promote to float64 once. All perturb/restore operations work on these
    # grids in-place, so no copies are needed inside the column loop.
    water_saturation_grid_f64 = water_saturation_grid.astype(np.float64, copy=True)
    oil_saturation_grid_f64 = oil_saturation_grid.astype(np.float64, copy=True)
    gas_saturation_grid_f64 = gas_saturation_grid.astype(np.float64, copy=True)

    # Shared kwargs for every `evaluate_residual` call
    residual_kwargs = dict(  # noqa
        old_water_saturation_grid=old_water_saturation_grid,
        old_gas_saturation_grid=old_gas_saturation_grid,
        oil_pressure_grid=oil_pressure_grid,
        pressure_change_grid=pressure_change_grid,
        rock_properties=rock_properties,
        fluid_properties=fluid_properties,
        wells=wells,
        config=config,
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
        thickness_grid=thickness_grid,
        cell_size_x=cell_size_x,
        cell_size_y=cell_size_y,
        elevation_grid=elevation_grid,
        porosity_grid=porosity_grid,
        time=time,
        time_step_in_days=time_step_in_days,
        gravitational_constant=gravitational_constant,
        water_compressibility_grid=water_compressibility_grid,
        gas_compressibility_grid=gas_compressibility_grid,
        rock_compressibility=rock_compressibility,
        boundary_conditions=boundary_conditions,
        injection_grid=injection_grid,
        production_grid=production_grid,
        pad_width=pad_width,
        dtype=dtype,
    )

    for cell_1d_idx in range(interior_cell_count):
        i, j, k = from_1D_index_interior_only(
            cell_1d_idx, cell_count_x, cell_count_y, cell_count_z
        )

        # Residual of cell (i,j,k) and its six face-neighbours are the only rows
        # that can be non-zero in column `col` due to the 7-point stencil.
        affected_cell_indices = [cell_1d_idx]
        for ni, nj, nk in (
            (i + 1, j, k),
            (i - 1, j, k),
            (i, j + 1, k),
            (i, j - 1, k),
            (i, j, k + 1),
            (i, j, k - 1),
        ):
            neighbor_1d = to_1D_index_interior_only(
                ni, nj, nk, cell_count_x, cell_count_y, cell_count_z
            )
            if neighbor_1d >= 0:
                affected_cell_indices.append(neighbor_1d)

        # Current saturations at the cell being perturbed — read once per cell.
        sw_cell = float(water_saturation_grid_f64[i, j, k])
        sg_cell = float(gas_saturation_grid_f64[i, j, k])
        so_cell = 1.0 - sw_cell - sg_cell

        for var_offset in range(2):
            col = 2 * cell_1d_idx + var_offset
            s_j = saturation_vector[col]

            # Perturbation magnitude — scaled to the saturation range [0, 1].
            epsilon = base_epsilon * max(abs(s_j), 1.0)

            # Direction: switch to backward difference when a forward step
            # would push So below zero (cell near the So = 0 boundary).
            if epsilon > so_cell:
                epsilon = -epsilon
                # Further safety: backward step must not drive the variable negative.
                if s_j + epsilon < 0.0:
                    epsilon = (-s_j * 0.5) if s_j > 1e-15 else (base_epsilon * 0.01)

            # In-place perturbation (no grid copies)
            # Save the three scalar originals for the cell being perturbed.
            sw_orig = water_saturation_grid_f64[i, j, k]
            sg_orig = gas_saturation_grid_f64[i, j, k]
            so_orig = oil_saturation_grid_f64[i, j, k]

            if var_offset == 0:
                water_saturation_grid_f64[i, j, k] = sw_orig + epsilon
            else:
                gas_saturation_grid_f64[i, j, k] = sg_orig + epsilon

            # Recompute So to maintain Sw + So + Sg = 1 at the perturbed cell.
            oil_saturation_grid_f64[i, j, k] = max(
                0.0,
                1.0
                - water_saturation_grid_f64[i, j, k]
                - gas_saturation_grid_f64[i, j, k],
            )

            # Evaluate residual with the shared float64 grids (now perturbed at
            # exactly one cell). All other cells are unchanged.
            residual_water_perturbed, residual_gas_perturbed = evaluate_residual(
                water_saturation_grid=water_saturation_grid_f64,
                oil_saturation_grid=oil_saturation_grid_f64,
                gas_saturation_grid=gas_saturation_grid_f64,
                **residual_kwargs,  # type: ignore[arg-type]
            )
            residual_perturbed = interleave_residuals(
                residual_water=residual_water_perturbed,
                residual_gas=residual_gas_perturbed,
            )

            # Restore in-place (three scalar writes)
            water_saturation_grid_f64[i, j, k] = sw_orig
            gas_saturation_grid_f64[i, j, k] = sg_orig
            oil_saturation_grid_f64[i, j, k] = so_orig

            # Finite-difference derivative for each affected row.
            for affected_idx in affected_cell_indices:
                row_water = 2 * affected_idx
                dR_water = (
                    residual_perturbed[row_water] - residual_base[row_water]
                ) / epsilon
                if abs(dR_water) > 1e-30:
                    rows_list.append(row_water)
                    cols_list.append(col)
                    vals_list.append(dR_water)

                row_gas = 2 * affected_idx + 1
                dR_gas = (
                    residual_perturbed[row_gas] - residual_base[row_gas]
                ) / epsilon
                if abs(dR_gas) > 1e-30:
                    rows_list.append(row_gas)
                    cols_list.append(col)
                    vals_list.append(dR_gas)

    jacobian_coo = coo_matrix(
        (
            np.array(vals_list, dtype=np.float64),
            (
                np.array(rows_list, dtype=np.int32),
                np.array(cols_list, dtype=np.int32),
            ),
        ),
        shape=(system_size, system_size),
    )
    return jacobian_coo.tocsr()


def solve_implicit_saturation(
    oil_pressure_grid: ThreeDimensionalGrid,
    pressure_change_grid: ThreeDimensionalGrid,
    old_water_saturation_grid: ThreeDimensionalGrid,
    old_oil_saturation_grid: ThreeDimensionalGrid,
    old_gas_saturation_grid: ThreeDimensionalGrid,
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
    thickness_grid: ThreeDimensionalGrid,
    cell_size_x: float,
    cell_size_y: float,
    elevation_grid: ThreeDimensionalGrid,
    porosity_grid: ThreeDimensionalGrid,
    time_step_size: float,
    time: float,
    rock_properties: RockProperties[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
    wells: Wells[ThreeDimensions],
    config: Config,
    boundary_conditions: BoundaryConditions[ThreeDimensions],
    injection_grid: typing.Optional[
        SupportsSetItem[ThreeDimensions, typing.Tuple[float, float, float]]
    ],
    production_grid: typing.Optional[
        SupportsSetItem[ThreeDimensions, typing.Tuple[float, float, float]]
    ],
    water_compressibility_grid: ThreeDimensionalGrid,
    gas_compressibility_grid: ThreeDimensionalGrid,
    rock_compressibility: float,
    pad_width: int = 1,
    max_newton_iterations: int = 12,
    newton_tolerance: float = 1e-6,
    line_search_max_cuts: int = 4,
    max_saturation_step: float = 0.05,
    saturation_convergence_tolerance: float = 1e-4,
) -> EvolutionResult[ImplicitSaturationSolution, typing.List[NewtonConvergenceInfo]]:
    """
    Solve the implicit saturation equations using Newton-Raphson iteration
    with backtracking line search.

    Pressure is fixed from the implicit pressure solve. The Newton loop
    iterates on saturation until either:
      1. The relative residual norm drops below `newton_tolerance`, OR
      2. The max saturation change per iteration drops below
         `saturation_convergence_tolerance` and the relative residual
         is below 1e-3 (i.e. the solution is effectively converged but
         the upwind scheme discontinuity prevents further residual reduction).

    The Newton update is damped so that no single cell changes saturation
    by more than `max_saturation_step` per iteration. This prevents
    upwind direction flipping that causes limit-cycle oscillation.
    """
    time_step_in_days = time_step_size * c.DAYS_PER_SECOND
    dtype = get_dtype()
    interior_cell_count = (cell_count_x - 2) * (cell_count_y - 2) * (cell_count_z - 2)

    gravitational_constant = (
        c.ACCELERATION_DUE_TO_GRAVITY_FEET_PER_SECONDS_SQUARE
        / c.GRAVITATIONAL_CONSTANT_LBM_FT_PER_LBF_S2
    )

    # Initial guess: old-time saturations
    water_saturation_grid = old_water_saturation_grid.copy()
    oil_saturation_grid = old_oil_saturation_grid.copy()
    gas_saturation_grid = old_gas_saturation_grid.copy()

    saturation_vector = saturations_to_vector(
        water_saturation_grid=water_saturation_grid,
        gas_saturation_grid=gas_saturation_grid,
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
    )

    convergence_history: typing.List[NewtonConvergenceInfo] = []
    initial_residual_norm = 0.0
    converged = False
    final_iteration = 0
    final_residual_norm = 0.0

    # Residual stagnation tracking: detect when Newton iterations stop making
    # progress (e.g., phase appearance discontinuity makes Jacobian ineffective).
    best_residual_norm = float("inf")
    stagnation_count = 0
    stagnation_patience = 3  # consecutive flat iterations before breaking
    stagnation_improvement_threshold = 0.01  # require 1% improvement

    residual_kwargs = dict(  # noqa
        old_water_saturation_grid=old_water_saturation_grid,
        old_gas_saturation_grid=old_gas_saturation_grid,
        oil_pressure_grid=oil_pressure_grid,
        pressure_change_grid=pressure_change_grid,
        rock_properties=rock_properties,
        fluid_properties=fluid_properties,
        wells=wells,
        config=config,
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
        thickness_grid=thickness_grid,
        cell_size_x=cell_size_x,
        cell_size_y=cell_size_y,
        elevation_grid=elevation_grid,
        porosity_grid=porosity_grid,
        time=time,
        time_step_in_days=time_step_in_days,
        gravitational_constant=gravitational_constant,
        water_compressibility_grid=water_compressibility_grid,
        gas_compressibility_grid=gas_compressibility_grid,
        rock_compressibility=rock_compressibility,
        boundary_conditions=boundary_conditions,
        injection_grid=injection_grid,
        production_grid=production_grid,
        pad_width=pad_width,
        dtype=dtype,
    )

    for iteration in range(max_newton_iterations):
        # Evaluate residual at current iterate
        residual_water, residual_gas = evaluate_residual(
            water_saturation_grid=water_saturation_grid,
            oil_saturation_grid=oil_saturation_grid,
            gas_saturation_grid=gas_saturation_grid,
            **residual_kwargs,  # type: ignore
        )
        residual_vector = interleave_residuals(residual_water, residual_gas)
        residual_norm = np.linalg.norm(residual_vector)

        if iteration == 0:
            initial_residual_norm = max(residual_norm, 1e-30)
            logger.info(f"Newton iteration 0: ||R0|| = {initial_residual_norm:.4e}")

        relative_residual_norm = residual_norm / initial_residual_norm

        # Check convergence using two criteria:
        # Relative residual below tolerance, or
        # Saturation changes negligible and residual is reasonable (<1e-3).
        # This second criterion handles the upwind scheme discontinuity
        # which creates an irreducible residual floor.
        residual_converged = relative_residual_norm < newton_tolerance and iteration > 0
        # Use the last recorded saturation update for the combined check
        last_max_ds = (
            convergence_history[-1].max_saturation_update
            if convergence_history
            else float("inf")
        )
        saturation_converged = (
            last_max_ds < saturation_convergence_tolerance
            and relative_residual_norm < 1e-3
            and iteration > 1
        )

        if residual_converged or saturation_converged:
            converged = True
            final_iteration = iteration
            final_residual_norm = residual_norm
            convergence_history.append(
                NewtonConvergenceInfo(
                    iteration=iteration,
                    residual_norm=residual_norm,  # type: ignore
                    relative_residual_norm=relative_residual_norm,  # type: ignore
                    max_saturation_update=0.0,
                    line_search_factor=1.0,
                )
            )
            reason = "residual" if residual_converged else "saturation change"
            logger.info(
                f"Newton converged at iteration {iteration} ({reason}): "
                f"||R||/||R0|| = {relative_residual_norm:.2e}, "
                f"max |dS| = {last_max_ds:.2e}"
            )
            break

        # Assemble Jacobian
        jacobian = assemble_numerical_jacobian(
            saturation_vector=saturation_vector,
            residual_base=residual_vector,
            interior_cell_count=interior_cell_count,
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
            rock_properties=rock_properties,
            fluid_properties=fluid_properties,
            wells=wells,
            config=config,
            thickness_grid=thickness_grid,
            cell_size_x=cell_size_x,
            cell_size_y=cell_size_y,
            elevation_grid=elevation_grid,
            porosity_grid=porosity_grid,
            time=time,
            time_step_in_days=time_step_in_days,
            gravitational_constant=gravitational_constant,
            water_compressibility_grid=water_compressibility_grid,
            gas_compressibility_grid=gas_compressibility_grid,
            rock_compressibility=rock_compressibility,
            boundary_conditions=boundary_conditions,
            injection_grid=injection_grid,
            production_grid=production_grid,
            pad_width=pad_width,
            dtype=dtype,
        )

        # Solve linear system: J * delta_saturation = -R
        negative_residual = -residual_vector
        delta_saturation, _ = solve_linear_system(
            A_csr=jacobian,
            b=negative_residual,
            solver=config.saturation_solver,
            preconditioner=config.saturation_preconditioner,
            rtol=config.saturation_convergence_tolerance,
            max_iterations=config.max_iterations,
            fallback_to_direct=True,
        )

        # Damp the Newton step so no single cell changes saturation by
        # more than max_saturation_step. This prevents large updates that
        # flip upwind directions and cause limit-cycle oscillation.
        max_raw_delta = float(np.max(np.abs(delta_saturation)))
        if max_raw_delta > max_saturation_step:
            damping_factor = max_saturation_step / max_raw_delta
            delta_saturation = delta_saturation * damping_factor
            logger.debug(
                f"Damped Newton step by {damping_factor:.3f} "
                f"(max |delta_S| = {max_raw_delta:.4f} > {max_saturation_step})"
            )

        # Backtracking line search: start with full Newton step, halve on failure.
        # Initialize trial state with full step (alpha=1) so variables are always
        # bound even if line_search_max_cuts is 0.
        line_search_factor = 1.0
        saturation_vector_trial = saturation_vector + delta_saturation
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

        for _ in range(line_search_max_cuts):
            # Evaluate residual at trial point
            residual_water_trial, residual_gas_trial = evaluate_residual(
                water_saturation_grid=water_saturation_grid_trial,
                oil_saturation_grid=oil_saturation_grid_trial,
                gas_saturation_grid=gas_saturation_grid_trial,
                **residual_kwargs,  # type: ignore
            )
            residual_trial = interleave_residuals(
                residual_water=residual_water_trial,
                residual_gas=residual_gas_trial,
            )
            residual_norm_trial = np.linalg.norm(residual_trial)

            if residual_norm_trial < residual_norm:
                break

            # Cut step size in half and recompute trial
            line_search_factor *= 0.5
            saturation_vector_trial = (
                saturation_vector + line_search_factor * delta_saturation
            )
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

        # Update state
        saturation_vector = saturation_vector_trial
        water_saturation_grid = water_saturation_grid_trial
        oil_saturation_grid = oil_saturation_grid_trial
        gas_saturation_grid = gas_saturation_grid_trial

        convergence_history.append(
            NewtonConvergenceInfo(
                iteration=iteration,
                residual_norm=residual_norm,  # type: ignore
                relative_residual_norm=relative_residual_norm,  # type: ignore
                max_saturation_update=max_saturation_update,
                line_search_factor=line_search_factor,
            )
        )

        logger.info(
            f"Newton iteration {iteration}: "
            f"||R|| = {residual_norm:.2e}, "
            f"||R||/||R0|| = {relative_residual_norm:.2e}, "
            f"max |dS| = {max_saturation_update:.2e}, "
            f"alpha = {line_search_factor:.3f}"
        )

        final_iteration = iteration + 1
        final_residual_norm = residual_norm

        # Detect stagnation with two mechanisms.
        # Saturation stagnation: Newton step produces negligible updates.
        # If the residual is acceptable, declare converged; otherwise break.
        if max_saturation_update < 1e-10:
            if relative_residual_norm < 1e-3:
                converged = True
                logger.info(
                    f"Newton converged (saturation stagnation) at iteration {iteration}: "
                    f"max |dS| = {max_saturation_update:.2e}, "
                    f"||R||/||R0|| = {relative_residual_norm:.2e}"
                )
            else:
                logger.warning(
                    f"Newton stagnated (negligible dS) at iteration {iteration}: "
                    f"max |dS| = {max_saturation_update:.2e}, "
                    f"||R||/||R0|| = {relative_residual_norm:.2e}"
                )
            break

        # Check for residual stagnation, i.e. when residual stops improving across iterations.
        # This check catches cases where the Jacobian is ineffective (e.g., phase
        # appearance discontinuity). The solver produces small but non-zero
        # saturation changes that don't reduce the residual, so breaking early
        # avoids wasting iterations and lets the timer cut dt.
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
                logger.info(
                    f"Newton converged (residual plateau) at iteration {iteration}: "
                    f"||R||/||R0|| = {relative_residual_norm:.2e}, "
                    f"no improvement for {stagnation_count} iterations"
                )
            else:
                logger.warning(
                    f"Newton stagnated (residual flat) at iteration {iteration}: "
                    f"||R||/||R0|| = {relative_residual_norm:.2e}, "
                    f"no improvement for {stagnation_count} iterations"
                )
            break

    # Compute saturation changes from old to final
    max_water_saturation_change = float(
        np.max(np.abs(water_saturation_grid - old_water_saturation_grid))
    )
    max_oil_saturation_change = float(
        np.max(np.abs(oil_saturation_grid - old_oil_saturation_grid))
    )
    max_gas_saturation_change = float(
        np.max(np.abs(gas_saturation_grid - old_gas_saturation_grid))
    )

    solution = ImplicitSaturationSolution(
        water_saturation_grid=water_saturation_grid.astype(dtype, copy=False),
        oil_saturation_grid=oil_saturation_grid.astype(dtype, copy=False),
        gas_saturation_grid=gas_saturation_grid.astype(dtype, copy=False),
        newton_iterations=final_iteration,
        final_residual_norm=final_residual_norm,  # type: ignore
        max_water_saturation_change=max_water_saturation_change,
        max_oil_saturation_change=max_oil_saturation_change,
        max_gas_saturation_change=max_gas_saturation_change,
    )
    if converged:
        return EvolutionResult(
            value=solution,
            scheme="implicit",
            success=True,
            message=f"Implicit saturation converged in {final_iteration} Newton iterations.",
            metadata=convergence_history,
        )
    return EvolutionResult(
        value=solution,
        scheme="implicit",
        success=False,
        message=(
            f"Newton did not converge after {max_newton_iterations} iterations. "
            f"Final relative residual: {final_residual_norm / initial_residual_norm:.2e}"
        ),
        metadata=convergence_history,
    )


def evolve_saturation(
    cell_dimension: typing.Tuple[float, float],
    thickness_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    time_step: int,
    time_step_size: float,
    time: float,
    rock_properties: RockProperties[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
    wells: Wells[ThreeDimensions],
    config: Config,
    boundary_conditions: BoundaryConditions[ThreeDimensions],
    injection_grid: typing.Optional[
        SupportsSetItem[ThreeDimensions, typing.Tuple[float, float, float]]
    ],
    production_grid: typing.Optional[
        SupportsSetItem[ThreeDimensions, typing.Tuple[float, float, float]]
    ],
    pressure_change_grid: ThreeDimensionalGrid,
    pad_width: int = 1,
) -> EvolutionResult[ImplicitSaturationSolution, typing.List[NewtonConvergenceInfo]]:
    """
    Solve the implicit saturation equations for a three-phase system.

    :param cell_dimension: (cell_size_x, cell_size_y) in feet.
    :param thickness_grid: 3D grid of cell thicknesses (ft).
    :param elevation_grid: 3D grid of cell elevations (ft).
    :param time_step: Current time step index.
    :param time_step_size: Time step size in seconds.
    :param rock_properties: Rock properties (permeability, porosity, etc.).
    :param fluid_properties: Fluid properties at new pressure level.
    :param wells: Well definitions.
    :param config: Simulation configuration.
    :param pressure_change_grid: P_new - P_old (psi) for PVT volume correction.
    :param pad_width: Ghost cell padding width.
    :return: `EvolutionResult` containing `ImplicitSaturationSolution`.
    """
    cell_size_x, cell_size_y = cell_dimension
    oil_pressure_grid = fluid_properties.pressure_grid
    cell_count_x, cell_count_y, cell_count_z = oil_pressure_grid.shape

    return solve_implicit_saturation(
        oil_pressure_grid=oil_pressure_grid,
        pressure_change_grid=pressure_change_grid,
        old_water_saturation_grid=fluid_properties.water_saturation_grid,
        old_oil_saturation_grid=fluid_properties.oil_saturation_grid,
        old_gas_saturation_grid=fluid_properties.gas_saturation_grid,
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
        thickness_grid=thickness_grid,
        cell_size_x=cell_size_x,
        cell_size_y=cell_size_y,
        elevation_grid=elevation_grid,
        porosity_grid=rock_properties.porosity_grid,
        time_step_size=time_step_size,
        time=time,
        rock_properties=rock_properties,
        fluid_properties=fluid_properties,
        wells=wells,
        config=config,
        boundary_conditions=boundary_conditions,
        injection_grid=injection_grid,
        production_grid=production_grid,
        water_compressibility_grid=fluid_properties.water_compressibility_grid,
        gas_compressibility_grid=fluid_properties.gas_compressibility_grid,
        rock_compressibility=rock_properties.compressibility,
        pad_width=pad_width,
        max_newton_iterations=config.max_newton_iterations,
        newton_tolerance=config.newton_tolerance,
        line_search_max_cuts=config.line_search_max_cuts,
        max_saturation_step=config.max_saturation_step,
        saturation_convergence_tolerance=config.saturation_convergence_tolerance,
    )
