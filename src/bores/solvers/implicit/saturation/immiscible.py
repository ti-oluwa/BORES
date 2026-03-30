import logging
import typing

import attrs
import numba
import numpy as np
import numpy.typing as npt
import scipy.sparse.linalg as spla
from scipy.sparse import coo_matrix, csr_matrix

from bores._precision import get_dtype
from bores.boundary_conditions import BoundaryConditions
from bores.config import Config
from bores.constants import c
from bores.correlations.core import compute_harmonic_mean
from bores.datastructures import PhaseTensorsProxy
from bores.grids.base import CapillaryPressureGrids, RelativeMobilityGrids
from bores.grids.boundary_conditions import apply_saturation_boundary_conditions
from bores.grids.rock_fluid import build_rock_fluid_properties_grids
from bores.models import FluidProperties, RockProperties
from bores.solvers.base import (
    EvolutionResult,
    compute_mobility_grids,
    from_1D_index_interior_only,
    solve_linear_system,
    to_1D_index_interior_only,
)
from bores.solvers.explicit.saturation.immiscible import (
    compute_fluxes_from_neighbour,
    compute_well_rate_grids,
)
from bores.tables.rock_fluid import RockFluidTables
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
def saturation_grids_to_vector(
    water_saturation_grid: ThreeDimensionalGrid,
    gas_saturation_grid: ThreeDimensionalGrid,
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
) -> npt.NDArray:
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
    saturation_vector: npt.NDArray,
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
        water_saturation = saturation_vector[2 * idx]
        gas_saturation = saturation_vector[2 * idx + 1]
        water_saturation_grid[i, j, k] = water_saturation
        gas_saturation_grid[i, j, k] = gas_saturation
        oil_saturation_grid[i, j, k] = max(0.0, 1.0 - water_saturation - gas_saturation)


@numba.njit(cache=True)
def project_to_feasible(saturation_vector: npt.NDArray) -> npt.NDArray:
    """
    Project saturation vector onto the feasible set:

        0 <= Sw, 0 <= Sg, and Sw + Sg <= 1.

    Uses proportional scaling if Sw + Sg > 1.
    """
    interior_count = len(saturation_vector) // 2
    for idx in range(interior_count):
        water_saturation = max(0.0, saturation_vector[2 * idx])
        gas_saturation = max(0.0, saturation_vector[2 * idx + 1])
        total = water_saturation + gas_saturation
        if total > 1.0:
            water_saturation = water_saturation / total
            gas_saturation = gas_saturation / total
        saturation_vector[2 * idx] = water_saturation
        saturation_vector[2 * idx + 1] = gas_saturation
    return saturation_vector


@numba.njit(cache=True)
def interleave_residuals(
    water_residual: npt.NDArray,
    gas_residual: npt.NDArray,
) -> npt.NDArray:
    """
    Interleave water and gas residual arrays into a single vector.

    Layout: [R_w_0, R_g_0, R_w_1, R_g_1, ..., R_w_{N-1}, R_g_{N-1}]
    """
    interior_count = len(water_residual)
    result = np.empty(2 * interior_count)
    for idx in range(interior_count):
        result[2 * idx] = water_residual[idx]
        result[2 * idx + 1] = gas_residual[idx]
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
) -> typing.Tuple[npt.NDArray, npt.NDArray]:
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

    Returns two 1D arrays: (water_residual, gas_residual), each of
    length interior_cell_count.
    """
    interior_count = (cell_count_x - 2) * (cell_count_y - 2) * (cell_count_z - 2)
    water_residual = np.zeros(interior_count)
    gas_residual = np.zeros(interior_count)

    for i in numba.prange(1, cell_count_x - 1):  # type: ignore
        for j in range(1, cell_count_y - 1):
            for k in range(1, cell_count_z - 1):
                cell_thickness = thickness_grid[i, j, k]
                cell_total_volume = cell_size_x * cell_size_y * cell_thickness
                cell_porosity = porosity_grid[i, j, k]
                cell_pore_volume = cell_total_volume * cell_porosity

                accumulation_coefficient = cell_pore_volume / time_step_in_days
                water_accumulation = accumulation_coefficient * (
                    water_saturation_grid[i, j, k] - old_water_saturation_grid[i, j, k]
                )
                gas_accumulation = accumulation_coefficient * (
                    gas_saturation_grid[i, j, k] - old_gas_saturation_grid[i, j, k]
                )

                net_water_flux = 0.0
                net_gas_flux = 0.0

                flow_length_x = cell_size_x
                east_neighbour_thickness = thickness_grid[i + 1, j, k]
                harmonic_thickness = compute_harmonic_mean(
                    cell_thickness, east_neighbour_thickness
                )
                east_flow_area = cell_size_y * harmonic_thickness
                water_flux, _, gas_flux = compute_fluxes_from_neighbour(
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

                west_neighbour_thickness = thickness_grid[i - 1, j, k]
                harmonic_thickness = compute_harmonic_mean(
                    cell_thickness, west_neighbour_thickness
                )
                west_flow_area = cell_size_y * harmonic_thickness
                water_flux, _, gas_flux = compute_fluxes_from_neighbour(
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

                flow_length_y = cell_size_y
                north_neighbour_thickness = thickness_grid[i, j - 1, k]
                harmonic_thickness = compute_harmonic_mean(
                    cell_thickness, north_neighbour_thickness
                )
                north_flow_area = cell_size_x * harmonic_thickness
                water_flux, _, gas_flux = compute_fluxes_from_neighbour(
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

                south_neighbour_thickness = thickness_grid[i, j + 1, k]
                harmonic_thickness = compute_harmonic_mean(
                    cell_thickness, south_neighbour_thickness
                )
                south_flow_area = cell_size_x * harmonic_thickness
                water_flux, _, gas_flux = compute_fluxes_from_neighbour(
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

                flow_area_z = cell_size_x * cell_size_y
                top_neighbour_thickness = thickness_grid[i, j, k - 1]
                harmonic_thickness = compute_harmonic_mean(
                    cell_thickness, top_neighbour_thickness
                )
                top_flow_length = harmonic_thickness
                water_flux, _, gas_flux = compute_fluxes_from_neighbour(
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

                bottom_neighbour_thickness = thickness_grid[i, j, k + 1]
                harmonic_thickness = compute_harmonic_mean(
                    cell_thickness, bottom_neighbour_thickness
                )
                bottom_flow_length = harmonic_thickness
                water_flux, _, gas_flux = compute_fluxes_from_neighbour(
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

                water_well_source = net_water_well_rate_grid[i, j, k]
                gas_well_source = net_gas_well_rate_grid[i, j, k]

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

                idx = to_1D_index_interior_only(
                    i, j, k, cell_count_x, cell_count_y, cell_count_z
                )
                water_residual[idx] = (
                    water_accumulation
                    - net_water_flux
                    - water_well_source
                    - water_pvt_correction
                )
                gas_residual[idx] = (
                    gas_accumulation
                    - net_gas_flux
                    - gas_well_source
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
    typing.Tuple[
        typing.Tuple[ThreeDimensionalGrid, ThreeDimensionalGrid, ThreeDimensionalGrid],
        typing.Tuple[ThreeDimensionalGrid, ThreeDimensionalGrid, ThreeDimensionalGrid],
        typing.Tuple[ThreeDimensionalGrid, ThreeDimensionalGrid, ThreeDimensionalGrid],
    ],
]:
    """
    Compute all saturation-dependent quantities at fixed pressure.

    Called at each Newton iteration. Returns updated relative mobilities,
    capillary pressures, and directional mobility grids.
    """
    if normalize_saturations:
        # Clamp and normalize saturations
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
            water_saturation_grid=water_sat_grid,  # type: ignore
            oil_saturation_grid=oil_sat_grid,  # type: ignore
            gas_saturation_grid=gas_sat_grid,  # type: ignore
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
    return (relative_mobility_grids, capillary_pressure_grids, mobility_grids)


def _compute_residual(
    water_saturation_grid: ThreeDimensionalGrid,
    gas_saturation_grid: ThreeDimensionalGrid,
    old_water_saturation_grid: ThreeDimensionalGrid,
    old_gas_saturation_grid: ThreeDimensionalGrid,
    oil_pressure_grid: ThreeDimensionalGrid,
    pressure_change_grid: ThreeDimensionalGrid,
    capillary_pressure_grids: CapillaryPressureGrids[ThreeDimensions],
    mobility_grids: typing.Tuple[
        typing.Tuple[ThreeDimensionalGrid, ThreeDimensionalGrid, ThreeDimensionalGrid],
        typing.Tuple[ThreeDimensionalGrid, ThreeDimensionalGrid, ThreeDimensionalGrid],
        typing.Tuple[ThreeDimensionalGrid, ThreeDimensionalGrid, ThreeDimensionalGrid],
    ],
    fluid_properties: FluidProperties[ThreeDimensions],
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
    thickness_grid: ThreeDimensionalGrid,
    cell_size_x: float,
    cell_size_y: float,
    elevation_grid: ThreeDimensionalGrid,
    porosity_grid: ThreeDimensionalGrid,
    time_step_in_days: float,
    gravitational_constant: float,
    water_compressibility_grid: ThreeDimensionalGrid,
    gas_compressibility_grid: ThreeDimensionalGrid,
    rock_compressibility: float,
    well_indices_cache: WellIndicesCache,
    injection_rates: PhaseTensorsProxy[float, ThreeDimensions],
    production_rates: PhaseTensorsProxy[float, ThreeDimensions],
    dtype: npt.DTypeLike,
    pad_width: int = 1,
) -> typing.Tuple[npt.NDArray, npt.NDArray]:
    """
    Compute the residual from pre-computed saturation-dependent rock-fluid properties.

    :return: `(water_residual, gas_residual)` as 1D arrays of length N.
    """
    oil_water_capillary_pressure_grid, gas_oil_capillary_pressure_grid = (
        capillary_pressure_grids
    )
    (
        (water_mobility_grid_x, oil_mobility_grid_x, gas_mobility_grid_x),
        (water_mobility_grid_y, oil_mobility_grid_y, gas_mobility_grid_y),
        (water_mobility_grid_z, oil_mobility_grid_z, gas_mobility_grid_z),
    ) = mobility_grids

    net_water_well_rate_grid, _, net_gas_well_rate_grid = compute_well_rate_grids(
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
        well_indices_cache=well_indices_cache,
        injection_rates=injection_rates,
        production_rates=production_rates,
        dtype=dtype,
        pad_width=pad_width,
    )
    return _compute_saturation_residual(
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
        time_step_in_days=time_step_in_days,
        net_water_well_rate_grid=net_water_well_rate_grid,
        net_gas_well_rate_grid=net_gas_well_rate_grid,
        pressure_change_grid=pressure_change_grid,
        water_compressibility_grid=water_compressibility_grid,
        gas_compressibility_grid=gas_compressibility_grid,
        rock_compressibility=rock_compressibility,
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
    config: Config,
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
    thickness_grid: ThreeDimensionalGrid,
    cell_size_x: float,
    cell_size_y: float,
    elevation_grid: ThreeDimensionalGrid,
    porosity_grid: ThreeDimensionalGrid,
    time_step_in_days: float,
    gravitational_constant: float,
    water_compressibility_grid: ThreeDimensionalGrid,
    gas_compressibility_grid: ThreeDimensionalGrid,
    rock_compressibility: float,
    well_indices_cache: WellIndicesCache,
    injection_rates: PhaseTensorsProxy[float, ThreeDimensions],
    production_rates: PhaseTensorsProxy[float, ThreeDimensions],
    dtype: npt.DTypeLike,
    pad_width: int = 1,
) -> typing.Tuple[npt.NDArray, npt.NDArray]:
    """
    Computes full residual. (Re-)computes saturation-dependent rock-fluid properties,
    well rates, and then calls the saturation residual kernel.

    :return: `(water_residual, gas_residual)` as 1D arrays of length N.
    """
    (
        _,
        capillary_pressure_grids,
        mobility_grids,
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
        capillary_pressure_grids=capillary_pressure_grids,
        mobility_grids=mobility_grids,
        fluid_properties=fluid_properties,
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
        thickness_grid=thickness_grid,
        cell_size_x=cell_size_x,
        cell_size_y=cell_size_y,
        elevation_grid=elevation_grid,
        porosity_grid=porosity_grid,
        time_step_in_days=time_step_in_days,
        gravitational_constant=gravitational_constant,
        water_compressibility_grid=water_compressibility_grid,
        gas_compressibility_grid=gas_compressibility_grid,
        rock_compressibility=rock_compressibility,
        well_indices_cache=well_indices_cache,
        injection_rates=injection_rates,
        production_rates=production_rates,
        dtype=dtype,
        pad_width=pad_width,
    )


def assemble_numerical_jacobian(
    saturation_vector: npt.NDArray,
    residual_base: npt.NDArray,
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
    config: Config,
    thickness_grid: ThreeDimensionalGrid,
    cell_size_x: float,
    cell_size_y: float,
    elevation_grid: ThreeDimensionalGrid,
    porosity_grid: ThreeDimensionalGrid,
    time_step_in_days: float,
    gravitational_constant: float,
    water_compressibility_grid: ThreeDimensionalGrid,
    gas_compressibility_grid: ThreeDimensionalGrid,
    rock_compressibility: float,
    well_indices_cache: WellIndicesCache,
    injection_rates: PhaseTensorsProxy[float, ThreeDimensions],
    production_rates: PhaseTensorsProxy[float, ThreeDimensions],
    dtype: npt.DTypeLike,
    pad_width: int = 1,
) -> csr_matrix:
    """
    Assemble the saturation Jacobian using column-wise forward finite differences.

    :return: Sparse Jacobian matrix of shape (2N, 2N) in CSR format.
    """
    rows = []
    cols = []
    vals = []

    machine_eps = np.finfo(dtype).eps  # type: ignore
    base_epsilon = float(np.sqrt(machine_eps))

    water_saturation_grid_f64 = water_saturation_grid.astype(np.float64, copy=True)
    oil_saturation_grid_f64 = oil_saturation_grid.astype(np.float64, copy=True)
    gas_saturation_grid_f64 = gas_saturation_grid.astype(np.float64, copy=True)

    residual_kwargs = dict(  # noqa
        old_water_saturation_grid=old_water_saturation_grid,
        old_gas_saturation_grid=old_gas_saturation_grid,
        oil_pressure_grid=oil_pressure_grid,
        pressure_change_grid=pressure_change_grid,
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
        porosity_grid=porosity_grid,
        time_step_in_days=time_step_in_days,
        gravitational_constant=gravitational_constant,
        water_compressibility_grid=water_compressibility_grid,
        gas_compressibility_grid=gas_compressibility_grid,
        rock_compressibility=rock_compressibility,
        well_indices_cache=well_indices_cache,
        injection_rates=injection_rates,
        production_rates=production_rates,
        dtype=dtype,
        pad_width=pad_width,
    )

    for cell_1d_idx in range(interior_cell_count):
        i, j, k = from_1D_index_interior_only(
            cell_1d_idx, cell_count_x, cell_count_y, cell_count_z
        )

        affected_cell_indices = [cell_1d_idx]
        for ni, nj, nk in (
            (i + 1, j, k),
            (i - 1, j, k),
            (i, j + 1, k),
            (i, j - 1, k),
            (i, j, k + 1),
            (i, j, k - 1),
        ):
            neighbor_1d_idx = to_1D_index_interior_only(
                ni, nj, nk, cell_count_x, cell_count_y, cell_count_z
            )
            if neighbor_1d_idx >= 0:
                affected_cell_indices.append(neighbor_1d_idx)

        cell_water_saturation = water_saturation_grid_f64[i, j, k]
        cell_gas_saturation = gas_saturation_grid_f64[i, j, k]
        cell_oil_saturation = 1.0 - cell_water_saturation - cell_gas_saturation

        for var_offset in range(2):
            col = 2 * cell_1d_idx + var_offset
            s_j = saturation_vector[col]

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
                perturbed_oil_saturation = max(
                    0.0, 1.0 - perturbed_water_saturation - perturbed_gas_saturation
                )
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
                    cols.append(col)
                    vals.append(dR_water)

                row_gas = 2 * affected_idx + 1
                dR_gas = (
                    residual_perturbed[row_gas] - residual_base[row_gas]
                ) / epsilon
                if abs(dR_gas) > 1e-30:
                    rows.append(row_gas)
                    cols.append(col)
                    vals.append(dR_gas)

    system_size = 2 * interior_cell_count
    jacobian_coo = coo_matrix(
        (
            np.array(vals, dtype=np.float64),
            (
                np.array(rows, dtype=np.int32),
                np.array(cols, dtype=np.int32),
            ),
        ),
        shape=(system_size, system_size),
        dtype=np.float64,
    )
    return jacobian_coo.tocsr()


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
    ThreeDimensionalGrid,  # dPcow_dSw_eff  (So eliminated: dPcow_dSw + dPcow_dSo*(-1))
    ThreeDimensionalGrid,  # dPcow_dSg_eff  (                dPcow_dSo * (-1))
    ThreeDimensionalGrid,  # dPcgo_dSw_eff  (                dPcgo_dSo * (-1))
    ThreeDimensionalGrid,  # dPcgo_dSg_eff  (dPcgo_dSg + dPcgo_dSo*(-1))
]:
    """
    Compute all relperm and capillary pressure derivative grids for the
    analytical Jacobian.

    **Relperm derivatives**

    All nine raw partials `d kr_alpha / d S_beta` (with `beta in {Sw, So, Sg}`)
    are returned unchanged. The Jacobian kernel projects out the oil saturation
    dependence inline using `dSo/dSw = -1` and `dSo/dSg = -1`.

    **Capillary pressure derivatives projected onto (Sw, Sg) basis**

    Capillary pressures may depend on oil saturation (e.g. a gas-oil table
    indexed by So). The constraint `So = 1 - Sw - Sg` gives
    `dSo/dSw_free = dSo/dSg_free = -1`, so the effective derivatives are:

    ```
    dPcow_dSw_eff = dPcow_dSw + dPcow_dSo * (-1)
    dPcow_dSg_eff =              dPcow_dSo * (-1)
    dPcgo_dSw_eff =              dPcgo_dSo * (-1)
    dPcgo_dSg_eff = dPcgo_dSg + dPcgo_dSo * (-1)
    ```

    These are pre-projected here so the Numba kernel receives simple scalars
    with no further chain-rule work.

    :param water_saturation_grid: Current water saturation (3D, clamped).
    :param oil_saturation_grid: Current oil saturation (3D, clamped).
    :param gas_saturation_grid: Current gas saturation (3D, clamped).
    :param rock_properties: Rock properties (residual saturations).
    :return: 13-tuple of derivative grids.
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

    dkrw_dSw = np.asarray(relperm_derivatives["dKrw_dSw"], dtype=np.float64)
    dkrw_dSo = np.asarray(relperm_derivatives["dKrw_dSo"], dtype=np.float64)
    dkrw_dSg = np.asarray(relperm_derivatives["dKrw_dSg"], dtype=np.float64)
    dkro_dSw = np.asarray(relperm_derivatives["dKro_dSw"], dtype=np.float64)
    dkro_dSo = np.asarray(relperm_derivatives["dKro_dSo"], dtype=np.float64)
    dkro_dSg = np.asarray(relperm_derivatives["dKro_dSg"], dtype=np.float64)
    dkrg_dSw = np.asarray(relperm_derivatives["dKrg_dSw"], dtype=np.float64)
    dkrg_dSo = np.asarray(relperm_derivatives["dKrg_dSo"], dtype=np.float64)
    dkrg_dSg = np.asarray(relperm_derivatives["dKrg_dSg"], dtype=np.float64)

    if capillary_table is not None and not disable_capillary_effects:
        capillary_pressure_derivatives = capillary_table.derivatives(
            water_saturation=water_saturation_grid,
            oil_saturation=oil_saturation_grid,
            gas_saturation=gas_saturation_grid,
            irreducible_water_saturation=irreducible_water_saturation_grid,
            residual_oil_saturation_water=residual_oil_saturation_water_grid,
            residual_oil_saturation_gas=residual_oil_saturation_gas_grid,
            residual_gas_saturation=residual_gas_saturation_grid,
        )
        raw_dPcow_dSw = (
            np.asarray(capillary_pressure_derivatives["dPcow_dSw"], dtype=np.float64)
            * capillary_strength_factor
        )
        raw_dPcow_dSo = (
            np.asarray(capillary_pressure_derivatives["dPcow_dSo"], dtype=np.float64)
            * capillary_strength_factor
        )
        raw_dPcgo_dSo = (
            np.asarray(capillary_pressure_derivatives["dPcgo_dSo"], dtype=np.float64)
            * capillary_strength_factor
        )
        raw_dPcgo_dSg = (
            np.asarray(capillary_pressure_derivatives["dPcgo_dSg"], dtype=np.float64)
            * capillary_strength_factor
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

    return (
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
    water_mobility_grid_x: ThreeDimensionalGrid,
    water_mobility_grid_y: ThreeDimensionalGrid,
    water_mobility_grid_z: ThreeDimensionalGrid,
    gas_mobility_grid_x: ThreeDimensionalGrid,
    gas_mobility_grid_y: ThreeDimensionalGrid,
    gas_mobility_grid_z: ThreeDimensionalGrid,
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
    absolute_permeability_x_grid: ThreeDimensionalGrid,
    absolute_permeability_y_grid: ThreeDimensionalGrid,
    absolute_permeability_z_grid: ThreeDimensionalGrid,
    porosity_grid: ThreeDimensionalGrid,
    time_step_in_days: float,
    md_per_cp_to_ft2_per_psi_per_day: float,
) -> typing.Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Assemble the (2N x 2N) saturation Jacobian in COO format using analytical
    derivatives of the residual.

    The residual equations are (per interior cell i):

        R_w[i] = (phi*V/dt)*(Sw_i - Sw_old) - sum_faces(F_w_face) - qw - PVT_w
        R_g[i] = (phi*V/dt)*(Sg_i - Sg_old) - sum_faces(F_g_face) - qg - PVT_g

    Face fluxes:

        F_w = lambda_w_up * water_potential * T
        F_g = lambda_g_up * gas_potential * T

        water_potential = (Po_j - Po_i) - (Pcow_j - Pcow_i) - rho_w*g*(z_j - z_i)/144
        gas_potential = (Po_j - Po_i) + (Pcgo_j - Pcgo_i) - rho_g*g*(z_j - z_i)/144

    **Jacobian contributions per face**

    Free variables are Sw and Sg only (So = 1 - Sw - Sg is derived).
    For any phase alpha and free variable beta:

    Effective relperm derivatives (So projected out via dSo/dSw = dSo/dSg = -1):

        dkr_alpha/dSw_eff = dkr_alpha/dSw + dkr_alpha/dSo * (-1)
        dkr_alpha/dSg_eff = dkr_alpha/dSg + dkr_alpha/dSo * (-1)

    Mobility derivative (upwind cell only):

        dF_alpha/dSbeta_up = (k_up * conv / mu_alpha) * (dkr_alpha/dSbeta_eff_up) * delta_Phi_alpha * T

    where 'conv' is 'md_per_cp_to_ft2_per_psi_per_day', the same unit-conversion
    factor applied when building the mobility grids.  Omitting it causes the
    mobility derivative to be mis-scaled relative to the mobility itself,
    producing a structurally incorrect Jacobian.

    Capillary potential derivative (both face cells):

        dF_w/dSbeta = lambda_w_up * (-dPcow_i/dSbeta_eff) * T   when beta belongs to cell i
        dF_w/dSbeta = lambda_w_up * (+dPcow_j/dSbeta_eff) * T   when beta belongs to cell j

    Same pattern for gas with dPcgo.

    Residual convention R = accum - sum_F - wells, so dR/dS = -dF/dS for flux
    terms and +phi*V/dt for the accumulation diagonal.

    Oil relative permeability derivatives (dkro grids) are not needed by the
    water and gas residual equations directly, since the water and gas
    mobilities depend only on krw and krg respectively.  They are omitted from
    the function signature.

    Note: well derivatives are omitted (second-order correction).

    :param cell_count_x: Number of cells in the x-direction (including ghost cells).
    :param cell_count_y: Number of cells in the y-direction (including ghost cells).
    :param cell_count_z: Number of cells in the z-direction (including ghost cells).
    :param thickness_grid: 3D grid of cell thicknesses (ft).
    :param cell_size_x: Cell size in the x-direction (ft).
    :param cell_size_y: Cell size in the y-direction (ft).
    :param oil_pressure_grid: 3D grid of oil phase pressures (psi).
    :param water_density_grid: 3D grid of water densities (lb/ft3). Used for
        the gravity potential in the upwind selection of the water phase.
    :param gas_density_grid: 3D grid of gas densities (lb/ft3). Used for
        the gravity potential in the upwind selection of the gas phase.
    :param elevation_grid: 3D grid of cell elevations (ft).
    :param gravitational_constant: Gravitational constant conversion factor
        (lbf/lbm), equal to g/gc in consistent imperial units.
    :param water_mobility_grid_x: Water phase mobility in the x-direction
        (ft2/psi/day), already including absolute permeability and the
        md-per-cp unit conversion.
    :param water_mobility_grid_y: Water phase mobility in the y-direction
        (ft2/psi/day).
    :param water_mobility_grid_z: Water phase mobility in the z-direction
        (ft2/psi/day).
    :param gas_mobility_grid_x: Gas phase mobility in the x-direction
        (ft2/psi/day).
    :param gas_mobility_grid_y: Gas phase mobility in the y-direction
        (ft2/psi/day).
    :param gas_mobility_grid_z: Gas phase mobility in the z-direction
        (ft2/psi/day).
    :param oil_water_capillary_pressure_grid: 3D grid of oil-water capillary
        pressures Pcow = Po - Pw (psi).
    :param gas_oil_capillary_pressure_grid: 3D grid of gas-oil capillary
        pressures Pcgo = Pg - Po (psi).
    :param dkrw_dSw_grid: Partial derivative of water relative permeability
        with respect to water saturation (dimensionless/dimensionless).
    :param dkrw_dSo_grid: Partial derivative of water relative permeability
        with respect to oil saturation.
    :param dkrw_dSg_grid: Partial derivative of water relative permeability
        with respect to gas saturation.
    :param dkrg_dSw_grid: Partial derivative of gas relative permeability
        with respect to water saturation.
    :param dkrg_dSo_grid: Partial derivative of gas relative permeability
        with respect to oil saturation.
    :param dkrg_dSg_grid: Partial derivative of gas relative permeability
        with respect to gas saturation.
    :param dPcow_dSw_eff_grid: Effective derivative of Pcow with respect to
        Sw after projecting out the So dependence via dSo/dSw = -1.
    :param dPcow_dSg_eff_grid: Effective derivative of Pcow with respect to
        Sg after projecting out the So dependence via dSo/dSg = -1.
    :param dPcgo_dSw_eff_grid: Effective derivative of Pcgo with respect to
        Sw after projecting out the So dependence.
    :param dPcgo_dSg_eff_grid: Effective derivative of Pcgo with respect to
        Sg after projecting out the So dependence.
    :param water_viscosity_grid: 3D grid of water viscosities (cP).
    :param gas_viscosity_grid: 3D grid of gas viscosities (cP).
    :param absolute_permeability_x_grid: Absolute permeability in the
        x-direction (mD).
    :param absolute_permeability_y_grid: Absolute permeability in the
        y-direction (mD).
    :param absolute_permeability_z_grid: Absolute permeability in the
        z-direction (mD).
    :param porosity_grid: 3D grid of cell porosities (fraction).
    :param time_step_in_days: Time step size (days).
    :param md_per_cp_to_ft2_per_psi_per_day: Unit conversion factor from
        mD/cP to ft2/psi/day. Must be the same constant used when building
        the mobility grids so that the mobility derivative is consistently
        scaled.
    :return: (rows, cols, vals) COO arrays of length <= 56*N.
    """
    cells_per_slice = (cell_count_y - 2) * (cell_count_z - 2)
    # Per interior cell, worst-case non-zero entries:
    #
    #  Accumulation (diagonal only, one per free variable):
    #      (Rw, Sw), (Rg, Sg) → 2 entries
    #
    #  Per face (up to 6 faces):
    #      Diagonal entries for cell i  (dRw/dSw_i, dRw/dSg_i, dRg/dSw_i, dRg/dSg_i) → 4
    #      Off-diagonal entries for neighbour n (same 4 combinations)                  → 4
    #      Total per face: 8
    #
    #  6 faces × 8 = 48 flux entries
    #  Total: 2 + 48 = 50 per cell
    #
    #  Multiply by cells_per_slice and add headroom.
    max_nnz_per_slice = 56 * cells_per_slice
    slice_count = cell_count_x - 2

    all_rows = np.empty((slice_count, max_nnz_per_slice), dtype=np.int32)
    all_cols = np.empty((slice_count, max_nnz_per_slice), dtype=np.int32)
    all_vals = np.empty((slice_count, max_nnz_per_slice), dtype=np.float64)
    slice_fill = np.zeros(slice_count, dtype=np.int64)

    for i in numba.prange(1, cell_count_x - 1):  # type: ignore
        slice_idx = i - 1
        local_ptr = 0

        for j in range(1, cell_count_y - 1):
            for k in range(1, cell_count_z - 1):
                cell_idx = to_1D_index_interior_only(
                    i, j, k, cell_count_x, cell_count_y, cell_count_z
                )
                water_row = 2 * cell_idx
                gas_row = 2 * cell_idx + 1
                cell_water_saturation_column = 2 * cell_idx
                cell_gas_saturation_column = 2 * cell_idx + 1

                cell_thickness = thickness_grid[i, j, k]
                cell_volume = cell_size_x * cell_size_y * cell_thickness
                accumulation_coefficient = (
                    porosity_grid[i, j, k] * cell_volume / time_step_in_days
                )

                # Accumulation — diagonal entries only
                all_rows[slice_idx, local_ptr] = water_row
                all_cols[slice_idx, local_ptr] = cell_water_saturation_column
                all_vals[slice_idx, local_ptr] = accumulation_coefficient
                local_ptr += 1

                all_rows[slice_idx, local_ptr] = gas_row
                all_cols[slice_idx, local_ptr] = cell_gas_saturation_column
                all_vals[slice_idx, local_ptr] = accumulation_coefficient
                local_ptr += 1

                # Effective kr derivatives at cell i (So projected out)
                # dkra/dSw_eff = dkra/dSw + dkra/dSo * (-1)
                # dkra/dSg_eff = dkra/dSg + dkra/dSo * (-1)
                dkrw_dSw_i_eff = dkrw_dSw_grid[i, j, k] - dkrw_dSo_grid[i, j, k]
                dkrw_dSg_i_eff = dkrw_dSg_grid[i, j, k] - dkrw_dSo_grid[i, j, k]
                dkrg_dSw_i_eff = dkrg_dSw_grid[i, j, k] - dkrg_dSo_grid[i, j, k]
                dkrg_dSg_i_eff = dkrg_dSg_grid[i, j, k] - dkrg_dSo_grid[i, j, k]

                # Projected capillary derivatives at cell i
                dPcow_dSw_i = dPcow_dSw_eff_grid[i, j, k]
                dPcow_dSg_i = dPcow_dSg_eff_grid[i, j, k]
                dPcgo_dSw_i = dPcgo_dSw_eff_grid[i, j, k]
                dPcgo_dSg_i = dPcgo_dSg_eff_grid[i, j, k]

                for face in range(6):
                    if face == 0:
                        ni, nj, nk = i + 1, j, k
                        face_harmonic_thickness = compute_harmonic_mean(
                            cell_thickness, thickness_grid[ni, nj, nk]
                        )
                        flow_area = cell_size_y * face_harmonic_thickness
                        flow_length = cell_size_x
                        cell_water_mobility = water_mobility_grid_x[i, j, k]
                        neighbour_water_mobility = water_mobility_grid_x[ni, nj, nk]
                        cell_gas_mobility = gas_mobility_grid_x[i, j, k]
                        neighbour_gas_mobility = gas_mobility_grid_x[ni, nj, nk]
                        cell_absolute_permeability = absolute_permeability_x_grid[
                            i, j, k
                        ]
                        neighbour_absolute_permeability = absolute_permeability_x_grid[
                            ni, nj, nk
                        ]
                        cell_water_viscosity = water_viscosity_grid[i, j, k]
                        neighbour_water_viscosity = water_viscosity_grid[ni, nj, nk]
                        cell_gas_viscosity = gas_viscosity_grid[i, j, k]
                        neighbour_gas_viscosity = gas_viscosity_grid[ni, nj, nk]
                    elif face == 1:
                        ni, nj, nk = i - 1, j, k
                        face_harmonic_thickness = compute_harmonic_mean(
                            cell_thickness, thickness_grid[ni, nj, nk]
                        )
                        flow_area = cell_size_y * face_harmonic_thickness
                        flow_length = cell_size_x
                        cell_water_mobility = water_mobility_grid_x[i, j, k]
                        neighbour_water_mobility = water_mobility_grid_x[ni, nj, nk]
                        cell_gas_mobility = gas_mobility_grid_x[i, j, k]
                        neighbour_gas_mobility = gas_mobility_grid_x[ni, nj, nk]
                        cell_absolute_permeability = absolute_permeability_x_grid[
                            i, j, k
                        ]
                        neighbour_absolute_permeability = absolute_permeability_x_grid[
                            ni, nj, nk
                        ]
                        cell_water_viscosity = water_viscosity_grid[i, j, k]
                        neighbour_water_viscosity = water_viscosity_grid[ni, nj, nk]
                        cell_gas_viscosity = gas_viscosity_grid[i, j, k]
                        neighbour_gas_viscosity = gas_viscosity_grid[ni, nj, nk]
                    elif face == 2:
                        ni, nj, nk = i, j + 1, k
                        face_harmonic_thickness = compute_harmonic_mean(
                            cell_thickness, thickness_grid[ni, nj, nk]
                        )
                        flow_area = cell_size_x * face_harmonic_thickness
                        flow_length = cell_size_y
                        cell_water_mobility = water_mobility_grid_y[i, j, k]
                        neighbour_water_mobility = water_mobility_grid_y[ni, nj, nk]
                        cell_gas_mobility = gas_mobility_grid_y[i, j, k]
                        neighbour_gas_mobility = gas_mobility_grid_y[ni, nj, nk]
                        cell_absolute_permeability = absolute_permeability_y_grid[
                            i, j, k
                        ]
                        neighbour_absolute_permeability = absolute_permeability_y_grid[
                            ni, nj, nk
                        ]
                        cell_water_viscosity = water_viscosity_grid[i, j, k]
                        neighbour_water_viscosity = water_viscosity_grid[ni, nj, nk]
                        cell_gas_viscosity = gas_viscosity_grid[i, j, k]
                        neighbour_gas_viscosity = gas_viscosity_grid[ni, nj, nk]
                    elif face == 3:
                        ni, nj, nk = i, j - 1, k
                        face_harmonic_thickness = compute_harmonic_mean(
                            cell_thickness, thickness_grid[ni, nj, nk]
                        )
                        flow_area = cell_size_x * face_harmonic_thickness
                        flow_length = cell_size_y
                        cell_water_mobility = water_mobility_grid_y[i, j, k]
                        neighbour_water_mobility = water_mobility_grid_y[ni, nj, nk]
                        cell_gas_mobility = gas_mobility_grid_y[i, j, k]
                        neighbour_gas_mobility = gas_mobility_grid_y[ni, nj, nk]
                        cell_absolute_permeability = absolute_permeability_y_grid[
                            i, j, k
                        ]
                        neighbour_absolute_permeability = absolute_permeability_y_grid[
                            ni, nj, nk
                        ]
                        cell_water_viscosity = water_viscosity_grid[i, j, k]
                        neighbour_water_viscosity = water_viscosity_grid[ni, nj, nk]
                        cell_gas_viscosity = gas_viscosity_grid[i, j, k]
                        neighbour_gas_viscosity = gas_viscosity_grid[ni, nj, nk]
                    elif face == 4:
                        ni, nj, nk = i, j, k + 1
                        face_harmonic_thickness = compute_harmonic_mean(
                            cell_thickness, thickness_grid[ni, nj, nk]
                        )
                        flow_area = cell_size_x * cell_size_y
                        flow_length = (
                            face_harmonic_thickness
                            if face_harmonic_thickness > 0.0
                            else 1.0
                        )
                        cell_water_mobility = water_mobility_grid_z[i, j, k]
                        neighbour_water_mobility = water_mobility_grid_z[ni, nj, nk]
                        cell_gas_mobility = gas_mobility_grid_z[i, j, k]
                        neighbour_gas_mobility = gas_mobility_grid_z[ni, nj, nk]
                        cell_absolute_permeability = absolute_permeability_z_grid[
                            i, j, k
                        ]
                        neighbour_absolute_permeability = absolute_permeability_z_grid[
                            ni, nj, nk
                        ]
                        cell_water_viscosity = water_viscosity_grid[i, j, k]
                        neighbour_water_viscosity = water_viscosity_grid[ni, nj, nk]
                        cell_gas_viscosity = gas_viscosity_grid[i, j, k]
                        neighbour_gas_viscosity = gas_viscosity_grid[ni, nj, nk]
                    else:  # face == 5
                        ni, nj, nk = i, j, k - 1
                        face_harmonic_thickness = compute_harmonic_mean(
                            cell_thickness, thickness_grid[ni, nj, nk]
                        )
                        flow_area = cell_size_x * cell_size_y
                        flow_length = (
                            face_harmonic_thickness
                            if face_harmonic_thickness > 0.0
                            else 1.0
                        )
                        cell_water_mobility = water_mobility_grid_z[i, j, k]
                        neighbour_water_mobility = water_mobility_grid_z[ni, nj, nk]
                        cell_gas_mobility = gas_mobility_grid_z[i, j, k]
                        neighbour_gas_mobility = gas_mobility_grid_z[ni, nj, nk]
                        cell_absolute_permeability = absolute_permeability_z_grid[
                            i, j, k
                        ]
                        neighbour_absolute_permeability = absolute_permeability_z_grid[
                            ni, nj, nk
                        ]
                        cell_water_viscosity = water_viscosity_grid[i, j, k]
                        neighbour_water_viscosity = water_viscosity_grid[ni, nj, nk]
                        cell_gas_viscosity = gas_viscosity_grid[i, j, k]
                        neighbour_gas_viscosity = gas_viscosity_grid[ni, nj, nk]

                    # Skip ghost cells
                    if (
                        ni < 1
                        or ni >= cell_count_x - 1
                        or nj < 1
                        or nj >= cell_count_y - 1
                        or nk < 1
                        or nk >= cell_count_z - 1
                    ):
                        continue

                    transmissibility = flow_area / flow_length

                    # The residual selects the upwind density based on the sign of the
                    # potential difference, where that potential difference itself uses
                    # the upwind density. We resolve this the same way the residual does:
                    # compute the potential with each cell's own density, then pick the
                    # upwind direction based on those values. For the residual, when
                    # water_potential_difference > 0, flow is from neighbour -> cell i,
                    # so neighbour is upwind (density = neighbour's). When <= 0, cell i
                    # is upwind. We use each cell's own density in its own potential
                    # estimate to determine which direction is upwind, matching the
                    # residual's two-pass approach.
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

                    # Pick upwind density on pressure difference sign
                    if water_pressure_difference > 0.0:
                        upwind_water_density = water_density_grid[ni, nj, nk]
                    else:
                        upwind_water_density = water_density_grid[i, j, k]

                    # Compute potential with upwind density (matching residual stage 2)
                    water_gravity_potential = (
                        upwind_water_density
                        * gravitational_constant
                        * elevation_difference
                        / 144.0
                    )
                    water_potential = (
                        water_pressure_difference + water_gravity_potential
                    )

                    # Residual: upwind = neighbour when potential > 0 (flow neighbour->i),
                    # upwind = cell i when potential <= 0 (flow i->neighbour).
                    # Use neighbour's perspective when neighbour is upwind, cell i's when i is upwind.
                    water_neighbour_is_upwind = water_potential > 0.0

                    # Gas phase potential — same pattern.
                    gas_pressure_difference = (
                        oil_pressure_difference + gas_oil_capillary_pressure_difference
                    )
                    if gas_pressure_difference > 0.0:
                        upwind_gas_density = gas_density_grid[ni, nj, nk]
                    else:
                        upwind_gas_density = gas_density_grid[i, j, k]

                    gas_gravity_potential = (
                        upwind_gas_density
                        * gravitational_constant
                        * elevation_difference
                        / 144.0
                    )
                    gas_potential = gas_pressure_difference + gas_gravity_potential
                    gas_neighbour_is_upwind = gas_potential > 0.0

                    # Neighbour 1D index
                    neighbour_idx = to_1D_index_interior_only(
                        ni, nj, nk, cell_count_x, cell_count_y, cell_count_z
                    )
                    neighbour_water_saturation_column = 2 * neighbour_idx
                    neigbour_gas_saturation_column = 2 * neighbour_idx + 1

                    # Effective kr derivatives at neighbour n
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

                    # Projected capillary derivatives at neighbour n
                    dPcow_dSw_n = dPcow_dSw_eff_grid[ni, nj, nk]
                    dPcow_dSg_n = dPcow_dSg_eff_grid[ni, nj, nk]
                    dPcgo_dSw_n = dPcgo_dSw_eff_grid[ni, nj, nk]
                    dPcgo_dSg_n = dPcgo_dSg_eff_grid[ni, nj, nk]

                    # WATER flux derivatives
                    # mob_term: only for upwind cell
                    #   dF_w/dSbeta_up = (k_abs_up * conv / mu_w) * dkrw/dSbeta_eff_up * water_potential * T
                    # cap_term: both cells
                    #   dF_w/dSbeta_i = upwind_water_mobility * (-dPcow_i/dSbeta_eff) * T
                    #   dF_w/dSbeta_n = upwind_water_mobility * (+dPcow_n/dSbeta_eff) * T
                    # dR_w/dS = -dF_w/dS
                    #
                    # water_neighbour_is_upwind mirrors the upwind_water_density decision
                    # in compute_fluxes_from_neighbour: neighbour is upwind when
                    # water_potential_difference > 0 (flow from neighbour into cell i).
                    upwind_water_mobility = (
                        neighbour_water_mobility
                        if water_neighbour_is_upwind
                        else cell_water_mobility
                    )

                    if not water_neighbour_is_upwind:
                        # cell i is upwind: mob contribution to diagonal (i)
                        inverse_cell_water_viscosity = (
                            1.0 / cell_water_viscosity
                            if cell_water_viscosity > 0.0
                            else 0.0
                        )
                        dFw_mob_dSw_i = (
                            cell_absolute_permeability
                            * md_per_cp_to_ft2_per_psi_per_day
                            * inverse_cell_water_viscosity
                            * dkrw_dSw_i_eff
                            * water_potential
                            * transmissibility
                        )
                        dFw_mob_dSg_i = (
                            cell_absolute_permeability
                            * md_per_cp_to_ft2_per_psi_per_day
                            * inverse_cell_water_viscosity
                            * dkrw_dSg_i_eff
                            * water_potential
                            * transmissibility
                        )
                        dFw_mob_dSw_n = 0.0
                        dFw_mob_dSg_n = 0.0
                    else:
                        # neighbour is upwind: mob contribution to off-diagonal (n)
                        inverse_neighbour_water_viscosity = (
                            1.0 / neighbour_water_viscosity
                            if neighbour_water_viscosity > 0.0
                            else 0.0
                        )
                        dFw_mob_dSw_i = 0.0
                        dFw_mob_dSg_i = 0.0
                        dFw_mob_dSw_n = (
                            neighbour_absolute_permeability
                            * md_per_cp_to_ft2_per_psi_per_day
                            * inverse_neighbour_water_viscosity
                            * dkrw_dSw_n_eff
                            * water_potential
                            * transmissibility
                        )
                        dFw_mob_dSg_n = (
                            neighbour_absolute_permeability
                            * md_per_cp_to_ft2_per_psi_per_day
                            * inverse_neighbour_water_viscosity
                            * dkrw_dSg_n_eff
                            * water_potential
                            * transmissibility
                        )

                    # Capillary contributions (always both cells)
                    dFw_cap_dSw_i = (
                        upwind_water_mobility * (+dPcow_dSw_i) * transmissibility
                    )
                    dFw_cap_dSg_i = (
                        upwind_water_mobility * (+dPcow_dSg_i) * transmissibility
                    )
                    dFw_cap_dSw_n = (
                        upwind_water_mobility * (-dPcow_dSw_n) * transmissibility
                    )
                    dFw_cap_dSg_n = (
                        upwind_water_mobility * (-dPcow_dSg_n) * transmissibility
                    )

                    # Total dF_w/dS -> dR_w/dS = -dF_w/dS
                    dRw_dSw_i = -(dFw_mob_dSw_i + dFw_cap_dSw_i)
                    dRw_dSg_i = -(dFw_mob_dSg_i + dFw_cap_dSg_i)
                    dRw_dSw_n = -(dFw_mob_dSw_n + dFw_cap_dSw_n)
                    dRw_dSg_n = -(dFw_mob_dSg_n + dFw_cap_dSg_n)

                    # GAS flux derivatives (same structure as water)
                    upwind_gas_mobility = (
                        neighbour_gas_mobility
                        if gas_neighbour_is_upwind
                        else cell_gas_mobility
                    )

                    if not gas_neighbour_is_upwind:
                        inverse_cell_gas_viscosity = (
                            1.0 / cell_gas_viscosity
                            if cell_gas_viscosity > 0.0
                            else 0.0
                        )
                        dFg_mob_dSw_i = (
                            cell_absolute_permeability
                            * md_per_cp_to_ft2_per_psi_per_day
                            * inverse_cell_gas_viscosity
                            * dkrg_dSw_i_eff
                            * gas_potential
                            * transmissibility
                        )
                        dFg_mob_dSg_i = (
                            cell_absolute_permeability
                            * md_per_cp_to_ft2_per_psi_per_day
                            * inverse_cell_gas_viscosity
                            * dkrg_dSg_i_eff
                            * gas_potential
                            * transmissibility
                        )
                        dFg_mob_dSw_n = 0.0
                        dFg_mob_dSg_n = 0.0
                    else:
                        inverse_neighbour_gas_viscosity = (
                            1.0 / neighbour_gas_viscosity
                            if neighbour_gas_viscosity > 0.0
                            else 0.0
                        )
                        dFg_mob_dSw_i = 0.0
                        dFg_mob_dSg_i = 0.0
                        dFg_mob_dSw_n = (
                            neighbour_absolute_permeability
                            * md_per_cp_to_ft2_per_psi_per_day
                            * inverse_neighbour_gas_viscosity
                            * dkrg_dSw_n_eff
                            * gas_potential
                            * transmissibility
                        )
                        dFg_mob_dSg_n = (
                            neighbour_absolute_permeability
                            * md_per_cp_to_ft2_per_psi_per_day
                            * inverse_neighbour_gas_viscosity
                            * dkrg_dSg_n_eff
                            * gas_potential
                            * transmissibility
                        )

                    dFg_cap_dSw_i = (
                        upwind_gas_mobility * (+dPcgo_dSw_i) * transmissibility
                    )
                    dFg_cap_dSg_i = (
                        upwind_gas_mobility * (+dPcgo_dSg_i) * transmissibility
                    )
                    dFg_cap_dSw_n = (
                        upwind_gas_mobility * (-dPcgo_dSw_n) * transmissibility
                    )
                    dFg_cap_dSg_n = (
                        upwind_gas_mobility * (-dPcgo_dSg_n) * transmissibility
                    )

                    dRg_dSw_i = -(dFg_mob_dSw_i + dFg_cap_dSw_i)
                    dRg_dSg_i = -(dFg_mob_dSg_i + dFg_cap_dSg_i)
                    dRg_dSw_n = -(dFg_mob_dSw_n + dFg_cap_dSw_n)
                    dRg_dSg_n = -(dFg_mob_dSg_n + dFg_cap_dSg_n)

                    # Write diagonal entries (cell i)
                    if dRw_dSw_i != 0.0:
                        all_rows[slice_idx, local_ptr] = water_row
                        all_cols[slice_idx, local_ptr] = cell_water_saturation_column
                        all_vals[slice_idx, local_ptr] = dRw_dSw_i
                        local_ptr += 1
                    if dRw_dSg_i != 0.0:
                        all_rows[slice_idx, local_ptr] = water_row
                        all_cols[slice_idx, local_ptr] = cell_gas_saturation_column
                        all_vals[slice_idx, local_ptr] = dRw_dSg_i
                        local_ptr += 1
                    if dRg_dSw_i != 0.0:
                        all_rows[slice_idx, local_ptr] = gas_row
                        all_cols[slice_idx, local_ptr] = cell_water_saturation_column
                        all_vals[slice_idx, local_ptr] = dRg_dSw_i
                        local_ptr += 1
                    if dRg_dSg_i != 0.0:
                        all_rows[slice_idx, local_ptr] = gas_row
                        all_cols[slice_idx, local_ptr] = cell_gas_saturation_column
                        all_vals[slice_idx, local_ptr] = dRg_dSg_i
                        local_ptr += 1

                    # Write off-diagonal entries (neighbour n, interior only)
                    if neighbour_idx >= 0:
                        if dRw_dSw_n != 0.0:
                            all_rows[slice_idx, local_ptr] = water_row
                            all_cols[slice_idx, local_ptr] = (
                                neighbour_water_saturation_column
                            )
                            all_vals[slice_idx, local_ptr] = dRw_dSw_n
                            local_ptr += 1
                        if dRw_dSg_n != 0.0:
                            all_rows[slice_idx, local_ptr] = water_row
                            all_cols[slice_idx, local_ptr] = (
                                neigbour_gas_saturation_column
                            )
                            all_vals[slice_idx, local_ptr] = dRw_dSg_n
                            local_ptr += 1
                        if dRg_dSw_n != 0.0:
                            all_rows[slice_idx, local_ptr] = gas_row
                            all_cols[slice_idx, local_ptr] = (
                                neighbour_water_saturation_column
                            )
                            all_vals[slice_idx, local_ptr] = dRg_dSw_n
                            local_ptr += 1
                        if dRg_dSg_n != 0.0:
                            all_rows[slice_idx, local_ptr] = gas_row
                            all_cols[slice_idx, local_ptr] = (
                                neigbour_gas_saturation_column
                            )
                            all_vals[slice_idx, local_ptr] = dRg_dSg_n
                            local_ptr += 1

                # end face loop
            # end k
        # end j

        slice_fill[slice_idx] = local_ptr
    # end parallel i

    # Sequential compaction
    total_nnz = 0
    for s in range(slice_count):
        total_nnz += slice_fill[s]

    out_rows = np.empty(total_nnz, dtype=np.int32)
    out_cols = np.empty(total_nnz, dtype=np.int32)
    out_vals = np.empty(total_nnz, dtype=np.float64)
    write_ptr = 0
    for s in range(slice_count):
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
    pad_width: int,
) -> typing.Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Compute the well contributions to the saturation Jacobian (dR/dS at perforated cells).

    Wells contribute only to the diagonal (each perforated cell couples only to itself).
    For BHP-controlled wells, the phase well rate depends on relative permeability:

        q_alpha = PI_alpha * (P_cell - BHP)     where  PI_alpha = WI * k_abs * conv * kr_alpha / mu_alpha

    Taking the derivative w.r.t. Sw (free variable, So eliminated):

        d(q_alpha)/dSw_eff = WI * k_abs * conv / mu_alpha * dkr_alpha/dSw_eff * (P_cell - BHP)
        dR_alpha/dSw       = -d(q_alpha)/dSw_eff

    For rate-controlled injection wells the rate is fixed, so the saturation derivative is zero.
    Production wells are always BHP-controlled in the saturation Jacobian (rate control fixes
    total surface rate but the phase split still depends on kr).

    The pattern mirrors `compute_well_contributions` in the implicit pressure solver exactly:
    same BHP call signatures, same well index computation, same phase dispatch.

    :return: (rows, cols, vals) COO arrays for the well Jacobian entries.
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
        """Append a single (row, col, val) triplet for a diagonal well entry."""
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
            water_bhp, _, gas_bhp = injection_bhps[
                i - pad_width, j - pad_width, k - pad_width
            ]

            # Injection only injects the injected phase; we differentiate the
            # injected-phase well rate w.r.t. Sw and Sg.
            if gas_bhp:
                drawdown = cell_pressure - gas_bhp
                # For gas PI, the kr dependence is PI ∝ krg/mu_g * k_abs * conv.
                # d(PI)/d(krg) * d(krg)/dSw_eff = base_pi / max(krg, eps) * dkrg/dSw_eff
                # is numerically fragile; instead use the raw mobility derivative:
                #   d(q_g)/dSw_eff = WI * conv / mu_g * dkrg/dSw_eff * drawdown
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
                # dR_g/dSw = -dq_g/dSw (residual = accum - flux - q_well)
                _add_diagonal_entry(cell_1d_index, 1, 0, -dqg_dSw)
                _add_diagonal_entry(cell_1d_index, 1, 1, -dqg_dSg)

            elif water_bhp:
                # Water injection: only water rate has kr sensitivity
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
                # dR_w/dSw = -dq_w/dSw
                _add_diagonal_entry(cell_1d_index, 0, 0, -dqw_dSw)
                _add_diagonal_entry(cell_1d_index, 0, 1, -dqw_dSg)

    for well_indices in well_indices_cache.production.values():
        for perforation_index in well_indices:
            i, j, k = perforation_index.cell
            cell_1d_index = perforation_index.cell_1d_index
            well_index = perforation_index.well_index
            cell_pressure = typing.cast(float, oil_pressure_grid[i, j, k])
            water_bhp, _, gas_bhp = production_bhps[
                i - pad_width, j - pad_width, k - pad_width
            ]

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

            # No contribution needed for produced oil. Water/gas equations have no direct oil-kr dependency
            # (So is derived; oil doesn't appear in the water or gas residual
            # equations directly since kro only enters the oil material balance
            # which we do not solve explicitly).

    return (
        np.array(rows, dtype=np.int32),
        np.array(cols, dtype=np.int32),
        np.array(vals, dtype=np.float64),
    )


def assemble_analytical_jacobian(
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
    interior_cell_count: int,
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
    absolute_permeability_x_grid: ThreeDimensionalGrid,
    absolute_permeability_y_grid: ThreeDimensionalGrid,
    absolute_permeability_z_grid: ThreeDimensionalGrid,
    porosity_grid: ThreeDimensionalGrid,
    rock_properties: RockProperties[ThreeDimensions],
    injection_bhps: PhaseTensorsProxy[float, ThreeDimensions],
    production_bhps: PhaseTensorsProxy[float, ThreeDimensions],
    capillary_pressure_grids: CapillaryPressureGrids[ThreeDimensions],
    mobility_grids: typing.Tuple[
        typing.Tuple[ThreeDimensionalGrid, ThreeDimensionalGrid, ThreeDimensionalGrid],
        typing.Tuple[ThreeDimensionalGrid, ThreeDimensionalGrid, ThreeDimensionalGrid],
        typing.Tuple[ThreeDimensionalGrid, ThreeDimensionalGrid, ThreeDimensionalGrid],
    ],
    elevation_grid: ThreeDimensionalGrid,
    gravitational_constant: float,
    time_step_in_days: float,
    config: Config,
    well_indices_cache: WellIndicesCache,
    pad_width: int = 1,
) -> csr_matrix:
    """
    Assemble the full analytical saturation Jacobian.

    Combines:
    - Inter-cell flux derivatives.
    - Well rate derivatives.

    Both parts are assembled as COO triplets and merged into a single CSR
    matrix, which automatically sums duplicate entries so diagonal contributions
    from multiple faces/wells add correctly.
    """
    (
        dkrw_dSw_grid,
        dkrw_dSo_grid,
        dkrw_dSg_grid,
        _,
        _,
        _,
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
        (water_mobility_grid_x, _, gas_mobility_grid_x),
        (water_mobility_grid_y, _, gas_mobility_grid_y),
        (water_mobility_grid_z, _, gas_mobility_grid_z),
    ) = mobility_grids

    md_per_cp_to_ft2_per_psi_per_day = (
        c.MILLIDARCIES_PER_CENTIPOISE_TO_SQUARE_FEET_PER_PSI_PER_DAY
    )

    # Assemble inter-cell flux Jacobian
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
        water_mobility_grid_x=water_mobility_grid_x,
        water_mobility_grid_y=water_mobility_grid_y,
        water_mobility_grid_z=water_mobility_grid_z,
        gas_mobility_grid_x=gas_mobility_grid_x,
        gas_mobility_grid_y=gas_mobility_grid_y,
        gas_mobility_grid_z=gas_mobility_grid_z,
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
        absolute_permeability_x_grid=absolute_permeability_x_grid,
        absolute_permeability_y_grid=absolute_permeability_y_grid,
        absolute_permeability_z_grid=absolute_permeability_z_grid,
        porosity_grid=porosity_grid,
        time_step_in_days=time_step_in_days,
        md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
    )

    # Assemble well rate Jacobian (diagonal contributions only)
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
        pad_width=pad_width,
    )

    # Merge both parts into a single COO matrix
    # coo_matrix.tocsr() sums duplicate (row, col) entries, so flux and well
    # contributions at the same diagonal position are correctly accumulated.
    system_size = 2 * interior_cell_count
    combined_rows = np.concatenate([flux_rows, well_rows])
    combined_cols = np.concatenate([flux_cols, well_cols])
    combined_vals = np.concatenate([flux_vals, well_vals])
    jacobian_coo = coo_matrix(
        (combined_vals, (combined_rows, combined_cols)),
        shape=(system_size, system_size),
        dtype=np.float64,
    )
    return jacobian_coo.tocsr()


def assemble_jacobian(
    config: Config,
    saturation_vector: npt.NDArray,
    residual_base: npt.NDArray,
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
    thickness_grid: ThreeDimensionalGrid,
    cell_size_x: float,
    cell_size_y: float,
    elevation_grid: ThreeDimensionalGrid,
    porosity_grid: ThreeDimensionalGrid,
    time_step_in_days: float,
    gravitational_constant: float,
    water_compressibility_grid: ThreeDimensionalGrid,
    gas_compressibility_grid: ThreeDimensionalGrid,
    rock_compressibility: float,
    well_indices_cache: WellIndicesCache,
    injection_bhps: PhaseTensorsProxy[float, ThreeDimensions],
    production_bhps: PhaseTensorsProxy[float, ThreeDimensions],
    injection_rates: PhaseTensorsProxy[float, ThreeDimensions],
    production_rates: PhaseTensorsProxy[float, ThreeDimensions],
    dtype: npt.DTypeLike,
    capillary_pressure_grids: CapillaryPressureGrids[ThreeDimensions],
    mobility_grids: typing.Tuple[
        typing.Tuple[ThreeDimensionalGrid, ThreeDimensionalGrid, ThreeDimensionalGrid],
        typing.Tuple[ThreeDimensionalGrid, ThreeDimensionalGrid, ThreeDimensionalGrid],
        typing.Tuple[ThreeDimensionalGrid, ThreeDimensionalGrid, ThreeDimensionalGrid],
    ],
    pad_width: int = 1,
) -> csr_matrix:
    """
    Assembles the Jacobian for the system.

    Dispatches to the numerical or analytical Jacobian assembler based on
    `config.jacobian_assembly_method`.

    :param capillary_pressure_grids: `(Pcow_grid, Pcgo_grid)` at the current
        saturation iterate.  Used by the analytical Jacobian kernel for upwind
        potential differences.
    :param relative_mobility_grids: `(lam_w, lam_o, lam_g)` relative mobilities
        at the current iterate.  Forwarded to the well Jacobian function.
    :param mobility_grids: Directional mobility grids at the current iterate.
        Used by the analytical Jacobian kernel.
    :return: Jacobian as a (2N x 2N) CSR sparse matrix.
    """
    if config.jacobian_assembly_method == "analytical":
        return assemble_analytical_jacobian(
            interior_cell_count=interior_cell_count,
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
            gravitational_constant=gravitational_constant,
            water_viscosity_grid=fluid_properties.water_viscosity_grid,
            gas_viscosity_grid=fluid_properties.gas_viscosity_grid,
            absolute_permeability_x_grid=rock_properties.absolute_permeability.x,
            absolute_permeability_y_grid=rock_properties.absolute_permeability.y,
            absolute_permeability_z_grid=rock_properties.absolute_permeability.z,
            porosity_grid=porosity_grid,
            mobility_grids=mobility_grids,
            capillary_pressure_grids=capillary_pressure_grids,
            time_step_in_days=time_step_in_days,
            config=config,
            well_indices_cache=well_indices_cache,
            injection_bhps=injection_bhps,
            production_bhps=production_bhps,
            pad_width=pad_width,
        )

    # Numerical path
    return assemble_numerical_jacobian(
        saturation_vector=saturation_vector,
        residual_base=residual_base,
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
        config=config,
        thickness_grid=thickness_grid,
        cell_size_x=cell_size_x,
        cell_size_y=cell_size_y,
        elevation_grid=elevation_grid,
        porosity_grid=porosity_grid,
        time_step_in_days=time_step_in_days,
        gravitational_constant=gravitational_constant,
        water_compressibility_grid=water_compressibility_grid,
        gas_compressibility_grid=gas_compressibility_grid,
        rock_compressibility=rock_compressibility,
        well_indices_cache=well_indices_cache,
        injection_rates=injection_rates,
        production_rates=production_rates,
        dtype=dtype,
        pad_width=pad_width,
    )


def solve_implicit_saturation(
    grid_shape: ThreeDimensions,
    cell_dimension: typing.Tuple[float, float],
    oil_pressure_grid: ThreeDimensionalGrid,
    pressure_change_grid: ThreeDimensionalGrid,
    old_water_saturation_grid: ThreeDimensionalGrid,
    old_oil_saturation_grid: ThreeDimensionalGrid,
    old_gas_saturation_grid: ThreeDimensionalGrid,
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
    thickness_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    porosity_grid: ThreeDimensionalGrid,
    time_step_size: float,
    time: float,
    rock_properties: RockProperties[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
    config: Config,
    well_indices_cache: WellIndicesCache,
    injection_bhps: PhaseTensorsProxy[float, ThreeDimensions],
    production_bhps: PhaseTensorsProxy[float, ThreeDimensions],
    injection_rates: PhaseTensorsProxy[float, ThreeDimensions],
    production_rates: PhaseTensorsProxy[float, ThreeDimensions],
    water_compressibility_grid: ThreeDimensionalGrid,
    gas_compressibility_grid: ThreeDimensionalGrid,
    rock_compressibility: float,
    boundary_conditions: BoundaryConditions[ThreeDimensions],
    pad_width: int = 1,
    max_newton_iterations: int = 12,
    newton_tolerance: float = 1e-6,
    line_search_max_cuts: int = 4,
    max_saturation_change: float = 0.05,
    saturation_convergence_tolerance: float = 1e-4,
) -> EvolutionResult[ImplicitSaturationSolution, typing.List[NewtonConvergenceInfo]]:
    """
    Solve the implicit saturation equations using Newton-Raphson iteration
    with backtracking line search.

    The Newton loop iterates on saturations until either the relative residual norm drops below `newton_tolerance`,
    or the maximum saturation change per iteration drops below `saturation_convergence_tolerance` and the relative
    residual is below 1e-3 (effective convergence despite the upwind discontinuity floor).
    """
    cell_size_x, cell_size_y = cell_dimension
    time_step_in_days = time_step_size * c.DAYS_PER_SECOND
    dtype = get_dtype()
    interior_cell_count = (cell_count_x - 2) * (cell_count_y - 2) * (cell_count_z - 2)

    gravitational_constant = (
        c.ACCELERATION_DUE_TO_GRAVITY_FEET_PER_SECONDS_SQUARE
        / c.GRAVITATIONAL_CONSTANT_LBM_FT_PER_LBF_S2
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

    convergence_history: typing.List[NewtonConvergenceInfo] = []
    initial_residual_norm = 0.0
    converged = False
    final_iteration = 0
    final_residual_norm = 0.0

    best_residual_norm = float("inf")
    stagnation_count = 0
    stagnation_patience = config.newton_stagnation_patience
    stagnation_improvement_threshold = config.newton_stagnation_improvement_threshold
    min_step_size = float(np.sqrt(np.finfo(dtype).eps))

    residual_kwargs = dict(  # noqa
        old_water_saturation_grid=old_water_saturation_grid,
        old_gas_saturation_grid=old_gas_saturation_grid,
        oil_pressure_grid=oil_pressure_grid,
        pressure_change_grid=pressure_change_grid,
        fluid_properties=fluid_properties,
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
        thickness_grid=thickness_grid,
        cell_size_x=cell_size_x,
        cell_size_y=cell_size_y,
        elevation_grid=elevation_grid,
        porosity_grid=porosity_grid,
        time_step_in_days=time_step_in_days,
        gravitational_constant=gravitational_constant,
        water_compressibility_grid=water_compressibility_grid,
        gas_compressibility_grid=gas_compressibility_grid,
        rock_compressibility=rock_compressibility,
        well_indices_cache=well_indices_cache,
        injection_rates=injection_rates,
        production_rates=production_rates,
        dtype=dtype,
        pad_width=pad_width,
    )

    for iteration in range(max_newton_iterations):
        # Apply saturation boundary condition to ensure consistency between bghost and edge/boundary cells
        apply_saturation_boundary_conditions(
            padded_water_saturation_grid=water_saturation_grid,
            padded_oil_saturation_grid=oil_saturation_grid,
            padded_gas_saturation_grid=gas_saturation_grid,
            boundary_conditions=boundary_conditions,
            cell_dimension=cell_dimension,
            grid_shape=grid_shape,
            thickness_grid=thickness_grid,
            time=time,
            pad_width=pad_width,
        )
        # (Re-)compute all saturation-dependent properties once per iteration
        (
            _,
            capillary_pressure_grids,
            mobility_grids,
        ) = compute_rock_fluid_properties(
            water_saturation_grid=water_saturation_grid,
            oil_saturation_grid=oil_saturation_grid,
            gas_saturation_grid=gas_saturation_grid,
            rock_properties=rock_properties,
            fluid_properties=fluid_properties,
            config=config,
            # We only normalize for "numerical" Jacobian assembly, as perturbations may cause
            # saturation sum to exceed 1.0 sometimes
            normalize_saturations=config.jacobian_assembly_method == "numerical",
        )

        # Evaluate residual using the pre-computed properties
        water_residual, gas_residual = _compute_residual(
            water_saturation_grid=water_saturation_grid,
            gas_saturation_grid=gas_saturation_grid,
            capillary_pressure_grids=capillary_pressure_grids,
            mobility_grids=mobility_grids,
            **residual_kwargs,  # type: ignore[arg-type]
        )
        residual_vector = interleave_residuals(water_residual, gas_residual)
        residual_norm = np.linalg.norm(residual_vector)

        if iteration == 0:
            initial_residual_norm = max(residual_norm, 1e-30)
            logger.debug(f"Newton iteration 0: ||R0|| = {initial_residual_norm:.4e}")

        relative_residual_norm = residual_norm / initial_residual_norm

        residual_converged = relative_residual_norm < newton_tolerance and iteration > 0
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
            logger.debug(
                f"Newton converged at iteration {iteration} ({reason}): "
                f"||R||/||R0|| = {relative_residual_norm:.2e}, "
                f"max |∆S| = {last_max_ds:.2e}"
            )
            break

        # Assemble Jacobian
        jacobian = assemble_jacobian(
            config=config,
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
            thickness_grid=thickness_grid,
            cell_size_x=cell_size_x,
            cell_size_y=cell_size_y,
            elevation_grid=elevation_grid,
            porosity_grid=porosity_grid,
            time_step_in_days=time_step_in_days,
            gravitational_constant=gravitational_constant,
            water_compressibility_grid=water_compressibility_grid,
            gas_compressibility_grid=gas_compressibility_grid,
            rock_compressibility=rock_compressibility,
            well_indices_cache=well_indices_cache,
            injection_bhps=injection_bhps,
            production_bhps=production_bhps,
            injection_rates=injection_rates,
            production_rates=production_rates,
            pad_width=pad_width,
            dtype=dtype,
            capillary_pressure_grids=capillary_pressure_grids,
            mobility_grids=mobility_grids,
        )
        try:
            condition_number = spla.norm(jacobian, ord=2) * spla.norm(jacobian.T, ord=2)
            logger.debug(f"Jacobian condition number estimate: {condition_number:.3e}")
        except Exception as exc:  # noqa
            logger.debug(
                f"Error occured when computing Jacobian condition number. {exc}"
            )

        # Solve the linear system: J * dS = -R
        saturation_change, _ = solve_linear_system(
            A_csr=jacobian,
            b=-residual_vector,
            solver=config.saturation_solver,
            preconditioner=config.saturation_preconditioner,
            rtol=config.saturation_convergence_tolerance,
            max_iterations=config.max_iterations,
            fallback_to_direct=True,
        )

        linear_residual = jacobian @ saturation_change + residual_vector
        linear_residual_norm = np.linalg.norm(linear_residual)
        logger.debug(
            f"Linear solver residual: ||J*dS + R|| = {linear_residual_norm:.2e}"
        )

        # Damp Newton step
        max_raw_change = float(np.max(np.abs(saturation_change)))
        if max_raw_change > max_saturation_change:
            damping_factor = max_saturation_change / max_raw_change
            saturation_change = saturation_change * damping_factor
            logger.debug(
                f"Damped Newton step by {damping_factor:.3f} "
                f"(max |∆S| = {max_raw_change:.4f} > {max_saturation_change})"
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

        for ls_iteration in range(line_search_max_cuts):
            water_residual_trial, gas_residual_trial = compute_residual(
                water_saturation_grid=water_saturation_grid_trial,
                oil_saturation_grid=oil_saturation_grid_trial,
                gas_saturation_grid=gas_saturation_grid_trial,
                rock_properties=rock_properties,
                config=config,
                **residual_kwargs,  # type: ignore[arg-type]
            )
            residual_trial = interleave_residuals(
                water_residual=water_residual_trial,
                gas_residual=gas_residual_trial,
            )
            residual_trial_norm = np.linalg.norm(residual_trial)
            logger.debug(
                f"Line search iteration {ls_iteration}, alpha={line_search_factor:.4f}, "
                f"||R||={residual_trial_norm:.4e} vs ||R_base||={residual_norm:.4e}"
            )
            if residual_trial_norm < residual_norm:
                logger.debug(f"Line search: Accepted at alpha={line_search_factor:.4f}")
                break

            line_search_factor *= 0.5
            if (line_search_factor * max_raw_change) < min_step_size:
                logger.debug(
                    f"Line search hit precision floor at iteration {iteration}, "
                    f"alpha={line_search_factor}, max_dS={line_search_factor * max_raw_change:.2e}"
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
                residual_norm=residual_norm,  # type: ignore
                relative_residual_norm=relative_residual_norm,  # type: ignore
                max_saturation_update=max_saturation_update,
                line_search_factor=line_search_factor,
            )
        )

        logger.debug(
            f"Newton iteration {iteration}: "
            f"||R|| = {residual_norm:.2e}, "
            f"||R||/||R0|| = {relative_residual_norm:.2e}, "
            f"max |∆S| = {max_saturation_update:.2e}, "
            f"alpha = {line_search_factor:.3f}"
        )

        final_iteration = iteration + 1
        final_residual_norm = residual_norm

        if max_saturation_update < 1e-10:
            if relative_residual_norm < 1e-3:
                converged = True
                logger.debug(
                    f"Newton converged (saturation stagnation) at iteration {iteration}: "
                    f"max |∆S| = {max_saturation_update:.2e}, "
                    f"||R||/||R0|| = {relative_residual_norm:.2e}"
                )
            else:
                logger.warning(
                    f"Newton stagnated (negligible dS) at iteration {iteration}: "
                    f"max |∆S| = {max_saturation_update:.2e}, "
                    f"||R||/||R0|| = {relative_residual_norm:.2e}"
                )
            break

        if residual_norm < (
            best_residual_norm * (1.0 - stagnation_improvement_threshold)
        ):
            best_residual_norm = residual_norm
            stagnation_count = 0
        else:
            stagnation_count += 1

        if stagnation_count >= stagnation_patience and iteration >= stagnation_patience:
            if relative_residual_norm < 1e-3:
                converged = True
                logger.debug(
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
    grid_shape: ThreeDimensions,
    cell_dimension: typing.Tuple[float, float],
    thickness_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    time_step_size: float,
    time: float,
    rock_properties: RockProperties[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
    config: Config,
    well_indices_cache: WellIndicesCache,
    pressure_change_grid: ThreeDimensionalGrid,
    injection_rates: PhaseTensorsProxy[float, ThreeDimensions],
    production_rates: PhaseTensorsProxy[float, ThreeDimensions],
    injection_bhps: PhaseTensorsProxy[float, ThreeDimensions],
    production_bhps: PhaseTensorsProxy[float, ThreeDimensions],
    boundary_conditions: BoundaryConditions[ThreeDimensions],
    pad_width: int = 1,
) -> EvolutionResult[ImplicitSaturationSolution, typing.List[NewtonConvergenceInfo]]:
    """
    Solve the implicit saturation equations for a three-phase system.

    :param cell_dimension: (cell_size_x, cell_size_y) in feet.
    :param thickness_grid: 3D grid of cell thicknesses (ft).
    :param elevation_grid: 3D grid of cell elevations (ft).
    :param time_step: Current time step index (unused, kept for interface symmetry).
    :param time_step_size: Time step size in seconds.
    :param rock_properties: Rock properties (permeability, porosity, etc.).
    :param fluid_properties: Fluid properties at the new pressure level.
    :param wells: Well definitions.
    :param config: Simulation configuration.
    :param pressure_change_grid: P_new - P_old (psi) for PVT volume correction.
    :param pad_width: Ghost cell padding width.
    :return: `EvolutionResult` containing `ImplicitSaturationSolution`.
    """
    oil_pressure_grid = fluid_properties.pressure_grid
    cell_count_x, cell_count_y, cell_count_z = oil_pressure_grid.shape

    return solve_implicit_saturation(
        grid_shape=grid_shape,
        cell_dimension=cell_dimension,
        oil_pressure_grid=oil_pressure_grid,
        pressure_change_grid=pressure_change_grid,
        old_water_saturation_grid=fluid_properties.water_saturation_grid,
        old_oil_saturation_grid=fluid_properties.oil_saturation_grid,
        old_gas_saturation_grid=fluid_properties.gas_saturation_grid,
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
        thickness_grid=thickness_grid,
        elevation_grid=elevation_grid,
        porosity_grid=rock_properties.porosity_grid,
        time_step_size=time_step_size,
        time=time,
        rock_properties=rock_properties,
        fluid_properties=fluid_properties,
        config=config,
        well_indices_cache=well_indices_cache,
        injection_bhps=injection_bhps,
        production_bhps=production_bhps,
        injection_rates=injection_rates,
        production_rates=production_rates,
        boundary_conditions=boundary_conditions,
        water_compressibility_grid=fluid_properties.water_compressibility_grid,
        gas_compressibility_grid=fluid_properties.gas_compressibility_grid,
        rock_compressibility=rock_properties.compressibility,
        pad_width=pad_width,
        max_newton_iterations=config.max_newton_iterations,
        newton_tolerance=config.newton_tolerance,
        line_search_max_cuts=config.line_search_max_cuts,
        max_saturation_change=config.max_saturation_change,
        saturation_convergence_tolerance=config.saturation_convergence_tolerance,
    )
