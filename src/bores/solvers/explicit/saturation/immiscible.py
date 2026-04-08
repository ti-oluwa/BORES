import logging
import typing

import attrs
import numba
import numpy as np
import numpy.typing as npt

from bores.config import Config
from bores.constants import c
from bores.correlations.core import compute_harmonic_mean
from bores.datastructures import PhaseTensorsProxy
from bores.grids.base import CapillaryPressureGrids, RelativeMobilityGrids
from bores.grids.utils import unpad_grid
from bores.models import FluidProperties, RockProperties
from bores.precision import get_dtype
from bores.solvers.base import EvolutionResult
from bores.transmissibility import FaceTransmissibilities
from bores.types import OneDimensionalGrid, ThreeDimensionalGrid, ThreeDimensions
from bores.wells.indices import WellIndicesCache

__all__ = ["evolve_saturation"]

logger = logging.getLogger(__name__)


@attrs.frozen
class CFLMeta:
    cfl_threshold: float
    maximum_cfl_encountered: float
    cell: typing.Tuple[int, int, int]
    time_step: int
    violated: bool


@attrs.frozen
class FluxesMeta:
    total_water_inflow: float
    total_water_outflow: float
    total_oil_inflow: float
    total_oil_outflow: float
    total_gas_inflow: float
    total_gas_outflow: float
    total_inflow: float
    total_outflow: float


@attrs.frozen
class VolumesMeta:
    oil_volume: float
    water_volume: float
    gas_volume: float
    pore_volume: float


@attrs.frozen
class SaturationEvolutionMeta:
    cfl_info: CFLMeta
    fluxes: typing.Optional[FluxesMeta] = None
    volumes: typing.Optional[VolumesMeta] = None


@attrs.frozen
class ExplicitSaturationSolution:
    water_saturation_grid: ThreeDimensionalGrid
    oil_saturation_grid: ThreeDimensionalGrid
    gas_saturation_grid: ThreeDimensionalGrid
    maximum_cfl_encountered: float
    cfl_threshold: float
    maximum_oil_saturation_change: float
    maximum_water_saturation_change: float
    maximum_gas_saturation_change: float
    solvent_concentration_grid: typing.Optional[ThreeDimensionalGrid] = None


def evolve_saturation(
    cell_dimension: typing.Tuple[float, float],
    thickness_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    time_step: int,
    time_step_size: float,
    rock_properties: RockProperties[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
    relative_mobility_grids: RelativeMobilityGrids[ThreeDimensions],
    capillary_pressure_grids: CapillaryPressureGrids[ThreeDimensions],
    face_transmissibilities: FaceTransmissibilities,
    config: Config,
    well_indices_cache: WellIndicesCache,
    injection_rates: PhaseTensorsProxy[float, ThreeDimensions],
    production_rates: PhaseTensorsProxy[float, ThreeDimensions],
    pressure_change_grid: typing.Optional[ThreeDimensionalGrid] = None,
    pad_width: int = 1,
) -> EvolutionResult[ExplicitSaturationSolution, SaturationEvolutionMeta]:
    """
    Computes the new/updated saturation distribution for water, oil, and gas
    across the reservoir grid using an explicit upwind finite difference method.

    :param cell_dimension: Tuple representing the dimensions of each grid cell (cell_size_x, cell_size_y) in feet (ft).
    :param thickness_grid: N-Dimensional numpy array representing the height of each cell in the grid (ft).
    :param elevation_grid: N-Dimensional numpy array representing the elevation of each cell in the grid (ft).
    :param time_step: Current time step index (starting from 0).
    :param time_step_size: Time step duration in seconds for the simulation.
    :param rock_properties: `RockProperties` object containing rock physical properties.
    :param fluid_properties: `FluidProperties` object containing fluid physical properties,
        including current pressure and saturation grids.
    :param relative_mobility_grids: Tuple of relative mobility grids for (water, oil, gas)
    :param capillary_pressure_grids: Tuple of capillary pressure grids for (oil-water, gas-oil)
    :param config: Simulation config and parameters.
    :param well_indices_cache: Cache of well indices for efficient lookup during pressure solve.
    :param injection_rates: Optional `PhaseTensorsProxy` of injection rates for each phase and cell.
    :param production_rates: Optional `PhaseTensorsProxy` of production rates for each phase and cell.
    :param pressure_change_grid: Pressure change grid (P_new - P_old) in psi for PVT volume correction.
    :param pad_width: Number of ghost cells used for grid padding. Well coordinates are offset by this amount.
    :return: `EvolutionResult` containing updated saturations.
    """
    time_step_in_days = time_step_size * c.DAYS_PER_SECOND
    porosity_grid = rock_properties.porosity_grid
    oil_density_grid = fluid_properties.oil_effective_density_grid
    water_density_grid = fluid_properties.water_density_grid
    gas_density_grid = fluid_properties.gas_density_grid

    oil_pressure_grid = fluid_properties.pressure_grid
    current_water_saturation_grid = fluid_properties.water_saturation_grid
    current_oil_saturation_grid = fluid_properties.oil_saturation_grid
    current_gas_saturation_grid = fluid_properties.gas_saturation_grid

    water_compressibility_grid = fluid_properties.water_compressibility_grid
    oil_compressibility_grid = fluid_properties.oil_compressibility_grid
    gas_compressibility_grid = fluid_properties.gas_compressibility_grid

    # Determine grid dimensions and cell sizes
    cell_count_x, cell_count_y, cell_count_z = oil_pressure_grid.shape
    cell_size_x, cell_size_y = cell_dimension

    (
        water_relative_mobility_grid,
        oil_relative_mobility_grid,
        gas_relative_mobility_grid,
    ) = relative_mobility_grids
    oil_water_capillary_pressure_grid, gas_oil_capillary_pressure_grid = (
        capillary_pressure_grids
    )

    md_per_cp_to_ft2_per_psi_per_day = (
        c.MILLIDARCIES_PER_CENTIPOISE_TO_SQUARE_FEET_PER_PSI_PER_DAY
    )
    dtype = get_dtype()

    # Compute net flux contributions
    # Compute gravitational constant conversion factor (ft/s² * lbf·s²/(lbm·ft) = lbf/lbm)
    # On Earth, this should normally be 1.0 in consistent units, but we include it for clarity
    # and say the acceleration due to gravity was changed to 12.0 ft/s² for some reason (say g on Mars)
    # then the conversion factor would be 12.0 / 32.174 = 0.373. Which would scale the gravity terms accordingly.
    gravitational_constant = (
        c.ACCELERATION_DUE_TO_GRAVITY_FEET_PER_SECONDS_SQUARE
        / c.GRAVITATIONAL_CONSTANT_LBM_FT_PER_LBF_S2
    )

    if (pool := config.task_pool) is not None:
        fluxes_future = pool.submit(
            compute_net_flux_contributions,
            oil_pressure_grid=oil_pressure_grid,
            cell_count_x=cell_count_x,
            cell_count_y=cell_count_y,
            cell_count_z=cell_count_z,
            water_relative_mobility_grid=water_relative_mobility_grid,
            oil_relative_mobility_grid=oil_relative_mobility_grid,
            gas_relative_mobility_grid=gas_relative_mobility_grid,
            face_transmissibilities_x=face_transmissibilities.x,
            face_transmissibilities_y=face_transmissibilities.y,
            face_transmissibilities_z=face_transmissibilities.z,
            oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
            gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
            oil_density_grid=oil_density_grid,
            water_density_grid=water_density_grid,
            gas_density_grid=gas_density_grid,
            elevation_grid=elevation_grid,
            gravitational_constant=gravitational_constant,
            md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
            dtype=dtype,
        )
        well_rates_future = pool.submit(
            compute_well_rate_grids,
            cell_count_x=cell_count_x,
            cell_count_y=cell_count_y,
            cell_count_z=cell_count_z,
            well_indices_cache=well_indices_cache,
            injection_rates=injection_rates,
            production_rates=production_rates,
            dtype=dtype,
        )
        net_water_flux_grid, net_oil_flux_grid, net_gas_flux_grid = (
            fluxes_future.result()
        )
        net_water_well_rate_grid, net_oil_well_rate_grid, net_gas_well_rate_grid = (
            well_rates_future.result()
        )

    else:
        net_water_flux_grid, net_oil_flux_grid, net_gas_flux_grid = (
            compute_net_flux_contributions(
                oil_pressure_grid=oil_pressure_grid,
                cell_count_x=cell_count_x,
                cell_count_y=cell_count_y,
                cell_count_z=cell_count_z,
                water_relative_mobility_grid=water_relative_mobility_grid,
                oil_relative_mobility_grid=oil_relative_mobility_grid,
                gas_relative_mobility_grid=gas_relative_mobility_grid,
                face_transmissibilities_x=face_transmissibilities.x,
                face_transmissibilities_y=face_transmissibilities.y,
                face_transmissibilities_z=face_transmissibilities.z,
                oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                oil_density_grid=oil_density_grid,
                water_density_grid=water_density_grid,
                gas_density_grid=gas_density_grid,
                elevation_grid=elevation_grid,
                gravitational_constant=gravitational_constant,
                md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
                dtype=dtype,
            )
        )
        net_water_well_rate_grid, net_oil_well_rate_grid, net_gas_well_rate_grid = (
            compute_well_rate_grids(
                cell_count_x=cell_count_x,
                cell_count_y=cell_count_y,
                cell_count_z=cell_count_z,
                well_indices_cache=well_indices_cache,
                injection_rates=injection_rates,
                production_rates=production_rates,
                dtype=dtype,
                pad_width=pad_width,
            )
        )

    # Apply saturation updates with PVT volume correction
    (
        updated_water_saturation_grid,
        updated_oil_saturation_grid,
        updated_gas_saturation_grid,
        cfl_violation_info,
    ) = apply_updates(
        water_saturation_grid=current_water_saturation_grid,
        oil_saturation_grid=current_oil_saturation_grid,
        gas_saturation_grid=current_gas_saturation_grid,
        net_water_flux_grid=net_water_flux_grid,
        net_oil_flux_grid=net_oil_flux_grid,
        net_gas_flux_grid=net_gas_flux_grid,
        net_water_well_rate_grid=net_water_well_rate_grid,
        net_oil_well_rate_grid=net_oil_well_rate_grid,
        net_gas_well_rate_grid=net_gas_well_rate_grid,
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
        thickness_grid=thickness_grid,
        porosity_grid=porosity_grid,
        cell_size_x=cell_size_x,
        cell_size_y=cell_size_y,
        time_step_in_days=time_step_in_days,
        cfl_threshold=config.saturation_cfl_threshold,
        dtype=dtype,
        pressure_change_grid=pressure_change_grid,
        oil_compressibility_grid=oil_compressibility_grid,
        water_compressibility_grid=water_compressibility_grid,
        gas_compressibility_grid=gas_compressibility_grid,
        rock_compressibility=rock_properties.compressibility,
    )
    maximum_oil_saturation_change = np.max(
        np.abs(updated_oil_saturation_grid - current_oil_saturation_grid)
    )
    maximum_water_saturation_change = np.max(
        np.abs(updated_water_saturation_grid - current_water_saturation_grid)
    )
    maximum_gas_saturation_change = np.max(
        np.abs(updated_gas_saturation_grid - current_gas_saturation_grid)
    )

    # Check for CFL violations
    if cfl_violation_info[0] > 0.0:
        i, j, k = (
            int(cfl_violation_info[1]),
            int(cfl_violation_info[2]),
            int(cfl_violation_info[3]),
        )
        maximum_cfl_encountered = cfl_violation_info[4]
        cfl_threshold = cfl_violation_info[5]
        # Compute details for error message
        cell_thickness = thickness_grid[i, j, k]
        cell_total_volume = cell_size_x * cell_size_y * cell_thickness
        cell_porosity = porosity_grid[i, j, k]
        cell_pore_volume = cell_total_volume * cell_porosity

        # Get fluxes for this cell
        net_water_flux = net_water_flux_grid[i, j, k]
        net_oil_flux = net_oil_flux_grid[i, j, k]
        net_gas_flux = net_gas_flux_grid[i, j, k]
        net_water_well_rate = net_water_well_rate_grid[i, j, k]
        net_oil_well_rate = net_oil_well_rate_grid[i, j, k]
        net_gas_well_rate = net_gas_well_rate_grid[i, j, k]

        # Calculate total outflows for CFL check
        water_outflow_advection = abs(min(0.0, net_water_flux))
        oil_outflow_advection = abs(min(0.0, net_oil_flux))
        gas_outflow_advection = abs(min(0.0, net_gas_flux))

        water_outflow_well = abs(min(0.0, net_water_well_rate))
        oil_outflow_well = abs(min(0.0, net_oil_well_rate))
        gas_outflow_well = abs(min(0.0, net_gas_well_rate))

        water_inflow_advection = max(0.0, net_water_flux)
        oil_inflow_advection = max(0.0, net_oil_flux)
        gas_inflow_advection = max(0.0, net_gas_flux)

        water_inflow_well = max(0.0, net_water_well_rate)
        oil_inflow_well = max(0.0, net_oil_well_rate)
        gas_inflow_well = max(0.0, net_gas_well_rate)

        total_water_inflow = water_inflow_advection + water_inflow_well
        total_oil_inflow = oil_inflow_advection + oil_inflow_well
        total_gas_inflow = gas_inflow_advection + gas_inflow_well

        total_water_outflow = water_outflow_advection + water_outflow_well
        total_oil_outflow = oil_outflow_advection + oil_outflow_well
        total_gas_outflow = gas_outflow_advection + gas_outflow_well

        total_outflow = total_water_outflow + total_oil_outflow + total_gas_outflow
        total_inflow = total_water_inflow + total_oil_inflow + total_gas_inflow

        oil_saturation = current_oil_saturation_grid[i, j, k]
        water_saturation = current_water_saturation_grid[i, j, k]
        gas_saturation = current_gas_saturation_grid[i, j, k]

        oil_volume = cell_pore_volume * oil_saturation
        water_volume = cell_pore_volume * water_saturation
        gas_volume = cell_pore_volume * gas_saturation

        cell_oil_saturation_change = (
            oil_saturation - updated_oil_saturation_grid[i, j, k]
        )
        cell_water_saturation_change = (
            water_saturation - updated_water_saturation_grid[i, j, k]
        )
        cell_gas_saturation_change = (
            gas_saturation - updated_gas_saturation_grid[i, j, k]
        )

        # Pressure diagnostics at the CFL-violating cell
        cell_pressure = float(oil_pressure_grid[i, j, k])
        cell_bubble_point = float(
            fluid_properties.oil_bubble_point_pressure_grid[i, j, k]
        )
        pressure_state = (
            "undersaturated" if cell_pressure > cell_bubble_point else "saturated"
        )
        interior_pressure_grid = unpad_grid(oil_pressure_grid, pad_width=pad_width)
        avg_reservoir_pressure = float(np.mean(interior_pressure_grid))

        msg = f"""
        CFL condition violated at cell ({i - pad_width}, {j - pad_width}, {k - pad_width}) at timestep {time_step}:

        Max CFL number {maximum_cfl_encountered:.4f} exceeds limit {cfl_threshold:.4f}.

        Pressure diagnostics:
        Cell pressure = {cell_pressure:.2f} psi, Bubble point = {cell_bubble_point:.2f} psi ({pressure_state})
        Avg reservoir pressure = {avg_reservoir_pressure:.2f} psi

        Water Inflow = {total_water_inflow:.12f} ft³/day, Water Outflow = {total_water_outflow:.12f} ft³/day,
        Oil Inflow = {total_oil_inflow:.12f} ft³/day, Oil Outflow = {total_oil_outflow:.12f} ft³/day,
        Gas Inflow = {total_gas_inflow:.12f} ft³/day, Gas Outflow = {total_gas_outflow:.12f} ft³/day,
        Oil Volume = {oil_volume:.12f} ft³, Water Volume = {water_volume:.12f} ft³, Gas Volume = {gas_volume:.12f} ft³,
        Total Inflow = {total_inflow:.12f} ft³/day, Total Outflow = {total_outflow:.12f} ft³/day,
        Oil Saturation Change = {cell_oil_saturation_change}, Water Saturation Change = {cell_water_saturation_change},
        Gas Saturation Change = {cell_gas_saturation_change}, Pore volume = {cell_pore_volume:.12f} ft³.

        Consider reducing time step size from {time_step_size} seconds.
        """
        return EvolutionResult(
            success=False,
            value=ExplicitSaturationSolution(
                water_saturation_grid=updated_water_saturation_grid.astype(
                    dtype, copy=False
                ),
                oil_saturation_grid=updated_oil_saturation_grid.astype(
                    dtype, copy=False
                ),
                gas_saturation_grid=updated_gas_saturation_grid.astype(
                    dtype, copy=False
                ),
                maximum_cfl_encountered=maximum_cfl_encountered,
                cfl_threshold=cfl_threshold,
                maximum_oil_saturation_change=maximum_oil_saturation_change,
                maximum_water_saturation_change=maximum_water_saturation_change,
                maximum_gas_saturation_change=maximum_gas_saturation_change,
            ),
            scheme="explicit",
            message=msg,
            metadata=SaturationEvolutionMeta(
                cfl_info=CFLMeta(
                    cfl_threshold=cfl_threshold,
                    maximum_cfl_encountered=maximum_cfl_encountered,
                    cell=(i - pad_width, j - pad_width, k - pad_width),
                    time_step=time_step,
                    violated=True,
                ),
                fluxes=FluxesMeta(
                    total_water_inflow=total_water_inflow,
                    total_water_outflow=total_water_outflow,
                    total_oil_inflow=total_oil_inflow,
                    total_oil_outflow=total_oil_outflow,
                    total_gas_inflow=total_gas_inflow,
                    total_gas_outflow=total_gas_outflow,
                    total_inflow=total_inflow,
                    total_outflow=total_outflow,
                ),
                volumes=VolumesMeta(
                    oil_volume=oil_volume,
                    water_volume=water_volume,
                    gas_volume=gas_volume,
                    pore_volume=cell_pore_volume,
                ),
            ),
        )

    cfl_threshold = cfl_violation_info[5]
    maximum_cfl_encountered = cfl_violation_info[4]
    # Convert padded index to unpadded index
    cfl_i, cfl_j, cfl_k = (
        int(cfl_violation_info[1]) - pad_width,
        int(cfl_violation_info[2]) - pad_width,
        int(cfl_violation_info[3]) - pad_width,
    )
    return EvolutionResult(
        value=ExplicitSaturationSolution(
            water_saturation_grid=updated_water_saturation_grid.astype(
                dtype, copy=False
            ),
            oil_saturation_grid=updated_oil_saturation_grid.astype(dtype, copy=False),
            gas_saturation_grid=updated_gas_saturation_grid.astype(dtype, copy=False),
            maximum_cfl_encountered=maximum_cfl_encountered,
            cfl_threshold=cfl_threshold,
            maximum_oil_saturation_change=maximum_oil_saturation_change,
            maximum_water_saturation_change=maximum_water_saturation_change,
            maximum_gas_saturation_change=maximum_gas_saturation_change,
        ),
        scheme="explicit",
        success=True,
        metadata=SaturationEvolutionMeta(
            cfl_info=CFLMeta(
                cfl_threshold=cfl_threshold,
                maximum_cfl_encountered=maximum_cfl_encountered,
                cell=(cfl_i, cfl_j, cfl_k),
                time_step=time_step,
                violated=False,
            )
        ),
        message=f"Explicit saturation evolution time step {time_step} successful.",
    )


@numba.njit(cache=True, inline="always")
def compute_fluxes_from_neighbour(
    cell_indices: ThreeDimensions,
    neighbour_indices: ThreeDimensions,
    oil_pressure_grid: ThreeDimensionalGrid,
    face_transmissibility: float,
    water_relative_mobility_grid: ThreeDimensionalGrid,
    oil_relative_mobility_grid: ThreeDimensionalGrid,
    gas_relative_mobility_grid: ThreeDimensionalGrid,
    oil_water_capillary_pressure_grid: ThreeDimensionalGrid,
    gas_oil_capillary_pressure_grid: ThreeDimensionalGrid,
    oil_density_grid: ThreeDimensionalGrid,
    water_density_grid: ThreeDimensionalGrid,
    gas_density_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    gravitational_constant: float,
    md_per_cp_to_ft2_per_psi_per_day: float,
) -> typing.Tuple[float, float, float]:
    """
    Compute volumetric fluxes for water, oil, and gas phases between a cell and its neighbour.

    :param cell_indices: Tuple of indices (i, j, k) for the current cell.
    :param neighbour_indices: Tuple of indices (i, j, k) for the neighbouring cell.
    :param flow_area: Cross-sectional area for flow between the cells (ft²).
    :param flow_length: Distance between the centers of the two cells (ft).
    :param oil_pressure_grid: 3D grid of oil pressures (psi).
    :param water_relative_mobility_grid: 3D grid of water mobilities (ft²/psi.day).
    :param oil_relative_mobility_grid: 3D grid of oil mobilities (ft²/psi.day).
    :param gas_relative_mobility_grid: 3D grid of gas mobilities (ft²/psi.day).
    :param oil_water_capillary_pressure_grid: 3D grid of oil-water capillary pressures (psi).
    :param gas_oil_capillary_pressure_grid: 3D grid of gas-oil capillary pressures (psi).
    :param oil_density_grid: 3D grid of oil densities (lb/ft³).
    :param water_density_grid: 3D grid of water densities (lb/ft³).
    :param gas_density_grid: 3D grid of gas densities (lb/ft³).
    :param elevation_grid: 3D grid of cell elevations (ft).
    :param gravitational_constant: Gravitational constant conversion factor (lbf/lbm).
    :return: Tuple of volumetric fluxes (water_flux, oil_flux, gas_flux) in ft³/day.
    """
    # Calculate pressure differences relative to current cell (Neighbour - Current)
    # These represent the gradients driving flow from neighbour to currrent cell, or vice versa
    oil_pressure_difference = (
        oil_pressure_grid[neighbour_indices] - oil_pressure_grid[cell_indices]
    )
    oil_water_capillary_pressure_difference = (
        oil_water_capillary_pressure_grid[neighbour_indices]
        - oil_water_capillary_pressure_grid[cell_indices]
    )
    water_pressure_difference = (
        oil_pressure_difference - oil_water_capillary_pressure_difference
    )
    # Gas pressure difference is calculated as:
    gas_oil_capillary_pressure_difference = (
        gas_oil_capillary_pressure_grid[neighbour_indices]
        - gas_oil_capillary_pressure_grid[cell_indices]
    )
    gas_pressure_difference = (
        oil_pressure_difference + gas_oil_capillary_pressure_difference
    )

    # Calculate the elevation difference between the neighbour and current cell
    elevation_difference = (
        elevation_grid[neighbour_indices] - elevation_grid[cell_indices]
    )
    # # Determine the upwind densities based on pressure difference
    # # If pressure difference is positive (P_neighbour - P_current > 0), we use the neighbour's density
    # upwind_water_density = (
    #     water_density_grid[neighbour_indices]
    #     if water_pressure_difference > 0.0
    #     else water_density_grid[cell_indices]
    # )
    # upwind_oil_density = (
    #     oil_density_grid[neighbour_indices]
    #     if oil_pressure_difference > 0.0
    #     else oil_density_grid[cell_indices]
    # )
    # upwind_gas_density = (
    #     gas_density_grid[neighbour_indices]
    #     if gas_pressure_difference > 0.0
    #     else gas_density_grid[cell_indices]
    # )

    # Rank phases by density at the interface and select upwind accordingly
    # Water > Oil > Gas typically
    # Heavier phases upwind to the cell with higher density
    upwind_water_density = max(
        water_density_grid[neighbour_indices], water_density_grid[cell_indices]
    )
    upwind_gas_density = min(
        gas_density_grid[neighbour_indices], gas_density_grid[cell_indices]
    )
    upwind_oil_density = (
        oil_density_grid[neighbour_indices]
        if oil_density_grid[neighbour_indices] > oil_density_grid[cell_indices]
        else oil_density_grid[cell_indices]
    )

    # Computing the potential difference for the three phases
    water_gravity_potential = (
        upwind_water_density * gravitational_constant * elevation_difference
    ) / 144.0
    # Calculate the total water phase potential
    water_potential_difference = water_pressure_difference + water_gravity_potential

    oil_gravity_potential = (
        upwind_oil_density * gravitational_constant * elevation_difference
    ) / 144.0
    # Calculate the total oil phase potential
    oil_potential_difference = oil_pressure_difference + oil_gravity_potential

    gas_gravity_potential = (
        upwind_gas_density * gravitational_constant * elevation_difference
    ) / 144.0
    # Calculate the total gas phase potential
    gas_potential_difference = gas_pressure_difference + gas_gravity_potential

    upwind_water_relative_mobility = (
        water_relative_mobility_grid[neighbour_indices]
        if water_potential_difference > 0.0  # Flow from neighbour to cell
        else water_relative_mobility_grid[cell_indices]
    )
    upwind_oil_relative_mobility = (
        oil_relative_mobility_grid[neighbour_indices]
        if oil_potential_difference > 0.0
        else oil_relative_mobility_grid[cell_indices]
    )
    upwind_gas_relative_mobility = (
        gas_relative_mobility_grid[neighbour_indices]
        if gas_potential_difference > 0.0
        else gas_relative_mobility_grid[cell_indices]
    )

    # Compute volumetric fluxes at the face for each phase
    water_flux = (
        upwind_water_relative_mobility
        * water_potential_difference
        * face_transmissibility
        * md_per_cp_to_ft2_per_psi_per_day
    )
    oil_flux = (
        upwind_oil_relative_mobility
        * oil_potential_difference
        * face_transmissibility
        * md_per_cp_to_ft2_per_psi_per_day
    )
    gas_flux = (
        upwind_gas_relative_mobility
        * gas_potential_difference
        * face_transmissibility
        * md_per_cp_to_ft2_per_psi_per_day
    )
    return water_flux, oil_flux, gas_flux


@numba.njit(parallel=True, cache=True)
def compute_net_flux_contributions(
    oil_pressure_grid: ThreeDimensionalGrid,
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
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
    md_per_cp_to_ft2_per_psi_per_day: float,
    dtype: npt.DTypeLike,
) -> typing.Tuple[ThreeDimensionalGrid, ThreeDimensionalGrid, ThreeDimensionalGrid]:
    """
    Compute net flux contributions for all three phases using parallel loops.

    :param oil_pressure_grid: 3D grid of oil pressures (psi).
    :param cell_count_x: Number of cells in the x-direction.
    :param cell_count_y: Number of cells in the y-direction.
    :param cell_count_z: Number of cells in the z-direction.
    :param thickness_grid: 3D grid of cell thicknesses (ft).
    :param cell_size_x: Size of each cell in the x-direction (ft).
    :param cell_size_y: Size of each cell in the y-direction (ft).
    :param water_mobility_grid_x: 3D grid of water mobilities in x-direction (ft²/psi.day).
    :param water_mobility_grid_y: 3D grid of water mobilities in y-direction (ft²/psi.day).
    :param water_mobility_grid_z: 3D grid of water mobilities in z-direction (ft²/psi.day).
    :param oil_mobility_grid_x: 3D grid of oil mobilities in x-direction (ft²/psi.day).
    :param oil_mobility_grid_y: 3D grid of oil mobilities in y-direction (ft²/psi.day).
    :param oil_mobility_grid_z: 3D grid of oil mobilities in z-direction (ft²/psi.day).
    :param gas_mobility_grid_x: 3D grid of gas mobilities in x-direction (ft²/psi.day).
    :param gas_mobility_grid_y: 3D grid of gas mobilities in y-direction (ft²/psi.day).
    :param gas_mobility_grid_z: 3D grid of gas mobilities in z-direction (ft²/psi.day).
    :param oil_water_capillary_pressure_grid: 3D grid of oil-water capillary pressures (psi).
    :param gas_oil_capillary_pressure_grid: 3D grid of gas-oil capillary pressures (psi).
    :param oil_density_grid: 3D grid of oil densities (lb/ft³).
    :param water_density_grid: 3D grid of water densities (lb/ft³).
    :param gas_density_grid: 3D grid of gas densities (lb/ft³).
    :param elevation_grid: 3D grid of cell elevations (ft).
    :param gravitational_constant: Gravitational constant conversion factor (lbf/lbm).
    :param dtype: Numpy data type for computations.
    :return: (net_water_flux_grid, net_oil_flux_grid, net_gas_flux_grid) (ft³/day)
    """
    # Initialize flux grids
    net_water_flux_grid = np.zeros(
        (cell_count_x, cell_count_y, cell_count_z), dtype=dtype
    )
    net_oil_flux_grid = np.zeros(
        (cell_count_x, cell_count_y, cell_count_z), dtype=dtype
    )
    net_gas_flux_grid = np.zeros(
        (cell_count_x, cell_count_y, cell_count_z), dtype=dtype
    )

    # Parallel loop over interior cells
    for i in numba.prange(1, cell_count_x - 1):  # type: ignore
        for j in range(1, cell_count_y - 1):
            for k in range(1, cell_count_z - 1):
                # Initialize net fluxes for this cell
                net_water_flux = 0.0
                net_oil_flux = 0.0
                net_gas_flux = 0.0

                # X-direction fluxes (East and West neighbors)
                # East neighbor (i+1, j, k)
                water_flux, oil_flux, gas_flux = compute_fluxes_from_neighbour(
                    cell_indices=(i, j, k),
                    neighbour_indices=(i + 1, j, k),
                    oil_pressure_grid=oil_pressure_grid,
                    water_relative_mobility_grid=water_relative_mobility_grid,
                    oil_relative_mobility_grid=oil_relative_mobility_grid,
                    gas_relative_mobility_grid=gas_relative_mobility_grid,
                    face_transmissibility=face_transmissibilities_x[i, j, k],
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
                net_oil_flux += oil_flux
                net_gas_flux += gas_flux

                # West neighbor (i-1, j, k)
                water_flux, oil_flux, gas_flux = compute_fluxes_from_neighbour(
                    cell_indices=(i, j, k),
                    neighbour_indices=(i - 1, j, k),
                    oil_pressure_grid=oil_pressure_grid,
                    water_relative_mobility_grid=water_relative_mobility_grid,
                    oil_relative_mobility_grid=oil_relative_mobility_grid,
                    gas_relative_mobility_grid=gas_relative_mobility_grid,
                    face_transmissibility=face_transmissibilities_x[i - 1, j, k],
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
                net_oil_flux += oil_flux
                net_gas_flux += gas_flux

                # Y-direction fluxes (North and South neighbors)
                # North neighbor (i, j-1, k)
                water_flux, oil_flux, gas_flux = compute_fluxes_from_neighbour(
                    cell_indices=(i, j, k),
                    neighbour_indices=(i, j - 1, k),
                    oil_pressure_grid=oil_pressure_grid,
                    water_relative_mobility_grid=water_relative_mobility_grid,
                    oil_relative_mobility_grid=oil_relative_mobility_grid,
                    gas_relative_mobility_grid=gas_relative_mobility_grid,
                    face_transmissibility=face_transmissibilities_y[i, j - 1, k],
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
                net_oil_flux += oil_flux
                net_gas_flux += gas_flux

                # South neighbor (i, j+1, k)
                water_flux, oil_flux, gas_flux = compute_fluxes_from_neighbour(
                    cell_indices=(i, j, k),
                    neighbour_indices=(i, j + 1, k),
                    oil_pressure_grid=oil_pressure_grid,
                    water_relative_mobility_grid=water_relative_mobility_grid,
                    oil_relative_mobility_grid=oil_relative_mobility_grid,
                    gas_relative_mobility_grid=gas_relative_mobility_grid,
                    face_transmissibility=face_transmissibilities_y[i, j, k],
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
                net_oil_flux += oil_flux
                net_gas_flux += gas_flux

                # Z-direction fluxes (Top and Bottom neighbors)
                water_flux, oil_flux, gas_flux = compute_fluxes_from_neighbour(
                    cell_indices=(i, j, k),
                    neighbour_indices=(i, j, k - 1),
                    oil_pressure_grid=oil_pressure_grid,
                    water_relative_mobility_grid=water_relative_mobility_grid,
                    oil_relative_mobility_grid=oil_relative_mobility_grid,
                    gas_relative_mobility_grid=gas_relative_mobility_grid,
                    face_transmissibility=face_transmissibilities_z[i, j, k - 1],
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
                net_oil_flux += oil_flux
                net_gas_flux += gas_flux

                # Bottom neighbor (i, j, k+1)
                water_flux, oil_flux, gas_flux = compute_fluxes_from_neighbour(
                    cell_indices=(i, j, k),
                    neighbour_indices=(i, j, k + 1),
                    oil_pressure_grid=oil_pressure_grid,
                    water_relative_mobility_grid=water_relative_mobility_grid,
                    oil_relative_mobility_grid=oil_relative_mobility_grid,
                    gas_relative_mobility_grid=gas_relative_mobility_grid,
                    face_transmissibility=face_transmissibilities_z[i, j, k],
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
                net_oil_flux += oil_flux
                net_gas_flux += gas_flux

                # Store net fluxes for this cell
                net_water_flux_grid[i, j, k] = net_water_flux
                net_oil_flux_grid[i, j, k] = net_oil_flux
                net_gas_flux_grid[i, j, k] = net_gas_flux

    return net_water_flux_grid, net_oil_flux_grid, net_gas_flux_grid


def compute_well_rate_grids(
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
    well_indices_cache: WellIndicesCache,
    injection_rates: PhaseTensorsProxy[float, ThreeDimensions],
    production_rates: PhaseTensorsProxy[float, ThreeDimensions],
    dtype: npt.DTypeLike,
    pad_width: int = 1,
) -> typing.Tuple[ThreeDimensionalGrid, ThreeDimensionalGrid, ThreeDimensionalGrid]:
    """
    Compute well rates for all cells (injection + production).

    :param cell_count_x: Number of cells in the x-direction.
    :param cell_count_y: Number of cells in the y-direction.
    :param cell_count_z: Number of cells in the z-direction.
    :param well_indices_cache: Cache of well indices for efficient lookup during pressure solve.
    :param injection_rates: Optional `PhaseTensorsProxy` of injection rates for each phase and cell.
    :param production_rates: Optional `PhaseTensorsProxy` of production rates for each phase and cell.
    :param dtype: Numpy data type for computations.
    :return: (net_water_well_rate_grid, net_oil_well_rate_grid, net_gas_well_rate_grid) (ft³/day)
    """
    # Initialize well rate grids
    net_water_well_rate_grid = np.zeros(
        (cell_count_x, cell_count_y, cell_count_z), dtype=dtype
    )
    net_oil_well_rate_grid = np.zeros(
        (cell_count_x, cell_count_y, cell_count_z), dtype=dtype
    )
    net_gas_well_rate_grid = np.zeros(
        (cell_count_x, cell_count_y, cell_count_z), dtype=dtype
    )

    # Update net grids
    for well_indices in well_indices_cache.injection.values():
        for perforation_index in well_indices:
            i, j, k = perforation_index.cell
            water_rate, _, gas_rate = injection_rates[
                i - pad_width, j - pad_width, k - pad_width
            ]
            net_water_well_rate_grid[i, j, k] += water_rate
            net_gas_well_rate_grid[i, j, k] += gas_rate

    for well_indices in well_indices_cache.production.values():
        for perforation_index in well_indices:
            i, j, k = perforation_index.cell
            water_rate, oil_rate, gas_rate = production_rates[
                i - pad_width, j - pad_width, k - pad_width
            ]
            net_water_well_rate_grid[i, j, k] += water_rate
            net_oil_well_rate_grid[i, j, k] += oil_rate
            net_gas_well_rate_grid[i, j, k] += gas_rate

    return net_water_well_rate_grid, net_oil_well_rate_grid, net_gas_well_rate_grid


@numba.njit(parallel=True, cache=True)
def apply_updates(
    water_saturation_grid: ThreeDimensionalGrid,
    oil_saturation_grid: ThreeDimensionalGrid,
    gas_saturation_grid: ThreeDimensionalGrid,
    net_water_flux_grid: ThreeDimensionalGrid,
    net_oil_flux_grid: ThreeDimensionalGrid,
    net_gas_flux_grid: ThreeDimensionalGrid,
    net_water_well_rate_grid: ThreeDimensionalGrid,
    net_oil_well_rate_grid: ThreeDimensionalGrid,
    net_gas_well_rate_grid: ThreeDimensionalGrid,
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
    thickness_grid: ThreeDimensionalGrid,
    porosity_grid: ThreeDimensionalGrid,
    cell_size_x: float,
    cell_size_y: float,
    time_step_in_days: float,
    cfl_threshold: float,
    dtype: npt.DTypeLike,
    pressure_change_grid: typing.Optional[ThreeDimensionalGrid] = None,
    oil_compressibility_grid: typing.Optional[ThreeDimensionalGrid] = None,
    water_compressibility_grid: typing.Optional[ThreeDimensionalGrid] = None,
    gas_compressibility_grid: typing.Optional[ThreeDimensionalGrid] = None,
    rock_compressibility: float = 0.0,
) -> typing.Tuple[
    ThreeDimensionalGrid, ThreeDimensionalGrid, ThreeDimensionalGrid, OneDimensionalGrid
]:
    """
    Apply saturation updates with CFL criteria checking and PVT volume correction.

    In IMPES, the pressure equation includes compressibility so the saturation
    transport must too. If not, we would creates a volume balance mismatch where So + Sw + Sg
    drifts from 1.0 over time. The PVT correction accounts for fluid expansion /
    contraction and pore volume change due to the pressure change:

        ΔSα_pvt = Sα x (cα + cf) x (-ΔP)

    This ensures each phase's saturation properly reflects both transport and
    pressure-driven volume changes, maintaining So + Sw + Sg ≈ 1.0 without
    artificial proportional normalization that would inflate immobile phases.

    :param water_saturation_grid: 3D grid of water saturations.
    :param oil_saturation_grid: 3D grid of oil saturations.
    :param gas_saturation_grid: 3D grid of gas saturations.
    :param net_water_flux_grid: 3D grid of net water fluxes (ft³/day).
    :param net_oil_flux_grid: 3D grid of net oil fluxes (ft³/day).
    :param net_gas_flux_grid: 3D grid of net gas fluxes (ft³/day).
    :param net_water_well_rate_grid: 3D grid of net water well rates (ft³/day).
    :param net_oil_well_rate_grid: 3D grid of net oil well rates (ft³/day).
    :param net_gas_well_rate_grid: 3D grid of net gas well rates (ft³/day).
    :param cell_count_x: Number of cells in the x-direction.
    :param cell_count_y: Number of cells in the y-direction.
    :param cell_count_z: Number of cells in the z-direction.
    :param thickness_grid: 3D grid of cell thicknesses (ft).
    :param porosity_grid: 3D grid of cell porosities (fraction).
    :param cell_size_x: Size of each cell in the x-direction (ft).
    :param cell_size_y: Size of each cell in the y-direction (ft).
    :param time_step_in_days: Time step size in days.
    :param cfl_threshold: Maximum allowed CFL number.
    :param dtype: Numpy data type for computations.
    :param pressure_change_grid: 3D grid of pressure changes (P_new - P_old) in psi.
        When provided along with compressibility grids, PVT volume correction is applied.
    :param oil_compressibility_grid: 3D grid of oil compressibilities (psi⁻¹).
    :param water_compressibility_grid: 3D grid of water compressibilities (psi⁻¹).
    :param gas_compressibility_grid: 3D grid of gas compressibilities (psi⁻¹).
    :param rock_compressibility: Scalar rock (pore) compressibility (psi⁻¹).
    :return: Tuple of (updated_water_sat, updated_oil_sat, updated_gas_sat, cfl_violation_info)
    where `cfl_violation_info` is array [violated (bool), i, j, k, cfl_number, maximum_cfl]
    """
    apply_pvt_correction = (
        pressure_change_grid is not None
        and oil_compressibility_grid is not None
        and water_compressibility_grid is not None
        and gas_compressibility_grid is not None
    )
    # Initialize updated saturation grids
    updated_water_saturation_grid = water_saturation_grid.copy()
    updated_oil_saturation_grid = oil_saturation_grid.copy()
    updated_gas_saturation_grid = gas_saturation_grid.copy()

    # CFL violation tracking: [violated, i, j, k, cfl_number, maximum_cfl]
    cfl_violation_info = np.zeros(6, dtype=dtype)
    maximum_cfl_encountered = 0.0

    # Parallel loop over interior cells
    for i in numba.prange(1, cell_count_x - 1):  # type: ignore
        for j in range(1, cell_count_y - 1):
            for k in range(1, cell_count_z - 1):
                cell_thickness = thickness_grid[i, j, k]
                cell_total_volume = cell_size_x * cell_size_y * cell_thickness
                cell_porosity = porosity_grid[i, j, k]
                cell_pore_volume = cell_total_volume * cell_porosity

                # Get net fluxes and well rates
                net_water_flux = net_water_flux_grid[i, j, k]
                net_oil_flux = net_oil_flux_grid[i, j, k]
                net_gas_flux = net_gas_flux_grid[i, j, k]

                net_water_well_rate = net_water_well_rate_grid[i, j, k]
                net_oil_well_rate = net_oil_well_rate_grid[i, j, k]
                net_gas_well_rate = net_gas_well_rate_grid[i, j, k]

                # Calculate total outflows for CFL check
                water_outflow_advection = abs(min(0.0, net_water_flux))
                oil_outflow_advection = abs(min(0.0, net_oil_flux))
                gas_outflow_advection = abs(min(0.0, net_gas_flux))

                water_outflow_well = abs(min(0.0, net_water_well_rate))
                oil_outflow_well = abs(min(0.0, net_oil_well_rate))
                gas_outflow_well = abs(min(0.0, net_gas_well_rate))

                total_water_outflow = water_outflow_advection + water_outflow_well
                total_oil_outflow = oil_outflow_advection + oil_outflow_well
                total_gas_outflow = gas_outflow_advection + gas_outflow_well

                total_outflow = (
                    total_water_outflow + total_oil_outflow + total_gas_outflow
                )

                # CFL check
                cfl_number = (total_outflow * time_step_in_days) / cell_pore_volume
                if cfl_number > cfl_threshold and cfl_number > maximum_cfl_encountered:
                    # Record max CFL encountered
                    cfl_violation_info[0] = 1.0  # violated flag
                    cfl_violation_info[1] = float(i)
                    cfl_violation_info[2] = float(j)
                    cfl_violation_info[3] = float(k)
                    cfl_violation_info[4] = cfl_number
                    cfl_violation_info[5] = cfl_threshold
                    maximum_cfl_encountered = cfl_number

                # Calculate saturation changes
                oil_saturation_change = (
                    (net_oil_flux + net_oil_well_rate)
                    * time_step_in_days
                    / cell_pore_volume
                )
                water_saturation_change = (
                    (net_water_flux + net_water_well_rate)
                    * time_step_in_days
                    / cell_pore_volume
                )
                gas_saturation_change = (
                    (net_gas_flux + net_gas_well_rate)
                    * time_step_in_days
                    / cell_pore_volume
                )

                # Transport-based saturation update
                old_water_saturation = water_saturation_grid[i, j, k]
                old_oil_saturation = oil_saturation_grid[i, j, k]
                old_gas_saturation = gas_saturation_grid[i, j, k]

                new_water_saturation = old_water_saturation + water_saturation_change
                new_oil_saturation = old_oil_saturation + oil_saturation_change
                new_gas_saturation = old_gas_saturation + gas_saturation_change

                # PVT volume correction: accounts for fluid expansion/contraction
                # and pore volume change due to pressure change.
                # ΔSα_pvt = Sα_old x (cα + cf) x (-ΔP)
                # When pressure decreases: fluids expand, pore volume contracts so Sα increases.
                # When pressure increases: fluids contract, pore volume expands so Sα decreases.
                if apply_pvt_correction:
                    delta_pressure = pressure_change_grid[i, j, k]  # type: ignore[index]
                    negative_delta_pressure = -delta_pressure

                    new_oil_saturation += (
                        old_oil_saturation
                        * (
                            oil_compressibility_grid[i, j, k] + rock_compressibility  # type: ignore[index]
                        )
                        * negative_delta_pressure
                    )
                    new_water_saturation += (
                        old_water_saturation
                        * (
                            water_compressibility_grid[i, j, k] + rock_compressibility  # type: ignore[index]
                        )
                        * negative_delta_pressure
                    )
                    new_gas_saturation += (
                        old_gas_saturation
                        * (
                            gas_compressibility_grid[i, j, k] + rock_compressibility  # type: ignore[index]
                        )
                        * negative_delta_pressure
                    )

                # Clamp negative saturations
                if new_water_saturation < 0.0:
                    new_water_saturation = 0.0
                if new_oil_saturation < 0.0:
                    new_oil_saturation = 0.0
                if new_gas_saturation < 0.0:
                    new_gas_saturation = 0.0

                # Residual volume balance correction: apply any remaining tiny
                # numerical gap to oil (the dominant phase) rather than
                # proportional normalization which would inflate immobile phases.
                total_saturation = (
                    new_water_saturation + new_oil_saturation + new_gas_saturation
                )
                if abs(total_saturation - 1.0) > 1e-12:
                    new_oil_saturation += 1.0 - total_saturation
                    if new_oil_saturation < 0.0:
                        new_oil_saturation = 0.0

                updated_water_saturation_grid[i, j, k] = new_water_saturation
                updated_oil_saturation_grid[i, j, k] = new_oil_saturation
                updated_gas_saturation_grid[i, j, k] = new_gas_saturation

    return (
        updated_water_saturation_grid,
        updated_oil_saturation_grid,
        updated_gas_saturation_grid,
        cfl_violation_info,
    )


"""
Explicit finite difference formulation for saturation transport in a 3D reservoir
(immiscible three-phase flow: oil, water, and gas with slightly compressible fluids):

The governing equation for saturation evolution is conservation of mass per phase:

    ∂S_x/∂t * (φ * V_cell) = Σ_faces [F_x_face] + q_x * V_cell

Where:
    ∂S_x/∂t * φ * V_cell = Accumulation term (change in phase volume) (ft³/day)
    Σ_faces [F_x_face] = Net advective flux from all 6 neighbours (ft³/day)
    q_x * V_cell = Source/sink term for the phase (injection/production) (ft³/day)

Phase Flux Computation (per-phase Darcy velocity, not fractional flow):

    Each phase has its own Darcy velocity driven by its own potential gradient:

    v_x_phase = λ_phase · ∂Φ_phase/∂dir / ΔL_dir    (ft/day)
    F_x_face = v_x_phase · A_face                      (ft³/day)

    where:
    λ_phase = k · kr_phase(S_upwind) / μ_phase         (ft²/psi·day)
    Φ_phase = P_phase + ρ_phase · (g/gc) · Δelevation / 144   (psi)

    Phase pressures include capillary corrections:
        P_water = P_oil - P_cow
        P_gas   = P_oil + P_cgo

    This per-phase formulation correctly handles capillary-driven counter-current flow,
    where different phases can flow in opposite directions at the same face.

Upwind Selection:

    Two-stage upwinding:
    1. Density upwinding: ρ_upwind selected based on pressure difference sign
       (used in gravity potential computation)
    2. Mobility upwinding: λ_upwind selected based on total phase potential sign
       (used in Darcy velocity computation)

    Convention: positive flux = flow from neighbour into current cell.

Gravity:

    gravity_potential_phase = ρ_upwind_phase · (g/gc) · Δelevation / 144   (psi)

    where:
    g/gc = 32.174 / 32.174 = 1.0 (dimensionless in consistent imperial units)
    Δelevation = elevation_neighbour - elevation_current (ft)
    144 converts lbf/ft² to psi

    Gravity drives lighter phases (gas) upward and heavier phases (water, oil) downward.

Discretization:

Time: Forward Euler
    ∂S/∂t ≈ (Sⁿ⁺¹_ijk - Sⁿ_ijk) / Δt

Space: First-order upwind scheme

    Sⁿ⁺¹_ijk = Sⁿ_ijk + Δt / (φ · V_cell) · [
        Σ_faces [F_phase_face] + q_phase_ijk · V_cell
    ]

Variables:
    Sⁿ_ijk = saturation at cell (i,j,k) at time step n
    Sⁿ⁺¹_ijk = updated saturation
    φ = porosity
    Δx, Δy = cell dimensions in x, y (ft)
    h_face = harmonic mean of adjacent cell thicknesses (ft)
    A_x = Δy · h_face; A_y = Δx · h_face; A_z = Δx · Δy (face areas, ft²)
    ΔL = flow length between cell centres (ft)
    q_phase_ijk = phase well rate (ft³/day)

Assumptions:
- Darcy flow
- No dispersion or diffusion (purely advective; miscible version adds optional diffusion)
- Saturation-dependent relative permeability (Corey, Brooks-Corey, LET, etc.)
- Time step satisfies CFL condition
- Pressure field is computed before solving saturation (IMPES splitting)

Stability (CFL) condition:

    max(|v_x|/Δx + |v_y|/Δy + |v_z|/Δz) · Δt / φ ≤ 1

Notes:
- Pressure field must be computed before solving saturation.
- Upwind selection is based on phase potential, not pressure alone.
- All three phase transport equations (water, oil, gas) are solved independently.
  A residual volume balance correction is applied to oil to enforce So + Sw + Sg = 1.
  (This differs from the implicit saturation solver, which solves only Sw and Sg
  and derives So = 1 - Sw - Sg from the constraint.)
"""
