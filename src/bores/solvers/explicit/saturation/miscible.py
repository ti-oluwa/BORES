import logging
import typing

import attrs
import numba
import numpy as np
import numpy.typing as npt

from bores._precision import get_dtype
from bores.config import Config
from bores.constants import c
from bores.correlations.core import compute_harmonic_mean
from bores.datastructures import PhaseTensorsProxy
from bores.grids.base import CapillaryPressureGrids, RelativeMobilityGrids
from bores.grids.utils import unpad_grid
from bores.models import FluidProperties, RockProperties
from bores.solvers.base import (
    EvolutionResult,
    _warn_injection_rate,
    _warn_production_rate,
    compute_mobility_grids,
)
from bores.types import (
    FluidPhase,
    OneDimensionalGrid,
    ThreeDimensionalGrid,
    ThreeDimensions,
)
from bores.utils import clip
from bores.wells.base import Wells
from bores.wells.controls import CoupledRateControl
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
    time: float,
    rock_properties: RockProperties[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
    relative_mobility_grids: RelativeMobilityGrids[ThreeDimensions],
    capillary_pressure_grids: CapillaryPressureGrids[ThreeDimensions],
    wells: Wells[ThreeDimensions],
    config: Config,
    well_indices_cache: WellIndicesCache,
    injection_rates: PhaseTensorsProxy[float, ThreeDimensions],
    production_rates: PhaseTensorsProxy[float, ThreeDimensions],
    pressure_change_grid: typing.Optional[ThreeDimensionalGrid] = None,
    pad_width: int = 1,
) -> EvolutionResult[ExplicitSaturationSolution, SaturationEvolutionMeta]:
    """
    Evolve saturations explicitly with Todd-Longstaff miscible displacement.

    Solvent (e.g., CO2) can exist as:
    1. Free gas phase (tracked by `gas_saturation`)
    2. Dissolved in oil (tracked by `solvent_concentration` in oil)

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
    :return: `EvolutionResult` containing updated saturations and solvent concentration
    """
    absolute_permeability = rock_properties.absolute_permeability
    porosity_grid = rock_properties.porosity_grid

    oil_pressure_grid = fluid_properties.pressure_grid
    current_water_saturation_grid = fluid_properties.water_saturation_grid
    current_oil_saturation_grid = fluid_properties.oil_saturation_grid
    current_gas_saturation_grid = fluid_properties.gas_saturation_grid
    current_solvent_concentration_grid = fluid_properties.solvent_concentration_grid

    oil_density_grid = fluid_properties.oil_effective_density_grid
    water_density_grid = fluid_properties.water_density_grid
    gas_density_grid = fluid_properties.gas_density_grid

    water_compressibility_grid = fluid_properties.water_compressibility_grid
    oil_compressibility_grid = fluid_properties.oil_compressibility_grid
    gas_compressibility_grid = fluid_properties.gas_compressibility_grid

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

    dtype = get_dtype()

    # Compute mobility grids for x, y, z directions
    (
        (water_mobility_grid_x, oil_mobility_grid_x, gas_mobility_grid_x),
        (water_mobility_grid_y, oil_mobility_grid_y, gas_mobility_grid_y),
        (water_mobility_grid_z, oil_mobility_grid_z, gas_mobility_grid_z),
    ) = compute_mobility_grids(
        absolute_permeability_x=absolute_permeability.x,
        absolute_permeability_y=absolute_permeability.y,
        absolute_permeability_z=absolute_permeability.z,
        water_relative_mobility_grid=water_relative_mobility_grid,
        oil_relative_mobility_grid=oil_relative_mobility_grid,
        gas_relative_mobility_grid=gas_relative_mobility_grid,
        md_per_cp_to_ft2_per_psi_per_day=c.MILLIDARCIES_PER_CENTIPOISE_TO_SQUARE_FEET_PER_PSI_PER_DAY,
    )

    # Compute net flux contributions for all cells
    # Compute gravitational constant conversion factor (ft/s² * lbf·s²/(lbm·ft) = lbf/lbm)
    # On Earth, this should normally be 1.0 in consistent units, but we include it for clarity
    # and say the acceleration due to gravity was changed to 12.0 ft/s² for some reason (say g on Mars)
    # then the conversion factor would be 12.0 / 32.174 = 0.373. Which would scale the gravity terms accordingly.
    gravitational_constant = (
        c.ACCELERATION_DUE_TO_GRAVITY_FEET_PER_SECONDS_SQUARE
        / c.GRAVITATIONAL_CONSTANT_LBM_FT_PER_LBF_S2
    )
    temperature_grid = fluid_properties.temperature_grid
    if (pool := config.task_pool) is not None:
        fluxes_future = pool.submit(
            compute_net_flux_contributions,
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
            solvent_concentration_grid=current_solvent_concentration_grid,
            oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
            gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
            oil_density_grid=oil_density_grid,
            water_density_grid=water_density_grid,
            gas_density_grid=gas_density_grid,
            elevation_grid=elevation_grid,
            gravitational_constant=gravitational_constant,
            dtype=dtype,
        )
        well_rates_future = pool.submit(
            compute_well_rate_grids,
            cell_count_x=cell_count_x,
            cell_count_y=cell_count_y,
            cell_count_z=cell_count_z,
            wells=wells,
            oil_pressure_grid=oil_pressure_grid,
            temperature_grid=temperature_grid,
            water_relative_mobility_grid=water_relative_mobility_grid,
            oil_relative_mobility_grid=oil_relative_mobility_grid,
            gas_relative_mobility_grid=gas_relative_mobility_grid,
            water_compressibility_grid=water_compressibility_grid,
            oil_compressibility_grid=oil_compressibility_grid,
            gas_compressibility_grid=gas_compressibility_grid,
            fluid_properties=fluid_properties,
            time=time,
            config=config,
            well_indices_cache=well_indices_cache,
            pad_width=pad_width,
            injection_rates=injection_rates,
            production_rates=production_rates,
            dtype=dtype,
        )
        (
            net_water_flux_grid,
            net_oil_flux_grid,
            net_gas_flux_grid,
            net_solvent_flux_grid,
        ) = fluxes_future.result()
        (
            net_water_well_rate_grid,
            net_oil_well_rate_grid,
            net_gas_well_rate_grid,
            solvent_injection_concentration_grid,
            gas_injection_rate_grid,
        ) = well_rates_future.result()

    else:
        (
            net_water_flux_grid,
            net_oil_flux_grid,
            net_gas_flux_grid,
            net_solvent_flux_grid,
        ) = compute_net_flux_contributions(
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
            solvent_concentration_grid=current_solvent_concentration_grid,
            oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
            gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
            oil_density_grid=oil_density_grid,
            water_density_grid=water_density_grid,
            gas_density_grid=gas_density_grid,
            elevation_grid=elevation_grid,
            gravitational_constant=gravitational_constant,
            dtype=dtype,
        )
        (
            net_water_well_rate_grid,
            net_oil_well_rate_grid,
            net_gas_well_rate_grid,
            solvent_injection_concentration_grid,
            gas_injection_rate_grid,
        ) = compute_well_rate_grids(
            cell_count_x=cell_count_x,
            cell_count_y=cell_count_y,
            cell_count_z=cell_count_z,
            wells=wells,
            oil_pressure_grid=oil_pressure_grid,
            temperature_grid=temperature_grid,
            water_relative_mobility_grid=water_relative_mobility_grid,
            oil_relative_mobility_grid=oil_relative_mobility_grid,
            gas_relative_mobility_grid=gas_relative_mobility_grid,
            water_compressibility_grid=water_compressibility_grid,
            oil_compressibility_grid=oil_compressibility_grid,
            gas_compressibility_grid=gas_compressibility_grid,
            fluid_properties=fluid_properties,
            time=time,
            config=config,
            well_indices_cache=well_indices_cache,
            pad_width=pad_width,
            injection_rates=injection_rates,
            production_rates=production_rates,
            dtype=dtype,
        )

    # Apply saturation and solvent concentration updates
    cfl_threshold = config.saturation_cfl_threshold
    time_step_in_days = time_step_size * c.DAYS_PER_SECOND
    (
        updated_water_saturation_grid,
        updated_oil_saturation_grid,
        updated_gas_saturation_grid,
        updated_solvent_concentration_grid,
        cfl_violation_info,
    ) = apply_updates(
        water_saturation_grid=current_water_saturation_grid,
        oil_saturation_grid=current_oil_saturation_grid,
        gas_saturation_grid=current_gas_saturation_grid,
        solvent_concentration_grid=current_solvent_concentration_grid,
        net_water_flux_grid=net_water_flux_grid,
        net_oil_flux_grid=net_oil_flux_grid,
        net_gas_flux_grid=net_gas_flux_grid,
        net_solvent_flux_grid=net_solvent_flux_grid,
        net_water_well_rate_grid=net_water_well_rate_grid,
        net_oil_well_rate_grid=net_oil_well_rate_grid,
        net_gas_well_rate_grid=net_gas_well_rate_grid,
        solvent_injection_concentration_grid=solvent_injection_concentration_grid,
        gas_injection_rate_grid=gas_injection_rate_grid,
        porosity_grid=porosity_grid,
        thickness_grid=thickness_grid,
        cell_size_x=cell_size_x,
        cell_size_y=cell_size_y,
        time_step_in_days=time_step_in_days,
        cfl_threshold=cfl_threshold,
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

    # Check for CFL violation
    if cfl_violation_info[0] > 0:
        i, j, k = (
            int(cfl_violation_info[1]),
            int(cfl_violation_info[2]),
            int(cfl_violation_info[3]),
        )
        maximum_cfl_encountered = cfl_violation_info[4]
        cfl_threshold = cfl_violation_info[5]
        cell_pore_volume = (
            porosity_grid[i, j, k] * cell_size_x * cell_size_y * thickness_grid[i, j, k]
        )
        # Compute total inflow/outflow for error message
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

        oil_saturation = current_oil_saturation_grid[i, j, k]
        water_saturation = current_water_saturation_grid[i, j, k]
        gas_saturation = current_gas_saturation_grid[i, j, k]

        oil_volume = cell_pore_volume * oil_saturation
        water_volume = cell_pore_volume * water_saturation
        gas_volume = cell_pore_volume * gas_saturation

        total_inflow = total_water_inflow + total_oil_inflow + total_gas_inflow
        total_outflow = total_water_outflow + total_oil_outflow + total_gas_outflow

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
                solvent_concentration_grid=updated_solvent_concentration_grid.astype(
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
    # Convert padded index to unpadded index
    cfl_i, cfl_j, cfl_k = (
        int(cfl_violation_info[1]) - pad_width,
        int(cfl_violation_info[2]) - pad_width,
        int(cfl_violation_info[3]) - pad_width,
    )
    maximum_cfl_encountered = cfl_violation_info[4]
    return EvolutionResult(
        value=ExplicitSaturationSolution(
            water_saturation_grid=updated_water_saturation_grid.astype(
                dtype, copy=False
            ),
            oil_saturation_grid=updated_oil_saturation_grid.astype(dtype, copy=False),
            gas_saturation_grid=updated_gas_saturation_grid.astype(dtype, copy=False),
            solvent_concentration_grid=updated_solvent_concentration_grid.astype(
                dtype, copy=False
            ),
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
            ),
        ),
        message=f"Explicit saturation evolution time step {time_step} successful.",
    )


@numba.njit(cache=True)
def compute_fluxes_from_neighbour(
    cell_indices: ThreeDimensions,
    neighbour_indices: ThreeDimensions,
    flow_area: float,
    flow_length: float,
    oil_pressure_grid: ThreeDimensionalGrid,
    water_mobility_grid: ThreeDimensionalGrid,
    oil_mobility_grid: ThreeDimensionalGrid,
    gas_mobility_grid: ThreeDimensionalGrid,
    solvent_concentration_grid: ThreeDimensionalGrid,
    oil_water_capillary_pressure_grid: ThreeDimensionalGrid,
    gas_oil_capillary_pressure_grid: ThreeDimensionalGrid,
    oil_density_grid: ThreeDimensionalGrid,
    water_density_grid: ThreeDimensionalGrid,
    gas_density_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    gravitational_constant: float,
) -> typing.Tuple[float, float, float, float]:  # water, oil, gas, solvent_in_oil
    """
    Compute phase volumetric fluxes including solvent concentration transport.

    The solvent flux is the volumetric flux of dissolved solvent
    moving with the oil phase (ft³/day * concentration).

    :param cell_indices: Indices of the current cell (i, j, k).
    :param neighbour_indices: Indices of the neighbouring cell (i, j, k).
    :param flow_area: Flow area between the two cells (ft²).
    :param flow_length: Flow length between the two cells (ft).
    :param oil_pressure_grid: 3D grid of oil pressures (psi).
    :param water_mobility_grid: 3D grid of water mobilities (ft²/psi.day).
    :param oil_mobility_grid: 3D grid of oil mobilities (ft²/psi.day).
    :param gas_mobility_grid: 3D grid of gas mobilities (ft²/psi.day).
    :param solvent_concentration_grid: 3D grid of solvent concentrations (fraction).
    :param oil_water_capillary_pressure_grid: 3D grid of oil-water capillary pressures (psi).
    :param gas_oil_capillary_pressure_grid: 3D grid of gas-oil capillary pressures (psi).
    :param oil_density_grid: 3D grid of oil densities (lb/ft³).
    :param water_density_grid: 3D grid of water densities (lb/ft³).
    :param gas_density_grid: 3D grid of gas densities (lb/ft³).
    :param elevation_grid: 3D grid of cell elevations (ft).
    :param gravitational_constant: Gravitational constant conversion factor (lbf/lbm).
    :return: Tuple of (water_flux, oil_flux, gas_flux, solvent_flux_in_oil)
        1. `water_flux`: Volumetric flux of water (ft³/day).
        2. `oil_flux`: Volumetric flux of oil (ft³/day).
        3. `gas_flux`: Volumetric flux of gas (ft³/day).
        4. `solvent_flux_in_oil`: Volumetric flux of solvent in oil phase (ft³/day).
    """
    # Calculate pressure differences relative to current cell (Neighbour - Current)
    # These represent the gradients driving flow from Neighbour to currrent cell, or vice versa
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
    gas_oil_capillary_pressure_difference = (
        gas_oil_capillary_pressure_grid[neighbour_indices]
        - gas_oil_capillary_pressure_grid[cell_indices]
    )
    gas_pressure_difference = (
        oil_pressure_difference + gas_oil_capillary_pressure_difference
    )
    cell_solvent_concentration = solvent_concentration_grid[cell_indices]
    neighbour_solvent_concentration = solvent_concentration_grid[neighbour_indices]

    elevation_difference = (
        elevation_grid[neighbour_indices] - elevation_grid[cell_indices]
    )

    # Upwind densities
    upwind_water_density = (
        water_density_grid[neighbour_indices]
        if water_pressure_difference > 0.0
        else water_density_grid[cell_indices]
    )
    upwind_oil_density = (
        oil_density_grid[neighbour_indices]
        if oil_pressure_difference > 0.0
        else oil_density_grid[cell_indices]
    )
    upwind_gas_density = (
        gas_density_grid[neighbour_indices]
        if gas_pressure_difference > 0.0
        else gas_density_grid[cell_indices]
    )

    # Darcy velocities with gravity
    water_gravity_potential = (
        upwind_water_density * gravitational_constant * elevation_difference
    ) / 144.0
    water_potential_difference = water_pressure_difference + water_gravity_potential

    oil_gravity_potential = (
        upwind_oil_density * gravitational_constant * elevation_difference
    ) / 144.0
    oil_potential_difference = oil_pressure_difference + oil_gravity_potential

    gas_gravity_potential = (
        upwind_gas_density * gravitational_constant * elevation_difference
    ) / 144.0
    gas_potential_difference = gas_pressure_difference + gas_gravity_potential

    upwind_water_mobility = (
        water_mobility_grid[neighbour_indices]
        if water_potential_difference > 0.0  # Flow from neighbour to cell
        else water_mobility_grid[cell_indices]
    )
    upwind_oil_mobility = (
        oil_mobility_grid[neighbour_indices]
        if oil_potential_difference > 0.0
        else oil_mobility_grid[cell_indices]
    )
    upwind_gas_mobility = (
        gas_mobility_grid[neighbour_indices]
        if gas_potential_difference > 0.0
        else gas_mobility_grid[cell_indices]
    )

    water_velocity = upwind_water_mobility * water_potential_difference / flow_length
    oil_velocity = upwind_oil_mobility * oil_potential_difference / flow_length
    gas_velocity = upwind_gas_mobility * gas_potential_difference / flow_length

    # Upwind solvent concentration (moves with oil)
    upwinded_solvent_concentration = (
        neighbour_solvent_concentration
        if oil_velocity > 0
        else cell_solvent_concentration
    )

    # Volumetric fluxes (ft³/day) = velocity * area
    water_flux_at_face = water_velocity * flow_area
    oil_flux_at_face = oil_velocity * flow_area
    gas_flux_at_face = gas_velocity * flow_area

    # Solvent mass flux in oil phase (ft³/day)
    # The solvent concentration travels with the oil phase
    solvent_flux_in_oil = oil_flux_at_face * upwinded_solvent_concentration
    return (water_flux_at_face, oil_flux_at_face, gas_flux_at_face, solvent_flux_in_oil)


@numba.njit(parallel=True, cache=True)
def compute_net_flux_contributions(
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
    solvent_concentration_grid: ThreeDimensionalGrid,
    oil_water_capillary_pressure_grid: ThreeDimensionalGrid,
    gas_oil_capillary_pressure_grid: ThreeDimensionalGrid,
    oil_density_grid: ThreeDimensionalGrid,
    water_density_grid: ThreeDimensionalGrid,
    gas_density_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    gravitational_constant: float,
    dtype: npt.DTypeLike,
) -> typing.Tuple[
    ThreeDimensionalGrid,
    ThreeDimensionalGrid,
    ThreeDimensionalGrid,
    ThreeDimensionalGrid,
]:
    """
    Compute net volumetric flux contributions for miscible flow (4 phases: water, oil, gas, solvent).

    :param oil_pressure_grid: 3D grid of current oil pressures (psi).
    :param cell_count_x: Number of cells in x-direction.
    :param cell_count_y: Number of cells in y-direction.
    :param cell_count_z: Number of cells in z-direction.
    :param thickness_grid: 3D grid of cell thicknesses (ft).
    :param cell_size_x: Cell size in x-direction (ft).
    :param cell_size_y: Cell size in y-direction (ft).
    :param water_mobility_grid_x: 3D grid of water mobilities in x-direction (ft²/psi.day).
    :param water_mobility_grid_y: 3D grid of water mobilities in y-direction (ft²/psi.day).
    :param water_mobility_grid_z: 3D grid of water mobilities in z-direction (ft²/psi.day).
    :param oil_mobility_grid_x: 3D grid of oil mobilities in x-direction (ft²/psi.day).
    :param oil_mobility_grid_y: 3D grid of oil mobilities in y-direction (ft²/psi.day).
    :param oil_mobility_grid_z: 3D grid of oil mobilities in z-direction (ft²/psi.day).
    :param gas_mobility_grid_x: 3D grid of gas mobilities in x-direction (ft²/psi.day).
    :param gas_mobility_grid_y: 3D grid of gas mobilities in y-direction (ft²/psi.day).
    :param gas_mobility_grid_z: 3D grid of gas mobilities in z-direction (ft²/psi.day).
    :param solvent_concentration_grid: 3D grid of solvent concentrations (fraction).
    :param oil_water_capillary_pressure_grid: 3D grid of oil-water capillary pressures (psi).
    :param gas_oil_capillary_pressure_grid: 3D grid of gas-oil capillary pressures (psi).
    :param oil_density_grid: 3D grid of oil densities (lb/ft³).
    :param water_density_grid: 3D grid of water densities (lb/ft³).
    :param gas_density_grid: 3D grid of gas densities (lb/ft³
    :param elevation_grid: 3D grid of cell elevations (ft).
    :param gravitational_constant: Gravitational constant conversion factor (lbf/lbm).
    :param dtype: Numpy data type for computations.
    :return: Tuple of net flux grids:
        1. `net_water_flux_grid`: 3D grid of net water fluxes (ft³/day).
        2. `net_oil_flux_grid`: 3D grid of net oil fluxes (ft³/day).
        3. `net_gas_flux_grid`: 3D grid of net gas fluxes (ft³/day).
        4. `net_solvent_flux_grid`: 3D grid of net solvent fluxes (ft³/day).
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
    net_solvent_flux_grid = np.zeros(
        (cell_count_x, cell_count_y, cell_count_z), dtype=dtype
    )

    # Parallel loop over interior cells
    for i in numba.prange(1, cell_count_x - 1):  # type: ignore
        for j in range(1, cell_count_y - 1):
            for k in range(1, cell_count_z - 1):
                cell_thickness = thickness_grid[i, j, k]

                # Initialize net fluxes for this cell
                net_water_flux = 0.0
                net_oil_flux = 0.0
                net_gas_flux = 0.0
                net_solvent_flux = 0.0

                # X-direction fluxes (East and West neighbors)
                flow_length_x = cell_size_x

                # East neighbor (i+1, j, k)
                east_neighbour_thickness = thickness_grid[i + 1, j, k]
                harmonic_thickness = compute_harmonic_mean(
                    cell_thickness, east_neighbour_thickness
                )
                east_flow_area = cell_size_y * harmonic_thickness
                water_flux, oil_flux, gas_flux, solvent_flux = (
                    compute_fluxes_from_neighbour(
                        cell_indices=(i, j, k),
                        neighbour_indices=(i + 1, j, k),
                        flow_area=east_flow_area,
                        flow_length=flow_length_x,
                        oil_pressure_grid=oil_pressure_grid,
                        water_mobility_grid=water_mobility_grid_x,
                        oil_mobility_grid=oil_mobility_grid_x,
                        gas_mobility_grid=gas_mobility_grid_x,
                        solvent_concentration_grid=solvent_concentration_grid,
                        oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                        gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                        oil_density_grid=oil_density_grid,
                        water_density_grid=water_density_grid,
                        gas_density_grid=gas_density_grid,
                        elevation_grid=elevation_grid,
                        gravitational_constant=gravitational_constant,
                    )
                )
                net_water_flux += water_flux
                net_oil_flux += oil_flux
                net_gas_flux += gas_flux
                net_solvent_flux += solvent_flux

                # West neighbor (i-1, j, k)
                west_neighbour_thickness = thickness_grid[i - 1, j, k]
                harmonic_thickness = compute_harmonic_mean(
                    cell_thickness, west_neighbour_thickness
                )
                west_flow_area = cell_size_y * harmonic_thickness
                water_flux, oil_flux, gas_flux, solvent_flux = (
                    compute_fluxes_from_neighbour(
                        cell_indices=(i, j, k),
                        neighbour_indices=(i - 1, j, k),
                        flow_area=west_flow_area,
                        flow_length=flow_length_x,
                        oil_pressure_grid=oil_pressure_grid,
                        water_mobility_grid=water_mobility_grid_x,
                        oil_mobility_grid=oil_mobility_grid_x,
                        gas_mobility_grid=gas_mobility_grid_x,
                        solvent_concentration_grid=solvent_concentration_grid,
                        oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                        gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                        oil_density_grid=oil_density_grid,
                        water_density_grid=water_density_grid,
                        gas_density_grid=gas_density_grid,
                        elevation_grid=elevation_grid,
                        gravitational_constant=gravitational_constant,
                    )
                )
                net_water_flux += water_flux
                net_oil_flux += oil_flux
                net_gas_flux += gas_flux
                net_solvent_flux += solvent_flux

                # Y-direction fluxes (North and South neighbors)
                flow_length_y = cell_size_y

                # North neighbor (i, j-1, k)
                north_neighbour_thickness = thickness_grid[i, j - 1, k]
                harmonic_thickness = compute_harmonic_mean(
                    cell_thickness, north_neighbour_thickness
                )
                north_flow_area = cell_size_x * harmonic_thickness
                water_flux, oil_flux, gas_flux, solvent_flux = (
                    compute_fluxes_from_neighbour(
                        cell_indices=(i, j, k),
                        neighbour_indices=(i, j - 1, k),
                        flow_area=north_flow_area,
                        flow_length=flow_length_y,
                        oil_pressure_grid=oil_pressure_grid,
                        water_mobility_grid=water_mobility_grid_y,
                        oil_mobility_grid=oil_mobility_grid_y,
                        gas_mobility_grid=gas_mobility_grid_y,
                        solvent_concentration_grid=solvent_concentration_grid,
                        oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                        gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                        oil_density_grid=oil_density_grid,
                        water_density_grid=water_density_grid,
                        gas_density_grid=gas_density_grid,
                        elevation_grid=elevation_grid,
                        gravitational_constant=gravitational_constant,
                    )
                )
                net_water_flux += water_flux
                net_oil_flux += oil_flux
                net_gas_flux += gas_flux
                net_solvent_flux += solvent_flux

                # South neighbor (i, j+1, k)
                south_neighbour_thickness = thickness_grid[i, j + 1, k]
                harmonic_thickness = compute_harmonic_mean(
                    cell_thickness, south_neighbour_thickness
                )
                south_flow_area = cell_size_x * harmonic_thickness
                water_flux, oil_flux, gas_flux, solvent_flux = (
                    compute_fluxes_from_neighbour(
                        cell_indices=(i, j, k),
                        neighbour_indices=(i, j + 1, k),
                        flow_area=south_flow_area,
                        flow_length=flow_length_y,
                        oil_pressure_grid=oil_pressure_grid,
                        water_mobility_grid=water_mobility_grid_y,
                        oil_mobility_grid=oil_mobility_grid_y,
                        gas_mobility_grid=gas_mobility_grid_y,
                        solvent_concentration_grid=solvent_concentration_grid,
                        oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                        gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                        oil_density_grid=oil_density_grid,
                        water_density_grid=water_density_grid,
                        gas_density_grid=gas_density_grid,
                        elevation_grid=elevation_grid,
                        gravitational_constant=gravitational_constant,
                    )
                )
                net_water_flux += water_flux
                net_oil_flux += oil_flux
                net_gas_flux += gas_flux
                net_solvent_flux += solvent_flux

                # Z-direction fluxes (Top and Bottom neighbors)
                flow_area_z = cell_size_x * cell_size_y

                # Note: For vertical flow, the flow area is simply the cell cross-sectional area
                # But the flow length is the harmonic mean of the two cell thicknesses

                # Top neighbor (i, j, k-1)
                top_neighbour_thickness = thickness_grid[i, j, k - 1]
                harmonic_thickness = compute_harmonic_mean(
                    cell_thickness, top_neighbour_thickness
                )
                top_flow_length = harmonic_thickness
                water_flux, oil_flux, gas_flux, solvent_flux = (
                    compute_fluxes_from_neighbour(
                        cell_indices=(i, j, k),
                        neighbour_indices=(i, j, k - 1),
                        flow_area=flow_area_z,
                        flow_length=top_flow_length,
                        oil_pressure_grid=oil_pressure_grid,
                        water_mobility_grid=water_mobility_grid_z,
                        oil_mobility_grid=oil_mobility_grid_z,
                        gas_mobility_grid=gas_mobility_grid_z,
                        solvent_concentration_grid=solvent_concentration_grid,
                        oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                        gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                        oil_density_grid=oil_density_grid,
                        water_density_grid=water_density_grid,
                        gas_density_grid=gas_density_grid,
                        elevation_grid=elevation_grid,
                        gravitational_constant=gravitational_constant,
                    )
                )
                net_water_flux += water_flux
                net_oil_flux += oil_flux
                net_gas_flux += gas_flux
                net_solvent_flux += solvent_flux

                # Bottom neighbor (i, j, k+1)
                bottom_neighbour_thickness = thickness_grid[i, j, k + 1]
                harmonic_thickness = compute_harmonic_mean(
                    cell_thickness, bottom_neighbour_thickness
                )
                bottom_flow_length = harmonic_thickness
                water_flux, oil_flux, gas_flux, solvent_flux = (
                    compute_fluxes_from_neighbour(
                        cell_indices=(i, j, k),
                        neighbour_indices=(i, j, k + 1),
                        flow_area=flow_area_z,
                        flow_length=bottom_flow_length,
                        oil_pressure_grid=oil_pressure_grid,
                        water_mobility_grid=water_mobility_grid_z,
                        oil_mobility_grid=oil_mobility_grid_z,
                        gas_mobility_grid=gas_mobility_grid_z,
                        solvent_concentration_grid=solvent_concentration_grid,
                        oil_water_capillary_pressure_grid=oil_water_capillary_pressure_grid,
                        gas_oil_capillary_pressure_grid=gas_oil_capillary_pressure_grid,
                        oil_density_grid=oil_density_grid,
                        water_density_grid=water_density_grid,
                        gas_density_grid=gas_density_grid,
                        elevation_grid=elevation_grid,
                        gravitational_constant=gravitational_constant,
                    )
                )
                net_water_flux += water_flux
                net_oil_flux += oil_flux
                net_gas_flux += gas_flux
                net_solvent_flux += solvent_flux

                # Store net fluxes for this cell
                net_water_flux_grid[i, j, k] = net_water_flux
                net_oil_flux_grid[i, j, k] = net_oil_flux
                net_gas_flux_grid[i, j, k] = net_gas_flux
                net_solvent_flux_grid[i, j, k] = net_solvent_flux

    return (
        net_water_flux_grid,
        net_oil_flux_grid,
        net_gas_flux_grid,
        net_solvent_flux_grid,
    )


def compute_well_rate_grids(
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
    wells: Wells[ThreeDimensions],
    oil_pressure_grid: ThreeDimensionalGrid,
    temperature_grid: ThreeDimensionalGrid,
    water_relative_mobility_grid: ThreeDimensionalGrid,
    oil_relative_mobility_grid: ThreeDimensionalGrid,
    gas_relative_mobility_grid: ThreeDimensionalGrid,
    water_compressibility_grid: ThreeDimensionalGrid,
    oil_compressibility_grid: ThreeDimensionalGrid,
    gas_compressibility_grid: ThreeDimensionalGrid,
    fluid_properties: FluidProperties[ThreeDimensions],
    time: float,
    config: Config,
    well_indices_cache: WellIndicesCache,
    dtype: npt.DTypeLike,
    injection_rates: PhaseTensorsProxy[float, ThreeDimensions],
    production_rates: PhaseTensorsProxy[float, ThreeDimensions],
    pad_width: int = 1,
) -> typing.Tuple[
    ThreeDimensionalGrid,
    ThreeDimensionalGrid,
    ThreeDimensionalGrid,
    ThreeDimensionalGrid,
    ThreeDimensionalGrid,
]:
    """
    Compute well rate contributions for each cell in the miscible case.

    This function handles injection and production wells and cannot be JIT compiled
    because it relies on well object methods.

    :param cell_count_x: Number of cells in x direction
    :param cell_count_y: Number of cells in y direction
    :param cell_count_z: Number of cells in z direction
    :param wells: Wells object containing injection and production wells
    :param oil_pressure_grid: Oil pressure values (psi)
    :param temperature_grid: Temperature values (°R)
    :param absolute_permeability: Absolute permeability object with x, y, z components (md)
    :param water_relative_mobility_grid: Water relative mobility (md/cP)
    :param oil_relative_mobility_grid: Oil relative mobility (md/cP)
    :param gas_relative_mobility_grid: Gas relative mobility (md/cP)
    :param water_compressibility_grid: Water compressibility values (psi⁻¹)
    :param oil_compressibility_grid: Oil compressibility values (psi⁻¹)
    :param gas_compressibility_grid: Gas compressibility values (psi⁻¹)
    :param fluid_properties: Fluid properties object containing FVF grids
    :param thickness_grid: Cell thickness values (ft)
    :param cell_size_x: Cell size in x direction (ft)
    :param cell_size_y: Cell size in y direction (ft)
    :param time_step: Current time step number
    :param time_step_size: Time step size (seconds)
    :param config: Evolution config
    :param injection_grid: Optional grid to store injection rates (ft³/day)
    :param production_grid: Optional grid to store production rates (ft³/day)
    :param dtype: Data type for output arrays
    :param pad_width: Number of ghost cells used for grid padding. Well coordinates are offset by this amount.
    :return: Tuple of (net_water_well_rate_grid, net_oil_well_rate_grid, net_gas_well_rate_grid,
        solvent_injection_concentration_grid, gas_injection_rate_grid)
        where rates are in ft³/day (volumetric rates) and concentration is dimensionless.
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
    solvent_injection_concentration_grid = np.zeros(
        (cell_count_x, cell_count_y, cell_count_z), dtype=dtype
    )
    gas_injection_rate_grid = np.zeros(
        (cell_count_x, cell_count_y, cell_count_z), dtype=dtype
    )
    bbl_to_ft3 = c.BARRELS_TO_CUBIC_FEET

    # Process all injection wells (compute total WI + rates in single pass)
    for well in wells.injection_wells:
        if not well.is_open or well.injected_fluid is None:
            continue

        injected_fluid = well.injected_fluid
        injected_phase = injected_fluid.phase
        use_pseudo_pressure = (
            config.use_pseudo_pressure and injected_phase == FluidPhase.GAS
        )

        water_bubble_point_pressure_grid = (
            fluid_properties.water_bubble_point_pressure_grid
        )
        gas_formation_volume_factor_grid = (
            fluid_properties.gas_formation_volume_factor_grid
        )
        gas_solubility_in_water_grid = fluid_properties.gas_solubility_in_water_grid

        well_indices = well_indices_cache.injection[well.name]
        # Compute rates for each perforated cell using cached well indices
        for perforation_index in well_indices:
            i, j, k = perforation_index.cell
            well_index = perforation_index.well_index
            allocation_fraction = well_indices.allocation_fraction(perforation_index)
            cell_temperature = typing.cast(float, temperature_grid[i, j, k])
            cell_oil_pressure = typing.cast(float, oil_pressure_grid[i, j, k])

            phase_fvf = injected_fluid.get_formation_volume_factor(
                pressure=cell_oil_pressure,
                temperature=cell_temperature,
            )
            phase_fvf = typing.cast(float, phase_fvf)

            if injected_phase == FluidPhase.GAS:
                phase_mobility = typing.cast(float, gas_relative_mobility_grid[i, j, k])
                compressibility_kwargs = {}
            else:
                phase_mobility = typing.cast(
                    float, water_relative_mobility_grid[i, j, k]
                )
                compressibility_kwargs = {
                    "bubble_point_pressure": water_bubble_point_pressure_grid[i, j, k],
                    "gas_formation_volume_factor": gas_formation_volume_factor_grid[
                        i, j, k
                    ],
                    "gas_solubility_in_water": gas_solubility_in_water_grid[i, j, k],
                }

            phase_compressibility = injected_fluid.get_compressibility(
                pressure=cell_oil_pressure,
                temperature=cell_temperature,
                **compressibility_kwargs,
            )
            phase_compressibility = typing.cast(float, phase_compressibility)

            # Injection wells use total mobility since injected fluid
            # displaces all existing phases in the cell
            total_mobility = (
                water_relative_mobility_grid[i, j, k]
                + oil_relative_mobility_grid[i, j, k]
                + gas_relative_mobility_grid[i, j, k]
            )
            effective_mobility = typing.cast(float, total_mobility)

            cell_injection_rate = well.get_flow_rate(
                pressure=cell_oil_pressure,
                temperature=cell_temperature,
                well_index=well_index,
                phase_mobility=effective_mobility,
                fluid=injected_fluid,
                fluid_compressibility=phase_compressibility,
                use_pseudo_pressure=use_pseudo_pressure,
                formation_volume_factor=phase_fvf,
                allocation_fraction=allocation_fraction,
                pvt_tables=None,
            )

            if cell_injection_rate < 0.0 and config.warn_well_anomalies:
                _warn_injection_rate(
                    injection_rate=cell_injection_rate,
                    well_name=well.name,
                    cell=(i - pad_width, j - pad_width, k - pad_width),
                    time=time,
                    rate_unit="ft³/day"
                    if injected_phase == FluidPhase.GAS
                    else "bbls/day",
                )

            # Handle miscible solvent injection
            cell_gas_injection_rate = 0.0
            cell_water_injection_rate = 0.0
            cell_oil_injection_rate = 0.0
            cell_solvent_injection_concentration = 0.0

            if injected_phase == FluidPhase.GAS and injected_fluid.is_miscible:
                # Miscible solvent injection (e.g., CO2)
                cell_gas_injection_rate = cell_injection_rate  # ft³/day
                # This will be mixed with existing oil in the mass balance
                cell_solvent_injection_concentration = injected_fluid.concentration

            elif injected_phase == FluidPhase.GAS:
                # Non-miscible gas injection
                cell_gas_injection_rate = cell_injection_rate

            else:  # Water injection
                cell_water_injection_rate = cell_injection_rate * bbl_to_ft3

            # Update net grids
            net_water_well_rate_grid[i, j, k] += cell_water_injection_rate
            net_gas_well_rate_grid[i, j, k] += cell_gas_injection_rate
            solvent_injection_concentration_grid[i, j, k] = (
                cell_solvent_injection_concentration
            )
            gas_injection_rate_grid[i, j, k] = cell_gas_injection_rate

    # Process all production wells (compute total WI + rates in single pass)
    for well in wells.production_wells:
        if not well.is_open:
            continue

        is_couple_controlled = isinstance(well.control, CoupledRateControl)
        well_indices = well_indices_cache.production[well.name]
        # Compute rates for each perforated cell using cached well indices
        for perforation_index in well_indices:
            i, j, k = perforation_index.cell
            well_index = perforation_index.well_index
            allocation_fraction = well_indices.allocation_fraction(perforation_index)
            cell_temperature = typing.cast(float, temperature_grid[i, j, k])
            cell_oil_pressure = typing.cast(float, oil_pressure_grid[i, j, k])

            water_formation_volume_factor_grid = (
                fluid_properties.water_formation_volume_factor_grid
            )
            oil_formation_volume_factor_grid = (
                fluid_properties.oil_formation_volume_factor_grid
            )
            gas_formation_volume_factor_grid = (
                fluid_properties.gas_formation_volume_factor_grid
            )

            cell_water_production_rate = 0.0
            cell_oil_production_rate = 0.0
            cell_gas_production_rate = 0.0

            # Build primary phase context if using CoupledRateControl
            primary_phase_context = {}
            if is_couple_controlled:
                primary_phase_context = well.control.build_primary_phase_context(  # type: ignore
                    produced_fluids=well.produced_fluids,
                    oil_mobility=typing.cast(
                        float, oil_relative_mobility_grid[i, j, k]
                    ),
                    water_mobility=typing.cast(
                        float, water_relative_mobility_grid[i, j, k]
                    ),
                    gas_mobility=typing.cast(
                        float, gas_relative_mobility_grid[i, j, k]
                    ),
                    oil_fvf=typing.cast(
                        float, oil_formation_volume_factor_grid[i, j, k]
                    ),
                    water_fvf=typing.cast(
                        float, water_formation_volume_factor_grid[i, j, k]
                    ),
                    gas_fvf=typing.cast(
                        float, gas_formation_volume_factor_grid[i, j, k]
                    ),
                    oil_compressibility=typing.cast(
                        float, oil_compressibility_grid[i, j, k]
                    ),
                    water_compressibility=typing.cast(
                        float, water_compressibility_grid[i, j, k]
                    ),
                    gas_compressibility=typing.cast(
                        float, gas_compressibility_grid[i, j, k]
                    ),
                )

            for produced_fluid in well.produced_fluids:
                produced_phase = produced_fluid.phase
                if produced_phase == FluidPhase.GAS:
                    phase_mobility = typing.cast(
                        float, gas_relative_mobility_grid[i, j, k]
                    )
                    fluid_compressibility = typing.cast(
                        float, gas_compressibility_grid[i, j, k]
                    )
                    fluid_formation_volume_factor = typing.cast(
                        float, gas_formation_volume_factor_grid[i, j, k]
                    )
                elif produced_phase == FluidPhase.WATER:
                    phase_mobility = typing.cast(
                        float, water_relative_mobility_grid[i, j, k]
                    )
                    fluid_compressibility = typing.cast(
                        float, water_compressibility_grid[i, j, k]
                    )
                    fluid_formation_volume_factor = typing.cast(
                        float, water_formation_volume_factor_grid[i, j, k]
                    )
                else:  # OIL
                    phase_mobility = typing.cast(
                        float, oil_relative_mobility_grid[i, j, k]
                    )
                    fluid_compressibility = typing.cast(
                        float, oil_compressibility_grid[i, j, k]
                    )
                    fluid_formation_volume_factor = typing.cast(
                        float, oil_formation_volume_factor_grid[i, j, k]
                    )

                production_rate = well.get_flow_rate(
                    pressure=cell_oil_pressure,
                    temperature=cell_temperature,
                    well_index=well_index,
                    phase_mobility=phase_mobility,
                    fluid=produced_fluid,
                    fluid_compressibility=fluid_compressibility,
                    use_pseudo_pressure=config.use_pseudo_pressure
                    and produced_phase == FluidPhase.GAS,
                    formation_volume_factor=fluid_formation_volume_factor,
                    allocation_fraction=allocation_fraction,
                    pvt_tables=config.pvt_tables,
                    **primary_phase_context,
                )

                if production_rate > 0.0 and config.warn_well_anomalies:
                    _warn_production_rate(
                        production_rate=production_rate,
                        well_name=well.name,
                        cell=(i - pad_width, j - pad_width, k - pad_width),
                        time=time,
                        rate_unit="ft³/day"
                        if produced_phase == FluidPhase.GAS
                        else "bbls/day",
                    )

                # Accumulate production rates by phase
                if produced_phase == FluidPhase.GAS:
                    cell_gas_production_rate += production_rate
                elif produced_phase == FluidPhase.WATER:
                    cell_water_production_rate += production_rate * bbl_to_ft3
                else:  # OIL
                    cell_oil_production_rate += production_rate * bbl_to_ft3

            if production_grid is not None:
                production_grid[i, j, k] = (
                    cell_oil_production_rate,
                    cell_water_production_rate,
                    cell_gas_production_rate,
                )

            # Update net grids
            net_water_well_rate_grid[i, j, k] += cell_water_production_rate
            net_oil_well_rate_grid[i, j, k] += cell_oil_production_rate
            net_gas_well_rate_grid[i, j, k] += cell_gas_production_rate

    return (
        net_water_well_rate_grid,
        net_oil_well_rate_grid,
        net_gas_well_rate_grid,
        solvent_injection_concentration_grid,
        gas_injection_rate_grid,
    )


@numba.njit(parallel=True, cache=True)
def apply_updates(
    water_saturation_grid: ThreeDimensionalGrid,
    oil_saturation_grid: ThreeDimensionalGrid,
    gas_saturation_grid: ThreeDimensionalGrid,
    solvent_concentration_grid: ThreeDimensionalGrid,
    net_water_flux_grid: ThreeDimensionalGrid,
    net_oil_flux_grid: ThreeDimensionalGrid,
    net_gas_flux_grid: ThreeDimensionalGrid,
    net_solvent_flux_grid: ThreeDimensionalGrid,
    net_water_well_rate_grid: ThreeDimensionalGrid,
    net_oil_well_rate_grid: ThreeDimensionalGrid,
    net_gas_well_rate_grid: ThreeDimensionalGrid,
    solvent_injection_concentration_grid: ThreeDimensionalGrid,
    gas_injection_rate_grid: ThreeDimensionalGrid,
    porosity_grid: ThreeDimensionalGrid,
    thickness_grid: ThreeDimensionalGrid,
    cell_size_x: float,
    cell_size_y: float,
    time_step_in_days: float,
    cfl_threshold: float,
    pressure_change_grid: typing.Optional[ThreeDimensionalGrid] = None,
    oil_compressibility_grid: typing.Optional[ThreeDimensionalGrid] = None,
    water_compressibility_grid: typing.Optional[ThreeDimensionalGrid] = None,
    gas_compressibility_grid: typing.Optional[ThreeDimensionalGrid] = None,
    rock_compressibility: float = 0.0,
) -> typing.Tuple[
    ThreeDimensionalGrid,
    ThreeDimensionalGrid,
    ThreeDimensionalGrid,
    ThreeDimensionalGrid,
    OneDimensionalGrid,
]:
    """
    Apply saturation and solvent concentration updates with CFL checking.

    :param water_saturation_grid: Current water saturation (-)
    :param oil_saturation_grid: Current oil saturation (-)
    :param gas_saturation_grid: Current gas saturation (-)
    :param solvent_concentration_grid: Current solvent concentration in oil (-)
    :param net_water_flux_grid: Net water flux for each cell (ft³/day)
    :param net_oil_flux_grid: Net oil flux for each cell (ft³/day)
    :param net_gas_flux_grid: Net gas flux for each cell (ft³/day)
    :param net_solvent_flux_grid: Net solvent flux for each cell (ft³/day)
    :param net_water_well_rate_grid: Net water well rate for each cell (ft³/day)
    :param net_oil_well_rate_grid: Net oil well rate for each cell (ft³/day)
    :param net_gas_well_rate_grid: Net gas well rate for each cell (ft³/day)
    :param solvent_injection_concentration_grid: Solvent concentration from injection (-)
    :param gas_injection_rate_grid: Gas injection rate (excluding production) for each cell (ft³/day)
    :param water_density_grid: Water density for each cell (lbm/ft³)
    :param oil_density_grid: Oil density for each cell (lbm/ft³)
    :param gas_density_grid: Gas density for each cell (lbm/ft³)
    :param porosity_grid: Porosity values (-)
    :param thickness_grid: Cell thickness values (ft)
    :param cell_size_x: Cell size in x direction (ft)
    :param cell_size_y: Cell size in y direction (ft)
    :param time_step_in_days: Time step size (days)
    :param cfl_threshold: Maximum allowed CFL number
    :return: Tuple of (updated_water_sat, updated_oil_sat, updated_gas_sat, updated_solvent_conc, cfl_violation_info)
             where `cfl_violation_info` is [violated, i, j, k, cfl_number, maximum_cfl]
    """
    apply_pvt_correction = (
        pressure_change_grid is not None
        and oil_compressibility_grid is not None
        and water_compressibility_grid is not None
        and gas_compressibility_grid is not None
    )
    nx, ny, nz = water_saturation_grid.shape

    # Initialize updated grids
    updated_water_saturation_grid = water_saturation_grid.copy()
    updated_oil_saturation_grid = oil_saturation_grid.copy()
    updated_gas_saturation_grid = gas_saturation_grid.copy()
    updated_solvent_concentration_grid = solvent_concentration_grid.copy()

    # CFL violation tracking: [violated (0 or 1), i, j, k, cfl_number, maximum_cfl]
    cfl_violation = np.zeros(6, dtype=np.float64)
    maximum_cfl_encountered = 0.0

    # Update saturations in parallel
    for i in numba.prange(1, nx - 1):  # type: ignore
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                # Get cell properties
                oil_saturation = oil_saturation_grid[i, j, k]
                cell_solvent_concentration = solvent_concentration_grid[i, j, k]
                porosity = porosity_grid[i, j, k]
                cell_thickness = thickness_grid[i, j, k]
                cell_volume = cell_size_x * cell_size_y * cell_thickness
                cell_pore_volume = porosity * cell_volume

                # Get fluxes and well rates for this cell
                net_water_flux = net_water_flux_grid[i, j, k]
                net_oil_flux = net_oil_flux_grid[i, j, k]
                net_gas_flux = net_gas_flux_grid[i, j, k]
                net_solvent_flux = net_solvent_flux_grid[i, j, k]

                net_water_well_rate = net_water_well_rate_grid[i, j, k]
                net_oil_well_rate = net_oil_well_rate_grid[i, j, k]
                net_gas_well_rate = net_gas_well_rate_grid[i, j, k]
                cell_solvent_injection_concentration = (
                    solvent_injection_concentration_grid[i, j, k]
                )
                cell_gas_injection_rate = gas_injection_rate_grid[i, j, k]

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
                    # Record max CFL violation
                    cfl_violation[0] = 1.0
                    cfl_violation[1] = float(i)
                    cfl_violation[2] = float(j)
                    cfl_violation[3] = float(k)
                    cfl_violation[4] = cfl_number
                    cfl_violation[5] = cfl_threshold
                    maximum_cfl_encountered = cfl_number

                # Total flow rates (advection + wells) in ft³/day
                total_water_flow = net_water_flux + net_water_well_rate
                total_oil_flow = net_oil_flux + net_oil_well_rate
                total_gas_flow = net_gas_flux + net_gas_well_rate

                # Calculate saturation changes from transport
                water_saturation_change = (total_water_flow * time_step_in_days) / (
                    cell_pore_volume
                )
                oil_saturation_change = (total_oil_flow * time_step_in_days) / (
                    cell_pore_volume
                )
                gas_saturation_change = (total_gas_flow * time_step_in_days) / (
                    cell_pore_volume
                )

                # Transport-based saturation update
                old_water_saturation = water_saturation_grid[i, j, k]
                old_oil_saturation = oil_saturation
                old_gas_saturation = gas_saturation_grid[i, j, k]

                new_water_saturation = old_water_saturation + water_saturation_change
                new_oil_saturation = old_oil_saturation + oil_saturation_change
                new_gas_saturation = old_gas_saturation + gas_saturation_change

                # PVT volume correction
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

                # Residual volume balance correction
                total_saturation = (
                    new_water_saturation + new_oil_saturation + new_gas_saturation
                )
                if abs(total_saturation - 1.0) > 1e-12:
                    new_oil_saturation += 1.0 - total_saturation
                    if new_oil_saturation < 0.0:
                        new_oil_saturation = 0.0

                updated_water_saturation_grid[i, j, k] = new_water_saturation
                updated_gas_saturation_grid[i, j, k] = new_gas_saturation
                updated_oil_saturation_grid[i, j, k] = new_oil_saturation

                # Update solvent concentration in oil phase
                # Mass balance: (C_old * V_oil_old) + (C_in * V_in) = (C_new * V_oil_new)
                if new_oil_saturation > 1e-9:  # Avoid division by zero
                    # New oil volume (after saturation updates)
                    new_oil_volume = new_oil_saturation * cell_pore_volume
                    # Current solvent volume in oil
                    old_solvent_volume = (
                        cell_solvent_concentration * oil_saturation * cell_pore_volume
                    )
                    # Solvent volume flux from advection
                    advected_solvent_volume = net_solvent_flux * time_step_in_days

                    # Solvent volume from injection (if miscible)
                    injected_solvent_volume = 0.0
                    if (
                        cell_gas_injection_rate > 0.0
                        and cell_solvent_injection_concentration > 0.0
                    ):
                        # Miscible solvent dissolves into oil immediately
                        # Limit dissolved volume to prevent C > 1
                        potential_injected_volume = (
                            cell_solvent_injection_concentration
                            * cell_gas_injection_rate
                            * time_step_in_days
                        )
                        # Maximum solvent that can dissolve without exceeding C=1
                        max_dissolvable = new_oil_volume - (
                            old_solvent_volume + advected_solvent_volume
                        )
                        # Take minimum to prevent oversaturation
                        injected_solvent_volume = max(
                            0.0, min(potential_injected_volume, max_dissolvable)
                        )

                    # Total solvent volume in oil
                    new_solvent_volume = (
                        old_solvent_volume
                        + advected_solvent_volume
                        + injected_solvent_volume
                    )
                    # New concentration
                    new_concentration = new_solvent_volume / new_oil_volume
                    # Clamp to [0, 1] (should already be satisfied, but ensure numerical stability)
                    updated_solvent_concentration_grid[i, j, k] = clip(
                        new_concentration, 0.0, 1.0
                    )
                else:
                    # No oil in cell, concentration is undefined (set to 0)
                    updated_solvent_concentration_grid[i, j, k] = 0.0

    return (
        updated_water_saturation_grid,
        updated_oil_saturation_grid,
        updated_gas_saturation_grid,
        updated_solvent_concentration_grid,
        cfl_violation,
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
