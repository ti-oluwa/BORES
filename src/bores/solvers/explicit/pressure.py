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
from bores.grids.pvt import build_total_fluid_compressibility_grid
from bores.models import FluidProperties, RockProperties
from bores.solvers.base import (
    EvolutionResult,
    _warn_injection_rate,
    _warn_production_rate,
)
from bores.transmissibility import FaceTransmissibilities
from bores.types import FluidPhase, ThreeDimensionalGrid, ThreeDimensions
from bores.wells.base import Wells
from bores.wells.controls import CoupledRateControl
from bores.wells.indices import WellIndicesCache

__all__ = ["evolve_pressure"]


@attrs.frozen
class ExplicitPressureSolution:
    pressure_grid: ThreeDimensionalGrid
    maximum_cfl_encountered: float
    cfl_threshold: float
    maximum_pressure_change: float


def evolve_pressure(
    cell_dimension: typing.Tuple[float, float],
    thickness_grid: ThreeDimensionalGrid,
    elevation_grid: ThreeDimensionalGrid,
    time_step: int,
    time_step_size: float,
    time: float,
    rock_properties: RockProperties[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
    face_transmissibilities: FaceTransmissibilities,
    pressure_boundaries: ThreeDimensionalGrid,
    flux_boundaries: ThreeDimensionalGrid,
    relative_mobility_grids: RelativeMobilityGrids[ThreeDimensions],
    capillary_pressure_grids: CapillaryPressureGrids[ThreeDimensions],
    wells: Wells[ThreeDimensions],
    config: Config,
    well_indices_cache: WellIndicesCache,
    injection_rates: typing.Optional[PhaseTensorsProxy[float, ThreeDimensions]] = None,
    production_rates: typing.Optional[PhaseTensorsProxy[float, ThreeDimensions]] = None,
    injection_fvfs: typing.Optional[PhaseTensorsProxy[float, ThreeDimensions]] = None,
    production_fvfs: typing.Optional[PhaseTensorsProxy[float, ThreeDimensions]] = None,
    injection_bhps: typing.Optional[PhaseTensorsProxy[float, ThreeDimensions]] = None,
    production_bhps: typing.Optional[PhaseTensorsProxy[float, ThreeDimensions]] = None,
    dtype: npt.DTypeLike = np.float64,
) -> EvolutionResult[ExplicitPressureSolution, None]:
    """
    Computes the pressure evolution (specifically, oil phase pressure P_oil) in the reservoir grid
    for one time step using an explicit finite volume method.

    :param cell_dimension: Tuple of (cell_size_x, cell_size_y) in feet (ft).
    :param thickness_grid: N-Dimensional numpy array representing the thickness of each cell in the reservoir (ft).
    :param elevation_grid: N-Dimensional numpy array representing the elevation of each cell in the reservoir (ft).
    :param time_step: Current time step number (starting from 0).
    :param time_step_size: Time step size (s) for each iteration.
    :param time: Total simulation time elapsed. This time step inclusive.
    :param rock_properties: `RockProperties` object containing rock physical properties including
        absolute permeability, porosity, residual saturations.
    :param fluid_properties: `FluidProperties` object containing fluid physical properties like
        pressure, temperature, saturations, viscosities, and compressibilities for
        water, oil, and gas.
    :param wells: `Wells` object containing well parameters for injection and production wells
    :param config: Simulation config.
    :param well_indices_cache: Cache of well indices for efficient lookup during pressure solve.
    :param injection_rates: Optional `PhaseTensorsProxy` of injection rates for each phase and cell.
    :param production_rates: Optional `PhaseTensorsProxy` of production rates for each phase and cell.
    :param injection_fvfs: Optional `PhaseTensorsProxy` of injection formation volume factors for each phase and cell.
    :param production_fvfs: Optional `PhaseTensorsProxy` of production formation volume factors for each phase and cell.
    :param injection_bhps: Optional `PhaseTensorsProxy` of injection bottom hole pressures for each phase and cell.
    :param production_bhps: Optional `PhaseTensorsProxy` of production bottom hole pressures for each phase and cell.
    :param pad_width: Number of ghost cells used for grid padding. Well coordinates are offset by this amount.
    :return: A N-Dimensional numpy array representing the updated oil phase pressure field (psi).
    """
    time_step_size_in_days = time_step_size * c.DAYS_PER_SECOND
    porosity_grid = rock_properties.porosity_grid
    rock_compressibility = rock_properties.compressibility
    oil_density_grid = fluid_properties.oil_effective_density_grid
    water_density_grid = fluid_properties.water_density_grid
    gas_density_grid = fluid_properties.gas_density_grid

    current_oil_pressure_grid = fluid_properties.pressure_grid
    current_water_saturation_grid = fluid_properties.water_saturation_grid
    current_oil_saturation_grid = fluid_properties.oil_saturation_grid
    current_gas_saturation_grid = fluid_properties.gas_saturation_grid

    water_compressibility_grid = fluid_properties.water_compressibility_grid
    oil_compressibility_grid = fluid_properties.oil_compressibility_grid
    gas_compressibility_grid = fluid_properties.gas_compressibility_grid

    # Determine grid dimensions and cell sizes
    cell_count_x, cell_count_y, cell_count_z = current_oil_pressure_grid.shape
    cell_size_x, cell_size_y = cell_dimension

    # Compute total fluid system compressibility for each cell
    total_fluid_compressibility_grid = build_total_fluid_compressibility_grid(
        oil_saturation_grid=current_oil_saturation_grid,
        oil_compressibility_grid=oil_compressibility_grid,
        water_saturation_grid=current_water_saturation_grid,
        water_compressibility_grid=water_compressibility_grid,
        gas_saturation_grid=current_gas_saturation_grid,
        gas_compressibility_grid=gas_compressibility_grid,
    )
    # Total compressibility (psi⁻¹) = fluid compressibility + rock compressibility
    total_compressibility_grid = np.add(
        total_fluid_compressibility_grid, rock_compressibility, dtype=dtype
    )
    # Clamp the compressibility within range
    total_compressibility_grid = config.total_compressibility_range.clip(
        total_compressibility_grid
    )
    md_per_cp_to_ft2_per_psi_per_day = (
        c.MILLIDARCIES_PER_CENTIPOISE_TO_SQUARE_FEET_PER_PSI_PER_DAY
    )

    (
        water_relative_mobility_grid,
        oil_relative_mobility_grid,
        gas_relative_mobility_grid,
    ) = relative_mobility_grids
    oil_water_capillary_pressure_grid, gas_oil_capillary_pressure_grid = (
        capillary_pressure_grids
    )

    # Compute CFL number for this time step
    pressure_cfl = compute_pressure_cfl_number(
        time_step_size_in_days=time_step_size_in_days,
        porosity_grid=porosity_grid,
        total_compressibility_grid=total_compressibility_grid,
        thickness_grid=thickness_grid,
        cell_size_x=cell_size_x,
        cell_size_y=cell_size_y,
        water_relative_mobility_grid=water_relative_mobility_grid,
        oil_relative_mobility_grid=oil_relative_mobility_grid,
        gas_relative_mobility_grid=gas_relative_mobility_grid,
        face_transmissibilities_x=face_transmissibilities.x,
        face_transmissibilities_y=face_transmissibilities.y,
        face_transmissibilities_z=face_transmissibilities.z,
        md_per_cp_to_ft2_per_psi_per_day=md_per_cp_to_ft2_per_psi_per_day,
    )
    max_pressure_cfl = config.pressure_cfl_threshold
    if pressure_cfl > max_pressure_cfl:
        return EvolutionResult(
            success=False,
            scheme="explicit",
            value=ExplicitPressureSolution(
                pressure_grid=current_oil_pressure_grid.astype(dtype, copy=False),
                maximum_cfl_encountered=pressure_cfl,
                cfl_threshold=max_pressure_cfl,
                maximum_pressure_change=0.0,
            ),
            message=f"Pressure evolution failed with CFL={pressure_cfl:.4f}.",
        )

    # Compute net flux contributions from neighbors
    # Compute gravitational constant conversion factor (ft/s² * lbf·s²/(lbm·ft) = lbf/lbm)
    # On Earth, this should normally be 1.0 in consistent units, but we include it for clarity
    # and say the acceleration due to gravity was changed to 12.0 ft/s² for some reason (say g on Mars)
    # then the conversion factor would be 12.0 / 32.174 = 0.373. Which would scale the gravity terms accordingly.
    gravitational_constant = (
        c.ACCELERATION_DUE_TO_GRAVITY_FEET_PER_SECONDS_SQUARE
        / c.GRAVITATIONAL_CONSTANT_LBM_FT_PER_LBF_S2
    )
    if (pool := config.task_pool) is not None:
        flux_future = pool.submit(
            compute_net_flux_contributions,
            current_oil_pressure_grid=current_oil_pressure_grid,
            cell_count_x=cell_count_x,
            cell_count_y=cell_count_y,
            cell_count_z=cell_count_z,
            pressure_boundaries=pressure_boundaries,
            flux_boundaries=flux_boundaries,
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
        well_rate_future = pool.submit(
            compute_well_rate_grid,
            cell_count_x=cell_count_x,
            cell_count_y=cell_count_y,
            cell_count_z=cell_count_z,
            wells=wells,
            current_oil_pressure_grid=current_oil_pressure_grid,
            temperature_grid=fluid_properties.temperature_grid,
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
            injection_rates=injection_rates,
            production_rates=production_rates,
            injection_fvfs=injection_fvfs,
            production_fvfs=production_fvfs,
            injection_bhps=injection_bhps,
            production_bhps=production_bhps,
            dtype=dtype,
        )
        net_flux_grid = flux_future.result()
        well_rate_grid = well_rate_future.result()

    else:
        net_flux_grid = compute_net_flux_contributions(
            current_oil_pressure_grid=current_oil_pressure_grid,
            cell_count_x=cell_count_x,
            cell_count_y=cell_count_y,
            cell_count_z=cell_count_z,
            pressure_boundaries=pressure_boundaries,
            flux_boundaries=flux_boundaries,
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
        well_rate_grid = compute_well_rate_grid(
            cell_count_x=cell_count_x,
            cell_count_y=cell_count_y,
            cell_count_z=cell_count_z,
            wells=wells,
            current_oil_pressure_grid=current_oil_pressure_grid,
            temperature_grid=fluid_properties.temperature_grid,
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
            injection_rates=injection_rates,
            injection_fvfs=injection_fvfs,
            production_fvfs=production_fvfs,
            production_rates=production_rates,
            injection_bhps=injection_bhps,
            production_bhps=production_bhps,
            dtype=dtype,
        )

    # Apply pressure updates
    updated_oil_pressure_grid = apply_updates(
        updated_grid=current_oil_pressure_grid.copy(),
        net_flux_grid=net_flux_grid,
        well_rate_grid=well_rate_grid,
        cell_count_x=cell_count_x,
        cell_count_y=cell_count_y,
        cell_count_z=cell_count_z,
        thickness_grid=thickness_grid,
        porosity_grid=porosity_grid,
        total_compressibility_grid=total_compressibility_grid,
        cell_size_x=cell_size_x,
        cell_size_y=cell_size_y,
        time_step_size_in_days=time_step_size_in_days,
    )
    maximum_pressure_change = np.max(
        np.abs(updated_oil_pressure_grid - current_oil_pressure_grid)
    )
    return EvolutionResult(
        success=True,
        scheme="explicit",
        value=ExplicitPressureSolution(
            pressure_grid=updated_oil_pressure_grid.astype(dtype, copy=False),
            maximum_cfl_encountered=pressure_cfl,
            cfl_threshold=max_pressure_cfl,
            maximum_pressure_change=maximum_pressure_change,
        ),
        message=f"Pressure evolution from time step {time_step} successful with CFL={pressure_cfl:.4f}.",
    )


@numba.njit(cache=True)
def compute_pressure_cfl_number(
    time_step_size_in_days: float,
    porosity_grid: ThreeDimensionalGrid,
    total_compressibility_grid: ThreeDimensionalGrid,
    thickness_grid: ThreeDimensionalGrid,
    cell_size_x: float,
    cell_size_y: float,
    water_relative_mobility_grid: ThreeDimensionalGrid,
    oil_relative_mobility_grid: ThreeDimensionalGrid,
    gas_relative_mobility_grid: ThreeDimensionalGrid,
    face_transmissibilities_x: ThreeDimensionalGrid,
    face_transmissibilities_y: ThreeDimensionalGrid,
    face_transmissibilities_z: ThreeDimensionalGrid,
    md_per_cp_to_ft2_per_psi_per_day: float,
) -> float:
    """
    Compute the maximum CFL number across all cells for pressure evolution,
    using precomputed geometric face transmissibilities.

    CFL = Δt * (Σ T_face * lambda_total_face) / (φ * c_t * V)

    Each T_face (mD·ft) already encodes k_harmonic * A / L.
    Multiplying by lambda_total (dimensionless relative mobility, in mD/cP units
    before conversion) and the unit conversion factor gives ft²/psi/day —
    the same transmissibility used in the pressure equation.

    For the CFL check we use the harmonic mean of the two adjacent cells'
    total relative mobilities as the face mobility estimate.
    """
    cell_count_x, cell_count_y, cell_count_z = porosity_grid.shape
    total_relative_mobility_grid = (
        water_relative_mobility_grid
        + oil_relative_mobility_grid
        + gas_relative_mobility_grid
    )
    maximum_cfl = 0.0

    for i in range(cell_count_x):
        for j in range(cell_count_y):
            for k in range(cell_count_z):
                cell_thickness = thickness_grid[i, j, k]
                cell_volume = cell_size_x * cell_size_y * cell_thickness
                cell_porosity = porosity_grid[i, j, k]
                cell_compressibility = total_compressibility_grid[i, j, k]

                if cell_compressibility <= 0.0 or cell_porosity <= 0.0:
                    continue

                cell_total_mobility = total_relative_mobility_grid[i, j, k]
                total_transmissibility = 0.0

                # X: forward face (i→i+1) and backward face (i-1→i)
                east_mobility = total_relative_mobility_grid[i + 1, j, k]
                east_face_mobility = compute_harmonic_mean(
                    cell_total_mobility, east_mobility
                )
                total_transmissibility += (
                    face_transmissibilities_x[i, j, k]
                    * east_face_mobility
                    * md_per_cp_to_ft2_per_psi_per_day
                )

                west_mobility = total_relative_mobility_grid[i - 1, j, k]
                west_face_mobility = compute_harmonic_mean(
                    cell_total_mobility, west_mobility
                )
                total_transmissibility += (
                    face_transmissibilities_x[i - 1, j, k]
                    * west_face_mobility
                    * md_per_cp_to_ft2_per_psi_per_day
                )

                # Y: forward face (j→j+1) and backward face (j-1→j)
                south_mobility = total_relative_mobility_grid[i, j + 1, k]
                south_face_mobility = compute_harmonic_mean(
                    cell_total_mobility, south_mobility
                )
                total_transmissibility += (
                    face_transmissibilities_y[i, j, k]
                    * south_face_mobility
                    * md_per_cp_to_ft2_per_psi_per_day
                )

                north_mobility = total_relative_mobility_grid[i, j - 1, k]
                north_face_mobility = compute_harmonic_mean(
                    cell_total_mobility, north_mobility
                )
                total_transmissibility += (
                    face_transmissibilities_y[i, j - 1, k]
                    * north_face_mobility
                    * md_per_cp_to_ft2_per_psi_per_day
                )

                # Z: forward face (k→k+1) and backward face (k-1→k)
                bottom_mobility = total_relative_mobility_grid[i, j, k + 1]
                bottom_face_mobility = compute_harmonic_mean(
                    cell_total_mobility, bottom_mobility
                )
                total_transmissibility += (
                    face_transmissibilities_z[i, j, k]
                    * bottom_face_mobility
                    * md_per_cp_to_ft2_per_psi_per_day
                )

                top_mobility = total_relative_mobility_grid[i, j, k - 1]
                top_face_mobility = compute_harmonic_mean(
                    cell_total_mobility, top_mobility
                )
                total_transmissibility += (
                    face_transmissibilities_z[i, j, k - 1]
                    * top_face_mobility
                    * md_per_cp_to_ft2_per_psi_per_day
                )

                cell_cfl = (time_step_size_in_days * total_transmissibility) / (
                    cell_porosity * cell_compressibility * cell_volume
                )
                maximum_cfl = max(maximum_cfl, cell_cfl)

    return maximum_cfl


@numba.njit(cache=True, inline="always")
def compute_flux_from_neighbour(
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
) -> float:
    """
    Computes the total volumetric flux from a neighbour cell into the current cell
    based on the pressure differences, mobilities, and capillary pressures.

    This function calculates the volumetric flux for each phase (water, oil, gas)
    from the neighbour cell into the current cell using the harmonic mobility approach.
    The total volumetric flux is the sum of the individual phase fluxes, converted to ft³/day.
    The formula used is:

    q_total = (λ_w * T_face * Water Phase Potential Difference)
              + (λ_o * T_face * Oil Phase Potential Difference)
              + (λ_g * T_face * Gas Phase Potential Difference)

    :param cell_indices: Indices of the current cell (i, j, k).
    :param neighbour_indices: Indices of the neighbour cell (i±1, j, k) or (i, j±1, k) or (i, j, k±1).
    :param oil_pressure_grid: N-Dimensional numpy array representing the oil phase pressure grid (psi).
    :param water_mobility_grid: N-Dimensional numpy array representing the water phase mobility grid (ft²/psi/day).
    :param oil_mobility_grid: N-Dimensional numpy array representing the oil phase mobility grid (ft²/psi/day).
    :param gas_mobility_grid: N-Dimensional numpy array representing the gas phase mobility grid (ft²/psi/day).
    :param oil_water_capillary_pressure_grid: N-Dimensional numpy array representing the oil-water capillary pressure grid (psi).
    :param gas_oil_capillary_pressure_grid: N-Dimensional numpy array representing the gas-oil capillary pressure grid (psi).
    :param oil_density_grid: N-Dimensional numpy array representing the oil phase density grid (lb/ft³).
    :param water_density_grid: N-Dimensional numpy array representing the water phase density grid (lb/ft³).
    :param gas_density_grid: N-Dimensional numpy array representing the gas phase density grid (lb/ft³).
    :param elevation_grid: N-Dimensional numpy array representing the thickness of each cell in the reservoir (ft).
    :param gravitational_constant: Gravitational constant conversion factor (lbf/lbm).
    :return: Total volumetric flux from neighbour to current cell (ft³/day).
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

    # Calculate the elevation difference between the neighbour and current cell
    elevation_difference = (
        elevation_grid[neighbour_indices] - elevation_grid[cell_indices]
    )
    # Determine the average densities for each phase across the face
    average_water_density = (
        water_density_grid[neighbour_indices] + water_density_grid[cell_indices]
    ) * 0.5
    average_oil_density = (
        oil_density_grid[neighbour_indices] + oil_density_grid[cell_indices]
    ) * 0.5
    average_gas_density = (
        gas_density_grid[neighbour_indices] + gas_density_grid[cell_indices]
    ) * 0.5

    # Calculate harmonic relative mobilities for each phase across the face (in the direction of flow)
    water_harmonic_relative_mobility = compute_harmonic_mean(
        water_relative_mobility_grid[neighbour_indices],
        water_relative_mobility_grid[cell_indices],
    )
    oil_harmonic_relative_mobility = compute_harmonic_mean(
        oil_relative_mobility_grid[neighbour_indices],
        oil_relative_mobility_grid[cell_indices],
    )
    gas_harmonic_relative_mobility = compute_harmonic_mean(
        gas_relative_mobility_grid[neighbour_indices],
        gas_relative_mobility_grid[cell_indices],
    )

    # Calculate volumetric flux for each phase from neighbour INTO the current cell
    # Flux_in = λ * T_face * 'Phase Potential Difference'

    # NOTE: Phase potential differences is the same as the pressure difference
    # For Oil and Water:
    # q = λ * (∆P + Gravity Potential) (ft³/day)
    # Calculate the water gravity potential (hydrostatic/gravity head)
    water_gravity_potential = (
        average_water_density * gravitational_constant * elevation_difference
    ) / 144.0
    # Calculate the total water phase potential
    water_potential_difference = water_pressure_difference + water_gravity_potential
    # Calculate the volumetric flux of water from neighbour to current cell
    water_volumetric_flux = (
        water_harmonic_relative_mobility
        * face_transmissibility
        * water_potential_difference
        * md_per_cp_to_ft2_per_psi_per_day
    )

    # For Oil:
    # Calculate the oil gravity potential (gravity head)
    oil_gravity_potential = (
        average_oil_density * gravitational_constant * elevation_difference
    ) / 144.0
    # Calculate the total oil phase potential
    oil_potential_difference = oil_pressure_difference + oil_gravity_potential
    # Calculate the volumetric flux of oil from neighbour to current cell
    oil_volumetric_flux = (
        oil_harmonic_relative_mobility
        * face_transmissibility
        * oil_potential_difference
        * md_per_cp_to_ft2_per_psi_per_day
    )

    # For Gas:
    # Calculate the gas gravity potential (gravity head)
    gas_gravity_potential = (
        average_gas_density * gravitational_constant * elevation_difference
    ) / 144.0
    # Calculate the total gas phase potential
    gas_potential_difference = gas_pressure_difference + gas_gravity_potential
    # Calculate the volumetric flux of gas from neighbour to current cell
    gas_volumetric_flux = (
        gas_harmonic_relative_mobility
        * face_transmissibility
        * gas_potential_difference
        * md_per_cp_to_ft2_per_psi_per_day
    )

    # Add these incoming fluxes to the net total for the cell, q (ft³/day)
    total_volumetric_flux = (
        water_volumetric_flux + oil_volumetric_flux + gas_volumetric_flux
    )
    return total_volumetric_flux


@numba.njit(parallel=True, cache=True)
def compute_net_flux_contributions(
    current_oil_pressure_grid: ThreeDimensionalGrid,
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
    pressure_boundaries: ThreeDimensionalGrid,
    flux_boundaries: ThreeDimensionalGrid,
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
) -> ThreeDimensionalGrid:
    """
    Compute net volumetric flux into each cell from all 6 neighbours (excluding wells).

    For each cell, all six face neighbours are checked:

    1. **Interior neighbour** (indices in [0, cell_count_*)):
       Full multi-phase flux via `compute_flux_from_neighbour` — pressure, capillary,
       and gravity contributions using harmonic face mobilities from both cells.

    2. **Out-of-bounds neighbour** — boundary face. Convert out-of-bounds indices
       to padded ghost-cell coordinates (i+1, j+1, k+1) and look up directly in
       the boundary grids:

       a. **Dirichlet** (pressure value is not NaN in pressure_boundaries):
          Known boundary pressure p_bc. One-sided flux using only the interior
          cell's total mobility (no harmonic mean — ghost has no real mobility)
          and only the oil pressure difference (ghost has no capillary or gravity
          properties):

              flux = T_geo * lambda_total[i,j,k] * md_per_cp * (p_bc - p_cell)

       b. **Neumann** (pressure value is NaN in pressure_boundaries):
          Known boundary flux q_bc in ft³/day from flux_boundaries. Added directly
          to the cell's net flux with no further modification.

    :param current_oil_pressure_grid: Current oil pressure grid (psi), shape (nx, ny, nz)
    :param cell_count_x: Number of cells in x-direction (real grid, no ghost cells)
    :param cell_count_y: Number of cells in y-direction (real grid, no ghost cells)
    :param cell_count_z: Number of cells in z-direction (real grid, no ghost cells)
    :param pressure_boundaries: 3D grid of boundary pressures, shape (nx+2, ny+2, nz+2).
        Ghost-cell region indexed by [i+1, j+1, k+1] for out-of-bounds cell (i, j, k).
        Contains pressure values for Dirichlet BCs; NaN indicates Neumann BC.
    :param flux_boundaries: 3D grid of boundary fluxes, shape (nx+2, ny+2, nz+2).
        Ghost-cell region indexed by [i+1, j+1, k+1] for out-of-bounds cell (i, j, k).
        Contains flux values for Neumann BCs (read when pressure_boundaries[...] is NaN).
    :param water_relative_mobility_grid: Water relative mobility grid (ft²/psi·day)
    :param oil_relative_mobility_grid: Oil relative mobility grid (ft²/psi·day)
    :param gas_relative_mobility_grid: Gas relative mobility grid (ft²/psi·day)
    :param face_transmissibilities_x: Geometric face transmissibilities in x-direction (mD·ft)
    :param face_transmissibilities_y: Geometric face transmissibilities in y-direction (mD·ft)
    :param face_transmissibilities_z: Geometric face transmissibilities in z-direction (mD·ft)
    :param oil_water_capillary_pressure_grid: Oil-water capillary pressure grid (psi)
    :param gas_oil_capillary_pressure_grid: Gas-oil capillary pressure grid (psi)
    :param oil_density_grid: Oil density grid (lb/ft³)
    :param water_density_grid: Water density grid (lb/ft³)
    :param gas_density_grid: Gas density grid (lb/ft³)
    :param elevation_grid: Cell elevation grid (ft)
    :param gravitational_constant: Gravitational constant conversion factor (lbf/lbm)
    :param md_per_cp_to_ft2_per_psi_per_day: Unit conversion factor
    :param dtype: NumPy dtype for array allocation (np.float32 or np.float64)
    :return: 3D grid of net volumetric fluxes (ft³/day), positive = net flow into cell
    """
    flux_grid = np.zeros((cell_count_x, cell_count_y, cell_count_z), dtype=dtype)

    for i in numba.prange(cell_count_x):  # type: ignore
        for j in range(cell_count_y):
            for k in range(cell_count_z):
                cell_pressure = current_oil_pressure_grid[i, j, k]
                cell_total_mobility = (
                    water_relative_mobility_grid[i, j, k]
                    + oil_relative_mobility_grid[i, j, k]
                    + gas_relative_mobility_grid[i, j, k]
                )
                net_flux = 0.0

                # Six neighbour directions with their associated geometric transmissibility.
                # For x/y/z: the forward face transmissibility is stored at the current cell
                # index, the backward face at the neighbour's index (one step back).
                #
                # Directions and their face transmissibility index:
                #   east   (i+1, j,   k  ) → face_transmissibilities_x[i,   j, k]
                #   west   (i-1, j,   k  ) → face_transmissibilities_x[i-1, j, k]
                #   south  (i,   j+1, k  ) → face_transmissibilities_y[i, j,   k]
                #   north  (i,   j-1, k  ) → face_transmissibilities_y[i, j-1, k]
                #   bottom (i,   j,   k+1) → face_transmissibilities_z[i, j, k  ]
                #   top    (i,   j,   k-1) → face_transmissibilities_z[i, j, k-1]

                # EAST (i+1, j, k)
                ei = i + 1
                if ei < cell_count_x:
                    net_flux += compute_flux_from_neighbour(
                        cell_indices=(i, j, k),
                        neighbour_indices=(ei, j, k),
                        oil_pressure_grid=current_oil_pressure_grid,
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
                else:
                    ni, nj, nk = ei + 1, j + 1, k + 1
                    pressure_boundary = pressure_boundaries[ni, nj, nk]
                    if not np.isnan(pressure_boundary):
                        net_flux += (
                            face_transmissibilities_x[i, j, k]
                            * cell_total_mobility
                            * md_per_cp_to_ft2_per_psi_per_day
                            * (pressure_boundary - cell_pressure)
                        )
                    else:
                        net_flux += flux_boundaries[ni, nj, nk]

                # WEST (i-1, j, k)
                wi = i - 1
                if wi >= 0:
                    net_flux += compute_flux_from_neighbour(
                        cell_indices=(i, j, k),
                        neighbour_indices=(wi, j, k),
                        oil_pressure_grid=current_oil_pressure_grid,
                        face_transmissibility=face_transmissibilities_x[wi, j, k],
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
                else:
                    ni, nj, nk = wi + 1, j + 1, k + 1
                    pressure_boundary = pressure_boundaries[ni, nj, nk]
                    if not np.isnan(pressure_boundary):
                        net_flux += (
                            face_transmissibilities_x[i, j, k]
                            * cell_total_mobility
                            * md_per_cp_to_ft2_per_psi_per_day
                            * (pressure_boundary - cell_pressure)
                        )
                    else:
                        net_flux += flux_boundaries[ni, nj, nk]

                # SOUTH (i, j+1, k)
                sj = j + 1
                if sj < cell_count_y:
                    net_flux += compute_flux_from_neighbour(
                        cell_indices=(i, j, k),
                        neighbour_indices=(i, sj, k),
                        oil_pressure_grid=current_oil_pressure_grid,
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
                else:
                    ni, nj, nk = i + 1, sj + 1, k + 1
                    pressure_boundary = pressure_boundaries[ni, nj, nk]
                    if not np.isnan(pressure_boundary):
                        net_flux += (
                            face_transmissibilities_y[i, j, k]
                            * cell_total_mobility
                            * md_per_cp_to_ft2_per_psi_per_day
                            * (pressure_boundary - cell_pressure)
                        )
                    else:
                        net_flux += flux_boundaries[ni, nj, nk]

                # NORTH (i, j-1, k)
                nj = j - 1
                if nj >= 0:
                    net_flux += compute_flux_from_neighbour(
                        cell_indices=(i, j, k),
                        neighbour_indices=(i, nj, k),
                        oil_pressure_grid=current_oil_pressure_grid,
                        face_transmissibility=face_transmissibilities_y[i, nj, k],
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
                else:
                    ni, nj, nk = i + 1, nj + 1, k + 1
                    pressure_boundary = pressure_boundaries[ni, nj, nk]
                    if not np.isnan(pressure_boundary):
                        net_flux += (
                            face_transmissibilities_y[i, j, k]
                            * cell_total_mobility
                            * md_per_cp_to_ft2_per_psi_per_day
                            * (pressure_boundary - cell_pressure)
                        )
                    else:
                        net_flux += flux_boundaries[ni, nj, nk]

                # BOTTOM (i, j, k+1)
                bk = k + 1
                if bk < cell_count_z:
                    net_flux += compute_flux_from_neighbour(
                        cell_indices=(i, j, k),
                        neighbour_indices=(i, j, bk),
                        oil_pressure_grid=current_oil_pressure_grid,
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
                else:
                    ni, nj, nk = i + 1, j + 1, bk + 1
                    pressure_boundary = pressure_boundaries[ni, nj, nk]
                    if not np.isnan(pressure_boundary):
                        net_flux += (
                            face_transmissibilities_z[i, j, k]
                            * cell_total_mobility
                            * md_per_cp_to_ft2_per_psi_per_day
                            * (pressure_boundary - cell_pressure)
                        )
                    else:
                        net_flux += flux_boundaries[ni, nj, nk]

                # TOP (i, j, k-1)
                tk = k - 1
                if tk >= 0:
                    net_flux += compute_flux_from_neighbour(
                        cell_indices=(i, j, k),
                        neighbour_indices=(i, j, tk),
                        oil_pressure_grid=current_oil_pressure_grid,
                        face_transmissibility=face_transmissibilities_z[i, j, tk],
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
                else:
                    ni, nj, nk = i + 1, j + 1, tk + 1
                    pressure_boundary = pressure_boundaries[ni, nj, nk]
                    if not np.isnan(pressure_boundary):
                        net_flux += (
                            face_transmissibilities_z[i, j, k]
                            * cell_total_mobility
                            * md_per_cp_to_ft2_per_psi_per_day
                            * (pressure_boundary - cell_pressure)
                        )
                    else:
                        net_flux += flux_boundaries[ni, nj, nk]

                flux_grid[i, j, k] = net_flux

    return flux_grid


def compute_well_rate_grid(
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
    wells: Wells[ThreeDimensions],
    current_oil_pressure_grid: ThreeDimensionalGrid,
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
    injection_rates: typing.Optional[PhaseTensorsProxy[float, ThreeDimensions]] = None,
    production_rates: typing.Optional[PhaseTensorsProxy[float, ThreeDimensions]] = None,
    injection_fvfs: typing.Optional[PhaseTensorsProxy[float, ThreeDimensions]] = None,
    production_fvfs: typing.Optional[PhaseTensorsProxy[float, ThreeDimensions]] = None,
    injection_bhps: typing.Optional[PhaseTensorsProxy[float, ThreeDimensions]] = None,
    production_bhps: typing.Optional[PhaseTensorsProxy[float, ThreeDimensions]] = None,
) -> ThreeDimensionalGrid:
    """
    Compute well rates for all cells (injection + production).

    :param cell_count_x: Number of cells in x-direction (including boundaries)
    :param cell_count_y: Number of cells in y-direction (including boundaries)
    :param cell_count_z: Number of cells in z-direction (including boundaries)
    :param wells: Wells grid containing injection and production wells
    :param current_oil_pressure_grid: Current oil pressure grid (psi)
    :param temperature_grid: Temperature grid (°F or °R)
    :param absolute_permeability: Absolute permeability in x, y, z directions (mD)
    :param water_relative_mobility_grid: Water relative mobility (1/cP)
    :param oil_relative_mobility_grid: Oil relative mobility (1/cP)
    :param gas_relative_mobility_grid: Gas relative mobility (1/cP)
    :param water_compressibility_grid: Water compressibility grid (1/psi)
    :param oil_compressibility_grid: Oil compressibility grid (1/psi)
    :param gas_compressibility_grid: Gas compressibility grid (1/psi)
    :param fluid_properties: Fluid properties container
    :param time: Total simulation time elapsed. This time step inclusive.
    :param config: Simulation config
    :param dtype: NumPy dtype for array allocation (np.float32 or np.float64)
    :param well_indices_cache: Cache of well indices for efficient lookup during pressure solve.
    :param injection_rates: Optional `PhaseTensorsProxy` of injection rates for each phase and cell.
    :param production_rates: Optional `PhaseTensorsProxy` of production rates for each phase and cell.
    :param injection_fvfs: Optional `PhaseTensorsProxy` of injection formation volume factors for each phase and cell.
    :param production_fvfs: Optional `PhaseTensorsProxy` of production formation volume factors for each phase and cell.
    :param injection_bhps: Optional `PhaseTensorsProxy` of injection bottom hole pressures for each phase and cell.
    :param production_bhps: Optional `PhaseTensorsProxy` of production bottom hole pressures for each phase and cell.
    :param pad_width: Number of ghost cells used for grid padding. Well coordinates are offset by this amount.
    :return: 3D grid of net well flow rates (ft³/day), positive = injection, negative = production
    """
    well_rate_grid = np.zeros((cell_count_x, cell_count_y, cell_count_z), dtype=dtype)
    bbl_to_ft3 = c.BARRELS_TO_CUBIC_FEET

    for well in wells.injection_wells:
        if not well.is_open or well.injected_fluid is None:
            continue

        injected_fluid = well.injected_fluid
        injected_phase = injected_fluid.phase
        use_pseudo_pressure = (
            config.use_pseudo_pressure and injected_phase == FluidPhase.GAS
        )

        # Compute rates using cached well indices
        water_bubble_point_pressure_grid = (
            fluid_properties.water_bubble_point_pressure_grid
        )
        gas_formation_volume_factor_grid = (
            fluid_properties.gas_formation_volume_factor_grid
        )
        gas_solubility_in_water_grid = fluid_properties.gas_solubility_in_water_grid

        well_indices = well_indices_cache.injection[well.name]
        for perforation_index in well_indices:
            i, j, k = perforation_index.cell
            well_index = perforation_index.well_index
            allocation_fraction = well_indices.allocation_fraction(perforation_index)
            cell_temperature = typing.cast(float, temperature_grid[i, j, k])
            cell_oil_pressure = typing.cast(float, current_oil_pressure_grid[i, j, k])

            phase_fvf = injected_fluid.get_formation_volume_factor(
                pressure=cell_oil_pressure,
                temperature=cell_temperature,
            )
            phase_fvf = typing.cast(float, phase_fvf)

            # Get phase mobility
            if injected_phase == FluidPhase.GAS:
                compressibility_kwargs = {}
            else:  # Water injection
                compressibility_kwargs = {
                    "bubble_point_pressure": water_bubble_point_pressure_grid[i, j, k],
                    "gas_formation_volume_factor": gas_formation_volume_factor_grid[
                        i, j, k
                    ],
                    "gas_solubility_in_water": gas_solubility_in_water_grid[i, j, k],
                }

            # Get fluid properties
            phase_compressibility = injected_fluid.get_compressibility(
                pressure=cell_oil_pressure,
                temperature=cell_temperature,
                **compressibility_kwargs,
            )
            phase_compressibility = typing.cast(float, phase_compressibility)
            phase_mobility = typing.cast(
                float,
                injected_fluid.get_mobility(
                    pressure=cell_oil_pressure, temperature=cell_temperature
                ),
            )

            flow_rate, effective_bhp = well.get_control(
                pressure=cell_oil_pressure,
                temperature=cell_temperature,
                well_index=well_index,
                phase_mobility=phase_mobility,
                fluid=injected_fluid,
                fluid_compressibility=phase_compressibility,
                use_pseudo_pressure=use_pseudo_pressure,
                formation_volume_factor=phase_fvf,
                allocation_fraction=allocation_fraction,
                pvt_tables=None,
            )

            # Check for backflow (negative injection)
            if flow_rate < 0.0 and config.warn_well_anomalies:
                _warn_injection_rate(
                    injection_rate=flow_rate,
                    well_name=well.name,
                    time=time,
                    cell=(i, j, k),
                    rate_unit="ft³/day"
                    if injected_phase == FluidPhase.GAS
                    else "bbls/day",
                )

            if injected_phase != FluidPhase.GAS:
                flow_rate *= bbl_to_ft3

            well_rate_grid[i, j, k] += flow_rate

            if injection_rates is not None:
                if injected_phase == FluidPhase.GAS:
                    injection_rates[i, j, k] = (0.0, 0.0, flow_rate)
                else:
                    injection_rates[i, j, k] = (flow_rate, 0.0, 0.0)

            if injection_fvfs is not None:
                if injected_phase == FluidPhase.GAS:
                    injection_fvfs[i, j, k] = (0.0, 0.0, phase_fvf)
                else:
                    injection_fvfs[i, j, k] = (phase_fvf, 0.0, 0.0)

            if injection_bhps is not None:
                if injected_phase == FluidPhase.GAS:
                    injection_bhps[i, j, k] = (0.0, 0.0, effective_bhp)
                else:
                    injection_bhps[i, j, k] = (effective_bhp, 0.0, 0.0)

    # Process production wells
    for well in wells.production_wells:
        if not well.is_open:
            continue

        # Compute rates using cached well indices
        water_formation_volume_factor_grid = (
            fluid_properties.water_formation_volume_factor_grid
        )
        oil_formation_volume_factor_grid = (
            fluid_properties.oil_formation_volume_factor_grid
        )
        gas_formation_volume_factor_grid = (
            fluid_properties.gas_formation_volume_factor_grid
        )
        is_couple_controlled = isinstance(well.control, CoupledRateControl)
        well_indices = well_indices_cache.production[well.name]

        for perforation_index in well_indices:
            i, j, k = perforation_index.cell
            well_index = perforation_index.well_index
            allocation_fraction = well_indices.allocation_fraction(perforation_index)
            cell_temperature = typing.cast(float, temperature_grid[i, j, k])
            cell_oil_pressure = typing.cast(float, current_oil_pressure_grid[i, j, k])

            context: dict = {}
            if is_couple_controlled:
                context = well.control.build_primary_phase_context(  # type: ignore
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

            water_rate = 0.0
            oil_rate = 0.0
            gas_rate = 0.0
            water_fvf = 0.0
            oil_fvf = 0.0
            gas_fvf = 0.0
            water_bhp = 0.0
            oil_bhp = 0.0
            gas_bhp = 0.0
            for produced_fluid in well.produced_fluids:
                produced_phase = produced_fluid.phase

                # Get phase-specific properties
                if produced_phase == FluidPhase.GAS:
                    phase_mobility = typing.cast(
                        float, gas_relative_mobility_grid[i, j, k]
                    )
                    phase_compressibility = typing.cast(
                        float, gas_compressibility_grid[i, j, k]
                    )
                    phase_fvf = typing.cast(
                        float, gas_formation_volume_factor_grid[i, j, k]
                    )
                elif produced_phase == FluidPhase.WATER:
                    phase_mobility = typing.cast(
                        float, water_relative_mobility_grid[i, j, k]
                    )
                    phase_compressibility = typing.cast(
                        float, water_compressibility_grid[i, j, k]
                    )
                    phase_fvf = typing.cast(
                        float, water_formation_volume_factor_grid[i, j, k]
                    )
                else:  # Oil
                    phase_mobility = typing.cast(
                        float, oil_relative_mobility_grid[i, j, k]
                    )
                    phase_compressibility = typing.cast(
                        float, oil_compressibility_grid[i, j, k]
                    )
                    phase_fvf = typing.cast(
                        float, oil_formation_volume_factor_grid[i, j, k]
                    )

                use_pseudo_pressure = (
                    config.use_pseudo_pressure and produced_phase == FluidPhase.GAS
                )
                # Compute production rate (bbls/day for liquids, ft³/day for gas)
                # Note: Production rates are negative by convention
                flow_rate, effective_bhp = well.get_control(
                    pressure=cell_oil_pressure,
                    temperature=cell_temperature,
                    well_index=well_index,
                    phase_mobility=phase_mobility,
                    fluid=produced_fluid,
                    fluid_compressibility=phase_compressibility,
                    use_pseudo_pressure=use_pseudo_pressure,
                    formation_volume_factor=phase_fvf,
                    allocation_fraction=allocation_fraction,
                    pvt_tables=config.pvt_tables,
                    **context,
                )

                # Check for backflow (positive production = injection)
                if flow_rate > 0.0 and config.warn_well_anomalies:
                    _warn_production_rate(
                        production_rate=flow_rate,
                        well_name=well.name,
                        time=time,
                        cell=(i, j, k),
                        rate_unit="ft³/day"
                        if produced_phase == FluidPhase.GAS
                        else "bbls/day",
                    )

                # Convert to ft³/day if not already
                if produced_phase != FluidPhase.GAS:
                    flow_rate *= bbl_to_ft3

                # Production rates are already negative
                well_rate_grid[i, j, k] += flow_rate

                if produced_phase == FluidPhase.GAS:
                    gas_rate += flow_rate
                    gas_fvf = phase_fvf
                    gas_bhp = effective_bhp
                elif produced_phase == FluidPhase.WATER:
                    water_rate += flow_rate
                    water_fvf = phase_fvf
                    water_bhp = effective_bhp
                else:
                    oil_rate += flow_rate
                    oil_fvf = phase_fvf
                    oil_bhp = effective_bhp

            if production_rates is not None:
                production_rates[i, j, k] = (water_rate, oil_rate, gas_rate)

            if production_fvfs is not None:
                production_fvfs[i, j, k] = (water_fvf, oil_fvf, gas_fvf)

            if production_bhps is not None:
                production_bhps[i, j, k] = (water_bhp, oil_bhp, gas_bhp)

    return well_rate_grid


@numba.njit(parallel=True, cache=True)
def apply_updates(
    updated_grid: ThreeDimensionalGrid,
    net_flux_grid: ThreeDimensionalGrid,
    well_rate_grid: ThreeDimensionalGrid,
    cell_count_x: int,
    cell_count_y: int,
    cell_count_z: int,
    thickness_grid: ThreeDimensionalGrid,
    porosity_grid: ThreeDimensionalGrid,
    total_compressibility_grid: ThreeDimensionalGrid,
    cell_size_x: float,
    cell_size_y: float,
    time_step_size_in_days: float,
) -> ThreeDimensionalGrid:
    """
    Apply pressure updates to all cells using pre-computed flux and well contributions.

    This function combines the net flux contributions (from neighbors) and well contributions
    to compute and apply pressure changes to each cell. The computation is parallelized
    across all cells using prange.

    :param updated_grid: Current oil pressure grid to be updated (psi)
    :param net_flux_grid: Pre-computed net flux contributions (ft³/day)
    :param well_rate_grid: Pre-computed well rate contributions (ft³/day)
    :param cell_count_x: Number of cells in x-direction (including boundaries)
    :param cell_count_y: Number of cells in y-direction (including boundaries)
    :param cell_count_z: Number of cells in z-direction (including boundaries)
    :param thickness_grid: Cell thickness grid (ft)
    :param porosity_grid: Cell porosity grid (fraction)
    :param total_compressibility_grid: Total compressibility grid (1/psi)
    :param cell_size_x: Cell size in x-direction (ft)
    :param cell_size_y: Cell size in y-direction (ft)
    :param time_step_size_in_days: Time step size (days)
    :return: Updated oil pressure grid (psi)
    """
    for i in numba.prange(cell_count_x):  # type: ignore
        for j in range(cell_count_y):
            for k in range(cell_count_z):
                # Cell properties
                cell_thickness = thickness_grid[i, j, k]
                cell_volume = cell_size_x * cell_size_y * cell_thickness
                cell_porosity = porosity_grid[i, j, k]
                cell_total_compressibility = total_compressibility_grid[i, j, k]

                # Total flow rate = flux from neighbors + well contribution
                net_volumetric_flow = net_flux_grid[i, j, k]
                net_well_flow = well_rate_grid[i, j, k]
                total_flow_rate = net_volumetric_flow + net_well_flow

                # Calculate pressure change
                # dP = (Δt / (φ * c_t * V)) * Q_total
                change_in_pressure = (
                    time_step_size_in_days
                    / (cell_porosity * cell_total_compressibility * cell_volume)
                ) * total_flow_rate

                # Apply the update
                updated_grid[i, j, k] += change_in_pressure

    return updated_grid


"""
Explicit finite difference formulation for pressure diffusion in a 3D reservoir
(slightly compressible three-phase fluid):

The governing equation is the 3D linear-flow diffusivity equation:

    ∂p/∂t * (φ·c_t) * V = ∇ · (λ_t · ∇Φ) * A + q * V

where:
    - ∂p/∂t * (φ·c_t) * V is the accumulation term (ft³/day)
    - ∇ · (λ_t · ∇Φ) * A is the diffusion term including gravity and capillary effects
    - q * V is the source/sink term (injection/production) (ft³/day)

    λ_t = total mobility = Σ_phases (k · kr_phase / μ_phase)  (ft²/psi·day)
    Φ_phase = phase potential = P_phase + ρ_phase · (g/gc) · Δz / 144  (psi)

    Phase pressures:
        P_water = P_oil - P_cow  (capillary correction)
        P_gas   = P_oil + P_cgo  (capillary correction)

Assumptions:
    - Slightly compressible fluids (properties frozen at current time level)
    - Cartesian grid (structured)
    - Mobility and density are evaluated at old time level (explicit)

The diffusion term is expanded as:

    ∇ · (λ_t · ∇Φ) = ∂/∂x (λ_t · ∂Φ/∂x) + ∂/∂y (λ_t · ∂Φ/∂y) + ∂/∂z (λ_t · ∂Φ/∂z)

    The total flux across each face sums phase contributions:
        q_face = Σ_phases [λ_phase · (ΔP_phase + ρ_phase · g/gc · Δelevation / 144)]

Explicit Discretization (Forward Euler in time, central difference in space):

    ∂p/∂t ≈ (pⁿ⁺¹_ijk - pⁿ_ijk) / Δt

    ∂/∂x (λ·∂Φ/∂x) ≈ (λ_{i+½}·Φⁿ_{i+1} - λ_{i+½}·Φⁿ_i - λ_{i-½}·Φⁿ_i + λ_{i-½}·Φⁿ_{i-1}) / Δx²

Final explicit update formula:

    pⁿ⁺¹_ijk = pⁿ_ijk + (Δt / (φ·c_t·V)) * [
        Σ_neighbours [λ_harmonic · (ΔP + capillary + gravity) · A_face / ΔL] + q_{i,j,k} * V
    ]

    Gravity potential at each face:
        gravity_potential_phase = harmonic_ρ_phase · (g/gc) · Δelevation / 144   (psi)

Where:
    - Δt is time step (s)
    - Δx, Δy, Δz are cell dimensions (ft)
    - A_x = Δy · h_face; A_y = Δx · h_face; A_z = Δx · Δy (ft²)
    - V = Δx · Δy · Δz (ft³)
    - h_face = harmonic mean of adjacent cell thicknesses (ft)
    - λ_{i±½,...} = harmonic average of total mobility between adjacent cells
    - q_{i,j,k} = well injection/production rate (ft³/day)

Stability Condition:
    Explicit scheme is conditionally stable. The 3D diffusion CFL criterion requires:
    Δt ≤ min( (φ·c_t·V) / (2 · λ_max · (A_x/Δx + A_y/Δy + A_z/Δz)) )

Notes:
    - Harmonic averaging is used for both λ and ρ at cell interfaces.
    - Gravity and capillary effects are included in the phase potential difference.
    - Volume-normalized source/sink terms only affect the cell where the well is located.
"""
