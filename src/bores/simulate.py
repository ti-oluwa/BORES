"""Run a simulation workflow on a multi-dimensional reservoir model."""

import copy
import logging
import typing
from datetime import datetime, timezone
from os import PathLike

import attrs
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from bores._precision import get_dtype
from bores.boundary_conditions import (
    BoundaryConditions,
    GridBoundaryCondition,
    default_bc,
)
from bores.config import Config
from bores.constants import c
from bores.datastructures import (
    BottomHolePressure,
    BottomHolePressures,
    PhaseTensorsProxy,
    Rates,
    SparseTensor,
)
from bores.errors import SimulationError, StopSimulation, TimingError, ValidationError
from bores.grids.base import CapillaryPressureGrids, RelativeMobilityGrids
from bores.grids.boundary_conditions import (
    apply_boundary_conditions,
    apply_pressure_boundary_condition,
    apply_saturation_boundary_conditions,
)
from bores.grids.pvt import (
    build_three_phase_relative_mobilities_grids,
    build_three_phase_relative_permeabilities_grids,
)
from bores.grids.rock_fluid import build_rock_fluid_properties_grids
from bores.grids.updates import (
    apply_solution_gas_updates,
    update_pvt_grids,
    update_residual_saturation_grids,
)
from bores.grids.utils import pad_grid
from bores.models import (
    FluidProperties,
    ReservoirModel,
    RockProperties,
    SaturationHistory,
)
from bores.solvers import explicit, implicit
from bores.solvers.base import normalize_saturations
from bores.states import ModelState
from bores.stores import StoreSerializable
from bores.tables.pvt import PVTDataSet, PVTTables
from bores.types import MiscibilityModel, NDimension, NDimensionalGrid, ThreeDimensions
from bores.wells.base import Wells
from bores.wells.indices import WellIndicesCache, build_well_indices_cache

__all__ = ["Run", "run"]

logger = logging.getLogger(__name__)


UNPHYSICAL_PRESSURE_ERROR_MSG = """
Unphysical pressure encountered in the pressure grid at the following indices:

{indices}

This indicates a likely issue with the simulation setup, numerical stability, or physical parameters.

Potential causes include:
1. Boundary conditions that allow for unphysical pressure drops.
2. Incompatible or unrealistic rock/fluid properties.
3. Time step size too large for explicit schemes, leading to instability.
4. Incorrect initial conditions or pressure distributions.
5. Unrealistic/improperly configured wells (e.g., injection/production rates or pressures).
6. Numerical issues due to discretization choices or solver settings.

Suggested actions:
- Validate boundary conditions and ensure fixed-pressure constraints are properly applied.
- Check permeability, porosity, and compressibility values.
- Cell dimensions and bulk volume should be appropriate for the physical scale of the reservoir.
- Use smaller time steps if using explicit updates.
- Cross-check well source/sink terms for sign and magnitude correctness.

Simulation aborted to avoid propagation of unphysical results.
"""


@attrs.frozen(slots=True)
class StepResult(typing.Generic[NDimension]):
    """
    Result from executing one time step of the simulation.
    """

    fluid_properties: FluidProperties[NDimension]
    """Updated fluid properties after the time step."""
    rock_properties: RockProperties[NDimension]
    """Updated rock properties after the time step."""
    saturation_history: SaturationHistory[NDimension]
    """Updated saturation history after the time step."""
    injection_rates: typing.Optional[Rates[float, NDimension]] = None
    """Phase injection rates during the time step."""
    production_rates: typing.Optional[Rates[float, NDimension]] = None
    """Phase production rates during the time step."""
    injection_bhps: typing.Optional[BottomHolePressures[float, NDimension]] = None
    """Phase injection bottom hole pressures during the time step"""
    production_bhps: typing.Optional[BottomHolePressures[float, NDimension]] = None
    """Phase production bottom hole pressures during the time step"""
    success: bool = True
    """Whether the time step evolution was successful."""
    message: typing.Optional[str] = None
    """Optional message providing additional information about the time step result."""
    timer_kwargs: typing.Dict[str, typing.Any] = attrs.field(factory=dict)
    """Kwargs that should be passed to the simulation timer on accepting or rejecting a step."""


@attrs.frozen(slots=True)
class SaturationChangeCheckResult:
    violated: bool
    max_phase_saturation_change: float
    max_allowed_phase_saturation_change: float
    message: typing.Optional[str] = None


def _validate_pressure_range(
    padded_pressure_grid: NDimensionalGrid[ThreeDimensions],
    time_step: int,
    padded_fluid_properties: FluidProperties[ThreeDimensions],
    padded_rock_properties: RockProperties[ThreeDimensions],
    padded_saturation_history: SaturationHistory[ThreeDimensions],
) -> typing.Optional[StepResult[ThreeDimensions]]:
    """
    Check for out-of-range pressures and return failure `StepResult` if found.

    :param padded_pressure_grid: Pressure grid to validate.
    :param time_step: Current time step index.
    :param padded_zeros_grid: Padded grid of zeros for rate tracking.
    :param padded_fluid_properties: Padded fluid properties.
    :param padded_rock_properties: Padded rock properties.
    :param padded_saturation_history: Padded saturation history.
    :return: `StepResult` with failure if pressures are out of range, None otherwise.
    """
    min_allowable_pressure = c.MINIMUM_VALID_PRESSURE - 1e-3
    max_allowable_pressure = c.MAXIMUM_VALID_PRESSURE + 1e-3
    out_of_range_mask = (padded_pressure_grid < min_allowable_pressure) | (
        padded_pressure_grid > max_allowable_pressure
    )
    out_of_range_indices = np.argwhere(out_of_range_mask)

    if out_of_range_indices.size > 0:
        min_pressure = np.min(padded_pressure_grid)
        max_pressure = np.max(padded_pressure_grid)
        logger.warning(
            f"Unphysical pressure detected at {out_of_range_indices.size} cells. "
            f"Range: [{min_pressure:.4f}, {max_pressure:.4f}] psi. "
            f"Allowed: [{min_allowable_pressure}, {max_allowable_pressure}]."
        )
        message = ""
        if min_pressure < min_allowable_pressure:
            message += f"Pressure dropped below {min_allowable_pressure} psi (Min: {min_pressure:.4f}).\n"
        if max_pressure > max_allowable_pressure:
            message += f"Pressure exceeded {max_allowable_pressure} psi (Max: {max_pressure:.4f}).\n"

        message += (
            UNPHYSICAL_PRESSURE_ERROR_MSG.format(indices=out_of_range_indices.tolist())
            + f"\nAt Time Step {time_step}."
        )
        return StepResult(
            fluid_properties=padded_fluid_properties,
            rock_properties=padded_rock_properties,
            saturation_history=padded_saturation_history,
            success=False,
            message=message,
        )
    return None


def _check_saturation_changes(
    maximum_oil_saturation_change: float,
    maximum_water_saturation_change: float,
    maximum_gas_saturation_change: float,
    max_allowed_oil_saturation_change: float,
    max_allowed_water_saturation_change: float,
    max_allowed_gas_saturation_change: float,
    tolerance: float = 1e-4,
) -> SaturationChangeCheckResult:
    """
    Check if the saturation changes for each phase exceed their allowed maximums.
    If any phase exceeds its limit, return a violation result. If multiple phases exceed
    their limits, return the maximum phase saturation change and the corresponding allowed maximum.
    If no violations occur, report a non-violated result with zeros for the max changes.

    :param maximum_oil_saturation_change: Maximum oil saturation change observed.
    :param maximum_water_saturation_change: Maximum water saturation change observed.
    :param maximum_gas_saturation_change: Maximum gas saturation change observed.
    :param max_allowed_oil_saturation_change: Maximum allowed oil saturation change.
    :param max_allowed_water_saturation_change: Maximum allowed water saturation change.
    :param max_allowed_gas_saturation_change: Maximum allowed gas saturation change.
    :param tolerance: Relative tolerance for saturation change checks.
    :return: `SaturationChangeCheckResult` indicating if any saturation change limits were violated.
    """
    violated = False
    messages = []
    max_phase_saturation_change = 0.0
    max_allowed_phase_saturation_change = 0.0

    oil_tolerance = max(tolerance, 0.005 * max_allowed_oil_saturation_change)
    max_tolerable_oil_saturation_change = max_allowed_oil_saturation_change * (
        1 + oil_tolerance
    )
    if maximum_oil_saturation_change > max_tolerable_oil_saturation_change:
        violated = True
        # Start assuming oil saturation change is the largest so far
        max_phase_saturation_change = maximum_oil_saturation_change
        max_allowed_phase_saturation_change = max_allowed_oil_saturation_change
        messages.append(
            f"Oil saturation change {maximum_oil_saturation_change:.6f} exceeded maximum allowed {max_allowed_oil_saturation_change:.6f}."
        )

    water_tolerance = max(tolerance, 0.005 * max_allowed_water_saturation_change)
    max_tolerable_water_saturation_change = max_allowed_water_saturation_change * (
        1 + water_tolerance
    )
    if maximum_water_saturation_change > max_tolerable_water_saturation_change:
        violated = True
        # If water saturation change is the largest so far, update the max values
        if maximum_water_saturation_change > max_phase_saturation_change:
            max_phase_saturation_change = maximum_water_saturation_change
            max_allowed_phase_saturation_change = max_allowed_water_saturation_change
        messages.append(
            f"Water saturation change {maximum_water_saturation_change:.6f} exceeded maximum allowed {max_allowed_water_saturation_change:.6f}."
        )

    gas_tolerance = max(tolerance, 0.005 * max_allowed_gas_saturation_change)
    max_tolerable_gas_saturation_change = max_allowed_gas_saturation_change * (
        1 + gas_tolerance
    )
    if maximum_gas_saturation_change > max_tolerable_gas_saturation_change:
        violated = True
        # If gas saturation change is the largest so far, update the max values
        if maximum_gas_saturation_change > max_phase_saturation_change:
            max_phase_saturation_change = maximum_gas_saturation_change
            max_allowed_phase_saturation_change = max_allowed_gas_saturation_change
        messages.append(
            f"Gas saturation change {maximum_gas_saturation_change:.6f} exceeded maximum allowed {max_allowed_gas_saturation_change:.6f}."
        )

    message = "\n".join(messages) if messages else None
    return SaturationChangeCheckResult(
        violated=violated,
        max_phase_saturation_change=max_phase_saturation_change,
        max_allowed_phase_saturation_change=max_allowed_phase_saturation_change,
        message=message,
    )


def _make_rates(grid_shape: NDimension) -> Rates[float, NDimension]:
    return Rates(
        oil=SparseTensor(grid_shape, dtype=float),
        water=SparseTensor(grid_shape, dtype=float),
        gas=SparseTensor(grid_shape, dtype=float),
    )


def _make_bhps(grid_shape: NDimension) -> BottomHolePressures[float, NDimension]:
    return BottomHolePressures(
        oil=BottomHolePressure(grid_shape, dtype=float),
        water=BottomHolePressure(grid_shape, dtype=float),
        gas=BottomHolePressure(grid_shape, dtype=float),
    )


def _rates_proxy(
    rates: Rates[float, NDimension],
) -> PhaseTensorsProxy[float, NDimension]:
    return PhaseTensorsProxy(oil=rates.oil, water=rates.water, gas=rates.gas)


def _bhps_proxy(
    bhps: BottomHolePressures[float, NDimension],
) -> PhaseTensorsProxy[float, NDimension]:
    return PhaseTensorsProxy(oil=bhps.oil, water=bhps.water, gas=bhps.gas)


def _run_impes_step(
    time_step: int,
    grid_shape: ThreeDimensions,
    cell_dimension: typing.Tuple[float, float],
    thickness_grid: NDimensionalGrid[ThreeDimensions],
    padded_thickness_grid: NDimensionalGrid[ThreeDimensions],
    padded_elevation_grid: NDimensionalGrid[ThreeDimensions],
    time_step_size: float,
    time: float,
    padded_rock_properties: RockProperties[ThreeDimensions],
    padded_fluid_properties: FluidProperties[ThreeDimensions],
    padded_saturation_history: SaturationHistory[ThreeDimensions],
    padded_relative_mobility_grids: RelativeMobilityGrids[ThreeDimensions],
    padded_capillary_pressure_grids: CapillaryPressureGrids[ThreeDimensions],
    wells: Wells[ThreeDimensions],
    miscibility_model: MiscibilityModel,
    config: Config,
    boundary_conditions: BoundaryConditions[ThreeDimensions],
    well_indices_cache: WellIndicesCache,
    dtype: npt.DTypeLike,
    pad_width: int = 1,
    min_valid_pressure: float = 14.7,
    max_valid_pressure: float = 14700,
    saturation_epsilon: float = 1e-12,
) -> StepResult[ThreeDimensions]:
    """
    Execute one time step using (semi-implicit) IMPES (Implicit Pressure, Explicit Saturation).

    :param time_step: Current time step index.
    :param padded_zeros_grid: Padded grid of zeros for rate tracking.
    :param grid_shape: Original model grid shape (nx, ny, nz).
    :param cell_dimension: Tuple of cell dimensions (dx, dy).
    :param thickness_grid: Un-padded thickness grid.
    :param padded_thickness_grid: Padded thickness grid.
    :param padded_elevation_grid: Padded elevation grid.
    :param time_step_size: Size of the current time step.
    :param time: Total simulation time elapsed. This time step inclusive.
    :param padded_rock_properties: Padded rock properties.
    :param padded_fluid_properties: Padded fluid properties.
    :param padded_saturation_history: Padded saturation history.
    :param padded_relative_mobility_grids: Padded relative mobility grids.
    :param padded_capillary_pressure_grids: Padded capillary pressure grids.
    :param wells: Wells in the reservoir.
    :param miscibility_model: Miscibility model used in the simulation.
    :param config: Simulation configuration.
    :param boundary_conditions: Model boundary conditions.
    :param pad_width: Number of ghost cells used for grid padding.
    :return: `StepResult` containing updated rates and fluid properties.
    """
    old_pressure_grid = padded_fluid_properties.pressure_grid.copy()

    logger.debug("Evolving pressure (implicit)...")
    injection_rates = _make_rates(grid_shape)
    production_rates = _make_rates(grid_shape)
    injection_bhps = _make_bhps(grid_shape)
    production_bhps = _make_bhps(grid_shape)
    pressure_result = implicit.evolve_pressure(
        cell_dimension=cell_dimension,
        thickness_grid=padded_thickness_grid,
        elevation_grid=padded_elevation_grid,
        time_step=time_step,
        time_step_size=time_step_size,
        time=time,
        rock_properties=padded_rock_properties,
        fluid_properties=padded_fluid_properties,
        relative_mobility_grids=padded_relative_mobility_grids,
        capillary_pressure_grids=padded_capillary_pressure_grids,
        wells=wells,
        config=config,
        injection_rates=_rates_proxy(injection_rates),
        production_rates=_rates_proxy(production_rates),
        injection_bhps=_bhps_proxy(injection_bhps),
        production_bhps=_bhps_proxy(production_bhps),
        pad_width=pad_width,
        well_indices_cache=well_indices_cache,
    )
    if not pressure_result.success:
        logger.error(
            f"Implicit pressure evolution failed at time step {time_step}: \n{pressure_result.message}"
        )
        return StepResult(
            fluid_properties=padded_fluid_properties,
            rock_properties=padded_rock_properties,
            saturation_history=padded_saturation_history,
            success=False,
            message=pressure_result.message,
        )

    pressure_solution = pressure_result.value
    padded_pressure_grid = pressure_solution.pressure_grid
    maximum_pressure_change = pressure_solution.maximum_pressure_change
    maximum_allowed_pressure_change = config.maximum_pressure_change
    if maximum_pressure_change > maximum_allowed_pressure_change:
        message = (
            f"Pressure change {maximum_pressure_change:.6f} psi "
            f"exceeded maximum allowed {maximum_allowed_pressure_change:.6f} psi "
            f"at time step {time_step}."
        )
        logger.error(message)
        return StepResult(
            fluid_properties=padded_fluid_properties,
            rock_properties=padded_rock_properties,
            saturation_history=padded_saturation_history,
            success=False,
            message=message,
            timer_kwargs={
                "maximum_pressure_change": maximum_pressure_change,
                "maximum_allowed_pressure_change": maximum_allowed_pressure_change,
            },
        )

    # Check for any out-of-range pressures
    pressure_validation_result = _validate_pressure_range(
        padded_pressure_grid=padded_pressure_grid,
        time_step=time_step,
        padded_fluid_properties=padded_fluid_properties,
        padded_rock_properties=padded_rock_properties,
        padded_saturation_history=padded_saturation_history,
    )
    if pressure_validation_result is not None:
        return pressure_validation_result

    # Apply boundary conditions to new pressure grid
    logger.debug(
        f"Applying pressure boundary condition after pressure evolution for time step {time_step}..."
    )
    apply_pressure_boundary_condition(
        padded_pressure_grid=padded_pressure_grid,
        boundary_conditions=boundary_conditions,
        cell_dimension=cell_dimension,
        grid_shape=grid_shape,
        thickness_grid=thickness_grid,
        time=time,
        pad_width=pad_width,
    )
    logger.debug("Pressure boundary condition applied.")

    # Clamp pressures to valid range just for additional safety and to remove numerical noise
    padded_pressure_grid = np.clip(
        padded_pressure_grid, min_valid_pressure, max_valid_pressure
    ).astype(dtype, copy=False)

    # Update fluid properties with new pressure grid
    logger.debug("Updating fluid properties with new pressure grid...")

    padded_fluid_properties = attrs.evolve(
        padded_fluid_properties, pressure_grid=padded_pressure_grid
    )
    logger.debug("Pressure evolution completed!")

    # IMPES specific: Update PVT properties after implicit pressure solve
    # but before explicit saturation evolution.
    # This ensures saturation transport uses the correct fluid properties at the new pressure.
    # Pressure changes affect viscosity, density, compressibility → affects mobility → affects transport.
    logger.debug("Updating PVT fluid properties for saturation evolution")

    # Save old solution gas-to-oil ratio and oil formation volume factor
    # before PVT update (needed for gas liberation flash)
    old_solution_gas_to_oil_ratio_grid = (
        padded_fluid_properties.solution_gas_to_oil_ratio_grid.copy()
    )
    old_oil_formation_volume_factor_grid = (
        padded_fluid_properties.oil_formation_volume_factor_grid.copy()
    )
    old_gas_solubility_in_water_grid = (
        padded_fluid_properties.gas_solubility_in_water_grid.copy()
    )
    old_water_formation_volume_factor_grid = (
        padded_fluid_properties.water_formation_volume_factor_grid.copy()
    )
    # Save pre-flash saturations to detect flash-induced saturation changes
    old_oil_saturation_grid = padded_fluid_properties.oil_saturation_grid.copy()
    old_gas_saturation_grid = padded_fluid_properties.gas_saturation_grid.copy()
    old_water_saturation_grid = padded_fluid_properties.water_saturation_grid.copy()

    padded_fluid_properties = update_pvt_grids(
        fluid_properties=padded_fluid_properties,
        wells=wells,
        miscibility_model=miscibility_model,
        pvt_tables=config.pvt_tables,
        freeze_saturation_pressure=config.freeze_saturation_pressure,
    )

    # Solution gas liberation / re-dissolution flash step.
    # When pressure drops below bubble point, solution gas-to-oil ratio decreases
    # and dissolved gas comes out of solution as free gas.
    # This updates oil saturation, gas saturation, and solution gas-to-oil ratio.
    padded_fluid_properties = apply_solution_gas_updates(
        fluid_properties=padded_fluid_properties,
        old_solution_gas_to_oil_ratio_grid=old_solution_gas_to_oil_ratio_grid,
        old_oil_formation_volume_factor_grid=old_oil_formation_volume_factor_grid,
        old_gas_solubility_in_water_grid=old_gas_solubility_in_water_grid,
        old_water_formation_volume_factor_grid=old_water_formation_volume_factor_grid,
    )
    # Check flash-induced saturation changes before proceeding to saturation solver.
    # If the liberation flash itself violates the saturation change limits, reject
    # the step immediately, as there's no point running the full saturation solver.
    flash_gas_saturation_change = float(
        np.max(
            np.abs(
                padded_fluid_properties.gas_saturation_grid - old_gas_saturation_grid
            )
        )
    )
    flash_oil_saturation_change = float(
        np.max(
            np.abs(
                padded_fluid_properties.oil_saturation_grid - old_oil_saturation_grid
            )
        )
    )
    flash_water_saturation_change = float(
        np.max(
            np.abs(
                padded_fluid_properties.water_saturation_grid
                - old_water_saturation_grid
            )
        )
    )
    flash_saturation_check = _check_saturation_changes(
        maximum_oil_saturation_change=flash_oil_saturation_change,
        maximum_water_saturation_change=flash_water_saturation_change,
        maximum_gas_saturation_change=flash_gas_saturation_change,
        max_allowed_oil_saturation_change=config.maximum_oil_saturation_change,
        max_allowed_water_saturation_change=config.maximum_water_saturation_change,
        max_allowed_gas_saturation_change=config.maximum_gas_saturation_change,
    )
    if flash_saturation_check.violated:
        message = (
            f"Solution gas liberation flash at time step {time_step} violated "
            f"saturation change limits: {flash_saturation_check.message}"
        )
        logger.debug(message)
        return StepResult(
            fluid_properties=padded_fluid_properties,
            rock_properties=padded_rock_properties,
            saturation_history=padded_saturation_history,
            success=False,
            message=message,
            timer_kwargs={
                "maximum_saturation_change": flash_saturation_check.max_phase_saturation_change,
                "maximum_allowed_saturation_change": flash_saturation_check.max_allowed_phase_saturation_change,
            },
        )

    # Updated fluid properties again as solution gas-to-oil ratio may have changed
    # and some PVT properties depend on it
    padded_fluid_properties = update_pvt_grids(
        fluid_properties=padded_fluid_properties,
        wells=wells,
        miscibility_model=miscibility_model,
        pvt_tables=config.pvt_tables,
        freeze_saturation_pressure=config.freeze_saturation_pressure,
    )

    # Apply boundary conditions to updated saturation grids
    logger.debug(
        f"Applying saturations boundary conditions after solution gas evolution for time step {time_step}..."
    )
    apply_saturation_boundary_conditions(
        padded_water_saturation_grid=padded_fluid_properties.water_saturation_grid,
        padded_oil_saturation_grid=padded_fluid_properties.oil_saturation_grid,
        padded_gas_saturation_grid=padded_fluid_properties.gas_saturation_grid,
        boundary_conditions=boundary_conditions,
        cell_dimension=cell_dimension,
        grid_shape=grid_shape,
        thickness_grid=thickness_grid,
        time=time,
        pad_width=pad_width,
    )
    logger.debug("Saturation boundary conditions applied.")

    # Rebuild relative permeability grids from post-flash saturations.
    # This ensures cells with newly liberated gas get krg > 0 for transport.
    logger.debug("Rebuilding relative permeability and mobility grids...")
    krw_grid, kro_grid, krg_grid = build_three_phase_relative_permeabilities_grids(
        water_saturation_grid=padded_fluid_properties.water_saturation_grid,
        oil_saturation_grid=padded_fluid_properties.oil_saturation_grid,
        gas_saturation_grid=padded_fluid_properties.gas_saturation_grid,
        irreducible_water_saturation_grid=padded_rock_properties.irreducible_water_saturation_grid,
        residual_oil_saturation_water_grid=padded_rock_properties.residual_oil_saturation_water_grid,
        residual_oil_saturation_gas_grid=padded_rock_properties.residual_oil_saturation_gas_grid,
        residual_gas_saturation_grid=padded_rock_properties.residual_gas_saturation_grid,
        relative_permeability_table=config.rock_fluid_tables.relative_permeability_table,
        phase_appearance_tolerance=config.phase_appearance_tolerance,
    )
    # Rebuild mobility grids (kr/μ) from new relative permeabilities and updated viscosities
    (
        padded_water_relative_mobility_grid,
        padded_oil_relative_mobility_grid,
        padded_gas_relative_mobility_grid,
    ) = build_three_phase_relative_mobilities_grids(
        oil_relative_permeability_grid=kro_grid,
        water_relative_permeability_grid=krw_grid,
        gas_relative_permeability_grid=krg_grid,
        water_viscosity_grid=padded_fluid_properties.water_viscosity_grid,
        oil_viscosity_grid=padded_fluid_properties.oil_effective_viscosity_grid,
        gas_viscosity_grid=padded_fluid_properties.gas_viscosity_grid,
    )

    # Clamp relative mobility grids to avoid numerical issues
    padded_water_relative_mobility_grid = config.relative_mobility_range["water"].clip(
        padded_water_relative_mobility_grid
    )
    padded_oil_relative_mobility_grid = config.relative_mobility_range["oil"].clip(
        padded_oil_relative_mobility_grid
    )
    padded_gas_relative_mobility_grid = config.relative_mobility_range["gas"].clip(
        padded_gas_relative_mobility_grid
    )
    padded_relative_mobility_grids = RelativeMobilityGrids(
        water_relative_mobility=padded_water_relative_mobility_grid,
        oil_relative_mobility=padded_oil_relative_mobility_grid,
        gas_relative_mobility=padded_gas_relative_mobility_grid,
    )
    logger.debug("Relative mobility grids rebuilt for saturation evolution.")

    # Saturation evolution (explicit)
    logger.debug("Evolving saturation (explicit)...")
    # Compute pressure change grid for PVT volume correction
    pressure_change_grid = padded_pressure_grid - old_pressure_grid

    if miscibility_model == "immiscible":
        saturation_result = explicit.evolve_saturation(
            cell_dimension=cell_dimension,
            thickness_grid=padded_thickness_grid,
            elevation_grid=padded_elevation_grid,
            time_step=time_step,
            time_step_size=time_step_size,
            rock_properties=padded_rock_properties,
            fluid_properties=padded_fluid_properties,
            relative_mobility_grids=padded_relative_mobility_grids,
            capillary_pressure_grids=padded_capillary_pressure_grids,
            config=config,
            well_indices_cache=well_indices_cache,
            injection_rates=_rates_proxy(injection_rates),
            production_rates=_rates_proxy(production_rates),
            pressure_change_grid=pressure_change_grid,
            pad_width=pad_width,
        )
    else:
        saturation_result = explicit.evolve_miscible_saturation(
            cell_dimension=cell_dimension,
            thickness_grid=padded_thickness_grid,
            elevation_grid=padded_elevation_grid,
            time_step=time_step,
            time_step_size=time_step_size,
            time=time,
            rock_properties=padded_rock_properties,
            fluid_properties=padded_fluid_properties,
            relative_mobility_grids=padded_relative_mobility_grids,
            capillary_pressure_grids=padded_capillary_pressure_grids,
            wells=wells,
            config=config,
            well_indices_cache=well_indices_cache,
            injection_rates=_rates_proxy(injection_rates),
            production_rates=_rates_proxy(production_rates),
            pressure_change_grid=pressure_change_grid,
            pad_width=pad_width,
        )

    saturation_solution = saturation_result.value
    saturation_change_result = _check_saturation_changes(
        maximum_oil_saturation_change=saturation_solution.maximum_oil_saturation_change,
        maximum_water_saturation_change=saturation_solution.maximum_water_saturation_change,
        maximum_gas_saturation_change=saturation_solution.maximum_gas_saturation_change,
        max_allowed_oil_saturation_change=config.maximum_oil_saturation_change,
        max_allowed_water_saturation_change=config.maximum_water_saturation_change,
        max_allowed_gas_saturation_change=config.maximum_gas_saturation_change,
    )
    timer_kwargs = {
        "maximum_cfl_encountered": saturation_solution.maximum_cfl_encountered,
        "cfl_threshold": saturation_solution.cfl_threshold,
        "maximum_pressure_change": maximum_pressure_change,
        "maximum_allowed_pressure_change": maximum_allowed_pressure_change,
        "maximum_saturation_change": saturation_change_result.max_phase_saturation_change
        or None,
        "maximum_allowed_saturation_change": saturation_change_result.max_allowed_phase_saturation_change
        or None,
    }

    if not saturation_result.success:
        logger.error(
            f"Explicit saturation evolution failed at time step {time_step}: \n{saturation_result.message}"
        )
        return StepResult(
            fluid_properties=padded_fluid_properties,
            rock_properties=padded_rock_properties,
            saturation_history=padded_saturation_history,
            success=False,
            message=saturation_result.message,
            timer_kwargs=timer_kwargs,
        )

    if saturation_change_result.violated:
        message = f"""
        At time step {time_step}, saturation change limits were violated:
        {saturation_change_result.message}

        Oil saturation change: {saturation_solution.maximum_oil_saturation_change:.6f},
        Water saturation change: {saturation_solution.maximum_water_saturation_change:.6f},
        Gas saturation change: {saturation_solution.maximum_gas_saturation_change:.6f}.
        """
        logger.debug(message)
        return StepResult(
            fluid_properties=padded_fluid_properties,
            rock_properties=padded_rock_properties,
            saturation_history=padded_saturation_history,
            success=False,
            message=message,
            timer_kwargs=timer_kwargs,
        )

    logger.debug("Updating fluid properties with new saturation grids...")
    padded_water_saturation_grid = saturation_solution.water_saturation_grid.astype(
        dtype, copy=False
    )
    padded_oil_saturation_grid = saturation_solution.oil_saturation_grid.astype(
        dtype, copy=False
    )
    padded_gas_saturation_grid = saturation_solution.gas_saturation_grid.astype(
        dtype, copy=False
    )
    padded_solvent_concentration_grid = saturation_solution.solvent_concentration_grid

    # Apply boundary conditions to updated saturation grids again
    logger.debug(
        f"Applying saturations boundary conditions after saturation evolution for time step {time_step}..."
    )
    apply_saturation_boundary_conditions(
        padded_water_saturation_grid=padded_water_saturation_grid,
        padded_oil_saturation_grid=padded_oil_saturation_grid,
        padded_gas_saturation_grid=padded_gas_saturation_grid,
        boundary_conditions=boundary_conditions,
        cell_dimension=cell_dimension,
        grid_shape=grid_shape,
        thickness_grid=thickness_grid,
        time=time,
        pad_width=pad_width,
    )
    logger.debug("Saturation boundary conditions applied.")

    if padded_solvent_concentration_grid is None:
        padded_fluid_properties = attrs.evolve(
            padded_fluid_properties,
            water_saturation_grid=padded_water_saturation_grid,
            oil_saturation_grid=padded_oil_saturation_grid,
            gas_saturation_grid=padded_gas_saturation_grid,
        )
    else:
        padded_fluid_properties = attrs.evolve(
            padded_fluid_properties,
            water_saturation_grid=padded_water_saturation_grid,
            oil_saturation_grid=padded_oil_saturation_grid,
            gas_saturation_grid=padded_gas_saturation_grid,
            solvent_concentration_grid=padded_solvent_concentration_grid.astype(
                dtype, copy=False
            ),
        )

    if config.normalize_saturations:
        # Normalize saturations (in-place) to ensure So + Sw + Sg = 1.0
        normalize_saturations(
            oil_saturation_grid=padded_fluid_properties.oil_saturation_grid,
            water_saturation_grid=padded_fluid_properties.water_saturation_grid,
            gas_saturation_grid=padded_fluid_properties.gas_saturation_grid,
            saturation_epsilon=saturation_epsilon,
        )

    # Update residual saturation grids based on new saturations
    padded_rock_properties, padded_saturation_history = (
        update_residual_saturation_grids(
            rock_properties=padded_rock_properties,
            saturation_history=padded_saturation_history,
            water_saturation_grid=padded_fluid_properties.water_saturation_grid,
            gas_saturation_grid=padded_fluid_properties.gas_saturation_grid,
            residual_oil_drainage_ratio_water_flood=config.residual_oil_drainage_ratio_water_flood,
            residual_oil_drainage_ratio_gas_flood=config.residual_oil_drainage_ratio_gas_flood,
            residual_gas_drainage_ratio=config.residual_gas_drainage_ratio,
        )
    )
    logger.debug("Saturation evolution completed!")
    return StepResult(
        fluid_properties=padded_fluid_properties,
        rock_properties=padded_rock_properties,
        saturation_history=padded_saturation_history,
        injection_rates=injection_rates,
        production_rates=production_rates,
        injection_bhps=injection_bhps,
        production_bhps=production_bhps,
        success=True,
        message=saturation_result.message,
        timer_kwargs=timer_kwargs,
    )


def _run_sequential_implicit_step(
    time_step: int,
    grid_shape: ThreeDimensions,
    cell_dimension: typing.Tuple[float, float],
    thickness_grid: NDimensionalGrid[ThreeDimensions],
    padded_thickness_grid: NDimensionalGrid[ThreeDimensions],
    padded_elevation_grid: NDimensionalGrid[ThreeDimensions],
    time_step_size: float,
    time: float,
    padded_rock_properties: RockProperties[ThreeDimensions],
    padded_fluid_properties: FluidProperties[ThreeDimensions],
    padded_saturation_history: SaturationHistory[ThreeDimensions],
    padded_relative_mobility_grids: RelativeMobilityGrids[ThreeDimensions],
    padded_capillary_pressure_grids: CapillaryPressureGrids[ThreeDimensions],
    wells: Wells[ThreeDimensions],
    miscibility_model: MiscibilityModel,
    config: Config,
    boundary_conditions: BoundaryConditions[ThreeDimensions],
    well_indices_cache: WellIndicesCache,
    dtype: npt.DTypeLike,
    pad_width: int = 1,
    min_valid_pressure: float = 14.7,
    max_valid_pressure: float = 14700,
    saturation_epsilon: float = 1e-12,
) -> StepResult[ThreeDimensions]:
    """
    Execute one time step using Sequential Implicit (SI) scheme.

    Pressure is solved implicitly, then saturation is solved
    implicitly using Newton-Raphson iteration. This eliminates the CFL
    stability constraint on saturation transport, allowing larger timesteps.

    :param time_step: Current time step index.
    :param padded_zeros_grid: Padded grid of zeros for rate tracking.
    :param grid_shape: Original model grid shape (nx, ny, nz).
    :param cell_dimension: Tuple of cell dimensions (dx, dy).
    :param thickness_grid: Un-padded thickness grid.
    :param padded_thickness_grid: Padded thickness grid.
    :param padded_elevation_grid: Padded elevation grid.
    :param time_step_size: Size of the current time step.
    :param time: Total simulation time elapsed. This time step inclusive.
    :param padded_rock_properties: Padded rock properties.
    :param padded_fluid_properties: Padded fluid properties.
    :param padded_saturation_history: Padded saturation history.
    :param padded_relative_mobility_grids: Padded relative mobility grids.
    :param padded_capillary_pressure_grids: Padded capillary pressure grids.
    :param wells: Wells in the reservoir.
    :param miscibility_model: Miscibility model used in the simulation.
    :param config: Simulation configuration.
    :param boundary_conditions: Model boundary conditions.
    :param pad_width: Number of ghost cells used for grid padding.
    :return: `StepResult` containing updated rates and fluid properties.
    """
    # Save old pressure grid before implicit solve (needed for PVT volume correction)
    old_pressure_grid = padded_fluid_properties.pressure_grid.copy()

    logger.debug("Evolving pressure (implicit)...")
    injection_rates = _make_rates(grid_shape)
    production_rates = _make_rates(grid_shape)
    injection_bhps = _make_bhps(grid_shape)
    production_bhps = _make_bhps(grid_shape)
    pressure_result = implicit.evolve_pressure(
        cell_dimension=cell_dimension,
        thickness_grid=padded_thickness_grid,
        elevation_grid=padded_elevation_grid,
        time_step=time_step,
        time_step_size=time_step_size,
        time=time,
        rock_properties=padded_rock_properties,
        fluid_properties=padded_fluid_properties,
        relative_mobility_grids=padded_relative_mobility_grids,
        capillary_pressure_grids=padded_capillary_pressure_grids,
        wells=wells,
        config=config,
        well_indices_cache=well_indices_cache,
        injection_rates=_rates_proxy(injection_rates),
        production_rates=_rates_proxy(production_rates),
        injection_bhps=_bhps_proxy(injection_bhps),
        production_bhps=_bhps_proxy(production_bhps),
        pad_width=pad_width,
    )
    if not pressure_result.success:
        logger.error(
            f"Implicit pressure evolution failed at time step {time_step}: \n{pressure_result.message}"
        )
        return StepResult(
            fluid_properties=padded_fluid_properties,
            rock_properties=padded_rock_properties,
            saturation_history=padded_saturation_history,
            success=False,
            message=pressure_result.message,
        )

    pressure_solution = pressure_result.value
    padded_pressure_grid = pressure_solution.pressure_grid
    maximum_pressure_change = pressure_solution.maximum_pressure_change
    maximum_allowed_pressure_change = config.maximum_pressure_change
    if maximum_pressure_change > maximum_allowed_pressure_change:
        message = (
            f"Pressure change {maximum_pressure_change:.6f} psi "
            f"exceeded maximum allowed {maximum_allowed_pressure_change:.6f} psi "
            f"at time step {time_step}."
        )
        logger.error(message)
        return StepResult(
            fluid_properties=padded_fluid_properties,
            rock_properties=padded_rock_properties,
            saturation_history=padded_saturation_history,
            success=False,
            message=message,
            timer_kwargs={
                "maximum_pressure_change": maximum_pressure_change,
                "maximum_allowed_pressure_change": maximum_allowed_pressure_change,
            },
        )

    # Check for any out-of-range pressures
    pressure_validation_result = _validate_pressure_range(
        padded_pressure_grid=padded_pressure_grid,
        time_step=time_step,
        padded_fluid_properties=padded_fluid_properties,
        padded_rock_properties=padded_rock_properties,
        padded_saturation_history=padded_saturation_history,
    )
    if pressure_validation_result is not None:
        return pressure_validation_result

    # Apply boundary conditions to new pressure grid
    logger.debug(
        f"Applying pressure boundary condition after pressure evolution for time step {time_step}..."
    )
    apply_pressure_boundary_condition(
        padded_pressure_grid=padded_pressure_grid,
        boundary_conditions=boundary_conditions,
        cell_dimension=cell_dimension,
        grid_shape=grid_shape,
        thickness_grid=thickness_grid,
        time=time,
        pad_width=pad_width,
    )
    logger.debug("Pressure boundary conditions applied.")

    padded_pressure_grid = np.clip(
        padded_pressure_grid, min_valid_pressure, max_valid_pressure
    ).astype(dtype, copy=False)

    old_solution_gas_to_oil_ratio_grid = (
        padded_fluid_properties.solution_gas_to_oil_ratio_grid.copy()
    )
    old_oil_formation_volume_factor_grid = (
        padded_fluid_properties.oil_formation_volume_factor_grid.copy()
    )
    old_gas_solubility_in_water_grid = (
        padded_fluid_properties.gas_solubility_in_water_grid.copy()
    )
    old_water_formation_volume_factor_grid = (
        padded_fluid_properties.water_formation_volume_factor_grid.copy()
    )
    # Save pre-flash saturations to detect flash-induced saturation changes
    old_oil_saturation_grid = padded_fluid_properties.oil_saturation_grid.copy()
    old_gas_saturation_grid = padded_fluid_properties.gas_saturation_grid.copy()
    old_water_saturation_grid = padded_fluid_properties.water_saturation_grid.copy()

    # PVT update at new pressure
    logger.debug("Updating PVT fluid properties for saturation evolution")
    padded_fluid_properties = attrs.evolve(
        padded_fluid_properties, pressure_grid=padded_pressure_grid
    )
    padded_fluid_properties = update_pvt_grids(
        fluid_properties=padded_fluid_properties,
        wells=wells,
        miscibility_model=miscibility_model,
        pvt_tables=config.pvt_tables,
        freeze_saturation_pressure=config.freeze_saturation_pressure,
    )

    # Solution gas liberation flash
    padded_fluid_properties = apply_solution_gas_updates(
        fluid_properties=padded_fluid_properties,
        old_solution_gas_to_oil_ratio_grid=old_solution_gas_to_oil_ratio_grid,
        old_oil_formation_volume_factor_grid=old_oil_formation_volume_factor_grid,
        old_gas_solubility_in_water_grid=old_gas_solubility_in_water_grid,
        old_water_formation_volume_factor_grid=old_water_formation_volume_factor_grid,
    )

    # Check flash-induced saturation changes before proceeding to saturation solver.
    # If the liberation flash itself violates the saturation change limits, reject
    # the step immediately, as there's no point running the full saturation solver.
    flash_gas_saturation_change = float(
        np.max(
            np.abs(
                padded_fluid_properties.gas_saturation_grid - old_gas_saturation_grid
            )
        )
    )
    flash_oil_saturation_change = float(
        np.max(
            np.abs(
                padded_fluid_properties.oil_saturation_grid - old_oil_saturation_grid
            )
        )
    )
    flash_water_saturation_change = float(
        np.max(
            np.abs(
                padded_fluid_properties.water_saturation_grid
                - old_water_saturation_grid
            )
        )
    )
    flash_saturation_check = _check_saturation_changes(
        maximum_oil_saturation_change=flash_oil_saturation_change,
        maximum_water_saturation_change=flash_water_saturation_change,
        maximum_gas_saturation_change=flash_gas_saturation_change,
        max_allowed_oil_saturation_change=config.maximum_oil_saturation_change,
        max_allowed_water_saturation_change=config.maximum_water_saturation_change,
        max_allowed_gas_saturation_change=config.maximum_gas_saturation_change,
    )
    if flash_saturation_check.violated:
        message = (
            f"Solution gas liberation flash at time step {time_step} violated "
            f"saturation change limits: {flash_saturation_check.message}"
        )
        logger.debug(message)
        return StepResult(
            fluid_properties=padded_fluid_properties,
            rock_properties=padded_rock_properties,
            saturation_history=padded_saturation_history,
            success=False,
            message=message,
            timer_kwargs={
                "maximum_saturation_change": flash_saturation_check.max_phase_saturation_change,
                "maximum_allowed_saturation_change": flash_saturation_check.max_allowed_phase_saturation_change,
            },
        )

    # No need to apply saturation boundary conditions here, they are applied at the
    # start of the newton iteration in the implicit saturation solver

    # Update fluid properties again as solution gas-to-oil ratio may have changed
    # and some PVT properties depend on it
    padded_fluid_properties = update_pvt_grids(
        fluid_properties=padded_fluid_properties,
        wells=wells,
        miscibility_model=miscibility_model,
        pvt_tables=config.pvt_tables,
        freeze_saturation_pressure=config.freeze_saturation_pressure,
    )

    logger.debug("Evolving saturation (implicit, Newton-Raphson)...")
    pressure_change_grid = padded_pressure_grid - old_pressure_grid

    saturation_result = implicit.evolve_saturation(
        grid_shape=grid_shape,
        cell_dimension=cell_dimension,
        thickness_grid=padded_thickness_grid,
        elevation_grid=padded_elevation_grid,
        time_step_size=time_step_size,
        time=time,
        rock_properties=padded_rock_properties,
        fluid_properties=padded_fluid_properties,
        config=config,
        well_indices_cache=well_indices_cache,
        injection_rates=_rates_proxy(injection_rates),
        production_rates=_rates_proxy(production_rates),
        injection_bhps=_bhps_proxy(injection_bhps),
        production_bhps=_bhps_proxy(production_bhps),
        boundary_conditions=boundary_conditions,
        pressure_change_grid=pressure_change_grid,
        pad_width=pad_width,
    )
    saturation_solution = saturation_result.value
    saturation_change_result = _check_saturation_changes(
        maximum_oil_saturation_change=saturation_solution.maximum_oil_saturation_change,
        maximum_water_saturation_change=saturation_solution.maximum_water_saturation_change,
        maximum_gas_saturation_change=saturation_solution.maximum_gas_saturation_change,
        max_allowed_oil_saturation_change=config.maximum_oil_saturation_change,
        max_allowed_water_saturation_change=config.maximum_water_saturation_change,
        max_allowed_gas_saturation_change=config.maximum_gas_saturation_change,
    )
    timer_kwargs = {
        "maximum_pressure_change": maximum_pressure_change,
        "maximum_allowed_pressure_change": maximum_allowed_pressure_change,
        "maximum_saturation_change": saturation_change_result.max_phase_saturation_change
        or None,
        "maximum_allowed_saturation_change": saturation_change_result.max_allowed_phase_saturation_change
        or None,
        "newton_iterations": saturation_solution.newton_iterations,
    }

    if not saturation_result.success:
        logger.error(
            f"Implicit saturation evolution failed at time step {time_step}: \n{saturation_result.message}"
        )
        return StepResult(
            fluid_properties=padded_fluid_properties,
            rock_properties=padded_rock_properties,
            saturation_history=padded_saturation_history,
            success=False,
            message=saturation_result.message,
            timer_kwargs=timer_kwargs,
        )

    if saturation_change_result.violated:
        message = f"""
        At time step {time_step}, saturation change limits were violated:
        {saturation_change_result.message}

        Oil saturation change: {saturation_solution.maximum_oil_saturation_change:.6f},
        Water saturation change: {saturation_solution.maximum_water_saturation_change:.6f},
        Gas saturation change: {saturation_solution.maximum_gas_saturation_change:.6f}.
        """
        logger.debug(message)
        return StepResult(
            fluid_properties=padded_fluid_properties,
            rock_properties=padded_rock_properties,
            saturation_history=padded_saturation_history,
            success=False,
            message=message,
            timer_kwargs=timer_kwargs,
        )

    # Update fluid properties with new saturations
    logger.debug("Updating fluid properties with new saturation grids...")
    padded_water_saturation_grid = saturation_solution.water_saturation_grid.astype(
        dtype, copy=False
    )
    padded_oil_saturation_grid = saturation_solution.oil_saturation_grid.astype(
        dtype, copy=False
    )
    padded_gas_saturation_grid = saturation_solution.gas_saturation_grid.astype(
        dtype, copy=False
    )

    # Apply boundary conditions to updated saturation grids again
    logger.debug(
        f"Applying saturations boundary conditions after saturation evolution for time step {time_step}..."
    )
    apply_saturation_boundary_conditions(
        padded_water_saturation_grid=padded_water_saturation_grid,
        padded_oil_saturation_grid=padded_oil_saturation_grid,
        padded_gas_saturation_grid=padded_gas_saturation_grid,
        boundary_conditions=boundary_conditions,
        cell_dimension=cell_dimension,
        grid_shape=grid_shape,
        thickness_grid=thickness_grid,
        time=time,
        pad_width=pad_width,
    )
    logger.debug("Saturation boundary conditions applied.")

    padded_fluid_properties = attrs.evolve(
        padded_fluid_properties,
        water_saturation_grid=padded_water_saturation_grid,
        oil_saturation_grid=padded_oil_saturation_grid,
        gas_saturation_grid=padded_gas_saturation_grid,
    )

    if config.normalize_saturations:
        normalize_saturations(
            oil_saturation_grid=padded_fluid_properties.oil_saturation_grid,
            water_saturation_grid=padded_fluid_properties.water_saturation_grid,
            gas_saturation_grid=padded_fluid_properties.gas_saturation_grid,
            saturation_epsilon=saturation_epsilon,
        )

    # Update residual saturations
    padded_rock_properties, padded_saturation_history = (
        update_residual_saturation_grids(
            rock_properties=padded_rock_properties,
            saturation_history=padded_saturation_history,
            water_saturation_grid=padded_fluid_properties.water_saturation_grid,
            gas_saturation_grid=padded_fluid_properties.gas_saturation_grid,
            residual_oil_drainage_ratio_water_flood=config.residual_oil_drainage_ratio_water_flood,
            residual_oil_drainage_ratio_gas_flood=config.residual_oil_drainage_ratio_gas_flood,
            residual_gas_drainage_ratio=config.residual_gas_drainage_ratio,
        )
    )

    logger.debug("Sequential implicit step completed!")
    return StepResult(
        fluid_properties=padded_fluid_properties,
        rock_properties=padded_rock_properties,
        saturation_history=padded_saturation_history,
        injection_rates=injection_rates,
        production_rates=production_rates,
        injection_bhps=injection_bhps,
        production_bhps=production_bhps,
        success=True,
        message=saturation_result.message,
        timer_kwargs=timer_kwargs,
    )


def _run_full_sequential_implicit_step(
    time_step: int,
    grid_shape: ThreeDimensions,
    cell_dimension: typing.Tuple[float, float],
    thickness_grid: NDimensionalGrid[ThreeDimensions],
    padded_thickness_grid: NDimensionalGrid[ThreeDimensions],
    padded_elevation_grid: NDimensionalGrid[ThreeDimensions],
    time_step_size: float,
    time: float,
    padded_rock_properties: RockProperties[ThreeDimensions],
    padded_fluid_properties: FluidProperties[ThreeDimensions],
    padded_saturation_history: SaturationHistory[ThreeDimensions],
    padded_relative_mobility_grids: RelativeMobilityGrids[ThreeDimensions],
    padded_capillary_pressure_grids: CapillaryPressureGrids[ThreeDimensions],
    wells: Wells[ThreeDimensions],
    miscibility_model: MiscibilityModel,
    config: Config,
    boundary_conditions: BoundaryConditions[ThreeDimensions],
    well_indices_cache: WellIndicesCache,
    dtype: npt.DTypeLike,
    pad_width: int = 1,
    min_valid_pressure: float = 14.7,
    max_valid_pressure: float = 14700,
    saturation_epsilon: float = 1e-12,
) -> StepResult[ThreeDimensions]:
    """
    Execute one time step using Sequential Implicit (SI) scheme with outer iterations.

    Pressure is solved implicitly, then saturation is solved implicitly using Newton-Raphson.
    An outer iteration loop enforces coupling consistency between pressure and saturation
    until convergence or maximum iterations is reached.

    :param time_step: Current time step index.
    :param grid_shape: Original model grid shape (nx, ny, nz).
    :param cell_dimension: Tuple of cell dimensions (dx, dy).
    :param thickness_grid: Un-padded thickness grid.
    :param padded_thickness_grid: Padded thickness grid.
    :param padded_elevation_grid: Padded elevation grid.
    :param time_step_size: Size of the current time step.
    :param time: Total simulation time elapsed. This time step inclusive.
    :param padded_rock_properties: Padded rock properties.
    :param padded_fluid_properties: Padded fluid properties.
    :param padded_saturation_history: Padded saturation history.
    :param padded_relative_mobility_grids: Padded relative mobility grids.
    :param padded_capillary_pressure_grids: Padded capillary pressure grids.
    :param wells: Wells in the reservoir.
    :param miscibility_model: Miscibility model used in the simulation.
    :param config: Simulation configuration.
    :param boundary_conditions: Model boundary conditions.
    :param pad_width: Number of ghost cells used for grid padding.
    :return: `StepResult` containing updated rates and fluid properties.
    """
    saturation_tolerance = config.saturation_outer_convergence_tolerance
    pressure_tolerance = config.pressure_outer_convergence_tolerance
    maximum_newton_iterations = config.maximum_newton_iterations
    maximum_outer_iterations = config.maximum_outer_iterations

    logger.debug(
        f"Outer iteration tolerances - "
        f"saturation (absolute): {saturation_tolerance:.2e}, "
        f"pressure (relative): {pressure_tolerance:.2e}"
    )

    # Snapshots of the start-of-timestep state, updated after each outer iteration
    # to compute inter-iterate drift for the convergence check.
    prev_pressure_grid = padded_fluid_properties.pressure_grid.copy()
    prev_water_saturation_grid = padded_fluid_properties.water_saturation_grid.copy()
    prev_oil_saturation_grid = padded_fluid_properties.oil_saturation_grid.copy()
    prev_gas_saturation_grid = padded_fluid_properties.gas_saturation_grid.copy()

    # Working state updated each outer iteration
    iter_fluid_properties = padded_fluid_properties
    iter_relative_mobility_grids = padded_relative_mobility_grids
    iter_capillary_pressure_grids = padded_capillary_pressure_grids

    injection_rates = _make_rates(grid_shape)
    production_rates = _make_rates(grid_shape)
    injection_bhps = _make_bhps(grid_shape)
    production_bhps = _make_bhps(grid_shape)

    outer_converged = False
    # Initialised before the loop so they are always bound after it
    saturation_result = None
    saturation_solution = None
    saturation_change_result = None
    maximum_pressure_change = 0.0
    final_timer_kwargs: typing.Dict[str, typing.Any] = {}

    logger.debug(
        f"Starting outer iteration loop (max {maximum_outer_iterations} iterations) "
        f"at time step {time_step}..."
    )

    for iteration in range(maximum_outer_iterations):
        logger.debug(f"Outer iteration {iteration + 1}/{maximum_outer_iterations}")
        injection_rates = _make_rates(grid_shape)
        production_rates = _make_rates(grid_shape)
        injection_bhps = _make_bhps(grid_shape)
        production_bhps = _make_bhps(grid_shape)

        # Implicit pressure solve
        pressure_result = implicit.evolve_pressure(
            cell_dimension=cell_dimension,
            thickness_grid=padded_thickness_grid,
            elevation_grid=padded_elevation_grid,
            time_step=time_step,
            time_step_size=time_step_size,
            time=time,
            rock_properties=padded_rock_properties,
            fluid_properties=iter_fluid_properties,
            relative_mobility_grids=iter_relative_mobility_grids,
            capillary_pressure_grids=iter_capillary_pressure_grids,
            wells=wells,
            config=config,
            well_indices_cache=well_indices_cache,
            injection_rates=_rates_proxy(injection_rates),
            production_rates=_rates_proxy(production_rates),
            injection_bhps=_bhps_proxy(injection_bhps),
            production_bhps=_bhps_proxy(production_bhps),
            pad_width=pad_width,
        )

        if not pressure_result.success:
            logger.error(
                f"Implicit pressure solve failed at outer iteration "
                f"{iteration + 1}, time step {time_step}:\n"
                f"{pressure_result.message}"
            )
            return StepResult(
                fluid_properties=padded_fluid_properties,
                rock_properties=padded_rock_properties,
                saturation_history=padded_saturation_history,
                success=False,
                message=pressure_result.message,
            )

        pressure_solution = pressure_result.value
        padded_pressure_grid = pressure_solution.pressure_grid
        maximum_pressure_change = pressure_solution.maximum_pressure_change
        maximum_allowed_pressure_change = config.maximum_pressure_change

        if maximum_pressure_change > maximum_allowed_pressure_change:
            message = (
                f"Pressure change {maximum_pressure_change:.6f} psi exceeded maximum "
                f"allowed {maximum_allowed_pressure_change:.6f} psi at time step "
                f"{time_step}, outer iteration {iteration + 1}."
            )
            logger.error(message)
            return StepResult(
                fluid_properties=padded_fluid_properties,
                rock_properties=padded_rock_properties,
                saturation_history=padded_saturation_history,
                success=False,
                message=message,
                timer_kwargs={
                    "maximum_pressure_change": maximum_pressure_change,
                    "maximum_allowed_pressure_change": maximum_allowed_pressure_change,
                },
            )

        pressure_validation_result = _validate_pressure_range(
            padded_pressure_grid=padded_pressure_grid,
            time_step=time_step,
            padded_fluid_properties=padded_fluid_properties,
            padded_rock_properties=padded_rock_properties,
            padded_saturation_history=padded_saturation_history,
        )
        if pressure_validation_result is not None:
            return pressure_validation_result

        apply_pressure_boundary_condition(
            padded_pressure_grid=padded_pressure_grid,
            boundary_conditions=boundary_conditions,
            cell_dimension=cell_dimension,
            grid_shape=grid_shape,
            thickness_grid=thickness_grid,
            time=time,
            pad_width=pad_width,
        )
        padded_pressure_grid = np.clip(
            padded_pressure_grid, min_valid_pressure, max_valid_pressure
        ).astype(dtype, copy=False)

        # PVT update at the new pressure, then solution gas flash
        old_solution_gas_to_oil_ratio_grid = (
            iter_fluid_properties.solution_gas_to_oil_ratio_grid.copy()
        )
        old_oil_formation_volume_factor_grid = (
            iter_fluid_properties.oil_formation_volume_factor_grid.copy()
        )
        old_gas_solubility_in_water_grid = (
            iter_fluid_properties.gas_solubility_in_water_grid.copy()
        )
        old_water_formation_volume_factor_grid = (
            iter_fluid_properties.water_formation_volume_factor_grid.copy()
        )
        pre_flash_oil_saturation_grid = iter_fluid_properties.oil_saturation_grid.copy()
        pre_flash_gas_saturation_grid = iter_fluid_properties.gas_saturation_grid.copy()
        pre_flash_water_saturation_grid = (
            iter_fluid_properties.water_saturation_grid.copy()
        )

        iter_fluid_properties = attrs.evolve(
            iter_fluid_properties, pressure_grid=padded_pressure_grid
        )
        iter_fluid_properties = update_pvt_grids(
            fluid_properties=iter_fluid_properties,
            wells=wells,
            miscibility_model=miscibility_model,
            pvt_tables=config.pvt_tables,
            freeze_saturation_pressure=config.freeze_saturation_pressure,
        )
        iter_fluid_properties = apply_solution_gas_updates(
            fluid_properties=iter_fluid_properties,
            old_solution_gas_to_oil_ratio_grid=old_solution_gas_to_oil_ratio_grid,
            old_oil_formation_volume_factor_grid=old_oil_formation_volume_factor_grid,
            old_gas_solubility_in_water_grid=old_gas_solubility_in_water_grid,
            old_water_formation_volume_factor_grid=old_water_formation_volume_factor_grid,
        )

        flash_saturation_check = _check_saturation_changes(
            maximum_oil_saturation_change=float(
                np.max(
                    np.abs(
                        iter_fluid_properties.oil_saturation_grid
                        - pre_flash_oil_saturation_grid
                    )
                )
            ),
            maximum_water_saturation_change=float(
                np.max(
                    np.abs(
                        iter_fluid_properties.water_saturation_grid
                        - pre_flash_water_saturation_grid
                    )
                )
            ),
            maximum_gas_saturation_change=float(
                np.max(
                    np.abs(
                        iter_fluid_properties.gas_saturation_grid
                        - pre_flash_gas_saturation_grid
                    )
                )
            ),
            max_allowed_oil_saturation_change=config.maximum_oil_saturation_change,
            max_allowed_water_saturation_change=config.maximum_water_saturation_change,
            max_allowed_gas_saturation_change=config.maximum_gas_saturation_change,
        )
        if flash_saturation_check.violated:
            message = (
                f"Solution gas flash at time step {time_step}, outer iteration "
                f"{iteration + 1} violated saturation change limits: "
                f"{flash_saturation_check.message}"
            )
            logger.debug(message)
            return StepResult(
                fluid_properties=padded_fluid_properties,
                rock_properties=padded_rock_properties,
                saturation_history=padded_saturation_history,
                success=False,
                message=message,
                timer_kwargs={
                    "maximum_saturation_change": flash_saturation_check.max_phase_saturation_change,
                    "maximum_allowed_saturation_change": flash_saturation_check.max_allowed_phase_saturation_change,
                },
            )

        # Second PVT pass: re-evaluate properties that depend on the updated Rs
        iter_fluid_properties = update_pvt_grids(
            fluid_properties=iter_fluid_properties,
            wells=wells,
            miscibility_model=miscibility_model,
            pvt_tables=config.pvt_tables,
            freeze_saturation_pressure=config.freeze_saturation_pressure,
        )

        # pressure_change_grid is relative to the start-of-timestep pressure so that
        # the PVT volume correction in the saturation solver sees the full timestep delta,
        # not just the inter-iterate delta.
        pressure_change_grid = (
            padded_pressure_grid - padded_fluid_properties.pressure_grid
        )

        saturation_result = implicit.evolve_saturation(
            grid_shape=grid_shape,
            cell_dimension=cell_dimension,
            thickness_grid=padded_thickness_grid,
            elevation_grid=padded_elevation_grid,
            time_step_size=time_step_size,
            time=time,
            rock_properties=padded_rock_properties,
            fluid_properties=iter_fluid_properties,
            config=config,
            well_indices_cache=well_indices_cache,
            injection_rates=_rates_proxy(injection_rates),
            production_rates=_rates_proxy(production_rates),
            injection_bhps=_bhps_proxy(injection_bhps),
            production_bhps=_bhps_proxy(production_bhps),
            boundary_conditions=boundary_conditions,
            pressure_change_grid=pressure_change_grid,
            pad_width=pad_width,
        )

        if not saturation_result.success:
            logger.error(
                f"Implicit saturation solve failed at outer iteration "
                f"{iteration + 1}, time step {time_step}:\n"
                f"{saturation_result.message}"
            )
            return StepResult(
                fluid_properties=padded_fluid_properties,
                rock_properties=padded_rock_properties,
                saturation_history=padded_saturation_history,
                success=False,
                message=saturation_result.message,
            )

        saturation_solution = saturation_result.value
        saturation_change_result = _check_saturation_changes(
            maximum_oil_saturation_change=saturation_solution.maximum_oil_saturation_change,
            maximum_water_saturation_change=saturation_solution.maximum_water_saturation_change,
            maximum_gas_saturation_change=saturation_solution.maximum_gas_saturation_change,
            max_allowed_oil_saturation_change=config.maximum_oil_saturation_change,
            max_allowed_water_saturation_change=config.maximum_water_saturation_change,
            max_allowed_gas_saturation_change=config.maximum_gas_saturation_change,
        )
        if saturation_change_result.violated:
            message = (
                f"At time step {time_step}, outer iteration {iteration + 1}, "
                f"saturation change limits were violated:\n"
                f"{saturation_change_result.message}\n\n"
                f"Oil saturation change:   {saturation_solution.maximum_oil_saturation_change:.6f}\n"
                f"Water saturation change: {saturation_solution.maximum_water_saturation_change:.6f}\n"
                f"Gas saturation change:   {saturation_solution.maximum_gas_saturation_change:.6f}"
            )
            logger.debug(message)
            return StepResult(
                fluid_properties=padded_fluid_properties,
                rock_properties=padded_rock_properties,
                saturation_history=padded_saturation_history,
                success=False,
                message=message,
                timer_kwargs={
                    "maximum_saturation_change": saturation_change_result.max_phase_saturation_change,
                    "maximum_allowed_saturation_change": saturation_change_result.max_allowed_phase_saturation_change,
                },
            )

        padded_water_saturation_grid = saturation_solution.water_saturation_grid.astype(
            dtype, copy=False
        )
        padded_oil_saturation_grid = saturation_solution.oil_saturation_grid.astype(
            dtype, copy=False
        )
        padded_gas_saturation_grid = saturation_solution.gas_saturation_grid.astype(
            dtype, copy=False
        )

        apply_saturation_boundary_conditions(
            padded_water_saturation_grid=padded_water_saturation_grid,
            padded_oil_saturation_grid=padded_oil_saturation_grid,
            padded_gas_saturation_grid=padded_gas_saturation_grid,
            boundary_conditions=boundary_conditions,
            cell_dimension=cell_dimension,
            grid_shape=grid_shape,
            thickness_grid=thickness_grid,
            time=time,
            pad_width=pad_width,
        )

        iter_fluid_properties = attrs.evolve(
            iter_fluid_properties,
            water_saturation_grid=padded_water_saturation_grid,
            oil_saturation_grid=padded_oil_saturation_grid,
            gas_saturation_grid=padded_gas_saturation_grid,
        )

        if config.normalize_saturations:
            normalize_saturations(
                oil_saturation_grid=iter_fluid_properties.oil_saturation_grid,
                water_saturation_grid=iter_fluid_properties.water_saturation_grid,
                gas_saturation_grid=iter_fluid_properties.gas_saturation_grid,
                saturation_epsilon=saturation_epsilon,
            )

        # If Newton converged very easily, coupling is weak, stop iteration early.
        newton_iterations = saturation_solution.newton_iterations
        newton_utilization = newton_iterations / maximum_newton_iterations
        if newton_utilization < 0.25:  # used less than 25% of Newton budget
            logger.debug(
                f"Newton converged in {newton_iterations} iterations (utilization {newton_utilization:.0%}), "
                f"skipping further outer iterations."
            )
            outer_converged = True
            break

        # Exit early, if total saturation movement from the start of the timestep is small,
        # the pressure-saturation coupling error is negligible regardless of iteration count.
        total_saturation_change_from_bop = max(
            float(
                np.max(
                    np.abs(
                        iter_fluid_properties.water_saturation_grid
                        - padded_fluid_properties.water_saturation_grid
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        iter_fluid_properties.oil_saturation_grid
                        - padded_fluid_properties.oil_saturation_grid
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        iter_fluid_properties.gas_saturation_grid
                        - padded_fluid_properties.gas_saturation_grid
                    )
                )
            ),
        )
        if total_saturation_change_from_bop < 0.1 * min(
            config.maximum_oil_saturation_change,
            config.maximum_water_saturation_change,
            config.maximum_gas_saturation_change,
        ):
            logger.debug(
                f"Total saturation change from start of timestep {total_saturation_change_from_bop:.3e} is small, "
                f"skipping further outer iterations."
            )
            outer_converged = True
            break

        # Outer convergence check.
        # For saturation, we check absolute inter-iterate change against an absolute tolerance
        # derived from the per-phase change limits.
        # for pressure, we check relative inter-iterate change normalised by mean field pressure so
        # the criterion is regime-independent; guard against degenerate zero-pressure fields.
        max_outer_saturation_change = max(
            float(
                np.max(
                    np.abs(
                        iter_fluid_properties.water_saturation_grid
                        - prev_water_saturation_grid
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        iter_fluid_properties.oil_saturation_grid
                        - prev_oil_saturation_grid
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        iter_fluid_properties.gas_saturation_grid
                        - prev_gas_saturation_grid
                    )
                )
            ),
        )
        reference_pressure = max(float(np.mean(np.abs(padded_pressure_grid))), 1.0)
        relative_outer_pressure_change = (
            float(np.max(np.abs(padded_pressure_grid - prev_pressure_grid)))
            / reference_pressure
        )

        logger.debug(
            f"Outer iteration {iteration + 1} convergence - "
            f"Δsat (absolute): {max_outer_saturation_change:.3e} (atol={saturation_tolerance:.3e}), "
            f"ΔP (relative): {relative_outer_pressure_change:.3e} "
            f"(rtol={pressure_tolerance:.3e})"
        )

        final_timer_kwargs = {
            "maximum_pressure_change": maximum_pressure_change,
            "maximum_allowed_pressure_change": maximum_allowed_pressure_change,
            "maximum_saturation_change": saturation_change_result.max_phase_saturation_change
            or None,
            "maximum_allowed_saturation_change": saturation_change_result.max_allowed_phase_saturation_change
            or None,
            "newton_iterations": newton_iterations,
        }

        if (
            max_outer_saturation_change < saturation_tolerance
            and relative_outer_pressure_change < pressure_tolerance
        ):
            logger.debug(
                f"Outer iteration converged after {iteration + 1} iteration(s)."
            )
            outer_converged = True
            break

        # Rebuild rock-fluid properties at the updated saturation state before the
        # next outer iteration so the pressure solve sees consistent mobilities.
        iter_fluid_properties = update_pvt_grids(
            fluid_properties=iter_fluid_properties,
            wells=wells,
            miscibility_model=miscibility_model,
            pvt_tables=config.pvt_tables,
            freeze_saturation_pressure=config.freeze_saturation_pressure,
        )
        (
            _,
            iter_relative_mobility_grids,
            iter_capillary_pressure_grids,
        ) = build_rock_fluid_properties_grids(
            water_saturation_grid=iter_fluid_properties.water_saturation_grid,
            oil_saturation_grid=iter_fluid_properties.oil_saturation_grid,
            gas_saturation_grid=iter_fluid_properties.gas_saturation_grid,
            irreducible_water_saturation_grid=padded_rock_properties.irreducible_water_saturation_grid,
            residual_oil_saturation_water_grid=padded_rock_properties.residual_oil_saturation_water_grid,
            residual_oil_saturation_gas_grid=padded_rock_properties.residual_oil_saturation_gas_grid,
            residual_gas_saturation_grid=padded_rock_properties.residual_gas_saturation_grid,
            water_viscosity_grid=iter_fluid_properties.water_viscosity_grid,
            oil_viscosity_grid=iter_fluid_properties.oil_effective_viscosity_grid,
            gas_viscosity_grid=iter_fluid_properties.gas_viscosity_grid,
            relative_permeability_table=config.rock_fluid_tables.relative_permeability_table,
            capillary_pressure_table=config.rock_fluid_tables.capillary_pressure_table,
            disable_capillary_effects=config.disable_capillary_effects,
            capillary_strength_factor=config.capillary_strength_factor,
            relative_mobility_range=config.relative_mobility_range,
            phase_appearance_tolerance=config.phase_appearance_tolerance,
        )

        prev_pressure_grid = padded_pressure_grid.copy()
        prev_water_saturation_grid = iter_fluid_properties.water_saturation_grid.copy()
        prev_oil_saturation_grid = iter_fluid_properties.oil_saturation_grid.copy()
        prev_gas_saturation_grid = iter_fluid_properties.gas_saturation_grid.copy()

    if not outer_converged:
        logger.warning(
            f"Outer iteration did not converge after {config.maximum_outer_iterations} "
            f"iteration(s) at time step {time_step}. Proceeding with last solution."
        )

    assert saturation_result is not None and saturation_solution is not None, (
        "Saturation solve must have run at least once."
    )

    padded_rock_properties, padded_saturation_history = (
        update_residual_saturation_grids(
            rock_properties=padded_rock_properties,
            saturation_history=padded_saturation_history,
            water_saturation_grid=iter_fluid_properties.water_saturation_grid,
            gas_saturation_grid=iter_fluid_properties.gas_saturation_grid,
            residual_oil_drainage_ratio_water_flood=config.residual_oil_drainage_ratio_water_flood,
            residual_oil_drainage_ratio_gas_flood=config.residual_oil_drainage_ratio_gas_flood,
            residual_gas_drainage_ratio=config.residual_gas_drainage_ratio,
        )
    )

    logger.debug("Sequential implicit step completed.")
    return StepResult(
        fluid_properties=iter_fluid_properties,
        rock_properties=padded_rock_properties,
        saturation_history=padded_saturation_history,
        injection_rates=injection_rates,
        production_rates=production_rates,
        injection_bhps=injection_bhps,
        production_bhps=production_bhps,
        success=True,
        message=saturation_result.message,
        timer_kwargs=final_timer_kwargs,
    )


def _run_explicit_step(
    time_step: int,
    grid_shape: ThreeDimensions,
    cell_dimension: typing.Tuple[float, float],
    thickness_grid: NDimensionalGrid[ThreeDimensions],
    padded_thickness_grid: NDimensionalGrid[ThreeDimensions],
    padded_elevation_grid: NDimensionalGrid[ThreeDimensions],
    time_step_size: float,
    time: float,
    padded_rock_properties: RockProperties[ThreeDimensions],
    padded_fluid_properties: FluidProperties[ThreeDimensions],
    padded_saturation_history: SaturationHistory[ThreeDimensions],
    padded_relative_mobility_grids: RelativeMobilityGrids[ThreeDimensions],
    padded_capillary_pressure_grids: CapillaryPressureGrids[ThreeDimensions],
    wells: Wells[ThreeDimensions],
    miscibility_model: MiscibilityModel,
    config: Config,
    boundary_conditions: BoundaryConditions[ThreeDimensions],
    well_indices_cache: WellIndicesCache,
    dtype: npt.DTypeLike,
    pad_width: int = 1,
    min_valid_pressure: float = 14.7,
    max_valid_pressure: float = 14700,
    saturation_epsilon: float = 1e-12,
) -> StepResult[ThreeDimensions]:
    """
    Execute one time step using fully explicit scheme (explicit pressure and saturation).

    :param time_step: Current time step index.
    :param padded_zeros_grid: Padded grid of zeros for rate tracking.
    :param grid_shape: Original model grid shape (nx, ny, nz).
    :param cell_dimension: Tuple of cell dimensions (dx, dy).
    :param thickness_grid: Un-padded thickness grid.
    :param padded_thickness_grid: Padded thickness grid.
    :param padded_elevation_grid: Padded elevation grid.
    :param time_step_size: Size of the current time step.
    :param time: Total simulation time elapsed. This time step inclusive.
    :param padded_rock_properties: Padded rock properties.
    :param padded_fluid_properties: Padded fluid properties.
    :param padded_saturation_history: Padded saturation history.
    :param padded_relative_mobility_grids: Padded relative mobility grids.
    :param padded_capillary_pressure_grids: Padded capillary pressure grids.
    :param wells: Wells in the reservoir.
    :param miscibility_model: Miscibility model used in the simulation.
    :param config: Simulation configuration.
    :param boundary_conditions: Model boundary conditions.
    :param pad_width: Number of ghost cells used for grid padding.
    :return: `StepResult` containing updated rates and fluid properties.
    """
    old_pressure_grid = padded_fluid_properties.pressure_grid.copy()

    logger.debug("Evolving pressure (explicit)...")
    injection_rates = _make_rates(grid_shape)
    production_rates = _make_rates(grid_shape)
    injection_bhps = _make_bhps(grid_shape)
    production_bhps = _make_bhps(grid_shape)
    pressure_result = explicit.evolve_pressure(
        cell_dimension=cell_dimension,
        thickness_grid=padded_thickness_grid,
        elevation_grid=padded_elevation_grid,
        time_step=time_step,
        time_step_size=time_step_size,
        time=time,
        rock_properties=padded_rock_properties,
        fluid_properties=padded_fluid_properties,
        relative_mobility_grids=padded_relative_mobility_grids,
        capillary_pressure_grids=padded_capillary_pressure_grids,
        wells=wells,
        config=config,
        well_indices_cache=well_indices_cache,
        injection_rates=_rates_proxy(injection_rates),
        production_rates=_rates_proxy(production_rates),
        injection_bhps=_bhps_proxy(injection_bhps),
        production_bhps=_bhps_proxy(production_bhps),
        pad_width=pad_width,
    )
    pressure_solution = pressure_result.value
    maximum_pressure_change = pressure_solution.maximum_pressure_change
    maximum_allowed_pressure_change = config.maximum_pressure_change
    timer_kwargs = {
        "maximum_cfl_encountered": pressure_solution.maximum_cfl_encountered,
        "cfl_threshold": pressure_solution.cfl_threshold,
        "maximum_pressure_change": maximum_pressure_change,
        "maximum_allowed_pressure_change": maximum_allowed_pressure_change,
    }

    if not pressure_result.success:
        logger.error(
            f"Explicit pressure evolution failed at time step {time_step}: \n{pressure_result.message}"
        )
        return StepResult(
            fluid_properties=padded_fluid_properties,
            rock_properties=padded_rock_properties,
            saturation_history=padded_saturation_history,
            success=False,
            message=pressure_result.message,
            timer_kwargs=timer_kwargs,
        )

    if maximum_pressure_change > maximum_allowed_pressure_change:
        message = (
            f"Pressure change {maximum_pressure_change:.6f} psi "
            f"exceeded maximum allowed {maximum_allowed_pressure_change:.6f} psi "
            f"at time step {time_step}."
        )
        logger.debug(message)
        return StepResult(
            fluid_properties=padded_fluid_properties,
            rock_properties=padded_rock_properties,
            saturation_history=padded_saturation_history,
            success=False,
            message=message,
            timer_kwargs={
                "maximum_pressure_change": maximum_pressure_change,
                "maximum_allowed_pressure_change": maximum_allowed_pressure_change,
            },
        )

    padded_pressure_grid = pressure_solution.pressure_grid

    # Check for any out-of-range pressures
    pressure_validation_result = _validate_pressure_range(
        padded_pressure_grid=padded_pressure_grid,
        time_step=time_step,
        padded_fluid_properties=padded_fluid_properties,
        padded_rock_properties=padded_rock_properties,
        padded_saturation_history=padded_saturation_history,
    )
    if pressure_validation_result is not None:
        # Add `timer_kwargs` to the result for explicit scheme
        return StepResult(
            fluid_properties=pressure_validation_result.fluid_properties,
            rock_properties=pressure_validation_result.rock_properties,
            saturation_history=pressure_validation_result.saturation_history,
            success=False,
            message=pressure_validation_result.message,
            timer_kwargs=timer_kwargs,
        )

    # Apply boundary conditions to new pressure grid
    logger.debug(
        f"Applying pressure boundary condition after pressure evolution for time step {time_step}..."
    )
    apply_pressure_boundary_condition(
        padded_pressure_grid=padded_pressure_grid,
        boundary_conditions=boundary_conditions,
        cell_dimension=cell_dimension,
        grid_shape=grid_shape,
        thickness_grid=thickness_grid,
        time=time,
        pad_width=pad_width,
    )
    logger.debug("Pressure boundary condition applied.")

    # Clamp pressures to valid range just for additional safety and to remove numerical noise
    padded_pressure_grid = np.clip(
        padded_pressure_grid, min_valid_pressure, max_valid_pressure
    ).astype(dtype, copy=False)
    logger.debug("Pressure evolution completed!")

    # Explicit specific: Re-use current fluid properties for saturation evolution.
    # Unlike IMPES, explicit pressure and saturation are fully decoupled at the current timestep.
    # PVT is updated after both pressure and saturation evolve (below at line ~866).
    # This is acceptable because explicit schemes use old-time values for transport coefficients.
    logger.debug(
        "Using current PVT fluid properties for saturation evolution (explicit scheme)"
    )

    # Saturation evolution (explicit)
    logger.debug("Evolving saturation (explicit)...")

    # Compute pressure change grid for PVT volume correction
    pressure_change_grid = padded_pressure_grid - old_pressure_grid

    if miscibility_model == "immiscible":
        saturation_result = explicit.evolve_saturation(
            cell_dimension=cell_dimension,
            thickness_grid=padded_thickness_grid,
            elevation_grid=padded_elevation_grid,
            time_step=time_step,
            time_step_size=time_step_size,
            rock_properties=padded_rock_properties,
            fluid_properties=padded_fluid_properties,
            relative_mobility_grids=padded_relative_mobility_grids,
            capillary_pressure_grids=padded_capillary_pressure_grids,
            config=config,
            well_indices_cache=well_indices_cache,
            injection_rates=_rates_proxy(injection_rates),
            production_rates=_rates_proxy(production_rates),
            pressure_change_grid=pressure_change_grid,
            pad_width=pad_width,
        )
    else:
        saturation_result = explicit.evolve_miscible_saturation(
            cell_dimension=cell_dimension,
            thickness_grid=padded_thickness_grid,
            elevation_grid=padded_elevation_grid,
            time_step=time_step,
            time_step_size=time_step_size,
            time=time,
            rock_properties=padded_rock_properties,
            fluid_properties=padded_fluid_properties,
            relative_mobility_grids=padded_relative_mobility_grids,
            capillary_pressure_grids=padded_capillary_pressure_grids,
            wells=wells,
            config=config,
            well_indices_cache=well_indices_cache,
            injection_rates=_rates_proxy(injection_rates),
            production_rates=_rates_proxy(production_rates),
            pressure_change_grid=pressure_change_grid,
            pad_width=pad_width,
        )

    saturation_solution = saturation_result.value
    saturation_change_result = _check_saturation_changes(
        maximum_oil_saturation_change=saturation_solution.maximum_oil_saturation_change,
        maximum_water_saturation_change=saturation_solution.maximum_water_saturation_change,
        maximum_gas_saturation_change=saturation_solution.maximum_gas_saturation_change,
        max_allowed_oil_saturation_change=config.maximum_oil_saturation_change,
        max_allowed_water_saturation_change=config.maximum_water_saturation_change,
        max_allowed_gas_saturation_change=config.maximum_gas_saturation_change,
    )
    timer_kwargs = {
        "maximum_cfl_encountered": saturation_solution.maximum_cfl_encountered,
        "cfl_threshold": saturation_solution.cfl_threshold,
        "maximum_pressure_change": maximum_pressure_change,
        "maximum_allowed_pressure_change": maximum_allowed_pressure_change,
        "maximum_saturation_change": saturation_change_result.max_phase_saturation_change
        or None,
        "maximum_allowed_saturation_change": saturation_change_result.max_allowed_phase_saturation_change
        or None,
    }

    if not saturation_result.success:
        logger.error(
            f"Explicit saturation evolution failed at time step {time_step}: \n{saturation_result.message}"
        )
        return StepResult(
            fluid_properties=padded_fluid_properties,
            rock_properties=padded_rock_properties,
            saturation_history=padded_saturation_history,
            success=False,
            message=saturation_result.message,
            timer_kwargs=timer_kwargs,
        )

    if saturation_change_result.violated:
        message = f"""
        At time step {time_step}, saturation change limits were violated:
        {saturation_change_result.message} 

        Oil saturation change: {saturation_solution.maximum_oil_saturation_change:.6f}, 
        Water saturation change: {saturation_solution.maximum_water_saturation_change:.6f}, 
        Gas saturation change: {saturation_solution.maximum_gas_saturation_change:.6f}.
        """
        logger.error(message)
        return StepResult(
            fluid_properties=padded_fluid_properties,
            rock_properties=padded_rock_properties,
            saturation_history=padded_saturation_history,
            success=False,
            message=message,
            timer_kwargs=timer_kwargs,
        )

    logger.debug("Saturation evolution completed!")

    # Update fluid properties with new pressure and saturations after saturation update
    logger.debug("Updating fluid properties with new pressure and saturation grids...")

    padded_water_saturation_grid = saturation_solution.water_saturation_grid.astype(
        dtype, copy=False
    )
    padded_oil_saturation_grid = saturation_solution.oil_saturation_grid.astype(
        dtype, copy=False
    )
    padded_gas_saturation_grid = saturation_solution.gas_saturation_grid.astype(
        dtype, copy=False
    )
    padded_solvent_concentration_grid = saturation_solution.solvent_concentration_grid
    if padded_solvent_concentration_grid is None:
        padded_fluid_properties = attrs.evolve(
            padded_fluid_properties,
            pressure_grid=padded_pressure_grid,
            water_saturation_grid=padded_water_saturation_grid,
            oil_saturation_grid=padded_oil_saturation_grid,
            gas_saturation_grid=padded_gas_saturation_grid,
        )
    else:
        padded_fluid_properties = attrs.evolve(
            padded_fluid_properties,
            pressure_grid=padded_pressure_grid,
            water_saturation_grid=padded_water_saturation_grid,
            oil_saturation_grid=padded_oil_saturation_grid,
            gas_saturation_grid=padded_gas_saturation_grid,
            solvent_concentration_grid=padded_solvent_concentration_grid.astype(
                dtype, copy=False
            ),
        )

    if config.normalize_saturations:
        # Normalize saturations (in-place) to ensure So + Sw + Sg = 1.0
        normalize_saturations(
            oil_saturation_grid=padded_fluid_properties.oil_saturation_grid,
            water_saturation_grid=padded_fluid_properties.water_saturation_grid,
            gas_saturation_grid=padded_fluid_properties.gas_saturation_grid,
            saturation_epsilon=saturation_epsilon,
        )

    # Update PVT properties with new state (pressure and saturations)
    logger.debug("Updating PVT fluid properties after explicit solve...")

    # Save old solution gas-to-oil ratio and oil formation volume factor
    # before PVT update (needed for gas liberation flash)
    old_solution_gas_to_oil_ratio_grid = (
        padded_fluid_properties.solution_gas_to_oil_ratio_grid.copy()
    )
    old_oil_formation_volume_factor_grid = (
        padded_fluid_properties.oil_formation_volume_factor_grid.copy()
    )
    old_gas_solubility_in_water_grid = (
        padded_fluid_properties.gas_solubility_in_water_grid.copy()
    )
    old_water_formation_volume_factor_grid = (
        padded_fluid_properties.water_formation_volume_factor_grid.copy()
    )
    # Save pre-flash saturations to detect flash-induced saturation changes
    old_oil_saturation_grid = padded_fluid_properties.oil_saturation_grid.copy()
    old_gas_saturation_grid = padded_fluid_properties.gas_saturation_grid.copy()
    old_water_saturation_grid = padded_fluid_properties.water_saturation_grid.copy()

    padded_fluid_properties = update_pvt_grids(
        fluid_properties=padded_fluid_properties,
        wells=wells,
        miscibility_model=miscibility_model,
        pvt_tables=config.pvt_tables,
        freeze_saturation_pressure=config.freeze_saturation_pressure,
    )

    # Solution gas liberation / re-dissolution flash step.
    # When pressure drops below bubble point, solution gas-to-oil ratio decreases
    # and dissolved gas comes out of solution as free gas.
    # This updates oil saturation, gas saturation, and solution gas-to-oil ratio.
    padded_fluid_properties = apply_solution_gas_updates(
        fluid_properties=padded_fluid_properties,
        old_solution_gas_to_oil_ratio_grid=old_solution_gas_to_oil_ratio_grid,
        old_oil_formation_volume_factor_grid=old_oil_formation_volume_factor_grid,
        old_gas_solubility_in_water_grid=old_gas_solubility_in_water_grid,
        old_water_formation_volume_factor_grid=old_water_formation_volume_factor_grid,
    )

    # Check flash-induced saturation changes so the next timestep doesn't
    # inherit a large uncontrolled saturation jump from the liberation flash.
    flash_gas_saturation_change = float(
        np.max(
            np.abs(
                padded_fluid_properties.gas_saturation_grid - old_gas_saturation_grid
            )
        )
    )
    flash_oil_saturation_change = float(
        np.max(
            np.abs(
                padded_fluid_properties.oil_saturation_grid - old_oil_saturation_grid
            )
        )
    )
    flash_water_saturation_change = float(
        np.max(
            np.abs(
                padded_fluid_properties.water_saturation_grid
                - old_water_saturation_grid
            )
        )
    )
    flash_saturation_check = _check_saturation_changes(
        maximum_oil_saturation_change=flash_oil_saturation_change,
        maximum_water_saturation_change=flash_water_saturation_change,
        maximum_gas_saturation_change=flash_gas_saturation_change,
        max_allowed_oil_saturation_change=config.maximum_oil_saturation_change,
        max_allowed_water_saturation_change=config.maximum_water_saturation_change,
        max_allowed_gas_saturation_change=config.maximum_gas_saturation_change,
    )
    if flash_saturation_check.violated:
        message = (
            f"Solution gas liberation flash at time step {time_step} violated "
            f"saturation change limits: {flash_saturation_check.message}"
        )
        logger.debug(message)
        return StepResult(
            fluid_properties=padded_fluid_properties,
            rock_properties=padded_rock_properties,
            saturation_history=padded_saturation_history,
            success=False,
            message=message,
            timer_kwargs={
                "maximum_saturation_change": flash_saturation_check.max_phase_saturation_change,
                "maximum_allowed_saturation_change": flash_saturation_check.max_allowed_phase_saturation_change,
            },
        )

    # Updated fluid properties again as solution gas-to-oil ratio may have changed
    # and some PVT properties depend on it
    padded_fluid_properties = update_pvt_grids(
        fluid_properties=padded_fluid_properties,
        wells=wells,
        miscibility_model=miscibility_model,
        pvt_tables=config.pvt_tables,
        freeze_saturation_pressure=config.freeze_saturation_pressure,
    )

    # Apply boundary conditions to updated saturation grids
    logger.debug(
        f"Applying saturations boundary conditions after solution gas evolution for time step {time_step}..."
    )
    apply_saturation_boundary_conditions(
        padded_water_saturation_grid=padded_fluid_properties.water_saturation_grid,
        padded_oil_saturation_grid=padded_fluid_properties.oil_saturation_grid,
        padded_gas_saturation_grid=padded_fluid_properties.gas_saturation_grid,
        boundary_conditions=boundary_conditions,
        cell_dimension=cell_dimension,
        grid_shape=grid_shape,
        thickness_grid=thickness_grid,
        time=time,
        pad_width=pad_width,
    )
    logger.debug("Saturation boundary conditions applied.")

    # Update residual saturation grids based on new saturations
    padded_rock_properties, padded_saturation_history = (
        update_residual_saturation_grids(
            rock_properties=padded_rock_properties,
            saturation_history=padded_saturation_history,
            water_saturation_grid=padded_fluid_properties.water_saturation_grid,
            gas_saturation_grid=padded_fluid_properties.gas_saturation_grid,
            residual_oil_drainage_ratio_water_flood=config.residual_oil_drainage_ratio_water_flood,
            residual_oil_drainage_ratio_gas_flood=config.residual_oil_drainage_ratio_gas_flood,
            residual_gas_drainage_ratio=config.residual_gas_drainage_ratio,
        )
    )
    return StepResult(
        fluid_properties=padded_fluid_properties,
        rock_properties=padded_rock_properties,
        saturation_history=padded_saturation_history,
        injection_rates=injection_rates,
        production_rates=production_rates,
        injection_bhps=injection_bhps,
        production_bhps=production_bhps,
        success=True,
        message=saturation_result.message,
        timer_kwargs=timer_kwargs,
    )


def log_progress(
    step: int,
    step_size: float,
    time_elapsed: float,
    total_time: float,
    is_last_step: bool = False,
    interval: int = 3,
):
    """Logs the simulation progress at specified intervals."""
    if step <= 1 or step % interval == 0 or is_last_step:
        percent_complete = (time_elapsed / total_time) * 100.0
        logger.info(
            f"Time Step {step} with Δt = {step_size:.4f}s - "
            f"({percent_complete:.4f}%) - "
            f"Elapsed Time: {time_elapsed:.4f}s / {total_time:.4f}s"
        )


StepCallback = typing.Callable[[StepResult[ThreeDimensions], float, float], None]
"""
A callback function or handler that accepts:

- `StepResult[ThreeDimensions]`: The result of the current simulation step, containing updated fluid properties, rock properties, saturation history, rates, bhps, success status, message, and timer kwargs.
- `float`: The current step size (time step size).
- `float`: The total (successful) simulation time elapsed up to the current step.

This callback can be used for custom logging, monitoring, or handling of simulation results at each step. 
It is called after each simulation step with the results and timing information.
"""


@attrs.define
class Run(StoreSerializable):
    """
    Simulation run specification.

    Executes a reservoir simulation on a 3D static reservoir model using the provided configuration.

    Example:
    ```python
    from bores import ReservoirModel, Config, Run

    model = ReservoirModel.from_file("path/to/3d_model.h5")
    config = Config.from_file("path/to/simulation_config.yaml")

    run = Run(model=model, config=config)
    for state in run:
        # Process the model state at each output interval
        process(state)
    ```
    """

    model: ReservoirModel[ThreeDimensions]
    """The reservoir model to simulate."""

    config: Config
    """Simulation configuration and parameters."""

    name: typing.Optional[str] = None
    """Human-readable name for this run."""

    description: typing.Optional[str] = None
    """Detailed description of the simulation."""

    tags: typing.Tuple[str, ...] = attrs.field(factory=tuple)
    """Tags for organizing runs."""

    created_at: typing.Optional[str] = attrs.field(
        factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    """ISO timestamp of when this run was created."""

    def __call__(
        self,
        *,
        on_step_rejected: typing.Optional[StepCallback] = None,
        on_step_accepted: typing.Optional[StepCallback] = None,
    ) -> typing.Generator[ModelState[ThreeDimensions], None, None]:
        """Returns a genrator that executes this simulation run."""
        return run(
            self.model,
            self.config,
            on_step_rejected=on_step_rejected,
            on_step_accepted=on_step_accepted,
        )

    def __iter__(self) -> typing.Iterator[ModelState[ThreeDimensions]]:
        return iter(self())

    @classmethod
    def from_files(
        cls,
        model_path: typing.Union[str, PathLike],
        config_path: typing.Union[str, PathLike],
        pvt_tables_path: typing.Optional[typing.Union[str, PathLike]] = None,
        pvt_data_path: typing.Optional[typing.Union[str, PathLike]] = None,
    ) -> Self:
        """
        Load run from separate model and config files.

        :param model_path: Path to the reservoir model file.
        :param config_path: Path to the simulation configuration file.
        :param pvt_tables_path: Optional path to dumped `PVTTables` file.
        :param pvt_data_path: Optional path to dumped `PVTDataSet` file.
        :return: `Run` instance with loaded model and config.
        """
        model = ReservoirModel.from_file(model_path)
        if not isinstance(model, ReservoirModel) or model.dimensions != 3:
            raise ValidationError(
                "Loaded model must be a 3D `ReservoirModel` instance."
            )

        config = Config.from_file(config_path)
        if config is None:
            raise ValidationError("Failed to load simulation config from file.")

        if pvt_tables_path is not None:
            pvt_tables = PVTTables.from_file(pvt_tables_path)
            if pvt_tables is None:
                raise ValidationError("Failed to load `PVTTables` data from file.")

            config = config.with_updates(pvt_tables=pvt_tables)

        if pvt_data_path is not None:
            pvt_dataset = PVTDataSet.from_file(pvt_data_path)
            if pvt_dataset is None:
                raise ValidationError("Failed to load `PVTDataSet` from file.")

            pvt_tables = PVTTables.from_dataset(pvt_dataset)
            config = config.with_updates(pvt_tables=pvt_tables)
        return cls(model=model, config=config)


_SCHEME_ALIASES = {
    "si": "Sequential Implicit",
    "full-si": "Full Sequential Implicit",
    "impes": "IMPES",
    "explicit": "Explicit",
}


def run(
    input: typing.Union[ReservoirModel[ThreeDimensions], Run],
    config: typing.Optional[Config] = None,
    *,
    on_step_rejected: typing.Optional[StepCallback] = None,
    on_step_accepted: typing.Optional[StepCallback] = None,
) -> typing.Generator[ModelState[ThreeDimensions], None, None]:
    """
    Run a simulation on a 3D reservoir model.

    The 3D simulation evolves pressure and saturation over time using the specified evolution scheme.
    3D simulations are computationally intensive and may require significant memory and processing power.

    Complex reservoir features such as faults, fractures, boundary condtions can be modeled, but may increase computational demands.
    Ensure that the model and configuration are appropriate for 3D simulations.

    :param input: Either a `ReservoirModel` instance or a `Run` instance containing the model and configuration.
    :param config: Simulation run configuration and parameters. Only required if `input` is a `ReservoirModel`.
        If `input` is a `Run`, the configuration from the `Run` instance will be used. If config is provided
        alongside a `Run` instance, it will override the config in the `Run`.
    :param on_step_rejected: Optional callback function that is called when a simulation step is rejected due to convergence or stability issues.
        The callback receives the `StepResult`, current step size, and total elapsed time.
    :param on_step_accepted: Optional callback function that is called when a simulation step is accepted and successfully completed.
        The callback receives the `StepResult`, current step size, and total elapsed time.
    :yield: Yields the model state at specified output intervals.

    Example:
    ```python
    import bores

    # Using `ReservoirModel` and `Config` directly
    model = bores.ReservoirModel.from_file("path/to/3d_model.h5")
    config = bores.Config.from_file("path/to/simulation_config.yaml")
    for state in bores.run(model, config):
        # Process the model state at each output interval
        process(state)

    # Using `Run` instance
    run = bores.Run(model=model, config=config)
    for state in bores.run(run):
        process(state)

    # Using `Run` instance with overridden config
    new_config = bores.Config.from_file("path/to/new_simulation_config.yaml")
    for state in bores.run(run, config=new_config):
        process(state)

    ```
    """
    if isinstance(input, Run):
        model = input.model
        if config is not None:
            logger.info(
                "Overriding `config` parameter from `Run` instance with provided `config` parameter."
            )
        config = config or input.config
    else:
        if config is None:
            raise ValueError(
                "Must provide `config` parameter when `input` is a `ReservoirModel`"
            )
        model = input

    rock_fluid_tables = config.rock_fluid_tables
    boundary_conditions = config.boundary_conditions
    timer = config.timer
    wells = config.wells
    well_schedules = config.well_schedules

    if wells is None:
        logger.debug("No wells provided, proceeding with no-well simulation.")
        wells = Wells()

    if boundary_conditions is None:
        logger.debug("No boundary conditions provided, applying no-flow boundaries.")
        boundary_conditions = BoundaryConditions[ThreeDimensions]()

    cell_dimension = model.cell_dimension
    grid_shape = model.grid_shape
    has_wells = wells.exists()
    output_frequency = config.output_frequency
    scheme = config.scheme.replace("_", "-").lower()
    miscibility_model = config.miscibility_model
    dtype = get_dtype()
    disable_capillary_effects = config.disable_capillary_effects
    capillary_strength_factor = config.capillary_strength_factor
    relative_mobility_range = config.relative_mobility_range
    phase_appearance_tolerance = config.phase_appearance_tolerance
    pvt_tables = config.pvt_tables
    freeze_saturation_pressure = config.freeze_saturation_pressure
    log_interval = config.log_interval

    logger.info("Starting simulation workflow...")
    logger.info(f"Grid dimensions: {grid_shape}")
    logger.info(f"Cell dimensions: {cell_dimension}")
    logger.info(
        f"Evolution scheme: {_SCHEME_ALIASES.get(scheme, scheme.replace('-', ' ').title())}"
    )
    logger.info(f"Total simulation time: {timer.simulation_time} seconds")
    logger.info(f"Output frequency: every {output_frequency} steps")
    logger.info(f"Has wells: {has_wells}")
    if has_wells:
        logger.debug("Checking well locations against grid shape")
        wells.check_location(grid_shape=grid_shape)

    # Use the config context manager to ensure that constants defined in config are utilized
    # throughout the simulation run
    with config.constants():
        # Pad fluid and rock properties grids and other necesary grids with ghost cells
        # for boundary condition application
        # Ensure ghost cells mirror neighbour values by default
        pad_width = 1
        min_valid_pressure = c.MINIMUM_VALID_PRESSURE
        max_valid_pressure = c.MAXIMUM_VALID_PRESSURE
        saturation_epsilon = c.SATURATION_EPSILON
        padded_fluid_properties = model.fluid_properties.pad(pad_width=pad_width)
        padded_rock_properties = model.rock_properties.pad(pad_width=pad_width)
        padded_saturation_history = model.saturation_history.pad(pad_width=pad_width)
        thickness_grid = model.thickness_grid
        padded_thickness_grid = pad_grid(thickness_grid, pad_width=pad_width)
        well_indices_cache = build_well_indices_cache(
            wells=wells,
            thickness_grid=padded_thickness_grid,
            absolute_permeability=padded_rock_properties.absolute_permeability,
            boundary_conditions=boundary_conditions,
            cell_size_x=cell_dimension[0],
            cell_size_y=cell_dimension[1],
            cell_count_x=padded_thickness_grid.shape[0],
            cell_count_y=padded_thickness_grid.shape[1],
            cell_count_z=padded_thickness_grid.shape[2],
            pad_width=pad_width,
        )
        elevation_grid = model.get_elevation_grid(
            apply_dip=not config.disable_structural_dip
        )
        padded_elevation_grid = pad_grid(elevation_grid, pad_width=pad_width)

        # Apply boundary conditions to relevant padded grids
        logger.debug("Applying boundary conditions to initial grids")
        padded_fluid_properties = apply_boundary_conditions(
            padded_fluid_properties=padded_fluid_properties,
            boundary_conditions=boundary_conditions,
            cell_dimension=cell_dimension,
            grid_shape=grid_shape,
            thickness_grid=thickness_grid,
            time=0.0,
            pad_width=pad_width,
        )

        # Initialize fluid properties before starting the simulation
        # To ensure all dependent properties are consistent with initial pressure and saturation conditions
        logger.debug("Initializing PVT fluid properties for simulation start")
        padded_fluid_properties = update_pvt_grids(
            fluid_properties=padded_fluid_properties,
            wells=wells,
            miscibility_model=miscibility_model,
            pvt_tables=pvt_tables,
            freeze_saturation_pressure=freeze_saturation_pressure,
        )

        # Unpad the fluid properties back to the original grid shape for model state snapshots
        model = model.evolve(
            fluid_properties=padded_fluid_properties.unpad(pad_width=pad_width)
        )
        padded_water_saturation_grid = padded_fluid_properties.water_saturation_grid
        padded_oil_saturation_grid = padded_fluid_properties.oil_saturation_grid
        padded_gas_saturation_grid = padded_fluid_properties.gas_saturation_grid
        padded_irreducible_water_saturation_grid = (
            padded_rock_properties.irreducible_water_saturation_grid
        )
        padded_residual_oil_saturation_water_grid = (
            padded_rock_properties.residual_oil_saturation_water_grid
        )
        padded_residual_oil_saturation_gas_grid = (
            padded_rock_properties.residual_oil_saturation_gas_grid
        )
        padded_residual_gas_saturation_grid = (
            padded_rock_properties.residual_gas_saturation_grid
        )
        padded_water_viscosity_grid = padded_fluid_properties.water_viscosity_grid
        padded_oil_viscosity_grid = padded_fluid_properties.oil_effective_viscosity_grid
        padded_gas_viscosity_grid = padded_fluid_properties.gas_viscosity_grid
        relative_permeability_table = rock_fluid_tables.relative_permeability_table
        capillary_pressure_table = rock_fluid_tables.capillary_pressure_table
        (
            padded_relperm_grids,
            padded_relative_mobility_grids,
            padded_capillary_pressure_grids,
        ) = build_rock_fluid_properties_grids(
            water_saturation_grid=padded_water_saturation_grid,
            oil_saturation_grid=padded_oil_saturation_grid,
            gas_saturation_grid=padded_gas_saturation_grid,
            irreducible_water_saturation_grid=padded_irreducible_water_saturation_grid,
            residual_oil_saturation_water_grid=padded_residual_oil_saturation_water_grid,
            residual_oil_saturation_gas_grid=padded_residual_oil_saturation_gas_grid,
            residual_gas_saturation_grid=padded_residual_gas_saturation_grid,
            water_viscosity_grid=padded_water_viscosity_grid,
            oil_viscosity_grid=padded_oil_viscosity_grid,
            gas_viscosity_grid=padded_gas_viscosity_grid,
            relative_permeability_table=relative_permeability_table,
            capillary_pressure_table=capillary_pressure_table,
            disable_capillary_effects=disable_capillary_effects,
            capillary_strength_factor=capillary_strength_factor,
            relative_mobility_range=relative_mobility_range,
            phase_appearance_tolerance=phase_appearance_tolerance,
        )
        relative_mobility_grids = padded_relative_mobility_grids.unpad(
            pad_width=pad_width
        )
        relperm_grids = padded_relperm_grids.unpad(pad_width=pad_width)
        capillary_pressure_grids = padded_capillary_pressure_grids.unpad(
            pad_width=pad_width
        )
        injection_rates = _make_rates(grid_shape)
        production_rates = _make_rates(grid_shape)
        injection_bhps = _make_bhps(grid_shape)
        production_bhps = _make_bhps(grid_shape)
        state = ModelState(
            step=timer.step,
            step_size=timer.step_size,
            time=timer.elapsed_time,
            model=model,
            wells=wells,
            relative_mobilities=relative_mobility_grids,
            relative_permeabilities=relperm_grids,
            capillary_pressures=capillary_pressure_grids,
            injection_rates=injection_rates,
            production_rates=production_rates,
            injection_bhps=injection_bhps,
            production_bhps=production_bhps,
            timer_state=timer.dump_state(),
        )

        # Yield the initial model state
        logger.debug("Yielding initial model state")
        yield state

        no_flow_pressure_bc = isinstance(
            boundary_conditions["pressure"], type(default_bc)
        )
        while not timer.done():
            # We first propose the time step size for the next step
            # `timer.step` is still the last accepted step
            # since we have not accepted the new step size proposal and hence, the step yet.
            # So we use `timer.next_step` to indicate the new step we are attempting
            new_step = timer.next_step
            step_size = timer.propose_step_size()
            time = timer.elapsed_time + step_size
            logger.debug(
                f"Attempting time step {new_step} with size {step_size} seconds..."
            )
            try:
                if has_wells and well_schedules is not None:
                    logger.debug(
                        f"Updating wells configuration for time step {new_step}"
                    )
                    well_schedules.apply(wells, state)
                    logger.debug("Wells updated.")

                if new_step > 1:
                    # Apply boundary conditions before update for the new time step
                    logger.debug(
                        f"Applying boundary conditions for time step {new_step}..."
                    )
                    padded_fluid_properties = apply_boundary_conditions(
                        padded_fluid_properties=padded_fluid_properties,
                        boundary_conditions=boundary_conditions,
                        cell_dimension=cell_dimension,
                        grid_shape=grid_shape,
                        thickness_grid=thickness_grid,
                        time=time,
                        pad_width=pad_width,
                    )
                    logger.debug("Boundary conditions applied.")
                    # If the pressure boundary condition is not no-flow, Then apply PVT update before next (pressure) evolution
                    # since most PVT properties depend on pressure. This is skipped for no-flow BCs to save computation.
                    # because mirroring neighbour values for PVT properties is sufficient for no-flow BCs.
                    if no_flow_pressure_bc is False:
                        logger.debug(
                            "Updating PVT fluid properties due to boundary condition changes..."
                        )
                        padded_fluid_properties = update_pvt_grids(
                            fluid_properties=padded_fluid_properties,
                            wells=wells,
                            miscibility_model=miscibility_model,
                            pvt_tables=pvt_tables,
                            freeze_saturation_pressure=freeze_saturation_pressure,
                        )
                        logger.debug("PVT fluid properties updated")

                    # Build relative permeability, relative mobility, and capillary pressure grids
                    logger.debug(
                        f"Rebuilding rock-fluid property grids for time step {new_step}..."
                    )

                    (
                        padded_relperm_grids,
                        padded_relative_mobility_grids,
                        padded_capillary_pressure_grids,
                    ) = build_rock_fluid_properties_grids(
                        water_saturation_grid=padded_fluid_properties.water_saturation_grid,
                        oil_saturation_grid=padded_fluid_properties.oil_saturation_grid,
                        gas_saturation_grid=padded_fluid_properties.gas_saturation_grid,
                        irreducible_water_saturation_grid=padded_rock_properties.irreducible_water_saturation_grid,
                        residual_oil_saturation_water_grid=padded_rock_properties.residual_oil_saturation_water_grid,
                        residual_oil_saturation_gas_grid=padded_rock_properties.residual_oil_saturation_gas_grid,
                        residual_gas_saturation_grid=padded_rock_properties.residual_gas_saturation_grid,
                        water_viscosity_grid=padded_fluid_properties.water_viscosity_grid,
                        oil_viscosity_grid=padded_fluid_properties.oil_viscosity_grid,
                        gas_viscosity_grid=padded_fluid_properties.gas_viscosity_grid,
                        relative_permeability_table=relative_permeability_table,
                        capillary_pressure_table=capillary_pressure_table,
                        disable_capillary_effects=disable_capillary_effects,
                        capillary_strength_factor=capillary_strength_factor,
                        relative_mobility_range=relative_mobility_range,
                        phase_appearance_tolerance=phase_appearance_tolerance,
                    )

                if scheme == "impes":
                    result = _run_impes_step(
                        time_step=new_step,
                        grid_shape=grid_shape,
                        cell_dimension=cell_dimension,
                        thickness_grid=thickness_grid,
                        padded_thickness_grid=padded_thickness_grid,
                        padded_elevation_grid=padded_elevation_grid,
                        time_step_size=step_size,
                        time=time,
                        padded_rock_properties=padded_rock_properties,
                        padded_fluid_properties=padded_fluid_properties,
                        padded_saturation_history=padded_saturation_history,
                        padded_relative_mobility_grids=padded_relative_mobility_grids,
                        padded_capillary_pressure_grids=padded_capillary_pressure_grids,
                        wells=wells,
                        miscibility_model=miscibility_model,
                        config=config,
                        boundary_conditions=boundary_conditions,
                        well_indices_cache=well_indices_cache,
                        dtype=dtype,
                        pad_width=pad_width,
                        min_valid_pressure=min_valid_pressure,
                        max_valid_pressure=max_valid_pressure,
                        saturation_epsilon=saturation_epsilon,
                    )
                elif scheme in {"sequential-implicit", "si"}:
                    result = _run_sequential_implicit_step(
                        time_step=new_step,
                        grid_shape=grid_shape,
                        cell_dimension=cell_dimension,
                        thickness_grid=thickness_grid,
                        padded_thickness_grid=padded_thickness_grid,
                        padded_elevation_grid=padded_elevation_grid,
                        time_step_size=step_size,
                        time=time,
                        padded_rock_properties=padded_rock_properties,
                        padded_fluid_properties=padded_fluid_properties,
                        padded_saturation_history=padded_saturation_history,
                        padded_relative_mobility_grids=padded_relative_mobility_grids,
                        padded_capillary_pressure_grids=padded_capillary_pressure_grids,
                        wells=wells,
                        miscibility_model=miscibility_model,
                        config=config,
                        boundary_conditions=boundary_conditions,
                        well_indices_cache=well_indices_cache,
                        dtype=dtype,
                        pad_width=pad_width,
                        min_valid_pressure=min_valid_pressure,
                        max_valid_pressure=max_valid_pressure,
                        saturation_epsilon=saturation_epsilon,
                    )
                elif scheme in {"full-sequential-implicit", "full-si"}:
                    result = _run_full_sequential_implicit_step(
                        time_step=new_step,
                        grid_shape=grid_shape,
                        cell_dimension=cell_dimension,
                        thickness_grid=thickness_grid,
                        padded_thickness_grid=padded_thickness_grid,
                        padded_elevation_grid=padded_elevation_grid,
                        time_step_size=step_size,
                        time=time,
                        padded_rock_properties=padded_rock_properties,
                        padded_fluid_properties=padded_fluid_properties,
                        padded_saturation_history=padded_saturation_history,
                        padded_relative_mobility_grids=padded_relative_mobility_grids,
                        padded_capillary_pressure_grids=padded_capillary_pressure_grids,
                        wells=wells,
                        miscibility_model=miscibility_model,
                        config=config,
                        boundary_conditions=boundary_conditions,
                        well_indices_cache=well_indices_cache,
                        dtype=dtype,
                        pad_width=pad_width,
                        min_valid_pressure=min_valid_pressure,
                        max_valid_pressure=max_valid_pressure,
                        saturation_epsilon=saturation_epsilon,
                    )
                elif scheme == "explicit":
                    result = _run_explicit_step(
                        time_step=new_step,
                        grid_shape=grid_shape,
                        cell_dimension=cell_dimension,
                        thickness_grid=thickness_grid,
                        padded_thickness_grid=padded_thickness_grid,
                        padded_elevation_grid=padded_elevation_grid,
                        time_step_size=step_size,
                        time=time,
                        padded_rock_properties=padded_rock_properties,
                        padded_fluid_properties=padded_fluid_properties,
                        padded_saturation_history=padded_saturation_history,
                        padded_relative_mobility_grids=padded_relative_mobility_grids,
                        padded_capillary_pressure_grids=padded_capillary_pressure_grids,
                        wells=wells,
                        miscibility_model=miscibility_model,
                        config=config,
                        boundary_conditions=boundary_conditions,
                        well_indices_cache=well_indices_cache,
                        dtype=dtype,
                        pad_width=pad_width,
                        min_valid_pressure=min_valid_pressure,
                        max_valid_pressure=max_valid_pressure,
                        saturation_epsilon=saturation_epsilon,
                    )
                else:
                    raise ValidationError(
                        f"Invalid simualtion scheme {scheme!r}. Available schemes include 'impes', 'sequential-implicit', 'full-sequential-implicit', or 'explicit'."
                    )

                # If the step was successful, accept that step proposal
                if result.success:
                    # Now we can accept the proposed time step size and we now agree that this is a new step
                    logger.debug(f"Time step {new_step} completed successfully.")
                    timer.accept_step(step_size=step_size, **result.timer_kwargs)
                    if log_interval:
                        log_progress(
                            step=timer.step,
                            step_size=step_size,
                            time_elapsed=timer.elapsed_time,
                            total_time=timer.simulation_time,
                            is_last_step=timer.is_last_step,
                            interval=log_interval,
                        )
                    if on_step_accepted is not None:
                        on_step_accepted(result, step_size, timer.elapsed_time)
                else:
                    # Reject the step, adjust the time step size, and retry
                    logger.debug(
                        f"Time step {new_step} failed with step size {step_size}. Retrying with smaller step size."
                    )
                    try:
                        timer.reject_step(
                            step_size=step_size,
                            aggressive=timer.rejection_count > 5,
                            **result.timer_kwargs,
                        )
                    except TimingError as exc:
                        raise SimulationError(
                            f"Simulation failed at time step {new_step} and cannot reduce step size further. {exc}."
                            f"\n{result.message}"
                        ) from exc

                    if on_step_rejected is not None:
                        on_step_rejected(result, step_size, timer.elapsed_time)

                    continue  # Retry the time step with a smaller size

                # Get the updated fluid properties, which will also be used for the next time step
                padded_fluid_properties = result.fluid_properties
                padded_rock_properties = result.rock_properties
                padded_saturation_history = result.saturation_history

                # Take a snapshot of the model state at start, at specified intervals and at the last time step
                if (
                    timer.step == 1
                    or (timer.step % output_frequency == 0)
                    or timer.is_last_step
                ):
                    logger.debug(f"Capturing model state at time step {timer.step}")
                    # The production rates are negative in the evolution
                    # so we need to negate them to report positive production values
                    logger.debug(
                        "Preparing injection and production rate grids for output"
                    )
                    injection_rates = result.injection_rates
                    production_rates = result.production_rates
                    injection_bhps = result.injection_bhps
                    production_bhps = result.production_bhps
                    assert injection_rates is not None
                    assert production_rates is not None
                    assert injection_bhps is not None
                    assert production_bhps is not None

                    logger.debug("Taking model state snapshot")
                    # Capture the current state of the wells
                    wells_snapshot = copy.deepcopy(wells)
                    # Capture the current model with updated fluid properties
                    model_snapshot = model.evolve(
                        fluid_properties=padded_fluid_properties.unpad(
                            pad_width=pad_width
                        ),
                        rock_properties=padded_rock_properties.unpad(
                            pad_width=pad_width
                        ),
                        saturation_history=padded_saturation_history.unpad(
                            pad_width=pad_width
                        ),
                    )
                    relative_mobility_grids = padded_relative_mobility_grids.unpad(
                        pad_width=pad_width
                    )
                    relperm_grids = padded_relperm_grids.unpad(pad_width=pad_width)
                    capillary_pressure_grids = padded_capillary_pressure_grids.unpad(
                        pad_width=pad_width
                    )
                    state = ModelState(
                        step=timer.step,
                        step_size=timer.step_size,
                        time=timer.elapsed_time,
                        model=model_snapshot,
                        wells=wells_snapshot,
                        relative_mobilities=relative_mobility_grids,
                        relative_permeabilities=relperm_grids,
                        capillary_pressures=capillary_pressure_grids,
                        injection_rates=injection_rates,
                        production_rates=production_rates.abs(),
                        injection_bhps=injection_bhps,
                        production_bhps=production_bhps,
                        timer_state=timer.dump_state(),
                    )
                    logger.debug("Yielding model state")
                    yield state

            except StopSimulation as exc:
                logger.info(f"Stopping simulation on request: {exc}")
                break
            except Exception as exc:
                raise SimulationError(
                    f"Simulation failed while attempting time step {new_step} due to error: {exc}"
                ) from exc

    logger.info(f"Simulation completed successfully after {timer.step} time steps")
