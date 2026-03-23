"""Run a simulation workflow on a multi-dimensional reservoir model."""

import copy
import logging
import typing
import warnings
from datetime import datetime, timezone
from os import PathLike

import attrs
import numpy as np
from typing_extensions import Self

from bores._precision import get_dtype
from bores.boundary_conditions import (
    BoundaryConditions,
    GridBoundaryCondition,
    default_bc,
)
from bores.config import Config
from bores.constants import c
from bores.errors import SimulationError, StopSimulation, TimingError, ValidationError
from bores.grids.base import (
    CapillaryPressureGrids,
    RateGrids,
    RelativeMobilityGrids,
    _RateGridsProxy,
    build_uniform_grid,
)
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
    apply_solution_gas_liberation,
    update_pvt_grids,
    update_residual_saturation_grids,
)
from bores.grids.utils import pad_grid, unpad_grid
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
from bores.utils import clip
from bores.wells import Wells

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
    oil_injection_grid: NDimensionalGrid[NDimension]
    """Grid of oil injection rates during the time step."""
    water_injection_grid: NDimensionalGrid[NDimension]
    """Grid of water injection rates during the time step."""
    gas_injection_grid: NDimensionalGrid[NDimension]
    """Grid of gas injection rates during the time step."""
    oil_production_grid: NDimensionalGrid[NDimension]
    """Grid of oil production rates during the time step."""
    water_production_grid: NDimensionalGrid[NDimension]
    """Grid of water production rates during the time step."""
    gas_production_grid: NDimensionalGrid[NDimension]
    """Grid of gas production rates during the time step."""
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
    padded_zeros_grid: NDimensionalGrid[ThreeDimensions],
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
            oil_injection_grid=padded_zeros_grid,
            water_injection_grid=padded_zeros_grid,
            gas_injection_grid=padded_zeros_grid,
            oil_production_grid=padded_zeros_grid,
            water_production_grid=padded_zeros_grid,
            gas_production_grid=padded_zeros_grid,
            fluid_properties=padded_fluid_properties,
            rock_properties=padded_rock_properties,
            saturation_history=padded_saturation_history,
            success=False,
            message=message,
        )

    return None


def _check_saturation_changes(
    max_oil_saturation_change: float,
    max_water_saturation_change: float,
    max_gas_saturation_change: float,
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

    :param max_oil_saturation_change: Maximum oil saturation change observed.
    :param max_water_saturation_change: Maximum water saturation change observed.
    :param max_gas_saturation_change: Maximum gas saturation change observed.
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
    if max_oil_saturation_change > max_tolerable_oil_saturation_change:
        violated = True
        # Start assuming oil saturation change is the largest so far
        max_phase_saturation_change = max_oil_saturation_change
        max_allowed_phase_saturation_change = max_allowed_oil_saturation_change
        messages.append(
            f"Oil saturation change {max_oil_saturation_change:.6f} exceeded maximum allowed {max_allowed_oil_saturation_change:.6f}."
        )

    water_tolerance = max(tolerance, 0.005 * max_allowed_water_saturation_change)
    max_tolerable_water_saturation_change = max_allowed_water_saturation_change * (
        1 + water_tolerance
    )
    if max_water_saturation_change > max_tolerable_water_saturation_change:
        violated = True
        # If water saturation change is the largest so far, update the max values
        if max_water_saturation_change > max_phase_saturation_change:
            max_phase_saturation_change = max_water_saturation_change
            max_allowed_phase_saturation_change = max_allowed_water_saturation_change
        messages.append(
            f"Water saturation change {max_water_saturation_change:.6f} exceeded maximum allowed {max_allowed_water_saturation_change:.6f}."
        )

    gas_tolerance = max(tolerance, 0.005 * max_allowed_gas_saturation_change)
    max_tolerable_gas_saturation_change = max_allowed_gas_saturation_change * (
        1 + gas_tolerance
    )
    if max_gas_saturation_change > max_tolerable_gas_saturation_change:
        violated = True
        # If gas saturation change is the largest so far, update the max values
        if max_gas_saturation_change > max_phase_saturation_change:
            max_phase_saturation_change = max_gas_saturation_change
            max_allowed_phase_saturation_change = max_allowed_gas_saturation_change
        messages.append(
            f"Gas saturation change {max_gas_saturation_change:.6f} exceeded maximum allowed {max_allowed_gas_saturation_change:.6f}."
        )

    message = "\n".join(messages) if messages else None
    return SaturationChangeCheckResult(
        violated=violated,
        max_phase_saturation_change=max_phase_saturation_change,
        max_allowed_phase_saturation_change=max_allowed_phase_saturation_change,
        message=message,
    )


def _run_impes_step(
    time_step: int,
    padded_zeros_grid: NDimensionalGrid[ThreeDimensions],
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
    pad_width: int = 1,
) -> StepResult[ThreeDimensions]:
    """
    Execute one time step using IMPES (Implicit Pressure, Explicit Saturation).

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
        pad_width=pad_width,
        boundary_conditions=boundary_conditions,
    )
    if not pressure_result.success:
        logger.error(
            f"Implicit pressure evolution failed at time step {time_step}: \n{pressure_result.message}"
        )
        return StepResult(
            oil_injection_grid=padded_zeros_grid,
            water_injection_grid=padded_zeros_grid,
            gas_injection_grid=padded_zeros_grid,
            oil_production_grid=padded_zeros_grid,
            water_production_grid=padded_zeros_grid,
            gas_production_grid=padded_zeros_grid,
            fluid_properties=padded_fluid_properties,
            rock_properties=padded_rock_properties,
            saturation_history=padded_saturation_history,
            success=False,
            message=pressure_result.message,
        )

    pressure_solution = pressure_result.value
    padded_pressure_grid = pressure_solution.pressure_grid
    max_pressure_change = pressure_solution.max_pressure_change
    max_allowed_pressure_change = config.max_pressure_change
    if max_pressure_change > max_allowed_pressure_change:
        message = (
            f"Pressure change {max_pressure_change:.6f} psi "
            f"exceeded maximum allowed {max_allowed_pressure_change:.6f} psi "
            f"at time step {time_step}."
        )
        logger.error(message)
        return StepResult(
            oil_injection_grid=padded_zeros_grid,
            water_injection_grid=padded_zeros_grid,
            gas_injection_grid=padded_zeros_grid,
            oil_production_grid=padded_zeros_grid,
            water_production_grid=padded_zeros_grid,
            gas_production_grid=padded_zeros_grid,
            fluid_properties=padded_fluid_properties,
            rock_properties=padded_rock_properties,
            saturation_history=padded_saturation_history,
            success=False,
            message=message,
            timer_kwargs={
                "max_pressure_change": max_pressure_change,
                "max_allowed_pressure_change": max_allowed_pressure_change,
            },
        )

    # Check for any out-of-range pressures
    pressure_validation_result = _validate_pressure_range(
        padded_pressure_grid=padded_pressure_grid,
        time_step=time_step,
        padded_zeros_grid=padded_zeros_grid,
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

    dtype = get_dtype()
    # Clamp pressures to valid range just for additional safety and to remove numerical noise
    padded_pressure_grid = clip(
        padded_pressure_grid, c.MINIMUM_VALID_PRESSURE, c.MAXIMUM_VALID_PRESSURE
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
    # Save pre-flash saturations to detect flash-induced saturation changes
    old_oil_saturation_grid = padded_fluid_properties.oil_saturation_grid.copy()
    old_gas_saturation_grid = padded_fluid_properties.gas_saturation_grid.copy()

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
    padded_fluid_properties = apply_solution_gas_liberation(
        fluid_properties=padded_fluid_properties,
        old_solution_gas_to_oil_ratio_grid=old_solution_gas_to_oil_ratio_grid,
        old_oil_formation_volume_factor_grid=old_oil_formation_volume_factor_grid,
    )

    # Check flash-induced saturation changes before proceeding to saturation solver.
    # If the liberation flash itself violates the saturation change limits, reject
    # the step immediately, as there's no point running the full saturation solver.
    flash_gas_sat_change = float(
        np.max(
            np.abs(
                padded_fluid_properties.gas_saturation_grid - old_gas_saturation_grid
            )
        )
    )
    flash_oil_sat_change = float(
        np.max(
            np.abs(
                padded_fluid_properties.oil_saturation_grid - old_oil_saturation_grid
            )
        )
    )
    flash_sat_check = _check_saturation_changes(
        max_oil_saturation_change=flash_oil_sat_change,
        max_water_saturation_change=0.0,
        max_gas_saturation_change=flash_gas_sat_change,
        max_allowed_oil_saturation_change=config.max_oil_saturation_change,
        max_allowed_water_saturation_change=config.max_water_saturation_change,
        max_allowed_gas_saturation_change=config.max_gas_saturation_change,
    )
    if flash_sat_check.violated:
        message = (
            f"Solution gas liberation flash at time step {time_step} violated "
            f"saturation change limits: {flash_sat_check.message}"
        )
        logger.debug(message)
        return StepResult(
            oil_injection_grid=padded_zeros_grid,
            water_injection_grid=padded_zeros_grid,
            gas_injection_grid=padded_zeros_grid,
            oil_production_grid=padded_zeros_grid,
            water_production_grid=padded_zeros_grid,
            gas_production_grid=padded_zeros_grid,
            fluid_properties=padded_fluid_properties,
            rock_properties=padded_rock_properties,
            saturation_history=padded_saturation_history,
            success=False,
            message=message,
            timer_kwargs={
                "max_saturation_change": flash_sat_check.max_phase_saturation_change,
                "max_allowed_saturation_change": flash_sat_check.max_allowed_phase_saturation_change,
            },
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
    # Build zeros grids to track production and injection at each time step
    oil_injection_grid = padded_zeros_grid.copy()
    water_injection_grid = padded_zeros_grid.copy()
    gas_injection_grid = padded_zeros_grid.copy()
    oil_production_grid = padded_zeros_grid.copy()
    water_production_grid = padded_zeros_grid.copy()
    gas_production_grid = padded_zeros_grid.copy()

    # Compute pressure change grid for PVT volume correction
    pressure_change_grid = padded_pressure_grid - old_pressure_grid

    if miscibility_model == "immiscible":
        evolve_saturation = explicit.evolve_saturation
    else:
        evolve_saturation = explicit.evolve_miscible_saturation

    saturation_result = evolve_saturation(
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
        boundary_conditions=boundary_conditions,
        injection_grid=_RateGridsProxy(
            oil=oil_injection_grid,
            water=water_injection_grid,
            gas=gas_injection_grid,
        ),
        production_grid=_RateGridsProxy(
            oil=oil_production_grid,
            water=water_production_grid,
            gas=gas_production_grid,
        ),
        pressure_change_grid=pressure_change_grid,
        pad_width=pad_width,
    )
    saturation_solution = saturation_result.value
    saturation_change_result = _check_saturation_changes(
        max_oil_saturation_change=saturation_solution.max_oil_saturation_change,
        max_water_saturation_change=saturation_solution.max_water_saturation_change,
        max_gas_saturation_change=saturation_solution.max_gas_saturation_change,
        max_allowed_oil_saturation_change=config.max_oil_saturation_change,
        max_allowed_water_saturation_change=config.max_water_saturation_change,
        max_allowed_gas_saturation_change=config.max_gas_saturation_change,
    )
    timer_kwargs = {
        "max_cfl_encountered": saturation_solution.max_cfl_encountered,
        "cfl_threshold": saturation_solution.cfl_threshold,
        "max_pressure_change": max_pressure_change,
        "max_allowed_pressure_change": max_allowed_pressure_change,
        "max_saturation_change": saturation_change_result.max_phase_saturation_change
        or None,
        "max_allowed_saturation_change": saturation_change_result.max_allowed_phase_saturation_change
        or None,
    }

    if not saturation_result.success:
        logger.error(
            f"Explicit saturation evolution failed at time step {time_step}: \n{saturation_result.message}"
        )
        return StepResult(
            oil_injection_grid=oil_injection_grid,
            water_injection_grid=water_injection_grid,
            gas_injection_grid=gas_injection_grid,
            oil_production_grid=oil_production_grid,
            water_production_grid=water_production_grid,
            gas_production_grid=gas_production_grid,
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

        Oil saturation change: {saturation_solution.max_oil_saturation_change:.6f},
        Water saturation change: {saturation_solution.max_water_saturation_change:.6f},
        Gas saturation change: {saturation_solution.max_gas_saturation_change:.6f}.
        """
        logger.debug(message)
        return StepResult(
            oil_injection_grid=oil_injection_grid,
            water_injection_grid=water_injection_grid,
            gas_injection_grid=gas_injection_grid,
            oil_production_grid=oil_production_grid,
            water_production_grid=water_production_grid,
            gas_production_grid=gas_production_grid,
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
            saturation_epsilon=c.SATURATION_EPSILON,
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
        oil_injection_grid=oil_injection_grid,
        water_injection_grid=water_injection_grid,
        gas_injection_grid=gas_injection_grid,
        oil_production_grid=oil_production_grid,
        water_production_grid=water_production_grid,
        gas_production_grid=gas_production_grid,
        fluid_properties=padded_fluid_properties,
        rock_properties=padded_rock_properties,
        saturation_history=padded_saturation_history,
        success=True,
        message=saturation_result.message,
        timer_kwargs=timer_kwargs,
    )


def _run_sequential_implicit_step(
    time_step: int,
    padded_zeros_grid: NDimensionalGrid[ThreeDimensions],
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
    pad_width: int = 1,
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
        boundary_conditions=boundary_conditions,
        pad_width=pad_width,
    )
    if not pressure_result.success:
        logger.error(
            f"Implicit pressure evolution failed at time step {time_step}: \n{pressure_result.message}"
        )
        return StepResult(
            oil_injection_grid=padded_zeros_grid,
            water_injection_grid=padded_zeros_grid,
            gas_injection_grid=padded_zeros_grid,
            oil_production_grid=padded_zeros_grid,
            water_production_grid=padded_zeros_grid,
            gas_production_grid=padded_zeros_grid,
            fluid_properties=padded_fluid_properties,
            rock_properties=padded_rock_properties,
            saturation_history=padded_saturation_history,
            success=False,
            message=pressure_result.message,
        )

    pressure_solution = pressure_result.value
    padded_pressure_grid = pressure_solution.pressure_grid
    max_pressure_change = pressure_solution.max_pressure_change
    max_allowed_pressure_change = config.max_pressure_change
    if max_pressure_change > max_allowed_pressure_change:
        message = (
            f"Pressure change {max_pressure_change:.6f} psi "
            f"exceeded maximum allowed {max_allowed_pressure_change:.6f} psi "
            f"at time step {time_step}."
        )
        logger.error(message)
        return StepResult(
            oil_injection_grid=padded_zeros_grid,
            water_injection_grid=padded_zeros_grid,
            gas_injection_grid=padded_zeros_grid,
            oil_production_grid=padded_zeros_grid,
            water_production_grid=padded_zeros_grid,
            gas_production_grid=padded_zeros_grid,
            fluid_properties=padded_fluid_properties,
            rock_properties=padded_rock_properties,
            saturation_history=padded_saturation_history,
            success=False,
            message=message,
            timer_kwargs={
                "max_pressure_change": max_pressure_change,
                "max_allowed_pressure_change": max_allowed_pressure_change,
            },
        )

    # Check for any out-of-range pressures
    pressure_validation_result = _validate_pressure_range(
        padded_pressure_grid=padded_pressure_grid,
        time_step=time_step,
        padded_zeros_grid=padded_zeros_grid,
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

    dtype = get_dtype()
    padded_pressure_grid = clip(
        padded_pressure_grid, c.MINIMUM_VALID_PRESSURE, c.MAXIMUM_VALID_PRESSURE
    ).astype(dtype, copy=False)

    padded_fluid_properties = attrs.evolve(
        padded_fluid_properties, pressure_grid=padded_pressure_grid
    )

    # PVT update at new pressure
    logger.debug("Updating PVT fluid properties for saturation evolution")
    old_solution_gas_to_oil_ratio_grid = (
        padded_fluid_properties.solution_gas_to_oil_ratio_grid.copy()
    )
    old_oil_formation_volume_factor_grid = (
        padded_fluid_properties.oil_formation_volume_factor_grid.copy()
    )
    # Save pre-flash saturations to detect flash-induced saturation changes
    old_oil_saturation_grid = padded_fluid_properties.oil_saturation_grid.copy()
    old_gas_saturation_grid = padded_fluid_properties.gas_saturation_grid.copy()

    padded_fluid_properties = update_pvt_grids(
        fluid_properties=padded_fluid_properties,
        wells=wells,
        miscibility_model=miscibility_model,
        pvt_tables=config.pvt_tables,
        freeze_saturation_pressure=config.freeze_saturation_pressure,
    )

    # Solution gas liberation flash
    padded_fluid_properties = apply_solution_gas_liberation(
        fluid_properties=padded_fluid_properties,
        old_solution_gas_to_oil_ratio_grid=old_solution_gas_to_oil_ratio_grid,
        old_oil_formation_volume_factor_grid=old_oil_formation_volume_factor_grid,
    )

    # Check flash-induced saturation changes before proceeding to saturation solver.
    # If the liberation flash itself violates the saturation change limits, reject
    # the step immediately, as there's no point running the full saturation solver.
    flash_gas_sat_change = float(
        np.max(
            np.abs(
                padded_fluid_properties.gas_saturation_grid - old_gas_saturation_grid
            )
        )
    )
    flash_oil_sat_change = float(
        np.max(
            np.abs(
                padded_fluid_properties.oil_saturation_grid - old_oil_saturation_grid
            )
        )
    )
    flash_sat_check = _check_saturation_changes(
        max_oil_saturation_change=flash_oil_sat_change,
        max_water_saturation_change=0.0,
        max_gas_saturation_change=flash_gas_sat_change,
        max_allowed_oil_saturation_change=config.max_oil_saturation_change,
        max_allowed_water_saturation_change=config.max_water_saturation_change,
        max_allowed_gas_saturation_change=config.max_gas_saturation_change,
    )
    if flash_sat_check.violated:
        message = (
            f"Solution gas liberation flash at time step {time_step} violated "
            f"saturation change limits: {flash_sat_check.message}"
        )
        logger.debug(message)
        return StepResult(
            oil_injection_grid=padded_zeros_grid,
            water_injection_grid=padded_zeros_grid,
            gas_injection_grid=padded_zeros_grid,
            oil_production_grid=padded_zeros_grid,
            water_production_grid=padded_zeros_grid,
            gas_production_grid=padded_zeros_grid,
            fluid_properties=padded_fluid_properties,
            rock_properties=padded_rock_properties,
            saturation_history=padded_saturation_history,
            success=False,
            message=message,
            timer_kwargs={
                "max_saturation_change": flash_sat_check.max_phase_saturation_change,
                "max_allowed_saturation_change": flash_sat_check.max_allowed_phase_saturation_change,
            },
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

    logger.debug("Evolving saturation (implicit, Newton-Raphson)...")
    pressure_change_grid = padded_pressure_grid - old_pressure_grid

    # Build zeros grids to track production and injection at each time step
    oil_injection_grid = padded_zeros_grid.copy()
    water_injection_grid = padded_zeros_grid.copy()
    gas_injection_grid = padded_zeros_grid.copy()
    oil_production_grid = padded_zeros_grid.copy()
    water_production_grid = padded_zeros_grid.copy()
    gas_production_grid = padded_zeros_grid.copy()

    saturation_result = implicit.evolve_saturation(
        cell_dimension=cell_dimension,
        thickness_grid=padded_thickness_grid,
        elevation_grid=padded_elevation_grid,
        time_step=time_step,
        time_step_size=time_step_size,
        time=time,
        rock_properties=padded_rock_properties,
        fluid_properties=padded_fluid_properties,
        wells=wells,
        config=config,
        boundary_conditions=boundary_conditions,
        injection_grid=_RateGridsProxy(
            oil=oil_injection_grid,
            water=water_injection_grid,
            gas=gas_injection_grid,
        ),
        production_grid=_RateGridsProxy(
            oil=oil_production_grid,
            water=water_production_grid,
            gas=gas_production_grid,
        ),
        pressure_change_grid=pressure_change_grid,
        pad_width=pad_width,
    )
    saturation_solution = saturation_result.value
    saturation_change_result = _check_saturation_changes(
        max_oil_saturation_change=saturation_solution.max_oil_saturation_change,
        max_water_saturation_change=saturation_solution.max_water_saturation_change,
        max_gas_saturation_change=saturation_solution.max_gas_saturation_change,
        max_allowed_oil_saturation_change=config.max_oil_saturation_change,
        max_allowed_water_saturation_change=config.max_water_saturation_change,
        max_allowed_gas_saturation_change=config.max_gas_saturation_change,
    )
    timer_kwargs = {
        "max_pressure_change": max_pressure_change,
        "max_allowed_pressure_change": max_allowed_pressure_change,
        "max_saturation_change": saturation_change_result.max_phase_saturation_change
        or None,
        "max_allowed_saturation_change": saturation_change_result.max_allowed_phase_saturation_change
        or None,
        "newton_iterations": saturation_solution.newton_iterations,
    }

    if not saturation_result.success:
        logger.error(
            f"Implicit saturation evolution failed at time step {time_step}: \n{saturation_result.message}"
        )
        return StepResult(
            oil_injection_grid=oil_injection_grid,
            water_injection_grid=water_injection_grid,
            gas_injection_grid=gas_injection_grid,
            oil_production_grid=oil_production_grid,
            water_production_grid=water_production_grid,
            gas_production_grid=gas_production_grid,
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

        Oil saturation change: {saturation_solution.max_oil_saturation_change:.6f},
        Water saturation change: {saturation_solution.max_water_saturation_change:.6f},
        Gas saturation change: {saturation_solution.max_gas_saturation_change:.6f}.
        """
        logger.debug(message)
        return StepResult(
            oil_injection_grid=oil_injection_grid,
            water_injection_grid=water_injection_grid,
            gas_injection_grid=gas_injection_grid,
            oil_production_grid=oil_production_grid,
            water_production_grid=water_production_grid,
            gas_production_grid=gas_production_grid,
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
            saturation_epsilon=c.SATURATION_EPSILON,
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
        oil_injection_grid=oil_injection_grid,
        water_injection_grid=water_injection_grid,
        gas_injection_grid=gas_injection_grid,
        oil_production_grid=oil_production_grid,
        water_production_grid=water_production_grid,
        gas_production_grid=gas_production_grid,
        fluid_properties=padded_fluid_properties,
        rock_properties=padded_rock_properties,
        saturation_history=padded_saturation_history,
        success=True,
        message=saturation_result.message,
        timer_kwargs=timer_kwargs,
    )


def _run_explicit_step(
    time_step: int,
    padded_zeros_grid: NDimensionalGrid[ThreeDimensions],
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
    pad_width: int = 1,
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
    # Save old pressure grid before explicit solve (needed for PVT volume correction)
    old_pressure_grid = padded_fluid_properties.pressure_grid.copy()

    logger.debug("Evolving pressure (explicit)...")
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
        boundary_conditions=boundary_conditions,
        pad_width=pad_width,
    )
    pressure_solution = pressure_result.value
    max_pressure_change = pressure_solution.max_pressure_change
    max_allowed_pressure_change = config.max_pressure_change
    timer_kwargs = {
        "max_cfl_encountered": pressure_solution.max_cfl_encountered,
        "cfl_threshold": pressure_solution.cfl_threshold,
        "max_pressure_change": max_pressure_change,
        "max_allowed_pressure_change": max_allowed_pressure_change,
    }

    if not pressure_result.success:
        logger.error(
            f"Explicit pressure evolution failed at time step {time_step}: \n{pressure_result.message}"
        )
        return StepResult(
            oil_injection_grid=padded_zeros_grid,
            water_injection_grid=padded_zeros_grid,
            gas_injection_grid=padded_zeros_grid,
            oil_production_grid=padded_zeros_grid,
            water_production_grid=padded_zeros_grid,
            gas_production_grid=padded_zeros_grid,
            fluid_properties=padded_fluid_properties,
            rock_properties=padded_rock_properties,
            saturation_history=padded_saturation_history,
            success=False,
            message=pressure_result.message,
            timer_kwargs=timer_kwargs,
        )

    if max_pressure_change > max_allowed_pressure_change:
        message = (
            f"Pressure change {max_pressure_change:.6f} psi "
            f"exceeded maximum allowed {max_allowed_pressure_change:.6f} psi "
            f"at time step {time_step}."
        )
        logger.debug(message)
        return StepResult(
            oil_injection_grid=padded_zeros_grid,
            water_injection_grid=padded_zeros_grid,
            gas_injection_grid=padded_zeros_grid,
            oil_production_grid=padded_zeros_grid,
            water_production_grid=padded_zeros_grid,
            gas_production_grid=padded_zeros_grid,
            fluid_properties=padded_fluid_properties,
            rock_properties=padded_rock_properties,
            saturation_history=padded_saturation_history,
            success=False,
            message=message,
            timer_kwargs={
                "max_pressure_change": max_pressure_change,
                "max_allowed_pressure_change": max_allowed_pressure_change,
            },
        )

    padded_pressure_grid = pressure_solution.pressure_grid

    # Check for any out-of-range pressures
    pressure_validation_result = _validate_pressure_range(
        padded_pressure_grid=padded_pressure_grid,
        time_step=time_step,
        padded_zeros_grid=padded_zeros_grid,
        padded_fluid_properties=padded_fluid_properties,
        padded_rock_properties=padded_rock_properties,
        padded_saturation_history=padded_saturation_history,
    )
    if pressure_validation_result is not None:
        # Add `timer_kwargs` to the result for explicit scheme
        return StepResult(
            oil_injection_grid=pressure_validation_result.oil_injection_grid,
            water_injection_grid=pressure_validation_result.water_injection_grid,
            gas_injection_grid=pressure_validation_result.gas_injection_grid,
            oil_production_grid=pressure_validation_result.oil_production_grid,
            water_production_grid=pressure_validation_result.water_production_grid,
            gas_production_grid=pressure_validation_result.gas_production_grid,
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

    dtype = get_dtype()
    # Clamp pressures to valid range just for additional safety and to remove numerical noise
    padded_pressure_grid = clip(
        padded_pressure_grid, c.MINIMUM_VALID_PRESSURE, c.MAXIMUM_VALID_PRESSURE
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

    # Build zeros grids to track production and injection at each time step
    oil_injection_grid = padded_zeros_grid.copy()
    water_injection_grid = padded_zeros_grid.copy()
    gas_injection_grid = padded_zeros_grid.copy()
    oil_production_grid = padded_zeros_grid.copy()
    water_production_grid = padded_zeros_grid.copy()
    gas_production_grid = padded_zeros_grid.copy()

    # Compute pressure change grid for PVT volume correction
    pressure_change_grid = padded_pressure_grid - old_pressure_grid

    if miscibility_model == "immiscible":
        evolve_saturation = explicit.evolve_saturation
    else:
        evolve_saturation = explicit.evolve_miscible_saturation

    saturation_result = evolve_saturation(
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
        boundary_conditions=boundary_conditions,
        injection_grid=_RateGridsProxy(
            oil=oil_injection_grid,
            water=water_injection_grid,
            gas=gas_injection_grid,
        ),
        production_grid=_RateGridsProxy(
            oil=oil_production_grid,
            water=water_production_grid,
            gas=gas_production_grid,
        ),
        pressure_change_grid=pressure_change_grid,
        pad_width=pad_width,
    )
    saturation_solution = saturation_result.value

    saturation_change_result = _check_saturation_changes(
        max_oil_saturation_change=saturation_solution.max_oil_saturation_change,
        max_water_saturation_change=saturation_solution.max_water_saturation_change,
        max_gas_saturation_change=saturation_solution.max_gas_saturation_change,
        max_allowed_oil_saturation_change=config.max_oil_saturation_change,
        max_allowed_water_saturation_change=config.max_water_saturation_change,
        max_allowed_gas_saturation_change=config.max_gas_saturation_change,
    )
    timer_kwargs = {
        "max_cfl_encountered": saturation_solution.max_cfl_encountered,
        "cfl_threshold": saturation_solution.cfl_threshold,
        "max_pressure_change": max_pressure_change,
        "max_allowed_pressure_change": max_allowed_pressure_change,
        "max_saturation_change": saturation_change_result.max_phase_saturation_change
        or None,
        "max_allowed_saturation_change": saturation_change_result.max_allowed_phase_saturation_change
        or None,
    }

    if not saturation_result.success:
        logger.error(
            f"Explicit saturation evolution failed at time step {time_step}: \n{saturation_result.message}"
        )
        return StepResult(
            oil_injection_grid=oil_injection_grid,
            water_injection_grid=water_injection_grid,
            gas_injection_grid=gas_injection_grid,
            oil_production_grid=oil_production_grid,
            water_production_grid=water_production_grid,
            gas_production_grid=gas_production_grid,
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

        Oil saturation change: {saturation_solution.max_oil_saturation_change:.6f}, 
        Water saturation change: {saturation_solution.max_water_saturation_change:.6f}, 
        Gas saturation change: {saturation_solution.max_gas_saturation_change:.6f}.
        """
        logger.error(message)
        return StepResult(
            oil_injection_grid=oil_injection_grid,
            water_injection_grid=water_injection_grid,
            gas_injection_grid=gas_injection_grid,
            oil_production_grid=oil_production_grid,
            water_production_grid=water_production_grid,
            gas_production_grid=gas_production_grid,
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
            saturation_epsilon=c.SATURATION_EPSILON,
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
    # Save pre-flash saturations to detect flash-induced saturation changes
    old_oil_saturation_grid = padded_fluid_properties.oil_saturation_grid.copy()
    old_gas_saturation_grid = padded_fluid_properties.gas_saturation_grid.copy()

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
    padded_fluid_properties = apply_solution_gas_liberation(
        fluid_properties=padded_fluid_properties,
        old_solution_gas_to_oil_ratio_grid=old_solution_gas_to_oil_ratio_grid,
        old_oil_formation_volume_factor_grid=old_oil_formation_volume_factor_grid,
    )

    # Check flash-induced saturation changes so the next timestep doesn't
    # inherit a large uncontrolled saturation jump from the liberation flash.
    flash_gas_sat_change = float(
        np.max(
            np.abs(
                padded_fluid_properties.gas_saturation_grid - old_gas_saturation_grid
            )
        )
    )
    flash_oil_sat_change = float(
        np.max(
            np.abs(
                padded_fluid_properties.oil_saturation_grid - old_oil_saturation_grid
            )
        )
    )
    flash_sat_check = _check_saturation_changes(
        max_oil_saturation_change=flash_oil_sat_change,
        max_water_saturation_change=0.0,
        max_gas_saturation_change=flash_gas_sat_change,
        max_allowed_oil_saturation_change=config.max_oil_saturation_change,
        max_allowed_water_saturation_change=config.max_water_saturation_change,
        max_allowed_gas_saturation_change=config.max_gas_saturation_change,
    )
    if flash_sat_check.violated:
        message = (
            f"Solution gas liberation flash at time step {time_step} violated "
            f"saturation change limits: {flash_sat_check.message}"
        )
        logger.debug(message)
        return StepResult(
            oil_injection_grid=oil_injection_grid,
            water_injection_grid=water_injection_grid,
            gas_injection_grid=gas_injection_grid,
            oil_production_grid=oil_production_grid,
            water_production_grid=water_production_grid,
            gas_production_grid=gas_production_grid,
            fluid_properties=padded_fluid_properties,
            rock_properties=padded_rock_properties,
            saturation_history=padded_saturation_history,
            success=False,
            message=message,
            timer_kwargs={
                "max_saturation_change": flash_sat_check.max_phase_saturation_change,
                "max_allowed_saturation_change": flash_sat_check.max_allowed_phase_saturation_change,
            },
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
        oil_injection_grid=oil_injection_grid,
        water_injection_grid=water_injection_grid,
        gas_injection_grid=gas_injection_grid,
        oil_production_grid=oil_production_grid,
        water_production_grid=water_production_grid,
        gas_production_grid=gas_production_grid,
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
    ) -> typing.Generator[ModelState[ThreeDimensions], None, None]:
        """Returns a genrator that executes this simulation run."""
        return run(self.model, self.config)

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


def run(
    input: typing.Union[ReservoirModel[ThreeDimensions], Run],
    config: typing.Optional[Config] = None,
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
        boundary_conditions = BoundaryConditions[ThreeDimensions](
            conditions={
                "pressure": GridBoundaryCondition(),
                "oil_saturation": GridBoundaryCondition(),
                "gas_saturation": GridBoundaryCondition(),
                "water_saturation": GridBoundaryCondition(),
            }
        )

    logger.info("Starting simulation workflow...")
    cell_dimension = model.cell_dimension
    grid_shape = model.grid_shape
    has_wells = wells.exists()
    output_frequency = config.output_frequency
    scheme = config.scheme
    miscibility_model = config.miscibility_model

    logger.debug(f"Grid dimensions: {grid_shape}")
    logger.debug(f"Cell dimensions: {cell_dimension}")
    logger.debug(f"Evolution scheme: {scheme}")
    logger.debug(f"Total simulation time: {timer.simulation_time} seconds")
    logger.debug(f"Output frequency: every {output_frequency} steps")
    logger.debug(f"Has wells: {has_wells}")
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
        padded_fluid_properties = model.fluid_properties.pad(pad_width=pad_width)
        padded_rock_properties = model.rock_properties.pad(pad_width=pad_width)
        padded_saturation_history = model.saturation_history.pad(pad_width=pad_width)
        thickness_grid = model.thickness_grid
        padded_thickness_grid = pad_grid(thickness_grid, pad_width=pad_width)
        elevation_grid = model.get_elevation_grid(
            apply_dip=not config.disable_structural_dip
        )
        padded_elevation_grid = pad_grid(elevation_grid, pad_width=pad_width)

        # Apply boundary conditions to relevant padded grids
        logger.debug("Applying boundary conditions to initial padded grids")
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
            pvt_tables=config.pvt_tables,
            freeze_saturation_pressure=config.freeze_saturation_pressure,
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
            disable_capillary_effects=config.disable_capillary_effects,
            capillary_strength_factor=config.capillary_strength_factor,
            relative_mobility_range=config.relative_mobility_range,
            phase_appearance_tolerance=config.phase_appearance_tolerance,
        )
        relative_mobility_grids = padded_relative_mobility_grids.unpad(
            pad_width=pad_width
        )
        relperm_grids = padded_relperm_grids.unpad(pad_width=pad_width)
        capillary_pressure_grids = padded_capillary_pressure_grids.unpad(
            pad_width=pad_width
        )
        zeros_grid = build_uniform_grid(model.grid_shape, value=0.0)
        oil_injection_grid = zeros_grid
        water_injection_grid = zeros_grid.copy()
        gas_injection_grid = zeros_grid.copy()
        oil_production_grid = zeros_grid.copy()
        water_production_grid = zeros_grid.copy()
        gas_production_grid = zeros_grid.copy()
        state = ModelState(
            step=timer.step,
            step_size=timer.step_size,
            time=timer.elapsed_time,
            model=model,
            wells=wells,
            relative_mobilities=relative_mobility_grids,
            relative_permeabilities=relperm_grids,
            capillary_pressures=capillary_pressure_grids,
            injection=RateGrids(
                oil=oil_injection_grid,
                water=water_injection_grid,
                gas=gas_injection_grid,
            ),
            production=RateGrids(
                oil=oil_production_grid,
                water=water_production_grid,
                gas=gas_production_grid,
            ),
            timer_state=timer.dump_state(),
        )

        # Yield the initial model state
        logger.debug("Yielding initial model state")
        yield state

        no_flow_pressure_bc = isinstance(
            boundary_conditions["pressure"], type(default_bc)
        )
        padded_zeros_grid = pad_grid(zeros_grid, pad_width=pad_width)
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
                            pvt_tables=config.pvt_tables,
                            freeze_saturation_pressure=config.freeze_saturation_pressure,
                        )
                        logger.debug("PVT fluid properties updated")

                    # Build relative permeability, relative mobility, and capillary pressure grids
                    logger.debug(
                        f"Rebuilding rock-fluid property grids for time step {new_step}..."
                    )
                    padded_water_saturation_grid = (
                        padded_fluid_properties.water_saturation_grid
                    )
                    padded_oil_saturation_grid = (
                        padded_fluid_properties.oil_saturation_grid
                    )
                    padded_gas_saturation_grid = (
                        padded_fluid_properties.gas_saturation_grid
                    )
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
                    padded_water_viscosity_grid = (
                        padded_fluid_properties.water_viscosity_grid
                    )
                    padded_oil_viscosity_grid = (
                        padded_fluid_properties.oil_effective_viscosity_grid
                    )
                    padded_gas_viscosity_grid = (
                        padded_fluid_properties.gas_viscosity_grid
                    )
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
                        disable_capillary_effects=config.disable_capillary_effects,
                        capillary_strength_factor=config.capillary_strength_factor,
                        relative_mobility_range=config.relative_mobility_range,
                        phase_appearance_tolerance=config.phase_appearance_tolerance,
                    )

                if scheme == "impes":
                    result = _run_impes_step(
                        time_step=new_step,
                        grid_shape=grid_shape,
                        padded_zeros_grid=padded_zeros_grid,
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
                        pad_width=pad_width,
                    )
                elif scheme == "implicit":
                    result = _run_sequential_implicit_step(
                        time_step=new_step,
                        grid_shape=grid_shape,
                        padded_zeros_grid=padded_zeros_grid,
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
                        pad_width=pad_width,
                    )
                elif scheme == "explicit":
                    result = _run_explicit_step(
                        time_step=new_step,
                        grid_shape=grid_shape,
                        padded_zeros_grid=padded_zeros_grid,
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
                        pad_width=pad_width,
                    )
                else:
                    raise ValidationError(
                        f"Invalid simualtion scheme {scheme!r}. Available schemes include 'impes', 'implicit' or 'explicit'."
                    )

                # If the step was successful, accept that step proposal
                if result.success:
                    # Now we can accept the proposed time step size and we now agree that this is a new step
                    logger.debug(f"Time step {new_step} completed successfully.")
                    timer.accept_step(step_size=step_size, **result.timer_kwargs)
                    if config.log_interval:
                        log_progress(
                            step=timer.step,
                            step_size=step_size,
                            time_elapsed=timer.elapsed_time,
                            total_time=timer.simulation_time,
                            is_last_step=timer.is_last_step,
                            interval=config.log_interval,
                        )
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
                    oil_injection_grid = unpad_grid(
                        result.oil_injection_grid, pad_width=pad_width
                    )
                    water_injection_grid = unpad_grid(
                        result.water_injection_grid, pad_width=pad_width
                    )
                    gas_injection_grid = unpad_grid(
                        result.gas_injection_grid, pad_width=pad_width
                    )
                    oil_production_grid = unpad_grid(
                        result.oil_production_grid, pad_width=pad_width
                    )
                    water_production_grid = unpad_grid(
                        result.water_production_grid, pad_width=pad_width
                    )
                    gas_production_grid = unpad_grid(
                        result.gas_production_grid, pad_width=pad_width
                    )

                    # Negate production rates in-place (use out parameter to avoid -0.0 issues)
                    np.negative(oil_production_grid, out=oil_production_grid)
                    np.negative(water_production_grid, out=water_production_grid)
                    np.negative(gas_production_grid, out=gas_production_grid)
                    injection_rates = RateGrids(
                        oil=oil_injection_grid,
                        water=water_injection_grid,
                        gas=gas_injection_grid,
                    )
                    production_rates = RateGrids(
                        oil=oil_production_grid,
                        water=water_production_grid,
                        gas=gas_production_grid,
                    )
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
                        injection=injection_rates,
                        production=production_rates,
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
