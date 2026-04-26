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

from bores.boundary_conditions import BoundaryConditions, build_boundary_metadata
from bores.config import Config
from bores.constants import c
from bores.datastructures import (
    BottomHolePressure,
    BottomHolePressures,
    FormationVolumeFactors,
    Rates,
    SparseTensor,
)
from bores.errors import SimulationError, StopSimulation, TimingError, ValidationError
from bores.grids.base import CapillaryPressureGrids, RelativeMobilityGrids, RelPermGrids
from bores.grids.pvt import build_three_phase_relative_mobilities_grids
from bores.grids.rock_fluid import build_rock_fluid_properties_grids
from bores.initialization import (
    apply_minimum_injector_saturations,
    check_zero_flow_initialization,
    seed_injection_saturations,
)
from bores.material_balance import (
    MaterialBalanceErrors,
    compute_material_balance_errors,
)
from bores.models import (
    FluidProperties,
    ReservoirModel,
    RockProperties,
    SaturationHistory,
)
from bores.precision import get_dtype
from bores.solvers import explicit, implicit
from bores.solvers.base import normalize_saturations
from bores.solvers.rates import WellRates, compute_well_rates
from bores.states import ModelState, RatesInfo
from bores.stores import StoreSerializable
from bores.tables.pvt import PVTDataSet, PVTTables
from bores.transmissibility import FaceTransmissibilities
from bores.types import MiscibilityModel, NDimension, NDimensionalGrid, ThreeDimensions
from bores.updates import update_fluid_properties, update_residual_saturation_grids
from bores.wells.base import Wells
from bores.wells.indices import (
    WellsIndices,
    build_wells_indices,
    update_wells_indices,
)

__all__ = ["Run", "run"]

logger = logging.getLogger(__name__)

PRESSURE_ERROR_MSG = """
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
    """Result from executing one time step of the simulation."""

    fluid_properties: FluidProperties[NDimension]
    """Updated fluid properties after the time step."""
    rock_properties: RockProperties[NDimension]
    """Updated rock properties after the time step."""
    saturation_history: typing.Optional[SaturationHistory[NDimension]] = None
    """Updated saturation history after the time step."""
    rates: typing.Optional[WellRates[NDimension]] = None
    """Well rate info for the step"""
    success: bool = True
    """Whether the time step evolution was successful."""
    message: typing.Optional[str] = None
    """Optional message providing additional information about the time step result."""
    material_balance_errors: typing.Optional[MaterialBalanceErrors] = None
    """Material balance errors for this time step. None for rejected steps or first step."""
    timer_context: typing.Dict[str, typing.Any] = attrs.field(factory=dict)
    """Keyword argument that should be passed to the simulation timer on accepting or rejecting a step."""


@attrs.frozen(slots=True)
class SaturationChangeCheckResult:
    violated: bool
    max_phase_saturation_change: typing.Optional[float]
    max_allowed_phase_saturation_change: typing.Optional[float]
    message: typing.Optional[str] = None


def _validate_pressure_range(
    pressure_grid: NDimensionalGrid[ThreeDimensions],
    time_step: int,
    fluid_properties: FluidProperties[ThreeDimensions],
    rock_properties: RockProperties[ThreeDimensions],
    saturation_history: typing.Optional[SaturationHistory[ThreeDimensions]] = None,
) -> typing.Optional[StepResult[ThreeDimensions]]:
    """
    Check for out-of-range pressures and return a failure `StepResult` if found.

    :param pressure_grid: Pressure grid to validate.
    :param time_step: Current time step index.
    :param fluid_properties: Fluid properties for the current state.
    :param rock_properties: Rock properties for the current state.
    :param saturation_history: Saturation history for the current state.
    :return: `StepResult` with failure if pressures are out of range, *None* otherwise.
    """
    min_allowable = c.MINIMUM_VALID_PRESSURE - 1e-3
    max_allowable = c.MAXIMUM_VALID_PRESSURE + 1e-3
    out_of_range_mask = (pressure_grid < min_allowable) | (
        pressure_grid > max_allowable
    )
    out_of_range_indices = np.argwhere(out_of_range_mask)

    if out_of_range_indices.size > 0:
        min_p = float(np.min(pressure_grid))
        max_p = float(np.max(pressure_grid))
        logger.warning(
            f"Unphysical pressure detected at {out_of_range_indices.size} cells. "
            f"Range: [{min_p:.4f}, {max_p:.4f}] psi. "
            f"Allowed: [{min_allowable}, {max_allowable}]."
        )
        message = ""
        if min_p < min_allowable:
            message += (
                f"Pressure dropped below {min_allowable} psi (Min: {min_p:.4f}).\n"
            )
        if max_p > max_allowable:
            message += f"Pressure exceeded {max_allowable} psi (Max: {max_p:.4f}).\n"
        message += (
            PRESSURE_ERROR_MSG.format(indices=out_of_range_indices.tolist())
            + f"\nAt Time Step {time_step}."
        )
        return StepResult(
            fluid_properties=fluid_properties,
            rock_properties=rock_properties,
            saturation_history=saturation_history,
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
    """Check whether any phase saturation change exceeds its configured limit.

    If multiple phases violate their limits the result carries the largest
    absolute violation and its corresponding allowed maximum.

    :param maximum_oil_saturation_change: Maximum oil saturation change observed.
    :param maximum_water_saturation_change: Maximum water saturation change observed.
    :param maximum_gas_saturation_change: Maximum gas saturation change observed.
    :param max_allowed_oil_saturation_change: Maximum allowed oil saturation change.
    :param max_allowed_water_saturation_change: Maximum allowed water saturation change.
    :param max_allowed_gas_saturation_change: Maximum allowed gas saturation change.
    :param tolerance: Relative tolerance applied on top of each phase limit before
        a violation is declared.
    :return: `SaturationChangeCheckResult` describing the outcome.
    """
    violated = False
    messages = []
    max_phase_saturation_change = None
    max_allowed_phase_saturation_change = None

    oil_tol = max(tolerance, 0.005 * max_allowed_oil_saturation_change)
    max_allowed_oil_saturation_change *= 1 + oil_tol
    if maximum_oil_saturation_change > max_allowed_oil_saturation_change:
        violated = True
        max_phase_saturation_change = maximum_oil_saturation_change
        max_allowed_phase_saturation_change = max_allowed_oil_saturation_change
        messages.append(
            f"Oil saturation change {maximum_oil_saturation_change:.9f} exceeded "
            f"maximum allowed {max_allowed_oil_saturation_change:.9f}."
        )

    water_tol = max(tolerance, 0.005 * max_allowed_water_saturation_change)
    max_allowed_water_saturation_change *= 1 + water_tol
    if maximum_water_saturation_change > max_allowed_water_saturation_change:
        violated = True
        if (
            max_phase_saturation_change is None
            or maximum_water_saturation_change > max_phase_saturation_change
        ):
            max_phase_saturation_change = maximum_water_saturation_change
            max_allowed_phase_saturation_change = max_allowed_water_saturation_change
        messages.append(
            f"Water saturation change {maximum_water_saturation_change:.9f} exceeded "
            f"maximum allowed {max_allowed_water_saturation_change:.9f}."
        )

    gas_tol = max(tolerance, 0.005 * max_allowed_gas_saturation_change)
    max_allowed_gas_saturation_change *= 1 + gas_tol
    if maximum_gas_saturation_change > max_allowed_gas_saturation_change:
        violated = True
        if (
            max_phase_saturation_change is None
            or maximum_gas_saturation_change > max_phase_saturation_change
        ):
            max_phase_saturation_change = maximum_gas_saturation_change
            max_allowed_phase_saturation_change = max_allowed_gas_saturation_change
        messages.append(
            f"Gas saturation change {maximum_gas_saturation_change:.9f} exceeded "
            f"maximum allowed {max_allowed_gas_saturation_change:.9f}."
        )

    return SaturationChangeCheckResult(
        violated=violated,
        max_phase_saturation_change=max_phase_saturation_change,
        max_allowed_phase_saturation_change=max_allowed_phase_saturation_change,
        message="\n".join(messages) if messages else None,
    )


def _make_rates(grid_shape: NDimension) -> Rates[float, NDimension]:
    return Rates(
        oil=SparseTensor(grid_shape, dtype=float),
        water=SparseTensor(grid_shape, dtype=float),
        gas=SparseTensor(grid_shape, dtype=float),
    )


def _make_fvfs(grid_shape: NDimension) -> FormationVolumeFactors[float, NDimension]:
    return FormationVolumeFactors(
        oil=SparseTensor(grid_shape, dtype=float, default=np.nan),
        water=SparseTensor(grid_shape, dtype=float, default=np.nan),
        gas=SparseTensor(grid_shape, dtype=float, default=np.nan),
    )


def _make_bhps(grid_shape: NDimension) -> BottomHolePressures[float, NDimension]:
    return BottomHolePressures(
        oil=BottomHolePressure(grid_shape, dtype=float, default=np.nan),
        water=BottomHolePressure(grid_shape, dtype=float, default=np.nan),
        gas=BottomHolePressure(grid_shape, dtype=float, default=np.nan),
    )


def _rebuild_rock_fluid_grids(
    fluid_properties: FluidProperties[ThreeDimensions],
    rock_properties: RockProperties[ThreeDimensions],
    config: Config,
) -> typing.Tuple[
    RelPermGrids[ThreeDimensions],
    RelativeMobilityGrids[ThreeDimensions],
    CapillaryPressureGrids[ThreeDimensions],
]:
    """
    Rebuild relative permeability, mobility, and capillary pressure grids from current saturations.

    :param fluid_properties: Current fluid properties.
    :param rock_properties: Current rock properties.
    :param config: Simulation configuration.
    :return: 3-tuple of `(relperm_grids, relative_mobility_grids, capillary_pressure_grids)`.
    """
    return build_rock_fluid_properties_grids(
        water_saturation_grid=fluid_properties.water_saturation_grid,
        oil_saturation_grid=fluid_properties.oil_saturation_grid,
        gas_saturation_grid=fluid_properties.gas_saturation_grid,
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


# STEP FUNCTIONS


def _run_impes_step(
    time_step: int,
    grid_shape: ThreeDimensions,
    cell_dimension: typing.Tuple[float, float],
    thickness_grid: NDimensionalGrid[ThreeDimensions],
    elevation_grid: NDimensionalGrid[ThreeDimensions],
    time_step_size: float,
    time: float,
    face_transmissibilities: FaceTransmissibilities,
    rock_properties: RockProperties[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
    saturation_history: typing.Optional[SaturationHistory[ThreeDimensions]],
    relperm_grids: RelPermGrids[ThreeDimensions],
    relative_mobility_grids: RelativeMobilityGrids[ThreeDimensions],
    capillary_pressure_grids: CapillaryPressureGrids[ThreeDimensions],
    wells: Wells[ThreeDimensions],
    miscibility_model: MiscibilityModel,
    config: Config,
    boundary_conditions: BoundaryConditions[ThreeDimensions],
    wells_indices: WellsIndices,
    dtype: npt.DTypeLike,
    min_valid_pressure: float = 14.7,
    max_valid_pressure: float = 14700,
    saturation_epsilon: float = 1e-12,
) -> StepResult[ThreeDimensions]:
    """
    Execute one time step using IMPES (Implicit Pressure, Explicit Saturation).

    :param time_step: Current time step index.
    :param grid_shape: Model grid shape (nx, ny, nz).
    :param cell_dimension: Tuple of cell dimensions (dx, dy) in feet.
    :param thickness_grid: Cell thickness grid (ft).
    :param elevation_grid: Cell elevation grid (ft).
    :param time_step_size: Size of the current time step (seconds).
    :param time: Total simulation time elapsed, this time step inclusive (seconds).
    :param face_transmissibilities: Precomputed geometric face transmissibilities.
    :param rock_properties: Rock properties.
    :param fluid_properties: Fluid properties.
    :param saturation_history: Saturation history, or *None* if hysteresis is disabled.
    :param relperm_grids: Three-phase relative permeability grids.
    :param relative_mobility_grids: Three-phase relative mobility grids.
    :param capillary_pressure_grids: Oil-water and gas-oil capillary pressure grids.
    :param wells: Wells in the reservoir.
    :param miscibility_model: Miscibility model used in the simulation.
    :param config: Simulation configuration.
    :param boundary_conditions: Model boundary conditions.
    :param wells_indices: Cache of well indices for efficient lookup during the pressure solve.
    :param dtype: Data type used for numerical arrays.
    :param min_valid_pressure: Minimum valid pressure (psi). Pressures below this trigger a failure.
    :param max_valid_pressure: Maximum valid pressure (psi). Pressures above this trigger a failure.
    :param saturation_epsilon: Small value to keep saturations strictly between 0 and 1.
    :return: `StepResult` containing updated fluid properties, rock properties, and rates.
    """
    old_pressure_grid = fluid_properties.pressure_grid.copy()
    initial_fluid_properties = fluid_properties

    # Build boundary metadata and get the initial ghost-cell maps.
    metadata = build_boundary_metadata(
        fluid_properties=fluid_properties,
        rock_properties=rock_properties,
        relperm_grids=relperm_grids,
        relative_mobility_grids=relative_mobility_grids,
        capillary_pressure_grids=capillary_pressure_grids,
        face_transmissibilities=face_transmissibilities,
        grid_shape=grid_shape,
        cell_dimension=cell_dimension,
        thickness_grid=thickness_grid,
        time=time,
        dtype=dtype,
    )
    flux_boundaries, pressure_boundaries = boundary_conditions.get_boundaries(
        grid_shape=grid_shape, metadata=metadata
    )

    if has_open_wells := wells.any_open():
        logger.debug("Computing well rates...")
        well_rates = compute_well_rates(
            fluid_properties=fluid_properties,
            water_relative_mobility_grid=relative_mobility_grids.water_relative_mobility,
            oil_relative_mobility_grid=relative_mobility_grids.oil_relative_mobility,
            gas_relative_mobility_grid=relative_mobility_grids.gas_relative_mobility,
            wells=wells,
            time=time,
            config=config,
            wells_indices=wells_indices,
            dtype=dtype,
        )

    else:
        well_rates = None

    logger.debug("Evolving pressure (implicit)...")
    pressure_result = implicit.solve_pressure(
        cell_dimension=cell_dimension,
        thickness_grid=thickness_grid,
        elevation_grid=elevation_grid,
        time_step=time_step,
        time_step_size=time_step_size,
        rock_properties=rock_properties,
        fluid_properties=fluid_properties,
        relative_mobility_grids=relative_mobility_grids,
        capillary_pressure_grids=capillary_pressure_grids,
        face_transmissibilities=face_transmissibilities,
        config=config,
        flux_boundaries=flux_boundaries,
        pressure_boundaries=pressure_boundaries,
        net_well_rate_grid=well_rates.net_well_rate_grid if has_open_wells else None,  # type: ignore
        dtype=dtype,
    )
    if not pressure_result.success:
        logger.error(
            f"Implicit pressure evolution failed at time step {time_step}: \n{pressure_result.message}"
        )
        return StepResult(
            fluid_properties=fluid_properties,
            rock_properties=rock_properties,
            saturation_history=saturation_history,
            success=False,
            message=pressure_result.message,
        )

    pressure_solution = pressure_result.value
    new_pressure_grid = pressure_solution.pressure_grid
    maximum_pressure_change = pressure_solution.maximum_pressure_change
    maximum_allowed_pressure_change = config.maximum_pressure_change * 1.0001

    if maximum_pressure_change > maximum_allowed_pressure_change:
        message = (
            f"Pressure change {maximum_pressure_change:.6f} psi "
            f"exceeded maximum allowed {maximum_allowed_pressure_change:.6f} psi "
            f"at time step {time_step}."
        )
        logger.warning(message)
        return StepResult(
            fluid_properties=fluid_properties,
            rock_properties=rock_properties,
            saturation_history=saturation_history,
            success=False,
            message=message,
            timer_context={
                "maximum_pressure_change": maximum_pressure_change,
                "maximum_allowed_pressure_change": maximum_allowed_pressure_change,
            },
        )

    result = _validate_pressure_range(
        pressure_grid=new_pressure_grid,
        time_step=time_step,
        fluid_properties=fluid_properties,
        rock_properties=rock_properties,
        saturation_history=saturation_history,
    )
    if result is not None:
        return result

    np.clip(
        new_pressure_grid,
        min_valid_pressure,
        max_valid_pressure,
        dtype=dtype,
        out=new_pressure_grid,
    )
    fluid_properties = attrs.evolve(fluid_properties, pressure_grid=new_pressure_grid)
    logger.debug("Pressure evolution completed.")

    # Copy before PVT updates
    old_solution_gor_grid = fluid_properties.solution_gas_to_oil_ratio_grid.copy()
    old_gas_solubility_in_water_grid = (
        fluid_properties.gas_solubility_in_water_grid.copy()
    )
    old_oil_fvf_grid = fluid_properties.oil_formation_volume_factor_grid.copy()
    old_gas_fvf_grid = fluid_properties.gas_formation_volume_factor_grid.copy()
    old_water_fvf_grid = fluid_properties.water_formation_volume_factor_grid.copy()
    old_water_density_grid = fluid_properties.water_density_grid.copy()
    old_oil_density_grid = fluid_properties.oil_effective_density_grid.copy()
    old_gas_density_grid = fluid_properties.gas_density_grid.copy()

    logger.debug("Updating PVT fluid properties to reflect pressure change...")
    fluid_properties = update_fluid_properties(
        fluid_properties=fluid_properties,
        wells=wells,
        miscibility_model=miscibility_model,
        pvt_tables=config.pvt_tables,
        freeze_saturation_pressure=config.freeze_saturation_pressure,
    )

    # Partially update mobility grids (kr/μ) using updated viscosities
    (
        water_relative_mobility_grid,
        oil_relative_mobility_grid,
        gas_relative_mobility_grid,
    ) = build_three_phase_relative_mobilities_grids(
        oil_relative_permeability_grid=relperm_grids.kro,
        water_relative_permeability_grid=relperm_grids.krw,
        gas_relative_permeability_grid=relperm_grids.krg,
        water_viscosity_grid=fluid_properties.water_viscosity_grid,
        oil_viscosity_grid=fluid_properties.oil_effective_viscosity_grid,
        gas_viscosity_grid=fluid_properties.gas_viscosity_grid,
    )
    relative_mobility_grids = RelativeMobilityGrids(
        oil_relative_mobility=oil_relative_mobility_grid,
        water_relative_mobility=water_relative_mobility_grid,
        gas_relative_mobility=gas_relative_mobility_grid,
    )

    # Refresh boundary conditions after pressure update so that dynamic BCs (Robin,
    # Carter-Tracy) see the new interior pressures before saturation evolves.
    logger.debug("Refreshing boundary conditions after pressure solve...")
    metadata = attrs.evolve(metadata, fluid_properties=fluid_properties)
    flux_boundaries, pressure_boundaries = boundary_conditions.refresh_boundaries(
        metadata=metadata
    )

    logger.debug("Evolving saturation (explicit)...")
    transport_result = explicit.solve_transport(
        cell_dimension=cell_dimension,
        thickness_grid=thickness_grid,
        elevation_grid=elevation_grid,
        time_step=time_step,
        time_step_size=time_step_size,
        rock_properties=rock_properties,
        fluid_properties=fluid_properties,
        old_water_density_grid=old_water_density_grid,
        old_oil_density_grid=old_oil_density_grid,
        old_gas_density_grid=old_gas_density_grid,
        old_solution_gas_to_oil_ratio_grid=old_solution_gor_grid,
        old_gas_solubility_in_water_grid=old_gas_solubility_in_water_grid,
        old_gas_formation_volume_factor_grid=old_gas_fvf_grid,
        old_oil_formation_volume_factor_grid=old_oil_fvf_grid,
        old_water_formation_volume_factor_grid=old_water_fvf_grid,
        relative_mobility_grids=relative_mobility_grids,
        capillary_pressure_grids=capillary_pressure_grids,
        face_transmissibilities=face_transmissibilities,
        config=config,
        flux_boundaries=flux_boundaries,
        pressure_boundaries=pressure_boundaries,
        rates=well_rates,
        dtype=dtype,
    )

    transport_solution = transport_result.value
    saturation_check = _check_saturation_changes(
        maximum_oil_saturation_change=transport_solution.maximum_oil_saturation_change,
        maximum_water_saturation_change=transport_solution.maximum_water_saturation_change,
        maximum_gas_saturation_change=transport_solution.maximum_gas_saturation_change,
        max_allowed_oil_saturation_change=config.maximum_oil_saturation_change,
        max_allowed_water_saturation_change=config.maximum_water_saturation_change,
        max_allowed_gas_saturation_change=config.maximum_gas_saturation_change,
        tolerance=1e-4,
    )
    timer_context = {
        "maximum_cfl_encountered": transport_solution.maximum_cfl_encountered,
        "cfl_threshold": transport_solution.cfl_threshold,
        "maximum_pressure_change": maximum_pressure_change,
        "maximum_allowed_pressure_change": maximum_allowed_pressure_change,
        "maximum_saturation_change": saturation_check.max_phase_saturation_change,
        "maximum_allowed_saturation_change": saturation_check.max_allowed_phase_saturation_change,
    }

    if not transport_result.success:
        logger.warning(
            f"Explicit saturation evolution failed at time step {time_step}: \n{transport_result.message}"
        )
        return StepResult(
            fluid_properties=fluid_properties,
            rock_properties=rock_properties,
            saturation_history=saturation_history,
            success=False,
            message=transport_result.message,
            timer_context=timer_context,
        )

    if saturation_check.violated:
        message = (
            f"At time step {time_step}, saturation change limits were violated:\n"
            f"{saturation_check.message}\n"
            f"Oil: {transport_solution.maximum_oil_saturation_change:.6f}, "
            f"Water: {transport_solution.maximum_water_saturation_change:.6f}, "
            f"Gas: {transport_solution.maximum_gas_saturation_change:.6f}."
        )
        logger.debug(message)
        return StepResult(
            fluid_properties=fluid_properties,
            rock_properties=rock_properties,
            saturation_history=saturation_history,
            success=False,
            message=message,
            timer_context=timer_context,
        )

    logger.debug("Updating fluid properties with new saturation grids...")
    water_saturation_grid = transport_solution.water_saturation_grid.astype(
        dtype, copy=False
    )
    oil_saturation_grid = transport_solution.oil_saturation_grid.astype(
        dtype, copy=False
    )
    gas_saturation_grid = transport_solution.gas_saturation_grid.astype(
        dtype, copy=False
    )
    solvent_concentration_grid = transport_solution.solvent_concentration_grid

    if solvent_concentration_grid is None:
        fluid_properties = attrs.evolve(
            fluid_properties,
            water_saturation_grid=water_saturation_grid,
            oil_saturation_grid=oil_saturation_grid,
            gas_saturation_grid=gas_saturation_grid,
        )
    else:
        fluid_properties = attrs.evolve(
            fluid_properties,
            water_saturation_grid=water_saturation_grid,
            oil_saturation_grid=oil_saturation_grid,
            gas_saturation_grid=gas_saturation_grid,
            solvent_concentration_grid=solvent_concentration_grid.astype(
                dtype, copy=False
            ),
        )

    if config.normalize_saturations:
        normalize_saturations(
            oil_saturation_grid=fluid_properties.oil_saturation_grid,
            water_saturation_grid=fluid_properties.water_saturation_grid,
            gas_saturation_grid=fluid_properties.gas_saturation_grid,
            saturation_epsilon=saturation_epsilon,
        )

    if saturation_history is not None:
        rock_properties, saturation_history = update_residual_saturation_grids(
            rock_properties=rock_properties,
            saturation_history=saturation_history,
            water_saturation_grid=fluid_properties.water_saturation_grid,
            gas_saturation_grid=fluid_properties.gas_saturation_grid,
            residual_oil_drainage_ratio_water_flood=config.residual_oil_drainage_ratio_water_flood,
            residual_oil_drainage_ratio_gas_flood=config.residual_oil_drainage_ratio_gas_flood,
            residual_gas_drainage_ratio=config.residual_gas_drainage_ratio,
        )

    material_balance_errors = compute_material_balance_errors(
        current_fluid_properties=fluid_properties,
        previous_fluid_properties=initial_fluid_properties,
        rock=rock_properties,
        thickness_grid=thickness_grid,
        cell_dimension=cell_dimension,
        injection_mass_rates=well_rates.injection_mass_rates  # type: ignore
        if has_open_wells
        else None,
        production_mass_rates=well_rates.production_mass_rates  # type: ignore
        if has_open_wells
        else None,
        time_step_size=time_step_size,
    )
    timer_context.update(
        {
            "absolute_oil_mbe": material_balance_errors.absolute_oil_mbe,
            "absolute_water_mbe": material_balance_errors.absolute_water_mbe,
            "absolute_gas_mbe": material_balance_errors.absolute_gas_mbe,
            "total_absolute_mbe": material_balance_errors.total_absolute_mbe,
            "relative_oil_mbe": material_balance_errors.relative_oil_mbe,
            "relative_water_mbe": material_balance_errors.relative_water_mbe,
            "relative_gas_mbe": material_balance_errors.relative_gas_mbe,
            "total_relative_mbe": material_balance_errors.total_relative_mbe,
        }
    )
    logger.debug("Saturation evolution completed.")
    return StepResult(
        fluid_properties=fluid_properties,
        rock_properties=rock_properties,
        saturation_history=saturation_history,
        rates=well_rates,
        success=True,
        message=transport_result.message,
        material_balance_errors=material_balance_errors,
        timer_context=timer_context,
    )


def _run_sequential_implicit_step(
    time_step: int,
    grid_shape: ThreeDimensions,
    cell_dimension: typing.Tuple[float, float],
    thickness_grid: NDimensionalGrid[ThreeDimensions],
    elevation_grid: NDimensionalGrid[ThreeDimensions],
    time_step_size: float,
    time: float,
    face_transmissibilities: FaceTransmissibilities,
    rock_properties: RockProperties[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
    saturation_history: typing.Optional[SaturationHistory[ThreeDimensions]],
    relperm_grids: RelPermGrids[ThreeDimensions],
    relative_mobility_grids: RelativeMobilityGrids[ThreeDimensions],
    capillary_pressure_grids: CapillaryPressureGrids[ThreeDimensions],
    wells: Wells[ThreeDimensions],
    miscibility_model: MiscibilityModel,
    config: Config,
    boundary_conditions: BoundaryConditions[ThreeDimensions],
    wells_indices: WellsIndices,
    dtype: npt.DTypeLike,
    min_valid_pressure: float = 14.7,
    max_valid_pressure: float = 14700,
    saturation_epsilon: float = 1e-12,
) -> StepResult[ThreeDimensions]:
    """
    Execute one time step using the Sequential Implicit (SI) scheme.

    Pressure is solved implicitly, then saturation is solved implicitly using
    Newton-Raphson iteration. This eliminates the CFL stability constraint on
    saturation transport, allowing larger time steps than IMPES.

    :param time_step: Current time step index.
    :param grid_shape: Model grid shape (nx, ny, nz).
    :param cell_dimension: Tuple of cell dimensions (dx, dy) in feet.
    :param thickness_grid: Cell thickness grid (ft).
    :param elevation_grid: Cell elevation grid (ft).
    :param time_step_size: Size of the current time step (seconds).
    :param time: Total simulation time elapsed, this time step inclusive (seconds).
    :param face_transmissibilities: Precomputed geometric face transmissibilities.
    :param rock_properties: Rock properties.
    :param fluid_properties: Fluid properties.
    :param saturation_history: Saturation history, or *None* if hysteresis is disabled.
    :param relperm_grids: Three-phase relative permeability grids.
    :param relative_mobility_grids: Three-phase relative mobility grids.
    :param capillary_pressure_grids: Oil-water and gas-oil capillary pressure grids.
    :param wells: Wells in the reservoir.
    :param miscibility_model: Miscibility model used in the simulation.
    :param config: Simulation configuration.
    :param boundary_conditions: Model boundary conditions.
    :param wells_indices: Cache of well indices for efficient lookup during the pressure solve.
    :param dtype: Data type used for numerical arrays.
    :param min_valid_pressure: Minimum valid pressure (psi). Pressures below this trigger a failure.
    :param max_valid_pressure: Maximum valid pressure (psi). Pressures above this trigger a failure.
    :param saturation_epsilon: Small value to keep saturations strictly between 0 and 1.
    :return: `StepResult` containing updated fluid properties, rock properties, and rates.
    """
    initial_fluid_properties = fluid_properties

    metadata = build_boundary_metadata(
        fluid_properties=fluid_properties,
        rock_properties=rock_properties,
        relperm_grids=relperm_grids,
        relative_mobility_grids=relative_mobility_grids,
        capillary_pressure_grids=capillary_pressure_grids,
        face_transmissibilities=face_transmissibilities,
        grid_shape=grid_shape,
        cell_dimension=cell_dimension,
        thickness_grid=thickness_grid,
        time=time,
        dtype=dtype,
    )
    flux_boundaries, pressure_boundaries = boundary_conditions.get_boundaries(
        grid_shape=grid_shape, metadata=metadata
    )

    if has_open_wells := wells.any_open():
        logger.debug("Computing well rates...")
        well_rates = compute_well_rates(
            fluid_properties=fluid_properties,
            water_relative_mobility_grid=relative_mobility_grids.water_relative_mobility,
            oil_relative_mobility_grid=relative_mobility_grids.oil_relative_mobility,
            gas_relative_mobility_grid=relative_mobility_grids.gas_relative_mobility,
            wells=wells,
            time=time,
            config=config,
            wells_indices=wells_indices,
            dtype=dtype,
        )

    else:
        well_rates = None

    logger.debug("Evolving pressure (implicit)...")
    pressure_result = implicit.solve_pressure(
        cell_dimension=cell_dimension,
        thickness_grid=thickness_grid,
        elevation_grid=elevation_grid,
        time_step=time_step,
        time_step_size=time_step_size,
        rock_properties=rock_properties,
        fluid_properties=fluid_properties,
        relative_mobility_grids=relative_mobility_grids,
        capillary_pressure_grids=capillary_pressure_grids,
        face_transmissibilities=face_transmissibilities,
        config=config,
        flux_boundaries=flux_boundaries,
        pressure_boundaries=pressure_boundaries,
        net_well_rate_grid=well_rates.net_well_rate_grid if has_open_wells else None,  # type: ignore
        dtype=dtype,
    )
    if not pressure_result.success:
        logger.warning(
            f"Implicit pressure evolution failed at time step {time_step}: \n{pressure_result.message}"
        )
        return StepResult(
            fluid_properties=fluid_properties,
            rock_properties=rock_properties,
            saturation_history=saturation_history,
            success=False,
            message=pressure_result.message,
        )

    pressure_solution = pressure_result.value
    new_pressure_grid = pressure_solution.pressure_grid
    maximum_pressure_change = pressure_solution.maximum_pressure_change
    maximum_allowed_pressure_change = config.maximum_pressure_change * 1.0001

    if maximum_pressure_change > maximum_allowed_pressure_change:
        message = (
            f"Pressure change {maximum_pressure_change:.6f} psi "
            f"exceeded maximum allowed {maximum_allowed_pressure_change:.6f} psi "
            f"at time step {time_step}."
        )
        logger.warning(message)
        return StepResult(
            fluid_properties=fluid_properties,
            rock_properties=rock_properties,
            saturation_history=saturation_history,
            success=False,
            message=message,
            timer_context={
                "maximum_pressure_change": maximum_pressure_change,
                "maximum_allowed_pressure_change": maximum_allowed_pressure_change,
            },
        )

    result = _validate_pressure_range(
        pressure_grid=new_pressure_grid,
        time_step=time_step,
        fluid_properties=fluid_properties,
        rock_properties=rock_properties,
        saturation_history=saturation_history,
    )
    if result is not None:
        return result

    np.clip(
        new_pressure_grid,
        min_valid_pressure,
        max_valid_pressure,
        dtype=dtype,
        out=new_pressure_grid,
    )
    fluid_properties = attrs.evolve(fluid_properties, pressure_grid=new_pressure_grid)
    logger.debug("Pressure evolution completed.")

    # Copy before PVT updates
    old_solution_gor_grid = fluid_properties.solution_gas_to_oil_ratio_grid.copy()
    old_gas_solubility_in_water_grid = (
        fluid_properties.gas_solubility_in_water_grid.copy()
    )
    old_oil_fvf_grid = fluid_properties.oil_formation_volume_factor_grid.copy()
    old_gas_fvf_grid = fluid_properties.gas_formation_volume_factor_grid.copy()
    old_water_fvf_grid = fluid_properties.water_formation_volume_factor_grid.copy()
    old_water_density_grid = fluid_properties.water_density_grid.copy()
    old_oil_density_grid = fluid_properties.oil_effective_density_grid.copy()
    old_gas_density_grid = fluid_properties.gas_density_grid.copy()

    logger.debug("Updating PVT fluid properties to reflect pressure changes...")
    fluid_properties = update_fluid_properties(
        fluid_properties=fluid_properties,
        wells=wells,
        miscibility_model=miscibility_model,
        pvt_tables=config.pvt_tables,
        freeze_saturation_pressure=config.freeze_saturation_pressure,
    )

    # Refresh boundary conditions so the saturation solve sees post-pressure BC values.
    metadata = attrs.evolve(metadata, fluid_properties=fluid_properties)
    flux_boundaries, pressure_boundaries = boundary_conditions.refresh_boundaries(
        metadata=metadata
    )

    logger.debug("Evolving saturation (implicit, Newton-Raphson)...")
    transport_result = implicit.solve_transport(
        cell_dimension=cell_dimension,
        thickness_grid=thickness_grid,
        elevation_grid=elevation_grid,
        time_step_size=time_step_size,
        rock_properties=rock_properties,
        fluid_properties=fluid_properties,
        old_water_density_grid=old_water_density_grid,
        old_oil_density_grid=old_oil_density_grid,
        old_gas_density_grid=old_gas_density_grid,
        old_solution_gas_to_oil_ratio_grid=old_solution_gor_grid,
        old_gas_solubility_in_water_grid=old_gas_solubility_in_water_grid,
        old_gas_formation_volume_factor_grid=old_gas_fvf_grid,
        old_oil_formation_volume_factor_grid=old_oil_fvf_grid,
        old_water_formation_volume_factor_grid=old_water_fvf_grid,
        face_transmissibilities=face_transmissibilities,
        config=config,
        flux_boundaries=flux_boundaries,
        pressure_boundaries=pressure_boundaries,
        rates=well_rates,
        dtype=dtype,
    )
    transport_solution = transport_result.value
    saturation_check = _check_saturation_changes(
        maximum_oil_saturation_change=transport_solution.maximum_oil_saturation_change,
        maximum_water_saturation_change=transport_solution.maximum_water_saturation_change,
        maximum_gas_saturation_change=transport_solution.maximum_gas_saturation_change,
        max_allowed_oil_saturation_change=config.maximum_oil_saturation_change,
        max_allowed_water_saturation_change=config.maximum_water_saturation_change,
        max_allowed_gas_saturation_change=config.maximum_gas_saturation_change,
        tolerance=1e-4,
    )
    timer_context = {
        "maximum_pressure_change": maximum_pressure_change,
        "maximum_allowed_pressure_change": maximum_allowed_pressure_change,
        "maximum_saturation_change": saturation_check.max_phase_saturation_change,
        "maximum_allowed_saturation_change": saturation_check.max_allowed_phase_saturation_change,
        "newton_iterations": transport_solution.newton_iterations,
    }

    if not transport_result.success:
        logger.warning(
            f"Implicit saturation evolution failed at time step {time_step}: \n{transport_result.message}"
        )
        return StepResult(
            fluid_properties=fluid_properties,
            rock_properties=rock_properties,
            saturation_history=saturation_history,
            success=False,
            message=transport_result.message,
            timer_context=timer_context,
        )

    if saturation_check.violated:
        message = (
            f"At time step {time_step}, saturation change limits were violated:\n"
            f"{saturation_check.message}\n"
            f"Oil: {transport_solution.maximum_oil_saturation_change:.6f}, "
            f"Water: {transport_solution.maximum_water_saturation_change:.6f}, "
            f"Gas: {transport_solution.maximum_gas_saturation_change:.6f}."
        )
        logger.warning(message)
        return StepResult(
            fluid_properties=fluid_properties,
            rock_properties=rock_properties,
            saturation_history=saturation_history,
            success=False,
            message=message,
            timer_context=timer_context,
        )

    logger.debug("Updating fluid properties with new saturation grids...")
    water_saturation_grid = transport_solution.water_saturation_grid.astype(
        dtype, copy=False
    )
    oil_saturation_grid = transport_solution.oil_saturation_grid.astype(
        dtype, copy=False
    )
    gas_saturation_grid = transport_solution.gas_saturation_grid.astype(
        dtype, copy=False
    )
    fluid_properties = attrs.evolve(
        fluid_properties,
        water_saturation_grid=water_saturation_grid,
        oil_saturation_grid=oil_saturation_grid,
        gas_saturation_grid=gas_saturation_grid,
    )

    if config.normalize_saturations:
        normalize_saturations(
            oil_saturation_grid=fluid_properties.oil_saturation_grid,
            water_saturation_grid=fluid_properties.water_saturation_grid,
            gas_saturation_grid=fluid_properties.gas_saturation_grid,
            saturation_epsilon=saturation_epsilon,
        )

    if saturation_history is not None:
        rock_properties, saturation_history = update_residual_saturation_grids(
            rock_properties=rock_properties,
            saturation_history=saturation_history,
            water_saturation_grid=fluid_properties.water_saturation_grid,
            gas_saturation_grid=fluid_properties.gas_saturation_grid,
            residual_oil_drainage_ratio_water_flood=config.residual_oil_drainage_ratio_water_flood,
            residual_oil_drainage_ratio_gas_flood=config.residual_oil_drainage_ratio_gas_flood,
            residual_gas_drainage_ratio=config.residual_gas_drainage_ratio,
        )

    material_balance_errors = compute_material_balance_errors(
        current_fluid_properties=fluid_properties,
        previous_fluid_properties=initial_fluid_properties,
        rock=rock_properties,
        thickness_grid=thickness_grid,
        cell_dimension=cell_dimension,
        injection_mass_rates=well_rates.injection_mass_rates  # type: ignore
        if has_open_wells
        else None,
        production_mass_rates=well_rates.production_mass_rates  # type: ignore
        if has_open_wells
        else None,
        time_step_size=time_step_size,
    )
    timer_context.update(
        {
            "absolute_oil_mbe": material_balance_errors.absolute_oil_mbe,
            "absolute_water_mbe": material_balance_errors.absolute_water_mbe,
            "absolute_gas_mbe": material_balance_errors.absolute_gas_mbe,
            "total_absolute_mbe": material_balance_errors.total_absolute_mbe,
            "relative_oil_mbe": material_balance_errors.relative_oil_mbe,
            "relative_water_mbe": material_balance_errors.relative_water_mbe,
            "relative_gas_mbe": material_balance_errors.relative_gas_mbe,
            "total_relative_mbe": material_balance_errors.total_relative_mbe,
        }
    )
    logger.debug("Sequential implicit step completed.")
    return StepResult(
        fluid_properties=fluid_properties,
        rock_properties=rock_properties,
        saturation_history=saturation_history,
        rates=well_rates,
        success=True,
        message=transport_result.message,
        material_balance_errors=material_balance_errors,
        timer_context=timer_context,
    )


def _run_full_sequential_implicit_step(
    time_step: int,
    grid_shape: ThreeDimensions,
    cell_dimension: typing.Tuple[float, float],
    thickness_grid: NDimensionalGrid[ThreeDimensions],
    elevation_grid: NDimensionalGrid[ThreeDimensions],
    time_step_size: float,
    time: float,
    face_transmissibilities: FaceTransmissibilities,
    rock_properties: RockProperties[ThreeDimensions],
    fluid_properties: FluidProperties[ThreeDimensions],
    saturation_history: typing.Optional[SaturationHistory[ThreeDimensions]],
    relperm_grids: RelPermGrids[ThreeDimensions],
    relative_mobility_grids: RelativeMobilityGrids[ThreeDimensions],
    capillary_pressure_grids: CapillaryPressureGrids[ThreeDimensions],
    wells: Wells[ThreeDimensions],
    miscibility_model: MiscibilityModel,
    config: Config,
    boundary_conditions: BoundaryConditions[ThreeDimensions],
    wells_indices: WellsIndices,
    dtype: npt.DTypeLike,
    min_valid_pressure: float = 14.7,
    max_valid_pressure: float = 14700,
    saturation_epsilon: float = 1e-12,
) -> StepResult[ThreeDimensions]:
    """
    Execute one time step using the Sequential Implicit scheme with outer iterations.

    Pressure is solved implicitly, then saturation is solved implicitly using
    Newton-Raphson. An outer iteration loop enforces coupling consistency
    between pressure and saturation until convergence or the maximum iteration
    count is reached.

    :param time_step: Current time step index.
    :param grid_shape: Model grid shape (nx, ny, nz).
    :param cell_dimension: Tuple of cell dimensions (dx, dy) in feet.
    :param thickness_grid: Cell thickness grid (ft).
    :param elevation_grid: Cell elevation grid (ft).
    :param time_step_size: Size of the current time step (seconds).
    :param time: Total simulation time elapsed, this time step inclusive (seconds).
    :param face_transmissibilities: Precomputed geometric face transmissibilities.
    :param rock_properties: Rock properties.
    :param fluid_properties: Fluid properties.
    :param saturation_history: Saturation history, or *None* if hysteresis is disabled.
    :param relperm_grids: Three-phase relative permeability grids.
    :param relative_mobility_grids: Three-phase relative mobility grids.
    :param capillary_pressure_grids: Oil-water and gas-oil capillary pressure grids.
    :param wells: Wells in the reservoir.
    :param miscibility_model: Miscibility model used in the simulation.
    :param config: Simulation configuration.
    :param boundary_conditions: Model boundary conditions.
    :param wells_indices: Cache of well indices for efficient lookup during the pressure solve.
    :param dtype: Data type used for numerical arrays.
    :param min_valid_pressure: Minimum valid pressure (psi). Pressures below this trigger a failure.
    :param max_valid_pressure: Maximum valid pressure (psi). Pressures above this trigger a failure.
    :param saturation_epsilon: Small value to keep saturations strictly between 0 and 1.
    :return: `StepResult` containing updated fluid properties, rock properties, and rates.
    """
    saturation_tolerance = config.transport_outer_convergence_tolerance
    pressure_tolerance = config.pressure_outer_convergence_tolerance
    maximum_newton_iterations = config.maximum_newton_iterations
    maximum_outer_iterations = config.maximum_outer_iterations
    initial_fluid_properties = fluid_properties

    logger.debug(
        "Outer iteration tolerances: saturation (absolute): %d:.2e, pressure (relative): %d:.2e",
        saturation_tolerance,
        pressure_tolerance,
    )

    previous_pressure_grid = fluid_properties.pressure_grid.copy()
    previous_water_saturation_grid = fluid_properties.water_saturation_grid.copy()
    previous_oil_saturation_grid = fluid_properties.oil_saturation_grid.copy()
    previous_gas_saturation_grid = fluid_properties.gas_saturation_grid.copy()

    iter_fluid_properties = fluid_properties
    iter_relative_mobility_grids = relative_mobility_grids
    iter_capillary_pressure_grids = capillary_pressure_grids
    iter_relperm_grids = relperm_grids

    outer_converged = False
    transport_result = None
    transport_solution = None
    well_rates = None
    has_open_wells = False
    final_timer_kwargs: typing.Dict[str, typing.Any] = {}

    logger.debug(
        "Starting outer iteration loop (max %d iterations) at time step %d...",
        maximum_outer_iterations,
        time_step,
    )

    for iteration in range(maximum_outer_iterations):
        logger.debug("Outer iteration %d/%d", iteration + 1, maximum_outer_iterations)

        metadata = build_boundary_metadata(
            fluid_properties=iter_fluid_properties,
            rock_properties=rock_properties,
            relperm_grids=iter_relperm_grids,
            relative_mobility_grids=iter_relative_mobility_grids,
            capillary_pressure_grids=iter_capillary_pressure_grids,
            face_transmissibilities=face_transmissibilities,
            grid_shape=grid_shape,
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            time=time,
            dtype=dtype,
        )
        flux_boundaries, pressure_boundaries = boundary_conditions.get_boundaries(
            grid_shape=grid_shape, metadata=metadata
        )

        if has_open_wells := wells.any_open():
            logger.debug("Computing well rates...")
            well_rates = compute_well_rates(
                fluid_properties=iter_fluid_properties,
                water_relative_mobility_grid=iter_relative_mobility_grids.water_relative_mobility,
                oil_relative_mobility_grid=iter_relative_mobility_grids.oil_relative_mobility,
                gas_relative_mobility_grid=iter_relative_mobility_grids.gas_relative_mobility,
                wells=wells,
                time=time,
                config=config,
                wells_indices=wells_indices,
                dtype=dtype,
            )

        else:
            well_rates = None

        logger.debug(
            "Evolving pressure (implicit) for outer iteration saturation solve..."
        )
        pressure_result = implicit.solve_pressure(
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            elevation_grid=elevation_grid,
            time_step=time_step,
            time_step_size=time_step_size,
            rock_properties=rock_properties,
            fluid_properties=iter_fluid_properties,
            relative_mobility_grids=iter_relative_mobility_grids,
            capillary_pressure_grids=iter_capillary_pressure_grids,
            face_transmissibilities=face_transmissibilities,
            config=config,
            flux_boundaries=flux_boundaries,
            pressure_boundaries=pressure_boundaries,
            net_well_rate_grid=well_rates.net_well_rate_grid  # type: ignore
            if has_open_wells
            else None,
            dtype=dtype,
        )

        if not pressure_result.success:
            logger.warning(
                f"Implicit pressure solve failed at outer iteration "
                f"{iteration + 1}, time step {time_step}:\n{pressure_result.message}"
            )
            return StepResult(
                fluid_properties=fluid_properties,
                rock_properties=rock_properties,
                saturation_history=saturation_history,
                success=False,
                message=pressure_result.message,
            )

        pressure_solution = pressure_result.value
        new_pressure_grid = pressure_solution.pressure_grid
        maximum_pressure_change = pressure_solution.maximum_pressure_change
        maximum_allowed_pressure_change = config.maximum_pressure_change * 1.0001

        if maximum_pressure_change > maximum_allowed_pressure_change:
            message = (
                f"Pressure change {maximum_pressure_change:.6f} psi exceeded maximum "
                f"allowed {maximum_allowed_pressure_change:.6f} psi at time step "
                f"{time_step}, outer iteration {iteration + 1}."
            )
            logger.warning(message)
            return StepResult(
                fluid_properties=fluid_properties,
                rock_properties=rock_properties,
                saturation_history=saturation_history,
                success=False,
                message=message,
                timer_context={
                    "maximum_pressure_change": maximum_pressure_change,
                    "maximum_allowed_pressure_change": maximum_allowed_pressure_change,
                },
            )

        result = _validate_pressure_range(
            pressure_grid=new_pressure_grid,
            time_step=time_step,
            fluid_properties=fluid_properties,
            rock_properties=rock_properties,
            saturation_history=saturation_history,
        )
        if result is not None:
            return result

        np.clip(
            new_pressure_grid,
            min_valid_pressure,
            max_valid_pressure,
            dtype=dtype,
            out=new_pressure_grid,
        )
        iter_fluid_properties = attrs.evolve(
            iter_fluid_properties, pressure_grid=new_pressure_grid
        )
        logger.debug(
            "Pressure updated in fluid properties for outer iteration saturation solve."
        )

        # Copy before PVT updates
        old_solution_gor_grid = (
            iter_fluid_properties.solution_gas_to_oil_ratio_grid.copy()
        )
        old_gas_solubility_in_water_grid = (
            iter_fluid_properties.gas_solubility_in_water_grid.copy()
        )
        old_oil_fvf_grid = iter_fluid_properties.oil_formation_volume_factor_grid.copy()
        old_gas_fvf_grid = iter_fluid_properties.gas_formation_volume_factor_grid.copy()
        old_water_fvf_grid = (
            iter_fluid_properties.water_formation_volume_factor_grid.copy()
        )
        old_water_density_grid = iter_fluid_properties.water_density_grid.copy()
        old_oil_density_grid = iter_fluid_properties.oil_effective_density_grid.copy()
        old_gas_density_grid = iter_fluid_properties.gas_density_grid.copy()

        logger.debug(
            "Updating fluid properties to reflect pressure change for outer iteration..."
        )
        iter_fluid_properties = update_fluid_properties(
            fluid_properties=iter_fluid_properties,
            wells=wells,
            miscibility_model=miscibility_model,
            pvt_tables=config.pvt_tables,
            freeze_saturation_pressure=config.freeze_saturation_pressure,
        )

        # Refresh boundary conditions with updated pressure for the saturation solve.
        metadata = attrs.evolve(metadata, fluid_properties=iter_fluid_properties)
        flux_boundaries, pressure_boundaries = boundary_conditions.refresh_boundaries(
            metadata=metadata
        )

        logger.debug(
            "Evolving saturation (implicit, Newton-Raphson) for outer iteration %d/%d...",
            iteration + 1,
            maximum_outer_iterations,
        )
        transport_result = implicit.solve_transport(
            cell_dimension=cell_dimension,
            thickness_grid=thickness_grid,
            elevation_grid=elevation_grid,
            time_step_size=time_step_size,
            rock_properties=rock_properties,
            fluid_properties=iter_fluid_properties,
            old_water_density_grid=old_water_density_grid,
            old_oil_density_grid=old_oil_density_grid,
            old_gas_density_grid=old_gas_density_grid,
            old_solution_gas_to_oil_ratio_grid=old_solution_gor_grid,
            old_gas_solubility_in_water_grid=old_gas_solubility_in_water_grid,
            old_gas_formation_volume_factor_grid=old_gas_fvf_grid,
            old_oil_formation_volume_factor_grid=old_oil_fvf_grid,
            old_water_formation_volume_factor_grid=old_water_fvf_grid,
            face_transmissibilities=face_transmissibilities,
            config=config,
            flux_boundaries=flux_boundaries,
            pressure_boundaries=pressure_boundaries,
            rates=well_rates,
            dtype=dtype,
        )

        if not transport_result.success:
            logger.warning(
                f"Implicit saturation solve failed at outer iteration "
                f"{iteration + 1}, time step {time_step}:\n{transport_result.message}"
            )
            return StepResult(
                fluid_properties=fluid_properties,
                rock_properties=rock_properties,
                saturation_history=saturation_history,
                success=False,
                message=transport_result.message,
            )

        transport_solution = transport_result.value
        saturation_check = _check_saturation_changes(
            maximum_oil_saturation_change=transport_solution.maximum_oil_saturation_change,
            maximum_water_saturation_change=transport_solution.maximum_water_saturation_change,
            maximum_gas_saturation_change=transport_solution.maximum_gas_saturation_change,
            max_allowed_oil_saturation_change=config.maximum_oil_saturation_change,
            max_allowed_water_saturation_change=config.maximum_water_saturation_change,
            max_allowed_gas_saturation_change=config.maximum_gas_saturation_change,
            tolerance=1e-4,
        )
        if saturation_check.violated:
            message = (
                f"At time step {time_step}, outer iteration {iteration + 1}, "
                f"saturation change limits were violated:\n{saturation_check.message}\n"
                f"Oil: {transport_solution.maximum_oil_saturation_change:.6f}, "
                f"Water: {transport_solution.maximum_water_saturation_change:.6f}, "
                f"Gas: {transport_solution.maximum_gas_saturation_change:.6f}."
            )
            logger.warning(message)
            return StepResult(
                fluid_properties=fluid_properties,
                rock_properties=rock_properties,
                saturation_history=saturation_history,
                success=False,
                message=message,
                timer_context={
                    "maximum_saturation_change": saturation_check.max_phase_saturation_change,
                    "maximum_allowed_saturation_change": saturation_check.max_allowed_phase_saturation_change,
                },
            )

        water_saturation_grid = transport_solution.water_saturation_grid.astype(
            dtype, copy=False
        )
        oil_saturation_grid = transport_solution.oil_saturation_grid.astype(
            dtype, copy=False
        )
        gas_saturation_grid = transport_solution.gas_saturation_grid.astype(
            dtype, copy=False
        )

        iter_fluid_properties = attrs.evolve(
            iter_fluid_properties,
            water_saturation_grid=water_saturation_grid,
            oil_saturation_grid=oil_saturation_grid,
            gas_saturation_grid=gas_saturation_grid,
        )

        if config.normalize_saturations:
            normalize_saturations(
                oil_saturation_grid=iter_fluid_properties.oil_saturation_grid,
                water_saturation_grid=iter_fluid_properties.water_saturation_grid,
                gas_saturation_grid=iter_fluid_properties.gas_saturation_grid,
                saturation_epsilon=saturation_epsilon,
            )

        newton_iterations = transport_solution.newton_iterations
        newton_utilization = newton_iterations / maximum_newton_iterations
        final_timer_kwargs = {
            "maximum_pressure_change": maximum_pressure_change,
            "maximum_allowed_pressure_change": maximum_allowed_pressure_change,
            "maximum_saturation_change": saturation_check.max_phase_saturation_change,
            "maximum_allowed_saturation_change": saturation_check.max_allowed_phase_saturation_change,
            "newton_iterations": newton_iterations,
        }

        if newton_utilization < 0.25:
            logger.debug(
                f"Newton converged in {newton_iterations} iterations "
                f"(utilization {newton_utilization:.0%}), skipping further outer iterations."
            )
            outer_converged = True
            break

        total_saturation_change_from_bop = max(
            float(
                np.max(
                    np.abs(
                        iter_fluid_properties.water_saturation_grid
                        - fluid_properties.water_saturation_grid
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        iter_fluid_properties.oil_saturation_grid
                        - fluid_properties.oil_saturation_grid
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        iter_fluid_properties.gas_saturation_grid
                        - fluid_properties.gas_saturation_grid
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
                f"Total saturation change from start of timestep {total_saturation_change_from_bop:.3e} "
                f"is small, skipping further outer iterations."
            )
            outer_converged = True
            break

        max_outer_saturation_change = max(
            float(
                np.max(
                    np.abs(
                        iter_fluid_properties.water_saturation_grid
                        - previous_water_saturation_grid
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        iter_fluid_properties.oil_saturation_grid
                        - previous_oil_saturation_grid
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        iter_fluid_properties.gas_saturation_grid
                        - previous_gas_saturation_grid
                    )
                )
            ),
        )
        reference_pressure = max(float(np.mean(np.abs(new_pressure_grid))), 1.0)
        relative_outer_pressure_change = (
            float(np.max(np.abs(new_pressure_grid - previous_pressure_grid)))
            / reference_pressure
        )

        logger.debug(
            f"Outer iteration {iteration + 1} convergence - "
            f"Δsat (absolute): {max_outer_saturation_change:.3e} (atol={saturation_tolerance:.3e}), "
            f"ΔP (relative): {relative_outer_pressure_change:.3e} (rtol={pressure_tolerance:.3e})"
        )

        if (
            max_outer_saturation_change < saturation_tolerance
            and relative_outer_pressure_change < pressure_tolerance
        ):
            logger.debug(
                f"Outer iteration converged after {iteration + 1} iteration(s)."
            )
            outer_converged = True
            break

        iter_fluid_properties = update_fluid_properties(
            fluid_properties=iter_fluid_properties,
            wells=wells,
            miscibility_model=miscibility_model,
            pvt_tables=config.pvt_tables,
            freeze_saturation_pressure=config.freeze_saturation_pressure,
        )
        (
            iter_relperm_grids,
            iter_relative_mobility_grids,
            iter_capillary_pressure_grids,
        ) = _rebuild_rock_fluid_grids(iter_fluid_properties, rock_properties, config)

        previous_pressure_grid = new_pressure_grid.copy()
        previous_water_saturation_grid = (
            iter_fluid_properties.water_saturation_grid.copy()
        )
        previous_oil_saturation_grid = iter_fluid_properties.oil_saturation_grid.copy()
        previous_gas_saturation_grid = iter_fluid_properties.gas_saturation_grid.copy()

    if not outer_converged:
        logger.warning(
            f"Outer iteration did not converge after {config.maximum_outer_iterations} "
            f"iteration(s) at time step {time_step}. Proceeding with last solution."
        )

    assert transport_result is not None and transport_solution is not None, (
        "Saturation solve must have run at least once."
    )

    if saturation_history is not None:
        rock_properties, saturation_history = update_residual_saturation_grids(
            rock_properties=rock_properties,
            saturation_history=saturation_history,
            water_saturation_grid=iter_fluid_properties.water_saturation_grid,
            gas_saturation_grid=iter_fluid_properties.gas_saturation_grid,
            residual_oil_drainage_ratio_water_flood=config.residual_oil_drainage_ratio_water_flood,
            residual_oil_drainage_ratio_gas_flood=config.residual_oil_drainage_ratio_gas_flood,
            residual_gas_drainage_ratio=config.residual_gas_drainage_ratio,
        )

    material_balance_errors = compute_material_balance_errors(
        current_fluid_properties=iter_fluid_properties,
        previous_fluid_properties=initial_fluid_properties,
        rock=rock_properties,
        thickness_grid=thickness_grid,
        cell_dimension=cell_dimension,
        injection_mass_rates=well_rates.injection_mass_rates  # type: ignore
        if has_open_wells
        else None,
        production_mass_rates=well_rates.production_mass_rates  # type: ignore
        if has_open_wells
        else None,
        time_step_size=time_step_size,
    )
    final_timer_kwargs.update(
        {
            "absolute_oil_mbe": material_balance_errors.absolute_oil_mbe,
            "absolute_water_mbe": material_balance_errors.absolute_water_mbe,
            "absolute_gas_mbe": material_balance_errors.absolute_gas_mbe,
            "total_absolute_mbe": material_balance_errors.total_absolute_mbe,
            "relative_oil_mbe": material_balance_errors.relative_oil_mbe,
            "relative_water_mbe": material_balance_errors.relative_water_mbe,
            "relative_gas_mbe": material_balance_errors.relative_gas_mbe,
            "total_relative_mbe": material_balance_errors.total_relative_mbe,
        }
    )
    logger.debug("Full sequential implicit step completed.")
    return StepResult(
        fluid_properties=iter_fluid_properties,
        rock_properties=rock_properties,
        saturation_history=saturation_history,
        rates=well_rates,
        success=True,
        message=transport_result.message,
        material_balance_errors=material_balance_errors,
        timer_context=final_timer_kwargs,
    )


def log_progress(
    step: int,
    step_size: float,
    time_elapsed: float,
    total_time: float,
    is_last_step: bool = False,
    interval: int = 3,
) -> None:
    """
    Log simulation progress at specified step intervals.

    :param step: Current accepted time step index.
    :param step_size: Size of the accepted time step (seconds).
    :param time_elapsed: Total elapsed simulation time (seconds).
    :param total_time: Total simulation duration (seconds).
    :param is_last_step: *True* if this is the final time step.
    :param interval: Log every *interval* steps (and always on the first and last).
    """
    if step <= 1 or step % interval == 0 or is_last_step:
        percent_complete = (time_elapsed / total_time) * 100.0
        logger.info(
            f"Time Step {step} with Δt = {step_size:.4f}s - "
            f"({percent_complete:.4f}%) - "
            f"Elapsed Time: {time_elapsed:.4f}s / {total_time:.4f}s"
        )


StepCallback = typing.Callable[[StepResult[ThreeDimensions], float, float], None]
"""
A callback invoked after each simulation step attempt.

Arguments:

- `StepResult[ThreeDimensions]`: The result of the current step, containing updated
  fluid properties, rock properties, saturation history, rates, BHPs, success status,
  message, and timer kwargs.
- `float`: The step size that was attempted (seconds).
- `float`: Total accepted simulation time elapsed up to the current step (seconds).
"""


@attrs.define
class Run(StoreSerializable):
    """
    Simulation run specification.

    Executes a reservoir simulation on a 3D static reservoir model using the
    provided configuration.

    Example:

    ```python
    from bores import ReservoirModel, Config, Run

    model = ReservoirModel.from_file("path/to/3d_model.h5")
    config = Config.from_file("path/to/simulation_config.yaml")

    run = Run(model=model, config=config)
    for state in run:
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
        config: typing.Optional[Config] = None,
        *,
        on_step_rejected: typing.Optional[StepCallback] = None,
        on_step_accepted: typing.Optional[StepCallback] = None,
    ) -> typing.Generator[ModelState[ThreeDimensions], None, None]:
        """
        Return a generator that executes this simulation run.

        :param config: Optional override for the run's configuration.
        :param on_step_rejected: Optional callback invoked when a step is rejected.
        :param on_step_accepted: Optional callback invoked when a step is accepted.
        :return: Generator yielding `ModelState` at each output interval.
        """
        return run(
            self.model,
            config if config is not None else self.config,
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
        Load a `Run` from separate model and configuration files.

        :param model_path: Path to the reservoir model file.
        :param config_path: Path to the simulation configuration file.
        :param pvt_tables_path: Optional path to a dumped `PVTTables` file.
        :param pvt_data_path: Optional path to a dumped `PVTDataSet` file.
        :return: `Run` instance with loaded model and config.
        :raises ValidationError: If the loaded model is not a 3-D `ReservoirModel`.
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

    The simulation evolves pressure and saturation over time using the chosen
    evolution scheme.

    :param input: Either a `ReservoirModel` or a `Run` instance. If a `Run` is
        supplied and *config* is also provided, the explicit *config* takes precedence.
    :param config: Simulation configuration. Required when *input* is a
        `ReservoirModel`; optional when *input* is a `Run`.
    :param on_step_rejected: Optional callback invoked each time a step is
        rejected due to stability or convergence issues. Receives the
        `StepResult`, the attempted step size (seconds), and the total elapsed
        simulation time (seconds).
    :param on_step_accepted: Optional callback invoked each time a step is
        successfully accepted. Same signature as *on_step_rejected*.
    :yield: `ModelState` snapshots at the frequency specified by
        `config.output_frequency`.
    :raises ValidationError: If *config* is missing when required.
    :raises SimulationError: If the simulation cannot recover from a failed step.

    Example:

    ```python

    import bores

    model = bores.ReservoirModel.from_file("path/to/3d_model.h5")
    config = bores.Config.from_file("path/to/simulation_config.yaml")
    for state in bores.run(model, config):
        process(state)
    ```
    """
    if isinstance(input, Run):
        model = input.model
        if config is not None:
            logger.info(
                "Overriding `config` from `Run` instance with the provided `config` parameter."
            )
        config = config or input.config
    else:
        if config is None:
            raise ValidationError(
                "Must provide `config` when `input` is a `ReservoirModel`."
            )
        model = input

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
    has_well_schedules = well_schedules is not None
    output_frequency = config.output_frequency
    scheme = config.scheme.replace("_", "-").lower()
    miscibility_model = config.miscibility_model
    dtype = get_dtype()
    pvt_tables = config.pvt_tables
    freeze_saturation_pressure = config.freeze_saturation_pressure
    log_interval = config.log_interval
    capture_timer_state = config.capture_timer_state
    enable_hysteresis = config.enable_hysteresis
    apply_dip = not config.disable_structural_dip

    logger.info("Starting simulation workflow...")
    logger.info(
        f"Grid dimensions: (nx={grid_shape[0]}, ny={grid_shape[1]}, nz={grid_shape[2]})"
    )
    logger.info(
        f"Cell dimensions: (dx={cell_dimension[0]}ft, dy={cell_dimension[1]}ft)"
    )
    logger.info(
        f"Evolution scheme: {_SCHEME_ALIASES.get(scheme, scheme.replace('-', ' ').title())}"
    )
    logger.info(f"Array numerical precision: {np.dtype(dtype).name!r}")
    logger.info("Total simulation time: %.1f seconds", timer.simulation_time)
    logger.info(
        f"Output frequency: Every {output_frequency} steps"
        if output_frequency > 1
        else "Output frequency: Every step",
    )
    logger.info("Has wells: %s", has_wells)
    if has_wells:
        logger.debug("Checking well locations against grid shape")
        wells.check_location(grid_shape=grid_shape)

    with config.constants():
        min_valid_pressure = c.MINIMUM_VALID_PRESSURE
        max_valid_pressure = c.MAXIMUM_VALID_PRESSURE
        saturation_epsilon = c.SATURATION_EPSILON

        fluid_properties = model.fluid_properties
        rock_properties = model.rock_properties
        saturation_history = model.saturation_history if enable_hysteresis else None
        thickness_grid = model.thickness_grid
        absolute_permeability = rock_properties.absolute_permeability
        net_to_gross_grid = rock_properties.net_to_gross_grid
        elevation_grid = model.build_elevation_grid(apply_dip=apply_dip)

        logger.debug("Building well indices cache")
        wells_indices = build_wells_indices(
            grid_shape=grid_shape,
            cell_size_x=cell_dimension[0],
            cell_size_y=cell_dimension[1],
            thickness_grid=thickness_grid,
            wells=wells,
            absolute_permeability=absolute_permeability,
            net_to_gross_grid=net_to_gross_grid,
            boundary_conditions=boundary_conditions,
            regime_constant=-0.75,  # Pseudo steady regime
        )

        logger.debug("Building face transmissibilities...")
        face_transmissibilities = model.build_face_transmissibilities(dtype=dtype)

        logger.debug("Initializing PVT fluid properties...")
        fluid_properties = update_fluid_properties(
            fluid_properties=fluid_properties,
            wells=wells,
            miscibility_model=miscibility_model,
            pvt_tables=pvt_tables,
            freeze_saturation_pressure=freeze_saturation_pressure,
        )
        model = model.evolve(fluid_properties=fluid_properties)

        # Seed injector saturations to avoid phase deadlock at t=0
        if has_wells:
            logger.debug("Seeding injection saturations in injector perforations...")
            fluid_properties = seed_injection_saturations(
                fluid_properties=fluid_properties,
                wells=wells,
                wells_indices=wells_indices,
                config=config,
                minimum_injector_water_saturation=config.minimum_injector_water_saturation,
                minimum_injector_gas_saturation=config.minimum_injector_gas_saturation,
                inplace=True,
            )

        if config.check_zero_flow_initialization:
            logger.debug("Checking initial state for zero-flow violations...")
            zero_flow_check = check_zero_flow_initialization(
                fluid_properties=fluid_properties,
                rock_properties=rock_properties,
                face_transmissibilities=face_transmissibilities,
                elevation_grid=elevation_grid,
                config=config,
                cell_dimension=cell_dimension,
                thickness_grid=thickness_grid,
                relative_tolerance=config.zero_flow_relative_flux_tolerance,
                max_reported_violations=10,
                dtype=dtype,
            )
            if zero_flow_check.passed:
                logger.info(
                    "Zero-Flow initialization check passed! Max relative flux = %.3e day⁻¹ (tolerance = %.3e day⁻¹). "
                    "Checked %d active cells.",
                    zero_flow_check.max_relative_flux,
                    zero_flow_check.relative_tolerance,
                    zero_flow_check.cells_checked,
                )
            else:
                logger.warning(
                    "Zero-Flow initialization check failed! %d of %d active cells exceed tolerance %.3e day⁻¹. "
                    "Max relative flux = %.3e day⁻¹ at cell %s. "
                    "Check pressure/saturation/density/capillary-pressure consistency.",
                    zero_flow_check.violation_count,
                    zero_flow_check.cells_checked,
                    zero_flow_check.relative_tolerance,
                    zero_flow_check.max_relative_flux,
                    zero_flow_check.worst_cell,
                )
                for violation in zero_flow_check.violations:
                    logger.warning(
                        "  Cell %s: relative_flux=%.3e day⁻¹  "
                        "(qw=%.3e, qo=%.3e, qg=%.3e lbm/day, PV=%.3e ft³)",
                        violation.cell,
                        violation.relative_flux,
                        violation.net_water_flux,
                        violation.net_oil_flux,
                        violation.net_gas_flux,
                        violation.pore_volume,
                    )

        logger.debug("Building initial rock-fluid property grids...")
        relperm_grids, relative_mobility_grids, capillary_pressure_grids = (
            _rebuild_rock_fluid_grids(fluid_properties, rock_properties, config)
        )
        rates = RatesInfo(
            injection_rates=_make_rates(grid_shape),
            production_rates=_make_rates(grid_shape),
            injection_mass_rates=_make_rates(grid_shape),
            production_mass_rates=_make_rates(grid_shape),
            injection_fvfs=_make_fvfs(grid_shape),
            production_fvfs=_make_fvfs(grid_shape),
            injection_bhps=_make_bhps(grid_shape),
            production_bhps=_make_bhps(grid_shape),
        )
        null_mbe = MaterialBalanceErrors.null()

        logger.debug("Yielding zero-time initial state...")
        state = ModelState(
            step=timer.step,
            step_size=timer.step_size,
            time=timer.elapsed_time,
            model=model,
            wells=wells,
            relative_mobilities=relative_mobility_grids,
            relative_permeabilities=relperm_grids,
            capillary_pressures=capillary_pressure_grids,
            rates=rates,
            timer_state=timer.dump_state(),
            material_balance_errors=null_mbe,
        )
        yield state

        while not timer.done():
            new_step = timer.next_step
            step_size = timer.propose_step_size()
            time = timer.elapsed_time + step_size
            logger.debug(
                "Attempting time step %d with size %d seconds...", new_step, step_size
            )
            try:
                if has_wells and has_well_schedules:
                    logger.debug(
                        "Updating wells configuration for time step %d", new_step
                    )
                    assert well_schedules is not None
                    well_schedules.apply(wells, state)  # type: ignore[attr-defined]

                if new_step > 1:
                    # Apply minimum injector saturations BEFORE mobility rebuild to ensure
                    # non-zero mobilities in the pressure Jacobian.
                    if has_wells:
                        logger.debug(
                            "Enforcing minimum injector saturations for time step %d...",
                            new_step,
                        )
                        fluid_properties = apply_minimum_injector_saturations(
                            fluid_properties=fluid_properties,
                            wells=wells,
                            wells_indices=wells_indices,
                            minimum_injector_water_saturation=config.minimum_injector_water_saturation,
                            minimum_injector_gas_saturation=config.minimum_injector_gas_saturation,
                            dtype=dtype,
                        )

                    # Rebuild rock-fluid grids from the current saturation state
                    # at the start of every step so the solvers see consistent mobilities.
                    logger.debug(
                        "Rebuilding rock-fluid property grids for time step %d...",
                        new_step,
                    )
                    relperm_grids, relative_mobility_grids, capillary_pressure_grids = (
                        _rebuild_rock_fluid_grids(
                            fluid_properties, rock_properties, config
                        )
                    )

                    with update_wells_indices as should_update:
                        if should_update:
                            logger.debug("Updating well indices cache")
                            wells_indices = build_wells_indices(
                                grid_shape=grid_shape,
                                cell_size_x=cell_dimension[0],
                                cell_size_y=cell_dimension[1],
                                thickness_grid=thickness_grid,
                                wells=wells,
                                absolute_permeability=absolute_permeability,
                                net_to_gross_grid=net_to_gross_grid,
                                boundary_conditions=boundary_conditions,
                                regime_constant=-0.75,
                            )

                kwds = dict(  # noqa
                    time_step=new_step,
                    grid_shape=grid_shape,
                    cell_dimension=cell_dimension,
                    thickness_grid=thickness_grid,
                    elevation_grid=elevation_grid,
                    time_step_size=step_size,
                    time=time,
                    face_transmissibilities=face_transmissibilities,
                    rock_properties=rock_properties,
                    fluid_properties=fluid_properties,
                    saturation_history=saturation_history,
                    relperm_grids=relperm_grids,
                    relative_mobility_grids=relative_mobility_grids,
                    capillary_pressure_grids=capillary_pressure_grids,
                    wells=wells,
                    miscibility_model=miscibility_model,
                    config=config,
                    boundary_conditions=boundary_conditions,
                    wells_indices=wells_indices,
                    dtype=dtype,
                    min_valid_pressure=min_valid_pressure,
                    max_valid_pressure=max_valid_pressure,
                    saturation_epsilon=saturation_epsilon,
                )

                if scheme == "impes":
                    result = _run_impes_step(**kwds)  # type: ignore[arg-type]
                elif scheme in {"sequential-implicit", "si"}:
                    result = _run_sequential_implicit_step(**kwds)  # type: ignore[arg-type]
                elif scheme in {"full-sequential-implicit", "full-si"}:
                    result = _run_full_sequential_implicit_step(**kwds)  # type: ignore[arg-type]
                else:
                    raise ValidationError(
                        f"Invalid simulation scheme {scheme!r}. Supported schemes: 'impes', 'sequential-implicit', or 'full-sequential-implicit'."
                    )

                acceptable = False
                timer_context = result.timer_context
                error_msg = None
                if result.success:
                    acceptable, error_msg = timer.is_acceptable(**timer_context)
                    if acceptable:
                        logger.debug("Time step %d completed successfully.", new_step)
                        timer.accept_step(step_size=step_size, **timer_context)
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

                if not acceptable:
                    logger.debug(
                        "Time step %d failed with step size %d. Retrying with smaller step size.",
                        new_step,
                        step_size,
                    )
                    try:
                        timer.reject_step(
                            step_size=step_size,
                            aggressive=timer.rejection_count > 5,
                            **timer_context,
                        )
                    except TimingError as exc:
                        raise SimulationError(
                            f"Simulation failed at time step {new_step} and cannot reduce "
                            f"step size further. {exc}.\n{error_msg or result.message or ''}"
                        ) from exc

                    if on_step_rejected is not None:
                        on_step_rejected(result, step_size, timer.elapsed_time)

                    continue

                fluid_properties = result.fluid_properties
                rock_properties = result.rock_properties
                saturation_history = result.saturation_history

                if (
                    timer.step == 1
                    or (timer.step % output_frequency == 0)
                    or timer.is_last_step
                ):
                    logger.debug("Capturing model state at time step %d", timer.step)
                    wells_snapshot = copy.deepcopy(wells)
                    if saturation_history is not None:
                        model_snapshot = model.evolve(
                            fluid_properties=fluid_properties,
                            rock_properties=rock_properties,
                            saturation_history=saturation_history,
                        )
                    else:
                        model_snapshot = model.evolve(
                            fluid_properties=fluid_properties,
                            rock_properties=rock_properties,
                        )

                    # Rebuild grids for the snapshot so the yielded state is
                    # consistent with the accepted fluid properties.
                    relperm_snapshot, mobilities_snapshot, capillary_snapshot = (
                        _rebuild_rock_fluid_grids(
                            fluid_properties, rock_properties, config
                        )
                    )

                    material_balance_errors = result.material_balance_errors
                    material_balance_errors = (
                        material_balance_errors
                        if material_balance_errors is not None
                        else null_mbe
                    )
                    if (well_rates := result.rates) is not None:
                        rates = RatesInfo(
                            injection_rates=well_rates.injection_rates,
                            production_rates=well_rates.production_rates.abs(),
                            injection_mass_rates=well_rates.injection_mass_rates,
                            production_mass_rates=well_rates.production_mass_rates.abs(),
                            injection_fvfs=well_rates.injection_fvfs,
                            production_fvfs=well_rates.production_fvfs,
                            injection_bhps=well_rates.injection_bhps,
                            production_bhps=well_rates.production_bhps,
                        )
                    else:
                        rates = RatesInfo(
                            injection_rates=_make_rates(grid_shape),
                            production_rates=_make_rates(grid_shape),
                            injection_mass_rates=_make_rates(grid_shape),
                            production_mass_rates=_make_rates(grid_shape),
                            injection_fvfs=_make_fvfs(grid_shape),
                            production_fvfs=_make_fvfs(grid_shape),
                            injection_bhps=_make_bhps(grid_shape),
                            production_bhps=_make_bhps(grid_shape),
                        )

                    state = ModelState(
                        step=timer.step,
                        step_size=timer.step_size,
                        time=timer.elapsed_time,
                        model=model_snapshot,
                        wells=wells_snapshot,
                        relative_mobilities=mobilities_snapshot,
                        relative_permeabilities=relperm_snapshot,
                        capillary_pressures=capillary_snapshot,
                        rates=rates,
                        timer_state=timer.dump_state() if capture_timer_state else None,
                        material_balance_errors=material_balance_errors,
                    )
                    logger.debug("Yielding model state")
                    yield state

            except StopSimulation as exc:
                logger.info("Stopping simulation on request: %s", exc)
                break
            except Exception as exc:
                raise SimulationError(
                    f"Simulation failed while attempting time step {new_step} due to error: {exc}"
                ) from exc

    logger.info("Simulation completed successfully after %d time steps", timer.step)
